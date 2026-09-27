using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Minimal InfiniTAM-style block extractor.  It partitions only extraction
    /// and presentation; every block still reads the one authoritative TSDF.
    /// No candidate history, smoothing, product court, quality gate or legacy
    /// paper/HERA policy is present on this route.
    /// </summary>
    internal sealed class InfiniTamBlockMeshPipeline : IDisposable
    {
        private sealed class Block : IDisposable
        {
            public int Index;
            public int3 Coordinate;
            public int3 CoreMin;
            public int3 CoreMax;
            public int3 MapMin;
            public int3 MapCount;
            public uint TargetEpoch;
            public uint ProcessedEpoch;
            public uint LastOwnerEpoch;
            public readonly uint[] LastBoundaryEpoch = new uint[6];
            public bool Built;
            public bool Queued;
            public bool CommitPending;
            public int CommitSerial;
            public int VertexCount;
            public int IndexCount;
            public GameObject GameObject;
            public GPUSurfaceNets Surface;
            public GPUChunkMeshSnapshot Front;
            public GPUChunkMeshSnapshot Back;
            public GPUMeshRenderer Renderer;

            public void Dispose()
            {
                if (Renderer != null)
                    Renderer.RenderVisible = false;
                Front?.Dispose();
                Back?.Dispose();
                Surface?.Dispose();
                Front = null;
                Back = null;
                Surface = null;
                if (GameObject != null)
                    UnityEngine.Object.Destroy(GameObject);
                GameObject = null;
                Renderer = null;
            }
        }

        private static readonly int3[] FaceNeighbours =
        {
            new int3(-1, 0, 0), new int3(1, 0, 0),
            new int3(0, -1, 0), new int3(0, 1, 0),
            new int3(0, 0, -1), new int3(0, 0, 1)
        };

        private readonly VolumeIntegrator _volume;
        private readonly ComputeShader _compute;
        private readonly Material _material;
        private readonly Transform _parent;
        private readonly int _layer;
        private readonly int _haloVoxels;
        private readonly int _maxBlocksPerBatch;
        private readonly float _dirtyReadbackHz;
        private readonly float _vertexBudgetPercent;
        private readonly Action<GPUSurfaceNets> _extract;
        private readonly List<Block> _blocks = new List<Block>();

        private int3 _blockCount;
        private int _generation;
        private int _readbackFailures;
        private int _inFlightCommits;
        private int _queuedBlockCount;
        private int _nextBatchSerial;
        private long _acceptedCommitCount;
        private long _completedBatchCount;
        private long _publishedBatchCount;
        private long _dirtyLedgerApplyCount;
        private long _staleCandidateDiscardCount;
        private bool _disposed;
        private bool _visible;
        private bool _readbackPending;
        private bool _ownerReady;
        private bool _boundaryReady;
        private bool _ledgerFailed;
        private bool _firstLedgerApplied;
        private bool _initialNeedsSettlement = true;
        private uint[] _ownerSnapshot;
        private uint[] _boundarySnapshot;
        private float _nextReadbackTime;
        private MeshBatch _activeBatch;

        // Mature incremental mesh scheduling has one batch in flight. Dirty
        // arrivals are idempotent block flags that accumulate for the next
        // batch; they are never appended as historical jobs.
        private const float BatchWatchdogSeconds = 10f;
        // Isolated A/B switch: keep the authoritative TSDF at 5 cm, but let
        // the InfiniTAM extractor build one vertex per globally anchored
        // 2x2x2 cell (10 cm topology).  No shader grid, corner treatment,
        // smoothing or constrained simplifier participates in this trial.
        // 09-25 device A/B result: stride-2 topology over a 5 cm TSDF makes
        // complete triangles and boundary patches toggle as the fine signs
        // evolve. Keep that experiment wired but dormant. The accepted 10 cm
        // route instead uses one native 10 cm TSDF and ordinary stride-1
        // extraction from integration through presentation.
        internal const bool DirectTenCentimeterExperiment = false;
        // Isolated native-10 cm crease trial.  This changes only the position
        // chosen for a stride-1 Surface-Nets vertex when two coherent normal
        // families prove a real fold.  TSDF integration, topology, publication
        // and the native 10 cm lattice remain exactly on the accepted baseline.
        internal const bool NativeTenCentimeterCreaseExperiment = true;

        public bool Failed { get; private set; }
        public string FailureReason { get; private set; } = string.Empty;
        public bool InitialBuildComplete { get; private set; }
        public int CommitWatchdogResets { get; private set; }
        public int BlockCount => _blocks.Count;
        public int PendingCount => _queuedBlockCount + _inFlightCommits;
        public int QueuedCount => _queuedBlockCount;
        public int InFlightCommitCount => _inFlightCommits;
        public long AcceptedCommitCount => _acceptedCommitCount;
        public long CompletedBatchCount => _completedBatchCount;
        public long PublishedBatchCount => _publishedBatchCount;
        public bool BatchInFlight => _activeBatch != null;
        public long DirtyLedgerApplyCount => _dirtyLedgerApplyCount;
        public long StaleCandidateDiscardCount => _staleCandidateDiscardCount;

        public int OutstandingBlockCount
        {
            get
            {
                int count = 0;
                for (int i = 0; i < _blocks.Count; i++)
                {
                    Block block = _blocks[i];
                    if (block.Queued || block.CommitPending ||
                        block.TargetEpoch > block.ProcessedEpoch)
                        count++;
                }
                return count;
            }
        }

        private sealed class BatchItem
        {
            public int BlockIndex;
            public uint CandidateEpoch;
            public int CommitSerial;
            public int VertexCount;
            public int IndexCount;
            public bool Failed;
            public bool Completed;
        }

        private sealed class MeshBatch
        {
            public int Serial;
            public int Generation;
            public float StartedAt;
            public int Remaining;
            public readonly List<BatchItem> Items = new List<BatchItem>();
        }

        public ulong EpochDebt
        {
            get
            {
                ulong debt = 0;
                for (int i = 0; i < _blocks.Count; i++)
                {
                    Block block = _blocks[i];
                    if (block.TargetEpoch > block.ProcessedEpoch)
                        debt += (ulong)(block.TargetEpoch - block.ProcessedEpoch);
                }
                return debt;
            }
        }

        public int VisibleBlockCount
        {
            get
            {
                int count = 0;
                for (int i = 0; i < _blocks.Count; i++)
                    if (_blocks[i].Built && _blocks[i].IndexCount > 0)
                        count++;
                return count;
            }
        }

        public long CommittedVertexCount
        {
            get
            {
                long count = 0;
                for (int i = 0; i < _blocks.Count; i++)
                    count += _blocks[i].VertexCount;
                return count;
            }
        }

        public long CommittedIndexCount
        {
            get
            {
                long count = 0;
                for (int i = 0; i < _blocks.Count; i++)
                    count += _blocks[i].IndexCount;
                return count;
            }
        }

        public long RecentlySubmittedVertexCount
        {
            get
            {
                long total = 0;
                int oldestAcceptedFrame = Time.frameCount - 2;
                for (int i = 0; i < _blocks.Count; i++)
                {
                    GPUMeshRenderer renderer = _blocks[i].Renderer;
                    if (renderer == null || renderer.LastSubmittedFrame < oldestAcceptedFrame ||
                        renderer.LastSubmittedVertexCount <= 0)
                        continue;
                    total += renderer.LastSubmittedVertexCount;
                }
                return total;
            }
        }

        public string CompactStats => Failed
            ? $"块失败[{FailureReason}]"
            : $"块{VisibleBlockCount}/{BlockCount} 队{_queuedBlockCount} " +
              $"批途{(_activeBatch != null ? 1 : 0)} " +
              $"读{_inFlightCommits} 旧{_staleCandidateDiscardCount} 复{CommitWatchdogResets}";

        public void RequestImmediateDirtyLedgerRefresh()
        {
            if (!_disposed)
                _nextReadbackTime = 0f;
        }

        public InfiniTamBlockMeshPipeline(
            VolumeIntegrator volume,
            ComputeShader compute,
            Material material,
            Transform parent,
            int layer,
            int haloVoxels,
            int maxBlocksPerBatch,
            float dirtyReadbackHz,
            float vertexBudgetPercent,
            Action<GPUSurfaceNets> extract)
        {
            _volume = volume ?? throw new ArgumentNullException(nameof(volume));
            _compute = compute ?? throw new ArgumentNullException(nameof(compute));
            _material = material;
            _parent = parent;
            _layer = layer;
            _haloVoxels = Mathf.Max(1, haloVoxels);
            _maxBlocksPerBatch = Mathf.Max(1, maxBlocksPerBatch);
            _dirtyReadbackHz = Mathf.Max(0.5f, dirtyReadbackHz);
            _vertexBudgetPercent = Mathf.Clamp(vertexBudgetPercent, 0.01f, 0.5f);
            _extract = extract ?? throw new ArgumentNullException(nameof(extract));

            _volume.SetDirtyBoundaryHalo(_haloVoxels);
            BuildLayout();
            _volume.Cleared += OnVolumeCleared;
            _volume.TopologyInvalidated += OnTopologyInvalidated;
        }

        public void Tick()
        {
            if (_disposed || Failed)
                return;

            RequestDirtyLedgerIfDue();
            if (_activeBatch == null && _queuedBlockCount > 0)
                StartNextBatch();

            if (_activeBatch != null &&
                Time.realtimeSinceStartup - _activeBatch.StartedAt > BatchWatchdogSeconds)
                ResetTimedOutBatch();

            TryFinishInitialBuild();
        }

        public void SetVisible(bool visible)
        {
            _visible = visible;
            ApplyVisibility();
        }

        private void BuildLayout()
        {
            int chunkSize = _volume.ExtractionChunkSize;
            int3 voxels = _volume.VoxelCount;
            int3 cellCount = math.max(voxels - 1, 0);
            _blockCount = (cellCount + chunkSize - 1) / chunkSize;

            for (int z = 0; z < _blockCount.z; z++)
            for (int y = 0; y < _blockCount.y; y++)
            for (int x = 0; x < _blockCount.x; x++)
            {
                int3 coordinate = new int3(x, y, z);
                int3 coreMin = coordinate * chunkSize;
                int3 coreMax = math.min(coreMin + chunkSize, voxels - 1);
                int3 mapMin = math.max(coreMin - _haloVoxels, 0);
                int3 mapMax = math.min(coreMax + _haloVoxels + 1, voxels);
                _blocks.Add(new Block
                {
                    Index = Flatten(coordinate),
                    Coordinate = coordinate,
                    CoreMin = coreMin,
                    CoreMax = coreMax,
                    MapMin = mapMin,
                    MapCount = mapMax - mapMin
                });
            }
        }

        private void EnsureBlockResources(Block block)
        {
            if (block.Surface != null)
                return;

            block.Surface = new GPUSurfaceNets(_compute)
            {
                MinMeshWeight = _volume.MinMeshWeight,
                SmoothIterations = 0,
                SmoothLambda = 0f,
                SmoothBeta = 0f,
                TemporalAlphaMax = 1f,
                TemporalAlphaMin = 1f,
                TemporalDecayRate = 0f,
                ConvergenceThreshold = 0f,
                TemporalDeadzone = 0f,
                StrictObservedEdges = false,
                CandidateHistoryUpdateEnabled = false,
                FoundationTopologyMode = DirectTenCentimeterExperiment,
                FoundationCellStride = DirectTenCentimeterExperiment ? 2 : 1,
                FoundationFeatureRecognition = NativeTenCentimeterCreaseExperiment,
                FoundationNormalClusterDotMin = 0.8660254f,
                FoundationCornerNormalDotMax = 0.7071068f,
                FoundationChamferWidthVoxels = 0f,
                FoundationConstrainedSimplification = false,
                FoundationVisibleSkirtVoxels = 0,
                VisualQualityDiagnosticsEnabled = false,
                DiagnosticRoiEnabled = false
            };
            block.Surface.EnsureBuffers(_volume.VoxelCount, block.MapMin,
                block.MapCount, block.CoreMin, block.CoreMax,
                _vertexBudgetPercent);

            block.GameObject = new GameObject($"InfiniTAM Block {block.Coordinate}");
            block.GameObject.layer = _layer;
            block.GameObject.transform.SetParent(_parent, false);
            block.Renderer = block.GameObject.AddComponent<GPUMeshRenderer>();
            block.Renderer.GpuMeshMaterial = _material;
            block.Renderer.Initialize(block.Surface,
                block.Surface.GetCoreBounds(_volume.VoxelSize));
            block.Renderer.SetStrictObservedDisplay(false);
            block.Renderer.SetJointDiagnosticDisplay(false);
            block.Renderer.SetTemporalIllegalCandidateActive(false);
            block.Renderer.SetProductGridDisplay(false);
            block.Renderer.SetPinkIsolation(true);
            block.Renderer.RenderVisible = false;
        }

        private void StartNextBatch()
        {
            var batch = new MeshBatch
            {
                Serial = ++_nextBatchSerial,
                Generation = _generation,
                StartedAt = Time.realtimeSinceStartup
            };

            int budget = Mathf.Min(_maxBlocksPerBatch, _queuedBlockCount);
            while (budget-- > 0)
            {
                int index = DequeueNextQueuedBlock();
                if (index < 0)
                    break;
                Block block = _blocks[index];
                if (block.CommitPending || block.ProcessedEpoch >= block.TargetEpoch)
                    continue;

                batch.Items.Add(new BatchItem
                {
                    BlockIndex = index,
                    CandidateEpoch = block.TargetEpoch,
                    CommitSerial = ++block.CommitSerial
                });
            }

            if (batch.Items.Count == 0)
                return;

            batch.Remaining = batch.Items.Count;
            _activeBatch = batch;
            for (int i = 0; i < batch.Items.Count; i++)
            {
                BatchItem item = batch.Items[i];
                Block block = _blocks[item.BlockIndex];
                try
                {
                    EnsureBlockResources(block);
                    _extract(block.Surface);
                    block.CommitPending = true;
                    _inFlightCommits++;
                    RequestBatchItemReadback(batch, item);
                }
                catch (Exception ex)
                {
                    Fail($"block {block.Coordinate} extraction failed: {ex.Message}");
                    return;
                }
            }
        }

        private void RequestBatchItemReadback(MeshBatch batch, BatchItem item)
        {
            Block block = _blocks[item.BlockIndex];
            AsyncGPUReadback.Request(block.Surface.CountersBuffer, request =>
            {
                if (_disposed || batch != _activeBatch ||
                    batch.Generation != _generation ||
                    item.CommitSerial != block.CommitSerial)
                    return;

                block.CommitPending = false;
                _inFlightCommits = Mathf.Max(0, _inFlightCommits - 1);
                if (request.hasError)
                {
                    item.Failed = true;
                }
                else
                {
                    var counters = request.GetData<uint>();
                    item.VertexCount = counters.Length > 0
                        ? Mathf.Max(0, (int)counters[0]) : 0;
                    item.IndexCount = counters.Length > 1
                        ? Mathf.Max(0, (int)counters[1]) : 0;

                    // Populate a private back snapshot, but keep the old front
                    // visible until every member of this batch is complete.
                    block.Back ??= new GPUChunkMeshSnapshot();
                    if (item.VertexCount > 0 && item.IndexCount >= 3)
                        block.Surface.CopyCurrentMeshTo(block.Back,
                            item.VertexCount, item.IndexCount);
                    else
                        block.Back.Clear();
                }

                item.Completed = true;
                batch.Remaining = Mathf.Max(0, batch.Remaining - 1);
                if (batch.Remaining == 0)
                    FinishBatch(batch);
            });
        }

        private void FinishBatch(MeshBatch batch)
        {
            if (_disposed || batch != _activeBatch ||
                batch.Generation != _generation)
                return;

            int acceptedInBatch = 0;
            for (int i = 0; i < batch.Items.Count; i++)
            {
                BatchItem item = batch.Items[i];
                Block block = _blocks[item.BlockIndex];
                if (item.Failed)
                {
                    QueueBlock(block.Index, block.TargetEpoch, true);
                    continue;
                }

                // A later fusion epoch supersedes this private back result.
                // Keep the immutable front and carry one dirty bit forward.
                if (item.CandidateEpoch < block.TargetEpoch)
                {
                    _staleCandidateDiscardCount++;
                    QueueBlock(block.Index, block.TargetEpoch, true);
                    continue;
                }

                GPUChunkMeshSnapshot oldFront = block.Front;
                block.Front = block.Back;
                block.Back = oldFront;
                block.Renderer.SetMeshSource(block.Front);
                block.VertexCount = item.VertexCount;
                block.IndexCount = item.IndexCount;
                block.ProcessedEpoch = item.CandidateEpoch;
                block.Built = true;
                _acceptedCommitCount++;
                acceptedInBatch++;
            }

            _completedBatchCount++;
            if (acceptedInBatch > 0)
                _publishedBatchCount++;
            _activeBatch = null;
            if (_queuedBlockCount == 0 && _inFlightCommits == 0)
                _initialNeedsSettlement = false;
            ApplyVisibility();
            if (!InitialBuildComplete && _queuedBlockCount == 0)
                _nextReadbackTime = 0f;
            TryFinishInitialBuild();
        }

        private void ResetTimedOutBatch()
        {
            MeshBatch batch = _activeBatch;
            if (batch == null)
                return;

            for (int i = 0; i < batch.Items.Count; i++)
            {
                BatchItem item = batch.Items[i];
                Block block = _blocks[item.BlockIndex];
                if (!item.Completed)
                {
                    block.CommitSerial++;
                    block.CommitPending = false;
                    _inFlightCommits = Mathf.Max(0, _inFlightCommits - 1);
                }
                QueueBlock(block.Index, block.TargetEpoch, true);
            }
            _activeBatch = null;
            CommitWatchdogResets++;
            Logger.Warning($"InfiniTAM mesh batch {batch.Serial} timed out; " +
                           "its dirty set was coalesced into the next batch.");
        }

        private void RequestDirtyLedgerIfDue()
        {
            if (_readbackPending || Time.realtimeSinceStartup < _nextReadbackTime)
                return;

            ComputeBuffer owner = _volume.DirtyChunkEpochs;
            ComputeBuffer boundary = _volume.DirtyBoundaryEpochs;
            if (owner == null || owner.count != _blocks.Count ||
                boundary == null || boundary.count != _blocks.Count * 6)
            {
                Fail("dirty owner/boundary ledger is unavailable or has the wrong size");
                return;
            }

            _readbackPending = true;
            _ownerReady = false;
            _boundaryReady = false;
            _ledgerFailed = false;
            _ownerSnapshot = null;
            _boundarySnapshot = null;
            _nextReadbackTime = Time.realtimeSinceStartup + 1f / _dirtyReadbackHz;
            int requestGeneration = _generation;

            AsyncGPUReadback.Request(owner, request =>
            {
                if (_disposed || requestGeneration != _generation)
                    return;
                if (request.hasError)
                    _ledgerFailed = true;
                else
                {
                    var data = request.GetData<uint>();
                    _ownerSnapshot = new uint[data.Length];
                    data.CopyTo(_ownerSnapshot);
                }
                _ownerReady = true;
                FinishDirtyLedger(requestGeneration);
            });

            AsyncGPUReadback.Request(boundary, request =>
            {
                if (_disposed || requestGeneration != _generation)
                    return;
                if (request.hasError)
                    _ledgerFailed = true;
                else
                {
                    var data = request.GetData<uint>();
                    _boundarySnapshot = new uint[data.Length];
                    data.CopyTo(_boundarySnapshot);
                }
                _boundaryReady = true;
                FinishDirtyLedger(requestGeneration);
            });
        }

        private void FinishDirtyLedger(int requestGeneration)
        {
            if (_disposed || requestGeneration != _generation ||
                !_ownerReady || !_boundaryReady)
                return;

            _readbackPending = false;
            if (_ledgerFailed || _ownerSnapshot == null || _boundarySnapshot == null)
            {
                if (++_readbackFailures >= 3)
                    Fail("dirty owner/boundary ledger GPU readback failed three times");
                return;
            }

            _readbackFailures = 0;
            _dirtyLedgerApplyCount++;
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                uint ownerEpoch = _ownerSnapshot[i];
                if (ownerEpoch > block.LastOwnerEpoch)
                {
                    block.LastOwnerEpoch = ownerEpoch;
                    QueueBlock(i, ownerEpoch);
                }

                int faceBase = i * 6;
                for (int face = 0; face < FaceNeighbours.Length; face++)
                {
                    uint boundaryEpoch = _boundarySnapshot[faceBase + face];
                    if (boundaryEpoch == 0 || boundaryEpoch <= block.LastBoundaryEpoch[face])
                        continue;
                    block.LastBoundaryEpoch[face] = boundaryEpoch;
                    int3 neighbour = block.Coordinate + FaceNeighbours[face];
                    if (math.any(neighbour < 0) || math.any(neighbour >= _blockCount))
                        continue;
                    QueueBlock(Flatten(neighbour), boundaryEpoch);
                }
            }

            // Unobserved blocks are already valid empty blocks.  Marking them
            // complete prevents an expensive full-volume empty proof at takeover.
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                if (block.LastOwnerEpoch == 0 && block.TargetEpoch == 0 &&
                    !block.Queued && !block.CommitPending && !block.Built)
                    block.Built = true;
            }
            _firstLedgerApplied = true;
            _initialNeedsSettlement = _queuedBlockCount > 0 ||
                _activeBatch != null;
            TryFinishInitialBuild();
        }

        private void QueueBlock(int index, uint epoch, bool force = false)
        {
            Block block = _blocks[index];
            block.TargetEpoch = math.max(block.TargetEpoch, epoch);
            if ((!force && block.ProcessedEpoch >= block.TargetEpoch) || block.Queued)
                return;
            block.Queued = true;
            if (!InitialBuildComplete)
                _initialNeedsSettlement = true;
            _queuedBlockCount++;
        }

        private int DequeueNextQueuedBlock()
        {
            int bestIndex = -1;
            bool bestVisible = false;
            uint bestDebt = 0;

            // The baseline currently owns only a handful of room-scale blocks,
            // so a linear scan is cheaper and more deterministic than maintaining
            // another heap.  Visible stale fronts go first; within that class the
            // largest epoch debt wins, then the stable block index breaks ties.
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                if (!block.Queued)
                    continue;
                bool visible = block.Built && block.IndexCount > 0;
                uint debt = block.TargetEpoch > block.ProcessedEpoch
                    ? block.TargetEpoch - block.ProcessedEpoch : 0u;
                if (bestIndex >= 0 &&
                    (visible ? 1 : 0) < (bestVisible ? 1 : 0))
                    continue;
                if (bestIndex >= 0 && visible == bestVisible && debt < bestDebt)
                    continue;
                if (bestIndex >= 0 && visible == bestVisible && debt == bestDebt &&
                    i >= bestIndex)
                    continue;

                bestIndex = i;
                bestVisible = visible;
                bestDebt = debt;
            }

            if (bestIndex >= 0)
            {
                _blocks[bestIndex].Queued = false;
                _queuedBlockCount = Mathf.Max(0, _queuedBlockCount - 1);
            }
            return bestIndex;
        }

        private void TryFinishInitialBuild()
        {
            if (InitialBuildComplete || !_firstLedgerApplied ||
                _initialNeedsSettlement || _readbackPending ||
                _queuedBlockCount > 0 || _activeBatch != null)
                return;
            for (int i = 0; i < _blocks.Count; i++)
                if (!_blocks[i].Built)
                    return;
            InitialBuildComplete = true;
            ApplyVisibility();
            Logger.Info($"InfiniTAM block front ready: blocks={_blocks.Count}, visible={VisibleBlockCount}");
        }

        private int Flatten(int3 coordinate)
        {
            return coordinate.x + _blockCount.x *
                (coordinate.y + _blockCount.y * coordinate.z);
        }

        private void ApplyVisibility()
        {
            for (int i = 0; i < _blocks.Count; i++)
                ApplyBlockVisibility(_blocks[i]);
        }

        private void ApplyBlockVisibility(Block block)
        {
            if (block.Renderer != null)
                block.Renderer.RenderVisible = _visible && InitialBuildComplete &&
                    !Failed && block.Built && block.Front != null && block.IndexCount > 0;
        }

        private void OnVolumeCleared()
        {
            Reset(knownEmpty: true);
        }

        private void OnTopologyInvalidated()
        {
            Reset(knownEmpty: false);
        }

        private void Reset(bool knownEmpty)
        {
            _generation++;
            _readbackPending = false;
            _ownerReady = false;
            _boundaryReady = false;
            _ledgerFailed = false;
            _ownerSnapshot = null;
            _boundarySnapshot = null;
            _queuedBlockCount = 0;
            _inFlightCommits = 0;
            _activeBatch = null;
            _firstLedgerApplied = knownEmpty;
            _initialNeedsSettlement = !knownEmpty;
            InitialBuildComplete = knownEmpty;
            uint epoch = knownEmpty ? _volume.DirtyEpoch : 0u;
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                block.CommitSerial++;
                block.CommitPending = false;
                block.Queued = false;
                block.Built = knownEmpty;
                block.TargetEpoch = epoch;
                block.ProcessedEpoch = epoch;
                block.LastOwnerEpoch = epoch;
                block.VertexCount = 0;
                block.IndexCount = 0;
                Array.Clear(block.LastBoundaryEpoch, 0, block.LastBoundaryEpoch.Length);
                block.Dispose();
            }
            ApplyVisibility();
        }

        private void Fail(string reason)
        {
            Failed = true;
            FailureReason = reason;
            ApplyVisibility();
            Logger.Error($"InfiniTAM block extraction disabled; whole-volume fallback remains active: {reason}");
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _generation++;
            _volume.Cleared -= OnVolumeCleared;
            _volume.TopologyInvalidated -= OnTopologyInvalidated;
            for (int i = 0; i < _blocks.Count; i++)
                _blocks[i].Dispose();
            _blocks.Clear();
            _queuedBlockCount = 0;
            _activeBatch = null;
        }
    }
}

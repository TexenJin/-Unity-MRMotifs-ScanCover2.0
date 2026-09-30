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
            public readonly uint[] LatestBoundaryEpoch = new uint[6];
            public readonly uint[] RequiredBoundaryEpoch = new uint[6];
            public bool Built;
            public bool Queued;
            public bool CommitPending;
            public bool HasStagedCandidate;
            public int CommitSerial;
            public int PublishVisitSerial;
            public int VertexCount;
            public int IndexCount;
            public uint StagedEpoch;
            public int StagedVertexCount;
            public int StagedIndexCount;
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
        private int _stagedBlockCount;
        private int _nextBatchSerial;
        private int _nextPublishVisitSerial;
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
        private uint[] _ownerSnapshot;
        private uint[] _boundarySnapshot;
        private float _nextReadbackTime;
        private MeshBatch _activeBatch;
        private readonly List<int> _publishComponent = new List<int>();

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
        // Isolated native-10 cm crease trial. Device review showed that its
        // unconstrained corner solve can turn transient normal families into
        // long scar faces. Keep the implementation available, but return the
        // production front to the ordinary Surface-Nets crossing mean until a
        // bounded crease solver can be validated separately.
        internal const bool NativeTenCentimeterCreaseExperiment = false;

        public bool Failed { get; private set; }
        public string FailureReason { get; private set; } = string.Empty;
        public bool InitialBuildComplete { get; private set; }
        public int CommitWatchdogResets { get; private set; }
        public int BlockCount => _blocks.Count;
        public int PendingCount => _queuedBlockCount + _inFlightCommits + _stagedBlockCount;
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
                    if (block.Queued || block.CommitPending || block.HasStagedCandidate ||
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

            // Counter readback completion only makes a batch eligible for
            // publication.  The snapshot copy is GPU work in its own right,
            // so execute it here on the next scanner-owned mesh-work slot
            // instead of submitting it from an arbitrary readback callback.
            // Return after publication: one slot owns either snapshot copy or
            // the next extraction dispatch, never both.
            if (_activeBatch != null && _activeBatch.Remaining == 0)
            {
                CommitReadyBatch(_activeBatch);
                TryFinishInitialBuild();
                return;
            }

            if (_activeBatch == null && _queuedBlockCount > 0)
                StartNextBatch();

            // A completed batch may deliberately wait for its next scheduled
            // publication slot.  Only missing readbacks are watchdog failures.
            if (_activeBatch != null && _activeBatch.Remaining > 0 &&
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
                // A missing TSDF endpoint is unknown space, not a zero
                // crossing.  Letting it close a face creates the long caps and
                // scar-like bridges that remain after the real surface heals.
                StrictObservedEdges = true,
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
            block.Renderer.SetTrueLineQuadTopology(true);
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
                if (block.CommitPending || block.HasStagedCandidate ||
                    block.ProcessedEpoch >= block.TargetEpoch)
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
                }

                item.Completed = true;
                batch.Remaining = Mathf.Max(0, batch.Remaining - 1);
            });
        }

        private void CommitReadyBatch(MeshBatch batch)
        {
            if (_disposed || batch != _activeBatch || batch.Remaining != 0 ||
                batch.Generation != _generation)
                return;

            try
            {
                for (int i = 0; i < batch.Items.Count; i++)
                {
                    BatchItem item = batch.Items[i];
                    if (item.Failed)
                        continue;

                    Block block = _blocks[item.BlockIndex];
                    // Populate a private back snapshot while the old front
                    // remains visible.  This dispatch now runs only inside the
                    // scanner's phased mesh-work budget.
                    block.Back ??= new GPUChunkMeshSnapshot();
                    if (item.VertexCount > 0 && item.IndexCount >= 3)
                        block.Surface.CopyCurrentMeshTo(block.Back,
                            item.VertexCount, item.IndexCount);
                    else
                        block.Back.Clear();
                }
            }
            catch (Exception ex)
            {
                Fail($"batch {batch.Serial} snapshot publication failed: {ex.Message}");
                return;
            }

            FinishBatch(batch);
        }

        private void FinishBatch(MeshBatch batch)
        {
            if (_disposed || batch != _activeBatch ||
                batch.Generation != _generation)
                return;

            for (int i = 0; i < batch.Items.Count; i++)
            {
                BatchItem item = batch.Items[i];
                Block block = _blocks[item.BlockIndex];
                if (item.Failed)
                {
                    QueueBlock(block.Index, block.TargetEpoch, true);
                    continue;
                }

                if (block.Built && item.CandidateEpoch <= block.ProcessedEpoch)
                {
                    _staleCandidateDiscardCount++;
                    if (block.TargetEpoch > block.ProcessedEpoch)
                        QueueBlock(block.Index, block.TargetEpoch, true);
                    continue;
                }

                // Interior changes retain bounded-lag publication. A shared
                // boundary uses a latched requirement: both sides must cover
                // this round, while newer fusion remains debt for the next
                // round instead of invalidating useful completed work forever.
                if (PendingBoundaryRequirement(block) > item.CandidateEpoch)
                {
                    _staleCandidateDiscardCount++;
                    QueueBlock(block.Index, block.TargetEpoch, true);
                    continue;
                }

                block.HasStagedCandidate = true;
                block.StagedEpoch = item.CandidateEpoch;
                block.StagedVertexCount = item.VertexCount;
                block.StagedIndexCount = item.IndexCount;
                _stagedBlockCount++;
                if (block.Queued)
                {
                    block.Queued = false;
                    _queuedBlockCount = Mathf.Max(0, _queuedBlockCount - 1);
                }
            }

            _completedBatchCount++;
            _activeBatch = null;
            if (TryPublishReadyStagedComponents() > 0)
                _publishedBatchCount++;
            ApplyVisibility();
            if (!InitialBuildComplete && _queuedBlockCount == 0)
                _nextReadbackTime = 0f;
            TryFinishInitialBuild();
        }

        private uint PendingBoundaryRequirement(Block block)
        {
            uint required = 0;
            for (int face = 0; face < block.RequiredBoundaryEpoch.Length; face++)
            {
                uint epoch = block.RequiredBoundaryEpoch[face];
                if (epoch > block.ProcessedEpoch)
                    required = math.max(required, epoch);
            }
            return required;
        }

        private void InvalidateStagedCandidate(Block block)
        {
            if (!block.HasStagedCandidate)
                return;
            block.HasStagedCandidate = false;
            block.StagedEpoch = 0;
            block.StagedVertexCount = 0;
            block.StagedIndexCount = 0;
            _stagedBlockCount = Mathf.Max(0, _stagedBlockCount - 1);
            _staleCandidateDiscardCount++;
        }

        /// <summary>
        /// Publish every currently ready seam-connected component. Extraction
        /// and snapshot copies still happened one block per mesh-work slot; this
        /// method only swaps already prepared fronts before LateUpdate draws.
        /// </summary>
        private int TryPublishReadyStagedComponents()
        {
            int published = 0;
            bool madeProgress;
            do
            {
                madeProgress = false;
                for (int i = 0; i < _blocks.Count; i++)
                {
                    if (!_blocks[i].HasStagedCandidate ||
                        !TryCollectReadyPublishComponent(i))
                        continue;
                    published += PublishCollectedComponent();
                    madeProgress = true;
                    break;
                }
            } while (madeProgress);
            return published;
        }

        private bool TryCollectReadyPublishComponent(int startIndex)
        {
            Block start = _blocks[startIndex];
            if (!start.HasStagedCandidate)
                return false;

            _nextPublishVisitSerial++;
            if (_nextPublishVisitSerial == 0)
            {
                for (int i = 0; i < _blocks.Count; i++)
                    _blocks[i].PublishVisitSerial = 0;
                _nextPublishVisitSerial = 1;
            }

            int visitSerial = _nextPublishVisitSerial;
            _publishComponent.Clear();
            _publishComponent.Add(startIndex);
            start.PublishVisitSerial = visitSerial;

            for (int cursor = 0; cursor < _publishComponent.Count; cursor++)
            {
                Block block = _blocks[_publishComponent[cursor]];
                for (int face = 0; face < FaceNeighbours.Length; face++)
                {
                    uint requiredEpoch = block.RequiredBoundaryEpoch[face];
                    if (requiredEpoch == 0)
                        continue;

                    int3 neighbourCoordinate = block.Coordinate + FaceNeighbours[face];
                    if (math.any(neighbourCoordinate < 0) ||
                        math.any(neighbourCoordinate >= _blockCount))
                        continue;
                    Block neighbour = _blocks[Flatten(neighbourCoordinate)];
                    bool blockReady = block.ProcessedEpoch >= requiredEpoch;
                    bool neighbourReady = neighbour.ProcessedEpoch >= requiredEpoch;
                    if (blockReady && neighbourReady)
                        continue;

                    if (!blockReady && (!block.HasStagedCandidate ||
                        block.StagedEpoch < requiredEpoch))
                        return false;
                    if (!neighbourReady && (!neighbour.HasStagedCandidate ||
                        neighbour.StagedEpoch < requiredEpoch))
                        return false;

                    // If the already-ready side also has a newer staged front,
                    // include it in the same visible swap rather than letting it
                    // advance alone while its neighbour is still catching up.
                    if (!neighbour.HasStagedCandidate ||
                        neighbour.PublishVisitSerial == visitSerial)
                        continue;
                    neighbour.PublishVisitSerial = visitSerial;
                    _publishComponent.Add(neighbour.Index);
                }
            }
            return true;
        }

        private int PublishCollectedComponent()
        {
            int published = 0;
            for (int i = 0; i < _publishComponent.Count; i++)
            {
                Block block = _blocks[_publishComponent[i]];
                if (!block.HasStagedCandidate)
                    continue;

                GPUChunkMeshSnapshot oldFront = block.Front;
                block.Front = block.Back;
                block.Back = oldFront;
                block.Renderer.SetMeshSource(block.Front);
                block.VertexCount = block.StagedVertexCount;
                block.IndexCount = block.StagedIndexCount;
                block.ProcessedEpoch = block.StagedEpoch;
                block.Built = true;
                block.HasStagedCandidate = false;
                block.StagedEpoch = 0;
                block.StagedVertexCount = 0;
                block.StagedIndexCount = 0;
                _stagedBlockCount = Mathf.Max(0, _stagedBlockCount - 1);
                _acceptedCommitCount++;
                published++;
            }

            AdvanceSatisfiedBoundaryRequirements();

            // A newer interior or boundary epoch may have arrived while this
            // component waited. Preserve one coalesced follow-up per block.
            for (int i = 0; i < _publishComponent.Count; i++)
            {
                Block block = _blocks[_publishComponent[i]];
                if (block.TargetEpoch > block.ProcessedEpoch)
                    QueueBlock(block.Index, block.TargetEpoch, true);
            }
            return published;
        }

        private void AdvanceSatisfiedBoundaryRequirements()
        {
            // Visit each undirected face once (+X,+Y,+Z). Once both published
            // fronts cover the latched epoch, roll any newer observed epoch
            // into the next transaction without revoking the completed one.
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                for (int face = 1; face < FaceNeighbours.Length; face += 2)
                {
                    int3 neighbourCoordinate = block.Coordinate + FaceNeighbours[face];
                    if (math.any(neighbourCoordinate < 0) ||
                        math.any(neighbourCoordinate >= _blockCount))
                        continue;
                    Block neighbour = _blocks[Flatten(neighbourCoordinate)];
                    int oppositeFace = face ^ 1;
                    uint required = math.max(block.RequiredBoundaryEpoch[face],
                        neighbour.RequiredBoundaryEpoch[oppositeFace]);
                    if (required == 0 || block.ProcessedEpoch < required ||
                        neighbour.ProcessedEpoch < required)
                        continue;
                    uint latest = math.max(block.LatestBoundaryEpoch[face],
                        neighbour.LatestBoundaryEpoch[oppositeFace]);
                    if (latest <= required)
                        continue;
                    block.RequiredBoundaryEpoch[face] = latest;
                    neighbour.RequiredBoundaryEpoch[oppositeFace] = latest;
                }
            }
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
                    RegisterBoundaryPair(block, face,
                        _blocks[Flatten(neighbour)], boundaryEpoch);
                }
            }

            if (TryPublishReadyStagedComponents() > 0)
            {
                _publishedBatchCount++;
                ApplyVisibility();
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
            TryFinishInitialBuild();
        }

        private void RegisterBoundaryPair(
            Block block, int face, Block neighbour, uint epoch)
        {
            int oppositeFace = face ^ 1;
            uint latest = math.max(epoch,
                math.max(block.LatestBoundaryEpoch[face],
                    neighbour.LatestBoundaryEpoch[oppositeFace]));
            block.LatestBoundaryEpoch[face] = latest;
            neighbour.LatestBoundaryEpoch[oppositeFace] = latest;

            uint required = math.max(block.RequiredBoundaryEpoch[face],
                neighbour.RequiredBoundaryEpoch[oppositeFace]);
            if (required == 0 ||
                (block.ProcessedEpoch >= required &&
                 neighbour.ProcessedEpoch >= required))
                required = latest;
            block.RequiredBoundaryEpoch[face] = required;
            neighbour.RequiredBoundaryEpoch[oppositeFace] = required;

            // Only the latched round can invalidate a too-old stage. Later
            // observations stay in LatestBoundaryEpoch for the next round.
            if (block.HasStagedCandidate && block.StagedEpoch < required)
                InvalidateStagedCandidate(block);
            if (neighbour.HasStagedCandidate &&
                neighbour.StagedEpoch < required)
                InvalidateStagedCandidate(neighbour);

            QueueBlock(block.Index, latest);
            QueueBlock(neighbour.Index, latest);
        }

        private void QueueBlock(int index, uint epoch, bool force = false)
        {
            Block block = _blocks[index];
            block.TargetEpoch = math.max(block.TargetEpoch, epoch);
            // A prepared snapshot cannot be overwritten before its seam group
            // publishes. Any newer debt remains in TargetEpoch and is queued
            // immediately after that atomic front swap.
            if (block.HasStagedCandidate)
                return;
            if ((!force && block.ProcessedEpoch >= block.TargetEpoch) || block.Queued)
                return;
            block.Queued = true;
            _queuedBlockCount++;
        }

        private int DequeueNextQueuedBlock()
        {
            int bestIndex = -1;
            int bestPriority = int.MinValue;
            uint bestDebt = 0;

            // The baseline currently owns only a handful of room-scale blocks,
            // so a linear scan is cheaper and more deterministic than maintaining
            // another heap. During the first build, blocks without any committed
            // front go first so takeover cannot expose a partial block set. After
            // takeover, visibility is deliberately not a priority class: a visible
            // block that stays hot must not starve a newly observed empty neighbour
            // at the shared face. All queued blocks then compete by epoch debt, and
            // the stable block index breaks exact ties.
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                if (!block.Queued)
                    continue;
                int priority = !InitialBuildComplete
                    ? (block.Built ? 0 : 1)
                    : 0;
                uint debt = block.TargetEpoch > block.ProcessedEpoch
                    ? block.TargetEpoch - block.ProcessedEpoch : 0u;
                if (bestIndex >= 0 && priority < bestPriority)
                    continue;
                if (bestIndex >= 0 && priority == bestPriority && debt < bestDebt)
                    continue;
                if (bestIndex >= 0 && priority == bestPriority && debt == bestDebt &&
                    i >= bestIndex)
                    continue;

                bestIndex = i;
                bestPriority = priority;
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
            if (InitialBuildComplete || !_firstLedgerApplied)
                return;

            // InitialBuildComplete means that every block has a usable first
            // front (including a proven-empty front), not that live fusion has
            // stopped producing dirty debt. Requiring an empty queue/readback
            // window makes takeover impossible during continuous scanning once
            // the scheduler intentionally publishes bounded-lag candidates and
            // immediately queues their latest epochs.
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
            _stagedBlockCount = 0;
            _activeBatch = null;
            _firstLedgerApplied = knownEmpty;
            InitialBuildComplete = knownEmpty;
            uint epoch = knownEmpty ? _volume.DirtyEpoch : 0u;
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                block.CommitSerial++;
                block.CommitPending = false;
                block.Queued = false;
                block.HasStagedCandidate = false;
                block.Built = knownEmpty;
                block.TargetEpoch = epoch;
                block.ProcessedEpoch = epoch;
                block.LastOwnerEpoch = epoch;
                block.VertexCount = 0;
                block.IndexCount = 0;
                block.StagedEpoch = 0;
                block.StagedVertexCount = 0;
                block.StagedIndexCount = 0;
                Array.Clear(block.LastBoundaryEpoch, 0, block.LastBoundaryEpoch.Length);
                Array.Clear(block.LatestBoundaryEpoch, 0, block.LatestBoundaryEpoch.Length);
                Array.Clear(block.RequiredBoundaryEpoch, 0, block.RequiredBoundaryEpoch.Length);
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
            _stagedBlockCount = 0;
            _activeBatch = null;
        }
    }
}

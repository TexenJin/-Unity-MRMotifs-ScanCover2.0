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
            public uint CandidateEpoch;
            public uint LastOwnerEpoch;
            public readonly uint[] LastBoundaryEpoch = new uint[6];
            public bool Built;
            public bool Queued;
            public bool CommitPending;
            public bool ReadyToPublish;
            // A candidate already ready or still in flight when the operator
            // freezes presentation must not replace/clear the visible Front on
            // resume. It is discarded and freshly extracted instead.
            public bool DiscardAfterPresentationFreeze;
            public float CommitStartedAt;
            public int CommitSerial;
            public int PendingVertexCount;
            public int PendingIndexCount;
            public int VertexCount;
            public int IndexCount;
            public GameObject GameObject;
            public GPUSurfaceNets Surface;
            public GPUChunkMeshSnapshot Front;
            public GPUChunkMeshSnapshot Back;
            public GPUMeshRenderer Renderer;
            public MeshLayerEvidenceRenderer LayerEvidenceRenderer;

            public void Dispose()
            {
                if (Renderer != null)
                    Renderer.RenderVisible = false;
                if (LayerEvidenceRenderer != null)
                    LayerEvidenceRenderer.SetMode(0);
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
                LayerEvidenceRenderer = null;
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
        private readonly int _maxBlocksPerTick;
        private readonly float _dirtyReadbackHz;
        private readonly float _vertexBudgetPercent;
        private readonly Action<GPUSurfaceNets> _extract;
        private readonly Queue<int> _dirtyQueue = new Queue<int>();
        private readonly List<Block> _blocks = new List<Block>();
        private readonly Material _layerEvidencePointMaterial;
        private readonly Material _layerEvidenceRelationMaterial;

        private int3 _blockCount;
        private int _generation;
        private int _readbackFailures;
        private int _inFlightCommits;
        private bool _disposed;
        private bool _visible;
        private int _layerEvidenceMode;
        private bool _presentationFrozen;
        private bool _readbackPending;
        private bool _ownerReady;
        private bool _boundaryReady;
        private bool _ledgerFailed;
        private bool _firstLedgerApplied;
        private uint[] _ownerSnapshot;
        private uint[] _boundarySnapshot;
        private float _nextReadbackTime;

        private const int MaximumConcurrentCommits = 1;

        public bool Failed { get; private set; }
        public string FailureReason { get; private set; } = string.Empty;
        public bool InitialBuildComplete { get; private set; }
        public int CommitWatchdogResets { get; private set; }
        public int BlockCount => _blocks.Count;
        public int PendingCount
        {
            get
            {
                int ready = 0;
                for (int i = 0; i < _blocks.Count; i++)
                    if (_blocks[i].ReadyToPublish)
                        ready++;
                return _dirtyQueue.Count + _inFlightCommits + ready;
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
            : $"块{VisibleBlockCount}/{BlockCount} 队{_dirtyQueue.Count} " +
              $"途{_inFlightCommits} 复{CommitWatchdogResets}";

        public InfiniTamBlockMeshPipeline(
            VolumeIntegrator volume,
            ComputeShader compute,
            Material material,
            Transform parent,
            int layer,
            int haloVoxels,
            int maxBlocksPerTick,
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
            _maxBlocksPerTick = Mathf.Max(1, maxBlocksPerTick);
            _dirtyReadbackHz = Mathf.Max(0.5f, dirtyReadbackHz);
            _vertexBudgetPercent = Mathf.Clamp(vertexBudgetPercent, 0.01f, 0.5f);
            _extract = extract ?? throw new ArgumentNullException(nameof(extract));

            Shader layerEvidenceShader = Resources.Load<Shader>("MeshLayerEvidence");
            if (layerEvidenceShader != null)
                _layerEvidencePointMaterial = new Material(layerEvidenceShader)
                {
                    name = "QRS Same-Candidate Mesh Layer Evidence"
                };
            else
                Logger.Warning("MeshLayerEvidence shader missing; causal profile view is unavailable.");

            Shader layerRelationShader = Resources.Load<Shader>("MeshLayerRelations");
            if (layerRelationShader != null)
                _layerEvidenceRelationMaterial = new Material(layerRelationShader)
                {
                    name = "QRS Same-Candidate Mesh Layer Relations"
                };
            else
                Logger.Warning("MeshLayerRelations shader missing; causal links and emitting-cell outlines are unavailable.");

            _volume.SetDirtyBoundaryHalo(_haloVoxels);
            BuildLayout();
            _volume.Cleared += OnVolumeCleared;
            _volume.TopologyInvalidated += OnTopologyInvalidated;
        }

        public void Tick()
        {
            if (_disposed || Failed)
                return;

            // A-key is a literal visual freeze: do not let a completed async
            // readback swap a newer Front after the operator has frozen the
            // scene. Pending work resumes without loss on unfreeze.
            if (_presentationFrozen)
                return;

            RequestDirtyLedgerIfDue();

            // A counter readback only declares a private candidate ready. The
            // GPU snapshot copy and Front swap run here, in the scanner-owned
            // mesh slot. One slot therefore performs publication OR extraction,
            // never both and never from an asynchronous callback.
            for (int i = 0; i < _blocks.Count; i++)
            {
                if (!_blocks[i].ReadyToPublish)
                    continue;
                if (_blocks[i].DiscardAfterPresentationFreeze)
                {
                    DiscardFrozenCandidateAndRequeue(_blocks[i]);
                    return;
                }
                PublishReadyBlock(_blocks[i]);
                TryFinishInitialBuild();
                return;
            }

            int budget = Mathf.Min(1, Mathf.Min(_maxBlocksPerTick,
                Mathf.Max(0, MaximumConcurrentCommits - _inFlightCommits)));
            budget = Mathf.Max(0, budget);
            while (budget-- > 0 && _dirtyQueue.Count > 0)
            {
                int index = _dirtyQueue.Dequeue();
                Block block = _blocks[index];
                if (!block.Queued)
                {
                    budget++;
                    continue;
                }

                block.Queued = false;
                if (block.CommitPending || block.ProcessedEpoch >= block.TargetEpoch)
                    continue;

                try
                {
                    EnsureBlockResources(block);
                    uint candidateEpoch = block.TargetEpoch;
                    _extract(block.Surface);
                    block.CandidateEpoch = candidateEpoch;
                    block.CommitPending = true;
                    block.CommitStartedAt = Time.realtimeSinceStartup;
                    _inFlightCommits++;
                    int commitSerial = ++block.CommitSerial;
                    RequestCommit(block, candidateEpoch, commitSerial);
                }
                catch (Exception ex)
                {
                    Fail($"block {block.Coordinate} extraction failed: {ex.Message}");
                    return;
                }
            }

            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                if (!block.CommitPending ||
                    Time.realtimeSinceStartup - block.CommitStartedAt <= 10f)
                    continue;
                block.CommitSerial++;
                block.CommitPending = false;
                _inFlightCommits = Mathf.Max(0, _inFlightCommits - 1);
                CommitWatchdogResets++;
                QueueBlock(block.Index, block.TargetEpoch, true);
                Logger.Warning($"InfiniTAM block {block.Coordinate} readback timed out; requeued.");
            }

            TryFinishInitialBuild();
        }

        public void SetVisible(bool visible)
        {
            _visible = visible;
            ApplyVisibility();
        }

        public int LayerEvidenceMode => _layerEvidenceMode;

        public void SetLayerEvidenceMode(int mode)
        {
            _layerEvidenceMode = Mathf.Clamp(mode, 0, 3);
            ApplyVisibility();
        }

        public void SetPresentationFrozen(bool frozen)
        {
            if (_presentationFrozen == frozen)
                return;
            _presentationFrozen = frozen;
            if (frozen)
            {
                for (int i = 0; i < _blocks.Count; i++)
                {
                    Block block = _blocks[i];
                    if (block.CommitPending || block.ReadyToPublish)
                        block.DiscardAfterPresentationFreeze = true;
                }
                return;
            }

            // Completed candidates are private Back data. Throwing them away
            // cannot affect the published Front. Pending callbacks retain the
            // flag and are discarded by Tick after they become ready.
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                if (block.ReadyToPublish && block.DiscardAfterPresentationFreeze)
                    DiscardFrozenCandidateAndRequeue(block);
            }
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
                FoundationTopologyMode = false,
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
            if (_layerEvidencePointMaterial != null)
            {
                block.LayerEvidenceRenderer =
                    block.GameObject.AddComponent<MeshLayerEvidenceRenderer>();
                block.LayerEvidenceRenderer.Initialize(_layerEvidencePointMaterial,
                    _layerEvidenceRelationMaterial,
                    block.Surface.GetCoreBounds(_volume.VoxelSize),
                    _volume.VoxelCount, _volume.VoxelSize);
                block.LayerEvidenceRenderer.SetMode(0);
            }
        }

        private void RequestCommit(Block block, uint candidateEpoch, int commitSerial)
        {
            int requestGeneration = _generation;
            AsyncGPUReadback.Request(block.Surface.CountersBuffer, request =>
            {
                if (_disposed || requestGeneration != _generation ||
                    commitSerial != block.CommitSerial)
                    return;

                block.CommitPending = false;
                _inFlightCommits = Mathf.Max(0, _inFlightCommits - 1);
                if (request.hasError)
                {
                    block.DiscardAfterPresentationFreeze = false;
                    QueueBlock(block.Index, candidateEpoch, true);
                    return;
                }

                var counters = request.GetData<uint>();
                int vertices = counters.Length > 0 ? Mathf.Max(0, (int)counters[0]) : 0;
                int indices = counters.Length > 1 ? Mathf.Max(0, (int)counters[1]) : 0;

                // A newer dirty epoch arriving during extraction does not make
                // this completed bounded-lag candidate useless. Publish it once,
                // then keep the newer epoch as one coalesced follow-up. Only a
                // candidate already superseded by a published Front is stale.
                if (block.Built && candidateEpoch <= block.ProcessedEpoch)
                {
                    block.DiscardAfterPresentationFreeze = false;
                    if (block.TargetEpoch > block.ProcessedEpoch)
                        QueueBlock(block.Index, block.TargetEpoch, true);
                    return;
                }

                block.CandidateEpoch = candidateEpoch;
                block.PendingVertexCount = vertices;
                block.PendingIndexCount = indices;
                block.ReadyToPublish = true;
            });
        }

        private void PublishReadyBlock(Block block)
        {
            if (block == null || !block.ReadyToPublish)
                return;

            int vertices = block.PendingVertexCount;
            int indices = block.PendingIndexCount;
            uint candidateEpoch = block.CandidateEpoch;
            try
            {
                if (vertices > 0 && indices >= 3)
                {
                    block.Back ??= new GPUChunkMeshSnapshot();
                    block.Surface.CopyCurrentMeshTo(block.Back, vertices, indices);
                    GPUChunkMeshSnapshot oldFront = block.Front;
                    block.Front = block.Back;
                    block.Back = oldFront;
                    block.Renderer.SetMeshSource(block.Front);
                    block.LayerEvidenceRenderer?.SetSource(block.Front);
                }
                else
                {
                    block.Front?.Clear();
                }
            }
            catch (Exception ex)
            {
                Fail($"block {block.Coordinate} snapshot publication failed: {ex.Message}");
                return;
            }

            block.ReadyToPublish = false;
            block.DiscardAfterPresentationFreeze = false;
            block.PendingVertexCount = 0;
            block.PendingIndexCount = 0;
            block.VertexCount = vertices;
            block.IndexCount = indices;
            block.ProcessedEpoch = candidateEpoch;
            block.Built = true;
            ApplyBlockVisibility(block);
            if (block.TargetEpoch > block.ProcessedEpoch)
                QueueBlock(block.Index, block.TargetEpoch, true);
            else if (!InitialBuildComplete && _dirtyQueue.Count == 0 &&
                     _inFlightCommits == 0)
                _nextReadbackTime = 0f;
        }

        private void DiscardFrozenCandidateAndRequeue(Block block)
        {
            if (block == null || !block.ReadyToPublish)
                return;

            uint retryEpoch = math.max(block.TargetEpoch, block.CandidateEpoch);
            block.ReadyToPublish = false;
            block.DiscardAfterPresentationFreeze = false;
            block.PendingVertexCount = 0;
            block.PendingIndexCount = 0;
            // Never invent a dirty epoch merely to force work: doing so could
            // advance ProcessedEpoch past the volume ledger and hide a later
            // real update. A stale/equal candidate is simply discarded while
            // the already published Front remains visible.
            if (retryEpoch > block.ProcessedEpoch)
                QueueBlock(block.Index, retryEpoch, true);
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
            TryFinishInitialBuild();
        }

        private void QueueBlock(int index, uint epoch, bool force = false)
        {
            Block block = _blocks[index];
            block.TargetEpoch = math.max(block.TargetEpoch, epoch);
            if (block.CommitPending || block.ReadyToPublish ||
                (!force && block.ProcessedEpoch >= block.TargetEpoch) ||
                block.Queued)
                return;
            block.Queued = true;
            _dirtyQueue.Enqueue(index);
        }

        private void TryFinishInitialBuild()
        {
            if (InitialBuildComplete || !_firstLedgerApplied)
                return;
            // Continuous fusion may keep the queue non-empty forever. Startup
            // is complete once every block owns a usable first Front (including
            // a proven-empty Front), not only after an impossible globally idle
            // instant.
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
            bool blockReady = InitialBuildComplete &&
                !Failed && block.Built && block.Front != null && block.IndexCount > 0;
            if (block.Renderer != null)
                block.Renderer.RenderVisible = blockReady && _visible;
            if (block.LayerEvidenceRenderer != null)
                block.LayerEvidenceRenderer.SetMode(blockReady
                    ? _layerEvidenceMode
                    : 0);
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
            _dirtyQueue.Clear();
            _inFlightCommits = 0;
            _firstLedgerApplied = knownEmpty;
            InitialBuildComplete = knownEmpty;
            uint epoch = knownEmpty ? _volume.DirtyEpoch : 0u;
            for (int i = 0; i < _blocks.Count; i++)
            {
                Block block = _blocks[i];
                block.CommitSerial++;
                block.CommitPending = false;
                block.ReadyToPublish = false;
                block.DiscardAfterPresentationFreeze = false;
                block.Queued = false;
                block.PendingVertexCount = 0;
                block.PendingIndexCount = 0;
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
            _dirtyQueue.Clear();
            if (_layerEvidencePointMaterial != null)
                UnityEngine.Object.Destroy(_layerEvidencePointMaterial);
            if (_layerEvidenceRelationMaterial != null)
                UnityEngine.Object.Destroy(_layerEvidenceRelationMaterial);
        }
    }
}

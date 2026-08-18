using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Persistent, dirty-chunk mesh front end.  The TSDF remains one authoritative
    /// global volume; only extraction and rendering are partitioned.  A chunk keeps
    /// its last complete GPU mesh until its replacement dispatch is submitted, so
    /// unchanged geometry is neither cleared nor rebuilt.
    /// </summary>
    internal sealed class PersistentChunkMeshPipeline : IDisposable
    {
        internal readonly struct Config
        {
            public readonly int HaloVoxels;
            public readonly int MaxChunksPerTick;
            public readonly float DirtyReadbackHz;
            public readonly float VertexBudgetPercent;
            public readonly int SmoothIterations;
            public readonly float SmoothLambda;
            public readonly float SmoothBeta;
            public readonly float TemporalAlphaMax;
            public readonly float TemporalAlphaMin;
            public readonly float TemporalDecayRate;
            public readonly float ConvergenceThreshold;
            public readonly float TemporalDeadzone;
            public readonly bool DiagnosticRoiEnabled;
            public readonly Vector4 DiagnosticRoiRect;
            public readonly Vector2 DiagnosticRoiSplitX;
            public readonly int ChunkSize;
            public readonly bool StaticReplay;
            public readonly bool StaticReplayAutoQueueAll;
            public readonly bool HeraFilterCleanTriangles;

            public Config(
                int haloVoxels,
                int maxChunksPerTick,
                float dirtyReadbackHz,
                float vertexBudgetPercent,
                int smoothIterations,
                float smoothLambda,
                float smoothBeta,
                float temporalAlphaMax,
                float temporalAlphaMin,
                float temporalDecayRate,
                float convergenceThreshold,
                float temporalDeadzone,
                bool diagnosticRoiEnabled,
                Vector4 diagnosticRoiRect,
                Vector2 diagnosticRoiSplitX,
                int chunkSize,
                bool staticReplay,
                bool staticReplayAutoQueueAll = true,
                bool heraFilterCleanTriangles = false)
            {
                HaloVoxels = Mathf.Max(1, haloVoxels);
                MaxChunksPerTick = Mathf.Max(1, maxChunksPerTick);
                DirtyReadbackHz = Mathf.Max(0.5f, dirtyReadbackHz);
                VertexBudgetPercent = vertexBudgetPercent;
                SmoothIterations = smoothIterations;
                SmoothLambda = smoothLambda;
                SmoothBeta = smoothBeta;
                TemporalAlphaMax = temporalAlphaMax;
                TemporalAlphaMin = temporalAlphaMin;
                TemporalDecayRate = temporalDecayRate;
                ConvergenceThreshold = convergenceThreshold;
                TemporalDeadzone = temporalDeadzone;
                DiagnosticRoiEnabled = diagnosticRoiEnabled;
                DiagnosticRoiRect = diagnosticRoiRect;
                DiagnosticRoiSplitX = diagnosticRoiSplitX;
                ChunkSize = Mathf.Max(4, chunkSize);
                StaticReplay = staticReplay;
                StaticReplayAutoQueueAll = staticReplayAutoQueueAll;
                HeraFilterCleanTriangles = heraFilterCleanTriangles;
            }
        }

        internal readonly struct StaticReplayPageResult
        {
            public readonly int3 Coordinate;
            public readonly int ChunkIndex;
            public readonly int SourceTriangles;
            public readonly int KeptTriangles;
            public readonly int DelegatedTriangles;
            public readonly int InternalMixedContactTriangles;
            public readonly int InternalMixedRealEdgeTriangles;
            public readonly int InternalMixedSuspectedCenterFalseTriangles;
            public readonly int InternalMixedAmbiguousTriangles;
            public readonly int DelegatedInternalMixedRealEdgeTriangles;
            public readonly int DelegatedInternalMixedSuspectedCenterFalseTriangles;
            public readonly int DelegatedInternalMixedAmbiguousTriangles;
            public readonly int DelegatedTransitionTriangles;
            public readonly int AmbiguousNormalTriangles;
            public readonly int AmbiguousWeightTriangles;
            public readonly int AmbiguousBoundaryTriangles;
            public readonly int AmbiguousMultiFailureTriangles;
            public readonly int DelegatedAmbiguousNormalTriangles;
            public readonly int DelegatedAmbiguousWeightTriangles;
            public readonly int DelegatedAmbiguousBoundaryTriangles;
            public readonly int DelegatedAmbiguousMultiFailureTriangles;
            public readonly int InteriorShadowTriangles;
            public readonly int BoundaryShadowTriangles;
            public readonly int DisplayRedTriangles;
            public readonly int ConfirmationUnknownTriangles;
            public readonly int ConfirmationPendingTriangles;
            public readonly int ConfirmationConfirmedTriangles;
            public readonly int ConfirmationMixedTriangles;

            public StaticReplayPageResult(
                int3 coordinate,
                int chunkIndex,
                int sourceTriangles,
                int keptTriangles,
                int delegatedTriangles,
                int internalMixedContactTriangles,
                int internalMixedRealEdgeTriangles,
                int internalMixedSuspectedCenterFalseTriangles,
                int internalMixedAmbiguousTriangles,
                int delegatedInternalMixedRealEdgeTriangles,
                int delegatedInternalMixedSuspectedCenterFalseTriangles,
                int delegatedInternalMixedAmbiguousTriangles,
                int delegatedTransitionTriangles,
                int ambiguousNormalTriangles,
                int ambiguousWeightTriangles,
                int ambiguousBoundaryTriangles,
                int ambiguousMultiFailureTriangles,
                int delegatedAmbiguousNormalTriangles,
                int delegatedAmbiguousWeightTriangles,
                int delegatedAmbiguousBoundaryTriangles,
                int delegatedAmbiguousMultiFailureTriangles,
                int interiorShadowTriangles,
                int boundaryShadowTriangles,
                int displayRedTriangles,
                int confirmationUnknownTriangles,
                int confirmationPendingTriangles,
                int confirmationConfirmedTriangles,
                int confirmationMixedTriangles)
            {
                Coordinate = coordinate;
                ChunkIndex = chunkIndex;
                SourceTriangles = sourceTriangles;
                KeptTriangles = keptTriangles;
                DelegatedTriangles = delegatedTriangles;
                InternalMixedContactTriangles = internalMixedContactTriangles;
                InternalMixedRealEdgeTriangles = internalMixedRealEdgeTriangles;
                InternalMixedSuspectedCenterFalseTriangles = internalMixedSuspectedCenterFalseTriangles;
                InternalMixedAmbiguousTriangles = internalMixedAmbiguousTriangles;
                DelegatedInternalMixedRealEdgeTriangles = delegatedInternalMixedRealEdgeTriangles;
                DelegatedInternalMixedSuspectedCenterFalseTriangles = delegatedInternalMixedSuspectedCenterFalseTriangles;
                DelegatedInternalMixedAmbiguousTriangles = delegatedInternalMixedAmbiguousTriangles;
                DelegatedTransitionTriangles = delegatedTransitionTriangles;
                AmbiguousNormalTriangles = ambiguousNormalTriangles;
                AmbiguousWeightTriangles = ambiguousWeightTriangles;
                AmbiguousBoundaryTriangles = ambiguousBoundaryTriangles;
                AmbiguousMultiFailureTriangles = ambiguousMultiFailureTriangles;
                DelegatedAmbiguousNormalTriangles = delegatedAmbiguousNormalTriangles;
                DelegatedAmbiguousWeightTriangles = delegatedAmbiguousWeightTriangles;
                DelegatedAmbiguousBoundaryTriangles = delegatedAmbiguousBoundaryTriangles;
                DelegatedAmbiguousMultiFailureTriangles = delegatedAmbiguousMultiFailureTriangles;
                InteriorShadowTriangles = interiorShadowTriangles;
                BoundaryShadowTriangles = boundaryShadowTriangles;
                DisplayRedTriangles = displayRedTriangles;
                ConfirmationUnknownTriangles = confirmationUnknownTriangles;
                ConfirmationPendingTriangles = confirmationPendingTriangles;
                ConfirmationConfirmedTriangles = confirmationConfirmedTriangles;
                ConfirmationMixedTriangles = confirmationMixedTriangles;
            }
        }

        private sealed class Chunk : IDisposable
        {
            public int Index;
            public int3 Coordinate;
            public int3 CoreMin;
            public int3 CoreMax;
            public int3 MapMin;
            public int3 MapCount;
            public uint TargetEpoch;
            public uint BuiltEpoch;
            public uint ProcessedEpoch;
            public uint CandidateEpoch;
            public uint LastOwnerEpoch;
            public readonly uint[] LastBoundaryEpoch = new uint[6];
            public bool Queued;
            /// <summary>入队时刻（Time.time）：计时账"排队→落地"起点。</summary>
            public float QueuedAt;
            public bool Built;
            public bool CommitPending;
            public float CommitPendingSince;
            public bool RequestedVisible = true;
            public int AcceptedVertices;
            public int AcceptedIndices;
            // Additive snapshots may contain candidate vertices that are not
            // referenced by an admitted triangle.  Keep physical payload
            // sizes separate from the logical mesh counts used by the
            // replacement gate.
            public int SnapshotVertexCount;
            public int SnapshotIndexCount;
            public int AdditiveMergePasses;
            // Frozen A/B replay-only accounting.  PageClass: 0 empty, 1 clean,
            // 2 mixed/questionable.  It never feeds extraction or TSDF state.
            public int ReplayPageClass;
            public int ReplayBuiltFrame = -1;
            public int ReplayBuildOrder = -1;
            public int ReplayTriangles;
            public int ReplayCleanTriangles;
            public int ReplayQuestionableTriangles;
            public uint ReplayTransitionTriangles;
            public uint ReplayPendingTriangles;
            public uint ReplayInternalMixedTriangles;
            public uint ReplayInternalMixedContactTriangles;
            public uint ReplayInternalMixedRealEdgeTriangles;
            public uint ReplayInternalMixedSuspectedCenterFalseTriangles;
            public uint ReplayInternalMixedAmbiguousTriangles;
            public uint ReplayDelegatedInternalMixedRealEdgeTriangles;
            public uint ReplayDelegatedInternalMixedSuspectedCenterFalseTriangles;
            public uint ReplayDelegatedInternalMixedAmbiguousTriangles;
            public uint ReplayDelegatedTransitionTriangles;
            public uint ReplayAmbiguousNormalTriangles;
            public uint ReplayAmbiguousWeightTriangles;
            public uint ReplayAmbiguousBoundaryTriangles;
            public uint ReplayAmbiguousMultiFailureTriangles;
            public uint ReplayDelegatedAmbiguousNormalTriangles;
            public uint ReplayDelegatedAmbiguousWeightTriangles;
            public uint ReplayDelegatedAmbiguousBoundaryTriangles;
            public uint ReplayDelegatedAmbiguousMultiFailureTriangles;
            public readonly uint[] ReplayForensicEvidence = new uint[4];
            public readonly uint[] ReplayForensicConfirmationEvidence = new uint[16];
            public readonly uint[] ReplayForensicSourceEvidence = new uint[20];
            public readonly uint[] ReplayForensicRiskEvidence = new uint[20];
            public readonly uint[] ReplayForensicRiskTotals = new uint[5];
            public readonly uint[] ReplayForensicSpatialConfirmationEvidence = new uint[64 * 16];
            public uint ReplayForensicTotal;
            public readonly uint[] AcceptedSpatialMature = new uint[SpatialLedgerBinCount];
            public readonly uint[] AcceptedSpatialOccupancy = new uint[SpatialOccupancyWordCount];
            public readonly byte[] SpatialStablePasses = new byte[SpatialLedgerBinCount];
            public readonly bool[] SpatialProtected = new bool[SpatialLedgerBinCount];
            public int DestructiveCandidateCount;
            public float DestructiveCandidateSince;
            public GameObject GameObject;
            public GPUSurfaceNets Surface;
            public GPUChunkMeshSnapshot Snapshot;
            public GPUMeshRenderer Renderer;
            public GPUChunkMeshSnapshot HeraInteriorShadowSnapshot;
            public GPUMeshRenderer HeraInteriorShadowRenderer;
            public int HeraInteriorShadowIndices;
            public bool HeraInteriorShadowRequestedVisible;
            public GPUChunkMeshSnapshot HeraBoundaryShadowSnapshot;
            public GPUMeshRenderer HeraBoundaryShadowRenderer;
            public int HeraBoundaryShadowIndices;
            public bool HeraBoundaryShadowRequestedVisible;

            public void Dispose()
            {
                if (Renderer != null)
                    Renderer.RenderVisible = false;
                if (HeraInteriorShadowRenderer != null)
                    HeraInteriorShadowRenderer.RenderVisible = false;
                if (HeraBoundaryShadowRenderer != null)
                    HeraBoundaryShadowRenderer.RenderVisible = false;
                Snapshot?.Dispose();
                Snapshot = null;
                HeraInteriorShadowSnapshot?.Dispose();
                HeraInteriorShadowSnapshot = null;
                HeraBoundaryShadowSnapshot?.Dispose();
                HeraBoundaryShadowSnapshot = null;
                Surface?.Dispose();
                Surface = null;
                if (GameObject != null)
                    UnityEngine.Object.Destroy(GameObject);
                GameObject = null;
                Renderer = null;
                HeraInteriorShadowRenderer = null;
                HeraBoundaryShadowRenderer = null;
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
        private readonly Config _config;
        private readonly Action<GPUSurfaceNets> _extract;
        private readonly Queue<int> _dirtyQueue = new Queue<int>();
        private readonly List<Chunk> _chunks = new List<Chunk>();
        private int3 _chunkCount;
        private int _generation;
        private int _readbackFailures;
        private bool _readbackPending;
        private bool _ownerLedgerReady;
        private bool _boundaryLedgerReady;
        private bool _ledgerRequestFailed;
        private uint[] _ownerEpochSnapshot;
        private uint[] _boundaryEpochSnapshot;
        private bool _disposed;
        private bool _visible;
        private bool _diagnosticColoring = true;
        private int _replayBuildSequence;
        private float _nextReadbackTime;

        public event Action<StaticReplayPageResult> StaticReplayPageCommitted;

        /// <summary>
        /// CPU-authoritative vertices currently eligible for direct snapshot
        /// rendering.  This is intentionally independent from the HERA colour
        /// ledger and from the legacy indirect argument buffer.
        /// </summary>
        public long VisibleKnownDrawVertexCount
        {
            get
            {
                long total = 0;
                for (int i = 0; i < _chunks.Count; i++)
                {
                    Chunk chunk = _chunks[i];
                    if (chunk.Renderer == null || !chunk.Renderer.RenderVisible || chunk.Snapshot == null)
                        continue;
                    total += Math.Max(0, chunk.Snapshot.KnownDrawVertexCount);
                }
                return total;
            }
        }

        /// <summary>
        /// Direct snapshot vertices actually submitted by renderers during the
        /// current or previous two frames.  A short window avoids Update versus
        /// LateUpdate ordering turning a healthy draw into a false zero.
        /// </summary>
        public long RecentlySubmittedVertexCount
        {
            get
            {
                long total = 0;
                int oldestAcceptedFrame = Time.frameCount - 2;
                for (int i = 0; i < _chunks.Count; i++)
                {
                    GPUMeshRenderer renderer = _chunks[i].Renderer;
                    if (renderer == null || renderer.LastSubmittedFrame < oldestAcceptedFrame ||
                        renderer.LastSubmittedVertexCount <= 0)
                        continue;
                    total += renderer.LastSubmittedVertexCount;
                }
                return total;
            }
        }

        // A mature visible surface is never replaced by a single sparse read.
        // Repeated evidence still permits real removals after a short delay.
        private const float MatureRetainedRatio = 0.75f;
        private const int DestructiveConfirmations = 3;
        private const float DestructiveConfirmSeconds = 0.75f;
        private const int MinMeaningfulVertexLoss = 128;
        private const int MinMeaningfulIndexLoss = 384;
        private const int SpatialLedgerBase = 138;
        private const int SpatialLedgerBinCount = 64;
        private const int SpatialOccupancyBase = 202;
        private const int SpatialOccupancyWordsPerBin = 2;
        private const int SpatialOccupancyWordCount = SpatialLedgerBinCount * SpatialOccupancyWordsPerBin;
        private const int ForensicEvidenceBase = 330;
        private const int ForensicConfirmationEvidenceBase = 334;
        private const int ForensicSourceEvidenceBase = 350;
        private const int ForensicRiskEvidenceBase = 370;
        private const int ForensicRiskTotalsBase = 390;
        private const int ForensicTotalIndex = 395;
        private const int ForensicSpatialBase = 396;
        private const int ForensicEvidenceCount = 4;
        private const int ForensicConfirmationCount = 4;
        private const int ForensicSourceCount = 5;
        private const int ForensicRiskCount = 5;
        private const int ForensicSpatialStride = ForensicConfirmationCount * ForensicEvidenceCount;
        private const int ForensicSpatialCount = SpatialLedgerBinCount * ForensicSpatialStride;
        // A few early triangles are usually a fragment, not an established
        // surface.  Protection starts only after a dense bin and one of its
        // face-neighbours have remained coherent for several accepted passes.
        private const uint SpatialEstablishedTriangles = 24;
        private const int SpatialProtectionConfirmations = 3;
        private const float SpatialStableLowerRatio = 0.65f;
        private const float SpatialStableUpperRatio = 1.55f;
        private const uint SpatialMeaningfulTriangleLoss = 12;
        private const float SpatialRetainedRatio = 0.55f;
        // Partial additive recovery is deliberately bounded.  A later clean,
        // non-regressive replacement compacts the snapshot back to one exact
        // extraction payload.
        private const int MaxAdditiveMergePasses = 6;
        private const int MaxAdditiveSnapshotVertices = 131072;
        private const int MaxAdditiveSnapshotIndices = 786432;
        // Disabled for production: a monotonic triangle union cannot both keep
        // stale geometry and correct a surface that moves into a neighbouring
        // voxel.  It also has no shared-boundary transaction, so it can create
        // duplicate shells and disconnect adjacent patches.  Keep the code and
        // counters available for post-mortem comparison, but do not publish it.
        private const bool EnablePartialAdditiveMerge = false;

        private sealed class LocalReplacementEvent
        {
            public ulong Sequence;
            public float Realtime;
            public int3 Chunk;
            public uint Epoch;
            public bool Initial;
            public bool Accepted;
            public string Decision;
            public int OldVertices;
            public int CandidateVertices;
            public int OldTriangles;
            public int CandidateTriangles;
            public uint OldCells;
            public uint CandidateCells;
            public uint SameCells;
            public uint LostCells;
            public uint AddedCells;
            public uint SuspectedMovedCells;
            public uint ChangedBins;
            public uint NovelTriangles;
            public uint SkippedOccupiedTriangles;
            public uint SkippedImmatureTriangles;
            public int SnapshotVertices;
            public int SnapshotTriangles;
            public int AdditivePass;
        }

        private readonly List<LocalReplacementEvent> _localReplacementEvents = new List<LocalReplacementEvent>(1024);
        private const int MaxLocalReplacementEvents = 65536;
        private ulong _localReplacementSequence;
        private ulong _localReplacementDroppedEvents;
        private ulong _localInitialPublishes;
        private ulong _localAcceptedCandidates;
        private ulong _localRejectedCandidates;
        private ulong _localAcceptedSameCells;
        private ulong _localAcceptedLostCells;
        private ulong _localAcceptedAddedCells;
        private ulong _localAcceptedMovedCells;
        private ulong _localRejectedSameCells;
        private ulong _localRejectedLostCells;
        private ulong _localRejectedAddedCells;
        private ulong _localRejectedMovedCells;
        private ulong _localPartialAcceptedCandidates;
        private ulong _localPartialNovelTriangles;
        private ulong _localPartialSkippedOccupiedTriangles;
        private ulong _localPartialSkippedImmatureTriangles;
        private ulong _localPartialCapRejectedCandidates;
        private LocalReplacementEvent _lastLocalReplacement;

        public bool InitialBuildComplete { get; private set; }
        public bool Failed { get; private set; }
        public string FailureReason { get; private set; }
        public int PendingChunkCount => _dirtyQueue.Count;
        // ── 计时账（EMA 平滑，α=0.25）──
        private float _emaQueueToCommitMs = -1f;
        private float _emaDispatchToCallbackMs = -1f;
        /// <summary>页从入队到提交落地的墙钟 EMA（ms）。高=队列/派发/回读链某处在堵。</summary>
        public float AvgQueueToCommitMs => _emaQueueToCommitMs;
        /// <summary>派发到首个回读回调的往返 EMA（ms）。高=GPU 回读延迟瓶颈。</summary>
        public float AvgDispatchToCallbackMs => _emaDispatchToCallbackMs;
        /// <summary>提交在途页数（已派发待回读）。HUD 拥塞判读：队高途高=回读延迟瓶颈；队高途低=派发被预算闸限流。</summary>
        public int CommitPendingCount
        {
            get
            {
                int n = 0;
                for (int i = 0; i < _chunks.Count; i++)
                    if (_chunks[i].CommitPending) n++;
                return n;
            }
        }
        /// <summary>提交看门狗复位次数（HUD 回读丢弃活度）。</summary>
        public int CommitWatchdogResets { get; private set; }
        public int ChunkCount => _chunks.Count;
        public int ChunkSize => _config.ChunkSize;
        public int3 ChunkGridCount => _chunkCount;
        public int BuiltChunkCount
        {
            get
            {
                int count = 0;
                for (int i = 0; i < _chunks.Count; i++)
                    if (_chunks[i].Built) count++;
                return count;
            }
        }
        public long AcceptedVertexCount
        {
            get
            {
                long count = 0;
                for (int i = 0; i < _chunks.Count; i++)
                    count += _chunks[i].AcceptedVertices;
                return count;
            }
        }
        public long AcceptedTriangleCount
        {
            get
            {
                long count = 0;
                for (int i = 0; i < _chunks.Count; i++)
                    count += _chunks[i].AcceptedIndices / 3;
                return count;
            }
        }
        public int ReplayCleanPageCount => CountReplayPages(1);
        public int ReplayQuestionablePageCount => CountReplayPages(2);
        public int ReplayEmptyPageCount => CountReplayPages(0);
        public long ReplayCollateralCleanTriangleLoss
        {
            get
            {
                long count = 0;
                for (int i = 0; i < _chunks.Count; i++)
                    if (_chunks[i].Built && _chunks[i].ReplayPageClass == 2)
                        count += _chunks[i].ReplayCleanTriangles;
                return count;
            }
        }

        public PersistentChunkMeshPipeline(
            VolumeIntegrator volume,
            ComputeShader compute,
            Material material,
            Transform parent,
            int layer,
            Config config,
            Action<GPUSurfaceNets> extract)
        {
            _volume = volume ?? throw new ArgumentNullException(nameof(volume));
            _compute = compute ?? throw new ArgumentNullException(nameof(compute));
            _material = material;
            _parent = parent;
            _layer = layer;
            _config = config;
            _extract = extract ?? throw new ArgumentNullException(nameof(extract));

            _volume.SetDirtyBoundaryHalo(config.HaloVoxels);
            BuildLayout();
            _volume.Cleared += OnVolumeCleared;
            _volume.TopologyInvalidated += OnTopologyInvalidated;
            if (_config.StaticReplay && _config.StaticReplayAutoQueueAll)
                QueueAll(_volume.DirtyEpoch);
        }

        public void Tick()
        {
            if (_disposed || Failed)
                return;

            // Surface workers retain the mature-triangle ledger.  Treating an
            // idle worker as disposable cache would make an out-of-view chunk
            // forget its accepted surface and relearn it from an empty state.
            if (!_config.StaticReplay)
                RequestDirtyLedgerIfDue();

            int budget = _config.MaxChunksPerTick;
            while (budget-- > 0 && _dirtyQueue.Count > 0)
            {
                int index = _dirtyQueue.Dequeue();
                Chunk chunk = _chunks[index];
                chunk.Queued = false;
                try
                {
                    EnsureChunkResources(chunk);
                    if (chunk.CommitPending)
                        continue;
                    uint candidateEpoch = chunk.TargetEpoch;
                    _extract(chunk.Surface);
                    chunk.Renderer.UpdateBounds(GetPaddedCoreBounds(chunk));
                    chunk.CandidateEpoch = candidateEpoch;
                    chunk.CommitPending = true;
                    chunk.CommitPendingSince = Time.time;
                    RequestCandidateCommit(chunk, candidateEpoch);
                }
                catch (Exception ex)
                {
                    Fail($"chunk {chunk.Coordinate} extraction failed: {ex.Message}");
                    return;
                }
            }

            // 提交看门狗：每页提交=两段串联 GPU 回读，Quest 高负载下回读会被静默
            // 丢弃（回调永不到达）→ CommitPending 永久卡死 → QueueChunk 被守卫挡在
            // 门外，该页永不再提——实机症状=几个块区后新页不再出现。超时强解重排。
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                if (chunk.CommitPending && Time.time - chunk.CommitPendingSince > 10f)
                {
                    chunk.CommitPending = false;
                    CommitWatchdogResets++;
                    Logger.Warning($"页 {chunk.Coordinate} 提交回读超时（>10s），看门狗复位重排");
                    QueueChunk(chunk.Index, chunk.TargetEpoch);
                }
            }

            if (!InitialBuildComplete && AllChunksBuilt())
            {
                InitialBuildComplete = true;
                ApplyVisibility();
                Logger.Info($"Persistent chunk mesh takeover ready: chunks={_chunks.Count}");
            }
        }

        public void SetVisible(bool visible)
        {
            _visible = visible;
            ApplyVisibility();
        }

        public void SetDiagnosticColoring(bool enabled)
        {
            _diagnosticColoring = enabled;
            for (int i = 0; i < _chunks.Count; i++)
            {
                GPUMeshRenderer renderer = _chunks[i].Renderer;
                if (renderer == null) continue;
                if (_config.StaticReplay && _config.HeraFilterCleanTriangles)
                {
                    renderer.SetHeraReplayDisplay(enabled);
                    if (_chunks[i].HeraInteriorShadowRenderer != null)
                        _chunks[i].HeraInteriorShadowRenderer.SetHeraLocalAcceptedPatchDisplay(enabled);
                    if (_chunks[i].HeraBoundaryShadowRenderer != null)
                        _chunks[i].HeraBoundaryShadowRenderer.SetHeraLocalAcceptedPatchDisplay(enabled);
                }
                else if (_config.StaticReplay)
                    renderer.SetReplayPageState(enabled ? _chunks[i].ReplayPageClass : -1);
                else
                    renderer.SetJointDiagnosticDisplay(enabled);
            }
        }

        private void ApplyVisibility()
        {
            bool show = _visible && (_config.StaticReplay || InitialBuildComplete) && !Failed;
            for (int i = 0; i < _chunks.Count; i++)
            {
                if (_chunks[i].Renderer != null)
                    _chunks[i].Renderer.RenderVisible = show && _chunks[i].RequestedVisible && _chunks[i].Built &&
                        _chunks[i].Snapshot != null && _chunks[i].AcceptedIndices > 0;
                ApplyHeraInteriorShadowVisibility(_chunks[i]);
                ApplyHeraBoundaryShadowVisibility(_chunks[i]);
            }
        }

        private void ApplyHeraInteriorShadowVisibility(Chunk chunk)
        {
            if (chunk.HeraInteriorShadowRenderer == null)
                return;
            chunk.HeraInteriorShadowRenderer.RenderVisible =
                _visible && chunk.HeraInteriorShadowRequestedVisible &&
                (_config.StaticReplay || InitialBuildComplete) && !Failed && chunk.Built &&
                chunk.HeraInteriorShadowSnapshot != null && chunk.HeraInteriorShadowIndices > 0;
        }

        private void ApplyHeraBoundaryShadowVisibility(Chunk chunk)
        {
            if (chunk.HeraBoundaryShadowRenderer == null)
                return;
            chunk.HeraBoundaryShadowRenderer.RenderVisible =
                _visible && chunk.HeraBoundaryShadowRequestedVisible &&
                (_config.StaticReplay || InitialBuildComplete) && !Failed && chunk.Built &&
                chunk.HeraBoundaryShadowSnapshot != null && chunk.HeraBoundaryShadowIndices > 0;
        }

        public bool QueueStaticReplayChunk(int3 coordinate)
        {
            if (_disposed || !_config.StaticReplay ||
                math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return false;
            QueueChunk(Flatten(coordinate), math.max(1u, _volume.DirtyEpoch));
            return true;
        }

        /// <summary>页是否已建（含空页）。增量精修 HUD 与邻页判定用。</summary>
        public bool IsChunkBuilt(int3 coordinate)
        {
            if (_disposed || math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return false;
            return _chunks[Flatten(coordinate)].Built;
        }

        /// <summary>页是否在队或提交在途（增量精修 HUD："精修中"判定）。</summary>
        public bool IsChunkInFlight(int3 coordinate)
        {
            if (_disposed || math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return false;
            Chunk chunk = _chunks[Flatten(coordinate)];
            return chunk.Queued || chunk.CommitPending;
        }

        /// <summary>
        /// 强制重建一页（增量精修：解冻→重冻后的重修）。与 QueueStaticReplayChunk
        /// 的差别：清零已处理 epoch，绕过"epoch 未推进就不重提"的幂等守卫。
        /// 旧页面前台快照保留到新提交原子替换，重建期间不闪空；HERA 过滤提交路径
        /// 无反闪烁门槛，真删除（家具搬走）不会被当成退化候选拦下。
        /// </summary>
        public bool RebuildStaticReplayChunk(int3 coordinate)
        {
            if (_disposed || !_config.StaticReplay ||
                math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return false;
            Chunk chunk = _chunks[Flatten(coordinate)];
            chunk.ProcessedEpoch = 0;
            chunk.TargetEpoch = 0;
            QueueChunk(chunk.Index, math.max(1u, _volume.DirtyEpoch));
            return true;
        }

        public void SetChunkVisible(int3 coordinate, bool visible)
        {
            if (_disposed || math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return;
            Chunk chunk = _chunks[Flatten(coordinate)];
            chunk.RequestedVisible = visible;
            if (chunk.Renderer != null)
                chunk.Renderer.RenderVisible = _visible && visible &&
                    (_config.StaticReplay || InitialBuildComplete) && !Failed && chunk.Built &&
                    chunk.Snapshot != null && chunk.AcceptedIndices > 0;
        }

        public void SetHeraInteriorShadowVisible(int3 coordinate, bool visible)
        {
            if (_disposed || math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return;
            Chunk chunk = _chunks[Flatten(coordinate)];
            chunk.HeraInteriorShadowRequestedVisible = visible;
            ApplyHeraInteriorShadowVisibility(chunk);
        }

        public void SetHeraBoundaryShadowVisible(int3 coordinate, bool visible)
        {
            if (_disposed || math.any(coordinate < 0) || math.any(coordinate >= _chunkCount))
                return;
            Chunk chunk = _chunks[Flatten(coordinate)];
            chunk.HeraBoundaryShadowRequestedVisible = visible;
            ApplyHeraBoundaryShadowVisibility(chunk);
        }

        private void BuildLayout()
        {
            int chunkSize = _config.ChunkSize;
            int3 voxels = _volume.VoxelCount;
            int3 cellCount = math.max(voxels - 1, 0);
            _chunkCount = (cellCount + chunkSize - 1) / chunkSize;

            for (int z = 0; z < _chunkCount.z; z++)
            for (int y = 0; y < _chunkCount.y; y++)
            for (int x = 0; x < _chunkCount.x; x++)
            {
                int3 coord = new int3(x, y, z);
                int3 coreMin = coord * chunkSize;
                int3 coreMax = math.min(coreMin + chunkSize, voxels - 1);
                // Core cells are half-open.  One extra voxel is needed because a
                // cell samples its +XYZ corners; the remaining halo feeds smoothing
                // and neighbouring-cell topology without granting emit ownership.
                int3 mapMin = math.max(coreMin - _config.HaloVoxels, 0);
                int3 mapMax = math.min(coreMax + _config.HaloVoxels + 1, voxels);
                int index = Flatten(coord);
                _chunks.Add(new Chunk
                {
                    Index = index,
                    Coordinate = coord,
                    CoreMin = coreMin,
                    CoreMax = coreMax,
                    MapMin = mapMin,
                    MapCount = mapMax - mapMin
                });
            }
        }

        private void EnsureChunkResources(Chunk chunk)
        {
            if (chunk.Surface != null)
                return;

            chunk.Surface = new GPUSurfaceNets(_compute)
            {
                MinMeshWeight = _volume.MinMeshWeight,
                SmoothIterations = _config.SmoothIterations,
                SmoothLambda = _config.SmoothLambda,
                SmoothBeta = _config.SmoothBeta,
                TemporalAlphaMax = _config.TemporalAlphaMax,
                TemporalAlphaMin = _config.TemporalAlphaMin,
                TemporalDecayRate = _config.TemporalDecayRate,
                ConvergenceThreshold = _config.ConvergenceThreshold,
                TemporalDeadzone = _config.TemporalDeadzone,
                StrictObservedEdges = false,
                CandidateHistoryUpdateEnabled = true,
                DiagnosticRoiEnabled = _config.DiagnosticRoiEnabled,
                DiagnosticRoiRect = _config.DiagnosticRoiRect,
                DiagnosticRoiSplitX = _config.DiagnosticRoiSplitX
            };
            chunk.Surface.EnsureBuffers(
                _volume.VoxelCount,
                chunk.MapMin,
                chunk.MapCount,
                chunk.CoreMin,
                chunk.CoreMax,
                _config.VertexBudgetPercent);

            if (chunk.GameObject == null)
            {
                chunk.GameObject = new GameObject($"QRS Persistent Chunk {chunk.Coordinate.x},{chunk.Coordinate.y},{chunk.Coordinate.z}");
                chunk.GameObject.layer = _layer;
                chunk.GameObject.transform.SetParent(_parent, false);
                chunk.Renderer = chunk.GameObject.AddComponent<GPUMeshRenderer>();
                chunk.Renderer.GpuMeshMaterial = _material;
                chunk.Renderer.Initialize(chunk.Surface, GetPaddedCoreBounds(chunk));
                chunk.Renderer.SetStrictObservedDisplay(false);
                if (_config.StaticReplay && _config.HeraFilterCleanTriangles)
                    chunk.Renderer.SetHeraReplayDisplay(_diagnosticColoring);
                else if (_config.StaticReplay)
                    chunk.Renderer.SetReplayPageState(_diagnosticColoring ? chunk.ReplayPageClass : -1);
                else
                    chunk.Renderer.SetJointDiagnosticDisplay(_diagnosticColoring);
                // HERA owns its presentation classes. Legacy pink isolation is
                // not allowed to remove a triangle after the HERA identity
                // ledger accepted it into the final stream.
                chunk.Renderer.SetPinkIsolation(!(_config.StaticReplay && _config.HeraFilterCleanTriangles));
                chunk.Renderer.RenderVisible = false;

                chunk.HeraInteriorShadowRenderer = chunk.GameObject.AddComponent<GPUMeshRenderer>();
                chunk.HeraInteriorShadowRenderer.GpuMeshMaterial = _material;
                chunk.HeraInteriorShadowRenderer.Initialize(null, GetPaddedCoreBounds(chunk));
                chunk.HeraInteriorShadowRenderer.SetHeraLocalAcceptedPatchDisplay(_diagnosticColoring);
                // This exact local-patch stream is already isolated by its own
                // index buffer and must not be filtered a second time.
                chunk.HeraInteriorShadowRenderer.SetPinkIsolation(false);
                chunk.HeraInteriorShadowRenderer.RenderVisible = false;

                chunk.HeraBoundaryShadowRenderer = chunk.GameObject.AddComponent<GPUMeshRenderer>();
                chunk.HeraBoundaryShadowRenderer.GpuMeshMaterial = _material;
                chunk.HeraBoundaryShadowRenderer.Initialize(null, GetPaddedCoreBounds(chunk));
                chunk.HeraBoundaryShadowRenderer.SetHeraLocalAcceptedPatchDisplay(_diagnosticColoring);
                // Boundary triangles are copied from the canonical source-page
                // index stream; this renderer must not classify them again.
                chunk.HeraBoundaryShadowRenderer.SetPinkIsolation(false);
                chunk.HeraBoundaryShadowRenderer.RenderVisible = false;
            }
        }

        private void RequestCandidateCommit(Chunk chunk, uint candidateEpoch)
        {
            int requestGeneration = _generation;
            AsyncGPUReadback.Request(chunk.Surface.CountersBuffer, request =>
            {
                if (_disposed || requestGeneration != _generation)
                    return;

                // 计时账：派发→首个回读回调的往返。
                float rttMs = (Time.time - chunk.CommitPendingSince) * 1000f;
                _emaDispatchToCallbackMs = _emaDispatchToCallbackMs < 0f
                    ? rttMs : Mathf.Lerp(_emaDispatchToCallbackMs, rttMs, 0.25f);

                if (request.hasError)
                {
                    // Keep the previous front-buffer and retry the same epoch.
                    chunk.CommitPending = false;
                    QueueChunk(chunk.Index, candidateEpoch);
                    return;
                }

                var counters = request.GetData<uint>();
                int vertices = counters.Length > 0 ? (int)counters[0] : 0;
                int indices = counters.Length > 1 ? (int)counters[1] : 0;
                if (_config.StaticReplay)
                    CaptureStaticReplayPage(chunk, counters, indices);
                if (_config.StaticReplay && _config.HeraFilterCleanTriangles)
                {
                    RequestHeraFilteredCommit(chunk, candidateEpoch, vertices, indices, requestGeneration);
                    return;
                }
                var spatialMature = new uint[SpatialLedgerBinCount];
                for (int i = 0; i < SpatialLedgerBinCount; i++)
                {
                    int counterIndex = SpatialLedgerBase + i;
                    spatialMature[i] = counterIndex < counters.Length ? counters[counterIndex] : 0u;
                }
                var spatialOccupancy = new uint[SpatialOccupancyWordCount];
                for (int i = 0; i < SpatialOccupancyWordCount; i++)
                {
                    int counterIndex = SpatialOccupancyBase + i;
                    spatialOccupancy[i] = counterIndex < counters.Length ? counters[counterIndex] : 0u;
                }
                bool spatialDestructive = IsSpatialRegression(chunk, spatialMature);
                bool destructive = IsDestructiveRegression(chunk, vertices, indices);
                // A local mature region may not be replaced by an empty region
                // merely because new geometry elsewhere keeps the total count
                // high.  Unlike the legacy global-count gate, this condition is
                // not aged through: real removal must first arrive as explicit
                // contradictory evidence in the extraction history.
                bool accept = !spatialDestructive &&
                              (!destructive || DestructiveCandidateConfirmed(chunk));
                if (accept)
                {
                    RecordLocalReplacement(
                        chunk, candidateEpoch, vertices, indices, spatialOccupancy,
                        true, "accepted");
                    PublishFullCandidate(
                        chunk, candidateEpoch, vertices, indices,
                        spatialMature, spatialOccupancy);
                    FinishCandidateCommit(chunk, candidateEpoch);
                    return;
                }

                uint addedCells = CountAddedSpatialCells(
                    chunk.AcceptedSpatialOccupancy, spatialOccupancy);
                bool additiveCapacityAvailable =
                    chunk.Snapshot != null &&
                    chunk.AdditiveMergePasses < MaxAdditiveMergePasses &&
                    chunk.SnapshotVertexCount + vertices <= MaxAdditiveSnapshotVertices &&
                    chunk.SnapshotIndexCount + indices <= MaxAdditiveSnapshotIndices;

                // A spatially destructive candidate used to be rejected as one
                // indivisible chunk.  Preserve the accepted mesh byte-for-byte,
                // but salvage mature triangles that occupy previously empty
                // fine cells.  This path cannot erase or move old geometry.
                if (EnablePartialAdditiveMerge &&
                    spatialDestructive && addedCells > 0u && additiveCapacityAvailable)
                {
                    GPUSurfaceNets.AdditiveMergeOperation operation;
                    try
                    {
                        operation = chunk.Surface.BeginAdditiveMerge(
                            chunk.Snapshot,
                            chunk.SnapshotVertexCount,
                            chunk.SnapshotIndexCount,
                            vertices,
                            indices,
                            chunk.AcceptedSpatialOccupancy);
                    }
                    catch (Exception ex)
                    {
                        Logger.Warning($"Partial additive merge could not start for chunk " +
                                    $"{chunk.Coordinate.x}/{chunk.Coordinate.y}/{chunk.Coordinate.z}: {ex.Message}");
                        RecordLocalReplacement(
                            chunk, candidateEpoch, vertices, indices, spatialOccupancy,
                            false, "partial_merge_start_failed");
                        ResetDestructiveCandidate(chunk);
                        FinishCandidateCommit(chunk, candidateEpoch);
                        return;
                    }

                    AsyncGPUReadback.Request(operation.Counters, mergeRequest =>
                    {
                        try
                        {
                            if (_disposed || requestGeneration != _generation)
                                return;

                            if (mergeRequest.hasError)
                            {
                                RecordLocalReplacement(
                                    chunk, candidateEpoch, vertices, indices, spatialOccupancy,
                                    false, "partial_merge_readback_failed");
                                ResetDestructiveCandidate(chunk);
                                FinishCandidateCommit(chunk, candidateEpoch);
                                return;
                            }

                            var mergeCounters = mergeRequest.GetData<uint>();
                            uint novelTriangles = mergeCounters.Length > 2 ? mergeCounters[2] : 0u;
                            uint skippedOccupied = mergeCounters.Length > 3 ? mergeCounters[3] : 0u;
                            uint skippedImmature = mergeCounters.Length > 4 ? mergeCounters[4] : 0u;
                            int mergedIndices = mergeCounters.Length > 1
                                ? (int)mergeCounters[1]
                                : chunk.SnapshotIndexCount;

                            if (novelTriangles == 0u ||
                                mergedIndices <= chunk.SnapshotIndexCount ||
                                mergedIndices > MaxAdditiveSnapshotIndices)
                            {
                                RecordLocalReplacement(
                                    chunk, candidateEpoch, vertices, indices, spatialOccupancy,
                                    false, "spatial_regression_no_novel_mature",
                                    novelTriangles, skippedOccupied, skippedImmature,
                                    operation.OutputVertexCount, mergedIndices,
                                    chunk.AdditiveMergePasses);
                                ResetDestructiveCandidate(chunk);
                                FinishCandidateCommit(chunk, candidateEpoch);
                                return;
                            }

                            // Record before changing the accepted occupancy so
                            // the event remains an old-versus-candidate account.
                            RecordLocalReplacement(
                                chunk, candidateEpoch, vertices, indices, spatialOccupancy,
                                true, "partial_additive",
                                novelTriangles, skippedOccupied, skippedImmature,
                                operation.OutputVertexCount, mergedIndices,
                                chunk.AdditiveMergePasses + 1);

                            GPUChunkMeshSnapshot nextSnapshot = operation.TakeSnapshot();
                            // The merge counter readback is authoritative for
                            // the immutable snapshot just like the HERA filter
                            // readback below.  Publish it for the direct snapshot
                            // draw path before handing the buffer to the renderer.
                            nextSnapshot?.SetDrawIndexCount(mergedIndices);
                            GPUChunkMeshSnapshot previousSnapshot = chunk.Snapshot;
                            chunk.Snapshot = nextSnapshot;
                            chunk.Renderer.SetMeshSource(nextSnapshot);
                            previousSnapshot?.Dispose();

                            chunk.SnapshotVertexCount = operation.OutputVertexCount;
                            chunk.SnapshotIndexCount = mergedIndices;
                            chunk.AcceptedVertices = Math.Max(chunk.AcceptedVertices, vertices);
                            chunk.AcceptedIndices = mergedIndices;
                            MergeAcceptedSpatialEvidence(chunk, spatialMature, spatialOccupancy);
                            chunk.AdditiveMergePasses++;
                            chunk.BuiltEpoch = candidateEpoch;
                            chunk.Built = true;
                            ResetDestructiveCandidate(chunk);
                            FinishCandidateCommit(chunk, candidateEpoch);
                        }
                        finally
                        {
                            operation.Dispose();
                        }
                    });
                    return;
                }

                string rejection = spatialDestructive
                    ? (!EnablePartialAdditiveMerge && addedCells > 0u
                        ? "partial_additive_disabled_unsafe"
                        : (addedCells == 0u ? "spatial_regression_no_added_cells" :
                       (chunk.AdditiveMergePasses >= MaxAdditiveMergePasses
                             ? "partial_merge_pass_cap"
                             : "partial_merge_capacity_cap")))
                    : "whole_chunk_regression";
                if (EnablePartialAdditiveMerge &&
                    spatialDestructive && addedCells > 0u && !additiveCapacityAvailable)
                    _localPartialCapRejectedCandidates++;
                RecordLocalReplacement(
                    chunk, candidateEpoch, vertices, indices, spatialOccupancy,
                    false, rejection);
                if (spatialDestructive)
                    ResetDestructiveCandidate(chunk);
                FinishCandidateCommit(chunk, candidateEpoch);
            });
        }

        private void RequestHeraFilteredCommit(
            Chunk chunk,
            uint candidateEpoch,
            int vertices,
            int sourceIndices,
            int requestGeneration)
        {
            GPUSurfaceNets.HeraFilterOperation operation;
            try
            {
                operation = chunk.Surface.BeginHeraCleanFilter(vertices, sourceIndices);
            }
            catch (Exception ex)
            {
                Logger.Warning($"HERA filter could not start for page {chunk.Coordinate}: {ex.Message}");
                chunk.CommitPending = false;
                QueueChunk(chunk.Index, candidateEpoch);
                return;
            }

            AsyncGPUReadback.Request(operation.Counters, filterRequest =>
            {
                try
                {
                    if (_disposed || requestGeneration != _generation)
                        return;
                    if (filterRequest.hasError)
                    {
                        chunk.CommitPending = false;
                        QueueChunk(chunk.Index, candidateEpoch);
                        return;
                    }

                    var filtered = filterRequest.GetData<uint>();
                    int storedIndices = filtered.Length > 0 ? (int)filtered[0] : 0;
                    int keptTriangles = filtered.Length > 1 ? (int)filtered[1] : storedIndices / 3;
                    int delegatedTriangles = filtered.Length > 2
                        ? (int)filtered[2]
                        : Mathf.Max(0, sourceIndices / 3 - keptTriangles);
                    int internalMixedRealEdge = filtered.Length > 3 ? (int)filtered[3] : 0;
                    int internalMixedSuspectedCenterFalse = filtered.Length > 4 ? (int)filtered[4] : 0;
                    int internalMixedAmbiguous = filtered.Length > 5 ? (int)filtered[5] : 0;
                    int delegatedInternalMixedRealEdge = filtered.Length > 6 ? (int)filtered[6] : 0;
                    int delegatedInternalMixedSuspectedCenterFalse = filtered.Length > 7 ? (int)filtered[7] : 0;
                    int delegatedInternalMixedAmbiguous = filtered.Length > 8 ? (int)filtered[8] : 0;
                    int delegatedTransition = filtered.Length > 9 ? (int)filtered[9] : 0;
                    int internalMixedContact = filtered.Length > 10 ? (int)filtered[10] : 0;
                    int ambiguousNormal = filtered.Length > 11 ? (int)filtered[11] : 0;
                    int ambiguousWeight = filtered.Length > 12 ? (int)filtered[12] : 0;
                    int ambiguousBoundary = filtered.Length > 13 ? (int)filtered[13] : 0;
                    int ambiguousMultiFailure = filtered.Length > 14 ? (int)filtered[14] : 0;
                    int delegatedAmbiguousNormal = filtered.Length > 15 ? (int)filtered[15] : 0;
                    int delegatedAmbiguousWeight = filtered.Length > 16 ? (int)filtered[16] : 0;
                    int delegatedAmbiguousBoundary = filtered.Length > 17 ? (int)filtered[17] : 0;
                    int delegatedAmbiguousMultiFailure = filtered.Length > 18 ? (int)filtered[18] : 0;
                    int interiorShadowIndices = filtered.Length > 19 ? (int)filtered[19] : 0;
                    int interiorShadowTriangles = interiorShadowIndices / 3;
                    int boundaryShadowIndices = filtered.Length > 20 ? (int)filtered[20] : 0;
                    int boundaryShadowTriangles = boundaryShadowIndices / 3;
                    int displayRedTriangles = filtered.Length > 21 ? (int)filtered[21] : delegatedTriangles;
                    int confirmationUnknownTriangles = filtered.Length > 22 ? (int)filtered[22] : 0;
                    int confirmationPendingTriangles = filtered.Length > 23 ? (int)filtered[23] : 0;
                    int confirmationConfirmedTriangles = filtered.Length > 24 ? (int)filtered[24] : 0;
                    int confirmationMixedTriangles = filtered.Length > 25 ? (int)filtered[25] : 0;
                    int sourceTriangles = Mathf.Max(0, sourceIndices / 3);

                    GPUChunkMeshSnapshot nextSnapshot = operation.TakeSnapshot();
                    GPUChunkMeshSnapshot nextInteriorShadowSnapshot = operation.TakeInteriorShadowSnapshot();
                    GPUChunkMeshSnapshot nextBoundaryShadowSnapshot = operation.TakeBoundaryShadowSnapshot();
                    // The readback is the authoritative admission result.  Do
                    // not leave visibility dependent on an earlier GPU args
                    // write: that produced pages which were retained in the
                    // HERA ledger but submitted zero indices to the renderer.
                    nextSnapshot?.SetDrawIndexCount(storedIndices);
                    nextInteriorShadowSnapshot?.SetDrawIndexCount(interiorShadowIndices);
                    nextBoundaryShadowSnapshot?.SetDrawIndexCount(boundaryShadowIndices);
                    GPUChunkMeshSnapshot previousSnapshot = chunk.Snapshot;
                    GPUChunkMeshSnapshot previousInteriorShadowSnapshot = chunk.HeraInteriorShadowSnapshot;
                    GPUChunkMeshSnapshot previousBoundaryShadowSnapshot = chunk.HeraBoundaryShadowSnapshot;
                    chunk.Snapshot = nextSnapshot;
                    chunk.HeraInteriorShadowSnapshot = nextInteriorShadowSnapshot;
                    chunk.HeraBoundaryShadowSnapshot = nextBoundaryShadowSnapshot;
                    chunk.Renderer.SetMeshSource(nextSnapshot);
                    chunk.HeraInteriorShadowRenderer.SetMeshSource(nextInteriorShadowSnapshot);
                    chunk.HeraBoundaryShadowRenderer.SetMeshSource(nextBoundaryShadowSnapshot);
                    previousSnapshot?.Dispose();
                    previousInteriorShadowSnapshot?.Dispose();
                    previousBoundaryShadowSnapshot?.Dispose();

                    chunk.AcceptedVertices = vertices;
                    chunk.AcceptedIndices = storedIndices;
                    chunk.SnapshotVertexCount = vertices;
                    chunk.SnapshotIndexCount = storedIndices;
                    chunk.HeraInteriorShadowIndices = interiorShadowIndices;
                    chunk.HeraBoundaryShadowIndices = boundaryShadowIndices;
                    chunk.AdditiveMergePasses = 0;
                    chunk.BuiltEpoch = candidateEpoch;
                    chunk.Built = true;
                    chunk.ReplayTriangles = sourceTriangles;
                    chunk.ReplayCleanTriangles = keptTriangles;
                    chunk.ReplayQuestionableTriangles = delegatedTriangles;
                    chunk.ReplayInternalMixedContactTriangles = (uint)internalMixedContact;
                    chunk.ReplayInternalMixedRealEdgeTriangles = (uint)internalMixedRealEdge;
                    chunk.ReplayInternalMixedSuspectedCenterFalseTriangles = (uint)internalMixedSuspectedCenterFalse;
                    chunk.ReplayInternalMixedAmbiguousTriangles = (uint)internalMixedAmbiguous;
                    chunk.ReplayDelegatedInternalMixedRealEdgeTriangles = (uint)delegatedInternalMixedRealEdge;
                    chunk.ReplayDelegatedInternalMixedSuspectedCenterFalseTriangles = (uint)delegatedInternalMixedSuspectedCenterFalse;
                    chunk.ReplayDelegatedInternalMixedAmbiguousTriangles = (uint)delegatedInternalMixedAmbiguous;
                    chunk.ReplayDelegatedTransitionTriangles = (uint)delegatedTransition;
                    chunk.ReplayAmbiguousNormalTriangles = (uint)ambiguousNormal;
                    chunk.ReplayAmbiguousWeightTriangles = (uint)ambiguousWeight;
                    chunk.ReplayAmbiguousBoundaryTriangles = (uint)ambiguousBoundary;
                    chunk.ReplayAmbiguousMultiFailureTriangles = (uint)ambiguousMultiFailure;
                    chunk.ReplayDelegatedAmbiguousNormalTriangles = (uint)delegatedAmbiguousNormal;
                    chunk.ReplayDelegatedAmbiguousWeightTriangles = (uint)delegatedAmbiguousWeight;
                    chunk.ReplayDelegatedAmbiguousBoundaryTriangles = (uint)delegatedAmbiguousBoundary;
                    chunk.ReplayDelegatedAmbiguousMultiFailureTriangles = (uint)delegatedAmbiguousMultiFailure;
                    chunk.ReplayPageClass = sourceTriangles == 0 ? 0 : delegatedTriangles > 0 ? 2 : 1;
                    chunk.Renderer.SetHeraReplayDisplay(_diagnosticColoring);
                    ResetDestructiveCandidate(chunk);

                    // Complete the page commit before notifying HERA.  The
                    // family callback is the final authority over parent/child
                    // visibility and must not be overwritten by commit cleanup.
                    FinishCandidateCommit(chunk, candidateEpoch);

                    StaticReplayPageCommitted?.Invoke(new StaticReplayPageResult(
                        chunk.Coordinate,
                        chunk.Index,
                        sourceTriangles,
                        keptTriangles,
                        delegatedTriangles,
                        internalMixedContact,
                        internalMixedRealEdge,
                        internalMixedSuspectedCenterFalse,
                        internalMixedAmbiguous,
                        delegatedInternalMixedRealEdge,
                        delegatedInternalMixedSuspectedCenterFalse,
                        delegatedInternalMixedAmbiguous,
                        delegatedTransition,
                        ambiguousNormal,
                        ambiguousWeight,
                        ambiguousBoundary,
                        ambiguousMultiFailure,
                        delegatedAmbiguousNormal,
                        delegatedAmbiguousWeight,
                        delegatedAmbiguousBoundary,
                        delegatedAmbiguousMultiFailure,
                        interiorShadowTriangles,
                        boundaryShadowTriangles,
                        displayRedTriangles,
                        confirmationUnknownTriangles,
                        confirmationPendingTriangles,
                        confirmationConfirmedTriangles,
                        confirmationMixedTriangles));
                    ApplyHeraInteriorShadowVisibility(chunk);
                    ApplyHeraBoundaryShadowVisibility(chunk);

                    // Frozen replay pages are one-shot jobs.  Once their
                    // immutable front buffers and ledgers are published, the
                    // extraction worker (temporal volume, candidate history,
                    // smoothing buffers and source mesh buffers) has no role in
                    // rendering.  Retaining one worker per 32/16 page caused
                    // GPU memory to grow throughout HERA and made the headset
                    // spend long periods showing only the first sparse rescue
                    // pages.  Keep the snapshots; retire only the worker.
                    if (_config.StaticReplay)
                    {
                        chunk.Surface?.Dispose();
                        chunk.Surface = null;
                    }
                }
                finally
                {
                    operation.Dispose();
                }
            });
        }

        private void PublishFullCandidate(
            Chunk chunk,
            uint candidateEpoch,
            int vertices,
            int indices,
            uint[] spatialMature,
            uint[] spatialOccupancy)
        {
            // Publish a complete replacement, then retire the old front
            // buffer. Rendering never observes a cleared/half-copied mesh.
            var nextSnapshot = new GPUChunkMeshSnapshot();
            chunk.Surface.CopyCurrentMeshTo(nextSnapshot, vertices, indices);
            GPUChunkMeshSnapshot previousSnapshot = chunk.Snapshot;
            chunk.Snapshot = nextSnapshot;
            chunk.Renderer.SetMeshSource(nextSnapshot);
            previousSnapshot?.Dispose();
            chunk.AcceptedVertices = vertices;
            chunk.AcceptedIndices = indices;
            chunk.SnapshotVertexCount = vertices;
            chunk.SnapshotIndexCount = indices;
            chunk.AdditiveMergePasses = 0;
            UpdateSpatialProtection(chunk, spatialMature);
            Array.Copy(spatialMature, chunk.AcceptedSpatialMature, SpatialLedgerBinCount);
            Array.Copy(spatialOccupancy, chunk.AcceptedSpatialOccupancy, SpatialOccupancyWordCount);
            chunk.BuiltEpoch = candidateEpoch;
            chunk.Built = true;
            if (_config.StaticReplay)
                chunk.Renderer.SetReplayPageState(_diagnosticColoring ? chunk.ReplayPageClass : -1);
            ResetDestructiveCandidate(chunk);
        }

        private void CaptureStaticReplayPage(Chunk chunk, Unity.Collections.NativeArray<uint> counters, int indices)
        {
            uint transition = counters.Length > 64 ? counters[64] : 0u;
            uint pending = counters.Length > 65 ? counters[65] : 0u;
            uint internalMixed = counters.Length > 66 ? counters[66] : 0u;
            int triangles = Mathf.Max(0, indices / 3);
            ulong questionableRaw = (ulong)transition + pending + internalMixed;
            int questionable = (int)Math.Min((ulong)triangles, questionableRaw);

            chunk.ReplayBuiltFrame = Time.frameCount;
            chunk.ReplayBuildOrder = _replayBuildSequence++;
            chunk.ReplayTriangles = triangles;
            chunk.ReplayTransitionTriangles = transition;
            chunk.ReplayPendingTriangles = pending;
            chunk.ReplayInternalMixedTriangles = internalMixed;
            chunk.ReplayQuestionableTriangles = questionable;
            chunk.ReplayCleanTriangles = Mathf.Max(0, triangles - questionable);
            chunk.ReplayPageClass = triangles == 0 ? 0 : questionable > 0 ? 2 : 1;

            CopyCounterRange(counters, ForensicEvidenceBase, chunk.ReplayForensicEvidence);
            CopyCounterRange(counters, ForensicConfirmationEvidenceBase,
                chunk.ReplayForensicConfirmationEvidence);
            CopyCounterRange(counters, ForensicSourceEvidenceBase,
                chunk.ReplayForensicSourceEvidence);
            CopyCounterRange(counters, ForensicRiskEvidenceBase,
                chunk.ReplayForensicRiskEvidence);
            CopyCounterRange(counters, ForensicRiskTotalsBase,
                chunk.ReplayForensicRiskTotals);
            CopyCounterRange(counters, ForensicSpatialBase,
                chunk.ReplayForensicSpatialConfirmationEvidence);
            chunk.ReplayForensicTotal = counters.Length > ForensicTotalIndex
                ? counters[ForensicTotalIndex]
                : 0u;
        }

        private static void CopyCounterRange(
            Unity.Collections.NativeArray<uint> counters,
            int sourceBase,
            uint[] destination)
        {
            for (int i = 0; i < destination.Length; i++)
            {
                int source = sourceBase + i;
                destination[i] = source < counters.Length ? counters[source] : 0u;
            }
        }

        private int CountReplayPages(int pageClass)
        {
            int count = 0;
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                if (chunk.Built && chunk.ReplayPageClass == pageClass)
                    count++;
            }
            return count;
        }

        public string GetStaticReplayStatsCompact()
        {
            if (!_config.StaticReplay) return "非冻结回放";
            return $"好{ReplayCleanPageCount} 坏{ReplayQuestionablePageCount} 空{ReplayEmptyPageCount} " +
                   $"牵连{ReplayCollateralCleanTriangleLoss}";
        }

        /// <summary>
        /// Counterfactual read-only account: if a page containing any known
        /// questionable triangle were rejected atomically, how much clean
        /// geometry would be lost with it? This is the coupling signal that the
        /// 64/32/16 experiment is designed to compare.
        /// </summary>
        public void AppendStaticReplayReport(StringBuilder sb)
        {
            if (sb == null || !_config.StaticReplay) return;

            long cleanTriangles = 0;
            long questionableTriangles = 0;
            long acceptedIfAtomic = 0;
            long collateralLoss = 0;
            long internalMixedTriangles = 0;
            long internalMixedContactTriangles = 0;
            long internalMixedRealEdgeTriangles = 0;
            long internalMixedSuspectedCenterFalseTriangles = 0;
            long internalMixedAmbiguousTriangles = 0;
            long delegatedInternalMixedRealEdgeTriangles = 0;
            long delegatedInternalMixedSuspectedCenterFalseTriangles = 0;
            long delegatedInternalMixedAmbiguousTriangles = 0;
            long delegatedTransitionTriangles = 0;
            long ambiguousNormalTriangles = 0;
            long ambiguousWeightTriangles = 0;
            long ambiguousBoundaryTriangles = 0;
            long ambiguousMultiFailureTriangles = 0;
            long delegatedAmbiguousNormalTriangles = 0;
            long delegatedAmbiguousWeightTriangles = 0;
            long delegatedAmbiguousBoundaryTriangles = 0;
            long delegatedAmbiguousMultiFailureTriangles = 0;
            long neighbourGapTotal = 0;
            int neighbourPairs = 0;
            int neighbourGapMax = 0;

            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                if (!chunk.Built) continue;
                cleanTriangles += chunk.ReplayCleanTriangles;
                questionableTriangles += chunk.ReplayQuestionableTriangles;
                internalMixedTriangles += chunk.ReplayInternalMixedTriangles;
                internalMixedContactTriangles += chunk.ReplayInternalMixedContactTriangles;
                internalMixedRealEdgeTriangles += chunk.ReplayInternalMixedRealEdgeTriangles;
                internalMixedSuspectedCenterFalseTriangles += chunk.ReplayInternalMixedSuspectedCenterFalseTriangles;
                internalMixedAmbiguousTriangles += chunk.ReplayInternalMixedAmbiguousTriangles;
                delegatedInternalMixedRealEdgeTriangles += chunk.ReplayDelegatedInternalMixedRealEdgeTriangles;
                delegatedInternalMixedSuspectedCenterFalseTriangles += chunk.ReplayDelegatedInternalMixedSuspectedCenterFalseTriangles;
                delegatedInternalMixedAmbiguousTriangles += chunk.ReplayDelegatedInternalMixedAmbiguousTriangles;
                delegatedTransitionTriangles += chunk.ReplayDelegatedTransitionTriangles;
                ambiguousNormalTriangles += chunk.ReplayAmbiguousNormalTriangles;
                ambiguousWeightTriangles += chunk.ReplayAmbiguousWeightTriangles;
                ambiguousBoundaryTriangles += chunk.ReplayAmbiguousBoundaryTriangles;
                ambiguousMultiFailureTriangles += chunk.ReplayAmbiguousMultiFailureTriangles;
                delegatedAmbiguousNormalTriangles += chunk.ReplayDelegatedAmbiguousNormalTriangles;
                delegatedAmbiguousWeightTriangles += chunk.ReplayDelegatedAmbiguousWeightTriangles;
                delegatedAmbiguousBoundaryTriangles += chunk.ReplayDelegatedAmbiguousBoundaryTriangles;
                delegatedAmbiguousMultiFailureTriangles += chunk.ReplayDelegatedAmbiguousMultiFailureTriangles;
                if (chunk.ReplayPageClass == 1)
                    acceptedIfAtomic += chunk.ReplayCleanTriangles;
                else if (chunk.ReplayPageClass == 2)
                    collateralLoss += chunk.ReplayCleanTriangles;

                for (int n = 1; n < FaceNeighbours.Length; n += 2)
                {
                    int3 neighbourCoord = chunk.Coordinate + FaceNeighbours[n];
                    if (math.any(neighbourCoord >= _chunkCount)) continue;
                    Chunk neighbour = _chunks[Flatten(neighbourCoord)];
                    if (!neighbour.Built || neighbour.ReplayBuiltFrame < 0) continue;
                    int gap = Math.Abs(chunk.ReplayBuiltFrame - neighbour.ReplayBuiltFrame);
                    neighbourGapTotal += gap;
                    neighbourGapMax = Math.Max(neighbourGapMax, gap);
                    neighbourPairs++;
                }
            }

            double neighbourGapMean = neighbourPairs > 0
                ? (double)neighbourGapTotal / neighbourPairs
                : 0.0;
            sb.AppendLine();
            sb.AppendLine("page_ab_summary:");
            sb.AppendLine("classification=empty:no triangles; clean:no counters[64..66]; questionable:any transition/pending/internal-mixed triangle");
            sb.AppendLine("counterfactual=atomically reject every questionable page; this does not alter the displayed mesh or frozen TSDF");
            sb.AppendLine($"pages_clean={ReplayCleanPageCount}");
            sb.AppendLine($"pages_questionable={ReplayQuestionablePageCount}");
            sb.AppendLine($"pages_empty={ReplayEmptyPageCount}");
            sb.AppendLine($"triangles_clean={cleanTriangles}");
            sb.AppendLine($"triangles_questionable={questionableTriangles}");
            sb.AppendLine($"triangles_accepted_if_atomic={acceptedIfAtomic}");
            sb.AppendLine($"clean_triangles_collateral_rejected={collateralLoss}");
            long contactLedgerClassified = internalMixedRealEdgeTriangles +
                                       internalMixedSuspectedCenterFalseTriangles +
                                       internalMixedAmbiguousTriangles;
            long delegatedLedgerTotal = delegatedInternalMixedRealEdgeTriangles +
                                        delegatedInternalMixedSuspectedCenterFalseTriangles +
                                        delegatedInternalMixedAmbiguousTriangles;
            sb.AppendLine("internal_mixed_contact_ledger:");
            sb.AppendLine("scope=all triangles containing at least one internal_mixed vertex,independent_of_keep_or_delegate");
            sb.AppendLine($"contact_triangles={internalMixedContactTriangles}");
            sb.AppendLine($"classified_total={contactLedgerClassified}");
            sb.AppendLine($"real_edge={internalMixedRealEdgeTriangles}");
            sb.AppendLine($"suspected_center_false={internalMixedSuspectedCenterFalseTriangles}");
            sb.AppendLine($"ambiguous={internalMixedAmbiguousTriangles}");
            sb.AppendLine($"ambiguous_normal_mid_or_invalid={ambiguousNormalTriangles}");
            sb.AppendLine($"ambiguous_insufficient_weight={ambiguousWeightTriangles}");
            sb.AppendLine($"ambiguous_page_boundary_near={ambiguousBoundaryTriangles}");
            sb.AppendLine($"ambiguous_multi_failure={ambiguousMultiFailureTriangles}");
            sb.AppendLine($"ambiguous_reason_reconcile_delta={internalMixedAmbiguousTriangles - ambiguousNormalTriangles - ambiguousWeightTriangles - ambiguousBoundaryTriangles - ambiguousMultiFailureTriangles}");
            sb.AppendLine($"reconcile_delta={internalMixedContactTriangles - contactLedgerClassified}");
            sb.AppendLine("internal_mixed_delegated_ledger:");
            sb.AppendLine("scope=triangles finally delegated by HERA and containing at least one internal_mixed vertex");
            sb.AppendLine($"delegated_triangles={questionableTriangles}");
            sb.AppendLine($"classified_total={delegatedLedgerTotal}");
            sb.AppendLine($"real_edge={delegatedInternalMixedRealEdgeTriangles}");
            sb.AppendLine($"suspected_center_false={delegatedInternalMixedSuspectedCenterFalseTriangles}");
            sb.AppendLine($"ambiguous={delegatedInternalMixedAmbiguousTriangles}");
            sb.AppendLine($"ambiguous_normal_mid_or_invalid={delegatedAmbiguousNormalTriangles}");
            sb.AppendLine($"ambiguous_insufficient_weight={delegatedAmbiguousWeightTriangles}");
            sb.AppendLine($"ambiguous_page_boundary_near={delegatedAmbiguousBoundaryTriangles}");
            sb.AppendLine($"ambiguous_multi_failure={delegatedAmbiguousMultiFailureTriangles}");
            sb.AppendLine($"ambiguous_reason_reconcile_delta={delegatedInternalMixedAmbiguousTriangles - delegatedAmbiguousNormalTriangles - delegatedAmbiguousWeightTriangles - delegatedAmbiguousBoundaryTriangles - delegatedAmbiguousMultiFailureTriangles}");
            sb.AppendLine($"delegated_without_internal_mixed={delegatedTransitionTriangles}");
            sb.AppendLine($"reconcile_delta={questionableTriangles - delegatedLedgerTotal - delegatedTransitionTriangles}");
            // Compatibility key: this is the historical extraction counter, not
            // the denominator of the broad contact ledger above.
            sb.AppendLine($"legacy_internal_mixed_counter={internalMixedTriangles}");
            sb.AppendLine($"delegated_transition_triangles={delegatedTransitionTriangles}");
            sb.AppendLine($"neighbour_completion_pairs={neighbourPairs}");
            sb.AppendLine($"neighbour_completion_gap_frames_mean={neighbourGapMean.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"neighbour_completion_gap_frames_max={neighbourGapMax}");
            sb.AppendLine();
            sb.AppendLine("internal_mixed_split_semantics=real_edge:min_normal_dot<=0.82; suspected_center_false:min_normal_dot>=0.94,all_three_tsdf_weight>=max(min_weight,0.08),all_three_at_least_2_voxels_inside_page; ambiguous:remaining_internal_mixed; delegated_transition:delegated_without_internal_mixed_vertex");
            sb.AppendLine("internal_mixed_split_is_read_only=true");
            sb.AppendLine("hera_route_source=chunk_local_packed_vertex_admission_ledger");
            sb.AppendLine("hera_local_recheck_source=triangle_normals_plus_tsdf_weight_plus_page_interior");
            sb.AppendLine("child_inherits_parent_decision_state=false");
            sb.AppendLine("child_reuses_parent_classification_semantics=true");
            AppendForensicReplayReport(sb);
            sb.AppendLine("page_ab_csv:");
            sb.AppendLine("chunk_x,chunk_y,chunk_z,core_min_x,core_min_y,core_min_z,core_max_x,core_max_y,core_max_z,build_frame,build_order,page_class,vertices,triangles,clean_triangles,questionable_triangles,transition_triangles,pending_triangles,legacy_internal_mixed_counter,contact_triangles,contact_real_edge,contact_suspected_center_false,contact_ambiguous,contact_ambiguous_normal,contact_ambiguous_weight,contact_ambiguous_boundary,contact_ambiguous_multi,delegated_real_edge,delegated_suspected_center_false,delegated_ambiguous,delegated_ambiguous_normal,delegated_ambiguous_weight,delegated_ambiguous_boundary,delegated_ambiguous_multi,delegated_without_internal_mixed,atomic_clean_collateral_loss");
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk c = _chunks[i];
                if (!c.Built) continue;
                string pageClass = c.ReplayPageClass == 0 ? "empty" : c.ReplayPageClass == 1 ? "clean" : "questionable";
                sb.Append(c.Coordinate.x).Append(',').Append(c.Coordinate.y).Append(',').Append(c.Coordinate.z).Append(',')
                  .Append(c.CoreMin.x).Append(',').Append(c.CoreMin.y).Append(',').Append(c.CoreMin.z).Append(',')
                  .Append(c.CoreMax.x).Append(',').Append(c.CoreMax.y).Append(',').Append(c.CoreMax.z).Append(',')
                  .Append(c.ReplayBuiltFrame).Append(',').Append(c.ReplayBuildOrder).Append(',').Append(pageClass).Append(',')
                  .Append(c.AcceptedVertices).Append(',').Append(c.ReplayTriangles).Append(',')
                  .Append(c.ReplayCleanTriangles).Append(',').Append(c.ReplayQuestionableTriangles).Append(',')
                  .Append(c.ReplayTransitionTriangles).Append(',').Append(c.ReplayPendingTriangles).Append(',')
                   .Append(c.ReplayInternalMixedTriangles).Append(',')
                   .Append(c.ReplayInternalMixedContactTriangles).Append(',')
                   .Append(c.ReplayInternalMixedRealEdgeTriangles).Append(',')
                  .Append(c.ReplayInternalMixedSuspectedCenterFalseTriangles).Append(',')
                  .Append(c.ReplayInternalMixedAmbiguousTriangles).Append(',')
                  .Append(c.ReplayAmbiguousNormalTriangles).Append(',')
                  .Append(c.ReplayAmbiguousWeightTriangles).Append(',')
                  .Append(c.ReplayAmbiguousBoundaryTriangles).Append(',')
                  .Append(c.ReplayAmbiguousMultiFailureTriangles).Append(',')
                  .Append(c.ReplayDelegatedInternalMixedRealEdgeTriangles).Append(',')
                  .Append(c.ReplayDelegatedInternalMixedSuspectedCenterFalseTriangles).Append(',')
                  .Append(c.ReplayDelegatedInternalMixedAmbiguousTriangles).Append(',')
                  .Append(c.ReplayDelegatedAmbiguousNormalTriangles).Append(',')
                  .Append(c.ReplayDelegatedAmbiguousWeightTriangles).Append(',')
                  .Append(c.ReplayDelegatedAmbiguousBoundaryTriangles).Append(',')
                  .Append(c.ReplayDelegatedAmbiguousMultiFailureTriangles).Append(',')
                  .Append(c.ReplayDelegatedTransitionTriangles).Append(',')
                  .Append(c.ReplayPageClass == 2 ? c.ReplayCleanTriangles : 0).AppendLine();
            }
        }

        private void AppendForensicReplayReport(StringBuilder sb)
        {
            var evidence = new long[ForensicEvidenceCount];
            var confirmationEvidence = new long[ForensicConfirmationCount * ForensicEvidenceCount];
            var sourceEvidence = new long[ForensicSourceCount * ForensicEvidenceCount];
            var riskEvidence = new long[ForensicRiskCount * ForensicEvidenceCount];
            var riskTotals = new long[ForensicRiskCount];
            long forensicTotal = 0;
            long sourceTriangles = 0;

            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                if (!chunk.Built) continue;
                sourceTriangles += chunk.ReplayTriangles;
                forensicTotal += chunk.ReplayForensicTotal;
                for (int n = 0; n < evidence.Length; n++)
                    evidence[n] += chunk.ReplayForensicEvidence[n];
                for (int n = 0; n < confirmationEvidence.Length; n++)
                    confirmationEvidence[n] += chunk.ReplayForensicConfirmationEvidence[n];
                for (int n = 0; n < sourceEvidence.Length; n++)
                    sourceEvidence[n] += chunk.ReplayForensicSourceEvidence[n];
                for (int n = 0; n < riskEvidence.Length; n++)
                    riskEvidence[n] += chunk.ReplayForensicRiskEvidence[n];
                for (int n = 0; n < riskTotals.Length; n++)
                    riskTotals[n] += chunk.ReplayForensicRiskTotals[n];
            }

            long evidenceSum = 0;
            long confirmationSum = 0;
            long sourceSum = 0;
            long spatialSum = 0;
            for (int i = 0; i < evidence.Length; i++) evidenceSum += evidence[i];
            for (int i = 0; i < confirmationEvidence.Length; i++) confirmationSum += confirmationEvidence[i];
            for (int i = 0; i < sourceEvidence.Length; i++) sourceSum += sourceEvidence[i];
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                if (!chunk.Built) continue;
                for (int n = 0; n < chunk.ReplayForensicSpatialConfirmationEvidence.Length; n++)
                    spatialSum += chunk.ReplayForensicSpatialConfirmationEvidence[n];
            }

            string[] evidenceNames = { "depth_support", "history_only", "free_space_contradiction", "insufficient_or_mixed" };
            string[] confirmationNames = { "unknown", "pending", "confirmed", "mixed" };
            string[] sourceNames = { "untracked", "self", "direct", "relay", "mixed" };
            string[] riskNames = { "plane_rescue_seen", "near_abstain_seen", "motion_gt_60_birth", "retired_edge_bit", "fast_promotion" };

            sb.AppendLine();
            sb.AppendLine("forensic_geometry_ledger:");
            sb.AppendLine($"chunk_size={_config.ChunkSize}");
            sb.AppendLine("scope=all_emitted_triangles_before_hera_routing;diagnostic_only=true");
            sb.AppendLine("evidence_semantics=depth_support:all_vertices_supported;free_space:contradicted_without_supported_vertex;insufficient:support_contradiction_mix_or_edge_multihop;history:remaining");
            sb.AppendLine($"emitted_total={forensicTotal}");
            for (int e = 0; e < ForensicEvidenceCount; e++)
                sb.AppendLine($"evidence_{evidenceNames[e]}={evidence[e]}");
            sb.AppendLine($"evidence_reconcile_delta={forensicTotal - evidenceSum}");
            sb.AppendLine($"source_triangle_reconcile_delta={sourceTriangles - forensicTotal}");

            sb.AppendLine("confirmation_x_evidence_csv:");
            sb.AppendLine("confirmation,depth_support,history_only,free_space_contradiction,insufficient_or_mixed,total");
            for (int q = 0; q < ForensicConfirmationCount; q++)
            {
                long row = 0;
                sb.Append(confirmationNames[q]);
                for (int e = 0; e < ForensicEvidenceCount; e++)
                {
                    long value = confirmationEvidence[q * ForensicEvidenceCount + e];
                    row += value;
                    sb.Append(',').Append(value);
                }
                sb.Append(',').Append(row).AppendLine();
            }
            sb.AppendLine($"confirmation_reconcile_delta={forensicTotal - confirmationSum}");

            sb.AppendLine("source_x_evidence_csv:");
            sb.AppendLine("source,depth_support,history_only,free_space_contradiction,insufficient_or_mixed,total");
            for (int s = 0; s < ForensicSourceCount; s++)
            {
                long row = 0;
                sb.Append(sourceNames[s]);
                for (int e = 0; e < ForensicEvidenceCount; e++)
                {
                    long value = sourceEvidence[s * ForensicEvidenceCount + e];
                    row += value;
                    sb.Append(',').Append(value);
                }
                sb.Append(',').Append(row).AppendLine();
            }
            sb.AppendLine($"source_reconcile_delta={forensicTotal - sourceSum}");

            sb.AppendLine("lifecycle_risk_x_evidence_csv:");
            sb.AppendLine("risk,depth_support,history_only,free_space_contradiction,insufficient_or_mixed,total");
            for (int r = 0; r < ForensicRiskCount; r++)
            {
                sb.Append(riskNames[r]);
                long row = 0;
                for (int e = 0; e < ForensicEvidenceCount; e++)
                {
                    long value = riskEvidence[r * ForensicEvidenceCount + e];
                    row += value;
                    sb.Append(',').Append(value);
                }
                sb.Append(',').Append(row)
                  .Append(",risk_total_check=").Append(riskTotals[r])
                  .Append(",delta=").Append(riskTotals[r] - row).AppendLine();
            }
            sb.AppendLine("risk_rows_overlap=true");
            sb.AppendLine($"spatial_reconcile_delta={forensicTotal - spatialSum}");

            sb.AppendLine("forensic_spatial_csv:");
            sb.AppendLine("chunk_x,chunk_y,chunk_z,bin_x,bin_y,bin_z,voxel_min_x,voxel_min_y,voxel_min_z,voxel_max_x,voxel_max_y,voxel_max_z,local_min_x_m,local_min_y_m,local_min_z_m,local_max_x_m,local_max_y_m,local_max_z_m,total,support,history,free_space,insufficient,green_support,green_history,green_free_space,green_insufficient,red_support,red_history,red_free_space,red_insufficient");
            int3 volumeCount = _volume.VoxelCount;
            float voxelSize = _volume.VoxelSize;
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                if (!chunk.Built) continue;
                int3 extent = math.max(chunk.CoreMax - chunk.CoreMin, new int3(1));
                for (int bin = 0; bin < SpatialLedgerBinCount; bin++)
                {
                    int bx = bin & 3;
                    int by = (bin >> 2) & 3;
                    int bz = (bin >> 4) & 3;
                    long[] totals = new long[ForensicEvidenceCount];
                    long[] green = new long[ForensicEvidenceCount];
                    long[] red = new long[ForensicEvidenceCount];
                    long binTotal = 0;
                    int baseIndex = bin * ForensicSpatialStride;
                    for (int q = 0; q < ForensicConfirmationCount; q++)
                    {
                        for (int e = 0; e < ForensicEvidenceCount; e++)
                        {
                            long value = chunk.ReplayForensicSpatialConfirmationEvidence[
                                baseIndex + q * ForensicEvidenceCount + e];
                            totals[e] += value;
                            if (q == 2) green[e] += value; else red[e] += value;
                            binTotal += value;
                        }
                    }
                    if (binTotal == 0) continue;

                    int3 binCoord = new int3(bx, by, bz);
                    int3 voxelMin = chunk.CoreMin + (extent * binCoord) / 4;
                    int3 voxelMax = chunk.CoreMin + (extent * (binCoord + 1)) / 4;
                    float3 localMin = ((float3)voxelMin - (float3)volumeCount * 0.5f) * voxelSize;
                    float3 localMax = ((float3)voxelMax - (float3)volumeCount * 0.5f) * voxelSize;

                    sb.Append(chunk.Coordinate.x).Append(',').Append(chunk.Coordinate.y).Append(',').Append(chunk.Coordinate.z).Append(',')
                      .Append(bx).Append(',').Append(by).Append(',').Append(bz).Append(',')
                      .Append(voxelMin.x).Append(',').Append(voxelMin.y).Append(',').Append(voxelMin.z).Append(',')
                      .Append(voxelMax.x).Append(',').Append(voxelMax.y).Append(',').Append(voxelMax.z).Append(',')
                      .Append(localMin.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                      .Append(localMin.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                      .Append(localMin.z.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                      .Append(localMax.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                      .Append(localMax.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                      .Append(localMax.z.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                      .Append(binTotal);
                    for (int e = 0; e < ForensicEvidenceCount; e++) sb.Append(',').Append(totals[e]);
                    for (int e = 0; e < ForensicEvidenceCount; e++) sb.Append(',').Append(green[e]);
                    for (int e = 0; e < ForensicEvidenceCount; e++) sb.Append(',').Append(red[e]);
                    sb.AppendLine();
                }
            }
        }

        private static void MergeAcceptedSpatialEvidence(
            Chunk chunk,
            uint[] candidateMature,
            uint[] candidateOccupancy)
        {
            var mergedMature = new uint[SpatialLedgerBinCount];
            for (int i = 0; i < SpatialLedgerBinCount; i++)
                mergedMature[i] = chunk.AcceptedSpatialMature[i] >= candidateMature[i]
                    ? chunk.AcceptedSpatialMature[i]
                    : candidateMature[i];

            UpdateSpatialProtection(chunk, mergedMature);
            Array.Copy(mergedMature, chunk.AcceptedSpatialMature, SpatialLedgerBinCount);
            for (int i = 0; i < SpatialOccupancyWordCount; i++)
                chunk.AcceptedSpatialOccupancy[i] |= candidateOccupancy[i];
        }

        private static uint CountAddedSpatialCells(uint[] accepted, uint[] candidate)
        {
            uint added = 0u;
            for (int i = 0; i < SpatialOccupancyWordCount; i++)
                added += PopCount(candidate[i] & ~accepted[i]);
            return added;
        }

        private static void ResetDestructiveCandidate(Chunk chunk)
        {
            chunk.DestructiveCandidateCount = 0;
            chunk.DestructiveCandidateSince = 0f;
        }

        private void FinishCandidateCommit(Chunk chunk, uint candidateEpoch)
        {
            chunk.CommitPending = false;
            // 计时账：入队→落地全链路墙钟。
            if (chunk.QueuedAt > 0f)
            {
                float q2cMs = (Time.time - chunk.QueuedAt) * 1000f;
                _emaQueueToCommitMs = _emaQueueToCommitMs < 0f
                    ? q2cMs : Mathf.Lerp(_emaQueueToCommitMs, q2cMs, 0.25f);
                chunk.QueuedAt = 0f;
            }
            chunk.ProcessedEpoch = math.max(chunk.ProcessedEpoch, candidateEpoch);

            if (!InitialBuildComplete && AllChunksBuilt())
            {
                InitialBuildComplete = true;
                ApplyVisibility();
                Logger.Info($"Persistent chunk mesh takeover ready: chunks={_chunks.Count}");
            }
            else if (InitialBuildComplete || _config.StaticReplay)
            {
                chunk.Renderer.RenderVisible = _visible && chunk.RequestedVisible && chunk.Built &&
                    chunk.Snapshot != null && chunk.AcceptedIndices > 0;
            }

            if (chunk.TargetEpoch > chunk.ProcessedEpoch)
                QueueChunk(chunk.Index, chunk.TargetEpoch);
        }

        private static bool IsDestructiveRegression(Chunk chunk, int vertices, int indices)
        {
            if (!chunk.Built || chunk.AcceptedVertices <= 0 || chunk.AcceptedIndices <= 0)
                return false;
            bool vertexLoss = chunk.AcceptedVertices - vertices >= MinMeaningfulVertexLoss &&
                              vertices < chunk.AcceptedVertices * MatureRetainedRatio;
            bool indexLoss = chunk.AcceptedIndices - indices >= MinMeaningfulIndexLoss &&
                             indices < chunk.AcceptedIndices * MatureRetainedRatio;
            return vertexLoss || indexLoss;
        }

        private static bool IsSpatialRegression(Chunk chunk, uint[] candidate)
        {
            if (!chunk.Built || candidate == null || candidate.Length < SpatialLedgerBinCount)
                return false;

            for (int i = 0; i < SpatialLedgerBinCount; i++)
            {
                if (!chunk.SpatialProtected[i])
                    continue;

                uint previous = chunk.AcceptedSpatialMature[i];

                uint current = candidate[i];
                uint loss = previous > current ? previous - current : 0u;
                if (loss >= SpatialMeaningfulTriangleLoss &&
                    current < previous * SpatialRetainedRatio)
                    return true;
            }
            return false;
        }

        private static void UpdateSpatialProtection(Chunk chunk, uint[] candidate)
        {
            for (int i = 0; i < SpatialLedgerBinCount; i++)
            {
                if (chunk.SpatialProtected[i])
                    continue;

                uint current = candidate[i];
                uint previous = chunk.AcceptedSpatialMature[i];
                bool dense = current >= SpatialEstablishedTriangles;
                bool connected = dense && HasDenseFaceNeighbour(candidate, i);
                bool stable = connected && previous >= SpatialEstablishedTriangles &&
                              current >= previous * SpatialStableLowerRatio &&
                              current <= previous * SpatialStableUpperRatio;

                if (!stable)
                {
                    chunk.SpatialStablePasses[i] = 0;
                    continue;
                }

                if (chunk.SpatialStablePasses[i] < byte.MaxValue)
                    chunk.SpatialStablePasses[i]++;
                if (chunk.SpatialStablePasses[i] >= SpatialProtectionConfirmations)
                    chunk.SpatialProtected[i] = true;
            }
        }

        private static bool HasDenseFaceNeighbour(uint[] bins, int index)
        {
            int x = index & 3;
            int y = (index >> 2) & 3;
            int z = (index >> 4) & 3;
            if (x > 0 && bins[index - 1] >= SpatialEstablishedTriangles) return true;
            if (x < 3 && bins[index + 1] >= SpatialEstablishedTriangles) return true;
            if (y > 0 && bins[index - 4] >= SpatialEstablishedTriangles) return true;
            if (y < 3 && bins[index + 4] >= SpatialEstablishedTriangles) return true;
            if (z > 0 && bins[index - 16] >= SpatialEstablishedTriangles) return true;
            if (z < 3 && bins[index + 16] >= SpatialEstablishedTriangles) return true;
            return false;
        }

        private void RecordLocalReplacement(
            Chunk chunk,
            uint candidateEpoch,
            int candidateVertices,
            int candidateIndices,
            uint[] candidateOccupancy,
            bool accepted,
            string decision,
            uint novelTriangles = 0u,
            uint skippedOccupiedTriangles = 0u,
            uint skippedImmatureTriangles = 0u,
            int snapshotVertices = -1,
            int snapshotIndices = -1,
            int additivePass = 0)
        {
            bool initial = !chunk.Built;
            uint oldCells = 0;
            uint candidateCells = 0;
            uint sameCells = 0;
            uint lostCells = 0;
            uint addedCells = 0;
            uint suspectedMovedCells = 0;
            uint changedBins = 0;

            for (int bin = 0; bin < SpatialLedgerBinCount; bin++)
            {
                uint binLost = 0;
                uint binAdded = 0;
                bool changed = false;
                int wordBase = bin * SpatialOccupancyWordsPerBin;
                for (int word = 0; word < SpatialOccupancyWordsPerBin; word++)
                {
                    uint previous = chunk.AcceptedSpatialOccupancy[wordBase + word];
                    uint current = candidateOccupancy[wordBase + word];
                    uint same = previous & current;
                    uint lost = previous & ~current;
                    uint added = current & ~previous;
                    oldCells += PopCount(previous);
                    candidateCells += PopCount(current);
                    sameCells += PopCount(same);
                    uint lostCount = PopCount(lost);
                    uint addedCount = PopCount(added);
                    lostCells += lostCount;
                    addedCells += addedCount;
                    binLost += lostCount;
                    binAdded += addedCount;
                    changed |= previous != current;
                }

                if (changed)
                    changedBins++;
                // This is deliberately labelled "suspected": equal losses and
                // additions inside one coarse neighbourhood look like relocation,
                // but no identity is inferred and production never consumes it.
                suspectedMovedCells += Math.Min(binLost, binAdded);
            }

            var entry = new LocalReplacementEvent
            {
                Sequence = ++_localReplacementSequence,
                Realtime = Time.realtimeSinceStartup,
                Chunk = chunk.Coordinate,
                Epoch = candidateEpoch,
                Initial = initial,
                Accepted = accepted,
                Decision = initial && accepted ? "initial_publish" : decision,
                OldVertices = chunk.AcceptedVertices,
                CandidateVertices = candidateVertices,
                OldTriangles = chunk.AcceptedIndices / 3,
                CandidateTriangles = candidateIndices / 3,
                OldCells = oldCells,
                CandidateCells = candidateCells,
                SameCells = sameCells,
                LostCells = lostCells,
                AddedCells = addedCells,
                SuspectedMovedCells = suspectedMovedCells,
                ChangedBins = changedBins,
                NovelTriangles = novelTriangles,
                SkippedOccupiedTriangles = skippedOccupiedTriangles,
                SkippedImmatureTriangles = skippedImmatureTriangles,
                SnapshotVertices = snapshotVertices >= 0
                    ? snapshotVertices
                    : (accepted ? candidateVertices : chunk.SnapshotVertexCount),
                SnapshotTriangles = snapshotIndices >= 0
                    ? snapshotIndices / 3
                    : (accepted ? candidateIndices / 3 : chunk.SnapshotIndexCount / 3),
                AdditivePass = additivePass
            };

            _lastLocalReplacement = entry;
            if (_localReplacementEvents.Count < MaxLocalReplacementEvents)
                _localReplacementEvents.Add(entry);
            else
                _localReplacementDroppedEvents++;

            if (initial)
            {
                if (accepted) _localInitialPublishes++;
                return;
            }

            if (accepted)
            {
                _localAcceptedCandidates++;
                _localAcceptedSameCells += sameCells;
                _localAcceptedLostCells += lostCells;
                _localAcceptedAddedCells += addedCells;
                _localAcceptedMovedCells += suspectedMovedCells;
                if (decision == "partial_additive")
                {
                    _localPartialAcceptedCandidates++;
                    _localPartialNovelTriangles += novelTriangles;
                    _localPartialSkippedOccupiedTriangles += skippedOccupiedTriangles;
                    _localPartialSkippedImmatureTriangles += skippedImmatureTriangles;
                }
            }
            else
            {
                _localRejectedCandidates++;
                _localRejectedSameCells += sameCells;
                _localRejectedLostCells += lostCells;
                _localRejectedAddedCells += addedCells;
                _localRejectedMovedCells += suspectedMovedCells;
            }
        }

        private static uint PopCount(uint value)
        {
            value -= (value >> 1) & 0x55555555u;
            value = (value & 0x33333333u) + ((value >> 2) & 0x33333333u);
            return (((value + (value >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
        }

        public void ResetLocalReplacementLedger()
        {
            _localReplacementEvents.Clear();
            _localReplacementSequence = 0;
            _localReplacementDroppedEvents = 0;
            _localInitialPublishes = 0;
            _localAcceptedCandidates = 0;
            _localRejectedCandidates = 0;
            _localAcceptedSameCells = 0;
            _localAcceptedLostCells = 0;
            _localAcceptedAddedCells = 0;
            _localAcceptedMovedCells = 0;
            _localRejectedSameCells = 0;
            _localRejectedLostCells = 0;
            _localRejectedAddedCells = 0;
            _localRejectedMovedCells = 0;
            _localPartialAcceptedCandidates = 0;
            _localPartialNovelTriangles = 0;
            _localPartialSkippedOccupiedTriangles = 0;
            _localPartialSkippedImmatureTriangles = 0;
            _localPartialCapRejectedCandidates = 0;
            _lastLocalReplacement = null;
        }

        public string GetLocalReplacementStatsCompact()
        {
            if (_lastLocalReplacement == null)
                return "候0 接0/拒0 尚无替换";
            LocalReplacementEvent last = _lastLocalReplacement;
            return $"候{_localAcceptedCandidates + _localRejectedCandidates} " +
                   $"接{_localAcceptedCandidates}/拒{_localRejectedCandidates} " +
                   $"末{(last.Accepted ? "收" : "拒")} 同{last.SameCells} 消{last.LostCells} " +
                   $"新{last.AddedCells} 搬?{last.SuspectedMovedCells}";
        }

        public void AppendLocalReplacementSummary(StringBuilder sb)
        {
            if (sb == null) return;
            sb.AppendLine();
            sb.AppendLine("局部替换账（只读，不参与生产决策）:");
            sb.AppendLine("口径: 每个持久块划为16×16×16局部占用格；同位=旧新均占用，消失=仅旧占用，新增=仅新占用；疑似搬位=min(同一4×4×4邻域内消失,新增)，仅是定位代理，不代表已证明同一表面移动。");
            sb.AppendLine($"初次发布={_localInitialPublishes} 替换候选={_localAcceptedCandidates + _localRejectedCandidates} 接受={_localAcceptedCandidates} 整块拒绝={_localRejectedCandidates} 事件丢弃={_localReplacementDroppedEvents}");
            sb.AppendLine($"接受累计: 同位={_localAcceptedSameCells} 消失={_localAcceptedLostCells} 新增={_localAcceptedAddedCells} 疑似搬位={_localAcceptedMovedCells}");
            sb.AppendLine($"拒绝累计: 同位={_localRejectedSameCells} 消失={_localRejectedLostCells} 新增={_localRejectedAddedCells} 疑似搬位={_localRejectedMovedCells}");
            sb.AppendLine($"partial_additive: accepted={_localPartialAcceptedCandidates} novel_triangles={_localPartialNovelTriangles} " +
                          $"skip_occupied={_localPartialSkippedOccupiedTriangles} skip_immature={_localPartialSkippedImmatureTriangles} " +
                          $"cap_rejected={_localPartialCapRejectedCandidates}");
            if (_lastLocalReplacement != null)
            {
                LocalReplacementEvent last = _lastLocalReplacement;
                sb.AppendLine($"末次: 块={last.Chunk.x}/{last.Chunk.y}/{last.Chunk.z} epoch={last.Epoch} 决策={last.Decision} " +
                              $"旧/候选格={last.OldCells}/{last.CandidateCells} 同位={last.SameCells} 消失={last.LostCells} " +
                              $"新增={last.AddedCells} 疑似搬位={last.SuspectedMovedCells} 变化粗格={last.ChangedBins}");
            }
        }

        public void AppendLocalReplacementCsv(StringBuilder sb, string sessionId)
        {
            if (sb == null) return;
            sb.AppendLine("session_id,sequence,realtime_s,chunk_x,chunk_y,chunk_z,epoch,initial,accepted,decision,old_vertices,candidate_vertices,old_triangles,candidate_triangles,old_cells,candidate_cells,same_cells,lost_cells,added_cells,suspected_moved_cells,changed_coarse_bins,novel_triangles,skipped_occupied_triangles,skipped_immature_triangles,snapshot_vertices,snapshot_triangles,additive_pass");
            for (int i = 0; i < _localReplacementEvents.Count; i++)
            {
                LocalReplacementEvent e = _localReplacementEvents[i];
                sb.Append(sessionId).Append(',').Append(e.Sequence).Append(',')
                  .Append(e.Realtime.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(e.Chunk.x).Append(',').Append(e.Chunk.y).Append(',').Append(e.Chunk.z).Append(',')
                  .Append(e.Epoch).Append(',').Append(e.Initial ? 1 : 0).Append(',')
                  .Append(e.Accepted ? 1 : 0).Append(',').Append(e.Decision).Append(',')
                  .Append(e.OldVertices).Append(',').Append(e.CandidateVertices).Append(',')
                  .Append(e.OldTriangles).Append(',').Append(e.CandidateTriangles).Append(',')
                  .Append(e.OldCells).Append(',').Append(e.CandidateCells).Append(',')
                  .Append(e.SameCells).Append(',').Append(e.LostCells).Append(',')
                  .Append(e.AddedCells).Append(',').Append(e.SuspectedMovedCells).Append(',')
                  .Append(e.ChangedBins).Append(',').Append(e.NovelTriangles).Append(',')
                  .Append(e.SkippedOccupiedTriangles).Append(',').Append(e.SkippedImmatureTriangles).Append(',')
                  .Append(e.SnapshotVertices).Append(',').Append(e.SnapshotTriangles).Append(',')
                  .Append(e.AdditivePass).AppendLine();
            }
        }

        private Bounds GetPaddedCoreBounds(Chunk chunk)
        {
            Bounds bounds = chunk.Surface.GetCoreBounds(_volume.VoxelSize);
            // Surface-Nets smoothing and halo-shared vertices can sit just
            // outside the exact core AABB.  A two-voxel pad prevents a valid
            // retained chunk from being culled at the edge of the eye frustum.
            float padding = _volume.VoxelSize * Mathf.Max(2, _config.HaloVoxels + 1);
            bounds.Expand(padding * 2f);
            return bounds;
        }

        private static bool DestructiveCandidateConfirmed(Chunk chunk)
        {
            float now = Time.realtimeSinceStartup;
            if (chunk.DestructiveCandidateCount == 0)
                chunk.DestructiveCandidateSince = now;
            chunk.DestructiveCandidateCount++;
            return chunk.DestructiveCandidateCount >= DestructiveConfirmations &&
                   now - chunk.DestructiveCandidateSince >= DestructiveConfirmSeconds;
        }

        private void RequestDirtyLedgerIfDue()
        {
            if (_readbackPending || Time.realtimeSinceStartup < _nextReadbackTime)
                return;

            ComputeBuffer ownerLedger = _volume.DirtyChunkEpochs;
            ComputeBuffer boundaryLedger = _volume.DirtyBoundaryEpochs;
            if (ownerLedger == null || ownerLedger.count != _chunks.Count ||
                boundaryLedger == null || boundaryLedger.count != _chunks.Count * 6)
            {
                Fail("dirty owner/boundary ledger is unavailable or has the wrong size");
                return;
            }

            _readbackPending = true;
            _ownerLedgerReady = false;
            _boundaryLedgerReady = false;
            _ledgerRequestFailed = false;
            _ownerEpochSnapshot = null;
            _boundaryEpochSnapshot = null;
            _nextReadbackTime = Time.realtimeSinceStartup + 1f / _config.DirtyReadbackHz;
            int requestGeneration = _generation;
            AsyncGPUReadback.Request(ownerLedger, request =>
            {
                if (_disposed || requestGeneration != _generation)
                    return;
                if (request.hasError)
                {
                    _ledgerRequestFailed = true;
                }
                else
                {
                    var data = request.GetData<uint>();
                    _ownerEpochSnapshot = new uint[data.Length];
                    data.CopyTo(_ownerEpochSnapshot);
                }
                _ownerLedgerReady = true;
                FinishDirtyLedgerReadback(requestGeneration);
            });

            AsyncGPUReadback.Request(boundaryLedger, request =>
            {
                if (_disposed || requestGeneration != _generation)
                    return;
                if (request.hasError)
                {
                    _ledgerRequestFailed = true;
                }
                else
                {
                    var data = request.GetData<uint>();
                    _boundaryEpochSnapshot = new uint[data.Length];
                    data.CopyTo(_boundaryEpochSnapshot);
                }
                _boundaryLedgerReady = true;
                FinishDirtyLedgerReadback(requestGeneration);
            });
        }

        private void FinishDirtyLedgerReadback(int requestGeneration)
        {
            if (_disposed || requestGeneration != _generation ||
                !_ownerLedgerReady || !_boundaryLedgerReady)
                return;

            _readbackPending = false;
            if (_ledgerRequestFailed || _ownerEpochSnapshot == null || _boundaryEpochSnapshot == null)
            {
                if (++_readbackFailures >= 3)
                    Fail("dirty owner/boundary ledger GPU readback failed three times");
                return;
            }

            _readbackFailures = 0;
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                uint ownerEpoch = _ownerEpochSnapshot[i];
                if (ownerEpoch > chunk.LastOwnerEpoch)
                {
                    chunk.LastOwnerEpoch = ownerEpoch;
                    QueueChunk(i, ownerEpoch);
                }

                int faceBase = i * 6;
                for (int face = 0; face < FaceNeighbours.Length; face++)
                {
                    uint boundaryEpoch = _boundaryEpochSnapshot[faceBase + face];
                    if (boundaryEpoch == 0 || boundaryEpoch <= chunk.LastBoundaryEpoch[face])
                        continue;

                    chunk.LastBoundaryEpoch[face] = boundaryEpoch;
                    int3 neighbour = chunk.Coordinate + FaceNeighbours[face];
                    if (math.any(neighbour < 0) || math.any(neighbour >= _chunkCount))
                        continue;
                    QueueChunk(Flatten(neighbour), boundaryEpoch);
                }
            }

            // A never-observed chunk is already a valid empty front buffer.  Do
            // not allocate a full extraction workset merely to prove emptiness.
            // Boundary propagation above has already queued any empty neighbour
            // whose halo can actually change a surface.
            if (_volume.IntegrationCount > 0)
            {
                for (int i = 0; i < _chunks.Count; i++)
                {
                    Chunk chunk = _chunks[i];
                    if (chunk.LastOwnerEpoch != 0 || chunk.TargetEpoch != 0 ||
                        chunk.Queued || chunk.CommitPending || chunk.Built)
                        continue;
                    chunk.Built = true;
                    chunk.BuiltEpoch = 0;
                    chunk.ProcessedEpoch = 0;
                }
            }
        }

        private void QueueChunk(int index, uint epoch)
        {
            Chunk chunk = _chunks[index];
            if (epoch > chunk.TargetEpoch)
                chunk.TargetEpoch = epoch;
            if (chunk.Queued || chunk.CommitPending || chunk.ProcessedEpoch >= chunk.TargetEpoch)
                return;
            chunk.Queued = true;
            chunk.QueuedAt = Time.time;
            _dirtyQueue.Enqueue(index);
        }

        private void QueueAll(uint epoch)
        {
            for (int i = 0; i < _chunks.Count; i++)
                QueueChunk(i, epoch == 0 ? 1u : epoch);
        }

        private bool AllChunksBuilt()
        {
            for (int i = 0; i < _chunks.Count; i++)
                if (!_chunks[i].Built)
                    return false;
            return _chunks.Count > 0;
        }

        private int Flatten(int3 c)
        {
            return c.x + _chunkCount.x * (c.y + _chunkCount.y * c.z);
        }

        private void OnVolumeCleared()
        {
            if (_disposed) return;
            ResetToKnownEmptyVolume();
        }

        private void OnTopologyInvalidated()
        {
            if (_disposed) return;
            ResetForGlobalInvalidation();
        }

        private void ResetToKnownEmptyVolume()
        {
            _generation++;
            _readbackPending = false;
            _ownerLedgerReady = false;
            _boundaryLedgerReady = false;
            _ledgerRequestFailed = false;
            _ownerEpochSnapshot = null;
            _boundaryEpochSnapshot = null;
            _dirtyQueue.Clear();
            uint clearEpoch = _volume.DirtyEpoch;
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                chunk.Built = true;
                chunk.BuiltEpoch = clearEpoch;
                chunk.ProcessedEpoch = clearEpoch;
                chunk.TargetEpoch = clearEpoch;
                chunk.Queued = false;
                chunk.CommitPending = false;
                chunk.AcceptedVertices = 0;
                chunk.AcceptedIndices = 0;
                chunk.SnapshotVertexCount = 0;
                chunk.SnapshotIndexCount = 0;
                chunk.AdditiveMergePasses = 0;
                Array.Clear(chunk.AcceptedSpatialMature, 0, chunk.AcceptedSpatialMature.Length);
                Array.Clear(chunk.AcceptedSpatialOccupancy, 0, chunk.AcceptedSpatialOccupancy.Length);
                Array.Clear(chunk.SpatialStablePasses, 0, chunk.SpatialStablePasses.Length);
                Array.Clear(chunk.SpatialProtected, 0, chunk.SpatialProtected.Length);
                chunk.DestructiveCandidateCount = 0;
                chunk.DestructiveCandidateSince = 0f;
                chunk.LastOwnerEpoch = clearEpoch;
                Array.Clear(chunk.LastBoundaryEpoch, 0, chunk.LastBoundaryEpoch.Length);
                chunk.Snapshot?.Dispose();
                chunk.Snapshot = null;
                chunk.Surface?.Dispose();
                chunk.Surface = null;
                if (chunk.Renderer != null)
                    chunk.Renderer.RenderVisible = false;
            }
            InitialBuildComplete = _chunks.Count > 0;
            ApplyVisibility();
            Logger.Info("Persistent chunk mesh cleared without rebuilding empty chunks.");
        }

        private void ResetForGlobalInvalidation()
        {
            _generation++;
            _readbackPending = false;
            _ownerLedgerReady = false;
            _boundaryLedgerReady = false;
            _ledgerRequestFailed = false;
            _ownerEpochSnapshot = null;
            _boundaryEpochSnapshot = null;
            _dirtyQueue.Clear();
            InitialBuildComplete = false;
            for (int i = 0; i < _chunks.Count; i++)
            {
                Chunk chunk = _chunks[i];
                chunk.Built = false;
                chunk.BuiltEpoch = 0;
                chunk.ProcessedEpoch = 0;
                chunk.TargetEpoch = 0;
                chunk.Queued = false;
                chunk.CommitPending = false;
                chunk.AcceptedVertices = 0;
                chunk.AcceptedIndices = 0;
                chunk.SnapshotVertexCount = 0;
                chunk.SnapshotIndexCount = 0;
                chunk.AdditiveMergePasses = 0;
                Array.Clear(chunk.AcceptedSpatialMature, 0, chunk.AcceptedSpatialMature.Length);
                Array.Clear(chunk.AcceptedSpatialOccupancy, 0, chunk.AcceptedSpatialOccupancy.Length);
                Array.Clear(chunk.SpatialStablePasses, 0, chunk.SpatialStablePasses.Length);
                Array.Clear(chunk.SpatialProtected, 0, chunk.SpatialProtected.Length);
                chunk.DestructiveCandidateCount = 0;
                chunk.DestructiveCandidateSince = 0f;
                chunk.LastOwnerEpoch = 0;
                Array.Clear(chunk.LastBoundaryEpoch, 0, chunk.LastBoundaryEpoch.Length);
                chunk.Snapshot?.Clear();
                chunk.Surface?.ResetTemporalState();
                if (chunk.Renderer != null)
                    chunk.Renderer.RenderVisible = false;
            }
            QueueAll(_volume.DirtyEpoch);
        }

        private void Fail(string reason)
        {
            Failed = true;
            FailureReason = reason;
            ApplyVisibility();
            Logger.Error($"Persistent chunk mesh disabled; falling back to global extraction: {reason}");
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _generation++;
            _volume.Cleared -= OnVolumeCleared;
            _volume.TopologyInvalidated -= OnTopologyInvalidated;
            for (int i = 0; i < _chunks.Count; i++)
                _chunks[i].Dispose();
            _chunks.Clear();
            _dirtyQueue.Clear();
        }
    }
}

using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    internal class GPUSurfaceNets : IGPUMeshBufferSource, IDisposable
    {
        private readonly ComputeShader _compute;

        // Kernel indices
        private readonly int _kClearCounters;
        private readonly int _kClassifyAndEmit;
        private readonly int _kBuildVertexDispatchArgs;
        private readonly int _kInitSmooth;
        private readonly int _kSmoothVertices;
        private readonly int _kApplySmooth;
        private readonly int _kTemporalBlend;
        private readonly int _kGenerateIndices;
        private readonly int _kBuildIndirectArgs;
        private readonly int _kInitTemporal;
        private readonly int _kClearCandidateHistory;
        private readonly int _kCopyMeshSnapshot;
        private readonly int _kCopyAdditiveMergePayload;
        private readonly int _kAppendNovelMatureTriangles;
        private readonly int _kBuildAdditiveMergeArgs;
        private readonly int _kCopyHeraVertexPayload;
        private readonly int _kFilterHeraCleanTriangles;
        private readonly int _kBuildHeraFilteredArgs;
        private readonly int _kClassifyHeraInternalMixed;
        private readonly int _kBuildHeraInteriorShadowArgs;
        private readonly int _kBuildHeraBoundaryShadowArgs;

        // GPU buffers
        private GraphicsBuffer _coordVertMap;
        private GraphicsBuffer _vertices;
        private GraphicsBuffer _indices;
        private GraphicsBuffer _vertexAdmissionClass;
        private GraphicsBuffer _counters;
        private GraphicsBuffer _dispatchArgs;
        private GraphicsBuffer _drawIndirectArgs;
        private GraphicsBuffer _smoothPosA;
        private GraphicsBuffer _smoothPosB;
        private GraphicsBuffer _candidateHistoryKeys;
        private RenderTexture _lastTsdfVolume;
        private bool _heraRescueWarningLogged;
        private GraphicsBuffer _candidateHistoryStates;

        // Temporal state as 3D texture (avoids 128MB structured buffer limit on Quest)
        private RenderTexture _temporalState;

        // Sizing
        private int3 _voxCount;
        private int3 _mapMin;
        private int3 _mapCount;
        private int3 _coreMin;
        private int3 _coreMax;
        private int _totalVoxels;
        private int _mapVoxels;
        private int _maxVertices;
        private int _maxIndices;
        private int _candidateHistoryCapacity;
        private bool _temporalInitialized;
        private bool _candidateHistoryInitialized;
        private uint _candidateExtractionEpoch;

        public float MinMeshWeight { get; set; } = 0.08f;
        public int SmoothIterations { get; set; } = 1;
        public float SmoothLambda { get; set; } = 0.33f;
        public float SmoothBeta { get; set; } = 0.5f;
        public float TemporalAlphaMax { get; set; } = 0.85f;
        public float TemporalAlphaMin { get; set; } = 0.1f;
        public float TemporalDecayRate { get; set; } = 0.15f;
        public float ConvergenceThreshold { get; set; } = 0.005f;
        public float TemporalDeadzone { get; set; } = 0.001f;
        public bool StrictObservedEdges { get; set; }
        public bool CandidateHistoryUpdateEnabled { get; set; } = true;
        public bool DiagnosticRoiEnabled { get; set; } = true;
        public Vector4 DiagnosticRoiRect { get; set; } = new Vector4(0.2f, 0.25f, 0.8f, 0.75f);
        public Vector2 DiagnosticRoiSplitX { get; set; } = new Vector2(0.44f, 0.56f);

        public GraphicsBuffer VertexBuffer => _vertices;
        public GraphicsBuffer IndexBuffer => _indices;
        public GraphicsBuffer VertexAdmissionClassBuffer => _vertexAdmissionClass;
        public GraphicsBuffer DrawIndirectArgs => _drawIndirectArgs;
        public int KnownDrawVertexCount => -1;
        public GraphicsBuffer CountersBuffer => _counters;

        private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ID_ColorVolume = Shader.PropertyToID("_ColorVolume");
        private static readonly int ID_AdmissionTraceVolume = Shader.PropertyToID("_AdmissionTraceVolume");
        private static readonly int ID_VoxCount = Shader.PropertyToID("_VoxCount");
        private static readonly int ID_MapMin = Shader.PropertyToID("_MapMin");
        private static readonly int ID_MapCount = Shader.PropertyToID("_MapCount");
        private static readonly int ID_CoreMin = Shader.PropertyToID("_CoreMin");
        private static readonly int ID_CoreMax = Shader.PropertyToID("_CoreMax");
        private static readonly int ID_VoxSize = Shader.PropertyToID("_VoxSize");
        private static readonly int ID_MinWeight = Shader.PropertyToID("_MinWeight");
        private static readonly int ID_TotalVoxels = Shader.PropertyToID("_TotalVoxels");
        private static readonly int ID_MaxVertices = Shader.PropertyToID("_MaxVertices");
        private static readonly int ID_SmoothLambda = Shader.PropertyToID("_SmoothLambda");
        private static readonly int ID_SmoothBeta = Shader.PropertyToID("_SmoothBeta");
        private static readonly int ID_TemporalAlphaMax = Shader.PropertyToID("_TemporalAlphaMax");
        private static readonly int ID_TemporalAlphaMin = Shader.PropertyToID("_TemporalAlphaMin");
        private static readonly int ID_TemporalDecayRate = Shader.PropertyToID("_TemporalDecayRate");
        private static readonly int ID_ConvergeThreshold = Shader.PropertyToID("_ConvergeThreshold");
        private static readonly int ID_TemporalDeadzone = Shader.PropertyToID("_TemporalDeadzone");
        private static readonly int ID_StrictObservedEdges = Shader.PropertyToID("_StrictObservedEdges");
        private static readonly int ID_CurrentDepthEvidenceAvailable = Shader.PropertyToID("_CurrentDepthEvidenceAvailable");
        private static readonly int ID_CurrentEdgeEvidenceAvailable = Shader.PropertyToID("_CurrentEdgeEvidenceAvailable");
        private static readonly int ID_DiagnosticRoiEnabled = Shader.PropertyToID("_DiagnosticRoiEnabled");
        private static readonly int ID_DiagnosticRoiRect = Shader.PropertyToID("_DiagnosticRoiRect");
        private static readonly int ID_DiagnosticRoiSplitX = Shader.PropertyToID("_DiagnosticRoiSplitX");

        private static readonly int ID_CoordVertMap = Shader.PropertyToID("_CoordVertMap");
        private static readonly int ID_Vertices = Shader.PropertyToID("_Vertices");
        private static readonly int ID_Indices = Shader.PropertyToID("_Indices");
        private static readonly int ID_VertexAdmissionClass = Shader.PropertyToID("_VertexAdmissionClass");
        private static readonly int ID_Counters = Shader.PropertyToID("_Counters");
        private static readonly int ID_DispatchArgs = Shader.PropertyToID("_DispatchArgs");
        private static readonly int ID_DrawIndirectArgs = Shader.PropertyToID("_DrawIndirectArgs");
        private static readonly int ID_SmoothPosA = Shader.PropertyToID("_SmoothPosA");
        private static readonly int ID_SmoothPosB = Shader.PropertyToID("_SmoothPosB");
        private static readonly int ID_TemporalState = Shader.PropertyToID("_TemporalState");
        private static readonly int ID_CandidateHistoryKeys = Shader.PropertyToID("_CandidateHistoryKeys");
        private static readonly int ID_CandidateHistoryStates = Shader.PropertyToID("_CandidateHistoryStates");
        private static readonly int ID_CandidateExtractionEpoch = Shader.PropertyToID("_CandidateExtractionEpoch");
        private static readonly int ID_CandidateHistoryUpdateEnabled = Shader.PropertyToID("_CandidateHistoryUpdateEnabled");
        private static readonly int ID_CandidateHistoryCapacity = Shader.PropertyToID("_CandidateHistoryCapacity");
        private static readonly int ID_CandidateHistoryMask = Shader.PropertyToID("_CandidateHistoryMask");
        private static readonly int ID_SnapshotVertices = Shader.PropertyToID("_SnapshotVertices");
        private static readonly int ID_SnapshotIndices = Shader.PropertyToID("_SnapshotIndices");
        private static readonly int ID_SnapshotAdmissionClass = Shader.PropertyToID("_SnapshotAdmissionClass");
        private static readonly int ID_SnapshotVertexCount = Shader.PropertyToID("_SnapshotVertexCount");
        private static readonly int ID_SnapshotIndexCount = Shader.PropertyToID("_SnapshotIndexCount");
        private static readonly int ID_PreviousVertices = Shader.PropertyToID("_PreviousVertices");
        private static readonly int ID_PreviousIndices = Shader.PropertyToID("_PreviousIndices");
        private static readonly int ID_PreviousAdmissionClass = Shader.PropertyToID("_PreviousAdmissionClass");
        private static readonly int ID_AcceptedSpatialOccupancy = Shader.PropertyToID("_AcceptedSpatialOccupancy");
        private static readonly int ID_MergeVertices = Shader.PropertyToID("_MergeVertices");
        private static readonly int ID_MergeIndices = Shader.PropertyToID("_MergeIndices");
        private static readonly int ID_MergeAdmissionClass = Shader.PropertyToID("_MergeAdmissionClass");
        private static readonly int ID_MergeCounters = Shader.PropertyToID("_MergeCounters");
        private static readonly int ID_MergeDrawIndirectArgs = Shader.PropertyToID("_MergeDrawIndirectArgs");
        private static readonly int ID_PreviousVertexCount = Shader.PropertyToID("_PreviousVertexCount");
        private static readonly int ID_PreviousIndexCount = Shader.PropertyToID("_PreviousIndexCount");
        private static readonly int ID_CandidateVertexCount = Shader.PropertyToID("_CandidateVertexCount");
        private static readonly int ID_CandidateIndexCount = Shader.PropertyToID("_CandidateIndexCount");
        private static readonly int ID_HeraFilterCounters = Shader.PropertyToID("_HeraFilterCounters");
        private static readonly int ID_HeraDrawIndirectArgs = Shader.PropertyToID("_HeraDrawIndirectArgs");
        private static readonly int ID_HeraSourceIndexCount = Shader.PropertyToID("_HeraSourceIndexCount");
        private static readonly int ID_HeraInteriorShadowIndices = Shader.PropertyToID("_HeraInteriorShadowIndices");
        private static readonly int ID_HeraInteriorShadowDrawIndirectArgs = Shader.PropertyToID("_HeraInteriorShadowDrawIndirectArgs");
        private static readonly int ID_HeraBoundaryShadowIndices = Shader.PropertyToID("_HeraBoundaryShadowIndices");
        private static readonly int ID_HeraBoundaryShadowDrawIndirectArgs = Shader.PropertyToID("_HeraBoundaryShadowDrawIndirectArgs");

        private const int VertexStride = 32;
        private const int Float3Stride = 12;
        // 0..5 production/strict extraction, 6..10 path cells,
        // 11..14 real-confirmation cells, 15..19 admission-source triangles,
        // 20..23 real-confirmation triangles, 24..43 joint cells,
        // 44..63 joint triangles (5 admission sources x 4 confirmation states),
        // 64..66 white-triangle cause split: boundary / pending / internal mixed total.
        // 67..69 are disjoint children of counter 66: admission-only /
        // confirmation-only / mixed on both ledgers.
        // 70..74 split counter 69 without changing extraction:
        // same-vertex / split-vertices, then 1/2/3 same-vertex double-mixed counts.
        // 75..93 are the read-only unique double-mixed-vertex combination ledger.
        // 94..115 exhaustively split the old association fallback (counter 93).
        // 116..119 are the final current-evidence split of the unknown-source
        // residual subset (94..109): support/history/free-space/insufficient.
        // total, source-set partition, confirmation-set partition, and exact
        // pending/confirmed source-association partition.
        // 120..131 are a read-only 3-column ROI split (near/edge/far), with
        // the same four evidence classes as counters 116..119.
        // 132..137 are candidate-production-B temporal triangle states:
        // pending / mature-current / grace-held / supported-edge-only /
        // retired / history-hash-overflow.  They never gate production A.
        // 138..201: 4 x 4 x 4 spatial ledger of mature output triangles.
        // Persistent chunks use this to detect a local coverage collapse that
        // would be hidden by an unchanged whole-chunk triangle count.
        // 202..329: two 32-bit occupancy words for every spatial bin.  Each bit
        // represents one 1/16-chunk mature cell and is read-only replacement
        // evidence; it never participates in extraction or admission.
        // 330..395: global emitted-triangle forensic joins: current depth
        // evidence, confirmation x evidence, source x evidence, lifecycle-risk
        // x evidence, risk totals and the emitted total.
        // 396..1419: 4^3 spatial bins x 4 confirmation states x 4 current-depth
        // evidence states.  This is the page-local join used to locate flying
        // red geometry; it is read-only and adds only a few KB of counters.
        private const int CounterCount = 1420;
        private const int CandidateHistoryCapacity = 1 << 19;

        public GPUSurfaceNets(ComputeShader compute)
        {
            _compute = compute;

            _kClearCounters = compute.FindKernel("ClearCounters");
            _kClassifyAndEmit = compute.FindKernel("ClassifyAndEmit");
            _kBuildVertexDispatchArgs = compute.FindKernel("BuildVertexDispatchArgs");
            _kInitSmooth = compute.FindKernel("InitSmooth");
            _kSmoothVertices = compute.FindKernel("SmoothVertices");
            _kApplySmooth = compute.FindKernel("ApplySmooth");
            _kTemporalBlend = compute.FindKernel("TemporalBlend");
            _kGenerateIndices = compute.FindKernel("GenerateIndices");
            _kBuildIndirectArgs = compute.FindKernel("BuildIndirectArgs");
            _kInitTemporal = compute.FindKernel("InitTemporal");
            _kClearCandidateHistory = compute.FindKernel("ClearCandidateHistory");
            _kCopyMeshSnapshot = compute.FindKernel("CopyMeshSnapshot");
            _kCopyAdditiveMergePayload = compute.FindKernel("CopyAdditiveMergePayload");
            _kAppendNovelMatureTriangles = compute.FindKernel("AppendNovelMatureTriangles");
            _kBuildAdditiveMergeArgs = compute.FindKernel("BuildAdditiveMergeArgs");
            _kCopyHeraVertexPayload = compute.FindKernel("CopyHeraVertexPayload");
            _kFilterHeraCleanTriangles = compute.FindKernel("FilterHeraCleanTriangles");
            _kBuildHeraFilteredArgs = compute.FindKernel("BuildHeraFilteredArgs");
            _kClassifyHeraInternalMixed = compute.FindKernel("ClassifyHeraInternalMixed");
            _kBuildHeraInteriorShadowArgs = compute.FindKernel("BuildHeraInteriorShadowArgs");
            _kBuildHeraBoundaryShadowArgs = compute.FindKernel("BuildHeraBoundaryShadowArgs");
        }

        internal sealed class HeraFilterOperation : IDisposable
        {
            private GPUChunkMeshSnapshot _snapshot;
            private GPUChunkMeshSnapshot _interiorShadowSnapshot;
            private GPUChunkMeshSnapshot _boundaryShadowSnapshot;
            public GraphicsBuffer Counters { get; private set; }
            public int VertexCount { get; }

            public HeraFilterOperation(
                GPUChunkMeshSnapshot snapshot,
                GPUChunkMeshSnapshot interiorShadowSnapshot,
                GPUChunkMeshSnapshot boundaryShadowSnapshot,
                GraphicsBuffer counters,
                int vertexCount)
            {
                _snapshot = snapshot;
                _interiorShadowSnapshot = interiorShadowSnapshot;
                _boundaryShadowSnapshot = boundaryShadowSnapshot;
                Counters = counters;
                VertexCount = vertexCount;
            }

            public GPUChunkMeshSnapshot TakeInteriorShadowSnapshot()
            {
                GPUChunkMeshSnapshot result = _interiorShadowSnapshot;
                _interiorShadowSnapshot = null;
                return result;
            }

            public GPUChunkMeshSnapshot TakeBoundaryShadowSnapshot()
            {
                GPUChunkMeshSnapshot result = _boundaryShadowSnapshot;
                _boundaryShadowSnapshot = null;
                return result;
            }

            public GPUChunkMeshSnapshot TakeSnapshot()
            {
                GPUChunkMeshSnapshot result = _snapshot;
                _snapshot = null;
                return result;
            }

            public void Dispose()
            {
                _snapshot?.Dispose();
                _snapshot = null;
                _interiorShadowSnapshot?.Dispose();
                _interiorShadowSnapshot = null;
                _boundaryShadowSnapshot?.Dispose();
                _boundaryShadowSnapshot = null;
                Counters?.Release();
                Counters = null;
            }
        }

        internal sealed class AdditiveMergeOperation : IDisposable
        {
            private GPUChunkMeshSnapshot _snapshot;
            public GraphicsBuffer Counters { get; private set; }
            public GraphicsBuffer AcceptedOccupancy { get; private set; }
            public int OutputVertexCount { get; }

            public AdditiveMergeOperation(
                GPUChunkMeshSnapshot snapshot,
                GraphicsBuffer counters,
                GraphicsBuffer acceptedOccupancy,
                int outputVertexCount)
            {
                _snapshot = snapshot;
                Counters = counters;
                AcceptedOccupancy = acceptedOccupancy;
                OutputVertexCount = outputVertexCount;
            }

            public GPUChunkMeshSnapshot TakeSnapshot()
            {
                GPUChunkMeshSnapshot result = _snapshot;
                _snapshot = null;
                return result;
            }

            public void Dispose()
            {
                _snapshot?.Dispose();
                _snapshot = null;
                Counters?.Release();
                Counters = null;
                AcceptedOccupancy?.Release();
                AcceptedOccupancy = null;
            }
        }

        public void CopyCurrentMeshTo(GPUChunkMeshSnapshot snapshot, int vertexCount, int indexCount)
        {
            if (snapshot == null) throw new ArgumentNullException(nameof(snapshot));
            vertexCount = Mathf.Clamp(vertexCount, 0, _maxVertices);
            indexCount = Mathf.Clamp(indexCount, 0, _maxIndices);
            snapshot.Prepare(vertexCount, indexCount, VertexStride);

            _compute.SetBuffer(_kCopyMeshSnapshot, ID_Vertices, _vertices);
            _compute.SetBuffer(_kCopyMeshSnapshot, ID_Indices, _indices);
            _compute.SetBuffer(_kCopyMeshSnapshot, ID_VertexAdmissionClass, _vertexAdmissionClass);
            _compute.SetBuffer(_kCopyMeshSnapshot, ID_SnapshotVertices, snapshot.VertexBuffer);
            _compute.SetBuffer(_kCopyMeshSnapshot, ID_SnapshotIndices, snapshot.IndexBuffer);
            _compute.SetBuffer(_kCopyMeshSnapshot, ID_SnapshotAdmissionClass, snapshot.VertexAdmissionClassBuffer);
            _compute.SetInt(ID_SnapshotVertexCount, vertexCount);
            _compute.SetInt(ID_SnapshotIndexCount, indexCount);
            int count = Mathf.Max(vertexCount, indexCount);
            if (count > 0)
                _compute.Dispatch(_kCopyMeshSnapshot, CeilDiv(count, 64), 1, 1);
        }

        public HeraFilterOperation BeginHeraCleanFilter(int vertexCount, int indexCount)
        {
            vertexCount = Mathf.Clamp(vertexCount, 0, _maxVertices);
            indexCount = Mathf.Clamp(indexCount, 0, _maxIndices);
            var snapshot = new GPUChunkMeshSnapshot();
            var interiorShadowSnapshot = new GPUChunkMeshSnapshot();
            var boundaryShadowSnapshot = new GPUChunkMeshSnapshot();
            GraphicsBuffer counters = null;
            try
            {
                int triangleCount = indexCount / 3;
                snapshot.PrepareCapacity(vertexCount, indexCount, VertexStride);
                interiorShadowSnapshot.PrepareCapacity(vertexCount, indexCount, VertexStride);
                boundaryShadowSnapshot.PrepareCapacity(vertexCount, indexCount, VertexStride);
                // 0..10 are the original HERA route/contact ledgers; 11..18
                // exhaustively split ambiguous contact/delegated triangles;
                // 19 is the exact interior-shadow index count; 20 is the exact
                // canonical page-boundary index count; 21 is the displayed red
                // triangle total; 22..25 split all triangles by confirmation
                // class unknown/pending/confirmed/mixed. GenerateIndices owns
                // triangles by its half-open core, so no second page may emit
                // the same source triangle.  No colour identity counters or
                // shared-edge colour buffers exist in the dual-colour path.
                counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 26, sizeof(uint));
                counters.SetData(new uint[26]);

                _compute.SetInt(ID_SnapshotVertexCount, vertexCount);
                _compute.SetInt(ID_HeraSourceIndexCount, indexCount);
                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_Vertices, _vertices);
                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_VertexAdmissionClass, _vertexAdmissionClass);
                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_SnapshotVertices, snapshot.VertexBuffer);
                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_SnapshotAdmissionClass, snapshot.VertexAdmissionClassBuffer);
                if (vertexCount > 0)
                    _compute.Dispatch(_kCopyHeraVertexPayload, CeilDiv(vertexCount, 64), 1, 1);

                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_SnapshotVertices, interiorShadowSnapshot.VertexBuffer);
                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_SnapshotAdmissionClass, interiorShadowSnapshot.VertexAdmissionClassBuffer);
                if (vertexCount > 0)
                    _compute.Dispatch(_kCopyHeraVertexPayload, CeilDiv(vertexCount, 64), 1, 1);

                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_SnapshotVertices, boundaryShadowSnapshot.VertexBuffer);
                _compute.SetBuffer(_kCopyHeraVertexPayload, ID_SnapshotAdmissionClass, boundaryShadowSnapshot.VertexAdmissionClassBuffer);
                if (vertexCount > 0)
                    _compute.Dispatch(_kCopyHeraVertexPayload, CeilDiv(vertexCount, 64), 1, 1);

                _compute.SetBuffer(_kFilterHeraCleanTriangles, ID_Indices, _indices);
                _compute.SetBuffer(_kFilterHeraCleanTriangles, ID_VertexAdmissionClass, _vertexAdmissionClass);
                _compute.SetBuffer(_kFilterHeraCleanTriangles, ID_SnapshotIndices, snapshot.IndexBuffer);
                _compute.SetBuffer(_kFilterHeraCleanTriangles, ID_HeraFilterCounters, counters);
                if (triangleCount > 0)
                    _compute.Dispatch(_kFilterHeraCleanTriangles, CeilDiv(triangleCount, 64), 1, 1);

                _compute.SetBuffer(_kBuildHeraFilteredArgs, ID_HeraFilterCounters, counters);
                _compute.SetBuffer(_kBuildHeraFilteredArgs, ID_HeraDrawIndirectArgs, snapshot.DrawIndirectArgs);
                _compute.Dispatch(_kBuildHeraFilteredArgs, 1, 1, 1);

                // Exact child16 rescue analysis runs after the resident parent
                // stream is complete.  It never rewrites parent indices.
                if (triangleCount > 0 && _lastTsdfVolume != null && _lastTsdfVolume.IsCreated())
                {
                    try
                    {
                        _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
                        _compute.SetInts(ID_CoreMin, _coreMin.x, _coreMin.y, _coreMin.z);
                        _compute.SetInts(ID_CoreMax, _coreMax.x, _coreMax.y, _coreMax.z);
                        _compute.SetFloat(ID_MinWeight, MinMeshWeight);
                        _compute.SetBuffer(_kClassifyHeraInternalMixed, ID_Indices, _indices);
                        _compute.SetBuffer(_kClassifyHeraInternalMixed, ID_Vertices, _vertices);
                        _compute.SetBuffer(_kClassifyHeraInternalMixed, ID_VertexAdmissionClass, _vertexAdmissionClass);
                        _compute.SetBuffer(_kClassifyHeraInternalMixed, ID_HeraFilterCounters, counters);
                        _compute.SetBuffer(_kClassifyHeraInternalMixed, ID_HeraInteriorShadowIndices, interiorShadowSnapshot.IndexBuffer);
                        _compute.SetBuffer(_kClassifyHeraInternalMixed, ID_HeraBoundaryShadowIndices, boundaryShadowSnapshot.IndexBuffer);
                        _compute.SetTexture(_kClassifyHeraInternalMixed, ID_TsdfVolume, _lastTsdfVolume);
                        _compute.Dispatch(_kClassifyHeraInternalMixed, CeilDiv(triangleCount, 64), 1, 1);
                    }
                    catch (Exception ex)
                    {
                        if (!_heraRescueWarningLogged)
                        {
                            Debug.LogWarning($"HERA exact rescue analysis disabled; resident parent mesh is unaffected. {ex.Message}");
                            _heraRescueWarningLogged = true;
                        }
                    }
                }

                _compute.SetBuffer(_kBuildHeraInteriorShadowArgs, ID_HeraFilterCounters, counters);
                _compute.SetBuffer(_kBuildHeraInteriorShadowArgs, ID_HeraInteriorShadowDrawIndirectArgs, interiorShadowSnapshot.DrawIndirectArgs);
                _compute.Dispatch(_kBuildHeraInteriorShadowArgs, 1, 1, 1);
                _compute.SetBuffer(_kBuildHeraBoundaryShadowArgs, ID_HeraFilterCounters, counters);
                _compute.SetBuffer(_kBuildHeraBoundaryShadowArgs, ID_HeraBoundaryShadowDrawIndirectArgs, boundaryShadowSnapshot.DrawIndirectArgs);
                _compute.Dispatch(_kBuildHeraBoundaryShadowArgs, 1, 1, 1);
                return new HeraFilterOperation(
                    snapshot,
                    interiorShadowSnapshot,
                    boundaryShadowSnapshot,
                    counters,
                    vertexCount);
            }
            catch
            {
                snapshot.Dispose();
                interiorShadowSnapshot.Dispose();
                boundaryShadowSnapshot.Dispose();
                counters?.Release();
                throw;
            }
        }

        public AdditiveMergeOperation BeginAdditiveMerge(
            GPUChunkMeshSnapshot previous,
            int previousVertexCount,
            int previousIndexCount,
            int candidateVertexCount,
            int candidateIndexCount,
            uint[] acceptedSpatialOccupancy)
        {
            if (previous == null) throw new ArgumentNullException(nameof(previous));
            if (acceptedSpatialOccupancy == null || acceptedSpatialOccupancy.Length < 128)
                throw new ArgumentException("Expected 128 accepted occupancy words.", nameof(acceptedSpatialOccupancy));

            previousVertexCount = Mathf.Clamp(previousVertexCount, 0, previous.VertexBuffer.count);
            previousIndexCount = Mathf.Clamp(previousIndexCount, 0, previous.IndexBuffer.count);
            candidateVertexCount = Mathf.Clamp(candidateVertexCount, 0, _maxVertices);
            candidateIndexCount = Mathf.Clamp(candidateIndexCount, 0, _maxIndices);

            int outputVertexCount = previousVertexCount + candidateVertexCount;
            int outputIndexCapacity = previousIndexCount + candidateIndexCount;
            var next = new GPUChunkMeshSnapshot();
            GraphicsBuffer counters = null;
            GraphicsBuffer occupancy = null;
            try
            {
                next.PrepareCapacity(outputVertexCount, outputIndexCapacity, VertexStride);
                counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 5, sizeof(uint));
                counters.SetData(new uint[]
                {
                    (uint)outputVertexCount, (uint)previousIndexCount, 0u, 0u, 0u
                });
                occupancy = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 128, sizeof(uint));
                occupancy.SetData(acceptedSpatialOccupancy, 0, 0, 128);

                _compute.SetInt(ID_PreviousVertexCount, previousVertexCount);
                _compute.SetInt(ID_PreviousIndexCount, previousIndexCount);
                _compute.SetInt(ID_CandidateVertexCount, candidateVertexCount);
                _compute.SetInt(ID_CandidateIndexCount, candidateIndexCount);

                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_PreviousVertices, previous.VertexBuffer);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_PreviousIndices, previous.IndexBuffer);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_PreviousAdmissionClass, previous.VertexAdmissionClassBuffer);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_Vertices, _vertices);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_VertexAdmissionClass, _vertexAdmissionClass);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_MergeVertices, next.VertexBuffer);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_MergeIndices, next.IndexBuffer);
                _compute.SetBuffer(_kCopyAdditiveMergePayload, ID_MergeAdmissionClass, next.VertexAdmissionClassBuffer);
                int copyCount = Mathf.Max(previousIndexCount,
                    Mathf.Max(previousVertexCount, candidateVertexCount));
                if (copyCount > 0)
                    _compute.Dispatch(_kCopyAdditiveMergePayload, CeilDiv(copyCount, 64), 1, 1);

                _compute.SetBuffer(_kAppendNovelMatureTriangles, ID_Vertices, _vertices);
                _compute.SetBuffer(_kAppendNovelMatureTriangles, ID_Indices, _indices);
                _compute.SetBuffer(_kAppendNovelMatureTriangles, ID_AcceptedSpatialOccupancy, occupancy);
                _compute.SetBuffer(_kAppendNovelMatureTriangles, ID_MergeIndices, next.IndexBuffer);
                _compute.SetBuffer(_kAppendNovelMatureTriangles, ID_MergeCounters, counters);
                int triangleCount = candidateIndexCount / 3;
                if (triangleCount > 0)
                    _compute.Dispatch(_kAppendNovelMatureTriangles, CeilDiv(triangleCount, 64), 1, 1);

                _compute.SetBuffer(_kBuildAdditiveMergeArgs, ID_MergeCounters, counters);
                _compute.SetBuffer(_kBuildAdditiveMergeArgs, ID_MergeDrawIndirectArgs, next.DrawIndirectArgs);
                _compute.Dispatch(_kBuildAdditiveMergeArgs, 1, 1, 1);
                return new AdditiveMergeOperation(next, counters, occupancy, outputVertexCount);
            }
            catch
            {
                next.Dispose();
                counters?.Release();
                occupancy?.Release();
                throw;
            }
        }

        public void EnsureBuffers(int3 voxCount, float vertexBudgetPercent = 0.05f)
        {
            EnsureBuffers(voxCount, int3.zero, voxCount, int3.zero, voxCount, vertexBudgetPercent);
        }

        /// <summary>
        /// Allocates a persistent extraction block.  mapMin/mapCount include the
        /// halo used for topology lookup; coreMin/coreMax are the half-open range
        /// allowed to emit indices.  Vertex coordinates remain in the full TSDF
        /// coordinate system, so block meshes share one world frame.
        /// </summary>
        public void EnsureBuffers(
            int3 voxCount,
            int3 mapMin,
            int3 mapCount,
            int3 coreMin,
            int3 coreMax,
            float vertexBudgetPercent = 0.12f)
        {
            int totalVoxels = voxCount.x * voxCount.y * voxCount.z;
            int mapVoxels = mapCount.x * mapCount.y * mapCount.z;
            if (_totalVoxels == totalVoxels && _mapVoxels == mapVoxels &&
                math.all(_mapMin == mapMin) && math.all(_mapCount == mapCount) &&
                math.all(_coreMin == coreMin) && math.all(_coreMax == coreMax) &&
                _coordVertMap != null)
                return;

            Dispose();

            _voxCount = voxCount;
            _mapMin = mapMin;
            _mapCount = mapCount;
            _coreMin = coreMin;
            _coreMax = coreMax;
            _totalVoxels = totalVoxels;
            _mapVoxels = mapVoxels;
            _maxVertices = Mathf.Max(1024, (int)(mapVoxels * vertexBudgetPercent));
            _maxIndices = _maxVertices * 18;
            _candidateHistoryCapacity = Mathf.NextPowerOfTwo(Mathf.Clamp(mapVoxels * 2, 4096, CandidateHistoryCapacity));

            const GraphicsBuffer.Target structuredIndirect =
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;

            _coordVertMap = new GraphicsBuffer(GraphicsBuffer.Target.Structured, mapVoxels, 4);
            _vertices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, VertexStride);
            _indices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxIndices, 4);
            _vertexAdmissionClass = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, 4);
            _counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CounterCount, 4);
            _dispatchArgs = new GraphicsBuffer(structuredIndirect, 3, 4);
            _drawIndirectArgs = new GraphicsBuffer(structuredIndirect, 5, 4);
            _smoothPosA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
            _smoothPosB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
            _candidateHistoryKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _candidateHistoryCapacity, 4);
            _candidateHistoryStates = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _candidateHistoryCapacity, 4);

            // Temporal state as RWTexture3D<float4> -- avoids the 128MB structured buffer limit.
            // 256^3 x RGBA32Float = 256MB as a 3D texture, which Quest supports (same as TSDF volume path).
            _temporalState = new RenderTexture(mapCount.x, mapCount.y, 0, GraphicsFormat.R32G32B32A32_SFloat)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = mapCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            _temporalState.Create();

            _temporalInitialized = false;
            _candidateHistoryInitialized = false;
            _candidateExtractionEpoch = 0;

            long totalBytes = (long)mapVoxels * 4
                            + (long)_maxVertices * VertexStride
                            + (long)_maxIndices * 4
                            + (long)_maxVertices * 4
                            + CounterCount * 4 + 3 * 4 + 5 * 4
                            + (long)_maxVertices * Float3Stride * 2
                            + (long)mapVoxels * 16
                            + (long)_candidateHistoryCapacity * 8;
            Logger.Info($"[GPUSurfaceNets] Allocated buffers: vox={voxCount}, map={mapMin}+{mapCount}, core={coreMin}..{coreMax}, " +
                      $"maxVerts={_maxVertices}, maxIdx={_maxIndices}, " +
                      $"totalGPU={totalBytes / (1024 * 1024)}MB");
        }

        public void InitTemporalState()
        {
            if (_temporalInitialized || _temporalState == null) return;

            _compute.SetTexture(_kInitTemporal, ID_TemporalState, _temporalState);
            SetRegionParams();
            int gx = CeilDiv(_mapCount.x, 4);
            int gy = CeilDiv(_mapCount.y, 4);
            int gz = CeilDiv(_mapCount.z, 4);
            _compute.Dispatch(_kInitTemporal, gx, gy, gz);
            _temporalInitialized = true;
        }

        /// <summary>
        /// Mark temporal history dirty without reallocating the large Surface Nets buffers.
        /// The next Extract reinitializes the temporal texture before consuming a different TSDF source.
        /// </summary>
        public void ResetTemporalState()
        {
            _temporalInitialized = false;
            _candidateHistoryInitialized = false;
            _candidateExtractionEpoch = 0;
        }

        public void Extract(
            RenderTexture tsdfVolume,
            RenderTexture colorVolume,
            RenderTexture admissionTraceVolume,
            float voxelSize,
            Texture currentDepthTexture,
            Texture currentEdgeReasonTexture,
            bool currentDepthEvidenceAvailable)
        {
            if (_coordVertMap == null)
                throw new InvalidOperationException("Call EnsureBuffers before Extract");

            if (!_temporalInitialized && TemporalAlphaMax < 1f)
                InitTemporalState();

            SetGlobalParams(voxelSize);
            BindAllBuffers();
            _lastTsdfVolume = tsdfVolume;

            if (!_candidateHistoryInitialized)
            {
                _compute.Dispatch(_kClearCandidateHistory, CeilDiv(_candidateHistoryCapacity, 256), 1, 1);
                _candidateHistoryInitialized = true;
            }

            if (CandidateHistoryUpdateEnabled)
            {
                _candidateExtractionEpoch++;
                if (_candidateExtractionEpoch == 0 || _candidateExtractionEpoch > 0x0FFFFFFFu)
                    _candidateExtractionEpoch = 1;
            }
            _compute.SetInt(ID_CandidateExtractionEpoch, unchecked((int)_candidateExtractionEpoch));
            _compute.SetInt(ID_CandidateHistoryUpdateEnabled, CandidateHistoryUpdateEnabled ? 1 : 0);

            _compute.SetTexture(_kClassifyAndEmit, ID_TsdfVolume, tsdfVolume);
            _compute.SetTexture(_kClassifyAndEmit, ID_ColorVolume, colorVolume);
            _compute.SetTexture(_kClassifyAndEmit, ID_AdmissionTraceVolume, admissionTraceVolume);
            bool hasDepth = currentDepthEvidenceAvailable && currentDepthTexture != null;
            bool hasEdge = hasDepth && currentEdgeReasonTexture != null;
            _compute.SetFloat(ID_CurrentDepthEvidenceAvailable, hasDepth ? 1f : 0f);
            _compute.SetFloat(ID_CurrentEdgeEvidenceAvailable, hasEdge ? 1f : 0f);
            if (currentDepthTexture != null)
                _compute.SetTexture(_kClassifyAndEmit, DepthCapture.DepthTexID, currentDepthTexture);
            if (currentEdgeReasonTexture != null)
                _compute.SetTexture(_kClassifyAndEmit, DepthCapture.EdgeReasonTexID, currentEdgeReasonTexture);
            _compute.SetTexture(_kGenerateIndices, ID_TsdfVolume, tsdfVolume);
            // 1. Clear counters
            _compute.Dispatch(_kClearCounters, 1, 1, 1);

            // 2. Classify & emit vertices
            int gx = CeilDiv(_mapCount.x, 4);
            int gy = CeilDiv(_mapCount.y, 4);
            int gz = CeilDiv(_mapCount.z, 4);
            _compute.Dispatch(_kClassifyAndEmit, gx, gy, gz);

            // 3. Build dispatch args from vertex count
            _compute.Dispatch(_kBuildVertexDispatchArgs, 1, 1, 1);

            // 4. Smoothing (optional)
            if (SmoothIterations > 0)
            {
                _compute.DispatchIndirect(_kInitSmooth, _dispatchArgs);

                for (int iter = 0; iter < SmoothIterations; iter++)
                {
                    if (iter % 2 == 0)
                    {
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosA);
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosB);
                    }
                    else
                    {
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosB);
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosA);
                    }
                    _compute.DispatchIndirect(_kSmoothVertices, _dispatchArgs);
                }

                if (SmoothIterations % 2 == 0)
                    _compute.SetBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosA);
                else
                    _compute.SetBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosB);

                _compute.DispatchIndirect(_kApplySmooth, _dispatchArgs);
            }

            // 5. Temporal blend (optional)
            if (TemporalAlphaMax < 1f)
            {
                _compute.SetTexture(_kTemporalBlend, ID_TemporalState, _temporalState);
                _compute.DispatchIndirect(_kTemporalBlend, _dispatchArgs);
            }

            // 6. Generate indices
            _compute.DispatchIndirect(_kGenerateIndices, _dispatchArgs);

            // 7. Build draw indirect args
            _compute.Dispatch(_kBuildIndirectArgs, 1, 1, 1);
        }

        public Bounds GetVolumeBounds(float voxelSize)
        {
            float3 halfExtent = (float3)_voxCount * voxelSize * 0.5f;
            return new Bounds(Vector3.zero, (Vector3)(halfExtent * 2));
        }

        public Bounds GetCoreBounds(float voxelSize)
        {
            float3 min = ((float3)_coreMin - (float3)_voxCount * 0.5f) * voxelSize;
            float3 max = ((float3)_coreMax - (float3)_voxCount * 0.5f) * voxelSize;
            return new Bounds((Vector3)((min + max) * 0.5f), (Vector3)(max - min));
        }

        private void SetGlobalParams(float voxelSize)
        {
            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
            SetRegionParams();
            _compute.SetFloat(ID_VoxSize, voxelSize);
            _compute.SetFloat(ID_MinWeight, MinMeshWeight);
            _compute.SetInt(ID_TotalVoxels, _totalVoxels);
            _compute.SetInt(ID_MaxVertices, _maxVertices);
            _compute.SetFloat(ID_SmoothLambda, SmoothLambda);
            _compute.SetFloat(ID_SmoothBeta, SmoothBeta);
            _compute.SetFloat(ID_TemporalAlphaMax, TemporalAlphaMax);
            _compute.SetFloat(ID_TemporalAlphaMin, TemporalAlphaMin);
            _compute.SetFloat(ID_TemporalDecayRate, TemporalDecayRate);
            _compute.SetFloat(ID_ConvergeThreshold, ConvergenceThreshold);
            _compute.SetFloat(ID_TemporalDeadzone, TemporalDeadzone);
            _compute.SetFloat(ID_StrictObservedEdges, StrictObservedEdges ? 1f : 0f);
            _compute.SetFloat(ID_DiagnosticRoiEnabled, DiagnosticRoiEnabled ? 1f : 0f);
            _compute.SetVector(ID_DiagnosticRoiRect, DiagnosticRoiRect);
            _compute.SetVector(ID_DiagnosticRoiSplitX,
                new Vector4(DiagnosticRoiSplitX.x, DiagnosticRoiSplitX.y, 0f, 0f));
        }

        private void SetRegionParams()
        {
            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
            _compute.SetInts(ID_MapMin, _mapMin.x, _mapMin.y, _mapMin.z);
            _compute.SetInts(ID_MapCount, _mapCount.x, _mapCount.y, _mapCount.z);
            _compute.SetInts(ID_CoreMin, _coreMin.x, _coreMin.y, _coreMin.z);
            _compute.SetInts(ID_CoreMax, _coreMax.x, _coreMax.y, _coreMax.z);
            _compute.SetInt(ID_CandidateHistoryCapacity, _candidateHistoryCapacity);
            _compute.SetInt(ID_CandidateHistoryMask, _candidateHistoryCapacity - 1);
        }

        private void BindAllBuffers()
        {
            BindBuffer(_kClearCounters, ID_Counters, _counters);
            BindBuffer(_kClearCandidateHistory, ID_CandidateHistoryKeys, _candidateHistoryKeys);
            BindBuffer(_kClearCandidateHistory, ID_CandidateHistoryStates, _candidateHistoryStates);

            BindBuffer(_kClassifyAndEmit, ID_CoordVertMap, _coordVertMap);
            BindBuffer(_kClassifyAndEmit, ID_Vertices, _vertices);
            BindBuffer(_kClassifyAndEmit, ID_VertexAdmissionClass, _vertexAdmissionClass);
            BindBuffer(_kClassifyAndEmit, ID_Counters, _counters);

            BindBuffer(_kBuildVertexDispatchArgs, ID_Counters, _counters);
            BindBuffer(_kBuildVertexDispatchArgs, ID_DispatchArgs, _dispatchArgs);

            BindBuffer(_kInitSmooth, ID_Vertices, _vertices);
            BindBuffer(_kInitSmooth, ID_SmoothPosA, _smoothPosA);
            BindBuffer(_kInitSmooth, ID_Counters, _counters);

            BindBuffer(_kSmoothVertices, ID_Vertices, _vertices);
            BindBuffer(_kSmoothVertices, ID_CoordVertMap, _coordVertMap);
            BindBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosA);
            BindBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosB);
            BindBuffer(_kSmoothVertices, ID_Counters, _counters);

            BindBuffer(_kApplySmooth, ID_Vertices, _vertices);
            BindBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosA);
            BindBuffer(_kApplySmooth, ID_Counters, _counters);

            BindBuffer(_kTemporalBlend, ID_Vertices, _vertices);
            BindBuffer(_kTemporalBlend, ID_Counters, _counters);

            BindBuffer(_kGenerateIndices, ID_Vertices, _vertices);
            BindBuffer(_kGenerateIndices, ID_CoordVertMap, _coordVertMap);
            BindBuffer(_kGenerateIndices, ID_Indices, _indices);
            BindBuffer(_kGenerateIndices, ID_VertexAdmissionClass, _vertexAdmissionClass);
            BindBuffer(_kGenerateIndices, ID_Counters, _counters);
            BindBuffer(_kGenerateIndices, ID_CandidateHistoryKeys, _candidateHistoryKeys);
            BindBuffer(_kGenerateIndices, ID_CandidateHistoryStates, _candidateHistoryStates);

            BindBuffer(_kBuildIndirectArgs, ID_Counters, _counters);
            BindBuffer(_kBuildIndirectArgs, ID_DrawIndirectArgs, _drawIndirectArgs);
        }

        private void BindBuffer(int kernel, int nameID, GraphicsBuffer buffer)
        {
            _compute.SetBuffer(kernel, nameID, buffer);
        }

        public void Dispose()
        {
            _coordVertMap?.Release();
            _vertices?.Release();
            _indices?.Release();
            _vertexAdmissionClass?.Release();
            _counters?.Release();
            _dispatchArgs?.Release();
            _drawIndirectArgs?.Release();
            _smoothPosA?.Release();
            _smoothPosB?.Release();
            _candidateHistoryKeys?.Release();
            _candidateHistoryStates?.Release();

            if (_temporalState != null)
            {
                _temporalState.Release();
                UnityEngine.Object.Destroy(_temporalState);
            }

            _coordVertMap = null;
            _vertices = null;
            _indices = null;
            _vertexAdmissionClass = null;
            _counters = null;
            _dispatchArgs = null;
            _drawIndirectArgs = null;
            _smoothPosA = null;
            _smoothPosB = null;
            _candidateHistoryKeys = null;
            _candidateHistoryStates = null;
            _temporalState = null;

            _totalVoxels = 0;
            _mapVoxels = 0;
            _candidateHistoryCapacity = 0;
            _temporalInitialized = false;
            _candidateHistoryInitialized = false;
            _candidateExtractionEpoch = 0;
        }

        private static int CeilDiv(int a, int b) => (a + b - 1) / b;
    }
}

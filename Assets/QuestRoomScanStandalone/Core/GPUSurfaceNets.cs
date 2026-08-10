using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    internal class GPUSurfaceNets : IDisposable
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
        private GraphicsBuffer _candidateHistoryStates;

        // Temporal state as 3D texture (avoids 128MB structured buffer limit on Quest)
        private RenderTexture _temporalState;

        // Sizing
        private int3 _voxCount;
        private int _totalVoxels;
        private int _maxVertices;
        private int _maxIndices;
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
        public GraphicsBuffer CountersBuffer => _counters;

        private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ID_ColorVolume = Shader.PropertyToID("_ColorVolume");
        private static readonly int ID_AdmissionTraceVolume = Shader.PropertyToID("_AdmissionTraceVolume");
        private static readonly int ID_VoxCount = Shader.PropertyToID("_VoxCount");
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
        private const int CounterCount = 138;
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
        }

        public void EnsureBuffers(int3 voxCount, float vertexBudgetPercent = 0.05f)
        {
            int totalVoxels = voxCount.x * voxCount.y * voxCount.z;
            if (_totalVoxels == totalVoxels && _coordVertMap != null)
                return;

            Dispose();

            _voxCount = voxCount;
            _totalVoxels = totalVoxels;
            _maxVertices = Mathf.Max(1024, (int)(totalVoxels * vertexBudgetPercent));
            _maxIndices = _maxVertices * 18;

            const GraphicsBuffer.Target structuredIndirect =
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;

            _coordVertMap = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVoxels, 4);
            _vertices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, VertexStride);
            _indices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxIndices, 4);
            _vertexAdmissionClass = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, 4);
            _counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CounterCount, 4);
            _dispatchArgs = new GraphicsBuffer(structuredIndirect, 3, 4);
            _drawIndirectArgs = new GraphicsBuffer(structuredIndirect, 5, 4);
            _smoothPosA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
            _smoothPosB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
            _candidateHistoryKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CandidateHistoryCapacity, 4);
            _candidateHistoryStates = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CandidateHistoryCapacity, 4);

            // Temporal state as RWTexture3D<float4> -- avoids the 128MB structured buffer limit.
            // 256^3 x RGBA32Float = 256MB as a 3D texture, which Quest supports (same as TSDF volume path).
            _temporalState = new RenderTexture(voxCount.x, voxCount.y, 0, GraphicsFormat.R32G32B32A32_SFloat)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            _temporalState.Create();

            _temporalInitialized = false;
            _candidateHistoryInitialized = false;
            _candidateExtractionEpoch = 0;

            long totalBytes = (long)totalVoxels * 4
                            + (long)_maxVertices * VertexStride
                            + (long)_maxIndices * 4
                            + (long)_maxVertices * 4
                            + CounterCount * 4 + 3 * 4 + 5 * 4
                            + (long)_maxVertices * Float3Stride * 2
                            + (long)totalVoxels * 16
                            + (long)CandidateHistoryCapacity * 8;
            Logger.Info($"[GPUSurfaceNets] Allocated buffers: vox={voxCount}, " +
                      $"maxVerts={_maxVertices}, maxIdx={_maxIndices}, " +
                      $"totalGPU={totalBytes / (1024 * 1024)}MB");
        }

        public void InitTemporalState()
        {
            if (_temporalInitialized || _temporalState == null) return;

            _compute.SetTexture(_kInitTemporal, ID_TemporalState, _temporalState);
            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
            int gx = CeilDiv(_voxCount.x, 4);
            int gy = CeilDiv(_voxCount.y, 4);
            int gz = CeilDiv(_voxCount.z, 4);
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

            if (!_candidateHistoryInitialized)
            {
                _compute.Dispatch(_kClearCandidateHistory, CeilDiv(CandidateHistoryCapacity, 256), 1, 1);
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
            int gx = CeilDiv(_voxCount.x, 4);
            int gy = CeilDiv(_voxCount.y, 4);
            int gz = CeilDiv(_voxCount.z, 4);
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

        private void SetGlobalParams(float voxelSize)
        {
            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
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
            _temporalInitialized = false;
            _candidateHistoryInitialized = false;
            _candidateExtractionEpoch = 0;
        }

        private static int CeilDiv(int a, int b) => (a + b - 1) / b;
    }
}

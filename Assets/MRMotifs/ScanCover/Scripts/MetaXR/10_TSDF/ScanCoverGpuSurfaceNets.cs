using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// GPU Surface Nets extraction orchestrator — faithful port of
/// QuestRoomScan's GPUSurfaceNets.cs, adapted to ScanCover's dual-volume
/// layout (separate TSDF + weight R32 textures, no color volume).
///
/// The extracted mesh can either stay on GPU (vertex/index buffers are
/// exposed for future DrawProceduralIndirect rendering) or be read back
/// to CPU lists so the existing mesh/snapshot display path keeps working
/// unchanged.
/// </summary>
public class ScanCoverGpuSurfaceNets : IDisposable
{
    private readonly ComputeShader _compute;

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

    private GraphicsBuffer _coordVertMap;
    private GraphicsBuffer _vertices;
    private GraphicsBuffer _indices;
    private GraphicsBuffer _counters;
    private GraphicsBuffer _dispatchArgs;
    private GraphicsBuffer _drawIndirectArgs;
    private GraphicsBuffer _smoothPosA;
    private GraphicsBuffer _smoothPosB;

    // Temporal state as 3D texture (avoids the 128MB structured buffer
    // limit on Quest), same as QuestRoomScan.
    private RenderTexture _temporalState;

    private Vector3Int _voxCount;
    private int _totalVoxels;
    private int _maxVertices;
    private int _maxIndices;
    private bool _temporalInitialized;

    // Async mesh readback state.
    private bool _hasPendingReadback;
    private AsyncGPUReadbackRequest _countersRequest;
    private AsyncGPUReadbackRequest _verticesRequest;
    private AsyncGPUReadbackRequest _indicesRequest;

    public float MinMeshWeight = 0.08f;
    public int SmoothIterations = 1;
    public float SmoothLambda = 0.33f;
    public float SmoothBeta = 0.5f;
    public float TemporalAlphaMax = 0.85f;
    public float TemporalAlphaMin = 0.1f;
    public float TemporalDecayRate = 0.15f;
    public float ConvergenceThreshold = 0.005f;
    public float TemporalDeadzone = 0.001f;

    public bool HasPendingReadback => _hasPendingReadback;
    public GraphicsBuffer VertexBuffer => _vertices;
    public GraphicsBuffer IndexBuffer => _indices;
    public GraphicsBuffer DrawIndirectArgsBuffer => _drawIndirectArgs;

    private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
    private static readonly int ID_WeightVolume = Shader.PropertyToID("_WeightVolume");
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

    private static readonly int ID_CoordVertMap = Shader.PropertyToID("_CoordVertMap");
    private static readonly int ID_Vertices = Shader.PropertyToID("_Vertices");
    private static readonly int ID_Indices = Shader.PropertyToID("_Indices");
    private static readonly int ID_Counters = Shader.PropertyToID("_Counters");
    private static readonly int ID_DispatchArgs = Shader.PropertyToID("_DispatchArgs");
    private static readonly int ID_DrawIndirectArgs = Shader.PropertyToID("_DrawIndirectArgs");
    private static readonly int ID_SmoothPosA = Shader.PropertyToID("_SmoothPosA");
    private static readonly int ID_SmoothPosB = Shader.PropertyToID("_SmoothPosB");
    private static readonly int ID_TemporalState = Shader.PropertyToID("_TemporalState");

    private const int VertexStride = 32;
    private const int Float3Stride = 12;

    // Must match the GPUVertex struct in ScanCoverGpuSurfaceNets.compute.
    private struct GPUVertexData
    {
        public Vector3 pos;
        public Vector3 norm;
        public uint voxelFlatIdx;
        public uint pad;
    }

    public ScanCoverGpuSurfaceNets(ComputeShader compute)
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
    }

    public void EnsureBuffers(Vector3Int voxCount, float vertexBudgetPercent = 0.05f)
    {
        int totalVoxels = voxCount.x * voxCount.y * voxCount.z;
        if (_totalVoxels == totalVoxels && _coordVertMap != null)
            return;

        DisposeBuffers();

        _voxCount = voxCount;
        _totalVoxels = totalVoxels;
        _maxVertices = Mathf.Max(1024, (int)(totalVoxels * vertexBudgetPercent));
        _maxIndices = _maxVertices * 18;

        const GraphicsBuffer.Target structuredIndirect =
            GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;

        _coordVertMap = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVoxels, 4);
        _vertices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, VertexStride);
        _indices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxIndices, 4);
        _counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 2, 4);
        _dispatchArgs = new GraphicsBuffer(structuredIndirect, 3, 4);
        _drawIndirectArgs = new GraphicsBuffer(structuredIndirect, 5, 4);
        _smoothPosA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
        _smoothPosB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);

        _temporalState = new RenderTexture(voxCount.x, voxCount.y, 0, UnityEngine.Experimental.Rendering.GraphicsFormat.R32G32B32A32_SFloat)
        {
            dimension = TextureDimension.Tex3D,
            volumeDepth = voxCount.z,
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        _temporalState.Create();

        _temporalInitialized = false;

        if (_totalVoxels > 0)
        {
            long totalBytes = (long)totalVoxels * 4
                            + (long)_maxVertices * VertexStride
                            + (long)_maxIndices * 4
                            + (long)_maxVertices * Float3Stride * 2
                            + (long)totalVoxels * 16;
            Debug.Log($"[ScanCoverGpuSurfaceNets] Allocated buffers: vox={voxCount}, " +
                      $"maxVerts={_maxVertices}, maxIdx={_maxIndices}, " +
                      $"totalGPU={totalBytes / (1024 * 1024)}MB");
        }
    }

    public void ResetTemporal()
    {
        _temporalInitialized = false;
    }

    private void InitTemporalState()
    {
        if (_temporalInitialized || _temporalState == null) return;

        _compute.SetTexture(_kInitTemporal, ID_TemporalState, _temporalState);
        _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
        _compute.Dispatch(_kInitTemporal,
            CeilDiv(_voxCount.x, 4), CeilDiv(_voxCount.y, 4), CeilDiv(_voxCount.z, 4));
        _temporalInitialized = true;
    }

    public void Extract(RenderTexture tsdfVolume, RenderTexture weightVolume, float voxelSize)
    {
        if (_coordVertMap == null)
            throw new InvalidOperationException("Call EnsureBuffers before Extract");

        if (!_temporalInitialized && TemporalAlphaMax < 1f)
            InitTemporalState();

        SetGlobalParams(voxelSize);
        BindAllBuffers();

        _compute.SetTexture(_kClassifyAndEmit, ID_TsdfVolume, tsdfVolume);
        _compute.SetTexture(_kClassifyAndEmit, ID_WeightVolume, weightVolume);
        _compute.SetTexture(_kGenerateIndices, ID_TsdfVolume, tsdfVolume);
        _compute.SetTexture(_kGenerateIndices, ID_WeightVolume, weightVolume);

        // 1. Clear counters
        _compute.Dispatch(_kClearCounters, 1, 1, 1);

        // 2. Classify & emit vertices
        _compute.Dispatch(_kClassifyAndEmit,
            CeilDiv(_voxCount.x, 4), CeilDiv(_voxCount.y, 4), CeilDiv(_voxCount.z, 4));

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

    /// <summary>
    /// Requests an async readback of the extracted mesh (counters +
    /// compacted vertex/index buffers).  Call TryConsumeMeshReadback from
    /// Update until it returns true.
    /// </summary>
    public bool RequestMeshReadback()
    {
        if (_hasPendingReadback || _vertices == null)
            return false;

        _countersRequest = AsyncGPUReadback.Request(_counters);
        _verticesRequest = AsyncGPUReadback.Request(_vertices);
        _indicesRequest = AsyncGPUReadback.Request(_indices);
        _hasPendingReadback = true;
        return true;
    }

    /// <summary>
    /// Polls the pending readback; when complete, fills vertices/triangles
    /// (volume-local positions, compact triangle indices) and returns true.
    /// Returns false while still pending.  On error, sets issue and
    /// returns true with empty outputs so the caller can fail gracefully.
    /// </summary>
    public bool TryConsumeMeshReadback(List<Vector3> vertices, List<int> triangles, out string issue)
    {
        issue = null;
        if (!_hasPendingReadback)
            return false;

        if (!_countersRequest.done || !_verticesRequest.done || !_indicesRequest.done)
            return false;

        _hasPendingReadback = false;
        vertices.Clear();
        triangles.Clear();

        if (_countersRequest.hasError || _verticesRequest.hasError || _indicesRequest.hasError)
        {
            issue = "GPU surface nets mesh readback failed.";
            return true;
        }

        uint[] counters = _countersRequest.GetData<uint>().ToArray();
        int vertCount = Mathf.Min((int)counters[0], _maxVertices);
        int indexCount = Mathf.Min((int)counters[1], _maxIndices);

        var verts = _verticesRequest.GetData<GPUVertexData>();
        for (int i = 0; i < vertCount; i++)
            vertices.Add(verts[i].pos);

        var indices = _indicesRequest.GetData<uint>();
        for (int i = 0; i + 2 < indexCount; i += 3)
        {
            uint a = indices[i];
            uint b = indices[i + 1];
            uint c = indices[i + 2];
            if (a >= (uint)vertCount || b >= (uint)vertCount || c >= (uint)vertCount)
                continue;
            triangles.Add((int)a);
            triangles.Add((int)b);
            triangles.Add((int)c);
        }

        return true;
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
    }

    private void BindAllBuffers()
    {
        _compute.SetBuffer(_kClearCounters, ID_Counters, _counters);

        _compute.SetBuffer(_kClassifyAndEmit, ID_CoordVertMap, _coordVertMap);
        _compute.SetBuffer(_kClassifyAndEmit, ID_Vertices, _vertices);
        _compute.SetBuffer(_kClassifyAndEmit, ID_Counters, _counters);

        _compute.SetBuffer(_kBuildVertexDispatchArgs, ID_Counters, _counters);
        _compute.SetBuffer(_kBuildVertexDispatchArgs, ID_DispatchArgs, _dispatchArgs);

        _compute.SetBuffer(_kInitSmooth, ID_Vertices, _vertices);
        _compute.SetBuffer(_kInitSmooth, ID_SmoothPosA, _smoothPosA);
        _compute.SetBuffer(_kInitSmooth, ID_Counters, _counters);

        _compute.SetBuffer(_kSmoothVertices, ID_Vertices, _vertices);
        _compute.SetBuffer(_kSmoothVertices, ID_CoordVertMap, _coordVertMap);
        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosA);
        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosB);
        _compute.SetBuffer(_kSmoothVertices, ID_Counters, _counters);

        _compute.SetBuffer(_kApplySmooth, ID_Vertices, _vertices);
        _compute.SetBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosA);
        _compute.SetBuffer(_kApplySmooth, ID_Counters, _counters);

        _compute.SetBuffer(_kTemporalBlend, ID_Vertices, _vertices);
        _compute.SetBuffer(_kTemporalBlend, ID_Counters, _counters);

        _compute.SetBuffer(_kGenerateIndices, ID_Vertices, _vertices);
        _compute.SetBuffer(_kGenerateIndices, ID_CoordVertMap, _coordVertMap);
        _compute.SetBuffer(_kGenerateIndices, ID_Indices, _indices);
        _compute.SetBuffer(_kGenerateIndices, ID_Counters, _counters);

        _compute.SetBuffer(_kBuildIndirectArgs, ID_Counters, _counters);
        _compute.SetBuffer(_kBuildIndirectArgs, ID_DrawIndirectArgs, _drawIndirectArgs);
    }

    public void Dispose()
    {
        DisposeBuffers();
    }

    private void DisposeBuffers()
    {
        _coordVertMap?.Release();
        _vertices?.Release();
        _indices?.Release();
        _counters?.Release();
        _dispatchArgs?.Release();
        _drawIndirectArgs?.Release();
        _smoothPosA?.Release();
        _smoothPosB?.Release();

        if (_temporalState != null)
        {
            _temporalState.Release();
            UnityEngine.Object.Destroy(_temporalState);
        }

        _coordVertMap = null;
        _vertices = null;
        _indices = null;
        _counters = null;
        _dispatchArgs = null;
        _drawIndirectArgs = null;
        _smoothPosA = null;
        _smoothPosB = null;
        _temporalState = null;

        _totalVoxels = 0;
        _temporalInitialized = false;
        _hasPendingReadback = false;
    }

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;
}

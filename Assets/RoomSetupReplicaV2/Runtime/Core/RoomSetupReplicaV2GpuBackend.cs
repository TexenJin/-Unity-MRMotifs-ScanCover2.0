using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

/// <summary>
/// Live RoomSetupReplica production backend. Depth stays on the GPU from the
/// callback-latched stereo texture through TSDF integration, the recovered
/// v0.1.1 aligned 10 cm Marching Cubes field, and indirect rendering.
/// </summary>
internal sealed class RoomSetupReplicaV2GpuBackend : IDisposable
{
    private const int TileSampleCount = 31;
    private const int MeshStride = 2;
    private const int TileCellCount =
        (TileSampleCount - 1) / MeshStride;
    private const int TileStepSamples = TileCellCount * MeshStride;
    private const int MaximumVerticesPerCell = 15;
    // Preserve the former dense route's worst-case global output budget while
    // allowing an unbounded number of world tiles to contribute sparsely.
    private const int MaximumTriangleVertices =
        79 * 47 * 79 * MaximumVerticesPerCell;
    private const int MaximumTriangleCount =
        MaximumTriangleVertices / 3;
    private const int UniqueEdgeHashCapacity = 2097152;
    private const float VoxelMeters = 0.05f;
    private const float TileStepMeters = TileStepSamples * VoxelMeters;
    private const float TileActivationPaddingMeters = 0.10f;
    private const float MinimumDepthMeters = 0.30f;
    private const float MaximumDepthMeters = 3.00f;
    private const float PositiveTruncationScale = 0.20f;
    private const float NegativeTruncationScale = 0.20f / 3f;
    private const float GradientThresholdSquared = 0.01f;
    private const float MaximumWeight = 1000f;
    private const float ProbeStableDeltaMeters = 0.025f;
    private const float ProbeUnstableDeltaMeters = 0.050f;
    private const float ProbeMaximumStability = 16f;
    private const float ProbeStableVoteThreshold = 8f;
    private const float ProbeUnstablePenalty = 4f;
    private const int RemoveSmallTriangleThreshold = 32;
    private const int RemoveSmallTriangleGate = 40000;
    private const int ConnectivityVertexCapacity =
        RemoveSmallTriangleGate * 3;
    private const int PositionHashCapacity = 262144;
    private const float SeamWeldUnitsPerMeter = 10000f;
    private const int MeshEveryIntegrations = 2;
    private const int MaximumStructuralSamples = 16384;
    // The official-style display reveals only the central 70% binocular-common
    // circle. The 70-80% annulus may fuse and correct in the background, while
    // samples outside the common 80% circle have no TSDF write authority.
    private const float RevealNdcRadius = 0.70f;
    private const float FusionNdcRadius = 0.80f;
    private const float RevealOcclusionToleranceMeters = 0.12f;

    private static readonly int CandidateTriangleVerticesId =
        Shader.PropertyToID("_RoomSetupMcCandidateVertices");
    private static readonly int FilteredTriangleVerticesId =
        Shader.PropertyToID("_RoomSetupMcFilteredVertices");
    private static readonly int MarchingCubesCountersId =
        Shader.PropertyToID("_RoomSetupMcCounters");
    private static readonly int CompactDisplayEdgesId =
        Shader.PropertyToID("_RoomSetupMcCompactDisplayEdges");
    private static readonly int CandidateTriangleRevealedId =
        Shader.PropertyToID("_RoomSetupMcCandidateRevealed");
    private static readonly int FilteredTriangleRevealedId =
        Shader.PropertyToID("_RoomSetupMcFilteredRevealed");
    private static readonly int BaseColorId =
        Shader.PropertyToID("_BaseColor");
    private static readonly int LineWidthPixelsId =
        Shader.PropertyToID("_LineWidthPixels");
    private static readonly int UniqueEdgesOnlyId =
        Shader.PropertyToID("_UniqueEdgesOnly");
    private readonly GameObject _owner;
    private readonly Color _lineColor;
    private readonly float _lineWidthPixels;
    private readonly Matrix4x4[] _environmentReprojection =
        new Matrix4x4[2];
    private readonly Matrix4x4[] _environmentReprojectionInverse =
        new Matrix4x4[2];
    private readonly Plane[] _leftFrustumPlanes = new Plane[6];
    private readonly Plane[] _rightFrustumPlanes = new Plane[6];
    private readonly Dictionary<Vector3Int, GpuTile> _tiles = new();
    private readonly List<GpuTile> _activeTiles = new();
    private sealed class GpuTile
    {
        public Vector3Int key;
        public Vector3 volumeOrigin;
        public Bounds bounds;
        public RenderTexture tsdf;
        public RenderTexture stabilityProbe;
        public RenderTexture revealedCells;
    }

    private ComputeShader _fusion;
    private ComputeShader _marchingCubes;
    // Readback-only companion for export and simplified-room generation. It
    // never participates in the live fusion or draw path.
    private ComputeShader _structuralReduction;
    private Material _wireMaterial;
    private Material _depthMaterial;
    private MaterialPropertyBlock _drawProperties;
    private int _clearKernel = -1;
    private int _integrateEnvironmentKernel = -1;
    private int _mcClearKernel = -1;
    private int _mcClearRevealKernel = -1;
    private int _mcEmitKernel = -1;
    private int _mcConnectivityArgsKernel = -1;
    private int _mcInitializeConnectivityKernel = -1;
    private int _mcUnionConnectivityKernel = -1;
    private int _mcWeldQuantizedPositionsKernel = -1;
    private int _mcCompressConnectivityKernel = -1;
    private int _mcCountComponentsKernel = -1;
    private int _mcFilterComponentsKernel = -1;
    private int _mcArgsKernel = -1;
    private int _mcClearDisplayEdgesKernel = -1;
    private int _mcClearUniqueEdgesKernel = -1;
    private int _mcBuildUniqueEdgeArgsKernel = -1;
    private int _mcBuildUniqueEdgesKernel = -1;
    private int _mcCompactDisplayEdgesKernel = -1;
    private int _mcBuildDisplayEdgeArgsKernel = -1;
    private int _structuralSampleKernel = -1;
    private GraphicsBuffer _triangleEdges;
    private GraphicsBuffer _candidateTriangleVertices;
    private GraphicsBuffer _filteredTriangleVertices;
    private GraphicsBuffer _candidateVertexStability;
    private GraphicsBuffer _filteredVertexStability;
    private GraphicsBuffer _candidateTriangleRevealed;
    private GraphicsBuffer _filteredTriangleRevealed;
    private GraphicsBuffer _connectivityParents;
    private GraphicsBuffer _componentTriangleCounts;
    private GraphicsBuffer _positionHashOwners;
    private GraphicsBuffer _mcCounters;
    private GraphicsBuffer _connectivityDispatchArgs;
    private GraphicsBuffer _drawArgs;
    private GraphicsBuffer _lineDrawArgs;
    private GraphicsBuffer _uniqueEdgeHashOwners;
    private GraphicsBuffer _uniqueEdgeMasks;
    private GraphicsBuffer _uniqueEdgeDispatchArgs;
    private GraphicsBuffer _compactDisplayEdges;
    private GraphicsBuffer _structuralSamples;
    private Bounds _drawBounds;
    private bool _hasDrawBounds;
    private bool _initialized;
    private bool _visible = true;
    private bool _failureLogged;
    // A manifold triangle mesh has about 1.5 unique edges per triangle. With
    // six portable triangle vertices per compacted edge this is about nine
    // vertex shader invocations per source triangle, half the old 18.
    private bool _uniqueEdgesOnly = true;
    private int _integrationCount;
    private bool _structuralReadbackPending;
    private bool _meshReadbackPending;
    private int _meshReadbackGeneration;

    public int IntegrationCount => _integrationCount;
    public int TileCount => _tiles.Count;
    public bool Ready => _initialized;
    public bool UniqueEdgesOnly => _uniqueEdgesOnly;

    public bool TryRequestStructuralSamples(Action<Vector4[]> completed)
    {
        if (!_initialized || _structuralReadbackPending ||
            _structuralReduction == null || _structuralSamples == null ||
            _candidateTriangleVertices == null ||
            _filteredTriangleVertices == null || _mcCounters == null)
            return false;

        _structuralReduction.SetInt(
            "_MaximumTriangleVertices", MaximumTriangleVertices);
        _structuralReduction.SetInt(
            "_MaximumStructuralSamples", MaximumStructuralSamples);
        _structuralReduction.SetBuffer(
            _structuralSampleKernel,
            "_CandidateTriangleVertices", _candidateTriangleVertices);
        _structuralReduction.SetBuffer(
            _structuralSampleKernel,
            "_FilteredTriangleVertices", _filteredTriangleVertices);
        _structuralReduction.SetBuffer(
            _structuralSampleKernel, "_Counters", _mcCounters);
        _structuralReduction.SetBuffer(
            _structuralSampleKernel,
            "_StructuralSamples", _structuralSamples);
        _structuralReduction.Dispatch(
            _structuralSampleKernel,
            Mathf.CeilToInt(MaximumStructuralSamples / 64f), 1, 1);

        _structuralReadbackPending = true;
        AsyncGPUReadback.Request(_structuralSamples, request =>
        {
            _structuralReadbackPending = false;
            if (request.hasError)
            {
                completed?.Invoke(null);
                return;
            }
            var data = request.GetData<Vector4>();
            var copy = new Vector4[data.Length];
            data.CopyTo(copy);
            completed?.Invoke(copy);
        });
        return true;
    }

    public bool TryRequestMeshSnapshot(
        Action<Vector3[], float[], uint[]> completed)
    {
        if (!_initialized || _meshReadbackPending ||
            _mcCounters == null || _candidateTriangleVertices == null ||
            _filteredTriangleVertices == null)
            return false;

        _meshReadbackPending = true;
        int generation = ++_meshReadbackGeneration;
        AsyncGPUReadback.Request(_mcCounters, counterRequest =>
        {
            if (!IsCurrentMeshReadback(generation))
                return;
            if (counterRequest.hasError)
            {
                _meshReadbackPending = false;
                completed?.Invoke(null, null, null);
                return;
            }
            var counterData = counterRequest.GetData<uint>();
            var counters = new uint[counterData.Length];
            counterData.CopyTo(counters);
            bool cleanupApplied = counters.Length > 6 && counters[6] != 0u;
            uint rawCount = cleanupApplied && counters.Length > 4
                ? counters[4]
                : counters.Length > 0 ? counters[0] : 0u;
            int vertexCount = Mathf.Min(
                (int)Math.Min(rawCount, int.MaxValue),
                cleanupApplied
                    ? ConnectivityVertexCapacity
                    : MaximumTriangleVertices);
            vertexCount -= vertexCount % 3;
            if (vertexCount <= 0)
            {
                _meshReadbackPending = false;
                completed?.Invoke(Array.Empty<Vector3>(),
                    Array.Empty<float>(), counters);
                return;
            }

            GraphicsBuffer source = cleanupApplied
                ? _filteredTriangleVertices
                : _candidateTriangleVertices;
            AsyncGPUReadback.Request(
                source, vertexCount * sizeof(float) * 3, 0,
                vertexRequest =>
                {
                    if (!IsCurrentMeshReadback(generation))
                        return;
                    _meshReadbackPending = false;
                    if (vertexRequest.hasError)
                    {
                        completed?.Invoke(null, null, counters);
                        return;
                    }
                    var vertexData = vertexRequest.GetData<Vector3>();
                    var vertices = new Vector3[vertexData.Length];
                    vertexData.CopyTo(vertices);
                    // v0.1.3 has no experimental stability channel.
                    completed?.Invoke(vertices, Array.Empty<float>(), counters);
                });
        });
        return true;
    }

    public void CancelMeshSnapshot()
    {
        if (!_meshReadbackPending)
            return;
        _meshReadbackPending = false;
        _meshReadbackGeneration++;
    }

    private bool IsCurrentMeshReadback(int generation)
    {
        return _meshReadbackPending && generation == _meshReadbackGeneration;
    }

    public RoomSetupReplicaV2GpuBackend(
        GameObject owner, Color lineColor, float lineWidthPixels)
    {
        _owner = owner;
        _lineColor = lineColor;
        _lineWidthPixels = lineWidthPixels;
    }

    /// <summary>
    /// Integrates the globally published Meta Environment Depth frame. This
    /// entry point is used only by the production Environment Depth scene, where the
    /// EnvironmentDepthManager exclusively owns depth acquisition.
    /// </summary>
    public bool TryObserveEnvironmentDepthGlobal(Camera camera)
    {
        if (camera == null)
            return false;
        Texture source = Shader.GetGlobalTexture(
            "_EnvironmentDepthTexture");
        Matrix4x4[] reprojection = Shader.GetGlobalMatrixArray(
            "_EnvironmentDepthReprojectionMatrices");
        Vector4 zBufferParams = Shader.GetGlobalVector(
            "_EnvironmentDepthZBufferParams");
        if (source == null || source.dimension != TextureDimension.Tex2DArray ||
            source.width <= 0 || source.height <= 0 ||
            reprojection == null || reprojection.Length < 2 ||
            !IsFinite(zBufferParams) ||
            Mathf.Abs(zBufferParams.x) <= 0.000001f)
            return false;

        for (int eye = 0; eye < 2; eye++)
        {
            Matrix4x4 worldToDepth = reprojection[eye];
            if (!IsFinite(worldToDepth) ||
                Mathf.Abs(worldToDepth.determinant) <= 0.000001f)
                return false;
            _environmentReprojection[eye] = worldToDepth;
            _environmentReprojectionInverse[eye] = worldToDepth.inverse;
        }

        if (!EnsureInitialized())
            return false;

        // The reveal decision uses the same latched stereo Environment Depth
        // frame as this integration. Both eyes contribute to one shared
        // world-space ledger, so they can never reveal different geometry.
        _marchingCubes.SetTexture(
            _mcEmitKernel, "_EnvironmentDepthTexture", source);
        _marchingCubes.SetInts("_DepthSize", source.width, source.height);
        _marchingCubes.SetMatrixArray(
            "_EnvironmentDepthReprojectionMatrices",
            _environmentReprojection);
        _marchingCubes.SetVector(
            "_EnvironmentDepthZBufferParams", zBufferParams);

        Vector3 eyeCentre = camera.transform.position;
        EnsureActiveTiles(
            eyeCentre,
            _environmentReprojection[0],
            _environmentReprojection[1]);
        _fusion.SetTexture(
            _integrateEnvironmentKernel,
            "_EnvironmentDepthTexture", source);
        _fusion.SetInts("_DepthSize", source.width, source.height);
        _fusion.SetMatrixArray(
            "_EnvironmentDepthReprojectionMatrices",
            _environmentReprojection);
        _fusion.SetMatrixArray(
            "_EnvironmentDepthReprojectionInverse",
            _environmentReprojectionInverse);
        _fusion.SetVector(
            "_EnvironmentDepthZBufferParams", zBufferParams);
        _fusion.SetVector("_EnvironmentEyeCentre", eyeCentre);
        for (int tileIndex = 0;
             tileIndex < _activeTiles.Count; tileIndex++)
        {
            GpuTile tile = _activeTiles[tileIndex];
            _fusion.SetVector("_VolumeOrigin", tile.volumeOrigin);
            BindFusionVolumes(_integrateEnvironmentKernel, tile);
            DispatchFusionVolume(_integrateEnvironmentKernel);
        }
        _integrationCount++;

        if (_integrationCount == 1 ||
            _integrationCount % MeshEveryIntegrations == 0)
        {
            ExtractMarchingCubes();
        }
        return true;
    }

    public void Render()
    {
        if (!_initialized || !_visible || _drawArgs == null ||
            _lineDrawArgs == null ||
            _candidateTriangleVertices == null || _drawProperties == null)
            return;

        if (_depthMaterial != null)
        {
            RenderParams depthParams = new RenderParams(_depthMaterial)
            {
                worldBounds = _drawBounds,
                matProps = _drawProperties,
                shadowCastingMode = ShadowCastingMode.Off,
                receiveShadows = false,
                layer = _owner != null ? _owner.layer : 0
            };
            Graphics.RenderPrimitivesIndirect(
                depthParams, MeshTopology.Triangles, _drawArgs, 1);
        }

        if (_wireMaterial == null)
            return;
        RenderParams wireParams = new RenderParams(_wireMaterial)
        {
            worldBounds = _drawBounds,
            matProps = _drawProperties,
            shadowCastingMode = ShadowCastingMode.Off,
            receiveShadows = false,
            layer = _owner != null ? _owner.layer : 0
        };
        Graphics.RenderPrimitivesIndirect(
            wireParams, MeshTopology.Triangles, _lineDrawArgs, 1);
    }

    public void SetVisible(bool visible)
    {
        _visible = visible;
    }

    public void SetUniqueEdgeDisplay(bool enabled)
    {
        _uniqueEdgesOnly = enabled;
        float value = enabled ? 1f : 0f;
        _wireMaterial?.SetFloat(UniqueEdgesOnlyId, value);
        _drawProperties?.SetFloat(UniqueEdgesOnlyId, value);
        if (_marchingCubes != null)
            _marchingCubes.SetInt("_UniqueEdgesOnly", enabled ? 1 : 0);
        if (_initialized)
            BuildUniqueEdgeVisibility();
    }

    public void Clear()
    {
        _integrationCount = 0;
        if (!_initialized)
            return;
        ReleaseAllTiles();
        ExtractMarchingCubes();
    }

    private bool EnsureInitialized()
    {
        if (_initialized)
            return true;
        if (_owner == null)
        {
            LogFailureOnce("GPU production owner is unavailable.");
            return false;
        }

        try
        {
            _fusion = Resources.Load<ComputeShader>(
                "RoomSetupReplicaV2GpuFusion");
            _marchingCubes = Resources.Load<ComputeShader>(
                "RoomSetupReplicaV2GpuMarchingCubes");
            _structuralReduction = Resources.Load<ComputeShader>(
                "RoomSetupReplicaV2StructuralReduction");
            Shader wireShader = Resources.Load<Shader>(
                "RoomSetupReplicaV2GpuMarchingCubes");
            Shader depthShader = Resources.Load<Shader>(
                "RoomSetupReplicaV2GpuMarchingCubesDepth");
            if (_fusion == null || _marchingCubes == null ||
                _structuralReduction == null ||
                wireShader == null || depthShader == null)
            {
                throw new InvalidOperationException(
                    "Dedicated GPU Marching Cubes resources are unavailable.");
            }

            _clearKernel = _fusion.FindKernel("ClearVolume");
            _integrateEnvironmentKernel = _fusion.FindKernel(
                "IntegrateEnvironmentDepth");
            _mcClearKernel = _marchingCubes.FindKernel(
                "ClearMarchingCubes");
            _mcClearRevealKernel = _marchingCubes.FindKernel(
                "ClearRevealVolume");
            _mcEmitKernel = _marchingCubes.FindKernel(
                "EmitMarchingCubes");
            _mcConnectivityArgsKernel = _marchingCubes.FindKernel(
                "BuildConnectivityArgs");
            _mcInitializeConnectivityKernel = _marchingCubes.FindKernel(
                "InitializeConnectivity");
            _mcUnionConnectivityKernel = _marchingCubes.FindKernel(
                "UnionConnectivity");
            _mcWeldQuantizedPositionsKernel = _marchingCubes.FindKernel(
                "WeldQuantizedPositions");
            _mcCompressConnectivityKernel = _marchingCubes.FindKernel(
                "CompressConnectivity");
            _mcCountComponentsKernel = _marchingCubes.FindKernel(
                "CountComponents");
            _mcFilterComponentsKernel = _marchingCubes.FindKernel(
                "FilterComponents");
            _mcArgsKernel = _marchingCubes.FindKernel(
                "BuildMarchingCubesArgs");
            _mcClearDisplayEdgesKernel = _marchingCubes.FindKernel(
                "ClearDisplayEdgeState");
            _mcClearUniqueEdgesKernel = _marchingCubes.FindKernel(
                "ClearUniqueEdgeState");
            _mcBuildUniqueEdgeArgsKernel = _marchingCubes.FindKernel(
                "BuildUniqueEdgeArgs");
            _mcBuildUniqueEdgesKernel = _marchingCubes.FindKernel(
                "BuildUniqueEdges");
            _mcCompactDisplayEdgesKernel = _marchingCubes.FindKernel(
                "CompactDisplayEdges");
            _mcBuildDisplayEdgeArgsKernel = _marchingCubes.FindKernel(
                "BuildDisplayEdgeArgs");
            _structuralSampleKernel = _structuralReduction.FindKernel(
                "BuildStructuralSamples");

            CreateMarchingCubesBuffers();
            _wireMaterial = new Material(wireShader)
            {
                name = "RoomSetup GPU Marching Cubes Wire",
                enableInstancing = true
            };
            _depthMaterial = new Material(depthShader)
            {
                name = "RoomSetup GPU Marching Cubes Depth",
                enableInstancing = true
            };
            Color white = new Color(1f, 1f, 1f, _lineColor.a);
            _wireMaterial.SetColor(BaseColorId, white);
            _wireMaterial.SetFloat(
                LineWidthPixelsId, Mathf.Clamp(_lineWidthPixels, 0.75f, 3f));
            _wireMaterial.SetFloat(
                UniqueEdgesOnlyId, _uniqueEdgesOnly ? 1f : 0f);
            _drawProperties = new MaterialPropertyBlock();
            _drawProperties.SetBuffer(
                CandidateTriangleVerticesId, _candidateTriangleVertices);
            _drawProperties.SetBuffer(
                FilteredTriangleVerticesId, _filteredTriangleVertices);
            _drawProperties.SetBuffer(
                MarchingCubesCountersId, _mcCounters);
            _drawProperties.SetBuffer(
                CompactDisplayEdgesId, _compactDisplayEdges);
            _drawProperties.SetBuffer(
                CandidateTriangleRevealedId,
                _candidateTriangleRevealed);
            _drawProperties.SetBuffer(
                FilteredTriangleRevealedId,
                _filteredTriangleRevealed);
            _drawProperties.SetColor(BaseColorId, white);
            _drawProperties.SetFloat(
                LineWidthPixelsId, Mathf.Clamp(_lineWidthPixels, 0.75f, 3f));
            _drawProperties.SetFloat(
                UniqueEdgesOnlyId, _uniqueEdgesOnly ? 1f : 0f);
            _drawBounds = new Bounds(Vector3.zero, Vector3.zero);
            _hasDrawBounds = false;

            ConfigureFusionConstants();
            ConfigureMarchingCubesConstants();
            ExtractMarchingCubes();

            _initialized = true;
            Debug.Log(
                "[RoomSetupReplicaV2] GPU production ready: " +
                $"worldTiles={TileSampleCount}^3 step={TileStepMeters:F2}m " +
                $"sourceVoxel={VoxelMeters:F2}m meshVoxel=" +
                $"{VoxelMeters * MeshStride:F2}m " +
                $"hardRange={MaximumDepthMeters:F2}m " +
                "fusion=v0.1.1-weighted gradient=0.1m " +
                "truncation=depth-scaled(+0.20,-0.0667) " +
                "maxWeight=1000 firstWrite=any-valid-eye " +
                "mesh=GPU-MarchingCubes direct-current-TSDF " +
                "display=white-persistent-binocular-view-sweep " +
                "cleanup=global-RemoveSmall<32-if-triangles<40000 " +
                "rangeOrigin=current-eyes " +
                "draw=compacted-instanced-2.25px-edge-quads " +
                "readback=export-or-simplified-room-only");
            return true;
        }
        catch (Exception exception)
        {
            LogFailureOnce("GPU production initialization failed: " + exception);
            Dispose();
            return false;
        }
    }

    private void CreateMarchingCubesBuffers()
    {
        int tableEntries =
            256 * RoomSetupReplicaV2McTables.TriangleEntriesPerIndex;
        int[] edges = new int[tableEntries];
        for (int configuration = 0; configuration < 256; configuration++)
        for (int entry = 0;
             entry < RoomSetupReplicaV2McTables.TriangleEntriesPerIndex;
             entry++)
        {
            edges[configuration *
                  RoomSetupReplicaV2McTables.TriangleEntriesPerIndex + entry] =
                RoomSetupReplicaV2McTables.TriangleEdge(configuration, entry);
        }

        _triangleEdges = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, edges.Length, sizeof(int));
        _triangleEdges.SetData(edges);
        _candidateTriangleVertices = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            MaximumTriangleVertices, sizeof(float) * 3);
        _filteredTriangleVertices = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            ConnectivityVertexCapacity, sizeof(float) * 3);
        _candidateVertexStability = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            MaximumTriangleVertices, sizeof(float));
        _filteredVertexStability = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            ConnectivityVertexCapacity, sizeof(float));
        _candidateTriangleRevealed = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            MaximumTriangleCount, sizeof(uint));
        _filteredTriangleRevealed = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            ConnectivityVertexCapacity / 3, sizeof(uint));
        _connectivityParents = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            ConnectivityVertexCapacity, sizeof(uint));
        _componentTriangleCounts = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            ConnectivityVertexCapacity, sizeof(uint));
        _positionHashOwners = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            PositionHashCapacity, sizeof(uint));
        _mcCounters = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, 9, sizeof(uint));
        _connectivityDispatchArgs = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured |
            GraphicsBuffer.Target.IndirectArguments,
            3, sizeof(uint));
        _drawArgs = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured |
            GraphicsBuffer.Target.IndirectArguments,
            5, sizeof(uint));
        _lineDrawArgs = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured |
            GraphicsBuffer.Target.IndirectArguments,
            5, sizeof(uint));
        _uniqueEdgeHashOwners = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            UniqueEdgeHashCapacity, sizeof(uint));
        _uniqueEdgeMasks = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            MaximumTriangleCount, sizeof(uint));
        _uniqueEdgeDispatchArgs = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured |
            GraphicsBuffer.Target.IndirectArguments,
            3, sizeof(uint));
        _compactDisplayEdges = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            MaximumTriangleVertices, sizeof(uint));
        _structuralSamples = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            MaximumStructuralSamples * 4, sizeof(float) * 4);
        _mcCounters.SetData(new uint[9]);
        _connectivityDispatchArgs.SetData(new uint[] { 0, 1, 1 });
        _drawArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
        _lineDrawArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
        _uniqueEdgeDispatchArgs.SetData(new uint[] { 0, 1, 1 });
    }

    private void ConfigureFusionConstants()
    {
        _fusion.SetInts(
            "_VoxelCount",
            TileSampleCount, TileSampleCount, TileSampleCount);
        _fusion.SetFloat("_VoxelSize", VoxelMeters);
        _fusion.SetFloat("_MinimumDepthMeters", MinimumDepthMeters);
        _fusion.SetFloat("_MaximumDepthMeters", MaximumDepthMeters);
        _fusion.SetFloat(
            "_PositiveTruncationScale", PositiveTruncationScale);
        _fusion.SetFloat(
            "_NegativeTruncationScale", NegativeTruncationScale);
        _fusion.SetFloat(
            "_GradientThresholdSquared", GradientThresholdSquared);
        _fusion.SetFloat("_MaximumWeight", MaximumWeight);
        _fusion.SetFloat(
            "_ProbeStableDeltaMeters", ProbeStableDeltaMeters);
        _fusion.SetFloat(
            "_ProbeUnstableDeltaMeters", ProbeUnstableDeltaMeters);
        _fusion.SetFloat(
            "_ProbeMaximumStability", ProbeMaximumStability);
        _fusion.SetFloat(
            "_ProbeUnstablePenalty", ProbeUnstablePenalty);
        _fusion.SetFloat("_FusionNdcRadius", FusionNdcRadius);
    }

    private void ConfigureMarchingCubesConstants()
    {
        _marchingCubes.SetInts(
            "_VoxelCount",
            TileSampleCount, TileSampleCount, TileSampleCount);
        _marchingCubes.SetInts(
            "_CellCount", TileCellCount, TileCellCount, TileCellCount);
        _marchingCubes.SetFloat("_VoxelSize", VoxelMeters);
        _marchingCubes.SetInt(
            "_MaximumTriangleVertices", MaximumTriangleVertices);
        _marchingCubes.SetInt(
            "_ConnectivityVertexCapacity", ConnectivityVertexCapacity);
        _marchingCubes.SetInt(
            "_PositionHashCapacity", PositionHashCapacity);
        _marchingCubes.SetFloat(
            "_SeamWeldUnitsPerMeter", SeamWeldUnitsPerMeter);
        _marchingCubes.SetInt(
            "_RemoveSmallTriangleThreshold", RemoveSmallTriangleThreshold);
        _marchingCubes.SetInt(
            "_RemoveSmallTriangleGate", RemoveSmallTriangleGate);
        _marchingCubes.SetInt(
            "_MaximumTriangleCount", MaximumTriangleCount);
        _marchingCubes.SetInt(
            "_UniqueEdgeHashCapacity", UniqueEdgeHashCapacity);
        _marchingCubes.SetInt(
            "_CompactDisplayEdgeCapacity", MaximumTriangleVertices);
        _marchingCubes.SetInt(
            "_UniqueEdgesOnly", _uniqueEdgesOnly ? 1 : 0);
        _marchingCubes.SetFloat(
            "_ProbeStableVoteThreshold", ProbeStableVoteThreshold);
        _marchingCubes.SetFloat("_RevealNdcRadius", RevealNdcRadius);
        _marchingCubes.SetFloat(
            "_RevealOcclusionToleranceMeters",
            RevealOcclusionToleranceMeters);
    }

    private void BindFusionVolumes(int kernel, GpuTile tile)
    {
        _fusion.SetTexture(kernel, "_TsdfVolume", tile.tsdf);
        _fusion.SetTexture(
            kernel, "_StabilityProbeVolume", tile.stabilityProbe);
    }

    private void EnsureActiveTiles(
        Vector3 eyeCentre,
        Matrix4x4 leftWorldToClip,
        Matrix4x4 rightWorldToClip)
    {
        _activeTiles.Clear();
        GeometryUtility.CalculateFrustumPlanes(
            leftWorldToClip, _leftFrustumPlanes);
        GeometryUtility.CalculateFrustumPlanes(
            rightWorldToClip, _rightFrustumPlanes);
        float activationRadius =
            MaximumDepthMeters + TileActivationPaddingMeters;
        float activationRadiusSquared =
            activationRadius * activationRadius;
        int minX = Mathf.FloorToInt(
            (eyeCentre.x - activationRadius) / TileStepMeters);
        int minY = Mathf.FloorToInt(
            (eyeCentre.y - activationRadius) / TileStepMeters);
        int minZ = Mathf.FloorToInt(
            (eyeCentre.z - activationRadius) / TileStepMeters);
        int maxX = Mathf.FloorToInt(
            (eyeCentre.x + activationRadius) / TileStepMeters);
        int maxY = Mathf.FloorToInt(
            (eyeCentre.y + activationRadius) / TileStepMeters);
        int maxZ = Mathf.FloorToInt(
            (eyeCentre.z + activationRadius) / TileStepMeters);

        for (int z = minZ; z <= maxZ; z++)
        for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            var key = new Vector3Int(x, y, z);
            Bounds tileBounds = GetTileBounds(key);
            Vector3 closest = tileBounds.ClosestPoint(eyeCentre);
            if ((closest - eyeCentre).sqrMagnitude >
                activationRadiusSquared)
                continue;
            if (!GeometryUtility.TestPlanesAABB(
                    _leftFrustumPlanes, tileBounds) &&
                !GeometryUtility.TestPlanesAABB(
                    _rightFrustumPlanes, tileBounds))
                continue;
            if (!_tiles.TryGetValue(key, out GpuTile tile))
            {
                tile = CreateTile(key, tileBounds);
                _tiles.Add(key, tile);
                ExpandDrawBounds(tileBounds);
            }
            _activeTiles.Add(tile);
        }
    }

    private GpuTile CreateTile(Vector3Int key, Bounds bounds)
    {
        var tile = new GpuTile
        {
            key = key,
            bounds = bounds,
            // VoxelToWorld subtracts half of the sample count. This places
            // coordinate zero exactly at the tile's world-space minimum.
            volumeOrigin = bounds.min + Vector3.one *
                (TileSampleCount * 0.5f * VoxelMeters)
        };
        string suffix = $"[{key.x},{key.y},{key.z}]";
        try
        {
            tile.tsdf = CreateVolume(
                "RoomSetup Tile TSDF " + suffix,
                SupportedLoadStoreFormat(
                    GraphicsFormat.R16G16_SFloat,
                    GraphicsFormat.R32G32_SFloat));
            tile.stabilityProbe = CreateVolume(
                "RoomSetup Tile Stability Probe " + suffix,
                SupportedLoadStoreFormat(
                    GraphicsFormat.R16G16_SFloat,
                    GraphicsFormat.R32G32_SFloat));
            tile.revealedCells = CreateCellVolume(
                "RoomSetup Tile Revealed Cells " + suffix,
                SupportedLoadStoreFormat(
                    GraphicsFormat.R16_SFloat,
                    GraphicsFormat.R32_SFloat));
            BindFusionVolumes(_clearKernel, tile);
            DispatchFusionVolume(_clearKernel);
            _marchingCubes.SetTexture(
                _mcClearRevealKernel,
                "_RevealCellVolume", tile.revealedCells);
            int revealGroups = Mathf.CeilToInt(TileCellCount / 4f);
            _marchingCubes.Dispatch(
                _mcClearRevealKernel,
                revealGroups, revealGroups, revealGroups);
            return tile;
        }
        catch
        {
            ReleaseTile(tile);
            throw;
        }
    }

    private static Bounds GetTileBounds(Vector3Int key)
    {
        Vector3 minimum = new Vector3(
            key.x * TileStepMeters,
            key.y * TileStepMeters,
            key.z * TileStepMeters);
        Vector3 size = Vector3.one * TileStepMeters;
        return new Bounds(minimum + size * 0.5f, size);
    }

    private void ExpandDrawBounds(Bounds tileBounds)
    {
        if (!_hasDrawBounds)
        {
            _drawBounds = tileBounds;
            _hasDrawBounds = true;
            return;
        }
        _drawBounds.Encapsulate(tileBounds.min);
        _drawBounds.Encapsulate(tileBounds.max);
    }

    private void ExtractMarchingCubes()
    {
        _marchingCubes.SetBuffer(
            _mcClearKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcClearKernel, "_ConnectivityDispatchArgs",
            _connectivityDispatchArgs);
        _marchingCubes.SetBuffer(
            _mcClearKernel, "_DrawArgs", _drawArgs);
        _marchingCubes.Dispatch(_mcClearKernel, 1, 1, 1);

        _marchingCubes.SetBuffer(
            _mcEmitKernel, "_TriangleEdges", _triangleEdges);
        _marchingCubes.SetBuffer(
            _mcEmitKernel, "_CandidateTriangleVertices",
            _candidateTriangleVertices);
        _marchingCubes.SetBuffer(
            _mcEmitKernel, "_CandidateVertexStability",
            _candidateVertexStability);
        _marchingCubes.SetBuffer(
            _mcEmitKernel, "_CandidateTriangleRevealed",
            _candidateTriangleRevealed);
        _marchingCubes.SetBuffer(
            _mcEmitKernel, "_Counters", _mcCounters);
        int tileCellGroups = Mathf.CeilToInt(TileCellCount / 4f);
        foreach (GpuTile tile in _tiles.Values)
        {
            _marchingCubes.SetVector("_VolumeOrigin", tile.volumeOrigin);
            _marchingCubes.SetTexture(
                _mcEmitKernel, "_TsdfVolume", tile.tsdf);
            _marchingCubes.SetTexture(
                _mcEmitKernel, "_StabilityProbeVolume",
                tile.stabilityProbe);
            _marchingCubes.SetTexture(
                _mcEmitKernel, "_RevealCellVolume",
                tile.revealedCells);
            _marchingCubes.Dispatch(
                _mcEmitKernel,
                tileCellGroups, tileCellGroups, tileCellGroups);
        }

        _marchingCubes.SetBuffer(
            _mcConnectivityArgsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcConnectivityArgsKernel, "_ConnectivityDispatchArgs",
            _connectivityDispatchArgs);
        _marchingCubes.Dispatch(_mcConnectivityArgsKernel, 1, 1, 1);

        _marchingCubes.SetBuffer(
            _mcInitializeConnectivityKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcInitializeConnectivityKernel, "_ConnectivityParents",
            _connectivityParents);
        _marchingCubes.SetBuffer(
            _mcInitializeConnectivityKernel, "_ComponentTriangleCounts",
            _componentTriangleCounts);
        _marchingCubes.SetBuffer(
            _mcInitializeConnectivityKernel, "_PositionHashOwners",
            _positionHashOwners);
        int connectivityGroups =
            Mathf.CeilToInt(
                Mathf.Max(
                    ConnectivityVertexCapacity, PositionHashCapacity) /
                128f);
        _marchingCubes.Dispatch(
            _mcInitializeConnectivityKernel, connectivityGroups, 1, 1);

        _marchingCubes.SetBuffer(
            _mcUnionConnectivityKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcUnionConnectivityKernel, "_ConnectivityParents",
            _connectivityParents);
        _marchingCubes.DispatchIndirect(
            _mcUnionConnectivityKernel, _connectivityDispatchArgs);

        _marchingCubes.SetBuffer(
            _mcWeldQuantizedPositionsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcWeldQuantizedPositionsKernel,
            "_CandidateTriangleVertices", _candidateTriangleVertices);
        _marchingCubes.SetBuffer(
            _mcWeldQuantizedPositionsKernel, "_ConnectivityParents",
            _connectivityParents);
        _marchingCubes.SetBuffer(
            _mcWeldQuantizedPositionsKernel, "_PositionHashOwners",
            _positionHashOwners);
        _marchingCubes.DispatchIndirect(
            _mcWeldQuantizedPositionsKernel, _connectivityDispatchArgs);

        _marchingCubes.SetBuffer(
            _mcCompressConnectivityKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcCompressConnectivityKernel, "_ConnectivityParents",
            _connectivityParents);
        _marchingCubes.Dispatch(
            _mcCompressConnectivityKernel, connectivityGroups, 1, 1);

        _marchingCubes.SetBuffer(
            _mcCountComponentsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcCountComponentsKernel, "_ConnectivityParents",
            _connectivityParents);
        _marchingCubes.SetBuffer(
            _mcCountComponentsKernel, "_ComponentTriangleCounts",
            _componentTriangleCounts);
        _marchingCubes.DispatchIndirect(
            _mcCountComponentsKernel, _connectivityDispatchArgs);

        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_CandidateTriangleVertices",
            _candidateTriangleVertices);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_FilteredTriangleVertices",
            _filteredTriangleVertices);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_CandidateVertexStability",
            _candidateVertexStability);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_FilteredVertexStability",
            _filteredVertexStability);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_CandidateTriangleRevealed",
            _candidateTriangleRevealed);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_FilteredTriangleRevealed",
            _filteredTriangleRevealed);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_ConnectivityParents",
            _connectivityParents);
        _marchingCubes.SetBuffer(
            _mcFilterComponentsKernel, "_ComponentTriangleCounts",
            _componentTriangleCounts);
        _marchingCubes.DispatchIndirect(
            _mcFilterComponentsKernel, _connectivityDispatchArgs);

        _marchingCubes.SetBuffer(
            _mcArgsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcArgsKernel, "_DrawArgs", _drawArgs);
        _marchingCubes.SetBuffer(
            _mcArgsKernel, "_LineDrawArgs", _lineDrawArgs);
        _marchingCubes.Dispatch(_mcArgsKernel, 1, 1, 1);
        BuildUniqueEdgeVisibility();
    }

    private void BuildUniqueEdgeVisibility()
    {
        if (_marchingCubes == null ||
            _uniqueEdgeHashOwners == null || _uniqueEdgeMasks == null ||
            _uniqueEdgeDispatchArgs == null ||
            _compactDisplayEdges == null || _lineDrawArgs == null)
            return;

        _marchingCubes.SetBuffer(
            _mcClearDisplayEdgesKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcClearDisplayEdgesKernel, "_LineDrawArgs", _lineDrawArgs);
        _marchingCubes.Dispatch(_mcClearDisplayEdgesKernel, 1, 1, 1);

        _marchingCubes.SetBuffer(
            _mcBuildUniqueEdgeArgsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcBuildUniqueEdgeArgsKernel,
            "_UniqueEdgeDispatchArgs", _uniqueEdgeDispatchArgs);
        _marchingCubes.Dispatch(
            _mcBuildUniqueEdgeArgsKernel, 1, 1, 1);

        if (_uniqueEdgesOnly)
        {
            _marchingCubes.SetBuffer(
                _mcClearUniqueEdgesKernel,
                "_UniqueEdgeHashOwners", _uniqueEdgeHashOwners);
            _marchingCubes.SetBuffer(
                _mcClearUniqueEdgesKernel,
                "_UniqueEdgeMasks", _uniqueEdgeMasks);
            int clearGroups = Mathf.CeilToInt(
                Mathf.Max(UniqueEdgeHashCapacity, MaximumTriangleCount) /
                128f);
            _marchingCubes.Dispatch(
                _mcClearUniqueEdgesKernel, clearGroups, 1, 1);

            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel, "_Counters", _mcCounters);
            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel,
                "_CandidateTriangleVertices", _candidateTriangleVertices);
            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel,
                "_FilteredTriangleVertices", _filteredTriangleVertices);
            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel,
                "_UniqueEdgeHashOwners", _uniqueEdgeHashOwners);
            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel,
                "_UniqueEdgeMasks", _uniqueEdgeMasks);
            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel,
                "_CandidateTriangleRevealed",
                _candidateTriangleRevealed);
            _marchingCubes.SetBuffer(
                _mcBuildUniqueEdgesKernel,
                "_FilteredTriangleRevealed",
                _filteredTriangleRevealed);
            _marchingCubes.DispatchIndirect(
                _mcBuildUniqueEdgesKernel, _uniqueEdgeDispatchArgs);
        }

        _marchingCubes.SetInt(
            "_UniqueEdgesOnly", _uniqueEdgesOnly ? 1 : 0);
        _marchingCubes.SetBuffer(
            _mcCompactDisplayEdgesKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcCompactDisplayEdgesKernel,
            "_UniqueEdgeMasks", _uniqueEdgeMasks);
        _marchingCubes.SetBuffer(
            _mcCompactDisplayEdgesKernel,
            "_CompactDisplayEdges", _compactDisplayEdges);
        _marchingCubes.SetBuffer(
            _mcCompactDisplayEdgesKernel,
            "_CandidateTriangleRevealed",
            _candidateTriangleRevealed);
        _marchingCubes.SetBuffer(
            _mcCompactDisplayEdgesKernel,
            "_FilteredTriangleRevealed",
            _filteredTriangleRevealed);
        _marchingCubes.DispatchIndirect(
            _mcCompactDisplayEdgesKernel, _uniqueEdgeDispatchArgs);

        _marchingCubes.SetBuffer(
            _mcBuildDisplayEdgeArgsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcBuildDisplayEdgeArgsKernel,
            "_LineDrawArgs", _lineDrawArgs);
        _marchingCubes.Dispatch(_mcBuildDisplayEdgeArgsKernel, 1, 1, 1);
    }

    private void DispatchFusionVolume(int kernel)
    {
        _fusion.Dispatch(
            kernel,
            Mathf.CeilToInt(TileSampleCount / 4f),
            Mathf.CeilToInt(TileSampleCount / 4f),
            Mathf.CeilToInt(TileSampleCount / 4f));
    }

    private static RenderTexture CreateVolume(
        string name, GraphicsFormat format)
    {
        var descriptor = new RenderTextureDescriptor(
            TileSampleCount, TileSampleCount, format, 0)
        {
            dimension = TextureDimension.Tex3D,
            volumeDepth = TileSampleCount,
            enableRandomWrite = true,
            msaaSamples = 1,
            useMipMap = false,
            autoGenerateMips = false
        };
        var texture = new RenderTexture(descriptor)
        {
            name = name,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        texture.Create();
        if (!texture.IsCreated())
        {
            Object.Destroy(texture);
            throw new InvalidOperationException(
                "Could not create GPU volume " + name + " (" + format + ")");
        }
        return texture;
    }

    private static RenderTexture CreateCellVolume(
        string name, GraphicsFormat format)
    {
        var descriptor = new RenderTextureDescriptor(
            TileCellCount, TileCellCount, format, 0)
        {
            dimension = TextureDimension.Tex3D,
            volumeDepth = TileCellCount,
            enableRandomWrite = true,
            msaaSamples = 1,
            useMipMap = false,
            autoGenerateMips = false
        };
        var texture = new RenderTexture(descriptor)
        {
            name = name,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        texture.Create();
        if (!texture.IsCreated())
        {
            Object.Destroy(texture);
            throw new InvalidOperationException(
                "Could not create GPU cell volume " + name +
                " (" + format + ")");
        }
        return texture;
    }

    private static GraphicsFormat SupportedLoadStoreFormat(
        GraphicsFormat preferred, GraphicsFormat fallback)
    {
        return SystemInfo.IsFormatSupported(preferred, FormatUsage.LoadStore)
            ? preferred
            : fallback;
    }

    private static bool IsFinite(Vector4 value)
    {
        return float.IsFinite(value.x) && float.IsFinite(value.y) &&
               float.IsFinite(value.z) && float.IsFinite(value.w);
    }

    private static bool IsFinite(Matrix4x4 value)
    {
        for (int row = 0; row < 4; row++)
        for (int column = 0; column < 4; column++)
        {
            if (!float.IsFinite(value[row, column]))
                return false;
        }
        return true;
    }

    private void LogFailureOnce(string message)
    {
        if (_failureLogged)
            return;
        _failureLogged = true;
        Debug.LogError("[RoomSetupReplicaV2] " + message);
    }

    public void Dispose()
    {
        _initialized = false;
        _integrationCount = 0;
        ReleaseAllTiles();
        Release(ref _triangleEdges);
        Release(ref _candidateTriangleVertices);
        Release(ref _filteredTriangleVertices);
        Release(ref _candidateVertexStability);
        Release(ref _filteredVertexStability);
        Release(ref _candidateTriangleRevealed);
        Release(ref _filteredTriangleRevealed);
        Release(ref _connectivityParents);
        Release(ref _componentTriangleCounts);
        Release(ref _positionHashOwners);
        Release(ref _mcCounters);
        Release(ref _connectivityDispatchArgs);
        Release(ref _drawArgs);
        Release(ref _lineDrawArgs);
        Release(ref _uniqueEdgeHashOwners);
        Release(ref _uniqueEdgeMasks);
        Release(ref _uniqueEdgeDispatchArgs);
        Release(ref _compactDisplayEdges);
        Release(ref _structuralSamples);
        if (_wireMaterial != null)
            Object.Destroy(_wireMaterial);
        if (_depthMaterial != null)
            Object.Destroy(_depthMaterial);
        _wireMaterial = null;
        _depthMaterial = null;
        _drawProperties = null;
        _fusion = null;
        _marchingCubes = null;
        _structuralReduction = null;
    }

    private void ReleaseAllTiles()
    {
        foreach (GpuTile tile in _tiles.Values)
            ReleaseTile(tile);
        _tiles.Clear();
        _activeTiles.Clear();
        _drawBounds = new Bounds(Vector3.zero, Vector3.zero);
        _hasDrawBounds = false;
    }

    private static void ReleaseTile(GpuTile tile)
    {
        if (tile == null)
            return;
        Release(ref tile.tsdf);
        Release(ref tile.stabilityProbe);
        Release(ref tile.revealedCells);
    }

    private static void Release(ref RenderTexture texture)
    {
        if (texture == null)
            return;
        if (texture.IsCreated())
            texture.Release();
        Object.Destroy(texture);
        texture = null;
    }

    private static void Release(ref GraphicsBuffer buffer)
    {
        buffer?.Release();
        buffer = null;
    }
}

using System;
using System.Collections.Generic;
using Genesis.RoomScan;
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
    private const int RemoveSmallTriangleThreshold = 32;
    private const int RemoveSmallTriangleGate = 40000;
    private const int ConnectivityVertexCapacity =
        RemoveSmallTriangleGate * 3;
    private const int PositionHashCapacity = 262144;
    private const float SeamWeldUnitsPerMeter = 10000f;
    private const int MeshEveryIntegrations = 2;

    private static readonly int CandidateTriangleVerticesId =
        Shader.PropertyToID("_RoomSetupMcCandidateVertices");
    private static readonly int FilteredTriangleVerticesId =
        Shader.PropertyToID("_RoomSetupMcFilteredVertices");
    private static readonly int MarchingCubesCountersId =
        Shader.PropertyToID("_RoomSetupMcCounters");
    private static readonly int UniqueEdgeMasksId =
        Shader.PropertyToID("_RoomSetupMcUniqueEdgeMasks");
    private static readonly int BaseColorId =
        Shader.PropertyToID("_BaseColor");
    private static readonly int LineWidthPixelsId =
        Shader.PropertyToID("_LineWidthPixels");
    private static readonly int UniqueEdgesOnlyId =
        Shader.PropertyToID("_UniqueEdgesOnly");
    private readonly GameObject _owner;
    private readonly Color _lineColor;
    private readonly float _lineWidthPixels;
    private readonly Matrix4x4[] _projection = new Matrix4x4[2];
    private readonly Matrix4x4[] _projectionInverse = new Matrix4x4[2];
    private readonly Matrix4x4[] _view = new Matrix4x4[2];
    private readonly Matrix4x4[] _viewInverse = new Matrix4x4[2];
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
    }

    private ComputeShader _fusion;
    private ComputeShader _marchingCubes;
    private Material _wireMaterial;
    private Material _depthMaterial;
    private MaterialPropertyBlock _drawProperties;
    private int _clearKernel = -1;
    private int _integrateKernel = -1;
    private int _integrateEnvironmentKernel = -1;
    private int _mcClearKernel = -1;
    private int _mcEmitKernel = -1;
    private int _mcConnectivityArgsKernel = -1;
    private int _mcInitializeConnectivityKernel = -1;
    private int _mcUnionConnectivityKernel = -1;
    private int _mcWeldQuantizedPositionsKernel = -1;
    private int _mcCompressConnectivityKernel = -1;
    private int _mcCountComponentsKernel = -1;
    private int _mcFilterComponentsKernel = -1;
    private int _mcArgsKernel = -1;
    private int _mcClearUniqueEdgesKernel = -1;
    private int _mcBuildUniqueEdgeArgsKernel = -1;
    private int _mcBuildUniqueEdgesKernel = -1;
    private GraphicsBuffer _triangleEdges;
    private GraphicsBuffer _candidateTriangleVertices;
    private GraphicsBuffer _filteredTriangleVertices;
    private GraphicsBuffer _connectivityParents;
    private GraphicsBuffer _componentTriangleCounts;
    private GraphicsBuffer _positionHashOwners;
    private GraphicsBuffer _mcCounters;
    private GraphicsBuffer _connectivityDispatchArgs;
    private GraphicsBuffer _drawArgs;
    private GraphicsBuffer _uniqueEdgeHashOwners;
    private GraphicsBuffer _uniqueEdgeMasks;
    private GraphicsBuffer _uniqueEdgeDispatchArgs;
    private Bounds _drawBounds;
    private bool _hasDrawBounds;
    private bool _initialized;
    private bool _visible = true;
    private bool _failureLogged;
    private bool _uniqueEdgesOnly;
    private int _integrationCount;

    public int IntegrationCount => _integrationCount;
    public int TileCount => _tiles.Count;
    public bool Ready => _initialized;
    public bool UniqueEdgesOnly => _uniqueEdgesOnly;

    public RoomSetupReplicaV2GpuBackend(
        GameObject owner, Color lineColor, float lineWidthPixels)
    {
        _owner = owner;
        _lineColor = lineColor;
        _lineWidthPixels = lineWidthPixels;
    }

    public bool TryObserveDepthFrame(DepthCapture depth)
    {
        if (depth == null || !DepthCapture.DepthAvailable)
            return false;
        Texture source = depth.ChiselSourceDepthTex;
        if (source == null || depth.Proj == null || depth.Proj.Length < 2 ||
            depth.ProjInv == null || depth.ProjInv.Length < 2 ||
            depth.View == null || depth.View.Length < 2 ||
            depth.ViewInv == null || depth.ViewInv.Length < 2)
            return false;

        if (!EnsureInitialized())
            return false;

        for (int eye = 0; eye < 2; eye++)
        {
            _projection[eye] = depth.Proj[eye];
            _projectionInverse[eye] = depth.ProjInv[eye];
            _view[eye] = depth.View[eye];
            _viewInverse[eye] = depth.ViewInv[eye];
        }

        Vector3 leftEye = _viewInverse[0].MultiplyPoint3x4(Vector3.zero);
        Vector3 rightEye = _viewInverse[1].MultiplyPoint3x4(Vector3.zero);
        EnsureActiveTiles(
            (leftEye + rightEye) * 0.5f,
            _projection[0] * _view[0],
            _projection[1] * _view[1]);
        _fusion.SetTexture(_integrateKernel, "_DepthTex", source);
        _fusion.SetInts("_DepthSize", source.width, source.height);
        _fusion.SetMatrixArray("_Projection", _projection);
        _fusion.SetMatrixArray("_ProjectionInverse", _projectionInverse);
        _fusion.SetMatrixArray("_View", _view);
        _fusion.SetMatrixArray("_ViewInverse", _viewInverse);
        for (int tileIndex = 0;
             tileIndex < _activeTiles.Count; tileIndex++)
        {
            GpuTile tile = _activeTiles[tileIndex];
            _fusion.SetVector("_VolumeOrigin", tile.volumeOrigin);
            BindFusionVolumes(_integrateKernel, tile);
            DispatchFusionVolume(_integrateKernel);
        }
        _integrationCount++;

        if (_integrationCount == 1 ||
            _integrationCount % MeshEveryIntegrations == 0)
        {
            ExtractMarchingCubes();
        }
        return true;
    }

    /// <summary>
    /// Integrates the globally published Meta Environment Depth frame. This
    /// entry point is used only by the separate sequential-B scene, where the
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
            wireParams, MeshTopology.Triangles, _drawArgs, 1);
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
        if (enabled && _initialized)
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
            Shader wireShader = Resources.Load<Shader>(
                "RoomSetupReplicaV2GpuMarchingCubes");
            Shader depthShader = Resources.Load<Shader>(
                "RoomSetupReplicaV2GpuMarchingCubesDepth");
            if (_fusion == null || _marchingCubes == null ||
                wireShader == null || depthShader == null)
            {
                throw new InvalidOperationException(
                    "Dedicated GPU Marching Cubes resources are unavailable.");
            }

            _clearKernel = _fusion.FindKernel("ClearVolume");
            _integrateKernel = _fusion.FindKernel("IntegrateStereo");
            _integrateEnvironmentKernel = _fusion.FindKernel(
                "IntegrateEnvironmentDepth");
            _mcClearKernel = _marchingCubes.FindKernel(
                "ClearMarchingCubes");
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
            _mcClearUniqueEdgesKernel = _marchingCubes.FindKernel(
                "ClearUniqueEdgeState");
            _mcBuildUniqueEdgeArgsKernel = _marchingCubes.FindKernel(
                "BuildUniqueEdgeArgs");
            _mcBuildUniqueEdgesKernel = _marchingCubes.FindKernel(
                "BuildUniqueEdges");

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
            _wireMaterial.SetColor(BaseColorId, _lineColor);
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
                UniqueEdgeMasksId, _uniqueEdgeMasks);
            _drawProperties.SetColor(BaseColorId, _lineColor);
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
                "cleanup=global-RemoveSmall<32-if-triangles<40000 " +
                "rangeOrigin=current-eyes draw=indirect cpuReadback=none");
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
            256 * ScanCoverPaperDmcTables.TriangleEntriesPerIndex;
        int[] edges = new int[tableEntries];
        for (int configuration = 0; configuration < 256; configuration++)
        for (int entry = 0;
             entry < ScanCoverPaperDmcTables.TriangleEntriesPerIndex;
             entry++)
        {
            edges[configuration *
                  ScanCoverPaperDmcTables.TriangleEntriesPerIndex + entry] =
                ScanCoverPaperDmcTables.TriangleEdge(configuration, entry);
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
            GraphicsBuffer.Target.Structured, 7, sizeof(uint));
        _connectivityDispatchArgs = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured |
            GraphicsBuffer.Target.IndirectArguments,
            3, sizeof(uint));
        _drawArgs = new GraphicsBuffer(
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
        _mcCounters.SetData(new uint[7]);
        _connectivityDispatchArgs.SetData(new uint[] { 0, 1, 1 });
        _drawArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
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
    }

    private void BindFusionVolumes(int kernel, GpuTile tile)
    {
        _fusion.SetTexture(kernel, "_TsdfVolume", tile.tsdf);
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
            BindFusionVolumes(_clearKernel, tile);
            DispatchFusionVolume(_clearKernel);
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
            _mcEmitKernel, "_Counters", _mcCounters);
        int tileCellGroups = Mathf.CeilToInt(TileCellCount / 4f);
        foreach (GpuTile tile in _tiles.Values)
        {
            _marchingCubes.SetVector("_VolumeOrigin", tile.volumeOrigin);
            _marchingCubes.SetTexture(
                _mcEmitKernel, "_TsdfVolume", tile.tsdf);
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
        _marchingCubes.Dispatch(_mcArgsKernel, 1, 1, 1);
        if (_uniqueEdgesOnly)
            BuildUniqueEdgeVisibility();
    }

    private void BuildUniqueEdgeVisibility()
    {
        if (_marchingCubes == null ||
            _uniqueEdgeHashOwners == null || _uniqueEdgeMasks == null ||
            _uniqueEdgeDispatchArgs == null)
            return;

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
            _mcBuildUniqueEdgeArgsKernel, "_Counters", _mcCounters);
        _marchingCubes.SetBuffer(
            _mcBuildUniqueEdgeArgsKernel,
            "_UniqueEdgeDispatchArgs", _uniqueEdgeDispatchArgs);
        _marchingCubes.Dispatch(
            _mcBuildUniqueEdgeArgsKernel, 1, 1, 1);

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
        _marchingCubes.DispatchIndirect(
            _mcBuildUniqueEdgesKernel, _uniqueEdgeDispatchArgs);
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
        Release(ref _connectivityParents);
        Release(ref _componentTriangleCounts);
        Release(ref _positionHashOwners);
        Release(ref _mcCounters);
        Release(ref _connectivityDispatchArgs);
        Release(ref _drawArgs);
        Release(ref _uniqueEdgeHashOwners);
        Release(ref _uniqueEdgeMasks);
        Release(ref _uniqueEdgeDispatchArgs);
        if (_wireMaterial != null)
            Object.Destroy(_wireMaterial);
        if (_depthMaterial != null)
            Object.Destroy(_depthMaterial);
        _wireMaterial = null;
        _depthMaterial = null;
        _drawProperties = null;
        _fusion = null;
        _marchingCubes = null;
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

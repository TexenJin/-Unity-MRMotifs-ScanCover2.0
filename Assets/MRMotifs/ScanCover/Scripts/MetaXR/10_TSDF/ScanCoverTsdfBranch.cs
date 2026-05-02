using Meta.XR.EnvironmentDepth;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverTsdfBranch : MonoBehaviour
{
    [Header("Refs")]
    public Transform referenceFrame;
    public ScanCoverSkeletonBuilder_A builder;
    public EnvironmentDepthManager environmentDepthManager;
    public ScanCoverDepthPreprocessor depthPreprocessor;

    [Header("Source")]
    public ScanCoverDepthPreprocessor.SourceEye preferredEye = ScanCoverDepthPreprocessor.SourceEye.Right;
    public bool autoResolveRefs = true;
    public bool integrateWhileScanning = true;
    [Min(0.05f)] public float integrateIntervalSeconds = 0.20f;

    [Header("TSDF Volume")]
    [Min(0.03f)] public float voxelSizeMeters = 0.10f;
    [Min(16)] public int volumeSizeX = 96;
    [Min(16)] public int volumeSizeY = 48;
    [Min(16)] public int volumeSizeZ = 96;
    public Vector3 volumeCenterLocal = new Vector3(0f, 1.4f, 0f);
    [Min(0.05f)] public float truncationPositiveMeters = 0.35f;
    public float truncationNegativeMeters = -0.15f;
    [Min(1f)] public float maxIntegratedWeight = 24f;

    [Header("Observation Filter")]
    [Range(0f, 1f)] public float minObservationConfidence = 0.18f;
    [Range(0f, 1f)] public float minNormalFacingDot = 0.20f;
    [Min(0f)] public float minDepthMeters = 0.35f;
    [Min(0f)] public float maxDepthMeters = 6f;

    [Header("Surface Output")]
    public bool renderDebugSurfaceObject = false;
    public bool hideSurfaceWhileScanning = true;
    public bool showSurfaceWhenFrozen = true;
    public bool buildCollider = false;
    [Min(0)] public int minTrianglesToShow = 24;
    [Range(0.01f, 4f)] public float minObservedWeightForMeshing = 0.5f;
    public Color surfaceColor = new Color(0.18f, 0.58f, 1.0f, 0.85f);

    [Header("Voxel Shell Supplement")]
    public bool mergeVoxelShell = true;
    [Min(1)] public int voxelShellMinHits = 2;
    [Min(0f)] public float voxelShellSkipIfTsdfWeightAtLeast = 1.5f;
    [Min(0f)] public float voxelShellInflateMeters = 0.002f;

    [Header("Debug")]
    public bool debugLog = false;

    public string LastIssue { get; private set; }
    public int TriangleCount { get; private set; }
    public int IntegrationCount { get; private set; }
    public bool HasPendingReadback => _hasPendingReadback;
    public bool HasSurfaceSnapshot => _surfaceTriangles.Count > 0;
    public event System.Action<ScanCoverTsdfBranch> SurfaceDataUpdated;

    private static readonly int ReprojectionMatricesId = Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");

    private static readonly int SourceWorldPositionTextureId = Shader.PropertyToID("_SourceWorldPositionTexture");
    private static readonly int SourceWorldNormalTextureId = Shader.PropertyToID("_SourceWorldNormalTexture");
    private static readonly int SourceObservationMetaTextureId = Shader.PropertyToID("_SourceObservationMetaTexture");
    private static readonly int TsdfVolumeId = Shader.PropertyToID("_TsdfVolume");
    private static readonly int WeightVolumeId = Shader.PropertyToID("_WeightVolume");
    private static readonly int TsdfReadbackBufferId = Shader.PropertyToID("_TsdfReadbackBuffer");
    private static readonly int WeightReadbackBufferId = Shader.PropertyToID("_WeightReadbackBuffer");
    private static readonly int SourceSizeId = Shader.PropertyToID("_SourceSize");
    private static readonly int VolumeSizeId = Shader.PropertyToID("_VolumeSize");
    private static readonly int VolumeVoxelCountId = Shader.PropertyToID("_VolumeVoxelCount");
    private static readonly int VoxelSizeId = Shader.PropertyToID("_VoxelSize");
    private static readonly int TruncationPositiveId = Shader.PropertyToID("_TruncationPositiveMeters");
    private static readonly int TruncationNegativeId = Shader.PropertyToID("_TruncationNegativeMeters");
    private static readonly int MinObservationConfidenceId = Shader.PropertyToID("_MinObservationConfidence");
    private static readonly int MinNormalFacingDotId = Shader.PropertyToID("_MinNormalFacingDot");
    private static readonly int MinDepthMetersId = Shader.PropertyToID("_MinDepthMeters");
    private static readonly int MaxDepthMetersId = Shader.PropertyToID("_MaxDepthMeters");
    private static readonly int MaxIntegratedWeightId = Shader.PropertyToID("_MaxIntegratedWeight");
    private static readonly int CameraWorldPositionId = Shader.PropertyToID("_CameraWorldPosition");
    private static readonly int WorldToClipId = Shader.PropertyToID("_WorldToClip");
    private static readonly int VolumeLocalToWorldId = Shader.PropertyToID("_VolumeLocalToWorld");

    private ComputeShader _computeShader;
    private int _clearKernel = -1;
    private int _integrateKernel = -1;
    private int _copyVolumeToBufferKernel = -1;
    private RenderTexture _tsdfVolume;
    private RenderTexture _weightVolume;
    private ComputeBuffer _tsdfReadbackBuffer;
    private ComputeBuffer _weightReadbackBuffer;
    private GameObject _surfaceRoot;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private MeshCollider _meshCollider;
    private Mesh _mesh;
    private Material _runtimeMaterial;
    private float _nextIntegrateTime;
    private bool _hasPendingReadback;
    private AsyncGPUReadbackRequest _tsdfReadbackRequest;
    private AsyncGPUReadbackRequest _weightReadbackRequest;
    private Vector3Int _pendingVolumeSize;
    private readonly List<ScanCoverSkeletonBuilder_A.CellInfo> _builderSnapshot = new List<ScanCoverSkeletonBuilder_A.CellInfo>(16384);
    private readonly HashSet<ScanCoverSkeletonBuilder_A.VoxelKey> _confirmedVoxels = new HashSet<ScanCoverSkeletonBuilder_A.VoxelKey>();
    private readonly List<Vector3> _combinedVertices = new List<Vector3>(65536);
    private readonly List<int> _combinedTriangles = new List<int>(131072);
    private readonly List<Vector3> _surfaceVertices = new List<Vector3>(65536);
    private readonly List<int> _surfaceTriangles = new List<int>(131072);
    private float[] _latestTsdfData;
    private float[] _latestWeightData;
    private Vector3Int _latestVolumeSize;

    private void Awake()
    {
        ResolveRefs();
        EnsureShader();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureShader();
    }

    private void Update()
    {
        if (_hasPendingReadback)
            UpdatePendingReadback();

        UpdateSurfaceVisibility();

        if (!integrateWhileScanning || builder == null || !builder.scanEnabled)
            return;

        if (Time.time < _nextIntegrateTime)
            return;

        _nextIntegrateTime = Time.time + Mathf.Max(0.05f, integrateIntervalSeconds);
        IntegrateNow();
    }

    private void OnDisable()
    {
        _hasPendingReadback = false;
    }

    private void OnDestroy()
    {
        ReleaseVolumes();
        ReleaseSurface();
    }

    public void EnsureInitialized()
    {
        ResolveRefs();
        EnsureShader();
        EnsureVolumes();
        EnsureSurfaceResources();
    }

    [ContextMenu("TSDF Integrate Now")]
    public bool IntegrateNow()
    {
        ResolveRefs();
        EnsureShader();
        if (!EnsureVolumes())
            return false;

        if (depthPreprocessor == null)
            return SetIssue("ScanCoverDepthPreprocessor was not found.");

        if (!depthPreprocessor.RefreshNow())
            return SetIssue(depthPreprocessor.LastIssue ?? "Depth preprocessor refresh failed.");

        if (!depthPreprocessor.TryGetOutputs(out RenderTexture worldPosition, out RenderTexture worldNormal, out RenderTexture observationMeta))
            return SetIssue(depthPreprocessor.LastIssue ?? "Depth preprocessor outputs are unavailable.");

        Matrix4x4[] reprojectionMatrices = Shader.GetGlobalMatrixArray(ReprojectionMatricesId);
        if (reprojectionMatrices == null || reprojectionMatrices.Length < 2)
            return SetIssue("_EnvironmentDepthReprojectionMatrices is unavailable.");

        int eyeIndex = Mathf.Clamp((int)depthPreprocessor.CurrentSourceEye, 0, 1);
        Matrix4x4 worldToClip = reprojectionMatrices[eyeIndex];

        Transform baseFrame = referenceFrame != null ? referenceFrame : transform;
        Matrix4x4 volumeLocalToWorld = baseFrame.localToWorldMatrix * Matrix4x4.TRS(volumeCenterLocal, Quaternion.identity, Vector3.one);
        Camera mainCamera = Camera.main;
        Vector3 cameraWorldPosition = mainCamera ? mainCamera.transform.position : baseFrame.position;

        _computeShader.SetTexture(_integrateKernel, SourceWorldPositionTextureId, worldPosition);
        _computeShader.SetTexture(_integrateKernel, SourceWorldNormalTextureId, worldNormal);
        _computeShader.SetTexture(_integrateKernel, SourceObservationMetaTextureId, observationMeta);
        _computeShader.SetTexture(_integrateKernel, TsdfVolumeId, _tsdfVolume);
        _computeShader.SetTexture(_integrateKernel, WeightVolumeId, _weightVolume);
        _computeShader.SetVector(SourceSizeId, new Vector4(worldPosition.width, worldPosition.height, 0f, 0f));
        _computeShader.SetInts(VolumeSizeId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _computeShader.SetFloat(VoxelSizeId, voxelSizeMeters);
        _computeShader.SetFloat(TruncationPositiveId, Mathf.Max(0.05f, truncationPositiveMeters));
        _computeShader.SetFloat(TruncationNegativeId, Mathf.Min(-0.001f, truncationNegativeMeters));
        _computeShader.SetFloat(MinObservationConfidenceId, minObservationConfidence);
        _computeShader.SetFloat(MinNormalFacingDotId, minNormalFacingDot);
        _computeShader.SetFloat(MinDepthMetersId, minDepthMeters);
        _computeShader.SetFloat(MaxDepthMetersId, Mathf.Max(minDepthMeters + 0.01f, maxDepthMeters));
        _computeShader.SetFloat(MaxIntegratedWeightId, Mathf.Max(1f, maxIntegratedWeight));
        _computeShader.SetVector(CameraWorldPositionId, new Vector4(cameraWorldPosition.x, cameraWorldPosition.y, cameraWorldPosition.z, 1f));
        _computeShader.SetMatrix(WorldToClipId, worldToClip);
        _computeShader.SetMatrix(VolumeLocalToWorldId, volumeLocalToWorld);

        DispatchVolumeKernel(_integrateKernel);

        IntegrationCount++;
        LastIssue = null;

        if (debugLog)
            Debug.Log($"[ScanCoverTsdfBranch] Integrate #{IntegrationCount}, eye={depthPreprocessor.CurrentSourceEye}, source={worldPosition.width}x{worldPosition.height}");

        return true;
    }

    [ContextMenu("Build TSDF Surface Now")]
    public bool BuildSurfaceNow()
    {
        EnsureInitialized();
        if (_hasPendingReadback)
            return false;

        if (_tsdfVolume == null || _weightVolume == null)
            return SetIssue("TSDF volume is not initialized.");

        if (!IntegrateNow())
            return false;

        if (!EnsureReadbackBuffers())
            return false;

        _pendingVolumeSize = new Vector3Int(volumeSizeX, volumeSizeY, volumeSizeZ);
        int voxelCount = GetExpectedVoxelCount();
        _computeShader.SetInts(VolumeSizeId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _computeShader.SetInt(VolumeVoxelCountId, voxelCount);
        _computeShader.SetTexture(_copyVolumeToBufferKernel, TsdfVolumeId, _tsdfVolume);
        _computeShader.SetTexture(_copyVolumeToBufferKernel, WeightVolumeId, _weightVolume);
        _computeShader.SetBuffer(_copyVolumeToBufferKernel, TsdfReadbackBufferId, _tsdfReadbackBuffer);
        _computeShader.SetBuffer(_copyVolumeToBufferKernel, WeightReadbackBufferId, _weightReadbackBuffer);
        DispatchLinearKernel(_copyVolumeToBufferKernel, voxelCount);

        _tsdfReadbackRequest = AsyncGPUReadback.Request(_tsdfReadbackBuffer);
        _weightReadbackRequest = AsyncGPUReadback.Request(_weightReadbackBuffer);
        _hasPendingReadback = true;
        LastIssue = null;

        if (debugLog)
            Debug.Log("[ScanCoverTsdfBranch] Requested TSDF readback.");

        return true;
    }

    public void ClearAll()
    {
        EnsureShader();
        EnsureVolumes();
        ClearVolumes();
        IntegrationCount = 0;
        TriangleCount = 0;
        LastIssue = null;
        _hasPendingReadback = false;
        if (_mesh != null)
            _mesh.Clear();
        if (_meshCollider != null)
            _meshCollider.sharedMesh = null;
        _surfaceVertices.Clear();
        _surfaceTriangles.Clear();
        _latestTsdfData = null;
        _latestWeightData = null;
        _latestVolumeSize = default;
        UpdateSurfaceVisibility();
        SurfaceDataUpdated?.Invoke(this);
    }

    private void UpdatePendingReadback()
    {
        if (!_tsdfReadbackRequest.done || !_weightReadbackRequest.done)
            return;

        _hasPendingReadback = false;

        if (_tsdfReadbackRequest.hasError || _weightReadbackRequest.hasError)
        {
            SetIssue("AsyncGPUReadback failed for TSDF surface.");
            return;
        }

        float[] tsdf = _tsdfReadbackRequest.GetData<float>().ToArray();
        float[] weights = _weightReadbackRequest.GetData<float>().ToArray();
        BuildMeshFromData(tsdf, weights, _pendingVolumeSize);
    }

    private void BuildMeshFromData(float[] tsdf, float[] weights, Vector3Int volumeSize)
    {
        EnsureSurfaceResources();
        if (_mesh == null)
        {
            SetIssue("TSDF mesh is missing.");
            return;
        }

        bool hasSurface;
        int triangles;
        string mesherIssue;
        try
        {
            hasSurface = ScanCoverTsdfMesherUtil.BuildMesh(
                tsdf,
                weights,
                volumeSize,
                voxelSizeMeters,
                Mathf.Max(0.01f, minObservedWeightForMeshing),
                _mesh,
                out triangles,
                out mesherIssue);
        }
        catch (System.Exception ex)
        {
            TriangleCount = 0;
            _mesh.Clear();
            _surfaceVertices.Clear();
            _surfaceTriangles.Clear();
            SetSurfaceVisible(false);
            SetIssue($"TSDF mesher failed: {ex.Message}");
            SurfaceDataUpdated?.Invoke(this);
            return;
        }

        if (!string.IsNullOrEmpty(mesherIssue))
        {
            TriangleCount = 0;
            _mesh.Clear();
            _surfaceVertices.Clear();
            _surfaceTriangles.Clear();
            SetSurfaceVisible(false);
            SetIssue(mesherIssue);
            SurfaceDataUpdated?.Invoke(this);
            return;
        }

        _latestTsdfData = tsdf;
        _latestWeightData = weights;
        _latestVolumeSize = volumeSize;
        _combinedVertices.Clear();
        _combinedTriangles.Clear();

        if (hasSurface)
        {
            _mesh.GetVertices(_combinedVertices);
            _combinedTriangles.AddRange(_mesh.triangles);
        }

        triangles = _combinedTriangles.Count / 3;
        TriangleCount = triangles;
        bool visible = triangles >= Mathf.Max(0, minTrianglesToShow);

        _surfaceVertices.Clear();
        _surfaceTriangles.Clear();

        _mesh.Clear();
        if (visible)
        {
            _mesh.SetVertices(_combinedVertices);
            _mesh.SetTriangles(_combinedTriangles, 0, true);
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
            _surfaceVertices.AddRange(_combinedVertices);
            _surfaceTriangles.AddRange(_combinedTriangles);
        }

        if (_meshCollider != null)
            _meshCollider.sharedMesh = null;
        if (renderDebugSurfaceObject && buildCollider && visible && _meshCollider != null)
            _meshCollider.sharedMesh = _mesh;

        SetSurfaceVisible(visible && (!hideSurfaceWhileScanning || builder == null || builder.IsFrozen));
        LastIssue = visible ? null : "TSDF surface generated too few triangles.";
        SurfaceDataUpdated?.Invoke(this);

        if (debugLog)
            Debug.Log($"[ScanCoverTsdfBranch] Surface build triangles={triangles}, visible={visible}");
    }

    private int AppendVoxelShellFallback(
        float[] tsdf,
        float[] weights,
        Vector3Int volumeSize,
        List<Vector3> vertices,
        List<int> triangles)
    {
        if (builder == null || vertices == null || triangles == null)
            return 0;

        _builderSnapshot.Clear();
        builder.GetCellsSnapshot(_builderSnapshot);
        _confirmedVoxels.Clear();

        int minHits = Mathf.Max(1, voxelShellMinHits);
        for (int i = 0; i < _builderSnapshot.Count; i++)
        {
            if (_builderSnapshot[i].count >= minHits)
                _confirmedVoxels.Add(_builderSnapshot[i].key);
        }

        if (_confirmedVoxels.Count <= 0)
            return 0;

        float cellSize = Mathf.Max(1e-4f, builder.cellSizeMeters);
        float inflate = Mathf.Max(0f, voxelShellInflateMeters);
        int trianglesBefore = triangles.Count;

        foreach (ScanCoverSkeletonBuilder_A.VoxelKey key in _confirmedVoxels)
        {
            if (ShouldSkipVoxelShell(key, weights, volumeSize, cellSize))
                continue;

            AddVoxelFallbackFaces(key, cellSize, inflate, vertices, triangles);
        }

        return (triangles.Count - trianglesBefore) / 3;
    }

    private bool ShouldSkipVoxelShell(ScanCoverSkeletonBuilder_A.VoxelKey key, float[] weights, Vector3Int volumeSize, float builderCellSize)
    {
        if (weights == null || weights.Length == 0)
            return false;

        Vector3 refLocalCenter = new Vector3(
            (key.x + 0.5f) * builderCellSize,
            (key.y + 0.5f) * builderCellSize,
            (key.z + 0.5f) * builderCellSize);

        Vector3 volumeLocalCenter = refLocalCenter - volumeCenterLocal;
        Vector3 volumeCoord = volumeLocalCenter / Mathf.Max(1e-4f, voxelSizeMeters) +
            0.5f * new Vector3(volumeSize.x, volumeSize.y, volumeSize.z);

        int x = Mathf.Clamp(Mathf.FloorToInt(volumeCoord.x), 0, volumeSize.x - 1);
        int y = Mathf.Clamp(Mathf.FloorToInt(volumeCoord.y), 0, volumeSize.y - 1);
        int z = Mathf.Clamp(Mathf.FloorToInt(volumeCoord.z), 0, volumeSize.z - 1);
        int index = x + y * volumeSize.x + z * volumeSize.x * volumeSize.y;
        return index >= 0 &&
            index < weights.Length &&
            weights[index] >= Mathf.Max(0f, voxelShellSkipIfTsdfWeightAtLeast);
    }

    private void AddVoxelFallbackFaces(
        ScanCoverSkeletonBuilder_A.VoxelKey key,
        float cellSize,
        float inflate,
        List<Vector3> vertices,
        List<int> triangles)
    {
        float x0 = key.x * cellSize - volumeCenterLocal.x - inflate;
        float y0 = key.y * cellSize - volumeCenterLocal.y - inflate;
        float z0 = key.z * cellSize - volumeCenterLocal.z - inflate;
        float x1 = x0 + cellSize + inflate * 2f;
        float y1 = y0 + cellSize + inflate * 2f;
        float z1 = z0 + cellSize + inflate * 2f;

        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x + 1, key.y, key.z)))
            AddQuad(new Vector3(x1, y0, z0), new Vector3(x1, y0, z1), new Vector3(x1, y1, z1), new Vector3(x1, y1, z0), vertices, triangles);
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x - 1, key.y, key.z)))
            AddQuad(new Vector3(x0, y0, z1), new Vector3(x0, y0, z0), new Vector3(x0, y1, z0), new Vector3(x0, y1, z1), vertices, triangles);
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y + 1, key.z)))
            AddQuad(new Vector3(x0, y1, z0), new Vector3(x1, y1, z0), new Vector3(x1, y1, z1), new Vector3(x0, y1, z1), vertices, triangles);
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y - 1, key.z)))
            AddQuad(new Vector3(x0, y0, z1), new Vector3(x1, y0, z1), new Vector3(x1, y0, z0), new Vector3(x0, y0, z0), vertices, triangles);
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y, key.z + 1)))
            AddQuad(new Vector3(x1, y0, z1), new Vector3(x0, y0, z1), new Vector3(x0, y1, z1), new Vector3(x1, y1, z1), vertices, triangles);
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y, key.z - 1)))
            AddQuad(new Vector3(x0, y0, z0), new Vector3(x1, y0, z0), new Vector3(x1, y1, z0), new Vector3(x0, y1, z0), vertices, triangles);
    }

    private static void AddQuad(
        Vector3 a,
        Vector3 b,
        Vector3 c,
        Vector3 d,
        List<Vector3> vertices,
        List<int> triangles)
    {
        int baseIndex = vertices.Count;
        vertices.Add(a);
        vertices.Add(b);
        vertices.Add(c);
        vertices.Add(d);
        triangles.Add(baseIndex + 0);
        triangles.Add(baseIndex + 1);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex + 0);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex + 3);
    }

    public bool TryCopySurfaceSnapshot(List<Vector3> vertices, List<int> triangles)
    {
        if (vertices == null || triangles == null)
            return false;

        vertices.Clear();
        triangles.Clear();

        if (_surfaceVertices.Count <= 0 || _surfaceTriangles.Count <= 0)
            return false;

        vertices.AddRange(_surfaceVertices);
        triangles.AddRange(_surfaceTriangles);
        return true;
    }

    public bool TryGetWeightAtReferenceLocalPosition(Vector3 referenceLocalPosition, out float weight)
    {
        weight = 0f;
        if (_latestWeightData == null || _latestWeightData.Length <= 0)
            return false;

        Vector3 volumeLocal = referenceLocalPosition - volumeCenterLocal;
        Vector3 volumeCoord = volumeLocal / Mathf.Max(1e-4f, voxelSizeMeters) +
            0.5f * new Vector3(_latestVolumeSize.x, _latestVolumeSize.y, _latestVolumeSize.z);

        int x = Mathf.Clamp(Mathf.FloorToInt(volumeCoord.x), 0, _latestVolumeSize.x - 1);
        int y = Mathf.Clamp(Mathf.FloorToInt(volumeCoord.y), 0, _latestVolumeSize.y - 1);
        int z = Mathf.Clamp(Mathf.FloorToInt(volumeCoord.z), 0, _latestVolumeSize.z - 1);
        int index = x + y * _latestVolumeSize.x + z * _latestVolumeSize.x * _latestVolumeSize.y;
        if (index < 0 || index >= _latestWeightData.Length)
            return false;

        weight = _latestWeightData[index];
        return true;
    }

    private void ResolveRefs()
    {
        if (!autoResolveRefs)
            return;

        if (builder == null)
            builder = GetComponent<ScanCoverSkeletonBuilder_A>();

        if (referenceFrame == null && builder != null && builder.referenceFrame != null)
            referenceFrame = builder.referenceFrame;

        if (environmentDepthManager == null)
            environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>();

        if (depthPreprocessor == null)
        {
            ScanCoverDepthPreprocessor[] preprocessors = FindObjectsByType<ScanCoverDepthPreprocessor>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            for (int i = 0; i < preprocessors.Length; i++)
            {
                if (preprocessors[i] != null && preprocessors[i].CurrentSourceEye == preferredEye)
                {
                    depthPreprocessor = preprocessors[i];
                    break;
                }
            }

            if (depthPreprocessor == null && preprocessors.Length > 0)
                depthPreprocessor = preprocessors[0];
        }
    }

    private void EnsureShader()
    {
        if (_computeShader == null)
            _computeShader = Resources.Load<ComputeShader>("ScanCoverTSDF");

        if (_computeShader == null)
            return;

        if (_clearKernel < 0)
            _clearKernel = _computeShader.FindKernel("Clear");

        if (_integrateKernel < 0)
            _integrateKernel = _computeShader.FindKernel("Integrate");

        if (_copyVolumeToBufferKernel < 0)
            _copyVolumeToBufferKernel = _computeShader.FindKernel("CopyVolumeToBuffer");
    }

    private bool EnsureVolumes()
    {
        if (_computeShader == null || _clearKernel < 0 || _integrateKernel < 0 || _copyVolumeToBufferKernel < 0)
            return SetIssue("ScanCoverTSDF.compute was not found in Resources.");

        if (_tsdfVolume != null &&
            _tsdfVolume.width == volumeSizeX &&
            _tsdfVolume.height == volumeSizeY &&
            _tsdfVolume.volumeDepth == volumeSizeZ)
        {
            return true;
        }

        ReleaseVolumes();

        _tsdfVolume = CreateVolumeTexture("ScanCover_TSDFVolume");
        _weightVolume = CreateVolumeTexture("ScanCover_TSDFWeight");
        ClearVolumes();
        return _tsdfVolume != null && _weightVolume != null;
    }

    private void ClearVolumes()
    {
        if (_computeShader == null || _tsdfVolume == null || _weightVolume == null)
            return;

        _computeShader.SetInts(VolumeSizeId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _computeShader.SetTexture(_clearKernel, TsdfVolumeId, _tsdfVolume);
        _computeShader.SetTexture(_clearKernel, WeightVolumeId, _weightVolume);
        DispatchVolumeKernel(_clearKernel);
    }

    private bool EnsureReadbackBuffers()
    {
        int voxelCount = GetExpectedVoxelCount();
        if (voxelCount <= 0)
            return SetIssue("TSDF readback buffer size is invalid.");

        if (_tsdfReadbackBuffer != null && _weightReadbackBuffer != null && _tsdfReadbackBuffer.count == voxelCount && _weightReadbackBuffer.count == voxelCount)
            return true;

        ReleaseReadbackBuffers();

        _tsdfReadbackBuffer = new ComputeBuffer(voxelCount, sizeof(float));
        _weightReadbackBuffer = new ComputeBuffer(voxelCount, sizeof(float));
        return _tsdfReadbackBuffer != null && _weightReadbackBuffer != null;
    }

    private void DispatchVolumeKernel(int kernel)
    {
        uint threadX;
        uint threadY;
        uint threadZ;
        _computeShader.GetKernelThreadGroupSizes(kernel, out threadX, out threadY, out threadZ);
        int dispatchX = Mathf.CeilToInt(volumeSizeX / (float)threadX);
        int dispatchY = Mathf.CeilToInt(volumeSizeY / (float)threadY);
        int dispatchZ = Mathf.CeilToInt(volumeSizeZ / (float)threadZ);
        _computeShader.Dispatch(kernel, dispatchX, dispatchY, dispatchZ);
    }

    private void DispatchLinearKernel(int kernel, int itemCount)
    {
        uint threadX;
        uint threadY;
        uint threadZ;
        _computeShader.GetKernelThreadGroupSizes(kernel, out threadX, out threadY, out threadZ);
        int dispatchX = Mathf.CeilToInt(itemCount / (float)threadX);
        _computeShader.Dispatch(kernel, dispatchX, 1, 1);
    }

    private RenderTexture CreateVolumeTexture(string textureName)
    {
        var texture = new RenderTexture(volumeSizeX, volumeSizeY, 0)
        {
            name = textureName,
            dimension = TextureDimension.Tex3D,
            volumeDepth = volumeSizeZ,
            enableRandomWrite = true,
            graphicsFormat = GraphicsFormat.R32_SFloat,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
            useMipMap = false,
            autoGenerateMips = false,
        };
        texture.Create();
        return texture;
    }

    private void ReleaseVolumes()
    {
        ReleaseTexture(ref _tsdfVolume);
        ReleaseTexture(ref _weightVolume);
        ReleaseReadbackBuffers();
    }

    private void ReleaseReadbackBuffers()
    {
        if (_tsdfReadbackBuffer != null)
        {
            _tsdfReadbackBuffer.Release();
            _tsdfReadbackBuffer = null;
        }

        if (_weightReadbackBuffer != null)
        {
            _weightReadbackBuffer.Release();
            _weightReadbackBuffer = null;
        }
    }

    private int GetExpectedVoxelCount()
    {
        return Mathf.Max(0, volumeSizeX) * Mathf.Max(0, volumeSizeY) * Mathf.Max(0, volumeSizeZ);
    }

    private void EnsureSurfaceResources()
    {
        if (_mesh == null)
        {
            _mesh = new Mesh
            {
                name = "ScanCover_TSDFSurfaceMesh",
                indexFormat = IndexFormat.UInt32,
            };
        }

        if (!renderDebugSurfaceObject)
        {
            if (_surfaceRoot != null)
            {
                if (_meshCollider != null)
                    _meshCollider.sharedMesh = null;
                if (_surfaceRoot != null)
                    Destroy(_surfaceRoot);
                _surfaceRoot = null;
                _meshFilter = null;
                _meshRenderer = null;
                _meshCollider = null;
            }
            if (_runtimeMaterial != null)
            {
                Destroy(_runtimeMaterial);
                _runtimeMaterial = null;
            }
            return;
        }

        if (_surfaceRoot == null)
        {
            _surfaceRoot = new GameObject("[ScanCover] TSDF Surface");
            Transform parent = referenceFrame != null ? referenceFrame : transform;
            _surfaceRoot.transform.SetParent(parent, false);
            _surfaceRoot.transform.localPosition = volumeCenterLocal;
            _surfaceRoot.transform.localRotation = Quaternion.identity;
            _surfaceRoot.transform.localScale = Vector3.one;
            _meshFilter = _surfaceRoot.AddComponent<MeshFilter>();
            _meshRenderer = _surfaceRoot.AddComponent<MeshRenderer>();
            _meshCollider = _surfaceRoot.AddComponent<MeshCollider>();
        }

        if (_meshFilter != null)
            _meshFilter.sharedMesh = _mesh;

        if (_runtimeMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader != null)
            {
                _runtimeMaterial = new Material(shader)
                {
                    name = "ScanCover_TSDFSurfaceMaterial"
                };

                if (_runtimeMaterial.HasProperty("_BaseColor"))
                    _runtimeMaterial.SetColor("_BaseColor", surfaceColor);
                if (_runtimeMaterial.HasProperty("_Color"))
                    _runtimeMaterial.SetColor("_Color", surfaceColor);
            }
        }
        if (_meshRenderer != null && _runtimeMaterial != null)
            _meshRenderer.sharedMaterial = _runtimeMaterial;
    }

    private void ReleaseSurface()
    {
        if (_mesh != null)
            Destroy(_mesh);
        _mesh = null;

        if (_runtimeMaterial != null)
            Destroy(_runtimeMaterial);
        _runtimeMaterial = null;

        if (_surfaceRoot != null)
            Destroy(_surfaceRoot);

        _surfaceRoot = null;
        _meshFilter = null;
        _meshRenderer = null;
        _meshCollider = null;
    }

    private void UpdateSurfaceVisibility()
    {
        bool shouldShow = _surfaceRoot != null &&
            _mesh != null &&
            TriangleCount >= Mathf.Max(0, minTrianglesToShow) &&
            (!hideSurfaceWhileScanning || builder == null || builder.IsFrozen) &&
            (showSurfaceWhenFrozen || builder == null || !builder.IsFrozen);

        SetSurfaceVisible(shouldShow);
    }

    private void SetSurfaceVisible(bool visible)
    {
        if (_surfaceRoot != null && _surfaceRoot.activeSelf != visible)
            _surfaceRoot.SetActive(visible);
    }

    private static void ReleaseTexture(ref RenderTexture texture)
    {
        if (texture == null)
            return;

        if (texture.IsCreated())
            texture.Release();

        Object.Destroy(texture);
        texture = null;
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog && !string.IsNullOrEmpty(issue))
            Debug.LogWarning($"[ScanCoverTsdfBranch] {issue}");
        return false;
    }
}

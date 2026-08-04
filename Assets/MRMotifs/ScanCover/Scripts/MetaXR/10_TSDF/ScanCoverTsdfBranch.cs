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
    [Min(0.01f)] public float integrateIntervalSeconds = 0.0333f;

    [Header("TSDF Volume")]
    [Min(0.03f)] public float voxelSizeMeters = 0.05f;
    [Min(16)] public int volumeSizeX = 192;
    [Min(16)] public int volumeSizeY = 96;
    [Min(16)] public int volumeSizeZ = 192;
    public Vector3 volumeCenterLocal = new Vector3(0f, 1.4f, 0f);
    [Min(0.05f)] public float truncationPositiveMeters = 0.15f;
    [Tooltip("负侧截断（米），QRS为负2体素=0.10（voxelMin），比正侧窄可减少背面渗色")]
    public float truncationNegativeMeters = -0.10f;
    [Tooltip("QuestRoomScan语义：0.5，低上限保证旧值永远可被矛盾证据纠正")]
    [Range(0.1f, 2f)] public float maxIntegratedWeight = 0.5f;

    [Header("TSDF Blending (QuestRoomScan semantics)")]
    [Tooltip("混合速率，QuestRoomScan默认0.8")]
    [Range(0.1f, 2f)] public float blendRate = 0.8f;
    [Tooltip("权重阻力系数，QuestRoomScan默认2.5")]
    [Range(0.5f, 5f)] public float stability = 2.5f;
    [Tooltip("权重增长率，QuestRoomScan默认0.025")]
    [Range(0.005f, 0.1f)] public float weightGrowth = 0.025f;

    [Header("Observation Filter")]
    [Range(0f, 1f)] public float minObservationConfidence = 0.18f;
    [Range(0f, 1f)] public float minNormalFacingDot = 0.20f;
    [Min(0f)] public float minDepthMeters = 0.35f;
    [Min(0f)] public float maxDepthMeters = 6f;

    [Header("Occlusion Gating (QuestRoomScan)")]
    [Tooltip("遮挡门控：膨胀深度拒绝在已观测表面后方写入TSDF，防止幽灵后壳")]
    public bool enableOcclusionGating = true;
    [Tooltip("深度差异阈值（米），QuestRoomScan默认0.5")]
    [Range(0.1f, 1f)] public float depthDisparityThreshold = 0.5f;
    [Tooltip("膨胀步数（jump-flood迭代次数）")]
    [Range(2, 10)] public int dilationSteps = 8;

    [Header("Surface Output")]
    public bool renderDebugSurfaceObject = false;
    public bool hideSurfaceWhileScanning = true;
    public bool showSurfaceWhenFrozen = true;
    public bool buildCollider = false;
    [Min(0)] public int minTrianglesToShow = 24;
    [Tooltip("QuestRoomScan语义：0.08，与maxIntegratedWeight=0.5配套")]
    [Range(0.01f, 0.5f)] public float minObservedWeightForMeshing = 0.08f;
    public Color surfaceColor = new Color(0.18f, 0.58f, 1.0f, 0.85f);

    [Header("Voxel Shell Supplement")]
    public bool mergeVoxelShell = true;
    [Min(1)] public int voxelShellMinHits = 2;
    [Tooltip("QuestRoomScan语义：与minObservedWeightForMeshing一致")]
    [Min(0f)] public float voxelShellSkipIfTsdfWeightAtLeast = 0.08f;
    [Min(0f)] public float voxelShellInflateMeters = 0.002f;

    [Header("Surface Smoothing (QuestRoomScan)")]
    [Tooltip("HC-Laplacian平滑迭代次数，QuestRoomScan默认1")]
    [Range(0, 4)] public int smoothIterations = 1;
    [Tooltip("Laplacian混合率，QuestRoomScan默认0.33")]
    [Range(0f, 1f)] public float smoothLambda = 0.33f;
    [Tooltip("HC收缩校正，QuestRoomScan默认0.5")]
    [Range(0f, 1f)] public float smoothBeta = 0.5f;

    [Header("Temporal Blend (QuestRoomScan)")]
    [Tooltip("时序混合：顶点位置帧间阻尼，减少跳动")]
    public bool enableTemporalBlend = true;
    [Tooltip("时序混合最小alpha，QuestRoomScan默认0.1")]
    [Range(0f, 1f)] public float temporalAlphaMin = 0.1f;
    [Tooltip("时序混合最大alpha，QuestRoomScan默认0.85")]
    [Range(0f, 1f)] public float temporalAlphaMax = 0.85f;
    [Tooltip("时序衰减率，QuestRoomScan默认0.15")]
    [Range(0.01f, 1f)] public float temporalDecayRate = 0.15f;
    [Tooltip("时序死区（米），QuestRoomScan默认0.001")]
    [Range(0.0001f, 0.01f)] public float temporalDeadzone = 0.001f;
    [Tooltip("收敛阈值（米）：顶点位移超过此值视为表面真实更新，age清零快速跟随。QuestRoomScan默认0.005")]
    [Range(0.001f, 0.02f)] public float temporalConvergeThreshold = 0.005f;

    [Header("GPU Extraction (QuestRoomScan)")]
    [Tooltip("开启后TSDF→网格提取在GPU完成（QuestRoomScan同构surface nets，含平滑+时序混合）；关闭回退CPU mesher")]
    public bool useGpuExtraction = true;
    [Tooltip("开启后网格直接由GPU缓冲间接渲染（QRS GPUMeshRenderer同构，需场景中有ScanCoverGpuMeshRenderer），提取随融合30Hz刷新且零CPU回读；快照/导出仍走BuildSurfaceNow原路径")]
    public bool useIndirectRendering = true;

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
    private static readonly int BlendRateId = Shader.PropertyToID("_BlendRate");
    private static readonly int StabilityId = Shader.PropertyToID("_Stability");
    private static readonly int WeightGrowthId = Shader.PropertyToID("_WeightGrowth");
    private static readonly int CameraWorldPositionId = Shader.PropertyToID("_CameraWorldPosition");
    private static readonly int VoxelOffsetId = Shader.PropertyToID("_VoxelOffset");
    private static readonly int WorldToClipId = Shader.PropertyToID("_WorldToClip");
    private static readonly int VolumeLocalToWorldId = Shader.PropertyToID("_VolumeLocalToWorld");
    private static readonly int DilationSrcId = Shader.PropertyToID("_DilationSrc");
    private static readonly int DilationDstId = Shader.PropertyToID("_DilationDst");
    private static readonly int DilationStepSizeId = Shader.PropertyToID("_DilationStepSize");
    private static readonly int DilationFocalLengthId = Shader.PropertyToID("_DilationFocalLength");
    private static readonly int DepthDisparityThresholdId = Shader.PropertyToID("_DepthDisparityThreshold");
    private static readonly int EnableOcclusionGatingId = Shader.PropertyToID("_EnableOcclusionGating");

    private ComputeShader _computeShader;
    private int _clearKernel = -1;
    private int _integrateKernel = -1;
    private int _copyVolumeToBufferKernel = -1;
    private int _pruneKernel = -1;
    private int _initDilationKernel = -1;
    private int _dilationStepKernel = -1;
    private RenderTexture _tsdfVolume;
    private RenderTexture _dilationTexA;
    private RenderTexture _dilationTexB;
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
    private float _nextPruneTime;

    // Temporal blend state: cellIndex → (position, age) from previous frame.
    private readonly Dictionary<int, TemporalVertexState> _temporalState = new Dictionary<int, TemporalVertexState>();
    private struct TemporalVertexState
    {
        public Vector3 Position;
        public float Age;
    }
    private readonly List<int> _cellIndices = new List<int>(16384);
    private int _integrationCountSinceClear;
    private const int WarmupIntegrations = 3;
    private const float PruneIntervalSeconds = 3f;
    private bool _hasPendingReadback;
    private AsyncGPUReadbackRequest _tsdfReadbackRequest;
    private AsyncGPUReadbackRequest _weightReadbackRequest;
    private Vector3Int _pendingVolumeSize;
    private readonly List<ScanCoverSkeletonBuilder_A.CellInfo> _builderSnapshot = new List<ScanCoverSkeletonBuilder_A.CellInfo>(16384);
    private readonly HashSet<ScanCoverSkeletonBuilder_A.VoxelKey> _confirmedVoxels = new HashSet<ScanCoverSkeletonBuilder_A.VoxelKey>();
    private readonly List<Vector3> _combinedVertices = new List<Vector3>(65536);
    private readonly List<int> _combinedTriangles = new List<int>(131072);
    private readonly List<Vector3> _combinedNormals = new List<Vector3>(65536);

    // GPU surface nets extraction (QuestRoomScan-faithful path).
    private ComputeShader _gpuSurfaceNetsShader;
    private ScanCoverGpuSurfaceNets _gpuSurfaceNets;
    private ScanCoverGpuMeshRenderer _gpuMeshRenderer;
    private bool _hasGpuRenderData;
    // GPU 间接渲染健康监测：每 ~1s 回读 20 字节 draw args（QRS 也只回读计数）。
    // 连续 3 次抽取索引为 0 → 判定间接渲染静默失败，自动回退 CPU 网格显示。
    private AsyncGPUReadbackRequest _gpuDrawArgsRequest;
    private bool _gpuDrawArgsReadbackPending;
    private float _nextGpuDrawArgsCheckTime;
    private int _gpuZeroDrawStreak;
    private Matrix4x4 _gpuRenderLocalToWorld = Matrix4x4.identity;
    private readonly List<Vector3> _gpuVertices = new List<Vector3>(65536);
    private readonly List<int> _gpuTriangles = new List<int>(131072);
    private readonly List<Vector3> _surfaceVertices = new List<Vector3>(65536);
    private readonly List<int> _surfaceTriangles = new List<int>(131072);
    private float[] _latestTsdfData;
    private float[] _latestWeightData;
    private Vector3Int _latestVolumeSize;

    private void Awake()
    {
        ResolveRefs();
        EnsureShader();
        EnsureGpuMeshRenderer();
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

        if (useGpuExtraction && _gpuSurfaceNets != null && _gpuSurfaceNets.HasPendingReadback)
            UpdatePendingGpuMesh();

        ConsumeGpuDrawArgsReadback();

        UpdateSurfaceVisibility();

        if (!integrateWhileScanning || builder == null || !builder.scanEnabled)
            return;

        // QuestRoomScan warmup: discard the first N integrations to avoid
        // Quest 3 depth sensor startup calibration noise.
        if (_integrationCountSinceClear > 0 &&
            _integrationCountSinceClear <= WarmupIntegrations &&
            Time.time >= _nextIntegrateTime)
        {
            ClearVolumes();
            _integrationCountSinceClear = 0;
            if (debugLog)
                Debug.Log("[ScanCoverTsdfBranch] Warmup clear after initial integrations.");
        }

        // QuestRoomScan periodic prune: reset very-low-weight voxels to empty.
        if (_pruneKernel >= 0 && Time.time >= _nextPruneTime)
        {
            _nextPruneTime = Time.time + PruneIntervalSeconds;
            _computeShader.SetInts(VolumeSizeId, volumeSizeX, volumeSizeY, volumeSizeZ);
            _computeShader.SetTexture(_pruneKernel, TsdfVolumeId, _tsdfVolume);
            _computeShader.SetTexture(_pruneKernel, WeightVolumeId, _weightVolume);
            DispatchVolumeKernel(_pruneKernel);
        }

        if (Time.time < _nextIntegrateTime)
            return;

        _nextIntegrateTime = Time.time + Mathf.Max(0.01f, integrateIntervalSeconds);
        if (IntegrateNow() && useGpuExtraction && useIndirectRendering)
        {
            RunGpuExtractionForDisplay();
            RequestGpuDrawArgsReadback();
        }
    }

    private void OnDisable()
    {
        _hasPendingReadback = false;
    }

    private void OnDestroy()
    {
        ReleaseVolumes();
        ReleaseSurface();
        _gpuSurfaceNets?.Dispose();
        _gpuSurfaceNets = null;
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
        _computeShader.SetFloat(MaxIntegratedWeightId, Mathf.Max(0.1f, maxIntegratedWeight));
        _computeShader.SetFloat(BlendRateId, blendRate);
        _computeShader.SetFloat(StabilityId, stability);
        _computeShader.SetFloat(WeightGrowthId, weightGrowth);

        // Occlusion gating: update dilated depth, then bind to Integrate.
        RenderTexture dilatedResult = null;
        if (enableOcclusionGating &&
            _initDilationKernel >= 0 && _dilationStepKernel >= 0 &&
            EnsureDilationTextures(observationMeta.width, observationMeta.height))
        {
            UpdateDilation(observationMeta, worldToClip);
            dilatedResult = _dilationTexA; // UpdateDilation leaves result in A
        }
        _computeShader.SetInt(EnableOcclusionGatingId,
            enableOcclusionGating && dilatedResult != null ? 1 : 0);
        _computeShader.SetFloat(DepthDisparityThresholdId, depthDisparityThreshold);
        if (dilatedResult != null)
            _computeShader.SetTexture(_integrateKernel, DilationSrcId, dilatedResult);

        _computeShader.SetVector(CameraWorldPositionId, new Vector4(cameraWorldPosition.x, cameraWorldPosition.y, cameraWorldPosition.z, 1f));
        _computeShader.SetMatrix(WorldToClipId, worldToClip);
        _computeShader.SetMatrix(VolumeLocalToWorldId, volumeLocalToWorld);

        DispatchIntegrateFrustumKernel(worldToClip, volumeLocalToWorld);

        IntegrationCount++;
        _integrationCountSinceClear++;
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

        if (useGpuExtraction && _gpuSurfaceNets != null && _gpuSurfaceNets.HasPendingReadback)
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

        if (useGpuExtraction)
        {
            // QuestRoomScan GPU surface nets: extraction + smoothing +
            // temporal blend all run on GPU directly from the volume
            // textures (no CPU volume readback involved in meshing).
            if (!RunGpuExtractionForDisplay())
                return false;

            _gpuSurfaceNets.RequestMeshReadback();
        }

        if (debugLog)
            Debug.Log("[ScanCoverTsdfBranch] Requested TSDF readback.");

        return true;
    }

    private bool EnsureGpuSurfaceNets()
    {
        if (_gpuSurfaceNets != null)
            return true;

        if (_gpuSurfaceNetsShader == null)
            _gpuSurfaceNetsShader = Resources.Load<ComputeShader>("ScanCoverGpuSurfaceNets");

        if (_gpuSurfaceNetsShader == null)
            return SetIssue("ScanCoverGpuSurfaceNets.compute was not found in Resources.");

        _gpuSurfaceNets = new ScanCoverGpuSurfaceNets(_gpuSurfaceNetsShader);
        return true;
    }

    /// <summary>
    /// Runs GPU surface nets extraction so the GPU buffers hold the latest
    /// mesh.  Used both by the 30Hz indirect-rendering display path and by
    /// BuildSurfaceNow (which additionally requests a CPU readback for the
    /// snapshot/export consumers).
    /// </summary>
    private bool RunGpuExtractionForDisplay()
    {
        if (!EnsureGpuSurfaceNets())
            return false;

        _gpuSurfaceNets.MinMeshWeight = Mathf.Max(0.01f, minObservedWeightForMeshing);
        _gpuSurfaceNets.SmoothIterations = smoothIterations;
        _gpuSurfaceNets.SmoothLambda = smoothLambda;
        _gpuSurfaceNets.SmoothBeta = smoothBeta;
        _gpuSurfaceNets.TemporalAlphaMax = enableTemporalBlend ? temporalAlphaMax : 1f;
        _gpuSurfaceNets.TemporalAlphaMin = temporalAlphaMin;
        _gpuSurfaceNets.TemporalDecayRate = temporalDecayRate;
        _gpuSurfaceNets.ConvergenceThreshold = temporalConvergeThreshold;
        _gpuSurfaceNets.TemporalDeadzone = temporalDeadzone;
        _gpuSurfaceNets.EnsureBuffers(new Vector3Int(volumeSizeX, volumeSizeY, volumeSizeZ));
        _gpuSurfaceNets.Extract(_tsdfVolume, _weightVolume, voxelSizeMeters);

        Transform baseFrame = referenceFrame != null ? referenceFrame : transform;
        _gpuRenderLocalToWorld = baseFrame.localToWorldMatrix * Matrix4x4.TRS(volumeCenterLocal, Quaternion.identity, Vector3.one);
        _hasGpuRenderData = true;
        return true;
    }

    /// <summary>
    /// Indirect-rendering data outlet for ScanCoverGpuMeshRenderer (QRS
    /// GPUMeshRenderer pattern): GPU buffers + transform, no CPU readback.
    /// </summary>
    public bool TryGetGpuRenderData(
        out GraphicsBuffer vertices,
        out GraphicsBuffer indices,
        out GraphicsBuffer drawArgs,
        out Matrix4x4 localToWorld)
    {
        localToWorld = _gpuRenderLocalToWorld;
        vertices = null;
        indices = null;
        drawArgs = null;

        if (!_hasGpuRenderData || _gpuSurfaceNets == null || !useGpuExtraction || !useIndirectRendering)
            return false;

        vertices = _gpuSurfaceNets.VertexBuffer;
        indices = _gpuSurfaceNets.IndexBuffer;
        drawArgs = _gpuSurfaceNets.DrawIndirectArgsBuffer;
        localToWorld = _gpuRenderLocalToWorld;
        return vertices != null && indices != null && drawArgs != null;
    }

    public Bounds GetGpuRenderWorldBounds()
    {
        Vector3 half = new Vector3(volumeSizeX, volumeSizeY, volumeSizeZ) * (voxelSizeMeters * 0.5f) + Vector3.one * 0.25f;
        Vector3 center = _gpuRenderLocalToWorld.GetColumn(3);
        return new Bounds(center, half * 2f);
    }

    public uint LastGpuDrawIndexCount { get; private set; }
    public int GpuZeroDrawStreak => _gpuZeroDrawStreak;
    /// <summary>间接渲染连续 3 次抽取为空 → 已自动回退 CPU 网格显示。</summary>
    public bool IndirectRenderingSuspect =>
        useGpuExtraction && useIndirectRendering && _gpuZeroDrawStreak >= 3;
    public string GpuRendererIssue => _gpuMeshRenderer != null ? _gpuMeshRenderer.LastIssue : null;

    private void RequestGpuDrawArgsReadback()
    {
        if (_gpuDrawArgsReadbackPending || _gpuSurfaceNets == null ||
            _gpuSurfaceNets.DrawIndirectArgsBuffer == null ||
            Time.unscaledTime < _nextGpuDrawArgsCheckTime)
            return;
        _nextGpuDrawArgsCheckTime = Time.unscaledTime + 1f;
        _gpuDrawArgsReadbackPending = true;
        _gpuDrawArgsRequest = AsyncGPUReadback.Request(_gpuSurfaceNets.DrawIndirectArgsBuffer);
    }

    private void ConsumeGpuDrawArgsReadback()
    {
        if (!_gpuDrawArgsReadbackPending || !_gpuDrawArgsRequest.done)
            return;
        _gpuDrawArgsReadbackPending = false;
        if (_gpuDrawArgsRequest.hasError)
        {
            _gpuZeroDrawStreak++;
        }
        else
        {
            var args = _gpuDrawArgsRequest.GetData<uint>();
            LastGpuDrawIndexCount = args.Length > 0 ? args[0] : 0u;
            _gpuZeroDrawStreak = LastGpuDrawIndexCount == 0 ? _gpuZeroDrawStreak + 1 : 0;
        }
        if (_gpuZeroDrawStreak == 3)
        {
            Debug.LogWarning(
                "[ScanCoverTsdfBranch] 间接渲染连续3次抽取索引为0——已自动回退CPU网格显示。" +
                "疑似设备端 shader 被剥离或提取为空。", this);
        }
    }

    private void UpdatePendingGpuMesh()
    {
        string gpuIssue;
        if (!_gpuSurfaceNets.TryConsumeMeshReadback(_gpuVertices, _gpuTriangles, out gpuIssue))
            return;

        if (!string.IsNullOrEmpty(gpuIssue))
        {
            SetIssue(gpuIssue);
            return;
        }

        BuildMeshFromGpuData(_gpuVertices, _gpuTriangles);
    }

    /// <summary>
    /// Builds the display/snapshot mesh from GPU-extracted surface nets
    /// data.  Positions are already volume-local and smoothing/temporal
    /// blending were applied on GPU, so this only feeds the existing
    /// mesh/snapshot path (hybrid receiver, collider, diagnostics).
    /// </summary>
    private void BuildMeshFromGpuData(List<Vector3> vertices, List<int> triangleIndices)
    {
        EnsureSurfaceResources();
        if (_mesh == null)
        {
            SetIssue("TSDF mesh is missing.");
            return;
        }

        _combinedVertices.Clear();
        _combinedTriangles.Clear();
        _combinedVertices.AddRange(vertices);
        _combinedTriangles.AddRange(triangleIndices);

        int triangles = _combinedTriangles.Count / 3;
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
            Debug.Log($"[ScanCoverTsdfBranch] GPU surface build verts={vertices.Count}, triangles={triangles}, visible={visible}");
    }

    public void ClearAll()
    {
        EnsureShader();
        EnsureVolumes();
        ClearVolumes();
        IntegrationCount = 0;
        _integrationCountSinceClear = 0;
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
        _hasGpuRenderData = false;
        _gpuSurfaceNets?.ResetTemporal();
        _latestVolumeSize = default;
        _temporalState.Clear();
        _cellIndices.Clear();
        _combinedNormals.Clear();
        _gpuSurfaceNets?.ResetTemporal();
        LastGpuDrawIndexCount = 0;
        _gpuZeroDrawStreak = 0;
        _gpuDrawArgsReadbackPending = false;
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

        if (useGpuExtraction)
        {
            // GPU extraction owns meshing (ScanCoverGpuSurfaceNets); the
            // CPU volume arrays are kept only for weight queries and
            // diagnostics consumers (TryGetWeightAtReferenceLocalPosition,
            // fused point cloud, reference shell).
            _latestTsdfData = tsdf;
            _latestWeightData = weights;
            _latestVolumeSize = _pendingVolumeSize;
            return;
        }

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
        _cellIndices.Clear();
        _combinedNormals.Clear();
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
                out mesherIssue,
                _cellIndices,
                _combinedNormals);
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

            // QuestRoomScan extraction post-processing:
            // 1. HC-Laplacian smoothing — grid 6-neighborhood weighted by
            //    normal agreement (corners preserved), ping-pong update.
            if (smoothIterations > 0)
            {
                ScanCoverTsdfMesherUtil.SmoothMesh(
                    _combinedVertices, _combinedNormals, _cellIndices,
                    volumeSize, smoothLambda, smoothBeta, smoothIterations);
            }

            // 2. Temporal blend — damps vertex positions between frames,
            //    preventing jitter and flicker.
            if (enableTemporalBlend)
            {
                ApplyTemporalBlend(_combinedVertices, _cellIndices);
            }
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

    /// <summary>
    /// QuestRoomScan TemporalBlend kernel — adaptive per-vertex damping.
    /// Each cell index maps to at most one vertex per frame, so the cell
    /// index serves as the temporal key.  Deadzone prevents micro-jitter;
    /// age-based alpha decay lets the mesh converge smoothly.
    /// </summary>
    private void ApplyTemporalBlend(List<Vector3> vertices, List<int> cellIndices)
    {
        if (vertices == null || cellIndices == null ||
            vertices.Count == 0 || vertices.Count != cellIndices.Count)
        {
            _temporalState.Clear();
            return;
        }

        var newState = new Dictionary<int, TemporalVertexState>(vertices.Count);

        for (int i = 0; i < vertices.Count; i++)
        {
            int cellIdx = cellIndices[i];
            Vector3 currentPos = vertices[i];

            if (!_temporalState.TryGetValue(cellIdx, out TemporalVertexState prev))
            {
                // First sighting: no blending, just record.
                newState[cellIdx] = new TemporalVertexState
                {
                    Position = currentPos,
                    Age = 0f
                };
                continue;
            }

            float dist = Vector3.Distance(currentPos, prev.Position);

            if (dist < temporalDeadzone)
            {
                // Within deadzone: hold previous position exactly.
                vertices[i] = prev.Position;
                newState[cellIdx] = new TemporalVertexState
                {
                    Position = prev.Position,
                    Age = prev.Age + 1f
                };
                continue;
            }

            // QuestRoomScan TemporalBlend: displacement beyond the converge
            // threshold means the surface genuinely updated — reset age to 0
            // so alpha jumps to alphaMax and the vertex follows quickly.
            // (Without this reset, converged vertices stay frozen at
            // alphaMin and lag behind the refining TSDF, producing ghost
            // duplicate layers and tangled bridging triangles.)
            float age = (dist > temporalConvergeThreshold) ? 0f : prev.Age + 1f;
            float alpha = temporalAlphaMin +
                (temporalAlphaMax - temporalAlphaMin) *
                Mathf.Exp(-age * temporalDecayRate);

            vertices[i] = Vector3.Lerp(prev.Position, currentPos, alpha);
            newState[cellIdx] = new TemporalVertexState
            {
                Position = vertices[i],
                Age = age
            };
        }

        // Replace old state with new state (cells not seen this frame are
        // dropped, matching QuestRoomScan's stateless temporal behavior).
        _temporalState.Clear();
        foreach (var kvp in newState)
            _temporalState[kvp.Key] = kvp.Value;
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

        if (_pruneKernel < 0)
            _pruneKernel = _computeShader.FindKernel("Prune");

        if (_initDilationKernel < 0)
            _initDilationKernel = _computeShader.FindKernel("InitDilation");

        if (_dilationStepKernel < 0)
            _dilationStepKernel = _computeShader.FindKernel("DilationStep");
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

    private bool EnsureDilationTextures(int width, int height)
    {
        if (_dilationTexA != null && _dilationTexB != null &&
            _dilationTexA.width == width && _dilationTexA.height == height)
            return true;

        ReleaseDilationTextures();
        _dilationTexA = CreateDilationTexture("ScanCover_DilationA", width, height);
        _dilationTexB = CreateDilationTexture("ScanCover_DilationB", width, height);
        return _dilationTexA != null && _dilationTexB != null;
    }

    private void UpdateDilation(RenderTexture observationMeta, Matrix4x4 worldToClip)
    {
        int width = observationMeta.width;
        int height = observationMeta.height;
        float focalLength = worldToClip.m00;

        // Init: copy depth from observation meta to both ping-pong textures.
        _computeShader.SetVector(SourceSizeId, new Vector4(width, height, 0f, 0f));
        _computeShader.SetTexture(_initDilationKernel, SourceObservationMetaTextureId, observationMeta);
        _computeShader.SetTexture(_initDilationKernel, DilationSrcId, _dilationTexA);
        _computeShader.SetTexture(_initDilationKernel, DilationDstId, _dilationTexB);
        _computeShader.Dispatch(_initDilationKernel,
            Mathf.CeilToInt(width / 8f), Mathf.CeilToInt(height / 8f), 1);

        // Jump-flood steps: propagate minimum depth with exponentially
        // decreasing step sizes.  Ping-pong between A and B.
        RenderTexture src = _dilationTexA;
        RenderTexture dst = _dilationTexB;
        for (int i = dilationSteps - 1; i >= 0; i--)
        {
            int stepSize = 1 << i;
            _computeShader.SetInt(DilationStepSizeId, stepSize);
            _computeShader.SetFloat(DilationFocalLengthId, focalLength);
            _computeShader.SetFloat(TruncationPositiveId, Mathf.Max(0.05f, truncationPositiveMeters));
            _computeShader.SetFloat(VoxelSizeId, voxelSizeMeters);
            _computeShader.SetVector(SourceSizeId, new Vector4(width, height, 0f, 0f));
            _computeShader.SetTexture(_dilationStepKernel, DilationSrcId, src);
            _computeShader.SetTexture(_dilationStepKernel, DilationDstId, dst);
            _computeShader.Dispatch(_dilationStepKernel,
                Mathf.CeilToInt(width / 8f), Mathf.CeilToInt(height / 8f), 1);
            (src, dst) = (dst, src);
        }

        // After the loop, src holds the final dilated result.  Copy it back
        // to _dilationTexA so the caller can always read from A.
        if (src != _dilationTexA)
        {
            Graphics.Blit(src, _dilationTexA);
        }
    }

    private static RenderTexture CreateDilationTexture(string name, int width, int height)
    {
        var tex = new RenderTexture(width, height, 0)
        {
            name = name,
            enableRandomWrite = true,
            graphicsFormat = GraphicsFormat.R32G32B32A32_SFloat,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Point,
            useMipMap = false,
            autoGenerateMips = false,
        };
        tex.Create();
        return tex;
    }

    private void ReleaseDilationTextures()
    {
        ReleaseTexture(ref _dilationTexA);
        ReleaseTexture(ref _dilationTexB);
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

    /// <summary>
    /// Frustum sub-dispatch (QRS sparse-dispatch equivalent): Integrate only
    /// runs over the voxel AABB covering frustum∩volume, which is what makes
    /// 30Hz affordable on Quest.  The shader's per-voxel NDC test remains the
    /// authoritative acceptance check; this only prunes thread groups that
    /// cannot possibly contribute.  z=-1/0/1 are all unprojected so the AABB
    /// is valid regardless of the platform's clip-z convention.
    /// </summary>
    private void DispatchIntegrateFrustumKernel(Matrix4x4 worldToClip, Matrix4x4 volumeLocalToWorld)
    {
        Matrix4x4 clipToWorld = worldToClip.inverse;
        Matrix4x4 worldToVolume = volumeLocalToWorld.inverse;

        Vector3 min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        bool any = false;
        for (int zi = 0; zi < 3; zi++)
        {
            float z = zi - 1f; // -1, 0, 1
            for (int yi = 0; yi < 2; yi++)
            for (int xi = 0; xi < 2; xi++)
            {
                Vector4 clip = new Vector4(xi == 0 ? -1f : 1f, yi == 0 ? -1f : 1f, z, 1f);
                Vector4 world = clipToWorld * clip;
                if (Mathf.Abs(world.w) < 1e-6f)
                    continue;
                Vector3 worldPos = ((Vector3)world) / world.w;
                Vector3 localPos = worldToVolume.MultiplyPoint(worldPos);
                Vector3 voxel = localPos / voxelSizeMeters +
                    new Vector3(volumeSizeX, volumeSizeY, volumeSizeZ) * 0.5f;
                min = Vector3.Min(min, voxel);
                max = Vector3.Max(max, voxel);
                any = true;
            }
        }

        if (!any)
        {
            _computeShader.SetInts(VoxelOffsetId, 0, 0, 0);
            return;
        }

        const int margin = 1;
        int x0 = Mathf.Clamp(Mathf.FloorToInt(min.x) - margin, 0, volumeSizeX - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(min.y) - margin, 0, volumeSizeY - 1);
        int z0 = Mathf.Clamp(Mathf.FloorToInt(min.z) - margin, 0, volumeSizeZ - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(max.x) + margin, 0, volumeSizeX - 1);
        int y1 = Mathf.Clamp(Mathf.CeilToInt(max.y) + margin, 0, volumeSizeY - 1);
        int z1 = Mathf.Clamp(Mathf.CeilToInt(max.z) + margin, 0, volumeSizeZ - 1);

        _computeShader.SetInts(VoxelOffsetId, x0, y0, z0);
        uint threadX;
        uint threadY;
        uint threadZ;
        _computeShader.GetKernelThreadGroupSizes(_integrateKernel, out threadX, out threadY, out threadZ);
        _computeShader.Dispatch(_integrateKernel,
            Mathf.CeilToInt((x1 - x0 + 1) / (float)threadX),
            Mathf.CeilToInt((y1 - y0 + 1) / (float)threadY),
            Mathf.CeilToInt((z1 - z0 + 1) / (float)threadZ));
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
        ReleaseDilationTextures();
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

        // Indirect rendering (QRS GPUMeshRenderer pattern) replaces the CPU
        // mesh display entirely; the snapshot mesh stays data-only.
        // Self-heal: if the indirect path keeps producing zero indices
        // (device shader stripping, extraction failure...), stop hiding the
        // CPU mesh so the legacy display path takes over again.
        if (useGpuExtraction && useIndirectRendering && _gpuZeroDrawStreak < 3)
            shouldShow = false;

        SetSurfaceVisible(shouldShow);
    }

    /// <summary>
    /// Runtime-created indirect renderer (same pattern as the QuestRoom
    /// pipeline's runtime AddComponent), so no scene/prefab wiring is needed
    /// and serialized defaults cannot fight the code.
    /// </summary>
    private void EnsureGpuMeshRenderer()
    {
        if (!useGpuExtraction || !useIndirectRendering)
            return;

        if (_gpuMeshRenderer == null)
        {
            _gpuMeshRenderer = GetComponent<ScanCoverGpuMeshRenderer>();
            if (_gpuMeshRenderer == null)
                _gpuMeshRenderer = gameObject.AddComponent<ScanCoverGpuMeshRenderer>();
        }

        _gpuMeshRenderer.branch = this;
        _gpuMeshRenderer.surfaceColor = surfaceColor;
        _gpuMeshRenderer.renderVisible = true;
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

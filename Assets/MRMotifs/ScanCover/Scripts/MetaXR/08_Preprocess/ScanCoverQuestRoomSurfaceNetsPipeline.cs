using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

/// <summary>
/// Independent QuestRoomScan-style production candidate:
/// Meta depth -> projective TSDF -> GPU Surface Nets -> persistent temporal wire mesh.
/// It deliberately does not read the legacy ScanCover TSDF, DMC, QEF or certificate state.
/// Architecture adapted from QuestRoomScan (MIT); see ThirdPartyNotices/QuestRoomScan-MIT.txt.
/// </summary>
[DefaultExecutionOrder(-25)]
[DisallowMultipleComponent]
public sealed class ScanCoverQuestRoomSurfaceNetsPipeline : MonoBehaviour
{
    [Serializable]
    public sealed class EyeInputDiagnostics
    {
        public int attempts;
        public int integrations;
        public int acceptedFrame = -1;
        public bool bound;
        public bool active;
        public bool ready;
        public bool expectedEye;
        public bool outputsAvailable;
        public bool reprojectionAvailable;
        public bool eyePoseAvailable;
        public bool integrated;
        public int sourceEye = -1;
        public int width;
        public int height;
        public int dispatchFrame = -1;
        public float dispatchDeltaMilliseconds = -1f;
        public string issue = "not_attempted";

        public void BeginAttempt(int frameOrdinal)
        {
            attempts++;
            acceptedFrame = frameOrdinal;
            bound = false;
            active = false;
            ready = false;
            expectedEye = false;
            outputsAvailable = false;
            reprojectionAvailable = false;
            eyePoseAvailable = false;
            integrated = false;
            sourceEye = -1;
            width = 0;
            height = 0;
            dispatchFrame = -1;
            dispatchDeltaMilliseconds = -1f;
            issue = "unresolved";
        }

        public void ResetAll()
        {
            attempts = 0;
            integrations = 0;
            acceptedFrame = -1;
            bound = false;
            active = false;
            ready = false;
            expectedEye = false;
            outputsAvailable = false;
            reprojectionAvailable = false;
            eyePoseAvailable = false;
            integrated = false;
            sourceEye = -1;
            width = 0;
            height = 0;
            dispatchFrame = -1;
            dispatchDeltaMilliseconds = -1f;
            issue = "not_attempted";
        }
    }

    [Header("Production candidate")]
    [SerializeField] private bool renderVisible = true;
    [SerializeField] private Color wireColor = new Color(1f, 0.62f, 0.02f, 0.96f);

    [Header("Single-resolution volume")]
    [SerializeField, Range(64, 192)] private int volumeSizeX = 144;
    [SerializeField, Range(48, 128)] private int volumeSizeY = 80;
    [SerializeField, Range(64, 192)] private int volumeSizeZ = 144;
    [SerializeField, Range(0.03f, 0.10f)] private float voxelSizeMeters = 0.05f;
    [SerializeField, Range(0.05f, 0.30f)] private float truncationDistanceMeters = 0.15f;

    [Header("Projective integration")]
    [SerializeField, Range(0f, 1f)] private float minimumObservationConfidence = 0.12f;
    [SerializeField, Range(0f, 1f)] private float minimumNormalFacingDot = 0.20f;
    [SerializeField, Min(0.05f)] private float minimumDepthMeters = 0.30f;
    [SerializeField, Min(0.5f)] private float maximumDepthMeters = 5.5f;
    [SerializeField, Range(0.1f, 2f)] private float blendRate = 0.8f;
    [SerializeField, Range(0.5f, 10f)] private float stability = 2.5f;
    [SerializeField, Range(0.005f, 0.1f)] private float weightGrowth = 0.025f;
    [SerializeField, Range(0.1f, 1f)] private float maximumWeight = 0.5f;

    [Header("Surface Nets")]
    [SerializeField, Range(0.01f, 0.5f)] private float minimumMeshWeight = 0.08f;
    [SerializeField, Range(0.03f, 0.25f)] private float vertexBudgetFraction = 0.12f;
    [SerializeField, Range(0, 3)] private int smoothIterations = 1;
    [SerializeField, Range(0f, 1f)] private float smoothLambda = 0.33f;
    [SerializeField, Range(0f, 1f)] private float smoothBeta = 0.50f;
    [SerializeField, Range(0.1f, 1f)] private float temporalAlphaMaximum = 0.85f;
    [SerializeField, Range(0.01f, 0.5f)] private float temporalAlphaMinimum = 0.10f;
    [SerializeField, Range(0.01f, 1f)] private float temporalDecayRate = 0.15f;
    [SerializeField, Range(0.001f, 0.03f)] private float convergenceThresholdMeters = 0.005f;
    [SerializeField, Range(0.0001f, 0.01f)] private float temporalDeadzoneMeters = 0.001f;

    [Header("Diagnostics")]
    [SerializeField] private bool debugLog = true;
    [SerializeField, Range(1, 30)] private int counterReadbackIntervalIntegrations = 2;
    [SerializeField, Range(5f, 250f)]
    private float maximumAcceptedStereoDispatchDeltaMilliseconds = 75f;

    public bool HasRenderableSurface { get; private set; }
    public int IntegrationCount { get; private set; }
    public uint LastVertexCount { get; private set; }
    public uint LastLineIndexCount { get; private set; }
    public uint LastOverflowCount { get; private set; }
    public string LastIssue { get; private set; }
    public int AcceptedFrameAttemptCount { get; private set; }
    public int StereoPairIntegrationCount { get; private set; }
    public int PartialStereoIntegrationCount { get; private set; }
    public int RejectedAcceptedFrameCount { get; private set; }
    public int StereoCompanionPreparationAttempts { get; private set; }
    public int StereoCompanionPreparationSuccesses { get; private set; }
    public int LastAcceptedSnapshotFrame { get; private set; } = -1;
    public EyeInputDiagnostics LeftEyeDiagnostics => _leftEyeDiagnostics;
    public EyeInputDiagnostics RightEyeDiagnostics => _rightEyeDiagnostics;

    private const int VertexStride = 32;
    private ComputeShader _tsdfCompute;
    private ComputeShader _surfaceCompute;
    private Material _wireMaterial;
    private RenderTexture _tsdfVolume;
    private RenderTexture _temporalState;
    private GraphicsBuffer _coordVertexMap;
    private GraphicsBuffer _vertices;
    private GraphicsBuffer _lineIndices;
    private GraphicsBuffer _counters;
    private GraphicsBuffer _dispatchArgs;
    private GraphicsBuffer _drawArgs;
    private GraphicsBuffer _smoothPositionA;
    private GraphicsBuffer _smoothPositionB;
    private int _maximumVertices;
    private int _maximumLineIndices;
    private int _totalVoxelCount;
    private int _clearTsdfKernel = -1;
    private int _integrateTsdfKernel = -1;
    private int _clearSurfaceKernel = -1;
    private int _classifyKernel = -1;
    private int _buildDispatchKernel = -1;
    private int _initSmoothKernel = -1;
    private int _smoothKernel = -1;
    private int _applySmoothKernel = -1;
    private int _temporalKernel = -1;
    private int _generateLinesKernel = -1;
    private int _buildDrawArgsKernel = -1;
    private int _initTemporalKernel = -1;
    private Matrix4x4 _volumeLocalToWorld = Matrix4x4.identity;
    private Bounds _worldBounds;
    private bool _volumePlaced;
    private bool _resourcesReady;
    private bool _counterReadbackPending;
    private int _lastAcceptedFrameOrdinal = -1;
    private ScanCoverTsdfSingleShellPrototype _owner;
    private MaterialPropertyBlock _propertyBlock;
    private ScanCoverDepthPreprocessor _acceptedLeftSource;
    private ScanCoverDepthPreprocessor _acceptedRightSource;
    private readonly EyeInputDiagnostics _leftEyeDiagnostics = new EyeInputDiagnostics();
    private readonly EyeInputDiagnostics _rightEyeDiagnostics = new EyeInputDiagnostics();

    private static readonly int VolumeId = Shader.PropertyToID("_Volume");
    private static readonly int WorldPositionId = Shader.PropertyToID("_WorldPosition");
    private static readonly int WorldNormalId = Shader.PropertyToID("_WorldNormal");
    private static readonly int ObservationMetaId = Shader.PropertyToID("_ObservationMeta");
    private static readonly int TsdfVolumeId = Shader.PropertyToID("_TsdfVolume");
    private static readonly int VoxelCountId = Shader.PropertyToID("_VoxelCount");
    private static readonly int VoxelSizeId = Shader.PropertyToID("_VoxelSize");
    private static readonly int TotalVoxelCountId = Shader.PropertyToID("_TotalVoxelCount");
    private static readonly int MaximumVerticesId = Shader.PropertyToID("_MaximumVertices");
    private static readonly int MaximumLineIndicesId = Shader.PropertyToID("_MaximumLineIndices");
    private static readonly int MinimumWeightId = Shader.PropertyToID("_MinimumWeight");
    private static readonly int CoordVertexMapId = Shader.PropertyToID("_CoordVertexMap");
    private static readonly int VerticesId = Shader.PropertyToID("_Vertices");
    private static readonly int LineIndicesId = Shader.PropertyToID("_LineIndices");
    private static readonly int CountersId = Shader.PropertyToID("_Counters");
    private static readonly int DispatchArgsId = Shader.PropertyToID("_DispatchArgs");
    private static readonly int DrawArgsId = Shader.PropertyToID("_DrawArgs");
    private static readonly int SmoothPositionAId = Shader.PropertyToID("_SmoothPositionA");
    private static readonly int SmoothPositionBId = Shader.PropertyToID("_SmoothPositionB");
    private static readonly int TemporalStateId = Shader.PropertyToID("_TemporalState");
    private static readonly int SurfaceVerticesId = Shader.PropertyToID("_SurfaceVertices");
    private static readonly int SurfaceLineIndicesId = Shader.PropertyToID("_SurfaceLineIndices");
    private static readonly int VolumeLocalToWorldId = Shader.PropertyToID("_VolumeLocalToWorld");
    private static readonly int WireColorId = Shader.PropertyToID("_WireColor");

    public void BindOwner(ScanCoverTsdfSingleShellPrototype owner)
    {
        _owner = owner;
    }

    public void BindAcceptedStereoSources(
        ScanCoverDepthPreprocessor leftSource,
        ScanCoverDepthPreprocessor rightSource)
    {
        _acceptedLeftSource = leftSource;
        _acceptedRightSource = rightSource;
    }

    /// <summary>
    /// The production raw-depth source refreshes the directly referenced eye first.
    /// This method stages only the explicitly bound companion eye from the same source
    /// frame; it never searches the scene or refreshes arbitrary preprocessors.
    /// </summary>
    public bool PrepareAcceptedStereoCompanionFrame()
    {
        StereoCompanionPreparationAttempts++;
        if (_acceptedLeftSource == null)
            return false;

        bool refreshed = _acceptedLeftSource.RefreshNow();
        if (refreshed)
            StereoCompanionPreparationSuccesses++;
        return refreshed;
    }

    public void SetRenderVisible(bool visible)
    {
        renderVisible = visible;
    }

    public void OnAcceptedDepthFrame(
        int frameOrdinal,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot acceptedSnapshot)
    {
        if (!isActiveAndEnabled || frameOrdinal == _lastAcceptedFrameOrdinal)
            return;
        _lastAcceptedFrameOrdinal = frameOrdinal;
        AcceptedFrameAttemptCount++;
        LastAcceptedSnapshotFrame = acceptedSnapshot != null
            ? acceptedSnapshot.frameIndex
            : frameOrdinal;

        if (!EnsureResources())
            return;
        if (!_volumePlaced)
            PlaceVolumeAroundViewer();

        int integratedEyes = 0;
        if (TryIntegrateBoundEye(
                _acceptedLeftSource,
                ScanCoverDepthPreprocessor.SourceEye.Left,
                acceptedSnapshot,
                _leftEyeDiagnostics))
            integratedEyes++;
        if (TryIntegrateBoundEye(
                _acceptedRightSource,
                ScanCoverDepthPreprocessor.SourceEye.Right,
                acceptedSnapshot,
                _rightEyeDiagnostics))
            integratedEyes++;

        if (integratedEyes == 0)
        {
            RejectedAcceptedFrameCount++;
            LastIssue =
                $"accepted_stereo_unavailable:left={_leftEyeDiagnostics.issue};" +
                $"right={_rightEyeDiagnostics.issue}";
            return;
        }

        if (integratedEyes == 2)
            StereoPairIntegrationCount++;
        else
            PartialStereoIntegrationCount++;

        IntegrationCount++;
        ExtractSurface();
        LastIssue = integratedEyes == 2
            ? null
            : $"accepted_stereo_partial:left={_leftEyeDiagnostics.issue};" +
              $"right={_rightEyeDiagnostics.issue}";
        if (debugLog && (IntegrationCount <= 2 || IntegrationCount % 12 == 0))
        {
            Debug.Log(
                $"[ScanCoverQuestRoomSurfaceNets] integrate={IntegrationCount} acceptedFrame={LastAcceptedSnapshotFrame} " +
                $"eyes={integratedEyes} left={_leftEyeDiagnostics.issue} right={_rightEyeDiagnostics.issue} " +
                $"verts={LastVertexCount} lineIndices={LastLineIndexCount} overflow={LastOverflowCount}",
                this);
        }
    }

    private bool TryIntegrateBoundEye(
        ScanCoverDepthPreprocessor source,
        ScanCoverDepthPreprocessor.SourceEye expectedEye,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot acceptedSnapshot,
        EyeInputDiagnostics diagnostics)
    {
        int acceptedFrame = acceptedSnapshot != null
            ? acceptedSnapshot.frameIndex
            : _lastAcceptedFrameOrdinal;
        diagnostics.BeginAttempt(acceptedFrame);
        diagnostics.bound = source != null;
        if (source == null)
        {
            diagnostics.issue = "source_not_bound";
            return false;
        }

        diagnostics.active = source.isActiveAndEnabled;
        diagnostics.ready = source.IsReady;
        diagnostics.sourceEye = (int)source.CurrentSourceEye;
        diagnostics.expectedEye = source.CurrentSourceEye == expectedEye;
        diagnostics.dispatchFrame = source.LastSuccessfulRefreshFrame;
        Vector2Int outputResolution = source.OutputResolution;
        diagnostics.width = outputResolution.x;
        diagnostics.height = outputResolution.y;

        if (!diagnostics.expectedEye)
        {
            diagnostics.issue = "wrong_eye_binding";
            return false;
        }
        if (!diagnostics.ready)
        {
            diagnostics.issue = string.IsNullOrEmpty(source.LastIssue)
                ? "source_not_ready"
                : "source_not_ready:" + source.LastIssue;
            return false;
        }
        if (!source.TryGetOutputs(
                out RenderTexture worldPosition,
                out RenderTexture worldNormal,
                out RenderTexture observationMeta))
        {
            diagnostics.issue = "outputs_unavailable";
            return false;
        }

        diagnostics.outputsAvailable = true;
        diagnostics.width = worldPosition != null ? worldPosition.width : diagnostics.width;
        diagnostics.height = worldPosition != null ? worldPosition.height : diagnostics.height;
        if (worldPosition == null || worldNormal == null || observationMeta == null ||
            diagnostics.width <= 1 || diagnostics.height <= 1)
        {
            diagnostics.issue = "invalid_output_size";
            return false;
        }

        bool isAcceptedSnapshotEye = acceptedSnapshot != null &&
                                     acceptedSnapshot.sourceEyeIndex == (int)expectedEye;
        bool hasReprojection = isAcceptedSnapshotEye && acceptedSnapshot.hasDepthReprojectionMatrix;
        Matrix4x4 reprojection = hasReprojection
            ? acceptedSnapshot.depthReprojectionMatrix
            : source.LastDispatchDepthReprojectionMatrix;
        if (!hasReprojection)
            hasReprojection = source.HasLastDispatchDepthReprojectionMatrix;
        diagnostics.reprojectionAvailable = hasReprojection;
        if (!hasReprojection)
        {
            diagnostics.issue = "reprojection_unavailable";
            return false;
        }

        bool hasEyePose = isAcceptedSnapshotEye && acceptedSnapshot.hasDispatchEyePosition;
        Vector3 eyeWorld = hasEyePose
            ? acceptedSnapshot.dispatchEyePosition
            : source.LastDispatchEyePosition;
        if (!hasEyePose)
            hasEyePose = source.HasLastDispatchEyePosition;
        diagnostics.eyePoseAvailable = hasEyePose;
        if (!hasEyePose)
        {
            diagnostics.issue = "eye_pose_unavailable";
            return false;
        }

        if (acceptedSnapshot != null &&
            acceptedSnapshot.dispatchRealtimeSeconds > 0d &&
            source.LastSuccessfulRefreshRealtimeSeconds > 0d)
        {
            diagnostics.dispatchDeltaMilliseconds = (float)Math.Abs(
                (source.LastSuccessfulRefreshRealtimeSeconds -
                 acceptedSnapshot.dispatchRealtimeSeconds) * 1000d);
            if (diagnostics.dispatchDeltaMilliseconds >
                maximumAcceptedStereoDispatchDeltaMilliseconds)
            {
                diagnostics.issue = "stale_dispatch";
                return false;
            }
        }

        DispatchProjectiveIntegration(
            worldPosition,
            worldNormal,
            observationMeta,
            reprojection,
            eyeWorld);
        diagnostics.integrated = true;
        diagnostics.integrations++;
        diagnostics.issue = "integrated";
        return true;
    }

    public void ClearAll()
    {
        if (!EnsureResources()) return;
        DispatchVolume(_tsdfCompute, _clearTsdfKernel);
        DispatchVolume(_surfaceCompute, _initTemporalKernel);
        HasRenderableSurface = false;
        IntegrationCount = 0;
        LastVertexCount = 0;
        LastLineIndexCount = 0;
        LastOverflowCount = 0;
        _lastAcceptedFrameOrdinal = -1;
        AcceptedFrameAttemptCount = 0;
        StereoPairIntegrationCount = 0;
        PartialStereoIntegrationCount = 0;
        RejectedAcceptedFrameCount = 0;
        StereoCompanionPreparationAttempts = 0;
        StereoCompanionPreparationSuccesses = 0;
        LastAcceptedSnapshotFrame = -1;
        _leftEyeDiagnostics.ResetAll();
        _rightEyeDiagnostics.ResetAll();
    }

    private void LateUpdate()
    {
        if (!renderVisible || !HasRenderableSurface || !_resourcesReady ||
            _wireMaterial == null || _vertices == null || _lineIndices == null || _drawArgs == null)
            return;

        _propertyBlock.SetBuffer(SurfaceVerticesId, _vertices);
        _propertyBlock.SetBuffer(SurfaceLineIndicesId, _lineIndices);
        _propertyBlock.SetMatrix(VolumeLocalToWorldId, _volumeLocalToWorld);
        _propertyBlock.SetColor(WireColorId, wireColor);
        RenderParams renderParams = new RenderParams(_wireMaterial)
        {
            worldBounds = _worldBounds,
            matProps = _propertyBlock,
            shadowCastingMode = ShadowCastingMode.Off,
            receiveShadows = false,
            layer = gameObject.layer
        };
        Graphics.RenderPrimitivesIndirect(
            renderParams,
            MeshTopology.Lines,
            _drawArgs,
            1);
    }

    private bool EnsureResources()
    {
        if (_resourcesReady) return true;
        _tsdfCompute = Resources.Load<ComputeShader>("ScanCoverQuestRoomTsdf");
        _surfaceCompute = Resources.Load<ComputeShader>("ScanCoverQuestRoomSurfaceNets");
        Shader wireShader = Resources.Load<Shader>("ScanCoverQuestRoomSurfaceNetsWire");
        if (wireShader == null)
            wireShader = Shader.Find("ScanCover/QuestRoomSurfaceNetsWire");
        if (_tsdfCompute == null || _surfaceCompute == null || wireShader == null)
        {
            LastIssue = "QuestRoom GPU shader 资源缺失";
            return false;
        }

        try
        {
            _clearTsdfKernel = _tsdfCompute.FindKernel("Clear");
            _integrateTsdfKernel = _tsdfCompute.FindKernel("Integrate");
            _clearSurfaceKernel = _surfaceCompute.FindKernel("ClearCountersAndMap");
            _classifyKernel = _surfaceCompute.FindKernel("ClassifyAndEmit");
            _buildDispatchKernel = _surfaceCompute.FindKernel("BuildDispatchArgs");
            _initSmoothKernel = _surfaceCompute.FindKernel("InitSmooth");
            _smoothKernel = _surfaceCompute.FindKernel("SmoothVertices");
            _applySmoothKernel = _surfaceCompute.FindKernel("ApplySmooth");
            _temporalKernel = _surfaceCompute.FindKernel("TemporalBlend");
            _generateLinesKernel = _surfaceCompute.FindKernel("GenerateLineIndices");
            _buildDrawArgsKernel = _surfaceCompute.FindKernel("BuildDrawArgs");
            _initTemporalKernel = _surfaceCompute.FindKernel("InitTemporal");
        }
        catch (Exception ex)
        {
            LastIssue = "QuestRoom GPU kernel 缺失: " + ex.Message;
            return false;
        }

        _totalVoxelCount = volumeSizeX * volumeSizeY * volumeSizeZ;
        _maximumVertices = Mathf.Max(4096, Mathf.CeilToInt(_totalVoxelCount * vertexBudgetFraction));
        // Each Surface Nets cell can own at most three quads. Two triangles per
        // quad and three line pairs per triangle gives a strict 36-index bound.
        _maximumLineIndices = _maximumVertices * 36;
        _tsdfVolume = CreateVolume(
            "ScanCover_QuestRoom_TSDF",
            GraphicsFormat.R16G16_SFloat);
        _temporalState = CreateVolume(
            "ScanCover_QuestRoom_Temporal",
            GraphicsFormat.R16G16B16A16_SFloat);

        _coordVertexMap = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _totalVoxelCount,
            sizeof(int));
        _vertices = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumVertices,
            VertexStride);
        _lineIndices = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumLineIndices,
            sizeof(uint));
        _counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, sizeof(uint));
        GraphicsBuffer.Target indirect =
            GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;
        _dispatchArgs = new GraphicsBuffer(indirect, 3, sizeof(uint));
        _drawArgs = new GraphicsBuffer(indirect, 5, sizeof(uint));
        _smoothPositionA = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumVertices,
            sizeof(float) * 3);
        _smoothPositionB = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumVertices,
            sizeof(float) * 3);
        _wireMaterial = new Material(wireShader)
        {
            name = "ScanCover_QuestRoom_SurfaceNets_Wire"
        };
        _propertyBlock = new MaterialPropertyBlock();
        BindSurfaceBuffers();
        SetSurfaceConstants();
        DispatchVolume(_tsdfCompute, _clearTsdfKernel);
        DispatchVolume(_surfaceCompute, _initTemporalKernel);
        _resourcesReady = true;

        if (debugLog)
        {
            long approximateBytes = (long)_totalVoxelCount * (4 + 8 + 4) +
                                    (long)_maximumVertices * VertexStride +
                                    (long)_maximumLineIndices * 4 +
                                    (long)_maximumVertices * sizeof(float) * 3 * 2;
            Debug.Log(
                $"[ScanCoverQuestRoomSurfaceNets] GPU ready volume={volumeSizeX}x{volumeSizeY}x{volumeSizeZ} " +
                $"voxel={voxelSizeMeters:F3}m budget~={approximateBytes / (1024 * 1024)}MB",
                this);
        }
        return true;
    }

    private void PlaceVolumeAroundViewer()
    {
        Vector3 viewer = Camera.main != null ? Camera.main.transform.position : transform.position;
        float halfHeight = volumeSizeY * voxelSizeMeters * 0.5f;
        float centerY = Mathf.Max(halfHeight - 0.15f, viewer.y + 0.15f);
        Vector3 center = new Vector3(viewer.x, centerY, viewer.z);
        _volumeLocalToWorld = Matrix4x4.TRS(center, Quaternion.identity, Vector3.one);
        Vector3 size = new Vector3(volumeSizeX, volumeSizeY, volumeSizeZ) * voxelSizeMeters;
        _worldBounds = new Bounds(center, size);
        _volumePlaced = true;
    }

    private void DispatchProjectiveIntegration(
        RenderTexture worldPosition,
        RenderTexture worldNormal,
        RenderTexture observationMeta,
        Matrix4x4 worldToClip,
        Vector3 eyeWorld)
    {
        _tsdfCompute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _tsdfCompute.SetFloat(VoxelSizeId, voxelSizeMeters);
        _tsdfCompute.SetFloat("_TruncationDistance", truncationDistanceMeters);
        _tsdfCompute.SetFloat("_MinimumDepth", minimumDepthMeters);
        _tsdfCompute.SetFloat("_MaximumDepth", maximumDepthMeters);
        _tsdfCompute.SetFloat("_MinimumConfidence", minimumObservationConfidence);
        _tsdfCompute.SetFloat("_MinimumNormalDot", minimumNormalFacingDot);
        _tsdfCompute.SetFloat("_BlendRate", blendRate);
        _tsdfCompute.SetFloat("_Stability", stability);
        _tsdfCompute.SetFloat("_WeightGrowth", weightGrowth);
        _tsdfCompute.SetFloat("_MaximumWeight", maximumWeight);
        _tsdfCompute.SetVector("_SourceSize", new Vector4(worldPosition.width, worldPosition.height, 0f, 0f));
        _tsdfCompute.SetVector("_EyeWorldPosition", new Vector4(eyeWorld.x, eyeWorld.y, eyeWorld.z, 1f));
        _tsdfCompute.SetMatrix("_WorldToClip", worldToClip);
        _tsdfCompute.SetMatrix(VolumeLocalToWorldId, _volumeLocalToWorld);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, VolumeId, _tsdfVolume);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, WorldPositionId, worldPosition);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, WorldNormalId, worldNormal);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, ObservationMetaId, observationMeta);
        DispatchVolume(_tsdfCompute, _integrateTsdfKernel);
    }

    private void ExtractSurface()
    {
        SetSurfaceConstants();
        BindSurfaceBuffers();
        _surfaceCompute.SetTexture(_classifyKernel, TsdfVolumeId, _tsdfVolume);
        _surfaceCompute.SetTexture(_generateLinesKernel, TsdfVolumeId, _tsdfVolume);

        int clearGroups = Mathf.CeilToInt(_totalVoxelCount / 64f);
        _surfaceCompute.Dispatch(_clearSurfaceKernel, clearGroups, 1, 1);
        DispatchVolume(_surfaceCompute, _classifyKernel);
        _surfaceCompute.Dispatch(_buildDispatchKernel, 1, 1, 1);

        int iterationCount = Mathf.Max(0, smoothIterations);
        if (iterationCount > 0)
        {
            _surfaceCompute.DispatchIndirect(_initSmoothKernel, _dispatchArgs);
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                GraphicsBuffer source = iteration % 2 == 0
                    ? _smoothPositionA
                    : _smoothPositionB;
                GraphicsBuffer destination = iteration % 2 == 0
                    ? _smoothPositionB
                    : _smoothPositionA;
                SetBuffer(_smoothKernel, SmoothPositionAId, source);
                SetBuffer(_smoothKernel, SmoothPositionBId, destination);
                _surfaceCompute.DispatchIndirect(_smoothKernel, _dispatchArgs);
            }

            GraphicsBuffer finalSmooth = iterationCount % 2 == 0
                ? _smoothPositionA
                : _smoothPositionB;
            SetBuffer(_applySmoothKernel, SmoothPositionAId, finalSmooth);
            _surfaceCompute.DispatchIndirect(_applySmoothKernel, _dispatchArgs);
        }

        _surfaceCompute.DispatchIndirect(_temporalKernel, _dispatchArgs);
        _surfaceCompute.DispatchIndirect(_generateLinesKernel, _dispatchArgs);
        _surfaceCompute.Dispatch(_buildDrawArgsKernel, 1, 1, 1);

        int interval = Mathf.Max(1, counterReadbackIntervalIntegrations);
        if (!_counterReadbackPending && (IntegrationCount <= 1 || IntegrationCount % interval == 0))
        {
            _counterReadbackPending = true;
            AsyncGPUReadback.Request(_counters, OnCounterReadback);
        }
    }

    private void OnCounterReadback(AsyncGPUReadbackRequest request)
    {
        _counterReadbackPending = false;
        if (!this || request.hasError) return;
        var values = request.GetData<uint>();
        if (values.Length < 3) return;
        LastVertexCount = Math.Min(values[0], (uint)_maximumVertices);
        LastLineIndexCount = Math.Min(values[1], (uint)_maximumLineIndices);
        LastOverflowCount = values[2];
        bool wasReady = HasRenderableSurface;
        HasRenderableSurface = StereoPairIntegrationCount > 0 &&
                               LastVertexCount > 0 &&
                               LastLineIndexCount >= 6;
        if (!wasReady && HasRenderableSurface)
            _owner?.NotifyQuestRoomSurfaceNetsReady();
    }

    private void BindSurfaceBuffers()
    {
        SetBuffer(_clearSurfaceKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_clearSurfaceKernel, CountersId, _counters);
        SetBuffer(_classifyKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_classifyKernel, VerticesId, _vertices);
        SetBuffer(_classifyKernel, CountersId, _counters);
        SetBuffer(_buildDispatchKernel, CountersId, _counters);
        SetBuffer(_buildDispatchKernel, DispatchArgsId, _dispatchArgs);
        SetBuffer(_initSmoothKernel, VerticesId, _vertices);
        SetBuffer(_initSmoothKernel, SmoothPositionAId, _smoothPositionA);
        SetBuffer(_initSmoothKernel, CountersId, _counters);
        SetBuffer(_smoothKernel, VerticesId, _vertices);
        SetBuffer(_smoothKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_smoothKernel, SmoothPositionAId, _smoothPositionA);
        SetBuffer(_smoothKernel, SmoothPositionBId, _smoothPositionB);
        SetBuffer(_smoothKernel, CountersId, _counters);
        SetBuffer(_applySmoothKernel, VerticesId, _vertices);
        SetBuffer(_applySmoothKernel, SmoothPositionAId, _smoothPositionA);
        SetBuffer(_applySmoothKernel, CountersId, _counters);
        SetBuffer(_temporalKernel, VerticesId, _vertices);
        SetBuffer(_temporalKernel, CountersId, _counters);
        _surfaceCompute.SetTexture(_temporalKernel, TemporalStateId, _temporalState);
        SetBuffer(_generateLinesKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_generateLinesKernel, VerticesId, _vertices);
        SetBuffer(_generateLinesKernel, LineIndicesId, _lineIndices);
        SetBuffer(_generateLinesKernel, CountersId, _counters);
        SetBuffer(_buildDrawArgsKernel, CountersId, _counters);
        SetBuffer(_buildDrawArgsKernel, DrawArgsId, _drawArgs);
        _surfaceCompute.SetTexture(_initTemporalKernel, TemporalStateId, _temporalState);
    }

    private void SetSurfaceConstants()
    {
        _surfaceCompute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _surfaceCompute.SetInt(TotalVoxelCountId, _totalVoxelCount);
        _surfaceCompute.SetInt(MaximumVerticesId, _maximumVertices);
        _surfaceCompute.SetInt(MaximumLineIndicesId, _maximumLineIndices);
        _surfaceCompute.SetFloat(VoxelSizeId, voxelSizeMeters);
        _surfaceCompute.SetFloat(MinimumWeightId, minimumMeshWeight);
        _surfaceCompute.SetFloat("_SmoothLambda", smoothLambda);
        _surfaceCompute.SetFloat("_SmoothBeta", smoothBeta);
        _surfaceCompute.SetFloat("_TemporalAlphaMaximum", temporalAlphaMaximum);
        _surfaceCompute.SetFloat("_TemporalAlphaMinimum", temporalAlphaMinimum);
        _surfaceCompute.SetFloat("_TemporalDecayRate", temporalDecayRate);
        _surfaceCompute.SetFloat("_ConvergenceThreshold", convergenceThresholdMeters);
        _surfaceCompute.SetFloat("_TemporalDeadzone", temporalDeadzoneMeters);
    }

    private void SetBuffer(int kernel, int property, GraphicsBuffer buffer)
    {
        _surfaceCompute.SetBuffer(kernel, property, buffer);
    }

    private void DispatchVolume(ComputeShader compute, int kernel)
    {
        compute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        uint tx;
        uint ty;
        uint tz;
        compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
        compute.Dispatch(
            kernel,
            Mathf.CeilToInt(volumeSizeX / (float)tx),
            Mathf.CeilToInt(volumeSizeY / (float)ty),
            Mathf.CeilToInt(volumeSizeZ / (float)tz));
    }

    private RenderTexture CreateVolume(string textureName, GraphicsFormat format)
    {
        RenderTexture texture = new RenderTexture(volumeSizeX, volumeSizeY, 0)
        {
            name = textureName,
            dimension = TextureDimension.Tex3D,
            volumeDepth = volumeSizeZ,
            graphicsFormat = format,
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false
        };
        texture.Create();
        return texture;
    }

    private void OnDestroy()
    {
        ReleaseResources();
    }

    private void ReleaseResources()
    {
        ReleaseBuffer(ref _coordVertexMap);
        ReleaseBuffer(ref _vertices);
        ReleaseBuffer(ref _lineIndices);
        ReleaseBuffer(ref _counters);
        ReleaseBuffer(ref _dispatchArgs);
        ReleaseBuffer(ref _drawArgs);
        ReleaseBuffer(ref _smoothPositionA);
        ReleaseBuffer(ref _smoothPositionB);
        ReleaseTexture(ref _tsdfVolume);
        ReleaseTexture(ref _temporalState);
        if (_wireMaterial != null) Destroy(_wireMaterial);
        _wireMaterial = null;
        _resourcesReady = false;
    }

    private static void ReleaseBuffer(ref GraphicsBuffer buffer)
    {
        if (buffer == null) return;
        buffer.Release();
        buffer = null;
    }

    private static void ReleaseTexture(ref RenderTexture texture)
    {
        if (texture == null) return;
        if (texture.IsCreated()) texture.Release();
        Destroy(texture);
        texture = null;
    }
}

using Meta.XR.EnvironmentDepth;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

[DefaultExecutionOrder(-45)]
[DisallowMultipleComponent]
public sealed class ScanCoverDepthPreprocessor : MonoBehaviour
{
    public enum SourceEye
    {
        Left = 0,
        Right = 1,
    }

    [Header("Refs")]
    [SerializeField] private EnvironmentDepthManager environmentDepthManager;
    [SerializeField] private ComputeShader computeShader;

    [Header("Sampling")]
    [SerializeField] private SourceEye sourceEye = SourceEye.Right;
    [SerializeField, Min(1)] private int downsample = 2;
    [SerializeField] private bool refreshEveryFrame = true;

    [Header("Validity")]
    [SerializeField, Range(0f, 0.1f)] private float minDepth01 = 0.001f;
    [SerializeField, Range(0f, 0.2f)] private float depthEdgeThreshold01 = 0.02f;
    [SerializeField, Range(0.0001f, 0.2f)] private float depthEdgeSoftness01 = 0.02f;
    [SerializeField, Min(0f)] private float minLinearDepthMeters = 0.35f;
    [SerializeField, Min(0f)] private float maxLinearDepthMeters = 6f;

    [Header("Smoothing")]
    [SerializeField, Min(0f)] private float smoothingDepthDeltaMeters = 0.08f;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public SourceEye CurrentSourceEye => sourceEye;
    public RenderTexture WorldPositionRawTexture => _worldPositionRawTexture;
    public RenderTexture WorldPositionTexture => _worldPositionTexture;
    public RenderTexture WorldNormalTexture => _worldNormalTexture;
    public RenderTexture WorldNormalNeighbourTexture => _worldNormalNeighbourTexture;
    public RenderTexture ObservationMetaTexture => _observationMetaTexture;
    public Vector2Int OutputResolution => _outputResolution;
    public bool IsReady => _isReady;
    public bool RefreshEveryFrame => refreshEveryFrame;
    public string LastIssue { get; private set; }
    public bool HasLastDispatchEyePosition { get; private set; }
    public Vector3 LastDispatchEyePosition { get; private set; }
    public bool HasLastDispatchDepthReprojectionMatrix { get; private set; }
    public Matrix4x4 LastDispatchDepthReprojectionMatrix { get; private set; }
    public int LastSuccessfulRefreshFrame { get; private set; } = -1;
    public double LastSuccessfulRefreshRealtimeSeconds { get; private set; } = -1d;
    public int SuccessfulRefreshCount { get; private set; }

    private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
    private static readonly int ReprojectionMatricesId = Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
    private static readonly int ZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");

    private static readonly int SourceDepthTextureId = Shader.PropertyToID("_SourceDepthTexture");
    private static readonly int InverseReprojectionMatricesId = Shader.PropertyToID("_InverseReprojectionMatrices");
    private static readonly int SourceSizeId = Shader.PropertyToID("_SourceSize");
    private static readonly int OutputSizeId = Shader.PropertyToID("_OutputSize");
    private static readonly int SelectedEyeId = Shader.PropertyToID("_SelectedEye");
    private static readonly int MinDepth01Id = Shader.PropertyToID("_MinDepth01");
    private static readonly int DepthEdgeThresholdId = Shader.PropertyToID("_DepthEdgeThreshold");
    private static readonly int DepthEdgeSoftnessId = Shader.PropertyToID("_DepthEdgeSoftness");
    private static readonly int MinLinearDepthMetersId = Shader.PropertyToID("_MinLinearDepthMeters");
    private static readonly int MaxLinearDepthMetersId = Shader.PropertyToID("_MaxLinearDepthMeters");
    private static readonly int SmoothingDepthDeltaMetersId = Shader.PropertyToID("_SmoothingDepthDeltaMeters");
    private static readonly int CameraWorldPositionId = Shader.PropertyToID("_CameraWorldPosition");
    private static readonly int WorldPositionRawTextureId = Shader.PropertyToID("_WorldPositionRawTexture");
    private static readonly int WorldPositionTextureId = Shader.PropertyToID("_WorldPositionTexture");
    private static readonly int WorldNormalTextureId = Shader.PropertyToID("_WorldNormalTexture");
    private static readonly int WorldNormalNeighbourTextureId = Shader.PropertyToID("_WorldNormalNeighbourTexture");
    private static readonly int ObservationMetaTextureId = Shader.PropertyToID("_ObservationMetaTexture");
    private static readonly int GlobalWorldPositionTextureId = Shader.PropertyToID("_ScanCoverDepthWorldPositionTexture");
    private static readonly int GlobalWorldNormalTextureId = Shader.PropertyToID("_ScanCoverDepthWorldNormalTexture");
    private static readonly int GlobalObservationMetaTextureId = Shader.PropertyToID("_ScanCoverDepthObservationMetaTexture");

    private readonly Matrix4x4[] _inverseReprojectionMatrices = new Matrix4x4[2];
    private RenderTexture _worldPositionRawTexture;
    private RenderTexture _worldPositionTexture;
    private RenderTexture _worldNormalTexture;
    private RenderTexture _worldNormalNeighbourTexture;
    private RenderTexture _observationMetaTexture;
    private Vector2Int _outputResolution;
    private int _kernelIndex = -1;
    private int _neighbourNormalKernelIndex = -1;
    private bool _isReady;

    private void Awake()
    {
        ResolveRefs();
        ResolveKernel();
    }

    private void OnEnable()
    {
        ResolveRefs();
        ResolveKernel();
    }

        private void OnDisable()
        {
            ReleaseOutputs();
            _isReady = false;
        }

    private void Update()
    {
        if (!refreshEveryFrame)
            return;

        RefreshNow();
    }

    [ContextMenu("Refresh Depth Preprocessor")]
    public bool RefreshNow()
    {
        ResolveRefs();
        ResolveKernel();
        _isReady = false;

        if (environmentDepthManager == null)
        {
            return SetIssue("EnvironmentDepthManager is missing.");
        }

        if (!environmentDepthManager.isActiveAndEnabled)
        {
            return SetIssue("EnvironmentDepthManager is present but disabled.");
        }

        if (computeShader == null || _kernelIndex < 0 || _neighbourNormalKernelIndex < 0)
        {
            return SetIssue("ComputeShader or kernel is missing.");
        }

        Texture sourceDepth = Shader.GetGlobalTexture(EnvironmentDepthTextureId);
        if (!environmentDepthManager.IsDepthAvailable && sourceDepth == null)
        {
            return SetIssue("EnvironmentDepthManager has not published depth yet.");
        }

        if (sourceDepth == null)
        {
            return SetIssue("_EnvironmentDepthTexture is null.");
        }

        Matrix4x4[] reprojectionMatrices = Shader.GetGlobalMatrixArray(ReprojectionMatricesId);
        if (reprojectionMatrices == null || reprojectionMatrices.Length < 2)
        {
            return SetIssue("_EnvironmentDepthReprojectionMatrices is unavailable.");
        }

        if (!EnsureOutputs(sourceDepth.width, sourceDepth.height))
        {
            return false;
        }

        _inverseReprojectionMatrices[0] = reprojectionMatrices[0].inverse;
        _inverseReprojectionMatrices[1] = reprojectionMatrices[1].inverse;
        int selectedEye = Mathf.Clamp((int)sourceEye, 0, 1);
        HasLastDispatchDepthReprojectionMatrix = true;
        LastDispatchDepthReprojectionMatrix = reprojectionMatrices[selectedEye];

        computeShader.SetTexture(_kernelIndex, SourceDepthTextureId, sourceDepth);
        computeShader.SetMatrixArray(InverseReprojectionMatricesId, _inverseReprojectionMatrices);
        computeShader.SetVector(SourceSizeId, new Vector4(sourceDepth.width, sourceDepth.height, 0f, 0f));
        computeShader.SetVector(OutputSizeId, new Vector4(_outputResolution.x, _outputResolution.y, 0f, 0f));
        computeShader.SetInt(SelectedEyeId, (int)sourceEye);
        computeShader.SetFloat(MinDepth01Id, minDepth01);
        computeShader.SetFloat(DepthEdgeThresholdId, depthEdgeThreshold01);
        computeShader.SetFloat(DepthEdgeSoftnessId, depthEdgeSoftness01);
        computeShader.SetFloat(MinLinearDepthMetersId, minLinearDepthMeters);
        computeShader.SetFloat(MaxLinearDepthMetersId, maxLinearDepthMeters);
        computeShader.SetFloat(SmoothingDepthDeltaMetersId, smoothingDepthDeltaMeters);
        Camera sourceCamera = Camera.main;
        Vector3 cameraWorldPosition = ResolveSelectedEyeWorldPosition(sourceCamera, out bool hasEyePosition);
        HasLastDispatchEyePosition = hasEyePosition;
        LastDispatchEyePosition = cameraWorldPosition;
        computeShader.SetVector(CameraWorldPositionId, new Vector4(
            cameraWorldPosition.x,
            cameraWorldPosition.y,
            cameraWorldPosition.z,
            1f));
        computeShader.SetTexture(_kernelIndex, WorldPositionRawTextureId, _worldPositionRawTexture);
        computeShader.SetTexture(_kernelIndex, WorldPositionTextureId, _worldPositionTexture);
        computeShader.SetTexture(_kernelIndex, WorldNormalTextureId, _worldNormalTexture);
        computeShader.SetTexture(_kernelIndex, WorldNormalNeighbourTextureId, _worldNormalNeighbourTexture);
        computeShader.SetTexture(_kernelIndex, ObservationMetaTextureId, _observationMetaTexture);

        uint threadX;
        uint threadY;
        uint threadZ;
        computeShader.GetKernelThreadGroupSizes(_kernelIndex, out threadX, out threadY, out threadZ);
        int dispatchX = Mathf.CeilToInt(_outputResolution.x / (float)threadX);
        int dispatchY = Mathf.CeilToInt(_outputResolution.y / (float)threadY);
        computeShader.Dispatch(_kernelIndex, dispatchX, dispatchY, 1);

        computeShader.SetVector(OutputSizeId, new Vector4(_outputResolution.x, _outputResolution.y, 0f, 0f));
        computeShader.SetTexture(_neighbourNormalKernelIndex, WorldPositionRawTextureId, _worldPositionRawTexture);
        computeShader.SetTexture(_neighbourNormalKernelIndex, WorldPositionTextureId, _worldPositionTexture);
        computeShader.SetTexture(_neighbourNormalKernelIndex, WorldNormalNeighbourTextureId, _worldNormalNeighbourTexture);
        computeShader.GetKernelThreadGroupSizes(_neighbourNormalKernelIndex, out threadX, out threadY, out threadZ);
        dispatchX = Mathf.CeilToInt(_outputResolution.x / (float)threadX);
        dispatchY = Mathf.CeilToInt(_outputResolution.y / (float)threadY);
        computeShader.Dispatch(_neighbourNormalKernelIndex, dispatchX, dispatchY, 1);

        Shader.SetGlobalTexture(GlobalWorldPositionTextureId, _worldPositionTexture);
        Shader.SetGlobalTexture(GlobalWorldNormalTextureId, _worldNormalTexture);
        Shader.SetGlobalTexture(GlobalObservationMetaTextureId, _observationMetaTexture);

        _isReady = true;
        LastIssue = null;
        LastSuccessfulRefreshFrame = Time.frameCount;
        LastSuccessfulRefreshRealtimeSeconds = Time.realtimeSinceStartupAsDouble;
        SuccessfulRefreshCount++;

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverDepthPreprocessor] eye={sourceEye}, source={sourceDepth.width}x{sourceDepth.height}, " +
                $"output={_outputResolution.x}x{_outputResolution.y}");
        }

        return true;
    }

    /// <summary>
    /// Creates a deterministic stereo companion from the exact preprocessor used
    /// by the production raw-depth source.  This avoids scene-wide runtime searches
    /// while keeping both eyes on identical depth, filtering and confidence rules.
    /// </summary>
    public bool ConfigureAsStereoCompanion(
        ScanCoverDepthPreprocessor productionTemplate,
        SourceEye eye)
    {
        if (productionTemplate == null)
            return SetIssue("Production depth preprocessor template is missing.");

        environmentDepthManager = productionTemplate.environmentDepthManager;
        computeShader = productionTemplate.computeShader;
        sourceEye = eye;
        downsample = productionTemplate.downsample;
        refreshEveryFrame = false;
        minDepth01 = productionTemplate.minDepth01;
        depthEdgeThreshold01 = productionTemplate.depthEdgeThreshold01;
        depthEdgeSoftness01 = productionTemplate.depthEdgeSoftness01;
        minLinearDepthMeters = productionTemplate.minLinearDepthMeters;
        maxLinearDepthMeters = productionTemplate.maxLinearDepthMeters;
        smoothingDepthDeltaMeters = productionTemplate.smoothingDepthDeltaMeters;
        debugLog = false;
        ResolveRefs();
        ResolveKernel();
        LastIssue = null;
        return computeShader != null && _kernelIndex >= 0 && _neighbourNormalKernelIndex >= 0;
    }

    public void SetSourceEye(SourceEye eye)
    {
        sourceEye = eye;
    }

    public void SetRefreshEveryFrame(bool enabled)
    {
        refreshEveryFrame = enabled;
    }

    public bool TryGetOutputs(
            out RenderTexture worldPositionTexture,
        out RenderTexture worldNormalTexture,
        out RenderTexture observationMetaTexture)
    {
        worldPositionTexture = _worldPositionTexture;
        worldNormalTexture = _worldNormalTexture;
        observationMetaTexture = _observationMetaTexture;
        return _isReady &&
            worldPositionTexture != null &&
            worldNormalTexture != null &&
            observationMetaTexture != null;
        }

    public bool TryGetPaperShadowOutputs(
        out RenderTexture worldPositionRawTexture,
        out RenderTexture worldNormalNeighbourTexture)
    {
        worldPositionRawTexture = _worldPositionRawTexture;
        worldNormalNeighbourTexture = _worldNormalNeighbourTexture;
        return _isReady &&
            worldPositionRawTexture != null &&
            worldNormalNeighbourTexture != null;
    }

        public void ReleaseOutputs()
        {
        Shader.SetGlobalTexture(GlobalWorldPositionTextureId, null);
        Shader.SetGlobalTexture(GlobalWorldNormalTextureId, null);
        Shader.SetGlobalTexture(GlobalObservationMetaTextureId, null);
        ReleaseTexture(ref _worldPositionRawTexture);
        ReleaseTexture(ref _worldPositionTexture);
        ReleaseTexture(ref _worldNormalTexture);
        ReleaseTexture(ref _worldNormalNeighbourTexture);
        ReleaseTexture(ref _observationMetaTexture);
        _outputResolution = Vector2Int.zero;
    }

    private void ResolveRefs()
    {
        if (environmentDepthManager == null)
            environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>();
    }

    private void ResolveKernel()
    {
        if (computeShader == null)
            return;

        if (_kernelIndex < 0)
            _kernelIndex = computeShader.FindKernel("CSMain");
        if (_neighbourNormalKernelIndex < 0)
            _neighbourNormalKernelIndex = computeShader.FindKernel("CSNeighbourNormals");
    }

    private bool EnsureOutputs(int sourceWidth, int sourceHeight)
    {
        int safeDownsample = Mathf.Max(1, downsample);
        Vector2Int wantedResolution = new Vector2Int(
            Mathf.Max(1, sourceWidth / safeDownsample),
            Mathf.Max(1, sourceHeight / safeDownsample));

        if (_outputResolution == wantedResolution &&
            _worldPositionRawTexture != null &&
            _worldPositionTexture != null &&
            _worldNormalTexture != null &&
            _worldNormalNeighbourTexture != null &&
            _observationMetaTexture != null)
        {
            return true;
        }

        ReleaseOutputs();

        _worldPositionRawTexture = CreateOutputTexture(
            "ScanCover_DepthWorldPositionRaw",
            wantedResolution,
            GraphicsFormat.R32G32B32A32_SFloat);
        _worldPositionTexture = CreateOutputTexture(
            "ScanCover_DepthWorldPosition",
            wantedResolution,
            GraphicsFormat.R32G32B32A32_SFloat);
        _worldNormalTexture = CreateOutputTexture(
            "ScanCover_DepthWorldNormal",
            wantedResolution,
            GraphicsFormat.R32G32B32A32_SFloat);
        _worldNormalNeighbourTexture = CreateOutputTexture(
            "ScanCover_DepthWorldNormalNeighbour",
            wantedResolution,
            GraphicsFormat.R32G32B32A32_SFloat);
        _observationMetaTexture = CreateOutputTexture(
            "ScanCover_DepthObservationMeta",
            wantedResolution,
            GraphicsFormat.R32G32B32A32_SFloat);

        _outputResolution = wantedResolution;
        return _worldPositionRawTexture != null &&
            _worldPositionTexture != null &&
            _worldNormalTexture != null &&
            _worldNormalNeighbourTexture != null &&
            _observationMetaTexture != null;
    }

    private Vector3 ResolveSelectedEyeWorldPosition(Camera sourceCamera, out bool valid)
    {
        valid = false;
        if (sourceCamera == null)
            return Vector3.zero;

        try
        {
            Camera.StereoscopicEye eye = sourceEye == SourceEye.Left
                ? Camera.StereoscopicEye.Left
                : Camera.StereoscopicEye.Right;
            Matrix4x4 eyeToWorld = sourceCamera.GetStereoViewMatrix(eye).inverse;
            Vector4 column = eyeToWorld.GetColumn(3);
            Vector3 position = new Vector3(column.x, column.y, column.z);
            if (float.IsFinite(position.x) && float.IsFinite(position.y) && float.IsFinite(position.z))
            {
                valid = true;
                return position;
            }
        }
        catch
        {
            // Non-stereo editor cameras fall back to the camera transform.
        }

        return sourceCamera.transform.position;
    }

    private static RenderTexture CreateOutputTexture(string textureName, Vector2Int size, GraphicsFormat format)
    {
        var texture = new RenderTexture(size.x, size.y, 0)
        {
            name = textureName,
            enableRandomWrite = true,
            graphicsFormat = format,
            dimension = UnityEngine.Rendering.TextureDimension.Tex2D,
            volumeDepth = 1,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
        };

        texture.Create();
        return texture;
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
        if (debugLog)
            Debug.LogWarning($"[ScanCoverDepthPreprocessor] {issue}");
        return false;
    }
}

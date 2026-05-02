using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(Renderer))]
public sealed class ScanCoverDepthPreprocessorDebugView : MonoBehaviour
{
    public enum ViewMode
    {
        ObservationValidity = 0,
        ObservationConfidence = 1,
        LinearDepth01 = 2,
        WorldNormal = 3,
        WorldPosition = 4,
        NormalValidity = 5,
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverDepthPreprocessor preprocessor;
    [SerializeField] private Shader debugShader;

    [Header("Display")]
    [SerializeField] private ViewMode viewMode = ViewMode.ObservationConfidence;
    [SerializeField] private bool refreshEveryFrame = true;
    [SerializeField] private bool hideWhenUnavailable = true;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    private static readonly int WorldPositionTextureId = Shader.PropertyToID("_ScanCoverDepthWorldPositionTexture");
    private static readonly int WorldNormalTextureId = Shader.PropertyToID("_ScanCoverDepthWorldNormalTexture");
    private static readonly int ObservationMetaTextureId = Shader.PropertyToID("_ScanCoverDepthObservationMetaTexture");
    private static readonly int ViewModeId = Shader.PropertyToID("_ViewMode");

    private Renderer _targetRenderer;
    private Material _runtimeMaterial;

    private void Awake()
    {
        ResolveRefs();
        EnsureMaterial();
        RefreshNow();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureMaterial();
        RefreshNow();
    }

    private void OnDisable()
    {
        ApplyVisible(false);
    }

    private void OnDestroy()
    {
        if (_runtimeMaterial != null)
        {
            Object.Destroy(_runtimeMaterial);
            _runtimeMaterial = null;
        }
    }

    private void Update()
    {
        if (!refreshEveryFrame)
            return;

        RefreshNow();
    }

    [ContextMenu("Refresh Debug View")]
    public void RefreshNow()
    {
        ResolveRefs();
        EnsureMaterial();

        if (_targetRenderer == null || _runtimeMaterial == null)
            return;

        if (preprocessor == null)
        {
            ApplyVisible(!hideWhenUnavailable);
            return;
        }

        bool available = preprocessor.TryGetOutputs(
            out RenderTexture worldPositionTexture,
            out RenderTexture worldNormalTexture,
            out RenderTexture observationMetaTexture);

        if (!available)
        {
            ApplyVisible(!hideWhenUnavailable);
            return;
        }

        _runtimeMaterial.SetTexture(WorldPositionTextureId, worldPositionTexture);
        _runtimeMaterial.SetTexture(WorldNormalTextureId, worldNormalTexture);
        _runtimeMaterial.SetTexture(ObservationMetaTextureId, observationMetaTexture);
        _runtimeMaterial.SetFloat(ViewModeId, (float)viewMode);
        ApplyVisible(true);

        if (debugLog)
        {
            Debug.Log($"[ScanCoverDepthPreprocessorDebugView] mode={viewMode}");
        }
    }

    private void ResolveRefs()
    {
        if (_targetRenderer == null)
            _targetRenderer = GetComponent<Renderer>();

        if (preprocessor == null)
            preprocessor = FindAnyObjectByType<ScanCoverDepthPreprocessor>();
    }

    private void EnsureMaterial()
    {
        if (_targetRenderer == null)
            return;

        if (debugShader == null)
            debugShader = Shader.Find("MRMotifs/ScanCover/DepthPreprocessorDebug");

        if (debugShader == null)
            return;

        if (_runtimeMaterial == null)
        {
            _runtimeMaterial = new Material(debugShader)
            {
                name = "ScanCoverDepthPreprocessorDebugView_Mat"
            };
            _targetRenderer.sharedMaterial = _runtimeMaterial;
        }
    }

    private void ApplyVisible(bool visible)
    {
        if (_targetRenderer != null)
            _targetRenderer.enabled = visible;
    }
}

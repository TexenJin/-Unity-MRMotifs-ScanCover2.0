using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-19)]
[DisallowMultipleComponent]
public sealed class ScanCoverRawDepthDebugWindow : MonoBehaviour
{
    private enum SourceEye
    {
        Left = 0,
        Right = 1
    }

    [Header("Refs")]
    [SerializeField] private Camera sourceCamera;
    [SerializeField] private Transform followTarget;
    [SerializeField] private Shader rawDepthPreviewShader;

    [Header("Raw Depth")]
    [SerializeField] private SourceEye sourceEye = SourceEye.Right;
    [SerializeField, Min(0.001f)] private float rawDepthDisplayScale = 1f;
    [SerializeField, Range(0f, 1f)] private float previewAlpha = 0.86f;

    [Header("Window")]
    [SerializeField] private bool showWindow = true;
    [SerializeField] private Vector3 localPosition = new Vector3(0f, 0.08f, 0.18f);
    [SerializeField, Min(0.03f)] private float sizeMeters = 0.22f;
    [SerializeField] private bool faceSourceCamera = true;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
    private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
    private static readonly int EyeIndexId = Shader.PropertyToID("_EyeIndex");
    private static readonly int RawDepthDisplayScaleId = Shader.PropertyToID("_RawDepthDisplayScale");
    private static readonly int AlphaId = Shader.PropertyToID("_Alpha");

    private GameObject _root;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;
    private Material _material;
    private int _lastLoggedFrame = -1;
    private ComputeShader _rawDepthProbeShader;
    private ComputeBuffer _rawDepthProbeBuffer;
    private bool _hasPendingReadback;

    private void Reset()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureResources();
    }

    private void Update()
    {
        ResolveRefs();
        EnsureResources();
        UpdateWindow();
    }

    private void OnDestroy()
    {
        if (_material != null)
            Destroy(_material);
        if (_mesh != null)
            Destroy(_mesh);
        if (_root != null)
            Destroy(_root);
        if (_rawDepthProbeBuffer != null)
        {
            _rawDepthProbeBuffer.Release();
            _rawDepthProbeBuffer = null;
        }
    }

    private void ResolveRefs()
    {
        if (sourceCamera == null)
            sourceCamera = Camera.main;
        if (followTarget == null)
            followTarget = ResolveRightHandFollowTarget();
        if (rawDepthPreviewShader == null)
            rawDepthPreviewShader = Shader.Find("Hidden/ScanCover/RawDepthDebugWindow");
    }

    private void EnsureResources()
    {
        if (_material == null && rawDepthPreviewShader != null)
            _material = new Material(rawDepthPreviewShader) { name = "ScanCover Raw Depth Debug Window Material" };

        if (_root == null)
        {
            _root = new GameObject("[ScanCover] Raw Depth Debug Window");
            _meshFilter = _root.AddComponent<MeshFilter>();
            _meshRenderer = _root.AddComponent<MeshRenderer>();
            _mesh = BuildQuadMesh();
            _meshFilter.sharedMesh = _mesh;
        }

        if (_meshRenderer != null && _material != null)
            _meshRenderer.sharedMaterial = _material;
    }

    private void UpdateWindow()
    {
        if (_root == null)
            return;

        if (_root.activeSelf != showWindow)
            _root.SetActive(showWindow);
        if (!showWindow)
            return;

        float size = Mathf.Max(0.03f, sizeMeters);
        Transform target = sourceCamera != null ? sourceCamera.transform : (followTarget != null ? followTarget : transform);
        if (_root.transform.parent != null)
            _root.transform.SetParent(null, true);

        _root.transform.position = target.TransformPoint(localPosition);
        _root.transform.rotation = faceSourceCamera && sourceCamera != null ? sourceCamera.transform.rotation : target.rotation;
        _root.transform.localScale = new Vector3(size, size, size);

        Texture rawDepth = Shader.GetGlobalTexture(EnvironmentDepthTextureId);
        if (_material != null)
        {
            _material.SetTexture(EnvironmentDepthTextureId, rawDepth);
            _material.SetFloat(EyeIndexId, (float)sourceEye);
            _material.SetFloat(RawDepthDisplayScaleId, Mathf.Max(0.001f, rawDepthDisplayScale));
            _material.SetFloat(AlphaId, previewAlpha);
        }

        if (debugLog && Time.frameCount - _lastLoggedFrame > 60)
        {
            _lastLoggedFrame = Time.frameCount;
            Debug.Log(rawDepth != null
                ? $"[ScanCoverRawDepthDebugWindow] rawDepth={rawDepth.GetType().Name} size={rawDepth.width}x{rawDepth.height} eye={sourceEye} displayScale={rawDepthDisplayScale:F6}"
                : "[ScanCoverRawDepthDebugWindow] _EnvironmentDepthTexture is missing.",
                this);
            RequestRawDepthStats(rawDepth);
        }
    }

    private void RequestRawDepthStats(Texture rawDepth)
    {
        if (rawDepth == null || _hasPendingReadback)
            return;

        if (_rawDepthProbeShader == null)
            _rawDepthProbeShader = Resources.Load<ComputeShader>("ScanCoverRawEnvironmentDepthProbe");
        if (_rawDepthProbeShader == null)
        {
            Debug.LogWarning("[ScanCoverRawDepthDebugWindow] Missing Resources/ScanCoverRawEnvironmentDepthProbe.compute.", this);
            return;
        }

        const int probeTextureSize = 128;
        int count = probeTextureSize * probeTextureSize * 2;
        if (_rawDepthProbeBuffer == null || _rawDepthProbeBuffer.count != count)
        {
            if (_rawDepthProbeBuffer != null)
                _rawDepthProbeBuffer.Release();
            _rawDepthProbeBuffer = new ComputeBuffer(count, sizeof(float));
        }

        int kernel = _rawDepthProbeShader.FindKernel("CopyRaw");
        _rawDepthProbeShader.SetTexture(kernel, EnvironmentDepthTextureId, rawDepth);
        _rawDepthProbeShader.SetFloat("_EnvironmentDepthTextureSize", rawDepth.width);
        _rawDepthProbeShader.SetBuffer(kernel, "_RawDepthProbe", _rawDepthProbeBuffer);
        _rawDepthProbeShader.Dispatch(kernel, 1, 1, 1);

        _hasPendingReadback = true;
        AsyncGPUReadback.Request(_rawDepthProbeBuffer, request =>
        {
            _hasPendingReadback = false;
            if (request.hasError)
            {
                Debug.LogWarning("[ScanCoverRawDepthDebugWindow] Raw depth GPU readback failed.", this);
                return;
            }

            LogRawDepthStats(request.GetData<float>(), probeTextureSize);
        });
    }

    private void LogRawDepthStats(Unity.Collections.NativeArray<float> data, int textureSize)
    {
        int eyeOffset = (int)sourceEye * textureSize * textureSize;
        int eyePixels = textureSize * textureSize;
        int finite = 0;
        int positive = 0;
        int zeroOrNegative = 0;
        float minRaw = 0f;
        float maxRaw = 0f;
        double sumRaw = 0.0;
        int positiveLinear = 0;
        float minLinear = 0f;
        float maxLinear = 0f;
        double sumLinear = 0.0;
        Vector4 zBufferParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);

        for (int i = 0; i < eyePixels; i++)
        {
            float v = data[eyeOffset + i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                continue;

            finite++;
            if (v <= 0f)
            {
                zeroOrNegative++;
                continue;
            }

            positive++;
            minRaw = positive == 1 ? v : Mathf.Min(minRaw, v);
            maxRaw = positive == 1 ? v : Mathf.Max(maxRaw, v);
            sumRaw += v;

            float linear = RawDepthToLinearMeters(v, zBufferParams);
            if (linear > 0f && !float.IsNaN(linear) && !float.IsInfinity(linear))
            {
                positiveLinear++;
                minLinear = positiveLinear == 1 ? linear : Mathf.Min(minLinear, linear);
                maxLinear = positiveLinear == 1 ? linear : Mathf.Max(maxLinear, linear);
                sumLinear += linear;
            }
        }

        int centerIndex = eyeOffset + textureSize / 2 + (textureSize / 2) * textureSize;
        float centerRaw = data[centerIndex];
        float avgRaw = positive > 0 ? (float)(sumRaw / positive) : 0f;
        float centerLinear = RawDepthToLinearMeters(centerRaw, zBufferParams);
        float avgLinear = positiveLinear > 0 ? (float)(sumLinear / positiveLinear) : 0f;
        float centerMapped = centerLinear > 0f ? Mathf.Pow(Mathf.Clamp01(centerLinear * Mathf.Max(0.001f, rawDepthDisplayScale)), 1.25f) : 0f;

        Debug.Log(
            $"[ScanCoverRawDepthDebugWindow] stats eye={sourceEye} {textureSize}x{textureSize} " +
            $"positive={positive}/{eyePixels} ({(100f * positive / eyePixels):F1}%) finite={finite} zeroOrNegative={zeroOrNegative} " +
            $"rawMinMax={FormatNumber(minRaw)}-{FormatNumber(maxRaw)} rawAvg={FormatNumber(avgRaw)} centerRaw={FormatNumber(centerRaw)} " +
            $"linearMinMax={FormatNumber(minLinear)}m-{FormatNumber(maxLinear)}m linearAvg={FormatNumber(avgLinear)}m centerLinear={FormatNumber(centerLinear)}m " +
            $"centerMapped={centerMapped:F3} displayScale={rawDepthDisplayScale:F6} zParams=({zBufferParams.x:F6},{zBufferParams.y:F6},{zBufferParams.z:F6},{zBufferParams.w:F6})",
            this);
    }

    private static float RawDepthToLinearMeters(float raw, Vector4 zBufferParams)
    {
        if (raw <= 0f || float.IsNaN(raw) || float.IsInfinity(raw))
            return 0f;

        float denominator = raw + zBufferParams.y;
        if (Mathf.Abs(denominator) < 1e-6f)
            return 0f;

        float linear = zBufferParams.x / denominator;
        return linear > 0f && !float.IsNaN(linear) && !float.IsInfinity(linear) ? linear : 0f;
    }

    private static string FormatNumber(float value)
    {
        return value > 0f && !float.IsNaN(value) && !float.IsInfinity(value) ? value.ToString("F6") : "n/a";
    }

    private static Transform ResolveRightHandFollowTarget()
    {
        Transform target = FindActiveTransformByName("RightHandAnchorDetached");
        if (target != null)
            return target;
        target = FindActiveTransformByName("RightControllerInHandAnchor");
        if (target != null)
            return target;
        target = FindActiveTransformByName("RightHandOnControllerAnchor");
        if (target != null)
            return target;
        target = FindActiveTransformByName("RightControllerAnchor");
        if (target != null)
            return target;
        return FindActiveTransformByName("RightHandAnchor");
    }

    private static Transform FindActiveTransformByName(string objectName)
    {
        Transform[] transforms = FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        for (int i = 0; i < transforms.Length; i++)
        {
            Transform candidate = transforms[i];
            if (candidate != null && candidate.name == objectName && candidate.gameObject.activeInHierarchy)
                return candidate;
        }

        for (int i = 0; i < transforms.Length; i++)
        {
            Transform candidate = transforms[i];
            if (candidate != null && candidate.name == objectName)
                return candidate;
        }

        return null;
    }

    private static Mesh BuildQuadMesh()
    {
        Mesh mesh = new Mesh { name = "ScanCover Raw Depth Debug Window Quad" };
        mesh.vertices = new[]
        {
            new Vector3(-0.5f, -0.5f, 0f),
            new Vector3(0.5f, -0.5f, 0f),
            new Vector3(-0.5f, 0.5f, 0f),
            new Vector3(0.5f, 0.5f, 0f)
        };
        mesh.uv = new[]
        {
            new Vector2(0f, 0f),
            new Vector2(1f, 0f),
            new Vector2(0f, 1f),
            new Vector2(1f, 1f)
        };
        mesh.triangles = new[] { 0, 2, 1, 2, 3, 1 };
        mesh.RecalculateBounds();
        return mesh;
    }
}

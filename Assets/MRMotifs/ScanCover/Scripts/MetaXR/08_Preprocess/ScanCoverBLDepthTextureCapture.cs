using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-20)]
[DisallowMultipleComponent]
public sealed class ScanCoverBLDepthTextureCapture : MonoBehaviour
{
    private enum ViewSourceMode
    {
        AssignedCamera,
        CameraMain,
        BestCameraForBLBounds
    }

    [Header("Source")]
    [SerializeField] private ScanCoverDepthGridPointCloud sourceGrid;
    [SerializeField] private Camera sourceCamera;
    [SerializeField] private Shader linearDepthShader;
    [SerializeField] private Shader previewShader;
    [SerializeField] private Shader previewLineShader;

    [Header("Capture")]
    [SerializeField] private ViewSourceMode viewSourceMode = ViewSourceMode.AssignedCamera;
    [SerializeField] private bool useSourceFacingBoundsCamera = true;
    [SerializeField] private bool useDiagnosticBoundsCamera = false;
    [SerializeField, Range(1f, 2f)] private float boundsCameraPadding = 1.08f;
    [SerializeField, Min(8)] private int textureWidth = 64;
    [SerializeField, Min(8)] private int textureHeight = 64;
    [SerializeField, Min(0.01f)] private float nearClipMeters = 0.03f;
    [SerializeField, Min(0.1f)] private float farClipMeters = 8f;
    [SerializeField] private bool captureEveryFrame = true;
    [SerializeField, Min(1)] private int captureFrameInterval = 5;
    [SerializeField] private bool requestReadbackStats = true;
    [SerializeField, Min(1)] private int statsLogFrameInterval = 30;
    [SerializeField] private bool logCameraAndBoundsDiagnostics = true;
    [SerializeField] private bool debugLog = true;

    [Header("Patch Preview")]
    [SerializeField] private bool showPatchPreview = true;
    [SerializeField] private Vector3 previewLocalPosition = new Vector3(0.34f, -0.22f, 0.72f);
    [SerializeField, Min(0.03f)] private float previewSizeMeters = 0.22f;
    [SerializeField, Min(0.01f)] private float previewDepthScaleMeters = 1.2f;
    [SerializeField, Range(0f, 1f)] private float previewAlpha = 0.82f;
    [SerializeField] private bool showProjectedMeshOutline = false;
    [SerializeField] private Color projectedMeshOutlineColor = new Color(1f, 1f, 1f, 0.95f);

    private Material _depthMaterial;
    private Material _previewMaterial;
    private Material _previewLineMaterial;
    private RenderTexture _depthTexture;
    private CommandBuffer _commandBuffer;
    private GameObject _previewRoot;
    private MeshFilter _previewFilter;
    private MeshRenderer _previewRenderer;
    private Mesh _previewMesh;
    private GameObject _previewLineRoot;
    private MeshFilter _previewLineFilter;
    private MeshRenderer _previewLineRenderer;
    private Mesh _previewLineMesh;
    private readonly List<Vector3> _previewLineVertices = new List<Vector3>(32768);
    private readonly List<int> _previewLineIndices = new List<int>(32768);
    private AsyncGPUReadbackRequest _readback;
    private bool _hasPendingReadback;
    private int _lastCaptureFrame = -1;
    private int _lastStatsLogFrame = -1;
    private int _lastDiagnosticsLogFrame = -1;

    public RenderTexture DepthTexture => _depthTexture;
    public Vector2Int Resolution => new Vector2Int(textureWidth, textureHeight);
    public string LastIssue { get; private set; }

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
        if (_hasPendingReadback)
            CompleteReadbackIfReady();

        if (!captureEveryFrame)
            return;

        int interval = Mathf.Max(1, captureFrameInterval);
        if (_lastCaptureFrame >= 0 && Time.frameCount - _lastCaptureFrame < interval)
            return;

        CaptureNow();
    }

    private void OnDisable()
    {
        _hasPendingReadback = false;
    }

    private void OnDestroy()
    {
        if (_commandBuffer != null)
            _commandBuffer.Release();
        if (_depthTexture != null)
            _depthTexture.Release();
        if (_depthMaterial != null)
            Destroy(_depthMaterial);
        if (_previewMaterial != null)
            Destroy(_previewMaterial);
        if (_previewLineMaterial != null)
            Destroy(_previewLineMaterial);
        if (_previewMesh != null)
            Destroy(_previewMesh);
        if (_previewLineMesh != null)
            Destroy(_previewLineMesh);
        if (_previewRoot != null)
            Destroy(_previewRoot);
    }

    [ContextMenu("Capture BL Depth Texture Now")]
    public bool CaptureNow()
    {
        ResolveRefs();
        EnsureResources();

        if (sourceGrid == null)
            return SetIssue("BL depth source grid is missing.");
        if (_depthMaterial == null)
            return SetIssue("Linear depth shader/material is unavailable.");
        if (!sourceGrid.TryGetPreviewSurfaceData(out Mesh sourceMesh, out Transform sourceTransform))
            return SetIssue("BL preview surface mesh is unavailable.");

        Bounds worldBounds = TransformBounds(sourceMesh.bounds, sourceTransform.localToWorldMatrix);
        Camera activeCamera = ResolveActiveCamera(worldBounds);
        if (activeCamera == null)
            return SetIssue("Source camera is missing.");

        if (debugLog && logCameraAndBoundsDiagnostics)
            LogCameraAndBoundsDiagnostics(sourceMesh, sourceTransform, activeCamera, worldBounds);

        Matrix4x4 view = activeCamera.worldToCameraMatrix;
        Matrix4x4 projection = activeCamera.projectionMatrix;
        float nearClip = Mathf.Max(0.001f, nearClipMeters);
        float farClip = Mathf.Max(nearClip + 0.01f, farClipMeters);
        if (activeCamera.nearClipPlane > 0f && activeCamera.farClipPlane > activeCamera.nearClipPlane)
        {
            nearClip = activeCamera.nearClipPlane;
            farClip = activeCamera.farClipPlane;
        }

        if (useSourceFacingBoundsCamera)
            BuildSourceFacingBoundsView(worldBounds, sourceTransform, activeCamera, out view, out projection, out nearClip, out farClip);
        else if (useDiagnosticBoundsCamera)
            BuildDiagnosticBoundsView(worldBounds, activeCamera, out view, out projection, out nearClip, out farClip);

        _depthMaterial.SetFloat("_NearClipMeters", nearClip);
        _depthMaterial.SetFloat("_FarClipMeters", farClip);

        _commandBuffer.Clear();
        _commandBuffer.name = "ScanCover BL Depth Texture Capture";
        _commandBuffer.SetRenderTarget(_depthTexture);
        _commandBuffer.ClearRenderTarget(true, true, Color.clear);
        _commandBuffer.SetViewProjectionMatrices(view, projection);
        _commandBuffer.DrawMesh(sourceMesh, sourceTransform.localToWorldMatrix, _depthMaterial, 0, 0);
        _commandBuffer.SetViewProjectionMatrices(Matrix4x4.identity, Matrix4x4.identity);
        Graphics.ExecuteCommandBuffer(_commandBuffer);
        UpdateProjectedMeshOutline(sourceMesh, sourceTransform, view, projection);

        _lastCaptureFrame = Time.frameCount;
        LastIssue = null;

        if (requestReadbackStats && !_hasPendingReadback)
        {
            _readback = AsyncGPUReadback.Request(_depthTexture, 0);
            _hasPendingReadback = true;
        }

        return true;
    }

    private void ResolveRefs()
    {
        if (sourceGrid == null)
            sourceGrid = GetComponent<ScanCoverDepthGridPointCloud>();
        if (sourceGrid == null)
            sourceGrid = FindAnyObjectByType<ScanCoverDepthGridPointCloud>(FindObjectsInactive.Include);
        if (sourceCamera == null)
            sourceCamera = Camera.main;
        if (linearDepthShader == null)
            linearDepthShader = Shader.Find("Hidden/ScanCover/BLLinearDepthRFloat");
        if (previewShader == null)
            previewShader = Shader.Find("Hidden/ScanCover/BLDepthPatchPreview");
        if (previewLineShader == null)
            previewLineShader = Shader.Find("Hidden/ScanCover/BLDepthPatchLineOverlay");
    }

    private void EnsureResources()
    {
        int width = Mathf.Max(8, textureWidth);
        int height = Mathf.Max(8, textureHeight);
        if (_depthTexture == null || _depthTexture.width != width || _depthTexture.height != height)
        {
            if (_depthTexture != null)
                _depthTexture.Release();

            RenderTextureDescriptor descriptor = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 24)
            {
                msaaSamples = 1,
                sRGB = false,
                useMipMap = false,
                autoGenerateMips = false
            };
            _depthTexture = new RenderTexture(descriptor)
            {
                name = "ScanCover_BLDepthTexture_RFloat",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            _depthTexture.Create();
        }

        if (_commandBuffer == null)
            _commandBuffer = new CommandBuffer { name = "ScanCover BL Depth Texture Capture" };

        if (_depthMaterial == null && linearDepthShader != null)
            _depthMaterial = new Material(linearDepthShader) { name = "ScanCover BL Linear Depth Capture Material" };

        if (_previewMaterial == null && previewShader != null)
            _previewMaterial = new Material(previewShader) { name = "ScanCover BL Depth Patch Preview Material" };

        if (_previewLineMaterial == null && previewLineShader != null)
            _previewLineMaterial = new Material(previewLineShader) { name = "ScanCover BL Depth Patch Line Overlay Material" };

        EnsurePatchPreview();
        UpdatePatchPreview();
    }

    private void EnsurePatchPreview()
    {
        if (!showPatchPreview)
        {
            if (_previewRoot != null && _previewRoot.activeSelf)
                _previewRoot.SetActive(false);
            return;
        }

        if (_previewRoot == null)
        {
            _previewRoot = new GameObject("[ScanCover] BL Depth Patch Preview");
            _previewFilter = _previewRoot.AddComponent<MeshFilter>();
            _previewRenderer = _previewRoot.AddComponent<MeshRenderer>();
            _previewMesh = BuildPreviewQuadMesh();
            _previewFilter.sharedMesh = _previewMesh;
        }

        if (_previewLineRoot == null)
        {
            _previewLineRoot = new GameObject("Projected BL Mesh Outline");
            _previewLineRoot.transform.SetParent(_previewRoot.transform, false);
            _previewLineRoot.transform.localPosition = new Vector3(0f, 0f, -0.001f);
            _previewLineRoot.transform.localRotation = Quaternion.identity;
            _previewLineRoot.transform.localScale = Vector3.one;
            _previewLineFilter = _previewLineRoot.AddComponent<MeshFilter>();
            _previewLineRenderer = _previewLineRoot.AddComponent<MeshRenderer>();
            _previewLineMesh = new Mesh { name = "ScanCover BL Projected Mesh Outline" };
            _previewLineMesh.MarkDynamic();
            _previewLineFilter.sharedMesh = _previewLineMesh;
        }

        if (_previewRenderer != null && _previewMaterial != null)
            _previewRenderer.sharedMaterial = _previewMaterial;
        if (_previewLineRenderer != null && _previewLineMaterial != null)
            _previewLineRenderer.sharedMaterial = _previewLineMaterial;

        if (!_previewRoot.activeSelf)
            _previewRoot.SetActive(true);
        if (_previewLineRoot != null && _previewLineRoot.activeSelf != showProjectedMeshOutline)
            _previewLineRoot.SetActive(showProjectedMeshOutline);
    }

    private void UpdatePatchPreview()
    {
        if (!showPatchPreview || _previewRoot == null)
            return;

        Camera camera = sourceCamera != null ? sourceCamera : Camera.main;
        Transform parent = camera != null ? camera.transform : transform;
        if (_previewRoot.transform.parent != parent)
            _previewRoot.transform.SetParent(parent, false);

        float size = Mathf.Max(0.03f, previewSizeMeters);
        _previewRoot.transform.localPosition = previewLocalPosition;
        _previewRoot.transform.localRotation = Quaternion.identity;
        _previewRoot.transform.localScale = new Vector3(size, size, size);

        if (_previewMaterial != null)
        {
            _previewMaterial.SetTexture("_DepthTex", _depthTexture);
            _previewMaterial.SetFloat("_DepthScaleMeters", Mathf.Max(0.01f, previewDepthScaleMeters));
            _previewMaterial.SetFloat("_Alpha", previewAlpha);
        }

        if (_previewLineMaterial != null)
            _previewLineMaterial.SetColor("_Color", projectedMeshOutlineColor);
    }

    private void UpdateProjectedMeshOutline(Mesh sourceMesh, Transform sourceTransform, Matrix4x4 view, Matrix4x4 projection)
    {
        if (!showPatchPreview || !showProjectedMeshOutline || _previewLineMesh == null || sourceMesh == null || sourceTransform == null)
            return;

        Vector3[] vertices = sourceMesh.vertices;
        int[] triangles = sourceMesh.triangles;
        Matrix4x4 localToClip = projection * view * sourceTransform.localToWorldMatrix;

        _previewLineVertices.Clear();
        _previewLineIndices.Clear();

        for (int i = 0; i + 2 < triangles.Length; i += 3)
        {
            AddProjectedEdge(vertices, localToClip, triangles[i], triangles[i + 1]);
            AddProjectedEdge(vertices, localToClip, triangles[i + 1], triangles[i + 2]);
            AddProjectedEdge(vertices, localToClip, triangles[i + 2], triangles[i]);
        }

        _previewLineMesh.Clear();
        if (_previewLineVertices.Count <= 0)
            return;

        _previewLineMesh.SetVertices(_previewLineVertices);
        _previewLineMesh.SetIndices(_previewLineIndices, MeshTopology.Lines, 0);
        _previewLineMesh.RecalculateBounds();
    }

    private void AddProjectedEdge(Vector3[] vertices, Matrix4x4 localToClip, int a, int b)
    {
        if (a < 0 || b < 0 || a >= vertices.Length || b >= vertices.Length)
            return;

        if (!TryProjectToPreview(vertices[a], localToClip, out Vector3 pa) ||
            !TryProjectToPreview(vertices[b], localToClip, out Vector3 pb))
            return;

        int start = _previewLineVertices.Count;
        _previewLineVertices.Add(pa);
        _previewLineVertices.Add(pb);
        _previewLineIndices.Add(start);
        _previewLineIndices.Add(start + 1);
    }

    private static bool TryProjectToPreview(Vector3 localPosition, Matrix4x4 localToClip, out Vector3 previewPosition)
    {
        Vector4 clip = localToClip * new Vector4(localPosition.x, localPosition.y, localPosition.z, 1f);
        if (Mathf.Abs(clip.w) <= 1e-5f)
        {
            previewPosition = default;
            return false;
        }

        float ndcX = clip.x / clip.w;
        float ndcY = clip.y / clip.w;
        float ndcZ = clip.z / clip.w;
        if (!float.IsFinite(ndcX) || !float.IsFinite(ndcY) || !float.IsFinite(ndcZ))
        {
            previewPosition = default;
            return false;
        }

        if (ndcX < -1.05f || ndcX > 1.05f || ndcY < -1.05f || ndcY > 1.05f)
        {
            previewPosition = default;
            return false;
        }

        previewPosition = new Vector3(ndcX * 0.5f, ndcY * 0.5f, -0.001f);
        return true;
    }

    private static Mesh BuildPreviewQuadMesh()
    {
        Mesh mesh = new Mesh { name = "ScanCover BL Depth Patch Preview Quad" };
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

    private void CompleteReadbackIfReady()
    {
        if (!_readback.done)
            return;

        _hasPendingReadback = false;
        if (_readback.hasError)
        {
            SetIssue("BL depth texture GPU readback failed.");
            return;
        }

        NativeArray<float> data = _readback.GetData<float>();
        int valid = 0;
        float minDepth = float.PositiveInfinity;
        float maxDepth = 0f;
        for (int i = 0; i < data.Length; i++)
        {
            float depth = data[i];
            if (!(depth > 0f) || !float.IsFinite(depth))
                continue;

            valid++;
            if (depth < minDepth) minDepth = depth;
            if (depth > maxDepth) maxDepth = depth;
        }

        if (!debugLog)
            return;

        int logInterval = Mathf.Max(1, statsLogFrameInterval);
        if (_lastStatsLogFrame >= 0 && Time.frameCount - _lastStatsLogFrame < logInterval)
            return;

        _lastStatsLogFrame = Time.frameCount;
        int centerIndex = Mathf.Clamp(textureHeight / 2, 0, textureHeight - 1) * textureWidth + Mathf.Clamp(textureWidth / 2, 0, textureWidth - 1);
        float centerDepth = centerIndex >= 0 && centerIndex < data.Length ? data[centerIndex] : 0f;
        float validRatio = data.Length > 0 ? valid / (float)data.Length : 0f;
        string minText = valid > 0 ? minDepth.ToString("F3") : "n/a";
        string maxText = valid > 0 ? maxDepth.ToString("F3") : "n/a";
        Debug.Log($"[ScanCoverBLDepthTextureCapture] {textureWidth}x{textureHeight} valid={valid}/{data.Length} ({validRatio:P1}) min={minText}m max={maxText}m center={centerDepth:F3}m", this);
    }

    private Camera ResolveActiveCamera(Bounds worldBounds)
    {
        switch (viewSourceMode)
        {
            case ViewSourceMode.CameraMain:
                return Camera.main != null ? Camera.main : sourceCamera;
            case ViewSourceMode.BestCameraForBLBounds:
                return FindBestCameraForBounds(worldBounds) ?? sourceCamera ?? Camera.main;
            default:
                return sourceCamera != null ? sourceCamera : Camera.main;
        }
    }

    private void LogCameraAndBoundsDiagnostics(Mesh sourceMesh, Transform sourceTransform, Camera activeCamera, Bounds worldBounds)
    {
        int logInterval = Mathf.Max(1, statsLogFrameInterval);
        if (_lastDiagnosticsLogFrame >= 0 && Time.frameCount - _lastDiagnosticsLogFrame < logInterval)
            return;

        _lastDiagnosticsLogFrame = Time.frameCount;

        Vector3 cameraPosition = activeCamera.transform.position;
        Vector3 cameraForward = activeCamera.transform.forward;
        Vector3 toBoundsCenter = worldBounds.center - cameraPosition;
        float centerDistance = toBoundsCenter.magnitude;
        float forwardDot = centerDistance > 1e-5f ? Vector3.Dot(cameraForward, toBoundsCenter / centerDistance) : 0f;
        Rect viewportRect = ProjectBoundsToViewport(worldBounds, activeCamera, out int cornersInFront, out int cornersInside);
        bool centerRayMayHitBounds = worldBounds.IntersectRay(new Ray(cameraPosition, cameraForward), out float centerRayBoundsDistance) && centerRayBoundsDistance >= 0f;
        Bounds cameraBounds = TransformBounds(sourceMesh.bounds, activeCamera.worldToCameraMatrix * sourceTransform.localToWorldMatrix);

        Debug.Log(
            $"[ScanCoverBLDepthTextureCapture] diag mode={viewSourceMode} cam={activeCamera.name} pos={cameraPosition.ToString("F3")} " +
            $"fwd={cameraForward.ToString("F3")} near={activeCamera.nearClipPlane:F3} far={activeCamera.farClipPlane:F3} " +
            $"fov={activeCamera.fieldOfView:F1} aspect={activeCamera.aspect:F3} patchCam={useSourceFacingBoundsCamera} diagBoundsCam={useDiagnosticBoundsCamera} " +
            $"meshVerts={sourceMesh.vertexCount} meshTris={sourceMesh.triangles.Length / 3} " +
            $"boundsCenter={worldBounds.center.ToString("F3")} boundsSize={worldBounds.size.ToString("F3")} " +
            $"camBoundsCenter={cameraBounds.center.ToString("F3")} camBoundsSize={cameraBounds.size.ToString("F3")} " +
            $"srcTf={sourceTransform.name} srcPos={sourceTransform.position.ToString("F3")} srcRot={sourceTransform.rotation.eulerAngles.ToString("F1")} " +
            $"centerDist={centerDistance:F3} forwardDot={forwardDot:F3} " +
            $"viewport={FormatViewportRect(viewportRect)} inFront={cornersInFront}/8 inside={cornersInside}/8 " +
            $"centerRayBounds={(centerRayMayHitBounds ? centerRayBoundsDistance.ToString("F3") + "m" : "miss")}",
            this);

        LogSourceTransformDiagnostics(sourceTransform, activeCamera);
        LogCameraCandidates(worldBounds);
    }

    private void LogSourceTransformDiagnostics(Transform sourceTransform, Camera activeCamera)
    {
        if (sourceTransform == null || activeCamera == null)
            return;

        Transform parent = sourceTransform.parent;
        Transform cameraTransform = activeCamera.transform;
        float sourceForwardDotCameraForward = Vector3.Dot(sourceTransform.forward, cameraTransform.forward);
        float sourceRightDotCameraRight = Vector3.Dot(sourceTransform.right, cameraTransform.right);
        float sourceUpDotCameraUp = Vector3.Dot(sourceTransform.up, cameraTransform.up);
        Vector3 sourceToCamera = cameraTransform.position - sourceTransform.position;
        float sourceForwardDotToCamera = sourceToCamera.sqrMagnitude > 1e-8f
            ? Vector3.Dot(sourceTransform.forward, sourceToCamera.normalized)
            : 0f;

        Debug.Log(
            $"[ScanCoverBLDepthTextureCapture] sourceTf name={sourceTransform.name} parent={(parent != null ? parent.name : "null")} " +
            $"localPos={sourceTransform.localPosition.ToString("F3")} localRot={sourceTransform.localRotation.eulerAngles.ToString("F1")} " +
            $"worldPos={sourceTransform.position.ToString("F3")} worldRot={sourceTransform.rotation.eulerAngles.ToString("F1")} " +
            $"axisDot camF={sourceForwardDotCameraForward:F3} camR={sourceRightDotCameraRight:F3} camU={sourceUpDotCameraUp:F3} " +
            $"sourceFToCam={sourceForwardDotToCamera:F3}",
            this);
    }

    private Camera FindBestCameraForBounds(Bounds worldBounds)
    {
        Camera[] cameras = FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        Camera best = null;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < cameras.Length; i++)
        {
            Camera candidate = cameras[i];
            if (candidate == null || !candidate.enabled || !candidate.gameObject.activeInHierarchy)
                continue;

            float score = ScoreCameraForBounds(candidate, worldBounds, out _, out _, out _, out _);
            if (score > bestScore)
            {
                bestScore = score;
                best = candidate;
            }
        }

        return best;
    }

    private void LogCameraCandidates(Bounds worldBounds)
    {
        Camera[] cameras = FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        Camera bestEnabled = null;
        float bestEnabledScore = float.NegativeInfinity;
        for (int i = 0; i < cameras.Length; i++)
        {
            Camera candidate = cameras[i];
            if (candidate == null)
                continue;

            float score = ScoreCameraForBounds(candidate, worldBounds, out float forwardDot, out Rect viewportRect, out int cornersInFront, out int cornersInside);
            if (candidate.enabled && candidate.gameObject.activeInHierarchy && score > bestEnabledScore)
            {
                bestEnabledScore = score;
                bestEnabled = candidate;
            }

            Debug.Log(
                $"[ScanCoverBLDepthTextureCapture] candidate cam={candidate.name} enabled={candidate.enabled} active={candidate.gameObject.activeInHierarchy} " +
                $"tag={candidate.tag} score={score:F3} forwardDot={forwardDot:F3} viewport={FormatViewportRect(viewportRect)} " +
                $"inFront={cornersInFront}/8 inside={cornersInside}/8 pos={candidate.transform.position.ToString("F3")} fwd={candidate.transform.forward.ToString("F3")}",
                this);
        }

        if (bestEnabled != null)
        {
            float score = ScoreCameraForBounds(bestEnabled, worldBounds, out float forwardDot, out Rect viewportRect, out int cornersInFront, out int cornersInside);
            Debug.Log(
                $"[ScanCoverBLDepthTextureCapture] bestEnabledCamera cam={bestEnabled.name} score={score:F3} forwardDot={forwardDot:F3} " +
                $"viewport={FormatViewportRect(viewportRect)} inFront={cornersInFront}/8 inside={cornersInside}/8",
                this);
        }
    }

    private static float ScoreCameraForBounds(Camera candidate, Bounds worldBounds, out float forwardDot, out Rect viewportRect, out int cornersInFront, out int cornersInside)
    {
        Vector3 toCenter = worldBounds.center - candidate.transform.position;
        float distance = toCenter.magnitude;
        forwardDot = distance > 1e-5f ? Vector3.Dot(candidate.transform.forward, toCenter / distance) : -1f;
        viewportRect = ProjectBoundsToViewport(worldBounds, candidate, out cornersInFront, out cornersInside);

        float insideRatio = cornersInside / 8f;
        float frontRatio = cornersInFront / 8f;
        float forwardScore = Mathf.Clamp01((forwardDot + 1f) * 0.5f);
        float centerPenalty = 0f;
        if (float.IsFinite(viewportRect.center.x) && float.IsFinite(viewportRect.center.y))
            centerPenalty = Mathf.Abs(viewportRect.center.x - 0.5f) + Mathf.Abs(viewportRect.center.y - 0.5f);

        return forwardScore * 2f + insideRatio * 2f + frontRatio - centerPenalty * 0.25f;
    }

    private static void BuildDiagnosticBoundsView(
        Bounds worldBounds,
        Camera referenceCamera,
        out Matrix4x4 view,
        out Matrix4x4 projection,
        out float nearClip,
        out float farClip)
    {
        Vector3 cameraForward = referenceCamera != null ? referenceCamera.transform.forward : Vector3.forward;
        if (cameraForward.sqrMagnitude < 1e-6f)
            cameraForward = Vector3.forward;
        cameraForward.Normalize();

        Vector3 cameraUp = referenceCamera != null ? referenceCamera.transform.up : Vector3.up;
        if (Vector3.Cross(cameraForward, cameraUp).sqrMagnitude < 1e-6f)
            cameraUp = Vector3.up;

        float radius = Mathf.Max(0.05f, worldBounds.extents.magnitude);
        float distance = Mathf.Max(0.25f, radius * 1.6f);
        Vector3 position = worldBounds.center - cameraForward * distance;
        Quaternion rotation = Quaternion.LookRotation(cameraForward, cameraUp);

        nearClip = 0.01f;
        farClip = Mathf.Max(distance + radius * 2.5f, 1f);
        float fov = referenceCamera != null ? referenceCamera.fieldOfView : 90f;
        float aspect = referenceCamera != null ? referenceCamera.aspect : 1f;
        projection = Matrix4x4.Perspective(Mathf.Clamp(fov, 20f, 140f), Mathf.Max(0.01f, aspect), nearClip, farClip);
        view = Matrix4x4.Scale(new Vector3(1f, 1f, -1f)) * Matrix4x4.TRS(position, rotation, Vector3.one).inverse;
    }

    private void BuildSourceFacingBoundsView(
        Bounds worldBounds,
        Transform sourceTransform,
        Camera referenceCamera,
        out Matrix4x4 view,
        out Matrix4x4 projection,
        out float nearClip,
        out float farClip)
    {
        Vector3 patchNormal = sourceTransform != null ? sourceTransform.forward : Vector3.forward;
        if (patchNormal.sqrMagnitude < 1e-6f)
            patchNormal = Vector3.forward;
        patchNormal.Normalize();

        Vector3 up = sourceTransform != null ? sourceTransform.up : Vector3.up;
        if (Vector3.Cross(patchNormal, up).sqrMagnitude < 1e-6f)
            up = referenceCamera != null ? referenceCamera.transform.up : Vector3.up;
        if (Vector3.Cross(patchNormal, up).sqrMagnitude < 1e-6f)
            up = Vector3.up;

        Vector3 right = Vector3.Cross(up, -patchNormal);
        if (right.sqrMagnitude < 1e-6f)
            right = Vector3.right;
        right.Normalize();
        up = Vector3.Cross(-patchNormal, right).normalized;

        Vector3 extents = worldBounds.extents;
        float halfWidth =
            Mathf.Abs(Vector3.Dot(right, Vector3.right)) * extents.x +
            Mathf.Abs(Vector3.Dot(right, Vector3.up)) * extents.y +
            Mathf.Abs(Vector3.Dot(right, Vector3.forward)) * extents.z;
        float halfHeight =
            Mathf.Abs(Vector3.Dot(up, Vector3.right)) * extents.x +
            Mathf.Abs(Vector3.Dot(up, Vector3.up)) * extents.y +
            Mathf.Abs(Vector3.Dot(up, Vector3.forward)) * extents.z;
        float halfDepth =
            Mathf.Abs(Vector3.Dot(patchNormal, Vector3.right)) * extents.x +
            Mathf.Abs(Vector3.Dot(patchNormal, Vector3.up)) * extents.y +
            Mathf.Abs(Vector3.Dot(patchNormal, Vector3.forward)) * extents.z;

        float padding = Mathf.Clamp(boundsCameraPadding, 1f, 2f);
        float aspect = textureHeight > 0 ? textureWidth / (float)textureHeight : 1f;
        float orthoHalfHeight = Mathf.Max(0.01f, halfHeight * padding, (halfWidth * padding) / Mathf.Max(0.01f, aspect));
        float distance = Mathf.Max(0.05f, halfDepth + 0.05f);
        Vector3 position = worldBounds.center + patchNormal * distance;
        Quaternion rotation = Quaternion.LookRotation(-patchNormal, up);

        nearClip = 0.01f;
        farClip = Mathf.Max(0.2f, halfDepth * 2f + 0.2f);
        projection = Matrix4x4.Ortho(
            -orthoHalfHeight * aspect,
            orthoHalfHeight * aspect,
            -orthoHalfHeight,
            orthoHalfHeight,
            nearClip,
            farClip);
        view = Matrix4x4.Scale(new Vector3(1f, 1f, -1f)) * Matrix4x4.TRS(position, rotation, Vector3.one).inverse;

        if (debugLog && logCameraAndBoundsDiagnostics)
        {
            Debug.Log(
                $"[ScanCoverBLDepthTextureCapture] patchView pos={position.ToString("F3")} fwd={(-patchNormal).ToString("F3")} " +
                $"up={up.ToString("F3")} orthoH={orthoHalfHeight:F3} aspect={aspect:F3} near={nearClip:F3} far={farClip:F3} " +
                $"halfW={halfWidth:F3} halfH={halfHeight:F3} halfD={halfDepth:F3}",
                this);
        }
    }

    private static Bounds TransformBounds(Bounds localBounds, Matrix4x4 localToWorld)
    {
        Vector3 center = localToWorld.MultiplyPoint3x4(localBounds.center);
        Vector3 extents = localBounds.extents;
        Vector3 axisX = localToWorld.MultiplyVector(new Vector3(extents.x, 0f, 0f));
        Vector3 axisY = localToWorld.MultiplyVector(new Vector3(0f, extents.y, 0f));
        Vector3 axisZ = localToWorld.MultiplyVector(new Vector3(0f, 0f, extents.z));
        extents.x = Mathf.Abs(axisX.x) + Mathf.Abs(axisY.x) + Mathf.Abs(axisZ.x);
        extents.y = Mathf.Abs(axisX.y) + Mathf.Abs(axisY.y) + Mathf.Abs(axisZ.y);
        extents.z = Mathf.Abs(axisX.z) + Mathf.Abs(axisY.z) + Mathf.Abs(axisZ.z);
        return new Bounds(center, extents * 2f);
    }

    private static Rect ProjectBoundsToViewport(Bounds bounds, Camera camera, out int cornersInFront, out int cornersInside)
    {
        Vector3 min = bounds.min;
        Vector3 max = bounds.max;
        Vector3[] corners =
        {
            new Vector3(min.x, min.y, min.z),
            new Vector3(max.x, min.y, min.z),
            new Vector3(min.x, max.y, min.z),
            new Vector3(max.x, max.y, min.z),
            new Vector3(min.x, min.y, max.z),
            new Vector3(max.x, min.y, max.z),
            new Vector3(min.x, max.y, max.z),
            new Vector3(max.x, max.y, max.z)
        };

        cornersInFront = 0;
        cornersInside = 0;
        float minX = float.PositiveInfinity;
        float minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float maxY = float.NegativeInfinity;

        for (int i = 0; i < corners.Length; i++)
        {
            Vector3 vp = camera.WorldToViewportPoint(corners[i]);
            if (vp.z <= 0f)
                continue;

            cornersInFront++;
            minX = Mathf.Min(minX, vp.x);
            minY = Mathf.Min(minY, vp.y);
            maxX = Mathf.Max(maxX, vp.x);
            maxY = Mathf.Max(maxY, vp.y);
            if (vp.x >= 0f && vp.x <= 1f && vp.y >= 0f && vp.y <= 1f)
                cornersInside++;
        }

        if (cornersInFront <= 0)
            return new Rect(float.NaN, float.NaN, float.NaN, float.NaN);

        return Rect.MinMaxRect(minX, minY, maxX, maxY);
    }

    private static string FormatViewportRect(Rect rect)
    {
        if (!float.IsFinite(rect.xMin) || !float.IsFinite(rect.yMin) ||
            !float.IsFinite(rect.xMax) || !float.IsFinite(rect.yMax))
            return "n/a";

        return $"({rect.xMin:F2},{rect.yMin:F2})-({rect.xMax:F2},{rect.yMax:F2})";
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverBLDepthTextureCapture] {issue}", this);
        return false;
    }
}

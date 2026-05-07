using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR.EnvironmentDepth;
using MyProject.XR;
using UnityEngine;

/// <summary>
/// Independent first-stage validator for the ScanCover seed confidence plan.
/// This component samples only a small center ROI, estimates whether it is one
/// stable local plane, and renders only diagnostics. It does not build grids,
/// remeshes, lattices, contours, or surface display meshes.
/// </summary>
[DefaultExecutionOrder(-35)]
[DisallowMultipleComponent]
public sealed class ScanCoverSeedConfidencePatch : MonoBehaviour
{
    public enum SeedState
    {
        None = 0,
        Candidate = 1,
        Trusted = 2,
        Stable = 3,
    }

    public enum SamplingCenterMode
    {
        DepthTextureCenter = 0,
        CameraCenterRaycast = 1,
    }

    [Header("Dependencies")]
    [SerializeField] private CustomEnvironmentDepthRaycaster depthRaycaster;
    [SerializeField] private EnvironmentDepthManager environmentDepthManager;
    [SerializeField] private ScanCoverSkeletonSessionController sessionController;
    [SerializeField] private Transform displayParent;
    [SerializeField] private Camera centerCamera;

    [Header("Collection")]
    [SerializeField] private bool collectOnFreeze = true;
    [SerializeField] private bool collectContinuously;
    [SerializeField] private Eye eye = Eye.Right;
    [SerializeField] private SamplingCenterMode samplingCenterMode = SamplingCenterMode.CameraCenterRaycast;
    [Min(4)]
    [SerializeField] private int roiPixels = 24;
    [Min(1)]
    [SerializeField] private int sampleStridePixels = 2;
    [Min(1)]
    [SerializeField] private int requiredFrames = 8;
    [Min(1)]
    [SerializeField] private int maxStoredFrames = 20;

    [Header("Depth Read")]
    [SerializeField] private bool depthPixelVFlip = true;
    [SerializeField] private bool neighborFill = true;
    [Min(0)]
    [SerializeField] private int neighborRadiusPixels = 1;
    [Min(0.01f)]
    [SerializeField] private float minLinearDepthMeters = 0.05f;
    [Min(0.1f)]
    [SerializeField] private float maxLinearDepthMeters = 8f;

    [Header("Seed Thresholds")]
    [Min(0.001f)]
    [SerializeField] private float baseResidualMeters = 0.01f;
    [Min(0f)]
    [SerializeField] private float residualPerMeter = 0.02f;
    [Min(1)]
    [SerializeField] private int candidateFrames = 3;
    [Min(1)]
    [SerializeField] private int trustedFrames = 8;
    [Min(1)]
    [SerializeField] private int stableFrames = 15;
    [Range(0f, 90f)]
    [SerializeField] private float maxNormalChangeDegrees = 15f;
    [Min(0f)]
    [SerializeField] private float maxCenterDriftMeters = 0.03f;
    [Range(0.1f, 1f)]
    [SerializeField] private float minValidRatio = 0.55f;
    [Range(0.1f, 1f)]
    [SerializeField] private float minInlierRatio = 0.70f;

    [Header("Diagnostics")]
    [SerializeField] private bool showDiagnostics = true;
    [Min(0.001f)]
    [SerializeField] private float diagnosticScaleMeters = 0.025f;
    [SerializeField] private Color candidateColor = new Color(0f, 1f, 1f, 1f);
    [SerializeField] private Color trustedColor = new Color(0.1f, 1f, 0.25f, 1f);
    [SerializeField] private Color stableColor = new Color(1f, 1f, 1f, 1f);
    [SerializeField] private Color failedColor = new Color(1f, 0.25f, 0.9f, 1f);
    [SerializeField] private bool debugLog;

    [Header("Export")]
    [SerializeField] private bool saveCaptureToDesktop = true;
    [SerializeField] private string desktopFolderName = "ScanCoverSeedConfidence";
    [Tooltip("Optional absolute folder path. When empty, uses Desktop/ScanCoverSeedConfidence.")]
    [SerializeField] private string exportFolderOverride;
    [SerializeField] private bool saveSummaryJson = true;
    [SerializeField] private bool saveFrameCsv = true;

    public SeedState State { get; private set; } = SeedState.None;
    public int StableFrameCount => _stableRunCount;

    private readonly List<FrameStats> _frames = new List<FrameStats>(20);
    private readonly List<Vector3> _samplePoints = new List<Vector3>(256);
    private readonly List<float> _sampleDepths = new List<float>(256);
    private readonly List<Vector3> _lineVertices = new List<Vector3>(32);
    private readonly List<int> _lineIndices = new List<int>(32);

    private GameObject _diagnosticObject;
    private Mesh _diagnosticMesh;
    private Material _diagnosticMaterial;
    private bool _collecting;
    private int _framesCollectedThisRun;
    private int _stableRunCount;
    private ScanCoverSkeletonSessionController.SessionState _lastSessionState;

    private struct FrameStats
    {
        public bool valid;
        public int sampleCount;
        public int validCount;
        public float validRatio;
        public float inlierRatio;
        public float residualMedian;
        public float residualP80;
        public float residualP95;
        public float depthMedian;
        public float depthP80;
        public float depthP95;
        public float normalJitter;
        public float centerDrift;
        public float bimodalityScore;
        public int sampleCenterX;
        public int sampleCenterY;
        public Vector3 center;
        public Vector3 normal;
        public Vector3 axisU;
        public Vector3 axisV;
    }

    private void Awake()
    {
        ResolveRefs();
        EnsureDiagnosticsObject();
        _lastSessionState = sessionController != null
            ? sessionController.State
            : ScanCoverSkeletonSessionController.SessionState.Idle;
    }

    private void Update()
    {
        if (collectOnFreeze && SessionFreezeTriggered())
            BeginSeedCapture();

        if (collectContinuously && !_collecting)
            BeginSeedCapture();

        if (_collecting || collectContinuously)
            CollectOneFrame();
    }

    [ContextMenu("Begin Seed Confidence Capture")]
    public void BeginSeedCapture()
    {
        ResolveRefs();
        _frames.Clear();
        _framesCollectedThisRun = 0;
        _stableRunCount = 0;
        State = SeedState.None;
        _collecting = true;
        ClearDiagnosticsMesh();
        Log("Seed confidence capture started.");
    }

    [ContextMenu("Clear Seed Diagnostics")]
    public void ClearDiagnostics()
    {
        _frames.Clear();
        _framesCollectedThisRun = 0;
        _stableRunCount = 0;
        _collecting = false;
        State = SeedState.None;
        ClearDiagnosticsMesh();
    }

    private bool SessionFreezeTriggered()
    {
        if (sessionController == null)
            ResolveRefs();

        if (sessionController == null)
            return false;

        var current = sessionController.State;
        bool triggered = current == ScanCoverSkeletonSessionController.SessionState.Frozen &&
                         _lastSessionState != ScanCoverSkeletonSessionController.SessionState.Frozen;
        _lastSessionState = current;
        return triggered;
    }

    private void CollectOneFrame()
    {
        ResolveRefs();
        if (depthRaycaster == null || !depthRaycaster.IsDepthTextureAvailable)
            return;

        depthRaycaster.SetEye(eye);
        if (!TryCaptureFrame(out FrameStats stats))
            return;

        _frames.Add(stats);
        while (_frames.Count > Mathf.Max(1, maxStoredFrames))
            _frames.RemoveAt(0);

        _framesCollectedThisRun++;
        UpdateState(stats);
        UpdateDiagnostics(stats);

        if (debugLog)
        {
            Log($"state={State} stable={_stableRunCount} valid={stats.validRatio:0.00} inlier={stats.inlierRatio:0.00} residualP95={stats.residualP95:0.000}m normalJitter={stats.normalJitter:0.0}deg drift={stats.centerDrift:0.000}m");
        }

        if (!collectContinuously && _framesCollectedThisRun >= requiredFrames)
        {
            _collecting = false;
            SaveCaptureIfNeeded();
        }
    }

    private bool TryCaptureFrame(out FrameStats stats)
    {
        stats = default;
        _samplePoints.Clear();
        _sampleDepths.Clear();

        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        Vector2Int sampleCenter = ResolveSampleCenter(textureSize);
        int half = Mathf.Clamp(roiPixels / 2, 1, textureSize / 2);
        int minX = Mathf.Max(0, sampleCenter.x - half);
        int maxX = Mathf.Min(textureSize - 1, sampleCenter.x + half);
        int minY = Mathf.Max(0, sampleCenter.y - half);
        int maxY = Mathf.Min(textureSize - 1, sampleCenter.y + half);

        int sampleCount = 0;
        var gridValid = new bool[textureSize * textureSize];
        var gridPoints = new Vector3[textureSize * textureSize];

        for (int y = minY; y <= maxY; y += sampleStridePixels)
        {
            for (int x = minX; x <= maxX; x += sampleStridePixels)
            {
                sampleCount++;
                if (TryReadWorldPoint(x, y, out Vector3 worldPoint, out float linearDepth) ||
                    neighborFill && TryReadNeighborWorldPoint(x, y, out worldPoint, out linearDepth))
                {
                    _samplePoints.Add(worldPoint);
                    _sampleDepths.Add(linearDepth);
                    int gridIndex = y * textureSize + x;
                    gridValid[gridIndex] = true;
                    gridPoints[gridIndex] = worldPoint;
                }
            }
        }

        if (_samplePoints.Count < 6 || sampleCount == 0)
            return false;

        Vector3 centerPoint = Average(_samplePoints);
        if (!TryEstimateNormalFromGrid(gridPoints, gridValid, textureSize, minX, maxX, minY, maxY, out Vector3 normal))
            normal = EstimateFallbackNormal(centerPoint);

        if (normal.sqrMagnitude < 0.25f)
            return false;

        normal.Normalize();
        OrientNormalTowardViewer(centerPoint, ref normal);
        Vector3 axisU = BuildAxisU(normal);
        Vector3 axisV = Vector3.Cross(normal, axisU).normalized;

        float[] residuals = new float[_samplePoints.Count];
        float[] depths = new float[_sampleDepths.Count];
        for (int i = 0; i < _samplePoints.Count; i++)
        {
            residuals[i] = Mathf.Abs(Vector3.Dot(_samplePoints[i] - centerPoint, normal));
            depths[i] = _sampleDepths[i];
        }

        Array.Sort(residuals);
        Array.Sort(depths);

        float medianDepth = PercentileSorted(depths, 0.5f);
        float residualLimit = baseResidualMeters + residualPerMeter * Mathf.Max(0f, medianDepth);
        int inliers = 0;
        for (int i = 0; i < residuals.Length; i++)
        {
            if (residuals[i] <= residualLimit)
                inliers++;
        }

        stats.valid = true;
        stats.sampleCount = sampleCount;
        stats.validCount = _samplePoints.Count;
        stats.validRatio = _samplePoints.Count / (float)sampleCount;
        stats.inlierRatio = inliers / (float)_samplePoints.Count;
        stats.residualMedian = PercentileSorted(residuals, 0.5f);
        stats.residualP80 = PercentileSorted(residuals, 0.8f);
        stats.residualP95 = PercentileSorted(residuals, 0.95f);
        stats.depthMedian = medianDepth;
        stats.depthP80 = PercentileSorted(depths, 0.8f);
        stats.depthP95 = PercentileSorted(depths, 0.95f);
        stats.normalJitter = ComputeNormalJitter(normal);
        stats.centerDrift = ComputeCenterDrift(centerPoint);
        stats.bimodalityScore = ComputeBimodalityScore(depths);
        stats.sampleCenterX = sampleCenter.x;
        stats.sampleCenterY = sampleCenter.y;
        stats.center = centerPoint;
        stats.normal = normal;
        stats.axisU = axisU;
        stats.axisV = axisV;
        return true;
    }

    private Vector2Int ResolveSampleCenter(int textureSize)
    {
        if (samplingCenterMode == SamplingCenterMode.CameraCenterRaycast &&
            TryResolveCameraCenterTexCoord(textureSize, out Vector2Int center))
        {
            return center;
        }

        int middle = textureSize / 2;
        return new Vector2Int(middle, middle);
    }

    private bool TryResolveCameraCenterTexCoord(int textureSize, out Vector2Int logicalTexCoord)
    {
        logicalTexCoord = default;
        Camera cam = ResolveCenterCamera();
        if (cam == null || depthRaycaster == null)
            return false;

        Ray ray = cam.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0f));
        var result = depthRaycaster.Raycast02(ray, maxLinearDepthMeters, eye, true);
        if (result.status != DepthRaycastResult.Success || !IsFinite(result.position))
            return false;

        depthRaycaster.SetEye(result.eyeIndex);
        Vector2Int depthTexCoord = depthRaycaster.WorldPosToNonNormalizedTextureCoords02(result.position);
        if (depthTexCoord.x < 0 || depthTexCoord.y < 0 ||
            depthTexCoord.x >= textureSize || depthTexCoord.y >= textureSize)
        {
            return false;
        }

        int logicalY = depthPixelVFlip ? textureSize - 1 - depthTexCoord.y : depthTexCoord.y;
        logicalTexCoord = new Vector2Int(depthTexCoord.x, logicalY);
        return true;
    }

    private void UpdateState(FrameStats stats)
    {
        float residualLimit = baseResidualMeters + residualPerMeter * Mathf.Max(0f, stats.depthMedian);
        bool framePass =
            stats.validRatio >= minValidRatio &&
            stats.inlierRatio >= minInlierRatio &&
            stats.residualP95 <= residualLimit * 2f &&
            stats.normalJitter <= maxNormalChangeDegrees &&
            stats.centerDrift <= maxCenterDriftMeters;

        if (framePass)
            _stableRunCount++;
        else
            _stableRunCount = 0;

        if (_stableRunCount >= stableFrames)
            State = SeedState.Stable;
        else if (_stableRunCount >= trustedFrames)
            State = SeedState.Trusted;
        else if (_stableRunCount >= candidateFrames)
            State = SeedState.Candidate;
        else
            State = SeedState.None;
    }

    private bool TryReadWorldPoint(int x, int y, out Vector3 worldPoint, out float linearDepthMeters)
    {
        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int sampledY = depthPixelVFlip ? textureSize - 1 - y : y;
        var texCoord = new Vector2Int(x, sampledY);
        worldPoint = depthRaycaster.WorldPosAtDepthTexCoord02(texCoord);
        linearDepthMeters = 0f;
        if (!IsFinite(worldPoint))
            return false;

        linearDepthMeters = depthRaycaster.WorldPosToLinearDepth02(worldPoint);
        return linearDepthMeters >= minLinearDepthMeters && linearDepthMeters <= maxLinearDepthMeters;
    }

    private bool TryReadNeighborWorldPoint(int centerX, int centerY, out Vector3 worldPoint, out float linearDepthMeters)
    {
        int radius = Mathf.Max(0, neighborRadiusPixels);
        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        for (int r = 1; r <= radius; r++)
        {
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    if (Mathf.Abs(dx) != r && Mathf.Abs(dy) != r)
                        continue;

                    int x = centerX + dx;
                    int y = centerY + dy;
                    if (x < 0 || y < 0 || x >= textureSize || y >= textureSize)
                        continue;

                    if (TryReadWorldPoint(x, y, out worldPoint, out linearDepthMeters))
                        return true;
                }
            }
        }

        worldPoint = default;
        linearDepthMeters = 0f;
        return false;
    }

    private bool TryEstimateNormalFromGrid(
        Vector3[] gridPoints,
        bool[] gridValid,
        int textureSize,
        int minX,
        int maxX,
        int minY,
        int maxY,
        out Vector3 normal)
    {
        normal = Vector3.zero;
        for (int y = minY; y + sampleStridePixels <= maxY; y += sampleStridePixels)
        {
            for (int x = minX; x + sampleStridePixels <= maxX; x += sampleStridePixels)
            {
                int i00 = y * textureSize + x;
                int i10 = y * textureSize + (x + sampleStridePixels);
                int i01 = (y + sampleStridePixels) * textureSize + x;
                int i11 = (y + sampleStridePixels) * textureSize + (x + sampleStridePixels);

                if (gridValid[i00] && gridValid[i10] && gridValid[i01])
                    AccumulateNormal(gridPoints[i00], gridPoints[i10], gridPoints[i01], ref normal);

                if (gridValid[i10] && gridValid[i11] && gridValid[i01])
                    AccumulateNormal(gridPoints[i10], gridPoints[i11], gridPoints[i01], ref normal);
            }
        }

        if (normal.sqrMagnitude < 1e-6f)
            return false;

        normal.Normalize();
        return true;
    }

    private static void AccumulateNormal(Vector3 a, Vector3 b, Vector3 c, ref Vector3 normalSum)
    {
        Vector3 n = Vector3.Cross(b - a, c - a);
        if (n.sqrMagnitude > 1e-8f)
            normalSum += n.normalized;
    }

    private Vector3 EstimateFallbackNormal(Vector3 centerPoint)
    {
        Camera cam = ResolveCenterCamera();
        Transform viewer = cam != null ? cam.transform : transform;
        Vector3 toViewer = viewer.position - centerPoint;
        return toViewer.sqrMagnitude > 1e-6f ? toViewer.normalized : -viewer.forward;
    }

    private void OrientNormalTowardViewer(Vector3 centerPoint, ref Vector3 normal)
    {
        Camera cam = ResolveCenterCamera();
        Transform viewer = cam != null ? cam.transform : transform;
        Vector3 toViewer = viewer.position - centerPoint;
        if (Vector3.Dot(normal, toViewer) < 0f)
            normal = -normal;
    }

    private Camera ResolveCenterCamera()
    {
        if (centerCamera != null)
            return centerCamera;

        centerCamera = Camera.main;
        if (centerCamera != null)
            return centerCamera;

        return FindAnyObjectByType<Camera>(FindObjectsInactive.Exclude);
    }

    private static Vector3 BuildAxisU(Vector3 normal)
    {
        Vector3 axisU = Vector3.Cross(Vector3.up, normal);
        if (axisU.sqrMagnitude < 1e-6f)
            axisU = Vector3.Cross(Vector3.right, normal);
        return axisU.normalized;
    }

    private float ComputeNormalJitter(Vector3 normal)
    {
        if (_frames.Count == 0)
            return 0f;

        float maxAngle = 0f;
        int start = Mathf.Max(0, _frames.Count - 5);
        for (int i = start; i < _frames.Count; i++)
        {
            if (!_frames[i].valid)
                continue;
            maxAngle = Mathf.Max(maxAngle, Vector3.Angle(_frames[i].normal, normal));
        }
        return maxAngle;
    }

    private float ComputeCenterDrift(Vector3 center)
    {
        if (_frames.Count == 0)
            return 0f;

        float maxDrift = 0f;
        int start = Mathf.Max(0, _frames.Count - 5);
        for (int i = start; i < _frames.Count; i++)
        {
            if (!_frames[i].valid)
                continue;
            maxDrift = Mathf.Max(maxDrift, Vector3.Distance(_frames[i].center, center));
        }
        return maxDrift;
    }

    private static float ComputeBimodalityScore(float[] sortedDepths)
    {
        if (sortedDepths.Length < 3)
            return 0f;

        float maxGap = 0f;
        for (int i = 1; i < sortedDepths.Length; i++)
            maxGap = Mathf.Max(maxGap, sortedDepths[i] - sortedDepths[i - 1]);

        float range = Mathf.Max(0.001f, sortedDepths[sortedDepths.Length - 1] - sortedDepths[0]);
        return maxGap / range;
    }

    private void UpdateDiagnostics(FrameStats stats)
    {
        if (!showDiagnostics)
        {
            ClearDiagnosticsMesh();
            return;
        }

        EnsureDiagnosticsObject();
        _diagnosticObject.SetActive(true);
        ConfigureDiagnosticMaterial();

        _lineVertices.Clear();
        _lineIndices.Clear();

        float s = diagnosticScaleMeters;
        AddLine(stats.center - stats.axisU * s, stats.center + stats.axisU * s);
        AddLine(stats.center - stats.axisV * s, stats.center + stats.axisV * s);
        AddLine(stats.center, stats.center + stats.normal * s * 2.5f);

        Vector3 c0 = stats.center - stats.axisU * s - stats.axisV * s;
        Vector3 c1 = stats.center + stats.axisU * s - stats.axisV * s;
        Vector3 c2 = stats.center + stats.axisU * s + stats.axisV * s;
        Vector3 c3 = stats.center - stats.axisU * s + stats.axisV * s;
        AddLine(c0, c1);
        AddLine(c1, c2);
        AddLine(c2, c3);
        AddLine(c3, c0);

        _diagnosticMesh.Clear();
        _diagnosticMesh.SetVertices(_lineVertices);
        _diagnosticMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);
        _diagnosticMesh.RecalculateBounds();
    }

    private void AddLine(Vector3 a, Vector3 b)
    {
        int index = _lineVertices.Count;
        _lineVertices.Add(a);
        _lineVertices.Add(b);
        _lineIndices.Add(index);
        _lineIndices.Add(index + 1);
    }

    private void EnsureDiagnosticsObject()
    {
        if (_diagnosticObject != null)
            return;

        _diagnosticObject = new GameObject("ScanCover Seed Confidence Diagnostics");
        _diagnosticObject.transform.SetParent(displayParent != null ? displayParent : transform, false);
        _diagnosticObject.transform.position = Vector3.zero;
        _diagnosticObject.transform.rotation = Quaternion.identity;
        _diagnosticObject.transform.localScale = Vector3.one;

        var filter = _diagnosticObject.AddComponent<MeshFilter>();
        var renderer = _diagnosticObject.AddComponent<MeshRenderer>();
        _diagnosticMesh = new Mesh { name = "ScanCover Seed Confidence Diagnostics Mesh" };
        _diagnosticMesh.MarkDynamic();
        filter.sharedMesh = _diagnosticMesh;

        _diagnosticMaterial = CreateLineMaterial();
        renderer.sharedMaterial = _diagnosticMaterial;
        _diagnosticObject.SetActive(false);
    }

    private void ConfigureDiagnosticMaterial()
    {
        if (_diagnosticMaterial == null)
            return;

        Color color = State switch
        {
            SeedState.Stable => stableColor,
            SeedState.Trusted => trustedColor,
            SeedState.Candidate => candidateColor,
            _ => failedColor,
        };

        _diagnosticMaterial.color = color;
        if (_diagnosticMaterial.HasProperty("_BaseColor"))
            _diagnosticMaterial.SetColor("_BaseColor", color);
        if (_diagnosticMaterial.HasProperty("_Color"))
            _diagnosticMaterial.SetColor("_Color", color);
    }

    private static Material CreateLineMaterial()
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Sprites/Default");

        var material = new Material(shader) { name = "ScanCover Seed Confidence Diagnostic Material" };
        material.renderQueue = 5000;
        return material;
    }

    private void ClearDiagnosticsMesh()
    {
        if (_diagnosticMesh != null)
            _diagnosticMesh.Clear();
        if (_diagnosticObject != null)
            _diagnosticObject.SetActive(false);
    }

    private void SaveCaptureIfNeeded()
    {
        if (!saveCaptureToDesktop || _frames.Count == 0)
            return;

        string folder = GetExportFolder();
        if (string.IsNullOrWhiteSpace(folder))
        {
            Debug.LogWarning("[ScanCoverSeedConfidencePatch] Export skipped because no valid folder was resolved.", this);
            return;
        }

        try
        {
            Directory.CreateDirectory(folder);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            string prefix = Path.Combine(folder, $"seed_{stamp}");

            if (saveSummaryJson)
                File.WriteAllText(prefix + "_summary.json", BuildSummaryJson(stamp), Encoding.UTF8);

            if (saveFrameCsv)
                File.WriteAllText(prefix + "_frames.csv", BuildFrameCsv(), Encoding.UTF8);

            Debug.Log($"[ScanCoverSeedConfidencePatch] Exported seed diagnostics to: {folder}", this);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ScanCoverSeedConfidencePatch] Export failed: {ex.Message}", this);
        }
    }

    private string GetExportFolder()
    {
        if (!string.IsNullOrWhiteSpace(exportFolderOverride))
            return exportFolderOverride;

        string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        if (string.IsNullOrWhiteSpace(desktop))
            desktop = Application.persistentDataPath;

        string folderName = string.IsNullOrWhiteSpace(desktopFolderName)
            ? "ScanCoverSeedConfidence"
            : desktopFolderName.Trim();
        return Path.Combine(desktop, folderName);
    }

    private string BuildSummaryJson(string timestamp)
    {
        FrameStats last = _frames[_frames.Count - 1];
        var sb = new StringBuilder(2048);
        sb.AppendLine("{");
        AppendJson(sb, "timestamp", timestamp, true);
        AppendJson(sb, "state", State.ToString(), true);
        AppendJson(sb, "stableFrameCount", _stableRunCount, true);
        AppendJson(sb, "framesCollected", _framesCollectedThisRun, true);
        AppendJson(sb, "storedFrameCount", _frames.Count, true);
        AppendJson(sb, "roiPixels", roiPixels, true);
        AppendJson(sb, "sampleStridePixels", sampleStridePixels, true);
        AppendJson(sb, "requiredFrames", requiredFrames, true);
        AppendJson(sb, "samplingCenterMode", samplingCenterMode.ToString(), true);
        AppendJson(sb, "sampleCenterX", last.sampleCenterX, true);
        AppendJson(sb, "sampleCenterY", last.sampleCenterY, true);
        AppendJson(sb, "validRatio", last.validRatio, true);
        AppendJson(sb, "inlierRatio", last.inlierRatio, true);
        AppendJson(sb, "residualMedian", last.residualMedian, true);
        AppendJson(sb, "residualP80", last.residualP80, true);
        AppendJson(sb, "residualP95", last.residualP95, true);
        AppendJson(sb, "depthMedian", last.depthMedian, true);
        AppendJson(sb, "depthP80", last.depthP80, true);
        AppendJson(sb, "depthP95", last.depthP95, true);
        AppendJson(sb, "normalJitter", last.normalJitter, true);
        AppendJson(sb, "centerDrift", last.centerDrift, true);
        AppendJson(sb, "bimodalityScore", last.bimodalityScore, true);
        AppendJsonVector(sb, "center", last.center, true);
        AppendJsonVector(sb, "normal", last.normal, true);
        AppendJsonVector(sb, "axisU", last.axisU, true);
        AppendJsonVector(sb, "axisV", last.axisV, false);
        sb.AppendLine("}");
        return sb.ToString();
    }

    private string BuildFrameCsv()
    {
        var sb = new StringBuilder(4096);
        sb.AppendLine("frameIndex,state,stableFrameCount,samplingCenterMode,sampleCenterX,sampleCenterY,sampleCount,validCount,validRatio,inlierRatio,residualMedian,residualP80,residualP95,depthMedian,depthP80,depthP95,normalJitter,centerDrift,bimodalityScore,centerX,centerY,centerZ,normalX,normalY,normalZ,axisUX,axisUY,axisUZ,axisVX,axisVY,axisVZ");
        for (int i = 0; i < _frames.Count; i++)
        {
            FrameStats f = _frames[i];
            sb.Append(i).Append(',');
            sb.Append(State).Append(',');
            sb.Append(_stableRunCount).Append(',');
            sb.Append(samplingCenterMode).Append(',');
            AppendCsv(sb, f.sampleCenterX);
            AppendCsv(sb, f.sampleCenterY);
            AppendCsv(sb, f.sampleCount);
            AppendCsv(sb, f.validCount);
            AppendCsv(sb, f.validRatio);
            AppendCsv(sb, f.inlierRatio);
            AppendCsv(sb, f.residualMedian);
            AppendCsv(sb, f.residualP80);
            AppendCsv(sb, f.residualP95);
            AppendCsv(sb, f.depthMedian);
            AppendCsv(sb, f.depthP80);
            AppendCsv(sb, f.depthP95);
            AppendCsv(sb, f.normalJitter);
            AppendCsv(sb, f.centerDrift);
            AppendCsv(sb, f.bimodalityScore);
            AppendCsv(sb, f.center.x);
            AppendCsv(sb, f.center.y);
            AppendCsv(sb, f.center.z);
            AppendCsv(sb, f.normal.x);
            AppendCsv(sb, f.normal.y);
            AppendCsv(sb, f.normal.z);
            AppendCsv(sb, f.axisU.x);
            AppendCsv(sb, f.axisU.y);
            AppendCsv(sb, f.axisU.z);
            AppendCsv(sb, f.axisV.x);
            AppendCsv(sb, f.axisV.y);
            sb.Append(FloatString(f.axisV.z)).AppendLine();
        }
        return sb.ToString();
    }

    private static void AppendCsv(StringBuilder sb, int value)
    {
        sb.Append(value).Append(',');
    }

    private static void AppendCsv(StringBuilder sb, float value)
    {
        sb.Append(FloatString(value)).Append(',');
    }

    private static void AppendJson(StringBuilder sb, string name, string value, bool comma)
    {
        sb.Append("  \"").Append(name).Append("\": \"").Append(value).Append('"');
        sb.AppendLine(comma ? "," : string.Empty);
    }

    private static void AppendJson(StringBuilder sb, string name, int value, bool comma)
    {
        sb.Append("  \"").Append(name).Append("\": ").Append(value);
        sb.AppendLine(comma ? "," : string.Empty);
    }

    private static void AppendJson(StringBuilder sb, string name, float value, bool comma)
    {
        sb.Append("  \"").Append(name).Append("\": ").Append(FloatString(value));
        sb.AppendLine(comma ? "," : string.Empty);
    }

    private static void AppendJsonVector(StringBuilder sb, string name, Vector3 value, bool comma)
    {
        sb.Append("  \"").Append(name).Append("\": { ");
        sb.Append("\"x\": ").Append(FloatString(value.x)).Append(", ");
        sb.Append("\"y\": ").Append(FloatString(value.y)).Append(", ");
        sb.Append("\"z\": ").Append(FloatString(value.z)).Append(" }");
        sb.AppendLine(comma ? "," : string.Empty);
    }

    private static string FloatString(float value)
    {
        return value.ToString("0.######", CultureInfo.InvariantCulture);
    }

    private void ResolveRefs()
    {
        if (environmentDepthManager == null)
            environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);

        if (sessionController == null)
            sessionController = FindAnyObjectByType<ScanCoverSkeletonSessionController>(FindObjectsInactive.Include);

        if (depthRaycaster == null)
            depthRaycaster = FindAnyObjectByType<CustomEnvironmentDepthRaycaster>(FindObjectsInactive.Include);

        if (depthRaycaster == null)
            depthRaycaster = CreateRuntimeRaycaster();

        if (depthRaycaster != null)
        {
            if (environmentDepthManager != null && depthRaycaster.depthManager == null)
                depthRaycaster.depthManager = environmentDepthManager;

            if (!depthRaycaster.gameObject.activeSelf)
                depthRaycaster.gameObject.SetActive(true);
        }
    }

    private CustomEnvironmentDepthRaycaster CreateRuntimeRaycaster()
    {
        var raycasterObject = new GameObject("ScanCover Seed Runtime Depth Raycaster");
        raycasterObject.SetActive(false);
        raycasterObject.transform.SetParent(transform, false);
        var raycaster = raycasterObject.AddComponent<CustomEnvironmentDepthRaycaster>();
        raycaster.depthManager = environmentDepthManager;
        raycasterObject.SetActive(true);
        return raycaster;
    }

    private static Vector3 Average(List<Vector3> points)
    {
        Vector3 sum = Vector3.zero;
        for (int i = 0; i < points.Count; i++)
            sum += points[i];
        return sum / Mathf.Max(1, points.Count);
    }

    private static float PercentileSorted(float[] sortedValues, float percentile)
    {
        if (sortedValues == null || sortedValues.Length == 0)
            return 0f;

        float index = Mathf.Clamp01(percentile) * (sortedValues.Length - 1);
        int low = Mathf.FloorToInt(index);
        int high = Mathf.CeilToInt(index);
        if (low == high)
            return sortedValues[low];
        return Mathf.Lerp(sortedValues[low], sortedValues[high], index - low);
    }

    private static bool IsFinite(Vector3 value)
    {
        return !float.IsNaN(value.x) && !float.IsNaN(value.y) && !float.IsNaN(value.z) &&
               !float.IsInfinity(value.x) && !float.IsInfinity(value.y) && !float.IsInfinity(value.z);
    }

    private void Log(string message)
    {
        if (debugLog)
            Debug.Log($"[ScanCoverSeedConfidencePatch] {message}", this);
    }
}

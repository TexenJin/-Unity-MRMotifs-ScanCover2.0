using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR.EnvironmentDepth;
using MyProject.XR;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverDepthPointBurstWindow : MonoBehaviour
{
    private enum PointBurstDisplayMode
    {
        Points,
        TrustedLattice,
        None
    }

    private enum SampleStatus
    {
        Valid,
        TrustedPlane,
        CandidatePlane,
        DepthEdge,
        EdgeEvidence,
        InvalidDepth
    }

    [Header("Refs")]
    [SerializeField] private CustomEnvironmentDepthRaycaster depthRaycaster;
    [SerializeField] private EnvironmentDepthManager environmentDepthManager;
    [SerializeField] private Transform displayParent;
    [SerializeField] private Camera centerCamera;

    [Header("Sampling Window")]
    [SerializeField] private Eye eye = Eye.Right;
    [SerializeField] private bool useCameraCenterHitPhysicalWindow = true;
    [SerializeField, Min(0.005f)] private float physicalWindowSizeMeters = 0.08f;
    [SerializeField, Min(2)] private int windowPixels = 24;
    [SerializeField, Min(1)] private int samplesPerFrame = 256;
    [SerializeField] private bool stratifiedJitter = true;
    [SerializeField] private bool randomizeBurstSeed = true;
    [SerializeField] private int deterministicSeed = 20260506;
    [SerializeField] private bool depthPixelVFlip = true;
    [SerializeField] private bool neighborFill = true;
    [SerializeField, Min(0)] private int neighborRadiusPixels = 1;
    [SerializeField, Min(0f)] private float minLinearDepthMeters = 0.05f;
    [SerializeField, Min(0.1f)] private float maxLinearDepthMeters = 8f;

    [Header("Sampling Window Display")]
    [SerializeField] private bool showSamplingWindow = true;
    [SerializeField, Min(0.05f)] private float samplingWindowDistanceMeters = 0.75f;
    [SerializeField, Min(0.1f)] private float samplingWindowScale = 1f;
    [SerializeField] private Color samplingWindowColor = new Color(0f, 1f, 1f, 0.9f);

    [Header("Capture")]
    [SerializeField, Min(1)] private int framesPerBurst = 30;
    [SerializeField, Min(1)] private int maxAccumulatedPoints = 20000;
    [SerializeField] private bool clearOnBeginCapture = true;
    [SerializeField] private bool collectContinuously = false;
    [SerializeField] private bool saveWhenBurstCompletes = true;
    [SerializeField] private string desktopFolderName = "ScanCoverDepthPointBurst";

    [Header("Point Display")]
    [SerializeField] private PointBurstDisplayMode displayMode = PointBurstDisplayMode.Points;
    [SerializeField] private bool showPoints = true;
    [SerializeField] private bool renderAsBillboardQuads = true;
    [SerializeField, Min(0.001f)] private float pointVisualSizeMeters = 0.008f;
    [SerializeField, Min(0f)] private float surfaceOffsetTowardCameraMeters = 0.004f;
    [SerializeField] private Color pointColor = new Color(1f, 1f, 1f, 0.95f);
    [SerializeField] private bool recordInvalidSamples = true;
    [SerializeField] private Color invalidPointColor = new Color(0.45f, 0.45f, 0.45f, 0.45f);
    [SerializeField] private bool colorizeValidPointsByPlaneResidual = true;
    [SerializeField] private Color candidatePointColor = new Color(1f, 0.85f, 0f, 0.95f);
    [SerializeField] private Color trustedPointColor = new Color(0.2f, 1f, 0.55f, 0.95f);
    [SerializeField] private Color depthEdgePointColor = new Color(1f, 0.05f, 0.05f, 0.95f);
    [SerializeField] private Color edgeEvidenceColor = new Color(1f, 0.45f, 0f, 0.95f);
    [SerializeField] private Color rejectedPointColor = new Color(1f, 0.1f, 0.1f, 0.9f);
    [SerializeField] private Color unknownPointColor = new Color(0.8f, 0f, 1f, 0.9f);
    [SerializeField, Min(0.001f)] private float trustedResidualBaseMeters = 0.012f;
    [SerializeField, Min(0.001f)] private float candidateResidualBaseMeters = 0.03f;

    [Header("Depth Edge Detection")]
    [SerializeField] private bool enableDepthEdgeDetection = true;
    [SerializeField] private bool excludeDepthEdgeSamplesFromPlaneFit = true;
    [SerializeField, Min(1)] private int depthEdgeProbeRadiusPixels = 1;
    [SerializeField, Min(0.001f)] private float minDepthEdgeJumpMeters = 0.035f;
    [SerializeField, Min(0f)] private float depthEdgeJumpDistanceScale = 0.018f;

    [Header("Trusted Lattice")]
    [SerializeField] private bool showTrustedLattice = false;
    [SerializeField, Min(16)] private int minLatticeSamples = 256;
    [SerializeField, Range(0.1f, 1f)] private float minTrustedInlierRatio = 0.8f;
    [SerializeField, Min(0.001f)] private float maxTrustedResidualP95Meters = 0.03f;
    [SerializeField, Min(0.001f)] private float latticeInlierResidualMeters = 0.02f;
    [SerializeField, Min(0.005f)] private float latticeCellSizeMeters = 0.03f;
    [SerializeField, Min(1)] private int minInliersPerLatticeCell = 2;
    [SerializeField] private Color latticeColor = new Color(1f, 1f, 1f, 0.92f);
    [SerializeField] private bool debugLog = false;

    private readonly List<SampleRecord> _samples = new List<SampleRecord>(4096);
    private readonly List<Vector3> _meshVertices = new List<Vector3>(4096);
    private readonly List<Color> _meshColors = new List<Color>(4096);
    private readonly List<int> _meshIndices = new List<int>(4096);

    private GameObject _displayObject;
    private Mesh _displayMesh;
    private Material _displayMaterial;
    private GameObject _windowDisplayObject;
    private Mesh _windowMesh;
    private Material _windowMaterial;
    private bool _burstActive;
    private int _framesCollectedThisBurst;
    private int _burstId;
    private int _totalFramesCollected;
    private System.Random _random = new System.Random();
    private bool _hasActiveWorldWindow;
    private WorldSamplingWindow _activeWorldWindow;

    private struct SampleRecord
    {
        public int burstId;
        public int burstFrame;
        public int globalFrame;
        public int sampleIndexInFrame;
        public int textureX;
        public int textureY;
        public int windowX;
        public int windowY;
        public float linearDepth;
        public Vector3 worldPosition;
        public float windowLocalU;
        public float windowLocalV;
        public Vector3 windowCenter;
        public Vector3 windowAxisU;
        public Vector3 windowAxisV;
        public float windowSizeMeters;
        public bool fromPhysicalWindow;
        public bool valid;
        public SampleStatus status;
    }

    private struct WorldSamplingWindow
    {
        public bool valid;
        public Vector3 center;
        public Vector3 axisU;
        public Vector3 axisV;
        public Vector3 normal;
        public float sizeMeters;
        public int centerTextureX;
        public int centerTextureY;
        public float centerLinearDepth;
    }

    private struct PlaneFitResult
    {
        public bool valid;
        public Vector3 center;
        public Vector3 normal;
        public Vector3 axisU;
        public Vector3 axisV;
        public float residualMedian;
        public float residualP80;
        public float residualP95;
        public float inlierRatio;
    }

    private void Awake()
    {
        ResolveRefs();
        EnsureDisplayObject();
        EnsureSamplingWindowDisplayObject();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureDisplayObject();
        EnsureSamplingWindowDisplayObject();
    }

    private void Update()
    {
        UpdateSamplingWindowDisplay();

        if (!_burstActive && !collectContinuously)
            return;

        CollectOneFrame();

        if (_burstActive)
        {
            _framesCollectedThisBurst++;
            if (_framesCollectedThisBurst >= Mathf.Max(1, framesPerBurst))
                CompleteBurst();
        }
    }

    [ContextMenu("Begin Point Burst Capture")]
    public void BeginBurstCapture()
    {
        ResolveRefs();
        EnsureDisplayObject();

        if (clearOnBeginCapture)
            ClearCapture();

        _hasActiveWorldWindow = false;
        if (useCameraCenterHitPhysicalWindow && TryBuildWorldSamplingWindow(out WorldSamplingWindow window))
        {
            _activeWorldWindow = window;
            _hasActiveWorldWindow = true;
        }

        _burstId++;
        _framesCollectedThisBurst = 0;
        _burstActive = true;
        _random = randomizeBurstSeed
            ? new System.Random(unchecked(Environment.TickCount ^ (_burstId * 397)))
            : new System.Random(deterministicSeed + _burstId);

        if (debugLog)
            Debug.Log($"[ScanCoverDepthPointBurstWindow] Begin burst {_burstId}, physicalWindow={useCameraCenterHitPhysicalWindow}, window={windowPixels}px/{physicalWindowSizeMeters:F3}m samplesPerFrame={samplesPerFrame} stratifiedJitter={stratifiedJitter}");
    }

    [ContextMenu("Stop Point Burst Capture")]
    public void StopBurstCapture()
    {
        if (!_burstActive)
            return;

        CompleteBurst();
    }

    [ContextMenu("Clear Point Burst Capture")]
    public void ClearCapture()
    {
        _burstActive = false;
        _framesCollectedThisBurst = 0;
        _hasActiveWorldWindow = false;
        _samples.Clear();
        ClearMesh();
    }

    public bool IsCapturing => _burstActive || collectContinuously;
    public int AccumulatedPointCount => _samples.Count;
    public int FramesCollectedThisBurst => _framesCollectedThisBurst;

    private void CollectOneFrame()
    {
        if (depthRaycaster == null)
            ResolveRefs();
        if (depthRaycaster == null || !depthRaycaster.IsDepthTextureAvailable)
            return;

        depthRaycaster.SetEye(eye);

        if (useCameraCenterHitPhysicalWindow)
        {
            WorldSamplingWindow window;
            if (_hasActiveWorldWindow)
            {
                window = _activeWorldWindow;
            }
            else if (!TryBuildWorldSamplingWindow(out window))
            {
                return;
            }

            int physicalGlobalFrame = _totalFramesCollected++;
            int physicalTargetSamples = Mathf.Max(1, samplesPerFrame);
            if (stratifiedJitter)
                CollectPhysicalWindowStratifiedFrame(window, physicalTargetSamples, physicalGlobalFrame);
            else
                CollectPhysicalWindowRandomFrame(window, physicalTargetSamples, physicalGlobalFrame);

            ClassifyValidSamplesForDisplay();
            RebuildPointDisplay();
            return;
        }

        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int center = textureSize / 2;
        int half = Mathf.Clamp(windowPixels / 2, 1, textureSize / 2);
        int minX = Mathf.Max(0, center - half);
        int maxX = Mathf.Min(textureSize - 1, center + half);
        int minY = Mathf.Max(0, center - half);
        int maxY = Mathf.Min(textureSize - 1, center + half);

        int globalFrame = _totalFramesCollected++;
        int targetSamples = Mathf.Max(1, samplesPerFrame);

        if (stratifiedJitter)
            CollectStratifiedJitteredFrame(minX, maxX, minY, maxY, center, targetSamples, globalFrame);
        else
            CollectUniformRandomFrame(minX, maxX, minY, maxY, center, targetSamples, globalFrame);

        ClassifyValidSamplesForDisplay();
        RebuildPointDisplay();
    }

    private bool TryBuildWorldSamplingWindow(out WorldSamplingWindow window)
    {
        window = default;

        if (depthRaycaster == null)
            ResolveRefs();
        if (depthRaycaster == null || !depthRaycaster.IsDepthTextureAvailable)
            return false;

        Camera cam = ResolveCenterCamera();
        if (cam == null)
            return false;

        depthRaycaster.SetEye(eye);

        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        Vector3 centerWorld;
        Vector2Int centerTexCoord;
        float linearDepth;
        Ray centerRay = new Ray(cam.transform.position, cam.transform.forward);
        var centerHit = depthRaycaster.Raycast02(centerRay, maxLinearDepthMeters, eye, true);
        if (centerHit.status == DepthRaycastResult.Success && IsFinite(centerHit.position))
        {
            centerWorld = centerHit.position;
            centerTexCoord = depthRaycaster.WorldPosToNonNormalizedTextureCoords02(centerWorld);
        }
        else
        {
            centerTexCoord = new Vector2Int(textureSize / 2, textureSize / 2);
            centerWorld = depthRaycaster.WorldPosAtDepthTexCoord02(centerTexCoord);
        }

        if (!IsFinite(centerWorld))
            return false;

        linearDepth = depthRaycaster.WorldPosToLinearDepth02(centerWorld);
        if (linearDepth < minLinearDepthMeters || linearDepth > maxLinearDepthMeters)
            return false;

        Vector3 axisU = cam.transform.right;
        Vector3 axisV = cam.transform.up;
        if (!IsFinite(axisU) || axisU.sqrMagnitude < 1e-6f)
            axisU = Vector3.right;
        if (!IsFinite(axisV) || axisV.sqrMagnitude < 1e-6f)
            axisV = Vector3.up;

        axisU.Normalize();
        axisV = Vector3.ProjectOnPlane(axisV, axisU);
        if (axisV.sqrMagnitude < 1e-6f)
            axisV = Vector3.up;
        axisV.Normalize();

        window = new WorldSamplingWindow
        {
            valid = true,
            center = centerWorld,
            axisU = axisU,
            axisV = axisV,
            normal = cam.transform.forward,
            sizeMeters = Mathf.Max(0.005f, physicalWindowSizeMeters * Mathf.Max(0.1f, samplingWindowScale)),
            centerTextureX = centerTexCoord.x,
            centerTextureY = centerTexCoord.y,
            centerLinearDepth = linearDepth
        };
        return true;
    }

    private void CollectPhysicalWindowStratifiedFrame(WorldSamplingWindow window, int targetSamples, int globalFrame)
    {
        int columns = Mathf.CeilToInt(Mathf.Sqrt(targetSamples));
        int rows = Mathf.CeilToInt(targetSamples / (float)columns);

        for (int i = 0; i < targetSamples; i++)
        {
            int col = i % columns;
            int row = i / columns;
            float u01 = (col + NextRandom01()) / columns;
            float v01 = (row + NextRandom01()) / rows;
            float localU = (u01 - 0.5f) * window.sizeMeters;
            float localV = (v01 - 0.5f) * window.sizeMeters;
            TryAddPhysicalWindowSample(window, localU, localV, i, globalFrame);
        }
    }

    private void CollectPhysicalWindowRandomFrame(WorldSamplingWindow window, int targetSamples, int globalFrame)
    {
        for (int i = 0; i < targetSamples; i++)
        {
            float localU = (NextRandom01() - 0.5f) * window.sizeMeters;
            float localV = (NextRandom01() - 0.5f) * window.sizeMeters;
            TryAddPhysicalWindowSample(window, localU, localV, i, globalFrame);
        }
    }

    private bool TryAddPhysicalWindowSample(WorldSamplingWindow window, float localU, float localV, int sampleIndex, int globalFrame)
    {
        if (_samples.Count >= Mathf.Max(1, maxAccumulatedPoints))
            return false;

        Vector3 targetWorld = window.center + window.axisU * localU + window.axisV * localV;
        Vector2Int texCoord = depthRaycaster.WorldPosToNonNormalizedTextureCoords02(targetWorld);
        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        if (texCoord.x < 0 || texCoord.y < 0 || texCoord.x >= textureSize || texCoord.y >= textureSize)
            return AddInvalidPhysicalWindowSample(window, localU, localV, targetWorld, texCoord, sampleIndex, globalFrame);

        if (!TryReadWorldPointAtDepthTexCoord(texCoord, out Vector3 worldPoint, out float linearDepth) &&
            (!neighborFill || !TryReadNeighborWorldPointAtDepthTexCoord(texCoord, out worldPoint, out linearDepth)))
        {
            return AddInvalidPhysicalWindowSample(window, localU, localV, targetWorld, texCoord, sampleIndex, globalFrame);
        }

        bool depthEdge = IsDepthEdgeTexCoord(texCoord, linearDepth);
        _samples.Add(new SampleRecord
        {
            burstId = _burstId,
            burstFrame = _framesCollectedThisBurst,
            globalFrame = globalFrame,
            sampleIndexInFrame = sampleIndex,
            textureX = texCoord.x,
            textureY = texCoord.y,
            windowX = Mathf.RoundToInt(localU * 1000f),
            windowY = Mathf.RoundToInt(localV * 1000f),
            linearDepth = linearDepth,
            worldPosition = worldPoint,
            windowLocalU = localU,
            windowLocalV = localV,
            windowCenter = window.center,
            windowAxisU = window.axisU,
            windowAxisV = window.axisV,
            windowSizeMeters = window.sizeMeters,
            fromPhysicalWindow = true,
            valid = true,
            status = depthEdge ? SampleStatus.DepthEdge : SampleStatus.Valid
        });

        return true;
    }

    private bool AddInvalidPhysicalWindowSample(WorldSamplingWindow window, float localU, float localV, Vector3 targetWorld, Vector2Int texCoord, int sampleIndex, int globalFrame)
    {
        if (!recordInvalidSamples || _samples.Count >= Mathf.Max(1, maxAccumulatedPoints))
            return false;

        _samples.Add(new SampleRecord
        {
            burstId = _burstId,
            burstFrame = _framesCollectedThisBurst,
            globalFrame = globalFrame,
            sampleIndexInFrame = sampleIndex,
            textureX = texCoord.x,
            textureY = texCoord.y,
            windowX = Mathf.RoundToInt(localU * 1000f),
            windowY = Mathf.RoundToInt(localV * 1000f),
            linearDepth = 0f,
            worldPosition = targetWorld,
            windowLocalU = localU,
            windowLocalV = localV,
            windowCenter = window.center,
            windowAxisU = window.axisU,
            windowAxisV = window.axisV,
            windowSizeMeters = window.sizeMeters,
            fromPhysicalWindow = true,
            valid = false,
            status = SampleStatus.InvalidDepth
        });

        return true;
    }

    private void ClassifyValidSamplesForDisplay()
    {
        if (!colorizeValidPointsByPlaneResidual)
        {
            for (int i = 0; i < _samples.Count; i++)
            {
                SampleRecord sample = _samples[i];
                if (sample.valid && sample.status != SampleStatus.DepthEdge)
                {
                    sample.status = SampleStatus.Valid;
                    _samples[i] = sample;
                }
            }
            return;
        }

        if (!TryFitValidSamplePlane(out PlaneFitResult plane, out List<int> validIndices))
            return;

        float spread = Mathf.Max(0.001f, plane.residualP80 - plane.residualMedian);
        float trustedLimit = Mathf.Max(trustedResidualBaseMeters, plane.residualMedian + spread * 2f);
        float candidateLimit = Mathf.Max(candidateResidualBaseMeters, plane.residualMedian + spread * 5f);

        for (int i = 0; i < validIndices.Count; i++)
        {
            int sampleIndex = validIndices[i];
            SampleRecord sample = _samples[sampleIndex];
            if (sample.status == SampleStatus.DepthEdge)
                continue;

            float residual = Mathf.Abs(Vector3.Dot(sample.worldPosition - plane.center, plane.normal));
            if (residual <= trustedLimit)
                sample.status = SampleStatus.TrustedPlane;
            else if (residual <= candidateLimit)
                sample.status = SampleStatus.CandidatePlane;
            else
                sample.status = SampleStatus.EdgeEvidence;

            _samples[sampleIndex] = sample;
        }
    }

    private void CollectStratifiedJitteredFrame(int minX, int maxX, int minY, int maxY, int center, int targetSamples, int globalFrame)
    {
        int columns = Mathf.CeilToInt(Mathf.Sqrt(targetSamples));
        int rows = Mathf.CeilToInt(targetSamples / (float)columns);
        float width = maxX - minX + 1f;
        float height = maxY - minY + 1f;

        for (int i = 0; i < targetSamples; i++)
        {
            int col = i % columns;
            int row = i / columns;
            float u = (col + NextRandom01()) / columns;
            float v = (row + NextRandom01()) / rows;
            int x = Mathf.Clamp(Mathf.FloorToInt(minX + u * width), minX, maxX);
            int y = Mathf.Clamp(Mathf.FloorToInt(minY + v * height), minY, maxY);
            TryAddSample(x, y, center, i, globalFrame);
        }
    }

    private void CollectUniformRandomFrame(int minX, int maxX, int minY, int maxY, int center, int targetSamples, int globalFrame)
    {
        for (int i = 0; i < targetSamples; i++)
        {
            int x = _random.Next(minX, maxX + 1);
            int y = _random.Next(minY, maxY + 1);
            TryAddSample(x, y, center, i, globalFrame);
        }
    }

    private bool TryAddSample(int x, int y, int center, int sampleIndex, int globalFrame)
    {
        if (_samples.Count >= Mathf.Max(1, maxAccumulatedPoints))
            return false;

        if (!TryReadWorldPoint(x, y, out Vector3 worldPoint, out float linearDepth) &&
            (!neighborFill || !TryReadNeighborWorldPoint(x, y, out worldPoint, out linearDepth)))
        {
            return false;
        }

        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int sampledY = depthPixelVFlip ? textureSize - 1 - y : y;
        bool depthEdge = IsDepthEdgeTexCoord(new Vector2Int(x, sampledY), linearDepth);
        _samples.Add(new SampleRecord
        {
            burstId = _burstId,
            burstFrame = _framesCollectedThisBurst,
            globalFrame = globalFrame,
            sampleIndexInFrame = sampleIndex,
            textureX = x,
            textureY = y,
            windowX = x - center,
            windowY = y - center,
            linearDepth = linearDepth,
            worldPosition = worldPoint,
            valid = true,
            status = depthEdge ? SampleStatus.DepthEdge : SampleStatus.Valid
        });

        return true;
    }

    private float NextRandom01()
    {
        return (float)_random.NextDouble();
    }

    private void CompleteBurst()
    {
        _burstActive = false;
        if (saveWhenBurstCompletes)
            SaveCaptureToDesktop();

        if (debugLog)
            Debug.Log($"[ScanCoverDepthPointBurstWindow] Complete burst {_burstId}: frames={_framesCollectedThisBurst}, points={_samples.Count}");
    }

    private bool TryReadWorldPoint(int x, int y, out Vector3 worldPoint, out float linearDepthMeters)
    {
        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int sampledY = depthPixelVFlip ? textureSize - 1 - y : y;
        Vector2Int texCoord = new Vector2Int(x, sampledY);
        return TryReadWorldPointAtDepthTexCoord(texCoord, out worldPoint, out linearDepthMeters);
    }

    private bool TryReadWorldPointAtDepthTexCoord(Vector2Int texCoord, out Vector3 worldPoint, out float linearDepthMeters)
    {
        worldPoint = depthRaycaster.WorldPosAtDepthTexCoord02(texCoord);
        linearDepthMeters = 0f;

        if (!IsFinite(worldPoint))
            return false;

        linearDepthMeters = depthRaycaster.WorldPosToLinearDepth02(worldPoint);
        return linearDepthMeters >= minLinearDepthMeters && linearDepthMeters <= maxLinearDepthMeters;
    }

    private bool TryReadNeighborWorldPointAtDepthTexCoord(Vector2Int centerTexCoord, out Vector3 worldPoint, out float linearDepthMeters)
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

                    int x = centerTexCoord.x + dx;
                    int y = centerTexCoord.y + dy;
                    if (x < 0 || y < 0 || x >= textureSize || y >= textureSize)
                        continue;

                    if (TryReadWorldPointAtDepthTexCoord(new Vector2Int(x, y), out worldPoint, out linearDepthMeters))
                        return true;
                }
            }
        }

        worldPoint = default;
        linearDepthMeters = 0f;
        return false;
    }

    private bool IsDepthEdgeTexCoord(Vector2Int centerTexCoord, float centerLinearDepth)
    {
        if (!enableDepthEdgeDetection || centerLinearDepth <= 0f)
            return false;

        int radius = Mathf.Max(1, depthEdgeProbeRadiusPixels);
        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        float minDepth = centerLinearDepth;
        float maxDepth = centerLinearDepth;
        int validNeighborCount = 0;

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;

                int x = centerTexCoord.x + dx;
                int y = centerTexCoord.y + dy;
                if (x < 0 || y < 0 || x >= textureSize || y >= textureSize)
                    continue;

                if (!TryReadWorldPointAtDepthTexCoord(new Vector2Int(x, y), out _, out float neighborDepth))
                    continue;

                validNeighborCount++;
                if (neighborDepth < minDepth)
                    minDepth = neighborDepth;
                if (neighborDepth > maxDepth)
                    maxDepth = neighborDepth;
            }
        }

        if (validNeighborCount < 2)
            return false;

        float jumpLimit = Mathf.Max(minDepthEdgeJumpMeters, centerLinearDepth * depthEdgeJumpDistanceScale);
        return maxDepth - minDepth > jumpLimit;
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

    private void RebuildPointDisplay()
    {
        EnsureDisplayObject();
        if (_displayMesh == null)
            return;

        bool drawPoints = displayMode == PointBurstDisplayMode.Points && showPoints;
        bool drawLattice = displayMode == PointBurstDisplayMode.TrustedLattice && showTrustedLattice;
        if (!drawPoints && !drawLattice)
        {
            ClearMesh();
            return;
        }

        _meshVertices.Clear();
        _meshColors.Clear();
        _meshIndices.Clear();

        Camera cam = ResolveCenterCamera();
        Transform displayTransform = _displayObject != null ? _displayObject.transform : transform;

        MeshTopology topology;
        if (drawLattice)
        {
            if (!TryAddTrustedLattice(displayTransform, cam))
            {
                ClearMesh();
                return;
            }

            topology = MeshTopology.Lines;
        }
        else
        {
            Vector3 camRight = cam != null ? cam.transform.right : Vector3.right;
            Vector3 camUp = cam != null ? cam.transform.up : Vector3.up;
            Vector3 camPos = cam != null ? cam.transform.position : transform.position;
            Vector3 localRight = displayTransform.InverseTransformDirection(camRight).normalized;
            Vector3 localUp = displayTransform.InverseTransformDirection(camUp).normalized;
            float halfSize = Mathf.Max(0.001f, pointVisualSizeMeters) * 0.5f;

            for (int i = 0; i < _samples.Count; i++)
            {
                SampleRecord sample = _samples[i];
                Vector3 p = sample.worldPosition;
                Color sampleColor = ResolveSampleColor(sample);
                if (sample.valid && surfaceOffsetTowardCameraMeters > 0f)
                {
                    Vector3 towardCamera = camPos - p;
                    if (towardCamera.sqrMagnitude > 1e-8f)
                        p += towardCamera.normalized * surfaceOffsetTowardCameraMeters;
                }

                p = displayTransform.InverseTransformPoint(p);
                if (renderAsBillboardQuads)
                    AddBillboardQuad(p, localRight, localUp, halfSize, sampleColor);
                else
                    AddMeshPoint(p, sampleColor);
            }

            topology = renderAsBillboardQuads ? MeshTopology.Triangles : MeshTopology.Points;
        }

        _displayMesh.Clear();
        _displayMesh.indexFormat = _meshVertices.Count > 65000 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _displayMesh.SetVertices(_meshVertices);
        _displayMesh.SetColors(_meshColors);
        _displayMesh.SetIndices(_meshIndices, topology, 0, true);

        _displayMesh.RecalculateBounds();
    }

    private Color ResolveSampleColor(SampleRecord sample)
    {
        switch (sample.status)
        {
            case SampleStatus.TrustedPlane:
                return trustedPointColor;
            case SampleStatus.CandidatePlane:
                return candidatePointColor;
            case SampleStatus.DepthEdge:
                return depthEdgePointColor;
            case SampleStatus.EdgeEvidence:
                return edgeEvidenceColor;
            case SampleStatus.InvalidDepth:
                return invalidPointColor;
            case SampleStatus.Valid:
            default:
                return sample.valid ? pointColor : unknownPointColor;
        }
    }

    private void AddMeshPoint(Vector3 point, Color color)
    {
        int index = _meshVertices.Count;
        _meshVertices.Add(point);
        _meshColors.Add(color);
        _meshIndices.Add(index);
    }

    private void AddBillboardQuad(Vector3 center, Vector3 right, Vector3 up, float halfSize, Color color)
    {
        int start = _meshVertices.Count;
        Vector3 r = right.normalized * halfSize;
        Vector3 u = up.normalized * halfSize;

        _meshVertices.Add(center - r - u);
        _meshVertices.Add(center - r + u);
        _meshVertices.Add(center + r + u);
        _meshVertices.Add(center + r - u);

        _meshColors.Add(color);
        _meshColors.Add(color);
        _meshColors.Add(color);
        _meshColors.Add(color);

        _meshIndices.Add(start);
        _meshIndices.Add(start + 1);
        _meshIndices.Add(start + 2);
        _meshIndices.Add(start);
        _meshIndices.Add(start + 2);
        _meshIndices.Add(start + 3);
    }

    private bool TryAddTrustedLattice(Transform displayTransform, Camera cam)
    {
        if (_samples.Count < Mathf.Max(16, minLatticeSamples))
            return false;

        if (!TryFitPlane(out PlaneFitResult plane, cam))
            return false;

        if (plane.inlierRatio < minTrustedInlierRatio || plane.residualP95 > maxTrustedResidualP95Meters)
            return false;

        float cellSize = Mathf.Max(0.005f, latticeCellSizeMeters);
        float inlierLimit = Mathf.Max(0.001f, latticeInlierResidualMeters);
        var projected = new List<Vector2>(_samples.Count);
        float minU = float.PositiveInfinity;
        float minV = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float maxV = float.NegativeInfinity;

        for (int i = 0; i < _samples.Count; i++)
        {
            Vector3 delta = _samples[i].worldPosition - plane.center;
            float residual = Mathf.Abs(Vector3.Dot(delta, plane.normal));
            if (residual > inlierLimit)
                continue;

            float u = Vector3.Dot(delta, plane.axisU);
            float v = Vector3.Dot(delta, plane.axisV);
            projected.Add(new Vector2(u, v));
            minU = Mathf.Min(minU, u);
            minV = Mathf.Min(minV, v);
            maxU = Mathf.Max(maxU, u);
            maxV = Mathf.Max(maxV, v);
        }

        if (projected.Count < Mathf.Max(16, minLatticeSamples / 2) ||
            !float.IsFinite(minU) || !float.IsFinite(minV) ||
            maxU - minU < cellSize || maxV - minV < cellSize)
        {
            return false;
        }

        int width = Mathf.Clamp(Mathf.CeilToInt((maxU - minU) / cellSize), 1, 128);
        int height = Mathf.Clamp(Mathf.CeilToInt((maxV - minV) / cellSize), 1, 128);
        var cellCounts = new Dictionary<string, int>(width * height);

        for (int i = 0; i < projected.Count; i++)
        {
            int x = Mathf.Clamp(Mathf.FloorToInt((projected[i].x - minU) / cellSize), 0, width - 1);
            int y = Mathf.Clamp(Mathf.FloorToInt((projected[i].y - minV) / cellSize), 0, height - 1);
            string key = CellKey(x, y);
            cellCounts.TryGetValue(key, out int count);
            cellCounts[key] = count + 1;
        }

        var edges = new HashSet<string>();
        int cellsDrawn = 0;
        foreach (KeyValuePair<string, int> pair in cellCounts)
        {
            if (pair.Value < Mathf.Max(1, minInliersPerLatticeCell))
                continue;

            if (!TryParseCellKey(pair.Key, out int x, out int y))
                continue;

            AddLatticeEdge(x, y, x + 1, y, minU, minV, cellSize, plane, displayTransform, edges);
            AddLatticeEdge(x + 1, y, x + 1, y + 1, minU, minV, cellSize, plane, displayTransform, edges);
            AddLatticeEdge(x + 1, y + 1, x, y + 1, minU, minV, cellSize, plane, displayTransform, edges);
            AddLatticeEdge(x, y + 1, x, y, minU, minV, cellSize, plane, displayTransform, edges);
            cellsDrawn++;
        }

        return cellsDrawn > 0;
    }

    private bool TryFitValidSamplePlane(out PlaneFitResult result, out List<int> validIndices)
    {
        result = default;
        validIndices = new List<int>(_samples.Count);

        for (int i = 0; i < _samples.Count; i++)
        {
            if (_samples[i].valid &&
                (!excludeDepthEdgeSamplesFromPlaneFit || _samples[i].status != SampleStatus.DepthEdge) &&
                IsFinite(_samples[i].worldPosition))
            {
                validIndices.Add(i);
            }
        }

        int count = validIndices.Count;
        if (count < 3)
            return false;

        Vector3 center = Vector3.zero;
        for (int i = 0; i < count; i++)
            center += _samples[validIndices[i]].worldPosition;
        center /= count;

        float xx = 0f;
        float xy = 0f;
        float xz = 0f;
        float yy = 0f;
        float yz = 0f;
        float zz = 0f;

        for (int i = 0; i < count; i++)
        {
            Vector3 d = _samples[validIndices[i]].worldPosition - center;
            xx += d.x * d.x;
            xy += d.x * d.y;
            xz += d.x * d.z;
            yy += d.y * d.y;
            yz += d.y * d.z;
            zz += d.z * d.z;
        }

        float inv = 1f / count;
        xx *= inv;
        xy *= inv;
        xz *= inv;
        yy *= inv;
        yz *= inv;
        zz *= inv;

        if (!TrySmallestEigenVectorSymmetric(xx, xy, xz, yy, yz, zz, out Vector3 normal))
            return false;

        Camera cam = centerCamera != null ? centerCamera : Camera.main;
        if (cam != null && Vector3.Dot(normal, cam.transform.position - center) < 0f)
            normal = -normal;

        Vector3 axisU = BuildPlaneAxisU(normal, cam);
        Vector3 axisV = Vector3.Cross(normal, axisU).normalized;

        var residuals = new List<float>(count);
        int inliers = 0;
        float inlierLimit = Mathf.Max(0.001f, latticeInlierResidualMeters);
        for (int i = 0; i < count; i++)
        {
            float residual = Mathf.Abs(Vector3.Dot(_samples[validIndices[i]].worldPosition - center, normal));
            residuals.Add(residual);
            if (residual <= inlierLimit)
                inliers++;
        }

        residuals.Sort();
        result = new PlaneFitResult
        {
            valid = true,
            center = center,
            normal = normal,
            axisU = axisU,
            axisV = axisV,
            residualMedian = QuantileSorted(residuals, 0.5f),
            residualP80 = QuantileSorted(residuals, 0.8f),
            residualP95 = QuantileSorted(residuals, 0.95f),
            inlierRatio = inliers / (float)count
        };

        return result.valid;
    }

    private bool TryFitPlane(out PlaneFitResult result, Camera cam)
    {
        result = default;
        int count = _samples.Count;
        if (count < 3)
            return false;

        Vector3 center = Vector3.zero;
        for (int i = 0; i < count; i++)
            center += _samples[i].worldPosition;
        center /= count;

        float xx = 0f;
        float xy = 0f;
        float xz = 0f;
        float yy = 0f;
        float yz = 0f;
        float zz = 0f;

        for (int i = 0; i < count; i++)
        {
            Vector3 d = _samples[i].worldPosition - center;
            xx += d.x * d.x;
            xy += d.x * d.y;
            xz += d.x * d.z;
            yy += d.y * d.y;
            yz += d.y * d.z;
            zz += d.z * d.z;
        }

        float inv = 1f / count;
        xx *= inv;
        xy *= inv;
        xz *= inv;
        yy *= inv;
        yz *= inv;
        zz *= inv;

        if (!TrySmallestEigenVectorSymmetric(xx, xy, xz, yy, yz, zz, out Vector3 normal))
            return false;

        if (cam != null && Vector3.Dot(normal, cam.transform.position - center) < 0f)
            normal = -normal;

        Vector3 axisU = BuildPlaneAxisU(normal, cam);
        Vector3 axisV = Vector3.Cross(normal, axisU).normalized;

        var residuals = new List<float>(count);
        int inliers = 0;
        float inlierLimit = Mathf.Max(0.001f, latticeInlierResidualMeters);
        for (int i = 0; i < count; i++)
        {
            float residual = Mathf.Abs(Vector3.Dot(_samples[i].worldPosition - center, normal));
            residuals.Add(residual);
            if (residual <= inlierLimit)
                inliers++;
        }

        residuals.Sort();
        result = new PlaneFitResult
        {
            valid = true,
            center = center,
            normal = normal,
            axisU = axisU,
            axisV = axisV,
            residualMedian = QuantileSorted(residuals, 0.5f),
            residualP80 = QuantileSorted(residuals, 0.8f),
            residualP95 = QuantileSorted(residuals, 0.95f),
            inlierRatio = inliers / (float)count
        };

        return result.valid;
    }

    private static bool TrySmallestEigenVectorSymmetric(
        float xx, float xy, float xz,
        float yy, float yz, float zz,
        out Vector3 vector)
    {
        float[,] a =
        {
            { xx, xy, xz },
            { xy, yy, yz },
            { xz, yz, zz }
        };
        float[,] v =
        {
            { 1f, 0f, 0f },
            { 0f, 1f, 0f },
            { 0f, 0f, 1f }
        };

        for (int iter = 0; iter < 24; iter++)
        {
            int p = 0;
            int q = 1;
            float max = Mathf.Abs(a[0, 1]);
            float abs02 = Mathf.Abs(a[0, 2]);
            float abs12 = Mathf.Abs(a[1, 2]);
            if (abs02 > max)
            {
                max = abs02;
                p = 0;
                q = 2;
            }
            if (abs12 > max)
            {
                max = abs12;
                p = 1;
                q = 2;
            }
            if (max < 1e-8f)
                break;

            float app = a[p, p];
            float aqq = a[q, q];
            float apq = a[p, q];
            float phi = 0.5f * Mathf.Atan2(2f * apq, aqq - app);
            float c = Mathf.Cos(phi);
            float s = Mathf.Sin(phi);

            for (int k = 0; k < 3; k++)
            {
                float aik = a[k, p];
                float akq = a[k, q];
                a[k, p] = c * aik - s * akq;
                a[k, q] = s * aik + c * akq;
            }

            for (int k = 0; k < 3; k++)
            {
                float aip = a[p, k];
                float aiq = a[q, k];
                a[p, k] = c * aip - s * aiq;
                a[q, k] = s * aip + c * aiq;
            }

            for (int k = 0; k < 3; k++)
            {
                float vip = v[k, p];
                float viq = v[k, q];
                v[k, p] = c * vip - s * viq;
                v[k, q] = s * vip + c * viq;
            }
        }

        int smallest = 0;
        if (a[1, 1] < a[smallest, smallest])
            smallest = 1;
        if (a[2, 2] < a[smallest, smallest])
            smallest = 2;

        vector = new Vector3(v[0, smallest], v[1, smallest], v[2, smallest]);
        if (!IsFinite(vector) || vector.sqrMagnitude < 1e-8f)
            return false;

        vector.Normalize();
        return true;
    }

    private static Vector3 BuildPlaneAxisU(Vector3 normal, Camera cam)
    {
        Vector3 preferred = cam != null ? cam.transform.right : Vector3.right;
        Vector3 axisU = Vector3.ProjectOnPlane(preferred, normal);
        if (axisU.sqrMagnitude < 1e-8f)
            axisU = Vector3.ProjectOnPlane(Vector3.right, normal);
        if (axisU.sqrMagnitude < 1e-8f)
            axisU = Vector3.Cross(normal, Vector3.up);
        if (axisU.sqrMagnitude < 1e-8f)
            axisU = Vector3.Cross(normal, Vector3.forward);
        return axisU.normalized;
    }

    private static float QuantileSorted(List<float> sortedValues, float quantile)
    {
        if (sortedValues == null || sortedValues.Count == 0)
            return 0f;

        int index = Mathf.Clamp(Mathf.RoundToInt((sortedValues.Count - 1) * quantile), 0, sortedValues.Count - 1);
        return sortedValues[index];
    }

    private void AddLatticeEdge(
        int ax, int ay,
        int bx, int by,
        float minU,
        float minV,
        float cellSize,
        PlaneFitResult plane,
        Transform displayTransform,
        HashSet<string> edges)
    {
        string key = EdgeKey(ax, ay, bx, by);
        if (!edges.Add(key))
            return;

        Vector3 a = LatticeVertexWorld(ax, ay, minU, minV, cellSize, plane);
        Vector3 b = LatticeVertexWorld(bx, by, minU, minV, cellSize, plane);
        AddLine(displayTransform.InverseTransformPoint(a), displayTransform.InverseTransformPoint(b), latticeColor);
    }

    private static Vector3 LatticeVertexWorld(int x, int y, float minU, float minV, float cellSize, PlaneFitResult plane)
    {
        return plane.center + plane.axisU * (minU + x * cellSize) + plane.axisV * (minV + y * cellSize);
    }

    private void AddLine(Vector3 a, Vector3 b, Color color)
    {
        int start = _meshVertices.Count;
        _meshVertices.Add(a);
        _meshVertices.Add(b);
        _meshColors.Add(color);
        _meshColors.Add(color);
        _meshIndices.Add(start);
        _meshIndices.Add(start + 1);
    }

    private static string CellKey(int x, int y)
    {
        return x.ToString(CultureInfo.InvariantCulture) + "," + y.ToString(CultureInfo.InvariantCulture);
    }

    private static bool TryParseCellKey(string key, out int x, out int y)
    {
        x = 0;
        y = 0;
        int comma = key.IndexOf(',');
        if (comma <= 0 || comma >= key.Length - 1)
            return false;

        return int.TryParse(key.Substring(0, comma), NumberStyles.Integer, CultureInfo.InvariantCulture, out x) &&
               int.TryParse(key.Substring(comma + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out y);
    }

    private static string EdgeKey(int ax, int ay, int bx, int by)
    {
        if (ax > bx || (ax == bx && ay > by))
        {
            int tx = ax;
            int ty = ay;
            ax = bx;
            ay = by;
            bx = tx;
            by = ty;
        }

        return ax.ToString(CultureInfo.InvariantCulture) + "," +
               ay.ToString(CultureInfo.InvariantCulture) + "-" +
               bx.ToString(CultureInfo.InvariantCulture) + "," +
               by.ToString(CultureInfo.InvariantCulture);
    }

    private void ClearMesh()
    {
        if (_displayMesh != null)
            _displayMesh.Clear();
    }

    private void UpdateSamplingWindowDisplay()
    {
        if (!showSamplingWindow)
        {
            if (_windowMesh != null)
                _windowMesh.Clear();
            return;
        }

        Camera cam = ResolveCenterCamera();
        if (cam == null)
            return;

        EnsureSamplingWindowDisplayObject();
        if (_windowDisplayObject == null || _windowMesh == null)
            return;

        if (useCameraCenterHitPhysicalWindow)
        {
            WorldSamplingWindow window;
            if (_hasActiveWorldWindow)
            {
                window = _activeWorldWindow;
            }
            else if (!TryBuildWorldSamplingWindow(out window))
            {
                _windowMesh.Clear();
                return;
            }

            Transform parent = displayParent != null ? displayParent : transform;
            if (_windowDisplayObject.transform.parent != parent)
                _windowDisplayObject.transform.SetParent(parent, false);

            _windowDisplayObject.transform.localPosition = Vector3.zero;
            _windowDisplayObject.transform.localRotation = Quaternion.identity;
            _windowDisplayObject.transform.localScale = Vector3.one;

            float half = window.sizeMeters * 0.5f;
            Vector3 aWorld = window.center - window.axisU * half - window.axisV * half;
            Vector3 bWorld = window.center - window.axisU * half + window.axisV * half;
            Vector3 cWorld = window.center + window.axisU * half + window.axisV * half;
            Vector3 dWorld = window.center + window.axisU * half - window.axisV * half;

            Transform displayTransform = _windowDisplayObject.transform;
            var windowVertices = new List<Vector3>(4)
            {
                displayTransform.InverseTransformPoint(aWorld),
                displayTransform.InverseTransformPoint(bWorld),
                displayTransform.InverseTransformPoint(cWorld),
                displayTransform.InverseTransformPoint(dWorld)
            };
            var windowColors = new List<Color>(4) { samplingWindowColor, samplingWindowColor, samplingWindowColor, samplingWindowColor };
            var windowIndices = new List<int>(8) { 0, 1, 1, 2, 2, 3, 3, 0 };

            _windowMesh.Clear();
            _windowMesh.SetVertices(windowVertices);
            _windowMesh.SetColors(windowColors);
            _windowMesh.SetIndices(windowIndices, MeshTopology.Lines, 0, true);
            _windowMesh.RecalculateBounds();
            return;
        }

        if (_windowDisplayObject.transform.parent != cam.transform)
            _windowDisplayObject.transform.SetParent(cam.transform, false);

        _windowDisplayObject.transform.localPosition = Vector3.zero;
        _windowDisplayObject.transform.localRotation = Quaternion.identity;
        _windowDisplayObject.transform.localScale = Vector3.one;

        float depthTextureSize = Mathf.Max(1, CustomEnvironmentDepthRaycaster.TextureSize);
        float viewportFraction = Mathf.Clamp01(windowPixels / depthTextureSize) * Mathf.Max(0.1f, samplingWindowScale);
        float distance = Mathf.Max(0.05f, samplingWindowDistanceMeters);
        float halfHeight = Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * distance * viewportFraction * 0.5f;
        float halfWidth = halfHeight * Mathf.Max(0.01f, cam.aspect);

        Vector3 a = new Vector3(-halfWidth, -halfHeight, distance);
        Vector3 b = new Vector3(-halfWidth, halfHeight, distance);
        Vector3 c = new Vector3(halfWidth, halfHeight, distance);
        Vector3 d = new Vector3(halfWidth, -halfHeight, distance);

        var vertices = new List<Vector3>(4) { a, b, c, d };
        var colors = new List<Color>(4) { samplingWindowColor, samplingWindowColor, samplingWindowColor, samplingWindowColor };
        var indices = new List<int>(8) { 0, 1, 1, 2, 2, 3, 3, 0 };

        _windowMesh.Clear();
        _windowMesh.SetVertices(vertices);
        _windowMesh.SetColors(colors);
        _windowMesh.SetIndices(indices, MeshTopology.Lines, 0, true);
        _windowMesh.RecalculateBounds();
    }

    private void SaveCaptureToDesktop()
    {
        try
        {
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            string folder = Path.Combine(desktop, string.IsNullOrWhiteSpace(desktopFolderName) ? "ScanCoverDepthPointBurst" : desktopFolderName);
            Directory.CreateDirectory(folder);

            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            string csvPath = Path.Combine(folder, $"point_burst_{stamp}.csv");
            string summaryPath = Path.Combine(folder, $"point_burst_{stamp}_summary.json");

            File.WriteAllText(csvPath, BuildCsv(), Encoding.UTF8);
            File.WriteAllText(summaryPath, BuildSummaryJson(), Encoding.UTF8);

            if (debugLog)
                Debug.Log($"[ScanCoverDepthPointBurstWindow] Saved {csvPath}");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ScanCoverDepthPointBurstWindow] Save failed: {ex.Message}");
        }
    }

    private string BuildCsv()
    {
        var sb = new StringBuilder(256 + _samples.Count * 96);
        sb.AppendLine("burstId,burstFrame,globalFrame,sampleIndex,textureX,textureY,windowX,windowY,windowLocalU,windowLocalV,linearDepth,worldX,worldY,worldZ,windowCenterX,windowCenterY,windowCenterZ,windowAxisUX,windowAxisUY,windowAxisUZ,windowAxisVX,windowAxisVY,windowAxisVZ,windowSizeMeters,fromPhysicalWindow,valid,status");
        for (int i = 0; i < _samples.Count; i++)
        {
            SampleRecord s = _samples[i];
            Append(sb, s.burstId).Append(',');
            Append(sb, s.burstFrame).Append(',');
            Append(sb, s.globalFrame).Append(',');
            Append(sb, s.sampleIndexInFrame).Append(',');
            Append(sb, s.textureX).Append(',');
            Append(sb, s.textureY).Append(',');
            Append(sb, s.windowX).Append(',');
            Append(sb, s.windowY).Append(',');
            Append(sb, s.windowLocalU).Append(',');
            Append(sb, s.windowLocalV).Append(',');
            Append(sb, s.linearDepth).Append(',');
            Append(sb, s.worldPosition.x).Append(',');
            Append(sb, s.worldPosition.y).Append(',');
            Append(sb, s.worldPosition.z).Append(',');
            Append(sb, s.windowCenter.x).Append(',');
            Append(sb, s.windowCenter.y).Append(',');
            Append(sb, s.windowCenter.z).Append(',');
            Append(sb, s.windowAxisU.x).Append(',');
            Append(sb, s.windowAxisU.y).Append(',');
            Append(sb, s.windowAxisU.z).Append(',');
            Append(sb, s.windowAxisV.x).Append(',');
            Append(sb, s.windowAxisV.y).Append(',');
            Append(sb, s.windowAxisV.z).Append(',');
            Append(sb, s.windowSizeMeters).Append(',');
            sb.Append(s.fromPhysicalWindow ? "1" : "0").Append(',');
            sb.Append(s.valid ? "1" : "0").Append(',');
            sb.AppendLine(s.status.ToString());
        }

        return sb.ToString();
    }

    private string BuildSummaryJson()
    {
        int validCount = 0;
        int invalidCount = 0;
        for (int i = 0; i < _samples.Count; i++)
        {
            if (_samples[i].valid)
                validCount++;
            else
                invalidCount++;
        }

        return "{\n" +
               $"  \"burstId\": {_burstId},\n" +
               $"  \"framesCollected\": {_framesCollectedThisBurst},\n" +
               $"  \"pointCount\": {_samples.Count},\n" +
               $"  \"validPointCount\": {validCount},\n" +
               $"  \"invalidPointCount\": {invalidCount},\n" +
               $"  \"recordInvalidSamples\": {recordInvalidSamples.ToString().ToLowerInvariant()},\n" +
               $"  \"textureSize\": {CustomEnvironmentDepthRaycaster.TextureSize},\n" +
               $"  \"useCameraCenterHitPhysicalWindow\": {useCameraCenterHitPhysicalWindow.ToString().ToLowerInvariant()},\n" +
               $"  \"physicalWindowSizeMeters\": {physicalWindowSizeMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"hasActiveWorldWindow\": {_hasActiveWorldWindow.ToString().ToLowerInvariant()},\n" +
               $"  \"activeWindowCenter\": [{_activeWorldWindow.center.x.ToString(CultureInfo.InvariantCulture)}, {_activeWorldWindow.center.y.ToString(CultureInfo.InvariantCulture)}, {_activeWorldWindow.center.z.ToString(CultureInfo.InvariantCulture)}],\n" +
               $"  \"activeWindowAxisU\": [{_activeWorldWindow.axisU.x.ToString(CultureInfo.InvariantCulture)}, {_activeWorldWindow.axisU.y.ToString(CultureInfo.InvariantCulture)}, {_activeWorldWindow.axisU.z.ToString(CultureInfo.InvariantCulture)}],\n" +
               $"  \"activeWindowAxisV\": [{_activeWorldWindow.axisV.x.ToString(CultureInfo.InvariantCulture)}, {_activeWorldWindow.axisV.y.ToString(CultureInfo.InvariantCulture)}, {_activeWorldWindow.axisV.z.ToString(CultureInfo.InvariantCulture)}],\n" +
               $"  \"activeWindowSizeMeters\": {_activeWorldWindow.sizeMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"activeWindowCenterLinearDepth\": {_activeWorldWindow.centerLinearDepth.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"windowPixels\": {windowPixels},\n" +
               $"  \"showSamplingWindow\": {showSamplingWindow.ToString().ToLowerInvariant()},\n" +
               $"  \"samplingWindowDistanceMeters\": {samplingWindowDistanceMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"samplingWindowScale\": {samplingWindowScale.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"samplesPerFrame\": {samplesPerFrame},\n" +
               $"  \"stratifiedJitter\": {stratifiedJitter.ToString().ToLowerInvariant()},\n" +
               $"  \"randomizeBurstSeed\": {randomizeBurstSeed.ToString().ToLowerInvariant()},\n" +
               $"  \"eye\": \"{eye}\",\n" +
               $"  \"displayMode\": \"{displayMode}\",\n" +
               $"  \"showTrustedLattice\": {showTrustedLattice.ToString().ToLowerInvariant()},\n" +
               $"  \"minLatticeSamples\": {minLatticeSamples},\n" +
               $"  \"minTrustedInlierRatio\": {minTrustedInlierRatio.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"maxTrustedResidualP95Meters\": {maxTrustedResidualP95Meters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"latticeInlierResidualMeters\": {latticeInlierResidualMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"latticeCellSizeMeters\": {latticeCellSizeMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"minLinearDepthMeters\": {minLinearDepthMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"maxLinearDepthMeters\": {maxLinearDepthMeters.ToString(CultureInfo.InvariantCulture)}\n" +
               "}\n";
    }

    private static StringBuilder Append(StringBuilder sb, int value)
    {
        return sb.Append(value.ToString(CultureInfo.InvariantCulture));
    }

    private static StringBuilder Append(StringBuilder sb, float value)
    {
        return sb.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private void EnsureDisplayObject()
    {
        if (_displayObject == null)
        {
            Transform parent = displayParent != null ? displayParent : transform;
            _displayObject = new GameObject("Depth Point Burst Window Samples");
            _displayObject.transform.SetParent(parent, false);
            _displayObject.transform.localPosition = Vector3.zero;
            _displayObject.transform.localRotation = Quaternion.identity;
            _displayObject.transform.localScale = Vector3.one;
            _displayObject.AddComponent<MeshFilter>();
            _displayObject.AddComponent<MeshRenderer>();
        }

        MeshFilter filter = _displayObject.GetComponent<MeshFilter>();
        MeshRenderer renderer = _displayObject.GetComponent<MeshRenderer>();

        if (_displayMesh == null)
        {
            _displayMesh = new Mesh { name = "Depth Point Burst Window Samples" };
            _displayMesh.MarkDynamic();
        }

        if (filter.sharedMesh != _displayMesh)
            filter.sharedMesh = _displayMesh;

        if (_displayMaterial == null)
        {
            Shader shader = Shader.Find("Sprites/Default");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            _displayMaterial = new Material(shader) { name = "Depth Point Burst Window Material" };
            _displayMaterial.color = pointColor;
        }

        if (renderer.sharedMaterial != _displayMaterial)
            renderer.sharedMaterial = _displayMaterial;
    }

    private void EnsureSamplingWindowDisplayObject()
    {
        Camera cam = ResolveCenterCamera();
        Transform parent = useCameraCenterHitPhysicalWindow
            ? (displayParent != null ? displayParent : transform)
            : (cam != null ? cam.transform : transform);

        if (_windowDisplayObject == null)
        {
            _windowDisplayObject = new GameObject("Depth Point Burst Sampling Window");
            _windowDisplayObject.transform.SetParent(parent, false);
            _windowDisplayObject.transform.localPosition = Vector3.zero;
            _windowDisplayObject.transform.localRotation = Quaternion.identity;
            _windowDisplayObject.transform.localScale = Vector3.one;
            _windowDisplayObject.AddComponent<MeshFilter>();
            _windowDisplayObject.AddComponent<MeshRenderer>();
        }

        MeshFilter filter = _windowDisplayObject.GetComponent<MeshFilter>();
        MeshRenderer renderer = _windowDisplayObject.GetComponent<MeshRenderer>();

        if (_windowMesh == null)
        {
            _windowMesh = new Mesh { name = "Depth Point Burst Sampling Window" };
            _windowMesh.MarkDynamic();
        }

        if (filter.sharedMesh != _windowMesh)
            filter.sharedMesh = _windowMesh;

        if (_windowMaterial == null)
        {
            Shader shader = Shader.Find("Sprites/Default");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            _windowMaterial = new Material(shader) { name = "Depth Point Burst Sampling Window Material" };
            _windowMaterial.color = samplingWindowColor;
        }

        if (renderer.sharedMaterial != _windowMaterial)
            renderer.sharedMaterial = _windowMaterial;
    }

    private void ResolveRefs()
    {
        if (depthRaycaster == null)
            depthRaycaster = GetComponentInChildren<CustomEnvironmentDepthRaycaster>(true);
        if (depthRaycaster == null)
            depthRaycaster = FindAnyObjectByType<CustomEnvironmentDepthRaycaster>(FindObjectsInactive.Include);

        if (environmentDepthManager == null && depthRaycaster != null)
            environmentDepthManager = depthRaycaster.depthManager;
        if (environmentDepthManager == null)
            environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);

        if (centerCamera == null)
            centerCamera = Camera.main;
        if (displayParent == null)
            displayParent = transform;
    }

    private Camera ResolveCenterCamera()
    {
        if (centerCamera == null)
            centerCamera = Camera.main;
        return centerCamera;
    }

    private static bool IsFinite(Vector3 value)
    {
        return float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);
    }
}

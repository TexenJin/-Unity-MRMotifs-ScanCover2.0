using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR.EnvironmentDepth;
using MyProject.XR;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

[DisallowMultipleComponent]
public sealed class ScanCoverDepthPointBurstWindow : MonoBehaviour
{
    private enum PointBurstDisplayMode
    {
        Points = 0,
        TrustedLattice = 1,
        None = 2,
        TrustedThinBox = 3
    }

    private enum SampleStatus
    {
        Valid,
        TrustedPlane,
        CandidatePlane,
        SecondaryTrustedPlane,
        SecondaryCandidatePlane,
        DepthEdge,
        EdgeEvidence,
        InvalidDepth
    }

    private enum TerrainCellClassification
    {
        StableSingleLayer,
        BoundaryDepthJump,
        MultiLayer,
        Unstable
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

    [Header("Point Cloud Terrain Diagnostics")]
    [SerializeField] private bool pointCloudTerrainDiagnosticsOnly = true;
    [SerializeField, Min(0.03f)] private float terrainDiagnosticsWindowSizeMeters = 0.35f;
    [SerializeField, Min(1)] private int terrainDiagnosticsSamplesPerFrame = 512;
    [SerializeField, Min(1)] private int terrainDiagnosticsFramesPerBurst = 45;
    [SerializeField, Min(1000)] private int terrainDiagnosticsMaxAccumulatedPoints = 30000;
    [SerializeField] private bool terrainDiagnosticsUsePureRandomSampling = false;
    [SerializeField] private bool terrainDiagnosticsKeepPointClassification = true;
    [SerializeField] private bool terrainDiagnosticsShowSamplingWindow = true;

    [Header("Terrain Cell Classification")]
    [SerializeField] private bool terrainDiagnosticsDisplayAggregatedCells = true;
    [SerializeField] private bool terrainDiagnosticsShowRawOverlay = false;
    [SerializeField, Min(0.005f)] private float terrainCellSizeMeters = 0.02f;
    [SerializeField, Min(1)] private int terrainStableMinSamplesPerCell = 4;
    [SerializeField, Min(1)] private int terrainStableMinSupportFrames = 3;
    [SerializeField, Min(0.001f)] private float terrainStableMaxThicknessMeters = 0.025f;
    [SerializeField, Min(0.001f)] private float terrainMultiLayerMinDepthGapMeters = 0.045f;
    [SerializeField, Min(0.001f)] private float terrainNeighborDepthJumpMeters = 0.055f;
    [SerializeField, Range(0.05f, 0.45f)] private float terrainMultiLayerMinFraction = 0.2f;
    [SerializeField, Range(0f, 1f)] private float terrainBoundaryMinDepthEdgeFraction = 0.25f;
    [SerializeField, Min(0.001f)] private float terrainRepresentativePointSizeMeters = 0.012f;
    [SerializeField] private Color terrainStableCellColor = new Color(0.1f, 1f, 0.45f, 0.95f);
    [SerializeField] private Color terrainBoundaryCellColor = new Color(1f, 0.48f, 0.05f, 0.95f);
    [SerializeField] private Color terrainMultiLayerNearColor = new Color(0.65f, 0.2f, 1f, 0.95f);
    [SerializeField] private Color terrainMultiLayerFarColor = new Color(1f, 0.78f, 0.05f, 0.95f);
    [SerializeField] private Color terrainUnstableCellColor = new Color(0.45f, 0.45f, 0.45f, 0.55f);

    [Header("Planarized Terrain Preview")]
    [SerializeField] private bool terrainDiagnosticsShowPlanarizedPreview = true;
    [SerializeField] private bool terrainDiagnosticsShowRejectedPlaneIslands = true;
    [SerializeField, Min(3)] private int terrainPlaneMinCells = 12;
    [SerializeField, Range(0.5f, 1f)] private float terrainPlaneTrimKeepRatio = 0.82f;
    [SerializeField, Min(0)] private int terrainPlaneRefineIterations = 2;
    [SerializeField, Min(0.001f)] private float terrainPlaneMaxResidualP95Meters = 0.035f;
    [SerializeField, Range(0.1f, 1f)] private float terrainPlaneMinInlierRatio = 0.75f;
    [SerializeField, Min(0.001f)] private float terrainPlaneInlierResidualMeters = 0.025f;
    [SerializeField, Min(0.001f)] private float terrainPlaneMaxProjectionOffsetMeters = 0.045f;
    [SerializeField] private bool terrainDiagnosticsShowPlanarizedMeshPreview = true;
    [SerializeField] private bool terrainDiagnosticsUsePlanarizedCellTiles = true;
    [SerializeField] private bool terrainDiagnosticsShowPlanarizedPointFallback = true;
    [SerializeField, Min(0.001f)] private float terrainPlaneMeshMaxEdgeMeters = 0.08f;
    [SerializeField, Range(0.5f, 1f)] private float terrainPlanarizedTileFillRatio = 1f;
    [SerializeField] private Color terrainPlanarizedAcceptedColor = new Color(0.15f, 0.75f, 1f, 0.98f);
    [SerializeField] private Color terrainPlanarizedMeshColor = new Color(0.05f, 0.55f, 1f, 0.38f);
    [SerializeField] private Color terrainPlanarizedRejectedColor = new Color(1f, 0.12f, 0.12f, 0.85f);

    [Header("Point Display")]
    [SerializeField] private PointBurstDisplayMode displayMode = PointBurstDisplayMode.TrustedThinBox;
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
    [SerializeField] private Color secondaryTrustedPointColor = new Color(0.2f, 0.65f, 1f, 0.95f);
    [SerializeField] private Color secondaryCandidatePointColor = new Color(0.55f, 0.25f, 1f, 0.95f);
    [SerializeField] private Color depthEdgePointColor = new Color(1f, 0.05f, 0.05f, 0.95f);
    [SerializeField] private Color edgeEvidenceColor = new Color(1f, 0.45f, 0f, 0.95f);
    [SerializeField] private Color rejectedPointColor = new Color(1f, 0.1f, 0.1f, 0.9f);
    [SerializeField] private Color unknownPointColor = new Color(0.8f, 0f, 1f, 0.9f);
    [SerializeField, Min(0.001f)] private float trustedResidualBaseMeters = 0.012f;
    [SerializeField, Min(0.001f)] private float candidateResidualBaseMeters = 0.03f;

    [Header("Secondary Plane Candidate")]
    [SerializeField] private bool enableSecondaryPlaneCandidate = true;
    [SerializeField, Min(8)] private int minSecondaryPlaneSamples = 32;
    [SerializeField, Min(0.001f)] private float secondaryPlaneTrustedResidualMeters = 0.025f;
    [SerializeField, Min(0.001f)] private float secondaryPlaneCandidateResidualMeters = 0.05f;
    [SerializeField, Range(5f, 90f)] private float minSecondaryPlaneNormalAngleDegrees = 20f;
    [SerializeField, Min(0.001f)] private float minSecondaryPlaneDistanceFromPrimaryMeters = 0.025f;
    [SerializeField, Min(1)] private int secondaryPlaneMaxThinBoxGroups = 4;
    [SerializeField] private Color secondaryThinBoxColor = new Color(0.15f, 0.55f, 1f, 0.55f);

    [Header("Depth Edge Detection")]
    [SerializeField] private bool enableDepthEdgeDetection = true;
    [SerializeField] private bool excludeDepthEdgeSamplesFromPlaneFit = true;
    [SerializeField, Min(1)] private int depthEdgeProbeRadiusPixels = 1;
    [SerializeField, Min(0.001f)] private float minDepthEdgeJumpMeters = 0.035f;
    [SerializeField, Min(0f)] private float depthEdgeJumpDistanceScale = 0.018f;
    [SerializeField] private bool depthEdgeUseLocalPlaneReject = true;
    [SerializeField, Min(0.001f)] private float localPlaneResidualBaseMeters = 0.012f;
    [SerializeField, Min(0f)] private float localPlaneResidualDistanceScale = 0.006f;
    [SerializeField, Range(0.1f, 1f)] private float localPlaneMinInlierRatio = 0.65f;
    [SerializeField, Min(3)] private int localPlaneMinNeighborPoints = 5;
    [SerializeField, Range(0.1f, 0.9f)] private float depthEdgeTwoClusterMinFraction = 0.25f;

    [Header("Trusted Lattice")]
    [SerializeField] private bool showTrustedLattice = false;
    [SerializeField, Min(16)] private int minLatticeSamples = 256;
    [SerializeField, Range(0.1f, 1f)] private float minTrustedInlierRatio = 0.8f;
    [SerializeField, Min(0.001f)] private float maxTrustedResidualP95Meters = 0.03f;
    [SerializeField, Min(0.001f)] private float latticeInlierResidualMeters = 0.02f;
    [SerializeField, Min(0.005f)] private float latticeCellSizeMeters = 0.03f;
    [SerializeField, Min(1)] private int minInliersPerLatticeCell = 2;
    [SerializeField] private Color latticeColor = new Color(1f, 1f, 1f, 0.92f);

    [Header("Trusted Thin Box")]
    [FormerlySerializedAs("showTrustedSurfaceMesh")]
    [SerializeField] private bool showTrustedThinBox = true;
    [FormerlySerializedAs("trustedSurfaceCellSizeMeters")]
    [SerializeField, Min(0.005f)] private float trustedThinBoxMinSpanMeters = 0.03f;
    [SerializeField, Min(0.005f)] private float trustedThinBoxGroupCellSizeMeters = 0.015f;
    [SerializeField, Min(1)] private int trustedThinBoxMinSamplesPerGroupCell = 2;
    [SerializeField, Min(0.005f)] private float trustedThinBoxMaxNeighborDepthJumpMeters = 0.04f;
    [SerializeField, Min(0.005f)] private float trustedThinBoxMaxNeighborLinearDepthJumpMeters = 0.05f;
    [SerializeField, Min(0.01f)] private float trustedThinBoxMaxNeighborWorldGapMeters = 0.08f;
    [SerializeField, Min(3)] private int trustedThinBoxMinGroupSamples = 24;
    [SerializeField, Min(1)] private int trustedThinBoxMaxGroups = 8;
    [SerializeField, Min(0.01f)] private float trustedThinBoxMaxGroupResidualP95Meters = 0.035f;
    [SerializeField, Min(1f)] private float trustedThinBoxMaxAspectRatio = 5f;
    [SerializeField, Min(0.05f)] private float trustedThinBoxMaxSpanMeters = 0.35f;
    [SerializeField] private bool trustedThinBoxUseTrimmedPlaneFit = true;
    [SerializeField, Range(0.5f, 0.98f)] private float trustedThinBoxPlaneTrimKeepRatio = 0.8f;
    [SerializeField, Min(1)] private int trustedThinBoxPlaneRefineIterations = 2;
    [SerializeField, Min(0.001f)] private float trustedThinBoxExtentInlierResidualMultiplier = 1.25f;
    [SerializeField, Min(0.001f)] private float trustedThinBoxLineWidthMeters = 0.006f;
    [SerializeField, Min(0f)] private float trustedThinBoxPaddingMeters = 0.01f;
    [SerializeField, Min(0.001f)] private float trustedThinBoxMinThicknessMeters = 0.012f;
    [FormerlySerializedAs("trustedSurfaceColor")]
    [SerializeField] private Color trustedThinBoxColor = new Color(0.1f, 1f, 0.65f, 0.55f);
    [HideInInspector, SerializeField, Min(1)] private int minTrustedSamplesPerSurfaceCell = 2;
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

    private struct LocalDepthProbe
    {
        public Vector2Int texCoord;
        public Vector3 worldPosition;
        public float linearDepth;
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

    private struct IndexResidual
    {
        public int index;
        public float residual;
    }

    private sealed class TrustedThinBoxCell
    {
        public readonly List<int> indices = new List<int>();
        public Vector3 worldSum;
        public float windowNormalOffsetSum;
        public float linearDepthSum;

        public Vector3 MeanWorld => indices.Count > 0 ? worldSum / indices.Count : Vector3.zero;
        public float MeanWindowNormalOffset => indices.Count > 0 ? windowNormalOffsetSum / indices.Count : 0f;
        public float MeanLinearDepth => indices.Count > 0 ? linearDepthSum / indices.Count : 0f;
    }

    private sealed class TerrainCell
    {
        public readonly List<int> indices = new List<int>();
        public readonly List<float> depths = new List<float>();
        public readonly HashSet<int> supportFrames = new HashSet<int>();
        public Vector3 worldSum;
        public int depthEdgeSamples;
        public float depthP10;
        public float depthMedian;
        public float depthP90;
        public float thickness;
        public float largestDepthGap;
        public int splitIndex;
        public TerrainCellClassification classification;

        public Vector3 MeanWorld => indices.Count > 0 ? worldSum / indices.Count : Vector3.zero;
    }

    private struct TerrainCellSummary
    {
        public int cellCount;
        public int stableSingleLayerCount;
        public int boundaryDepthJumpCount;
        public int multiLayerCount;
        public int unstableCount;
    }

    private sealed class TerrainPlaneIsland
    {
        public readonly List<Vector2Int> cellKeys = new List<Vector2Int>();
        public readonly List<int> representativeIndices = new List<int>();
        public readonly Dictionary<Vector2Int, int> representativeIndexByCell = new Dictionary<Vector2Int, int>();
        public PlaneFitResult plane;
        public bool accepted;
        public float maxProjectionOffset;
    }

    private struct TerrainPlaneSummary
    {
        public int islandCount;
        public int acceptedIslandCount;
        public int rejectedIslandCount;
        public int acceptedCellCount;
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
            if (_framesCollectedThisBurst >= Mathf.Max(1, EffectiveFramesPerBurst))
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
            Debug.Log($"[ScanCoverDepthPointBurstWindow] Begin burst {_burstId}, physicalWindow={useCameraCenterHitPhysicalWindow}, window={windowPixels}px/{EffectivePhysicalWindowSizeMeters:F3}m samplesPerFrame={EffectiveSamplesPerFrame} stratifiedJitter={EffectiveStratifiedJitter}");
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
    private float EffectivePhysicalWindowSizeMeters => pointCloudTerrainDiagnosticsOnly ? terrainDiagnosticsWindowSizeMeters : physicalWindowSizeMeters;
    private int EffectiveSamplesPerFrame => pointCloudTerrainDiagnosticsOnly ? terrainDiagnosticsSamplesPerFrame : samplesPerFrame;
    private int EffectiveFramesPerBurst => pointCloudTerrainDiagnosticsOnly ? terrainDiagnosticsFramesPerBurst : framesPerBurst;
    private int EffectiveMaxAccumulatedPoints => pointCloudTerrainDiagnosticsOnly ? terrainDiagnosticsMaxAccumulatedPoints : maxAccumulatedPoints;
    private bool EffectiveStratifiedJitter => pointCloudTerrainDiagnosticsOnly ? !terrainDiagnosticsUsePureRandomSampling : stratifiedJitter;
    private bool EffectiveShowSamplingWindow => pointCloudTerrainDiagnosticsOnly ? terrainDiagnosticsShowSamplingWindow : showSamplingWindow;
    private bool EffectivePlanePointClassification => !pointCloudTerrainDiagnosticsOnly || terrainDiagnosticsKeepPointClassification;

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
            int physicalTargetSamples = Mathf.Max(1, EffectiveSamplesPerFrame);
            if (EffectiveStratifiedJitter)
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
        int targetSamples = Mathf.Max(1, EffectiveSamplesPerFrame);

        if (EffectiveStratifiedJitter)
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
            sizeMeters = Mathf.Max(0.005f, EffectivePhysicalWindowSizeMeters * Mathf.Max(0.1f, samplingWindowScale)),
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
        if (_samples.Count >= Mathf.Max(1, EffectiveMaxAccumulatedPoints))
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
        if (!recordInvalidSamples || _samples.Count >= Mathf.Max(1, EffectiveMaxAccumulatedPoints))
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
        if (!colorizeValidPointsByPlaneResidual || !EffectivePlanePointClassification)
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

        ClassifySecondaryPlaneCandidate(plane, validIndices);
    }

    private void ClassifySecondaryPlaneCandidate(PlaneFitResult primaryPlane, List<int> validIndices)
    {
        if (!enableSecondaryPlaneCandidate || validIndices == null)
            return;

        float minPrimaryDistance = Mathf.Max(0.001f, minSecondaryPlaneDistanceFromPrimaryMeters);
        var candidateIndices = new List<int>(validIndices.Count);
        for (int i = 0; i < validIndices.Count; i++)
        {
            int sampleIndex = validIndices[i];
            SampleRecord sample = _samples[sampleIndex];
            if (!sample.valid ||
                sample.status == SampleStatus.DepthEdge ||
                sample.status == SampleStatus.TrustedPlane ||
                !IsFinite(sample.worldPosition))
            {
                continue;
            }

            float primaryResidual = Mathf.Abs(Vector3.Dot(sample.worldPosition - primaryPlane.center, primaryPlane.normal));
            if (primaryResidual < minPrimaryDistance)
                continue;

            candidateIndices.Add(sampleIndex);
        }

        if (candidateIndices.Count < Mathf.Max(8, minSecondaryPlaneSamples))
            return;

        Camera cam = centerCamera != null ? centerCamera : Camera.main;
        if (!TryFitTrimmedSamplePlaneFromIndices(
                candidateIndices,
                cam,
                trustedThinBoxPlaneTrimKeepRatio,
                trustedThinBoxPlaneRefineIterations,
                out PlaneFitResult secondaryPlane,
                out List<int> secondaryInliers))
        {
            return;
        }

        if (secondaryInliers == null || secondaryInliers.Count < Mathf.Max(8, minSecondaryPlaneSamples))
            return;

        float residualLimit = Mathf.Max(secondaryPlaneCandidateResidualMeters, trustedThinBoxMaxGroupResidualP95Meters);
        if (secondaryPlane.residualP95 > residualLimit)
            return;

        float normalAngle = Vector3.Angle(primaryPlane.normal, secondaryPlane.normal);
        normalAngle = Mathf.Min(normalAngle, 180f - normalAngle);
        float planeOffset = Mathf.Abs(Vector3.Dot(secondaryPlane.center - primaryPlane.center, primaryPlane.normal));
        bool sufficientlyDifferent =
            normalAngle >= minSecondaryPlaneNormalAngleDegrees ||
            planeOffset >= minPrimaryDistance * 2f;
        if (!sufficientlyDifferent)
            return;

        float trustedLimit = Mathf.Max(0.001f, secondaryPlaneTrustedResidualMeters);
        float candidateLimit = Mathf.Max(trustedLimit, secondaryPlaneCandidateResidualMeters);
        for (int i = 0; i < candidateIndices.Count; i++)
        {
            int sampleIndex = candidateIndices[i];
            SampleRecord sample = _samples[sampleIndex];
            if (!sample.valid ||
                sample.status == SampleStatus.DepthEdge ||
                sample.status == SampleStatus.TrustedPlane)
            {
                continue;
            }

            float residual = Mathf.Abs(Vector3.Dot(sample.worldPosition - secondaryPlane.center, secondaryPlane.normal));
            if (residual <= trustedLimit)
                sample.status = SampleStatus.SecondaryTrustedPlane;
            else if (residual <= candidateLimit)
                sample.status = SampleStatus.SecondaryCandidatePlane;

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
        if (_samples.Count >= Mathf.Max(1, EffectiveMaxAccumulatedPoints))
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
        var probes = new List<LocalDepthProbe>((radius * 2 + 1) * (radius * 2 + 1));

        if (TryReadWorldPointAtDepthTexCoord(centerTexCoord, out Vector3 centerWorld, out float readCenterDepth))
        {
            centerLinearDepth = readCenterDepth;
            minDepth = centerLinearDepth;
            maxDepth = centerLinearDepth;
            probes.Add(new LocalDepthProbe
            {
                texCoord = centerTexCoord,
                worldPosition = centerWorld,
                linearDepth = centerLinearDepth
            });
        }

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

                Vector2Int texCoord = new Vector2Int(x, y);
                if (!TryReadWorldPointAtDepthTexCoord(texCoord, out Vector3 neighborWorld, out float neighborDepth))
                    continue;

                probes.Add(new LocalDepthProbe
                {
                    texCoord = texCoord,
                    worldPosition = neighborWorld,
                    linearDepth = neighborDepth
                });

                if (neighborDepth < minDepth)
                    minDepth = neighborDepth;
                if (neighborDepth > maxDepth)
                    maxDepth = neighborDepth;
            }
        }

        if (probes.Count < 3)
            return false;

        float jumpLimit = Mathf.Max(minDepthEdgeJumpMeters, centerLinearDepth * depthEdgeJumpDistanceScale);
        if (maxDepth - minDepth <= jumpLimit)
            return false;

        if (depthEdgeUseLocalPlaneReject && probes.Count >= Mathf.Max(3, localPlaneMinNeighborPoints))
        {
            Camera cam = centerCamera != null ? centerCamera : Camera.main;
            float residualLimit = Mathf.Max(localPlaneResidualBaseMeters, centerLinearDepth * localPlaneResidualDistanceScale);
            if (TryFitLocalProbePlane(probes, cam, residualLimit, out PlaneFitResult localPlane) &&
                localPlane.inlierRatio >= localPlaneMinInlierRatio &&
                localPlane.residualP95 <= residualLimit)
            {
                return false;
            }
        }

        if (!depthEdgeUseLocalPlaneReject)
            return true;

        return HasTwoClusterDepthJump(probes, jumpLimit);
    }

    private bool TryFitLocalProbePlane(List<LocalDepthProbe> probes, Camera cam, float inlierLimit, out PlaneFitResult result)
    {
        result = default;
        int count = probes != null ? probes.Count : 0;
        if (count < 3)
            return false;

        Vector3 center = Vector3.zero;
        for (int i = 0; i < count; i++)
            center += probes[i].worldPosition;
        center /= count;

        float xx = 0f;
        float xy = 0f;
        float xz = 0f;
        float yy = 0f;
        float yz = 0f;
        float zz = 0f;

        for (int i = 0; i < count; i++)
        {
            Vector3 d = probes[i].worldPosition - center;
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
        float limit = Mathf.Max(0.001f, inlierLimit);
        for (int i = 0; i < count; i++)
        {
            float residual = Mathf.Abs(Vector3.Dot(probes[i].worldPosition - center, normal));
            residuals.Add(residual);
            if (residual <= limit)
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

        return true;
    }

    private bool HasTwoClusterDepthJump(List<LocalDepthProbe> probes, float jumpLimit)
    {
        int count = probes != null ? probes.Count : 0;
        if (count < 4)
            return false;

        var depths = new List<float>(count);
        for (int i = 0; i < count; i++)
            depths.Add(probes[i].linearDepth);
        depths.Sort();

        float largestGap = 0f;
        int splitIndex = -1;
        for (int i = 0; i < depths.Count - 1; i++)
        {
            float gap = depths[i + 1] - depths[i];
            if (gap > largestGap)
            {
                largestGap = gap;
                splitIndex = i + 1;
            }
        }

        if (splitIndex <= 0 || splitIndex >= count)
            return false;

        float minFraction = Mathf.Clamp01(depthEdgeTwoClusterMinFraction);
        float nearFraction = splitIndex / (float)count;
        float farFraction = (count - splitIndex) / (float)count;
        return largestGap > jumpLimit && nearFraction >= minFraction && farFraction >= minFraction;
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

        bool drawTerrainCells = pointCloudTerrainDiagnosticsOnly && terrainDiagnosticsDisplayAggregatedCells;
        bool drawPoints = showPoints &&
                          (displayMode == PointBurstDisplayMode.Points ||
                           displayMode == PointBurstDisplayMode.TrustedThinBox);
        if (drawTerrainCells)
            drawPoints = terrainDiagnosticsShowRawOverlay && showPoints;

        bool drawLattice = !drawTerrainCells && displayMode == PointBurstDisplayMode.TrustedLattice && showTrustedLattice;
        bool drawTrustedThinBox = showTrustedThinBox &&
                                  !drawTerrainCells &&
                                  (displayMode == PointBurstDisplayMode.TrustedThinBox ||
                                   displayMode == PointBurstDisplayMode.Points);
        if (!drawPoints && !drawLattice && !drawTrustedThinBox && !drawTerrainCells)
        {
            ClearMesh();
            return;
        }

        _meshVertices.Clear();
        _meshColors.Clear();
        _meshIndices.Clear();

        Camera cam = ResolveCenterCamera();
        Transform displayTransform = _displayObject != null ? _displayObject.transform : transform;

        MeshTopology topology = MeshTopology.Triangles;
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
            if (drawTerrainCells)
                TryAddTerrainCellDiagnostics(displayTransform, cam);

            bool trustedThinBoxAdded = false;
            if (drawTrustedThinBox)
            {
                trustedThinBoxAdded = TryAddTrustedThinBox(displayTransform, cam);
                if (!trustedThinBoxAdded && !drawPoints)
                {
                    ClearMesh();
                    return;
                }
            }

            if (drawPoints)
            {
                Vector3 camRight = cam != null ? cam.transform.right : Vector3.right;
                Vector3 camUp = cam != null ? cam.transform.up : Vector3.up;
                Vector3 camPos = cam != null ? cam.transform.position : transform.position;
                Vector3 localRight = displayTransform.InverseTransformDirection(camRight).normalized;
                Vector3 localUp = displayTransform.InverseTransformDirection(camUp).normalized;
                float halfSize = Mathf.Max(0.001f, pointVisualSizeMeters) * 0.5f;
                bool drawPointQuads = drawTrustedThinBox || renderAsBillboardQuads;

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
                    if (drawPointQuads)
                        AddBillboardQuad(p, localRight, localUp, halfSize, sampleColor);
                    else
                        AddMeshPoint(p, sampleColor);
                }

                topology = drawPointQuads ? MeshTopology.Triangles : MeshTopology.Points;
            }
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
            case SampleStatus.SecondaryTrustedPlane:
                return secondaryTrustedPointColor;
            case SampleStatus.SecondaryCandidatePlane:
                return secondaryCandidatePointColor;
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

    private bool TryAddTerrainCellDiagnostics(Transform displayTransform, Camera cam)
    {
        if (!TryBuildTerrainCells(out Dictionary<Vector2Int, TerrainCell> cells, out _))
            return false;

        Vector3 camRight = cam != null ? cam.transform.right : Vector3.right;
        Vector3 camUp = cam != null ? cam.transform.up : Vector3.up;
        Vector3 localRight = displayTransform.InverseTransformDirection(camRight).normalized;
        Vector3 localUp = displayTransform.InverseTransformDirection(camUp).normalized;
        float halfSize = Mathf.Max(0.001f, terrainRepresentativePointSizeMeters) * 0.5f;

        int drawn = 0;
        var planarizedCells = new HashSet<Vector2Int>();
        if (terrainDiagnosticsShowPlanarizedPreview)
        {
            drawn += AddPlanarizedTerrainPreview(cells, planarizedCells, displayTransform, cam, localRight, localUp, halfSize);
        }

        foreach (KeyValuePair<Vector2Int, TerrainCell> pair in cells)
        {
            if (planarizedCells.Contains(pair.Key))
                continue;

            TerrainCell cell = pair.Value;
            if (cell.indices.Count == 0)
                continue;

            if (cell.classification == TerrainCellClassification.MultiLayer &&
                TryGetTerrainLayerRepresentatives(cell, out int nearIndex, out int farIndex))
            {
                AddTerrainRepresentative(_samples[nearIndex], terrainMultiLayerNearColor, displayTransform, localRight, localUp, halfSize);
                AddTerrainRepresentative(_samples[farIndex], terrainMultiLayerFarColor, displayTransform, localRight, localUp, halfSize);
                drawn += 2;
                continue;
            }

            int representativeIndex = FindSampleClosestToDepth(cell, cell.depthMedian, float.NegativeInfinity, float.PositiveInfinity);
            if (representativeIndex < 0)
                continue;

            AddTerrainRepresentative(
                _samples[representativeIndex],
                ResolveTerrainCellColor(cell.classification),
                displayTransform,
                localRight,
                localUp,
                halfSize);
            drawn++;
        }

        return drawn > 0;
    }

    private int AddPlanarizedTerrainPreview(
        Dictionary<Vector2Int, TerrainCell> cells,
        HashSet<Vector2Int> planarizedCells,
        Transform displayTransform,
        Camera cam,
        Vector3 localRight,
        Vector3 localUp,
        float halfSize)
    {
        if (!TryBuildTerrainPlaneIslands(cells, cam, out List<TerrainPlaneIsland> islands, out _))
            return 0;

        int drawn = 0;
        for (int i = 0; i < islands.Count; i++)
        {
            TerrainPlaneIsland island = islands[i];
            if (island.accepted)
            {
                int meshTriangles = terrainDiagnosticsShowPlanarizedMeshPreview
                    ? AddPlanarizedTerrainIslandMesh(island, displayTransform)
                    : 0;
                drawn += meshTriangles;

                bool drawPointFallback = !terrainDiagnosticsShowPlanarizedMeshPreview ||
                                         (meshTriangles == 0 && terrainDiagnosticsShowPlanarizedPointFallback);
                if (drawPointFallback)
                {
                    for (int j = 0; j < island.representativeIndices.Count; j++)
                    {
                        int sampleIndex = island.representativeIndices[j];
                        if (sampleIndex < 0 || sampleIndex >= _samples.Count)
                            continue;

                        SampleRecord sample = _samples[sampleIndex];
                        Vector3 projected = ProjectPointToPlane(sample.worldPosition, island.plane);
                        AddTerrainRepresentative(projected, terrainPlanarizedAcceptedColor, displayTransform, localRight, localUp, halfSize);
                        drawn++;
                    }
                }

                for (int j = 0; j < island.cellKeys.Count; j++)
                    planarizedCells.Add(island.cellKeys[j]);

                continue;
            }

            if (!terrainDiagnosticsShowRejectedPlaneIslands)
                continue;

            for (int j = 0; j < island.representativeIndices.Count; j++)
            {
                int sampleIndex = island.representativeIndices[j];
                if (sampleIndex < 0 || sampleIndex >= _samples.Count)
                    continue;

                AddTerrainRepresentative(_samples[sampleIndex], terrainPlanarizedRejectedColor, displayTransform, localRight, localUp, halfSize);
                drawn++;
            }
        }

        return drawn;
    }

    private int AddPlanarizedTerrainIslandMesh(TerrainPlaneIsland island, Transform displayTransform)
    {
        if (island == null || !island.accepted || island.cellKeys.Count < 4)
            return 0;

        if (terrainDiagnosticsUsePlanarizedCellTiles)
            return AddPlanarizedTerrainIslandTileMesh(island, displayTransform);

        var vertexByCell = new Dictionary<Vector2Int, int>(island.cellKeys.Count);
        float maxProjectionOffset = Mathf.Max(0.001f, terrainPlaneMaxProjectionOffsetMeters);
        for (int i = 0; i < island.cellKeys.Count; i++)
        {
            Vector2Int cellKey = island.cellKeys[i];
            if (!island.representativeIndexByCell.TryGetValue(cellKey, out int sampleIndex))
                continue;

            if (sampleIndex < 0 || sampleIndex >= _samples.Count)
                continue;

            SampleRecord sample = _samples[sampleIndex];
            float projectionOffset = Mathf.Abs(Vector3.Dot(sample.worldPosition - island.plane.center, island.plane.normal));
            if (projectionOffset > maxProjectionOffset)
                continue;

            Vector3 projectedWorld = ProjectPointToPlane(sample.worldPosition, island.plane);
            int vertexIndex = _meshVertices.Count;
            vertexByCell[cellKey] = vertexIndex;
            _meshVertices.Add(displayTransform.InverseTransformPoint(projectedWorld));
            _meshColors.Add(terrainPlanarizedMeshColor);
        }

        if (vertexByCell.Count < 4)
            return 0;

        int triangles = 0;
        foreach (KeyValuePair<Vector2Int, int> pair in vertexByCell)
        {
            Vector2Int aKey = pair.Key;
            Vector2Int bKey = aKey + new Vector2Int(1, 0);
            Vector2Int cKey = aKey + new Vector2Int(0, 1);
            Vector2Int dKey = aKey + new Vector2Int(1, 1);

            if (!vertexByCell.TryGetValue(bKey, out int b) ||
                !vertexByCell.TryGetValue(cKey, out int c) ||
                !vertexByCell.TryGetValue(dKey, out int d))
            {
                continue;
            }

            int a = pair.Value;
            if (!PlanarizedMeshEdgeOk(a, b) ||
                !PlanarizedMeshEdgeOk(a, c) ||
                !PlanarizedMeshEdgeOk(b, d) ||
                !PlanarizedMeshEdgeOk(c, d))
            {
                continue;
            }

            bool diagonalAC = (_meshVertices[a] - _meshVertices[d]).sqrMagnitude <=
                              (_meshVertices[b] - _meshVertices[c]).sqrMagnitude;
            if (diagonalAC && PlanarizedMeshEdgeOk(a, d))
            {
                AddTriangle(a, b, d);
                AddTriangle(a, d, c);
                triangles += 2;
            }
            else if (!diagonalAC && PlanarizedMeshEdgeOk(b, c))
            {
                AddTriangle(a, b, c);
                AddTriangle(b, d, c);
                triangles += 2;
            }
        }

        return triangles;
    }

    private int AddPlanarizedTerrainIslandTileMesh(TerrainPlaneIsland island, Transform displayTransform)
    {
        float maxProjectionOffset = Mathf.Max(0.001f, terrainPlaneMaxProjectionOffsetMeters);
        float fillRatio = Mathf.Clamp01(terrainPlanarizedTileFillRatio);
        float cellSize = Mathf.Max(0.001f, terrainCellSizeMeters);
        var vertexByCorner = new Dictionary<Vector2Int, int>(island.cellKeys.Count * 4);
        int triangles = 0;

        for (int i = 0; i < island.cellKeys.Count; i++)
        {
            Vector2Int cellKey = island.cellKeys[i];
            if (!island.representativeIndexByCell.TryGetValue(cellKey, out int sampleIndex) ||
                sampleIndex < 0 ||
                sampleIndex >= _samples.Count)
            {
                continue;
            }

            SampleRecord sample = _samples[sampleIndex];
            float projectionOffset = Mathf.Abs(Vector3.Dot(sample.worldPosition - island.plane.center, island.plane.normal));
            if (projectionOffset > maxProjectionOffset)
                continue;

            int a = GetOrAddPlanarizedGridCornerVertex(vertexByCorner, cellKey, 0, 0, sample, island.plane, displayTransform, cellSize, fillRatio);
            int b = GetOrAddPlanarizedGridCornerVertex(vertexByCorner, cellKey, 1, 0, sample, island.plane, displayTransform, cellSize, fillRatio);
            int c = GetOrAddPlanarizedGridCornerVertex(vertexByCorner, cellKey, 1, 1, sample, island.plane, displayTransform, cellSize, fillRatio);
            int d = GetOrAddPlanarizedGridCornerVertex(vertexByCorner, cellKey, 0, 1, sample, island.plane, displayTransform, cellSize, fillRatio);
            if (a < 0 || b < 0 || c < 0 || d < 0)
                continue;

            AddTriangle(a, b, c);
            AddTriangle(a, c, d);
            triangles += 2;
        }

        return triangles;
    }

    private int GetOrAddPlanarizedGridCornerVertex(
        Dictionary<Vector2Int, int> vertexByCorner,
        Vector2Int cellKey,
        int cornerX,
        int cornerY,
        SampleRecord referenceSample,
        PlaneFitResult plane,
        Transform displayTransform,
        float cellSize,
        float fillRatio)
    {
        Vector2Int cornerKey = new Vector2Int(cellKey.x + cornerX, cellKey.y + cornerY);
        bool useSharedCorner = fillRatio >= 0.999f;
        if (useSharedCorner && vertexByCorner.TryGetValue(cornerKey, out int existing))
            return existing;

        float shrink = Mathf.Clamp01(fillRatio);
        float localX = cellKey.x + (cornerX == 0 ? (1f - shrink) * 0.5f : 1f - (1f - shrink) * 0.5f);
        float localY = cellKey.y + (cornerY == 0 ? (1f - shrink) * 0.5f : 1f - (1f - shrink) * 0.5f);
        float u = localX * cellSize;
        float v = localY * cellSize;
        Vector3 rawWorld = referenceSample.windowCenter +
                           referenceSample.windowAxisU * u +
                           referenceSample.windowAxisV * v;
        Vector3 projectedWorld = ProjectPointToPlane(rawWorld, plane);

        int index = _meshVertices.Count;
        _meshVertices.Add(displayTransform.InverseTransformPoint(projectedWorld));
        _meshColors.Add(terrainPlanarizedMeshColor);

        if (useSharedCorner)
            vertexByCorner[cornerKey] = index;

        return index;
    }

    private bool PlanarizedMeshEdgeOk(int a, int b)
    {
        float maxEdge = Mathf.Max(terrainPlaneMeshMaxEdgeMeters, terrainCellSizeMeters * 2.5f);
        return (_meshVertices[a] - _meshVertices[b]).sqrMagnitude <= maxEdge * maxEdge;
    }

    private void AddTriangle(int a, int b, int c)
    {
        _meshIndices.Add(a);
        _meshIndices.Add(b);
        _meshIndices.Add(c);
        _meshIndices.Add(c);
        _meshIndices.Add(b);
        _meshIndices.Add(a);
    }

    private void AddTerrainRepresentative(
        Vector3 worldPosition,
        Color color,
        Transform displayTransform,
        Vector3 localRight,
        Vector3 localUp,
        float halfSize)
    {
        Vector3 localPoint = displayTransform.InverseTransformPoint(worldPosition);
        AddBillboardQuad(localPoint, localRight, localUp, halfSize, color);
    }

    private void AddTerrainRepresentative(
        SampleRecord sample,
        Color color,
        Transform displayTransform,
        Vector3 localRight,
        Vector3 localUp,
        float halfSize)
    {
        Vector3 localPoint = displayTransform.InverseTransformPoint(sample.worldPosition);
        AddBillboardQuad(localPoint, localRight, localUp, halfSize, color);
    }

    private Color ResolveTerrainCellColor(TerrainCellClassification classification)
    {
        switch (classification)
        {
            case TerrainCellClassification.StableSingleLayer:
                return terrainStableCellColor;
            case TerrainCellClassification.BoundaryDepthJump:
                return terrainBoundaryCellColor;
            case TerrainCellClassification.MultiLayer:
                return terrainMultiLayerNearColor;
            case TerrainCellClassification.Unstable:
            default:
                return terrainUnstableCellColor;
        }
    }

    private bool TryBuildTerrainCells(out Dictionary<Vector2Int, TerrainCell> cells, out TerrainCellSummary summary)
    {
        cells = new Dictionary<Vector2Int, TerrainCell>();
        summary = default;
        float cellSize = Mathf.Max(0.005f, terrainCellSizeMeters);

        for (int i = 0; i < _samples.Count; i++)
        {
            SampleRecord sample = _samples[i];
            if (!sample.valid || !IsFinite(sample.worldPosition) || !TryGetTerrainCell(sample, cellSize, out Vector2Int key))
                continue;

            if (!cells.TryGetValue(key, out TerrainCell cell))
            {
                cell = new TerrainCell();
                cells.Add(key, cell);
            }

            cell.indices.Add(i);
            cell.depths.Add(sample.linearDepth);
            cell.supportFrames.Add(sample.globalFrame);
            cell.worldSum += sample.worldPosition;
            if (sample.status == SampleStatus.DepthEdge || sample.status == SampleStatus.EdgeEvidence)
                cell.depthEdgeSamples++;
        }

        foreach (KeyValuePair<Vector2Int, TerrainCell> pair in cells)
        {
            ClassifyTerrainCell(pair.Value);
        }

        MarkNeighborDepthJumpCells(cells);

        foreach (KeyValuePair<Vector2Int, TerrainCell> pair in cells)
        {
            summary.cellCount++;
            switch (pair.Value.classification)
            {
                case TerrainCellClassification.StableSingleLayer:
                    summary.stableSingleLayerCount++;
                    break;
                case TerrainCellClassification.BoundaryDepthJump:
                    summary.boundaryDepthJumpCount++;
                    break;
                case TerrainCellClassification.MultiLayer:
                    summary.multiLayerCount++;
                    break;
                case TerrainCellClassification.Unstable:
                    summary.unstableCount++;
                    break;
            }
        }

        return cells.Count > 0;
    }

    private void MarkNeighborDepthJumpCells(Dictionary<Vector2Int, TerrainCell> cells)
    {
        if (cells == null || cells.Count == 0)
            return;

        float jumpLimit = Mathf.Max(
            terrainNeighborDepthJumpMeters,
            terrainMultiLayerMinDepthGapMeters,
            minDepthEdgeJumpMeters);
        Vector2Int[] neighbors =
        {
            new Vector2Int(1, 0),
            new Vector2Int(0, 1)
        };

        foreach (KeyValuePair<Vector2Int, TerrainCell> pair in cells)
        {
            TerrainCell current = pair.Value;
            if (!CanPromoteToNeighborBoundary(current))
                continue;

            for (int i = 0; i < neighbors.Length; i++)
            {
                Vector2Int nextKey = pair.Key + neighbors[i];
                if (!cells.TryGetValue(nextKey, out TerrainCell next) ||
                    !CanPromoteToNeighborBoundary(next))
                {
                    continue;
                }

                if (Mathf.Abs(current.depthMedian - next.depthMedian) <= jumpLimit)
                    continue;

                current.classification = TerrainCellClassification.BoundaryDepthJump;
                next.classification = TerrainCellClassification.BoundaryDepthJump;
            }
        }
    }

    private static bool CanPromoteToNeighborBoundary(TerrainCell cell)
    {
        return cell != null &&
               cell.classification != TerrainCellClassification.MultiLayer &&
               cell.classification != TerrainCellClassification.Unstable;
    }

    private bool TryBuildTerrainPlaneIslands(
        Dictionary<Vector2Int, TerrainCell> cells,
        Camera cam,
        out List<TerrainPlaneIsland> islands,
        out TerrainPlaneSummary summary)
    {
        islands = new List<TerrainPlaneIsland>();
        summary = default;
        if (cells == null || cells.Count == 0)
            return false;

        var visited = new HashSet<Vector2Int>();
        var queue = new Queue<Vector2Int>();
        Vector2Int[] neighbors =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        foreach (KeyValuePair<Vector2Int, TerrainCell> pair in cells)
        {
            if (!IsStableTerrainPlaneCell(pair.Value) || !visited.Add(pair.Key))
                continue;

            var island = new TerrainPlaneIsland();
            queue.Enqueue(pair.Key);
            while (queue.Count > 0)
            {
                Vector2Int currentKey = queue.Dequeue();
                TerrainCell current = cells[currentKey];
                island.cellKeys.Add(currentKey);

                int representativeIndex = FindSampleClosestToDepth(current, current.depthMedian, float.NegativeInfinity, float.PositiveInfinity);
                if (representativeIndex >= 0)
                {
                    island.representativeIndices.Add(representativeIndex);
                    island.representativeIndexByCell[currentKey] = representativeIndex;
                }

                for (int i = 0; i < neighbors.Length; i++)
                {
                    Vector2Int nextKey = currentKey + neighbors[i];
                    if (!cells.TryGetValue(nextKey, out TerrainCell next) ||
                        !IsStableTerrainPlaneCell(next) ||
                        !AreTerrainStableCellsPlaneCompatible(current, next) ||
                        !visited.Add(nextKey))
                    {
                        continue;
                    }

                    queue.Enqueue(nextKey);
                }
            }

            if (island.cellKeys.Count < Mathf.Max(3, terrainPlaneMinCells) ||
                island.representativeIndices.Count < Mathf.Max(3, terrainPlaneMinCells))
            {
                island.accepted = false;
                islands.Add(island);
                continue;
            }

            EvaluateTerrainPlaneIsland(island, cam);
            islands.Add(island);
        }

        for (int i = 0; i < islands.Count; i++)
        {
            summary.islandCount++;
            if (islands[i].accepted)
            {
                summary.acceptedIslandCount++;
                summary.acceptedCellCount += islands[i].cellKeys.Count;
            }
            else
            {
                summary.rejectedIslandCount++;
            }
        }

        return islands.Count > 0;
    }

    private bool IsStableTerrainPlaneCell(TerrainCell cell)
    {
        return cell != null && cell.classification == TerrainCellClassification.StableSingleLayer;
    }

    private bool AreTerrainStableCellsPlaneCompatible(TerrainCell a, TerrainCell b)
    {
        if (a == null || b == null)
            return false;

        float jumpLimit = Mathf.Max(terrainNeighborDepthJumpMeters, terrainMultiLayerMinDepthGapMeters, minDepthEdgeJumpMeters);
        return Mathf.Abs(a.depthMedian - b.depthMedian) <= jumpLimit;
    }

    private void EvaluateTerrainPlaneIsland(TerrainPlaneIsland island, Camera cam)
    {
        if (island == null || island.representativeIndices.Count < 3)
            return;

        if (!TryFitTrimmedSamplePlaneFromIndices(
                island.representativeIndices,
                cam,
                terrainPlaneTrimKeepRatio,
                terrainPlaneRefineIterations,
                out PlaneFitResult plane,
                out List<int> inlierIndices))
        {
            island.accepted = false;
            return;
        }

        island.plane = plane;
        var inlierSet = new HashSet<int>(inlierIndices);
        float residualLimit = Mathf.Max(0.001f, terrainPlaneInlierResidualMeters);
        int inlierCount = 0;
        float maxOffset = 0f;

        for (int i = 0; i < island.representativeIndices.Count; i++)
        {
            int sampleIndex = island.representativeIndices[i];
            if (sampleIndex < 0 || sampleIndex >= _samples.Count)
                continue;

            float residual = Mathf.Abs(Vector3.Dot(_samples[sampleIndex].worldPosition - plane.center, plane.normal));
            if (residual <= residualLimit || inlierSet.Contains(sampleIndex))
                inlierCount++;
            if (residual > maxOffset)
                maxOffset = residual;
        }

        island.maxProjectionOffset = maxOffset;
        float inlierRatio = island.representativeIndices.Count > 0
            ? inlierCount / (float)island.representativeIndices.Count
            : 0f;

        island.accepted =
            plane.valid &&
            plane.residualP95 <= Mathf.Max(0.001f, terrainPlaneMaxResidualP95Meters) &&
            inlierRatio >= terrainPlaneMinInlierRatio &&
            maxOffset <= Mathf.Max(0.001f, terrainPlaneMaxProjectionOffsetMeters);
    }

    private static Vector3 ProjectPointToPlane(Vector3 point, PlaneFitResult plane)
    {
        return point - plane.normal * Vector3.Dot(point - plane.center, plane.normal);
    }

    private void ClassifyTerrainCell(TerrainCell cell)
    {
        int count = cell.depths.Count;
        if (count == 0)
        {
            cell.classification = TerrainCellClassification.Unstable;
            return;
        }

        cell.depths.Sort();
        cell.depthP10 = QuantileSorted(cell.depths, 0.1f);
        cell.depthMedian = QuantileSorted(cell.depths, 0.5f);
        cell.depthP90 = QuantileSorted(cell.depths, 0.9f);
        cell.thickness = cell.depthP90 - cell.depthP10;
        cell.largestDepthGap = 0f;
        cell.splitIndex = -1;

        for (int i = 0; i < cell.depths.Count - 1; i++)
        {
            float gap = cell.depths[i + 1] - cell.depths[i];
            if (gap > cell.largestDepthGap)
            {
                cell.largestDepthGap = gap;
                cell.splitIndex = i + 1;
            }
        }

        if (count < Mathf.Max(1, terrainStableMinSamplesPerCell) ||
            cell.supportFrames.Count < Mathf.Max(1, terrainStableMinSupportFrames))
        {
            cell.classification = TerrainCellClassification.Unstable;
            return;
        }

        float minLayerFraction = Mathf.Clamp01(terrainMultiLayerMinFraction);
        bool hasLayerSplit = cell.splitIndex > 0 &&
                             cell.splitIndex < count &&
                             cell.largestDepthGap >= Mathf.Max(0.001f, terrainMultiLayerMinDepthGapMeters) &&
                             cell.splitIndex / (float)count >= minLayerFraction &&
                             (count - cell.splitIndex) / (float)count >= minLayerFraction;
        if (hasLayerSplit)
        {
            cell.classification = TerrainCellClassification.MultiLayer;
            return;
        }

        float edgeFraction = count > 0 ? cell.depthEdgeSamples / (float)count : 0f;
        if (cell.thickness > Mathf.Max(0.001f, terrainStableMaxThicknessMeters) ||
            edgeFraction >= terrainBoundaryMinDepthEdgeFraction)
        {
            cell.classification = TerrainCellClassification.BoundaryDepthJump;
            return;
        }

        cell.classification = TerrainCellClassification.StableSingleLayer;
    }

    private bool TryGetTerrainCell(SampleRecord sample, float cellSize, out Vector2Int cell)
    {
        if (sample.fromPhysicalWindow)
        {
            cell = new Vector2Int(
                Mathf.FloorToInt(sample.windowLocalU / cellSize),
                Mathf.FloorToInt(sample.windowLocalV / cellSize));
            return true;
        }

        cell = new Vector2Int(
            Mathf.FloorToInt(sample.windowX / Mathf.Max(1f, cellSize * 1000f)),
            Mathf.FloorToInt(sample.windowY / Mathf.Max(1f, cellSize * 1000f)));
        return true;
    }

    private bool TryGetTerrainLayerRepresentatives(TerrainCell cell, out int nearIndex, out int farIndex)
    {
        nearIndex = -1;
        farIndex = -1;
        if (cell.splitIndex <= 0 || cell.splitIndex >= cell.depths.Count)
            return false;

        float splitDepth = (cell.depths[cell.splitIndex - 1] + cell.depths[cell.splitIndex]) * 0.5f;
        float nearDepth = cell.depths[Mathf.Clamp(cell.splitIndex / 2, 0, cell.splitIndex - 1)];
        float farDepth = cell.depths[Mathf.Clamp(cell.splitIndex + (cell.depths.Count - cell.splitIndex) / 2, cell.splitIndex, cell.depths.Count - 1)];
        nearIndex = FindSampleClosestToDepth(cell, nearDepth, float.NegativeInfinity, splitDepth);
        farIndex = FindSampleClosestToDepth(cell, farDepth, splitDepth, float.PositiveInfinity);
        return nearIndex >= 0 && farIndex >= 0;
    }

    private int FindSampleClosestToDepth(TerrainCell cell, float targetDepth, float minDepthInclusive, float maxDepthExclusive)
    {
        int bestIndex = -1;
        float bestDelta = float.PositiveInfinity;

        for (int i = 0; i < cell.indices.Count; i++)
        {
            int sampleIndex = cell.indices[i];
            float depth = _samples[sampleIndex].linearDepth;
            if (depth < minDepthInclusive || depth >= maxDepthExclusive)
                continue;

            float delta = Mathf.Abs(depth - targetDepth);
            if (delta < bestDelta)
            {
                bestDelta = delta;
                bestIndex = sampleIndex;
            }
        }

        return bestIndex;
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

    private bool TryAddTrustedThinBox(Transform displayTransform, Camera cam)
    {
        bool drew = TryAddTrustedThinBoxForStatus(
            displayTransform,
            cam,
            SampleStatus.TrustedPlane,
            trustedThinBoxColor,
            Mathf.Max(1, trustedThinBoxMaxGroups),
            Mathf.Max(3, trustedThinBoxMinGroupSamples));

        if (enableSecondaryPlaneCandidate)
        {
            int secondaryMinGroupSamples = Mathf.Max(3, Mathf.Min(trustedThinBoxMinGroupSamples, minSecondaryPlaneSamples));
            drew |= TryAddTrustedThinBoxForStatus(
                displayTransform,
                cam,
                SampleStatus.SecondaryTrustedPlane,
                secondaryThinBoxColor,
                Mathf.Max(1, secondaryPlaneMaxThinBoxGroups),
                secondaryMinGroupSamples);
        }

        return drew;
    }

    private bool TryAddTrustedThinBoxForStatus(
        Transform displayTransform,
        Camera cam,
        SampleStatus trustedStatus,
        Color boxColor,
        int maxGroups,
        int minGroupSamples)
    {
        float cellSize = Mathf.Max(0.005f, trustedThinBoxGroupCellSizeMeters);
        int minCellSamples = Mathf.Max(1, trustedThinBoxMinSamplesPerGroupCell);
        var cells = new Dictionary<Vector2Int, TrustedThinBoxCell>();

        for (int i = 0; i < _samples.Count; i++)
        {
            SampleRecord sample = _samples[i];
            if (!sample.valid ||
                sample.status != trustedStatus ||
                !IsFinite(sample.worldPosition) ||
                !TryGetTrustedThinBoxCell(sample, cellSize, out Vector2Int cell))
            {
                continue;
            }

            if (!cells.TryGetValue(cell, out TrustedThinBoxCell cellStats))
            {
                cellStats = new TrustedThinBoxCell();
                cells.Add(cell, cellStats);
            }

            cellStats.indices.Add(i);
            cellStats.worldSum += sample.worldPosition;
            cellStats.windowNormalOffsetSum += GetSampleWindowNormalOffset(sample);
            cellStats.linearDepthSum += sample.linearDepth;
        }

        if (cells.Count == 0)
            return false;

        var usableCells = new Dictionary<Vector2Int, TrustedThinBoxCell>();
        foreach (KeyValuePair<Vector2Int, TrustedThinBoxCell> pair in cells)
        {
            if (pair.Value.indices.Count >= minCellSamples)
                usableCells.Add(pair.Key, pair.Value);
        }

        if (usableCells.Count == 0)
            return false;

        var visited = new HashSet<Vector2Int>();
        var components = new List<List<int>>();
        var queue = new Queue<Vector2Int>();
        Vector2Int[] neighbors =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        foreach (KeyValuePair<Vector2Int, TrustedThinBoxCell> pair in usableCells)
        {
            if (!visited.Add(pair.Key))
                continue;

            var component = new List<int>();
            queue.Enqueue(pair.Key);
            while (queue.Count > 0)
            {
                Vector2Int current = queue.Dequeue();
                TrustedThinBoxCell currentStats = usableCells[current];
                component.AddRange(currentStats.indices);

                for (int i = 0; i < neighbors.Length; i++)
                {
                    Vector2Int next = current + neighbors[i];
                    if (!usableCells.TryGetValue(next, out TrustedThinBoxCell nextStats) ||
                        !AreTrustedThinBoxCellsCompatible(currentStats, nextStats) ||
                        !visited.Add(next))
                    {
                        continue;
                    }

                    queue.Enqueue(next);
                }
            }

            components.Add(component);
        }

        components.Sort((a, b) => b.Count.CompareTo(a.Count));

        int groupsDrawn = 0;
        for (int i = 0; i < components.Count && groupsDrawn < maxGroups; i++)
        {
            List<int> component = components[i];
            if (component.Count < minGroupSamples)
                continue;

            if (TryAddTrustedThinBoxForGroup(displayTransform, cam, component, boxColor))
                groupsDrawn++;
        }

        return groupsDrawn > 0;
    }

    private bool AreTrustedThinBoxCellsCompatible(TrustedThinBoxCell a, TrustedThinBoxCell b)
    {
        float depthJumpLimit = Mathf.Max(0.001f, trustedThinBoxMaxNeighborDepthJumpMeters);
        float normalOffsetJump = Mathf.Abs(a.MeanWindowNormalOffset - b.MeanWindowNormalOffset);
        if (normalOffsetJump > depthJumpLimit)
            return false;

        float linearDepthJumpLimit = Mathf.Max(0.001f, trustedThinBoxMaxNeighborLinearDepthJumpMeters);
        if (Mathf.Abs(a.MeanLinearDepth - b.MeanLinearDepth) > linearDepthJumpLimit)
            return false;

        float worldGapLimit = Mathf.Max(trustedThinBoxGroupCellSizeMeters * 2f, trustedThinBoxMaxNeighborWorldGapMeters);
        if ((a.MeanWorld - b.MeanWorld).sqrMagnitude > worldGapLimit * worldGapLimit)
            return false;

        return true;
    }

    private static float GetSampleWindowNormalOffset(SampleRecord sample)
    {
        Vector3 normal = Vector3.Cross(sample.windowAxisU, sample.windowAxisV);
        if (normal.sqrMagnitude < 1e-8f)
            return sample.linearDepth;

        return Vector3.Dot(sample.worldPosition - sample.windowCenter, normal.normalized);
    }

    private bool TryGetTrustedThinBoxCell(SampleRecord sample, float cellSize, out Vector2Int cell)
    {
        if (sample.fromPhysicalWindow)
        {
            cell = new Vector2Int(
                Mathf.FloorToInt(sample.windowLocalU / cellSize),
                Mathf.FloorToInt(sample.windowLocalV / cellSize));
            return true;
        }

        cell = new Vector2Int(
            Mathf.FloorToInt(sample.windowX / 4f),
            Mathf.FloorToInt(sample.windowY / 4f));
        return true;
    }

    private bool TryAddTrustedThinBoxForGroup(Transform displayTransform, Camera cam, List<int> trustedIndices, Color boxColor)
    {
        PlaneFitResult plane;
        List<int> boxIndices;
        bool fitOk;
        if (trustedThinBoxUseTrimmedPlaneFit)
        {
            fitOk = TryFitTrimmedSamplePlaneFromIndices(
                trustedIndices,
                cam,
                trustedThinBoxPlaneTrimKeepRatio,
                trustedThinBoxPlaneRefineIterations,
                out plane,
                out boxIndices);
        }
        else
        {
            fitOk = TryFitSamplePlaneFromIndices(trustedIndices, cam, out plane);
            boxIndices = trustedIndices;
        }

        if (!fitOk || boxIndices == null)
            return false;

        if (plane.residualP95 > Mathf.Max(0.001f, trustedThinBoxMaxGroupResidualP95Meters))
            return false;

        int count = boxIndices.Count;
        if (count < Mathf.Max(3, minTrustedSamplesPerSurfaceCell * 2))
            return false;

        float extentResidualLimit = Mathf.Max(
            trustedThinBoxMaxGroupResidualP95Meters,
            plane.residualP95 * Mathf.Max(1f, trustedThinBoxExtentInlierResidualMultiplier));
        float minU = float.PositiveInfinity;
        float minV = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float maxV = float.NegativeInfinity;
        var normalOffsets = new List<float>(count);

        for (int i = 0; i < count; i++)
        {
            SampleRecord sample = _samples[boxIndices[i]];
            Vector3 delta = sample.worldPosition - plane.center;
            float residual = Mathf.Abs(Vector3.Dot(delta, plane.normal));
            if (residual > extentResidualLimit)
                continue;

            float u = Vector3.Dot(delta, plane.axisU);
            float v = Vector3.Dot(delta, plane.axisV);
            float n = Vector3.Dot(delta, plane.normal);

            minU = Mathf.Min(minU, u);
            minV = Mathf.Min(minV, v);
            maxU = Mathf.Max(maxU, u);
            maxV = Mathf.Max(maxV, v);
            normalOffsets.Add(n);
        }

        if (normalOffsets.Count < Mathf.Max(3, minTrustedSamplesPerSurfaceCell * 2))
            return false;

        if (!float.IsFinite(minU) || !float.IsFinite(minV) ||
            !float.IsFinite(maxU) || !float.IsFinite(maxV))
        {
            return false;
        }

        float rawSpanU = maxU - minU;
        float rawSpanV = maxV - minV;
        float maxSpan = Mathf.Max(rawSpanU, rawSpanV);
        float minSpanForRatio = Mathf.Max(0.005f, Mathf.Min(rawSpanU, rawSpanV));
        if (maxSpan > Mathf.Max(0.05f, trustedThinBoxMaxSpanMeters))
            return false;
        if (maxSpan / minSpanForRatio > Mathf.Max(1f, trustedThinBoxMaxAspectRatio))
            return false;

        float padding = Mathf.Max(0f, trustedThinBoxPaddingMeters);
        float minSpan = Mathf.Max(0.01f, trustedThinBoxMinSpanMeters);
        if (maxU - minU < minSpan)
        {
            float centerU = (minU + maxU) * 0.5f;
            minU = centerU - minSpan * 0.5f;
            maxU = centerU + minSpan * 0.5f;
        }

        if (maxV - minV < minSpan)
        {
            float centerV = (minV + maxV) * 0.5f;
            minV = centerV - minSpan * 0.5f;
            maxV = centerV + minSpan * 0.5f;
        }

        minU -= padding;
        minV -= padding;
        maxU += padding;
        maxV += padding;

        normalOffsets.Sort();
        float lowN = QuantileSorted(normalOffsets, 0.05f);
        float highN = QuantileSorted(normalOffsets, 0.95f);
        float centerN = (lowN + highN) * 0.5f;
        float halfThickness = Mathf.Max(
            Mathf.Max(trustedThinBoxMinThicknessMeters * 0.5f, (highN - lowN) * 0.5f),
            plane.residualP95);
        lowN = centerN - halfThickness;
        highN = centerN + halfThickness;

        Vector3 p000 = OrientedBoxPoint(plane, minU, minV, lowN);
        Vector3 p100 = OrientedBoxPoint(plane, maxU, minV, lowN);
        Vector3 p110 = OrientedBoxPoint(plane, maxU, maxV, lowN);
        Vector3 p010 = OrientedBoxPoint(plane, minU, maxV, lowN);
        Vector3 p001 = OrientedBoxPoint(plane, minU, minV, highN);
        Vector3 p101 = OrientedBoxPoint(plane, maxU, minV, highN);
        Vector3 p111 = OrientedBoxPoint(plane, maxU, maxV, highN);
        Vector3 p011 = OrientedBoxPoint(plane, minU, maxV, highN);

        float width = Mathf.Max(0.001f, trustedThinBoxLineWidthMeters);
        Color color = boxColor;
        AddWorldLineQuad(p000, p100, width, color, cam, displayTransform);
        AddWorldLineQuad(p100, p110, width, color, cam, displayTransform);
        AddWorldLineQuad(p110, p010, width, color, cam, displayTransform);
        AddWorldLineQuad(p010, p000, width, color, cam, displayTransform);

        AddWorldLineQuad(p001, p101, width, color, cam, displayTransform);
        AddWorldLineQuad(p101, p111, width, color, cam, displayTransform);
        AddWorldLineQuad(p111, p011, width, color, cam, displayTransform);
        AddWorldLineQuad(p011, p001, width, color, cam, displayTransform);

        AddWorldLineQuad(p000, p001, width, color, cam, displayTransform);
        AddWorldLineQuad(p100, p101, width, color, cam, displayTransform);
        AddWorldLineQuad(p110, p111, width, color, cam, displayTransform);
        AddWorldLineQuad(p010, p011, width, color, cam, displayTransform);
        return true;
    }

    private static Vector3 OrientedBoxPoint(PlaneFitResult plane, float u, float v, float n)
    {
        return plane.center + plane.axisU * u + plane.axisV * v + plane.normal * n;
    }

    private void AddWorldLineQuad(Vector3 worldA, Vector3 worldB, float width, Color color, Camera cam, Transform displayTransform)
    {
        Vector3 line = worldB - worldA;
        if (line.sqrMagnitude < 1e-8f)
            return;

        Vector3 lineDir = line.normalized;
        Vector3 mid = (worldA + worldB) * 0.5f;
        Vector3 toCamera = cam != null ? cam.transform.position - mid : Vector3.up;
        Vector3 side = Vector3.Cross(lineDir, toCamera.sqrMagnitude > 1e-8f ? toCamera.normalized : Vector3.up);
        if (side.sqrMagnitude < 1e-8f)
            side = Vector3.Cross(lineDir, Vector3.up);
        if (side.sqrMagnitude < 1e-8f)
            side = Vector3.Cross(lineDir, Vector3.right);
        if (side.sqrMagnitude < 1e-8f)
            return;

        Vector3 localA = displayTransform.InverseTransformPoint(worldA);
        Vector3 localB = displayTransform.InverseTransformPoint(worldB);
        Vector3 localOffset = displayTransform.InverseTransformDirection(side.normalized * (width * 0.5f));

        int start = _meshVertices.Count;
        _meshVertices.Add(localA - localOffset);
        _meshVertices.Add(localA + localOffset);
        _meshVertices.Add(localB + localOffset);
        _meshVertices.Add(localB - localOffset);

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

        Camera cam = centerCamera != null ? centerCamera : Camera.main;
        return TryFitSamplePlaneFromIndices(validIndices, cam, out result);
    }

    private bool TryFitTrustedSamplePlane(Camera cam, out PlaneFitResult result, out List<int> trustedIndices)
    {
        result = default;
        trustedIndices = new List<int>(_samples.Count);

        for (int i = 0; i < _samples.Count; i++)
        {
            if (_samples[i].valid &&
                _samples[i].status == SampleStatus.TrustedPlane &&
                IsFinite(_samples[i].worldPosition))
            {
                trustedIndices.Add(i);
            }
        }

        return TryFitSamplePlaneFromIndices(trustedIndices, cam, out result);
    }

    private bool TryFitSamplePlaneFromIndices(List<int> indices, Camera cam, out PlaneFitResult result)
    {
        result = default;
        int count = indices != null ? indices.Count : 0;
        if (count < 3)
            return false;

        Vector3 center = Vector3.zero;
        for (int i = 0; i < count; i++)
            center += _samples[indices[i]].worldPosition;
        center /= count;

        float xx = 0f;
        float xy = 0f;
        float xz = 0f;
        float yy = 0f;
        float yz = 0f;
        float zz = 0f;

        for (int i = 0; i < count; i++)
        {
            Vector3 d = _samples[indices[i]].worldPosition - center;
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
            float residual = Mathf.Abs(Vector3.Dot(_samples[indices[i]].worldPosition - center, normal));
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

    private bool TryFitTrimmedSamplePlaneFromIndices(
        List<int> indices,
        Camera cam,
        float keepRatio,
        int iterations,
        out PlaneFitResult result,
        out List<int> inlierIndices)
    {
        result = default;
        inlierIndices = null;

        int count = indices != null ? indices.Count : 0;
        if (count < 3)
            return false;

        var sourceIndices = new List<int>(count);
        for (int i = 0; i < count; i++)
        {
            int sampleIndex = indices[i];
            if (sampleIndex < 0 ||
                sampleIndex >= _samples.Count ||
                !_samples[sampleIndex].valid ||
                !IsFinite(_samples[sampleIndex].worldPosition))
            {
                continue;
            }

            sourceIndices.Add(sampleIndex);
        }

        if (sourceIndices.Count < 3)
            return false;

        var workingIndices = new List<int>(sourceIndices);
        if (!TryFitSamplePlaneFromIndices(workingIndices, cam, out result))
            return false;

        float clampedKeepRatio = Mathf.Clamp(keepRatio, 0.5f, 1f);
        int keepCount = Mathf.Clamp(Mathf.CeilToInt(sourceIndices.Count * clampedKeepRatio), 3, sourceIndices.Count);
        int refineIterations = Mathf.Max(0, iterations);
        var residuals = new List<IndexResidual>(sourceIndices.Count);

        for (int iteration = 0; iteration < refineIterations; iteration++)
        {
            residuals.Clear();
            for (int i = 0; i < sourceIndices.Count; i++)
            {
                int sampleIndex = sourceIndices[i];
                float residual = Mathf.Abs(Vector3.Dot(_samples[sampleIndex].worldPosition - result.center, result.normal));
                if (!float.IsFinite(residual))
                    continue;

                residuals.Add(new IndexResidual
                {
                    index = sampleIndex,
                    residual = residual
                });
            }

            if (residuals.Count < 3)
                return false;

            residuals.Sort((a, b) => a.residual.CompareTo(b.residual));

            int currentKeepCount = Mathf.Min(keepCount, residuals.Count);
            workingIndices.Clear();
            for (int i = 0; i < currentKeepCount; i++)
                workingIndices.Add(residuals[i].index);

            if (workingIndices.Count < 3 ||
                !TryFitSamplePlaneFromIndices(workingIndices, cam, out result))
            {
                return false;
            }
        }

        inlierIndices = workingIndices;
        return result.valid && inlierIndices.Count >= 3;
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
        if (!EffectiveShowSamplingWindow)
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
            string terrainCsvPath = Path.Combine(folder, $"point_burst_{stamp}_terrain_cells.csv");
            string summaryPath = Path.Combine(folder, $"point_burst_{stamp}_summary.json");

            File.WriteAllText(csvPath, BuildCsv(), Encoding.UTF8);
            File.WriteAllText(terrainCsvPath, BuildTerrainCellsCsv(), Encoding.UTF8);
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

    private string BuildTerrainCellsCsv()
    {
        if (!TryBuildTerrainCells(out Dictionary<Vector2Int, TerrainCell> cells, out _))
            return "cellX,cellY,sampleCount,supportFrames,depthP10,depthMedian,depthP90,thickness,largestDepthGap,depthEdgeFraction,classification,worldX,worldY,worldZ\n";

        var sb = new StringBuilder(256 + cells.Count * 128);
        sb.AppendLine("cellX,cellY,sampleCount,supportFrames,depthP10,depthMedian,depthP90,thickness,largestDepthGap,depthEdgeFraction,classification,worldX,worldY,worldZ");
        foreach (KeyValuePair<Vector2Int, TerrainCell> pair in cells)
        {
            TerrainCell cell = pair.Value;
            float edgeFraction = cell.indices.Count > 0 ? cell.depthEdgeSamples / (float)cell.indices.Count : 0f;
            Append(sb, pair.Key.x).Append(',');
            Append(sb, pair.Key.y).Append(',');
            Append(sb, cell.indices.Count).Append(',');
            Append(sb, cell.supportFrames.Count).Append(',');
            Append(sb, cell.depthP10).Append(',');
            Append(sb, cell.depthMedian).Append(',');
            Append(sb, cell.depthP90).Append(',');
            Append(sb, cell.thickness).Append(',');
            Append(sb, cell.largestDepthGap).Append(',');
            Append(sb, edgeFraction).Append(',');
            sb.Append(cell.classification).Append(',');
            Vector3 mean = cell.MeanWorld;
            Append(sb, mean.x).Append(',');
            Append(sb, mean.y).Append(',');
            Append(sb, mean.z).AppendLine();
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

        TerrainPlaneSummary terrainPlaneSummary = default;
        if (TryBuildTerrainCells(out Dictionary<Vector2Int, TerrainCell> terrainCells, out TerrainCellSummary terrainSummary))
        {
            Camera cam = ResolveCenterCamera();
            TryBuildTerrainPlaneIslands(terrainCells, cam, out _, out terrainPlaneSummary);
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
               $"  \"effectivePhysicalWindowSizeMeters\": {EffectivePhysicalWindowSizeMeters.ToString(CultureInfo.InvariantCulture)},\n" +
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
               $"  \"effectiveSamplesPerFrame\": {EffectiveSamplesPerFrame},\n" +
               $"  \"effectiveFramesPerBurst\": {EffectiveFramesPerBurst},\n" +
               $"  \"effectiveMaxAccumulatedPoints\": {EffectiveMaxAccumulatedPoints},\n" +
               $"  \"stratifiedJitter\": {stratifiedJitter.ToString().ToLowerInvariant()},\n" +
               $"  \"effectiveStratifiedJitter\": {EffectiveStratifiedJitter.ToString().ToLowerInvariant()},\n" +
               $"  \"pointCloudTerrainDiagnosticsOnly\": {pointCloudTerrainDiagnosticsOnly.ToString().ToLowerInvariant()},\n" +
               $"  \"terrainDiagnosticsDisplayAggregatedCells\": {terrainDiagnosticsDisplayAggregatedCells.ToString().ToLowerInvariant()},\n" +
               $"  \"terrainCellSizeMeters\": {terrainCellSizeMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainStableMinSamplesPerCell\": {terrainStableMinSamplesPerCell},\n" +
               $"  \"terrainStableMinSupportFrames\": {terrainStableMinSupportFrames},\n" +
               $"  \"terrainStableMaxThicknessMeters\": {terrainStableMaxThicknessMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainMultiLayerMinDepthGapMeters\": {terrainMultiLayerMinDepthGapMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainNeighborDepthJumpMeters\": {terrainNeighborDepthJumpMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainDiagnosticsShowPlanarizedPreview\": {terrainDiagnosticsShowPlanarizedPreview.ToString().ToLowerInvariant()},\n" +
               $"  \"terrainDiagnosticsShowPlanarizedMeshPreview\": {terrainDiagnosticsShowPlanarizedMeshPreview.ToString().ToLowerInvariant()},\n" +
               $"  \"terrainDiagnosticsUsePlanarizedCellTiles\": {terrainDiagnosticsUsePlanarizedCellTiles.ToString().ToLowerInvariant()},\n" +
               $"  \"terrainPlanarizedTileFillRatio\": {terrainPlanarizedTileFillRatio.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainPlaneMeshMaxEdgeMeters\": {terrainPlaneMeshMaxEdgeMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainPlaneMinCells\": {terrainPlaneMinCells},\n" +
               $"  \"terrainPlaneMaxResidualP95Meters\": {terrainPlaneMaxResidualP95Meters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainPlaneMinInlierRatio\": {terrainPlaneMinInlierRatio.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainPlaneMaxProjectionOffsetMeters\": {terrainPlaneMaxProjectionOffsetMeters.ToString(CultureInfo.InvariantCulture)},\n" +
               $"  \"terrainPlaneIslandCount\": {terrainPlaneSummary.islandCount},\n" +
               $"  \"terrainAcceptedPlaneIslandCount\": {terrainPlaneSummary.acceptedIslandCount},\n" +
               $"  \"terrainRejectedPlaneIslandCount\": {terrainPlaneSummary.rejectedIslandCount},\n" +
               $"  \"terrainAcceptedPlaneCellCount\": {terrainPlaneSummary.acceptedCellCount},\n" +
               $"  \"terrainCellCount\": {terrainSummary.cellCount},\n" +
               $"  \"terrainStableSingleLayerCellCount\": {terrainSummary.stableSingleLayerCount},\n" +
               $"  \"terrainBoundaryDepthJumpCellCount\": {terrainSummary.boundaryDepthJumpCount},\n" +
               $"  \"terrainMultiLayerCellCount\": {terrainSummary.multiLayerCount},\n" +
               $"  \"terrainUnstableCellCount\": {terrainSummary.unstableCount},\n" +
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

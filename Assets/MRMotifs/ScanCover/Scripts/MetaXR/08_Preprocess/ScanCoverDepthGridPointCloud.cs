using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using System;

[DefaultExecutionOrder(-43)]
[DisallowMultipleComponent]
public sealed class ScanCoverDepthGridPointCloud : MonoBehaviour
{
    [Serializable]
    public struct GridStateEntry
    {
        public int index;
        public int group;
        public int row;
        public int col;
        public bool valid;
        public Vector3 worldPos;
        public Vector3 normal;
        public float confidence;
    }

    [Serializable]
    public sealed class GridStateSnapshot
    {
        public string componentName;
        public string samplingMode;
        public int frameIndex;
        public int resolutionWidth;
        public int resolutionHeight;
        public int cellCount;
        public int visibleCount;
        public GridStateEntry[] entries;
    }

    [Serializable]
    public sealed class RawDepthFrameSnapshot
    {
        public string componentName;
        public int frameIndex;
        public int resolutionWidth;
        public int resolutionHeight;
        public double snapshotRealtimeSeconds;
        public bool hasSnapshotCameraPose;
        public Vector3 snapshotCameraPosition;
        public Quaternion snapshotCameraRotation;
        // Capture provenance. The legacy snapshot fields above intentionally keep
        // their completion-time meaning; these fields identify the pose/matrices
        // that actually produced the GPU sample.
        public int sourceEyeIndex;
        public double dispatchRealtimeSeconds;
        public double completionRealtimeSeconds;
        public bool hasDispatchCameraPose;
        public Vector3 dispatchCameraPosition;
        public Quaternion dispatchCameraRotation;
        public bool hasCompletionCameraPose;
        public Vector3 completionCameraPosition;
        public Quaternion completionCameraRotation;
        public bool hasProjectionMatrix;
        public Matrix4x4 projectionMatrix;
        public bool hasWorldToCameraMatrix;
        public Matrix4x4 worldToCameraMatrix;
        public bool hasDepthReprojectionMatrix;
        public Matrix4x4 depthReprojectionMatrix;
        public bool hasDispatchEyePosition;
        public Vector3 dispatchEyePosition;
        // Legacy production fields remain the filtered point and raw finite-
        // difference normal. The paper DMC shadow uses the two explicit fields
        // below and never substitutes the production values.
        public Vector3[] worldPositionsRaw;
        public Vector3[] worldPositions;
        public Vector3[] worldNormals;
        public Vector3[] worldNormalsNeighbour;
        public bool[] worldNormalsNeighbourValid;
        public Color[] observationMeta;
    }

    private enum SamplingMode { RegularGrid, AdaptiveTiles, ViewLockedVolume }
    private enum VolumeFace { Front, Left, Right, Top, Bottom }
    private enum GridInteriorDisplayMode { Hidden, Mesh }
    private struct Cell { public int minX, maxX, minY, maxY, centerX, centerY, group, row, col; public bool hasSubpixelCenter; public float centerXF, centerYF; public VolumeFace face; }
    private struct GridGroup { public int startIndex, columns, rows, group; public VolumeFace face; }
    private struct VolumeBin { public bool hasValue; public int sampleCount; public float weightSum, confidenceSum, bestPlaneDistance; public Vector3 weightedPosition, weightedNormal; }
    private struct DirectGridPlaneCandidate { public int group, startRow, endRow, startCol, endCol, pointCount; public float score, halfWidth, halfHeight; public Vector3 center, normal, right, up; }
    private struct PlaneFamilySample { public Vector3 point, normal; public bool active; }
    private struct PlaneFamilyModel { public Vector3 center, normal; public float d; public int inlierCount; }
    private struct PlaneFamilyProjection { public bool valid; public Vector3 right, up; public float minU, maxU, minV, maxV; }

    [Header("Refs")] [SerializeField] private ScanCoverDepthPreprocessor preprocessor; [SerializeField] private GameObject markerPrefab; [SerializeField] private Transform markerParent;
[Header("Sampling")] [SerializeField] private SamplingMode samplingMode = SamplingMode.RegularGrid; [SerializeField, Min(1)] private int stridePixels = 4; [SerializeField, Min(0)] private int stridePixelsX = 0; [SerializeField, Min(0)] private int stridePixelsY = 0; [SerializeField, Min(0)] private int regularGridMaxColumns = 30; [SerializeField, Min(0)] private int regularGridMaxRows = 40; [SerializeField] private bool centerRegularGridWindow = true; [SerializeField] private bool centerRegularGridWindowOnHeadsetForward = true; [SerializeField] private bool regularGridUseViewportCoverage = true; [SerializeField, Range(0.25f, 1.25f)] private float regularGridViewportCoverageScale = 1f; [SerializeField] private bool regularGridUseFixedWorldSize = false; [SerializeField] private bool regularGridUseDepthHitPlaneOnly = false; [SerializeField] private bool regularGridUseVerticalDepthPlaneExperiment = false; [SerializeField] private bool regularGridUseSurfacePlaneAxesForVerticalDepthGrid = false; [SerializeField, Range(0f, 1f)] private float verticalDepthGridSurfaceAxisMinFacingDot = 0.18f; [SerializeField, Range(0f, 1f)] private float verticalDepthGridSurfaceAxisMaxWorldUpDot = 0.82f; [SerializeField, Range(0.25f, 2f)] private float verticalDepthGridSampleRadiusMultiplier = 0.85f; [SerializeField, Min(0.02f)] private float verticalDepthGridMaxForwardOffsetMeters = 2.5f; [SerializeField, Min(0.005f)] private float regularGridWorldCellSizeMeters = 0.04f; [SerializeField, Range(0.25f, 1.5f)] private float regularGridFixedWorldWindowScale = 0.5f; [SerializeField] private bool regularGridUseSmoothDistanceScale = true; [SerializeField, Range(0.01f, 4f)] private float regularGridFarMinStepPixels = 0.05f; [SerializeField, Range(0.01f, 4f)] private float regularGridStepSoftFloorPixels = 0.1f; [SerializeField, Min(1)] private int regularGridFixedWorldDepthFillRadiusPixels = 4; [SerializeField] private bool depthPixelVFlip = false;
    [Header("Adaptive Tiles")] [SerializeField, Min(2)] private int adaptiveTileSizePixels = 8; [SerializeField, Min(1)] private int adaptiveTileSampleStride = 1; [SerializeField, Range(0f, 1f)] private float adaptiveMinTileValidRatio = 0.25f; [SerializeField, Range(-1f, 1f)] private float adaptiveMinNormalDot = 0.72f; [SerializeField, Min(0f)] private float adaptiveMaxPlaneDeviationMeters = 0.03f;
    [Header("View Locked Volume")] [SerializeField] private Transform viewLockedOrigin; [SerializeField] private bool volumeLockPitch = true; [SerializeField] private bool volumeLockRoll = true; [SerializeField, Min(0.05f)] private float volumeForwardOffsetMeters = 1.2f; [SerializeField] private Vector3 volumeHalfExtents = new Vector3(1.2f, 0.9f, 0.8f); [SerializeField, Min(0.02f)] private float volumeFaceSampleMeters = 0.04f; [SerializeField, Min(0f)] private float volumeCapturePaddingMeters = 0.08f; [SerializeField] private bool volumeUseSideFaces = true; [SerializeField] private bool volumeUseTopBottomFaces = true; [SerializeField, Range(0f, 1f)] private float volumeMinBinCoverage = 0.2f; [SerializeField, Range(-1f, 1f)] private float volumeMinFaceNormalDot = -1f; [SerializeField, Min(0f)] private float volumePlaneBlendMeters = 0.06f;
    [Header("Filter")] [SerializeField, Range(0f, 1f)] private float minConfidence = 0.2f; [SerializeField, Min(0f)] private float minLinearDepthMeters = 0.35f; [SerializeField, Min(0f)] private float maxLinearDepthMeters = 5f; [SerializeField] private bool requireValidNormal = false; [SerializeField] private bool neighborFill = true; [SerializeField, Min(1)] private int neighborRadiusPixels = 1;
    [Header("Markers")] [SerializeField] private bool showMarkers = false; [SerializeField] private bool updateEveryFrame = true; [SerializeField] private bool orientToNormal = false; [SerializeField, Min(0f)] private float surfaceBiasMeters = 0.0015f; [SerializeField, Min(0.001f)] private float fallbackMarkerScaleMeters = 0.012f; [SerializeField] private Color snapshotGridUniformColor = new Color(0.55f, 0.55f, 0.55f, 1f);
    [Header("Preview Visibility")] [SerializeField] private bool previewDisplayVisible = true;
    [Header("Display Frame")] [SerializeField] private bool useWorldSpaceDisplayRoots = false; [SerializeField] private bool lockUnfrozenDisplayRoll = false; [SerializeField] private bool lockUnfrozenDisplayPitch = false; [SerializeField] private bool lockUnfrozenDisplayYaw = false; [SerializeField] private bool compensateRegularGridRollSampling = false; [SerializeField] private bool showRuntimeAxisDebug = false; [SerializeField] private Vector3 runtimeAxisOffset = new Vector3(0f, 0f, 0.8f); [SerializeField, Min(0.01f)] private float runtimeAxisCubeSize = 0.12f; [SerializeField, Min(0.02f)] private float runtimeAxisLength = 0.4f; [SerializeField, Min(0.002f)] private float runtimeAxisThickness = 0.03f;
    [Header("Grid Lines")] [SerializeField] private bool showGridLines = false; [SerializeField] private bool showGridOuterContourOnly = true; [SerializeField] private bool showRectilinearFillInsideContour = false; [SerializeField, Range(1, 4)] private int rectilinearRennetStride = 2; [SerializeField, Range(0f, 1f)] private float rectilinearRennetMinNormalDot = 0.45f; [SerializeField, Range(1f, 8f)] private float rectilinearRennetMaxEdgeSpanMultiplier = 3f; [SerializeField] private bool showGridTriangulation = false; [SerializeField] private bool rectifyGridLinesAfterDepthWrap = false; [SerializeField] private bool gridLineRequireContinuousSurface = true; [SerializeField, Range(1.05f, 6f)] private float rectifiedGridLineMaxSpanMultiplier = 2.35f; [SerializeField, Range(0f, 1f)] private float rectifiedGridLineMinValidRatio = 0.18f; [SerializeField, Range(-1f, 1f)] private float gridLineMinNeighborNormalDot = 0.2f; [SerializeField] private bool gridLineRequireCompleteCellSupport = false; [SerializeField, Min(1)] private int gridLineMinCompleteCellIslandCount = 8; [SerializeField, Min(1)] private int gridLineKeepLargestCompleteCellIslands = 1; [SerializeField] private Material gridLineMaterialOverride; [SerializeField] private Color gridLineColor = Color.white; [SerializeField, Min(0f)] private float gridLineSurfaceOffsetMeters = 0.004f; [SerializeField] private bool gridLinesRenderBehindCandidatePatch = false; [SerializeField] private bool syncGridLinesToFocusedCandidate = false; [SerializeField, Min(0f)] private float focusedGridPlaneToleranceMeters = 0.08f; [SerializeField, Min(0f)] private float focusedGridExpandMeters = 0.06f;
    [Header("Grid Interior Display")] [SerializeField] private bool showGridInteriorMesh = false; [SerializeField] private GridInteriorDisplayMode gridInteriorDisplayMode = GridInteriorDisplayMode.Hidden;
    [Header("Surface Mesh")] [SerializeField] private bool showSurfaceMesh = false; [SerializeField] private bool keepSurfaceMeshAvailableWhenHidden = false; [SerializeField] private bool useIndexConnectivity = true; [SerializeField] private Material surfaceMaterialOverride; [SerializeField, Min(0.01f)] private float maxEdgeLengthMeters = 0.20f; [SerializeField, Range(-1f, 1f)] private float minNeighborNormalDot = 0.35f; [SerializeField] private Color surfaceColor = new Color(0.18f, 0.95f, 0.98f, 1f); [SerializeField] private bool surfaceDoubleSided = true;
    [Header("Geometric Surface Grid")] [SerializeField] private bool showGeometricSurfaceGrid = true; [SerializeField] private bool geometricSurfaceGridUseRansacPatches = true; [SerializeField, Min(0.01f)] private float geometricSurfaceGridSpacingMeters = 0.12f; [SerializeField, Min(0.0005f)] private float geometricSurfaceGridLineWidthMeters = 0.0035f; [SerializeField, Min(0f)] private float geometricSurfaceGridSurfaceOffsetMeters = 0.008f; [SerializeField] private Color geometricSurfaceGridColor = new Color(1f, 1f, 1f, 0.92f); [SerializeField, Range(128, 2048)] private int ransacPatchMaxSamples = 768; [SerializeField, Range(8, 128)] private int ransacPatchIterations = 48; [SerializeField, Range(1, 4)] private int ransacPatchPlanesPerBucket = 2; [SerializeField, Range(0.005f, 0.12f)] private float ransacPatchInlierDistanceMeters = 0.035f; [SerializeField, Range(0.25f, 0.98f)] private float ransacPatchNormalDot = 0.72f; [SerializeField, Min(8)] private int ransacPatchMinInliers = 28; [SerializeField, Min(0.04f)] private float ransacPatchGridCellMeters = 0.18f; [SerializeField, Range(4, 48)] private int ransacPatchMaxGridCellsPerAxis = 24;
    [Header("Probe Row Experiment")] [SerializeField] private bool showProbeRowExperiment = false; [SerializeField] private bool probeRowUseCurrentMeshHorizontalExtent = true; [SerializeField, Min(2)] private int probeRowPointCount = 13; [SerializeField, Min(2)] private int probeRowMaxPointCount = 128; [SerializeField, Min(0.005f)] private float probeRowSpacingMeters = 0.06f; [SerializeField, Min(0f)] private float probeRowHorizontalPaddingMeters = 0.12f; [SerializeField, Min(0.05f)] private float probeRowTargetDistanceMeters = 1.2f; [SerializeField, Min(0.1f)] private float probeRowMaxRayDistanceMeters = 6f; [SerializeField, Min(0.005f)] private float probeRowMarkerScaleMeters = 0.025f; [SerializeField, Min(0.0005f)] private float probeRowLineWidthMeters = 0.006f; [SerializeField, Min(0f)] private float probeRowSurfaceOffsetMeters = 0.012f; [SerializeField, Min(0.005f)] private float probeRowRecognitionRadiusMeters = 0.09f; [SerializeField, Min(1)] private int probeRowMinNeighborhoodSamples = 3; [SerializeField, Range(-1f, 1f)] private float probeRowStableNormalDot = 0.82f; [SerializeField, Range(-1f, 1f)] private float probeRowRefineNormalDot = 0.45f; [SerializeField, Min(0.001f)] private float probeRowStablePlaneDeviationMeters = 0.025f; [SerializeField, Range(1f, 5f)] private float probeRowStableDistanceMultiplier = 1.8f; [SerializeField, Range(1f, 8f)] private float probeRowRefineDistanceMultiplier = 3.0f; [SerializeField] private Color probeRowPointColor = new Color(0f, 1f, 1f, 1f); [SerializeField] private Color probeRowStableLineColor = new Color(0.1f, 1f, 0.3f, 1f); [SerializeField] private Color probeRowRefineLineColor = new Color(1f, 0.85f, 0.05f, 1f); [SerializeField] private Color probeRowBreakLineColor = new Color(1f, 0.08f, 0.08f, 1f);
    [Header("Height Slice Contour")] [SerializeField] private bool showHeightSliceContour = false; [SerializeField] private bool heightSliceUseFrozenScreenCenterHeight = true; [SerializeField, Min(1)] private int heightSliceRowCount = 32; [SerializeField] private bool heightSliceShowPerpendicularColumns = true; [SerializeField, Min(1)] private int heightSliceColumnCount = 32; [SerializeField] private bool showHeightSlicePlaneFrame = true; [SerializeField] private bool heightSliceShowSampleColumnPlaneFrames = false; [SerializeField, Range(1, 16)] private int heightSliceSampleColumnPlaneFrameCount = 5; [SerializeField, Min(0.001f)] private float heightSliceEpsilonMeters = 0.01f; [SerializeField, Min(0.01f)] private float heightSliceMaxSegmentMeters = 0.28f; [SerializeField, Min(0.0005f)] private float heightSliceLineWidthMeters = 0.006f; [SerializeField] private Color heightSliceContourColor = new Color(0.08f, 0.85f, 1f, 0.95f); [SerializeField] private Color heightSlicePlaneFrameColor = new Color(1f, 1f, 1f, 0.35f);
    [Header("Plane Family Classification")] [SerializeField] private bool showPlaneFamilyClassification = false; [SerializeField] private bool planeFamilyDisplayAsPointQuads = false; [SerializeField, Min(0.002f)] private float planeFamilyPointSizeMeters = 0.014f; [SerializeField, Range(128, 4096)] private int planeFamilyMaxSamples = 4096; [SerializeField, Range(8, 256)] private int planeFamilyRansacIterations = 96; [SerializeField, Range(1, 16)] private int planeFamilyMaxFamilies = 10; [SerializeField, Min(12)] private int planeFamilyMinInliers = 96; [SerializeField, Range(0.005f, 0.15f)] private float planeFamilyFitDistanceMeters = 0.055f; [SerializeField, Range(0.02f, 0.18f)] private float planeFamilyClassifyDistanceMeters = 0.09f; [SerializeField, Range(0f, 60f)] private float planeFamilyClassifyNormalDegrees = 48f; [SerializeField, Range(0f, 20f)] private float planeFamilyMergeNormalDegrees = 16f; [SerializeField, Range(0.02f, 0.25f)] private float planeFamilyMergeDistanceMeters = 0.18f; [SerializeField] private bool planeFamilyUseSpatialConsistency = true; [SerializeField, Range(0, 4)] private int planeFamilySpatialSmoothingPasses = 2; [SerializeField, Min(1)] private int planeFamilyMinIslandPoints = 18; [SerializeField, Range(1, 8)] private int planeFamilyNeighborVoteThreshold = 3; [SerializeField, Min(1)] private int planeFamilyWeakIslandMaxPoints = 260; [SerializeField, Range(0f, 0.5f)] private float planeFamilyWeakIslandMaxRatio = 0.08f; [SerializeField, Range(0f, 1f)] private float planeFamilyWeakIslandBorderRatio = 0.55f; [SerializeField, Range(0f, 89f)] private float planeFamilyWeakIslandRelaxNormalDegrees = 68f; [SerializeField, Range(1f, 3f)] private float planeFamilyWeakIslandRelaxDistanceMultiplier = 1.25f; [SerializeField, Range(0.02f, 1f)] private float planeFamilySurfaceAlpha = 1f; [SerializeField] private Color planeFamilyOutlierColor = new Color(0.55f, 0.55f, 0.55f, 1f);
    [Header("Plane Family Structural Consensus")] [SerializeField] private bool planeFamilyUseStructuralConsensus = true; [SerializeField, Range(0.1f, 1f)] private float planeFamilyStrongDistanceRatio = 0.45f; [SerializeField, Min(0f)] private float planeFamilyProjectionPaddingMeters = 0.22f; [SerializeField, Range(0f, 0.25f)] private float planeFamilyNormalScoreWeight = 0.03f; [SerializeField, Min(1)] private int planeFamilyStructuralNeighborMinSame = 4; [SerializeField, Range(0f, 1f)] private float planeFamilyStructuralNeighborMinRatio = 0.45f;
    [Header("Plane Family Diagnostics")] [SerializeField] private bool planeFamilyDiagnostics = true; [SerializeField] private bool planeFamilyDiagnosticsExportCsv = true; [SerializeField] private bool planeFamilyDiagnosticsLogSummary = true; [SerializeField, Min(0.1f)] private float planeFamilyDiagnosticsMinIntervalSeconds = 2f;
    [Header("Surface Regions")] [SerializeField] private bool colorizeSurfaceRegions = false; [SerializeField] private bool showIrregularSurfaceBucket = true; [SerializeField, Range(0f, 1f)] private float irregularSurfaceBucketAlpha = 0.18f; [SerializeField, Range(1f, 89f)] private float surfaceRegionCreaseAngleDegrees = 12f; [SerializeField, Min(1)] private int surfaceRegionMinQuadCount = 12; [SerializeField, Range(1f, 45f)] private float surfaceNormalPatchMergeAngleDegrees = 10f; [SerializeField, Range(1f, 45f)] private float surfaceNormalFamilyAngleDegrees = 18f; [SerializeField, Range(0.5f, 0.999f)] private float surfaceNormalAxisBucketMinDot = 0.82f; [SerializeField, Min(1)] private int surfaceLargeComponentMinTriangleCount = 48; [SerializeField, Range(0.05f, 1f)] private float surfaceLargeComponentKeepRatio = 0.22f; [SerializeField, Min(0.001f)] private float surfaceRegionMaxNeighborDistanceMeters = 0.12f; [SerializeField, Min(0.001f)] private float surfaceRegionMaxPlaneOffsetMeters = 0.03f; [SerializeField, Range(0f, 1f)] private float surfaceRegionColorSaturation = 0.62f; [SerializeField, Range(0f, 1f)] private float surfaceRegionColorValue = 0.95f;
    [Header("Candidate Surface Display")] [SerializeField] private bool isolateTopCandidateSurfaces = false; [SerializeField, Min(0)] private int topCandidateSurfaceCount = 4; [SerializeField, Min(0)] private int topCandidateSurfaceMinTriangleCount = 48; [SerializeField] private bool candidatePreferViewCenter = true; [SerializeField, Range(0.05f, 0.75f)] private float candidateViewCenterRadius = 0.28f; [SerializeField, Range(0f, 8f)] private float candidateViewCenterScoreWeight = 1.25f; [SerializeField, Range(0f, 2f)] private float candidateFaceCountScoreWeight = 0.35f; [SerializeField, Range(0f, 8f)] private float candidateFacingScoreWeight = 0.9f; [SerializeField, Range(0f, 1f)] private float candidateMinFacingScore = 0.55f; [SerializeField, Range(0f, 1f)] private float candidateCenterFacingRelaxation = 0.12f; [SerializeField, Range(1f, 45f)] private float candidateSurfaceMergeAngleDegrees = 14f; [SerializeField, Min(0.001f)] private float candidateSurfaceMergePlaneOffsetMeters = 0.05f; [SerializeField, Min(1)] private int candidateSurfaceMergeMinSharedEdges = 2; [SerializeField] private bool rebuildLargestCandidateAsRegularGrid = true; [SerializeField] private bool largestCandidateUseTriangularLattice = true; [SerializeField] private bool largestCandidateUseOriginalGridTerrain = true; [SerializeField] private bool largestCandidateProjectRegularGridToMeshTerrain = true; [SerializeField] private bool largestCandidateShowFill = false; [SerializeField, Min(0.01f)] private float largestCandidateGridCellSizeMeters = 0.08f; [SerializeField, Min(4)] private int largestCandidateGridMaxColumns = 64; [SerializeField, Min(4)] private int largestCandidateGridMaxRows = 64; [SerializeField, Range(0.05f, 1f)] private float largestCandidateGridFillAlpha = 0.28f; [SerializeField] private Color largestCandidateGridLineColor = new Color(1f, 1f, 1f, 0.92f); [SerializeField, Min(0.0005f)] private float largestCandidateGridLineWidthMeters = 0.006f; [SerializeField] private bool largestCandidateGridShowCellDiagonals = false; [SerializeField, Range(0, 8)] private int largestCandidateGridMinNeighborCount = 2; [SerializeField, Min(0)] private int largestCandidateGridMinIslandCellCount = 3; [SerializeField, Min(0)] private int largestCandidateGridMinCellCount = 12; [SerializeField, Min(0)] private int largestCandidateGridMinIslandSpanCells = 3; [SerializeField, Min(0)] private int largestCandidateGridKeepTopIslandCount = 2; [SerializeField] private bool largestCandidateGridUseViewportFilter = true; [SerializeField, Range(0f, 1f)] private float largestCandidateGridMinViewportMaxSpan = 0.10f; [SerializeField, Range(0f, 1f)] private float largestCandidateGridMinViewportArea = 0.008f;
    [Header("Candidate Plane Objects")] [SerializeField] private bool showCandidatePlaneObjects = true; [SerializeField, Min(1)] private int candidatePlaneObjectCount = 4; [SerializeField, Min(0.01f)] private float candidatePlaneMinSizeMeters = 0.08f; [SerializeField, Min(0f)] private float candidatePlanePaddingMeters = 0.01f; [SerializeField, Min(0f)] private float candidatePlaneSurfaceOffsetMeters = 0.004f; [SerializeField, Range(0.05f, 1f)] private float candidatePlaneAlpha = 0.32f; [SerializeField] private Color candidatePlaneColor = new Color(1f, 0.18f, 0.12f, 0.32f);
    [NonSerialized] private float surfacePlaneClusterNormalDot = 0.965f;
    [NonSerialized] private float surfacePlaneClusterOffsetMeters = 0.08f;
    [Header("Surface Normal Indicators")] [SerializeField] private bool showSurfaceNormalIndicators = false; [SerializeField, Min(0.002f)] private float surfaceNormalIndicatorLengthMeters = 0.035f; [SerializeField, Min(0.0005f)] private float surfaceNormalIndicatorThicknessMeters = 0.0025f; [SerializeField] private Color surfaceNormalIndicatorColor = new Color(1f, 0.95f, 0.2f, 0.95f);
    [Header("Center Debug")] [SerializeField] private bool showCenterDebugMarkers = false; [SerializeField] private bool showScreenCenterDebugMarker = false; [SerializeField] private bool showFocusedGridCenterDebugMarker = false; [SerializeField] private bool showPatchCenterDebugMarker = false; [SerializeField, Min(0.002f)] private float centerDebugMarkerScaleMeters = 0.025f; [SerializeField, Min(0f)] private float centerDebugSurfaceOffsetMeters = 0.01f; [SerializeField] private Color screenCenterDebugColor = new Color(1f, 0.2f, 0.2f, 0.95f); [SerializeField] private Color gridCenterDebugColor = new Color(1f, 0.9f, 0.2f, 0.95f); [SerializeField] private Color patchCenterDebugColor = new Color(0.2f, 1f, 1f, 0.95f);
    [Header("Headset Screen Center Debug")] [SerializeField] private bool showHeadsetScreenCenterMarker = false; [SerializeField, Min(0.05f)] private float headsetScreenCenterMarkerDistanceMeters = 1f; [SerializeField, Min(0.002f)] private float headsetScreenCenterMarkerScaleMeters = 0.025f; [SerializeField] private Color headsetScreenCenterMarkerColor = new Color(1f, 0.05f, 0.05f, 1f);
    [Header("Raw Depth Screen Center Debug")] [SerializeField] private bool showRawDepthScreenCenterMarker = true; [SerializeField, Min(0.002f)] private float rawDepthScreenCenterMarkerScaleMeters = 0.03f; [SerializeField] private Color rawDepthScreenCenterMarkerColor = new Color(1f, 0.05f, 0.05f, 1f);
    [Header("Original Grid Center Debug")] [SerializeField] private bool showOriginalGridCenterMarker = false; [SerializeField, Min(0.002f)] private float originalGridCenterMarkerScaleMeters = 0.035f; [SerializeField] private Color originalGridCenterMarkerColor = new Color(1f, 0f, 1f, 1f);
    [Header("Debug")] [SerializeField] private bool debugLog; [SerializeField] private bool logBounds; [SerializeField] private bool dumpRosterOnceOnPlay = false; [SerializeField] private bool dumpOnlyValidCellsInRoster = false;

    public int VisibleCount => _visibleCount; public int SurfaceTriangleCount => _surfaceMesh != null ? _surfaceMesh.triangles.Length / 3 : 0; public int FrameIndex => _frameIndex; public bool HasPendingReadback => _hasPendingReadback; public string LastIssue { get; private set; }
    public bool UpdateEveryFrame => updateEveryFrame;
    public ScanCoverDepthPreprocessor Preprocessor => preprocessor;
    public bool UseIndexConnectivity => useIndexConnectivity;
    public float SurfaceMaxEdgeLengthMeters => maxEdgeLengthMeters;
    public float SurfaceMinNeighborNormalDot => minNeighborNormalDot;
    public bool PreviewVisible => _previewVisible;
    public bool PreviewDisplayVisible => previewDisplayVisible;
    public string LastPlaneFamilyDiagnosticsPath => _lastPlaneFamilyDiagnosticsPath;
    public Transform SnapshotCaptureRoot => useWorldSpaceDisplayRoots ? (_displayRoot != null ? _displayRoot.transform : null) : transform;
    public Vector3 CurrentObservationOrigin
    {
        get
        {
            Transform origin = ResolveViewLockedOrigin();
            return origin != null ? origin.position : transform.position;
        }
    }

    private readonly List<Cell> _cells = new List<Cell>(4096); private readonly List<GridGroup> _groups = new List<GridGroup>(8); private readonly List<GameObject> _pool = new List<GameObject>(4096); private readonly List<Renderer[]> _rendererCache = new List<Renderer[]>(4096); private readonly List<bool> _validScratch = new List<bool>(4096); private readonly List<bool> _gridLineValidScratch = new List<bool>(4096); private readonly List<Vector3> _verts = new List<Vector3>(4096); private readonly List<Vector3> _normals = new List<Vector3>(4096); private readonly List<int> _tris = new List<int>(8192); private readonly List<int> _lineIndices = new List<int>(8192);
    private MaterialPropertyBlock _propertyBlock; private Material _fallbackMaterial; private GameObject _displayRoot; private Transform _runtimeMarkerRoot; private GameObject _lineRoot; private MeshFilter _lineFilter; private MeshRenderer _lineRenderer; private Mesh _lineMesh; private Material _lineMaterial; private GameObject _remeshLineRoot; private MeshFilter _remeshLineFilter; private MeshRenderer _remeshLineRenderer; private Mesh _remeshLineMesh; private Material _remeshLineMaterial; private GameObject _geometricSurfaceGridRoot; private MeshFilter _geometricSurfaceGridFilter; private MeshRenderer _geometricSurfaceGridRenderer; private Mesh _geometricSurfaceGridMesh; private Material _geometricSurfaceGridMaterial; private GameObject _candidatePlaneRoot; private readonly List<MeshFilter> _candidatePlaneFilters = new List<MeshFilter>(8); private readonly List<MeshRenderer> _candidatePlaneRenderers = new List<MeshRenderer>(8); private readonly List<Mesh> _candidatePlaneMeshes = new List<Mesh>(8); private Material _candidatePlaneMaterial; private GameObject _surfaceRoot; private MeshFilter _surfaceFilter; private MeshRenderer _surfaceRenderer; private Mesh _surfaceMesh; private Material _surfaceMaterial; private readonly List<Material> _surfaceRegionMaterials = new List<Material>(16); private GameObject _surfaceNormalRoot; private MeshFilter _surfaceNormalFilter; private MeshRenderer _surfaceNormalRenderer; private Mesh _surfaceNormalMesh; private Material _surfaceNormalMaterial; private GameObject _probeRowRoot; private Material _probeRowPointMaterial; private Material _probeRowStableLineMaterial; private Material _probeRowRefineLineMaterial; private Material _probeRowBreakLineMaterial; private readonly List<GameObject> _probeRowMarkers = new List<GameObject>(32); private readonly List<LineRenderer> _probeRowLines = new List<LineRenderer>(32); private readonly List<int> _probeTriangleIndices = new List<int>(8192); private GameObject _runtimeAxisRoot; private readonly List<Material> _runtimeAxisMaterials = new List<Material>(4); private GameObject _centerDebugRoot; private readonly GameObject[] _centerDebugMarkers = new GameObject[3]; private readonly Material[] _centerDebugMaterials = new Material[3]; private GameObject _headsetScreenCenterMarker; private Material _headsetScreenCenterMarkerMaterial; private GameObject _rawDepthScreenCenterMarker; private Material _rawDepthScreenCenterMarkerMaterial; private GameObject _originalGridCenterMarker; private Material _originalGridCenterMarkerMaterial; private FocusedGridOverlayState _focusedGridOverlayState;
    private Vector3[] _rawSurfaceExportVertices = Array.Empty<Vector3>();
    private Vector3[] _rawSurfaceExportNormals = Array.Empty<Vector3>();
    private int[] _rawSurfaceExportTriangles = Array.Empty<int>();
    private AsyncGPUReadbackRequest _worldPositionRawRequest, _worldPositionRequest, _worldNormalRequest, _worldNormalNeighbourRequest, _observationMetaRequest; private bool _hasPendingReadback; private Vector2Int _pendingResolution; private int _visibleCount; private int _frameIndex; private bool _hasDumpedRoster;
    private int _pendingSourceEyeIndex = -1;
    private double _pendingDispatchRealtimeSeconds;
    private bool _pendingHasCameraPose;
    private Vector3 _pendingCameraPosition;
    private Quaternion _pendingCameraRotation = Quaternion.identity;
    private bool _pendingHasProjectionMatrix;
    private Matrix4x4 _pendingProjectionMatrix = Matrix4x4.identity;
    private bool _pendingHasWorldToCameraMatrix;
    private Matrix4x4 _pendingWorldToCameraMatrix = Matrix4x4.identity;
    private bool _pendingHasDepthReprojectionMatrix;
    private Matrix4x4 _pendingDepthReprojectionMatrix = Matrix4x4.identity;
    private bool _pendingHasEyePosition;
    private Vector3 _pendingEyePosition;
    private static readonly int EnvironmentDepthReprojectionMatricesId = Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
    private RawDepthFrameSnapshot _latestRawDepthFrameSnapshot;
    private int _rawDepthSnapshotFrameIndex;
    private bool[] _currentValid = Array.Empty<bool>(); private bool[] _currentLineValid = Array.Empty<bool>(); private Vector3[] _currentPositions = Array.Empty<Vector3>(); private Vector3[] _currentNormals = Array.Empty<Vector3>(); private float[] _currentConfidences = Array.Empty<float>(); private Vector2Int _currentResolution;
    private bool _previewVisible = true;
    private bool _captureOnlySnapshotMode;
    private bool _snapshotGridExternalControlActive;
    private bool _snapshotGridExternalSavedUpdateEveryFrame;
    private bool _snapshotGridExternalSavedCaptureOnlySnapshotMode;
    private bool _hasFrozenHeadsetScreenCenterPoint;
    private Vector3 _frozenHeadsetScreenCenterPoint;
    private Vector3 _frozenHeadsetScreenCenterNormal;
    private bool _frozenHeadsetScreenCenterNormalValid;
    private bool _hasFrozenHeightSliceFrame;
    private Vector3 _frozenHeightSliceRightWorld = Vector3.right;
    private Vector3 _frozenHeightSliceForwardWorld = Vector3.forward;
    private bool _exportSurfaceMeshObjAfterNextBuild;
    private int _planeFamilyOutlierSubmeshIndex = -1;
    private float _lastPlaneFamilyDiagnosticsTime = -999f;
    private string _lastPlaneFamilyDiagnosticsPath;

    private void Reset() { }
    private void Awake() { ResolveRefs(); EnsurePropertyBlock(); EnsureDisplayRoots(); if (updateEveryFrame && !_snapshotGridExternalControlActive) RefreshNow(); }
    private void OnEnable() { _hasDumpedRoster = false; ResolveRefs(); EnsurePropertyBlock(); EnsureDisplayRoots(); if (updateEveryFrame && !_snapshotGridExternalControlActive) RefreshNow(); }
    private void Update() { if (!_snapshotGridExternalControlActive) UpdateDisplayRootsTransform(); if (_hasPendingReadback) { UpdatePendingReadback(); return; } if (updateEveryFrame && !_snapshotGridExternalControlActive) RefreshNow(); }
    private void OnDestroy() { if (_lineMesh != null) Destroy(_lineMesh); if (_lineMaterial != null) Destroy(_lineMaterial); if (_lineRoot != null) Destroy(_lineRoot); if (_remeshLineMesh != null) Destroy(_remeshLineMesh); if (_remeshLineMaterial != null) Destroy(_remeshLineMaterial); if (_remeshLineRoot != null) Destroy(_remeshLineRoot); if (_geometricSurfaceGridMesh != null) Destroy(_geometricSurfaceGridMesh); if (_geometricSurfaceGridMaterial != null) Destroy(_geometricSurfaceGridMaterial); if (_geometricSurfaceGridRoot != null) Destroy(_geometricSurfaceGridRoot); if (_surfaceMesh != null) Destroy(_surfaceMesh); if (_surfaceMaterial != null) Destroy(_surfaceMaterial); ClearSurfaceRegionMaterials(); if (_surfaceNormalMesh != null) Destroy(_surfaceNormalMesh); if (_surfaceNormalMaterial != null) Destroy(_surfaceNormalMaterial); if (_surfaceNormalRoot != null) Destroy(_surfaceNormalRoot); if (_fallbackMaterial != null) Destroy(_fallbackMaterial); if (_surfaceRoot != null) Destroy(_surfaceRoot); if (_probeRowRoot != null) Destroy(_probeRowRoot); if (_probeRowPointMaterial != null) Destroy(_probeRowPointMaterial); if (_probeRowStableLineMaterial != null) Destroy(_probeRowStableLineMaterial); if (_probeRowRefineLineMaterial != null) Destroy(_probeRowRefineLineMaterial); if (_probeRowBreakLineMaterial != null) Destroy(_probeRowBreakLineMaterial); if (_runtimeAxisRoot != null) Destroy(_runtimeAxisRoot); if (_centerDebugRoot != null) Destroy(_centerDebugRoot); if (_headsetScreenCenterMarker != null) Destroy(_headsetScreenCenterMarker); if (_headsetScreenCenterMarkerMaterial != null) Destroy(_headsetScreenCenterMarkerMaterial); if (_rawDepthScreenCenterMarker != null) Destroy(_rawDepthScreenCenterMarker); if (_rawDepthScreenCenterMarkerMaterial != null) Destroy(_rawDepthScreenCenterMarkerMaterial); if (_originalGridCenterMarker != null) Destroy(_originalGridCenterMarker); if (_originalGridCenterMarkerMaterial != null) Destroy(_originalGridCenterMarkerMaterial); for (int i = 0; i < _runtimeAxisMaterials.Count; i++) if (_runtimeAxisMaterials[i] != null) Destroy(_runtimeAxisMaterials[i]); for (int i = 0; i < _centerDebugMaterials.Length; i++) if (_centerDebugMaterials[i] != null) Destroy(_centerDebugMaterials[i]); if (_displayRoot != null) Destroy(_displayRoot); }

    [ContextMenu("Refresh Depth Grid Point Cloud")]
    public bool RefreshNow()
    {
        return RefreshNow(forcePreprocessorRefresh: false);
    }

    public bool RefreshNow(bool forcePreprocessorRefresh)
    {
        if (_snapshotGridExternalControlActive)
            return false;

        ResolveRefs(); EnsurePropertyBlock();
        if (preprocessor == null) return SetIssue("ScanCoverDepthPreprocessor is missing.");
        if (_hasPendingReadback) return false;
        if (!TryPrepareOutputs(preprocessor, forcePreprocessorRefresh, out RenderTexture worldPosRawTex, out RenderTexture worldPosTex, out RenderTexture worldNormalTex, out RenderTexture worldNormalNeighbourTex, out RenderTexture metaTex, out Vector2Int primaryResolution))
            return false;
        BuildCells(primaryResolution);
        UpdateDisplayRootsTransform(force: true);
        if (ShouldShowMarkers()) EnsurePoolSize(_cells.Count); else HideAllMarkers();
        if (ShouldShowGridLines()) EnsureGridLineObjects(); else SetGridLinesVisible(false);
        if (ShouldMaintainSurfaceMesh()) EnsureSurfaceObjects(); else SetSurfaceVisible(false);
        _pendingResolution = primaryResolution;
        CapturePendingReadbackProvenance();
        _worldPositionRawRequest = AsyncGPUReadback.Request(worldPosRawTex);
        _worldPositionRequest = AsyncGPUReadback.Request(worldPosTex);
        _worldNormalRequest = AsyncGPUReadback.Request(worldNormalTex);
        _worldNormalNeighbourRequest = AsyncGPUReadback.Request(worldNormalNeighbourTex);
        _observationMetaRequest = AsyncGPUReadback.Request(metaTex);
        _hasPendingReadback = true; LastIssue = null; return true;
    }

    public void SetPreviewVisible(bool visible)
    {
        _previewVisible = visible;
        if (!_previewVisible)
        {
            HideAllMarkers();
            SetGridLinesVisible(false);
            SetSurfaceVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetGeometricSurfaceGridVisible(false);
            SetHeadsetScreenCenterMarkerVisible(false);
            SetRawDepthScreenCenterMarkerVisible(false);
            SetProbeRowExperimentVisible(false);
            return;
        }

        if (updateEveryFrame && !_snapshotGridExternalControlActive)
            RefreshNow();
    }

    public void SetUpdateEveryFrame(bool enabled)
    {
        if (_snapshotGridExternalControlActive)
        {
            _snapshotGridExternalSavedUpdateEveryFrame = enabled;
            return;
        }

        updateEveryFrame = enabled;
    }

    private bool ShouldShowMarkers()
    {
        return _previewVisible && previewDisplayVisible && showMarkers;
    }

    private bool ShouldShowGridLines()
    {
        if (showHeightSliceContour)
            return false;
        return _previewVisible && previewDisplayVisible && showGridLines;
    }

    private bool ShouldShowSurfaceMesh()
    {
        return _previewVisible && previewDisplayVisible && (showSurfaceMesh || ShouldShowGridInteriorMesh());
    }

    public void RequestExportSurfaceMeshObjAfterNextBuild()
    {
        _exportSurfaceMeshObjAfterNextBuild = true;
    }

    [ContextMenu("Export Plane Family Diagnostics Now")]
    public void ExportPlaneFamilyDiagnosticsNow()
    {
        if (_verts == null || _normals == null || _verts.Count < planeFamilyMinInliers || _normals.Count != _verts.Count)
        {
            Debug.LogWarning("[ScanCoverPlaneFamilyDiagnostics] No current BL surface vertices are available.", this);
            return;
        }

        if (!TryExtractPlaneFamilyModels(_verts, _normals, out List<PlaneFamilyModel> families) || families.Count <= 0)
        {
            Debug.LogWarning("[ScanCoverPlaneFamilyDiagnostics] No plane family candidates were extracted.", this);
            return;
        }

        float maxDistance = Mathf.Max(0.005f, planeFamilyClassifyDistanceMeters);
        float normalDotThreshold = Mathf.Cos(planeFamilyClassifyNormalDegrees * Mathf.Deg2Rad);
        int[] assignments = BuildPlaneFamilyAssignments(_verts, _normals, families, maxDistance, normalDotThreshold);
        WritePlaneFamilyDiagnostics(_verts, _normals, families, assignments, maxDistance, normalDotThreshold, force: true);
    }

    [ContextMenu("Apply Rule Hardening v01")]
    public void ApplyRuleHardeningProfileV01()
    {
        previewDisplayVisible = false;
        showSurfaceMesh = false;
        keepSurfaceMeshAvailableWhenHidden = true;
        colorizeSurfaceRegions = false;
        showIrregularSurfaceBucket = false;
        showGeometricSurfaceGrid = false;
        showGridLines = false;
        showGridTriangulation = false;
        showGridInteriorMesh = false;
        showHeightSliceContour = false;
        showHeightSlicePlaneFrame = false;
        showCandidatePlaneObjects = false;
        showRawDepthScreenCenterMarker = false;
        showHeadsetScreenCenterMarker = false;
        showOriginalGridCenterMarker = false;
        maxLinearDepthMeters = 5f;

        showPlaneFamilyClassification = false;
        planeFamilyDisplayAsPointQuads = false;
        planeFamilyPointSizeMeters = 0.014f;
        planeFamilyMaxSamples = 4096;
        planeFamilyRansacIterations = 96;
        planeFamilyMaxFamilies = 10;
        planeFamilyMinInliers = 96;
        planeFamilyFitDistanceMeters = 0.055f;
        planeFamilyClassifyDistanceMeters = 0.09f;
        planeFamilyClassifyNormalDegrees = 48f;
        planeFamilyMergeNormalDegrees = 16f;
        planeFamilyMergeDistanceMeters = 0.18f;
        planeFamilyUseSpatialConsistency = true;
        planeFamilySpatialSmoothingPasses = 2;
        planeFamilyMinIslandPoints = 18;
        planeFamilyNeighborVoteThreshold = 3;
        planeFamilyWeakIslandMaxPoints = 260;
        planeFamilyWeakIslandMaxRatio = 0.08f;
        planeFamilyWeakIslandBorderRatio = 0.55f;
        planeFamilyWeakIslandRelaxNormalDegrees = 68f;
        planeFamilyWeakIslandRelaxDistanceMultiplier = 1.25f;
        planeFamilySurfaceAlpha = 1f;
        planeFamilyOutlierColor = snapshotGridUniformColor;

        planeFamilyUseStructuralConsensus = true;
        planeFamilyStrongDistanceRatio = 0.45f;
        planeFamilyProjectionPaddingMeters = 0.22f;
        planeFamilyNormalScoreWeight = 0.03f;
        planeFamilyStructuralNeighborMinSame = 4;
        planeFamilyStructuralNeighborMinRatio = 0.45f;
    }

    private bool ShouldShowGridInteriorMesh()
    {
        return showGridInteriorMesh && gridInteriorDisplayMode == GridInteriorDisplayMode.Mesh;
    }

    public void SetPreviewDisplayVisible(bool visible)
    {
        if (previewDisplayVisible == visible)
            return;

        previewDisplayVisible = visible;
        if (!previewDisplayVisible)
        {
            HideAllMarkers();
            SetGridLinesVisible(false);
            SetSurfaceVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetGeometricSurfaceGridVisible(false);
            SetCenterDebugMarkersVisible(false);
            SetHeadsetScreenCenterMarkerVisible(false);
            SetRawDepthScreenCenterMarkerVisible(false);
            SetProbeRowExperimentVisible(false);
            return;
        }

        if (_previewVisible && updateEveryFrame && !_snapshotGridExternalControlActive)
            RefreshNow();
    }

    public void SetCaptureOnlySnapshotMode(bool enabled)
    {
        if (_snapshotGridExternalControlActive)
        {
            _snapshotGridExternalSavedCaptureOnlySnapshotMode = enabled;
            return;
        }

        if (_captureOnlySnapshotMode == enabled)
            return;

        _captureOnlySnapshotMode = enabled;
        if (_captureOnlySnapshotMode)
        {
            HideAllMarkers();
            SetGridLinesVisible(false);
            SetSurfaceVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetGeometricSurfaceGridVisible(false);
            SetCenterDebugMarkersVisible(false);
            SetHeadsetScreenCenterMarkerVisible(false);
            SetRawDepthScreenCenterMarkerVisible(false);
            SetProbeRowExperimentVisible(false);
            if (_lineMesh != null)
                _lineMesh.Clear();
            if (_surfaceMesh != null)
                _surfaceMesh.Clear();
            if (_surfaceNormalMesh != null)
                _surfaceNormalMesh.Clear();
            if (_geometricSurfaceGridMesh != null)
                _geometricSurfaceGridMesh.Clear();
        }
    }

    public void SetSnapshotGridExternalControlActive(bool active)
    {
        if (_snapshotGridExternalControlActive == active)
            return;

        if (active)
        {
            _snapshotGridExternalSavedUpdateEveryFrame = updateEveryFrame;
            _snapshotGridExternalSavedCaptureOnlySnapshotMode = _captureOnlySnapshotMode;
            _snapshotGridExternalControlActive = true;
            updateEveryFrame = false;
            _captureOnlySnapshotMode = false;
            previewDisplayVisible = true;
            _previewVisible = true;
            return;
        }

        _snapshotGridExternalControlActive = false;
        updateEveryFrame = _snapshotGridExternalSavedUpdateEveryFrame;
        _captureOnlySnapshotMode = _snapshotGridExternalSavedCaptureOnlySnapshotMode;
    }

    [ContextMenu("Clear Depth Grid Runtime State")]
    public void ClearRuntimeState()
    {
        ClearRuntimeState(hidePreview: false);
    }

    public void ClearRuntimeState(bool hidePreview)
    {
        _hasPendingReadback = false;
        _pendingResolution = Vector2Int.zero;
        ResetPendingReadbackProvenance();
        _visibleCount = 0;
        _currentResolution = Vector2Int.zero;
        _currentValid = Array.Empty<bool>();
        _currentPositions = Array.Empty<Vector3>();
        _currentNormals = Array.Empty<Vector3>();
        _currentConfidences = Array.Empty<float>();
        _hasFrozenHeadsetScreenCenterPoint = false;
        _frozenHeadsetScreenCenterPoint = Vector3.zero;
        _frozenHeadsetScreenCenterNormal = Vector3.forward;
        _frozenHeadsetScreenCenterNormalValid = false;
        _hasFrozenHeightSliceFrame = false;
        _frozenHeightSliceRightWorld = Vector3.right;
        _frozenHeightSliceForwardWorld = Vector3.forward;
        _cells.Clear();
        _groups.Clear();
        _validScratch.Clear();
        _verts.Clear();
        _normals.Clear();
        _tris.Clear();
        _lineIndices.Clear();
        _frameIndex++;
        _hasDumpedRoster = false;
        LastIssue = null;

        if (hidePreview)
            _previewVisible = false;

        HideAllMarkers();
        SetGridLinesVisible(false);
        SetSurfaceVisible(false);
        SetCandidatePlaneObjectsVisible(false);
        SetSurfaceNormalIndicatorsVisible(false);
        SetGeometricSurfaceGridVisible(false);
        SetCenterDebugMarkersVisible(false);
        SetHeadsetScreenCenterMarkerVisible(false);
        SetRawDepthScreenCenterMarkerVisible(false);
        SetProbeRowExperimentVisible(false);
        if (_lineMesh != null)
            _lineMesh.Clear();
        if (_surfaceMesh != null)
            _surfaceMesh.Clear();
        if (_surfaceNormalMesh != null)
            _surfaceNormalMesh.Clear();
        if (_geometricSurfaceGridMesh != null)
            _geometricSurfaceGridMesh.Clear();
    }

    private void UpdatePendingReadback()
    {
        if (!_worldPositionRawRequest.done || !_worldPositionRequest.done || !_worldNormalRequest.done || !_worldNormalNeighbourRequest.done || !_observationMetaRequest.done) return;
        _hasPendingReadback = false;
        if (_snapshotGridExternalControlActive)
            return;
        if (_worldPositionRawRequest.hasError || _worldPositionRequest.hasError || _worldNormalRequest.hasError || _worldNormalNeighbourRequest.hasError || _observationMetaRequest.hasError) { SetIssue("AsyncGPUReadback failed for depth grid point cloud."); return; }
        NativeArray<Color> worldPositionsRaw = _worldPositionRawRequest.GetData<Color>();
        NativeArray<Color> worldPositions = _worldPositionRequest.GetData<Color>();
        NativeArray<Color> worldNormals = _worldNormalRequest.GetData<Color>();
        NativeArray<Color> worldNormalsNeighbour = _worldNormalNeighbourRequest.GetData<Color>();
        NativeArray<Color> observationMeta = _observationMetaRequest.GetData<Color>();

        if (ShouldRectifyRegularGridBuffers())
        {
            NativeArray<Color> rectifiedWorldPositionsRaw = default;
            NativeArray<Color> rectifiedWorldPositions = default;
            NativeArray<Color> rectifiedWorldNormals = default;
            NativeArray<Color> rectifiedWorldNormalsNeighbour = default;
            NativeArray<Color> rectifiedObservationMeta = default;
            try
            {
                int pixelCount = worldPositions.Length;
                rectifiedWorldPositionsRaw = new NativeArray<Color>(pixelCount, Allocator.Temp);
                rectifiedWorldPositions = new NativeArray<Color>(pixelCount, Allocator.Temp);
                rectifiedWorldNormals = new NativeArray<Color>(pixelCount, Allocator.Temp);
                rectifiedWorldNormalsNeighbour = new NativeArray<Color>(pixelCount, Allocator.Temp);
                rectifiedObservationMeta = new NativeArray<Color>(pixelCount, Allocator.Temp);
                RectifyRegularGridBuffers(worldPositionsRaw, worldPositions, worldNormals, worldNormalsNeighbour, observationMeta, _pendingResolution, rectifiedWorldPositionsRaw, rectifiedWorldPositions, rectifiedWorldNormals, rectifiedWorldNormalsNeighbour, rectifiedObservationMeta);
                StoreLatestRawDepthFrameSnapshot(rectifiedWorldPositionsRaw, rectifiedWorldPositions, rectifiedWorldNormals, rectifiedWorldNormalsNeighbour, rectifiedObservationMeta, _pendingResolution);
                if (!_captureOnlySnapshotMode)
                    BuildVisualization(rectifiedWorldPositions, rectifiedWorldNormals, rectifiedObservationMeta, _pendingResolution);
            }
            finally
            {
                if (rectifiedWorldPositionsRaw.IsCreated) rectifiedWorldPositionsRaw.Dispose();
                if (rectifiedWorldPositions.IsCreated) rectifiedWorldPositions.Dispose();
                if (rectifiedWorldNormals.IsCreated) rectifiedWorldNormals.Dispose();
                if (rectifiedWorldNormalsNeighbour.IsCreated) rectifiedWorldNormalsNeighbour.Dispose();
                if (rectifiedObservationMeta.IsCreated) rectifiedObservationMeta.Dispose();
            }

            return;
        }

        StoreLatestRawDepthFrameSnapshot(worldPositionsRaw, worldPositions, worldNormals, worldNormalsNeighbour, observationMeta, _pendingResolution);
        if (!_captureOnlySnapshotMode)
            BuildVisualization(worldPositions, worldNormals, observationMeta, _pendingResolution);
    }

    private void BuildCells(Vector2Int resolution)
    {
        _cells.Clear(); _groups.Clear();
        if (samplingMode == SamplingMode.ViewLockedVolume) { BuildVolumeCells(); return; }

        int width = Mathf.Max(1, resolution.x), height = Mathf.Max(1, resolution.y), rows = 0, columns = 0;
        if (samplingMode == SamplingMode.RegularGrid)
        {
            int stepX = Mathf.Max(1, stridePixelsX > 0 ? stridePixelsX : stridePixels);
            int stepY = Mathf.Max(1, stridePixelsY > 0 ? stridePixelsY : stridePixels);
            int columnLimit = regularGridMaxColumns > 0 ? regularGridMaxColumns : int.MaxValue;
            int rowLimit = regularGridMaxRows > 0 ? regularGridMaxRows : int.MaxValue;
            int flipBase = height - 1;

            List<int> sampleXs = new List<int>(width / stepX + 1);
            List<int> sampleYs = new List<int>(height / stepY + 1);
            for (int x = 0; x < width; x += stepX)
                sampleXs.Add(x);
            for (int y = 0; y < height; y += stepY)
                sampleYs.Add(y);

            int totalColumns = sampleXs.Count;
            int totalRows = sampleYs.Count;
            int desiredColumns = Mathf.Min(totalColumns, columnLimit);
            int desiredRows = Mathf.Min(totalRows, rowLimit);
            if (centerRegularGridWindow)
            {
                rows = 0;
                columns = desiredColumns;
                float centerX = (width - 1) * 0.5f;
                float centerY = (height - 1) * 0.5f;
                float firstX = centerX - (desiredColumns - 1) * stepX * 0.5f;
                float firstY = centerY - (desiredRows - 1) * stepY * 0.5f;
                for (int row = 0; row < desiredRows; row++)
                {
                    int srcY = Mathf.Clamp(Mathf.RoundToInt(firstY + row * stepY), 0, height - 1);
                    int py = depthPixelVFlip ? (flipBase - srcY) : srcY;
                    for (int col = 0; col < desiredColumns; col++)
                    {
                        int x = Mathf.Clamp(Mathf.RoundToInt(firstX + col * stepX), 0, width - 1);
                        _cells.Add(new Cell { minX = x, maxX = Mathf.Min(width, x + 1), minY = py, maxY = Mathf.Min(height, py + 1), centerX = x, centerY = py, group = 0, row = row, col = col, face = VolumeFace.Front });
                    }
                    rows++;
                }
                if (_cells.Count > 0) _groups.Add(new GridGroup { startIndex = 0, columns = columns, rows = rows, group = 0, face = VolumeFace.Front });
                return;
            }

            int startColumn = centerRegularGridWindow ? Mathf.Max(0, (totalColumns - desiredColumns) / 2) : 0;
            int startRow = centerRegularGridWindow ? Mathf.Max(0, (totalRows - desiredRows) / 2) : 0;
            int endColumn = Mathf.Min(totalColumns, startColumn + desiredColumns);
            int endRow = Mathf.Min(totalRows, startRow + desiredRows);

            for (int rowIndex = startRow; rowIndex < endRow; rowIndex++)
            {
                int srcY = sampleYs[rowIndex];
                int py = depthPixelVFlip ? (flipBase - srcY) : srcY;
                int rowCount = 0;
                for (int columnIndex = startColumn; columnIndex < endColumn; columnIndex++)
                {
                    int x = sampleXs[columnIndex];
                    _cells.Add(new Cell { minX = x, maxX = Mathf.Min(width, x + 1), minY = py, maxY = Mathf.Min(height, py + 1), centerX = x, centerY = py, group = 0, row = rows, col = rowCount, face = VolumeFace.Front });
                    rowCount++;
                }
                if (columns == 0) columns = rowCount;
                rows++;
            }
            if (_cells.Count > 0) _groups.Add(new GridGroup { startIndex = 0, columns = columns, rows = rows, group = 0, face = VolumeFace.Front });
            return;
        }

        int tile = Mathf.Max(2, adaptiveTileSizePixels), yFlipBase = height - 1;
        for (int srcY = 0; srcY < height; srcY += tile)
        {
            int srcYMax = Mathf.Min(height, srcY + tile), rowCount = 0; int minY = depthPixelVFlip ? yFlipBase - (srcYMax - 1) : srcY; int maxY = depthPixelVFlip ? yFlipBase - srcY + 1 : srcYMax;
            for (int x = 0; x < width; x += tile)
            {
                int maxX = Mathf.Min(width, x + tile);
                _cells.Add(new Cell { minX = x, maxX = maxX, minY = minY, maxY = maxY, centerX = Mathf.Clamp((x + maxX - 1) / 2, x, maxX - 1), centerY = Mathf.Clamp((minY + maxY - 1) / 2, minY, maxY - 1), group = 0, row = rows, col = rowCount, face = VolumeFace.Front });
                rowCount++;
            }
            if (columns == 0) columns = rowCount; rows++;
        }
        if (_cells.Count > 0) _groups.Add(new GridGroup { startIndex = 0, columns = columns, rows = rows, group = 0, face = VolumeFace.Front });
    }

    private void BuildRegularGridCellsAroundPixel(Vector2Int resolution, int centerX, int centerY)
    {
        _cells.Clear();
        _groups.Clear();
        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int stepX = Mathf.Max(1, stridePixelsX > 0 ? stridePixelsX : stridePixels);
        int stepY = Mathf.Max(1, stridePixelsY > 0 ? stridePixelsY : stridePixels);
        int columnLimit = regularGridMaxColumns > 0 ? regularGridMaxColumns : int.MaxValue;
        int rowLimit = regularGridMaxRows > 0 ? regularGridMaxRows : int.MaxValue;
        int desiredColumns = Mathf.Min(Mathf.CeilToInt(width / (float)stepX), columnLimit);
        int desiredRows = Mathf.Min(Mathf.CeilToInt(height / (float)stepY), rowLimit);
        int flipBase = height - 1;
        float firstX = Mathf.Clamp(centerX, 0, width - 1) - (desiredColumns - 1) * stepX * 0.5f;
        float firstY = Mathf.Clamp(centerY, 0, height - 1) - (desiredRows - 1) * stepY * 0.5f;

        for (int row = 0; row < desiredRows; row++)
        {
            int srcY = Mathf.Clamp(Mathf.RoundToInt(firstY + row * stepY), 0, height - 1);
            int py = depthPixelVFlip ? (flipBase - srcY) : srcY;
            for (int col = 0; col < desiredColumns; col++)
            {
                int x = Mathf.Clamp(Mathf.RoundToInt(firstX + col * stepX), 0, width - 1);
                _cells.Add(new Cell { minX = x, maxX = Mathf.Min(width, x + 1), minY = py, maxY = Mathf.Min(height, py + 1), centerX = x, centerY = py, group = 0, row = row, col = col, face = VolumeFace.Front });
            }
        }

        if (_cells.Count > 0)
            _groups.Add(new GridGroup { startIndex = 0, columns = desiredColumns, rows = desiredRows, group = 0, face = VolumeFace.Front });
    }

    private void BuildViewportCoverageRegularGridCells(Vector2Int resolution, Vector2Int centerPixel)
    {
        _cells.Clear();
        _groups.Clear();

        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int baseStepX = Mathf.Max(1, stridePixelsX > 0 ? stridePixelsX : stridePixels);
        int baseStepY = Mathf.Max(1, stridePixelsY > 0 ? stridePixelsY : stridePixels);
        int desiredColumns = regularGridMaxColumns > 0 ? regularGridMaxColumns : Mathf.CeilToInt(width / (float)baseStepX);
        int desiredRows = regularGridMaxRows > 0 ? regularGridMaxRows : Mathf.CeilToInt(height / (float)baseStepY);
        desiredColumns = Mathf.Max(1, desiredColumns);
        desiredRows = Mathf.Max(1, desiredRows);

        float fitStepX = desiredColumns > 1 ? (width - 1f) / (desiredColumns - 1f) : 0f;
        float fitStepY = desiredRows > 1 ? (height - 1f) / (desiredRows - 1f) : 0f;
        float sharedStep = Mathf.Max(1f, Mathf.Min(fitStepX > 0f ? fitStepX : fitStepY, fitStepY > 0f ? fitStepY : fitStepX));
        sharedStep *= Mathf.Clamp(regularGridViewportCoverageScale, 0.25f, 1.25f);

        float centerX = Mathf.Clamp(centerPixel.x, 0, width - 1);
        float centerY = Mathf.Clamp(centerPixel.y, 0, height - 1);
        float firstX = centerX - (desiredColumns - 1) * sharedStep * 0.5f;
        float firstY = centerY - (desiredRows - 1) * sharedStep * 0.5f;
        int flipBase = height - 1;

        for (int row = 0; row < desiredRows; row++)
        {
            float sampleYF = Mathf.Clamp(firstY + row * sharedStep, 0f, height - 1f);
            int sampleY = Mathf.Clamp(Mathf.RoundToInt(sampleYF), 0, height - 1);
            for (int col = 0; col < desiredColumns; col++)
            {
                float sampleXF = Mathf.Clamp(firstX + col * sharedStep, 0f, width - 1f);
                int sampleX = Mathf.Clamp(Mathf.RoundToInt(sampleXF), 0, width - 1);
                float pyF = depthPixelVFlip ? (flipBase - sampleYF) : sampleYF;
                int py = depthPixelVFlip ? (flipBase - sampleY) : sampleY;
                _cells.Add(new Cell { minX = sampleX, maxX = Mathf.Min(width, sampleX + 1), minY = py, maxY = Mathf.Min(height, py + 1), centerX = sampleX, centerY = py, hasSubpixelCenter = true, centerXF = sampleXF, centerYF = pyF, group = 0, row = row, col = col, face = VolumeFace.Front });
            }
        }

        if (_cells.Count > 0)
            _groups.Add(new GridGroup { startIndex = 0, columns = desiredColumns, rows = desiredRows, group = 0, face = VolumeFace.Front });
    }

    private void BuildVerticalDepthPlaneGridCells(Vector2Int resolution)
    {
        _cells.Clear();
        _groups.Clear();

        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int baseStepX = Mathf.Max(1, stridePixelsX > 0 ? stridePixelsX : stridePixels);
        int baseStepY = Mathf.Max(1, stridePixelsY > 0 ? stridePixelsY : stridePixels);
        int desiredColumns = regularGridMaxColumns > 0 ? regularGridMaxColumns : Mathf.CeilToInt(width / (float)baseStepX);
        int desiredRows = regularGridMaxRows > 0 ? regularGridMaxRows : Mathf.CeilToInt(height / (float)baseStepY);
        desiredColumns = Mathf.Max(1, desiredColumns);
        desiredRows = Mathf.Max(1, desiredRows);

        for (int row = 0; row < desiredRows; row++)
        {
            for (int col = 0; col < desiredColumns; col++)
                _cells.Add(new Cell { group = 0, row = row, col = col, face = VolumeFace.Front });
        }

        if (_cells.Count > 0)
            _groups.Add(new GridGroup { startIndex = 0, columns = desiredColumns, rows = desiredRows, group = 0, face = VolumeFace.Front });
    }

    private bool TryBuildFixedWorldRegularGridCells(
        Vector2Int resolution,
        Vector2Int anchorPixel,
        Vector3 anchorPoint,
        Transform origin,
        NativeArray<Color> worldPositions,
        List<bool> validScratch)
    {
        _cells.Clear();
        _groups.Clear();

        if (worldPositions.Length <= 0 || validScratch == null)
            return false;

        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int baseStepX = Mathf.Max(1, stridePixelsX > 0 ? stridePixelsX : stridePixels);
        int baseStepY = Mathf.Max(1, stridePixelsY > 0 ? stridePixelsY : stridePixels);
        int columnLimit = regularGridMaxColumns > 0 ? regularGridMaxColumns : int.MaxValue;
        int rowLimit = regularGridMaxRows > 0 ? regularGridMaxRows : int.MaxValue;
        int desiredColumns = regularGridMaxColumns > 0 ? regularGridMaxColumns : Mathf.CeilToInt(width / (float)baseStepX);
        int desiredRows = regularGridMaxRows > 0 ? regularGridMaxRows : Mathf.CeilToInt(height / (float)baseStepY);
        desiredColumns = Mathf.Max(1, Mathf.Min(desiredColumns, columnLimit));
        desiredRows = Mathf.Max(1, Mathf.Min(desiredRows, rowLimit));
        if (desiredColumns <= 0 || desiredRows <= 0)
            return false;

        Camera viewCamera = Camera.main;
        Transform viewOrigin = origin != null ? origin : (viewCamera != null ? viewCamera.transform : null);
        if (viewOrigin == null)
            return false;

        Vector3 forward = viewOrigin.forward.sqrMagnitude > 1e-8f ? viewOrigin.forward.normalized : Vector3.forward;
        float centerDistance = Vector3.Dot(anchorPoint - viewOrigin.position, forward);
        if (centerDistance <= 0.05f)
            centerDistance = Vector3.Distance(anchorPoint, viewOrigin.position);
        if (centerDistance <= 0.05f)
            return false;

        float cellSize = Mathf.Max(0.005f, regularGridWorldCellSizeMeters);
        int flipBase = height - 1;
        float verticalFovRad = viewCamera != null ? viewCamera.fieldOfView * Mathf.Deg2Rad : 90f * Mathf.Deg2Rad;
        float cellAngularSize = 2f * Mathf.Atan(cellSize * 0.5f / centerDistance);
        float projectedStep = height * cellAngularSize / Mathf.Max(0.001f, verticalFovRad);
        float windowScale = Mathf.Clamp(regularGridFixedWorldWindowScale, 0.25f, 1.5f);
        float projectedScaledStep = Mathf.Max(0.01f, projectedStep * windowScale);
        float sharedStep = ResolveFixedWorldGridStep(projectedScaledStep);
        float stepX = desiredColumns > 1 ? sharedStep : baseStepX;
        float stepY = desiredRows > 1 ? sharedStep : baseStepY;

        float firstX = Mathf.Clamp(anchorPixel.x, 0, width - 1) - (desiredColumns - 1) * stepX * 0.5f;
        float firstY = Mathf.Clamp(anchorPixel.y, 0, height - 1) - (desiredRows - 1) * stepY * 0.5f;

        for (int row = 0; row < desiredRows; row++)
        {
            float sampleYF = Mathf.Clamp(firstY + row * stepY, 0f, height - 1f);
            int sampleY = Mathf.Clamp(Mathf.RoundToInt(sampleYF), 0, height - 1);
            for (int col = 0; col < desiredColumns; col++)
            {
                float sampleXF = Mathf.Clamp(firstX + col * stepX, 0f, width - 1f);
                int sampleX = Mathf.Clamp(Mathf.RoundToInt(sampleXF), 0, width - 1);
                float pyF = depthPixelVFlip ? (flipBase - sampleYF) : sampleYF;
                int py = depthPixelVFlip ? (flipBase - sampleY) : sampleY;
                _cells.Add(new Cell { minX = sampleX, maxX = Mathf.Min(width, sampleX + 1), minY = py, maxY = Mathf.Min(height, py + 1), centerX = sampleX, centerY = py, hasSubpixelCenter = true, centerXF = sampleXF, centerYF = pyF, group = 0, row = row, col = col, face = VolumeFace.Front });
            }
        }

        if (_cells.Count > 0)
        {
            _groups.Add(new GridGroup { startIndex = 0, columns = desiredColumns, rows = desiredRows, group = 0, face = VolumeFace.Front });
            return true;
        }

        return false;
    }

    private float ResolveFixedWorldGridStep(float projectedScaledStep)
    {
        float farStep = Mathf.Max(0.01f, regularGridFarMinStepPixels);
        if (!regularGridUseSmoothDistanceScale)
            return Mathf.Max(farStep, projectedScaledStep);

        float softness = Mathf.Max(0.01f, regularGridStepSoftFloorPixels);
        float maxStep = Mathf.Max(projectedScaledStep, farStep);
        float softStep = maxStep + softness * Mathf.Log(
            Mathf.Exp((projectedScaledStep - maxStep) / softness) +
            Mathf.Exp((farStep - maxStep) / softness));
        return Mathf.Max(0.01f, softStep);
    }

    private void BuildVolumeCells()
    {
        float width = Mathf.Max(0.1f, volumeHalfExtents.x * 2f), height = Mathf.Max(0.1f, volumeHalfExtents.y * 2f), depth = Mathf.Max(0.1f, volumeHalfExtents.z * 2f);
        int groupIndex = 0;
        AddVolumeFaceGroup(VolumeFace.Front, width, height, ref groupIndex);
        if (volumeUseSideFaces) { AddVolumeFaceGroup(VolumeFace.Left, depth, height, ref groupIndex); AddVolumeFaceGroup(VolumeFace.Right, depth, height, ref groupIndex); }
        if (volumeUseTopBottomFaces) { AddVolumeFaceGroup(VolumeFace.Top, width, depth, ref groupIndex); AddVolumeFaceGroup(VolumeFace.Bottom, width, depth, ref groupIndex); }
    }

    private void AddVolumeFaceGroup(VolumeFace face, float axisUSize, float axisVSize, ref int groupIndex)
    {
        int columns = Mathf.Max(2, Mathf.CeilToInt(axisUSize / Mathf.Max(0.02f, volumeFaceSampleMeters)) + 1), rows = Mathf.Max(2, Mathf.CeilToInt(axisVSize / Mathf.Max(0.02f, volumeFaceSampleMeters)) + 1), start = _cells.Count;
        for (int row = 0; row < rows; row++) for (int col = 0; col < columns; col++) _cells.Add(new Cell { group = groupIndex, row = row, col = col, face = face });
        _groups.Add(new GridGroup { startIndex = start, columns = columns, rows = rows, group = groupIndex, face = face });
        groupIndex++;
    }

    private void BuildVisualization(NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, Vector2Int resolution)
    {
        if (samplingMode == SamplingMode.ViewLockedVolume) BuildVolumeVisualization(worldPositions, worldNormals, observationMeta, resolution);
        else BuildGridVisualization(worldPositions, worldNormals, observationMeta, resolution);
    }

    private bool TryFindHeadsetForwardDepthPixel(int width, int height, NativeArray<Color> worldPositions, List<bool> validScratch, out Vector2Int pixel)
    {
        pixel = default;
        Transform origin = ResolveViewLockedOrigin();
        if (origin == null || validScratch == null || worldPositions.Length <= 0)
            return false;

        Vector3 originPosition = origin.position;
        Vector3 forward = origin.forward;
        if (forward.sqrMagnitude <= 1e-8f)
            return false;

        forward.Normalize();
        float bestDot = -1f;
        float bestDistanceToRaySqr = float.PositiveInfinity;
        for (int y = 0; y < height; y++)
        {
            int rowStart = y * width;
            for (int x = 0; x < width; x++)
            {
                int index = rowStart + x;
                if (index < 0 || index >= validScratch.Count || index >= worldPositions.Length || !validScratch[index])
                    continue;

                Vector3 toPoint = WorldPos(worldPositions[index]) - originPosition;
                float distanceSqr = toPoint.sqrMagnitude;
                if (distanceSqr <= 1e-8f)
                    continue;

                float distance = Mathf.Sqrt(distanceSqr);
                float dot = Vector3.Dot(toPoint / distance, forward);
                if (dot <= 0f)
                    continue;

                float distanceToRaySqr = Mathf.Max(0f, distanceSqr * (1f - dot * dot));
                if (dot > bestDot + 1e-5f || (Mathf.Abs(dot - bestDot) <= 1e-5f && distanceToRaySqr < bestDistanceToRaySqr))
                {
                    bestDot = dot;
                    bestDistanceToRaySqr = distanceToRaySqr;
                    pixel = new Vector2Int(x, y);
                }
            }
        }

        return bestDot > 0f;
    }

    private void BuildGridVisualization(NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, Vector2Int resolution)
    {
        int width = Mathf.Max(1, resolution.x), height = Mathf.Max(1, resolution.y), pixelCount = width * height;
        RebuildValidScratch(_validScratch, pixelCount, worldPositions, worldNormals, observationMeta);
        RebuildGridLineValidScratch(_gridLineValidScratch, pixelCount, worldPositions, worldNormals, observationMeta);
        _hasFrozenHeadsetScreenCenterPoint = false;
        _frozenHeadsetScreenCenterNormalValid = false;
        _hasFrozenHeightSliceFrame = false;
        bool useDepthHitPlane = false;
        bool useFixedWorldWindow = false;
        bool useVerticalDepthPlane = false;
        Vector3 depthHitPlaneCenter = Vector3.zero;
        Transform depthHitPlaneOrigin = null;
        if (centerRegularGridWindowOnHeadsetForward && samplingMode == SamplingMode.RegularGrid && centerRegularGridWindow &&
            TryFindHeadsetForwardDepthPixel(width, height, worldPositions, _gridLineValidScratch, out Vector2Int headsetCenterPixel))
        {
            if (TryGetRegularSample(headsetCenterPixel.x, headsetCenterPixel.y, width, height, worldPositions, worldNormals, observationMeta, _gridLineValidScratch, out Vector3 headsetCenterPoint, out Vector3 headsetCenterNormal, out _, out bool headsetCenterNormalValid))
            {
                Transform origin = ResolveViewLockedOrigin();
                Vector3 centerPoint = headsetCenterPoint;
                if (origin != null && origin.forward.sqrMagnitude > 1e-8f)
                {
                    Vector3 forward = origin.forward.normalized;
                    centerPoint = origin.position + forward * Vector3.Dot(headsetCenterPoint - origin.position, forward);
                }
                _hasFrozenHeadsetScreenCenterPoint = true;
                _frozenHeadsetScreenCenterPoint = centerPoint;
                _frozenHeadsetScreenCenterNormal = headsetCenterNormal.sqrMagnitude > 1e-8f ? headsetCenterNormal.normalized : Vector3.forward;
                _frozenHeadsetScreenCenterNormalValid = headsetCenterNormalValid && _frozenHeadsetScreenCenterNormal.sqrMagnitude > 1e-8f && Finite(_frozenHeadsetScreenCenterNormal);
                CaptureFrozenHeightSliceFrame(origin);
                depthHitPlaneCenter = centerPoint;
                depthHitPlaneOrigin = origin;
                bool allowVerticalDepthPlane = regularGridUseVerticalDepthPlaneExperiment && !regularGridUseViewportCoverage;
                if (allowVerticalDepthPlane && origin != null)
                {
                    BuildVerticalDepthPlaneGridCells(resolution);
                    useVerticalDepthPlane = true;
                }
                else if (regularGridUseViewportCoverage)
                {
                    BuildViewportCoverageRegularGridCells(resolution, headsetCenterPixel);
                }
                else if (regularGridUseDepthHitPlaneOnly)
                {
                    BuildRegularGridCellsAroundPixel(resolution, headsetCenterPixel.x, headsetCenterPixel.y);
                    useDepthHitPlane = origin != null;
                }
                else if (regularGridUseFixedWorldSize && TryBuildFixedWorldRegularGridCells(resolution, headsetCenterPixel, headsetCenterPoint, origin, worldPositions, _validScratch))
                {
                    useFixedWorldWindow = true;
                }
                else
                {
                    BuildRegularGridCellsAroundPixel(resolution, headsetCenterPixel.x, headsetCenterPixel.y);
                }
            }
            else
            {
                BuildRegularGridCellsAroundPixel(resolution, headsetCenterPixel.x, headsetCenterPixel.y);
            }
        }
        if (_cells.Count <= 0 && samplingMode == SamplingMode.RegularGrid && centerRegularGridWindow && regularGridUseViewportCoverage)
            BuildViewportCoverageRegularGridCells(resolution, new Vector2Int(width / 2, height / 2));
        UpdateRawDepthScreenCenterMarker(width, height, worldPositions, worldNormals, observationMeta, _validScratch);

        Vector3[] positions = new Vector3[_cells.Count]; Vector3[] normals = new Vector3[_cells.Count]; float[] confidences = new float[_cells.Count]; bool[] valid = new bool[_cells.Count]; bool[] lineValid = new bool[_cells.Count];
        _visibleCount = 0; _frameIndex++;
        Vector3 boundsMin = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity), boundsMax = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

        if (useVerticalDepthPlane &&
            TryBuildVerticalDepthPlaneGridSamples(width, height, worldPositions, worldNormals, observationMeta, _gridLineValidScratch, depthHitPlaneCenter, _frozenHeadsetScreenCenterNormal, _frozenHeadsetScreenCenterNormalValid, depthHitPlaneOrigin, positions, normals, confidences, valid, lineValid, ref boundsMin, ref boundsMax))
        {
            for (int i = 0; i < _cells.Count; i++)
            {
                if (!valid[i])
                {
                    DisableMarker(i);
                    continue;
                }

                if (surfaceBiasMeters > 0f) positions[i] += normals[i] * surfaceBiasMeters;
                _visibleCount++;
                if (ShouldShowMarkers()) UpdateMarker(i, positions[i], normals[i], confidences[i]); else DisableMarker(i);
            }
        }
        else for (int i = 0; i < _cells.Count; i++)
        {
            Vector3 p;
            Vector3 n;
            float c;
            bool sampleValid = useDepthHitPlane
                ? TryGetDepthHitPlaneSample(_cells[i], depthHitPlaneCenter, depthHitPlaneOrigin, out p, out n, out c)
                : TryGetCellSample(_cells[i], width, height, worldPositions, worldNormals, observationMeta, _validScratch, out p, out n, out c);
            if (!sampleValid && useFixedWorldWindow && !useDepthHitPlane)
                sampleValid = TryGetFixedWorldWindowFillSample(_cells[i], width, height, worldPositions, worldNormals, observationMeta, _validScratch, out p, out n, out c);
            if (!sampleValid)
            {
                bool lineSampleValid = !useDepthHitPlane &&
                    TryGetCellSample(_cells[i], width, height, worldPositions, worldNormals, observationMeta, _gridLineValidScratch, out p, out n, out c);
                if (!lineSampleValid && useFixedWorldWindow && !useDepthHitPlane)
                    lineSampleValid = TryGetFixedWorldWindowFillSample(_cells[i], width, height, worldPositions, worldNormals, observationMeta, _gridLineValidScratch, out p, out n, out c);
                if (!lineSampleValid && useFixedWorldWindow && depthHitPlaneOrigin != null &&
                    TryGetDepthHitPlaneSample(_cells[i], depthHitPlaneCenter, depthHitPlaneOrigin, out p, out n, out c))
                    lineSampleValid = true;
                if (lineSampleValid)
                {
                    if (surfaceBiasMeters > 0f) p += n * surfaceBiasMeters;
                    positions[i] = p;
                    normals[i] = n;
                    confidences[i] = Mathf.Min(c, 0.25f);
                    lineValid[i] = true;
                }
                DisableMarker(i);
                continue;
            }
            if (surfaceBiasMeters > 0f) p += n * surfaceBiasMeters;
            positions[i] = p; normals[i] = n; confidences[i] = c; valid[i] = true; lineValid[i] = true; _visibleCount++;
            boundsMin = Vector3.Min(boundsMin, p); boundsMax = Vector3.Max(boundsMax, p);
            if (ShouldShowMarkers()) UpdateMarker(i, p, n, c); else DisableMarker(i);
        }

        if (!ShouldShowMarkers()) HideAllMarkers();
        bool[] gridLineSourceValid = showRectilinearFillInsideContour ? valid : lineValid;
        if (ShouldShowGridLines()) BuildGridLineMesh(gridLineSourceValid, positions, normals, confidences); else SetGridLinesVisible(false);
        if (showCandidatePlaneObjects) BuildCandidatePlaneObjectsFromGrid(valid, positions, normals, ShouldShowSurfaceMesh());
        else
        {
            SetCandidatePlaneObjectsVisible(false);
            if (ShouldMaintainSurfaceMesh())
            {
                BuildSurfaceMesh(valid, positions, normals, confidences, ShouldShowSurfaceMesh());
                ExportSurfaceMeshObjIfRequested();
            }
            else { SetSurfaceVisible(false); SetGeometricSurfaceGridVisible(false); SetCenterDebugMarkersVisible(false); }
        }
        StoreCurrentGridState(valid, lineValid, positions, normals, confidences, resolution);
        UpdateProbeRowExperiment(valid, positions, normals);
        DumpRosterIfNeeded(valid, lineValid, positions, normals, confidences, resolution);
        LastIssue = _visibleCount > 0 ? null : "Grid sampling produced no visible points.";
        if (debugLog)
        {
            Debug.Log($"[ScanCoverDepthGridPointCloud] mode={samplingMode}, visible={_visibleCount}, triangles={SurfaceTriangleCount}, cells={_cells.Count}, resolution={width}x{height}");
            if (_visibleCount > 0 && logBounds) Debug.Log($"[ScanCoverDepthGridPointCloud] boundsMin={boundsMin}, boundsMax={boundsMax}");
        }
    }

    private bool TryBuildVerticalDepthPlaneGridSamples(
        int width,
        int height,
        NativeArray<Color> worldPositions,
        NativeArray<Color> worldNormals,
        NativeArray<Color> observationMeta,
        List<bool> validScratch,
        Vector3 anchorPoint,
        Vector3 anchorNormal,
        bool anchorNormalValid,
        Transform origin,
        Vector3[] positions,
        Vector3[] normals,
        float[] confidences,
        bool[] valid,
        bool[] lineValid,
        ref Vector3 boundsMin,
        ref Vector3 boundsMax)
    {
        if (origin == null || _groups.Count <= 0 || positions == null || normals == null || confidences == null || valid == null || lineValid == null)
            return false;

        GridGroup group = _groups[0];
        if (group.columns <= 0 || group.rows <= 0 || positions.Length < _cells.Count)
            return false;

        Vector3 viewForward = origin.forward.sqrMagnitude > 1e-8f ? origin.forward.normalized : Vector3.forward;
        Vector3 forward = viewForward;
        if (regularGridUseSurfacePlaneAxesForVerticalDepthGrid && anchorNormalValid && anchorNormal.sqrMagnitude > 1e-8f && Finite(anchorNormal))
        {
            Vector3 surfaceForward = anchorNormal.normalized;
            if (Vector3.Dot(surfaceForward, viewForward) > 0f)
                surfaceForward = -surfaceForward;

            float facingDot = Mathf.Abs(Vector3.Dot(surfaceForward, viewForward));
            float worldUpDot = Mathf.Abs(Vector3.Dot(surfaceForward, Vector3.up));
            if (facingDot >= verticalDepthGridSurfaceAxisMinFacingDot &&
                worldUpDot <= verticalDepthGridSurfaceAxisMaxWorldUpDot)
            {
                forward = surfaceForward;
            }
        }
        Vector3 up = ResolveWorldUp(forward, origin.up);
        Vector3 right = Vector3.Cross(up, forward);
        if (right.sqrMagnitude <= 1e-8f)
            right = origin.right.sqrMagnitude > 1e-8f ? origin.right.normalized : Vector3.right;
        else
            right.Normalize();
        up = Vector3.Cross(forward, right);
        if (up.sqrMagnitude <= 1e-8f)
            up = Vector3.up;
        else
            up.Normalize();
        float cellSize = Mathf.Max(0.005f, regularGridWorldCellSizeMeters);
        float centerCol = (group.columns - 1) * 0.5f;
        float centerRow = (group.rows - 1) * 0.5f;
        float sampleRadius = cellSize * Mathf.Clamp(verticalDepthGridSampleRadiusMultiplier, 0.25f, 2f);
        float sampleRadiusSqr = sampleRadius * sampleRadius;
        float maxForwardOffset = Mathf.Max(0.02f, verticalDepthGridMaxForwardOffsetMeters);
        int cellCount = _cells.Count;

        Vector3[] weightedPositions = new Vector3[cellCount];
        Vector3[] weightedNormals = new Vector3[cellCount];
        float[] weightedConfidences = new float[cellCount];
        float[] weightSums = new float[cellCount];

        int sampleCount = Mathf.Min(width * height, worldPositions.Length);
        for (int index = 0; index < sampleCount; index++)
        {
            if (validScratch == null || index >= validScratch.Count || !validScratch[index] || index >= observationMeta.Length)
                continue;

            Vector3 samplePos = WorldPos(worldPositions[index]);
            if (!Finite(samplePos))
                continue;

            Vector3 delta = samplePos - anchorPoint;
            float u = Vector3.Dot(delta, right);
            float v = Vector3.Dot(delta, up);
            float depthOffset = Vector3.Dot(delta, forward);
            if (Mathf.Abs(depthOffset) > maxForwardOffset)
                continue;

            int col = Mathf.RoundToInt(u / cellSize + centerCol);
            int row = Mathf.RoundToInt(centerRow - v / cellSize);
            if ((uint)col >= (uint)group.columns || (uint)row >= (uint)group.rows)
                continue;

            float cellU = (col - centerCol) * cellSize;
            float cellV = (centerRow - row) * cellSize;
            float du = u - cellU;
            float dv = v - cellV;
            float lateralSqr = du * du + dv * dv;
            if (lateralSqr > sampleRadiusSqr)
                continue;

            int cellIndex = group.startIndex + row * group.columns + col;
            if ((uint)cellIndex >= (uint)cellCount)
                continue;

            float confidence = Mathf.Clamp01(Confidence(observationMeta[index]));
            float lateralWeight = 1f / (1f + lateralSqr / Mathf.Max(1e-6f, sampleRadiusSqr));
            float weight = Mathf.Max(0.001f, confidence) * lateralWeight;
            Vector3 gridPoint = anchorPoint + right * cellU + up * cellV + forward * depthOffset;
            weightedPositions[cellIndex] += gridPoint * weight;
            if (index < worldNormals.Length)
                weightedNormals[cellIndex] += WorldNormal(worldNormals[index], observationMeta[index].a >= 0.5f) * weight;
            weightedConfidences[cellIndex] += confidence * weight;
            weightSums[cellIndex] += weight;
        }

        int validCount = 0;
        for (int i = 0; i < cellCount; i++)
        {
            if (weightSums[i] <= 1e-6f)
                continue;

            Vector3 p = weightedPositions[i] / weightSums[i];
            Vector3 n = weightedNormals[i].sqrMagnitude > 1e-8f ? weightedNormals[i].normalized : -forward;
            positions[i] = p;
            normals[i] = n;
            confidences[i] = Mathf.Clamp01(weightedConfidences[i] / weightSums[i]);
            valid[i] = true;
            lineValid[i] = true;
            boundsMin = Vector3.Min(boundsMin, p);
            boundsMax = Vector3.Max(boundsMax, p);
            validCount++;
        }

        return validCount > 0;
    }

    private void BuildVolumeVisualization(NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, Vector2Int resolution)
    {
        int pixelCount = Mathf.Max(1, resolution.x * resolution.y); VolumeBin[] bins = new VolumeBin[_cells.Count]; Transform origin = ResolveViewLockedOrigin();
        Quaternion volumeRotation = ResolveVolumeRotation(origin);
        Vector3 volumePosition = origin.position;
        Vector3 half = new Vector3(Mathf.Max(0.05f, volumeHalfExtents.x), Mathf.Max(0.05f, volumeHalfExtents.y), Mathf.Max(0.05f, volumeHalfExtents.z));
        float minZ = volumeForwardOffsetMeters - half.z, maxZ = volumeForwardOffsetMeters + half.z, pad = Mathf.Max(0f, volumeCapturePaddingMeters);

        for (int i = 0; i < pixelCount; i++)
        {
            if (!IsSampleUsable(i, worldPositions, worldNormals, observationMeta)) continue;
            Vector3 posWorld = WorldPos(worldPositions[i]), normalWorld = WorldNormal(worldNormals[i], observationMeta[i].a >= 0.5f), local = Quaternion.Inverse(volumeRotation) * (posWorld - volumePosition);
            float confidence = Confidence(observationMeta[i]);
            if (local.x < -half.x - pad || local.x > half.x + pad || local.y < -half.y - pad || local.y > half.y + pad || local.z < minZ - pad || local.z > maxZ + pad) continue;

            Vector3 localNormal = (Quaternion.Inverse(volumeRotation) * normalWorld).normalized;
            if (!TryMapVolumeSample(local, localNormal, half, minZ, maxZ, out int cellIndex, out float planeDistance, out float faceNormalDot)) continue;
            if (faceNormalDot < volumeMinFaceNormalDot) continue;

            VolumeBin bin = bins[cellIndex];
            float replaceThreshold = Mathf.Max(0.01f, volumePlaneBlendMeters), weight = Mathf.Max(0.001f, confidence) / (1f + planeDistance * 12f);
            if (!bin.hasValue || planeDistance < bin.bestPlaneDistance - replaceThreshold)
            {
                bin.hasValue = true; bin.sampleCount = 1; bin.weightSum = weight; bin.confidenceSum = confidence; bin.bestPlaneDistance = planeDistance; bin.weightedPosition = posWorld * weight; bin.weightedNormal = normalWorld * weight; bins[cellIndex] = bin;
                continue;
            }
            if (planeDistance > bin.bestPlaneDistance + replaceThreshold) continue;

            bin.sampleCount++; bin.weightSum += weight; bin.confidenceSum += confidence; bin.bestPlaneDistance = Mathf.Min(bin.bestPlaneDistance, planeDistance); bin.weightedPosition += posWorld * weight; bin.weightedNormal += normalWorld * weight;
            bins[cellIndex] = bin;
        }

        Vector3[] positions = new Vector3[_cells.Count]; Vector3[] normals = new Vector3[_cells.Count]; float[] confidences = new float[_cells.Count]; bool[] valid = new bool[_cells.Count];
        _visibleCount = 0; _frameIndex++;
        Vector3 boundsMin = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity), boundsMax = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

        for (int i = 0; i < _cells.Count; i++)
        {
            VolumeBin bin = bins[i];
            if (!bin.hasValue || bin.weightSum <= 1e-6f) { DisableMarker(i); continue; }

            float coverage = Mathf.Clamp01(bin.sampleCount / 3f);
            if (coverage < volumeMinBinCoverage) { DisableMarker(i); continue; }

            Vector3 p = bin.weightedPosition / bin.weightSum, n = bin.weightedNormal.sqrMagnitude > 1e-6f ? bin.weightedNormal.normalized : Vector3.up;
            float c = Mathf.Clamp01((bin.confidenceSum / Mathf.Max(1, bin.sampleCount)) * Mathf.Lerp(0.6f, 1f, coverage));
            if (surfaceBiasMeters > 0f) p += n * surfaceBiasMeters;
            positions[i] = p; normals[i] = n; confidences[i] = c; valid[i] = true; _visibleCount++;
            boundsMin = Vector3.Min(boundsMin, p); boundsMax = Vector3.Max(boundsMax, p);
            if (ShouldShowMarkers()) UpdateMarker(i, p, n, c); else DisableMarker(i);
        }

        if (!ShouldShowMarkers()) HideAllMarkers();
        if (ShouldShowGridLines()) BuildGridLineMesh(valid, positions, normals, confidences); else SetGridLinesVisible(false);
        if (showCandidatePlaneObjects) BuildCandidatePlaneObjectsFromGrid(valid, positions, normals, ShouldShowSurfaceMesh());
        else
        {
            SetCandidatePlaneObjectsVisible(false);
            if (ShouldMaintainSurfaceMesh())
            {
                BuildSurfaceMesh(valid, positions, normals, confidences, ShouldShowSurfaceMesh());
                ExportSurfaceMeshObjIfRequested();
            }
            else { SetSurfaceVisible(false); SetGeometricSurfaceGridVisible(false); SetCenterDebugMarkersVisible(false); }
        }
        StoreCurrentGridState(valid, valid, positions, normals, confidences, resolution);
        UpdateProbeRowExperiment(valid, positions, normals);
        DumpRosterIfNeeded(valid, valid, positions, normals, confidences, resolution);
        LastIssue = _visibleCount > 0 ? null : "View-locked volume sampling produced no visible points.";
        if (debugLog)
        {
            Debug.Log($"[ScanCoverDepthGridPointCloud] mode={samplingMode}, visible={_visibleCount}, triangles={SurfaceTriangleCount}, cells={_cells.Count}, resolution={resolution.x}x{resolution.y}");
            if (_visibleCount > 0 && logBounds) Debug.Log($"[ScanCoverDepthGridPointCloud] boundsMin={boundsMin}, boundsMax={boundsMax}");
        }
    }

    private bool TryMapVolumeSample(Vector3 local, Vector3 localNormal, Vector3 half, float minZ, float maxZ, out int cellIndex, out float planeDistance, out float faceNormalDot)
    {
        cellIndex = -1; planeDistance = float.PositiveInfinity; faceNormalDot = -1f;
        if (local.z < minZ || local.z > maxZ) return false;

        float bestDistance = float.PositiveInfinity, bestU = 0f, bestV = 0f, bestDot = -1f; VolumeFace bestFace = VolumeFace.Front; bool hasCandidate = false;
        TryConsiderVolumeFace(VolumeFace.Front, Mathf.Abs(maxZ - local.z), Remap(local.x, -half.x, half.x), Remap(local.y, -half.y, half.y), Vector3.back, localNormal, ref bestDistance, ref bestFace, ref bestU, ref bestV, ref bestDot, ref hasCandidate);
        if (volumeUseSideFaces)
        {
            TryConsiderVolumeFace(VolumeFace.Left, Mathf.Abs(local.x + half.x), Remap(local.z, minZ, maxZ), Remap(local.y, -half.y, half.y), Vector3.right, localNormal, ref bestDistance, ref bestFace, ref bestU, ref bestV, ref bestDot, ref hasCandidate);
            TryConsiderVolumeFace(VolumeFace.Right, Mathf.Abs(half.x - local.x), Remap(local.z, minZ, maxZ), Remap(local.y, -half.y, half.y), Vector3.left, localNormal, ref bestDistance, ref bestFace, ref bestU, ref bestV, ref bestDot, ref hasCandidate);
        }
        if (volumeUseTopBottomFaces)
        {
            TryConsiderVolumeFace(VolumeFace.Top, Mathf.Abs(half.y - local.y), Remap(local.x, -half.x, half.x), Remap(local.z, minZ, maxZ), Vector3.down, localNormal, ref bestDistance, ref bestFace, ref bestU, ref bestV, ref bestDot, ref hasCandidate);
            TryConsiderVolumeFace(VolumeFace.Bottom, Mathf.Abs(local.y + half.y), Remap(local.x, -half.x, half.x), Remap(local.z, minZ, maxZ), Vector3.up, localNormal, ref bestDistance, ref bestFace, ref bestU, ref bestV, ref bestDot, ref hasCandidate);
        }

        if (!hasCandidate || !TryGetGroup(bestFace, out GridGroup group)) return false;
        int col = Mathf.Clamp(Mathf.RoundToInt(bestU * (group.columns - 1)), 0, group.columns - 1), row = Mathf.Clamp(Mathf.RoundToInt(bestV * (group.rows - 1)), 0, group.rows - 1);
        cellIndex = group.startIndex + row * group.columns + col; planeDistance = bestDistance; faceNormalDot = bestDot; return true;
    }

    private bool TryConsiderVolumeFace(VolumeFace face, float distance, float u, float v, Vector3 captureDirection, Vector3 localNormal, ref float bestDistance, ref VolumeFace bestFace, ref float bestU, ref float bestV, ref float bestDot, ref bool hasCandidate)
    {
        if (u < 0f || u > 1f || v < 0f || v > 1f) return false;
        if (!hasCandidate || distance < bestDistance)
        {
            hasCandidate = true; bestDistance = distance; bestFace = face; bestU = u; bestV = v; bestDot = localNormal.sqrMagnitude > 1e-6f ? Vector3.Dot(localNormal.normalized, captureDirection) : 1f; return true;
        }
        return false;
    }

    private bool TryGetGroup(VolumeFace face, out GridGroup group)
    {
        for (int i = 0; i < _groups.Count; i++) if (_groups[i].face == face) { group = _groups[i]; return true; }
        group = default; return false;
    }

    private bool TryGetCellSample(Cell cell, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence)
        => TryGetCellSample(cell, width, height, worldPositions, worldNormals, observationMeta, validScratch, neighborFill, out pos, out normal, out confidence);

    private bool TryGetCellSample(Cell cell, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, bool allowNeighborFill, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        if (samplingMode != SamplingMode.RegularGrid)
            return TryGetAdaptiveSample(cell, width, height, worldPositions, worldNormals, observationMeta, validScratch, out pos, out normal, out confidence);

        if (cell.hasSubpixelCenter && TryGetRegularSampleBilinear(cell.centerXF, cell.centerYF, width, height, worldPositions, worldNormals, observationMeta, validScratch, out pos, out normal, out confidence))
            return true;

        return TryGetRegularSample(cell.centerX, cell.centerY, width, height, worldPositions, worldNormals, observationMeta, validScratch, allowNeighborFill, out pos, out normal, out confidence);
    }

    private bool TryGetDepthHitPlaneSample(Cell cell, Vector3 center, Transform origin, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        pos = default;
        normal = default;
        confidence = 1f;
        if (origin == null || _groups.Count <= 0)
            return false;

        GridGroup group = _groups[Mathf.Clamp(cell.group, 0, _groups.Count - 1)];
        float cellSize = Mathf.Max(0.005f, regularGridWorldCellSizeMeters);
        float centerCol = (group.columns - 1) * 0.5f;
        float centerRow = (group.rows - 1) * 0.5f;
        Vector3 forward = origin.forward.sqrMagnitude > 1e-8f ? origin.forward.normalized : Vector3.forward;
        Vector3 up = ResolveWorldUp(forward, origin.up);
        Vector3 right = Vector3.Cross(up, forward);
        if (right.sqrMagnitude <= 1e-8f)
            right = origin.right.sqrMagnitude > 1e-8f ? origin.right.normalized : Vector3.right;
        else
            right.Normalize();
        up = Vector3.Cross(forward, right);
        if (up.sqrMagnitude <= 1e-8f)
            up = Vector3.up;
        else
            up.Normalize();

        pos = center + right * ((cell.col - centerCol) * cellSize) + up * ((centerRow - cell.row) * cellSize);
        normal = -forward;
        return true;
    }

    private bool TryGetRegularSample(int x, int y, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        return TryGetRegularSample(x, y, width, height, worldPositions, worldNormals, observationMeta, validScratch, neighborFill, out pos, out normal, out confidence, out _);
    }

    private bool TryGetRegularSample(int x, int y, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, bool allowNeighborFill, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        return TryGetRegularSample(x, y, width, height, worldPositions, worldNormals, observationMeta, validScratch, allowNeighborFill, out pos, out normal, out confidence, out _);
    }

    private bool TryGetRegularSample(int x, int y, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence, out bool normalValid)
        => TryGetRegularSample(x, y, width, height, worldPositions, worldNormals, observationMeta, validScratch, neighborFill, out pos, out normal, out confidence, out normalValid);

    private bool TryGetRegularSample(int x, int y, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, bool allowNeighborFill, out Vector3 pos, out Vector3 normal, out float confidence, out bool normalValid)
    {
        normalValid = false;
        if ((uint)x >= (uint)width || (uint)y >= (uint)height) { pos = default; normal = default; confidence = 0f; return false; }
        int index = x + y * width;
        if (index >= 0 && index < validScratch.Count && validScratch[index] && index < worldPositions.Length)
        {
            bool hasMeta = index < observationMeta.Length;
            bool hasNormal = index < worldNormals.Length && hasMeta && observationMeta[index].a >= 0.5f;
            pos = WorldPos(worldPositions[index]);
            normal = hasNormal ? WorldNormal(worldNormals[index], true) : Vector3.up;
            normalValid = hasNormal && normal.sqrMagnitude > 1e-8f && Finite(normal);
            confidence = hasMeta ? Confidence(observationMeta[index]) : 0f;
            return true;
        }
        if (allowNeighborFill) return TryGetNeighborSample(x, y, width, height, worldPositions, worldNormals, observationMeta, validScratch, out pos, out normal, out confidence);
        pos = default; normal = default; confidence = 0f; return false;
    }

    private bool TryGetRegularSampleBilinear(float x, float y, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        if (width <= 0 || height <= 0)
        {
            pos = default;
            normal = default;
            confidence = 0f;
            return false;
        }

        x = Mathf.Clamp(x, 0f, width - 1f);
        y = Mathf.Clamp(y, 0f, height - 1f);
        int x0 = Mathf.Clamp(Mathf.FloorToInt(x), 0, width - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(y), 0, height - 1);
        int x1 = Mathf.Clamp(x0 + 1, 0, width - 1);
        int y1 = Mathf.Clamp(y0 + 1, 0, height - 1);
        float tx = x - x0;
        float ty = y - y0;

        Vector3 weightedPos = Vector3.zero;
        Vector3 weightedNormal = Vector3.zero;
        float weightedConfidence = 0f;
        float weightSum = 0f;
        AddBilinearSample(x0, y0, (1f - tx) * (1f - ty), width, worldPositions, worldNormals, observationMeta, validScratch, ref weightedPos, ref weightedNormal, ref weightedConfidence, ref weightSum);
        AddBilinearSample(x1, y0, tx * (1f - ty), width, worldPositions, worldNormals, observationMeta, validScratch, ref weightedPos, ref weightedNormal, ref weightedConfidence, ref weightSum);
        AddBilinearSample(x0, y1, (1f - tx) * ty, width, worldPositions, worldNormals, observationMeta, validScratch, ref weightedPos, ref weightedNormal, ref weightedConfidence, ref weightSum);
        AddBilinearSample(x1, y1, tx * ty, width, worldPositions, worldNormals, observationMeta, validScratch, ref weightedPos, ref weightedNormal, ref weightedConfidence, ref weightSum);

        if (weightSum <= 1e-6f)
        {
            pos = default;
            normal = default;
            confidence = 0f;
            return false;
        }

        pos = weightedPos / weightSum;
        normal = weightedNormal.sqrMagnitude > 1e-8f ? weightedNormal.normalized : Vector3.up;
        confidence = Mathf.Clamp01(weightedConfidence / weightSum);
        return true;
    }

    private void AddBilinearSample(int x, int y, float weight, int width, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, ref Vector3 weightedPos, ref Vector3 weightedNormal, ref float weightedConfidence, ref float weightSum)
    {
        if (weight <= 1e-6f)
            return;

        int index = x + y * width;
        if (index < 0 || index >= validScratch.Count || index >= worldPositions.Length || index >= observationMeta.Length || !validScratch[index])
            return;

        weightedPos += WorldPos(worldPositions[index]) * weight;
        weightedNormal += WorldNormal(worldNormals[index], observationMeta[index].a >= 0.5f) * weight;
        weightedConfidence += Confidence(observationMeta[index]) * weight;
        weightSum += weight;
    }

    private bool TryGetFixedWorldWindowFillSample(Cell cell, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence)
        => TryGetNeighborSample(cell.centerX, cell.centerY, width, height, worldPositions, worldNormals, observationMeta, validScratch, Mathf.Max(1, regularGridFixedWorldDepthFillRadiusPixels), out pos, out normal, out confidence);

    private bool TryGetAdaptiveSample(Cell cell, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        int step = Mathf.Max(1, adaptiveTileSampleStride), total = 0, validCount = 0; Vector3 weightedPos = Vector3.zero, weightedNormal = Vector3.zero; float weightSum = 0f;
        for (int y = cell.minY; y < cell.maxY; y += step) for (int x = cell.minX; x < cell.maxX; x += step)
        {
            if ((uint)x >= (uint)width || (uint)y >= (uint)height) continue;
            total++; int index = x + y * width; if (index < 0 || index >= validScratch.Count || !validScratch[index]) continue;
            float weight = Mathf.Max(0.001f, Confidence(observationMeta[index])); weightedPos += WorldPos(worldPositions[index]) * weight; weightedNormal += WorldNormal(worldNormals[index], observationMeta[index].a >= 0.5f) * weight; weightSum += weight; validCount++;
        }

        if (total <= 0 || validCount <= 0 || weightSum <= 1e-6f) { pos = default; normal = default; confidence = 0f; return false; }
        float validRatio = validCount / (float)total; if (validRatio < adaptiveMinTileValidRatio) { pos = default; normal = default; confidence = 0f; return false; }

        Vector3 centroid = weightedPos / weightSum, avgNormal = weightedNormal.sqrMagnitude > 1e-6f ? weightedNormal.normalized : Vector3.up, coherentPos = Vector3.zero, coherentNormal = Vector3.zero; float coherentWeight = 0f; int coherentCount = 0;
        for (int y = cell.minY; y < cell.maxY; y += step) for (int x = cell.minX; x < cell.maxX; x += step)
        {
            if ((uint)x >= (uint)width || (uint)y >= (uint)height) continue;
            int index = x + y * width; if (index < 0 || index >= validScratch.Count || !validScratch[index]) continue;
            Vector3 samplePos = WorldPos(worldPositions[index]), sampleNormal = WorldNormal(worldNormals[index], observationMeta[index].a >= 0.5f); if (Vector3.Dot(avgNormal, sampleNormal) < adaptiveMinNormalDot) continue; if (Mathf.Abs(Vector3.Dot(samplePos - centroid, avgNormal)) > adaptiveMaxPlaneDeviationMeters) continue;
            float weight = Mathf.Max(0.001f, Confidence(observationMeta[index])); coherentPos += samplePos * weight; coherentNormal += sampleNormal * weight; coherentWeight += weight; coherentCount++;
        }

        if (coherentCount <= 0 || coherentWeight <= 1e-6f) { pos = default; normal = default; confidence = 0f; return false; }
        pos = coherentPos / coherentWeight; normal = coherentNormal.sqrMagnitude > 1e-6f ? coherentNormal.normalized : avgNormal; confidence = Mathf.Clamp01((coherentWeight / coherentCount) * validRatio); return true;
    }

    private struct SurfaceQuadRegionInfo
    {
        public int[] regionIds;
        public int totalRegionCount;
        public int[] regionColorIds;
        public int colorCount;
    }

    private void BuildSurfaceMesh(bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences, bool visibleAfterBuild)
    {
        EnsureSurfaceObjects(); if (_surfaceMesh == null || _groups.Count <= 0) { SetSurfaceVisible(false); return; }
        bool buildGridInteriorMesh = ShouldShowGridInteriorMesh();
        _verts.Clear(); _normals.Clear(); _tris.Clear(); int[] vertexIndices = new int[_cells.Count]; for (int i = 0; i < vertexIndices.Length; i++) vertexIndices[i] = -1;
        Transform surfaceTransform = ResolveDisplayLocalTransform();
        for (int i = 0; i < _cells.Count; i++)
        {
            if (!valid[i]) continue;
            Vector3 vertexPosition = positions[i];
            Vector3 vertexNormal = normals[i];
            if (surfaceTransform != null)
            {
                vertexPosition = surfaceTransform.InverseTransformPoint(vertexPosition);
                vertexNormal = surfaceTransform.InverseTransformDirection(vertexNormal);
                vertexNormal = vertexNormal.sqrMagnitude > 1e-6f ? vertexNormal.normalized : Vector3.up;
            }

            vertexIndices[i] = _verts.Count;
            _verts.Add(vertexPosition);
            _normals.Add(vertexNormal);
        }

        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g]; if (group.columns <= 1 || group.rows <= 1) continue;
            for (int row = 0; row < group.rows - 1; row++)
            {
                int rowStart = group.startIndex + row * group.columns, nextRowStart = group.startIndex + (row + 1) * group.columns;
                for (int col = 0; col < group.columns - 1; col++)
                {
                    int i00 = rowStart + col, i10 = rowStart + col + 1, i01 = nextRowStart + col, i11 = nextRowStart + col + 1;
                    int count = (valid[i00] ? 1 : 0) + (valid[i10] ? 1 : 0) + (valid[i01] ? 1 : 0) + (valid[i11] ? 1 : 0); if (count < 3) continue;
                    if (buildGridInteriorMesh)
                    {
                        if (count != 4)
                            continue;
                        TryAddTriangle(i00, i10, i11, vertexIndices, positions, normals, confidences);
                        TryAddTriangle(i00, i11, i01, vertexIndices, positions, normals, confidences);
                        continue;
                    }

                    if (count == 4)
                    {
                        bool diagA = EdgeOk(i00, i11, positions, normals, confidences), diagB = EdgeOk(i10, i01, positions, normals, confidences);
                        if (diagA && diagB) { float scoreA = Vector3.Distance(positions[i00], positions[i11]) / Mathf.Max(0.01f, confidences[i00] + confidences[i11]); float scoreB = Vector3.Distance(positions[i10], positions[i01]) / Mathf.Max(0.01f, confidences[i10] + confidences[i01]); diagA = scoreA <= scoreB; diagB = !diagA; }
                        if (diagA) { TryAddTriangle(i00, i10, i11, vertexIndices, positions, normals, confidences); TryAddTriangle(i00, i11, i01, vertexIndices, positions, normals, confidences); }
                        else if (diagB) { TryAddTriangle(i00, i10, i01, vertexIndices, positions, normals, confidences); TryAddTriangle(i10, i11, i01, vertexIndices, positions, normals, confidences); }
                        continue;
                    }
                    if (valid[i00] && valid[i10] && valid[i11]) TryAddTriangle(i00, i10, i11, vertexIndices, positions, normals, confidences);
                    if (valid[i00] && valid[i11] && valid[i01]) TryAddTriangle(i00, i11, i01, vertexIndices, positions, normals, confidences);
                    if (valid[i00] && valid[i10] && valid[i01]) TryAddTriangle(i00, i10, i01, vertexIndices, positions, normals, confidences);
                    if (valid[i10] && valid[i11] && valid[i01]) TryAddTriangle(i10, i11, i01, vertexIndices, positions, normals, confidences);
                }
            }
        }

        _surfaceMesh.Clear();
        ApplySurfaceMaterialSettings();
        if (_tris.Count <= 0)
        {
            ClearRawSurfaceExportCache();
            SetSurfaceVisible(false);
            SetGeometricSurfaceGridVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            if (showHeightSliceContour)
                BuildLargestCandidateGridLineMesh(new List<Vector3>(), new List<Vector3>(), new List<int>(), false);
            return;
        }
        _surfaceMesh.SetVertices(_verts);
        _surfaceMesh.SetNormals(_normals);
        CacheRawSurfaceMeshForExport(_verts, _normals, _tris);
        if (showHeightSliceContour)
        {
            BuildHeightSliceContourDisplay(_verts, _normals, _tris, visibleAfterBuild);
            _surfaceMesh.Clear();
            SetSurfaceVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetCenterDebugMarkersVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            return;
        }

        if (buildGridInteriorMesh)
        {
            _surfaceMesh.SetTriangles(_tris, 0, true);
            _surfaceMesh.RecalculateBounds();
            _planeFamilyOutlierSubmeshIndex = -1;
            SetSurfaceRendererMaterials(1);
            SetSurfaceVisible(visibleAfterBuild);
            BuildGeometricSurfaceGridOverlay(_verts, _normals, _tris, visibleAfterBuild);
            SetSurfaceNormalIndicatorsVisible(false);
            SetCenterDebugMarkersVisible(false);
            return;
        }

        if (showPlaneFamilyClassification &&
            planeFamilyDisplayAsPointQuads &&
            TryBuildPlaneFamilyPointCloudMesh(_verts, _normals, out List<Vector3> guidancePointVertices, out List<Vector3> guidancePointNormals, out List<List<int>> guidancePointSubMeshes, out int guidanceOutlierSubmeshIndex))
        {
            _surfaceMesh.Clear();
            _surfaceMesh.SetVertices(guidancePointVertices);
            _surfaceMesh.SetNormals(guidancePointNormals);
            _surfaceMesh.subMeshCount = guidancePointSubMeshes.Count;
            for (int i = 0; i < guidancePointSubMeshes.Count; i++)
                _surfaceMesh.SetTriangles(guidancePointSubMeshes[i], i, true);
            _surfaceMesh.RecalculateBounds();
            _planeFamilyOutlierSubmeshIndex = guidanceOutlierSubmeshIndex;
            SetSurfaceRendererMaterials(guidancePointSubMeshes.Count);
            SetSurfaceVisible(visibleAfterBuild);
            SetGeometricSurfaceGridVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetCenterDebugMarkersVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            return;
        }

        if (showPlaneFamilyClassification && TryBuildPlaneFamilyClassificationSubMeshes(_verts, _normals, _tris, out List<List<int>> planeFamilySubMeshes, out int outlierSubmeshIndex))
        {
            if (planeFamilyDisplayAsPointQuads &&
                TryBuildPlaneFamilyPointQuadMesh(_verts, _normals, planeFamilySubMeshes, out List<Vector3> pointVertices, out List<Vector3> pointNormals, out List<List<int>> pointSubMeshes))
            {
                _surfaceMesh.Clear();
                _surfaceMesh.SetVertices(pointVertices);
                _surfaceMesh.SetNormals(pointNormals);
                _surfaceMesh.subMeshCount = pointSubMeshes.Count;
                for (int i = 0; i < pointSubMeshes.Count; i++)
                    _surfaceMesh.SetTriangles(pointSubMeshes[i], i, true);
                _surfaceMesh.RecalculateBounds();
                _planeFamilyOutlierSubmeshIndex = outlierSubmeshIndex;
                SetSurfaceRendererMaterials(pointSubMeshes.Count);
                SetSurfaceVisible(visibleAfterBuild);
                SetGeometricSurfaceGridVisible(false);
                SetSurfaceNormalIndicatorsVisible(false);
                SetCenterDebugMarkersVisible(false);
                SetCandidatePlaneObjectsVisible(false);
                return;
            }

            _surfaceMesh.subMeshCount = planeFamilySubMeshes.Count;
            for (int i = 0; i < planeFamilySubMeshes.Count; i++)
                _surfaceMesh.SetTriangles(planeFamilySubMeshes[i], i, true);
            _surfaceMesh.RecalculateBounds();
            _planeFamilyOutlierSubmeshIndex = outlierSubmeshIndex;
            SetSurfaceRendererMaterials(planeFamilySubMeshes.Count);
            SetSurfaceVisible(visibleAfterBuild);
            SetGeometricSurfaceGridVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetCenterDebugMarkersVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            return;
        }

        if (showPlaneFamilyClassification && planeFamilyDisplayAsPointQuads)
        {
            _surfaceMesh.Clear();
            _planeFamilyOutlierSubmeshIndex = -1;
            SetSurfaceVisible(false);
            SetGeometricSurfaceGridVisible(false);
            SetSurfaceNormalIndicatorsVisible(false);
            SetCenterDebugMarkersVisible(false);
            SetCandidatePlaneObjectsVisible(false);
            return;
        }

        _planeFamilyOutlierSubmeshIndex = -1;
        if (!colorizeSurfaceRegions)
        {
            _surfaceMesh.SetTriangles(_tris, 0, true);
            _surfaceMesh.RecalculateBounds();
            SetSurfaceRendererMaterials(1);
            SetSurfaceVisible(visibleAfterBuild);
            BuildGeometricSurfaceGridOverlay(_verts, _normals, _tris, visibleAfterBuild);
            BuildSurfaceNormalIndicatorMesh(_verts, _tris, visibleAfterBuild);
            return;
        }

        if (isolateTopCandidateSurfaces)
        {
            List<CandidateSurfaceInfo> candidateSurfaces = BuildCandidateSurfaces(_verts, _normals, _tris);
            UpdateFocusedGridCellMask(valid, positions, _verts, _tris, candidateSurfaces);
            bool hasFocusedCandidate = TryGetFocusedCandidateSurface(candidateSurfaces, _verts, _tris, out CandidateSurfaceInfo focusedCandidate);
            if (showCandidatePlaneObjects)
            {
                bool hasPlanes = BuildCandidatePlaneObjects(candidateSurfaces, _verts, _tris, visibleAfterBuild);
                _surfaceMesh.Clear();
                SetSurfaceVisible(false);
                SetGeometricSurfaceGridVisible(false);
                SetSurfaceNormalIndicatorsVisible(false);
                BuildLargestCandidateGridLineMesh(new List<Vector3>(), new List<Vector3>(), new List<int>(), false);
                UpdateCenterDebugMarkers(hasFocusedCandidate, focusedCandidate, hasFocusedCandidate, focusedCandidate.averageCenter, focusedCandidate.averageNormal);
                if (!hasPlanes)
                    SetCandidatePlaneObjectsVisible(false);
                return;
            }
            SetCandidatePlaneObjectsVisible(false);
            if (rebuildLargestCandidateAsRegularGrid)
            {
                if (TryBuildTopCandidateRegularGridMeshes(_verts, _tris, candidateSurfaces, valid, positions, normals, confidences, out List<Vector3> remeshVertices, out List<Vector3> remeshNormals, out List<List<int>> remeshSubMeshes, out List<Vector3> remeshLineVertices, out List<Vector3> remeshLineNormals, out List<int> remeshLineIndices))
                {
                    bool showFill = !largestCandidateUseTriangularLattice
                        && largestCandidateShowFill
                        && remeshSubMeshes.Count > 0
                        && remeshVertices.Count > 0
                        && remeshNormals.Count == remeshVertices.Count;
                    List<int> remeshVisibleTriangles = new List<int>(remeshVertices.Count);
                    if (showFill)
                    {
                        _surfaceMesh.SetVertices(remeshVertices);
                        _surfaceMesh.SetNormals(remeshNormals);
                        _surfaceMesh.subMeshCount = remeshSubMeshes.Count;
                        for (int i = 0; i < remeshSubMeshes.Count; i++)
                        {
                            _surfaceMesh.SetTriangles(remeshSubMeshes[i], i, true);
                            remeshVisibleTriangles.AddRange(remeshSubMeshes[i]);
                        }
                        _surfaceMesh.RecalculateBounds();
                        SetSurfaceRendererMaterials(remeshSubMeshes.Count);
                        bool useRegionMaterials = colorizeSurfaceRegions && remeshSubMeshes.Count > 1;
                        for (int i = 0; i < remeshSubMeshes.Count; i++)
                        {
                            Color regionColor = GetSurfaceRegionColor(i);
                            bool useOpaqueFill = i >= 6;
                            Color fillColor = new Color(regionColor.r, regionColor.g, regionColor.b, useOpaqueFill ? 1f : Mathf.Clamp01(largestCandidateGridFillAlpha));
                            if (useRegionMaterials)
                            {
                                if (i < _surfaceRegionMaterials.Count && _surfaceRegionMaterials[i] != null)
                                    ApplySurfaceMaterialSettings(_surfaceRegionMaterials[i], fillColor, !useOpaqueFill);
                            }
                            else if (i == 0)
                            {
                                Material activeMaterial = _surfaceRenderer != null ? _surfaceRenderer.sharedMaterial : null;
                                if (activeMaterial != null)
                                    ApplySurfaceMaterialSettings(activeMaterial, fillColor, !useOpaqueFill);
                            }
                        }
                        SetSurfaceVisible(visibleAfterBuild);
                        BuildGeometricSurfaceGridOverlay(remeshVertices, remeshNormals, remeshVisibleTriangles, visibleAfterBuild);
                    }
                    else
                    {
                        _surfaceMesh.Clear();
                        SetSurfaceVisible(false);
                    }
                    BuildLargestCandidateGridLineMesh(remeshLineVertices, remeshLineNormals, remeshLineIndices, visibleAfterBuild);
                    if (ShouldShowGridLines())
                        BuildGridLineMesh(valid, positions, normals, confidences);
                    Vector3 patchDebugCenter = Vector3.zero;
                    Vector3 patchDebugNormal = Vector3.up;
                    bool hasPatchDebugCenter = showFill && TryComputeDisplayedPatchCenter(remeshVertices, remeshVisibleTriangles, out patchDebugCenter, out patchDebugNormal);
                    UpdateCenterDebugMarkers(hasFocusedCandidate, focusedCandidate, hasPatchDebugCenter, patchDebugCenter, patchDebugNormal);
                    if (showFill)
                        BuildSurfaceNormalIndicatorMesh(remeshVertices, remeshVisibleTriangles, visibleAfterBuild);
                    else
                        SetSurfaceNormalIndicatorsVisible(false);
                    return;
                }
            }

            List<List<int>> candidateSurfaceSubMeshes = BuildTopCandidateSurfaceSubMeshes(_tris, candidateSurfaces);
            if (ShouldShowGridLines())
                BuildGridLineMesh(valid, positions, normals, confidences);
            if (candidateSurfaceSubMeshes.Count <= 0)
            {
                _surfaceMesh.SetTriangles(_tris, 0, true);
                _surfaceMesh.RecalculateBounds();
                SetSurfaceRendererMaterials(1);
                SetSurfaceVisible(visibleAfterBuild);
                BuildGeometricSurfaceGridOverlay(_verts, _normals, _tris, visibleAfterBuild);
                Vector3 patchDebugCenter = Vector3.zero;
                Vector3 patchDebugNormal = Vector3.up;
                bool hasPatchDebugCenter = TryComputeDisplayedPatchCenter(_verts, _tris, out patchDebugCenter, out patchDebugNormal);
                UpdateCenterDebugMarkers(hasFocusedCandidate, focusedCandidate, hasPatchDebugCenter, patchDebugCenter, patchDebugNormal);
                BuildSurfaceNormalIndicatorMesh(_verts, _tris, visibleAfterBuild);
                return;
            }

            _surfaceMesh.subMeshCount = candidateSurfaceSubMeshes.Count;
            List<int> visibleTriangles = new List<int>(candidateSurfaceSubMeshes.Count * 256);
            for (int i = 0; i < candidateSurfaceSubMeshes.Count; i++)
            {
                _surfaceMesh.SetTriangles(candidateSurfaceSubMeshes[i], i, true);
                visibleTriangles.AddRange(candidateSurfaceSubMeshes[i]);
            }

            _surfaceMesh.RecalculateBounds();
            SetSurfaceRendererMaterials(candidateSurfaceSubMeshes.Count);
            SetSurfaceVisible(visibleAfterBuild);
            BuildGeometricSurfaceGridOverlay(_verts, _normals, visibleTriangles, visibleAfterBuild);
            Vector3 candidatePatchDebugCenter = Vector3.zero;
            Vector3 candidatePatchDebugNormal = Vector3.up;
            bool hasCandidatePatchDebugCenter = TryComputeDisplayedPatchCenter(_verts, visibleTriangles, out candidatePatchDebugCenter, out candidatePatchDebugNormal);
            UpdateCenterDebugMarkers(hasFocusedCandidate, focusedCandidate, hasCandidatePatchDebugCenter, candidatePatchDebugCenter, candidatePatchDebugNormal);
            BuildSurfaceNormalIndicatorMesh(_verts, visibleTriangles, visibleAfterBuild);
            return;
        }

        _focusedGridOverlayState = default;
        SetCenterDebugMarkersVisible(false);

        List<List<int>> normalBuckets = BuildMeshNormalBuckets(_verts, _normals, _tris);
        int activeBucketCount = 0;
        for (int i = 0; i < normalBuckets.Count; i++)
        {
            if (!showIrregularSurfaceBucket && i == 6)
                continue;
            if (normalBuckets[i] != null && normalBuckets[i].Count > 0)
                activeBucketCount++;
        }

        if (activeBucketCount <= 0)
        {
            _surfaceMesh.SetTriangles(_tris, 0, true);
            _surfaceMesh.RecalculateBounds();
            SetSurfaceRendererMaterials(1);
            SetSurfaceVisible(visibleAfterBuild);
            BuildGeometricSurfaceGridOverlay(_verts, _normals, _tris, visibleAfterBuild);
            BuildSurfaceNormalIndicatorMesh(_verts, _tris, visibleAfterBuild);
            return;
        }

        _surfaceMesh.subMeshCount = activeBucketCount;
        int subMesh = 0;
        for (int i = 0; i < normalBuckets.Count; i++)
        {
            if (!showIrregularSurfaceBucket && i == 6)
                continue;
            if (normalBuckets[i] == null || normalBuckets[i].Count <= 0)
                continue;
            _surfaceMesh.SetTriangles(normalBuckets[i], subMesh, true);
            subMesh++;
        }
        _surfaceMesh.RecalculateBounds();
        SetSurfaceRendererMaterials(activeBucketCount);
        SetSurfaceVisible(visibleAfterBuild);
        BuildGeometricSurfaceGridOverlay(_verts, _normals, _tris, visibleAfterBuild);
        BuildSurfaceNormalIndicatorMesh(_verts, _tris, visibleAfterBuild);
    }

    private bool TryBuildPlaneFamilyClassificationSubMeshes(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<int> triangles,
        out List<List<int>> subMeshes,
        out int outlierSubmeshIndex)
    {
        subMeshes = new List<List<int>>();
        outlierSubmeshIndex = -1;
        if (vertices == null || normals == null || triangles == null ||
            vertices.Count < planeFamilyMinInliers || normals.Count != vertices.Count || triangles.Count < 3)
            return false;

        if (!TryExtractPlaneFamilyModels(vertices, normals, out List<PlaneFamilyModel> families) || families.Count <= 0)
            return false;

        for (int i = 0; i < families.Count; i++)
            subMeshes.Add(new List<int>(Mathf.Max(64, triangles.Count / Mathf.Max(1, families.Count))));
        List<int> outliers = new List<int>(Mathf.Max(64, triangles.Count / 8));

        float maxDistance = Mathf.Max(0.005f, planeFamilyClassifyDistanceMeters);
        float normalDotThreshold = Mathf.Cos(planeFamilyClassifyNormalDegrees * Mathf.Deg2Rad);
        int[] assignments = BuildPlaneFamilyAssignments(vertices, normals, families, maxDistance, normalDotThreshold);
        WritePlaneFamilyDiagnostics(vertices, normals, families, assignments, maxDistance, normalDotThreshold, force: false);
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int ia = triangles[i];
            int ib = triangles[i + 1];
            int ic = triangles[i + 2];
            if ((uint)ia >= (uint)vertices.Count || (uint)ib >= (uint)vertices.Count || (uint)ic >= (uint)vertices.Count)
                continue;

            int bestFamily = ChooseTrianglePlaneFamily(assignments, ia, ib, ic);
            List<int> target = bestFamily >= 0 ? subMeshes[bestFamily] : outliers;
            target.Add(ia);
            target.Add(ib);
            target.Add(ic);
        }

        for (int i = subMeshes.Count - 1; i >= 0; i--)
        {
            if (subMeshes[i].Count <= 0)
                subMeshes.RemoveAt(i);
        }

        if (outliers.Count > 0)
        {
            outlierSubmeshIndex = subMeshes.Count;
            subMeshes.Add(outliers);
        }

        return subMeshes.Count > 0;
    }

    private bool TryBuildPlaneFamilyPointCloudMesh(
        List<Vector3> vertices,
        List<Vector3> normals,
        out List<Vector3> pointVertices,
        out List<Vector3> pointNormals,
        out List<List<int>> pointSubMeshes,
        out int outlierSubmeshIndex)
    {
        pointVertices = new List<Vector3>(vertices != null ? vertices.Count * 4 : 0);
        pointNormals = new List<Vector3>(vertices != null ? vertices.Count * 4 : 0);
        pointSubMeshes = new List<List<int>>();
        outlierSubmeshIndex = -1;
        if (vertices == null || normals == null ||
            vertices.Count < planeFamilyMinInliers || normals.Count != vertices.Count)
            return false;

        if (!TryExtractPlaneFamilyModels(vertices, normals, out List<PlaneFamilyModel> families) || families.Count <= 0)
            return false;

        float maxDistance = Mathf.Max(0.005f, planeFamilyClassifyDistanceMeters);
        float normalDotThreshold = Mathf.Cos(planeFamilyClassifyNormalDegrees * Mathf.Deg2Rad);
        float halfSize = Mathf.Max(0.001f, planeFamilyPointSizeMeters * 0.5f);
        int[] assignments = BuildPlaneFamilyAssignments(vertices, normals, families, maxDistance, normalDotThreshold);
        WritePlaneFamilyDiagnostics(vertices, normals, families, assignments, maxDistance, normalDotThreshold, force: false);

        for (int i = 0; i < families.Count; i++)
            pointSubMeshes.Add(new List<int>(Mathf.Max(64, vertices.Count / Mathf.Max(1, families.Count))));

        List<int> outliers = new List<int>(Mathf.Max(64, vertices.Count / 8));
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 point = vertices[i];
            Vector3 normal = normals[i];
            if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude <= 1e-8f)
                continue;
            normal.Normalize();
            int bestFamily = assignments[i];
            List<int> target = bestFamily >= 0 ? pointSubMeshes[bestFamily] : outliers;
            AppendPlaneFamilyPointQuad(point, normal, halfSize, pointVertices, pointNormals, target);
        }

        for (int i = pointSubMeshes.Count - 1; i >= 0; i--)
        {
            if (pointSubMeshes[i].Count <= 0)
                pointSubMeshes.RemoveAt(i);
        }

        if (outliers.Count > 0)
        {
            outlierSubmeshIndex = pointSubMeshes.Count;
            pointSubMeshes.Add(outliers);
        }

        return pointVertices.Count > 0 && pointSubMeshes.Count > 0;
    }

    private int[] BuildPlaneFamilyAssignments(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<PlaneFamilyModel> families,
        float maxDistance,
        float normalDotThreshold)
    {
        int[] assignments = new int[vertices != null ? vertices.Count : 0];
        for (int i = 0; i < assignments.Length; i++)
            assignments[i] = -1;

        if (vertices == null || normals == null || families == null ||
            vertices.Count <= 0 || normals.Count != vertices.Count || families.Count <= 0)
            return assignments;

        if (!planeFamilyUseStructuralConsensus)
        {
            for (int i = 0; i < vertices.Count; i++)
            {
                Vector3 point = vertices[i];
                Vector3 normal = normals[i];
                if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude <= 1e-8f)
                    continue;

                normal.Normalize();
                assignments[i] = ClassifyPointToPlaneFamily(point, normal, families, maxDistance, normalDotThreshold);
            }

            if (planeFamilyUseSpatialConsistency && _tris.Count >= 3)
                SmoothPlaneFamilyAssignments(vertices, normals, families, assignments, maxDistance, normalDotThreshold);
            return assignments;
        }

        PlaneFamilyProjection[] projections = BuildPlaneFamilyProjections(vertices, normals, families, maxDistance, normalDotThreshold);
        int[] candidates = new int[vertices.Count];
        bool[] strong = new bool[vertices.Count];
        for (int i = 0; i < candidates.Length; i++)
        {
            candidates[i] = -1;
            Vector3 point = vertices[i];
            Vector3 normal = normals[i];
            if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude <= 1e-8f)
                continue;

            normal.Normalize();
            candidates[i] = ClassifyPointToPlaneFamilyStructural(point, normal, families, projections, maxDistance, out float bestDistance);
            if (candidates[i] >= 0 && bestDistance <= maxDistance * Mathf.Clamp01(planeFamilyStrongDistanceRatio))
            {
                assignments[i] = candidates[i];
                strong[i] = true;
            }
        }

        ApplyStructuralConsensusAssignments(vertices.Count, candidates, strong, assignments);

        if (planeFamilyUseSpatialConsistency && _tris.Count >= 3)
            SmoothPlaneFamilyAssignments(vertices, normals, families, assignments, maxDistance, -1f);

        return assignments;
    }

    private PlaneFamilyProjection[] BuildPlaneFamilyProjections(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<PlaneFamilyModel> families,
        float maxDistance,
        float normalDotThreshold)
    {
        PlaneFamilyProjection[] projections = new PlaneFamilyProjection[families.Count];
        int[] counts = new int[families.Count];
        float looseNormalDot = Mathf.Min(normalDotThreshold, Mathf.Cos(70f * Mathf.Deg2Rad));
        for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
        {
            PlaneFamilyModel family = families[familyIndex];
            Vector3 up = Vector3.ProjectOnPlane(Vector3.up, family.normal);
            if (up.sqrMagnitude <= 1e-8f)
                up = Vector3.ProjectOnPlane(Vector3.forward, family.normal);
            if (up.sqrMagnitude <= 1e-8f)
                up = Vector3.ProjectOnPlane(Vector3.right, family.normal);
            if (up.sqrMagnitude <= 1e-8f)
                up = Vector3.up;
            up.Normalize();

            Vector3 right = Vector3.Cross(up, family.normal);
            if (right.sqrMagnitude <= 1e-8f)
                right = Vector3.Cross(Vector3.right, family.normal);
            right = right.sqrMagnitude > 1e-8f ? right.normalized : Vector3.right;
            up = Vector3.Cross(family.normal, right).normalized;

            projections[familyIndex] = new PlaneFamilyProjection
            {
                valid = false,
                right = right,
                up = up,
                minU = float.PositiveInfinity,
                maxU = float.NegativeInfinity,
                minV = float.PositiveInfinity,
                maxV = float.NegativeInfinity
            };
        }

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 point = vertices[i];
            Vector3 normal = normals[i];
            if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude <= 1e-8f)
                continue;
            normal.Normalize();

            for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
            {
                PlaneFamilyModel family = families[familyIndex];
                float distance = Mathf.Abs(Vector3.Dot(family.normal, point) + family.d);
                if (distance > maxDistance)
                    continue;
                if (Mathf.Abs(Vector3.Dot(normal, family.normal)) < looseNormalDot)
                    continue;

                PlaneFamilyProjection projection = projections[familyIndex];
                Vector3 delta = point - family.center;
                float u = Vector3.Dot(delta, projection.right);
                float v = Vector3.Dot(delta, projection.up);
                projection.minU = Mathf.Min(projection.minU, u);
                projection.maxU = Mathf.Max(projection.maxU, u);
                projection.minV = Mathf.Min(projection.minV, v);
                projection.maxV = Mathf.Max(projection.maxV, v);
                projections[familyIndex] = projection;
                counts[familyIndex]++;
            }
        }

        int minCoverageSamples = Mathf.Max(6, planeFamilyMinInliers / 2);
        for (int familyIndex = 0; familyIndex < projections.Length; familyIndex++)
        {
            PlaneFamilyProjection projection = projections[familyIndex];
            projection.valid = counts[familyIndex] >= minCoverageSamples &&
                               projection.minU <= projection.maxU &&
                               projection.minV <= projection.maxV;
            projections[familyIndex] = projection;
        }

        return projections;
    }

    private int ClassifyPointToPlaneFamilyStructural(
        Vector3 point,
        Vector3 normal,
        List<PlaneFamilyModel> families,
        PlaneFamilyProjection[] projections,
        float maxDistance,
        out float bestDistance)
    {
        int bestFamily = -1;
        bestDistance = float.PositiveInfinity;
        float bestScore = float.PositiveInfinity;
        float normalScoreWeight = Mathf.Max(0f, planeFamilyNormalScoreWeight);
        for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
        {
            PlaneFamilyModel family = families[familyIndex];
            float distance = Mathf.Abs(Vector3.Dot(family.normal, point) + family.d);
            if (distance > maxDistance)
                continue;
            if (projections != null &&
                familyIndex < projections.Length &&
                projections[familyIndex].valid &&
                !PointWithinPlaneFamilyProjection(point, family, projections[familyIndex]))
                continue;

            float normalDot = Mathf.Abs(Vector3.Dot(normal, family.normal));
            float score = distance - normalDot * maxDistance * normalScoreWeight;
            if (score < bestScore)
            {
                bestScore = score;
                bestDistance = distance;
                bestFamily = familyIndex;
            }
        }

        return bestFamily;
    }

    private bool PointWithinPlaneFamilyProjection(Vector3 point, PlaneFamilyModel family, PlaneFamilyProjection projection)
    {
        Vector3 delta = point - family.center;
        float padding = Mathf.Max(0f, planeFamilyProjectionPaddingMeters);
        float u = Vector3.Dot(delta, projection.right);
        float v = Vector3.Dot(delta, projection.up);
        return u >= projection.minU - padding &&
               u <= projection.maxU + padding &&
               v >= projection.minV - padding &&
               v <= projection.maxV + padding;
    }

    private void ApplyStructuralConsensusAssignments(int vertexCount, int[] candidates, bool[] strong, int[] assignments)
    {
        if (vertexCount <= 0 || candidates == null || strong == null || assignments == null ||
            candidates.Length != vertexCount || strong.Length != vertexCount || assignments.Length != vertexCount)
            return;

        List<int>[] neighbors = BuildPlaneFamilyVertexNeighbors(vertexCount);
        if (neighbors == null)
            return;

        int minSame = Mathf.Max(1, planeFamilyStructuralNeighborMinSame);
        float minRatio = Mathf.Clamp01(planeFamilyStructuralNeighborMinRatio);
        for (int i = 0; i < vertexCount; i++)
        {
            if (assignments[i] >= 0 || strong[i] || candidates[i] < 0)
                continue;

            List<int> localNeighbors = neighbors[i];
            if (localNeighbors == null || localNeighbors.Count <= 0)
                continue;

            int same = 0;
            int candidateNeighborCount = 0;
            for (int n = 0; n < localNeighbors.Count; n++)
            {
                int neighbor = localNeighbors[n];
                if ((uint)neighbor >= (uint)candidates.Length || candidates[neighbor] < 0)
                    continue;
                candidateNeighborCount++;
                if (candidates[neighbor] == candidates[i])
                    same++;
            }

            float ratio = candidateNeighborCount > 0 ? same / (float)candidateNeighborCount : 0f;
            if (same >= minSame && ratio >= minRatio)
                assignments[i] = candidates[i];
        }
    }

    private static int ChooseTrianglePlaneFamily(int[] assignments, int ia, int ib, int ic)
    {
        if (assignments == null ||
            (uint)ia >= (uint)assignments.Length ||
            (uint)ib >= (uint)assignments.Length ||
            (uint)ic >= (uint)assignments.Length)
            return -1;

        int a = assignments[ia];
        int b = assignments[ib];
        int c = assignments[ic];
        if (a >= 0 && (a == b || a == c))
            return a;
        if (b >= 0 && b == c)
            return b;
        if (a >= 0)
            return a;
        if (b >= 0)
            return b;
        return c >= 0 ? c : -1;
    }

    private int ClassifyPointToPlaneFamily(
        Vector3 point,
        Vector3 normal,
        List<PlaneFamilyModel> families,
        float maxDistance,
        float normalDotThreshold)
    {
        int bestFamily = -1;
        float bestDistance = maxDistance;
        for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
        {
            PlaneFamilyModel family = families[familyIndex];
            float normalDot = Mathf.Abs(Vector3.Dot(normal, family.normal));
            if (normalDot < normalDotThreshold)
                continue;

            float distance = Mathf.Abs(Vector3.Dot(family.normal, point) + family.d);
            if (distance <= bestDistance)
            {
                bestDistance = distance;
                bestFamily = familyIndex;
            }
        }

        return bestFamily;
    }

    private void SmoothPlaneFamilyAssignments(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<PlaneFamilyModel> families,
        int[] assignments,
        float maxDistance,
        float normalDotThreshold)
    {
        if (vertices == null || normals == null || families == null || assignments == null ||
            vertices.Count <= 0 || normals.Count != vertices.Count || assignments.Length != vertices.Count)
            return;

        List<int>[] neighbors = BuildPlaneFamilyVertexNeighbors(vertices.Count);
        if (neighbors == null)
            return;

        int passCount = Mathf.Clamp(planeFamilySpatialSmoothingPasses, 0, 4);
        int[] counts = new int[Mathf.Max(1, families.Count)];
        for (int pass = 0; pass < passCount; pass++)
        {
            int[] next = new int[assignments.Length];
            Array.Copy(assignments, next, assignments.Length);
            for (int i = 0; i < assignments.Length; i++)
            {
                List<int> localNeighbors = neighbors[i];
                if (localNeighbors == null || localNeighbors.Count <= 0)
                    continue;

                Array.Clear(counts, 0, counts.Length);
                for (int n = 0; n < localNeighbors.Count; n++)
                {
                    int label = assignments[localNeighbors[n]];
                    if ((uint)label < (uint)counts.Length)
                        counts[label]++;
                }

                int bestLabel = assignments[i];
                int bestVotes = 0;
                for (int label = 0; label < counts.Length; label++)
                {
                    if (counts[label] > bestVotes)
                    {
                        bestVotes = counts[label];
                        bestLabel = label;
                    }
                }

                if (bestLabel >= 0 &&
                    bestLabel != assignments[i] &&
                    bestVotes >= Mathf.Max(1, planeFamilyNeighborVoteThreshold) &&
                    PointFitsPlaneFamily(vertices[i], normals[i], families[bestLabel], maxDistance, normalDotThreshold))
                {
                    next[i] = bestLabel;
                }
            }

            Array.Copy(next, assignments, assignments.Length);
        }

        MergeSmallPlaneFamilyIslands(vertices, normals, families, assignments, neighbors, maxDistance, normalDotThreshold);
    }

    private List<int>[] BuildPlaneFamilyVertexNeighbors(int vertexCount)
    {
        if (vertexCount <= 0 || _tris.Count < 3)
            return null;

        List<int>[] neighbors = new List<int>[vertexCount];
        for (int i = 0; i + 2 < _tris.Count; i += 3)
        {
            int a = _tris[i];
            int b = _tris[i + 1];
            int c = _tris[i + 2];
            if ((uint)a >= (uint)vertexCount || (uint)b >= (uint)vertexCount || (uint)c >= (uint)vertexCount)
                continue;
            AddPlaneFamilyNeighbor(neighbors, a, b);
            AddPlaneFamilyNeighbor(neighbors, b, a);
            AddPlaneFamilyNeighbor(neighbors, b, c);
            AddPlaneFamilyNeighbor(neighbors, c, b);
            AddPlaneFamilyNeighbor(neighbors, c, a);
            AddPlaneFamilyNeighbor(neighbors, a, c);
        }

        return neighbors;
    }

    private static void AddPlaneFamilyNeighbor(List<int>[] neighbors, int index, int neighbor)
    {
        List<int> list = neighbors[index];
        if (list == null)
        {
            list = new List<int>(6);
            neighbors[index] = list;
        }

        for (int i = 0; i < list.Count; i++)
        {
            if (list[i] == neighbor)
                return;
        }

        list.Add(neighbor);
    }

    private void WritePlaneFamilyDiagnostics(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<PlaneFamilyModel> families,
        int[] assignments,
        float maxDistance,
        float normalDotThreshold,
        bool force)
    {
        if (!planeFamilyDiagnostics || vertices == null || normals == null || families == null || assignments == null ||
            vertices.Count <= 0 || normals.Count != vertices.Count || assignments.Length != vertices.Count)
            return;

        float now = Application.isPlaying ? Time.realtimeSinceStartup : 0f;
        if (!force && Application.isPlaying &&
            now - _lastPlaneFamilyDiagnosticsTime < Mathf.Max(0.1f, planeFamilyDiagnosticsMinIntervalSeconds))
            return;

        _lastPlaneFamilyDiagnosticsTime = now;
        List<int>[] neighbors = BuildPlaneFamilyVertexNeighbors(vertices.Count);
        int familyCount = families.Count;
        int[] counts = new int[familyCount];
        int[] componentCounts = new int[familyCount];
        int[] largestComponents = new int[familyCount];
        int[] smallComponents = new int[familyCount];
        int[] familyEdges = new int[familyCount];
        int[] familySameEdges = new int[familyCount];
        float[] distanceSums = new float[familyCount];
        float[] maxDistances = new float[familyCount];
        float[] angleSums = new float[familyCount];
        float[] maxAngles = new float[familyCount];

        int assignedCount = 0;
        int outlierCount = 0;
        for (int i = 0; i < assignments.Length; i++)
        {
            int label = assignments[i];
            if ((uint)label >= (uint)familyCount)
            {
                outlierCount++;
                continue;
            }

            assignedCount++;
            counts[label]++;
            PlaneFamilyModel family = families[label];
            float distance = Mathf.Abs(Vector3.Dot(family.normal, vertices[i]) + family.d);
            distanceSums[label] += distance;
            maxDistances[label] = Mathf.Max(maxDistances[label], distance);
            float angle = 0f;
            Vector3 normal = normals[i];
            if (Finite(normal) && normal.sqrMagnitude > 1e-8f)
            {
                normal.Normalize();
                angle = Mathf.Acos(Mathf.Clamp(Mathf.Abs(Vector3.Dot(normal, family.normal)), -1f, 1f)) * Mathf.Rad2Deg;
            }

            angleSums[label] += angle;
            maxAngles[label] = Mathf.Max(maxAngles[label], angle);
        }

        int labeledEdges = 0;
        int sameEdges = 0;
        int mixedEdges = 0;
        if (neighbors != null)
        {
            for (int i = 0; i < neighbors.Length; i++)
            {
                List<int> localNeighbors = neighbors[i];
                if (localNeighbors == null)
                    continue;

                int a = assignments[i];
                for (int n = 0; n < localNeighbors.Count; n++)
                {
                    int j = localNeighbors[n];
                    if (j <= i || (uint)j >= (uint)assignments.Length)
                        continue;

                    int b = assignments[j];
                    if ((uint)a >= (uint)familyCount || (uint)b >= (uint)familyCount)
                        continue;

                    labeledEdges++;
                    if (a == b)
                    {
                        sameEdges++;
                        familyEdges[a]++;
                        familySameEdges[a]++;
                    }
                    else
                    {
                        mixedEdges++;
                        familyEdges[a]++;
                        familyEdges[b]++;
                    }
                }
            }

            for (int label = 0; label < familyCount; label++)
                CountPlaneFamilyComponents(assignments, neighbors, label, out componentCounts[label], out largestComponents[label], out smallComponents[label]);
        }

        float outlierRatio = outlierCount / (float)Mathf.Max(1, vertices.Count);
        float sameEdgeRatio = sameEdges / (float)Mathf.Max(1, labeledEdges);
        float mixedEdgeRatio = mixedEdges / (float)Mathf.Max(1, labeledEdges);
        string diagnosis = BuildPlaneFamilyDiagnosticSummary(familyCount, outlierRatio, sameEdgeRatio, mixedEdgeRatio, componentCounts, counts);
        string path = null;
        if (planeFamilyDiagnosticsExportCsv)
            path = ExportPlaneFamilyDiagnosticsCsv(vertices.Count, assignedCount, outlierCount, labeledEdges, sameEdges, mixedEdges, maxDistance, normalDotThreshold, families, counts, componentCounts, largestComponents, smallComponents, familyEdges, familySameEdges, distanceSums, maxDistances, angleSums, maxAngles, diagnosis);

        if (!string.IsNullOrEmpty(path))
            _lastPlaneFamilyDiagnosticsPath = path;

        if (planeFamilyDiagnosticsLogSummary)
        {
            string pathSuffix = string.IsNullOrEmpty(path) ? string.Empty : $" path={path}";
            Debug.Log($"[ScanCoverPlaneFamilyDiagnostics] families={familyCount} assigned={assignedCount}/{vertices.Count} outlier={outlierRatio:0.0%} sameEdge={sameEdgeRatio:0.0%} mixedEdge={mixedEdgeRatio:0.0%} diagnosis={diagnosis}{pathSuffix}", this);
        }
    }

    private static void CountPlaneFamilyComponents(
        int[] assignments,
        List<int>[] neighbors,
        int targetLabel,
        out int componentCount,
        out int largestComponent,
        out int smallComponentCount)
    {
        componentCount = 0;
        largestComponent = 0;
        smallComponentCount = 0;
        if (assignments == null || neighbors == null || assignments.Length != neighbors.Length)
            return;

        bool[] visited = new bool[assignments.Length];
        List<int> stack = new List<int>(64);
        for (int start = 0; start < assignments.Length; start++)
        {
            if (visited[start] || assignments[start] != targetLabel)
                continue;

            int size = 0;
            componentCount++;
            stack.Clear();
            stack.Add(start);
            visited[start] = true;
            while (stack.Count > 0)
            {
                int current = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                size++;
                List<int> localNeighbors = neighbors[current];
                if (localNeighbors == null)
                    continue;

                for (int i = 0; i < localNeighbors.Count; i++)
                {
                    int neighbor = localNeighbors[i];
                    if ((uint)neighbor >= (uint)assignments.Length || visited[neighbor] || assignments[neighbor] != targetLabel)
                        continue;
                    visited[neighbor] = true;
                    stack.Add(neighbor);
                }
            }

            largestComponent = Mathf.Max(largestComponent, size);
            if (size < 18)
                smallComponentCount++;
        }
    }

    private string BuildPlaneFamilyDiagnosticSummary(
        int familyCount,
        float outlierRatio,
        float sameEdgeRatio,
        float mixedEdgeRatio,
        int[] componentCounts,
        int[] counts)
    {
        int activeFamilies = 0;
        int fragmentedFamilies = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            if (counts[i] <= 0)
                continue;
            activeFamilies++;
            if (i < componentCounts.Length && componentCounts[i] >= 3)
                fragmentedFamilies++;
        }

        if (familyCount >= Mathf.Max(1, planeFamilyMaxFamilies) && mixedEdgeRatio > 0.18f)
            return "候选主面数量打满且邻接混色偏高：同一物理面很可能被拆成多个 family。";
        if (sameEdgeRatio < 0.55f || mixedEdgeRatio > 0.35f)
            return "邻域一致率低：分类标签在网格邻接层已经碎，不是单纯显示问题。";
        if (outlierRatio > 0.25f)
            return "未归类点偏多：深度/法线/投影范围或阈值造成大量风险点。";
        if (fragmentedFamilies >= Mathf.Max(1, activeFamilies / 2))
            return "连通块碎片偏多：邻域合并没有覆盖到这些小岛。";
        return "整体指标未显示严重碎片，需看每个 family 的距离/法线统计定位。";
    }

    private string ExportPlaneFamilyDiagnosticsCsv(
        int vertexCount,
        int assignedCount,
        int outlierCount,
        int labeledEdges,
        int sameEdges,
        int mixedEdges,
        float classifyDistance,
        float normalDotThreshold,
        List<PlaneFamilyModel> families,
        int[] counts,
        int[] componentCounts,
        int[] largestComponents,
        int[] smallComponents,
        int[] familyEdges,
        int[] familySameEdges,
        float[] distanceSums,
        float[] maxDistances,
        float[] angleSums,
        float[] maxAngles,
        string diagnosis)
    {
        try
        {
            string root = Path.Combine(Application.dataPath, "..", "ScanCoverExports", "PlaneFamilyDiagnostics");
            Directory.CreateDirectory(root);
            string path = Path.Combine(root, $"ScanCover_PlaneFamilyDiagnostics_{DateTime.Now:yyyyMMdd_HHmmss_fff}.csv");
            StringBuilder sb = new StringBuilder(4096);
            sb.AppendLine("section,key,value");
            AppendCsvRow(sb, "summary", "vertexCount", vertexCount.ToString(CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "familyCount", families.Count.ToString(CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "assignedCount", assignedCount.ToString(CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "outlierCount", outlierCount.ToString(CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "outlierRatio", (outlierCount / (float)Mathf.Max(1, vertexCount)).ToString("0.######", CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "labeledEdges", labeledEdges.ToString(CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "sameEdgeRatio", (sameEdges / (float)Mathf.Max(1, labeledEdges)).ToString("0.######", CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "mixedEdgeRatio", (mixedEdges / (float)Mathf.Max(1, labeledEdges)).ToString("0.######", CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "classifyDistanceMeters", classifyDistance.ToString("0.######", CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "classifyNormalDegrees", (Mathf.Acos(Mathf.Clamp(normalDotThreshold, -1f, 1f)) * Mathf.Rad2Deg).ToString("0.###", CultureInfo.InvariantCulture));
            AppendCsvRow(sb, "summary", "diagnosis", diagnosis);
            sb.AppendLine();
            sb.AppendLine("family,count,inlierCount,components,largestComponent,smallComponents,sameNeighborRatio,avgPlaneDistance,maxPlaneDistance,avgNormalAngle,maxNormalAngle,normalX,normalY,normalZ,centerX,centerY,centerZ,d");
            for (int i = 0; i < families.Count; i++)
            {
                int count = counts[i];
                float sameRatio = familySameEdges[i] / (float)Mathf.Max(1, familyEdges[i]);
                float avgDistance = distanceSums[i] / Mathf.Max(1, count);
                float avgAngle = angleSums[i] / Mathf.Max(1, count);
                PlaneFamilyModel family = families[i];
                sb.Append(i.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(count.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.inlierCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(componentCounts[i].ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(largestComponents[i].ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(smallComponents[i].ToString(CultureInfo.InvariantCulture)).Append(',')
                    .Append(sameRatio.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(avgDistance.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(maxDistances[i].ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(avgAngle.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
                    .Append(maxAngles[i].ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.normal.x.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.normal.y.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.normal.z.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.center.x.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.center.y.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.center.z.ToString("0.######", CultureInfo.InvariantCulture)).Append(',')
                    .Append(family.d.ToString("0.######", CultureInfo.InvariantCulture)).AppendLine();
            }

            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
            return path;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ScanCoverPlaneFamilyDiagnostics] Failed to export diagnostics: {ex.Message}", this);
            return null;
        }
    }

    private static void AppendCsvRow(StringBuilder sb, string section, string key, string value)
    {
        sb.Append(section).Append(',').Append(key).Append(',').Append(EscapeCsv(value)).AppendLine();
    }

    private static string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;
        return value.IndexOfAny(new[] { ',', '"', '\r', '\n' }) >= 0
            ? "\"" + value.Replace("\"", "\"\"") + "\""
            : value;
    }

    private void MergeSmallPlaneFamilyIslands(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<PlaneFamilyModel> families,
        int[] assignments,
        List<int>[] neighbors,
        float maxDistance,
        float normalDotThreshold)
    {
        int minIslandPoints = Mathf.Max(1, planeFamilyMinIslandPoints);
        if (minIslandPoints <= 1 || assignments.Length <= 0)
            return;

        int assignedCount = 0;
        for (int i = 0; i < assignments.Length; i++)
        {
            if (assignments[i] >= 0)
                assignedCount++;
        }

        int weakLimitByRatio = planeFamilyWeakIslandMaxRatio > 0f
            ? Mathf.CeilToInt(assignedCount * planeFamilyWeakIslandMaxRatio)
            : int.MaxValue;
        int weakComponentLimit = Mathf.Max(
            minIslandPoints,
            Mathf.Min(Mathf.Max(minIslandPoints, planeFamilyWeakIslandMaxPoints), weakLimitByRatio));
        float weakBorderRatio = Mathf.Clamp01(planeFamilyWeakIslandBorderRatio);
        int weakMinVotes = Mathf.Max(planeFamilyNeighborVoteThreshold, 4);
        bool[] visited = new bool[assignments.Length];
        List<int> stack = new List<int>(64);
        List<int> component = new List<int>(64);
        int[] borderCounts = new int[Mathf.Max(1, families.Count)];
        for (int start = 0; start < assignments.Length; start++)
        {
            if (visited[start] || assignments[start] < 0)
                continue;

            int sourceLabel = assignments[start];
            stack.Clear();
            component.Clear();
            stack.Add(start);
            visited[start] = true;
            while (stack.Count > 0)
            {
                int current = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                component.Add(current);

                List<int> localNeighbors = neighbors[current];
                if (localNeighbors == null)
                    continue;

                for (int i = 0; i < localNeighbors.Count; i++)
                {
                    int neighbor = localNeighbors[i];
                    if ((uint)neighbor >= (uint)assignments.Length || visited[neighbor] || assignments[neighbor] != sourceLabel)
                        continue;
                    visited[neighbor] = true;
                    stack.Add(neighbor);
                }
            }

            bool tinyIsland = component.Count < minIslandPoints;
            bool weakIsland = component.Count <= weakComponentLimit;
            if (!tinyIsland && !weakIsland)
                continue;

            Array.Clear(borderCounts, 0, borderCounts.Length);
            int borderTotal = 0;
            for (int i = 0; i < component.Count; i++)
            {
                List<int> localNeighbors = neighbors[component[i]];
                if (localNeighbors == null)
                    continue;

                for (int n = 0; n < localNeighbors.Count; n++)
                {
                    int label = assignments[localNeighbors[n]];
                    if (label >= 0 && label != sourceLabel && label < borderCounts.Length)
                    {
                        borderCounts[label]++;
                        borderTotal++;
                    }
                }
            }

            int targetLabel = -1;
            int targetVotes = 0;
            for (int label = 0; label < borderCounts.Length; label++)
            {
                if (borderCounts[label] > targetVotes)
                {
                    targetVotes = borderCounts[label];
                    targetLabel = label;
                }
            }

            if (targetLabel < 0 || targetVotes <= 0)
                continue;

            float targetBorderRatio = borderTotal > 0 ? targetVotes / (float)borderTotal : 0f;
            if (!tinyIsland && (targetVotes < weakMinVotes || targetBorderRatio < weakBorderRatio))
                continue;

            for (int i = 0; i < component.Count; i++)
            {
                int vertexIndex = component[i];
                bool fits = tinyIsland
                    ? PointFitsPlaneFamily(vertices[vertexIndex], normals[vertexIndex], families[targetLabel], maxDistance, normalDotThreshold)
                    : PointFitsPlaneFamilyRelaxed(vertices[vertexIndex], normals[vertexIndex], families[targetLabel], maxDistance, normalDotThreshold);
                if (fits)
                    assignments[vertexIndex] = targetLabel;
            }
        }
    }

    private bool PointFitsPlaneFamilyRelaxed(
        Vector3 point,
        Vector3 normal,
        PlaneFamilyModel family,
        float maxDistance,
        float normalDotThreshold)
    {
        if (!Finite(point))
            return false;

        float distanceLimit = Mathf.Max(0.005f, maxDistance) * Mathf.Max(1f, planeFamilyWeakIslandRelaxDistanceMultiplier);
        if (Mathf.Abs(Vector3.Dot(family.normal, point) + family.d) > distanceLimit)
            return false;

        if (!Finite(normal) || normal.sqrMagnitude <= 1e-8f)
            return true;

        normal.Normalize();
        float relaxedDot = Mathf.Cos(Mathf.Clamp(planeFamilyWeakIslandRelaxNormalDegrees, 0f, 89f) * Mathf.Deg2Rad);
        if (normalDotThreshold >= 0f)
            relaxedDot = Mathf.Min(normalDotThreshold, relaxedDot);
        return Mathf.Abs(Vector3.Dot(normal, family.normal)) >= relaxedDot;
    }

    private static bool PointFitsPlaneFamily(
        Vector3 point,
        Vector3 normal,
        PlaneFamilyModel family,
        float maxDistance,
        float normalDotThreshold)
    {
        if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude <= 1e-8f)
            return false;
        normal.Normalize();
        return Mathf.Abs(Vector3.Dot(normal, family.normal)) >= normalDotThreshold &&
               Mathf.Abs(Vector3.Dot(family.normal, point) + family.d) <= maxDistance;
    }

    private bool TryBuildPlaneFamilyPointQuadMesh(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<List<int>> triangleSubMeshes,
        out List<Vector3> pointVertices,
        out List<Vector3> pointNormals,
        out List<List<int>> pointSubMeshes)
    {
        pointVertices = new List<Vector3>(vertices != null ? vertices.Count * 4 : 0);
        pointNormals = new List<Vector3>(vertices != null ? vertices.Count * 4 : 0);
        pointSubMeshes = new List<List<int>>();
        if (vertices == null || normals == null || triangleSubMeshes == null || vertices.Count <= 0 || normals.Count != vertices.Count)
            return false;

        float halfSize = Mathf.Max(0.001f, planeFamilyPointSizeMeters * 0.5f);
        bool[] emitted = new bool[vertices.Count];
        for (int subMeshIndex = 0; subMeshIndex < triangleSubMeshes.Count; subMeshIndex++)
        {
            List<int> source = triangleSubMeshes[subMeshIndex];
            List<int> target = new List<int>(source != null ? source.Count * 2 : 0);
            pointSubMeshes.Add(target);
            if (source == null)
                continue;

            Array.Clear(emitted, 0, emitted.Length);
            for (int i = 0; i < source.Count; i++)
            {
                int vertexIndex = source[i];
                if ((uint)vertexIndex >= (uint)vertices.Count || emitted[vertexIndex])
                    continue;

                emitted[vertexIndex] = true;
                AppendPlaneFamilyPointQuad(vertices[vertexIndex], normals[vertexIndex], halfSize, pointVertices, pointNormals, target);
            }
        }

        for (int i = pointSubMeshes.Count - 1; i >= 0; i--)
        {
            if (pointSubMeshes[i].Count <= 0)
                pointSubMeshes.RemoveAt(i);
        }

        return pointVertices.Count > 0 && pointSubMeshes.Count > 0;
    }

    private static void AppendPlaneFamilyPointQuad(
        Vector3 center,
        Vector3 normal,
        float halfSize,
        List<Vector3> vertices,
        List<Vector3> normals,
        List<int> triangles)
    {
        Vector3 n = normal.sqrMagnitude > 1e-8f && Finite(normal) ? normal.normalized : Vector3.forward;
        Vector3 right = Vector3.Cross(Vector3.up, n);
        if (right.sqrMagnitude <= 1e-8f)
            right = Vector3.Cross(Vector3.right, n);
        right = SafeNormalized(right, Vector3.right);
        Vector3 up = SafeNormalized(Vector3.Cross(n, right), Vector3.up);
        Vector3 bias = n * Mathf.Max(0.0005f, halfSize * 0.25f);

        int start = vertices.Count;
        vertices.Add(center - right * halfSize - up * halfSize + bias);
        vertices.Add(center + right * halfSize - up * halfSize + bias);
        vertices.Add(center + right * halfSize + up * halfSize + bias);
        vertices.Add(center - right * halfSize + up * halfSize + bias);
        normals.Add(n);
        normals.Add(n);
        normals.Add(n);
        normals.Add(n);
        triangles.Add(start);
        triangles.Add(start + 1);
        triangles.Add(start + 2);
        triangles.Add(start);
        triangles.Add(start + 2);
        triangles.Add(start + 3);
    }

    private bool TryExtractPlaneFamilyModels(List<Vector3> vertices, List<Vector3> normals, out List<PlaneFamilyModel> families)
    {
        families = new List<PlaneFamilyModel>(Mathf.Max(1, planeFamilyMaxFamilies));
        List<PlaneFamilySample> samples = BuildPlaneFamilySamples(vertices, normals);
        if (samples.Count < planeFamilyMinInliers)
            return false;

        List<PlaneFamilyModel> extracted = new List<PlaneFamilyModel>(Mathf.Max(1, planeFamilyMaxFamilies));
        int maxFamilies = Mathf.Max(1, planeFamilyMaxFamilies);
        for (int i = 0; i < maxFamilies; i++)
        {
            if (!TryExtractPlaneFamilyModel(samples, i, out PlaneFamilyModel model))
                break;
            extracted.Add(model);
        }

        if (extracted.Count <= 0)
            return false;

        families = MergePlaneFamilyModels(extracted);
        return families.Count > 0;
    }

    private List<PlaneFamilySample> BuildPlaneFamilySamples(List<Vector3> vertices, List<Vector3> normals)
    {
        int maxSamples = Mathf.Max(128, planeFamilyMaxSamples);
        int step = Mathf.Max(1, Mathf.CeilToInt(vertices.Count / (float)maxSamples));
        List<PlaneFamilySample> samples = new List<PlaneFamilySample>(Mathf.Min(vertices.Count, maxSamples));
        for (int i = 0; i < vertices.Count; i += step)
        {
            Vector3 point = vertices[i];
            Vector3 normal = normals[i];
            if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude <= 1e-8f)
                continue;

            samples.Add(new PlaneFamilySample
            {
                point = point,
                normal = normal.normalized,
                active = true
            });
        }

        return samples;
    }

    private bool TryExtractPlaneFamilyModel(List<PlaneFamilySample> samples, int planeIndex, out PlaneFamilyModel model)
    {
        model = default;
        List<int> activeIndices = new List<int>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            if (samples[i].active)
                activeIndices.Add(i);
        }

        int minInliers = Mathf.Max(12, planeFamilyMinInliers);
        if (activeIndices.Count < minInliers)
            return false;

        float fitDistance = Mathf.Max(0.005f, planeFamilyFitDistanceMeters);
        float normalDotThreshold = Mathf.Cos(planeFamilyClassifyNormalDegrees * Mathf.Deg2Rad);
        System.Random rng = new System.Random(1469598101 ^ samples.Count * 73856093 ^ (planeIndex + 17) * 19349663);
        int bestInliers = 0;
        Vector3 bestPoint = Vector3.zero;
        Vector3 bestNormal = Vector3.zero;
        for (int iteration = 0; iteration < planeFamilyRansacIterations; iteration++)
        {
            int ia = activeIndices[rng.Next(activeIndices.Count)];
            int ib = activeIndices[rng.Next(activeIndices.Count)];
            int ic = activeIndices[rng.Next(activeIndices.Count)];
            if (ia == ib || ia == ic || ib == ic)
                continue;

            Vector3 a = samples[ia].point;
            Vector3 b = samples[ib].point;
            Vector3 c = samples[ic].point;
            Vector3 candidateNormal = Vector3.Cross(b - a, c - a);
            if (!Finite(candidateNormal) || candidateNormal.sqrMagnitude <= 1e-8f)
                continue;
            candidateNormal.Normalize();

            int inliers = 0;
            for (int i = 0; i < activeIndices.Count; i++)
            {
                PlaneFamilySample sample = samples[activeIndices[i]];
                float distance = Mathf.Abs(Vector3.Dot(candidateNormal, sample.point - a));
                float normalDot = Mathf.Abs(Vector3.Dot(candidateNormal, sample.normal));
                if (distance <= fitDistance && normalDot >= normalDotThreshold)
                    inliers++;
            }

            if (inliers > bestInliers)
            {
                bestInliers = inliers;
                bestPoint = a;
                bestNormal = candidateNormal;
            }
        }

        if (bestInliers < minInliers || bestNormal.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 center = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        int refinedInliers = 0;
        for (int i = 0; i < activeIndices.Count; i++)
        {
            int sampleIndex = activeIndices[i];
            PlaneFamilySample sample = samples[sampleIndex];
            float distance = Mathf.Abs(Vector3.Dot(bestNormal, sample.point - bestPoint));
            float normalDot = Mathf.Abs(Vector3.Dot(bestNormal, sample.normal));
            if (distance > fitDistance || normalDot < normalDotThreshold)
                continue;

            center += sample.point;
            normalSum += Vector3.Dot(sample.normal, bestNormal) >= 0f ? sample.normal : -sample.normal;
            refinedInliers++;
            PlaneFamilySample updated = sample;
            updated.active = false;
            samples[sampleIndex] = updated;
        }

        if (refinedInliers < minInliers)
            return false;

        center /= refinedInliers;
        Vector3 normal = normalSum.sqrMagnitude > 1e-8f ? normalSum.normalized : bestNormal;
        float d = -Vector3.Dot(normal, center);
        model = new PlaneFamilyModel
        {
            center = center,
            normal = normal,
            d = d,
            inlierCount = refinedInliers
        };
        return true;
    }

    private List<PlaneFamilyModel> MergePlaneFamilyModels(List<PlaneFamilyModel> planes)
    {
        List<PlaneFamilyModel> families = new List<PlaneFamilyModel>(planes.Count);
        float mergeDot = Mathf.Cos(planeFamilyMergeNormalDegrees * Mathf.Deg2Rad);
        float mergeDistance = Mathf.Max(0.005f, planeFamilyMergeDistanceMeters);
        for (int i = 0; i < planes.Count; i++)
        {
            PlaneFamilyModel plane = planes[i];
            int target = -1;
            for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
            {
                PlaneFamilyModel family = families[familyIndex];
                float dot = Vector3.Dot(plane.normal, family.normal);
                float signedD = plane.d;
                Vector3 signedNormal = plane.normal;
                if (dot < 0f)
                {
                    dot = -dot;
                    signedD = -signedD;
                    signedNormal = -signedNormal;
                }

                if (dot >= mergeDot && Mathf.Abs(signedD - family.d) <= mergeDistance)
                {
                    int total = family.inlierCount + plane.inlierCount;
                    family.center = (family.center * family.inlierCount + plane.center * plane.inlierCount) / Mathf.Max(1, total);
                    Vector3 mergedNormal = family.normal * family.inlierCount + signedNormal * plane.inlierCount;
                    family.normal = mergedNormal.sqrMagnitude > 1e-8f ? mergedNormal.normalized : family.normal;
                    family.d = (family.d * family.inlierCount + signedD * plane.inlierCount) / Mathf.Max(1, total);
                    family.inlierCount = total;
                    families[familyIndex] = family;
                    target = familyIndex;
                    break;
                }
            }

            if (target < 0)
                families.Add(plane);
        }

        families.Sort((a, b) => b.inlierCount.CompareTo(a.inlierCount));
        return families;
    }

    private bool BuildCandidatePlaneObjects(List<CandidateSurfaceInfo> candidateSurfaces, List<Vector3> vertices, List<int> triangles, bool visibleAfterBuild)
    {
        if (candidateSurfaces == null || candidateSurfaces.Count <= 0 || vertices == null || triangles == null)
            return false;

        EnsureCandidatePlaneObjects(Mathf.Max(1, candidatePlaneObjectCount));
        SortCandidateSurfacesForDisplay(candidateSurfaces, vertices, triangles);

        bool disableMinTriangleFilter = topCandidateSurfaceMinTriangleCount <= 0;
        int minTriangleCount = disableMinTriangleFilter ? 0 : Mathf.Max(1, topCandidateSurfaceMinTriangleCount);
        int shownCount = 0;
        for (int i = 0; i < candidateSurfaces.Count && shownCount < candidatePlaneObjectCount; i++)
        {
            CandidateSurfaceInfo candidate = candidateSurfaces[i];
            if (candidate.faceIndices == null || (!disableMinTriangleFilter && candidate.faceIndices.Count < minTriangleCount))
                continue;

            if (!TryBuildCandidatePlaneMesh(candidate, vertices, triangles, shownCount))
                continue;

            if (shownCount < _candidatePlaneRenderers.Count && _candidatePlaneRenderers[shownCount] != null)
                _candidatePlaneRenderers[shownCount].enabled = visibleAfterBuild;
            shownCount++;
        }

        for (int i = shownCount; i < _candidatePlaneRenderers.Count; i++)
        {
            if (_candidatePlaneRenderers[i] != null)
                _candidatePlaneRenderers[i].enabled = false;
        }

        SetCandidatePlaneObjectsVisible(visibleAfterBuild && shownCount > 0);
        return shownCount > 0;
    }

    private bool TryBuildCandidatePlaneMesh(CandidateSurfaceInfo candidate, List<Vector3> vertices, List<int> triangles, int planeIndex)
    {
        if (planeIndex < 0 || planeIndex >= _candidatePlaneMeshes.Count || _candidatePlaneMeshes[planeIndex] == null)
            return false;

        Vector3 planeNormal = candidate.averageNormal.sqrMagnitude > 1e-8f ? candidate.averageNormal.normalized : Vector3.forward;
        Vector3 planeOrigin = candidate.averageCenter;
        Vector3 localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            return false;

        localUp.Normalize();
        Vector3 localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return false;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;

        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        Vector3 centerSum = Vector3.zero;
        int pointCount = 0;
        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            if (triStart < 0 || triStart + 2 >= triangles.Count)
                continue;

            for (int j = 0; j < 3; j++)
            {
                int vertexIndex = triangles[triStart + j];
                if (vertexIndex < 0 || vertexIndex >= vertices.Count)
                    continue;

                Vector3 point = vertices[vertexIndex];
                Vector3 delta = point - planeOrigin;
                float u = Vector3.Dot(delta, localRight);
                float v = Vector3.Dot(delta, localUp);
                minU = Mathf.Min(minU, u);
                maxU = Mathf.Max(maxU, u);
                minV = Mathf.Min(minV, v);
                maxV = Mathf.Max(maxV, v);
                centerSum += point;
                pointCount++;
            }
        }

        if (pointCount <= 0 || !float.IsFinite(minU) || !float.IsFinite(maxU) || !float.IsFinite(minV) || !float.IsFinite(maxV))
            return false;

        planeOrigin = centerSum / pointCount;
        float halfWidth = Mathf.Max(candidatePlaneMinSizeMeters * 0.5f, (maxU - minU) * 0.5f + candidatePlanePaddingMeters);
        float halfHeight = Mathf.Max(candidatePlaneMinSizeMeters * 0.5f, (maxV - minV) * 0.5f + candidatePlanePaddingMeters);
        Vector3 offset = planeNormal * Mathf.Max(0f, candidatePlaneSurfaceOffsetMeters);
        Vector3 p00 = planeOrigin - localRight * halfWidth - localUp * halfHeight + offset;
        Vector3 p10 = planeOrigin + localRight * halfWidth - localUp * halfHeight + offset;
        Vector3 p11 = planeOrigin + localRight * halfWidth + localUp * halfHeight + offset;
        Vector3 p01 = planeOrigin - localRight * halfWidth + localUp * halfHeight + offset;

        Mesh mesh = _candidatePlaneMeshes[planeIndex];
        mesh.Clear();
        mesh.SetVertices(new List<Vector3> { p00, p10, p11, p01 });
        mesh.SetNormals(new List<Vector3> { planeNormal, planeNormal, planeNormal, planeNormal });
        mesh.SetUVs(0, new List<Vector2> { Vector2.zero, Vector2.right, Vector2.one, Vector2.up });
        mesh.SetTriangles(new[] { 0, 1, 2, 0, 2, 3 }, 0, true);
        mesh.RecalculateBounds();
        return true;
    }

    private bool BuildCandidatePlaneObjectsFromGrid(bool[] valid, Vector3[] positions, Vector3[] normals, bool visibleAfterBuild)
    {
        SetSurfaceVisible(false);
        SetGeometricSurfaceGridVisible(false);
        SetSurfaceNormalIndicatorsVisible(false);
        SetRemeshGridLinesVisible(false);
        SetCenterDebugMarkersVisible(false);

        if (valid == null || positions == null || normals == null || _groups.Count <= 0)
        {
            SetCandidatePlaneObjectsVisible(false);
            return false;
        }

        EnsureCandidatePlaneObjects(Mathf.Max(1, candidatePlaneObjectCount));
        if (_candidatePlaneMeshes.Count <= 0)
        {
            SetCandidatePlaneObjectsVisible(false);
            return false;
        }

        List<DirectGridPlaneCandidate> candidates = new List<DirectGridPlaneCandidate>(32);
        Transform localTransform = ResolveDisplayLocalTransform();

        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            if (group.columns < 2 || group.rows < 2)
                continue;

            int tileColumns = Mathf.Max(3, Mathf.CeilToInt(group.columns / 4f));
            int tileRows = Mathf.Max(3, Mathf.CeilToInt(group.rows / 4f));
            int stepColumns = Mathf.Max(2, tileColumns / 2);
            int stepRows = Mathf.Max(2, tileRows / 2);

            for (int row = 0; row < group.rows; row += stepRows)
            {
                int endRow = Mathf.Min(group.rows, row + tileRows);
                if (endRow - row < 2)
                    continue;

                for (int col = 0; col < group.columns; col += stepColumns)
                {
                    int endCol = Mathf.Min(group.columns, col + tileColumns);
                    if (endCol - col < 2)
                        continue;

                    if (TryBuildDirectGridPlaneCandidate(group, row, endRow, col, endCol, valid, positions, normals, localTransform, out DirectGridPlaneCandidate candidate))
                        candidates.Add(candidate);
                }
            }
        }

        if (candidates.Count <= 0)
        {
            SetCandidatePlaneObjectsVisible(false);
            return false;
        }

        candidates.Sort((a, b) => b.score.CompareTo(a.score));
        List<DirectGridPlaneCandidate> selected = new List<DirectGridPlaneCandidate>(Mathf.Max(1, candidatePlaneObjectCount));
        for (int i = 0; i < candidates.Count && selected.Count < candidatePlaneObjectCount; i++)
        {
            bool overlaps = false;
            for (int j = 0; j < selected.Count; j++)
            {
                if (DirectPlaneCandidatesOverlap(candidates[i], selected[j]))
                {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps)
                selected.Add(candidates[i]);
        }

        int shownCount = 0;
        for (int i = 0; i < selected.Count && shownCount < _candidatePlaneMeshes.Count; i++)
        {
            if (!BuildDirectGridPlaneMesh(selected[i], shownCount))
                continue;

            if (shownCount < _candidatePlaneRenderers.Count && _candidatePlaneRenderers[shownCount] != null)
                _candidatePlaneRenderers[shownCount].enabled = visibleAfterBuild;
            shownCount++;
        }

        for (int i = shownCount; i < _candidatePlaneRenderers.Count; i++)
        {
            if (_candidatePlaneRenderers[i] != null)
                _candidatePlaneRenderers[i].enabled = false;
        }

        SetCandidatePlaneObjectsVisible(visibleAfterBuild && shownCount > 0);
        return shownCount > 0;
    }

    private bool TryBuildDirectGridPlaneCandidate(GridGroup group, int startRow, int endRow, int startCol, int endCol, bool[] valid, Vector3[] positions, Vector3[] normals, Transform localTransform, out DirectGridPlaneCandidate candidate)
    {
        candidate = default;
        Vector3 centerSum = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        int sampleCount = 0;

        for (int row = startRow; row < endRow; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = startCol; col < endCol; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || !valid[index])
                    continue;

                Vector3 point = positions[index];
                Vector3 normal = normals[index];
                if (localTransform != null)
                {
                    point = localTransform.InverseTransformPoint(point);
                    normal = localTransform.InverseTransformDirection(normal);
                }

                centerSum += point;
                if (normal.sqrMagnitude > 1e-8f)
                    normalSum += normal.normalized;
                sampleCount++;
            }
        }

        if (sampleCount < 4)
            return false;

        Vector3 planeCenter = centerSum / sampleCount;
        Vector3 planeNormal = normalSum.sqrMagnitude > 1e-8f ? normalSum.normalized : Vector3.forward;
        float minNormalDot = Mathf.Cos(22f * Mathf.Deg2Rad);
        float maxPlaneOffset = Mathf.Max(0.05f, surfaceRegionMaxPlaneOffsetMeters, candidateSurfaceMergePlaneOffsetMeters);

        Vector3 acceptedCenterSum = Vector3.zero;
        Vector3 acceptedNormalSum = Vector3.zero;
        int acceptedCount = 0;
        for (int row = startRow; row < endRow; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = startCol; col < endCol; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || !valid[index])
                    continue;

                Vector3 point = positions[index];
                Vector3 normal = normals[index];
                if (localTransform != null)
                {
                    point = localTransform.InverseTransformPoint(point);
                    normal = localTransform.InverseTransformDirection(normal);
                }

                if (normal.sqrMagnitude > 1e-8f && Vector3.Dot(planeNormal, normal.normalized) < minNormalDot)
                    continue;
                if (Mathf.Abs(Vector3.Dot(point - planeCenter, planeNormal)) > maxPlaneOffset)
                    continue;

                acceptedCenterSum += point;
                if (normal.sqrMagnitude > 1e-8f)
                    acceptedNormalSum += normal.normalized;
                acceptedCount++;
            }
        }

        if (acceptedCount < 4 || acceptedCount < Mathf.Max(4, sampleCount / 3))
            return false;

        planeCenter = acceptedCenterSum / acceptedCount;
        if (acceptedNormalSum.sqrMagnitude > 1e-8f)
            planeNormal = acceptedNormalSum.normalized;

        Vector3 localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            return false;

        localUp.Normalize();
        Vector3 localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return false;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;

        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        for (int row = startRow; row < endRow; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = startCol; col < endCol; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || !valid[index])
                    continue;

                Vector3 point = positions[index];
                Vector3 normal = normals[index];
                if (localTransform != null)
                {
                    point = localTransform.InverseTransformPoint(point);
                    normal = localTransform.InverseTransformDirection(normal);
                }

                if (normal.sqrMagnitude > 1e-8f && Vector3.Dot(planeNormal, normal.normalized) < minNormalDot)
                    continue;
                if (Mathf.Abs(Vector3.Dot(point - planeCenter, planeNormal)) > maxPlaneOffset)
                    continue;

                Vector3 delta = point - planeCenter;
                float u = Vector3.Dot(delta, localRight);
                float v = Vector3.Dot(delta, localUp);
                minU = Mathf.Min(minU, u);
                maxU = Mathf.Max(maxU, u);
                minV = Mathf.Min(minV, v);
                maxV = Mathf.Max(maxV, v);
            }
        }

        if (!float.IsFinite(minU) || !float.IsFinite(maxU) || !float.IsFinite(minV) || !float.IsFinite(maxV))
            return false;

        float halfWidth = Mathf.Max(candidatePlaneMinSizeMeters * 0.5f, (maxU - minU) * 0.5f + candidatePlanePaddingMeters);
        float halfHeight = Mathf.Max(candidatePlaneMinSizeMeters * 0.5f, (maxV - minV) * 0.5f + candidatePlanePaddingMeters);
        if (halfWidth <= 1e-4f || halfHeight <= 1e-4f)
            return false;

        float groupCenterRow = (group.rows - 1) * 0.5f;
        float groupCenterCol = (group.columns - 1) * 0.5f;
        float tileCenterRow = (startRow + endRow - 1) * 0.5f;
        float tileCenterCol = (startCol + endCol - 1) * 0.5f;
        float normalizedCenterDistance = Vector2.Distance(new Vector2(tileCenterCol, tileCenterRow), new Vector2(groupCenterCol, groupCenterRow)) / Mathf.Max(1f, Mathf.Max(group.columns, group.rows));

        candidate = new DirectGridPlaneCandidate
        {
            group = group.group,
            startRow = startRow,
            endRow = endRow,
            startCol = startCol,
            endCol = endCol,
            pointCount = acceptedCount,
            score = acceptedCount * (1.35f - Mathf.Clamp01(normalizedCenterDistance)) + halfWidth * halfHeight * 16f,
            center = planeCenter,
            normal = planeNormal,
            right = localRight,
            up = localUp,
            halfWidth = halfWidth,
            halfHeight = halfHeight
        };
        return true;
    }

    private static bool DirectPlaneCandidatesOverlap(DirectGridPlaneCandidate a, DirectGridPlaneCandidate b)
    {
        if (a.group != b.group)
            return false;

        int overlapRows = Mathf.Min(a.endRow, b.endRow) - Mathf.Max(a.startRow, b.startRow);
        int overlapCols = Mathf.Min(a.endCol, b.endCol) - Mathf.Max(a.startCol, b.startCol);
        if (overlapRows <= 0 || overlapCols <= 0)
            return false;

        int overlapArea = overlapRows * overlapCols;
        int areaA = Mathf.Max(1, (a.endRow - a.startRow) * (a.endCol - a.startCol));
        int areaB = Mathf.Max(1, (b.endRow - b.startRow) * (b.endCol - b.startCol));
        return overlapArea > Mathf.Min(areaA, areaB) / 4;
    }

    private bool BuildDirectGridPlaneMesh(DirectGridPlaneCandidate candidate, int planeIndex)
    {
        if (planeIndex < 0 || planeIndex >= _candidatePlaneMeshes.Count || _candidatePlaneMeshes[planeIndex] == null)
            return false;

        Vector3 planeNormal = candidate.normal.sqrMagnitude > 1e-8f ? candidate.normal.normalized : Vector3.forward;
        Vector3 offset = planeNormal * Mathf.Max(0f, candidatePlaneSurfaceOffsetMeters);
        Vector3 p00 = candidate.center - candidate.right * candidate.halfWidth - candidate.up * candidate.halfHeight + offset;
        Vector3 p10 = candidate.center + candidate.right * candidate.halfWidth - candidate.up * candidate.halfHeight + offset;
        Vector3 p11 = candidate.center + candidate.right * candidate.halfWidth + candidate.up * candidate.halfHeight + offset;
        Vector3 p01 = candidate.center - candidate.right * candidate.halfWidth + candidate.up * candidate.halfHeight + offset;

        Mesh mesh = _candidatePlaneMeshes[planeIndex];
        mesh.Clear();
        mesh.SetVertices(new List<Vector3> { p00, p10, p11, p01 });
        mesh.SetNormals(new List<Vector3> { planeNormal, planeNormal, planeNormal, planeNormal });
        mesh.SetUVs(0, new List<Vector2> { Vector2.zero, Vector2.right, Vector2.one, Vector2.up });
        mesh.SetTriangles(new[] { 0, 1, 2, 0, 2, 3 }, 0, true);
        mesh.RecalculateBounds();
        return true;
    }

    private void BuildSurfaceNormalIndicatorMesh(List<Vector3> vertices, List<int> triangles, bool visibleAfterBuild)
    {
        if (!showSurfaceNormalIndicators)
        {
            SetSurfaceNormalIndicatorsVisible(false);
            return;
        }

        EnsureSurfaceNormalObjects();
        if (_surfaceNormalMesh == null)
        {
            SetSurfaceNormalIndicatorsVisible(false);
            return;
        }

        int faceCount = triangles.Count / 3;
        if (faceCount <= 0)
        {
            _surfaceNormalMesh.Clear();
            SetSurfaceNormalIndicatorsVisible(false);
            return;
        }

        List<Vector3> indicatorVerts = new List<Vector3>(faceCount * 4);
        List<int> indicatorTris = new List<int>(faceCount * 6);
        float length = Mathf.Max(0.002f, surfaceNormalIndicatorLengthMeters);
        float halfThickness = Mathf.Max(0.00025f, surfaceNormalIndicatorThicknessMeters * 0.5f);

        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int triStart = faceIndex * 3;
            int ia = triangles[triStart];
            int ib = triangles[triStart + 1];
            int ic = triangles[triStart + 2];
            Vector3 va = vertices[ia];
            Vector3 vb = vertices[ib];
            Vector3 vc = vertices[ic];
            Vector3 normal = Vector3.Cross(vb - va, vc - va);
            if (normal.sqrMagnitude <= 1e-8f)
                continue;
            normal.Normalize();

            Vector3 averagedVertexNormal = Vector3.zero;
            if (ia >= 0 && ia < _normals.Count)
                averagedVertexNormal += _normals[ia];
            if (ib >= 0 && ib < _normals.Count)
                averagedVertexNormal += _normals[ib];
            if (ic >= 0 && ic < _normals.Count)
                averagedVertexNormal += _normals[ic];
            if (averagedVertexNormal.sqrMagnitude > 1e-8f && Vector3.Dot(normal, averagedVertexNormal.normalized) < 0f)
                normal = -normal;

            Vector3 center = (va + vb + vc) / 3f;
            Vector3 tangent = Vector3.Cross(normal, Vector3.up);
            if (tangent.sqrMagnitude <= 1e-8f)
                tangent = Vector3.Cross(normal, Vector3.right);
            if (tangent.sqrMagnitude <= 1e-8f)
                continue;
            tangent.Normalize();

            Vector3 side = tangent * halfThickness;
            Vector3 start = center + normal * 0.001f;
            Vector3 end = start + normal * length;

            int baseIndex = indicatorVerts.Count;
            indicatorVerts.Add(start - side);
            indicatorVerts.Add(start + side);
            indicatorVerts.Add(end + side);
            indicatorVerts.Add(end - side);

            indicatorTris.Add(baseIndex);
            indicatorTris.Add(baseIndex + 1);
            indicatorTris.Add(baseIndex + 2);
            indicatorTris.Add(baseIndex);
            indicatorTris.Add(baseIndex + 2);
            indicatorTris.Add(baseIndex + 3);
        }

        _surfaceNormalMesh.Clear();
        if (indicatorVerts.Count <= 0 || indicatorTris.Count <= 0)
        {
            SetSurfaceNormalIndicatorsVisible(false);
            return;
        }

        _surfaceNormalMesh.SetVertices(indicatorVerts);
        _surfaceNormalMesh.SetTriangles(indicatorTris, 0, true);
        _surfaceNormalMesh.RecalculateNormals();
        _surfaceNormalMesh.RecalculateBounds();
        SetSurfaceNormalIndicatorsVisible(visibleAfterBuild);
    }

    private void BuildGridLineMesh(bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        EnsureGridLineObjects();
        if (_lineMesh == null)
        {
            SetGridLinesVisible(false);
            return;
        }

        if (syncGridLinesToFocusedCandidate && _focusedGridOverlayState.isValid)
        {
            SetHeadsetScreenCenterMarkerVisible(false);
            SetOriginalGridCenterMarkerVisible(false);
            BuildFocusedGridOverlayLineMesh(valid, positions, normals);
            return;
        }

        if (_groups.Count <= 0)
        {
            SetGridLinesVisible(false);
            return;
        }

        _verts.Clear();
        _lineIndices.Clear();
        int[] vertexIndices = new int[_cells.Count];
        Vector3[] sourceLocalPositions = new Vector3[_cells.Count];
        Vector3[] sourceLocalNormals = new Vector3[_cells.Count];
        Vector3[] displayPositions = new Vector3[_cells.Count];
        bool[] displayValid = new bool[_cells.Count];
        for (int i = 0; i < vertexIndices.Length; i++) vertexIndices[i] = -1;

        Transform lineTransform = ResolveDisplayLocalTransform();
        for (int i = 0; i < _cells.Count; i++)
        {
            if (!valid[i]) continue;
            Vector3 vertexPosition = positions[i];
            Vector3 vertexNormal = normals[i];
            if (gridLineSurfaceOffsetMeters > 0f && normals[i].sqrMagnitude > 1e-6f)
                vertexPosition += normals[i].normalized * gridLineSurfaceOffsetMeters;
            if (lineTransform != null)
            {
                vertexPosition = lineTransform.InverseTransformPoint(vertexPosition);
                vertexNormal = lineTransform.InverseTransformDirection(vertexNormal);
            }
            sourceLocalPositions[i] = vertexPosition;
            sourceLocalNormals[i] = vertexNormal.sqrMagnitude > 1e-6f ? vertexNormal.normalized : Vector3.zero;
            displayPositions[i] = vertexPosition;
            displayValid[i] = true;
        }

        bool useRectilinearContourFill = showRectilinearFillInsideContour && !regularGridUseVerticalDepthPlaneExperiment;
        if (rectifyGridLinesAfterDepthWrap && !useRectilinearContourFill && !regularGridUseVerticalDepthPlaneExperiment)
        {
            for (int g = 0; g < _groups.Count; g++)
            {
                GridGroup group = _groups[g];
                TryBuildTerrainConformedGridLinePositions(group, displayValid, sourceLocalPositions, sourceLocalNormals, displayPositions);
            }
        }

        for (int i = 0; i < _cells.Count; i++)
        {
            if (!displayValid[i]) continue;
            vertexIndices[i] = _verts.Count;
            _verts.Add(displayPositions[i]);
        }

        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            if (group.columns <= 0 || group.rows <= 0) continue;
            float maxLineSpan = 0f;
            if (useRectilinearContourFill)
            {
                maxLineSpan = EstimateGridLineMaxSpan(group, displayValid, sourceLocalPositions);
                AddRectilinearFilledContourGridEdges(group, displayValid, vertexIndices, sourceLocalPositions, maxLineSpan);
                continue;
            }

            if (showGridOuterContourOnly)
            {
                AddGridOuterContourEdges(group, displayValid, vertexIndices, sourceLocalPositions, maxLineSpan);
                continue;
            }

            if (gridLineRequireContinuousSurface || gridLineRequireCompleteCellSupport)
                maxLineSpan = EstimateGridLineMaxSpan(group, displayValid, sourceLocalPositions);

            bool[] supportedHorizontalEdges = null;
            bool[] supportedVerticalEdges = null;
            bool[] supportedCells = null;
            if (gridLineRequireCompleteCellSupport && group.columns > 1 && group.rows > 1)
            {
                supportedHorizontalEdges = new bool[group.rows * (group.columns - 1)];
                supportedVerticalEdges = new bool[(group.rows - 1) * group.columns];
                supportedCells = new bool[(group.rows - 1) * (group.columns - 1)];
                BuildCompleteCellSupportedGridEdges(group, displayValid, sourceLocalPositions, sourceLocalNormals, maxLineSpan, supportedHorizontalEdges, supportedVerticalEdges, supportedCells);
                PruneCompleteCellIslands(group, supportedCells);
                RebuildCompleteCellSupportedGridEdges(group, supportedCells, supportedHorizontalEdges, supportedVerticalEdges);
            }

            for (int row = 0; row < group.rows; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                for (int col = 0; col < group.columns - 1; col++)
                {
                    if (supportedHorizontalEdges != null && !supportedHorizontalEdges[row * (group.columns - 1) + col])
                        continue;
                    AddGridLineEdge(rowStart + col, rowStart + col + 1, displayValid, vertexIndices, sourceLocalPositions, sourceLocalNormals, maxLineSpan);
                }
            }

            for (int row = 0; row < group.rows - 1; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                int nextRowStart = group.startIndex + (row + 1) * group.columns;
                for (int col = 0; col < group.columns; col++)
                {
                    if (supportedVerticalEdges != null && !supportedVerticalEdges[row * group.columns + col])
                        continue;
                    AddGridLineEdge(rowStart + col, nextRowStart + col, displayValid, vertexIndices, sourceLocalPositions, sourceLocalNormals, maxLineSpan);
                }
            }

            if (!showGridTriangulation)
                continue;

            for (int row = 0; row < group.rows - 1; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                int nextRowStart = group.startIndex + (row + 1) * group.columns;
                for (int col = 0; col < group.columns - 1; col++)
                {
                    if (supportedCells != null && !supportedCells[row * (group.columns - 1) + col])
                        continue;
                    int i00 = rowStart + col;
                    int i10 = rowStart + col + 1;
                    int i01 = nextRowStart + col;
                    int i11 = nextRowStart + col + 1;
                    AddTriangulationEdge(i00, i10, i01, i11, displayValid, vertexIndices, positions, normals, confidences);
                }
            }
        }

        // Keep the raw depth-wrapped grid lines; cleanup passes can remove valid reference lines.

        ApplyBuiltGridLineMesh(valid, positions, normals);
    }

    private void ApplyBuiltGridLineMesh(bool[] valid, Vector3[] positions, Vector3[] normals)
    {
        _lineMesh.Clear();
        if (_lineIndices.Count <= 0)
        {
            SetGridLinesVisible(false);
            return;
        }

        _lineMesh.SetVertices(_verts);
        _lineMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0, true);
        _lineMesh.RecalculateBounds();
        UpdateOriginalGridCenterMarker(valid, positions, normals);
        UpdateHeadsetScreenCenterMarker();
        SetGridLinesVisible(true);
    }

    private bool TryBuildIndependentRectilinearRennetGrid(bool[] valid, Vector3[] sourceLocalPositions, Vector3[] sourceLocalNormals)
    {
        if (valid == null || sourceLocalPositions == null || _groups.Count <= 0)
            return false;

        _verts.Clear();
        _lineIndices.Clear();

        bool builtAny = false;
        List<Vector3> renetVertexNormals = new List<Vector3>(_cells.Count);
        int stride = Mathf.Max(1, rectilinearRennetStride);
        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            if (group.columns < 2 || group.rows < 2)
                continue;

            List<int> sourceRows = BuildStridedGridAxisIndices(group.rows, stride);
            List<int> sourceColumns = BuildStridedGridAxisIndices(group.columns, stride);
            if (sourceRows.Count < 2 || sourceColumns.Count < 2)
                continue;

            int renetColumns = sourceColumns.Count;
            int renetRows = sourceRows.Count;
            float maxRennetEdgeSpan = EstimateGridLineMaxSpan(group, valid, sourceLocalPositions) *
                                      Mathf.Max(1, stride) *
                                      Mathf.Max(1f, rectilinearRennetMaxEdgeSpanMultiplier);

            int[] renetVertexIndices = new int[renetColumns * renetRows];
            for (int i = 0; i < renetVertexIndices.Length; i++)
                renetVertexIndices[i] = -1;

            for (int row = 0; row < renetRows; row++)
            {
                int sourceRow = sourceRows[row];
                int sourceRowStart = group.startIndex + sourceRow * group.columns;
                for (int col = 0; col < renetColumns; col++)
                {
                    int sourceColumn = sourceColumns[col];
                    int sourceIndex = sourceRowStart + sourceColumn;
                    if (sourceIndex < 0 ||
                        sourceIndex >= valid.Length ||
                        sourceIndex >= sourceLocalPositions.Length ||
                        !valid[sourceIndex] ||
                        !Finite(sourceLocalPositions[sourceIndex]))
                        continue;

                    renetVertexIndices[row * renetColumns + col] = _verts.Count;
                    _verts.Add(sourceLocalPositions[sourceIndex]);
                    Vector3 normal = sourceLocalNormals != null && sourceIndex < sourceLocalNormals.Length
                        ? sourceLocalNormals[sourceIndex]
                        : Vector3.zero;
                    renetVertexNormals.Add(normal.sqrMagnitude > 1e-8f ? normal.normalized : Vector3.zero);
                }
            }

            for (int row = 0; row < renetRows; row++)
            {
                int rowStart = row * renetColumns;
                for (int col = 0; col < renetColumns - 1; col++)
                    AddRennetLineEdge(renetVertexIndices[rowStart + col], renetVertexIndices[rowStart + col + 1], renetVertexNormals, maxRennetEdgeSpan);
            }

            for (int row = 0; row < renetRows - 1; row++)
            {
                int rowStart = row * renetColumns;
                int nextRowStart = (row + 1) * renetColumns;
                for (int col = 0; col < renetColumns; col++)
                    AddRennetLineEdge(renetVertexIndices[rowStart + col], renetVertexIndices[nextRowStart + col], renetVertexNormals, maxRennetEdgeSpan);
            }

            builtAny |= _lineIndices.Count > 0;
        }

        return builtAny;
    }

    private static List<int> BuildStridedGridAxisIndices(int count, int stride)
    {
        List<int> indices = new List<int>(Mathf.Max(2, Mathf.CeilToInt(count / (float)Mathf.Max(1, stride))));
        if (count <= 0)
            return indices;

        stride = Mathf.Max(1, stride);
        for (int i = 0; i < count; i += stride)
            indices.Add(i);

        int last = count - 1;
        if (indices.Count == 0 || indices[indices.Count - 1] != last)
            indices.Add(last);

        return indices;
    }

    private bool TryGetGroupProjectedBounds(GridGroup group, bool[] valid, Vector3[] sourceLocalPositions, out float minX, out float maxX, out float minY, out float maxY)
    {
        minX = float.PositiveInfinity;
        maxX = float.NegativeInfinity;
        minY = float.PositiveInfinity;
        maxY = float.NegativeInfinity;
        int count = 0;

        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || index >= sourceLocalPositions.Length || !valid[index])
                    continue;

                Vector3 p = sourceLocalPositions[index];
                if (!Finite(p))
                    continue;

                minX = Mathf.Min(minX, p.x);
                maxX = Mathf.Max(maxX, p.x);
                minY = Mathf.Min(minY, p.y);
                maxY = Mathf.Max(maxY, p.y);
                count++;
            }
        }

        return count >= 4 &&
               float.IsFinite(minX) && float.IsFinite(maxX) &&
               float.IsFinite(minY) && float.IsFinite(maxY) &&
               maxX > minX && maxY > minY;
    }

    private bool TryProjectRectilinearPointToSourceSurface(
        GridGroup group,
        float x,
        float y,
        bool[] valid,
        Vector3[] sourceLocalPositions,
        Vector3[] sourceLocalNormals,
        float sourceMaxSpan,
        out Vector3 hitPosition,
        out Vector3 hitNormal)
    {
        hitPosition = Vector3.zero;
        hitNormal = Vector3.forward;
        bool found = false;
        float bestZ = float.PositiveInfinity;

        for (int row = 0; row < group.rows - 1; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < group.columns - 1; col++)
            {
                int i00 = rowStart + col;
                int i10 = i00 + 1;
                int i01 = nextRowStart + col;
                int i11 = i01 + 1;
                if (!IsCoherentSourceQuad(i00, i10, i01, i11, valid, sourceLocalPositions, sourceLocalNormals, sourceMaxSpan))
                    continue;

                TrySampleProjectedTriangle(x, y, i00, i10, i11, sourceLocalPositions, sourceLocalNormals, ref found, ref bestZ, ref hitPosition, ref hitNormal);
                TrySampleProjectedTriangle(x, y, i00, i11, i01, sourceLocalPositions, sourceLocalNormals, ref found, ref bestZ, ref hitPosition, ref hitNormal);
            }
        }

        return found;
    }

    private bool IsCoherentSourceQuad(int i00, int i10, int i01, int i11, bool[] valid, Vector3[] sourceLocalPositions, Vector3[] sourceLocalNormals, float maxSpan)
    {
        if (!IsValidSourcePoint(i00, valid, sourceLocalPositions) ||
            !IsValidSourcePoint(i10, valid, sourceLocalPositions) ||
            !IsValidSourcePoint(i01, valid, sourceLocalPositions) ||
            !IsValidSourcePoint(i11, valid, sourceLocalPositions))
            return false;

        float spanLimit = Mathf.Max(0.01f, maxSpan);
        if (!SourceEdgeWithinSpan(i00, i10, sourceLocalPositions, spanLimit) ||
            !SourceEdgeWithinSpan(i00, i01, sourceLocalPositions, spanLimit) ||
            !SourceEdgeWithinSpan(i10, i11, sourceLocalPositions, spanLimit) ||
            !SourceEdgeWithinSpan(i01, i11, sourceLocalPositions, spanLimit) ||
            !SourceEdgeWithinSpan(i00, i11, sourceLocalPositions, spanLimit * 1.45f) ||
            !SourceEdgeWithinSpan(i10, i01, sourceLocalPositions, spanLimit * 1.45f))
            return false;

        return SourceNormalsAreCoherent(i00, i10, i01, i11, sourceLocalNormals);
    }

    private static bool SourceEdgeWithinSpan(int a, int b, Vector3[] sourceLocalPositions, float maxSpan)
    {
        float distance = Vector3.Distance(sourceLocalPositions[a], sourceLocalPositions[b]);
        return float.IsFinite(distance) && distance <= maxSpan;
    }

    private bool SourceNormalsAreCoherent(int i00, int i10, int i01, int i11, Vector3[] sourceLocalNormals)
    {
        if (sourceLocalNormals == null)
            return true;

        return SourceNormalPairIsCoherent(i00, i10, sourceLocalNormals) &&
               SourceNormalPairIsCoherent(i00, i01, sourceLocalNormals) &&
               SourceNormalPairIsCoherent(i10, i11, sourceLocalNormals) &&
               SourceNormalPairIsCoherent(i01, i11, sourceLocalNormals);
    }

    private bool SourceNormalPairIsCoherent(int a, int b, Vector3[] sourceLocalNormals)
    {
        if (a < 0 || b < 0 || a >= sourceLocalNormals.Length || b >= sourceLocalNormals.Length)
            return true;

        Vector3 na = sourceLocalNormals[a];
        Vector3 nb = sourceLocalNormals[b];
        if (na.sqrMagnitude <= 1e-8f || nb.sqrMagnitude <= 1e-8f)
            return true;

        return Vector3.Dot(na.normalized, nb.normalized) >= rectilinearRennetMinNormalDot;
    }

    private static bool IsValidSourcePoint(int index, bool[] valid, Vector3[] sourceLocalPositions)
    {
        return index >= 0 &&
               valid != null &&
               sourceLocalPositions != null &&
               index < valid.Length &&
               index < sourceLocalPositions.Length &&
               valid[index] &&
               Finite(sourceLocalPositions[index]);
    }

    private void TrySampleProjectedTriangle(
        float x,
        float y,
        int ia,
        int ib,
        int ic,
        Vector3[] sourceLocalPositions,
        Vector3[] sourceLocalNormals,
        ref bool found,
        ref float bestZ,
        ref Vector3 hitPosition,
        ref Vector3 hitNormal)
    {
        Vector3 a = sourceLocalPositions[ia];
        Vector3 b = sourceLocalPositions[ib];
        Vector3 c = sourceLocalPositions[ic];
        if (!TryBarycentricXY(x, y, a, b, c, out float wa, out float wb, out float wc))
            return;

        float z = a.z * wa + b.z * wb + c.z * wc;
        if (!float.IsFinite(z) || z < 0f || z >= bestZ)
            return;

        Vector3 normal = Vector3.zero;
        if (sourceLocalNormals != null)
        {
            if (ia < sourceLocalNormals.Length) normal += sourceLocalNormals[ia] * wa;
            if (ib < sourceLocalNormals.Length) normal += sourceLocalNormals[ib] * wb;
            if (ic < sourceLocalNormals.Length) normal += sourceLocalNormals[ic] * wc;
        }

        if (normal.sqrMagnitude <= 1e-8f)
            normal = Vector3.Cross(b - a, c - a);

        bestZ = z;
        found = true;
        hitPosition = new Vector3(x, y, z);
        hitNormal = normal.sqrMagnitude > 1e-8f ? normal.normalized : Vector3.forward;
    }

    private static bool TryBarycentricXY(float x, float y, Vector3 a, Vector3 b, Vector3 c, out float wa, out float wb, out float wc)
    {
        Vector2 p = new Vector2(x, y);
        Vector2 a2 = new Vector2(a.x, a.y);
        Vector2 b2 = new Vector2(b.x, b.y);
        Vector2 c2 = new Vector2(c.x, c.y);
        Vector2 v0 = b2 - a2;
        Vector2 v1 = c2 - a2;
        Vector2 v2 = p - a2;
        float denom = v0.x * v1.y - v1.x * v0.y;
        if (Mathf.Abs(denom) <= 1e-8f)
        {
            wa = wb = wc = 0f;
            return false;
        }

        wb = (v2.x * v1.y - v1.x * v2.y) / denom;
        wc = (v0.x * v2.y - v2.x * v0.y) / denom;
        wa = 1f - wb - wc;
        const float epsilon = -0.0001f;
        return wa >= epsilon && wb >= epsilon && wc >= epsilon;
    }

    private void AddRennetLineEdge(int a, int b, List<Vector3> renetVertexNormals, float maxSpan)
    {
        if (a < 0 || b < 0 || a >= _verts.Count || b >= _verts.Count)
            return;

        float distance = Vector3.Distance(_verts[a], _verts[b]);
        if (!float.IsFinite(distance) || distance <= 1e-5f || distance > Mathf.Max(0.01f, maxSpan))
            return;

        if (renetVertexNormals != null && a < renetVertexNormals.Count && b < renetVertexNormals.Count)
        {
            Vector3 na = renetVertexNormals[a];
            Vector3 nb = renetVertexNormals[b];
            if (na.sqrMagnitude > 1e-8f && nb.sqrMagnitude > 1e-8f &&
                Vector3.Dot(na.normalized, nb.normalized) < rectilinearRennetMinNormalDot)
                return;
        }

        _lineIndices.Add(a);
        _lineIndices.Add(b);
    }

    private void BuildFocusedGridOverlayLineMesh(bool[] valid, Vector3[] positions, Vector3[] normals)
    {
        _verts.Clear();
        _lineIndices.Clear();
        FocusedGridOverlayState state = _focusedGridOverlayState;
        if (!state.isValid || state.cellColumns <= 0 || state.cellRows <= 0 || _groups.Count <= 0)
        {
            SetGridLinesVisible(false);
            return;
        }

        int vertexColumns = Mathf.Max(1, state.cellColumns);
        int vertexRows = Mathf.Max(1, state.cellRows);
        if (vertexColumns <= 0 || vertexRows <= 0)
        {
            SetGridLinesVisible(false);
            return;
        }

        Transform lineTransform = ResolveDisplayLocalTransform();
        Vector3 planeNormal = state.planeNormal.sqrMagnitude > 1e-8f ? state.planeNormal.normalized : Vector3.forward;
        Vector3 planeOffset = gridLineSurfaceOffsetMeters > 0f ? planeNormal * gridLineSurfaceOffsetMeters : Vector3.zero;
        float planeTolerance = Mathf.Max(0.001f, focusedGridPlaneToleranceMeters);
        float sampleSearchRadius = Mathf.Max(state.stepU, state.stepV) * 1.25f;
        List<Vector2> sampleUvs = new List<Vector2>();
        List<float> samplePlaneOffsets = new List<float>();

        if (valid != null && positions != null)
        {
            GridGroup group = _groups[0];
            for (int row = 0; row < group.rows; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                for (int col = 0; col < group.columns; col++)
                {
                    int sourceIndex = rowStart + col;
                    if (sourceIndex < 0 || sourceIndex >= _cells.Count || sourceIndex >= valid.Length || !valid[sourceIndex] || sourceIndex >= positions.Length)
                        continue;

                    Vector3 sourceLocal = lineTransform != null ? lineTransform.InverseTransformPoint(positions[sourceIndex]) : positions[sourceIndex];
                    Vector3 localDelta = sourceLocal - state.planeOrigin;
                    float sampleU = Vector3.Dot(localDelta, state.localRight);
                    float sampleV = Vector3.Dot(localDelta, state.localUp);
                    float samplePlaneOffset = Vector3.Dot(localDelta, planeNormal);
                    if (Mathf.Abs(samplePlaneOffset) > planeTolerance)
                        continue;

                    sampleUvs.Add(new Vector2(sampleU, sampleV));
                    samplePlaneOffsets.Add(samplePlaneOffset);
                }
            }
        }

        int[] vertexIndices = new int[vertexColumns * vertexRows];
        for (int i = 0; i < vertexIndices.Length; i++)
            vertexIndices[i] = -1;

        for (int y = 0; y < vertexRows; y++)
        {
            float v = state.minV + (y + 0.5f) * state.stepV;
            for (int x = 0; x < vertexColumns; x++)
            {
                float u = state.minU + (x + 0.5f) * state.stepU;
                Vector3 vertexPosition = state.planeOrigin + state.localRight * u + state.localUp * v;
                if (sampleUvs.Count > 0)
                {
                    float bestDistanceSqr = float.PositiveInfinity;
                    float bestPlaneOffset = 0f;
                    for (int i = 0; i < sampleUvs.Count; i++)
                    {
                        Vector2 delta = sampleUvs[i] - new Vector2(u, v);
                        if (Mathf.Abs(delta.x) > sampleSearchRadius || Mathf.Abs(delta.y) > sampleSearchRadius)
                            continue;

                        float distanceSqr = delta.sqrMagnitude;
                        if (distanceSqr >= bestDistanceSqr)
                            continue;

                        bestDistanceSqr = distanceSqr;
                        bestPlaneOffset = samplePlaneOffsets[i];
                    }

                    if (float.IsFinite(bestDistanceSqr))
                        vertexPosition += planeNormal * bestPlaneOffset;
                }

                vertexPosition += planeOffset;
                vertexIndices[y * vertexColumns + x] = _verts.Count;
                _verts.Add(vertexPosition);
            }
        }

        int VertexIndex(int x, int y) => vertexIndices[y * vertexColumns + x];
        for (int y = 0; y < vertexRows; y++)
        {
            for (int x = 0; x < vertexColumns - 1; x++)
            {
                AddFocusedLineEdge(VertexIndex(x, y), VertexIndex(x + 1, y));
            }
        }

        for (int y = 0; y < vertexRows - 1; y++)
        {
            for (int x = 0; x < vertexColumns; x++)
            {
                AddFocusedLineEdge(VertexIndex(x, y), VertexIndex(x, y + 1));
            }
        }

        if (showGridTriangulation)
        {
            for (int y = 0; y < vertexRows - 1; y++)
            {
                for (int x = 0; x < vertexColumns - 1; x++)
                {
                    AddFocusedLineEdge(VertexIndex(x, y), VertexIndex(x + 1, y + 1));
                }
            }
        }

        _lineMesh.Clear();
        if (_verts.Count <= 0 || _lineIndices.Count <= 0)
        {
            SetGridLinesVisible(false);
            return;
        }

        _lineMesh.SetVertices(_verts);
        _lineMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0, true);
        _lineMesh.RecalculateBounds();
        SetGridLinesVisible(true);
    }

    private void AddFocusedLineEdge(int a, int b)
    {
        if (a < 0 || b < 0 || a == b)
            return;

        _lineIndices.Add(a);
        _lineIndices.Add(b);
    }

    private void BuildLargestCandidateGridLineMesh(List<Vector3> lineVertices, List<Vector3> lineNormals, List<int> lineIndices, bool visibleAfterBuild)
    {
        EnsureRemeshGridLineObjects();
        if (_remeshLineMesh == null)
        {
            SetRemeshGridLinesVisible(false);
            return;
        }

        _remeshLineMesh.Clear();
        if (lineVertices == null || lineIndices == null || lineVertices.Count <= 0 || lineIndices.Count <= 0)
        {
            SetRemeshGridLinesVisible(false);
            return;
        }

        if (!TryBuildCandidateLineRibbonMesh(lineVertices, lineNormals, lineIndices, largestCandidateGridLineWidthMeters, out List<Vector3> ribbonVertices, out List<int> ribbonTriangles))
        {
            SetRemeshGridLinesVisible(false);
            return;
        }

        _remeshLineMesh.SetVertices(ribbonVertices);
        _remeshLineMesh.SetTriangles(ribbonTriangles, 0, true);
        _remeshLineMesh.RecalculateBounds();
        ApplyRemeshGridLineRendererColor(largestCandidateGridLineColor);
        SetRemeshGridLinesVisible(visibleAfterBuild);
    }

    private void BuildHeightSliceContourDisplay(List<Vector3> vertices, List<Vector3> normals, List<int> triangles, bool visibleAfterBuild)
    {
        EnsureRemeshGridLineObjects();
        bool contourVisible = _previewVisible && previewDisplayVisible && showHeightSliceContour;
        if (_remeshLineMesh == null)
        {
            SetRemeshGridLinesVisible(false);
            return;
        }

        if (vertices == null || triangles == null || vertices.Count <= 0 || triangles.Count < 3 ||
            !TryGetHeightSlicePlaneLocal(vertices, out Vector3 planeOrigin, out Vector3 planeNormal, out Vector3 planeRight, out Vector3 planeForward))
        {
            _remeshLineMesh.Clear();
            SetRemeshGridLinesVisible(false);
            return;
        }

        List<Vector3> lineVertices = new List<Vector3>(triangles.Count);
        List<Vector3> lineNormals = new List<Vector3>(triangles.Count);
        List<int> lineIndices = new List<int>(triangles.Count);
        float epsilon = Mathf.Max(0.0005f, heightSliceEpsilonMeters);
        float maxSegmentLength = Mathf.Max(0.01f, heightSliceMaxSegmentMeters);

        AppendHeightSliceIntersectionRows(vertices, triangles, planeOrigin, planeNormal, Mathf.Max(1, heightSliceRowCount), epsilon, maxSegmentLength, lineVertices, lineNormals, lineIndices);
        if (heightSliceShowPerpendicularColumns)
            AppendHeightSliceIntersectionRows(vertices, triangles, planeOrigin, planeRight, Mathf.Max(1, heightSliceColumnCount), epsilon, maxSegmentLength, lineVertices, lineNormals, lineIndices);

        if (showHeightSlicePlaneFrame)
            AddHeightSlicePlaneFrame(vertices, planeOrigin, planeNormal, planeRight, planeForward, lineVertices, lineNormals, lineIndices);
        if (heightSliceShowPerpendicularColumns && heightSliceShowSampleColumnPlaneFrames)
            AddHeightSliceSamplePlaneFrames(vertices, planeOrigin, planeRight, planeNormal, planeForward, Mathf.Max(1, heightSliceSampleColumnPlaneFrameCount), lineVertices, lineNormals, lineIndices);

        _remeshLineMesh.Clear();
        if (lineIndices.Count <= 0 ||
            !TryBuildCandidateLineRibbonMesh(lineVertices, lineNormals, lineIndices, heightSliceLineWidthMeters, out List<Vector3> ribbonVertices, out List<int> ribbonTriangles))
        {
            SetRemeshGridLinesVisible(false);
            return;
        }

        _remeshLineMesh.SetVertices(ribbonVertices);
        _remeshLineMesh.SetTriangles(ribbonTriangles, 0, true);
        _remeshLineMesh.RecalculateBounds();
        ApplyRemeshGridLineRendererColor(heightSliceContourColor);
        SetRemeshGridLinesVisible(contourVisible);
    }

    private void BuildGeometricSurfaceGridOverlay(List<Vector3> vertices, List<Vector3> normals, List<int> triangles, bool visibleAfterBuild)
    {
        bool overlayVisible = _previewVisible && previewDisplayVisible && showGeometricSurfaceGrid;
        if (!overlayVisible || vertices == null || triangles == null || vertices.Count <= 0 || triangles.Count < 3)
        {
            SetGeometricSurfaceGridVisible(false);
            return;
        }

        EnsureGeometricSurfaceGridObjects();
        if (_geometricSurfaceGridMesh == null)
        {
            SetGeometricSurfaceGridVisible(false);
            return;
        }

        if (geometricSurfaceGridUseRansacPatches &&
            TryBuildRansacPatchGridOverlay(vertices, normals, out List<Vector3> ransacLineVertices, out List<Vector3> ransacLineNormals, out List<int> ransacLineIndices))
        {
            _geometricSurfaceGridMesh.Clear();
            if (TryBuildCandidateLineRibbonMesh(ransacLineVertices, ransacLineNormals, ransacLineIndices, geometricSurfaceGridLineWidthMeters, out List<Vector3> ransacRibbonVertices, out List<int> ransacRibbonTriangles))
            {
                _geometricSurfaceGridMesh.SetVertices(ransacRibbonVertices);
                _geometricSurfaceGridMesh.SetTriangles(ransacRibbonTriangles, 0, true);
                _geometricSurfaceGridMesh.RecalculateBounds();
                ApplyGeometricSurfaceGridRendererColor(geometricSurfaceGridColor);
                SetGeometricSurfaceGridVisible(true);
                return;
            }
        }

        if (geometricSurfaceGridUseRansacPatches)
        {
            _geometricSurfaceGridMesh.Clear();
            SetGeometricSurfaceGridVisible(false);
            return;
        }

        List<Vector3> lineVertices = new List<Vector3>(triangles.Count);
        List<Vector3> lineNormals = new List<Vector3>(triangles.Count);
        List<int> lineIndices = new List<int>(triangles.Count);
        Vector3 planeNormal = ResolveDisplayLocalTransform() != null
            ? ResolveDisplayLocalTransform().InverseTransformDirection(Vector3.up)
            : Vector3.up;
        if (planeNormal.sqrMagnitude <= 1e-8f || !Finite(planeNormal))
        {
            SetGeometricSurfaceGridVisible(false);
            return;
        }

        planeNormal.Normalize();
        float minDistance = float.PositiveInfinity;
        float maxDistance = float.NegativeInfinity;
        Vector3 center = Vector3.zero;
        int finiteCount = 0;
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 vertex = vertices[i];
            if (!Finite(vertex))
                continue;
            float distance = Vector3.Dot(vertex, planeNormal);
            minDistance = Mathf.Min(minDistance, distance);
            maxDistance = Mathf.Max(maxDistance, distance);
            center += vertex;
            finiteCount++;
        }

        if (finiteCount <= 0 || !float.IsFinite(minDistance) || !float.IsFinite(maxDistance) || maxDistance <= minDistance)
        {
            SetGeometricSurfaceGridVisible(false);
            return;
        }

        center /= finiteCount;
        float spacing = Mathf.Max(0.01f, geometricSurfaceGridSpacingMeters);
        int sliceCount = Mathf.Clamp(Mathf.FloorToInt((maxDistance - minDistance) / spacing) + 1, 1, 160);
        float epsilon = Mathf.Max(0.0005f, spacing * 0.01f);
        float maxSegmentLength = Mathf.Max(spacing * 4f, 0.35f);
        AppendHeightSliceIntersectionRows(vertices, triangles, center, planeNormal, sliceCount, epsilon, maxSegmentLength, lineVertices, lineNormals, lineIndices);

        _geometricSurfaceGridMesh.Clear();
        if (lineIndices.Count <= 0 ||
            !TryBuildCandidateLineRibbonMesh(lineVertices, lineNormals, lineIndices, geometricSurfaceGridLineWidthMeters, out List<Vector3> ribbonVertices, out List<int> ribbonTriangles))
        {
            SetGeometricSurfaceGridVisible(false);
            return;
        }

        _geometricSurfaceGridMesh.SetVertices(ribbonVertices);
        _geometricSurfaceGridMesh.SetTriangles(ribbonTriangles, 0, true);
        _geometricSurfaceGridMesh.RecalculateBounds();
        ApplyGeometricSurfaceGridRendererColor(geometricSurfaceGridColor);
        SetGeometricSurfaceGridVisible(true);
    }

    private bool TryBuildRansacPatchGridOverlay(
        List<Vector3> vertices,
        List<Vector3> normals,
        out List<Vector3> lineVertices,
        out List<Vector3> lineNormals,
        out List<int> lineIndices)
    {
        lineVertices = new List<Vector3>(1024);
        lineNormals = new List<Vector3>(1024);
        lineIndices = new List<int>(2048);
        if (vertices == null || vertices.Count < ransacPatchMinInliers)
            return false;

        Vector3[] axes = GetRansacLocalAxes();
        List<RansacPatchSample>[] buckets = new List<RansacPatchSample>[axes.Length];
        for (int i = 0; i < buckets.Length; i++)
            buckets[i] = new List<RansacPatchSample>(Mathf.Max(16, ransacPatchMaxSamples / buckets.Length));

        int maxSamples = Mathf.Max(128, ransacPatchMaxSamples);
        int step = Mathf.Max(1, Mathf.CeilToInt(vertices.Count / (float)maxSamples));
        float looseNormalDot = Mathf.Clamp(ransacPatchNormalDot * 0.65f, 0.25f, 0.85f);
        for (int i = 0; i < vertices.Count; i += step)
        {
            Vector3 point = vertices[i];
            if (!Finite(point))
                continue;

            Vector3 normal = Vector3.zero;
            if (normals != null && i < normals.Count && Finite(normals[i]) && normals[i].sqrMagnitude > 1e-8f)
                normal = normals[i].normalized;
            else
                continue;

            int bestBucket = -1;
            float bestDot = looseNormalDot;
            for (int axisIndex = 0; axisIndex < axes.Length; axisIndex++)
            {
                float dot = Vector3.Dot(normal, axes[axisIndex]);
                if (dot > bestDot)
                {
                    bestDot = dot;
                    bestBucket = axisIndex;
                }
            }

            if (bestBucket < 0)
                continue;

            buckets[bestBucket].Add(new RansacPatchSample
            {
                point = point,
                normal = normal,
                active = true
            });
        }

        bool appendedAny = false;
        for (int bucketIndex = 0; bucketIndex < buckets.Length; bucketIndex++)
        {
            List<RansacPatchSample> bucket = buckets[bucketIndex];
            if (bucket.Count < ransacPatchMinInliers)
                continue;

            for (int planeIndex = 0; planeIndex < ransacPatchPlanesPerBucket; planeIndex++)
            {
                if (!TryExtractRansacPatchPlane(bucket, axes[bucketIndex], bucketIndex, planeIndex, out RansacPatchPlane plane))
                    break;

                int before = lineIndices.Count;
                AppendRansacPatchGrid(bucket, plane, axes[bucketIndex], lineVertices, lineNormals, lineIndices);
                appendedAny |= lineIndices.Count > before;
            }
        }

        return appendedAny && lineIndices.Count > 0;
    }

    private Vector3[] GetRansacLocalAxes()
    {
        Transform localTransform = ResolveDisplayLocalTransform();
        Vector3 right = localTransform != null ? localTransform.InverseTransformDirection(Vector3.right) : Vector3.right;
        Vector3 up = localTransform != null ? localTransform.InverseTransformDirection(Vector3.up) : Vector3.up;
        Vector3 forward = localTransform != null ? localTransform.InverseTransformDirection(Vector3.forward) : Vector3.forward;
        right = SafeNormalized(right, Vector3.right);
        up = SafeNormalized(up, Vector3.up);
        forward = SafeNormalized(forward, Vector3.forward);
        return new[]
        {
            right,
            -right,
            up,
            -up,
            forward,
            -forward
        };
    }

    private bool TryExtractRansacPatchPlane(
        List<RansacPatchSample> samples,
        Vector3 preferredNormal,
        int bucketIndex,
        int planeIndex,
        out RansacPatchPlane plane)
    {
        plane = default;
        List<int> activeIndices = new List<int>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            if (samples[i].active)
                activeIndices.Add(i);
        }

        if (activeIndices.Count < ransacPatchMinInliers)
            return false;

        System.Random rng = new System.Random((_frameIndex + 1) * 73856093 ^ (bucketIndex + 3) * 19349663 ^ (planeIndex + 5) * 83492791);
        int bestInlierCount = 0;
        Vector3 bestPoint = Vector3.zero;
        Vector3 bestNormal = Vector3.zero;
        float normalDotThreshold = Mathf.Clamp01(ransacPatchNormalDot);
        float distanceThreshold = Mathf.Max(0.002f, ransacPatchInlierDistanceMeters);

        for (int iteration = 0; iteration < ransacPatchIterations; iteration++)
        {
            int ia = activeIndices[rng.Next(activeIndices.Count)];
            int ib = activeIndices[rng.Next(activeIndices.Count)];
            int ic = activeIndices[rng.Next(activeIndices.Count)];
            if (ia == ib || ia == ic || ib == ic)
                continue;

            Vector3 a = samples[ia].point;
            Vector3 b = samples[ib].point;
            Vector3 c = samples[ic].point;
            Vector3 candidateNormal = Vector3.Cross(b - a, c - a);
            if (candidateNormal.sqrMagnitude <= 1e-8f || !Finite(candidateNormal))
                continue;

            candidateNormal.Normalize();
            if (Vector3.Dot(candidateNormal, preferredNormal) < 0f)
                candidateNormal = -candidateNormal;
            if (Vector3.Dot(candidateNormal, preferredNormal) < normalDotThreshold * 0.75f)
                continue;

            int inlierCount = 0;
            for (int active = 0; active < activeIndices.Count; active++)
            {
                RansacPatchSample sample = samples[activeIndices[active]];
                float distance = Mathf.Abs(Vector3.Dot(sample.point - a, candidateNormal));
                if (distance <= distanceThreshold && Vector3.Dot(sample.normal, candidateNormal) >= normalDotThreshold)
                    inlierCount++;
            }

            if (inlierCount > bestInlierCount)
            {
                bestInlierCount = inlierCount;
                bestPoint = a;
                bestNormal = candidateNormal;
            }
        }

        if (bestInlierCount < ransacPatchMinInliers || bestNormal.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 center = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        int refinedCount = 0;
        for (int active = 0; active < activeIndices.Count; active++)
        {
            int sampleIndex = activeIndices[active];
            RansacPatchSample sample = samples[sampleIndex];
            float distance = Mathf.Abs(Vector3.Dot(sample.point - bestPoint, bestNormal));
            if (distance > distanceThreshold || Vector3.Dot(sample.normal, bestNormal) < normalDotThreshold)
                continue;

            center += sample.point;
            normalSum += sample.normal;
            refinedCount++;

            RansacPatchSample updated = sample;
            updated.active = false;
            samples[sampleIndex] = updated;
        }

        if (refinedCount < ransacPatchMinInliers)
            return false;

        center /= refinedCount;
        Vector3 refinedNormal = normalSum.sqrMagnitude > 1e-8f ? normalSum.normalized : bestNormal;
        if (Vector3.Dot(refinedNormal, preferredNormal) < 0f)
            refinedNormal = -refinedNormal;

        plane = new RansacPatchPlane
        {
            center = center,
            normal = refinedNormal,
            inlierCount = refinedCount
        };
        return true;
    }

    private void AppendRansacPatchGrid(
        List<RansacPatchSample> samples,
        RansacPatchPlane plane,
        Vector3 preferredAxis,
        List<Vector3> lineVertices,
        List<Vector3> lineNormals,
        List<int> lineIndices)
    {
        Vector3 axisU = BuildPlaneTangent(plane.normal, preferredAxis);
        Vector3 axisV = Vector3.Cross(plane.normal, axisU);
        if (axisV.sqrMagnitude <= 1e-8f)
            return;
        axisV.Normalize();

        float distanceThreshold = Mathf.Max(0.002f, ransacPatchInlierDistanceMeters);
        float normalDotThreshold = Mathf.Clamp01(ransacPatchNormalDot);
        List<Vector2> support = new List<Vector2>(samples.Count);
        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < samples.Count; i++)
        {
            RansacPatchSample sample = samples[i];
            float distance = Mathf.Abs(Vector3.Dot(sample.point - plane.center, plane.normal));
            if (distance > distanceThreshold || Vector3.Dot(sample.normal, plane.normal) < normalDotThreshold)
                continue;

            Vector3 offset = sample.point - plane.center;
            float u = Vector3.Dot(offset, axisU);
            float v = Vector3.Dot(offset, axisV);
            support.Add(new Vector2(u, v));
            minU = Mathf.Min(minU, u);
            maxU = Mathf.Max(maxU, u);
            minV = Mathf.Min(minV, v);
            maxV = Mathf.Max(maxV, v);
        }

        if (support.Count < ransacPatchMinInliers ||
            !float.IsFinite(minU) || !float.IsFinite(maxU) || !float.IsFinite(minV) || !float.IsFinite(maxV))
            return;

        float baseCell = Mathf.Max(0.04f, ransacPatchGridCellMeters);
        float extentU = maxU - minU;
        float extentV = maxV - minV;
        if (extentU < baseCell * 0.75f || extentV < baseCell * 0.75f)
            return;

        int maxCells = Mathf.Max(4, ransacPatchMaxGridCellsPerAxis);
        float cellU = Mathf.Max(baseCell, extentU / maxCells);
        float cellV = Mathf.Max(baseCell, extentV / maxCells);
        int countU = Mathf.Clamp(Mathf.FloorToInt(extentU / cellU) + 1, 2, maxCells + 1);
        int countV = Mathf.Clamp(Mathf.FloorToInt(extentV / cellV) + 1, 2, maxCells + 1);
        if (countU < 2 || countV < 2)
            return;

        int[,] gridIndices = new int[countU, countV];
        for (int uIndex = 0; uIndex < countU; uIndex++)
        {
            for (int vIndex = 0; vIndex < countV; vIndex++)
                gridIndices[uIndex, vIndex] = -1;
        }

        float supportRadius = Mathf.Max(cellU, cellV) * 1.45f;
        Vector3 displayNormal = plane.normal;
        Vector3 surfaceBias = displayNormal * Mathf.Max(0f, geometricSurfaceGridSurfaceOffsetMeters);
        for (int uIndex = 0; uIndex < countU; uIndex++)
        {
            float tu = countU <= 1 ? 0.5f : uIndex / (float)(countU - 1);
            float u = Mathf.Lerp(minU, maxU, tu);
            for (int vIndex = 0; vIndex < countV; vIndex++)
            {
                float tv = countV <= 1 ? 0.5f : vIndex / (float)(countV - 1);
                float v = Mathf.Lerp(minV, maxV, tv);
                if (!HasRansacPatchSupport(support, u, v, supportRadius))
                    continue;

                int lineVertex = lineVertices.Count;
                lineVertices.Add(plane.center + axisU * u + axisV * v + surfaceBias);
                lineNormals.Add(displayNormal);
                gridIndices[uIndex, vIndex] = lineVertex;
            }
        }

        for (int uIndex = 0; uIndex < countU; uIndex++)
        {
            for (int vIndex = 0; vIndex < countV; vIndex++)
            {
                int current = gridIndices[uIndex, vIndex];
                if (current < 0)
                    continue;
                AddRansacGridEdge(current, uIndex + 1 < countU ? gridIndices[uIndex + 1, vIndex] : -1, lineIndices);
                AddRansacGridEdge(current, vIndex + 1 < countV ? gridIndices[uIndex, vIndex + 1] : -1, lineIndices);
                AddRansacGridEdge(current, uIndex + 1 < countU && vIndex + 1 < countV ? gridIndices[uIndex + 1, vIndex + 1] : -1, lineIndices);
            }
        }
    }

    private static bool HasRansacPatchSupport(List<Vector2> support, float u, float v, float radius)
    {
        float radiusSqr = radius * radius;
        Vector2 candidate = new Vector2(u, v);
        for (int i = 0; i < support.Count; i++)
        {
            if ((support[i] - candidate).sqrMagnitude <= radiusSqr)
                return true;
        }

        return false;
    }

    private static void AddRansacGridEdge(int a, int b, List<int> lineIndices)
    {
        if (a < 0 || b < 0 || a == b)
            return;
        lineIndices.Add(a);
        lineIndices.Add(b);
    }

    private static Vector3 BuildPlaneTangent(Vector3 normal, Vector3 preferredAxis)
    {
        Vector3 tangent = Vector3.ProjectOnPlane(preferredAxis, normal);
        if (tangent.sqrMagnitude <= 1e-8f)
            tangent = Vector3.ProjectOnPlane(Vector3.up, normal);
        if (tangent.sqrMagnitude <= 1e-8f)
            tangent = Vector3.ProjectOnPlane(Vector3.right, normal);
        if (tangent.sqrMagnitude <= 1e-8f)
            tangent = Vector3.Cross(normal, Vector3.forward);
        return SafeNormalized(tangent, Vector3.right);
    }

    private static Vector3 SafeNormalized(Vector3 value, Vector3 fallback)
    {
        if (!Finite(value) || value.sqrMagnitude <= 1e-8f)
            return fallback;
        return value.normalized;
    }

    private void AppendProjectedGeometricGridForAxis(
        int normalAxis,
        Vector3 min,
        Vector3 max,
        float spacing,
        float castDistance,
        int maxSamplesPerLine,
        List<Vector3> meshVertices,
        List<int> meshTriangles,
        List<Vector3> lineVertices,
        List<Vector3> lineNormals,
        List<int> lineIndices)
    {
        int axisU = (normalAxis + 1) % 3;
        int axisV = (normalAxis + 2) % 3;
        AppendProjectedGeometricGridLines(normalAxis, axisU, axisV, min, max, spacing, castDistance, maxSamplesPerLine, meshVertices, meshTriangles, lineVertices, lineNormals, lineIndices);
        AppendProjectedGeometricGridLines(normalAxis, axisV, axisU, min, max, spacing, castDistance, maxSamplesPerLine, meshVertices, meshTriangles, lineVertices, lineNormals, lineIndices);
    }

    private void AppendProjectedGeometricGridLines(
        int normalAxis,
        int lineAxis,
        int fixedAxis,
        Vector3 min,
        Vector3 max,
        float spacing,
        float castDistance,
        int maxSamplesPerLine,
        List<Vector3> meshVertices,
        List<int> meshTriangles,
        List<Vector3> lineVertices,
        List<Vector3> lineNormals,
        List<int> lineIndices)
    {
        float fixedMin = Component(min, fixedAxis);
        float fixedMax = Component(max, fixedAxis);
        float lineMin = Component(min, lineAxis);
        float lineMax = Component(max, lineAxis);
        float normalMin = Component(min, normalAxis);
        float normalMax = Component(max, normalAxis);
        if (fixedMax <= fixedMin || lineMax <= lineMin || normalMax <= normalMin)
            return;

        int lineCount = Mathf.Clamp(Mathf.FloorToInt((fixedMax - fixedMin) / spacing) + 1, 1, 128);
        int sampleCount = Mathf.Clamp(Mathf.FloorToInt((lineMax - lineMin) / spacing) + 1, 2, maxSamplesPerLine);
        Vector3 direction = AxisVector(normalAxis);
        float normalCenter = (normalMin + normalMax) * 0.5f;
        float halfCast = castDistance * 0.5f;

        for (int lineIndex = 0; lineIndex < lineCount; lineIndex++)
        {
            float fixedValue = lineCount <= 1 ? (fixedMin + fixedMax) * 0.5f : Mathf.Lerp(fixedMin, fixedMax, lineIndex / (float)(lineCount - 1));
            int previousVertex = -1;
            Vector3 previousPoint = Vector3.zero;
            Vector3 previousNormal = Vector3.zero;

            for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
            {
                float lineValue = sampleCount <= 1 ? (lineMin + lineMax) * 0.5f : Mathf.Lerp(lineMin, lineMax, sampleIndex / (float)(sampleCount - 1));
                Vector3 gridPoint = Vector3.zero;
                SetComponent(ref gridPoint, normalAxis, normalCenter + halfCast);
                SetComponent(ref gridPoint, lineAxis, lineValue);
                SetComponent(ref gridPoint, fixedAxis, fixedValue);

                Ray ray = new Ray(gridPoint, -direction);
                if (!TryRaycastLocalTriangleMesh(ray, meshVertices, meshTriangles, castDistance, out Vector3 hitPoint, out Vector3 hitNormal))
                {
                    previousVertex = -1;
                    continue;
                }

                Vector3 displayPoint = hitPoint + hitNormal * Mathf.Max(0f, geometricSurfaceGridSurfaceOffsetMeters);
                int currentVertex = lineVertices.Count;
                lineVertices.Add(displayPoint);
                lineNormals.Add(hitNormal);
                if (previousVertex >= 0 &&
                    Vector3.Distance(previousPoint, displayPoint) <= spacing * 2.25f &&
                    Vector3.Dot(previousNormal, hitNormal) >= 0.45f)
                {
                    lineIndices.Add(previousVertex);
                    lineIndices.Add(currentVertex);
                }

                previousVertex = currentVertex;
                previousPoint = displayPoint;
                previousNormal = hitNormal;
            }
        }
    }

    private bool TryRaycastLocalTriangleMesh(Ray ray, List<Vector3> vertices, List<int> triangles, float maxDistance, out Vector3 hitPoint, out Vector3 hitNormal)
    {
        hitPoint = Vector3.zero;
        hitNormal = Vector3.zero;
        float bestDistance = maxDistance;
        bool hasHit = false;
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int ia = triangles[i];
            int ib = triangles[i + 1];
            int ic = triangles[i + 2];
            if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
                continue;

            Vector3 a = vertices[ia];
            Vector3 b = vertices[ib];
            Vector3 c = vertices[ic];
            if (!Finite(a) || !Finite(b) || !Finite(c))
                continue;

            if (!TryIntersectRayTriangle(ray, a, b, c, bestDistance, out float distance, out _, out _))
                continue;

            bestDistance = distance;
            hitPoint = ray.origin + ray.direction * distance;
            Vector3 normal = Vector3.Cross(b - a, c - a);
            hitNormal = normal.sqrMagnitude > 1e-8f ? normal.normalized : -ray.direction;
            if (Vector3.Dot(hitNormal, ray.direction) > 0f)
                hitNormal = -hitNormal;
            hasHit = true;
        }

        return hasHit;
    }

    private static float Component(Vector3 vector, int axis)
    {
        switch (axis)
        {
            case 0:
                return vector.x;
            case 1:
                return vector.y;
            default:
                return vector.z;
        }
    }

    private static void SetComponent(ref Vector3 vector, int axis, float value)
    {
        switch (axis)
        {
            case 0:
                vector.x = value;
                break;
            case 1:
                vector.y = value;
                break;
            default:
                vector.z = value;
                break;
        }
    }

    private static Vector3 AxisVector(int axis)
    {
        switch (axis)
        {
            case 0:
                return Vector3.right;
            case 1:
                return Vector3.up;
            default:
                return Vector3.forward;
        }
    }

    private void AddHeightSliceSamplePlaneFrames(List<Vector3> vertices, Vector3 baseOrigin, Vector3 sliceNormal, Vector3 planeAxisA, Vector3 planeAxisB, int sampleCount, List<Vector3> lineVertices, List<Vector3> lineNormals, List<int> lineIndices)
    {
        if (vertices == null || vertices.Count <= 0 || sliceNormal.sqrMagnitude <= 1e-8f || planeAxisA.sqrMagnitude <= 1e-8f || planeAxisB.sqrMagnitude <= 1e-8f)
            return;

        sliceNormal.Normalize();
        planeAxisA = Vector3.ProjectOnPlane(planeAxisA, sliceNormal);
        planeAxisB = Vector3.ProjectOnPlane(planeAxisB, sliceNormal);
        if (planeAxisA.sqrMagnitude <= 1e-8f || planeAxisB.sqrMagnitude <= 1e-8f)
            return;
        planeAxisA.Normalize();
        planeAxisB.Normalize();

        float minDistance = float.PositiveInfinity;
        float maxDistance = float.NegativeInfinity;
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 vertex = vertices[i];
            if (!Finite(vertex))
                continue;
            float distance = Vector3.Dot(vertex - baseOrigin, sliceNormal);
            minDistance = Mathf.Min(minDistance, distance);
            maxDistance = Mathf.Max(maxDistance, distance);
        }

        if (!float.IsFinite(minDistance) || !float.IsFinite(maxDistance))
            return;

        for (int i = 0; i < sampleCount; i++)
        {
            float t = sampleCount <= 1 ? 0.5f : i / (float)(sampleCount - 1);
            Vector3 planeOrigin = baseOrigin + sliceNormal * Mathf.Lerp(minDistance, maxDistance, t);
            AddHeightSlicePlaneFrame(vertices, planeOrigin, sliceNormal, planeAxisA, planeAxisB, lineVertices, lineNormals, lineIndices);
        }
    }

    private void AppendHeightSliceIntersectionRows(List<Vector3> vertices, List<int> triangles, Vector3 baseOrigin, Vector3 sliceNormal, int sliceCount, float epsilon, float maxSegmentLength, List<Vector3> lineVertices, List<Vector3> lineNormals, List<int> lineIndices)
    {
        if (vertices == null || triangles == null || sliceNormal.sqrMagnitude <= 1e-8f)
            return;

        sliceNormal.Normalize();
        float minPlaneDistance = float.PositiveInfinity;
        float maxPlaneDistance = float.NegativeInfinity;
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 vertex = vertices[i];
            if (!Finite(vertex))
                continue;
            float distance = Vector3.Dot(vertex - baseOrigin, sliceNormal);
            minPlaneDistance = Mathf.Min(minPlaneDistance, distance);
            maxPlaneDistance = Mathf.Max(maxPlaneDistance, distance);
        }

        if (!float.IsFinite(minPlaneDistance) || !float.IsFinite(maxPlaneDistance))
            return;

        List<Vector3> intersections = new List<Vector3>(3);
        Vector3 surfaceOffset = sliceNormal * Mathf.Max(0f, gridLineSurfaceOffsetMeters);
        for (int row = 0; row < sliceCount; row++)
        {
            float t = sliceCount <= 1 ? 0.5f : row / (float)(sliceCount - 1);
            Vector3 sliceOrigin = baseOrigin + sliceNormal * Mathf.Lerp(minPlaneDistance, maxPlaneDistance, t);

            for (int i = 0; i + 2 < triangles.Count; i += 3)
            {
                int ia = triangles[i];
                int ib = triangles[i + 1];
                int ic = triangles[i + 2];
                if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
                    continue;

                Vector3 a = vertices[ia];
                Vector3 b = vertices[ib];
                Vector3 c = vertices[ic];
                if (!Finite(a) || !Finite(b) || !Finite(c))
                    continue;

                intersections.Clear();
                float da = Vector3.Dot(a - sliceOrigin, sliceNormal);
                float db = Vector3.Dot(b - sliceOrigin, sliceNormal);
                float dc = Vector3.Dot(c - sliceOrigin, sliceNormal);
                TryAddPlaneIntersection(a, da, b, db, epsilon, intersections);
                TryAddPlaneIntersection(b, db, c, dc, epsilon, intersections);
                TryAddPlaneIntersection(c, dc, a, da, epsilon, intersections);

                if (intersections.Count != 2)
                    continue;

                Vector3 p0 = intersections[0] + surfaceOffset;
                Vector3 p1 = intersections[1] + surfaceOffset;
                if ((p1 - p0).magnitude > maxSegmentLength)
                    continue;

                AddDirectLineSegment(p0, p1, sliceNormal, lineVertices, lineNormals, lineIndices);
            }
        }
    }

    private bool TryGetHeightSlicePlaneLocal(List<Vector3> vertices, out Vector3 planeOrigin, out Vector3 planeNormal, out Vector3 planeRight, out Vector3 planeForward)
    {
        Transform surfaceTransform = ResolveDisplayLocalTransform();
        planeOrigin = Vector3.zero;
        planeNormal = Vector3.up;
        planeRight = Vector3.right;
        planeForward = Vector3.forward;

        if (heightSliceUseFrozenScreenCenterHeight && _hasFrozenHeadsetScreenCenterPoint)
        {
            Vector3 worldOrigin = _frozenHeadsetScreenCenterPoint;
            Vector3 worldNormal = Vector3.up;
            planeOrigin = surfaceTransform != null ? surfaceTransform.InverseTransformPoint(worldOrigin) : worldOrigin;
            planeNormal = surfaceTransform != null ? surfaceTransform.InverseTransformDirection(worldNormal) : worldNormal;
        }
        else
        {
            if (vertices == null || vertices.Count <= 0)
                return false;
            Vector3 sum = Vector3.zero;
            int count = 0;
            for (int i = 0; i < vertices.Count; i++)
            {
                if (!Finite(vertices[i]))
                    continue;
                sum += vertices[i];
                count++;
            }
            if (count <= 0)
                return false;
            planeOrigin = sum / count;
            planeNormal = surfaceTransform != null ? surfaceTransform.InverseTransformDirection(Vector3.up) : Vector3.up;
        }

        if (planeNormal.sqrMagnitude <= 1e-8f || !Finite(planeNormal))
            return false;
        planeNormal.Normalize();

        Vector3 rightWorld = _hasFrozenHeightSliceFrame ? _frozenHeightSliceRightWorld : Vector3.right;
        Vector3 forwardWorld = _hasFrozenHeightSliceFrame ? _frozenHeightSliceForwardWorld : Vector3.forward;
        Vector3 rightHint = surfaceTransform != null ? surfaceTransform.InverseTransformDirection(rightWorld) : rightWorld;
        planeRight = Vector3.ProjectOnPlane(rightHint, planeNormal);
        if (planeRight.sqrMagnitude <= 1e-8f)
        {
            Vector3 forwardHint = surfaceTransform != null ? surfaceTransform.InverseTransformDirection(forwardWorld) : forwardWorld;
            planeRight = Vector3.ProjectOnPlane(forwardHint, planeNormal);
        }
        if (planeRight.sqrMagnitude <= 1e-8f)
            return false;
        planeRight.Normalize();
        planeForward = Vector3.Cross(planeRight, planeNormal);
        if (planeForward.sqrMagnitude <= 1e-8f)
            return false;
        planeForward.Normalize();
        return true;
    }

    private void CaptureFrozenHeightSliceFrame(Transform origin)
    {
        _hasFrozenHeightSliceFrame = false;
        Vector3 forward = origin != null ? origin.forward : Vector3.forward;
        forward = Vector3.ProjectOnPlane(forward, Vector3.up);
        if (forward.sqrMagnitude <= 1e-8f || !Finite(forward))
            forward = Vector3.forward;
        forward.Normalize();

        Vector3 right = Vector3.Cross(Vector3.up, forward);
        if (right.sqrMagnitude <= 1e-8f || !Finite(right))
            right = Vector3.right;
        right.Normalize();

        _frozenHeightSliceForwardWorld = forward;
        _frozenHeightSliceRightWorld = right;
        _hasFrozenHeightSliceFrame = true;
    }

    private void TryAddPlaneIntersection(Vector3 from, float fromDistance, Vector3 to, float toDistance, float epsilon, List<Vector3> intersections)
    {
        bool fromOnPlane = Mathf.Abs(fromDistance) <= epsilon;
        bool toOnPlane = Mathf.Abs(toDistance) <= epsilon;
        if (fromOnPlane && toOnPlane)
            return;
        if (!fromOnPlane && !toOnPlane && Mathf.Sign(fromDistance) == Mathf.Sign(toDistance))
            return;

        Vector3 point;
        if (fromOnPlane)
            point = from;
        else if (toOnPlane)
            point = to;
        else
        {
            float denom = fromDistance - toDistance;
            if (Mathf.Abs(denom) <= 1e-8f)
                return;
            point = Vector3.Lerp(from, to, Mathf.Clamp01(fromDistance / denom));
        }

        AddUniquePlaneIntersection(intersections, point);
    }

    private static void AddUniquePlaneIntersection(List<Vector3> intersections, Vector3 point)
    {
        const float MinDistanceSqr = 0.000001f;
        for (int i = 0; i < intersections.Count; i++)
        {
            if ((intersections[i] - point).sqrMagnitude <= MinDistanceSqr)
                return;
        }

        intersections.Add(point);
    }

    private void AddHeightSlicePlaneFrame(List<Vector3> vertices, Vector3 planeOrigin, Vector3 planeNormal, Vector3 planeRight, Vector3 planeForward, List<Vector3> lineVertices, List<Vector3> lineNormals, List<int> lineIndices)
    {
        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 p = vertices[i];
            if (!Finite(p))
                continue;
            Vector3 offset = p - planeOrigin;
            float u = Vector3.Dot(offset, planeRight);
            float v = Vector3.Dot(offset, planeForward);
            minU = Mathf.Min(minU, u);
            maxU = Mathf.Max(maxU, u);
            minV = Mathf.Min(minV, v);
            maxV = Mathf.Max(maxV, v);
        }

        if (!float.IsFinite(minU) || !float.IsFinite(maxU) || !float.IsFinite(minV) || !float.IsFinite(maxV))
            return;

        Vector3 bias = planeNormal * Mathf.Max(0f, gridLineSurfaceOffsetMeters);
        Vector3 p00 = planeOrigin + planeRight * minU + planeForward * minV + bias;
        Vector3 p10 = planeOrigin + planeRight * maxU + planeForward * minV + bias;
        Vector3 p11 = planeOrigin + planeRight * maxU + planeForward * maxV + bias;
        Vector3 p01 = planeOrigin + planeRight * minU + planeForward * maxV + bias;
        AddDirectLineSegment(p00, p10, planeNormal, lineVertices, lineNormals, lineIndices);
        AddDirectLineSegment(p10, p11, planeNormal, lineVertices, lineNormals, lineIndices);
        AddDirectLineSegment(p11, p01, planeNormal, lineVertices, lineNormals, lineIndices);
        AddDirectLineSegment(p01, p00, planeNormal, lineVertices, lineNormals, lineIndices);
    }

    private bool TryBuildCandidateLineRibbonMesh(List<Vector3> lineVertices, List<Vector3> lineNormals, List<int> lineIndices, float lineWidthMeters, out List<Vector3> ribbonVertices, out List<int> ribbonTriangles)
    {
        ribbonVertices = new List<Vector3>(Mathf.Max(4, lineIndices != null ? lineIndices.Count * 2 : 4));
        ribbonTriangles = new List<int>(Mathf.Max(6, lineIndices != null ? lineIndices.Count * 3 : 6));
        float halfWidth = Mathf.Max(0.00025f, lineWidthMeters * 0.5f);
        if (lineIndices == null || lineVertices == null || lineIndices.Count < 2 || lineVertices.Count <= 0)
            return false;

        for (int i = 0; i + 1 < lineIndices.Count; i += 2)
        {
            int ia = lineIndices[i];
            int ib = lineIndices[i + 1];
            if (ia < 0 || ib < 0 || ia >= lineVertices.Count || ib >= lineVertices.Count)
                continue;

            Vector3 a = lineVertices[ia];
            Vector3 b = lineVertices[ib];
            Vector3 segment = b - a;
            float segmentLength = segment.magnitude;
            if (segmentLength <= 1e-6f)
                continue;

            Vector3 direction = segment / segmentLength;
            Vector3 segmentNormal = Vector3.zero;
            if (lineNormals != null)
            {
                if (ia < lineNormals.Count)
                    segmentNormal += lineNormals[ia];
                if (ib < lineNormals.Count)
                    segmentNormal += lineNormals[ib];
            }

            if (segmentNormal.sqrMagnitude <= 1e-8f)
                segmentNormal = Vector3.up;
            else
                segmentNormal.Normalize();

            Vector3 side = Vector3.Cross(segmentNormal, direction);
            if (side.sqrMagnitude <= 1e-8f)
                side = Vector3.Cross(direction, Vector3.forward);
            if (side.sqrMagnitude <= 1e-8f)
                side = Vector3.Cross(direction, Vector3.up);
            if (side.sqrMagnitude <= 1e-8f)
                continue;

            side = side.normalized * halfWidth;
            int baseIndex = ribbonVertices.Count;
            ribbonVertices.Add(a - side);
            ribbonVertices.Add(a + side);
            ribbonVertices.Add(b - side);
            ribbonVertices.Add(b + side);

            ribbonTriangles.Add(baseIndex + 0);
            ribbonTriangles.Add(baseIndex + 1);
            ribbonTriangles.Add(baseIndex + 3);
            ribbonTriangles.Add(baseIndex + 0);
            ribbonTriangles.Add(baseIndex + 3);
            ribbonTriangles.Add(baseIndex + 2);
        }

        return ribbonVertices.Count > 0 && ribbonTriangles.Count > 0;
    }

    private bool TryBuildTerrainConformedGridLinePositions(GridGroup group, bool[] valid, Vector3[] sourcePositions, Vector3[] sourceNormals, Vector3[] displayPositions)
    {
        if (group.columns <= 1 || group.rows <= 1 || valid == null || sourcePositions == null || sourceNormals == null || displayPositions == null)
            return false;

        int groupCellCount = group.columns * group.rows;
        int validCount = 0;
        Vector3 center = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || index >= sourcePositions.Length || !valid[index])
                    continue;

                center += sourcePositions[index];
                if (index < sourceNormals.Length && sourceNormals[index].sqrMagnitude > 1e-8f)
                    normalSum += sourceNormals[index].normalized;
                validCount++;
            }
        }

        if (validCount < 4 || validCount < Mathf.CeilToInt(groupCellCount * Mathf.Clamp01(rectifiedGridLineMinValidRatio)))
            return false;

        center /= validCount;
        Vector3 planeNormal = normalSum.sqrMagnitude > 1e-8f ? normalSum.normalized : Vector3.zero;

        Vector3 rightSum = Vector3.zero;
        Vector3 downSum = Vector3.zero;
        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns - 1; col++)
            {
                int a = rowStart + col;
                int b = a + 1;
                if (a >= 0 && b >= 0 && a < valid.Length && b < valid.Length && valid[a] && valid[b])
                    rightSum += sourcePositions[b] - sourcePositions[a];
            }
        }

        for (int row = 0; row < group.rows - 1; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                int a = rowStart + col;
                int b = nextRowStart + col;
                if (a >= 0 && b >= 0 && a < valid.Length && b < valid.Length && valid[a] && valid[b])
                    downSum += sourcePositions[b] - sourcePositions[a];
            }
        }

        if (planeNormal.sqrMagnitude <= 1e-8f && rightSum.sqrMagnitude > 1e-8f && downSum.sqrMagnitude > 1e-8f)
            planeNormal = Vector3.Cross(rightSum, downSum).normalized;
        if (planeNormal.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 right = Vector3.ProjectOnPlane(rightSum, planeNormal);
        if (right.sqrMagnitude <= 1e-8f)
            right = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (right.sqrMagnitude <= 1e-8f)
            right = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (right.sqrMagnitude <= 1e-8f)
            return false;
        right.Normalize();

        Vector3 up = Vector3.Cross(planeNormal, right);
        if (up.sqrMagnitude <= 1e-8f)
            return false;
        up.Normalize();
        if (downSum.sqrMagnitude > 1e-8f && Vector3.Dot(up, downSum.normalized) > 0f)
            up = -up;

        float[] colCoords = new float[group.columns];
        float[] rowCoords = new float[group.rows];
        int[] colCounts = new int[group.columns];
        int[] rowCounts = new int[group.rows];
        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || index >= sourcePositions.Length || !valid[index])
                    continue;

                Vector3 delta = sourcePositions[index] - center;
                colCoords[col] += Vector3.Dot(delta, right);
                rowCoords[row] += Vector3.Dot(delta, up);
                colCounts[col]++;
                rowCounts[row]++;
            }
        }

        if (!TryResolveGridAxisCoordinates(colCoords, colCounts, group.columns))
            return false;
        if (!TryResolveGridAxisCoordinates(rowCoords, rowCounts, group.rows))
            return false;

        float[] terrainHeights = new float[groupCellCount];
        bool[] heightValid = new bool[groupCellCount];
        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                int index = rowStart + col;
                if (index < 0 || index >= valid.Length || index >= displayPositions.Length || !valid[index])
                    continue;

                Vector3 regularPosition = center + right * colCoords[col] + up * rowCoords[row];
                terrainHeights[row * group.columns + col] = Vector3.Dot(sourcePositions[index] - regularPosition, planeNormal);
                heightValid[row * group.columns + col] = true;
            }
        }

        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                int index = rowStart + col;
                int localIndex = row * group.columns + col;
                if (index < 0 || index >= valid.Length || index >= displayPositions.Length || !valid[index] || !heightValid[localIndex])
                    continue;

                Vector3 regularPosition = center + right * colCoords[col] + up * rowCoords[row];
                displayPositions[index] = regularPosition + planeNormal * terrainHeights[localIndex];
            }
        }

        return true;
    }

    private static bool TryResolveGridAxisCoordinates(float[] coordinates, int[] counts, int length)
    {
        if (coordinates == null || counts == null || length <= 0 || coordinates.Length < length || counts.Length < length)
            return false;

        int firstValid = -1;
        int lastValid = -1;
        for (int i = 0; i < length; i++)
        {
            if (counts[i] <= 0)
                continue;

            coordinates[i] /= counts[i];
            if (firstValid < 0)
                firstValid = i;
            lastValid = i;
        }

        if (firstValid < 0 || lastValid < 0)
            return false;

        float fallbackStep = length > 1 ? (coordinates[lastValid] - coordinates[firstValid]) / Mathf.Max(1, lastValid - firstValid) : 0f;
        if (Mathf.Abs(fallbackStep) <= 1e-5f)
            fallbackStep = 0.02f;

        for (int i = 0; i < firstValid; i++)
            coordinates[i] = coordinates[firstValid] - fallbackStep * (firstValid - i);
        for (int i = lastValid + 1; i < length; i++)
            coordinates[i] = coordinates[lastValid] + fallbackStep * (i - lastValid);

        int segmentStart = firstValid;
        while (segmentStart < lastValid)
        {
            if (counts[segmentStart] <= 0)
            {
                segmentStart++;
                continue;
            }

            int segmentEnd = segmentStart + 1;
            while (segmentEnd <= lastValid && counts[segmentEnd] <= 0)
                segmentEnd++;
            if (segmentEnd > lastValid)
                break;

            int gap = segmentEnd - segmentStart;
            if (gap > 1)
            {
                float start = coordinates[segmentStart];
                float end = coordinates[segmentEnd];
                for (int i = 1; i < gap; i++)
                    coordinates[segmentStart + i] = Mathf.Lerp(start, end, i / (float)gap);
            }

            segmentStart = segmentEnd;
        }

        return true;
    }

    private float EstimateGridLineMaxSpan(GridGroup group, bool[] valid, Vector3[] displayPositions)
    {
        List<float> distances = new List<float>(group.columns * group.rows * 2);
        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns - 1; col++)
                AddGridLineSpanSample(rowStart + col, rowStart + col + 1, valid, displayPositions, distances);
        }

        for (int row = 0; row < group.rows - 1; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < group.columns; col++)
                AddGridLineSpanSample(rowStart + col, nextRowStart + col, valid, displayPositions, distances);
        }

        if (distances.Count <= 0)
            return Mathf.Max(0.01f, maxEdgeLengthMeters);

        distances.Sort();
        float median = distances[distances.Count / 2];
        return Mathf.Max(0.01f, median * Mathf.Max(1.05f, rectifiedGridLineMaxSpanMultiplier));
    }

    private static void AddGridLineSpanSample(int a, int b, bool[] valid, Vector3[] positions, List<float> distances)
    {
        if (a < 0 || b < 0 || valid == null || positions == null || distances == null || a >= valid.Length || b >= valid.Length || a >= positions.Length || b >= positions.Length)
            return;
        if (!valid[a] || !valid[b])
            return;

        float distance = Vector3.Distance(positions[a], positions[b]);
        if (!float.IsFinite(distance) || distance <= 1e-5f)
            return;

        distances.Add(distance);
    }

    private void BuildCompleteCellSupportedGridEdges(GridGroup group, bool[] valid, Vector3[] sourcePositions, Vector3[] sourceNormals, float maxSpan, bool[] horizontalEdges, bool[] verticalEdges, bool[] cells)
    {
        if (valid == null || horizontalEdges == null || verticalEdges == null || cells == null || group.columns < 2 || group.rows < 2)
            return;

        int cellStride = group.columns - 1;
        for (int row = 0; row < group.rows - 1; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < group.columns - 1; col++)
            {
                int i00 = rowStart + col;
                int i10 = i00 + 1;
                int i01 = nextRowStart + col;
                int i11 = i01 + 1;
                bool complete =
                    IsGridLineEdgeUsable(i00, i10, valid, sourcePositions, sourceNormals, maxSpan) &&
                    IsGridLineEdgeUsable(i01, i11, valid, sourcePositions, sourceNormals, maxSpan) &&
                    IsGridLineEdgeUsable(i00, i01, valid, sourcePositions, sourceNormals, maxSpan) &&
                    IsGridLineEdgeUsable(i10, i11, valid, sourcePositions, sourceNormals, maxSpan);
                if (!complete)
                    continue;

                cells[row * cellStride + col] = true;
            }
        }

        RebuildCompleteCellSupportedGridEdges(group, cells, horizontalEdges, verticalEdges);
    }

    private void PruneCompleteCellIslands(GridGroup group, bool[] cells)
    {
        if (cells == null || group.columns < 2 || group.rows < 2)
            return;

        int cellColumns = group.columns - 1;
        int cellRows = group.rows - 1;
        int cellCount = cellColumns * cellRows;
        int[] componentIds = new int[cellCount];
        for (int i = 0; i < componentIds.Length; i++)
            componentIds[i] = -1;

        List<int> stack = new List<int>(cellCount);
        List<int> componentSizes = new List<int>();
        for (int start = 0; start < cellCount; start++)
        {
            if (!cells[start] || componentIds[start] >= 0)
                continue;

            int componentId = componentSizes.Count;
            int size = 0;
            stack.Clear();
            stack.Add(start);
            componentIds[start] = componentId;

            while (stack.Count > 0)
            {
                int current = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                size++;

                int row = current / cellColumns;
                int col = current - row * cellColumns;
                TryVisitCompleteCellNeighbor(row, col - 1, cellColumns, cellRows, cells, componentIds, stack, componentId);
                TryVisitCompleteCellNeighbor(row, col + 1, cellColumns, cellRows, cells, componentIds, stack, componentId);
                TryVisitCompleteCellNeighbor(row - 1, col, cellColumns, cellRows, cells, componentIds, stack, componentId);
                TryVisitCompleteCellNeighbor(row + 1, col, cellColumns, cellRows, cells, componentIds, stack, componentId);
            }

            componentSizes.Add(size);
        }

        if (componentSizes.Count <= 1)
            return;

        bool[] keep = new bool[componentSizes.Count];
        int keepCount = Mathf.Clamp(gridLineKeepLargestCompleteCellIslands, 1, componentSizes.Count);
        int minSize = Mathf.Max(1, gridLineMinCompleteCellIslandCount);
        for (int keepIndex = 0; keepIndex < keepCount; keepIndex++)
        {
            int best = -1;
            int bestSize = -1;
            for (int component = 0; component < componentSizes.Count; component++)
            {
                if (keep[component] || componentSizes[component] < minSize || componentSizes[component] <= bestSize)
                    continue;

                best = component;
                bestSize = componentSizes[component];
            }

            if (best < 0)
                break;
            keep[best] = true;
        }

        for (int i = 0; i < cells.Length; i++)
        {
            int component = componentIds[i];
            cells[i] = component >= 0 && component < keep.Length && keep[component];
        }
    }

    private static void TryVisitCompleteCellNeighbor(int row, int col, int columns, int rows, bool[] cells, int[] componentIds, List<int> stack, int componentId)
    {
        if ((uint)row >= (uint)rows || (uint)col >= (uint)columns)
            return;

        int index = row * columns + col;
        if (!cells[index] || componentIds[index] >= 0)
            return;

        componentIds[index] = componentId;
        stack.Add(index);
    }

    private static void RebuildCompleteCellSupportedGridEdges(GridGroup group, bool[] cells, bool[] horizontalEdges, bool[] verticalEdges)
    {
        if (cells == null || horizontalEdges == null || verticalEdges == null || group.columns < 2 || group.rows < 2)
            return;

        Array.Clear(horizontalEdges, 0, horizontalEdges.Length);
        Array.Clear(verticalEdges, 0, verticalEdges.Length);

        int cellColumns = group.columns - 1;
        for (int row = 0; row < group.rows - 1; row++)
        {
            for (int col = 0; col < group.columns - 1; col++)
            {
                if (!cells[row * cellColumns + col])
                    continue;

                horizontalEdges[row * cellColumns + col] = true;
                horizontalEdges[(row + 1) * cellColumns + col] = true;
                verticalEdges[row * group.columns + col] = true;
                verticalEdges[row * group.columns + col + 1] = true;
            }
        }
    }

    private void AddGridOuterContourEdges(GridGroup group, bool[] valid, int[] vertexIndices, Vector3[] sourcePositions, float maxSpan)
    {
        if (valid == null || vertexIndices == null || group.columns < 2 || group.rows < 2)
            return;

        int cellColumns = group.columns - 1;
        int cellRows = group.rows - 1;
        bool[] cells = new bool[cellColumns * cellRows];
        for (int row = 0; row < cellRows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < cellColumns; col++)
            {
                int i00 = rowStart + col;
                int i10 = i00 + 1;
                int i01 = nextRowStart + col;
                int i11 = i01 + 1;
                cells[row * cellColumns + col] =
                    IsGridLineEdgeUsable(i00, i10, valid, sourcePositions, null, maxSpan) &&
                    IsGridLineEdgeUsable(i01, i11, valid, sourcePositions, null, maxSpan) &&
                    IsGridLineEdgeUsable(i00, i01, valid, sourcePositions, null, maxSpan) &&
                    IsGridLineEdgeUsable(i10, i11, valid, sourcePositions, null, maxSpan);
            }
        }

        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < cellColumns; col++)
            {
                bool cellAbove = row > 0 && cells[(row - 1) * cellColumns + col];
                bool cellBelow = row < cellRows && cells[row * cellColumns + col];
                if (cellAbove == cellBelow)
                    continue;

                AddGridLineEdge(rowStart + col, rowStart + col + 1, valid, vertexIndices, sourcePositions, null, maxSpan);
            }
        }

        for (int row = 0; row < cellRows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                bool cellLeft = col > 0 && cells[row * cellColumns + col - 1];
                bool cellRight = col < cellColumns && cells[row * cellColumns + col];
                if (cellLeft == cellRight)
                    continue;

                AddGridLineEdge(rowStart + col, nextRowStart + col, valid, vertexIndices, sourcePositions, null, maxSpan);
            }
        }
    }

    private void AddRectilinearFilledContourGridEdges(GridGroup group, bool[] valid, int[] vertexIndices, Vector3[] sourcePositions, float maxSpan)
    {
        if (valid == null || vertexIndices == null || group.columns < 2 || group.rows < 2)
            return;

        bool[] supportedHorizontalEdges = new bool[group.rows * (group.columns - 1)];
        bool[] supportedVerticalEdges = new bool[(group.rows - 1) * group.columns];
        bool[] supportedCells = new bool[(group.rows - 1) * (group.columns - 1)];
        BuildMeshBackedCompleteCellGridEdges(group, valid, supportedHorizontalEdges, supportedVerticalEdges, supportedCells);
        PruneCompleteCellIslands(group, supportedCells);
        RebuildCompleteCellSupportedGridEdges(group, supportedCells, supportedHorizontalEdges, supportedVerticalEdges);

        for (int row = 0; row < group.rows; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            for (int col = 0; col < group.columns - 1; col++)
            {
                if (!supportedHorizontalEdges[row * (group.columns - 1) + col])
                    continue;

                AddLineEdge(rowStart + col, rowStart + col + 1, valid, vertexIndices);
            }
        }

        for (int row = 0; row < group.rows - 1; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < group.columns; col++)
            {
                if (!supportedVerticalEdges[row * group.columns + col])
                    continue;

                AddLineEdge(rowStart + col, nextRowStart + col, valid, vertexIndices);
            }
        }
    }

    private static void BuildMeshBackedCompleteCellGridEdges(GridGroup group, bool[] valid, bool[] horizontalEdges, bool[] verticalEdges, bool[] cells)
    {
        if (valid == null || horizontalEdges == null || verticalEdges == null || cells == null || group.columns < 2 || group.rows < 2)
            return;

        int cellColumns = group.columns - 1;
        for (int row = 0; row < group.rows - 1; row++)
        {
            int rowStart = group.startIndex + row * group.columns;
            int nextRowStart = group.startIndex + (row + 1) * group.columns;
            for (int col = 0; col < cellColumns; col++)
            {
                int i00 = rowStart + col;
                int i10 = i00 + 1;
                int i01 = nextRowStart + col;
                int i11 = i01 + 1;
                cells[row * cellColumns + col] = valid[i00] && valid[i10] && valid[i01] && valid[i11];
            }
        }

        RebuildCompleteCellSupportedGridEdges(group, cells, horizontalEdges, verticalEdges);
    }

    private void AddGridLineEdge(int a, int b, bool[] valid, int[] vertexIndices, Vector3[] sourcePositions, Vector3[] sourceNormals, float maxSpan)
    {
        if (a < 0 || b < 0 || valid == null || vertexIndices == null || a >= valid.Length || b >= valid.Length || a >= vertexIndices.Length || b >= vertexIndices.Length)
            return;
        if (!IsGridLineEdgeUsable(a, b, valid, sourcePositions, sourceNormals, maxSpan))
            return;

        AddLineEdge(a, b, valid, vertexIndices);
    }

    private bool IsGridLineEdgeUsable(int a, int b, bool[] valid, Vector3[] sourcePositions, Vector3[] sourceNormals, float maxSpan)
    {
        if (a < 0 || b < 0 || valid == null || a >= valid.Length || b >= valid.Length)
            return false;
        if (!valid[a] || !valid[b])
            return false;

        if (gridLineRequireContinuousSurface && sourceNormals != null && a < sourceNormals.Length && b < sourceNormals.Length)
        {
            Vector3 normalA = sourceNormals[a];
            Vector3 normalB = sourceNormals[b];
            if (normalA.sqrMagnitude > 1e-6f && normalB.sqrMagnitude > 1e-6f)
            {
                float normalDot = Vector3.Dot(normalA.normalized, normalB.normalized);
                if (!float.IsFinite(normalDot) || normalDot < gridLineMinNeighborNormalDot)
                    return false;
            }
        }

        if (sourcePositions != null && a < sourcePositions.Length && b < sourcePositions.Length && maxSpan > 0f)
        {
            float sourceDistance = Vector3.Distance(sourcePositions[a], sourcePositions[b]);
            if (!float.IsFinite(sourceDistance) || sourceDistance > maxSpan)
                return false;
        }

        return true;
    }

    private void RemoveSingleGridLineSegments()
    {
        if (_lineIndices.Count <= 0)
            return;

        if (_lineIndices.Count < 4 || _verts.Count <= 0)
        {
            _lineIndices.Clear();
            return;
        }

        int[] degree = new int[_verts.Count];
        for (int i = 0; i + 1 < _lineIndices.Count; i += 2)
        {
            int a = _lineIndices[i];
            int b = _lineIndices[i + 1];
            if ((uint)a >= (uint)degree.Length || (uint)b >= (uint)degree.Length)
                continue;

            degree[a]++;
            degree[b]++;
        }

        int write = 0;
        int originalCount = _lineIndices.Count;
        for (int i = 0; i + 1 < originalCount; i += 2)
        {
            int a = _lineIndices[i];
            int b = _lineIndices[i + 1];
            if ((uint)a >= (uint)degree.Length || (uint)b >= (uint)degree.Length)
                continue;
            if (degree[a] <= 1 && degree[b] <= 1)
                continue;

            _lineIndices[write++] = a;
            _lineIndices[write++] = b;
        }

        if (write < originalCount)
            _lineIndices.RemoveRange(write, originalCount - write);
    }

    private void AddLineEdge(int a, int b, bool[] valid, int[] vertexIndices)
    {
        if (!valid[a] || !valid[b]) return;
        int va = vertexIndices[a], vb = vertexIndices[b];
        if (va < 0 || vb < 0) return;
        _lineIndices.Add(va);
        _lineIndices.Add(vb);
    }

    private void AddTriangulationEdge(int i00, int i10, int i01, int i11, bool[] valid, int[] vertexIndices, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        int count = (valid[i00] ? 1 : 0) + (valid[i10] ? 1 : 0) + (valid[i01] ? 1 : 0) + (valid[i11] ? 1 : 0);
        if (count < 3)
            return;

        if (count == 4)
        {
            bool diagA = EdgeOk(i00, i11, positions, normals, confidences);
            bool diagB = EdgeOk(i10, i01, positions, normals, confidences);
            if (diagA && diagB)
            {
                float scoreA = Vector3.Distance(positions[i00], positions[i11]) / Mathf.Max(0.01f, confidences[i00] + confidences[i11]);
                float scoreB = Vector3.Distance(positions[i10], positions[i01]) / Mathf.Max(0.01f, confidences[i10] + confidences[i01]);
                diagA = scoreA <= scoreB;
                diagB = !diagA;
            }

            if (diagA)
                AddLineEdge(i00, i11, valid, vertexIndices);
            else if (diagB)
                AddLineEdge(i10, i01, valid, vertexIndices);
            return;
        }

        if (valid[i00] && valid[i10] && valid[i11] && EdgeOk(i10, i11, positions, normals, confidences))
            AddLineEdge(i10, i11, valid, vertexIndices);
        if (valid[i00] && valid[i11] && valid[i01] && EdgeOk(i00, i11, positions, normals, confidences))
            AddLineEdge(i00, i11, valid, vertexIndices);
        if (valid[i00] && valid[i10] && valid[i01] && EdgeOk(i00, i10, positions, normals, confidences))
            AddLineEdge(i00, i10, valid, vertexIndices);
        if (valid[i10] && valid[i11] && valid[i01] && EdgeOk(i10, i01, positions, normals, confidences))
            AddLineEdge(i10, i01, valid, vertexIndices);
    }

    private void TryAddTriangle(int ia, int ib, int ic, int[] vertexIndices, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        int va = vertexIndices[ia], vb = vertexIndices[ib], vc = vertexIndices[ic]; if (va < 0 || vb < 0 || vc < 0) return;
        if (!EdgeOk(ia, ib, positions, normals, confidences) || !EdgeOk(ib, ic, positions, normals, confidences) || !EdgeOk(ic, ia, positions, normals, confidences)) return;
        if (Vector3.Cross(positions[ib] - positions[ia], positions[ic] - positions[ia]).sqrMagnitude <= 1e-8f) return;
        _tris.Add(va); _tris.Add(vb); _tris.Add(vc);
    }

    private bool EdgeOk(int a, int b, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        if (useIndexConnectivity && samplingMode == SamplingMode.RegularGrid) return true;
        float confidence = Mathf.Clamp01((confidences[a] + confidences[b]) * 0.5f), allowed = Mathf.Lerp(Mathf.Max(0.02f, maxEdgeLengthMeters * 0.65f), maxEdgeLengthMeters, confidence);
        if (Vector3.Distance(positions[a], positions[b]) > Mathf.Max(0.01f, allowed)) return false;
        if (normals[a].sqrMagnitude > 1e-6f && normals[b].sqrMagnitude > 1e-6f && Vector3.Dot(normals[a].normalized, normals[b].normalized) < minNeighborNormalDot) return false;
        return true;
    }

    private List<List<int>> BuildMeshNormalBuckets(List<Vector3> vertices, List<Vector3> vertexNormals, List<int> triangles)
    {
        List<List<int>> buckets = new List<List<int>>(7);
        for (int i = 0; i < 7; i++)
            buckets.Add(new List<int>(256));

        List<CandidateSurfaceInfo> candidateSurfaces = BuildCandidateSurfaces(vertices, vertexNormals, triangles);
        int candidateSurfaceCount = candidateSurfaces.Count;
        if (candidateSurfaceCount <= 0)
            return buckets;

        float minAxisDot = Mathf.Clamp(surfaceNormalAxisBucketMinDot, 0.5f, 0.999f);
        int minTriangleCount = Mathf.Max(1, surfaceLargeComponentMinTriangleCount);
        float keepRatio = Mathf.Clamp(surfaceLargeComponentKeepRatio, 0.05f, 1f);

        int[] candidateBuckets = new int[candidateSurfaceCount];
        int[] candidateTriangleCounts = new int[candidateSurfaceCount];
        int[] largestTrianglesPerBucket = new int[7];

        for (int surfaceIndex = 0; surfaceIndex < candidateSurfaceCount; surfaceIndex++)
        {
            int bucketIndex = GetSurfaceNormalBucketIndex(candidateSurfaces[surfaceIndex].averageNormal, minAxisDot);
            int triCount = candidateSurfaces[surfaceIndex].faceIndices.Count;
            candidateBuckets[surfaceIndex] = bucketIndex;
            candidateTriangleCounts[surfaceIndex] = triCount;
            largestTrianglesPerBucket[bucketIndex] = Mathf.Max(largestTrianglesPerBucket[bucketIndex], triCount);
        }

        for (int surfaceIndex = 0; surfaceIndex < candidateSurfaceCount; surfaceIndex++)
        {
            int bucketIndex = candidateBuckets[surfaceIndex];
            int triCount = candidateTriangleCounts[surfaceIndex];
            int bucketLargest = largestTrianglesPerBucket[bucketIndex];
            bool keep = triCount >= minTriangleCount || triCount >= Mathf.CeilToInt(bucketLargest * keepRatio);
            int outputBucket = keep ? bucketIndex : 6;

            AddCandidateSurfaceTriangles(candidateSurfaces[surfaceIndex].faceIndices, triangles, buckets[outputBucket]);
        }

        return buckets;
    }

    private List<List<int>> BuildTopCandidateSurfaceSubMeshes(List<int> triangles, List<CandidateSurfaceInfo> candidateSurfaces)
    {
        if (candidateSurfaces.Count <= 0)
            return new List<List<int>>();

        SortCandidateSurfacesForDisplay(candidateSurfaces, _verts, _tris);

        bool disableCountFilter = topCandidateSurfaceCount <= 0;
        bool disableMinTriangleFilter = topCandidateSurfaceMinTriangleCount <= 0;
        int keepCount = disableCountFilter ? candidateSurfaces.Count : Mathf.Max(1, topCandidateSurfaceCount);
        int minTriangleCount = disableMinTriangleFilter ? 0 : Mathf.Max(1, topCandidateSurfaceMinTriangleCount);
        List<List<int>> subMeshes = new List<List<int>>(Mathf.Max(1, keepCount));
        for (int i = 0; i < candidateSurfaces.Count && subMeshes.Count < keepCount; i++)
        {
            if (!disableMinTriangleFilter && candidateSurfaces[i].faceIndices.Count < minTriangleCount)
                continue;

            List<int> subMeshTriangles = new List<int>(candidateSurfaces[i].faceIndices.Count * 3);
            AddCandidateSurfaceTriangles(candidateSurfaces[i].faceIndices, triangles, subMeshTriangles);
            if (subMeshTriangles.Count > 0)
                subMeshes.Add(subMeshTriangles);
        }

        return subMeshes;
    }

    private bool TryBuildTopCandidateRegularGridMeshes(List<Vector3> vertices, List<int> triangles, List<CandidateSurfaceInfo> candidateSurfaces, bool[] sourceValid, Vector3[] sourcePositions, Vector3[] sourceNormals, float[] sourceConfidences, out List<Vector3> remeshVertices, out List<Vector3> remeshNormals, out List<List<int>> remeshSubMeshes, out List<Vector3> remeshLineVertices, out List<Vector3> remeshLineNormals, out List<int> remeshLineIndices)
    {
        List<Vector3> localVertices = new List<Vector3>();
        List<Vector3> localNormals = new List<Vector3>();
        List<List<int>> localSubMeshes = new List<List<int>>();
        List<Vector3> localLineVertices = new List<Vector3>();
        List<Vector3> localLineNormals = new List<Vector3>();
        List<int> localLineIndices = new List<int>();
        remeshVertices = localVertices;
        remeshNormals = localNormals;
        remeshSubMeshes = localSubMeshes;
        remeshLineVertices = localLineVertices;
        remeshLineNormals = localLineNormals;
        remeshLineIndices = localLineIndices;
        if (candidateSurfaces == null || candidateSurfaces.Count <= 0)
            return false;

        SortCandidateSurfacesForDisplay(candidateSurfaces, vertices, triangles);
        bool disableCountFilter = topCandidateSurfaceCount <= 0;
        bool disableMinTriangleFilter = topCandidateSurfaceMinTriangleCount <= 0;
        int keepCount = disableCountFilter ? candidateSurfaces.Count : Mathf.Max(1, topCandidateSurfaceCount);
        int minTriangleCount = disableMinTriangleFilter ? 0 : Mathf.Max(1, topCandidateSurfaceMinTriangleCount);
        for (int candidateIndex = 0; candidateIndex < candidateSurfaces.Count && localSubMeshes.Count < keepCount; candidateIndex++)
        {
            CandidateSurfaceInfo candidate = candidateSurfaces[candidateIndex];
            if (candidate.faceIndices == null || (!disableMinTriangleFilter && candidate.faceIndices.Count < minTriangleCount))
                continue;

            bool built;
            List<int> subMeshTriangles;
            if (largestCandidateUseOriginalGridTerrain && largestCandidateProjectRegularGridToMeshTerrain && !largestCandidateUseTriangularLattice)
            {
                built = TryBuildCandidateProjectedRennetTerrainMesh(
                    vertices,
                    triangles,
                    candidate,
                    localVertices,
                    localNormals,
                    out subMeshTriangles,
                    localLineVertices,
                    localLineNormals,
                    localLineIndices);
            }
            else
            {
                built = largestCandidateUseOriginalGridTerrain && !largestCandidateUseTriangularLattice
                    ? TryBuildCandidateTerrainGridMesh(vertices, triangles, candidate, sourceValid, sourcePositions, sourceNormals, sourceConfidences, localVertices, localNormals, out subMeshTriangles, localLineVertices, localLineNormals, localLineIndices)
                    : TryBuildCandidateRegularGridMesh(vertices, triangles, candidate, localVertices, localNormals, out subMeshTriangles, localLineVertices, localLineNormals, localLineIndices);
            }
            if (!built)
                continue;

            localSubMeshes.Add(subMeshTriangles ?? new List<int>());
        }

        bool hasFill = localVertices.Count > 0 && localSubMeshes.Count > 0;
        bool hasLines = localLineVertices.Count > 0 && localLineIndices.Count > 0;
        return hasFill || hasLines;
    }

    private void UpdateFocusedGridCellMask(bool[] valid, Vector3[] positions, List<Vector3> vertices, List<int> triangles, List<CandidateSurfaceInfo> candidateSurfaces)
    {
        _focusedGridOverlayState = default;
        if (!syncGridLinesToFocusedCandidate || !showGridLines || valid == null || positions == null || _cells.Count <= 0)
        {
            return;
        }

        if (!TryGetFocusedCandidateSurface(candidateSurfaces, vertices, triangles, out CandidateSurfaceInfo focusedCandidate))
        {
            return;
        }

        _focusedGridOverlayState = BuildFocusedGridOverlayState(vertices, triangles, focusedCandidate);
    }

    private bool TryGetFocusedCandidateSurface(List<CandidateSurfaceInfo> candidateSurfaces, List<Vector3> vertices, List<int> triangles, out CandidateSurfaceInfo focusedCandidate)
    {
        focusedCandidate = default;
        if (candidateSurfaces == null || candidateSurfaces.Count <= 0)
            return false;

        SortCandidateSurfacesForDisplay(candidateSurfaces, vertices, triangles);
        bool disableMinTriangleFilter = topCandidateSurfaceMinTriangleCount <= 0;
        int minTriangleCount = disableMinTriangleFilter ? 0 : Mathf.Max(1, topCandidateSurfaceMinTriangleCount);
        for (int i = 0; i < candidateSurfaces.Count; i++)
        {
            CandidateSurfaceInfo candidate = candidateSurfaces[i];
            if (candidate.faceIndices == null || candidate.faceIndices.Count <= 0)
                continue;
            if (!disableMinTriangleFilter && candidate.faceIndices.Count < minTriangleCount)
                continue;
            focusedCandidate = candidate;
            return true;
        }

        return false;
    }

    private FocusedGridOverlayState BuildFocusedGridOverlayState(List<Vector3> vertices, List<int> triangles, CandidateSurfaceInfo candidate)
    {
        if (vertices == null || triangles == null || candidate.faceIndices == null || candidate.faceIndices.Count <= 0)
            return default;

        Vector3 planeNormal = candidate.averageNormal.sqrMagnitude > 1e-8f ? candidate.averageNormal.normalized : Vector3.forward;
        Vector3 planeOrigin = candidate.averageCenter;
        Vector3 localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            return default;

        localUp.Normalize();
        Vector3 localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return default;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;

        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            if (triStart < 0 || triStart + 2 >= triangles.Count)
                continue;
            Vector3 pa = vertices[triangles[triStart]];
            Vector3 pb = vertices[triangles[triStart + 1]];
            Vector3 pc = vertices[triangles[triStart + 2]];
            Vector2 a = new Vector2(Vector3.Dot(pa - planeOrigin, localRight), Vector3.Dot(pa - planeOrigin, localUp));
            Vector2 b = new Vector2(Vector3.Dot(pb - planeOrigin, localRight), Vector3.Dot(pb - planeOrigin, localUp));
            Vector2 c = new Vector2(Vector3.Dot(pc - planeOrigin, localRight), Vector3.Dot(pc - planeOrigin, localUp));
            minU = Mathf.Min(minU, a.x, b.x, c.x);
            maxU = Mathf.Max(maxU, a.x, b.x, c.x);
            minV = Mathf.Min(minV, a.y, b.y, c.y);
            maxV = Mathf.Max(maxV, a.y, b.y, c.y);
        }

        if (!float.IsFinite(minU) || !float.IsFinite(maxU) || !float.IsFinite(minV) || !float.IsFinite(maxV))
            return default;

        float stepU = Mathf.Max(0.01f, largestCandidateGridCellSizeMeters);
        float stepV = stepU;
        GridGroup sourceGroup = _groups.Count > 0 ? _groups[0] : default;
        int sourceColumns = Mathf.Max(1, sourceGroup.columns);
        int sourceRows = Mathf.Max(1, sourceGroup.rows);
        int targetCellColumns = sourceColumns;
        int targetCellRows = sourceRows;

        float centerU = candidate.hasFocusAnchor ? candidate.focusAnchorLocalU : (minU + maxU) * 0.5f;
        float centerV = candidate.hasFocusAnchor ? candidate.focusAnchorLocalV : (minV + maxV) * 0.5f;
        float gridWidth = targetCellColumns * stepU;
        float gridHeight = targetCellRows * stepV;

        return new FocusedGridOverlayState
        {
            isValid = true,
            planeOrigin = planeOrigin,
            planeNormal = planeNormal,
            localRight = localRight,
            localUp = localUp,
            minU = centerU - gridWidth * 0.5f,
            minV = centerV - gridHeight * 0.5f,
            stepU = stepU,
            stepV = stepV,
            cellColumns = targetCellColumns,
            cellRows = targetCellRows
        };
    }

    private void UpdateCenterDebugMarkers(bool hasFocusedCandidate, CandidateSurfaceInfo focusedCandidate, bool hasPatchDebugCenter, Vector3 patchDebugCenter, Vector3 patchDebugNormal)
    {
        if (!showCenterDebugMarkers || !hasFocusedCandidate)
        {
            SetCenterDebugMarkersVisible(false);
            return;
        }

        if (!TryGetCenterDebugPositions(focusedCandidate, hasPatchDebugCenter, patchDebugCenter, patchDebugNormal, out Vector3 screenCenterWorld, out Vector3 gridCenterWorld, out Vector3 patchCenterWorld))
        {
            SetCenterDebugMarkersVisible(false);
            return;
        }

        EnsureCenterDebugObjects();
        bool hasAnyCenterMarker = showScreenCenterDebugMarker || showFocusedGridCenterDebugMarker || showPatchCenterDebugMarker;
        if (!hasAnyCenterMarker)
        {
            SetCenterDebugMarkersVisible(false);
            return;
        }

        if (_centerDebugRoot != null && !_centerDebugRoot.activeSelf)
            _centerDebugRoot.SetActive(true);

        if (showScreenCenterDebugMarker) UpdateCenterDebugMarker(0, screenCenterWorld, screenCenterDebugColor); else SetCenterDebugMarkerVisible(0, false);
        if (showFocusedGridCenterDebugMarker) UpdateCenterDebugMarker(1, gridCenterWorld, gridCenterDebugColor); else SetCenterDebugMarkerVisible(1, false);
        if (showPatchCenterDebugMarker) UpdateCenterDebugMarker(2, patchCenterWorld, patchCenterDebugColor); else SetCenterDebugMarkerVisible(2, false);
    }

    private bool TryGetCenterDebugPositions(CandidateSurfaceInfo focusedCandidate, bool hasPatchDebugCenter, Vector3 patchDebugCenter, Vector3 patchDebugNormal, out Vector3 screenCenterWorld, out Vector3 gridCenterWorld, out Vector3 patchCenterWorld)
    {
        screenCenterWorld = Vector3.zero;
        gridCenterWorld = Vector3.zero;
        patchCenterWorld = Vector3.zero;

        Transform displayTransform = ResolveDisplayLocalTransform();
        Vector3 patchCenter = hasPatchDebugCenter ? patchDebugCenter : focusedCandidate.averageCenter;
        Vector3 patchNormal = hasPatchDebugCenter && patchDebugNormal.sqrMagnitude > 1e-8f
            ? patchDebugNormal.normalized
            : (focusedCandidate.averageNormal.sqrMagnitude > 1e-8f ? focusedCandidate.averageNormal.normalized : Vector3.forward);
        patchCenterWorld = TransformDebugPointToWorld(displayTransform, patchCenter);
        Vector3 patchNormalWorld = TransformDebugDirectionToWorld(displayTransform, patchNormal);

        if (_focusedGridOverlayState.isValid)
        {
            FocusedGridOverlayState state = _focusedGridOverlayState;
            float centerU = state.minU + state.cellColumns * state.stepU * 0.5f;
            float centerV = state.minV + state.cellRows * state.stepV * 0.5f;
            Vector3 gridCenter = state.planeOrigin + state.localRight * centerU + state.localUp * centerV;
            gridCenterWorld = TransformDebugPointToWorld(displayTransform, gridCenter);

            Vector3 planeOriginWorld = TransformDebugPointToWorld(displayTransform, state.planeOrigin);
            Vector3 planeNormalWorld = TransformDebugDirectionToWorld(displayTransform, state.planeNormal);
            if (TryGetScreenCenterOnPlaneWorld(planeOriginWorld, planeNormalWorld, out Vector3 screenHitWorld))
                screenCenterWorld = screenHitWorld;
            else
                screenCenterWorld = gridCenterWorld;
        }
        else
        {
            gridCenterWorld = patchCenterWorld;
            screenCenterWorld = patchCenterWorld;
        }

        Vector3 surfaceOffset = patchNormalWorld.sqrMagnitude > 1e-8f
            ? patchNormalWorld.normalized * Mathf.Max(0f, centerDebugSurfaceOffsetMeters)
            : Vector3.zero;
        screenCenterWorld += surfaceOffset;
        gridCenterWorld += surfaceOffset;
        patchCenterWorld += surfaceOffset;
        return true;
    }

    private bool TryComputeDisplayedPatchCenter(List<Vector3> vertices, List<int> triangles, out Vector3 center, out Vector3 normal)
    {
        center = Vector3.zero;
        normal = Vector3.up;
        if (vertices == null || triangles == null || triangles.Count < 3)
            return false;

        Vector3 weightedCenterSum = Vector3.zero;
        Vector3 weightedNormalSum = Vector3.zero;
        float totalArea = 0f;
        int validTriangleCount = 0;

        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int ia = triangles[i];
            int ib = triangles[i + 1];
            int ic = triangles[i + 2];
            if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
                continue;

            Vector3 va = vertices[ia];
            Vector3 vb = vertices[ib];
            Vector3 vc = vertices[ic];
            Vector3 cross = Vector3.Cross(vb - va, vc - va);
            float doubleArea = cross.magnitude;
            if (doubleArea <= 1e-8f)
                continue;

            float area = doubleArea * 0.5f;
            weightedCenterSum += (va + vb + vc) / 3f * area;
            weightedNormalSum += cross;
            totalArea += area;
            validTriangleCount++;
        }

        if (validTriangleCount <= 0 || totalArea <= 1e-8f)
            return false;

        center = weightedCenterSum / totalArea;
        normal = weightedNormalSum.sqrMagnitude > 1e-8f ? weightedNormalSum.normalized : Vector3.up;
        return true;
    }

    private bool TryGetScreenCenterOnPlaneWorld(Vector3 planeOriginWorld, Vector3 planeNormalWorld, out Vector3 hitWorld)
    {
        hitWorld = Vector3.zero;
        Camera viewCamera = Camera.main;
        if (viewCamera == null)
            return false;

        Ray centerRay = viewCamera.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0f));
        float denominator = Vector3.Dot(planeNormalWorld, centerRay.direction);
        if (Mathf.Abs(denominator) <= 1e-6f)
            return false;

        float distance = Vector3.Dot(planeOriginWorld - centerRay.origin, planeNormalWorld) / denominator;
        if (distance <= 0f)
            return false;

        hitWorld = centerRay.origin + centerRay.direction * distance;
        return true;
    }

    private static Vector3 TransformDebugPointToWorld(Transform displayTransform, Vector3 point)
    {
        return displayTransform != null ? displayTransform.TransformPoint(point) : point;
    }

    private static Vector3 TransformDebugDirectionToWorld(Transform displayTransform, Vector3 direction)
    {
        if (direction.sqrMagnitude <= 1e-8f)
            return Vector3.forward;

        Vector3 worldDirection = displayTransform != null ? displayTransform.TransformDirection(direction) : direction;
        return worldDirection.sqrMagnitude > 1e-8f ? worldDirection.normalized : Vector3.forward;
    }

    private bool TryBuildCandidateRegularGridMesh(List<Vector3> vertices, List<int> triangles, CandidateSurfaceInfo candidate, List<Vector3> localVertices, List<Vector3> localNormals, out List<int> subMeshTriangles, List<Vector3> localLineVertices, List<Vector3> localLineNormals, List<int> localLineIndices)
    {
        if (largestCandidateUseTriangularLattice)
            return TryBuildCandidateTriangularLatticeMesh(vertices, triangles, candidate, localVertices, localNormals, out subMeshTriangles, localLineVertices, localLineNormals, localLineIndices);

        subMeshTriangles = new List<int>();
        if (candidate.faceIndices == null || candidate.faceIndices.Count <= 0)
            return false;

        int baseVertexCount = localVertices.Count;
        int baseNormalCount = localNormals.Count;
        int baseLineVertexCount = localLineVertices.Count;
        int baseLineIndexCount = localLineIndices.Count;

        Vector3 planeNormal;
        Vector3 planeOrigin;
        Vector3 localRight;
        Vector3 localUp;
        float stepU;
        float stepV;
        int columns;
        int rows;
        float minU;
        float minV;
        float maxU;
        float maxV;
        planeNormal = candidate.averageNormal.sqrMagnitude > 1e-8f ? candidate.averageNormal.normalized : Vector3.forward;
        planeOrigin = candidate.averageCenter;

        localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            return false;

        localUp.Normalize();
        localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return false;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;
        minU = float.PositiveInfinity;
        maxU = float.NegativeInfinity;
        minV = float.PositiveInfinity;
        maxV = float.NegativeInfinity;
        stepU = Mathf.Max(0.01f, largestCandidateGridCellSizeMeters);
        stepV = stepU;
        columns = 0;
        rows = 0;

        List<ProjectedTriangle2D> projectedTriangles = new List<ProjectedTriangle2D>(candidate.faceIndices.Count);
        float candidateMinU = float.PositiveInfinity;
        float candidateMaxU = float.NegativeInfinity;
        float candidateMinV = float.PositiveInfinity;
        float candidateMaxV = float.NegativeInfinity;
        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            Vector3 pa = vertices[triangles[triStart]];
            Vector3 pb = vertices[triangles[triStart + 1]];
            Vector3 pc = vertices[triangles[triStart + 2]];
            Vector2 a = new Vector2(Vector3.Dot(pa - planeOrigin, localRight), Vector3.Dot(pa - planeOrigin, localUp));
            Vector2 b = new Vector2(Vector3.Dot(pb - planeOrigin, localRight), Vector3.Dot(pb - planeOrigin, localUp));
            Vector2 c = new Vector2(Vector3.Dot(pc - planeOrigin, localRight), Vector3.Dot(pc - planeOrigin, localUp));
            projectedTriangles.Add(new ProjectedTriangle2D
            {
                a = a,
                b = b,
                c = c,
                minX = Mathf.Min(a.x, b.x, c.x),
                maxX = Mathf.Max(a.x, b.x, c.x),
                minY = Mathf.Min(a.y, b.y, c.y),
                maxY = Mathf.Max(a.y, b.y, c.y)
            });
            candidateMinU = Mathf.Min(candidateMinU, a.x, b.x, c.x);
            candidateMaxU = Mathf.Max(candidateMaxU, a.x, b.x, c.x);
            candidateMinV = Mathf.Min(candidateMinV, a.y, b.y, c.y);
            candidateMaxV = Mathf.Max(candidateMaxV, a.y, b.y, c.y);
        }

        if (!float.IsFinite(candidateMinU) || !float.IsFinite(candidateMinV) || !float.IsFinite(candidateMaxU) || !float.IsFinite(candidateMaxV))
            return false;

        minU = candidateMinU;
        minV = candidateMinV;
        maxU = candidateMaxU;
        maxV = candidateMaxV;
        columns = Mathf.Clamp(Mathf.CeilToInt((maxU - minU) / stepU), 1, Mathf.Max(4, largestCandidateGridMaxColumns));
        rows = Mathf.Clamp(Mathf.CeilToInt((maxV - minV) / stepV), 1, Mathf.Max(4, largestCandidateGridMaxRows));

        if (columns <= 0 || rows <= 0)
            return false;

        bool[] cellMask = new bool[columns * rows];
        List<Vector2>[] cellPolygons = new List<Vector2>[columns * rows];
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                float u0 = minU + x * stepU;
                float u1 = minU + (x + 1) * stepU;
                float v0 = minV + y * stepV;
                float v1 = minV + (y + 1) * stepV;
                if (TryBuildProjectedCellPolygon(projectedTriangles, u0, u1, v0, v1, out List<Vector2> polygon))
                {
                    int index = y * columns + x;
                    cellMask[index] = true;
                    cellPolygons[index] = polygon;
                }
            }
        }

        FilterCandidateGridCellMask(cellMask, columns, rows);
        int activeCellCount = 0;
        for (int i = 0; i < cellMask.Length; i++)
        {
            if (cellMask[i])
                activeCellCount++;
        }

        if (activeCellCount < Mathf.Max(0, largestCandidateGridMinCellCount))
            return false;

        Dictionary<Vector2Int, int> lineVertexIndices = new Dictionary<Vector2Int, int>();
        int EnsureLineVertex(int gridX, int gridY)
        {
            Vector2Int key = new Vector2Int(gridX, gridY);
            if (lineVertexIndices.TryGetValue(key, out int existing))
                return existing;

            float u = minU + gridX * stepU;
            float v = minV + gridY * stepV;
            Vector3 position = planeOrigin + localRight * u + localUp * v;
            if (gridLineSurfaceOffsetMeters > 0f)
                position += planeNormal * gridLineSurfaceOffsetMeters;
            int created = localLineVertices.Count;
            localLineVertices.Add(position);
            localLineNormals.Add(planeNormal);
            lineVertexIndices[key] = created;
            return created;
        }

        HashSet<EdgeKey> emittedEdges = new HashSet<EdgeKey>();
        void AddCellEdge(int ax, int ay, int bx, int by)
        {
            int ia = EnsureLineVertex(ax, ay);
            int ib = EnsureLineVertex(bx, by);
            EdgeKey key = new EdgeKey(ia, ib);
            if (!emittedEdges.Add(key))
                return;
            localLineIndices.Add(ia);
            localLineIndices.Add(ib);
        }

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                if (!cellMask[y * columns + x])
                    continue;

                float u0 = minU + x * stepU;
                float u1 = minU + (x + 1) * stepU;
                float v0 = minV + y * stepV;
                float v1 = minV + (y + 1) * stepV;

                int cellIndex = y * columns + x;
                List<Vector2> cellPolygon = cellPolygons[cellIndex];
                bool useFullCell = cellPolygon == null || IsProjectedCellNearlyFullRect(cellPolygon, u0, u1, v0, v1);
                if (useFullCell)
                {
                    Vector3 p00 = planeOrigin + localRight * u0 + localUp * v0;
                    Vector3 p10 = planeOrigin + localRight * u1 + localUp * v0;
                    Vector3 p01 = planeOrigin + localRight * u0 + localUp * v1;
                    Vector3 p11 = planeOrigin + localRight * u1 + localUp * v1;

                    int i00 = localVertices.Count;
                    localVertices.Add(p00);
                    localNormals.Add(planeNormal);
                    int i10 = localVertices.Count;
                    localVertices.Add(p10);
                    localNormals.Add(planeNormal);
                    int i01 = localVertices.Count;
                    localVertices.Add(p01);
                    localNormals.Add(planeNormal);
                    int i11 = localVertices.Count;
                    localVertices.Add(p11);
                    localNormals.Add(planeNormal);

                    subMeshTriangles.Add(i00);
                    subMeshTriangles.Add(i10);
                    subMeshTriangles.Add(i11);
                    subMeshTriangles.Add(i00);
                    subMeshTriangles.Add(i11);
                    subMeshTriangles.Add(i01);

                    AddCellEdge(x, y, x + 1, y);
                    AddCellEdge(x + 1, y, x + 1, y + 1);
                    AddCellEdge(x + 1, y + 1, x, y + 1);
                    AddCellEdge(x, y + 1, x, y);
                    if (largestCandidateGridShowCellDiagonals)
                        AddCellEdge(x, y, x + 1, y + 1);
                }
                else
                {
                    int polygonStart = localVertices.Count;
                    for (int i = 0; i < cellPolygon.Count; i++)
                    {
                        Vector2 point = cellPolygon[i];
                        Vector3 position = planeOrigin + localRight * point.x + localUp * point.y;
                        localVertices.Add(position);
                        localNormals.Add(planeNormal);
                    }

                    for (int i = 1; i < cellPolygon.Count - 1; i++)
                    {
                        subMeshTriangles.Add(polygonStart);
                        subMeshTriangles.Add(polygonStart + i);
                        subMeshTriangles.Add(polygonStart + i + 1);
                    }

                    for (int i = 0; i < cellPolygon.Count; i++)
                    {
                        Vector2 pointA = cellPolygon[i];
                        Vector2 pointB = cellPolygon[(i + 1) % cellPolygon.Count];
                        AddDirectLineSegment(
                            planeOrigin + localRight * pointA.x + localUp * pointA.y + planeNormal * gridLineSurfaceOffsetMeters,
                            planeOrigin + localRight * pointB.x + localUp * pointB.y + planeNormal * gridLineSurfaceOffsetMeters,
                            planeNormal,
                            localLineVertices,
                            localLineNormals,
                            localLineIndices);
                    }

                    if (largestCandidateGridShowCellDiagonals &&
                        TryClipSegmentToConvexPolygon(
                            new Vector2(u0, v0),
                            new Vector2(u1, v1),
                            cellPolygon,
                            out Vector2 clippedDiagonalA,
                            out Vector2 clippedDiagonalB))
                    {
                        AddDirectLineSegment(
                            planeOrigin + localRight * clippedDiagonalA.x + localUp * clippedDiagonalA.y + planeNormal * gridLineSurfaceOffsetMeters,
                            planeOrigin + localRight * clippedDiagonalB.x + localUp * clippedDiagonalB.y + planeNormal * gridLineSurfaceOffsetMeters,
                            planeNormal,
                            localLineVertices,
                            localLineNormals,
                            localLineIndices);
                    }
                }
            }
        }

        if (!PassesCandidateViewportFilter(localVertices, baseVertexCount))
        {
            RollBackCandidateRegularGridBuild(localVertices, localNormals, localLineVertices, localLineNormals, localLineIndices, baseVertexCount, baseNormalCount, baseLineVertexCount, baseLineIndexCount);
            subMeshTriangles.Clear();
            return false;
        }

        return subMeshTriangles.Count > 0;
    }

    private bool TryBuildCandidateProjectedRennetTerrainMesh(List<Vector3> vertices, List<int> triangles, CandidateSurfaceInfo candidate, List<Vector3> localVertices, List<Vector3> localNormals, out List<int> subMeshTriangles, List<Vector3> localLineVertices, List<Vector3> localLineNormals, List<int> localLineIndices)
    {
        subMeshTriangles = new List<int>();
        if (candidate.faceIndices == null || candidate.faceIndices.Count <= 0 || vertices == null || triangles == null)
            return false;

        int baseVertexCount = localVertices.Count;
        int baseNormalCount = localNormals.Count;
        int baseLineVertexCount = localLineVertices.Count;
        int baseLineIndexCount = localLineIndices.Count;

        if (!TryGetCandidatePlaneFrame(candidate, out Vector3 planeOrigin, out Vector3 planeNormal, out Vector3 localRight, out Vector3 localUp))
            return false;

        List<RennetTerrainTriangle> terrainTriangles = new List<RennetTerrainTriangle>(candidate.faceIndices.Count);
        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;

        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            if (triStart < 0 || triStart + 2 >= triangles.Count)
                continue;

            int ia = triangles[triStart];
            int ib = triangles[triStart + 1];
            int ic = triangles[triStart + 2];
            if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
                continue;

            Vector3 pa = vertices[ia];
            Vector3 pb = vertices[ib];
            Vector3 pc = vertices[ic];
            if (!Finite(pa) || !Finite(pb) || !Finite(pc))
                continue;

            Vector3 triangleNormal = Vector3.Cross(pb - pa, pc - pa);
            if (triangleNormal.sqrMagnitude <= 1e-10f)
                continue;

            triangleNormal.Normalize();
            if (Vector3.Dot(triangleNormal, planeNormal) < 0f)
                triangleNormal = -triangleNormal;

            Vector2 a = new Vector2(Vector3.Dot(pa - planeOrigin, localRight), Vector3.Dot(pa - planeOrigin, localUp));
            Vector2 b = new Vector2(Vector3.Dot(pb - planeOrigin, localRight), Vector3.Dot(pb - planeOrigin, localUp));
            Vector2 c = new Vector2(Vector3.Dot(pc - planeOrigin, localRight), Vector3.Dot(pc - planeOrigin, localUp));
            if (Mathf.Abs(SignedTriangleArea2D(a, b, c)) <= 1e-7f)
                continue;

            terrainTriangles.Add(new RennetTerrainTriangle
            {
                a = a,
                b = b,
                c = c,
                pa = pa,
                pb = pb,
                pc = pc,
                normal = triangleNormal,
                minX = Mathf.Min(a.x, b.x, c.x),
                maxX = Mathf.Max(a.x, b.x, c.x),
                minY = Mathf.Min(a.y, b.y, c.y),
                maxY = Mathf.Max(a.y, b.y, c.y)
            });

            minU = Mathf.Min(minU, a.x, b.x, c.x);
            maxU = Mathf.Max(maxU, a.x, b.x, c.x);
            minV = Mathf.Min(minV, a.y, b.y, c.y);
            maxV = Mathf.Max(maxV, a.y, b.y, c.y);
        }

        if (terrainTriangles.Count <= 0 ||
            !float.IsFinite(minU) || !float.IsFinite(maxU) ||
            !float.IsFinite(minV) || !float.IsFinite(maxV) ||
            maxU <= minU || maxV <= minV)
        {
            return false;
        }

        float step = Mathf.Max(0.01f, largestCandidateGridCellSizeMeters);
        int cellColumns = Mathf.Clamp(Mathf.CeilToInt((maxU - minU) / step), 1, Mathf.Max(4, largestCandidateGridMaxColumns));
        int cellRows = Mathf.Clamp(Mathf.CeilToInt((maxV - minV) / step), 1, Mathf.Max(4, largestCandidateGridMaxRows));
        int vertexColumns = cellColumns + 1;
        int vertexRows = cellRows + 1;
        int[] gridVertexIndices = new int[vertexColumns * vertexRows];
        Vector3[] gridNormals = new Vector3[vertexColumns * vertexRows];
        for (int i = 0; i < gridVertexIndices.Length; i++)
            gridVertexIndices[i] = -1;

        int GridIndex(int x, int y) => y * vertexColumns + x;
        for (int y = 0; y < vertexRows; y++)
        {
            float v = Mathf.Lerp(minV, maxV, y / (float)Mathf.Max(1, vertexRows - 1));
            for (int x = 0; x < vertexColumns; x++)
            {
                float u = Mathf.Lerp(minU, maxU, x / (float)Mathf.Max(1, vertexColumns - 1));
                if (!TrySampleProjectedRennetTerrain(terrainTriangles, new Vector2(u, v), planeOrigin, planeNormal, out Vector3 position, out Vector3 normal))
                    continue;

                int index = GridIndex(x, y);
                gridVertexIndices[index] = localVertices.Count;
                gridNormals[index] = normal;
                localVertices.Add(position);
                localNormals.Add(normal);
            }
        }

        bool[] cellMask = new bool[cellColumns * cellRows];
        int activeCellCount = 0;
        for (int y = 0; y < cellRows; y++)
        {
            for (int x = 0; x < cellColumns; x++)
            {
                int i00 = gridVertexIndices[GridIndex(x, y)];
                int i10 = gridVertexIndices[GridIndex(x + 1, y)];
                int i01 = gridVertexIndices[GridIndex(x, y + 1)];
                int i11 = gridVertexIndices[GridIndex(x + 1, y + 1)];
                if (i00 < 0 || i10 < 0 || i01 < 0 || i11 < 0)
                    continue;

                cellMask[y * cellColumns + x] = true;
            }
        }

        FilterCandidateGridCellMask(cellMask, cellColumns, cellRows);

        Dictionary<int, int> lineVertexIndices = new Dictionary<int, int>();
        int EnsureLineVertex(int gridIndex)
        {
            if (lineVertexIndices.TryGetValue(gridIndex, out int existing))
                return existing;

            int sourceIndex = gridVertexIndices[gridIndex];
            if (sourceIndex < 0 || sourceIndex >= localVertices.Count)
                return -1;

            Vector3 normal = gridNormals[gridIndex].sqrMagnitude > 1e-8f ? gridNormals[gridIndex].normalized : planeNormal;
            int created = localLineVertices.Count;
            localLineVertices.Add(localVertices[sourceIndex] + normal * Mathf.Max(0f, gridLineSurfaceOffsetMeters));
            localLineNormals.Add(normal);
            lineVertexIndices[gridIndex] = created;
            return created;
        }

        HashSet<EdgeKey> emittedEdges = new HashSet<EdgeKey>();
        void AddGridEdge(int ax, int ay, int bx, int by)
        {
            int ia = EnsureLineVertex(GridIndex(ax, ay));
            int ib = EnsureLineVertex(GridIndex(bx, by));
            if (ia < 0 || ib < 0)
                return;

            EdgeKey edge = new EdgeKey(ia, ib);
            if (!emittedEdges.Add(edge))
                return;

            localLineIndices.Add(ia);
            localLineIndices.Add(ib);
        }

        for (int y = 0; y < cellRows; y++)
        {
            for (int x = 0; x < cellColumns; x++)
            {
                if (!cellMask[y * cellColumns + x])
                    continue;

                int i00 = gridVertexIndices[GridIndex(x, y)];
                int i10 = gridVertexIndices[GridIndex(x + 1, y)];
                int i01 = gridVertexIndices[GridIndex(x, y + 1)];
                int i11 = gridVertexIndices[GridIndex(x + 1, y + 1)];
                if (i00 < 0 || i10 < 0 || i01 < 0 || i11 < 0)
                    continue;

                activeCellCount++;
                subMeshTriangles.Add(i00);
                subMeshTriangles.Add(i10);
                subMeshTriangles.Add(i11);
                subMeshTriangles.Add(i00);
                subMeshTriangles.Add(i11);
                subMeshTriangles.Add(i01);

                AddGridEdge(x, y, x + 1, y);
                AddGridEdge(x + 1, y, x + 1, y + 1);
                AddGridEdge(x + 1, y + 1, x, y + 1);
                AddGridEdge(x, y + 1, x, y);
                if (largestCandidateGridShowCellDiagonals)
                    AddGridEdge(x, y, x + 1, y + 1);
            }
        }

        if (activeCellCount < Mathf.Max(0, largestCandidateGridMinCellCount) ||
            (!PassesCandidateViewportFilter(localVertices.Count > baseVertexCount ? localVertices : localLineVertices, localVertices.Count > baseVertexCount ? baseVertexCount : baseLineVertexCount)))
        {
            RollBackCandidateRegularGridBuild(localVertices, localNormals, localLineVertices, localLineNormals, localLineIndices, baseVertexCount, baseNormalCount, baseLineVertexCount, baseLineIndexCount);
            subMeshTriangles.Clear();
            return false;
        }

        return subMeshTriangles.Count > 0 || localLineIndices.Count > baseLineIndexCount;
    }

    private bool TrySampleProjectedRennetTerrain(List<RennetTerrainTriangle> terrainTriangles, Vector2 uv, Vector3 planeOrigin, Vector3 planeNormal, out Vector3 position, out Vector3 normal)
    {
        position = Vector3.zero;
        normal = planeNormal;
        if (terrainTriangles == null || terrainTriangles.Count <= 0)
            return false;

        bool found = false;
        float bestPlaneDistance = float.PositiveInfinity;
        for (int i = 0; i < terrainTriangles.Count; i++)
        {
            RennetTerrainTriangle triangle = terrainTriangles[i];
            if (uv.x < triangle.minX || uv.x > triangle.maxX || uv.y < triangle.minY || uv.y > triangle.maxY)
                continue;

            if (!TryBarycentric2D(uv, triangle.a, triangle.b, triangle.c, out float wa, out float wb, out float wc))
                continue;

            Vector3 candidatePosition = triangle.pa * wa + triangle.pb * wb + triangle.pc * wc;
            float planeDistance = Mathf.Abs(Vector3.Dot(candidatePosition - planeOrigin, planeNormal));
            if (!float.IsFinite(planeDistance) || planeDistance >= bestPlaneDistance)
                continue;

            bestPlaneDistance = planeDistance;
            position = candidatePosition;
            normal = triangle.normal.sqrMagnitude > 1e-8f ? triangle.normal : planeNormal;
            found = true;
        }

        return found;
    }

    private static bool TryBarycentric2D(Vector2 p, Vector2 a, Vector2 b, Vector2 c, out float wa, out float wb, out float wc)
    {
        Vector2 v0 = b - a;
        Vector2 v1 = c - a;
        Vector2 v2 = p - a;
        float denom = v0.x * v1.y - v1.x * v0.y;
        if (Mathf.Abs(denom) <= 1e-8f)
        {
            wa = wb = wc = 0f;
            return false;
        }

        wb = (v2.x * v1.y - v1.x * v2.y) / denom;
        wc = (v0.x * v2.y - v2.x * v0.y) / denom;
        wa = 1f - wb - wc;
        const float epsilon = -0.0001f;
        return wa >= epsilon && wb >= epsilon && wc >= epsilon;
    }

    private bool TryBuildCandidateTerrainGridMesh(List<Vector3> vertices, List<int> triangles, CandidateSurfaceInfo candidate, bool[] sourceValid, Vector3[] sourcePositions, Vector3[] sourceNormals, float[] sourceConfidences, List<Vector3> localVertices, List<Vector3> localNormals, out List<int> subMeshTriangles, List<Vector3> localLineVertices, List<Vector3> localLineNormals, List<int> localLineIndices)
    {
        subMeshTriangles = new List<int>();
        if (candidate.faceIndices == null || candidate.faceIndices.Count <= 0 || sourceValid == null || sourcePositions == null || sourceNormals == null || _groups.Count <= 0)
            return false;

        int baseVertexCount = localVertices.Count;
        int baseNormalCount = localNormals.Count;
        int baseLineVertexCount = localLineVertices.Count;
        int baseLineIndexCount = localLineIndices.Count;

        Vector3 planeNormal = candidate.averageNormal.sqrMagnitude > 1e-8f ? candidate.averageNormal.normalized : Vector3.forward;
        Vector3 planeOrigin = candidate.averageCenter;
        Vector3 localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            return false;

        localUp.Normalize();
        Vector3 localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return false;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;

        List<ProjectedTriangle2D> projectedTriangles = new List<ProjectedTriangle2D>(candidate.faceIndices.Count);
        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            if (triStart < 0 || triStart + 2 >= triangles.Count)
                continue;

            int ia = triangles[triStart];
            int ib = triangles[triStart + 1];
            int ic = triangles[triStart + 2];
            if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
                continue;

            Vector3 pa = vertices[ia];
            Vector3 pb = vertices[ib];
            Vector3 pc = vertices[ic];
            Vector2 a = new Vector2(Vector3.Dot(pa - planeOrigin, localRight), Vector3.Dot(pa - planeOrigin, localUp));
            Vector2 b = new Vector2(Vector3.Dot(pb - planeOrigin, localRight), Vector3.Dot(pb - planeOrigin, localUp));
            Vector2 c = new Vector2(Vector3.Dot(pc - planeOrigin, localRight), Vector3.Dot(pc - planeOrigin, localUp));
            projectedTriangles.Add(new ProjectedTriangle2D
            {
                a = a,
                b = b,
                c = c,
                minX = Mathf.Min(a.x, b.x, c.x),
                maxX = Mathf.Max(a.x, b.x, c.x),
                minY = Mathf.Min(a.y, b.y, c.y),
                maxY = Mathf.Max(a.y, b.y, c.y)
            });
        }

        if (projectedTriangles.Count <= 0)
            return false;

        Transform surfaceTransform = ResolveDisplayLocalTransform();
        Vector3 ToLocalPoint(Vector3 point) => surfaceTransform != null ? surfaceTransform.InverseTransformPoint(point) : point;
        Vector3 ToLocalNormal(Vector3 normal)
        {
            Vector3 localNormal = surfaceTransform != null ? surfaceTransform.InverseTransformDirection(normal) : normal;
            return localNormal.sqrMagnitude > 1e-8f ? localNormal.normalized : planeNormal;
        }

        bool PointInsideCandidate(Vector3 localPoint)
        {
            Vector2 sample = new Vector2(Vector3.Dot(localPoint - planeOrigin, localRight), Vector3.Dot(localPoint - planeOrigin, localUp));
            for (int i = 0; i < projectedTriangles.Count; i++)
            {
                ProjectedTriangle2D tri = projectedTriangles[i];
                if (sample.x < tri.minX || sample.x > tri.maxX || sample.y < tri.minY || sample.y > tri.maxY)
                    continue;
                if (PointInProjectedTriangle(sample, tri))
                    return true;
            }

            return false;
        }

        bool IsSourceValid(int index) => index >= 0 && index < sourceValid.Length && index < sourcePositions.Length && index < sourceNormals.Length && sourceValid[index];

        Dictionary<int, int> lineVertexIndices = new Dictionary<int, int>();
        int EnsureLineVertex(int sourceIndex)
        {
            if (lineVertexIndices.TryGetValue(sourceIndex, out int existing))
                return existing;

            Vector3 localNormal = ToLocalNormal(sourceNormals[sourceIndex]);
            Vector3 localPosition = ToLocalPoint(sourcePositions[sourceIndex]);
            if (gridLineSurfaceOffsetMeters > 0f)
                localPosition += localNormal * gridLineSurfaceOffsetMeters;

            int created = localLineVertices.Count;
            localLineVertices.Add(localPosition);
            localLineNormals.Add(localNormal);
            lineVertexIndices[sourceIndex] = created;
            return created;
        }

        HashSet<EdgeKey> emittedEdges = new HashSet<EdgeKey>();
        void AddTerrainEdge(int sourceA, int sourceB)
        {
            if (!IsSourceValid(sourceA) || !IsSourceValid(sourceB))
                return;

            int ia = EnsureLineVertex(sourceA);
            int ib = EnsureLineVertex(sourceB);
            EdgeKey key = new EdgeKey(ia, ib);
            if (!emittedEdges.Add(key))
                return;

            localLineIndices.Add(ia);
            localLineIndices.Add(ib);
        }

        int activeCellCount = 0;
        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            if (group.columns <= 1 || group.rows <= 1)
                continue;

            int columns = group.columns - 1;
            int rows = group.rows - 1;
            bool[] cellMask = new bool[columns * rows];
            for (int row = 0; row < rows; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                int nextRowStart = group.startIndex + (row + 1) * group.columns;
                for (int col = 0; col < columns; col++)
                {
                    int i00 = rowStart + col;
                    int i10 = rowStart + col + 1;
                    int i01 = nextRowStart + col;
                    int i11 = nextRowStart + col + 1;
                    int validCount = (IsSourceValid(i00) ? 1 : 0) + (IsSourceValid(i10) ? 1 : 0) + (IsSourceValid(i01) ? 1 : 0) + (IsSourceValid(i11) ? 1 : 0);
                    if (validCount < 3)
                        continue;

                    Vector3 center = Vector3.zero;
                    if (IsSourceValid(i00)) center += ToLocalPoint(sourcePositions[i00]);
                    if (IsSourceValid(i10)) center += ToLocalPoint(sourcePositions[i10]);
                    if (IsSourceValid(i01)) center += ToLocalPoint(sourcePositions[i01]);
                    if (IsSourceValid(i11)) center += ToLocalPoint(sourcePositions[i11]);
                    center /= validCount;

                    if (PointInsideCandidate(center))
                        cellMask[row * columns + col] = true;
                }
            }

            FilterCandidateGridCellMask(cellMask, columns, rows);
            for (int row = 0; row < rows; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                int nextRowStart = group.startIndex + (row + 1) * group.columns;
                for (int col = 0; col < columns; col++)
                {
                    if (!cellMask[row * columns + col])
                        continue;

                    int i00 = rowStart + col;
                    int i10 = rowStart + col + 1;
                    int i01 = nextRowStart + col;
                    int i11 = nextRowStart + col + 1;
                    activeCellCount++;

                    AddTerrainEdge(i00, i10);
                    AddTerrainEdge(i10, i11);
                    AddTerrainEdge(i11, i01);
                    AddTerrainEdge(i01, i00);
                    if (largestCandidateGridShowCellDiagonals)
                        AddTerrainEdge(i00, i11);

                    if (IsSourceValid(i00) && IsSourceValid(i10) && IsSourceValid(i11))
                    {
                        int baseIndex = localVertices.Count;
                        localVertices.Add(ToLocalPoint(sourcePositions[i00]));
                        localVertices.Add(ToLocalPoint(sourcePositions[i10]));
                        localVertices.Add(ToLocalPoint(sourcePositions[i11]));
                        localNormals.Add(ToLocalNormal(sourceNormals[i00]));
                        localNormals.Add(ToLocalNormal(sourceNormals[i10]));
                        localNormals.Add(ToLocalNormal(sourceNormals[i11]));
                        subMeshTriangles.Add(baseIndex);
                        subMeshTriangles.Add(baseIndex + 1);
                        subMeshTriangles.Add(baseIndex + 2);
                    }

                    if (IsSourceValid(i00) && IsSourceValid(i11) && IsSourceValid(i01))
                    {
                        int baseIndex = localVertices.Count;
                        localVertices.Add(ToLocalPoint(sourcePositions[i00]));
                        localVertices.Add(ToLocalPoint(sourcePositions[i11]));
                        localVertices.Add(ToLocalPoint(sourcePositions[i01]));
                        localNormals.Add(ToLocalNormal(sourceNormals[i00]));
                        localNormals.Add(ToLocalNormal(sourceNormals[i11]));
                        localNormals.Add(ToLocalNormal(sourceNormals[i01]));
                        subMeshTriangles.Add(baseIndex);
                        subMeshTriangles.Add(baseIndex + 1);
                        subMeshTriangles.Add(baseIndex + 2);
                    }
                }
            }
        }

        if (activeCellCount < Mathf.Max(0, largestCandidateGridMinCellCount) ||
            (!PassesCandidateViewportFilter(localVertices.Count > baseVertexCount ? localVertices : localLineVertices, localVertices.Count > baseVertexCount ? baseVertexCount : baseLineVertexCount)))
        {
            RollBackCandidateRegularGridBuild(localVertices, localNormals, localLineVertices, localLineNormals, localLineIndices, baseVertexCount, baseNormalCount, baseLineVertexCount, baseLineIndexCount);
            subMeshTriangles.Clear();
            return false;
        }

        return subMeshTriangles.Count > 0 || localLineIndices.Count > baseLineIndexCount;
    }

    private bool TryBuildCandidateTriangularLatticeMesh(List<Vector3> vertices, List<int> triangles, CandidateSurfaceInfo candidate, List<Vector3> localVertices, List<Vector3> localNormals, out List<int> subMeshTriangles, List<Vector3> localLineVertices, List<Vector3> localLineNormals, List<int> localLineIndices)
    {
        subMeshTriangles = new List<int>();
        if (candidate.faceIndices == null || candidate.faceIndices.Count <= 0)
            return false;

        int baseLineVertexCount = localLineVertices.Count;
        int baseLineIndexCount = localLineIndices.Count;

        Vector3 planeNormal = candidate.averageNormal.sqrMagnitude > 1e-8f ? candidate.averageNormal.normalized : Vector3.forward;
        Vector3 planeOrigin = candidate.averageCenter;

        Vector3 localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            return false;

        localUp.Normalize();
        Vector3 localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return false;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;

        List<ProjectedTriangle2D> projectedTriangles = new List<ProjectedTriangle2D>(candidate.faceIndices.Count);
        float minU = float.PositiveInfinity, maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity, maxV = float.NegativeInfinity;
        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            Vector3 pa = vertices[triangles[triStart]];
            Vector3 pb = vertices[triangles[triStart + 1]];
            Vector3 pc = vertices[triangles[triStart + 2]];
            Vector2 a = new Vector2(Vector3.Dot(pa - planeOrigin, localRight), Vector3.Dot(pa - planeOrigin, localUp));
            Vector2 b = new Vector2(Vector3.Dot(pb - planeOrigin, localRight), Vector3.Dot(pb - planeOrigin, localUp));
            Vector2 c = new Vector2(Vector3.Dot(pc - planeOrigin, localRight), Vector3.Dot(pc - planeOrigin, localUp));
            projectedTriangles.Add(new ProjectedTriangle2D
            {
                a = a,
                b = b,
                c = c,
                minX = Mathf.Min(a.x, b.x, c.x),
                maxX = Mathf.Max(a.x, b.x, c.x),
                minY = Mathf.Min(a.y, b.y, c.y),
                maxY = Mathf.Max(a.y, b.y, c.y)
            });
            minU = Mathf.Min(minU, a.x, b.x, c.x);
            maxU = Mathf.Max(maxU, a.x, b.x, c.x);
            minV = Mathf.Min(minV, a.y, b.y, c.y);
            maxV = Mathf.Max(maxV, a.y, b.y, c.y);
        }

        if (!float.IsFinite(minU) || !float.IsFinite(minV) || !float.IsFinite(maxU) || !float.IsFinite(maxV))
            return false;

        float step = Mathf.Max(0.01f, largestCandidateGridCellSizeMeters);
        float rowSpacing = step * 0.8660254f;
        int columns = Mathf.Clamp(Mathf.CeilToInt((maxU - minU) / step) + 3, 2, Mathf.Max(4, largestCandidateGridMaxColumns));
        int rows = Mathf.Clamp(Mathf.CeilToInt((maxV - minV) / rowSpacing) + 3, 2, Mathf.Max(4, largestCandidateGridMaxRows));
        if (columns <= 1 || rows <= 1)
            return false;

        bool[] pointMask = new bool[columns * rows];
        int IndexOf(int x, int y) => y * columns + x;
        bool IsActivePoint(int x, int y) => x >= 0 && x < columns && y >= 0 && y < rows && pointMask[IndexOf(x, y)];

        for (int y = 0; y < rows; y++)
        {
            float rowOffset = (y & 1) != 0 ? step * 0.5f : 0f;
            for (int x = 0; x < columns; x++)
            {
                Vector2 sample = new Vector2(minU + x * step + rowOffset, minV + y * rowSpacing);
                for (int triIndex = 0; triIndex < projectedTriangles.Count; triIndex++)
                {
                    if (PointInProjectedTriangle(sample, projectedTriangles[triIndex]))
                    {
                        pointMask[IndexOf(x, y)] = true;
                        break;
                    }
                }
            }
        }

        bool[] filteredMask = (bool[])pointMask.Clone();
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                int index = IndexOf(x, y);
                if (!pointMask[index])
                    continue;

                int neighborCount = 0;
                if (IsActivePoint(x - 1, y)) neighborCount++;
                if (IsActivePoint(x + 1, y)) neighborCount++;
                if ((y & 1) == 0)
                {
                    if (IsActivePoint(x - 1, y - 1)) neighborCount++;
                    if (IsActivePoint(x, y - 1)) neighborCount++;
                    if (IsActivePoint(x - 1, y + 1)) neighborCount++;
                    if (IsActivePoint(x, y + 1)) neighborCount++;
                }
                else
                {
                    if (IsActivePoint(x, y - 1)) neighborCount++;
                    if (IsActivePoint(x + 1, y - 1)) neighborCount++;
                    if (IsActivePoint(x, y + 1)) neighborCount++;
                    if (IsActivePoint(x + 1, y + 1)) neighborCount++;
                }

                if (neighborCount < 2)
                    filteredMask[index] = false;
            }
        }
        pointMask = filteredMask;

        int activePointCount = 0;
        for (int i = 0; i < pointMask.Length; i++)
        {
            if (pointMask[i])
                activePointCount++;
        }
        if (activePointCount < Mathf.Max(4, largestCandidateGridMinCellCount))
            return false;

        Dictionary<Vector2Int, int> pointVertexIndices = new Dictionary<Vector2Int, int>();
        int EnsurePointVertex(int gridX, int gridY)
        {
            Vector2Int key = new Vector2Int(gridX, gridY);
            if (pointVertexIndices.TryGetValue(key, out int existing))
                return existing;

            float rowOffset = (gridY & 1) != 0 ? step * 0.5f : 0f;
            float u = minU + gridX * step + rowOffset;
            float v = minV + gridY * rowSpacing;
            Vector3 position = planeOrigin + localRight * u + localUp * v;
            if (gridLineSurfaceOffsetMeters > 0f)
                position += planeNormal * gridLineSurfaceOffsetMeters;
            int created = localLineVertices.Count;
            localLineVertices.Add(position);
            localLineNormals.Add(planeNormal);
            pointVertexIndices[key] = created;
            return created;
        }

        HashSet<EdgeKey> emittedEdges = new HashSet<EdgeKey>();
        void AddLatticeEdge(int ax, int ay, int bx, int by)
        {
            if (!IsActivePoint(ax, ay) || !IsActivePoint(bx, by))
                return;

            int ia = EnsurePointVertex(ax, ay);
            int ib = EnsurePointVertex(bx, by);
            EdgeKey key = new EdgeKey(ia, ib);
            if (!emittedEdges.Add(key))
                return;
            localLineIndices.Add(ia);
            localLineIndices.Add(ib);
        }

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                if (!IsActivePoint(x, y))
                    continue;

                AddLatticeEdge(x, y, x + 1, y);
                if ((y & 1) == 0)
                {
                    AddLatticeEdge(x, y, x, y + 1);
                    AddLatticeEdge(x, y, x - 1, y + 1);
                }
                else
                {
                    AddLatticeEdge(x, y, x, y + 1);
                    AddLatticeEdge(x, y, x + 1, y + 1);
                }
            }
        }

        if (!PassesCandidateViewportFilter(localLineVertices, baseLineVertexCount))
        {
            RollBackCandidateRegularGridBuild(localVertices, localNormals, localLineVertices, localLineNormals, localLineIndices, localVertices.Count, localNormals.Count, baseLineVertexCount, baseLineIndexCount);
            return false;
        }

        return localLineIndices.Count > baseLineIndexCount;
    }

    private bool PassesCandidateViewportFilter(List<Vector3> vertices, int startVertexIndex)
    {
        if (!largestCandidateGridUseViewportFilter)
            return true;

        Camera viewCamera = Camera.main;
        if (viewCamera == null || vertices == null || startVertexIndex < 0 || startVertexIndex >= vertices.Count)
            return true;

        float minX = float.PositiveInfinity;
        float minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float maxY = float.NegativeInfinity;
        bool hasVisiblePoint = false;

        for (int i = startVertexIndex; i < vertices.Count; i++)
        {
            Vector3 viewport = viewCamera.WorldToViewportPoint(vertices[i]);
            if (viewport.z <= 0f)
                continue;

            hasVisiblePoint = true;
            if (viewport.x < minX) minX = viewport.x;
            if (viewport.x > maxX) maxX = viewport.x;
            if (viewport.y < minY) minY = viewport.y;
            if (viewport.y > maxY) maxY = viewport.y;
        }

        if (!hasVisiblePoint)
            return false;

        float width = Mathf.Max(0f, maxX - minX);
        float height = Mathf.Max(0f, maxY - minY);
        float maxSpan = Mathf.Max(width, height);
        float area = width * height;
        return maxSpan >= largestCandidateGridMinViewportMaxSpan || area >= largestCandidateGridMinViewportArea;
    }

    private static void RollBackCandidateRegularGridBuild(List<Vector3> localVertices, List<Vector3> localNormals, List<Vector3> localLineVertices, List<Vector3> localLineNormals, List<int> localLineIndices, int baseVertexCount, int baseNormalCount, int baseLineVertexCount, int baseLineIndexCount)
    {
        if (localVertices.Count > baseVertexCount)
            localVertices.RemoveRange(baseVertexCount, localVertices.Count - baseVertexCount);
        if (localNormals.Count > baseNormalCount)
            localNormals.RemoveRange(baseNormalCount, localNormals.Count - baseNormalCount);
        if (localLineVertices.Count > baseLineVertexCount)
            localLineVertices.RemoveRange(baseLineVertexCount, localLineVertices.Count - baseLineVertexCount);
        if (localLineNormals.Count > baseLineVertexCount)
            localLineNormals.RemoveRange(baseLineVertexCount, localLineNormals.Count - baseLineVertexCount);
        if (localLineIndices.Count > baseLineIndexCount)
            localLineIndices.RemoveRange(baseLineIndexCount, localLineIndices.Count - baseLineIndexCount);
    }

    private void FilterCandidateGridCellMask(bool[] cellMask, int columns, int rows)
    {
        if (cellMask == null || cellMask.Length != columns * rows || columns <= 0 || rows <= 0)
            return;

        int minNeighborCount = Mathf.Clamp(largestCandidateGridMinNeighborCount, 0, 8);
        if (minNeighborCount > 0)
        {
            bool[] filtered = (bool[])cellMask.Clone();
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    int index = y * columns + x;
                    if (!cellMask[index])
                        continue;

                    int neighborCount = 0;
                    for (int oy = -1; oy <= 1; oy++)
                    {
                        for (int ox = -1; ox <= 1; ox++)
                        {
                            if (ox == 0 && oy == 0)
                                continue;

                            int nx = x + ox;
                            int ny = y + oy;
                            if (nx < 0 || nx >= columns || ny < 0 || ny >= rows)
                                continue;
                            if (cellMask[ny * columns + nx])
                                neighborCount++;
                        }
                    }

                    if (neighborCount < minNeighborCount)
                        filtered[index] = false;
                }
            }

            Array.Copy(filtered, cellMask, cellMask.Length);
        }

        int minIslandCellCount = Mathf.Max(0, largestCandidateGridMinIslandCellCount);
        int minIslandSpanCells = Mathf.Max(0, largestCandidateGridMinIslandSpanCells);
        int keepTopIslandCount = Mathf.Max(0, largestCandidateGridKeepTopIslandCount);
        if (minIslandCellCount <= 1 && minIslandSpanCells <= 1 && keepTopIslandCount <= 0)
            return;

        bool[] visited = new bool[cellMask.Length];
        int[] offsetsX = { 1, -1, 0, 0 };
        int[] offsetsY = { 0, 0, 1, -1 };
        Queue<int> queue = new Queue<int>();
        List<int> island = new List<int>(32);
        List<GridIslandInfo> keptIslands = new List<GridIslandInfo>(16);
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                int startIndex = y * columns + x;
                if (!cellMask[startIndex] || visited[startIndex])
                    continue;

                queue.Clear();
                island.Clear();
                int minIslandX = x, maxIslandX = x;
                int minIslandY = y, maxIslandY = y;
                queue.Enqueue(startIndex);
                visited[startIndex] = true;
                while (queue.Count > 0)
                {
                    int index = queue.Dequeue();
                    island.Add(index);
                    int cx = index % columns;
                    int cy = index / columns;
                    if (cx < minIslandX) minIslandX = cx;
                    if (cx > maxIslandX) maxIslandX = cx;
                    if (cy < minIslandY) minIslandY = cy;
                    if (cy > maxIslandY) maxIslandY = cy;
                    for (int dir = 0; dir < 4; dir++)
                    {
                        int nx = cx + offsetsX[dir];
                        int ny = cy + offsetsY[dir];
                        if (nx < 0 || nx >= columns || ny < 0 || ny >= rows)
                            continue;

                        int neighborIndex = ny * columns + nx;
                        if (!cellMask[neighborIndex] || visited[neighborIndex])
                            continue;

                        visited[neighborIndex] = true;
                        queue.Enqueue(neighborIndex);
                    }
                }

                int islandWidth = maxIslandX - minIslandX + 1;
                int islandHeight = maxIslandY - minIslandY + 1;
                bool keepByCount = island.Count >= minIslandCellCount;
                bool keepBySpan = islandWidth >= minIslandSpanCells && islandHeight >= minIslandSpanCells;
                if (keepByCount && keepBySpan)
                {
                    keptIslands.Add(new GridIslandInfo
                    {
                        cells = new List<int>(island),
                        count = island.Count,
                        width = islandWidth,
                        height = islandHeight
                    });
                    continue;
                }

                for (int i = 0; i < island.Count; i++)
                    cellMask[island[i]] = false;
            }
        }

        if (keepTopIslandCount <= 0 || keptIslands.Count <= keepTopIslandCount)
            return;

        keptIslands.Sort((a, b) =>
        {
            int byCount = b.count.CompareTo(a.count);
            if (byCount != 0)
                return byCount;
            int byArea = (b.width * b.height).CompareTo(a.width * a.height);
            if (byArea != 0)
                return byArea;
            return b.height.CompareTo(a.height);
        });

        bool[] keepMask = new bool[cellMask.Length];
        for (int islandIndex = 0; islandIndex < keepTopIslandCount; islandIndex++)
        {
            List<int> keptCells = keptIslands[islandIndex].cells;
            for (int i = 0; i < keptCells.Count; i++)
                keepMask[keptCells[i]] = true;
        }

        for (int i = 0; i < cellMask.Length; i++)
            cellMask[i] = cellMask[i] && keepMask[i];
    }

    private List<CandidateSurfaceInfo> BuildCandidateSurfaces(List<Vector3> vertices, List<Vector3> vertexNormals, List<int> triangles)
    {
        int faceCount = triangles.Count / 3;
        List<CandidateSurfaceInfo> candidateSurfaces = new List<CandidateSurfaceInfo>();
        if (faceCount <= 0)
            return candidateSurfaces;

        TriangleFaceInfo[] faces = BuildTriangleFaceInfos(vertices, vertexNormals, triangles);
        Dictionary<EdgeKey, List<int>> edgeToFaces = BuildTriangleEdgeAdjacency(triangles);
        float creaseDot = Mathf.Cos(Mathf.Clamp(surfaceRegionCreaseAngleDegrees, 1f, 60f) * Mathf.Deg2Rad);
        float maxPlaneOffset = Mathf.Max(0.001f, surfaceRegionMaxPlaneOffsetMeters);

        int[] faceSurfaceIds = new int[faceCount];
        for (int i = 0; i < faceSurfaceIds.Length; i++)
            faceSurfaceIds[i] = -1;

        Queue<int> queue = new Queue<int>();
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            if (faceSurfaceIds[faceIndex] >= 0)
                continue;

            int surfaceId = candidateSurfaces.Count;
            List<int> currentSurfaceFaces = new List<int>(32);
            Vector3 surfaceNormalSum = faces[faceIndex].normal;
            Vector3 surfaceCenterSum = faces[faceIndex].center;
            int surfaceFaceCount = 1;

            faceSurfaceIds[faceIndex] = surfaceId;
            currentSurfaceFaces.Add(faceIndex);
            queue.Enqueue(faceIndex);

            while (queue.Count > 0)
            {
                int currentFace = queue.Dequeue();
                Vector3 averageNormal = surfaceNormalSum.sqrMagnitude > 1e-8f ? surfaceNormalSum.normalized : faces[currentFace].normal;
                Vector3 averageCenter = surfaceCenterSum / Mathf.Max(1, surfaceFaceCount);

                ExpandCandidateSurfaceNeighbor(currentFace, new EdgeKey(faces[currentFace].a, faces[currentFace].b), edgeToFaces, faces, faceSurfaceIds, surfaceId, creaseDot, maxPlaneOffset, averageNormal, averageCenter, queue, currentSurfaceFaces, ref surfaceNormalSum, ref surfaceCenterSum, ref surfaceFaceCount);
                ExpandCandidateSurfaceNeighbor(currentFace, new EdgeKey(faces[currentFace].b, faces[currentFace].c), edgeToFaces, faces, faceSurfaceIds, surfaceId, creaseDot, maxPlaneOffset, averageNormal, averageCenter, queue, currentSurfaceFaces, ref surfaceNormalSum, ref surfaceCenterSum, ref surfaceFaceCount);
                ExpandCandidateSurfaceNeighbor(currentFace, new EdgeKey(faces[currentFace].c, faces[currentFace].a), edgeToFaces, faces, faceSurfaceIds, surfaceId, creaseDot, maxPlaneOffset, averageNormal, averageCenter, queue, currentSurfaceFaces, ref surfaceNormalSum, ref surfaceCenterSum, ref surfaceFaceCount);
            }

            candidateSurfaces.Add(BuildCandidateSurfaceInfo(currentSurfaceFaces, faces));
        }

        MergeAdjacentCandidateSurfaces(candidateSurfaces, faceSurfaceIds, faces, edgeToFaces);
        return candidateSurfaces;
    }

    private void MergeAdjacentCandidateSurfaces(List<CandidateSurfaceInfo> candidateSurfaces, int[] faceSurfaceIds, TriangleFaceInfo[] faces, Dictionary<EdgeKey, List<int>> edgeToFaces)
    {
        if (candidateSurfaces == null || candidateSurfaces.Count <= 1 || faceSurfaceIds == null || faces == null || edgeToFaces == null)
            return;

        float mergeDot = Mathf.Cos(Mathf.Clamp(candidateSurfaceMergeAngleDegrees, 1f, 60f) * Mathf.Deg2Rad);
        float mergePlaneOffset = Mathf.Max(0.001f, candidateSurfaceMergePlaneOffsetMeters);
        int minSharedEdges = Mathf.Max(1, candidateSurfaceMergeMinSharedEdges);
        int surfaceCount = candidateSurfaces.Count;
        Dictionary<EdgeKey, int> pairSharedEdgeCounts = new Dictionary<EdgeKey, int>(surfaceCount * 2);

        foreach (KeyValuePair<EdgeKey, List<int>> entry in edgeToFaces)
        {
            List<int> sharedFaces = entry.Value;
            if (sharedFaces == null || sharedFaces.Count < 2)
                continue;

            for (int i = 0; i < sharedFaces.Count - 1; i++)
            {
                int surfaceA = faceSurfaceIds[sharedFaces[i]];
                if (surfaceA < 0)
                    continue;

                for (int j = i + 1; j < sharedFaces.Count; j++)
                {
                    int surfaceB = faceSurfaceIds[sharedFaces[j]];
                    if (surfaceB < 0 || surfaceA == surfaceB)
                        continue;

                    EdgeKey pairKey = new EdgeKey(surfaceA, surfaceB);
                    pairSharedEdgeCounts.TryGetValue(pairKey, out int count);
                    pairSharedEdgeCounts[pairKey] = count + 1;
                }
            }
        }

        UnionFind unionFind = new UnionFind(surfaceCount);
        foreach (KeyValuePair<EdgeKey, int> pair in pairSharedEdgeCounts)
        {
            if (pair.Value < minSharedEdges)
                continue;

            int surfaceA = pair.Key.lo;
            int surfaceB = pair.Key.hi;
            CandidateSurfaceInfo a = candidateSurfaces[surfaceA];
            CandidateSurfaceInfo b = candidateSurfaces[surfaceB];
            if (a.faceIndices == null || b.faceIndices == null || a.faceIndices.Count <= 0 || b.faceIndices.Count <= 0)
                continue;

            if (Vector3.Dot(a.averageNormal, b.averageNormal) < mergeDot)
                continue;

            float planeOffsetAB = Mathf.Abs(Vector3.Dot(a.averageNormal, b.averageCenter - a.averageCenter));
            float planeOffsetBA = Mathf.Abs(Vector3.Dot(b.averageNormal, a.averageCenter - b.averageCenter));
            if (Mathf.Max(planeOffsetAB, planeOffsetBA) > mergePlaneOffset)
                continue;

            unionFind.Union(surfaceA, surfaceB);
        }

        Dictionary<int, List<int>> groupedSurfaceIds = new Dictionary<int, List<int>>(surfaceCount);
        for (int surfaceIndex = 0; surfaceIndex < surfaceCount; surfaceIndex++)
        {
            int root = unionFind.Find(surfaceIndex);
            if (!groupedSurfaceIds.TryGetValue(root, out List<int> grouped))
            {
                grouped = new List<int>(4);
                groupedSurfaceIds[root] = grouped;
            }
            grouped.Add(surfaceIndex);
        }

        if (groupedSurfaceIds.Count >= surfaceCount)
            return;

        List<CandidateSurfaceInfo> mergedSurfaces = new List<CandidateSurfaceInfo>(groupedSurfaceIds.Count);
        foreach (List<int> grouped in groupedSurfaceIds.Values)
        {
            List<int> mergedFaceIndices = new List<int>(64);
            Vector3 normalSum = Vector3.zero;
            Vector3 centerSum = Vector3.zero;
            int faceCountSum = 0;

            for (int i = 0; i < grouped.Count; i++)
            {
                CandidateSurfaceInfo source = candidateSurfaces[grouped[i]];
                if (source.faceIndices == null || source.faceIndices.Count <= 0)
                    continue;

                mergedFaceIndices.AddRange(source.faceIndices);
                int sourceFaceCount = source.faceIndices.Count;
                normalSum += source.averageNormal * sourceFaceCount;
                centerSum += source.averageCenter * sourceFaceCount;
                faceCountSum += sourceFaceCount;
            }

            if (mergedFaceIndices.Count <= 0)
                continue;

            mergedSurfaces.Add(BuildCandidateSurfaceInfo(mergedFaceIndices, faces));
        }

        candidateSurfaces.Clear();
        candidateSurfaces.AddRange(mergedSurfaces);
    }

    private TriangleFaceInfo[] BuildTriangleFaceInfos(List<Vector3> vertices, List<Vector3> vertexNormals, List<int> triangles)
    {
        int faceCount = triangles.Count / 3;
        TriangleFaceInfo[] faces = new TriangleFaceInfo[faceCount];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int triStart = faceIndex * 3;
            int a = triangles[triStart];
            int b = triangles[triStart + 1];
            int c = triangles[triStart + 2];
            Vector3 va = vertices[a];
            Vector3 vb = vertices[b];
            Vector3 vc = vertices[c];
            Vector3 normal = Vector3.Cross(vb - va, vc - va);
            if (normal.sqrMagnitude <= 1e-8f)
                normal = Vector3.up;
            else
                normal.Normalize();

            Vector3 averagedVertexNormal = Vector3.zero;
            if (a >= 0 && a < vertexNormals.Count)
                averagedVertexNormal += vertexNormals[a];
            if (b >= 0 && b < vertexNormals.Count)
                averagedVertexNormal += vertexNormals[b];
            if (c >= 0 && c < vertexNormals.Count)
                averagedVertexNormal += vertexNormals[c];
            if (averagedVertexNormal.sqrMagnitude > 1e-8f && Vector3.Dot(normal, averagedVertexNormal.normalized) < 0f)
                normal = -normal;

            Vector3 center = (va + vb + vc) / 3f;
            faces[faceIndex] = new TriangleFaceInfo
            {
                a = a,
                b = b,
                c = c,
                normal = normal,
                center = center,
                planeOffset = Vector3.Dot(normal, center)
            };
        }

        return faces;
    }

    private Dictionary<EdgeKey, List<int>> BuildTriangleEdgeAdjacency(List<int> triangles)
    {
        int faceCount = triangles.Count / 3;
        Dictionary<EdgeKey, List<int>> edgeToFaces = new Dictionary<EdgeKey, List<int>>(faceCount * 2);
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int triStart = faceIndex * 3;
            int a = triangles[triStart];
            int b = triangles[triStart + 1];
            int c = triangles[triStart + 2];
            AddFaceEdge(edgeToFaces, new EdgeKey(a, b), faceIndex);
            AddFaceEdge(edgeToFaces, new EdgeKey(b, c), faceIndex);
            AddFaceEdge(edgeToFaces, new EdgeKey(c, a), faceIndex);
        }

        return edgeToFaces;
    }

    private void AddCandidateSurfaceTriangles(List<int> faceIndices, List<int> triangles, List<int> destination)
    {
        for (int i = 0; i < faceIndices.Count; i++)
        {
            int triStart = faceIndices[i] * 3;
            destination.Add(triangles[triStart]);
            destination.Add(triangles[triStart + 1]);
            destination.Add(triangles[triStart + 2]);
        }
    }

    private void ExpandCandidateSurfaceNeighbor(
        int currentFace,
        EdgeKey edge,
        Dictionary<EdgeKey, List<int>> edgeToFaces,
        TriangleFaceInfo[] faces,
        int[] faceSurfaceIds,
        int surfaceId,
        float creaseDot,
        float maxPlaneOffset,
        Vector3 averageNormal,
        Vector3 averageCenter,
        Queue<int> queue,
        List<int> currentSurfaceFaces,
        ref Vector3 surfaceNormalSum,
        ref Vector3 surfaceCenterSum,
        ref int surfaceFaceCount)
    {
        if (!edgeToFaces.TryGetValue(edge, out List<int> sharedFaces))
            return;

        for (int i = 0; i < sharedFaces.Count; i++)
        {
            int neighborFace = sharedFaces[i];
            if (neighborFace == currentFace || faceSurfaceIds[neighborFace] >= 0)
                continue;

            if (Vector3.Dot(faces[currentFace].normal, faces[neighborFace].normal) < creaseDot)
                continue;

            float planeOffset = Mathf.Abs(Vector3.Dot(averageNormal, faces[neighborFace].center - averageCenter));
            if (planeOffset > maxPlaneOffset)
                continue;

            faceSurfaceIds[neighborFace] = surfaceId;
            currentSurfaceFaces.Add(neighborFace);
            surfaceNormalSum += faces[neighborFace].normal;
            surfaceCenterSum += faces[neighborFace].center;
            surfaceFaceCount++;
            queue.Enqueue(neighborFace);
        }
    }


    private int GetSurfaceNormalBucketIndex(Vector3 normal, float minAxisDot)
    {
        Vector3 n = normal.sqrMagnitude > 1e-8f ? normal.normalized : Vector3.up;
        Vector3[] axes =
        {
            Vector3.right,
            Vector3.left,
            Vector3.up,
            Vector3.down,
            Vector3.forward,
            Vector3.back
        };

        int bestIndex = -1;
        float bestDot = minAxisDot;
        for (int i = 0; i < axes.Length; i++)
        {
            float dot = Vector3.Dot(n, axes[i]);
            if (dot > bestDot)
            {
                bestDot = dot;
                bestIndex = i;
            }
        }

        return bestIndex >= 0 ? bestIndex : 6;
    }

    private bool PointInProjectedTriangle(Vector2 point, ProjectedTriangle2D triangle)
    {
        float d1 = SignedTriangleArea2D(point, triangle.a, triangle.b);
        float d2 = SignedTriangleArea2D(point, triangle.b, triangle.c);
        float d3 = SignedTriangleArea2D(point, triangle.c, triangle.a);
        bool hasNeg = (d1 < 0f) || (d2 < 0f) || (d3 < 0f);
        bool hasPos = (d1 > 0f) || (d2 > 0f) || (d3 > 0f);
        return !(hasNeg && hasPos);
    }

    private float SignedTriangleArea2D(Vector2 p1, Vector2 p2, Vector2 p3)
    {
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    }

    private float DistancePointToSegment2DSqr(Vector2 point, Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a;
        float lengthSqr = ab.sqrMagnitude;
        if (lengthSqr <= 1e-8f)
            return (point - a).sqrMagnitude;

        float t = Mathf.Clamp01(Vector2.Dot(point - a, ab) / lengthSqr);
        Vector2 closest = a + ab * t;
        return (point - closest).sqrMagnitude;
    }

    private bool TryBuildProjectedCellPolygon(List<ProjectedTriangle2D> projectedTriangles, float minX, float maxX, float minY, float maxY, out List<Vector2> polygon)
    {
        polygon = null;
        if (projectedTriangles == null || projectedTriangles.Count <= 0)
            return false;

        List<Vector2> points = new List<Vector2>(8);
        for (int i = 0; i < projectedTriangles.Count; i++)
        {
            if (!TryClipTriangleToRect(projectedTriangles[i], minX, maxX, minY, maxY, out List<Vector2> clipped))
                continue;

            for (int pointIndex = 0; pointIndex < clipped.Count; pointIndex++)
                AppendUniquePoint(points, clipped[pointIndex], 0.0005f);
        }

        if (points.Count < 3)
            return false;

        polygon = BuildConvexHull(points);
        return polygon != null && polygon.Count >= 3 && ComputePolygonAreaAbs(polygon) > 1e-6f;
    }

    private bool TryClipTriangleToRect(ProjectedTriangle2D triangle, float minX, float maxX, float minY, float maxY, out List<Vector2> clippedPolygon)
    {
        clippedPolygon = null;
        if (triangle.maxX < minX || triangle.minX > maxX || triangle.maxY < minY || triangle.minY > maxY)
            return false;

        List<Vector2> polygon = new List<Vector2>(3) { triangle.a, triangle.b, triangle.c };
        polygon = ClipPolygonAgainstX(polygon, minX, true);
        polygon = ClipPolygonAgainstX(polygon, maxX, false);
        polygon = ClipPolygonAgainstY(polygon, minY, true);
        polygon = ClipPolygonAgainstY(polygon, maxY, false);
        if (polygon == null || polygon.Count < 3 || ComputePolygonAreaAbs(polygon) <= 1e-6f)
            return false;

        clippedPolygon = polygon;
        return true;
    }

    private List<Vector2> ClipPolygonAgainstX(List<Vector2> input, float boundary, bool keepGreater)
    {
        if (input == null || input.Count <= 0)
            return null;

        List<Vector2> output = new List<Vector2>(input.Count + 2);
        Vector2 previous = input[input.Count - 1];
        bool previousInside = keepGreater ? previous.x >= boundary : previous.x <= boundary;
        for (int i = 0; i < input.Count; i++)
        {
            Vector2 current = input[i];
            bool currentInside = keepGreater ? current.x >= boundary : current.x <= boundary;
            if (currentInside != previousInside && Mathf.Abs(current.x - previous.x) > 1e-6f)
            {
                float t = (boundary - previous.x) / (current.x - previous.x);
                output.Add(Vector2.Lerp(previous, current, Mathf.Clamp01(t)));
            }

            if (currentInside)
                output.Add(current);

            previous = current;
            previousInside = currentInside;
        }

        return output;
    }

    private List<Vector2> ClipPolygonAgainstY(List<Vector2> input, float boundary, bool keepGreater)
    {
        if (input == null || input.Count <= 0)
            return null;

        List<Vector2> output = new List<Vector2>(input.Count + 2);
        Vector2 previous = input[input.Count - 1];
        bool previousInside = keepGreater ? previous.y >= boundary : previous.y <= boundary;
        for (int i = 0; i < input.Count; i++)
        {
            Vector2 current = input[i];
            bool currentInside = keepGreater ? current.y >= boundary : current.y <= boundary;
            if (currentInside != previousInside && Mathf.Abs(current.y - previous.y) > 1e-6f)
            {
                float t = (boundary - previous.y) / (current.y - previous.y);
                output.Add(Vector2.Lerp(previous, current, Mathf.Clamp01(t)));
            }

            if (currentInside)
                output.Add(current);

            previous = current;
            previousInside = currentInside;
        }

        return output;
    }

    private void AppendUniquePoint(List<Vector2> points, Vector2 candidate, float epsilon)
    {
        if (points == null)
            return;

        float epsilonSqr = epsilon * epsilon;
        for (int i = 0; i < points.Count; i++)
        {
            if ((points[i] - candidate).sqrMagnitude <= epsilonSqr)
                return;
        }

        points.Add(candidate);
    }

    private List<Vector2> BuildConvexHull(List<Vector2> points)
    {
        if (points == null || points.Count < 3)
            return points;

        List<Vector2> sorted = new List<Vector2>(points);
        sorted.Sort((a, b) =>
        {
            int byX = a.x.CompareTo(b.x);
            return byX != 0 ? byX : a.y.CompareTo(b.y);
        });

        List<Vector2> lower = new List<Vector2>(sorted.Count);
        for (int i = 0; i < sorted.Count; i++)
        {
            while (lower.Count >= 2 && SignedTriangleArea2D(lower[lower.Count - 2], lower[lower.Count - 1], sorted[i]) <= 1e-6f)
                lower.RemoveAt(lower.Count - 1);
            lower.Add(sorted[i]);
        }

        List<Vector2> upper = new List<Vector2>(sorted.Count);
        for (int i = sorted.Count - 1; i >= 0; i--)
        {
            while (upper.Count >= 2 && SignedTriangleArea2D(upper[upper.Count - 2], upper[upper.Count - 1], sorted[i]) <= 1e-6f)
                upper.RemoveAt(upper.Count - 1);
            upper.Add(sorted[i]);
        }

        lower.RemoveAt(lower.Count - 1);
        upper.RemoveAt(upper.Count - 1);
        lower.AddRange(upper);
        return lower;
    }

    private float ComputePolygonAreaAbs(List<Vector2> polygon)
    {
        if (polygon == null || polygon.Count < 3)
            return 0f;

        float area = 0f;
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            area += a.x * b.y - b.x * a.y;
        }

        return Mathf.Abs(area) * 0.5f;
    }

    private bool IsProjectedCellNearlyFullRect(List<Vector2> polygon, float minX, float maxX, float minY, float maxY)
    {
        float rectArea = Mathf.Max(1e-6f, (maxX - minX) * (maxY - minY));
        return ComputePolygonAreaAbs(polygon) >= rectArea * 0.995f;
    }

    private void AddDirectLineSegment(Vector3 a, Vector3 b, Vector3 normal, List<Vector3> localLineVertices, List<Vector3> localLineNormals, List<int> localLineIndices)
    {
        int start = localLineVertices.Count;
        localLineVertices.Add(a);
        localLineNormals.Add(normal);
        localLineVertices.Add(b);
        localLineNormals.Add(normal);
        localLineIndices.Add(start);
        localLineIndices.Add(start + 1);
    }

    private bool TryClipSegmentToConvexPolygon(Vector2 start, Vector2 end, List<Vector2> polygon, out Vector2 clippedStart, out Vector2 clippedEnd)
    {
        clippedStart = Vector2.zero;
        clippedEnd = Vector2.zero;
        if (polygon == null || polygon.Count < 3)
            return false;

        List<float> parameters = new List<float>(8) { 0f, 1f };
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            if (TryGetSegmentIntersectionParameter(start, end, a, b, out float t))
                AppendUniqueParameter(parameters, t, 0.0001f);
        }

        parameters.Sort();
        for (int i = 0; i < parameters.Count - 1; i++)
        {
            float t0 = parameters[i];
            float t1 = parameters[i + 1];
            if (t1 - t0 <= 1e-5f)
                continue;

            float midpointT = (t0 + t1) * 0.5f;
            Vector2 midpoint = Vector2.Lerp(start, end, midpointT);
            if (!IsPointInsideConvexPolygon(midpoint, polygon))
                continue;

            clippedStart = Vector2.Lerp(start, end, t0);
            clippedEnd = Vector2.Lerp(start, end, t1);
            return (clippedEnd - clippedStart).sqrMagnitude > 1e-8f;
        }

        return false;
    }

    private void AppendUniqueParameter(List<float> parameters, float value, float epsilon)
    {
        for (int i = 0; i < parameters.Count; i++)
        {
            if (Mathf.Abs(parameters[i] - value) <= epsilon)
                return;
        }

        parameters.Add(Mathf.Clamp01(value));
    }

    private bool TryGetSegmentIntersectionParameter(Vector2 segmentA, Vector2 segmentB, Vector2 edgeA, Vector2 edgeB, out float segmentT)
    {
        segmentT = 0f;
        Vector2 r = segmentB - segmentA;
        Vector2 s = edgeB - edgeA;
        float denominator = r.x * s.y - r.y * s.x;
        if (Mathf.Abs(denominator) <= 1e-6f)
            return false;

        Vector2 delta = edgeA - segmentA;
        float t = (delta.x * s.y - delta.y * s.x) / denominator;
        float u = (delta.x * r.y - delta.y * r.x) / denominator;
        if (t < 0f || t > 1f || u < 0f || u > 1f)
            return false;

        segmentT = t;
        return true;
    }

    private bool IsPointInsideConvexPolygon(Vector2 point, List<Vector2> polygon)
    {
        if (polygon == null || polygon.Count < 3)
            return false;

        bool hasNegative = false;
        bool hasPositive = false;
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            float cross = SignedTriangleArea2D(point, a, b);
            if (cross < -1e-5f) hasNegative = true;
            else if (cross > 1e-5f) hasPositive = true;
            if (hasNegative && hasPositive)
                return false;
        }

        return true;
    }

    private struct TriangleFaceInfo
    {
        public int a;
        public int b;
        public int c;
        public Vector3 normal;
        public Vector3 center;
        public float planeOffset;
    }

    private struct CandidateSurfaceInfo
    {
        public List<int> faceIndices;
        public Vector3 averageNormal;
        public Vector3 averageCenter;
        public float averagePlaneOffset;
        public float averageNormalAlignment;
        public float stabilityScore;
        public float viewCenterScore;
        public float facingScore;
        public float displayPriorityScore;
        public bool isViewCenterQualified;
        public bool hasFocusAnchor;
        public Vector3 focusAnchorWorld;
        public float focusAnchorLocalU;
        public float focusAnchorLocalV;
        public float focusAnchorDistance;
    }

    private void SortCandidateSurfacesForDisplay(List<CandidateSurfaceInfo> candidateSurfaces, List<Vector3> vertices = null, List<int> triangles = null)
    {
        if (candidateSurfaces == null || candidateSurfaces.Count <= 1)
            return;

        Camera viewCamera = candidatePreferViewCenter ? Camera.main : null;
        float centerRadius = Mathf.Max(0.05f, candidateViewCenterRadius);
        float facingThreshold = Mathf.Clamp01(candidateMinFacingScore);
        float facingRelaxation = Mathf.Clamp01(candidateCenterFacingRelaxation);
        Ray focusRay = default;
        bool hasFocusRay = false;
        if (viewCamera != null)
        {
            focusRay = viewCamera.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0f));
            hasFocusRay = focusRay.direction.sqrMagnitude > 1e-8f;
        }
        for (int i = 0; i < candidateSurfaces.Count; i++)
        {
            CandidateSurfaceInfo candidate = candidateSurfaces[i];
            int faceCount = candidate.faceIndices != null ? candidate.faceIndices.Count : 0;
            float basePriorityScore = candidate.stabilityScore + Mathf.Sqrt(Mathf.Max(0, faceCount)) * Mathf.Max(0f, candidateFaceCountScoreWeight);
            candidate.viewCenterScore = 0f;
            candidate.facingScore = 0f;
            candidate.displayPriorityScore = basePriorityScore;
            candidate.isViewCenterQualified = false;
            candidate.hasFocusAnchor = false;
            candidate.focusAnchorWorld = Vector3.zero;
            candidate.focusAnchorLocalU = 0f;
            candidate.focusAnchorLocalV = 0f;
            candidate.focusAnchorDistance = float.PositiveInfinity;

            if (viewCamera != null)
            {
                Vector3 viewport = viewCamera.WorldToViewportPoint(candidate.averageCenter);
                if (viewport.z > 0f)
                {
                    float deltaX = viewport.x - 0.5f;
                    float deltaY = viewport.y - 0.5f;
                    float radialDistance = Mathf.Sqrt(deltaX * deltaX + deltaY * deltaY);
                    candidate.viewCenterScore = Mathf.Clamp01(1f - radialDistance / centerRadius);
                    Vector3 toCandidate = candidate.averageCenter - viewCamera.transform.position;
                    if (toCandidate.sqrMagnitude > 1e-8f)
                        candidate.facingScore = Mathf.Clamp01(Mathf.Abs(Vector3.Dot(candidate.averageNormal.normalized, toCandidate.normalized)));
                    float adaptiveFacingThreshold = Mathf.Max(0f, facingThreshold - candidate.viewCenterScore * facingRelaxation);
                    candidate.isViewCenterQualified = candidate.viewCenterScore > 0f && candidate.facingScore >= adaptiveFacingThreshold;
                    if (candidate.isViewCenterQualified)
                    {
                        candidate.displayPriorityScore = basePriorityScore
                            + candidate.viewCenterScore * Mathf.Max(0f, candidateViewCenterScoreWeight)
                            + candidate.facingScore * Mathf.Max(0f, candidateFacingScoreWeight);
                    }
                }
            }

            if (hasFocusRay
                && vertices != null && triangles != null
                && TryGetCandidatePlaneFrame(candidate, out Vector3 planeOrigin, out Vector3 planeNormal, out Vector3 localRight, out Vector3 localUp)
                && TryProjectFocusRayOntoCandidate(focusRay, planeOrigin, planeNormal, localRight, localUp, candidate, vertices, triangles, out Vector3 focusWorld, out float focusLocalU, out float focusLocalV, out float focusDistance))
            {
                candidate.hasFocusAnchor = true;
                candidate.focusAnchorWorld = focusWorld;
                candidate.focusAnchorLocalU = focusLocalU;
                candidate.focusAnchorLocalV = focusLocalV;
                candidate.focusAnchorDistance = focusDistance;
                candidate.displayPriorityScore += Mathf.Max(0f, candidateViewCenterScoreWeight) * 4f;
            }

            candidateSurfaces[i] = candidate;
        }

        candidateSurfaces.Sort(CompareCandidateSurfaces);
    }

    private static int CompareCandidateSurfaces(CandidateSurfaceInfo a, CandidateSurfaceInfo b)
    {
        int byFocusAnchor = b.hasFocusAnchor.CompareTo(a.hasFocusAnchor);
        if (byFocusAnchor != 0)
            return byFocusAnchor;

        if (a.hasFocusAnchor && b.hasFocusAnchor)
        {
            int byFocusDistance = a.focusAnchorDistance.CompareTo(b.focusAnchorDistance);
            if (byFocusDistance != 0)
                return byFocusDistance;
        }

        int byViewCenterQualified = b.isViewCenterQualified.CompareTo(a.isViewCenterQualified);
        if (byViewCenterQualified != 0)
            return byViewCenterQualified;

        if (a.isViewCenterQualified && b.isViewCenterQualified)
        {
            int byViewCenterScoreLocked = b.viewCenterScore.CompareTo(a.viewCenterScore);
            if (byViewCenterScoreLocked != 0)
                return byViewCenterScoreLocked;

            int byFacingLocked = b.facingScore.CompareTo(a.facingScore);
            if (byFacingLocked != 0)
                return byFacingLocked;
        }

        int byDisplayPriority = b.displayPriorityScore.CompareTo(a.displayPriorityScore);
        if (byDisplayPriority != 0)
            return byDisplayPriority;

        int byScore = b.stabilityScore.CompareTo(a.stabilityScore);
        if (byScore != 0)
            return byScore;

        int byFaceCount = b.faceIndices.Count.CompareTo(a.faceIndices.Count);
        if (byFaceCount != 0)
            return byFaceCount;

        int byViewCenterScore = b.viewCenterScore.CompareTo(a.viewCenterScore);
        if (byViewCenterScore != 0)
            return byViewCenterScore;

        int byFacingScore = b.facingScore.CompareTo(a.facingScore);
        if (byFacingScore != 0)
            return byFacingScore;

        return b.averageNormalAlignment.CompareTo(a.averageNormalAlignment);
    }

    private bool TryGetCandidatePlaneFrame(CandidateSurfaceInfo candidate, out Vector3 planeOrigin, out Vector3 planeNormal, out Vector3 localRight, out Vector3 localUp)
    {
        planeOrigin = candidate.averageCenter;
        planeNormal = candidate.averageNormal.sqrMagnitude > 1e-8f ? candidate.averageNormal.normalized : Vector3.forward;
        localUp = Vector3.ProjectOnPlane(Vector3.up, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.forward, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
            localUp = Vector3.ProjectOnPlane(Vector3.right, planeNormal);
        if (localUp.sqrMagnitude <= 1e-8f)
        {
            localRight = Vector3.right;
            return false;
        }

        localUp.Normalize();
        localRight = Vector3.Cross(localUp, planeNormal);
        if (localRight.sqrMagnitude <= 1e-8f)
            return false;
        localRight.Normalize();
        localUp = Vector3.Cross(planeNormal, localRight).normalized;
        return true;
    }

    private bool TryProjectFocusRayOntoCandidate(
        Ray focusRay,
        Vector3 planeOrigin,
        Vector3 planeNormal,
        Vector3 localRight,
        Vector3 localUp,
        CandidateSurfaceInfo candidate,
        List<Vector3> vertices,
        List<int> triangles,
        out Vector3 focusWorld,
        out float focusLocalU,
        out float focusLocalV,
        out float focusDistance)
    {
        focusWorld = Vector3.zero;
        focusLocalU = 0f;
        focusLocalV = 0f;
        focusDistance = float.PositiveInfinity;
        if (candidate.faceIndices == null || candidate.faceIndices.Count <= 0 || vertices == null || triangles == null)
            return false;

        float denom = Vector3.Dot(planeNormal, focusRay.direction);
        if (Mathf.Abs(denom) <= 1e-6f)
            return false;

        float hitDistance = Vector3.Dot(planeOrigin - focusRay.origin, planeNormal) / denom;
        if (hitDistance <= 0f)
            return false;

        focusWorld = focusRay.origin + focusRay.direction * hitDistance;
        Vector3 local = focusWorld - planeOrigin;
        focusLocalU = Vector3.Dot(local, localRight);
        focusLocalV = Vector3.Dot(local, localUp);
        Vector2 focusPoint = new Vector2(focusLocalU, focusLocalV);

        float bestDistanceSqr = float.PositiveInfinity;
        for (int i = 0; i < candidate.faceIndices.Count; i++)
        {
            int triStart = candidate.faceIndices[i] * 3;
            if (triStart < 0 || triStart + 2 >= triangles.Count)
                continue;

            Vector3 pa = vertices[triangles[triStart]];
            Vector3 pb = vertices[triangles[triStart + 1]];
            Vector3 pc = vertices[triangles[triStart + 2]];
            ProjectedTriangle2D triangle = new ProjectedTriangle2D
            {
                a = new Vector2(Vector3.Dot(pa - planeOrigin, localRight), Vector3.Dot(pa - planeOrigin, localUp)),
                b = new Vector2(Vector3.Dot(pb - planeOrigin, localRight), Vector3.Dot(pb - planeOrigin, localUp)),
                c = new Vector2(Vector3.Dot(pc - planeOrigin, localRight), Vector3.Dot(pc - planeOrigin, localUp))
            };
            triangle.minX = Mathf.Min(triangle.a.x, triangle.b.x, triangle.c.x);
            triangle.maxX = Mathf.Max(triangle.a.x, triangle.b.x, triangle.c.x);
            triangle.minY = Mathf.Min(triangle.a.y, triangle.b.y, triangle.c.y);
            triangle.maxY = Mathf.Max(triangle.a.y, triangle.b.y, triangle.c.y);

            if (PointInProjectedTriangle(focusPoint, triangle))
            {
                focusDistance = 0f;
                return true;
            }

            float triangleDistanceSqr = Mathf.Min(
                DistancePointToSegment2DSqr(focusPoint, triangle.a, triangle.b),
                DistancePointToSegment2DSqr(focusPoint, triangle.b, triangle.c),
                DistancePointToSegment2DSqr(focusPoint, triangle.c, triangle.a));
            if (triangleDistanceSqr < bestDistanceSqr)
                bestDistanceSqr = triangleDistanceSqr;
        }

        if (!float.IsFinite(bestDistanceSqr))
            return false;

        focusDistance = Mathf.Sqrt(bestDistanceSqr);
        float acceptanceDistance = Mathf.Max(0.02f, largestCandidateGridCellSizeMeters * 0.75f);
        return focusDistance <= acceptanceDistance;
    }

    private static CandidateSurfaceInfo BuildCandidateSurfaceInfo(List<int> faceIndices, TriangleFaceInfo[] faces)
    {
        List<int> safeFaceIndices = faceIndices ?? new List<int>();
        if (safeFaceIndices.Count <= 0 || faces == null || faces.Length <= 0)
        {
            return new CandidateSurfaceInfo
            {
                faceIndices = safeFaceIndices,
                averageNormal = Vector3.up,
                averageCenter = Vector3.zero,
                averagePlaneOffset = 0f,
                averageNormalAlignment = 0f,
                stabilityScore = 0f,
                viewCenterScore = 0f,
                facingScore = 0f,
                displayPriorityScore = 0f,
                isViewCenterQualified = false,
                hasFocusAnchor = false,
                focusAnchorWorld = Vector3.zero,
                focusAnchorLocalU = 0f,
                focusAnchorLocalV = 0f,
                focusAnchorDistance = float.PositiveInfinity
            };
        }

        Vector3 normalSum = Vector3.zero;
        Vector3 centerSum = Vector3.zero;
        int validFaceCount = 0;
        for (int i = 0; i < safeFaceIndices.Count; i++)
        {
            int faceIndex = safeFaceIndices[i];
            if (faceIndex < 0 || faceIndex >= faces.Length)
                continue;

            normalSum += faces[faceIndex].normal;
            centerSum += faces[faceIndex].center;
            validFaceCount++;
        }

        if (validFaceCount <= 0)
        {
            return new CandidateSurfaceInfo
            {
                faceIndices = safeFaceIndices,
                averageNormal = Vector3.up,
                averageCenter = Vector3.zero,
                averagePlaneOffset = 0f,
                averageNormalAlignment = 0f,
                stabilityScore = 0f,
                viewCenterScore = 0f,
                facingScore = 0f,
                displayPriorityScore = 0f,
                isViewCenterQualified = false,
                hasFocusAnchor = false,
                focusAnchorWorld = Vector3.zero,
                focusAnchorLocalU = 0f,
                focusAnchorLocalV = 0f,
                focusAnchorDistance = float.PositiveInfinity
            };
        }

        Vector3 averageNormal = normalSum.sqrMagnitude > 1e-8f ? normalSum.normalized : Vector3.up;
        Vector3 averageCenter = centerSum / validFaceCount;
        float alignmentSum = 0f;
        float planeOffsetSum = 0f;
        for (int i = 0; i < safeFaceIndices.Count; i++)
        {
            int faceIndex = safeFaceIndices[i];
            if (faceIndex < 0 || faceIndex >= faces.Length)
                continue;

            TriangleFaceInfo face = faces[faceIndex];
            alignmentSum += Mathf.Clamp01(Vector3.Dot(averageNormal, face.normal));
            planeOffsetSum += Mathf.Abs(Vector3.Dot(averageNormal, face.center - averageCenter));
        }

        float averageAlignment = alignmentSum / validFaceCount;
        float averagePlaneOffset = planeOffsetSum / validFaceCount;
        float stabilityScore = validFaceCount * averageAlignment / (1f + averagePlaneOffset * 20f);
        return new CandidateSurfaceInfo
        {
            faceIndices = safeFaceIndices,
            averageNormal = averageNormal,
            averageCenter = averageCenter,
            averagePlaneOffset = averagePlaneOffset,
            averageNormalAlignment = averageAlignment,
            stabilityScore = stabilityScore,
            viewCenterScore = 0f,
            facingScore = 0f,
            displayPriorityScore = stabilityScore,
            isViewCenterQualified = false,
            hasFocusAnchor = false,
            focusAnchorWorld = Vector3.zero,
            focusAnchorLocalU = 0f,
            focusAnchorLocalV = 0f,
            focusAnchorDistance = float.PositiveInfinity
        };
    }

    private struct GridIslandInfo
    {
        public List<int> cells;
        public int count;
        public int width;
        public int height;
    }

    private sealed class UnionFind
    {
        private readonly int[] _parent;
        private readonly byte[] _rank;

        public UnionFind(int count)
        {
            _parent = new int[count];
            _rank = new byte[count];
            for (int i = 0; i < count; i++)
                _parent[i] = i;
        }

        public int Find(int value)
        {
            if (_parent[value] != value)
                _parent[value] = Find(_parent[value]);
            return _parent[value];
        }

        public void Union(int a, int b)
        {
            int rootA = Find(a);
            int rootB = Find(b);
            if (rootA == rootB)
                return;

            if (_rank[rootA] < _rank[rootB])
            {
                _parent[rootA] = rootB;
                return;
            }

            if (_rank[rootA] > _rank[rootB])
            {
                _parent[rootB] = rootA;
                return;
            }

            _parent[rootB] = rootA;
            _rank[rootA]++;
        }
    }

    private struct ProjectedTriangle2D
    {
        public Vector2 a;
        public Vector2 b;
        public Vector2 c;
        public float minX;
        public float maxX;
        public float minY;
        public float maxY;
    }

    private struct RennetTerrainTriangle
    {
        public Vector2 a;
        public Vector2 b;
        public Vector2 c;
        public Vector3 pa;
        public Vector3 pb;
        public Vector3 pc;
        public Vector3 normal;
        public float minX;
        public float maxX;
        public float minY;
        public float maxY;
    }

    private struct FocusedGridOverlayState
    {
        public bool isValid;
        public Vector3 planeOrigin;
        public Vector3 planeNormal;
        public Vector3 localRight;
        public Vector3 localUp;
        public float minU;
        public float minV;
        public float stepU;
        public float stepV;
        public int cellColumns;
        public int cellRows;
    }

    private readonly struct EdgeKey
    {
        public readonly int lo;
        public readonly int hi;

        public EdgeKey(int a, int b)
        {
            if (a < b) { lo = a; hi = b; }
            else { lo = b; hi = a; }
        }
    }

    private List<List<int>> BuildMeshTriangleRegions(List<Vector3> vertices, List<int> triangles)
    {
        int faceCount = triangles.Count / 3;
        List<List<int>> regions = new List<List<int>>();
        if (faceCount <= 0)
            return regions;

        TriangleFaceInfo[] faces = new TriangleFaceInfo[faceCount];
        Dictionary<EdgeKey, List<int>> edgeToFaces = new Dictionary<EdgeKey, List<int>>(faceCount * 2);
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int triStart = faceIndex * 3;
            int a = triangles[triStart];
            int b = triangles[triStart + 1];
            int c = triangles[triStart + 2];
            Vector3 va = vertices[a];
            Vector3 vb = vertices[b];
            Vector3 vc = vertices[c];
            Vector3 normal = Vector3.Cross(vb - va, vc - va);
            if (normal.sqrMagnitude <= 1e-8f)
                normal = Vector3.up;
            else
                normal.Normalize();

            Vector3 center = (va + vb + vc) / 3f;
            faces[faceIndex] = new TriangleFaceInfo
            {
                a = a,
                b = b,
                c = c,
                normal = normal,
                center = center,
                planeOffset = Vector3.Dot(normal, center)
            };

            AddFaceEdge(edgeToFaces, new EdgeKey(a, b), faceIndex);
            AddFaceEdge(edgeToFaces, new EdgeKey(b, c), faceIndex);
            AddFaceEdge(edgeToFaces, new EdgeKey(c, a), faceIndex);
        }

        int[] faceRegionIds = new int[faceCount];
        for (int i = 0; i < faceRegionIds.Length; i++)
            faceRegionIds[i] = -1;

        int nextRegionId = 0;
        Queue<int> queue = new Queue<int>();
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            if (faceRegionIds[faceIndex] >= 0)
                continue;

            faceRegionIds[faceIndex] = nextRegionId;
            queue.Enqueue(faceIndex);
            regions.Add(new List<int>(256));

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                int triStart = current * 3;
                regions[nextRegionId].Add(triangles[triStart]);
                regions[nextRegionId].Add(triangles[triStart + 1]);
                regions[nextRegionId].Add(triangles[triStart + 2]);

                ExpandTriangleNeighbor(current, new EdgeKey(faces[current].a, faces[current].b), edgeToFaces, faces, faceRegionIds, nextRegionId, queue);
                ExpandTriangleNeighbor(current, new EdgeKey(faces[current].b, faces[current].c), edgeToFaces, faces, faceRegionIds, nextRegionId, queue);
                ExpandTriangleNeighbor(current, new EdgeKey(faces[current].c, faces[current].a), edgeToFaces, faces, faceRegionIds, nextRegionId, queue);
            }

            nextRegionId++;
        }

        return regions;
    }

    private void AddFaceEdge(Dictionary<EdgeKey, List<int>> edgeToFaces, EdgeKey edge, int faceIndex)
    {
        if (!edgeToFaces.TryGetValue(edge, out List<int> faces))
        {
            faces = new List<int>(2);
            edgeToFaces[edge] = faces;
        }

        faces.Add(faceIndex);
    }

    private void ExpandNormalPatchNeighbor(
        int currentFace,
        EdgeKey edge,
        Dictionary<EdgeKey, List<int>> edgeToFaces,
        TriangleFaceInfo[] faces,
        int[] facePatchIds,
        int patchId,
        float patchDot,
        Queue<int> queue)
    {
        if (!edgeToFaces.TryGetValue(edge, out List<int> sharedFaces))
            return;

        for (int i = 0; i < sharedFaces.Count; i++)
        {
            int neighbor = sharedFaces[i];
            if (neighbor == currentFace || facePatchIds[neighbor] >= 0)
                continue;

            if (Vector3.Dot(faces[currentFace].normal, faces[neighbor].normal) < patchDot)
                continue;

            facePatchIds[neighbor] = patchId;
            queue.Enqueue(neighbor);
        }
    }

    private void ExpandTriangleNeighbor(int currentFace, EdgeKey edge, Dictionary<EdgeKey, List<int>> edgeToFaces, TriangleFaceInfo[] faces, int[] faceRegionIds, int regionId, Queue<int> queue)
    {
        if (!edgeToFaces.TryGetValue(edge, out List<int> sharedFaces))
            return;

        for (int i = 0; i < sharedFaces.Count; i++)
        {
            int neighborFace = sharedFaces[i];
            if (neighborFace == currentFace || faceRegionIds[neighborFace] >= 0)
                continue;

            if (!MeshTriangleBoundaryOk(faces[currentFace], faces[neighborFace]))
                continue;

            faceRegionIds[neighborFace] = regionId;
            queue.Enqueue(neighborFace);
        }
    }

    private bool MeshTriangleBoundaryOk(TriangleFaceInfo a, TriangleFaceInfo b)
    {
        float minNormalDot = Mathf.Clamp(surfacePlaneClusterNormalDot, 0.5f, 0.999f);
        if (Vector3.Dot(a.normal, b.normal) < minNormalDot)
            return false;

        float maxOffsetDelta = Mathf.Max(0.001f, surfacePlaneClusterOffsetMeters);
        if (Mathf.Abs(a.planeOffset - b.planeOffset) > maxOffsetDelta)
            return false;

        float planeOffset = Mathf.Max(
            Mathf.Abs(Vector3.Dot(b.center - a.center, a.normal)),
            Mathf.Abs(Vector3.Dot(a.center - b.center, b.normal)));
        if (planeOffset > Mathf.Max(0.001f, surfaceRegionMaxPlaneOffsetMeters))
            return false;

        return true;
    }

    private SurfaceQuadRegionInfo BuildSurfaceQuadRegions(bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        int totalQuadCount = 0;
        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            totalQuadCount += Mathf.Max(0, group.columns - 1) * Mathf.Max(0, group.rows - 1);
        }

        int[] regionIds = new int[totalQuadCount];
        Array.Fill(regionIds, -1);
        Queue<int> queue = new Queue<int>();
        int nextRegionId = 0;
        int quadOffset = 0;

        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            int quadColumns = Mathf.Max(0, group.columns - 1);
            int quadRows = Mathf.Max(0, group.rows - 1);
            if (quadColumns <= 0 || quadRows <= 0)
                continue;

            for (int row = 0; row < quadRows; row++)
            {
                for (int col = 0; col < quadColumns; col++)
                {
                    int quadIndex = quadOffset + row * quadColumns + col;
                    if (regionIds[quadIndex] >= 0 || !IsSurfaceQuadUsable(group, row, col, valid, positions, normals, confidences))
                        continue;

                    regionIds[quadIndex] = nextRegionId;
                    queue.Enqueue(quadIndex);

                    while (queue.Count > 0)
                    {
                        int currentQuad = queue.Dequeue();
                        int localQuadIndex = currentQuad - quadOffset;
                        int currentRow = localQuadIndex / quadColumns;
                        int currentCol = localQuadIndex % quadColumns;

                        TryExpandSurfaceQuadNeighbor(currentRow, currentCol, currentRow, currentCol - 1, group, quadOffset, quadColumns, quadRows, valid, positions, normals, confidences, regionIds, nextRegionId, queue);
                        TryExpandSurfaceQuadNeighbor(currentRow, currentCol, currentRow, currentCol + 1, group, quadOffset, quadColumns, quadRows, valid, positions, normals, confidences, regionIds, nextRegionId, queue);
                        TryExpandSurfaceQuadNeighbor(currentRow, currentCol, currentRow - 1, currentCol, group, quadOffset, quadColumns, quadRows, valid, positions, normals, confidences, regionIds, nextRegionId, queue);
                        TryExpandSurfaceQuadNeighbor(currentRow, currentCol, currentRow + 1, currentCol, group, quadOffset, quadColumns, quadRows, valid, positions, normals, confidences, regionIds, nextRegionId, queue);
                    }

                    nextRegionId++;
                }
            }

            quadOffset += quadColumns * quadRows;
        }

        MergeSmallSurfaceQuadRegions(regionIds, valid, positions, normals, confidences);

        int[] regionColorIds = BuildSurfaceRegionColorIds(regionIds, positions);
        return new SurfaceQuadRegionInfo
        {
            regionIds = regionIds,
            totalRegionCount = CountActiveIds(regionIds),
            regionColorIds = regionColorIds,
            colorCount = CountActiveIds(regionColorIds)
        };
    }

    private int CountActiveIds(int[] ids)
    {
        if (ids == null || ids.Length == 0)
            return 0;

        HashSet<int> active = new HashSet<int>();
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] >= 0)
                active.Add(ids[i]);
        }

        return active.Count;
    }

    private void MergeSmallSurfaceQuadRegions(int[] regionIds, bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        if (regionIds == null || regionIds.Length == 0)
            return;

        int minimumCount = Mathf.Max(1, surfaceRegionMinQuadCount);
        if (minimumCount <= 1)
            return;

        Dictionary<int, int> regionCounts = new Dictionary<int, int>();
        for (int i = 0; i < regionIds.Length; i++)
        {
            int regionId = regionIds[i];
            if (regionId < 0)
                continue;

            regionCounts.TryGetValue(regionId, out int count);
            regionCounts[regionId] = count + 1;
        }

        int quadOffset = 0;
        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            int quadColumns = Mathf.Max(0, group.columns - 1);
            int quadRows = Mathf.Max(0, group.rows - 1);
            int quadCount = quadColumns * quadRows;
            if (quadCount <= 0)
                continue;

            for (int row = 0; row < quadRows; row++)
            {
                for (int col = 0; col < quadColumns; col++)
                {
                    int quadIndex = quadOffset + row * quadColumns + col;
                    int regionId = regionIds[quadIndex];
                    if (regionId < 0)
                        continue;

                    if (!regionCounts.TryGetValue(regionId, out int regionCount) || regionCount >= minimumCount)
                        continue;

                    int mergeTarget = FindBestSurfaceQuadMergeTarget(regionId, row, col, g, quadOffset, quadColumns, quadRows, regionIds, valid, positions, normals, confidences, regionCounts);
                    if (mergeTarget < 0 || mergeTarget == regionId)
                        continue;

                    regionIds[quadIndex] = mergeTarget;
                    regionCounts[regionId] = Mathf.Max(0, regionCounts[regionId] - 1);
                    regionCounts.TryGetValue(mergeTarget, out int targetCount);
                    regionCounts[mergeTarget] = targetCount + 1;
                }
            }

            quadOffset += quadCount;
        }
    }

    private int FindBestSurfaceQuadMergeTarget(int sourceRegionId, int row, int col, int groupIndex, int quadOffset, int quadColumns, int quadRows, int[] regionIds, bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences, Dictionary<int, int> regionCounts)
    {
        int bestTarget = -1;
        float bestScore = float.NegativeInfinity;
        ConsiderSurfaceQuadMergeTarget(sourceRegionId, row, col - 1, row, col, groupIndex, quadOffset, quadColumns, quadRows, regionIds, valid, positions, normals, confidences, regionCounts, ref bestTarget, ref bestScore);
        ConsiderSurfaceQuadMergeTarget(sourceRegionId, row, col + 1, row, col, groupIndex, quadOffset, quadColumns, quadRows, regionIds, valid, positions, normals, confidences, regionCounts, ref bestTarget, ref bestScore);
        ConsiderSurfaceQuadMergeTarget(sourceRegionId, row - 1, col, row, col, groupIndex, quadOffset, quadColumns, quadRows, regionIds, valid, positions, normals, confidences, regionCounts, ref bestTarget, ref bestScore);
        ConsiderSurfaceQuadMergeTarget(sourceRegionId, row + 1, col, row, col, groupIndex, quadOffset, quadColumns, quadRows, regionIds, valid, positions, normals, confidences, regionCounts, ref bestTarget, ref bestScore);
        return bestTarget;
    }

    private void ConsiderSurfaceQuadMergeTarget(int sourceRegionId, int neighborRow, int neighborCol, int row, int col, int groupIndex, int quadOffset, int quadColumns, int quadRows, int[] regionIds, bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences, Dictionary<int, int> regionCounts, ref int bestTarget, ref float bestScore)
    {
        if (neighborRow < 0 || neighborRow >= quadRows || neighborCol < 0 || neighborCol >= quadColumns)
            return;

        int neighborIndex = quadOffset + neighborRow * quadColumns + neighborCol;
        int neighborRegionId = regionIds[neighborIndex];
        if (neighborRegionId < 0 || neighborRegionId == sourceRegionId)
            return;

        if (!regionCounts.TryGetValue(neighborRegionId, out int neighborCount) || neighborCount <= 0)
            return;

        GridGroup group = _groups[groupIndex];
        if (!IsSurfaceQuadUsable(group, neighborRow, neighborCol, valid, positions, normals, confidences))
            return;

        if (!SurfaceQuadBoundaryOk(row, col, neighborRow, neighborCol, group, positions, normals))
            return;

        float score = neighborCount;
        if (TryGetSurfaceQuadNormal(group, row, col, positions, out Vector3 sourceNormal) &&
            TryGetSurfaceQuadNormal(group, neighborRow, neighborCol, positions, out Vector3 neighborNormal))
        {
            score += Vector3.Dot(sourceNormal, neighborNormal) * 100f;
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestTarget = neighborRegionId;
        }
    }

    private void TryExpandSurfaceQuadNeighbor(int currentRow, int currentCol, int neighborRow, int neighborCol, GridGroup group, int quadOffset, int quadColumns, int quadRows, bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences, int[] regionIds, int regionId, Queue<int> queue)
    {
        if (neighborRow < 0 || neighborRow >= quadRows || neighborCol < 0 || neighborCol >= quadColumns)
            return;

        int neighborQuad = quadOffset + neighborRow * quadColumns + neighborCol;
        if (regionIds[neighborQuad] >= 0)
            return;

        if (!IsSurfaceQuadUsable(group, neighborRow, neighborCol, valid, positions, normals, confidences))
            return;

        if (!SurfaceQuadBoundaryOk(currentRow, currentCol, neighborRow, neighborCol, group, positions, normals))
            return;

        regionIds[neighborQuad] = regionId;
        queue.Enqueue(neighborQuad);
    }

    private bool IsSurfaceQuadUsable(GridGroup group, int row, int col, bool[] valid, Vector3[] positions, Vector3[] normals, float[] confidences)
    {
        int i00 = group.startIndex + row * group.columns + col;
        int i10 = i00 + 1;
        int i01 = i00 + group.columns;
        int i11 = i01 + 1;
        int count = (valid[i00] ? 1 : 0) + (valid[i10] ? 1 : 0) + (valid[i01] ? 1 : 0) + (valid[i11] ? 1 : 0);
        if (count < 3)
            return false;

        if (count == 4)
        {
            bool diagA = EdgeOk(i00, i11, positions, normals, confidences);
            bool diagB = EdgeOk(i10, i01, positions, normals, confidences);
            return diagA || diagB;
        }

        return true;
    }

    private bool SurfaceQuadBoundaryOk(int rowA, int colA, int rowB, int colB, GridGroup group, Vector3[] positions, Vector3[] normals)
    {
        if (!TryGetSurfaceQuadCenter(group, rowA, colA, positions, out Vector3 centerA) ||
            !TryGetSurfaceQuadCenter(group, rowB, colB, positions, out Vector3 centerB))
            return false;

        float maxDistance = Mathf.Max(0.001f, surfaceRegionMaxNeighborDistanceMeters) * 1.5f;
        if (Vector3.Distance(centerA, centerB) > maxDistance)
            return false;

        float creaseDot = Mathf.Cos(Mathf.Clamp(surfaceRegionCreaseAngleDegrees, 1f, 89f) * Mathf.Deg2Rad);
        if (TryGetRegionEdgeCreaseDot(rowA, colA, rowB, colB, group, positions, out float edgeCreaseDot) && edgeCreaseDot < creaseDot)
            return false;

        if (TryGetSurfaceQuadNormal(group, rowA, colA, positions, out Vector3 normalA) &&
            TryGetSurfaceQuadNormal(group, rowB, colB, positions, out Vector3 normalB))
        {
            if (Vector3.Dot(normalA, normalB) < creaseDot)
                return false;

            float planeOffset = Mathf.Max(
                Mathf.Abs(Vector3.Dot(centerB - centerA, normalA)),
                Mathf.Abs(Vector3.Dot(centerA - centerB, normalB)));
            if (planeOffset > Mathf.Max(0.001f, surfaceRegionMaxPlaneOffsetMeters))
                return false;
        }

        return true;
    }

    private int GetSurfaceQuadRegionId(int[] regionIds, int groupIndex, int row, int col, GridGroup group)
    {
        if (regionIds == null)
            return -1;

        int offset = 0;
        for (int i = 0; i < groupIndex; i++)
        {
            GridGroup prior = _groups[i];
            offset += Mathf.Max(0, prior.columns - 1) * Mathf.Max(0, prior.rows - 1);
        }

        int quadColumns = Mathf.Max(0, group.columns - 1);
        if (quadColumns <= 0)
            return -1;

        int index = offset + row * quadColumns + col;
        if (index < 0 || index >= regionIds.Length)
            return -1;

        return regionIds[index];
    }

    private int GetSurfaceQuadColorId(SurfaceQuadRegionInfo info, int groupIndex, int row, int col, GridGroup group)
    {
        int regionId = GetSurfaceQuadRegionId(info.regionIds, groupIndex, row, col, group);
        if (regionId < 0 || info.regionColorIds == null || regionId >= info.regionColorIds.Length)
            return -1;

        return info.regionColorIds[regionId];
    }

    private int[] BuildSurfaceRegionColorIds(int[] regionIds, Vector3[] positions)
    {
        if (regionIds == null || regionIds.Length == 0)
            return Array.Empty<int>();

        int maxRegionId = -1;
        for (int i = 0; i < regionIds.Length; i++)
            maxRegionId = Mathf.Max(maxRegionId, regionIds[i]);
        if (maxRegionId < 0)
            return Array.Empty<int>();

        Vector3[] normalSums = new Vector3[maxRegionId + 1];
        Vector3[] centerSums = new Vector3[maxRegionId + 1];
        int[] counts = new int[maxRegionId + 1];

        int quadOffset = 0;
        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            int quadColumns = Mathf.Max(0, group.columns - 1);
            int quadRows = Mathf.Max(0, group.rows - 1);
            for (int row = 0; row < quadRows; row++)
            {
                for (int col = 0; col < quadColumns; col++)
                {
                    int quadIndex = quadOffset + row * quadColumns + col;
                    int regionId = regionIds[quadIndex];
                    if (regionId < 0)
                        continue;

                    if (!TryGetSurfaceQuadNormal(group, row, col, positions, out Vector3 quadNormal))
                        continue;
                    if (!TryGetSurfaceQuadCenter(group, row, col, positions, out Vector3 quadCenter))
                        continue;

                    normalSums[regionId] += quadNormal;
                    centerSums[regionId] += quadCenter;
                    counts[regionId]++;
                }
            }

            quadOffset += quadColumns * quadRows;
        }

        List<Vector4> clusterPlanes = new List<Vector4>();
        int[] colorIds = new int[maxRegionId + 1];
        for (int regionId = 0; regionId <= maxRegionId; regionId++)
        {
            colorIds[regionId] = -1;
            if (counts[regionId] <= 0 || normalSums[regionId].sqrMagnitude <= 1e-8f)
                continue;

            Vector3 normal = normalSums[regionId].normalized;
            Vector3 center = centerSums[regionId] / counts[regionId];
            float planeOffset = Vector3.Dot(normal, center);
            int clusterId = FindSurfacePlaneCluster(clusterPlanes, normal, planeOffset);
            if (clusterId < 0)
            {
                clusterId = clusterPlanes.Count;
                clusterPlanes.Add(new Vector4(normal.x, normal.y, normal.z, planeOffset));
            }

            colorIds[regionId] = clusterId;
        }

        return colorIds;
    }

    private int FindSurfacePlaneCluster(List<Vector4> clusterPlanes, Vector3 normal, float planeOffset)
    {
        float minNormalDot = Mathf.Clamp(surfacePlaneClusterNormalDot, 0.5f, 0.999f);
        float maxOffsetDelta = Mathf.Max(0.001f, surfacePlaneClusterOffsetMeters);
        for (int i = 0; i < clusterPlanes.Count; i++)
        {
            Vector3 clusterNormal = new Vector3(clusterPlanes[i].x, clusterPlanes[i].y, clusterPlanes[i].z);
            float normalDot = Vector3.Dot(clusterNormal, normal);
            if (normalDot < minNormalDot)
                continue;

            if (Mathf.Abs(clusterPlanes[i].w - planeOffset) > maxOffsetDelta)
                continue;

            return i;
        }

        return -1;
    }

    private bool TryGetSurfaceQuadCenter(GridGroup group, int row, int col, Vector3[] positions, out Vector3 center)
    {
        center = Vector3.zero;
        int sampleCount = 0;
        if (TryGetGroupPoint(group, row, col, positions, out Vector3 p00)) { center += p00; sampleCount++; }
        if (TryGetGroupPoint(group, row, col + 1, positions, out Vector3 p10)) { center += p10; sampleCount++; }
        if (TryGetGroupPoint(group, row + 1, col, positions, out Vector3 p01)) { center += p01; sampleCount++; }
        if (TryGetGroupPoint(group, row + 1, col + 1, positions, out Vector3 p11)) { center += p11; sampleCount++; }
        if (sampleCount <= 0)
            return false;

        center /= sampleCount;
        return true;
    }

    private bool TryGetSurfaceQuadNormal(GridGroup group, int row, int col, Vector3[] positions, out Vector3 quadNormal)
    {
        quadNormal = Vector3.zero;
        List<Vector3> triangleNormals = new List<Vector3>(4);

        if (TryGetGroupPoint(group, row, col, positions, out Vector3 p00) &&
            TryGetGroupPoint(group, row, col + 1, positions, out Vector3 p10) &&
            TryGetGroupPoint(group, row + 1, col, positions, out Vector3 p01))
            TryAddTriangleNormal(p00, p10, p01, triangleNormals);

        if (TryGetGroupPoint(group, row, col + 1, positions, out p10) &&
            TryGetGroupPoint(group, row + 1, col + 1, positions, out Vector3 p11) &&
            TryGetGroupPoint(group, row + 1, col, positions, out p01))
            TryAddTriangleNormal(p10, p11, p01, triangleNormals);

        if (triangleNormals.Count <= 0)
            return false;

        Vector3 sum = Vector3.zero;
        for (int i = 0; i < triangleNormals.Count; i++)
            sum += triangleNormals[i];

        if (sum.sqrMagnitude <= 1e-8f)
            return false;

        quadNormal = sum.normalized;
        return true;
    }

    private bool TryGetRegionEdgeCreaseDot(int rowA, int colA, int rowB, int colB, GridGroup group, Vector3[] positions, out float edgeCreaseDot)
    {
        edgeCreaseDot = 1f;
        bool horizontalNeighbor = rowA == rowB && Mathf.Abs(colA - colB) == 1;
        bool verticalNeighbor = colA == colB && Mathf.Abs(rowA - rowB) == 1;
        if (!horizontalNeighbor && !verticalNeighbor)
            return false;

        if (horizontalNeighbor)
        {
            int row = rowA;
            int leftCol = Mathf.Min(colA, colB);
            bool hasAbove = TryComputeEdgeSideNormal(group, row, leftCol, true, true, positions, out Vector3 aboveNormal);
            bool hasBelow = TryComputeEdgeSideNormal(group, row, leftCol, true, false, positions, out Vector3 belowNormal);
            if (!hasAbove || !hasBelow)
                return false;

            edgeCreaseDot = Vector3.Dot(aboveNormal, belowNormal);
            return true;
        }

        int topRow = Mathf.Min(rowA, rowB);
        int col = colA;
        bool hasLeft = TryComputeEdgeSideNormal(group, topRow, col, false, true, positions, out Vector3 leftNormal);
        bool hasRight = TryComputeEdgeSideNormal(group, topRow, col, false, false, positions, out Vector3 rightNormal);
        if (!hasLeft || !hasRight)
            return false;

        edgeCreaseDot = Vector3.Dot(leftNormal, rightNormal);
        return true;
    }

    private bool TryComputeEdgeSideNormal(GridGroup group, int anchorRow, int anchorCol, bool horizontalEdge, bool primarySide, Vector3[] positions, out Vector3 sideNormal)
    {
        sideNormal = Vector3.zero;
        List<Vector3> triangleNormals = new List<Vector3>(2);

        if (horizontalEdge)
        {
            int row0 = anchorRow;
            int col0 = anchorCol;
            int rowSide = primarySide ? row0 - 1 : row0 + 1;
            if (rowSide < 0 || rowSide >= group.rows)
                return false;

            if (!TryGetGroupPoint(group, row0, col0, positions, out Vector3 a) ||
                !TryGetGroupPoint(group, row0, col0 + 1, positions, out Vector3 b))
                return false;

            if (TryGetGroupPoint(group, rowSide, col0, positions, out Vector3 c0))
                TryAddTriangleNormal(a, b, c0, triangleNormals);
            if (TryGetGroupPoint(group, rowSide, col0 + 1, positions, out Vector3 c1))
                TryAddTriangleNormal(a, b, c1, triangleNormals);
        }
        else
        {
            int row0 = anchorRow;
            int col0 = anchorCol;
            int colSide = primarySide ? col0 - 1 : col0 + 1;
            if (colSide < 0 || colSide >= group.columns)
                return false;

            if (!TryGetGroupPoint(group, row0, col0, positions, out Vector3 a) ||
                !TryGetGroupPoint(group, row0 + 1, col0, positions, out Vector3 b))
                return false;

            if (TryGetGroupPoint(group, row0, colSide, positions, out Vector3 c0))
                TryAddTriangleNormal(a, b, c0, triangleNormals);
            if (TryGetGroupPoint(group, row0 + 1, colSide, positions, out Vector3 c1))
                TryAddTriangleNormal(a, b, c1, triangleNormals);
        }

        if (triangleNormals.Count <= 0)
            return false;

        Vector3 sum = Vector3.zero;
        for (int i = 0; i < triangleNormals.Count; i++)
            sum += triangleNormals[i];

        if (sum.sqrMagnitude <= 1e-8f)
            return false;

        sideNormal = sum.normalized;
        return true;
    }

    private bool TryGetGroupPoint(GridGroup group, int row, int col, Vector3[] positions, out Vector3 point)
    {
        point = Vector3.zero;
        if (row < 0 || row >= group.rows || col < 0 || col >= group.columns)
            return false;

        int index = group.startIndex + row * group.columns + col;
        if (index < 0 || index >= positions.Length)
            return false;

        point = positions[index];
        return Finite(point);
    }

    private void TryAddTriangleNormal(Vector3 a, Vector3 b, Vector3 c, List<Vector3> triangleNormals)
    {
        Vector3 normal = Vector3.Cross(b - a, c - a);
        if (normal.sqrMagnitude <= 1e-8f)
            return;

        triangleNormals.Add(normal.normalized);
    }

    private bool TryGetNeighborSample(int centerX, int centerY, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 pos, out Vector3 normal, out float confidence)
        => TryGetNeighborSample(centerX, centerY, width, height, worldPositions, worldNormals, observationMeta, validScratch, Mathf.Max(1, neighborRadiusPixels), out pos, out normal, out confidence);

    private bool TryGetNeighborSample(int centerX, int centerY, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, int radiusMax, out Vector3 pos, out Vector3 normal, out float confidence)
    {
        radiusMax = Mathf.Max(1, radiusMax);
        for (int radius = 1; radius <= radiusMax; radius++) for (int dy = -radius; dy <= radius; dy++) for (int dx = -radius; dx <= radius; dx++)
        {
            if (dx == 0 && dy == 0) continue;
            int x = centerX + dx, y = centerY + dy; if ((uint)x >= (uint)width || (uint)y >= (uint)height) continue;
            int index = x + y * width; if (index < 0 || index >= validScratch.Count || !validScratch[index]) continue;
            pos = WorldPos(worldPositions[index]); normal = WorldNormal(worldNormals[index], observationMeta[index].a >= 0.5f); confidence = Confidence(observationMeta[index]); return true;
        }
        pos = default; normal = default; confidence = 0f; return false;
    }

    private bool IsSampleUsable(int index, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta)
    {
        Color meta = observationMeta[index];
        if (meta.r < 0.5f || meta.g < minConfidence) return false;
        if (meta.b < minLinearDepthMeters || meta.b > Mathf.Max(minLinearDepthMeters, maxLinearDepthMeters)) return false;
        if (requireValidNormal && meta.a < 0.5f) return false;
        if (!Finite(WorldPos(worldPositions[index]))) return false;
        if (meta.a >= 0.5f && !Finite(WorldNormal(worldNormals[index], true))) return false;
        return true;
    }

    private bool IsSampleUsableForGridLine(int index, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta)
    {
        Color meta = observationMeta[index];
        if (meta.r < 0.5f || meta.g < minConfidence) return false;
        if (meta.b < minLinearDepthMeters) return false;
        if (requireValidNormal && meta.a < 0.5f) return false;
        if (!Finite(WorldPos(worldPositions[index]))) return false;
        if (meta.a >= 0.5f && !Finite(WorldNormal(worldNormals[index], true))) return false;
        return true;
    }


    private void ResolveRefs()
    {
        if (preprocessor == null)
        {
            ScanCoverDepthPreprocessor[] preprocessors = FindObjectsByType<ScanCoverDepthPreprocessor>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            for (int i = 0; i < preprocessors.Length; i++) if (preprocessors[i] != null && preprocessors[i].CurrentSourceEye == ScanCoverDepthPreprocessor.SourceEye.Right) { preprocessor = preprocessors[i]; break; }
            if (preprocessor == null && preprocessors.Length > 0) preprocessor = preprocessors[0];
        }
        if (viewLockedOrigin == null) { Camera mainCamera = Camera.main; if (mainCamera != null) viewLockedOrigin = mainCamera.transform; }
    }

    private Transform ResolveViewLockedOrigin()
    {
        if (viewLockedOrigin != null) return viewLockedOrigin;
        Camera mainCamera = Camera.main; if (mainCamera != null) return mainCamera.transform;
        if (preprocessor != null) return preprocessor.transform;
        return transform;
    }

    private Quaternion ResolveVolumeRotation(Transform origin)
    {
        if (origin == null) return transform.rotation;
        Vector3 forward = origin.forward;
        if (volumeLockPitch || volumeLockRoll)
        {
            Vector3 flatForward = Vector3.ProjectOnPlane(forward, Vector3.up);
            if (flatForward.sqrMagnitude > 1e-6f) forward = flatForward.normalized;
            else forward = Vector3.forward;
            return Quaternion.LookRotation(forward, Vector3.up);
        }
        return origin.rotation;
    }

    private void EnsurePoolSize(int count)
    {
        while (_pool.Count < count)
        {
            GameObject go = markerPrefab != null ? Instantiate(markerPrefab, ResolveMarkerParent(), false) : CreateFallbackMarker();
            go.SetActive(false); _pool.Add(go); _rendererCache.Add(go.GetComponentsInChildren<Renderer>(true));
        }
    }

    private void UpdateMarker(int index, Vector3 position, Vector3 normal, float confidence)
    {
        if (index < 0 || index >= _pool.Count) return;
        GameObject go = _pool[index]; if (go == null) return;
        Transform markerTransform = go.transform;
        if (useWorldSpaceDisplayRoots && _runtimeMarkerRoot != null)
        {
            Transform localTransform = ResolveDisplayLocalTransform();
            markerTransform.localPosition = localTransform != null ? localTransform.InverseTransformPoint(position) : position;
            if (orientToNormal && normal.sqrMagnitude > 1e-6f)
            {
                Vector3 localNormal = localTransform != null ? localTransform.InverseTransformDirection(normal) : normal;
                localNormal = localNormal.sqrMagnitude > 1e-6f ? localNormal.normalized : Vector3.forward;
                markerTransform.localRotation = Quaternion.LookRotation(localNormal, ResolveLocalUp(localNormal, Vector3.up));
            }
        }
        else
        {
            markerTransform.position = position;
            if (orientToNormal && normal.sqrMagnitude > 1e-6f) markerTransform.rotation = Quaternion.LookRotation(normal, Vector3.up);
        }
        ApplyColor(index, snapshotGridUniformColor);
        if (!go.activeSelf) go.SetActive(true);
    }

    private void DisableMarker(int index) { if (index < 0 || index >= _pool.Count) return; GameObject go = _pool[index]; if (go != null && go.activeSelf) go.SetActive(false); }
    private void HideAllMarkers() { for (int i = 0; i < _pool.Count; i++) DisableMarker(i); }

    private void DumpRosterIfNeeded(bool[] valid, bool[] lineValid, Vector3[] positions, Vector3[] normals, float[] confidences, Vector2Int resolution)
    {
        if (!dumpRosterOnceOnPlay || _hasDumpedRoster || !Application.isPlaying || _cells.Count <= 0)
            return;

        StringBuilder builder = new StringBuilder(_cells.Count * 96);
        builder.AppendLine("# ScanCover Depth Grid Point Cloud Roster");
        builder.Append("samplingMode=").Append(samplingMode).AppendLine();
        builder.Append("preprocessorOutput=").Append(resolution.x).Append('x').Append(resolution.y).AppendLine();
        builder.Append("cellCount=").Append(_cells.Count).AppendLine();
        builder.Append("visibleCount=").Append(_visibleCount).AppendLine();
        builder.AppendLine("index,row,col,group,face,centerX,centerY,meshValid,lineValid,confidence,posX,posY,posZ,normX,normY,normZ");

        for (int i = 0; i < _cells.Count; i++)
        {
            if (dumpOnlyValidCellsInRoster && (valid == null || i >= valid.Length || !valid[i]))
                continue;

            Cell cell = _cells[i];
            bool isValid = valid != null && i < valid.Length && valid[i];
            bool isLineValid = lineValid != null && i < lineValid.Length && lineValid[i];
            Vector3 pos = isValid && positions != null && i < positions.Length ? positions[i] : Vector3.zero;
            Vector3 normal = isValid && normals != null && i < normals.Length ? normals[i] : Vector3.zero;
            float confidence = isValid && confidences != null && i < confidences.Length ? confidences[i] : 0f;

            builder.Append(i).Append(',')
                .Append(cell.row).Append(',')
                .Append(cell.col).Append(',')
                .Append(cell.group).Append(',')
                .Append(cell.face).Append(',')
                .Append(cell.centerX).Append(',')
                .Append(cell.centerY).Append(',')
                .Append(isValid ? 1 : 0).Append(',')
                .Append(isLineValid ? 1 : 0).Append(',')
                .Append(confidence.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
                .Append(pos.x.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
                .Append(pos.y.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
                .Append(pos.z.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
                .Append(normal.x.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
                .Append(normal.y.ToString("F5", CultureInfo.InvariantCulture)).Append(',')
                .Append(normal.z.ToString("F5", CultureInfo.InvariantCulture)).AppendLine();
        }

        string exportDirectory = ResolveDebugExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        string path = Path.Combine(exportDirectory, $"DepthGridRoster_{System.DateTime.Now:yyyyMMdd_HHmmss}.txt");
        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
        _hasDumpedRoster = true;
        Debug.Log($"[ScanCoverDepthGridPointCloud] Roster exported => {path}");
    }

    private static string ResolveDebugExportDirectory()
    {
        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName ?? Application.dataPath;
        return Path.Combine(projectRoot, "ScanCoverExports");
    }

    public bool ExportCurrentGridStateAsJson(out string exportPath)
    {
        exportPath = null;
        if (!TryGetCurrentGridState(out GridStateSnapshot snapshot))
            return false;

        string exportDirectory = ResolveDebugExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        exportPath = Path.Combine(exportDirectory, $"GridState_{System.DateTime.Now:yyyyMMdd_HHmmss}.json");
        File.WriteAllText(exportPath, JsonUtility.ToJson(snapshot, true), Encoding.UTF8);
        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridPointCloud] Grid state exported => {exportPath}");
        return true;
    }

    [ContextMenu("Export Current Grid Nodes As CSV")]
    public void ExportCurrentGridNodesAsCsvFromContextMenu() => ExportCurrentGridNodesAsCsv(out _);

    public bool ExportCurrentGridNodesAsCsv(out string exportPath)
    {
        exportPath = null;
        if (_cells.Count <= 0 || _currentPositions == null || _currentPositions.Length != _cells.Count)
            return SetIssue("Grid node state is not ready.");

        string exportDirectory = ResolveDebugExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        exportPath = Path.Combine(exportDirectory, $"ScanCover_DepthGridNodes_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}.csv");

        Transform localTransform = ResolveDisplayLocalTransform();
        StringBuilder builder = new StringBuilder(Mathf.Max(1024, _cells.Count * 240));
        builder.AppendLine("# ScanCover Depth Grid Nodes");
        builder.Append("component=").Append(name).AppendLine();
        builder.Append("samplingMode=").Append(samplingMode).AppendLine();
        builder.Append("frameIndex=").Append(_frameIndex).AppendLine();
        builder.Append("preprocessorOutput=").Append(_currentResolution.x).Append('x').Append(_currentResolution.y).AppendLine();
        builder.Append("cellCount=").Append(_cells.Count).AppendLine();
        builder.Append("visibleCount=").Append(_visibleCount).AppendLine();
        builder.AppendLine("index,group,row,col,face,minPixelX,maxPixelX,minPixelY,maxPixelY,centerPixelX,centerPixelY,centerPixelXF,centerPixelYF,hasSubpixelCenter,meshValid,lineValid,hasPosition,confidence,worldX,worldY,worldZ,localX,localY,localZ,normalX,normalY,normalZ,leftDistance,rightDistance,downDistance,upDistance,leftNormalDelta,rightNormalDelta,downNormalDelta,upNormalDelta,leftVerticalDelta,rightVerticalDelta,downVerticalDelta,upVerticalDelta");

        for (int i = 0; i < _cells.Count; i++)
        {
            Cell cell = _cells[i];
            bool meshValid = _currentValid != null && i < _currentValid.Length && _currentValid[i];
            bool lineValid = _currentLineValid != null && i < _currentLineValid.Length ? _currentLineValid[i] : meshValid;
            bool hasPosition = lineValid && i < _currentPositions.Length && Finite(_currentPositions[i]);
            Vector3 world = hasPosition ? _currentPositions[i] : Vector3.zero;
            Vector3 local = hasPosition && localTransform != null ? localTransform.InverseTransformPoint(world) : world;
            Vector3 normal = hasPosition && i < _currentNormals.Length && Finite(_currentNormals[i]) ? _currentNormals[i] : Vector3.zero;
            float confidence = hasPosition && i < _currentConfidences.Length ? _currentConfidences[i] : 0f;

            TryGetNeighborMetrics(i, 0, -1, out float leftDistance, out float leftNormalDelta, out float leftVerticalDelta);
            TryGetNeighborMetrics(i, 0, 1, out float rightDistance, out float rightNormalDelta, out float rightVerticalDelta);
            TryGetNeighborMetrics(i, -1, 0, out float downDistance, out float downNormalDelta, out float downVerticalDelta);
            TryGetNeighborMetrics(i, 1, 0, out float upDistance, out float upNormalDelta, out float upVerticalDelta);

            builder.Append(i).Append(',')
                .Append(cell.group).Append(',')
                .Append(cell.row).Append(',')
                .Append(cell.col).Append(',')
                .Append(cell.face).Append(',')
                .Append(cell.minX).Append(',')
                .Append(cell.maxX).Append(',')
                .Append(cell.minY).Append(',')
                .Append(cell.maxY).Append(',')
                .Append(cell.centerX).Append(',')
                .Append(cell.centerY).Append(',')
                .Append(FormatFloat(cell.hasSubpixelCenter ? cell.centerXF : cell.centerX)).Append(',')
                .Append(FormatFloat(cell.hasSubpixelCenter ? cell.centerYF : cell.centerY)).Append(',')
                .Append(cell.hasSubpixelCenter ? 1 : 0).Append(',')
                .Append(meshValid ? 1 : 0).Append(',')
                .Append(lineValid ? 1 : 0).Append(',')
                .Append(hasPosition ? 1 : 0).Append(',')
                .Append(FormatFloat(confidence)).Append(',')
                .Append(FormatVector(world)).Append(',')
                .Append(FormatVector(local)).Append(',')
                .Append(FormatVector(normal)).Append(',')
                .Append(FormatNullableFloat(leftDistance)).Append(',')
                .Append(FormatNullableFloat(rightDistance)).Append(',')
                .Append(FormatNullableFloat(downDistance)).Append(',')
                .Append(FormatNullableFloat(upDistance)).Append(',')
                .Append(FormatNullableFloat(leftNormalDelta)).Append(',')
                .Append(FormatNullableFloat(rightNormalDelta)).Append(',')
                .Append(FormatNullableFloat(downNormalDelta)).Append(',')
                .Append(FormatNullableFloat(upNormalDelta)).Append(',')
                .Append(FormatNullableFloat(leftVerticalDelta)).Append(',')
                .Append(FormatNullableFloat(rightVerticalDelta)).Append(',')
                .Append(FormatNullableFloat(downVerticalDelta)).Append(',')
                .Append(FormatNullableFloat(upVerticalDelta)).AppendLine();
        }

        File.WriteAllText(exportPath, builder.ToString(), Encoding.UTF8);
        LastIssue = null;
        Debug.Log($"[ScanCoverDepthGridPointCloud] Grid node CSV exported => {exportPath}");
        return true;
    }

    [ContextMenu("Export Current BL Surface Mesh As OBJ")]
    public void ExportCurrentSurfaceMeshAsObjFromContextMenu() => ExportCurrentSurfaceMeshAsObj(out _);

    private void CacheRawSurfaceMeshForExport(List<Vector3> vertices, List<Vector3> normals, List<int> triangles)
    {
        if (vertices == null || vertices.Count <= 0 || triangles == null || triangles.Count < 3)
        {
            ClearRawSurfaceExportCache();
            return;
        }

        _rawSurfaceExportVertices = vertices.ToArray();
        _rawSurfaceExportNormals = normals != null && normals.Count == vertices.Count ? normals.ToArray() : Array.Empty<Vector3>();
        _rawSurfaceExportTriangles = triangles.ToArray();
    }

    private void ClearRawSurfaceExportCache()
    {
        _rawSurfaceExportVertices = Array.Empty<Vector3>();
        _rawSurfaceExportNormals = Array.Empty<Vector3>();
        _rawSurfaceExportTriangles = Array.Empty<int>();
    }

    public int CurrentSurfaceMeshFrameIndex => _frameIndex;

    public bool TryGetLatestRawDepthFrameSnapshot(out RawDepthFrameSnapshot snapshot)
    {
        snapshot = null;
        if (_latestRawDepthFrameSnapshot == null ||
            _latestRawDepthFrameSnapshot.worldPositions == null ||
            _latestRawDepthFrameSnapshot.worldPositions.Length <= 0)
            return false;

        snapshot = new RawDepthFrameSnapshot
        {
            componentName = _latestRawDepthFrameSnapshot.componentName,
            frameIndex = _latestRawDepthFrameSnapshot.frameIndex,
            resolutionWidth = _latestRawDepthFrameSnapshot.resolutionWidth,
            resolutionHeight = _latestRawDepthFrameSnapshot.resolutionHeight,
            snapshotRealtimeSeconds = _latestRawDepthFrameSnapshot.snapshotRealtimeSeconds,
            hasSnapshotCameraPose = _latestRawDepthFrameSnapshot.hasSnapshotCameraPose,
            snapshotCameraPosition = _latestRawDepthFrameSnapshot.snapshotCameraPosition,
            snapshotCameraRotation = _latestRawDepthFrameSnapshot.snapshotCameraRotation,
            sourceEyeIndex = _latestRawDepthFrameSnapshot.sourceEyeIndex,
            dispatchRealtimeSeconds = _latestRawDepthFrameSnapshot.dispatchRealtimeSeconds,
            completionRealtimeSeconds = _latestRawDepthFrameSnapshot.completionRealtimeSeconds,
            hasDispatchCameraPose = _latestRawDepthFrameSnapshot.hasDispatchCameraPose,
            dispatchCameraPosition = _latestRawDepthFrameSnapshot.dispatchCameraPosition,
            dispatchCameraRotation = _latestRawDepthFrameSnapshot.dispatchCameraRotation,
            hasCompletionCameraPose = _latestRawDepthFrameSnapshot.hasCompletionCameraPose,
            completionCameraPosition = _latestRawDepthFrameSnapshot.completionCameraPosition,
            completionCameraRotation = _latestRawDepthFrameSnapshot.completionCameraRotation,
            hasProjectionMatrix = _latestRawDepthFrameSnapshot.hasProjectionMatrix,
            projectionMatrix = _latestRawDepthFrameSnapshot.projectionMatrix,
            hasWorldToCameraMatrix = _latestRawDepthFrameSnapshot.hasWorldToCameraMatrix,
            worldToCameraMatrix = _latestRawDepthFrameSnapshot.worldToCameraMatrix,
            hasDepthReprojectionMatrix = _latestRawDepthFrameSnapshot.hasDepthReprojectionMatrix,
            depthReprojectionMatrix = _latestRawDepthFrameSnapshot.depthReprojectionMatrix,
            hasDispatchEyePosition = _latestRawDepthFrameSnapshot.hasDispatchEyePosition,
            dispatchEyePosition = _latestRawDepthFrameSnapshot.dispatchEyePosition,
            worldPositionsRaw = _latestRawDepthFrameSnapshot.worldPositionsRaw != null
                ? (Vector3[])_latestRawDepthFrameSnapshot.worldPositionsRaw.Clone()
                : null,
            worldPositions = (Vector3[])_latestRawDepthFrameSnapshot.worldPositions.Clone(),
            worldNormals = (Vector3[])_latestRawDepthFrameSnapshot.worldNormals.Clone(),
            worldNormalsNeighbour = _latestRawDepthFrameSnapshot.worldNormalsNeighbour != null
                ? (Vector3[])_latestRawDepthFrameSnapshot.worldNormalsNeighbour.Clone()
                : null,
            worldNormalsNeighbourValid = _latestRawDepthFrameSnapshot.worldNormalsNeighbourValid != null
                ? (bool[])_latestRawDepthFrameSnapshot.worldNormalsNeighbourValid.Clone()
                : null,
            observationMeta = (Color[])_latestRawDepthFrameSnapshot.observationMeta.Clone()
        };
        return true;
    }

    public bool ExportCurrentSurfaceMeshFramePackage(
        string frameDirectory,
        string frameName,
        Camera captureCamera,
        Transform poseSource,
        out string objPath,
        out string verticesCsvPath,
        out string trianglesCsvPath,
        out string cameraJsonPath)
    {
        objPath = null;
        verticesCsvPath = null;
        trianglesCsvPath = null;
        cameraJsonPath = null;

        if (string.IsNullOrWhiteSpace(frameDirectory))
            return SetIssue("Frame export directory is empty.");
        if (string.IsNullOrWhiteSpace(frameName))
            frameName = $"frame_{_frameIndex:000000}";

        if (!TryGetSurfaceMeshExportData(out Vector3[] vertices, out Vector3[] normals, out int[] triangles, out Transform surfaceTransform, out string sourceLabel))
            return false;

        Directory.CreateDirectory(frameDirectory);
        objPath = Path.Combine(frameDirectory, frameName + ".obj");
        verticesCsvPath = Path.Combine(frameDirectory, frameName + "_vertices.csv");
        trianglesCsvPath = Path.Combine(frameDirectory, frameName + "_triangles.csv");
        cameraJsonPath = Path.Combine(frameDirectory, frameName + "_camera.json");

        WriteSurfaceMeshObj(objPath, frameName, vertices, normals, triangles, surfaceTransform, sourceLabel);
        WriteSurfaceVerticesCsv(verticesCsvPath, vertices, normals, surfaceTransform);
        WriteSurfaceTrianglesCsv(trianglesCsvPath, vertices, normals, triangles, surfaceTransform);
        WriteCaptureCameraJson(cameraJsonPath, frameName, captureCamera, poseSource, surfaceTransform, sourceLabel, vertices.Length, triangles.Length / 3);

        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridPointCloud] Surface frame package exported => {frameDirectory}");
        return true;
    }

    private bool TryGetSurfaceMeshExportData(out Vector3[] vertices, out Vector3[] normals, out int[] triangles, out Transform surfaceTransform, out string sourceLabel)
    {
        vertices = null;
        normals = null;
        triangles = null;
        surfaceTransform = null;
        sourceLabel = null;

        bool hasRawExportCache = _rawSurfaceExportVertices != null && _rawSurfaceExportVertices.Length > 0 &&
                                 _rawSurfaceExportTriangles != null && _rawSurfaceExportTriangles.Length >= 3;
        if (!hasRawExportCache && (_surfaceMesh == null || _surfaceMesh.vertexCount <= 0))
            return SetIssue("BL surface mesh is not ready.");

        vertices = hasRawExportCache ? (Vector3[])_rawSurfaceExportVertices.Clone() : _surfaceMesh.vertices;
        triangles = hasRawExportCache ? (int[])_rawSurfaceExportTriangles.Clone() : _surfaceMesh.triangles;
        if (vertices == null || vertices.Length <= 0 || triangles == null || triangles.Length < 3)
            return SetIssue("BL surface mesh has no exportable triangles.");

        Vector3[] sourceNormals = hasRawExportCache ? _rawSurfaceExportNormals : _surfaceMesh.normals;
        normals = BuildSurfaceMeshExportNormals(vertices, triangles, sourceNormals);
        surfaceTransform = _surfaceRoot != null ? _surfaceRoot.transform : ResolveDisplayLocalTransform();
        sourceLabel = hasRawExportCache ? "raw-bl-surface-cache" : "display-surface-mesh-fallback";
        return true;
    }

    public bool ExportCurrentSurfaceMeshAsObj(out string exportPath)
    {
        exportPath = null;
        if (!TryGetSurfaceMeshExportData(out Vector3[] vertices, out Vector3[] normals, out int[] triangles, out Transform surfaceTransform, out string sourceLabel))
            return false;

        string exportDirectory = ResolveDebugExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        exportPath = Path.Combine(exportDirectory, $"ScanCover_BLSurfaceMesh_{timestamp}.obj");
        string manifestPath = Path.ChangeExtension(exportPath, ".txt");

        WriteSurfaceMeshObj(exportPath, "ScanCover_BLSurfaceMesh", vertices, normals, triangles, surfaceTransform, sourceLabel);

        StringBuilder manifest = new StringBuilder(512);
        manifest.AppendLine("ScanCover BL Surface Mesh Export");
        manifest.Append("obj=").Append(exportPath).AppendLine();
        manifest.Append("component=").Append(name).AppendLine();
        manifest.Append("frameIndex=").Append(_frameIndex).AppendLine();
        manifest.Append("vertexCount=").Append(vertices.Length).AppendLine();
        manifest.Append("triangleCount=").Append(triangles.Length / 3).AppendLine();
        manifest.Append("source=").Append(sourceLabel).AppendLine();
        manifest.Append("coordinateSpace=").Append(surfaceTransform != null ? "world" : "mesh-local").AppendLine();
        manifest.Append("showSurfaceMesh=").Append(showSurfaceMesh).AppendLine();
        manifest.Append("keepSurfaceMeshAvailableWhenHidden=").Append(keepSurfaceMeshAvailableWhenHidden).AppendLine();
        manifest.Append("useIndexConnectivity=").Append(useIndexConnectivity).AppendLine();
        manifest.Append("maxEdgeLengthMeters=").Append(FormatFloat(maxEdgeLengthMeters)).AppendLine();
        manifest.Append("minNeighborNormalDot=").Append(FormatFloat(minNeighborNormalDot)).AppendLine();
        File.WriteAllText(manifestPath, manifest.ToString(), Encoding.UTF8);

        LastIssue = null;
        Debug.Log($"[ScanCoverDepthGridPointCloud] BL surface mesh OBJ exported => {exportPath}");
        return true;
    }

    private void WriteSurfaceMeshObj(string path, string objectName, Vector3[] vertices, Vector3[] normals, int[] triangles, Transform surfaceTransform, string sourceLabel)
    {
        StringBuilder obj = new StringBuilder(Mathf.Max(4096, vertices.Length * 96 + triangles.Length * 24));
        obj.AppendLine("# ScanCover BL Surface Mesh OBJ");
        obj.Append("# component=").Append(name).AppendLine();
        obj.Append("# frameIndex=").Append(_frameIndex).AppendLine();
        obj.Append("# vertexCount=").Append(vertices.Length).AppendLine();
        obj.Append("# triangleCount=").Append(triangles.Length / 3).AppendLine();
        obj.Append("# source=").Append(sourceLabel).AppendLine();
        obj.Append("o ").Append(SanitizeObjName(objectName)).AppendLine();

        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 p = surfaceTransform != null ? surfaceTransform.TransformPoint(vertices[i]) : vertices[i];
            obj.Append("v ")
                .Append(FormatObjFloat(p.x)).Append(' ')
                .Append(FormatObjFloat(p.y)).Append(' ')
                .Append(FormatObjFloat(p.z)).AppendLine();
        }

        for (int i = 0; i < normals.Length; i++)
        {
            Vector3 n = normals[i];
            if (surfaceTransform != null)
                n = surfaceTransform.TransformDirection(n);
            n = SafeNormalized(n, Vector3.up);
            obj.Append("vn ")
                .Append(FormatObjFloat(n.x)).Append(' ')
                .Append(FormatObjFloat(n.y)).Append(' ')
                .Append(FormatObjFloat(n.z)).AppendLine();
        }

        for (int i = 0; i + 2 < triangles.Length; i += 3)
        {
            int a = triangles[i] + 1;
            int b = triangles[i + 1] + 1;
            int c = triangles[i + 2] + 1;
            if (a <= 0 || b <= 0 || c <= 0 || a > vertices.Length || b > vertices.Length || c > vertices.Length)
                continue;
            obj.Append("f ")
                .Append(a).Append("//").Append(a).Append(' ')
                .Append(b).Append("//").Append(b).Append(' ')
                .Append(c).Append("//").Append(c).AppendLine();
        }

        File.WriteAllText(path, obj.ToString(), Encoding.UTF8);
    }

    private void WriteSurfaceVerticesCsv(string path, Vector3[] vertices, Vector3[] normals, Transform surfaceTransform)
    {
        StringBuilder builder = new StringBuilder(Mathf.Max(1024, vertices.Length * 160));
        builder.AppendLine("# ScanCover BL Surface Mesh Vertices");
        builder.Append("component=").Append(name).AppendLine();
        builder.Append("frameIndex=").Append(_frameIndex).AppendLine();
        builder.AppendLine("index,localX,localY,localZ,worldX,worldY,worldZ,normalLocalX,normalLocalY,normalLocalZ,normalWorldX,normalWorldY,normalWorldZ");

        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 local = vertices[i];
            Vector3 world = surfaceTransform != null ? surfaceTransform.TransformPoint(local) : local;
            Vector3 normalLocal = i < normals.Length ? SafeNormalized(normals[i], Vector3.up) : Vector3.up;
            Vector3 normalWorld = surfaceTransform != null ? SafeNormalized(surfaceTransform.TransformDirection(normalLocal), Vector3.up) : normalLocal;
            builder.Append(i).Append(',')
                .Append(FormatVector(local)).Append(',')
                .Append(FormatVector(world)).Append(',')
                .Append(FormatVector(normalLocal)).Append(',')
                .Append(FormatVector(normalWorld)).AppendLine();
        }

        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private void WriteSurfaceTrianglesCsv(string path, Vector3[] vertices, Vector3[] normals, int[] triangles, Transform surfaceTransform)
    {
        StringBuilder builder = new StringBuilder(Mathf.Max(1024, triangles.Length * 96));
        builder.AppendLine("# ScanCover BL Surface Mesh Triangles");
        builder.Append("component=").Append(name).AppendLine();
        builder.Append("frameIndex=").Append(_frameIndex).AppendLine();
        builder.AppendLine("triangle,index0,index1,index2,centerX,centerY,centerZ,normalX,normalY,normalZ,area,edge01,edge12,edge20");

        int triangleIndex = 0;
        for (int i = 0; i + 2 < triangles.Length; i += 3)
        {
            int ia = triangles[i];
            int ib = triangles[i + 1];
            int ic = triangles[i + 2];
            if ((uint)ia >= (uint)vertices.Length || (uint)ib >= (uint)vertices.Length || (uint)ic >= (uint)vertices.Length)
                continue;

            Vector3 a = surfaceTransform != null ? surfaceTransform.TransformPoint(vertices[ia]) : vertices[ia];
            Vector3 b = surfaceTransform != null ? surfaceTransform.TransformPoint(vertices[ib]) : vertices[ib];
            Vector3 c = surfaceTransform != null ? surfaceTransform.TransformPoint(vertices[ic]) : vertices[ic];
            Vector3 center = (a + b + c) / 3f;
            Vector3 normal = SafeNormalized(Vector3.Cross(b - a, c - a), Vector3.up);
            float area = Vector3.Cross(b - a, c - a).magnitude * 0.5f;
            builder.Append(triangleIndex).Append(',')
                .Append(ia).Append(',').Append(ib).Append(',').Append(ic).Append(',')
                .Append(FormatVector(center)).Append(',')
                .Append(FormatVector(normal)).Append(',')
                .Append(FormatFloat(area)).Append(',')
                .Append(FormatFloat(Vector3.Distance(a, b))).Append(',')
                .Append(FormatFloat(Vector3.Distance(b, c))).Append(',')
                .Append(FormatFloat(Vector3.Distance(c, a))).AppendLine();
            triangleIndex++;
        }

        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private void WriteCaptureCameraJson(
        string path,
        string frameName,
        Camera captureCamera,
        Transform poseSource,
        Transform surfaceTransform,
        string sourceLabel,
        int vertexCount,
        int triangleCount)
    {
        Transform pose = poseSource != null ? poseSource : captureCamera != null ? captureCamera.transform : null;
        StringBuilder builder = new StringBuilder(2048);
        builder.AppendLine("{");
        AppendJsonString(builder, "component", name, 1, true);
        AppendJsonString(builder, "frameName", frameName, 1, true);
        AppendJsonNumber(builder, "depthGridFrameIndex", _frameIndex, 1, true);
        AppendJsonNumber(builder, "unityFrameCount", Time.frameCount, 1, true);
        AppendJsonNumber(builder, "time", Time.time, 1, true);
        AppendJsonNumber(builder, "unscaledTime", Time.unscaledTime, 1, true);
        AppendJsonString(builder, "source", sourceLabel, 1, true);
        AppendJsonNumber(builder, "vertexCount", vertexCount, 1, true);
        AppendJsonNumber(builder, "triangleCount", triangleCount, 1, true);
        AppendJsonString(builder, "cameraName", captureCamera != null ? captureCamera.name : "", 1, true);
        AppendJsonString(builder, "poseSourceName", pose != null ? pose.name : "", 1, true);

        builder.Append("  \"pose\": ");
        AppendTransformJson(builder, pose);
        builder.AppendLine(",");

        builder.Append("  \"surfaceTransform\": ");
        AppendTransformJson(builder, surfaceTransform);
        builder.AppendLine(",");

        builder.Append("  \"camera\": ");
        AppendCameraJson(builder, captureCamera);
        builder.AppendLine();
        builder.AppendLine("}");
        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private void ExportSurfaceMeshObjIfRequested()
    {
        if (!_exportSurfaceMeshObjAfterNextBuild)
            return;

        if (ExportCurrentSurfaceMeshAsObj(out _))
            _exportSurfaceMeshObjAfterNextBuild = false;
    }

    public bool TryGetCurrentGridState(out GridStateSnapshot snapshot)
    {
        snapshot = null;
        if (_cells.Count <= 0 || _currentValid == null || _currentValid.Length != _cells.Count)
            return SetIssue("Grid state is not ready.");

        GridStateEntry[] entries = new GridStateEntry[_cells.Count];
        for (int i = 0; i < _cells.Count; i++)
        {
            Cell cell = _cells[i];
            bool isValid = i < _currentValid.Length && _currentValid[i];
            entries[i] = new GridStateEntry
            {
                index = i,
                group = cell.group,
                row = cell.row,
                col = cell.col,
                valid = isValid,
                worldPos = isValid && i < _currentPositions.Length ? _currentPositions[i] : Vector3.zero,
                normal = isValid && i < _currentNormals.Length ? _currentNormals[i] : Vector3.zero,
                confidence = isValid && i < _currentConfidences.Length ? _currentConfidences[i] : 0f
            };
        }

        snapshot = new GridStateSnapshot
        {
            componentName = name,
            samplingMode = samplingMode.ToString(),
            frameIndex = _frameIndex,
            resolutionWidth = _currentResolution.x,
            resolutionHeight = _currentResolution.y,
            cellCount = _cells.Count,
            visibleCount = _visibleCount,
            entries = entries
        };
        LastIssue = null;
        return true;
    }

    public int CopyCurrentValidGridPositions(List<Vector3> target)
    {
        if (target == null)
            return 0;

        target.Clear();
        if (_cells.Count <= 0 || _currentValid == null || _currentPositions == null)
            return 0;

        if (target.Capacity < _visibleCount)
            target.Capacity = _visibleCount;
        int count = Mathf.Min(_cells.Count, Mathf.Min(_currentValid.Length, _currentPositions.Length));
        for (int i = 0; i < count; i++)
        {
            Vector3 position = _currentPositions[i];
            if (_currentValid[i] &&
                float.IsFinite(position.x) && float.IsFinite(position.y) && float.IsFinite(position.z))
                target.Add(position);
        }

        return target.Count;
    }

    public bool UsesPreprocessor(ScanCoverDepthPreprocessor targetPreprocessor)
    {
        return targetPreprocessor != null && preprocessor == targetPreprocessor;
    }

    public void SetKeepSurfaceMeshAvailableWhenHidden(bool enabled)
    {
        keepSurfaceMeshAvailableWhenHidden = enabled;
    }

    public void SetPreviewSurfaceMeshVisible(bool visible)
    {
        showSurfaceMesh = visible;
        if (_previewVisible)
            SetSurfaceVisible(ShouldShowSurfaceMesh() && _surfaceMesh != null && _surfaceMesh.vertexCount > 0);
    }

    public bool TryGetPreviewSurfaceData(out Mesh mesh, out Transform surfaceTransform)
    {
        mesh = null;
        surfaceTransform = null;
        if (_surfaceMesh == null || _surfaceMesh.vertexCount <= 0 || _surfaceMesh.triangles == null || _surfaceMesh.triangles.Length <= 0)
            return false;

        mesh = _surfaceMesh;
        surfaceTransform = _surfaceRoot != null ? _surfaceRoot.transform : null;
        return surfaceTransform != null;
    }

    public Material GetPreviewSurfaceMaterial()
    {
        return _surfaceRenderer != null ? _surfaceRenderer.sharedMaterial : null;
    }

    private bool TryPrepareOutputs(
        ScanCoverDepthPreprocessor targetPreprocessor,
        bool forcePreprocessorRefresh,
        out RenderTexture worldPosRawTex,
        out RenderTexture worldPosTex,
        out RenderTexture worldNormalTex,
        out RenderTexture worldNormalNeighbourTex,
        out RenderTexture metaTex,
        out Vector2Int outputResolution)
    {
        outputResolution = Vector2Int.zero;
        worldPosRawTex = null;
        worldPosTex = null;
        worldNormalTex = null;
        worldNormalNeighbourTex = null;
        metaTex = null;
        if (targetPreprocessor == null)
            return false;

        bool needsRefresh =
            forcePreprocessorRefresh ||
            !targetPreprocessor.isActiveAndEnabled ||
            !targetPreprocessor.IsReady;
        if (needsRefresh && !targetPreprocessor.RefreshNow())
            return SetIssue(targetPreprocessor.LastIssue ?? "Depth preprocessor refresh failed.");
        if (!targetPreprocessor.TryGetOutputs(out worldPosTex, out worldNormalTex, out metaTex))
        {
            if (!needsRefresh && !targetPreprocessor.RefreshNow())
                return SetIssue(targetPreprocessor.LastIssue ?? "Depth preprocessor refresh failed.");
            if (!targetPreprocessor.TryGetOutputs(out worldPosTex, out worldNormalTex, out metaTex))
                return SetIssue(targetPreprocessor.LastIssue ?? "Depth preprocessor outputs are unavailable.");
        }
        if (!targetPreprocessor.TryGetPaperShadowOutputs(
                out worldPosRawTex, out worldNormalNeighbourTex))
            return SetIssue(targetPreprocessor.LastIssue ?? "Paper DMC shadow inputs are unavailable.");

        outputResolution = targetPreprocessor.OutputResolution;
        return true;
    }

    private void RebuildValidScratch(List<bool> target, int pixelCount, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta)
    {
        target.Clear();
        if (target.Capacity < pixelCount) target.Capacity = pixelCount;
        for (int i = 0; i < pixelCount; i++) target.Add(IsSampleUsable(i, worldPositions, worldNormals, observationMeta));
    }

    private void RebuildGridLineValidScratch(List<bool> target, int pixelCount, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta)
    {
        target.Clear();
        if (target.Capacity < pixelCount) target.Capacity = pixelCount;
        for (int i = 0; i < pixelCount; i++) target.Add(IsSampleUsableForGridLine(i, worldPositions, worldNormals, observationMeta));
    }

    private void StoreCurrentGridState(bool[] valid, bool[] lineValid, Vector3[] positions, Vector3[] normals, float[] confidences, Vector2Int resolution)
    {
        _currentResolution = resolution;
        _currentValid = valid != null ? (bool[])valid.Clone() : Array.Empty<bool>();
        _currentLineValid = lineValid != null ? (bool[])lineValid.Clone() : Array.Empty<bool>();
        _currentPositions = positions != null ? (Vector3[])positions.Clone() : Array.Empty<Vector3>();
        _currentNormals = normals != null ? (Vector3[])normals.Clone() : Array.Empty<Vector3>();
        _currentConfidences = confidences != null ? (float[])confidences.Clone() : Array.Empty<float>();
    }

    private void StoreLatestRawDepthFrameSnapshot(
        NativeArray<Color> worldPositionsRaw,
        NativeArray<Color> worldPositions,
        NativeArray<Color> worldNormals,
        NativeArray<Color> worldNormalsNeighbour,
        NativeArray<Color> observationMeta,
        Vector2Int resolution)
    {
        int count = Mathf.Min(
            Mathf.Min(worldPositionsRaw.Length, worldPositions.Length),
            Mathf.Min(worldNormals.Length, Mathf.Min(worldNormalsNeighbour.Length, observationMeta.Length)));
        if (count <= 0)
        {
            _latestRawDepthFrameSnapshot = null;
            return;
        }

        Vector3[] rawPositions = new Vector3[count];
        Vector3[] positions = new Vector3[count];
        Vector3[] normals = new Vector3[count];
        Vector3[] neighbourNormals = new Vector3[count];
        bool[] neighbourNormalValid = new bool[count];
        Color[] meta = new Color[count];
        for (int i = 0; i < count; i++)
        {
            rawPositions[i] = WorldPos(worldPositionsRaw[i]);
            positions[i] = WorldPos(worldPositions[i]);
            normals[i] = WorldNormal(worldNormals[i], observationMeta[i].a >= 0.5f);
            neighbourNormalValid[i] = worldNormalsNeighbour[i].a >= 0.5f;
            neighbourNormals[i] = neighbourNormalValid[i]
                ? WorldNormal(worldNormalsNeighbour[i], true)
                : Vector3.zero;
            meta[i] = observationMeta[i];
        }
        _rawDepthSnapshotFrameIndex++;
        Camera snapshotCamera = Camera.main;
        double completionRealtimeSeconds = Time.realtimeSinceStartupAsDouble;

        _latestRawDepthFrameSnapshot = new RawDepthFrameSnapshot
        {
            componentName = name,
            frameIndex = _rawDepthSnapshotFrameIndex,
            resolutionWidth = resolution.x,
            resolutionHeight = resolution.y,
            snapshotRealtimeSeconds = Time.realtimeSinceStartupAsDouble,
            hasSnapshotCameraPose = snapshotCamera != null,
            snapshotCameraPosition = snapshotCamera != null ? snapshotCamera.transform.position : Vector3.zero,
            snapshotCameraRotation = snapshotCamera != null ? snapshotCamera.transform.rotation : Quaternion.identity,
            sourceEyeIndex = _pendingSourceEyeIndex,
            dispatchRealtimeSeconds = _pendingDispatchRealtimeSeconds,
            completionRealtimeSeconds = completionRealtimeSeconds,
            hasDispatchCameraPose = _pendingHasCameraPose,
            dispatchCameraPosition = _pendingCameraPosition,
            dispatchCameraRotation = _pendingCameraRotation,
            hasCompletionCameraPose = snapshotCamera != null,
            completionCameraPosition = snapshotCamera != null ? snapshotCamera.transform.position : Vector3.zero,
            completionCameraRotation = snapshotCamera != null ? snapshotCamera.transform.rotation : Quaternion.identity,
            hasProjectionMatrix = _pendingHasProjectionMatrix,
            projectionMatrix = _pendingProjectionMatrix,
            hasWorldToCameraMatrix = _pendingHasWorldToCameraMatrix,
            worldToCameraMatrix = _pendingWorldToCameraMatrix,
            hasDepthReprojectionMatrix = _pendingHasDepthReprojectionMatrix,
            depthReprojectionMatrix = _pendingDepthReprojectionMatrix,
            hasDispatchEyePosition = _pendingHasEyePosition,
            dispatchEyePosition = _pendingEyePosition,
            worldPositionsRaw = rawPositions,
            worldPositions = positions,
            worldNormals = normals,
            worldNormalsNeighbour = neighbourNormals,
            worldNormalsNeighbourValid = neighbourNormalValid,
            observationMeta = meta
        };
    }

    private void CapturePendingReadbackProvenance()
    {
        ResetPendingReadbackProvenance();
        _pendingDispatchRealtimeSeconds = Time.realtimeSinceStartupAsDouble;
        _pendingSourceEyeIndex = preprocessor != null ? (int)preprocessor.CurrentSourceEye : -1;
        _pendingHasEyePosition = preprocessor != null && preprocessor.HasLastDispatchEyePosition;
        _pendingEyePosition = _pendingHasEyePosition
            ? preprocessor.LastDispatchEyePosition
            : Vector3.zero;

        Camera camera = Camera.main;
        if (camera != null)
        {
            _pendingHasCameraPose = true;
            _pendingCameraPosition = camera.transform.position;
            _pendingCameraRotation = camera.transform.rotation;
            _pendingHasProjectionMatrix = true;
            _pendingHasWorldToCameraMatrix = true;
            if (camera.stereoEnabled && _pendingSourceEyeIndex >= 0)
            {
                Camera.StereoscopicEye stereoEye = _pendingSourceEyeIndex == (int)ScanCoverDepthPreprocessor.SourceEye.Left
                    ? Camera.StereoscopicEye.Left
                    : Camera.StereoscopicEye.Right;
                _pendingProjectionMatrix = camera.GetStereoProjectionMatrix(stereoEye);
                _pendingWorldToCameraMatrix = camera.GetStereoViewMatrix(stereoEye);
                Matrix4x4 eyeToWorld = _pendingWorldToCameraMatrix.inverse;
                Vector4 eyePosition = eyeToWorld.GetColumn(3);
                Vector4 eyeForward = eyeToWorld.GetColumn(2);
                Vector4 eyeUp = eyeToWorld.GetColumn(1);
                _pendingCameraPosition = new Vector3(eyePosition.x, eyePosition.y, eyePosition.z);
                Vector3 forward = new Vector3(eyeForward.x, eyeForward.y, eyeForward.z);
                Vector3 up = new Vector3(eyeUp.x, eyeUp.y, eyeUp.z);
                if (forward.sqrMagnitude > 1e-8f && up.sqrMagnitude > 1e-8f)
                    _pendingCameraRotation = Quaternion.LookRotation(forward, up);
            }
            else
            {
                _pendingProjectionMatrix = camera.projectionMatrix;
                _pendingWorldToCameraMatrix = camera.worldToCameraMatrix;
            }
        }

        Matrix4x4[] matrices = Shader.GetGlobalMatrixArray(EnvironmentDepthReprojectionMatricesId);
        if (matrices != null && _pendingSourceEyeIndex >= 0 && _pendingSourceEyeIndex < matrices.Length)
        {
            _pendingHasDepthReprojectionMatrix = true;
            _pendingDepthReprojectionMatrix = matrices[_pendingSourceEyeIndex];
        }
    }

    private void ResetPendingReadbackProvenance()
    {
        _pendingSourceEyeIndex = -1;
        _pendingDispatchRealtimeSeconds = 0d;
        _pendingHasCameraPose = false;
        _pendingCameraPosition = Vector3.zero;
        _pendingCameraRotation = Quaternion.identity;
        _pendingHasProjectionMatrix = false;
        _pendingProjectionMatrix = Matrix4x4.identity;
        _pendingHasWorldToCameraMatrix = false;
        _pendingWorldToCameraMatrix = Matrix4x4.identity;
        _pendingHasDepthReprojectionMatrix = false;
        _pendingDepthReprojectionMatrix = Matrix4x4.identity;
        _pendingHasEyePosition = false;
        _pendingEyePosition = Vector3.zero;
    }

    private bool TryGetNeighborMetrics(int index, int rowDelta, int colDelta, out float distance, out float normalDelta, out float verticalDelta)
    {
        distance = float.NaN;
        normalDelta = float.NaN;
        verticalDelta = float.NaN;
        if (!TryGetGridNeighborIndex(index, rowDelta, colDelta, out int neighborIndex))
            return false;
        if (!TryGetStoredNodePosition(index, out Vector3 point) || !TryGetStoredNodePosition(neighborIndex, out Vector3 neighborPoint))
            return false;

        Vector3 offset = neighborPoint - point;
        distance = offset.magnitude;
        Vector3 normal = index < _currentNormals.Length && Finite(_currentNormals[index]) && _currentNormals[index].sqrMagnitude > 1e-8f
            ? _currentNormals[index].normalized
            : Vector3.zero;
        Vector3 neighborNormal = neighborIndex < _currentNormals.Length && Finite(_currentNormals[neighborIndex]) && _currentNormals[neighborIndex].sqrMagnitude > 1e-8f
            ? _currentNormals[neighborIndex].normalized
            : Vector3.zero;
        normalDelta = normal.sqrMagnitude > 0f && neighborNormal.sqrMagnitude > 0f
            ? 1f - Mathf.Clamp(Vector3.Dot(normal, neighborNormal), -1f, 1f)
            : float.NaN;
        verticalDelta = normal.sqrMagnitude > 0f ? Vector3.Dot(offset, normal) : float.NaN;
        return true;
    }

    private bool TryGetGridNeighborIndex(int index, int rowDelta, int colDelta, out int neighborIndex)
    {
        neighborIndex = -1;
        if (index < 0 || index >= _cells.Count)
            return false;

        Cell cell = _cells[index];
        for (int i = 0; i < _groups.Count; i++)
        {
            GridGroup group = _groups[i];
            int count = group.columns * group.rows;
            if (index < group.startIndex || index >= group.startIndex + count)
                continue;

            int row = cell.row + rowDelta;
            int col = cell.col + colDelta;
            if (row < 0 || row >= group.rows || col < 0 || col >= group.columns)
                return false;

            neighborIndex = group.startIndex + row * group.columns + col;
            return neighborIndex >= 0 && neighborIndex < _cells.Count;
        }

        return false;
    }

    private bool TryGetStoredNodePosition(int index, out Vector3 position)
    {
        position = Vector3.zero;
        if (index < 0 || _currentPositions == null || index >= _currentPositions.Length)
            return false;

        bool hasLinePosition = _currentLineValid != null && index < _currentLineValid.Length && _currentLineValid[index];
        bool hasMeshPosition = _currentValid != null && index < _currentValid.Length && _currentValid[index];
        if (!hasLinePosition && !hasMeshPosition)
            return false;

        position = _currentPositions[index];
        return Finite(position);
    }

    private static Vector3[] BuildSurfaceMeshExportNormals(Vector3[] vertices, int[] triangles, Vector3[] meshNormals)
    {
        if (meshNormals != null && meshNormals.Length == vertices.Length)
        {
            Vector3[] copiedNormals = new Vector3[meshNormals.Length];
            for (int i = 0; i < meshNormals.Length; i++)
                copiedNormals[i] = SafeNormalized(meshNormals[i], Vector3.up);
            return copiedNormals;
        }

        Vector3[] normals = new Vector3[vertices.Length];
        if (triangles != null)
        {
            for (int i = 0; i + 2 < triangles.Length; i += 3)
            {
                int ia = triangles[i];
                int ib = triangles[i + 1];
                int ic = triangles[i + 2];
                if ((uint)ia >= (uint)vertices.Length || (uint)ib >= (uint)vertices.Length || (uint)ic >= (uint)vertices.Length)
                    continue;

                Vector3 a = vertices[ia];
                Vector3 b = vertices[ib];
                Vector3 c = vertices[ic];
                Vector3 normal = Vector3.Cross(b - a, c - a);
                if (!Finite(normal) || normal.sqrMagnitude <= 1e-8f)
                    continue;

                normal.Normalize();
                normals[ia] += normal;
                normals[ib] += normal;
                normals[ic] += normal;
            }
        }

        for (int i = 0; i < normals.Length; i++)
            normals[i] = SafeNormalized(normals[i], Vector3.up);
        return normals;
    }

    private static string FormatVector(Vector3 value)
        => string.Concat(FormatFloat(value.x), ",", FormatFloat(value.y), ",", FormatFloat(value.z));

    private static string FormatFloat(float value)
        => value.ToString("F6", CultureInfo.InvariantCulture);

    private static string FormatObjFloat(float value)
        => value.ToString("G9", CultureInfo.InvariantCulture);

    private static string FormatNullableFloat(float value)
        => float.IsNaN(value) || float.IsInfinity(value) ? string.Empty : value.ToString("F6", CultureInfo.InvariantCulture);

    private static string SanitizeObjName(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "ScanCover_Object";
        StringBuilder builder = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            builder.Append(char.IsLetterOrDigit(c) || c == '_' || c == '-' ? c : '_');
        }
        return builder.ToString();
    }

    private static string EscapeJson(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "";
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private static void AppendJsonString(StringBuilder builder, string key, string value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": \"").Append(EscapeJson(value)).Append('"');
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonNumber(StringBuilder builder, string key, float value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": ").Append(FormatFloat(value));
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonNumber(StringBuilder builder, string key, int value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": ").Append(value.ToString(CultureInfo.InvariantCulture));
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendTransformJson(StringBuilder builder, Transform transform)
    {
        if (transform == null)
        {
            builder.Append("null");
            return;
        }

        builder.Append("{");
        builder.Append("\"position\":");
        AppendVectorJson(builder, transform.position);
        builder.Append(",\"rotationEuler\":");
        AppendVectorJson(builder, transform.rotation.eulerAngles);
        builder.Append(",\"forward\":");
        AppendVectorJson(builder, transform.forward);
        builder.Append(",\"right\":");
        AppendVectorJson(builder, transform.right);
        builder.Append(",\"up\":");
        AppendVectorJson(builder, transform.up);
        builder.Append(",\"localToWorld\":");
        AppendMatrixJson(builder, transform.localToWorldMatrix);
        builder.Append(",\"worldToLocal\":");
        AppendMatrixJson(builder, transform.worldToLocalMatrix);
        builder.Append("}");
    }

    private static void AppendCameraJson(StringBuilder builder, Camera camera)
    {
        if (camera == null)
        {
            builder.Append("null");
            return;
        }

        builder.Append("{");
        builder.Append("\"nearClipPlane\":").Append(FormatFloat(camera.nearClipPlane));
        builder.Append(",\"farClipPlane\":").Append(FormatFloat(camera.farClipPlane));
        builder.Append(",\"fieldOfView\":").Append(FormatFloat(camera.fieldOfView));
        builder.Append(",\"aspect\":").Append(FormatFloat(camera.aspect));
        builder.Append(",\"worldToCamera\":");
        AppendMatrixJson(builder, camera.worldToCameraMatrix);
        builder.Append(",\"projection\":");
        AppendMatrixJson(builder, camera.projectionMatrix);
        builder.Append("}");
    }

    private static void AppendVectorJson(StringBuilder builder, Vector3 value)
    {
        builder.Append('[')
            .Append(FormatFloat(value.x)).Append(',')
            .Append(FormatFloat(value.y)).Append(',')
            .Append(FormatFloat(value.z)).Append(']');
    }

    private static void AppendMatrixJson(StringBuilder builder, Matrix4x4 matrix)
    {
        builder.Append('[');
        for (int row = 0; row < 4; row++)
        {
            if (row > 0)
                builder.Append(',');
            builder.Append('[');
            for (int col = 0; col < 4; col++)
            {
                if (col > 0)
                    builder.Append(',');
                builder.Append(FormatFloat(matrix[row, col]));
            }
            builder.Append(']');
        }
        builder.Append(']');
    }

    private static void AppendIndent(StringBuilder builder, int indent)
    {
        for (int i = 0; i < indent; i++)
            builder.Append("  ");
    }

    private GameObject CreateFallbackMarker()
    {
        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere); go.name = "ScanCoverDepthGridMarker"; go.transform.SetParent(ResolveMarkerParent(), false); go.transform.localScale = Vector3.one * Mathf.Max(0.001f, fallbackMarkerScaleMeters);
        Collider collider = go.GetComponent<Collider>(); if (collider != null) Destroy(collider);
        Renderer renderer = go.GetComponent<Renderer>();
        if (renderer != null)
        {
            if (_fallbackMaterial == null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit"); if (shader == null) shader = Shader.Find("Unlit/Color");
                _fallbackMaterial = new Material(shader) { name = "ScanCoverDepthGridFallback_Mat" }; _fallbackMaterial.SetColor("_BaseColor", Color.cyan); _fallbackMaterial.SetColor("_Color", Color.cyan);
            }
            renderer.sharedMaterial = _fallbackMaterial; renderer.shadowCastingMode = ShadowCastingMode.Off; renderer.receiveShadows = false;
        }
        return go;
    }

    private void EnsureCenterDebugObjects()
    {
        EnsureDisplayRoots();
        Transform parent = ResolveSurfaceParent();
        if (parent == null)
            return;

        if (_centerDebugRoot == null)
        {
            _centerDebugRoot = new GameObject("[Debug] CenterMarkers");
            _centerDebugRoot.transform.SetParent(parent, false);
            _centerDebugRoot.transform.localPosition = Vector3.zero;
            _centerDebugRoot.transform.localRotation = Quaternion.identity;
            _centerDebugRoot.transform.localScale = Vector3.one;
        }
        else if (_centerDebugRoot.transform.parent != parent)
        {
            _centerDebugRoot.transform.SetParent(parent, false);
        }

        EnsureCenterDebugMarker(0, "ScreenCenter", screenCenterDebugColor);
        EnsureCenterDebugMarker(1, "GridCenter", gridCenterDebugColor);
        EnsureCenterDebugMarker(2, "PatchCenter", patchCenterDebugColor);
    }

    private void EnsureCenterDebugMarker(int index, string name, Color color)
    {
        if (_centerDebugRoot == null || index < 0 || index >= _centerDebugMarkers.Length)
            return;

        if (_centerDebugMarkers[index] == null)
        {
            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.name = name;
            marker.transform.SetParent(_centerDebugRoot.transform, false);
            Collider markerCollider = marker.GetComponent<Collider>();
            if (markerCollider != null)
                Destroy(markerCollider);

            Renderer renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Unlit/Color");
                Material material = new Material(shader) { name = $"ScanCover_{name}_Mat" };
                if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
                if (material.HasProperty("_Color")) material.SetColor("_Color", color);
                if (material.HasProperty("_Surface")) material.SetFloat("_Surface", 1f);
                if (material.HasProperty("_ZWrite")) material.SetFloat("_ZWrite", 0f);
                material.renderQueue = 3100;
                renderer.sharedMaterial = material;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                _centerDebugMaterials[index] = material;
            }

            _centerDebugMarkers[index] = marker;
        }

        _centerDebugMarkers[index].transform.localScale = Vector3.one * Mathf.Max(0.002f, centerDebugMarkerScaleMeters);
    }

    private void UpdateCenterDebugMarker(int index, Vector3 worldPosition, Color color)
    {
        if (_centerDebugRoot == null || index < 0 || index >= _centerDebugMarkers.Length || _centerDebugMarkers[index] == null)
            return;

        Transform markerTransform = _centerDebugMarkers[index].transform;
        Transform localTransform = ResolveDisplayLocalTransform();
        if (useWorldSpaceDisplayRoots && _centerDebugRoot.transform.parent != null)
            markerTransform.localPosition = localTransform != null ? localTransform.InverseTransformPoint(worldPosition) : worldPosition;
        else
            markerTransform.position = worldPosition;

        markerTransform.localRotation = Quaternion.identity;
        markerTransform.localScale = Vector3.one * Mathf.Max(0.002f, centerDebugMarkerScaleMeters);

        Material material = index >= 0 && index < _centerDebugMaterials.Length ? _centerDebugMaterials[index] : null;
        if (material != null)
        {
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
            if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        }

        if (!_centerDebugMarkers[index].activeSelf)
            _centerDebugMarkers[index].SetActive(true);
    }

    private void SetCenterDebugMarkersVisible(bool visible)
    {
        if (_centerDebugRoot == null)
            return;

        if (_centerDebugRoot.activeSelf != visible)
            _centerDebugRoot.SetActive(visible);

        if (!visible)
            return;

        for (int i = 0; i < _centerDebugMarkers.Length; i++)
        {
            if (_centerDebugMarkers[i] != null && !_centerDebugMarkers[i].activeSelf)
                _centerDebugMarkers[i].SetActive(true);
        }
    }

    private void SetCenterDebugMarkerVisible(int index, bool visible)
    {
        if (index < 0 || index >= _centerDebugMarkers.Length || _centerDebugMarkers[index] == null)
            return;

        if (_centerDebugMarkers[index].activeSelf != visible)
            _centerDebugMarkers[index].SetActive(visible);
    }

    private void UpdateHeadsetScreenCenterMarker()
    {
        if (!showHeadsetScreenCenterMarker || !_previewVisible || !previewDisplayVisible)
        {
            SetHeadsetScreenCenterMarkerVisible(false);
            return;
        }

        if (!_hasFrozenHeadsetScreenCenterPoint)
        {
            SetHeadsetScreenCenterMarkerVisible(false);
            return;
        }

        Vector3 worldPosition = _frozenHeadsetScreenCenterPoint;
        if (_frozenHeadsetScreenCenterNormal.sqrMagnitude > 1e-8f)
            worldPosition += _frozenHeadsetScreenCenterNormal.normalized * Mathf.Max(0f, gridLineSurfaceOffsetMeters);
        Transform parent = ResolveSurfaceParent();
        EnsureHeadsetScreenCenterMarker(parent);
        if (_headsetScreenCenterMarker == null)
            return;

        if (_headsetScreenCenterMarker.transform.parent != parent)
            _headsetScreenCenterMarker.transform.SetParent(parent, false);

        Transform localTransform = ResolveDisplayLocalTransform();
        if (useWorldSpaceDisplayRoots && _headsetScreenCenterMarker.transform.parent != null)
            _headsetScreenCenterMarker.transform.localPosition = localTransform != null ? localTransform.InverseTransformPoint(worldPosition) : worldPosition;
        else
            _headsetScreenCenterMarker.transform.position = worldPosition;
        _headsetScreenCenterMarker.transform.localRotation = Quaternion.identity;
        _headsetScreenCenterMarker.transform.localScale = Vector3.one * Mathf.Max(0.002f, headsetScreenCenterMarkerScaleMeters);
        SetHeadsetScreenCenterMarkerVisible(true);
    }

    private void EnsureHeadsetScreenCenterMarker(Transform parent)
    {
        if (parent == null)
            return;

        if (_headsetScreenCenterMarker == null)
        {
            _headsetScreenCenterMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _headsetScreenCenterMarker.name = "[Debug] HeadsetScreenCenter";
            _headsetScreenCenterMarker.transform.SetParent(parent, false);
            Collider markerCollider = _headsetScreenCenterMarker.GetComponent<Collider>();
            if (markerCollider != null)
                Destroy(markerCollider);

            Renderer renderer = _headsetScreenCenterMarker.GetComponent<Renderer>();
            if (renderer != null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Unlit/Color");
                _headsetScreenCenterMarkerMaterial = new Material(shader) { name = "ScanCover_HeadsetScreenCenter_Mat" };
                if (_headsetScreenCenterMarkerMaterial.HasProperty("_BaseColor")) _headsetScreenCenterMarkerMaterial.SetColor("_BaseColor", headsetScreenCenterMarkerColor);
                if (_headsetScreenCenterMarkerMaterial.HasProperty("_Color")) _headsetScreenCenterMarkerMaterial.SetColor("_Color", headsetScreenCenterMarkerColor);
                if (_headsetScreenCenterMarkerMaterial.HasProperty("_Surface")) _headsetScreenCenterMarkerMaterial.SetFloat("_Surface", 1f);
                if (_headsetScreenCenterMarkerMaterial.HasProperty("_ZWrite")) _headsetScreenCenterMarkerMaterial.SetFloat("_ZWrite", 0f);
                _headsetScreenCenterMarkerMaterial.renderQueue = 3100;
                renderer.sharedMaterial = _headsetScreenCenterMarkerMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
        }
        else if (_headsetScreenCenterMarker.transform.parent != parent)
        {
            _headsetScreenCenterMarker.transform.SetParent(parent, false);
        }

        _headsetScreenCenterMarker.transform.localScale = Vector3.one * Mathf.Max(0.002f, headsetScreenCenterMarkerScaleMeters);
    }

    private void SetHeadsetScreenCenterMarkerVisible(bool visible)
    {
        if (_headsetScreenCenterMarker != null && _headsetScreenCenterMarker.activeSelf != visible)
            _headsetScreenCenterMarker.SetActive(visible);
    }

    private void UpdateRawDepthScreenCenterMarker(int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch)
    {
        if (!showRawDepthScreenCenterMarker || !_previewVisible || !previewDisplayVisible)
        {
            SetRawDepthScreenCenterMarkerVisible(false);
            return;
        }

        if (!TryGetRawDepthScreenCenter(width, height, worldPositions, worldNormals, observationMeta, validScratch, out Vector3 center, out Vector3 normal))
        {
            SetRawDepthScreenCenterMarkerVisible(false);
            return;
        }

        if (normal.sqrMagnitude > 1e-8f)
            center += normal.normalized * Mathf.Max(0f, gridLineSurfaceOffsetMeters);

        EnsureRawDepthScreenCenterMarker();
        if (_rawDepthScreenCenterMarker == null)
            return;

        Transform localTransform = ResolveDisplayLocalTransform();
        if (useWorldSpaceDisplayRoots && _rawDepthScreenCenterMarker.transform.parent != null)
            _rawDepthScreenCenterMarker.transform.localPosition = localTransform != null ? localTransform.InverseTransformPoint(center) : center;
        else
            _rawDepthScreenCenterMarker.transform.position = center;

        _rawDepthScreenCenterMarker.transform.localRotation = Quaternion.identity;
        _rawDepthScreenCenterMarker.transform.localScale = Vector3.one * Mathf.Max(0.002f, rawDepthScreenCenterMarkerScaleMeters);
        SetRawDepthScreenCenterMarkerVisible(true);
    }

    private bool TryGetRawDepthScreenCenter(int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, out Vector3 center, out Vector3 normal)
    {
        center = Vector3.zero;
        normal = Vector3.zero;
        if (width <= 0 || height <= 0 || worldPositions.Length <= 0 || observationMeta.Length <= 0 || validScratch == null)
            return false;

        float centerX = (width - 1) * 0.5f;
        float centerY = (height - 1) * 0.5f;
        int x0 = Mathf.FloorToInt(centerX);
        int x1 = Mathf.CeilToInt(centerX);
        int y0 = Mathf.FloorToInt(centerY);
        int y1 = Mathf.CeilToInt(centerY);

        int count = 0;
        AddRawDepthScreenCenterSample(x0, y0, width, height, worldPositions, worldNormals, observationMeta, validScratch, ref center, ref normal, ref count);
        if (x1 != x0) AddRawDepthScreenCenterSample(x1, y0, width, height, worldPositions, worldNormals, observationMeta, validScratch, ref center, ref normal, ref count);
        if (y1 != y0) AddRawDepthScreenCenterSample(x0, y1, width, height, worldPositions, worldNormals, observationMeta, validScratch, ref center, ref normal, ref count);
        if (x1 != x0 && y1 != y0) AddRawDepthScreenCenterSample(x1, y1, width, height, worldPositions, worldNormals, observationMeta, validScratch, ref center, ref normal, ref count);

        if (count <= 0)
            return false;

        center /= count;
        if (normal.sqrMagnitude > 1e-8f)
            normal.Normalize();

        return true;
    }

    private void AddRawDepthScreenCenterSample(int x, int y, int width, int height, NativeArray<Color> worldPositions, NativeArray<Color> worldNormals, NativeArray<Color> observationMeta, List<bool> validScratch, ref Vector3 center, ref Vector3 normal, ref int count)
    {
        if ((uint)x >= (uint)width || (uint)y >= (uint)height)
            return;

        int index = x + y * width;
        if (index < 0 || index >= validScratch.Count || index >= worldPositions.Length || index >= observationMeta.Length || !validScratch[index])
            return;

        center += WorldPos(worldPositions[index]);
        if (index < worldNormals.Length && observationMeta[index].a >= 0.5f)
        {
            Vector3 sampleNormal = WorldNormal(worldNormals[index], true);
            if (sampleNormal.sqrMagnitude > 1e-8f)
                normal += sampleNormal.normalized;
        }
        count++;
    }

    private void EnsureRawDepthScreenCenterMarker()
    {
        EnsureDisplayRoots();
        Transform parent = ResolveSurfaceParent();
        if (parent == null)
            return;

        if (_rawDepthScreenCenterMarker == null)
        {
            _rawDepthScreenCenterMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _rawDepthScreenCenterMarker.name = "[Debug] RawDepthScreenCenter";
            _rawDepthScreenCenterMarker.transform.SetParent(parent, false);
            Collider markerCollider = _rawDepthScreenCenterMarker.GetComponent<Collider>();
            if (markerCollider != null)
                Destroy(markerCollider);

            Renderer renderer = _rawDepthScreenCenterMarker.GetComponent<Renderer>();
            if (renderer != null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Unlit/Color");
                _rawDepthScreenCenterMarkerMaterial = new Material(shader) { name = "ScanCover_RawDepthScreenCenter_Mat" };
                if (_rawDepthScreenCenterMarkerMaterial.HasProperty("_BaseColor")) _rawDepthScreenCenterMarkerMaterial.SetColor("_BaseColor", rawDepthScreenCenterMarkerColor);
                if (_rawDepthScreenCenterMarkerMaterial.HasProperty("_Color")) _rawDepthScreenCenterMarkerMaterial.SetColor("_Color", rawDepthScreenCenterMarkerColor);
                if (_rawDepthScreenCenterMarkerMaterial.HasProperty("_Surface")) _rawDepthScreenCenterMarkerMaterial.SetFloat("_Surface", 1f);
                if (_rawDepthScreenCenterMarkerMaterial.HasProperty("_ZWrite")) _rawDepthScreenCenterMarkerMaterial.SetFloat("_ZWrite", 0f);
                _rawDepthScreenCenterMarkerMaterial.renderQueue = 3100;
                renderer.sharedMaterial = _rawDepthScreenCenterMarkerMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
        }
        else if (_rawDepthScreenCenterMarker.transform.parent != parent)
        {
            _rawDepthScreenCenterMarker.transform.SetParent(parent, false);
        }

        _rawDepthScreenCenterMarker.transform.localScale = Vector3.one * Mathf.Max(0.002f, rawDepthScreenCenterMarkerScaleMeters);
    }

    private void SetRawDepthScreenCenterMarkerVisible(bool visible)
    {
        if (_rawDepthScreenCenterMarker != null && _rawDepthScreenCenterMarker.activeSelf != visible)
            _rawDepthScreenCenterMarker.SetActive(visible);
    }

    private void UpdateOriginalGridCenterMarker(bool[] valid, Vector3[] positions, Vector3[] normals)
    {
        if (!showOriginalGridCenterMarker)
        {
            SetOriginalGridCenterMarkerVisible(false);
            return;
        }

        if (!TryGetOriginalGridCenter(valid, positions, normals, out Vector3 center, out Vector3 normal))
        {
            SetOriginalGridCenterMarkerVisible(false);
            return;
        }

        if (normal.sqrMagnitude > 1e-8f)
            center += normal.normalized * Mathf.Max(0f, gridLineSurfaceOffsetMeters);

        EnsureOriginalGridCenterMarker();
        if (_originalGridCenterMarker == null)
            return;

        Transform localTransform = ResolveDisplayLocalTransform();
        if (useWorldSpaceDisplayRoots && _originalGridCenterMarker.transform.parent != null)
            _originalGridCenterMarker.transform.localPosition = localTransform != null ? localTransform.InverseTransformPoint(center) : center;
        else
            _originalGridCenterMarker.transform.position = center;

        _originalGridCenterMarker.transform.localRotation = Quaternion.identity;
        _originalGridCenterMarker.transform.localScale = Vector3.one * Mathf.Max(0.002f, originalGridCenterMarkerScaleMeters);
        SetOriginalGridCenterMarkerVisible(true);
    }

    private bool TryGetOriginalGridCenter(bool[] valid, Vector3[] positions, Vector3[] normals, out Vector3 center, out Vector3 normal)
    {
        center = Vector3.zero;
        normal = Vector3.zero;
        if (valid == null || positions == null || _groups.Count <= 0)
        {
            return false;
        }

        GridGroup group = _groups[0];
        if (group.columns <= 0 || group.rows <= 0)
        {
            return false;
        }

        float centerRow = (group.rows - 1) * 0.5f;
        float centerCol = (group.columns - 1) * 0.5f;
        int row0 = Mathf.FloorToInt(centerRow);
        int row1 = Mathf.CeilToInt(centerRow);
        int col0 = Mathf.FloorToInt(centerCol);
        int col1 = Mathf.CeilToInt(centerCol);

        int count = 0;
        AddOriginalGridCenterSample(row0, col0, group, valid, positions, normals, ref center, ref normal, ref count);
        if (col1 != col0) AddOriginalGridCenterSample(row0, col1, group, valid, positions, normals, ref center, ref normal, ref count);
        if (row1 != row0) AddOriginalGridCenterSample(row1, col0, group, valid, positions, normals, ref center, ref normal, ref count);
        if (row1 != row0 && col1 != col0) AddOriginalGridCenterSample(row1, col1, group, valid, positions, normals, ref center, ref normal, ref count);

        if (count <= 0)
            return false;

        center /= count;
        if (normal.sqrMagnitude > 1e-8f)
            normal.Normalize();

        return true;
    }

    private void AddOriginalGridCenterSample(int row, int col, GridGroup group, bool[] valid, Vector3[] positions, Vector3[] normals, ref Vector3 center, ref Vector3 normal, ref int count)
    {
        if (row < 0 || row >= group.rows || col < 0 || col >= group.columns)
            return;

        int index = group.startIndex + row * group.columns + col;
        if (index < 0 || index >= valid.Length || index >= positions.Length || !valid[index])
            return;

        center += positions[index];
        if (normals != null && index < normals.Length && normals[index].sqrMagnitude > 1e-8f)
            normal += normals[index].normalized;
        count++;
    }

    private void EnsureOriginalGridCenterMarker()
    {
        EnsureDisplayRoots();
        Transform parent = ResolveSurfaceParent();
        if (parent == null)
            return;

        if (_originalGridCenterMarker == null)
        {
            _originalGridCenterMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _originalGridCenterMarker.name = "[Debug] Original32x32GridCenter";
            _originalGridCenterMarker.transform.SetParent(parent, false);
            Collider markerCollider = _originalGridCenterMarker.GetComponent<Collider>();
            if (markerCollider != null)
                Destroy(markerCollider);

            Renderer renderer = _originalGridCenterMarker.GetComponent<Renderer>();
            if (renderer != null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Unlit/Color");
                _originalGridCenterMarkerMaterial = new Material(shader) { name = "ScanCover_Original32x32GridCenter_Mat" };
                if (_originalGridCenterMarkerMaterial.HasProperty("_BaseColor")) _originalGridCenterMarkerMaterial.SetColor("_BaseColor", originalGridCenterMarkerColor);
                if (_originalGridCenterMarkerMaterial.HasProperty("_Color")) _originalGridCenterMarkerMaterial.SetColor("_Color", originalGridCenterMarkerColor);
                if (_originalGridCenterMarkerMaterial.HasProperty("_Surface")) _originalGridCenterMarkerMaterial.SetFloat("_Surface", 1f);
                if (_originalGridCenterMarkerMaterial.HasProperty("_ZWrite")) _originalGridCenterMarkerMaterial.SetFloat("_ZWrite", 0f);
                _originalGridCenterMarkerMaterial.renderQueue = 3100;
                renderer.sharedMaterial = _originalGridCenterMarkerMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
        }
        else if (_originalGridCenterMarker.transform.parent != parent)
        {
            _originalGridCenterMarker.transform.SetParent(parent, false);
        }

        _originalGridCenterMarker.transform.localScale = Vector3.one * Mathf.Max(0.002f, originalGridCenterMarkerScaleMeters);
    }

    private void SetOriginalGridCenterMarkerVisible(bool visible)
    {
        if (_originalGridCenterMarker != null && _originalGridCenterMarker.activeSelf != visible)
            _originalGridCenterMarker.SetActive(visible);
    }

    private void EnsureGridLineObjects()
    {
        EnsureDisplayRoots();
        if (_lineRoot == null)
        {
            _lineRoot = new GameObject("[ScanCover] DepthGrid Lines");
            _lineRoot.transform.SetParent(ResolveSurfaceParent(), false);
            _lineFilter = _lineRoot.AddComponent<MeshFilter>();
            _lineRenderer = _lineRoot.AddComponent<MeshRenderer>();
        }
        else if (_lineRoot.transform.parent != ResolveSurfaceParent())
        {
            _lineRoot.transform.SetParent(ResolveSurfaceParent(), false);
        }

        UpdateDisplayRootsTransform();

        if (_lineMesh == null)
        {
            _lineMesh = new Mesh { name = "ScanCover_DepthGridLines" };
            _lineMesh.MarkDynamic();
        }

        Material targetMaterial = ResolveGridLineMaterial();
        if (targetMaterial != null && _lineRenderer.sharedMaterial != targetMaterial)
            _lineRenderer.sharedMaterial = targetMaterial;

        _lineFilter.sharedMesh = _lineMesh;
        _lineRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _lineRenderer.receiveShadows = false;
        if (_propertyBlock != null)
        {
            Color lineColor = gridLineColor;
            _propertyBlock.Clear();
            _propertyBlock.SetColor("_BaseColor", lineColor);
            _propertyBlock.SetColor("_Color", lineColor);
            _lineRenderer.SetPropertyBlock(_propertyBlock);
        }
    }

    private Material ResolveGridLineMaterial()
    {
        if (gridLineMaterialOverride != null)
        {
            if (_lineMaterial != null && _lineMaterial != gridLineMaterialOverride)
                Destroy(_lineMaterial);
            _lineMaterial = null;
            return gridLineMaterialOverride;
        }

        if (_lineMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader != null)
            {
                _lineMaterial = new Material(shader) { name = "ScanCoverDepthGridLine_Mat" };
            }
        }

        ConfigureGridLineMaterial(_lineMaterial);
        return _lineMaterial;
    }

    private void ConfigureGridLineMaterial(Material targetMaterial)
    {
        if (targetMaterial == null)
            return;

        targetMaterial.SetColor("_BaseColor", gridLineColor);
        targetMaterial.SetColor("_Color", gridLineColor);
        if (targetMaterial.HasProperty("_Surface")) targetMaterial.SetFloat("_Surface", 1f);
        if (targetMaterial.HasProperty("_Blend")) targetMaterial.SetFloat("_Blend", 0f);
        if (targetMaterial.HasProperty("_ZWrite")) targetMaterial.SetFloat("_ZWrite", 0f);
        if (targetMaterial.HasProperty("_SrcBlend")) targetMaterial.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (targetMaterial.HasProperty("_DstBlend")) targetMaterial.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        if (targetMaterial.HasProperty("_Cull")) targetMaterial.SetFloat("_Cull", (float)CullMode.Off);
        if (targetMaterial.HasProperty("_CullMode")) targetMaterial.SetFloat("_CullMode", (float)CullMode.Off);
        if (targetMaterial.HasProperty("_ZTest"))
            targetMaterial.SetFloat("_ZTest", gridLinesRenderBehindCandidatePatch ? (float)CompareFunction.LessEqual : (float)CompareFunction.Always);
        targetMaterial.renderQueue = gridLinesRenderBehindCandidatePatch ? (int)RenderQueue.Transparent - 20 : (int)RenderQueue.Overlay;
    }

    private void EnsureRemeshGridLineObjects()
    {
        EnsureDisplayRoots();
        if (_remeshLineRoot == null)
        {
            _remeshLineRoot = new GameObject("[ScanCover] CandidateRemesh Lines");
            _remeshLineRoot.transform.SetParent(ResolveSurfaceParent(), false);
            _remeshLineFilter = _remeshLineRoot.AddComponent<MeshFilter>();
            _remeshLineRenderer = _remeshLineRoot.AddComponent<MeshRenderer>();
        }
        else if (_remeshLineRoot.transform.parent != ResolveSurfaceParent())
        {
            _remeshLineRoot.transform.SetParent(ResolveSurfaceParent(), false);
        }

        UpdateDisplayRootsTransform();
        if (_remeshLineMesh == null)
        {
            _remeshLineMesh = new Mesh { name = "ScanCover_CandidateRemeshLines" };
            _remeshLineMesh.MarkDynamic();
        }

        if (_remeshLineMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader != null)
            {
                _remeshLineMaterial = new Material(shader) { name = "ScanCover_CandidateRemeshLine_Mat" };
                _remeshLineMaterial.SetColor("_BaseColor", largestCandidateGridLineColor);
                _remeshLineMaterial.SetColor("_Color", largestCandidateGridLineColor);
                if (_remeshLineMaterial.HasProperty("_Surface")) _remeshLineMaterial.SetFloat("_Surface", 1f);
                if (_remeshLineMaterial.HasProperty("_Blend")) _remeshLineMaterial.SetFloat("_Blend", 0f);
                if (_remeshLineMaterial.HasProperty("_ZWrite")) _remeshLineMaterial.SetFloat("_ZWrite", 0f);
                if (_remeshLineMaterial.HasProperty("_SrcBlend")) _remeshLineMaterial.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                if (_remeshLineMaterial.HasProperty("_DstBlend")) _remeshLineMaterial.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
                if (_remeshLineMaterial.HasProperty("_Cull")) _remeshLineMaterial.SetFloat("_Cull", (float)CullMode.Off);
                if (_remeshLineMaterial.HasProperty("_CullMode")) _remeshLineMaterial.SetFloat("_CullMode", (float)CullMode.Off);
                if (_remeshLineMaterial.HasProperty("_ZTest")) _remeshLineMaterial.SetFloat("_ZTest", 8f);
                _remeshLineMaterial.renderQueue = (int)RenderQueue.Overlay;
            }
        }

        if (_remeshLineRenderer != null && _remeshLineMaterial != null && _remeshLineRenderer.sharedMaterial != _remeshLineMaterial)
            _remeshLineRenderer.sharedMaterial = _remeshLineMaterial;
        if (_remeshLineFilter != null)
            _remeshLineFilter.sharedMesh = _remeshLineMesh;
        if (_remeshLineRenderer != null)
        {
            _remeshLineRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _remeshLineRenderer.receiveShadows = false;
        }
    }

    private void EnsureGeometricSurfaceGridObjects()
    {
        EnsureDisplayRoots();
        Transform parent = ResolveSurfaceParent();
        if (_geometricSurfaceGridRoot == null)
        {
            _geometricSurfaceGridRoot = new GameObject("[ScanCover] Geometric Surface Grid");
            _geometricSurfaceGridRoot.transform.SetParent(parent, false);
            _geometricSurfaceGridFilter = _geometricSurfaceGridRoot.AddComponent<MeshFilter>();
            _geometricSurfaceGridRenderer = _geometricSurfaceGridRoot.AddComponent<MeshRenderer>();
        }
        else if (_geometricSurfaceGridRoot.transform.parent != parent)
        {
            _geometricSurfaceGridRoot.transform.SetParent(parent, false);
        }

        UpdateDisplayRootsTransform();
        if (_geometricSurfaceGridMesh == null)
        {
            _geometricSurfaceGridMesh = new Mesh { name = "ScanCover_GeometricSurfaceGrid" };
            _geometricSurfaceGridMesh.MarkDynamic();
        }

        if (_geometricSurfaceGridMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader != null)
                _geometricSurfaceGridMaterial = new Material(shader) { name = "ScanCover_GeometricSurfaceGrid_Mat" };
        }

        if (_geometricSurfaceGridMaterial != null)
        {
            if (_geometricSurfaceGridMaterial.HasProperty("_Surface")) _geometricSurfaceGridMaterial.SetFloat("_Surface", 1f);
            if (_geometricSurfaceGridMaterial.HasProperty("_Blend")) _geometricSurfaceGridMaterial.SetFloat("_Blend", 0f);
            if (_geometricSurfaceGridMaterial.HasProperty("_ZWrite")) _geometricSurfaceGridMaterial.SetFloat("_ZWrite", 0f);
            if (_geometricSurfaceGridMaterial.HasProperty("_SrcBlend")) _geometricSurfaceGridMaterial.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
            if (_geometricSurfaceGridMaterial.HasProperty("_DstBlend")) _geometricSurfaceGridMaterial.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
            if (_geometricSurfaceGridMaterial.HasProperty("_Cull")) _geometricSurfaceGridMaterial.SetFloat("_Cull", (float)CullMode.Off);
            if (_geometricSurfaceGridMaterial.HasProperty("_CullMode")) _geometricSurfaceGridMaterial.SetFloat("_CullMode", (float)CullMode.Off);
            if (_geometricSurfaceGridMaterial.HasProperty("_ZTest")) _geometricSurfaceGridMaterial.SetFloat("_ZTest", 8f);
            _geometricSurfaceGridMaterial.renderQueue = (int)RenderQueue.Overlay;
        }

        if (_geometricSurfaceGridFilter != null)
            _geometricSurfaceGridFilter.sharedMesh = _geometricSurfaceGridMesh;
        if (_geometricSurfaceGridRenderer != null)
        {
            _geometricSurfaceGridRenderer.sharedMaterial = _geometricSurfaceGridMaterial;
            _geometricSurfaceGridRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _geometricSurfaceGridRenderer.receiveShadows = false;
        }
    }

    private enum ProbeRowLinkKind
    {
        Stable,
        Refine,
        Break
    }

    private struct ProbeRowHit
    {
        public bool valid;
        public Vector3 point;
        public Vector3 normal;
        public int neighborhoodCount;
        public float planeDeviation;
    }

    private struct ProbeSliceSegment
    {
        public Vector3 a;
        public Vector3 b;
    }

    private struct RansacPatchSample
    {
        public Vector3 point;
        public Vector3 normal;
        public bool active;
    }

    private struct RansacPatchPlane
    {
        public Vector3 center;
        public Vector3 normal;
        public int inlierCount;
    }

    private struct ProbeRowLink
    {
        public int a;
        public int b;
    }

    private void UpdateProbeRowExperiment(bool[] valid, Vector3[] positions, Vector3[] normals)
    {
        if (!showProbeRowExperiment || !_previewVisible || !previewDisplayVisible)
        {
            SetProbeRowExperimentVisible(false);
            return;
        }

        if (valid == null || positions == null || normals == null || valid.Length == 0 || positions.Length != valid.Length)
        {
            SetProbeRowExperimentVisible(false);
            return;
        }

        Transform origin = ResolveViewLockedOrigin();
        if (origin == null)
        {
            SetProbeRowExperimentVisible(false);
            return;
        }

        BuildProbeTriangleIndexCache(valid);
        if (_probeTriangleIndices.Count <= 0)
        {
            SetProbeRowExperimentVisible(false);
            return;
        }

        Vector3 originPos = origin.position;
        Vector3 forward = origin.forward.sqrMagnitude > 1e-8f ? origin.forward.normalized : Vector3.forward;
        float spacing = Mathf.Max(0.005f, probeRowSpacingMeters);
        float targetDistance = Mathf.Max(0.05f, probeRowTargetDistanceMeters);
        Ray centerRay = new Ray(originPos, forward);
        if (!TryRaycastProbeMesh(centerRay, positions, normals, Mathf.Max(targetDistance + 0.1f, probeRowMaxRayDistanceMeters), out Vector3 anchorPoint, out Vector3 anchorNormal))
        {
            SetProbeRowExperimentVisible(false);
            return;
        }

        anchorNormal = anchorNormal.sqrMagnitude > 1e-8f ? anchorNormal.normalized : -forward;
        int maxPointCount = Mathf.Max(2, probeRowMaxPointCount, 768);
        float maxDistance = Mathf.Max(targetDistance + 0.1f, probeRowMaxRayDistanceMeters);
        float surfaceOffset = Mathf.Max(0f, probeRowSurfaceOffsetMeters);
        List<ProbeRowHit> hits = new List<ProbeRowHit>(maxPointCount);
        List<ProbeRowLink> links = new List<ProbeRowLink>(maxPointCount);
        Vector3 horizontalForward = Vector3.ProjectOnPlane(forward, Vector3.up);
        if (horizontalForward.sqrMagnitude <= 1e-8f)
            horizontalForward = Vector3.forward;
        horizontalForward.Normalize();
        Vector3 horizontalRight = Vector3.Cross(Vector3.up, horizontalForward);
        if (horizontalRight.sqrMagnitude <= 1e-8f)
            horizontalRight = Vector3.right;
        horizontalRight.Normalize();

        AddProbeSurfaceGridOverlay(anchorPoint, anchorNormal, horizontalRight, spacing, maxDistance, maxPointCount, positions, normals, valid, hits, links);

        int pointCount = hits.Count;
        EnsureProbeRowObjects();
        EnsureProbeMarkerPool(pointCount);
        EnsureProbeLinePool(links.Count);

        int visiblePoints = 0;
        for (int i = 0; i < pointCount; i++)
        {
            if (!hits[i].valid)
            {
                SetProbeMarkerVisible(i, false);
                continue;
            }

            GameObject marker = _probeRowMarkers[i];
            ProbeRowHit hit = hits[i];
            marker.transform.position = hit.point + hit.normal * surfaceOffset;
            marker.transform.localScale = Vector3.one * Mathf.Max(0.002f, probeRowMarkerScaleMeters);
            SetProbeMarkerVisible(i, true);
            visiblePoints++;
        }

        for (int i = pointCount; i < _probeRowMarkers.Count; i++)
            SetProbeMarkerVisible(i, false);

        for (int i = 0; i < links.Count; i++)
        {
            LineRenderer line = _probeRowLines[i];
            int a = links[i].a;
            int b = links[i].b;
            if ((uint)a >= (uint)hits.Count || (uint)b >= (uint)hits.Count || !hits[a].valid || !hits[b].valid)
            {
                SetProbeLineVisible(line, false);
                continue;
            }

            ProbeRowLinkKind kind = ClassifyProbeLink(hits[a], hits[b], spacing);
            line.sharedMaterial = ResolveProbeLineMaterial(kind);
            line.startWidth = Mathf.Max(0.0005f, probeRowLineWidthMeters);
            line.endWidth = line.startWidth;
            line.positionCount = 2;
            line.useWorldSpace = true;
            line.SetPosition(0, hits[a].point + hits[a].normal * surfaceOffset);
            line.SetPosition(1, hits[b].point + hits[b].normal * surfaceOffset);
            SetProbeLineVisible(line, true);
        }

        for (int i = links.Count; i < _probeRowLines.Count; i++)
            SetProbeLineVisible(_probeRowLines[i], false);

        SetProbeRowExperimentVisible(visiblePoints > 0);
    }

    private void AddProbeSurfaceGridOverlay(Vector3 anchorPoint, Vector3 anchorNormal, Vector3 preferredRight, float spacing, float maxDistance, int maxPointCount, Vector3[] positions, Vector3[] normals, bool[] valid, List<ProbeRowHit> hits, List<ProbeRowLink> links)
    {
        Vector3 normal = anchorNormal.sqrMagnitude > 1e-8f ? anchorNormal.normalized : Vector3.up;
        Vector3 axisU = Vector3.ProjectOnPlane(preferredRight, normal);
        if (axisU.sqrMagnitude <= 1e-8f)
            axisU = Vector3.Cross(Vector3.up, normal);
        if (axisU.sqrMagnitude <= 1e-8f)
            axisU = Vector3.Cross(Vector3.right, normal);
        if (axisU.sqrMagnitude <= 1e-8f)
            return;
        axisU.Normalize();

        Vector3 axisV = Vector3.Cross(normal, axisU);
        if (Vector3.Dot(axisV, Vector3.up) < 0f)
            axisV = -axisV;
        if (axisV.sqrMagnitude <= 1e-8f)
            return;
        axisV.Normalize();

        if (!TryResolveProbeGridRanges(valid, positions, anchorPoint, normal, axisU, axisV, spacing, maxDistance, out float minU, out float maxU, out float minV, out float maxV))
            return;

        float gridSpacing = Mathf.Max(0.005f, spacing);
        int estimatedU = Mathf.Max(2, Mathf.FloorToInt((maxU - minU) / gridSpacing) + 1);
        int estimatedV = Mathf.Max(2, Mathf.FloorToInt((maxV - minV) / gridSpacing) + 1);
        int estimatedPoints = estimatedU * estimatedV * 2;
        if (estimatedPoints > maxPointCount)
        {
            float scale = Mathf.Sqrt(estimatedPoints / (float)Mathf.Max(1, maxPointCount));
            gridSpacing *= Mathf.Max(1f, scale);
            estimatedU = Mathf.Max(2, Mathf.FloorToInt((maxU - minU) / gridSpacing) + 1);
            estimatedV = Mathf.Max(2, Mathf.FloorToInt((maxV - minV) / gridSpacing) + 1);
        }

        AddProbeSurfaceGridFamily(anchorPoint, normal, axisU, axisV, minU, maxU, minV, maxV, gridSpacing, maxDistance, maxPointCount, positions, normals, valid, hits, links, linesAlongU: true);
        AddProbeSurfaceGridFamily(anchorPoint, normal, axisU, axisV, minU, maxU, minV, maxV, gridSpacing, maxDistance, maxPointCount, positions, normals, valid, hits, links, linesAlongU: false);
    }

    private bool TryResolveProbeGridRanges(bool[] valid, Vector3[] positions, Vector3 anchorPoint, Vector3 normal, Vector3 axisU, Vector3 axisV, float spacing, float maxDistance, out float minU, out float maxU, out float minV, out float maxV)
    {
        minU = float.PositiveInfinity;
        maxU = float.NegativeInfinity;
        minV = float.PositiveInfinity;
        maxV = float.NegativeInfinity;
        if (valid == null || positions == null)
            return false;

        float halfSpanLimit = Mathf.Max(spacing * 4f, Mathf.Min(Mathf.Max(0.3f, maxDistance), spacing * 32f));
        float normalRange = Mathf.Max(spacing * 6f, probeRowRecognitionRadiusMeters * 4f);
        int count = Mathf.Min(valid.Length, positions.Length);
        int accepted = 0;
        for (int i = 0; i < count; i++)
        {
            if (!valid[i] || !Finite(positions[i]))
                continue;

            Vector3 delta = positions[i] - anchorPoint;
            float normalOffset = Mathf.Abs(Vector3.Dot(delta, normal));
            if (normalOffset > normalRange)
                continue;

            float u = Mathf.Clamp(Vector3.Dot(delta, axisU), -halfSpanLimit, halfSpanLimit);
            float v = Mathf.Clamp(Vector3.Dot(delta, axisV), -halfSpanLimit, halfSpanLimit);
            minU = Mathf.Min(minU, u);
            maxU = Mathf.Max(maxU, u);
            minV = Mathf.Min(minV, v);
            maxV = Mathf.Max(maxV, v);
            accepted++;
        }

        if (accepted < 4 || !float.IsFinite(minU) || !float.IsFinite(maxU) || !float.IsFinite(minV) || !float.IsFinite(maxV))
        {
            float fallbackHalfSpan = Mathf.Max(spacing * 4f, Mathf.Min(0.9f, halfSpanLimit));
            minU = -fallbackHalfSpan;
            maxU = fallbackHalfSpan;
            minV = -fallbackHalfSpan;
            maxV = fallbackHalfSpan;
            return true;
        }

        minU = Mathf.Max(minU, -halfSpanLimit);
        maxU = Mathf.Min(maxU, halfSpanLimit);
        minV = Mathf.Max(minV, -halfSpanLimit);
        maxV = Mathf.Min(maxV, halfSpanLimit);
        return maxU > minU + spacing && maxV > minV + spacing;
    }

    private void AddProbeSurfaceGridFamily(Vector3 anchorPoint, Vector3 normal, Vector3 axisU, Vector3 axisV, float minU, float maxU, float minV, float maxV, float spacing, float maxDistance, int maxPointCount, Vector3[] positions, Vector3[] normals, bool[] valid, List<ProbeRowHit> hits, List<ProbeRowLink> links, bool linesAlongU)
    {
        float lineMin = linesAlongU ? minV : minU;
        float lineMax = linesAlongU ? maxV : maxU;
        float sampleMin = linesAlongU ? minU : minV;
        float sampleMax = linesAlongU ? maxU : maxV;
        int lineCount = Mathf.Max(1, Mathf.FloorToInt((lineMax - lineMin) / spacing) + 1);
        int sampleCount = Mathf.Max(2, Mathf.FloorToInt((sampleMax - sampleMin) / spacing) + 1);

        for (int lineIndex = 0; lineIndex < lineCount && hits.Count < maxPointCount; lineIndex++)
        {
            float lineT = lineCount > 1 ? lineIndex / (float)(lineCount - 1) : 0.5f;
            float lineOffset = Mathf.Lerp(lineMin, lineMax, lineT);
            int previousIndex = -1;
            for (int sampleIndex = 0; sampleIndex < sampleCount && hits.Count < maxPointCount; sampleIndex++)
            {
                float sampleT = sampleCount > 1 ? sampleIndex / (float)(sampleCount - 1) : 0.5f;
                float sampleOffset = Mathf.Lerp(sampleMin, sampleMax, sampleT);
                float u = linesAlongU ? sampleOffset : lineOffset;
                float v = linesAlongU ? lineOffset : sampleOffset;
                Vector3 planePoint = anchorPoint + axisU * u + axisV * v;
                if (!TryProjectProbeSurfacePoint(planePoint, normal, positions, normals, maxDistance, out Vector3 projectedPoint, out Vector3 projectedNormal))
                {
                    previousIndex = -1;
                    continue;
                }

                Vector3 sampleNormal = projectedNormal.sqrMagnitude > 1e-8f ? projectedNormal.normalized : normal;
                ProbeRowHit hit = new ProbeRowHit
                {
                    valid = true,
                    point = projectedPoint,
                    normal = sampleNormal
                };
                EvaluateProbeNeighborhood(hit.point, hit.normal, valid, positions, normals, out hit.neighborhoodCount, out hit.planeDeviation);
                int currentIndex = hits.Count;
                hits.Add(hit);
                if (previousIndex >= 0)
                    links.Add(new ProbeRowLink { a = previousIndex, b = currentIndex });
                previousIndex = currentIndex;
            }
        }
    }

    private void AddProbeCenterSliceLine(Vector3 planeNormal, Vector3 anchorPoint, Vector3 fallbackNormal, float spacing, float maxDistance, int maxPointCount, Vector3[] positions, Vector3[] normals, bool[] valid, List<ProbeRowHit> hits, List<ProbeRowLink> links, List<ProbeSliceSegment> sliceSegments, List<Vector3> slicePolyline)
    {
        if (planeNormal.sqrMagnitude <= 1e-8f || hits.Count >= maxPointCount)
            return;

        Vector3 n = planeNormal.normalized;
        BuildProbeSliceSegments(anchorPoint, n, positions, sliceSegments);
        if (sliceSegments.Count <= 0)
            return;

        slicePolyline.Clear();
        if (!TryBuildProbeSlicePolyline(sliceSegments, anchorPoint, spacing, slicePolyline) || slicePolyline.Count < 2)
            return;

        float[] cumulative = BuildProbePolylineDistances(slicePolyline);
        float totalLength = cumulative.Length > 0 ? cumulative[cumulative.Length - 1] : 0f;
        if (totalLength <= spacing * 0.5f)
            return;

        int lineStart = hits.Count;
        int availablePoints = Mathf.Max(0, maxPointCount - hits.Count);
        int sampleCount = Mathf.Min(availablePoints, Mathf.Max(2, Mathf.FloorToInt(totalLength / spacing) + 1));
        if (sampleCount < 2)
            return;

        for (int i = 0; i < sampleCount && hits.Count < maxPointCount; i++)
        {
            float sampleDistance = sampleCount > 1
                ? Mathf.Lerp(0f, totalLength, i / (float)(sampleCount - 1))
                : 0f;
            if (sampleDistance < -1e-4f || sampleDistance > totalLength + 1e-4f ||
                !TrySampleProbePolyline(slicePolyline, cumulative, sampleDistance, out Vector3 samplePoint))
                continue;

            Vector3 sampleNormal = ResolveNearestProbeNormal(samplePoint, valid, positions, normals, fallbackNormal, maxDistance);
            ProbeRowHit hit = new ProbeRowHit
            {
                valid = true,
                point = samplePoint,
                normal = sampleNormal
            };
            EvaluateProbeNeighborhood(hit.point, hit.normal, valid, positions, normals, out hit.neighborhoodCount, out hit.planeDeviation);
            int index = hits.Count;
            hits.Add(hit);
            if (index > lineStart)
                links.Add(new ProbeRowLink { a = index - 1, b = index });
        }
    }

    private bool TryResolveProbeBounds(bool[] valid, Vector3[] positions, out Bounds bounds)
    {
        bounds = default;
        if (valid == null || positions == null)
            return false;

        bool hasValue = false;
        int count = Mathf.Min(valid.Length, positions.Length);
        for (int i = 0; i < count; i++)
        {
            if (!valid[i] || !Finite(positions[i]))
                continue;

            if (!hasValue)
            {
                bounds = new Bounds(positions[i], Vector3.zero);
                hasValue = true;
            }
            else
            {
                bounds.Encapsulate(positions[i]);
            }
        }

        return hasValue;
    }

    private static void GetProbeBoundsAxisRange(Bounds bounds, Vector3 axis, out float minOffset, out float maxOffset)
    {
        minOffset = float.PositiveInfinity;
        maxOffset = float.NegativeInfinity;
        Vector3 center = bounds.center;
        Vector3 extents = bounds.extents;
        for (int x = -1; x <= 1; x += 2)
        {
            for (int y = -1; y <= 1; y += 2)
            {
                for (int z = -1; z <= 1; z += 2)
                {
                    Vector3 corner = center + new Vector3(extents.x * x, extents.y * y, extents.z * z);
                    float offset = Vector3.Dot(corner, axis);
                    minOffset = Mathf.Min(minOffset, offset);
                    maxOffset = Mathf.Max(maxOffset, offset);
                }
            }
        }
    }

    private void BuildProbeSliceSegments(Vector3 planePoint, Vector3 planeNormal, Vector3[] positions, List<ProbeSliceSegment> segments)
    {
        segments.Clear();
        if (positions == null || planeNormal.sqrMagnitude <= 1e-8f)
            return;

        Vector3 n = planeNormal.normalized;
        const float epsilon = 0.0005f;
        List<Vector3> crossings = new List<Vector3>(3);
        for (int i = 0; i + 2 < _probeTriangleIndices.Count; i += 3)
        {
            int ia = _probeTriangleIndices[i];
            int ib = _probeTriangleIndices[i + 1];
            int ic = _probeTriangleIndices[i + 2];
            if ((uint)ia >= (uint)positions.Length || (uint)ib >= (uint)positions.Length || (uint)ic >= (uint)positions.Length)
                continue;

            Vector3 a = positions[ia];
            Vector3 b = positions[ib];
            Vector3 c = positions[ic];
            if (!Finite(a) || !Finite(b) || !Finite(c))
                continue;

            float da = Vector3.Dot(a - planePoint, n);
            float db = Vector3.Dot(b - planePoint, n);
            float dc = Vector3.Dot(c - planePoint, n);
            if ((da > epsilon && db > epsilon && dc > epsilon) || (da < -epsilon && db < -epsilon && dc < -epsilon))
                continue;

            crossings.Clear();
            AddProbeSliceEdgeCrossing(a, da, b, db, epsilon, crossings);
            AddProbeSliceEdgeCrossing(b, db, c, dc, epsilon, crossings);
            AddProbeSliceEdgeCrossing(c, dc, a, da, epsilon, crossings);
            RemoveDuplicateProbeSlicePoints(crossings, epsilon * 4f);
            if (crossings.Count < 2)
                continue;

            Vector3 p0 = crossings[0];
            Vector3 p1 = crossings[1];
            float bestSqr = (p1 - p0).sqrMagnitude;
            for (int p = 0; p < crossings.Count; p++)
            {
                for (int q = p + 1; q < crossings.Count; q++)
                {
                    float sqr = (crossings[q] - crossings[p]).sqrMagnitude;
                    if (sqr > bestSqr)
                    {
                        bestSqr = sqr;
                        p0 = crossings[p];
                        p1 = crossings[q];
                    }
                }
            }

            if (bestSqr > 1e-8f)
                segments.Add(new ProbeSliceSegment { a = p0, b = p1 });
        }
    }

    private static void AddProbeSliceEdgeCrossing(Vector3 a, float da, Vector3 b, float db, float epsilon, List<Vector3> crossings)
    {
        bool aOn = Mathf.Abs(da) <= epsilon;
        bool bOn = Mathf.Abs(db) <= epsilon;
        if (aOn)
            crossings.Add(a);
        if (bOn)
            crossings.Add(b);
        if (aOn || bOn || da * db > 0f)
            return;

        float t = da / (da - db);
        if (t >= 0f && t <= 1f)
            crossings.Add(Vector3.LerpUnclamped(a, b, t));
    }

    private static void RemoveDuplicateProbeSlicePoints(List<Vector3> points, float tolerance)
    {
        float toleranceSqr = tolerance * tolerance;
        for (int i = points.Count - 1; i >= 0; i--)
        {
            for (int j = 0; j < i; j++)
            {
                if ((points[i] - points[j]).sqrMagnitude <= toleranceSqr)
                {
                    points.RemoveAt(i);
                    break;
                }
            }
        }
    }

    private bool TryBuildProbeSlicePolyline(List<ProbeSliceSegment> segments, Vector3 anchorPoint, float spacing, List<Vector3> polyline)
    {
        polyline.Clear();
        if (segments == null || segments.Count <= 0)
            return false;

        int seed = -1;
        float bestSqr = float.PositiveInfinity;
        for (int i = 0; i < segments.Count; i++)
        {
            float sqr = DistancePointSegmentSqr(anchorPoint, segments[i].a, segments[i].b);
            if (sqr < bestSqr)
            {
                bestSqr = sqr;
                seed = i;
            }
        }

        if (seed < 0)
            return false;

        bool[] used = new bool[segments.Count];
        used[seed] = true;
        Vector3 seedA = segments[seed].a;
        Vector3 seedB = segments[seed].b;
        if ((seedA - anchorPoint).sqrMagnitude < (seedB - anchorPoint).sqrMagnitude)
        {
            polyline.Add(seedB);
            polyline.Add(seedA);
        }
        else
        {
            polyline.Add(seedA);
            polyline.Add(seedB);
        }

        float connectTolerance = Mathf.Max(0.01f, Mathf.Min(0.08f, spacing * 1.25f));
        ExtendProbeSlicePolyline(polyline, segments, used, connectTolerance, appendToEnd: true);
        ExtendProbeSlicePolyline(polyline, segments, used, connectTolerance, appendToEnd: false);
        return polyline.Count >= 2;
    }

    private static void ExtendProbeSlicePolyline(List<Vector3> polyline, List<ProbeSliceSegment> segments, bool[] used, float connectTolerance, bool appendToEnd)
    {
        float toleranceSqr = connectTolerance * connectTolerance;
        bool added = true;
        while (added)
        {
            added = false;
            Vector3 end = appendToEnd ? polyline[polyline.Count - 1] : polyline[0];
            int bestIndex = -1;
            bool useA = true;
            float bestSqr = toleranceSqr;
            for (int i = 0; i < segments.Count; i++)
            {
                if (used[i])
                    continue;

                float aSqr = (segments[i].a - end).sqrMagnitude;
                if (aSqr <= bestSqr)
                {
                    bestSqr = aSqr;
                    bestIndex = i;
                    useA = true;
                }

                float bSqr = (segments[i].b - end).sqrMagnitude;
                if (bSqr <= bestSqr)
                {
                    bestSqr = bSqr;
                    bestIndex = i;
                    useA = false;
                }
            }

            if (bestIndex < 0)
                continue;

            used[bestIndex] = true;
            Vector3 next = useA ? segments[bestIndex].b : segments[bestIndex].a;
            if (appendToEnd)
                polyline.Add(next);
            else
                polyline.Insert(0, next);
            added = true;
        }
    }

    private static float[] BuildProbePolylineDistances(List<Vector3> polyline)
    {
        float[] distances = new float[polyline != null ? polyline.Count : 0];
        if (polyline == null || polyline.Count <= 0)
            return distances;

        distances[0] = 0f;
        for (int i = 1; i < polyline.Count; i++)
            distances[i] = distances[i - 1] + Vector3.Distance(polyline[i - 1], polyline[i]);
        return distances;
    }

    private static float FindClosestDistanceOnProbePolyline(List<Vector3> polyline, float[] cumulative, Vector3 point)
    {
        if (polyline == null || cumulative == null || polyline.Count < 2 || cumulative.Length != polyline.Count)
            return 0f;

        float bestSqr = float.PositiveInfinity;
        float bestDistance = 0f;
        for (int i = 0; i < polyline.Count - 1; i++)
        {
            Vector3 a = polyline[i];
            Vector3 b = polyline[i + 1];
            Vector3 ab = b - a;
            float abSqr = ab.sqrMagnitude;
            if (abSqr <= 1e-8f)
                continue;

            float t = Mathf.Clamp01(Vector3.Dot(point - a, ab) / abSqr);
            Vector3 closest = a + ab * t;
            float sqr = (point - closest).sqrMagnitude;
            if (sqr < bestSqr)
            {
                bestSqr = sqr;
                bestDistance = cumulative[i] + Mathf.Sqrt(abSqr) * t;
            }
        }

        return bestDistance;
    }

    private static bool TrySampleProbePolyline(List<Vector3> polyline, float[] cumulative, float distance, out Vector3 point)
    {
        point = Vector3.zero;
        if (polyline == null || cumulative == null || polyline.Count < 2 || cumulative.Length != polyline.Count)
            return false;

        float total = cumulative[cumulative.Length - 1];
        if (distance < -1e-4f || distance > total + 1e-4f)
            return false;

        distance = Mathf.Clamp(distance, 0f, total);
        for (int i = 0; i < cumulative.Length - 1; i++)
        {
            float a = cumulative[i];
            float b = cumulative[i + 1];
            if (distance > b && i < cumulative.Length - 2)
                continue;

            float span = b - a;
            float t = span > 1e-8f ? (distance - a) / span : 0f;
            point = Vector3.LerpUnclamped(polyline[i], polyline[i + 1], t);
            return true;
        }

        point = polyline[polyline.Count - 1];
        return true;
    }

    private Vector3 ResolveNearestProbeNormal(Vector3 point, bool[] valid, Vector3[] positions, Vector3[] normals, Vector3 fallbackNormal, float maxDistance)
    {
        Vector3 bestNormal = fallbackNormal.sqrMagnitude > 1e-8f ? fallbackNormal.normalized : Vector3.up;
        if (valid == null || positions == null || normals == null)
            return bestNormal;

        float bestSqr = Mathf.Max(0.0025f, maxDistance * maxDistance);
        int count = Mathf.Min(valid.Length, Mathf.Min(positions.Length, normals.Length));
        for (int i = 0; i < count; i++)
        {
            if (!valid[i] || !Finite(positions[i]) || !Finite(normals[i]) || normals[i].sqrMagnitude <= 1e-8f)
                continue;

            float sqr = (positions[i] - point).sqrMagnitude;
            if (sqr >= bestSqr)
                continue;

            bestSqr = sqr;
            bestNormal = normals[i].normalized;
        }

        return bestNormal;
    }

    private static float DistancePointSegmentSqr(Vector3 point, Vector3 a, Vector3 b)
    {
        Vector3 ab = b - a;
        float abSqr = ab.sqrMagnitude;
        if (abSqr <= 1e-8f)
            return (point - a).sqrMagnitude;

        float t = Mathf.Clamp01(Vector3.Dot(point - a, ab) / abSqr);
        Vector3 closest = a + ab * t;
        return (point - closest).sqrMagnitude;
    }

    private bool TryStepProbeSurface(ProbeRowHit from, Vector3 preferredDirection, float spacing, float projectDistance, Vector3[] positions, Vector3[] normals, bool[] valid, out ProbeRowHit hit)
    {
        hit = default;
        if (!from.valid || preferredDirection.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 normal = from.normal.sqrMagnitude > 1e-8f ? from.normal.normalized : Vector3.up;
        Vector3 tangent = Vector3.ProjectOnPlane(preferredDirection, normal);
        if (tangent.sqrMagnitude <= 1e-8f)
            tangent = preferredDirection;
        if (tangent.sqrMagnitude <= 1e-8f)
            return false;

        tangent.Normalize();
        Vector3 surfaceTarget = from.point + tangent * Mathf.Max(0.005f, spacing);
        if (!TryProjectProbeSurfacePoint(surfaceTarget, normal, positions, normals, projectDistance, out Vector3 hitPoint, out Vector3 hitNormal))
            return false;

        Vector3 resolvedNormal = hitNormal.sqrMagnitude > 1e-8f ? hitNormal.normalized : normal;
        hit = new ProbeRowHit
        {
            valid = true,
            point = hitPoint,
            normal = resolvedNormal
        };
        EvaluateProbeNeighborhood(hit.point, hit.normal, valid, positions, normals, out hit.neighborhoodCount, out hit.planeDeviation);
        return true;
    }

    private bool TryResolveProbeHorizontalExtent(bool[] valid, Vector3[] positions, Vector3 originPos, Vector3 forward, Vector3 right, float targetDistance, out float minOffset, out float maxOffset)
    {
        minOffset = float.PositiveInfinity;
        maxOffset = float.NegativeInfinity;
        if (valid == null || positions == null || forward.sqrMagnitude <= 1e-8f || right.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 fwd = forward.normalized;
        Vector3 axis = right.normalized;
        float planeDistance = Mathf.Max(0.05f, targetDistance);
        int count = Mathf.Min(valid.Length, positions.Length);
        int validCount = 0;
        for (int i = 0; i < count; i++)
        {
            if (!valid[i] || !Finite(positions[i]))
                continue;

            Vector3 toPoint = positions[i] - originPos;
            float depth = Vector3.Dot(toPoint, fwd);
            if (depth <= 0.02f)
                continue;

            float offset = Vector3.Dot(toPoint, axis) * (planeDistance / depth);
            if (!float.IsFinite(offset))
                continue;

            minOffset = Mathf.Min(minOffset, offset);
            maxOffset = Mathf.Max(maxOffset, offset);
            validCount++;
        }

        return validCount >= 2 && float.IsFinite(minOffset) && float.IsFinite(maxOffset) && maxOffset > minOffset + 0.01f;
    }

    private bool TryResolveProbeSurfaceHorizontalExtent(bool[] valid, Vector3[] positions, Vector3[] normals, Vector3 anchorPoint, Vector3 anchorNormal, Vector3 surfaceRight, out float minOffset, out float maxOffset)
    {
        minOffset = float.PositiveInfinity;
        maxOffset = float.NegativeInfinity;
        if (valid == null || positions == null || surfaceRight.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 axis = surfaceRight.normalized;
        Vector3 normal = anchorNormal.sqrMagnitude > 1e-8f ? anchorNormal.normalized : Vector3.up;
        int count = Mathf.Min(valid.Length, positions.Length);
        int validCount = 0;
        for (int i = 0; i < count; i++)
        {
            if (!valid[i] || !Finite(positions[i]))
                continue;

            if (normals != null && i < normals.Length && Finite(normals[i]) && normals[i].sqrMagnitude > 1e-8f &&
                Vector3.Dot(normal, normals[i].normalized) < probeRowRefineNormalDot)
                continue;

            float offset = Vector3.Dot(positions[i] - anchorPoint, axis);
            if (!float.IsFinite(offset))
                continue;

            minOffset = Mathf.Min(minOffset, offset);
            maxOffset = Mathf.Max(maxOffset, offset);
            validCount++;
        }

        return validCount >= 2 && float.IsFinite(minOffset) && float.IsFinite(maxOffset) && maxOffset > minOffset + 0.01f;
    }

    private bool TryProjectProbeSurfacePoint(Vector3 surfaceTarget, Vector3 anchorNormal, Vector3[] positions, Vector3[] normals, float maxDistance, out Vector3 hitPoint, out Vector3 hitNormal)
    {
        hitPoint = Vector3.zero;
        hitNormal = Vector3.zero;
        Vector3 normal = anchorNormal.sqrMagnitude > 1e-8f ? anchorNormal.normalized : Vector3.up;
        float castDistance = Mathf.Max(0.1f, maxDistance);
        Vector3 start = surfaceTarget + normal * (castDistance * 0.5f);
        Ray ray = new Ray(start, -normal);
        if (TryRaycastProbeMesh(ray, positions, normals, castDistance, out hitPoint, out hitNormal))
            return true;

        start = surfaceTarget - normal * (castDistance * 0.5f);
        ray = new Ray(start, normal);
        return TryRaycastProbeMesh(ray, positions, normals, castDistance, out hitPoint, out hitNormal);
    }

    private void BuildProbeTriangleIndexCache(bool[] valid)
    {
        _probeTriangleIndices.Clear();
        if (valid == null || valid.Length < _cells.Count || _groups.Count <= 0)
            return;

        for (int g = 0; g < _groups.Count; g++)
        {
            GridGroup group = _groups[g];
            if (group.columns <= 1 || group.rows <= 1)
                continue;

            for (int row = 0; row < group.rows - 1; row++)
            {
                int rowStart = group.startIndex + row * group.columns;
                int nextRowStart = group.startIndex + (row + 1) * group.columns;
                for (int col = 0; col < group.columns - 1; col++)
                {
                    int i00 = rowStart + col;
                    int i10 = rowStart + col + 1;
                    int i01 = nextRowStart + col;
                    int i11 = nextRowStart + col + 1;
                    if ((uint)i00 >= (uint)valid.Length || (uint)i10 >= (uint)valid.Length ||
                        (uint)i01 >= (uint)valid.Length || (uint)i11 >= (uint)valid.Length)
                        continue;

                    if (valid[i00] && valid[i10] && valid[i11])
                        AddProbeTriangle(i00, i10, i11);
                    if (valid[i00] && valid[i11] && valid[i01])
                        AddProbeTriangle(i00, i11, i01);
                    if (!valid[i11] && valid[i00] && valid[i10] && valid[i01])
                        AddProbeTriangle(i00, i10, i01);
                    if (!valid[i00] && valid[i10] && valid[i11] && valid[i01])
                        AddProbeTriangle(i10, i11, i01);
                }
            }
        }
    }

    private void AddProbeTriangle(int a, int b, int c)
    {
        _probeTriangleIndices.Add(a);
        _probeTriangleIndices.Add(b);
        _probeTriangleIndices.Add(c);
    }

    private bool TryRaycastProbeMesh(Ray ray, Vector3[] positions, Vector3[] normals, float maxDistance, out Vector3 hitPoint, out Vector3 hitNormal)
    {
        hitPoint = Vector3.zero;
        hitNormal = Vector3.zero;
        float bestDistance = maxDistance;
        bool hasHit = false;

        for (int i = 0; i + 2 < _probeTriangleIndices.Count; i += 3)
        {
            int ia = _probeTriangleIndices[i];
            int ib = _probeTriangleIndices[i + 1];
            int ic = _probeTriangleIndices[i + 2];
            if ((uint)ia >= (uint)positions.Length || (uint)ib >= (uint)positions.Length || (uint)ic >= (uint)positions.Length)
                continue;

            Vector3 a = positions[ia];
            Vector3 b = positions[ib];
            Vector3 c = positions[ic];
            if (!Finite(a) || !Finite(b) || !Finite(c))
                continue;

            if (!TryIntersectRayTriangle(ray, a, b, c, bestDistance, out float distance, out _, out _))
                continue;

            bestDistance = distance;
            hitPoint = ray.origin + ray.direction * distance;
            Vector3 normal = Vector3.zero;
            if ((uint)ia < (uint)normals.Length && Finite(normals[ia])) normal += normals[ia];
            if ((uint)ib < (uint)normals.Length && Finite(normals[ib])) normal += normals[ib];
            if ((uint)ic < (uint)normals.Length && Finite(normals[ic])) normal += normals[ic];
            if (normal.sqrMagnitude <= 1e-8f)
                normal = Vector3.Cross(b - a, c - a);
            hitNormal = normal.sqrMagnitude > 1e-8f ? normal.normalized : -ray.direction;
            if (Vector3.Dot(hitNormal, ray.direction) > 0f)
                hitNormal = -hitNormal;
            hasHit = true;
        }

        return hasHit;
    }

    private static bool TryIntersectRayTriangle(Ray ray, Vector3 a, Vector3 b, Vector3 c, float maxDistance, out float distance, out float u, out float v)
    {
        distance = 0f;
        u = 0f;
        v = 0f;
        Vector3 edge1 = b - a;
        Vector3 edge2 = c - a;
        Vector3 p = Vector3.Cross(ray.direction, edge2);
        float det = Vector3.Dot(edge1, p);
        if (Mathf.Abs(det) < 1e-7f)
            return false;

        float invDet = 1f / det;
        Vector3 t = ray.origin - a;
        u = Vector3.Dot(t, p) * invDet;
        if (u < 0f || u > 1f)
            return false;

        Vector3 q = Vector3.Cross(t, edge1);
        v = Vector3.Dot(ray.direction, q) * invDet;
        if (v < 0f || u + v > 1f)
            return false;

        distance = Vector3.Dot(edge2, q) * invDet;
        return distance > 1e-5f && distance <= maxDistance;
    }

    private void EvaluateProbeNeighborhood(Vector3 point, Vector3 normal, bool[] valid, Vector3[] positions, Vector3[] normals, out int count, out float planeDeviation)
    {
        count = 0;
        planeDeviation = float.PositiveInfinity;
        if (valid == null || positions == null)
            return;

        float radius = Mathf.Max(0.005f, probeRowRecognitionRadiusMeters);
        float radiusSqr = radius * radius;
        float totalAbsPlaneDistance = 0f;
        Vector3 n = normal.sqrMagnitude > 1e-8f ? normal.normalized : Vector3.up;
        for (int i = 0; i < valid.Length && i < positions.Length; i++)
        {
            if (!valid[i])
                continue;
            Vector3 sample = positions[i];
            if (!Finite(sample) || (sample - point).sqrMagnitude > radiusSqr)
                continue;

            if (i < normals.Length && Finite(normals[i]) && normals[i].sqrMagnitude > 1e-8f &&
                Vector3.Dot(n, normals[i].normalized) < probeRowRefineNormalDot)
                continue;

            totalAbsPlaneDistance += Mathf.Abs(Vector3.Dot(sample - point, n));
            count++;
        }

        if (count > 0)
            planeDeviation = totalAbsPlaneDistance / count;
    }

    private ProbeRowLinkKind ClassifyProbeLink(ProbeRowHit a, ProbeRowHit b, float spacing)
    {
        float distance = Vector3.Distance(a.point, b.point);
        float normalDot = a.normal.sqrMagnitude > 1e-8f && b.normal.sqrMagnitude > 1e-8f
            ? Vector3.Dot(a.normal.normalized, b.normal.normalized)
            : -1f;
        bool enoughSamples = a.neighborhoodCount >= probeRowMinNeighborhoodSamples &&
                             b.neighborhoodCount >= probeRowMinNeighborhoodSamples;
        float maxDeviation = Mathf.Max(a.planeDeviation, b.planeDeviation);
        float stableDistance = spacing * Mathf.Max(1f, probeRowStableDistanceMultiplier);
        float refineDistance = spacing * Mathf.Max(probeRowStableDistanceMultiplier, probeRowRefineDistanceMultiplier);

        if (enoughSamples &&
            distance <= stableDistance &&
            normalDot >= probeRowStableNormalDot &&
            maxDeviation <= probeRowStablePlaneDeviationMeters)
            return ProbeRowLinkKind.Stable;

        if (distance <= refineDistance && normalDot >= probeRowRefineNormalDot)
            return ProbeRowLinkKind.Refine;

        return ProbeRowLinkKind.Break;
    }

    private void EnsureProbeRowObjects()
    {
        if (_probeRowRoot == null)
        {
            _probeRowRoot = new GameObject("[ScanCover] Probe Row Experiment");
            _probeRowRoot.transform.SetParent(null, false);
        }

        EnsureProbeMaterials();
    }

    private void EnsureProbeMaterials()
    {
        _probeRowPointMaterial = EnsureProbeMaterial(_probeRowPointMaterial, "ScanCover_ProbeRow_Point", probeRowPointColor);
        _probeRowStableLineMaterial = EnsureProbeMaterial(_probeRowStableLineMaterial, "ScanCover_ProbeRow_StableLine", probeRowStableLineColor);
        _probeRowRefineLineMaterial = EnsureProbeMaterial(_probeRowRefineLineMaterial, "ScanCover_ProbeRow_RefineLine", probeRowRefineLineColor);
        _probeRowBreakLineMaterial = EnsureProbeMaterial(_probeRowBreakLineMaterial, "ScanCover_ProbeRow_BreakLine", probeRowBreakLineColor);
    }

    private Material EnsureProbeMaterial(Material material, string materialName, Color color)
    {
        if (material == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            material = new Material(shader) { name = materialName };
        }

        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        material.renderQueue = 3100;
        return material;
    }

    private void EnsureProbeMarkerPool(int count)
    {
        while (_probeRowMarkers.Count < count)
        {
            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.name = $"ProbePoint_{_probeRowMarkers.Count:00}";
            marker.transform.SetParent(_probeRowRoot.transform, false);
            Collider markerCollider = marker.GetComponent<Collider>();
            if (markerCollider != null)
                Destroy(markerCollider);
            Renderer renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = _probeRowPointMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
            marker.SetActive(false);
            _probeRowMarkers.Add(marker);
        }
    }

    private void EnsureProbeLinePool(int count)
    {
        while (_probeRowLines.Count < count)
        {
            GameObject lineObject = new GameObject($"ProbeLink_{_probeRowLines.Count:00}");
            lineObject.transform.SetParent(_probeRowRoot.transform, false);
            LineRenderer line = lineObject.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.positionCount = 0;
            line.numCornerVertices = 2;
            line.numCapVertices = 2;
            line.shadowCastingMode = ShadowCastingMode.Off;
            line.receiveShadows = false;
            line.sharedMaterial = _probeRowStableLineMaterial;
            lineObject.SetActive(false);
            _probeRowLines.Add(line);
        }
    }

    private Material ResolveProbeLineMaterial(ProbeRowLinkKind kind)
    {
        switch (kind)
        {
            case ProbeRowLinkKind.Stable:
                return _probeRowStableLineMaterial;
            case ProbeRowLinkKind.Refine:
                return _probeRowRefineLineMaterial;
            default:
                return _probeRowBreakLineMaterial;
        }
    }

    private void SetProbeMarkerVisible(int index, bool visible)
    {
        if (index < 0 || index >= _probeRowMarkers.Count || _probeRowMarkers[index] == null)
            return;
        if (_probeRowMarkers[index].activeSelf != visible)
            _probeRowMarkers[index].SetActive(visible);
    }

    private static void SetProbeLineVisible(LineRenderer line, bool visible)
    {
        if (line == null)
            return;
        if (line.gameObject.activeSelf != visible)
            line.gameObject.SetActive(visible);
        if (!visible)
            line.positionCount = 0;
    }

    private void SetProbeRowExperimentVisible(bool visible)
    {
        if (_probeRowRoot != null && _probeRowRoot.activeSelf != visible)
            _probeRowRoot.SetActive(visible);
    }

    private void EnsureSurfaceObjects()
    {
        EnsureDisplayRoots();
        if (_surfaceRoot == null) { _surfaceRoot = new GameObject("[ScanCover] DepthGrid Surface"); _surfaceRoot.transform.SetParent(ResolveSurfaceParent(), false); _surfaceFilter = _surfaceRoot.AddComponent<MeshFilter>(); _surfaceRenderer = _surfaceRoot.AddComponent<MeshRenderer>(); }
        else if (_surfaceRoot.transform.parent != ResolveSurfaceParent()) { _surfaceRoot.transform.SetParent(ResolveSurfaceParent(), false); }
        UpdateDisplayRootsTransform();
        if (_surfaceMesh == null) { _surfaceMesh = new Mesh { name = "ScanCover_DepthGridSurface" }; _surfaceMesh.MarkDynamic(); }
        if (surfaceMaterialOverride != null)
        {
            if (_surfaceMaterial != null && _surfaceMaterial != surfaceMaterialOverride) Destroy(_surfaceMaterial);
            _surfaceMaterial = null;
        }
        else if (_surfaceMaterial == null || !IsLitShader(_surfaceMaterial.shader))
        {
            if (_surfaceMaterial != null) Destroy(_surfaceMaterial);
            Shader shader = Shader.Find("Universal Render Pipeline/Lit"); if (shader == null) shader = Shader.Find("Standard");
            _surfaceMaterial = new Material(shader) { name = "ScanCover_DepthGridSurface_Mat" };
        }
        ApplySurfaceMaterialSettings();
        _surfaceFilter.sharedMesh = _surfaceMesh;
        Material targetMaterial = surfaceMaterialOverride != null ? surfaceMaterialOverride : _surfaceMaterial;
        if (_surfaceRenderer.sharedMaterial != targetMaterial) _surfaceRenderer.sharedMaterial = targetMaterial;
        _surfaceRenderer.shadowCastingMode = ShadowCastingMode.Off; _surfaceRenderer.receiveShadows = true;
    }

    private void EnsureCandidatePlaneObjects(int count)
    {
        EnsureDisplayRoots();
        Transform parent = ResolveSurfaceParent();
        if (parent == null)
            return;

        if (_candidatePlaneRoot == null)
        {
            _candidatePlaneRoot = new GameObject("[ScanCover] Candidate Plane Objects");
            _candidatePlaneRoot.transform.SetParent(parent, false);
        }
        else if (_candidatePlaneRoot.transform.parent != parent)
        {
            _candidatePlaneRoot.transform.SetParent(parent, false);
        }

        UpdateDisplayRootsTransform();
        EnsureCandidatePlaneMaterial();
        while (_candidatePlaneFilters.Count < count)
        {
            int index = _candidatePlaneFilters.Count;
            GameObject planeObject = new GameObject($"[ScanCover] CandidatePlane_{index:00}");
            planeObject.transform.SetParent(_candidatePlaneRoot.transform, false);
            MeshFilter filter = planeObject.AddComponent<MeshFilter>();
            MeshRenderer renderer = planeObject.AddComponent<MeshRenderer>();
            Mesh mesh = new Mesh { name = $"ScanCover_CandidatePlane_{index:00}" };
            mesh.MarkDynamic();
            filter.sharedMesh = mesh;
            renderer.sharedMaterial = _candidatePlaneMaterial;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.enabled = false;
            _candidatePlaneFilters.Add(filter);
            _candidatePlaneRenderers.Add(renderer);
            _candidatePlaneMeshes.Add(mesh);
        }

        for (int i = 0; i < _candidatePlaneRenderers.Count; i++)
        {
            if (_candidatePlaneRenderers[i] != null)
                _candidatePlaneRenderers[i].sharedMaterial = _candidatePlaneMaterial;
        }
    }

    private void EnsureCandidatePlaneMaterial()
    {
        Color color = candidatePlaneColor;
        color.a = Mathf.Clamp01(candidatePlaneAlpha);
        if (_candidatePlaneMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            _candidatePlaneMaterial = new Material(shader) { name = "ScanCover_CandidatePlane_Mat" };
        }

        if (_candidatePlaneMaterial.HasProperty("_BaseColor")) _candidatePlaneMaterial.SetColor("_BaseColor", color);
        if (_candidatePlaneMaterial.HasProperty("_Color")) _candidatePlaneMaterial.SetColor("_Color", color);
        if (_candidatePlaneMaterial.HasProperty("_Surface")) _candidatePlaneMaterial.SetFloat("_Surface", 1f);
        if (_candidatePlaneMaterial.HasProperty("_Blend")) _candidatePlaneMaterial.SetFloat("_Blend", 0f);
        if (_candidatePlaneMaterial.HasProperty("_ZWrite")) _candidatePlaneMaterial.SetFloat("_ZWrite", 0f);
        if (_candidatePlaneMaterial.HasProperty("_SrcBlend")) _candidatePlaneMaterial.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (_candidatePlaneMaterial.HasProperty("_DstBlend")) _candidatePlaneMaterial.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        if (_candidatePlaneMaterial.HasProperty("_Cull")) _candidatePlaneMaterial.SetFloat("_Cull", (float)CullMode.Off);
        if (_candidatePlaneMaterial.HasProperty("_CullMode")) _candidatePlaneMaterial.SetFloat("_CullMode", (float)CullMode.Off);
        _candidatePlaneMaterial.renderQueue = 3050;
    }

    private void EnsureSurfaceNormalObjects()
    {
        EnsureSurfaceObjects();
        if (_surfaceNormalRoot == null)
        {
            _surfaceNormalRoot = new GameObject("[ScanCover] DepthGrid SurfaceNormals");
            _surfaceNormalRoot.transform.SetParent(ResolveSurfaceParent(), false);
            _surfaceNormalFilter = _surfaceNormalRoot.AddComponent<MeshFilter>();
            _surfaceNormalRenderer = _surfaceNormalRoot.AddComponent<MeshRenderer>();
        }
        else if (_surfaceNormalRoot.transform.parent != ResolveSurfaceParent())
        {
            _surfaceNormalRoot.transform.SetParent(ResolveSurfaceParent(), false);
        }

        UpdateDisplayRootsTransform();
        if (_surfaceNormalMesh == null)
        {
            _surfaceNormalMesh = new Mesh { name = "ScanCover_DepthGridSurfaceNormals" };
            _surfaceNormalMesh.MarkDynamic();
        }

        if (_surfaceNormalMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader != null)
                _surfaceNormalMaterial = new Material(shader) { name = "ScanCover_DepthGridSurfaceNormals_Mat" };
        }

        if (_surfaceNormalMaterial != null)
        {
            if (_surfaceNormalMaterial.HasProperty("_BaseColor"))
                _surfaceNormalMaterial.SetColor("_BaseColor", surfaceNormalIndicatorColor);
            if (_surfaceNormalMaterial.HasProperty("_Color"))
                _surfaceNormalMaterial.SetColor("_Color", surfaceNormalIndicatorColor);
            if (_surfaceNormalMaterial.HasProperty("_Cull"))
                _surfaceNormalMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            if (_surfaceNormalMaterial.HasProperty("_CullMode"))
                _surfaceNormalMaterial.SetInt("_CullMode", (int)UnityEngine.Rendering.CullMode.Off);
            _surfaceNormalRenderer.sharedMaterial = _surfaceNormalMaterial;
        }

        _surfaceNormalFilter.sharedMesh = _surfaceNormalMesh;
        _surfaceNormalRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _surfaceNormalRenderer.receiveShadows = false;
    }

    private void ApplySurfaceMaterialSettings()
    {
        Material targetMaterial = surfaceMaterialOverride != null ? surfaceMaterialOverride : _surfaceMaterial;
        if (targetMaterial == null) return;
        ApplySurfaceMaterialSettings(targetMaterial, surfaceColor, false);
    }

    private void ApplySurfaceMaterialSettings(Material targetMaterial, Color color, bool transparent)
    {
        if (targetMaterial == null)
            return;
        color.a = transparent ? Mathf.Clamp01(color.a) : 1f;
        if (targetMaterial.HasProperty("_BaseColor")) targetMaterial.SetColor("_BaseColor", color);
        if (targetMaterial.HasProperty("_Color")) targetMaterial.SetColor("_Color", color);
        if (targetMaterial.HasProperty("_Smoothness")) targetMaterial.SetFloat("_Smoothness", 0.35f);
        if (targetMaterial.HasProperty("_Metallic")) targetMaterial.SetFloat("_Metallic", 0f);
        if (targetMaterial.HasProperty("_Surface")) targetMaterial.SetFloat("_Surface", transparent ? 1f : 0f);
        if (targetMaterial.HasProperty("_Blend")) targetMaterial.SetFloat("_Blend", transparent ? 0f : 0f);
        if (targetMaterial.HasProperty("_ZWrite")) targetMaterial.SetFloat("_ZWrite", transparent ? 0f : 1f);
        if (targetMaterial.HasProperty("_Cull")) targetMaterial.SetFloat("_Cull", surfaceDoubleSided ? (float)CullMode.Off : (float)CullMode.Back);
        if (targetMaterial.HasProperty("_SrcBlend")) targetMaterial.SetFloat("_SrcBlend", transparent ? (float)BlendMode.SrcAlpha : (float)BlendMode.One);
        if (targetMaterial.HasProperty("_DstBlend")) targetMaterial.SetFloat("_DstBlend", transparent ? (float)BlendMode.OneMinusSrcAlpha : (float)BlendMode.Zero);
        if (targetMaterial.HasProperty("_Mode")) targetMaterial.SetFloat("_Mode", transparent ? 3f : 0f);
        targetMaterial.renderQueue = transparent ? (int)RenderQueue.Transparent : (int)RenderQueue.Geometry;
    }

    private void ApplyGridLineRendererColor(Color color)
    {
        if (_lineRenderer == null || _propertyBlock == null)
            return;
        _propertyBlock.Clear();
        _propertyBlock.SetColor("_BaseColor", color);
        _propertyBlock.SetColor("_Color", color);
        _lineRenderer.SetPropertyBlock(_propertyBlock);
    }

    private void ApplyRemeshGridLineRendererColor(Color color)
    {
        if (_remeshLineRenderer == null || _propertyBlock == null)
            return;
        _propertyBlock.Clear();
        _propertyBlock.SetColor("_BaseColor", color);
        _propertyBlock.SetColor("_Color", color);
        _remeshLineRenderer.SetPropertyBlock(_propertyBlock);
    }

    private void ApplyGeometricSurfaceGridRendererColor(Color color)
    {
        if (_geometricSurfaceGridRenderer == null || _propertyBlock == null)
            return;
        _propertyBlock.Clear();
        _propertyBlock.SetColor("_BaseColor", color);
        _propertyBlock.SetColor("_Color", color);
        _geometricSurfaceGridRenderer.SetPropertyBlock(_propertyBlock);
    }

    private void SetSurfaceRendererMaterials(int regionCount)
    {
        if (_surfaceRenderer == null)
            return;

        bool useRegionMaterials = colorizeSurfaceRegions;
        if (!useRegionMaterials || regionCount <= 1)
        {
            Material targetMaterial = surfaceMaterialOverride != null ? surfaceMaterialOverride : _surfaceMaterial;
            if (_surfaceRenderer.sharedMaterial != targetMaterial)
                _surfaceRenderer.sharedMaterial = targetMaterial;
            return;
        }

        EnsureSurfaceRegionMaterials(regionCount);
        Material[] materials = new Material[regionCount];
        for (int i = 0; i < regionCount; i++)
            materials[i] = _surfaceRegionMaterials[i];
        _surfaceRenderer.sharedMaterials = materials;
    }

    private void EnsureSurfaceRegionMaterials(int count)
    {
        Material prototype = surfaceMaterialOverride != null ? surfaceMaterialOverride : _surfaceMaterial;
        if (prototype == null)
            return;

        while (_surfaceRegionMaterials.Count < count)
        {
            Material material = new Material(prototype)
            {
                name = $"ScanCover_DepthGridSurface_Region_{_surfaceRegionMaterials.Count}"
            };
            int materialIndex = _surfaceRegionMaterials.Count;
            bool transparent = !showPlaneFamilyClassification && materialIndex == _planeFamilyOutlierSubmeshIndex;
            ApplySurfaceMaterialSettings(material, GetSurfaceRegionColor(materialIndex), transparent);
            _surfaceRegionMaterials.Add(material);
        }

        for (int i = 0; i < _surfaceRegionMaterials.Count; i++)
        {
            if (_surfaceRegionMaterials[i] == null)
                continue;
            bool transparent = !showPlaneFamilyClassification && i == _planeFamilyOutlierSubmeshIndex;
            ApplySurfaceMaterialSettings(_surfaceRegionMaterials[i], GetSurfaceRegionColor(i), transparent);
        }
    }

    private void ClearSurfaceRegionMaterials()
    {
        for (int i = 0; i < _surfaceRegionMaterials.Count; i++)
        {
            if (_surfaceRegionMaterials[i] != null)
                Destroy(_surfaceRegionMaterials[i]);
        }
        _surfaceRegionMaterials.Clear();
    }

    private Color GetSurfaceRegionColor(int regionIndex)
    {
        return snapshotGridUniformColor;
    }

    private void SetGridLinesVisible(bool visible)
    {
        if (_lineRoot != null && _lineRoot.activeSelf != visible) _lineRoot.SetActive(visible);
        if (_lineMesh != null && !visible) _lineMesh.Clear();
        if (!visible)
        {
            SetHeadsetScreenCenterMarkerVisible(false);
            SetOriginalGridCenterMarkerVisible(false);
        }
    }

    private void SetRemeshGridLinesVisible(bool visible)
    {
        if (_remeshLineRoot != null && _remeshLineRoot.activeSelf != visible) _remeshLineRoot.SetActive(visible);
        if (_remeshLineMesh != null && !visible) _remeshLineMesh.Clear();
    }

    private void SetGeometricSurfaceGridVisible(bool visible)
    {
        if (_geometricSurfaceGridRoot != null && _geometricSurfaceGridRoot.activeSelf != visible)
            _geometricSurfaceGridRoot.SetActive(visible);
        if (_geometricSurfaceGridMesh != null && !visible)
            _geometricSurfaceGridMesh.Clear();
    }

    private void SetSurfaceVisible(bool visible) { if (_surfaceRoot != null && _surfaceRoot.activeSelf != visible) _surfaceRoot.SetActive(visible); }

    private void SetCandidatePlaneObjectsVisible(bool visible)
    {
        if (_candidatePlaneRoot != null && _candidatePlaneRoot.activeSelf != visible)
            _candidatePlaneRoot.SetActive(visible);
    }

    private void SetSurfaceNormalIndicatorsVisible(bool visible)
    {
        if (_surfaceNormalRoot != null && _surfaceNormalRoot.activeSelf != visible)
            _surfaceNormalRoot.SetActive(visible);
        if (_surfaceNormalMesh != null && !visible)
            _surfaceNormalMesh.Clear();
    }

    private bool ShouldMaintainSurfaceMesh()
    {
        return !showCandidatePlaneObjects &&
               (showSurfaceMesh ||
                ShouldShowGridInteriorMesh() ||
                keepSurfaceMeshAvailableWhenHidden ||
                showHeightSliceContour ||
                showPlaneFamilyClassification);
    }

    private void ApplyColor(int index, Color color)
    {
        if (_propertyBlock == null || index < 0 || index >= _rendererCache.Count) return;
        Renderer[] renderers = _rendererCache[index]; if (renderers == null) return;
        _propertyBlock.Clear(); _propertyBlock.SetColor("_BaseColor", color); _propertyBlock.SetColor("_Color", color); _propertyBlock.SetColor("_EmissionColor", Color.black);
        for (int i = 0; i < renderers.Length; i++) if (renderers[i] != null) renderers[i].SetPropertyBlock(_propertyBlock);
    }

    public bool TrySetSnapshotGridPointColor(int index, Color color)
    {
        EnsurePropertyBlock();
        if (index < 0 || index >= _rendererCache.Count)
            return false;

        ApplyColor(index, color);
        return true;
    }

    public void RestoreSnapshotGridPointColors()
    {
        if (_snapshotGridExternalControlActive)
            return;

        EnsurePropertyBlock();
        if (_currentValid == null)
            return;

        int count = Mathf.Min(_currentValid.Length, _rendererCache.Count);
        for (int i = 0; i < count; i++)
        {
            if (!_currentValid[i])
                continue;

            ApplyColor(i, snapshotGridUniformColor);
        }
    }

    private void EnsurePropertyBlock() { if (_propertyBlock == null) _propertyBlock = new MaterialPropertyBlock(); }
    private bool SetIssue(string issue) { LastIssue = issue; if (debugLog && !string.IsNullOrEmpty(issue)) Debug.LogWarning($"[ScanCoverDepthGridPointCloud] {issue}"); return false; }

    private void EnsureDisplayRoots()
    {
        if (!useWorldSpaceDisplayRoots)
            return;

        if (_displayRoot == null)
        {
            _displayRoot = new GameObject("[ScanCover] DepthGrid Runtime");
            _displayRoot.transform.SetParent(null, false);
            _displayRoot.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _displayRoot.transform.localScale = Vector3.one;
        }

        if (_runtimeMarkerRoot == null)
        {
            GameObject markerRoot = new GameObject("Markers");
            _runtimeMarkerRoot = markerRoot.transform;
            _runtimeMarkerRoot.SetParent(_displayRoot.transform, false);
            _runtimeMarkerRoot.localPosition = Vector3.zero;
            _runtimeMarkerRoot.localRotation = Quaternion.identity;
            _runtimeMarkerRoot.localScale = Vector3.one;
        }

        EnsureRuntimeAxisDebug();
    }

    private void UpdateDisplayRootsTransform(bool force = false)
    {
        if (!useWorldSpaceDisplayRoots || _displayRoot == null)
            return;
        if (_snapshotGridExternalControlActive)
            return;
        if (!force && !updateEveryFrame)
            return;

        Transform origin = ResolveDisplayOrigin();
        _displayRoot.transform.position = origin != null ? origin.position : Vector3.zero;
        _displayRoot.transform.rotation = ResolveDisplayRotation(origin);
        _displayRoot.transform.localScale = Vector3.one;

        if (_runtimeMarkerRoot != null)
        {
            _runtimeMarkerRoot.localPosition = Vector3.zero;
            _runtimeMarkerRoot.localRotation = Quaternion.identity;
            _runtimeMarkerRoot.localScale = Vector3.one;
        }

        if (_centerDebugRoot != null)
        {
            _centerDebugRoot.transform.localPosition = Vector3.zero;
            _centerDebugRoot.transform.localRotation = Quaternion.identity;
            _centerDebugRoot.transform.localScale = Vector3.one;
        }

        if (_surfaceRoot != null)
        {
            _surfaceRoot.transform.localPosition = Vector3.zero;
            _surfaceRoot.transform.localRotation = Quaternion.identity;
            _surfaceRoot.transform.localScale = Vector3.one;
        }

        if (_surfaceNormalRoot != null)
        {
            _surfaceNormalRoot.transform.localPosition = Vector3.zero;
            _surfaceNormalRoot.transform.localRotation = Quaternion.identity;
            _surfaceNormalRoot.transform.localScale = Vector3.one;
        }

        if (_runtimeAxisRoot != null)
        {
            _runtimeAxisRoot.transform.localPosition = runtimeAxisOffset;
            _runtimeAxisRoot.transform.localRotation = Quaternion.identity;
            _runtimeAxisRoot.transform.localScale = Vector3.one;
        }
    }

    private Transform ResolveMarkerParent()
    {
        if (useWorldSpaceDisplayRoots)
        {
            EnsureDisplayRoots();
            return _runtimeMarkerRoot != null ? _runtimeMarkerRoot : transform;
        }

        return markerParent ? markerParent : transform;
    }

    private Transform ResolveSurfaceParent()
    {
        if (useWorldSpaceDisplayRoots)
        {
            EnsureDisplayRoots();
            return _displayRoot != null ? _displayRoot.transform : null;
        }

        return transform;
    }

    private Transform ResolveDisplayOrigin() => ResolveViewLockedOrigin();

    private bool ShouldRectifyRegularGridBuffers()
    {
        return samplingMode == SamplingMode.RegularGrid &&
               useWorldSpaceDisplayRoots &&
               lockUnfrozenDisplayRoll &&
               compensateRegularGridRollSampling;
    }

    private Quaternion ResolveDisplayRotation(Transform origin)
    {
        if (origin == null) return transform.rotation;
        if (!lockUnfrozenDisplayRoll && !lockUnfrozenDisplayPitch && !lockUnfrozenDisplayYaw) return origin.rotation;
        if (lockUnfrozenDisplayYaw && lockUnfrozenDisplayPitch && lockUnfrozenDisplayRoll) return Quaternion.identity;

        Vector3 forward = origin.forward;
        if (forward.sqrMagnitude <= 1e-6f)
            return origin.rotation;

        forward.Normalize();
        if (lockUnfrozenDisplayYaw)
        {
            float pitchDegrees = -Mathf.Asin(Mathf.Clamp(forward.y, -1f, 1f)) * Mathf.Rad2Deg;
            forward = Quaternion.AngleAxis(pitchDegrees, Vector3.right) * Vector3.forward;
        }
        else if (lockUnfrozenDisplayPitch)
        {
            Vector3 planarForward = Vector3.ProjectOnPlane(forward, Vector3.up);
            if (planarForward.sqrMagnitude > 1e-6f)
                forward = planarForward.normalized;
        }

        Vector3 up = ResolveWorldUp(forward, origin.up);
        return up.sqrMagnitude > 1e-6f ? Quaternion.LookRotation(forward, up) : origin.rotation;
    }

    private static float ResolveDisplayRollOffset(Transform origin, Quaternion displayRotation)
    {
        Quaternion delta = Quaternion.Inverse(displayRotation) * origin.rotation;
        Vector3 deltaEuler = delta.eulerAngles;
        float signedRoll = deltaEuler.z > 180f ? deltaEuler.z - 360f : deltaEuler.z;
        return -signedRoll;
    }

    private static void ResolveRegularGridSampleCoord(int baseX, int baseY, int width, int height, float rollCompensationDegrees, out int sampleX, out int sampleY)
    {
        if (Mathf.Abs(rollCompensationDegrees) <= 1e-3f)
        {
            sampleX = Mathf.Clamp(baseX, 0, width - 1);
            sampleY = Mathf.Clamp(baseY, 0, height - 1);
            return;
        }

        float radians = rollCompensationDegrees * Mathf.Deg2Rad;
        float sin = Mathf.Sin(radians);
        float cos = Mathf.Cos(radians);
        Vector2 center = new Vector2((width - 1) * 0.5f, (height - 1) * 0.5f);
        Vector2 offset = new Vector2(baseX, baseY) - center;
        Vector2 rotated = new Vector2(offset.x * cos - offset.y * sin, offset.x * sin + offset.y * cos) + center;
        sampleX = Mathf.Clamp(Mathf.RoundToInt(rotated.x), 0, width - 1);
        sampleY = Mathf.Clamp(Mathf.RoundToInt(rotated.y), 0, height - 1);
    }

    private void RectifyRegularGridBuffers(
        NativeArray<Color> sourceWorldPositionsRaw,
        NativeArray<Color> sourceWorldPositions,
        NativeArray<Color> sourceWorldNormals,
        NativeArray<Color> sourceWorldNormalsNeighbour,
        NativeArray<Color> sourceObservationMeta,
        Vector2Int resolution,
        NativeArray<Color> rectifiedWorldPositionsRaw,
        NativeArray<Color> rectifiedWorldPositions,
        NativeArray<Color> rectifiedWorldNormals,
        NativeArray<Color> rectifiedWorldNormalsNeighbour,
        NativeArray<Color> rectifiedObservationMeta)
    {
        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int pixelCount = width * height;
        if (sourceWorldPositionsRaw.Length != pixelCount ||
            sourceWorldPositions.Length != pixelCount ||
            sourceWorldNormals.Length != pixelCount ||
            sourceWorldNormalsNeighbour.Length != pixelCount ||
            sourceObservationMeta.Length != pixelCount ||
            rectifiedWorldPositionsRaw.Length != pixelCount ||
            rectifiedWorldPositions.Length != pixelCount ||
            rectifiedWorldNormals.Length != pixelCount ||
            rectifiedWorldNormalsNeighbour.Length != pixelCount ||
            rectifiedObservationMeta.Length != pixelCount)
        {
            for (int i = 0; i < Mathf.Min(pixelCount, rectifiedWorldPositions.Length); i++)
            {
                rectifiedWorldPositionsRaw[i] = i < sourceWorldPositionsRaw.Length ? sourceWorldPositionsRaw[i] : default;
                rectifiedWorldPositions[i] = i < sourceWorldPositions.Length ? sourceWorldPositions[i] : default;
                rectifiedWorldNormals[i] = i < sourceWorldNormals.Length ? sourceWorldNormals[i] : default;
                rectifiedWorldNormalsNeighbour[i] = i < sourceWorldNormalsNeighbour.Length ? sourceWorldNormalsNeighbour[i] : default;
                rectifiedObservationMeta[i] = i < sourceObservationMeta.Length ? sourceObservationMeta[i] : default;
            }
            return;
        }

        Transform origin = ResolveDisplayOrigin();
        Quaternion displayRotation = ResolveDisplayRotation(origin);
        float rollCompensationDegrees = origin != null ? ResolveDisplayRollOffset(origin, displayRotation) : 0f;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                ResolveRegularGridSampleCoord(x, y, width, height, rollCompensationDegrees, out int sampleX, out int sampleY);
                int dstIndex = x + y * width;
                int srcIndex = sampleX + sampleY * width;
                rectifiedWorldPositionsRaw[dstIndex] = sourceWorldPositionsRaw[srcIndex];
                rectifiedWorldPositions[dstIndex] = sourceWorldPositions[srcIndex];
                rectifiedWorldNormals[dstIndex] = sourceWorldNormals[srcIndex];
                rectifiedWorldNormalsNeighbour[dstIndex] = sourceWorldNormalsNeighbour[srcIndex];
                rectifiedObservationMeta[dstIndex] = sourceObservationMeta[srcIndex];
            }
        }
    }

    private Transform ResolveDisplayLocalTransform()
    {
        if (!useWorldSpaceDisplayRoots)
            return null;

        return _displayRoot != null ? _displayRoot.transform : ResolveDisplayOrigin();
    }

    private void EnsureRuntimeAxisDebug()
    {
        if (!useWorldSpaceDisplayRoots || _displayRoot == null)
            return;

        if (!showRuntimeAxisDebug)
        {
            if (_runtimeAxisRoot != null)
                _runtimeAxisRoot.SetActive(false);
            return;
        }

        if (_runtimeAxisRoot == null)
        {
            _runtimeAxisRoot = new GameObject("[Debug] Runtime Axes");
            _runtimeAxisRoot.transform.SetParent(_displayRoot.transform, false);
            _runtimeAxisRoot.transform.localPosition = runtimeAxisOffset;
            _runtimeAxisRoot.transform.localRotation = Quaternion.identity;
            _runtimeAxisRoot.transform.localScale = Vector3.one;

            CreateRuntimeAxisCube("CenterCube", Color.white, Vector3.zero, Vector3.one * runtimeAxisCubeSize);
            CreateRuntimeAxisCube("PitchAxis_X", new Color(1f, 0.25f, 0.25f, 1f), new Vector3(runtimeAxisLength * 0.5f, 0f, 0f), new Vector3(runtimeAxisLength, runtimeAxisThickness, runtimeAxisThickness));
            CreateRuntimeAxisCube("YawAxis_Y", new Color(0.25f, 1f, 0.35f, 1f), new Vector3(0f, runtimeAxisLength * 0.5f, 0f), new Vector3(runtimeAxisThickness, runtimeAxisLength, runtimeAxisThickness));
            CreateRuntimeAxisCube("RollAxis_Z", new Color(0.3f, 0.6f, 1f, 1f), new Vector3(0f, 0f, runtimeAxisLength * 0.5f), new Vector3(runtimeAxisThickness, runtimeAxisThickness, runtimeAxisLength));
            CreateRuntimeAxisCube("PitchTip_X", new Color(1f, 0.45f, 0.45f, 1f), new Vector3(runtimeAxisLength, 0f, 0f), Vector3.one * (runtimeAxisCubeSize * 0.6f));
            CreateRuntimeAxisCube("YawTip_Y", new Color(0.45f, 1f, 0.5f, 1f), new Vector3(0f, runtimeAxisLength, 0f), Vector3.one * (runtimeAxisCubeSize * 0.6f));
            CreateRuntimeAxisCube("RollTip_Z", new Color(0.45f, 0.7f, 1f, 1f), new Vector3(0f, 0f, runtimeAxisLength), Vector3.one * (runtimeAxisCubeSize * 0.6f));
        }
        else if (!_runtimeAxisRoot.activeSelf)
        {
            _runtimeAxisRoot.SetActive(true);
        }
    }

    private void CreateRuntimeAxisCube(string name, Color color, Vector3 localPosition, Vector3 localScale)
    {
        if (_runtimeAxisRoot == null)
            return;

        GameObject axisObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
        axisObject.name = name;
        axisObject.transform.SetParent(_runtimeAxisRoot.transform, false);
        axisObject.transform.localPosition = localPosition;
        axisObject.transform.localRotation = Quaternion.identity;
        axisObject.transform.localScale = localScale;

        Collider axisCollider = axisObject.GetComponent<Collider>();
        if (axisCollider != null)
            Destroy(axisCollider);

        Renderer renderer = axisObject.GetComponent<Renderer>();
        if (renderer != null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            Material axisMaterial = new Material(shader) { name = $"ScanCover_RuntimeAxis_{name}" };
            if (axisMaterial.HasProperty("_BaseColor")) axisMaterial.SetColor("_BaseColor", color);
            if (axisMaterial.HasProperty("_Color")) axisMaterial.SetColor("_Color", color);
            if (axisMaterial.HasProperty("_Smoothness")) axisMaterial.SetFloat("_Smoothness", 0.15f);
            if (axisMaterial.HasProperty("_Metallic")) axisMaterial.SetFloat("_Metallic", 0f);
            renderer.sharedMaterial = axisMaterial;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            _runtimeAxisMaterials.Add(axisMaterial);
        }
    }

    private static Vector3 ResolveWorldUp(Vector3 forward, Vector3 fallbackUp)
    {
        Vector3 up = Vector3.ProjectOnPlane(Vector3.up, forward);
        if (up.sqrMagnitude > 1e-6f) return up.normalized;

        up = Vector3.ProjectOnPlane(fallbackUp, forward);
        if (up.sqrMagnitude > 1e-6f) return up.normalized;

        up = Vector3.ProjectOnPlane(Vector3.forward, forward);
        if (up.sqrMagnitude > 1e-6f) return up.normalized;

        up = Vector3.ProjectOnPlane(Vector3.right, forward);
        return up.sqrMagnitude > 1e-6f ? up.normalized : Vector3.up;
    }

    private static Vector3 ResolveLocalUp(Vector3 forward, Vector3 fallbackUp)
    {
        Vector3 up = Vector3.ProjectOnPlane(fallbackUp, forward);
        if (up.sqrMagnitude > 1e-6f) return up.normalized;

        up = Vector3.ProjectOnPlane(Vector3.forward, forward);
        if (up.sqrMagnitude > 1e-6f) return up.normalized;

        up = Vector3.ProjectOnPlane(Vector3.right, forward);
        return up.sqrMagnitude > 1e-6f ? up.normalized : Vector3.up;
    }

    private static float Remap(float value, float min, float max) => Mathf.Abs(max - min) <= 1e-6f ? 0.5f : Mathf.InverseLerp(min, max, value);
    private static Vector3 WorldPos(Color color) => new Vector3(color.r, color.g, color.b);
    private static Vector3 WorldNormal(Color color, bool valid) { if (!valid) return Vector3.up; Vector3 normal = new Vector3(color.r, color.g, color.b); return normal.sqrMagnitude <= 1e-6f ? Vector3.up : normal.normalized; }
    private static float Confidence(Color color) => color.g;
    private static bool Finite(Vector3 value) => !float.IsNaN(value.x) && !float.IsNaN(value.y) && !float.IsNaN(value.z) && !float.IsInfinity(value.x) && !float.IsInfinity(value.y) && !float.IsInfinity(value.z);
    private static bool IsLitShader(Shader shader) => shader != null && (shader.name.Contains("/Lit") || shader.name == "Standard");
}




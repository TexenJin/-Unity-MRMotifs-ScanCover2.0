using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

[DisallowMultipleComponent]
public sealed class ScanCoverTsdfSingleShellPrototype : MonoBehaviour
{
    private static readonly Vector3Int[] FaceDirections =
    {
        new Vector3Int(-1, 0, 0), new Vector3Int(1, 0, 0),
        new Vector3Int(0, -1, 0), new Vector3Int(0, 1, 0),
        new Vector3Int(0, 0, -1), new Vector3Int(0, 0, 1)
    };
    [Header("Refs")]
    [SerializeField] private ScanCoverDepthGridPointCloud rawDepthSource;
    [SerializeField] private Transform volumeAnchor;
    [SerializeField] private Transform displayRoot;
    [SerializeField] private bool forceWorldSpaceDisplay = true;

    [Header("Raw Snapshot Refresh")]
    [SerializeField] private bool hideSourcePreview = true;
    [SerializeField] private bool forceRawRefreshOnSnapshot = true;
    [SerializeField, Min(0.05f)] private float waitForRawFrameTimeoutSeconds = 0.85f;
    [SerializeField, Min(0.05f)] private float fusionBurstDurationSeconds = 2.0f;
    [SerializeField, Min(1)] private int maxFusionFramesPerTrigger = 24;
    [SerializeField] private bool rebuildMeshAfterFusionBurstOnly = false;

    [Header("Distilled TSDF Shell Parameters")]
    [SerializeField, Min(0.02f)] private float voxelSizeMeters = 0.065f;
    [SerializeField, Min(0.02f)] private float truncationMeters = 0.195f;
    [SerializeField] private Vector3 volumeSizeMeters = new Vector3(12f, 4f, 12f);
    [SerializeField, Range(1, 64)] private int maxFusionWeight = 12;
    [SerializeField, Range(1, 8)] private int minSurfaceCornerWeight = 2;
    [SerializeField] private bool useProjectiveTsdfIntegration = true;

    [Header("Depth Filtering")]
    [SerializeField, Min(1)] private int sampleStridePixels = 1;
    [SerializeField, Min(0f)] private float minDepthMeters = 0.35f;
    [SerializeField, Min(0f)] private float maxDepthMeters = 5.5f;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0f;
    [SerializeField] private bool requireValidNormal = false;
    [SerializeField, Range(-1f, 1f)] private float minNormalFacingCameraDot = -0.25f;
    [SerializeField] private bool fillProjectiveDepthFromNeighbors = false;
    [SerializeField, Range(1, 4)] private int projectiveNeighborFillRadiusPixels = 2;
    [SerializeField] private bool rejectDepthDiscontinuities = true;
    [SerializeField, Range(1, 3)] private int depthDiscontinuityNeighborRadiusPixels = 1;
    [SerializeField, Min(0.01f)] private float maxDepthDiscontinuityMeters = 0.07f;
    [SerializeField, Range(0f, 0.2f)] private float maxDepthDiscontinuityRatio = 0.03f;
    [SerializeField, Range(1, 8)] private int minDepthConsistentNeighbors = 5;
    [SerializeField] private bool allowSparseDepthNeighborhood = false;
    [SerializeField] private bool useRobustDepthPrefilter = true;
    [SerializeField, Range(1, 3)] private int robustDepthFilterRadiusPixels = 1;
    [SerializeField, Range(2, 8)] private int minRobustDepthNeighbors = 6;
    [SerializeField, Range(0f, 1f)] private float minRobustDepthConsistencyRatio = 0.62f;
    [SerializeField, Min(0.01f)] private float maxRobustMedianDepthDeviationMeters = 0.055f;
    [SerializeField] private bool correctDepthToNeighborhoodMedian = true;
    [SerializeField, Range(0.05f, 1f)] private float minRobustDepthWeightScale = 0.28f;
    [SerializeField, Range(0.01f, 1f)] private float minRobustTsdfSampleWeight = 0.28f;
    [SerializeField] private bool erodeDepthEdgesBeforeTsdf = true;
    [SerializeField, Range(1, 3)] private int depthEdgeErosionRadiusPixels = 1;
    [SerializeField, Range(1, 16)] private int maxRobustDepthEdgeNeighbors = 2;
    [SerializeField, Min(0.005f)] private float robustDepthEdgeDeltaMeters = 0.055f;
    [SerializeField, Range(0f, 0.2f)] private float robustDepthEdgeDeltaRatio = 0.025f;
    [SerializeField, Range(0.05f, 1f)] private float depthEdgeWeightScale = 0.35f;
    [SerializeField] private bool rejectStrongDepthEdges = true;
    [SerializeField] private bool gateTsdfWritesByDepthSupport = true;
    [SerializeField, Range(1, 24)] private int minTsdfDepthCheckedNeighbors = 4;
    [SerializeField, Range(0f, 1f)] private float minTsdfDepthConsistencyRatio = 0.55f;
    [SerializeField] private bool rejectConflictingTsdfWrites = true;
    [SerializeField, Range(0.05f, 1f)] private float tsdfConflictThreshold = 0.45f;
    [SerializeField, Range(1, 32)] private int minConflictVoxelWeight = 3;
    [SerializeField] private bool correctStableConflictingTsdf = true;
    [SerializeField, Range(2, 12)] private int minConflictCorrectionFrames = 4;
    [SerializeField, Range(0.05f, 1f)] private float conflictCorrectionAgreementThreshold = 0.30f;
    [SerializeField, Range(1, 64)] private int maxCorrectableTsdfWeight = 12;
    [SerializeField, Range(0.1f, 1f)] private float conflictCorrectionBlend = 0.72f;
    [SerializeField, Range(1, 16)] private int correctedTsdfWeight = 3;
    [SerializeField] private bool replaceDirtyTsdfOnStableConflict = true;
    [SerializeField, Range(1, 8)] private int minDirtyTsdfReplaceFrames = 2;
    [SerializeField, Range(1, 64)] private int maxDirtyTsdfReplaceWeight = 12;
    [SerializeField, Range(1, 16)] private int dirtyTsdfReplaceWeight = 2;
    [SerializeField] private bool enableGuardedDirtyTsdfFastReplace = true;
    [SerializeField, Range(1, 4)] private int minGuardedDirtyTsdfReplaceFrames = 1;
    [SerializeField, Range(1, 24)] private int maxGuardedDirtyTsdfReplaceWeight = 12;
    [SerializeField, Range(0.05f, 1f)] private float maxGuardedDirtyTsdfReplaceAbsValue = 0.75f;
    [SerializeField] private bool repairDirtyTsdfConflictBands = true;
    [SerializeField, Range(0.05f, 1f)] private float minDirtyBandConflictRatio = 0.18f;
    [SerializeField, Range(1, 64)] private int maxDirtyBandRepairWeight = 12;
    [SerializeField, Range(0.05f, 1f)] private float minDirtyBandRepairResidual = 0.45f;
    [SerializeField, Range(0f, 1f)] private float dirtyBandRepairWeightKeepRatio = 0.35f;
    [SerializeField, Range(1, 32)] private int maxDirtyBandRepairsPerSample = 8;
    [SerializeField] private bool cleanupDirtyTsdfNeighborhood = true;
    [SerializeField, Range(0, 2)] private int dirtyTsdfCleanupRadiusVoxels = 1;
    [SerializeField, Range(1, 64)] private int maxDirtyTsdfCleanupWeight = 4;
    [SerializeField] private bool cleanupOnlyPendingDirtyTsdfNeighbors = true;
    [SerializeField, Range(0, 12)] private int dirtyTsdfQuarantineFrames = 4;
    [SerializeField] private bool requireMultiFrameStableTsdf = true;
    [SerializeField, Range(2, 8)] private int minStableTsdfFrames = 3;
    [SerializeField, Range(0.05f, 1f)] private float stableTsdfAgreementThreshold = 0.35f;
    [SerializeField, Range(1, 32)] private int stableTsdfBypassWeight = 3;
    [SerializeField] private bool enableProjectiveFreeSpaceCarving = true;
    [SerializeField, Range(0.1f, 1f)] private float freeSpaceCarvingWeightScale = 0.35f;
    [SerializeField, Range(0.05f, 1f)] private float freeSpaceCarvingBlend = 0.45f;
    [SerializeField, Range(1, 32)] private int maxFreeSpaceCarvableWeight = 10;
    [SerializeField, Range(1, 8)] private int freeSpaceCarvingPixelStride = 3;
    [SerializeField] private bool useTemporalFreeSpaceEvidence = true;
    [SerializeField, Range(2, 8)] private int minFreeSpaceEvidenceFrames = 3;
    [SerializeField, Range(2, 32)] private int maxFreeSpaceEvidenceFrameGap = 10;
    [SerializeField, Range(1, 4)] private int freeSpaceEvidenceWeightDecay = 1;
    [SerializeField, Range(0, 4)] private int freeSpaceEvidenceClearWeight = 1;
    [SerializeField, Range(0.5f, 1f)] private float freeSpaceEvidenceClearTsdf = 0.85f;
    [SerializeField] private bool protectSameFrameSurfaceFromClearing = true;

    [Header("Surface Extraction")]
    [SerializeField] private bool rebuildMeshAfterEachCapture = true;
    [SerializeField] private bool showExtractedSurfaceMesh = true;
    [SerializeField] private bool useLegacyTsdfMeshDisplay = true;
    [Header("Stage 03A Clean TSDF Iso-Surface")]
    [SerializeField] private bool useStage03ACleanIsoSurface = true;
    [SerializeField] private bool useV09LegacyExtractorForStage03A = true;
    [SerializeField] private bool useV09ExactCornerEligibilityForDiagnosis = true;
    [SerializeField] private bool useV08DirectBandWriteForDiagnosis = true;
    [SerializeField] private bool requireCleanMeshVoxelProvenance = true;
    [SerializeField] private bool allowVerifiedContinuityInCleanMesh = true;
    [Header("Hard TSDF Write Audit")]
    [SerializeField] private bool enableHardTsdfWriteAudit = true;
    [SerializeField] private bool writeHardTsdfAuditOnCapture = true;
    [SerializeField, Range(1, 128)] private int maxHardAuditVoxelSamplesPerStage = 32;
    [Header("TSDF Surface Profile Diagnostics")]
    [SerializeField] private bool writeHoleAndLayerDiagnosticsOnCapture = true;
    [SerializeField, Range(12, 96)] private int layerDiagnosticRaysX = 48;
    [SerializeField, Range(9, 72)] private int layerDiagnosticRaysY = 36;
    [SerializeField, Range(0.25f, 1f)] private float layerDiagnosticStepVoxels = 0.5f;
    [SerializeField, Range(256, 12000)] private int maxHoleDiagnosticRows = 4000;
    [Header("Guarded Hole Side Repair")]
    [SerializeField] private bool enableGuardedHoleSideRepair = true;
    [SerializeField, Range(1, 2048)] private int maxHoleSideRepairsPerRebuild = 512;
    [SerializeField, Range(1, 6)] private int minHoleSideRepairOppositeFaceSupport = 2;
    [SerializeField, Range(1, 6)] private int minHoleSideRepairExistingFaceSupport = 2;
    [SerializeField, Range(1, 8)] private int minHoleSideRepairBridgingCells = 1;
    [SerializeField, Range(0.01f, 0.3f)] private float holeSideRepairAbsTsdf = 0.06f;
    [SerializeField, Range(0.05f, 0.5f)] private float maxHoleSideRepairNeighborAbsTsdf = 0.25f;
    [SerializeField] private bool blockHoleRepairThatCreatesMultipleZeroCrossings = true;
    [SerializeField, Range(2, 8)] private int holeRepairZeroCrossingCheckRadiusVoxels = 4;
    [SerializeField] private bool usePrimaryPlaneHoleRepair = true;
    [SerializeField, Range(1, 4)] private int primaryPlaneHoleAnchorRadiusVoxels = 2;
    [SerializeField, Range(3, 16)] private int minPrimaryPlaneHoleAnchors = 4;
    [SerializeField, Range(0.75f, 1f)] private float minPrimaryPlaneHoleNormalDot = 0.92f;
    [SerializeField, Range(0.01f, 0.2f)] private float maxPrimaryPlaneHoleResidualMeters = 0.06f;
    [SerializeField, Range(0, 2)] private int primaryPlaneHoleBandRadiusVoxels = 1;
    [SerializeField] private bool confirmPrimaryPlaneHoleBandsFromNearbyAccept = true;
    [SerializeField, Range(1, 3)] private int primaryPlaneHoleAcceptRadiusVoxels = 1;
    [SerializeField, Range(1, 4)] private int primaryPlaneHoleAcceptTangentialRadiusVoxels = 3;
    [SerializeField, Range(0.5f, 1.5f)] private float primaryPlaneHoleAcceptNormalHalfThicknessVoxels = 0.75f;
    [SerializeField, Range(0.01f, 0.12f)] private float primaryPlaneHoleAcceptMaxPlaneDistanceMeters = 0.05f;
    [SerializeField, Range(0.01f, 0.3f)] private float primaryPlaneHoleAcceptMaxTsdfDelta = 0.18f;
    [SerializeField, Range(1, 8)] private int primaryPlaneHoleMaxAgeFrames = 3;
    [Header("Layer Pair Classification")]
    [SerializeField, Range(0.03f, 0.25f)] private float duplicateLayerMaxGapMeters = 0.12f;
    [SerializeField, Range(0.5f, 1f)] private float duplicateLayerMinNormalDot = 0.9f;
    [SerializeField, Range(0.15f, 1f)] private float legalOcclusionMinGapMeters = 0.30f;
    [SerializeField] private bool gateAtomicAcceptDuplicateLayers = true;
    [SerializeField, Range(1, 5)] private int atomicAcceptDuplicateSearchRadiusVoxels = 2;
    [SerializeField, Range(0.01f, 0.08f)] private float atomicAcceptDuplicateMinGapMeters = 0.03f;
    [SerializeField, Range(1, 2048)] private int maxDuplicateLayerCleanupVoxelsPerRebuild = 512;
    [SerializeField, Range(1, 4)] private int duplicateLayerCleanupWeightStep = 1;
    [SerializeField, Range(2, 8)] private int duplicateLayerRepeatBoostThreshold = 2;
    [SerializeField, Range(2, 16)] private int duplicateLayerMaxBoostedWeightStep = 8;
    [Header("Atomic Observation TSDF Bands")]
    [SerializeField] private bool useAtomicObservationTsdfBands = true;
    [SerializeField, Range(0.05f, 0.5f)] private float atomicProvisionalBandWeightScale = 0.2f;
    [SerializeField, Range(1, 4)] private int atomicProvisionalBandMaxWeight = 2;
    [SerializeField, Range(0.5f, 1.25f)] private float projectiveVoxelCenterRadiusScale = 0.9f;
    [Header("Accepted Sign Recovery")]
    [SerializeField] private bool enableAcceptedSignRecovery = true;
    [SerializeField, Range(2, 6)] private int acceptedSignRecoveryMinFrames = 3;
    [SerializeField, Range(1, 6)] private int acceptedSignRecoveryMinNeighborSupport = 2;
    [SerializeField, Range(0.05f, 0.6f)] private float acceptedSignRecoveryMaxAbsTsdf = 0.35f;
    [SerializeField, Range(1, 16)] private int acceptedSignRecoveryOldWeightCap = 4;
    [Header("Hole Boundary Diagnosis")]
    [SerializeField] private bool showHoleBoundaryDiagnosis = true;
    [SerializeField] private bool hideMeshWhileShowingHoleBoundaryDiagnosis = true;
    [SerializeField, Range(128, 12000)] private int maxHoleBoundaryMarkers = 4000;
    [SerializeField, Range(0.005f, 0.08f)] private float holeBoundaryMarkerSizeMeters = 0.025f;
    [SerializeField, Range(8, 600)] private int holeBoundaryRetiredMemoryFrames = 180;
    [SerializeField] private Color holeBoundaryWaitingColor = new Color(0.05f, 0.45f, 1f, 1f);
    [SerializeField] private Color holeBoundaryRetiredColor = new Color(0.9f, 0.05f, 1f, 1f);
    [SerializeField] private Color holeBoundaryNoBandColor = new Color(1f, 0.85f, 0.02f, 1f);
    [SerializeField] private Color holeBoundaryPositiveOnlyColor = new Color(1f, 0.38f, 0.02f, 1f);
    [SerializeField] private Color holeBoundaryNegativeOnlyColor = new Color(1f, 0.04f, 0.02f, 1f);
    [SerializeField] private Color holeBoundarySpatialMissColor = new Color(1f, 0.05f, 0.58f, 1f);
    [SerializeField] private Color holeBoundaryWeakCornersColor = new Color(0.05f, 1f, 0.95f, 1f);
    [SerializeField] private Color holeCauseNoRawColor = new Color(0.45f, 0.45f, 0.45f, 1f);
    [SerializeField] private Color holeCausePendingColor = new Color(0.05f, 0.45f, 1f, 1f);
    [SerializeField] private Color holeCauseNoBandColor = new Color(1f, 0.85f, 0.02f, 1f);
    [SerializeField] private Color holeCauseNoZeroCrossColor = new Color(1f, 0.32f, 0.02f, 1f);
    [SerializeField] private Color holeCauseVertexRejectColor = new Color(0.05f, 1f, 0.95f, 1f);
    [SerializeField] private Color holeCauseQuadRejectColor = new Color(0.95f, 0.08f, 0.75f, 1f);
    [SerializeField] private Color holeCauseClearedColor = new Color(1f, 0.03f, 0.02f, 1f);
    [SerializeField] private Color holeCauseFrameMismatchColor = Color.white;
    [SerializeField] private Color holeCauseCrossFrameColor = new Color(0.62f, 0.08f, 1f, 1f);
    [SerializeField] private Color primaryLayerDiagnosticColor = new Color(0.05f, 0.8f, 1f, 1f);
    [SerializeField] private Color secondaryLayerDiagnosticColor = new Color(1f, 0.05f, 0.72f, 1f);
    [Header("Layer Diagnostic Display")]
    [SerializeField] private bool showMeshBehindLayerDiagnostics = true;
    [SerializeField] private bool showMeshAsWireOnlyBehindLayerDiagnostics = true;
    [SerializeField] private bool showGlobalPairedLayerDiagnostics = true;
    [SerializeField] private bool showAllLayerDiagnosticMarkers = true;
    [SerializeField, Range(12000, 100000)] private int maxFullLayerDiagnosticMarkers = 72000;
    [SerializeField, Range(0.02f, 0.5f)] private float layerDiagnosticMaxAbsTsdf = 0.22f;
    [SerializeField, Range(0.25f, 1f)] private float primaryLayerDiagnosticSizeScale = 0.6f;
    [SerializeField, Min(0)] private int maxSurfaceVertices = 180000;
    [SerializeField] private bool relaxSurfaceExtractionNearZero = true;
    [SerializeField, Range(0.02f, 0.45f)] private float nearZeroSurfaceThreshold = 0.14f;
    [SerializeField, Min(2)] private int minStrictSurfaceSupportedCorners = 3;
    [SerializeField, Min(2)] private int minRelaxedSurfaceSupportedCorners = 6;
    [SerializeField, Min(1)] private int minRelaxedNearZeroCorners = 4;
    [SerializeField, Min(1)] private int minStrictSurfaceEdgeHits = 2;
    [SerializeField, Min(1)] private int minRelaxedSurfaceEdgeHits = 7;
    [SerializeField, Range(0.02f, 0.45f)] private float relaxedEdgeThreshold = 0.04f;
    [SerializeField] private bool requireRelaxedSurfaceNearStrictCell = false;
    [SerializeField, Range(1, 3)] private int relaxedStrictNeighborRadiusCells = 1;
    [SerializeField] private bool gateRelaxedQuadsByStrictSupport = false;
    [SerializeField, Range(1, 3)] private int relaxedQuadStrictNeighborRadiusCells = 1;
    [SerializeField, Range(1, 12)] private int minRelaxedQuadStrictNeighborCells = 3;
    [SerializeField] private bool rejectRelaxedOnlySignChangeQuads = false;
    [SerializeField] private bool suppressExtractionNearPendingTsdfCorrection = true;
    [SerializeField, Range(0, 2)] private int pendingCorrectionExtractionRadiusCells = 1;
    [SerializeField] private bool suppressExtractionNearDirtyTsdfQuarantine = true;
    [SerializeField, Range(0, 2)] private int dirtyTsdfExtractionRadiusCells = 1;
    [SerializeField] private bool suppressQuadOnPendingTsdfCorrectionEdge = true;
    [SerializeField] private bool suppressQuadOnDirtyTsdfQuarantineEdge = true;
    [SerializeField, Range(1, 12)] private int minPendingCorrectionHitsToSuppressQuad = 2;
    [SerializeField] private bool protectDisplayedMeshFromBadRebuild = false;
    [SerializeField, Range(0.1f, 1f)] private float minAcceptedRebuildTriangleRatio = 0.55f;
    [SerializeField, Range(0f, 1f)] private float maxAcceptedPrunedTriangleRatio = 0.45f;
    [Header("Committed Mesh Stability")]
    [SerializeField] private bool guardCommittedMeshSpatialCoverage = true;
    [SerializeField, Range(0.5f, 1f)] private float minCommittedMeshBlockRetention = 0.90f;
    [SerializeField, Range(1, 8)] private int committedMeshBlockSizeVoxels = 2;
    [SerializeField] private bool useStrictCommittedMeshRetention = true;
    [SerializeField, Range(0.8f, 1f)] private float minCommittedMeshTriangleRetention = 0.90f;
    [SerializeField] private bool absorbNewBlocksFromHeldCandidate = false;
    [SerializeField, Range(256, 20000)] private int maxHeldCandidateTrianglesToAbsorb = 6000;
    [SerializeField] private bool requireMainShellPromotionGate = true;
    [SerializeField, Range(1, 24)] private int minMainShellFusedFrames = 8;
    [SerializeField, Range(1, 3)] private int mainShellNeighborRadiusCells = 1;
    [SerializeField, Range(1, 64)] private int minMainShellNeighborSurfaceCells = 9;
    [SerializeField, Range(1, 12)] private int minMainShellAverageCornerWeight = 3;
    [SerializeField] private bool separateComplexDetailCandidates = true;
    [SerializeField] private bool promoteComplexDetailCandidatesToMesh = false;
    [SerializeField, Range(1, 64)] private int minDetailPromotionNeighborSurfaceCells = 14;
    [SerializeField, Range(2, 24)] private int minDetailCandidateFrames = 6;
    [SerializeField, Range(1, 64)] private int minDetailCandidateNeighborSurfaceCells = 9;
    [SerializeField] private bool showFallbackWhenMainShellHeld = true;
    [SerializeField] private bool rejectStretchedExtractedQuads = true;
    [SerializeField, Range(1.2f, 4.0f)] private float maxExtractedQuadEdgeVoxelScale = 1.7f;
    [SerializeField] private bool pruneSmallExtractedMeshComponents = true;
    [SerializeField, Min(0)] private int minExtractedComponentTriangles = 96;
    [SerializeField, Range(2f, 16f)] private float minExtractedComponentExtentVoxelScale = 7.0f;
    [SerializeField, Range(1, 8)] private int suspectComponentTriangleMultiplier = 3;
    [SerializeField] private bool pruneDanglingExtractedMeshTriangles = true;
    [SerializeField, Range(1, 6)] private int danglingTrianglePruneIterations = 2;
    [SerializeField, Range(0, 2)] private int minTriangleSharedEdgesToKeep = 1;
    [SerializeField] private bool pruneSpikePatchTriangles = true;
    [SerializeField, Range(1, 8)] private int spikePatchPruneIterations = 3;
    [SerializeField, Range(2, 48)] private int maxSpikePatchTriangles = 18;
    [SerializeField, Range(0.05f, 0.95f)] private float minSpikePatchBoundaryEdgeRatio = 0.45f;
    [SerializeField, Range(0.5f, 4f)] private float maxSpikePatchExtentVoxelScale = 2.2f;
    [SerializeField, Range(-1f, 1f)] private float minSpikePatchNormalDot = 0.72f;
    [SerializeField] private bool fillCleanTsdfContinuityGaps = true;
    [SerializeField, Range(1, 2)] private int tsdfContinuityFillRadiusVoxels = 2;
    [SerializeField, Range(3, 26)] private int minTsdfContinuityNeighborVoxels = 6;
    [SerializeField, Range(1, 6)] private int minTsdfContinuityFaceNeighborVoxels = 2;
    [SerializeField, Range(3, 26)] private int minTsdfContinuitySameSignNeighborVoxels = 10;
    [SerializeField, Range(0.01f, 0.35f)] private float maxTsdfContinuityNeighborAbsValue = 0.24f;
    [SerializeField, Range(0.005f, 0.2f)] private float continuityFillAbsTsdf = 0.035f;
    [SerializeField, Range(1, 12)] private int continuityFillWeight = 2;
    [SerializeField, Min(0)] private int maxTsdfContinuityFillVoxelsPerRebuild = 4096;
    [SerializeField] private bool allowSameSignContinuityBaseFill = true;
    [SerializeField] private bool allowStableProvisionalContinuitySources = true;
    [SerializeField] private bool rejectContinuityFillFromUnsettledNeighbors = true;
    [SerializeField] private bool fillBoundaryNoTsdfGaps = true;
    [SerializeField, Range(3, 26)] private int minBoundaryNoTsdfNeighborVoxels = 8;
    [SerializeField, Range(2, 6)] private int minBoundaryNoTsdfFaceNeighborVoxels = 3;
    [SerializeField] private bool allowBoundaryNoTsdfProvisionalAnchor = true;
    [SerializeField, Range(0, 6)] private int minBoundaryNoTsdfCleanFaceAnchors = 0;
    [SerializeField, Range(1, 12)] private int minBoundaryNoTsdfProvisionalAnchors = 2;
    [SerializeField, Range(1, 6)] private int minBoundaryNoTsdfProvisionalFaceAnchors = 1;
    [SerializeField, Min(0)] private int maxBoundaryNoTsdfFillVoxelsPerRebuild = 1024;
    [SerializeField] private bool bridgeCleanCoplanarBoundaryEdges = true;
    [SerializeField, Range(1f, 4f)] private float maxBoundaryBridgeDistanceVoxelScale = 2.0f;
    [SerializeField, Range(0.75f, 1f)] private float minBoundaryBridgeNormalDot = 0.92f;
    [SerializeField, Range(0.05f, 1.5f)] private float maxBoundaryBridgePlaneDistanceVoxelScale = 0.65f;
    [SerializeField, Range(0.01f, 0.8f)] private float maxBoundaryBridgeAbsTsdf = 0.35f;
    [SerializeField, Range(0, 3)] private int boundaryBridgeCleanRadiusVoxels = 1;
    [SerializeField, Min(0)] private int maxBoundaryBridgesPerRebuild = 1200;
    [SerializeField] private bool doubleSidedTriangles = true;
    [SerializeField] private bool showFallbackSurfaceSamples = false;
    [SerializeField] private bool useFallbackMeshWhenTsdfExtractionEmpty = false;
    [SerializeField] private bool addFallbackOnlyAfterTsdfSupport = true;
    [SerializeField, Range(0.01f, 1f)] private float minFallbackSurfaceSampleWeight = 0.65f;
    [SerializeField, Min(256)] private int maxFallbackSurfaceSamples = 72000;
    [SerializeField, Min(0.002f)] private float fallbackSurfaceSampleSizeMeters = 0.028f;
    [SerializeField] private bool displayFallbackAsCoverageTiles = true;
    [SerializeField, Range(0.25f, 1.5f)] private float coverageTileVoxelScale = 0.45f;
    [SerializeField, Min(256)] private int maxRenderedLightCoverTiles = 72000;
    [SerializeField] private bool renderOnlyStableLightCoverCells = true;
    [SerializeField, Range(1, 16)] private int minLightCoverCellHits = 4;
    [SerializeField] private bool renderOnlyCleanLightCoverCells = true;
    [SerializeField] private bool showUnstableLightCoverRiskCells = true;
    [SerializeField] private bool showDirtyLightCoverRiskCells = true;
    [SerializeField, Range(0, 3)] private int cleanLightCoverCheckRadiusVoxels = 0;
    [SerializeField, Range(1, 16)] private int minCleanLightCoverVoxelWeight = 2;
    [SerializeField, Range(0.01f, 0.8f)] private float maxCleanLightCoverAbsTsdf = 0.35f;
    [SerializeField] private bool requireNearZeroTsdfForLightCover = false;
    [SerializeField] private bool useChunkedSurfaceSamples = true;
    [SerializeField] private bool replaceFallbackSamplesWhenFull = true;
    [SerializeField, Min(0.25f)] private float surfaceChunkSizeMeters = 2.0f;
    [SerializeField, Min(1)] private int maxSurfaceChunks = 1024;
    [SerializeField, Min(64)] private int maxFallbackSamplesPerChunk = 16384;
    [SerializeField] private bool sideAwareSurfaceCells = true;
    [SerializeField] private bool useSideAwareLightCoverKeys = false;
    [SerializeField] private bool splitLightCoverNormalConflicts = false;
    [SerializeField, Range(0f, 1f)] private float surfaceSideOffsetVoxelScale = 0.45f;
    [SerializeField, Min(1)] private int stableSurfaceCellLockHits = 4;
    [SerializeField, Min(0.001f)] private float maxStableCellPositionUpdateMeters = 0.015f;
    [SerializeField] private bool renderStableCellsAsPlanarMeshCover = false;
    [SerializeField] private bool useConnectedPlanarCoverMesh = false;
    [SerializeField] private bool showPlanarCoverSurfaceFill = true;
    [SerializeField, Range(0.6f, 1.8f)] private float planarCoverTileScale = 1.0f;
    [SerializeField] private bool showPlanarCoverWireOverlay = false;
    [SerializeField] private bool showMeshEdgeWireOverlay = true;
    [SerializeField, Range(0.6f, 1.5f)] private float wireTileScale = 1.06f;
    [SerializeField] private Color wireColor = new Color(0.1f, 1f, 0.15f, 0.95f);
    [SerializeField] private Color shellColor = new Color(0.12f, 0.9f, 1f, 1f);
    [SerializeField] private Color cleanCoverCellColor = new Color(0.82f, 1f, 0.96f, 1f);
    [SerializeField] private Color unstableCoverCellColor = new Color(1f, 0.74f, 0.22f, 1f);
    [SerializeField] private Color riskCoverCellColor = new Color(1f, 0.16f, 0.38f, 1f);
    [SerializeField] private Material shellMaterialOverride;
    [SerializeField] private bool debugLog = true;

    [Header("Diagnostic HUD")]
    [SerializeField] private bool showDiagnosticHud = true;
    [SerializeField] private Vector3 hudLocalPosition = new Vector3(0f, -0.18f, 0.78f);
    [SerializeField] private Vector3 hudLocalEuler = new Vector3(8f, 0f, 0f);
    [SerializeField] private Vector3 hudLocalScale = new Vector3(0.00075f, 0.00075f, 0.00075f);
    [SerializeField, Min(0.05f)] private float hudRefreshIntervalSeconds = 0.12f;
    [SerializeField] private Color hudHealthyColor = new Color(0.7f, 1f, 0.92f, 1f);
    [SerializeField] private Color hudWarningColor = new Color(1f, 0.72f, 0.25f, 1f);

    [Header("Confidence Audit")]
    [SerializeField] private bool writeConfidenceAuditOnCapture = true;
    [SerializeField, Min(1)] private int confidenceAuditSampleStride = 1;
    [SerializeField, Min(1000)] private int maxConfidenceAuditRowsPerCapture = 250000;
    [SerializeField] private string confidenceAuditFolderName = "ScanCoverDiagnostics";
    [SerializeField, Range(0.5f, 1f)] private float auditDepthEdgeSupportRatio = 0.88f;
    [SerializeField, Min(0.001f)] private float auditRobustShiftRiskMeters = 0.025f;
    [SerializeField, Range(0f, 1f)] private float auditGrazingViewFacingThreshold = 0.42f;
    [SerializeField, Range(1, 64)] private int auditOldLockedWeightThreshold = 10;
    [SerializeField] private bool clearRawDepthDebugAtCaptureStart = true;

    [Header("Observation Vote - Stage 2")]
    [SerializeField] private ObservationVoteMode observationVoteMode = ObservationVoteMode.Shadow;
    [SerializeField, Range(0f, 1f)] private float voteAcceptScore = 0.72f;
    [SerializeField, Range(0f, 1f)] private float voteRejectScore = 0.38f;
    [SerializeField, Range(0f, 1f)] private float voteGoodSupportRatio = 0.82f;
    [SerializeField, Range(0f, 1f)] private float voteBadSupportRatio = 0.55f;
    [SerializeField, Min(0.001f)] private float voteGoodRobustShiftMeters = 0.015f;
    [SerializeField, Min(0.001f)] private float voteBadRobustShiftMeters = 0.060f;
    [SerializeField, Range(0f, 1f)] private float voteGoodViewFacing = 0.50f;
    [SerializeField, Range(0f, 1f)] private float voteBadViewFacing = 0.15f;
    [SerializeField, Range(0f, 1f)] private float voteGoodSampleWeight = 0.65f;
    [SerializeField, Range(0f, 1f)] private float voteBadSampleWeight = 0.20f;
    [SerializeField, Min(0.001f)] private float voteGoodTemporalDepthDeltaMeters = 0.020f;
    [SerializeField, Min(0.001f)] private float voteBadTemporalDepthDeltaMeters = 0.090f;
    [SerializeField, Min(0.001f)] private float voteGoodHistoricalSurfaceDistanceMeters = 0.050f;
    [SerializeField, Min(0.001f)] private float voteBadHistoricalSurfaceDistanceMeters = 0.160f;
    [SerializeField, Range(0f, 1f)] private float voteGoodOldTsdfResidual = 0.15f;
    [SerializeField, Range(0f, 1f)] private float voteBadOldTsdfResidual = 0.55f;
    [SerializeField, Range(0f, 1f)] private float voteGoodHistoryAgreement = 0.70f;
    [SerializeField, Range(0f, 1f)] private float voteBadHistoryAgreement = 0.25f;
    [SerializeField, Range(1, 4)] private int voteHistoryNeighborhoodRadiusVoxels = 1;
    [SerializeField, Min(0.02f)] private float voteCorrespondenceCellSizeMeters = 0.08f;
    [SerializeField, Min(0.02f)] private float voteMaxCorrespondenceDistanceMeters = 0.20f;
    [SerializeField, Range(0f, 1f)] private float voteGoodBandConflictRatio = 0.05f;
    [SerializeField, Range(0f, 1f)] private float voteBadBandConflictRatio = 0.35f;
    [SerializeField, Range(0f, 2f)] private float voteGoodBandMeanResidual = 0.15f;
    [SerializeField, Range(0f, 2f)] private float voteBadBandMeanResidual = 0.75f;
    [SerializeField] private bool enforceObservationWriteGate = true;
    [SerializeField] private bool holdUnconfirmedPendingWrites = true;
    [SerializeField, Range(0.05f, 1f)] private float confirmedPendingWriteWeightScale = 0.35f;
    [SerializeField] private bool allowPendingBootstrapSeeds = true;
    [SerializeField, Range(0.05f, 0.5f)] private float pendingBootstrapWeightScale = 0.20f;
    [SerializeField] private bool allowStrongSampleSeedWrites = true;
    [SerializeField] private bool lockStrongSampleSeedToTemporaryTsdf = true;
    [SerializeField, Range(0.05f, 0.5f)] private float strongSampleSeedWeightScale = 0.25f;
    [SerializeField, Range(0f, 1f)] private float maxStrongSampleSeedOldTsdfResidual = 0.12f;
    [SerializeField] private bool allowProvisionalTsdfSupportWrites = true;
    [SerializeField, Range(1, 2)] private int provisionalTsdfSupportWeight = 2;
    [SerializeField, Range(0.05f, 0.6f)] private float provisionalTsdfSupportBlend = 0.25f;
    [SerializeField, Range(0.05f, 1f)] private float provisionalTsdfNearSurfaceAbs = 0.30f;
    [SerializeField] private bool allowNearSurfaceProvisionalBootstrap = true;
    [SerializeField, Range(0.02f, 0.5f)] private float maxNearSurfaceProvisionalBootstrapAbsTsdf = 0.18f;
    [SerializeField, Range(0f, 1f)] private float minNearSurfaceProvisionalBootstrapVoteScore = 0.85f;
    [SerializeField, Range(0f, 1f)] private float minNearSurfaceProvisionalBootstrapSupportRatio = 0.90f;
    [SerializeField] private bool requireProvisionalPlaneCompatibility = true;
    [SerializeField, Range(1, 2)] private int provisionalPlaneCompatibilityRadiusVoxels = 1;
    [SerializeField, Range(1, 8)] private int minProvisionalPlaneCompatibleNeighbors = 1;
    [SerializeField, Range(0.5f, 1f)] private float minProvisionalPlaneNormalDot = 0.88f;
    [SerializeField, Range(0.05f, 2f)] private float maxProvisionalPlaneDistanceVoxelScale = 0.85f;
    [SerializeField, Range(0.02f, 0.5f)] private float maxProvisionalPlaneNeighborAbsTsdf = 0.30f;
    [SerializeField] private bool requireProvisionalLocalSupport = true;
    [SerializeField, Range(1, 2)] private int provisionalLocalSupportRadiusVoxels = 1;
    [SerializeField, Range(1, 12)] private int minProvisionalLocalSupportVoxels = 3;
    [SerializeField, Range(0, 6)] private int minProvisionalLocalStableVoxels = 1;
    [SerializeField, Range(0, 6)] private int minProvisionalLocalAxialVoxels = 1;
    [SerializeField, Range(0.02f, 0.5f)] private float maxProvisionalLocalSupportAbsTsdf = 0.25f;
    [SerializeField, Range(0.05f, 1f)] private float maxProvisionalLocalSupportResidual = 0.45f;
    [SerializeField] private bool retireUnconfirmedProvisionalTsdf = true;
    [SerializeField, Range(1, 12)] private int provisionalTsdfMaxAgeFrames = 4;
    [SerializeField, Range(0f, 1f)] private float minProvisionalVoteScore = 0.70f;
    [SerializeField, Range(0f, 1f)] private float minProvisionalSupportRatio = 0.82f;
    [SerializeField, Range(0f, 0.5f)] private float maxProvisionalBandConflictRatio = 0.05f;
    [SerializeField] private bool rejectProvisionalCleanHistoryMismatch = true;
    [SerializeField, Min(0.001f)] private float maxProvisionalSameFrameDepthDeltaMeters = 0.03f;
    [SerializeField, Range(0, 8)] private int maxProvisionalExistingWeight = 2;
    [SerializeField, Min(0.01f)] private float sameFrameConflictDepthMeters = 0.03f;
    [SerializeField, Min(0.03f)] private float sameFrameConflictSearchMeters = 0.18f;
    [SerializeField, Range(1f, 45f)] private float sameFrameConflictMaxNormalAngleDegrees = 20f;
    [SerializeField] private bool enableSameFrameEntryGate = true;
    [SerializeField, Min(0.01f)] private float maxSameFrameEntryProvisionalDepthMeters = 0.06f;
    [SerializeField, Range(0.05f, 0.6f)] private float sameFrameEntryProvisionalWeightScale = 0.20f;
    [SerializeField] private bool gateFormalIntegrateWrites = true;
    [SerializeField] private bool downgradeBlockedFormalIntegrateToProvisional = true;
    [SerializeField, Range(0f, 1f)] private float minFormalIntegrateVoteScore = 0.72f;
    [SerializeField, Range(0f, 1f)] private float minFormalIntegrateSupportRatio = 0.82f;
    [SerializeField, Range(0f, 0.5f)] private float maxFormalIntegrateBandConflictRatio = 0.05f;
    [SerializeField, Range(0f, 2f)] private float maxFormalIntegrateBandMeanResidual = 0.20f;
    [SerializeField] private bool rejectFormalCleanHistoryMismatch = true;
    [SerializeField] private bool rejectFormalIntegrateWeakHistory = true;
    [SerializeField] private bool rejectFormalIntegrateTemporalDepthJump = true;
    [SerializeField] private bool requireHistoryForFormalStrongCurrentClean = true;
    [SerializeField, Range(0, 16)] private int minFormalStrongCurrentOldWeight = 5;
    [SerializeField, Range(0, 16)] private int minFormalStrongCurrentBandHistory = 2;
    [SerializeField, Range(0f, 1f)] private float minFormalStrongCurrentHistoryAgreement = 0.65f;
    [SerializeField] private bool allowLocalSupportForFormalStrongCurrentHistory = true;
    [SerializeField, Range(1, 12)] private int minFormalStrongCurrentLocalSupportVoxels = 4;
    [SerializeField, Range(0, 6)] private int minFormalStrongCurrentLocalStableVoxels = 1;
    [SerializeField, Range(0, 6)] private int minFormalStrongCurrentLocalAxialVoxels = 2;
    [SerializeField] private bool promoteStrongCurrentProvisionalToFormal = true;
    [SerializeField, Range(1, 8)] private int minStrongCurrentProvisionalHits = 2;
    [SerializeField, Range(1, 8)] private int minStrongCurrentProvisionalWeight = 2;
    [SerializeField, Range(0, 16)] private int minStrongCurrentProvisionalBandHistory = 1;
    [SerializeField, Range(0f, 1f)] private float minStrongCurrentProvisionalHistoryAgreement = 0.65f;
    [SerializeField, Range(0.05f, 1f)] private float strongCurrentProvisionalPromoteWeight = 1f;
    [SerializeField] private bool allowProvisionalNeighborPromotionSupport = true;
    [SerializeField, Range(1, 12)] private int minStrongCurrentPromotionLocalSupportVoxels = 2;
    [SerializeField, Range(0, 6)] private int minStrongCurrentPromotionStableVoxels = 0;
    [SerializeField, Range(0, 6)] private int minStrongCurrentPromotionAxialVoxels = 1;
    [SerializeField] private bool rejectStrongCurrentPromotionDoubleLayer = true;
    [SerializeField, Range(1, 4)] private int strongCurrentPromotionDoubleLayerSearchRadiusVoxels = 2;
    [SerializeField, Min(0.01f)] private float maxStrongCurrentPromotionDoubleLayerSeparationMeters = 0.085f;
    [SerializeField, Range(0.05f, 0.75f)] private float maxStrongCurrentPromotionDoubleLayerNeighborAbsTsdf = 0.5f;
    [SerializeField, Range(0f, 1f)] private float maxFormalCleanHistoryOldTsdfResidual = 0.12f;
    [SerializeField, Range(0f, 2f)] private float maxFormalCleanHistoryBandMeanResidual = 0.16f;
    [SerializeField, Range(0f, 0.5f)] private float maxFormalCleanHistoryBandConflictRatio = 0.02f;
    [SerializeField, Min(0.001f)] private float maxFormalIntegrateSameFrameDepthDeltaMeters = 0.03f;
    [SerializeField, Min(0.001f)] private float maxFormalIntegrateCrossFrameDepthDeltaMeters = 0.03f;
    [SerializeField, Range(0f, 1f)] private float maxFormalIntegrateOldTsdfResidual = 0.25f;
    [SerializeField] private bool rejectDirtyReplaceOnCleanHistoryMismatch = true;
    [SerializeField] private bool rejectCrossFrameCleanSurfaceConflict = true;
    [SerializeField, Min(0.01f)] private float crossFrameCleanConflictDepthMeters = 0.03f;
    [SerializeField, Min(0.03f)] private float crossFrameCleanConflictSearchMeters = 0.18f;
    [SerializeField, Range(0.25f, 4f)] private float crossFrameCleanConflictMaxLateralVoxelScale = 1.5f;
    [SerializeField, Range(1f, 60f)] private float crossFrameCleanConflictMaxNormalAngleDegrees = 30f;
    [SerializeField] private bool enableOldCleanTsdfMetabolism = true;
    [SerializeField, Range(1, 8)] private int minOldCleanMetabolismConflictHits = 3;
    [SerializeField, Range(1, 64)] private int maxOldCleanMetabolismWeight = 18;
    [SerializeField, Range(0.05f, 1f)] private float minOldCleanMetabolismResidual = 0.35f;
    [SerializeField, Range(0.05f, 1f)] private float minOldCleanMetabolismCrossFrameVoxelResidual = 0.35f;
    [SerializeField, Range(0.05f, 1f)] private float minOldCleanMetabolismBandVoxelResidual = 0.35f;
    [SerializeField, Range(0f, 1f)] private float minOldCleanMetabolismSupportRatio = 0.82f;
    [SerializeField, Range(0f, 0.5f)] private float maxOldCleanMetabolismBandConflictRatio = 0.08f;
    [SerializeField, Range(0.005f, 0.08f)] private float maxOldCleanMetabolismSameFrameDeltaMeters = 0.035f;
    [SerializeField, Range(1, 8)] private int oldCleanMetabolismDecayWeight = 2;
    [SerializeField, Range(0, 8)] private int oldCleanMetabolismClearWeight = 2;

    [Header("Mesh Diagnostics")]
    [SerializeField] private bool runMeshDiagnosticsOnRebuild = false;
    [SerializeField] private bool showMeshDiagnosticHotspots = false;
    [SerializeField, Range(1, 12)] private int maxDiagnosticHotspots = 6;
    [SerializeField, Range(-1f, 1f)] private float diagnosticNormalPatchMinDot = 0.74f;
    [SerializeField, Range(4, 512)] private int diagnosticSuspectPatchMaxTriangles = 180;
    [SerializeField, Range(0.05f, 0.95f)] private float diagnosticSuspectBoundaryRatio = 0.30f;
    [SerializeField, Range(1f, 12f)] private float diagnosticSuspectExtentVoxelScale = 5.0f;
    [SerializeField, Min(0.005f)] private float diagnosticHotspotMarkerSizeMeters = 0.045f;
    [SerializeField] private Color diagnosticHotspotColor = new Color(1f, 0.08f, 0.02f, 0.95f);
    [SerializeField] private Color diagnosticSecondaryHotspotColor = new Color(1f, 0.85f, 0.05f, 0.9f);
    [SerializeField] private Color pureGeometrySuspectColor = new Color(0.1f, 0.85f, 1f, 1f);
    [SerializeField] private bool showMeshTumorSuspectOverlay = true;
    [SerializeField, Range(0, 2)] private int tumorSuspectTsdfRadiusVoxels = 1;
    [SerializeField, Range(0.05f, 0.95f)] private float tumorSuspectOpenBoundaryRatio = 0.62f;
    [SerializeField, Range(2f, 20f)] private float tumorSuspectSmallIslandExtentVoxelScale = 4.5f;
    [SerializeField, Range(4, 1024)] private int tumorSuspectSmallIslandMaxTriangles = 96;
    [SerializeField, Range(1.2f, 5f)] private float tumorSuspectLongEdgeVoxelScale = 3.2f;
    [SerializeField, Range(10f, 80f)] private float pureGeometryNormalDeviationDegrees = 38f;
    [SerializeField, Range(0.2f, 2f)] private float pureGeometryPlaneOffsetVoxelScale = 0.65f;
    [SerializeField, Range(0.75f, 1f)] private float pureGeometryParallelNormalDot = 0.92f;
    [SerializeField, Range(0.15f, 1.5f)] private float pureGeometryMinLayerSeparationVoxelScale = 0.35f;
    [SerializeField, Range(1f, 5f)] private float pureGeometryMaxLayerSeparationVoxelScale = 3f;
    [SerializeField, Range(0.25f, 2f)] private float pureGeometryMaxLayerLateralVoxelScale = 0.8f;
    [SerializeField] private bool validateDoubleLayerWithWriteProvenance = true;
    [SerializeField, Min(0.01f)] private float doubleLayerMinSourcePlaneSeparationMeters = 0.03f;
    [SerializeField, Range(1f, 60f)] private float doubleLayerMaxSourceNormalAngleDegrees = 30f;
    [SerializeField, Range(0.25f, 3f)] private float doubleLayerMaxSourceLateralVoxelScale = 1.5f;
    [SerializeField, Range(0f, 1f)] private float doubleLayerMinMeshSourceNormalDot = 0.70f;
    [SerializeField] private Color tumorSuspectHighColor = new Color(1f, 0.03f, 0.02f, 1f);
    [SerializeField] private Color tumorSuspectMediumColor = new Color(1f, 0.72f, 0.08f, 0.95f);
    [SerializeField] private Color tumorCauseDirtyColor = new Color(1f, 0.02f, 0.04f, 1f);
    [SerializeField] private Color tumorCauseNonManifoldColor = new Color(0.74f, 0.1f, 1f, 1f);
    [SerializeField] private Color tumorCauseIslandColor = new Color(1f, 0.58f, 0.02f, 1f);
    [SerializeField] private Color tumorCauseOpenColor = new Color(1f, 0.92f, 0.08f, 0.95f);
    [SerializeField] private Color tumorCauseLongEdgeColor = new Color(0.05f, 0.85f, 1f, 0.95f);
    [SerializeField] private bool showDirtyTsdfEvidenceOverlay = false;
    [SerializeField, Range(16, 512)] private int maxDirtyTsdfEvidenceMarkers = 160;
    [SerializeField, Min(0.005f)] private float dirtyTsdfEvidenceMarkerSizeMeters = 0.07f;
    [SerializeField, Range(0.05f, 1f)] private float dirtyTsdfEvidenceLineAlpha = 1f;
    [SerializeField] private bool backfillCurrentDirtyTsdfMarkers = true;
    [SerializeField] private bool forceMeshGridDisplayOnEnable = true;
    [SerializeField] private bool showRawDepthDebugView = false;
    [SerializeField] private bool restrictRawDepthDebugToNearTsdfSurface = true;
    [SerializeField, Range(0, 3)] private int rawDepthDebugNearSurfaceRadiusVoxels = 1;
    [SerializeField, Range(0, 2)] private int rawDepthDebugDirtyRadiusVoxels = 0;
    [SerializeField, Range(0.01f, 0.8f)] private float rawDepthDebugNearSurfaceAbsTsdf = 0.28f;
    [SerializeField] private bool showAcceptedRawDepthDebugSamples = false;
    [SerializeField, Range(1, 64)] private int acceptedRawDepthDebugSampleStride = 12;
    [SerializeField, Range(1, 64)] private int rawDepthDebugDepthEdgeSampleStride = 5;
    [SerializeField, Range(1, 64)] private int rawDepthDebugRobustSampleStride = 8;
    [SerializeField, Range(1, 64)] private int rawDepthDebugRejectedSampleStride = 8;
    [SerializeField, Min(256)] private int maxRawDepthDebugSamples = 12000;
    [SerializeField, Min(0.002f)] private float rawDepthDebugSampleSizeMeters = 0.024f;
    [SerializeField, Range(0.05f, 1f)] private float rawDepthDebugAlpha = 0.96f;
    [SerializeField] private Color rawDepthDebugAcceptedColor = new Color(0.92f, 1f, 1f, 0.24f);
    [SerializeField] private Color rawDepthDebugDirtyColor = new Color(1f, 0.02f, 0.02f, 1f);
    [SerializeField] private Color rawDepthDebugPendingColor = new Color(1f, 0.05f, 0.85f, 1f);
    [SerializeField] private Color rawDepthDebugStableBandColor = new Color(0.05f, 0.9f, 1f, 1f);
    [SerializeField] private Color rawDepthDebugConflictEdgeColor = new Color(1f, 0.92f, 0.02f, 1f);
    [SerializeField] private Color rawDepthDebugConflictViewColor = new Color(1f, 0.42f, 0.02f, 1f);
    [SerializeField] private Color rawDepthDebugConflictLockedColor = new Color(1f, 0.02f, 0.02f, 1f);
    [SerializeField] private Color rawDepthDebugDepthEdgeColor = new Color(1f, 0.9f, 0.05f, 1f);
    [SerializeField] private Color rawDepthDebugRobustColor = new Color(1f, 0.45f, 0.02f, 1f);
    [SerializeField] private Color rawDepthDebugSupportColor = new Color(1f, 0.05f, 0.85f, 1f);
    [SerializeField] private Color rawDepthDebugOutsideColor = new Color(0.05f, 0.35f, 1f, 1f);
    [SerializeField] private Color rawDepthDebugRejectedColor = new Color(0.45f, 0.45f, 0.45f, 1f);
    [Header("Raw Coverage Grid Diagnostics")]
    [SerializeField] private bool enableRawCoverageGridDiagnostics = true;
    [SerializeField] private bool showRawCoverageGridOverlay = true;
    [SerializeField] private bool hideMeshWhenRawCoverageGridOverlay = true;
    [SerializeField] private bool renderRawCoverageGridAsWorldCubes = false;
    [SerializeField, Min(0.03f)] private float rawCoverageGridCellSizeMeters = 0.10f;
    [SerializeField, Min(256)] private int maxRawCoverageGridCells = 4096;
    [SerializeField, Range(1, 64)] private int rawCoverageAcceptedDisplayStride = 5;

    private GameObject _meshObject;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;
    private Material _runtimeMaterial;
    private GameObject _wireObject;
    private MeshFilter _wireMeshFilter;
    private MeshRenderer _wireMeshRenderer;
    private Mesh _wireMesh;
    private Material _wireRuntimeMaterial;
    private Coroutine _captureRoutine;
    private GameObject _hudRoot;
    private Canvas _hudCanvas;
    private Text _hudText;
    private Image _hudPanel;
    private float _nextHudRefresh;
    private GameObject _diagnosticHotspotRoot;
    private Material _diagnosticHotspotMaterial;
    private Material _diagnosticSecondaryHotspotMaterial;
    private readonly List<GameObject> _diagnosticHotspotMarkers = new List<GameObject>(8);
    private readonly List<MeshDiagnosticHotspot> _meshDiagnosticHotspots = new List<MeshDiagnosticHotspot>(16);
    private GameObject _dirtyEvidenceObject;
    private MeshFilter _dirtyEvidenceMeshFilter;
    private MeshRenderer _dirtyEvidenceRenderer;
    private Mesh _dirtyEvidenceMesh;
    private Material _dirtyEvidenceMaterial;
    private readonly List<DirtyTsdfEvidence> _dirtyTsdfEvidence = new List<DirtyTsdfEvidence>(256);
    private GameObject _rawDepthDebugObject;
    private MeshFilter _rawDepthDebugMeshFilter;
    private MeshRenderer _rawDepthDebugRenderer;
    private Mesh _rawDepthDebugMesh;
    private GameObject _holeBoundaryDiagnosticObject;
    private MeshFilter _holeBoundaryDiagnosticMeshFilter;
    private MeshRenderer _holeBoundaryDiagnosticRenderer;
    private Mesh _holeBoundaryDiagnosticMesh;
    private Material _rawDepthDebugMaterial;
    private readonly List<Vector3> _rawDepthDebugPoints = new List<Vector3>(12000);
    private readonly List<RawDepthDebugKind> _rawDepthDebugKinds = new List<RawDepthDebugKind>(12000);
    private readonly Dictionary<Vector3Int, RawCoverageGridCell> _rawCoverageGridCells = new Dictionary<Vector3Int, RawCoverageGridCell>(4096);
    private int _rawDepthDebugReplaceCursor;
    private int _rawDepthDebugClassifierRevision = -1;
    private int _acceptedRawDepthDebugCounter;
    private int _rawDepthDebugDepthEdgeCounter;
    private int _rawDepthDebugRobustCounter;
    private int _rawDepthDebugRejectedCounter;
    private int _rawCoverageOverlayBuildVersion = -1;
    private int _rawCoverageGridVersion;
    private Vector3 _activeTsdfSourceCamera;
    private Vector3 _activeTsdfSourceSurface;
    private bool _activeTsdfSourceValid;
    private int _surfaceDiagnosticCaptureIndex;
    private string _lastHoleDiagnosticPath = "off";
    private string _lastLayerDiagnosticPath = "off";
    private int _lastLayerNoZeroRayCount;
    private int _lastLayerSingleZeroRayCount;
    private int _lastLayerMultiZeroRayCount;
    private int _lastLayerDisplayPairRayCount;
    private int _lastLayerLegalOcclusionRayCount;
    private int _lastLayerDuplicateRayCount;
    private int _lastLayerAmbiguousRayCount;
    private int _lastHoleDiagnosticRowCount;
    public int LastHoleSideRepairCandidateCount { get; private set; }
    public int LastHoleSideRepairAppliedCount { get; private set; }
    public int LastHoleSideRepairBlockedSupportCount { get; private set; }
    public int LastHoleSideRepairBlockedDirtyCount { get; private set; }
    public int LastHoleSideRepairBlockedMultiZeroCount { get; private set; }
    public int LastHoleSideRepairBlockedPlaneCount { get; private set; }
    public int LastHoleSideRepairPlaneBandVoxelCount { get; private set; }
    public int LastHoleSideRepairPlaneConfirmedCount { get; private set; }
    public int LastHoleSideRepairPlaneRetiredCount { get; private set; }
    public int LastHoleSideRepairRetiredNoNearAcceptCount { get; private set; }
    public int LastHoleSideRepairRetiredSignMismatchCount { get; private set; }
    public int LastHoleSideRepairRetiredPlaneMismatchCount { get; private set; }
    public int LastHoleSideRepairRetiredTsdfDeltaCount { get; private set; }
    public int LastHoleSideRepairRetiredExpiredAfterPassCount { get; private set; }
    public int LastHoleSideRepairPlaneDistance0ToHalfCount { get; private set; }
    public int LastHoleSideRepairPlaneDistanceHalfTo075Count { get; private set; }
    public int LastHoleSideRepairPlaneDistance075To1Count { get; private set; }
    public int LastHoleSideRepairPlaneDistance1To15Count { get; private set; }
    public int LastHoleSideRepairPlaneDistance15To2Count { get; private set; }
    public int LastHoleSideRepairPlaneDistanceOver2Count { get; private set; }
    public int LastAtomicAcceptDuplicateCandidateCount { get; private set; }
    public int LastAtomicAcceptDuplicateDowngradeCount { get; private set; }
    public int LastAtomicAcceptDuplicateSameSurfaceCount { get; private set; }
    public int LastDuplicateLayerCleanupQueuedCount { get; private set; }
    public int LastDuplicateLayerCleanupDecayedCount { get; private set; }
    public int LastDuplicateLayerCleanupClearedCount { get; private set; }
    public int LastDuplicateLayerCleanupBoostedCount { get; private set; }
    public int LastDuplicateLayerCleanupMaxEvidence { get; private set; }

    private float[] _tsdf;
    private byte[] _weights;
    private float[] _atomicProvisionalBandTsdf;
    private byte[] _atomicProvisionalBandWeight;
    private int[] _atomicProvisionalBandRetiredLastFrame;
    private byte[] _atomicProvisionalBandRetiredSign;
    private byte[] _surfaceBandVisitFlags;
    private byte[] _surfaceBandAcceptSignLostFlags;
    private int[] _acceptedPositiveLastSequence;
    private int[] _acceptedNegativeLastSequence;
    private int[] _acceptedPositiveRetainedLastSequence;
    private int[] _acceptedNegativeRetainedLastSequence;
    private int _acceptedWriteSequence;
    private byte[] _acceptedSignRecoveryPositiveFrames;
    private byte[] _acceptedSignRecoveryNegativeFrames;
    private int[] _acceptedSignRecoveryPositiveLastFrame;
    private int[] _acceptedSignRecoveryNegativeLastFrame;
    private byte[] _surfaceObservationState;
    private int[] _surfaceObservationLastFrame;
    private int[] _clearedVoxelLastFrame;
    private float[] _pendingTsdf;
    private byte[] _pendingTsdfHits;
    private int[] _pendingTsdfLastFrame;
    private float[] _correctionTsdf;
    private byte[] _correctionTsdfHits;
    private int[] _correctionTsdfLastFrame;
    private int[] _dirtyTsdfLastFrame;
    private byte[] _provisionalTsdf;
    private int[] _provisionalTsdfLastFrame;
    private byte[] _provisionalTsdfHits;
    private readonly Dictionary<int, int> _primaryPlaneHoleConfirmations = new Dictionary<int, int>();
    private readonly Dictionary<int, int> _primaryPlaneHoleLastValidationFrame = new Dictionary<int, int>();
    private readonly Dictionary<int, byte> _primaryPlaneHoleAcceptEvidence = new Dictionary<int, byte>();
    private readonly Dictionary<int, float> _primaryPlaneHoleMinPlaneDistanceVoxels = new Dictionary<int, float>();
    private int _pendingPrimaryPlaneHoleAcceptConfirmedCount;
    private readonly HashSet<Vector3Int> _committedMeshBlocks = new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _candidateMeshBlocks = new HashSet<Vector3Int>();
    private byte[] _oldCleanConflictHits;
    private int[] _oldCleanConflictLastFrame;
    private byte[] _freeSpaceEvidenceHits;
    private int[] _freeSpaceEvidenceLastFrame;
    private int[] _cellVertexIndices;
    private byte[] _detailCandidateCellHits;
    private int[] _detailCandidateCellLastFrame;
    private readonly List<Vector3> _fallbackSurfacePoints = new List<Vector3>(12000);
    private readonly List<Vector3> _fallbackSurfaceNormals = new List<Vector3>(12000);
    private readonly List<int> _fallbackSurfaceVoxelKeys = new List<int>(12000);
    private readonly HashSet<int> _fallbackSurfaceVoxels = new HashSet<int>();
    private readonly Dictionary<Vector3Int, SurfaceChunk> _surfaceChunks = new Dictionary<Vector3Int, SurfaceChunk>();
    private readonly List<Vector3Int> _surfaceChunkOrder = new List<Vector3Int>();
    private readonly float[] _robustDepthSamples = new float[49];
    private int _fallbackSurfaceReplaceCursor;
    private int _dimX;
    private int _dimY;
    private int _dimZ;
    private Vector3 _volumeOriginWorld;
    private bool _volumeInitialized;

    private static readonly int[] CornerOffsetX = { 0, 1, 1, 0, 0, 1, 1, 0 };
    private static readonly int[] CornerOffsetY = { 0, 0, 1, 1, 0, 0, 1, 1 };
    private static readonly int[] CornerOffsetZ = { 0, 0, 0, 0, 1, 1, 1, 1 };

    private static readonly int[,] CubeEdges =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };

    public int IntegratedFrameCount { get; private set; }
    public int LastRawFrameIndex { get; private set; } = -1;
    public int LastInputSampleCount { get; private set; }
    public int LastIntegratedSampleCount { get; private set; }
    public int LastUpdatedVoxelCount { get; private set; }
    public int LastMeshVertexCount { get; private set; }
    public int LastMeshTriangleCount { get; private set; }
    public bool LastMeshUsedExtractedTsdf { get; private set; }
    public int LastRejectedInvalidPositionCount { get; private set; }
    public int LastRejectedDepthRangeCount { get; private set; }
    public int LastRejectedNormalCount { get; private set; }
    public int LastRejectedFacingCount { get; private set; }
    public int LastRejectedConfidenceCount { get; private set; }
    public int LastRejectedOutsideVolumeCount { get; private set; }
    public int LastRejectedDepthDiscontinuityCount { get; private set; }
    public int LastRejectedTsdfDepthSupportCount { get; private set; }
    public int LastRejectedRobustDepthCount { get; private set; }
    public int LastRobustDepthDownweightedCount { get; private set; }
    public int LastRobustDepthCorrectedCount { get; private set; }
    public int LastRejectedDepthEdgeErosionCount { get; private set; }
    public int LastDepthEdgeDownweightedCount { get; private set; }
    public int LastRejectedTsdfConflictCount { get; private set; }
    public int LastPendingStableTsdfCount { get; private set; }
    public int LastProvisionalTsdfSupportWriteCount { get; private set; }
    public int LastProvisionalTsdfSupportBlockedCount { get; private set; }
    public int LastProvisionalTsdfNearSurfaceBlockedCount { get; private set; }
    public int LastProvisionalTsdfNearSurfacePositiveBlockedCount { get; private set; }
    public int LastProvisionalTsdfNearSurfaceNegativeBlockedCount { get; private set; }
    public int LastProvisionalTsdfFarBandSkippedCount { get; private set; }
    public int LastProvisionalTsdfCleanHistoryBlockedCount { get; private set; }
    public int LastProvisionalTsdfDisabledBlockedCount { get; private set; }
    public int LastProvisionalTsdfInvalidBlockedCount { get; private set; }
    public int LastProvisionalTsdfVoteBlockedCount { get; private set; }
    public int LastProvisionalTsdfScoreBlockedCount { get; private set; }
    public int LastProvisionalTsdfSupportRatioBlockedCount { get; private set; }
    public int LastProvisionalTsdfBandBlockedCount { get; private set; }
    public int LastProvisionalTsdfSameFrameBlockedCount { get; private set; }
    public int LastProvisionalTsdfCrossFrameBlockedCount { get; private set; }
    public int LastProvisionalTsdfDirtyPendingBlockedCount { get; private set; }
    public int LastProvisionalTsdfOldWeightBlockedCount { get; private set; }
    public int LastProvisionalTsdfConflictBlockedCount { get; private set; }
    public int LastProvisionalTsdfPlaneBlockedCount { get; private set; }
    public int LastProvisionalTsdfPendingStabilityBlockedCount { get; private set; }
    public int LastProvisionalTsdfFormalDowngradeBlockedCount { get; private set; }
    public int LastProvisionalTsdfExistingWeightBypassCount { get; private set; }
    public int LastProvisionalTsdfBootstrapLocalBypassCount { get; private set; }
    public int LastProvisionalPlaneCompatibilityPassCount { get; private set; }
    public int LastProvisionalPlaneCompatibilityBlockedCount { get; private set; }
    public int LastProvisionalPlaneCompatibilityNoReferenceCount { get; private set; }
    public int LastProvisionalPlaneCompatibilityCandidateCount { get; private set; }
    public int LastProvisionalPlaneCompatibilityNormalRejectedCount { get; private set; }
    public int LastProvisionalPlaneCompatibilityDistanceRejectedCount { get; private set; }
    public int LastProvisionalLocalSupportPassCount { get; private set; }
    public int LastProvisionalLocalSupportBlockedCount { get; private set; }
    public int LastProvisionalLocalSupportNeighborCount { get; private set; }
    public int LastProvisionalLocalSupportStableNeighborCount { get; private set; }
    public int LastProvisionalLocalSupportAxialNeighborCount { get; private set; }
    public int LastProvisionalTsdfConfirmedCount { get; private set; }
    public int LastProvisionalTsdfConfirmedByWeightCount { get; private set; }
    public int LastProvisionalTsdfRetiredCount { get; private set; }
    public int LastAtomicAcceptedBandVoxelWriteCount { get; private set; }
    public int LastAtomicProvisionalBandVoxelWriteCount { get; private set; }
    public int LastAtomicPromotedProvisionalVoxelCount { get; private set; }
    public int LastAtomicRetiredProvisionalVoxelCount { get; private set; }
    public int LastHoleBoundaryWaitingCount { get; private set; }
    public int LastHoleBoundaryRetiredCount { get; private set; }
    public int LastHoleBoundaryNoBandCount { get; private set; }
    public int LastHoleBoundaryPositiveOnlyCount { get; private set; }
    public int LastHoleBoundaryNegativeOnlyCount { get; private set; }
    public int LastHoleBoundarySpatialMissCount { get; private set; }
    public int LastHoleBoundaryWeakCornersCount { get; private set; }
    public int LastHoleBoundaryRenderedMarkerCount { get; private set; }
    public int LastPrimaryLayerDiagnosticMarkerCount { get; private set; }
    public int LastSecondaryLayerDiagnosticMarkerCount { get; private set; }
    public int LastUnclassifiedLayerDiagnosticMarkerCount { get; private set; }
    public int LastConfirmedSecondaryLayerVoxelCount { get; private set; }
    public int LastLayerHoleTrendHoleLoad { get; private set; }
    public int LastLayerHoleTrendSecondaryDelta { get; private set; }
    public int LastLayerHoleTrendHoleDelta { get; private set; }
    public float LastLayerHoleTrendCorrelation { get; private set; }
    public int LastLayerHoleTrendSampleCount { get; private set; }
    public int LastHoleCauseNoRawCount { get; private set; }
    public int LastHoleCausePendingCount { get; private set; }
    public int LastHoleCauseNoBandCount { get; private set; }
    public int LastHoleCauseNoZeroCrossCount { get; private set; }
    public int LastHoleCausePositiveOnlyCount { get; private set; }
    public int LastHoleCauseNegativeOnlyCount { get; private set; }
    public int LastMissingSideNoVisitCount { get; private set; }
    public int LastMissingSideProvisionalCount { get; private set; }
    public int LastMissingSideBlockedCount { get; private set; }
    public int LastMissingSideAcceptLostCount { get; private set; }
    public int LastMissingSidePendingNoPromoteCount { get; private set; }
    public int LastMissingSideRejectCount { get; private set; }
    public int LastMissingSideOtherBlockCount { get; private set; }
    public int LastMissingSideAcceptedOverwrittenCount { get; private set; }
    public int LastMissingSideAcceptedSpatialMissCount { get; private set; }
    public int LastMissingSideAcceptedReflattenedCount { get; private set; }
    public int LastReflattenedByCarveCount { get; private set; }
    public int LastReflattenedByClearCount { get; private set; }
    public int LastReflattenedByProvisionalCount { get; private set; }
    public int LastReflattenedByOtherCount { get; private set; }
    public int LastReflattenedByAcceptedWriterCount { get; private set; }
    public int LastReflattenedByContinuityWriterCount { get; private set; }
    public int LastReflattenedByFusionWriterCount { get; private set; }
    public int LastReflattenedByUntrackedWriterCount { get; private set; }
    public int LastHardAuditIntegrationCount { get; private set; }
    public int LastHardAuditRetireCount { get; private set; }
    public int LastHardAuditContinuityCount { get; private set; }
    public int LastHardAuditExtractionCount { get; private set; }
    public int LastHardAuditTsdfOnlyCount { get; private set; }
    public int LastHardAuditWeightOnlyCount { get; private set; }
    public int LastHardAuditBothCount { get; private set; }
    public int LastMissingSideGoneCount { get; private set; }
    public int LastMissingSideInvalidCount { get; private set; }
    public int LastAcceptedSignRecoveryCandidateCount { get; private set; }
    public int LastAcceptedSignRecoveryNeighborBlockedCount { get; private set; }
    public int LastAcceptedSignRecoveryAppliedCount { get; private set; }
    public int LastHoleCauseVertexRejectCount { get; private set; }
    public int LastHoleCauseQuadRejectCount { get; private set; }
    public int LastHoleCauseClearedCount { get; private set; }
    public int LastHoleCauseLocalOffsetCount { get; private set; }
    public int LastHoleCauseCrossFrameCount { get; private set; }
    public int LastProvisionalTsdfRetiredExpiredCount { get; private set; }
    public int LastProvisionalTsdfDirtyClearedCount { get; private set; }
    public int LastOldCleanMetabolismWatchCount { get; private set; }
    public int LastOldCleanMetabolismDecayCount { get; private set; }
    public int LastOldCleanMetabolismClearCount { get; private set; }
    public int LastOldCleanMetabolismBlockedCount { get; private set; }
    public int LastOldCleanMetabolismCandidateCount { get; private set; }
    public int LastOldCleanMetabolismWaitingHitsCount { get; private set; }
    public int LastOldCleanMetabolismBlockedSupportCount { get; private set; }
    public int LastOldCleanMetabolismBlockedSameFrameCount { get; private set; }
    public int LastOldCleanMetabolismBlockedWeightCount { get; private set; }
    public int LastOldCleanMetabolismBlockedResidualCount { get; private set; }
    public int LastOldCleanMetabolismBlockedDirtyPendingCount { get; private set; }
    public int LastOldCleanMetabolismSkippedWeakBandCount { get; private set; }
    public int LastOldCleanMetabolismSkippedWeakCrossFrameCount { get; private set; }
    public int LastPendingTsdfCorrectionCount { get; private set; }
    public int LastCorrectedTsdfCount { get; private set; }
    public int LastReplacedDirtyTsdfCount { get; private set; }
    public int LastGuardedDirtyTsdfReplaceCount { get; private set; }
    public int LastDirtyTsdfBandRepairCount { get; private set; }
    public int LastDirtyTsdfBandRepairSampleCount { get; private set; }
    public int LastDirtyTsdfBandRepairTriggerCount { get; private set; }
    public int LastDirtyTsdfBandRepairProbeCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedDisabledCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedNoSampleCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedNoHistoryCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedLowConflictCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedBudgetCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedOutsideCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedEmptyCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedWeightCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedSameSignCount { get; private set; }
    public int LastDirtyTsdfBandRepairBlockedResidualCount { get; private set; }
    public int LastCleanedDirtyTsdfNeighborCount { get; private set; }
    public int LastDirtyTsdfQuarantineCount { get; private set; }
    public int LastDirtyTsdfPendingNewCount { get; private set; }
    public int LastDirtyTsdfPendingRepeatCount { get; private set; }
    public int LastDirtyTsdfRejectedConflictCount { get; private set; }
    public int LastDirtyTsdfReplaceBlockedDisabledCount { get; private set; }
    public int LastDirtyTsdfReplaceBlockedWeightCount { get; private set; }
    public int LastDirtyTsdfReplaceBlockedHitsCount { get; private set; }
    public int LastDirtyTsdfReplaceBlockedCleanHistoryCount { get; private set; }
    public int LastGuardedDirtyTsdfBlockedHitsCount { get; private set; }
    public int LastGuardedDirtyTsdfBlockedWeightCount { get; private set; }
    public int LastGuardedDirtyTsdfBlockedValueCount { get; private set; }
    public int LastDirtyTsdfQuarantineNewCount { get; private set; }
    public int LastDirtyTsdfQuarantineRefreshCount { get; private set; }
    public int LastDirtyTsdfActiveCount { get; private set; }
    public int LastDirtyTsdfExpiredCount { get; private set; }
    public int LastCarvedFreeSpaceVoxelCount { get; private set; }
    public int LastFreeSpaceEvidenceCandidateCount { get; private set; }
    public int LastFreeSpaceEvidenceNewCount { get; private set; }
    public int LastFreeSpaceEvidenceRepeatCount { get; private set; }
    public int LastFreeSpaceEvidenceWaitingCount { get; private set; }
    public int LastFreeSpaceEvidenceAppliedCount { get; private set; }
    public int LastFreeSpaceEvidenceClearedCount { get; private set; }
    public int LastFreeSpaceEvidenceBlockedHighWeightCount { get; private set; }
    public int LastFreeSpaceEvidenceBlockedSameFrameCount { get; private set; }
    public int LastFreeSpaceEvidenceDuplicateFrameCount { get; private set; }
    public int LastFreeSpaceEvidenceCancelledBySurfaceCount { get; private set; }
    public int LastNeighborFilledSampleCount { get; private set; }
    public int LastRejectedFallbackWeakCount { get; private set; }
    public int LastSkippedFallbackCapacityCount { get; private set; }
    public int LastSurfaceChunkCount { get; private set; }
    public int LastRenderedFallbackSampleCount { get; private set; }
    public int LastEvictedSurfaceChunkCount { get; private set; }
    public int LastEvictedSurfaceCellCount { get; private set; }
    public int TotalEvictedSurfaceChunkCount { get; private set; }
    public int TotalEvictedSurfaceCellCount { get; private set; }
    public int LastSurfaceCellConflictCount { get; private set; }
    public int LastPlanarCoverTileCount { get; private set; }
    public int LastSkippedDirtyLightCoverCellCount { get; private set; }
    public int LastBlockedDirtyLightCoverSampleCount { get; private set; }
    public int TotalBlockedDirtyLightCoverSampleCount { get; private set; }
    public int LastSkippedUnstableLightCoverCellCount { get; private set; }
    public int LastCleanLightCoverCellCount { get; private set; }
    public int LastUnstableLightCoverCellCount { get; private set; }
    public int LastRiskLightCoverCellCount { get; private set; }
    public int LastWireSegmentCount { get; private set; }
    public int LastPrunedMeshComponentCount { get; private set; }
    public int LastPrunedMeshTriangleCount { get; private set; }
    public int LastPrunedComponentTriangleCount { get; private set; }
    public int LastPrunedDanglingTriangleCount { get; private set; }
    public int LastPrunedSpikeTriangleCount { get; private set; }
    public int LastStrictSurfaceEdgeCount { get; private set; }
    public int LastRelaxedSurfaceEdgeCount { get; private set; }
    public int LastAddedSurfaceQuadCount { get; private set; }
    public int LastSurfaceCellScanCount { get; private set; }
    public int LastStrictSurfaceCellCandidateCount { get; private set; }
    public int LastRelaxedSurfaceCellCandidateCount { get; private set; }
    public int LastBuiltSurfaceCellVertexCount { get; private set; }
    public int LastRejectedNoSurfaceCellCount { get; private set; }
    public int LastRejectedMainGateCellCount { get; private set; }
    public int LastRejectedNoEdgeCellCount { get; private set; }
    public int LastRejectedEdgeCountCellCount { get; private set; }
    public int LastSurfaceQuadCandidateCount { get; private set; }
    public int LastPrePruneMeshTriangleCount { get; private set; }
    public int LastPostComponentMeshTriangleCount { get; private set; }
    public int LastPostBridgeMeshTriangleCount { get; private set; }
    public int LastRejectedRelaxedQuadCount { get; private set; }
    public int LastRejectedWeakQuadCount { get; private set; }
    public int LastRejectedCorrectionPendingCellCount { get; private set; }
    public int LastRejectedCorrectionPendingQuadCount { get; private set; }
    public int LastRejectedDirtyQuarantineCellCount { get; private set; }
    public int LastRejectedDirtyQuarantineQuadCount { get; private set; }
    public int LastBoundaryBridgeCandidateCount { get; private set; }
    public int LastAddedBoundaryBridgeCount { get; private set; }
    public int LastRejectedBoundaryBridgeCount { get; private set; }
    public int LastRejectedBadMeshRebuildCount { get; private set; }
    public int LastHeldDisplayTriangleCount { get; private set; }
    public int LastCommittedMeshBlockCount { get; private set; }
    public int LastCandidateMeshBlockCount { get; private set; }
    public int LastRetainedCommittedMeshBlockCount { get; private set; }
    public int LastCommittedMeshHoldSpatialCount { get; private set; }
    public int LastCommittedMeshHoldTriangleCount { get; private set; }
    public int LastCommittedMeshGrowthTriangleCount { get; private set; }
    public int LastDiagMeshComponentCount { get; private set; }
    public int LastDiagLargestComponentTriangles { get; private set; }
    public int LastDiagBoundaryEdgeCount { get; private set; }
    public int LastDiagNonManifoldEdgeCount { get; private set; }
    public int LastDiagNormalPatchCount { get; private set; }
    public int LastDiagSuspectPatchCount { get; private set; }
    public float LastDiagWorstPatchScore { get; private set; }
    public int LastDiagWorstPatchTriangles { get; private set; }
    public float LastDiagWorstPatchBoundaryRatio { get; private set; }
    public float LastDiagWorstPatchExtentMeters { get; private set; }
    public string LastDiagLikelyCause { get; private set; } = "none";
    public int LastTumorSuspectTriangleCount { get; private set; }
    public int LastTumorRiskVertexCount { get; private set; }
    public int LastTumorDirtyTriangleCount { get; private set; }
    public int LastTumorPendingTriangleCount { get; private set; }
    public int LastTumorReplaceHitBlockedTriangleCount { get; private set; }
    public int LastTumorReplaceWeightBlockedTriangleCount { get; private set; }
    public int LastTumorReplaceReadyTriangleCount { get; private set; }
    public int LastSuspectMeshSourceIntegrateVoxelCount { get; private set; }
    public int LastSuspectMeshSourceProvisionalVoxelCount { get; private set; }
    public int LastSuspectMeshSourceStrongSeedVoxelCount { get; private set; }
    public int LastSuspectMeshSourceContinuityVoxelCount { get; private set; }
    public int LastSuspectMeshSourceCarveVoxelCount { get; private set; }
    public int LastSuspectMeshSourceReplaceVoxelCount { get; private set; }
    public int LastSuspectMeshSourceRepairVoxelCount { get; private set; }
    public int LastSuspectMeshSourceOtherVoxelCount { get; private set; }
    public int LastSuspectMeshSourceUnknownVoxelCount { get; private set; }
    public int LastTumorIslandTriangleCount { get; private set; }
    public int LastTumorOpenEdgeTriangleCount { get; private set; }
    public int LastTumorNonManifoldTriangleCount { get; private set; }
    public int LastTumorLongEdgeTriangleCount { get; private set; }
    public int LastIslandCauseComponentCount { get; private set; }
    public int LastIslandCauseBoundaryVoxelCount { get; private set; }
    public int LastIslandCauseNoTsdfCount { get; private set; }
    public int LastIslandCausePendingCount { get; private set; }
    public int LastIslandCauseDirtyCount { get; private set; }
    public int LastIslandCauseLowWeightCount { get; private set; }
    public int LastIslandCausePlaneMismatchCount { get; private set; }
    public int LastIslandCausePrunedCount { get; private set; }
    public int LastTsdfContinuityCandidateCount { get; private set; }
    public int LastTsdfContinuityFilledCount { get; private set; }
    public int LastTsdfContinuitySameSignFilledCount { get; private set; }
    public int LastTsdfContinuityMixedSignFilledCount { get; private set; }
    public int LastTsdfContinuityProvisionalNeighborCount { get; private set; }
    public int LastTsdfContinuityBlockedDirtyPendingCount { get; private set; }
    public int LastTsdfContinuityBlockedLowSupportCount { get; private set; }
    public int LastTsdfContinuityBlockedBudgetCount { get; private set; }
    public int LastTsdfContinuityBlockedUnsettledNeighborCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfCandidateCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFilledCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedSupportCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedFaceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedAxisCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedCleanAnchorCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedProvisionalAnchorCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedNearSurfaceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfBlockedBudgetCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfRelaxedAnchorFilledCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfVerifiedJointAnchorFilledCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfZeroCleanJointAnchorFilledCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalPresentNeighborCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalFacePresentCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalAcceptedNeighborCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalAcceptedFaceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalBlockedWeightCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalBlockedDirtyPendingCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalBlockedTsdfCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalBlockedResidualCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalBlockedProvenanceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfProvisionalBlockedNotFaceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport0Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport1Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport2Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport3Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport4Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport5Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSupport6Count { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceDeficitOneCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceDeficitTwoPlusCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotAcceptedCleanCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotAcceptedProvisionalCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotBlockedWeightCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotBlockedDirtyPendingCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotBlockedProvenanceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotBlockedTsdfCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotBlockedResidualCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfFaceSlotOutOfBoundsCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfVerifiedTwoFaceCandidateCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfVerifiedTwoFaceFilledCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportPassedCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportDeficitOneCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportDeficitTwoCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportDeficitThreeCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportDeficitFourPlusCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotAcceptedCleanCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotAcceptedProvisionalCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotBlockedWeightCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotBlockedDirtyPendingCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotBlockedProvenanceCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotBlockedTsdfCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotBlockedResidualCount { get; private set; }
    public int LastTsdfBoundaryNoTsdfSupportSlotOutOfBoundsCount { get; private set; }
    public int LastPureGeometrySuspectTriangleCount { get; private set; }
    public int LastPureGeometryNormalTriangleCount { get; private set; }
    public int LastPureGeometryProtrusionTriangleCount { get; private set; }
    public int LastPureGeometryDoubleLayerTriangleCount { get; private set; }
    public int LastDoubleLayerSourceDirtyCount { get; private set; }
    public int LastDoubleLayerSourcePendingCount { get; private set; }
    public int LastDoubleLayerSourceBandConflictCount { get; private set; }
    public int LastDoubleLayerSourceOldLockedCount { get; private set; }
    public int LastDoubleLayerSourceCleanCount { get; private set; }
    public int LastDoubleLayerSourceEmptyCount { get; private set; }
    public int LastDoubleLayerSourceMixedCount { get; private set; }
    public int LastDoubleLayerSourceUnknownCount { get; private set; }
    public int LastDoubleLayerCleanNoLifecycleCount { get; private set; }
    public int LastDoubleLayerCleanNoProvenanceCount { get; private set; }
    public int LastDoubleLayerCleanConflictHistoryCount { get; private set; }
    public int LastDoubleLayerCleanReplaceHistoryCount { get; private set; }
    public int LastDoubleLayerCleanExpiredDirtyCount { get; private set; }
    public int LastDoubleLayerCleanLowWeightCount { get; private set; }
    public int LastDoubleLayerCleanHighWeightCount { get; private set; }
    public int LastDoubleLayerCleanLastIntegrateCount { get; private set; }
    public int LastDoubleLayerCleanLastCarveCount { get; private set; }
    public int LastDoubleLayerCleanLastReplaceCount { get; private set; }
    public int LastDoubleLayerCleanLastRepairCount { get; private set; }
    public int LastDoubleLayerCleanIntegrateCarveHistoryCount { get; private set; }
    public int LastDoubleLayerCleanMultiFrameHistoryCount { get; private set; }
    public int LastVoteSameFrameConflictRejectCount { get; private set; }
    public int LastVotePendingHoldCount { get; private set; }
    public int LastVotePendingConfirmedWriteCount { get; private set; }
    public int LastVotePendingBootstrapWriteCount { get; private set; }
    public int LastStrongSampleSeedWriteCount { get; private set; }
    public int LastStrongSampleSeedBlockedCount { get; private set; }
    public int LastStrongSampleSeedTemporaryBlockedCount { get; private set; }
    public int LastStrongSampleSeedTempNearSurfaceBlockCount { get; private set; }
    public int LastStrongSampleSeedTempVoteBlockCount { get; private set; }
    public int LastStrongSampleSeedTempScoreBlockCount { get; private set; }
    public int LastStrongSampleSeedTempSupportBlockCount { get; private set; }
    public int LastStrongSampleSeedTempBandBlockCount { get; private set; }
    public int LastStrongSampleSeedTempSameFrameBlockCount { get; private set; }
    public int LastStrongSampleSeedTempCrossFrameBlockCount { get; private set; }
    public int LastStrongSampleSeedTempDirtyPendingBlockCount { get; private set; }
    public int LastStrongSampleSeedTempOldWeightBlockCount { get; private set; }
    public int LastStrongSampleSeedTempConflictBlockCount { get; private set; }
    public int LastStrongSampleSeedTempLocalBlockCount { get; private set; }
    public int LastVoteStrongCurrentAcceptCount { get; private set; }
    public int LastVoteCrossFrameCleanConflictRejectCount { get; private set; }
    public int LastSameFrameEntryStableCount { get; private set; }
    public int LastSameFrameEntryProvisionalCount { get; private set; }
    public int LastSameFrameEntryProvisionalWriteCount { get; private set; }
    public int LastSameFrameEntryRejectedCount { get; private set; }
    public int LastSameFrameEntryHeldCount { get; private set; }
    public int LastFormalIntegrateWriteCount { get; private set; }
    public int LastFormalIntegrateBlockedCount { get; private set; }
    public int LastFormalIntegrateProvisionalCount { get; private set; }
    public int LastFormalIntegrateBlockVoteCount { get; private set; }
    public int LastFormalIntegrateBlockScoreCount { get; private set; }
    public int LastFormalIntegrateBlockSupportCount { get; private set; }
    public int LastFormalIntegrateBlockBandCount { get; private set; }
    public int LastFormalIntegrateBlockResidualCount { get; private set; }
    public int LastFormalIntegrateBlockCleanHistoryCount { get; private set; }
    public int LastFormalIntegrateBlockWeakHistoryCount { get; private set; }
    public int LastFormalIntegrateBlockTemporalDepthJumpCount { get; private set; }
    public int LastFormalIntegrateBlockStrongCurrentHistoryCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBypassCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockLocalCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockSupportCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockStableCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockAxialCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockResidualCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockPlaneCount { get; private set; }
    public int LastFormalStrongCurrentLocalHistoryBlockDoubleLayerCount { get; private set; }
    public int LastStrongCurrentProvisionalPromotedCount { get; private set; }
    public int LastStrongCurrentProvisionalPromotionBlockedCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockDisabledCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockTagCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockStorageCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockInvalidCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockNoProvisionalCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockNoProvisionalNearSurfaceCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockNoProvisionalFarBandCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockHitsCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockWeightCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockBandHistoryCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockAgreementCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockCleanHistoryCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockDirtyPendingCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockSameFrameCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockCrossFrameCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockConflictCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockLocalCount { get; private set; }
    public int LastStrongCurrentProvisionalBlockPlaneCount { get; private set; }
    public int LastStrongCurrentPromotionLocalPassCount { get; private set; }
    public int LastStrongCurrentPromotionLocalBlockCount { get; private set; }
    public int LastStrongCurrentPromotionLocalNeighborCount { get; private set; }
    public int LastStrongCurrentPromotionLocalProvisionalNeighborCount { get; private set; }
    public int LastStrongCurrentPromotionLocalStableNeighborCount { get; private set; }
    public int LastStrongCurrentPromotionLocalAxialNeighborCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerCandidateCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerBlockCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerNormalRejectCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerPlaneRejectCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerLateralRejectCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerDirtyNeighborCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerIntegrateBlockCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerReplaceBlockCount { get; private set; }
    public int LastStrongCurrentPromotionDoubleLayerOtherBlockCount { get; private set; }
    public int LastFormalIntegrateBlockSameFrameCount { get; private set; }
    public int LastFormalIntegrateBlockCrossFrameCount { get; private set; }
    public int LastFormalIntegrateBlockOldTsdfCount { get; private set; }
    public int LastFormalIntegrateBlockDirtyPendingCount { get; private set; }
    public int LastDoubleLayerValidatedPairCount { get; private set; }
    public int LastDoubleLayerRejectedMissingSourceCount { get; private set; }
    public int LastDoubleLayerRejectedSourceNormalCount { get; private set; }
    public int LastDoubleLayerRejectedSourcePlaneCount { get; private set; }
    public int LastDoubleLayerRejectedSourceLateralCount { get; private set; }
    public int LastDoubleLayerMeshAlignmentMismatchCount { get; private set; }
    public int LastDirtyEvidenceRenderedCount { get; private set; }
    public int LastDirtyEvidenceBackfillCount { get; private set; }
    public int LastRawDepthDebugRenderedCount { get; private set; }
    public int LastRawDepthDebugAcceptedCount { get; private set; }
    public int LastRawDepthDebugDirtyCount { get; private set; }
    public int LastRawDepthDebugPendingCount { get; private set; }
    public int LastRawDepthDebugStableBandCount { get; private set; }
    public int LastRawDepthDebugConflictEdgeCount { get; private set; }
    public int LastRawDepthDebugConflictViewCount { get; private set; }
    public int LastRawDepthDebugConflictLockedCount { get; private set; }
    public int LastRawDepthDebugRejectedCount { get; private set; }
    public int LastRawDepthDebugDepthEdgeCount { get; private set; }
    public int LastRawDepthDebugRobustCount { get; private set; }
    public int LastRawDepthDebugSupportCount { get; private set; }
    public int LastRawDepthDebugOutsideCount { get; private set; }
    public int LastRawCoverageGridSampleCount { get; private set; }
    public int LastRawCoverageGridDroppedSampleCount { get; private set; }
    public int LastRawCoverageGridCellCount { get; private set; }
    public int LastRawCoverageAcceptedCellCount { get; private set; }
    public int LastRawCoverageProblemCellCount { get; private set; }
    public int LastRawCoverageMixedCellCount { get; private set; }
    public int LastRawCoverageAcceptedComponentCount { get; private set; }
    public int LastRawCoverageLargestAcceptedComponentCells { get; private set; }
    public int LastRawCoverageProblemComponentCount { get; private set; }
    public int LastRawCoverageLargestProblemComponentCells { get; private set; }
    public int LastRawCoverageRenderedCellCount { get; private set; }
    public int LastMainShellPromotedCellCount { get; private set; }
    public int LastCandidateShellHeldCellCount { get; private set; }
    public int LastDetailCandidateHeldCellCount { get; private set; }
    public int LastDetailCandidatePromotedCellCount { get; private set; }

    private sealed class SurfaceChunk
    {
        public readonly List<Vector3> Points = new List<Vector3>(256);
        public readonly List<Vector3> Normals = new List<Vector3>(256);
        public readonly List<Vector3Int> LocalCellKeys = new List<Vector3Int>(256);
        public readonly HashSet<Vector3Int> LocalCells = new HashSet<Vector3Int>();
        public readonly Dictionary<Vector3Int, SurfaceCell> Cells = new Dictionary<Vector3Int, SurfaceCell>();
        public int ReplaceCursor;
    }

    private sealed class SurfaceCell
    {
        public Vector3 Point;
        public Vector3 Normal;
        public int Hits;
    }

    private readonly struct CoverVertexKey
    {
        public readonly int Axis;
        public readonly int Sign;
        public readonly int Plane;
        public readonly int U;
        public readonly int V;

        public CoverVertexKey(int axis, int sign, int plane, int u, int v)
        {
            Axis = axis;
            Sign = sign;
            Plane = plane;
            U = u;
            V = v;
        }

        public override bool Equals(object obj)
        {
            return obj is CoverVertexKey other &&
                   Axis == other.Axis &&
                   Sign == other.Sign &&
                   Plane == other.Plane &&
                   U == other.U &&
                   V == other.V;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = Axis;
                hash = hash * 397 ^ Sign;
                hash = hash * 397 ^ Plane;
                hash = hash * 397 ^ U;
                hash = hash * 397 ^ V;
                return hash;
            }
        }
    }

    private struct MeshDiagnosticHotspot
    {
        public Vector3 Center;
        public Vector3 Normal;
        public float Score;
        public float BoundaryRatio;
        public float ExtentMeters;
        public int Triangles;
        public string Reason;
    }

    private enum DirtyTsdfEvidenceReason
    {
        PendingConflict,
        RepeatedPending,
        ReplaceDirty,
        CorrectedOldDepth,
        CleanupNeighbor,
        RejectedConflict
    }

    private struct DirtyTsdfEvidence
    {
        public int VoxelIndex;
        public int FrameIndex;
        public int OldWeight;
        public float OldTsdf;
        public float NewTsdf;
        public Vector3 VoxelCenter;
        public Vector3 CameraPosition;
        public Vector3 SurfacePoint;
        public DirtyTsdfEvidenceReason Reason;
    }

    private enum SurfaceEdgeKind
    {
        None,
        Strict,
        Relaxed
    }

    private enum LightCoverCellRisk
    {
        Clean,
        Unstable,
        Risk
    }

    private enum RawDepthDebugKind
    {
        Accepted,
        Dirty,
        Pending,
        StableBand,
        ConflictEdge,
        ConflictView,
        ConflictLocked,
        Rejected,
        DepthEdge,
        Robust,
        Support,
        Outside
    }

    private enum HoleBoundaryDiagnosticKind
    {
        WaitingPromotion,
        Retired,
        NoBand,
        PositiveOnly,
        NegativeOnly,
        SpatialMiss,
        WeakCorners
    }

    private enum HoleSupportCause
    {
        NoRaw,
        Pending,
        NoBand,
        NoZeroCross,
        VertexReject,
        QuadReject,
        Cleared,
        LocalOffset,
        CrossFrame
    }

    private enum DiagnosticLayerRole
    {
        None,
        Primary,
        Secondary
    }

    private enum MissingSideCause
    {
        NoVisit,
        Provisional,
        AcceptLost,
        PendingNoPromote,
        Reject,
        AcceptedOverwritten,
        AcceptedSpatialMiss,
        AcceptedReflattenedCarve,
        AcceptedReflattenedClear,
        AcceptedReflattenedProvisional,
        AcceptedReflattenedAcceptedWriter,
        AcceptedReflattenedContinuityWriter,
        AcceptedReflattenedFusionWriter,
        AcceptedReflattenedUntrackedWriter,
        AcceptedReflattenedOther,
        OtherBlock,
        Gone,
        Invalid
    }

    private struct RawCoverageGridCell
    {
        public Vector3 PositionSum;
        public int AcceptedCount;
        public int ProblemCount;
        public RawDepthDebugKind WorstKind;
    }

    private enum ObservationVoteMode
    {
        Shadow,
        RejectOnly
    }

    private enum ObservationVoteState
    {
        Unassessed,
        Accept,
        Pending,
        Reject
    }

    private StringBuilder _confidenceAuditRows;
    private StringBuilder _confidenceAuditFrameRows;
    private StringBuilder _doubleLayerPairRows;
    private int _doubleLayerPairRowCount;
    private int _doubleLayerPairDroppedRows;
    private StreamWriter _contributionLedgerWriter;
    private string _contributionLedgerPath;
    private int _contributionLedgerRowCount;
    private int _contributionLedgerWriteFailures;
    private int _activeLedgerSampleIndex = -1;
    private int _activeLedgerPixelX = -1;
    private int _activeLedgerPixelY = -1;
    private string _confidenceAuditDirectory;
    private string _confidenceAuditStem;
    private int _confidenceAuditRowCount;
    private int _confidenceAuditDroppedRows;
    private int _confidenceAuditCaptureIndex;
    private int _confidenceAuditIntegratedFrames;
    private string _lastConfidenceAuditPath = "off";
    private static readonly CultureInfo AuditCulture = CultureInfo.InvariantCulture;
    private bool _auditSampleActive;
    private float _auditCenterAbsSampleTsdf;
    private float _auditCenterSampleTsdf;
    private float _auditCenterOldTsdf;
    private int _auditCenterOldWeight;
    private int _auditCenterStableHitsBefore;
    private int _auditCenterStableHitsAfter;
    private int _auditCenterCorrectionHitsBefore;
    private int _auditCenterCorrectionHitsAfter;
    private bool _auditCenterSignFlip;
    private float _auditCenterResidual;
    private string _auditCenterOutcome = "none";
    private int _auditBandWritten;
    private int _auditBandPendingStable;
    private int _auditBandPendingCorrection;
    private int _auditBandReplaced;
    private int _auditBandCorrected;
    private int _auditBandRejectedConflict;
    private float _auditSampleSupportRatio;
    private float _auditSampleRobustShiftMeters;
    private float _auditSampleViewFacing;
    private Vector3 _auditSampleNormal;
    private float _auditTemporalDepthDeltaMeters = -1f;
    private float _auditTemporalWorldDeltaMeters = -1f;
    private float _auditTemporalNormalDeltaDegrees = -1f;
    private float _auditSameFrameDepthDeltaMeters = -1f;
    private float _auditSameFrameNormalDeltaDegrees = -1f;
    private float _auditCrossFrameCleanDepthDeltaMeters = -1f;
    private float _auditCrossFrameCleanLateralMeters = -1f;
    private float _auditCrossFrameCleanNormalDeltaDegrees = -1f;
    private int _auditCrossFrameCleanFrameGap = -1;
    private int _auditCrossFrameCleanWeight = 0;
    private float _auditHistoricalSurfaceDistanceMeters = -1f;
    private float _auditOldTsdfResidual = -1f;
    private int _auditOldTsdfWeight;
    private float _auditHistoryAgreement = -1f;
    private int _auditHistorySupportCount;
    private int _auditBandHistoryCount;
    private int _auditBandConflictCount;
    private int _auditBandHighWeightConflictCount;
    private float _auditBandConflictRatio = -1f;
    private float _auditBandHighWeightConflictRatio = -1f;
    private float _auditBandMeanResidual = -1f;
    private float _auditBandMaxResidual = -1f;
    private ObservationVoteState _auditVoteState;
    private ObservationVoteState _activeAtomicBandVote;
    private bool _activeAtomicBandWrite;
    private byte _activeSurfaceObservationState;
    private float _auditVoteScore;
    private string _auditVoteReasons = "unassessed";
    private bool _auditVoteEnforced;
    private bool _auditStrongSampleSeedWrite;
    private int _auditVoteAcceptCount;
    private int _auditVotePendingCount;
    private int _auditVoteRejectCount;
    private int _auditVoteEnforcedRejectCount;
    private Dictionary<Vector3Int, VoteHistoryCell> _votePreviousFrameHistory = new Dictionary<Vector3Int, VoteHistoryCell>(8192);
    private Dictionary<Vector3Int, VoteHistoryCell> _voteCurrentFrameHistory = new Dictionary<Vector3Int, VoteHistoryCell>(8192);
    private readonly Dictionary<Vector3Int, VoteHistoryCell> _voteCurrentFrameRawHistory = new Dictionary<Vector3Int, VoteHistoryCell>(8192);

    private struct VoteHistoryCell
    {
        public Vector3 PositionSum;
        public Vector3 NormalSum;
        public int Count;
    }
    private float _auditMaxPosePositionDelta;
    private float _auditMaxPoseAngleDelta;
    private float _auditMaxSnapshotLatencyMs;
    private readonly Dictionary<int, VoxelAuditLifecycle> _voxelAuditLifecycle = new Dictionary<int, VoxelAuditLifecycle>(4096);
    private readonly Dictionary<int, VoxelWriteProvenance> _voxelWriteProvenance = new Dictionary<int, VoxelWriteProvenance>(32768);
    private readonly Dictionary<int, int> _duplicateLayerCleanupEvidence = new Dictionary<int, int>();
    private readonly List<Vector2> _layerHoleTrendSamples = new List<Vector2>(24);
    private int _lastLayerHoleTrendIntegratedFrame = -1;
    private int _voxelWriteSequence;
    private float[] _hardAuditExpectedTsdf;
    private byte[] _hardAuditExpectedWeights;
    private StringBuilder _hardAuditRows;
    private StringBuilder _hardAuditSampleRows;
    private int _hardAuditCaptureIndex;
    private string _hardAuditStem;
    private string _lastHardAuditPath = "off";
    private readonly HashSet<int> _captureAuditVoxels = new HashSet<int>();
    private readonly HashSet<int> _auditSuspectMeshVoxels = new HashSet<int>();
    private readonly HashSet<int> _auditPureGeometryMeshVoxels = new HashSet<int>();
    private int _auditMeshAssociatedVoxelCount;
    private int _auditRecurrentAfterReplaceCount;
    private int _auditLikelyDepthInstabilityCount;
    private int _auditViewOrRegistrationCount;
    private int _auditSuspectAssociatedConflictVoxelCount;
    private int _auditSuspectVoxelsLinkedToConflictCount;
    private int _auditPureGeometryAssociatedConflictVoxelCount;
    private int _auditPureGeometryVoxelsLinkedToConflictCount;
    private int _auditLockedAboveCorrectableWeightCount;
    private int _auditLockedBandConflictCount;
    private int _auditLockedReplaceRecurrentCount;
    private int _auditLockedDepthUnstableCount;
    private int _auditLockedViewChangedCount;
    private int _auditLockedUnderSupportedCount;
    private int _auditLockedStableOldSurfaceCount;
    private int _auditFrameReferenceCount;
    private Vector3 _auditFrameReferenceVectorSum;
    private float _auditFrameReferenceMagnitudeSum;

    private struct VoxelAuditLifecycle
    {
        public int FirstConflictFrame;
        public int LastConflictFrame;
        public int ConflictCount;
        public int ReplaceCount;
        public int ConflictAfterReplaceCount;
        public int LastReplaceFrame;
        public int LastOldWeight;
        public float LastOldTsdf;
        public float LastSampleTsdf;
        public float LastResidual;
        public string LastCause;
        public string LastLockSubtype;
        public int LastCorrectionHits;
        public float LastBandConflictRatio;
        public float LastBandHighWeightConflictRatio;
        public float LastHistoryAgreement;
        public float LastHistoricalSurfaceDistanceMeters;
        public int MeshNearbyVertices;
        public bool SuspectMeshNearby;
        public bool PureGeometryMeshNearby;
        public int PriorWriteFrame;
        public int PriorWriteFrameGap;
        public float PriorCameraTravelMeters;
        public float PriorSurfaceShiftMeters;
        public float PriorCameraRotationDeltaDegrees;
        public float PriorRayAngleDeltaDegrees;
        public float PriorDepthDeltaMeters;
        public float PriorNormalAngleDeltaDegrees;
        public bool PriorWriteKnown;
    }

    private struct VoxelWriteProvenance
    {
        public int FirstFrame;
        public int Frame;
        public int Capture;
        public int Sample;
        public int PixelX;
        public int PixelY;
        public int WriteCount;
        public int WriteSequence;
        public int IntegrateCount;
        public int CarveCount;
        public int ReplaceCount;
        public int RepairCount;
        public string LastOperation;
        public Vector3 CameraPosition;
        public Quaternion CameraRotation;
        public Vector3 SurfacePoint;
        public Vector3 RayDirection;
        public Vector3 SurfaceNormal;
        public float SurfaceDepth;
        public float OldTsdf;
        public int OldWeight;
        public float Tsdf;
        public int Weight;
    }

    private void ResetStrongSampleSeedTemporaryBlockDiagnostics()
    {
        LastStrongSampleSeedTemporaryBlockedCount = 0;
        LastStrongSampleSeedTempNearSurfaceBlockCount = 0;
        LastStrongSampleSeedTempVoteBlockCount = 0;
        LastStrongSampleSeedTempScoreBlockCount = 0;
        LastStrongSampleSeedTempSupportBlockCount = 0;
        LastStrongSampleSeedTempBandBlockCount = 0;
        LastStrongSampleSeedTempSameFrameBlockCount = 0;
        LastStrongSampleSeedTempCrossFrameBlockCount = 0;
        LastStrongSampleSeedTempDirtyPendingBlockCount = 0;
        LastStrongSampleSeedTempOldWeightBlockCount = 0;
        LastStrongSampleSeedTempConflictBlockCount = 0;
        LastStrongSampleSeedTempLocalBlockCount = 0;
    }

    private void ResetProvisionalTsdfBlockDiagnostics()
    {
        LastProvisionalTsdfDisabledBlockedCount = 0;
        LastProvisionalTsdfInvalidBlockedCount = 0;
        LastProvisionalTsdfNearSurfaceBlockedCount = 0;
        LastProvisionalTsdfNearSurfacePositiveBlockedCount = 0;
        LastProvisionalTsdfNearSurfaceNegativeBlockedCount = 0;
        LastProvisionalTsdfFarBandSkippedCount = 0;
        LastProvisionalTsdfVoteBlockedCount = 0;
        LastProvisionalTsdfScoreBlockedCount = 0;
        LastProvisionalTsdfSupportRatioBlockedCount = 0;
        LastProvisionalTsdfBandBlockedCount = 0;
        LastProvisionalTsdfCleanHistoryBlockedCount = 0;
        LastProvisionalTsdfSameFrameBlockedCount = 0;
        LastProvisionalTsdfCrossFrameBlockedCount = 0;
        LastProvisionalTsdfDirtyPendingBlockedCount = 0;
        LastProvisionalTsdfOldWeightBlockedCount = 0;
        LastProvisionalTsdfConflictBlockedCount = 0;
        LastProvisionalTsdfPlaneBlockedCount = 0;
        LastProvisionalTsdfPendingStabilityBlockedCount = 0;
        LastProvisionalTsdfFormalDowngradeBlockedCount = 0;
        LastProvisionalTsdfExistingWeightBypassCount = 0;
        LastProvisionalTsdfBootstrapLocalBypassCount = 0;
        LastProvisionalPlaneCompatibilityPassCount = 0;
        LastProvisionalPlaneCompatibilityBlockedCount = 0;
        LastProvisionalPlaneCompatibilityNoReferenceCount = 0;
        LastProvisionalPlaneCompatibilityCandidateCount = 0;
        LastProvisionalPlaneCompatibilityNormalRejectedCount = 0;
        LastProvisionalPlaneCompatibilityDistanceRejectedCount = 0;
        LastProvisionalLocalSupportBlockedCount = 0;
    }

    private void RecordProvisionalTsdfBlock(string reason)
    {
        if (_auditStrongSampleSeedWrite)
            return;

        switch (reason)
        {
            case "disabled":
                LastProvisionalTsdfDisabledBlockedCount++;
                break;
            case "invalid":
                LastProvisionalTsdfInvalidBlockedCount++;
                break;
            case "near_surface":
                LastProvisionalTsdfNearSurfaceBlockedCount++;
                break;
            case "near_surface_positive":
                LastProvisionalTsdfNearSurfaceBlockedCount++;
                LastProvisionalTsdfNearSurfacePositiveBlockedCount++;
                break;
            case "near_surface_negative":
                LastProvisionalTsdfNearSurfaceBlockedCount++;
                LastProvisionalTsdfNearSurfaceNegativeBlockedCount++;
                break;
            case "vote":
                LastProvisionalTsdfVoteBlockedCount++;
                break;
            case "score":
                LastProvisionalTsdfScoreBlockedCount++;
                break;
            case "support":
                LastProvisionalTsdfSupportRatioBlockedCount++;
                break;
            case "band":
                LastProvisionalTsdfBandBlockedCount++;
                break;
            case "clean_history":
                LastProvisionalTsdfCleanHistoryBlockedCount++;
                break;
            case "same_frame":
                LastProvisionalTsdfSameFrameBlockedCount++;
                break;
            case "cross_frame":
                LastProvisionalTsdfCrossFrameBlockedCount++;
                break;
            case "dirty_pending":
                LastProvisionalTsdfDirtyPendingBlockedCount++;
                break;
            case "old_weight":
                LastProvisionalTsdfOldWeightBlockedCount++;
                break;
            case "conflict":
                LastProvisionalTsdfConflictBlockedCount++;
                break;
            case "plane":
                LastProvisionalTsdfPlaneBlockedCount++;
                break;
            case "local":
                LastProvisionalLocalSupportBlockedCount++;
                break;
        }
    }

    private void ResetFormalIntegrateGateDiagnostics()
    {
        LastFormalIntegrateWriteCount = 0;
        LastFormalIntegrateBlockedCount = 0;
        LastFormalIntegrateProvisionalCount = 0;
        LastFormalIntegrateBlockVoteCount = 0;
        LastFormalIntegrateBlockScoreCount = 0;
        LastFormalIntegrateBlockSupportCount = 0;
        LastFormalIntegrateBlockBandCount = 0;
        LastFormalIntegrateBlockResidualCount = 0;
        LastFormalIntegrateBlockCleanHistoryCount = 0;
        LastFormalIntegrateBlockWeakHistoryCount = 0;
        LastFormalIntegrateBlockTemporalDepthJumpCount = 0;
        LastFormalIntegrateBlockStrongCurrentHistoryCount = 0;
        LastFormalStrongCurrentLocalHistoryBypassCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockLocalCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockSupportCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockStableCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockAxialCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockResidualCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockPlaneCount = 0;
        LastFormalStrongCurrentLocalHistoryBlockDoubleLayerCount = 0;
        LastStrongCurrentProvisionalPromotedCount = 0;
        LastStrongCurrentProvisionalPromotionBlockedCount = 0;
        LastStrongCurrentProvisionalBlockDisabledCount = 0;
        LastStrongCurrentProvisionalBlockTagCount = 0;
        LastStrongCurrentProvisionalBlockStorageCount = 0;
        LastStrongCurrentProvisionalBlockInvalidCount = 0;
        LastStrongCurrentProvisionalBlockNoProvisionalCount = 0;
        LastStrongCurrentProvisionalBlockNoProvisionalNearSurfaceCount = 0;
        LastStrongCurrentProvisionalBlockNoProvisionalFarBandCount = 0;
        LastStrongCurrentProvisionalBlockHitsCount = 0;
        LastStrongCurrentProvisionalBlockWeightCount = 0;
        LastStrongCurrentProvisionalBlockBandHistoryCount = 0;
        LastStrongCurrentProvisionalBlockAgreementCount = 0;
        LastStrongCurrentProvisionalBlockCleanHistoryCount = 0;
        LastStrongCurrentProvisionalBlockDirtyPendingCount = 0;
        LastStrongCurrentProvisionalBlockSameFrameCount = 0;
        LastStrongCurrentProvisionalBlockCrossFrameCount = 0;
        LastStrongCurrentProvisionalBlockConflictCount = 0;
        LastStrongCurrentProvisionalBlockLocalCount = 0;
        LastStrongCurrentProvisionalBlockPlaneCount = 0;
        LastStrongCurrentPromotionLocalPassCount = 0;
        LastStrongCurrentPromotionLocalBlockCount = 0;
        LastStrongCurrentPromotionLocalNeighborCount = 0;
        LastStrongCurrentPromotionLocalProvisionalNeighborCount = 0;
        LastStrongCurrentPromotionLocalStableNeighborCount = 0;
        LastStrongCurrentPromotionLocalAxialNeighborCount = 0;
        LastStrongCurrentPromotionDoubleLayerCandidateCount = 0;
        LastStrongCurrentPromotionDoubleLayerBlockCount = 0;
        LastStrongCurrentPromotionDoubleLayerNormalRejectCount = 0;
        LastStrongCurrentPromotionDoubleLayerPlaneRejectCount = 0;
        LastStrongCurrentPromotionDoubleLayerLateralRejectCount = 0;
        LastStrongCurrentPromotionDoubleLayerDirtyNeighborCount = 0;
        LastStrongCurrentPromotionDoubleLayerIntegrateBlockCount = 0;
        LastStrongCurrentPromotionDoubleLayerReplaceBlockCount = 0;
        LastStrongCurrentPromotionDoubleLayerOtherBlockCount = 0;
        LastFormalIntegrateBlockSameFrameCount = 0;
        LastFormalIntegrateBlockCrossFrameCount = 0;
        LastFormalIntegrateBlockOldTsdfCount = 0;
        LastFormalIntegrateBlockDirtyPendingCount = 0;
    }

    private void ResetSuspectMeshSourceDiagnostics()
    {
        LastSuspectMeshSourceIntegrateVoxelCount = 0;
        LastSuspectMeshSourceProvisionalVoxelCount = 0;
        LastSuspectMeshSourceStrongSeedVoxelCount = 0;
        LastSuspectMeshSourceContinuityVoxelCount = 0;
        LastSuspectMeshSourceCarveVoxelCount = 0;
        LastSuspectMeshSourceReplaceVoxelCount = 0;
        LastSuspectMeshSourceRepairVoxelCount = 0;
        LastSuspectMeshSourceOtherVoxelCount = 0;
        LastSuspectMeshSourceUnknownVoxelCount = 0;
    }

    private void Awake()
    {
        ApplyStage03ADisplayPreset();
        ApplyContinuityBaseFillPreset();
        ResolveRefs();
        EnsureObjects();
    }

    private void OnEnable()
    {
        ApplyStage03ADisplayPreset();
        ApplyContinuityBaseFillPreset();
        ResolveRefs();
        EnsureObjects();
    }

    private void ApplyStage03ADisplayPreset()
    {
        if (!forceMeshGridDisplayOnEnable && !useStage03ACleanIsoSurface)
            return;

        showRawDepthDebugView = false;
        showRawCoverageGridOverlay = false;
        hideMeshWhenRawCoverageGridOverlay = false;
        showExtractedSurfaceMesh = true;
        showPlanarCoverSurfaceFill = true;
        showMeshEdgeWireOverlay = true;
        if (useStage03ACleanIsoSurface)
        {
            DisableDiagnosticDiskWrites();
            useAtomicObservationTsdfBands = true;
            useV08DirectBandWriteForDiagnosis = false;
            useV09ExactCornerEligibilityForDiagnosis = false;
            useLegacyTsdfMeshDisplay = useV09LegacyExtractorForStage03A;
            suppressExtractionNearPendingTsdfCorrection = false;
            suppressExtractionNearDirtyTsdfQuarantine = false;
            suppressQuadOnPendingTsdfCorrectionEdge = false;
            suppressQuadOnDirtyTsdfQuarantineEdge = false;
            requireMainShellPromotionGate = false;
            bridgeCleanCoplanarBoundaryEdges = false;
            pruneSmallExtractedMeshComponents = false;
            pruneDanglingExtractedMeshTriangles = false;
            pruneSpikePatchTriangles = false;
            showMeshTumorSuspectOverlay = false;
        }
    }

    private void DisableDiagnosticDiskWrites()
    {
        writeConfidenceAuditOnCapture = false;
        if (_contributionLedgerWriter != null)
        {
            _contributionLedgerWriter.Dispose();
            _contributionLedgerWriter = null;
        }
        _confidenceAuditRows = null;
        _confidenceAuditFrameRows = null;
        _doubleLayerPairRows = null;
    }

    private void ApplyContinuityBaseFillPreset()
    {
        if (!fillCleanTsdfContinuityGaps)
            return;

        tsdfContinuityFillRadiusVoxels = Mathf.Max(tsdfContinuityFillRadiusVoxels, 2);
        minTsdfContinuityFaceNeighborVoxels = Mathf.Max(minTsdfContinuityFaceNeighborVoxels, 2);
        minTsdfContinuitySameSignNeighborVoxels = Mathf.Min(minTsdfContinuitySameSignNeighborVoxels, 10);
        maxTsdfContinuityNeighborAbsValue = Mathf.Max(maxTsdfContinuityNeighborAbsValue, 0.24f);
        maxTsdfContinuityFillVoxelsPerRebuild = Mathf.Max(maxTsdfContinuityFillVoxelsPerRebuild, 4096);
        allowSameSignContinuityBaseFill = true;
        allowStableProvisionalContinuitySources = true;
        fillBoundaryNoTsdfGaps = true;
        minBoundaryNoTsdfNeighborVoxels = Mathf.Min(minBoundaryNoTsdfNeighborVoxels, 8);
        minBoundaryNoTsdfFaceNeighborVoxels = Mathf.Max(minBoundaryNoTsdfFaceNeighborVoxels, 3);
        allowBoundaryNoTsdfProvisionalAnchor = true;
        minBoundaryNoTsdfCleanFaceAnchors = Mathf.Clamp(minBoundaryNoTsdfCleanFaceAnchors, 0, 6);
        minBoundaryNoTsdfProvisionalAnchors = Mathf.Clamp(minBoundaryNoTsdfProvisionalAnchors, 2, 12);
        minBoundaryNoTsdfProvisionalFaceAnchors = Mathf.Clamp(minBoundaryNoTsdfProvisionalFaceAnchors, 1, 6);
        maxBoundaryNoTsdfFillVoxelsPerRebuild = Mathf.Max(maxBoundaryNoTsdfFillVoxelsPerRebuild, 1024);
    }

    private void OnDisable()
    {
        if (_confidenceAuditRows != null)
            EndConfidenceAuditCapture(_confidenceAuditIntegratedFrames, "component disabled");
    }

    private void OnDestroy()
    {
        if (_confidenceAuditRows != null)
            EndConfidenceAuditCapture(_confidenceAuditIntegratedFrames, "component destroyed");
        if (_mesh != null)
            Destroy(_mesh);
        if (_runtimeMaterial != null)
            Destroy(_runtimeMaterial);
        if (_meshObject != null)
            Destroy(_meshObject);
        if (_wireMesh != null)
            Destroy(_wireMesh);
        if (_wireRuntimeMaterial != null)
            Destroy(_wireRuntimeMaterial);
        if (_wireObject != null)
            Destroy(_wireObject);
        if (_dirtyEvidenceMesh != null)
            Destroy(_dirtyEvidenceMesh);
        if (_dirtyEvidenceMaterial != null)
            Destroy(_dirtyEvidenceMaterial);
        if (_dirtyEvidenceObject != null)
            Destroy(_dirtyEvidenceObject);
        if (_rawDepthDebugMesh != null)
            Destroy(_rawDepthDebugMesh);
        if (_rawDepthDebugMaterial != null)
            Destroy(_rawDepthDebugMaterial);
        if (_holeBoundaryDiagnosticMesh != null)
            Destroy(_holeBoundaryDiagnosticMesh);
        if (_holeBoundaryDiagnosticObject != null)
            Destroy(_holeBoundaryDiagnosticObject);
        if (_rawDepthDebugObject != null)
            Destroy(_rawDepthDebugObject);
        if (_hudRoot != null)
            Destroy(_hudRoot);
        if (_diagnosticHotspotRoot != null)
            Destroy(_diagnosticHotspotRoot);
        if (_diagnosticHotspotMaterial != null)
            Destroy(_diagnosticHotspotMaterial);
        if (_diagnosticSecondaryHotspotMaterial != null)
            Destroy(_diagnosticSecondaryHotspotMaterial);
    }

    private void LateUpdate()
    {
        UpdateDiagnosticHud();
    }

    [ContextMenu("Prepare TSDF Single Shell Route")]
    public void PrepareRoute()
    {
        ResolveRefs();
        EnsureObjects();
        if (rawDepthSource == null)
            return;

        rawDepthSource.enabled = true;
        SuppressSourcePreview();
    }

    [ContextMenu("Capture Raw Snapshot And Integrate")]
    public void CaptureRawSnapshotAndIntegrate()
    {
        ResolveRefs();
        if (_captureRoutine != null)
            StopCoroutine(_captureRoutine);
        _captureRoutine = StartCoroutine(CaptureRawSnapshotAndIntegrateRoutine());
    }

    [ContextMenu("Integrate Latest Raw Snapshot")]
    public bool IntegrateLatestRawSnapshot()
    {
        ResolveRefs();
        if (rawDepthSource == null)
            return Warn("Raw depth source is missing.");

        if (!rawDepthSource.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot))
            return Warn("No latest raw depth frame snapshot is available.");

        bool integrated = IntegrateSnapshot(snapshot);
        if (integrated && rebuildMeshAfterEachCapture)
            RebuildMesh();
        return integrated;
    }

    [ContextMenu("Clear TSDF Shell")]
    public void ClearShell()
    {
        if (_captureRoutine != null)
        {
            StopCoroutine(_captureRoutine);
            _captureRoutine = null;
        }

        _volumeInitialized = false;
        _tsdf = null;
        _weights = null;
        _atomicProvisionalBandTsdf = null;
        _atomicProvisionalBandWeight = null;
        _atomicProvisionalBandRetiredLastFrame = null;
        _atomicProvisionalBandRetiredSign = null;
        _surfaceBandVisitFlags = null;
        _surfaceBandAcceptSignLostFlags = null;
        _acceptedPositiveLastSequence = null;
        _acceptedNegativeLastSequence = null;
        _acceptedPositiveRetainedLastSequence = null;
        _acceptedNegativeRetainedLastSequence = null;
        _acceptedWriteSequence = 0;
        _hardAuditExpectedTsdf = null;
        _hardAuditExpectedWeights = null;
        ResetHardTsdfWriteAuditCounters();
        _surfaceObservationState = null;
        _surfaceObservationLastFrame = null;
        _clearedVoxelLastFrame = null;
        _pendingTsdf = null;
        _pendingTsdfHits = null;
        _pendingTsdfLastFrame = null;
        _correctionTsdf = null;
        _correctionTsdfHits = null;
        _correctionTsdfLastFrame = null;
        _dirtyTsdfLastFrame = null;
        _provisionalTsdf = null;
        _provisionalTsdfLastFrame = null;
        _provisionalTsdfHits = null;
        _oldCleanConflictHits = null;
        _oldCleanConflictLastFrame = null;
        _freeSpaceEvidenceHits = null;
        _freeSpaceEvidenceLastFrame = null;
        _cellVertexIndices = null;
        _detailCandidateCellHits = null;
        _detailCandidateCellLastFrame = null;
        _fallbackSurfacePoints.Clear();
        _fallbackSurfaceNormals.Clear();
        _fallbackSurfaceVoxelKeys.Clear();
        _fallbackSurfaceVoxels.Clear();
        _surfaceChunks.Clear();
        _voxelAuditLifecycle.Clear();
        _voxelWriteProvenance.Clear();
        _voxelWriteSequence = 0;
        _captureAuditVoxels.Clear();
        _surfaceChunkOrder.Clear();
        _dirtyTsdfEvidence.Clear();
        if (_dirtyEvidenceMesh != null)
            _dirtyEvidenceMesh.Clear();
        if (_dirtyEvidenceObject != null)
            _dirtyEvidenceObject.SetActive(false);
        if (_rawDepthDebugMesh != null)
            _rawDepthDebugMesh.Clear();
        if (_rawDepthDebugObject != null)
            _rawDepthDebugObject.SetActive(false);
        if (_holeBoundaryDiagnosticMesh != null)
            _holeBoundaryDiagnosticMesh.Clear();
        if (_holeBoundaryDiagnosticObject != null)
            _holeBoundaryDiagnosticObject.SetActive(false);
        ClearRawDepthDebugCache();
        ResetRawDepthDebugCounters();
        _fallbackSurfaceReplaceCursor = 0;
        LastSurfaceChunkCount = 0;
        LastEvictedSurfaceChunkCount = 0;
        LastEvictedSurfaceCellCount = 0;
        TotalEvictedSurfaceChunkCount = 0;
        TotalEvictedSurfaceCellCount = 0;
        LastSurfaceCellConflictCount = 0;
        LastProvisionalTsdfSupportWriteCount = 0;
        LastProvisionalTsdfSupportBlockedCount = 0;
        ResetProvisionalTsdfBlockDiagnostics();
        LastStrongSampleSeedWriteCount = 0;
        LastStrongSampleSeedBlockedCount = 0;
        ResetStrongSampleSeedTemporaryBlockDiagnostics();
        ResetFormalIntegrateGateDiagnostics();
        LastProvisionalLocalSupportPassCount = 0;
        LastProvisionalLocalSupportNeighborCount = 0;
        LastProvisionalLocalSupportStableNeighborCount = 0;
        LastProvisionalLocalSupportAxialNeighborCount = 0;
        LastProvisionalTsdfConfirmedCount = 0;
        LastProvisionalTsdfConfirmedByWeightCount = 0;
        LastProvisionalTsdfRetiredCount = 0;
        LastAtomicAcceptedBandVoxelWriteCount = 0;
        LastAtomicProvisionalBandVoxelWriteCount = 0;
        LastAtomicAcceptDuplicateCandidateCount = 0;
        LastAtomicAcceptDuplicateDowngradeCount = 0;
        LastAtomicAcceptDuplicateSameSurfaceCount = 0;
        LastAtomicPromotedProvisionalVoxelCount = 0;
        LastAtomicRetiredProvisionalVoxelCount = 0;
        LastHoleBoundaryWaitingCount = 0;
        LastHoleBoundaryRetiredCount = 0;
        LastHoleBoundaryNoBandCount = 0;
        LastHoleBoundaryPositiveOnlyCount = 0;
        LastHoleBoundaryNegativeOnlyCount = 0;
        LastHoleBoundarySpatialMissCount = 0;
        LastHoleBoundaryWeakCornersCount = 0;
        LastHoleBoundaryRenderedMarkerCount = 0;
        LastProvisionalTsdfRetiredExpiredCount = 0;
        LastProvisionalTsdfDirtyClearedCount = 0;
        LastOldCleanMetabolismWatchCount = 0;
        LastOldCleanMetabolismDecayCount = 0;
        LastOldCleanMetabolismClearCount = 0;
        LastOldCleanMetabolismBlockedCount = 0;
        LastOldCleanMetabolismCandidateCount = 0;
        LastOldCleanMetabolismWaitingHitsCount = 0;
        LastOldCleanMetabolismBlockedSupportCount = 0;
        LastOldCleanMetabolismBlockedSameFrameCount = 0;
        LastOldCleanMetabolismBlockedWeightCount = 0;
        LastOldCleanMetabolismBlockedResidualCount = 0;
        LastOldCleanMetabolismBlockedDirtyPendingCount = 0;
        LastOldCleanMetabolismSkippedWeakBandCount = 0;
        LastOldCleanMetabolismSkippedWeakCrossFrameCount = 0;
        LastPendingTsdfCorrectionCount = 0;
        LastCorrectedTsdfCount = 0;
        LastReplacedDirtyTsdfCount = 0;
        LastGuardedDirtyTsdfReplaceCount = 0;
        LastDirtyTsdfBandRepairCount = 0;
        LastDirtyTsdfBandRepairSampleCount = 0;
        LastDirtyTsdfBandRepairTriggerCount = 0;
        LastDirtyTsdfBandRepairProbeCount = 0;
        LastDirtyTsdfBandRepairBlockedDisabledCount = 0;
        LastDirtyTsdfBandRepairBlockedNoSampleCount = 0;
        LastDirtyTsdfBandRepairBlockedNoHistoryCount = 0;
        LastDirtyTsdfBandRepairBlockedLowConflictCount = 0;
        LastDirtyTsdfBandRepairBlockedBudgetCount = 0;
        LastDirtyTsdfBandRepairBlockedOutsideCount = 0;
        LastDirtyTsdfBandRepairBlockedEmptyCount = 0;
        LastDirtyTsdfBandRepairBlockedWeightCount = 0;
        LastDirtyTsdfBandRepairBlockedSameSignCount = 0;
        LastDirtyTsdfBandRepairBlockedResidualCount = 0;
        LastCleanedDirtyTsdfNeighborCount = 0;
        LastDirtyTsdfQuarantineCount = 0;
        ResetDirtyTsdfDiagnostics();
        LastRejectedTsdfDepthSupportCount = 0;
        LastRejectedRobustDepthCount = 0;
        LastRobustDepthDownweightedCount = 0;
        LastRobustDepthCorrectedCount = 0;
        LastRejectedDepthEdgeErosionCount = 0;
        LastDepthEdgeDownweightedCount = 0;
        LastRejectedFallbackWeakCount = 0;
        LastCarvedFreeSpaceVoxelCount = 0;
        ResetFreeSpaceEvidenceDiagnostics();
        LastRejectedBadMeshRebuildCount = 0;
        LastHeldDisplayTriangleCount = 0;
        LastCommittedMeshBlockCount = 0;
        LastCandidateMeshBlockCount = 0;
        LastRetainedCommittedMeshBlockCount = 0;
        LastCommittedMeshHoldSpatialCount = 0;
        LastCommittedMeshHoldTriangleCount = 0;
        LastCommittedMeshGrowthTriangleCount = 0;
        _committedMeshBlocks.Clear();
        _candidateMeshBlocks.Clear();
        ResetExtractionDiagnostics();
        IntegratedFrameCount = 0;
        LastRawFrameIndex = -1;
        LastInputSampleCount = 0;
        LastIntegratedSampleCount = 0;
        LastUpdatedVoxelCount = 0;
        LastPendingStableTsdfCount = 0;
        LastMeshVertexCount = 0;
        LastMeshTriangleCount = 0;
        LastRenderedFallbackSampleCount = 0;
        LastPlanarCoverTileCount = 0;
        LastSkippedDirtyLightCoverCellCount = 0;
        LastBlockedDirtyLightCoverSampleCount = 0;
        TotalBlockedDirtyLightCoverSampleCount = 0;
        LastSkippedUnstableLightCoverCellCount = 0;
        LastCleanLightCoverCellCount = 0;
        LastUnstableLightCoverCellCount = 0;
        LastRiskLightCoverCellCount = 0;
        LastWireSegmentCount = 0;
        LastPrunedMeshComponentCount = 0;
        LastPrunedMeshTriangleCount = 0;
        LastPrunedComponentTriangleCount = 0;
        LastPrunedDanglingTriangleCount = 0;
        LastPrunedSpikeTriangleCount = 0;
        LastMainShellPromotedCellCount = 0;
        LastCandidateShellHeldCellCount = 0;
        LastDetailCandidateHeldCellCount = 0;
        LastDetailCandidatePromotedCellCount = 0;

        if (_mesh != null)
            _mesh.Clear();
        if (_wireMesh != null)
            _wireMesh.Clear();
    }

    [ContextMenu("Rebuild Mesh From TSDF")]
    public bool RebuildMesh()
    {
        EnsureObjects();
        int previousVertexCount = LastMeshVertexCount;
        int previousTriangleCount = LastMeshTriangleCount;
        LastMeshUsedExtractedTsdf = false;
        LastPrunedMeshComponentCount = 0;
        LastPrunedMeshTriangleCount = 0;
        LastPrunedComponentTriangleCount = 0;
        LastPrunedDanglingTriangleCount = 0;
        LastPrunedSpikeTriangleCount = 0;
        LastMainShellPromotedCellCount = 0;
        LastCandidateShellHeldCellCount = 0;
        LastDetailCandidateHeldCellCount = 0;
        LastDetailCandidatePromotedCellCount = 0;

        if (!_volumeInitialized || _tsdf == null || _weights == null)
            return Warn("TSDF volume is empty.");

        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);
        if (cellX <= 0 || cellY <= 0 || cellZ <= 0)
            return Warn("TSDF volume dimensions are invalid.");

        int cellCount = cellX * cellY * cellZ;
        if (_cellVertexIndices == null || _cellVertexIndices.Length != cellCount)
            _cellVertexIndices = new int[cellCount];
        if (_detailCandidateCellHits == null || _detailCandidateCellHits.Length != cellCount)
            _detailCandidateCellHits = new byte[cellCount];
        if (_detailCandidateCellLastFrame == null || _detailCandidateCellLastFrame.Length != cellCount)
        {
            _detailCandidateCellLastFrame = new int[cellCount];
            for (int i = 0; i < _detailCandidateCellLastFrame.Length; i++)
                _detailCandidateCellLastFrame[i] = int.MinValue;
        }
        for (int i = 0; i < _cellVertexIndices.Length; i++)
            _cellVertexIndices[i] = -1;

        List<Vector3> vertices = new List<Vector3>(Mathf.Min(maxSurfaceVertices, cellCount));
        List<int> triangles = new List<int>(Mathf.Min(maxSurfaceVertices * 3, cellCount * 6));
        List<Color> colors = new List<Color>(Mathf.Min(maxSurfaceVertices, cellCount));
        ResetExtractionDiagnostics();
        LastHoleSideRepairCandidateCount = 0;
        LastHoleSideRepairAppliedCount = 0;
        LastHoleSideRepairBlockedSupportCount = 0;
        LastHoleSideRepairBlockedDirtyCount = 0;
        LastHoleSideRepairBlockedMultiZeroCount = 0;
        LastHoleSideRepairBlockedPlaneCount = 0;
        LastHoleSideRepairPlaneBandVoxelCount = 0;
        LastHoleSideRepairPlaneConfirmedCount = _pendingPrimaryPlaneHoleAcceptConfirmedCount;
        _pendingPrimaryPlaneHoleAcceptConfirmedCount = 0;
        LastHoleSideRepairPlaneRetiredCount = 0;
        LastHoleSideRepairRetiredNoNearAcceptCount = 0;
        LastHoleSideRepairRetiredSignMismatchCount = 0;
        LastHoleSideRepairRetiredPlaneMismatchCount = 0;
        LastHoleSideRepairRetiredTsdfDeltaCount = 0;
        LastHoleSideRepairRetiredExpiredAfterPassCount = 0;
        LastHoleSideRepairPlaneDistance0ToHalfCount = 0;
        LastHoleSideRepairPlaneDistanceHalfTo075Count = 0;
        LastHoleSideRepairPlaneDistance075To1Count = 0;
        LastHoleSideRepairPlaneDistance1To15Count = 0;
        LastHoleSideRepairPlaneDistance15To2Count = 0;
        LastHoleSideRepairPlaneDistanceOver2Count = 0;
        LastDuplicateLayerCleanupQueuedCount = _duplicateLayerCleanupEvidence.Count;
        LastDuplicateLayerCleanupDecayedCount = 0;
        LastDuplicateLayerCleanupClearedCount = 0;
        LastDuplicateLayerCleanupBoostedCount = 0;
        LastDuplicateLayerCleanupMaxEvidence = 0;
        AuditUntrackedTsdfWrites("integration");
        ConfirmPrimaryPlaneHoleBands();
        RetireExpiredProvisionalTsdf();
        AuditUntrackedTsdfWrites("retire");
        CleanupConfirmedDuplicateLayers();
        FillCleanTsdfContinuityGaps();
        RepairGuardedHoleSides();
        AuditUntrackedTsdfWrites("continuity");

        Vector3[] cornerPositions = new Vector3[8];
        float[] cornerValues = new float[8];
        byte[] cornerWeights = new byte[8];

        if (showExtractedSurfaceMesh)
        {
            for (int z = 0; z < cellZ; z++)
            {
                for (int y = 0; y < cellY; y++)
                {
                    for (int x = 0; x < cellX; x++)
                    {
                        if (vertices.Count >= maxSurfaceVertices)
                            break;

                        LastSurfaceCellScanCount++;
                        if (!TryBuildCellVertex(x, y, z, cornerPositions, cornerValues, cornerWeights, out Vector3 vertex))
                            continue;

                        int vertexIndex = vertices.Count;
                        vertices.Add(vertex);
                        colors.Add(cleanCoverCellColor);
                        _cellVertexIndices[CellIndex(x, y, z, cellX, cellY)] = vertexIndex;
                        LastBuiltSurfaceCellVertexCount++;
                    }
                }
            }

            for (int z = 0; z < _dimZ; z++)
            {
                for (int y = 0; y < _dimY; y++)
                {
                    for (int x = 0; x < _dimX; x++)
                    {
                        int xIndex = Index(x, y, z);
                        if (UseLegacyMeshExtraction)
                        {
                            if (x + 1 < _dimX && IsLegacySignChange(xIndex, Index(x + 1, y, z)))
                            {
                                LastSurfaceQuadCandidateCount++;
                                AddQuadAroundXEdge(x, y, z, cellX, cellY, vertices, triangles);
                            }
                            if (y + 1 < _dimY && IsLegacySignChange(xIndex, Index(x, y + 1, z)))
                            {
                                LastSurfaceQuadCandidateCount++;
                                AddQuadAroundYEdge(x, y, z, cellX, cellY, vertices, triangles);
                            }
                            if (z + 1 < _dimZ && IsLegacySignChange(xIndex, Index(x, y, z + 1)))
                            {
                                LastSurfaceQuadCandidateCount++;
                                AddQuadAroundZEdge(x, y, z, cellX, cellY, vertices, triangles);
                            }
                        }
                        else
                        {
                            SurfaceEdgeKind xEdge = x + 1 < _dimX ? GetSurfaceEdgeKind(xIndex, Index(x + 1, y, z)) : SurfaceEdgeKind.None;
                            if (xEdge != SurfaceEdgeKind.None)
                                LastSurfaceQuadCandidateCount++;
                            if (xEdge != SurfaceEdgeKind.None && PassSurfaceQuadExtractionGate(xEdge, x, y, z, xIndex, Index(x + 1, y, z)))
                                AddQuadAroundXEdge(x, y, z, cellX, cellY, vertices, triangles);
                            SurfaceEdgeKind yEdge = y + 1 < _dimY ? GetSurfaceEdgeKind(xIndex, Index(x, y + 1, z)) : SurfaceEdgeKind.None;
                            if (yEdge != SurfaceEdgeKind.None)
                                LastSurfaceQuadCandidateCount++;
                            if (yEdge != SurfaceEdgeKind.None && PassSurfaceQuadExtractionGate(yEdge, x, y, z, xIndex, Index(x, y + 1, z)))
                                AddQuadAroundYEdge(x, y, z, cellX, cellY, vertices, triangles);
                            SurfaceEdgeKind zEdge = z + 1 < _dimZ ? GetSurfaceEdgeKind(xIndex, Index(x, y, z + 1)) : SurfaceEdgeKind.None;
                            if (zEdge != SurfaceEdgeKind.None)
                                LastSurfaceQuadCandidateCount++;
                            if (zEdge != SurfaceEdgeKind.None && PassSurfaceQuadExtractionGate(zEdge, x, y, z, xIndex, Index(x, y, z + 1)))
                                AddQuadAroundZEdge(x, y, z, cellX, cellY, vertices, triangles);
                        }
                    }
                }
            }
            RebuildHoleBoundaryDiagnostic(cellX, cellY, cellZ, vertices.Count, triangles);
        }

        bool mainShellHeld = requireMainShellPromotionGate && (LastCandidateShellHeldCellCount + LastDetailCandidateHeldCellCount) > 0;
        if (useFallbackMeshWhenTsdfExtractionEmpty &&
            (!showExtractedSurfaceMesh || vertices.Count == 0 || triangles.Count == 0) &&
            (!mainShellHeld || showFallbackWhenMainShellHeld))
        {
            vertices.Clear();
            triangles.Clear();
            colors.Clear();
            BuildFallbackSurfaceSampleMesh(vertices, triangles, colors);
        }
        else
        {
            LastMeshUsedExtractedTsdf = true;
            LastPrePruneMeshTriangleCount = triangles.Count / 3;
            if (UseLegacyMeshExtraction)
            {
                LastPostComponentMeshTriangleCount = triangles.Count / 3;
                LastPostBridgeMeshTriangleCount = triangles.Count / 3;
            }
            else
            {
                PruneSmallExtractedMeshComponents(vertices, triangles, colors);
                LastPostComponentMeshTriangleCount = triangles.Count / 3;
                BridgeCleanCoplanarBoundaryEdges(vertices, triangles, colors);
                LastPostBridgeMeshTriangleCount = triangles.Count / 3;
                PruneDanglingExtractedMeshTriangles(vertices, triangles);
                PruneSpikePatchTriangles(vertices, triangles);
            }
        }

        AnalyzeMeshDiagnostics(vertices, triangles);
        ApplyMeshTumorSuspectOverlay(vertices, triangles, colors);
        AuditUntrackedTsdfWrites("extraction");

        if (ShouldKeepDisplayedMesh(previousTriangleCount, vertices, triangles.Count / 3))
        {
            // A candidate rejected by the stability gate must stay atomic.  Appending only its
            // new blocks to the old mesh creates unvalidated seams at the frozen boundary.
            int absorbedTriangles = 0;
            LastCommittedMeshGrowthTriangleCount = absorbedTriangles;
            LastMeshVertexCount = _mesh != null ? _mesh.vertexCount : previousVertexCount;
            LastMeshTriangleCount = _mesh != null ? _mesh.triangles.Length / 3 : previousTriangleCount;
            LastHeldDisplayTriangleCount = LastMeshTriangleCount;
            LastRejectedBadMeshRebuildCount++;
            if (debugLog)
            {
                Debug.LogWarning(
                    $"[ScanCoverTsdfSingleShellPrototype] Rejected weaker mesh rebuild: newVerts={vertices.Count} newTris={triangles.Count / 3} previousTris={previousTriangleCount} pruned={LastPrunedMeshTriangleCount}",
                    this);
            }
            return previousTriangleCount > 0;
        }

        _mesh.Clear();
        _mesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _mesh.SetVertices(vertices);
        if (colors.Count == vertices.Count)
            _mesh.SetColors(colors);
        _mesh.SetTriangles(triangles, 0, true);
        _mesh.RecalculateNormals();
        _mesh.RecalculateBounds();
        CommitDisplayedMeshBlocks(vertices);

        LastMeshVertexCount = vertices.Count;
        LastMeshTriangleCount = triangles.Count / 3;
        RebuildWireOverlay(vertices, triangles);
        RebuildRawDepthDebugMesh();
        ApplyRawDepthDebugDisplayMode();

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverTsdfSingleShellPrototype] Mesh rebuilt: vertices={LastMeshVertexCount} triangles={LastMeshTriangleCount} fallbackSamples={_fallbackSurfacePoints.Count} frames={IntegratedFrameCount} voxel={voxelSizeMeters:F3} trunc={truncationMeters:F3}",
                this);
        }

        return vertices.Count > 0;
    }

    private bool ShouldKeepDisplayedMesh(int previousTriangleCount, List<Vector3> candidateVertices, int newTriangleCount)
    {
        if (previousTriangleCount <= 0)
            return false;
        int newVertexCount = candidateVertices != null ? candidateVertices.Count : 0;
        if (newVertexCount <= 0 || newTriangleCount <= 0)
            return true;

        if (guardCommittedMeshSpatialCoverage)
        {
            EnsureCommittedMeshBlocksFromDisplayedMesh();
            BuildMeshBlockSet(candidateVertices, _candidateMeshBlocks);
            LastCommittedMeshBlockCount = _committedMeshBlocks.Count;
            LastCandidateMeshBlockCount = _candidateMeshBlocks.Count;
            LastRetainedCommittedMeshBlockCount = CountRetainedCommittedMeshBlocks();
            if (_committedMeshBlocks.Count > 0)
            {
                float retainedRatio = (float)LastRetainedCommittedMeshBlockCount / _committedMeshBlocks.Count;
                int newBlockCount = Mathf.Max(0, _candidateMeshBlocks.Count - LastRetainedCommittedMeshBlockCount);
                bool extendsObservedArea = newBlockCount >= Mathf.Max(4, _committedMeshBlocks.Count / 50);
                float requiredBlockRetention = useStrictCommittedMeshRetention
                    ? Mathf.Min(0.92f, Mathf.Clamp01(minCommittedMeshBlockRetention))
                    : Mathf.Clamp01(minCommittedMeshBlockRetention);
                float growthRetentionFloor = Mathf.Min(requiredBlockRetention, 0.82f);
                if (retainedRatio < requiredBlockRetention &&
                    (!extendsObservedArea || retainedRatio < growthRetentionFloor))
                {
                    LastCommittedMeshHoldSpatialCount++;
                    return true;
                }
            }
        }

        if (useStrictCommittedMeshRetention &&
            newTriangleCount < previousTriangleCount * Mathf.Min(0.92f, Mathf.Clamp(minCommittedMeshTriangleRetention, 0.8f, 1f)))
        {
            LastCommittedMeshHoldTriangleCount++;
            return true;
        }

        if (protectDisplayedMeshFromBadRebuild)
        {
            float minRatio = Mathf.Clamp(minAcceptedRebuildTriangleRatio, 0.1f, 1f);
            if (newTriangleCount < previousTriangleCount * minRatio)
                return true;

            int candidateTriangleBudget = newTriangleCount + LastPrunedMeshTriangleCount;
            if (candidateTriangleBudget > 0)
            {
                float prunedRatio = (float)LastPrunedMeshTriangleCount / candidateTriangleBudget;
                if (prunedRatio > Mathf.Clamp01(maxAcceptedPrunedTriangleRatio) &&
                    newTriangleCount < previousTriangleCount)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private void EnsureCommittedMeshBlocksFromDisplayedMesh()
    {
        if (_committedMeshBlocks.Count > 0 || _mesh == null || _mesh.vertexCount <= 0)
            return;
        BuildMeshBlockSet(_mesh.vertices, _committedMeshBlocks);
    }

    private void CommitDisplayedMeshBlocks(IList<Vector3> vertices)
    {
        BuildMeshBlockSet(vertices, _committedMeshBlocks);
        LastCommittedMeshBlockCount = _committedMeshBlocks.Count;
        LastCandidateMeshBlockCount = _committedMeshBlocks.Count;
        LastRetainedCommittedMeshBlockCount = _committedMeshBlocks.Count;
    }

    private void BuildMeshBlockSet(IList<Vector3> vertices, HashSet<Vector3Int> destination)
    {
        destination.Clear();
        if (vertices == null)
            return;
        int blockVoxels = useStrictCommittedMeshRetention
            ? Mathf.Min(2, Mathf.Clamp(committedMeshBlockSizeVoxels, 1, 8))
            : Mathf.Clamp(committedMeshBlockSizeVoxels, 1, 8);
        float blockSize = Mathf.Max(0.001f, voxelSizeMeters * blockVoxels);
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 local = (vertices[i] - _volumeOriginWorld) / blockSize;
            destination.Add(new Vector3Int(
                Mathf.FloorToInt(local.x),
                Mathf.FloorToInt(local.y),
                Mathf.FloorToInt(local.z)));
        }
    }

    private int CountRetainedCommittedMeshBlocks()
    {
        int retained = 0;
        foreach (Vector3Int block in _committedMeshBlocks)
        {
            if (_candidateMeshBlocks.Contains(block))
                retained++;
        }
        return retained;
    }

    private int AbsorbNewBlocksFromHeldCandidate(
        List<Vector3> candidateVertices,
        List<int> candidateTriangles,
        List<Color> candidateColors)
    {
        if (_mesh == null || candidateVertices == null || candidateTriangles == null || candidateTriangles.Count < 3)
            return 0;

        EnsureCommittedMeshBlocksFromDisplayedMesh();
        List<Vector3> mergedVertices = new List<Vector3>(_mesh.vertexCount + 1024);
        List<int> mergedTriangles = new List<int>(_mesh.triangles.Length + 3072);
        List<Color> mergedColors = new List<Color>(_mesh.vertexCount + 1024);
        _mesh.GetVertices(mergedVertices);
        _mesh.GetTriangles(mergedTriangles, 0);
        _mesh.GetColors(mergedColors);
        while (mergedColors.Count < mergedVertices.Count)
            mergedColors.Add(Color.white);

        Dictionary<int, int> remappedVertices = new Dictionary<int, int>();
        int addedTriangles = 0;
        int triangleBudget = Mathf.Clamp(maxHeldCandidateTrianglesToAbsorb, 256, 20000);
        for (int i = 0; i + 2 < candidateTriangles.Count && addedTriangles < triangleBudget; i += 3)
        {
            int ia = candidateTriangles[i];
            int ib = candidateTriangles[i + 1];
            int ic = candidateTriangles[i + 2];
            if (ia < 0 || ib < 0 || ic < 0 ||
                ia >= candidateVertices.Count || ib >= candidateVertices.Count || ic >= candidateVertices.Count)
            {
                continue;
            }

            Vector3 centroid = (candidateVertices[ia] + candidateVertices[ib] + candidateVertices[ic]) / 3f;
            if (_committedMeshBlocks.Contains(MeshBlockForPoint(centroid)))
                continue;
            if (mergedVertices.Count + 3 > maxSurfaceVertices)
                break;

            mergedTriangles.Add(RemapCandidateVertex(ia, candidateVertices, candidateColors, mergedVertices, mergedColors, remappedVertices));
            mergedTriangles.Add(RemapCandidateVertex(ib, candidateVertices, candidateColors, mergedVertices, mergedColors, remappedVertices));
            mergedTriangles.Add(RemapCandidateVertex(ic, candidateVertices, candidateColors, mergedVertices, mergedColors, remappedVertices));
            addedTriangles++;
        }

        if (addedTriangles <= 0)
            return 0;

        _mesh.Clear();
        _mesh.indexFormat = mergedVertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _mesh.SetVertices(mergedVertices);
        _mesh.SetColors(mergedColors);
        _mesh.SetTriangles(mergedTriangles, 0, true);
        _mesh.RecalculateNormals();
        _mesh.RecalculateBounds();
        CommitDisplayedMeshBlocks(mergedVertices);
        RebuildWireOverlay(mergedVertices, mergedTriangles);
        return addedTriangles;
    }

    private int RemapCandidateVertex(
        int candidateIndex,
        List<Vector3> candidateVertices,
        List<Color> candidateColors,
        List<Vector3> mergedVertices,
        List<Color> mergedColors,
        Dictionary<int, int> remappedVertices)
    {
        if (remappedVertices.TryGetValue(candidateIndex, out int mergedIndex))
            return mergedIndex;
        mergedIndex = mergedVertices.Count;
        remappedVertices[candidateIndex] = mergedIndex;
        mergedVertices.Add(candidateVertices[candidateIndex]);
        mergedColors.Add(candidateColors != null && candidateIndex < candidateColors.Count
            ? candidateColors[candidateIndex]
            : Color.white);
        return mergedIndex;
    }

    private Vector3Int MeshBlockForPoint(Vector3 point)
    {
        int blockVoxels = useStrictCommittedMeshRetention
            ? Mathf.Min(2, Mathf.Clamp(committedMeshBlockSizeVoxels, 1, 8))
            : Mathf.Clamp(committedMeshBlockSizeVoxels, 1, 8);
        float blockSize = Mathf.Max(0.001f, voxelSizeMeters * blockVoxels);
        Vector3 local = (point - _volumeOriginWorld) / blockSize;
        return new Vector3Int(
            Mathf.FloorToInt(local.x),
            Mathf.FloorToInt(local.y),
            Mathf.FloorToInt(local.z));
    }

    private IEnumerator CaptureRawSnapshotAndIntegrateRoutine()
    {
        ResolveRefs();
        EnsureObjects();
        BeginConfidenceAuditCapture();

        if (rawDepthSource == null)
        {
            Warn("Raw depth source is missing.");
            EndConfidenceAuditCapture(0, "raw depth source missing");
            _captureRoutine = null;
            yield break;
        }

        rawDepthSource.enabled = true;
        SuppressSourcePreview();

        int lastAcceptedFrame = int.MinValue;
        int integratedFrames = 0;
        float burstStart = Time.unscaledTime;
        float burstDuration = Mathf.Max(0.05f, fusionBurstDurationSeconds);
        int maxFrames = Mathf.Max(1, maxFusionFramesPerTrigger);

        while (Time.unscaledTime - burstStart < burstDuration && integratedFrames < maxFrames)
        {
            if (forceRawRefreshOnSnapshot && !rawDepthSource.HasPendingReadback)
                rawDepthSource.RefreshNow(forcePreprocessorRefresh: true);

            float frameStart = Time.unscaledTime;
            bool integratedThisFrame = false;
            while (Time.unscaledTime - frameStart < waitForRawFrameTimeoutSeconds)
            {
                if (!rawDepthSource.HasPendingReadback &&
                    rawDepthSource.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot ready) &&
                    ready.worldPositions != null &&
                    ready.worldPositions.Length > 0 &&
                    ready.frameIndex != lastAcceptedFrame)
                {
                    integratedThisFrame = IntegrateSnapshot(ready);
                    // A fully gated frame is still consumed; do not repeatedly reprocess it.
                    lastAcceptedFrame = ready.frameIndex;
                    if (integratedThisFrame)
                    {
                        integratedFrames++;
                    }
                    if (integratedThisFrame && rebuildMeshAfterEachCapture && !rebuildMeshAfterFusionBurstOnly)
                        RebuildMesh();
                    break;
                }

                yield return null;
            }

            if (!integratedThisFrame)
                yield return null;
        }

        if (integratedFrames <= 0)
        {
            bool integratedLatest = IntegrateLatestRawSnapshot();
            if (integratedLatest)
                integratedFrames = 1;
        }

        if (rebuildMeshAfterEachCapture && (rebuildMeshAfterFusionBurstOnly || integratedFrames <= 0))
            RebuildMesh();

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverTsdfSingleShellPrototype] Fusion trigger complete: integratedFrames={integratedFrames} totalFrames={IntegratedFrameCount} fallbackSamples={_fallbackSurfacePoints.Count} vertices={LastMeshVertexCount} triangles={LastMeshTriangleCount}",
                this);
        }

        EndConfidenceAuditCapture(integratedFrames, integratedFrames > 0 ? "complete" : "no integrated frame");
        _captureRoutine = null;
    }

    private bool IntegrateSnapshot(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot)
    {
        if (snapshot == null ||
            snapshot.worldPositions == null ||
            snapshot.worldPositions.Length <= 0 ||
            snapshot.resolutionWidth <= 1 ||
            snapshot.resolutionHeight <= 1)
            return Warn("Raw depth snapshot is empty.");

        EnsureVolumeInitialized();

        Vector3 cameraPosition = GetCameraPosition();
        double integrationRealtime = Time.realtimeSinceStartupAsDouble;
        float posePositionDelta = snapshot.hasSnapshotCameraPose
            ? Vector3.Distance(snapshot.snapshotCameraPosition, cameraPosition)
            : -1f;
        Quaternion integrationRotation = Camera.main != null ? Camera.main.transform.rotation : Quaternion.identity;
        float poseAngleDelta = snapshot.hasSnapshotCameraPose
            ? Quaternion.Angle(snapshot.snapshotCameraRotation, integrationRotation)
            : -1f;
        float snapshotLatencyMs = snapshot.snapshotRealtimeSeconds > 0d
            ? (float)((integrationRealtime - snapshot.snapshotRealtimeSeconds) * 1000d)
            : -1f;
        ResetFrameReferenceAudit();
        Vector3[] positions = snapshot.worldPositions;
        Vector3[] normals = snapshot.worldNormals;
        Color[] meta = snapshot.observationMeta;
        int width = snapshot.resolutionWidth;
        int height = snapshot.resolutionHeight;
        int expected = Mathf.Min(width * height, positions.Length);
        BeginVoteHistoryFrame();
        int stride = Mathf.Max(1, sampleStridePixels);
        int halfBandSteps = Mathf.Max(1, Mathf.CeilToInt(truncationMeters / voxelSizeMeters));

        LastRawFrameIndex = snapshot.frameIndex;
        LastInputSampleCount = 0;
        LastIntegratedSampleCount = 0;
        LastUpdatedVoxelCount = 0;
        LastNeighborFilledSampleCount = 0;
        ResetRejectCounters();

        for (int y = 0; y < height; y += stride)
        {
            for (int x = 0; x < width; x += stride)
            {
                int index = y * width + x;
                if (index < 0 || index >= expected)
                    continue;

                LastInputSampleCount++;
                _activeLedgerSampleIndex = index;
                _activeLedgerPixelX = x;
                _activeLedgerPixelY = y;
                bool auditThisSample = ShouldAuditSample(index);
                BeginConfidenceAuditSample(auditThisSample);
                Vector3 rawPoint = positions[index];

                if (!TryGetUsableSampleOrNeighbor(x, y, width, height, positions, normals, meta, cameraPosition, out Vector3 point, out Vector3 normal, out float sampleWeight))
                {
                    SetObservationVote(ObservationVoteState.Reject, 0f, "invalid_or_basic_filter");
                    RecordRawCoverageGridSample(positions[index], RawDepthDebugKind.Rejected, false);
                    CacheRawDepthDebugSample(positions[index], RawDepthDebugKind.Rejected);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, rawPoint, 0f, 0, 0, 0f, 0f, 0, 0, "discard", "invalid_or_basic_filter", auditThisSample);
                    continue;
                }

                if (!PassDepthNeighborhoodConsistency(x, y, width, height, positions, cameraPosition, point, out int checkedDepthNeighbors, out int consistentDepthNeighbors))
                {
                    SetObservationVote(ObservationVoteState.Reject, 0.05f, "depth_discontinuity");
                    LastRejectedDepthDiscontinuityCount++;
                    RecordRawCoverageGridSample(point, RawDepthDebugKind.DepthEdge, false);
                    CacheRawDepthDebugSample(point, RawDepthDebugKind.DepthEdge);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, 0f, 0f, 0, 0, "discard", "depth_discontinuity", auditThisSample);
                    continue;
                }

                Vector3 pointBeforeRobust = point;
                if (!ApplyRobustDepthPrefilter(x, y, width, height, positions, cameraPosition, ref point, ref sampleWeight))
                {
                    SetObservationVote(ObservationVoteState.Reject, 0.10f, "robust_depth_outlier");
                    LastRejectedRobustDepthCount++;
                    RecordRawCoverageGridSample(point, RawDepthDebugKind.Robust, false);
                    CacheRawDepthDebugSample(point, RawDepthDebugKind.Robust);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), 0f, 0, 0, "discard", "robust_depth_outlier", auditThisSample);
                    continue;
                }
                _auditSampleSupportRatio = checkedDepthNeighbors > 0 ? (float)consistentDepthNeighbors / checkedDepthNeighbors : 0f;
                _auditSampleRobustShiftMeters = Vector3.Distance(pointBeforeRobust, point);
                Vector3 auditViewDirection = (cameraPosition - point).normalized;
                _auditSampleViewFacing = normal.sqrMagnitude > 0.0001f
                    ? Mathf.Abs(Vector3.Dot(normal.normalized, auditViewDirection))
                    : 0f;
                _auditSampleNormal = normal.sqrMagnitude > 0.0001f ? normal.normalized : Vector3.zero;
                MeasureTemporalVoteFeatures(cameraPosition, point, _auditSampleNormal);
                MeasureHistoricalTsdfVoteFeatures(point);
                MeasureTsdfBandVoteFeatures(cameraPosition, point, _auditSampleNormal, halfBandSteps);
                MeasureCrossFrameCleanSurfaceConflict(point, _auditSampleNormal);
                RecordCurrentFrameRawVoteSample(point, _auditSampleNormal);
                MarkSurfaceObservationState(point, 1);

                if (!PassTsdfDepthSupport(checkedDepthNeighbors, consistentDepthNeighbors))
                {
                    SetObservationVote(ObservationVoteState.Reject, 0.15f, "insufficient_depth_support");
                    LastRejectedTsdfDepthSupportCount++;
                    RecordRawCoverageGridSample(point, RawDepthDebugKind.Support, false);
                    CacheRawDepthDebugSample(point, RawDepthDebugKind.Support);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), 0f, 0, 0, "discard", "insufficient_depth_support", auditThisSample);
                    continue;
                }
                ObservationVoteState observationVote = EvaluateObservationVote(sampleWeight);
                if (observationVote == ObservationVoteState.Accept &&
                    WouldAtomicAcceptCreateDuplicateLayer(point, _auditSampleNormal))
                {
                    observationVote = ObservationVoteState.Pending;
                    LastAtomicAcceptDuplicateDowngradeCount++;
                    SetObservationVote(
                        ObservationVoteState.Pending,
                        _auditVoteScore,
                        string.IsNullOrEmpty(_auditVoteReasons)
                            ? "duplicate_layer_guard"
                            : _auditVoteReasons + "+duplicate_layer_guard",
                        true);
                }
                MarkSurfaceObservationState(point, observationVote == ObservationVoteState.Accept ? (byte)3 : observationVote == ObservationVoteState.Pending ? (byte)2 : (byte)4);
                if (observationVote == ObservationVoteState.Accept)
                    RecordVoteHistorySample(point, _auditSampleNormal);
                if (observationVote == ObservationVoteState.Reject &&
                    (useAtomicObservationTsdfBands || enforceObservationWriteGate || observationVoteMode == ObservationVoteMode.RejectOnly))
                {
                    MarkRejectedObservationBand(cameraPosition, point, normal, halfBandSteps);
                    TryMetabolizeOldCleanTsdfBand(cameraPosition, point, sampleWeight, halfBandSteps);
                    RecordRawCoverageGridSample(point, RawDepthDebugKind.Rejected, false);
                    CacheRawDepthDebugSample(point, RawDepthDebugKind.Rejected);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), 0f, 0, 0, "discard", "vote_reject", auditThisSample);
                    continue;
                }
                if (!useAtomicObservationTsdfBands &&
                    observationVote == ObservationVoteState.Pending &&
                    enforceObservationWriteGate &&
                    holdUnconfirmedPendingWrites)
                {
                    if (!PendingObservationHasWriteSupport(sampleWeight, out bool bootstrapWrite, out bool sameFrameProvisionalWrite, out bool strongSampleSeedWrite))
                    {
                        LastVotePendingHoldCount++;
                        if (HasSameFrameEntryConflict())
                            LastSameFrameEntryHeldCount++;
                        RecordRawCoverageGridSample(point, RawDepthDebugKind.Pending, false);
                        CacheRawDepthDebugSample(point, RawDepthDebugKind.Rejected);
                        AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), 0f, 0, 0, "hold", "vote_pending_hold", auditThisSample);
                        continue;
                    }
                    if (strongSampleSeedWrite)
                    {
                        _auditStrongSampleSeedWrite = true;
                        sampleWeight *= Mathf.Clamp(strongSampleSeedWeightScale, 0.05f, 0.5f);
                        LastStrongSampleSeedWriteCount++;
                        if (bootstrapWrite)
                            LastVotePendingBootstrapWriteCount++;
                    }
                    else if (sameFrameProvisionalWrite)
                    {
                        sampleWeight *= Mathf.Clamp(sameFrameEntryProvisionalWeightScale, 0.05f, 0.6f);
                        LastSameFrameEntryProvisionalWriteCount++;
                    }
                    else if (bootstrapWrite)
                    {
                        sampleWeight *= Mathf.Clamp(pendingBootstrapWeightScale, 0.05f, 0.5f);
                        LastVotePendingBootstrapWriteCount++;
                    }
                    else
                    {
                        sampleWeight *= Mathf.Clamp(confirmedPendingWriteWeightScale, 0.05f, 1f);
                        LastVotePendingConfirmedWriteCount++;
                    }
                    RecordVoteHistorySample(point, _auditSampleNormal);
                }
                else if (useAtomicObservationTsdfBands && observationVote == ObservationVoteState.Pending)
                {
                    sampleWeight *= Mathf.Clamp(atomicProvisionalBandWeightScale, 0.05f, 0.5f);
                    RecordVoteHistorySample(point, _auditSampleNormal);
                }
                AccumulateFrameReferenceAudit(point);

                float oldTsdfAtSurface = 0f;
                int oldWeightAtSurface = 0;
                int oldPendingHitsAtSurface = 0;
                if (TryWorldToVoxel(point, out int auditVx, out int auditVy, out int auditVz))
                {
                    int auditVoxelIndex = Index(auditVx, auditVy, auditVz);
                    oldTsdfAtSurface = _tsdf[auditVoxelIndex];
                    oldWeightAtSurface = _weights[auditVoxelIndex];
                    if (_correctionTsdfHits != null && auditVoxelIndex >= 0 && auditVoxelIndex < _correctionTsdfHits.Length)
                        oldPendingHitsAtSurface = _correctionTsdfHits[auditVoxelIndex];
                }
                int dirtyReplaceBeforeSample = LastReplacedDirtyTsdfCount;
                int dirtyCleanBeforeSample = LastCleanedDirtyTsdfNeighborCount;
                int dirtyQuarantineBeforeSample = LastDirtyTsdfQuarantineCount;
                bool touchedDenseTsdf;
                _activeAtomicBandWrite = useAtomicObservationTsdfBands;
                _activeAtomicBandVote = observationVote;
                _activeSurfaceObservationState = observationVote == ObservationVoteState.Accept ? (byte)3 : (byte)2;
                try
                {
                    bool allowClearingRay = observationVote == ObservationVoteState.Accept && ShouldCarveFreeSpaceAtPixel(x, y, stride);
                    touchedDenseTsdf = useProjectiveTsdfIntegration
                        ? IntegrateProjectiveTsdfBand(cameraPosition, point, sampleWeight, halfBandSteps, allowClearingRay)
                        : IntegrateNormalTsdfBand(point, normal, sampleWeight, halfBandSteps);
                }
                finally
                {
                    _activeAtomicBandWrite = false;
                    _activeAtomicBandVote = ObservationVoteState.Unassessed;
                    _activeSurfaceObservationState = 0;
                }
                if (!touchedDenseTsdf)
                {
                    LastRejectedOutsideVolumeCount++;
                    RecordRawCoverageGridSample(point, RawDepthDebugKind.Outside, false);
                    CacheRawDepthDebugSample(point, RawDepthDebugKind.Outside);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), oldTsdfAtSurface, oldWeightAtSurface, oldPendingHitsAtSurface, "discard", "outside_tsdf_volume", auditThisSample);
                    continue;
                }

                LastIntegratedSampleCount++;
                bool sampleHitDirtyRepair =
                    LastReplacedDirtyTsdfCount > dirtyReplaceBeforeSample ||
                    LastCleanedDirtyTsdfNeighborCount > dirtyCleanBeforeSample ||
                    LastDirtyTsdfQuarantineCount > dirtyQuarantineBeforeSample;
                if (sampleHitDirtyRepair)
                {
                    LastBlockedDirtyLightCoverSampleCount++;
                    TotalBlockedDirtyLightCoverSampleCount++;
                    RecordRawCoverageGridSample(point, RawDepthKindFromAuditOutcome(), false);
                    CacheRawDepthDebugSample(point, RawDepthKindFromAuditOutcome());
                    string repairReason = LastReplacedDirtyTsdfCount > dirtyReplaceBeforeSample ? "dirty_replaced" :
                        (LastDirtyTsdfQuarantineCount > dirtyQuarantineBeforeSample ? "dirty_quarantined" : "dirty_neighbor_cleaned");
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), oldTsdfAtSurface, oldWeightAtSurface, oldPendingHitsAtSurface, "repair", repairReason, auditThisSample);
                }
                else if (!addFallbackOnlyAfterTsdfSupport || sampleWeight >= minFallbackSurfaceSampleWeight)
                {
                    AddFallbackSurfaceSample(point, normal);
                    RawDepthDebugKind outcomeKind = RawDepthKindFromAuditOutcome();
                    string decision = _auditCenterOutcome == "pending_correction" || _auditCenterOutcome == "pending_stability"
                        ? "pending"
                        : (_auditCenterOutcome == "rejected_conflict" ? "discard" : "accept");
                    RecordRawCoverageGridSample(point, outcomeKind, observationVote == ObservationVoteState.Accept);
                    CacheRawDepthDebugSample(point, outcomeKind);
                    string reason = _auditCenterOutcome;
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), oldTsdfAtSurface, oldWeightAtSurface, oldPendingHitsAtSurface, decision, reason, auditThisSample);
                }
                else
                {
                    LastRejectedFallbackWeakCount++;
                    RecordRawCoverageGridSample(point, RawDepthDebugKind.Rejected, false);
                    CacheRawDepthDebugSample(point, RawDepthDebugKind.Rejected);
                    AppendConfidenceAuditRow(snapshot.frameIndex, index, x, y, point, sampleWeight, checkedDepthNeighbors, consistentDepthNeighbors, Vector3.Distance(pointBeforeRobust, point), oldTsdfAtSurface, oldWeightAtSurface, oldPendingHitsAtSurface, "discard", "fallback_weight_weak", auditThisSample);
                }
            }
        }

        FinalizeRawCoverageGridDiagnostics();
        AppendConfidenceAuditFrame(snapshot.frameIndex, snapshotLatencyMs, posePositionDelta, poseAngleDelta, cameraPosition);
        EndVoteHistoryFrame();
        _activeLedgerSampleIndex = -1;
        _activeLedgerPixelX = -1;
        _activeLedgerPixelY = -1;
        IntegratedFrameCount++;
        _confidenceAuditIntegratedFrames++;
        UpdateDirtyTsdfLifecycleDiagnostics();
        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverTsdfSingleShellPrototype] Integrated raw frame={snapshot.frameIndex} resolution={width}x{height} stride={stride} input={LastInputSampleCount} accepted={LastIntegratedSampleCount} acceptRatio={(LastInputSampleCount > 0 ? (float)LastIntegratedSampleCount / LastInputSampleCount : 0f):P1} voxelUpdates={LastUpdatedVoxelCount} fallback={GetFallbackSurfaceSampleCount()}/{maxFallbackSurfaceSamples} chunks={LastSurfaceChunkCount}/{maxSurfaceChunks} uniqueVoxels={_fallbackSurfaceVoxels.Count} rejects invalid={LastRejectedInvalidPositionCount} depth={LastRejectedDepthRangeCount} edge={LastRejectedDepthDiscontinuityCount} robust={LastRejectedRobustDepthCount}/{LastRobustDepthDownweightedCount}/{LastRobustDepthCorrectedCount} edgeErode={LastRejectedDepthEdgeErosionCount}/{LastDepthEdgeDownweightedCount} support={LastRejectedTsdfDepthSupportCount} normal={LastRejectedNormalCount} facing={LastRejectedFacingCount} confidence={LastRejectedConfidenceCount} tsdfConflict={LastRejectedTsdfConflictCount} stable={LastPendingStableTsdfCount} corrWait={LastPendingTsdfCorrectionCount} corrected={LastCorrectedTsdfCount} dirtyReplace={LastReplacedDirtyTsdfCount} dirtyClean={LastCleanedDirtyTsdfNeighborCount} dirtyQ={LastDirtyTsdfQuarantineCount} carve={LastCarvedFreeSpaceVoxelCount} outsideVolume={LastRejectedOutsideVolumeCount} fbWeak={LastRejectedFallbackWeakCount} fallbackReplaced={LastSkippedFallbackCapacityCount} volume={_dimX}x{_dimY}x{_dimZ} origin={_volumeOriginWorld:F2} size={volumeSizeMeters:F1}",
                this);
        }

        return LastIntegratedSampleCount > 0;
    }

    private void BeginSurfaceProfileDiagnostics()
    {
        _surfaceDiagnosticCaptureIndex++;
        _lastLayerNoZeroRayCount = 0;
        _lastLayerSingleZeroRayCount = 0;
        _lastLayerMultiZeroRayCount = 0;
        _lastLayerDisplayPairRayCount = 0;
        _lastLayerLegalOcclusionRayCount = 0;
        _lastLayerDuplicateRayCount = 0;
        _lastLayerAmbiguousRayCount = 0;
        _lastHoleDiagnosticRowCount = 0;
    }

    private void EndSurfaceProfileDiagnostics(int integratedFrames, string status)
    {
        if (!writeHoleAndLayerDiagnosticsOnCapture || _tsdf == null || _weights == null)
            return;

        try
        {
            string directory = ResolveConfidenceAuditDirectory();
            Directory.CreateDirectory(directory);
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", AuditCulture);
            string stem = $"surface_profile_{timestamp}_{_surfaceDiagnosticCaptureIndex:D3}";
            string holePath = Path.Combine(directory, stem + "_holes.csv");
            string rayPath = Path.Combine(directory, stem + "_rays.csv");
            string layerPath = Path.Combine(directory, stem + "_layers.csv");
            string pairPath = Path.Combine(directory, stem + "_layer_pairs.csv");
            string summaryPath = Path.Combine(directory, stem + "_summary.txt");

            StringBuilder holes = BuildHoleProfileCsv();
            BuildLayerProfileCsv(out StringBuilder rays, out StringBuilder layers, out StringBuilder pairs);
            File.WriteAllText(holePath, holes.ToString(), new UTF8Encoding(false));
            File.WriteAllText(rayPath, rays.ToString(), new UTF8Encoding(false));
            File.WriteAllText(layerPath, layers.ToString(), new UTF8Encoding(false));
            File.WriteAllText(pairPath, pairs.ToString(), new UTF8Encoding(false));

            StringBuilder summary = new StringBuilder(512);
            summary.AppendLine("ScanCover TSDF surface profile");
            summary.AppendLine("status=" + status);
            summary.AppendLine("capture=" + _surfaceDiagnosticCaptureIndex);
            summary.AppendLine("integrated_frames=" + integratedFrames);
            summary.AppendLine("hole_rows=" + _lastHoleDiagnosticRowCount);
            summary.AppendLine("ray_no_zero=" + _lastLayerNoZeroRayCount);
            summary.AppendLine("ray_single_zero=" + _lastLayerSingleZeroRayCount);
            summary.AppendLine("ray_multi_zero=" + _lastLayerMultiZeroRayCount);
            summary.AppendLine("ray_display_pair=" + _lastLayerDisplayPairRayCount);
            summary.AppendLine("ray_legal_occlusion=" + _lastLayerLegalOcclusionRayCount);
            summary.AppendLine("ray_duplicate_layer=" + _lastLayerDuplicateRayCount);
            summary.AppendLine("ray_ambiguous_multi=" + _lastLayerAmbiguousRayCount);
            summary.AppendLine("layer_viz_primary=" + LastPrimaryLayerDiagnosticMarkerCount);
            summary.AppendLine("layer_viz_secondary=" + LastSecondaryLayerDiagnosticMarkerCount);
            summary.AppendLine("layer_viz_unclassified=" + LastUnclassifiedLayerDiagnosticMarkerCount);
            summary.AppendLine("layer_confirmed_secondary_voxels=" + LastConfirmedSecondaryLayerVoxelCount);
            summary.AppendLine("layer_hole_trend_hole_load=" + LastLayerHoleTrendHoleLoad);
            summary.AppendLine("layer_hole_trend_secondary_delta=" + LastLayerHoleTrendSecondaryDelta);
            summary.AppendLine("layer_hole_trend_hole_delta=" + LastLayerHoleTrendHoleDelta);
            summary.AppendLine("layer_hole_trend_correlation=" + AuditFloat(LastLayerHoleTrendCorrelation));
            summary.AppendLine("layer_hole_trend_samples=" + LastLayerHoleTrendSampleCount);
            summary.AppendLine("hole_repair_candidate=" + LastHoleSideRepairCandidateCount);
            summary.AppendLine("hole_repair_applied=" + LastHoleSideRepairAppliedCount);
            summary.AppendLine("hole_repair_block_support=" + LastHoleSideRepairBlockedSupportCount);
            summary.AppendLine("hole_repair_block_dirty=" + LastHoleSideRepairBlockedDirtyCount);
            summary.AppendLine("hole_repair_block_multizero=" + LastHoleSideRepairBlockedMultiZeroCount);
            summary.AppendLine("hole_repair_block_plane=" + LastHoleSideRepairBlockedPlaneCount);
            summary.AppendLine("hole_repair_plane_band_voxels=" + LastHoleSideRepairPlaneBandVoxelCount);
            summary.AppendLine("hole_repair_plane_confirmed=" + LastHoleSideRepairPlaneConfirmedCount);
            summary.AppendLine("hole_repair_plane_retired=" + LastHoleSideRepairPlaneRetiredCount);
            summary.AppendLine("hole_repair_retired_no_near_accept=" + LastHoleSideRepairRetiredNoNearAcceptCount);
            summary.AppendLine("hole_repair_retired_sign_mismatch=" + LastHoleSideRepairRetiredSignMismatchCount);
            summary.AppendLine("hole_repair_retired_plane_mismatch=" + LastHoleSideRepairRetiredPlaneMismatchCount);
            summary.AppendLine("hole_repair_retired_tsdf_delta=" + LastHoleSideRepairRetiredTsdfDeltaCount);
            summary.AppendLine("hole_repair_retired_expired_after_pass=" + LastHoleSideRepairRetiredExpiredAfterPassCount);
            summary.AppendLine("hole_repair_plane_distance_0_0p5_vox=" + LastHoleSideRepairPlaneDistance0ToHalfCount);
            summary.AppendLine("hole_repair_plane_distance_0p5_0p75_vox=" + LastHoleSideRepairPlaneDistanceHalfTo075Count);
            summary.AppendLine("hole_repair_plane_distance_0p75_1_vox=" + LastHoleSideRepairPlaneDistance075To1Count);
            summary.AppendLine("hole_repair_plane_distance_1_1p5_vox=" + LastHoleSideRepairPlaneDistance1To15Count);
            summary.AppendLine("hole_repair_plane_distance_1p5_2_vox=" + LastHoleSideRepairPlaneDistance15To2Count);
            summary.AppendLine("hole_repair_plane_distance_over_2_vox=" + LastHoleSideRepairPlaneDistanceOver2Count);
            summary.AppendLine("mesh_commit_blocks=" + LastCommittedMeshBlockCount);
            summary.AppendLine("mesh_candidate_blocks=" + LastCandidateMeshBlockCount);
            summary.AppendLine("mesh_retained_commit_blocks=" + LastRetainedCommittedMeshBlockCount);
            summary.AppendLine("mesh_hold_spatial=" + LastCommittedMeshHoldSpatialCount);
            summary.AppendLine("mesh_hold_triangles=" + LastCommittedMeshHoldTriangleCount);
            summary.AppendLine("mesh_growth_triangles=" + LastCommittedMeshGrowthTriangleCount);
            summary.AppendLine("accept_duplicate_candidates=" + LastAtomicAcceptDuplicateCandidateCount);
            summary.AppendLine("accept_duplicate_downgraded=" + LastAtomicAcceptDuplicateDowngradeCount);
            summary.AppendLine("accept_duplicate_same_surface=" + LastAtomicAcceptDuplicateSameSurfaceCount);
            summary.AppendLine("duplicate_cleanup_queued=" + LastDuplicateLayerCleanupQueuedCount);
            summary.AppendLine("duplicate_cleanup_decayed=" + LastDuplicateLayerCleanupDecayedCount);
            summary.AppendLine("duplicate_cleanup_cleared=" + LastDuplicateLayerCleanupClearedCount);
            summary.AppendLine("duplicate_cleanup_boosted=" + LastDuplicateLayerCleanupBoostedCount);
            summary.AppendLine("duplicate_cleanup_max_evidence=" + LastDuplicateLayerCleanupMaxEvidence);
            File.WriteAllText(summaryPath, summary.ToString(), new UTF8Encoding(false));
            _lastHoleDiagnosticPath = holePath;
            _lastLayerDiagnosticPath = rayPath;
        }
        catch (System.Exception ex)
        {
            _lastHoleDiagnosticPath = "WRITE FAILED";
            _lastLayerDiagnosticPath = "WRITE FAILED";
            Debug.LogWarning("[ScanCover] Surface profile write failed: " + ex.Message, this);
        }
    }

    private StringBuilder BuildHoleProfileCsv()
    {
        StringBuilder csv = new StringBuilder(256 * 1024);
        csv.AppendLine("capture,frame,cell_x,cell_y,cell_z,world_x,world_y,world_z,cause,positive_only,negative_only,vertex_index,corner,voxel_index,tsdf,weight,last_operation,last_write_sequence");
        if (_cellVertexIndices == null || _mesh == null)
            return csv;

        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);
        bool[] vertexUsed = new bool[Mathf.Max(0, _mesh.vertexCount)];
        int[] meshTriangles = _mesh.triangles;
        for (int i = 0; i < meshTriangles.Length; i++)
        {
            int vertex = meshTriangles[i];
            if (vertex >= 0 && vertex < vertexUsed.Length)
                vertexUsed[vertex] = true;
        }

        int limit = Mathf.Max(1, maxHoleDiagnosticRows);
        for (int z = 0; z < cellZ && _lastHoleDiagnosticRowCount < limit; z++)
        for (int y = 0; y < cellY && _lastHoleDiagnosticRowCount < limit; y++)
        for (int x = 0; x < cellX && _lastHoleDiagnosticRowCount < limit; x++)
        {
            int cellIndex = CellIndex(x, y, z, cellX, cellY);
            int vertexIndex = _cellVertexIndices[cellIndex];
            bool used = vertexIndex >= 0 && vertexIndex < vertexUsed.Length && vertexUsed[vertexIndex];
            if (used || (vertexIndex < 0 && !CellTouchesUsedSurface(x, y, z, cellX, cellY, cellZ, vertexUsed)))
                continue;

            HoleSupportCause cause = ClassifyHoleSupportCause(x, y, z, vertexIndex, out bool positiveOnly, out bool negativeOnly);
            Vector3 center = CellCenter(x, y, z);
            for (int corner = 0; corner < 8; corner++)
            {
                int vx = x + ((corner & 1) != 0 ? 1 : 0);
                int vy = y + ((corner & 2) != 0 ? 1 : 0);
                int vz = z + ((corner & 4) != 0 ? 1 : 0);
                int voxelIndex = Index(vx, vy, vz);
                bool hasProvenance = _voxelWriteProvenance.TryGetValue(voxelIndex, out VoxelWriteProvenance provenance);
                csv.Append(_surfaceDiagnosticCaptureIndex).Append(',').Append(IntegratedFrameCount).Append(',')
                    .Append(x).Append(',').Append(y).Append(',').Append(z).Append(',')
                    .Append(AuditFloat(center.x)).Append(',').Append(AuditFloat(center.y)).Append(',').Append(AuditFloat(center.z)).Append(',')
                    .Append(cause).Append(',').Append(positiveOnly ? 1 : 0).Append(',').Append(negativeOnly ? 1 : 0).Append(',')
                    .Append(vertexIndex).Append(',').Append(corner).Append(',').Append(voxelIndex).Append(',')
                    .Append(AuditFloat(_tsdf[voxelIndex])).Append(',').Append(_weights[voxelIndex]).Append(',')
                    .Append(hasProvenance ? provenance.LastOperation : "untracked").Append(',')
                    .Append(hasProvenance ? provenance.WriteSequence : 0).AppendLine();
            }
            _lastHoleDiagnosticRowCount++;
        }
        return csv;
    }

    private void BuildLayerProfileCsv(out StringBuilder rays, out StringBuilder layers, out StringBuilder pairs)
    {
        rays = new StringBuilder(128 * 1024);
        layers = new StringBuilder(256 * 1024);
        pairs = new StringBuilder(128 * 1024);
        rays.AppendLine("capture,frame,ray_x,ray_y,zero_crossings,near_zero_runs,classification,first_depth_m,last_depth_m,max_layer_gap_m");
        layers.AppendLine("capture,frame,ray_x,ray_y,layer,depth_m,world_x,world_y,world_z,front_voxel,front_tsdf,front_weight,front_operation,front_sequence,back_voxel,back_tsdf,back_weight,back_operation,back_sequence");
        pairs.AppendLine("capture,frame,ray_x,ray_y,pair,classification,front_depth_m,back_depth_m,gap_m,normal_dot,lateral_m,front_operation,front_sequence,front_capture,front_surface_x,front_surface_y,front_surface_z,back_operation,back_sequence,back_capture,back_surface_x,back_surface_y,back_surface_z");
        Camera camera = Camera.main;
        if (camera == null)
            return;

        int countX = Mathf.Max(1, layerDiagnosticRaysX);
        int countY = Mathf.Max(1, layerDiagnosticRaysY);
        for (int ry = 0; ry < countY; ry++)
        for (int rx = 0; rx < countX; rx++)
        {
            Ray ray = camera.ViewportPointToRay(new Vector3((rx + 0.5f) / countX, (ry + 0.5f) / countY, 0f));
            if (!TryIntersectVolume(ray, out float enter, out float exit))
                continue;

            List<float> depths = new List<float>(4);
            List<int> frontIndices = new List<int>(4);
            List<int> backIndices = new List<int>(4);
            int nearZeroRuns = 0;
            bool inNearZeroRun = false;
            bool hasPrevious = false;
            float previousTsdf = 0f;
            int previousIndex = -1;
            float previousDepth = 0f;
            int lastIndex = -1;
            float step = Mathf.Max(0.001f, voxelSizeMeters * Mathf.Max(0.25f, layerDiagnosticStepVoxels));
            for (float depth = Mathf.Max(0f, enter); depth <= exit; depth += step)
            {
                Vector3 point = ray.origin + ray.direction * depth;
                if (!TryWorldToVoxel(point, out int vx, out int vy, out int vz))
                    continue;
                int index = Index(vx, vy, vz);
                if (index == lastIndex)
                    continue;
                lastIndex = index;
                if (_weights[index] <= 0)
                {
                    hasPrevious = false;
                    inNearZeroRun = false;
                    continue;
                }

                float value = _tsdf[index];
                bool nearZero = Mathf.Abs(value) <= 0.25f;
                if (nearZero && !inNearZeroRun)
                    nearZeroRuns++;
                inNearZeroRun = nearZero;
                if (hasPrevious && ((previousTsdf < 0f && value >= 0f) || (previousTsdf > 0f && value <= 0f)))
                {
                    float denominator = Mathf.Abs(previousTsdf) + Mathf.Abs(value);
                    float fraction = denominator > 0.000001f ? Mathf.Abs(previousTsdf) / denominator : 0.5f;
                    depths.Add(Mathf.Lerp(previousDepth, depth, fraction));
                    frontIndices.Add(previousIndex);
                    backIndices.Add(index);
                }
                previousTsdf = value;
                previousIndex = index;
                previousDepth = depth;
                hasPrevious = true;
            }

            string classification;
            if (depths.Count <= 0)
            {
                classification = "NO_ZERO";
                _lastLayerNoZeroRayCount++;
            }
            else if (depths.Count == 1 && nearZeroRuns > 1)
            {
                classification = "DISPLAY_PAIR";
                _lastLayerSingleZeroRayCount++;
                _lastLayerDisplayPairRayCount++;
            }
            else if (depths.Count == 1)
            {
                classification = "SINGLE_ZERO";
                _lastLayerSingleZeroRayCount++;
            }
            else
            {
                _lastLayerMultiZeroRayCount++;
                bool hasDuplicate = false;
                bool hasAmbiguous = false;
                for (int pair = 0; pair < depths.Count - 1; pair++)
                {
                    string pairClass = AppendLayerPairProfileRow(
                        pairs, rx, ry, pair, depths[pair], depths[pair + 1],
                        frontIndices[pair], backIndices[pair], frontIndices[pair + 1], backIndices[pair + 1]);
                    hasDuplicate |= pairClass == "DUPLICATE_LAYER";
                    hasAmbiguous |= pairClass == "AMBIGUOUS_MULTI";
                }
                if (hasDuplicate)
                {
                    classification = "DUPLICATE_LAYER";
                    _lastLayerDuplicateRayCount++;
                }
                else if (hasAmbiguous)
                {
                    classification = "AMBIGUOUS_MULTI";
                    _lastLayerAmbiguousRayCount++;
                }
                else
                {
                    classification = "LEGAL_OCCLUSION";
                    _lastLayerLegalOcclusionRayCount++;
                }
            }

            float maxGap = 0f;
            for (int i = 1; i < depths.Count; i++)
                maxGap = Mathf.Max(maxGap, depths[i] - depths[i - 1]);
            rays.Append(_surfaceDiagnosticCaptureIndex).Append(',').Append(IntegratedFrameCount).Append(',')
                .Append(rx).Append(',').Append(ry).Append(',').Append(depths.Count).Append(',').Append(nearZeroRuns).Append(',')
                .Append(classification).Append(',')
                .Append(depths.Count > 0 ? AuditFloat(depths[0]) : string.Empty).Append(',')
                .Append(depths.Count > 0 ? AuditFloat(depths[depths.Count - 1]) : string.Empty).Append(',')
                .Append(AuditFloat(maxGap)).AppendLine();

            for (int layer = 0; layer < depths.Count; layer++)
                AppendLayerProfileRow(layers, rx, ry, layer, depths[layer], ray, frontIndices[layer], backIndices[layer]);
        }
    }

    private string AppendLayerPairProfileRow(
        StringBuilder csv, int rayX, int rayY, int pair, float frontDepth, float backDepth,
        int frontA, int frontB, int backA, int backB)
    {
        bool hasFront = TryGetNewestLayerProvenance(frontA, frontB, out VoxelWriteProvenance front);
        bool hasBack = TryGetNewestLayerProvenance(backA, backB, out VoxelWriteProvenance back);
        float gap = Mathf.Max(0f, backDepth - frontDepth);
        float normalDot = -1f;
        float lateral = -1f;
        string classification = "AMBIGUOUS_MULTI";
        if (hasFront && hasBack && front.SurfaceNormal.sqrMagnitude > 0.0001f && back.SurfaceNormal.sqrMagnitude > 0.0001f)
        {
            Vector3 frontNormal = front.SurfaceNormal.normalized;
            Vector3 backNormal = back.SurfaceNormal.normalized;
            normalDot = Mathf.Abs(Vector3.Dot(frontNormal, backNormal));
            Vector3 meanNormal = (frontNormal + (Vector3.Dot(frontNormal, backNormal) >= 0f ? backNormal : -backNormal)).normalized;
            Vector3 delta = back.SurfacePoint - front.SurfacePoint;
            lateral = (delta - meanNormal * Vector3.Dot(delta, meanNormal)).magnitude;
            if (gap <= duplicateLayerMaxGapMeters && normalDot >= duplicateLayerMinNormalDot && lateral <= duplicateLayerMaxGapMeters * 1.5f)
                classification = "DUPLICATE_LAYER";
            else if (gap >= legalOcclusionMinGapMeters || normalDot < 0.7f || lateral > 0.25f)
                classification = "LEGAL_OCCLUSION";
        }

        csv.Append(_surfaceDiagnosticCaptureIndex).Append(',').Append(IntegratedFrameCount).Append(',')
            .Append(rayX).Append(',').Append(rayY).Append(',').Append(pair).Append(',').Append(classification).Append(',')
            .Append(AuditFloat(frontDepth)).Append(',').Append(AuditFloat(backDepth)).Append(',').Append(AuditFloat(gap)).Append(',')
            .Append(AuditFloat(normalDot)).Append(',').Append(AuditFloat(lateral)).Append(',')
            .Append(hasFront ? front.LastOperation : "untracked").Append(',').Append(hasFront ? front.WriteSequence : 0).Append(',').Append(hasFront ? front.Capture : 0).Append(',')
            .Append(hasFront ? AuditFloat(front.SurfacePoint.x) : string.Empty).Append(',').Append(hasFront ? AuditFloat(front.SurfacePoint.y) : string.Empty).Append(',').Append(hasFront ? AuditFloat(front.SurfacePoint.z) : string.Empty).Append(',')
            .Append(hasBack ? back.LastOperation : "untracked").Append(',').Append(hasBack ? back.WriteSequence : 0).Append(',').Append(hasBack ? back.Capture : 0).Append(',')
            .Append(hasBack ? AuditFloat(back.SurfacePoint.x) : string.Empty).Append(',').Append(hasBack ? AuditFloat(back.SurfacePoint.y) : string.Empty).Append(',').Append(hasBack ? AuditFloat(back.SurfacePoint.z) : string.Empty).AppendLine();
        return classification;
    }

    private bool TryGetNewestLayerProvenance(int a, int b, out VoxelWriteProvenance provenance)
    {
        bool hasA = _voxelWriteProvenance.TryGetValue(a, out VoxelWriteProvenance pa);
        bool hasB = _voxelWriteProvenance.TryGetValue(b, out VoxelWriteProvenance pb);
        if (hasA && (!hasB || pa.WriteSequence >= pb.WriteSequence))
        {
            provenance = pa;
            return true;
        }
        if (hasB)
        {
            provenance = pb;
            return true;
        }
        provenance = default;
        return false;
    }

    private void AppendLayerProfileRow(StringBuilder csv, int rayX, int rayY, int layer, float depth, Ray ray, int frontIndex, int backIndex)
    {
        Vector3 world = ray.origin + ray.direction * depth;
        bool hasFront = _voxelWriteProvenance.TryGetValue(frontIndex, out VoxelWriteProvenance front);
        bool hasBack = _voxelWriteProvenance.TryGetValue(backIndex, out VoxelWriteProvenance back);
        csv.Append(_surfaceDiagnosticCaptureIndex).Append(',').Append(IntegratedFrameCount).Append(',')
            .Append(rayX).Append(',').Append(rayY).Append(',').Append(layer).Append(',').Append(AuditFloat(depth)).Append(',')
            .Append(AuditFloat(world.x)).Append(',').Append(AuditFloat(world.y)).Append(',').Append(AuditFloat(world.z)).Append(',')
            .Append(frontIndex).Append(',').Append(AuditFloat(_tsdf[frontIndex])).Append(',').Append(_weights[frontIndex]).Append(',')
            .Append(hasFront ? front.LastOperation : "untracked").Append(',').Append(hasFront ? front.WriteSequence : 0).Append(',')
            .Append(backIndex).Append(',').Append(AuditFloat(_tsdf[backIndex])).Append(',').Append(_weights[backIndex]).Append(',')
            .Append(hasBack ? back.LastOperation : "untracked").Append(',').Append(hasBack ? back.WriteSequence : 0).AppendLine();
    }

    private bool TryIntersectVolume(Ray ray, out float enter, out float exit)
    {
        Vector3 min = _volumeOriginWorld;
        Vector3 max = _volumeOriginWorld + new Vector3((_dimX - 1) * voxelSizeMeters, (_dimY - 1) * voxelSizeMeters, (_dimZ - 1) * voxelSizeMeters);
        enter = 0f;
        exit = float.MaxValue;
        for (int axis = 0; axis < 3; axis++)
        {
            float origin = ray.origin[axis];
            float direction = ray.direction[axis];
            if (Mathf.Abs(direction) < 0.000001f)
            {
                if (origin < min[axis] || origin > max[axis])
                    return false;
                continue;
            }
            float inverse = 1f / direction;
            float a = (min[axis] - origin) * inverse;
            float b = (max[axis] - origin) * inverse;
            if (a > b)
            {
                float swap = a;
                a = b;
                b = swap;
            }
            enter = Mathf.Max(enter, a);
            exit = Mathf.Min(exit, b);
            if (exit < enter)
                return false;
        }
        return exit >= Mathf.Max(0f, enter);
    }

    private void BeginConfidenceAuditCapture()
    {
        BeginSurfaceProfileDiagnostics();
        ResetHardTsdfWriteAuditCounters();
        BeginHardTsdfWriteAuditCapture();
        ResetRawCoverageGridDiagnostics();
        if (!writeConfidenceAuditOnCapture)
            return;

        if (clearRawDepthDebugAtCaptureStart)
            ClearRawDepthDebugCache();
        _confidenceAuditCaptureIndex++;
        _confidenceAuditRowCount = 0;
        _confidenceAuditDroppedRows = 0;
        _confidenceAuditIntegratedFrames = 0;
        _auditVoteAcceptCount = 0;
        _auditVotePendingCount = 0;
        _auditVoteRejectCount = 0;
        _auditVoteEnforcedRejectCount = 0;
        LastVoteSameFrameConflictRejectCount = 0;
        LastVotePendingHoldCount = 0;
        LastVotePendingConfirmedWriteCount = 0;
        LastVotePendingBootstrapWriteCount = 0;
        LastStrongSampleSeedWriteCount = 0;
        LastStrongSampleSeedBlockedCount = 0;
        ResetStrongSampleSeedTemporaryBlockDiagnostics();
        ResetFormalIntegrateGateDiagnostics();
        LastVoteStrongCurrentAcceptCount = 0;
        LastVoteCrossFrameCleanConflictRejectCount = 0;
        LastSameFrameEntryStableCount = 0;
        LastSameFrameEntryProvisionalCount = 0;
        LastSameFrameEntryProvisionalWriteCount = 0;
        LastSameFrameEntryRejectedCount = 0;
        LastSameFrameEntryHeldCount = 0;
        _confidenceAuditRows = new StringBuilder(1024 * 1024);
        _confidenceAuditFrameRows = new StringBuilder(2048);
        _doubleLayerPairRows = new StringBuilder(256 * 1024);
        _doubleLayerPairRowCount = 0;
        _doubleLayerPairDroppedRows = 0;
        _doubleLayerPairRows.AppendLine(
            "capture,detect_frame,triangle_a,triangle_b,separation_m,lateral_m,normal_dot," +
            "a_center_x,a_center_y,a_center_z,a_voxel_index,a_voxel_x,a_voxel_y,a_voxel_z,a_tsdf,a_weight,a_first_frame,a_last_frame,a_capture,a_sample,a_pixel_x,a_pixel_y,a_last_operation,a_write_count,a_integrate_count,a_carve_count,a_replace_count,a_repair_count,a_surface_depth_m,a_ray_x,a_ray_y,a_ray_z,a_dirty,a_pending," +
            "b_center_x,b_center_y,b_center_z,b_voxel_index,b_voxel_x,b_voxel_y,b_voxel_z,b_tsdf,b_weight,b_first_frame,b_last_frame,b_capture,b_sample,b_pixel_x,b_pixel_y,b_last_operation,b_write_count,b_integrate_count,b_carve_count,b_replace_count,b_repair_count,b_surface_depth_m,b_ray_x,b_ray_y,b_ray_z,b_dirty,b_pending," +
            "same_last_frame,same_sample,frame_gap,surface_depth_delta_m,ray_angle_deg,both_integrated,either_carved");
        _confidenceAuditFrameRows.AppendLine(
            "capture,frame,snapshot_latency_ms,pose_position_delta_m,pose_angle_delta_deg,integration_camera_x,integration_camera_y,integration_camera_z,stable_reference_count,reference_mean_residual_m,reference_vector_coherence,reference_mean_dx,reference_mean_dy,reference_mean_dz");
        _captureAuditVoxels.Clear();
        _auditMaxPosePositionDelta = 0f;
        _auditMaxPoseAngleDelta = 0f;
        _auditMaxSnapshotLatencyMs = 0f;
        _auditMeshAssociatedVoxelCount = 0;
        _auditRecurrentAfterReplaceCount = 0;
        _auditLikelyDepthInstabilityCount = 0;
        _auditViewOrRegistrationCount = 0;
        _auditSuspectAssociatedConflictVoxelCount = 0;
        _auditSuspectVoxelsLinkedToConflictCount = 0;
        _auditPureGeometryAssociatedConflictVoxelCount = 0;
        _auditPureGeometryVoxelsLinkedToConflictCount = 0;
        _auditPureGeometryAssociatedConflictVoxelCount = 0;
        _auditPureGeometryVoxelsLinkedToConflictCount = 0;
        _confidenceAuditRows.AppendLine(
            "capture,frame,sample,pixel_x,pixel_y,world_x,world_y,world_z,sample_weight,depth_checked,depth_consistent,support_ratio,robust_shift_m,view_facing,temporal_depth_delta_m,temporal_world_delta_m,temporal_normal_delta_deg,historical_surface_distance_m,old_tsdf_residual,old_tsdf_history_weight,history_agreement,history_support_count,band_history_count,band_conflict_count,band_high_weight_conflict_count,band_conflict_ratio,band_high_weight_conflict_ratio,band_mean_residual,band_max_residual,vote_state,vote_score,vote_reasons,vote_enforced,old_tsdf,old_weight,pending_hits,decision,reason,center_sample_tsdf,center_old_tsdf,center_old_weight,center_stable_hits_before,center_stable_hits_after,center_correction_hits_before,center_correction_hits_after,center_sign_flip,center_residual,center_outcome,center_conflict_cause,band_written,band_pending_stable,band_pending_correction,band_replaced,band_corrected,band_rejected_conflict,same_frame_depth_delta_m,same_frame_normal_delta_deg,cross_frame_clean_depth_delta_m,cross_frame_clean_lateral_m,cross_frame_clean_normal_delta_deg,cross_frame_clean_frame_gap,cross_frame_clean_weight");
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", AuditCulture);
        _confidenceAuditStem = $"confidence_audit_{timestamp}_{_confidenceAuditCaptureIndex:D3}";
        _confidenceAuditDirectory = ResolveConfidenceAuditDirectory();
        BeginContributionLedger();
    }

    private void BeginContributionLedger()
    {
        _contributionLedgerRowCount = 0;
        _contributionLedgerWriteFailures = 0;
        _contributionLedgerPath = null;
        try
        {
            Directory.CreateDirectory(_confidenceAuditDirectory);
            _contributionLedgerPath = Path.Combine(
                _confidenceAuditDirectory,
                _confidenceAuditStem + "_contribution_ledger.csv");
            _contributionLedgerWriter = new StreamWriter(
                _contributionLedgerPath,
                false,
                new UTF8Encoding(false),
                1024 * 1024);
            _contributionLedgerWriter.WriteLine(
                "capture,frame,sample,pixel_x,pixel_y,voxel_index,voxel_x,voxel_y,voxel_z,world_x,world_y,world_z,operation,old_tsdf,old_weight,observed_tsdf,observed_weight,new_tsdf,new_weight,old_numerator,observed_numerator,new_numerator,pending_stable_hits,pending_correction_hits,support_ratio,robust_shift_m,view_facing,temporal_depth_delta_m,temporal_world_delta_m,temporal_normal_delta_deg,historical_surface_distance_m,old_tsdf_residual,old_tsdf_history_weight,history_agreement,history_support_count,band_history_count,band_conflict_count,band_high_weight_conflict_count,band_conflict_ratio,band_high_weight_conflict_ratio,band_mean_residual,band_max_residual,vote_state,vote_score,vote_reasons,vote_enforced,normal_x,normal_y,normal_z,same_frame_depth_delta_m,same_frame_normal_delta_deg,cross_frame_clean_depth_delta_m,cross_frame_clean_lateral_m,cross_frame_clean_normal_delta_deg,cross_frame_clean_frame_gap,cross_frame_clean_weight");
        }
        catch (System.Exception error)
        {
            _contributionLedgerWriter = null;
            _contributionLedgerWriteFailures++;
            Debug.LogWarning($"[ScanCover] Contribution ledger could not start: {error.Message}", this);
        }
    }

    private void RecordContributionLedger(
        int index,
        string operation,
        float oldTsdf,
        int oldWeight,
        float observedTsdf,
        float observedWeight,
        float newTsdf,
        int newWeight)
    {
        if (index < 0)
            return;

        UpdateProvisionalTsdfLifecycle(index, operation);
        RecordVoxelWriteProvenance(index, operation, oldTsdf, oldWeight, newTsdf, newWeight);
        if (ShouldCancelFreeSpaceEvidence(index, operation, observedTsdf))
            CancelFreeSpaceEvidence(index);
        if (_contributionLedgerWriter == null)
            return;

        try
        {
            IndexToVoxel(index, out int x, out int y, out int z);
            Vector3 world = VoxelCenter(x, y, z);
            _contributionLedgerWriter.Write(_confidenceAuditCaptureIndex);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(LastRawFrameIndex);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_activeLedgerSampleIndex);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_activeLedgerPixelX);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_activeLedgerPixelY);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(index);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(x);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(y);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(z);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(world.x);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(world.y);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(world.z);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(operation);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(oldTsdf);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(oldWeight);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(observedTsdf);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(observedWeight);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(newTsdf);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(newWeight);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(oldTsdf * oldWeight);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(observedTsdf * observedWeight);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(newTsdf * newWeight);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(ReadAuditHit(_pendingTsdfHits, index));
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(ReadAuditHit(_correctionTsdfHits, index));
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSampleSupportRatio);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSampleRobustShiftMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSampleViewFacing);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditTemporalDepthDeltaMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditTemporalWorldDeltaMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditTemporalNormalDeltaDegrees);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditHistoricalSurfaceDistanceMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditOldTsdfResidual);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditOldTsdfWeight);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditHistoryAgreement);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditHistorySupportCount);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditBandHistoryCount);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditBandConflictCount);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditBandHighWeightConflictCount);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditBandConflictRatio);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditBandHighWeightConflictRatio);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditBandMeanResidual);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditBandMaxResidual);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditVoteState.ToString().ToUpperInvariant());
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditVoteScore);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditVoteReasons);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditVoteEnforced ? 1 : 0);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSampleNormal.x);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSampleNormal.y);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSampleNormal.z);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSameFrameDepthDeltaMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditSameFrameNormalDeltaDegrees);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditCrossFrameCleanDepthDeltaMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditCrossFrameCleanLateralMeters);
            _contributionLedgerWriter.Write(',');
            WriteLedgerFloat(_auditCrossFrameCleanNormalDeltaDegrees);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditCrossFrameCleanFrameGap);
            _contributionLedgerWriter.Write(',');
            _contributionLedgerWriter.Write(_auditCrossFrameCleanWeight);
            _contributionLedgerWriter.WriteLine();
            _contributionLedgerRowCount++;
        }
        catch (System.Exception error)
        {
            _contributionLedgerWriteFailures++;
            if (_contributionLedgerWriteFailures == 1)
                Debug.LogWarning($"[ScanCover] Contribution ledger write failed: {error.Message}", this);
        }
    }

    private void UpdateProvisionalTsdfLifecycle(int index, string operation)
    {
        if (_provisionalTsdf == null || _provisionalTsdfLastFrame == null || index < 0 || index >= _provisionalTsdf.Length)
            return;

        bool provisionalWrite =
            operation == "provisional_support" ||
            operation == "strong_sample_seed" ||
            operation == "same_frame_provisional" ||
            operation == "hole_side_repair" ||
            operation == "primary_plane_hole_band";

        if (provisionalWrite)
        {
            _provisionalTsdf[index] = 1;
            _provisionalTsdfLastFrame[index] = LastRawFrameIndex;
            if (_provisionalTsdfHits != null && index < _provisionalTsdfHits.Length)
                _provisionalTsdfHits[index] = (byte)Mathf.Min(255, _provisionalTsdfHits[index] + 1);
            return;
        }

        if (_provisionalTsdf[index] == 0)
            return;

        if (operation == "provisional_retire" || operation == "provisional_dirty_clear" ||
            operation == "primary_plane_hole_retire")
        {
            _provisionalTsdf[index] = 0;
            _provisionalTsdfLastFrame[index] = int.MinValue;
            if (_provisionalTsdfHits != null && index < _provisionalTsdfHits.Length)
                _provisionalTsdfHits[index] = 0;
            return;
        }

        _provisionalTsdf[index] = 0;
        _provisionalTsdfLastFrame[index] = int.MinValue;
        if (_provisionalTsdfHits != null && index < _provisionalTsdfHits.Length)
            _provisionalTsdfHits[index] = 0;
        LastProvisionalTsdfConfirmedCount++;
    }

    private void RecordVoxelWriteProvenance(
        int index,
        string operation,
        float oldTsdf,
        int oldWeight,
        float newTsdf,
        int newWeight)
    {
        if (_weights == null || index < 0 || index >= _weights.Length)
            return;

        bool hadPrior = _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance);
        if (!hadPrior)
            provenance.FirstFrame = LastRawFrameIndex;

        provenance.Frame = LastRawFrameIndex;
        provenance.Capture = _confidenceAuditCaptureIndex;
        provenance.Sample = _activeLedgerSampleIndex;
        provenance.PixelX = _activeLedgerPixelX;
        provenance.PixelY = _activeLedgerPixelY;
        provenance.WriteCount++;
        provenance.WriteSequence = ++_voxelWriteSequence;
        provenance.LastOperation = operation ?? "unknown";
        if (operation == "integrate")
            provenance.IntegrateCount++;
        else if (operation == "free_space_carve")
            provenance.CarveCount++;
        else if (operation == "replace" || operation == "guarded_replace")
            provenance.ReplaceCount++;
        else if (operation == "band_repair" || operation == "conflict_correct" || operation == "cleanup_neighbor")
            provenance.RepairCount++;

        Vector3 writeCamera = _activeTsdfSourceValid ? _activeTsdfSourceCamera : GetCameraPosition();
        Vector3 writeSurface = _activeTsdfSourceValid ? _activeTsdfSourceSurface : VoxelCenterFromIndex(index);
        provenance.CameraPosition = writeCamera;
        provenance.CameraRotation = Camera.main != null ? Camera.main.transform.rotation : Quaternion.identity;
        provenance.SurfacePoint = writeSurface;
        provenance.RayDirection = (writeSurface - writeCamera).normalized;
        provenance.SurfaceNormal = _auditSampleNormal;
        provenance.SurfaceDepth = Vector3.Distance(writeCamera, writeSurface);
        provenance.OldTsdf = oldTsdf;
        provenance.OldWeight = oldWeight;
        provenance.Tsdf = newTsdf;
        provenance.Weight = newWeight;
        _voxelWriteProvenance[index] = provenance;
        if (_hardAuditExpectedTsdf != null && index < _hardAuditExpectedTsdf.Length)
            _hardAuditExpectedTsdf[index] = newTsdf;
        if (_hardAuditExpectedWeights != null && index < _hardAuditExpectedWeights.Length)
            _hardAuditExpectedWeights[index] = (byte)Mathf.Clamp(newWeight, 0, 255);
    }

    private void ResetHardTsdfWriteAuditCounters()
    {
        LastHardAuditIntegrationCount = 0;
        LastHardAuditRetireCount = 0;
        LastHardAuditContinuityCount = 0;
        LastHardAuditExtractionCount = 0;
        LastHardAuditTsdfOnlyCount = 0;
        LastHardAuditWeightOnlyCount = 0;
        LastHardAuditBothCount = 0;
    }

    private void BeginHardTsdfWriteAuditCapture()
    {
        if (!enableHardTsdfWriteAudit || !writeHardTsdfAuditOnCapture)
        {
            _hardAuditRows = null;
            _hardAuditSampleRows = null;
            return;
        }

        _hardAuditCaptureIndex++;
        _hardAuditStem = $"hard_write_audit_{System.DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", AuditCulture)}_{_hardAuditCaptureIndex:D3}";
        _hardAuditRows = new StringBuilder(1024);
        _hardAuditRows.AppendLine("capture,frame,stage,changed,tsdf_only,weight_only,both,first_voxel,max_tsdf_delta,max_weight_delta");
        _hardAuditSampleRows = new StringBuilder(8192);
        _hardAuditSampleRows.AppendLine("capture,frame,stage,voxel_index,x,y,z,expected_tsdf,actual_tsdf,expected_weight,actual_weight,last_operation,last_write_sequence");
    }

    private void EndHardTsdfWriteAuditCapture()
    {
        if (_hardAuditRows == null || !writeHardTsdfAuditOnCapture)
            return;

        try
        {
            string directory = ResolveConfidenceAuditDirectory();
            Directory.CreateDirectory(directory);
            string summaryPath = Path.Combine(directory, _hardAuditStem + ".csv");
            string samplePath = Path.Combine(directory, _hardAuditStem + "_voxels.csv");
            File.WriteAllText(summaryPath, _hardAuditRows.ToString(), new UTF8Encoding(false));
            File.WriteAllText(samplePath, _hardAuditSampleRows != null ? _hardAuditSampleRows.ToString() : string.Empty, new UTF8Encoding(false));
            _lastHardAuditPath = summaryPath;
        }
        catch (System.Exception error)
        {
            _lastHardAuditPath = "WRITE FAILED";
            Debug.LogWarning($"[ScanCover] Hard write audit could not be written: {error.Message}", this);
        }
        finally
        {
            _hardAuditRows = null;
            _hardAuditSampleRows = null;
        }
    }

    private void AuditUntrackedTsdfWrites(string stage)
    {
        if (!enableHardTsdfWriteAudit || _tsdf == null || _weights == null)
            return;
        if (_hardAuditExpectedTsdf == null || _hardAuditExpectedTsdf.Length != _tsdf.Length ||
            _hardAuditExpectedWeights == null || _hardAuditExpectedWeights.Length != _weights.Length)
        {
            _hardAuditExpectedTsdf = (float[])_tsdf.Clone();
            _hardAuditExpectedWeights = (byte[])_weights.Clone();
            return;
        }

        int changed = 0;
        int tsdfOnly = 0;
        int weightOnly = 0;
        int both = 0;
        int firstVoxel = -1;
        int sampled = 0;
        float maxTsdfDelta = 0f;
        int maxWeightDelta = 0;
        for (int i = 0; i < _tsdf.Length; i++)
        {
            bool tsdfChanged = !Mathf.Approximately(_tsdf[i], _hardAuditExpectedTsdf[i]);
            bool weightChanged = _weights[i] != _hardAuditExpectedWeights[i];
            if (!tsdfChanged && !weightChanged)
                continue;

            changed++;
            if (firstVoxel < 0)
                firstVoxel = i;
            maxTsdfDelta = Mathf.Max(maxTsdfDelta, Mathf.Abs(_tsdf[i] - _hardAuditExpectedTsdf[i]));
            maxWeightDelta = Mathf.Max(maxWeightDelta, Mathf.Abs(_weights[i] - _hardAuditExpectedWeights[i]));
            if (tsdfChanged && weightChanged)
            {
                both++;
                LastHardAuditBothCount++;
            }
            else if (tsdfChanged)
            {
                tsdfOnly++;
                LastHardAuditTsdfOnlyCount++;
            }
            else
            {
                weightOnly++;
                LastHardAuditWeightOnlyCount++;
            }

            if (_hardAuditSampleRows != null && sampled < Mathf.Max(1, maxHardAuditVoxelSamplesPerStage))
            {
                IndexToVoxel(i, out int x, out int y, out int z);
                bool hasProvenance = _voxelWriteProvenance.TryGetValue(i, out VoxelWriteProvenance provenance);
                _hardAuditSampleRows
                    .Append(_hardAuditCaptureIndex).Append(',')
                    .Append(LastRawFrameIndex).Append(',')
                    .Append(stage).Append(',')
                    .Append(i).Append(',').Append(x).Append(',').Append(y).Append(',').Append(z).Append(',')
                    .Append(AuditFloat(_hardAuditExpectedTsdf[i])).Append(',')
                    .Append(AuditFloat(_tsdf[i])).Append(',')
                    .Append(_hardAuditExpectedWeights[i]).Append(',').Append(_weights[i]).Append(',')
                    .Append(hasProvenance ? provenance.LastOperation : "untracked").Append(',')
                    .Append(hasProvenance ? provenance.WriteSequence : 0)
                    .AppendLine();
                sampled++;
            }

            // Adopt the observed value after reporting it so a later checkpoint
            // cannot charge the same untracked mutation to a second stage.
            _hardAuditExpectedTsdf[i] = _tsdf[i];
            _hardAuditExpectedWeights[i] = _weights[i];
        }

        switch (stage)
        {
            case "integration": LastHardAuditIntegrationCount += changed; break;
            case "retire": LastHardAuditRetireCount += changed; break;
            case "continuity": LastHardAuditContinuityCount += changed; break;
            case "extraction": LastHardAuditExtractionCount += changed; break;
        }

        if (_hardAuditRows != null)
        {
            _hardAuditRows
                .Append(_hardAuditCaptureIndex).Append(',')
                .Append(LastRawFrameIndex).Append(',')
                .Append(stage).Append(',')
                .Append(changed).Append(',').Append(tsdfOnly).Append(',').Append(weightOnly).Append(',').Append(both).Append(',')
                .Append(firstVoxel).Append(',').Append(AuditFloat(maxTsdfDelta)).Append(',').Append(maxWeightDelta)
                .AppendLine();
        }
    }

    private void WriteLedgerFloat(float value)
    {
        _contributionLedgerWriter.Write(value.ToString("0.######", AuditCulture));
    }

    private void AppendConfidenceAuditFrame(int frameIndex, float latencyMs, float positionDelta, float angleDelta, Vector3 integrationCamera)
    {
        if (_confidenceAuditFrameRows == null)
            return;
        _auditMaxSnapshotLatencyMs = Mathf.Max(_auditMaxSnapshotLatencyMs, Mathf.Max(0f, latencyMs));
        _auditMaxPosePositionDelta = Mathf.Max(_auditMaxPosePositionDelta, Mathf.Max(0f, positionDelta));
        _auditMaxPoseAngleDelta = Mathf.Max(_auditMaxPoseAngleDelta, Mathf.Max(0f, angleDelta));
        Vector3 meanVector = _auditFrameReferenceCount > 0
            ? _auditFrameReferenceVectorSum / _auditFrameReferenceCount
            : Vector3.zero;
        float meanResidual = _auditFrameReferenceCount > 0
            ? _auditFrameReferenceMagnitudeSum / _auditFrameReferenceCount
            : 0f;
        float coherence = _auditFrameReferenceMagnitudeSum > 0.000001f
            ? _auditFrameReferenceVectorSum.magnitude / _auditFrameReferenceMagnitudeSum
            : 0f;
        _confidenceAuditFrameRows
            .Append(_confidenceAuditCaptureIndex).Append(',')
            .Append(frameIndex).Append(',')
            .Append(AuditFloat(latencyMs)).Append(',')
            .Append(AuditFloat(positionDelta)).Append(',')
            .Append(AuditFloat(angleDelta)).Append(',')
            .Append(AuditFloat(integrationCamera.x)).Append(',')
            .Append(AuditFloat(integrationCamera.y)).Append(',')
            .Append(AuditFloat(integrationCamera.z)).Append(',')
            .Append(_auditFrameReferenceCount).Append(',')
            .Append(AuditFloat(meanResidual)).Append(',')
            .Append(AuditFloat(coherence)).Append(',')
            .Append(AuditFloat(meanVector.x)).Append(',')
            .Append(AuditFloat(meanVector.y)).Append(',')
            .Append(AuditFloat(meanVector.z)).AppendLine();
    }

    private void ResetFrameReferenceAudit()
    {
        _auditFrameReferenceCount = 0;
        _auditFrameReferenceVectorSum = Vector3.zero;
        _auditFrameReferenceMagnitudeSum = 0f;
    }

    private void AccumulateFrameReferenceAudit(Vector3 point)
    {
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(point, out int cx, out int cy, out int cz))
            return;

        int minWeight = Mathf.Max(minConflictVoxelWeight, Mathf.CeilToInt(maxFusionWeight * 0.75f));
        float bestDistanceSq = float.PositiveInfinity;
        Vector3 bestCenter = Vector3.zero;
        for (int dz = -2; dz <= 2; dz++)
        for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
        {
            int x = cx + dx;
            int y = cy + dy;
            int z = cz + dz;
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;
            int index = Index(x, y, z);
            if (_weights[index] < minWeight || Mathf.Abs(_tsdf[index]) > rawDepthDebugNearSurfaceAbsTsdf)
                continue;
            Vector3 center = VoxelCenter(x, y, z);
            float distanceSq = (point - center).sqrMagnitude;
            if (distanceSq >= bestDistanceSq)
                continue;
            bestDistanceSq = distanceSq;
            bestCenter = center;
        }

        if (float.IsNaN(bestDistanceSq) || float.IsInfinity(bestDistanceSq) || bestDistanceSq > Mathf.Pow(voxelSizeMeters * 2.5f, 2f))
            return;
        Vector3 residual = point - bestCenter;
        _auditFrameReferenceVectorSum += residual;
        _auditFrameReferenceMagnitudeSum += residual.magnitude;
        _auditFrameReferenceCount++;
    }

    private bool ShouldAuditSample(int sampleIndex)
    {
        return writeConfidenceAuditOnCapture &&
            _confidenceAuditRows != null &&
            sampleIndex >= 0 &&
            sampleIndex % Mathf.Max(1, confidenceAuditSampleStride) == 0;
    }

    private void BeginConfidenceAuditSample(bool enabled)
    {
        // The same per-sample classification drives the overlay even when this row is not persisted.
        _auditSampleActive = true;
        _auditCenterAbsSampleTsdf = float.PositiveInfinity;
        _auditCenterSampleTsdf = 0f;
        _auditCenterOldTsdf = 0f;
        _auditCenterOldWeight = 0;
        _auditCenterStableHitsBefore = 0;
        _auditCenterStableHitsAfter = 0;
        _auditCenterCorrectionHitsBefore = 0;
        _auditCenterCorrectionHitsAfter = 0;
        _auditCenterSignFlip = false;
        _auditCenterResidual = 0f;
        _auditCenterOutcome = "none";
        _auditBandWritten = 0;
        _auditBandPendingStable = 0;
        _auditBandPendingCorrection = 0;
        _auditBandReplaced = 0;
        _auditBandCorrected = 0;
        _auditBandRejectedConflict = 0;
        _auditSampleSupportRatio = 0f;
        _auditSampleRobustShiftMeters = 0f;
        _auditSampleViewFacing = 0f;
        _auditSampleNormal = Vector3.zero;
        _auditTemporalDepthDeltaMeters = -1f;
        _auditTemporalWorldDeltaMeters = -1f;
        _auditTemporalNormalDeltaDegrees = -1f;
        _auditSameFrameDepthDeltaMeters = -1f;
        _auditSameFrameNormalDeltaDegrees = -1f;
        _auditCrossFrameCleanDepthDeltaMeters = -1f;
        _auditCrossFrameCleanLateralMeters = -1f;
        _auditCrossFrameCleanNormalDeltaDegrees = -1f;
        _auditCrossFrameCleanFrameGap = -1;
        _auditCrossFrameCleanWeight = 0;
        _auditHistoricalSurfaceDistanceMeters = -1f;
        _auditOldTsdfResidual = -1f;
        _auditOldTsdfWeight = 0;
        _auditHistoryAgreement = -1f;
        _auditHistorySupportCount = 0;
        _auditBandHistoryCount = 0;
        _auditBandConflictCount = 0;
        _auditBandHighWeightConflictCount = 0;
        _auditBandConflictRatio = -1f;
        _auditBandHighWeightConflictRatio = -1f;
        _auditBandMeanResidual = -1f;
        _auditBandMaxResidual = -1f;
        _auditVoteState = ObservationVoteState.Unassessed;
        _auditVoteScore = 0f;
        _auditVoteReasons = "unassessed";
        _auditVoteEnforced = false;
        _auditStrongSampleSeedWrite = false;
    }

    private void SetObservationVote(ObservationVoteState state, float score, string reasons, bool enforced = false)
    {
        if (_auditVoteState != ObservationVoteState.Unassessed)
            return;

        _auditVoteState = state;
        _auditVoteScore = Mathf.Clamp01(score);
        _auditVoteReasons = string.IsNullOrEmpty(reasons) ? "none" : reasons.Replace(',', '|');
        _auditVoteEnforced = enforced;
        switch (state)
        {
            case ObservationVoteState.Accept:
                _auditVoteAcceptCount++;
                break;
            case ObservationVoteState.Pending:
                _auditVotePendingCount++;
                break;
            case ObservationVoteState.Reject:
                _auditVoteRejectCount++;
                if (enforced)
                    _auditVoteEnforcedRejectCount++;
                break;
        }
    }

    private Vector3Int VoteHistoryKey(Vector3 point)
    {
        float inv = 1f / Mathf.Max(0.02f, voteCorrespondenceCellSizeMeters);
        return new Vector3Int(
            Mathf.FloorToInt(point.x * inv),
            Mathf.FloorToInt(point.y * inv),
            Mathf.FloorToInt(point.z * inv));
    }

    private void BeginVoteHistoryFrame()
    {
        _voteCurrentFrameHistory.Clear();
        _voteCurrentFrameRawHistory.Clear();
    }

    private void EndVoteHistoryFrame()
    {
        Dictionary<Vector3Int, VoteHistoryCell> previous = _votePreviousFrameHistory;
        _votePreviousFrameHistory = _voteCurrentFrameHistory;
        _voteCurrentFrameHistory = previous;
        _voteCurrentFrameHistory.Clear();
        _voteCurrentFrameRawHistory.Clear();
    }

    private void MeasureTemporalVoteFeatures(Vector3 cameraPosition, Vector3 point, Vector3 normal)
    {
        Vector3Int key = VoteHistoryKey(point);
        float bestDistance = Mathf.Max(0.02f, voteMaxCorrespondenceDistanceMeters);
        bool found = false;
        Vector3 bestPoint = Vector3.zero;
        Vector3 bestNormal = Vector3.zero;
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            Vector3Int candidateKey = key + new Vector3Int(dx, dy, dz);
            if (!_votePreviousFrameHistory.TryGetValue(candidateKey, out VoteHistoryCell cell) || cell.Count <= 0)
                continue;
            Vector3 candidate = cell.PositionSum / cell.Count;
            float distance = Vector3.Distance(point, candidate);
            if (distance >= bestDistance)
                continue;
            bestDistance = distance;
            bestPoint = candidate;
            bestNormal = cell.NormalSum.sqrMagnitude > 0.0001f ? cell.NormalSum.normalized : Vector3.zero;
            found = true;
        }

        if (found)
        {
            Vector3 ray = point - cameraPosition;
            ray = ray.sqrMagnitude > 0.0001f ? ray.normalized : Vector3.forward;
            Vector3 delta = point - bestPoint;
            _auditTemporalDepthDeltaMeters = Mathf.Abs(Vector3.Dot(delta, ray));
            _auditTemporalWorldDeltaMeters = delta.magnitude;
            if (normal.sqrMagnitude > 0.0001f && bestNormal.sqrMagnitude > 0.0001f)
                _auditTemporalNormalDeltaDegrees = Vector3.Angle(normal, bestNormal);
        }

        float sameFrameSearch = Mathf.Max(0.03f, sameFrameConflictSearchMeters);
        float sameFrameNormalLimit = Mathf.Clamp(sameFrameConflictMaxNormalAngleDegrees, 1f, 45f);
        for (int dz = -2; dz <= 2; dz++)
        for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
        {
            Vector3Int candidateKey = key + new Vector3Int(dx, dy, dz);
            if (!_voteCurrentFrameRawHistory.TryGetValue(candidateKey, out VoteHistoryCell cell) || cell.Count <= 0)
                continue;
            Vector3 candidate = cell.PositionSum / cell.Count;
            if (Vector3.Distance(point, candidate) > sameFrameSearch)
                continue;
            Vector3 candidateNormal = cell.NormalSum.sqrMagnitude > 0.0001f ? cell.NormalSum.normalized : Vector3.zero;
            float normalAngle = normal.sqrMagnitude > 0.0001f && candidateNormal.sqrMagnitude > 0.0001f
                ? Vector3.Angle(normal, candidateNormal)
                : 0f;
            if (normalAngle > sameFrameNormalLimit)
                continue;
            Vector3 planeNormal = candidateNormal.sqrMagnitude > 0.0001f
                ? candidateNormal
                : (normal.sqrMagnitude > 0.0001f ? normal.normalized : Vector3.up);
            float depthDelta = Mathf.Abs(Vector3.Dot(point - candidate, planeNormal));
            if (depthDelta <= _auditSameFrameDepthDeltaMeters)
                continue;
            _auditSameFrameDepthDeltaMeters = depthDelta;
            _auditSameFrameNormalDeltaDegrees = normalAngle;
        }
    }

    private void RecordVoteHistorySample(Vector3 point, Vector3 normal)
    {
        Vector3Int key = VoteHistoryKey(point);
        _voteCurrentFrameHistory.TryGetValue(key, out VoteHistoryCell current);
        current.PositionSum += point;
        current.NormalSum += normal;
        current.Count++;
        _voteCurrentFrameHistory[key] = current;
    }

    private void RecordCurrentFrameRawVoteSample(Vector3 point, Vector3 normal)
    {
        Vector3Int key = VoteHistoryKey(point);
        _voteCurrentFrameRawHistory.TryGetValue(key, out VoteHistoryCell current);
        current.PositionSum += point;
        current.NormalSum += normal;
        current.Count++;
        _voteCurrentFrameRawHistory[key] = current;
    }

    private void MarkSurfaceObservationState(Vector3 point, byte state)
    {
        if (_surfaceObservationState == null || _surfaceObservationLastFrame == null || !Finite(point))
            return;
        if (!TryWorldToVoxel(point, out int x, out int y, out int z))
            return;
        int index = Index(x, y, z);
        _surfaceObservationState[index] = state;
        _surfaceObservationLastFrame[index] = LastRawFrameIndex;
    }

    private void MarkSurfaceObservationVoxel(int index, byte state)
    {
        if (state == 0 || _surfaceObservationState == null || _surfaceObservationLastFrame == null ||
            index < 0 || index >= _surfaceObservationState.Length)
            return;
        _surfaceObservationState[index] = state;
        _surfaceObservationLastFrame[index] = LastRawFrameIndex;
    }

    private void MeasureCrossFrameCleanSurfaceConflict(Vector3 point, Vector3 normal)
    {
        if (_tsdf == null ||
            _weights == null ||
            _voxelWriteProvenance == null ||
            _voxelWriteProvenance.Count <= 0 ||
            normal.sqrMagnitude < 0.0001f ||
            !TryWorldToVoxel(point, out int cx, out int cy, out int cz))
        {
            return;
        }

        Vector3 sampleNormal = normal.normalized;
        float searchMeters = Mathf.Max(0.03f, crossFrameCleanConflictSearchMeters);
        int radius = Mathf.Clamp(Mathf.CeilToInt(searchMeters / Mathf.Max(0.001f, voxelSizeMeters)), 1, 5);
        float minPlane = Mathf.Max(0.01f, crossFrameCleanConflictDepthMeters);
        float maxLateral = Mathf.Max(
            voxelSizeMeters * 0.25f,
            voxelSizeMeters * Mathf.Max(0.25f, crossFrameCleanConflictMaxLateralVoxelScale));
        float normalLimit = Mathf.Clamp(crossFrameCleanConflictMaxNormalAngleDegrees, 1f, 60f);

        for (int dz = -radius; dz <= radius; dz++)
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = cx + dx;
            int y = cy + dy;
            int z = cz + dz;
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;

            int index = Index(x, y, z);
            int oldWeight = _weights[index];
            if (oldWeight < minSurfaceCornerWeight || Mathf.Abs(_tsdf[index]) > maxCleanLightCoverAbsTsdf)
                continue;
            if (!_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance))
                continue;
            if (provenance.Frame == LastRawFrameIndex || provenance.IntegrateCount <= 0)
                continue;
            if (provenance.SurfaceNormal.sqrMagnitude < 0.0001f)
                continue;

            Vector3 sourceNormal = provenance.SurfaceNormal.normalized;
            float normalAngle = Vector3.Angle(sampleNormal, sourceNormal);
            if (normalAngle > 90f)
            {
                sourceNormal = -sourceNormal;
                normalAngle = 180f - normalAngle;
            }
            if (normalAngle > normalLimit)
                continue;

            Vector3 planeNormal = (sampleNormal + sourceNormal).normalized;
            if (planeNormal.sqrMagnitude < 0.0001f)
                planeNormal = sampleNormal;

            Vector3 delta = point - provenance.SurfacePoint;
            float planeSeparation = Mathf.Abs(Vector3.Dot(delta, planeNormal));
            if (planeSeparation < minPlane)
                continue;
            float lateralSq = Mathf.Max(0f, delta.sqrMagnitude - planeSeparation * planeSeparation);
            if (lateralSq > maxLateral * maxLateral)
                continue;

            if (planeSeparation <= _auditCrossFrameCleanDepthDeltaMeters)
                continue;
            _auditCrossFrameCleanDepthDeltaMeters = planeSeparation;
            _auditCrossFrameCleanLateralMeters = Mathf.Sqrt(lateralSq);
            _auditCrossFrameCleanNormalDeltaDegrees = normalAngle;
            _auditCrossFrameCleanFrameGap = Mathf.Abs(LastRawFrameIndex - provenance.Frame);
            _auditCrossFrameCleanWeight = oldWeight;
        }
    }

    private void MeasureHistoricalTsdfVoteFeatures(Vector3 point)
    {
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(point, out int cx, out int cy, out int cz))
            return;

        int centerIndex = Index(cx, cy, cz);
        if (_weights[centerIndex] > 0)
        {
            _auditOldTsdfWeight = _weights[centerIndex];
            _auditOldTsdfResidual = Mathf.Abs(_tsdf[centerIndex]);
        }

        int radius = Mathf.Clamp(voteHistoryNeighborhoodRadiusVoxels, 1, 4);
        int historyCount = 0;
        int agreementCount = 0;
        float nearest = float.PositiveInfinity;
        float agreementDistance = Mathf.Max(
            voteGoodHistoricalSurfaceDistanceMeters,
            voxelSizeMeters * 1.5f);
        for (int dz = -radius; dz <= radius; dz++)
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = cx + dx;
            int y = cy + dy;
            int z = cz + dz;
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;
            int index = Index(x, y, z);
            if (_weights[index] < minSurfaceCornerWeight || Mathf.Abs(_tsdf[index]) > maxCleanLightCoverAbsTsdf)
                continue;

            float distance = Vector3.Distance(point, VoxelCenter(x, y, z));
            nearest = Mathf.Min(nearest, distance);
            historyCount++;
            if (distance <= agreementDistance)
                agreementCount++;
        }

        _auditHistorySupportCount = historyCount;
        if (historyCount > 0)
        {
            _auditHistoricalSurfaceDistanceMeters = nearest;
            _auditHistoryAgreement = (float)agreementCount / historyCount;
        }
    }

    private void MeasureTsdfBandVoteFeatures(
        Vector3 cameraPosition,
        Vector3 point,
        Vector3 normal,
        int halfBandSteps)
    {
        float residualSum = 0f;
        if (useProjectiveTsdfIntegration)
        {
            Vector3 toSurface = point - cameraPosition;
            float surfaceDepth = toSurface.magnitude;
            if (surfaceDepth <= 0.0001f)
                return;
            Vector3 ray = toSurface / surfaceDepth;
            float voxel = Mathf.Max(0.0001f, voxelSizeMeters);
            float startDepth = Mathf.Max(0.01f, surfaceDepth - halfBandSteps * voxel);
            float endDepth = surfaceDepth + halfBandSteps * voxel;
            for (float voxelDepth = startDepth; voxelDepth <= endDepth + voxel * 0.25f; voxelDepth += voxel)
            {
                Vector3 world = cameraPosition + ray * voxelDepth;
                float sampleTsdf = Mathf.Clamp((surfaceDepth - voxelDepth) / truncationMeters, -1f, 1f);
                AccumulateTsdfBandVoteVoxel(world, sampleTsdf, ref residualSum);
            }
        }
        else
        {
            for (int step = -halfBandSteps; step <= halfBandSteps; step++)
            {
                float signedDistance = step * voxelSizeMeters;
                Vector3 world = point + normal * signedDistance;
                float sampleTsdf = Mathf.Clamp(signedDistance / truncationMeters, -1f, 1f);
                AccumulateTsdfBandVoteVoxel(world, sampleTsdf, ref residualSum);
            }
        }

        if (_auditBandHistoryCount <= 0)
            return;
        _auditBandConflictRatio = (float)_auditBandConflictCount / _auditBandHistoryCount;
        _auditBandHighWeightConflictRatio = (float)_auditBandHighWeightConflictCount / _auditBandHistoryCount;
        _auditBandMeanResidual = residualSum / _auditBandHistoryCount;
    }

    private void AccumulateTsdfBandVoteVoxel(Vector3 world, float sampleTsdf, ref float residualSum)
    {
        if (!TryWorldToVoxel(world, out int x, out int y, out int z))
            return;
        int index = Index(x, y, z);
        int oldWeight = _weights[index];
        if (oldWeight <= 0)
            return;

        float residual = Mathf.Abs(_tsdf[index] - sampleTsdf);
        _auditBandHistoryCount++;
        residualSum += residual;
        _auditBandMaxResidual = Mathf.Max(_auditBandMaxResidual, residual);
        bool signConflict =
            Mathf.Sign(_tsdf[index]) != Mathf.Sign(sampleTsdf) &&
            residual >= tsdfConflictThreshold;
        if (!signConflict)
            return;
        _auditBandConflictCount++;
        if (oldWeight >= minConflictVoxelWeight)
            _auditBandHighWeightConflictCount++;
    }

    private static float VoteScoreOrNeutral(float measured, float bad, float good, bool higherIsBetter)
    {
        if (measured < 0f)
            return 0.5f;
        float low = Mathf.Min(bad, good);
        float high = Mathf.Max(bad, good);
        float score = Mathf.InverseLerp(low, high, measured);
        return higherIsBetter ? score : 1f - score;
    }

    private ObservationVoteState EvaluateObservationVote(float sampleWeight)
    {
        float supportScore = Mathf.InverseLerp(
            Mathf.Min(voteBadSupportRatio, voteGoodSupportRatio),
            Mathf.Max(voteBadSupportRatio, voteGoodSupportRatio),
            _auditSampleSupportRatio);
        float robustScore = 1f - Mathf.InverseLerp(
            Mathf.Min(voteGoodRobustShiftMeters, voteBadRobustShiftMeters),
            Mathf.Max(voteGoodRobustShiftMeters, voteBadRobustShiftMeters),
            _auditSampleRobustShiftMeters);
        float viewScore = Mathf.InverseLerp(
            Mathf.Min(voteBadViewFacing, voteGoodViewFacing),
            Mathf.Max(voteBadViewFacing, voteGoodViewFacing),
            _auditSampleViewFacing);
        float weightScore = Mathf.InverseLerp(
            Mathf.Min(voteBadSampleWeight, voteGoodSampleWeight),
            Mathf.Max(voteBadSampleWeight, voteGoodSampleWeight),
            sampleWeight);
        float temporalDepthScore = VoteScoreOrNeutral(
            _auditTemporalDepthDeltaMeters,
            voteGoodTemporalDepthDeltaMeters,
            voteBadTemporalDepthDeltaMeters,
            false);
        float historicalSurfaceScore = VoteScoreOrNeutral(
            _auditHistoricalSurfaceDistanceMeters,
            voteGoodHistoricalSurfaceDistanceMeters,
            voteBadHistoricalSurfaceDistanceMeters,
            false);
        float oldTsdfScore = VoteScoreOrNeutral(
            _auditOldTsdfResidual,
            voteGoodOldTsdfResidual,
            voteBadOldTsdfResidual,
            false);
        float historyAgreementScore = VoteScoreOrNeutral(
            _auditHistoryAgreement,
            voteBadHistoryAgreement,
            voteGoodHistoryAgreement,
            true);
        float bandConflictRisk = Mathf.Max(
            _auditBandConflictRatio,
            _auditBandHighWeightConflictRatio);
        float bandConflictScore = VoteScoreOrNeutral(
            bandConflictRisk,
            voteGoodBandConflictRatio,
            voteBadBandConflictRatio,
            false);
        float bandResidualScore = VoteScoreOrNeutral(
            _auditBandMeanResidual,
            voteGoodBandMeanResidual,
            voteBadBandMeanResidual,
            false);
        float score =
            supportScore * 0.05f +
            robustScore * 0.05f +
            viewScore * 0.05f +
            weightScore * 0.025f +
            temporalDepthScore * 0.20f +
            historicalSurfaceScore * 0.15f +
            oldTsdfScore * 0.15f +
            historyAgreementScore * 0.025f +
            bandConflictScore * 0.225f +
            bandResidualScore * 0.075f;

        StringBuilder reasons = new StringBuilder(48);
        if (supportScore < 0.5f)
            reasons.Append("weak_support|");
        if (robustScore < 0.5f)
            reasons.Append("depth_shift|");
        if (viewScore < 0.5f)
            reasons.Append("grazing_view|");
        if (weightScore < 0.5f)
            reasons.Append("low_weight|");
        if (temporalDepthScore < 0.5f)
            reasons.Append("temporal_depth_jump|");
        if (historicalSurfaceScore < 0.5f)
            reasons.Append("off_history_surface|");
        if (oldTsdfScore < 0.5f)
            reasons.Append("old_tsdf_residual|");
        if (historyAgreementScore < 0.5f)
            reasons.Append("weak_history|");
        if (bandConflictScore < 0.5f)
            reasons.Append("band_conflict|");
        if (bandResidualScore < 0.5f)
            reasons.Append("band_residual|");
        bool sameFrameConflict = HasSameFrameEntryConflict();
        bool sameFrameProvisional = SameFrameEntryCanUseProvisional(sampleWeight);
        if (sameFrameConflict)
            reasons.Append(sameFrameProvisional ? "same_frame_provisional|" : "same_frame_depth_conflict|");
        bool crossFrameCleanConflict =
            rejectCrossFrameCleanSurfaceConflict &&
            _auditCrossFrameCleanDepthDeltaMeters >= Mathf.Max(0.01f, crossFrameCleanConflictDepthMeters);
        if (crossFrameCleanConflict)
            reasons.Append("cross_frame_clean_conflict|");
        bool strongCurrentEvidence =
            _auditSampleSupportRatio >= Mathf.Clamp01(voteGoodSupportRatio) &&
            _auditSampleRobustShiftMeters <= Mathf.Max(0.001f, voteGoodRobustShiftMeters) &&
            _auditSampleViewFacing >= Mathf.Clamp01(voteGoodViewFacing) &&
            sampleWeight >= Mathf.Clamp01(voteGoodSampleWeight);
        bool strongCurrentAccept =
            strongCurrentEvidence &&
            !sameFrameConflict &&
            !crossFrameCleanConflict &&
            reasons.Length == 0;
        if (reasons.Length == 0)
            reasons.Append("clean");
        else
            reasons.Length--;

        ObservationVoteState state = (sameFrameConflict && !sameFrameProvisional) || crossFrameCleanConflict
            ? ObservationVoteState.Reject
            : (strongCurrentAccept || score >= Mathf.Max(voteAcceptScore, voteRejectScore)
                ? ObservationVoteState.Accept
                : (score < Mathf.Min(voteAcceptScore, voteRejectScore)
                    ? ObservationVoteState.Reject
                    : ObservationVoteState.Pending));
        if (sameFrameProvisional && state == ObservationVoteState.Accept)
            state = ObservationVoteState.Pending;
        bool enforced = state == ObservationVoteState.Reject &&
            (enforceObservationWriteGate || observationVoteMode == ObservationVoteMode.RejectOnly);
        SetObservationVote(
            state,
            score,
            strongCurrentAccept ? "strong_current_clean" : reasons.ToString(),
            enforced);
        if (!sameFrameConflict)
            LastSameFrameEntryStableCount++;
        else if (sameFrameProvisional)
            LastSameFrameEntryProvisionalCount++;
        else
            LastSameFrameEntryRejectedCount++;
        if (sameFrameConflict)
            LastVoteSameFrameConflictRejectCount++;
        if (crossFrameCleanConflict)
            LastVoteCrossFrameCleanConflictRejectCount++;
        if (strongCurrentAccept)
            LastVoteStrongCurrentAcceptCount++;
        return state;
    }

    private bool HasSameFrameEntryConflict()
    {
        return _auditSameFrameDepthDeltaMeters >= Mathf.Max(0.01f, sameFrameConflictDepthMeters) &&
               (_auditSameFrameNormalDeltaDegrees < 0f ||
                _auditSameFrameNormalDeltaDegrees <= Mathf.Clamp(sameFrameConflictMaxNormalAngleDegrees, 1f, 45f));
    }

    private bool SameFrameEntryCanUseProvisional(float sampleWeight)
    {
        if (!enableSameFrameEntryGate || !allowProvisionalTsdfSupportWrites || !HasSameFrameEntryConflict())
            return false;
        if (_auditSameFrameDepthDeltaMeters > Mathf.Max(sameFrameConflictDepthMeters, maxSameFrameEntryProvisionalDepthMeters))
            return false;
        bool bandSafe =
            _auditBandConflictRatio < 0f ||
            _auditBandConflictRatio <= Mathf.Clamp01(voteGoodBandConflictRatio);
        return bandSafe &&
               _auditSampleSupportRatio >= Mathf.Clamp01(voteGoodSupportRatio) &&
               _auditSampleRobustShiftMeters <= Mathf.Max(0.001f, voteGoodRobustShiftMeters * 2f) &&
               _auditSampleViewFacing >= Mathf.Clamp01(voteGoodViewFacing) &&
               sampleWeight >= Mathf.Clamp01(voteBadSampleWeight);
    }

    private bool PendingObservationHasWriteSupport(float sampleWeight, out bool bootstrapWrite, out bool sameFrameProvisionalWrite, out bool strongSampleSeedWrite)
    {
        bootstrapWrite = false;
        sameFrameProvisionalWrite = false;
        strongSampleSeedWrite = false;
        bool temporalConfirmed =
            _auditTemporalDepthDeltaMeters >= 0f &&
            _auditTemporalDepthDeltaMeters <= Mathf.Max(0.001f, voteGoodTemporalDepthDeltaMeters);
        bool historyConfirmed =
            _auditHistorySupportCount > 0 &&
            _auditHistoryAgreement >= Mathf.Clamp01(voteGoodHistoryAgreement) &&
            (_auditHistoricalSurfaceDistanceMeters < 0f ||
             _auditHistoricalSurfaceDistanceMeters <= Mathf.Max(0.001f, voteGoodHistoricalSurfaceDistanceMeters));
        bool bandSafe =
            _auditBandConflictRatio < 0f ||
            _auditBandConflictRatio <= Mathf.Clamp01(voteGoodBandConflictRatio);
        bool sameFrameSafe =
            _auditSameFrameDepthDeltaMeters < 0f ||
            _auditSameFrameDepthDeltaMeters < Mathf.Max(0.01f, sameFrameConflictDepthMeters);
        bool crossFrameCleanSafe =
            !rejectCrossFrameCleanSurfaceConflict ||
            _auditCrossFrameCleanDepthDeltaMeters < 0f ||
            _auditCrossFrameCleanDepthDeltaMeters < Mathf.Max(0.01f, crossFrameCleanConflictDepthMeters);
        bool strongLocalObservation =
            _auditSampleSupportRatio >= Mathf.Clamp01(voteGoodSupportRatio) &&
            _auditSampleRobustShiftMeters <= Mathf.Max(0.001f, voteGoodRobustShiftMeters) &&
            _auditSampleViewFacing >= Mathf.Clamp01(voteGoodViewFacing) &&
            sampleWeight >= Mathf.Clamp01(voteGoodSampleWeight);
        bool oldTsdfSafe =
            _auditOldTsdfWeight <= 0 ||
            _auditOldTsdfResidual < 0f ||
            _auditOldTsdfResidual <= Mathf.Clamp01(maxStrongSampleSeedOldTsdfResidual);
        bool strongSeedAllowed =
            allowStrongSampleSeedWrites &&
            strongLocalObservation &&
            bandSafe &&
            sameFrameSafe &&
            crossFrameCleanSafe &&
            oldTsdfSafe;
        bool bootstrapAllowed =
            allowPendingBootstrapSeeds &&
            IntegratedFrameCount <= 0 &&
            strongLocalObservation &&
            bandSafe &&
            sameFrameSafe &&
            crossFrameCleanSafe;
        if (strongSeedAllowed)
        {
            strongSampleSeedWrite = true;
            bootstrapWrite = bootstrapAllowed;
            return true;
        }
        if (allowStrongSampleSeedWrites && strongLocalObservation)
            LastStrongSampleSeedBlockedCount++;

        if (bootstrapAllowed)
        {
            bootstrapWrite = true;
            return true;
        }

        if (SameFrameEntryCanUseProvisional(sampleWeight))
        {
            sameFrameProvisionalWrite = true;
            return true;
        }

        return (temporalConfirmed || historyConfirmed) &&
               bandSafe &&
               sameFrameSafe &&
               crossFrameCleanSafe &&
               _auditSampleSupportRatio >= Mathf.Clamp01(voteGoodSupportRatio);
    }

    private RawDepthDebugKind RawDepthKindFromAuditOutcome()
    {
        if (_auditCenterOutcome == "replaced" ||
            _auditCenterOutcome == "corrected" ||
            _auditCenterOutcome == "rejected_conflict" ||
            _auditCenterOutcome == "pending_correction")
        {
            switch (AuditCenterConflictCause())
            {
                case "depth_edge":
                    return RawDepthDebugKind.ConflictEdge;
                case "view_angle":
                    return RawDepthDebugKind.ConflictView;
                case "old_weight_locked":
                    return RawDepthDebugKind.ConflictLocked;
                default:
                    return RawDepthDebugKind.Pending;
            }
        }
        if (_auditCenterOutcome == "pending_stability" || _auditBandPendingStable > 0)
            return RawDepthDebugKind.StableBand;
        return RawDepthDebugKind.Accepted;
    }

    private string AuditCenterConflictCause()
    {
        if (!_auditCenterSignFlip && _auditCenterResidual < tsdfConflictThreshold)
            return "none";
        if (_auditSampleSupportRatio < Mathf.Clamp01(auditDepthEdgeSupportRatio) ||
            _auditSampleRobustShiftMeters >= Mathf.Max(0.001f, auditRobustShiftRiskMeters))
            return "depth_edge";
        if (_auditSampleViewFacing > 0f &&
            _auditSampleViewFacing < Mathf.Clamp01(auditGrazingViewFacingThreshold))
            return "view_angle";
        if (_auditCenterOldWeight >= Mathf.Max(1, auditOldLockedWeightThreshold))
            return "old_weight_locked";
        return "unclassified";
    }

    private void AppendConfidenceAuditRow(
        int frameIndex,
        int sampleIndex,
        int pixelX,
        int pixelY,
        Vector3 point,
        float sampleWeight,
        int checkedNeighbors,
        int consistentNeighbors,
        float robustShiftMeters,
        float oldTsdf,
        int oldWeight,
        int pendingHits,
        string decision,
        string reason,
        bool enabled)
    {
        if (!enabled || _confidenceAuditRows == null)
            return;
        if (_confidenceAuditRowCount >= Mathf.Max(1000, maxConfidenceAuditRowsPerCapture))
        {
            _confidenceAuditDroppedRows++;
            return;
        }

        float supportRatio = checkedNeighbors > 0 ? (float)consistentNeighbors / checkedNeighbors : 0f;
        _confidenceAuditRows
            .Append(_confidenceAuditCaptureIndex).Append(',')
            .Append(frameIndex).Append(',')
            .Append(sampleIndex).Append(',')
            .Append(pixelX).Append(',')
            .Append(pixelY).Append(',')
            .Append(AuditFloat(point.x)).Append(',')
            .Append(AuditFloat(point.y)).Append(',')
            .Append(AuditFloat(point.z)).Append(',')
            .Append(AuditFloat(sampleWeight)).Append(',')
            .Append(checkedNeighbors).Append(',')
            .Append(consistentNeighbors).Append(',')
            .Append(AuditFloat(supportRatio)).Append(',')
            .Append(AuditFloat(robustShiftMeters)).Append(',')
            .Append(AuditFloat(_auditSampleViewFacing)).Append(',')
            .Append(AuditFloat(_auditTemporalDepthDeltaMeters)).Append(',')
            .Append(AuditFloat(_auditTemporalWorldDeltaMeters)).Append(',')
            .Append(AuditFloat(_auditTemporalNormalDeltaDegrees)).Append(',')
            .Append(AuditFloat(_auditHistoricalSurfaceDistanceMeters)).Append(',')
            .Append(AuditFloat(_auditOldTsdfResidual)).Append(',')
            .Append(_auditOldTsdfWeight).Append(',')
            .Append(AuditFloat(_auditHistoryAgreement)).Append(',')
            .Append(_auditHistorySupportCount).Append(',')
            .Append(_auditBandHistoryCount).Append(',')
            .Append(_auditBandConflictCount).Append(',')
            .Append(_auditBandHighWeightConflictCount).Append(',')
            .Append(AuditFloat(_auditBandConflictRatio)).Append(',')
            .Append(AuditFloat(_auditBandHighWeightConflictRatio)).Append(',')
            .Append(AuditFloat(_auditBandMeanResidual)).Append(',')
            .Append(AuditFloat(_auditBandMaxResidual)).Append(',')
            .Append(_auditVoteState.ToString().ToUpperInvariant()).Append(',')
            .Append(AuditFloat(_auditVoteScore)).Append(',')
            .Append(_auditVoteReasons).Append(',')
            .Append(_auditVoteEnforced ? 1 : 0).Append(',')
            .Append(AuditFloat(oldTsdf)).Append(',')
            .Append(oldWeight).Append(',')
            .Append(pendingHits).Append(',')
            .Append(decision).Append(',')
            .Append(reason).Append(',')
            .Append(AuditFloat(_auditCenterSampleTsdf)).Append(',')
            .Append(AuditFloat(_auditCenterOldTsdf)).Append(',')
            .Append(_auditCenterOldWeight).Append(',')
            .Append(_auditCenterStableHitsBefore).Append(',')
            .Append(_auditCenterStableHitsAfter).Append(',')
            .Append(_auditCenterCorrectionHitsBefore).Append(',')
            .Append(_auditCenterCorrectionHitsAfter).Append(',')
            .Append(_auditCenterSignFlip ? 1 : 0).Append(',')
            .Append(AuditFloat(_auditCenterResidual)).Append(',')
            .Append(_auditCenterOutcome).Append(',')
            .Append(AuditCenterConflictCause()).Append(',')
            .Append(_auditBandWritten).Append(',')
            .Append(_auditBandPendingStable).Append(',')
            .Append(_auditBandPendingCorrection).Append(',')
            .Append(_auditBandReplaced).Append(',')
            .Append(_auditBandCorrected).Append(',')
            .Append(_auditBandRejectedConflict).Append(',')
            .Append(AuditFloat(_auditSameFrameDepthDeltaMeters)).Append(',')
            .Append(AuditFloat(_auditSameFrameNormalDeltaDegrees)).Append(',')
            .Append(AuditFloat(_auditCrossFrameCleanDepthDeltaMeters)).Append(',')
            .Append(AuditFloat(_auditCrossFrameCleanLateralMeters)).Append(',')
            .Append(AuditFloat(_auditCrossFrameCleanNormalDeltaDegrees)).Append(',')
            .Append(_auditCrossFrameCleanFrameGap).Append(',')
            .Append(_auditCrossFrameCleanWeight)
            .AppendLine();
        _confidenceAuditRowCount++;
    }

    private void EndConfidenceAuditCapture(int integratedFrames, string status)
    {
        EndSurfaceProfileDiagnostics(integratedFrames, status);
        EndHardTsdfWriteAuditCapture();
        if (!writeConfidenceAuditOnCapture || _confidenceAuditRows == null)
            return;

        try
        {
            Directory.CreateDirectory(_confidenceAuditDirectory);
            string csvPath = Path.Combine(_confidenceAuditDirectory, _confidenceAuditStem + "_samples.csv");
            string framesPath = Path.Combine(_confidenceAuditDirectory, _confidenceAuditStem + "_frames.csv");
            string lifecyclePath = Path.Combine(_confidenceAuditDirectory, _confidenceAuditStem + "_voxel_lifecycle.csv");
            string doubleLayerPairsPath = Path.Combine(_confidenceAuditDirectory, _confidenceAuditStem + "_double_layer_pairs.csv");
            string summaryPath = Path.Combine(_confidenceAuditDirectory, _confidenceAuditStem + "_summary.txt");
            if (_contributionLedgerWriter != null)
            {
                _contributionLedgerWriter.Flush();
                _contributionLedgerWriter.Dispose();
                _contributionLedgerWriter = null;
            }
            AnalyzeAuditMeshAssociation();
            File.WriteAllText(csvPath, _confidenceAuditRows.ToString(), new UTF8Encoding(false));
            File.WriteAllText(framesPath, _confidenceAuditFrameRows != null ? _confidenceAuditFrameRows.ToString() : string.Empty, new UTF8Encoding(false));
            File.WriteAllText(lifecyclePath, BuildVoxelAuditLifecycleCsv(), new UTF8Encoding(false));
            File.WriteAllText(doubleLayerPairsPath, _doubleLayerPairRows != null ? _doubleLayerPairRows.ToString() : string.Empty, new UTF8Encoding(false));

            StringBuilder summary = new StringBuilder(1024);
            summary.AppendLine("ScanCover confidence audit");
            summary.AppendLine("status=" + status);
            summary.AppendLine("capture=" + _confidenceAuditCaptureIndex);
            summary.AppendLine("integrated_frames=" + integratedFrames);
            summary.AppendLine("observed_frames=" + _confidenceAuditIntegratedFrames);
            summary.AppendLine("rows=" + _confidenceAuditRowCount);
            summary.AppendLine("dropped_rows=" + _confidenceAuditDroppedRows);
            summary.AppendLine("sample_stride=" + Mathf.Max(1, confidenceAuditSampleStride));
            summary.AppendLine("vote_mode=" + observationVoteMode);
            summary.AppendLine("vote_accept=" + _auditVoteAcceptCount);
            summary.AppendLine("vote_pending=" + _auditVotePendingCount);
            summary.AppendLine("vote_reject=" + _auditVoteRejectCount);
            summary.AppendLine("vote_enforced_reject=" + _auditVoteEnforcedRejectCount);
            summary.AppendLine("vote_write_gate_enabled=" + (enforceObservationWriteGate ? 1 : 0));
            summary.AppendLine("vote_same_frame_conflict_reject=" + LastVoteSameFrameConflictRejectCount);
            summary.AppendLine("vote_cross_frame_clean_conflict_reject=" + LastVoteCrossFrameCleanConflictRejectCount);
            summary.AppendLine("same_frame_entry_gate_enabled=" + (enableSameFrameEntryGate ? 1 : 0));
            summary.AppendLine("same_frame_entry_stable=" + LastSameFrameEntryStableCount);
            summary.AppendLine("same_frame_entry_provisional=" + LastSameFrameEntryProvisionalCount);
            summary.AppendLine("same_frame_entry_provisional_write=" + LastSameFrameEntryProvisionalWriteCount);
            summary.AppendLine("same_frame_entry_rejected=" + LastSameFrameEntryRejectedCount);
            summary.AppendLine("same_frame_entry_held=" + LastSameFrameEntryHeldCount);
            summary.AppendLine("same_frame_entry_max_provisional_depth_m=" + AuditFloat(maxSameFrameEntryProvisionalDepthMeters));
            summary.AppendLine("same_frame_entry_provisional_weight_scale=" + AuditFloat(sameFrameEntryProvisionalWeightScale));
            summary.AppendLine("formal_integrate_gate_enabled=" + (gateFormalIntegrateWrites ? 1 : 0));
            summary.AppendLine("formal_integrate_write=" + LastFormalIntegrateWriteCount);
            summary.AppendLine("formal_integrate_block=" + LastFormalIntegrateBlockedCount);
            summary.AppendLine("formal_integrate_provisional=" + LastFormalIntegrateProvisionalCount);
            summary.AppendLine("formal_integrate_block_vote=" + LastFormalIntegrateBlockVoteCount);
            summary.AppendLine("formal_integrate_block_score=" + LastFormalIntegrateBlockScoreCount);
            summary.AppendLine("formal_integrate_block_support=" + LastFormalIntegrateBlockSupportCount);
            summary.AppendLine("formal_integrate_block_band=" + LastFormalIntegrateBlockBandCount);
            summary.AppendLine("formal_integrate_block_residual=" + LastFormalIntegrateBlockResidualCount);
            summary.AppendLine("formal_integrate_block_clean_history=" + LastFormalIntegrateBlockCleanHistoryCount);
            summary.AppendLine("formal_integrate_block_weak_history=" + LastFormalIntegrateBlockWeakHistoryCount);
            summary.AppendLine("formal_integrate_block_temporal_depth_jump=" + LastFormalIntegrateBlockTemporalDepthJumpCount);
            summary.AppendLine("formal_integrate_block_strong_current_history=" + LastFormalIntegrateBlockStrongCurrentHistoryCount);
            summary.AppendLine("formal_integrate_block_same_frame=" + LastFormalIntegrateBlockSameFrameCount);
            summary.AppendLine("formal_integrate_block_cross_frame=" + LastFormalIntegrateBlockCrossFrameCount);
            summary.AppendLine("formal_integrate_block_old_tsdf=" + LastFormalIntegrateBlockOldTsdfCount);
            summary.AppendLine("formal_integrate_block_dirty_pending=" + LastFormalIntegrateBlockDirtyPendingCount);
            summary.AppendLine("formal_integrate_min_score=" + AuditFloat(minFormalIntegrateVoteScore));
            summary.AppendLine("formal_integrate_min_support=" + AuditFloat(minFormalIntegrateSupportRatio));
            summary.AppendLine("formal_integrate_max_band_conflict=" + AuditFloat(maxFormalIntegrateBandConflictRatio));
            summary.AppendLine("formal_integrate_max_band_residual=" + AuditFloat(maxFormalIntegrateBandMeanResidual));
            summary.AppendLine("formal_clean_history_gate_enabled=" + (rejectFormalCleanHistoryMismatch ? 1 : 0));
            summary.AppendLine("formal_weak_history_gate_enabled=" + (rejectFormalIntegrateWeakHistory ? 1 : 0));
            summary.AppendLine("formal_temporal_depth_jump_gate_enabled=" + (rejectFormalIntegrateTemporalDepthJump ? 1 : 0));
            summary.AppendLine("formal_strong_current_history_gate_enabled=" + (requireHistoryForFormalStrongCurrentClean ? 1 : 0));
            summary.AppendLine("formal_strong_current_min_old_weight=" + Mathf.Clamp(minFormalStrongCurrentOldWeight, 0, 16));
            summary.AppendLine("formal_strong_current_min_band_history=" + Mathf.Clamp(minFormalStrongCurrentBandHistory, 0, 16));
            summary.AppendLine("formal_strong_current_min_history_agreement=" + AuditFloat(minFormalStrongCurrentHistoryAgreement));
            summary.AppendLine("formal_strong_current_local_history_bypass_enabled=" + (allowLocalSupportForFormalStrongCurrentHistory ? 1 : 0));
            summary.AppendLine("formal_strong_current_local_history_bypass=" + LastFormalStrongCurrentLocalHistoryBypassCount);
            summary.AppendLine("formal_strong_current_local_history_block_local=" + LastFormalStrongCurrentLocalHistoryBlockLocalCount);
            summary.AppendLine("formal_strong_current_local_history_block_support=" + LastFormalStrongCurrentLocalHistoryBlockSupportCount);
            summary.AppendLine("formal_strong_current_local_history_block_stable=" + LastFormalStrongCurrentLocalHistoryBlockStableCount);
            summary.AppendLine("formal_strong_current_local_history_block_axial=" + LastFormalStrongCurrentLocalHistoryBlockAxialCount);
            summary.AppendLine("formal_strong_current_local_history_block_residual=" + LastFormalStrongCurrentLocalHistoryBlockResidualCount);
            summary.AppendLine("formal_strong_current_local_history_block_plane=" + LastFormalStrongCurrentLocalHistoryBlockPlaneCount);
            summary.AppendLine("formal_strong_current_local_history_block_double_layer=" + LastFormalStrongCurrentLocalHistoryBlockDoubleLayerCount);
            summary.AppendLine("formal_strong_current_local_min=" + Mathf.Clamp(minFormalStrongCurrentLocalSupportVoxels, 1, 26));
            summary.AppendLine("formal_strong_current_local_min_stable=" + Mathf.Clamp(minFormalStrongCurrentLocalStableVoxels, 0, 26));
            summary.AppendLine("formal_strong_current_local_min_axial=" + Mathf.Clamp(minFormalStrongCurrentLocalAxialVoxels, 0, 6));
            summary.AppendLine("strong_current_provisional_promote_enabled=" + (promoteStrongCurrentProvisionalToFormal ? 1 : 0));
            summary.AppendLine("strong_current_provisional_promote=" + LastStrongCurrentProvisionalPromotedCount);
            summary.AppendLine("strong_current_provisional_promote_block=" + LastStrongCurrentProvisionalPromotionBlockedCount);
            summary.AppendLine("strong_current_provisional_block_disabled=" + LastStrongCurrentProvisionalBlockDisabledCount);
            summary.AppendLine("strong_current_provisional_block_tag=" + LastStrongCurrentProvisionalBlockTagCount);
            summary.AppendLine("strong_current_provisional_block_storage=" + LastStrongCurrentProvisionalBlockStorageCount);
            summary.AppendLine("strong_current_provisional_block_invalid=" + LastStrongCurrentProvisionalBlockInvalidCount);
            summary.AppendLine("strong_current_provisional_block_no_provisional=" + LastStrongCurrentProvisionalBlockNoProvisionalCount);
            summary.AppendLine("strong_current_provisional_block_no_provisional_near_surface=" + LastStrongCurrentProvisionalBlockNoProvisionalNearSurfaceCount);
            summary.AppendLine("strong_current_provisional_block_no_provisional_far_band=" + LastStrongCurrentProvisionalBlockNoProvisionalFarBandCount);
            summary.AppendLine("strong_current_provisional_block_hits=" + LastStrongCurrentProvisionalBlockHitsCount);
            summary.AppendLine("strong_current_provisional_block_weight=" + LastStrongCurrentProvisionalBlockWeightCount);
            summary.AppendLine("strong_current_provisional_block_band_history=" + LastStrongCurrentProvisionalBlockBandHistoryCount);
            summary.AppendLine("strong_current_provisional_block_agreement=" + LastStrongCurrentProvisionalBlockAgreementCount);
            summary.AppendLine("strong_current_provisional_block_clean_history=" + LastStrongCurrentProvisionalBlockCleanHistoryCount);
            summary.AppendLine("strong_current_provisional_block_dirty_pending=" + LastStrongCurrentProvisionalBlockDirtyPendingCount);
            summary.AppendLine("strong_current_provisional_block_same_frame=" + LastStrongCurrentProvisionalBlockSameFrameCount);
            summary.AppendLine("strong_current_provisional_block_cross_frame=" + LastStrongCurrentProvisionalBlockCrossFrameCount);
            summary.AppendLine("strong_current_provisional_block_conflict=" + LastStrongCurrentProvisionalBlockConflictCount);
            summary.AppendLine("strong_current_provisional_block_local=" + LastStrongCurrentProvisionalBlockLocalCount);
            summary.AppendLine("strong_current_provisional_block_plane=" + LastStrongCurrentProvisionalBlockPlaneCount);
            summary.AppendLine("strong_current_promotion_neighbor_support_enabled=" + (allowProvisionalNeighborPromotionSupport ? 1 : 0));
            summary.AppendLine("strong_current_promotion_local_pass=" + LastStrongCurrentPromotionLocalPassCount);
            summary.AppendLine("strong_current_promotion_local_block=" + LastStrongCurrentPromotionLocalBlockCount);
            summary.AppendLine("strong_current_promotion_local_neighbors=" + LastStrongCurrentPromotionLocalNeighborCount);
            summary.AppendLine("strong_current_promotion_local_provisional_neighbors=" + LastStrongCurrentPromotionLocalProvisionalNeighborCount);
            summary.AppendLine("strong_current_promotion_local_stable_neighbors=" + LastStrongCurrentPromotionLocalStableNeighborCount);
            summary.AppendLine("strong_current_promotion_local_axial_neighbors=" + LastStrongCurrentPromotionLocalAxialNeighborCount);
            summary.AppendLine("strong_current_promotion_local_min=" + Mathf.Clamp(minStrongCurrentPromotionLocalSupportVoxels, 1, 26));
            summary.AppendLine("strong_current_promotion_local_min_stable=" + Mathf.Clamp(minStrongCurrentPromotionStableVoxels, 0, 26));
            summary.AppendLine("strong_current_promotion_local_min_axial=" + Mathf.Clamp(minStrongCurrentPromotionAxialVoxels, 0, 6));
            summary.AppendLine("strong_current_promotion_double_layer_gate_enabled=" + (rejectStrongCurrentPromotionDoubleLayer ? 1 : 0));
            summary.AppendLine("strong_current_promotion_double_layer_candidates=" + LastStrongCurrentPromotionDoubleLayerCandidateCount);
            summary.AppendLine("strong_current_promotion_double_layer_block=" + LastStrongCurrentPromotionDoubleLayerBlockCount);
            summary.AppendLine("strong_current_promotion_double_layer_normal_reject=" + LastStrongCurrentPromotionDoubleLayerNormalRejectCount);
            summary.AppendLine("strong_current_promotion_double_layer_plane_reject=" + LastStrongCurrentPromotionDoubleLayerPlaneRejectCount);
            summary.AppendLine("strong_current_promotion_double_layer_lateral_reject=" + LastStrongCurrentPromotionDoubleLayerLateralRejectCount);
            summary.AppendLine("strong_current_promotion_double_layer_dirty_neighbors=" + LastStrongCurrentPromotionDoubleLayerDirtyNeighborCount);
            summary.AppendLine("strong_current_promotion_double_layer_block_integrate=" + LastStrongCurrentPromotionDoubleLayerIntegrateBlockCount);
            summary.AppendLine("strong_current_promotion_double_layer_block_replace=" + LastStrongCurrentPromotionDoubleLayerReplaceBlockCount);
            summary.AppendLine("strong_current_promotion_double_layer_block_other=" + LastStrongCurrentPromotionDoubleLayerOtherBlockCount);
            summary.AppendLine("strong_current_promotion_double_layer_radius=" + Mathf.Clamp(strongCurrentPromotionDoubleLayerSearchRadiusVoxels, 1, 4));
            summary.AppendLine("strong_current_promotion_double_layer_max_sep_m=" + AuditFloat(maxStrongCurrentPromotionDoubleLayerSeparationMeters));
            summary.AppendLine("strong_current_promotion_double_layer_max_neighbor_abs=" + AuditFloat(maxStrongCurrentPromotionDoubleLayerNeighborAbsTsdf));
            summary.AppendLine("strong_current_provisional_min_hits=" + Mathf.Clamp(minStrongCurrentProvisionalHits, 1, 8));
            summary.AppendLine("strong_current_provisional_min_weight=" + Mathf.Clamp(minStrongCurrentProvisionalWeight, 1, 8));
            summary.AppendLine("strong_current_provisional_min_band_history=" + Mathf.Clamp(minStrongCurrentProvisionalBandHistory, 0, 16));
            summary.AppendLine("strong_current_provisional_min_history_agreement=" + AuditFloat(minStrongCurrentProvisionalHistoryAgreement));
            summary.AppendLine("strong_current_provisional_promote_weight=" + AuditFloat(strongCurrentProvisionalPromoteWeight));
            summary.AppendLine("formal_clean_history_max_old_tsdf=" + AuditFloat(maxFormalCleanHistoryOldTsdfResidual));
            summary.AppendLine("formal_clean_history_max_band_residual=" + AuditFloat(maxFormalCleanHistoryBandMeanResidual));
            summary.AppendLine("formal_clean_history_max_band_conflict=" + AuditFloat(maxFormalCleanHistoryBandConflictRatio));
            summary.AppendLine("vote_pending_hold=" + LastVotePendingHoldCount);
            summary.AppendLine("vote_pending_confirmed_write=" + LastVotePendingConfirmedWriteCount);
            summary.AppendLine("vote_pending_bootstrap_write=" + LastVotePendingBootstrapWriteCount);
            summary.AppendLine("strong_sample_seed_enabled=" + (allowStrongSampleSeedWrites ? 1 : 0));
            summary.AppendLine("strong_sample_seed_temporary_lock=" + (lockStrongSampleSeedToTemporaryTsdf ? 1 : 0));
            summary.AppendLine("strong_sample_seed_write=" + LastStrongSampleSeedWriteCount);
            summary.AppendLine("strong_sample_seed_block=" + LastStrongSampleSeedBlockedCount);
            summary.AppendLine("strong_sample_seed_temp_block=" + LastStrongSampleSeedTemporaryBlockedCount);
            summary.AppendLine("strong_sample_seed_temp_block_near_surface=" + LastStrongSampleSeedTempNearSurfaceBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_vote=" + LastStrongSampleSeedTempVoteBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_score=" + LastStrongSampleSeedTempScoreBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_support=" + LastStrongSampleSeedTempSupportBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_band=" + LastStrongSampleSeedTempBandBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_same_frame=" + LastStrongSampleSeedTempSameFrameBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_cross_frame=" + LastStrongSampleSeedTempCrossFrameBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_dirty_pending=" + LastStrongSampleSeedTempDirtyPendingBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_old_weight=" + LastStrongSampleSeedTempOldWeightBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_conflict=" + LastStrongSampleSeedTempConflictBlockCount);
            summary.AppendLine("strong_sample_seed_temp_block_local=" + LastStrongSampleSeedTempLocalBlockCount);
            summary.AppendLine("strong_sample_seed_weight_scale=" + AuditFloat(strongSampleSeedWeightScale));
            summary.AppendLine("strong_sample_seed_max_old_tsdf_residual=" + AuditFloat(maxStrongSampleSeedOldTsdfResidual));
            summary.AppendLine("vote_strong_current_accept=" + LastVoteStrongCurrentAcceptCount);
            summary.AppendLine("vote_same_frame_conflict_depth_m=" + AuditFloat(sameFrameConflictDepthMeters));
            summary.AppendLine("vote_cross_frame_clean_conflict_depth_m=" + AuditFloat(crossFrameCleanConflictDepthMeters));
            summary.AppendLine("vote_cross_frame_clean_conflict_search_m=" + AuditFloat(crossFrameCleanConflictSearchMeters));
            summary.AppendLine("vote_pending_weight_scale=" + AuditFloat(confirmedPendingWriteWeightScale));
            summary.AppendLine("provisional_tsdf_enabled=" + (allowProvisionalTsdfSupportWrites ? 1 : 0));
            summary.AppendLine("provisional_tsdf_write=" + LastProvisionalTsdfSupportWriteCount);
            summary.AppendLine("provisional_tsdf_block=" + LastProvisionalTsdfSupportBlockedCount);
            summary.AppendLine("provisional_tsdf_near_surface_abs=" + AuditFloat(provisionalTsdfNearSurfaceAbs));
            summary.AppendLine("provisional_tsdf_near_surface_block=" + LastProvisionalTsdfNearSurfaceBlockedCount);
            summary.AppendLine("provisional_tsdf_near_surface_positive_block=" + LastProvisionalTsdfNearSurfacePositiveBlockedCount);
            summary.AppendLine("provisional_tsdf_near_surface_negative_block=" + LastProvisionalTsdfNearSurfaceNegativeBlockedCount);
            summary.AppendLine("provisional_tsdf_far_band_skip=" + LastProvisionalTsdfFarBandSkippedCount);
            summary.AppendLine("provisional_tsdf_disabled_block=" + LastProvisionalTsdfDisabledBlockedCount);
            summary.AppendLine("provisional_tsdf_invalid_block=" + LastProvisionalTsdfInvalidBlockedCount);
            summary.AppendLine("provisional_tsdf_vote_block=" + LastProvisionalTsdfVoteBlockedCount);
            summary.AppendLine("provisional_tsdf_score_block=" + LastProvisionalTsdfScoreBlockedCount);
            summary.AppendLine("provisional_tsdf_support_block=" + LastProvisionalTsdfSupportRatioBlockedCount);
            summary.AppendLine("provisional_tsdf_band_block=" + LastProvisionalTsdfBandBlockedCount);
            summary.AppendLine("provisional_tsdf_clean_history_block=" + LastProvisionalTsdfCleanHistoryBlockedCount);
            summary.AppendLine("provisional_tsdf_same_frame_block=" + LastProvisionalTsdfSameFrameBlockedCount);
            summary.AppendLine("provisional_tsdf_cross_frame_block=" + LastProvisionalTsdfCrossFrameBlockedCount);
            summary.AppendLine("provisional_tsdf_dirty_pending_block=" + LastProvisionalTsdfDirtyPendingBlockedCount);
            summary.AppendLine("provisional_tsdf_old_weight_block=" + LastProvisionalTsdfOldWeightBlockedCount);
            summary.AppendLine("provisional_tsdf_conflict_block=" + LastProvisionalTsdfConflictBlockedCount);
            summary.AppendLine("provisional_tsdf_plane_block=" + LastProvisionalTsdfPlaneBlockedCount);
            summary.AppendLine("provisional_tsdf_pending_stability_block=" + LastProvisionalTsdfPendingStabilityBlockedCount);
            summary.AppendLine("provisional_tsdf_formal_downgrade_block=" + LastProvisionalTsdfFormalDowngradeBlockedCount);
            summary.AppendLine("provisional_tsdf_existing_weight_bypass=" + LastProvisionalTsdfExistingWeightBypassCount);
            summary.AppendLine("provisional_tsdf_bootstrap_local_bypass=" + LastProvisionalTsdfBootstrapLocalBypassCount);
            summary.AppendLine("provisional_tsdf_bootstrap_enabled=" + (allowNearSurfaceProvisionalBootstrap ? 1 : 0));
            summary.AppendLine("provisional_tsdf_bootstrap_max_abs=" + AuditFloat(maxNearSurfaceProvisionalBootstrapAbsTsdf));
            summary.AppendLine("provisional_tsdf_bootstrap_min_vote=" + AuditFloat(minNearSurfaceProvisionalBootstrapVoteScore));
            summary.AppendLine("provisional_tsdf_bootstrap_min_support=" + AuditFloat(minNearSurfaceProvisionalBootstrapSupportRatio));
            summary.AppendLine("provisional_plane_compat_enabled=" + (requireProvisionalPlaneCompatibility ? 1 : 0));
            summary.AppendLine("provisional_plane_compat_pass=" + LastProvisionalPlaneCompatibilityPassCount);
            summary.AppendLine("provisional_plane_compat_block=" + LastProvisionalPlaneCompatibilityBlockedCount);
            summary.AppendLine("provisional_plane_compat_no_ref=" + LastProvisionalPlaneCompatibilityNoReferenceCount);
            summary.AppendLine("provisional_plane_compat_candidates=" + LastProvisionalPlaneCompatibilityCandidateCount);
            summary.AppendLine("provisional_plane_compat_normal_reject=" + LastProvisionalPlaneCompatibilityNormalRejectedCount);
            summary.AppendLine("provisional_plane_compat_distance_reject=" + LastProvisionalPlaneCompatibilityDistanceRejectedCount);
            summary.AppendLine("provisional_plane_compat_radius=" + provisionalPlaneCompatibilityRadiusVoxels);
            summary.AppendLine("provisional_plane_compat_min_neighbors=" + minProvisionalPlaneCompatibleNeighbors);
            summary.AppendLine("provisional_plane_compat_min_normal_dot=" + AuditFloat(minProvisionalPlaneNormalDot));
            summary.AppendLine("provisional_plane_compat_max_distance_vox=" + AuditFloat(maxProvisionalPlaneDistanceVoxelScale));
            summary.AppendLine("provisional_local_support_enabled=" + (requireProvisionalLocalSupport ? 1 : 0));
            summary.AppendLine("provisional_local_support_pass=" + LastProvisionalLocalSupportPassCount);
            summary.AppendLine("provisional_local_support_block=" + LastProvisionalLocalSupportBlockedCount);
            summary.AppendLine("provisional_local_support_neighbors=" + LastProvisionalLocalSupportNeighborCount);
            summary.AppendLine("provisional_local_support_stable_neighbors=" + LastProvisionalLocalSupportStableNeighborCount);
            summary.AppendLine("provisional_local_support_axial_neighbors=" + LastProvisionalLocalSupportAxialNeighborCount);
            summary.AppendLine("provisional_local_support_radius=" + provisionalLocalSupportRadiusVoxels);
            summary.AppendLine("provisional_local_support_min=" + minProvisionalLocalSupportVoxels);
            summary.AppendLine("provisional_local_support_min_stable=" + minProvisionalLocalStableVoxels);
            summary.AppendLine("provisional_local_support_min_axial=" + minProvisionalLocalAxialVoxels);
            summary.AppendLine("provisional_local_support_max_abs=" + AuditFloat(maxProvisionalLocalSupportAbsTsdf));
            summary.AppendLine("provisional_local_support_max_residual=" + AuditFloat(maxProvisionalLocalSupportResidual));
            summary.AppendLine("provisional_tsdf_confirmed=" + LastProvisionalTsdfConfirmedCount);
            summary.AppendLine("provisional_tsdf_confirmed_by_weight=" + LastProvisionalTsdfConfirmedByWeightCount);
            summary.AppendLine("provisional_tsdf_retired=" + LastProvisionalTsdfRetiredCount);
            summary.AppendLine("provisional_tsdf_retired_expired=" + LastProvisionalTsdfRetiredExpiredCount);
            summary.AppendLine("provisional_tsdf_dirty_cleared=" + LastProvisionalTsdfDirtyClearedCount);
            summary.AppendLine("provisional_tsdf_weight=" + provisionalTsdfSupportWeight);
            summary.AppendLine("provisional_tsdf_blend=" + AuditFloat(provisionalTsdfSupportBlend));
            summary.AppendLine("provisional_tsdf_max_age_frames=" + provisionalTsdfMaxAgeFrames);
            summary.AppendLine("old_clean_metabolism_enabled=" + (enableOldCleanTsdfMetabolism ? 1 : 0));
            summary.AppendLine("old_clean_metabolism_watch=" + LastOldCleanMetabolismWatchCount);
            summary.AppendLine("old_clean_metabolism_decay=" + LastOldCleanMetabolismDecayCount);
            summary.AppendLine("old_clean_metabolism_clear=" + LastOldCleanMetabolismClearCount);
            summary.AppendLine("old_clean_metabolism_block=" + LastOldCleanMetabolismBlockedCount);
            summary.AppendLine("old_clean_metabolism_candidate=" + LastOldCleanMetabolismCandidateCount);
            summary.AppendLine("old_clean_metabolism_waiting_hits=" + LastOldCleanMetabolismWaitingHitsCount);
            summary.AppendLine("old_clean_metabolism_block_support=" + LastOldCleanMetabolismBlockedSupportCount);
            summary.AppendLine("old_clean_metabolism_block_same_frame=" + LastOldCleanMetabolismBlockedSameFrameCount);
            summary.AppendLine("old_clean_metabolism_block_weight=" + LastOldCleanMetabolismBlockedWeightCount);
            summary.AppendLine("old_clean_metabolism_block_residual=" + LastOldCleanMetabolismBlockedResidualCount);
            summary.AppendLine("old_clean_metabolism_block_dirty_pending=" + LastOldCleanMetabolismBlockedDirtyPendingCount);
            summary.AppendLine("old_clean_metabolism_skip_weak_band=" + LastOldCleanMetabolismSkippedWeakBandCount);
            summary.AppendLine("old_clean_metabolism_skip_weak_cross_frame=" + LastOldCleanMetabolismSkippedWeakCrossFrameCount);
            summary.AppendLine("old_clean_metabolism_min_hits=" + minOldCleanMetabolismConflictHits);
            summary.AppendLine("old_clean_metabolism_max_weight=" + maxOldCleanMetabolismWeight);
            summary.AppendLine("old_clean_metabolism_clear_weight=" + oldCleanMetabolismClearWeight);
            summary.AppendLine("old_clean_metabolism_cross_frame_voxel_residual=" + AuditFloat(minOldCleanMetabolismCrossFrameVoxelResidual));
            summary.AppendLine("old_clean_metabolism_band_voxel_residual=" + AuditFloat(minOldCleanMetabolismBandVoxelResidual));
            summary.AppendLine("fusion_total_frames=" + IntegratedFrameCount);
            summary.AppendLine("free_space_evidence_enabled=" + (useTemporalFreeSpaceEvidence ? 1 : 0));
            summary.AppendLine("free_space_evidence_candidates=" + LastFreeSpaceEvidenceCandidateCount);
            summary.AppendLine("free_space_evidence_new=" + LastFreeSpaceEvidenceNewCount);
            summary.AppendLine("free_space_evidence_repeat=" + LastFreeSpaceEvidenceRepeatCount);
            summary.AppendLine("free_space_evidence_waiting=" + LastFreeSpaceEvidenceWaitingCount);
            summary.AppendLine("free_space_evidence_applied=" + LastFreeSpaceEvidenceAppliedCount);
            summary.AppendLine("free_space_evidence_cleared=" + LastFreeSpaceEvidenceClearedCount);
            summary.AppendLine("free_space_evidence_block_high_weight=" + LastFreeSpaceEvidenceBlockedHighWeightCount);
            summary.AppendLine("free_space_evidence_block_same_frame=" + LastFreeSpaceEvidenceBlockedSameFrameCount);
            summary.AppendLine("free_space_evidence_duplicate_frame=" + LastFreeSpaceEvidenceDuplicateFrameCount);
            summary.AppendLine("free_space_evidence_cancelled_by_surface=" + LastFreeSpaceEvidenceCancelledBySurfaceCount);
            summary.AppendLine("free_space_evidence_min_frames=" + Mathf.Clamp(minFreeSpaceEvidenceFrames, 2, 8));
            summary.AppendLine("free_space_evidence_max_gap=" + Mathf.Max(2, maxFreeSpaceEvidenceFrameGap));
            summary.AppendLine("free_space_evidence_weight_decay=" + Mathf.Clamp(freeSpaceEvidenceWeightDecay, 1, 4));
            summary.AppendLine("pending_wait=" + LastPendingTsdfCorrectionCount);
            summary.AppendLine("dirty_replace=" + LastReplacedDirtyTsdfCount);
            summary.AppendLine("dirty_guard=" + LastGuardedDirtyTsdfReplaceCount);
            summary.AppendLine("dirty_replace_block_clean_history=" + LastDirtyTsdfReplaceBlockedCleanHistoryCount);
            summary.AppendLine("dirty_band_repair=" + LastDirtyTsdfBandRepairCount);
            summary.AppendLine("dirty_band_repair_samples=" + LastDirtyTsdfBandRepairSampleCount);
            summary.AppendLine("dirty_band_repair_triggers=" + LastDirtyTsdfBandRepairTriggerCount);
            summary.AppendLine("dirty_band_repair_probes=" + LastDirtyTsdfBandRepairProbeCount);
            summary.AppendLine("dirty_band_repair_block_disabled=" + LastDirtyTsdfBandRepairBlockedDisabledCount);
            summary.AppendLine("dirty_band_repair_block_no_sample=" + LastDirtyTsdfBandRepairBlockedNoSampleCount);
            summary.AppendLine("dirty_band_repair_block_no_history=" + LastDirtyTsdfBandRepairBlockedNoHistoryCount);
            summary.AppendLine("dirty_band_repair_block_low_conflict=" + LastDirtyTsdfBandRepairBlockedLowConflictCount);
            summary.AppendLine("dirty_band_repair_block_budget=" + LastDirtyTsdfBandRepairBlockedBudgetCount);
            summary.AppendLine("dirty_band_repair_block_outside=" + LastDirtyTsdfBandRepairBlockedOutsideCount);
            summary.AppendLine("dirty_band_repair_block_empty=" + LastDirtyTsdfBandRepairBlockedEmptyCount);
            summary.AppendLine("dirty_band_repair_block_weight=" + LastDirtyTsdfBandRepairBlockedWeightCount);
            summary.AppendLine("dirty_band_repair_block_same_sign=" + LastDirtyTsdfBandRepairBlockedSameSignCount);
            summary.AppendLine("dirty_band_repair_block_residual=" + LastDirtyTsdfBandRepairBlockedResidualCount);
            summary.AppendLine("dirty_active=" + LastDirtyTsdfActiveCount);
            summary.AppendLine("mesh_triangles=" + LastMeshTriangleCount);
            summary.AppendLine("island_cause_components=" + LastIslandCauseComponentCount);
            summary.AppendLine("island_cause_boundary_voxels=" + LastIslandCauseBoundaryVoxelCount);
            summary.AppendLine("island_cause_no_tsdf=" + LastIslandCauseNoTsdfCount);
            summary.AppendLine("island_cause_pending=" + LastIslandCausePendingCount);
            summary.AppendLine("island_cause_dirty=" + LastIslandCauseDirtyCount);
            summary.AppendLine("island_cause_low_weight=" + LastIslandCauseLowWeightCount);
            summary.AppendLine("island_cause_plane_mismatch=" + LastIslandCausePlaneMismatchCount);
            summary.AppendLine("island_cause_pruned_or_topology=" + LastIslandCausePrunedCount);
            summary.AppendLine("tsdf_continuity_candidates=" + LastTsdfContinuityCandidateCount);
            summary.AppendLine("tsdf_continuity_filled=" + LastTsdfContinuityFilledCount);
            summary.AppendLine("tsdf_continuity_same_sign_filled=" + LastTsdfContinuitySameSignFilledCount);
            summary.AppendLine("tsdf_continuity_mixed_sign_filled=" + LastTsdfContinuityMixedSignFilledCount);
            summary.AppendLine("tsdf_continuity_provisional_neighbors=" + LastTsdfContinuityProvisionalNeighborCount);
            summary.AppendLine("tsdf_continuity_block_dirty_pending=" + LastTsdfContinuityBlockedDirtyPendingCount);
            summary.AppendLine("tsdf_continuity_block_low_support=" + LastTsdfContinuityBlockedLowSupportCount);
            summary.AppendLine("tsdf_continuity_block_budget=" + LastTsdfContinuityBlockedBudgetCount);
            summary.AppendLine("tsdf_continuity_block_unsettled_neighbor=" + LastTsdfContinuityBlockedUnsettledNeighborCount);
            summary.AppendLine("tsdf_continuity_allow_same_sign=" + (allowSameSignContinuityBaseFill ? 1 : 0));
            summary.AppendLine("tsdf_continuity_allow_provisional_sources=" + (allowStableProvisionalContinuitySources ? 1 : 0));
            summary.AppendLine("tsdf_continuity_same_sign_min=" + Mathf.Clamp(minTsdfContinuitySameSignNeighborVoxels, 3, 26));
            summary.AppendLine("tsdf_continuity_face_min=" + Mathf.Clamp(minTsdfContinuityFaceNeighborVoxels, 1, 6));
            summary.AppendLine("tsdf_boundary_no_tsdf_candidates=" + LastTsdfBoundaryNoTsdfCandidateCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_filled=" + LastTsdfBoundaryNoTsdfFilledCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_blocked=" + LastTsdfBoundaryNoTsdfBlockedCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_support=" + LastTsdfBoundaryNoTsdfBlockedSupportCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_face=" + LastTsdfBoundaryNoTsdfBlockedFaceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_axis=" + LastTsdfBoundaryNoTsdfBlockedAxisCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_clean_anchor=" + LastTsdfBoundaryNoTsdfBlockedCleanAnchorCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_provisional_anchor=" + LastTsdfBoundaryNoTsdfBlockedProvisionalAnchorCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_near_surface=" + LastTsdfBoundaryNoTsdfBlockedNearSurfaceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_block_budget=" + LastTsdfBoundaryNoTsdfBlockedBudgetCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_relaxed_anchor_filled=" + LastTsdfBoundaryNoTsdfRelaxedAnchorFilledCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_verified_joint_anchor_filled=" + LastTsdfBoundaryNoTsdfVerifiedJointAnchorFilledCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_zero_clean_joint_anchor_filled=" + LastTsdfBoundaryNoTsdfZeroCleanJointAnchorFilledCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_present=" + LastTsdfBoundaryNoTsdfProvisionalPresentNeighborCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_face_present=" + LastTsdfBoundaryNoTsdfProvisionalFacePresentCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_accept=" + LastTsdfBoundaryNoTsdfProvisionalAcceptedNeighborCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_face_accept=" + LastTsdfBoundaryNoTsdfProvisionalAcceptedFaceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_block_weight=" + LastTsdfBoundaryNoTsdfProvisionalBlockedWeightCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_block_dirty_pending=" + LastTsdfBoundaryNoTsdfProvisionalBlockedDirtyPendingCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_block_tsdf=" + LastTsdfBoundaryNoTsdfProvisionalBlockedTsdfCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_block_residual=" + LastTsdfBoundaryNoTsdfProvisionalBlockedResidualCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_block_provenance=" + LastTsdfBoundaryNoTsdfProvisionalBlockedProvenanceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_prov_block_not_face=" + LastTsdfBoundaryNoTsdfProvisionalBlockedNotFaceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_0=" + LastTsdfBoundaryNoTsdfFaceSupport0Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_1=" + LastTsdfBoundaryNoTsdfFaceSupport1Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_2=" + LastTsdfBoundaryNoTsdfFaceSupport2Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_3=" + LastTsdfBoundaryNoTsdfFaceSupport3Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_4=" + LastTsdfBoundaryNoTsdfFaceSupport4Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_5=" + LastTsdfBoundaryNoTsdfFaceSupport5Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_hist_6=" + LastTsdfBoundaryNoTsdfFaceSupport6Count);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_deficit_one=" + LastTsdfBoundaryNoTsdfFaceDeficitOneCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_deficit_two_plus=" + LastTsdfBoundaryNoTsdfFaceDeficitTwoPlusCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_accept_clean=" + LastTsdfBoundaryNoTsdfFaceSlotAcceptedCleanCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_accept_provisional=" + LastTsdfBoundaryNoTsdfFaceSlotAcceptedProvisionalCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_block_weight=" + LastTsdfBoundaryNoTsdfFaceSlotBlockedWeightCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_block_dirty_pending=" + LastTsdfBoundaryNoTsdfFaceSlotBlockedDirtyPendingCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_block_provenance=" + LastTsdfBoundaryNoTsdfFaceSlotBlockedProvenanceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_block_tsdf=" + LastTsdfBoundaryNoTsdfFaceSlotBlockedTsdfCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_block_residual=" + LastTsdfBoundaryNoTsdfFaceSlotBlockedResidualCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_face_slot_out_of_bounds=" + LastTsdfBoundaryNoTsdfFaceSlotOutOfBoundsCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_verified_two_face_candidates=" + LastTsdfBoundaryNoTsdfVerifiedTwoFaceCandidateCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_verified_two_face_filled=" + LastTsdfBoundaryNoTsdfVerifiedTwoFaceFilledCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_passed=" + LastTsdfBoundaryNoTsdfSupportPassedCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_deficit_one=" + LastTsdfBoundaryNoTsdfSupportDeficitOneCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_deficit_two=" + LastTsdfBoundaryNoTsdfSupportDeficitTwoCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_deficit_three=" + LastTsdfBoundaryNoTsdfSupportDeficitThreeCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_deficit_four_plus=" + LastTsdfBoundaryNoTsdfSupportDeficitFourPlusCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_accept_clean=" + LastTsdfBoundaryNoTsdfSupportSlotAcceptedCleanCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_accept_provisional=" + LastTsdfBoundaryNoTsdfSupportSlotAcceptedProvisionalCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_block_weight=" + LastTsdfBoundaryNoTsdfSupportSlotBlockedWeightCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_block_dirty_pending=" + LastTsdfBoundaryNoTsdfSupportSlotBlockedDirtyPendingCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_block_provenance=" + LastTsdfBoundaryNoTsdfSupportSlotBlockedProvenanceCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_block_tsdf=" + LastTsdfBoundaryNoTsdfSupportSlotBlockedTsdfCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_block_residual=" + LastTsdfBoundaryNoTsdfSupportSlotBlockedResidualCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_support_slot_out_of_bounds=" + LastTsdfBoundaryNoTsdfSupportSlotOutOfBoundsCount);
            summary.AppendLine("tsdf_boundary_no_tsdf_min=" + Mathf.Clamp(minBoundaryNoTsdfNeighborVoxels, 3, 26));
            summary.AppendLine("tsdf_boundary_no_tsdf_face_min=" + Mathf.Clamp(minBoundaryNoTsdfFaceNeighborVoxels, 2, 6));
            summary.AppendLine("tsdf_boundary_no_tsdf_provisional_anchor_enabled=" + (allowBoundaryNoTsdfProvisionalAnchor ? 1 : 0));
            summary.AppendLine("tsdf_boundary_no_tsdf_clean_face_anchor_min=0");
            summary.AppendLine("tsdf_boundary_no_tsdf_clean_face_anchor_serialized=" + Mathf.Clamp(minBoundaryNoTsdfCleanFaceAnchors, 0, 6));
            summary.AppendLine("tsdf_boundary_no_tsdf_provisional_anchor_min=" + Mathf.Clamp(minBoundaryNoTsdfProvisionalAnchors, 1, 12));
            summary.AppendLine("tsdf_boundary_no_tsdf_provisional_face_anchor_min=" + Mathf.Clamp(minBoundaryNoTsdfProvisionalFaceAnchors, 1, 6));
            summary.AppendLine("pose_max_position_delta_m=" + AuditFloat(_auditMaxPosePositionDelta));
            summary.AppendLine("pose_max_angle_delta_deg=" + AuditFloat(_auditMaxPoseAngleDelta));
            summary.AppendLine("snapshot_max_latency_ms=" + AuditFloat(_auditMaxSnapshotLatencyMs));
            summary.AppendLine("conflict_voxels=" + _captureAuditVoxels.Count);
            summary.AppendLine("mesh_associated_conflict_voxels=" + _auditMeshAssociatedVoxelCount);
            summary.AppendLine("recurrent_after_replace_voxels=" + _auditRecurrentAfterReplaceCount);
            summary.AppendLine("likely_depth_instability_voxels=" + _auditLikelyDepthInstabilityCount);
            summary.AppendLine("view_or_registration_ambiguous_voxels=" + _auditViewOrRegistrationCount);
            summary.AppendLine("suspect_mesh_voxels=" + _auditSuspectMeshVoxels.Count);
            summary.AppendLine("suspect_mesh_source_integrate=" + LastSuspectMeshSourceIntegrateVoxelCount);
            summary.AppendLine("suspect_mesh_source_provisional=" + LastSuspectMeshSourceProvisionalVoxelCount);
            summary.AppendLine("suspect_mesh_source_strong_seed=" + LastSuspectMeshSourceStrongSeedVoxelCount);
            summary.AppendLine("suspect_mesh_source_continuity=" + LastSuspectMeshSourceContinuityVoxelCount);
            summary.AppendLine("suspect_mesh_source_carve=" + LastSuspectMeshSourceCarveVoxelCount);
            summary.AppendLine("suspect_mesh_source_replace=" + LastSuspectMeshSourceReplaceVoxelCount);
            summary.AppendLine("suspect_mesh_source_repair=" + LastSuspectMeshSourceRepairVoxelCount);
            summary.AppendLine("suspect_mesh_source_other=" + LastSuspectMeshSourceOtherVoxelCount);
            summary.AppendLine("suspect_mesh_source_unknown=" + LastSuspectMeshSourceUnknownVoxelCount);
            summary.AppendLine("conflict_voxels_linked_to_suspect_mesh=" + _auditSuspectAssociatedConflictVoxelCount);
            summary.AppendLine("suspect_mesh_voxels_linked_to_conflict=" + _auditSuspectVoxelsLinkedToConflictCount);
            summary.AppendLine("pure_geometry_triangles=" + LastPureGeometrySuspectTriangleCount);
            summary.AppendLine("pure_geometry_normal_triangles=" + LastPureGeometryNormalTriangleCount);
            summary.AppendLine("pure_geometry_protrusion_triangles=" + LastPureGeometryProtrusionTriangleCount);
            summary.AppendLine("pure_geometry_double_layer_triangles=" + LastPureGeometryDoubleLayerTriangleCount);
            summary.AppendLine("double_layer_source_dirty=" + LastDoubleLayerSourceDirtyCount);
            summary.AppendLine("double_layer_source_pending=" + LastDoubleLayerSourcePendingCount);
            summary.AppendLine("double_layer_source_band_conflict=" + LastDoubleLayerSourceBandConflictCount);
            summary.AppendLine("double_layer_source_old_locked=" + LastDoubleLayerSourceOldLockedCount);
            summary.AppendLine("double_layer_source_clean=" + LastDoubleLayerSourceCleanCount);
            summary.AppendLine("double_layer_source_empty=" + LastDoubleLayerSourceEmptyCount);
            summary.AppendLine("double_layer_source_mixed=" + LastDoubleLayerSourceMixedCount);
            summary.AppendLine("double_layer_source_unknown=" + LastDoubleLayerSourceUnknownCount);
            summary.AppendLine("double_layer_clean_definition=weighted_without_active_dirty_pending_band_or_old_lock");
            summary.AppendLine("double_layer_clean_no_lifecycle=" + LastDoubleLayerCleanNoLifecycleCount);
            summary.AppendLine("double_layer_clean_no_provenance=" + LastDoubleLayerCleanNoProvenanceCount);
            summary.AppendLine("double_layer_clean_conflict_history=" + LastDoubleLayerCleanConflictHistoryCount);
            summary.AppendLine("double_layer_clean_replace_history=" + LastDoubleLayerCleanReplaceHistoryCount);
            summary.AppendLine("double_layer_clean_expired_dirty=" + LastDoubleLayerCleanExpiredDirtyCount);
            summary.AppendLine("double_layer_clean_low_weight=" + LastDoubleLayerCleanLowWeightCount);
            summary.AppendLine("double_layer_clean_high_weight=" + LastDoubleLayerCleanHighWeightCount);
            summary.AppendLine("double_layer_clean_last_integrate=" + LastDoubleLayerCleanLastIntegrateCount);
            summary.AppendLine("double_layer_clean_last_carve=" + LastDoubleLayerCleanLastCarveCount);
            summary.AppendLine("double_layer_clean_last_replace=" + LastDoubleLayerCleanLastReplaceCount);
            summary.AppendLine("double_layer_clean_last_repair=" + LastDoubleLayerCleanLastRepairCount);
            summary.AppendLine("double_layer_clean_integrate_carve_history=" + LastDoubleLayerCleanIntegrateCarveHistoryCount);
            summary.AppendLine("double_layer_clean_multi_frame_history=" + LastDoubleLayerCleanMultiFrameHistoryCount);
            summary.AppendLine("double_layer_pair_rows=" + _doubleLayerPairRowCount);
            summary.AppendLine("double_layer_pair_dropped_rows=" + _doubleLayerPairDroppedRows);
            summary.AppendLine("double_layer_pairs_csv=" + doubleLayerPairsPath);
            summary.AppendLine("double_layer_validated_pairs=" + LastDoubleLayerValidatedPairCount);
            summary.AppendLine("double_layer_reject_missing_source=" + LastDoubleLayerRejectedMissingSourceCount);
            summary.AppendLine("double_layer_reject_source_normal=" + LastDoubleLayerRejectedSourceNormalCount);
            summary.AppendLine("double_layer_reject_source_plane=" + LastDoubleLayerRejectedSourcePlaneCount);
            summary.AppendLine("double_layer_reject_source_lateral=" + LastDoubleLayerRejectedSourceLateralCount);
            summary.AppendLine("double_layer_mesh_alignment_mismatch=" + LastDoubleLayerMeshAlignmentMismatchCount);
            summary.AppendLine("raw_coverage_grid_enabled=" + (enableRawCoverageGridDiagnostics ? 1 : 0));
            summary.AppendLine("raw_coverage_grid_cell_size_m=" + AuditFloat(rawCoverageGridCellSizeMeters));
            summary.AppendLine("raw_coverage_grid_samples=" + LastRawCoverageGridSampleCount);
            summary.AppendLine("raw_coverage_grid_dropped_samples=" + LastRawCoverageGridDroppedSampleCount);
            summary.AppendLine("raw_coverage_grid_cells=" + LastRawCoverageGridCellCount);
            summary.AppendLine("raw_coverage_grid_accepted_cells=" + LastRawCoverageAcceptedCellCount);
            summary.AppendLine("raw_coverage_grid_problem_cells=" + LastRawCoverageProblemCellCount);
            summary.AppendLine("raw_coverage_grid_mixed_cells=" + LastRawCoverageMixedCellCount);
            summary.AppendLine("raw_coverage_grid_accepted_components=" + LastRawCoverageAcceptedComponentCount);
            summary.AppendLine("raw_coverage_grid_largest_accepted_component=" + LastRawCoverageLargestAcceptedComponentCells);
            summary.AppendLine("raw_coverage_grid_problem_components=" + LastRawCoverageProblemComponentCount);
            summary.AppendLine("raw_coverage_grid_largest_problem_component=" + LastRawCoverageLargestProblemComponentCells);
            summary.AppendLine("raw_coverage_grid_rendered_cells=" + LastRawCoverageRenderedCellCount);
            summary.AppendLine("pure_geometry_voxels=" + _auditPureGeometryMeshVoxels.Count);
            summary.AppendLine("conflict_voxels_linked_to_pure_geometry=" + _auditPureGeometryAssociatedConflictVoxelCount);
            summary.AppendLine("pure_geometry_voxels_linked_to_conflict=" + _auditPureGeometryVoxelsLinkedToConflictCount);
            summary.AppendLine("old_locked_above_correctable_weight=" + _auditLockedAboveCorrectableWeightCount);
            summary.AppendLine("old_locked_band_conflict=" + _auditLockedBandConflictCount);
            summary.AppendLine("old_locked_replace_recurrent=" + _auditLockedReplaceRecurrentCount);
            summary.AppendLine("old_locked_depth_unstable=" + _auditLockedDepthUnstableCount);
            summary.AppendLine("old_locked_view_changed=" + _auditLockedViewChangedCount);
            summary.AppendLine("old_locked_under_supported=" + _auditLockedUnderSupportedCount);
            summary.AppendLine("old_locked_stable_old_surface=" + _auditLockedStableOldSurfaceCount);
            summary.AppendLine("snapshot_pose_sync_suspect=" + ((_auditMaxPosePositionDelta > 0.025f || _auditMaxPoseAngleDelta > 2f || _auditMaxSnapshotLatencyMs > 50f) ? 1 : 0));
            summary.AppendLine("samples_csv=" + csvPath);
            summary.AppendLine("frames_csv=" + framesPath);
            summary.AppendLine("voxel_lifecycle_csv=" + lifecyclePath);
            summary.AppendLine("contribution_ledger_rows=" + _contributionLedgerRowCount);
            summary.AppendLine("contribution_ledger_write_failures=" + _contributionLedgerWriteFailures);
            summary.AppendLine("contribution_ledger_csv=" + (_contributionLedgerPath ?? "unavailable"));
            File.WriteAllText(summaryPath, summary.ToString(), new UTF8Encoding(false));
            _lastConfidenceAuditPath = csvPath;
            if (debugLog)
                Debug.Log($"[ScanCover] Confidence audit written: {csvPath} rows={_confidenceAuditRowCount} dropped={_confidenceAuditDroppedRows}", this);
        }
        catch (System.Exception firstError)
        {
            try
            {
                string fallbackDirectory = Path.Combine(Application.persistentDataPath, SanitizeAuditFolderName());
                Directory.CreateDirectory(fallbackDirectory);
                string fallbackPath = Path.Combine(fallbackDirectory, _confidenceAuditStem + "_samples.csv");
                File.WriteAllText(fallbackPath, _confidenceAuditRows.ToString(), new UTF8Encoding(false));
                _lastConfidenceAuditPath = fallbackPath + " (fallback)";
                Debug.LogWarning($"[ScanCover] Install directory audit write failed; wrote fallback file to {fallbackPath}. Error: {firstError.Message}", this);
            }
            catch (System.Exception fallbackError)
            {
                _lastConfidenceAuditPath = "WRITE FAILED";
                Debug.LogError($"[ScanCover] Confidence audit write failed. install={firstError.Message}; fallback={fallbackError.Message}", this);
            }
        }
        finally
        {
            if (_contributionLedgerWriter != null)
            {
                _contributionLedgerWriter.Dispose();
                _contributionLedgerWriter = null;
            }
            _confidenceAuditRows = null;
            _confidenceAuditFrameRows = null;
            _doubleLayerPairRows = null;
        }
    }

    private void AnalyzeAuditMeshAssociation()
    {
        _auditMeshAssociatedVoxelCount = 0;
        _auditSuspectAssociatedConflictVoxelCount = 0;
        _auditSuspectVoxelsLinkedToConflictCount = 0;
        if (_mesh == null || _captureAuditVoxels.Count <= 0)
            return;

        HashSet<int> meshVoxels = new HashSet<int>();
        Vector3[] meshVertices = _mesh.vertices;
        for (int i = 0; i < meshVertices.Length; i++)
        {
            Vector3 world = _meshObject != null ? _meshObject.transform.TransformPoint(meshVertices[i]) : meshVertices[i];
            if (TryWorldToVoxel(world, out int x, out int y, out int z))
                meshVoxels.Add(Index(x, y, z));
        }

        foreach (int voxelIndex in _captureAuditVoxels)
        {
            if (!_voxelAuditLifecycle.TryGetValue(voxelIndex, out VoxelAuditLifecycle life))
                continue;
            IndexToVoxel(voxelIndex, out int cx, out int cy, out int cz);
            int nearby = 0;
            for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                int x = cx + dx;
                int y = cy + dy;
                int z = cz + dz;
                if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                    continue;
                if (meshVoxels.Contains(Index(x, y, z)))
                    nearby++;
            }
            life.MeshNearbyVertices = nearby;
            life.SuspectMeshNearby = NeighborhoodContainsVoxel(_auditSuspectMeshVoxels, cx, cy, cz, 1);
            life.PureGeometryMeshNearby = NeighborhoodContainsVoxel(_auditPureGeometryMeshVoxels, cx, cy, cz, 1);
            _voxelAuditLifecycle[voxelIndex] = life;
            if (nearby > 0)
                _auditMeshAssociatedVoxelCount++;
            if (life.SuspectMeshNearby)
                _auditSuspectAssociatedConflictVoxelCount++;
            if (life.PureGeometryMeshNearby)
                _auditPureGeometryAssociatedConflictVoxelCount++;
        }

        foreach (int suspectVoxel in _auditSuspectMeshVoxels)
        {
            IndexToVoxel(suspectVoxel, out int x, out int y, out int z);
            if (NeighborhoodContainsVoxel(_captureAuditVoxels, x, y, z, 1))
                _auditSuspectVoxelsLinkedToConflictCount++;
        }
        foreach (int pureVoxel in _auditPureGeometryMeshVoxels)
        {
            IndexToVoxel(pureVoxel, out int x, out int y, out int z);
            if (NeighborhoodContainsVoxel(_captureAuditVoxels, x, y, z, 1))
                _auditPureGeometryVoxelsLinkedToConflictCount++;
        }
    }

    private bool NeighborhoodContainsVoxel(HashSet<int> voxels, int cx, int cy, int cz, int radius)
    {
        if (voxels == null || voxels.Count <= 0)
            return false;
        int r = Mathf.Max(0, radius);
        for (int dz = -r; dz <= r; dz++)
        for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
        {
            int x = cx + dx;
            int y = cy + dy;
            int z = cz + dz;
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;
            if (voxels.Contains(Index(x, y, z)))
                return true;
        }
        return false;
    }

    private string BuildVoxelAuditLifecycleCsv()
    {
        StringBuilder csv = new StringBuilder(Mathf.Max(1024, _captureAuditVoxels.Count * 120));
        csv.AppendLine("voxel_index,x,y,z,world_x,world_y,world_z,first_conflict_frame,last_conflict_frame,conflicts,replaces,conflicts_after_replace,last_replace_frame,old_weight,old_tsdf,sample_tsdf,residual,cause,lock_subtype,correction_hits,band_conflict_ratio,band_high_weight_conflict_ratio,history_agreement,historical_surface_distance_m,prior_write_known,prior_write_frame,prior_write_frame_gap,prior_camera_travel_m,prior_camera_rotation_delta_deg,prior_ray_angle_delta_deg,prior_depth_delta_m,prior_normal_angle_delta_deg,prior_surface_shift_m,mesh_nearby_voxels,suspect_mesh_nearby,pure_geometry_mesh_nearby");
        _auditRecurrentAfterReplaceCount = 0;
        _auditLikelyDepthInstabilityCount = 0;
        _auditViewOrRegistrationCount = 0;
        _auditLockedAboveCorrectableWeightCount = 0;
        _auditLockedBandConflictCount = 0;
        _auditLockedReplaceRecurrentCount = 0;
        _auditLockedDepthUnstableCount = 0;
        _auditLockedViewChangedCount = 0;
        _auditLockedUnderSupportedCount = 0;
        _auditLockedStableOldSurfaceCount = 0;
        foreach (int voxelIndex in _captureAuditVoxels)
        {
            if (!_voxelAuditLifecycle.TryGetValue(voxelIndex, out VoxelAuditLifecycle life))
                continue;
            IndexToVoxel(voxelIndex, out int x, out int y, out int z);
            Vector3 center = VoxelCenter(x, y, z);
            if (life.ConflictAfterReplaceCount > 0)
                _auditRecurrentAfterReplaceCount++;
            if (life.PriorWriteKnown && life.PriorSurfaceShiftMeters >= 0.08f)
            {
                if (life.PriorCameraTravelMeters < 0.08f)
                    _auditLikelyDepthInstabilityCount++;
                else
                    _auditViewOrRegistrationCount++;
            }
            CountOldWeightLockSubtype(life.LastLockSubtype);
            csv.Append(voxelIndex).Append(',').Append(x).Append(',').Append(y).Append(',').Append(z).Append(',')
                .Append(AuditFloat(center.x)).Append(',').Append(AuditFloat(center.y)).Append(',').Append(AuditFloat(center.z)).Append(',')
                .Append(life.FirstConflictFrame).Append(',').Append(life.LastConflictFrame).Append(',')
                .Append(life.ConflictCount).Append(',').Append(life.ReplaceCount).Append(',')
                .Append(life.ConflictAfterReplaceCount).Append(',').Append(life.LastReplaceFrame).Append(',')
                .Append(life.LastOldWeight).Append(',').Append(AuditFloat(life.LastOldTsdf)).Append(',')
                .Append(AuditFloat(life.LastSampleTsdf)).Append(',').Append(AuditFloat(life.LastResidual)).Append(',')
                .Append(life.LastCause).Append(',').Append(life.LastLockSubtype).Append(',')
                .Append(life.LastCorrectionHits).Append(',').Append(AuditFloat(life.LastBandConflictRatio)).Append(',')
                .Append(AuditFloat(life.LastBandHighWeightConflictRatio)).Append(',')
                .Append(AuditFloat(life.LastHistoryAgreement)).Append(',')
                .Append(AuditFloat(life.LastHistoricalSurfaceDistanceMeters)).Append(',')
                .Append(life.PriorWriteKnown ? 1 : 0).Append(',')
                .Append(life.PriorWriteFrame).Append(',').Append(life.PriorWriteFrameGap).Append(',')
                .Append(AuditFloat(life.PriorCameraTravelMeters)).Append(',')
                .Append(AuditFloat(life.PriorCameraRotationDeltaDegrees)).Append(',')
                .Append(AuditFloat(life.PriorRayAngleDeltaDegrees)).Append(',')
                .Append(AuditFloat(life.PriorDepthDeltaMeters)).Append(',')
                .Append(AuditFloat(life.PriorNormalAngleDeltaDegrees)).Append(',')
                .Append(AuditFloat(life.PriorSurfaceShiftMeters)).Append(',')
                .Append(life.MeshNearbyVertices).Append(',').Append(life.SuspectMeshNearby ? 1 : 0).Append(',')
                .Append(life.PureGeometryMeshNearby ? 1 : 0).AppendLine();
        }
        return csv.ToString();
    }

    private void CountOldWeightLockSubtype(string subtype)
    {
        switch (subtype)
        {
            case "above_correctable_weight":
                _auditLockedAboveCorrectableWeightCount++;
                break;
            case "band_conflict":
                _auditLockedBandConflictCount++;
                break;
            case "replace_recurrent":
                _auditLockedReplaceRecurrentCount++;
                break;
            case "depth_unstable":
                _auditLockedDepthUnstableCount++;
                break;
            case "view_changed":
                _auditLockedViewChangedCount++;
                break;
            case "under_supported_pending":
                _auditLockedUnderSupportedCount++;
                break;
            case "stable_old_surface":
                _auditLockedStableOldSurfaceCount++;
                break;
        }
    }

    private string ResolveConfidenceAuditDirectory()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        // APK contents are read-only. persistentDataPath is this installed app's writable data directory.
        return Path.Combine(Application.persistentDataPath, SanitizeAuditFolderName());
#else
        string installDirectory = Application.dataPath;
        try
        {
            DirectoryInfo dataDirectory = Directory.GetParent(Application.dataPath);
            if (dataDirectory != null)
                installDirectory = dataDirectory.FullName;
        }
        catch (System.Exception)
        {
            // Application.dataPath remains a usable fallback candidate.
        }
        return Path.Combine(installDirectory, SanitizeAuditFolderName());
#endif
    }

    private string SanitizeAuditFolderName()
    {
        string folder = string.IsNullOrWhiteSpace(confidenceAuditFolderName) ? "ScanCoverDiagnostics" : confidenceAuditFolderName.Trim();
        foreach (char invalid in Path.GetInvalidFileNameChars())
            folder = folder.Replace(invalid, '_');
        return folder;
    }

    private static string AuditFloat(float value)
    {
        return float.IsNaN(value) || float.IsInfinity(value) ? string.Empty : value.ToString("0.######", AuditCulture);
    }

    private void ResetRawDepthDebugCounters()
    {
        LastRawDepthDebugRenderedCount = 0;
        LastRawDepthDebugAcceptedCount = 0;
        LastRawDepthDebugDirtyCount = 0;
        LastRawDepthDebugPendingCount = 0;
        LastRawDepthDebugStableBandCount = 0;
        LastRawDepthDebugConflictEdgeCount = 0;
        LastRawDepthDebugConflictViewCount = 0;
        LastRawDepthDebugConflictLockedCount = 0;
        LastRawDepthDebugRejectedCount = 0;
        LastRawDepthDebugDepthEdgeCount = 0;
        LastRawDepthDebugRobustCount = 0;
        LastRawDepthDebugSupportCount = 0;
        LastRawDepthDebugOutsideCount = 0;
    }

    private void ClearRawDepthDebugCache()
    {
        _rawDepthDebugPoints.Clear();
        _rawDepthDebugKinds.Clear();
        _rawDepthDebugReplaceCursor = 0;
        _acceptedRawDepthDebugCounter = 0;
        _rawDepthDebugDepthEdgeCounter = 0;
        _rawDepthDebugRobustCounter = 0;
        _rawDepthDebugRejectedCounter = 0;
    }

    private void ResetRawCoverageGridDiagnostics()
    {
        _rawCoverageGridCells.Clear();
        _rawCoverageGridVersion++;
        _rawCoverageOverlayBuildVersion = -1;
        LastRawCoverageGridSampleCount = 0;
        LastRawCoverageGridDroppedSampleCount = 0;
        LastRawCoverageGridCellCount = 0;
        LastRawCoverageAcceptedCellCount = 0;
        LastRawCoverageProblemCellCount = 0;
        LastRawCoverageMixedCellCount = 0;
        LastRawCoverageAcceptedComponentCount = 0;
        LastRawCoverageLargestAcceptedComponentCells = 0;
        LastRawCoverageProblemComponentCount = 0;
        LastRawCoverageLargestProblemComponentCells = 0;
        LastRawCoverageRenderedCellCount = 0;
    }

    private void RecordRawCoverageGridSample(Vector3 point, RawDepthDebugKind kind, bool accepted)
    {
        if (!enableRawCoverageGridDiagnostics || !Finite(point))
            return;

        LastRawCoverageGridSampleCount++;
        Vector3Int key = RawCoverageGridKey(point);
        if (!_rawCoverageGridCells.TryGetValue(key, out RawCoverageGridCell cell))
        {
            if (_rawCoverageGridCells.Count >= Mathf.Max(1, maxRawCoverageGridCells))
            {
                LastRawCoverageGridDroppedSampleCount++;
                return;
            }

            cell = new RawCoverageGridCell
            {
                PositionSum = Vector3.zero,
                AcceptedCount = 0,
                ProblemCount = 0,
                WorstKind = kind
            };
        }

        cell.PositionSum += point;
        if (accepted)
            cell.AcceptedCount++;
        else
            cell.ProblemCount++;

        if (RawDepthDebugPriority(kind) > RawDepthDebugPriority(cell.WorstKind))
            cell.WorstKind = kind;

        _rawCoverageGridCells[key] = cell;
        _rawCoverageGridVersion++;
        _rawCoverageOverlayBuildVersion = -1;
    }

    private Vector3Int RawCoverageGridKey(Vector3 point)
    {
        float cellSize = Mathf.Max(0.03f, rawCoverageGridCellSizeMeters);
        return new Vector3Int(
            Mathf.FloorToInt(point.x / cellSize),
            Mathf.FloorToInt(point.y / cellSize),
            Mathf.FloorToInt(point.z / cellSize));
    }

    private void FinalizeRawCoverageGridDiagnostics()
    {
        if (!enableRawCoverageGridDiagnostics)
            return;

        LastRawCoverageGridCellCount = _rawCoverageGridCells.Count;
        HashSet<Vector3Int> acceptedCells = new HashSet<Vector3Int>();
        HashSet<Vector3Int> problemCells = new HashSet<Vector3Int>();
        LastRawCoverageAcceptedCellCount = 0;
        LastRawCoverageProblemCellCount = 0;
        LastRawCoverageMixedCellCount = 0;

        foreach (KeyValuePair<Vector3Int, RawCoverageGridCell> pair in _rawCoverageGridCells)
        {
            bool hasAccepted = pair.Value.AcceptedCount > 0;
            bool hasProblem = pair.Value.ProblemCount > 0;
            if (hasAccepted)
            {
                acceptedCells.Add(pair.Key);
                LastRawCoverageAcceptedCellCount++;
            }
            if (hasProblem)
            {
                problemCells.Add(pair.Key);
                LastRawCoverageProblemCellCount++;
            }
            if (hasAccepted && hasProblem)
                LastRawCoverageMixedCellCount++;
        }

        LastRawCoverageAcceptedComponentCount = CountRawCoverageComponents(acceptedCells, out int largestAccepted);
        LastRawCoverageLargestAcceptedComponentCells = largestAccepted;
        LastRawCoverageProblemComponentCount = CountRawCoverageComponents(problemCells, out int largestProblem);
        LastRawCoverageLargestProblemComponentCells = largestProblem;
    }

    private int CountRawCoverageComponents(HashSet<Vector3Int> cells, out int largest)
    {
        largest = 0;
        if (cells == null || cells.Count <= 0)
            return 0;

        int components = 0;
        HashSet<Vector3Int> visited = new HashSet<Vector3Int>();
        Queue<Vector3Int> queue = new Queue<Vector3Int>();
        Vector3Int[] offsets =
        {
            new Vector3Int(1, 0, 0), new Vector3Int(-1, 0, 0),
            new Vector3Int(0, 1, 0), new Vector3Int(0, -1, 0),
            new Vector3Int(0, 0, 1), new Vector3Int(0, 0, -1)
        };

        foreach (Vector3Int seed in cells)
        {
            if (visited.Contains(seed))
                continue;

            components++;
            int size = 0;
            visited.Add(seed);
            queue.Enqueue(seed);
            while (queue.Count > 0)
            {
                Vector3Int current = queue.Dequeue();
                size++;
                for (int i = 0; i < offsets.Length; i++)
                {
                    Vector3Int next = current + offsets[i];
                    if (!cells.Contains(next) || visited.Contains(next))
                        continue;
                    visited.Add(next);
                    queue.Enqueue(next);
                }
            }

            if (size > largest)
                largest = size;
        }

        return components;
    }

    private void BuildRawCoverageGridDebugCache()
    {
        if (!enableRawCoverageGridDiagnostics || !showRawCoverageGridOverlay)
            return;
        if (_rawCoverageOverlayBuildVersion == _rawCoverageGridVersion)
            return;

        ClearRawDepthDebugCache();
        int limit = Mathf.Max(1, maxRawDepthDebugSamples);
        int acceptedStride = Mathf.Max(1, rawCoverageAcceptedDisplayStride);
        int acceptedCounter = 0;
        foreach (KeyValuePair<Vector3Int, RawCoverageGridCell> pair in _rawCoverageGridCells)
        {
            RawCoverageGridCell cell = pair.Value;
            bool hasAccepted = cell.AcceptedCount > 0;
            bool hasProblem = cell.ProblemCount > 0;
            RawDepthDebugKind kind = hasProblem ? cell.WorstKind : RawDepthDebugKind.Accepted;
            if (hasAccepted && !hasProblem)
            {
                acceptedCounter++;
                if (acceptedCounter % acceptedStride != 0)
                    continue;
            }

            int total = Mathf.Max(1, cell.AcceptedCount + cell.ProblemCount);
            _rawDepthDebugPoints.Add(cell.PositionSum / total);
            _rawDepthDebugKinds.Add(kind);
            if (_rawDepthDebugPoints.Count >= limit)
                break;
        }

        LastRawCoverageRenderedCellCount = _rawDepthDebugPoints.Count;
        _rawCoverageOverlayBuildVersion = _rawCoverageGridVersion;
    }

    private void CacheRawDepthDebugSample(Vector3 point, RawDepthDebugKind kind)
    {
        if (!showRawDepthDebugView || !Finite(point))
            return;
        const int classifierRevision = 6;
        if (_rawDepthDebugClassifierRevision != classifierRevision)
        {
            ClearRawDepthDebugCache();
            _rawDepthDebugClassifierRevision = classifierRevision;
        }
        if (restrictRawDepthDebugToNearTsdfSurface && !RawDepthDebugPointTouchesNearTsdfSurface(point))
            return;
        if (kind == RawDepthDebugKind.Accepted)
        {
            if (!showAcceptedRawDepthDebugSamples)
                return;
            _acceptedRawDepthDebugCounter++;
            if (_acceptedRawDepthDebugCounter % Mathf.Max(1, acceptedRawDepthDebugSampleStride) != 0)
                return;
        }
        else if (kind == RawDepthDebugKind.DepthEdge)
        {
            _rawDepthDebugDepthEdgeCounter++;
            if (_rawDepthDebugDepthEdgeCounter % Mathf.Max(1, rawDepthDebugDepthEdgeSampleStride) != 0)
                return;
        }
        else if (kind == RawDepthDebugKind.Robust)
        {
            _rawDepthDebugRobustCounter++;
            if (_rawDepthDebugRobustCounter % Mathf.Max(1, rawDepthDebugRobustSampleStride) != 0)
                return;
        }
        else if (kind == RawDepthDebugKind.Rejected)
        {
            _rawDepthDebugRejectedCounter++;
            if (_rawDepthDebugRejectedCounter % Mathf.Max(1, rawDepthDebugRejectedSampleStride) != 0)
                return;
        }

        int limit = Mathf.Max(1, maxRawDepthDebugSamples);
        if (_rawDepthDebugPoints.Count < limit)
        {
            _rawDepthDebugPoints.Add(point);
            _rawDepthDebugKinds.Add(kind);
            return;
        }

        int index = FindRawDepthDebugReplacementIndex(kind, limit);
        _rawDepthDebugPoints[index] = point;
        _rawDepthDebugKinds[index] = kind;
        _rawDepthDebugReplaceCursor = (index + 1) % limit;
    }

    private int FindRawDepthDebugReplacementIndex(RawDepthDebugKind incomingKind, int limit)
    {
        limit = Mathf.Max(1, Mathf.Min(limit, _rawDepthDebugKinds.Count));
        int incomingPriority = RawDepthDebugPriority(incomingKind);
        int start = Mathf.Abs(_rawDepthDebugReplaceCursor) % limit;
        int best = start;
        int bestPriority = int.MaxValue;
        for (int i = 0; i < limit; i++)
        {
            int index = (start + i) % limit;
            int priority = RawDepthDebugPriority(_rawDepthDebugKinds[index]);
            if (priority < incomingPriority)
                return index;
            if (priority < bestPriority)
            {
                best = index;
                bestPriority = priority;
            }
        }

        return bestPriority <= incomingPriority ? best : start;
    }

    private static int RawDepthDebugPriority(RawDepthDebugKind kind)
    {
        switch (kind)
        {
            case RawDepthDebugKind.Dirty:
                return 6;
            case RawDepthDebugKind.ConflictLocked:
            case RawDepthDebugKind.ConflictView:
            case RawDepthDebugKind.ConflictEdge:
                return 6;
            case RawDepthDebugKind.Pending:
                return 5;
            case RawDepthDebugKind.StableBand:
                return 2;
            case RawDepthDebugKind.Support:
                return 4;
            case RawDepthDebugKind.Outside:
                return 3;
            case RawDepthDebugKind.Robust:
                return 2;
            case RawDepthDebugKind.DepthEdge:
                return 1;
            case RawDepthDebugKind.Rejected:
            case RawDepthDebugKind.Accepted:
            default:
                return 0;
        }
    }

    private bool RawDepthDebugPointTouchesNearTsdfSurface(Vector3 point)
    {
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(point, out int x, out int y, out int z))
            return false;

        int radius = Mathf.Clamp(rawDepthDebugDirtyRadiusVoxels, 0, 2);
        float maxAbsTsdf = Mathf.Clamp(rawDepthDebugNearSurfaceAbsTsdf, 0.01f, 0.8f);
        for (int dz = -radius; dz <= radius; dz++)
        {
            int vz = z + dz;
            if (vz < 0 || vz >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int vy = y + dy;
                if (vy < 0 || vy >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    if (vx < 0 || vx >= _dimX)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (_weights[index] < minSurfaceCornerWeight)
                        continue;
                    if (Mathf.Abs(_tsdf[index]) <= maxAbsTsdf)
                        return true;
                    if (VoxelHasPendingTsdfCorrection(index) || VoxelIsDirtyQuarantined(index))
                        return true;
                }
            }
        }

        return false;
    }

    private bool TryGetRawDepthDebugDirtyKind(Vector3 point, out RawDepthDebugKind kind)
    {
        kind = RawDepthDebugKind.Rejected;
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(point, out int x, out int y, out int z))
            return false;

        int radius = Mathf.Clamp(rawDepthDebugNearSurfaceRadiusVoxels, 0, 3);
        bool pending = false;
        for (int dz = -radius; dz <= radius; dz++)
        {
            int vz = z + dz;
            if (vz < 0 || vz >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int vy = y + dy;
                if (vy < 0 || vy >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    if (vx < 0 || vx >= _dimX)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (VoxelIsDirtyQuarantined(index))
                    {
                        kind = RawDepthDebugKind.Dirty;
                        return true;
                    }
                    if (VoxelHasPendingTsdfCorrection(index))
                        pending = true;
                }
            }
        }

        if (pending)
        {
            kind = RawDepthDebugKind.Pending;
            return true;
        }

        return false;
    }

    private void AddRawDepthDebugSample(
        List<Vector3> vertices,
        List<int> triangles,
        List<Color> colors,
        Vector3 point,
        Vector3 cameraPosition,
        RawDepthDebugKind kind)
    {
        if ((!showRawDepthDebugView && !showRawCoverageGridOverlay) || vertices == null || triangles == null || colors == null)
            return;
        if (LastRawDepthDebugRenderedCount >= Mathf.Max(1, maxRawDepthDebugSamples))
            return;
        if (!Finite(point))
            return;

        Color color = RawDepthDebugColor(kind);
        color.a = Mathf.Clamp01(rawDepthDebugAlpha);

        Vector3 view = point - cameraPosition;
        if (!Finite(view) || view.sqrMagnitude < 0.0001f)
            view = Vector3.forward;
        view.Normalize();

        Vector3 tangent = Vector3.Cross(Vector3.up, view);
        if (tangent.sqrMagnitude < 0.0001f)
            tangent = Vector3.Cross(Vector3.right, view);
        tangent.Normalize();
        Vector3 bitangent = Vector3.Cross(view, tangent).normalized;

        float half = Mathf.Max(0.001f, rawDepthDebugSampleSizeMeters) * 0.5f;
        int baseIndex = vertices.Count;
        vertices.Add(point - tangent * half - bitangent * half);
        vertices.Add(point + tangent * half - bitangent * half);
        vertices.Add(point + tangent * half + bitangent * half);
        vertices.Add(point - tangent * half + bitangent * half);
        AddVertexColors(colors, color, 4);

        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 1);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex + 3);

        LastRawDepthDebugRenderedCount++;
        switch (kind)
        {
            case RawDepthDebugKind.Accepted:
                LastRawDepthDebugAcceptedCount++;
                break;
            case RawDepthDebugKind.Dirty:
                LastRawDepthDebugDirtyCount++;
                break;
            case RawDepthDebugKind.Pending:
                LastRawDepthDebugPendingCount++;
                break;
            case RawDepthDebugKind.StableBand:
                LastRawDepthDebugStableBandCount++;
                break;
            case RawDepthDebugKind.ConflictEdge:
                LastRawDepthDebugConflictEdgeCount++;
                break;
            case RawDepthDebugKind.ConflictView:
                LastRawDepthDebugConflictViewCount++;
                break;
            case RawDepthDebugKind.ConflictLocked:
                LastRawDepthDebugConflictLockedCount++;
                break;
            case RawDepthDebugKind.DepthEdge:
                LastRawDepthDebugDepthEdgeCount++;
                break;
            case RawDepthDebugKind.Robust:
                LastRawDepthDebugRobustCount++;
                break;
            case RawDepthDebugKind.Support:
                LastRawDepthDebugSupportCount++;
                break;
            case RawDepthDebugKind.Outside:
                LastRawDepthDebugOutsideCount++;
                break;
            case RawDepthDebugKind.Rejected:
            default:
                LastRawDepthDebugRejectedCount++;
                break;
        }
    }

    private void AddRawDepthDebugCubeSample(
        List<Vector3> vertices,
        List<int> triangles,
        List<Color> colors,
        Vector3 point,
        RawDepthDebugKind kind)
    {
        if ((!showRawDepthDebugView && !showRawCoverageGridOverlay) || vertices == null || triangles == null || colors == null)
            return;
        if (LastRawDepthDebugRenderedCount >= Mathf.Max(1, maxRawDepthDebugSamples))
            return;
        if (!Finite(point))
            return;

        Color color = RawDepthDebugColor(kind);
        color.a = Mathf.Clamp01(rawDepthDebugAlpha);

        float half = Mathf.Max(0.001f, rawDepthDebugSampleSizeMeters) * 0.5f;
        int baseIndex = vertices.Count;
        vertices.Add(point + new Vector3(-half, -half, -half));
        vertices.Add(point + new Vector3( half, -half, -half));
        vertices.Add(point + new Vector3( half,  half, -half));
        vertices.Add(point + new Vector3(-half,  half, -half));
        vertices.Add(point + new Vector3(-half, -half,  half));
        vertices.Add(point + new Vector3( half, -half,  half));
        vertices.Add(point + new Vector3( half,  half,  half));
        vertices.Add(point + new Vector3(-half,  half,  half));
        AddVertexColors(colors, color, 8);

        AddCubeFace(triangles, baseIndex, 0, 2, 1, 3);
        AddCubeFace(triangles, baseIndex, 4, 5, 6, 7);
        AddCubeFace(triangles, baseIndex, 0, 1, 5, 4);
        AddCubeFace(triangles, baseIndex, 2, 3, 7, 6);
        AddCubeFace(triangles, baseIndex, 1, 2, 6, 5);
        AddCubeFace(triangles, baseIndex, 3, 0, 4, 7);

        LastRawDepthDebugRenderedCount++;
        switch (kind)
        {
            case RawDepthDebugKind.Accepted:
                LastRawDepthDebugAcceptedCount++;
                break;
            case RawDepthDebugKind.Dirty:
                LastRawDepthDebugDirtyCount++;
                break;
            case RawDepthDebugKind.Pending:
                LastRawDepthDebugPendingCount++;
                break;
            case RawDepthDebugKind.StableBand:
                LastRawDepthDebugStableBandCount++;
                break;
            case RawDepthDebugKind.ConflictEdge:
                LastRawDepthDebugConflictEdgeCount++;
                break;
            case RawDepthDebugKind.ConflictView:
                LastRawDepthDebugConflictViewCount++;
                break;
            case RawDepthDebugKind.ConflictLocked:
                LastRawDepthDebugConflictLockedCount++;
                break;
            case RawDepthDebugKind.DepthEdge:
                LastRawDepthDebugDepthEdgeCount++;
                break;
            case RawDepthDebugKind.Robust:
                LastRawDepthDebugRobustCount++;
                break;
            case RawDepthDebugKind.Support:
                LastRawDepthDebugSupportCount++;
                break;
            case RawDepthDebugKind.Outside:
                LastRawDepthDebugOutsideCount++;
                break;
            case RawDepthDebugKind.Rejected:
            default:
                LastRawDepthDebugRejectedCount++;
                break;
        }
    }

    private static void AddCubeFace(List<int> triangles, int baseIndex, int a, int b, int c, int d)
    {
        triangles.Add(baseIndex + a);
        triangles.Add(baseIndex + b);
        triangles.Add(baseIndex + c);
        triangles.Add(baseIndex + a);
        triangles.Add(baseIndex + c);
        triangles.Add(baseIndex + d);
    }

    private Color RawDepthDebugColor(RawDepthDebugKind kind)
    {
        switch (kind)
        {
            case RawDepthDebugKind.Accepted:
                return rawDepthDebugAcceptedColor;
            case RawDepthDebugKind.Dirty:
                return rawDepthDebugDirtyColor;
            case RawDepthDebugKind.Pending:
                return rawDepthDebugPendingColor;
            case RawDepthDebugKind.StableBand:
                return rawDepthDebugStableBandColor;
            case RawDepthDebugKind.ConflictEdge:
                return rawDepthDebugConflictEdgeColor;
            case RawDepthDebugKind.ConflictView:
                return rawDepthDebugConflictViewColor;
            case RawDepthDebugKind.ConflictLocked:
                return rawDepthDebugConflictLockedColor;
            case RawDepthDebugKind.DepthEdge:
                return rawDepthDebugDepthEdgeColor;
            case RawDepthDebugKind.Robust:
                return rawDepthDebugRobustColor;
            case RawDepthDebugKind.Support:
                return rawDepthDebugSupportColor;
            case RawDepthDebugKind.Outside:
                return rawDepthDebugOutsideColor;
            case RawDepthDebugKind.Rejected:
            default:
                return rawDepthDebugRejectedColor;
        }
    }

    private struct HoleBoundaryMarkerCandidate
    {
        public Vector3 Center;
        public Color Color;
        public float ViewScore;
        public int Priority;
        public float SizeScale;
    }

    private struct HoleSideRepairProposal
    {
        public int Index;
        public float Tsdf;
    }

    private void RebuildHoleBoundaryDiagnostic(int cellX, int cellY, int cellZ, int meshVertexCount, List<int> meshTriangles)
    {
        LastHoleCauseNoRawCount = 0;
        LastHoleCausePendingCount = 0;
        LastHoleCauseNoBandCount = 0;
        LastHoleCauseNoZeroCrossCount = 0;
        LastHoleCausePositiveOnlyCount = 0;
        LastHoleCauseNegativeOnlyCount = 0;
        LastMissingSideNoVisitCount = 0;
        LastMissingSideProvisionalCount = 0;
        LastMissingSideBlockedCount = 0;
        LastMissingSideAcceptLostCount = 0;
        LastMissingSidePendingNoPromoteCount = 0;
        LastMissingSideRejectCount = 0;
        LastMissingSideOtherBlockCount = 0;
        LastMissingSideAcceptedOverwrittenCount = 0;
        LastMissingSideAcceptedSpatialMissCount = 0;
        LastMissingSideAcceptedReflattenedCount = 0;
        LastReflattenedByCarveCount = 0;
        LastReflattenedByClearCount = 0;
        LastReflattenedByProvisionalCount = 0;
        LastReflattenedByOtherCount = 0;
        LastReflattenedByAcceptedWriterCount = 0;
        LastReflattenedByContinuityWriterCount = 0;
        LastReflattenedByFusionWriterCount = 0;
        LastReflattenedByUntrackedWriterCount = 0;
        LastMissingSideGoneCount = 0;
        LastMissingSideInvalidCount = 0;
        LastHoleCauseVertexRejectCount = 0;
        LastHoleCauseQuadRejectCount = 0;
        LastHoleCauseClearedCount = 0;
        LastHoleCauseLocalOffsetCount = 0;
        LastHoleCauseCrossFrameCount = 0;
        LastHoleBoundaryRenderedMarkerCount = 0;
        LastPrimaryLayerDiagnosticMarkerCount = 0;
        LastSecondaryLayerDiagnosticMarkerCount = 0;
        LastUnclassifiedLayerDiagnosticMarkerCount = 0;
        LastConfirmedSecondaryLayerVoxelCount = 0;
        EnsureHoleBoundaryDiagnosticObjects();
        if (_holeBoundaryDiagnosticMesh == null)
            return;

        _holeBoundaryDiagnosticMesh.Clear();
        if (!showHoleBoundaryDiagnosis || _cellVertexIndices == null)
        {
            _holeBoundaryDiagnosticObject.SetActive(false);
            return;
        }

        int limit = Mathf.Max(1, maxHoleBoundaryMarkers);
        List<Vector3> vertices = new List<Vector3>(limit * 4);
        List<int> triangles = new List<int>(limit * 6);
        List<Color> colors = new List<Color>(limit * 4);
        Vector3 cameraPosition = GetCameraPosition();
        Vector3 cameraForward = GetCameraForward();
        List<HoleBoundaryMarkerCandidate> markerCandidates = new List<HoleBoundaryMarkerCandidate>(limit * 2);
        bool[] vertexUsed = new bool[Mathf.Max(0, meshVertexCount)];
        if (meshTriangles != null)
        {
            for (int i = 0; i < meshTriangles.Count; i++)
            {
                int vertexIndex = meshTriangles[i];
                if (vertexIndex >= 0 && vertexIndex < vertexUsed.Length)
                    vertexUsed[vertexIndex] = true;
            }
        }
        for (int z = 0; z < cellZ; z++)
        {
            for (int y = 0; y < cellY; y++)
            {
                for (int x = 0; x < cellX; x++)
                {
                    int cellIndex = CellIndex(x, y, z, cellX, cellY);
                    int vertexIndex = _cellVertexIndices[cellIndex];
                    bool usedByMesh = vertexIndex >= 0 && vertexIndex < vertexUsed.Length && vertexUsed[vertexIndex];
                    if (usedByMesh)
                        continue;
                    if (vertexIndex < 0 && !CellTouchesUsedSurface(x, y, z, cellX, cellY, cellZ, vertexUsed))
                        continue;

                    HoleSupportCause cause = ClassifyHoleSupportCause(
                        x, y, z, vertexIndex, out bool positiveOnly, out bool negativeOnly);
                    Color color = HoleSupportCauseColor(cause);
                    if (positiveOnly)
                    {
                        LastHoleCausePositiveOnlyCount++;
                        CountMissingSideCause(x, y, z, true);
                    }
                    if (negativeOnly)
                    {
                        LastHoleCauseNegativeOnlyCount++;
                        CountMissingSideCause(x, y, z, false);
                    }
                    if (cause == HoleSupportCause.NoZeroCross && !positiveOnly && !negativeOnly)
                        LastMissingSideInvalidCount += 2;
                    switch (cause)
                    {
                        case HoleSupportCause.NoRaw:
                            LastHoleCauseNoRawCount++;
                            break;
                        case HoleSupportCause.Pending:
                            LastHoleCausePendingCount++;
                            break;
                        case HoleSupportCause.NoBand:
                            LastHoleCauseNoBandCount++;
                            break;
                        case HoleSupportCause.NoZeroCross:
                            LastHoleCauseNoZeroCrossCount++;
                            break;
                        case HoleSupportCause.VertexReject:
                            LastHoleCauseVertexRejectCount++;
                            break;
                        case HoleSupportCause.QuadReject:
                            LastHoleCauseQuadRejectCount++;
                            break;
                        case HoleSupportCause.Cleared:
                            LastHoleCauseClearedCount++;
                            break;
                        case HoleSupportCause.LocalOffset:
                            LastHoleCauseLocalOffsetCount++;
                            break;
                        case HoleSupportCause.CrossFrame:
                            LastHoleCauseCrossFrameCount++;
                            break;
                    }

                    Vector3 center = VoxelCenter(x, y, z) + Vector3.one * (Mathf.Max(0.0001f, voxelSizeMeters) * 0.5f);
                    DiagnosticLayerRole layerRole = ClassifyDiagnosticLayerRole(center);
                    if (layerRole == DiagnosticLayerRole.None)
                    {
                        LastUnclassifiedLayerDiagnosticMarkerCount++;
                        continue;
                    }
                    if (layerRole == DiagnosticLayerRole.Primary)
                    {
                        LastPrimaryLayerDiagnosticMarkerCount++;
                        if (!showAllLayerDiagnosticMarkers && !PassLayerDiagnosticSpatialSample(x, y, z, 3))
                            continue;
                        color = primaryLayerDiagnosticColor;
                    }
                    else
                    {
                        LastSecondaryLayerDiagnosticMarkerCount++;
                        if (!showAllLayerDiagnosticMarkers && !PassLayerDiagnosticSpatialSample(x, y, z, 3))
                            continue;
                        color = secondaryLayerDiagnosticColor;
                    }
                    Vector3 toMarker = center - cameraPosition;
                    float distanceSq = toMarker.sqrMagnitude;
                    float forwardDot = distanceSq > 0.000001f
                        ? Vector3.Dot(cameraForward, toMarker / Mathf.Sqrt(distanceSq))
                        : 1f;
                    // Prefer nearby markers in the current view. Markers behind the user
                    // remain eligible, but naturally leave the bounded render budget.
                    float viewPenalty = Mathf.Max(0f, 1f - forwardDot) * (distanceSq + 1f) * 4f;
                    markerCandidates.Add(new HoleBoundaryMarkerCandidate
                    {
                        Center = center,
                        Color = color,
                        ViewScore = distanceSq + viewPenalty,
                        Priority = 0,
                        SizeScale = primaryLayerDiagnosticSizeScale
                    });
                }
            }
        }

        if (showGlobalPairedLayerDiagnostics)
        {
            // Hole-cause accounting above remains intact for disk diagnostics. Rendering,
            // however, comes from every written near-surface voxel so both sides of a
            // competing layer pair remain visible even away from a mesh hole.
            markerCandidates.Clear();
            LastPrimaryLayerDiagnosticMarkerCount = 0;
            LastSecondaryLayerDiagnosticMarkerCount = 0;
            LastUnclassifiedLayerDiagnosticMarkerCount = 0;
            AppendGlobalPairedLayerDiagnosticCandidates(markerCandidates, cameraPosition, cameraForward);
        }
        UpdateLayerHoleTrend();

        markerCandidates.Sort((a, b) =>
        {
            int priority = a.Priority.CompareTo(b.Priority);
            return priority != 0 ? priority : a.ViewScore.CompareTo(b.ViewScore);
        });
        int renderLimit = showAllLayerDiagnosticMarkers
            ? Mathf.Max(12000, maxFullLayerDiagnosticMarkers)
            : limit;
        int renderedMarkers = Mathf.Min(renderLimit, markerCandidates.Count);
        for (int i = 0; i < renderedMarkers; i++)
        {
            HoleBoundaryMarkerCandidate candidate = markerCandidates[i];
            AddHoleBoundaryPoint(vertices, triangles, colors, candidate.Center, cameraPosition, candidate.Color, candidate.SizeScale);
        }
        LastHoleBoundaryRenderedMarkerCount = renderedMarkers;

        if (vertices.Count == 0)
        {
            _holeBoundaryDiagnosticObject.SetActive(false);
            return;
        }

        _holeBoundaryDiagnosticMesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _holeBoundaryDiagnosticMesh.SetVertices(vertices);
        _holeBoundaryDiagnosticMesh.SetColors(colors);
        _holeBoundaryDiagnosticMesh.SetTriangles(triangles, 0, true);
        _holeBoundaryDiagnosticMesh.RecalculateBounds();
        _holeBoundaryDiagnosticObject.SetActive(true);
    }

    private void AppendGlobalPairedLayerDiagnosticCandidates(
        List<HoleBoundaryMarkerCandidate> markerCandidates,
        Vector3 cameraPosition,
        Vector3 cameraForward)
    {
        if (_tsdf == null || _weights == null || _voxelWriteProvenance.Count == 0)
            return;

        float maxAbsTsdf = Mathf.Clamp(layerDiagnosticMaxAbsTsdf, 0.02f, 0.5f);
        foreach (KeyValuePair<int, VoxelWriteProvenance> entry in _voxelWriteProvenance)
        {
            int index = entry.Key;
            if (index < 0 || index >= _tsdf.Length || index >= _weights.Length ||
                _weights[index] < minSurfaceCornerWeight || Mathf.Abs(_tsdf[index]) > maxAbsTsdf)
                continue;

            IndexToVoxel(index, out int x, out int y, out int z);
            Vector3 center = VoxelCenter(x, y, z);
            DiagnosticLayerRole layerRole = ClassifyDiagnosticLayerRole(center);
            if (layerRole == DiagnosticLayerRole.Secondary)
                LastConfirmedSecondaryLayerVoxelCount++;
            if (layerRole == DiagnosticLayerRole.None)
            {
                LastUnclassifiedLayerDiagnosticMarkerCount++;
                // A trustworthy near-zero voxel without a detected competitor is the
                // visible baseline surface. Keep it in the primary layer display while
                // preserving the unclassified count in diagnostics.
                layerRole = DiagnosticLayerRole.Primary;
            }

            if (!showAllLayerDiagnosticMarkers && !PassLayerDiagnosticSpatialSample(x, y, z, 3))
                continue;

            if (layerRole == DiagnosticLayerRole.Secondary)
                LastSecondaryLayerDiagnosticMarkerCount++;
            else
                LastPrimaryLayerDiagnosticMarkerCount++;

            Vector3 toMarker = center - cameraPosition;
            float distanceSq = toMarker.sqrMagnitude;
            float forwardDot = distanceSq > 0.000001f
                ? Vector3.Dot(cameraForward, toMarker / Mathf.Sqrt(distanceSq))
                : 1f;
            float viewPenalty = Mathf.Max(0f, 1f - forwardDot) * (distanceSq + 1f) * 4f;
            markerCandidates.Add(new HoleBoundaryMarkerCandidate
            {
                Center = center,
                Color = layerRole == DiagnosticLayerRole.Secondary
                    ? secondaryLayerDiagnosticColor
                    : primaryLayerDiagnosticColor,
                ViewScore = distanceSq + viewPenalty,
                Priority = 0,
                SizeScale = primaryLayerDiagnosticSizeScale
            });
        }
    }

    private void UpdateLayerHoleTrend()
    {
        int frame = IntegratedFrameCount;
        int holeLoad = LastHoleCauseNoBandCount + LastHoleCauseNoZeroCrossCount +
            LastHoleCauseVertexRejectCount + LastHoleCauseQuadRejectCount;
        LastLayerHoleTrendHoleLoad = holeLoad;
        if (frame == _lastLayerHoleTrendIntegratedFrame)
            return;

        _lastLayerHoleTrendIntegratedFrame = frame;
        if (_layerHoleTrendSamples.Count > 0)
        {
            Vector2 previous = _layerHoleTrendSamples[_layerHoleTrendSamples.Count - 1];
            LastLayerHoleTrendSecondaryDelta = LastConfirmedSecondaryLayerVoxelCount - Mathf.RoundToInt(previous.x);
            LastLayerHoleTrendHoleDelta = holeLoad - Mathf.RoundToInt(previous.y);
        }
        else
        {
            LastLayerHoleTrendSecondaryDelta = 0;
            LastLayerHoleTrendHoleDelta = 0;
        }

        _layerHoleTrendSamples.Add(new Vector2(LastConfirmedSecondaryLayerVoxelCount, holeLoad));
        if (_layerHoleTrendSamples.Count > 24)
            _layerHoleTrendSamples.RemoveAt(0);
        LastLayerHoleTrendSampleCount = _layerHoleTrendSamples.Count;
        LastLayerHoleTrendCorrelation = ComputeLayerHoleTrendCorrelation(_layerHoleTrendSamples);
    }

    private static float ComputeLayerHoleTrendCorrelation(List<Vector2> samples)
    {
        if (samples == null || samples.Count < 3)
            return 0f;

        float meanX = 0f;
        float meanY = 0f;
        for (int i = 0; i < samples.Count; i++)
        {
            meanX += samples[i].x;
            meanY += samples[i].y;
        }
        meanX /= samples.Count;
        meanY /= samples.Count;

        float covariance = 0f;
        float varianceX = 0f;
        float varianceY = 0f;
        for (int i = 0; i < samples.Count; i++)
        {
            float dx = samples[i].x - meanX;
            float dy = samples[i].y - meanY;
            covariance += dx * dy;
            varianceX += dx * dx;
            varianceY += dy * dy;
        }
        float denominator = Mathf.Sqrt(varianceX * varianceY);
        return denominator > 0.0001f ? Mathf.Clamp(covariance / denominator, -1f, 1f) : 0f;
    }

    private string LayerHoleTrendRelationText()
    {
        if (LastLayerHoleTrendSampleCount < 3)
            return "WARMUP";
        if (LastLayerHoleTrendCorrelation >= 0.35f)
            return "COUPLED";
        if (LastLayerHoleTrendCorrelation <= -0.35f)
            return "TRADEOFF";
        return "WEAK";
    }

    private static bool PassLayerDiagnosticSpatialSample(int x, int y, int z, int stride)
    {
        stride = Mathf.Max(1, stride);
        if (stride == 1)
            return true;

        unchecked
        {
            uint hash = (uint)x * 73856093u;
            hash ^= (uint)y * 19349663u;
            hash ^= (uint)z * 83492791u;
            return hash % (uint)stride == 0u;
        }
    }

    private bool CellTouchesUsedSurface(int x, int y, int z, int cellX, int cellY, int cellZ, bool[] vertexUsed)
    {
        return CellVertexIsUsed(x - 1, y, z, cellX, cellY, cellZ, vertexUsed) ||
               CellVertexIsUsed(x + 1, y, z, cellX, cellY, cellZ, vertexUsed) ||
               CellVertexIsUsed(x, y - 1, z, cellX, cellY, cellZ, vertexUsed) ||
               CellVertexIsUsed(x, y + 1, z, cellX, cellY, cellZ, vertexUsed) ||
               CellVertexIsUsed(x, y, z - 1, cellX, cellY, cellZ, vertexUsed) ||
               CellVertexIsUsed(x, y, z + 1, cellX, cellY, cellZ, vertexUsed);
    }

    private bool CellVertexIsUsed(int x, int y, int z, int cellX, int cellY, int cellZ, bool[] vertexUsed)
    {
        if (x < 0 || y < 0 || z < 0 || x >= cellX || y >= cellY || z >= cellZ)
            return false;
        int vertexIndex = _cellVertexIndices[CellIndex(x, y, z, cellX, cellY)];
        return vertexIndex >= 0 && vertexIndex < vertexUsed.Length && vertexUsed[vertexIndex];
    }

    private HoleSupportCause ClassifyHoleSupportCause(
        int x, int y, int z, int vertexIndex, out bool positiveOnly, out bool negativeOnly)
    {
        positiveOnly = false;
        negativeOnly = false;
        bool pending = false;
        bool cleared = false;
        bool rawAtCell = false;
        int formalCorners = 0;
        bool hasPositive = false;
        bool hasNegative = false;
        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            int index = Index(x + dx, y + dy, z + dz);
            pending |= _atomicProvisionalBandWeight != null && _atomicProvisionalBandWeight[index] > 0;
            cleared |= _clearedVoxelLastFrame != null && _clearedVoxelLastFrame[index] != int.MinValue;
            rawAtCell |= _surfaceObservationState != null && _surfaceObservationState[index] > 0;
            if (_weights[index] <= 0)
                continue;
            formalCorners++;
            hasPositive |= _tsdf[index] >= 0f;
            hasNegative |= _tsdf[index] <= 0f;
        }

        if (cleared)
            return HoleSupportCause.Cleared;
        if (pending)
            return HoleSupportCause.Pending;
        if (!rawAtCell)
        {
            if (!TryFindNearestObservationFrame(x, y, z, 2, out int observationFrame))
                return HoleSupportCause.NoRaw;
            return CellHasFormalWriteFromFrame(x, y, z, observationFrame)
                ? HoleSupportCause.LocalOffset
                : (CellHasFormalWriteFromOtherFrame(x, y, z, observationFrame)
                    ? HoleSupportCause.CrossFrame
                    : HoleSupportCause.LocalOffset);
        }
        if (formalCorners == 0)
            return HoleSupportCause.NoBand;
        if (!hasPositive || !hasNegative)
        {
            positiveOnly = hasPositive && !hasNegative;
            negativeOnly = hasNegative && !hasPositive;
            return HoleSupportCause.NoZeroCross;
        }
        return vertexIndex < 0 ? HoleSupportCause.VertexReject : HoleSupportCause.QuadReject;
    }

    private bool TryFindNearestObservationFrame(int x, int y, int z, int radius, out int frame)
    {
        frame = int.MinValue;
        if (_surfaceObservationState == null || _surfaceObservationLastFrame == null)
            return false;
        int bestDistanceSquared = int.MaxValue;
        for (int vz = Mathf.Max(0, z - radius); vz <= Mathf.Min(_dimZ - 1, z + 1 + radius); vz++)
        for (int vy = Mathf.Max(0, y - radius); vy <= Mathf.Min(_dimY - 1, y + 1 + radius); vy++)
        for (int vx = Mathf.Max(0, x - radius); vx <= Mathf.Min(_dimX - 1, x + 1 + radius); vx++)
        {
            int index = Index(vx, vy, vz);
            if (_surfaceObservationState[index] <= 0 || _surfaceObservationLastFrame[index] == int.MinValue)
                continue;
            int dx = vx < x ? x - vx : (vx > x + 1 ? vx - (x + 1) : 0);
            int dy = vy < y ? y - vy : (vy > y + 1 ? vy - (y + 1) : 0);
            int dz = vz < z ? z - vz : (vz > z + 1 ? vz - (z + 1) : 0);
            int distanceSquared = dx * dx + dy * dy + dz * dz;
            if (distanceSquared >= bestDistanceSquared)
                continue;
            bestDistanceSquared = distanceSquared;
            frame = _surfaceObservationLastFrame[index];
        }
        return frame != int.MinValue;
    }

    private bool CellHasFormalWriteFromFrame(int x, int y, int z, int frame)
    {
        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            int index = Index(x + dx, y + dy, z + dz);
            if (_weights[index] > 0 &&
                _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) &&
                provenance.Frame == frame)
                return true;
        }
        return false;
    }

    private bool CellHasFormalWriteFromOtherFrame(int x, int y, int z, int frame)
    {
        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            int index = Index(x + dx, y + dy, z + dz);
            if (_weights[index] > 0 &&
                _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) &&
                provenance.Frame != int.MinValue && provenance.Frame != frame)
                return true;
        }
        return false;
    }

    private void CountMissingSideCause(int x, int y, int z, bool missingPositive)
    {
        switch (ClassifyMissingSideCause(x, y, z, missingPositive))
        {
            case MissingSideCause.Provisional: LastMissingSideProvisionalCount++; break;
            case MissingSideCause.AcceptLost:
                LastMissingSideAcceptLostCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.PendingNoPromote:
                LastMissingSidePendingNoPromoteCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.Reject:
                LastMissingSideRejectCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedOverwritten:
                LastMissingSideAcceptedOverwrittenCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedSpatialMiss:
                LastMissingSideAcceptedSpatialMissCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedCarve:
                LastReflattenedByCarveCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedClear:
                LastReflattenedByClearCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedProvisional:
                LastReflattenedByProvisionalCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedOther:
                LastReflattenedByOtherCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedAcceptedWriter:
                LastReflattenedByAcceptedWriterCount++;
                LastReflattenedByOtherCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedContinuityWriter:
                LastReflattenedByContinuityWriterCount++;
                LastReflattenedByOtherCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedFusionWriter:
                LastReflattenedByFusionWriterCount++;
                LastReflattenedByOtherCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.AcceptedReflattenedUntrackedWriter:
                LastReflattenedByUntrackedWriterCount++;
                LastReflattenedByOtherCount++;
                LastMissingSideAcceptedReflattenedCount++;
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.OtherBlock:
                LastMissingSideOtherBlockCount++;
                LastMissingSideBlockedCount++;
                break;
            case MissingSideCause.Gone: LastMissingSideGoneCount++; break;
            case MissingSideCause.Invalid: LastMissingSideInvalidCount++; break;
            default: LastMissingSideNoVisitCount++; break;
        }
    }

    private MissingSideCause ClassifyMissingSideCause(int x, int y, int z, bool positive)
    {
        byte acceptedFlag = positive ? (byte)1 : (byte)2;
        byte pendingFlag = positive ? (byte)4 : (byte)8;
        byte signFlag = positive ? (byte)1 : (byte)2;
        bool acceptedVisited = false;
        bool pendingVisited = false;
        bool rejectedVisited = false;
        bool acceptSignLost = false;
        int desiredRetainedLastSequence = int.MinValue;
        int oppositeAcceptedLastSequence = int.MinValue;
        int reflattenedOperationSequence = int.MinValue;
        string reflattenedOperation = null;
        int memory = Mathf.Max(1, holeBoundaryRetiredMemoryFrames);

        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            int index = Index(x + dx, y + dy, z + dz);
            if (_weights[index] > 0 && (float.IsNaN(_tsdf[index]) || float.IsInfinity(_tsdf[index])))
                return MissingSideCause.Invalid;
            if (_atomicProvisionalBandWeight != null && _atomicProvisionalBandWeight[index] > 0)
            {
                float provisional = _atomicProvisionalBandTsdf[index];
                if (float.IsNaN(provisional) || float.IsInfinity(provisional))
                    return MissingSideCause.Invalid;
                if ((positive && provisional >= 0f) || (!positive && provisional <= 0f))
                    return MissingSideCause.Provisional;
            }
            if (_clearedVoxelLastFrame != null && RecentDiagnosticFrame(_clearedVoxelLastFrame[index], memory))
                return MissingSideCause.Gone;
            if (_atomicProvisionalBandRetiredLastFrame != null &&
                _atomicProvisionalBandRetiredSign != null &&
                RecentDiagnosticFrame(_atomicProvisionalBandRetiredLastFrame[index], memory) &&
                (_atomicProvisionalBandRetiredSign[index] & signFlag) != 0)
                return MissingSideCause.Gone;
            if (_surfaceBandVisitFlags != null)
            {
                acceptedVisited |= (_surfaceBandVisitFlags[index] & acceptedFlag) != 0;
                pendingVisited |= (_surfaceBandVisitFlags[index] & pendingFlag) != 0;
                byte rejectFlag = positive ? (byte)16 : (byte)32;
                rejectedVisited |= (_surfaceBandVisitFlags[index] & rejectFlag) != 0;
            }
            if (_surfaceBandAcceptSignLostFlags != null)
                acceptSignLost |= (_surfaceBandAcceptSignLostFlags[index] & signFlag) != 0;
            int[] desiredRetained = positive ? _acceptedPositiveRetainedLastSequence : _acceptedNegativeRetainedLastSequence;
            int[] oppositeAccepted = positive ? _acceptedNegativeLastSequence : _acceptedPositiveLastSequence;
            if (desiredRetained != null)
            {
                desiredRetainedLastSequence = Mathf.Max(desiredRetainedLastSequence, desiredRetained[index]);
                bool currentlyMissingDesiredSign = _weights[index] <= 0 ||
                    (positive ? _tsdf[index] < 0f : _tsdf[index] > 0f);
                if (desiredRetained[index] != int.MinValue && currentlyMissingDesiredSign &&
                    _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) &&
                    provenance.WriteSequence >= reflattenedOperationSequence)
                {
                    reflattenedOperationSequence = provenance.WriteSequence;
                    reflattenedOperation = provenance.LastOperation;
                }
            }
            if (oppositeAccepted != null)
                oppositeAcceptedLastSequence = Mathf.Max(oppositeAcceptedLastSequence, oppositeAccepted[index]);
        }

        if (acceptedVisited)
        {
            if (acceptSignLost)
                return MissingSideCause.AcceptLost;
            if (desiredRetainedLastSequence == int.MinValue)
                return MissingSideCause.AcceptedSpatialMiss;
            if (oppositeAcceptedLastSequence > desiredRetainedLastSequence)
                return MissingSideCause.AcceptedOverwritten;
            return ClassifyReflattenedOperation(reflattenedOperation);
        }
        if (pendingVisited)
            return MissingSideCause.PendingNoPromote;
        if (rejectedVisited)
            return MissingSideCause.Reject;
        return MissingSideCause.NoVisit;
    }

    private MissingSideCause ClassifyReflattenedOperation(string operation)
    {
        if (operation == "free_space_carve")
            return MissingSideCause.AcceptedReflattenedCarve;
        if (operation == "old_clean_clear" || operation == "provisional_dirty_clear" ||
            operation == "provisional_retire" || operation == "cleanup_neighbor")
            return MissingSideCause.AcceptedReflattenedClear;
        if (operation == "provisional_support" || operation == "strong_sample_seed" ||
            operation == "same_frame_provisional")
            return MissingSideCause.AcceptedReflattenedProvisional;
        if (operation == "atomic_accept_band" || operation == "integrate" ||
            operation == "strong_current_promote" || operation == "v08_direct_diag")
            return MissingSideCause.AcceptedReflattenedAcceptedWriter;
        if (operation == "continuity_fill" || operation == "boundary_no_tsdf_fill")
            return MissingSideCause.AcceptedReflattenedContinuityWriter;
        if (operation == "replace" || operation == "guarded_replace" ||
            operation == "band_repair" || operation == "conflict_correct" ||
            operation == "old_clean_decay")
            return MissingSideCause.AcceptedReflattenedFusionWriter;
        if (string.IsNullOrEmpty(operation) || operation == "unknown")
            return MissingSideCause.AcceptedReflattenedUntrackedWriter;
        return MissingSideCause.AcceptedReflattenedUntrackedWriter;
    }

    private bool RecentDiagnosticFrame(int frame, int memory)
    {
        if (frame == int.MinValue)
            return false;
        int age = LastRawFrameIndex - frame;
        return age >= 0 && age <= memory;
    }

    private void MarkSurfaceBandVisit(int index, float sampleTsdf)
    {
        if (_surfaceBandVisitFlags == null || index < 0 || index >= _surfaceBandVisitFlags.Length)
            return;
        byte flags = 0;
        if (_activeAtomicBandVote == ObservationVoteState.Accept)
            flags = sampleTsdf >= 0f ? (byte)1 : (byte)2;
        else if (_activeAtomicBandVote == ObservationVoteState.Pending)
            flags = sampleTsdf >= 0f ? (byte)4 : (byte)8;
        else if (_activeAtomicBandVote == ObservationVoteState.Reject)
            flags = sampleTsdf >= 0f ? (byte)16 : (byte)32;
        if (Mathf.Approximately(sampleTsdf, 0f))
            flags |= _activeAtomicBandVote == ObservationVoteState.Accept ? (byte)3 :
                     _activeAtomicBandVote == ObservationVoteState.Pending ? (byte)12 :
                     _activeAtomicBandVote == ObservationVoteState.Reject ? (byte)48 : (byte)0;
        _surfaceBandVisitFlags[index] |= flags;
    }

    private Color HoleSupportCauseColor(HoleSupportCause cause)
    {
        switch (cause)
        {
            case HoleSupportCause.NoRaw: return holeCauseNoRawColor;
            case HoleSupportCause.Pending: return holeCausePendingColor;
            case HoleSupportCause.NoBand: return holeCauseNoBandColor;
            case HoleSupportCause.NoZeroCross: return holeCauseNoZeroCrossColor;
            case HoleSupportCause.VertexReject: return holeCauseVertexRejectColor;
            case HoleSupportCause.QuadReject: return holeCauseQuadRejectColor;
            case HoleSupportCause.Cleared: return holeCauseClearedColor;
            case HoleSupportCause.LocalOffset: return holeCauseFrameMismatchColor;
            default: return holeCauseCrossFrameColor;
        }
    }

    private HoleBoundaryDiagnosticKind ClassifyHoleBoundaryCell(int x, int y, int z)
    {
        bool waiting = false;
        bool retired = false;
        int formalCorners = 0;
        bool hasPositive = false;
        bool hasNegative = false;
        int retiredMemory = Mathf.Max(1, holeBoundaryRetiredMemoryFrames);
        for (int dz = 0; dz <= 1; dz++)
        {
            for (int dy = 0; dy <= 1; dy++)
            {
                for (int dx = 0; dx <= 1; dx++)
                {
                    int index = Index(x + dx, y + dy, z + dz);
                    if (_weights != null && index < _weights.Length && _weights[index] > 0)
                    {
                        formalCorners++;
                        if (_tsdf[index] >= 0f)
                            hasPositive = true;
                        if (_tsdf[index] <= 0f)
                            hasNegative = true;
                    }
                    if (_atomicProvisionalBandWeight != null && index < _atomicProvisionalBandWeight.Length && _atomicProvisionalBandWeight[index] > 0)
                        waiting = true;
                    if (_atomicProvisionalBandRetiredLastFrame != null && index < _atomicProvisionalBandRetiredLastFrame.Length)
                    {
                        int retiredFrame = _atomicProvisionalBandRetiredLastFrame[index];
                        int age = LastRawFrameIndex - retiredFrame;
                        if (retiredFrame != int.MinValue && age >= 0 && age <= retiredMemory)
                            retired = true;
                    }
                }
            }
        }

        if (waiting)
            return HoleBoundaryDiagnosticKind.WaitingPromotion;
        if (retired)
            return HoleBoundaryDiagnosticKind.Retired;
        if (formalCorners == 0)
            return HoleBoundaryDiagnosticKind.NoBand;
        if (hasPositive && hasNegative)
            return HoleBoundaryDiagnosticKind.WeakCorners;
        if (NeighborContainsOppositeTsdfSign(x, y, z, hasPositive))
            return HoleBoundaryDiagnosticKind.SpatialMiss;
        return hasPositive
            ? HoleBoundaryDiagnosticKind.PositiveOnly
            : HoleBoundaryDiagnosticKind.NegativeOnly;
    }

    private bool NeighborContainsOppositeTsdfSign(int cellX, int cellY, int cellZ, bool currentPositive)
    {
        int minX = Mathf.Max(0, cellX - 1);
        int minY = Mathf.Max(0, cellY - 1);
        int minZ = Mathf.Max(0, cellZ - 1);
        int maxX = Mathf.Min(_dimX - 1, cellX + 2);
        int maxY = Mathf.Min(_dimY - 1, cellY + 2);
        int maxZ = Mathf.Min(_dimZ - 1, cellZ + 2);
        for (int z = minZ; z <= maxZ; z++)
        {
            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    int index = Index(x, y, z);
                    if (_weights == null || index >= _weights.Length || _weights[index] == 0)
                        continue;
                    if (currentPositive ? _tsdf[index] < 0f : _tsdf[index] > 0f)
                        return true;
                }
            }
        }
        return false;
    }

    private DiagnosticLayerRole ClassifyDiagnosticLayerRole(Vector3 center)
    {
        if (!TryResolveDoubleLayerSide(center, out DoubleLayerSideAudit current) ||
            !current.HasProvenance ||
            current.Provenance.SurfaceNormal.sqrMagnitude < 0.0001f)
        {
            return DiagnosticLayerRole.None;
        }

        Vector3 normal = current.Provenance.SurfaceNormal.normalized;
        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        int maxSteps = Mathf.Clamp(Mathf.CeilToInt(duplicateLayerMaxGapMeters / voxel) + 1, 2, 12);
        DoubleLayerSideAudit partner = default;
        bool foundPartner = false;
        float bestSeparation = float.PositiveInfinity;
        for (int sign = -1; sign <= 1; sign += 2)
        {
            for (int step = 1; step <= maxSteps; step++)
            {
                Vector3 probe = center + normal * (sign * step * voxel);
                if (!TryResolveDoubleLayerSide(probe, out DoubleLayerSideAudit candidate) ||
                    candidate.Index == current.Index ||
                    !candidate.HasProvenance ||
                    candidate.Provenance.SurfaceNormal.sqrMagnitude < 0.0001f)
                {
                    continue;
                }

                Vector3 candidateNormal = candidate.Provenance.SurfaceNormal.normalized;
                if (Mathf.Abs(Vector3.Dot(normal, candidateNormal)) < duplicateLayerMinNormalDot)
                    continue;
                Vector3 delta = candidate.Provenance.SurfacePoint - current.Provenance.SurfacePoint;
                float separation = Mathf.Abs(Vector3.Dot(delta, normal));
                if (separation < atomicAcceptDuplicateMinGapMeters || separation > duplicateLayerMaxGapMeters)
                    continue;
                float lateralSq = Mathf.Max(0f, delta.sqrMagnitude - separation * separation);
                if (lateralSq > duplicateLayerMaxGapMeters * duplicateLayerMaxGapMeters * 2.25f)
                    continue;
                if (separation >= bestSeparation)
                    continue;

                bestSeparation = separation;
                partner = candidate;
                foundPartner = true;
            }
        }

        if (!foundPartner)
            return DiagnosticLayerRole.None;

        float currentScore = DiagnosticLayerStrength(current);
        float partnerScore = DiagnosticLayerStrength(partner);
        if (Mathf.Abs(currentScore - partnerScore) < 0.001f)
        {
            currentScore += current.Provenance.WriteSequence >= partner.Provenance.WriteSequence ? 0.01f : 0f;
            partnerScore += partner.Provenance.WriteSequence > current.Provenance.WriteSequence ? 0.01f : 0f;
        }
        return currentScore >= partnerScore ? DiagnosticLayerRole.Primary : DiagnosticLayerRole.Secondary;
    }

    private float DiagnosticLayerStrength(DoubleLayerSideAudit side)
    {
        VoxelWriteProvenance provenance = side.Provenance;
        float score = Mathf.Max(0, side.Weight) * 4f;
        score += Mathf.Min(32, provenance.WriteCount) * 0.5f;
        score += Mathf.Min(16, provenance.IntegrateCount) * 2f;
        score += DiagnosticLayerOperationStrength(provenance.LastOperation);
        score -= Mathf.Min(16, provenance.RepairCount) * 1.5f;
        if (side.Pending)
            score -= 16f;
        if (side.Dirty)
            score -= 32f;
        return score;
    }

    private static float DiagnosticLayerOperationStrength(string operation)
    {
        switch (operation)
        {
            case "atomic_accept_band":
            case "strong_current_promote":
            case "integrate":
                return 24f;
            case "guarded_replace":
            case "replace":
            case "conflict_correct":
                return 12f;
            case "continuity_fill":
            case "boundary_no_tsdf_fill":
                return 6f;
            case "hole_side_repair":
                return -8f;
            case "duplicate_layer_decay":
            case "duplicate_layer_clear":
            case "free_space_carve":
                return -12f;
            default:
                return 0f;
        }
    }

    private void AddHoleBoundaryPoint(List<Vector3> vertices, List<int> triangles, List<Color> colors, Vector3 center, Vector3 cameraPosition, Color color, float sizeScale = 1f)
    {
        float half = Mathf.Max(0.0025f, holeBoundaryMarkerSizeMeters * Mathf.Max(0.1f, sizeScale) * 0.5f);
        Vector3 view = center - cameraPosition;
        if (!Finite(view) || view.sqrMagnitude < 0.0001f)
            view = Vector3.forward;
        view.Normalize();
        Vector3 tangent = Vector3.Cross(Vector3.up, view);
        if (tangent.sqrMagnitude < 0.0001f)
            tangent = Vector3.Cross(Vector3.right, view);
        tangent.Normalize();
        Vector3 bitangent = Vector3.Cross(view, tangent).normalized;
        int start = vertices.Count;
        vertices.Add(center - tangent * half - bitangent * half);
        vertices.Add(center + tangent * half - bitangent * half);
        vertices.Add(center + tangent * half + bitangent * half);
        vertices.Add(center - tangent * half + bitangent * half);
        colors.Add(color);
        colors.Add(color);
        colors.Add(color);
        colors.Add(color);
        triangles.Add(start);
        triangles.Add(start + 1);
        triangles.Add(start + 2);
        triangles.Add(start);
        triangles.Add(start + 2);
        triangles.Add(start + 3);
    }

    private void RebuildRawDepthDebugMesh()
    {
        const int classifierRevision = 6;
        bool coverageOverlay = enableRawCoverageGridDiagnostics && showRawCoverageGridOverlay;
        if (coverageOverlay)
        {
            BuildRawCoverageGridDebugCache();
        }
        else if (_rawDepthDebugClassifierRevision != classifierRevision)
        {
            ClearRawDepthDebugCache();
            _rawDepthDebugClassifierRevision = classifierRevision;
        }

        if (!showRawDepthDebugView && !coverageOverlay)
        {
            if (_rawDepthDebugObject != null)
                _rawDepthDebugObject.SetActive(false);
            ResetRawDepthDebugCounters();
            return;
        }

        EnsureRawDepthDebugObjects();
        if (_rawDepthDebugMesh == null)
            return;

        _rawDepthDebugMesh.Clear();
        ResetRawDepthDebugCounters();
        if (_rawDepthDebugPoints.Count <= 0 || _rawDepthDebugKinds.Count <= 0)
        {
            _rawDepthDebugObject.SetActive(false);
            return;
        }

        int count = Mathf.Min(_rawDepthDebugPoints.Count, _rawDepthDebugKinds.Count, Mathf.Max(1, maxRawDepthDebugSamples));
        bool worldCubes = false;
        List<Vector3> vertices = new List<Vector3>(count * (worldCubes ? 8 : 4));
        List<int> triangles = new List<int>(count * (worldCubes ? 36 : 6));
        List<Color> colors = new List<Color>(count * (worldCubes ? 8 : 4));
        Vector3 cameraPosition = GetCameraPosition();
        for (int i = 0; i < count; i++)
        {
            if (worldCubes)
                AddRawDepthDebugCubeSample(vertices, triangles, colors, _rawDepthDebugPoints[i], _rawDepthDebugKinds[i]);
            else
                AddRawDepthDebugSample(vertices, triangles, colors, _rawDepthDebugPoints[i], cameraPosition, _rawDepthDebugKinds[i]);
        }

        if (vertices.Count <= 0 || triangles.Count <= 0)
        {
            _rawDepthDebugObject.SetActive(false);
            return;
        }

        _rawDepthDebugMesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _rawDepthDebugMesh.SetVertices(vertices);
        _rawDepthDebugMesh.SetColors(colors);
        _rawDepthDebugMesh.SetTriangles(triangles, 0, true);
        _rawDepthDebugMesh.RecalculateBounds();
        _rawDepthDebugObject.SetActive(true);
    }

    private bool TryGetUsableSample(
        int index,
        Vector3[] positions,
        Vector3[] normals,
        Color[] meta,
        Vector3 cameraPosition,
        out Vector3 point,
        out Vector3 normal,
        out float sampleWeight)
    {
        point = Vector3.zero;
        normal = Vector3.up;
        sampleWeight = 1f;

        if (index < 0 || index >= positions.Length)
            return false;

        point = positions[index];
        if (!Finite(point))
        {
            LastRejectedInvalidPositionCount++;
            return false;
        }

        Vector3 toPoint = point - cameraPosition;
        float depth = toPoint.magnitude;
        if (depth < minDepthMeters || depth > maxDepthMeters)
        {
            LastRejectedDepthRangeCount++;
            return false;
        }

        bool hasMeta = meta != null && index < meta.Length;
        Vector3 viewDirection = toPoint.sqrMagnitude > 0.0001f ? toPoint.normalized : Vector3.forward;
        bool normalValid = false;
        if (normals != null && index < normals.Length)
        {
            if (!hasMeta || meta[index].a >= 0.5f)
            {
                normal = normals[index];
                normalValid = Finite(normal) && normal.sqrMagnitude > 0.01f;
            }
        }

        if (!normalValid && !requireValidNormal)
        {
            normal = -viewDirection;
            normalValid = true;
        }

        if (!normalValid)
        {
            LastRejectedNormalCount++;
            return false;
        }

        normal.Normalize();
        if (Vector3.Dot(normal, viewDirection) > 0f)
            normal = -normal;
        if (Vector3.Dot(normal, -viewDirection) < minNormalFacingCameraDot)
        {
            LastRejectedFacingCount++;
            return false;
        }

        float confidence = hasMeta ? meta[index].g : 1f;
        if (confidence < minConfidence)
        {
            LastRejectedConfidenceCount++;
            return false;
        }

        sampleWeight = Mathf.Clamp01(confidence);
        return true;
    }

    private bool TryGetUsableSampleOrNeighbor(
        int x,
        int y,
        int width,
        int height,
        Vector3[] positions,
        Vector3[] normals,
        Color[] meta,
        Vector3 cameraPosition,
        out Vector3 point,
        out Vector3 normal,
        out float sampleWeight)
    {
        int index = y * width + x;
        if (TryGetUsableSample(index, positions, normals, meta, cameraPosition, out point, out normal, out sampleWeight))
            return true;

        if (!fillProjectiveDepthFromNeighbors)
            return false;

        int radius = Mathf.Clamp(projectiveNeighborFillRadiusPixels, 1, 4);
        for (int r = 1; r <= radius; r++)
        {
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    if (Mathf.Abs(dx) != r && Mathf.Abs(dy) != r)
                        continue;

                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                        continue;

                    int neighborIndex = ny * width + nx;
                    if (!TryGetUsableSampleNoRejectCount(neighborIndex, positions, normals, meta, cameraPosition, out point, out normal, out sampleWeight))
                        continue;

                    float distanceScale = 1f - Mathf.Clamp01((r - 1f) / Mathf.Max(1f, radius));
                    sampleWeight *= Mathf.Lerp(0.45f, 0.75f, distanceScale);
                    LastNeighborFilledSampleCount++;
                    return true;
                }
            }
        }

        return false;
    }

    private bool TryGetUsableSampleNoRejectCount(
        int index,
        Vector3[] positions,
        Vector3[] normals,
        Color[] meta,
        Vector3 cameraPosition,
        out Vector3 point,
        out Vector3 normal,
        out float sampleWeight)
    {
        int invalid = LastRejectedInvalidPositionCount;
        int depth = LastRejectedDepthRangeCount;
        int normalReject = LastRejectedNormalCount;
        int facing = LastRejectedFacingCount;
        int confidence = LastRejectedConfidenceCount;
        bool ok = TryGetUsableSample(index, positions, normals, meta, cameraPosition, out point, out normal, out sampleWeight);
        LastRejectedInvalidPositionCount = invalid;
        LastRejectedDepthRangeCount = depth;
        LastRejectedNormalCount = normalReject;
        LastRejectedFacingCount = facing;
        LastRejectedConfidenceCount = confidence;
        return ok;
    }

    private bool PassDepthNeighborhoodConsistency(
        int x,
        int y,
        int width,
        int height,
        Vector3[] positions,
        Vector3 cameraPosition,
        Vector3 point,
        out int checkedNeighbors,
        out int consistentNeighbors)
    {
        checkedNeighbors = 0;
        consistentNeighbors = 0;

        if (!rejectDepthDiscontinuities || positions == null)
            return true;

        float depth = (point - cameraPosition).magnitude;
        if (!Finite(point) || depth < minDepthMeters || depth > maxDepthMeters)
            return false;

        int radius = Mathf.Clamp(depthDiscontinuityNeighborRadiusPixels, 1, 3);
        float maxDelta = Mathf.Max(maxDepthDiscontinuityMeters, depth * maxDepthDiscontinuityRatio);

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;

                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                    continue;

                int neighborIndex = ny * width + nx;
                if (neighborIndex < 0 || neighborIndex >= positions.Length)
                    continue;

                Vector3 neighbor = positions[neighborIndex];
                if (!Finite(neighbor))
                    continue;

                float neighborDepth = (neighbor - cameraPosition).magnitude;
                if (neighborDepth < minDepthMeters || neighborDepth > maxDepthMeters)
                    continue;

                checkedNeighbors++;
                if (Mathf.Abs(neighborDepth - depth) <= maxDelta)
                    consistentNeighbors++;
            }
        }

        int minNeighbors = Mathf.Max(1, minDepthConsistentNeighbors);
        if (checkedNeighbors < minNeighbors)
            return allowSparseDepthNeighborhood;
        return consistentNeighbors >= minNeighbors;
    }

    private bool PassTsdfDepthSupport(int checkedNeighbors, int consistentNeighbors)
    {
        if (!gateTsdfWritesByDepthSupport || !rejectDepthDiscontinuities)
            return true;
        int minChecked = Mathf.Max(1, minTsdfDepthCheckedNeighbors);
        if (checkedNeighbors < minChecked)
            return allowSparseDepthNeighborhood;
        float ratio = checkedNeighbors > 0 ? (float)consistentNeighbors / checkedNeighbors : 0f;
        return ratio >= Mathf.Clamp01(minTsdfDepthConsistencyRatio);
    }

    private bool ApplyRobustDepthPrefilter(
        int x,
        int y,
        int width,
        int height,
        Vector3[] positions,
        Vector3 cameraPosition,
        ref Vector3 point,
        ref float sampleWeight)
    {
        if (!useRobustDepthPrefilter || positions == null || !Finite(point))
            return true;

        Vector3 toPoint = point - cameraPosition;
        float depth = toPoint.magnitude;
        if (depth < minDepthMeters || depth > maxDepthMeters || depth <= 0.0001f)
            return false;

        Vector3 viewDirection = toPoint / depth;
        int radius = Mathf.Clamp(robustDepthFilterRadiusPixels, 1, 3);
        int sampleCount = 0;
        int checkedNeighbors = 0;
        int consistentNeighbors = 0;
        float maxDelta = Mathf.Max(maxDepthDiscontinuityMeters, depth * maxDepthDiscontinuityRatio);
        int edgeCheckedNeighbors = 0;
        int edgeNeighbors = 0;
        int edgeRadius = Mathf.Clamp(depthEdgeErosionRadiusPixels, 1, radius);
        float edgeDelta = Mathf.Max(robustDepthEdgeDeltaMeters, depth * robustDepthEdgeDeltaRatio);

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                    continue;

                int neighborIndex = ny * width + nx;
                if (neighborIndex < 0 || neighborIndex >= positions.Length)
                    continue;

                Vector3 neighbor = positions[neighborIndex];
                if (!Finite(neighbor))
                    continue;

                float neighborDepth = (neighbor - cameraPosition).magnitude;
                if (neighborDepth < minDepthMeters || neighborDepth > maxDepthMeters)
                    continue;

                if (sampleCount < _robustDepthSamples.Length)
                    _robustDepthSamples[sampleCount++] = neighborDepth;
                if (dx == 0 && dy == 0)
                    continue;

                checkedNeighbors++;
                float depthDelta = Mathf.Abs(neighborDepth - depth);
                if (depthDelta <= maxDelta)
                    consistentNeighbors++;

                if (erodeDepthEdgesBeforeTsdf && Mathf.Abs(dx) <= edgeRadius && Mathf.Abs(dy) <= edgeRadius)
                {
                    edgeCheckedNeighbors++;
                    if (depthDelta > edgeDelta)
                        edgeNeighbors++;
                }
            }
        }

        if (erodeDepthEdgesBeforeTsdf && edgeCheckedNeighbors > 0 && edgeNeighbors > maxRobustDepthEdgeNeighbors)
        {
            if (rejectStrongDepthEdges)
            {
                LastRejectedDepthEdgeErosionCount++;
                return false;
            }

            sampleWeight *= Mathf.Clamp01(depthEdgeWeightScale);
            LastDepthEdgeDownweightedCount++;
        }

        int minNeighbors = Mathf.Max(1, minRobustDepthNeighbors);
        if (checkedNeighbors >= minNeighbors)
        {
            float consistency = (float)consistentNeighbors / checkedNeighbors;
            if (consistency < Mathf.Clamp01(minRobustDepthConsistencyRatio))
                return false;

            float weightScale = Mathf.Lerp(
                Mathf.Clamp01(minRobustDepthWeightScale),
                1f,
                Mathf.Clamp01(consistency));
            if (weightScale < 0.98f)
                LastRobustDepthDownweightedCount++;
            sampleWeight *= weightScale;
        }

        if (sampleCount >= 3)
        {
            float medianDepth = MedianInPlace(_robustDepthSamples, sampleCount);
            float medianDelta = Mathf.Abs(depth - medianDepth);
            if (medianDelta > maxRobustMedianDepthDeviationMeters)
            {
                if (!correctDepthToNeighborhoodMedian)
                    return false;

                point = cameraPosition + viewDirection * medianDepth;
                float correctionScale = Mathf.Clamp01(1f - medianDelta / Mathf.Max(0.001f, truncationMeters * 2f));
                sampleWeight *= Mathf.Lerp(minRobustDepthWeightScale, 0.75f, correctionScale);
                LastRobustDepthCorrectedCount++;
            }
        }

        return sampleWeight >= minRobustTsdfSampleWeight;
    }

    private static float MedianInPlace(float[] values, int count)
    {
        for (int i = 1; i < count; i++)
        {
            float value = values[i];
            int j = i - 1;
            while (j >= 0 && values[j] > value)
            {
                values[j + 1] = values[j];
                j--;
            }
            values[j + 1] = value;
        }

        int mid = count / 2;
        if ((count & 1) == 1)
            return values[mid];
        return (values[mid - 1] + values[mid]) * 0.5f;
    }

    private bool PassFormalIntegrateGate(int index, float sampleTsdf, int oldWeight, out string reason)
    {
        reason = null;
        if (!gateFormalIntegrateWrites)
            return true;

        if (_auditVoteState != ObservationVoteState.Accept)
        {
            reason = "vote";
            return false;
        }

        if (_auditVoteScore < Mathf.Clamp01(minFormalIntegrateVoteScore))
        {
            reason = "score";
            return false;
        }

        if (_auditSampleSupportRatio < Mathf.Clamp01(minFormalIntegrateSupportRatio))
        {
            reason = "support";
            return false;
        }

        if (rejectFormalCleanHistoryMismatch &&
            oldWeight > 0 &&
            AuditVoteHasTag("clean") &&
            FormalCleanHasHistoryMismatch())
        {
            reason = "clean_history";
            return false;
        }

        if (rejectFormalIntegrateWeakHistory && AuditVoteHasExactTag("weak_history"))
        {
            reason = "weak_history";
            return false;
        }

        if (rejectFormalIntegrateTemporalDepthJump && AuditVoteHasExactTag("temporal_depth_jump"))
        {
            reason = "temporal_depth_jump";
            return false;
        }

        if (requireHistoryForFormalStrongCurrentClean &&
            AuditVoteHasExactTag("strong_current_clean") &&
            !StrongCurrentCleanHasFormalHistory(index, sampleTsdf, oldWeight))
        {
            reason = "strong_current_history";
            return false;
        }

        if (_auditBandConflictRatio >= 0f &&
            _auditBandConflictRatio > Mathf.Clamp01(maxFormalIntegrateBandConflictRatio))
        {
            reason = "band";
            return false;
        }

        if (_auditBandMeanResidual >= 0f &&
            _auditBandMeanResidual > Mathf.Clamp(maxFormalIntegrateBandMeanResidual, 0f, 2f))
        {
            reason = "residual";
            return false;
        }

        if (_auditSameFrameDepthDeltaMeters >= 0f &&
            _auditSameFrameDepthDeltaMeters >= Mathf.Max(0.001f, maxFormalIntegrateSameFrameDepthDeltaMeters))
        {
            reason = "same_frame";
            return false;
        }

        if (rejectCrossFrameCleanSurfaceConflict &&
            _auditCrossFrameCleanDepthDeltaMeters >= 0f &&
            _auditCrossFrameCleanDepthDeltaMeters >= Mathf.Max(0.001f, maxFormalIntegrateCrossFrameDepthDeltaMeters))
        {
            reason = "cross_frame";
            return false;
        }

        if (oldWeight > 0 &&
            _auditOldTsdfResidual >= 0f &&
            _auditOldTsdfResidual > Mathf.Clamp01(maxFormalIntegrateOldTsdfResidual))
        {
            reason = "old_tsdf";
            return false;
        }

        if (VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index))
        {
            reason = "dirty_pending";
            return false;
        }

        return true;
    }

    private bool AuditVoteHasExactTag(string tag)
    {
        if (string.IsNullOrEmpty(_auditVoteReasons))
            return false;

        string[] parts = _auditVoteReasons.Split('|');
        for (int i = 0; i < parts.Length; i++)
        {
            if (parts[i] == tag)
                return true;
        }

        return false;
    }

    private bool StrongCurrentCleanHasFormalHistory(int index, float sampleTsdf, int oldWeight)
    {
        int requiredOldWeight = Mathf.Clamp(minFormalStrongCurrentOldWeight, 0, 16);
        int requiredBandHistory = Mathf.Clamp(minFormalStrongCurrentBandHistory, 0, 16);
        float requiredAgreement = Mathf.Clamp01(minFormalStrongCurrentHistoryAgreement);

        bool hasFormalHistory =
            oldWeight >= requiredOldWeight &&
            (requiredBandHistory <= 0 || _auditBandHistoryCount >= requiredBandHistory) &&
            (requiredAgreement <= 0f || _auditHistoryAgreement < 0f || _auditHistoryAgreement >= requiredAgreement);
        if (hasFormalHistory)
            return true;

        if (!allowLocalSupportForFormalStrongCurrentHistory)
            return false;

        if (!HasFormalStrongCurrentLocalSupport(index, sampleTsdf, out int localSupport, out int localStable, out int localAxial, out int localResidualRejects))
        {
            LastFormalStrongCurrentLocalHistoryBlockLocalCount++;
            int minSupport = Mathf.Clamp(minFormalStrongCurrentLocalSupportVoxels, 1, 26);
            int minStable = Mathf.Clamp(minFormalStrongCurrentLocalStableVoxels, 0, 26);
            int minAxial = Mathf.Clamp(minFormalStrongCurrentLocalAxialVoxels, 0, 6);
            if (localSupport < minSupport)
                LastFormalStrongCurrentLocalHistoryBlockSupportCount++;
            if (localStable < minStable)
                LastFormalStrongCurrentLocalHistoryBlockStableCount++;
            if (localAxial < minAxial)
                LastFormalStrongCurrentLocalHistoryBlockAxialCount++;
            if (localSupport <= 0 && localResidualRejects > 0)
                LastFormalStrongCurrentLocalHistoryBlockResidualCount++;
            return false;
        }

        if (!HasProvisionalPlaneCompatibility(index, sampleTsdf))
        {
            LastFormalStrongCurrentLocalHistoryBlockPlaneCount++;
            return false;
        }

        if (WouldStrongCurrentPromotionCreateDoubleLayer(index, sampleTsdf))
        {
            LastFormalStrongCurrentLocalHistoryBlockDoubleLayerCount++;
            return false;
        }

        LastFormalStrongCurrentLocalHistoryBypassCount++;
        return true;
    }

    private bool HasFormalStrongCurrentLocalSupport(int index, float sampleTsdf, out int support, out int stable, out int axial, out int residualRejects)
    {
        support = 0;
        stable = 0;
        axial = 0;
        residualRejects = 0;
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return false;

        IndexToVoxel(index, out int x, out int y, out int z);
        int radius = Mathf.Clamp(provisionalLocalSupportRadiusVoxels, 1, 2);
        int minSupport = Mathf.Clamp(minFormalStrongCurrentLocalSupportVoxels, 1, 26);
        int minStable = Mathf.Clamp(minFormalStrongCurrentLocalStableVoxels, 0, 26);
        int minAxial = Mathf.Clamp(minFormalStrongCurrentLocalAxialVoxels, 0, 6);
        float maxAbs = Mathf.Clamp(maxProvisionalLocalSupportAbsTsdf, 0.01f, 0.5f);
        float maxResidual = Mathf.Clamp(maxProvisionalLocalSupportResidual, 0.01f, 1f);
        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                        continue;

                    int neighbor = Index(nx, ny, nz);
                    int weight = _weights[neighbor];
                    if (weight < minSurfaceCornerWeight ||
                        VoxelIsDirtyQuarantined(neighbor) ||
                        VoxelHasPendingTsdfCorrection(neighbor))
                    {
                        continue;
                    }

                    float value = _tsdf[neighbor];
                    if (float.IsNaN(value) || float.IsInfinity(value))
                        continue;
                    if (Mathf.Abs(value) > maxAbs || Mathf.Abs(value - sampleTsdf) > maxResidual)
                    {
                        residualRejects++;
                        continue;
                    }

                    support++;
                    if (weight >= stableTsdfBypassWeight)
                        stable++;
                    if (Mathf.Abs(dx) + Mathf.Abs(dy) + Mathf.Abs(dz) == 1)
                        axial++;
                }
            }
        }

        return support >= minSupport && stable >= minStable && axial >= minAxial;
    }

    private bool TryPromoteStrongCurrentProvisional(int index, float sampleTsdf, float sampleWeight, float oldTsdf, int oldWeight)
    {
        if (!promoteStrongCurrentProvisionalToFormal)
        {
            RecordStrongCurrentProvisionalPromotionBlock("disabled");
            return false;
        }
        if (!AuditVoteHasExactTag("strong_current_clean"))
        {
            RecordStrongCurrentProvisionalPromotionBlock("tag");
            return false;
        }
        if (_provisionalTsdf == null || _provisionalTsdfLastFrame == null || _provisionalTsdfHits == null)
        {
            RecordStrongCurrentProvisionalPromotionBlock("storage");
            return false;
        }
        if (index < 0 || index >= _tsdf.Length || index >= _weights.Length || index >= _provisionalTsdf.Length || index >= _provisionalTsdfHits.Length)
        {
            RecordStrongCurrentProvisionalPromotionBlock("invalid");
            return false;
        }
        if (_provisionalTsdf[index] == 0)
        {
            string noProvisionalReason = Mathf.Abs(sampleTsdf) <= Mathf.Clamp01(provisionalTsdfNearSurfaceAbs)
                ? "no_provisional_near_surface"
                : "no_provisional_far_band";
            RecordStrongCurrentProvisionalPromotionBlock(noProvisionalReason);
            return false;
        }

        int requiredHits = Mathf.Clamp(minStrongCurrentProvisionalHits, 1, 8);
        int hits = _provisionalTsdfHits[index];
        if (hits < requiredHits)
        {
            RecordStrongCurrentProvisionalPromotionBlock("hits");
            return false;
        }

        int requiredWeight = Mathf.Clamp(minStrongCurrentProvisionalWeight, 1, 8);
        if (oldWeight < requiredWeight)
        {
            RecordStrongCurrentProvisionalPromotionBlock("weight");
            return false;
        }

        int requiredBandHistory = Mathf.Clamp(minStrongCurrentProvisionalBandHistory, 0, 16);
        if (requiredBandHistory > 0 && _auditBandHistoryCount < requiredBandHistory)
        {
            RecordStrongCurrentProvisionalPromotionBlock("band_history");
            return false;
        }

        float requiredAgreement = Mathf.Clamp01(minStrongCurrentProvisionalHistoryAgreement);
        if (requiredAgreement > 0f && _auditHistoryAgreement >= 0f && _auditHistoryAgreement < requiredAgreement)
        {
            RecordStrongCurrentProvisionalPromotionBlock("agreement");
            return false;
        }

        if (FormalCleanHasHistoryMismatch())
        {
            RecordStrongCurrentProvisionalPromotionBlock("clean_history");
            return false;
        }

        if (VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index))
        {
            RecordStrongCurrentProvisionalPromotionBlock("dirty_pending");
            return false;
        }

        if (_auditSameFrameDepthDeltaMeters >= 0f &&
            _auditSameFrameDepthDeltaMeters >= Mathf.Max(0.001f, maxFormalIntegrateSameFrameDepthDeltaMeters))
        {
            RecordStrongCurrentProvisionalPromotionBlock("same_frame");
            return false;
        }

        if (rejectCrossFrameCleanSurfaceConflict &&
            _auditCrossFrameCleanDepthDeltaMeters >= 0f &&
            _auditCrossFrameCleanDepthDeltaMeters >= Mathf.Max(0.001f, maxFormalIntegrateCrossFrameDepthDeltaMeters))
        {
            RecordStrongCurrentProvisionalPromotionBlock("cross_frame");
            return false;
        }

        if (oldWeight > 0 &&
            Mathf.Sign(oldTsdf) != Mathf.Sign(sampleTsdf) &&
            Mathf.Abs(oldTsdf - sampleTsdf) >= tsdfConflictThreshold)
        {
            RecordStrongCurrentProvisionalPromotionBlock("conflict");
            return false;
        }

        if (!HasProvisionalPlaneCompatibility(index, sampleTsdf))
        {
            RecordStrongCurrentProvisionalPromotionBlock("plane");
            return false;
        }

        if (!HasStrongCurrentPromotionLocalSupport(index, sampleTsdf))
        {
            RecordStrongCurrentProvisionalPromotionBlock("local");
            return false;
        }

        if (WouldStrongCurrentPromotionCreateDoubleLayer(index, sampleTsdf))
        {
            RecordStrongCurrentProvisionalPromotionBlock("double_layer");
            return false;
        }

        float promoteWeight = Mathf.Clamp(strongCurrentProvisionalPromoteWeight, 0.05f, 1f);
        promoteWeight = Mathf.Max(promoteWeight, Mathf.Clamp01(sampleWeight));
        int cappedWeight = Mathf.Min(maxFusionWeight, oldWeight + Mathf.Max(1, Mathf.RoundToInt(promoteWeight)));
        float denominator = Mathf.Max(0.0001f, oldWeight + promoteWeight);
        _tsdf[index] = Mathf.Clamp((oldTsdf * oldWeight + sampleTsdf * promoteWeight) / denominator, -1f, 1f);
        _weights[index] = (byte)cappedWeight;
        RecordContributionLedger(index, "strong_current_promote", oldTsdf, oldWeight, sampleTsdf, promoteWeight, _tsdf[index], _weights[index]);
        LastStrongCurrentProvisionalPromotedCount++;
        LastFormalIntegrateWriteCount++;
        LastUpdatedVoxelCount++;
        return true;
    }

    private bool IsProvisionalSurfaceCandidate(float sampleTsdf)
    {
        return Mathf.Abs(sampleTsdf) <= Mathf.Clamp01(provisionalTsdfNearSurfaceAbs);
    }

    private bool IsProvisionalTracked(int index)
    {
        return _provisionalTsdf != null &&
               index >= 0 &&
               index < _provisionalTsdf.Length &&
               _provisionalTsdf[index] != 0;
    }

    private void RecordProvisionalFarBandSkip(float sampleTsdf)
    {
        LastProvisionalTsdfFarBandSkippedCount++;
        RecordStrongSampleSeedTemporaryBlock("near_surface");
    }

    private bool CanBootstrapNearSurfaceProvisional(float sampleTsdf)
    {
        if (!allowNearSurfaceProvisionalBootstrap)
            return false;
        if (Mathf.Abs(sampleTsdf) > Mathf.Clamp01(maxNearSurfaceProvisionalBootstrapAbsTsdf))
            return false;
        if (_auditVoteState != ObservationVoteState.Accept && !AuditVoteHasExactTag("strong_current_clean"))
            return false;
        if (_auditVoteScore < Mathf.Clamp01(minNearSurfaceProvisionalBootstrapVoteScore))
            return false;
        if (_auditSampleSupportRatio < Mathf.Clamp01(minNearSurfaceProvisionalBootstrapSupportRatio))
            return false;
        if (_auditBandConflictRatio >= 0f && _auditBandConflictRatio > Mathf.Clamp01(maxProvisionalBandConflictRatio))
            return false;
        return true;
    }

    private bool AuditVoteHasTag(string tag)
    {
        if (string.IsNullOrEmpty(_auditVoteReasons))
            return false;

        string[] parts = _auditVoteReasons.Split('|');
        for (int i = 0; i < parts.Length; i++)
        {
            string part = parts[i];
            if (part == tag || part == "strong_current_" + tag)
                return true;
        }

        return false;
    }

    private bool FormalCleanHasHistoryMismatch()
    {
        if (_auditOldTsdfResidual >= 0f &&
            _auditOldTsdfResidual > Mathf.Clamp01(maxFormalCleanHistoryOldTsdfResidual))
            return true;

        if (_auditBandMeanResidual >= 0f &&
            _auditBandMeanResidual > Mathf.Clamp(maxFormalCleanHistoryBandMeanResidual, 0f, 2f))
            return true;

        if (_auditBandConflictRatio >= 0f &&
            _auditBandConflictRatio > Mathf.Clamp01(maxFormalCleanHistoryBandConflictRatio))
            return true;

        return false;
    }

    private void RecordFormalIntegrateBlock(string reason)
    {
        LastFormalIntegrateBlockedCount++;
        switch (reason)
        {
            case "vote":
                LastFormalIntegrateBlockVoteCount++;
                break;
            case "score":
                LastFormalIntegrateBlockScoreCount++;
                break;
            case "support":
                LastFormalIntegrateBlockSupportCount++;
                break;
            case "band":
                LastFormalIntegrateBlockBandCount++;
                break;
            case "residual":
                LastFormalIntegrateBlockResidualCount++;
                break;
            case "clean_history":
                LastFormalIntegrateBlockCleanHistoryCount++;
                break;
            case "weak_history":
                LastFormalIntegrateBlockWeakHistoryCount++;
                break;
            case "temporal_depth_jump":
                LastFormalIntegrateBlockTemporalDepthJumpCount++;
                break;
            case "strong_current_history":
                LastFormalIntegrateBlockStrongCurrentHistoryCount++;
                break;
            case "same_frame":
                LastFormalIntegrateBlockSameFrameCount++;
                break;
            case "cross_frame":
                LastFormalIntegrateBlockCrossFrameCount++;
                break;
            case "old_tsdf":
                LastFormalIntegrateBlockOldTsdfCount++;
                break;
            case "dirty_pending":
                LastFormalIntegrateBlockDirtyPendingCount++;
                break;
        }
    }

    private void RecordStrongCurrentProvisionalPromotionBlock(string reason)
    {
        LastStrongCurrentProvisionalPromotionBlockedCount++;
        switch (reason)
        {
            case "disabled":
                LastStrongCurrentProvisionalBlockDisabledCount++;
                break;
            case "tag":
                LastStrongCurrentProvisionalBlockTagCount++;
                break;
            case "storage":
                LastStrongCurrentProvisionalBlockStorageCount++;
                break;
            case "invalid":
                LastStrongCurrentProvisionalBlockInvalidCount++;
                break;
            case "no_provisional":
                LastStrongCurrentProvisionalBlockNoProvisionalCount++;
                break;
            case "no_provisional_near_surface":
                LastStrongCurrentProvisionalBlockNoProvisionalCount++;
                LastStrongCurrentProvisionalBlockNoProvisionalNearSurfaceCount++;
                break;
            case "no_provisional_far_band":
                LastStrongCurrentProvisionalBlockNoProvisionalCount++;
                LastStrongCurrentProvisionalBlockNoProvisionalFarBandCount++;
                break;
            case "hits":
                LastStrongCurrentProvisionalBlockHitsCount++;
                break;
            case "weight":
                LastStrongCurrentProvisionalBlockWeightCount++;
                break;
            case "band_history":
                LastStrongCurrentProvisionalBlockBandHistoryCount++;
                break;
            case "agreement":
                LastStrongCurrentProvisionalBlockAgreementCount++;
                break;
            case "clean_history":
                LastStrongCurrentProvisionalBlockCleanHistoryCount++;
                break;
            case "dirty_pending":
                LastStrongCurrentProvisionalBlockDirtyPendingCount++;
                break;
            case "same_frame":
                LastStrongCurrentProvisionalBlockSameFrameCount++;
                break;
            case "cross_frame":
                LastStrongCurrentProvisionalBlockCrossFrameCount++;
                break;
            case "conflict":
                LastStrongCurrentProvisionalBlockConflictCount++;
                break;
            case "local":
                LastStrongCurrentProvisionalBlockLocalCount++;
                break;
            case "plane":
                LastStrongCurrentProvisionalBlockPlaneCount++;
                break;
            case "double_layer":
                LastStrongCurrentPromotionDoubleLayerBlockCount++;
                break;
        }
    }

    private void IntegrateVoxel(int x, int y, int z, float sampleTsdf, float sampleWeight)
    {
        int index = Index(x, y, z);
        if (useAtomicObservationTsdfBands && _activeAtomicBandWrite)
        {
            if (_activeAtomicBandVote == ObservationVoteState.Accept)
                IntegrateAtomicAcceptedBandVoxel(index, sampleTsdf, sampleWeight);
            else if (_activeAtomicBandVote == ObservationVoteState.Pending)
                IntegrateAtomicProvisionalBandVoxel(index, sampleTsdf, sampleWeight);
            return;
        }

        if (useStage03ACleanIsoSurface && useV08DirectBandWriteForDiagnosis)
        {
            IntegrateVoxelV08Diagnostic(index, sampleTsdf, sampleWeight);
            return;
        }

        int oldWeight = _weights[index];
        float oldTsdf = _tsdf[index];
        bool auditCenter = SelectConfidenceAuditCenter(index, sampleTsdf, oldTsdf, oldWeight);
        if (rejectConflictingTsdfWrites &&
            oldWeight >= minConflictVoxelWeight &&
            Mathf.Sign(oldTsdf) != Mathf.Sign(sampleTsdf) &&
            Mathf.Abs(oldTsdf - sampleTsdf) >= tsdfConflictThreshold)
        {
            if (auditCenter)
                RecordVoxelAuditConflict(index, oldWeight, oldTsdf, sampleTsdf);
            int replacedBefore = LastReplacedDirtyTsdfCount;
            int correctedBefore = LastCorrectedTsdfCount;
            if (TryCorrectConflictingTsdf(index, sampleTsdf, oldWeight, oldTsdf))
            {
                if (LastReplacedDirtyTsdfCount > replacedBefore)
                {
                    RecordVoxelAuditReplace(index, auditCenter);
                    RecordConfidenceAuditVoxelOutcome(index, auditCenter, "replaced");
                }
                else if (LastCorrectedTsdfCount > correctedBefore)
                    RecordConfidenceAuditVoxelOutcome(index, auditCenter, "corrected");
                else
                    RecordConfidenceAuditVoxelOutcome(index, auditCenter, "pending_correction");
                return;
            }

            RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.RejectedConflict, oldWeight, oldTsdf, sampleTsdf);
            LastDirtyTsdfRejectedConflictCount++;
            LastRejectedTsdfConflictCount++;
            RecordConfidenceAuditVoxelOutcome(index, auditCenter, "rejected_conflict");
            return;
        }

        if (_auditStrongSampleSeedWrite && lockStrongSampleSeedToTemporaryTsdf)
        {
            if (TryWriteProvisionalTsdfSupport(index, sampleTsdf, sampleWeight, oldTsdf, oldWeight))
            {
                RecordConfidenceAuditVoxelOutcome(index, auditCenter, "strong_sample_seed");
                return;
            }

            LastStrongSampleSeedTemporaryBlockedCount++;
            RecordConfidenceAuditVoxelOutcome(index, auditCenter, "strong_sample_seed_blocked");
            return;
        }

        if (!PassMultiFrameTsdfStability(index, sampleTsdf, oldWeight))
        {
            if (TryMetabolizeOldCleanTsdf(index, sampleTsdf, oldTsdf, oldWeight))
            {
                RecordConfidenceAuditVoxelOutcome(index, auditCenter, "old_clean_metabolism");
                return;
            }

            if (IsProvisionalSurfaceCandidate(sampleTsdf) &&
                TryWriteProvisionalTsdfSupport(index, sampleTsdf, sampleWeight, oldTsdf, oldWeight))
            {
                RecordConfidenceAuditVoxelOutcome(index, auditCenter, "provisional_support");
                return;
            }

            if (IsProvisionalSurfaceCandidate(sampleTsdf))
            {
                LastProvisionalTsdfSupportBlockedCount++;
                LastProvisionalTsdfPendingStabilityBlockedCount++;
            }
            else
            {
                RecordProvisionalFarBandSkip(sampleTsdf);
            }
            RecordConfidenceAuditVoxelOutcome(index, auditCenter, "pending_stability");
            return;
        }
        if (requireMultiFrameStableTsdf && oldWeight <= 0)
            sampleWeight = Mathf.Max(sampleWeight, minStableTsdfFrames * Mathf.Clamp01(sampleWeight));

        if (!PassFormalIntegrateGate(index, sampleTsdf, oldWeight, out string formalBlockReason))
        {
            RecordFormalIntegrateBlock(formalBlockReason);
            if (formalBlockReason == "strong_current_history" &&
                TryPromoteStrongCurrentProvisional(index, sampleTsdf, sampleWeight, oldTsdf, oldWeight))
            {
                RecordConfidenceAuditVoxelOutcome(index, auditCenter, "strong_current_promoted");
                return;
            }

            if (downgradeBlockedFormalIntegrateToProvisional &&
                IsProvisionalSurfaceCandidate(sampleTsdf) &&
                TryWriteProvisionalTsdfSupport(index, sampleTsdf, sampleWeight, oldTsdf, oldWeight))
            {
                LastFormalIntegrateProvisionalCount++;
                RecordConfidenceAuditVoxelOutcome(index, auditCenter, "formal_provisional");
                return;
            }

            if (downgradeBlockedFormalIntegrateToProvisional && IsProvisionalSurfaceCandidate(sampleTsdf))
            {
                LastProvisionalTsdfSupportBlockedCount++;
                LastProvisionalTsdfFormalDowngradeBlockedCount++;
            }
            else if (downgradeBlockedFormalIntegrateToProvisional)
            {
                RecordProvisionalFarBandSkip(sampleTsdf);
            }
            RecordConfidenceAuditVoxelOutcome(index, auditCenter, "formal_blocked");
            return;
        }

        int cappedWeight = Mathf.Min(maxFusionWeight, oldWeight + Mathf.Max(1, Mathf.RoundToInt(sampleWeight)));
        float weightedOld = oldTsdf * oldWeight;
        float weightedNew = sampleTsdf * sampleWeight;
        float denominator = Mathf.Max(0.0001f, oldWeight + sampleWeight);
        _tsdf[index] = Mathf.Clamp(weightedOld + weightedNew, -maxFusionWeight, maxFusionWeight) / Mathf.Min(denominator, maxFusionWeight);
        _weights[index] = (byte)cappedWeight;
        RecordContributionLedger(index, "integrate", oldTsdf, oldWeight, sampleTsdf, sampleWeight, _tsdf[index], _weights[index]);
        LastFormalIntegrateWriteCount++;
        LastUpdatedVoxelCount++;
        RecordConfidenceAuditVoxelOutcome(index, auditCenter, "written");
    }

    private void IntegrateAtomicAcceptedBandVoxel(int index, float sampleTsdf, float sampleWeight)
    {
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return;

        int oldWeight = _weights[index];
        float oldTsdf = _tsdf[index];
        float writeWeight = Mathf.Max(0.0001f, sampleWeight);
        int fusionOldWeight = oldWeight;
        if (ShouldRecoverAcceptedSign(index, sampleTsdf, oldTsdf))
        {
            LastAcceptedSignRecoveryCandidateCount++;
            if (HasAcceptedSignRecoveryNeighborSupport(index, sampleTsdf))
            {
                fusionOldWeight = Mathf.Min(oldWeight, Mathf.Max(1, acceptedSignRecoveryOldWeightCap));
                LastAcceptedSignRecoveryAppliedCount++;
            }
            else
            {
                LastAcceptedSignRecoveryNeighborBlockedCount++;
            }
        }
        float denominator = Mathf.Max(0.0001f, fusionOldWeight + writeWeight);
        _tsdf[index] = Mathf.Clamp((oldTsdf * fusionOldWeight + sampleTsdf * writeWeight) / denominator, -1f, 1f);
        _weights[index] = (byte)Mathf.Min(maxFusionWeight, fusionOldWeight + Mathf.Max(1, Mathf.RoundToInt(writeWeight)));
        RecordAtomicAcceptSignResult(index, sampleTsdf, _tsdf[index]);

        bool promoted = _atomicProvisionalBandWeight != null &&
                        index < _atomicProvisionalBandWeight.Length &&
                        _atomicProvisionalBandWeight[index] > 0;
        RecordContributionLedger(index, "atomic_accept_band", oldTsdf, oldWeight, sampleTsdf, writeWeight, _tsdf[index], _weights[index]);
        ConfirmNearbyPrimaryPlaneHoleBandsFromAccept(index, sampleTsdf);
        ClearAtomicProvisionalBandVoxel(index);
        if (_atomicProvisionalBandRetiredLastFrame != null && index < _atomicProvisionalBandRetiredLastFrame.Length)
            _atomicProvisionalBandRetiredLastFrame[index] = int.MinValue;
        if (promoted)
            LastAtomicPromotedProvisionalVoxelCount++;
        LastAtomicAcceptedBandVoxelWriteCount++;
        LastFormalIntegrateWriteCount++;
        LastUpdatedVoxelCount++;
    }

    private void ConfirmNearbyPrimaryPlaneHoleBandsFromAccept(int acceptedIndex, float acceptedTsdf)
    {
        if (!usePrimaryPlaneHoleRepair || !confirmPrimaryPlaneHoleBandsFromNearbyAccept ||
            !_activeTsdfSourceValid || _auditSampleNormal.sqrMagnitude < 0.0001f ||
            _provisionalTsdf == null || _weights == null ||
            Mathf.Abs(acceptedTsdf) > Mathf.Clamp(primaryPlaneHoleAcceptMaxTsdfDelta, 0.01f, 0.3f) + holeSideRepairAbsTsdf)
        {
            return;
        }

        Vector3 planeNormal = _auditSampleNormal.normalized;
        Vector3 planePoint = _activeTsdfSourceSurface;
        int acceptedSign = acceptedTsdf >= 0f ? 1 : -1;
        int tangentialRadiusVoxels = Mathf.Clamp(primaryPlaneHoleAcceptTangentialRadiusVoxels, 1, 4);
        int radius = Mathf.Max(Mathf.Clamp(primaryPlaneHoleAcceptRadiusVoxels, 1, 3), tangentialRadiusVoxels);
        float maxTangentialDistance = Mathf.Max(voxelSizeMeters, voxelSizeMeters * tangentialRadiusVoxels);
        float maxPlaneDistance = Mathf.Min(
            Mathf.Max(voxelSizeMeters * 0.5f, primaryPlaneHoleAcceptMaxPlaneDistanceMeters),
            voxelSizeMeters * Mathf.Clamp(primaryPlaneHoleAcceptNormalHalfThicknessVoxels, 0.5f, 1.5f));
        float maxTsdfDelta = Mathf.Clamp(primaryPlaneHoleAcceptMaxTsdfDelta, 0.01f, 0.3f);
        IndexToVoxel(acceptedIndex, out int cx, out int cy, out int cz);

        for (int z = Mathf.Max(0, cz - radius); z <= Mathf.Min(_dimZ - 1, cz + radius); z++)
        for (int y = Mathf.Max(0, cy - radius); y <= Mathf.Min(_dimY - 1, cy + radius); y++)
        for (int x = Mathf.Max(0, cx - radius); x <= Mathf.Min(_dimX - 1, cx + radius); x++)
        {
            int candidate = Index(x, y, z);
            if (candidate == acceptedIndex || !HasProvisionalTsdfMarker(candidate) ||
                !_voxelWriteProvenance.TryGetValue(candidate, out VoxelWriteProvenance provenance) ||
                provenance.LastOperation != "primary_plane_hole_band")
            {
                continue;
            }

            Vector3 candidateOffset = VoxelCenter(x, y, z) - planePoint;
            float signedPlaneDistance = Vector3.Dot(candidateOffset, planeNormal);
            Vector3 tangentialOffset = candidateOffset - planeNormal * signedPlaneDistance;
            if (tangentialOffset.sqrMagnitude > maxTangentialDistance * maxTangentialDistance)
                continue;

            byte evidence = _primaryPlaneHoleAcceptEvidence.TryGetValue(candidate, out byte priorEvidence)
                ? priorEvidence
                : (byte)0;
            evidence |= 1; // A real Accept reached the candidate's search neighborhood.
            float candidateTsdf = _tsdf[candidate];
            int candidateSign = candidateTsdf >= 0f ? 1 : -1;
            if (candidateSign != acceptedSign)
            {
                _primaryPlaneHoleAcceptEvidence[candidate] = evidence;
                continue;
            }
            evidence |= 2;

            float planeDistance = Mathf.Abs(signedPlaneDistance);
            if (planeDistance > maxPlaneDistance)
            {
                float distanceVoxels = planeDistance / Mathf.Max(0.001f, voxelSizeMeters);
                if (!_primaryPlaneHoleMinPlaneDistanceVoxels.TryGetValue(candidate, out float priorDistance) ||
                    distanceVoxels < priorDistance)
                {
                    _primaryPlaneHoleMinPlaneDistanceVoxels[candidate] = distanceVoxels;
                }
                _primaryPlaneHoleAcceptEvidence[candidate] = evidence;
                continue;
            }
            evidence |= 4;

            if (Mathf.Abs(candidateTsdf - acceptedTsdf) > maxTsdfDelta)
            {
                _primaryPlaneHoleAcceptEvidence[candidate] = evidence;
                continue;
            }
            evidence |= 8;
            _primaryPlaneHoleAcceptEvidence[candidate] = evidence;

            int oldWeight = _weights[candidate];
            _weights[candidate] = (byte)Mathf.Max(minSurfaceCornerWeight, oldWeight);
            RecordContributionLedger(
                candidate,
                "primary_plane_hole_accept_confirmed",
                candidateTsdf,
                oldWeight,
                candidateTsdf,
                1f,
                candidateTsdf,
                _weights[candidate]);
            _primaryPlaneHoleConfirmations.Remove(candidate);
            _primaryPlaneHoleLastValidationFrame.Remove(candidate);
            _primaryPlaneHoleAcceptEvidence.Remove(candidate);
            _primaryPlaneHoleMinPlaneDistanceVoxels.Remove(candidate);
            _pendingPrimaryPlaneHoleAcceptConfirmedCount++;
        }
    }

    private bool ShouldRecoverAcceptedSign(int index, float sampleTsdf, float oldTsdf)
    {
        if (!enableAcceptedSignRecovery || Mathf.Abs(sampleTsdf) > acceptedSignRecoveryMaxAbsTsdf)
            return false;
        if (Mathf.Approximately(sampleTsdf, 0f) || Mathf.Sign(sampleTsdf) == Mathf.Sign(oldTsdf))
        {
            ClearOppositeAcceptedSignRecoveryEvidence(index, sampleTsdf);
            return false;
        }

        byte[] frames = sampleTsdf > 0f ? _acceptedSignRecoveryPositiveFrames : _acceptedSignRecoveryNegativeFrames;
        int[] lastFrames = sampleTsdf > 0f ? _acceptedSignRecoveryPositiveLastFrame : _acceptedSignRecoveryNegativeLastFrame;
        if (frames == null || lastFrames == null || index < 0 || index >= frames.Length)
            return false;

        if (lastFrames[index] != IntegratedFrameCount)
        {
            frames[index] = (byte)Mathf.Min(byte.MaxValue, frames[index] + 1);
            lastFrames[index] = IntegratedFrameCount;
        }
        return frames[index] >= Mathf.Max(2, acceptedSignRecoveryMinFrames);
    }

    private void ClearOppositeAcceptedSignRecoveryEvidence(int index, float sampleTsdf)
    {
        byte[] frames = sampleTsdf >= 0f ? _acceptedSignRecoveryNegativeFrames : _acceptedSignRecoveryPositiveFrames;
        if (frames != null && index >= 0 && index < frames.Length)
            frames[index] = 0;
    }

    private bool HasAcceptedSignRecoveryNeighborSupport(int index, float sampleTsdf)
    {
        byte[] frames = sampleTsdf > 0f ? _acceptedSignRecoveryPositiveFrames : _acceptedSignRecoveryNegativeFrames;
        if (frames == null || index < 0 || index >= frames.Length)
            return false;

        int xy = _dimX * _dimY;
        int z = index / xy;
        int rem = index - z * xy;
        int y = rem / _dimX;
        int x = rem - y * _dimX;
        int requiredFrames = Mathf.Max(1, acceptedSignRecoveryMinFrames - 1);
        int support = 0;
        if (x > 0 && frames[index - 1] >= requiredFrames) support++;
        if (x + 1 < _dimX && frames[index + 1] >= requiredFrames) support++;
        if (y > 0 && frames[index - _dimX] >= requiredFrames) support++;
        if (y + 1 < _dimY && frames[index + _dimX] >= requiredFrames) support++;
        if (z > 0 && frames[index - xy] >= requiredFrames) support++;
        if (z + 1 < _dimZ && frames[index + xy] >= requiredFrames) support++;
        return support >= Mathf.Max(1, acceptedSignRecoveryMinNeighborSupport);
    }

    private void RecordAtomicAcceptSignResult(int index, float sampleTsdf, float fusedTsdf)
    {
        if (_surfaceBandAcceptSignLostFlags == null || index < 0 || index >= _surfaceBandAcceptSignLostFlags.Length)
            return;
        if (_acceptedWriteSequence == int.MaxValue)
            ResetAcceptedWriteSequenceHistory();
        int writeSequence = ++_acceptedWriteSequence;
        byte signFlag = sampleTsdf >= 0f ? (byte)1 : (byte)2;
        bool retained = sampleTsdf >= 0f ? fusedTsdf >= 0f : fusedTsdf <= 0f;
        int[] acceptedSequences = sampleTsdf >= 0f ? _acceptedPositiveLastSequence : _acceptedNegativeLastSequence;
        int[] retainedSequences = sampleTsdf >= 0f ? _acceptedPositiveRetainedLastSequence : _acceptedNegativeRetainedLastSequence;
        if (acceptedSequences != null && index < acceptedSequences.Length)
            acceptedSequences[index] = writeSequence;
        if (Mathf.Approximately(sampleTsdf, 0f))
        {
            signFlag = 3;
            retained = Mathf.Approximately(fusedTsdf, 0f);
        }
        if (retained)
        {
            _surfaceBandAcceptSignLostFlags[index] = (byte)(_surfaceBandAcceptSignLostFlags[index] & ~signFlag);
            if (retainedSequences != null && index < retainedSequences.Length)
                retainedSequences[index] = writeSequence;
        }
        else
            _surfaceBandAcceptSignLostFlags[index] |= signFlag;
    }

    private void ResetAcceptedWriteSequenceHistory()
    {
        _acceptedWriteSequence = 0;
        if (_acceptedPositiveLastSequence != null) System.Array.Fill(_acceptedPositiveLastSequence, int.MinValue);
        if (_acceptedNegativeLastSequence != null) System.Array.Fill(_acceptedNegativeLastSequence, int.MinValue);
        if (_acceptedPositiveRetainedLastSequence != null) System.Array.Fill(_acceptedPositiveRetainedLastSequence, int.MinValue);
        if (_acceptedNegativeRetainedLastSequence != null) System.Array.Fill(_acceptedNegativeRetainedLastSequence, int.MinValue);
    }

    private void IntegrateAtomicProvisionalBandVoxel(int index, float sampleTsdf, float sampleWeight)
    {
        if (_atomicProvisionalBandTsdf == null || _atomicProvisionalBandWeight == null ||
            index < 0 || index >= _atomicProvisionalBandTsdf.Length || index >= _atomicProvisionalBandWeight.Length)
            return;

        int oldWeight = _atomicProvisionalBandWeight[index];
        float oldTsdf = oldWeight > 0 ? _atomicProvisionalBandTsdf[index] : sampleTsdf;
        float writeWeight = Mathf.Max(0.0001f, sampleWeight);
        float denominator = Mathf.Max(0.0001f, oldWeight + writeWeight);
        _atomicProvisionalBandTsdf[index] = Mathf.Clamp((oldTsdf * oldWeight + sampleTsdf * writeWeight) / denominator, -1f, 1f);
        _atomicProvisionalBandWeight[index] = (byte)Mathf.Min(
            Mathf.Max(1, atomicProvisionalBandMaxWeight),
            oldWeight + Mathf.Max(1, Mathf.RoundToInt(writeWeight)));

        if (_provisionalTsdf != null && index < _provisionalTsdf.Length)
            _provisionalTsdf[index] = 1;
        if (_provisionalTsdfHits != null && index < _provisionalTsdfHits.Length)
            _provisionalTsdfHits[index] = (byte)Mathf.Min(byte.MaxValue, _provisionalTsdfHits[index] + 1);
        if (_provisionalTsdfLastFrame != null && index < _provisionalTsdfLastFrame.Length)
            _provisionalTsdfLastFrame[index] = LastRawFrameIndex;
        if (_atomicProvisionalBandRetiredLastFrame != null && index < _atomicProvisionalBandRetiredLastFrame.Length)
            _atomicProvisionalBandRetiredLastFrame[index] = int.MinValue;

        LastAtomicProvisionalBandVoxelWriteCount++;
        LastUpdatedVoxelCount++;
    }

    private void ClearAtomicProvisionalBandVoxel(int index)
    {
        if (_atomicProvisionalBandTsdf != null && index >= 0 && index < _atomicProvisionalBandTsdf.Length)
            _atomicProvisionalBandTsdf[index] = 1f;
        if (_atomicProvisionalBandWeight != null && index >= 0 && index < _atomicProvisionalBandWeight.Length)
            _atomicProvisionalBandWeight[index] = 0;
        if (_provisionalTsdf != null && index >= 0 && index < _provisionalTsdf.Length)
            _provisionalTsdf[index] = 0;
        if (_provisionalTsdfHits != null && index >= 0 && index < _provisionalTsdfHits.Length)
            _provisionalTsdfHits[index] = 0;
        if (_provisionalTsdfLastFrame != null && index >= 0 && index < _provisionalTsdfLastFrame.Length)
            _provisionalTsdfLastFrame[index] = int.MinValue;
    }

    private void IntegrateVoxelV08Diagnostic(int index, float sampleTsdf, float sampleWeight)
    {
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return;

        int oldWeight = _weights[index];
        float oldTsdf = _tsdf[index];
        int addedWeight = Mathf.Max(1, Mathf.RoundToInt(sampleWeight));
        int cappedWeight = Mathf.Min(maxFusionWeight, oldWeight + addedWeight);
        float denominator = Mathf.Max(0.0001f, oldWeight + sampleWeight);
        _tsdf[index] = Mathf.Clamp(
            (oldTsdf * oldWeight + sampleTsdf * sampleWeight) / Mathf.Min(denominator, maxFusionWeight),
            -1f,
            1f);
        _weights[index] = (byte)cappedWeight;

        if (_provisionalTsdf != null && index < _provisionalTsdf.Length)
            _provisionalTsdf[index] = 0;
        if (_provisionalTsdfHits != null && index < _provisionalTsdfHits.Length)
            _provisionalTsdfHits[index] = 0;
        if (_provisionalTsdfLastFrame != null && index < _provisionalTsdfLastFrame.Length)
            _provisionalTsdfLastFrame[index] = int.MinValue;
        if (_correctionTsdfHits != null && index < _correctionTsdfHits.Length)
            _correctionTsdfHits[index] = 0;
        if (_correctionTsdfLastFrame != null && index < _correctionTsdfLastFrame.Length)
            _correctionTsdfLastFrame[index] = int.MinValue;
        if (_dirtyTsdfLastFrame != null && index < _dirtyTsdfLastFrame.Length)
            _dirtyTsdfLastFrame[index] = int.MinValue;

        RecordContributionLedger(index, "v08_direct_diag", oldTsdf, oldWeight, sampleTsdf, sampleWeight, _tsdf[index], _weights[index]);
        LastUpdatedVoxelCount++;
    }

    private bool TryMetabolizeOldCleanTsdf(int index, float sampleTsdf, float oldTsdf, int oldWeight)
    {
        if (!enableOldCleanTsdfMetabolism)
            return false;
        if (_oldCleanConflictHits == null || _oldCleanConflictLastFrame == null)
            return false;
        if (index < 0 || index >= _tsdf.Length || index >= _weights.Length || index >= _oldCleanConflictHits.Length)
            return false;

        bool rawCrossFrameCleanConflict =
            rejectCrossFrameCleanSurfaceConflict &&
            _auditCrossFrameCleanDepthDeltaMeters >= Mathf.Max(0.01f, crossFrameCleanConflictDepthMeters);
        float voxelResidual = Mathf.Abs(oldTsdf - sampleTsdf);
        bool crossFrameCleanConflict =
            rawCrossFrameCleanConflict &&
            voxelResidual >= Mathf.Clamp01(minOldCleanMetabolismCrossFrameVoxelResidual);
        bool oldTsdfResidualConflict =
            oldWeight > 0 &&
            voxelResidual >= Mathf.Clamp01(minOldCleanMetabolismResidual);
        bool rawLockedBandConflict =
            _auditBandHighWeightConflictCount > 0 &&
            (_auditBandHighWeightConflictRatio < 0f ||
             _auditBandHighWeightConflictRatio >= Mathf.Clamp01(maxOldCleanMetabolismBandConflictRatio));
        bool lockedBandConflict =
            rawLockedBandConflict &&
            voxelResidual >= Mathf.Clamp01(minOldCleanMetabolismBandVoxelResidual);
        if (!crossFrameCleanConflict && !oldTsdfResidualConflict && rawLockedBandConflict && !lockedBandConflict)
        {
            LastOldCleanMetabolismSkippedWeakBandCount++;
            return false;
        }
        if (!crossFrameCleanConflict && !oldTsdfResidualConflict && rawCrossFrameCleanConflict)
        {
            LastOldCleanMetabolismSkippedWeakCrossFrameCount++;
            return false;
        }
        if (!crossFrameCleanConflict && !oldTsdfResidualConflict && !lockedBandConflict)
            return false;

        LastOldCleanMetabolismCandidateCount++;

        if (_auditSampleSupportRatio < Mathf.Clamp01(minOldCleanMetabolismSupportRatio))
        {
            LastOldCleanMetabolismBlockedCount++;
            LastOldCleanMetabolismBlockedSupportCount++;
            return false;
        }

        if (_auditSameFrameDepthDeltaMeters >= 0f &&
            _auditSameFrameDepthDeltaMeters > Mathf.Max(0.001f, maxOldCleanMetabolismSameFrameDeltaMeters))
        {
            LastOldCleanMetabolismBlockedCount++;
            LastOldCleanMetabolismBlockedSameFrameCount++;
            return false;
        }

        if (oldWeight <= 0 || oldWeight > Mathf.Clamp(maxOldCleanMetabolismWeight, 1, maxFusionWeight))
        {
            LastOldCleanMetabolismBlockedCount++;
            LastOldCleanMetabolismBlockedWeightCount++;
            return false;
        }

        if (voxelResidual < Mathf.Clamp01(minOldCleanMetabolismResidual))
        {
            LastOldCleanMetabolismBlockedCount++;
            LastOldCleanMetabolismBlockedResidualCount++;
            return false;
        }

        if (VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index))
        {
            LastOldCleanMetabolismBlockedCount++;
            LastOldCleanMetabolismBlockedDirtyPendingCount++;
            return false;
        }

        int frameIndex = LastRawFrameIndex;
        if (_oldCleanConflictLastFrame[index] != frameIndex)
        {
            _oldCleanConflictHits[index] = (byte)Mathf.Min(255, _oldCleanConflictHits[index] + 1);
            _oldCleanConflictLastFrame[index] = frameIndex;
        }

        LastOldCleanMetabolismWatchCount++;
        if (_oldCleanConflictHits[index] < Mathf.Max(1, minOldCleanMetabolismConflictHits))
        {
            LastOldCleanMetabolismWaitingHitsCount++;
            return false;
        }

        int currentWeight = _weights[index];
        float currentTsdf = _tsdf[index];
        int newWeight = Mathf.Max(0, currentWeight - Mathf.Max(1, oldCleanMetabolismDecayWeight));
        if (newWeight <= Mathf.Max(0, oldCleanMetabolismClearWeight))
        {
            _tsdf[index] = 1f;
            _weights[index] = 0;
            _oldCleanConflictHits[index] = 0;
            _oldCleanConflictLastFrame[index] = int.MinValue;
            RecordContributionLedger(index, "old_clean_clear", currentTsdf, currentWeight, sampleTsdf, 0f, _tsdf[index], _weights[index]);
            LastOldCleanMetabolismClearCount++;
        }
        else
        {
            _weights[index] = (byte)newWeight;
            RecordContributionLedger(index, "old_clean_decay", currentTsdf, currentWeight, sampleTsdf, 0f, _tsdf[index], _weights[index]);
            LastOldCleanMetabolismDecayCount++;
        }

        LastUpdatedVoxelCount++;
        return true;
    }

    private int TryMetabolizeOldCleanTsdfBand(Vector3 cameraPosition, Vector3 surfacePoint, float sampleWeight, int halfBandSteps)
    {
        if (!enableOldCleanTsdfMetabolism)
            return 0;

        Vector3 toSurface = surfacePoint - cameraPosition;
        float surfaceDepth = toSurface.magnitude;
        if (!Finite(toSurface) || surfaceDepth <= 0.0001f)
            return 0;

        Vector3 rayDirection = toSurface / surfaceDepth;
        float voxel = Mathf.Max(0.0001f, voxelSizeMeters);
        float startDepth = Mathf.Max(0.01f, surfaceDepth - Mathf.Max(1, halfBandSteps) * voxel);
        float endDepth = surfaceDepth + Mathf.Max(1, halfBandSteps) * voxel;
        int changed = 0;
        for (float voxelDepth = startDepth; voxelDepth <= endDepth + voxel * 0.25f; voxelDepth += voxel)
        {
            Vector3 voxelWorld = cameraPosition + rayDirection * voxelDepth;
            if (!TryWorldToVoxel(voxelWorld, out int vx, out int vy, out int vz))
                continue;

            int index = Index(vx, vy, vz);
            float signedDistance = surfaceDepth - voxelDepth;
            float sampleTsdf = Mathf.Clamp(signedDistance / truncationMeters, -1f, 1f);
            if (TryMetabolizeOldCleanTsdf(index, sampleTsdf, _tsdf[index], _weights[index]))
                changed++;
        }

        return changed;
    }

    private bool TryWriteProvisionalTsdfSupport(int index, float sampleTsdf, float sampleWeight, float oldTsdf, int oldWeight)
    {
        if (!allowProvisionalTsdfSupportWrites)
        {
            RecordProvisionalTsdfBlock("disabled");
            RecordStrongSampleSeedTemporaryBlock("disabled");
            return false;
        }
        if (index < 0 || index >= _tsdf.Length || index >= _weights.Length)
        {
            RecordProvisionalTsdfBlock("invalid");
            RecordStrongSampleSeedTemporaryBlock("invalid");
            return false;
        }
        if (Mathf.Abs(sampleTsdf) > Mathf.Clamp01(provisionalTsdfNearSurfaceAbs))
        {
            RecordProvisionalTsdfBlock(sampleTsdf >= 0f ? "near_surface_positive" : "near_surface_negative");
            RecordStrongSampleSeedTemporaryBlock("near_surface");
            return false;
        }
        if (_auditVoteState == ObservationVoteState.Reject || _auditVoteEnforced)
        {
            RecordProvisionalTsdfBlock("vote");
            RecordStrongSampleSeedTemporaryBlock("vote");
            return false;
        }
        if (!_auditStrongSampleSeedWrite &&
            _auditVoteState != ObservationVoteState.Accept &&
            _auditVoteScore < Mathf.Clamp01(minProvisionalVoteScore))
        {
            RecordProvisionalTsdfBlock("score");
            RecordStrongSampleSeedTemporaryBlock("score");
            return false;
        }
        if (_auditSampleSupportRatio < Mathf.Clamp01(minProvisionalSupportRatio))
        {
            RecordProvisionalTsdfBlock("support");
            RecordStrongSampleSeedTemporaryBlock("support");
            return false;
        }
        if (_auditBandConflictRatio >= 0f && _auditBandConflictRatio > Mathf.Clamp01(maxProvisionalBandConflictRatio))
        {
            RecordProvisionalTsdfBlock("band");
            RecordStrongSampleSeedTemporaryBlock("band");
            return false;
        }
        if (rejectProvisionalCleanHistoryMismatch &&
            AuditVoteHasTag("clean") &&
            FormalCleanHasHistoryMismatch())
        {
            RecordProvisionalTsdfBlock("clean_history");
            RecordStrongSampleSeedTemporaryBlock("clean_history");
            return false;
        }
        if (_auditSameFrameDepthDeltaMeters >= 0f &&
            _auditSameFrameDepthDeltaMeters >= Mathf.Max(0.001f, maxProvisionalSameFrameDepthDeltaMeters))
        {
            RecordProvisionalTsdfBlock("same_frame");
            RecordStrongSampleSeedTemporaryBlock("same_frame");
            return false;
        }
        if (rejectCrossFrameCleanSurfaceConflict &&
            _auditCrossFrameCleanDepthDeltaMeters >= 0f &&
            _auditCrossFrameCleanDepthDeltaMeters >= Mathf.Max(0.001f, crossFrameCleanConflictDepthMeters))
        {
            RecordProvisionalTsdfBlock("cross_frame");
            RecordStrongSampleSeedTemporaryBlock("cross_frame");
            return false;
        }
        if (VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index))
        {
            RecordProvisionalTsdfBlock("dirty_pending");
            RecordStrongSampleSeedTemporaryBlock("dirty_pending");
            return false;
        }
        bool alreadyProvisional = IsProvisionalTracked(index);
        if (oldWeight > Mathf.Max(0, maxProvisionalExistingWeight) && !alreadyProvisional)
        {
            RecordProvisionalTsdfBlock("old_weight");
            RecordStrongSampleSeedTemporaryBlock("old_weight");
            return false;
        }
        if (oldWeight > Mathf.Max(0, maxProvisionalExistingWeight))
            LastProvisionalTsdfExistingWeightBypassCount++;
        if (oldWeight > 0 &&
            Mathf.Sign(oldTsdf) != Mathf.Sign(sampleTsdf) &&
            Mathf.Abs(oldTsdf - sampleTsdf) >= tsdfConflictThreshold)
        {
            RecordProvisionalTsdfBlock("conflict");
            RecordStrongSampleSeedTemporaryBlock("conflict");
            return false;
        }
        if (!alreadyProvisional && !HasProvisionalPlaneCompatibility(index, sampleTsdf))
        {
            RecordProvisionalTsdfBlock("plane");
            RecordStrongSampleSeedTemporaryBlock("plane");
            return false;
        }
        if (!_auditStrongSampleSeedWrite && !HasProvisionalLocalSupport(index, sampleTsdf))
        {
            if (!alreadyProvisional && !CanBootstrapNearSurfaceProvisional(sampleTsdf))
            {
                RecordProvisionalTsdfBlock("local");
                RecordStrongSampleSeedTemporaryBlock("local");
                return false;
            }
            LastProvisionalTsdfBootstrapLocalBypassCount++;
        }

        int maxTemporaryWeight = Mathf.Max(1, Mathf.Min(maxFusionWeight, stableTsdfBypassWeight - 1));
        int targetWeight = Mathf.Clamp(
            Mathf.Max(oldWeight, Mathf.Clamp(provisionalTsdfSupportWeight, 1, 2)),
            1,
            maxTemporaryWeight);
        float provisionalWeight = Mathf.Clamp(
            sampleWeight * Mathf.Clamp(provisionalTsdfSupportBlend, 0.05f, 0.6f),
            0.05f,
            targetWeight);

        float newTsdf;
        if (oldWeight <= 0)
        {
            newTsdf = sampleTsdf;
        }
        else
        {
            float denominator = Mathf.Max(0.0001f, oldWeight + provisionalWeight);
            newTsdf = (oldTsdf * oldWeight + sampleTsdf * provisionalWeight) / denominator;
        }

        _tsdf[index] = Mathf.Clamp(newTsdf, -1f, 1f);
        _weights[index] = (byte)targetWeight;
        RecordContributionLedger(index, _auditStrongSampleSeedWrite ? "strong_sample_seed" : "provisional_support", oldTsdf, oldWeight, sampleTsdf, provisionalWeight, _tsdf[index], _weights[index]);
        LastUpdatedVoxelCount++;
        LastProvisionalTsdfSupportWriteCount++;
        return true;
    }

    private void RecordStrongSampleSeedTemporaryBlock(string reason)
    {
        if (!_auditStrongSampleSeedWrite)
            return;

        switch (reason)
        {
            case "near_surface":
                LastStrongSampleSeedTempNearSurfaceBlockCount++;
                break;
            case "vote":
                LastStrongSampleSeedTempVoteBlockCount++;
                break;
            case "score":
                LastStrongSampleSeedTempScoreBlockCount++;
                break;
            case "support":
                LastStrongSampleSeedTempSupportBlockCount++;
                break;
            case "band":
                LastStrongSampleSeedTempBandBlockCount++;
                break;
            case "same_frame":
                LastStrongSampleSeedTempSameFrameBlockCount++;
                break;
            case "cross_frame":
                LastStrongSampleSeedTempCrossFrameBlockCount++;
                break;
            case "dirty_pending":
                LastStrongSampleSeedTempDirtyPendingBlockCount++;
                break;
            case "old_weight":
                LastStrongSampleSeedTempOldWeightBlockCount++;
                break;
            case "conflict":
                LastStrongSampleSeedTempConflictBlockCount++;
                break;
            case "local":
                LastStrongSampleSeedTempLocalBlockCount++;
                break;
        }
    }

    private bool HasProvisionalLocalSupport(int index, float sampleTsdf)
    {
        if (!requireProvisionalLocalSupport)
            return true;
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return false;

        IndexToVoxel(index, out int x, out int y, out int z);
        int radius = Mathf.Clamp(provisionalLocalSupportRadiusVoxels, 1, 2);
        int minSupport = Mathf.Clamp(minProvisionalLocalSupportVoxels, 1, 26);
        int minStable = Mathf.Clamp(minProvisionalLocalStableVoxels, 0, 26);
        int minAxial = Mathf.Clamp(minProvisionalLocalAxialVoxels, 0, 6);
        float maxAbs = Mathf.Clamp(maxProvisionalLocalSupportAbsTsdf, 0.01f, 0.5f);
        float maxResidual = Mathf.Clamp(maxProvisionalLocalSupportResidual, 0.01f, 1f);
        int support = 0;
        int stable = 0;
        int axial = 0;

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                        continue;

                    int neighbor = Index(nx, ny, nz);
                    int weight = _weights[neighbor];
                    if (weight < minSurfaceCornerWeight ||
                        VoxelIsDirtyQuarantined(neighbor) ||
                        VoxelHasPendingTsdfCorrection(neighbor))
                    {
                        continue;
                    }

                    float value = _tsdf[neighbor];
                    if (float.IsNaN(value) ||
                        float.IsInfinity(value) ||
                        Mathf.Abs(value) > maxAbs ||
                        Mathf.Abs(value - sampleTsdf) > maxResidual)
                    {
                        continue;
                    }

                    support++;
                    if (weight >= stableTsdfBypassWeight)
                        stable++;
                    if (Mathf.Abs(dx) + Mathf.Abs(dy) + Mathf.Abs(dz) == 1)
                        axial++;
                }
            }
        }

        LastProvisionalLocalSupportNeighborCount += support;
        LastProvisionalLocalSupportStableNeighborCount += stable;
        LastProvisionalLocalSupportAxialNeighborCount += axial;

        if (support < minSupport || stable < minStable || axial < minAxial)
            return false;

        LastProvisionalLocalSupportPassCount++;
        return true;
    }

    private bool HasStrongCurrentPromotionLocalSupport(int index, float sampleTsdf)
    {
        if (!allowProvisionalNeighborPromotionSupport)
            return HasProvisionalLocalSupport(index, sampleTsdf);
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return false;

        IndexToVoxel(index, out int x, out int y, out int z);
        int radius = Mathf.Clamp(provisionalLocalSupportRadiusVoxels, 1, 2);
        int minSupport = Mathf.Clamp(minStrongCurrentPromotionLocalSupportVoxels, 1, 26);
        int minStable = Mathf.Clamp(minStrongCurrentPromotionStableVoxels, 0, 26);
        int minAxial = Mathf.Clamp(minStrongCurrentPromotionAxialVoxels, 0, 6);
        float maxAbs = Mathf.Clamp(maxProvisionalLocalSupportAbsTsdf, 0.01f, 0.5f);
        float maxResidual = Mathf.Clamp(maxProvisionalLocalSupportResidual, 0.01f, 1f);
        int support = 0;
        int provisionalSupport = 0;
        int stable = 0;
        int axial = 0;

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                        continue;

                    int neighbor = Index(nx, ny, nz);
                    int weight = _weights[neighbor];
                    bool provisional = IsProvisionalTracked(neighbor);
                    if ((!provisional && weight < minSurfaceCornerWeight) ||
                        VoxelIsDirtyQuarantined(neighbor) ||
                        VoxelHasPendingTsdfCorrection(neighbor))
                    {
                        continue;
                    }

                    float value = _tsdf[neighbor];
                    if (float.IsNaN(value) ||
                        float.IsInfinity(value) ||
                        Mathf.Abs(value) > maxAbs ||
                        Mathf.Abs(value - sampleTsdf) > maxResidual)
                    {
                        continue;
                    }

                    support++;
                    if (provisional)
                        provisionalSupport++;
                    if (weight >= stableTsdfBypassWeight)
                        stable++;
                    if (Mathf.Abs(dx) + Mathf.Abs(dy) + Mathf.Abs(dz) == 1)
                        axial++;
                }
            }
        }

        LastStrongCurrentPromotionLocalNeighborCount += support;
        LastStrongCurrentPromotionLocalProvisionalNeighborCount += provisionalSupport;
        LastStrongCurrentPromotionLocalStableNeighborCount += stable;
        LastStrongCurrentPromotionLocalAxialNeighborCount += axial;

        if (support < minSupport || stable < minStable || axial < minAxial)
        {
            LastStrongCurrentPromotionLocalBlockCount++;
            return false;
        }

        LastStrongCurrentPromotionLocalPassCount++;
        return true;
    }

    private bool WouldStrongCurrentPromotionCreateDoubleLayer(int index, float sampleTsdf)
    {
        if (!rejectStrongCurrentPromotionDoubleLayer)
            return false;
        if (_tsdf == null || _weights == null || _voxelWriteProvenance == null ||
            index < 0 || index >= _tsdf.Length || index >= _weights.Length)
        {
            return false;
        }

        Vector3 sampleNormal = _auditSampleNormal;
        if (!Finite(sampleNormal) || sampleNormal.sqrMagnitude < 0.0001f)
            return false;
        sampleNormal.Normalize();

        Vector3 sampleSurface = _activeTsdfSourceValid ? _activeTsdfSourceSurface : VoxelCenterFromIndex(index);
        if (!Finite(sampleSurface))
            return false;

        IndexToVoxel(index, out int cx, out int cy, out int cz);
        int radius = Mathf.Clamp(strongCurrentPromotionDoubleLayerSearchRadiusVoxels, 1, 4);
        float minNormalDot = Mathf.Cos(Mathf.Clamp(doubleLayerMaxSourceNormalAngleDegrees, 1f, 60f) * Mathf.Deg2Rad);
        float minPlaneSeparation = Mathf.Max(0.01f, doubleLayerMinSourcePlaneSeparationMeters);
        float maxPlaneSeparation = Mathf.Max(
            minPlaneSeparation,
            Mathf.Max(
                maxStrongCurrentPromotionDoubleLayerSeparationMeters,
                Mathf.Max(voxelSizeMeters * pureGeometryMaxLayerSeparationVoxelScale, minPlaneSeparation * 1.75f)));
        float maxLateral = Mathf.Max(voxelSizeMeters * 0.25f, voxelSizeMeters * doubleLayerMaxSourceLateralVoxelScale);
        float maxAbs = Mathf.Clamp(maxStrongCurrentPromotionDoubleLayerNeighborAbsTsdf, 0.05f, 0.75f);

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    int nx = cx + dx;
                    int ny = cy + dy;
                    int nz = cz + dz;
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                        continue;

                    int neighbor = Index(nx, ny, nz);
                    int weight = _weights[neighbor];
                    bool dirtyOrPending = VoxelIsDirtyQuarantined(neighbor) || VoxelHasPendingTsdfCorrection(neighbor);
                    if (weight <= 0)
                    {
                        continue;
                    }

                    float neighborTsdf = _tsdf[neighbor];
                    if (float.IsNaN(neighborTsdf) ||
                        float.IsInfinity(neighborTsdf) ||
                        Mathf.Abs(neighborTsdf) > maxAbs)
                    {
                        continue;
                    }

                    if (!dirtyOrPending && weight < minSurfaceCornerWeight)
                        continue;

                    if (!_voxelWriteProvenance.TryGetValue(neighbor, out VoxelWriteProvenance provenance))
                        continue;
                    if (!Finite(provenance.SurfacePoint) ||
                        !Finite(provenance.SurfaceNormal) ||
                        provenance.SurfaceNormal.sqrMagnitude < 0.0001f)
                    {
                        continue;
                    }

                    LastStrongCurrentPromotionDoubleLayerCandidateCount++;
                    if (dirtyOrPending)
                        LastStrongCurrentPromotionDoubleLayerDirtyNeighborCount++;

                    Vector3 neighborNormal = provenance.SurfaceNormal.normalized;
                    float normalDot = Vector3.Dot(sampleNormal, neighborNormal);
                    if (Mathf.Abs(normalDot) < minNormalDot)
                    {
                        LastStrongCurrentPromotionDoubleLayerNormalRejectCount++;
                        continue;
                    }
                    if (normalDot < 0f)
                        neighborNormal = -neighborNormal;

                    Vector3 layerNormal = sampleNormal + neighborNormal;
                    if (layerNormal.sqrMagnitude < 0.0001f)
                    {
                        LastStrongCurrentPromotionDoubleLayerNormalRejectCount++;
                        continue;
                    }
                    layerNormal.Normalize();

                    Vector3 delta = sampleSurface - provenance.SurfacePoint;
                    float planeSeparation = Mathf.Abs(Vector3.Dot(delta, layerNormal));
                    if (planeSeparation < minPlaneSeparation || planeSeparation > maxPlaneSeparation)
                    {
                        LastStrongCurrentPromotionDoubleLayerPlaneRejectCount++;
                        continue;
                    }

                    float lateralSq = Mathf.Max(0f, delta.sqrMagnitude - planeSeparation * planeSeparation);
                    if (lateralSq > maxLateral * maxLateral)
                    {
                        LastStrongCurrentPromotionDoubleLayerLateralRejectCount++;
                        continue;
                    }

                    CountStrongCurrentPromotionDoubleLayerBlockSource(provenance, dirtyOrPending);
                    return true;
                }
            }
        }

        return false;
    }

    private bool WouldAtomicAcceptCreateDuplicateLayer(Vector3 sampleSurface, Vector3 sampleNormal)
    {
        if (!gateAtomicAcceptDuplicateLayers || _weights == null || _voxelWriteProvenance == null ||
            !Finite(sampleSurface) || !Finite(sampleNormal) || sampleNormal.sqrMagnitude < 0.0001f ||
            !TryWorldToVoxel(sampleSurface, out int cx, out int cy, out int cz))
        {
            return false;
        }

        sampleNormal.Normalize();
        int radius = Mathf.Clamp(atomicAcceptDuplicateSearchRadiusVoxels, 1, 5);
        float minGap = Mathf.Max(0.005f, atomicAcceptDuplicateMinGapMeters);
        float maxGap = Mathf.Max(minGap, duplicateLayerMaxGapMeters);
        float minNormalDot = Mathf.Clamp01(duplicateLayerMinNormalDot);
        float maxLateral = Mathf.Max(voxelSizeMeters * 1.5f, maxGap * 1.5f);
        bool hasSameSurface = false;
        int duplicateIndex = -1;
        float closestDuplicateGap = float.MaxValue;

        for (int dz = -radius; dz <= radius; dz++)
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int nx = cx + dx;
            int ny = cy + dy;
            int nz = cz + dz;
            if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                continue;

            int neighbor = Index(nx, ny, nz);
            if (_weights[neighbor] < minSurfaceCornerWeight ||
                !_voxelWriteProvenance.TryGetValue(neighbor, out VoxelWriteProvenance provenance) ||
                !Finite(provenance.SurfacePoint) || !Finite(provenance.SurfaceNormal) ||
                provenance.SurfaceNormal.sqrMagnitude < 0.0001f)
            {
                continue;
            }

            Vector3 priorNormal = provenance.SurfaceNormal.normalized;
            float normalDot = Mathf.Abs(Vector3.Dot(sampleNormal, priorNormal));
            if (normalDot < minNormalDot)
                continue;

            LastAtomicAcceptDuplicateCandidateCount++;
            Vector3 alignedNormal = Vector3.Dot(sampleNormal, priorNormal) < 0f ? -priorNormal : priorNormal;
            Vector3 layerNormal = sampleNormal + alignedNormal;
            if (layerNormal.sqrMagnitude < 0.0001f)
                continue;
            layerNormal.Normalize();

            Vector3 delta = sampleSurface - provenance.SurfacePoint;
            float planeGap = Mathf.Abs(Vector3.Dot(delta, layerNormal));
            float lateralSq = Mathf.Max(0f, delta.sqrMagnitude - planeGap * planeGap);
            if (planeGap < minGap)
            {
                LastAtomicAcceptDuplicateSameSurfaceCount++;
                hasSameSurface = true;
                continue;
            }
            if (planeGap <= maxGap && lateralSq <= maxLateral * maxLateral && planeGap < closestDuplicateGap)
            {
                closestDuplicateGap = planeGap;
                duplicateIndex = neighbor;
            }
        }

        if (duplicateIndex < 0)
            return false;

        if (hasSameSurface)
        {
            _duplicateLayerCleanupEvidence.TryGetValue(duplicateIndex, out int evidence);
            _duplicateLayerCleanupEvidence[duplicateIndex] = Mathf.Min(255, evidence + 1);
            return false;
        }

        return true;
    }

    private void CountStrongCurrentPromotionDoubleLayerBlockSource(VoxelWriteProvenance provenance, bool dirtyOrPending)
    {
        if (dirtyOrPending)
        {
            LastStrongCurrentPromotionDoubleLayerReplaceBlockCount++;
            return;
        }

        switch (provenance.LastOperation)
        {
            case "integrate":
                LastStrongCurrentPromotionDoubleLayerIntegrateBlockCount++;
                break;
            case "replace":
            case "guarded_replace":
            case "band_repair":
            case "conflict_correct":
            case "cleanup_neighbor":
                LastStrongCurrentPromotionDoubleLayerReplaceBlockCount++;
                break;
            default:
                LastStrongCurrentPromotionDoubleLayerOtherBlockCount++;
                break;
        }
    }

    private void CleanupConfirmedDuplicateLayers()
    {
        if (_tsdf == null || _weights == null || _duplicateLayerCleanupEvidence.Count == 0)
            return;

        int budget = Mathf.Max(1, maxDuplicateLayerCleanupVoxelsPerRebuild);
        int baseWeightStep = Mathf.Max(1, duplicateLayerCleanupWeightStep);
        int boostThreshold = Mathf.Max(2, duplicateLayerRepeatBoostThreshold);
        int maxBoostedStep = Mathf.Max(baseWeightStep, duplicateLayerMaxBoostedWeightStep);
        List<KeyValuePair<int, int>> ranked = new List<KeyValuePair<int, int>>(_duplicateLayerCleanupEvidence);
        ranked.Sort((a, b) => b.Value.CompareTo(a.Value));
        List<int> processed = new List<int>(Mathf.Min(budget, ranked.Count));
        for (int candidate = 0; candidate < ranked.Count && processed.Count < budget; candidate++)
        {
            int index = ranked[candidate].Key;
            int evidence = Mathf.Max(1, ranked[candidate].Value);
            processed.Add(index);
            LastDuplicateLayerCleanupMaxEvidence = Mathf.Max(LastDuplicateLayerCleanupMaxEvidence, evidence);
            if (index < 0 || index >= _weights.Length || _weights[index] <= 0)
                continue;

            int weightStep = baseWeightStep;
            if (evidence >= boostThreshold)
            {
                weightStep = Mathf.Min(maxBoostedStep, baseWeightStep + evidence - boostThreshold + 1);
                LastDuplicateLayerCleanupBoostedCount++;
            }
            float oldTsdf = _tsdf[index];
            int oldWeight = _weights[index];
            int newWeight = Mathf.Max(0, oldWeight - weightStep);
            if (newWeight <= 0)
            {
                _tsdf[index] = 1f;
                _weights[index] = 0;
                RecordContributionLedger(index, "duplicate_layer_clear", oldTsdf, oldWeight, 1f, 0f, 1f, 0);
                LastDuplicateLayerCleanupClearedCount++;
            }
            else
            {
                _weights[index] = (byte)newWeight;
                RecordContributionLedger(index, "duplicate_layer_decay", oldTsdf, oldWeight, oldTsdf, 0f, oldTsdf, newWeight);
                LastDuplicateLayerCleanupDecayedCount++;
            }
        }

        for (int i = 0; i < processed.Count; i++)
            _duplicateLayerCleanupEvidence.Remove(processed[i]);
    }

    private bool HasProvisionalPlaneCompatibility(int index, float sampleTsdf)
    {
        if (!requireProvisionalPlaneCompatibility)
            return true;
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return false;

        Vector3 sampleNormal = _auditSampleNormal;
        if (!Finite(sampleNormal) || sampleNormal.sqrMagnitude < 0.0001f)
        {
            LastProvisionalPlaneCompatibilityNoReferenceCount++;
            return true;
        }

        sampleNormal.Normalize();
        Vector3 sampleSurface = _activeTsdfSourceValid ? _activeTsdfSourceSurface : VoxelCenterFromIndex(index);
        if (!Finite(sampleSurface))
        {
            LastProvisionalPlaneCompatibilityNoReferenceCount++;
            return true;
        }

        IndexToVoxel(index, out int x, out int y, out int z);
        int radius = Mathf.Clamp(provisionalPlaneCompatibilityRadiusVoxels, 1, 2);
        int minCompatible = Mathf.Clamp(minProvisionalPlaneCompatibleNeighbors, 1, 8);
        float minNormalDot = Mathf.Clamp(minProvisionalPlaneNormalDot, 0.5f, 1f);
        float maxPlaneDistance = Mathf.Max(0.005f, maxProvisionalPlaneDistanceVoxelScale * Mathf.Max(0.0001f, voxelSizeMeters));
        float maxAbs = Mathf.Clamp(maxProvisionalPlaneNeighborAbsTsdf, 0.01f, 0.5f);
        int candidates = 0;
        int compatible = 0;

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                        continue;

                    int neighbor = Index(nx, ny, nz);
                    bool trackedProvisional = IsProvisionalTracked(neighbor);
                    int weight = _weights[neighbor];
                    if (!trackedProvisional && weight < minSurfaceCornerWeight)
                        continue;
                    if (VoxelIsDirtyQuarantined(neighbor) || VoxelHasPendingTsdfCorrection(neighbor))
                        continue;

                    float neighborTsdf = _tsdf[neighbor];
                    if (float.IsNaN(neighborTsdf) ||
                        float.IsInfinity(neighborTsdf) ||
                        Mathf.Abs(neighborTsdf) > maxAbs ||
                        Mathf.Abs(neighborTsdf - sampleTsdf) > maxProvisionalLocalSupportResidual)
                    {
                        continue;
                    }

                    if (!_voxelWriteProvenance.TryGetValue(neighbor, out VoxelWriteProvenance provenance))
                        continue;
                    if (!Finite(provenance.SurfacePoint) ||
                        !Finite(provenance.SurfaceNormal) ||
                        provenance.SurfaceNormal.sqrMagnitude < 0.0001f)
                    {
                        continue;
                    }

                    candidates++;
                    Vector3 neighborNormal = provenance.SurfaceNormal.normalized;
                    float normalDot = Mathf.Abs(Vector3.Dot(sampleNormal, neighborNormal));
                    if (normalDot < minNormalDot)
                    {
                        LastProvisionalPlaneCompatibilityNormalRejectedCount++;
                        continue;
                    }

                    float planeDistance = Mathf.Abs(Vector3.Dot(sampleSurface - provenance.SurfacePoint, sampleNormal));
                    if (planeDistance > maxPlaneDistance)
                    {
                        LastProvisionalPlaneCompatibilityDistanceRejectedCount++;
                        continue;
                    }

                    compatible++;
                    if (compatible >= minCompatible)
                    {
                        LastProvisionalPlaneCompatibilityCandidateCount += candidates;
                        LastProvisionalPlaneCompatibilityPassCount++;
                        return true;
                    }
                }
            }
        }

        LastProvisionalPlaneCompatibilityCandidateCount += candidates;
        if (candidates <= 0)
        {
            LastProvisionalPlaneCompatibilityNoReferenceCount++;
            return true;
        }

        LastProvisionalPlaneCompatibilityBlockedCount++;
        return false;
    }

    private void RetireExpiredProvisionalTsdf()
    {
        if (useAtomicObservationTsdfBands)
        {
            RetireExpiredAtomicProvisionalBands();
            return;
        }

        if (!retireUnconfirmedProvisionalTsdf || _provisionalTsdf == null || _provisionalTsdfLastFrame == null || _tsdf == null || _weights == null)
            return;

        int maxAge = Mathf.Max(1, provisionalTsdfMaxAgeFrames);
        int maxWeight = Mathf.Clamp(provisionalTsdfSupportWeight, 1, 2);
        int count = Mathf.Min(_provisionalTsdf.Length, Mathf.Min(_tsdf.Length, _weights.Length));
        for (int index = 0; index < count; index++)
        {
            if (_provisionalTsdf[index] == 0)
                continue;

            bool dirtyOrPending = VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index);
            int age = LastRawFrameIndex - _provisionalTsdfLastFrame[index];
            if (!dirtyOrPending && age >= 0 && age < maxAge)
                continue;

            int oldWeight = _weights[index];
            if (!dirtyOrPending && oldWeight > maxWeight)
            {
                _provisionalTsdf[index] = 0;
                _provisionalTsdfLastFrame[index] = int.MinValue;
                LastProvisionalTsdfConfirmedCount++;
                LastProvisionalTsdfConfirmedByWeightCount++;
                continue;
            }

            float oldTsdf = _tsdf[index];
            _tsdf[index] = 1f;
            _weights[index] = 0;
            string operation = dirtyOrPending ? "provisional_dirty_clear" : "provisional_retire";
            if (dirtyOrPending)
                LastProvisionalTsdfDirtyClearedCount++;
            else
            {
                LastProvisionalTsdfRetiredCount++;
                LastProvisionalTsdfRetiredExpiredCount++;
            }

            if (_provisionalTsdfHits != null && index < _provisionalTsdfHits.Length)
                _provisionalTsdfHits[index] = 0;

            RecordContributionLedger(index, operation, oldTsdf, oldWeight, 1f, 0f, _tsdf[index], _weights[index]);
        }
    }

    private void RetireExpiredAtomicProvisionalBands()
    {
        if (!retireUnconfirmedProvisionalTsdf || _provisionalTsdf == null || _provisionalTsdfLastFrame == null ||
            _atomicProvisionalBandTsdf == null || _atomicProvisionalBandWeight == null)
            return;

        int maxAge = Mathf.Max(1, provisionalTsdfMaxAgeFrames);
        int count = Mathf.Min(_provisionalTsdf.Length, _atomicProvisionalBandWeight.Length);
        for (int index = 0; index < count; index++)
        {
            if (_provisionalTsdf[index] == 0 || _atomicProvisionalBandWeight[index] == 0)
                continue;

            bool dirtyOrPending = VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index);
            int age = LastRawFrameIndex - _provisionalTsdfLastFrame[index];
            if (!dirtyOrPending && age >= 0 && age < maxAge)
                continue;

            if (_atomicProvisionalBandRetiredLastFrame != null && index < _atomicProvisionalBandRetiredLastFrame.Length)
                _atomicProvisionalBandRetiredLastFrame[index] = LastRawFrameIndex;
            if (_atomicProvisionalBandRetiredSign != null && index < _atomicProvisionalBandRetiredSign.Length)
            {
                float retiredTsdf = _atomicProvisionalBandTsdf[index];
                _atomicProvisionalBandRetiredSign[index] = retiredTsdf >= 0f ? (byte)1 : (byte)2;
                if (Mathf.Approximately(retiredTsdf, 0f))
                    _atomicProvisionalBandRetiredSign[index] = 3;
            }
            ClearAtomicProvisionalBandVoxel(index);
            LastAtomicRetiredProvisionalVoxelCount++;
            if (dirtyOrPending)
                LastProvisionalTsdfDirtyClearedCount++;
            else
            {
                LastProvisionalTsdfRetiredCount++;
                LastProvisionalTsdfRetiredExpiredCount++;
            }
        }
    }

    private bool SelectConfidenceAuditCenter(int index, float sampleTsdf, float oldTsdf, int oldWeight)
    {
        if (!_auditSampleActive)
            return false;
        float absSample = Mathf.Abs(sampleTsdf);
        if (absSample > _auditCenterAbsSampleTsdf)
            return false;

        _auditCenterAbsSampleTsdf = absSample;
        _auditCenterSampleTsdf = sampleTsdf;
        _auditCenterOldTsdf = oldTsdf;
        _auditCenterOldWeight = oldWeight;
        _auditCenterStableHitsBefore = ReadAuditHit(_pendingTsdfHits, index);
        _auditCenterStableHitsAfter = _auditCenterStableHitsBefore;
        _auditCenterCorrectionHitsBefore = ReadAuditHit(_correctionTsdfHits, index);
        _auditCenterCorrectionHitsAfter = _auditCenterCorrectionHitsBefore;
        _auditCenterSignFlip = oldWeight > 0 && Mathf.Sign(oldTsdf) != Mathf.Sign(sampleTsdf);
        _auditCenterResidual = oldWeight > 0 ? Mathf.Abs(oldTsdf - sampleTsdf) : 0f;
        _auditCenterOutcome = "none";
        return true;
    }

    private void RecordConfidenceAuditVoxelOutcome(int index, bool auditCenter, string outcome)
    {
        if (!_auditSampleActive)
            return;

        switch (outcome)
        {
            case "written":
            case "provisional_support":
                _auditBandWritten++;
                break;
            case "pending_stability":
                _auditBandPendingStable++;
                break;
            case "pending_correction":
                _auditBandPendingCorrection++;
                break;
            case "replaced":
                _auditBandReplaced++;
                break;
            case "corrected":
                _auditBandCorrected++;
                break;
            case "rejected_conflict":
                _auditBandRejectedConflict++;
                break;
        }

        if (!auditCenter)
            return;
        _auditCenterStableHitsAfter = ReadAuditHit(_pendingTsdfHits, index);
        _auditCenterCorrectionHitsAfter = ReadAuditHit(_correctionTsdfHits, index);
        _auditCenterOutcome = outcome;
    }

    private static int ReadAuditHit(byte[] hits, int index)
    {
        return hits != null && index >= 0 && index < hits.Length ? hits[index] : 0;
    }

    private void RecordVoxelAuditConflict(int index, int oldWeight, float oldTsdf, float sampleTsdf)
    {
        if (!_voxelAuditLifecycle.TryGetValue(index, out VoxelAuditLifecycle life))
        {
            life.FirstConflictFrame = LastRawFrameIndex;
            life.LastReplaceFrame = int.MinValue;
        }
        if (life.LastReplaceFrame != int.MinValue && LastRawFrameIndex > life.LastReplaceFrame)
            life.ConflictAfterReplaceCount++;
        life.LastConflictFrame = LastRawFrameIndex;
        life.ConflictCount++;
        life.LastOldWeight = oldWeight;
        life.LastOldTsdf = oldTsdf;
        life.LastSampleTsdf = sampleTsdf;
        life.LastResidual = Mathf.Abs(oldTsdf - sampleTsdf);
        life.LastCause = AuditCenterConflictCause();
        life.LastCorrectionHits = _auditCenterCorrectionHitsAfter;
        life.LastBandConflictRatio = _auditBandConflictRatio;
        life.LastBandHighWeightConflictRatio = _auditBandHighWeightConflictRatio;
        life.LastHistoryAgreement = _auditHistoryAgreement;
        life.LastHistoricalSurfaceDistanceMeters = _auditHistoricalSurfaceDistanceMeters;
        if (_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance prior))
        {
            Vector3 currentCamera = _activeTsdfSourceValid ? _activeTsdfSourceCamera : GetCameraPosition();
            Vector3 currentSurface = _activeTsdfSourceValid ? _activeTsdfSourceSurface : VoxelCenterFromIndex(index);
            life.PriorWriteKnown = true;
            life.PriorWriteFrame = prior.Frame;
            life.PriorWriteFrameGap = LastRawFrameIndex - prior.Frame;
            life.PriorCameraTravelMeters = Vector3.Distance(prior.CameraPosition, currentCamera);
            life.PriorSurfaceShiftMeters = Vector3.Distance(prior.SurfacePoint, currentSurface);
            Quaternion currentRotation = Camera.main != null ? Camera.main.transform.rotation : Quaternion.identity;
            Vector3 currentRay = (currentSurface - currentCamera).normalized;
            life.PriorCameraRotationDeltaDegrees = Quaternion.Angle(prior.CameraRotation, currentRotation);
            life.PriorRayAngleDeltaDegrees = prior.RayDirection.sqrMagnitude > 0.0001f && currentRay.sqrMagnitude > 0.0001f
                ? Vector3.Angle(prior.RayDirection, currentRay)
                : 0f;
            life.PriorDepthDeltaMeters = Mathf.Abs(prior.SurfaceDepth - Vector3.Distance(currentCamera, currentSurface));
            life.PriorNormalAngleDeltaDegrees = prior.SurfaceNormal.sqrMagnitude > 0.0001f && _auditSampleNormal.sqrMagnitude > 0.0001f
                ? Vector3.Angle(prior.SurfaceNormal, _auditSampleNormal)
                : 0f;
        }
        life.LastLockSubtype = AuditOldWeightLockSubtype(life);
        _voxelAuditLifecycle[index] = life;
        _captureAuditVoxels.Add(index);
    }

    private string AuditOldWeightLockSubtype(VoxelAuditLifecycle life)
    {
        if (life.LastCause != "old_weight_locked")
            return "none";

        if (life.ConflictAfterReplaceCount > 0 || life.ReplaceCount > 0)
            return "replace_recurrent";

        if (life.LastOldWeight > Mathf.Max(1, maxCorrectableTsdfWeight))
            return "above_correctable_weight";

        if (life.LastBandConflictRatio >= 0.20f || life.LastBandHighWeightConflictRatio >= 0.10f)
            return "band_conflict";

        if (life.PriorWriteKnown && life.PriorSurfaceShiftMeters >= 0.08f)
        {
            if (life.PriorCameraTravelMeters < 0.08f && life.PriorCameraRotationDeltaDegrees < 3f)
                return "depth_unstable";
            if (life.PriorRayAngleDeltaDegrees >= 25f ||
                life.PriorCameraTravelMeters >= 0.15f ||
                life.PriorNormalAngleDeltaDegrees >= 35f)
                return "view_changed";
        }

        if (life.LastCorrectionHits > 0 && life.LastCorrectionHits < Mathf.Max(1, minDirtyTsdfReplaceFrames))
            return "under_supported_pending";

        return "stable_old_surface";
    }

    private void RecordVoxelAuditReplace(int index, bool auditCenter)
    {
        if (!auditCenter || !_voxelAuditLifecycle.TryGetValue(index, out VoxelAuditLifecycle life))
            return;
        life.ReplaceCount++;
        life.LastReplaceFrame = LastRawFrameIndex;
        _voxelAuditLifecycle[index] = life;
    }

    private Vector3 VoxelCenterFromIndex(int index)
    {
        IndexToVoxel(index, out int x, out int y, out int z);
        return VoxelCenter(x, y, z);
    }

    private bool TryCorrectConflictingTsdf(int index, float sampleTsdf, int oldWeight, float oldTsdf)
    {
        if (!correctStableConflictingTsdf || oldWeight > maxCorrectableTsdfWeight)
            return false;
        if (_correctionTsdf == null || _correctionTsdfHits == null || _correctionTsdfLastFrame == null || index < 0 || index >= _correctionTsdf.Length)
            return false;

        int frameIndex = LastRawFrameIndex;
        if (_correctionTsdfHits[index] <= 0 ||
            Mathf.Sign(_correctionTsdf[index]) != Mathf.Sign(sampleTsdf) ||
            Mathf.Abs(_correctionTsdf[index] - sampleTsdf) > conflictCorrectionAgreementThreshold)
        {
            _correctionTsdf[index] = sampleTsdf;
            _correctionTsdfHits[index] = 1;
            _correctionTsdfLastFrame[index] = frameIndex;
            RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.PendingConflict, oldWeight, oldTsdf, sampleTsdf);
            LastDirtyTsdfPendingNewCount++;
            LastPendingTsdfCorrectionCount++;
            TryRepairDirtyTsdfConflictBand();
            return true;
        }

        if (_correctionTsdfLastFrame[index] != frameIndex)
        {
            int nextHits = Mathf.Min(255, _correctionTsdfHits[index] + 1);
            _correctionTsdfHits[index] = (byte)nextHits;
            _correctionTsdfLastFrame[index] = frameIndex;
            _correctionTsdf[index] = Mathf.Lerp(_correctionTsdf[index], sampleTsdf, 0.35f);
        }

        if (_correctionTsdfHits[index] < minConflictCorrectionFrames)
        {
            if (TryReplaceDirtyTsdf(index, oldWeight))
                return true;

            RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.RepeatedPending, oldWeight, oldTsdf, sampleTsdf);
            LastDirtyTsdfPendingRepeatCount++;
            LastPendingTsdfCorrectionCount++;
            TryRepairDirtyTsdfConflictBand();
            return true;
        }

        if (TryReplaceDirtyTsdf(index, oldWeight))
            return true;

        float corrected = Mathf.Lerp(oldTsdf, _correctionTsdf[index], conflictCorrectionBlend);
        _tsdf[index] = Mathf.Clamp(corrected, -1f, 1f);
        _weights[index] = (byte)Mathf.Clamp(correctedTsdfWeight, minSurfaceCornerWeight, maxFusionWeight);
        RecordContributionLedger(index, "conflict_correct", oldTsdf, oldWeight, _correctionTsdf[index], conflictCorrectionBlend, _tsdf[index], _weights[index]);
        _correctionTsdfHits[index] = 0;
        _correctionTsdfLastFrame[index] = int.MinValue;
        RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.CorrectedOldDepth, oldWeight, oldTsdf, sampleTsdf);
        LastCorrectedTsdfCount++;
        LastUpdatedVoxelCount++;
        TryRepairDirtyTsdfConflictBand();
        return true;
    }

    private bool TryRepairDirtyTsdfConflictBand()
    {
        if (!repairDirtyTsdfConflictBands)
        {
            LastDirtyTsdfBandRepairBlockedDisabledCount++;
            return false;
        }
        if (!_auditSampleActive)
        {
            LastDirtyTsdfBandRepairBlockedNoSampleCount++;
            return false;
        }
        if (_auditBandHistoryCount <= 0)
        {
            LastDirtyTsdfBandRepairBlockedNoHistoryCount++;
            return false;
        }
        if (_auditBandConflictRatio < Mathf.Clamp01(minDirtyBandConflictRatio))
        {
            LastDirtyTsdfBandRepairBlockedLowConflictCount++;
            return false;
        }

        LastDirtyTsdfBandRepairTriggerCount++;
        int repaired = useProjectiveTsdfIntegration
            ? RepairProjectiveDirtyTsdfConflictBand()
            : RepairNormalDirtyTsdfConflictBand();
        if (repaired <= 0)
            return false;

        LastDirtyTsdfBandRepairCount += repaired;
        LastDirtyTsdfBandRepairSampleCount++;
        return true;
    }

    private int RepairProjectiveDirtyTsdfConflictBand()
    {
        Vector3 cameraPosition = _activeTsdfSourceValid ? _activeTsdfSourceCamera : GetCameraPosition();
        Vector3 surfacePoint = _activeTsdfSourceValid ? _activeTsdfSourceSurface : VoxelCenterFromIndex(0);
        Vector3 toSurface = surfacePoint - cameraPosition;
        float surfaceDepth = toSurface.magnitude;
        if (!Finite(toSurface) || surfaceDepth <= 0.0001f)
            return 0;

        Vector3 ray = toSurface / surfaceDepth;
        float voxel = Mathf.Max(0.0001f, voxelSizeMeters);
        int halfBandSteps = Mathf.Max(1, Mathf.CeilToInt(truncationMeters / voxel));
        float startDepth = Mathf.Max(0.01f, surfaceDepth - halfBandSteps * voxel);
        float endDepth = surfaceDepth + halfBandSteps * voxel;
        int repaired = 0;
        for (float voxelDepth = startDepth; voxelDepth <= endDepth + voxel * 0.25f; voxelDepth += voxel)
        {
            Vector3 world = cameraPosition + ray * voxelDepth;
            float sampleTsdf = Mathf.Clamp((surfaceDepth - voxelDepth) / truncationMeters, -1f, 1f);
            repaired += RepairDirtyTsdfConflictBandVoxel(world, sampleTsdf, repaired);
            if (repaired >= Mathf.Max(1, maxDirtyBandRepairsPerSample))
                break;
        }
        return repaired;
    }

    private int RepairNormalDirtyTsdfConflictBand()
    {
        Vector3 surfacePoint = _activeTsdfSourceValid ? _activeTsdfSourceSurface : Vector3.zero;
        Vector3 normal = _auditSampleNormal.sqrMagnitude > 0.0001f ? _auditSampleNormal.normalized : Vector3.up;
        int halfBandSteps = Mathf.Max(1, Mathf.CeilToInt(truncationMeters / Mathf.Max(0.0001f, voxelSizeMeters)));
        int repaired = 0;
        for (int step = -halfBandSteps; step <= halfBandSteps; step++)
        {
            float signedDistance = step * voxelSizeMeters;
            Vector3 world = surfacePoint + normal * signedDistance;
            float sampleTsdf = Mathf.Clamp(signedDistance / truncationMeters, -1f, 1f);
            repaired += RepairDirtyTsdfConflictBandVoxel(world, sampleTsdf, repaired);
            if (repaired >= Mathf.Max(1, maxDirtyBandRepairsPerSample))
                break;
        }
        return repaired;
    }

    private int RepairDirtyTsdfConflictBandVoxel(Vector3 world, float sampleTsdf, int repairedSoFar)
    {
        if (repairedSoFar >= Mathf.Max(1, maxDirtyBandRepairsPerSample))
        {
            LastDirtyTsdfBandRepairBlockedBudgetCount++;
            return 0;
        }
        if (!TryWorldToVoxel(world, out int x, out int y, out int z))
        {
            LastDirtyTsdfBandRepairBlockedOutsideCount++;
            return 0;
        }

        LastDirtyTsdfBandRepairProbeCount++;
        int index = Index(x, y, z);
        int oldWeight = _weights[index];
        if (oldWeight <= 0)
        {
            LastDirtyTsdfBandRepairBlockedEmptyCount++;
            return 0;
        }
        if (oldWeight > Mathf.Max(1, maxDirtyBandRepairWeight))
        {
            LastDirtyTsdfBandRepairBlockedWeightCount++;
            return 0;
        }

        float oldTsdf = _tsdf[index];
        float residual = Mathf.Abs(oldTsdf - sampleTsdf);
        if (Mathf.Sign(oldTsdf) == Mathf.Sign(sampleTsdf))
        {
            LastDirtyTsdfBandRepairBlockedSameSignCount++;
            return 0;
        }
        if (residual < Mathf.Clamp(minDirtyBandRepairResidual, 0.05f, 1f))
        {
            LastDirtyTsdfBandRepairBlockedResidualCount++;
            return 0;
        }

        int keptWeight = Mathf.FloorToInt(oldWeight * Mathf.Clamp01(dirtyBandRepairWeightKeepRatio));
        float newTsdf = keptWeight > 0
            ? Mathf.Clamp(Mathf.Lerp(oldTsdf, sampleTsdf, 0.5f), -1f, 1f)
            : 1f;
        int newWeight = Mathf.Clamp(keptWeight, 0, maxFusionWeight);
        _tsdf[index] = newTsdf;
        _weights[index] = (byte)newWeight;
        if (_correctionTsdfHits != null && index >= 0 && index < _correctionTsdfHits.Length)
        {
            _correctionTsdf[index] = sampleTsdf;
            _correctionTsdfHits[index] = (byte)Mathf.Max(_correctionTsdfHits[index], 1);
            _correctionTsdfLastFrame[index] = LastRawFrameIndex;
        }
        RecordContributionLedger(index, "band_repair", oldTsdf, oldWeight, sampleTsdf, 1f, _tsdf[index], _weights[index]);
        RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.CleanupNeighbor, oldWeight, oldTsdf, sampleTsdf);
        MarkDirtyTsdfQuarantine(index);
        LastUpdatedVoxelCount++;
        return 1;
    }

    private bool TryReplaceDirtyTsdf(int index, int oldWeight)
    {
        if (!replaceDirtyTsdfOnStableConflict)
        {
            LastDirtyTsdfReplaceBlockedDisabledCount++;
            return false;
        }

        if (oldWeight > Mathf.Max(1, maxDirtyTsdfReplaceWeight))
        {
            LastDirtyTsdfReplaceBlockedWeightCount++;
            return false;
        }
        if (rejectDirtyReplaceOnCleanHistoryMismatch &&
            AuditVoteHasTag("clean") &&
            FormalCleanHasHistoryMismatch())
        {
            LastDirtyTsdfReplaceBlockedCleanHistoryCount++;
            return false;
        }

        if (_correctionTsdfHits[index] < Mathf.Max(1, minDirtyTsdfReplaceFrames))
        {
            if (TryGuardedReplaceDirtyTsdf(index, oldWeight))
                return true;
            LastDirtyTsdfReplaceBlockedHitsCount++;
            return false;
        }

        ReplaceDirtyTsdfNow(index, oldWeight, false);
        return true;
    }

    private bool TryGuardedReplaceDirtyTsdf(int index, int oldWeight)
    {
        if (!enableGuardedDirtyTsdfFastReplace ||
            _correctionTsdfHits == null ||
            _correctionTsdf == null ||
            index < 0 ||
            index >= _correctionTsdfHits.Length)
        {
            return false;
        }

        int hits = _correctionTsdfHits[index];
        if (hits < Mathf.Max(1, minGuardedDirtyTsdfReplaceFrames))
        {
            LastGuardedDirtyTsdfBlockedHitsCount++;
            return false;
        }
        if (oldWeight > Mathf.Max(1, maxGuardedDirtyTsdfReplaceWeight))
        {
            LastGuardedDirtyTsdfBlockedWeightCount++;
            return false;
        }

        float replacementTsdf = Mathf.Clamp(_correctionTsdf[index], -1f, 1f);
        if (Mathf.Abs(replacementTsdf) > Mathf.Clamp(maxGuardedDirtyTsdfReplaceAbsValue, 0.05f, 1f))
        {
            LastGuardedDirtyTsdfBlockedValueCount++;
            return false;
        }

        ReplaceDirtyTsdfNow(index, oldWeight, true);
        return true;
    }

    private void ReplaceDirtyTsdfNow(int index, int oldWeight, bool guarded)
    {
        float oldTsdf = _tsdf[index];
        float replacementTsdf = Mathf.Clamp(_correctionTsdf[index], -1f, 1f);
        _tsdf[index] = replacementTsdf;
        _weights[index] = (byte)Mathf.Clamp(dirtyTsdfReplaceWeight, minSurfaceCornerWeight, maxFusionWeight);
        RecordContributionLedger(index, guarded ? "guarded_replace" : "replace", oldTsdf, oldWeight, replacementTsdf, dirtyTsdfReplaceWeight, _tsdf[index], _weights[index]);
        RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.ReplaceDirty, oldWeight, oldTsdf, replacementTsdf);
        MarkDirtyTsdfQuarantine(index);
        CleanupDirtyTsdfNeighborhood(index, replacementTsdf);
        _correctionTsdfHits[index] = 0;
        _correctionTsdfLastFrame[index] = int.MinValue;
        LastReplacedDirtyTsdfCount++;
        if (guarded)
            LastGuardedDirtyTsdfReplaceCount++;
        LastUpdatedVoxelCount++;
    }

    private void CleanupDirtyTsdfNeighborhood(int centerIndex, float replacementTsdf)
    {
        if (!cleanupDirtyTsdfNeighborhood || _tsdf == null || _weights == null)
            return;

        int radius = Mathf.Clamp(dirtyTsdfCleanupRadiusVoxels, 0, 2);
        if (radius <= 0)
            return;

        IndexToVoxel(centerIndex, out int cx, out int cy, out int cz);
        int maxWeight = Mathf.Max(1, maxDirtyTsdfCleanupWeight);
        float replacementSign = Mathf.Sign(replacementTsdf);

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = cx + dx;
                    int vy = cy + dy;
                    int vz = cz + dz;
                    if (vx < 0 || vy < 0 || vz < 0 || vx >= _dimX || vy >= _dimY || vz >= _dimZ)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (index == centerIndex || _weights[index] > maxWeight)
                        continue;

                    bool pendingConflict = VoxelHasPendingTsdfCorrection(index);
                    bool signConflict =
                        !cleanupOnlyPendingDirtyTsdfNeighbors &&
                        Mathf.Sign(_tsdf[index]) != replacementSign &&
                        Mathf.Abs(_tsdf[index] - replacementTsdf) >= conflictCorrectionAgreementThreshold;
                    if (!signConflict && !pendingConflict)
                        continue;

                    int neighborOldWeight = _weights[index];
                    float neighborOldTsdf = _tsdf[index];
                    RecordDirtyTsdfEvidence(index, DirtyTsdfEvidenceReason.CleanupNeighbor, neighborOldWeight, neighborOldTsdf, replacementTsdf);
                    _tsdf[index] = 1f;
                    _weights[index] = 0;
                    RecordContributionLedger(index, "cleanup_neighbor", neighborOldTsdf, neighborOldWeight, replacementTsdf, 0f, _tsdf[index], _weights[index]);
                    if (_correctionTsdfHits != null && index < _correctionTsdfHits.Length)
                    {
                        _correctionTsdfHits[index] = 0;
                        _correctionTsdfLastFrame[index] = int.MinValue;
                    }
                    MarkDirtyTsdfQuarantine(index);
                    LastCleanedDirtyTsdfNeighborCount++;
                    LastUpdatedVoxelCount++;
                }
            }
        }
    }

    private void MarkDirtyTsdfQuarantine(int index)
    {
        if (_dirtyTsdfLastFrame == null || index < 0 || index >= _dirtyTsdfLastFrame.Length)
            return;
        int previousFrame = _dirtyTsdfLastFrame[index];
        bool wasActive =
            previousFrame != int.MinValue &&
            LastRawFrameIndex - previousFrame <= Mathf.Max(0, dirtyTsdfQuarantineFrames);
        if (wasActive)
            LastDirtyTsdfQuarantineRefreshCount++;
        else
            LastDirtyTsdfQuarantineNewCount++;
        _dirtyTsdfLastFrame[index] = LastRawFrameIndex;
        LastDirtyTsdfQuarantineCount++;
    }

    private void UpdateDirtyTsdfLifecycleDiagnostics()
    {
        LastDirtyTsdfActiveCount = 0;
        LastDirtyTsdfExpiredCount = 0;
        if (_dirtyTsdfLastFrame == null)
            return;

        int maxAge = Mathf.Max(0, dirtyTsdfQuarantineFrames);
        for (int i = 0; i < _dirtyTsdfLastFrame.Length; i++)
        {
            int frame = _dirtyTsdfLastFrame[i];
            if (frame == int.MinValue)
                continue;
            if (LastRawFrameIndex - frame <= maxAge)
                LastDirtyTsdfActiveCount++;
            else
                LastDirtyTsdfExpiredCount++;
        }
    }

    private void RecordDirtyTsdfEvidence(int index, DirtyTsdfEvidenceReason reason, int oldWeight, float oldTsdf, float newTsdf)
    {
        if (!showDirtyTsdfEvidenceOverlay || index < 0 || index >= _dimX * _dimY * _dimZ)
            return;

        IndexToVoxel(index, out int x, out int y, out int z);
        DirtyTsdfEvidence evidence = new DirtyTsdfEvidence
        {
            VoxelIndex = index,
            FrameIndex = LastRawFrameIndex,
            OldWeight = oldWeight,
            OldTsdf = oldTsdf,
            NewTsdf = newTsdf,
            VoxelCenter = VoxelCenter(x, y, z),
            CameraPosition = _activeTsdfSourceValid ? _activeTsdfSourceCamera : GetCameraPosition(),
            SurfacePoint = _activeTsdfSourceValid ? _activeTsdfSourceSurface : VoxelCenter(x, y, z),
            Reason = reason
        };

        int limit = Mathf.Max(1, maxDirtyTsdfEvidenceMarkers);
        while (_dirtyTsdfEvidence.Count >= limit)
            _dirtyTsdfEvidence.RemoveAt(0);
        _dirtyTsdfEvidence.Add(evidence);
    }

    private void UpdateDirtyTsdfEvidenceOverlay()
    {
        if (!showDirtyTsdfEvidenceOverlay)
        {
            if (_dirtyEvidenceObject != null)
                _dirtyEvidenceObject.SetActive(false);
            return;
        }

        EnsureDirtyTsdfEvidenceObjects();
        if (_dirtyEvidenceMesh == null)
            return;

        _dirtyEvidenceMesh.Clear();
        int start = Mathf.Max(0, _dirtyTsdfEvidence.Count - Mathf.Max(1, maxDirtyTsdfEvidenceMarkers));
        float marker = Mathf.Max(0.005f, dirtyTsdfEvidenceMarkerSizeMeters);
        List<Vector3> vertices = new List<Vector3>((_dirtyTsdfEvidence.Count - start) * 32);
        List<int> indices = new List<int>((_dirtyTsdfEvidence.Count - start) * 48);
        List<Color> colors = new List<Color>((_dirtyTsdfEvidence.Count - start) * 32);
        LastDirtyEvidenceRenderedCount = 0;
        LastDirtyEvidenceBackfillCount = 0;

        for (int i = start; i < _dirtyTsdfEvidence.Count; i++)
        {
            DirtyTsdfEvidence evidence = _dirtyTsdfEvidence[i];
            Color color = DirtyTsdfEvidenceColor(evidence.Reason);
            Vector3 lineStart = evidence.CameraPosition;
            Vector3 lineEnd = evidence.VoxelCenter;
            if ((lineEnd - lineStart).sqrMagnitude > 9f)
                lineStart = Vector3.Lerp(lineEnd, lineStart, 3f / Mathf.Max(0.001f, (lineEnd - lineStart).magnitude));

            AddEvidenceLine(vertices, indices, colors, lineStart, lineEnd, color);
            AddEvidenceCube(vertices, indices, colors, lineEnd, marker, color);
            LastDirtyEvidenceRenderedCount++;
        }

        AddCurrentDirtyTsdfBackfillMarkers(vertices, indices, colors, marker);

        _dirtyEvidenceMesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _dirtyEvidenceMesh.SetVertices(vertices);
        _dirtyEvidenceMesh.SetColors(colors);
        _dirtyEvidenceMesh.SetIndices(indices, MeshTopology.Lines, 0, true);
        _dirtyEvidenceMesh.RecalculateBounds();
        _dirtyEvidenceObject.SetActive(vertices.Count > 0);
    }

    private void AddCurrentDirtyTsdfBackfillMarkers(List<Vector3> vertices, List<int> indices, List<Color> colors, float marker)
    {
        if (!backfillCurrentDirtyTsdfMarkers || _tsdf == null || _weights == null)
            return;

        int maxMarkers = Mathf.Max(1, maxDirtyTsdfEvidenceMarkers);
        int stride = Mathf.Max(1, Mathf.CeilToInt(Mathf.Pow((_dimX * _dimY * _dimZ) / Mathf.Max(1f, maxMarkers * 12f), 1f / 3f)));
        for (int z = 0; z < _dimZ && LastDirtyEvidenceBackfillCount < maxMarkers; z += stride)
        {
            for (int y = 0; y < _dimY && LastDirtyEvidenceBackfillCount < maxMarkers; y += stride)
            {
                for (int x = 0; x < _dimX && LastDirtyEvidenceBackfillCount < maxMarkers; x += stride)
                {
                    int index = Index(x, y, z);
                    bool dirty = VoxelIsDirtyQuarantined(index);
                    bool pending = VoxelHasPendingTsdfCorrection(index);
                    if (!dirty && !pending)
                        continue;

                    Color color = dirty ? new Color(0f, 0.35f, 1f, 1f) : Color.white;
                    color.a = Mathf.Clamp01(dirtyTsdfEvidenceLineAlpha);
                    AddEvidenceCube(vertices, indices, colors, VoxelCenter(x, y, z), marker * 0.8f, color);
                    LastDirtyEvidenceBackfillCount++;
                }
            }
        }
    }

    private static void AddEvidenceLine(List<Vector3> vertices, List<int> indices, List<Color> colors, Vector3 a, Vector3 b, Color color)
    {
        int start = vertices.Count;
        vertices.Add(a);
        vertices.Add(b);
        indices.Add(start);
        indices.Add(start + 1);
        colors.Add(color);
        colors.Add(color);
    }

    private static void AddEvidenceCube(List<Vector3> vertices, List<int> indices, List<Color> colors, Vector3 center, float size, Color color)
    {
        float h = Mathf.Max(0.002f, size) * 0.5f;
        int start = vertices.Count;
        vertices.Add(center + new Vector3(-h, -h, -h));
        vertices.Add(center + new Vector3( h, -h, -h));
        vertices.Add(center + new Vector3( h,  h, -h));
        vertices.Add(center + new Vector3(-h,  h, -h));
        vertices.Add(center + new Vector3(-h, -h,  h));
        vertices.Add(center + new Vector3( h, -h,  h));
        vertices.Add(center + new Vector3( h,  h,  h));
        vertices.Add(center + new Vector3(-h,  h,  h));

        AddEvidenceEdge(indices, start, 0, 1);
        AddEvidenceEdge(indices, start, 1, 2);
        AddEvidenceEdge(indices, start, 2, 3);
        AddEvidenceEdge(indices, start, 3, 0);
        AddEvidenceEdge(indices, start, 4, 5);
        AddEvidenceEdge(indices, start, 5, 6);
        AddEvidenceEdge(indices, start, 6, 7);
        AddEvidenceEdge(indices, start, 7, 4);
        AddEvidenceEdge(indices, start, 0, 4);
        AddEvidenceEdge(indices, start, 1, 5);
        AddEvidenceEdge(indices, start, 2, 6);
        AddEvidenceEdge(indices, start, 3, 7);

        for (int i = 0; i < 8; i++)
            colors.Add(color);
    }

    private static void AddEvidenceEdge(List<int> indices, int start, int a, int b)
    {
        indices.Add(start + a);
        indices.Add(start + b);
    }

    private Color DirtyTsdfEvidenceColor(DirtyTsdfEvidenceReason reason)
    {
        Color color;
        switch (reason)
        {
            case DirtyTsdfEvidenceReason.CleanupNeighbor:
                color = Color.black;
                break;
            case DirtyTsdfEvidenceReason.CorrectedOldDepth:
                color = Color.white;
                break;
            case DirtyTsdfEvidenceReason.RejectedConflict:
                color = new Color(1f, 0.55f, 0f, 1f);
                break;
            case DirtyTsdfEvidenceReason.ReplaceDirty:
            case DirtyTsdfEvidenceReason.PendingConflict:
            case DirtyTsdfEvidenceReason.RepeatedPending:
            default:
                color = new Color(0f, 0.35f, 1f, 1f);
                break;
        }
        color.a = Mathf.Clamp01(dirtyTsdfEvidenceLineAlpha);
        return color;
    }

    private bool PassMultiFrameTsdfStability(int index, float sampleTsdf, int oldWeight)
    {
        if (!requireMultiFrameStableTsdf || oldWeight >= stableTsdfBypassWeight)
            return true;
        if (_pendingTsdf == null || _pendingTsdfHits == null || _pendingTsdfLastFrame == null || index < 0 || index >= _pendingTsdf.Length)
            return true;

        int frameIndex = LastRawFrameIndex;
        if (_pendingTsdfHits[index] <= 0 ||
            Mathf.Sign(_pendingTsdf[index]) != Mathf.Sign(sampleTsdf) ||
            Mathf.Abs(_pendingTsdf[index] - sampleTsdf) > stableTsdfAgreementThreshold)
        {
            _pendingTsdf[index] = sampleTsdf;
            _pendingTsdfHits[index] = 1;
            _pendingTsdfLastFrame[index] = frameIndex;
            LastPendingStableTsdfCount++;
            return false;
        }

        if (_pendingTsdfLastFrame[index] != frameIndex)
        {
            int nextHits = Mathf.Min(255, _pendingTsdfHits[index] + 1);
            _pendingTsdfHits[index] = (byte)nextHits;
            _pendingTsdfLastFrame[index] = frameIndex;
            _pendingTsdf[index] = Mathf.Lerp(_pendingTsdf[index], sampleTsdf, 0.35f);
        }

        if (_pendingTsdfHits[index] < minStableTsdfFrames)
        {
            LastPendingStableTsdfCount++;
            return false;
        }

        return true;
    }

    private bool ShouldCarveFreeSpaceAtPixel(int x, int y, int stride)
    {
        if (!enableProjectiveFreeSpaceCarving)
            return false;
        int carveStride = Mathf.Max(1, freeSpaceCarvingPixelStride);
        int sx = stride > 1 ? x / stride : x;
        int sy = stride > 1 ? y / stride : y;
        return ((sx + sy) % carveStride) == 0;
    }

    private void MarkRejectedObservationBand(Vector3 cameraPosition, Vector3 point, Vector3 normal, int halfBandSteps)
    {
        if (!showHoleBoundaryDiagnosis)
            return;
        ObservationVoteState savedVote = _activeAtomicBandVote;
        byte savedState = _activeSurfaceObservationState;
        _activeAtomicBandVote = ObservationVoteState.Reject;
        _activeSurfaceObservationState = 4;
        try
        {
            if (useProjectiveTsdfIntegration)
                IntegrateProjectiveTsdfBand(cameraPosition, point, 0f, halfBandSteps, false, true);
            else
                IntegrateNormalTsdfBand(point, normal, 0f, halfBandSteps, true);
        }
        finally
        {
            _activeAtomicBandVote = savedVote;
            _activeSurfaceObservationState = savedState;
        }
    }

    private bool IntegrateProjectiveTsdfBand(Vector3 cameraPosition, Vector3 surfacePoint, float sampleWeight, int halfBandSteps, bool allowFreeSpaceCarving, bool diagnosticOnly = false)
    {
        Vector3 toSurface = surfacePoint - cameraPosition;
        float surfaceDepth = toSurface.magnitude;
        if (!Finite(toSurface) || surfaceDepth <= 0.0001f)
            return false;

        Vector3 rayDirection = toSurface / surfaceDepth;
        float voxel = Mathf.Max(0.0001f, voxelSizeMeters);
        float startDepth = Mathf.Max(0.01f, surfaceDepth - halfBandSteps * voxel);
        float endDepth = surfaceDepth + halfBandSteps * voxel;
        bool touchedDenseTsdf = false;
        _activeTsdfSourceCamera = cameraPosition;
        _activeTsdfSourceSurface = surfacePoint;
        _activeTsdfSourceValid = true;

        if (allowFreeSpaceCarving)
            CarveProjectiveFreeSpace(cameraPosition, rayDirection, surfaceDepth, sampleWeight);

        Vector3 segmentStart = cameraPosition + rayDirection * startDepth;
        Vector3 segmentEnd = cameraPosition + rayDirection * endDepth;
        float lateralRadius = voxel * Mathf.Clamp(projectiveVoxelCenterRadiusScale, 0.5f, 1.25f);
        Vector3 boundsMin = Vector3.Min(segmentStart, segmentEnd) - Vector3.one * lateralRadius;
        Vector3 boundsMax = Vector3.Max(segmentStart, segmentEnd) + Vector3.one * lateralRadius;
        Vector3 minGrid = (boundsMin - _volumeOriginWorld) / voxel;
        Vector3 maxGrid = (boundsMax - _volumeOriginWorld) / voxel;
        int minX = Mathf.Clamp(Mathf.FloorToInt(minGrid.x), 0, _dimX - 1);
        int minY = Mathf.Clamp(Mathf.FloorToInt(minGrid.y), 0, _dimY - 1);
        int minZ = Mathf.Clamp(Mathf.FloorToInt(minGrid.z), 0, _dimZ - 1);
        int maxX = Mathf.Clamp(Mathf.CeilToInt(maxGrid.x), 0, _dimX - 1);
        int maxY = Mathf.Clamp(Mathf.CeilToInt(maxGrid.y), 0, _dimY - 1);
        int maxZ = Mathf.Clamp(Mathf.CeilToInt(maxGrid.z), 0, _dimZ - 1);
        float lateralRadiusSq = lateralRadius * lateralRadius;
        float depthEpsilon = voxel * 0.5f;

        for (int vz = minZ; vz <= maxZ; vz++)
        {
            for (int vy = minY; vy <= maxY; vy++)
            {
                for (int vx = minX; vx <= maxX; vx++)
                {
                    Vector3 center = VoxelCenter(vx, vy, vz);
                    float centerDepth = Vector3.Dot(center - cameraPosition, rayDirection);
                    if (centerDepth < startDepth - depthEpsilon || centerDepth > endDepth + depthEpsilon)
                        continue;

                    Vector3 projectedCenter = cameraPosition + rayDirection * centerDepth;
                    if ((center - projectedCenter).sqrMagnitude > lateralRadiusSq)
                        continue;

                    float signedDistance = surfaceDepth - centerDepth;
                    float sampleTsdf = Mathf.Clamp(signedDistance / Mathf.Max(0.0001f, truncationMeters), -1f, 1f);
                    int voxelIndex = Index(vx, vy, vz);
                    MarkSurfaceBandVisit(voxelIndex, sampleTsdf);
                    if (Mathf.Abs(signedDistance) <= voxel * 0.75f)
                        MarkSurfaceObservationVoxel(voxelIndex, _activeSurfaceObservationState);
                    if (!diagnosticOnly)
                        IntegrateVoxel(vx, vy, vz, sampleTsdf, sampleWeight);
                    touchedDenseTsdf = true;
                }
            }
        }

        _activeTsdfSourceValid = false;
        return touchedDenseTsdf;
    }

    private void CarveProjectiveFreeSpace(Vector3 cameraPosition, Vector3 rayDirection, float surfaceDepth, float sampleWeight)
    {
        float carveEnd = surfaceDepth - Mathf.Max(truncationMeters, voxelSizeMeters);
        if (carveEnd <= minDepthMeters)
            return;

        float carveWeight = Mathf.Clamp01(sampleWeight) * Mathf.Clamp01(freeSpaceCarvingWeightScale);
        if (carveWeight <= 0.0001f)
            return;

        float voxel = Mathf.Max(0.0001f, voxelSizeMeters);
        Vector3 halfVoxel = Vector3.one * (voxel * 0.5f);
        Vector3 volumeMin = _volumeOriginWorld - halfVoxel;
        Vector3 volumeMax = _volumeOriginWorld + new Vector3(
            Mathf.Max(0, _dimX - 1) * voxel,
            Mathf.Max(0, _dimY - 1) * voxel,
            Mathf.Max(0, _dimZ - 1) * voxel) + halfVoxel;
        float traversalStart = Mathf.Max(0.01f, minDepthMeters);
        float traversalEnd = carveEnd;
        if (!ClipRaySegmentToBounds(cameraPosition, rayDirection, volumeMin, volumeMax, ref traversalStart, ref traversalEnd))
            return;

        float interiorStart = Mathf.Min(traversalEnd, traversalStart + voxel * 0.0001f);
        Vector3 startPoint = cameraPosition + rayDirection * interiorStart;
        Vector3 grid = (startPoint - volumeMin) / voxel;
        int vx = Mathf.Clamp(Mathf.FloorToInt(grid.x), 0, _dimX - 1);
        int vy = Mathf.Clamp(Mathf.FloorToInt(grid.y), 0, _dimY - 1);
        int vz = Mathf.Clamp(Mathf.FloorToInt(grid.z), 0, _dimZ - 1);
        int stepX = rayDirection.x > 0f ? 1 : rayDirection.x < 0f ? -1 : 0;
        int stepY = rayDirection.y > 0f ? 1 : rayDirection.y < 0f ? -1 : 0;
        int stepZ = rayDirection.z > 0f ? 1 : rayDirection.z < 0f ? -1 : 0;
        float tMaxX = NextVoxelBoundaryDepth(cameraPosition.x, rayDirection.x, volumeMin.x, voxel, vx, stepX);
        float tMaxY = NextVoxelBoundaryDepth(cameraPosition.y, rayDirection.y, volumeMin.y, voxel, vy, stepY);
        float tMaxZ = NextVoxelBoundaryDepth(cameraPosition.z, rayDirection.z, volumeMin.z, voxel, vz, stepZ);
        float tDeltaX = stepX == 0 ? float.PositiveInfinity : voxel / Mathf.Abs(rayDirection.x);
        float tDeltaY = stepY == 0 ? float.PositiveInfinity : voxel / Mathf.Abs(rayDirection.y);
        float tDeltaZ = stepZ == 0 ? float.PositiveInfinity : voxel / Mathf.Abs(rayDirection.z);

        while (vx >= 0 && vy >= 0 && vz >= 0 && vx < _dimX && vy < _dimY && vz < _dimZ)
        {
            CarveFreeSpaceVoxel(Index(vx, vy, vz), carveWeight);

            float nextDepth = Mathf.Min(tMaxX, Mathf.Min(tMaxY, tMaxZ));
            if (nextDepth > traversalEnd + 0.0001f)
                break;

            const float boundaryEpsilon = 0.00001f;
            if (tMaxX <= nextDepth + boundaryEpsilon)
            {
                vx += stepX;
                tMaxX += tDeltaX;
            }
            if (tMaxY <= nextDepth + boundaryEpsilon)
            {
                vy += stepY;
                tMaxY += tDeltaY;
            }
            if (tMaxZ <= nextDepth + boundaryEpsilon)
            {
                vz += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }

    private static bool ClipRaySegmentToBounds(
        Vector3 origin,
        Vector3 direction,
        Vector3 boundsMin,
        Vector3 boundsMax,
        ref float segmentStart,
        ref float segmentEnd)
    {
        return ClipRayAxis(origin.x, direction.x, boundsMin.x, boundsMax.x, ref segmentStart, ref segmentEnd) &&
               ClipRayAxis(origin.y, direction.y, boundsMin.y, boundsMax.y, ref segmentStart, ref segmentEnd) &&
               ClipRayAxis(origin.z, direction.z, boundsMin.z, boundsMax.z, ref segmentStart, ref segmentEnd) &&
               segmentEnd >= segmentStart;
    }

    private static bool ClipRayAxis(
        float origin,
        float direction,
        float axisMin,
        float axisMax,
        ref float segmentStart,
        ref float segmentEnd)
    {
        if (Mathf.Abs(direction) <= 0.000001f)
            return origin >= axisMin && origin <= axisMax;

        float inverse = 1f / direction;
        float near = (axisMin - origin) * inverse;
        float far = (axisMax - origin) * inverse;
        if (near > far)
        {
            float swap = near;
            near = far;
            far = swap;
        }

        segmentStart = Mathf.Max(segmentStart, near);
        segmentEnd = Mathf.Min(segmentEnd, far);
        return segmentEnd >= segmentStart;
    }

    private static float NextVoxelBoundaryDepth(
        float rayOrigin,
        float rayDirection,
        float gridMin,
        float voxel,
        int voxelIndex,
        int step)
    {
        if (step == 0 || Mathf.Abs(rayDirection) <= 0.000001f)
            return float.PositiveInfinity;
        float boundary = gridMin + (voxelIndex + (step > 0 ? 1 : 0)) * voxel;
        return (boundary - rayOrigin) / rayDirection;
    }

    private void CarveFreeSpaceVoxel(int index, float carveWeight)
    {
        if (_weights == null || _tsdf == null || index < 0 || index >= _weights.Length)
            return;

        int oldWeight = _weights[index];
        if (oldWeight <= 0)
            return;

        LastFreeSpaceEvidenceCandidateCount++;
        if (protectSameFrameSurfaceFromClearing &&
            _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance sameFrameProvenance) &&
            IsConfirmedSameFrameSurfaceWrite(index, sameFrameProvenance))
        {
            LastFreeSpaceEvidenceBlockedSameFrameCount++;
            return;
        }

        if (oldWeight > maxFreeSpaceCarvableWeight)
        {
            LastFreeSpaceEvidenceBlockedHighWeightCount++;
            return;
        }

        float oldTsdf = _tsdf[index];
        if (oldTsdf >= 0.98f)
            return;

        if (useTemporalFreeSpaceEvidence)
        {
            if (_freeSpaceEvidenceHits == null || _freeSpaceEvidenceLastFrame == null || index >= _freeSpaceEvidenceHits.Length)
                return;

            int previousFrame = _freeSpaceEvidenceLastFrame[index];
            if (previousFrame == LastRawFrameIndex)
            {
                LastFreeSpaceEvidenceDuplicateFrameCount++;
                return;
            }

            int frameGap = previousFrame == int.MinValue ? int.MaxValue : LastRawFrameIndex - previousFrame;
            if (previousFrame == int.MinValue || frameGap <= 0 || frameGap > Mathf.Max(2, maxFreeSpaceEvidenceFrameGap))
            {
                _freeSpaceEvidenceHits[index] = 1;
                LastFreeSpaceEvidenceNewCount++;
            }
            else
            {
                _freeSpaceEvidenceHits[index] = (byte)Mathf.Min(255, _freeSpaceEvidenceHits[index] + 1);
                LastFreeSpaceEvidenceRepeatCount++;
            }
            _freeSpaceEvidenceLastFrame[index] = LastRawFrameIndex;

            if (_freeSpaceEvidenceHits[index] < Mathf.Clamp(minFreeSpaceEvidenceFrames, 2, 8))
            {
                LastFreeSpaceEvidenceWaitingCount++;
                return;
            }
        }

        float blend = Mathf.Clamp01(freeSpaceCarvingBlend) * Mathf.Clamp01(carveWeight);
        float candidateTsdf = Mathf.Lerp(oldTsdf, 1f, blend);
        int decay = useTemporalFreeSpaceEvidence ? Mathf.Clamp(freeSpaceEvidenceWeightDecay, 1, 4) : 1;
        int candidateWeight = Mathf.Max(0, oldWeight - decay);
        int clearWeight = Mathf.Clamp(freeSpaceEvidenceClearWeight, 0, 4);
        bool commitClear = candidateWeight <= clearWeight &&
                           candidateTsdf >= Mathf.Clamp01(freeSpaceEvidenceClearTsdf);

        _tsdf[index] = candidateTsdf;
        _weights[index] = (byte)(commitClear ? 0 : Mathf.Max(1, candidateWeight));
        LastFreeSpaceEvidenceAppliedCount++;
        if (commitClear)
        {
            _tsdf[index] = 1f;
            if (_clearedVoxelLastFrame != null && index < _clearedVoxelLastFrame.Length)
                _clearedVoxelLastFrame[index] = LastRawFrameIndex;
            if (_provisionalTsdf != null && index < _provisionalTsdf.Length)
            {
                _provisionalTsdf[index] = 0;
                _provisionalTsdfHits[index] = 0;
                _provisionalTsdfLastFrame[index] = int.MinValue;
            }
            if (_freeSpaceEvidenceHits != null && index < _freeSpaceEvidenceHits.Length)
            {
                _freeSpaceEvidenceHits[index] = 0;
                _freeSpaceEvidenceLastFrame[index] = int.MinValue;
            }
            LastFreeSpaceEvidenceClearedCount++;
        }
        RecordContributionLedger(index, "free_space_carve", oldTsdf, oldWeight, 1f, carveWeight, _tsdf[index], _weights[index]);
        LastCarvedFreeSpaceVoxelCount++;
    }

    private bool IsConfirmedSameFrameSurfaceWrite(int index, VoxelWriteProvenance provenance)
    {
        if (provenance.Frame != LastRawFrameIndex || provenance.LastOperation == "free_space_carve")
            return false;

        bool trustedSurfaceOperation =
            provenance.LastOperation == "integrate" ||
            provenance.LastOperation == "strong_current_promote" ||
            provenance.LastOperation == "replace" ||
            provenance.LastOperation == "guarded_replace" ||
            provenance.LastOperation == "band_repair" ||
            provenance.LastOperation == "conflict_correct";
        if (!trustedSurfaceOperation)
            return false;

        Vector3 voxelCenter = VoxelCenterFromIndex(index);
        float surfaceDistance = Vector3.Distance(voxelCenter, provenance.SurfacePoint);
        float surfaceWindow = Mathf.Max(
            Mathf.Max(0.005f, voxelSizeMeters) * 0.9f,
            Mathf.Max(0.001f, truncationMeters) * Mathf.Clamp(nearZeroSurfaceThreshold, 0.02f, 0.45f));
        if (surfaceDistance <= surfaceWindow)
            return true;

        float nearZeroWindow = Mathf.Max(
            Mathf.Clamp(nearZeroSurfaceThreshold, 0.02f, 0.45f),
            Mathf.Max(0.005f, voxelSizeMeters) / Mathf.Max(0.001f, truncationMeters) * 0.5f);
        return Mathf.Abs(provenance.Tsdf) <= nearZeroWindow;
    }

    private bool ShouldCancelFreeSpaceEvidence(int index, string operation, float observedTsdf)
    {
        bool trustedSurfaceOperation =
            operation == "integrate" ||
            operation == "strong_current_promote" ||
            operation == "replace" ||
            operation == "guarded_replace" ||
            operation == "band_repair" ||
            operation == "conflict_correct";
        if (!trustedSurfaceOperation)
            return false;

        float nearZeroWindow = Mathf.Max(
            Mathf.Clamp(nearZeroSurfaceThreshold, 0.02f, 0.45f),
            Mathf.Max(0.005f, voxelSizeMeters) / Mathf.Max(0.001f, truncationMeters) * 0.5f);
        if (Mathf.Abs(observedTsdf) > nearZeroWindow)
            return false;

        return _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) &&
               IsConfirmedSameFrameSurfaceWrite(index, provenance);
    }

    private void CancelFreeSpaceEvidence(int index)
    {
        if (_freeSpaceEvidenceHits == null || _freeSpaceEvidenceLastFrame == null || index < 0 || index >= _freeSpaceEvidenceHits.Length)
            return;
        if (_freeSpaceEvidenceHits[index] == 0)
            return;
        _freeSpaceEvidenceHits[index] = 0;
        _freeSpaceEvidenceLastFrame[index] = int.MinValue;
        LastFreeSpaceEvidenceCancelledBySurfaceCount++;
    }

    private void ResetFreeSpaceEvidenceDiagnostics()
    {
        LastFreeSpaceEvidenceCandidateCount = 0;
        LastFreeSpaceEvidenceNewCount = 0;
        LastFreeSpaceEvidenceRepeatCount = 0;
        LastFreeSpaceEvidenceWaitingCount = 0;
        LastFreeSpaceEvidenceAppliedCount = 0;
        LastFreeSpaceEvidenceClearedCount = 0;
        LastFreeSpaceEvidenceBlockedHighWeightCount = 0;
        LastFreeSpaceEvidenceBlockedSameFrameCount = 0;
        LastFreeSpaceEvidenceDuplicateFrameCount = 0;
        LastFreeSpaceEvidenceCancelledBySurfaceCount = 0;
    }

    private bool IntegrateNormalTsdfBand(Vector3 point, Vector3 normal, float sampleWeight, int halfBandSteps, bool diagnosticOnly = false)
    {
        bool touchedDenseTsdf = false;
        _activeTsdfSourceCamera = GetCameraPosition();
        _activeTsdfSourceSurface = point;
        _activeTsdfSourceValid = true;
        for (int step = -halfBandSteps; step <= halfBandSteps; step++)
        {
            float signedDistance = step * voxelSizeMeters;
            Vector3 voxelWorld = point + normal * signedDistance;
            if (!TryWorldToVoxel(voxelWorld, out int vx, out int vy, out int vz))
                continue;

            touchedDenseTsdf = true;
            float sampleTsdf = Mathf.Clamp(signedDistance / truncationMeters, -1f, 1f);
            int voxelIndex = Index(vx, vy, vz);
            MarkSurfaceBandVisit(voxelIndex, sampleTsdf);
            if (Mathf.Abs(signedDistance) <= Mathf.Max(0.0001f, voxelSizeMeters) * 0.75f)
                MarkSurfaceObservationVoxel(voxelIndex, _activeSurfaceObservationState);
            if (!diagnosticOnly)
                IntegrateVoxel(vx, vy, vz, sampleTsdf, sampleWeight);
        }

        _activeTsdfSourceValid = false;
        return touchedDenseTsdf;
    }

    private void AddFallbackSurfaceSample(Vector3 point, Vector3 normal)
    {
        if (!showFallbackSurfaceSamples)
            return;
        if (useChunkedSurfaceSamples)
        {
            AddChunkedFallbackSurfaceSample(point, normal);
            return;
        }

        if (!TryWorldToVoxel(point, out int x, out int y, out int z))
        {
            LastRejectedOutsideVolumeCount++;
            return;
        }

        int voxelIndex = Index(x, y, z);
        if (!_fallbackSurfaceVoxels.Add(voxelIndex))
            return;

        Vector3 stableNormal = normal.sqrMagnitude > 0.0001f ? normal.normalized : Vector3.up;
        int maxSamples = Mathf.Max(1, maxFallbackSurfaceSamples);
        if (_fallbackSurfacePoints.Count < maxSamples)
        {
            _fallbackSurfacePoints.Add(point);
            _fallbackSurfaceNormals.Add(stableNormal);
            _fallbackSurfaceVoxelKeys.Add(voxelIndex);
            return;
        }

        LastSkippedFallbackCapacityCount++;
        if (!replaceFallbackSamplesWhenFull)
            return;

        int replaceIndex = _fallbackSurfaceReplaceCursor % maxSamples;
        _fallbackSurfaceReplaceCursor = (_fallbackSurfaceReplaceCursor + 1) % maxSamples;
        if (replaceIndex < _fallbackSurfaceVoxelKeys.Count)
        {
            int oldVoxelIndex = _fallbackSurfaceVoxelKeys[replaceIndex];
            if (oldVoxelIndex != voxelIndex)
                _fallbackSurfaceVoxels.Remove(oldVoxelIndex);
            _fallbackSurfaceVoxelKeys[replaceIndex] = voxelIndex;
        }
        else
        {
            _fallbackSurfaceVoxelKeys.Add(voxelIndex);
        }
        _fallbackSurfacePoints[replaceIndex] = point;
        if (replaceIndex < _fallbackSurfaceNormals.Count)
            _fallbackSurfaceNormals[replaceIndex] = stableNormal;
    }

    private void AddChunkedFallbackSurfaceSample(Vector3 point, Vector3 normal)
    {
        Vector3Int chunkKey = WorldToSurfaceChunkKey(point);
        if (!_surfaceChunks.TryGetValue(chunkKey, out SurfaceChunk chunk))
        {
            if (_surfaceChunks.Count >= Mathf.Max(1, maxSurfaceChunks))
                RemoveOldestSurfaceChunk();

            chunk = new SurfaceChunk();
            _surfaceChunks[chunkKey] = chunk;
            _surfaceChunkOrder.Add(chunkKey);
            LastSurfaceChunkCount = _surfaceChunks.Count;
        }

        Vector3 stableNormal = normal.sqrMagnitude > 0.0001f ? normal.normalized : Vector3.up;
        Vector3Int localCell = WorldToSurfaceLocalCell(point, stableNormal);
        if (chunk.Cells.TryGetValue(localCell, out SurfaceCell existingCell))
        {
            int oldHits = Mathf.Max(1, existingCell.Hits);
            int newHits = Mathf.Min(oldHits + 1, maxFusionWeight);
            float oldWeight = oldHits;
            float normalAgreement = Vector3.Dot(existingCell.Normal, stableNormal);
            if (normalAgreement < 0.25f)
            {
                LastSurfaceCellConflictCount++;
                if (!splitLightCoverNormalConflicts)
                    return;

                Vector3Int alternateCell = localCell + SurfaceNormalBin(stableNormal) * 2000000 + new Vector3Int(17, 31, 43);
                if (!TryAddNewSurfaceCell(chunk, alternateCell, point, stableNormal))
                    LastSurfaceCellConflictCount++;
                return;
            }

            float pointDelta = Vector3.Distance(existingCell.Point, point);
            if (oldHits < stableSurfaceCellLockHits || pointDelta <= maxStableCellPositionUpdateMeters)
                existingCell.Point = (existingCell.Point * oldWeight + point) / (oldWeight + 1f);

            Vector3 blendedNormal = existingCell.Normal * oldWeight + stableNormal;
            existingCell.Normal = blendedNormal.sqrMagnitude > 0.0001f ? blendedNormal.normalized : stableNormal;
            existingCell.Hits = newHits;
            return;
        }

        TryAddNewSurfaceCell(chunk, localCell, point, stableNormal);
    }

    private bool TryAddNewSurfaceCell(SurfaceChunk chunk, Vector3Int localCell, Vector3 point, Vector3 stableNormal)
    {
        if (chunk.Cells.ContainsKey(localCell))
            return false;

        int maxPerChunk = Mathf.Max(1, maxFallbackSamplesPerChunk);
        if (chunk.Cells.Count >= maxPerChunk)
        {
            LastSkippedFallbackCapacityCount++;
            return false;
        }

        chunk.LocalCells.Add(localCell);
        chunk.LocalCellKeys.Add(localCell);
        chunk.Points.Add(point);
        chunk.Normals.Add(stableNormal);
        chunk.Cells.Add(localCell, new SurfaceCell
        {
            Point = point,
            Normal = stableNormal,
            Hits = 1
        });
        return true;
    }

    private void RemoveOldestSurfaceChunk()
    {
        if (_surfaceChunkOrder.Count <= 0)
            return;

        Vector3Int oldKey = _surfaceChunkOrder[0];
        _surfaceChunkOrder.RemoveAt(0);
        int evictedCells = 0;
        if (_surfaceChunks.TryGetValue(oldKey, out SurfaceChunk oldChunk) && oldChunk != null)
            evictedCells = oldChunk.Cells.Count;
        _surfaceChunks.Remove(oldKey);
        LastSurfaceChunkCount = _surfaceChunks.Count;
        LastEvictedSurfaceChunkCount++;
        LastEvictedSurfaceCellCount += evictedCells;
        TotalEvictedSurfaceChunkCount++;
        TotalEvictedSurfaceCellCount += evictedCells;
        LastSkippedFallbackCapacityCount++;
    }

    private void ResetRejectCounters()
    {
        LastRejectedInvalidPositionCount = 0;
        LastRejectedDepthRangeCount = 0;
        LastRejectedNormalCount = 0;
        LastRejectedFacingCount = 0;
        LastRejectedConfidenceCount = 0;
        LastRejectedOutsideVolumeCount = 0;
        LastRejectedDepthDiscontinuityCount = 0;
        LastRejectedTsdfDepthSupportCount = 0;
        LastRejectedRobustDepthCount = 0;
        LastRobustDepthDownweightedCount = 0;
        LastRobustDepthCorrectedCount = 0;
        LastRejectedDepthEdgeErosionCount = 0;
        LastDepthEdgeDownweightedCount = 0;
        LastRejectedTsdfConflictCount = 0;
        LastPendingStableTsdfCount = 0;
        LastProvisionalTsdfSupportWriteCount = 0;
        LastProvisionalTsdfSupportBlockedCount = 0;
        ResetProvisionalTsdfBlockDiagnostics();
        LastStrongSampleSeedWriteCount = 0;
        LastStrongSampleSeedBlockedCount = 0;
        ResetStrongSampleSeedTemporaryBlockDiagnostics();
        ResetFormalIntegrateGateDiagnostics();
        LastProvisionalLocalSupportPassCount = 0;
        LastProvisionalLocalSupportNeighborCount = 0;
        LastProvisionalLocalSupportStableNeighborCount = 0;
        LastProvisionalLocalSupportAxialNeighborCount = 0;
        LastProvisionalTsdfConfirmedCount = 0;
        LastProvisionalTsdfConfirmedByWeightCount = 0;
        LastProvisionalTsdfRetiredCount = 0;
        LastAtomicAcceptedBandVoxelWriteCount = 0;
        LastAtomicProvisionalBandVoxelWriteCount = 0;
        LastAcceptedSignRecoveryCandidateCount = 0;
        LastAcceptedSignRecoveryNeighborBlockedCount = 0;
        LastAcceptedSignRecoveryAppliedCount = 0;
        LastAtomicPromotedProvisionalVoxelCount = 0;
        LastAtomicRetiredProvisionalVoxelCount = 0;
        LastProvisionalTsdfRetiredExpiredCount = 0;
        LastProvisionalTsdfDirtyClearedCount = 0;
        LastOldCleanMetabolismWatchCount = 0;
        LastOldCleanMetabolismDecayCount = 0;
        LastOldCleanMetabolismClearCount = 0;
        LastOldCleanMetabolismBlockedCount = 0;
        LastOldCleanMetabolismCandidateCount = 0;
        LastOldCleanMetabolismWaitingHitsCount = 0;
        LastOldCleanMetabolismBlockedSupportCount = 0;
        LastOldCleanMetabolismBlockedSameFrameCount = 0;
        LastOldCleanMetabolismBlockedWeightCount = 0;
        LastOldCleanMetabolismBlockedResidualCount = 0;
        LastOldCleanMetabolismBlockedDirtyPendingCount = 0;
        LastOldCleanMetabolismSkippedWeakBandCount = 0;
        LastOldCleanMetabolismSkippedWeakCrossFrameCount = 0;
        LastPendingTsdfCorrectionCount = 0;
        LastCorrectedTsdfCount = 0;
        LastReplacedDirtyTsdfCount = 0;
        LastGuardedDirtyTsdfReplaceCount = 0;
        LastDirtyTsdfBandRepairCount = 0;
        LastDirtyTsdfBandRepairSampleCount = 0;
        LastCleanedDirtyTsdfNeighborCount = 0;
        LastDirtyTsdfQuarantineCount = 0;
        ResetDirtyTsdfDiagnostics();
        LastCarvedFreeSpaceVoxelCount = 0;
        ResetFreeSpaceEvidenceDiagnostics();
        LastRejectedFallbackWeakCount = 0;
        LastSkippedFallbackCapacityCount = 0;
        LastEvictedSurfaceChunkCount = 0;
        LastEvictedSurfaceCellCount = 0;
        LastSurfaceCellConflictCount = 0;
        LastSkippedDirtyLightCoverCellCount = 0;
        LastBlockedDirtyLightCoverSampleCount = 0;
        LastSkippedUnstableLightCoverCellCount = 0;
    }

    private void ResetDirtyTsdfDiagnostics()
    {
        LastDirtyTsdfPendingNewCount = 0;
        LastDirtyTsdfPendingRepeatCount = 0;
        LastDirtyTsdfRejectedConflictCount = 0;
        LastDirtyTsdfBandRepairTriggerCount = 0;
        LastDirtyTsdfBandRepairProbeCount = 0;
        LastDirtyTsdfBandRepairBlockedDisabledCount = 0;
        LastDirtyTsdfBandRepairBlockedNoSampleCount = 0;
        LastDirtyTsdfBandRepairBlockedNoHistoryCount = 0;
        LastDirtyTsdfBandRepairBlockedLowConflictCount = 0;
        LastDirtyTsdfBandRepairBlockedBudgetCount = 0;
        LastDirtyTsdfBandRepairBlockedOutsideCount = 0;
        LastDirtyTsdfBandRepairBlockedEmptyCount = 0;
        LastDirtyTsdfBandRepairBlockedWeightCount = 0;
        LastDirtyTsdfBandRepairBlockedSameSignCount = 0;
        LastDirtyTsdfBandRepairBlockedResidualCount = 0;
        LastDirtyTsdfReplaceBlockedDisabledCount = 0;
        LastDirtyTsdfReplaceBlockedWeightCount = 0;
        LastDirtyTsdfReplaceBlockedHitsCount = 0;
        LastDirtyTsdfReplaceBlockedCleanHistoryCount = 0;
        LastGuardedDirtyTsdfBlockedHitsCount = 0;
        LastGuardedDirtyTsdfBlockedWeightCount = 0;
        LastGuardedDirtyTsdfBlockedValueCount = 0;
        LastDirtyTsdfQuarantineNewCount = 0;
        LastDirtyTsdfQuarantineRefreshCount = 0;
        LastDirtyTsdfActiveCount = 0;
        LastDirtyTsdfExpiredCount = 0;
    }

    private void BuildFallbackSurfaceSampleMesh(List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        if (!showFallbackSurfaceSamples)
            return;
        if (useChunkedSurfaceSamples)
        {
            BuildChunkedFallbackSurfaceSampleMesh(vertices, triangles, colors);
            return;
        }
        if (_fallbackSurfacePoints.Count <= 0)
            return;

        float halfSize = GetCoverageTileHalfSize();
        int count = Mathf.Min(_fallbackSurfacePoints.Count, Mathf.Max(1, maxFallbackSurfaceSamples));
        for (int i = 0; i < count; i++)
        {
            Vector3 normal = i < _fallbackSurfaceNormals.Count ? _fallbackSurfaceNormals[i] : Vector3.up;
            if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
                normal = Vector3.up;
            normal.Normalize();

            Vector3 tangent = Vector3.Cross(normal, Vector3.up);
            if (tangent.sqrMagnitude < 0.0001f)
                tangent = Vector3.Cross(normal, Vector3.right);
            tangent.Normalize();
            Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;
            Vector3 center = _fallbackSurfacePoints[i] + normal * 0.004f;

            int baseIndex = vertices.Count;
            vertices.Add(center - tangent * halfSize - bitangent * halfSize);
            vertices.Add(center + tangent * halfSize - bitangent * halfSize);
            vertices.Add(center + tangent * halfSize + bitangent * halfSize);
            vertices.Add(center - tangent * halfSize + bitangent * halfSize);
            AddVertexColors(colors, cleanCoverCellColor, 4);

            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 1);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex + 3);

            if (!doubleSidedTriangles)
                continue;

            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex + 1);
            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 3);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex);
        }
    }

    private void BuildChunkedFallbackSurfaceSampleMesh(List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        LastRenderedFallbackSampleCount = 0;
        LastPlanarCoverTileCount = 0;
        LastSkippedDirtyLightCoverCellCount = 0;
        LastSkippedUnstableLightCoverCellCount = 0;
        LastCleanLightCoverCellCount = 0;
        LastUnstableLightCoverCellCount = 0;
        LastRiskLightCoverCellCount = 0;
        if (_surfaceChunks.Count <= 0)
            return;

        if (renderStableCellsAsPlanarMeshCover && useConnectedPlanarCoverMesh)
        {
            BuildConnectedPlanarCoverageMesh(vertices, triangles, colors);
            return;
        }

        float halfSize = GetCoverageTileHalfSize();
        int renderBudget = GetLightCoverRenderBudget();
        int storedCellCount = GetFallbackSurfaceSampleCount();
        if (storedCellCount <= 0)
            return;

        float sampleStep = storedCellCount > renderBudget ? (float)storedCellCount / renderBudget : 1f;
        float nextEmitIndex = 0f;
        int visitedCellIndex = 0;

        for (int orderIndex = 0; orderIndex < _surfaceChunkOrder.Count && LastRenderedFallbackSampleCount < renderBudget; orderIndex++)
        {
            Vector3Int chunkKey = _surfaceChunkOrder[orderIndex];
            if (!_surfaceChunks.TryGetValue(chunkKey, out SurfaceChunk chunk))
                continue;

            foreach (KeyValuePair<Vector3Int, SurfaceCell> pair in chunk.Cells)
            {
                if (visitedCellIndex + 0.0001f < nextEmitIndex)
                {
                    visitedCellIndex++;
                    continue;
                }

                SurfaceCell cell = pair.Value;
                LightCoverCellRisk risk = ClassifyLightCoverCell(cell);
                if (!ShouldRenderLightCoverCell(risk))
                {
                    visitedCellIndex++;
                    continue;
                }
                CountRenderedLightCoverRisk(risk);

                if (renderStableCellsAsPlanarMeshCover)
                    AddPlanarCoverageTile(pair.Key, cell, vertices, triangles, colors, ColorForLightCoverRisk(risk));
                else
                    AddFallbackQuad(cell.Point, cell.Normal, halfSize, vertices, triangles, colors, ColorForLightCoverRisk(risk));
                LastRenderedFallbackSampleCount++;
                nextEmitIndex += sampleStep;
                visitedCellIndex++;
                if (LastRenderedFallbackSampleCount >= renderBudget)
                    break;
            }
        }
    }

    private void BuildConnectedPlanarCoverageMesh(List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        int renderBudget = GetLightCoverRenderBudget();
        int storedCellCount = GetFallbackSurfaceSampleCount();
        if (storedCellCount <= 0)
            return;

        Dictionary<CoverVertexKey, int> vertexLookup = new Dictionary<CoverVertexKey, int>(Mathf.Min(storedCellCount * 2, renderBudget * 2));
        float sampleStep = storedCellCount > renderBudget ? (float)storedCellCount / renderBudget : 1f;
        float nextEmitIndex = 0f;
        int visitedCellIndex = 0;

        for (int orderIndex = 0; orderIndex < _surfaceChunkOrder.Count && LastRenderedFallbackSampleCount < renderBudget; orderIndex++)
        {
            Vector3Int chunkKey = _surfaceChunkOrder[orderIndex];
            if (!_surfaceChunks.TryGetValue(chunkKey, out SurfaceChunk chunk))
                continue;

            foreach (KeyValuePair<Vector3Int, SurfaceCell> pair in chunk.Cells)
            {
                LightCoverCellRisk risk = ClassifyLightCoverCell(pair.Value);
                if (!ShouldRenderLightCoverCell(risk))
                {
                    visitedCellIndex++;
                    continue;
                }

                if (visitedCellIndex + 0.0001f < nextEmitIndex)
                {
                    visitedCellIndex++;
                    continue;
                }

                CountRenderedLightCoverRisk(risk);
                AddConnectedPlanarCoverageCell(pair.Value, vertexLookup, vertices, triangles, colors, ColorForLightCoverRisk(risk));
                LastRenderedFallbackSampleCount++;
                nextEmitIndex += sampleStep;
                visitedCellIndex++;
                if (LastRenderedFallbackSampleCount >= renderBudget)
                    break;
            }
        }
    }

    private int GetLightCoverRenderBudget()
    {
        return Mathf.Clamp(maxRenderedLightCoverTiles, 256, Mathf.Max(256, maxFallbackSurfaceSamples));
    }

    private LightCoverCellRisk ClassifyLightCoverCell(SurfaceCell cell)
    {
        if (renderOnlyStableLightCoverCells &&
            cell.Hits < Mathf.Max(1, minLightCoverCellHits))
        {
            return LightCoverCellRisk.Unstable;
        }

        if (renderOnlyCleanLightCoverCells && !LightCoverCellHasCleanTsdf(cell))
        {
            return LightCoverCellRisk.Risk;
        }

        return LightCoverCellRisk.Clean;
    }

    private bool ShouldRenderLightCoverCell(LightCoverCellRisk risk)
    {
        if (risk == LightCoverCellRisk.Unstable && !showUnstableLightCoverRiskCells)
        {
            LastSkippedUnstableLightCoverCellCount++;
            return false;
        }

        if (risk == LightCoverCellRisk.Risk && !showDirtyLightCoverRiskCells)
        {
            LastSkippedDirtyLightCoverCellCount++;
            return false;
        }

        return true;
    }

    private void CountRenderedLightCoverRisk(LightCoverCellRisk risk)
    {
        if (risk == LightCoverCellRisk.Risk)
            LastRiskLightCoverCellCount++;
        else if (risk == LightCoverCellRisk.Unstable)
            LastUnstableLightCoverCellCount++;
        else
            LastCleanLightCoverCellCount++;
    }

    private Color ColorForLightCoverRisk(LightCoverCellRisk risk)
    {
        if (risk == LightCoverCellRisk.Risk)
            return riskCoverCellColor;
        if (risk == LightCoverCellRisk.Unstable)
            return unstableCoverCellColor;
        return cleanCoverCellColor;
    }

    private bool LightCoverCellHasCleanTsdf(SurfaceCell cell)
    {
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(cell.Point, out int x, out int y, out int z))
            return false;

        int radius = Mathf.Clamp(cleanLightCoverCheckRadiusVoxels, 0, 3);
        int minWeight = Mathf.Max(1, minCleanLightCoverVoxelWeight);
        float maxAbsTsdf = Mathf.Clamp(maxCleanLightCoverAbsTsdf, 0.01f, 0.8f);
        int supportedNearSurface = 0;
        bool requireNearZero = requireNearZeroTsdfForLightCover;

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    int vy = y + dy;
                    int vz = z + dz;
                    if (vx < 0 || vy < 0 || vz < 0 || vx >= _dimX || vy >= _dimY || vz >= _dimZ)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (VoxelIsDirtyQuarantined(index))
                        return false;
                    if (VoxelHasPendingTsdfCorrection(index))
                        return false;
                    if (!requireNearZero)
                        continue;

                    if (_weights[index] < minWeight)
                        continue;

                    if (Mathf.Abs(_tsdf[index]) <= maxAbsTsdf)
                        supportedNearSurface++;
                }
            }
        }

        return !requireNearZero || supportedNearSurface > 0;
    }

    private void AddConnectedPlanarCoverageCell(
        SurfaceCell cell,
        Dictionary<CoverVertexKey, int> vertexLookup,
        List<Vector3> vertices,
        List<int> triangles,
        List<Color> colors,
        Color color)
    {
        Vector3 normal = cell.Normal;
        if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
            normal = Vector3.up;
        normal.Normalize();

        int axis = DominantAxis(normal);
        int sign = GetAxis(normal, axis) >= 0f ? 1 : -1;
        int uAxis;
        int vAxis;
        GetCoverUvAxes(axis, out uAxis, out vAxis);

        float pitch = Mathf.Max(0.005f, voxelSizeMeters * Mathf.Clamp(planarCoverTileScale, 0.6f, 1.8f));
        int plane = Mathf.RoundToInt(GetAxis(cell.Point, axis) / pitch);
        int centerU = Mathf.RoundToInt(GetAxis(cell.Point, uAxis) / pitch);
        int centerV = Mathf.RoundToInt(GetAxis(cell.Point, vAxis) / pitch);

        CoverVertexKey k00 = new CoverVertexKey(axis, sign, plane, centerU * 2 - 1, centerV * 2 - 1);
        CoverVertexKey k10 = new CoverVertexKey(axis, sign, plane, centerU * 2 + 1, centerV * 2 - 1);
        CoverVertexKey k11 = new CoverVertexKey(axis, sign, plane, centerU * 2 + 1, centerV * 2 + 1);
        CoverVertexKey k01 = new CoverVertexKey(axis, sign, plane, centerU * 2 - 1, centerV * 2 + 1);

        int i00 = GetOrAddCoverVertex(k00, pitch, vertexLookup, vertices, colors, color);
        int i10 = GetOrAddCoverVertex(k10, pitch, vertexLookup, vertices, colors, color);
        int i11 = GetOrAddCoverVertex(k11, pitch, vertexLookup, vertices, colors, color);
        int i01 = GetOrAddCoverVertex(k01, pitch, vertexLookup, vertices, colors, color);

        Vector3 a = vertices[i00];
        Vector3 b = vertices[i10];
        Vector3 c = vertices[i11];
        if (Vector3.Dot(Vector3.Cross(b - a, c - a), normal) >= 0f)
        {
            triangles.Add(i00);
            triangles.Add(i10);
            triangles.Add(i11);
            triangles.Add(i00);
            triangles.Add(i11);
            triangles.Add(i01);
            if (doubleSidedTriangles)
            {
                triangles.Add(i11);
                triangles.Add(i10);
                triangles.Add(i00);
                triangles.Add(i01);
                triangles.Add(i11);
                triangles.Add(i00);
            }
        }
        else
        {
            triangles.Add(i11);
            triangles.Add(i10);
            triangles.Add(i00);
            triangles.Add(i01);
            triangles.Add(i11);
            triangles.Add(i00);
            if (doubleSidedTriangles)
            {
                triangles.Add(i00);
                triangles.Add(i10);
                triangles.Add(i11);
                triangles.Add(i00);
                triangles.Add(i11);
                triangles.Add(i01);
            }
        }

        LastPlanarCoverTileCount++;
    }

    private int GetOrAddCoverVertex(
        CoverVertexKey key,
        float pitch,
        Dictionary<CoverVertexKey, int> vertexLookup,
        List<Vector3> vertices,
        List<Color> colors,
        Color color)
    {
        if (vertexLookup.TryGetValue(key, out int index))
        {
            PromoteCoverVertexColor(colors, index, color);
            return index;
        }

        int uAxis;
        int vAxis;
        GetCoverUvAxes(key.Axis, out uAxis, out vAxis);

        Vector3 position = Vector3.zero;
        SetAxis(ref position, key.Axis, key.Plane * pitch + key.Sign * 0.004f);
        SetAxis(ref position, uAxis, key.U * pitch * 0.5f);
        SetAxis(ref position, vAxis, key.V * pitch * 0.5f);

        index = vertices.Count;
        vertices.Add(position);
        colors.Add(color);
        vertexLookup.Add(key, index);
        return index;
    }

    private void AddPlanarCoverageTile(Vector3Int cellKey, SurfaceCell cell, List<Vector3> vertices, List<int> triangles, List<Color> colors, Color color)
    {
        Vector3 normal = cell.Normal;
        if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
            normal = Vector3.up;
        normal.Normalize();

        int axis = DominantAxis(normal);
        Vector3 tangent;
        Vector3 bitangent;
        if (axis == 0)
        {
            tangent = Vector3.forward;
            bitangent = Vector3.up;
        }
        else if (axis == 1)
        {
            tangent = Vector3.right;
            bitangent = Vector3.forward;
        }
        else
        {
            tangent = Vector3.right;
            bitangent = Vector3.up;
        }

        if (Vector3.Dot(Vector3.Cross(tangent, bitangent), normal) < 0f)
        {
            Vector3 swap = tangent;
            tangent = bitangent;
            bitangent = swap;
        }

        float halfSize = Mathf.Max(0.001f, voxelSizeMeters * Mathf.Clamp(planarCoverTileScale, 0.6f, 1.8f)) * 0.5f;
        Vector3 center = cell.Point + normal * 0.004f;
        int baseIndex = vertices.Count;
        vertices.Add(center - tangent * halfSize - bitangent * halfSize);
        vertices.Add(center + tangent * halfSize - bitangent * halfSize);
        vertices.Add(center + tangent * halfSize + bitangent * halfSize);
        vertices.Add(center - tangent * halfSize + bitangent * halfSize);
        AddVertexColors(colors, color, 4);

        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 1);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex + 3);

        if (doubleSidedTriangles)
        {
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex + 1);
            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 3);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex);
        }

        LastPlanarCoverTileCount++;
    }

    private void RebuildWireOverlay(List<Vector3> sourceVertices, List<int> sourceTriangles)
    {
        if (showMeshEdgeWireOverlay)
        {
            RebuildMeshEdgeWireOverlay(sourceVertices, sourceTriangles);
            return;
        }

        RebuildPlanarCoverWireOverlay();
    }

    private void RebuildMeshEdgeWireOverlay(List<Vector3> sourceVertices, List<int> sourceTriangles)
    {
        LastWireSegmentCount = 0;
        if (_wireMesh == null)
            return;

        _wireMesh.Clear();

        if (sourceVertices == null || sourceTriangles == null || sourceVertices.Count <= 0 || sourceTriangles.Count < 3)
        {
            if (_wireObject != null)
                _wireObject.SetActive(false);
            return;
        }

        List<int> indices = new List<int>(sourceTriangles.Count * 2);
        HashSet<ulong> edgeKeys = new HashSet<ulong>();
        for (int i = 0; i + 2 < sourceTriangles.Count; i += 3)
        {
            AddUniqueWireEdge(sourceTriangles[i], sourceTriangles[i + 1], sourceVertices.Count, edgeKeys, indices);
            AddUniqueWireEdge(sourceTriangles[i + 1], sourceTriangles[i + 2], sourceVertices.Count, edgeKeys, indices);
            AddUniqueWireEdge(sourceTriangles[i + 2], sourceTriangles[i], sourceVertices.Count, edgeKeys, indices);
        }

        _wireMesh.indexFormat = sourceVertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _wireMesh.SetVertices(sourceVertices);
        _wireMesh.SetIndices(indices, MeshTopology.Lines, 0, true);
        _wireMesh.RecalculateBounds();
        LastWireSegmentCount = indices.Count / 2;
        if (_wireObject != null)
            _wireObject.SetActive(indices.Count > 0);
    }

    private static void AddUniqueWireEdge(int a, int b, int vertexCount, HashSet<ulong> edgeKeys, List<int> indices)
    {
        if (a < 0 || b < 0 || a >= vertexCount || b >= vertexCount || a == b)
            return;

        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        ulong key = ((ulong)min << 32) | max;
        if (!edgeKeys.Add(key))
            return;

        indices.Add(a);
        indices.Add(b);
    }

    private void RebuildPlanarCoverWireOverlay()
    {
        LastWireSegmentCount = 0;
        if (_wireMesh == null)
            return;

        _wireMesh.Clear();

        if (!showPlanarCoverWireOverlay || !renderStableCellsAsPlanarMeshCover || !useChunkedSurfaceSamples || _surfaceChunks.Count <= 0)
        {
            if (_wireObject != null)
                _wireObject.SetActive(false);
            return;
        }

        int renderBudget = Mathf.Max(1, maxFallbackSurfaceSamples);
        int storedCellCount = GetFallbackSurfaceSampleCount();
        if (storedCellCount <= 0)
        {
            if (_wireObject != null)
                _wireObject.SetActive(false);
            return;
        }

        List<Vector3> vertices = new List<Vector3>(Mathf.Min(storedCellCount, renderBudget) * 4);
        List<int> indices = new List<int>(Mathf.Min(storedCellCount, renderBudget) * 8);
        float sampleStep = storedCellCount > renderBudget ? (float)storedCellCount / renderBudget : 1f;
        float nextEmitIndex = 0f;
        int visitedCellIndex = 0;
        int emitted = 0;

        for (int orderIndex = 0; orderIndex < _surfaceChunkOrder.Count && emitted < renderBudget; orderIndex++)
        {
            Vector3Int chunkKey = _surfaceChunkOrder[orderIndex];
            if (!_surfaceChunks.TryGetValue(chunkKey, out SurfaceChunk chunk))
                continue;

            foreach (KeyValuePair<Vector3Int, SurfaceCell> pair in chunk.Cells)
            {
                if (visitedCellIndex + 0.0001f < nextEmitIndex)
                {
                    visitedCellIndex++;
                    continue;
                }

                AddPlanarCoverageWireTile(pair.Value, vertices, indices);
                emitted++;
                nextEmitIndex += sampleStep;
                visitedCellIndex++;
                if (emitted >= renderBudget)
                    break;
            }
        }

        _wireMesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _wireMesh.SetVertices(vertices);
        _wireMesh.SetIndices(indices, MeshTopology.Lines, 0, true);
        _wireMesh.RecalculateBounds();
        if (_wireObject != null)
            _wireObject.SetActive(vertices.Count > 0);
    }

    private void AddPlanarCoverageWireTile(SurfaceCell cell, List<Vector3> vertices, List<int> indices)
    {
        Vector3 normal = cell.Normal;
        if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
            normal = Vector3.up;
        normal.Normalize();

        int axis = DominantAxis(normal);
        Vector3 tangent;
        Vector3 bitangent;
        if (axis == 0)
        {
            tangent = Vector3.forward;
            bitangent = Vector3.up;
        }
        else if (axis == 1)
        {
            tangent = Vector3.right;
            bitangent = Vector3.forward;
        }
        else
        {
            tangent = Vector3.right;
            bitangent = Vector3.up;
        }

        if (Vector3.Dot(Vector3.Cross(tangent, bitangent), normal) < 0f)
        {
            Vector3 swap = tangent;
            tangent = bitangent;
            bitangent = swap;
        }

        float halfSize = Mathf.Max(0.004f, voxelSizeMeters * Mathf.Clamp(wireTileScale, 0.6f, 1.5f)) * 0.5f;
        Vector3 center = cell.Point + normal * 0.007f;
        int start = vertices.Count;
        vertices.Add(center - tangent * halfSize - bitangent * halfSize);
        vertices.Add(center + tangent * halfSize - bitangent * halfSize);
        vertices.Add(center + tangent * halfSize + bitangent * halfSize);
        vertices.Add(center - tangent * halfSize + bitangent * halfSize);

        indices.Add(start);
        indices.Add(start + 1);
        indices.Add(start + 1);
        indices.Add(start + 2);
        indices.Add(start + 2);
        indices.Add(start + 3);
        indices.Add(start + 3);
        indices.Add(start);
        LastWireSegmentCount += 4;
    }

    private void AddFallbackQuad(Vector3 point, Vector3 normal, float halfSize, List<Vector3> vertices, List<int> triangles, List<Color> colors, Color color)
    {
        if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
            normal = Vector3.up;
        normal.Normalize();

        Vector3 tangent = Vector3.Cross(normal, Vector3.up);
        if (tangent.sqrMagnitude < 0.0001f)
            tangent = Vector3.Cross(normal, Vector3.right);
        tangent.Normalize();
        Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;
        Vector3 center = point + normal * 0.004f;

        int baseIndex = vertices.Count;
        vertices.Add(center - tangent * halfSize - bitangent * halfSize);
        vertices.Add(center + tangent * halfSize - bitangent * halfSize);
        vertices.Add(center + tangent * halfSize + bitangent * halfSize);
        vertices.Add(center - tangent * halfSize + bitangent * halfSize);
        AddVertexColors(colors, color, 4);

        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 1);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex + 3);

        if (!doubleSidedTriangles)
            return;

        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex + 1);
        triangles.Add(baseIndex);
        triangles.Add(baseIndex + 3);
        triangles.Add(baseIndex + 2);
        triangles.Add(baseIndex);
    }

    private static void AddVertexColors(List<Color> colors, Color color, int count)
    {
        if (colors == null)
            return;
        for (int i = 0; i < count; i++)
            colors.Add(color);
    }

    private void PromoteCoverVertexColor(List<Color> colors, int index, Color color)
    {
        if (colors == null || index < 0 || index >= colors.Count)
            return;
        if (LightCoverColorRank(color) > LightCoverColorRank(colors[index]))
            colors[index] = color;
    }

    private int LightCoverColorRank(Color color)
    {
        if (ApproximatelyColor(color, riskCoverCellColor))
            return 2;
        if (ApproximatelyColor(color, unstableCoverCellColor))
            return 1;
        return 0;
    }

    private static bool ApproximatelyColor(Color a, Color b)
    {
        return Mathf.Abs(a.r - b.r) < 0.001f &&
               Mathf.Abs(a.g - b.g) < 0.001f &&
               Mathf.Abs(a.b - b.b) < 0.001f &&
               Mathf.Abs(a.a - b.a) < 0.001f;
    }

    private float GetCoverageTileHalfSize()
    {
        float size = fallbackSurfaceSampleSizeMeters;
        if (displayFallbackAsCoverageTiles)
            size = Mathf.Max(size, Mathf.Max(0.005f, voxelSizeMeters) * Mathf.Clamp(coverageTileVoxelScale, 0.25f, 1.5f));
        return Mathf.Max(0.001f, size) * 0.5f;
    }

    private int GetFallbackSurfaceSampleCount()
    {
        if (!useChunkedSurfaceSamples)
            return _fallbackSurfacePoints.Count;

        int count = 0;
        foreach (KeyValuePair<Vector3Int, SurfaceChunk> pair in _surfaceChunks)
            count += pair.Value.Cells.Count;
        LastSurfaceChunkCount = _surfaceChunks.Count;
        return count;
    }

    private Vector3Int WorldToSurfaceChunkKey(Vector3 world)
    {
        float size = Mathf.Max(0.25f, surfaceChunkSizeMeters);
        return new Vector3Int(
            Mathf.FloorToInt(world.x / size),
            Mathf.FloorToInt(world.y / size),
            Mathf.FloorToInt(world.z / size));
    }

    private Vector3Int WorldToSurfaceLocalCell(Vector3 world, Vector3 normal)
    {
        float cell = Mathf.Max(0.005f, voxelSizeMeters);
        Vector3 keyedWorld = world;
        bool useSideAwareKey = sideAwareSurfaceCells && useSideAwareLightCoverKeys && Finite(normal) && normal.sqrMagnitude > 0.0001f;
        if (useSideAwareKey)
            keyedWorld += normal.normalized * cell * Mathf.Clamp01(surfaceSideOffsetVoxelScale);
        Vector3Int cellKey = new Vector3Int(
            Mathf.FloorToInt(keyedWorld.x / cell),
            Mathf.FloorToInt(keyedWorld.y / cell),
            Mathf.FloorToInt(keyedWorld.z / cell));
        if (useSideAwareKey)
            cellKey += SurfaceNormalBin(normal) * 1000000;
        return cellKey;
    }

    private Vector3Int SurfaceNormalBin(Vector3 normal)
    {
        if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
            return Vector3Int.zero;
        normal.Normalize();
        float ax = Mathf.Abs(normal.x);
        float ay = Mathf.Abs(normal.y);
        float az = Mathf.Abs(normal.z);
        if (ax >= ay && ax >= az)
            return normal.x >= 0f ? new Vector3Int(1, 0, 0) : new Vector3Int(-1, 0, 0);
        if (ay >= ax && ay >= az)
            return normal.y >= 0f ? new Vector3Int(0, 1, 0) : new Vector3Int(0, -1, 0);
        return normal.z >= 0f ? new Vector3Int(0, 0, 1) : new Vector3Int(0, 0, -1);
    }

    private int DominantAxis(Vector3 normal)
    {
        float ax = Mathf.Abs(normal.x);
        float ay = Mathf.Abs(normal.y);
        float az = Mathf.Abs(normal.z);
        if (ax >= ay && ax >= az)
            return 0;
        if (ay >= ax && ay >= az)
            return 1;
        return 2;
    }

    private void GetCoverUvAxes(int normalAxis, out int uAxis, out int vAxis)
    {
        if (normalAxis == 0)
        {
            uAxis = 2;
            vAxis = 1;
        }
        else if (normalAxis == 1)
        {
            uAxis = 0;
            vAxis = 2;
        }
        else
        {
            uAxis = 0;
            vAxis = 1;
        }
    }

    private float GetAxis(Vector3 value, int axis)
    {
        if (axis == 0)
            return value.x;
        if (axis == 1)
            return value.y;
        return value.z;
    }

    private void SetAxis(ref Vector3 value, int axis, float component)
    {
        if (axis == 0)
            value.x = component;
        else if (axis == 1)
            value.y = component;
        else
            value.z = component;
    }

    private void EnsureVolumeInitialized()
    {
        if (_volumeInitialized && _tsdf != null && _weights != null)
            return;

        float voxel = Mathf.Max(0.02f, voxelSizeMeters);
        _dimX = Mathf.Max(4, Mathf.CeilToInt(Mathf.Max(voxel, volumeSizeMeters.x) / voxel) + 1);
        _dimY = Mathf.Max(4, Mathf.CeilToInt(Mathf.Max(voxel, volumeSizeMeters.y) / voxel) + 1);
        _dimZ = Mathf.Max(4, Mathf.CeilToInt(Mathf.Max(voxel, volumeSizeMeters.z) / voxel) + 1);

        Vector3 center = volumeAnchor != null ? volumeAnchor.position : GetCameraPosition();
        _volumeOriginWorld = center - new Vector3((_dimX - 1) * voxel, (_dimY - 1) * voxel, (_dimZ - 1) * voxel) * 0.5f;

        int count = _dimX * _dimY * _dimZ;
        _voxelAuditLifecycle.Clear();
        _voxelWriteProvenance.Clear();
        _voxelWriteSequence = 0;
        _captureAuditVoxels.Clear();
        _tsdf = new float[count];
        _weights = new byte[count];
        _atomicProvisionalBandTsdf = new float[count];
        _atomicProvisionalBandWeight = new byte[count];
        _atomicProvisionalBandRetiredLastFrame = new int[count];
        _atomicProvisionalBandRetiredSign = new byte[count];
        _surfaceBandVisitFlags = new byte[count];
        _surfaceBandAcceptSignLostFlags = new byte[count];
        _acceptedPositiveLastSequence = new int[count];
        _acceptedNegativeLastSequence = new int[count];
        _acceptedPositiveRetainedLastSequence = new int[count];
        _acceptedNegativeRetainedLastSequence = new int[count];
        _acceptedWriteSequence = 0;
        _acceptedSignRecoveryPositiveFrames = new byte[count];
        _acceptedSignRecoveryNegativeFrames = new byte[count];
        _acceptedSignRecoveryPositiveLastFrame = new int[count];
        _acceptedSignRecoveryNegativeLastFrame = new int[count];
        _surfaceObservationState = new byte[count];
        _surfaceObservationLastFrame = new int[count];
        _clearedVoxelLastFrame = new int[count];
        _pendingTsdf = new float[count];
        _pendingTsdfHits = new byte[count];
        _pendingTsdfLastFrame = new int[count];
        _correctionTsdf = new float[count];
        _correctionTsdfHits = new byte[count];
        _correctionTsdfLastFrame = new int[count];
        _dirtyTsdfLastFrame = new int[count];
        _provisionalTsdf = new byte[count];
        _provisionalTsdfLastFrame = new int[count];
        _provisionalTsdfHits = new byte[count];
        _oldCleanConflictHits = new byte[count];
        _oldCleanConflictLastFrame = new int[count];
        _freeSpaceEvidenceHits = new byte[count];
        _freeSpaceEvidenceLastFrame = new int[count];
        for (int i = 0; i < count; i++)
        {
            _tsdf[i] = 1f;
            _atomicProvisionalBandTsdf[i] = 1f;
            _atomicProvisionalBandRetiredLastFrame[i] = int.MinValue;
            _acceptedSignRecoveryPositiveLastFrame[i] = int.MinValue;
            _acceptedSignRecoveryNegativeLastFrame[i] = int.MinValue;
            _acceptedPositiveLastSequence[i] = int.MinValue;
            _acceptedNegativeLastSequence[i] = int.MinValue;
            _acceptedPositiveRetainedLastSequence[i] = int.MinValue;
            _acceptedNegativeRetainedLastSequence[i] = int.MinValue;
            _surfaceObservationLastFrame[i] = int.MinValue;
            _clearedVoxelLastFrame[i] = int.MinValue;
            _pendingTsdfLastFrame[i] = int.MinValue;
            _correctionTsdfLastFrame[i] = int.MinValue;
            _dirtyTsdfLastFrame[i] = int.MinValue;
            _provisionalTsdfLastFrame[i] = int.MinValue;
            _oldCleanConflictLastFrame[i] = int.MinValue;
            _freeSpaceEvidenceLastFrame[i] = int.MinValue;
        }

        _hardAuditExpectedTsdf = (float[])_tsdf.Clone();
        _hardAuditExpectedWeights = (byte[])_weights.Clone();
        ResetHardTsdfWriteAuditCounters();

        _volumeInitialized = true;
    }

    private bool TryBuildCellVertex(
        int x,
        int y,
        int z,
        Vector3[] cornerPositions,
        float[] cornerValues,
        byte[] cornerWeights,
        out Vector3 vertex)
    {
        if (UseLegacyMeshExtraction)
            return TryBuildLegacyCellVertex(x, y, z, cornerPositions, cornerValues, cornerWeights, out vertex);

        vertex = Vector3.zero;
        bool hasPositive = false;
        bool hasNegative = false;
        bool hasNearZero = false;
        int supportedCorners = 0;
        int nearZeroCorners = 0;

        for (int i = 0; i < 8; i++)
        {
            int vx = x + CornerOffsetX[i];
            int vy = y + CornerOffsetY[i];
            int vz = z + CornerOffsetZ[i];
            int index = Index(vx, vy, vz);
            cornerPositions[i] = VoxelCenter(vx, vy, vz);
            cornerValues[i] = _tsdf[index];
            cornerWeights[i] = IsCleanMeshTsdfVoxel(index) ? _weights[index] : (byte)0;

            if (cornerWeights[i] >= minSurfaceCornerWeight)
            {
                supportedCorners++;
                if (cornerValues[i] < 0f) hasNegative = true;
                if (cornerValues[i] > 0f) hasPositive = true;
                if (Mathf.Abs(cornerValues[i]) <= nearZeroSurfaceThreshold)
                {
                    hasNearZero = true;
                    nearZeroCorners++;
                }
            }
        }

        bool hasStrictSurface = hasPositive && hasNegative;
        bool hasRelaxedSurface =
            AllowRelaxedMeshExtraction &&
            hasNearZero &&
            supportedCorners >= Mathf.Max(2, minRelaxedSurfaceSupportedCorners) &&
            nearZeroCorners >= Mathf.Max(1, minRelaxedNearZeroCorners);
        if (hasRelaxedSurface &&
            !hasStrictSurface &&
            requireRelaxedSurfaceNearStrictCell &&
            !HasStrictSurfaceNeighborCell(x, y, z, Mathf.Max(1, relaxedStrictNeighborRadiusCells)))
        {
            hasRelaxedSurface = false;
        }

        if (hasStrictSurface)
            LastStrictSurfaceCellCandidateCount++;
        else if (hasRelaxedSurface)
            LastRelaxedSurfaceCellCandidateCount++;

        if ((hasStrictSurface && supportedCorners < Mathf.Max(2, minStrictSurfaceSupportedCorners)) ||
            (!hasStrictSurface && !hasRelaxedSurface))
        {
            LastRejectedNoSurfaceCellCount++;
            return false;
        }

        if (suppressExtractionNearPendingTsdfCorrection &&
            CellTouchesPendingTsdfCorrection(x, y, z, pendingCorrectionExtractionRadiusCells))
        {
            LastRejectedCorrectionPendingCellCount++;
            return false;
        }

        if (suppressExtractionNearDirtyTsdfQuarantine &&
            CellTouchesDirtyTsdfQuarantine(x, y, z, dirtyTsdfExtractionRadiusCells))
        {
            LastRejectedDirtyQuarantineCellCount++;
            return false;
        }

        if (!PassMainShellPromotionGate(x, y, z, cornerWeights, hasStrictSurface))
        {
            LastRejectedMainGateCellCount++;
            return false;
        }

        Vector3 sum = Vector3.zero;
        int count = 0;
        int strictEdgeCount = 0;
        int relaxedEdgeCount = 0;
        for (int edge = 0; edge < 12; edge++)
        {
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            if (cornerWeights[a] < minSurfaceCornerWeight || cornerWeights[b] < minSurfaceCornerWeight)
                continue;

            float va = cornerValues[a];
            float vb = cornerValues[b];
            bool strictEdge = (va < 0f && vb > 0f) || (va > 0f && vb < 0f);
            bool relaxedEdge = hasRelaxedSurface && IsRelaxedSurfaceEdge(va, vb);
            if ((!strictEdge && !relaxedEdge) || Mathf.Approximately(va, vb))
                continue;

            if (strictEdge)
                strictEdgeCount++;
            else
                relaxedEdgeCount++;

            float t = strictEdge ? Mathf.Clamp01(va / (va - vb)) : (Mathf.Abs(va) <= Mathf.Abs(vb) ? 0f : 1f);
            sum += Vector3.Lerp(cornerPositions[a], cornerPositions[b], t);
            count++;
        }

        if (count <= 0)
        {
            LastRejectedNoEdgeCellCount++;
            return false;
        }
        if (hasStrictSurface && strictEdgeCount < Mathf.Max(1, minStrictSurfaceEdgeHits))
        {
            LastRejectedEdgeCountCellCount++;
            return false;
        }
        if (!hasStrictSurface && relaxedEdgeCount < Mathf.Max(1, minRelaxedSurfaceEdgeHits))
        {
            LastRejectedEdgeCountCellCount++;
            return false;
        }

        vertex = sum / count;
        return Finite(vertex);
    }

    private void RepairGuardedHoleSides()
    {
        if (!enableGuardedHoleSideRepair || _tsdf == null || _weights == null)
            return;

        int budget = Mathf.Max(1, maxHoleSideRepairsPerRebuild);
        Dictionary<int, HoleSideRepairProposal> proposals = new Dictionary<int, HoleSideRepairProposal>(budget);
        HashSet<int> conflicted = new HashSet<int>();
        float maxAbs = Mathf.Max(0.01f, maxHoleSideRepairNeighborAbsTsdf);
        int requiredBridgingCells = Mathf.Clamp(minHoleSideRepairBridgingCells, 1, 8);
        int requiredOpposite = Mathf.Clamp(minHoleSideRepairOppositeFaceSupport, 1, 6);
        int requiredExisting = Mathf.Clamp(minHoleSideRepairExistingFaceSupport, 1, 6);

        for (int z = 0; z < _dimZ - 1 && proposals.Count < budget; z++)
        for (int y = 0; y < _dimY - 1 && proposals.Count < budget; y++)
        for (int x = 0; x < _dimX - 1 && proposals.Count < budget; x++)
        {
            int positive = 0;
            int negative = 0;
            int empty = 0;
            for (int corner = 0; corner < 8; corner++)
            {
                int index = Index(x + ((corner & 1) != 0 ? 1 : 0), y + ((corner & 2) != 0 ? 1 : 0), z + ((corner & 4) != 0 ? 1 : 0));
                if (_weights[index] <= 0)
                    empty++;
                else if (_tsdf[index] >= 0f)
                    positive++;
                else
                    negative++;
            }
            if (empty <= 0 || (positive > 0 && negative > 0) || (positive <= 0 && negative <= 0))
                continue;

            int desiredSign = positive > 0 ? -1 : 1;
            for (int corner = 0; corner < 8 && proposals.Count < budget; corner++)
            {
                int index = Index(x + ((corner & 1) != 0 ? 1 : 0), y + ((corner & 2) != 0 ? 1 : 0), z + ((corner & 4) != 0 ? 1 : 0));
                if (_weights[index] > 0 || conflicted.Contains(index))
                    continue;
                LastHoleSideRepairCandidateCount++;
                IndexToVoxel(index, out int vx, out int vy, out int vz);
                if (NeighborhoodTouchesDirtyOrPending(vx, vy, vz, 1))
                {
                    LastHoleSideRepairBlockedDirtyCount++;
                    continue;
                }

                List<HoleSideRepairProposal> candidateBand = null;
                bool primaryPlaneValid = usePrimaryPlaneHoleRepair &&
                    TryBuildPrimaryPlaneHoleBand(index, desiredSign, out candidateBand);
                if (usePrimaryPlaneHoleRepair && !primaryPlaneValid)
                {
                    LastHoleSideRepairBlockedPlaneCount++;
                    continue;
                }

                bool legacyTopologyValid = HasHoleRepairCellTopology(
                        vx,
                        vy,
                        vz,
                        desiredSign,
                        maxAbs,
                        requiredBridgingCells,
                        requiredOpposite,
                        requiredExisting);
                if (!legacyTopologyValid && !primaryPlaneValid)
                {
                    LastHoleSideRepairBlockedSupportCount++;
                    continue;
                }

                if (candidateBand == null)
                {
                    candidateBand = new List<HoleSideRepairProposal>(1)
                    {
                        new HoleSideRepairProposal
                        {
                            Index = index,
                            Tsdf = desiredSign * Mathf.Max(0.01f, holeSideRepairAbsTsdf)
                        }
                    };
                }

                bool bandConflict = false;
                for (int bandIndex = 0; bandIndex < candidateBand.Count; bandIndex++)
                {
                    HoleSideRepairProposal candidate = candidateBand[bandIndex];
                    if (proposals.TryGetValue(candidate.Index, out HoleSideRepairProposal prior) &&
                        Mathf.Sign(prior.Tsdf) != Mathf.Sign(candidate.Tsdf))
                    {
                        proposals.Remove(candidate.Index);
                        conflicted.Add(candidate.Index);
                        bandConflict = true;
                    }
                }
                if (bandConflict)
                {
                    LastHoleSideRepairBlockedSupportCount++;
                    continue;
                }
                for (int bandIndex = 0; bandIndex < candidateBand.Count && proposals.Count < budget; bandIndex++)
                {
                    HoleSideRepairProposal candidate = candidateBand[bandIndex];
                    if (!conflicted.Contains(candidate.Index) && _weights[candidate.Index] <= 0)
                        proposals[candidate.Index] = candidate;
                }
            }
        }

        foreach (HoleSideRepairProposal proposal in proposals.Values)
        {
            int index = proposal.Index;
            if (_weights[index] > 0)
                continue;
            if (blockHoleRepairThatCreatesMultipleZeroCrossings &&
                WouldHoleRepairCreateMultipleZeroCrossings(index, proposal.Tsdf, proposals))
            {
                LastHoleSideRepairBlockedMultiZeroCount++;
                continue;
            }
            float oldTsdf = _tsdf[index];
            int oldWeight = _weights[index];
            _tsdf[index] = proposal.Tsdf;
            _weights[index] = 1;
            string operation = usePrimaryPlaneHoleRepair ? "primary_plane_hole_band" : "hole_side_repair";
            RecordContributionLedger(index, operation, oldTsdf, oldWeight, proposal.Tsdf, 1f, proposal.Tsdf, 1);
            LastHoleSideRepairAppliedCount++;
            if (usePrimaryPlaneHoleRepair)
            {
                LastHoleSideRepairPlaneBandVoxelCount++;
                if (!_primaryPlaneHoleConfirmations.ContainsKey(index))
                    _primaryPlaneHoleConfirmations[index] = 1;
                _primaryPlaneHoleLastValidationFrame[index] = LastRawFrameIndex;
                _primaryPlaneHoleAcceptEvidence[index] = 0;
                _primaryPlaneHoleMinPlaneDistanceVoxels.Remove(index);
            }
        }
    }

    private void ConfirmPrimaryPlaneHoleBands()
    {
        if (!usePrimaryPlaneHoleRepair || _provisionalTsdf == null || _weights == null)
            return;

        List<int> candidates = new List<int>();
        foreach (KeyValuePair<int, VoxelWriteProvenance> pair in _voxelWriteProvenance)
        {
            int index = pair.Key;
            if (pair.Value.LastOperation == "primary_plane_hole_band" &&
                index >= 0 && index < _weights.Length &&
                HasProvisionalTsdfMarker(index))
            {
                candidates.Add(index);
            }
        }

        for (int i = 0; i < candidates.Count; i++)
        {
            int index = candidates[i];
            int writtenFrame = _provisionalTsdfLastFrame != null && index < _provisionalTsdfLastFrame.Length
                ? _provisionalTsdfLastFrame[index]
                : LastRawFrameIndex;
            if (LastRawFrameIndex - writtenFrame < Mathf.Max(1, primaryPlaneHoleMaxAgeFrames))
                continue;

            float oldTsdf = _tsdf[index];
            int oldWeight = _weights[index];
            _tsdf[index] = 1f;
            _weights[index] = 0;
            RecordContributionLedger(index, "primary_plane_hole_retire", oldTsdf, oldWeight, 1f, 0f, 1f, 0);
            _primaryPlaneHoleConfirmations.Remove(index);
            _primaryPlaneHoleLastValidationFrame.Remove(index);
            byte evidence = _primaryPlaneHoleAcceptEvidence.TryGetValue(index, out byte recordedEvidence)
                ? recordedEvidence
                : (byte)0;
            _primaryPlaneHoleAcceptEvidence.Remove(index);
            float closestPlaneDistanceVoxels = _primaryPlaneHoleMinPlaneDistanceVoxels.TryGetValue(index, out float recordedDistance)
                ? recordedDistance
                : float.PositiveInfinity;
            _primaryPlaneHoleMinPlaneDistanceVoxels.Remove(index);
            if ((evidence & 1) == 0)
                LastHoleSideRepairRetiredNoNearAcceptCount++;
            else if ((evidence & 2) == 0)
                LastHoleSideRepairRetiredSignMismatchCount++;
            else if ((evidence & 4) == 0)
            {
                LastHoleSideRepairRetiredPlaneMismatchCount++;
                RecordPrimaryPlaneHoleDistanceBin(closestPlaneDistanceVoxels);
            }
            else if ((evidence & 8) == 0)
                LastHoleSideRepairRetiredTsdfDeltaCount++;
            else
                LastHoleSideRepairRetiredExpiredAfterPassCount++;
            LastHoleSideRepairPlaneRetiredCount++;
            LastProvisionalTsdfRetiredCount++;
            LastProvisionalTsdfRetiredExpiredCount++;
        }
    }

    private void RecordPrimaryPlaneHoleDistanceBin(float distanceVoxels)
    {
        if (distanceVoxels < 0.5f)
            LastHoleSideRepairPlaneDistance0ToHalfCount++;
        else if (distanceVoxels < 0.75f)
            LastHoleSideRepairPlaneDistanceHalfTo075Count++;
        else if (distanceVoxels < 1f)
            LastHoleSideRepairPlaneDistance075To1Count++;
        else if (distanceVoxels < 1.5f)
            LastHoleSideRepairPlaneDistance1To15Count++;
        else if (distanceVoxels < 2f)
            LastHoleSideRepairPlaneDistance15To2Count++;
        else
            LastHoleSideRepairPlaneDistanceOver2Count++;
    }

    private bool TryBuildPrimaryPlaneHoleBand(
        int candidateIndex,
        int desiredSign,
        out List<HoleSideRepairProposal> band)
    {
        band = null;
        IndexToVoxel(candidateIndex, out int cx, out int cy, out int cz);
        int radius = Mathf.Clamp(primaryPlaneHoleAnchorRadiusVoxels, 1, 4);
        int requiredAnchors = Mathf.Clamp(minPrimaryPlaneHoleAnchors, 3, 16);
        List<Vector3> points = new List<Vector3>(requiredAnchors * 2);
        List<Vector3> normals = new List<Vector3>(requiredAnchors * 2);
        HashSet<long> observations = new HashSet<long>();
        Vector3 referenceNormal = Vector3.zero;

        for (int dz = -radius; dz <= radius; dz++)
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = cx + dx;
            int y = cy + dy;
            int z = cz + dz;
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;
            int index = Index(x, y, z);
            if (_weights[index] < minSurfaceCornerWeight ||
                VoxelIsDirtyQuarantined(index) ||
                VoxelHasPendingTsdfCorrection(index) ||
                HasProvisionalTsdfMarker(index) ||
                !_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) ||
                !IsPrimaryPlaneHoleAnchorOperation(provenance.LastOperation) ||
                !Finite(provenance.SurfacePoint) ||
                !Finite(provenance.SurfaceNormal) ||
                provenance.SurfaceNormal.sqrMagnitude < 0.25f)
            {
                continue;
            }

            long observation = ((long)provenance.Capture << 32) ^ (uint)provenance.Sample;
            if (!observations.Add(observation))
                continue;
            Vector3 normal = provenance.SurfaceNormal.normalized;
            if (referenceNormal == Vector3.zero)
                referenceNormal = normal;
            if (Vector3.Dot(normal, referenceNormal) < 0f)
                normal = -normal;
            if (Vector3.Dot(normal, referenceNormal) < Mathf.Clamp01(minPrimaryPlaneHoleNormalDot))
                continue;
            points.Add(provenance.SurfacePoint);
            normals.Add(normal);
        }

        if (points.Count < requiredAnchors)
            return false;

        Vector3 planePoint = Vector3.zero;
        Vector3 planeNormal = Vector3.zero;
        for (int i = 0; i < points.Count; i++)
        {
            planePoint += points[i];
            planeNormal += normals[i];
        }
        planePoint /= points.Count;
        if (planeNormal.sqrMagnitude < 0.25f)
            return false;
        planeNormal.Normalize();

        float maxResidual = 0f;
        for (int i = 0; i < points.Count; i++)
            maxResidual = Mathf.Max(maxResidual, Mathf.Abs(Vector3.Dot(points[i] - planePoint, planeNormal)));
        if (maxResidual > Mathf.Max(0.001f, maxPrimaryPlaneHoleResidualMeters))
            return false;

        Vector3 candidateCenter = VoxelCenter(cx, cy, cz);
        float candidateDistance = Vector3.Dot(candidateCenter - planePoint, planeNormal);
        if (Mathf.Abs(candidateDistance) > Mathf.Max(truncationMeters, maxPrimaryPlaneHoleResidualMeters))
            return false;
        if (!Mathf.Approximately(candidateDistance, 0f) && Mathf.Sign(candidateDistance) != desiredSign)
            planeNormal = -planeNormal;

        int axis = 0;
        Vector3 absNormal = new Vector3(Mathf.Abs(planeNormal.x), Mathf.Abs(planeNormal.y), Mathf.Abs(planeNormal.z));
        if (absNormal.y > absNormal.x && absNormal.y >= absNormal.z)
            axis = 1;
        else if (absNormal.z > absNormal.x && absNormal.z > absNormal.y)
            axis = 2;

        int bandRadius = Mathf.Clamp(primaryPlaneHoleBandRadiusVoxels, 0, 2);
        band = new List<HoleSideRepairProposal>(bandRadius * 2 + 1);
        for (int offset = -bandRadius; offset <= bandRadius; offset++)
        {
            int x = cx + (axis == 0 ? offset : 0);
            int y = cy + (axis == 1 ? offset : 0);
            int z = cz + (axis == 2 ? offset : 0);
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;
            int index = Index(x, y, z);
            if ((_weights[index] > 0 && index != candidateIndex) || NeighborhoodTouchesDirtyOrPending(x, y, z, 1))
                continue;
            float signedDistance = Vector3.Dot(VoxelCenter(x, y, z) - planePoint, planeNormal);
            float value = Mathf.Clamp(signedDistance / Mathf.Max(0.0001f, truncationMeters), -1f, 1f);
            if (index == candidateIndex && Mathf.Abs(value) < 0.01f)
                value = desiredSign * Mathf.Max(0.01f, holeSideRepairAbsTsdf);
            if (!Mathf.Approximately(value, 0f))
                band.Add(new HoleSideRepairProposal { Index = index, Tsdf = value });
        }
        return band.Count > 0;
    }

    private static bool IsPrimaryPlaneHoleAnchorOperation(string operation)
    {
        return operation == "atomic_accept_band" ||
               operation == "integrate" ||
               operation == "strong_current_promote" ||
               operation == "guarded_replace" ||
               operation == "replace" ||
               operation == "v08_direct_diag";
    }

    private bool WouldHoleRepairCreateMultipleZeroCrossings(
        int candidateIndex,
        float candidateTsdf,
        Dictionary<int, HoleSideRepairProposal> proposals)
    {
        IndexToVoxel(candidateIndex, out int cx, out int cy, out int cz);
        int radius = Mathf.Clamp(holeRepairZeroCrossingCheckRadiusVoxels, 2, 8);
        for (int axis = 0; axis < 3; axis++)
        {
            int transitions = 0;
            int previousSign = 0;
            for (int offset = -radius; offset <= radius; offset++)
            {
                int x = cx + (axis == 0 ? offset : 0);
                int y = cy + (axis == 1 ? offset : 0);
                int z = cz + (axis == 2 ? offset : 0);
                if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                    continue;

                int index = Index(x, y, z);
                float value;
                bool supported;
                if (index == candidateIndex)
                {
                    value = candidateTsdf;
                    supported = true;
                }
                else if (proposals != null && proposals.TryGetValue(index, out HoleSideRepairProposal simulated))
                {
                    value = simulated.Tsdf;
                    supported = true;
                }
                else
                {
                    value = _tsdf[index];
                    supported = _weights[index] > 0;
                }

                if (!supported || Mathf.Approximately(value, 0f))
                    continue;
                int sign = value > 0f ? 1 : -1;
                if (previousSign != 0 && sign != previousSign)
                    transitions++;
                previousSign = sign;
            }

            if (transitions > 1)
                return true;
        }
        return false;
    }

    private bool HasHoleRepairCellTopology(
        int x,
        int y,
        int z,
        int desiredSign,
        float maxAbs,
        int requiredBridgingCells,
        int requiredOpposite,
        int requiredExisting)
    {
        int bridgingCells = 0;
        int desiredSupport = 0;
        int existingSupport = 0;
        for (int oz = -1; oz <= 0; oz++)
        for (int oy = -1; oy <= 0; oy++)
        for (int ox = -1; ox <= 0; ox++)
        {
            int cellX = x + ox;
            int cellY = y + oy;
            int cellZ = z + oz;
            if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                cellX >= _dimX - 1 || cellY >= _dimY - 1 || cellZ >= _dimZ - 1)
                continue;

            int cellPositive = 0;
            int cellNegative = 0;
            for (int corner = 0; corner < 8; corner++)
            {
                int vx = cellX + ((corner & 1) != 0 ? 1 : 0);
                int vy = cellY + ((corner & 2) != 0 ? 1 : 0);
                int vz = cellZ + ((corner & 4) != 0 ? 1 : 0);
                if (vx == x && vy == y && vz == z)
                    continue;
                int neighbor = Index(vx, vy, vz);
                if (_weights[neighbor] < minSurfaceCornerWeight || Mathf.Abs(_tsdf[neighbor]) > maxAbs ||
                    VoxelIsDirtyQuarantined(neighbor) || VoxelHasPendingTsdfCorrection(neighbor) ||
                    !IsAllowedContinuitySupportSource(neighbor))
                {
                    continue;
                }
                if (_tsdf[neighbor] >= 0f)
                    cellPositive++;
                else
                    cellNegative++;
            }

            if (cellPositive > 0 && cellNegative > 0)
                bridgingCells++;
            if (desiredSign > 0)
            {
                desiredSupport += cellPositive;
                existingSupport += cellNegative;
            }
            else
            {
                desiredSupport += cellNegative;
                existingSupport += cellPositive;
            }
        }

        return bridgingCells >= requiredBridgingCells &&
               desiredSupport >= requiredOpposite &&
               existingSupport >= requiredExisting;
    }

    private void FillCleanTsdfContinuityGaps()
    {
        if (!fillCleanTsdfContinuityGaps || _tsdf == null || _weights == null)
            return;

        int total = _dimX * _dimY * _dimZ;
        if (total <= 0 || _tsdf.Length < total || _weights.Length < total)
            return;

        int radius = Mathf.Clamp(tsdfContinuityFillRadiusVoxels, 1, 2);
        int requiredSupport = Mathf.Clamp(minTsdfContinuityNeighborVoxels, 3, 26);
        int requiredFaceSupport = Mathf.Clamp(minTsdfContinuityFaceNeighborVoxels, 1, 6);
        int requiredSameSignSupport = Mathf.Clamp(minTsdfContinuitySameSignNeighborVoxels, requiredSupport, 26);
        int requiredBoundaryNoTsdfSupport = Mathf.Clamp(minBoundaryNoTsdfNeighborVoxels, requiredSupport, requiredSameSignSupport);
        int requiredBoundaryNoTsdfFaceSupport = Mathf.Clamp(minBoundaryNoTsdfFaceNeighborVoxels, requiredFaceSupport, 6);
        int maxBoundaryNoTsdfFill = Mathf.Max(0, maxBoundaryNoTsdfFillVoxelsPerRebuild);
        int maxFill = Mathf.Max(0, maxTsdfContinuityFillVoxelsPerRebuild);
        float maxAbs = Mathf.Clamp(maxTsdfContinuityNeighborAbsValue, 0.01f, 0.35f);
        float fillAbs = Mathf.Clamp(continuityFillAbsTsdf, 0.005f, maxAbs);
        int fillWeight = Mathf.Clamp(Mathf.Max(continuityFillWeight, minSurfaceCornerWeight), 1, 255);

        for (int z = 1; z < _dimZ - 1; z++)
        {
            for (int y = 1; y < _dimY - 1; y++)
            {
                for (int x = 1; x < _dimX - 1; x++)
                {
                    if (maxFill > 0 && LastTsdfContinuityFilledCount >= maxFill)
                    {
                        LastTsdfContinuityBlockedBudgetCount++;
                        return;
                    }

                    int index = Index(x, y, z);
                    if (_weights[index] > 0)
                        continue;

                    LastTsdfContinuityCandidateCount++;
                    if (NeighborhoodTouchesDirtyOrPending(x, y, z, radius))
                    {
                        LastTsdfContinuityBlockedDirtyPendingCount++;
                        continue;
                    }
                    if (rejectContinuityFillFromUnsettledNeighbors &&
                        NeighborhoodHasUnsettledContinuitySource(x, y, z, radius))
                    {
                        LastTsdfContinuityBlockedUnsettledNeighborCount++;
                        continue;
                    }

                    int support = 0;
                    int faceSupport = 0;
                    int cleanAnchorSupport = 0;
                    int cleanAnchorFaceSupport = 0;
                    int stableProvisionalSupport = 0;
                    int stableProvisionalFaceSupport = 0;
                    int nearSurfaceFaceSupport = 0;
                    int faceAxisMask = 0;
                    int positive = 0;
                    int negative = 0;
                    float weightedTsdf = 0f;
                    int weightSum = 0;
                    int bNoProvPresent = 0;
                    int bNoProvFacePresent = 0;
                    int bNoProvAccepted = 0;
                    int bNoProvFaceAccepted = 0;
                    int bNoProvBlockWeight = 0;
                    int bNoProvBlockDirtyPending = 0;
                    int bNoProvBlockTsdf = 0;
                    int bNoProvBlockResidual = 0;
                    int bNoProvBlockProvenance = 0;
                    int bNoProvBlockNotFace = 0;
                    int bNoFaceAcceptClean = 0;
                    int bNoFaceAcceptProvisional = 0;
                    int bNoFaceBlockWeight = 0;
                    int bNoFaceBlockDirtyPending = 0;
                    int bNoFaceBlockProvenance = 0;
                    int bNoFaceBlockTsdf = 0;
                    int bNoFaceBlockResidual = 0;
                    int bNoFaceOutOfBounds = 0;
                    int bNoSupportBlockWeight = 0;
                    int bNoSupportBlockDirtyPending = 0;
                    int bNoSupportBlockProvenance = 0;
                    int bNoSupportBlockTsdf = 0;
                    int bNoSupportBlockResidual = 0;
                    int bNoSupportOutOfBounds = 0;

                    for (int dz = -radius; dz <= radius; dz++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            for (int dx = -radius; dx <= radius; dx++)
                            {
                                if (dx == 0 && dy == 0 && dz == 0)
                                    continue;

                                int nx = x + dx;
                                int ny = y + dy;
                                int nz = z + dz;
                                bool faceNeighbor = Mathf.Abs(dx) + Mathf.Abs(dy) + Mathf.Abs(dz) == 1;
                                if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
                                {
                                    bNoSupportOutOfBounds++;
                                    if (faceNeighbor)
                                        bNoFaceOutOfBounds++;
                                    continue;
                                }

                                int neighbor = Index(nx, ny, nz);
                                bool provisionalFlag = HasProvisionalTsdfMarker(neighbor);
                                if (provisionalFlag)
                                {
                                    bNoProvPresent++;
                                    if (faceNeighbor)
                                        bNoProvFacePresent++;
                                    else
                                        bNoProvBlockNotFace++;
                                }

                                if (_weights[neighbor] < minSurfaceCornerWeight)
                                {
                                    bNoSupportBlockWeight++;
                                    if (faceNeighbor)
                                        bNoFaceBlockWeight++;
                                    if (provisionalFlag)
                                        bNoProvBlockWeight++;
                                    continue;
                                }
                                if (VoxelIsDirtyQuarantined(neighbor) ||
                                    VoxelHasPendingTsdfCorrection(neighbor))
                                {
                                    bNoSupportBlockDirtyPending++;
                                    if (faceNeighbor)
                                        bNoFaceBlockDirtyPending++;
                                    if (provisionalFlag)
                                        bNoProvBlockDirtyPending++;
                                    continue;
                                }

                                if (!IsAllowedContinuitySupportSource(neighbor))
                                {
                                    bNoSupportBlockProvenance++;
                                    if (faceNeighbor)
                                        bNoFaceBlockProvenance++;
                                    if (provisionalFlag)
                                        bNoProvBlockProvenance++;
                                    continue;
                                }

                                float value = _tsdf[neighbor];
                                if (float.IsNaN(value) || float.IsInfinity(value))
                                {
                                    bNoSupportBlockTsdf++;
                                    if (faceNeighbor)
                                        bNoFaceBlockTsdf++;
                                    if (provisionalFlag)
                                        bNoProvBlockTsdf++;
                                    continue;
                                }
                                if (Mathf.Abs(value) > maxAbs)
                                {
                                    bNoSupportBlockResidual++;
                                    if (faceNeighbor)
                                        bNoFaceBlockResidual++;
                                    if (provisionalFlag)
                                        bNoProvBlockResidual++;
                                    continue;
                                }

                                bool stableProvisional = IsStableProvisionalContinuitySource(neighbor);
                                if (stableProvisional)
                                {
                                    bNoProvAccepted++;
                                    if (faceNeighbor)
                                        bNoProvFaceAccepted++;
                                }
                                int weight = Mathf.Max(1, _weights[neighbor]);
                                weightedTsdf += value * weight;
                                weightSum += weight;
                                support++;
                                if (faceNeighbor)
                                {
                                    faceSupport++;
                                    if (Mathf.Abs(value) <= fillAbs * 1.5f)
                                        nearSurfaceFaceSupport++;
                                    if (dx != 0)
                                        faceAxisMask |= 1;
                                    else if (dy != 0)
                                        faceAxisMask |= 2;
                                    else
                                        faceAxisMask |= 4;
                                }
                                if (stableProvisional)
                                {
                                    LastTsdfContinuityProvisionalNeighborCount++;
                                    stableProvisionalSupport++;
                                    if (faceNeighbor)
                                    {
                                        stableProvisionalFaceSupport++;
                                        bNoFaceAcceptProvisional++;
                                    }
                                }
                                else
                                {
                                    cleanAnchorSupport++;
                                    if (faceNeighbor)
                                    {
                                        cleanAnchorFaceSupport++;
                                        bNoFaceAcceptClean++;
                                    }
                                }
                                if (value >= 0f)
                                    positive++;
                                else
                                    negative++;
                            }
                        }
                    }

                    bool hasMixedSign = positive > 0 && negative > 0;
                    bool allSameSign = support > 0 && (positive == support || negative == support);
                    bool ordinaryCleanAnchored = cleanAnchorSupport >= requiredSupport && cleanAnchorFaceSupport >= requiredFaceSupport;
                    bool mixedSign = hasMixedSign && ordinaryCleanAnchored;
                    bool sameSignBase = allowSameSignContinuityBaseFill &&
                        ordinaryCleanAnchored &&
                        support >= requiredSameSignSupport &&
                        allSameSign;
                    bool boundaryNoTsdfFill = false;
                    bool verifiedJointAnchor = false;
                    bool verifiedTwoFaceBoundary = false;
                    int faceAxisCount = CountBits3(faceAxisMask);
                    bool boundaryNoTsdfCandidate = fillBoundaryNoTsdfGaps &&
                        !hasMixedSign &&
                        !sameSignBase &&
                        allSameSign &&
                        faceAxisCount >= 2 &&
                        faceSupport >= requiredFaceSupport;
                    if (boundaryNoTsdfCandidate)
                    {
                        LastTsdfBoundaryNoTsdfCandidateCount++;
                        LastTsdfBoundaryNoTsdfProvisionalPresentNeighborCount += bNoProvPresent;
                        LastTsdfBoundaryNoTsdfProvisionalFacePresentCount += bNoProvFacePresent;
                        LastTsdfBoundaryNoTsdfProvisionalAcceptedNeighborCount += bNoProvAccepted;
                        LastTsdfBoundaryNoTsdfProvisionalAcceptedFaceCount += bNoProvFaceAccepted;
                        LastTsdfBoundaryNoTsdfProvisionalBlockedWeightCount += bNoProvBlockWeight;
                        LastTsdfBoundaryNoTsdfProvisionalBlockedDirtyPendingCount += bNoProvBlockDirtyPending;
                        LastTsdfBoundaryNoTsdfProvisionalBlockedTsdfCount += bNoProvBlockTsdf;
                        LastTsdfBoundaryNoTsdfProvisionalBlockedResidualCount += bNoProvBlockResidual;
                        LastTsdfBoundaryNoTsdfProvisionalBlockedProvenanceCount += bNoProvBlockProvenance;
                        LastTsdfBoundaryNoTsdfProvisionalBlockedNotFaceCount += bNoProvBlockNotFace;
                        switch (Mathf.Clamp(faceSupport, 0, 6))
                        {
                            case 0: LastTsdfBoundaryNoTsdfFaceSupport0Count++; break;
                            case 1: LastTsdfBoundaryNoTsdfFaceSupport1Count++; break;
                            case 2: LastTsdfBoundaryNoTsdfFaceSupport2Count++; break;
                            case 3: LastTsdfBoundaryNoTsdfFaceSupport3Count++; break;
                            case 4: LastTsdfBoundaryNoTsdfFaceSupport4Count++; break;
                            case 5: LastTsdfBoundaryNoTsdfFaceSupport5Count++; break;
                            case 6: LastTsdfBoundaryNoTsdfFaceSupport6Count++; break;
                        }
                        int faceDeficit = Mathf.Max(0, requiredBoundaryNoTsdfFaceSupport - faceSupport);
                        if (faceDeficit == 1)
                            LastTsdfBoundaryNoTsdfFaceDeficitOneCount++;
                        else if (faceDeficit >= 2)
                            LastTsdfBoundaryNoTsdfFaceDeficitTwoPlusCount++;
                        LastTsdfBoundaryNoTsdfFaceSlotAcceptedCleanCount += bNoFaceAcceptClean;
                        LastTsdfBoundaryNoTsdfFaceSlotAcceptedProvisionalCount += bNoFaceAcceptProvisional;
                        LastTsdfBoundaryNoTsdfFaceSlotBlockedWeightCount += bNoFaceBlockWeight;
                        LastTsdfBoundaryNoTsdfFaceSlotBlockedDirtyPendingCount += bNoFaceBlockDirtyPending;
                        LastTsdfBoundaryNoTsdfFaceSlotBlockedProvenanceCount += bNoFaceBlockProvenance;
                        LastTsdfBoundaryNoTsdfFaceSlotBlockedTsdfCount += bNoFaceBlockTsdf;
                        LastTsdfBoundaryNoTsdfFaceSlotBlockedResidualCount += bNoFaceBlockResidual;
                        LastTsdfBoundaryNoTsdfFaceSlotOutOfBoundsCount += bNoFaceOutOfBounds;
                        int supportDeficit = Mathf.Max(0, requiredBoundaryNoTsdfSupport - support);
                        if (supportDeficit == 0)
                            LastTsdfBoundaryNoTsdfSupportPassedCount++;
                        else if (supportDeficit == 1)
                            LastTsdfBoundaryNoTsdfSupportDeficitOneCount++;
                        else if (supportDeficit == 2)
                            LastTsdfBoundaryNoTsdfSupportDeficitTwoCount++;
                        else if (supportDeficit == 3)
                            LastTsdfBoundaryNoTsdfSupportDeficitThreeCount++;
                        else
                            LastTsdfBoundaryNoTsdfSupportDeficitFourPlusCount++;
                        LastTsdfBoundaryNoTsdfSupportSlotAcceptedCleanCount += cleanAnchorSupport;
                        LastTsdfBoundaryNoTsdfSupportSlotAcceptedProvisionalCount += stableProvisionalSupport;
                        LastTsdfBoundaryNoTsdfSupportSlotBlockedWeightCount += bNoSupportBlockWeight;
                        LastTsdfBoundaryNoTsdfSupportSlotBlockedDirtyPendingCount += bNoSupportBlockDirtyPending;
                        LastTsdfBoundaryNoTsdfSupportSlotBlockedProvenanceCount += bNoSupportBlockProvenance;
                        LastTsdfBoundaryNoTsdfSupportSlotBlockedTsdfCount += bNoSupportBlockTsdf;
                        LastTsdfBoundaryNoTsdfSupportSlotBlockedResidualCount += bNoSupportBlockResidual;
                        LastTsdfBoundaryNoTsdfSupportSlotOutOfBoundsCount += bNoSupportOutOfBounds;
                        bool hasBudget = maxBoundaryNoTsdfFill <= 0 || LastTsdfBoundaryNoTsdfFilledCount < maxBoundaryNoTsdfFill;
                        bool enoughSupport = support >= requiredBoundaryNoTsdfSupport;
                        bool enoughAxis = faceAxisCount >= 2;
                        bool enoughNearSurface = nearSurfaceFaceSupport >= 1;
                        verifiedTwoFaceBoundary = faceSupport == 2 &&
                            faceAxisCount == 2 &&
                            stableProvisionalFaceSupport == 2 &&
                            cleanAnchorFaceSupport == 0 &&
                            enoughSupport &&
                            enoughNearSurface &&
                            bNoFaceBlockDirtyPending == 0 &&
                            bNoFaceBlockProvenance == 0 &&
                            bNoFaceBlockTsdf == 0 &&
                            bNoFaceBlockResidual == 0;
                        if (verifiedTwoFaceBoundary)
                            LastTsdfBoundaryNoTsdfVerifiedTwoFaceCandidateCount++;
                        bool enoughFace = faceSupport >= requiredBoundaryNoTsdfFaceSupport || verifiedTwoFaceBoundary;
                        bool strictCleanAnchor = cleanAnchorFaceSupport >= 2;
                        verifiedJointAnchor = allowBoundaryNoTsdfProvisionalAnchor &&
                            stableProvisionalSupport >= Mathf.Clamp(minBoundaryNoTsdfProvisionalAnchors, 1, 12) &&
                            stableProvisionalFaceSupport >= Mathf.Clamp(minBoundaryNoTsdfProvisionalFaceAnchors, 1, 6) &&
                            cleanAnchorSupport + stableProvisionalSupport >= requiredBoundaryNoTsdfSupport &&
                            (cleanAnchorFaceSupport + stableProvisionalFaceSupport >= requiredBoundaryNoTsdfFaceSupport || verifiedTwoFaceBoundary);
                        bool enoughCleanAnchor = strictCleanAnchor || verifiedJointAnchor;
                        boundaryNoTsdfFill = hasBudget &&
                            enoughSupport &&
                            enoughFace &&
                            enoughCleanAnchor &&
                            enoughNearSurface &&
                            enoughAxis;
                        if (!boundaryNoTsdfFill)
                        {
                            LastTsdfBoundaryNoTsdfBlockedCount++;
                            if (!hasBudget)
                                LastTsdfBoundaryNoTsdfBlockedBudgetCount++;
                            if (!enoughSupport)
                                LastTsdfBoundaryNoTsdfBlockedSupportCount++;
                            if (!enoughFace)
                                LastTsdfBoundaryNoTsdfBlockedFaceCount++;
                            if (!enoughAxis)
                                LastTsdfBoundaryNoTsdfBlockedAxisCount++;
                            if (!enoughCleanAnchor)
                            {
                                LastTsdfBoundaryNoTsdfBlockedCleanAnchorCount++;
                                if (!verifiedJointAnchor)
                                    LastTsdfBoundaryNoTsdfBlockedProvisionalAnchorCount++;
                            }
                            if (!enoughNearSurface)
                                LastTsdfBoundaryNoTsdfBlockedNearSurfaceCount++;
                        }
                    }

                    if (support < requiredSupport || faceSupport < requiredFaceSupport || weightSum <= 0 || (!mixedSign && !sameSignBase && !boundaryNoTsdfFill))
                    {
                        LastTsdfContinuityBlockedLowSupportCount++;
                        continue;
                    }

                    float oldTsdf = _tsdf[index];
                    int oldWeight = _weights[index];
                    float filledTsdf = Mathf.Clamp(weightedTsdf / weightSum, -fillAbs, fillAbs);
                    if (Mathf.Abs(filledTsdf) < 0.001f)
                        filledTsdf = positive >= negative ? fillAbs : -fillAbs;
                    if (sameSignBase || boundaryNoTsdfFill)
                        filledTsdf = positive >= negative ? fillAbs : -fillAbs;

                    _tsdf[index] = filledTsdf;
                    _weights[index] = (byte)fillWeight;
                    RecordContributionLedger(index, boundaryNoTsdfFill ? "boundary_no_tsdf_fill" : "continuity_fill", oldTsdf, oldWeight, filledTsdf, fillWeight, filledTsdf, _weights[index]);
                    LastTsdfContinuityFilledCount++;
                    if (boundaryNoTsdfFill)
                    {
                        LastTsdfBoundaryNoTsdfFilledCount++;
                        if (cleanAnchorFaceSupport < 2)
                            LastTsdfBoundaryNoTsdfRelaxedAnchorFilledCount++;
                        if (verifiedJointAnchor)
                        {
                            LastTsdfBoundaryNoTsdfVerifiedJointAnchorFilledCount++;
                            if (cleanAnchorFaceSupport == 0)
                                LastTsdfBoundaryNoTsdfZeroCleanJointAnchorFilledCount++;
                        }
                        if (verifiedTwoFaceBoundary)
                            LastTsdfBoundaryNoTsdfVerifiedTwoFaceFilledCount++;
                        LastTsdfContinuitySameSignFilledCount++;
                    }
                    else if (sameSignBase)
                        LastTsdfContinuitySameSignFilledCount++;
                    else
                        LastTsdfContinuityMixedSignFilledCount++;
                }
            }
        }
    }

    private bool IsAllowedContinuitySupportSource(int index)
    {
        if (!rejectContinuityFillFromUnsettledNeighbors)
            return true;

        if (!_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance))
            return false;

        string operation = provenance.LastOperation;
        if (IsStableProvisionalContinuityOperation(operation))
            return allowStableProvisionalContinuitySources;

        return !IsUnsettledContinuitySource(operation);
    }

    private bool IsStableProvisionalContinuitySource(int index)
    {
        return _voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) &&
               IsStableProvisionalContinuityOperation(provenance.LastOperation);
    }

    private bool HasProvisionalTsdfMarker(int index)
    {
        return _provisionalTsdf != null &&
               index >= 0 &&
               index < _provisionalTsdf.Length &&
               _provisionalTsdf[index] != 0;
    }

    private bool UseLegacyMeshExtraction =>
        useLegacyTsdfMeshDisplay && (!useStage03ACleanIsoSurface || useV09LegacyExtractorForStage03A);

    private bool AllowRelaxedMeshExtraction => relaxSurfaceExtractionNearZero && !useStage03ACleanIsoSurface;

    private bool IsCleanMeshTsdfVoxel(int index)
    {
        if (_tsdf == null || _weights == null || index < 0 || index >= _tsdf.Length || index >= _weights.Length)
            return false;
        if (_weights[index] < minSurfaceCornerWeight || float.IsNaN(_tsdf[index]) || float.IsInfinity(_tsdf[index]))
            return false;
        if (!useStage03ACleanIsoSurface)
            return true;
        if (HasProvisionalTsdfMarker(index) || VoxelHasPendingTsdfCorrection(index) || VoxelIsDirtyQuarantined(index))
            return false;
        if (!_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance))
            return !requireCleanMeshVoxelProvenance;

        string operation = provenance.LastOperation;
        if (string.IsNullOrEmpty(operation) || operation == "unknown")
            return false;
        if (operation == "provisional_support" || operation == "strong_sample_seed" ||
            operation == "cleanup_neighbor" || operation == "old_clean_decay" ||
            operation == "old_clean_clear")
        {
            return false;
        }
        if (!allowVerifiedContinuityInCleanMesh &&
            (operation == "continuity_fill" || operation == "boundary_no_tsdf_fill"))
        {
            return false;
        }

        return true;
    }

    private static bool IsStableProvisionalContinuityOperation(string operation)
    {
        return operation == "provisional_support" ||
               operation == "strong_sample_seed" ||
               operation == "primary_plane_hole_band";
    }

    private static int CountBits3(int value)
    {
        value &= 7;
        int count = 0;
        if ((value & 1) != 0)
            count++;
        if ((value & 2) != 0)
            count++;
        if ((value & 4) != 0)
            count++;
        return count;
    }

    private bool NeighborhoodTouchesDirtyOrPending(int x, int y, int z, int radius)
    {
        int r = Mathf.Clamp(radius, 0, 2);
        for (int dz = -r; dz <= r; dz++)
        {
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    int vx = x + dx;
                    int vy = y + dy;
                    int vz = z + dz;
                    if (vx < 0 || vy < 0 || vz < 0 || vx >= _dimX || vy >= _dimY || vz >= _dimZ)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index))
                        return true;
                }
            }
        }

        return false;
    }

    private bool NeighborhoodHasUnsettledContinuitySource(int x, int y, int z, int radius)
    {
        int r = Mathf.Clamp(radius, 0, 2);
        for (int dz = -r; dz <= r; dz++)
        {
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    int vx = x + dx;
                    int vy = y + dy;
                    int vz = z + dz;
                    if (vx < 0 || vy < 0 || vz < 0 || vx >= _dimX || vy >= _dimY || vz >= _dimZ)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (_weights[index] <= 0)
                        continue;

                    if (!_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance))
                        return true;

                    if (!allowStableProvisionalContinuitySources &&
                        IsStableProvisionalContinuityOperation(provenance.LastOperation))
                    {
                        return true;
                    }

                    if (IsUnsettledContinuitySource(provenance.LastOperation))
                        return true;
                }
            }
        }

        return false;
    }

    private bool IsUnsettledContinuitySource(string operation)
    {
        return operation == "continuity_fill" ||
               operation == "boundary_no_tsdf_fill" ||
               operation == "replace" ||
               operation == "guarded_replace" ||
               operation == "band_repair" ||
               operation == "conflict_correct" ||
               operation == "cleanup_neighbor";
    }

    private bool TryBuildLegacyCellVertex(
        int x,
        int y,
        int z,
        Vector3[] cornerPositions,
        float[] cornerValues,
        byte[] cornerWeights,
        out Vector3 vertex)
    {
        vertex = Vector3.zero;
        bool hasPositive = false;
        bool hasNegative = false;
        int supportedCorners = 0;

        for (int i = 0; i < 8; i++)
        {
            int vx = x + CornerOffsetX[i];
            int vy = y + CornerOffsetY[i];
            int vz = z + CornerOffsetZ[i];
            int index = Index(vx, vy, vz);
            cornerPositions[i] = VoxelCenter(vx, vy, vz);
            cornerValues[i] = _tsdf[index];
            cornerWeights[i] = useStage03ACleanIsoSurface && !useV09ExactCornerEligibilityForDiagnosis
                ? (IsCleanMeshTsdfVoxel(index) ? _weights[index] : (byte)0)
                : _weights[index];

            if (cornerWeights[i] >= minSurfaceCornerWeight)
            {
                supportedCorners++;
                if (cornerValues[i] < 0f) hasNegative = true;
                if (cornerValues[i] > 0f) hasPositive = true;
            }
        }

        if (supportedCorners < Mathf.Max(2, minSurfaceCornerWeight) || !hasPositive || !hasNegative)
        {
            LastRejectedNoSurfaceCellCount++;
            return false;
        }

        LastStrictSurfaceCellCandidateCount++;

        Vector3 sum = Vector3.zero;
        int count = 0;
        for (int edge = 0; edge < 12; edge++)
        {
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            if (cornerWeights[a] < minSurfaceCornerWeight || cornerWeights[b] < minSurfaceCornerWeight)
                continue;

            float va = cornerValues[a];
            float vb = cornerValues[b];
            if ((va < 0f && vb < 0f) || (va > 0f && vb > 0f) || Mathf.Approximately(va, vb))
                continue;

            LastStrictSurfaceEdgeCount++;
            float t = Mathf.Clamp01(va / (va - vb));
            sum += Vector3.Lerp(cornerPositions[a], cornerPositions[b], t);
            count++;
        }

        if (count <= 0)
        {
            LastRejectedNoEdgeCellCount++;
            return false;
        }

        vertex = sum / count;
        return Finite(vertex);
    }

    private bool PassMainShellPromotionGate(int x, int y, int z, byte[] cornerWeights, bool hasStrictSurface)
    {
        if (!requireMainShellPromotionGate)
        {
            LastMainShellPromotedCellCount++;
            return true;
        }

        if (IntegratedFrameCount < Mathf.Max(1, minMainShellFusedFrames))
        {
            LastCandidateShellHeldCellCount++;
            return false;
        }

        int weightedCornerCount = 0;
        int weightSum = 0;
        for (int i = 0; i < cornerWeights.Length; i++)
        {
            if (cornerWeights[i] <= 0)
                continue;
            weightedCornerCount++;
            weightSum += cornerWeights[i];
        }

        float averageWeight = weightedCornerCount > 0 ? (float)weightSum / weightedCornerCount : 0f;
        if (averageWeight < Mathf.Max(1, minMainShellAverageCornerWeight))
        {
            LastCandidateShellHeldCellCount++;
            return false;
        }

        int radius = Mathf.Max(1, mainShellNeighborRadiusCells);
        int neighborSurfaceCells = CountPotentialSurfaceNeighborCells(x, y, z, radius);
        int requiredNeighbors = Mathf.Max(1, minMainShellNeighborSurfaceCells);
        bool detailCandidate = separateComplexDetailCandidates && !hasStrictSurface;
        if (detailCandidate)
            requiredNeighbors = Mathf.Max(requiredNeighbors, minDetailPromotionNeighborSurfaceCells);

        if (neighborSurfaceCells < requiredNeighbors)
        {
            if (detailCandidate && !promoteComplexDetailCandidatesToMesh)
            {
                LastDetailCandidateHeldCellCount++;
                return false;
            }

            if (detailCandidate &&
                neighborSurfaceCells >= Mathf.Max(1, minDetailCandidateNeighborSurfaceCells) &&
                TryPromoteDetailCandidateCell(x, y, z))
            {
                LastMainShellPromotedCellCount++;
                LastDetailCandidatePromotedCellCount++;
                return true;
            }

            if (detailCandidate)
                LastDetailCandidateHeldCellCount++;
            else
                LastCandidateShellHeldCellCount++;
            return false;
        }

        if (detailCandidate && !promoteComplexDetailCandidatesToMesh)
        {
            LastDetailCandidateHeldCellCount++;
            return false;
        }

        if (detailCandidate && !TryPromoteDetailCandidateCell(x, y, z))
        {
            LastDetailCandidateHeldCellCount++;
            return false;
        }
        if (detailCandidate)
            LastDetailCandidatePromotedCellCount++;

        LastMainShellPromotedCellCount++;
        return true;
    }

    private bool TryPromoteDetailCandidateCell(int x, int y, int z)
    {
        if (_detailCandidateCellHits == null || _detailCandidateCellLastFrame == null)
            return false;

        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);
        if (x < 0 || y < 0 || z < 0 || x >= cellX || y >= cellY || z >= cellZ)
            return false;

        int index = CellIndex(x, y, z, cellX, cellY);
        if (index < 0 || index >= _detailCandidateCellHits.Length || index >= _detailCandidateCellLastFrame.Length)
            return false;

        int frameIndex = Mathf.Max(0, IntegratedFrameCount);
        if (_detailCandidateCellLastFrame[index] != frameIndex)
        {
            _detailCandidateCellHits[index] = (byte)Mathf.Min(255, _detailCandidateCellHits[index] + 1);
            _detailCandidateCellLastFrame[index] = frameIndex;
        }

        return _detailCandidateCellHits[index] >= Mathf.Max(2, minDetailCandidateFrames);
    }

    private int CountPotentialSurfaceNeighborCells(int x, int y, int z, int radius)
    {
        int count = 0;
        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);
        for (int dz = -radius; dz <= radius; dz++)
        {
            int cz = z + dz;
            if (cz < 0 || cz >= cellZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int cy = y + dy;
                if (cy < 0 || cy >= cellY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int cx = x + dx;
                    if (cx < 0 || cx >= cellX)
                        continue;
                    if (CellHasPotentialMainShellSurface(cx, cy, cz))
                        count++;
                }
            }
        }

        return count;
    }

    private bool CellHasPotentialMainShellSurface(int x, int y, int z)
    {
        bool hasPositive = false;
        bool hasNegative = false;
        int supportedCorners = 0;
        int nearZeroCorners = 0;
        int weightSum = 0;
        for (int i = 0; i < 8; i++)
        {
            int index = Index(x + CornerOffsetX[i], y + CornerOffsetY[i], z + CornerOffsetZ[i]);
            int weight = IsCleanMeshTsdfVoxel(index) ? _weights[index] : 0;
            if (weight < minSurfaceCornerWeight)
                continue;

            supportedCorners++;
            weightSum += weight;
            float value = _tsdf[index];
            if (value < 0f) hasNegative = true;
            if (value > 0f) hasPositive = true;
            if (Mathf.Abs(value) <= nearZeroSurfaceThreshold)
                nearZeroCorners++;
        }

        if (supportedCorners <= 0)
            return false;
        float averageWeight = (float)weightSum / supportedCorners;
        if (averageWeight < Mathf.Max(1, minMainShellAverageCornerWeight))
            return false;
        if (supportedCorners >= Mathf.Max(2, minStrictSurfaceSupportedCorners) && hasPositive && hasNegative)
            return true;
        return AllowRelaxedMeshExtraction &&
               supportedCorners >= Mathf.Max(2, minRelaxedSurfaceSupportedCorners) &&
               nearZeroCorners >= Mathf.Max(1, minRelaxedNearZeroCorners);
    }

    private bool IsSignChange(int a, int b)
    {
        return GetSurfaceEdgeKind(a, b) != SurfaceEdgeKind.None;
    }

    private bool IsLegacySignChange(int a, int b)
    {
        if (a < 0 || b < 0 || a >= _tsdf.Length || b >= _tsdf.Length)
            return false;
        if (useStage03ACleanIsoSurface && !useV09ExactCornerEligibilityForDiagnosis
            ? (!IsCleanMeshTsdfVoxel(a) || !IsCleanMeshTsdfVoxel(b))
            : (_weights[a] < minSurfaceCornerWeight || _weights[b] < minSurfaceCornerWeight))
            return false;

        float va = _tsdf[a];
        float vb = _tsdf[b];
        return (va < 0f && vb > 0f) || (va > 0f && vb < 0f);
    }

    private SurfaceEdgeKind GetSurfaceEdgeKind(int a, int b)
    {
        if (a < 0 || b < 0 || a >= _tsdf.Length || b >= _tsdf.Length)
            return SurfaceEdgeKind.None;
        if (!IsCleanMeshTsdfVoxel(a) || !IsCleanMeshTsdfVoxel(b))
            return SurfaceEdgeKind.None;
        float va = _tsdf[a];
        float vb = _tsdf[b];
        if ((va < 0f && vb > 0f) || (va > 0f && vb < 0f))
            return SurfaceEdgeKind.Strict;
        if (AllowRelaxedMeshExtraction && IsRelaxedSurfaceEdge(va, vb))
            return SurfaceEdgeKind.Relaxed;
        return SurfaceEdgeKind.None;
    }

    private bool PassSurfaceQuadExtractionGate(SurfaceEdgeKind edgeKind, int x, int y, int z, int edgeA, int edgeB)
    {
        if (suppressQuadOnPendingTsdfCorrectionEdge &&
            SurfaceEdgeTouchesPendingTsdfCorrection(edgeA, edgeB))
        {
            LastRejectedCorrectionPendingQuadCount++;
            return false;
        }

        if (suppressQuadOnDirtyTsdfQuarantineEdge &&
            SurfaceEdgeTouchesDirtyTsdfQuarantine(edgeA, edgeB))
        {
            LastRejectedDirtyQuarantineQuadCount++;
            return false;
        }

        if (edgeKind == SurfaceEdgeKind.Strict)
        {
            LastStrictSurfaceEdgeCount++;
            return true;
        }

        if (edgeKind != SurfaceEdgeKind.Relaxed)
            return false;

        LastRelaxedSurfaceEdgeCount++;
        if (rejectRelaxedOnlySignChangeQuads)
        {
            LastRejectedRelaxedQuadCount++;
            return false;
        }

        if (!gateRelaxedQuadsByStrictSupport)
            return true;

        int strictCells = CountStrictSurfaceNeighborCells(x, y, z, relaxedQuadStrictNeighborRadiusCells);
        if (strictCells >= Mathf.Max(1, minRelaxedQuadStrictNeighborCells))
            return true;

        LastRejectedRelaxedQuadCount++;
        return false;
    }

    private bool SurfaceEdgeTouchesPendingTsdfCorrection(int a, int b)
    {
        int minHits = Mathf.Max(1, minPendingCorrectionHitsToSuppressQuad);
        int hitsA = PendingTsdfCorrectionHits(a);
        int hitsB = PendingTsdfCorrectionHits(b);
        if (hitsA >= minHits || hitsB >= minHits)
            return true;
        return hitsA > 0 && hitsB > 0;
    }

    private bool SurfaceEdgeTouchesDirtyTsdfQuarantine(int a, int b)
    {
        return VoxelIsDirtyQuarantined(a) || VoxelIsDirtyQuarantined(b);
    }

    private bool VoxelHasPendingTsdfCorrection(int index)
    {
        return PendingTsdfCorrectionHits(index) > 0;
    }

    private bool VoxelIsDirtyQuarantined(int index)
    {
        if (_dirtyTsdfLastFrame == null ||
            dirtyTsdfQuarantineFrames <= 0 ||
            index < 0 ||
            index >= _dirtyTsdfLastFrame.Length)
        {
            return false;
        }

        int lastDirtyFrame = _dirtyTsdfLastFrame[index];
        return lastDirtyFrame != int.MinValue &&
               LastRawFrameIndex - lastDirtyFrame <= Mathf.Max(0, dirtyTsdfQuarantineFrames);
    }

    private int PendingTsdfCorrectionHits(int index)
    {
        if (_correctionTsdfHits == null ||
            _correctionTsdfLastFrame == null ||
            index < 0 ||
            index >= _correctionTsdfHits.Length ||
            _correctionTsdfLastFrame[index] == int.MinValue)
        {
            return 0;
        }

        return _correctionTsdfHits[index];
    }

    private bool HasPendingTsdfCorrection(int index)
    {
        return _correctionTsdfHits != null &&
               _correctionTsdfLastFrame != null &&
               index >= 0 &&
               index < _correctionTsdfHits.Length &&
               _correctionTsdfHits[index] > 0 &&
               _correctionTsdfLastFrame[index] != int.MinValue;
    }

    private bool CellTouchesPendingTsdfCorrection(int x, int y, int z, int radius)
    {
        if (_correctionTsdfHits == null || _correctionTsdfLastFrame == null)
            return false;

        int r = Mathf.Max(0, radius);
        for (int dz = -r; dz <= r + 1; dz++)
        {
            for (int dy = -r; dy <= r + 1; dy++)
            {
                for (int dx = -r; dx <= r + 1; dx++)
                {
                    int vx = x + dx;
                    int vy = y + dy;
                    int vz = z + dz;
                    if (vx < 0 || vy < 0 || vz < 0 || vx >= _dimX || vy >= _dimY || vz >= _dimZ)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (_correctionTsdfHits[index] > 0 && _correctionTsdfLastFrame[index] != int.MinValue)
                        return true;
                }
            }
        }

        return false;
    }

    private bool CellTouchesDirtyTsdfQuarantine(int x, int y, int z, int radius)
    {
        int r = Mathf.Clamp(radius, 0, 2);
        for (int dz = -r; dz <= r; dz++)
        {
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    int vx = x + dx;
                    int vy = y + dy;
                    int vz = z + dz;
                    if (vx < 0 || vy < 0 || vz < 0 || vx >= _dimX || vy >= _dimY || vz >= _dimZ)
                        continue;

                    if (VoxelIsDirtyQuarantined(Index(vx, vy, vz)))
                        return true;
                }
            }
        }
        return false;
    }

    private bool IsRelaxedSurfaceEdge(float va, float vb)
    {
        float threshold = Mathf.Min(Mathf.Max(0.001f, relaxedEdgeThreshold), Mathf.Max(0.001f, nearZeroSurfaceThreshold));
        return Mathf.Abs(va) <= threshold && Mathf.Abs(vb) <= nearZeroSurfaceThreshold;
    }

    private bool HasStrictSurfaceNeighborCell(int x, int y, int z, int radius)
    {
        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int cx = x + dx;
                    int cy = y + dy;
                    int cz = z + dz;
                    if (cx < 0 || cy < 0 || cz < 0 || cx >= cellX || cy >= cellY || cz >= cellZ)
                        continue;
                    if (CellHasStrictSurface(cx, cy, cz))
                        return true;
                }
            }
        }

        return false;
    }

    private int CountStrictSurfaceNeighborCells(int x, int y, int z, int radius)
    {
        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);
        int count = 0;

        for (int dz = -radius; dz <= radius; dz++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int cx = x + dx;
                    int cy = y + dy;
                    int cz = z + dz;
                    if (cx < 0 || cy < 0 || cz < 0 || cx >= cellX || cy >= cellY || cz >= cellZ)
                        continue;
                    if (CellHasStrictSurface(cx, cy, cz))
                        count++;
                }
            }
        }

        return count;
    }

    private bool CellHasStrictSurface(int x, int y, int z)
    {
        bool hasPositive = false;
        bool hasNegative = false;
        int supportedCorners = 0;

        for (int i = 0; i < 8; i++)
        {
            int index = Index(x + CornerOffsetX[i], y + CornerOffsetY[i], z + CornerOffsetZ[i]);
            if (!IsCleanMeshTsdfVoxel(index))
                continue;

            supportedCorners++;
            float value = _tsdf[index];
            if (value < 0f)
                hasNegative = true;
            else if (value > 0f)
                hasPositive = true;
        }

        return supportedCorners >= Mathf.Max(2, minStrictSurfaceSupportedCorners) && hasPositive && hasNegative;
    }

    private void AddQuadAroundXEdge(int x, int y, int z, int cellX, int cellY, List<Vector3> vertices, List<int> triangles)
    {
        if (y <= 0 || z <= 0 || x >= cellX)
            return;
        AddQuad(
            TryGetCellVertex(x, y, z, cellX, cellY),
            TryGetCellVertex(x, y - 1, z, cellX, cellY),
            TryGetCellVertex(x, y - 1, z - 1, cellX, cellY),
            TryGetCellVertex(x, y, z - 1, cellX, cellY),
            vertices,
            triangles);
    }

    private void AddQuadAroundYEdge(int x, int y, int z, int cellX, int cellY, List<Vector3> vertices, List<int> triangles)
    {
        if (x <= 0 || z <= 0 || y >= cellY)
            return;
        AddQuad(
            TryGetCellVertex(x, y, z, cellX, cellY),
            TryGetCellVertex(x, y, z - 1, cellX, cellY),
            TryGetCellVertex(x - 1, y, z - 1, cellX, cellY),
            TryGetCellVertex(x - 1, y, z, cellX, cellY),
            vertices,
            triangles);
    }

    private void AddQuadAroundZEdge(int x, int y, int z, int cellX, int cellY, List<Vector3> vertices, List<int> triangles)
    {
        if (x <= 0 || y <= 0 || z >= _dimZ - 1)
            return;
        AddQuad(
            TryGetCellVertex(x, y, z, cellX, cellY),
            TryGetCellVertex(x - 1, y, z, cellX, cellY),
            TryGetCellVertex(x - 1, y - 1, z, cellX, cellY),
            TryGetCellVertex(x, y - 1, z, cellX, cellY),
            vertices,
            triangles);
    }

    private int TryGetCellVertex(int x, int y, int z, int cellX, int cellY)
    {
        int cellZ = _dimZ - 1;
        if (x < 0 || y < 0 || z < 0 || x >= cellX || y >= cellY || z >= cellZ)
            return -1;
        return _cellVertexIndices[CellIndex(x, y, z, cellX, cellY)];
    }

    private void AddQuad(int a, int b, int c, int d, List<Vector3> vertices, List<int> triangles)
    {
        if (a < 0 || b < 0 || c < 0 || d < 0)
        {
            LastRejectedWeakQuadCount++;
            return;
        }
        if (!UseLegacyMeshExtraction && rejectStretchedExtractedQuads && IsStretchedExtractedQuad(a, b, c, d, vertices))
        {
            LastRejectedWeakQuadCount++;
            return;
        }

        LastAddedSurfaceQuadCount++;
        triangles.Add(a);
        triangles.Add(b);
        triangles.Add(c);
        triangles.Add(a);
        triangles.Add(c);
        triangles.Add(d);

        if (!doubleSidedTriangles)
            return;

        triangles.Add(c);
        triangles.Add(b);
        triangles.Add(a);
        triangles.Add(d);
        triangles.Add(c);
        triangles.Add(a);
    }

    private bool IsStretchedExtractedQuad(int a, int b, int c, int d, List<Vector3> vertices)
    {
        if (vertices == null ||
            a >= vertices.Count ||
            b >= vertices.Count ||
            c >= vertices.Count ||
            d >= vertices.Count)
            return false;

        Vector3 va = vertices[a];
        Vector3 vb = vertices[b];
        Vector3 vc = vertices[c];
        Vector3 vd = vertices[d];
        float maxEdge = Mathf.Max(0.03f, voxelSizeMeters * maxExtractedQuadEdgeVoxelScale);
        float maxEdgeSq = maxEdge * maxEdge;
        if ((va - vb).sqrMagnitude > maxEdgeSq ||
            (vb - vc).sqrMagnitude > maxEdgeSq ||
            (vc - vd).sqrMagnitude > maxEdgeSq ||
            (vd - va).sqrMagnitude > maxEdgeSq)
            return true;

        float minArea = voxelSizeMeters * voxelSizeMeters * 0.01f;
        float area0 = Vector3.Cross(vb - va, vc - va).magnitude * 0.5f;
        float area1 = Vector3.Cross(vc - va, vd - va).magnitude * 0.5f;
        return area0 + area1 < minArea;
    }

    private void PruneSmallExtractedMeshComponents(List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        LastPrunedMeshComponentCount = 0;
        LastPrunedMeshTriangleCount = 0;

        int triangleCount = triangles.Count / 3;
        int minTriangles = Mathf.Max(0, minExtractedComponentTriangles);
        if (!pruneSmallExtractedMeshComponents || minTriangles <= 0 || triangleCount <= minTriangles)
            return;

        Dictionary<int, List<int>> vertexToTriangles = new Dictionary<int, List<int>>(vertices.Count);
        for (int tri = 0; tri < triangleCount; tri++)
        {
            for (int corner = 0; corner < 3; corner++)
            {
                int vertexIndex = triangles[tri * 3 + corner];
                if (vertexIndex < 0 || vertexIndex >= vertices.Count)
                    continue;

                if (!vertexToTriangles.TryGetValue(vertexIndex, out List<int> linked))
                {
                    linked = new List<int>(4);
                    vertexToTriangles.Add(vertexIndex, linked);
                }
                linked.Add(tri);
            }
        }

        bool[] visited = new bool[triangleCount];
        List<List<int>> components = new List<List<int>>();
        Queue<int> queue = new Queue<int>();

        for (int start = 0; start < triangleCount; start++)
        {
            if (visited[start])
                continue;

            List<int> component = new List<int>(64);
            visited[start] = true;
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                int tri = queue.Dequeue();
                component.Add(tri);

                for (int corner = 0; corner < 3; corner++)
                {
                    int vertexIndex = triangles[tri * 3 + corner];
                    if (!vertexToTriangles.TryGetValue(vertexIndex, out List<int> linked))
                        continue;

                    for (int i = 0; i < linked.Count; i++)
                    {
                        int neighbor = linked[i];
                        if (visited[neighbor])
                            continue;

                        visited[neighbor] = true;
                        queue.Enqueue(neighbor);
                    }
                }
            }

            components.Add(component);
        }

        if (components.Count <= 1)
            return;

        List<Vector3> compactVertices = new List<Vector3>(vertices.Count);
        List<Color> compactColors = colors != null && colors.Count == vertices.Count ? new List<Color>(vertices.Count) : null;
        List<int> compactTriangles = new List<int>(triangles.Count);
        Dictionary<int, int> compactLookup = new Dictionary<int, int>(vertices.Count);
        float minExtent = Mathf.Max(voxelSizeMeters * 2f, voxelSizeMeters * minExtractedComponentExtentVoxelScale);
        int suspectTriangleLimit = Mathf.Max(minTriangles + 1, minTriangles * Mathf.Max(1, suspectComponentTriangleMultiplier));

        for (int componentIndex = 0; componentIndex < components.Count; componentIndex++)
        {
            List<int> component = components[componentIndex];
            float componentExtent = ComputeComponentMaxExtent(vertices, triangles, component);
            bool compactSuspect = component.Count < suspectTriangleLimit && componentExtent < minExtent;
            bool keep = component.Count >= minTriangles && !compactSuspect;
            if (!keep)
            {
                LastPrunedMeshComponentCount++;
                LastPrunedMeshTriangleCount += component.Count;
                LastPrunedComponentTriangleCount += component.Count;
                continue;
            }

            for (int i = 0; i < component.Count; i++)
            {
                int tri = component[i];
                for (int corner = 0; corner < 3; corner++)
                {
                    int oldIndex = triangles[tri * 3 + corner];
                    if (!compactLookup.TryGetValue(oldIndex, out int newIndex))
                    {
                        newIndex = compactVertices.Count;
                        compactVertices.Add(vertices[oldIndex]);
                        if (compactColors != null)
                            compactColors.Add(component.Count < suspectTriangleLimit ? unstableCoverCellColor : cleanCoverCellColor);
                        compactLookup.Add(oldIndex, newIndex);
                    }
                    compactTriangles.Add(newIndex);
                }
            }
        }

        if (compactTriangles.Count <= 0)
            return;

        vertices.Clear();
        vertices.AddRange(compactVertices);
        if (compactColors != null)
        {
            colors.Clear();
            colors.AddRange(compactColors);
        }
        triangles.Clear();
        triangles.AddRange(compactTriangles);
    }

    private static float ComputeComponentMaxExtent(List<Vector3> vertices, List<int> triangles, List<int> component)
    {
        if (vertices == null || triangles == null || component == null || component.Count <= 0)
            return 0f;

        Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);
        for (int i = 0; i < component.Count; i++)
        {
            int tri = component[i];
            int baseIndex = tri * 3;
            if (baseIndex + 2 >= triangles.Count)
                continue;

            for (int corner = 0; corner < 3; corner++)
            {
                int vertexIndex = triangles[baseIndex + corner];
                if (vertexIndex < 0 || vertexIndex >= vertices.Count)
                    continue;

                Vector3 p = vertices[vertexIndex];
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
        }

        if (!Finite(min) || !Finite(max))
            return 0f;

        Vector3 extent = max - min;
        return Mathf.Max(extent.x, Mathf.Max(extent.y, extent.z));
    }

    private struct BoundaryEdgeInfo
    {
        public int A;
        public int B;
        public Vector3 Center;
        public Vector3 Direction;
        public Vector3 Normal;
        public float Length;
    }

    private void BridgeCleanCoplanarBoundaryEdges(List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        if (!bridgeCleanCoplanarBoundaryEdges ||
            vertices == null ||
            triangles == null ||
            vertices.Count <= 0 ||
            triangles.Count < 3 ||
            maxBoundaryBridgesPerRebuild <= 0)
        {
            return;
        }

        List<BoundaryEdgeInfo> edges = CollectBoundaryEdges(vertices, triangles);
        if (edges.Count < 2)
            return;

        float maxDistance = Mathf.Max(0.01f, voxelSizeMeters * maxBoundaryBridgeDistanceVoxelScale);
        float maxDistanceSq = maxDistance * maxDistance;
        float maxPlaneDistance = Mathf.Max(0.002f, voxelSizeMeters * maxBoundaryBridgePlaneDistanceVoxelScale);
        float minNormalDot = Mathf.Clamp(minBoundaryBridgeNormalDot, 0.75f, 1f);
        HashSet<ulong> bridgedVertices = new HashSet<ulong>();
        int bridgeLimit = Mathf.Max(1, maxBoundaryBridgesPerRebuild);
        Dictionary<Vector3Int, List<int>> buckets = new Dictionary<Vector3Int, List<int>>();
        for (int i = 0; i < edges.Count; i++)
        {
            Vector3Int bucket = BoundaryBridgeBucket(edges[i].Center, maxDistance);
            if (!buckets.TryGetValue(bucket, out List<int> list))
            {
                list = new List<int>(8);
                buckets.Add(bucket, list);
            }
            list.Add(i);
        }

        for (int i = 0; i < edges.Count && LastAddedBoundaryBridgeCount < bridgeLimit; i++)
        {
            BoundaryEdgeInfo a = edges[i];
            Vector3Int bucket = BoundaryBridgeBucket(a.Center, maxDistance);
            for (int bz = -1; bz <= 1 && LastAddedBoundaryBridgeCount < bridgeLimit; bz++)
            {
                for (int by = -1; by <= 1 && LastAddedBoundaryBridgeCount < bridgeLimit; by++)
                {
                    for (int bx = -1; bx <= 1 && LastAddedBoundaryBridgeCount < bridgeLimit; bx++)
                    {
                        Vector3Int neighborBucket = new Vector3Int(bucket.x + bx, bucket.y + by, bucket.z + bz);
                        if (!buckets.TryGetValue(neighborBucket, out List<int> candidates))
                            continue;

                        for (int c = 0; c < candidates.Count && LastAddedBoundaryBridgeCount < bridgeLimit; c++)
                        {
                            int j = candidates[c];
                            if (j <= i)
                                continue;

                            BoundaryEdgeInfo b = edges[j];
                            LastBoundaryBridgeCandidateCount++;

                            if (!PassBoundaryBridgeGeometry(a, b, vertices, maxDistanceSq, maxPlaneDistance, minNormalDot, out bool flip))
                            {
                                LastRejectedBoundaryBridgeCount++;
                                continue;
                            }

                            int b0 = flip ? b.B : b.A;
                            int b1 = flip ? b.A : b.B;
                            if (SharesEndpoint(a.A, a.B, b0, b1))
                            {
                                LastRejectedBoundaryBridgeCount++;
                                continue;
                            }

                            ulong bridgeKey0 = EdgeKey(a.A, b0);
                            ulong bridgeKey1 = EdgeKey(a.B, b1);
                            if (bridgedVertices.Contains(bridgeKey0) || bridgedVertices.Contains(bridgeKey1))
                            {
                                LastRejectedBoundaryBridgeCount++;
                                continue;
                            }

                            Vector3 mid0 = (vertices[a.A] + vertices[b0]) * 0.5f;
                            Vector3 mid1 = (vertices[a.B] + vertices[b1]) * 0.5f;
                            Vector3 center = (mid0 + mid1) * 0.5f;
                            if (!BridgePathIsClean(center, mid0, mid1))
                            {
                                LastRejectedBoundaryBridgeCount++;
                                continue;
                            }

                            if (!AddBoundaryBridgeQuad(a.A, a.B, b1, b0, vertices, triangles, colors))
                            {
                                LastRejectedBoundaryBridgeCount++;
                                continue;
                            }
                            bridgedVertices.Add(bridgeKey0);
                            bridgedVertices.Add(bridgeKey1);
                            LastAddedBoundaryBridgeCount++;
                        }
                    }
                }
            }
        }
    }

    private static Vector3Int BoundaryBridgeBucket(Vector3 point, float cellSize)
    {
        float inv = 1f / Mathf.Max(0.001f, cellSize);
        return new Vector3Int(
            Mathf.FloorToInt(point.x * inv),
            Mathf.FloorToInt(point.y * inv),
            Mathf.FloorToInt(point.z * inv));
    }

    private List<BoundaryEdgeInfo> CollectBoundaryEdges(List<Vector3> vertices, List<int> triangles)
    {
        int triangleCount = triangles.Count / 3;
        Dictionary<ulong, List<int>> edgeToTriangles = new Dictionary<ulong, List<int>>(triangleCount * 3);
        Vector3[] triangleNormals = new Vector3[triangleCount];
        for (int tri = 0; tri < triangleCount; tri++)
        {
            triangleNormals[tri] = ComputeTriangleNormal(vertices, triangles, tri);
            AddEdgeTriangle(edgeToTriangles, triangles[tri * 3], triangles[tri * 3 + 1], tri);
            AddEdgeTriangle(edgeToTriangles, triangles[tri * 3 + 1], triangles[tri * 3 + 2], tri);
            AddEdgeTriangle(edgeToTriangles, triangles[tri * 3 + 2], triangles[tri * 3], tri);
        }

        List<BoundaryEdgeInfo> result = new List<BoundaryEdgeInfo>();
        foreach (KeyValuePair<ulong, List<int>> pair in edgeToTriangles)
        {
            if (pair.Value == null || pair.Value.Count != 1)
                continue;

            DecodeEdgeKey(pair.Key, out int a, out int b);
            if (a < 0 || b < 0 || a >= vertices.Count || b >= vertices.Count)
                continue;

            Vector3 va = vertices[a];
            Vector3 vb = vertices[b];
            Vector3 delta = vb - va;
            float length = delta.magnitude;
            if (length < voxelSizeMeters * 0.25f || length > voxelSizeMeters * 2.5f)
                continue;

            Vector3 normal = triangleNormals[pair.Value[0]];
            if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
                continue;

            result.Add(new BoundaryEdgeInfo
            {
                A = a,
                B = b,
                Center = (va + vb) * 0.5f,
                Direction = delta / length,
                Normal = normal.normalized,
                Length = length
            });
        }

        return result;
    }

    private bool PassBoundaryBridgeGeometry(
        BoundaryEdgeInfo a,
        BoundaryEdgeInfo b,
        List<Vector3> vertices,
        float maxDistanceSq,
        float maxPlaneDistance,
        float minNormalDot,
        out bool flip)
    {
        flip = false;
        float normalDot = Mathf.Abs(Vector3.Dot(a.Normal, b.Normal));
        if (normalDot < minNormalDot)
            return false;

        float dirDot = Mathf.Abs(Vector3.Dot(a.Direction, b.Direction));
        if (dirDot < 0.80f)
            return false;

        float planeAB = Mathf.Abs(Vector3.Dot(b.Center - a.Center, a.Normal));
        float planeBA = Mathf.Abs(Vector3.Dot(a.Center - b.Center, b.Normal));
        if (planeAB > maxPlaneDistance || planeBA > maxPlaneDistance)
            return false;

        float same = (vertices[a.A] - vertices[b.A]).sqrMagnitude + (vertices[a.B] - vertices[b.B]).sqrMagnitude;
        float crossed = (vertices[a.A] - vertices[b.B]).sqrMagnitude + (vertices[a.B] - vertices[b.A]).sqrMagnitude;
        flip = crossed < same;
        float pairDistance = Mathf.Min(same, crossed);
        if (pairDistance > maxDistanceSq * 2f)
            return false;

        float centerDistanceSq = (a.Center - b.Center).sqrMagnitude;
        if (centerDistanceSq > maxDistanceSq * 4f)
            return false;

        return true;
    }

    private bool BridgePathIsClean(Vector3 center, Vector3 mid0, Vector3 mid1)
    {
        return BridgePointIsClean(center) && BridgePointIsClean(mid0) && BridgePointIsClean(mid1);
    }

    private bool BridgePointIsClean(Vector3 world)
    {
        if (!TryWorldToVoxel(world, out int x, out int y, out int z))
            return false;

        int radius = Mathf.Clamp(boundaryBridgeCleanRadiusVoxels, 0, 3);
        for (int dz = -radius; dz <= radius; dz++)
        {
            int vz = z + dz;
            if (vz < 0 || vz >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int vy = y + dy;
                if (vy < 0 || vy >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    if (vx < 0 || vx >= _dimX)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (_weights[index] < minSurfaceCornerWeight)
                        return false;
                    if (Mathf.Abs(_tsdf[index]) > maxBoundaryBridgeAbsTsdf)
                        return false;
                    if (VoxelHasPendingTsdfCorrection(index) || VoxelIsDirtyQuarantined(index))
                        return false;
                }
            }
        }

        return true;
    }

    private bool AddBoundaryBridgeQuad(int a, int b, int c, int d, List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        if (rejectStretchedExtractedQuads && IsStretchedExtractedQuad(a, b, c, d, vertices))
            return false;

        triangles.Add(a);
        triangles.Add(b);
        triangles.Add(c);
        triangles.Add(a);
        triangles.Add(c);
        triangles.Add(d);

        if (colors != null && colors.Count == vertices.Count)
        {
            colors[a] = cleanCoverCellColor;
            colors[b] = cleanCoverCellColor;
            colors[c] = cleanCoverCellColor;
            colors[d] = cleanCoverCellColor;
        }

        return true;
    }

    private static bool SharesEndpoint(int a0, int a1, int b0, int b1)
    {
        return a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
    }

    private void PruneDanglingExtractedMeshTriangles(List<Vector3> vertices, List<int> triangles)
    {
        if (!pruneDanglingExtractedMeshTriangles || triangles.Count < 3)
            return;

        int minSharedEdges = Mathf.Clamp(minTriangleSharedEdgesToKeep, 0, 2);
        if (minSharedEdges <= 0)
            return;

        int iterations = Mathf.Max(1, danglingTrianglePruneIterations);
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            int triangleCount = triangles.Count / 3;
            if (triangleCount <= 0)
                return;

            Dictionary<ulong, int> edgeUseCounts = new Dictionary<ulong, int>(triangleCount * 3);
            for (int tri = 0; tri < triangleCount; tri++)
            {
                AddEdgeUse(edgeUseCounts, triangles[tri * 3], triangles[tri * 3 + 1]);
                AddEdgeUse(edgeUseCounts, triangles[tri * 3 + 1], triangles[tri * 3 + 2]);
                AddEdgeUse(edgeUseCounts, triangles[tri * 3 + 2], triangles[tri * 3]);
            }

            List<int> keptTriangles = new List<int>(triangles.Count);
            int prunedThisPass = 0;
            for (int tri = 0; tri < triangleCount; tri++)
            {
                int a = triangles[tri * 3];
                int b = triangles[tri * 3 + 1];
                int c = triangles[tri * 3 + 2];
                int sharedEdges = 0;
                if (GetEdgeUseCount(edgeUseCounts, a, b) > 1)
                    sharedEdges++;
                if (GetEdgeUseCount(edgeUseCounts, b, c) > 1)
                    sharedEdges++;
                if (GetEdgeUseCount(edgeUseCounts, c, a) > 1)
                    sharedEdges++;

                if (sharedEdges < minSharedEdges)
                {
                    prunedThisPass++;
                    continue;
                }

                keptTriangles.Add(a);
                keptTriangles.Add(b);
                keptTriangles.Add(c);
            }

            if (prunedThisPass <= 0)
                return;

            LastPrunedMeshTriangleCount += prunedThisPass;
            LastPrunedDanglingTriangleCount += prunedThisPass;
            triangles.Clear();
            triangles.AddRange(keptTriangles);
        }
    }

    private void PruneSpikePatchTriangles(List<Vector3> vertices, List<int> triangles)
    {
        if (!pruneSpikePatchTriangles || vertices == null || triangles == null || triangles.Count < 3)
            return;

        int iterations = Mathf.Max(1, spikePatchPruneIterations);
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            int triangleCount = triangles.Count / 3;
            if (triangleCount <= 0)
                return;

            Dictionary<ulong, List<int>> edgeToTriangles = new Dictionary<ulong, List<int>>(triangleCount * 3);
            Vector3[] triangleNormals = new Vector3[triangleCount];
            for (int tri = 0; tri < triangleCount; tri++)
            {
                triangleNormals[tri] = ComputeTriangleNormal(vertices, triangles, tri);
                AddEdgeTriangle(edgeToTriangles, triangles[tri * 3], triangles[tri * 3 + 1], tri);
                AddEdgeTriangle(edgeToTriangles, triangles[tri * 3 + 1], triangles[tri * 3 + 2], tri);
                AddEdgeTriangle(edgeToTriangles, triangles[tri * 3 + 2], triangles[tri * 3], tri);
            }

            bool[] visited = new bool[triangleCount];
            bool[] remove = new bool[triangleCount];
            int prunedThisPass = 0;
            Queue<int> queue = new Queue<int>();
            List<int> patch = new List<int>(32);
            int maxTriangles = Mathf.Max(2, maxSpikePatchTriangles);

            for (int start = 0; start < triangleCount; start++)
            {
                if (visited[start])
                    continue;

                patch.Clear();
                visited[start] = true;
                queue.Enqueue(start);
                while (queue.Count > 0)
                {
                    int tri = queue.Dequeue();
                    patch.Add(tri);
                    EnqueueSimilarNormalTriNeighbors(triangles, edgeToTriangles, triangleNormals, tri, visited, queue);
                }

                if (patch.Count <= 0 || patch.Count > maxTriangles)
                    continue;

                if (!IsSpikePatch(vertices, triangles, edgeToTriangles, patch))
                    continue;

                for (int i = 0; i < patch.Count; i++)
                {
                    int tri = patch[i];
                    if (remove[tri])
                        continue;
                    remove[tri] = true;
                    prunedThisPass++;
                }
            }

            if (prunedThisPass <= 0)
                return;

            List<int> keptTriangles = new List<int>(triangles.Count - prunedThisPass * 3);
            for (int tri = 0; tri < triangleCount; tri++)
            {
                if (remove[tri])
                    continue;
                keptTriangles.Add(triangles[tri * 3]);
                keptTriangles.Add(triangles[tri * 3 + 1]);
                keptTriangles.Add(triangles[tri * 3 + 2]);
            }

            triangles.Clear();
            triangles.AddRange(keptTriangles);
            LastPrunedMeshTriangleCount += prunedThisPass;
            LastPrunedSpikeTriangleCount += prunedThisPass;
        }
    }

    private void AnalyzeMeshDiagnostics(List<Vector3> vertices, List<int> triangles)
    {
        ResetMeshDiagnostics();
        if (!runMeshDiagnosticsOnRebuild || vertices == null || triangles == null || triangles.Count < 3)
        {
            UpdateMeshDiagnosticHotspots();
            return;
        }

        int triangleCount = triangles.Count / 3;
        Dictionary<ulong, List<int>> edgeToTriangles = new Dictionary<ulong, List<int>>(triangleCount * 3);
        Vector3[] triangleNormals = new Vector3[triangleCount];
        Vector3[] triangleCenters = new Vector3[triangleCount];
        for (int tri = 0; tri < triangleCount; tri++)
        {
            triangleNormals[tri] = ComputeTriangleNormal(vertices, triangles, tri);
            triangleCenters[tri] = ComputeTriangleCenter(vertices, triangles, tri);
            AddEdgeTriangle(edgeToTriangles, triangles[tri * 3], triangles[tri * 3 + 1], tri);
            AddEdgeTriangle(edgeToTriangles, triangles[tri * 3 + 1], triangles[tri * 3 + 2], tri);
            AddEdgeTriangle(edgeToTriangles, triangles[tri * 3 + 2], triangles[tri * 3], tri);
        }

        foreach (KeyValuePair<ulong, List<int>> pair in edgeToTriangles)
        {
            int count = pair.Value != null ? pair.Value.Count : 0;
            if (count == 1)
                LastDiagBoundaryEdgeCount++;
            else if (count > 2)
                LastDiagNonManifoldEdgeCount++;
        }

        AnalyzeMeshComponents(triangles, edgeToTriangles, triangleCount);
        AnalyzeNormalPatches(vertices, triangles, edgeToTriangles, triangleNormals, triangleCenters, triangleCount);
        SortMeshDiagnosticHotspots();
        LastDiagLikelyCause = GuessMeshDiagnosticCause(triangleCount);
        UpdateMeshDiagnosticHotspots();
    }

    private void ResetExtractionDiagnostics()
    {
        LastStrictSurfaceEdgeCount = 0;
        LastRelaxedSurfaceEdgeCount = 0;
        LastAddedSurfaceQuadCount = 0;
        LastSurfaceCellScanCount = 0;
        LastStrictSurfaceCellCandidateCount = 0;
        LastRelaxedSurfaceCellCandidateCount = 0;
        LastBuiltSurfaceCellVertexCount = 0;
        LastRejectedNoSurfaceCellCount = 0;
        LastRejectedMainGateCellCount = 0;
        LastRejectedNoEdgeCellCount = 0;
        LastRejectedEdgeCountCellCount = 0;
        LastSurfaceQuadCandidateCount = 0;
        LastPrePruneMeshTriangleCount = 0;
        LastPostComponentMeshTriangleCount = 0;
        LastPostBridgeMeshTriangleCount = 0;
        LastRejectedRelaxedQuadCount = 0;
        LastRejectedWeakQuadCount = 0;
        LastRejectedCorrectionPendingCellCount = 0;
        LastRejectedCorrectionPendingQuadCount = 0;
        LastRejectedDirtyQuarantineCellCount = 0;
        LastRejectedDirtyQuarantineQuadCount = 0;
        LastBoundaryBridgeCandidateCount = 0;
        LastAddedBoundaryBridgeCount = 0;
        LastRejectedBoundaryBridgeCount = 0;
        LastHeldDisplayTriangleCount = 0;
        LastCommittedMeshHoldSpatialCount = 0;
        LastCommittedMeshHoldTriangleCount = 0;
        LastCommittedMeshGrowthTriangleCount = 0;
        LastTsdfContinuityCandidateCount = 0;
        LastTsdfContinuityFilledCount = 0;
        LastTsdfContinuitySameSignFilledCount = 0;
        LastTsdfContinuityMixedSignFilledCount = 0;
        LastTsdfContinuityProvisionalNeighborCount = 0;
        LastTsdfContinuityBlockedDirtyPendingCount = 0;
        LastTsdfContinuityBlockedLowSupportCount = 0;
        LastTsdfContinuityBlockedBudgetCount = 0;
        LastTsdfContinuityBlockedUnsettledNeighborCount = 0;
        LastTsdfBoundaryNoTsdfCandidateCount = 0;
        LastTsdfBoundaryNoTsdfFilledCount = 0;
        LastTsdfBoundaryNoTsdfBlockedCount = 0;
        LastTsdfBoundaryNoTsdfBlockedSupportCount = 0;
        LastTsdfBoundaryNoTsdfBlockedFaceCount = 0;
        LastTsdfBoundaryNoTsdfBlockedAxisCount = 0;
        LastTsdfBoundaryNoTsdfBlockedCleanAnchorCount = 0;
        LastTsdfBoundaryNoTsdfBlockedProvisionalAnchorCount = 0;
        LastTsdfBoundaryNoTsdfBlockedNearSurfaceCount = 0;
        LastTsdfBoundaryNoTsdfBlockedBudgetCount = 0;
        LastTsdfBoundaryNoTsdfRelaxedAnchorFilledCount = 0;
        LastTsdfBoundaryNoTsdfVerifiedJointAnchorFilledCount = 0;
        LastTsdfBoundaryNoTsdfZeroCleanJointAnchorFilledCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalPresentNeighborCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalFacePresentCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalAcceptedNeighborCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalAcceptedFaceCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalBlockedWeightCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalBlockedDirtyPendingCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalBlockedTsdfCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalBlockedResidualCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalBlockedProvenanceCount = 0;
        LastTsdfBoundaryNoTsdfProvisionalBlockedNotFaceCount = 0;
        LastTsdfBoundaryNoTsdfFaceSupport0Count = 0;
        LastTsdfBoundaryNoTsdfFaceSupport1Count = 0;
        LastTsdfBoundaryNoTsdfFaceSupport2Count = 0;
        LastTsdfBoundaryNoTsdfFaceSupport3Count = 0;
        LastTsdfBoundaryNoTsdfFaceSupport4Count = 0;
        LastTsdfBoundaryNoTsdfFaceSupport5Count = 0;
        LastTsdfBoundaryNoTsdfFaceSupport6Count = 0;
        LastTsdfBoundaryNoTsdfFaceDeficitOneCount = 0;
        LastTsdfBoundaryNoTsdfFaceDeficitTwoPlusCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotAcceptedCleanCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotAcceptedProvisionalCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotBlockedWeightCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotBlockedDirtyPendingCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotBlockedProvenanceCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotBlockedTsdfCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotBlockedResidualCount = 0;
        LastTsdfBoundaryNoTsdfFaceSlotOutOfBoundsCount = 0;
        LastTsdfBoundaryNoTsdfVerifiedTwoFaceCandidateCount = 0;
        LastTsdfBoundaryNoTsdfVerifiedTwoFaceFilledCount = 0;
        LastTsdfBoundaryNoTsdfSupportPassedCount = 0;
        LastTsdfBoundaryNoTsdfSupportDeficitOneCount = 0;
        LastTsdfBoundaryNoTsdfSupportDeficitTwoCount = 0;
        LastTsdfBoundaryNoTsdfSupportDeficitThreeCount = 0;
        LastTsdfBoundaryNoTsdfSupportDeficitFourPlusCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotAcceptedCleanCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotAcceptedProvisionalCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotBlockedWeightCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotBlockedDirtyPendingCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotBlockedProvenanceCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotBlockedTsdfCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotBlockedResidualCount = 0;
        LastTsdfBoundaryNoTsdfSupportSlotOutOfBoundsCount = 0;
    }

    private void ResetMeshDiagnostics()
    {
        LastDiagMeshComponentCount = 0;
        LastDiagLargestComponentTriangles = 0;
        LastDiagBoundaryEdgeCount = 0;
        LastDiagNonManifoldEdgeCount = 0;
        LastDiagNormalPatchCount = 0;
        LastDiagSuspectPatchCount = 0;
        LastDiagWorstPatchScore = 0f;
        LastDiagWorstPatchTriangles = 0;
        LastDiagWorstPatchBoundaryRatio = 0f;
        LastDiagWorstPatchExtentMeters = 0f;
        LastDiagLikelyCause = "none";
        _meshDiagnosticHotspots.Clear();
    }

    private void ResetTumorSuspectOverlayDiagnostics()
    {
        _auditSuspectMeshVoxels.Clear();
        _auditPureGeometryMeshVoxels.Clear();
        LastTumorSuspectTriangleCount = 0;
        LastTumorRiskVertexCount = 0;
        LastTumorDirtyTriangleCount = 0;
        LastTumorPendingTriangleCount = 0;
        LastTumorReplaceHitBlockedTriangleCount = 0;
        LastTumorReplaceWeightBlockedTriangleCount = 0;
        LastTumorReplaceReadyTriangleCount = 0;
        ResetSuspectMeshSourceDiagnostics();
        LastTumorIslandTriangleCount = 0;
        LastTumorOpenEdgeTriangleCount = 0;
        LastTumorNonManifoldTriangleCount = 0;
        LastTumorLongEdgeTriangleCount = 0;
        LastIslandCauseComponentCount = 0;
        LastIslandCauseBoundaryVoxelCount = 0;
        LastIslandCauseNoTsdfCount = 0;
        LastIslandCausePendingCount = 0;
        LastIslandCauseDirtyCount = 0;
        LastIslandCauseLowWeightCount = 0;
        LastIslandCausePlaneMismatchCount = 0;
        LastIslandCausePrunedCount = 0;
        LastPureGeometrySuspectTriangleCount = 0;
        LastPureGeometryNormalTriangleCount = 0;
        LastPureGeometryProtrusionTriangleCount = 0;
        LastPureGeometryDoubleLayerTriangleCount = 0;
        LastDoubleLayerSourceDirtyCount = 0;
        LastDoubleLayerSourcePendingCount = 0;
        LastDoubleLayerSourceBandConflictCount = 0;
        LastDoubleLayerSourceOldLockedCount = 0;
        LastDoubleLayerSourceCleanCount = 0;
        LastDoubleLayerSourceEmptyCount = 0;
        LastDoubleLayerSourceMixedCount = 0;
        LastDoubleLayerSourceUnknownCount = 0;
        LastDoubleLayerCleanNoLifecycleCount = 0;
        LastDoubleLayerCleanNoProvenanceCount = 0;
        LastDoubleLayerCleanConflictHistoryCount = 0;
        LastDoubleLayerCleanReplaceHistoryCount = 0;
        LastDoubleLayerCleanExpiredDirtyCount = 0;
        LastDoubleLayerCleanLowWeightCount = 0;
        LastDoubleLayerCleanHighWeightCount = 0;
        LastDoubleLayerCleanLastIntegrateCount = 0;
        LastDoubleLayerCleanLastCarveCount = 0;
        LastDoubleLayerCleanLastReplaceCount = 0;
        LastDoubleLayerCleanLastRepairCount = 0;
        LastDoubleLayerCleanIntegrateCarveHistoryCount = 0;
        LastDoubleLayerCleanMultiFrameHistoryCount = 0;
        LastDoubleLayerValidatedPairCount = 0;
        LastDoubleLayerRejectedMissingSourceCount = 0;
        LastDoubleLayerRejectedSourceNormalCount = 0;
        LastDoubleLayerRejectedSourcePlaneCount = 0;
        LastDoubleLayerRejectedSourceLateralCount = 0;
        LastDoubleLayerMeshAlignmentMismatchCount = 0;
    }

    private void ApplyMeshTumorSuspectOverlay(List<Vector3> vertices, List<int> triangles, List<Color> colors)
    {
        ResetTumorSuspectOverlayDiagnostics();
        if (!showMeshTumorSuspectOverlay || vertices == null || triangles == null || colors == null || triangles.Count < 3)
            return;

        while (colors.Count < vertices.Count)
            colors.Add(cleanCoverCellColor);
        if (colors.Count != vertices.Count)
            return;

        int triangleCount = triangles.Count / 3;
        Dictionary<ulong, List<int>> edgeToTriangles = new Dictionary<ulong, List<int>>(triangleCount * 3);
        Vector3[] triangleCenters = new Vector3[triangleCount];
        Vector3[] triangleNormals = new Vector3[triangleCount];
        bool[] riskHigh = new bool[triangleCount];
        bool[] riskMedium = new bool[triangleCount];
        bool[] dirtyRisk = new bool[triangleCount];
        bool[] pendingRisk = new bool[triangleCount];
        bool[] hitBlockedRisk = new bool[triangleCount];
        bool[] weightBlockedRisk = new bool[triangleCount];
        bool[] replaceReadyRisk = new bool[triangleCount];
        bool[] nonManifoldRisk = new bool[triangleCount];
        bool[] islandRisk = new bool[triangleCount];
        bool[] openRisk = new bool[triangleCount];
        bool[] longEdgeRisk = new bool[triangleCount];
        bool[] pureNormalRisk = new bool[triangleCount];
        bool[] pureProtrusionRisk = new bool[triangleCount];
        bool[] pureDoubleLayerRisk = new bool[triangleCount];
        bool[] pureGeometryRisk = new bool[triangleCount];

        for (int tri = 0; tri < triangleCount; tri++)
        {
            int a = triangles[tri * 3];
            int b = triangles[tri * 3 + 1];
            int c = triangles[tri * 3 + 2];
            if (!TriangleIndicesValid(vertices, a, b, c))
                continue;

            triangleCenters[tri] = (vertices[a] + vertices[b] + vertices[c]) / 3f;
            triangleNormals[tri] = ComputeTriangleNormal(vertices, triangles, tri);
            AddEdgeTriangle(edgeToTriangles, a, b, tri);
            AddEdgeTriangle(edgeToTriangles, b, c, tri);
            AddEdgeTriangle(edgeToTriangles, c, a, tri);
        }

        int[] boundaryEdgesPerTri = new int[triangleCount];
        int[] nonManifoldEdgesPerTri = new int[triangleCount];
        int boundaryUseLimit = doubleSidedTriangles ? 2 : 1;
        int manifoldUseLimit = doubleSidedTriangles ? 4 : 2;
        foreach (KeyValuePair<ulong, List<int>> pair in edgeToTriangles)
        {
            int count = pair.Value != null ? pair.Value.Count : 0;
            if (count > boundaryUseLimit && count <= manifoldUseLimit)
                continue;

            for (int i = 0; i < pair.Value.Count; i++)
            {
                int tri = pair.Value[i];
                if (tri < 0 || tri >= triangleCount)
                    continue;
                if (count <= boundaryUseLimit)
                    boundaryEdgesPerTri[tri]++;
                else
                    nonManifoldEdgesPerTri[tri]++;
            }
        }

        float longEdgeSq = Mathf.Pow(Mathf.Max(0.02f, voxelSizeMeters * tumorSuspectLongEdgeVoxelScale), 2f);
        for (int tri = 0; tri < triangleCount; tri++)
        {
            int a = triangles[tri * 3];
            int b = triangles[tri * 3 + 1];
            int c = triangles[tri * 3 + 2];
            if (!TriangleIndicesValid(vertices, a, b, c))
                continue;

            bool dirty = false;
            bool pending = false;
            bool hitBlocked = false;
            bool weightBlocked = false;
            bool replaceReady = false;
            AccumulateMeshTsdfRepairState(triangleCenters[tri], ref dirty, ref pending, ref hitBlocked, ref weightBlocked, ref replaceReady);
            AccumulateMeshTsdfRepairState(vertices[a], ref dirty, ref pending, ref hitBlocked, ref weightBlocked, ref replaceReady);
            AccumulateMeshTsdfRepairState(vertices[b], ref dirty, ref pending, ref hitBlocked, ref weightBlocked, ref replaceReady);
            AccumulateMeshTsdfRepairState(vertices[c], ref dirty, ref pending, ref hitBlocked, ref weightBlocked, ref replaceReady);
            if (dirty)
            {
                riskHigh[tri] = true;
                dirtyRisk[tri] = true;
                LastTumorDirtyTriangleCount++;
            }
            if (pending)
                pendingRisk[tri] = true;
            if (hitBlocked)
                hitBlockedRisk[tri] = true;
            if (weightBlocked)
                weightBlockedRisk[tri] = true;
            if (replaceReady)
                replaceReadyRisk[tri] = true;

            if (nonManifoldEdgesPerTri[tri] > 0)
            {
                riskHigh[tri] = true;
                nonManifoldRisk[tri] = true;
                LastTumorNonManifoldTriangleCount++;
            }

            if (boundaryEdgesPerTri[tri] >= 3)
            {
                riskMedium[tri] = true;
                openRisk[tri] = true;
                LastTumorOpenEdgeTriangleCount++;
            }

            bool longEdge = (vertices[a] - vertices[b]).sqrMagnitude > longEdgeSq ||
                            (vertices[b] - vertices[c]).sqrMagnitude > longEdgeSq ||
                            (vertices[c] - vertices[a]).sqrMagnitude > longEdgeSq;
            if (longEdge)
            {
                if (dirty)
                    riskHigh[tri] = true;
                else
                    riskMedium[tri] = true;
                longEdgeRisk[tri] = true;
                LastTumorLongEdgeTriangleCount++;
            }
        }

        MarkSmallOpenMeshIslands(vertices, triangles, edgeToTriangles, triangleCenters, boundaryEdgesPerTri, riskHigh, riskMedium, islandRisk);
        DetectPureGeometrySuspects(
            vertices,
            triangles,
            edgeToTriangles,
            triangleCenters,
            triangleNormals,
            nonManifoldRisk,
            islandRisk,
            pureNormalRisk,
            pureProtrusionRisk,
            pureDoubleLayerRisk,
            pureGeometryRisk);

        bool[] vertexRisk = new bool[vertices.Count];
        for (int tri = 0; tri < triangleCount; tri++)
        {
            if (pureGeometryRisk[tri])
                riskMedium[tri] = true;
            if (!riskHigh[tri] && !riskMedium[tri])
                continue;

            LastTumorSuspectTriangleCount++;
            if (pendingRisk[tri])
                LastTumorPendingTriangleCount++;
            if (hitBlockedRisk[tri])
                LastTumorReplaceHitBlockedTriangleCount++;
            if (weightBlockedRisk[tri])
                LastTumorReplaceWeightBlockedTriangleCount++;
            if (replaceReadyRisk[tri])
                LastTumorReplaceReadyTriangleCount++;
            Color color = pureGeometryRisk[tri]
                ? pureGeometrySuspectColor
                : MeshTumorCauseColor(
                dirtyRisk[tri],
                nonManifoldRisk[tri],
                islandRisk[tri],
                openRisk[tri],
                longEdgeRisk[tri],
                riskHigh[tri]);
            int a = triangles[tri * 3];
            int b = triangles[tri * 3 + 1];
            int c = triangles[tri * 3 + 2];
            AddAuditSuspectMeshVoxel(triangleCenters[tri]);
            AddAuditSuspectMeshVoxel(vertices[a]);
            AddAuditSuspectMeshVoxel(vertices[b]);
            AddAuditSuspectMeshVoxel(vertices[c]);
            if (pureGeometryRisk[tri])
            {
                LastPureGeometrySuspectTriangleCount++;
                if (pureNormalRisk[tri])
                    LastPureGeometryNormalTriangleCount++;
                if (pureProtrusionRisk[tri])
                    LastPureGeometryProtrusionTriangleCount++;
                if (pureDoubleLayerRisk[tri])
                {
                    LastPureGeometryDoubleLayerTriangleCount++;
                    CountDoubleLayerSource(triangleCenters[tri], vertices[a], vertices[b], vertices[c]);
                }
                AddAuditPureGeometryMeshVoxel(triangleCenters[tri]);
                AddAuditPureGeometryMeshVoxel(vertices[a]);
                AddAuditPureGeometryMeshVoxel(vertices[b]);
                AddAuditPureGeometryMeshVoxel(vertices[c]);
            }
            PaintRiskVertex(colors, vertexRisk, a, color);
            PaintRiskVertex(colors, vertexRisk, b, color);
            PaintRiskVertex(colors, vertexRisk, c, color);
        }

        for (int i = 0; i < vertexRisk.Length; i++)
        {
            if (vertexRisk[i])
                LastTumorRiskVertexCount++;
        }
    }

    private void AddAuditSuspectMeshVoxel(Vector3 world)
    {
        if (TryWorldToVoxel(world, out int x, out int y, out int z))
        {
            int index = Index(x, y, z);
            if (_auditSuspectMeshVoxels.Add(index))
                CountSuspectMeshSource(index);
        }
    }

    private void CountSuspectMeshSource(int index)
    {
        if (!_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance) ||
            string.IsNullOrEmpty(provenance.LastOperation))
        {
            LastSuspectMeshSourceUnknownVoxelCount++;
            return;
        }

        switch (provenance.LastOperation)
        {
            case "integrate":
                LastSuspectMeshSourceIntegrateVoxelCount++;
                break;
            case "provisional_support":
                LastSuspectMeshSourceProvisionalVoxelCount++;
                break;
            case "strong_sample_seed":
                LastSuspectMeshSourceStrongSeedVoxelCount++;
                break;
            case "continuity_fill":
                LastSuspectMeshSourceContinuityVoxelCount++;
                break;
            case "free_space_carve":
                LastSuspectMeshSourceCarveVoxelCount++;
                break;
            case "replace":
            case "guarded_replace":
                LastSuspectMeshSourceReplaceVoxelCount++;
                break;
            case "band_repair":
            case "conflict_correct":
            case "cleanup_neighbor":
                LastSuspectMeshSourceRepairVoxelCount++;
                break;
            default:
                LastSuspectMeshSourceOtherVoxelCount++;
                break;
        }
    }

    private void AddAuditPureGeometryMeshVoxel(Vector3 world)
    {
        if (TryWorldToVoxel(world, out int x, out int y, out int z))
            _auditPureGeometryMeshVoxels.Add(Index(x, y, z));
    }

    private void CountDoubleLayerSource(Vector3 center, Vector3 a, Vector3 b, Vector3 c)
    {
        bool dirty = false;
        bool pending = false;
        bool bandConflict = false;
        bool oldLocked = false;
        bool cleanWeighted = false;
        bool emptyWeighted = false;
        bool anySample = false;
        DoubleLayerCleanEvidence cleanEvidence = new DoubleLayerCleanEvidence();

        AccumulateDoubleLayerSource(center, ref dirty, ref pending, ref bandConflict, ref oldLocked, ref cleanWeighted, ref emptyWeighted, ref anySample, ref cleanEvidence);
        AccumulateDoubleLayerSource(a, ref dirty, ref pending, ref bandConflict, ref oldLocked, ref cleanWeighted, ref emptyWeighted, ref anySample, ref cleanEvidence);
        AccumulateDoubleLayerSource(b, ref dirty, ref pending, ref bandConflict, ref oldLocked, ref cleanWeighted, ref emptyWeighted, ref anySample, ref cleanEvidence);
        AccumulateDoubleLayerSource(c, ref dirty, ref pending, ref bandConflict, ref oldLocked, ref cleanWeighted, ref emptyWeighted, ref anySample, ref cleanEvidence);

        int riskyKinds = 0;
        if (dirty) riskyKinds++;
        if (bandConflict) riskyKinds++;
        if (oldLocked) riskyKinds++;
        if (pending) riskyKinds++;

        if (riskyKinds > 1)
            LastDoubleLayerSourceMixedCount++;
        else if (dirty)
            LastDoubleLayerSourceDirtyCount++;
        else if (bandConflict)
            LastDoubleLayerSourceBandConflictCount++;
        else if (oldLocked)
            LastDoubleLayerSourceOldLockedCount++;
        else if (pending)
            LastDoubleLayerSourcePendingCount++;
        else if (cleanWeighted)
        {
            LastDoubleLayerSourceCleanCount++;
            CountDoubleLayerCleanEvidence(cleanEvidence);
        }
        else if (emptyWeighted || anySample)
            LastDoubleLayerSourceEmptyCount++;
        else
            LastDoubleLayerSourceUnknownCount++;
    }

    private struct DoubleLayerCleanEvidence
    {
        public bool NoLifecycle;
        public bool NoProvenance;
        public bool ConflictHistory;
        public bool ReplaceHistory;
        public bool ExpiredDirty;
        public bool LowWeight;
        public bool HighWeight;
        public bool LastIntegrate;
        public bool LastCarve;
        public bool LastReplace;
        public bool LastRepair;
        public bool IntegrateCarveHistory;
        public bool MultiFrameHistory;
    }

    private void CountDoubleLayerCleanEvidence(DoubleLayerCleanEvidence evidence)
    {
        if (evidence.NoLifecycle)
            LastDoubleLayerCleanNoLifecycleCount++;
        if (evidence.NoProvenance)
            LastDoubleLayerCleanNoProvenanceCount++;
        if (evidence.ConflictHistory)
            LastDoubleLayerCleanConflictHistoryCount++;
        if (evidence.ReplaceHistory)
            LastDoubleLayerCleanReplaceHistoryCount++;
        if (evidence.ExpiredDirty)
            LastDoubleLayerCleanExpiredDirtyCount++;
        if (evidence.LowWeight)
            LastDoubleLayerCleanLowWeightCount++;
        if (evidence.HighWeight)
            LastDoubleLayerCleanHighWeightCount++;
        if (evidence.LastIntegrate)
            LastDoubleLayerCleanLastIntegrateCount++;
        if (evidence.LastCarve)
            LastDoubleLayerCleanLastCarveCount++;
        if (evidence.LastReplace)
            LastDoubleLayerCleanLastReplaceCount++;
        if (evidence.LastRepair)
            LastDoubleLayerCleanLastRepairCount++;
        if (evidence.IntegrateCarveHistory)
            LastDoubleLayerCleanIntegrateCarveHistoryCount++;
        if (evidence.MultiFrameHistory)
            LastDoubleLayerCleanMultiFrameHistoryCount++;
    }

    private void AccumulateDoubleLayerSource(
        Vector3 world,
        ref bool dirty,
        ref bool pending,
        ref bool bandConflict,
        ref bool oldLocked,
        ref bool cleanWeighted,
        ref bool emptyWeighted,
        ref bool anySample,
        ref DoubleLayerCleanEvidence cleanEvidence)
    {
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(world, out int x, out int y, out int z))
            return;

        anySample = true;
        int radius = Mathf.Clamp(tumorSuspectTsdfRadiusVoxels, 0, 2);
        int weighted = 0;
        int risky = 0;
        for (int dz = -radius; dz <= radius; dz++)
        {
            int vz = z + dz;
            if (vz < 0 || vz >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int vy = y + dy;
                if (vy < 0 || vy >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    if (vx < 0 || vx >= _dimX)
                        continue;

                    int index = Index(vx, vy, vz);
                    int weight = _weights[index];
                    if (weight > 0)
                    {
                        weighted++;
                        if (weight <= Mathf.Max(1, minConflictVoxelWeight))
                            cleanEvidence.LowWeight = true;
                        if (weight > Mathf.Max(minConflictVoxelWeight, maxDirtyTsdfReplaceWeight))
                            cleanEvidence.HighWeight = true;
                        if (!_voxelWriteProvenance.TryGetValue(index, out VoxelWriteProvenance provenance))
                        {
                            cleanEvidence.NoProvenance = true;
                        }
                        else
                        {
                            if (provenance.LastOperation == "integrate")
                                cleanEvidence.LastIntegrate = true;
                            else if (provenance.LastOperation == "free_space_carve")
                                cleanEvidence.LastCarve = true;
                            else if (provenance.LastOperation == "replace" || provenance.LastOperation == "guarded_replace")
                                cleanEvidence.LastReplace = true;
                            else if (provenance.LastOperation == "band_repair" ||
                                     provenance.LastOperation == "conflict_correct" ||
                                     provenance.LastOperation == "cleanup_neighbor")
                                cleanEvidence.LastRepair = true;
                            if (provenance.IntegrateCount > 0 && provenance.CarveCount > 0)
                                cleanEvidence.IntegrateCarveHistory = true;
                            if (provenance.Frame > provenance.FirstFrame)
                                cleanEvidence.MultiFrameHistory = true;
                        }
                        if (!_voxelAuditLifecycle.TryGetValue(index, out VoxelAuditLifecycle history))
                        {
                            cleanEvidence.NoLifecycle = true;
                        }
                        else
                        {
                            if (history.ConflictCount > 0)
                                cleanEvidence.ConflictHistory = true;
                            if (history.ReplaceCount > 0)
                                cleanEvidence.ReplaceHistory = true;
                        }
                        if (_dirtyTsdfLastFrame != null &&
                            index < _dirtyTsdfLastFrame.Length &&
                            _dirtyTsdfLastFrame[index] != int.MinValue &&
                            !VoxelIsDirtyQuarantined(index))
                        {
                            cleanEvidence.ExpiredDirty = true;
                        }
                    }
                    if (VoxelIsDirtyQuarantined(index))
                    {
                        dirty = true;
                        risky++;
                    }
                    if (VoxelHasPendingTsdfCorrection(index))
                    {
                        pending = true;
                        risky++;
                    }
                    if (_voxelAuditLifecycle.TryGetValue(index, out VoxelAuditLifecycle life))
                    {
                        if (life.LastCause == "old_weight_locked")
                        {
                            oldLocked = true;
                            risky++;
                        }
                        if (life.LastLockSubtype == "band_conflict")
                        {
                            bandConflict = true;
                            risky++;
                        }
                    }
                }
            }
        }

        if (weighted > 0 && risky <= 0)
            cleanWeighted = true;
        if (weighted <= 0)
            emptyWeighted = true;
    }

    private void DetectPureGeometrySuspects(
        List<Vector3> vertices,
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        Vector3[] centers,
        Vector3[] normals,
        bool[] nonManifold,
        bool[] island,
        bool[] normalRisk,
        bool[] protrusionRisk,
        bool[] doubleLayerRisk,
        bool[] pureRisk)
    {
        int triangleCount = triangles.Count / 3;
        float normalDotLimit = Mathf.Cos(Mathf.Clamp(pureGeometryNormalDeviationDegrees, 10f, 80f) * Mathf.Deg2Rad);
        float parallelDot = Mathf.Clamp(pureGeometryParallelNormalDot, 0.75f, 1f);
        float planeOffset = Mathf.Max(0.005f, voxelSizeMeters * pureGeometryPlaneOffsetVoxelScale);
        List<int> neighbors = new List<int>(16);

        for (int tri = 0; tri < triangleCount; tri++)
        {
            if (normals[tri].sqrMagnitude < 0.0001f)
                continue;
            CollectAdjacentTriangles(triangles, edgeToTriangles, tri, neighbors);
            if (neighbors.Count >= 2)
            {
                Vector3 reference = normals[neighbors[0]].normalized;
                Vector3 normalSum = Vector3.zero;
                Vector3 centerSum = Vector3.zero;
                int valid = 0;
                for (int i = 0; i < neighbors.Count; i++)
                {
                    int other = neighbors[i];
                    Vector3 n = normals[other];
                    if (n.sqrMagnitude < 0.0001f)
                        continue;
                    n.Normalize();
                    if (Vector3.Dot(n, reference) < 0f)
                        n = -n;
                    normalSum += n;
                    centerSum += centers[other];
                    valid++;
                }

                if (valid >= 2 && normalSum.sqrMagnitude > 0.0001f)
                {
                    float neighborCoherence = normalSum.magnitude / valid;
                    Vector3 averageNormal = normalSum.normalized;
                    float currentDot = Mathf.Abs(Vector3.Dot(normals[tri].normalized, averageNormal));
                    if (neighborCoherence >= parallelDot && currentDot < normalDotLimit)
                        normalRisk[tri] = true;

                    Vector3 averageCenter = centerSum / valid;
                    float offset = Mathf.Abs(Vector3.Dot(centers[tri] - averageCenter, averageNormal));
                    if (neighborCoherence >= parallelDot && currentDot >= parallelDot && offset >= planeOffset)
                        protrusionRisk[tri] = true;
                }
            }

            pureRisk[tri] = nonManifold[tri] || island[tri] || normalRisk[tri] || protrusionRisk[tri];
        }

        Dictionary<int, List<int>> spatial = new Dictionary<int, List<int>>(triangleCount);
        for (int tri = 0; tri < triangleCount; tri++)
        {
            if (!TryWorldToVoxel(centers[tri], out int x, out int y, out int z))
                continue;
            int key = Index(x, y, z);
            if (!spatial.TryGetValue(key, out List<int> list))
            {
                list = new List<int>(4);
                spatial.Add(key, list);
            }
            list.Add(tri);
        }

        float minSeparation = Mathf.Max(0.005f, voxelSizeMeters * pureGeometryMinLayerSeparationVoxelScale);
        float maxSeparation = Mathf.Max(minSeparation, voxelSizeMeters * pureGeometryMaxLayerSeparationVoxelScale);
        float maxLateral = Mathf.Max(0.005f, voxelSizeMeters * pureGeometryMaxLayerLateralVoxelScale);
        int radius = Mathf.Clamp(Mathf.CeilToInt(pureGeometryMaxLayerSeparationVoxelScale), 1, 5);
        HashSet<ulong> recordedPairs = new HashSet<ulong>();
        HashSet<ulong> evaluatedPairs = new HashSet<ulong>();
        for (int tri = 0; tri < triangleCount; tri++)
        {
            if (!TryWorldToVoxel(centers[tri], out int cx, out int cy, out int cz) || normals[tri].sqrMagnitude < 0.0001f)
                continue;
            Vector3 normal = normals[tri].normalized;
            bool found = false;
            for (int dz = -radius; dz <= radius && !found; dz++)
            for (int dy = -radius; dy <= radius && !found; dy++)
            for (int dx = -radius; dx <= radius && !found; dx++)
            {
                if (dx * dx + dy * dy + dz * dz > radius * radius)
                    continue;
                int x = cx + dx;
                int y = cy + dy;
                int z = cz + dz;
                if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                    continue;
                if (!spatial.TryGetValue(Index(x, y, z), out List<int> candidates))
                    continue;
                for (int i = 0; i < candidates.Count; i++)
                {
                    int other = candidates[i];
                    if (other == tri || normals[other].sqrMagnitude < 0.0001f)
                        continue;
                    if (Mathf.Abs(Vector3.Dot(normal, normals[other].normalized)) < parallelDot)
                        continue;
                    Vector3 delta = centers[other] - centers[tri];
                    float separation = Mathf.Abs(Vector3.Dot(delta, normal));
                    if (separation < minSeparation || separation > maxSeparation)
                        continue;
                    float lateralSq = Mathf.Max(0f, delta.sqrMagnitude - separation * separation);
                    if (lateralSq > maxLateral * maxLateral)
                        continue;
                    int pairA = Mathf.Min(tri, other);
                    int pairB = Mathf.Max(tri, other);
                    ulong pairKey = ((ulong)(uint)pairA << 32) | (uint)pairB;
                    if (!evaluatedPairs.Add(pairKey))
                        continue;
                    if (!ValidateDoubleLayerPairSources(
                            centers[pairA],
                            centers[pairB],
                            normals[pairA],
                            normals[pairB]))
                    {
                        continue;
                    }
                    doubleLayerRisk[tri] = true;
                    doubleLayerRisk[other] = true;
                    pureRisk[tri] = true;
                    pureRisk[other] = true;
                    if (recordedPairs.Add(pairKey))
                    {
                        AppendDoubleLayerPairAudit(
                            pairA,
                            pairB,
                            centers[pairA],
                            centers[pairB],
                            normals[pairA],
                            normals[pairB],
                            separation,
                            Mathf.Sqrt(lateralSq));
                    }
                    found = true;
                    break;
                }
            }
        }
    }

    private bool ValidateDoubleLayerPairSources(
        Vector3 centerA,
        Vector3 centerB,
        Vector3 meshNormalA,
        Vector3 meshNormalB)
    {
        if (!validateDoubleLayerWithWriteProvenance)
        {
            LastDoubleLayerValidatedPairCount++;
            return true;
        }
        if (!TryResolveDoubleLayerSide(centerA, out DoubleLayerSideAudit sideA) ||
            !TryResolveDoubleLayerSide(centerB, out DoubleLayerSideAudit sideB) ||
            !sideA.HasProvenance ||
            !sideB.HasProvenance)
        {
            LastDoubleLayerRejectedMissingSourceCount++;
            return false;
        }

        Vector3 sourceNormalA = sideA.Provenance.SurfaceNormal;
        Vector3 sourceNormalB = sideB.Provenance.SurfaceNormal;
        if (sourceNormalA.sqrMagnitude < 0.0001f || sourceNormalB.sqrMagnitude < 0.0001f)
        {
            LastDoubleLayerRejectedSourceNormalCount++;
            return false;
        }
        sourceNormalA.Normalize();
        sourceNormalB.Normalize();
        float sourceNormalDot = Vector3.Dot(sourceNormalA, sourceNormalB);
        if (Mathf.Abs(sourceNormalDot) <
            Mathf.Cos(Mathf.Clamp(doubleLayerMaxSourceNormalAngleDegrees, 1f, 60f) * Mathf.Deg2Rad))
        {
            LastDoubleLayerRejectedSourceNormalCount++;
            return false;
        }
        if (sourceNormalDot < 0f)
            sourceNormalB = -sourceNormalB;
        Vector3 sourceNormal = sourceNormalA + sourceNormalB;
        if (sourceNormal.sqrMagnitude < 0.0001f)
        {
            LastDoubleLayerRejectedSourceNormalCount++;
            return false;
        }
        sourceNormal.Normalize();

        Vector3 sourceDelta = sideB.Provenance.SurfacePoint - sideA.Provenance.SurfacePoint;
        float sourcePlaneSeparation = Mathf.Abs(Vector3.Dot(sourceDelta, sourceNormal));
        if (sourcePlaneSeparation < Mathf.Max(0.01f, doubleLayerMinSourcePlaneSeparationMeters))
        {
            LastDoubleLayerRejectedSourcePlaneCount++;
            return false;
        }
        float sourceLateralSq = Mathf.Max(
            0f,
            sourceDelta.sqrMagnitude - sourcePlaneSeparation * sourcePlaneSeparation);
        float maxSourceLateral = Mathf.Max(
            voxelSizeMeters * 0.25f,
            voxelSizeMeters * doubleLayerMaxSourceLateralVoxelScale);
        if (sourceLateralSq > maxSourceLateral * maxSourceLateral)
        {
            LastDoubleLayerRejectedSourceLateralCount++;
            return false;
        }

        Vector3 meshNormal = meshNormalA.normalized;
        Vector3 alignedMeshB = meshNormalB.normalized;
        if (Vector3.Dot(meshNormal, alignedMeshB) < 0f)
            alignedMeshB = -alignedMeshB;
        meshNormal = (meshNormal + alignedMeshB).normalized;
        if (meshNormal.sqrMagnitude < 0.0001f ||
            Mathf.Abs(Vector3.Dot(meshNormal, sourceNormal)) <
            Mathf.Clamp01(doubleLayerMinMeshSourceNormalDot))
        {
            LastDoubleLayerMeshAlignmentMismatchCount++;
        }

        LastDoubleLayerValidatedPairCount++;
        return true;
    }

    private struct DoubleLayerSideAudit
    {
        public bool Found;
        public int Index;
        public int X;
        public int Y;
        public int Z;
        public float Tsdf;
        public int Weight;
        public bool Dirty;
        public bool Pending;
        public bool HasProvenance;
        public VoxelWriteProvenance Provenance;
    }

    private void AppendDoubleLayerPairAudit(
        int triangleA,
        int triangleB,
        Vector3 centerA,
        Vector3 centerB,
        Vector3 normalA,
        Vector3 normalB,
        float separation,
        float lateral)
    {
        if (_doubleLayerPairRows == null)
            return;
        if (_doubleLayerPairRowCount >= Mathf.Max(1000, maxConfidenceAuditRowsPerCapture))
        {
            _doubleLayerPairDroppedRows++;
            return;
        }

        TryResolveDoubleLayerSide(centerA, out DoubleLayerSideAudit sideA);
        TryResolveDoubleLayerSide(centerB, out DoubleLayerSideAudit sideB);
        VoxelWriteProvenance provenanceA = sideA.Provenance;
        VoxelWriteProvenance provenanceB = sideB.Provenance;
        bool bothProvenance = sideA.HasProvenance && sideB.HasProvenance;
        int frameGap = bothProvenance ? Mathf.Abs(provenanceA.Frame - provenanceB.Frame) : -1;
        float depthDelta = bothProvenance ? Mathf.Abs(provenanceA.SurfaceDepth - provenanceB.SurfaceDepth) : -1f;
        float rayAngle = bothProvenance &&
                         provenanceA.RayDirection.sqrMagnitude > 0.0001f &&
                         provenanceB.RayDirection.sqrMagnitude > 0.0001f
            ? Vector3.Angle(provenanceA.RayDirection, provenanceB.RayDirection)
            : -1f;

        StringBuilder row = _doubleLayerPairRows;
        row.Append(_confidenceAuditCaptureIndex).Append(',')
            .Append(LastRawFrameIndex).Append(',')
            .Append(triangleA).Append(',')
            .Append(triangleB).Append(',')
            .Append(AuditFloat(separation)).Append(',')
            .Append(AuditFloat(lateral)).Append(',')
            .Append(AuditFloat(Mathf.Abs(Vector3.Dot(normalA.normalized, normalB.normalized)))).Append(',');
        AppendDoubleLayerSideAudit(row, centerA, sideA);
        row.Append(',');
        AppendDoubleLayerSideAudit(row, centerB, sideB);
        row.Append(',')
            .Append(bothProvenance && provenanceA.Frame == provenanceB.Frame ? 1 : 0).Append(',')
            .Append(bothProvenance && provenanceA.Sample == provenanceB.Sample ? 1 : 0).Append(',')
            .Append(frameGap).Append(',')
            .Append(AuditFloat(depthDelta)).Append(',')
            .Append(AuditFloat(rayAngle)).Append(',')
            .Append(bothProvenance && provenanceA.IntegrateCount > 0 && provenanceB.IntegrateCount > 0 ? 1 : 0).Append(',')
            .Append(bothProvenance && (provenanceA.CarveCount > 0 || provenanceB.CarveCount > 0) ? 1 : 0)
            .AppendLine();
        _doubleLayerPairRowCount++;
    }

    private bool TryResolveDoubleLayerSide(Vector3 world, out DoubleLayerSideAudit side)
    {
        side = new DoubleLayerSideAudit();
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(world, out int cx, out int cy, out int cz))
            return false;

        int radius = Mathf.Clamp(tumorSuspectTsdfRadiusVoxels, 1, 2);
        float bestScore = float.PositiveInfinity;
        for (int dz = -radius; dz <= radius; dz++)
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = cx + dx;
            int y = cy + dy;
            int z = cz + dz;
            if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;
            int index = Index(x, y, z);
            int weight = _weights[index];
            if (weight <= 0)
                continue;
            float distanceSq = (VoxelCenter(x, y, z) - world).sqrMagnitude;
            float score = Mathf.Abs(_tsdf[index]) * voxelSizeMeters * voxelSizeMeters + distanceSq;
            if (score >= bestScore)
                continue;

            bestScore = score;
            side.Found = true;
            side.Index = index;
            side.X = x;
            side.Y = y;
            side.Z = z;
            side.Tsdf = _tsdf[index];
            side.Weight = weight;
            side.Dirty = VoxelIsDirtyQuarantined(index);
            side.Pending = VoxelHasPendingTsdfCorrection(index);
            side.HasProvenance = _voxelWriteProvenance.TryGetValue(index, out side.Provenance);
        }
        return side.Found;
    }

    private static void AppendDoubleLayerSideAudit(StringBuilder row, Vector3 center, DoubleLayerSideAudit side)
    {
        VoxelWriteProvenance provenance = side.Provenance;
        row.Append(AuditFloat(center.x)).Append(',')
            .Append(AuditFloat(center.y)).Append(',')
            .Append(AuditFloat(center.z)).Append(',')
            .Append(side.Found ? side.Index : -1).Append(',')
            .Append(side.Found ? side.X : -1).Append(',')
            .Append(side.Found ? side.Y : -1).Append(',')
            .Append(side.Found ? side.Z : -1).Append(',')
            .Append(side.Found ? AuditFloat(side.Tsdf) : "-1").Append(',')
            .Append(side.Found ? side.Weight : 0).Append(',')
            .Append(side.HasProvenance ? provenance.FirstFrame : -1).Append(',')
            .Append(side.HasProvenance ? provenance.Frame : -1).Append(',')
            .Append(side.HasProvenance ? provenance.Capture : -1).Append(',')
            .Append(side.HasProvenance ? provenance.Sample : -1).Append(',')
            .Append(side.HasProvenance ? provenance.PixelX : -1).Append(',')
            .Append(side.HasProvenance ? provenance.PixelY : -1).Append(',')
            .Append(side.HasProvenance ? provenance.LastOperation : "untracked").Append(',')
            .Append(side.HasProvenance ? provenance.WriteCount : 0).Append(',')
            .Append(side.HasProvenance ? provenance.IntegrateCount : 0).Append(',')
            .Append(side.HasProvenance ? provenance.CarveCount : 0).Append(',')
            .Append(side.HasProvenance ? provenance.ReplaceCount : 0).Append(',')
            .Append(side.HasProvenance ? provenance.RepairCount : 0).Append(',')
            .Append(side.HasProvenance ? AuditFloat(provenance.SurfaceDepth) : "-1").Append(',')
            .Append(side.HasProvenance ? AuditFloat(provenance.RayDirection.x) : "-1").Append(',')
            .Append(side.HasProvenance ? AuditFloat(provenance.RayDirection.y) : "-1").Append(',')
            .Append(side.HasProvenance ? AuditFloat(provenance.RayDirection.z) : "-1").Append(',')
            .Append(side.Dirty ? 1 : 0).Append(',')
            .Append(side.Pending ? 1 : 0);
    }

    private static void CollectAdjacentTriangles(
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        int tri,
        List<int> neighbors)
    {
        neighbors.Clear();
        int a = triangles[tri * 3];
        int b = triangles[tri * 3 + 1];
        int c = triangles[tri * 3 + 2];
        CollectEdgeAdjacent(edgeToTriangles, EdgeKey(a, b), tri, neighbors);
        CollectEdgeAdjacent(edgeToTriangles, EdgeKey(b, c), tri, neighbors);
        CollectEdgeAdjacent(edgeToTriangles, EdgeKey(c, a), tri, neighbors);
    }

    private static void CollectEdgeAdjacent(
        Dictionary<ulong, List<int>> edgeToTriangles,
        ulong edge,
        int tri,
        List<int> neighbors)
    {
        if (!edgeToTriangles.TryGetValue(edge, out List<int> linked))
            return;
        for (int i = 0; i < linked.Count; i++)
        {
            int other = linked[i];
            if (other != tri && !neighbors.Contains(other))
                neighbors.Add(other);
        }
    }

    private void MarkSmallOpenMeshIslands(
        List<Vector3> vertices,
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        Vector3[] triangleCenters,
        int[] boundaryEdgesPerTri,
        bool[] riskHigh,
        bool[] riskMedium,
        bool[] islandRisk)
    {
        int triangleCount = triangles.Count / 3;
        bool[] visited = new bool[triangleCount];
        Queue<int> queue = new Queue<int>();
        List<int> component = new List<int>(256);
        int maxIslandTriangles = Mathf.Max(4, tumorSuspectSmallIslandMaxTriangles);
        float extentLimit = Mathf.Max(0.02f, voxelSizeMeters * tumorSuspectSmallIslandExtentVoxelScale);
        float boundaryLimit = Mathf.Clamp01(tumorSuspectOpenBoundaryRatio);

        for (int start = 0; start < triangleCount; start++)
        {
            if (visited[start])
                continue;

            component.Clear();
            visited[start] = true;
            queue.Enqueue(start);
            Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
            Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);
            int edgeTotal = 0;
            int boundaryTotal = 0;

            while (queue.Count > 0)
            {
                int tri = queue.Dequeue();
                component.Add(tri);
                edgeTotal += 3;
                boundaryTotal += tri < boundaryEdgesPerTri.Length ? boundaryEdgesPerTri[tri] : 0;

                int a = triangles[tri * 3];
                int b = triangles[tri * 3 + 1];
                int c = triangles[tri * 3 + 2];
                if (TriangleIndicesValid(vertices, a, b, c))
                {
                    min = Vector3.Min(min, vertices[a]);
                    min = Vector3.Min(min, vertices[b]);
                    min = Vector3.Min(min, vertices[c]);
                    max = Vector3.Max(max, vertices[a]);
                    max = Vector3.Max(max, vertices[b]);
                    max = Vector3.Max(max, vertices[c]);
                }

                EnqueueTriNeighbors(triangles, edgeToTriangles, tri, visited, queue);
            }

            if (component.Count <= 0 || edgeTotal <= 0 || !Finite(min) || !Finite(max))
                continue;

            Vector3 extent = max - min;
            float maxExtent = Mathf.Max(extent.x, Mathf.Max(extent.y, extent.z));
            float boundaryRatio = (float)boundaryTotal / edgeTotal;
            bool smallOpenIsland = component.Count <= maxIslandTriangles &&
                                   maxExtent <= extentLimit &&
                                   boundaryRatio >= boundaryLimit;
            if (!smallOpenIsland)
                continue;

            LastTumorIslandTriangleCount += component.Count;
            AnalyzeIslandBoundaryCauses(component, vertices, triangles, triangleCenters, boundaryEdgesPerTri);
            for (int i = 0; i < component.Count; i++)
            {
                int tri = component[i];
                islandRisk[tri] = true;
                bool unsafeTsdf = MeshPointTouchesUnsafeTsdf(triangleCenters[tri]);
                if (unsafeTsdf && boundaryRatio >= Mathf.Min(0.9f, boundaryLimit + 0.18f))
                    riskHigh[tri] = true;
                else
                    riskMedium[tri] = true;
            }
        }
    }

    private void AnalyzeIslandBoundaryCauses(
        List<int> component,
        List<Vector3> vertices,
        List<int> triangles,
        Vector3[] triangleCenters,
        int[] boundaryEdgesPerTri)
    {
        if (component == null || component.Count <= 0 || vertices == null || triangles == null)
            return;

        LastIslandCauseComponentCount++;
        HashSet<int> sampled = new HashSet<int>();
        int supportedCleanNearSurface = 0;
        int blockedOrMissing = 0;

        for (int i = 0; i < component.Count; i++)
        {
            int tri = component[i];
            if (tri < 0 || tri * 3 + 2 >= triangles.Count)
                continue;
            if (boundaryEdgesPerTri != null && tri < boundaryEdgesPerTri.Length && boundaryEdgesPerTri[tri] <= 0)
                continue;

            CountIslandBoundaryCauseSample(triangleCenters != null && tri < triangleCenters.Length ? triangleCenters[tri] : Vector3.zero, sampled, ref supportedCleanNearSurface, ref blockedOrMissing);

            int a = triangles[tri * 3];
            int b = triangles[tri * 3 + 1];
            int c = triangles[tri * 3 + 2];
            if (!TriangleIndicesValid(vertices, a, b, c))
                continue;

            CountIslandBoundaryCauseSample(vertices[a], sampled, ref supportedCleanNearSurface, ref blockedOrMissing);
            CountIslandBoundaryCauseSample(vertices[b], sampled, ref supportedCleanNearSurface, ref blockedOrMissing);
            CountIslandBoundaryCauseSample(vertices[c], sampled, ref supportedCleanNearSurface, ref blockedOrMissing);
        }

        LastIslandCauseBoundaryVoxelCount += sampled.Count;
        if (sampled.Count <= 0 || supportedCleanNearSurface >= Mathf.Max(1, blockedOrMissing))
            LastIslandCausePrunedCount += component.Count;
    }

    private void CountIslandBoundaryCauseSample(
        Vector3 world,
        HashSet<int> sampled,
        ref int supportedCleanNearSurface,
        ref int blockedOrMissing)
    {
        if (!Finite(world) || sampled == null)
            return;

        if (_tsdf == null || _weights == null || !TryWorldToVoxel(world, out int cx, out int cy, out int cz))
        {
            LastIslandCauseNoTsdfCount++;
            blockedOrMissing++;
            return;
        }

        int radius = Mathf.Clamp(tumorSuspectTsdfRadiusVoxels, 0, 2);
        float nearSurface = Mathf.Max(0.02f, maxCleanLightCoverAbsTsdf);
        for (int dz = -radius; dz <= radius; dz++)
        {
            int z = cz + dz;
            if (z < 0 || z >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int y = cy + dy;
                if (y < 0 || y >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int x = cx + dx;
                    if (x < 0 || x >= _dimX)
                        continue;

                    int index = Index(x, y, z);
                    if (!sampled.Add(index))
                        continue;

                    int weight = _weights[index];
                    if (VoxelIsDirtyQuarantined(index))
                    {
                        LastIslandCauseDirtyCount++;
                        blockedOrMissing++;
                    }
                    else if (VoxelHasPendingTsdfCorrection(index))
                    {
                        LastIslandCausePendingCount++;
                        blockedOrMissing++;
                    }
                    else if (weight <= 0)
                    {
                        LastIslandCauseNoTsdfCount++;
                        blockedOrMissing++;
                    }
                    else if (weight < Mathf.Max(1, minSurfaceCornerWeight))
                    {
                        LastIslandCauseLowWeightCount++;
                        blockedOrMissing++;
                    }
                    else if (Mathf.Abs(_tsdf[index]) > nearSurface)
                    {
                        LastIslandCausePlaneMismatchCount++;
                        blockedOrMissing++;
                    }
                    else
                    {
                        supportedCleanNearSurface++;
                    }
                }
            }
        }
    }

    private Color MeshTumorCauseColor(bool dirty, bool nonManifold, bool island, bool open, bool longEdge, bool high)
    {
        if (dirty)
            return tumorCauseDirtyColor;
        if (nonManifold)
            return tumorCauseNonManifoldColor;
        if (island)
            return tumorCauseIslandColor;
        if (longEdge)
            return tumorCauseLongEdgeColor;
        if (open)
            return tumorCauseOpenColor;
        return high ? tumorSuspectHighColor : tumorSuspectMediumColor;
    }

    private bool MeshPointTouchesUnsafeTsdf(Vector3 world)
    {
        if (_tsdf == null || _weights == null || !TryWorldToVoxel(world, out int x, out int y, out int z))
            return false;

        int radius = Mathf.Clamp(tumorSuspectTsdfRadiusVoxels, 0, 2);
        for (int dz = -radius; dz <= radius; dz++)
        {
            int vz = z + dz;
            if (vz < 0 || vz >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int vy = y + dy;
                if (vy < 0 || vy >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    if (vx < 0 || vx >= _dimX)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (VoxelIsDirtyQuarantined(index) || VoxelHasPendingTsdfCorrection(index))
                        return true;
                }
            }
        }

        return false;
    }

    private void AccumulateMeshTsdfRepairState(
        Vector3 world,
        ref bool dirty,
        ref bool pending,
        ref bool hitBlocked,
        ref bool weightBlocked,
        ref bool replaceReady)
    {
        if (_tsdf == null ||
            _weights == null ||
            _correctionTsdfHits == null ||
            !TryWorldToVoxel(world, out int x, out int y, out int z))
        {
            return;
        }

        int radius = Mathf.Clamp(tumorSuspectTsdfRadiusVoxels, 0, 2);
        int minHits = Mathf.Max(1, minDirtyTsdfReplaceFrames);
        int maxWeight = Mathf.Max(1, maxDirtyTsdfReplaceWeight);
        for (int dz = -radius; dz <= radius; dz++)
        {
            int vz = z + dz;
            if (vz < 0 || vz >= _dimZ)
                continue;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int vy = y + dy;
                if (vy < 0 || vy >= _dimY)
                    continue;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int vx = x + dx;
                    if (vx < 0 || vx >= _dimX)
                        continue;

                    int index = Index(vx, vy, vz);
                    if (VoxelIsDirtyQuarantined(index))
                        dirty = true;

                    int hits = PendingTsdfCorrectionHits(index);
                    if (hits <= 0)
                        continue;

                    pending = true;
                    int weight = _weights[index];
                    if (!replaceDirtyTsdfOnStableConflict)
                        continue;
                    if (weight > maxWeight)
                    {
                        weightBlocked = true;
                        continue;
                    }
                    if (hits < minHits)
                        hitBlocked = true;
                    else
                        replaceReady = true;
                }
            }
        }
    }

    private static bool TriangleIndicesValid(List<Vector3> vertices, int a, int b, int c)
    {
        return vertices != null &&
               a >= 0 && b >= 0 && c >= 0 &&
               a < vertices.Count && b < vertices.Count && c < vertices.Count;
    }

    private static void PaintRiskVertex(List<Color> colors, bool[] vertexRisk, int index, Color color)
    {
        if (colors == null || vertexRisk == null || index < 0 || index >= colors.Count || index >= vertexRisk.Length)
            return;

        if (!vertexRisk[index] || RiskColorRank(color) > RiskColorRank(colors[index]))
            colors[index] = color;
        vertexRisk[index] = true;
    }

    private static int RiskColorRank(Color color)
    {
        if (color.r > 0.9f && color.g < 0.2f)
            return 5;
        if (color.b > 0.8f && color.r > 0.5f && color.g < 0.35f)
            return 4;
        if (color.r > 0.9f && color.g > 0.35f && color.g < 0.8f)
            return 3;
        if (color.b > 0.8f && color.g > 0.6f)
            return 2;
        if (color.r > 0.9f && color.g >= 0.8f)
            return 2;
        return 1;
    }

    private void AnalyzeMeshComponents(List<int> triangles, Dictionary<ulong, List<int>> edgeToTriangles, int triangleCount)
    {
        bool[] visited = new bool[triangleCount];
        Queue<int> queue = new Queue<int>();
        for (int start = 0; start < triangleCount; start++)
        {
            if (visited[start])
                continue;
            int count = 0;
            visited[start] = true;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int tri = queue.Dequeue();
                count++;
                EnqueueTriNeighbors(triangles, edgeToTriangles, tri, visited, queue);
            }

            LastDiagMeshComponentCount++;
            if (count > LastDiagLargestComponentTriangles)
                LastDiagLargestComponentTriangles = count;
        }
    }

    private void AnalyzeNormalPatches(
        List<Vector3> vertices,
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        Vector3[] triangleNormals,
        Vector3[] triangleCenters,
        int triangleCount)
    {
        bool[] visited = new bool[triangleCount];
        Queue<int> queue = new Queue<int>();
        List<int> patch = new List<int>(256);
        float minDot = Mathf.Clamp(diagnosticNormalPatchMinDot, -1f, 1f);

        for (int start = 0; start < triangleCount; start++)
        {
            if (visited[start])
                continue;

            patch.Clear();
            visited[start] = true;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int tri = queue.Dequeue();
                patch.Add(tri);
                EnqueueSimilarNormalTriNeighbors(triangles, edgeToTriangles, triangleNormals, tri, minDot, visited, queue);
            }

            LastDiagNormalPatchCount++;
            EvaluateDiagnosticPatch(vertices, triangles, edgeToTriangles, triangleNormals, triangleCenters, patch);
        }
    }

    private void EvaluateDiagnosticPatch(
        List<Vector3> vertices,
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        Vector3[] triangleNormals,
        Vector3[] triangleCenters,
        List<int> patch)
    {
        if (patch.Count <= 0)
            return;

        HashSet<int> patchSet = new HashSet<int>(patch);
        HashSet<int> vertexSet = new HashSet<int>();
        Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);
        Vector3 centerSum = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        int edgeCount = 0;
        int boundaryEdges = 0;

        for (int i = 0; i < patch.Count; i++)
        {
            int tri = patch[i];
            int a = triangles[tri * 3];
            int b = triangles[tri * 3 + 1];
            int c = triangles[tri * 3 + 2];
            AddPatchVertex(vertices, vertexSet, a, ref min, ref max);
            AddPatchVertex(vertices, vertexSet, b, ref min, ref max);
            AddPatchVertex(vertices, vertexSet, c, ref min, ref max);
            CountPatchEdge(edgeToTriangles, patchSet, a, b, ref edgeCount, ref boundaryEdges);
            CountPatchEdge(edgeToTriangles, patchSet, b, c, ref edgeCount, ref boundaryEdges);
            CountPatchEdge(edgeToTriangles, patchSet, c, a, ref edgeCount, ref boundaryEdges);
            centerSum += triangleCenters[tri];
            normalSum += triangleNormals[tri];
        }

        if (edgeCount <= 0)
            return;

        float boundaryRatio = (float)boundaryEdges / edgeCount;
        Vector3 extent = max - min;
        float maxExtent = Mathf.Max(extent.x, Mathf.Max(extent.y, extent.z));
        float extentLimit = Mathf.Max(0.01f, voxelSizeMeters * Mathf.Clamp(diagnosticSuspectExtentVoxelScale, 1f, 12f));
        bool compact = maxExtent <= extentLimit;
        bool smallEnough = patch.Count <= Mathf.Max(4, diagnosticSuspectPatchMaxTriangles);
        bool open = boundaryRatio >= diagnosticSuspectBoundaryRatio;
        if (!smallEnough || !open || !compact)
            return;

        MeshDiagnosticHotspot hotspot = new MeshDiagnosticHotspot
        {
            Center = centerSum / Mathf.Max(1, patch.Count),
            Normal = normalSum.sqrMagnitude > 0.0001f ? normalSum.normalized : Vector3.up,
            BoundaryRatio = boundaryRatio,
            ExtentMeters = maxExtent,
            Triangles = patch.Count,
            Score = boundaryRatio * 2f + (1f - Mathf.Clamp01(maxExtent / extentLimit)) + Mathf.Clamp01(1f - (float)patch.Count / Mathf.Max(1, diagnosticSuspectPatchMaxTriangles)),
            Reason = "open-normal-patch"
        };

        LastDiagSuspectPatchCount++;
        if (hotspot.Score > LastDiagWorstPatchScore)
        {
            LastDiagWorstPatchScore = hotspot.Score;
            LastDiagWorstPatchTriangles = hotspot.Triangles;
            LastDiagWorstPatchBoundaryRatio = hotspot.BoundaryRatio;
            LastDiagWorstPatchExtentMeters = hotspot.ExtentMeters;
        }
        _meshDiagnosticHotspots.Add(hotspot);
    }

    private string GuessMeshDiagnosticCause(int triangleCount)
    {
        if (LastDiagSuspectPatchCount > 0 && LastDiagBoundaryEdgeCount > triangleCount / 3)
            return "OPEN SPIKE PATCHES";
        if (LastDiagNonManifoldEdgeCount > 0)
            return "NON-MANIFOLD MESH";
        if (LastDiagMeshComponentCount > 1)
            return "SEPARATE COMPONENTS";
        if (LastCorrectedTsdfCount > 0 || LastPendingTsdfCorrectionCount > LastRejectedTsdfConflictCount)
            return "OLD TSDF CONFLICT";
        if (LastPendingStableTsdfCount > LastUpdatedVoxelCount / 3)
            return "UNSTABLE TSDF WRITES";
        return "NO CLEAR MESH HOTSPOT";
    }

    private static Vector3 ComputeTriangleCenter(List<Vector3> vertices, List<int> triangles, int tri)
    {
        int baseIndex = tri * 3;
        if (baseIndex + 2 >= triangles.Count)
            return Vector3.zero;
        int ia = triangles[baseIndex];
        int ib = triangles[baseIndex + 1];
        int ic = triangles[baseIndex + 2];
        if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
            return Vector3.zero;
        return (vertices[ia] + vertices[ib] + vertices[ic]) / 3f;
    }

    private void SortMeshDiagnosticHotspots()
    {
        for (int i = 1; i < _meshDiagnosticHotspots.Count; i++)
        {
            MeshDiagnosticHotspot value = _meshDiagnosticHotspots[i];
            int j = i - 1;
            while (j >= 0 && _meshDiagnosticHotspots[j].Score < value.Score)
            {
                _meshDiagnosticHotspots[j + 1] = _meshDiagnosticHotspots[j];
                j--;
            }
            _meshDiagnosticHotspots[j + 1] = value;
        }
    }

    private void UpdateMeshDiagnosticHotspots()
    {
        if (!showMeshDiagnosticHotspots || _meshDiagnosticHotspots.Count <= 0)
        {
            SetMeshDiagnosticHotspotsVisible(0);
            return;
        }

        EnsureMeshDiagnosticHotspotObjects();
        int count = Mathf.Min(Mathf.Max(1, maxDiagnosticHotspots), _meshDiagnosticHotspots.Count);
        SetMeshDiagnosticHotspotsVisible(count);
        float size = Mathf.Max(0.005f, diagnosticHotspotMarkerSizeMeters);
        for (int i = 0; i < count; i++)
        {
            MeshDiagnosticHotspot hotspot = _meshDiagnosticHotspots[i];
            GameObject marker = _diagnosticHotspotMarkers[i];
            marker.transform.position = hotspot.Center + hotspot.Normal * size * 0.35f;
            marker.transform.rotation = Quaternion.identity;
            marker.transform.localScale = Vector3.one * size * (i == 0 ? 1.25f : 0.9f);
            Renderer renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
                renderer.sharedMaterial = i == 0 ? ResolveMeshDiagnosticHotspotMaterial(true) : ResolveMeshDiagnosticHotspotMaterial(false);
        }
    }

    private void EnsureMeshDiagnosticHotspotObjects()
    {
        if (_diagnosticHotspotRoot == null)
        {
            _diagnosticHotspotRoot = new GameObject("[ScanCover] Mesh Diagnostic Hotspots");
            _diagnosticHotspotRoot.hideFlags = HideFlags.DontSave;
        }

        if (forceWorldSpaceDisplay)
            _diagnosticHotspotRoot.transform.SetParent(null, true);
        else if (_diagnosticHotspotRoot.transform.parent != displayRoot)
            _diagnosticHotspotRoot.transform.SetParent(displayRoot, false);

        int target = Mathf.Max(1, maxDiagnosticHotspots);
        while (_diagnosticHotspotMarkers.Count < target)
        {
            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.name = $"MeshDiagHotspot_{_diagnosticHotspotMarkers.Count}";
            marker.hideFlags = HideFlags.DontSave;
            marker.transform.SetParent(_diagnosticHotspotRoot.transform, true);
            Collider collider = marker.GetComponent<Collider>();
            if (collider != null)
                Destroy(collider);
            _diagnosticHotspotMarkers.Add(marker);
        }
    }

    private void SetMeshDiagnosticHotspotsVisible(int visibleCount)
    {
        for (int i = 0; i < _diagnosticHotspotMarkers.Count; i++)
        {
            GameObject marker = _diagnosticHotspotMarkers[i];
            bool visible = i < visibleCount;
            if (marker != null && marker.activeSelf != visible)
                marker.SetActive(visible);
        }
    }

    private Material ResolveMeshDiagnosticHotspotMaterial(bool primary)
    {
        Material material = primary ? _diagnosticHotspotMaterial : _diagnosticSecondaryHotspotMaterial;
        if (material == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");
            material = new Material(shader);
            material.name = primary ? "ScanCover_MeshDiagnosticHotspot_Primary" : "ScanCover_MeshDiagnosticHotspot_Secondary";
            if (primary)
                _diagnosticHotspotMaterial = material;
            else
                _diagnosticSecondaryHotspotMaterial = material;
        }

        ApplyMaterialColor(material, primary ? diagnosticHotspotColor : diagnosticSecondaryHotspotColor);
        ConfigureTransparentMaterial(material);
        material.renderQueue = (int)RenderQueue.Transparent + 60;
        return material;
    }

    private bool IsSpikePatch(
        List<Vector3> vertices,
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        List<int> patch)
    {
        HashSet<int> patchSet = new HashSet<int>(patch);
        HashSet<int> vertexSet = new HashSet<int>();
        int edgeCount = 0;
        int boundaryEdges = 0;
        Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

        for (int i = 0; i < patch.Count; i++)
        {
            int tri = patch[i];
            int a = triangles[tri * 3];
            int b = triangles[tri * 3 + 1];
            int c = triangles[tri * 3 + 2];
            AddPatchVertex(vertices, vertexSet, a, ref min, ref max);
            AddPatchVertex(vertices, vertexSet, b, ref min, ref max);
            AddPatchVertex(vertices, vertexSet, c, ref min, ref max);
            CountPatchEdge(edgeToTriangles, patchSet, a, b, ref edgeCount, ref boundaryEdges);
            CountPatchEdge(edgeToTriangles, patchSet, b, c, ref edgeCount, ref boundaryEdges);
            CountPatchEdge(edgeToTriangles, patchSet, c, a, ref edgeCount, ref boundaryEdges);
        }

        if (edgeCount <= 0 || vertexSet.Count <= 0)
            return false;

        float boundaryRatio = (float)boundaryEdges / edgeCount;
        Vector3 extent = max - min;
        float maxExtent = Mathf.Max(extent.x, Mathf.Max(extent.y, extent.z));
        float extentLimit = Mathf.Max(0.01f, voxelSizeMeters * Mathf.Clamp(maxSpikePatchExtentVoxelScale, 0.5f, 4f));
        return boundaryRatio >= minSpikePatchBoundaryEdgeRatio && maxExtent <= extentLimit;
    }

    private static void AddPatchVertex(List<Vector3> vertices, HashSet<int> vertexSet, int index, ref Vector3 min, ref Vector3 max)
    {
        if (index < 0 || index >= vertices.Count || !vertexSet.Add(index))
            return;
        Vector3 p = vertices[index];
        min = Vector3.Min(min, p);
        max = Vector3.Max(max, p);
    }

    private static void CountPatchEdge(Dictionary<ulong, List<int>> edgeToTriangles, HashSet<int> patchSet, int a, int b, ref int edgeCount, ref int boundaryEdges)
    {
        edgeCount++;
        ulong key = EdgeKey(a, b);
        if (!edgeToTriangles.TryGetValue(key, out List<int> linked))
        {
            boundaryEdges++;
            return;
        }

        int inside = 0;
        for (int i = 0; i < linked.Count; i++)
        {
            if (patchSet.Contains(linked[i]))
                inside++;
        }
        if (inside < linked.Count)
            boundaryEdges++;
    }

    private static void AddEdgeTriangle(Dictionary<ulong, List<int>> edgeToTriangles, int a, int b, int tri)
    {
        ulong key = EdgeKey(a, b);
        if (!edgeToTriangles.TryGetValue(key, out List<int> linked))
        {
            linked = new List<int>(2);
            edgeToTriangles.Add(key, linked);
        }
        linked.Add(tri);
    }

    private static void EnqueueTriNeighbors(Dictionary<ulong, List<int>> edgeToTriangles, ulong edgeKey, int tri, bool[] visited, Queue<int> queue)
    {
        if (!edgeToTriangles.TryGetValue(edgeKey, out List<int> linked))
            return;
        for (int i = 0; i < linked.Count; i++)
        {
            int neighbor = linked[i];
            if (neighbor == tri || neighbor < 0 || neighbor >= visited.Length || visited[neighbor])
                continue;
            visited[neighbor] = true;
            queue.Enqueue(neighbor);
        }
    }

    private static void EnqueueTriNeighbors(List<int> triangles, Dictionary<ulong, List<int>> edgeToTriangles, int tri, bool[] visited, Queue<int> queue)
    {
        int a = triangles[tri * 3];
        int b = triangles[tri * 3 + 1];
        int c = triangles[tri * 3 + 2];
        EnqueueTriNeighbors(edgeToTriangles, EdgeKey(a, b), tri, visited, queue);
        EnqueueTriNeighbors(edgeToTriangles, EdgeKey(b, c), tri, visited, queue);
        EnqueueTriNeighbors(edgeToTriangles, EdgeKey(c, a), tri, visited, queue);
    }

    private Vector3 ComputeTriangleNormal(List<Vector3> vertices, List<int> triangles, int tri)
    {
        int baseIndex = tri * 3;
        if (baseIndex + 2 >= triangles.Count)
            return Vector3.up;
        int ia = triangles[baseIndex];
        int ib = triangles[baseIndex + 1];
        int ic = triangles[baseIndex + 2];
        if (ia < 0 || ib < 0 || ic < 0 || ia >= vertices.Count || ib >= vertices.Count || ic >= vertices.Count)
            return Vector3.up;

        Vector3 normal = Vector3.Cross(vertices[ib] - vertices[ia], vertices[ic] - vertices[ia]);
        return normal.sqrMagnitude > 0.000001f ? normal.normalized : Vector3.up;
    }

    private void EnqueueSimilarNormalTriNeighbors(
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        Vector3[] triangleNormals,
        int tri,
        bool[] visited,
        Queue<int> queue)
    {
        EnqueueSimilarNormalTriNeighbors(
            triangles,
            edgeToTriangles,
            triangleNormals,
            tri,
            Mathf.Clamp(minSpikePatchNormalDot, -1f, 1f),
            visited,
            queue);
    }

    private static void EnqueueSimilarNormalTriNeighbors(
        List<int> triangles,
        Dictionary<ulong, List<int>> edgeToTriangles,
        Vector3[] triangleNormals,
        int tri,
        float minDot,
        bool[] visited,
        Queue<int> queue)
    {
        int a = triangles[tri * 3];
        int b = triangles[tri * 3 + 1];
        int c = triangles[tri * 3 + 2];
        Vector3 normal = tri >= 0 && tri < triangleNormals.Length ? triangleNormals[tri] : Vector3.up;
        EnqueueSimilarNormalTriNeighbors(edgeToTriangles, EdgeKey(a, b), tri, normal, minDot, triangleNormals, visited, queue);
        EnqueueSimilarNormalTriNeighbors(edgeToTriangles, EdgeKey(b, c), tri, normal, minDot, triangleNormals, visited, queue);
        EnqueueSimilarNormalTriNeighbors(edgeToTriangles, EdgeKey(c, a), tri, normal, minDot, triangleNormals, visited, queue);
    }

    private static void EnqueueSimilarNormalTriNeighbors(
        Dictionary<ulong, List<int>> edgeToTriangles,
        ulong edgeKey,
        int tri,
        Vector3 normal,
        float minDot,
        Vector3[] triangleNormals,
        bool[] visited,
        Queue<int> queue)
    {
        if (!edgeToTriangles.TryGetValue(edgeKey, out List<int> linked))
            return;
        for (int i = 0; i < linked.Count; i++)
        {
            int neighbor = linked[i];
            if (neighbor == tri || neighbor < 0 || neighbor >= visited.Length || visited[neighbor])
                continue;
            Vector3 neighborNormal = neighbor < triangleNormals.Length ? triangleNormals[neighbor] : Vector3.up;
            if (Vector3.Dot(normal, neighborNormal) < minDot)
                continue;
            visited[neighbor] = true;
            queue.Enqueue(neighbor);
        }
    }

    private static void AddEdgeUse(Dictionary<ulong, int> edgeUseCounts, int a, int b)
    {
        ulong key = EdgeKey(a, b);
        edgeUseCounts.TryGetValue(key, out int count);
        edgeUseCounts[key] = count + 1;
    }

    private static int GetEdgeUseCount(Dictionary<ulong, int> edgeUseCounts, int a, int b)
    {
        return edgeUseCounts.TryGetValue(EdgeKey(a, b), out int count) ? count : 0;
    }

    private static ulong EdgeKey(int a, int b)
    {
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        return ((ulong)min << 32) | max;
    }

    private static void DecodeEdgeKey(ulong key, out int a, out int b)
    {
        a = (int)(key >> 32);
        b = (int)(key & 0xffffffff);
    }

    private bool TryWorldToVoxel(Vector3 world, out int x, out int y, out int z)
    {
        Vector3 local = world - _volumeOriginWorld;
        float invVoxel = 1f / Mathf.Max(0.0001f, voxelSizeMeters);
        x = Mathf.RoundToInt(local.x * invVoxel);
        y = Mathf.RoundToInt(local.y * invVoxel);
        z = Mathf.RoundToInt(local.z * invVoxel);
        return x >= 0 && y >= 0 && z >= 0 && x < _dimX && y < _dimY && z < _dimZ;
    }

    private Vector3 VoxelCenter(int x, int y, int z)
    {
        return _volumeOriginWorld + new Vector3(x * voxelSizeMeters, y * voxelSizeMeters, z * voxelSizeMeters);
    }

    private Vector3 CellCenter(int x, int y, int z)
    {
        return _volumeOriginWorld + new Vector3(
            (x + 0.5f) * voxelSizeMeters,
            (y + 0.5f) * voxelSizeMeters,
            (z + 0.5f) * voxelSizeMeters);
    }

    private int Index(int x, int y, int z)
    {
        return x + _dimX * (y + _dimY * z);
    }

    private void IndexToVoxel(int index, out int x, out int y, out int z)
    {
        int plane = Mathf.Max(1, _dimX * _dimY);
        z = index / plane;
        int rem = index - z * plane;
        y = rem / Mathf.Max(1, _dimX);
        x = rem - y * Mathf.Max(1, _dimX);
    }

    private static int CellIndex(int x, int y, int z, int cellX, int cellY)
    {
        return x + cellX * (y + cellY * z);
    }

    private Vector3 GetCameraPosition()
    {
        if (volumeAnchor != null)
            return volumeAnchor.position;
        Camera camera = Camera.main;
        return camera != null ? camera.transform.position : transform.position;
    }

    private Vector3 GetCameraForward()
    {
        if (volumeAnchor != null)
            return volumeAnchor.forward;
        Camera camera = Camera.main;
        return camera != null ? camera.transform.forward : transform.forward;
    }

    private void ResolveRefs()
    {
        if (rawDepthSource == null)
            rawDepthSource = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (rawDepthSource == null)
            rawDepthSource = FindAnyObjectByType<ScanCoverDepthGridPointCloud>(FindObjectsInactive.Include);
        if (volumeAnchor == null && Camera.main != null)
            volumeAnchor = Camera.main.transform;
    }

    private void EnsureObjects()
    {
        if (displayRoot == null)
            displayRoot = transform;

        if (_meshObject == null)
        {
            _meshObject = new GameObject("[ScanCover] TSDF Single Shell");
        }
        if (_wireObject == null)
        {
            _wireObject = new GameObject("[ScanCover] TSDF Cover Wire");
        }

        if (forceWorldSpaceDisplay)
        {
            _meshObject.transform.SetParent(null, true);
            _meshObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _meshObject.transform.localScale = Vector3.one;
            _wireObject.transform.SetParent(null, true);
            _wireObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _wireObject.transform.localScale = Vector3.one;
        }
        else if (_meshObject.transform.parent != displayRoot)
        {
            _meshObject.transform.SetParent(displayRoot, false);
            _wireObject.transform.SetParent(displayRoot, false);
        }
        else if (_wireObject.transform.parent != displayRoot)
        {
            _wireObject.transform.SetParent(displayRoot, false);
        }

        if (_meshFilter == null)
            _meshFilter = _meshObject.GetComponent<MeshFilter>();
        if (_meshFilter == null)
            _meshFilter = _meshObject.AddComponent<MeshFilter>();

        if (_meshRenderer == null)
            _meshRenderer = _meshObject.GetComponent<MeshRenderer>();
        if (_meshRenderer == null)
            _meshRenderer = _meshObject.AddComponent<MeshRenderer>();

        if (_mesh == null)
        {
            _mesh = new Mesh { name = "ScanCover_TSDF_SingleShell" };
            _mesh.indexFormat = IndexFormat.UInt32;
        }

        _meshFilter.sharedMesh = _mesh;
        _meshRenderer.sharedMaterial = ResolveMaterial();
        _meshRenderer.enabled = showPlanarCoverSurfaceFill || !renderStableCellsAsPlanarMeshCover || !showPlanarCoverWireOverlay;

        if (_wireMeshFilter == null)
            _wireMeshFilter = _wireObject.GetComponent<MeshFilter>();
        if (_wireMeshFilter == null)
            _wireMeshFilter = _wireObject.AddComponent<MeshFilter>();

        if (_wireMeshRenderer == null)
            _wireMeshRenderer = _wireObject.GetComponent<MeshRenderer>();
        if (_wireMeshRenderer == null)
            _wireMeshRenderer = _wireObject.AddComponent<MeshRenderer>();

        if (_wireMesh == null)
        {
            _wireMesh = new Mesh { name = "ScanCover_TSDF_CoverWire" };
            _wireMesh.indexFormat = IndexFormat.UInt32;
        }

        _wireMeshFilter.sharedMesh = _wireMesh;
        _wireMeshRenderer.sharedMaterial = ResolveWireMaterial();
        _wireObject.SetActive(showMeshEdgeWireOverlay || showPlanarCoverWireOverlay);
        EnsureDirtyTsdfEvidenceObjects();
        EnsureRawDepthDebugObjects();
        EnsureHoleBoundaryDiagnosticObjects();
        ApplyRawDepthDebugDisplayMode();
        EnsureDiagnosticHud();
    }

    private void ApplyRawDepthDebugDisplayMode()
    {
        bool coverageOverlay = enableRawCoverageGridDiagnostics && showRawCoverageGridOverlay;
        bool hideMeshForCoverage = coverageOverlay && hideMeshWhenRawCoverageGridOverlay;
        bool hideMeshForHoleDiagnosis = showHoleBoundaryDiagnosis && hideMeshWhileShowingHoleBoundaryDiagnosis && !showMeshBehindLayerDiagnostics;
        bool layerWireOnly = showHoleBoundaryDiagnosis && showMeshBehindLayerDiagnostics && showMeshAsWireOnlyBehindLayerDiagnostics;
        if (_meshRenderer != null)
            _meshRenderer.enabled = !showRawDepthDebugView && !hideMeshForCoverage && !hideMeshForHoleDiagnosis && !layerWireOnly && (showPlanarCoverSurfaceFill || !renderStableCellsAsPlanarMeshCover || !showPlanarCoverWireOverlay);
        if (_wireObject != null)
            _wireObject.SetActive(!showRawDepthDebugView && !hideMeshForCoverage && !hideMeshForHoleDiagnosis && (layerWireOnly || showMeshEdgeWireOverlay || showPlanarCoverWireOverlay));
        if (_rawDepthDebugObject != null)
            _rawDepthDebugObject.SetActive((showRawDepthDebugView || coverageOverlay) && LastRawDepthDebugRenderedCount > 0);
    }

    private void EnsureDirtyTsdfEvidenceObjects()
    {
        if (_dirtyEvidenceObject == null)
            _dirtyEvidenceObject = new GameObject("[ScanCover] Dirty TSDF Evidence");

        if (forceWorldSpaceDisplay)
        {
            _dirtyEvidenceObject.transform.SetParent(null, true);
            _dirtyEvidenceObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _dirtyEvidenceObject.transform.localScale = Vector3.one;
        }
        else if (_dirtyEvidenceObject.transform.parent != displayRoot)
        {
            _dirtyEvidenceObject.transform.SetParent(displayRoot, false);
        }

        if (_dirtyEvidenceMeshFilter == null)
            _dirtyEvidenceMeshFilter = _dirtyEvidenceObject.GetComponent<MeshFilter>();
        if (_dirtyEvidenceMeshFilter == null)
            _dirtyEvidenceMeshFilter = _dirtyEvidenceObject.AddComponent<MeshFilter>();

        if (_dirtyEvidenceRenderer == null)
            _dirtyEvidenceRenderer = _dirtyEvidenceObject.GetComponent<MeshRenderer>();
        if (_dirtyEvidenceRenderer == null)
            _dirtyEvidenceRenderer = _dirtyEvidenceObject.AddComponent<MeshRenderer>();

        if (_dirtyEvidenceMesh == null)
        {
            _dirtyEvidenceMesh = new Mesh { name = "ScanCover_DirtyTsdfEvidence" };
            _dirtyEvidenceMesh.indexFormat = IndexFormat.UInt32;
        }

        _dirtyEvidenceMeshFilter.sharedMesh = _dirtyEvidenceMesh;
        _dirtyEvidenceRenderer.sharedMaterial = ResolveDirtyEvidenceMaterial();
        _dirtyEvidenceObject.SetActive(false);
    }

    private void EnsureRawDepthDebugObjects()
    {
        if (_rawDepthDebugObject == null)
            _rawDepthDebugObject = new GameObject("[ScanCover] Raw Depth Debug");

        if (forceWorldSpaceDisplay)
        {
            _rawDepthDebugObject.transform.SetParent(null, true);
            _rawDepthDebugObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _rawDepthDebugObject.transform.localScale = Vector3.one;
        }
        else if (_rawDepthDebugObject.transform.parent != displayRoot)
        {
            _rawDepthDebugObject.transform.SetParent(displayRoot, false);
        }

        if (_rawDepthDebugMeshFilter == null)
            _rawDepthDebugMeshFilter = _rawDepthDebugObject.GetComponent<MeshFilter>();
        if (_rawDepthDebugMeshFilter == null)
            _rawDepthDebugMeshFilter = _rawDepthDebugObject.AddComponent<MeshFilter>();

        if (_rawDepthDebugRenderer == null)
            _rawDepthDebugRenderer = _rawDepthDebugObject.GetComponent<MeshRenderer>();
        if (_rawDepthDebugRenderer == null)
            _rawDepthDebugRenderer = _rawDepthDebugObject.AddComponent<MeshRenderer>();

        if (_rawDepthDebugMesh == null)
        {
            _rawDepthDebugMesh = new Mesh { name = "ScanCover_RawDepthDebug" };
            _rawDepthDebugMesh.indexFormat = IndexFormat.UInt32;
        }

        _rawDepthDebugMeshFilter.sharedMesh = _rawDepthDebugMesh;
        _rawDepthDebugRenderer.sharedMaterial = ResolveRawDepthDebugMaterial();
        _rawDepthDebugObject.SetActive((showRawDepthDebugView || (enableRawCoverageGridDiagnostics && showRawCoverageGridOverlay)) && LastRawDepthDebugRenderedCount > 0);
    }

    private void EnsureHoleBoundaryDiagnosticObjects()
    {
        if (_holeBoundaryDiagnosticObject == null)
            _holeBoundaryDiagnosticObject = new GameObject("[ScanCover] Hole Boundary Diagnosis");

        if (forceWorldSpaceDisplay)
        {
            _holeBoundaryDiagnosticObject.transform.SetParent(null, true);
            _holeBoundaryDiagnosticObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _holeBoundaryDiagnosticObject.transform.localScale = Vector3.one;
        }
        else if (_holeBoundaryDiagnosticObject.transform.parent != displayRoot)
        {
            _holeBoundaryDiagnosticObject.transform.SetParent(displayRoot, false);
        }

        if (_holeBoundaryDiagnosticMeshFilter == null)
            _holeBoundaryDiagnosticMeshFilter = _holeBoundaryDiagnosticObject.GetComponent<MeshFilter>();
        if (_holeBoundaryDiagnosticMeshFilter == null)
            _holeBoundaryDiagnosticMeshFilter = _holeBoundaryDiagnosticObject.AddComponent<MeshFilter>();
        if (_holeBoundaryDiagnosticRenderer == null)
            _holeBoundaryDiagnosticRenderer = _holeBoundaryDiagnosticObject.GetComponent<MeshRenderer>();
        if (_holeBoundaryDiagnosticRenderer == null)
            _holeBoundaryDiagnosticRenderer = _holeBoundaryDiagnosticObject.AddComponent<MeshRenderer>();
        if (_holeBoundaryDiagnosticMesh == null)
        {
            _holeBoundaryDiagnosticMesh = new Mesh { name = "ScanCover_HoleBoundaryDiagnosis" };
            _holeBoundaryDiagnosticMesh.indexFormat = IndexFormat.UInt32;
        }

        _holeBoundaryDiagnosticMeshFilter.sharedMesh = _holeBoundaryDiagnosticMesh;
        _holeBoundaryDiagnosticRenderer.sharedMaterial = ResolveRawDepthDebugMaterial();
        _holeBoundaryDiagnosticObject.SetActive(showHoleBoundaryDiagnosis && _holeBoundaryDiagnosticMesh.vertexCount > 0);
    }

    private void EnsureDiagnosticHud()
    {
        if (!showDiagnosticHud)
        {
            if (_hudRoot != null && _hudRoot.activeSelf)
                _hudRoot.SetActive(false);
            return;
        }

        if (_hudCanvas != null && _hudText != null)
            return;

        GameObject root = new GameObject("[ScanCover] TSDF Diagnostic HUD", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler));
        root.hideFlags = HideFlags.DontSave;
        _hudRoot = root;
        _hudCanvas = root.GetComponent<Canvas>();
        _hudCanvas.renderMode = RenderMode.WorldSpace;

        CanvasScaler scaler = root.GetComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;
        scaler.referencePixelsPerUnit = 100f;

        GameObject panelGO = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelGO.transform.SetParent(root.transform, false);
        RectTransform panelRect = panelGO.GetComponent<RectTransform>();
        panelRect.sizeDelta = new Vector2(760f, 250f);
        panelRect.anchorMin = new Vector2(0.5f, 0.5f);
        panelRect.anchorMax = new Vector2(0.5f, 0.5f);
        panelRect.pivot = new Vector2(0.5f, 0.5f);
        _hudPanel = panelGO.GetComponent<Image>();
        _hudPanel.color = new Color(0f, 0.02f, 0.03f, 0.72f);
        _hudPanel.raycastTarget = false;

        GameObject textGO = new GameObject("Text", typeof(RectTransform), typeof(Text));
        textGO.transform.SetParent(panelGO.transform, false);
        RectTransform textRect = textGO.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(18f, 14f);
        textRect.offsetMax = new Vector2(-18f, -14f);

        _hudText = textGO.GetComponent<Text>();
        _hudText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (_hudText.font == null)
            _hudText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        _hudText.fontSize = 20;
        _hudText.alignment = TextAnchor.UpperLeft;
        _hudText.horizontalOverflow = HorizontalWrapMode.Wrap;
        _hudText.verticalOverflow = VerticalWrapMode.Overflow;
        _hudText.color = hudHealthyColor;
        _hudText.raycastTarget = false;
        _hudText.text = "TSDF HUD";
    }

    private string DominantHoleCauseText()
    {
        int[] counts =
        {
            LastHoleCauseNoRawCount, LastHoleCausePendingCount, LastHoleCauseNoBandCount, LastHoleCauseNoZeroCrossCount,
            LastHoleCauseVertexRejectCount, LastHoleCauseQuadRejectCount, LastHoleCauseClearedCount,
            LastHoleCauseLocalOffsetCount, LastHoleCauseCrossFrameCount
        };
        string[] names = { "NO_RAW", "PENDING", "NO_BAND", "NO_ZERO", "VERTEX", "QUAD", "CLEARED", "LOCAL", "XFRAME" };
        int best = 0;
        for (int i = 1; i < counts.Length; i++)
            if (counts[i] > counts[best])
                best = i;
        return names[best] + "=" + counts[best];
    }

    private void UpdateDiagnosticHud()
    {
        if (!showDiagnosticHud)
        {
            if (_hudRoot != null && _hudRoot.activeSelf)
                _hudRoot.SetActive(false);
            return;
        }

        EnsureDiagnosticHud();
        if (_hudRoot == null || _hudText == null)
            return;

        Camera camera = Camera.main;
        if (camera != null)
        {
            Transform hudTransform = _hudRoot.transform;
            hudTransform.SetParent(camera.transform, false);
            hudTransform.localPosition = hudLocalPosition;
            hudTransform.localEulerAngles = hudLocalEuler;
            hudTransform.localScale = hudLocalScale;
            _hudCanvas.worldCamera = camera;
        }
        else if (_hudRoot.transform.parent != transform)
        {
            _hudRoot.transform.SetParent(transform, false);
        }

        if (!_hudRoot.activeSelf)
            _hudRoot.SetActive(true);

        if (Time.unscaledTime < _nextHudRefresh)
            return;
        _nextHudRefresh = Time.unscaledTime + Mathf.Max(0.05f, hudRefreshIntervalSeconds);

        float acceptRatio = LastInputSampleCount > 0 ? (float)LastIntegratedSampleCount / LastInputSampleCount : 0f;
        float invalidRatio = LastInputSampleCount > 0 ? (float)LastRejectedInvalidPositionCount / LastInputSampleCount : 0f;
        float outsideRatio = LastInputSampleCount > 0 ? (float)LastRejectedOutsideVolumeCount / LastInputSampleCount : 0f;
        int fallbackSampleCount = GetFallbackSurfaceSampleCount();
        float fallbackFill = maxFallbackSurfaceSamples > 0 ? (float)fallbackSampleCount / maxFallbackSurfaceSamples : 0f;
        int lightCoverBudget = GetLightCoverRenderBudget();
        float renderedFill = lightCoverBudget > 0 ? (float)LastRenderedFallbackSampleCount / lightCoverBudget : 0f;
        int totalRejects =
            LastRejectedInvalidPositionCount +
            LastRejectedDepthRangeCount +
            LastRejectedNormalCount +
            LastRejectedFacingCount +
            LastRejectedConfidenceCount +
            LastRejectedDepthDiscontinuityCount +
            LastRejectedTsdfDepthSupportCount +
            LastRejectedRobustDepthCount +
            LastRejectedTsdfConflictCount +
            LastPendingStableTsdfCount +
            LastPendingTsdfCorrectionCount +
            LastRejectedFallbackWeakCount +
            LastRejectedOutsideVolumeCount;

        bool denseOutsideIsBlocking = !useChunkedSurfaceSamples && outsideRatio > 0.08f;
        bool renderBudgetExceeded = useChunkedSurfaceSamples && fallbackSampleCount > maxFallbackSurfaceSamples;
        bool mainShellIsHolding = LastCandidateShellHeldCellCount + LastDetailCandidateHeldCellCount > Mathf.Max(128, LastMainShellPromotedCellCount);
        bool meshDiagnosticsWarning = LastDiagSuspectPatchCount > 0 || LastDiagNonManifoldEdgeCount > 0;
        bool tumorSuspectWarning = LastTumorSuspectTriangleCount > 0;
        bool warning = invalidRatio > 0.35f || denseOutsideIsBlocking || acceptRatio < 0.25f || LastSkippedFallbackCapacityCount > 0 || LastSurfaceCellConflictCount > 0 || renderBudgetExceeded || mainShellIsHolding || meshDiagnosticsWarning || tumorSuspectWarning;
        _hudText.color = warning ? hudWarningColor : hudHealthyColor;

        string likelyCause = "OK";
        if (LastInputSampleCount <= 0)
            likelyCause = "NO RAW FRAME";
        else if (LastRejectedInvalidPositionCount > Mathf.Max(64, LastIntegratedSampleCount))
            likelyCause = "RAW DEPTH HOLE";
        else if (denseOutsideIsBlocking && LastRejectedOutsideVolumeCount > Mathf.Max(32, LastIntegratedSampleCount / 8))
            likelyCause = "OUTSIDE TSDF VOLUME / START BIAS";
        else if (LastSkippedFallbackCapacityCount > 0)
            likelyCause = "SURFACE CELL CAP HIT";
        else if (LastSurfaceCellConflictCount > 0)
            likelyCause = "SURFACE SIDE CONFLICT";
        else if (tumorSuspectWarning)
            likelyCause = "MESH SUSPECT OVERLAY";
        else if (meshDiagnosticsWarning)
            likelyCause = LastDiagLikelyCause;
        else if (LastRejectedDepthDiscontinuityCount > Mathf.Max(32, LastIntegratedSampleCount / 8))
            likelyCause = "DEPTH EDGE FILTER";
        else if (LastRejectedTsdfDepthSupportCount > Mathf.Max(32, LastIntegratedSampleCount / 8))
            likelyCause = "TSDF DEPTH SUPPORT GATE";
        else if (LastRejectedTsdfConflictCount > Mathf.Max(64, LastUpdatedVoxelCount / 12))
            likelyCause = "TSDF CONFLICT GUARD";
        else if (LastCorrectedTsdfCount > 0)
            likelyCause = "TSDF OLD DEPTH CORRECTED";
        else if (LastPendingTsdfCorrectionCount > Mathf.Max(64, LastRejectedTsdfConflictCount))
            likelyCause = "WAITING OLD DEPTH CORRECTION";
        else if (LastPendingStableTsdfCount > Mathf.Max(128, LastUpdatedVoxelCount / 4))
            likelyCause = "WAITING MULTI-FRAME STABILITY";
        else if (mainShellIsHolding)
            likelyCause = "WAITING MAIN SHELL PROMOTION";
        else if (LastPrunedComponentTriangleCount > Mathf.Max(128, LastPrePruneMeshTriangleCount / 3))
            likelyCause = "MESH ISLAND COMPONENT PRUNE";
        else if (LastSurfaceQuadCandidateCount > 0 && LastAddedSurfaceQuadCount < Mathf.Max(16, LastSurfaceQuadCandidateCount / 8))
            likelyCause = "MESH QUAD ASSEMBLY WEAK";
        else if (LastBoundaryBridgeCandidateCount > 0 && LastAddedBoundaryBridgeCount <= 0)
            likelyCause = "BOUNDARY BRIDGE REJECTED";
        else if (renderBudgetExceeded)
            likelyCause = "RENDER BUDGET THINNING";
        else if (LastRejectedDepthRangeCount > Mathf.Max(32, LastIntegratedSampleCount / 6))
            likelyCause = "DEPTH RANGE FILTER";

        string mainGateText = requireMainShellPromotionGate ? minMainShellFusedFrames + "/" + minMainShellNeighborSurfaceCells : "off";
        string detailText = separateComplexDetailCandidates ? (promoteComplexDetailCandidatesToMesh ? "promote" : "hold") + ":" + minDetailCandidateFrames + "/" + minDetailCandidateNeighborSurfaceCells + "/" + minDetailPromotionNeighborSurfaceCells : "off";
        string supportGateText = gateTsdfWritesByDepthSupport ? minTsdfDepthCheckedNeighbors + "/" + minTsdfDepthConsistencyRatio.ToString("0.00") : "off";
        bool guardedMeshPreview = showExtractedSurfaceMesh && !useFallbackMeshWhenTsdfExtractionEmpty;
        bool strictMeshPreview = guardedMeshPreview && !AllowRelaxedMeshExtraction;
        string displayModeText = LastMeshUsedExtractedTsdf
            ? (useAtomicObservationTsdfBands ? "ATOMIC A/P/R VOXEL-PROJ" : (useStage03ACleanIsoSurface ? (useV08DirectBandWriteForDiagnosis ? "03A V0.8 WRITE / V0.9 MESH" : (UseLegacyMeshExtraction ? (useV09ExactCornerEligibilityForDiagnosis ? "03A V0.9 EXACT DIAG" : "03A CLEAN / V0.9 EXTRACT") : "STAGE 03A CLEAN TSDF")) : (UseLegacyMeshExtraction ? "TSDF EXTRACTION" : (strictMeshPreview ? "STRICT TSDF PREVIEW" : (guardedMeshPreview ? "GUARDED TSDF PREVIEW" : "TSDF EXTRACTION")))))
            : (renderStableCellsAsPlanarMeshCover ? "LIGHT TSDF COVER" : "FALLBACK RAW CELLS");
        if (useStage03ACleanIsoSurface)
        {
            bool meshVisible = _meshRenderer != null && _meshRenderer.enabled;
            bool wireVisible = _wireObject != null && _wireObject.activeSelf;
            bool meshReady = LastMeshTriangleCount > 0 && (meshVisible || wireVisible);
            _hudText.color = meshReady ? hudHealthyColor : hudWarningColor;
            _hudText.text =
                "ScanCover STAGE 03A\n" +
                $"State : {(_captureRoutine != null ? "FUSING" : "READY")} / {(meshReady ? "MESH READY" : likelyCause)}  f={LastRawFrameIndex}/{IntegratedFrameCount}\n" +
                $"Input : {LastIntegratedSampleCount}/{LastInputSampleCount} {acceptRatio:P0}  updated={LastUpdatedVoxelCount}\n" +
                $"Bands : A/P={LastAtomicAcceptedBandVoxelWriteCount}/{LastAtomicProvisionalBandVoxelWriteCount}  up/down={LastAtomicPromotedProvisionalVoxelCount}/{LastAtomicRetiredProvisionalVoxelCount}\n" +
                $"Holes : band/ok/drop={LastHoleSideRepairPlaneBandVoxelCount}/{LastHoleSideRepairPlaneConfirmedCount}/{LastHoleSideRepairPlaneRetiredCount}  N/S/P/T={LastHoleSideRepairRetiredNoNearAcceptCount}/{LastHoleSideRepairRetiredSignMismatchCount}/{LastHoleSideRepairRetiredPlaneMismatchCount}/{LastHoleSideRepairRetiredTsdfDeltaCount}\n" +
                $"Pdist : .5/.75/1/1.5/2+={LastHoleSideRepairPlaneDistance0ToHalfCount}/{LastHoleSideRepairPlaneDistanceHalfTo075Count}/{LastHoleSideRepairPlaneDistance075To1Count}/{LastHoleSideRepairPlaneDistance1To15Count}/{LastHoleSideRepairPlaneDistance15To2Count + LastHoleSideRepairPlaneDistanceOver2Count}\n" +
                $"Commit: old/new/keep={LastCommittedMeshBlockCount}/{LastCandidateMeshBlockCount}/{LastRetainedCommittedMeshBlockCount} hold={LastHeldDisplayTriangleCount} grow={LastCommittedMeshGrowthTriangleCount}\n" +
                $"Mesh  : tris={LastMeshTriangleCount} wire={LastWireSegmentCount} vis={(meshVisible ? 1 : 0)}/{(wireVisible ? 1 : 0)}  log={(string.IsNullOrEmpty(_lastHoleDiagnosticPath) ? "off" : Path.GetFileName(_lastHoleDiagnosticPath))}";
            return;
        }
        _hudText.text =
            "ScanCover SAFEHUD\n" +
            $"State : {(_captureRoutine != null ? "FUSING" : "READY")} / {likelyCause}\n" +
            $"Mode  : {displayModeText} f={LastRawFrameIndex} fused={IntegratedFrameCount} in={LastIntegratedSampleCount}/{LastInputSampleCount} {acceptRatio:P0}\n" +
            $"03A   : scan/cand/vert={LastSurfaceCellScanCount}/{LastStrictSurfaceCellCandidateCount}/{LastBuiltSurfaceCellVertexCount} quad/add={LastSurfaceQuadCandidateCount}/{LastAddedSurfaceQuadCount} tris={LastMeshTriangleCount} vis={(_meshRenderer != null && _meshRenderer.enabled ? 1 : 0)}/{(_wireObject != null && _wireObject.activeSelf ? 1 : 0)}\n" +
            $"Vote  : {observationVoteMode} A/P/R={_auditVoteAcceptCount}/{_auditVotePendingCount}/{_auditVoteRejectCount} blocked={_auditVoteEnforcedRejectCount}\n" +
            $"Entry : strongA={LastVoteStrongCurrentAcceptCount} same/cross={LastVoteSameFrameConflictRejectCount}/{LastVoteCrossFrameCleanConflictRejectCount} pending={LastVotePendingHoldCount}/{LastVotePendingConfirmedWriteCount}\n" +
            $"Formal: ok/block/prov={LastFormalIntegrateWriteCount}/{LastFormalIntegrateBlockedCount}/{LastFormalIntegrateProvisionalCount} strongBlock={LastFormalIntegrateBlockStrongCurrentHistoryCount}\n" +
            $"FLocal: pass/block s/st/a/r={LastFormalStrongCurrentLocalHistoryBypassCount}/{LastFormalStrongCurrentLocalHistoryBlockLocalCount} {LastFormalStrongCurrentLocalHistoryBlockSupportCount}/{LastFormalStrongCurrentLocalHistoryBlockStableCount}/{LastFormalStrongCurrentLocalHistoryBlockAxialCount}/{LastFormalStrongCurrentLocalHistoryBlockResidualCount}\n" +
            $"Promote: strong={LastStrongCurrentProvisionalPromotedCount}/{LastStrongCurrentProvisionalPromotionBlockedCount} hits>={minStrongCurrentProvisionalHits} w>={minStrongCurrentProvisionalWeight}\n" +
            $"PBlock: no(n/f)={LastStrongCurrentProvisionalBlockNoProvisionalCount}({LastStrongCurrentProvisionalBlockNoProvisionalNearSurfaceCount}/{LastStrongCurrentProvisionalBlockNoProvisionalFarBandCount}) hit/w/hist/frm/local={LastStrongCurrentProvisionalBlockHitsCount}/{LastStrongCurrentProvisionalBlockWeightCount}/{LastStrongCurrentProvisionalBlockCleanHistoryCount + LastStrongCurrentProvisionalBlockAgreementCount + LastStrongCurrentProvisionalBlockBandHistoryCount}/{LastStrongCurrentProvisionalBlockSameFrameCount + LastStrongCurrentProvisionalBlockCrossFrameCount}/{LastStrongCurrentProvisionalBlockLocalCount}\n" +
            $"Seed  : strong={LastStrongSampleSeedWriteCount}/{LastStrongSampleSeedBlockedCount} tempBlock={LastStrongSampleSeedTemporaryBlockedCount} lock={(lockStrongSampleSeedToTemporaryTsdf ? 1 : 0)}\n" +
            $"RawCov: cells={LastRawCoverageGridCellCount} A/P/M={LastRawCoverageAcceptedCellCount}/{LastRawCoverageProblemCellCount}/{LastRawCoverageMixedCellCount} compA={LastRawCoverageAcceptedComponentCount}/{LastRawCoverageLargestAcceptedComponentCells} compP={LastRawCoverageProblemComponentCount}/{LastRawCoverageLargestProblemComponentCells}\n" +
            $"Dirty : wait={LastPendingTsdfCorrectionCount} rep={LastReplacedDirtyTsdfCount} guard={LastGuardedDirtyTsdfReplaceCount} hBlock={LastDirtyTsdfReplaceBlockedHitsCount} cleanH={LastDirtyTsdfReplaceBlockedCleanHistoryCount} active={LastDirtyTsdfActiveCount}\n" +
            $"Clear : cand={LastFreeSpaceEvidenceCandidateCount} new/rep/wait={LastFreeSpaceEvidenceNewCount}/{LastFreeSpaceEvidenceRepeatCount}/{LastFreeSpaceEvidenceWaitingCount} apply/clear={LastFreeSpaceEvidenceAppliedCount}/{LastFreeSpaceEvidenceClearedCount} block hi/same/dup={LastFreeSpaceEvidenceBlockedHighWeightCount}/{LastFreeSpaceEvidenceBlockedSameFrameCount}/{LastFreeSpaceEvidenceDuplicateFrameCount} cancel={LastFreeSpaceEvidenceCancelledBySurfaceCount}\n" +
            $"Metab : cand={LastOldCleanMetabolismCandidateCount} skip={LastOldCleanMetabolismSkippedWeakBandCount}/{LastOldCleanMetabolismSkippedWeakCrossFrameCount} watch={LastOldCleanMetabolismWatchCount} wait={LastOldCleanMetabolismWaitingHitsCount} d/c={LastOldCleanMetabolismDecayCount}/{LastOldCleanMetabolismClearCount} block={LastOldCleanMetabolismBlockedCount}\n" +
            $"Mesh  : tris={LastMeshTriangleCount} suspect={LastTumorSuspectTriangleCount} dirty={LastTumorDirtyTriangleCount}\n" +
            $"Shape : island/open/long={LastTumorIslandTriangleCount}/{LastTumorOpenEdgeTriangleCount}/{LastTumorLongEdgeTriangleCount}\n" +
            $"Island: comp={LastIslandCauseComponentCount} no/p/d/low/plane/prn={LastIslandCauseNoTsdfCount}/{LastIslandCausePendingCount}/{LastIslandCauseDirtyCount}/{LastIslandCauseLowWeightCount}/{LastIslandCausePlaneMismatchCount}/{LastIslandCausePrunedCount}\n" +
            $"Cont  : prov={LastProvisionalTsdfSupportWriteCount}/{LastProvisionalTsdfSupportBlockedCount} conf/ret={LastProvisionalTsdfConfirmedCount}/{LastProvisionalTsdfRetiredCount} fill={LastTsdfContinuityFilledCount} same/mix/pN={LastTsdfContinuitySameSignFilledCount}/{LastTsdfContinuityMixedSignFilledCount}/{LastTsdfContinuityProvisionalNeighborCount} bNo={LastTsdfBoundaryNoTsdfFilledCount}/{LastTsdfBoundaryNoTsdfCandidateCount}\n" +
            $"BNoBlk: sup/face/axis/anchor/prov/near={LastTsdfBoundaryNoTsdfBlockedSupportCount}/{LastTsdfBoundaryNoTsdfBlockedFaceCount}/{LastTsdfBoundaryNoTsdfBlockedAxisCount}/{LastTsdfBoundaryNoTsdfBlockedCleanAnchorCount}/{LastTsdfBoundaryNoTsdfBlockedProvisionalAnchorCount}/{LastTsdfBoundaryNoTsdfBlockedNearSurfaceCount} joint/zero={LastTsdfBoundaryNoTsdfVerifiedJointAnchorFilledCount}/{LastTsdfBoundaryNoTsdfZeroCleanJointAnchorFilledCount}\n" +
            $"BNoPrv: present/face/ok/okF={LastTsdfBoundaryNoTsdfProvisionalPresentNeighborCount}/{LastTsdfBoundaryNoTsdfProvisionalFacePresentCount}/{LastTsdfBoundaryNoTsdfProvisionalAcceptedNeighborCount}/{LastTsdfBoundaryNoTsdfProvisionalAcceptedFaceCount} block w/d/t/r/p/nf={LastTsdfBoundaryNoTsdfProvisionalBlockedWeightCount}/{LastTsdfBoundaryNoTsdfProvisionalBlockedDirtyPendingCount}/{LastTsdfBoundaryNoTsdfProvisionalBlockedTsdfCount}/{LastTsdfBoundaryNoTsdfProvisionalBlockedResidualCount}/{LastTsdfBoundaryNoTsdfProvisionalBlockedProvenanceCount}/{LastTsdfBoundaryNoTsdfProvisionalBlockedNotFaceCount}\n" +
            $"BNoFace: hist0..6={LastTsdfBoundaryNoTsdfFaceSupport0Count}/{LastTsdfBoundaryNoTsdfFaceSupport1Count}/{LastTsdfBoundaryNoTsdfFaceSupport2Count}/{LastTsdfBoundaryNoTsdfFaceSupport3Count}/{LastTsdfBoundaryNoTsdfFaceSupport4Count}/{LastTsdfBoundaryNoTsdfFaceSupport5Count}/{LastTsdfBoundaryNoTsdfFaceSupport6Count} deficit1/2+={LastTsdfBoundaryNoTsdfFaceDeficitOneCount}/{LastTsdfBoundaryNoTsdfFaceDeficitTwoPlusCount} two={LastTsdfBoundaryNoTsdfVerifiedTwoFaceFilledCount}/{LastTsdfBoundaryNoTsdfVerifiedTwoFaceCandidateCount}\n" +
            $"FaceWhy: okC/P={LastTsdfBoundaryNoTsdfFaceSlotAcceptedCleanCount}/{LastTsdfBoundaryNoTsdfFaceSlotAcceptedProvisionalCount} block w/d/p/t/r/oob={LastTsdfBoundaryNoTsdfFaceSlotBlockedWeightCount}/{LastTsdfBoundaryNoTsdfFaceSlotBlockedDirtyPendingCount}/{LastTsdfBoundaryNoTsdfFaceSlotBlockedProvenanceCount}/{LastTsdfBoundaryNoTsdfFaceSlotBlockedTsdfCount}/{LastTsdfBoundaryNoTsdfFaceSlotBlockedResidualCount}/{LastTsdfBoundaryNoTsdfFaceSlotOutOfBoundsCount}\n" +
            $"BNoSup: pass/d1/d2/d3/d4+={LastTsdfBoundaryNoTsdfSupportPassedCount}/{LastTsdfBoundaryNoTsdfSupportDeficitOneCount}/{LastTsdfBoundaryNoTsdfSupportDeficitTwoCount}/{LastTsdfBoundaryNoTsdfSupportDeficitThreeCount}/{LastTsdfBoundaryNoTsdfSupportDeficitFourPlusCount} okC/P={LastTsdfBoundaryNoTsdfSupportSlotAcceptedCleanCount}/{LastTsdfBoundaryNoTsdfSupportSlotAcceptedProvisionalCount}\n" +
            $"SupWhy: block w/d/p/t/r/oob={LastTsdfBoundaryNoTsdfSupportSlotBlockedWeightCount}/{LastTsdfBoundaryNoTsdfSupportSlotBlockedDirtyPendingCount}/{LastTsdfBoundaryNoTsdfSupportSlotBlockedProvenanceCount}/{LastTsdfBoundaryNoTsdfSupportSlotBlockedTsdfCount}/{LastTsdfBoundaryNoTsdfSupportSlotBlockedResidualCount}/{LastTsdfBoundaryNoTsdfSupportSlotOutOfBoundsCount}\n" +
            $"VBlock: near(+/-)={LastProvisionalTsdfNearSurfaceBlockedCount}({LastProvisionalTsdfNearSurfacePositiveBlockedCount}/{LastProvisionalTsdfNearSurfaceNegativeBlockedCount}) far={LastProvisionalTsdfFarBandSkippedCount} pending/formal={LastProvisionalTsdfPendingStabilityBlockedCount}/{LastProvisionalTsdfFormalDowngradeBlockedCount}\n" +
            $"VPass : oldW/bootstrap={LastProvisionalTsdfExistingWeightBypassCount}/{LastProvisionalTsdfBootstrapLocalBypassCount} vote/sup/band/hist/frm/old/plane/local={LastProvisionalTsdfVoteBlockedCount + LastProvisionalTsdfScoreBlockedCount}/{LastProvisionalTsdfSupportRatioBlockedCount}/{LastProvisionalTsdfBandBlockedCount}/{LastProvisionalTsdfCleanHistoryBlockedCount}/{LastProvisionalTsdfSameFrameBlockedCount + LastProvisionalTsdfCrossFrameBlockedCount}/{LastProvisionalTsdfOldWeightBlockedCount + LastProvisionalTsdfConflictBlockedCount}/{LastProvisionalTsdfPlaneBlockedCount}/{LastProvisionalLocalSupportBlockedCount}\n" +
            $"Plane : pass/block/noRef cand n/d={LastProvisionalPlaneCompatibilityPassCount}/{LastProvisionalPlaneCompatibilityBlockedCount}/{LastProvisionalPlaneCompatibilityNoReferenceCount} {LastProvisionalPlaneCompatibilityCandidateCount} {LastProvisionalPlaneCompatibilityNormalRejectedCount}/{LastProvisionalPlaneCompatibilityDistanceRejectedCount}\n" +
            $"Geom  : pure={LastPureGeometrySuspectTriangleCount} bump={LastPureGeometryProtrusionTriangleCount} layer={LastPureGeometryDoubleLayerTriangleCount} ok={LastDoubleLayerValidatedPairCount}\n" +
            $"Layer : ok={LastDoubleLayerValidatedPairCount} miss/norm/plane/lat={LastDoubleLayerRejectedMissingSourceCount}/{LastDoubleLayerRejectedSourceNormalCount}/{LastDoubleLayerRejectedSourcePlaneCount}/{LastDoubleLayerRejectedSourceLateralCount}\n" +
            $"Cause : pend={LastTumorPendingTriangleCount} hBlock={LastTumorReplaceHitBlockedTriangleCount} wBlock={LastTumorReplaceWeightBlockedTriangleCount} band={_auditLockedBandConflictCount}\n" +
            $"Audit : rows={_confidenceAuditRowCount} drop={_confidenceAuditDroppedRows} file={AuditDisplayName()}\n" +
            $"Gate  : dirty={minDirtyTsdfReplaceFrames}/{maxDirtyTsdfReplaceWeight} guard={minGuardedDirtyTsdfReplaceFrames}/{maxGuardedDirtyTsdfReplaceWeight}/{maxGuardedDirtyTsdfReplaceAbsValue:F2}\n" +
            (showRawDepthDebugView || showRawCoverageGridOverlay
                ? "Legend: red=dirty/locked gray=reject yellow=edge pink=pending cyan=stable white=accepted\n"
                : "Legend: red=dirty orange=island yellow=open cyan=stretch\n");
    }

    private string AuditDisplayName()
    {
        if (_confidenceAuditRows != null)
            return _confidenceAuditStem + " (recording)";
        if (string.IsNullOrEmpty(_lastConfidenceAuditPath))
            return "off";
        try
        {
            return Path.GetFileName(_lastConfidenceAuditPath);
        }
        catch (System.Exception)
        {
            return _lastConfidenceAuditPath;
        }
    }

    private Material ResolveMaterial()
    {
        if (shellMaterialOverride != null)
        {
            ApplyMaterialColor(shellMaterialOverride, shellColor);
            ConfigureTransparentMaterial(shellMaterialOverride);
            return shellMaterialOverride;
        }

        if (_runtimeMaterial == null)
        {
            Shader shader = Resources.Load<Shader>("ScanCoverSoftTransparentUnlit");
            if (shader == null)
                shader = Shader.Find("ScanCover/SoftTransparentUnlit");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");

            _runtimeMaterial = new Material(shader);
            _runtimeMaterial.name = "ScanCover_TSDF_SingleShell_Material";
        }

        ApplyMaterialColor(_runtimeMaterial, shellColor);
        ConfigureTransparentMaterial(_runtimeMaterial);
        return _runtimeMaterial;
    }

    private Material ResolveWireMaterial()
    {
        if (_wireRuntimeMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");

            _wireRuntimeMaterial = new Material(shader);
            _wireRuntimeMaterial.name = "ScanCover_TSDF_CoverWire_Material";
        }

        ApplyMaterialColor(_wireRuntimeMaterial, wireColor);
        ConfigureTransparentMaterial(_wireRuntimeMaterial);
        _wireRuntimeMaterial.renderQueue = (int)RenderQueue.Transparent + 20;
        return _wireRuntimeMaterial;
    }

    private Material ResolveDirtyEvidenceMaterial()
    {
        if (_dirtyEvidenceMaterial == null)
        {
            Shader shader = Resources.Load<Shader>("ScanCoverSoftTransparentUnlit");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");
            _dirtyEvidenceMaterial = new Material(shader);
            _dirtyEvidenceMaterial.name = "ScanCover_DirtyTsdfEvidence_Material";
        }

        ApplyMaterialColor(_dirtyEvidenceMaterial, Color.white);
        ConfigureTransparentMaterial(_dirtyEvidenceMaterial);
        _dirtyEvidenceMaterial.renderQueue = (int)RenderQueue.Overlay;
        return _dirtyEvidenceMaterial;
    }

    private Material ResolveRawDepthDebugMaterial()
    {
        if (_rawDepthDebugMaterial == null)
        {
            Shader shader = Resources.Load<Shader>("ScanCoverSoftTransparentUnlit");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");
            _rawDepthDebugMaterial = new Material(shader);
            _rawDepthDebugMaterial.name = "ScanCover_RawDepthDebug_Material";
        }

        ApplyMaterialColor(_rawDepthDebugMaterial, Color.white);
        ConfigureTransparentMaterial(_rawDepthDebugMaterial);
        _rawDepthDebugMaterial.renderQueue = (int)RenderQueue.Transparent + 60;
        return _rawDepthDebugMaterial;
    }

    private static void ApplyMaterialColor(Material material, Color color)
    {
        if (material == null)
            return;
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        if (material.HasProperty("_Cull"))
            material.SetFloat("_Cull", (float)CullMode.Off);
    }

    private static void ConfigureTransparentMaterial(Material material)
    {
        if (material == null)
            return;

        material.SetOverrideTag("RenderType", "Transparent");
        material.SetOverrideTag("Queue", "Transparent");
        if (material.HasProperty("_Surface"))
            material.SetFloat("_Surface", 1f);
        if (material.HasProperty("_AlphaClip"))
            material.SetFloat("_AlphaClip", 0f);
        if (material.HasProperty("_Blend"))
            material.SetFloat("_Blend", 0f);
        if (material.HasProperty("_SrcBlend"))
            material.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (material.HasProperty("_DstBlend"))
            material.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        if (material.HasProperty("_SrcBlendAlpha"))
            material.SetFloat("_SrcBlendAlpha", (float)BlendMode.One);
        if (material.HasProperty("_DstBlendAlpha"))
            material.SetFloat("_DstBlendAlpha", (float)BlendMode.OneMinusSrcAlpha);
        if (material.HasProperty("_ZWrite"))
            material.SetFloat("_ZWrite", 0f);
        if (material.HasProperty("_QueueOffset"))
            material.SetFloat("_QueueOffset", 0f);

        material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.EnableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHATEST_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.renderQueue = (int)RenderQueue.Transparent;
    }

    private static void ConfigureOpaqueMaterial(Material material)
    {
        if (material == null)
            return;

        material.SetOverrideTag("RenderType", "Opaque");
        material.SetOverrideTag("Queue", "Geometry");
        if (material.HasProperty("_Surface"))
            material.SetFloat("_Surface", 0f);
        if (material.HasProperty("_AlphaClip"))
            material.SetFloat("_AlphaClip", 0f);
        if (material.HasProperty("_Blend"))
            material.SetFloat("_Blend", 0f);
        if (material.HasProperty("_SrcBlend"))
            material.SetFloat("_SrcBlend", (float)BlendMode.One);
        if (material.HasProperty("_DstBlend"))
            material.SetFloat("_DstBlend", (float)BlendMode.Zero);
        if (material.HasProperty("_SrcBlendAlpha"))
            material.SetFloat("_SrcBlendAlpha", (float)BlendMode.One);
        if (material.HasProperty("_DstBlendAlpha"))
            material.SetFloat("_DstBlendAlpha", (float)BlendMode.Zero);
        if (material.HasProperty("_ZWrite"))
            material.SetFloat("_ZWrite", 1f);
        if (material.HasProperty("_QueueOffset"))
            material.SetFloat("_QueueOffset", 0f);

        material.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.DisableKeyword("_ALPHABLEND_ON");
        material.DisableKeyword("_ALPHATEST_ON");
        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        material.renderQueue = (int)RenderQueue.Geometry;
    }

    private void SuppressSourcePreview()
    {
        if (!hideSourcePreview || rawDepthSource == null)
            return;

        rawDepthSource.SetPreviewDisplayVisible(false);
    }

    private bool Warn(string message)
    {
        if (debugLog)
            Debug.LogWarning($"[ScanCoverTsdfSingleShellPrototype] {message}", this);
        return false;
    }

    private static bool Finite(Vector3 value)
    {
        return !(float.IsNaN(value.x) || float.IsNaN(value.y) || float.IsNaN(value.z) ||
                 float.IsInfinity(value.x) || float.IsInfinity(value.y) || float.IsInfinity(value.z));
    }
}

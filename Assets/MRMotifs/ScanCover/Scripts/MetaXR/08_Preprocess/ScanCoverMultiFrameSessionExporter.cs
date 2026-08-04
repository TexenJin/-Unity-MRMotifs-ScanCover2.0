using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using UnityEngine.UI;

[DefaultExecutionOrder(-41)]
[DisallowMultipleComponent]
public sealed class ScanCoverMultiFrameSessionExporter : MonoBehaviour
{
    public enum ObservationCaptureMode
    {
        FullCoreAndDirected = 0,
        NearEdgeSupplement = 1,
        BoundaryRiskSupplement = 2
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverDepthGridPointCloud depthGridPointCloud;
    [SerializeField] private Camera captureCamera;
    [SerializeField] private Transform poseSource;
    [SerializeField] private bool autoResolveRefs = true;

    [Header("Session")]
    [SerializeField] private string sessionGroupDirectoryName = "RepeatCoverageSessions";
    [SerializeField] private string sessionNamePrefix = "ScanCover_RepeatCoverage";
    [SerializeField] private bool startSessionOnEnable = false;
    [SerializeField] private bool captureFirstFrameOnStart = true;
    [SerializeField] private bool captureWhileSessionActive = true;
    [SerializeField, Min(1)] private int maxFramesPerSession = 1800;
    [SerializeField, Min(0.02f)] private float minCaptureIntervalSeconds = 0.20f;
    [SerializeField, Min(0f)] private float minMoveMeters = 0.04f;
    [SerializeField, Min(0f)] private float minRotateDegrees = 6f;
    [SerializeField] private bool requestFreshDepthGridBeforeCapture = true;
    [SerializeField, Range(0.02f, 1f)] private float refreshWaitTimeoutSeconds = 0.35f;

    [Header("Editor Local Session Export")]
    [SerializeField] private bool enableEditorLocalSessionExport = false;
    [SerializeField] private string editorLocalSessionExportRoot = "ScanCoverExports";

    [Header("Quest3 Clone Data")]
    [SerializeField] private bool questCloneContinuousCaptureMode = false;
    [SerializeField] private bool questCloneCaptureStaticFrames = true;
    [SerializeField] private bool questCloneUseCompactBinary = true;
    [SerializeField, Range(1, 4)] private int questCloneMaxPendingWrites = 2;
    [SerializeField, Range(1, 32)] private int questCloneProgressSampleStride = 8;
    [SerializeField, Min(1000)] private int questCloneMaxTrackedVoxels = 60000;
    [SerializeField] private string questCloneTargetLabel = "free_scan";
    [SerializeField] private string questCloneActionLabel = "move_smoothly";
    [SerializeField] private Color questCloneHudColor = new Color(0.2f, 0.92f, 1f, 1f);
    [SerializeField] private Color questCloneProgressPendingColor = new Color(0.34f, 0.34f, 0.38f, 0.85f);
    [SerializeField] private Color questCloneProgressSupportedColor = new Color(0.65f, 0.32f, 1f, 0.92f);
    [SerializeField] private Color questCloneProgressStableColor = new Color(0.12f, 0.9f, 1f, 0.95f);
    [SerializeField] private Color questCloneProgressRiskColor = new Color(0.95f, 0.2f, 0.78f, 0.95f);
    [SerializeField, Range(0f, 0.08f)] private float questCloneProgressSurfaceOffsetMeters = 0.025f;
    [SerializeField, Range(0f, 1f)] private float questCloneProgressLowConfidenceThreshold = 0.12f;
    [SerializeField, Range(0.01f, 1f)] private float questCloneProgressRiskEmaAlpha = 0.12f;
    [SerializeField, Min(0.001f)] private float questCloneProgressStablePositionStdMeters = 0.04f;
    [SerializeField, Min(0.001f)] private float questCloneProgressHardPositionStdMeters = 0.07f;
    [SerializeField, Range(0f, 1f)] private float questCloneProgressWarnRecentRiskRatio = 0.35f;
    [SerializeField, Range(0f, 1f)] private float questCloneProgressBadRecentRiskRatio = 0.75f;
    [SerializeField] private bool exportGridStateCsv = true;
        [SerializeField] private bool exportRawDepthProbe = false;
        [SerializeField] private bool exportObservationStats = true;
        [SerializeField] private bool exportQuest3ObservationBadnessStats = false;
    [SerializeField] private ComputeShader rawDepthProbeShader;
    [SerializeField, Min(0.005f)] private float stabilityVoxelSizeMeters = 0.02f;
    [SerializeField, Range(1f, 85f)] private float meshCreaseRiskAngleDegrees = 28f;
    [SerializeField, Min(1)] private int rawDepthHoleMinPixels = 8;
    [SerializeField, Min(0.005f)] private float rawDepthEdgeDepthJumpMeters = 0.08f;

    [Header("Repeat Coverage Gate")]
    [SerializeField] private bool gateFramesByRepeatCoverage = true;
    [SerializeField, Min(0.01f)] private float repeatCoverageVoxelSizeMeters = 0.04f;
    [SerializeField, Min(1)] private int repeatCoverageMinDistinctHits = 3;
    [SerializeField, Min(1)] private int repeatCoverageTargetStableVoxels = 12000;
    [SerializeField, Min(1)] private int repeatCoverageMinCandidateVoxels = 600;
    [SerializeField, Min(1)] private int repeatCoverageMinNewStableVoxelsPerFrame = 60;
    [SerializeField, Min(1)] private int repeatCoverageMinNewOrRehitVoxelsPerFrame = 220;
    [SerializeField, Min(0f)] private float repeatCoverageMinParallaxMeters = 0.025f;
    [SerializeField, Range(0f, 89f)] private float repeatCoverageMaxViewAngleDegrees = 72f;

    [Header("Room Raw Coverage HUD")]
    [SerializeField] private bool trackRoomRawCoverage = true;
    [SerializeField] private bool useLegacyFullViewRoomRawCapture = true;
    [SerializeField] private bool roomRawCoverageHudOnlyCapture = false;
    [SerializeField] private bool hideDepthGridPreviewWhileRecording = true;
    [SerializeField] private bool lockRoomRawCoverageToStartView = false;
    [SerializeField] private bool allowSameViewStableRoomRawHits = true;
    [SerializeField, Min(0.01f)] private float roomRawCoverageVoxelSizeMeters = 0.08f;
    [SerializeField, Min(1)] private int roomRawCoverageTargetVoxels = 25000;
    [SerializeField, Min(1)] private int roomRawCoverageTargetStableVoxels = 12000;
    [SerializeField, Min(1)] private int roomRawCoverageTargetHighVoxels = 2500;
    [SerializeField, Min(1)] private int roomRawCoverageTargetHighStableVoxels = 1200;
    [SerializeField, Min(1)] private int roomRawCoverageTargetLowVoxels = 2500;
    [SerializeField, Min(1)] private int roomRawCoverageTargetLowStableVoxels = 1200;
    [SerializeField, Min(1)] private int roomRawCoverageStableHitTarget = 3;
    [SerializeField, Min(1)] private int roomRawCoverageTargetPoseCells = 80;
    [SerializeField, Min(1)] private int roomRawCoverageTargetValidSamples = 750000;
    [SerializeField, Min(1)] private int roomRawCoverageTargetFrames = 180;
    [SerializeField, Min(1)] private int roomRawCoverageLocalTargetVoxels = 6500;
    [SerializeField, Min(1)] private int roomRawCoverageLocalTargetStableVoxels = 4200;
    [SerializeField, Min(1)] private int roomRawCoverageLocalTargetValidSamples = 180000;
    [SerializeField, Min(1)] private int roomRawCoverageLocalTargetFrames = 45;
    [SerializeField, Min(0f)] private float roomRawCoverageStableParallaxMeters = 0.03f;
    [SerializeField, Min(0.1f)] private float roomRawCoveragePoseCellSizeMeters = 0.25f;
    [SerializeField, Range(0f, 89f)] private float roomRawCoverageRiskViewAngleDegrees = 72f;
    [SerializeField, Min(0.1f)] private float roomRawCoverageHighPointMinDeltaYMeters = 0.75f;
    [SerializeField, Min(0.1f)] private float roomRawCoverageLowPointMinDeltaYMeters = 0.45f;
    [SerializeField, Min(0.005f)] private float roomRawCoverageNeighborDepthJumpMeters = 0.08f;
    [SerializeField, Range(0f, 1f)] private float roomRawCoverageTargetRiskRatio = 0.08f;
    [SerializeField] private bool gateRoomRawCoverageFrames = false;
    [SerializeField, Range(0f, 0.45f)] private float roomRawCoverageFocusMargin = 0.18f;
    [SerializeField] private bool roomRawCoverageUseFallbackTolerance = true;
    [SerializeField, Range(0f, 0.45f)] private float roomRawCoverageCoreMargin = 0.24f;
    [SerializeField, Range(0f, 0.45f)] private float roomRawCoverageEdgeBufferMargin = 0.08f;
    [SerializeField, Range(0f, 30f)] private float roomRawCoverageAnchorAngleFallbackDegrees = 8f;
    [SerializeField, Min(0f)] private float roomRawCoverageAnchorMoveFallbackMeters = 0.12f;
    [SerializeField, Range(1f, 45f)] private float roomRawCoverageMaxAnchorAngleDegrees = 14f;
    [SerializeField, Min(0f)] private float roomRawCoverageMaxAnchorMoveMeters = 0.25f;
    [SerializeField, Min(0.01f)] private float roomRawCoverageMinDepthMeters = 0.15f;
    [SerializeField, Min(0.1f)] private float roomRawCoverageMaxDepthMeters = 5f;
    [SerializeField, Min(0.01f)] private float roomRawCoverageMaxDepthProjectionErrorMeters = 0.45f;
    [SerializeField, Min(1)] private int roomRawCoverageMinFocusVoxels = 160;
    [SerializeField, Min(1)] private int roomRawCoverageMinNewVoxelsPerFrame = 90;
    [SerializeField, Min(1)] private int roomRawCoverageMinOverlapNewVoxelsPerFrame = 20;
    [SerializeField, Min(1)] private int roomRawCoverageMinNewStableVoxelsPerFrame = 18;
    [SerializeField, Min(1)] private int roomRawCoverageMinNewHighLowVoxelsPerFrame = 16;
    [SerializeField, Min(1)] private int roomRawCoverageMinParallaxRehitVoxelsPerFrame = 30;
    [SerializeField, Min(1)] private int roomRawCoverageMinOverlapRehitVoxelsPerFrame = 80;
    [SerializeField, Min(1)] private int roomRawCoverageMinCoreVoxels = 80;
    [SerializeField, Min(1)] private int roomRawCoverageMinEdgeBufferVoxels = 80;
    [SerializeField, Min(1)] private int roomRawCoverageMinHistoryRehitVoxelsPerFrame = 120;
    [SerializeField, Min(1)] private int roomRawCoverageMinRiskSamplesPerFrame = 80;
    [SerializeField, Range(0f, 1f)] private float roomRawCoverageIdealOverlapMinRatio = 0.20f;
    [SerializeField, Range(0f, 1f)] private float roomRawCoverageIdealOverlapMaxRatio = 0.40f;
    [SerializeField, Range(0f, 1f)] private float roomRawCoverageMinHistoryAgreementRatio = 0.70f;
    [SerializeField, Range(5f, 90f)] private float roomRawCoverageViewMeterHalfAngleDegrees = 34f;
    [SerializeField, Min(0.1f)] private float roomRawCoverageViewMeterFarDepthMeters = 1.5f;
    [SerializeField, Min(1)] private int roomRawCoverageViewMeterMinVoxels = 80;
    [SerializeField] private bool loadPreviousRoomRawCoverageOnStart = true;
    [SerializeField, Min(1)] private int maxPreviousRoomRawCoverageSessionsToLoad = 1;
    [SerializeField] private string previousRoomRawCoverageOverrideDirectory = "";
    [SerializeField] private bool autoStopLegacyRoomRawCoverageWhenTargetsComplete = true;
    [SerializeField] private bool autoStopLegacyRoomRawCoverageOnPlateau = true;
    [SerializeField, Min(1)] private int legacyRoomRawCoverageMinFramesBeforePlateau = 24;
    [SerializeField, Min(0.5f)] private float legacyRoomRawCoveragePlateauSeconds = 8f;
    [SerializeField, Min(0)] private int legacyRoomRawCoveragePlateauMinCoveredVoxels = 140;
    [SerializeField, Min(0)] private int legacyRoomRawCoveragePlateauMinStableVoxels = 28;

    private enum RoomRawCoverageDisorderFuseMode
    {
        Strict,
        RejectOnly,
        Off
    }

    private enum BinocularRoomRawDepthSnapshotStage
    {
        None,
        RightRequested,
        LeftRequested,
        Ready
    }

    [Header("Room Raw Coverage Disorder Fuse")]
    [SerializeField] private RoomRawCoverageDisorderFuseMode roomRawCoverageDisorderFuseMode = RoomRawCoverageDisorderFuseMode.Strict;
    [SerializeField] private bool enforceObservationOrderScoreGate = true;
    [SerializeField] private bool autoStopLegacyRoomRawCoverageOnDisorder = true;
    [SerializeField, Min(1)] private int legacyRoomRawCoverageDisorderMinFrames = 1;
    [SerializeField, Min(1)] private int legacyRoomRawCoverageDisorderConsecutiveFrames = 1;
    [SerializeField, Min(1)] private int legacyRoomRawCoverageDisorderMinValidSamples = 32;
    [SerializeField, Range(0f, 1f)] private float legacyRoomRawCoverageDisorderRiskRatio = 0.06f;
    [SerializeField, Min(0)] private int legacyRoomRawCoverageDisorderMinUsefulVoxels = 32;
    [SerializeField, Range(0f, 1f)] private float legacyRoomRawCoverageDisorderLowUsefulRiskRatio = 0.05f;
    [SerializeField, Range(0f, 1f)] private float legacyRoomRawCoverageHardFuseBadEdgeRatio = 0.055f;
    [SerializeField, Range(0f, 1f)] private float legacyRoomRawCoverageHardFuseCenterBadEdgeRatio = 0.055f;
    [SerializeField, Range(0f, 1f)] private float legacyRoomRawCoverageHardFuseBadQuadRatio = 0.040f;
    [SerializeField, Range(0f, 1f)] private float legacyRoomRawCoverageHardFuseRiskRatio = 0.22f;
    [SerializeField] private bool useRawObservationOrderScore = true;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderWarnRatio = 0.68f;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderFuseRatio = 0.45f;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderWarnBadEdgeRatio = 0.02f;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderFuseBadEdgeRatio = 0.045f;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderWarnCenterBadEdgeRatio = 0.015f;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderFuseCenterBadEdgeRatio = 0.035f;
    [SerializeField, Range(0f, 1f)] private float rawObservationOrderFuseBadQuadRatio = 0.06f;
    [SerializeField, Min(1)] private int rawObservationOrderMinEdges = 32;
    [SerializeField, Range(1, 8)] private int rawObservationOrderPixelStride = 1;
    [SerializeField, Min(0.001f)] private float rawObservationOrderMaxNeighborDistanceMeters = 0.18f;
    [SerializeField, Min(0.001f)] private float rawObservationOrderDepthJumpMeters = 0.08f;
    [SerializeField, Range(-1f, 1f)] private float rawObservationOrderMinNormalDot = 0.15f;
    [SerializeField, Min(0.001f)] private float rawObservationOrderQuadMaxDiagonalSkewMeters = 0.12f;

    [Header("Room Raw Depth Dense Export")]
    [SerializeField] private bool exportRoomRawDepthFrames = true;
    [SerializeField] private bool exportRoomRawDepthOnlyAcceptedFrames = false;
    [SerializeField, Min(1)] private int roomRawDepthFrameStride = 1;
    [SerializeField] private bool roomRawDepthFocusOnly = false;
    [SerializeField, Min(0)] private int roomRawDepthMaxSamplesPerFrame = 0;

    [Header("Room Raw Depth Snapshot")]
    [SerializeField] private bool exportRoomRawDepthSnapshots = true;
    [SerializeField] private bool showRoomRawDepthSnapshotOverlay = true;
    [SerializeField] private bool captureNowUsesIndependentSnapshotLine = true;
    [SerializeField] private string independentSnapshotDirectoryName = "SnapshotCaptures";
    [SerializeField] private string independentSnapshotNamePrefix = "ScanCover_BinocularSnapshot";
    [SerializeField] private bool useBinocularRoomRawDepthSnapshots = true;
    [SerializeField] private bool binocularRoomRawDepthSnapshotsManualOnly = true;
    [SerializeField, Min(0.05f)] private float binocularRoomRawDepthSnapshotTimeoutSeconds = 0.80f;
    [SerializeField] private bool exportVirtualCloneInputMetadata = true;
    [SerializeField] private string virtualCloneInputDirectoryName = "virtual_clone_input";
    [SerializeField, Min(0.001f)] private float virtualCloneEyeBaselineMeters = 0.063f;
    [SerializeField, Min(1f)] private float virtualCloneFallbackFieldOfViewDegrees = 100.2439f;
    [SerializeField, Min(0.002f)] private float roomRawDepthSnapshotPointSize = 0.014f;
    [SerializeField, Min(1024)] private int roomRawDepthSnapshotMaxVisualPoints = 51200;
    [SerializeField] private Color roomRawDepthSnapshotColor = new Color(0.12f, 0.85f, 1f, 1f);
    [SerializeField] private bool fuseRoomRawDepthSnapshotOverlayOnly = true;
    [SerializeField, Min(0.002f)] private float roomRawDepthSnapshotOverlayFuseVoxelMeters = 0.015f;

    [Header("Room Raw Depth Completion Overlay")]
    [SerializeField] private bool showRoomRawDepthCompletionOverlay = false;
    [SerializeField] private bool roomRawDepthCompletionOnlyWhileRecording = false;
    [SerializeField, Min(0.02f)] private float roomRawDepthCompletionVoxelSizeMeters = 0.08f;
    [SerializeField, Min(1)] private int roomRawDepthCompletionMinHits = 3;
    [SerializeField, Min(1)] private int roomRawDepthCompletionStableHits = 8;
    [SerializeField, Min(1)] private int roomRawDepthCompletionCandidateStableHits = 14;
    [SerializeField, Min(1)] private int roomRawDepthCompletionSameViewStableHits = 24;
    [SerializeField, Range(0f, 90f)] private float roomRawDepthCompletionMinAngleSpanDegrees = 12f;
    [SerializeField, Min(0.001f)] private float roomRawDepthCompletionStableDepthStdMeters = 0.08f;
    [SerializeField, Min(0.001f)] private float roomRawDepthCompletionHardDepthStdMeters = 0.20f;
    [SerializeField, Range(0f, 1f)] private float roomRawDepthCompletionWarnRiskRatio = 0.25f;
    [SerializeField, Range(0f, 1f)] private float roomRawDepthCompletionRecoverableRiskRatio = 0.45f;
    [SerializeField, Range(0f, 1f)] private float roomRawDepthCompletionBadRiskRatio = 0.65f;
    [SerializeField, Min(0)] private int roomRawDepthCompletionNeighborStableSupport = 4;
    [SerializeField, Min(0.001f)] private float roomRawDepthCompletionNeighborDepthDeltaMeters = 0.12f;
    [SerializeField, Min(0.001f)] private float roomRawDepthCompletionNeighborStableDepthStdMeters = 0.12f;
    [SerializeField, Range(0f, 1f)] private float roomRawDepthCompletionNeighborRiskRatio = 0.50f;
    [SerializeField] private bool roomRawDepthCompletionUseVoxelCenterPresentation = false;
    [SerializeField] private bool roomRawDepthCompletionUseGpuInstancing = false;
    [SerializeField, Min(0.005f)] private float roomRawDepthCompletionPointSize = 0.025f;
    [SerializeField, Min(64)] private int roomRawDepthCompletionMaxVisualPoints = 1200;
    [SerializeField, Min(0.05f)] private float roomRawDepthCompletionRefreshSeconds = 0.25f;

    [Header("Snapshot Grid Guided Raw Depth Capture")]
    [SerializeField] private bool useSnapshotGridCaptureMask = false;
    [SerializeField] private bool snapshotGridMaskRequiredForRawDepthExport = false;
    [SerializeField, Min(0.01f)] private float snapshotGridCaptureRadiusMeters = 0.08f;
    [SerializeField, Min(1)] private int snapshotGridSeenHits = 2;
    [SerializeField, Min(1)] private int snapshotGridStableHits = 6;
    [SerializeField, Range(0f, 1f)] private float snapshotGridRiskRatio = 0.35f;
    [SerializeField] private Color snapshotGridPendingColor = new Color(0.35f, 0.35f, 0.35f, 1f);
    [SerializeField] private Color snapshotGridSeenColor = new Color(0.55f, 0.55f, 0.55f, 1f);
    [SerializeField] private Color snapshotGridStableColor = new Color(0.55f, 0.55f, 0.55f, 1f);
    [SerializeField] private Color snapshotGridRiskColor = new Color(0.55f, 0.55f, 0.55f, 1f);

    [Header("Quest3 Observation Badness Targets")]
    [SerializeField, Min(1)] private int targetBadnessFrames = 120;
    [SerializeField, Min(1)] private int targetBadnessInvalidFrames = 20;
    [SerializeField, Min(1)] private int targetBadnessLargeHoleFrames = 12;
    [SerializeField, Min(1)] private int targetBadnessEdgeJumpFrames = 60;
    [SerializeField, Min(1)] private int targetBadnessPersistentInvalidFrames = 12;
    [SerializeField, Min(1)] private int minBadnessInvalidPixelsPerEye = 1;
    [SerializeField, Min(1)] private int minBadnessEdgeRiskPixelsPerEye = 500;

    [Header("Targeted Observation Gate")]
    [SerializeField] private bool gateFramesByObservationTargets = false;
    [SerializeField] private ObservationCaptureMode observationCaptureMode = ObservationCaptureMode.FullCoreAndDirected;
    [SerializeField, Min(1)] private int targetFramesPerObservationBucket = 60;
    [SerializeField] private bool useMultiCoreObservationTargets = true;
    [SerializeField, Min(1)] private int targetFramesPerCoreMetric = 45;
    [SerializeField, Min(1)] private int targetNearEdgeSupplementFrames = 45;
    [SerializeField, Min(1)] private int minStableMainPointsPerFrame = 450;
    [SerializeField, Min(1)] private int minCoverageBandPointsPerFrame = 450;
    [SerializeField, Min(1)] private int minTypeBucketPointsPerFrame = 260;
    [SerializeField, Min(1)] private int minRiskLayerPointsPerFrame = 120;
    [SerializeField, Min(1)] private int minFarSurfacePointsPerFrame = 300;
    [SerializeField, Min(1)] private int minHighAngleRiskPointsPerFrame = 80;
    [SerializeField, Min(1)] private int minNearEdgeRiskPointsPerFrame = 80;
    [SerializeField, Min(1)] private int minNearEdgeSupplementRiskPointsPerFrame = 80;
    [SerializeField, Min(1)] private int minNearEdgeSupplementCreasePointsPerFrame = 60;
    [SerializeField, Min(1)] private int minNearEdgeSupplementBoundaryPointsPerFrame = 40;
    [SerializeField, Min(1)] private int minDirectedMainPointsPerFrame = 180;
    [SerializeField, Min(1)] private int minDirectionalBoundaryPointsPerFrame = 60;
    [SerializeField, Min(1)] private int minCornerJunctionPointsPerFrame = 60;
    [SerializeField] private Vector2 farSurfaceDistanceRangeMeters = new Vector2(3f, 5f);
    [SerializeField] private Vector2 midSurfaceDistanceRangeMeters = new Vector2(1f, 3f);
    [SerializeField] private Vector2 nearEdgeDistanceRangeMeters = new Vector2(0.3f, 1f);
    [SerializeField] private Vector2 nearSurfaceDistanceRangeMeters = new Vector2(0.3f, 1f);
    [SerializeField, Range(0f, 89f)] private float stableMainMaxAngleDegrees = 45f;
    [SerializeField, Range(0f, 89f)] private float frontViewMaxAngleDegrees = 40f;
    [SerializeField, Range(0f, 89f)] private float obliqueViewMinAngleDegrees = 40f;
    [SerializeField, Range(0f, 89f)] private float obliqueViewMaxAngleDegrees = 75f;
    [SerializeField, Range(0f, 89f)] private float highAngleThresholdDegrees = 60f;
    [SerializeField, Range(0f, 89f)] private float nearEdgeSupplementObliqueMinAngleDegrees = 35f;
    [SerializeField, Range(0f, 89f)] private float nearEdgeSupplementExtremeMinAngleDegrees = 60f;
    [SerializeField, Range(0.05f, 0.45f)] private float verticalCoverageBandRatio = 0.28f;
    [SerializeField, Range(0.05f, 0.8f)] private float directionalSideThreshold = 0.22f;

    [Header("Input")]
    [SerializeField] private bool enableKeyboardInput = true;
    [SerializeField] private Key keyboardStartStopKey = Key.F9;
    [SerializeField] private Key keyboardCaptureNowKey = Key.F10;
    [SerializeField] private bool enableOvrInput = true;
    [SerializeField] private bool enableOvrStartStopInput = true;
    [SerializeField] private bool enableOvrCaptureNowInput = false;
    [SerializeField] private bool useRightIndexTriggerForCaptureNow = false;
    [SerializeField] private OVRInput.RawButton ovrStartStopButton = OVRInput.RawButton.B;
    [SerializeField] private OVRInput.RawButton ovrCaptureNowButton = OVRInput.RawButton.A;

    [Header("HUD")]
    [SerializeField] private bool showHud = true;
    [SerializeField] private Vector3 hudLocalPosition = new Vector3(0f, -0.12f, 0.95f);
    [SerializeField] private Vector3 hudLocalEuler = Vector3.zero;
    [SerializeField] private Vector3 hudLocalScale = new Vector3(0.000675f, 0.000675f, 0.000675f);
    [SerializeField, Min(180)] private int hudPanelWidth = 820;
    [SerializeField, Min(90)] private int hudPanelHeight = 420;
    [SerializeField, Min(0.05f)] private float hudRefreshIntervalSeconds = 0.15f;
    [SerializeField] private Color hudIdleColor = new Color(0.78f, 0.9f, 1f, 1f);
    [SerializeField] private Color hudRecordingColor = new Color(0.45f, 1f, 0.62f, 1f);
    [SerializeField] private Color hudPendingColor = new Color(1f, 0.86f, 0.28f, 1f);
    [SerializeField] private Color hudTextColor = new Color(1f, 1f, 1f, 1f);
    [SerializeField] private Color hudIdleTextColor = new Color(0.68f, 0.72f, 0.76f, 1f);
    [SerializeField] private Color hudRecordingTextColor = new Color(1f, 1f, 1f, 1f);
    [SerializeField] private Color hudSamplingTextColor = new Color(1f, 0.86f, 0.35f, 1f);
    [SerializeField] private Color hudPanelColor = new Color(0f, 0f, 0f, 0.78f);

    [Header("Room Raw Coverage View Frame")]
    [SerializeField] private bool showRoomRawCoverageViewFrame = false;
    [SerializeField] private Vector3 roomRawCoverageViewFrameLocalPosition = new Vector3(0f, 0f, 1.05f);
    [SerializeField, Min(0.05f)] private float roomRawCoverageViewFrameWidth = 0.72f;
    [SerializeField, Min(0.05f)] private float roomRawCoverageViewFrameHeight = 0.46f;
    [SerializeField, Min(0.001f)] private float roomRawCoverageViewFrameLineWidth = 0.006f;
    [SerializeField] private Color roomRawCoverageViewFrameIdleColor = new Color(0f, 0.85f, 1f, 0.9f);
    [SerializeField] private Color roomRawCoverageViewFrameRecordingColor = new Color(0.25f, 1f, 0.35f, 1f);
    [SerializeField] private Color roomRawCoverageViewFrameOutOfAnchorColor = new Color(1f, 0.25f, 0.15f, 1f);
    [SerializeField] private bool freezeRoomRawCoverageViewFrameWhileRecording = false;
    [SerializeField, Range(0f, 0.4f)] private float roomRawCoverageViewFrameLeaveMargin = 0.08f;
    [SerializeField, Range(0f, 0.8f)] private float roomRawCoverageViewFrameVerticalLeaveMargin = 0.22f;
    [SerializeField, Range(0f, 0.8f)] private float roomRawCoverageViewFrameTopExtraLeaveMargin = 0.18f;
    [SerializeField] private bool showRoomRawCoverageTileHints = false;
    [SerializeField, Range(2, 8)] private int roomRawCoverageTileColumns = 5;
    [SerializeField, Range(2, 6)] private int roomRawCoverageTileRows = 3;
    [SerializeField, Min(1)] private int roomRawCoverageTileTargetCoveredVoxels = 28;
    [SerializeField, Min(1)] private int roomRawCoverageTileTargetStableVoxels = 10;
    [SerializeField, Min(0.001f)] private float roomRawCoverageTileLineWidth = 0.003f;
    [SerializeField] private Color roomRawCoverageTileMissingColor = new Color(1f, 0.08f, 0.04f, 0.95f);
    [SerializeField] private Color roomRawCoverageTilePartialColor = new Color(1f, 0.82f, 0.05f, 0.95f);
    [SerializeField] private Color roomRawCoverageTileCompleteColor = new Color(0.08f, 1f, 0.32f, 0.95f);
    [SerializeField] private bool showRoomRawCoverageReticle = false;
    [SerializeField, Min(0.005f)] private float roomRawCoverageReticleSize = 0.055f;
    [SerializeField, Min(0.001f)] private float roomRawCoverageReticleLineWidth = 0.006f;
    [SerializeField] private Color roomRawCoverageReticleColor = new Color(0f, 1f, 1f, 1f);
    [SerializeField, Min(1f)] private float roomRawCoverageAimTileLineWidthMultiplier = 2.4f;

    [Header("Room Raw Coverage Preview Points")]
    [SerializeField] private bool showRoomRawCoveragePreviewPoints = false;
    [SerializeField, Min(32)] private int roomRawCoveragePreviewMaxPoints = 700;
    [SerializeField, Min(0.002f)] private float roomRawCoveragePreviewPointSize = 0.008f;
    [SerializeField, Min(0.05f)] private float roomRawCoveragePreviewRefreshSeconds = 0.18f;
    [SerializeField] private Color roomRawCoveragePreviewUniformColor = new Color(0.55f, 0.55f, 0.55f, 1f);

    public bool IsRecording => _isRecording;
    public int CapturedFrameCount => _capturedFrameCount;
    public string CurrentSessionDirectory => _sessionDirectory;

    private bool _isRecording;
    private bool _pendingCapture;
    private bool _pendingRefreshRequested;
    private bool _stopAfterPendingManualCapture;
    private bool _independentSnapshotCaptureActive;
    private int _pendingStartDepthFrame = -1;
    private float _pendingStartTime;
    private string _pendingReason;
    private bool _restoreDepthGridPreviewDisplay;
    private bool _hasDepthGridPreviewDisplayRestoreState;
    private bool _sessionFileExportEnabled;
    private string _sessionDirectory;
    private string _framesDirectory;
    private string _manifestPath;
    private string _observationDirectory;
    private string _frameObservationStatsPath;
    private string _distanceBinsPath;
    private string _angleBinsPath;
    private string _edgeRiskStatsPath;
    private string _targetGatePath;
    private string _quest3BadnessDirectory;
    private string _rawDepthBadnessFramesPath;
    private string _rawDepthHoleComponentsPath;
    private string _rawDepthBadnessSummaryPath;
    private string _repeatCoverageDirectory;
    private string _repeatCoverageGatePath;
    private string _repeatCoverageVoxelsPath;
    private string _roomRawCoverageDirectory;
    private string _roomRawCoverageFramesPath;
    private string _roomRawCoverageVoxelsPath;
    private string _roomRawCoverageSummaryPath;
    private string _roomRawDepthDirectory;
    private string _roomRawDepthManifestPath;
    private string _roomRawDepthSummaryPath;
    private string _roomRawDepthSnapshotDirectory;
    private string _roomRawDepthSnapshotManifestPath;
    private string _virtualCloneInputDirectory;
    private string _virtualCloneInputManifestPath;
    private int _capturedFrameCount;
    private int _roomRawDepthExportedFrames;
    private int _roomRawDepthExportedSamples;
    private int _roomRawDepthSkippedFrames;
    private int _roomRawDepthSnapshotExportedFrames;
    private int _roomRawDepthSnapshotTotalPixels;
    private int _roomRawDepthSnapshotValidPixels;
    private readonly Dictionary<Vector3Int, List<int>> _snapshotGridMaskCells = new Dictionary<Vector3Int, List<int>>(4096);
    private Vector3[] _snapshotGridMaskPositions = new Vector3[0];
    private int[] _snapshotGridMaskEntryIndices = new int[0];
    private int[] _snapshotGridMaskHits = new int[0];
    private int[] _snapshotGridMaskRiskHits = new int[0];
    private byte[] _snapshotGridMaskVisualStates = new byte[0];
    private int _snapshotGridMaskTotal;
    private int _snapshotGridMaskSeen;
    private int _snapshotGridMaskStable;
    private int _snapshotGridMaskRisk;
    private bool _snapshotGridMaskActive;
    private bool _snapshotGridMaskHasSignature;
    private int _snapshotGridMaskSignature;
    private bool _snapshotGridMaskSuppressUntilSignatureChanges;
    private int _snapshotGridMaskSuppressedSignature;
    private string _snapshotGridMaskStatus = "inactive";
    private int _rawDepthBadnessFrameCount;
    private int _rawDepthBadnessEyeSamples;
    private int _rawDepthBadnessValidSamples;
    private int _rawDepthBadnessInvalidSamples;
    private int _rawDepthBadnessLargeHoleComponents;
    private int _rawDepthBadnessPersistentInvalidPixels;
    private int _rawDepthBadnessNewlyInvalidPixels;
    private int _rawDepthBadnessRecoveredPixels;
    private int _rawDepthBadnessEdgeRiskPixels;
    private int _rawDepthBadnessInvalidFrames;
    private int _rawDepthBadnessLargeHoleFrames;
    private int _rawDepthBadnessEdgeJumpFrames;
    private int _rawDepthBadnessPersistentInvalidFrames;
    private int _targetFarSurfaceFrames;
    private int _targetHighAngleRiskFrames;
    private int _targetNearEdgeRiskFrames;
    private int _targetStableMainFrames;
    private int _targetTopCoverageFrames;
    private int _targetMiddleCoverageFrames;
    private int _targetBottomCoverageFrames;
    private int _targetNearDistanceFrames;
    private int _targetMidDistanceFrames;
    private int _targetFarDistanceFrames;
    private int _targetFrontAngleFrames;
    private int _targetObliqueAngleFrames;
    private int _targetExtremeAngleFrames;
    private int _targetRiskLayerFrames;
    private int _targetMainFrontFrames;
    private int _targetMainLeftObliqueFrames;
    private int _targetMainRightObliqueFrames;
    private int _targetMainUpObliqueFrames;
    private int _targetMainDownObliqueFrames;
    private int _targetBoundaryLeftFrames;
    private int _targetBoundaryRightFrames;
    private int _targetCornerJunctionFrames;
    private int _targetNearEdgeSupplementRiskFrames;
    private int _targetNearEdgeSupplementCreaseFrames;
    private int _targetNearEdgeSupplementBoundaryFrames;
    private int _targetNearEdgeSupplementLeftFrames;
    private int _targetNearEdgeSupplementRightFrames;
    private int _targetNearEdgeSupplementUpFrames;
    private int _targetNearEdgeSupplementDownFrames;
    private int _targetNearEdgeSupplementObliqueFrames;
    private int _targetNearEdgeSupplementExtremeFrames;
    private int _targetRejectedFrames;
    private int _repeatCoverageAcceptedFrames;
    private int _repeatCoverageRejectedFrames;
    private int _repeatCoverageStableVoxelCount;
    private int _lastRepeatCandidateVoxels;
    private int _lastRepeatNewVoxels;
    private int _lastRepeatRehitVoxels;
    private int _lastRepeatParallaxRehitVoxels;
    private int _lastRepeatNewStableVoxels;
    private string _lastRepeatCoverageStatus = "none";
    private int _roomRawCoverageFrames;
    private int _roomRawCoverageValidSamples;
    private int _roomRawCoverageTotalSamples;
    private int _roomRawCoverageCoveredVoxels;
    private int _roomRawCoverageStableVoxels;
    private int _roomRawCoverageRiskVoxels;
    private int _roomRawCoverageHighVoxels;
    private int _roomRawCoverageHighStableVoxels;
    private int _roomRawCoverageLowVoxels;
    private int _roomRawCoverageLowStableVoxels;
    private int _roomRawCoveragePoseCellCount;
    private int _roomRawCoverageRejectedFrames;
    private int _lastRoomRawValidSamples;
    private int _lastRoomRawFocusVoxels;
    private int _lastRoomRawFocusNewVoxels;
    private int _lastRoomRawNewVoxels;
    private int _lastRoomRawRehitVoxels;
    private int _lastRoomRawHistoryRehitVoxels;
    private int _lastRoomRawHistoryAgreementVoxels;
    private int _lastRoomRawHistoryConflictVoxels;
    private int _lastRoomRawParallaxRehitVoxels;
    private int _lastRoomRawStableNewVoxels;
    private int _lastRoomRawHighFrameVoxels;
    private int _lastRoomRawNewHighVoxels;
    private int _lastRoomRawNewHighStableVoxels;
    private int _lastRoomRawLowFrameVoxels;
    private int _lastRoomRawNewLowVoxels;
    private int _lastRoomRawNewLowStableVoxels;
    private float _lastRoomRawRiskRatio;
    private float _lastRoomRawOverlapRatio;
    private float _lastRoomRawHistoryAgreementRatio;
    private bool _roomRawCoverageWindowActive;
    private int _roomRawCoverageWindowStartFrames;
    private int _roomRawCoverageWindowStartValidSamples;
    private int _roomRawCoverageWindowStartCoveredVoxels;
    private int _roomRawCoverageWindowStartStableVoxels;
    private int _lastRoomRawWindowFrames;
    private int _lastRoomRawWindowValidSamples;
    private int _lastRoomRawWindowCoveredVoxels;
    private int _lastRoomRawWindowStableVoxels;
    private int _loadedRoomRawCoverageSessions;
    private int _loadedRoomRawCoverageVoxels;
    private int _loadedRoomRawCoverageStableVoxels;
    private string _loadedRoomRawCoverageSource = "none";
    private bool _roomRawCoverageAnchorSet;
    private Vector3 _roomRawCoverageAnchorPosition;
    private Quaternion _roomRawCoverageAnchorRotation = Quaternion.identity;
    private float _lastRoomRawAnchorAngle;
    private float _lastRoomRawAnchorMove;
    private string _lastRoomRawStatus = "none";
    private string _lastRoomRawFuseStatus = string.Empty;
    private ObservationOrderScore _lastObservationOrderScore;
    private float _legacyRoomRawCoverageLastProgressTime;
    private int _legacyRoomRawCoverageLastCoveredVoxels;
    private int _legacyRoomRawCoverageLastStableVoxels;
    private int _legacyRoomRawCoverageDisorderFrames;
    private float _lastCaptureTime = float.NegativeInfinity;
    private Vector3 _lastCapturePosition;
    private Quaternion _lastCaptureRotation = Quaternion.identity;
    private bool _hasLastCapturePose;
    private GameObject _hudRoot;
    private Canvas _hudCanvas;
    private Text _hudText;
    private Image _hudPanelImage;
    private float _nextHudRefreshTime;
    // 网格链诊断行：2s 节流查找，判空即跳过（两链都是运行时创建）。
    private ScanCoverTsdfBranch _diagTsdfBranch;
    private ScanCoverQuestRoomSurfaceNetsPipeline _diagQuestRoomPipeline;
    private float _nextDiagLookupTime;
    private GameObject _roomRawCoverageViewFrameRoot;
    private LineRenderer _roomRawCoverageViewFrameLine;
    private Material _roomRawCoverageViewFrameMaterial;
    private LineRenderer[] _roomRawCoverageTileLines = new LineRenderer[0];
    private Material[] _roomRawCoverageTileMaterials = new Material[0];
    private GameObject _roomRawCoverageReticleRoot;
    private LineRenderer _roomRawCoverageReticleHorizontalLine;
    private LineRenderer _roomRawCoverageReticleVerticalLine;
    private Material _roomRawCoverageReticleMaterial;
    private bool _roomRawCoverageViewFrameLocked;
    private Vector3 _roomRawCoverageViewFrameWorldPosition;
    private Quaternion _roomRawCoverageViewFrameWorldRotation = Quaternion.identity;
    private GameObject _roomRawCoveragePreviewRoot;
    private float _nextRoomRawCoveragePreviewRefreshTime;
    private readonly Mesh[] _roomRawCoveragePreviewMeshes = new Mesh[3];
    private readonly MeshFilter[] _roomRawCoveragePreviewMeshFilters = new MeshFilter[3];
    private readonly MeshRenderer[] _roomRawCoveragePreviewMeshRenderers = new MeshRenderer[3];
    private readonly Material[] _roomRawCoveragePreviewMaterials = new Material[3];
    private GameObject _roomRawDepthCompletionRoot;
    private Mesh _roomRawDepthCompletionMesh;
    private Material _roomRawDepthCompletionMaterial;
    private readonly Material[] _roomRawDepthCompletionInstancedMaterials = new Material[5];
    private MeshFilter _roomRawDepthCompletionCombinedMeshFilter;
    private MeshRenderer _roomRawDepthCompletionCombinedMeshRenderer;
    private Mesh _roomRawDepthCompletionCombinedMesh;
    private readonly List<Vector3> _roomRawDepthCompletionCombinedVertices = new List<Vector3>(9600);
    private readonly List<int>[] _roomRawDepthCompletionCombinedIndices =
    {
        new List<int>(2048),
        new List<int>(2048),
        new List<int>(2048),
        new List<int>(2048),
        new List<int>(2048)
    };
    private Material _roomRawDepthSnapshotMaterial;
    private GameObject _roomRawDepthSnapshotMeshRoot;
    private MeshFilter _roomRawDepthSnapshotMeshFilter;
    private MeshRenderer _roomRawDepthSnapshotMeshRenderer;
    private Mesh _roomRawDepthSnapshotMesh;
    private MaterialPropertyBlock _roomRawDepthCompletionPropertyBlock;
    private readonly List<GameObject> _roomRawDepthCompletionPointObjects = new List<GameObject>(2048);
    private readonly List<Matrix4x4>[] _roomRawDepthCompletionMatrices =
    {
        new List<Matrix4x4>(256),
        new List<Matrix4x4>(256),
        new List<Matrix4x4>(256),
        new List<Matrix4x4>(256),
        new List<Matrix4x4>(256)
    };
    private readonly Matrix4x4[] _roomRawDepthCompletionDrawBatch = new Matrix4x4[1023];
    private readonly List<Matrix4x4> _roomRawDepthSnapshotMatrices = new List<Matrix4x4>(51200);
    private readonly List<Vector3> _roomRawDepthSnapshotVertices = new List<Vector3>(51200 * 4);
    private readonly List<Color> _roomRawDepthSnapshotColors = new List<Color>(51200 * 4);
    private readonly List<int> _roomRawDepthSnapshotIndices = new List<int>(51200 * 6);
    private readonly Dictionary<Vector3Int, RoomRawDepthCompletionVoxel> _roomRawDepthCompletionVoxels = new Dictionary<Vector3Int, RoomRawDepthCompletionVoxel>(32768);
    // Surface cells discovered by the production grid are the capture target map.
    // Keeping this separate from capture evidence lets the overlay represent cells
    // that are known to exist but have not yet received enough raw-depth support.
    private readonly Dictionary<Vector3Int, Vector3> _questCloneGuideVoxels = new Dictionary<Vector3Int, Vector3>(32768);
    private readonly List<Vector3> _questCloneGuideScratchPositions = new List<Vector3>(4096);
    private float _nextRoomRawDepthCompletionRefreshTime;
    private int _roomRawDepthCompletionGrayCount;
    private int _roomRawDepthCompletionBlueCount;
    private int _roomRawDepthCompletionGreenCount;
    private int _roomRawDepthCompletionYellowCount;
    private int _roomRawDepthCompletionRedCount;
    private string _roomRawDepthCompletionHint = "Waiting";
    private BinocularRoomRawDepthSnapshotStage _binocularRoomRawDepthSnapshotStage;
    private ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot _binocularRoomRawDepthRightSnapshot;
    private ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot _binocularRoomRawDepthLeftSnapshot;
    private ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot _binocularRoomRawDepthFusedSnapshot;
    private ScanCoverDepthPreprocessor.SourceEye _binocularRoomRawDepthOriginalEye = ScanCoverDepthPreprocessor.SourceEye.Right;
    private int _binocularRoomRawDepthStartSnapshotIndex = -1;
    private float _binocularRoomRawDepthStartTime;
    private string _binocularRoomRawDepthStatus = "idle";
    private int _roomRawDepthSnapshotVisiblePoints;
    private int _roomRawDepthSnapshotLastTotalPixels;
    private int _roomRawDepthSnapshotLastValidPixels;
    private string _roomRawDepthSnapshotLastFrame = "none";
    private string _lastExportedFrameName = "";
    private string _questCloneBinaryDirectory;
    private string _questCloneBinaryManifestPath;
    private int _questCloneWritesInFlight;
    private int _questCloneDroppedWrites;
    private double _questCloneLastInterEyeDeltaMs;
    private string _questCloneLastWriteError = string.Empty;
    private readonly object _questCloneWriteLock = new object();
    private string _lastGateStatus = "none";
    private ComputeBuffer _rawDepthProbeBuffer;
    private readonly Dictionary<Vector3Int, StabilityVoxel> _stabilityVoxels = new Dictionary<Vector3Int, StabilityVoxel>(32768);
    private readonly Dictionary<Vector3Int, RepeatCoverageVoxel> _repeatCoverageVoxels = new Dictionary<Vector3Int, RepeatCoverageVoxel>(32768);
    private readonly Dictionary<Vector3Int, RoomRawCoverageVoxel> _roomRawCoverageVoxels = new Dictionary<Vector3Int, RoomRawCoverageVoxel>(65536);
    private readonly Dictionary<Vector3Int, RoomRawCoverageVoxel> _roomRawCoverageHistoryVoxels = new Dictionary<Vector3Int, RoomRawCoverageVoxel>(131072);
    private readonly HashSet<Vector3Int> _roomRawCoverageWindowCoveredKeys = new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _roomRawCoverageWindowStableKeys = new HashSet<Vector3Int>();
    private HashSet<Vector3Int>[] _roomRawCoverageTileCoveredKeys = new HashSet<Vector3Int>[0];
    private HashSet<Vector3Int>[] _roomRawCoverageTileStableKeys = new HashSet<Vector3Int>[0];
    private int[] _roomRawCoverageTileSamples = new int[0];
    private string _roomRawCoverageMissingHint = "none";
    private readonly HashSet<Vector3Int> _roomRawCoveragePosePositionCells = new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _roomRawCoveragePoseOrientationCells = new HashSet<Vector3Int>();
    private readonly bool[][] _previousRawDepthValidMasks = { null, null };
    private readonly int[][] _rawDepthInvalidRunLengths = { null, null };

    private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
    private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
    private const int RawDepthProbeSize = 128;
    private static readonly float[] DistanceBinEdges = { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 5f, 8f, float.PositiveInfinity };
    private static readonly string[] DistanceBinLabels = { "0.0-0.5m", "0.5-1.0m", "1.0-1.5m", "1.5-2.0m", "2.0-3.0m", "3.0-5.0m", "5.0-8.0m", "8.0m+" };
    private static readonly float[] AngleBinEdges = { 0f, 20f, 40f, 60f, 75f, 90.001f };
    private static readonly string[] AngleBinLabels = { "0-20deg", "20-40deg", "40-60deg", "60-75deg", "75deg+" };
    private const int CoreMetricCount = 11;
    private const int DirectedMetricCount = 8;
    private const int NearEdgeSupplementMetricCount = 9;
    private const int DirectedRiskSupplementMetricCount = 3;

    private struct StabilityVoxel
    {
        public int frameHits;
        public int pointHits;
        public int firstFrame;
        public int lastFrame;
        public Vector3 positionSum;
        public Vector3 min;
        public Vector3 max;
    }

    private struct RepeatCoverageVoxel
    {
        public int frameHits;
        public int pointHits;
        public int firstFrame;
        public int lastFrame;
        public bool stable;
        public Vector3 positionSum;
        public Vector3 normalSum;
        public Vector3 min;
        public Vector3 max;
        public Vector3 firstCameraPosition;
        public Vector3 lastCameraPosition;
    }

    private struct RepeatCoverageCandidateVoxel
    {
        public int pointHits;
        public Vector3 positionSum;
        public Vector3 normalSum;
        public Vector3 min;
        public Vector3 max;
    }

    private struct RepeatCoverageGateResult
    {
        public bool hasData;
        public bool accepted;
        public int candidateVoxels;
        public int newVoxels;
        public int rehitVoxels;
        public int parallaxRehitVoxels;
        public int newStableVoxels;
        public string reason;
    }

    private enum RoomRawDepthCompletionState
    {
        Empty,
        Scanning,
        Supported,
        Stable,
        Risk,
        Conflict
    }

    private sealed class RoomRawDepthCompletionVoxel
    {
        public int hits;
        public int riskHits;
        public float sumDepth;
        public float sumDepthSq;
        public Vector3 positionSum;
        public float positionMagnitudeSqSum;
        private float recentRiskRatio;
        private bool hasRecentRisk;
        private bool hasFirstView;
        private Vector3 firstViewDirection;
        private float minViewDot = 1f;

        public void Add(Vector3 point, Vector3 cameraPosition, float depth, bool risk, float riskEmaAlpha)
        {
            hits++;
            if (risk)
                riskHits++;

            sumDepth += depth;
            sumDepthSq += depth * depth;
            positionSum += point;
            positionMagnitudeSqSum += point.sqrMagnitude;
            float riskValue = risk ? 1f : 0f;
            if (!hasRecentRisk)
            {
                recentRiskRatio = riskValue;
                hasRecentRisk = true;
            }
            else
            {
                recentRiskRatio = Mathf.Lerp(recentRiskRatio, riskValue, Mathf.Clamp01(riskEmaAlpha));
            }

            Vector3 viewDirection = point - cameraPosition;
            if (!IsFinite(viewDirection) || viewDirection.sqrMagnitude <= 1e-8f)
                return;

            viewDirection.Normalize();
            if (!hasFirstView)
            {
                firstViewDirection = viewDirection;
                hasFirstView = true;
                minViewDot = 1f;
                return;
            }

            minViewDot = Mathf.Min(minViewDot, Vector3.Dot(firstViewDirection, viewDirection));
        }

        public Vector3 AveragePosition => hits > 0 ? positionSum / hits : Vector3.zero;
        public float MeanDepth => hits > 0 ? sumDepth / hits : 0f;

        public float DepthStd
        {
            get
            {
                if (hits <= 1)
                    return 0f;

                float mean = sumDepth / hits;
                float variance = Mathf.Max(0f, sumDepthSq / hits - mean * mean);
                return Mathf.Sqrt(variance);
            }
        }

        public float RiskRatio => hits > 0 ? riskHits / (float)hits : 0f;
        public float RecentRiskRatio => hasRecentRisk ? recentRiskRatio : 0f;

        public float PositionStd
        {
            get
            {
                if (hits <= 1)
                    return 0f;

                Vector3 mean = positionSum / hits;
                float variance = Mathf.Max(0f, positionMagnitudeSqSum / hits - mean.sqrMagnitude);
                return Mathf.Sqrt(variance);
            }
        }

        public float AngleSpanDegrees
        {
            get
            {
                if (!hasFirstView)
                    return 0f;

                return Mathf.Acos(Mathf.Clamp(minViewDot, -1f, 1f)) * Mathf.Rad2Deg;
            }
        }
    }

    private struct RoomRawCoverageVoxel
    {
        public int frameHits;
        public int pointHits;
        public int firstFrame;
        public int lastFrame;
        public bool stable;
        public bool risk;
        public bool high;
        public bool highStable;
        public bool low;
        public bool lowStable;
        public Vector3 positionSum;
        public Vector3 normalSum;
        public Vector3 firstCameraPosition;
        public Vector3 lastCameraPosition;
    }

    private struct RoomRawCoverageCandidateVoxel
    {
        public int pointHits;
        public bool high;
        public bool low;
        public bool risk;
        public bool focus;
        public bool core;
        public bool edgeBuffer;
        public Vector3 positionSum;
        public Vector3 normalSum;
    }

    private struct RoomRawCoverageFrameResult
    {
        public bool hasData;
        public bool accepted;
        public int totalSamples;
        public int validSamples;
        public int frameVoxels;
        public int focusVoxels;
        public int coreVoxels;
        public int edgeBufferVoxels;
        public int focusNewVoxels;
        public int edgeBufferNewVoxels;
        public int newVoxels;
        public int rehitVoxels;
        public int historyRehitVoxels;
        public int historyAgreementVoxels;
        public int historyConflictVoxels;
        public int parallaxRehitVoxels;
        public int newStableVoxels;
        public int highFrameVoxels;
        public int newHighVoxels;
        public int newHighStableVoxels;
        public int lowFrameVoxels;
        public int newLowVoxels;
        public int newLowStableVoxels;
        public int riskSamples;
        public int focusRiskSamples;
        public int riskVoxels;
        public float riskRatio;
        public float overlapRatio;
        public float historyAgreementRatio;
        public float anchorAngle;
        public float anchorMove;
        public bool anchorFallback;
        public RawObservationOrderResult rawObservationOrder;
        public ObservationOrderScore observationOrderScore;
        public string status;
    }

    private sealed class QuestCloneWriteJob
    {
        public string rightPath;
        public string leftPath;
        public string metadataPath;
        public string manifestPath;
        public string metadataJson;
        public string manifestRow;
        public ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot right;
        public ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot left;
    }

    private struct RoomRawCoverageViewMeter
    {
        public int covered;
        public int stable;
        public int singleHit;
        public int needsRevisit;
        public int farCovered;
        public int farStable;
        public int completionCandidates;
        public int completionStable;
        public int completionRisk;

        public float StableRatio => covered > 0 ? stable / (float)covered : 0f;
        public float FarStableRatio => farCovered > 0 ? farStable / (float)farCovered : 0f;
        public float CompletionRiskRatio
        {
            get
            {
                int total = completionStable + completionCandidates + completionRisk;
                return total > 0 ? completionRisk / (float)total : 0f;
            }
        }
    }

    private struct RawObservationOrderResult
    {
        public bool hasData;
        public int validSamples;
        public int testedEdges;
        public int orderedEdges;
        public int badEdges;
        public int badDistanceEdges;
        public int badDepthEdges;
        public int badNormalEdges;
        public int testedQuads;
        public int badQuads;
        public int centerTestedEdges;
        public int centerBadEdges;
        public float orderRatio;
        public float centerOrderRatio;
        public float badEdgeRatio;
        public float badQuadRatio;
        public byte[] sampleStates;
        public string reason;
    }

    private struct ObservationOrderScore
    {
        public bool hasData;
        public int samples;
        public int support;
        public int useful;
        public int riskHits;
        public float riskRatio;
        public float supportRatio;
        public float usefulRatio;
        public float score01;
        public float rawOrderRatio;
        public float rawCenterOrderRatio;
        public float rawBadEdgeRatio;
        public float rawBadQuadRatio;
        public int rawTestedEdges;
        public int rawBadEdges;
        public int rawTestedQuads;
        public int rawBadQuads;
        public byte representationState;
        public bool disordered;
        public bool shouldFuse;
        public string reason;
    }

    private struct VertexObservation
    {
        public int index;
        public Vector3 world;
        public Vector3 normal;
        public float viewDepth;
        public float euclideanDistance;
        public float viewAngleDegrees;
        public int triangleCount;
        public float maxFaceNormalAngleDegrees;
        public bool boundaryRisk;
        public bool creaseRisk;
        public float cameraRayRight;
        public float cameraRayUp;
    }

    private struct TriangleInfo
    {
        public int a;
        public int b;
        public int c;
        public Vector3 normal;
        public float area;
        public float maxEdge;
    }

    private struct BinStats
    {
        public int count;
        public double viewDepthSum;
        public double distanceSum;
        public double angleSum;
        public int boundaryRiskCount;
        public int creaseRiskCount;
        public int riskCount;

        public void Add(VertexObservation observation)
        {
            count++;
            viewDepthSum += observation.viewDepth;
            distanceSum += observation.euclideanDistance;
            angleSum += observation.viewAngleDegrees;
            if (observation.boundaryRisk)
                boundaryRiskCount++;
            if (observation.creaseRisk)
                creaseRiskCount++;
            if (observation.boundaryRisk || observation.creaseRisk)
                riskCount++;
        }
    }

    private struct ObservationExportResult
    {
        public bool hasData;
        public int pointCount;
        public int farSurfaceCount;
        public int highAngleRiskCount;
        public int nearEdgeRiskCount;
        public int stableMainCount;
        public int topCoverageCount;
        public int middleCoverageCount;
        public int bottomCoverageCount;
        public int nearDistanceCount;
        public int midDistanceCount;
        public int farDistanceCount;
        public int frontAngleCount;
        public int obliqueAngleCount;
        public int extremeAngleCount;
        public int riskLayerCount;
        public int mainFrontCount;
        public int mainLeftObliqueCount;
        public int mainRightObliqueCount;
        public int mainUpObliqueCount;
        public int mainDownObliqueCount;
        public int boundaryLeftCount;
        public int boundaryRightCount;
        public int cornerJunctionCount;
        public int nearEdgeSupplementRiskCount;
        public int nearEdgeSupplementCreaseCount;
        public int nearEdgeSupplementBoundaryCount;
        public int nearEdgeSupplementLeftCount;
        public int nearEdgeSupplementRightCount;
        public int nearEdgeSupplementUpCount;
        public int nearEdgeSupplementDownCount;
        public int nearEdgeSupplementObliqueCount;
        public int nearEdgeSupplementExtremeCount;
        public bool farSurfaceQualified;
        public bool highAngleRiskQualified;
        public bool nearEdgeRiskQualified;
        public bool stableMainQualified;
        public bool topCoverageQualified;
        public bool middleCoverageQualified;
        public bool bottomCoverageQualified;
        public bool nearDistanceQualified;
        public bool midDistanceQualified;
        public bool farDistanceQualified;
        public bool frontAngleQualified;
        public bool obliqueAngleQualified;
        public bool extremeAngleQualified;
        public bool riskLayerQualified;
        public bool mainFrontQualified;
        public bool mainLeftObliqueQualified;
        public bool mainRightObliqueQualified;
        public bool mainUpObliqueQualified;
        public bool mainDownObliqueQualified;
        public bool boundaryLeftQualified;
        public bool boundaryRightQualified;
        public bool cornerJunctionQualified;
        public bool nearEdgeSupplementRiskQualified;
        public bool nearEdgeSupplementCreaseQualified;
        public bool nearEdgeSupplementBoundaryQualified;
        public bool nearEdgeSupplementLeftQualified;
        public bool nearEdgeSupplementRightQualified;
        public bool nearEdgeSupplementUpQualified;
        public bool nearEdgeSupplementDownQualified;
        public bool nearEdgeSupplementObliqueQualified;
        public bool nearEdgeSupplementExtremeQualified;
    }

    private struct RawDepthBadnessStats
    {
        public int eye;
        public int totalPixels;
        public int validPixels;
        public int invalidPixels;
        public int invalidComponentCount;
        public int largeHoleComponentCount;
        public int largestHolePixels;
        public int persistentInvalidPixels;
        public int newlyInvalidPixels;
        public int recoveredPixels;
        public int edgeRiskValidPixels;
        public float minLinearMeters;
        public float maxLinearMeters;
        public float avgLinearMeters;
    }

    private struct RawDepthHoleComponent
    {
        public int id;
        public int pixelCount;
        public int minX;
        public int minY;
        public int maxX;
        public int maxY;
        public bool touchesBorder;
    }

    private void Awake()
    {
        ApplyQuestCloneRuntimeSafetyDefaults();
        ResolveRefs();
    }

    private void OnEnable()
    {
        ApplyQuestCloneRuntimeSafetyDefaults();
        ResolveRefs();
        if (questCloneContinuousCaptureMode)
            Debug.Log($"[ScanCoverQuestCapture] ready hud={showHud} productionGridVisible=true target={questCloneTargetLabel} action={questCloneActionLabel}", this);
        if (startSessionOnEnable)
            StartSession();
    }

    private void ApplyQuestCloneRuntimeSafetyDefaults()
    {
        if (!questCloneContinuousCaptureMode)
            return;

        // The collection chain is an overlay/recorder, not a replacement for the
        // production mesh. Keep a visible boot state even if an older serialized
        // scene still contains the previous capture-HUD defaults.
        showHud = true;
        enableOvrInput = true;
        enableOvrStartStopInput = true;
        captureWhileSessionActive = true;
        startSessionOnEnable = true;
        useBinocularRoomRawDepthSnapshots = true;
        hideDepthGridPreviewWhileRecording = false;
        roomRawDepthCompletionUseGpuInstancing = true;
        roomRawDepthCompletionUseVoxelCenterPresentation = true;
        roomRawDepthCompletionPointSize = Mathf.Max(0.055f, roomRawDepthCompletionPointSize);
        questCloneProgressStablePositionStdMeters = Mathf.Max(0.04f, questCloneProgressStablePositionStdMeters);
        questCloneProgressHardPositionStdMeters = Mathf.Max(0.07f, questCloneProgressHardPositionStdMeters);
        roomRawDepthCompletionMaxVisualPoints = Mathf.Clamp(roomRawDepthCompletionMaxVisualPoints, 64, 2400);
        roomRawDepthCompletionRefreshSeconds = Mathf.Max(0.35f, roomRawDepthCompletionRefreshSeconds);
    }

    private void Update()
    {
        ResolveRefs();
        HandleInput();
        CompletePendingCaptureIfReady();
        UpdateSnapshotGridCapturePreview();
        UpdateLegacyRoomRawCoverageCompletion();
        UpdateRoomRawCoverageViewFrame();
        UpdateRoomRawCoveragePreviewPoints();
        UpdateRoomRawDepthCompletionOverlay();
        DrawRoomRawDepthCompletionOverlay();
        DrawRoomRawDepthSnapshotOverlay();
        UpdateHud();

        if (!_isRecording || !captureWhileSessionActive || _pendingCapture || _capturedFrameCount >= maxFramesPerSession)
            return;

        if (ShouldCaptureKeyframe())
            RequestCapture("keyframe");
    }

    private void OnDisable()
    {
        SaveActiveSessionForAppExit();
        ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: true);
        _independentSnapshotCaptureActive = false;
        ApplyRecordingDepthGridDisplayMode(false);
        ClearSnapshotGridCaptureMask(true);
        SetHudVisible(false);
        SetRoomRawCoverageViewFrameVisible(false);
        SetRoomRawCoveragePreviewVisible(false);
        SetRoomRawDepthCompletionOverlayVisible(false);
    }

    private void OnDestroy()
    {
        ReleaseRawDepthResources();
        DestroyRoomRawCoverageViewFrame();
        DestroyRoomRawCoveragePreview();
        DestroyRoomRawDepthCompletionOverlay();
        DestroyRoomRawDepthSnapshotOverlay();
        if (_hudRoot == null)
            return;
        if (Application.isPlaying)
            Destroy(_hudRoot);
        else
            DestroyImmediate(_hudRoot);
    }

    private void OnApplicationQuit()
    {
        SaveActiveSessionForAppExit();
        ReleaseRawDepthResources();
    }

    private void OnApplicationPause(bool paused)
    {
        if (paused)
            SaveActiveSessionForAppExit();
    }

    private void SaveActiveSessionForAppExit()
    {
        if (!_isRecording)
            return;

        StopSession();
    }

    [ContextMenu("Start Multi Frame Session")]
    public void StartSession()
    {
        ResolveRefs();
        if (depthGridPointCloud == null)
        {
            Debug.LogWarning("[ScanCoverMultiFrameSessionExporter] Cannot start: depth grid source is missing.", this);
            return;
        }

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        string exportRoot = ResolveSessionExportRoot();
        _sessionFileExportEnabled = ShouldEnableSessionFileExport();
        if (_sessionFileExportEnabled)
        {
            _sessionDirectory = Path.Combine(exportRoot, sessionGroupDirectoryName, $"{sessionNamePrefix}_{timestamp}");
            _framesDirectory = Path.Combine(_sessionDirectory, "frames");
            _observationDirectory = Path.Combine(_sessionDirectory, "observation_stats");
            _quest3BadnessDirectory = Path.Combine(_sessionDirectory, "quest3_observation_badness");
            _repeatCoverageDirectory = Path.Combine(_sessionDirectory, "repeat_coverage");
            _roomRawCoverageDirectory = Path.Combine(_sessionDirectory, "room_raw_coverage");
            _roomRawDepthDirectory = Path.Combine(_sessionDirectory, "room_raw_depth_frames");
            _roomRawDepthSnapshotDirectory = Path.Combine(_sessionDirectory, "room_raw_depth_snapshots");
            _virtualCloneInputDirectory = Path.Combine(_sessionDirectory, SafeDirectoryName(virtualCloneInputDirectoryName, "virtual_clone_input"));
            _questCloneBinaryDirectory = Path.Combine(_sessionDirectory, "quest_clone_capture");
            Directory.CreateDirectory(_framesDirectory);
            Directory.CreateDirectory(_observationDirectory);
            if (exportQuest3ObservationBadnessStats)
                Directory.CreateDirectory(_quest3BadnessDirectory);
            if (gateFramesByRepeatCoverage)
                Directory.CreateDirectory(_repeatCoverageDirectory);
            if (trackRoomRawCoverage)
                Directory.CreateDirectory(_roomRawCoverageDirectory);
            if (ShouldExportRoomRawDepthFrames())
                Directory.CreateDirectory(_roomRawDepthDirectory);
            if (ShouldExportRoomRawDepthSnapshots())
                Directory.CreateDirectory(_roomRawDepthSnapshotDirectory);
            if (ShouldExportVirtualCloneInputMetadata())
                Directory.CreateDirectory(_virtualCloneInputDirectory);
            if (questCloneContinuousCaptureMode && questCloneUseCompactBinary)
                Directory.CreateDirectory(_questCloneBinaryDirectory);
            _manifestPath = Path.Combine(_sessionDirectory, "session_manifest.csv");
            _frameObservationStatsPath = Path.Combine(_observationDirectory, "frame_observation_stats.csv");
            _distanceBinsPath = Path.Combine(_observationDirectory, "distance_bins.csv");
            _angleBinsPath = Path.Combine(_observationDirectory, "angle_bins.csv");
            _edgeRiskStatsPath = Path.Combine(_observationDirectory, "edge_risk_stats.csv");
            _targetGatePath = Path.Combine(_observationDirectory, "target_gate.csv");
            _repeatCoverageGatePath = Path.Combine(_repeatCoverageDirectory, "repeat_coverage_gate.csv");
            _repeatCoverageVoxelsPath = Path.Combine(_repeatCoverageDirectory, "repeat_coverage_voxels.csv");
            _roomRawCoverageFramesPath = Path.Combine(_roomRawCoverageDirectory, "room_raw_coverage_frames.csv");
            _roomRawCoverageVoxelsPath = Path.Combine(_roomRawCoverageDirectory, "room_raw_coverage_voxels.csv");
            _roomRawCoverageSummaryPath = Path.Combine(_roomRawCoverageDirectory, "room_raw_coverage_summary.json");
            _roomRawDepthManifestPath = Path.Combine(_roomRawDepthDirectory, "room_raw_depth_manifest.csv");
            _roomRawDepthSummaryPath = Path.Combine(_roomRawDepthDirectory, "room_raw_depth_summary.json");
            _roomRawDepthSnapshotManifestPath = Path.Combine(_roomRawDepthSnapshotDirectory, "room_raw_depth_snapshot_manifest.csv");
            _virtualCloneInputManifestPath = Path.Combine(_virtualCloneInputDirectory, "virtual_clone_input_manifest.csv");
            _questCloneBinaryManifestPath = Path.Combine(_questCloneBinaryDirectory, "quest_clone_capture_manifest.csv");
            _rawDepthBadnessFramesPath = Path.Combine(_quest3BadnessDirectory, "raw_depth_badness_frames.csv");
            _rawDepthHoleComponentsPath = Path.Combine(_quest3BadnessDirectory, "raw_depth_hole_components.csv");
            _rawDepthBadnessSummaryPath = Path.Combine(_quest3BadnessDirectory, "raw_depth_badness_summary.json");
            WriteManifestHeader();
            WriteObservationStatsHeaders();
            WriteQuest3BadnessHeaders();
            WriteRepeatCoverageHeaders();
            WriteRoomRawCoverageHeaders();
            WriteRoomRawDepthHeaders();
            WriteRoomRawDepthSnapshotHeaders();
            WriteVirtualCloneInputHeaders();
            WriteQuestCloneBinaryHeader();
            WriteSessionInfo(timestamp);
        }
        else
        {
            ClearSessionExportPaths();
        }

        _capturedFrameCount = 0;
        _roomRawDepthExportedFrames = 0;
        _roomRawDepthExportedSamples = 0;
        _roomRawDepthSkippedFrames = 0;
        _roomRawDepthSnapshotExportedFrames = 0;
        _roomRawDepthSnapshotTotalPixels = 0;
        _roomRawDepthSnapshotValidPixels = 0;
        Interlocked.Exchange(ref _questCloneDroppedWrites, 0);
        _questCloneLastWriteError = string.Empty;
        _questCloneLastInterEyeDeltaMs = 0d;
        ResetRoomRawDepthCompletionOverlay();
        ClearRoomRawDepthSnapshotOverlay();
        PrepareSnapshotGridCaptureMaskForSession(!_snapshotGridMaskActive);
        _targetFarSurfaceFrames = 0;
        _targetHighAngleRiskFrames = 0;
        _targetNearEdgeRiskFrames = 0;
        _targetStableMainFrames = 0;
        _targetTopCoverageFrames = 0;
        _targetMiddleCoverageFrames = 0;
        _targetBottomCoverageFrames = 0;
        _targetNearDistanceFrames = 0;
        _targetMidDistanceFrames = 0;
        _targetFarDistanceFrames = 0;
        _targetFrontAngleFrames = 0;
        _targetObliqueAngleFrames = 0;
        _targetExtremeAngleFrames = 0;
        _targetRiskLayerFrames = 0;
        _targetMainFrontFrames = 0;
        _targetMainLeftObliqueFrames = 0;
        _targetMainRightObliqueFrames = 0;
        _targetMainUpObliqueFrames = 0;
        _targetMainDownObliqueFrames = 0;
        _targetBoundaryLeftFrames = 0;
        _targetBoundaryRightFrames = 0;
        _targetCornerJunctionFrames = 0;
        _targetNearEdgeSupplementRiskFrames = 0;
        _targetNearEdgeSupplementCreaseFrames = 0;
        _targetNearEdgeSupplementBoundaryFrames = 0;
        _targetNearEdgeSupplementLeftFrames = 0;
        _targetNearEdgeSupplementRightFrames = 0;
        _targetNearEdgeSupplementUpFrames = 0;
        _targetNearEdgeSupplementDownFrames = 0;
        _targetNearEdgeSupplementObliqueFrames = 0;
        _targetNearEdgeSupplementExtremeFrames = 0;
        _targetRejectedFrames = 0;
        _repeatCoverageAcceptedFrames = 0;
        _repeatCoverageRejectedFrames = 0;
        _repeatCoverageStableVoxelCount = 0;
        _lastRepeatCandidateVoxels = 0;
        _lastRepeatNewVoxels = 0;
        _lastRepeatRehitVoxels = 0;
        _lastRepeatParallaxRehitVoxels = 0;
        _lastRepeatNewStableVoxels = 0;
        _lastRepeatCoverageStatus = "none";
        _roomRawCoverageFrames = 0;
        _roomRawCoverageValidSamples = 0;
        _roomRawCoverageTotalSamples = 0;
        _roomRawCoverageCoveredVoxels = 0;
        _roomRawCoverageStableVoxels = 0;
        _roomRawCoverageRiskVoxels = 0;
        _roomRawCoverageHighVoxels = 0;
        _roomRawCoverageHighStableVoxels = 0;
        _roomRawCoverageLowVoxels = 0;
        _roomRawCoverageLowStableVoxels = 0;
        _roomRawCoveragePoseCellCount = 0;
        _roomRawCoverageRejectedFrames = 0;
        _lastRoomRawValidSamples = 0;
        _lastRoomRawFocusVoxels = 0;
        _lastRoomRawFocusNewVoxels = 0;
        _lastRoomRawNewVoxels = 0;
        _lastRoomRawRehitVoxels = 0;
        _lastRoomRawHistoryRehitVoxels = 0;
        _lastRoomRawHistoryAgreementVoxels = 0;
        _lastRoomRawHistoryConflictVoxels = 0;
        _lastRoomRawParallaxRehitVoxels = 0;
        _lastRoomRawStableNewVoxels = 0;
        _lastRoomRawHighFrameVoxels = 0;
        _lastRoomRawNewHighVoxels = 0;
        _lastRoomRawNewHighStableVoxels = 0;
        _lastRoomRawLowFrameVoxels = 0;
        _lastRoomRawNewLowVoxels = 0;
        _lastRoomRawNewLowStableVoxels = 0;
        _lastRoomRawRiskRatio = 0f;
        _lastRoomRawOverlapRatio = 0f;
        _lastRoomRawHistoryAgreementRatio = 1f;
        _lastRoomRawStatus = "none";
        _roomRawCoverageWindowActive = false;
        _roomRawCoverageWindowStartFrames = 0;
        _roomRawCoverageWindowStartValidSamples = 0;
        _roomRawCoverageWindowStartCoveredVoxels = 0;
        _roomRawCoverageWindowStartStableVoxels = 0;
        _lastRoomRawWindowFrames = 0;
        _lastRoomRawWindowValidSamples = 0;
        _lastRoomRawWindowCoveredVoxels = 0;
        _lastRoomRawWindowStableVoxels = 0;
        _loadedRoomRawCoverageSessions = 0;
        _loadedRoomRawCoverageVoxels = 0;
        _loadedRoomRawCoverageStableVoxels = 0;
        _loadedRoomRawCoverageSource = "none";
        _roomRawCoverageAnchorSet = false;
        _roomRawCoverageViewFrameLocked = false;
        _lastRoomRawAnchorAngle = 0f;
        _lastRoomRawAnchorMove = 0f;
        _rawDepthBadnessFrameCount = 0;
        _rawDepthBadnessEyeSamples = 0;
        _rawDepthBadnessValidSamples = 0;
        _rawDepthBadnessInvalidSamples = 0;
        _rawDepthBadnessLargeHoleComponents = 0;
        _rawDepthBadnessPersistentInvalidPixels = 0;
        _rawDepthBadnessNewlyInvalidPixels = 0;
        _rawDepthBadnessRecoveredPixels = 0;
        _rawDepthBadnessEdgeRiskPixels = 0;
        _rawDepthBadnessInvalidFrames = 0;
        _rawDepthBadnessLargeHoleFrames = 0;
        _rawDepthBadnessEdgeJumpFrames = 0;
        _rawDepthBadnessPersistentInvalidFrames = 0;
        _previousRawDepthValidMasks[0] = null;
        _previousRawDepthValidMasks[1] = null;
        _rawDepthInvalidRunLengths[0] = null;
        _rawDepthInvalidRunLengths[1] = null;
        _lastGateStatus = "none";
        _lastCaptureTime = float.NegativeInfinity;
        _hasLastCapturePose = false;
        _pendingCapture = false;
        _pendingRefreshRequested = false;
        ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: true);
        _independentSnapshotCaptureActive = false;
        _binocularRoomRawDepthStatus = "idle";
        _stabilityVoxels.Clear();
        _repeatCoverageVoxels.Clear();
        _roomRawCoverageVoxels.Clear();
        _roomRawCoverageHistoryVoxels.Clear();
        _roomRawCoverageWindowCoveredKeys.Clear();
        _roomRawCoverageWindowStableKeys.Clear();
        _roomRawCoveragePosePositionCells.Clear();
        _roomRawCoveragePoseOrientationCells.Clear();
        if (_sessionFileExportEnabled)
            LoadPreviousRoomRawCoverageIfNeeded(exportRoot);
        InitializeRoomRawCoverageAnchor();
        ResetLegacyRoomRawCoverageProgressWatch();
        if (ShouldUseRoomRawCoverageHudOnlyCapture())
            _lastRoomRawStatus = "local frame started";
        else if (ShouldUseLegacyFullViewRoomRawCoverage())
            _lastRoomRawStatus = "full-view scan started";
        ClearRoomRawCoveragePreviewMeshes();
        ApplyRecordingDepthGridDisplayMode(true);
        _isRecording = true;

        Debug.Log($"[ScanCoverMultiFrameSessionExporter] Multi-frame session started => {_sessionDirectory}", this);
        if (captureFirstFrameOnStart)
            RequestCapture("session-start");
    }

    [ContextMenu("Stop Multi Frame Session")]
    public void StopSession()
    {
        if (!_isRecording)
            return;

        _isRecording = false;
        _pendingCapture = false;
        _pendingRefreshRequested = false;
        _stopAfterPendingManualCapture = false;
        ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: true);
        _independentSnapshotCaptureActive = false;
        _roomRawCoverageWindowActive = false;
        _roomRawCoverageViewFrameLocked = false;
        _roomRawCoverageAnchorSet = false;
        ApplyRecordingDepthGridDisplayMode(false);
        HideSnapshotGridCaptureMask();
        ClearSnapshotGridCaptureMask(false);
        WriteStabilityOutputs();
        WriteRepeatCoverageOutputs();
        WriteRoomRawCoverageOutputs();
        WriteRoomRawDepthSummary();
        WriteQuest3BadnessSummary();
        if (_snapshotGridMaskActive)
            _snapshotGridMaskStatus = "\u91c7\u96c6\u5b8c\u6210";
        Debug.Log($"[ScanCoverMultiFrameSessionExporter] Multi-frame session stopped. frames={_capturedFrameCount} dir={_sessionDirectory}", this);
    }

    [ContextMenu("Capture Frame Now")]
    public void CaptureFrameNow()
    {
        if (ShouldUseIndependentSnapshotLine())
        {
            CaptureIndependentSnapshotNow();
            return;
        }

        bool startOneShotSession = !_isRecording;
        if (startOneShotSession)
            _stopAfterPendingManualCapture = true;
        if (!_isRecording)
            StartSession();
        RequestCapture("manual");
        StopOneShotManualSessionIfReady();
    }

    private bool ShouldUseIndependentSnapshotLine()
    {
        return captureNowUsesIndependentSnapshotLine &&
            exportRoomRawDepthSnapshots &&
            useBinocularRoomRawDepthSnapshots;
    }

    private void CaptureIndependentSnapshotNow()
    {
        ResolveRefs();
        if (depthGridPointCloud == null)
        {
            Debug.LogWarning("[ScanCoverMultiFrameSessionExporter] Cannot capture snapshot: depth grid source is missing.", this);
            return;
        }

        if (_pendingCapture)
            return;

        BeginIndependentSnapshotExportFolder();
        RequestCapture("manual");
    }

    private void BeginIndependentSnapshotExportFolder()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        string exportRoot = ResolveSessionExportRoot();
        _sessionFileExportEnabled = ShouldEnableSessionFileExport();
        _independentSnapshotCaptureActive = true;
        _capturedFrameCount = 0;
        _roomRawDepthSnapshotExportedFrames = 0;
        _roomRawDepthSnapshotTotalPixels = 0;
        _roomRawDepthSnapshotValidPixels = 0;
        ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: true);
        ClearRoomRawDepthSnapshotOverlay();

        if (!_sessionFileExportEnabled)
        {
            ClearSessionExportPaths();
            _binocularRoomRawDepthStatus = "independent snapshot: file export disabled";
            return;
        }

        string groupName = string.IsNullOrWhiteSpace(independentSnapshotDirectoryName)
            ? "SnapshotCaptures"
            : independentSnapshotDirectoryName.Trim();
        string prefix = string.IsNullOrWhiteSpace(independentSnapshotNamePrefix)
            ? "ScanCover_BinocularSnapshot"
            : independentSnapshotNamePrefix.Trim();

        _sessionDirectory = Path.Combine(exportRoot, groupName, $"{prefix}_{timestamp}");
        _roomRawDepthSnapshotDirectory = Path.Combine(_sessionDirectory, "room_raw_depth_snapshots");
        _virtualCloneInputDirectory = Path.Combine(_sessionDirectory, SafeDirectoryName(virtualCloneInputDirectoryName, "virtual_clone_input"));
        Directory.CreateDirectory(_roomRawDepthSnapshotDirectory);
        if (ShouldExportVirtualCloneInputMetadata())
            Directory.CreateDirectory(_virtualCloneInputDirectory);
        _roomRawDepthSnapshotManifestPath = Path.Combine(_roomRawDepthSnapshotDirectory, "room_raw_depth_snapshot_manifest.csv");
        _virtualCloneInputManifestPath = Path.Combine(_virtualCloneInputDirectory, "virtual_clone_input_manifest.csv");
        WriteRoomRawDepthSnapshotHeaders();
        WriteVirtualCloneInputHeaders();
        _binocularRoomRawDepthStatus = "独立快照已准备";
    }

    private void ToggleRoomRawCoverageWindow()
    {
        if (!_isRecording)
        {
            StartSession();
            return;
        }

        if (_roomRawCoverageWindowActive || _roomRawCoverageViewFrameLocked)
        {
            CompleteRoomRawCoverageWindow("local frame paused; aim next area and press B");
            return;
        }

        BeginRoomRawCoverageWindow("local frame started");
    }

    private void BeginRoomRawCoverageWindow(string status)
    {
        InitializeRoomRawCoverageAnchor();
        _lastCaptureTime = float.NegativeInfinity;
        _hasLastCapturePose = false;
        ClearRoomRawCoveragePreviewMeshes();
        _lastRoomRawStatus = status;
        if (captureFirstFrameOnStart)
            RequestCapture("window-start");
    }

    private void CompleteRoomRawCoverageWindow(string status = "local frame complete; aim next area and press B")
    {
        if (!_roomRawCoverageWindowActive && !_roomRawCoverageViewFrameLocked)
            return;

        _pendingCapture = false;
        _pendingRefreshRequested = false;
        _roomRawCoverageWindowActive = false;
        _roomRawCoverageViewFrameLocked = false;
        _roomRawCoverageAnchorSet = false;
        _lastRoomRawWindowFrames = GetRoomRawCoverageWindowFrames();
        _lastRoomRawWindowValidSamples = GetRoomRawCoverageWindowValidSamples();
        _lastRoomRawWindowCoveredVoxels = GetRoomRawCoverageWindowCoveredVoxels();
        _lastRoomRawWindowStableVoxels = GetRoomRawCoverageWindowStableVoxels();
        _lastRoomRawStatus = status;
        ClearRoomRawCoveragePreviewMeshes();
        HideSnapshotGridCaptureMask();
        _snapshotGridMaskStatus = "閲囬泦瀹屾垚";
    }

    private void HandleInput()
    {
        if (enableKeyboardInput && Keyboard.current != null)
        {
            if (Keyboard.current[keyboardStartStopKey].wasPressedThisFrame)
            {
                if (_isRecording) StopSession();
                else StartSession();
            }

            if (Keyboard.current[keyboardCaptureNowKey].wasPressedThisFrame)
                CaptureFrameNow();
        }

        if (enableOvrInput)
        {
            if (enableOvrStartStopInput && OVRInput.GetDown(ovrStartStopButton))
            {
                if (ShouldUseRoomRawCoverageHudOnlyCapture())
                    ToggleRoomRawCoverageWindow();
                else if (_isRecording)
                    StopSession();
                else
                    StartSession();
            }

            if (enableOvrCaptureNowInput && WasOvrCaptureNowPressed())
                CaptureFrameNow();
        }
    }

    private bool WasOvrCaptureNowPressed()
    {
        if (useRightIndexTriggerForCaptureNow)
            return OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger);

        return OVRInput.GetDown(ovrCaptureNowButton);
    }

    private bool ShouldCaptureKeyframe()
    {
        Transform pose = ResolvePoseTransform();
        if (pose == null)
            return !_hasLastCapturePose && Time.unscaledTime - _lastCaptureTime >= minCaptureIntervalSeconds;

        if (!_hasLastCapturePose)
            return true;

        if (Time.unscaledTime - _lastCaptureTime < minCaptureIntervalSeconds)
            return false;

        // Degradation-model collection needs stationary samples as well as motion.
        // Target/action labels are guidance only; they must never reject raw data.
        if (questCloneContinuousCaptureMode && questCloneCaptureStaticFrames)
            return true;

        if (ShouldUseRoomRawCoverageHudOnlyCapture() && lockRoomRawCoverageToStartView)
        {
            if (!_roomRawCoverageWindowActive || !_roomRawCoverageViewFrameLocked)
            {
                _lastRoomRawStatus = "paused: aim next area and press B";
                return false;
            }

            if (!IsWithinRoomRawCoverageAnchor(pose, out _lastRoomRawAnchorAngle, out _lastRoomRawAnchorMove))
            {
                _lastRoomRawStatus = $"paused: gaze outside frame angle={FormatFloat(_lastRoomRawAnchorAngle)}deg";
                return false;
            }

            return true;
        }

        float moved = Vector3.Distance(pose.position, _lastCapturePosition);
        float rotated = Quaternion.Angle(pose.rotation, _lastCaptureRotation);
        return moved >= minMoveMeters || rotated >= minRotateDegrees;
    }

    private void RequestCapture(string reason)
    {
        if (depthGridPointCloud == null || _pendingCapture || _capturedFrameCount >= maxFramesPerSession)
            return;

        _pendingCapture = true;
        _pendingReason = reason;
        _pendingStartDepthFrame = depthGridPointCloud.CurrentSurfaceMeshFrameIndex;
        _pendingStartTime = Time.unscaledTime;
        _pendingRefreshRequested = false;

        if (ShouldUseBinocularRoomRawDepthSnapshotCapture())
        {
            BeginBinocularRoomRawDepthSnapshotCapture();
            return;
        }

        if (_snapshotGridMaskActive)
        {
            TryExportPendingCapture(force: true);
            return;
        }

        if (!requestFreshDepthGridBeforeCapture)
        {
            TryExportPendingCapture(force: true);
            return;
        }

        _pendingRefreshRequested = depthGridPointCloud.RefreshNow(forcePreprocessorRefresh: true);
    }

    private void CompletePendingCaptureIfReady()
    {
        if (!_pendingCapture)
            return;

        if (_binocularRoomRawDepthSnapshotStage != BinocularRoomRawDepthSnapshotStage.None)
        {
            UpdateBinocularRoomRawDepthSnapshotCapture();
            StopOneShotManualSessionIfReady();
            return;
        }

        bool frameAdvanced = depthGridPointCloud != null && depthGridPointCloud.CurrentSurfaceMeshFrameIndex != _pendingStartDepthFrame;
        bool timedOut = Time.unscaledTime - _pendingStartTime >= refreshWaitTimeoutSeconds;
        if (frameAdvanced || timedOut)
        {
            TryExportPendingCapture(force: true);
            StopOneShotManualSessionIfReady();
        }
    }

    private void StopOneShotManualSessionIfReady()
    {
        if (!_stopAfterPendingManualCapture || _pendingCapture)
            return;

        _stopAfterPendingManualCapture = false;
        if (_isRecording)
            StopSession();
    }

    private void TryExportPendingCapture(bool force)
    {
        if (!_pendingCapture || depthGridPointCloud == null)
            return;

        string frameName = $"frame_{_capturedFrameCount:0000}";
        Transform pose = ResolvePoseTransform();
        if (_binocularRoomRawDepthSnapshotStage == BinocularRoomRawDepthSnapshotStage.Ready)
        {
            if (questCloneContinuousCaptureMode && questCloneUseCompactBinary)
            {
                ExportQuestCloneCapture(frameName, pose);
                _lastExportedFrameName = frameName;
                _capturedFrameCount++;
                MarkCaptureAttemptPose(pose);
                _pendingCapture = false;
                _pendingRefreshRequested = false;
                ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: true);
                if (_capturedFrameCount >= maxFramesPerSession)
                    StopSession();
                return;
            }

            if (_binocularRoomRawDepthFusedSnapshot != null)
                MaybeExportRoomRawDepthSnapshot(frameName, _binocularRoomRawDepthFusedSnapshot, pose);

            bool wasIndependentSnapshot = _independentSnapshotCaptureActive;
            _lastExportedFrameName = frameName;
            _capturedFrameCount++;
            MarkCaptureAttemptPose(pose);
            _pendingCapture = false;
            _pendingRefreshRequested = false;
            ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: true);
            if (wasIndependentSnapshot)
            {
                _independentSnapshotCaptureActive = false;
                _binocularRoomRawDepthStatus = "独立双眼快照完成";
                return;
            }

            if (_capturedFrameCount >= maxFramesPerSession)
                StopSession();
            return;
        }

        if (ShouldUseRoomRawCoverageHudOnlyCapture())
        {
            if (!UpdateRoomRawCoverage(frameName, pose, _capturedFrameCount))
            {
                MarkCaptureAttemptPose(pose);
                _pendingCapture = false;
                _pendingRefreshRequested = false;
                return;
            }

            _lastExportedFrameName = frameName;
            _capturedFrameCount++;
            MarkCaptureAttemptPose(pose);
            _pendingCapture = false;
            _pendingRefreshRequested = false;

            if (AreRoomRawCoverageTargetsComplete())
                CompleteRoomRawCoverageWindow();
            else if (_capturedFrameCount >= maxFramesPerSession)
                StopSession();
            return;
        }

        string frameDirectory = _sessionFileExportEnabled ? Path.Combine(_framesDirectory, frameName) : "";
        string rawDepthCsvPath = _sessionFileExportEnabled && exportRawDepthProbe ? Path.Combine(frameDirectory, frameName + "_raw_depth_probe.csv") : "";
        string rawDepthJsonPath = _sessionFileExportEnabled && exportRawDepthProbe ? Path.Combine(frameDirectory, frameName + "_raw_depth_probe.json") : "";
        string gridStateCsvPath = _sessionFileExportEnabled && exportGridStateCsv ? Path.Combine(frameDirectory, frameName + "_grid_state.csv") : "";
        bool rawCoverageUpdatedBeforeMeshExport = false;
        bool rawCoverageAcceptedBeforeMeshExport = false;
        if (ShouldUseLegacyFullViewRoomRawCoverage())
        {
            rawCoverageAcceptedBeforeMeshExport = UpdateRoomRawCoverage(frameName, pose, _capturedFrameCount);
            rawCoverageUpdatedBeforeMeshExport = !string.Equals(_lastRoomRawStatus, "raw snapshot unavailable", StringComparison.Ordinal);
        }
        else
        {
            TryExportLatestRoomRawDepthSnapshot(frameName, pose);
        }

        if (!_sessionFileExportEnabled)
        {
            if (rawCoverageAcceptedBeforeMeshExport)
            {
                _lastExportedFrameName = frameName;
                _capturedFrameCount++;
            }
            MarkCaptureAttemptPose(pose);
            _pendingCapture = false;
            _pendingRefreshRequested = false;

            if (ShouldUseLegacyFullViewRoomRawCoverage() && AreRoomRawCoverageTargetsComplete())
                StopSession();
            else if (_capturedFrameCount >= maxFramesPerSession)
                StopSession();
            return;
        }

        if (!depthGridPointCloud.ExportCurrentSurfaceMeshFramePackage(
                frameDirectory,
                frameName,
                captureCamera,
                pose,
                out string objPath,
                out string verticesCsvPath,
                out string trianglesCsvPath,
                out string cameraJsonPath))
        {
            if (force)
                Debug.LogWarning($"[ScanCoverMultiFrameSessionExporter] Frame export failed: {depthGridPointCloud.LastIssue}", this);

            if (rawCoverageAcceptedBeforeMeshExport)
            {
                AppendManifestRow(frameName, _pendingReason, "", "", "", "", rawDepthCsvPath, rawDepthJsonPath, gridStateCsvPath, pose);
                _lastExportedFrameName = frameName;
                _capturedFrameCount++;
                MarkCaptureAttemptPose(pose);

                if (AreRoomRawCoverageTargetsComplete())
                    StopSession();
                else if (_capturedFrameCount >= maxFramesPerSession)
                    StopSession();
            }

            _pendingCapture = false;
            _pendingRefreshRequested = false;
            return;
        }

        ObservationExportResult observationResult = default;
        bool hasObservationResult = false;
        if (exportObservationStats)
        {
            observationResult = ExportObservationStats(frameDirectory, frameName, verticesCsvPath, trianglesCsvPath, pose, writeOutputs: !gateFramesByObservationTargets && !gateFramesByRepeatCoverage);
            hasObservationResult = observationResult.hasData;
        }

        if (gateFramesByObservationTargets && exportObservationStats)
        {
            int legacyTarget = Mathf.Max(1, targetFramesPerObservationBucket);
            int coreTarget = Mathf.Max(1, targetFramesPerCoreMetric);
            int supplementTarget = Mathf.Max(1, targetNearEdgeSupplementFrames);
            bool supplementMode = observationCaptureMode == ObservationCaptureMode.NearEdgeSupplement;
            bool boundaryRiskSupplementMode = observationCaptureMode == ObservationCaptureMode.BoundaryRiskSupplement;
            bool farWouldCount = observationResult.farSurfaceQualified && _targetFarSurfaceFrames < legacyTarget;
            bool highAngleWouldCount = observationResult.highAngleRiskQualified && _targetHighAngleRiskFrames < legacyTarget;
            bool nearEdgeWouldCount = observationResult.nearEdgeRiskQualified && _targetNearEdgeRiskFrames < legacyTarget;

            bool stableMainWouldCount = observationResult.stableMainQualified && _targetStableMainFrames < coreTarget;
            bool topCoverageWouldCount = observationResult.topCoverageQualified && _targetTopCoverageFrames < coreTarget;
            bool middleCoverageWouldCount = observationResult.middleCoverageQualified && _targetMiddleCoverageFrames < coreTarget;
            bool bottomCoverageWouldCount = observationResult.bottomCoverageQualified && _targetBottomCoverageFrames < coreTarget;
            bool nearDistanceWouldCount = observationResult.nearDistanceQualified && _targetNearDistanceFrames < coreTarget;
            bool midDistanceWouldCount = observationResult.midDistanceQualified && _targetMidDistanceFrames < coreTarget;
            bool farDistanceWouldCount = observationResult.farDistanceQualified && _targetFarDistanceFrames < coreTarget;
            bool frontAngleWouldCount = observationResult.frontAngleQualified && _targetFrontAngleFrames < coreTarget;
            bool obliqueAngleWouldCount = observationResult.obliqueAngleQualified && _targetObliqueAngleFrames < coreTarget;
            bool extremeAngleWouldCount = observationResult.extremeAngleQualified && _targetExtremeAngleFrames < coreTarget;
            bool riskLayerWouldCount = observationResult.riskLayerQualified && _targetRiskLayerFrames < coreTarget;
            bool mainFrontWouldCount = observationResult.mainFrontQualified && _targetMainFrontFrames < coreTarget;
            bool mainLeftObliqueWouldCount = observationResult.mainLeftObliqueQualified && _targetMainLeftObliqueFrames < coreTarget;
            bool mainRightObliqueWouldCount = observationResult.mainRightObliqueQualified && _targetMainRightObliqueFrames < coreTarget;
            bool mainUpObliqueWouldCount = observationResult.mainUpObliqueQualified && _targetMainUpObliqueFrames < coreTarget;
            bool mainDownObliqueWouldCount = observationResult.mainDownObliqueQualified && _targetMainDownObliqueFrames < coreTarget;
            bool boundaryLeftWouldCount = observationResult.boundaryLeftQualified && _targetBoundaryLeftFrames < coreTarget;
            bool boundaryRightWouldCount = observationResult.boundaryRightQualified && _targetBoundaryRightFrames < coreTarget;
            bool cornerJunctionWouldCount = observationResult.cornerJunctionQualified && _targetCornerJunctionFrames < coreTarget;
            bool supplementRiskWouldCount = observationResult.nearEdgeSupplementRiskQualified && _targetNearEdgeSupplementRiskFrames < supplementTarget;
            bool supplementCreaseWouldCount = observationResult.nearEdgeSupplementCreaseQualified && _targetNearEdgeSupplementCreaseFrames < supplementTarget;
            bool supplementBoundaryWouldCount = observationResult.nearEdgeSupplementBoundaryQualified && _targetNearEdgeSupplementBoundaryFrames < supplementTarget;
            bool supplementLeftWouldCount = observationResult.nearEdgeSupplementLeftQualified && _targetNearEdgeSupplementLeftFrames < supplementTarget;
            bool supplementRightWouldCount = observationResult.nearEdgeSupplementRightQualified && _targetNearEdgeSupplementRightFrames < supplementTarget;
            bool supplementUpWouldCount = observationResult.nearEdgeSupplementUpQualified && _targetNearEdgeSupplementUpFrames < supplementTarget;
            bool supplementDownWouldCount = observationResult.nearEdgeSupplementDownQualified && _targetNearEdgeSupplementDownFrames < supplementTarget;
            bool supplementObliqueWouldCount = observationResult.nearEdgeSupplementObliqueQualified && _targetNearEdgeSupplementObliqueFrames < supplementTarget;
            bool supplementExtremeWouldCount = observationResult.nearEdgeSupplementExtremeQualified && _targetNearEdgeSupplementExtremeFrames < supplementTarget;

            bool acceptedByTargets = hasObservationResult && (boundaryRiskSupplementMode
                ? supplementBoundaryWouldCount || supplementCreaseWouldCount || supplementObliqueWouldCount
                : supplementMode
                ? supplementRiskWouldCount || supplementCreaseWouldCount || supplementBoundaryWouldCount ||
                  supplementLeftWouldCount || supplementRightWouldCount || supplementUpWouldCount || supplementDownWouldCount ||
                  supplementObliqueWouldCount || supplementExtremeWouldCount
                : useMultiCoreObservationTargets
                    ? stableMainWouldCount || topCoverageWouldCount || middleCoverageWouldCount || bottomCoverageWouldCount ||
                  nearDistanceWouldCount || midDistanceWouldCount || farDistanceWouldCount ||
                  frontAngleWouldCount || obliqueAngleWouldCount || extremeAngleWouldCount || riskLayerWouldCount ||
                  mainFrontWouldCount || mainLeftObliqueWouldCount || mainRightObliqueWouldCount ||
                  mainUpObliqueWouldCount || mainDownObliqueWouldCount ||
                  boundaryLeftWouldCount || boundaryRightWouldCount || cornerJunctionWouldCount
                    : farWouldCount || highAngleWouldCount || nearEdgeWouldCount);

            if (!acceptedByTargets)
            {
                _targetRejectedFrames++;
                _lastGateStatus = hasObservationResult ? BuildGateRejectStatus(observationResult) : "reject no observation data";
                AppendTargetGateRow(frameName, false, observationResult, false, false, false, _lastGateStatus);
                TryDeleteFrameDirectory(frameDirectory);
                MarkCaptureAttemptPose(pose);
                _pendingCapture = false;
                _pendingRefreshRequested = false;
                return;
            }

            if (boundaryRiskSupplementMode)
            {
                if (supplementBoundaryWouldCount) _targetNearEdgeSupplementBoundaryFrames++;
                if (supplementCreaseWouldCount) _targetNearEdgeSupplementCreaseFrames++;
                if (supplementObliqueWouldCount) _targetNearEdgeSupplementObliqueFrames++;
            }
            else if (supplementMode)
            {
                if (supplementRiskWouldCount) _targetNearEdgeSupplementRiskFrames++;
                if (supplementCreaseWouldCount) _targetNearEdgeSupplementCreaseFrames++;
                if (supplementBoundaryWouldCount) _targetNearEdgeSupplementBoundaryFrames++;
                if (supplementLeftWouldCount) _targetNearEdgeSupplementLeftFrames++;
                if (supplementRightWouldCount) _targetNearEdgeSupplementRightFrames++;
                if (supplementUpWouldCount) _targetNearEdgeSupplementUpFrames++;
                if (supplementDownWouldCount) _targetNearEdgeSupplementDownFrames++;
                if (supplementObliqueWouldCount) _targetNearEdgeSupplementObliqueFrames++;
                if (supplementExtremeWouldCount) _targetNearEdgeSupplementExtremeFrames++;
            }
            else
            {
                if (farWouldCount)
                    _targetFarSurfaceFrames++;
                if (highAngleWouldCount)
                    _targetHighAngleRiskFrames++;
                if (nearEdgeWouldCount)
                    _targetNearEdgeRiskFrames++;
            }

            if (!boundaryRiskSupplementMode && !supplementMode && useMultiCoreObservationTargets)
            {
                if (stableMainWouldCount) _targetStableMainFrames++;
                if (topCoverageWouldCount) _targetTopCoverageFrames++;
                if (middleCoverageWouldCount) _targetMiddleCoverageFrames++;
                if (bottomCoverageWouldCount) _targetBottomCoverageFrames++;
                if (nearDistanceWouldCount) _targetNearDistanceFrames++;
                if (midDistanceWouldCount) _targetMidDistanceFrames++;
                if (farDistanceWouldCount) _targetFarDistanceFrames++;
                if (frontAngleWouldCount) _targetFrontAngleFrames++;
                if (obliqueAngleWouldCount) _targetObliqueAngleFrames++;
                if (extremeAngleWouldCount) _targetExtremeAngleFrames++;
                if (riskLayerWouldCount) _targetRiskLayerFrames++;
                if (mainFrontWouldCount) _targetMainFrontFrames++;
                if (mainLeftObliqueWouldCount) _targetMainLeftObliqueFrames++;
                if (mainRightObliqueWouldCount) _targetMainRightObliqueFrames++;
                if (mainUpObliqueWouldCount) _targetMainUpObliqueFrames++;
                if (mainDownObliqueWouldCount) _targetMainDownObliqueFrames++;
                if (boundaryLeftWouldCount) _targetBoundaryLeftFrames++;
                if (boundaryRightWouldCount) _targetBoundaryRightFrames++;
                if (cornerJunctionWouldCount) _targetCornerJunctionFrames++;
            }

            _lastGateStatus = boundaryRiskSupplementMode
                ? $"accept directed risk {CompletedDirectedRiskSupplementMetricCount()}/{DirectedRiskSupplementMetricCount}"
                : supplementMode
                    ? $"accept near-edge supplement {CompletedNearEdgeSupplementMetricCount()}/{NearEdgeSupplementMetricCount}"
                    : useMultiCoreObservationTargets
                    ? $"accept core {CompletedCoreMetricCount()}/{CoreMetricCount} directed {CompletedDirectedMetricCount()}/{DirectedMetricCount}"
                    : $"accept far+{(farWouldCount ? 1 : 0)} angle+{(highAngleWouldCount ? 1 : 0)} near+{(nearEdgeWouldCount ? 1 : 0)}";
            AppendTargetGateRow(frameName, true, observationResult, farWouldCount, highAngleWouldCount, nearEdgeWouldCount, _lastGateStatus);

            ExportObservationStats(frameDirectory, frameName, verticesCsvPath, trianglesCsvPath, pose, writeOutputs: true);
        }

        if (gateFramesByRepeatCoverage)
        {
            RepeatCoverageGateResult repeatResult = EvaluateRepeatCoverageGate(verticesCsvPath, pose, _capturedFrameCount, apply: false);
            if (!repeatResult.accepted)
            {
                _repeatCoverageRejectedFrames++;
                _lastRepeatCoverageStatus = repeatResult.reason;
                StoreRepeatCoverageLastStats(repeatResult);
                AppendRepeatCoverageGateRow(frameName, false, repeatResult);
                TryDeleteFrameDirectory(frameDirectory);
                MarkCaptureAttemptPose(pose);
                _pendingCapture = false;
                _pendingRefreshRequested = false;
                return;
            }

            repeatResult = EvaluateRepeatCoverageGate(verticesCsvPath, pose, _capturedFrameCount, apply: true);
            _repeatCoverageAcceptedFrames++;
            _lastRepeatCoverageStatus = repeatResult.reason;
            StoreRepeatCoverageLastStats(repeatResult);
            AppendRepeatCoverageGateRow(frameName, true, repeatResult);
        }

        if (gateFramesByRepeatCoverage && exportObservationStats && !gateFramesByObservationTargets)
            ExportObservationStats(frameDirectory, frameName, verticesCsvPath, trianglesCsvPath, pose, writeOutputs: true);

        if (exportGridStateCsv)
            ExportGridStateCsv(gridStateCsvPath);
        if (exportRawDepthProbe)
            RequestRawDepthProbeExport(rawDepthCsvPath, rawDepthJsonPath, frameName);
        UpdateStabilityFromVerticesCsv(verticesCsvPath, _capturedFrameCount);
        if (!rawCoverageUpdatedBeforeMeshExport)
            UpdateRoomRawCoverage(frameName, pose, _capturedFrameCount);

        AppendManifestRow(frameName, _pendingReason, objPath, verticesCsvPath, trianglesCsvPath, cameraJsonPath, rawDepthCsvPath, rawDepthJsonPath, gridStateCsvPath, pose);
        _lastExportedFrameName = frameName;
        _capturedFrameCount++;
        MarkCaptureAttemptPose(pose);
        _pendingCapture = false;
        _pendingRefreshRequested = false;

        if (ShouldUseLegacyFullViewRoomRawCoverage() && AreRoomRawCoverageTargetsComplete())
            StopSession();
        else if (AreObservationTargetsComplete())
            StopSession();
        else if (_capturedFrameCount >= maxFramesPerSession)
            StopSession();
    }

    private void MarkCaptureAttemptPose(Transform pose)
    {
        _lastCaptureTime = Time.unscaledTime;
        if (pose == null)
            return;

        _lastCapturePosition = pose.position;
        _lastCaptureRotation = pose.rotation;
        _hasLastCapturePose = true;
    }

    private bool AreObservationTargetsComplete()
    {
        bool repeatComplete = !gateFramesByRepeatCoverage || _repeatCoverageStableVoxelCount >= Mathf.Max(1, repeatCoverageTargetStableVoxels);
        if (gateFramesByRepeatCoverage && (!gateFramesByObservationTargets || !exportObservationStats))
            return repeatComplete;

        if (!gateFramesByObservationTargets || !exportObservationStats)
            return false;

        int target = Mathf.Max(1, targetFramesPerObservationBucket);
        if (observationCaptureMode == ObservationCaptureMode.BoundaryRiskSupplement)
        {
            int supplementTarget = Mathf.Max(1, targetNearEdgeSupplementFrames);
            bool complete = _targetNearEdgeSupplementBoundaryFrames >= supplementTarget &&
                   _targetNearEdgeSupplementCreaseFrames >= supplementTarget &&
                   _targetNearEdgeSupplementObliqueFrames >= supplementTarget;
            return complete && repeatComplete;
        }

        if (observationCaptureMode == ObservationCaptureMode.NearEdgeSupplement)
        {
            int supplementTarget = Mathf.Max(1, targetNearEdgeSupplementFrames);
            bool complete = _targetNearEdgeSupplementRiskFrames >= supplementTarget &&
                   _targetNearEdgeSupplementCreaseFrames >= supplementTarget &&
                   _targetNearEdgeSupplementBoundaryFrames >= supplementTarget &&
                   _targetNearEdgeSupplementLeftFrames >= supplementTarget &&
                   _targetNearEdgeSupplementRightFrames >= supplementTarget &&
                   _targetNearEdgeSupplementUpFrames >= supplementTarget &&
                   _targetNearEdgeSupplementDownFrames >= supplementTarget &&
                   _targetNearEdgeSupplementObliqueFrames >= supplementTarget &&
                   _targetNearEdgeSupplementExtremeFrames >= supplementTarget;
            return complete && repeatComplete;
        }

        if (useMultiCoreObservationTargets)
        {
            int coreTarget = Mathf.Max(1, targetFramesPerCoreMetric);
            bool complete = _targetStableMainFrames >= coreTarget &&
                   _targetTopCoverageFrames >= coreTarget &&
                   _targetMiddleCoverageFrames >= coreTarget &&
                   _targetBottomCoverageFrames >= coreTarget &&
                   _targetNearDistanceFrames >= coreTarget &&
                   _targetMidDistanceFrames >= coreTarget &&
                   _targetFarDistanceFrames >= coreTarget &&
                   _targetFrontAngleFrames >= coreTarget &&
                   _targetObliqueAngleFrames >= coreTarget &&
                   _targetExtremeAngleFrames >= coreTarget &&
                   _targetRiskLayerFrames >= coreTarget &&
                   _targetMainFrontFrames >= coreTarget &&
                   _targetMainLeftObliqueFrames >= coreTarget &&
                   _targetMainRightObliqueFrames >= coreTarget &&
                   _targetMainUpObliqueFrames >= coreTarget &&
                   _targetMainDownObliqueFrames >= coreTarget &&
                   _targetBoundaryLeftFrames >= coreTarget &&
                   _targetBoundaryRightFrames >= coreTarget &&
                   _targetCornerJunctionFrames >= coreTarget;
            return complete && repeatComplete;
        }

        bool legacyComplete = _targetFarSurfaceFrames >= target &&
               _targetHighAngleRiskFrames >= target &&
               _targetNearEdgeRiskFrames >= target;
        return legacyComplete && repeatComplete;
    }

    private int CompletedNearEdgeSupplementMetricCount()
    {
        int target = Mathf.Max(1, targetNearEdgeSupplementFrames);
        int complete = 0;
        if (_targetNearEdgeSupplementRiskFrames >= target) complete++;
        if (_targetNearEdgeSupplementCreaseFrames >= target) complete++;
        if (_targetNearEdgeSupplementBoundaryFrames >= target) complete++;
        if (_targetNearEdgeSupplementLeftFrames >= target) complete++;
        if (_targetNearEdgeSupplementRightFrames >= target) complete++;
        if (_targetNearEdgeSupplementUpFrames >= target) complete++;
        if (_targetNearEdgeSupplementDownFrames >= target) complete++;
        if (_targetNearEdgeSupplementObliqueFrames >= target) complete++;
        if (_targetNearEdgeSupplementExtremeFrames >= target) complete++;
        return complete;
    }

    private int CompletedDirectedRiskSupplementMetricCount()
    {
        int target = Mathf.Max(1, targetNearEdgeSupplementFrames);
        int complete = 0;
        if (_targetNearEdgeSupplementBoundaryFrames >= target) complete++;
        if (_targetNearEdgeSupplementCreaseFrames >= target) complete++;
        if (_targetNearEdgeSupplementObliqueFrames >= target) complete++;
        return complete;
    }

    private int CompletedCoreMetricCount()
    {
        int target = Mathf.Max(1, targetFramesPerCoreMetric);
        int complete = 0;
        if (_targetStableMainFrames >= target) complete++;
        if (_targetTopCoverageFrames >= target) complete++;
        if (_targetMiddleCoverageFrames >= target) complete++;
        if (_targetBottomCoverageFrames >= target) complete++;
        if (_targetNearDistanceFrames >= target) complete++;
        if (_targetMidDistanceFrames >= target) complete++;
        if (_targetFarDistanceFrames >= target) complete++;
        if (_targetFrontAngleFrames >= target) complete++;
        if (_targetObliqueAngleFrames >= target) complete++;
        if (_targetExtremeAngleFrames >= target) complete++;
        if (_targetRiskLayerFrames >= target) complete++;
        return complete;
    }

    private int CompletedDirectedMetricCount()
    {
        int target = Mathf.Max(1, targetFramesPerCoreMetric);
        int complete = 0;
        if (_targetMainFrontFrames >= target) complete++;
        if (_targetMainLeftObliqueFrames >= target) complete++;
        if (_targetMainRightObliqueFrames >= target) complete++;
        if (_targetMainUpObliqueFrames >= target) complete++;
        if (_targetMainDownObliqueFrames >= target) complete++;
        if (_targetBoundaryLeftFrames >= target) complete++;
        if (_targetBoundaryRightFrames >= target) complete++;
        if (_targetCornerJunctionFrames >= target) complete++;
        return complete;
    }

    private string BuildGateRejectStatus(ObservationExportResult result)
    {
        if (observationCaptureMode == ObservationCaptureMode.BoundaryRiskSupplement)
            return $"reject directedRisk boundary={result.nearEdgeSupplementBoundaryCount}/{minNearEdgeSupplementBoundaryPointsPerFrame} crease={result.nearEdgeSupplementCreaseCount}/{minNearEdgeSupplementCreasePointsPerFrame} oblique={result.nearEdgeSupplementObliqueCount}/{Mathf.Max(1, minNearEdgeSupplementRiskPointsPerFrame / 2)} nearRange={FormatFloat(nearEdgeDistanceRangeMeters.x)}-{FormatFloat(nearEdgeDistanceRangeMeters.y)}m";

        if (observationCaptureMode == ObservationCaptureMode.NearEdgeSupplement)
            return $"reject near-edge supplement risk={result.nearEdgeSupplementRiskCount} crease={result.nearEdgeSupplementCreaseCount} boundary={result.nearEdgeSupplementBoundaryCount} lr={result.nearEdgeSupplementLeftCount}/{result.nearEdgeSupplementRightCount} ud={result.nearEdgeSupplementUpCount}/{result.nearEdgeSupplementDownCount} oblique={result.nearEdgeSupplementObliqueCount} extreme={result.nearEdgeSupplementExtremeCount}";

        if (!useMultiCoreObservationTargets)
            return $"reject far={result.farSurfaceCount} angleRisk={result.highAngleRiskCount} nearEdge={result.nearEdgeRiskCount}";

        return $"reject main={result.stableMainCount} top/mid/bot={result.topCoverageCount}/{result.middleCoverageCount}/{result.bottomCoverageCount} dist={result.nearDistanceCount}/{result.midDistanceCount}/{result.farDistanceCount} angle={result.frontAngleCount}/{result.obliqueAngleCount}/{result.extremeAngleCount} risk={result.riskLayerCount} directed={result.mainFrontCount}/{result.mainLeftObliqueCount}/{result.mainRightObliqueCount}/{result.mainUpObliqueCount}/{result.mainDownObliqueCount} edgeLR={result.boundaryLeftCount}/{result.boundaryRightCount} corner={result.cornerJunctionCount}";
    }

    private static void TryDeleteFrameDirectory(string frameDirectory)
    {
        if (string.IsNullOrEmpty(frameDirectory) || !Directory.Exists(frameDirectory))
            return;

        try
        {
            Directory.Delete(frameDirectory, recursive: true);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ScanCoverMultiFrameSessionExporter] Could not delete rejected frame directory: {frameDirectory}\n{ex.Message}");
        }
    }

    private void ResolveRefs()
    {
        if (!autoResolveRefs)
            return;

        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponent<ScanCoverDepthGridPointCloud>();
        if (depthGridPointCloud == null)
            depthGridPointCloud = FindObjectOfType<ScanCoverDepthGridPointCloud>();

        if (captureCamera == null)
            captureCamera = Camera.main;
        if (captureCamera == null)
        {
            Camera[] cameras = FindObjectsOfType<Camera>();
            for (int i = 0; i < cameras.Length; i++)
            {
                if (cameras[i] != null && cameras[i].isActiveAndEnabled && cameras[i].name.Contains("CenterEye"))
                {
                    captureCamera = cameras[i];
                    break;
                }
            }
        }

        if (poseSource == null && captureCamera != null)
            poseSource = captureCamera.transform;
    }

    private Transform ResolvePoseTransform()
    {
        if (poseSource != null)
            return poseSource;
        return captureCamera != null ? captureCamera.transform : null;
    }

    private void InitializeRoomRawCoverageAnchor()
    {
        if (!ShouldUseRoomRawCoverageHudOnlyCapture() || !lockRoomRawCoverageToStartView)
            return;

        Transform pose = ResolvePoseTransform();
        if (pose == null)
            return;

        _roomRawCoverageAnchorPosition = pose.position;
        _roomRawCoverageAnchorRotation = pose.rotation;
        _roomRawCoverageAnchorSet = true;
        _roomRawCoverageWindowActive = true;
        _roomRawCoverageWindowStartFrames = _roomRawCoverageFrames;
        _roomRawCoverageWindowStartValidSamples = _roomRawCoverageValidSamples;
        _roomRawCoverageWindowStartCoveredVoxels = _roomRawCoverageCoveredVoxels;
        _roomRawCoverageWindowStartStableVoxels = _roomRawCoverageStableVoxels;
        _roomRawCoverageWindowCoveredKeys.Clear();
        _roomRawCoverageWindowStableKeys.Clear();
        ResetRoomRawCoverageTileStats();
        if (freezeRoomRawCoverageViewFrameWhileRecording)
        {
            _roomRawCoverageViewFrameWorldPosition = pose.TransformPoint(roomRawCoverageViewFrameLocalPosition);
            _roomRawCoverageViewFrameWorldRotation = pose.rotation;
            _roomRawCoverageViewFrameLocked = true;
        }
    }

    private bool IsWithinRoomRawCoverageAnchor(Transform pose, out float angleDegrees, out float moveMeters)
    {
        angleDegrees = 0f;
        moveMeters = 0f;
        if (!lockRoomRawCoverageToStartView)
            return true;
        if (!_roomRawCoverageAnchorSet || pose == null)
            return true;

        Vector3 frameForward = _roomRawCoverageViewFrameLocked
            ? _roomRawCoverageViewFrameWorldRotation * Vector3.forward
            : _roomRawCoverageAnchorRotation * Vector3.forward;
        Vector3 framePosition = _roomRawCoverageViewFrameLocked
            ? _roomRawCoverageViewFrameWorldPosition
            : _roomRawCoverageAnchorPosition;

        angleDegrees = Vector3.Angle(frameForward, pose.forward);
        moveMeters = Vector3.Distance(framePosition, pose.position);

        if (!_roomRawCoverageViewFrameLocked)
        {
            return angleDegrees <= Mathf.Max(1f, roomRawCoverageMaxAnchorAngleDegrees) &&
                   moveMeters <= Mathf.Max(0f, roomRawCoverageMaxAnchorMoveMeters);
        }

        Plane framePlane = new Plane(frameForward, framePosition);
        Ray gazeRay = new Ray(pose.position, pose.forward);
        if (!framePlane.Raycast(gazeRay, out float enter) || enter <= 0f)
            return false;

        Vector3 hit = gazeRay.GetPoint(enter);
        Vector3 localHit = Quaternion.Inverse(_roomRawCoverageViewFrameWorldRotation) * (hit - framePosition);
        float halfWidth = Mathf.Max(0.05f, roomRawCoverageViewFrameWidth) * 0.5f;
        float halfHeight = Mathf.Max(0.05f, roomRawCoverageViewFrameHeight) * 0.5f;
        float horizontalScale = 1f + Mathf.Clamp(roomRawCoverageViewFrameLeaveMargin, 0f, 0.4f);
        float bottomScale = 1f + Mathf.Clamp(roomRawCoverageViewFrameVerticalLeaveMargin, 0f, 0.8f);
        float topScale = bottomScale + Mathf.Clamp(roomRawCoverageViewFrameTopExtraLeaveMargin, 0f, 0.8f);
        return Mathf.Abs(localHit.x) <= halfWidth * horizontalScale &&
               localHit.y >= -halfHeight * bottomScale &&
               localHit.y <= halfHeight * topScale;
    }

    private bool ShouldUseRoomRawCoverageHudOnlyCapture()
    {
        return roomRawCoverageHudOnlyCapture && trackRoomRawCoverage && !ShouldUseLegacyFullViewRoomRawCoverage();
    }

    private bool ShouldUseLegacyFullViewRoomRawCoverage()
    {
        return useLegacyFullViewRoomRawCapture && trackRoomRawCoverage;
    }

    private bool ShouldUseRoomRawCoverageSessionMode()
    {
        return trackRoomRawCoverage && (ShouldUseLegacyFullViewRoomRawCoverage() || ShouldUseRoomRawCoverageHudOnlyCapture());
    }

    private void ResetLegacyRoomRawCoverageProgressWatch()
    {
        _legacyRoomRawCoverageLastProgressTime = Time.unscaledTime;
        _legacyRoomRawCoverageLastCoveredVoxels = _roomRawCoverageCoveredVoxels;
        _legacyRoomRawCoverageLastStableVoxels = _roomRawCoverageStableVoxels;
        _legacyRoomRawCoverageDisorderFrames = 0;
        _lastRoomRawFuseStatus = string.Empty;
        _lastObservationOrderScore = default;
    }

    private ObservationOrderScore BuildObservationOrderScore(
        int samples,
        int support,
        int useful,
        int riskHits,
        float riskRatio,
        int minSamples,
        int stableSupport,
        int minUseful,
        float disorderRiskThreshold,
        float lowUsefulRiskThreshold,
        bool focusLost,
        bool allowFuse)
    {
        return BuildObservationOrderScore(
            samples,
            support,
            useful,
            riskHits,
            riskRatio,
            minSamples,
            stableSupport,
            minUseful,
            disorderRiskThreshold,
            lowUsefulRiskThreshold,
            focusLost,
            allowFuse,
            default);
    }

    private ObservationOrderScore BuildObservationOrderScore(
        int samples,
        int support,
        int useful,
        int riskHits,
        float riskRatio,
        int minSamples,
        int stableSupport,
        int minUseful,
        float disorderRiskThreshold,
        float lowUsefulRiskThreshold,
        bool focusLost,
        bool allowFuse,
        RawObservationOrderResult rawOrder)
    {
        ObservationOrderScore score = default;
        score.samples = Mathf.Max(0, samples);
        score.support = Mathf.Max(0, support);
        score.useful = Mathf.Max(0, useful);
        score.riskHits = Mathf.Max(0, riskHits);
        score.riskRatio = Mathf.Clamp01(riskRatio);
        score.hasData = score.samples > 0;

        int safeStableSupport = Mathf.Max(1, stableSupport);
        int safeUseful = Mathf.Max(1, minUseful);
        score.supportRatio = Mathf.Clamp01(score.support / (float)safeStableSupport);
        score.usefulRatio = Mathf.Clamp01(score.useful / (float)safeUseful);
        float riskBudget = Mathf.Max(0.01f, disorderRiskThreshold);
        float riskPenalty = Mathf.Clamp01(score.riskRatio / riskBudget);
        score.score01 = Mathf.Clamp01(score.supportRatio * 0.45f + score.usefulRatio * 0.35f + (1f - riskPenalty) * 0.20f);

        bool enoughSamples = score.samples >= Mathf.Max(1, minSamples);
        bool highRisk = enoughSamples && score.riskRatio >= Mathf.Clamp01(disorderRiskThreshold);
        bool lowUseful = score.useful < Mathf.Max(0, minUseful);
        bool lowUsefulRisk = enoughSamples && score.riskRatio >= Mathf.Clamp01(lowUsefulRiskThreshold);
        bool rawHasEnoughEdges = useRawObservationOrderScore && rawOrder.hasData && rawOrder.testedEdges >= Mathf.Max(1, rawObservationOrderMinEdges);
        float rawContinuityScore = rawHasEnoughEdges
            ? (rawOrder.centerTestedEdges >= Mathf.Max(1, rawObservationOrderMinEdges / 2)
                ? Mathf.Clamp01(rawOrder.orderRatio * 0.35f + rawOrder.centerOrderRatio * 0.65f)
                : Mathf.Clamp01(rawOrder.orderRatio))
            : 0f;
        float centerBadRatio = rawHasEnoughEdges ? Mathf.Clamp01(1f - rawOrder.centerOrderRatio) : 0f;
        float badEdgeWarn = Mathf.Clamp01(rawObservationOrderWarnBadEdgeRatio);
        float badEdgeFuse = Mathf.Max(badEdgeWarn + 0.001f, Mathf.Clamp01(rawObservationOrderFuseBadEdgeRatio));
        float centerBadWarn = Mathf.Clamp01(rawObservationOrderWarnCenterBadEdgeRatio);
        float centerBadFuse = Mathf.Max(centerBadWarn + 0.001f, Mathf.Clamp01(rawObservationOrderFuseCenterBadEdgeRatio));
        float badQuadFuse = Mathf.Max(0.001f, Mathf.Clamp01(rawObservationOrderFuseBadQuadRatio));
        float badEdgePressure = rawHasEnoughEdges ? Mathf.Max(
            rawOrder.badEdgeRatio / badEdgeFuse,
            centerBadRatio / centerBadFuse,
            rawOrder.badQuadRatio / badQuadFuse) : 0f;
        float rawScore = rawHasEnoughEdges ? Mathf.Min(rawContinuityScore, Mathf.Clamp01(1f - badEdgePressure)) : 0f;
        bool rawWarn = rawHasEnoughEdges && (
            rawContinuityScore <= Mathf.Clamp01(rawObservationOrderWarnRatio) ||
            rawOrder.badEdgeRatio >= badEdgeWarn ||
            centerBadRatio >= centerBadWarn);
        bool rawFuse = rawHasEnoughEdges && (
            rawContinuityScore <= Mathf.Clamp01(rawObservationOrderFuseRatio) ||
            rawOrder.badEdgeRatio >= badEdgeFuse ||
            centerBadRatio >= centerBadFuse ||
            rawOrder.badQuadRatio >= badQuadFuse);
        if (rawHasEnoughEdges)
            score.score01 = rawScore;

        score.rawOrderRatio = rawHasEnoughEdges ? rawOrder.orderRatio : 0f;
        score.rawCenterOrderRatio = rawHasEnoughEdges ? rawOrder.centerOrderRatio : 0f;
        score.rawBadEdgeRatio = rawHasEnoughEdges ? rawOrder.badEdgeRatio : 0f;
        score.rawBadQuadRatio = rawHasEnoughEdges ? rawOrder.badQuadRatio : 0f;
        score.rawTestedEdges = rawOrder.testedEdges;
        score.rawBadEdges = rawOrder.badEdges;
        score.rawTestedQuads = rawOrder.testedQuads;
        score.rawBadQuads = rawOrder.badQuads;

        score.disordered = focusLost || rawFuse || highRisk || (lowUseful && lowUsefulRisk);
        score.shouldFuse = allowFuse && score.disordered;

        if (score.disordered)
            score.representationState = 3;
        else if (rawWarn)
            score.representationState = 1;
        else if (score.support >= safeStableSupport)
            score.representationState = 2;
        else if (score.samples > 0)
            score.representationState = 1;
        else
            score.representationState = 0;

        if (focusLost)
            score.reason = "focus-lost";
        else if (rawFuse)
            score.reason = $"raw-order-fuse score={rawScore:0.00} badEdge={rawOrder.badEdgeRatio:0.000}/{badEdgeFuse:0.000} centerBad={centerBadRatio:0.000}/{centerBadFuse:0.000} {rawOrder.reason}";
        else if (rawWarn)
            score.reason = $"raw-order-warn score={rawScore:0.00} badEdge={rawOrder.badEdgeRatio:0.000}/{badEdgeWarn:0.000} centerBad={centerBadRatio:0.000}/{centerBadWarn:0.000} {rawOrder.reason}";
        else if (highRisk)
            score.reason = $"risk {score.riskRatio:0.00}>={Mathf.Clamp01(disorderRiskThreshold):0.00}";
        else if (lowUseful && lowUsefulRisk)
            score.reason = $"low-useful {score.useful}/{Mathf.Max(0, minUseful)} risk {score.riskRatio:0.00}";
        else if (rawHasEnoughEdges)
            score.reason = $"raw-order {rawScore:0.00} {rawOrder.reason}";
        else
            score.reason = $"ordered support {score.support} useful {score.useful} risk {score.riskRatio:0.00}";

        return score;
    }

    private Color GetObservationOrderScoreColor(ObservationOrderScore score)
    {
        switch (score.representationState)
        {
            case 3:
                return snapshotGridRiskColor;
            case 2:
                return snapshotGridStableColor;
            case 1:
                return snapshotGridSeenColor;
            default:
                return snapshotGridPendingColor;
        }
    }

    private bool TryAutoStopLegacyRoomRawCoverageOnDisorder()
    {
        if (roomRawCoverageDisorderFuseMode != RoomRawCoverageDisorderFuseMode.Strict ||
            !autoStopLegacyRoomRawCoverageOnDisorder ||
            !_isRecording ||
            !ShouldUseRoomRawCoverageSessionMode())
            return false;

        int minFrames = Mathf.Max(1, legacyRoomRawCoverageDisorderMinFrames);
        int minSamples = Mathf.Max(1, legacyRoomRawCoverageDisorderMinValidSamples);
        int requiredFrames = Mathf.Max(1, legacyRoomRawCoverageDisorderConsecutiveFrames);
        ObservationOrderScore score = _lastObservationOrderScore;
        bool focusLost = score.hasData && score.reason == "focus-lost";

        if (!focusLost && (_roomRawCoverageFrames < minFrames || _lastRoomRawValidSamples < minSamples))
        {
            _legacyRoomRawCoverageDisorderFrames = 0;
            _lastRoomRawFuseStatus = $"\u7194\u65ad\u5f85\u5224\uff1a\u5e27 {_roomRawCoverageFrames}/{minFrames}\uff0c\u6709\u6548 {_lastRoomRawValidSamples}/{minSamples}";
            return false;
        }

        float disorderRiskThreshold = Mathf.Clamp01(legacyRoomRawCoverageDisorderRiskRatio);
        bool disordered = score.hasData && score.shouldFuse;
        bool seamConsistent = _lastRoomRawHistoryRehitVoxels >= Mathf.Max(1, roomRawCoverageMinHistoryRehitVoxelsPerFrame) &&
            _lastRoomRawHistoryAgreementRatio >= Mathf.Clamp01(roomRawCoverageMinHistoryAgreementRatio);
        bool lowOverlapTurn = _loadedRoomRawCoverageSessions > 0 &&
            _lastRoomRawOverlapRatio < Mathf.Clamp01(roomRawCoverageIdealOverlapMinRatio);
        bool hardDisorder = IsHardRoomRawCoverageDisorder(score);

        if (!disordered)
        {
            _legacyRoomRawCoverageDisorderFrames = 0;
            _lastRoomRawFuseStatus = $"熔断监测：OrderScore {score.score01:0.00}，风险 {score.riskRatio:0.00}/{disorderRiskThreshold:0.00}，新增 {score.useful}，接缝 {FormatFloat(_lastRoomRawHistoryAgreementRatio)}";
            return false;
        }

        if (!hardDisorder && (seamConsistent || lowOverlapTurn))
        {
            _legacyRoomRawCoverageDisorderFrames = Mathf.Max(0, _legacyRoomRawCoverageDisorderFrames - 1);
            string toleranceReason = seamConsistent
                ? "接缝一致，坏帧仅拒绝不终止"
                : "重叠偏低疑似换视角，坏帧仅拒绝不终止";
            _lastRoomRawFuseStatus = $"软熔断容错：{toleranceReason}，OrderScore {score.score01:0.00}，overlap {FormatFloat(_lastRoomRawOverlapRatio)}，seam {FormatFloat(_lastRoomRawHistoryAgreementRatio)}";
            return false;
        }

        _legacyRoomRawCoverageDisorderFrames++;
        int requiredNow = hardDisorder ? 1 : requiredFrames;
        _lastRoomRawFuseStatus = $"熔断倒计时：{_legacyRoomRawCoverageDisorderFrames}/{requiredNow}，{(hardDisorder ? "硬坏帧" : "连续软坏帧")}，OrderScore {score.score01:0.00}，{score.reason}";
        if (_legacyRoomRawCoverageDisorderFrames < requiredNow)
            return false;

        string reason = focusLost
            ? "\u5df2\u7194\u65ad\u5e76\u843d\u76d8\uff1a\u89c6\u7ebf\u79bb\u5f00\u91c7\u96c6\u533a\u57df\uff0c\u8bf7\u6362\u89d2\u5ea6\u540e\u6309 B \u91cd\u65b0\u91c7\u96c6\u3002"
            : "\u5df2\u7194\u65ad\u5e76\u843d\u76d8\uff1a\u70b9\u9635\u65e0\u5e8f/\u98ce\u9669\u8fc7\u9ad8\uff0c\u8bf7\u6362\u89d2\u5ea6\u540e\u6309 B \u91cd\u65b0\u91c7\u96c6\u3002";
        _lastRoomRawStatus = reason;
        _lastRoomRawFuseStatus = reason;
        Debug.LogWarning("[ScanCover] " + reason, this);
        StopSession();
        return true;
    }

    private bool IsHardRoomRawCoverageDisorder(ObservationOrderScore score)
    {
        if (!score.hasData)
            return false;

        float centerBadRatio = score.rawCenterOrderRatio > 0f ? Mathf.Clamp01(1f - score.rawCenterOrderRatio) : 0f;
        if (score.rawBadEdgeRatio >= Mathf.Clamp01(legacyRoomRawCoverageHardFuseBadEdgeRatio))
            return true;
        if (centerBadRatio >= Mathf.Clamp01(legacyRoomRawCoverageHardFuseCenterBadEdgeRatio))
            return true;
        if (score.rawBadQuadRatio >= Mathf.Clamp01(legacyRoomRawCoverageHardFuseBadQuadRatio))
            return true;
        if (score.riskRatio >= Mathf.Clamp01(legacyRoomRawCoverageHardFuseRiskRatio))
            return true;
        return false;
    }

    private void UpdateLegacyRoomRawCoverageCompletion()
    {
        if (!_isRecording || !ShouldUseRoomRawCoverageSessionMode())
            return;

        int coveredDelta = _roomRawCoverageCoveredVoxels - _legacyRoomRawCoverageLastCoveredVoxels;
        int stableDelta = _roomRawCoverageStableVoxels - _legacyRoomRawCoverageLastStableVoxels;
        if (coveredDelta >= Mathf.Max(0, legacyRoomRawCoveragePlateauMinCoveredVoxels) ||
            stableDelta >= Mathf.Max(0, legacyRoomRawCoveragePlateauMinStableVoxels))
        {
            ResetLegacyRoomRawCoverageProgressWatch();
        }

        if (TryAutoStopLegacyRoomRawCoverageOnDisorder())
            return;

        if (_pendingCapture)
            return;

        if (autoStopLegacyRoomRawCoverageWhenTargetsComplete && AreRoomRawCoverageTargetsComplete())
        {
            _lastRoomRawStatus = "\u5df2\u5b8c\u6210\uff1a\u5168\u5c40\u8d28\u91cf\u76ee\u6807\u5df2\u8fbe\u5230\u3002";
            StopSession();
            return;
        }

        if (autoStopLegacyRoomRawCoverageOnPlateau &&
            _roomRawCoverageFrames >= Mathf.Max(1, legacyRoomRawCoverageMinFramesBeforePlateau) &&
            Time.unscaledTime - _legacyRoomRawCoverageLastProgressTime >= Mathf.Max(0.5f, legacyRoomRawCoveragePlateauSeconds))
        {
            _lastRoomRawStatus = "\u5df2\u5b8c\u6210\uff1a\u5f53\u524d\u533a\u57df\u589e\u957f\u505c\u6ede\uff0c\u8bf7\u79fb\u52a8\u5230\u4e0b\u4e00\u4e2a\u89c6\u89d2\u3002";
            StopSession();
        }
    }

    private string GetLegacyRoomRawCoverageHint()
    {
        if (!ShouldUseRoomRawCoverageSessionMode())
            return string.Empty;

        if (!_isRecording)
            return "\u6309 B \u5f00\u59cb\u5168\u89c6\u91ce\u91c7\u96c6\u3002";

        if (_lastRoomRawRiskRatio > Mathf.Clamp01(roomRawCoverageTargetRiskRatio) &&
            _lastRoomRawValidSamples >= roomRawCoverageMinRiskSamplesPerFrame)
            return "\u6362\u89d2\u5ea6\uff1a\u8fb9\u7f18/\u659c\u89c6\u98ce\u9669\u504f\u9ad8\u3002";

        if (_lastRoomRawHistoryRehitVoxels >= Mathf.Max(1, roomRawCoverageMinHistoryRehitVoxelsPerFrame) &&
            _lastRoomRawHistoryAgreementRatio < Mathf.Clamp01(roomRawCoverageMinHistoryAgreementRatio))
            return "接缝冲突：当前 Raw 和历史重叠区对不上，请退回或换角度。";

        if (_lastRoomRawOverlapRatio > Mathf.Clamp01(roomRawCoverageIdealOverlapMaxRatio) &&
            _lastRoomRawNewVoxels < Mathf.Max(1, roomRawCoverageMinOverlapNewVoxelsPerFrame))
            return "重叠过高：主要在重复旧区域，向未扫区域平移。";

        if (_lastRoomRawOverlapRatio < Mathf.Clamp01(roomRawCoverageIdealOverlapMinRatio) &&
            _roomRawCoverageFrames > 1 &&
            _loadedRoomRawCoverageSessions > 0)
            return "重叠偏低：保留一点已扫区域作为接缝。";

        if (_lastRoomRawNewVoxels >= Mathf.Max(1, roomRawCoverageMinNewVoxelsPerFrame))
            return "\u7ee7\u7eed\u4fdd\u6301\uff1a\u65b0\u8986\u76d6\u4ecd\u5728\u589e\u957f\u3002";

        if (_lastRoomRawParallaxRehitVoxels >= Mathf.Max(1, roomRawCoverageMinParallaxRehitVoxelsPerFrame) &&
            _lastRoomRawStableNewVoxels >= Mathf.Max(1, roomRawCoverageMinNewStableVoxelsPerFrame))
            return "\u5f53\u524d\u533a\u57df\u53ef\u7528\uff1a\u53ef\u4ee5\u6362\u4e0b\u4e00\u4e2a\u89c6\u89d2\u3002";

        if (_lastRoomRawRehitVoxels >= Mathf.Max(1, roomRawCoverageMinOverlapRehitVoxelsPerFrame))
            return "\u91cd\u53e0\u6709\u6548\uff1a\u6162\u6162\u79fb\u5411\u76f8\u90bb\u672a\u626b\u533a\u57df\u3002";

        return "\u589e\u957f\u53d8\u6162\uff1a\u4fa7\u79fb\u6216\u770b\u5411\u672a\u626b\u89d2\u5ea6\u3002";
    }


    private void ApplyRecordingDepthGridDisplayMode(bool recording)
    {
        if (depthGridPointCloud == null)
            return;

        if (recording && questCloneContinuousCaptureMode)
        {
            if (!_hasDepthGridPreviewDisplayRestoreState)
            {
                _restoreDepthGridPreviewDisplay = depthGridPointCloud.PreviewDisplayVisible;
                _hasDepthGridPreviewDisplayRestoreState = true;
            }

            depthGridPointCloud.SetPreviewDisplayVisible(true);
            depthGridPointCloud.SetCaptureOnlySnapshotMode(false);
            return;
        }

        if (recording && (ShouldUseRoomRawCoverageHudOnlyCapture() || ShouldUseLegacyFullViewRoomRawCoverage()))
        {
            if (!_hasDepthGridPreviewDisplayRestoreState)
            {
                _restoreDepthGridPreviewDisplay = depthGridPointCloud.PreviewDisplayVisible;
                _hasDepthGridPreviewDisplayRestoreState = true;
            }

            bool keepSnapshotGridVisible = _snapshotGridMaskActive || ShouldUseSnapshotGridCaptureMask();
            if (keepSnapshotGridVisible)
            {
                depthGridPointCloud.SetPreviewDisplayVisible(true);
                depthGridPointCloud.SetCaptureOnlySnapshotMode(false);
            }
            else
            {
                if (hideDepthGridPreviewWhileRecording)
                    depthGridPointCloud.SetPreviewDisplayVisible(false);
                depthGridPointCloud.SetCaptureOnlySnapshotMode(ShouldUseRoomRawCoverageHudOnlyCapture());
            }
            return;
        }

        depthGridPointCloud.SetCaptureOnlySnapshotMode(false);
        if (_hasDepthGridPreviewDisplayRestoreState)
        {
            depthGridPointCloud.SetPreviewDisplayVisible(_restoreDepthGridPreviewDisplay);
            _hasDepthGridPreviewDisplayRestoreState = false;
        }
    }

    private bool AreRoomRawCoverageTargetsComplete()
    {
        if (!trackRoomRawCoverage)
            return false;

        if (ShouldUseRoomRawCoverageHudOnlyCapture() && lockRoomRawCoverageToStartView)
        {
            return _roomRawCoverageWindowActive &&
                   GetRoomRawCoverageWindowFrames() >= Mathf.Max(1, roomRawCoverageLocalTargetFrames) &&
                   GetRoomRawCoverageWindowValidSamples() >= Mathf.Max(1, roomRawCoverageLocalTargetValidSamples) &&
                   GetRoomRawCoverageWindowCoveredVoxels() >= Mathf.Max(1, roomRawCoverageLocalTargetVoxels) &&
                   GetRoomRawCoverageWindowStableVoxels() >= Mathf.Max(1, roomRawCoverageLocalTargetStableVoxels);
        }

        return _roomRawCoverageFrames >= Mathf.Max(1, roomRawCoverageTargetFrames) &&
               _roomRawCoverageValidSamples >= Mathf.Max(1, roomRawCoverageTargetValidSamples) &&
               _roomRawCoverageCoveredVoxels >= Mathf.Max(1, roomRawCoverageTargetVoxels) &&
               _roomRawCoverageStableVoxels >= Mathf.Max(1, roomRawCoverageTargetStableVoxels) &&
               _roomRawCoverageHighVoxels >= Mathf.Max(1, roomRawCoverageTargetHighVoxels) &&
               _roomRawCoverageHighStableVoxels >= Mathf.Max(1, roomRawCoverageTargetHighStableVoxels) &&
               _roomRawCoverageLowVoxels >= Mathf.Max(1, roomRawCoverageTargetLowVoxels) &&
               _roomRawCoverageLowStableVoxels >= Mathf.Max(1, roomRawCoverageTargetLowStableVoxels) &&
               _roomRawCoverageRiskVoxels >= Mathf.Max(1, Mathf.RoundToInt(roomRawCoverageTargetVoxels * Mathf.Clamp01(roomRawCoverageTargetRiskRatio))) &&
               _roomRawCoveragePoseCellCount >= Mathf.Max(1, roomRawCoverageTargetPoseCells);
    }

    private int GetRoomRawCoverageWindowFrames()
    {
        return Mathf.Max(0, _roomRawCoverageFrames - _roomRawCoverageWindowStartFrames);
    }

    private int GetRoomRawCoverageWindowValidSamples()
    {
        return Mathf.Max(0, _roomRawCoverageValidSamples - _roomRawCoverageWindowStartValidSamples);
    }

    private int GetRoomRawCoverageWindowCoveredVoxels()
    {
        if (ShouldUseRoomRawCoverageHudOnlyCapture() && lockRoomRawCoverageToStartView)
            return _roomRawCoverageWindowCoveredKeys.Count;
        return Mathf.Max(0, _roomRawCoverageCoveredVoxels - _roomRawCoverageWindowStartCoveredVoxels);
    }

    private int GetRoomRawCoverageWindowStableVoxels()
    {
        if (ShouldUseRoomRawCoverageHudOnlyCapture() && lockRoomRawCoverageToStartView)
            return _roomRawCoverageWindowStableKeys.Count;
        return Mathf.Max(0, _roomRawCoverageStableVoxels - _roomRawCoverageWindowStartStableVoxels);
    }

    private int GetRoomRawCoverageTileCount()
    {
        return Mathf.Max(1, roomRawCoverageTileColumns) * Mathf.Max(1, roomRawCoverageTileRows);
    }

    private void EnsureRoomRawCoverageTileStats()
    {
        int count = GetRoomRawCoverageTileCount();
        if (_roomRawCoverageTileCoveredKeys != null &&
            _roomRawCoverageTileStableKeys != null &&
            _roomRawCoverageTileSamples != null &&
            _roomRawCoverageTileCoveredKeys.Length == count &&
            _roomRawCoverageTileStableKeys.Length == count &&
            _roomRawCoverageTileSamples.Length == count)
            return;

        _roomRawCoverageTileCoveredKeys = new HashSet<Vector3Int>[count];
        _roomRawCoverageTileStableKeys = new HashSet<Vector3Int>[count];
        _roomRawCoverageTileSamples = new int[count];
        for (int i = 0; i < count; i++)
        {
            _roomRawCoverageTileCoveredKeys[i] = new HashSet<Vector3Int>();
            _roomRawCoverageTileStableKeys[i] = new HashSet<Vector3Int>();
        }
    }

    private void ResetRoomRawCoverageTileStats()
    {
        EnsureRoomRawCoverageTileStats();
        for (int i = 0; i < _roomRawCoverageTileSamples.Length; i++)
        {
            _roomRawCoverageTileSamples[i] = 0;
            _roomRawCoverageTileCoveredKeys[i].Clear();
            _roomRawCoverageTileStableKeys[i].Clear();
        }
        _roomRawCoverageMissingHint = "none";
    }

    private void RegisterRoomRawCoverageTileHit(float u, float v, Vector3Int key, bool stable)
    {
        EnsureRoomRawCoverageTileStats();
        float margin = Mathf.Clamp(roomRawCoverageFocusMargin, 0f, 0.45f);
        float focusSize = Mathf.Max(0.001f, 1f - margin * 2f);
        float focusU = Mathf.Clamp01((u - margin) / focusSize);
        float focusV = Mathf.Clamp01((v - margin) / focusSize);
        int cols = Mathf.Max(1, roomRawCoverageTileColumns);
        int rows = Mathf.Max(1, roomRawCoverageTileRows);
        int col = Mathf.Clamp(Mathf.FloorToInt(focusU * cols), 0, cols - 1);
        int row = Mathf.Clamp(Mathf.FloorToInt(focusV * rows), 0, rows - 1);
        int index = row * cols + col;
        if (index < 0 || index >= _roomRawCoverageTileSamples.Length)
            return;

        _roomRawCoverageTileSamples[index]++;
        _roomRawCoverageTileCoveredKeys[index].Add(key);
        if (stable)
            _roomRawCoverageTileStableKeys[index].Add(key);
    }

    private bool GetRoomRawCoverageFocusFlags(float u, float v, out bool core, out bool edgeBuffer)
    {
        float focusMargin = Mathf.Clamp(roomRawCoverageFocusMargin, 0f, 0.45f);
        float coreMargin = focusMargin;
        float bufferMargin = focusMargin;
        if (roomRawCoverageUseFallbackTolerance)
        {
            coreMargin = Mathf.Clamp(Mathf.Max(focusMargin, roomRawCoverageCoreMargin), 0f, 0.45f);
            bufferMargin = Mathf.Clamp(Mathf.Min(focusMargin, roomRawCoverageEdgeBufferMargin), 0f, 0.45f);
        }

        bool focus = u >= bufferMargin && u <= 1f - bufferMargin &&
            v >= bufferMargin && v <= 1f - bufferMargin;
        core = u >= coreMargin && u <= 1f - coreMargin &&
            v >= coreMargin && v <= 1f - coreMargin;
        edgeBuffer = focus && !core;
        return focus;
    }

    private bool IsRoomRawCoverageAnchorUsable(Transform pose, out float angle, out float move, out bool fallback)
    {
        fallback = false;
        if (IsWithinRoomRawCoverageAnchor(pose, out angle, out move))
            return true;

        if (!roomRawCoverageUseFallbackTolerance)
            return false;

        float maxAngle = Mathf.Clamp(roomRawCoverageMaxAnchorAngleDegrees, 1f, 45f) +
            Mathf.Max(0f, roomRawCoverageAnchorAngleFallbackDegrees);
        float maxMove = Mathf.Max(0f, roomRawCoverageMaxAnchorMoveMeters) +
            Mathf.Max(0f, roomRawCoverageAnchorMoveFallbackMeters);
        fallback = angle <= maxAngle && move <= maxMove;
        return fallback;
    }

    private float GetRoomRawCoverageTileProgress01(int index)
    {
        EnsureRoomRawCoverageTileStats();
        if (index < 0 || index >= _roomRawCoverageTileSamples.Length)
            return 0f;

        float covered = _roomRawCoverageTileCoveredKeys[index].Count / (float)Mathf.Max(1, roomRawCoverageTileTargetCoveredVoxels);
        float stable = _roomRawCoverageTileStableKeys[index].Count / (float)Mathf.Max(1, roomRawCoverageTileTargetStableVoxels);
        return Mathf.Clamp01(Mathf.Min(covered, stable));
    }

    private int GetRoomRawCoverageCompleteTileCount()
    {
        EnsureRoomRawCoverageTileStats();
        int complete = 0;
        for (int i = 0; i < _roomRawCoverageTileSamples.Length; i++)
        {
            if (GetRoomRawCoverageTileProgress01(i) >= 0.95f)
                complete++;
        }
        return complete;
    }

    private int GetRoomRawCoverageAimTileIndex()
    {
        if (TryGetRoomRawCoverageAimTileIndex(out int index))
            return index;

        int cols = Mathf.Max(1, roomRawCoverageTileColumns);
        int rows = Mathf.Max(1, roomRawCoverageTileRows);
        int col = Mathf.Clamp(cols / 2, 0, cols - 1);
        int row = Mathf.Clamp(rows / 2, 0, rows - 1);
        return row * cols + col;
    }

    private bool TryGetRoomRawCoverageAimTileIndex(out int index)
    {
        index = -1;
        Transform pose = ResolvePoseTransform();
        if (pose == null || !_roomRawCoverageViewFrameLocked)
            return false;

        Vector3 frameForward = _roomRawCoverageViewFrameWorldRotation * Vector3.forward;
        Plane framePlane = new Plane(frameForward, _roomRawCoverageViewFrameWorldPosition);
        Ray gazeRay = new Ray(pose.position, pose.forward);
        if (!framePlane.Raycast(gazeRay, out float enter) || enter <= 0f)
            return false;

        Vector3 hit = gazeRay.GetPoint(enter);
        Vector3 localHit = Quaternion.Inverse(_roomRawCoverageViewFrameWorldRotation) * (hit - _roomRawCoverageViewFrameWorldPosition);
        float halfWidth = Mathf.Max(0.05f, roomRawCoverageViewFrameWidth) * 0.5f;
        float halfHeight = Mathf.Max(0.05f, roomRawCoverageViewFrameHeight) * 0.5f;
        if (Mathf.Abs(localHit.x) > halfWidth || Mathf.Abs(localHit.y) > halfHeight)
            return false;

        float u = Mathf.InverseLerp(-halfWidth, halfWidth, localHit.x);
        float v = Mathf.InverseLerp(-halfHeight, halfHeight, localHit.y);
        int cols = Mathf.Max(1, roomRawCoverageTileColumns);
        int rows = Mathf.Max(1, roomRawCoverageTileRows);
        int col = Mathf.Clamp(Mathf.FloorToInt(u * cols), 0, cols - 1);
        int row = Mathf.Clamp(Mathf.FloorToInt(v * rows), 0, rows - 1);
        index = row * cols + col;
        return true;
    }

    private string GetRoomRawCoverageTileName(int index)
    {
        int cols = Mathf.Max(1, roomRawCoverageTileColumns);
        int rows = Mathf.Max(1, roomRawCoverageTileRows);
        int row = Mathf.Clamp(index / cols, 0, rows - 1);
        int col = Mathf.Clamp(index % cols, 0, cols - 1);
        string vertical = row == rows - 1 ? "top" : row == 0 ? "bottom" : "middle";
        string horizontal = col == 0 ? "left" : col == cols - 1 ? "right" : col == cols / 2 ? "center" : col < cols / 2 ? "left-center" : "right-center";
        return vertical + "-" + horizontal;
    }


    private string BuildRoomRawCoverageMissingHint()
    {
        EnsureRoomRawCoverageTileStats();
        int[] worst = { -1, -1, -1 };
        float[] worstProgress = { 2f, 2f, 2f };
        for (int i = 0; i < _roomRawCoverageTileSamples.Length; i++)
        {
            float progress = GetRoomRawCoverageTileProgress01(i);
            if (progress >= 0.95f)
                continue;

            for (int slot = 0; slot < worst.Length; slot++)
            {
                if (progress >= worstProgress[slot])
                    continue;

                for (int shift = worst.Length - 1; shift > slot; shift--)
                {
                    worst[shift] = worst[shift - 1];
                    worstProgress[shift] = worstProgress[shift - 1];
                }
                worst[slot] = i;
                worstProgress[slot] = progress;
                break;
            }
        }

        if (worst[0] < 0)
            return "none";

        StringBuilder builder = new StringBuilder(96);
        for (int i = 0; i < worst.Length; i++)
        {
            if (worst[i] < 0)
                continue;
            if (builder.Length > 0)
                builder.Append(", ");
            builder.Append(GetRoomRawCoverageTileName(worst[i]))
                .Append(' ')
                .Append(Mathf.RoundToInt(Mathf.Clamp01(worstProgress[i]) * 100f))
                .Append('%');
        }
        return builder.ToString();
    }


    private float GetRoomRawCoverageLocalProgress01()
    {
        float frames = GetRoomRawCoverageWindowFrames() / (float)Mathf.Max(1, roomRawCoverageLocalTargetFrames);
        float samples = GetRoomRawCoverageWindowValidSamples() / (float)Mathf.Max(1, roomRawCoverageLocalTargetValidSamples);
        float cover = GetRoomRawCoverageWindowCoveredVoxels() / (float)Mathf.Max(1, roomRawCoverageLocalTargetVoxels);
        float stable = GetRoomRawCoverageWindowStableVoxels() / (float)Mathf.Max(1, roomRawCoverageLocalTargetStableVoxels);
        return Mathf.Clamp01(Mathf.Min(Mathf.Min(frames, samples), Mathf.Min(cover, stable)));
    }

    private RoomRawCoverageViewMeter BuildRoomRawCoverageViewMeter()
    {
        RoomRawCoverageViewMeter meter = default;
        Transform pose = ResolvePoseTransform();
        if (pose == null)
            return meter;

        float farDepth = Mathf.Max(roomRawCoverageMinDepthMeters + 0.1f, roomRawCoverageViewMeterFarDepthMeters);
        foreach (KeyValuePair<Vector3Int, RoomRawCoverageVoxel> pair in _roomRawCoverageVoxels)
        {
            RoomRawCoverageVoxel voxel = pair.Value;
            if (voxel.pointHits <= 0)
                continue;

            Vector3 center = GetRoomRawCoverageVoxelAverage(voxel);
            if (!IsPointInsideRoomRawCoverageViewMeter(center, pose, out float depth))
                continue;

            meter.covered++;
            if (voxel.stable)
                meter.stable++;
            if (voxel.frameHits <= 1)
                meter.singleHit++;
            if (!voxel.stable || voxel.frameHits < Mathf.Max(1, roomRawCoverageStableHitTarget))
                meter.needsRevisit++;
            if (depth >= farDepth)
            {
                meter.farCovered++;
                if (voxel.stable)
                    meter.farStable++;
            }
        }

        foreach (KeyValuePair<Vector3Int, RoomRawDepthCompletionVoxel> pair in _roomRawDepthCompletionVoxels)
        {
            RoomRawDepthCompletionVoxel voxel = pair.Value;
            if (voxel == null || voxel.hits <= 0)
                continue;

            Vector3 center = GetRoomRawDepthCompletionDisplayPosition(pair.Key, voxel);
            if (!IsPointInsideRoomRawCoverageViewMeter(center, pose, out _))
                continue;

            RoomRawDepthCompletionState state = ClassifyRoomRawDepthCompletionVoxel(pair.Key, voxel);
            if (state == RoomRawDepthCompletionState.Stable)
                meter.completionStable++;
            else if (state == RoomRawDepthCompletionState.Risk || state == RoomRawDepthCompletionState.Conflict)
                meter.completionRisk++;
            else if (state == RoomRawDepthCompletionState.Scanning || state == RoomRawDepthCompletionState.Supported)
                meter.completionCandidates++;
        }

        return meter;
    }

    private bool IsPointInsideRoomRawCoverageViewMeter(Vector3 point, Transform pose, out float depth)
    {
        depth = 0f;
        if (pose == null || !IsFinite(point))
            return false;

        Vector3 toPoint = point - pose.position;
        depth = Vector3.Dot(toPoint, pose.forward);
        float minDepth = Mathf.Max(0.01f, roomRawCoverageMinDepthMeters);
        float maxDepth = Mathf.Max(minDepth + 0.01f, roomRawCoverageMaxDepthMeters);
        if (!float.IsFinite(depth) || depth < minDepth || depth > maxDepth)
            return false;

        if (toPoint.sqrMagnitude <= 1e-8f)
            return false;

        float angle = Vector3.Angle(pose.forward, toPoint);
        return angle <= Mathf.Clamp(roomRawCoverageViewMeterHalfAngleDegrees, 5f, 90f);
    }

    private void UpdateHud()
    {
        if (!showHud)
        {
            SetHudVisible(false);
            return;
        }

        EnsureHud();
        PositionHud();
        if (_hudText == null || Time.unscaledTime < _nextHudRefreshTime)
            return;

        _nextHudRefreshTime = Time.unscaledTime + Mathf.Max(0.05f, hudRefreshIntervalSeconds);
        RefreshHudText();
    }

    private void EnsureHud()
    {
        if (_hudCanvas != null && _hudText != null)
        {
            SetHudVisible(true);
            return;
        }

        GameObject root = new GameObject("[ScanCover] MultiFrameCaptureHUD", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler));
        root.hideFlags = HideFlags.DontSave;
        _hudRoot = root;
        _hudCanvas = root.GetComponent<Canvas>();
        _hudCanvas.renderMode = RenderMode.WorldSpace;
        _hudCanvas.worldCamera = captureCamera;

        CanvasScaler scaler = root.GetComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;
        scaler.referencePixelsPerUnit = 100f;

        GameObject panelObject = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelObject.transform.SetParent(root.transform, false);
        RectTransform panel = panelObject.GetComponent<RectTransform>();
        panel.sizeDelta = new Vector2(hudPanelWidth, hudPanelHeight);
        panel.anchorMin = new Vector2(0.5f, 0.5f);
        panel.anchorMax = new Vector2(0.5f, 0.5f);
        panel.pivot = new Vector2(0.5f, 0.5f);
        _hudPanelImage = panelObject.GetComponent<Image>();
        _hudPanelImage.color = hudPanelColor;
        _hudPanelImage.raycastTarget = false;

        GameObject textObject = new GameObject("Text", typeof(RectTransform), typeof(Text));
        textObject.transform.SetParent(panelObject.transform, false);
        RectTransform textRect = textObject.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(18f, 14f);
        textRect.offsetMax = new Vector2(-18f, -14f);

        _hudText = textObject.GetComponent<Text>();
        _hudText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (_hudText.font == null)
            _hudText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        _hudText.fontSize = 15;
        _hudText.alignment = TextAnchor.UpperLeft;
        _hudText.horizontalOverflow = HorizontalWrapMode.Wrap;
        _hudText.verticalOverflow = VerticalWrapMode.Overflow;
        _hudText.color = GetHudTextColor();
        _hudText.raycastTarget = false;
        _hudText.text = "ScanCover HUD";

        SetHudVisible(true);
    }

    private void PositionHud()
    {
        if (_hudCanvas == null)
            return;

        Transform parent = ResolvePoseTransform();
        if (parent == null)
            parent = transform;

        _hudCanvas.worldCamera = captureCamera;
        Transform hudTransform = _hudCanvas.transform;
        if (hudTransform.parent != parent)
            hudTransform.SetParent(parent, false);
        hudTransform.localPosition = hudLocalPosition;
        hudTransform.localEulerAngles = hudLocalEuler;
        hudTransform.localScale = hudLocalScale;
    }

    private void UpdateRoomRawCoverageViewFrame()
    {
        bool shouldShow = showRoomRawCoverageViewFrame && trackRoomRawCoverage && ShouldUseRoomRawCoverageHudOnlyCapture();
        if (!shouldShow)
        {
            SetRoomRawCoverageViewFrameVisible(false);
            SetRoomRawCoverageReticleVisible(false);
            return;
        }

        EnsureRoomRawCoverageViewFrame();
        PositionRoomRawCoverageViewFrame();
        SetRoomRawCoverageViewFrameVisible(true);
        UpdateRoomRawCoverageViewFrameColor();
        UpdateRoomRawCoverageTileHintLines();
        UpdateRoomRawCoverageReticle();
    }

    private void EnsureRoomRawCoverageViewFrame()
    {
        if (_roomRawCoverageViewFrameRoot != null && _roomRawCoverageViewFrameLine != null)
            return;

        _roomRawCoverageViewFrameRoot = new GameObject("[ScanCover] RoomRawCoverageViewFrame");
        _roomRawCoverageViewFrameRoot.hideFlags = HideFlags.DontSave;

        _roomRawCoverageViewFrameLine = _roomRawCoverageViewFrameRoot.AddComponent<LineRenderer>();
        _roomRawCoverageViewFrameLine.useWorldSpace = false;
        _roomRawCoverageViewFrameLine.loop = false;
        _roomRawCoverageViewFrameLine.positionCount = 5;
        _roomRawCoverageViewFrameLine.numCornerVertices = 2;
        _roomRawCoverageViewFrameLine.numCapVertices = 2;
        _roomRawCoverageViewFrameLine.textureMode = LineTextureMode.Stretch;
        _roomRawCoverageViewFrameLine.alignment = LineAlignment.View;

        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null)
            shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader != null)
        {
            _roomRawCoverageViewFrameMaterial = new Material(shader);
            _roomRawCoverageViewFrameMaterial.hideFlags = HideFlags.DontSave;
            _roomRawCoverageViewFrameLine.sharedMaterial = _roomRawCoverageViewFrameMaterial;
        }

        EnsureRoomRawCoverageTileLines();
        EnsureRoomRawCoverageReticle();
    }

    private void PositionRoomRawCoverageViewFrame()
    {
        if (_roomRawCoverageViewFrameRoot == null || _roomRawCoverageViewFrameLine == null)
            return;

        Transform parent = ResolvePoseTransform();
        if (parent == null)
            parent = transform;

        Transform frameTransform = _roomRawCoverageViewFrameRoot.transform;
        if (_roomRawCoverageViewFrameLocked && freezeRoomRawCoverageViewFrameWhileRecording)
        {
            if (frameTransform.parent != null)
                frameTransform.SetParent(null, true);
            frameTransform.SetPositionAndRotation(_roomRawCoverageViewFrameWorldPosition, _roomRawCoverageViewFrameWorldRotation);
            frameTransform.localScale = Vector3.one;
        }
        else
        {
            if (frameTransform.parent != parent)
                frameTransform.SetParent(parent, false);
            frameTransform.localPosition = roomRawCoverageViewFrameLocalPosition;
            frameTransform.localRotation = Quaternion.identity;
            frameTransform.localScale = Vector3.one;
        }

        float halfWidth = Mathf.Max(0.05f, roomRawCoverageViewFrameWidth) * 0.5f;
        float halfHeight = Mathf.Max(0.05f, roomRawCoverageViewFrameHeight) * 0.5f;
        _roomRawCoverageViewFrameLine.widthMultiplier = Mathf.Max(0.001f, roomRawCoverageViewFrameLineWidth);
        _roomRawCoverageViewFrameLine.SetPosition(0, new Vector3(-halfWidth, -halfHeight, 0f));
        _roomRawCoverageViewFrameLine.SetPosition(1, new Vector3(-halfWidth, halfHeight, 0f));
        _roomRawCoverageViewFrameLine.SetPosition(2, new Vector3(halfWidth, halfHeight, 0f));
        _roomRawCoverageViewFrameLine.SetPosition(3, new Vector3(halfWidth, -halfHeight, 0f));
        _roomRawCoverageViewFrameLine.SetPosition(4, new Vector3(-halfWidth, -halfHeight, 0f));
    }

    private void UpdateRoomRawCoverageViewFrameColor()
    {
        if (_roomRawCoverageViewFrameLine == null)
            return;

        Color color = roomRawCoverageViewFrameIdleColor;
        if (_isRecording && _roomRawCoverageWindowActive)
        {
            bool inside = true;
            if (lockRoomRawCoverageToStartView)
            {
                Transform pose = ResolvePoseTransform();
                inside = IsWithinRoomRawCoverageAnchor(pose, out _, out _);
            }
            color = inside ? GetRoomRawCoverageFrameProgressColor() : roomRawCoverageViewFrameOutOfAnchorColor;
        }

        _roomRawCoverageViewFrameLine.startColor = color;
        _roomRawCoverageViewFrameLine.endColor = color;
        if (_roomRawCoverageViewFrameMaterial != null)
            _roomRawCoverageViewFrameMaterial.color = color;
    }

    private Color GetRoomRawCoverageFrameProgressColor()
    {
        if (!_roomRawCoverageWindowActive)
            return roomRawCoverageViewFrameIdleColor;

        float progress = GetRoomRawCoverageLocalProgress01();
        if (progress >= 0.95f)
            return roomRawCoverageTileCompleteColor;
        if (progress >= 0.45f)
            return roomRawCoverageTilePartialColor;
        return roomRawCoverageTileMissingColor;
    }

    private void EnsureRoomRawCoverageTileLines()
    {
        int count = GetRoomRawCoverageTileCount();
        if (_roomRawCoverageTileLines != null &&
            _roomRawCoverageTileMaterials != null &&
            _roomRawCoverageTileLines.Length == count &&
            _roomRawCoverageTileMaterials.Length == count)
            return;

        DestroyRoomRawCoverageTileLines();
        _roomRawCoverageTileLines = new LineRenderer[count];
        _roomRawCoverageTileMaterials = new Material[count];

        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null)
            shader = Shader.Find("Universal Render Pipeline/Unlit");

        for (int i = 0; i < count; i++)
        {
            GameObject tile = new GameObject($"Tile_{i:00}");
            tile.hideFlags = HideFlags.DontSave;
            tile.transform.SetParent(_roomRawCoverageViewFrameRoot.transform, false);

            LineRenderer line = tile.AddComponent<LineRenderer>();
            line.useWorldSpace = false;
            line.loop = false;
            line.positionCount = 5;
            line.numCornerVertices = 1;
            line.numCapVertices = 1;
            line.textureMode = LineTextureMode.Stretch;
            line.alignment = LineAlignment.View;
            if (shader != null)
            {
                Material material = new Material(shader);
                material.hideFlags = HideFlags.DontSave;
                line.sharedMaterial = material;
                _roomRawCoverageTileMaterials[i] = material;
            }
            _roomRawCoverageTileLines[i] = line;
        }
    }

    private void UpdateRoomRawCoverageTileHintLines()
    {
        if (!showRoomRawCoverageTileHints || _roomRawCoverageViewFrameRoot == null)
        {
            SetRoomRawCoverageTileHintsVisible(false);
            return;
        }

        EnsureRoomRawCoverageTileStats();
        EnsureRoomRawCoverageTileLines();
        bool visible = _isRecording && _roomRawCoverageWindowActive && _roomRawCoverageViewFrameLocked;
        SetRoomRawCoverageTileHintsVisible(visible);
        if (!visible)
            return;

        int cols = Mathf.Max(1, roomRawCoverageTileColumns);
        int rows = Mathf.Max(1, roomRawCoverageTileRows);
        float halfWidth = Mathf.Max(0.05f, roomRawCoverageViewFrameWidth) * 0.5f;
        float halfHeight = Mathf.Max(0.05f, roomRawCoverageViewFrameHeight) * 0.5f;
        float cellWidth = (halfWidth * 2f) / cols;
        float cellHeight = (halfHeight * 2f) / rows;
        bool hasAimTile = TryGetRoomRawCoverageAimTileIndex(out int aimTileIndex);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                int index = row * cols + col;
                if (index < 0 || index >= _roomRawCoverageTileLines.Length || _roomRawCoverageTileLines[index] == null)
                    continue;

                float x0 = -halfWidth + col * cellWidth;
                float x1 = x0 + cellWidth;
                float y0 = -halfHeight + row * cellHeight;
                float y1 = y0 + cellHeight;
                LineRenderer line = _roomRawCoverageTileLines[index];
                bool aimTile = hasAimTile && index == aimTileIndex;
                float width = Mathf.Max(0.001f, roomRawCoverageTileLineWidth);
                if (aimTile)
                    width *= Mathf.Max(1f, roomRawCoverageAimTileLineWidthMultiplier);
                line.widthMultiplier = width;
                line.SetPosition(0, new Vector3(x0, y0, 0.001f));
                line.SetPosition(1, new Vector3(x0, y1, 0.001f));
                line.SetPosition(2, new Vector3(x1, y1, 0.001f));
                line.SetPosition(3, new Vector3(x1, y0, 0.001f));
                line.SetPosition(4, new Vector3(x0, y0, 0.001f));

                Color color = GetRoomRawCoverageTileColor(index);
                if (aimTile)
                    color = Color.Lerp(color, roomRawCoverageReticleColor, 0.45f);
                line.startColor = color;
                line.endColor = color;
                if (index < _roomRawCoverageTileMaterials.Length && _roomRawCoverageTileMaterials[index] != null)
                    _roomRawCoverageTileMaterials[index].color = color;
            }
        }
    }

    private Color GetRoomRawCoverageTileColor(int index)
    {
        float progress = GetRoomRawCoverageTileProgress01(index);
        if (progress >= 0.95f)
            return roomRawCoverageTileCompleteColor;
        if (progress >= 0.45f)
            return roomRawCoverageTilePartialColor;
        return roomRawCoverageTileMissingColor;
    }

    private void SetRoomRawCoverageTileHintsVisible(bool visible)
    {
        if (_roomRawCoverageTileLines == null)
            return;
        for (int i = 0; i < _roomRawCoverageTileLines.Length; i++)
        {
            if (_roomRawCoverageTileLines[i] != null)
                _roomRawCoverageTileLines[i].enabled = visible;
        }
    }

    private void EnsureRoomRawCoverageReticle()
    {
        if (_roomRawCoverageReticleRoot != null)
            return;

        _roomRawCoverageReticleRoot = new GameObject("AimReticle");
        _roomRawCoverageReticleRoot.hideFlags = HideFlags.DontSave;

        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null)
            shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader != null)
        {
            _roomRawCoverageReticleMaterial = new Material(shader);
            _roomRawCoverageReticleMaterial.hideFlags = HideFlags.DontSave;
        }

        _roomRawCoverageReticleHorizontalLine = CreateRoomRawCoverageReticleLine("Horizontal");
        _roomRawCoverageReticleVerticalLine = CreateRoomRawCoverageReticleLine("Vertical");
    }

    private LineRenderer CreateRoomRawCoverageReticleLine(string name)
    {
        GameObject lineObject = new GameObject(name);
        lineObject.hideFlags = HideFlags.DontSave;
        lineObject.transform.SetParent(_roomRawCoverageReticleRoot.transform, false);

        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.useWorldSpace = false;
        line.loop = false;
        line.positionCount = 2;
        line.numCornerVertices = 1;
        line.numCapVertices = 1;
        line.textureMode = LineTextureMode.Stretch;
        line.alignment = LineAlignment.View;
        if (_roomRawCoverageReticleMaterial != null)
            line.sharedMaterial = _roomRawCoverageReticleMaterial;
        return line;
    }

    private void UpdateRoomRawCoverageReticle()
    {
        if (!showRoomRawCoverageReticle)
        {
            SetRoomRawCoverageReticleVisible(false);
            return;
        }

        EnsureRoomRawCoverageReticle();
        bool visible = _isRecording && _roomRawCoverageWindowActive && _roomRawCoverageViewFrameLocked;
        SetRoomRawCoverageReticleVisible(visible);
        if (!visible || _roomRawCoverageReticleHorizontalLine == null || _roomRawCoverageReticleVerticalLine == null)
            return;

        PositionRoomRawCoverageReticle();

        float size = Mathf.Max(0.005f, roomRawCoverageReticleSize);
        float width = Mathf.Max(0.001f, roomRawCoverageReticleLineWidth);
        _roomRawCoverageReticleHorizontalLine.widthMultiplier = width;
        _roomRawCoverageReticleVerticalLine.widthMultiplier = width;
        _roomRawCoverageReticleHorizontalLine.SetPosition(0, new Vector3(-size, 0f, 0.002f));
        _roomRawCoverageReticleHorizontalLine.SetPosition(1, new Vector3(size, 0f, 0.002f));
        _roomRawCoverageReticleVerticalLine.SetPosition(0, new Vector3(0f, -size, 0.002f));
        _roomRawCoverageReticleVerticalLine.SetPosition(1, new Vector3(0f, size, 0.002f));

        Color color = roomRawCoverageReticleColor;
        if (!_roomRawCoverageWindowActive)
            color = roomRawCoverageViewFrameIdleColor;
        else if (GetRoomRawCoverageLocalProgress01() >= 0.95f)
            color = roomRawCoverageTileCompleteColor;
        if (_roomRawCoverageReticleMaterial != null)
            _roomRawCoverageReticleMaterial.color = color;
        _roomRawCoverageReticleHorizontalLine.startColor = color;
        _roomRawCoverageReticleHorizontalLine.endColor = color;
        _roomRawCoverageReticleVerticalLine.startColor = color;
        _roomRawCoverageReticleVerticalLine.endColor = color;
    }

    private void PositionRoomRawCoverageReticle()
    {
        if (_roomRawCoverageReticleRoot == null)
            return;

        Transform parent = ResolvePoseTransform();
        if (parent == null)
            parent = transform;

        Transform reticleTransform = _roomRawCoverageReticleRoot.transform;
        if (reticleTransform.parent != parent)
            reticleTransform.SetParent(parent, false);
        reticleTransform.localPosition = roomRawCoverageViewFrameLocalPosition;
        reticleTransform.localRotation = Quaternion.identity;
        reticleTransform.localScale = Vector3.one;
    }

    private void SetRoomRawCoverageReticleVisible(bool visible)
    {
        if (_roomRawCoverageReticleHorizontalLine != null)
            _roomRawCoverageReticleHorizontalLine.enabled = visible;
        if (_roomRawCoverageReticleVerticalLine != null)
            _roomRawCoverageReticleVerticalLine.enabled = visible;
    }

    private void DestroyRoomRawCoverageReticle()
    {
        if (_roomRawCoverageReticleRoot != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawCoverageReticleRoot);
            else
                DestroyImmediate(_roomRawCoverageReticleRoot);
            _roomRawCoverageReticleRoot = null;
            _roomRawCoverageReticleHorizontalLine = null;
            _roomRawCoverageReticleVerticalLine = null;
        }

        if (_roomRawCoverageReticleMaterial != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawCoverageReticleMaterial);
            else
                DestroyImmediate(_roomRawCoverageReticleMaterial);
            _roomRawCoverageReticleMaterial = null;
        }
    }

    private void DestroyRoomRawCoverageTileLines()
    {
        if (_roomRawCoverageTileLines != null)
        {
            for (int i = 0; i < _roomRawCoverageTileLines.Length; i++)
            {
                if (_roomRawCoverageTileLines[i] == null)
                    continue;
                GameObject lineObject = _roomRawCoverageTileLines[i].gameObject;
                if (Application.isPlaying)
                    Destroy(lineObject);
                else
                    DestroyImmediate(lineObject);
            }
        }

        if (_roomRawCoverageTileMaterials != null)
        {
            for (int i = 0; i < _roomRawCoverageTileMaterials.Length; i++)
            {
                if (_roomRawCoverageTileMaterials[i] == null)
                    continue;
                if (Application.isPlaying)
                    Destroy(_roomRawCoverageTileMaterials[i]);
                else
                    DestroyImmediate(_roomRawCoverageTileMaterials[i]);
            }
        }
        _roomRawCoverageTileMaterials = new Material[0];
        _roomRawCoverageTileLines = new LineRenderer[0];
    }

    private void SetRoomRawCoverageViewFrameVisible(bool visible)
    {
        if (_roomRawCoverageViewFrameRoot != null && _roomRawCoverageViewFrameRoot.activeSelf != visible)
            _roomRawCoverageViewFrameRoot.SetActive(visible);
    }

    private void DestroyRoomRawCoverageViewFrame()
    {
        DestroyRoomRawCoverageReticle();
        DestroyRoomRawCoverageTileLines();

        if (_roomRawCoverageViewFrameMaterial != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawCoverageViewFrameMaterial);
            else
                DestroyImmediate(_roomRawCoverageViewFrameMaterial);
            _roomRawCoverageViewFrameMaterial = null;
        }

        if (_roomRawCoverageViewFrameRoot == null)
            return;

        if (Application.isPlaying)
            Destroy(_roomRawCoverageViewFrameRoot);
        else
            DestroyImmediate(_roomRawCoverageViewFrameRoot);
        _roomRawCoverageViewFrameRoot = null;
        _roomRawCoverageViewFrameLine = null;
    }

    private void UpdateRoomRawCoveragePreviewPoints()
    {
        bool shouldShow = showRoomRawCoveragePreviewPoints &&
            _isRecording &&
            trackRoomRawCoverage &&
            ShouldUseRoomRawCoverageHudOnlyCapture() &&
            _roomRawCoverageViewFrameLocked;
        if (!shouldShow)
        {
            SetRoomRawCoveragePreviewVisible(false);
            return;
        }

        EnsureRoomRawCoveragePreview();
        SetRoomRawCoveragePreviewVisible(true);
        if (Time.unscaledTime < _nextRoomRawCoveragePreviewRefreshTime)
            return;

        _nextRoomRawCoveragePreviewRefreshTime = Time.unscaledTime + Mathf.Max(0.05f, roomRawCoveragePreviewRefreshSeconds);
        RebuildRoomRawCoveragePreviewMeshes();
    }

    private void EnsureRoomRawCoveragePreview()
    {
        if (_roomRawCoveragePreviewRoot != null)
            return;

        _roomRawCoveragePreviewRoot = new GameObject("[ScanCover] RoomRawCoveragePreviewPoints");
        _roomRawCoveragePreviewRoot.hideFlags = HideFlags.DontSave;

        Color[] colors =
        {
            roomRawCoveragePreviewUniformColor,
            roomRawCoveragePreviewUniformColor,
            roomRawCoveragePreviewUniformColor
        };
        string[] names = { "Stable", "Risk", "Unstable" };
        for (int i = 0; i < _roomRawCoveragePreviewMeshes.Length; i++)
        {
            GameObject layer = new GameObject(names[i], typeof(MeshFilter), typeof(MeshRenderer));
            layer.hideFlags = HideFlags.DontSave;
            layer.transform.SetParent(_roomRawCoveragePreviewRoot.transform, false);

            _roomRawCoveragePreviewMeshes[i] = new Mesh { name = $"RoomRawCoveragePreview_{names[i]}" };
            _roomRawCoveragePreviewMeshes[i].hideFlags = HideFlags.DontSave;
            _roomRawCoveragePreviewMeshFilters[i] = layer.GetComponent<MeshFilter>();
            _roomRawCoveragePreviewMeshFilters[i].sharedMesh = _roomRawCoveragePreviewMeshes[i];
            _roomRawCoveragePreviewMeshRenderers[i] = layer.GetComponent<MeshRenderer>();

            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");
            if (shader != null)
            {
                _roomRawCoveragePreviewMaterials[i] = new Material(shader);
                _roomRawCoveragePreviewMaterials[i].hideFlags = HideFlags.DontSave;
                _roomRawCoveragePreviewMaterials[i].color = colors[i];
                _roomRawCoveragePreviewMeshRenderers[i].sharedMaterial = _roomRawCoveragePreviewMaterials[i];
            }
        }
    }

    private void RebuildRoomRawCoveragePreviewMeshes()
    {
        if (_roomRawCoveragePreviewRoot == null)
            return;

        Transform pose = ResolvePoseTransform();
        Vector3 right = pose != null ? pose.right : Vector3.right;
        Vector3 up = pose != null ? pose.up : Vector3.up;
        float size = Mathf.Max(0.002f, roomRawCoveragePreviewPointSize);
        int maxPoints = Mathf.Max(32, roomRawCoveragePreviewMaxPoints);
        int stableBudget = Mathf.Max(1, Mathf.RoundToInt(maxPoints * 0.55f));
        int riskBudget = Mathf.Max(1, Mathf.RoundToInt(maxPoints * 0.20f));
        int unstableBudget = Mathf.Max(1, maxPoints - stableBudget - riskBudget);

        List<Vector3>[] vertices =
        {
            new List<Vector3>(stableBudget * 4),
            new List<Vector3>(riskBudget * 4),
            new List<Vector3>(unstableBudget * 4)
        };
        List<int>[] triangles =
        {
            new List<int>(stableBudget * 6),
            new List<int>(riskBudget * 6),
            new List<int>(unstableBudget * 6)
        };
        int[] counts = { 0, 0, 0 };
        int[] budgets = { stableBudget, riskBudget, unstableBudget };
        int[] eligibleTotals = { 0, 0, 0 };

        foreach (KeyValuePair<Vector3Int, RoomRawCoverageVoxel> pair in _roomRawCoverageVoxels)
        {
            RoomRawCoverageVoxel voxel = pair.Value;
            if (voxel.pointHits <= 0)
                continue;

            Vector3 center = voxel.positionSum / Mathf.Max(1, voxel.pointHits);
            if (!IsPointInsideLockedRoomRawCoverageFrame(center))
                continue;
            int layer = voxel.stable && !voxel.risk ? 0 : voxel.risk ? 1 : 2;
            eligibleTotals[layer]++;
        }

        int[] seenByLayer = { 0, 0, 0 };
        int[] strides =
        {
            Mathf.Max(1, eligibleTotals[0] / Mathf.Max(1, stableBudget)),
            Mathf.Max(1, eligibleTotals[1] / Mathf.Max(1, riskBudget)),
            Mathf.Max(1, eligibleTotals[2] / Mathf.Max(1, unstableBudget))
        };

        foreach (KeyValuePair<Vector3Int, RoomRawCoverageVoxel> pair in _roomRawCoverageVoxels)
        {
            RoomRawCoverageVoxel voxel = pair.Value;
            int layer = voxel.stable && !voxel.risk ? 0 : voxel.risk ? 1 : 2;
            if (counts[layer] >= budgets[layer] || voxel.pointHits <= 0)
                continue;

            Vector3 center = voxel.positionSum / Mathf.Max(1, voxel.pointHits);
            if (!IsPointInsideLockedRoomRawCoverageFrame(center))
                continue;
            if (seenByLayer[layer]++ % strides[layer] != 0)
                continue;

            AddPreviewQuad(vertices[layer], triangles[layer], center, right, up, size);
            counts[layer]++;

            if (counts[0] >= budgets[0] && counts[1] >= budgets[1] && counts[2] >= budgets[2])
                break;
        }

        for (int i = 0; i < _roomRawCoveragePreviewMeshes.Length; i++)
            ApplyPreviewMesh(_roomRawCoveragePreviewMeshes[i], vertices[i], triangles[i]);
    }

    private bool IsPointInsideLockedRoomRawCoverageFrame(Vector3 point)
    {
        if (!_roomRawCoverageViewFrameLocked || !_roomRawCoverageAnchorSet)
            return true;

        Vector3 ray = point - _roomRawCoverageAnchorPosition;
        if (ray.sqrMagnitude <= 1e-8f)
            return false;

        Vector3 frameForward = _roomRawCoverageViewFrameWorldRotation * Vector3.forward;
        float forwardDepth = Vector3.Dot(ray, frameForward);
        if (forwardDepth <= Mathf.Max(0.01f, roomRawCoverageMinDepthMeters) ||
            forwardDepth > Mathf.Max(roomRawCoverageMinDepthMeters + 0.01f, roomRawCoverageMaxDepthMeters))
            return false;

        Plane framePlane = new Plane(frameForward, _roomRawCoverageViewFrameWorldPosition);
        Ray anchorRay = new Ray(_roomRawCoverageAnchorPosition, ray.normalized);
        if (!framePlane.Raycast(anchorRay, out float enter) || enter <= 0f)
            return false;

        Vector3 hit = anchorRay.GetPoint(enter);
        Vector3 localHit = Quaternion.Inverse(_roomRawCoverageViewFrameWorldRotation) * (hit - _roomRawCoverageViewFrameWorldPosition);
        float halfWidth = Mathf.Max(0.05f, roomRawCoverageViewFrameWidth) * 0.5f;
        float halfHeight = Mathf.Max(0.05f, roomRawCoverageViewFrameHeight) * 0.5f;
        return Mathf.Abs(localHit.x) <= halfWidth &&
               Mathf.Abs(localHit.y) <= halfHeight;
    }

    private static void AddPreviewQuad(List<Vector3> vertices, List<int> triangles, Vector3 center, Vector3 right, Vector3 up, float size)
    {
        int start = vertices.Count;
        Vector3 r = right.normalized * size;
        Vector3 u = up.normalized * size;
        vertices.Add(center - r - u);
        vertices.Add(center - r + u);
        vertices.Add(center + r + u);
        vertices.Add(center + r - u);
        triangles.Add(start);
        triangles.Add(start + 1);
        triangles.Add(start + 2);
        triangles.Add(start);
        triangles.Add(start + 2);
        triangles.Add(start + 3);
    }

    private static void ApplyPreviewMesh(Mesh mesh, List<Vector3> vertices, List<int> triangles)
    {
        if (mesh == null)
            return;

        mesh.Clear();
        if (vertices.Count <= 0)
            return;

        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateBounds();
    }

    private void ClearRoomRawCoveragePreviewMeshes()
    {
        for (int i = 0; i < _roomRawCoveragePreviewMeshes.Length; i++)
        {
            if (_roomRawCoveragePreviewMeshes[i] != null)
                _roomRawCoveragePreviewMeshes[i].Clear();
        }
    }

    private void SetRoomRawCoveragePreviewVisible(bool visible)
    {
        if (_roomRawCoveragePreviewRoot != null && _roomRawCoveragePreviewRoot.activeSelf != visible)
            _roomRawCoveragePreviewRoot.SetActive(visible);
    }

    private void DestroyRoomRawCoveragePreview()
    {
        for (int i = 0; i < _roomRawCoveragePreviewMaterials.Length; i++)
        {
            if (_roomRawCoveragePreviewMaterials[i] != null)
            {
                if (Application.isPlaying)
                    Destroy(_roomRawCoveragePreviewMaterials[i]);
                else
                    DestroyImmediate(_roomRawCoveragePreviewMaterials[i]);
                _roomRawCoveragePreviewMaterials[i] = null;
            }

            if (_roomRawCoveragePreviewMeshes[i] != null)
            {
                if (Application.isPlaying)
                    Destroy(_roomRawCoveragePreviewMeshes[i]);
                else
                    DestroyImmediate(_roomRawCoveragePreviewMeshes[i]);
                _roomRawCoveragePreviewMeshes[i] = null;
            }
            _roomRawCoveragePreviewMeshFilters[i] = null;
            _roomRawCoveragePreviewMeshRenderers[i] = null;
        }

        if (_roomRawCoveragePreviewRoot == null)
            return;

        if (Application.isPlaying)
            Destroy(_roomRawCoveragePreviewRoot);
        else
            DestroyImmediate(_roomRawCoveragePreviewRoot);
        _roomRawCoveragePreviewRoot = null;
    }

    private void RefreshHudText()
    {
        if (_hudText == null)
            return;

        if (questCloneContinuousCaptureMode)
        {
            RefreshQuestCloneCaptureHudText();
            return;
        }

        string state = _isRecording ? "RECORDING" : "IDLE";
        if (_pendingCapture)
            state = "SAMPLING";
        if (ShouldUseRoomRawCoverageHudOnlyCapture() && _isRecording && !_roomRawCoverageWindowActive)
            state = "PAUSED: outside capture view";

        _hudText.color = GetHudTextColor();
        if (_hudPanelImage != null)
            _hudPanelImage.color = hudPanelColor;

        float nextInterval = Mathf.Max(0f, minCaptureIntervalSeconds - (Time.unscaledTime - _lastCaptureTime));
        string directory = string.IsNullOrEmpty(_sessionDirectory) ? "no session" : _sessionDirectory;
        if (directory.Length > 52)
            directory = "..." + directory.Substring(directory.Length - 49);

        RoomRawCoverageViewMeter viewMeter = BuildRoomRawCoverageViewMeter();
        StringBuilder builder = new StringBuilder(1400);
        builder.Append("ScanCover Capture  ").Append(state).AppendLine();
        if (enableOvrStartStopInput)
            builder.Append("B: start/stop capture");
        else
            builder.Append("B: disabled");
        if (enableOvrCaptureNowInput)
            builder.Append(useRightIndexTriggerForCaptureNow ? "  Trigger: snapshot" : "  A: capture one frame");
        builder.Append("  F9/F10: editor").AppendLine();
        builder.Append("Frames: ").Append(_capturedFrameCount).Append('/').Append(maxFramesPerSession)
            .Append("  DepthFrame: ").Append(depthGridPointCloud != null ? depthGridPointCloud.CurrentSurfaceMeshFrameIndex : -1).AppendLine();
        AppendMeshChainDiagnosticLine(builder);

        string captureMode = ShouldUseRoomRawCoverageHudOnlyCapture()
            ? "view-window raw depth"
            : ShouldUseLegacyFullViewRoomRawCoverage()
                ? "full-view raw depth"
                : "mesh export";
        builder.Append("Mode: ").Append(captureMode)
            .Append("  RawDepth: ").Append(exportRawDepthProbe ? "on" : "off")
            .Append("  Stats: ").Append(exportObservationStats ? "on" : "off")
            .Append("  Fuse: ").Append(roomRawCoverageDisorderFuseMode.ToString()).AppendLine();

        if (trackRoomRawCoverage)
        {
            string areaHint = GetLegacyRoomRawCoverageHint();
            if (!string.IsNullOrEmpty(areaHint))
                builder.Append("Hint: ").Append(areaHint).AppendLine();

            builder.Append("Depth: ")
                .Append(FormatFloat(roomRawCoverageMinDepthMeters)).Append('-')
                .Append(FormatFloat(roomRawCoverageMaxDepthMeters)).Append("m").AppendLine();

            builder.Append("Coverage frames/samples/covered/stable: ")
                .Append(_roomRawCoverageFrames).Append('/')
                .Append(_roomRawCoverageValidSamples).Append('/')
                .Append(_roomRawCoverageCoveredVoxels).Append('/')
                .Append(_roomRawCoverageStableVoxels).AppendLine();

            builder.Append("Last raw valid/new/stable/risk: ")
                .Append(_lastRoomRawValidSamples).Append('/')
                .Append(_lastRoomRawNewVoxels).Append('/')
                .Append(_lastRoomRawStableNewVoxels).Append('/')
                .Append(FormatFloat(_lastRoomRawRiskRatio)).AppendLine();
            builder.Append("Overlap seam ratio/agree/conflict: ")
                .Append(FormatFloat(_lastRoomRawOverlapRatio)).Append('/')
                .Append(FormatFloat(_lastRoomRawHistoryAgreementRatio)).Append('/')
                .Append(_lastRoomRawHistoryConflictVoxels).AppendLine();
            AppendRoomRawCoverageViewMeterHudLine(builder, viewMeter);

            builder.Append("ObservationOrderScore: ")
                .Append(FormatFloat(_lastObservationOrderScore.score01))
                .Append(" raw=").Append(FormatFloat(_lastObservationOrderScore.rawOrderRatio))
                .Append(" badEdges=").Append(_lastObservationOrderScore.rawBadEdges).Append('/').Append(_lastObservationOrderScore.rawTestedEdges)
                .Append(" support=").Append(_lastObservationOrderScore.support)
                .Append(" useful=").Append(_lastObservationOrderScore.useful)
                .Append(" state=").Append(_lastObservationOrderScore.representationState)
                .Append(" reason=").Append(_lastObservationOrderScore.reason ?? "none")
                .AppendLine();

            int riskVoxelTarget = Mathf.Max(1, Mathf.RoundToInt(roomRawCoverageTargetStableVoxels * Mathf.Clamp01(roomRawCoverageTargetRiskRatio)));
            AppendProgressLine(builder, "stable voxels", _roomRawCoverageStableVoxels, roomRawCoverageTargetStableVoxels);
            AppendProgressLine(builder, "high stable", _roomRawCoverageHighStableVoxels, roomRawCoverageTargetHighStableVoxels);
            AppendProgressLine(builder, "low stable", _roomRawCoverageLowStableVoxels, roomRawCoverageTargetLowStableVoxels);
            AppendProgressLine(builder, "risk voxels", _roomRawCoverageRiskVoxels, riskVoxelTarget);
        }

        if (exportRoomRawDepthFrames)
        {
            builder.Append("Raw depth frames/samples/skipped: ")
                .Append(_roomRawDepthExportedFrames).Append('/')
                .Append(_roomRawDepthExportedSamples).Append('/')
                .Append(_roomRawDepthSkippedFrames).AppendLine();
        }

        AppendSnapshotGridMaskHudLine(builder);
        AppendRoomRawDepthSnapshotHudLine(builder);
        AppendBinocularRoomRawDepthSnapshotHudLine(builder);
        AppendRoomRawDepthCompletionHudLine(builder);

        builder.Append("Next: ").Append(FormatFloat(nextInterval)).Append("s").AppendLine();
        builder.Append("Dir: ").Append(directory);

        _hudText.text = builder.ToString();
    }

    /// <summary>
    /// 一瞥定位"扫描在跑但无网格"：A 链（积分/GPU三角形/间接渲染状态）+
    /// B 链（积分/顶点/线索引/双眼配对 + 各眼 issue）。只读、判空即略。
    /// </summary>
    private void AppendMeshChainDiagnosticLine(StringBuilder builder)
    {
        if (Time.unscaledTime >= _nextDiagLookupTime)
        {
            _nextDiagLookupTime = Time.unscaledTime + 2f;
            if (_diagTsdfBranch == null)
                _diagTsdfBranch = FindFirstObjectByType<ScanCoverTsdfBranch>(FindObjectsInactive.Include);
            if (_diagQuestRoomPipeline == null)
                _diagQuestRoomPipeline = FindFirstObjectByType<ScanCoverQuestRoomSurfaceNetsPipeline>(FindObjectsInactive.Include);
        }

        if (_diagTsdfBranch == null && _diagQuestRoomPipeline == null)
            return;

        builder.Append("网格 A[");
        if (_diagTsdfBranch != null)
        {
            builder.Append("积分").Append(_diagTsdfBranch.IntegrationCount)
                .Append(" 三角").Append(_diagTsdfBranch.LastGpuDrawIndexCount)
                .Append(" 间接").Append(_diagTsdfBranch.useIndirectRendering ? "开" : "关");
            if (_diagTsdfBranch.IndirectRenderingSuspect)
                builder.Append("(空→已回退CPU)");
            string gpuIssue = _diagTsdfBranch.GpuRendererIssue;
            if (!string.IsNullOrEmpty(gpuIssue))
                builder.Append(' ').Append(gpuIssue);
        }
        else
        {
            builder.Append("无");
        }

        builder.Append("] B[");
        if (_diagQuestRoomPipeline != null)
        {
            builder.Append("积分").Append(_diagQuestRoomPipeline.IntegrationCount)
                .Append(" 顶点").Append(_diagQuestRoomPipeline.LastVertexCount)
                .Append(" 线").Append(_diagQuestRoomPipeline.LastLineIndexCount)
                .Append(" 双眼").Append(_diagQuestRoomPipeline.StereoPairIntegrationCount)
                .Append('/').Append(_diagQuestRoomPipeline.PartialStereoIntegrationCount);
            if (!string.IsNullOrEmpty(_diagQuestRoomPipeline.LastIssue))
                builder.Append(' ').Append(_diagQuestRoomPipeline.LastIssue);
            if (_diagQuestRoomPipeline.LastVertexCount == 0)
            {
                string topReject = _diagQuestRoomPipeline.GetTopRejectReason();
                if (topReject != null)
                    builder.Append(" 拒:").Append(topReject);
                builder.Append(" 预L[").Append(_diagQuestRoomPipeline.GetLeftPrepStatsCompact())
                    .Append("] 预R[").Append(_diagQuestRoomPipeline.GetRightPrepStatsCompact())
                    .Append(']');
            }
        }
        else
        {
            builder.Append("无");
        }

        builder.Append(']').AppendLine();
    }

    private void RefreshQuestCloneCaptureHudText()
    {
        string state = _isRecording ? (_pendingCapture ? "SAMPLING" : "RECORDING") : "READY";
        _hudText.color = questCloneHudColor;
        if (_hudPanelImage != null)
            _hudPanelImage.color = hudPanelColor;

        int risk = _roomRawDepthCompletionYellowCount + _roomRawDepthCompletionRedCount;
        BuildQuestCloneViewGuidance(out int viewTotal, out int viewStable, out string viewHint);
        int viewPercent = Mathf.RoundToInt(viewStable * 100f / Mathf.Max(1, viewTotal));
        StringBuilder builder = new StringBuilder(320);
        builder.Append("QUEST CLONE CAPTURE  ").Append(state).AppendLine();
        builder.Append("VIEW: ").Append(viewHint)
            .Append("  COMPLETE ").Append(viewPercent).Append('%')
            .Append(" (").Append(viewStable).Append('/').Append(viewTotal).AppendLine(")");
        builder.Append("FRAMES: ").Append(_capturedFrameCount).Append('/').Append(maxFramesPerSession)
            .Append("  EYE DELTA: ").Append(_questCloneLastInterEyeDeltaMs.ToString("0.0", CultureInfo.InvariantCulture)).AppendLine("ms");
        builder.AppendLine("MAP: GRAY TODO | PURPLE ANGLE | CYAN DONE | MAGENTA BAD");
        builder.Append("CELLS D/A/B/T: ")
            .Append(_roomRawDepthCompletionGreenCount).Append('/')
            .Append(_roomRawDepthCompletionBlueCount).Append('/')
            .Append(risk).Append('/')
            .Append(_roomRawDepthCompletionGrayCount).AppendLine();
        builder.Append("WRITER active/drop: ")
            .Append(Volatile.Read(ref _questCloneWritesInFlight)).Append('/')
            .Append(Volatile.Read(ref _questCloneDroppedWrites));
        if (!string.IsNullOrEmpty(_questCloneLastWriteError))
            builder.Append("  ERR: ").Append(_questCloneLastWriteError);
        _hudText.text = builder.ToString();
    }

    private void BuildQuestCloneViewGuidance(out int viewTotal, out int viewStable, out string hint)
    {
        viewTotal = 0;
        viewStable = 0;
        int viewPending = 0;
        int viewSupported = 0;
        int viewRisk = 0;
        Transform pose = ResolvePoseTransform();
        if (pose == null || _questCloneGuideVoxels.Count <= 0)
        {
            hint = "LOOK AROUND TO DISCOVER SURFACES";
            return;
        }

        int stride = Mathf.Max(1, Mathf.CeilToInt(_questCloneGuideVoxels.Count / 10000f));
        int index = 0;
        bool hasOutsideTarget = false;
        Vector3 bestOutsideTarget = Vector3.zero;
        float bestOutsideScore = float.PositiveInfinity;
        foreach (KeyValuePair<Vector3Int, Vector3> guide in _questCloneGuideVoxels)
        {
            if ((index++ % stride) != 0)
                continue;

            _roomRawDepthCompletionVoxels.TryGetValue(guide.Key, out RoomRawDepthCompletionVoxel voxel);
            RoomRawDepthCompletionState state = ClassifyRoomRawDepthCompletionVoxel(guide.Key, voxel);
            Vector3 position = voxel != null
                ? GetRoomRawDepthCompletionDisplayPosition(guide.Key, voxel)
                : GetQuestCloneGuideDisplayPosition(guide.Key);
            bool inView = IsPointInsideRoomRawCoverageViewMeter(position, pose, out _);
            if (inView)
            {
                viewTotal++;
                switch (state)
                {
                    case RoomRawDepthCompletionState.Stable:
                        viewStable++;
                        break;
                    case RoomRawDepthCompletionState.Supported:
                        viewSupported++;
                        break;
                    case RoomRawDepthCompletionState.Risk:
                    case RoomRawDepthCompletionState.Conflict:
                        viewRisk++;
                        break;
                    default:
                        viewPending++;
                        break;
                }
            }
            else if (state != RoomRawDepthCompletionState.Stable)
            {
                Vector3 toTarget = position - pose.position;
                if (toTarget.sqrMagnitude <= 1e-6f)
                    continue;
                float angle = Vector3.Angle(pose.forward, toTarget);
                float score = angle + Mathf.Sqrt(toTarget.sqrMagnitude) * 2f;
                if (score < bestOutsideScore)
                {
                    bestOutsideScore = score;
                    bestOutsideTarget = position;
                    hasOutsideTarget = true;
                }
            }
        }

        if (viewTotal <= 0)
        {
            hint = hasOutsideTarget ? GetQuestCloneTurnHint(pose, bestOutsideTarget) : "LOOK AROUND TO DISCOVER SURFACES";
            return;
        }
        if (viewPending > 0)
            hint = "HOLD ON GRAY CELLS";
        else if (viewRisk > Mathf.Max(2, viewTotal / 5))
            hint = "STEP SIDEWAYS FROM MAGENTA";
        else if (viewSupported > 0)
            hint = "CHANGE ANGLE ON PURPLE";
        else if (viewStable * 100 < viewTotal * 80)
            hint = "SWEEP THIS VIEW SLOWLY";
        else
            hint = hasOutsideTarget ? GetQuestCloneTurnHint(pose, bestOutsideTarget) : "VIEW COMPLETE - DISCOVER MORE";
    }

    private static string GetQuestCloneTurnHint(Transform pose, Vector3 target)
    {
        Vector3 local = pose.InverseTransformPoint(target);
        if (local.z < 0f && Mathf.Abs(local.z) > Mathf.Abs(local.x))
            return "TURN AROUND TO PENDING AREA";
        if (Mathf.Abs(local.x) > Mathf.Abs(local.y) * 0.8f)
            return local.x >= 0f ? "TURN RIGHT TO PENDING AREA" : "TURN LEFT TO PENDING AREA";
        return local.y >= 0f ? "LOOK UP TO PENDING AREA" : "LOOK DOWN TO PENDING AREA";
    }


    private void AppendRoomRawCoverageViewMeterHudLine(StringBuilder builder, RoomRawCoverageViewMeter meter)
    {
        int minVoxels = Mathf.Max(1, roomRawCoverageViewMeterMinVoxels);
        int stablePercent = Mathf.RoundToInt(meter.StableRatio * 100f);
        int farStablePercent = Mathf.RoundToInt(meter.FarStableRatio * 100f);
        int riskPercent = Mathf.RoundToInt(meter.CompletionRiskRatio * 100f);
        bool enoughSamples = meter.covered >= minVoxels;
        bool farEnough = meter.farCovered <= minVoxels / 3 || meter.FarStableRatio >= 0.50f;
        bool areaComplete = enoughSamples && meter.StableRatio >= 0.75f && farEnough;
        bool riskDominant = meter.completionRisk >= 16 &&
            meter.CompletionRiskRatio >= 0.35f &&
            meter.completionRisk > meter.completionStable;
        string hint;
        if (!enoughSamples)
            hint = "样本偏少，慢扫当前方向";
        else if (areaComplete)
            hint = "当前区域基本完成，可转向";
        else if (meter.farCovered > minVoxels / 3 && meter.FarStableRatio < 0.45f)
            hint = $"远处需重访，侧移后再扫一次（远处{farStablePercent}%）";
        else if (meter.needsRevisit > Mathf.Max(meter.stable, minVoxels / 2))
            hint = "重访不足，当前方向再停留或侧移";
        else if (meter.completionCandidates > Mathf.Max(meter.completionStable, minVoxels / 2))
            hint = "候选偏多，继续停留到变绿";
        else if (riskDominant)
            hint = $"风险偏高，换一点角度（风险{riskPercent}%）";
        else
            hint = "当前区域可用，继续平滑扫过";

        builder.Append("区域提示: ")
            .Append(hint)
            .Append("  稳定").Append(stablePercent).Append('%')
            .AppendLine();
    }

    private void RefreshCompactRoomRawCoverageHudText(string state)
    {
        if (_hudText == null)
            return;

        _hudText.color = GetHudTextColor();
        if (_hudPanelImage != null)
            _hudPanelImage.color = hudPanelColor;

        float nextInterval = Mathf.Max(0f, minCaptureIntervalSeconds - (Time.unscaledTime - _lastCaptureTime));
        string directory = string.IsNullOrEmpty(_sessionDirectory) ? "no session" : _sessionDirectory;
        if (directory.Length > 52)
            directory = "..." + directory.Substring(directory.Length - 49);

        EnsureRoomRawCoverageTileStats();
        _roomRawCoverageMissingHint = BuildRoomRawCoverageMissingHint();
        int tileCount = GetRoomRawCoverageTileCount();
        int completeTiles = GetRoomRawCoverageCompleteTileCount();
        bool hasAimTile = TryGetRoomRawCoverageAimTileIndex(out int aimTile);
        int localProgress = Mathf.RoundToInt(GetRoomRawCoverageLocalProgress01() * 100f);
        int aimProgress = hasAimTile ? Mathf.RoundToInt(GetRoomRawCoverageTileProgress01(aimTile) * 100f) : 0;

        RoomRawCoverageViewMeter viewMeter = BuildRoomRawCoverageViewMeter();
        StringBuilder builder = new StringBuilder(1400);
        builder.Append("ScanCover Raw Coverage  ").Append(state).AppendLine();
        builder.Append("B: start/stop capture  App exit: auto save  Fuse: ")
            .Append(roomRawCoverageDisorderFuseMode.ToString()).AppendLine();
        builder.Append("View: ")
            .Append(_roomRawCoverageViewFrameLocked ? "locked" : "head-follow")
            .Append("  Capture: ")
            .Append(_roomRawCoverageWindowActive ? "on" : "paused")
            .Append("  Next: ")
            .Append(FormatFloat(nextInterval)).Append('s')
            .AppendLine();

        builder.Append("Local ").Append(localProgress).Append("%  Tiles ")
            .Append(completeTiles).Append('/').Append(tileCount)
            .Append("  Frames ")
            .Append(GetRoomRawCoverageWindowFrames()).Append('/').Append(Mathf.Max(1, roomRawCoverageLocalTargetFrames))
            .AppendLine();
        builder.Append("OrderScore ")
            .Append(FormatFloat(_lastObservationOrderScore.score01))
            .Append(" raw ").Append(FormatFloat(_lastObservationOrderScore.rawOrderRatio))
            .Append("  ").Append(_lastObservationOrderScore.reason ?? "none")
            .AppendLine();
        builder.Append("Crosshair: ").Append(hasAimTile ? GetRoomRawCoverageTileName(aimTile) : "outside view");
        if (hasAimTile)
            builder.Append(' ').Append(aimProgress).Append('%');
        builder.AppendLine();
        builder.Append("Covered ")
            .Append(GetRoomRawCoverageWindowCoveredVoxels()).Append('/').Append(Mathf.Max(1, roomRawCoverageLocalTargetVoxels))
            .Append("  Stable ")
            .Append(GetRoomRawCoverageWindowStableVoxels()).Append('/').Append(Mathf.Max(1, roomRawCoverageLocalTargetStableVoxels))
            .AppendLine();
        builder.Append("Session ")
            .Append(_roomRawCoverageCoveredVoxels).Append('/')
            .Append(_roomRawCoverageStableVoxels)
            .Append("  Overlap ")
            .Append(FormatFloat(_lastRoomRawOverlapRatio))
            .Append(" agree ")
            .Append(FormatFloat(_lastRoomRawHistoryAgreementRatio))
            .Append(" new ")
            .Append(_lastRoomRawNewVoxels)
            .AppendLine();
        AppendRoomRawCoverageViewMeterHudLine(builder, viewMeter);
        builder.Append("Missing: ").Append(_roomRawCoverageMissingHint).AppendLine();
        builder.Append("Memory covered/stable: ")
            .Append(_loadedRoomRawCoverageVoxels).Append('/')
            .Append(_loadedRoomRawCoverageStableVoxels);
        if (_loadedRoomRawCoverageSessions > 0)
            builder.Append("  Memory: ").Append(_loadedRoomRawCoverageSessions).Append(" sessions");
        builder.AppendLine();
        builder.Append("Status: ").Append(string.IsNullOrEmpty(_lastRoomRawStatus) ? "none" : _lastRoomRawStatus).AppendLine();
        if (!string.IsNullOrEmpty(_lastRoomRawFuseStatus))
            builder.Append(_lastRoomRawFuseStatus).AppendLine();

        AppendSnapshotGridMaskHudLine(builder);
        AppendRoomRawDepthSnapshotHudLine(builder);
        AppendBinocularRoomRawDepthSnapshotHudLine(builder);
        AppendRoomRawDepthCompletionHudLine(builder);
        builder.Append("Dir: ").Append(directory);
        _hudText.text = builder.ToString();
    }


    private static void AppendProgressLine(StringBuilder builder, string label, int value, int target)
    {
        target = Mathf.Max(1, target);
        int clamped = Mathf.Clamp(value, 0, target);
        int width = 10;
        int filled = Mathf.RoundToInt(width * (clamped / (float)target));
        builder.Append(label.PadRight(12)).Append(" [");
        for (int i = 0; i < width; i++)
            builder.Append(i < filled ? '#' : '-');
        builder.Append("] ").Append(value).Append('/').Append(target).AppendLine();
    }

    private void SetHudVisible(bool visible)
    {
        if (_hudRoot != null && _hudRoot.activeSelf != visible)
            _hudRoot.SetActive(visible);
    }

    private Color GetHudTextColor()
    {
        if (_pendingCapture)
            return hudSamplingTextColor;
        if (ShouldUseRoomRawCoverageHudOnlyCapture() && _isRecording && !_roomRawCoverageWindowActive)
            return hudIdleTextColor;
        if (_isRecording)
            return hudRecordingTextColor;
        return hudIdleTextColor;
    }

    private bool ShouldEnableSessionFileExport()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        return true;
#elif UNITY_EDITOR
        return enableEditorLocalSessionExport;
#else
        return false;
#endif
    }

    private string ResolveSessionExportRoot()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        return Path.Combine(Application.persistentDataPath, "ScanCoverExports");
#else
        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName ?? Application.dataPath;
        string root = string.IsNullOrWhiteSpace(editorLocalSessionExportRoot) ? "ScanCoverExports" : editorLocalSessionExportRoot.Trim();
        return Path.IsPathRooted(root) ? root : Path.Combine(projectRoot, root);
#endif
    }

    private void ClearSessionExportPaths()
    {
        _sessionDirectory = string.Empty;
        _framesDirectory = string.Empty;
        _manifestPath = string.Empty;
        _observationDirectory = string.Empty;
        _frameObservationStatsPath = string.Empty;
        _distanceBinsPath = string.Empty;
        _angleBinsPath = string.Empty;
        _edgeRiskStatsPath = string.Empty;
        _targetGatePath = string.Empty;
        _quest3BadnessDirectory = string.Empty;
        _rawDepthBadnessFramesPath = string.Empty;
        _rawDepthHoleComponentsPath = string.Empty;
        _rawDepthBadnessSummaryPath = string.Empty;
        _repeatCoverageDirectory = string.Empty;
        _repeatCoverageGatePath = string.Empty;
        _repeatCoverageVoxelsPath = string.Empty;
        _roomRawCoverageDirectory = string.Empty;
        _roomRawCoverageFramesPath = string.Empty;
        _roomRawCoverageVoxelsPath = string.Empty;
        _roomRawCoverageSummaryPath = string.Empty;
        _roomRawDepthDirectory = string.Empty;
        _roomRawDepthManifestPath = string.Empty;
        _roomRawDepthSummaryPath = string.Empty;
        _roomRawDepthSnapshotDirectory = string.Empty;
        _roomRawDepthSnapshotManifestPath = string.Empty;
        _virtualCloneInputDirectory = string.Empty;
        _virtualCloneInputManifestPath = string.Empty;
        _questCloneBinaryDirectory = string.Empty;
        _questCloneBinaryManifestPath = string.Empty;
    }

    private void WriteQuestCloneBinaryHeader()
    {
        if (!_sessionFileExportEnabled || !questCloneContinuousCaptureMode || !questCloneUseCompactBinary ||
            string.IsNullOrEmpty(_questCloneBinaryManifestPath))
            return;

        File.WriteAllText(_questCloneBinaryManifestPath,
            "# ScanCover Quest3 degradation-model capture. Target/action labels guide collection and never gate frames.\n" +
            "frame,rightFrame,leftFrame,rightDispatchSeconds,leftDispatchSeconds,interEyeDeltaMs,target,action,rightBinary,leftBinary,metadataJson,status\n",
            Encoding.UTF8);
    }

    private void WriteManifestHeader()
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(_manifestPath))
            return;

        StringBuilder builder = new StringBuilder(512);
        builder.AppendLine("# ScanCover Quest3 clone capture session");
        builder.AppendLine("# data groups: pose_camera, bl_mesh_raw_depth, multi_frame_stability, observation_stats");
        builder.AppendLine("frame,reason,time,unscaledTime,unityFrame,depthGridFrame,poseX,poseY,poseZ,rotX,rotY,rotZ,obj,verticesCsv,trianglesCsv,cameraJson,rawDepthCsv,rawDepthJson,gridStateCsv");
        File.WriteAllText(_manifestPath, builder.ToString(), Encoding.UTF8);
    }

    private void WriteObservationStatsHeaders()
    {
        if (!_sessionFileExportEnabled || !exportObservationStats || string.IsNullOrEmpty(_observationDirectory))
            return;

        File.WriteAllText(_frameObservationStatsPath,
            "# ScanCover per-frame observation summary\n" +
            "frame,pointCount,triangleCount,boundaryRiskCount,creaseRiskCount,anyRiskCount,avgViewDepth,avgEuclideanDistance,avgViewAngleDeg,avgFaceNormalMaxAngleDeg,minViewDepth,maxViewDepth,minDistance,maxDistance,pointsCsv\n",
            Encoding.UTF8);

        File.WriteAllText(_distanceBinsPath,
            "# ScanCover distance-binned observation summary\n" +
            "frame,bin,count,avgViewDepth,avgEuclideanDistance,avgViewAngleDeg,boundaryRiskCount,creaseRiskCount,anyRiskCount\n",
            Encoding.UTF8);

        File.WriteAllText(_angleBinsPath,
            "# ScanCover view-angle-binned observation summary\n" +
            "frame,bin,count,avgViewDepth,avgEuclideanDistance,avgViewAngleDeg,boundaryRiskCount,creaseRiskCount,anyRiskCount\n",
            Encoding.UTF8);

        File.WriteAllText(_edgeRiskStatsPath,
            "# ScanCover edge/risk observation summary\n" +
            "frame,total,boundaryRiskCount,creaseRiskCount,anyRiskCount,boundaryRiskRatio,creaseRiskRatio,anyRiskRatio,creaseAngleThresholdDeg\n",
            Encoding.UTF8);

        File.WriteAllText(_targetGatePath,
            "# ScanCover targeted observation gate\n" +
            "frame,accepted,farSurfaceQualified,highAngleRiskQualified,nearEdgeRiskQualified,farSurfaceCount,highAngleRiskCount,nearEdgeRiskCount,targetFarFrames,targetHighAngleFrames,targetNearEdgeFrames,rejectedFrames,pointCount,reason,stableMainQualified,topCoverageQualified,middleCoverageQualified,bottomCoverageQualified,nearDistanceQualified,midDistanceQualified,farDistanceQualified,frontAngleQualified,obliqueAngleQualified,extremeAngleQualified,riskLayerQualified,mainFrontQualified,mainLeftObliqueQualified,mainRightObliqueQualified,mainUpObliqueQualified,mainDownObliqueQualified,boundaryLeftQualified,boundaryRightQualified,cornerJunctionQualified,nearEdgeSupplementRiskQualified,nearEdgeSupplementCreaseQualified,nearEdgeSupplementBoundaryQualified,nearEdgeSupplementLeftQualified,nearEdgeSupplementRightQualified,nearEdgeSupplementUpQualified,nearEdgeSupplementDownQualified,nearEdgeSupplementObliqueQualified,nearEdgeSupplementExtremeQualified,stableMainCount,topCoverageCount,middleCoverageCount,bottomCoverageCount,nearDistanceCount,midDistanceCount,farDistanceCount,frontAngleCount,obliqueAngleCount,extremeAngleCount,riskLayerCount,mainFrontCount,mainLeftObliqueCount,mainRightObliqueCount,mainUpObliqueCount,mainDownObliqueCount,boundaryLeftCount,boundaryRightCount,cornerJunctionCount,nearEdgeSupplementRiskCount,nearEdgeSupplementCreaseCount,nearEdgeSupplementBoundaryCount,nearEdgeSupplementLeftCount,nearEdgeSupplementRightCount,nearEdgeSupplementUpCount,nearEdgeSupplementDownCount,nearEdgeSupplementObliqueCount,nearEdgeSupplementExtremeCount,targetStableMainFrames,targetTopCoverageFrames,targetMiddleCoverageFrames,targetBottomCoverageFrames,targetNearDistanceFrames,targetMidDistanceFrames,targetFarDistanceFrames,targetFrontAngleFrames,targetObliqueAngleFrames,targetExtremeAngleFrames,targetRiskLayerFrames,targetMainFrontFrames,targetMainLeftObliqueFrames,targetMainRightObliqueFrames,targetMainUpObliqueFrames,targetMainDownObliqueFrames,targetBoundaryLeftFrames,targetBoundaryRightFrames,targetCornerJunctionFrames,targetNearEdgeSupplementRiskFrames,targetNearEdgeSupplementCreaseFrames,targetNearEdgeSupplementBoundaryFrames,targetNearEdgeSupplementLeftFrames,targetNearEdgeSupplementRightFrames,targetNearEdgeSupplementUpFrames,targetNearEdgeSupplementDownFrames,targetNearEdgeSupplementObliqueFrames,targetNearEdgeSupplementExtremeFrames,completedCoreMetrics,completedDirectedMetrics,completedNearEdgeSupplementMetrics,coreMetricTarget,nearEdgeSupplementTarget,captureMode\n",
            Encoding.UTF8);
    }

    private void WriteQuest3BadnessHeaders()
    {
        if (!_sessionFileExportEnabled || !exportQuest3ObservationBadnessStats || string.IsNullOrEmpty(_quest3BadnessDirectory))
            return;

        File.WriteAllText(_rawDepthBadnessFramesPath,
            "# ScanCover Quest3 raw-depth observation badness summary\n" +
            "# invalid components are connected 2D valid-mask holes in the raw-depth probe, tracked per eye.\n" +
            "frame,eye,totalPixels,validPixels,invalidPixels,validRatio,invalidRatio,invalidComponentCount,largeHoleComponentCount,largestHolePixels,persistentInvalidPixels,newlyInvalidPixels,recoveredPixels,edgeRiskValidPixels,minLinearMeters,maxLinearMeters,avgLinearMeters,edgeDepthJumpMeters,holeMinPixels\n",
            Encoding.UTF8);

        File.WriteAllText(_rawDepthHoleComponentsPath,
            "# ScanCover raw-depth invalid connected components\n" +
            "frame,eye,componentId,pixelCount,minX,minY,maxX,maxY,touchesBorder,isLargeHole\n",
            Encoding.UTF8);

        File.WriteAllText(_rawDepthBadnessSummaryPath, "{\n  \"status\": \"session-active\"\n}\n", Encoding.UTF8);
    }

    private void WriteRepeatCoverageHeaders()
    {
        if (!_sessionFileExportEnabled || !gateFramesByRepeatCoverage || string.IsNullOrEmpty(_repeatCoverageDirectory))
            return;

        File.WriteAllText(_repeatCoverageGatePath,
            "# ScanCover repeat coverage / stable hit gate\n" +
            "frame,accepted,candidateVoxels,newVoxels,rehitVoxels,parallaxRehitVoxels,newStableVoxels,stableVoxelTotal,targetStableVoxels,acceptedFrames,rejectedFrames,reason\n",
            Encoding.UTF8);

        File.WriteAllText(_repeatCoverageVoxelsPath,
            "# session-active; final values are written when the session stops\n",
            Encoding.UTF8);
    }

    private void WriteRoomRawCoverageHeaders()
    {
        if (!_sessionFileExportEnabled || !trackRoomRawCoverage || string.IsNullOrEmpty(_roomRawCoverageDirectory))
            return;

        File.WriteAllText(_roomRawCoverageFramesPath,
            "# ScanCover room-scale raw-depth coverage summary. This is HUD/data only; full pixels are not visualized.\n" +
            "frame,rawDepthFrame,accepted,totalSamples,validSamples,frameVoxels,focusVoxels,coreVoxels,edgeBufferVoxels,focusNewVoxels,edgeBufferNewVoxels,newVoxels,rehitVoxels,historyRehitVoxels,parallaxRehitVoxels,newStableVoxels,highFrameVoxels,newHighVoxels,newHighStableVoxels,lowFrameVoxels,newLowVoxels,newLowStableVoxels,coveredVoxels,stableVoxels,highVoxels,highStableVoxels,lowVoxels,lowStableVoxels,riskSamples,focusRiskSamples,riskVoxels,riskRatio,orderScore,rawOrderRatio,rawCenterOrderRatio,rawBadEdgeRatio,rawBadQuadRatio,rawTestedEdges,rawBadEdges,rawTestedQuads,rawBadQuads,orderReason,poseCells,acceptedFrames,rejectedFrames,anchorAngleDeg,anchorMoveM,anchorFallback,overlapRatio,historyAgreementVoxels,historyConflictVoxels,historyAgreementRatio,sessionNewCoveredVoxels,sessionNewStableVoxels,status\n",
            Encoding.UTF8);

        File.WriteAllText(_roomRawCoverageVoxelsPath,
            "# session-active; final values are written when the session stops\n",
            Encoding.UTF8);

        File.WriteAllText(_roomRawCoverageSummaryPath, "{\n  \"status\": \"session-active\"\n}\n", Encoding.UTF8);
    }

    private bool ShouldExportRoomRawDepthFrames()
        => exportRoomRawDepthFrames;

    private bool ShouldExportRoomRawDepthSnapshots()
        => exportRoomRawDepthSnapshots;

    private bool ShouldExportVirtualCloneInputMetadata()
        => exportVirtualCloneInputMetadata && ShouldExportRoomRawDepthSnapshots() && useBinocularRoomRawDepthSnapshots;

    private void WriteRoomRawDepthHeaders()
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthFrames() || string.IsNullOrEmpty(_roomRawDepthDirectory))
            return;

        File.WriteAllText(_roomRawDepthManifestPath,
            "# ScanCover dense raw-depth frames. Per-frame CSV files keep raw pixel samples before 8cm coverage aggregation.\n" +
            "frame,rawDepthFrame,accepted,totalSamples,validSamples,exportedSamples,width,height,cameraX,cameraY,cameraZ,rotX,rotY,rotZ,path,orderScore,rawOrderRatio,rawBadEdgeRatio,rawBadEdges,rawTestedEdges,orderReason,status\n",
            Encoding.UTF8);

        File.WriteAllText(_roomRawDepthSummaryPath, "{\n  \"status\": \"session-active\"\n}\n", Encoding.UTF8);
    }

    private void WriteRoomRawDepthSnapshotHeaders()
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthSnapshots() || string.IsNullOrEmpty(_roomRawDepthSnapshotDirectory))
            return;

        File.WriteAllText(_roomRawDepthSnapshotManifestPath,
            "# ScanCover one-shot raw-depth snapshots. Each CSV keeps every raw 160x160 pixel row; invalid pixels remain rows with valid=0.\n" +
            "frame,rawDepthFrame,totalPixels,validPixels,visiblePoints,width,height,cameraX,cameraY,cameraZ,rotX,rotY,rotZ,path,status\n",
            Encoding.UTF8);
    }

    private void WriteVirtualCloneInputHeaders()
    {
        if (!_sessionFileExportEnabled || !ShouldExportVirtualCloneInputMetadata() || string.IsNullOrEmpty(_virtualCloneInputManifestPath))
            return;

        File.WriteAllText(_virtualCloneInputManifestPath,
            "# ScanCover binocular virtual-clone input. This is metadata for offline Replica/shell raycast; Raw Depth CSV remains unchanged.\n" +
            "frame,rawDepthFrame,width,height,rightStart,rightCount,leftStart,leftCount,eyeBaselineMeters,fieldOfViewDegrees,aspect,rightDispatchSeconds,leftDispatchSeconds,interEyeDeltaMs,target,action,rawSnapshotCsv,metadataJson,status\n",
            Encoding.UTF8);
    }

    private void AppendManifestRow(string frameName, string reason, string objPath, string verticesCsvPath, string trianglesCsvPath, string cameraJsonPath, string rawDepthCsvPath, string rawDepthJsonPath, string gridStateCsvPath, Transform pose)
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(_manifestPath))
            return;

        Vector3 position = pose != null ? pose.position : Vector3.zero;
        Vector3 rotation = pose != null ? pose.rotation.eulerAngles : Vector3.zero;
        StringBuilder row = new StringBuilder(512);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(EscapeCsv(reason)).Append(',')
            .Append(FormatFloat(Time.time)).Append(',')
            .Append(FormatFloat(Time.unscaledTime)).Append(',')
            .Append(Time.frameCount).Append(',')
            .Append(depthGridPointCloud != null ? depthGridPointCloud.CurrentSurfaceMeshFrameIndex : -1).Append(',')
            .Append(FormatVector(position)).Append(',')
            .Append(FormatVector(rotation)).Append(',')
            .Append(EscapeCsv(objPath)).Append(',')
            .Append(EscapeCsv(verticesCsvPath)).Append(',')
            .Append(EscapeCsv(trianglesCsvPath)).Append(',')
            .Append(EscapeCsv(cameraJsonPath)).Append(',')
            .Append(EscapeCsv(rawDepthCsvPath)).Append(',')
            .Append(EscapeCsv(rawDepthJsonPath)).Append(',')
            .Append(EscapeCsv(gridStateCsvPath)).AppendLine();
        File.AppendAllText(_manifestPath, row.ToString(), Encoding.UTF8);
    }

    private void WriteSessionInfo(string timestamp)
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(_sessionDirectory))
            return;

        string path = Path.Combine(_sessionDirectory, "session_info.json");
        Transform pose = ResolvePoseTransform();
        StringBuilder builder = new StringBuilder(1024);
        builder.AppendLine("{");
        builder.Append("  \"timestamp\": \"").Append(timestamp).AppendLine("\",");
        builder.Append("  \"questCloneContinuousCaptureMode\": ").Append(questCloneContinuousCaptureMode ? "true" : "false").AppendLine(",");
        builder.Append("  \"questCloneTarget\": \"").Append(EscapeJson(questCloneTargetLabel)).AppendLine("\",");
        builder.Append("  \"questCloneAction\": \"").Append(EscapeJson(questCloneActionLabel)).AppendLine("\",");
        builder.Append("  \"questCloneCaptureStaticFrames\": ").Append(questCloneCaptureStaticFrames ? "true" : "false").AppendLine(",");
        builder.Append("  \"depthGrid\": \"").Append(EscapeJson(depthGridPointCloud != null ? depthGridPointCloud.name : "")).AppendLine("\",");
        builder.Append("  \"camera\": \"").Append(EscapeJson(captureCamera != null ? captureCamera.name : "")).AppendLine("\",");
        builder.Append("  \"poseSource\": \"").Append(EscapeJson(pose != null ? pose.name : "")).AppendLine("\",");
        builder.Append("  \"minCaptureIntervalSeconds\": ").Append(FormatFloat(minCaptureIntervalSeconds)).AppendLine(",");
        builder.Append("  \"minMoveMeters\": ").Append(FormatFloat(minMoveMeters)).AppendLine(",");
        builder.Append("  \"minRotateDegrees\": ").Append(FormatFloat(minRotateDegrees)).AppendLine(",");
        builder.Append("  \"maxFramesPerSession\": ").Append(maxFramesPerSession.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"sessionGroupDirectoryName\": \"").Append(EscapeJson(sessionGroupDirectoryName)).AppendLine("\",");
        builder.Append("  \"sessionNamePrefix\": \"").Append(EscapeJson(sessionNamePrefix)).AppendLine("\",");
        builder.Append("  \"exportRawDepthProbe\": ").Append(exportRawDepthProbe ? "true" : "false").AppendLine(",");
        builder.Append("  \"rawDepthProbeSize\": ").Append(RawDepthProbeSize.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"exportQuest3ObservationBadnessStats\": ").Append(exportQuest3ObservationBadnessStats ? "true" : "false").AppendLine(",");
        builder.Append("  \"rawDepthHoleMinPixels\": ").Append(Mathf.Max(1, rawDepthHoleMinPixels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"rawDepthEdgeDepthJumpMeters\": ").Append(FormatFloat(Mathf.Max(0.001f, rawDepthEdgeDepthJumpMeters))).AppendLine(",");
        builder.Append("  \"targetBadnessFrames\": ").Append(Mathf.Max(1, targetBadnessFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"targetBadnessInvalidFrames\": ").Append(Mathf.Max(1, targetBadnessInvalidFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"targetBadnessLargeHoleFrames\": ").Append(Mathf.Max(1, targetBadnessLargeHoleFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"targetBadnessEdgeJumpFrames\": ").Append(Mathf.Max(1, targetBadnessEdgeJumpFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"targetBadnessPersistentInvalidFrames\": ").Append(Mathf.Max(1, targetBadnessPersistentInvalidFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minBadnessInvalidPixelsPerEye\": ").Append(Mathf.Max(1, minBadnessInvalidPixelsPerEye).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minBadnessEdgeRiskPixelsPerEye\": ").Append(Mathf.Max(1, minBadnessEdgeRiskPixelsPerEye).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"exportGridStateCsv\": ").Append(exportGridStateCsv ? "true" : "false").AppendLine(",");
        builder.Append("  \"stabilityVoxelSizeMeters\": ").Append(FormatFloat(stabilityVoxelSizeMeters)).AppendLine(",");
        builder.Append("  \"gateFramesByRepeatCoverage\": ").Append(gateFramesByRepeatCoverage ? "true" : "false").AppendLine(",");
        builder.Append("  \"repeatCoverageVoxelSizeMeters\": ").Append(FormatFloat(repeatCoverageVoxelSizeMeters)).AppendLine(",");
        builder.Append("  \"repeatCoverageMinDistinctHits\": ").Append(Mathf.Max(1, repeatCoverageMinDistinctHits).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"repeatCoverageTargetStableVoxels\": ").Append(Mathf.Max(1, repeatCoverageTargetStableVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"repeatCoverageMinCandidateVoxels\": ").Append(Mathf.Max(1, repeatCoverageMinCandidateVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"repeatCoverageMinNewStableVoxelsPerFrame\": ").Append(Mathf.Max(1, repeatCoverageMinNewStableVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"repeatCoverageMinNewOrRehitVoxelsPerFrame\": ").Append(Mathf.Max(1, repeatCoverageMinNewOrRehitVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"repeatCoverageMinParallaxMeters\": ").Append(FormatFloat(repeatCoverageMinParallaxMeters)).AppendLine(",");
        builder.Append("  \"repeatCoverageMaxViewAngleDegrees\": ").Append(FormatFloat(repeatCoverageMaxViewAngleDegrees)).AppendLine(",");
        builder.Append("  \"trackRoomRawCoverage\": ").Append(trackRoomRawCoverage ? "true" : "false").AppendLine(",");
        builder.Append("  \"roomRawCoverageHudOnlyCapture\": ").Append(roomRawCoverageHudOnlyCapture ? "true" : "false").AppendLine(",");
        builder.Append("  \"lockRoomRawCoverageToStartView\": ").Append(lockRoomRawCoverageToStartView ? "true" : "false").AppendLine(",");
        builder.Append("  \"allowSameViewStableRoomRawHits\": ").Append(allowSameViewStableRoomRawHits ? "true" : "false").AppendLine(",");
        builder.Append("  \"loadPreviousRoomRawCoverageOnStart\": ").Append(loadPreviousRoomRawCoverageOnStart ? "true" : "false").AppendLine(",");
        builder.Append("  \"maxPreviousRoomRawCoverageSessionsToLoad\": ").Append(Mathf.Max(1, maxPreviousRoomRawCoverageSessionsToLoad).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"previousRoomRawCoverageOverrideDirectory\": \"").Append(EscapeJson(previousRoomRawCoverageOverrideDirectory ?? "")).AppendLine("\",");
        builder.Append("  \"roomRawCoverageDisorderFuseMode\": \"").Append(roomRawCoverageDisorderFuseMode.ToString()).AppendLine("\",");
        builder.Append("  \"legacyRoomRawCoverageDisorderConsecutiveFrames\": ").Append(Mathf.Max(1, legacyRoomRawCoverageDisorderConsecutiveFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"legacyRoomRawCoverageHardFuseBadEdgeRatio\": ").Append(FormatFloat(Mathf.Clamp01(legacyRoomRawCoverageHardFuseBadEdgeRatio))).AppendLine(",");
        builder.Append("  \"legacyRoomRawCoverageHardFuseCenterBadEdgeRatio\": ").Append(FormatFloat(Mathf.Clamp01(legacyRoomRawCoverageHardFuseCenterBadEdgeRatio))).AppendLine(",");
        builder.Append("  \"legacyRoomRawCoverageHardFuseBadQuadRatio\": ").Append(FormatFloat(Mathf.Clamp01(legacyRoomRawCoverageHardFuseBadQuadRatio))).AppendLine(",");
        builder.Append("  \"legacyRoomRawCoverageHardFuseRiskRatio\": ").Append(FormatFloat(Mathf.Clamp01(legacyRoomRawCoverageHardFuseRiskRatio))).AppendLine(",");
        builder.Append("  \"hideDepthGridPreviewWhileRecording\": ").Append(hideDepthGridPreviewWhileRecording ? "true" : "false").AppendLine(",");
        builder.Append("  \"exportRoomRawDepthFrames\": ").Append(exportRoomRawDepthFrames ? "true" : "false").AppendLine(",");
        builder.Append("  \"exportRoomRawDepthOnlyAcceptedFrames\": ").Append(exportRoomRawDepthOnlyAcceptedFrames ? "true" : "false").AppendLine(",");
        builder.Append("  \"roomRawDepthFrameStride\": ").Append(Mathf.Max(1, roomRawDepthFrameStride).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawDepthFocusOnly\": ").Append(roomRawDepthFocusOnly ? "true" : "false").AppendLine(",");
        builder.Append("  \"roomRawDepthMaxSamplesPerFrame\": ").Append(Mathf.Max(0, roomRawDepthMaxSamplesPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"exportRoomRawDepthSnapshots\": ").Append(exportRoomRawDepthSnapshots ? "true" : "false").AppendLine(",");
        builder.Append("  \"showRoomRawDepthSnapshotOverlay\": ").Append(showRoomRawDepthSnapshotOverlay ? "true" : "false").AppendLine(",");
        builder.Append("  \"roomRawDepthSnapshotPointSize\": ").Append(FormatFloat(roomRawDepthSnapshotPointSize)).AppendLine(",");
        builder.Append("  \"roomRawDepthSnapshotMaxVisualPoints\": ").Append(Mathf.Max(1024, roomRawDepthSnapshotMaxVisualPoints).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageVoxelSizeMeters\": ").Append(FormatFloat(roomRawCoverageVoxelSizeMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetVoxels\": ").Append(Mathf.Max(1, roomRawCoverageTargetVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetStableVoxels\": ").Append(Mathf.Max(1, roomRawCoverageTargetStableVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetHighVoxels\": ").Append(Mathf.Max(1, roomRawCoverageTargetHighVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetHighStableVoxels\": ").Append(Mathf.Max(1, roomRawCoverageTargetHighStableVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetLowVoxels\": ").Append(Mathf.Max(1, roomRawCoverageTargetLowVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetLowStableVoxels\": ").Append(Mathf.Max(1, roomRawCoverageTargetLowStableVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageStableHitTarget\": ").Append(Mathf.Max(1, roomRawCoverageStableHitTarget).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetPoseCells\": ").Append(Mathf.Max(1, roomRawCoverageTargetPoseCells).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetValidSamples\": ").Append(Mathf.Max(1, roomRawCoverageTargetValidSamples).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetFrames\": ").Append(Mathf.Max(1, roomRawCoverageTargetFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageLocalTargetVoxels\": ").Append(Mathf.Max(1, roomRawCoverageLocalTargetVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageLocalTargetStableVoxels\": ").Append(Mathf.Max(1, roomRawCoverageLocalTargetStableVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageLocalTargetValidSamples\": ").Append(Mathf.Max(1, roomRawCoverageLocalTargetValidSamples).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageLocalTargetFrames\": ").Append(Mathf.Max(1, roomRawCoverageLocalTargetFrames).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageStableParallaxMeters\": ").Append(FormatFloat(roomRawCoverageStableParallaxMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoveragePoseCellSizeMeters\": ").Append(FormatFloat(roomRawCoveragePoseCellSizeMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageRiskViewAngleDegrees\": ").Append(FormatFloat(roomRawCoverageRiskViewAngleDegrees)).AppendLine(",");
        builder.Append("  \"roomRawCoverageHighPointMinDeltaYMeters\": ").Append(FormatFloat(roomRawCoverageHighPointMinDeltaYMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageLowPointMinDeltaYMeters\": ").Append(FormatFloat(roomRawCoverageLowPointMinDeltaYMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageNeighborDepthJumpMeters\": ").Append(FormatFloat(roomRawCoverageNeighborDepthJumpMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageTargetRiskRatio\": ").Append(FormatFloat(roomRawCoverageTargetRiskRatio)).AppendLine(",");
        builder.Append("  \"gateRoomRawCoverageFrames\": ").Append(gateRoomRawCoverageFrames ? "true" : "false").AppendLine(",");
        builder.Append("  \"roomRawCoverageFocusMargin\": ").Append(FormatFloat(roomRawCoverageFocusMargin)).AppendLine(",");
        builder.Append("  \"roomRawCoverageUseFallbackTolerance\": ").Append(roomRawCoverageUseFallbackTolerance ? "true" : "false").AppendLine(",");
        builder.Append("  \"roomRawCoverageCoreMargin\": ").Append(FormatFloat(roomRawCoverageCoreMargin)).AppendLine(",");
        builder.Append("  \"roomRawCoverageEdgeBufferMargin\": ").Append(FormatFloat(roomRawCoverageEdgeBufferMargin)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMaxAnchorAngleDegrees\": ").Append(FormatFloat(roomRawCoverageMaxAnchorAngleDegrees)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMaxAnchorMoveMeters\": ").Append(FormatFloat(roomRawCoverageMaxAnchorMoveMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageAnchorAngleFallbackDegrees\": ").Append(FormatFloat(roomRawCoverageAnchorAngleFallbackDegrees)).AppendLine(",");
        builder.Append("  \"roomRawCoverageAnchorMoveFallbackMeters\": ").Append(FormatFloat(roomRawCoverageAnchorMoveFallbackMeters)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinFocusVoxels\": ").Append(Mathf.Max(1, roomRawCoverageMinFocusVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinCoreVoxels\": ").Append(Mathf.Max(1, roomRawCoverageMinCoreVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinEdgeBufferVoxels\": ").Append(Mathf.Max(1, roomRawCoverageMinEdgeBufferVoxels).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinNewVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinNewVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinOverlapNewVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinOverlapNewVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinNewStableVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinNewStableVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinNewHighLowVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinNewHighLowVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinParallaxRehitVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinParallaxRehitVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinOverlapRehitVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinOverlapRehitVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinHistoryRehitVoxelsPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinHistoryRehitVoxelsPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinRiskSamplesPerFrame\": ").Append(Mathf.Max(1, roomRawCoverageMinRiskSamplesPerFrame).ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"roomRawCoverageIdealOverlapMinRatio\": ").Append(FormatFloat(Mathf.Clamp01(roomRawCoverageIdealOverlapMinRatio))).AppendLine(",");
        builder.Append("  \"roomRawCoverageIdealOverlapMaxRatio\": ").Append(FormatFloat(Mathf.Clamp01(roomRawCoverageIdealOverlapMaxRatio))).AppendLine(",");
        builder.Append("  \"roomRawCoverageMinHistoryAgreementRatio\": ").Append(FormatFloat(Mathf.Clamp01(roomRawCoverageMinHistoryAgreementRatio))).AppendLine(",");
        builder.Append("  \"exportObservationStats\": ").Append(exportObservationStats ? "true" : "false").AppendLine(",");
        builder.Append("  \"meshCreaseRiskAngleDegrees\": ").Append(FormatFloat(meshCreaseRiskAngleDegrees)).AppendLine(",");
        builder.Append("  \"gateFramesByObservationTargets\": ").Append(gateFramesByObservationTargets ? "true" : "false").AppendLine(",");
        builder.Append("  \"observationCaptureMode\": \"").Append(observationCaptureMode.ToString()).AppendLine("\",");
        builder.Append("  \"targetFramesPerObservationBucket\": ").Append(targetFramesPerObservationBucket.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"useMultiCoreObservationTargets\": ").Append(useMultiCoreObservationTargets ? "true" : "false").AppendLine(",");
        builder.Append("  \"targetFramesPerCoreMetric\": ").Append(targetFramesPerCoreMetric.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"targetNearEdgeSupplementFrames\": ").Append(targetNearEdgeSupplementFrames.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minStableMainPointsPerFrame\": ").Append(minStableMainPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minCoverageBandPointsPerFrame\": ").Append(minCoverageBandPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minTypeBucketPointsPerFrame\": ").Append(minTypeBucketPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minRiskLayerPointsPerFrame\": ").Append(minRiskLayerPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minFarSurfacePointsPerFrame\": ").Append(minFarSurfacePointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minHighAngleRiskPointsPerFrame\": ").Append(minHighAngleRiskPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minNearEdgeRiskPointsPerFrame\": ").Append(minNearEdgeRiskPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minNearEdgeSupplementRiskPointsPerFrame\": ").Append(minNearEdgeSupplementRiskPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minNearEdgeSupplementCreasePointsPerFrame\": ").Append(minNearEdgeSupplementCreasePointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minNearEdgeSupplementBoundaryPointsPerFrame\": ").Append(minNearEdgeSupplementBoundaryPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minDirectedMainPointsPerFrame\": ").Append(minDirectedMainPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minDirectionalBoundaryPointsPerFrame\": ").Append(minDirectionalBoundaryPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"minCornerJunctionPointsPerFrame\": ").Append(minCornerJunctionPointsPerFrame.ToString(CultureInfo.InvariantCulture)).AppendLine(",");
        builder.Append("  \"farSurfaceDistanceRangeMeters\": [").Append(FormatFloat(farSurfaceDistanceRangeMeters.x)).Append(", ").Append(FormatFloat(farSurfaceDistanceRangeMeters.y)).AppendLine("],");
        builder.Append("  \"midSurfaceDistanceRangeMeters\": [").Append(FormatFloat(midSurfaceDistanceRangeMeters.x)).Append(", ").Append(FormatFloat(midSurfaceDistanceRangeMeters.y)).AppendLine("],");
        builder.Append("  \"nearSurfaceDistanceRangeMeters\": [").Append(FormatFloat(nearSurfaceDistanceRangeMeters.x)).Append(", ").Append(FormatFloat(nearSurfaceDistanceRangeMeters.y)).AppendLine("],");
        builder.Append("  \"nearEdgeDistanceRangeMeters\": [").Append(FormatFloat(nearEdgeDistanceRangeMeters.x)).Append(", ").Append(FormatFloat(nearEdgeDistanceRangeMeters.y)).AppendLine("],");
        builder.Append("  \"stableMainMaxAngleDegrees\": ").Append(FormatFloat(stableMainMaxAngleDegrees)).AppendLine(",");
        builder.Append("  \"frontViewMaxAngleDegrees\": ").Append(FormatFloat(frontViewMaxAngleDegrees)).AppendLine(",");
        builder.Append("  \"obliqueViewMinAngleDegrees\": ").Append(FormatFloat(obliqueViewMinAngleDegrees)).AppendLine(",");
        builder.Append("  \"obliqueViewMaxAngleDegrees\": ").Append(FormatFloat(obliqueViewMaxAngleDegrees)).AppendLine(",");
        builder.Append("  \"nearEdgeSupplementObliqueMinAngleDegrees\": ").Append(FormatFloat(nearEdgeSupplementObliqueMinAngleDegrees)).AppendLine(",");
        builder.Append("  \"nearEdgeSupplementExtremeMinAngleDegrees\": ").Append(FormatFloat(nearEdgeSupplementExtremeMinAngleDegrees)).AppendLine(",");
        builder.Append("  \"verticalCoverageBandRatio\": ").Append(FormatFloat(verticalCoverageBandRatio)).AppendLine(",");
        builder.Append("  \"highAngleThresholdDegrees\": ").Append(FormatFloat(highAngleThresholdDegrees)).AppendLine(",");
        builder.Append("  \"directionalSideThreshold\": ").Append(FormatFloat(directionalSideThreshold)).AppendLine();
        builder.AppendLine("}");
        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private void ExportGridStateCsv(string path)
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(path) || depthGridPointCloud == null)
            return;
        if (!depthGridPointCloud.TryGetCurrentGridState(out ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot) || snapshot == null)
            return;

        StringBuilder builder = new StringBuilder(Mathf.Max(1024, snapshot.entries.Length * 128));
        builder.AppendLine("# ScanCover grid state");
        builder.Append("component=").Append(snapshot.componentName).AppendLine();
        builder.Append("samplingMode=").Append(snapshot.samplingMode).AppendLine();
        builder.Append("frameIndex=").Append(snapshot.frameIndex).AppendLine();
        builder.Append("resolution=").Append(snapshot.resolutionWidth).Append('x').Append(snapshot.resolutionHeight).AppendLine();
        builder.Append("cellCount=").Append(snapshot.cellCount).AppendLine();
        builder.Append("visibleCount=").Append(snapshot.visibleCount).AppendLine();
        builder.AppendLine("index,group,row,col,valid,worldX,worldY,worldZ,normalX,normalY,normalZ,confidence");
        for (int i = 0; i < snapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
            builder.Append(entry.index).Append(',')
                .Append(entry.group).Append(',')
                .Append(entry.row).Append(',')
                .Append(entry.col).Append(',')
                .Append(entry.valid ? 1 : 0).Append(',')
                .Append(FormatVector(entry.worldPos)).Append(',')
                .Append(FormatVector(entry.normal)).Append(',')
                .Append(FormatFloat(entry.confidence)).AppendLine();
        }
        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private ObservationExportResult ExportObservationStats(string frameDirectory, string frameName, string verticesCsvPath, string trianglesCsvPath, Transform pose, bool writeOutputs)
    {
        if (!_sessionFileExportEnabled)
            return default;

        if (string.IsNullOrEmpty(frameDirectory) || string.IsNullOrEmpty(verticesCsvPath) || !File.Exists(verticesCsvPath))
            return default;

        if (!TryReadSurfaceVertices(verticesCsvPath, out List<VertexObservation> observations))
            return default;

        List<TriangleInfo> triangles = TryReadSurfaceTriangles(trianglesCsvPath, out List<TriangleInfo> parsedTriangles)
            ? parsedTriangles
            : new List<TriangleInfo>(0);

        ApplyCameraObservationMetrics(observations, pose);
        ApplyTriangleRiskMetrics(observations, triangles);

        if (writeOutputs)
        {
            string pointsPath = Path.Combine(frameDirectory, frameName + "_observation_points.csv");
            WriteObservationPointsCsv(pointsPath, frameName, observations);
            AppendObservationSummary(frameName, observations, triangles.Count, pointsPath);
        }
        return EvaluateObservationTargets(observations);
    }

    private ObservationExportResult EvaluateObservationTargets(List<VertexObservation> observations)
    {
        ObservationExportResult result = new ObservationExportResult
        {
            hasData = observations != null && observations.Count > 0,
            pointCount = observations != null ? observations.Count : 0
        };

        if (!result.hasData)
            return result;

        float farMin = Mathf.Min(farSurfaceDistanceRangeMeters.x, farSurfaceDistanceRangeMeters.y);
        float farMax = Mathf.Max(farSurfaceDistanceRangeMeters.x, farSurfaceDistanceRangeMeters.y);
        float midMin = Mathf.Min(midSurfaceDistanceRangeMeters.x, midSurfaceDistanceRangeMeters.y);
        float midMax = Mathf.Max(midSurfaceDistanceRangeMeters.x, midSurfaceDistanceRangeMeters.y);
        float nearSurfaceMin = Mathf.Min(nearSurfaceDistanceRangeMeters.x, nearSurfaceDistanceRangeMeters.y);
        float nearSurfaceMax = Mathf.Max(nearSurfaceDistanceRangeMeters.x, nearSurfaceDistanceRangeMeters.y);
        float nearMin = Mathf.Min(nearEdgeDistanceRangeMeters.x, nearEdgeDistanceRangeMeters.y);
        float nearMax = Mathf.Max(nearEdgeDistanceRangeMeters.x, nearEdgeDistanceRangeMeters.y);
        float stableAngle = Mathf.Clamp(stableMainMaxAngleDegrees, 0f, 89f);
        float frontAngle = Mathf.Clamp(frontViewMaxAngleDegrees, 0f, 89f);
        float obliqueMin = Mathf.Clamp(Mathf.Min(obliqueViewMinAngleDegrees, obliqueViewMaxAngleDegrees), 0f, 89f);
        float obliqueMax = Mathf.Clamp(Mathf.Max(obliqueViewMinAngleDegrees, obliqueViewMaxAngleDegrees), 0f, 89f);
        float highAngle = Mathf.Clamp(highAngleThresholdDegrees, 0f, 89f);
        float supplementObliqueMin = Mathf.Clamp(nearEdgeSupplementObliqueMinAngleDegrees, 0f, 89f);
        float supplementExtremeMin = Mathf.Clamp(nearEdgeSupplementExtremeMinAngleDegrees, 0f, 89f);
        float sideThreshold = Mathf.Clamp(directionalSideThreshold, 0.05f, 0.8f);
        float minY = float.PositiveInfinity;
        float maxY = float.NegativeInfinity;

        for (int i = 0; i < observations.Count; i++)
        {
            float y = observations[i].world.y;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        float height = Mathf.Max(0.001f, maxY - minY);
        float bandRatio = Mathf.Clamp(verticalCoverageBandRatio, 0.05f, 0.45f);
        float bottomCut = minY + height * bandRatio;
        float topCut = maxY - height * bandRatio;

        for (int i = 0; i < observations.Count; i++)
        {
            VertexObservation observation = observations[i];
            bool anyRisk = observation.boundaryRisk || observation.creaseRisk;
            bool inFarRange = observation.euclideanDistance >= farMin && observation.euclideanDistance < farMax;
            bool inMidRange = observation.euclideanDistance >= midMin && observation.euclideanDistance < midMax;
            bool inNearSurfaceRange = observation.euclideanDistance >= nearSurfaceMin && observation.euclideanDistance < nearSurfaceMax;
            bool inNearRange = observation.euclideanDistance >= nearMin && observation.euclideanDistance < nearMax;
            bool stableSurface = !anyRisk;
            bool frontSurface = stableSurface && observation.viewAngleDegrees <= frontAngle;
            bool obliqueSurface = stableSurface && observation.viewAngleDegrees >= obliqueMin && observation.viewAngleDegrees < obliqueMax;
            bool nearEdgeRisk = inNearRange && anyRisk;

            if (inFarRange && !anyRisk)
                result.farSurfaceCount++;
            if (observation.viewAngleDegrees >= highAngle && anyRisk)
                result.highAngleRiskCount++;
            if (nearEdgeRisk)
                result.nearEdgeRiskCount++;

            if (stableSurface && observation.viewAngleDegrees <= stableAngle && observation.euclideanDistance >= nearSurfaceMin && observation.euclideanDistance < farMax)
                result.stableMainCount++;
            if (stableSurface && observation.world.y >= topCut)
                result.topCoverageCount++;
            if (stableSurface && observation.world.y > bottomCut && observation.world.y < topCut)
                result.middleCoverageCount++;
            if (stableSurface && observation.world.y <= bottomCut)
                result.bottomCoverageCount++;
            if (stableSurface && inNearSurfaceRange)
                result.nearDistanceCount++;
            if (stableSurface && inMidRange)
                result.midDistanceCount++;
            if (stableSurface && inFarRange)
                result.farDistanceCount++;
            if (stableSurface && observation.viewAngleDegrees <= frontAngle)
                result.frontAngleCount++;
            if (stableSurface && observation.viewAngleDegrees >= obliqueMin && observation.viewAngleDegrees < obliqueMax)
                result.obliqueAngleCount++;
            if (observation.viewAngleDegrees >= highAngle)
                result.extremeAngleCount++;
            if (anyRisk)
                result.riskLayerCount++;

            if (frontSurface)
                result.mainFrontCount++;
            if (obliqueSurface && observation.cameraRayRight <= -sideThreshold)
                result.mainLeftObliqueCount++;
            if (obliqueSurface && observation.cameraRayRight >= sideThreshold)
                result.mainRightObliqueCount++;
            if (obliqueSurface && observation.cameraRayUp >= sideThreshold)
                result.mainUpObliqueCount++;
            if (obliqueSurface && observation.cameraRayUp <= -sideThreshold)
                result.mainDownObliqueCount++;
            if (anyRisk && observation.cameraRayRight <= -sideThreshold)
                result.boundaryLeftCount++;
            if (anyRisk && observation.cameraRayRight >= sideThreshold)
                result.boundaryRightCount++;
            if (observation.creaseRisk && observation.maxFaceNormalAngleDegrees >= meshCreaseRiskAngleDegrees)
                result.cornerJunctionCount++;

            if (nearEdgeRisk)
            {
                result.nearEdgeSupplementRiskCount++;
                if (observation.creaseRisk)
                    result.nearEdgeSupplementCreaseCount++;
                if (observation.boundaryRisk)
                    result.nearEdgeSupplementBoundaryCount++;
                if (observation.cameraRayRight <= -sideThreshold)
                    result.nearEdgeSupplementLeftCount++;
                if (observation.cameraRayRight >= sideThreshold)
                    result.nearEdgeSupplementRightCount++;
                if (observation.cameraRayUp >= sideThreshold)
                    result.nearEdgeSupplementUpCount++;
                if (observation.cameraRayUp <= -sideThreshold)
                    result.nearEdgeSupplementDownCount++;
                if (observation.viewAngleDegrees >= supplementObliqueMin)
                    result.nearEdgeSupplementObliqueCount++;
                if (observation.viewAngleDegrees >= supplementExtremeMin)
                    result.nearEdgeSupplementExtremeCount++;
            }
        }

        result.farSurfaceQualified = result.farSurfaceCount >= Mathf.Max(1, minFarSurfacePointsPerFrame);
        result.highAngleRiskQualified = result.highAngleRiskCount >= Mathf.Max(1, minHighAngleRiskPointsPerFrame);
        result.nearEdgeRiskQualified = result.nearEdgeRiskCount >= Mathf.Max(1, minNearEdgeRiskPointsPerFrame);
        result.stableMainQualified = result.stableMainCount >= Mathf.Max(1, minStableMainPointsPerFrame);
        result.topCoverageQualified = result.topCoverageCount >= Mathf.Max(1, minCoverageBandPointsPerFrame);
        result.middleCoverageQualified = result.middleCoverageCount >= Mathf.Max(1, minCoverageBandPointsPerFrame);
        result.bottomCoverageQualified = result.bottomCoverageCount >= Mathf.Max(1, minCoverageBandPointsPerFrame);
        result.nearDistanceQualified = result.nearDistanceCount >= Mathf.Max(1, minTypeBucketPointsPerFrame);
        result.midDistanceQualified = result.midDistanceCount >= Mathf.Max(1, minTypeBucketPointsPerFrame);
        result.farDistanceQualified = result.farDistanceCount >= Mathf.Max(1, minTypeBucketPointsPerFrame);
        result.frontAngleQualified = result.frontAngleCount >= Mathf.Max(1, minTypeBucketPointsPerFrame);
        result.obliqueAngleQualified = result.obliqueAngleCount >= Mathf.Max(1, minTypeBucketPointsPerFrame);
        result.extremeAngleQualified = result.extremeAngleCount >= Mathf.Max(1, minTypeBucketPointsPerFrame);
        result.riskLayerQualified = result.riskLayerCount >= Mathf.Max(1, minRiskLayerPointsPerFrame);
        result.mainFrontQualified = result.mainFrontCount >= Mathf.Max(1, minDirectedMainPointsPerFrame);
        result.mainLeftObliqueQualified = result.mainLeftObliqueCount >= Mathf.Max(1, minDirectedMainPointsPerFrame);
        result.mainRightObliqueQualified = result.mainRightObliqueCount >= Mathf.Max(1, minDirectedMainPointsPerFrame);
        result.mainUpObliqueQualified = result.mainUpObliqueCount >= Mathf.Max(1, minDirectedMainPointsPerFrame);
        result.mainDownObliqueQualified = result.mainDownObliqueCount >= Mathf.Max(1, minDirectedMainPointsPerFrame);
        result.boundaryLeftQualified = result.boundaryLeftCount >= Mathf.Max(1, minDirectionalBoundaryPointsPerFrame);
        result.boundaryRightQualified = result.boundaryRightCount >= Mathf.Max(1, minDirectionalBoundaryPointsPerFrame);
        result.cornerJunctionQualified = result.cornerJunctionCount >= Mathf.Max(1, minCornerJunctionPointsPerFrame);
        int supplementRiskMin = Mathf.Max(1, minNearEdgeSupplementRiskPointsPerFrame);
        int supplementCreaseMin = Mathf.Max(1, minNearEdgeSupplementCreasePointsPerFrame);
        int supplementBoundaryMin = Mathf.Max(1, minNearEdgeSupplementBoundaryPointsPerFrame);
        int supplementSideMin = Mathf.Max(1, supplementRiskMin / 2);
        int supplementExtremeMinPoints = Mathf.Max(1, supplementRiskMin / 3);
        result.nearEdgeSupplementRiskQualified = result.nearEdgeSupplementRiskCount >= supplementRiskMin;
        result.nearEdgeSupplementCreaseQualified = result.nearEdgeSupplementCreaseCount >= supplementCreaseMin;
        result.nearEdgeSupplementBoundaryQualified = result.nearEdgeSupplementBoundaryCount >= supplementBoundaryMin;
        result.nearEdgeSupplementLeftQualified = result.nearEdgeSupplementLeftCount >= supplementSideMin;
        result.nearEdgeSupplementRightQualified = result.nearEdgeSupplementRightCount >= supplementSideMin;
        result.nearEdgeSupplementUpQualified = result.nearEdgeSupplementUpCount >= supplementSideMin;
        result.nearEdgeSupplementDownQualified = result.nearEdgeSupplementDownCount >= supplementSideMin;
        result.nearEdgeSupplementObliqueQualified = result.nearEdgeSupplementObliqueCount >= supplementSideMin;
        result.nearEdgeSupplementExtremeQualified = result.nearEdgeSupplementExtremeCount >= supplementExtremeMinPoints;
        return result;
    }

    private bool TryReadSurfaceVertices(string path, out List<VertexObservation> observations)
    {
        observations = new List<VertexObservation>(4096);
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#' || line.IndexOf('=') >= 0 || line.StartsWith("index,", StringComparison.Ordinal))
                continue;

            string[] parts = line.Split(',');
            if (parts.Length < 13 ||
                !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int index) ||
                !TryParseVector(parts, 4, out Vector3 world) ||
                !TryParseVector(parts, 10, out Vector3 normal))
                continue;

            normal = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up;
            observations.Add(new VertexObservation
            {
                index = index,
                world = world,
                normal = normal,
                viewDepth = 0f,
                euclideanDistance = 0f,
                viewAngleDegrees = 0f,
                triangleCount = 0,
                maxFaceNormalAngleDegrees = 0f,
                boundaryRisk = false,
                creaseRisk = false,
                cameraRayRight = 0f,
                cameraRayUp = 0f
            });
        }
        return observations.Count > 0;
    }

    private bool TryReadSurfaceTriangles(string path, out List<TriangleInfo> triangles)
    {
        triangles = new List<TriangleInfo>(8192);
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return false;

        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#' || line.IndexOf('=') >= 0 || line.StartsWith("triangle,", StringComparison.Ordinal))
                continue;

            string[] parts = line.Split(',');
            if (parts.Length < 14 ||
                !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int a) ||
                !int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int b) ||
                !int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out int c) ||
                !TryParseVector(parts, 7, out Vector3 normal) ||
                !float.TryParse(parts[10], NumberStyles.Float, CultureInfo.InvariantCulture, out float area))
                continue;

            float maxEdge = 0f;
            if (parts.Length >= 14 &&
                float.TryParse(parts[11], NumberStyles.Float, CultureInfo.InvariantCulture, out float edge01) &&
                float.TryParse(parts[12], NumberStyles.Float, CultureInfo.InvariantCulture, out float edge12) &&
                float.TryParse(parts[13], NumberStyles.Float, CultureInfo.InvariantCulture, out float edge20))
            {
                maxEdge = Mathf.Max(edge01, Mathf.Max(edge12, edge20));
            }

            triangles.Add(new TriangleInfo
            {
                a = a,
                b = b,
                c = c,
                normal = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up,
                area = area,
                maxEdge = maxEdge
            });
        }

        return triangles.Count > 0;
    }

    private void ApplyCameraObservationMetrics(List<VertexObservation> observations, Transform pose)
    {
        Vector3 cameraPosition = pose != null ? pose.position : captureCamera != null ? captureCamera.transform.position : Vector3.zero;
        Vector3 cameraForward = pose != null ? pose.forward : captureCamera != null ? captureCamera.transform.forward : Vector3.forward;
        Vector3 cameraRight = pose != null ? pose.right : captureCamera != null ? captureCamera.transform.right : Vector3.right;
        Vector3 cameraUp = pose != null ? pose.up : captureCamera != null ? captureCamera.transform.up : Vector3.up;
        cameraForward = cameraForward.sqrMagnitude > 1e-6f ? cameraForward.normalized : Vector3.forward;
        cameraRight = cameraRight.sqrMagnitude > 1e-6f ? cameraRight.normalized : Vector3.right;
        cameraUp = cameraUp.sqrMagnitude > 1e-6f ? cameraUp.normalized : Vector3.up;

        for (int i = 0; i < observations.Count; i++)
        {
            VertexObservation observation = observations[i];
            Vector3 cameraToPoint = observation.world - cameraPosition;
            float distance = cameraToPoint.magnitude;
            observation.euclideanDistance = distance;
            observation.viewDepth = Vector3.Dot(cameraToPoint, cameraForward);

            Vector3 rayDir = distance > 1e-6f ? cameraToPoint / distance : cameraForward;
            observation.cameraRayRight = Vector3.Dot(rayDir, cameraRight);
            observation.cameraRayUp = Vector3.Dot(rayDir, cameraUp);

            Vector3 pointToCamera = distance > 1e-6f ? -cameraToPoint / distance : cameraForward;
            float facing = Mathf.Abs(Vector3.Dot(observation.normal, pointToCamera));
            observation.viewAngleDegrees = Mathf.Acos(Mathf.Clamp(facing, -1f, 1f)) * Mathf.Rad2Deg;
            observations[i] = observation;
        }
    }

    private void ApplyTriangleRiskMetrics(List<VertexObservation> observations, List<TriangleInfo> triangles)
    {
        if (observations.Count <= 0)
            return;

        Dictionary<int, int> indexToObservation = new Dictionary<int, int>(observations.Count);
        for (int i = 0; i < observations.Count; i++)
            indexToObservation[observations[i].index] = i;

        int[] triangleCounts = new int[observations.Count];
        float[] maxAngles = new float[observations.Count];
        for (int i = 0; i < triangles.Count; i++)
        {
            TriangleInfo triangle = triangles[i];
            ApplyTriangleVertexRisk(triangle.a, triangle.normal, observations, indexToObservation, triangleCounts, maxAngles);
            ApplyTriangleVertexRisk(triangle.b, triangle.normal, observations, indexToObservation, triangleCounts, maxAngles);
            ApplyTriangleVertexRisk(triangle.c, triangle.normal, observations, indexToObservation, triangleCounts, maxAngles);
        }

        float creaseThreshold = Mathf.Max(1f, meshCreaseRiskAngleDegrees);
        for (int i = 0; i < observations.Count; i++)
        {
            VertexObservation observation = observations[i];
            observation.triangleCount = triangleCounts[i];
            observation.maxFaceNormalAngleDegrees = maxAngles[i];
            observation.boundaryRisk = triangleCounts[i] > 0 && triangleCounts[i] < 3;
            observation.creaseRisk = maxAngles[i] >= creaseThreshold;
            observations[i] = observation;
        }
    }

    private static void ApplyTriangleVertexRisk(int vertexIndex, Vector3 triangleNormal, List<VertexObservation> observations, Dictionary<int, int> indexToObservation, int[] triangleCounts, float[] maxAngles)
    {
        if (!indexToObservation.TryGetValue(vertexIndex, out int observationIndex))
            return;
        triangleCounts[observationIndex]++;
        VertexObservation observation = observations[observationIndex];
        float dot = Mathf.Clamp(Mathf.Abs(Vector3.Dot(observation.normal, triangleNormal)), -1f, 1f);
        float angle = Mathf.Acos(dot) * Mathf.Rad2Deg;
        if (angle > maxAngles[observationIndex])
            maxAngles[observationIndex] = angle;
    }

    private void WriteObservationPointsCsv(string path, string frameName, List<VertexObservation> observations)
    {
        if (!_sessionFileExportEnabled)
            return;

        StringBuilder builder = new StringBuilder(Mathf.Max(1024, observations.Count * 180));
        builder.AppendLine("# ScanCover per-vertex observation labels");
        builder.Append("frame=").Append(frameName).AppendLine();
        builder.Append("creaseRiskAngleDegrees=").Append(FormatFloat(meshCreaseRiskAngleDegrees)).AppendLine();
        builder.AppendLine("index,worldX,worldY,worldZ,normalX,normalY,normalZ,viewDepthMeters,euclideanDistanceMeters,viewAngleDegrees,cameraRayRight,cameraRayUp,triangleCount,maxFaceNormalAngleDegrees,boundaryRisk,creaseRisk,anyRisk,distanceBin,angleBin");
        for (int i = 0; i < observations.Count; i++)
        {
            VertexObservation observation = observations[i];
            bool anyRisk = observation.boundaryRisk || observation.creaseRisk;
            builder.Append(observation.index).Append(',')
                .Append(FormatVector(observation.world)).Append(',')
                .Append(FormatVector(observation.normal)).Append(',')
                .Append(FormatFloat(observation.viewDepth)).Append(',')
                .Append(FormatFloat(observation.euclideanDistance)).Append(',')
                .Append(FormatFloat(observation.viewAngleDegrees)).Append(',')
                .Append(FormatFloat(observation.cameraRayRight)).Append(',')
                .Append(FormatFloat(observation.cameraRayUp)).Append(',')
                .Append(observation.triangleCount).Append(',')
                .Append(FormatFloat(observation.maxFaceNormalAngleDegrees)).Append(',')
                .Append(observation.boundaryRisk ? 1 : 0).Append(',')
                .Append(observation.creaseRisk ? 1 : 0).Append(',')
                .Append(anyRisk ? 1 : 0).Append(',')
                .Append(DistanceBinLabel(observation.euclideanDistance)).Append(',')
                .Append(AngleBinLabel(observation.viewAngleDegrees)).AppendLine();
        }
        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private void AppendObservationSummary(string frameName, List<VertexObservation> observations, int triangleCount, string pointsPath)
    {
        if (!_sessionFileExportEnabled || observations.Count <= 0)
            return;

        int boundaryRiskCount = 0;
        int creaseRiskCount = 0;
        int anyRiskCount = 0;
        double viewDepthSum = 0.0;
        double distanceSum = 0.0;
        double angleSum = 0.0;
        double faceAngleSum = 0.0;
        float minViewDepth = float.PositiveInfinity;
        float maxViewDepth = float.NegativeInfinity;
        float minDistance = float.PositiveInfinity;
        float maxDistance = float.NegativeInfinity;
        BinStats[] distanceBins = new BinStats[DistanceBinLabels.Length];
        BinStats[] angleBins = new BinStats[AngleBinLabels.Length];

        for (int i = 0; i < observations.Count; i++)
        {
            VertexObservation observation = observations[i];
            bool anyRisk = observation.boundaryRisk || observation.creaseRisk;
            if (observation.boundaryRisk)
                boundaryRiskCount++;
            if (observation.creaseRisk)
                creaseRiskCount++;
            if (anyRisk)
                anyRiskCount++;

            viewDepthSum += observation.viewDepth;
            distanceSum += observation.euclideanDistance;
            angleSum += observation.viewAngleDegrees;
            faceAngleSum += observation.maxFaceNormalAngleDegrees;
            minViewDepth = Mathf.Min(minViewDepth, observation.viewDepth);
            maxViewDepth = Mathf.Max(maxViewDepth, observation.viewDepth);
            minDistance = Mathf.Min(minDistance, observation.euclideanDistance);
            maxDistance = Mathf.Max(maxDistance, observation.euclideanDistance);
            distanceBins[DistanceBinIndex(observation.euclideanDistance)].Add(observation);
            angleBins[AngleBinIndex(observation.viewAngleDegrees)].Add(observation);
        }

        float invCount = 1f / observations.Count;
        StringBuilder frameRow = new StringBuilder(512);
        frameRow.Append(EscapeCsv(frameName)).Append(',')
            .Append(observations.Count).Append(',')
            .Append(triangleCount).Append(',')
            .Append(boundaryRiskCount).Append(',')
            .Append(creaseRiskCount).Append(',')
            .Append(anyRiskCount).Append(',')
            .Append(FormatFloat((float)(viewDepthSum * invCount))).Append(',')
            .Append(FormatFloat((float)(distanceSum * invCount))).Append(',')
            .Append(FormatFloat((float)(angleSum * invCount))).Append(',')
            .Append(FormatFloat((float)(faceAngleSum * invCount))).Append(',')
            .Append(FormatFloat(minViewDepth)).Append(',')
            .Append(FormatFloat(maxViewDepth)).Append(',')
            .Append(FormatFloat(minDistance)).Append(',')
            .Append(FormatFloat(maxDistance)).Append(',')
            .Append(EscapeCsv(pointsPath)).AppendLine();
        File.AppendAllText(_frameObservationStatsPath, frameRow.ToString(), Encoding.UTF8);

        AppendBinRows(_distanceBinsPath, frameName, DistanceBinLabels, distanceBins);
        AppendBinRows(_angleBinsPath, frameName, AngleBinLabels, angleBins);

        StringBuilder edgeRow = new StringBuilder(256);
        edgeRow.Append(EscapeCsv(frameName)).Append(',')
            .Append(observations.Count).Append(',')
            .Append(boundaryRiskCount).Append(',')
            .Append(creaseRiskCount).Append(',')
            .Append(anyRiskCount).Append(',')
            .Append(FormatFloat(boundaryRiskCount / (float)observations.Count)).Append(',')
            .Append(FormatFloat(creaseRiskCount / (float)observations.Count)).Append(',')
            .Append(FormatFloat(anyRiskCount / (float)observations.Count)).Append(',')
            .Append(FormatFloat(meshCreaseRiskAngleDegrees)).AppendLine();
        File.AppendAllText(_edgeRiskStatsPath, edgeRow.ToString(), Encoding.UTF8);
    }

    private static void AppendBinRows(string path, string frameName, string[] labels, BinStats[] bins)
    {
        StringBuilder rows = new StringBuilder(labels.Length * 160);
        for (int i = 0; i < labels.Length; i++)
        {
            BinStats bin = bins[i];
            float inv = bin.count > 0 ? 1f / bin.count : 0f;
            rows.Append(EscapeCsv(frameName)).Append(',')
                .Append(labels[i]).Append(',')
                .Append(bin.count).Append(',')
                .Append(FormatFloat((float)(bin.viewDepthSum * inv))).Append(',')
                .Append(FormatFloat((float)(bin.distanceSum * inv))).Append(',')
                .Append(FormatFloat((float)(bin.angleSum * inv))).Append(',')
                .Append(bin.boundaryRiskCount).Append(',')
                .Append(bin.creaseRiskCount).Append(',')
                .Append(bin.riskCount).AppendLine();
        }
        File.AppendAllText(path, rows.ToString(), Encoding.UTF8);
    }

    private void AppendTargetGateRow(string frameName, bool accepted, ObservationExportResult result, bool countedFarSurface, bool countedHighAngleRisk, bool countedNearEdgeRisk, string reason)
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(_targetGatePath))
            return;

        StringBuilder row = new StringBuilder(256);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(accepted ? 1 : 0).Append(',')
            .Append(result.farSurfaceQualified ? 1 : 0).Append(',')
            .Append(result.highAngleRiskQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeRiskQualified ? 1 : 0).Append(',')
            .Append(result.farSurfaceCount).Append(',')
            .Append(result.highAngleRiskCount).Append(',')
            .Append(result.nearEdgeRiskCount).Append(',')
            .Append(_targetFarSurfaceFrames).Append(',')
            .Append(_targetHighAngleRiskFrames).Append(',')
            .Append(_targetNearEdgeRiskFrames).Append(',')
            .Append(_targetRejectedFrames).Append(',')
            .Append(result.pointCount).Append(',')
            .Append(EscapeCsv(reason)).Append(',')
            .Append(result.stableMainQualified ? 1 : 0).Append(',')
            .Append(result.topCoverageQualified ? 1 : 0).Append(',')
            .Append(result.middleCoverageQualified ? 1 : 0).Append(',')
            .Append(result.bottomCoverageQualified ? 1 : 0).Append(',')
            .Append(result.nearDistanceQualified ? 1 : 0).Append(',')
            .Append(result.midDistanceQualified ? 1 : 0).Append(',')
            .Append(result.farDistanceQualified ? 1 : 0).Append(',')
            .Append(result.frontAngleQualified ? 1 : 0).Append(',')
            .Append(result.obliqueAngleQualified ? 1 : 0).Append(',')
            .Append(result.extremeAngleQualified ? 1 : 0).Append(',')
            .Append(result.riskLayerQualified ? 1 : 0).Append(',')
            .Append(result.mainFrontQualified ? 1 : 0).Append(',')
            .Append(result.mainLeftObliqueQualified ? 1 : 0).Append(',')
            .Append(result.mainRightObliqueQualified ? 1 : 0).Append(',')
            .Append(result.mainUpObliqueQualified ? 1 : 0).Append(',')
            .Append(result.mainDownObliqueQualified ? 1 : 0).Append(',')
            .Append(result.boundaryLeftQualified ? 1 : 0).Append(',')
            .Append(result.boundaryRightQualified ? 1 : 0).Append(',')
            .Append(result.cornerJunctionQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementRiskQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementCreaseQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementBoundaryQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementLeftQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementRightQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementUpQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementDownQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementObliqueQualified ? 1 : 0).Append(',')
            .Append(result.nearEdgeSupplementExtremeQualified ? 1 : 0).Append(',')
            .Append(result.stableMainCount).Append(',')
            .Append(result.topCoverageCount).Append(',')
            .Append(result.middleCoverageCount).Append(',')
            .Append(result.bottomCoverageCount).Append(',')
            .Append(result.nearDistanceCount).Append(',')
            .Append(result.midDistanceCount).Append(',')
            .Append(result.farDistanceCount).Append(',')
            .Append(result.frontAngleCount).Append(',')
            .Append(result.obliqueAngleCount).Append(',')
            .Append(result.extremeAngleCount).Append(',')
            .Append(result.riskLayerCount).Append(',')
            .Append(result.mainFrontCount).Append(',')
            .Append(result.mainLeftObliqueCount).Append(',')
            .Append(result.mainRightObliqueCount).Append(',')
            .Append(result.mainUpObliqueCount).Append(',')
            .Append(result.mainDownObliqueCount).Append(',')
            .Append(result.boundaryLeftCount).Append(',')
            .Append(result.boundaryRightCount).Append(',')
            .Append(result.cornerJunctionCount).Append(',')
            .Append(result.nearEdgeSupplementRiskCount).Append(',')
            .Append(result.nearEdgeSupplementCreaseCount).Append(',')
            .Append(result.nearEdgeSupplementBoundaryCount).Append(',')
            .Append(result.nearEdgeSupplementLeftCount).Append(',')
            .Append(result.nearEdgeSupplementRightCount).Append(',')
            .Append(result.nearEdgeSupplementUpCount).Append(',')
            .Append(result.nearEdgeSupplementDownCount).Append(',')
            .Append(result.nearEdgeSupplementObliqueCount).Append(',')
            .Append(result.nearEdgeSupplementExtremeCount).Append(',')
            .Append(_targetStableMainFrames).Append(',')
            .Append(_targetTopCoverageFrames).Append(',')
            .Append(_targetMiddleCoverageFrames).Append(',')
            .Append(_targetBottomCoverageFrames).Append(',')
            .Append(_targetNearDistanceFrames).Append(',')
            .Append(_targetMidDistanceFrames).Append(',')
            .Append(_targetFarDistanceFrames).Append(',')
            .Append(_targetFrontAngleFrames).Append(',')
            .Append(_targetObliqueAngleFrames).Append(',')
            .Append(_targetExtremeAngleFrames).Append(',')
            .Append(_targetRiskLayerFrames).Append(',')
            .Append(_targetMainFrontFrames).Append(',')
            .Append(_targetMainLeftObliqueFrames).Append(',')
            .Append(_targetMainRightObliqueFrames).Append(',')
            .Append(_targetMainUpObliqueFrames).Append(',')
            .Append(_targetMainDownObliqueFrames).Append(',')
            .Append(_targetBoundaryLeftFrames).Append(',')
            .Append(_targetBoundaryRightFrames).Append(',')
            .Append(_targetCornerJunctionFrames).Append(',')
            .Append(_targetNearEdgeSupplementRiskFrames).Append(',')
            .Append(_targetNearEdgeSupplementCreaseFrames).Append(',')
            .Append(_targetNearEdgeSupplementBoundaryFrames).Append(',')
            .Append(_targetNearEdgeSupplementLeftFrames).Append(',')
            .Append(_targetNearEdgeSupplementRightFrames).Append(',')
            .Append(_targetNearEdgeSupplementUpFrames).Append(',')
            .Append(_targetNearEdgeSupplementDownFrames).Append(',')
            .Append(_targetNearEdgeSupplementObliqueFrames).Append(',')
            .Append(_targetNearEdgeSupplementExtremeFrames).Append(',')
            .Append(CompletedCoreMetricCount()).Append(',')
            .Append(CompletedDirectedMetricCount()).Append(',')
            .Append(CompletedNearEdgeSupplementMetricCount()).Append(',')
            .Append(Mathf.Max(1, targetFramesPerCoreMetric)).Append(',')
            .Append(Mathf.Max(1, targetNearEdgeSupplementFrames)).Append(',')
            .Append(EscapeCsv(observationCaptureMode.ToString())).AppendLine();
        File.AppendAllText(_targetGatePath, row.ToString(), Encoding.UTF8);
    }

    private static int DistanceBinIndex(float distance)
    {
        for (int i = 0; i < DistanceBinLabels.Length; i++)
        {
            if (distance >= DistanceBinEdges[i] && distance < DistanceBinEdges[i + 1])
                return i;
        }
        return DistanceBinLabels.Length - 1;
    }

    private static string DistanceBinLabel(float distance)
        => DistanceBinLabels[DistanceBinIndex(distance)];

    private static int AngleBinIndex(float angle)
    {
        for (int i = 0; i < AngleBinLabels.Length; i++)
        {
            if (angle >= AngleBinEdges[i] && angle < AngleBinEdges[i + 1])
                return i;
        }
        return AngleBinLabels.Length - 1;
    }

    private static string AngleBinLabel(float angle)
        => AngleBinLabels[AngleBinIndex(angle)];

    private static bool TryParseVector(string[] parts, int startIndex, out Vector3 value)
    {
        value = Vector3.zero;
        if (parts.Length <= startIndex + 2 ||
            !float.TryParse(parts[startIndex], NumberStyles.Float, CultureInfo.InvariantCulture, out float x) ||
            !float.TryParse(parts[startIndex + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float y) ||
            !float.TryParse(parts[startIndex + 2], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
            return false;

        value = new Vector3(x, y, z);
        return true;
    }

    private void RequestRawDepthProbeExport(string csvPath, string jsonPath, string frameName)
    {
        if (!_sessionFileExportEnabled)
            return;

        Texture rawDepth = Shader.GetGlobalTexture(EnvironmentDepthTextureId);
        if (rawDepth == null || string.IsNullOrEmpty(csvPath))
            return;

        if (rawDepthProbeShader == null)
            rawDepthProbeShader = Resources.Load<ComputeShader>("ScanCoverRawEnvironmentDepthProbe");
        if (rawDepthProbeShader == null)
            return;

        int count = RawDepthProbeSize * RawDepthProbeSize * 2;
        if (_rawDepthProbeBuffer == null || _rawDepthProbeBuffer.count != count)
        {
            ReleaseRawDepthResources();
            _rawDepthProbeBuffer = new ComputeBuffer(count, sizeof(float));
        }

        int kernel = rawDepthProbeShader.FindKernel("CopyRaw");
        rawDepthProbeShader.SetTexture(kernel, EnvironmentDepthTextureId, rawDepth);
        rawDepthProbeShader.SetFloat("_EnvironmentDepthTextureSize", rawDepth.width);
        rawDepthProbeShader.SetBuffer(kernel, "_RawDepthProbe", _rawDepthProbeBuffer);
        rawDepthProbeShader.Dispatch(kernel, 1, 1, 1);

        Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
        int rawWidth = rawDepth.width;
        int rawHeight = rawDepth.height;
        AsyncGPUReadback.Request(_rawDepthProbeBuffer, request =>
        {
            if (request.hasError)
                return;
            WriteRawDepthProbeFiles(csvPath, jsonPath, frameName, request.GetData<float>(), RawDepthProbeSize, rawWidth, rawHeight, zParams);
        });
    }

    private void WriteRawDepthProbeFiles(string csvPath, string jsonPath, string frameName, Unity.Collections.NativeArray<float> rawValues, int textureSize, int rawWidth, int rawHeight, Vector4 zParams)
    {
        if (!_sessionFileExportEnabled)
            return;

        StringBuilder csv = new StringBuilder(textureSize * textureSize * 80);
        csv.AppendLine("# ScanCover raw environment depth probe");
        csv.Append("frame=").Append(frameName).AppendLine();
        csv.Append("sourceTextureSize=").Append(rawWidth).Append('x').Append(rawHeight).AppendLine();
        csv.Append("probeSize=").Append(textureSize).Append('x').Append(textureSize).AppendLine();
        csv.AppendLine("eye,x,y,raw,linearMeters,valid");

        int[] positive = new int[2];
        float[] minLinear = { 0f, 0f };
        float[] maxLinear = { 0f, 0f };
        double[] sumLinear = { 0.0, 0.0 };
        bool[][] validMasks = { new bool[textureSize * textureSize], new bool[textureSize * textureSize] };
        float[][] linearMeters = { new float[textureSize * textureSize], new float[textureSize * textureSize] };
        for (int eye = 0; eye < 2; eye++)
        {
            int eyeOffset = eye * textureSize * textureSize;
            for (int y = 0; y < textureSize; y++)
            {
                for (int x = 0; x < textureSize; x++)
                {
                    float raw = rawValues[eyeOffset + x + y * textureSize];
                    float linear = RawDepthToLinearMeters(raw, zParams);
                    bool valid = linear > 0f;
                    int index = x + y * textureSize;
                    validMasks[eye][index] = valid;
                    linearMeters[eye][index] = linear;
                    if (valid)
                    {
                        positive[eye]++;
                        minLinear[eye] = positive[eye] == 1 ? linear : Mathf.Min(minLinear[eye], linear);
                        maxLinear[eye] = positive[eye] == 1 ? linear : Mathf.Max(maxLinear[eye], linear);
                        sumLinear[eye] += linear;
                    }

                    csv.Append(eye).Append(',')
                        .Append(x).Append(',')
                        .Append(y).Append(',')
                        .Append(FormatFloat(raw)).Append(',')
                        .Append(FormatFloat(linear)).Append(',')
                        .Append(valid ? 1 : 0).AppendLine();
                }
            }
        }
        File.WriteAllText(csvPath, csv.ToString(), Encoding.UTF8);

        if (exportQuest3ObservationBadnessStats)
            WriteRawDepthBadnessStats(frameName, validMasks, linearMeters, textureSize, minLinear, maxLinear, sumLinear, positive);

        if (string.IsNullOrEmpty(jsonPath))
            return;
        StringBuilder json = new StringBuilder(512);
        json.AppendLine("{");
        AppendJsonString(json, "frame", frameName, 1, true);
        AppendJsonNumber(json, "rawWidth", rawWidth, 1, true);
        AppendJsonNumber(json, "rawHeight", rawHeight, 1, true);
        AppendJsonNumber(json, "probeSize", textureSize, 1, true);
        json.Append("  \"zBufferParams\": [")
            .Append(FormatFloat(zParams.x)).Append(", ")
            .Append(FormatFloat(zParams.y)).Append(", ")
            .Append(FormatFloat(zParams.z)).Append(", ")
            .Append(FormatFloat(zParams.w)).AppendLine("],");
        json.Append("  \"eyes\": [").AppendLine();
        for (int eye = 0; eye < 2; eye++)
        {
            float avg = positive[eye] > 0 ? (float)(sumLinear[eye] / positive[eye]) : 0f;
            json.Append("    { \"eye\": ").Append(eye)
                .Append(", \"valid\": ").Append(positive[eye])
                .Append(", \"minLinearMeters\": ").Append(FormatFloat(minLinear[eye]))
                .Append(", \"maxLinearMeters\": ").Append(FormatFloat(maxLinear[eye]))
                .Append(", \"avgLinearMeters\": ").Append(FormatFloat(avg)).Append(" }");
            json.AppendLine(eye == 0 ? "," : "");
        }
        json.AppendLine("  ]");
        json.AppendLine("}");
        File.WriteAllText(jsonPath, json.ToString(), Encoding.UTF8);
    }

    private void WriteRawDepthBadnessStats(
        string frameName,
        bool[][] validMasks,
        float[][] linearMeters,
        int textureSize,
        float[] minLinear,
        float[] maxLinear,
        double[] sumLinear,
        int[] positive)
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(_rawDepthBadnessFramesPath) || validMasks == null || linearMeters == null)
            return;

        _rawDepthBadnessFrameCount++;
        StringBuilder frameRows = new StringBuilder(512);
        StringBuilder componentRows = new StringBuilder(1024);
        int totalPixels = textureSize * textureSize;
        int largeHoleMinPixels = Mathf.Max(1, rawDepthHoleMinPixels);
        float edgeJump = Mathf.Max(0.001f, rawDepthEdgeDepthJumpMeters);
        bool frameHasInvalid = false;
        bool frameHasLargeHole = false;
        bool frameHasEdgeJump = false;
        bool frameHasPersistentInvalid = false;

        for (int eye = 0; eye < 2; eye++)
        {
            bool[] valid = validMasks[eye];
            float[] linear = linearMeters[eye];
            if (valid == null || linear == null || valid.Length != totalPixels || linear.Length != totalPixels)
                continue;

            if (_previousRawDepthValidMasks[eye] == null || _previousRawDepthValidMasks[eye].Length != totalPixels)
            {
                _previousRawDepthValidMasks[eye] = new bool[totalPixels];
                _rawDepthInvalidRunLengths[eye] = new int[totalPixels];
            }

            RawDepthBadnessStats stats = BuildRawDepthBadnessStats(
                eye,
                valid,
                linear,
                textureSize,
                minLinear[eye],
                maxLinear[eye],
                positive[eye] > 0 ? (float)(sumLinear[eye] / positive[eye]) : 0f,
                edgeJump,
                largeHoleMinPixels,
                _previousRawDepthValidMasks[eye],
                _rawDepthInvalidRunLengths[eye],
                out List<RawDepthHoleComponent> components);

            _rawDepthBadnessEyeSamples++;
            _rawDepthBadnessValidSamples += stats.validPixels;
            _rawDepthBadnessInvalidSamples += stats.invalidPixels;
            _rawDepthBadnessLargeHoleComponents += stats.largeHoleComponentCount;
            _rawDepthBadnessPersistentInvalidPixels += stats.persistentInvalidPixels;
            _rawDepthBadnessNewlyInvalidPixels += stats.newlyInvalidPixels;
            _rawDepthBadnessRecoveredPixels += stats.recoveredPixels;
            _rawDepthBadnessEdgeRiskPixels += stats.edgeRiskValidPixels;
            if (stats.invalidPixels >= Mathf.Max(1, minBadnessInvalidPixelsPerEye))
                frameHasInvalid = true;
            if (stats.largeHoleComponentCount > 0)
                frameHasLargeHole = true;
            if (stats.edgeRiskValidPixels >= Mathf.Max(1, minBadnessEdgeRiskPixelsPerEye))
                frameHasEdgeJump = true;
            if (stats.persistentInvalidPixels > 0)
                frameHasPersistentInvalid = true;

            float validRatio = stats.totalPixels > 0 ? stats.validPixels / (float)stats.totalPixels : 0f;
            float invalidRatio = stats.totalPixels > 0 ? stats.invalidPixels / (float)stats.totalPixels : 0f;
            frameRows.Append(EscapeCsv(frameName)).Append(',')
                .Append(stats.eye).Append(',')
                .Append(stats.totalPixels).Append(',')
                .Append(stats.validPixels).Append(',')
                .Append(stats.invalidPixels).Append(',')
                .Append(FormatFloat(validRatio)).Append(',')
                .Append(FormatFloat(invalidRatio)).Append(',')
                .Append(stats.invalidComponentCount).Append(',')
                .Append(stats.largeHoleComponentCount).Append(',')
                .Append(stats.largestHolePixels).Append(',')
                .Append(stats.persistentInvalidPixels).Append(',')
                .Append(stats.newlyInvalidPixels).Append(',')
                .Append(stats.recoveredPixels).Append(',')
                .Append(stats.edgeRiskValidPixels).Append(',')
                .Append(FormatFloat(stats.minLinearMeters)).Append(',')
                .Append(FormatFloat(stats.maxLinearMeters)).Append(',')
                .Append(FormatFloat(stats.avgLinearMeters)).Append(',')
                .Append(FormatFloat(edgeJump)).Append(',')
                .Append(largeHoleMinPixels).AppendLine();

            for (int i = 0; i < components.Count; i++)
            {
                RawDepthHoleComponent c = components[i];
                componentRows.Append(EscapeCsv(frameName)).Append(',')
                    .Append(eye).Append(',')
                    .Append(c.id).Append(',')
                    .Append(c.pixelCount).Append(',')
                    .Append(c.minX).Append(',')
                    .Append(c.minY).Append(',')
                    .Append(c.maxX).Append(',')
                    .Append(c.maxY).Append(',')
                    .Append(c.touchesBorder ? 1 : 0).Append(',')
                    .Append(c.pixelCount >= largeHoleMinPixels ? 1 : 0).AppendLine();
            }

            Array.Copy(valid, _previousRawDepthValidMasks[eye], totalPixels);
        }

        if (frameHasInvalid)
            _rawDepthBadnessInvalidFrames++;
        if (frameHasLargeHole)
            _rawDepthBadnessLargeHoleFrames++;
        if (frameHasEdgeJump)
            _rawDepthBadnessEdgeJumpFrames++;
        if (frameHasPersistentInvalid)
            _rawDepthBadnessPersistentInvalidFrames++;

        File.AppendAllText(_rawDepthBadnessFramesPath, frameRows.ToString(), Encoding.UTF8);
        if (componentRows.Length > 0 && !string.IsNullOrEmpty(_rawDepthHoleComponentsPath))
            File.AppendAllText(_rawDepthHoleComponentsPath, componentRows.ToString(), Encoding.UTF8);
        WriteQuest3BadnessSummary();
    }

    private RawDepthBadnessStats BuildRawDepthBadnessStats(
        int eye,
        bool[] valid,
        float[] linear,
        int textureSize,
        float minLinear,
        float maxLinear,
        float avgLinear,
        float edgeJump,
        int largeHoleMinPixels,
        bool[] previousValid,
        int[] invalidRunLengths,
        out List<RawDepthHoleComponent> components)
    {
        int totalPixels = textureSize * textureSize;
        components = new List<RawDepthHoleComponent>(64);
        RawDepthBadnessStats stats = new RawDepthBadnessStats
        {
            eye = eye,
            totalPixels = totalPixels,
            minLinearMeters = minLinear,
            maxLinearMeters = maxLinear,
            avgLinearMeters = avgLinear
        };

        bool[] visited = new bool[totalPixels];
        Queue<int> queue = new Queue<int>(256);
        for (int i = 0; i < totalPixels; i++)
        {
            if (valid[i])
            {
                stats.validPixels++;
                if (previousValid != null && previousValid.Length == totalPixels && !previousValid[i])
                    stats.recoveredPixels++;
                if (invalidRunLengths != null && invalidRunLengths.Length == totalPixels)
                    invalidRunLengths[i] = 0;
                continue;
            }

            stats.invalidPixels++;
            if (previousValid != null && previousValid.Length == totalPixels && previousValid[i])
                stats.newlyInvalidPixels++;
            if (invalidRunLengths != null && invalidRunLengths.Length == totalPixels)
            {
                invalidRunLengths[i]++;
                if (invalidRunLengths[i] >= 2)
                    stats.persistentInvalidPixels++;
            }
        }

        for (int y = 0; y < textureSize; y++)
        {
            for (int x = 0; x < textureSize; x++)
            {
                int start = x + y * textureSize;
                if (valid[start] || visited[start])
                    continue;

                RawDepthHoleComponent component = new RawDepthHoleComponent
                {
                    id = components.Count,
                    minX = x,
                    minY = y,
                    maxX = x,
                    maxY = y
                };
                visited[start] = true;
                queue.Enqueue(start);
                while (queue.Count > 0)
                {
                    int index = queue.Dequeue();
                    int cx = index % textureSize;
                    int cy = index / textureSize;
                    component.pixelCount++;
                    component.minX = Mathf.Min(component.minX, cx);
                    component.minY = Mathf.Min(component.minY, cy);
                    component.maxX = Mathf.Max(component.maxX, cx);
                    component.maxY = Mathf.Max(component.maxY, cy);
                    if (cx == 0 || cy == 0 || cx == textureSize - 1 || cy == textureSize - 1)
                        component.touchesBorder = true;

                    EnqueueInvalidNeighbor(cx - 1, cy, textureSize, valid, visited, queue);
                    EnqueueInvalidNeighbor(cx + 1, cy, textureSize, valid, visited, queue);
                    EnqueueInvalidNeighbor(cx, cy - 1, textureSize, valid, visited, queue);
                    EnqueueInvalidNeighbor(cx, cy + 1, textureSize, valid, visited, queue);
                }

                stats.invalidComponentCount++;
                stats.largestHolePixels = Mathf.Max(stats.largestHolePixels, component.pixelCount);
                if (component.pixelCount >= largeHoleMinPixels)
                    stats.largeHoleComponentCount++;
                components.Add(component);
            }
        }

        stats.edgeRiskValidPixels = CountRawDepthEdgeRiskPixels(valid, linear, textureSize, edgeJump);
        return stats;
    }

    private static void EnqueueInvalidNeighbor(int x, int y, int textureSize, bool[] valid, bool[] visited, Queue<int> queue)
    {
        if (x < 0 || y < 0 || x >= textureSize || y >= textureSize)
            return;
        int index = x + y * textureSize;
        if (visited[index] || valid[index])
            return;
        visited[index] = true;
        queue.Enqueue(index);
    }

    private static int CountRawDepthEdgeRiskPixels(bool[] valid, float[] linear, int textureSize, float edgeJump)
    {
        int count = 0;
        for (int y = 0; y < textureSize; y++)
        {
            for (int x = 0; x < textureSize; x++)
            {
                int index = x + y * textureSize;
                if (!valid[index])
                    continue;
                float center = linear[index];
                if (RawDepthNeighborIsRisk(x - 1, y, textureSize, valid, linear, center, edgeJump) ||
                    RawDepthNeighborIsRisk(x + 1, y, textureSize, valid, linear, center, edgeJump) ||
                    RawDepthNeighborIsRisk(x, y - 1, textureSize, valid, linear, center, edgeJump) ||
                    RawDepthNeighborIsRisk(x, y + 1, textureSize, valid, linear, center, edgeJump))
                    count++;
            }
        }
        return count;
    }

    private static bool RawDepthNeighborIsRisk(int x, int y, int textureSize, bool[] valid, float[] linear, float center, float edgeJump)
    {
        if (x < 0 || y < 0 || x >= textureSize || y >= textureSize)
            return true;
        int index = x + y * textureSize;
        if (!valid[index])
            return true;
        return Mathf.Abs(center - linear[index]) >= edgeJump;
    }

    private void WriteQuest3BadnessSummary()
    {
        if (!_sessionFileExportEnabled || !exportQuest3ObservationBadnessStats || string.IsNullOrEmpty(_rawDepthBadnessSummaryPath))
            return;

        int totalSamples = Mathf.Max(1, _rawDepthBadnessValidSamples + _rawDepthBadnessInvalidSamples);
        StringBuilder json = new StringBuilder(1024);
        json.AppendLine("{");
        AppendJsonNumber(json, "rawDepthBadnessFrames", _rawDepthBadnessFrameCount, 1, true);
        AppendJsonNumber(json, "eyeSamples", _rawDepthBadnessEyeSamples, 1, true);
        AppendJsonNumber(json, "validPixels", _rawDepthBadnessValidSamples, 1, true);
        AppendJsonNumber(json, "invalidPixels", _rawDepthBadnessInvalidSamples, 1, true);
        AppendJsonNumber(json, "invalidRatio", _rawDepthBadnessInvalidSamples / (float)totalSamples, 1, true);
        AppendJsonNumber(json, "largeHoleComponents", _rawDepthBadnessLargeHoleComponents, 1, true);
        AppendJsonNumber(json, "invalidFrames", _rawDepthBadnessInvalidFrames, 1, true);
        AppendJsonNumber(json, "largeHoleFrames", _rawDepthBadnessLargeHoleFrames, 1, true);
        AppendJsonNumber(json, "edgeJumpFrames", _rawDepthBadnessEdgeJumpFrames, 1, true);
        AppendJsonNumber(json, "persistentInvalidFrames", _rawDepthBadnessPersistentInvalidFrames, 1, true);
        AppendJsonNumber(json, "persistentInvalidPixels", _rawDepthBadnessPersistentInvalidPixels, 1, true);
        AppendJsonNumber(json, "newlyInvalidPixels", _rawDepthBadnessNewlyInvalidPixels, 1, true);
        AppendJsonNumber(json, "recoveredPixels", _rawDepthBadnessRecoveredPixels, 1, true);
        AppendJsonNumber(json, "edgeRiskValidPixels", _rawDepthBadnessEdgeRiskPixels, 1, true);
        AppendJsonNumber(json, "rawDepthHoleMinPixels", Mathf.Max(1, rawDepthHoleMinPixels), 1, true);
        AppendJsonNumber(json, "rawDepthEdgeDepthJumpMeters", Mathf.Max(0.001f, rawDepthEdgeDepthJumpMeters), 1, true);
        AppendJsonNumber(json, "targetBadnessFrames", Mathf.Max(1, targetBadnessFrames), 1, true);
        AppendJsonNumber(json, "targetBadnessInvalidFrames", Mathf.Max(1, targetBadnessInvalidFrames), 1, true);
        AppendJsonNumber(json, "targetBadnessLargeHoleFrames", Mathf.Max(1, targetBadnessLargeHoleFrames), 1, true);
        AppendJsonNumber(json, "targetBadnessEdgeJumpFrames", Mathf.Max(1, targetBadnessEdgeJumpFrames), 1, true);
        AppendJsonNumber(json, "targetBadnessPersistentInvalidFrames", Mathf.Max(1, targetBadnessPersistentInvalidFrames), 1, false);
        json.AppendLine("}");
        File.WriteAllText(_rawDepthBadnessSummaryPath, json.ToString(), Encoding.UTF8);
    }

    private RepeatCoverageGateResult EvaluateRepeatCoverageGate(string verticesCsvPath, Transform pose, int frameIndex, bool apply)
    {
        RepeatCoverageGateResult result = new RepeatCoverageGateResult
        {
            hasData = false,
            accepted = false,
            reason = "reject no repeat data"
        };

        if (string.IsNullOrEmpty(verticesCsvPath) || !File.Exists(verticesCsvPath))
            return result;
        if (!TryReadSurfaceVertices(verticesCsvPath, out List<VertexObservation> observations))
            return result;

        ApplyCameraObservationMetrics(observations, pose);
        Dictionary<Vector3Int, RepeatCoverageCandidateVoxel> candidate = BuildRepeatCoverageCandidate(observations);
        result.hasData = candidate.Count > 0;
        result.candidateVoxels = candidate.Count;
        if (!result.hasData)
        {
            result.reason = "reject no candidate voxels";
            return result;
        }

        Vector3 cameraPosition = pose != null ? pose.position : Vector3.zero;
        float parallaxMin = Mathf.Max(0f, repeatCoverageMinParallaxMeters);
        int minHits = Mathf.Max(1, repeatCoverageMinDistinctHits);

        foreach (KeyValuePair<Vector3Int, RepeatCoverageCandidateVoxel> pair in candidate)
        {
            if (!_repeatCoverageVoxels.TryGetValue(pair.Key, out RepeatCoverageVoxel voxel) || voxel.frameHits <= 0)
            {
                result.newVoxels++;
                if (minHits <= 1)
                    result.newStableVoxels++;
                continue;
            }

            result.rehitVoxels++;
            float firstParallax = Vector3.Distance(cameraPosition, voxel.firstCameraPosition);
            float lastParallax = Vector3.Distance(cameraPosition, voxel.lastCameraPosition);
            if (firstParallax >= parallaxMin || lastParallax >= parallaxMin)
                result.parallaxRehitVoxels++;
            if (!voxel.stable && voxel.frameHits + 1 >= minHits)
                result.newStableVoxels++;
        }

        int minCandidate = Mathf.Max(1, repeatCoverageMinCandidateVoxels);
        int minNewStable = Mathf.Max(1, repeatCoverageMinNewStableVoxelsPerFrame);
        int minNewOrRehit = Mathf.Max(1, repeatCoverageMinNewOrRehitVoxelsPerFrame);
        bool firstUsefulFrame = _repeatCoverageAcceptedFrames <= 0 && result.candidateVoxels >= minCandidate;
        bool improvesStableCoverage = result.newStableVoxels >= minNewStable;
        bool improvesRepeatCoverage = result.newVoxels + result.rehitVoxels >= minNewOrRehit && result.parallaxRehitVoxels >= Mathf.Max(1, minNewStable / 2);
        bool stillNeedsStableCoverage = _repeatCoverageStableVoxelCount < Mathf.Max(1, repeatCoverageTargetStableVoxels);

        result.accepted = firstUsefulFrame || improvesStableCoverage || (stillNeedsStableCoverage && improvesRepeatCoverage);
        if (firstUsefulFrame)
            result.reason = "accept first coverage";
        else if (improvesStableCoverage)
            result.reason = "accept new stable hits";
        else if (improvesRepeatCoverage)
            result.reason = "accept repeat parallax";
        else
            result.reason = "reject low repeat gain";

        if (apply && result.accepted)
            ApplyRepeatCoverageCandidate(candidate, cameraPosition, frameIndex, minHits);

        return result;
    }

    private Dictionary<Vector3Int, RepeatCoverageCandidateVoxel> BuildRepeatCoverageCandidate(List<VertexObservation> observations)
    {
        Dictionary<Vector3Int, RepeatCoverageCandidateVoxel> candidate = new Dictionary<Vector3Int, RepeatCoverageCandidateVoxel>(4096);
        if (observations == null)
            return candidate;

        float voxelSize = Mathf.Max(0.005f, repeatCoverageVoxelSizeMeters);
        float maxViewAngle = Mathf.Clamp(repeatCoverageMaxViewAngleDegrees, 0f, 89f);
        for (int i = 0; i < observations.Count; i++)
        {
            VertexObservation observation = observations[i];
            if (observation.viewAngleDegrees > maxViewAngle)
                continue;

            Vector3 point = observation.world;
            Vector3Int key = new Vector3Int(
                Mathf.FloorToInt(point.x / voxelSize),
                Mathf.FloorToInt(point.y / voxelSize),
                Mathf.FloorToInt(point.z / voxelSize));

            candidate.TryGetValue(key, out RepeatCoverageCandidateVoxel voxel);
            if (voxel.pointHits <= 0)
            {
                voxel.min = point;
                voxel.max = point;
            }

            voxel.pointHits++;
            voxel.positionSum += point;
            voxel.normalSum += observation.normal;
            voxel.min = Vector3.Min(voxel.min, point);
            voxel.max = Vector3.Max(voxel.max, point);
            candidate[key] = voxel;
        }

        return candidate;
    }

    private void ApplyRepeatCoverageCandidate(Dictionary<Vector3Int, RepeatCoverageCandidateVoxel> candidate, Vector3 cameraPosition, int frameIndex, int minHits)
    {
        foreach (KeyValuePair<Vector3Int, RepeatCoverageCandidateVoxel> pair in candidate)
        {
            RepeatCoverageCandidateVoxel source = pair.Value;
            _repeatCoverageVoxels.TryGetValue(pair.Key, out RepeatCoverageVoxel voxel);
            if (voxel.frameHits <= 0)
            {
                voxel.firstFrame = frameIndex;
                voxel.firstCameraPosition = cameraPosition;
                voxel.min = source.min;
                voxel.max = source.max;
            }

            voxel.frameHits++;
            voxel.pointHits += source.pointHits;
            voxel.lastFrame = frameIndex;
            voxel.lastCameraPosition = cameraPosition;
            voxel.positionSum += source.positionSum;
            voxel.normalSum += source.normalSum;
            voxel.min = Vector3.Min(voxel.min, source.min);
            voxel.max = Vector3.Max(voxel.max, source.max);
            if (!voxel.stable && voxel.frameHits >= minHits)
            {
                voxel.stable = true;
                _repeatCoverageStableVoxelCount++;
            }
            _repeatCoverageVoxels[pair.Key] = voxel;
        }
    }

    private void StoreRepeatCoverageLastStats(RepeatCoverageGateResult result)
    {
        _lastRepeatCandidateVoxels = result.candidateVoxels;
        _lastRepeatNewVoxels = result.newVoxels;
        _lastRepeatRehitVoxels = result.rehitVoxels;
        _lastRepeatParallaxRehitVoxels = result.parallaxRehitVoxels;
        _lastRepeatNewStableVoxels = result.newStableVoxels;
    }

    private void AppendRepeatCoverageGateRow(string frameName, bool accepted, RepeatCoverageGateResult result)
    {
        if (!_sessionFileExportEnabled || !gateFramesByRepeatCoverage || string.IsNullOrEmpty(_repeatCoverageGatePath))
            return;

        StringBuilder row = new StringBuilder(256);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(accepted ? 1 : 0).Append(',')
            .Append(result.candidateVoxels).Append(',')
            .Append(result.newVoxels).Append(',')
            .Append(result.rehitVoxels).Append(',')
            .Append(result.parallaxRehitVoxels).Append(',')
            .Append(result.newStableVoxels).Append(',')
            .Append(_repeatCoverageStableVoxelCount).Append(',')
            .Append(Mathf.Max(1, repeatCoverageTargetStableVoxels)).Append(',')
            .Append(_repeatCoverageAcceptedFrames).Append(',')
            .Append(_repeatCoverageRejectedFrames).Append(',')
            .Append(EscapeCsv(result.reason)).AppendLine();
        File.AppendAllText(_repeatCoverageGatePath, row.ToString(), Encoding.UTF8);
    }

    private void WriteRepeatCoverageOutputs()
    {
        if (!_sessionFileExportEnabled || !gateFramesByRepeatCoverage || string.IsNullOrEmpty(_repeatCoverageVoxelsPath))
            return;

        float voxelSize = Mathf.Max(0.005f, repeatCoverageVoxelSizeMeters);
        StringBuilder builder = new StringBuilder(Mathf.Max(1024, _repeatCoverageVoxels.Count * 150));
        builder.AppendLine("# ScanCover repeat coverage voxels");
        builder.Append("voxelSizeMeters=").Append(FormatFloat(voxelSize)).AppendLine();
        builder.Append("capturedFrames=").Append(_capturedFrameCount).AppendLine();
        builder.Append("acceptedFrames=").Append(_repeatCoverageAcceptedFrames).AppendLine();
        builder.Append("rejectedFrames=").Append(_repeatCoverageRejectedFrames).AppendLine();
        builder.Append("stableVoxelCount=").Append(_repeatCoverageStableVoxelCount).AppendLine();
        builder.AppendLine("voxelX,voxelY,voxelZ,stable,frameHits,pointHits,firstFrame,lastFrame,avgX,avgY,avgZ,avgNormalX,avgNormalY,avgNormalZ,minX,minY,minZ,maxX,maxY,maxZ");
        foreach (KeyValuePair<Vector3Int, RepeatCoverageVoxel> pair in _repeatCoverageVoxels)
        {
            RepeatCoverageVoxel voxel = pair.Value;
            Vector3 avg = voxel.pointHits > 0 ? voxel.positionSum / voxel.pointHits : Vector3.zero;
            Vector3 avgNormal = voxel.normalSum.sqrMagnitude > 1e-6f ? voxel.normalSum.normalized : Vector3.up;
            builder.Append(pair.Key.x).Append(',')
                .Append(pair.Key.y).Append(',')
                .Append(pair.Key.z).Append(',')
                .Append(voxel.stable ? 1 : 0).Append(',')
                .Append(voxel.frameHits).Append(',')
                .Append(voxel.pointHits).Append(',')
                .Append(voxel.firstFrame).Append(',')
                .Append(voxel.lastFrame).Append(',')
                .Append(FormatVector(avg)).Append(',')
                .Append(FormatVector(avgNormal)).Append(',')
                .Append(FormatVector(voxel.min)).Append(',')
                .Append(FormatVector(voxel.max)).AppendLine();
        }
        File.WriteAllText(_repeatCoverageVoxelsPath, builder.ToString(), Encoding.UTF8);
    }

    private bool UpdateRoomRawCoverage(string frameName, Transform pose, int frameIndex)
    {
        if (!trackRoomRawCoverage || depthGridPointCloud == null)
            return false;

        if (!depthGridPointCloud.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot))
        {
            _lastRoomRawStatus = "raw snapshot unavailable";
            return false;
        }

        RoomRawCoverageFrameResult result = EvaluateRoomRawCoverage(snapshot, pose, frameIndex);
        if (result.hasData && result.accepted)
        {
            ApplyRoomRawCoverage(snapshot, pose, frameIndex, result);
            result.riskVoxels = _roomRawCoverageRiskVoxels;
        }
        else if (result.hasData)
            _roomRawCoverageRejectedFrames++;

        _lastRoomRawValidSamples = result.validSamples;
        _lastRoomRawFocusVoxels = result.focusVoxels;
        _lastRoomRawFocusNewVoxels = result.focusNewVoxels;
        _lastRoomRawNewVoxels = result.newVoxels;
        _lastRoomRawRehitVoxels = result.rehitVoxels;
        _lastRoomRawHistoryRehitVoxels = result.historyRehitVoxels;
        _lastRoomRawHistoryAgreementVoxels = result.historyAgreementVoxels;
        _lastRoomRawHistoryConflictVoxels = result.historyConflictVoxels;
        _lastRoomRawParallaxRehitVoxels = result.parallaxRehitVoxels;
        _lastRoomRawStableNewVoxels = result.newStableVoxels;
        _lastRoomRawHighFrameVoxels = result.highFrameVoxels;
        _lastRoomRawNewHighVoxels = result.newHighVoxels;
        _lastRoomRawNewHighStableVoxels = result.newHighStableVoxels;
        _lastRoomRawLowFrameVoxels = result.lowFrameVoxels;
        _lastRoomRawNewLowVoxels = result.newLowVoxels;
        _lastRoomRawNewLowStableVoxels = result.newLowStableVoxels;
        _lastRoomRawRiskRatio = result.riskRatio;
        _lastRoomRawOverlapRatio = result.overlapRatio;
        _lastRoomRawHistoryAgreementRatio = result.historyAgreementRatio;
        _lastRoomRawAnchorAngle = result.anchorAngle;
        _lastRoomRawAnchorMove = result.anchorMove;
        _lastObservationOrderScore = result.observationOrderScore;
        _lastRoomRawStatus = result.status;
        MaybeExportRoomRawDepthSnapshot(frameName, snapshot, pose);
        MaybeExportRoomRawDepthFrame(frameName, snapshot, pose, frameIndex, result);
        AppendRoomRawCoverageFrameRow(frameName, snapshot.frameIndex, result);
        return result.hasData && result.accepted;
    }

    private RoomRawCoverageFrameResult EvaluateRoomRawCoverage(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Transform pose, int frameIndex)
    {
        RoomRawCoverageFrameResult result = default;
        if (snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null)
        {
            result.status = "no snapshot";
            return result;
        }

        int width = Mathf.Max(1, snapshot.resolutionWidth);
        int height = Mathf.Max(1, snapshot.resolutionHeight);
        int count = Mathf.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length);
        if (count <= 0)
        {
            result.status = "empty snapshot";
            return result;
        }

        Vector3 cameraPosition = pose != null ? pose.position : Vector3.zero;
        Vector3 cameraForward = pose != null ? pose.forward : Vector3.forward;
        result.totalSamples = count;
        if (!IsRoomRawCoverageAnchorUsable(pose, out result.anchorAngle, out result.anchorMove, out result.anchorFallback))
        {
            result.hasData = true;
            result.status = $"outside locked view angle={FormatFloat(result.anchorAngle)}deg move={FormatFloat(result.anchorMove)}m";
            return result;
        }
        result.rawObservationOrder = EvaluateRawObservationOrder(snapshot, cameraPosition, cameraForward);

        float voxelSize = Mathf.Max(0.005f, roomRawCoverageVoxelSizeMeters);
        float riskAngle = Mathf.Clamp(roomRawCoverageRiskViewAngleDegrees, 0f, 89f);
        float riskFacingMin = Mathf.Cos(riskAngle * Mathf.Deg2Rad);
        float highMinY = cameraPosition.y + Mathf.Max(0.1f, roomRawCoverageHighPointMinDeltaYMeters);
        float lowMaxY = cameraPosition.y - Mathf.Max(0.1f, roomRawCoverageLowPointMinDeltaYMeters);
        float jumpMeters = Mathf.Max(0.001f, roomRawCoverageNeighborDepthJumpMeters);
        int stableHitTarget = Mathf.Max(1, roomRawCoverageStableHitTarget);
        float parallaxMin = allowSameViewStableRoomRawHits && lockRoomRawCoverageToStartView ? 0f : Mathf.Max(0f, roomRawCoverageStableParallaxMeters);
        Dictionary<Vector3Int, RoomRawCoverageCandidateVoxel> candidates = new Dictionary<Vector3Int, RoomRawCoverageCandidateVoxel>(4096);

        result.hasData = true;
        for (int i = 0; i < count; i++)
        {
            if (!IsRawDepthSampleUsable(snapshot, i, cameraPosition, cameraForward))
                continue;

            int x = i % width;
            int y = i / width;
            float u = width > 1 ? x / (float)(width - 1) : 0.5f;
            float v = height > 1 ? y / (float)(height - 1) : 0.5f;
            bool focus = GetRoomRawCoverageFocusFlags(u, v, out bool core, out bool edgeBuffer);
            Vector3 point = snapshot.worldPositions[i];
            bool highPoint = point.y >= highMinY;
            bool lowPoint = point.y <= lowMaxY;
            Vector3 normal = (snapshot.worldNormals != null && i < snapshot.worldNormals.Length && IsFinite(snapshot.worldNormals[i]) && snapshot.worldNormals[i].sqrMagnitude > 1e-8f)
                ? snapshot.worldNormals[i].normalized
                : Vector3.up;

            bool risk = IsRawDepthNeighborJump(snapshot, i, width, height, jumpMeters, cameraPosition, cameraForward) ||
                IsRawObservationOrderRisk(result.rawObservationOrder, i);
            Vector3 toCamera = cameraPosition - point;
            if (toCamera.sqrMagnitude > 1e-8f)
            {
                float facing = Mathf.Abs(Vector3.Dot(normal, toCamera.normalized));
                if (facing < riskFacingMin)
                    risk = true;
            }

            Vector3Int key = WorldToVoxelKey(point, voxelSize);
            candidates.TryGetValue(key, out RoomRawCoverageCandidateVoxel candidate);
            candidate.pointHits++;
            candidate.high |= highPoint;
            candidate.low |= lowPoint;
            candidate.risk |= risk;
            candidate.focus |= focus;
            candidate.core |= core;
            candidate.edgeBuffer |= edgeBuffer;
            candidate.positionSum += point;
            candidate.normalSum += normal;
            candidates[key] = candidate;

            result.validSamples++;
            if (risk)
            {
                result.riskSamples++;
                if (focus)
                    result.focusRiskSamples++;
            }
        }

        result.frameVoxels = candidates.Count;
        foreach (KeyValuePair<Vector3Int, RoomRawCoverageCandidateVoxel> pair in candidates)
        {
            RoomRawCoverageCandidateVoxel candidate = pair.Value;
            bool focus = candidate.focus;
            if (focus)
                result.focusVoxels++;
            if (candidate.core)
                result.coreVoxels++;
            if (candidate.edgeBuffer)
                result.edgeBufferVoxels++;
            if (candidate.high)
                result.highFrameVoxels++;
            if (candidate.low)
                result.lowFrameVoxels++;

            bool hasPriorVoxel = _roomRawCoverageVoxels.TryGetValue(pair.Key, out RoomRawCoverageVoxel voxel) && voxel.frameHits > 0;
            bool hasHistoryVoxel = _roomRawCoverageHistoryVoxels.TryGetValue(pair.Key, out RoomRawCoverageVoxel historyVoxel) && historyVoxel.frameHits > 0;
            if (hasHistoryVoxel)
            {
                result.historyRehitVoxels++;
                Vector3 candidateCenter = candidate.positionSum / Mathf.Max(1, candidate.pointHits);
                Vector3 historyCenter = GetRoomRawCoverageVoxelAverage(historyVoxel);
                float seamTolerance = Mathf.Max(
                    Mathf.Max(0.005f, roomRawCoverageVoxelSizeMeters) * 1.5f,
                    Mathf.Max(0.001f, roomRawCoverageNeighborDepthJumpMeters) * 2f);
                if (Vector3.Distance(candidateCenter, historyCenter) <= seamTolerance)
                    result.historyAgreementVoxels++;
                else
                    result.historyConflictVoxels++;
            }

            if (!hasPriorVoxel)
            {
                result.newVoxels++;
                if (focus)
                    result.focusNewVoxels++;
                if (candidate.edgeBuffer)
                    result.edgeBufferNewVoxels++;
                if (candidate.high)
                    result.newHighVoxels++;
                if (candidate.low)
                    result.newLowVoxels++;
                continue;
            }

            result.rehitVoxels++;
            bool hasParallax = parallaxMin <= 0f ||
                Vector3.Distance(voxel.firstCameraPosition, cameraPosition) >= parallaxMin ||
                Vector3.Distance(voxel.lastCameraPosition, cameraPosition) >= parallaxMin;
            if (hasParallax)
                result.parallaxRehitVoxels++;

            int nextFrameHits = voxel.frameHits + 1;
            if (!voxel.stable && nextFrameHits >= stableHitTarget && hasParallax)
                result.newStableVoxels++;
            if (candidate.high && !voxel.high)
                result.newHighVoxels++;
            if (candidate.low && !voxel.low)
                result.newLowVoxels++;
            if ((candidate.high || voxel.high) && !voxel.highStable && nextFrameHits >= stableHitTarget && hasParallax)
                result.newHighStableVoxels++;
            if ((candidate.low || voxel.low) && !voxel.lowStable && nextFrameHits >= stableHitTarget && hasParallax)
                result.newLowStableVoxels++;
        }

        result.riskVoxels = _roomRawCoverageRiskVoxels;
        result.riskRatio = result.validSamples > 0 ? result.riskSamples / (float)result.validSamples : 0f;
        result.overlapRatio = result.frameVoxels > 0 ? result.historyRehitVoxels / (float)result.frameVoxels : 0f;
        result.historyAgreementRatio = result.historyRehitVoxels > 0 ? result.historyAgreementVoxels / (float)result.historyRehitVoxels : 1f;
        int usefulVoxels = result.newVoxels + result.newStableVoxels + result.rehitVoxels + result.parallaxRehitVoxels;
        bool focusLost = _roomRawCoverageWindowActive && _roomRawCoverageFrames >= 2 && result.focusVoxels <= 0;
        result.observationOrderScore = BuildObservationOrderScore(
            result.validSamples,
            result.focusVoxels,
            usefulVoxels,
            result.riskSamples,
            result.riskRatio,
            legacyRoomRawCoverageDisorderMinValidSamples,
            roomRawCoverageMinFocusVoxels,
            legacyRoomRawCoverageDisorderMinUsefulVoxels,
            legacyRoomRawCoverageDisorderRiskRatio,
            legacyRoomRawCoverageDisorderLowUsefulRiskRatio,
            focusLost,
            autoStopLegacyRoomRawCoverageOnDisorder,
            result.rawObservationOrder);
        result.accepted = ShouldAcceptRoomRawCoverageFrame(result);
        string anchorMode = result.anchorFallback ? " fallback-anchor" : "";
        result.status = result.validSamples <= 0
            ? "no valid raw depth"
            : result.accepted
                ? $"accepted{anchorMode} order={result.observationOrderScore.score01:0.00} {result.validSamples}/{result.totalSamples} valid new={result.newVoxels} overlap={result.overlapRatio:0.00} seam={result.historyAgreementRatio:0.00}"
                : result.observationOrderScore.disordered
                    ? $"rejected observation-order {result.observationOrderScore.reason}"
                    : $"rejected duplicate/low contribution core={result.coreVoxels} focus={result.focusVoxels} edge={result.edgeBufferVoxels} new={result.focusNewVoxels} stable={result.newStableVoxels} overlap={result.overlapRatio:0.00}";
        return result;
    }

    private bool ShouldAcceptRoomRawCoverageFrame(RoomRawCoverageFrameResult result)
    {
        bool useObservationGate = roomRawCoverageDisorderFuseMode != RoomRawCoverageDisorderFuseMode.Off &&
            enforceObservationOrderScoreGate;
        if (useObservationGate && result.observationOrderScore.hasData && result.observationOrderScore.disordered)
            return false;
        if (!gateRoomRawCoverageFrames)
            return result.validSamples > 0;
        if (result.validSamples <= 0)
            return false;

        int minFocusVoxels = Mathf.Max(1, roomRawCoverageMinFocusVoxels);
        int minCoreVoxels = Mathf.Max(1, roomRawCoverageMinCoreVoxels);
        int minEdgeBufferVoxels = Mathf.Max(1, roomRawCoverageMinEdgeBufferVoxels);
        int minHistoryRehitVoxels = Mathf.Max(1, roomRawCoverageMinHistoryRehitVoxelsPerFrame);
        float minHistoryAgreement = Mathf.Clamp01(roomRawCoverageMinHistoryAgreementRatio);
        bool hasNormalFocus = result.focusVoxels >= minFocusVoxels;
        bool hasUsefulSeam = result.historyRehitVoxels >= minHistoryRehitVoxels &&
            result.historyAgreementRatio >= minHistoryAgreement;
        bool hasLargeSurfaceFallback = roomRawCoverageUseFallbackTolerance &&
            result.coreVoxels >= minCoreVoxels &&
            (result.edgeBufferVoxels >= minEdgeBufferVoxels || hasUsefulSeam);
        if (!hasNormalFocus && !hasLargeSurfaceFallback)
            return false;
        if (_roomRawCoverageFrames <= 0)
            return true;

        bool improvesCoverage = result.focusNewVoxels >= Mathf.Max(1, roomRawCoverageMinNewVoxelsPerFrame) ||
            result.newVoxels >= Mathf.Max(1, roomRawCoverageMinNewVoxelsPerFrame * 2) ||
            (hasLargeSurfaceFallback && result.edgeBufferNewVoxels >= Mathf.Max(1, roomRawCoverageMinOverlapNewVoxelsPerFrame));
        bool usefulOverlap = result.rehitVoxels >= Mathf.Max(1, roomRawCoverageMinOverlapRehitVoxelsPerFrame) &&
            (result.focusNewVoxels >= Mathf.Max(1, roomRawCoverageMinOverlapNewVoxelsPerFrame) ||
                result.newStableVoxels >= Mathf.Max(1, roomRawCoverageMinNewStableVoxelsPerFrame / 2) ||
                result.newHighVoxels + result.newLowVoxels >= Mathf.Max(1, roomRawCoverageMinNewHighLowVoxelsPerFrame / 2) ||
                result.focusRiskSamples >= Mathf.Max(1, roomRawCoverageMinRiskSamplesPerFrame / 2));
        bool improvesStable = result.newStableVoxels >= Mathf.Max(1, roomRawCoverageMinNewStableVoxelsPerFrame) ||
            result.parallaxRehitVoxels >= Mathf.Max(1, roomRawCoverageMinParallaxRehitVoxelsPerFrame);
        bool improvesHighLow = result.newHighVoxels + result.newLowVoxels >= Mathf.Max(1, roomRawCoverageMinNewHighLowVoxelsPerFrame) ||
            result.newHighStableVoxels + result.newLowStableVoxels >= Mathf.Max(1, roomRawCoverageMinNewStableVoxelsPerFrame);
        bool improvesRisk = result.coreVoxels >= minCoreVoxels &&
            result.focusRiskSamples >= Mathf.Max(1, roomRawCoverageMinRiskSamplesPerFrame) &&
            result.riskRatio >= Mathf.Max(0.01f, roomRawCoverageTargetRiskRatio * 0.35f);

        return improvesCoverage || usefulOverlap || improvesStable || improvesHighLow || improvesRisk;
    }

    private void ApplyRoomRawCoverage(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Transform pose, int frameIndex, RoomRawCoverageFrameResult result)
    {
        int width = Mathf.Max(1, snapshot.resolutionWidth);
        int height = Mathf.Max(1, snapshot.resolutionHeight);
        int count = Mathf.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length);
        Vector3 cameraPosition = pose != null ? pose.position : Vector3.zero;
        Vector3 cameraForward = pose != null ? pose.forward : Vector3.forward;
        float voxelSize = Mathf.Max(0.005f, roomRawCoverageVoxelSizeMeters);
        float riskAngle = Mathf.Clamp(roomRawCoverageRiskViewAngleDegrees, 0f, 89f);
        float riskFacingMin = Mathf.Cos(riskAngle * Mathf.Deg2Rad);
        float highMinY = cameraPosition.y + Mathf.Max(0.1f, roomRawCoverageHighPointMinDeltaYMeters);
        float lowMaxY = cameraPosition.y - Mathf.Max(0.1f, roomRawCoverageLowPointMinDeltaYMeters);
        float jumpMeters = Mathf.Max(0.001f, roomRawCoverageNeighborDepthJumpMeters);
        int stableHitTarget = Mathf.Max(1, roomRawCoverageStableHitTarget);
        float parallaxMin = allowSameViewStableRoomRawHits && lockRoomRawCoverageToStartView ? 0f : Mathf.Max(0f, roomRawCoverageStableParallaxMeters);
        HashSet<Vector3Int> frameVoxels = new HashSet<Vector3Int>();

        for (int i = 0; i < count; i++)
        {
            if (!IsRawDepthSampleUsable(snapshot, i, cameraPosition, cameraForward))
                continue;

            int x = i % width;
            int y = i / width;
            float u = width > 1 ? x / (float)(width - 1) : 0.5f;
            float v = height > 1 ? y / (float)(height - 1) : 0.5f;
            bool focus = GetRoomRawCoverageFocusFlags(u, v, out _, out _);
            Vector3 point = snapshot.worldPositions[i];
            bool highPoint = point.y >= highMinY;
            bool lowPoint = point.y <= lowMaxY;
            Vector3 normal = (snapshot.worldNormals != null && i < snapshot.worldNormals.Length && IsFinite(snapshot.worldNormals[i]) && snapshot.worldNormals[i].sqrMagnitude > 1e-8f)
                ? snapshot.worldNormals[i].normalized
                : Vector3.up;

            bool risk = IsRawDepthNeighborJump(snapshot, i, width, height, jumpMeters, cameraPosition, cameraForward) ||
                IsRawObservationOrderRisk(result.rawObservationOrder, i);
            Vector3 toCamera = cameraPosition - point;
            if (toCamera.sqrMagnitude > 1e-8f)
            {
                float facing = Mathf.Abs(Vector3.Dot(normal, toCamera.normalized));
                if (facing < riskFacingMin)
                    risk = true;
            }

            Vector3Int key = WorldToVoxelKey(point, voxelSize);
            bool firstVoxelHitThisFrame = frameVoxels.Add(key);
            bool hasLocalVoxel = _roomRawCoverageVoxels.TryGetValue(key, out RoomRawCoverageVoxel voxel) && voxel.frameHits > 0;

            if (!hasLocalVoxel || voxel.frameHits <= 0)
            {
                voxel = default;
                voxel.firstFrame = frameIndex;
                voxel.firstCameraPosition = cameraPosition;
            }

            if (firstVoxelHitThisFrame)
                voxel.frameHits++;
            if (highPoint && !voxel.high)
            {
                voxel.high = true;
                _roomRawCoverageHighVoxels++;
            }
            if (lowPoint && !voxel.low)
            {
                voxel.low = true;
                _roomRawCoverageLowVoxels++;
            }
            voxel.pointHits++;
            if (firstVoxelHitThisFrame)
            {
                voxel.lastFrame = frameIndex;
                voxel.lastCameraPosition = cameraPosition;
            }
            voxel.positionSum += point;
            voxel.normalSum += normal;

            if (risk && !voxel.risk)
            {
                voxel.risk = true;
                _roomRawCoverageRiskVoxels++;
            }

            if (!voxel.stable &&
                voxel.frameHits >= stableHitTarget &&
                (parallaxMin <= 0f || Vector3.Distance(voxel.firstCameraPosition, cameraPosition) >= parallaxMin))
            {
                voxel.stable = true;
                _roomRawCoverageStableVoxels++;
            }
            if (voxel.high && voxel.stable && !voxel.highStable)
            {
                voxel.highStable = true;
                _roomRawCoverageHighStableVoxels++;
            }
            if (voxel.low && voxel.stable && !voxel.lowStable)
            {
                voxel.lowStable = true;
                _roomRawCoverageLowStableVoxels++;
            }

            _roomRawCoverageVoxels[key] = voxel;
            _roomRawCoverageHistoryVoxels[key] = voxel;
            if (_roomRawCoverageWindowActive && focus)
            {
                _roomRawCoverageWindowCoveredKeys.Add(key);
                if (voxel.stable)
                    _roomRawCoverageWindowStableKeys.Add(key);
                RegisterRoomRawCoverageTileHit(u, v, key, voxel.stable);
            }
        }

        RecalculateRoomRawCoverageCounters();
        _roomRawCoverageFrames++;
        _roomRawCoverageTotalSamples += result.totalSamples;
        _roomRawCoverageValidSamples += result.validSamples;
        if (pose != null)
        {
            _roomRawCoveragePosePositionCells.Add(WorldToVoxelKey(pose.position, Mathf.Max(0.05f, roomRawCoveragePoseCellSizeMeters)));
            _roomRawCoveragePoseOrientationCells.Add(new Vector3Int(
                Mathf.FloorToInt((pose.rotation.eulerAngles.y + 180f) / 30f),
                Mathf.FloorToInt((pose.rotation.eulerAngles.x + 180f) / 30f),
                Mathf.FloorToInt((pose.rotation.eulerAngles.z + 180f) / 45f)));
        }
        _roomRawCoveragePoseCellCount = _roomRawCoveragePosePositionCells.Count + _roomRawCoveragePoseOrientationCells.Count;
    }

    private void RecalculateRoomRawCoverageCounters()
    {
        _roomRawCoverageCoveredVoxels = _roomRawCoverageVoxels.Count;
        _roomRawCoverageStableVoxels = 0;
        _roomRawCoverageHighVoxels = 0;
        _roomRawCoverageHighStableVoxels = 0;
        _roomRawCoverageLowVoxels = 0;
        _roomRawCoverageLowStableVoxels = 0;
        _roomRawCoverageRiskVoxels = 0;

        foreach (RoomRawCoverageVoxel voxel in _roomRawCoverageVoxels.Values)
        {
            if (voxel.stable)
                _roomRawCoverageStableVoxels++;
            if (voxel.high)
                _roomRawCoverageHighVoxels++;
            if (voxel.highStable)
                _roomRawCoverageHighStableVoxels++;
            if (voxel.low)
                _roomRawCoverageLowVoxels++;
            if (voxel.lowStable)
                _roomRawCoverageLowStableVoxels++;
            if (voxel.risk)
                _roomRawCoverageRiskVoxels++;
        }
    }

    private static Vector3 GetRoomRawCoverageVoxelAverage(RoomRawCoverageVoxel voxel)
    {
        return voxel.pointHits > 0 ? voxel.positionSum / voxel.pointHits : Vector3.zero;
    }

    private static int CountStableRoomRawCoverageVoxels(Dictionary<Vector3Int, RoomRawCoverageVoxel> voxels)
    {
        if (voxels == null || voxels.Count <= 0)
            return 0;

        int stable = 0;
        foreach (RoomRawCoverageVoxel voxel in voxels.Values)
        {
            if (voxel.stable)
                stable++;
        }
        return stable;
    }

    private void LoadPreviousRoomRawCoverageIfNeeded(string exportRoot)
    {
        if (!trackRoomRawCoverage || !loadPreviousRoomRawCoverageOnStart)
            return;

        List<string> voxelFiles = new List<string>();
        string overridePath = (previousRoomRawCoverageOverrideDirectory ?? "").Trim();
        if (!string.IsNullOrEmpty(overridePath))
        {
            string resolved = ResolveRoomRawCoverageVoxelsPath(overridePath);
            if (!string.IsNullOrEmpty(resolved))
                voxelFiles.Add(resolved);
        }
        else
        {
            string groupDirectory = Path.Combine(exportRoot, sessionGroupDirectoryName);
            if (Directory.Exists(groupDirectory))
            {
                DirectoryInfo groupInfo = new DirectoryInfo(groupDirectory);
                DirectoryInfo[] sessions = groupInfo.GetDirectories($"{sessionNamePrefix}_*");
                Array.Sort(sessions, (a, b) => b.LastWriteTimeUtc.CompareTo(a.LastWriteTimeUtc));
                int maxSessions = Mathf.Max(1, maxPreviousRoomRawCoverageSessionsToLoad);
                for (int i = 0; i < sessions.Length && voxelFiles.Count < maxSessions; i++)
                {
                    if (string.Equals(sessions[i].FullName, _sessionDirectory, StringComparison.OrdinalIgnoreCase))
                        continue;

                    string resolved = ResolveRoomRawCoverageVoxelsPath(sessions[i].FullName);
                    if (!string.IsNullOrEmpty(resolved))
                        voxelFiles.Add(resolved);
                }
            }
        }

        for (int i = 0; i < voxelFiles.Count; i++)
            LoadRoomRawCoverageVoxelsFile(voxelFiles[i]);

        if (_loadedRoomRawCoverageSessions > 0)
        {
            RecalculateRoomRawCoverageCounters();
            _loadedRoomRawCoverageVoxels = _roomRawCoverageHistoryVoxels.Count;
            _loadedRoomRawCoverageStableVoxels = CountStableRoomRawCoverageVoxels(_roomRawCoverageHistoryVoxels);
            Debug.Log($"[ScanCoverMultiFrameSessionExporter] Loaded previous room raw coverage sessions={_loadedRoomRawCoverageSessions} voxels={_loadedRoomRawCoverageVoxels} stable={_loadedRoomRawCoverageStableVoxels} source={_loadedRoomRawCoverageSource}", this);
        }
    }

    private static string ResolveRoomRawCoverageVoxelsPath(string path)
    {
        if (string.IsNullOrEmpty(path))
            return null;

        if (File.Exists(path))
            return path;

        if (!Directory.Exists(path))
            return null;

        string direct = Path.Combine(path, "room_raw_coverage_voxels.csv");
        if (File.Exists(direct))
            return direct;

        string nested = Path.Combine(path, "room_raw_coverage", "room_raw_coverage_voxels.csv");
        return File.Exists(nested) ? nested : null;
    }

    private void LoadRoomRawCoverageVoxelsFile(string path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return;

        int loaded = 0;
        try
        {
            float currentVoxelSize = Mathf.Max(0.005f, roomRawCoverageVoxelSizeMeters);
            bool voxelSizeChecked = false;
            foreach (string line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line) || line[0] == '#')
                    continue;

                if (!voxelSizeChecked && line.StartsWith("voxelSizeMeters=", StringComparison.Ordinal))
                {
                    voxelSizeChecked = true;
                    string value = line.Substring("voxelSizeMeters=".Length);
                    if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float loadedVoxelSize) &&
                        Mathf.Abs(loadedVoxelSize - currentVoxelSize) > 0.001f)
                    {
                        Debug.LogWarning($"[ScanCoverMultiFrameSessionExporter] Skipped previous room raw coverage because voxel size changed. file={FormatFloat(loadedVoxelSize)} current={FormatFloat(currentVoxelSize)} path={path}", this);
                        return;
                    }
                    continue;
                }

                if (line.IndexOf('=') >= 0 || line.StartsWith("voxelX", StringComparison.Ordinal))
                    continue;

                string[] parts = line.Split(',');
                if (parts.Length < 17 ||
                    !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int vx) ||
                    !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int vy) ||
                    !int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int vz) ||
                    !int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out int frameHits) ||
                    !int.TryParse(parts[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out int pointHits) ||
                    !TryParseBoolInt(parts[5], out bool stable) ||
                    !TryParseBoolInt(parts[6], out bool risk) ||
                    !TryParseBoolInt(parts[7], out bool high) ||
                    !TryParseBoolInt(parts[8], out bool highStable) ||
                    !TryParseBoolInt(parts[9], out bool low) ||
                    !TryParseBoolInt(parts[10], out bool lowStable) ||
                    !TryParseVector(parts, 11, out Vector3 avgPosition) ||
                    !TryParseVector(parts, 14, out Vector3 avgNormal))
                    continue;

                pointHits = Mathf.Max(1, pointHits);
                frameHits = Mathf.Max(1, frameHits);
                RoomRawCoverageVoxel loadedVoxel = default;
                loadedVoxel.frameHits = frameHits;
                loadedVoxel.pointHits = pointHits;
                loadedVoxel.firstFrame = 0;
                loadedVoxel.lastFrame = 0;
                loadedVoxel.stable = stable;
                loadedVoxel.risk = risk;
                loadedVoxel.high = high;
                loadedVoxel.highStable = highStable;
                loadedVoxel.low = low;
                loadedVoxel.lowStable = lowStable;
                loadedVoxel.positionSum = avgPosition * pointHits;
                loadedVoxel.normalSum = avgNormal.sqrMagnitude > 1e-8f ? avgNormal.normalized * pointHits : Vector3.up * pointHits;
                loadedVoxel.firstCameraPosition = Vector3.zero;
                loadedVoxel.lastCameraPosition = Vector3.zero;

                Vector3Int key = new Vector3Int(vx, vy, vz);
                MergeLoadedRoomRawCoverageVoxel(key, loadedVoxel);
                loaded++;
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ScanCoverMultiFrameSessionExporter] Failed to load previous room raw coverage: {path}\n{ex.Message}", this);
            return;
        }

        if (loaded <= 0)
            return;

        _loadedRoomRawCoverageSessions++;
        string parentDirectory = Path.GetDirectoryName(path);
        string sessionDirectory = parentDirectory;
        if (!string.IsNullOrEmpty(parentDirectory) &&
            string.Equals(Path.GetFileName(parentDirectory), "room_raw_coverage", StringComparison.OrdinalIgnoreCase))
            sessionDirectory = Path.GetDirectoryName(parentDirectory);
        _loadedRoomRawCoverageSource = !string.IsNullOrEmpty(sessionDirectory) ? Path.GetFileName(sessionDirectory) : Path.GetFileName(path);
    }

    private void MergeLoadedRoomRawCoverageVoxel(Vector3Int key, RoomRawCoverageVoxel loadedVoxel)
    {
        if (_roomRawCoverageHistoryVoxels.TryGetValue(key, out RoomRawCoverageVoxel existing) && existing.frameHits > 0)
        {
            existing.frameHits += loadedVoxel.frameHits;
            existing.pointHits += loadedVoxel.pointHits;
            existing.stable |= loadedVoxel.stable;
            existing.risk |= loadedVoxel.risk;
            existing.high |= loadedVoxel.high;
            existing.highStable |= loadedVoxel.highStable;
            existing.low |= loadedVoxel.low;
            existing.lowStable |= loadedVoxel.lowStable;
            existing.positionSum += loadedVoxel.positionSum;
            existing.normalSum += loadedVoxel.normalSum;
            _roomRawCoverageHistoryVoxels[key] = existing;
            return;
        }

        _roomRawCoverageHistoryVoxels[key] = loadedVoxel;
    }

    private static bool TryParseBoolInt(string value, out bool result)
    {
        value = (value ?? "").Trim();
        if (value == "1")
        {
            result = true;
            return true;
        }
        if (value == "0")
        {
            result = false;
            return true;
        }
        return bool.TryParse(value, out result);
    }

    private bool IsRawDepthSampleUsable(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, int index, Vector3 cameraPosition, Vector3 cameraForward)
    {
        if (snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null ||
            index < 0 || index >= snapshot.worldPositions.Length || index >= snapshot.observationMeta.Length)
            return false;

        Color meta = snapshot.observationMeta[index];
        if (meta.r < 0.5f)
            return false;
        if (meta.b <= 0f || !float.IsFinite(meta.b))
            return false;
        float minDepth = Mathf.Max(0.01f, roomRawCoverageMinDepthMeters);
        float maxDepth = Mathf.Max(minDepth + 0.01f, roomRawCoverageMaxDepthMeters);
        if (meta.b < minDepth || meta.b > maxDepth)
            return false;

        Vector3 point = snapshot.worldPositions[index];
        if (!IsFinite(point))
            return false;

        if (!IsFinite(cameraPosition) || !IsFinite(cameraForward) || cameraForward.sqrMagnitude <= 1e-8f)
            return true;

        Vector3 toPoint = point - cameraPosition;
        float forwardDepth = Vector3.Dot(toPoint, cameraForward.normalized);
        if (!float.IsFinite(forwardDepth) || forwardDepth < minDepth || forwardDepth > maxDepth)
            return false;

        float maxProjectionError = Mathf.Max(0.01f, roomRawCoverageMaxDepthProjectionErrorMeters);
        if (Mathf.Abs(forwardDepth - meta.b) > maxProjectionError)
            return false;

        return true;
    }

    private bool IsRawDepthSnapshotPixelValid(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, int index)
    {
        if (snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null ||
            index < 0 || index >= snapshot.worldPositions.Length || index >= snapshot.observationMeta.Length)
            return false;

        Color meta = snapshot.observationMeta[index];
        if (meta.r < 0.5f || meta.b <= 0f || !float.IsFinite(meta.b))
            return false;

        float minDepth = Mathf.Max(0.01f, roomRawCoverageMinDepthMeters);
        float maxDepth = Mathf.Max(minDepth + 0.01f, roomRawCoverageMaxDepthMeters);
        if (meta.b < minDepth || meta.b > maxDepth)
            return false;

        return IsFinite(snapshot.worldPositions[index]);
    }

    private bool IsRawDepthNeighborJump(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, int index, int width, int height, float thresholdMeters, Vector3 cameraPosition, Vector3 cameraForward)
    {
        int x = index % width;
        int y = index / width;
        float depth = snapshot.observationMeta[index].b;
        if (x + 1 < width)
        {
            int right = index + 1;
            if (right < snapshot.observationMeta.Length && IsRawDepthSampleUsable(snapshot, right, cameraPosition, cameraForward) &&
                Mathf.Abs(snapshot.observationMeta[right].b - depth) >= thresholdMeters)
                return true;
        }
        if (y + 1 < height)
        {
            int up = index + width;
            if (up < snapshot.observationMeta.Length && IsRawDepthSampleUsable(snapshot, up, cameraPosition, cameraForward) &&
                Mathf.Abs(snapshot.observationMeta[up].b - depth) >= thresholdMeters)
                return true;
        }
        return false;
    }

    private RawObservationOrderResult EvaluateRawObservationOrder(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Vector3 cameraPosition, Vector3 cameraForward)
    {
        RawObservationOrderResult result = default;
        if (!useRawObservationOrderScore || snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null)
            return result;

        int width = Mathf.Max(1, snapshot.resolutionWidth);
        int height = Mathf.Max(1, snapshot.resolutionHeight);
        int count = Mathf.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length);
        if (count <= 0)
            return result;
        int sampleStride = Mathf.Clamp(rawObservationOrderPixelStride, 1, 8);

        result.sampleStates = new byte[count];
        for (int y = 0; y < height; y += sampleStride)
        {
            for (int x = 0; x < width; x += sampleStride)
            {
                int i = x + y * width;
                if (i >= count || !IsRawDepthSampleUsable(snapshot, i, cameraPosition, cameraForward))
                    continue;

                result.validSamples++;
                result.sampleStates[i] = 1;
            }
        }

        if (result.validSamples <= 0)
        {
            result.reason = "no-valid-raw-order-samples";
            return result;
        }

        float depthJump = Mathf.Max(0.001f, rawObservationOrderDepthJumpMeters);
        if (roomRawCoverageNeighborDepthJumpMeters > 0f)
            depthJump = Mathf.Min(depthJump, Mathf.Max(0.001f, roomRawCoverageNeighborDepthJumpMeters));
        float maxNeighborDistance = Mathf.Max(0.001f, rawObservationOrderMaxNeighborDistanceMeters);
        float minNormalDot = Mathf.Clamp(rawObservationOrderMinNormalDot, -1f, 1f);
        float quadSkew = Mathf.Max(0.001f, rawObservationOrderQuadMaxDiagonalSkewMeters);

        for (int y = 0; y < height; y += sampleStride)
        {
            for (int x = 0; x < width; x += sampleStride)
            {
                int index = x + y * width;
                if (index >= count || result.sampleStates[index] == 0)
                    continue;

                if (x + sampleStride < width)
                    EvaluateRawObservationEdge(snapshot, index, index + sampleStride, width, height, maxNeighborDistance * sampleStride, depthJump, minNormalDot, ref result);
                if (y + sampleStride < height)
                    EvaluateRawObservationEdge(snapshot, index, index + width * sampleStride, width, height, maxNeighborDistance * sampleStride, depthJump, minNormalDot, ref result);

                if (x + sampleStride < width && y + sampleStride < height)
                {
                    int right = index + sampleStride;
                    int up = index + width * sampleStride;
                    int upRight = up + sampleStride;
                    if (upRight < count &&
                        result.sampleStates[right] != 0 &&
                        result.sampleStates[up] != 0 &&
                        result.sampleStates[upRight] != 0)
                    {
                        result.testedQuads++;
                        float d0 = Vector3.Distance(snapshot.worldPositions[index], snapshot.worldPositions[upRight]);
                        float d1 = Vector3.Distance(snapshot.worldPositions[right], snapshot.worldPositions[up]);
                        if (!float.IsFinite(d0) || !float.IsFinite(d1) || Mathf.Abs(d0 - d1) >= quadSkew)
                        {
                            result.badQuads++;
                            MarkRawObservationState(result.sampleStates, index, 3);
                            MarkRawObservationState(result.sampleStates, right, 3);
                            MarkRawObservationState(result.sampleStates, up, 3);
                            MarkRawObservationState(result.sampleStates, upRight, 3);
                        }
                    }
                }
            }
        }

        result.hasData = result.validSamples > 0 && result.testedEdges > 0;
        result.orderedEdges = Mathf.Max(0, result.testedEdges - result.badEdges);
        result.orderRatio = result.testedEdges > 0 ? Mathf.Clamp01(result.orderedEdges / (float)result.testedEdges) : 0f;
        result.centerOrderRatio = result.centerTestedEdges > 0
            ? Mathf.Clamp01((result.centerTestedEdges - result.centerBadEdges) / (float)result.centerTestedEdges)
            : result.orderRatio;
        result.badEdgeRatio = result.testedEdges > 0 ? Mathf.Clamp01(result.badEdges / (float)result.testedEdges) : 0f;
        result.badQuadRatio = result.testedQuads > 0 ? Mathf.Clamp01(result.badQuads / (float)result.testedQuads) : 0f;
        result.reason = $"edges {result.orderedEdges}/{result.testedEdges} bad={result.badEdges} quads={result.badQuads}/{result.testedQuads}";
        return result;
    }

    private void EvaluateRawObservationEdge(
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot,
        int a,
        int b,
        int width,
        int height,
        float maxNeighborDistance,
        float depthJump,
        float minNormalDot,
        ref RawObservationOrderResult result)
    {
        if (result.sampleStates == null || a < 0 || b < 0 || a >= result.sampleStates.Length || b >= result.sampleStates.Length)
            return;
        if (result.sampleStates[a] == 0 || result.sampleStates[b] == 0)
            return;

        result.testedEdges++;
        float uA = width > 1 ? (a % width) / (float)(width - 1) : 0.5f;
        float vA = height > 1 ? (a / width) / (float)(height - 1) : 0.5f;
        float uB = width > 1 ? (b % width) / (float)(width - 1) : 0.5f;
        float vB = height > 1 ? (b / width) / (float)(height - 1) : 0.5f;
        bool centerEdge = GetRoomRawCoverageFocusFlags(uA, vA, out _, out _) || GetRoomRawCoverageFocusFlags(uB, vB, out _, out _);
        if (centerEdge)
            result.centerTestedEdges++;

        Vector3 pa = snapshot.worldPositions[a];
        Vector3 pb = snapshot.worldPositions[b];
        float distance = Vector3.Distance(pa, pb);
        float da = snapshot.observationMeta[a].b;
        float db = snapshot.observationMeta[b].b;
        bool badDistance = !float.IsFinite(distance) || distance >= maxNeighborDistance;
        bool badDepth = Mathf.Abs(da - db) >= depthJump;
        bool badNormal = false;
        if (snapshot.worldNormals != null && a < snapshot.worldNormals.Length && b < snapshot.worldNormals.Length)
        {
            Vector3 na = snapshot.worldNormals[a];
            Vector3 nb = snapshot.worldNormals[b];
            if (IsFinite(na) && IsFinite(nb) && na.sqrMagnitude > 1e-8f && nb.sqrMagnitude > 1e-8f)
                badNormal = Vector3.Dot(na.normalized, nb.normalized) < minNormalDot;
        }

        if (!badDistance && !badDepth && !badNormal)
            return;

        result.badEdges++;
        if (centerEdge)
            result.centerBadEdges++;
        if (badDistance)
            result.badDistanceEdges++;
        if (badDepth)
            result.badDepthEdges++;
        if (badNormal)
            result.badNormalEdges++;
        MarkRawObservationState(result.sampleStates, a, 3);
        MarkRawObservationState(result.sampleStates, b, 3);
    }

    private static void MarkRawObservationState(byte[] states, int index, byte state)
    {
        if (states == null || index < 0 || index >= states.Length)
            return;
        if (state > states[index])
            states[index] = state;
    }

    private static bool IsRawObservationOrderRisk(RawObservationOrderResult result, int index)
    {
        return result.sampleStates != null && index >= 0 && index < result.sampleStates.Length && result.sampleStates[index] >= 3;
    }

    private static byte GetRawObservationOrderState(RawObservationOrderResult result, int index)
    {
        if (result.sampleStates == null || index < 0 || index >= result.sampleStates.Length)
            return 0;
        return result.sampleStates[index];
    }

    private bool ShouldUseSnapshotGridCaptureMask()
    {
        return useSnapshotGridCaptureMask && depthGridPointCloud != null && !ShouldUseLegacyFullViewRoomRawCoverage();
    }

    private void UpdateSnapshotGridCapturePreview()
    {
        if (_isRecording)
            return;

        if (!ShouldUseSnapshotGridCaptureMask())
        {
            if (_snapshotGridMaskActive)
                ClearSnapshotGridCaptureMask(true);
            _snapshotGridMaskStatus = "\u5173\u95ed";
            return;
        }

        PrepareSnapshotGridCaptureMaskForSession(false);
    }

    private void PrepareSnapshotGridCaptureMaskForSession(bool forceReset = false)
    {
        if (!ShouldUseSnapshotGridCaptureMask())
        {
            ClearSnapshotGridCaptureMask(true);
            _snapshotGridMaskStatus = "\u5173\u95ed";
            return;
        }

        if (BuildSnapshotGridCaptureMask(forceReset))
        {
            if (forceReset || _snapshotGridMaskStatus == "inactive" || _snapshotGridMaskStatus == "\u5173\u95ed" || _snapshotGridMaskStatus.StartsWith("\u6ca1\u6709", StringComparison.Ordinal))
                _snapshotGridMaskStatus = "\u5df2\u9501\u5b9a\u5feb\u7167\u70b9\u9635";
            return;
        }

        if (!_snapshotGridMaskActive)
            _snapshotGridMaskStatus = snapshotGridMaskRequiredForRawDepthExport ? "\u6ca1\u6709\u5feb\u7167\u70b9\u9635\uff0cRaw Depth \u6682\u505c" : "\u6ca1\u6709\u5feb\u7167\u70b9\u9635\uff0c\u5141\u8bb8\u5168\u91cf Raw Depth";
    }

    private void ClearSnapshotGridCaptureMask(bool restoreColors)
    {
        if (depthGridPointCloud != null)
        {
            depthGridPointCloud.SetSnapshotGridExternalControlActive(false);
            if (restoreColors)
                depthGridPointCloud.RestoreSnapshotGridPointColors();
        }

        _snapshotGridMaskCells.Clear();
        _snapshotGridMaskPositions = new Vector3[0];
        _snapshotGridMaskEntryIndices = new int[0];
        _snapshotGridMaskHits = new int[0];
        _snapshotGridMaskRiskHits = new int[0];
        _snapshotGridMaskVisualStates = new byte[0];
        _snapshotGridMaskTotal = 0;
        _snapshotGridMaskSeen = 0;
        _snapshotGridMaskStable = 0;
        _snapshotGridMaskRisk = 0;
        _snapshotGridMaskActive = false;
        _snapshotGridMaskHasSignature = false;
        _snapshotGridMaskSignature = 0;
    }

    private void HideSnapshotGridCaptureMask()
    {
        if (!_snapshotGridMaskActive)
            return;

        if (_snapshotGridMaskHasSignature)
        {
            _snapshotGridMaskSuppressUntilSignatureChanges = true;
            _snapshotGridMaskSuppressedSignature = _snapshotGridMaskSignature;
        }

        ClearSnapshotGridCaptureMask(false);
        if (depthGridPointCloud != null)
            depthGridPointCloud.SetPreviewDisplayVisible(false);
    }

    private bool BuildSnapshotGridCaptureMask(bool forceReset = false)
    {
        if (depthGridPointCloud == null || !depthGridPointCloud.TryGetCurrentGridState(out ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot) || snapshot == null || snapshot.entries == null)
            return false;

        int signature = ComputeSnapshotGridCaptureSignature(snapshot);
        if (_snapshotGridMaskSuppressUntilSignatureChanges)
        {
            if (signature == _snapshotGridMaskSuppressedSignature)
            {
                _snapshotGridMaskStatus = "閲囬泦瀹屾垚锛岀瓑寰呮柊蹇収";
                return false;
            }

            _snapshotGridMaskSuppressUntilSignatureChanges = false;
        }

        if (!forceReset && _snapshotGridMaskActive && _snapshotGridMaskHasSignature && _snapshotGridMaskSignature == signature)
        {
            depthGridPointCloud.SetSnapshotGridExternalControlActive(true);
            depthGridPointCloud.SetPreviewDisplayVisible(true);
            depthGridPointCloud.SetCaptureOnlySnapshotMode(false);
            return true;
        }

        List<Vector3> positions = new List<Vector3>(snapshot.entries.Length);
        List<int> indices = new List<int>(snapshot.entries.Length);
        Dictionary<Vector3Int, List<int>> cells = new Dictionary<Vector3Int, List<int>>();
        for (int i = 0; i < snapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
            if (!entry.valid || !IsFinite(entry.worldPos))
                continue;

            int slot = positions.Count;
            positions.Add(entry.worldPos);
            indices.Add(entry.index);

            Vector3Int cell = SnapshotGridCaptureCell(entry.worldPos);
            if (!cells.TryGetValue(cell, out List<int> slots))
            {
                slots = new List<int>(4);
                cells[cell] = slots;
            }
            slots.Add(slot);
        }

        if (positions.Count <= 0)
            return false;

        ClearSnapshotGridCaptureMask(true);
        foreach (KeyValuePair<Vector3Int, List<int>> pair in cells)
            _snapshotGridMaskCells[pair.Key] = pair.Value;

        _snapshotGridMaskPositions = positions.ToArray();
        _snapshotGridMaskEntryIndices = indices.ToArray();
        _snapshotGridMaskHits = new int[_snapshotGridMaskPositions.Length];
        _snapshotGridMaskRiskHits = new int[_snapshotGridMaskPositions.Length];
        _snapshotGridMaskVisualStates = new byte[_snapshotGridMaskPositions.Length];
        _snapshotGridMaskTotal = _snapshotGridMaskPositions.Length;
        _snapshotGridMaskActive = true;
        _snapshotGridMaskHasSignature = true;
        _snapshotGridMaskSignature = signature;
        _snapshotGridMaskSuppressUntilSignatureChanges = false;

        depthGridPointCloud.SetSnapshotGridExternalControlActive(true);
        depthGridPointCloud.SetPreviewDisplayVisible(true);
        depthGridPointCloud.SetCaptureOnlySnapshotMode(false);

        RefreshSnapshotGridCaptureCounts();
        return true;
    }

    private int ComputeSnapshotGridCaptureSignature(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
    {
        unchecked
        {
            int hash = 17;
            int validCount = 0;
            if (snapshot.entries != null)
            {
                for (int i = 0; i < snapshot.entries.Length; i++)
                {
                    ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
                    if (!entry.valid || !IsFinite(entry.worldPos))
                        continue;

                    validCount++;
                    Vector3 p = entry.worldPos;
                    hash = hash * 31 + entry.index;
                    hash = hash * 31 + Mathf.RoundToInt(p.x * 1000f);
                    hash = hash * 31 + Mathf.RoundToInt(p.y * 1000f);
                    hash = hash * 31 + Mathf.RoundToInt(p.z * 1000f);
                }
            }

            hash = hash * 31 + validCount;
            return hash;
        }
    }
    private Vector3Int SnapshotGridCaptureCell(Vector3 point)
    {
        float size = Mathf.Max(0.01f, snapshotGridCaptureRadiusMeters);
        return new Vector3Int(
            Mathf.FloorToInt(point.x / size),
            Mathf.FloorToInt(point.y / size),
            Mathf.FloorToInt(point.z / size));
    }

    private bool TryFindSnapshotGridMaskPoint(Vector3 point, out int slot, out float sqrDistance)
    {
        slot = -1;
        sqrDistance = float.PositiveInfinity;
        if (!_snapshotGridMaskActive || _snapshotGridMaskPositions == null || _snapshotGridMaskPositions.Length <= 0 || !IsFinite(point))
            return false;

        float radius = Mathf.Max(0.01f, snapshotGridCaptureRadiusMeters);
        float radiusSqr = radius * radius;
        Vector3Int center = SnapshotGridCaptureCell(point);
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            Vector3Int cell = new Vector3Int(center.x + dx, center.y + dy, center.z + dz);
            if (!_snapshotGridMaskCells.TryGetValue(cell, out List<int> slots))
                continue;

            for (int i = 0; i < slots.Count; i++)
            {
                int candidate = slots[i];
                if (candidate < 0 || candidate >= _snapshotGridMaskPositions.Length)
                    continue;

                float d = (point - _snapshotGridMaskPositions[candidate]).sqrMagnitude;
                if (d < sqrDistance)
                {
                    sqrDistance = d;
                    slot = candidate;
                }
            }
        }

        return slot >= 0 && sqrDistance <= radiusSqr;
    }

    private bool AcceptSnapshotGridRawDepthSample(Vector3 point, bool rawRisk, out int slot)
    {
        slot = -1;
        if (!ShouldUseSnapshotGridCaptureMask())
            return true;

        if (!_snapshotGridMaskActive)
            return !snapshotGridMaskRequiredForRawDepthExport;

        if (!TryFindSnapshotGridMaskPoint(point, out slot, out float sqrDistance))
            return false;

        float radius = Mathf.Max(0.01f, snapshotGridCaptureRadiusMeters);
        bool distanceRisk = sqrDistance > radius * radius * 0.72f;
        RegisterSnapshotGridCaptureHit(slot, rawRisk || distanceRisk);
        return true;
    }

    private void RegisterSnapshotGridCaptureHit(int slot, bool risk)
    {
        if (slot < 0 || slot >= _snapshotGridMaskHits.Length)
            return;

        _snapshotGridMaskHits[slot]++;
        if (risk)
            _snapshotGridMaskRiskHits[slot]++;
        UpdateSnapshotGridCaptureVisual(slot);
    }

    private void UpdateSnapshotGridCaptureVisual(int slot)
    {
        if (depthGridPointCloud == null || slot < 0 || slot >= _snapshotGridMaskVisualStates.Length || slot >= _snapshotGridMaskEntryIndices.Length)
            return;

        int hits = _snapshotGridMaskHits[slot];
        int riskHits = _snapshotGridMaskRiskHits[slot];
        float riskRatio = hits > 0 ? riskHits / (float)hits : 0f;
        ObservationOrderScore score = BuildObservationOrderScore(
            hits,
            hits,
            hits,
            riskHits,
            riskRatio,
            1,
            snapshotGridStableHits,
            snapshotGridSeenHits,
            snapshotGridRiskRatio,
            snapshotGridRiskRatio,
            false,
            false);
        byte state = score.representationState;

        if (_snapshotGridMaskVisualStates[slot] == state)
            return;

        _snapshotGridMaskVisualStates[slot] = state;
        depthGridPointCloud.TrySetSnapshotGridPointColor(_snapshotGridMaskEntryIndices[slot], GetObservationOrderScoreColor(score));
    }

    private void RefreshSnapshotGridCaptureCounts()
    {
        _snapshotGridMaskSeen = 0;
        _snapshotGridMaskStable = 0;
        _snapshotGridMaskRisk = 0;
        if (_snapshotGridMaskHits == null || _snapshotGridMaskHits.Length <= 0)
            return;

        int seenHits = Mathf.Max(1, snapshotGridSeenHits);
        int stableHits = Mathf.Max(1, snapshotGridStableHits);
        for (int i = 0; i < _snapshotGridMaskHits.Length; i++)
        {
            int hits = _snapshotGridMaskHits[i];
            int riskHits = _snapshotGridMaskRiskHits[i];
            float riskRatio = hits > 0 ? riskHits / (float)hits : 0f;
            ObservationOrderScore score = BuildObservationOrderScore(
                hits,
                hits,
                hits,
                riskHits,
                riskRatio,
                1,
                stableHits,
                seenHits,
                snapshotGridRiskRatio,
                snapshotGridRiskRatio,
                false,
                false);
            if (score.representationState >= 1)
                _snapshotGridMaskSeen++;
            if (score.representationState == 2)
                _snapshotGridMaskStable++;
            if (score.representationState == 3)
                _snapshotGridMaskRisk++;
        }
    }

    private void AppendSnapshotGridMaskHudLine(StringBuilder builder)
    {
        if (!ShouldUseSnapshotGridCaptureMask())
            return;

        RefreshSnapshotGridCaptureCounts();
        builder.Append("\u5feb\u7167\u70b9\u9635\u91c7\u96c6: ").Append(_snapshotGridMaskStatus);
        if (_snapshotGridMaskTotal > 0)
        {
            builder.Append(" \u7a33\u5b9a ").Append(_snapshotGridMaskStable).Append('/').Append(_snapshotGridMaskTotal)
                .Append(" \u5df2\u89c1 ").Append(_snapshotGridMaskSeen).Append('/').Append(_snapshotGridMaskTotal)
                .Append(" \u98ce\u9669 ").Append(_snapshotGridMaskRisk);
        }
        builder.AppendLine();
    }

    private void ResetRoomRawDepthCompletionOverlay()
    {
        _roomRawDepthCompletionVoxels.Clear();
        _questCloneGuideVoxels.Clear();
        _roomRawDepthCompletionGrayCount = 0;
        _roomRawDepthCompletionBlueCount = 0;
        _roomRawDepthCompletionGreenCount = 0;
        _roomRawDepthCompletionYellowCount = 0;
        _roomRawDepthCompletionRedCount = 0;
        _roomRawDepthCompletionHint = "Waiting";
        _nextRoomRawDepthCompletionRefreshTime = 0f;

        for (int i = 0; i < _roomRawDepthCompletionPointObjects.Count; i++)
        {
            if (_roomRawDepthCompletionPointObjects[i] != null)
                _roomRawDepthCompletionPointObjects[i].SetActive(false);
        }
    }

    private void AddRoomRawDepthCompletionSample(Vector3 point, Vector3 cameraPosition, float depth, bool risk)
    {
        if (!ShouldTrackRoomRawDepthCompletion() || !IsFinite(point) || float.IsNaN(depth) || float.IsInfinity(depth))
            return;

        Vector3Int key = GetRoomRawDepthCompletionKey(point);
        if (!_roomRawDepthCompletionVoxels.TryGetValue(key, out RoomRawDepthCompletionVoxel voxel))
        {
            if (questCloneContinuousCaptureMode && _roomRawDepthCompletionVoxels.Count >= Mathf.Max(1000, questCloneMaxTrackedVoxels))
                return;
            voxel = new RoomRawDepthCompletionVoxel();
            _roomRawDepthCompletionVoxels.Add(key, voxel);
        }

        voxel.Add(point, cameraPosition, depth, risk, questCloneContinuousCaptureMode ? questCloneProgressRiskEmaAlpha : 1f);
    }

    private Vector3Int GetRoomRawDepthCompletionKey(Vector3 point)
    {
        float size = Mathf.Max(0.001f, roomRawDepthCompletionVoxelSizeMeters);
        return new Vector3Int(
            Mathf.FloorToInt(point.x / size),
            Mathf.FloorToInt(point.y / size),
            Mathf.FloorToInt(point.z / size));
    }

    private Vector3 GetRoomRawDepthCompletionDisplayPosition(Vector3Int key, RoomRawDepthCompletionVoxel voxel)
    {
        if (!roomRawDepthCompletionUseVoxelCenterPresentation || voxel == null)
            return voxel != null ? voxel.AveragePosition : Vector3.zero;

        float size = Mathf.Max(0.001f, roomRawDepthCompletionVoxelSizeMeters);
        return new Vector3(
            (key.x + 0.5f) * size,
            (key.y + 0.5f) * size,
            (key.z + 0.5f) * size);
    }

    private Vector3 GetQuestCloneGuideDisplayPosition(Vector3Int key)
    {
        if (!roomRawDepthCompletionUseVoxelCenterPresentation && _questCloneGuideVoxels.TryGetValue(key, out Vector3 position))
            return position;

        float size = Mathf.Max(0.001f, roomRawDepthCompletionVoxelSizeMeters);
        return new Vector3(
            (key.x + 0.5f) * size,
            (key.y + 0.5f) * size,
            (key.z + 0.5f) * size);
    }

    private void SeedQuestCloneGuideFromProductionGrid()
    {
        if (!questCloneContinuousCaptureMode || depthGridPointCloud == null)
            return;
        if (depthGridPointCloud.CopyCurrentValidGridPositions(_questCloneGuideScratchPositions) <= 0)
            return;

        int maxTracked = Mathf.Max(1000, questCloneMaxTrackedVoxels);
        for (int i = 0; i < _questCloneGuideScratchPositions.Count && _questCloneGuideVoxels.Count < maxTracked; i++)
        {
            Vector3 position = _questCloneGuideScratchPositions[i];
            Vector3Int key = GetRoomRawDepthCompletionKey(position);
            if (!_questCloneGuideVoxels.ContainsKey(key))
                _questCloneGuideVoxels.Add(key, position);
        }
    }

    private void UpdateRoomRawDepthCompletionOverlay()
    {
        if (!ShouldTrackRoomRawDepthCompletion())
        {
            SetRoomRawDepthCompletionOverlayVisible(false);
            return;
        }

        if (questCloneContinuousCaptureMode && _isRecording)
            SeedQuestCloneGuideFromProductionGrid();

        bool shouldShow = (_roomRawDepthCompletionVoxels.Count > 0 || _questCloneGuideVoxels.Count > 0) &&
            (questCloneContinuousCaptureMode ? _isRecording : (!roomRawDepthCompletionOnlyWhileRecording || _isRecording));
        if (!shouldShow)
        {
            SetRoomRawDepthCompletionOverlayVisible(false);
            return;
        }

        float now = Time.unscaledTime;
        if (_roomRawDepthCompletionRoot != null && now < _nextRoomRawDepthCompletionRefreshTime)
            return;

        _nextRoomRawDepthCompletionRefreshTime = now + Mathf.Max(0.05f, roomRawDepthCompletionRefreshSeconds);
        EnsureRoomRawDepthCompletionOverlay();
        SetRoomRawDepthCompletionOverlayVisible(true);

        _roomRawDepthCompletionGrayCount = 0;
        _roomRawDepthCompletionBlueCount = 0;
        _roomRawDepthCompletionGreenCount = 0;
        _roomRawDepthCompletionYellowCount = 0;
        _roomRawDepthCompletionRedCount = 0;

        int total = _questCloneGuideVoxels.Count;
        foreach (KeyValuePair<Vector3Int, RoomRawDepthCompletionVoxel> pair in _roomRawDepthCompletionVoxels)
        {
            if (!_questCloneGuideVoxels.ContainsKey(pair.Key))
                total++;
        }
        int maxVisual = Mathf.Max(1, roomRawDepthCompletionMaxVisualPoints);
        int stride = Mathf.Max(1, Mathf.CeilToInt(total / (float)maxVisual));
        int sourceIndex = 0;
        int visualIndex = 0;
        ClearRoomRawDepthCompletionMatrices();

        foreach (KeyValuePair<Vector3Int, Vector3> guide in _questCloneGuideVoxels)
        {
            _roomRawDepthCompletionVoxels.TryGetValue(guide.Key, out RoomRawDepthCompletionVoxel voxel);
            RoomRawDepthCompletionState state = ClassifyRoomRawDepthCompletionVoxel(guide.Key, voxel);
            CountRoomRawDepthCompletionState(state);

            if (sourceIndex % stride == 0)
            {
                Vector3 position = voxel != null
                    ? GetRoomRawDepthCompletionDisplayPosition(guide.Key, voxel)
                    : GetQuestCloneGuideDisplayPosition(guide.Key);
                if (roomRawDepthCompletionUseGpuInstancing)
                    AddRoomRawDepthCompletionMatrix(state, position);
                else
                {
                    GameObject pointObject = GetOrCreateRoomRawDepthCompletionPoint(visualIndex);
                    ApplyRoomRawDepthCompletionPoint(pointObject, position, GetRoomRawDepthCompletionColor(state));
                    visualIndex++;
                }
            }

            sourceIndex++;
        }

        foreach (KeyValuePair<Vector3Int, RoomRawDepthCompletionVoxel> pair in _roomRawDepthCompletionVoxels)
        {
            if (_questCloneGuideVoxels.ContainsKey(pair.Key))
                continue;
            RoomRawDepthCompletionState state = ClassifyRoomRawDepthCompletionVoxel(pair.Key, pair.Value);
            CountRoomRawDepthCompletionState(state);

            if (sourceIndex % stride == 0)
            {
                Vector3 position = GetRoomRawDepthCompletionDisplayPosition(pair.Key, pair.Value);
                if (roomRawDepthCompletionUseGpuInstancing)
                    AddRoomRawDepthCompletionMatrix(state, position);
                else
                {
                    GameObject pointObject = GetOrCreateRoomRawDepthCompletionPoint(visualIndex);
                    ApplyRoomRawDepthCompletionPoint(pointObject, position, GetRoomRawDepthCompletionColor(state));
                    visualIndex++;
                }
            }

            sourceIndex++;
        }

        if (questCloneContinuousCaptureMode)
            RebuildRoomRawDepthCompletionCombinedMesh();

        int firstHiddenIndex = roomRawDepthCompletionUseGpuInstancing ? 0 : visualIndex;
        for (int i = firstHiddenIndex; i < _roomRawDepthCompletionPointObjects.Count; i++)
        {
            if (_roomRawDepthCompletionPointObjects[i] != null)
                _roomRawDepthCompletionPointObjects[i].SetActive(false);
        }

        UpdateRoomRawDepthCompletionHint(total);
    }

    private void ClearRoomRawDepthCompletionMatrices()
    {
        for (int i = 0; i < _roomRawDepthCompletionMatrices.Length; i++)
            _roomRawDepthCompletionMatrices[i].Clear();
    }

    private void RebuildRoomRawDepthCompletionCombinedMesh()
    {
        EnsureRoomRawDepthCompletionOverlay();
        EnsureRoomRawDepthCompletionCombinedRenderer();
        if (_roomRawDepthCompletionCombinedMesh == null || _roomRawDepthCompletionCombinedMeshRenderer == null)
            return;

        _roomRawDepthCompletionCombinedVertices.Clear();
        for (int i = 0; i < _roomRawDepthCompletionCombinedIndices.Length; i++)
            _roomRawDepthCompletionCombinedIndices[i].Clear();

        Transform pose = ResolvePoseTransform();
        if (pose == null)
        {
            _roomRawDepthCompletionCombinedMeshRenderer.enabled = false;
            return;
        }

        Vector3 right = pose.right * (Mathf.Max(0.002f, roomRawDepthCompletionPointSize) * 0.5f);
        Vector3 up = pose.up * (Mathf.Max(0.002f, roomRawDepthCompletionPointSize) * 0.5f);
        for (int stateIndex = 0; stateIndex < _roomRawDepthCompletionMatrices.Length; stateIndex++)
        {
            List<Matrix4x4> matrices = _roomRawDepthCompletionMatrices[stateIndex];
            List<int> indices = _roomRawDepthCompletionCombinedIndices[stateIndex];
            for (int i = 0; i < matrices.Count; i++)
            {
                Vector4 column = matrices[i].GetColumn(3);
                Vector3 center = new Vector3(column.x, column.y, column.z);
                int first = _roomRawDepthCompletionCombinedVertices.Count;
                _roomRawDepthCompletionCombinedVertices.Add(center - right - up);
                _roomRawDepthCompletionCombinedVertices.Add(center + right - up);
                _roomRawDepthCompletionCombinedVertices.Add(center + right + up);
                _roomRawDepthCompletionCombinedVertices.Add(center - right + up);
                indices.Add(first);
                indices.Add(first + 2);
                indices.Add(first + 1);
                indices.Add(first);
                indices.Add(first + 3);
                indices.Add(first + 2);
            }
        }

        _roomRawDepthCompletionCombinedMesh.Clear(false);
        _roomRawDepthCompletionCombinedMesh.SetVertices(_roomRawDepthCompletionCombinedVertices);
        _roomRawDepthCompletionCombinedMesh.subMeshCount = _roomRawDepthCompletionCombinedIndices.Length;
        for (int i = 0; i < _roomRawDepthCompletionCombinedIndices.Length; i++)
            _roomRawDepthCompletionCombinedMesh.SetTriangles(_roomRawDepthCompletionCombinedIndices[i], i, false);
        _roomRawDepthCompletionCombinedMesh.RecalculateBounds();
        _roomRawDepthCompletionCombinedMeshRenderer.enabled = _roomRawDepthCompletionCombinedVertices.Count > 0;
    }

    private void AddRoomRawDepthCompletionMatrix(RoomRawDepthCompletionState state, Vector3 position)
    {
        int index = Mathf.Clamp((int)state, 0, _roomRawDepthCompletionMatrices.Length - 1);
        float size = Mathf.Max(0.002f, roomRawDepthCompletionPointSize);
        position = OffsetQuestCloneProgressPointTowardViewer(position);
        _roomRawDepthCompletionMatrices[index].Add(Matrix4x4.TRS(position, Quaternion.identity, Vector3.one * size));
    }

    private Vector3 OffsetQuestCloneProgressPointTowardViewer(Vector3 position)
    {
        if (!questCloneContinuousCaptureMode)
            return position;

        float offset = Mathf.Max(0f, questCloneProgressSurfaceOffsetMeters);
        Transform pose = ResolvePoseTransform();
        if (offset <= 0f || pose == null)
            return position;

        Vector3 towardViewer = pose.position - position;
        if (!IsFinite(towardViewer) || towardViewer.sqrMagnitude <= 1e-8f)
            return position;
        return position + towardViewer.normalized * offset;
    }

    private void DrawRoomRawDepthCompletionOverlay()
    {
        if (questCloneContinuousCaptureMode)
            return;

        if (!roomRawDepthCompletionUseGpuInstancing ||
            !ShouldTrackRoomRawDepthCompletion() ||
            (_roomRawDepthCompletionVoxels.Count <= 0 && _questCloneGuideVoxels.Count <= 0) ||
            ((questCloneContinuousCaptureMode || roomRawDepthCompletionOnlyWhileRecording) && !_isRecording))
            return;

        EnsureRoomRawDepthCompletionOverlay();
        EnsureRoomRawDepthCompletionInstancedMaterials();
        if (_roomRawDepthCompletionMesh == null)
            return;

        for (int stateIndex = 0; stateIndex < _roomRawDepthCompletionMatrices.Length; stateIndex++)
        {
            List<Matrix4x4> matrices = _roomRawDepthCompletionMatrices[stateIndex];
            if (matrices == null || matrices.Count <= 0)
                continue;

            Material material = stateIndex < _roomRawDepthCompletionInstancedMaterials.Length
                ? _roomRawDepthCompletionInstancedMaterials[stateIndex]
                : null;
            if (material == null)
                continue;

            for (int start = 0; start < matrices.Count; start += 1023)
            {
                int batchCount = Mathf.Min(1023, matrices.Count - start);
                matrices.CopyTo(start, _roomRawDepthCompletionDrawBatch, 0, batchCount);
                Graphics.DrawMeshInstanced(_roomRawDepthCompletionMesh, 0, material, _roomRawDepthCompletionDrawBatch, batchCount);
            }
        }
    }

    private void ClearRoomRawDepthSnapshotOverlay()
    {
        _roomRawDepthSnapshotMatrices.Clear();
        _roomRawDepthSnapshotVertices.Clear();
        _roomRawDepthSnapshotColors.Clear();
        _roomRawDepthSnapshotIndices.Clear();
        if (_roomRawDepthSnapshotMesh != null)
            _roomRawDepthSnapshotMesh.Clear();
        if (_roomRawDepthSnapshotMeshRoot != null)
            _roomRawDepthSnapshotMeshRoot.SetActive(false);
        _roomRawDepthSnapshotVisiblePoints = 0;
        _roomRawDepthSnapshotLastTotalPixels = 0;
        _roomRawDepthSnapshotLastValidPixels = 0;
        _roomRawDepthSnapshotLastFrame = "none";
    }

    public void SuppressRawSnapshotVisualOverlayAndManualInput()
    {
        showRoomRawDepthSnapshotOverlay = false;
        showRoomRawDepthCompletionOverlay = false;
        showRoomRawCoverageViewFrame = false;
        showRoomRawCoverageTileHints = false;
        showRoomRawCoverageReticle = false;
        showRoomRawCoveragePreviewPoints = false;
        enableOvrCaptureNowInput = false;
        ClearRoomRawDepthSnapshotOverlay();
        SetRoomRawDepthCompletionOverlayVisible(false);
    }

    private void DrawRoomRawDepthSnapshotOverlay()
    {
        if (_roomRawDepthSnapshotMeshRoot != null)
        {
            bool visible = showRoomRawDepthSnapshotOverlay && _roomRawDepthSnapshotVisiblePoints > 0;
            if (_roomRawDepthSnapshotMeshRoot.activeSelf != visible)
                _roomRawDepthSnapshotMeshRoot.SetActive(visible);
            return;
        }

        if (!showRoomRawDepthSnapshotOverlay || _roomRawDepthSnapshotMatrices.Count <= 0)
            return;

        EnsureRoomRawDepthCompletionMesh();
        EnsureRoomRawDepthSnapshotMaterial();
        if (_roomRawDepthCompletionMesh == null || _roomRawDepthSnapshotMaterial == null)
            return;

        for (int start = 0; start < _roomRawDepthSnapshotMatrices.Count; start += 1023)
        {
            int batchCount = Mathf.Min(1023, _roomRawDepthSnapshotMatrices.Count - start);
            _roomRawDepthSnapshotMatrices.CopyTo(start, _roomRawDepthCompletionDrawBatch, 0, batchCount);
            Graphics.DrawMeshInstanced(_roomRawDepthCompletionMesh, 0, _roomRawDepthSnapshotMaterial, _roomRawDepthCompletionDrawBatch, batchCount);
        }
    }

    private void EnsureRoomRawDepthSnapshotMaterial()
    {
        if (_roomRawDepthSnapshotMaterial != null)
            return;

        Shader shader = Shader.Find("Hidden/ScanCover/RawDepthProjectedPointCloud");
        if (shader == null)
            shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Standard");

        _roomRawDepthSnapshotMaterial = new Material(shader);
        _roomRawDepthSnapshotMaterial.hideFlags = HideFlags.DontSave;
        _roomRawDepthSnapshotMaterial.enableInstancing = true;
        _roomRawDepthSnapshotMaterial.SetColor("_BaseColor", roomRawDepthSnapshotColor);
        _roomRawDepthSnapshotMaterial.SetColor("_Color", roomRawDepthSnapshotColor);
        if (_roomRawDepthSnapshotMaterial.HasProperty("_Alpha"))
            _roomRawDepthSnapshotMaterial.SetFloat("_Alpha", roomRawDepthSnapshotColor.a);
        if (_roomRawDepthSnapshotMaterial.HasProperty("_Brightness"))
            _roomRawDepthSnapshotMaterial.SetFloat("_Brightness", 1.35f);
    }

    private void EnsureRoomRawDepthSnapshotMeshOverlay()
    {
        EnsureRoomRawDepthSnapshotMaterial();

        if (_roomRawDepthSnapshotMesh == null)
        {
            _roomRawDepthSnapshotMesh = new Mesh { name = "ScanCover_RoomRawDepthSnapshotOverlay" };
            _roomRawDepthSnapshotMesh.hideFlags = HideFlags.DontSave;
            _roomRawDepthSnapshotMesh.MarkDynamic();
            _roomRawDepthSnapshotMesh.indexFormat = IndexFormat.UInt32;
        }

        if (_roomRawDepthSnapshotMeshRoot != null)
            return;

        _roomRawDepthSnapshotMeshRoot = new GameObject("[ScanCover] Room Raw Depth Snapshot Overlay");
        _roomRawDepthSnapshotMeshRoot.hideFlags = HideFlags.DontSave;
        _roomRawDepthSnapshotMeshRoot.transform.position = Vector3.zero;
        _roomRawDepthSnapshotMeshRoot.transform.rotation = Quaternion.identity;
        _roomRawDepthSnapshotMeshRoot.transform.localScale = Vector3.one;

        _roomRawDepthSnapshotMeshFilter = _roomRawDepthSnapshotMeshRoot.AddComponent<MeshFilter>();
        _roomRawDepthSnapshotMeshRenderer = _roomRawDepthSnapshotMeshRoot.AddComponent<MeshRenderer>();
        _roomRawDepthSnapshotMeshFilter.sharedMesh = _roomRawDepthSnapshotMesh;
        _roomRawDepthSnapshotMeshRenderer.sharedMaterial = _roomRawDepthSnapshotMaterial;
        _roomRawDepthSnapshotMeshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _roomRawDepthSnapshotMeshRenderer.receiveShadows = false;
        _roomRawDepthSnapshotMeshRoot.SetActive(false);
    }

    private void AddRoomRawDepthSnapshotQuad(Vector3 point, Vector3 right, Vector3 up, float size)
    {
        int baseIndex = _roomRawDepthSnapshotVertices.Count;
        float half = Mathf.Max(0.001f, size) * 0.5f;
        Vector3 x = right * half;
        Vector3 y = up * half;
        Color color = roomRawDepthSnapshotColor;

        _roomRawDepthSnapshotVertices.Add(point - x - y);
        _roomRawDepthSnapshotVertices.Add(point - x + y);
        _roomRawDepthSnapshotVertices.Add(point + x + y);
        _roomRawDepthSnapshotVertices.Add(point + x - y);

        _roomRawDepthSnapshotColors.Add(color);
        _roomRawDepthSnapshotColors.Add(color);
        _roomRawDepthSnapshotColors.Add(color);
        _roomRawDepthSnapshotColors.Add(color);

        _roomRawDepthSnapshotIndices.Add(baseIndex);
        _roomRawDepthSnapshotIndices.Add(baseIndex + 1);
        _roomRawDepthSnapshotIndices.Add(baseIndex + 2);
        _roomRawDepthSnapshotIndices.Add(baseIndex);
        _roomRawDepthSnapshotIndices.Add(baseIndex + 2);
        _roomRawDepthSnapshotIndices.Add(baseIndex + 3);
    }

    private void UploadRoomRawDepthSnapshotOverlayMesh()
    {
        EnsureRoomRawDepthSnapshotMeshOverlay();
        if (_roomRawDepthSnapshotMesh == null)
            return;

        _roomRawDepthSnapshotMesh.Clear();
        if (_roomRawDepthSnapshotVertices.Count > 0)
        {
            _roomRawDepthSnapshotMesh.indexFormat = _roomRawDepthSnapshotVertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            _roomRawDepthSnapshotMesh.SetVertices(_roomRawDepthSnapshotVertices);
            _roomRawDepthSnapshotMesh.SetColors(_roomRawDepthSnapshotColors);
            _roomRawDepthSnapshotMesh.SetIndices(_roomRawDepthSnapshotIndices, MeshTopology.Triangles, 0, true);
            _roomRawDepthSnapshotMesh.RecalculateBounds();
        }

        if (_roomRawDepthSnapshotMeshFilter != null)
            _roomRawDepthSnapshotMeshFilter.sharedMesh = _roomRawDepthSnapshotMesh;
        if (_roomRawDepthSnapshotMeshRenderer != null)
            _roomRawDepthSnapshotMeshRenderer.sharedMaterial = _roomRawDepthSnapshotMaterial;
        if (_roomRawDepthSnapshotMeshRoot != null)
            _roomRawDepthSnapshotMeshRoot.SetActive(showRoomRawDepthSnapshotOverlay && _roomRawDepthSnapshotVertices.Count > 0);
    }

    private void DestroyRoomRawDepthSnapshotOverlay()
    {
        ClearRoomRawDepthSnapshotOverlay();
        if (_roomRawDepthSnapshotMeshRoot != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawDepthSnapshotMeshRoot);
            else
                DestroyImmediate(_roomRawDepthSnapshotMeshRoot);
            _roomRawDepthSnapshotMeshRoot = null;
            _roomRawDepthSnapshotMeshFilter = null;
            _roomRawDepthSnapshotMeshRenderer = null;
        }
        if (_roomRawDepthSnapshotMesh != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawDepthSnapshotMesh);
            else
                DestroyImmediate(_roomRawDepthSnapshotMesh);
            _roomRawDepthSnapshotMesh = null;
        }
        if (_roomRawDepthSnapshotMaterial == null)
            return;

        if (Application.isPlaying)
            Destroy(_roomRawDepthSnapshotMaterial);
        else
            DestroyImmediate(_roomRawDepthSnapshotMaterial);
        _roomRawDepthSnapshotMaterial = null;
    }

    private void CountRoomRawDepthCompletionState(RoomRawDepthCompletionState state)
    {
        switch (state)
        {
            case RoomRawDepthCompletionState.Stable:
                _roomRawDepthCompletionGreenCount++;
                break;
            case RoomRawDepthCompletionState.Supported:
                _roomRawDepthCompletionBlueCount++;
                break;
            case RoomRawDepthCompletionState.Risk:
                _roomRawDepthCompletionYellowCount++;
                break;
            case RoomRawDepthCompletionState.Conflict:
                _roomRawDepthCompletionRedCount++;
                break;
            default:
                _roomRawDepthCompletionGrayCount++;
                break;
        }
    }

    private void EnsureRoomRawDepthCompletionOverlay()
    {
        if (_roomRawDepthCompletionRoot != null)
            return;

        _roomRawDepthCompletionRoot = new GameObject("[ScanCover] Raw Depth Completion Overlay");
        _roomRawDepthCompletionRoot.hideFlags = HideFlags.DontSave;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Standard");

        _roomRawDepthCompletionMaterial = new Material(shader);
        _roomRawDepthCompletionMaterial.hideFlags = HideFlags.DontSave;
        _roomRawDepthCompletionMaterial.enableInstancing = true;
        _roomRawDepthCompletionPropertyBlock = new MaterialPropertyBlock();
        EnsureRoomRawDepthCompletionMesh();
        EnsureRoomRawDepthCompletionInstancedMaterials();
        EnsureRoomRawDepthCompletionCombinedRenderer();
    }

    private void EnsureRoomRawDepthCompletionCombinedRenderer()
    {
        if (!questCloneContinuousCaptureMode || _roomRawDepthCompletionRoot == null)
            return;

        if (_roomRawDepthCompletionCombinedMesh == null)
        {
            _roomRawDepthCompletionCombinedMesh = new Mesh
            {
                name = "ScanCover Quest Capture Progress Quads",
                hideFlags = HideFlags.DontSave,
                indexFormat = IndexFormat.UInt32
            };
            _roomRawDepthCompletionCombinedMesh.MarkDynamic();
        }

        if (_roomRawDepthCompletionCombinedMeshFilter == null)
            _roomRawDepthCompletionCombinedMeshFilter = _roomRawDepthCompletionRoot.AddComponent<MeshFilter>();
        _roomRawDepthCompletionCombinedMeshFilter.sharedMesh = _roomRawDepthCompletionCombinedMesh;

        if (_roomRawDepthCompletionCombinedMeshRenderer == null)
        {
            _roomRawDepthCompletionCombinedMeshRenderer = _roomRawDepthCompletionRoot.AddComponent<MeshRenderer>();
            _roomRawDepthCompletionCombinedMeshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _roomRawDepthCompletionCombinedMeshRenderer.receiveShadows = false;
            _roomRawDepthCompletionCombinedMeshRenderer.lightProbeUsage = LightProbeUsage.Off;
            _roomRawDepthCompletionCombinedMeshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }
        _roomRawDepthCompletionCombinedMeshRenderer.sharedMaterials = _roomRawDepthCompletionInstancedMaterials;
    }

    private void EnsureRoomRawDepthCompletionMesh()
    {
        if (_roomRawDepthCompletionMesh != null)
            return;

        _roomRawDepthCompletionMesh = new Mesh { name = "ScanCover Raw Depth Completion Cube" };
        _roomRawDepthCompletionMesh.hideFlags = HideFlags.DontSave;
        Vector3[] vertices =
        {
            new Vector3(-0.5f, -0.5f, -0.5f), new Vector3(0.5f, -0.5f, -0.5f), new Vector3(0.5f, 0.5f, -0.5f), new Vector3(-0.5f, 0.5f, -0.5f),
            new Vector3(-0.5f, -0.5f, 0.5f), new Vector3(0.5f, -0.5f, 0.5f), new Vector3(0.5f, 0.5f, 0.5f), new Vector3(-0.5f, 0.5f, 0.5f)
        };
        int[] triangles =
        {
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            1, 2, 6, 1, 6, 5,
            3, 0, 4, 3, 4, 7
        };
        _roomRawDepthCompletionMesh.vertices = vertices;
        _roomRawDepthCompletionMesh.triangles = triangles;
        _roomRawDepthCompletionMesh.RecalculateNormals();
        _roomRawDepthCompletionMesh.RecalculateBounds();
    }

    private void EnsureRoomRawDepthCompletionInstancedMaterials()
    {
        if (_roomRawDepthCompletionMaterial == null)
            return;

        for (int i = 0; i < _roomRawDepthCompletionInstancedMaterials.Length; i++)
        {
            if (_roomRawDepthCompletionInstancedMaterials[i] != null)
                continue;

            Material material = new Material(_roomRawDepthCompletionMaterial);
            material.hideFlags = HideFlags.DontSave;
            material.enableInstancing = true;
            Color color = GetRoomRawDepthCompletionColor((RoomRawDepthCompletionState)i);
            material.SetColor("_BaseColor", color);
            material.SetColor("_Color", color);
            if (material.HasProperty("_Cull"))
                material.SetFloat("_Cull", (float)CullMode.Off);
            material.renderQueue = 3000;
            _roomRawDepthCompletionInstancedMaterials[i] = material;
        }
    }

    private GameObject GetOrCreateRoomRawDepthCompletionPoint(int index)
    {
        while (_roomRawDepthCompletionPointObjects.Count <= index)
        {
            GameObject pointObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pointObject.name = $"Raw Depth Completion Point {_roomRawDepthCompletionPointObjects.Count:0000}";
            pointObject.hideFlags = HideFlags.DontSave;
            pointObject.transform.SetParent(_roomRawDepthCompletionRoot.transform, false);

            Collider pointCollider = pointObject.GetComponent<Collider>();
            if (pointCollider != null)
            {
                if (Application.isPlaying)
                    Destroy(pointCollider);
                else
                    DestroyImmediate(pointCollider);
            }

            Renderer renderer = pointObject.GetComponent<Renderer>();
            if (renderer != null)
                renderer.sharedMaterial = _roomRawDepthCompletionMaterial;

            _roomRawDepthCompletionPointObjects.Add(pointObject);
        }

        return _roomRawDepthCompletionPointObjects[index];
    }

    private void ApplyRoomRawDepthCompletionPoint(GameObject pointObject, Vector3 position, Color color)
    {
        if (pointObject == null)
            return;

        pointObject.SetActive(true);
        pointObject.transform.position = OffsetQuestCloneProgressPointTowardViewer(position);
        pointObject.transform.localScale = Vector3.one * Mathf.Max(0.002f, roomRawDepthCompletionPointSize);

        Renderer renderer = pointObject.GetComponent<Renderer>();
        if (renderer == null)
            return;

        if (_roomRawDepthCompletionPropertyBlock == null)
            _roomRawDepthCompletionPropertyBlock = new MaterialPropertyBlock();

        renderer.GetPropertyBlock(_roomRawDepthCompletionPropertyBlock);
        _roomRawDepthCompletionPropertyBlock.SetColor("_BaseColor", color);
        _roomRawDepthCompletionPropertyBlock.SetColor("_Color", color);
        renderer.SetPropertyBlock(_roomRawDepthCompletionPropertyBlock);
    }

    private void SetRoomRawDepthCompletionOverlayVisible(bool visible)
    {
        if (_roomRawDepthCompletionRoot != null && _roomRawDepthCompletionRoot.activeSelf != visible)
            _roomRawDepthCompletionRoot.SetActive(visible);
    }

    private void DestroyRoomRawDepthCompletionOverlay()
    {
        if (_roomRawDepthCompletionRoot != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawDepthCompletionRoot);
            else
                DestroyImmediate(_roomRawDepthCompletionRoot);
            _roomRawDepthCompletionRoot = null;
        }
        _roomRawDepthCompletionCombinedMeshFilter = null;
        _roomRawDepthCompletionCombinedMeshRenderer = null;

        if (_roomRawDepthCompletionCombinedMesh != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawDepthCompletionCombinedMesh);
            else
                DestroyImmediate(_roomRawDepthCompletionCombinedMesh);
            _roomRawDepthCompletionCombinedMesh = null;
        }
        _roomRawDepthCompletionCombinedVertices.Clear();
        for (int i = 0; i < _roomRawDepthCompletionCombinedIndices.Length; i++)
            _roomRawDepthCompletionCombinedIndices[i].Clear();

        if (_roomRawDepthCompletionMaterial != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawDepthCompletionMaterial);
            else
                DestroyImmediate(_roomRawDepthCompletionMaterial);
            _roomRawDepthCompletionMaterial = null;
        }

        for (int i = 0; i < _roomRawDepthCompletionInstancedMaterials.Length; i++)
        {
            if (_roomRawDepthCompletionInstancedMaterials[i] == null)
                continue;
            if (Application.isPlaying)
                Destroy(_roomRawDepthCompletionInstancedMaterials[i]);
            else
                DestroyImmediate(_roomRawDepthCompletionInstancedMaterials[i]);
            _roomRawDepthCompletionInstancedMaterials[i] = null;
        }

        if (_roomRawDepthCompletionMesh != null)
        {
            if (Application.isPlaying)
                Destroy(_roomRawDepthCompletionMesh);
            else
                DestroyImmediate(_roomRawDepthCompletionMesh);
            _roomRawDepthCompletionMesh = null;
        }

        _roomRawDepthCompletionPointObjects.Clear();
        ClearRoomRawDepthCompletionMatrices();
    }

    private RoomRawDepthCompletionState ClassifyRoomRawDepthCompletionVoxel(Vector3Int key, RoomRawDepthCompletionVoxel voxel)
    {
        if (voxel == null || voxel.hits <= 0)
            return RoomRawDepthCompletionState.Empty;

        if (voxel.hits < roomRawDepthCompletionMinHits)
            return RoomRawDepthCompletionState.Scanning;

        float consistencyStd = questCloneContinuousCaptureMode ? voxel.PositionStd : voxel.DepthStd;
        float riskRatio = questCloneContinuousCaptureMode ? voxel.RecentRiskRatio : voxel.RiskRatio;
        float hardStd = questCloneContinuousCaptureMode
            ? GetQuestCloneHardPositionStdMeters(voxel.MeanDepth)
            : roomRawDepthCompletionHardDepthStdMeters;
        float badRisk = questCloneContinuousCaptureMode
            ? questCloneProgressBadRecentRiskRatio
            : roomRawDepthCompletionBadRiskRatio;
        float warnRisk = questCloneContinuousCaptureMode
            ? questCloneProgressWarnRecentRiskRatio
            : roomRawDepthCompletionWarnRiskRatio;
        if (consistencyStd >= hardStd || riskRatio >= badRisk)
            return RoomRawDepthCompletionState.Conflict;

        if (IsRoomRawDepthCompletionStableCandidate(key, voxel, consistencyStd, riskRatio))
            return RoomRawDepthCompletionState.Stable;

        float stableStd = questCloneContinuousCaptureMode
            ? GetQuestCloneStablePositionStdMeters(voxel.MeanDepth)
            : roomRawDepthCompletionStableDepthStdMeters;
        if (consistencyStd >= stableStd || riskRatio >= warnRisk)
            return RoomRawDepthCompletionState.Risk;

        return RoomRawDepthCompletionState.Supported;
    }

    private float GetQuestClonePositionToleranceScale(float depthMeters)
    {
        if (!float.IsFinite(depthMeters) || depthMeters <= 0f)
            return 1f;

        if (depthMeters < 0.75f)
        {
            float nearT = Mathf.InverseLerp(0.35f, 0.75f, depthMeters);
            return Mathf.Lerp(1.20f, 1f, nearT);
        }
        if (depthMeters > 3f)
        {
            float farT = Mathf.InverseLerp(3f, 5f, depthMeters);
            return Mathf.Lerp(1f, 1.25f, farT);
        }
        return 1f;
    }

    private float GetQuestCloneStablePositionStdMeters(float depthMeters)
        => Mathf.Max(0.001f, questCloneProgressStablePositionStdMeters) * GetQuestClonePositionToleranceScale(depthMeters);

    private float GetQuestCloneHardPositionStdMeters(float depthMeters)
        => Mathf.Max(0.001f, questCloneProgressHardPositionStdMeters) * GetQuestClonePositionToleranceScale(depthMeters);

    private bool IsRoomRawDepthCompletionStableCandidate(Vector3Int key, RoomRawDepthCompletionVoxel voxel, float consistencyStd, float riskRatio)
    {
        int stableHits = Mathf.Max(1, roomRawDepthCompletionStableHits);
        int candidateHits = Mathf.Max(stableHits, roomRawDepthCompletionCandidateStableHits);
        int sameViewHits = Mathf.Max(candidateHits, roomRawDepthCompletionSameViewStableHits);
        float minAngle = Mathf.Clamp(roomRawDepthCompletionMinAngleSpanDegrees, 0f, 90f);
        bool hasAngleSupport = minAngle <= 0f || voxel.AngleSpanDegrees >= minAngle;
        bool hasLongSameViewSupport = voxel.hits >= sameViewHits;
        bool hasEnoughHits = voxel.hits >= stableHits && (hasAngleSupport || hasLongSameViewSupport);
        if (!hasEnoughHits)
            return false;

        float stableDepthStd = Mathf.Max(0.001f, questCloneContinuousCaptureMode
            ? GetQuestCloneStablePositionStdMeters(voxel.MeanDepth)
            : roomRawDepthCompletionStableDepthStdMeters);
        float warnRisk = questCloneContinuousCaptureMode
            ? questCloneProgressWarnRecentRiskRatio
            : roomRawDepthCompletionWarnRiskRatio;
        bool cleanStable = consistencyStd <= stableDepthStd && riskRatio <= warnRisk;
        if (cleanStable)
            return true;

        bool recoverableRisk = voxel.hits >= candidateHits &&
            consistencyStd <= Mathf.Max(stableDepthStd, questCloneContinuousCaptureMode
                ? GetQuestCloneHardPositionStdMeters(voxel.MeanDepth)
                : roomRawDepthCompletionNeighborStableDepthStdMeters) &&
            riskRatio <= Mathf.Max(roomRawDepthCompletionWarnRiskRatio, roomRawDepthCompletionRecoverableRiskRatio) &&
            CountRoomRawDepthCompletionNeighborSupport(key, voxel) >= Mathf.Max(0, roomRawDepthCompletionNeighborStableSupport);
        return recoverableRisk;
    }

    private int CountRoomRawDepthCompletionNeighborSupport(Vector3Int key, RoomRawDepthCompletionVoxel center)
    {
        int required = Mathf.Max(0, roomRawDepthCompletionNeighborStableSupport);
        if (required <= 0 || center == null)
            return required;

        int support = 0;
        float maxDepthDelta = Mathf.Max(0.001f, roomRawDepthCompletionNeighborDepthDeltaMeters);
        float maxDepthStd = Mathf.Max(0.001f, questCloneContinuousCaptureMode
            ? GetQuestCloneHardPositionStdMeters(center.MeanDepth)
            : roomRawDepthCompletionNeighborStableDepthStdMeters);
        float maxRisk = Mathf.Clamp01(questCloneContinuousCaptureMode
            ? questCloneProgressBadRecentRiskRatio
            : roomRawDepthCompletionNeighborRiskRatio);
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0 && dz == 0)
                continue;

            Vector3Int neighborKey = new Vector3Int(key.x + dx, key.y + dy, key.z + dz);
            if (!_roomRawDepthCompletionVoxels.TryGetValue(neighborKey, out RoomRawDepthCompletionVoxel neighbor) || neighbor == null)
                continue;
            if (neighbor.hits < roomRawDepthCompletionMinHits)
                continue;
            if (!questCloneContinuousCaptureMode && Mathf.Abs(neighbor.MeanDepth - center.MeanDepth) > maxDepthDelta)
                continue;
            float neighborStd = questCloneContinuousCaptureMode ? neighbor.PositionStd : neighbor.DepthStd;
            float neighborRisk = questCloneContinuousCaptureMode ? neighbor.RecentRiskRatio : neighbor.RiskRatio;
            if (neighborStd > maxDepthStd || neighborRisk > maxRisk)
                continue;

            support++;
            if (support >= required)
                return support;
        }

        return support;
    }

    private Color GetRoomRawDepthCompletionColor(RoomRawDepthCompletionState state)
    {
        if (questCloneContinuousCaptureMode)
        {
            switch (state)
            {
                case RoomRawDepthCompletionState.Stable:
                    return questCloneProgressStableColor;
                case RoomRawDepthCompletionState.Supported:
                    return questCloneProgressSupportedColor;
                case RoomRawDepthCompletionState.Risk:
                case RoomRawDepthCompletionState.Conflict:
                    return questCloneProgressRiskColor;
                default:
                    return questCloneProgressPendingColor;
            }
        }

        switch (state)
        {
            case RoomRawDepthCompletionState.Stable:
                return new Color(0f, 1f, 0.25f, 1f);
            case RoomRawDepthCompletionState.Supported:
                return new Color(0.05f, 0.35f, 1f, 1f);
            case RoomRawDepthCompletionState.Risk:
                return new Color(1f, 0.82f, 0f, 1f);
            case RoomRawDepthCompletionState.Conflict:
                return new Color(1f, 0.04f, 0.02f, 1f);
            default:
                return new Color(0.45f, 0.45f, 0.45f, 1f);
        }
    }

    private void UpdateRoomRawDepthCompletionHint(int total)
    {
        if (total <= 0)
        {
            _roomRawDepthCompletionHint = "Waiting";
            return;
        }

        float greenRatio = _roomRawDepthCompletionGreenCount / (float)total;
        float yellowRedRatio = (_roomRawDepthCompletionYellowCount + _roomRawDepthCompletionRedCount) / (float)total;

        if (_roomRawDepthCompletionRedCount > total * 0.2f)
            _roomRawDepthCompletionHint = "Too much red: depth conflict or floating points; rescan from another distance/angle";
        else if (yellowRedRatio > 0.35f)
            _roomRawDepthCompletionHint = "Too much yellow: edge/oblique risk; side-step or move closer";
        else if (greenRatio < 0.35f)
            _roomRawDepthCompletionHint = "Blue/gray dominant: supported but not stable; scan slowly or change angle";
        else
            _roomRawDepthCompletionHint = "Green dominant: this area is mostly complete";
    }

    private void AppendRoomRawDepthCompletionHudLine(StringBuilder builder)
    {
        if (!ShouldTrackRoomRawDepthCompletion())
            return;

        int total = _roomRawDepthCompletionVoxels.Count;
        if (total <= 0)
        {
            builder.AppendLine("Raw completion: waiting");
            return;
        }

        float stableRatio = _roomRawDepthCompletionGreenCount / (float)Mathf.Max(1, total);
        if (questCloneContinuousCaptureMode)
        {
            builder.Append("Capture grid cyan/purple/risk/pending: ")
                .Append(_roomRawDepthCompletionGreenCount.ToString(CultureInfo.InvariantCulture)).Append('/')
                .Append(_roomRawDepthCompletionBlueCount.ToString(CultureInfo.InvariantCulture)).Append('/')
                .Append((_roomRawDepthCompletionYellowCount + _roomRawDepthCompletionRedCount).ToString(CultureInfo.InvariantCulture)).Append('/')
                .Append(_roomRawDepthCompletionGrayCount.ToString(CultureInfo.InvariantCulture))
                .Append(" complete ")
                .Append((stableRatio * 100f).ToString("0", CultureInfo.InvariantCulture))
                .AppendLine("%");
            return;
        }

        builder.Append("Raw completion G/B/Y/R: ")
            .Append(_roomRawDepthCompletionGreenCount.ToString(CultureInfo.InvariantCulture)).Append('/')
            .Append(_roomRawDepthCompletionBlueCount.ToString(CultureInfo.InvariantCulture)).Append('/')
            .Append(_roomRawDepthCompletionYellowCount.ToString(CultureInfo.InvariantCulture)).Append('/')
            .Append(_roomRawDepthCompletionRedCount.ToString(CultureInfo.InvariantCulture))
            .Append(" stable ")
            .Append((stableRatio * 100f).ToString("0", CultureInfo.InvariantCulture))
            .AppendLine("%");
        builder.Append("Scan hint: ").AppendLine(_roomRawDepthCompletionHint);
    }

    private bool ShouldTrackRoomRawDepthCompletion()
        => showRoomRawDepthCompletionOverlay || questCloneContinuousCaptureMode;

    private void AppendRoomRawDepthSnapshotHudLine(StringBuilder builder)
    {
        if (!exportRoomRawDepthSnapshots && !showRoomRawDepthSnapshotOverlay)
            return;

        builder.Append("Raw快照: ")
            .Append(_roomRawDepthSnapshotLastValidPixels).Append('/')
            .Append(_roomRawDepthSnapshotLastTotalPixels)
            .Append(" 显示")
            .Append(_roomRawDepthSnapshotVisiblePoints)
            .Append(" 帧 ")
            .Append(_roomRawDepthSnapshotLastFrame ?? "none")
            .AppendLine();
    }

    private void AppendBinocularRoomRawDepthSnapshotHudLine(StringBuilder builder)
    {
        if (!useBinocularRoomRawDepthSnapshots)
            return;

        builder.Append("双眼快照: ").AppendLine(_binocularRoomRawDepthStatus ?? "idle");
    }

    private bool ShouldUseBinocularRoomRawDepthSnapshotCapture()
    {
        if (!useBinocularRoomRawDepthSnapshots || depthGridPointCloud == null)
            return false;
        if (!(questCloneContinuousCaptureMode && questCloneUseCompactBinary) && !ShouldExportRoomRawDepthSnapshots())
            return false;

        if (!questCloneContinuousCaptureMode && binocularRoomRawDepthSnapshotsManualOnly &&
            !string.Equals(_pendingReason, "manual", StringComparison.OrdinalIgnoreCase))
            return false;

        return depthGridPointCloud.Preprocessor != null;
    }

    private bool BeginBinocularRoomRawDepthSnapshotCapture()
    {
        ResetBinocularRoomRawDepthSnapshotCapture(restoreOriginalEye: false);
        ScanCoverDepthPreprocessor preprocessor = depthGridPointCloud != null ? depthGridPointCloud.Preprocessor : null;
        if (preprocessor == null)
        {
            _binocularRoomRawDepthStatus = "missing preprocessor";
            return false;
        }

        _binocularRoomRawDepthOriginalEye = preprocessor.CurrentSourceEye;
        _binocularRoomRawDepthStatus = "右眼采集中";
        return RequestBinocularRoomRawDepthEye(ScanCoverDepthPreprocessor.SourceEye.Right, BinocularRoomRawDepthSnapshotStage.RightRequested);
    }

    private bool RequestBinocularRoomRawDepthEye(ScanCoverDepthPreprocessor.SourceEye eye, BinocularRoomRawDepthSnapshotStage stage)
    {
        ScanCoverDepthPreprocessor preprocessor = depthGridPointCloud != null ? depthGridPointCloud.Preprocessor : null;
        if (preprocessor == null || depthGridPointCloud == null)
            return FailBinocularRoomRawDepthSnapshotCapture("missing depth source");

        _binocularRoomRawDepthStartSnapshotIndex = GetLatestRawDepthSnapshotFrameIndex();
        _binocularRoomRawDepthStartTime = Time.unscaledTime;
        _binocularRoomRawDepthSnapshotStage = stage;
        preprocessor.SetSourceEye(eye);

        if (depthGridPointCloud.HasPendingReadback)
        {
            _binocularRoomRawDepthStatus = eye == ScanCoverDepthPreprocessor.SourceEye.Right
                ? "waiting for right-eye source"
                : "waiting for left-eye source";
            _pendingRefreshRequested = false;
            return true;
        }

        if (!depthGridPointCloud.RefreshNow(forcePreprocessorRefresh: true))
            return FailBinocularRoomRawDepthSnapshotCapture(depthGridPointCloud.LastIssue ?? "depth refresh failed");

        _pendingRefreshRequested = true;
        return true;
    }

    private void UpdateBinocularRoomRawDepthSnapshotCapture()
    {
        if (_binocularRoomRawDepthSnapshotStage == BinocularRoomRawDepthSnapshotStage.None ||
            _binocularRoomRawDepthSnapshotStage == BinocularRoomRawDepthSnapshotStage.Ready)
            return;

        bool timedOut = Time.unscaledTime - _binocularRoomRawDepthStartTime >= Mathf.Max(0.05f, binocularRoomRawDepthSnapshotTimeoutSeconds);
        int latestSnapshotFrame = GetLatestRawDepthSnapshotFrameIndex();
        bool hasFreshSnapshot = latestSnapshotFrame > _binocularRoomRawDepthStartSnapshotIndex;

        if (!hasFreshSnapshot)
        {
            if (!depthGridPointCloud.HasPendingReadback)
            {
                if (depthGridPointCloud.RefreshNow(forcePreprocessorRefresh: true))
                    _pendingRefreshRequested = true;
            }
            if (timedOut)
                FailBinocularRoomRawDepthSnapshotCapture("timeout");
            return;
        }

        if (!depthGridPointCloud.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot))
        {
            if (timedOut)
                FailBinocularRoomRawDepthSnapshotCapture("snapshot unavailable");
            return;
        }

        int expectedEye = _binocularRoomRawDepthSnapshotStage == BinocularRoomRawDepthSnapshotStage.RightRequested
            ? (int)ScanCoverDepthPreprocessor.SourceEye.Right
            : (int)ScanCoverDepthPreprocessor.SourceEye.Left;
        if (snapshot.sourceEyeIndex != expectedEye)
        {
            _binocularRoomRawDepthStartSnapshotIndex = snapshot.frameIndex;
            if (!depthGridPointCloud.HasPendingReadback && depthGridPointCloud.RefreshNow(forcePreprocessorRefresh: true))
                _pendingRefreshRequested = true;
            if (timedOut)
                FailBinocularRoomRawDepthSnapshotCapture("eye provenance timeout");
            return;
        }

        if (_binocularRoomRawDepthSnapshotStage == BinocularRoomRawDepthSnapshotStage.RightRequested)
        {
            _binocularRoomRawDepthRightSnapshot = snapshot;
            _binocularRoomRawDepthStatus = "左眼采集中";
            RequestBinocularRoomRawDepthEye(ScanCoverDepthPreprocessor.SourceEye.Left, BinocularRoomRawDepthSnapshotStage.LeftRequested);
            return;
        }

        if (_binocularRoomRawDepthSnapshotStage == BinocularRoomRawDepthSnapshotStage.LeftRequested)
        {
            _binocularRoomRawDepthLeftSnapshot = snapshot;
            // Compact collection writes the two eye buffers independently. Avoid
            // allocating and copying a third, fused 2-eye buffer on the main thread.
            _binocularRoomRawDepthFusedSnapshot = questCloneContinuousCaptureMode && questCloneUseCompactBinary
                ? _binocularRoomRawDepthLeftSnapshot
                : BuildBinocularRoomRawDepthSnapshot(
                    _binocularRoomRawDepthRightSnapshot,
                    _binocularRoomRawDepthLeftSnapshot);

            if (_binocularRoomRawDepthFusedSnapshot == null)
            {
                FailBinocularRoomRawDepthSnapshotCapture("merge failed");
                return;
            }

            RestoreBinocularRoomRawDepthOriginalEye();
            _binocularRoomRawDepthSnapshotStage = BinocularRoomRawDepthSnapshotStage.Ready;
            _binocularRoomRawDepthStatus = "双眼完成";
            TryExportPendingCapture(force: true);
        }
    }

    private ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot BuildBinocularRoomRawDepthSnapshot(
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot right,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot left)
    {
        if (right == null || left == null ||
            right.worldPositions == null || right.worldNormals == null || right.observationMeta == null ||
            left.worldPositions == null || left.worldNormals == null || left.observationMeta == null)
            return null;

        int rightCount = Mathf.Min(right.worldPositions.Length, Mathf.Min(right.worldNormals.Length, right.observationMeta.Length));
        int leftCount = Mathf.Min(left.worldPositions.Length, Mathf.Min(left.worldNormals.Length, left.observationMeta.Length));
        if (rightCount <= 0 || leftCount <= 0)
            return null;

        int total = rightCount + leftCount;
        Vector3[] positions = new Vector3[total];
        Vector3[] normals = new Vector3[total];
        Color[] meta = new Color[total];
        Array.Copy(right.worldPositions, 0, positions, 0, rightCount);
        Array.Copy(right.worldNormals, 0, normals, 0, rightCount);
        Array.Copy(right.observationMeta, 0, meta, 0, rightCount);
        Array.Copy(left.worldPositions, 0, positions, rightCount, leftCount);
        Array.Copy(left.worldNormals, 0, normals, rightCount, leftCount);
        Array.Copy(left.observationMeta, 0, meta, rightCount, leftCount);

        int width = right.resolutionWidth == left.resolutionWidth
            ? Mathf.Max(1, right.resolutionWidth)
            : Mathf.Max(1, Mathf.Max(right.resolutionWidth, left.resolutionWidth));
        int height = right.resolutionWidth == left.resolutionWidth
            ? Mathf.Max(1, right.resolutionHeight) + Mathf.Max(1, left.resolutionHeight)
            : Mathf.Max(1, Mathf.CeilToInt(total / (float)width));

        return new ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot
        {
            componentName = (right.componentName ?? "ScanCoverDepthGridPointCloud") + "+binocular",
            frameIndex = Mathf.Max(right.frameIndex, left.frameIndex),
            resolutionWidth = width,
            resolutionHeight = height,
            sourceEyeIndex = -1,
            dispatchRealtimeSeconds = Math.Min(right.dispatchRealtimeSeconds, left.dispatchRealtimeSeconds),
            completionRealtimeSeconds = Math.Max(right.completionRealtimeSeconds, left.completionRealtimeSeconds),
            snapshotRealtimeSeconds = Math.Max(right.snapshotRealtimeSeconds, left.snapshotRealtimeSeconds),
            hasSnapshotCameraPose = left.hasSnapshotCameraPose || right.hasSnapshotCameraPose,
            snapshotCameraPosition = left.hasSnapshotCameraPose ? left.snapshotCameraPosition : right.snapshotCameraPosition,
            snapshotCameraRotation = left.hasSnapshotCameraPose ? left.snapshotCameraRotation : right.snapshotCameraRotation,
            worldPositions = positions,
            worldNormals = normals,
            observationMeta = meta
        };
    }

    private void ExportQuestCloneCapture(string frameName, Transform pose)
    {
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot right = _binocularRoomRawDepthRightSnapshot;
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot left = _binocularRoomRawDepthLeftSnapshot;
        if (!_sessionFileExportEnabled || right == null || left == null || string.IsNullOrEmpty(_questCloneBinaryDirectory))
            return;

        UpdateQuestCloneProgress(right);
        UpdateQuestCloneProgress(left);

        double rightTime = right.dispatchRealtimeSeconds > 0d ? right.dispatchRealtimeSeconds : right.snapshotRealtimeSeconds;
        double leftTime = left.dispatchRealtimeSeconds > 0d ? left.dispatchRealtimeSeconds : left.snapshotRealtimeSeconds;
        _questCloneLastInterEyeDeltaMs = Math.Abs(leftTime - rightTime) * 1000d;

        int maxPending = Mathf.Max(1, questCloneMaxPendingWrites);
        if (Volatile.Read(ref _questCloneWritesInFlight) >= maxPending)
        {
            Interlocked.Increment(ref _questCloneDroppedWrites);
            AppendQuestCloneDroppedRow(frameName, right, left, "dropped-writer-backpressure");
            return;
        }

        Directory.CreateDirectory(_questCloneBinaryDirectory);
        string safeFrame = SafeDirectoryName(frameName, $"frame_{_capturedFrameCount:D4}");
        string rightPath = Path.Combine(_questCloneBinaryDirectory, safeFrame + "_right.scq3bin");
        string leftPath = Path.Combine(_questCloneBinaryDirectory, safeFrame + "_left.scq3bin");
        string metadataPath = Path.Combine(_questCloneBinaryDirectory, safeFrame + "_capture.json");
        string metadataJson = BuildQuestCloneCaptureMetadata(frameName, right, left, pose, rightPath, leftPath);
        string row = BuildQuestCloneManifestRow(frameName, right, left, rightPath, leftPath, metadataPath, "exported");
        QuestCloneWriteJob job = new QuestCloneWriteJob
        {
            rightPath = rightPath,
            leftPath = leftPath,
            metadataPath = metadataPath,
            manifestPath = _questCloneBinaryManifestPath,
            metadataJson = metadataJson,
            manifestRow = row,
            right = right,
            left = left
        };

        Interlocked.Increment(ref _questCloneWritesInFlight);
        ThreadPool.QueueUserWorkItem(_ => WriteQuestCloneJob(job));
    }

    private void UpdateQuestCloneProgress(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot)
    {
        if (snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null)
            return;

        int count = Mathf.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length);
        int stride = Mathf.Max(1, questCloneProgressSampleStride);
        Vector3 cameraPosition = snapshot.hasDispatchCameraPose
            ? snapshot.dispatchCameraPosition
            : snapshot.hasSnapshotCameraPose ? snapshot.snapshotCameraPosition : Vector3.zero;
        for (int i = 0; i < count; i += stride)
        {
            if (!IsRawDepthSnapshotPixelValid(snapshot, i))
                continue;

            Color meta = snapshot.observationMeta[i];
            bool risk = meta.g < Mathf.Clamp01(questCloneProgressLowConfidenceThreshold);
            AddRoomRawDepthCompletionSample(snapshot.worldPositions[i], cameraPosition, meta.b, risk);
        }
    }

    private void WriteQuestCloneJob(QuestCloneWriteJob job)
    {
        try
        {
            WriteQuestCloneSnapshotBinary(job.rightPath, job.right);
            WriteQuestCloneSnapshotBinary(job.leftPath, job.left);
            File.WriteAllText(job.metadataPath, job.metadataJson, Encoding.UTF8);
            lock (_questCloneWriteLock)
                File.AppendAllText(job.manifestPath, job.manifestRow, Encoding.UTF8);
            _questCloneLastWriteError = string.Empty;
        }
        catch (Exception ex)
        {
            _questCloneLastWriteError = ex.GetType().Name + ": " + ex.Message;
            Interlocked.Increment(ref _questCloneDroppedWrites);
        }
        finally
        {
            Interlocked.Decrement(ref _questCloneWritesInFlight);
        }
    }

    private static void WriteQuestCloneSnapshotBinary(string path, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot)
    {
        int count = snapshot != null && snapshot.worldPositions != null && snapshot.observationMeta != null
            ? Math.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length)
            : 0;
        using (FileStream stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 65536, FileOptions.SequentialScan))
        using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8))
        {
            writer.Write(Encoding.ASCII.GetBytes("SCQ3BIN2"));
            writer.Write(2);
            writer.Write(snapshot != null ? snapshot.sourceEyeIndex : -1);
            writer.Write(snapshot != null ? snapshot.frameIndex : -1);
            writer.Write(snapshot != null ? snapshot.resolutionWidth : 0);
            writer.Write(snapshot != null ? snapshot.resolutionHeight : 0);
            writer.Write(count);
            writer.Write(snapshot != null ? snapshot.dispatchRealtimeSeconds : 0d);
            writer.Write(snapshot != null ? snapshot.completionRealtimeSeconds : 0d);
            WriteQuestClonePose(writer, snapshot != null && snapshot.hasDispatchCameraPose,
                snapshot != null ? snapshot.dispatchCameraPosition : Vector3.zero,
                snapshot != null ? snapshot.dispatchCameraRotation : Quaternion.identity);
            WriteQuestCloneMatrix(writer, snapshot != null && snapshot.hasProjectionMatrix, snapshot != null ? snapshot.projectionMatrix : Matrix4x4.identity);
            WriteQuestCloneMatrix(writer, snapshot != null && snapshot.hasWorldToCameraMatrix, snapshot != null ? snapshot.worldToCameraMatrix : Matrix4x4.identity);
            WriteQuestCloneMatrix(writer, snapshot != null && snapshot.hasDepthReprojectionMatrix, snapshot != null ? snapshot.depthReprojectionMatrix : Matrix4x4.identity);

            for (int i = 0; i < count; i++)
            {
                Vector3 point = snapshot.worldPositions[i];
                Color meta = snapshot.observationMeta[i];
                bool valid = meta.r >= 0.5f && meta.b > 0f &&
                    float.IsFinite(point.x) && float.IsFinite(point.y) && float.IsFinite(point.z);
                writer.Write((byte)(valid ? 1 : 0));
                writer.Write(point.x); writer.Write(point.y); writer.Write(point.z);
                writer.Write(meta.r); writer.Write(meta.g); writer.Write(meta.b); writer.Write(meta.a);
            }
        }
    }

    private static void WriteQuestClonePose(BinaryWriter writer, bool valid, Vector3 position, Quaternion rotation)
    {
        writer.Write((byte)(valid ? 1 : 0));
        writer.Write(position.x); writer.Write(position.y); writer.Write(position.z);
        writer.Write(rotation.x); writer.Write(rotation.y); writer.Write(rotation.z); writer.Write(rotation.w);
    }

    private static void WriteQuestCloneMatrix(BinaryWriter writer, bool valid, Matrix4x4 matrix)
    {
        writer.Write((byte)(valid ? 1 : 0));
        for (int row = 0; row < 4; row++)
            for (int column = 0; column < 4; column++)
                writer.Write(matrix[row, column]);
    }

    private string BuildQuestCloneCaptureMetadata(
        string frameName,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot right,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot left,
        Transform pose,
        string rightPath,
        string leftPath)
    {
        StringBuilder json = new StringBuilder(4096);
        json.AppendLine("{");
        AppendJsonString(json, "schema", "ScanCoverQuestCloneCapture/v2", 1, true);
        AppendJsonString(json, "frame", frameName ?? "", 1, true);
        AppendJsonString(json, "target", questCloneTargetLabel ?? "", 1, true);
        AppendJsonString(json, "action", questCloneActionLabel ?? "", 1, true);
        AppendJsonNumber(json, "interEyeDeltaMs", _questCloneLastInterEyeDeltaMs, 1, true);
        AppendJsonString(json, "rightBinary", rightPath ?? "", 1, true);
        AppendJsonString(json, "leftBinary", leftPath ?? "", 1, true);
        AppendIndent(json, 1); json.AppendLine("\"headPoseAtExport\": {");
        AppendJsonVector(json, "position", pose != null ? pose.position : Vector3.zero, 2, true);
        AppendJsonQuaternion(json, "rotation", pose != null ? pose.rotation : Quaternion.identity, 2, false);
        AppendIndent(json, 1); json.AppendLine("},");
        AppendIndent(json, 1); json.AppendLine("\"eyes\": [");
        AppendQuestCloneEyeMetadata(json, "Right", right, 2, true);
        AppendQuestCloneEyeMetadata(json, "Left", left, 2, false);
        AppendIndent(json, 1); json.AppendLine("]");
        json.AppendLine("}");
        return json.ToString();
    }

    private static void AppendQuestCloneEyeMetadata(StringBuilder json, string name, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, int indent, bool trailingComma)
    {
        AppendIndent(json, indent); json.AppendLine("{");
        AppendJsonString(json, "name", name, indent + 1, true);
        AppendJsonNumber(json, "sourceEyeIndex", snapshot != null ? snapshot.sourceEyeIndex : -1, indent + 1, true);
        AppendJsonNumber(json, "frame", snapshot != null ? snapshot.frameIndex : -1, indent + 1, true);
        AppendJsonNumber(json, "dispatchRealtimeSeconds", snapshot != null ? snapshot.dispatchRealtimeSeconds : 0d, indent + 1, true);
        AppendJsonNumber(json, "completionRealtimeSeconds", snapshot != null ? snapshot.completionRealtimeSeconds : 0d, indent + 1, true);
        AppendJsonBool(json, "hasDispatchPose", snapshot != null && snapshot.hasDispatchCameraPose, indent + 1, true);
        AppendJsonBool(json, "hasProjectionMatrix", snapshot != null && snapshot.hasProjectionMatrix, indent + 1, true);
        AppendJsonBool(json, "hasWorldToCameraMatrix", snapshot != null && snapshot.hasWorldToCameraMatrix, indent + 1, true);
        AppendJsonBool(json, "hasDepthReprojectionMatrix", snapshot != null && snapshot.hasDepthReprojectionMatrix, indent + 1, true);
        AppendJsonVector(json, "dispatchPosition", snapshot != null ? snapshot.dispatchCameraPosition : Vector3.zero, indent + 1, true);
        AppendJsonQuaternion(json, "dispatchRotation", snapshot != null ? snapshot.dispatchCameraRotation : Quaternion.identity, indent + 1, true);
        AppendJsonMatrix(json, "projectionMatrix", snapshot != null ? snapshot.projectionMatrix : Matrix4x4.identity, indent + 1, true);
        AppendJsonMatrix(json, "worldToCameraMatrix", snapshot != null ? snapshot.worldToCameraMatrix : Matrix4x4.identity, indent + 1, true);
        AppendJsonMatrix(json, "depthReprojectionMatrix", snapshot != null ? snapshot.depthReprojectionMatrix : Matrix4x4.identity, indent + 1, false);
        AppendIndent(json, indent); json.Append('}');
        if (trailingComma) json.Append(',');
        json.AppendLine();
    }

    private string BuildQuestCloneManifestRow(string frameName, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot right, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot left, string rightPath, string leftPath, string metadataPath, string status)
    {
        return EscapeCsv(frameName) + "," + right.frameIndex.ToString(CultureInfo.InvariantCulture) + "," + left.frameIndex.ToString(CultureInfo.InvariantCulture) + "," +
            right.dispatchRealtimeSeconds.ToString("R", CultureInfo.InvariantCulture) + "," + left.dispatchRealtimeSeconds.ToString("R", CultureInfo.InvariantCulture) + "," +
            _questCloneLastInterEyeDeltaMs.ToString("0.###", CultureInfo.InvariantCulture) + "," + EscapeCsv(questCloneTargetLabel) + "," + EscapeCsv(questCloneActionLabel) + "," +
            EscapeCsv(rightPath) + "," + EscapeCsv(leftPath) + "," + EscapeCsv(metadataPath) + "," + EscapeCsv(status) + Environment.NewLine;
    }

    private void AppendQuestCloneDroppedRow(string frameName, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot right, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot left, string status)
    {
        if (string.IsNullOrEmpty(_questCloneBinaryManifestPath))
            return;
        string row = BuildQuestCloneManifestRow(frameName, right, left, "", "", "", status);
        File.AppendAllText(_questCloneBinaryManifestPath, row, Encoding.UTF8);
    }

    private int GetLatestRawDepthSnapshotFrameIndex()
    {
        if (depthGridPointCloud == null ||
            !depthGridPointCloud.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot) ||
            snapshot == null)
            return -1;

        return snapshot.frameIndex;
    }

    private bool FailBinocularRoomRawDepthSnapshotCapture(string reason)
    {
        _binocularRoomRawDepthStatus = "失败: " + (reason ?? "unknown");
        RestoreBinocularRoomRawDepthOriginalEye();
        _binocularRoomRawDepthSnapshotStage = BinocularRoomRawDepthSnapshotStage.None;
        _binocularRoomRawDepthRightSnapshot = null;
        _binocularRoomRawDepthLeftSnapshot = null;
        _binocularRoomRawDepthFusedSnapshot = null;
        _pendingCapture = false;
        _pendingRefreshRequested = false;
        _independentSnapshotCaptureActive = false;
        return false;
    }

    private void ResetBinocularRoomRawDepthSnapshotCapture(bool restoreOriginalEye)
    {
        if (restoreOriginalEye)
            RestoreBinocularRoomRawDepthOriginalEye();

        _binocularRoomRawDepthSnapshotStage = BinocularRoomRawDepthSnapshotStage.None;
        _binocularRoomRawDepthRightSnapshot = null;
        _binocularRoomRawDepthLeftSnapshot = null;
        _binocularRoomRawDepthFusedSnapshot = null;
        _binocularRoomRawDepthStartSnapshotIndex = -1;
        _binocularRoomRawDepthStartTime = 0f;
    }

    private void RestoreBinocularRoomRawDepthOriginalEye()
    {
        ScanCoverDepthPreprocessor preprocessor = depthGridPointCloud != null ? depthGridPointCloud.Preprocessor : null;
        if (preprocessor != null)
            preprocessor.SetSourceEye(_binocularRoomRawDepthOriginalEye);
    }

    private void TryExportLatestRoomRawDepthSnapshot(string frameName, Transform pose)
    {
        if (!ShouldExportRoomRawDepthSnapshots() || depthGridPointCloud == null)
            return;

        if (!depthGridPointCloud.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot))
            return;

        MaybeExportRoomRawDepthSnapshot(frameName, snapshot, pose);
    }

    private void MaybeExportRoomRawDepthSnapshot(string frameName, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Transform pose)
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthSnapshots() || snapshot == null || string.IsNullOrEmpty(_roomRawDepthSnapshotDirectory))
            return;

        Directory.CreateDirectory(_roomRawDepthSnapshotDirectory);
        string safeFrameName = string.IsNullOrWhiteSpace(frameName) ? $"frame_{_capturedFrameCount:D4}" : frameName;
        foreach (char invalid in Path.GetInvalidFileNameChars())
            safeFrameName = safeFrameName.Replace(invalid, '_');
        string path = Path.Combine(_roomRawDepthSnapshotDirectory, $"{safeFrameName}_raw{snapshot.frameIndex:D6}_raw_snapshot.csv");

        int totalPixels = WriteRoomRawDepthSnapshotCsv(path, snapshot, pose, out int validPixels, out int visiblePoints);
        if (totalPixels <= 0)
        {
            AppendRoomRawDepthSnapshotManifestRow(frameName, snapshot.frameIndex, 0, 0, 0, snapshot.resolutionWidth, "", "skipped-empty", pose);
            return;
        }

        _roomRawDepthSnapshotExportedFrames++;
        _roomRawDepthSnapshotTotalPixels += totalPixels;
        _roomRawDepthSnapshotValidPixels += validPixels;
        AppendRoomRawDepthSnapshotManifestRow(frameName, snapshot.frameIndex, totalPixels, validPixels, visiblePoints, snapshot.resolutionWidth, path, "exported", pose);
        ExportVirtualCloneInputMetadata(frameName, snapshot, pose, path, totalPixels, validPixels);
    }

    private void ExportVirtualCloneInputMetadata(
        string frameName,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot,
        Transform pose,
        string rawSnapshotCsvPath,
        int totalPixels,
        int validPixels)
    {
        if (!_sessionFileExportEnabled || !ShouldExportVirtualCloneInputMetadata() ||
            snapshot == null || string.IsNullOrEmpty(_virtualCloneInputDirectory) ||
            string.IsNullOrEmpty(_virtualCloneInputManifestPath))
            return;

        int rightCount = 0;
        int leftCount = 0;
        if (_binocularRoomRawDepthRightSnapshot != null && _binocularRoomRawDepthLeftSnapshot != null)
        {
            rightCount = SnapshotCount(_binocularRoomRawDepthRightSnapshot);
            leftCount = SnapshotCount(_binocularRoomRawDepthLeftSnapshot);
        }

        if (rightCount <= 0 || leftCount <= 0)
        {
            int half = totalPixels / 2;
            rightCount = half;
            leftCount = Mathf.Max(0, totalPixels - half);
        }

        if (rightCount <= 0 || leftCount <= 0)
            return;

        Directory.CreateDirectory(_virtualCloneInputDirectory);
        string safeFrameName = string.IsNullOrWhiteSpace(frameName) ? $"frame_{_capturedFrameCount:D4}" : frameName;
        foreach (char invalid in Path.GetInvalidFileNameChars())
            safeFrameName = safeFrameName.Replace(invalid, '_');

        string metadataPath = Path.Combine(_virtualCloneInputDirectory, $"{safeFrameName}_virtual_clone_input.json");
        float fov = captureCamera != null ? captureCamera.fieldOfView : virtualCloneFallbackFieldOfViewDegrees;
        float aspect = captureCamera != null ? captureCamera.aspect : snapshot.resolutionWidth / (float)Mathf.Max(1, snapshot.resolutionHeight / 2);

        Vector3 position = pose != null ? pose.position : Vector3.zero;
        Vector3 forward = pose != null ? pose.forward : Vector3.forward;
        Vector3 right = pose != null ? pose.right : Vector3.right;
        Vector3 up = pose != null ? pose.up : Vector3.up;
        Vector3 rotation = pose != null ? pose.rotation.eulerAngles : Vector3.zero;

        StringBuilder json = new StringBuilder(2048);
        json.AppendLine("{");
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot rightSnapshot = _binocularRoomRawDepthRightSnapshot;
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot leftSnapshot = _binocularRoomRawDepthLeftSnapshot;
        double rightDispatch = rightSnapshot != null ? rightSnapshot.dispatchRealtimeSeconds : 0d;
        double leftDispatch = leftSnapshot != null ? leftSnapshot.dispatchRealtimeSeconds : 0d;
        double interEyeDeltaMs = Math.Abs(leftDispatch - rightDispatch) * 1000d;

        AppendJsonString(json, "schema", "ScanCoverVirtualCloneInput/v2", 1, true);
        AppendJsonString(json, "frame", frameName ?? "", 1, true);
        AppendJsonString(json, "target", questCloneTargetLabel ?? "", 1, true);
        AppendJsonString(json, "action", questCloneActionLabel ?? "", 1, true);
        AppendJsonNumber(json, "interEyeDeltaMs", interEyeDeltaMs, 1, true);
        AppendJsonNumber(json, "rawDepthFrame", snapshot.frameIndex, 1, true);
        AppendJsonString(json, "rawSnapshotCsv", rawSnapshotCsvPath ?? "", 1, true);
        AppendJsonNumber(json, "totalPixels", totalPixels, 1, true);
        AppendJsonNumber(json, "validPixels", validPixels, 1, true);
        AppendJsonNumber(json, "width", Mathf.Max(1, snapshot.resolutionWidth), 1, true);
        AppendJsonNumber(json, "height", Mathf.Max(1, snapshot.resolutionHeight), 1, true);
        AppendJsonNumber(json, "fieldOfViewDegrees", fov, 1, true);
        AppendJsonNumber(json, "aspect", aspect, 1, true);
        AppendJsonNumber(json, "eyeBaselineMeters", virtualCloneEyeBaselineMeters, 1, true);
        AppendIndent(json, 1); json.AppendLine("\"pose\": {");
        AppendJsonVector(json, "position", position, 2, true);
        AppendJsonVector(json, "rotationEuler", rotation, 2, true);
        AppendJsonVector(json, "forward", forward, 2, true);
        AppendJsonVector(json, "right", right, 2, true);
        AppendJsonVector(json, "up", up, 2, false);
        AppendIndent(json, 1); json.AppendLine("},");
        AppendIndent(json, 1); json.AppendLine("\"captureEyes\": [");
        AppendQuestCloneEyeMetadata(json, "Right", rightSnapshot, 2, true);
        AppendQuestCloneEyeMetadata(json, "Left", leftSnapshot, 2, false);
        AppendIndent(json, 1); json.AppendLine("],");
        AppendIndent(json, 1); json.AppendLine("\"eyes\": [");
        AppendEyeSegmentJson(json, "Right", 0, rightCount, snapshot.resolutionWidth, 2, true);
        AppendEyeSegmentJson(json, "Left", rightCount, leftCount, snapshot.resolutionWidth, 2, false);
        AppendIndent(json, 1); json.AppendLine("]");
        json.AppendLine("}");
        File.WriteAllText(metadataPath, json.ToString(), Encoding.UTF8);

        StringBuilder row = new StringBuilder(512);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(snapshot.frameIndex).Append(',')
            .Append(Mathf.Max(1, snapshot.resolutionWidth)).Append(',')
            .Append(Mathf.Max(1, snapshot.resolutionHeight)).Append(',')
            .Append(0).Append(',')
            .Append(rightCount).Append(',')
            .Append(rightCount).Append(',')
            .Append(leftCount).Append(',')
            .Append(FormatFloat(virtualCloneEyeBaselineMeters)).Append(',')
            .Append(FormatFloat(fov)).Append(',')
            .Append(FormatFloat(aspect)).Append(',')
            .Append(rightDispatch.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(leftDispatch.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(interEyeDeltaMs.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
            .Append(EscapeCsv(questCloneTargetLabel)).Append(',')
            .Append(EscapeCsv(questCloneActionLabel)).Append(',')
            .Append(EscapeCsv(rawSnapshotCsvPath)).Append(',')
            .Append(EscapeCsv(metadataPath)).Append(',')
            .Append("exported").AppendLine();
        File.AppendAllText(_virtualCloneInputManifestPath, row.ToString(), Encoding.UTF8);
    }

    private static int SnapshotCount(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot)
    {
        if (snapshot == null || snapshot.worldPositions == null || snapshot.worldNormals == null || snapshot.observationMeta == null)
            return 0;
        return Mathf.Min(snapshot.worldPositions.Length, Mathf.Min(snapshot.worldNormals.Length, snapshot.observationMeta.Length));
    }

    private int WriteRoomRawDepthSnapshotCsv(string path, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Transform pose, out int validPixels, out int visiblePoints)
    {
        validPixels = 0;
        visiblePoints = 0;
        ClearRoomRawDepthSnapshotOverlay();

        if (snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null)
            return 0;

        int width = Mathf.Max(1, snapshot.resolutionWidth);
        int height = Mathf.Max(1, snapshot.resolutionHeight);
        int count = Mathf.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length);
        if (count <= 0)
            return 0;

        Vector3 cameraPosition = pose != null ? pose.position : Vector3.zero;
        Vector3 cameraForward = pose != null ? pose.forward : Vector3.forward;
        if (!IsFinite(cameraForward) || cameraForward.sqrMagnitude <= 1e-8f)
            cameraForward = Vector3.forward;
        cameraForward.Normalize();
        Vector3 cameraRight = pose != null ? pose.right : Vector3.right;
        if (!IsFinite(cameraRight) || cameraRight.sqrMagnitude <= 1e-8f)
            cameraRight = Vector3.right;
        cameraRight.Normalize();
        Vector3 cameraUp = pose != null ? pose.up : Vector3.up;
        if (!IsFinite(cameraUp) || cameraUp.sqrMagnitude <= 1e-8f)
            cameraUp = Vector3.up;
        cameraUp.Normalize();

        int maxVisual = Mathf.Max(1, roomRawDepthSnapshotMaxVisualPoints);
        float pointSize = Mathf.Max(0.002f, roomRawDepthSnapshotPointSize);
        bool fuseVisualOverlay = fuseRoomRawDepthSnapshotOverlayOnly;
        float visualFuseVoxel = Mathf.Max(0.002f, roomRawDepthSnapshotOverlayFuseVoxelMeters);
        HashSet<Vector3Int> visualOverlayVoxels = fuseVisualOverlay ? new HashSet<Vector3Int>() : null;
        StringBuilder rows = new StringBuilder(Mathf.Max(4096, count * 128));
        rows.AppendLine("# ScanCover full raw-depth view snapshot");
        rows.Append("resolution=").Append(width).Append('x').Append(height).AppendLine();
        rows.Append("index,pixelX,pixelY,u,v,valid,depthM,forwardDepthM,worldX,worldY,worldZ,normalX,normalY,normalZ,confidence").AppendLine();

        for (int i = 0; i < count; i++)
        {
            int x = i % width;
            int y = i / width;
            float u = width > 1 ? x / (float)(width - 1) : 0.5f;
            float v = height > 1 ? y / (float)(height - 1) : 0.5f;
            bool valid = IsRawDepthSnapshotPixelValid(snapshot, i);
            Color meta = snapshot.observationMeta[i];

            rows.Append(i).Append(',')
                .Append(x).Append(',')
                .Append(y).Append(',')
                .Append(FormatFloat(u)).Append(',')
                .Append(FormatFloat(v)).Append(',')
                .Append(valid ? 1 : 0).Append(',');

            if (valid)
            {
                Vector3 point = snapshot.worldPositions[i];
                Vector3 normal = (snapshot.worldNormals != null && i < snapshot.worldNormals.Length && IsFinite(snapshot.worldNormals[i]) && snapshot.worldNormals[i].sqrMagnitude > 1e-8f)
                    ? snapshot.worldNormals[i].normalized
                    : Vector3.up;
                float forwardDepth = Vector3.Dot(point - cameraPosition, cameraForward);
                rows.Append(FormatFloat(meta.b)).Append(',')
                    .Append(FormatFloat(forwardDepth)).Append(',')
                    .Append(FormatVector(point)).Append(',')
                    .Append(FormatVector(normal)).Append(',')
                    .Append(FormatFloat(meta.g)).AppendLine();

                validPixels++;
                if (_roomRawDepthSnapshotMatrices.Count < maxVisual)
                {
                    bool showVisualPoint = true;
                    if (fuseVisualOverlay)
                    {
                        Vector3Int key = new Vector3Int(
                            Mathf.RoundToInt(point.x / visualFuseVoxel),
                            Mathf.RoundToInt(point.y / visualFuseVoxel),
                            Mathf.RoundToInt(point.z / visualFuseVoxel));
                        showVisualPoint = visualOverlayVoxels.Add(key);
                    }

                    if (showVisualPoint)
                    {
                        _roomRawDepthSnapshotMatrices.Add(Matrix4x4.TRS(point, Quaternion.identity, Vector3.one * pointSize));
                        AddRoomRawDepthSnapshotQuad(point, cameraRight, cameraUp, pointSize);
                        visiblePoints++;
                    }
                }
            }
            else
            {
                rows.Append(",,,,,,,,,").AppendLine();
            }
        }

        _roomRawDepthSnapshotVisiblePoints = visiblePoints;
        _roomRawDepthSnapshotLastTotalPixels = count;
        _roomRawDepthSnapshotLastValidPixels = validPixels;
        _roomRawDepthSnapshotLastFrame = Path.GetFileNameWithoutExtension(path);
        UploadRoomRawDepthSnapshotOverlayMesh();
        File.WriteAllText(path, rows.ToString(), Encoding.UTF8);
        return count;
    }

    private void AppendRoomRawDepthSnapshotManifestRow(string frameName, int rawDepthFrameIndex, int totalPixels, int validPixels, int visiblePoints, int width, string path, string status, Transform pose)
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthSnapshots() || string.IsNullOrEmpty(_roomRawDepthSnapshotManifestPath))
            return;

        Vector3 position = pose != null ? pose.position : Vector3.zero;
        Vector3 rotation = pose != null ? pose.rotation.eulerAngles : Vector3.zero;
        int height = totalPixels > 0 && width > 0 ? Mathf.Max(1, Mathf.CeilToInt(totalPixels / (float)width)) : 0;
        StringBuilder row = new StringBuilder(384);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(rawDepthFrameIndex).Append(',')
            .Append(totalPixels).Append(',')
            .Append(validPixels).Append(',')
            .Append(visiblePoints).Append(',')
            .Append(width).Append(',')
            .Append(height).Append(',')
            .Append(FormatVector(position)).Append(',')
            .Append(FormatVector(rotation)).Append(',')
            .Append(EscapeCsv(path)).Append(',')
            .Append(EscapeCsv(status)).AppendLine();
        File.AppendAllText(_roomRawDepthSnapshotManifestPath, row.ToString(), Encoding.UTF8);
    }

    private void MaybeExportRoomRawDepthFrame(string frameName, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Transform pose, int frameIndex, RoomRawCoverageFrameResult result)
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthFrames() || snapshot == null || string.IsNullOrEmpty(_roomRawDepthDirectory))
            return;

        bool accepted = result.hasData && result.accepted;
        if (enforceObservationOrderScoreGate && result.observationOrderScore.hasData && result.observationOrderScore.disordered)
        {
            _roomRawDepthSkippedFrames++;
            AppendRoomRawDepthManifestRow(frameName, snapshot.frameIndex, accepted, result, 0, 0, "", "skipped-observation-order", pose);
            return;
        }

        if (exportRoomRawDepthOnlyAcceptedFrames && !accepted)
        {
            _roomRawDepthSkippedFrames++;
            AppendRoomRawDepthManifestRow(frameName, snapshot.frameIndex, accepted, result, 0, 0, "", "skipped-rejected", pose);
            return;
        }

        int stride = Mathf.Max(1, roomRawDepthFrameStride);
        if (stride > 1 && frameIndex % stride != 0)
        {
            _roomRawDepthSkippedFrames++;
            AppendRoomRawDepthManifestRow(frameName, snapshot.frameIndex, accepted, result, 0, 0, "", "skipped-stride", pose);
            return;
        }

        Directory.CreateDirectory(_roomRawDepthDirectory);
        string safeFrameName = string.IsNullOrWhiteSpace(frameName) ? $"frame_{frameIndex:D4}" : frameName;
        foreach (char invalid in Path.GetInvalidFileNameChars())
            safeFrameName = safeFrameName.Replace(invalid, '_');
        string path = Path.Combine(_roomRawDepthDirectory, $"{safeFrameName}_raw_depth.csv");

        int exportedSamples = WriteRoomRawDepthFrameCsv(path, snapshot, pose, result);
        if (exportedSamples > 0)
        {
            _roomRawDepthExportedFrames++;
            _roomRawDepthExportedSamples += exportedSamples;
            AppendRoomRawDepthManifestRow(frameName, snapshot.frameIndex, accepted, result, exportedSamples, snapshot.resolutionWidth, path, "exported", pose);
        }
        else
        {
            _roomRawDepthSkippedFrames++;
            AppendRoomRawDepthManifestRow(frameName, snapshot.frameIndex, accepted, result, 0, snapshot.resolutionWidth, "", "skipped-empty", pose);
        }
    }

    private int WriteRoomRawDepthFrameCsv(string path, ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot, Transform pose, RoomRawCoverageFrameResult result)
    {
        if (!_sessionFileExportEnabled)
            return 0;

        if (snapshot == null || snapshot.worldPositions == null || snapshot.observationMeta == null)
            return 0;

        int width = Mathf.Max(1, snapshot.resolutionWidth);
        int height = Mathf.Max(1, snapshot.resolutionHeight);
        int count = Mathf.Min(snapshot.worldPositions.Length, snapshot.observationMeta.Length);
        if (count <= 0)
            return 0;

        Vector3 cameraPosition = pose != null ? pose.position : Vector3.zero;
        Vector3 cameraForward = pose != null ? pose.forward : Vector3.forward;
        if (!IsFinite(cameraForward) || cameraForward.sqrMagnitude <= 1e-8f)
            cameraForward = Vector3.forward;
        cameraForward.Normalize();

        float riskAngle = Mathf.Clamp(roomRawCoverageRiskViewAngleDegrees, 0f, 89f);
        float riskFacingMin = Mathf.Cos(riskAngle * Mathf.Deg2Rad);
        float highMinY = cameraPosition.y + Mathf.Max(0.1f, roomRawCoverageHighPointMinDeltaYMeters);
        float lowMaxY = cameraPosition.y - Mathf.Max(0.1f, roomRawCoverageLowPointMinDeltaYMeters);
        float jumpMeters = Mathf.Max(0.001f, roomRawCoverageNeighborDepthJumpMeters);
        int maxSamples = Mathf.Max(0, roomRawDepthMaxSamplesPerFrame);
        int step = maxSamples > 0 ? Mathf.Max(1, Mathf.CeilToInt(count / (float)maxSamples)) : 1;

        int exported = 0;
        StringBuilder rows = new StringBuilder(4096);
        rows.AppendLine("# ScanCover dense raw-depth frame samples");
        rows.Append("resolution=").Append(width).Append('x').Append(height).AppendLine();
        rows.Append("index,pixelX,pixelY,u,v,focus,risk,orderState,orderScore,high,low,depthM,forwardDepthM,projectionErrorM,viewAngleDeg,worldX,worldY,worldZ,normalX,normalY,normalZ,confidence").AppendLine();

        for (int i = 0; i < count; i += step)
        {
            if (!IsRawDepthSampleUsable(snapshot, i, cameraPosition, cameraForward))
                continue;

            int x = i % width;
            int y = i / width;
            float u = width > 1 ? x / (float)(width - 1) : 0.5f;
            float v = height > 1 ? y / (float)(height - 1) : 0.5f;
            bool focus = GetRoomRawCoverageFocusFlags(u, v, out _, out _);
            if (roomRawDepthFocusOnly && !focus)
                continue;

            Vector3 point = snapshot.worldPositions[i];
            Vector3 normal = (snapshot.worldNormals != null && i < snapshot.worldNormals.Length && IsFinite(snapshot.worldNormals[i]) && snapshot.worldNormals[i].sqrMagnitude > 1e-8f)
                ? snapshot.worldNormals[i].normalized
                : Vector3.up;
            Color meta = snapshot.observationMeta[i];
            Vector3 toPoint = point - cameraPosition;
            float forwardDepth = Vector3.Dot(toPoint, cameraForward);
            float projectionError = Mathf.Abs(forwardDepth - meta.b);
            Vector3 toCamera = cameraPosition - point;
            float viewAngleDeg = 90f;
            byte orderState = GetRawObservationOrderState(result.rawObservationOrder, i);
            bool risk = IsRawDepthNeighborJump(snapshot, i, width, height, jumpMeters, cameraPosition, cameraForward) ||
                IsRawObservationOrderRisk(result.rawObservationOrder, i);
            if (toCamera.sqrMagnitude > 1e-8f)
            {
                float facing = Mathf.Abs(Vector3.Dot(normal, toCamera.normalized));
                viewAngleDeg = Mathf.Acos(Mathf.Clamp(facing, -1f, 1f)) * Mathf.Rad2Deg;
                if (facing < riskFacingMin)
                    risk = true;
            }

            int snapshotMaskSlot;
            if (!AcceptSnapshotGridRawDepthSample(point, risk, out snapshotMaskSlot))
                continue;

            AddRoomRawDepthCompletionSample(point, cameraPosition, meta.b, risk);

            rows.Append(i).Append(',')
                .Append(x).Append(',')
                .Append(y).Append(',')
                .Append(FormatFloat(u)).Append(',')
                .Append(FormatFloat(v)).Append(',')
                .Append(focus ? 1 : 0).Append(',')
                .Append(risk ? 1 : 0).Append(',')
                .Append(orderState).Append(',')
                .Append(FormatFloat(orderState >= 3 ? 0f : (orderState > 0 ? 1f : result.observationOrderScore.score01))).Append(',')
                .Append(point.y >= highMinY ? 1 : 0).Append(',')
                .Append(point.y <= lowMaxY ? 1 : 0).Append(',')
                .Append(FormatFloat(meta.b)).Append(',')
                .Append(FormatFloat(forwardDepth)).Append(',')
                .Append(FormatFloat(projectionError)).Append(',')
                .Append(FormatFloat(viewAngleDeg)).Append(',')
                .Append(FormatVector(point)).Append(',')
                .Append(FormatVector(normal)).Append(',')
                .Append(FormatFloat(meta.g)).AppendLine();
            exported++;
        }

        RefreshSnapshotGridCaptureCounts();

        if (exported <= 0)
            return 0;

        File.WriteAllText(path, rows.ToString(), Encoding.UTF8);
        return exported;
    }

    private void AppendRoomRawDepthManifestRow(string frameName, int rawDepthFrameIndex, bool accepted, RoomRawCoverageFrameResult result, int exportedSamples, int width, string path, string status, Transform pose)
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthFrames() || string.IsNullOrEmpty(_roomRawDepthManifestPath))
            return;

        Vector3 position = pose != null ? pose.position : Vector3.zero;
        Vector3 rotation = pose != null ? pose.rotation.eulerAngles : Vector3.zero;
        int height = result.totalSamples > 0 && width > 0 ? Mathf.Max(1, Mathf.CeilToInt(result.totalSamples / (float)width)) : 0;
        StringBuilder row = new StringBuilder(512);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(rawDepthFrameIndex).Append(',')
            .Append(accepted ? 1 : 0).Append(',')
            .Append(result.totalSamples).Append(',')
            .Append(result.validSamples).Append(',')
            .Append(exportedSamples).Append(',')
            .Append(width).Append(',')
            .Append(height).Append(',')
            .Append(FormatVector(position)).Append(',')
            .Append(FormatVector(rotation)).Append(',')
            .Append(EscapeCsv(path)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.score01)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.rawOrderRatio)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.rawBadEdgeRatio)).Append(',')
            .Append(result.observationOrderScore.rawBadEdges).Append(',')
            .Append(result.observationOrderScore.rawTestedEdges).Append(',')
            .Append(EscapeCsv(result.observationOrderScore.reason)).Append(',')
            .Append(EscapeCsv(status)).AppendLine();
        File.AppendAllText(_roomRawDepthManifestPath, row.ToString(), Encoding.UTF8);
    }

    private void AppendRoomRawCoverageFrameRow(string frameName, int rawDepthFrameIndex, RoomRawCoverageFrameResult result)
    {
        if (!_sessionFileExportEnabled || !trackRoomRawCoverage || string.IsNullOrEmpty(_roomRawCoverageFramesPath))
            return;

        StringBuilder row = new StringBuilder(256);
        row.Append(EscapeCsv(frameName)).Append(',')
            .Append(rawDepthFrameIndex).Append(',')
            .Append(result.accepted ? 1 : 0).Append(',')
            .Append(result.totalSamples).Append(',')
            .Append(result.validSamples).Append(',')
            .Append(result.frameVoxels).Append(',')
            .Append(result.focusVoxels).Append(',')
            .Append(result.coreVoxels).Append(',')
            .Append(result.edgeBufferVoxels).Append(',')
            .Append(result.focusNewVoxels).Append(',')
            .Append(result.edgeBufferNewVoxels).Append(',')
            .Append(result.newVoxels).Append(',')
            .Append(result.rehitVoxels).Append(',')
            .Append(result.historyRehitVoxels).Append(',')
            .Append(result.parallaxRehitVoxels).Append(',')
            .Append(result.newStableVoxels).Append(',')
            .Append(result.highFrameVoxels).Append(',')
            .Append(result.newHighVoxels).Append(',')
            .Append(result.newHighStableVoxels).Append(',')
            .Append(result.lowFrameVoxels).Append(',')
            .Append(result.newLowVoxels).Append(',')
            .Append(result.newLowStableVoxels).Append(',')
            .Append(_roomRawCoverageCoveredVoxels).Append(',')
            .Append(_roomRawCoverageStableVoxels).Append(',')
            .Append(_roomRawCoverageHighVoxels).Append(',')
            .Append(_roomRawCoverageHighStableVoxels).Append(',')
            .Append(_roomRawCoverageLowVoxels).Append(',')
            .Append(_roomRawCoverageLowStableVoxels).Append(',')
            .Append(result.riskSamples).Append(',')
            .Append(result.focusRiskSamples).Append(',')
            .Append(result.riskVoxels).Append(',')
            .Append(FormatFloat(result.riskRatio)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.score01)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.rawOrderRatio)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.rawCenterOrderRatio)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.rawBadEdgeRatio)).Append(',')
            .Append(FormatFloat(result.observationOrderScore.rawBadQuadRatio)).Append(',')
            .Append(result.observationOrderScore.rawTestedEdges).Append(',')
            .Append(result.observationOrderScore.rawBadEdges).Append(',')
            .Append(result.observationOrderScore.rawTestedQuads).Append(',')
            .Append(result.observationOrderScore.rawBadQuads).Append(',')
            .Append(EscapeCsv(result.observationOrderScore.reason)).Append(',')
            .Append(_roomRawCoveragePoseCellCount).Append(',')
            .Append(_roomRawCoverageFrames).Append(',')
            .Append(_roomRawCoverageRejectedFrames).Append(',')
            .Append(FormatFloat(result.anchorAngle)).Append(',')
            .Append(FormatFloat(result.anchorMove)).Append(',')
            .Append(result.anchorFallback ? 1 : 0).Append(',')
            .Append(FormatFloat(result.overlapRatio)).Append(',')
            .Append(result.historyAgreementVoxels).Append(',')
            .Append(result.historyConflictVoxels).Append(',')
            .Append(FormatFloat(result.historyAgreementRatio)).Append(',')
            .Append(_roomRawCoverageCoveredVoxels).Append(',')
            .Append(_roomRawCoverageStableVoxels).Append(',')
            .Append(EscapeCsv(result.status)).AppendLine();
        File.AppendAllText(_roomRawCoverageFramesPath, row.ToString(), Encoding.UTF8);
    }

    private void WriteRoomRawCoverageOutputs()
    {
        if (!_sessionFileExportEnabled || !trackRoomRawCoverage || string.IsNullOrEmpty(_roomRawCoverageDirectory))
            return;

        float voxelSize = Mathf.Max(0.005f, roomRawCoverageVoxelSizeMeters);
        Dictionary<Vector3Int, RoomRawCoverageVoxel> memoryVoxels = _roomRawCoverageHistoryVoxels.Count > 0
            ? _roomRawCoverageHistoryVoxels
            : _roomRawCoverageVoxels;
        int memoryStableVoxels = CountStableRoomRawCoverageVoxels(memoryVoxels);
        StringBuilder voxels = new StringBuilder(Mathf.Max(1024, memoryVoxels.Count * 128));
        voxels.AppendLine("# ScanCover room raw coverage voxels");
        voxels.Append("voxelSizeMeters=").Append(FormatFloat(voxelSize)).AppendLine();
        voxels.Append("frames=").Append(_roomRawCoverageFrames).AppendLine();
        voxels.Append("coveredVoxels=").Append(memoryVoxels.Count).AppendLine();
        voxels.Append("stableVoxels=").Append(memoryStableVoxels).AppendLine();
        voxels.Append("sessionNewCoveredVoxels=").Append(_roomRawCoverageCoveredVoxels).AppendLine();
        voxels.Append("sessionNewStableVoxels=").Append(_roomRawCoverageStableVoxels).AppendLine();
        voxels.Append("highVoxels=").Append(_roomRawCoverageHighVoxels).AppendLine();
        voxels.Append("highStableVoxels=").Append(_roomRawCoverageHighStableVoxels).AppendLine();
        voxels.Append("lowVoxels=").Append(_roomRawCoverageLowVoxels).AppendLine();
        voxels.Append("lowStableVoxels=").Append(_roomRawCoverageLowStableVoxels).AppendLine();
        voxels.Append("riskVoxels=").Append(_roomRawCoverageRiskVoxels).AppendLine();
        voxels.AppendLine("voxelX,voxelY,voxelZ,frameHits,pointHits,stable,risk,high,highStable,low,lowStable,avgX,avgY,avgZ,avgNormalX,avgNormalY,avgNormalZ");
        foreach (KeyValuePair<Vector3Int, RoomRawCoverageVoxel> pair in memoryVoxels)
        {
            RoomRawCoverageVoxel voxel = pair.Value;
            Vector3 avg = voxel.pointHits > 0 ? voxel.positionSum / voxel.pointHits : Vector3.zero;
            Vector3 normal = voxel.normalSum.sqrMagnitude > 1e-8f ? voxel.normalSum.normalized : Vector3.up;
            voxels.Append(pair.Key.x).Append(',')
                .Append(pair.Key.y).Append(',')
                .Append(pair.Key.z).Append(',')
                .Append(voxel.frameHits).Append(',')
                .Append(voxel.pointHits).Append(',')
                .Append(voxel.stable ? 1 : 0).Append(',')
                .Append(voxel.risk ? 1 : 0).Append(',')
                .Append(voxel.high ? 1 : 0).Append(',')
                .Append(voxel.highStable ? 1 : 0).Append(',')
                .Append(voxel.low ? 1 : 0).Append(',')
                .Append(voxel.lowStable ? 1 : 0).Append(',')
                .Append(FormatVector(avg)).Append(',')
                .Append(FormatVector(normal)).AppendLine();
        }
        File.WriteAllText(_roomRawCoverageVoxelsPath, voxels.ToString(), Encoding.UTF8);

        StringBuilder summary = new StringBuilder(512);
        summary.AppendLine("{");
        AppendJsonString(summary, "status", "complete", 1, true);
        AppendJsonNumber(summary, "frames", _roomRawCoverageFrames, 1, true);
        AppendJsonNumber(summary, "rejectedFrames", _roomRawCoverageRejectedFrames, 1, true);
        AppendJsonNumber(summary, "totalSamples", _roomRawCoverageTotalSamples, 1, true);
        AppendJsonNumber(summary, "validSamples", _roomRawCoverageValidSamples, 1, true);
        AppendJsonNumber(summary, "coveredVoxels", _roomRawCoverageCoveredVoxels, 1, true);
        AppendJsonNumber(summary, "sessionNewCoveredVoxels", _roomRawCoverageCoveredVoxels, 1, true);
        AppendJsonNumber(summary, "sessionNewStableVoxels", _roomRawCoverageStableVoxels, 1, true);
        AppendJsonNumber(summary, "targetCoveredVoxels", Mathf.Max(1, roomRawCoverageTargetVoxels), 1, true);
        AppendJsonNumber(summary, "historyCoveredVoxels", _roomRawCoverageHistoryVoxels.Count, 1, true);
        AppendJsonNumber(summary, "historyStableVoxels", CountStableRoomRawCoverageVoxels(_roomRawCoverageHistoryVoxels), 1, true);
        AppendJsonNumber(summary, "loadedPreviousCoverageSessions", _loadedRoomRawCoverageSessions, 1, true);
        AppendJsonNumber(summary, "loadedPreviousCoverageVoxels", _loadedRoomRawCoverageVoxels, 1, true);
        AppendJsonNumber(summary, "loadedPreviousCoverageStableVoxels", _loadedRoomRawCoverageStableVoxels, 1, true);
        AppendJsonString(summary, "loadedPreviousCoverageSource", _loadedRoomRawCoverageSource, 1, true);
        AppendJsonBool(summary, "lockedToStartView", ShouldUseRoomRawCoverageHudOnlyCapture() && lockRoomRawCoverageToStartView, 1, true);
        AppendJsonNumber(summary, "localTargetFrames", Mathf.Max(1, roomRawCoverageLocalTargetFrames), 1, true);
        AppendJsonNumber(summary, "localTargetValidSamples", Mathf.Max(1, roomRawCoverageLocalTargetValidSamples), 1, true);
        AppendJsonNumber(summary, "localTargetCoveredVoxels", Mathf.Max(1, roomRawCoverageLocalTargetVoxels), 1, true);
        AppendJsonNumber(summary, "localTargetStableVoxels", Mathf.Max(1, roomRawCoverageLocalTargetStableVoxels), 1, true);
        AppendJsonNumber(summary, "stableVoxels", _roomRawCoverageStableVoxels, 1, true);
        AppendJsonNumber(summary, "targetStableVoxels", Mathf.Max(1, roomRawCoverageTargetStableVoxels), 1, true);
        AppendJsonNumber(summary, "highVoxels", _roomRawCoverageHighVoxels, 1, true);
        AppendJsonNumber(summary, "targetHighVoxels", Mathf.Max(1, roomRawCoverageTargetHighVoxels), 1, true);
        AppendJsonNumber(summary, "highStableVoxels", _roomRawCoverageHighStableVoxels, 1, true);
        AppendJsonNumber(summary, "targetHighStableVoxels", Mathf.Max(1, roomRawCoverageTargetHighStableVoxels), 1, true);
        AppendJsonNumber(summary, "lowVoxels", _roomRawCoverageLowVoxels, 1, true);
        AppendJsonNumber(summary, "targetLowVoxels", Mathf.Max(1, roomRawCoverageTargetLowVoxels), 1, true);
        AppendJsonNumber(summary, "lowStableVoxels", _roomRawCoverageLowStableVoxels, 1, true);
        AppendJsonNumber(summary, "targetLowStableVoxels", Mathf.Max(1, roomRawCoverageTargetLowStableVoxels), 1, true);
        AppendJsonNumber(summary, "riskVoxels", _roomRawCoverageRiskVoxels, 1, true);
        AppendJsonNumber(summary, "posePositionCells", _roomRawCoveragePosePositionCells.Count, 1, true);
        AppendJsonNumber(summary, "poseOrientationCells", _roomRawCoveragePoseOrientationCells.Count, 1, true);
        AppendJsonNumber(summary, "poseCells", _roomRawCoveragePoseCellCount, 1, true);
        AppendJsonNumber(summary, "targetPoseCells", Mathf.Max(1, roomRawCoverageTargetPoseCells), 1, false);
        summary.AppendLine("}");
        File.WriteAllText(_roomRawCoverageSummaryPath, summary.ToString(), Encoding.UTF8);
    }

    private void WriteRoomRawDepthSummary()
    {
        if (!_sessionFileExportEnabled || !ShouldExportRoomRawDepthFrames() || string.IsNullOrEmpty(_roomRawDepthSummaryPath))
            return;

        StringBuilder summary = new StringBuilder(512);
        summary.AppendLine("{");
        AppendJsonString(summary, "status", "complete", 1, true);
        AppendJsonBool(summary, "onlyAcceptedFrames", exportRoomRawDepthOnlyAcceptedFrames, 1, true);
        AppendJsonNumber(summary, "frameStride", Mathf.Max(1, roomRawDepthFrameStride), 1, true);
        AppendJsonBool(summary, "focusOnly", roomRawDepthFocusOnly, 1, true);
        AppendJsonNumber(summary, "maxSamplesPerFrame", Mathf.Max(0, roomRawDepthMaxSamplesPerFrame), 1, true);
        AppendJsonNumber(summary, "exportedFrames", _roomRawDepthExportedFrames, 1, true);
        AppendJsonNumber(summary, "skippedFrames", _roomRawDepthSkippedFrames, 1, true);
        AppendJsonNumber(summary, "exportedSamples", _roomRawDepthExportedSamples, 1, true);
        AppendJsonString(summary, "directory", _roomRawDepthDirectory ?? "", 1, false);
        summary.AppendLine("}");
        File.WriteAllText(_roomRawDepthSummaryPath, summary.ToString(), Encoding.UTF8);
    }

    private static Vector3Int WorldToVoxelKey(Vector3 point, float voxelSize)
    {
        float inv = 1f / Mathf.Max(0.005f, voxelSize);
        return new Vector3Int(
            Mathf.FloorToInt(point.x * inv),
            Mathf.FloorToInt(point.y * inv),
            Mathf.FloorToInt(point.z * inv));
    }

    private void UpdateStabilityFromVerticesCsv(string verticesCsvPath, int frameIndex)
    {
        if (string.IsNullOrEmpty(verticesCsvPath) || !File.Exists(verticesCsvPath))
            return;

        float voxelSize = Mathf.Max(0.005f, stabilityVoxelSizeMeters);
        HashSet<Vector3Int> touchedThisFrame = new HashSet<Vector3Int>();
        foreach (string line in File.ReadLines(verticesCsvPath))
        {
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#' || line.IndexOf('=') >= 0 || line.StartsWith("index,", StringComparison.Ordinal))
                continue;
            string[] parts = line.Split(',');
            if (parts.Length < 7 ||
                !float.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out float x) ||
                !float.TryParse(parts[5], NumberStyles.Float, CultureInfo.InvariantCulture, out float y) ||
                !float.TryParse(parts[6], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
                continue;

            Vector3 point = new Vector3(x, y, z);
            Vector3Int key = new Vector3Int(
                Mathf.FloorToInt(point.x / voxelSize),
                Mathf.FloorToInt(point.y / voxelSize),
                Mathf.FloorToInt(point.z / voxelSize));

            _stabilityVoxels.TryGetValue(key, out StabilityVoxel voxel);
            if (voxel.pointHits == 0)
            {
                voxel.firstFrame = frameIndex;
                voxel.lastFrame = frameIndex;
                voxel.min = point;
                voxel.max = point;
            }
            voxel.pointHits++;
            voxel.positionSum += point;
            voxel.min = Vector3.Min(voxel.min, point);
            voxel.max = Vector3.Max(voxel.max, point);
            if (touchedThisFrame.Add(key))
            {
                voxel.frameHits++;
                voxel.lastFrame = frameIndex;
            }
            _stabilityVoxels[key] = voxel;
        }
    }

    private void WriteStabilityOutputs()
    {
        if (!_sessionFileExportEnabled || string.IsNullOrEmpty(_sessionDirectory) || _stabilityVoxels.Count <= 0)
            return;

        string path = Path.Combine(_sessionDirectory, "multi_frame_stability_voxels.csv");
        float voxelSize = Mathf.Max(0.005f, stabilityVoxelSizeMeters);
        StringBuilder builder = new StringBuilder(Mathf.Max(1024, _stabilityVoxels.Count * 120));
        builder.AppendLine("# ScanCover multi-frame stability voxels");
        builder.Append("voxelSizeMeters=").Append(FormatFloat(voxelSize)).AppendLine();
        builder.Append("capturedFrames=").Append(_capturedFrameCount).AppendLine();
        builder.AppendLine("voxelX,voxelY,voxelZ,frameHits,pointHits,firstFrame,lastFrame,avgX,avgY,avgZ,minX,minY,minZ,maxX,maxY,maxZ");
        foreach (KeyValuePair<Vector3Int, StabilityVoxel> pair in _stabilityVoxels)
        {
            StabilityVoxel voxel = pair.Value;
            Vector3 avg = voxel.pointHits > 0 ? voxel.positionSum / voxel.pointHits : Vector3.zero;
            builder.Append(pair.Key.x).Append(',')
                .Append(pair.Key.y).Append(',')
                .Append(pair.Key.z).Append(',')
                .Append(voxel.frameHits).Append(',')
                .Append(voxel.pointHits).Append(',')
                .Append(voxel.firstFrame).Append(',')
                .Append(voxel.lastFrame).Append(',')
                .Append(FormatVector(avg)).Append(',')
                .Append(FormatVector(voxel.min)).Append(',')
                .Append(FormatVector(voxel.max)).AppendLine();
        }
        File.WriteAllText(path, builder.ToString(), Encoding.UTF8);
    }

    private void ReleaseRawDepthResources()
    {
        if (_rawDepthProbeBuffer == null)
            return;
        _rawDepthProbeBuffer.Release();
        _rawDepthProbeBuffer = null;
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

    private static string FormatVector(Vector3 value)
        => string.Concat(FormatFloat(value.x), ",", FormatFloat(value.y), ",", FormatFloat(value.z));

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);

    private static string FormatFloat(float value)
        => value.ToString("F6", CultureInfo.InvariantCulture);

    private static string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "";
        bool needsQuotes = value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0;
        if (!needsQuotes)
            return value;
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string SafeDirectoryName(string value, string fallback)
    {
        string safe = string.IsNullOrWhiteSpace(value) ? fallback : value.Trim();
        foreach (char invalid in Path.GetInvalidFileNameChars())
            safe = safe.Replace(invalid, '_');
        return string.IsNullOrWhiteSpace(safe) ? fallback : safe;
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

    private static void AppendJsonNumber(StringBuilder builder, string key, int value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": ").Append(value.ToString(CultureInfo.InvariantCulture));
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

    private static void AppendJsonBool(StringBuilder builder, string key, bool value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": ").Append(value ? "true" : "false");
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonNumber(StringBuilder builder, string key, double value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": ").Append(value.ToString("R", CultureInfo.InvariantCulture));
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonVector(StringBuilder builder, string key, Vector3 value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": [")
            .Append(FormatFloat(value.x)).Append(", ")
            .Append(FormatFloat(value.y)).Append(", ")
            .Append(FormatFloat(value.z)).Append(']');
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonQuaternion(StringBuilder builder, string key, Quaternion value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": [")
            .Append(FormatFloat(value.x)).Append(", ")
            .Append(FormatFloat(value.y)).Append(", ")
            .Append(FormatFloat(value.z)).Append(", ")
            .Append(FormatFloat(value.w)).Append(']');
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonMatrix(StringBuilder builder, string key, Matrix4x4 value, int indent, bool trailingComma)
    {
        AppendIndent(builder, indent);
        builder.Append('"').Append(EscapeJson(key)).Append("\": [");
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                if (row != 0 || column != 0)
                    builder.Append(", ");
                builder.Append(FormatFloat(value[row, column]));
            }
        }
        builder.Append(']');
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendEyeSegmentJson(StringBuilder builder, string eye, int start, int count, int width, int indent, bool trailingComma)
    {
        int safeWidth = Mathf.Max(1, width);
        int height = count > 0 ? Mathf.Max(1, Mathf.CeilToInt(count / (float)safeWidth)) : 0;
        AppendIndent(builder, indent);
        builder.AppendLine("{");
        AppendJsonString(builder, "eye", eye, indent + 1, true);
        AppendJsonNumber(builder, "startIndex", Mathf.Max(0, start), indent + 1, true);
        AppendJsonNumber(builder, "count", Mathf.Max(0, count), indent + 1, true);
        AppendJsonNumber(builder, "width", safeWidth, indent + 1, true);
        AppendJsonNumber(builder, "height", height, indent + 1, false);
        AppendIndent(builder, indent);
        builder.Append('}');
        if (trailingComma)
            builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendIndent(StringBuilder builder, int indent)
    {
        for (int i = 0; i < indent; i++)
            builder.Append("  ");
    }
}






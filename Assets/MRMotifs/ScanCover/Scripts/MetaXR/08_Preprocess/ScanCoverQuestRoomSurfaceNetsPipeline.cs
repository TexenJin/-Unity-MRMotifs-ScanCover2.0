using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

/// <summary>
/// Independent QuestRoomScan-style production candidate:
/// Meta depth -> projective TSDF -> GPU Surface Nets -> persistent temporal wire mesh.
/// It deliberately does not read the legacy ScanCover TSDF, DMC, QEF or certificate state.
/// Architecture adapted from QuestRoomScan (MIT); see ThirdPartyNotices/QuestRoomScan-MIT.txt.
/// </summary>
[DefaultExecutionOrder(-25)]
[DisallowMultipleComponent]
public sealed class ScanCoverQuestRoomSurfaceNetsPipeline : MonoBehaviour
{
    [Serializable]
    public sealed class EyeInputDiagnostics
    {
        public int attempts;
        public int integrations;
        public int acceptedFrame = -1;
        public bool bound;
        public bool active;
        public bool ready;
        public bool expectedEye;
        public bool outputsAvailable;
        public bool reprojectionAvailable;
        public bool eyePoseAvailable;
        public bool integrated;
        public bool compatibilityIntegrated;
        public int sourceEye = -1;
        public int width;
        public int height;
        public int dispatchFrame = -1;
        public float dispatchDeltaMilliseconds = -1f;
        public string issue = "not_attempted";

        public void BeginAttempt(int frameOrdinal)
        {
            attempts++;
            acceptedFrame = frameOrdinal;
            bound = false;
            active = false;
            ready = false;
            expectedEye = false;
            outputsAvailable = false;
            reprojectionAvailable = false;
            eyePoseAvailable = false;
            integrated = false;
            compatibilityIntegrated = false;
            sourceEye = -1;
            width = 0;
            height = 0;
            dispatchFrame = -1;
            dispatchDeltaMilliseconds = -1f;
            issue = "unresolved";
        }

        public void ResetAll()
        {
            attempts = 0;
            integrations = 0;
            acceptedFrame = -1;
            bound = false;
            active = false;
            ready = false;
            expectedEye = false;
            outputsAvailable = false;
            reprojectionAvailable = false;
            eyePoseAvailable = false;
            integrated = false;
            compatibilityIntegrated = false;
            sourceEye = -1;
            width = 0;
            height = 0;
            dispatchFrame = -1;
            dispatchDeltaMilliseconds = -1f;
            issue = "not_attempted";
        }
    }

    [Header("Production candidate")]
    [SerializeField] private bool renderVisible = true;
    [SerializeField] private Color wireColor = new Color(1f, 0.62f, 0.02f, 0.96f);

    [Header("Single-resolution volume")]
    [SerializeField, Range(64, 192)] private int volumeSizeX = 144;
    [SerializeField, Range(48, 128)] private int volumeSizeY = 80;
    [SerializeField, Range(64, 192)] private int volumeSizeZ = 144;
    [SerializeField, Range(0.03f, 0.10f)] private float voxelSizeMeters = 0.05f;
    [SerializeField, Range(0.05f, 0.30f)] private float truncationDistanceMeters = 0.15f;

    [Header("Projective integration")]
    [SerializeField, Range(0f, 1f)] private float minimumObservationConfidence = 0.12f;
    [SerializeField, Range(0f, 1f)] private float minimumNormalFacingDot = 0.20f;
    [SerializeField, Min(0.05f)] private float minimumDepthMeters = 0.30f;
    [SerializeField, Min(0.5f)] private float maximumDepthMeters = 5.5f;
    [SerializeField, Range(0.1f, 2f)] private float blendRate = 0.8f;
    [SerializeField, Range(0.5f, 10f)] private float stability = 2.5f;
    [SerializeField, Range(0.005f, 0.1f)] private float weightGrowth = 0.025f;
    [SerializeField, Range(0.1f, 1f)] private float maximumWeight = 0.5f;

    [Header("Surface Nets")]
    [SerializeField, Range(0.01f, 0.5f)] private float minimumMeshWeight = 0.08f;
    [SerializeField, Range(0.03f, 0.25f)] private float vertexBudgetFraction = 0.12f;
    [SerializeField, Range(0, 3)] private int smoothIterations = 1;
    [SerializeField, Range(0f, 1f)] private float smoothLambda = 0.33f;
    [SerializeField, Range(0f, 1f)] private float smoothBeta = 0.50f;
    [SerializeField, Range(0.1f, 1f)] private float temporalAlphaMaximum = 0.85f;
    [SerializeField, Range(0.01f, 0.5f)] private float temporalAlphaMinimum = 0.10f;
    [SerializeField, Range(0.01f, 1f)] private float temporalDecayRate = 0.15f;
    [SerializeField, Range(0.001f, 0.03f)] private float convergenceThresholdMeters = 0.005f;
    [SerializeField, Range(0.0001f, 0.01f)] private float temporalDeadzoneMeters = 0.001f;

    [Header("Diagnostics")]
    [SerializeField] private bool debugLog = true;
    [SerializeField] private bool renderEvidenceDiagnostics = true;
    [SerializeField] private bool enableReadOnlyLifecycleDiagnostics = true;
    [SerializeField, Range(0.25f, 1.5f)]
    private float directSurfaceBandInVoxels = 0.75f;
    [SerializeField, Range(4, 240)]
    private int staleDirectSupportIntegrationThreshold = 48;
    [SerializeField, Range(1, 48)]
    private int freeSpaceContradictionIntegrationThreshold = 4;
    [SerializeField] private bool enableReadOnlyHoldAndLegalityShadows = true;
    [Tooltip("仅把多视角自由空间一致判定为非法的固定体素从前台隔离；不删除 TSDF 或历史证据。")]
    [SerializeField] private bool enableConservativeProductionLegalityGate = true;
    [SerializeField, Range(1, 4)]
    private int productionLegalityEvaluationIntervalIntegrations = 2;
    [SerializeField, Range(2, 32)]
    private int shadowLegalityEvaluationIntervalIntegrations = 8;
    [SerializeField, Range(4096, 131072)]
    private int shadowMaximumEvaluatedVertices = 65536;
    [SerializeField, Range(1, 6)]
    private int shadowDuplicateSearchVoxels = 4;
    [SerializeField, Range(0.25f, 1.5f)]
    private float shadowDepthToleranceInVoxels = 0.75f;
    [SerializeField, Range(0.75f, 0.999f)]
    private float shadowParallelNormalDot = 0.94f;
    [SerializeField, Range(0.25f, 2f)]
    private float shadowDuplicateMinimumSeparationInVoxels = 0.60f;
    [SerializeField, Range(1f, 8f)]
    private float shadowDuplicateMaximumSeparationInVoxels = 4.5f;
    [SerializeField, Range(0.25f, 3f)]
    private float shadowDuplicateMaximumTangentialDistanceInVoxels = 1.5f;
    [SerializeField, Range(4, 128)]
    private int shadowRecentCurrentDepthSupportIntegrations = 24;
    [SerializeField, Range(2, 8)]
    private int shadowRayFreeMinimumConfirmations = 2;
    [SerializeField, Range(3, 16)]
    private int shadowRayFreeSingleViewConfirmations = 4;
    [SerializeField, Range(2, 12)]
    private int shadowRayBehindMinimumConfirmations = 3;
    [SerializeField, Range(0.005f, 0.10f)]
    private float shadowViewBaselineNearMeters = 0.02f;
    [SerializeField, Range(0.02f, 0.20f)]
    private float shadowViewBaselineStrongMeters = 0.05f;
    [SerializeField, Range(0.05f, 0.40f)]
    private float shadowViewBaselineWideMeters = 0.10f;
    [SerializeField, Range(0.25f, 5f)]
    private float shadowViewParallaxNearDegrees = 1f;
    [SerializeField, Range(1f, 10f)]
    private float shadowViewParallaxStrongDegrees = 3f;
    [SerializeField, Range(2f, 20f)]
    private float shadowViewParallaxWideDegrees = 5f;
    [SerializeField, Range(1, 30)] private int counterReadbackIntervalIntegrations = 2;
    [SerializeField, Range(5f, 250f)]
    private float maximumAcceptedStereoDispatchDeltaMilliseconds = 75f;

    [Header("QuestRoomScan compatibility shadow (read-only)")]
    [Tooltip("Independent TSDF/topology eligibility chain. It never writes production volume or rendering buffers.")]
    [SerializeField] private bool enableQuestRoomCompatibilityShadow = true;
    [SerializeField, Range(1, 10)] private int compatibilityDilationSteps = 8;
    [SerializeField, Range(0.05f, 1f)]
    private float compatibilityDepthDisparityThresholdMeters = 0.50f;
    [SerializeField, Min(0.1f)] private float compatibilityMinimumDepthMeters = 0.50f;
    [SerializeField, Min(0.5f)] private float compatibilityMaximumDepthMeters = 5.0f;
    [SerializeField, Range(1, 30)] private int compatibilityReadbackIntervalIntegrations = 2;

    public bool HasRenderableSurface { get; private set; }
    public int IntegrationCount { get; private set; }
    public uint LastVertexCount { get; private set; }
    public uint LastLineIndexCount { get; private set; }
    public uint LastOverflowCount { get; private set; }
    public uint LastDiagnosticDirectVertexCount { get; private set; }
    public uint LastDiagnosticBackCapVertexCount { get; private set; }
    public uint LastDiagnosticFreeContradictedVertexCount { get; private set; }
    public uint LastDiagnosticUnknownVertexCount { get; private set; }
    public uint LastDiagnosticRearVertexCount { get; private set; }
    public uint LastDiagnosticStaleVertexCount { get; private set; }
    public uint LastDiagnosticUnresolvedVertexCount { get; private set; }
    public uint LastDiagnosticQuadCount { get; private set; }
    public uint LastDiagnosticDirectQuadCount { get; private set; }
    public uint LastDiagnosticBackCapQuadCount { get; private set; }
    public uint LastDiagnosticFreeContradictedQuadCount { get; private set; }
    public uint LastDiagnosticUnknownQuadCount { get; private set; }
    public uint LastDiagnosticRearQuadCount { get; private set; }
    public uint LastDiagnosticStaleQuadCount { get; private set; }
    public uint LastDiagnosticUnresolvedQuadCount { get; private set; }
    public uint LastDiagnosticSuppressedLineSegmentCount { get; private set; }
    public uint LastLifecycleActiveCellCount { get; private set; }
    public uint LastLifecycleVisibleCellCount { get; private set; }
    public uint LastLifecycleHiddenCellCount { get; private set; }
    public uint LastLifecycleVisibleToHiddenCount { get; private set; }
    public uint LastLifecycleHiddenToVisibleCount { get; private set; }
    public uint LastLifecycleFreshDirectToStaleCount { get; private set; }
    public uint LastLifecycleActiveToInactiveCount { get; private set; }
    public uint LastLifecycleInactiveToActiveCount { get; private set; }
    public uint LastLifecycleClassChangedCount { get; private set; }
    public uint LastLifecycleDirectAge0To7Count { get; private set; }
    public uint LastLifecycleDirectAge8To15Count { get; private set; }
    public uint LastLifecycleDirectAge16To31Count { get; private set; }
    public uint LastLifecycleDirectAge32To47Count { get; private set; }
    public uint LastLifecycleDirectAge48PlusCount { get; private set; }
    public uint LastTemporalFirstSeenCount { get; private set; }
    public uint LastTemporalDeadzoneHeldCount { get; private set; }
    public uint LastTemporalSmallBlendCount { get; private set; }
    public uint LastTemporalLargeResetCount { get; private set; }
    public uint LastLifecycleHiddenDirectBearingCount { get; private set; }
    public uint LastLifecycleYellowUnknownCount { get; private set; }
    public uint LastLifecycleGreenDirectCount { get; private set; }
    public uint LastLifecycleVisibleUnresolvedCount { get; private set; }
    public uint LastStaleDirectCorners1Count { get; private set; }
    public uint LastStaleDirectCorners2To3Count { get; private set; }
    public uint LastStaleDirectCorners4PlusCount { get; private set; }
    public uint LastStaleDirectWrites1To3Count { get; private set; }
    public uint LastStaleDirectWrites4To15Count { get; private set; }
    public uint LastStaleDirectWrites16To63Count { get; private set; }
    public uint LastStaleDirectWrites64PlusCount { get; private set; }
    public uint LastStaleDirectCrossingNoDirectEndpointCount { get; private set; }
    public uint LastStaleDirectCrossingOneSidedOnlyCount { get; private set; }
    public uint LastStaleDirectCrossingAnyTwoSidedCount { get; private set; }
    public uint LastStaleDirectRecentFreeBelowGateCount { get; private set; }
    public uint LastStaleDirectNewerFreeBelowGateCount { get; private set; }
    public uint LastStaleDirectRearMixedCount { get; private set; }
    public uint LastStaleDirectNoConflictHintCount { get; private set; }
    public uint LastStaleDirectStrongCandidateCount { get; private set; }
    public uint LastStaleDirectWeakCandidateCount { get; private set; }
    public uint LastShadowMatureActiveCount { get; private set; }
    public uint LastShadowMatureInactiveCount { get; private set; }
    public uint LastShadowHoldAge0To8Count { get; private set; }
    public uint LastShadowHoldAge9To24Count { get; private set; }
    public uint LastShadowHoldAge25To64Count { get; private set; }
    public uint LastShadowHoldExpired65PlusCount { get; private set; }
    public uint LastShadowHoldContradictedCount { get; private set; }
    public uint LastShadowRecovered0To8Count { get; private set; }
    public uint LastShadowRecovered9To24Count { get; private set; }
    public uint LastShadowRecovered25To64Count { get; private set; }
    public uint LastShadowRecovered65PlusCount { get; private set; }
    public uint LastShadowInactiveCrossingBelow3Count { get; private set; }
    public uint LastShadowInactiveNoObservedCrossingCount { get; private set; }
    public uint LastShadowInactiveTemporalPositionAvailableCount { get; private set; }
    public uint LastShadowDepthEvaluatedCount { get; private set; }
    public uint LastShadowDepthAlignedCount { get; private set; }
    public uint LastShadowDepthFreeContradictedCount { get; private set; }
    public uint LastShadowDepthBehindCurrentCount { get; private set; }
    public uint LastShadowDepthUnobservedCount { get; private set; }
    public uint LastShadowDuplicatePairCount { get; private set; }
    public uint LastShadowDuplicateFrontAlignedCount { get; private set; }
    public uint LastShadowDuplicateBackAlignedCount { get; private set; }
    public uint LastShadowDuplicateBackFreeContradictedCount { get; private set; }
    public uint LastShadowDuplicateBackBehindCurrentCount { get; private set; }
    public uint LastShadowDuplicateBackUnobservedCount { get; private set; }
    public uint LastShadowDuplicateRevokeCandidateCount { get; private set; }
    public uint LastShadowDuplicateAmbiguousCount { get; private set; }
    public uint LastShadowDuplicateBackRecentlySupportedCount { get; private set; }
    public uint LastShadowDuplicateBackSupportStaleCount { get; private set; }
    public uint LastShadowEverRecoveredCount { get; private set; }
    public uint LastShadowEverContradictedCount { get; private set; }
    public uint LastShadowEverExpiredCount { get; private set; }
    public uint LastShadowEverDuplicateCandidateCount { get; private set; }
    public uint LastShadowEverRevokeCandidateCount { get; private set; }
    public uint LastShadowEverAmbiguousCount { get; private set; }
    public uint LastShadowRayTrackedCount { get; private set; }
    public uint LastShadowRayEverAlignedCount { get; private set; }
    public uint LastShadowRayEverFreeCount { get; private set; }
    public uint LastShadowRayEverBehindCount { get; private set; }
    public uint LastShadowRayMultiView2PlusCount { get; private set; }
    public uint LastShadowRayMultiView3PlusCount { get; private set; }
    public uint LastShadowRayFreeCandidateCount { get; private set; }
    public uint LastShadowRayBehindOnlySuspectCount { get; private set; }
    public uint LastShadowRayAmbiguousCount { get; private set; }
    public uint LastShadowRayNeverAlignedCount { get; private set; }
    public uint LastShadowRayRecentlySupportedCount { get; private set; }
    public uint LastShadowRayFreeAlignedMixedCount { get; private set; }
    public uint LastShadowRayBehindAlignedMixedCount { get; private set; }
    public uint LastShadowRayDiversityFlagTotalCount { get; private set; }
    public uint LastShadowRayBaselineNearCount { get; private set; }
    public uint LastShadowRayBaselineStrongCount { get; private set; }
    public uint LastShadowRayBaselineWideCount { get; private set; }
    public uint LastShadowRayParallaxNearCount { get; private set; }
    public uint LastShadowRayParallaxStrongCount { get; private set; }
    public uint LastShadowRayParallaxWideCount { get; private set; }
    public uint LastShadowRayDiverseCount { get; private set; }
    public uint LastShadowRayStrongDiverseCount { get; private set; }
    public uint LastShadowRaySingleViewFreeRepeatCount { get; private set; }
    public uint LastShadowRayOriginClampedCount { get; private set; }
    public uint LastProductionLegalitySuppressedCount { get; private set; }
    public uint LastProductionDuplicateSuppressedPairCount { get; private set; }
    public uint LastProductionDuplicateAwaitingRepeatCount { get; private set; }
    public uint LastProductionDuplicateRejectedRayHistoryCount { get; private set; }
    public uint LastProductionLegalityRecoveredAlignedCount { get; private set; }
    public uint LastShadowRayRevisited2PlusCount { get; private set; }
    public uint LastShadowRayRevisited4PlusCount { get; private set; }
    public uint LastShadowRayRevisitedWithoutBaselineCount { get; private set; }
    public int ShadowEvaluationPoseCount { get; private set; }
    public float ShadowEvaluationPoseMaxBaselineMeters { get; private set; }
    public float ShadowEvaluationPoseAccumulatedTravelMeters { get; private set; }
    public float ShadowEvaluationPoseMaxStepMeters { get; private set; }
    public bool ConservativeProductionLegalityGateEnabled =>
        enableConservativeProductionLegalityGate;
    public int ShadowRayFreeMinimumConfirmations =>
        Mathf.Clamp(shadowRayFreeMinimumConfirmations, 2, 8);
    public int ShadowRayFreeSingleViewConfirmations =>
        Mathf.Clamp(shadowRayFreeSingleViewConfirmations, 3, 16);
    public int ShadowRayBehindMinimumConfirmations =>
        Mathf.Clamp(shadowRayBehindMinimumConfirmations, 2, 12);
    public float ShadowViewBaselineNearMeters => shadowViewBaselineNearMeters;
    public float ShadowViewBaselineStrongMeters => shadowViewBaselineStrongMeters;
    public float ShadowViewBaselineWideMeters => shadowViewBaselineWideMeters;
    public float ShadowViewParallaxNearDegrees => shadowViewParallaxNearDegrees;
    public float ShadowViewParallaxStrongDegrees => shadowViewParallaxStrongDegrees;
    public float ShadowViewParallaxWideDegrees => shadowViewParallaxWideDegrees;
    public int LastShadowLegalityEvaluationIntegration { get; private set; } = -1;
    public float LastShadowDiagnosticCpuEnqueueMilliseconds { get; private set; }
    public string LastIssue { get; private set; }
    public int AcceptedFrameAttemptCount { get; private set; }
    public int StereoPairIntegrationCount { get; private set; }
    public int PartialStereoIntegrationCount { get; private set; }
    public int RejectedAcceptedFrameCount { get; private set; }
    public int StereoCompanionPreparationAttempts { get; private set; }
    public int StereoCompanionPreparationSuccesses { get; private set; }
    public int LastAcceptedSnapshotFrame { get; private set; } = -1;
    public EyeInputDiagnostics LeftEyeDiagnostics => _leftEyeDiagnostics;
    public EyeInputDiagnostics RightEyeDiagnostics => _rightEyeDiagnostics;
    public bool QuestRoomCompatibilityShadowEnabled => enableQuestRoomCompatibilityShadow;
    public uint LastCompatibilityProjectedVoxelCount { get; private set; }
    public uint LastCompatibilitySeedWriteCount { get; private set; }
    public uint LastCompatibilityUpdateWriteCount { get; private set; }
    public uint LastCompatibilityOcclusionRejectCount { get; private set; }
    public uint LastCompatibilitySeedQualityRejectCount { get; private set; }
    public uint LastCompatibilityActiveCellCount { get; private set; }
    public uint LastCompatibilityAllUnknownCrossingRejectCount { get; private set; }
    public uint LastCompatibilityCrossingCellCount { get; private set; }
    public uint LastCompatibilityLowBlendRejectCount { get; private set; }
    public uint LastCompatibilityMissingDilationCount { get; private set; }
    public uint LastCompatibilityProductionActiveCellCount { get; private set; }
    public uint LastCompatibilitySharedActiveCellCount { get; private set; }
    public uint LastCompatibilityProductionOnlyActiveCellCount { get; private set; }
    public uint LastCompatibilityShadowOnlyActiveCellCount { get; private set; }
    public uint LastCompatibilityProductionOnlyAlignedCount { get; private set; }
    public uint LastCompatibilityProductionOnlyFreeContradictedCount { get; private set; }
    public uint LastCompatibilityProductionOnlyBehindCurrentCount { get; private set; }
    public uint LastCompatibilityProductionOnlyUnobservedCount { get; private set; }
    public uint LastCompatibilityProductionOnlyDirectCount { get; private set; }
    public uint LastCompatibilityProductionOnlyStaleCount { get; private set; }
    public uint LastCompatibilityProductionOnlyRearCount { get; private set; }
    public uint LastCompatibilityProductionOnlyNoEvidenceCount { get; private set; }
    public uint LastCompatibilityProductionOnlyAlignedDirectCount { get; private set; }
    public uint LastCompatibilityProductionOnlyAlignedStaleCount { get; private set; }
    public uint LastCompatibilityProductionOnlyAlignedRearCount { get; private set; }
    public uint LastCompatibilityProductionOnlyFreeDirectCount { get; private set; }
    public uint LastCompatibilityProductionOnlyFreeStaleCount { get; private set; }
    public uint LastCompatibilityProductionOnlyFreeRearCount { get; private set; }
    public uint LastCompatibilityProductionOnlyBehindDirectCount { get; private set; }
    public uint LastCompatibilityProductionOnlyBehindStaleCount { get; private set; }
    public uint LastCompatibilityProductionOnlyBehindRearCount { get; private set; }
    public uint LastCompatibilityProductionOnlyUnobservedDirectCount { get; private set; }
    public uint LastCompatibilityProductionOnlyUnobservedStaleCount { get; private set; }
    public uint LastCompatibilityProductionOnlyUnobservedRearCount { get; private set; }
    public uint LastCompatibilityShadowOnlyAlignedCount { get; private set; }
    public uint LastCompatibilityShadowOnlyFreeContradictedCount { get; private set; }
    public uint LastCompatibilityShadowOnlyBehindCurrentCount { get; private set; }
    public uint LastCompatibilityShadowOnlyUnobservedCount { get; private set; }

    private const int VertexStride = 32;
    private const int CounterCount = 124;
    private const int CompatibilityCounterCount = 40;
    private ComputeShader _tsdfCompute;
    private ComputeShader _surfaceCompute;
    private Material _wireMaterial;
    private RenderTexture _tsdfVolume;
    private RenderTexture _evidenceVolume;
    private RenderTexture _temporalState;
    private RenderTexture _compatibilityTsdfVolume;
    private RenderTexture _compatibilityDilationA;
    private RenderTexture _compatibilityDilationB;
    private GraphicsBuffer _coordVertexMap;
    private GraphicsBuffer _vertices;
    private GraphicsBuffer _lineIndices;
    private GraphicsBuffer _counters;
    private GraphicsBuffer _dispatchArgs;
    private GraphicsBuffer _drawArgs;
    private GraphicsBuffer _smoothPositionA;
    private GraphicsBuffer _smoothPositionB;
    private GraphicsBuffer _diagnosticPreviousCellState;
    private GraphicsBuffer _diagnosticShadowCellState;
    private GraphicsBuffer _diagnosticShadowRayState;
    private GraphicsBuffer _diagnosticShadowViewOriginState;
    private GraphicsBuffer _compatibilityCounters;
    private int _maximumVertices;
    private int _maximumLineIndices;
    private int _totalVoxelCount;
    private int _clearTsdfKernel = -1;
    private int _integrateTsdfKernel = -1;
    private int _clearSurfaceKernel = -1;
    private int _classifyKernel = -1;
    private int _buildDispatchKernel = -1;
    private int _initSmoothKernel = -1;
    private int _smoothKernel = -1;
    private int _applySmoothKernel = -1;
    private int _temporalKernel = -1;
    private int _generateLinesKernel = -1;
    private int _buildDrawArgsKernel = -1;
    private int _initTemporalKernel = -1;
    private int _evaluateShadowDepthKernel = -1;
    private int _evaluateShadowDuplicateKernel = -1;
    private int _clearCompatibilityVolumeKernel = -1;
    private int _clearCompatibilityCountersKernel = -1;
    private int _initCompatibilityDilationKernel = -1;
    private int _compatibilityDilationStepKernel = -1;
    private int _integrateCompatibilityKernel = -1;
    private int _countCompatibilitySurfaceCellsKernel = -1;
    private Matrix4x4 _volumeLocalToWorld = Matrix4x4.identity;
    private Bounds _worldBounds;
    private bool _volumePlaced;
    private bool _resourcesReady;
    private bool _counterReadbackPending;
    private bool _compatibilityCounterReadbackPending;
    private int _lastAcceptedFrameOrdinal = -1;
    private int _evidenceIntegrationOrdinal;
    private ScanCoverTsdfSingleShellPrototype _owner;
    private MaterialPropertyBlock _propertyBlock;
    private ScanCoverDepthPreprocessor _acceptedLeftSource;
    private ScanCoverDepthPreprocessor _acceptedRightSource;
    private RenderTexture _shadowWorldPosition;
    private RenderTexture _shadowObservationMeta;
    private Matrix4x4 _shadowWorldToClip = Matrix4x4.identity;
    private Vector3 _shadowEyeWorld;
    private Vector2Int _shadowSourceSize;
    private bool _shadowCurrentInputValid;
    private bool _hasShadowEvaluationPose;
    private Vector3 _firstShadowEvaluationEyeWorld;
    private Vector3 _lastShadowEvaluationEyeWorld;
    private readonly EyeInputDiagnostics _leftEyeDiagnostics = new EyeInputDiagnostics();
    private readonly EyeInputDiagnostics _rightEyeDiagnostics = new EyeInputDiagnostics();

    private static readonly int VolumeId = Shader.PropertyToID("_Volume");
    private static readonly int EvidenceVolumeId = Shader.PropertyToID("_EvidenceVolume");
    private static readonly int WorldPositionId = Shader.PropertyToID("_WorldPosition");
    private static readonly int WorldNormalId = Shader.PropertyToID("_WorldNormal");
    private static readonly int ObservationMetaId = Shader.PropertyToID("_ObservationMeta");
    private static readonly int CompatibilityVolumeId =
        Shader.PropertyToID("_CompatibilityVolume");
    private static readonly int CompatibilityDilationSourceId =
        Shader.PropertyToID("_CompatibilityDilationSource");
    private static readonly int CompatibilityDilationDestinationId =
        Shader.PropertyToID("_CompatibilityDilationDestination");
    private static readonly int CompatibilityCountersId =
        Shader.PropertyToID("_CompatibilityCounters");
    private static readonly int TsdfVolumeId = Shader.PropertyToID("_TsdfVolume");
    private static readonly int VoxelCountId = Shader.PropertyToID("_VoxelCount");
    private static readonly int VoxelSizeId = Shader.PropertyToID("_VoxelSize");
    private static readonly int TotalVoxelCountId = Shader.PropertyToID("_TotalVoxelCount");
    private static readonly int MaximumVerticesId = Shader.PropertyToID("_MaximumVertices");
    private static readonly int MaximumLineIndicesId = Shader.PropertyToID("_MaximumLineIndices");
    private static readonly int MinimumWeightId = Shader.PropertyToID("_MinimumWeight");
    private static readonly int CoordVertexMapId = Shader.PropertyToID("_CoordVertexMap");
    private static readonly int VerticesId = Shader.PropertyToID("_Vertices");
    private static readonly int LineIndicesId = Shader.PropertyToID("_LineIndices");
    private static readonly int CountersId = Shader.PropertyToID("_Counters");
    private static readonly int DispatchArgsId = Shader.PropertyToID("_DispatchArgs");
    private static readonly int DrawArgsId = Shader.PropertyToID("_DrawArgs");
    private static readonly int SmoothPositionAId = Shader.PropertyToID("_SmoothPositionA");
    private static readonly int SmoothPositionBId = Shader.PropertyToID("_SmoothPositionB");
    private static readonly int TemporalStateId = Shader.PropertyToID("_TemporalState");
    private static readonly int DiagnosticPreviousCellStateId =
        Shader.PropertyToID("_DiagnosticPreviousCellState");
    private static readonly int DiagnosticShadowCellStateId =
        Shader.PropertyToID("_DiagnosticShadowCellState");
    private static readonly int DiagnosticShadowRayStateId =
        Shader.PropertyToID("_DiagnosticShadowRayState");
    private static readonly int DiagnosticShadowViewOriginStateId =
        Shader.PropertyToID("_DiagnosticShadowViewOriginState");
    private static readonly int ShadowWorldPositionId =
        Shader.PropertyToID("_ShadowWorldPosition");
    private static readonly int ShadowObservationMetaId =
        Shader.PropertyToID("_ShadowObservationMeta");
    private static readonly int SurfaceVerticesId = Shader.PropertyToID("_SurfaceVertices");
    private static readonly int SurfaceLineIndicesId = Shader.PropertyToID("_SurfaceLineIndices");
    private static readonly int VolumeLocalToWorldId = Shader.PropertyToID("_VolumeLocalToWorld");
    private static readonly int WireColorId = Shader.PropertyToID("_WireColor");
    private static readonly int EvidenceDiagnosticModeId =
        Shader.PropertyToID("_EvidenceDiagnosticMode");

    public void BindOwner(ScanCoverTsdfSingleShellPrototype owner)
    {
        _owner = owner;
    }

    public void BindAcceptedStereoSources(
        ScanCoverDepthPreprocessor leftSource,
        ScanCoverDepthPreprocessor rightSource)
    {
        _acceptedLeftSource = leftSource;
        _acceptedRightSource = rightSource;
    }

    /// <summary>
    /// The production raw-depth source refreshes the directly referenced eye first.
    /// This method stages only the explicitly bound companion eye from the same source
    /// frame; it never searches the scene or refreshes arbitrary preprocessors.
    /// </summary>
    public bool PrepareAcceptedStereoCompanionFrame()
    {
        StereoCompanionPreparationAttempts++;
        if (_acceptedLeftSource == null)
            return false;

        bool refreshed = _acceptedLeftSource.RefreshNow();
        if (refreshed)
            StereoCompanionPreparationSuccesses++;
        return refreshed;
    }

    public void SetRenderVisible(bool visible)
    {
        renderVisible = visible;
    }

    public void OnAcceptedDepthFrame(
        int frameOrdinal,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot acceptedSnapshot)
    {
        if (!isActiveAndEnabled || frameOrdinal == _lastAcceptedFrameOrdinal)
            return;
        _lastAcceptedFrameOrdinal = frameOrdinal;
        AcceptedFrameAttemptCount++;
        LastAcceptedSnapshotFrame = acceptedSnapshot != null
            ? acceptedSnapshot.frameIndex
            : frameOrdinal;

        if (!EnsureResources())
            return;
        if (!_volumePlaced)
            PlaceVolumeAroundViewer();

        if (enableQuestRoomCompatibilityShadow)
            ClearCompatibilityCounters();

        int integratedEyes = 0;
        if (TryIntegrateBoundEye(
                _acceptedLeftSource,
                ScanCoverDepthPreprocessor.SourceEye.Left,
                acceptedSnapshot,
                _leftEyeDiagnostics))
            integratedEyes++;
        if (TryIntegrateBoundEye(
                _acceptedRightSource,
                ScanCoverDepthPreprocessor.SourceEye.Right,
                acceptedSnapshot,
                _rightEyeDiagnostics))
            integratedEyes++;

        if (integratedEyes == 0)
        {
            RejectedAcceptedFrameCount++;
            LastIssue =
                $"accepted_stereo_unavailable:left={_leftEyeDiagnostics.issue};" +
                $"right={_rightEyeDiagnostics.issue}";
            return;
        }

        if (integratedEyes == 2)
            StereoPairIntegrationCount++;
        else
            PartialStereoIntegrationCount++;

        IntegrationCount++;
        if (enableQuestRoomCompatibilityShadow)
            RebuildCompatibilityTopologyEligibility();
        ExtractSurface();
        LastIssue = integratedEyes == 2
            ? null
            : $"accepted_stereo_partial:left={_leftEyeDiagnostics.issue};" +
              $"right={_rightEyeDiagnostics.issue}";
        if (debugLog && (IntegrationCount <= 2 || IntegrationCount % 12 == 0))
        {
            Debug.Log(
                $"[ScanCoverQuestRoomSurfaceNets] integrate={IntegrationCount} acceptedFrame={LastAcceptedSnapshotFrame} " +
                $"eyes={integratedEyes} left={_leftEyeDiagnostics.issue} right={_rightEyeDiagnostics.issue} " +
                $"verts={LastVertexCount} lineIndices={LastLineIndexCount} overflow={LastOverflowCount}",
                this);
        }
    }

    private bool TryIntegrateBoundEye(
        ScanCoverDepthPreprocessor source,
        ScanCoverDepthPreprocessor.SourceEye expectedEye,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot acceptedSnapshot,
        EyeInputDiagnostics diagnostics)
    {
        int acceptedFrame = acceptedSnapshot != null
            ? acceptedSnapshot.frameIndex
            : _lastAcceptedFrameOrdinal;
        diagnostics.BeginAttempt(acceptedFrame);
        diagnostics.bound = source != null;
        if (source == null)
        {
            diagnostics.issue = "source_not_bound";
            return false;
        }

        diagnostics.active = source.isActiveAndEnabled;
        diagnostics.ready = source.IsReady;
        diagnostics.sourceEye = (int)source.CurrentSourceEye;
        diagnostics.expectedEye = source.CurrentSourceEye == expectedEye;
        diagnostics.dispatchFrame = source.LastSuccessfulRefreshFrame;
        Vector2Int outputResolution = source.OutputResolution;
        diagnostics.width = outputResolution.x;
        diagnostics.height = outputResolution.y;

        if (!diagnostics.expectedEye)
        {
            diagnostics.issue = "wrong_eye_binding";
            return false;
        }
        if (!diagnostics.ready)
        {
            diagnostics.issue = string.IsNullOrEmpty(source.LastIssue)
                ? "source_not_ready"
                : "source_not_ready:" + source.LastIssue;
            return false;
        }
        if (!source.TryGetOutputs(
                out RenderTexture worldPosition,
                out RenderTexture worldNormal,
                out RenderTexture observationMeta))
        {
            diagnostics.issue = "outputs_unavailable";
            return false;
        }

        diagnostics.outputsAvailable = true;
        diagnostics.width = worldPosition != null ? worldPosition.width : diagnostics.width;
        diagnostics.height = worldPosition != null ? worldPosition.height : diagnostics.height;
        if (worldPosition == null || worldNormal == null || observationMeta == null ||
            diagnostics.width <= 1 || diagnostics.height <= 1)
        {
            diagnostics.issue = "invalid_output_size";
            return false;
        }

        bool isAcceptedSnapshotEye = acceptedSnapshot != null &&
                                     acceptedSnapshot.sourceEyeIndex == (int)expectedEye;
        bool hasReprojection = isAcceptedSnapshotEye && acceptedSnapshot.hasDepthReprojectionMatrix;
        Matrix4x4 reprojection = hasReprojection
            ? acceptedSnapshot.depthReprojectionMatrix
            : source.LastDispatchDepthReprojectionMatrix;
        if (!hasReprojection)
            hasReprojection = source.HasLastDispatchDepthReprojectionMatrix;
        diagnostics.reprojectionAvailable = hasReprojection;
        if (!hasReprojection)
        {
            diagnostics.issue = "reprojection_unavailable";
            return false;
        }

        bool hasEyePose = isAcceptedSnapshotEye && acceptedSnapshot.hasDispatchEyePosition;
        Vector3 eyeWorld = hasEyePose
            ? acceptedSnapshot.dispatchEyePosition
            : source.LastDispatchEyePosition;
        if (!hasEyePose)
            hasEyePose = source.HasLastDispatchEyePosition;
        diagnostics.eyePoseAvailable = hasEyePose;
        if (!hasEyePose)
        {
            diagnostics.issue = "eye_pose_unavailable";
            return false;
        }

        if (acceptedSnapshot != null &&
            acceptedSnapshot.dispatchRealtimeSeconds > 0d &&
            source.LastSuccessfulRefreshRealtimeSeconds > 0d)
        {
            diagnostics.dispatchDeltaMilliseconds = (float)Math.Abs(
                (source.LastSuccessfulRefreshRealtimeSeconds -
                 acceptedSnapshot.dispatchRealtimeSeconds) * 1000d);
            if (diagnostics.dispatchDeltaMilliseconds >
                maximumAcceptedStereoDispatchDeltaMilliseconds)
            {
                diagnostics.issue = "stale_dispatch";
                return false;
            }
        }

        DispatchProjectiveIntegration(
            worldPosition,
            worldNormal,
            observationMeta,
            reprojection,
            eyeWorld);
        if (enableQuestRoomCompatibilityShadow &&
            source.HasLastDispatchDepthReprojectionMatrix &&
            source.HasLastDispatchEyePosition)
        {
            // The compatibility chain deliberately uses the transform captured
            // by the same preprocessor dispatch that produced these textures.
            // Production remains untouched so this is a true A/B comparison.
            DispatchQuestRoomCompatibilityIntegration(
                worldPosition,
                worldNormal,
                observationMeta,
                source.LastDispatchDepthReprojectionMatrix,
                source.LastDispatchEyePosition);
            diagnostics.compatibilityIntegrated = true;
        }
        diagnostics.integrated = true;
        diagnostics.integrations++;
        diagnostics.issue = "integrated";
        return true;
    }

    public void ClearAll()
    {
        if (!EnsureResources()) return;
        BindTsdfVolumes();
        DispatchVolume(_tsdfCompute, _clearTsdfKernel);
        if (_compatibilityTsdfVolume != null)
        {
            _tsdfCompute.SetTexture(
                _clearCompatibilityVolumeKernel,
                CompatibilityVolumeId,
                _compatibilityTsdfVolume);
            DispatchVolume(_tsdfCompute, _clearCompatibilityVolumeKernel);
            ClearCompatibilityCounters();
        }
        DispatchVolume(_surfaceCompute, _initTemporalKernel);
        HasRenderableSurface = false;
        IntegrationCount = 0;
        LastVertexCount = 0;
        LastLineIndexCount = 0;
        LastOverflowCount = 0;
        ResetEvidenceDiagnosticCounts();
        _evidenceIntegrationOrdinal = 0;
        _lastAcceptedFrameOrdinal = -1;
        AcceptedFrameAttemptCount = 0;
        StereoPairIntegrationCount = 0;
        PartialStereoIntegrationCount = 0;
        RejectedAcceptedFrameCount = 0;
        StereoCompanionPreparationAttempts = 0;
        StereoCompanionPreparationSuccesses = 0;
        LastAcceptedSnapshotFrame = -1;
        LastShadowLegalityEvaluationIntegration = -1;
        LastShadowDiagnosticCpuEnqueueMilliseconds = 0f;
        ShadowEvaluationPoseCount = 0;
        ShadowEvaluationPoseMaxBaselineMeters = 0f;
        ShadowEvaluationPoseAccumulatedTravelMeters = 0f;
        ShadowEvaluationPoseMaxStepMeters = 0f;
        _hasShadowEvaluationPose = false;
        _shadowCurrentInputValid = false;
        _shadowWorldPosition = null;
        _shadowObservationMeta = null;
        _leftEyeDiagnostics.ResetAll();
        _rightEyeDiagnostics.ResetAll();
        ResetCompatibilityDiagnosticCounts();
    }

    private void LateUpdate()
    {
        if (!renderVisible || !HasRenderableSurface || !_resourcesReady ||
            _wireMaterial == null || _vertices == null || _lineIndices == null || _drawArgs == null)
            return;

        _propertyBlock.SetBuffer(SurfaceVerticesId, _vertices);
        _propertyBlock.SetBuffer(SurfaceLineIndicesId, _lineIndices);
        _propertyBlock.SetMatrix(VolumeLocalToWorldId, _volumeLocalToWorld);
        _propertyBlock.SetColor(WireColorId, wireColor);
        _propertyBlock.SetFloat(
            EvidenceDiagnosticModeId,
            renderEvidenceDiagnostics ? 1f : 0f);
        RenderParams renderParams = new RenderParams(_wireMaterial)
        {
            worldBounds = _worldBounds,
            matProps = _propertyBlock,
            shadowCastingMode = ShadowCastingMode.Off,
            receiveShadows = false,
            layer = gameObject.layer
        };
        Graphics.RenderPrimitivesIndirect(
            renderParams,
            MeshTopology.Lines,
            _drawArgs,
            1);
    }

    private bool EnsureResources()
    {
        if (_resourcesReady) return true;
        _tsdfCompute = Resources.Load<ComputeShader>("ScanCoverQuestRoomTsdf");
        _surfaceCompute = Resources.Load<ComputeShader>("ScanCoverQuestRoomSurfaceNets");
        Shader wireShader = Resources.Load<Shader>("ScanCoverQuestRoomSurfaceNetsWire");
        if (wireShader == null)
            wireShader = Shader.Find("ScanCover/QuestRoomSurfaceNetsWire");
        if (_tsdfCompute == null || _surfaceCompute == null || wireShader == null)
        {
            LastIssue = "QuestRoom GPU shader 资源缺失";
            return false;
        }

        try
        {
            _clearTsdfKernel = _tsdfCompute.FindKernel("Clear");
            _integrateTsdfKernel = _tsdfCompute.FindKernel("Integrate");
            _clearCompatibilityVolumeKernel = _tsdfCompute.FindKernel(
                "ClearCompatibilityVolume");
            _clearCompatibilityCountersKernel = _tsdfCompute.FindKernel(
                "ClearCompatibilityCounters");
            _initCompatibilityDilationKernel = _tsdfCompute.FindKernel(
                "InitCompatibilityDilation");
            _compatibilityDilationStepKernel = _tsdfCompute.FindKernel(
                "CompatibilityDilationStep");
            _integrateCompatibilityKernel = _tsdfCompute.FindKernel(
                "IntegrateCompatibility");
            _countCompatibilitySurfaceCellsKernel = _tsdfCompute.FindKernel(
                "CountCompatibilitySurfaceCells");
            _clearSurfaceKernel = _surfaceCompute.FindKernel("ClearCountersAndMap");
            _classifyKernel = _surfaceCompute.FindKernel("ClassifyAndEmit");
            _buildDispatchKernel = _surfaceCompute.FindKernel("BuildDispatchArgs");
            _initSmoothKernel = _surfaceCompute.FindKernel("InitSmooth");
            _smoothKernel = _surfaceCompute.FindKernel("SmoothVertices");
            _applySmoothKernel = _surfaceCompute.FindKernel("ApplySmooth");
            _temporalKernel = _surfaceCompute.FindKernel("TemporalBlend");
            _generateLinesKernel = _surfaceCompute.FindKernel("GenerateLineIndices");
            _buildDrawArgsKernel = _surfaceCompute.FindKernel("BuildDrawArgs");
            _initTemporalKernel = _surfaceCompute.FindKernel("InitTemporal");
            _evaluateShadowDepthKernel = _surfaceCompute.FindKernel(
                "EvaluateShadowDepthConsistency");
            _evaluateShadowDuplicateKernel = _surfaceCompute.FindKernel(
                "EvaluateShadowDuplicateShells");
        }
        catch (Exception ex)
        {
            LastIssue = "QuestRoom GPU kernel 缺失: " + ex.Message;
            return false;
        }

        _totalVoxelCount = volumeSizeX * volumeSizeY * volumeSizeZ;
        _maximumVertices = Mathf.Max(4096, Mathf.CeilToInt(_totalVoxelCount * vertexBudgetFraction));
        // Each Surface Nets cell can own at most three quads. Two triangles per
        // quad and three line pairs per triangle gives a strict 36-index bound.
        _maximumLineIndices = _maximumVertices * 36;
        _tsdfVolume = CreateVolume(
            "ScanCover_QuestRoom_TSDF",
            GraphicsFormat.R16G16_SFloat);
        _evidenceVolume = CreateVolume(
            "ScanCover_QuestRoom_SurfaceEvidence",
            GraphicsFormat.R16G16B16A16_SFloat);
        _temporalState = CreateVolume(
            "ScanCover_QuestRoom_Temporal",
            GraphicsFormat.R16G16B16A16_SFloat);
        _compatibilityTsdfVolume = CreateVolume(
            "ScanCover_QuestRoom_Compatibility_TSDF",
            GraphicsFormat.R16G16_SFloat);

        _coordVertexMap = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _totalVoxelCount,
            sizeof(int));
        _vertices = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumVertices,
            VertexStride);
        _lineIndices = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumLineIndices,
            sizeof(uint));
        _counters = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            CounterCount,
            sizeof(uint));
        GraphicsBuffer.Target indirect =
            GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;
        _dispatchArgs = new GraphicsBuffer(indirect, 3, sizeof(uint));
        _drawArgs = new GraphicsBuffer(indirect, 5, sizeof(uint));
        _smoothPositionA = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumVertices,
            sizeof(float) * 3);
        _smoothPositionB = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _maximumVertices,
            sizeof(float) * 3);
        _diagnosticPreviousCellState = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _totalVoxelCount,
            sizeof(uint));
        _diagnosticShadowCellState = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _totalVoxelCount,
            sizeof(uint));
        _diagnosticShadowRayState = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _totalVoxelCount,
            sizeof(uint));
        _diagnosticShadowViewOriginState = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            _totalVoxelCount,
            sizeof(uint));
        _compatibilityCounters = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            CompatibilityCounterCount,
            sizeof(uint));
        _wireMaterial = new Material(wireShader)
        {
            name = "ScanCover_QuestRoom_SurfaceNets_Wire"
        };
        _propertyBlock = new MaterialPropertyBlock();
        BindSurfaceBuffers();
        SetSurfaceConstants();
        BindTsdfVolumes();
        DispatchVolume(_tsdfCompute, _clearTsdfKernel);
        _tsdfCompute.SetTexture(
            _clearCompatibilityVolumeKernel,
            CompatibilityVolumeId,
            _compatibilityTsdfVolume);
        DispatchVolume(_tsdfCompute, _clearCompatibilityVolumeKernel);
        ClearCompatibilityCounters();
        DispatchVolume(_surfaceCompute, _initTemporalKernel);
        _resourcesReady = true;

        if (debugLog)
        {
            long approximateBytes = (long)_totalVoxelCount * (4 + 8 + 8 + 4 + 4 + 4 + 4 + 4) +
                                    (long)_maximumVertices * VertexStride +
                                    (long)_maximumLineIndices * 4 +
                                    (long)_maximumVertices * sizeof(float) * 3 * 2;
            Debug.Log(
                $"[ScanCoverQuestRoomSurfaceNets] GPU ready volume={volumeSizeX}x{volumeSizeY}x{volumeSizeZ} " +
                $"voxel={voxelSizeMeters:F3}m budget~={approximateBytes / (1024 * 1024)}MB",
                this);
        }
        return true;
    }

    private void PlaceVolumeAroundViewer()
    {
        Vector3 viewer = Camera.main != null ? Camera.main.transform.position : transform.position;
        float halfHeight = volumeSizeY * voxelSizeMeters * 0.5f;
        float centerY = Mathf.Max(halfHeight - 0.15f, viewer.y + 0.15f);
        Vector3 center = new Vector3(viewer.x, centerY, viewer.z);
        _volumeLocalToWorld = Matrix4x4.TRS(center, Quaternion.identity, Vector3.one);
        Vector3 size = new Vector3(volumeSizeX, volumeSizeY, volumeSizeZ) * voxelSizeMeters;
        _worldBounds = new Bounds(center, size);
        _volumePlaced = true;
    }

    private void DispatchProjectiveIntegration(
        RenderTexture worldPosition,
        RenderTexture worldNormal,
        RenderTexture observationMeta,
        Matrix4x4 worldToClip,
        Vector3 eyeWorld)
    {
        _tsdfCompute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _tsdfCompute.SetFloat(VoxelSizeId, voxelSizeMeters);
        _tsdfCompute.SetFloat("_TruncationDistance", truncationDistanceMeters);
        _tsdfCompute.SetFloat("_MinimumDepth", minimumDepthMeters);
        _tsdfCompute.SetFloat("_MaximumDepth", maximumDepthMeters);
        _tsdfCompute.SetFloat("_MinimumConfidence", minimumObservationConfidence);
        _tsdfCompute.SetFloat("_MinimumNormalDot", minimumNormalFacingDot);
        _tsdfCompute.SetFloat("_BlendRate", blendRate);
        _tsdfCompute.SetFloat("_Stability", stability);
        _tsdfCompute.SetFloat("_WeightGrowth", weightGrowth);
        _tsdfCompute.SetFloat("_MaximumWeight", maximumWeight);
        _evidenceIntegrationOrdinal++;
        _tsdfCompute.SetFloat(
            "_EvidenceSurfaceBand",
            Mathf.Max(0.001f, voxelSizeMeters * directSurfaceBandInVoxels));
        _tsdfCompute.SetInt("_EvidenceIntegrationOrdinal", _evidenceIntegrationOrdinal);
        _tsdfCompute.SetVector("_SourceSize", new Vector4(worldPosition.width, worldPosition.height, 0f, 0f));
        _tsdfCompute.SetVector("_EyeWorldPosition", new Vector4(eyeWorld.x, eyeWorld.y, eyeWorld.z, 1f));
        _tsdfCompute.SetMatrix("_WorldToClip", worldToClip);
        _tsdfCompute.SetMatrix(VolumeLocalToWorldId, _volumeLocalToWorld);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, VolumeId, _tsdfVolume);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, EvidenceVolumeId, _evidenceVolume);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, WorldPositionId, worldPosition);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, WorldNormalId, worldNormal);
        _tsdfCompute.SetTexture(_integrateTsdfKernel, ObservationMetaId, observationMeta);
        DispatchVolume(_tsdfCompute, _integrateTsdfKernel);

        // Read-only shadow input: retain the exact accepted eye that just fed
        // production integration. The shadow kernels only sample these handles;
        // they never refresh a source or write production textures.
        _shadowWorldPosition = worldPosition;
        _shadowObservationMeta = observationMeta;
        _shadowWorldToClip = worldToClip;
        _shadowEyeWorld = eyeWorld;
        _shadowSourceSize = new Vector2Int(worldPosition.width, worldPosition.height);
        _shadowCurrentInputValid = true;
    }

    private void DispatchQuestRoomCompatibilityIntegration(
        RenderTexture worldPosition,
        RenderTexture worldNormal,
        RenderTexture observationMeta,
        Matrix4x4 worldToClip,
        Vector3 eyeWorld)
    {
        if (_compatibilityTsdfVolume == null || _compatibilityCounters == null ||
            worldPosition == null || worldNormal == null || observationMeta == null)
            return;
        if (!EnsureCompatibilityDilationTextures(
                worldPosition.width,
                worldPosition.height))
            return;

        SetCompatibilityCommonConstants(
            worldPosition,
            worldToClip,
            eyeWorld);
        _tsdfCompute.SetTexture(
            _initCompatibilityDilationKernel,
            WorldPositionId,
            worldPosition);
        _tsdfCompute.SetTexture(
            _initCompatibilityDilationKernel,
            ObservationMetaId,
            observationMeta);
        _tsdfCompute.SetTexture(
            _initCompatibilityDilationKernel,
            CompatibilityDilationDestinationId,
            _compatibilityDilationA);
        DispatchTexture2D(
            _tsdfCompute,
            _initCompatibilityDilationKernel,
            worldPosition.width,
            worldPosition.height);

        RenderTexture dilationSource = _compatibilityDilationA;
        RenderTexture dilationDestination = _compatibilityDilationB;
        int requestedSteps = Mathf.Clamp(compatibilityDilationSteps, 1, 10);
        // QuestRoomScan starts jump flooding at 2^N and dispatches exactly N
        // passes. Keep this independent of the input texture dimensions so the
        // shadow follows the reference policy rather than adapting its policy
        // to our capture resolution.
        int step = 1 << requestedSteps;
        int dispatchedSteps = 0;
        while (step >= 1 && dispatchedSteps < requestedSteps)
        {
            _tsdfCompute.SetInt("_CompatibilityDilationStep", step);
            _tsdfCompute.SetTexture(
                _compatibilityDilationStepKernel,
                CompatibilityDilationSourceId,
                dilationSource);
            _tsdfCompute.SetTexture(
                _compatibilityDilationStepKernel,
                CompatibilityDilationDestinationId,
                dilationDestination);
            DispatchTexture2D(
                _tsdfCompute,
                _compatibilityDilationStepKernel,
                worldPosition.width,
                worldPosition.height);
            RenderTexture swap = dilationSource;
            dilationSource = dilationDestination;
            dilationDestination = swap;
            step >>= 1;
            dispatchedSteps++;
        }

        SetCompatibilityCommonConstants(
            worldPosition,
            worldToClip,
            eyeWorld);
        _tsdfCompute.SetTexture(
            _integrateCompatibilityKernel,
            CompatibilityVolumeId,
            _compatibilityTsdfVolume);
        _tsdfCompute.SetTexture(
            _integrateCompatibilityKernel,
            WorldPositionId,
            worldPosition);
        _tsdfCompute.SetTexture(
            _integrateCompatibilityKernel,
            WorldNormalId,
            worldNormal);
        _tsdfCompute.SetTexture(
            _integrateCompatibilityKernel,
            ObservationMetaId,
            observationMeta);
        _tsdfCompute.SetTexture(
            _integrateCompatibilityKernel,
            CompatibilityDilationSourceId,
            dilationSource);
        _tsdfCompute.SetBuffer(
            _integrateCompatibilityKernel,
            CompatibilityCountersId,
            _compatibilityCounters);
        DispatchVolume(_tsdfCompute, _integrateCompatibilityKernel);
    }

    private void SetCompatibilityCommonConstants(
        RenderTexture worldPosition,
        Matrix4x4 worldToClip,
        Vector3 eyeWorld)
    {
        _tsdfCompute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _tsdfCompute.SetFloat(VoxelSizeId, voxelSizeMeters);
        _tsdfCompute.SetFloat("_TruncationDistance", truncationDistanceMeters);
        _tsdfCompute.SetFloat("_BlendRate", blendRate);
        _tsdfCompute.SetFloat("_Stability", stability);
        _tsdfCompute.SetFloat("_WeightGrowth", weightGrowth);
        _tsdfCompute.SetFloat("_MaximumWeight", maximumWeight);
        _tsdfCompute.SetFloat(
            "_CompatibilityDepthDisparityThreshold",
            compatibilityDepthDisparityThresholdMeters);
        _tsdfCompute.SetFloat(
            "_CompatibilityMinimumDepth",
            compatibilityMinimumDepthMeters);
        _tsdfCompute.SetFloat(
            "_CompatibilityMaximumDepth",
            compatibilityMaximumDepthMeters);
        _tsdfCompute.SetFloat(
            "_CompatibilityMinimumMeshWeight",
            minimumMeshWeight);
        _tsdfCompute.SetVector(
            "_SourceSize",
            new Vector4(worldPosition.width, worldPosition.height, 0f, 0f));
        _tsdfCompute.SetVector(
            "_EyeWorldPosition",
            new Vector4(eyeWorld.x, eyeWorld.y, eyeWorld.z, 1f));
        _tsdfCompute.SetMatrix("_WorldToClip", worldToClip);
        _tsdfCompute.SetMatrix(VolumeLocalToWorldId, _volumeLocalToWorld);
    }

    private void ClearCompatibilityCounters()
    {
        if (_compatibilityCounters == null || _clearCompatibilityCountersKernel < 0)
            return;
        _tsdfCompute.SetBuffer(
            _clearCompatibilityCountersKernel,
            CompatibilityCountersId,
            _compatibilityCounters);
        // ClearCompatibilityCounters uses [numthreads(16, 1, 1)].  Dispatch
        // enough groups for the complete 40-slot read-only ledger; one group
        // left slots 16..39 accumulating across integrations.
        int clearGroupCount = (CompatibilityCounterCount + 15) / 16;
        _tsdfCompute.Dispatch(
            _clearCompatibilityCountersKernel,
            clearGroupCount,
            1,
            1);
    }

    private void RebuildCompatibilityTopologyEligibility()
    {
        if (_compatibilityTsdfVolume == null || _compatibilityCounters == null)
            return;
        if (!_shadowCurrentInputValid || _shadowWorldPosition == null ||
            _shadowObservationMeta == null)
            return;
        SetCompatibilityCommonConstants(
            _shadowWorldPosition,
            _shadowWorldToClip,
            _shadowEyeWorld);
        _tsdfCompute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _tsdfCompute.SetFloat(
            "_CompatibilityMinimumMeshWeight",
            minimumMeshWeight);
        _tsdfCompute.SetFloat(
            "_CompatibilityDepthTolerance",
            Mathf.Max(0.001f, voxelSizeMeters * shadowDepthToleranceInVoxels));
        _tsdfCompute.SetFloat(
            "_CompatibilityStaleIntegrationThreshold",
            Mathf.Max(1, staleDirectSupportIntegrationThreshold));
        _tsdfCompute.SetTexture(
            _countCompatibilitySurfaceCellsKernel,
            CompatibilityVolumeId,
            _compatibilityTsdfVolume);
        _tsdfCompute.SetTexture(
            _countCompatibilitySurfaceCellsKernel,
            VolumeId,
            _tsdfVolume);
        _tsdfCompute.SetTexture(
            _countCompatibilitySurfaceCellsKernel,
            EvidenceVolumeId,
            _evidenceVolume);
        _tsdfCompute.SetTexture(
            _countCompatibilitySurfaceCellsKernel,
            WorldPositionId,
            _shadowWorldPosition);
        _tsdfCompute.SetTexture(
            _countCompatibilitySurfaceCellsKernel,
            ObservationMetaId,
            _shadowObservationMeta);
        _tsdfCompute.SetBuffer(
            _countCompatibilitySurfaceCellsKernel,
            CompatibilityCountersId,
            _compatibilityCounters);
        DispatchVolume(_tsdfCompute, _countCompatibilitySurfaceCellsKernel);

        int interval = Mathf.Max(1, compatibilityReadbackIntervalIntegrations);
        if (!_compatibilityCounterReadbackPending &&
            (IntegrationCount <= 1 || IntegrationCount % interval == 0))
        {
            _compatibilityCounterReadbackPending = true;
            AsyncGPUReadback.Request(
                _compatibilityCounters,
                OnCompatibilityCounterReadback);
        }
    }

    private void OnCompatibilityCounterReadback(
        AsyncGPUReadbackRequest request)
    {
        _compatibilityCounterReadbackPending = false;
        if (!this || request.hasError) return;
        var values = request.GetData<uint>();
        if (values.Length < 38) return;
        LastCompatibilityProjectedVoxelCount = values[0];
        LastCompatibilitySeedWriteCount = values[1];
        LastCompatibilityUpdateWriteCount = values[2];
        LastCompatibilityOcclusionRejectCount = values[3];
        LastCompatibilitySeedQualityRejectCount = values[4];
        LastCompatibilityActiveCellCount = values[5];
        LastCompatibilityAllUnknownCrossingRejectCount = values[6];
        LastCompatibilityCrossingCellCount = values[7];
        LastCompatibilityLowBlendRejectCount = values[8];
        LastCompatibilityMissingDilationCount = values[9];
        LastCompatibilityProductionActiveCellCount = values[10];
        LastCompatibilitySharedActiveCellCount = values[11];
        LastCompatibilityProductionOnlyActiveCellCount = values[12];
        LastCompatibilityShadowOnlyActiveCellCount = values[13];
        LastCompatibilityProductionOnlyAlignedCount = values[14];
        LastCompatibilityProductionOnlyFreeContradictedCount = values[15];
        LastCompatibilityProductionOnlyBehindCurrentCount = values[16];
        LastCompatibilityProductionOnlyUnobservedCount = values[17];
        LastCompatibilityProductionOnlyDirectCount = values[18];
        LastCompatibilityProductionOnlyStaleCount = values[19];
        LastCompatibilityProductionOnlyRearCount = values[20];
        LastCompatibilityProductionOnlyNoEvidenceCount = values[21];
        LastCompatibilityProductionOnlyAlignedDirectCount = values[22];
        LastCompatibilityProductionOnlyAlignedStaleCount = values[23];
        LastCompatibilityProductionOnlyAlignedRearCount = values[24];
        LastCompatibilityProductionOnlyFreeDirectCount = values[25];
        LastCompatibilityProductionOnlyFreeStaleCount = values[26];
        LastCompatibilityProductionOnlyFreeRearCount = values[27];
        LastCompatibilityProductionOnlyBehindDirectCount = values[28];
        LastCompatibilityProductionOnlyBehindStaleCount = values[29];
        LastCompatibilityProductionOnlyBehindRearCount = values[30];
        LastCompatibilityProductionOnlyUnobservedDirectCount = values[31];
        LastCompatibilityProductionOnlyUnobservedStaleCount = values[32];
        LastCompatibilityProductionOnlyUnobservedRearCount = values[33];
        LastCompatibilityShadowOnlyAlignedCount = values[34];
        LastCompatibilityShadowOnlyFreeContradictedCount = values[35];
        LastCompatibilityShadowOnlyBehindCurrentCount = values[36];
        LastCompatibilityShadowOnlyUnobservedCount = values[37];
    }

    private void ResetCompatibilityDiagnosticCounts()
    {
        LastCompatibilityProjectedVoxelCount = 0;
        LastCompatibilitySeedWriteCount = 0;
        LastCompatibilityUpdateWriteCount = 0;
        LastCompatibilityOcclusionRejectCount = 0;
        LastCompatibilitySeedQualityRejectCount = 0;
        LastCompatibilityActiveCellCount = 0;
        LastCompatibilityAllUnknownCrossingRejectCount = 0;
        LastCompatibilityCrossingCellCount = 0;
        LastCompatibilityLowBlendRejectCount = 0;
        LastCompatibilityMissingDilationCount = 0;
        LastCompatibilityProductionActiveCellCount = 0;
        LastCompatibilitySharedActiveCellCount = 0;
        LastCompatibilityProductionOnlyActiveCellCount = 0;
        LastCompatibilityShadowOnlyActiveCellCount = 0;
        LastCompatibilityProductionOnlyAlignedCount = 0;
        LastCompatibilityProductionOnlyFreeContradictedCount = 0;
        LastCompatibilityProductionOnlyBehindCurrentCount = 0;
        LastCompatibilityProductionOnlyUnobservedCount = 0;
        LastCompatibilityProductionOnlyDirectCount = 0;
        LastCompatibilityProductionOnlyStaleCount = 0;
        LastCompatibilityProductionOnlyRearCount = 0;
        LastCompatibilityProductionOnlyNoEvidenceCount = 0;
        LastCompatibilityProductionOnlyAlignedDirectCount = 0;
        LastCompatibilityProductionOnlyAlignedStaleCount = 0;
        LastCompatibilityProductionOnlyAlignedRearCount = 0;
        LastCompatibilityProductionOnlyFreeDirectCount = 0;
        LastCompatibilityProductionOnlyFreeStaleCount = 0;
        LastCompatibilityProductionOnlyFreeRearCount = 0;
        LastCompatibilityProductionOnlyBehindDirectCount = 0;
        LastCompatibilityProductionOnlyBehindStaleCount = 0;
        LastCompatibilityProductionOnlyBehindRearCount = 0;
        LastCompatibilityProductionOnlyUnobservedDirectCount = 0;
        LastCompatibilityProductionOnlyUnobservedStaleCount = 0;
        LastCompatibilityProductionOnlyUnobservedRearCount = 0;
        LastCompatibilityShadowOnlyAlignedCount = 0;
        LastCompatibilityShadowOnlyFreeContradictedCount = 0;
        LastCompatibilityShadowOnlyBehindCurrentCount = 0;
        LastCompatibilityShadowOnlyUnobservedCount = 0;
    }

    private void ExtractSurface()
    {
        SetSurfaceConstants();
        BindSurfaceBuffers();
        _surfaceCompute.SetTexture(_classifyKernel, TsdfVolumeId, _tsdfVolume);
        _surfaceCompute.SetTexture(_classifyKernel, EvidenceVolumeId, _evidenceVolume);
        _surfaceCompute.SetTexture(_generateLinesKernel, TsdfVolumeId, _tsdfVolume);

        int clearGroups = Mathf.CeilToInt(_totalVoxelCount / 64f);
        _surfaceCompute.Dispatch(_clearSurfaceKernel, clearGroups, 1, 1);
        DispatchVolume(_surfaceCompute, _classifyKernel);
        _surfaceCompute.Dispatch(_buildDispatchKernel, 1, 1, 1);

        int iterationCount = Mathf.Max(0, smoothIterations);
        if (iterationCount > 0)
        {
            _surfaceCompute.DispatchIndirect(_initSmoothKernel, _dispatchArgs);
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                GraphicsBuffer source = iteration % 2 == 0
                    ? _smoothPositionA
                    : _smoothPositionB;
                GraphicsBuffer destination = iteration % 2 == 0
                    ? _smoothPositionB
                    : _smoothPositionA;
                SetBuffer(_smoothKernel, SmoothPositionAId, source);
                SetBuffer(_smoothKernel, SmoothPositionBId, destination);
                _surfaceCompute.DispatchIndirect(_smoothKernel, _dispatchArgs);
            }

            GraphicsBuffer finalSmooth = iterationCount % 2 == 0
                ? _smoothPositionA
                : _smoothPositionB;
            SetBuffer(_applySmoothKernel, SmoothPositionAId, finalSmooth);
            _surfaceCompute.DispatchIndirect(_applySmoothKernel, _dispatchArgs);
        }

        _surfaceCompute.DispatchIndirect(_temporalKernel, _dispatchArgs);
        RunReadOnlyShadowLegalityDiagnostics();
        _surfaceCompute.DispatchIndirect(_generateLinesKernel, _dispatchArgs);
        _surfaceCompute.Dispatch(_buildDrawArgsKernel, 1, 1, 1);

        int interval = Mathf.Max(1, counterReadbackIntervalIntegrations);
        if (!_counterReadbackPending && (IntegrationCount <= 1 || IntegrationCount % interval == 0))
        {
            _counterReadbackPending = true;
            AsyncGPUReadback.Request(_counters, OnCounterReadback);
        }
    }

    private void RunReadOnlyShadowLegalityDiagnostics()
    {
        if ((!enableReadOnlyHoldAndLegalityShadows &&
             !enableConservativeProductionLegalityGate) ||
            !_shadowCurrentInputValid ||
            _shadowWorldPosition == null ||
            _shadowObservationMeta == null)
            return;

        int interval = Mathf.Max(2, shadowLegalityEvaluationIntervalIntegrations);
        if (enableConservativeProductionLegalityGate)
        {
            interval = Mathf.Min(
                interval,
                Mathf.Clamp(
                    productionLegalityEvaluationIntervalIntegrations,
                    1,
                    4));
        }
        if (IntegrationCount > 1 && IntegrationCount % interval != 0)
            return;

        RecordShadowEvaluationPose();
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        _surfaceCompute.SetTexture(
            _evaluateShadowDepthKernel,
            ShadowWorldPositionId,
            _shadowWorldPosition);
        _surfaceCompute.SetTexture(
            _evaluateShadowDepthKernel,
            ShadowObservationMetaId,
            _shadowObservationMeta);
        _surfaceCompute.SetTexture(
            _evaluateShadowDuplicateKernel,
            ShadowWorldPositionId,
            _shadowWorldPosition);
        _surfaceCompute.SetTexture(
            _evaluateShadowDuplicateKernel,
            ShadowObservationMetaId,
            _shadowObservationMeta);
        _surfaceCompute.DispatchIndirect(_evaluateShadowDepthKernel, _dispatchArgs);
        _surfaceCompute.DispatchIndirect(
            _evaluateShadowDuplicateKernel,
            _dispatchArgs);
        long finished = System.Diagnostics.Stopwatch.GetTimestamp();
        LastShadowDiagnosticCpuEnqueueMilliseconds =
            (float)((finished - started) * 1000.0 /
                    System.Diagnostics.Stopwatch.Frequency);
        LastShadowLegalityEvaluationIntegration = IntegrationCount;
    }

    private void OnCounterReadback(AsyncGPUReadbackRequest request)
    {
        _counterReadbackPending = false;
        if (!this || request.hasError) return;
        var values = request.GetData<uint>();
        if (values.Length < CounterCount) return;
        LastVertexCount = Math.Min(values[0], (uint)_maximumVertices);
        LastLineIndexCount = Math.Min(values[1], (uint)_maximumLineIndices);
        LastOverflowCount = values[2];
        LastDiagnosticDirectVertexCount = values[3];
        LastDiagnosticBackCapVertexCount = values[4];
        LastDiagnosticFreeContradictedVertexCount = values[5];
        LastDiagnosticUnknownVertexCount = values[6];
        LastDiagnosticRearVertexCount = values[7];
        LastDiagnosticStaleVertexCount = values[8];
        LastDiagnosticUnresolvedVertexCount = values[9];
        LastDiagnosticQuadCount = values[10];
        LastDiagnosticDirectQuadCount = values[11];
        LastDiagnosticBackCapQuadCount = values[12];
        LastDiagnosticFreeContradictedQuadCount = values[13];
        LastDiagnosticUnknownQuadCount = values[14];
        LastDiagnosticRearQuadCount = values[15];
        LastDiagnosticStaleQuadCount = values[16];
        LastDiagnosticUnresolvedQuadCount = values[17];
        LastDiagnosticSuppressedLineSegmentCount = values[18];
        LastLifecycleActiveCellCount = values[19];
        LastLifecycleVisibleCellCount = values[20];
        LastLifecycleHiddenCellCount = values[21];
        LastLifecycleVisibleToHiddenCount = values[22];
        LastLifecycleHiddenToVisibleCount = values[23];
        LastLifecycleFreshDirectToStaleCount = values[24];
        LastLifecycleActiveToInactiveCount = values[25];
        LastLifecycleInactiveToActiveCount = values[26];
        LastLifecycleClassChangedCount = values[27];
        LastLifecycleDirectAge0To7Count = values[28];
        LastLifecycleDirectAge8To15Count = values[29];
        LastLifecycleDirectAge16To31Count = values[30];
        LastLifecycleDirectAge32To47Count = values[31];
        LastLifecycleDirectAge48PlusCount = values[32];
        LastTemporalFirstSeenCount = values[33];
        LastTemporalDeadzoneHeldCount = values[34];
        LastTemporalSmallBlendCount = values[35];
        LastTemporalLargeResetCount = values[36];
        LastLifecycleHiddenDirectBearingCount = values[37];
        LastLifecycleYellowUnknownCount = values[38];
        LastLifecycleGreenDirectCount = values[39];
        LastLifecycleVisibleUnresolvedCount = values[40];
        LastStaleDirectCorners1Count = values[41];
        LastStaleDirectCorners2To3Count = values[42];
        LastStaleDirectCorners4PlusCount = values[43];
        LastStaleDirectWrites1To3Count = values[44];
        LastStaleDirectWrites4To15Count = values[45];
        LastStaleDirectWrites16To63Count = values[46];
        LastStaleDirectWrites64PlusCount = values[47];
        LastStaleDirectCrossingNoDirectEndpointCount = values[48];
        LastStaleDirectCrossingOneSidedOnlyCount = values[49];
        LastStaleDirectCrossingAnyTwoSidedCount = values[50];
        LastStaleDirectRecentFreeBelowGateCount = values[51];
        LastStaleDirectNewerFreeBelowGateCount = values[52];
        LastStaleDirectRearMixedCount = values[53];
        LastStaleDirectNoConflictHintCount = values[54];
        LastStaleDirectStrongCandidateCount = values[55];
        LastStaleDirectWeakCandidateCount = values[56];
        LastShadowMatureActiveCount = values[57];
        LastShadowMatureInactiveCount = values[58];
        LastShadowHoldAge0To8Count = values[59];
        LastShadowHoldAge9To24Count = values[60];
        LastShadowHoldAge25To64Count = values[61];
        LastShadowHoldExpired65PlusCount = values[62];
        LastShadowHoldContradictedCount = values[63];
        LastShadowRecovered0To8Count = values[64];
        LastShadowRecovered9To24Count = values[65];
        LastShadowRecovered25To64Count = values[66];
        LastShadowRecovered65PlusCount = values[67];
        LastShadowInactiveCrossingBelow3Count = values[68];
        LastShadowInactiveNoObservedCrossingCount = values[69];
        LastShadowInactiveTemporalPositionAvailableCount = values[70];
        if (values[71] > 0 || values[76] > 0 || values[92] > 0)
        {
            LastShadowDepthEvaluatedCount = values[71];
            LastShadowDepthAlignedCount = values[72];
            LastShadowDepthFreeContradictedCount = values[73];
            LastShadowDepthBehindCurrentCount = values[74];
            LastShadowDepthUnobservedCount = values[75];
            LastShadowDuplicatePairCount = values[76];
            LastShadowDuplicateFrontAlignedCount = values[77];
            LastShadowDuplicateBackAlignedCount = values[78];
            LastShadowDuplicateBackFreeContradictedCount = values[79];
            LastShadowDuplicateBackBehindCurrentCount = values[80];
            LastShadowDuplicateBackUnobservedCount = values[81];
            LastShadowDuplicateRevokeCandidateCount = values[82];
            LastShadowDuplicateAmbiguousCount = values[83];
            LastShadowDuplicateBackRecentlySupportedCount = values[84];
            LastShadowDuplicateBackSupportStaleCount = values[85];
            LastShadowRayTrackedCount = values[92];
            LastShadowRayEverAlignedCount = values[93];
            LastShadowRayEverFreeCount = values[94];
            LastShadowRayEverBehindCount = values[95];
            LastShadowRayMultiView2PlusCount = values[96];
            LastShadowRayMultiView3PlusCount = values[97];
            LastShadowRayFreeCandidateCount = values[98];
            LastShadowRayBehindOnlySuspectCount = values[99];
            LastShadowRayAmbiguousCount = values[100];
            LastShadowRayNeverAlignedCount = values[101];
            LastShadowRayRecentlySupportedCount = values[102];
            LastShadowRayFreeAlignedMixedCount = values[103];
            LastShadowRayBehindAlignedMixedCount = values[104];
            LastShadowRayDiversityFlagTotalCount = values[105];
            LastShadowRayBaselineNearCount = values[106];
            LastShadowRayBaselineStrongCount = values[107];
            LastShadowRayBaselineWideCount = values[108];
            LastShadowRayParallaxNearCount = values[109];
            LastShadowRayParallaxStrongCount = values[110];
            LastShadowRayParallaxWideCount = values[111];
            LastShadowRayDiverseCount = values[112];
            LastShadowRayStrongDiverseCount = values[113];
            LastShadowRaySingleViewFreeRepeatCount = values[114];
            LastShadowRayOriginClampedCount = values[115];
            LastProductionLegalitySuppressedCount = values[116];
            LastShadowRayRevisited2PlusCount = values[117];
            LastShadowRayRevisited4PlusCount = values[118];
            LastShadowRayRevisitedWithoutBaselineCount = values[119];
            LastProductionDuplicateSuppressedPairCount = values[120];
            LastProductionDuplicateAwaitingRepeatCount = values[121];
            LastProductionDuplicateRejectedRayHistoryCount = values[122];
            LastProductionLegalityRecoveredAlignedCount = values[123];
        }
        LastShadowEverRecoveredCount = values[86];
        LastShadowEverContradictedCount = values[87];
        LastShadowEverExpiredCount = values[88];
        LastShadowEverDuplicateCandidateCount = values[89];
        LastShadowEverRevokeCandidateCount = values[90];
        LastShadowEverAmbiguousCount = values[91];
        bool wasReady = HasRenderableSurface;
        HasRenderableSurface = StereoPairIntegrationCount > 0 &&
                               LastVertexCount > 0 &&
                               LastLineIndexCount >= 6;
        if (!wasReady && HasRenderableSurface)
            _owner?.NotifyQuestRoomSurfaceNetsReady();
        if (debugLog && IntegrationCount > 0 && IntegrationCount % 12 == 0)
        {
            Debug.Log(
                $"[ScanCoverQuestRoomSurfaceNetsEvidence] vertex direct/backcap/free/unknown/rear/stale/unresolved=" +
                $"{LastDiagnosticDirectVertexCount}/{LastDiagnosticBackCapVertexCount}/" +
                $"{LastDiagnosticFreeContradictedVertexCount}/" +
                $"{LastDiagnosticUnknownVertexCount}/{LastDiagnosticRearVertexCount}/" +
                $"{LastDiagnosticStaleVertexCount}/{LastDiagnosticUnresolvedVertexCount} " +
                $"quad total/direct/backcap/free/unknown/rear/stale/unresolved=" +
                $"{LastDiagnosticQuadCount}/{LastDiagnosticDirectQuadCount}/" +
                $"{LastDiagnosticBackCapQuadCount}/{LastDiagnosticFreeContradictedQuadCount}/" +
                $"{LastDiagnosticUnknownQuadCount}/" +
                $"{LastDiagnosticRearQuadCount}/{LastDiagnosticStaleQuadCount}/" +
                $"{LastDiagnosticUnresolvedQuadCount} " +
                $"displaySuppressedLines={LastDiagnosticSuppressedLineSegmentCount}",
                this);
            Debug.Log(
                $"[ScanCoverQuestRoomLifecycleReadOnly] active/visible/hidden=" +
                $"{LastLifecycleActiveCellCount}/{LastLifecycleVisibleCellCount}/{LastLifecycleHiddenCellCount} " +
                $"v2h/h2v/a2i/i2a/classChange=" +
                $"{LastLifecycleVisibleToHiddenCount}/{LastLifecycleHiddenToVisibleCount}/" +
                $"{LastLifecycleActiveToInactiveCount}/{LastLifecycleInactiveToActiveCount}/" +
                $"{LastLifecycleClassChangedCount} directFreshToStale=" +
                $"{LastLifecycleFreshDirectToStaleCount} temporal first/hold/small/reset=" +
                $"{LastTemporalFirstSeenCount}/{LastTemporalDeadzoneHeldCount}/" +
                $"{LastTemporalSmallBlendCount}/{LastTemporalLargeResetCount}",
                this);
            Debug.Log(
                $"[ScanCoverQuestRoomStaleDirectReadOnly] corners1/2to3/4plus=" +
                $"{LastStaleDirectCorners1Count}/{LastStaleDirectCorners2To3Count}/" +
                $"{LastStaleDirectCorners4PlusCount} writes1to3/4to15/16to63/64plus=" +
                $"{LastStaleDirectWrites1To3Count}/{LastStaleDirectWrites4To15Count}/" +
                $"{LastStaleDirectWrites16To63Count}/{LastStaleDirectWrites64PlusCount} " +
                $"crossingNone/oneSided/twoSided=" +
                $"{LastStaleDirectCrossingNoDirectEndpointCount}/" +
                $"{LastStaleDirectCrossingOneSidedOnlyCount}/" +
                $"{LastStaleDirectCrossingAnyTwoSidedCount} conflict recentFree/newerFree/rear/noHint=" +
                $"{LastStaleDirectRecentFreeBelowGateCount}/" +
                $"{LastStaleDirectNewerFreeBelowGateCount}/" +
                $"{LastStaleDirectRearMixedCount}/{LastStaleDirectNoConflictHintCount} " +
                $"candidate strong/weak={LastStaleDirectStrongCandidateCount}/" +
                $"{LastStaleDirectWeakCandidateCount}",
                this);
            Debug.Log(
                $"[ScanCoverQuestRoomHoldShadowReadOnly] mature active/inactive=" +
                $"{LastShadowMatureActiveCount}/{LastShadowMatureInactiveCount} " +
                $"wouldHold age0to8/9to24/25to64/expired/contradicted=" +
                $"{LastShadowHoldAge0To8Count}/{LastShadowHoldAge9To24Count}/" +
                $"{LastShadowHoldAge25To64Count}/{LastShadowHoldExpired65PlusCount}/" +
                $"{LastShadowHoldContradictedCount} recovered=" +
                $"{LastShadowRecovered0To8Count}/{LastShadowRecovered9To24Count}/" +
                $"{LastShadowRecovered25To64Count}/{LastShadowRecovered65PlusCount}",
                this);
            Debug.Log(
                $"[ScanCoverQuestRoomLegalityShadowReadOnly] eval/aligned/free/behind/unobserved=" +
                $"{LastShadowDepthEvaluatedCount}/{LastShadowDepthAlignedCount}/" +
                $"{LastShadowDepthFreeContradictedCount}/" +
                $"{LastShadowDepthBehindCurrentCount}/{LastShadowDepthUnobservedCount} " +
                $"duplicate pairs/revoke/ambiguous=" +
                $"{LastShadowDuplicatePairCount}/" +
                $"{LastShadowDuplicateRevokeCandidateCount}/" +
                $"{LastShadowDuplicateAmbiguousCount} enqueueMs=" +
                $"{LastShadowDiagnosticCpuEnqueueMilliseconds:F3}",
                this);
            Debug.Log(
                $"[ScanCoverQuestRoomRayLayerShadowReadOnly] tracked/aligned/free/behind=" +
                $"{LastShadowRayTrackedCount}/{LastShadowRayEverAlignedCount}/" +
                $"{LastShadowRayEverFreeCount}/{LastShadowRayEverBehindCount} " +
                $"multiView2/3={LastShadowRayMultiView2PlusCount}/" +
                $"{LastShadowRayMultiView3PlusCount} freeCandidate/behindOnly/ambiguous=" +
                $"{LastShadowRayFreeCandidateCount}/" +
                $"{LastShadowRayBehindOnlySuspectCount}/" +
                $"{LastShadowRayAmbiguousCount} baselineNear/strong/wide=" +
                $"{LastShadowRayBaselineNearCount}/{LastShadowRayBaselineStrongCount}/" +
                $"{LastShadowRayBaselineWideCount} parallaxNear/strong/wide=" +
                $"{LastShadowRayParallaxNearCount}/{LastShadowRayParallaxStrongCount}/" +
                $"{LastShadowRayParallaxWideCount} diverse/strong/singleFree/originClamp=" +
                $"{LastShadowRayDiverseCount}/{LastShadowRayStrongDiverseCount}/" +
                $"{LastShadowRaySingleViewFreeRepeatCount}/{LastShadowRayOriginClampedCount} " +
                $"gate/revisit2/4/noBase={LastProductionLegalitySuppressedCount}/" +
                $"{LastShadowRayRevisited2PlusCount}/" +
                $"{LastShadowRayRevisited4PlusCount}/" +
                $"{LastShadowRayRevisitedWithoutBaselineCount} " +
                $"dupGate/await/reject/recover={LastProductionDuplicateSuppressedPairCount}/" +
                $"{LastProductionDuplicateAwaitingRepeatCount}/" +
                $"{LastProductionDuplicateRejectedRayHistoryCount}/" +
                $"{LastProductionLegalityRecoveredAlignedCount} pose=" +
                $"{ShadowEvaluationPoseCount}/" +
                $"{ShadowEvaluationPoseMaxBaselineMeters:F3}/" +
                $"{ShadowEvaluationPoseAccumulatedTravelMeters:F3}",
                this);
        }
    }

    private void BindSurfaceBuffers()
    {
        SetBuffer(_clearSurfaceKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_clearSurfaceKernel, CountersId, _counters);
        SetBuffer(_classifyKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_classifyKernel, VerticesId, _vertices);
        SetBuffer(_classifyKernel, CountersId, _counters);
        SetBuffer(
            _classifyKernel,
            DiagnosticPreviousCellStateId,
            _diagnosticPreviousCellState);
        SetBuffer(
            _classifyKernel,
            DiagnosticShadowCellStateId,
            _diagnosticShadowCellState);
        SetBuffer(
            _classifyKernel,
            DiagnosticShadowRayStateId,
            _diagnosticShadowRayState);
        _surfaceCompute.SetTexture(
            _classifyKernel,
            TemporalStateId,
            _temporalState);
        SetBuffer(_buildDispatchKernel, CountersId, _counters);
        SetBuffer(_buildDispatchKernel, DispatchArgsId, _dispatchArgs);
        SetBuffer(_initSmoothKernel, VerticesId, _vertices);
        SetBuffer(_initSmoothKernel, SmoothPositionAId, _smoothPositionA);
        SetBuffer(_initSmoothKernel, CountersId, _counters);
        SetBuffer(_smoothKernel, VerticesId, _vertices);
        SetBuffer(_smoothKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_smoothKernel, SmoothPositionAId, _smoothPositionA);
        SetBuffer(_smoothKernel, SmoothPositionBId, _smoothPositionB);
        SetBuffer(_smoothKernel, CountersId, _counters);
        SetBuffer(_applySmoothKernel, VerticesId, _vertices);
        SetBuffer(_applySmoothKernel, SmoothPositionAId, _smoothPositionA);
        SetBuffer(_applySmoothKernel, CountersId, _counters);
        SetBuffer(_temporalKernel, VerticesId, _vertices);
        SetBuffer(_temporalKernel, CountersId, _counters);
        _surfaceCompute.SetTexture(_temporalKernel, TemporalStateId, _temporalState);
        SetBuffer(_generateLinesKernel, CoordVertexMapId, _coordVertexMap);
        SetBuffer(_generateLinesKernel, VerticesId, _vertices);
        SetBuffer(_generateLinesKernel, LineIndicesId, _lineIndices);
        SetBuffer(_generateLinesKernel, CountersId, _counters);
        SetBuffer(_buildDrawArgsKernel, CountersId, _counters);
        SetBuffer(_buildDrawArgsKernel, DrawArgsId, _drawArgs);
        SetBuffer(_evaluateShadowDepthKernel, VerticesId, _vertices);
        SetBuffer(_evaluateShadowDepthKernel, CountersId, _counters);
        SetBuffer(
            _evaluateShadowDepthKernel,
            DiagnosticShadowCellStateId,
            _diagnosticShadowCellState);
        SetBuffer(
            _evaluateShadowDepthKernel,
            DiagnosticShadowRayStateId,
            _diagnosticShadowRayState);
        SetBuffer(
            _evaluateShadowDepthKernel,
            DiagnosticShadowViewOriginStateId,
            _diagnosticShadowViewOriginState);
        SetBuffer(_evaluateShadowDuplicateKernel, VerticesId, _vertices);
        SetBuffer(
            _evaluateShadowDuplicateKernel,
            CoordVertexMapId,
            _coordVertexMap);
        SetBuffer(_evaluateShadowDuplicateKernel, CountersId, _counters);
        SetBuffer(
            _evaluateShadowDuplicateKernel,
            DiagnosticShadowCellStateId,
            _diagnosticShadowCellState);
        SetBuffer(
            _evaluateShadowDuplicateKernel,
            DiagnosticShadowRayStateId,
            _diagnosticShadowRayState);
        _surfaceCompute.SetTexture(_initTemporalKernel, TemporalStateId, _temporalState);
        SetBuffer(
            _initTemporalKernel,
            DiagnosticPreviousCellStateId,
            _diagnosticPreviousCellState);
        SetBuffer(
            _initTemporalKernel,
            DiagnosticShadowCellStateId,
            _diagnosticShadowCellState);
        SetBuffer(
            _initTemporalKernel,
            DiagnosticShadowRayStateId,
            _diagnosticShadowRayState);
        SetBuffer(
            _initTemporalKernel,
            DiagnosticShadowViewOriginStateId,
            _diagnosticShadowViewOriginState);
    }

    private void SetSurfaceConstants()
    {
        _surfaceCompute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        _surfaceCompute.SetInt(TotalVoxelCountId, _totalVoxelCount);
        _surfaceCompute.SetInt(MaximumVerticesId, _maximumVertices);
        _surfaceCompute.SetInt(MaximumLineIndicesId, _maximumLineIndices);
        _surfaceCompute.SetFloat(VoxelSizeId, voxelSizeMeters);
        _surfaceCompute.SetFloat(MinimumWeightId, minimumMeshWeight);
        _surfaceCompute.SetFloat("_SmoothLambda", smoothLambda);
        _surfaceCompute.SetFloat("_SmoothBeta", smoothBeta);
        _surfaceCompute.SetFloat("_TemporalAlphaMaximum", temporalAlphaMaximum);
        _surfaceCompute.SetFloat("_TemporalAlphaMinimum", temporalAlphaMinimum);
        _surfaceCompute.SetFloat("_TemporalDecayRate", temporalDecayRate);
        _surfaceCompute.SetFloat("_ConvergenceThreshold", convergenceThresholdMeters);
        _surfaceCompute.SetFloat("_TemporalDeadzone", temporalDeadzoneMeters);
        _surfaceCompute.SetFloat(
            "_EvidenceIntegrationOrdinal",
            _evidenceIntegrationOrdinal);
        _surfaceCompute.SetFloat(
            "_DiagnosticStaleIntegrationThreshold",
            Mathf.Max(1, staleDirectSupportIntegrationThreshold));
        _surfaceCompute.SetFloat(
            "_DiagnosticFreeContradictionIntegrationThreshold",
            Mathf.Max(1, freeSpaceContradictionIntegrationThreshold));
        _surfaceCompute.SetInt(
            "_EnableLifecycleDiagnostics",
            enableReadOnlyLifecycleDiagnostics ? 1 : 0);
        _surfaceCompute.SetInt(
            "_EnableShadowDiagnostics",
            (enableReadOnlyHoldAndLegalityShadows ||
             enableConservativeProductionLegalityGate) ? 1 : 0);
        _surfaceCompute.SetInt(
            "_EnableProductionLegalityGate",
            enableConservativeProductionLegalityGate ? 1 : 0);
        _surfaceCompute.SetInt(
            "_ShadowMaximumVertices",
            Mathf.Clamp(
                shadowMaximumEvaluatedVertices,
                4096,
                _maximumVertices));
        _surfaceCompute.SetInt(
            "_ShadowDuplicateSearchVoxels",
            Mathf.Clamp(shadowDuplicateSearchVoxels, 1, 6));
        _surfaceCompute.SetFloat(
            "_ShadowDepthTolerance",
            Mathf.Max(0.001f, voxelSizeMeters * shadowDepthToleranceInVoxels));
        _surfaceCompute.SetFloat(
            "_ShadowParallelNormalDot",
            Mathf.Clamp(shadowParallelNormalDot, 0.75f, 0.999f));
        _surfaceCompute.SetFloat(
            "_ShadowDuplicateMinimumSeparation",
            Mathf.Max(
                0.001f,
                voxelSizeMeters * shadowDuplicateMinimumSeparationInVoxels));
        _surfaceCompute.SetFloat(
            "_ShadowDuplicateMaximumSeparation",
            Mathf.Max(
                voxelSizeMeters,
                voxelSizeMeters * shadowDuplicateMaximumSeparationInVoxels));
        _surfaceCompute.SetFloat(
            "_ShadowDuplicateMaximumTangentialDistance",
            Mathf.Max(
                0.001f,
                voxelSizeMeters *
                    shadowDuplicateMaximumTangentialDistanceInVoxels));
        _surfaceCompute.SetInt(
            "_ShadowRecentSupportWindow",
            Mathf.Clamp(shadowRecentCurrentDepthSupportIntegrations, 4, 128));
        _surfaceCompute.SetInt(
            "_ShadowRayFreeMinimumConfirmations",
            Mathf.Clamp(shadowRayFreeMinimumConfirmations, 2, 8));
        _surfaceCompute.SetInt(
            "_ShadowRayFreeSingleViewConfirmations",
            Mathf.Clamp(shadowRayFreeSingleViewConfirmations, 3, 16));
        _surfaceCompute.SetInt(
            "_ShadowRayBehindMinimumConfirmations",
            Mathf.Clamp(shadowRayBehindMinimumConfirmations, 2, 12));
        float baselineNear = Mathf.Max(0.005f, shadowViewBaselineNearMeters);
        float baselineStrong = Mathf.Max(baselineNear, shadowViewBaselineStrongMeters);
        float baselineWide = Mathf.Max(baselineStrong, shadowViewBaselineWideMeters);
        float parallaxNear = Mathf.Clamp(shadowViewParallaxNearDegrees, 0.25f, 5f);
        float parallaxStrong = Mathf.Max(parallaxNear, shadowViewParallaxStrongDegrees);
        float parallaxWide = Mathf.Max(parallaxStrong, shadowViewParallaxWideDegrees);
        _surfaceCompute.SetFloat("_ShadowViewBaselineNear", baselineNear);
        _surfaceCompute.SetFloat("_ShadowViewBaselineStrong", baselineStrong);
        _surfaceCompute.SetFloat("_ShadowViewBaselineWide", baselineWide);
        _surfaceCompute.SetFloat(
            "_ShadowViewParallaxNearCos",
            Mathf.Cos(parallaxNear * Mathf.Deg2Rad));
        _surfaceCompute.SetFloat(
            "_ShadowViewParallaxStrongCos",
            Mathf.Cos(parallaxStrong * Mathf.Deg2Rad));
        _surfaceCompute.SetFloat(
            "_ShadowViewParallaxWideCos",
            Mathf.Cos(parallaxWide * Mathf.Deg2Rad));
        _surfaceCompute.SetVector(
            "_ShadowSourceSize",
            new Vector4(_shadowSourceSize.x, _shadowSourceSize.y, 0f, 0f));
        _surfaceCompute.SetVector(
            "_ShadowEyeWorldPosition",
            new Vector4(
                _shadowEyeWorld.x,
                _shadowEyeWorld.y,
                _shadowEyeWorld.z,
                1f));
        Vector3 shadowEyeLocal = _volumeLocalToWorld.inverse.MultiplyPoint3x4(
            _shadowEyeWorld);
        _surfaceCompute.SetVector(
            "_ShadowEyeLocalPosition",
            new Vector4(
                shadowEyeLocal.x,
                shadowEyeLocal.y,
                shadowEyeLocal.z,
                1f));
        _surfaceCompute.SetMatrix("_ShadowWorldToClip", _shadowWorldToClip);
        _surfaceCompute.SetMatrix(VolumeLocalToWorldId, _volumeLocalToWorld);
    }

    private void BindTsdfVolumes()
    {
        _tsdfCompute.SetTexture(_clearTsdfKernel, VolumeId, _tsdfVolume);
        _tsdfCompute.SetTexture(_clearTsdfKernel, EvidenceVolumeId, _evidenceVolume);
    }

    private void ResetEvidenceDiagnosticCounts()
    {
        LastDiagnosticDirectVertexCount = 0;
        LastDiagnosticBackCapVertexCount = 0;
        LastDiagnosticFreeContradictedVertexCount = 0;
        LastDiagnosticUnknownVertexCount = 0;
        LastDiagnosticRearVertexCount = 0;
        LastDiagnosticStaleVertexCount = 0;
        LastDiagnosticUnresolvedVertexCount = 0;
        LastDiagnosticQuadCount = 0;
        LastDiagnosticDirectQuadCount = 0;
        LastDiagnosticBackCapQuadCount = 0;
        LastDiagnosticFreeContradictedQuadCount = 0;
        LastDiagnosticUnknownQuadCount = 0;
        LastDiagnosticRearQuadCount = 0;
        LastDiagnosticStaleQuadCount = 0;
        LastDiagnosticUnresolvedQuadCount = 0;
        LastDiagnosticSuppressedLineSegmentCount = 0;
        LastLifecycleActiveCellCount = 0;
        LastLifecycleVisibleCellCount = 0;
        LastLifecycleHiddenCellCount = 0;
        LastLifecycleVisibleToHiddenCount = 0;
        LastLifecycleHiddenToVisibleCount = 0;
        LastLifecycleFreshDirectToStaleCount = 0;
        LastLifecycleActiveToInactiveCount = 0;
        LastLifecycleInactiveToActiveCount = 0;
        LastLifecycleClassChangedCount = 0;
        LastLifecycleDirectAge0To7Count = 0;
        LastLifecycleDirectAge8To15Count = 0;
        LastLifecycleDirectAge16To31Count = 0;
        LastLifecycleDirectAge32To47Count = 0;
        LastLifecycleDirectAge48PlusCount = 0;
        LastTemporalFirstSeenCount = 0;
        LastTemporalDeadzoneHeldCount = 0;
        LastTemporalSmallBlendCount = 0;
        LastTemporalLargeResetCount = 0;
        LastLifecycleHiddenDirectBearingCount = 0;
        LastLifecycleYellowUnknownCount = 0;
        LastLifecycleGreenDirectCount = 0;
        LastLifecycleVisibleUnresolvedCount = 0;
        LastStaleDirectCorners1Count = 0;
        LastStaleDirectCorners2To3Count = 0;
        LastStaleDirectCorners4PlusCount = 0;
        LastStaleDirectWrites1To3Count = 0;
        LastStaleDirectWrites4To15Count = 0;
        LastStaleDirectWrites16To63Count = 0;
        LastStaleDirectWrites64PlusCount = 0;
        LastStaleDirectCrossingNoDirectEndpointCount = 0;
        LastStaleDirectCrossingOneSidedOnlyCount = 0;
        LastStaleDirectCrossingAnyTwoSidedCount = 0;
        LastStaleDirectRecentFreeBelowGateCount = 0;
        LastStaleDirectNewerFreeBelowGateCount = 0;
        LastStaleDirectRearMixedCount = 0;
        LastStaleDirectNoConflictHintCount = 0;
        LastStaleDirectStrongCandidateCount = 0;
        LastStaleDirectWeakCandidateCount = 0;
        LastShadowMatureActiveCount = 0;
        LastShadowMatureInactiveCount = 0;
        LastShadowHoldAge0To8Count = 0;
        LastShadowHoldAge9To24Count = 0;
        LastShadowHoldAge25To64Count = 0;
        LastShadowHoldExpired65PlusCount = 0;
        LastShadowHoldContradictedCount = 0;
        LastShadowRecovered0To8Count = 0;
        LastShadowRecovered9To24Count = 0;
        LastShadowRecovered25To64Count = 0;
        LastShadowRecovered65PlusCount = 0;
        LastShadowInactiveCrossingBelow3Count = 0;
        LastShadowInactiveNoObservedCrossingCount = 0;
        LastShadowInactiveTemporalPositionAvailableCount = 0;
        LastShadowDepthEvaluatedCount = 0;
        LastShadowDepthAlignedCount = 0;
        LastShadowDepthFreeContradictedCount = 0;
        LastShadowDepthBehindCurrentCount = 0;
        LastShadowDepthUnobservedCount = 0;
        LastShadowDuplicatePairCount = 0;
        LastShadowDuplicateFrontAlignedCount = 0;
        LastShadowDuplicateBackAlignedCount = 0;
        LastShadowDuplicateBackFreeContradictedCount = 0;
        LastShadowDuplicateBackBehindCurrentCount = 0;
        LastShadowDuplicateBackUnobservedCount = 0;
        LastShadowDuplicateRevokeCandidateCount = 0;
        LastShadowDuplicateAmbiguousCount = 0;
        LastShadowDuplicateBackRecentlySupportedCount = 0;
        LastShadowDuplicateBackSupportStaleCount = 0;
        LastShadowEverRecoveredCount = 0;
        LastShadowEverContradictedCount = 0;
        LastShadowEverExpiredCount = 0;
        LastShadowEverDuplicateCandidateCount = 0;
        LastShadowEverRevokeCandidateCount = 0;
        LastShadowEverAmbiguousCount = 0;
        LastShadowRayTrackedCount = 0;
        LastShadowRayEverAlignedCount = 0;
        LastShadowRayEverFreeCount = 0;
        LastShadowRayEverBehindCount = 0;
        LastShadowRayMultiView2PlusCount = 0;
        LastShadowRayMultiView3PlusCount = 0;
        LastShadowRayFreeCandidateCount = 0;
        LastShadowRayBehindOnlySuspectCount = 0;
        LastShadowRayAmbiguousCount = 0;
        LastShadowRayNeverAlignedCount = 0;
        LastShadowRayRecentlySupportedCount = 0;
        LastShadowRayFreeAlignedMixedCount = 0;
        LastShadowRayBehindAlignedMixedCount = 0;
        LastShadowRayDiversityFlagTotalCount = 0;
        LastShadowRayBaselineNearCount = 0;
        LastShadowRayBaselineStrongCount = 0;
        LastShadowRayBaselineWideCount = 0;
        LastShadowRayParallaxNearCount = 0;
        LastShadowRayParallaxStrongCount = 0;
        LastShadowRayParallaxWideCount = 0;
        LastShadowRayDiverseCount = 0;
        LastShadowRayStrongDiverseCount = 0;
        LastShadowRaySingleViewFreeRepeatCount = 0;
        LastShadowRayOriginClampedCount = 0;
        LastProductionLegalitySuppressedCount = 0;
        LastProductionDuplicateSuppressedPairCount = 0;
        LastProductionDuplicateAwaitingRepeatCount = 0;
        LastProductionDuplicateRejectedRayHistoryCount = 0;
        LastProductionLegalityRecoveredAlignedCount = 0;
        LastShadowRayRevisited2PlusCount = 0;
        LastShadowRayRevisited4PlusCount = 0;
        LastShadowRayRevisitedWithoutBaselineCount = 0;
    }

    private void RecordShadowEvaluationPose()
    {
        if (!_hasShadowEvaluationPose)
        {
            _hasShadowEvaluationPose = true;
            _firstShadowEvaluationEyeWorld = _shadowEyeWorld;
            _lastShadowEvaluationEyeWorld = _shadowEyeWorld;
            ShadowEvaluationPoseCount = 1;
            return;
        }

        float step = Vector3.Distance(
            _lastShadowEvaluationEyeWorld,
            _shadowEyeWorld);
        ShadowEvaluationPoseCount++;
        ShadowEvaluationPoseAccumulatedTravelMeters += step;
        ShadowEvaluationPoseMaxStepMeters = Mathf.Max(
            ShadowEvaluationPoseMaxStepMeters,
            step);
        ShadowEvaluationPoseMaxBaselineMeters = Mathf.Max(
            ShadowEvaluationPoseMaxBaselineMeters,
            Vector3.Distance(
                _firstShadowEvaluationEyeWorld,
                _shadowEyeWorld));
        _lastShadowEvaluationEyeWorld = _shadowEyeWorld;
    }

    private void SetBuffer(int kernel, int property, GraphicsBuffer buffer)
    {
        _surfaceCompute.SetBuffer(kernel, property, buffer);
    }

    private void DispatchVolume(ComputeShader compute, int kernel)
    {
        compute.SetInts(VoxelCountId, volumeSizeX, volumeSizeY, volumeSizeZ);
        uint tx;
        uint ty;
        uint tz;
        compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
        compute.Dispatch(
            kernel,
            Mathf.CeilToInt(volumeSizeX / (float)tx),
            Mathf.CeilToInt(volumeSizeY / (float)ty),
            Mathf.CeilToInt(volumeSizeZ / (float)tz));
    }

    private static void DispatchTexture2D(
        ComputeShader compute,
        int kernel,
        int width,
        int height)
    {
        compute.GetKernelThreadGroupSizes(
            kernel,
            out uint threadX,
            out uint threadY,
            out _);
        compute.Dispatch(
            kernel,
            Mathf.CeilToInt(width / (float)threadX),
            Mathf.CeilToInt(height / (float)threadY),
            1);
    }

    private bool EnsureCompatibilityDilationTextures(int width, int height)
    {
        if (width <= 1 || height <= 1) return false;
        if (_compatibilityDilationA != null &&
            _compatibilityDilationB != null &&
            _compatibilityDilationA.width == width &&
            _compatibilityDilationA.height == height &&
            _compatibilityDilationB.width == width &&
            _compatibilityDilationB.height == height)
            return true;

        ReleaseTexture(ref _compatibilityDilationA);
        ReleaseTexture(ref _compatibilityDilationB);
        _compatibilityDilationA = CreateCompatibilityDilationTexture(
            "ScanCover_QuestRoom_Compatibility_DilationA",
            width,
            height);
        _compatibilityDilationB = CreateCompatibilityDilationTexture(
            "ScanCover_QuestRoom_Compatibility_DilationB",
            width,
            height);
        return _compatibilityDilationA != null &&
               _compatibilityDilationB != null &&
               _compatibilityDilationA.IsCreated() &&
               _compatibilityDilationB.IsCreated();
    }

    private static RenderTexture CreateCompatibilityDilationTexture(
        string textureName,
        int width,
        int height)
    {
        RenderTexture texture = new RenderTexture(width, height, 0)
        {
            name = textureName,
            dimension = TextureDimension.Tex2D,
            graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat,
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false
        };
        texture.Create();
        return texture;
    }

    private RenderTexture CreateVolume(string textureName, GraphicsFormat format)
    {
        RenderTexture texture = new RenderTexture(volumeSizeX, volumeSizeY, 0)
        {
            name = textureName,
            dimension = TextureDimension.Tex3D,
            volumeDepth = volumeSizeZ,
            graphicsFormat = format,
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false
        };
        texture.Create();
        return texture;
    }

    private void OnDestroy()
    {
        ReleaseResources();
    }

    private void ReleaseResources()
    {
        ReleaseBuffer(ref _coordVertexMap);
        ReleaseBuffer(ref _vertices);
        ReleaseBuffer(ref _lineIndices);
        ReleaseBuffer(ref _counters);
        ReleaseBuffer(ref _dispatchArgs);
        ReleaseBuffer(ref _drawArgs);
        ReleaseBuffer(ref _smoothPositionA);
        ReleaseBuffer(ref _smoothPositionB);
        ReleaseBuffer(ref _diagnosticPreviousCellState);
        ReleaseBuffer(ref _diagnosticShadowCellState);
        ReleaseBuffer(ref _diagnosticShadowRayState);
        ReleaseBuffer(ref _diagnosticShadowViewOriginState);
        ReleaseBuffer(ref _compatibilityCounters);
        ReleaseTexture(ref _tsdfVolume);
        ReleaseTexture(ref _evidenceVolume);
        ReleaseTexture(ref _temporalState);
        ReleaseTexture(ref _compatibilityTsdfVolume);
        ReleaseTexture(ref _compatibilityDilationA);
        ReleaseTexture(ref _compatibilityDilationB);
        if (_wireMaterial != null) Destroy(_wireMaterial);
        _wireMaterial = null;
        _shadowWorldPosition = null;
        _shadowObservationMeta = null;
        _shadowCurrentInputValid = false;
        _compatibilityCounterReadbackPending = false;
        _resourcesReady = false;
    }

    private static void ReleaseBuffer(ref GraphicsBuffer buffer)
    {
        if (buffer == null) return;
        buffer.Release();
        buffer = null;
    }

    private static void ReleaseTexture(ref RenderTexture texture)
    {
        if (texture == null) return;
        if (texture.IsCreated()) texture.Release();
        Destroy(texture);
        texture = null;
    }
}

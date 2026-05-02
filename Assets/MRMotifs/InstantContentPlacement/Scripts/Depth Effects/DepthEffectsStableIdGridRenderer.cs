using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR.EnvironmentDepth;
using Meta.XR.Samples;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsStableIdGridRenderer : MonoBehaviour
    {
        private enum DisplayMode
        {
            StableWorldCells,
            GridpointSurfaceMask,
            SurfaceLatticeWorldCells,
            WorldCellLinks,
            LocalPlaneCandidateCenters,
            ConfirmedStructureCenters,
        }

        private const int CopyTextureSize = 128;
        private const int NumEyes = 2;
        private static readonly Vector3Int InvalidWorldCellId = new(int.MinValue, int.MinValue, int.MinValue);
        private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        private static readonly int EnvironmentDepthTextureSizeId = Shader.PropertyToID("_EnvironmentDepthTextureSize");
        private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
        private static readonly int EnvironmentDepthInverseReprojectionMatricesId =
            Shader.PropertyToID("_EnvironmentDepthInverseReprojectionMatrices");
        private static readonly int EnvironmentDepthReprojectionMatricesId =
            Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
        private static readonly int CopiedDepthTextureId = Shader.PropertyToID("_CopiedDepthTexture");

        [Header("Refs")]
        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private Material materialOverride;

        [Header("Sampling")]
        [SerializeField]
        private int gridColumns = 24;

        [SerializeField]
        private int gridRows = 18;

        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportCenterY = 0.62f;

        [SerializeField]
        private float viewportWidth = 0.88f;

        [SerializeField]
        private float viewportHeight = 0.76f;

        [SerializeField]
        private float sampleIntervalSeconds = 0.08f;

        [SerializeField]
        private float minLinearDepthMeters = 0.25f;

        [SerializeField]
        private float maxLinearDepthMeters = 3.5f;

        [Header("World Cells")]
        [SerializeField]
        private float worldCellSizeMeters = 0.12f;

        [SerializeField]
        private float worldCellMergeDistanceMeters = 0.08f;

        [SerializeField]
        private int minStableHits = 3;

        [SerializeField]
        private float holdSeconds = 6.0f;

        [SerializeField]
        private float stableCellHoldSeconds = 18.0f;

        [SerializeField]
        private float worldCellReconnectDistanceMeters = 0.18f;

        [SerializeField]
        private float worldCellReconnectNormalDot = 0.7f;

        [Header("World Cell Memory")]
        [SerializeField]
        private int retainedMemoryMinStableHits = 5;

        [SerializeField]
        private float retainedMemoryMinConfidence = 0.35f;

        [SerializeField]
        private float retainedMemoryMinQualityScore = 0.58f;

        [SerializeField]
        private float retainedMemoryMaxUnseenSeconds = 45.0f;

        [SerializeField]
        private float retainedMemoryReconnectDistanceMeters = 0.22f;

        [SerializeField]
        private float retainedMemoryReconnectNormalDot = 0.82f;

        [SerializeField]
        private float retainedMemoryOverwriteDistanceMeters = 0.12f;

        [SerializeField]
        private float retainedMemoryOverwriteNormalDot = 0.88f;

        [SerializeField]
        private int retainedMemoryReconnectSupportFrames = 2;

        [SerializeField]
        private float retainedMemoryCooldownSeconds = 6.0f;

        [SerializeField]
        private float activeRetainedMaxUnseenSeconds = 5.0f;

        [SerializeField]
        private float warmStaleCandidateMaxUnseenSeconds = 9.0f;

        [SerializeField]
        private float positionBlend = 0.24f;

        [SerializeField]
        private float normalBlend = 0.18f;

        [SerializeField]
        private float confidenceFloor = 0.22f;

        [Header("Display Stability")]
        [SerializeField]
        private int freezeAfterStableHits = 5;

        [SerializeField]
        private float displayUnlockDistanceMeters = 0.08f;

        [SerializeField]
        private int unlockAfterOutlierFrames = 3;

        [SerializeField]
        private int relockAfterStableFrames = 3;

        [SerializeField]
        private float relocatingDisplayPositionBlend = 0.2f;

        [SerializeField]
        private float relocatingDisplayNormalBlend = 0.2f;

        [Header("Display")]
        [SerializeField]
        private bool showGridPoints = true;

        [SerializeField]
        private DisplayMode displayMode = DisplayMode.LocalPlaneCandidateCenters;

        [SerializeField]
        private PrimitiveType pointPrimitive = PrimitiveType.Cube;

        [SerializeField]
        private float pointScaleMeters = 0.02f;

        [SerializeField]
        private float surfaceLatticePointScaleRatio = 0.35f;

        [SerializeField]
        private float surfaceLatticeThicknessMeters = 0.006f;

        [SerializeField]
        private float gridpointSurfaceSpacingMeters = 0.16f;

        [SerializeField]
        private float gridpointSurfacePointScaleRatio = 0.22f;

        [SerializeField]
        private float surfaceOffsetMeters = 0.01f;

        [SerializeField]
        private Color pointColor = new(0.92f, 0.96f, 0.99f, 0.96f);

        [SerializeField]
        private Color lowSupportWorldCellColor = new(0.98f, 0.42f, 0.32f, 0.98f);

        [SerializeField]
        private float worldCellLinkThicknessMeters = 0.01f;

        [SerializeField]
        private float worldCellLinkMaxDistanceMeters = 0.22f;

        [SerializeField]
        private Color worldCellLinkColor = new(0.62f, 0.82f, 0.98f, 0.92f);

        [Header("Local Plane Candidates")]
        [SerializeField]
        private int minCandidateStableHits = 1;

        [SerializeField]
        private int minCandidateCells = 2;

        [SerializeField]
        private float maxCandidatePlaneResidualMeters = 0.12f;

        [SerializeField]
        private float minCandidateAxisAlignmentScore = 0.0f;

        [SerializeField]
        private float minCandidateNormalDot = 0.25f;

        [SerializeField]
        private float maxCandidateNeighborOffsetMeters = 0.45f;

        [SerializeField]
        private float candidateCenterScaleMeters = 0.035f;

        [SerializeField]
        private Color candidateCenterColor = new(0.96f, 0.83f, 0.32f, 0.98f);

        [SerializeField]
        private Color lowSupportCandidateColor = new(0.98f, 0.35f, 0.72f, 0.98f);

        [Header("Low Support Debug")]
        [SerializeField]
        private bool highlightLowSupport = true;

        [SerializeField]
        private float lowSupportCellMaxConfidence = 0.72f;

        [SerializeField]
        private float lowSupportCellMaxQualityScore = 0.82f;

        [SerializeField]
        private int lowSupportCellMaxStableHits = 6;

        [SerializeField]
        private float lowSupportCellMinUnseenSeconds = 0.75f;

        [SerializeField]
        private int lowSupportCandidateMaxMemberCount = 4;

        [SerializeField]
        private float lowSupportCandidateMinResidualMeters = 0.05f;

        [SerializeField]
        private float lowSupportCandidateMinDeltaMeters = 0.06f;

        [SerializeField]
        private float lowSupportCandidateMinUnseenSeconds = 0.35f;

        [SerializeField]
        private float lowSupportCandidateMaxAxisAlignmentScore = 0.88f;

        [Header("Candidate Persistence")]
        [SerializeField]
        private float candidateMatchDistanceMeters = 0.5f;

        [SerializeField]
        private float persistentCandidateMaxReassociationDistanceMeters = 0.22f;

        [SerializeField]
        private float matureCandidateMaxReassociationDistanceMeters = 0.10f;

        [SerializeField]
        private float candidateHoldSeconds = 10.0f;

        [SerializeField]
        private float visibleCandidateGraceSeconds = 3.0f;

        [SerializeField]
        private int minPersistentCandidateFrames = 2;

        [SerializeField]
        private float minCandidateMatchNormalDot = 0.8f;

        [SerializeField]
        private int maxCandidateMemberCountDelta = 16;

        [SerializeField]
        private float candidateMatchDistanceWeight = 1.0f;

        [SerializeField]
        private float candidateMatchNormalWeight = 0.45f;

        [SerializeField]
        private float candidateMatchMemberWeight = 0.15f;

        [SerializeField]
        private float candidateMatchResidualWeight = 0.2f;

        [SerializeField]
        private int freezeCandidateAfterFrames = 3;

        [SerializeField]
        private float candidateCenterBlend = 0.2f;

        [SerializeField]
        private float candidateNormalBlend = 0.18f;

        [SerializeField]
        private float candidateUnlockDistanceMeters = 0.12f;

        [SerializeField]
        private float candidateHardRelocationDistanceMeters = 0.2f;

        [SerializeField]
        private int candidateHardRelocationResetStableFrames = 2;

        [SerializeField]
        private float candidateMediumRelocationDistanceMeters = 0.07f;

        [SerializeField]
        private int candidateMediumRelocationResetStableFrames = 1;

        [SerializeField]
        private int candidateUnlockAfterOutlierFrames = 2;

        [SerializeField]
        private int candidateRelockAfterStableFrames = 2;

        [SerializeField]
        private float candidateRelocatingCenterBlend = 0.22f;

        [SerializeField]
        private float candidateRelocatingNormalBlend = 0.2f;

        [Header("Confirmed Structures")]
        [SerializeField]
        private int minConfirmedCandidateFrames = 4;

        [SerializeField]
        private int minConfirmedCandidateMembers = 6;

        [SerializeField]
        private float maxConfirmedCandidateResidualMeters = 0.04f;

        [SerializeField]
        private float minConfirmedAxisAlignmentScore = 0.78f;

        [SerializeField]
        private float confirmedMatchDistanceMeters = 0.45f;

        [SerializeField]
        private float confirmedMaxReassociationDistanceMeters = 0.24f;

        [SerializeField]
        private float minConfirmedMatchNormalDot = 0.88f;

        [SerializeField]
        private float confirmedCenterScaleMeters = 0.05f;

        [SerializeField]
        private Color confirmedCenterColor = new(0.22f, 0.9f, 0.55f, 0.98f);

        [SerializeField]
        private float confirmedCenterBlend = 0.16f;

        [SerializeField]
        private float confirmedNormalBlend = 0.16f;

        [SerializeField]
        private float confirmedUnlockDistanceMeters = 0.2f;

        [SerializeField]
        private int confirmedUnlockAfterOutlierFrames = 3;

        [SerializeField]
        private int confirmedRelockAfterStableFrames = 2;

        [SerializeField]
        private float confirmedRelocatingCenterBlend = 0.22f;

        [SerializeField]
        private float confirmedRelocatingNormalBlend = 0.2f;

        [SerializeField]
        private float confirmedMergeDistanceMeters = 0.32f;

        [SerializeField]
        private float confirmedMergeNormalDot = 0.90f;

        [SerializeField]
        private float replacedConfirmedCleanupSeconds = 1.5f;

        [SerializeField]
        private float lowQualityConfirmedCleanupSeconds = 5.0f;

        [SerializeField]
        private float lowQualityConfirmedMinSupportScore = 26.0f;

        [SerializeField]
        private float lowQualityConfirmedMaxResidualMeters = 0.045f;

        [SerializeField]
        private float lowQualityConfirmedMinAxisAlignmentScore = 0.89f;

        [SerializeField]
        private float confirmedDisplayMinSupportScore = 30.0f;

        [SerializeField]
        private float confirmedDisplayMaxResidualMeters = 0.04f;

        [SerializeField]
        private float confirmedDisplayMinAxisAlignmentScore = 0.90f;

        [SerializeField]
        private float confirmedDisplayLowQualityGraceSeconds = 0.75f;

        [SerializeField]
        private float confirmedViewportPadding = 0.12f;

        [SerializeField]
        private float confirmedDisplayViewportPadding = 0.18f;

        [SerializeField]
        private float confirmedRecentDisplaySeconds = 6.0f;

        [SerializeField]
        private float confirmedPeripheralRecentDisplaySeconds = 1.8f;

        [SerializeField]
        private int confirmedMaxVisibleStructures = 96;

        [SerializeField]
        private int confirmedCenterVisibleBudget = 36;

        [SerializeField]
        private int confirmedViewportVisibleBudget = 44;

        [SerializeField]
        private int confirmedRecentVisibleBudget = 12;

        [SerializeField]
        private int viewPriorityConfirmedFrameReduction = 2;

        [SerializeField]
        private int viewPriorityConfirmedMemberReduction = 2;

        [SerializeField]
        private float viewPriorityConfirmedResidualScale = 1.35f;

        [SerializeField]
        private float viewPriorityConfirmedAxisAlignmentReduction = 0.08f;

        [Header("Center Priority")]
        [SerializeField]
        private float centerPromotionViewportRadius = 0.34f;

        [SerializeField]
        private float centerDisplayPriorityViewportRadius = 0.30f;

        [SerializeField]
        private float centerConfirmedRecentSupportSeconds = 8.0f;

        [SerializeField]
        private float centerConfirmedViewportDisplaySeconds = 4.5f;

        [SerializeField]
        private float centerConfirmedMatchBias = 0.28f;

        [SerializeField]
        private float centerConfirmedSupportIncrement = 1.5f;

        [SerializeField]
        private int centerPriorityConfirmedFrameReduction = 4;

        [SerializeField]
        private int centerPriorityConfirmedMemberReduction = 4;

        [SerializeField]
        private float centerPriorityConfirmedResidualScale = 1.6f;

        [SerializeField]
        private float centerPriorityConfirmedAxisAlignmentReduction = 0.12f;

        [Header("Ceiling Priority")]
        [SerializeField]
        private float ceilingPriorityMinForwardY = 0.18f;

        [SerializeField]
        private float ceilingPriorityViewportMinY = 0.35f;

        [SerializeField]
        private float ceilingPriorityMinHeightOffsetMeters = 0.55f;

        [SerializeField]
        private int ceilingPriorityConfirmedFrameReduction = 2;

        [SerializeField]
        private int ceilingPriorityConfirmedMemberReduction = 2;

        [SerializeField]
        private float ceilingPriorityConfirmedResidualScale = 1.35f;

        [Header("Desktop Monitor")]
        [SerializeField]
        private bool exportDesktopMonitor = true;

        [SerializeField]
        private float desktopMonitorWriteIntervalSeconds = 0.5f;

        [SerializeField]
        private string desktopMonitorFileName = "DepthEffectsStableIdGridMonitor.md";

        [SerializeField]
        private bool debugLog;

        [Header("Response Monitor")]
        [SerializeField]
        private float responseTriggerPositionDeltaMeters = 0.12f;

        [SerializeField]
        private float responseTriggerAngularDeltaDegrees = 18f;

        [SerializeField]
        private float responseRetriggerCooldownSeconds = 0.75f;

        [SerializeField]
        private int responseMinViewportConfirmedCount = 1;

        private ComputeShader m_copyShader;
        private ComputeBuffer m_computeBuffer;
        private NativeArray<float> m_depthPixels;
        private NativeArray<float> m_gpuReadbackBuffer;
        private AsyncGPUReadbackRequest m_pendingReadback;
        private bool m_hasPendingReadback;
        private float m_nextSampleTime;
        private float m_nextMonitorWriteTime;

        private Mesh m_pointMesh;
        private Material m_runtimeMaterial;
        private Matrix4x4[] m_batchMatrices = Array.Empty<Matrix4x4>();
        private Matrix4x4[] m_secondaryBatchMatrices = Array.Empty<Matrix4x4>();
        private MaterialPropertyBlock m_propertyBlock;
        private bool m_hasLastMonitorPose;
        private Vector3 m_lastMonitorCameraPosition;
        private Quaternion m_lastMonitorCameraRotation = Quaternion.identity;
        private bool m_hasResponseReferencePose;
        private Vector3 m_responseReferenceCameraPosition;
        private Quaternion m_responseReferenceCameraRotation = Quaternion.identity;
        private bool m_responseSessionActive;
        private float m_responseSessionStartTime;
        private float m_responseLastTriggerTime = float.NegativeInfinity;
        private float m_responseFirstConfirmedLatency = -1f;
        private float m_responsePeakConfirmedLatency = -1f;
        private int m_responseCurrentViewportConfirmedCount;
        private int m_responsePeakViewportConfirmedCount;

        private readonly Dictionary<Vector3Int, CellState> m_worldCells = new();
        private readonly List<Vector3Int> m_cleanupKeys = new();
        private readonly List<LocalPlaneCandidate> m_localPlaneCandidates = new();
        private readonly Dictionary<int, PersistentCandidateState> m_persistentCandidates = new();
        private readonly Dictionary<int, ConfirmedStructureState> m_confirmedStructures = new();
        private readonly List<int> m_candidateCleanupKeys = new();
        private readonly List<int> m_confirmedCleanupKeys = new();
        private readonly List<int> m_confirmedInvalidateKeys = new();
        private readonly List<int> m_visibleConfirmedIds = new();
        private readonly List<int> m_visibleConfirmedCenterIds = new();
        private readonly List<int> m_visibleConfirmedViewportIds = new();
        private readonly List<int> m_visibleConfirmedRecentIds = new();
        private readonly HashSet<Vector3Int> m_candidateVisited = new();
        private readonly Queue<Vector3Int> m_candidateQueue = new();
        private readonly List<CellState> m_candidateMembers = new();
        private readonly Dictionary<Vector3Int, float> m_worldCellCooldownUntil = new();
        private int m_nextPersistentCandidateId = 1;
        private int m_nextConfirmedStructureId = 1;

        private struct LocalPlaneCandidate
        {
            public int candidateId;
            public int memberCount;
            public Vector3 center;
            public Vector3 normal;
            public float fitResidualMeters;
            public float axisAlignmentScore;
            public int axisIndex;
        }

        private struct PersistentCandidateState
        {
            public int persistentId;
            public int axisIndex;
            public CellState.DisplayLockState displayLockState;
            public int stableFrames;
            public int unlockOutlierFrames;
            public int relockStableFrames;
            public float lastSeenTime;
            public int memberCount;
            public float fitResidualMeters;
            public float axisAlignmentScore;
            public Vector3 observedCenter;
            public Vector3 observedNormal;
            public Vector3 displayCenter;
            public Vector3 displayNormal;
            public Vector3 relocatingCenter;
            public Vector3 relocatingNormal;
        }

        private enum ConfirmedStructureStatus
        {
            Confirmed,
            Challenged,
            Replaced,
            Invalidated,
        }

        private struct ConfirmedStructureState
        {
            public int confirmedId;
            public ConfirmedStructureStatus status;
            public int axisIndex;
            public int sourcePersistentCandidateId;
            public CellState.DisplayLockState displayLockState;
            public int unlockOutlierFrames;
            public int relockStableFrames;
            public int memberCount;
            public int stableFrames;
            public float lastObservedTime;
            public float supportScore;
            public float conflictScore;
            public float replacementScore;
            public float fitResidualMeters;
            public float axisAlignmentScore;
            public float lastCenterObservedTime;
            public float centerSupportScore;
            public Vector3 observedCenter;
            public Vector3 observedNormal;
            public Vector3 displayCenter;
            public Vector3 displayNormal;
            public Vector3 relocatingCenter;
            public Vector3 relocatingNormal;
        }

        private struct CellState
        {
            public enum DisplayLockState
            {
                Unlocked,
                Locked,
                Relocating,
            }

            public Vector3Int worldCellId;
            public bool valid;
            public DisplayLockState displayLockState;
            public int unlockOutlierFrames;
            public int relockStableFrames;
            public int stableHits;
            public bool retainedMemory;
            public float memoryQualityScore;
            public int pendingReconnectFrames;
            public float lastSeenTime;
            public Vector3 worldPos;
            public Vector3 normal;
            public Vector3 displayWorldPos;
            public Vector3 displayNormal;
            public Vector3 relocatingWorldPos;
            public Vector3 relocatingNormal;
            public Vector3 pendingReconnectWorldPos;
            public Vector3 pendingReconnectNormal;
            public float confidence;
        }

        private readonly struct SurfaceGridpointKey : IEquatable<SurfaceGridpointKey>
        {
            public readonly int axisIndex;
            public readonly int sliceIndex;
            public readonly int uIndex;
            public readonly int vIndex;

            public SurfaceGridpointKey(int axisIndex, int sliceIndex, int uIndex, int vIndex)
            {
                this.axisIndex = axisIndex;
                this.sliceIndex = sliceIndex;
                this.uIndex = uIndex;
                this.vIndex = vIndex;
            }

            public bool Equals(SurfaceGridpointKey other)
            {
                return axisIndex == other.axisIndex &&
                       sliceIndex == other.sliceIndex &&
                       uIndex == other.uIndex &&
                       vIndex == other.vIndex;
            }

            public override bool Equals(object obj)
            {
                return obj is SurfaceGridpointKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = axisIndex;
                    hash = (hash * 397) ^ sliceIndex;
                    hash = (hash * 397) ^ uIndex;
                    hash = (hash * 397) ^ vIndex;
                    return hash;
                }
            }
        }

        public void Configure(Camera camera, EnvironmentDepthManager depthManager, Material overrideMaterial)
        {
            sampleCamera = camera;
            environmentDepthManager = depthManager;
            materialOverride = overrideMaterial;
        }

        private void OnEnable()
        {
            ResolveRefs();
            EnsureResources();
            m_nextSampleTime = Time.unscaledTime;
            m_nextMonitorWriteTime = Time.unscaledTime;
        }

        private void OnDisable()
        {
            DisposeResources();
        }

        private void Update()
        {
            ResolveRefs();
            UpdatePendingReadback();

            if (!showGridPoints || environmentDepthManager == null || sampleCamera == null)
                return;

            if (Time.unscaledTime >= m_nextSampleTime && !m_hasPendingReadback)
            {
                m_nextSampleTime = Time.unscaledTime + Mathf.Max(0.03f, sampleIntervalSeconds);
                RequestDepthCopy();
            }
        }

        private void LateUpdate()
        {
            if (!showGridPoints)
                return;

            UpdateResponseMonitor(Time.unscaledTime);
            RenderStablePoints();
        }

        private void ResolveRefs()
        {
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);
        }

        private void EnsureResources()
        {
            if (m_copyShader == null)
                m_copyShader = Resources.Load<ComputeShader>("CopyDepthTexture");

            if (m_computeBuffer == null)
            {
                int numPixels = CopyTextureSize * CopyTextureSize * NumEyes;
                m_computeBuffer = new ComputeBuffer(numPixels, sizeof(float));
                m_depthPixels = new NativeArray<float>(numPixels, Allocator.Persistent);
                m_gpuReadbackBuffer = new NativeArray<float>(numPixels, Allocator.Persistent);
            }

            if (m_pointMesh == null)
                m_pointMesh = ResolvePrimitiveMesh(pointPrimitive);

            if (m_runtimeMaterial == null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Lit");
                if (shader == null)
                    shader = Shader.Find("Standard");
                if (shader != null)
                {
                    m_runtimeMaterial = new Material(shader)
                    {
                        name = "DepthEffectsStableIdGridRenderer_Runtime"
                    };
                    m_runtimeMaterial.enableInstancing = true;
                    m_runtimeMaterial.color = pointColor;
                    if (m_runtimeMaterial.HasProperty("_Surface"))
                        m_runtimeMaterial.SetFloat("_Surface", 0f);
                    if (m_runtimeMaterial.HasProperty("_ZWrite"))
                        m_runtimeMaterial.SetFloat("_ZWrite", 1f);
                    if (m_runtimeMaterial.HasProperty("_Cull"))
                        m_runtimeMaterial.SetFloat("_Cull", (float)CullMode.Off);
                }
            }

            m_propertyBlock ??= new MaterialPropertyBlock();
        }

        private void RequestDepthCopy()
        {
            if (m_copyShader == null || m_computeBuffer == null || !m_gpuReadbackBuffer.IsCreated)
                return;
            if (environmentDepthManager == null || !environmentDepthManager.isActiveAndEnabled || !environmentDepthManager.IsDepthAvailable)
                return;

            RenderTexture depthTexture = Shader.GetGlobalTexture(EnvironmentDepthTextureId) as RenderTexture;
            if (depthTexture == null)
                return;

            Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
            m_copyShader.SetTexture(0, EnvironmentDepthTextureId, depthTexture);
            m_copyShader.SetFloat(EnvironmentDepthTextureSizeId, depthTexture.width);
            m_copyShader.SetVector(EnvironmentDepthZBufferParamsId, zParams);
            m_copyShader.SetBuffer(0, CopiedDepthTextureId, m_computeBuffer);
            m_copyShader.Dispatch(0, 1, 1, 1);

            m_pendingReadback = AsyncGPUReadback.RequestIntoNativeArray(ref m_gpuReadbackBuffer, m_computeBuffer);
            m_hasPendingReadback = true;
        }

        private void UpdatePendingReadback()
        {
            if (!m_hasPendingReadback || !m_pendingReadback.done)
                return;

            m_hasPendingReadback = false;
            if (m_pendingReadback.hasError)
            {
                if (debugLog)
                    Debug.LogWarning("[DepthEffectsStableIdGridRenderer] AsyncGPUReadback failed.");
                return;
            }

            (m_depthPixels, m_gpuReadbackBuffer) = (m_gpuReadbackBuffer, m_depthPixels);
            ConsumeDepthSamples();
        }

        private void ConsumeDepthSamples()
        {
            PruneWorldCellCooldowns(Time.unscaledTime);

            Matrix4x4[] inverseMatrices = Shader.GetGlobalMatrixArray(EnvironmentDepthInverseReprojectionMatricesId);
            if (inverseMatrices == null || inverseMatrices.Length <= 0)
            {
                Matrix4x4[] reprojectionMatrices = Shader.GetGlobalMatrixArray(EnvironmentDepthReprojectionMatricesId);
                if (reprojectionMatrices == null || reprojectionMatrices.Length <= 0)
                    return;

                inverseMatrices = new Matrix4x4[reprojectionMatrices.Length];
                for (int i = 0; i < reprojectionMatrices.Length; i++)
                    inverseMatrices[i] = reprojectionMatrices[i].inverse;
            }

            int eyeIndex = 0;
            float now = Time.unscaledTime;
            for (int row = 0; row < gridRows; row++)
            {
                float v = gridRows > 1 ? (float)row / (gridRows - 1) : 0.5f;
                float viewportY = viewportCenterY + (v - 0.5f) * viewportHeight;
                for (int column = 0; column < gridColumns; column++)
                {
                    float u = gridColumns > 1 ? (float)column / (gridColumns - 1) : 0.5f;
                    float viewportX = viewportCenterX + (u - 0.5f) * viewportWidth;

                    Vector2Int texCoord = ViewportToDepthCoord(viewportX, viewportY);
                    float linearDepth = SampleLinearDepth(texCoord, eyeIndex);
                    if (linearDepth < minLinearDepthMeters || linearDepth > maxLinearDepthMeters)
                        continue;

                    if (!TryReconstructWorld(texCoord, linearDepth, inverseMatrices[Mathf.Min(eyeIndex, inverseMatrices.Length - 1)], out Vector3 worldPos))
                        continue;

                    Vector3 normal = ReconstructNormal(texCoord, eyeIndex, inverseMatrices[Mathf.Min(eyeIndex, inverseMatrices.Length - 1)]);
                    float confidence = Mathf.Clamp01(1f - (linearDepth / Mathf.Max(minLinearDepthMeters + 0.1f, maxLinearDepthMeters)));
                    confidence = Mathf.Max(confidenceFloor, confidence);

                    Vector3Int worldCellId = ResolveObservedWorldCellId(worldPos, normal, now);
                    if (worldCellId == InvalidWorldCellId)
                        continue;

                    if (!m_worldCells.TryGetValue(worldCellId, out CellState cell))
                    {
                        cell.worldCellId = worldCellId;
                        cell.valid = true;
                        cell.displayLockState = CellState.DisplayLockState.Unlocked;
                        cell.unlockOutlierFrames = 0;
                        cell.relockStableFrames = 0;
                        cell.stableHits = 1;
                        cell.retainedMemory = false;
                        cell.memoryQualityScore = 0f;
                        cell.pendingReconnectFrames = 0;
                        cell.lastSeenTime = now;
                        cell.worldPos = worldPos;
                        cell.normal = normal;
                        cell.displayWorldPos = worldPos;
                        cell.displayNormal = normal;
                        cell.relocatingWorldPos = worldPos;
                        cell.relocatingNormal = normal;
                        cell.pendingReconnectWorldPos = worldPos;
                        cell.pendingReconnectNormal = normal;
                        cell.confidence = confidence;
                    }
                    else
                    {
                        if (cell.retainedMemory)
                        {
                            TryApplyRetainedMemoryObservation(ref cell, worldPos, normal, confidence, now);
                        }
                        else
                        {
                            float positionDeltaMeters = Vector3.Distance(cell.worldPos, worldPos);
                            float blend = positionDeltaMeters <= Mathf.Max(0.001f, worldCellMergeDistanceMeters)
                                ? Mathf.Clamp01(positionBlend)
                                : 0.08f;

                            cell.stableHits = Mathf.Min(999, cell.stableHits + 1);
                            cell.lastSeenTime = now;
                            cell.worldPos = Vector3.Lerp(cell.worldPos, worldPos, blend);

                            Vector3 blendedNormal = Vector3.Lerp(cell.normal, normal, Mathf.Clamp01(normalBlend));
                            cell.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
                            cell.confidence = Mathf.Lerp(cell.confidence, confidence, 0.2f);
                            cell.pendingReconnectFrames = 0;
                            cell.pendingReconnectWorldPos = cell.worldPos;
                            cell.pendingReconnectNormal = cell.normal;
                        }
                    }

                    float displayDeltaMeters = Vector3.Distance(cell.displayWorldPos, cell.worldPos);
                    float observedQualityScore = ComputeObservedMemoryQuality(cell.stableHits, cell.confidence, displayDeltaMeters);
                    cell.memoryQualityScore = Mathf.Lerp(cell.memoryQualityScore, observedQualityScore, 0.2f);
                    if (ShouldRetainCellMemory(cell))
                        cell.retainedMemory = true;

                    UpdateDisplayState(ref cell);

                    m_worldCells[worldCellId] = cell;
                }
            }

            m_cleanupKeys.Clear();
            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                CellState cell = pair.Value;
                if (!cell.valid)
                    continue;

                if (cell.retainedMemory)
                {
                    if (ShouldInvalidateRetainedMemory(cell, now))
                    {
                        m_worldCellCooldownUntil[pair.Key] = now + Mathf.Max(0.5f, retainedMemoryCooldownSeconds);
                        m_cleanupKeys.Add(pair.Key);
                    }
                    continue;
                }

                float cellHold = cell.stableHits >= minStableHits
                    ? Mathf.Max(holdSeconds, stableCellHoldSeconds)
                    : holdSeconds;
                if (now - cell.lastSeenTime > cellHold)
                    m_cleanupKeys.Add(pair.Key);
            }

            for (int i = 0; i < m_cleanupKeys.Count; i++)
                m_worldCells.Remove(m_cleanupKeys[i]);

            RebuildLocalPlaneCandidates(now);
            UpdatePersistentCandidates(now);
            UpdateConfirmedStructures(now);
            TryWriteDesktopMonitor(now);

            if (debugLog)
            {
                int validCount = 0;
                int stableCount = 0;
                int retainedCount = 0;
                foreach (CellState cell in m_worldCells.Values)
                {
                    if (cell.valid)
                        validCount++;
                    if (cell.valid && cell.stableHits >= minStableHits)
                        stableCount++;
                    if (cell.valid && cell.retainedMemory)
                        retainedCount++;
                }

                int persistentVisible = CountVisiblePersistentCandidates(now);
                int confirmedVisible = CountVisibleConfirmedStructures(now);

                Debug.Log(
                    $"[DepthEffectsStableIdGridRenderer] worldCells valid={validCount}, stable={stableCount}, retained={retainedCount}, rawCandidates={m_localPlaneCandidates.Count}, persistentCandidates={persistentVisible}, confirmedStructures={confirmedVisible}");
            }
        }

        private void RenderStablePoints()
        {
            if (m_pointMesh == null)
                return;

            Material material = materialOverride != null ? materialOverride : m_runtimeMaterial;
            if (material == null)
                return;

            float now = Time.unscaledTime;
            int stableCount = displayMode == DisplayMode.LocalPlaneCandidateCenters
                ? CountVisiblePersistentCandidates(now)
                : displayMode == DisplayMode.ConfirmedStructureCenters
                    ? CountVisibleConfirmedStructures(now)
                    : displayMode == DisplayMode.GridpointSurfaceMask
                        ? CountGridpointSurfaceMaskPoints(now)
                    : displayMode == DisplayMode.SurfaceLatticeWorldCells
                        ? CountSurfaceLatticeWorldCellFaces()
                    : displayMode == DisplayMode.WorldCellLinks
                        ? CountWorldCellLinks()
                        : CountStableWorldCells();

            if (stableCount <= 0)
                return;

            EnsureMatrixCapacity(stableCount);

            if ((displayMode == DisplayMode.StableWorldCells || displayMode == DisplayMode.SurfaceLatticeWorldCells || displayMode == DisplayMode.GridpointSurfaceMask) && highlightLowSupport)
            {
                int normalCount = displayMode == DisplayMode.GridpointSurfaceMask
                    ? BuildGridpointSurfaceMaskMatrices(now, false, m_batchMatrices)
                    : displayMode == DisplayMode.SurfaceLatticeWorldCells
                    ? BuildSurfaceLatticeWorldCellMatrices(now, false, m_batchMatrices)
                    : BuildStableWorldCellMatrices(now, false, m_batchMatrices);
                int lowSupportCount = displayMode == DisplayMode.GridpointSurfaceMask
                    ? BuildGridpointSurfaceMaskMatrices(now, true, m_secondaryBatchMatrices)
                    : displayMode == DisplayMode.SurfaceLatticeWorldCells
                    ? BuildSurfaceLatticeWorldCellMatrices(now, true, m_secondaryBatchMatrices)
                    : BuildStableWorldCellMatrices(now, true, m_secondaryBatchMatrices);
                DrawMatrixBatch(material, m_batchMatrices, normalCount, pointColor);
                DrawMatrixBatch(material, m_secondaryBatchMatrices, lowSupportCount, lowSupportWorldCellColor);
                return;
            }

            if (displayMode == DisplayMode.LocalPlaneCandidateCenters && highlightLowSupport)
            {
                int normalCount = BuildCandidateCenterMatrices(now, false, m_batchMatrices);
                int lowSupportCount = BuildCandidateCenterMatrices(now, true, m_secondaryBatchMatrices);
                DrawMatrixBatch(material, m_batchMatrices, normalCount, candidateCenterColor);
                DrawMatrixBatch(material, m_secondaryBatchMatrices, lowSupportCount, lowSupportCandidateColor);
                return;
            }

            int matrixCount = displayMode == DisplayMode.LocalPlaneCandidateCenters
                ? BuildCandidateCenterMatrices(now)
                : displayMode == DisplayMode.ConfirmedStructureCenters
                    ? BuildConfirmedStructureMatrices(now)
                    : displayMode == DisplayMode.GridpointSurfaceMask
                        ? BuildGridpointSurfaceMaskMatrices()
                    : displayMode == DisplayMode.SurfaceLatticeWorldCells
                        ? BuildSurfaceLatticeWorldCellMatrices()
                    : displayMode == DisplayMode.WorldCellLinks
                        ? BuildWorldCellLinkMatrices()
                    : BuildStableWorldCellMatrices();

            Color renderColor = displayMode == DisplayMode.LocalPlaneCandidateCenters
                ? candidateCenterColor
                : displayMode == DisplayMode.ConfirmedStructureCenters
                    ? confirmedCenterColor
                    : displayMode == DisplayMode.WorldCellLinks
                        ? worldCellLinkColor
                    : pointColor;
            DrawMatrixBatch(material, m_batchMatrices, matrixCount, renderColor);
        }

        private void EnsureMatrixCapacity(int count)
        {
            int required = Mathf.Max(1, Mathf.NextPowerOfTwo(count));
            if (m_batchMatrices.Length < required)
                m_batchMatrices = new Matrix4x4[required];
            if (m_secondaryBatchMatrices.Length < required)
                m_secondaryBatchMatrices = new Matrix4x4[required];
        }

        private void DrawMatrixBatch(Material material, Matrix4x4[] matrices, int matrixCount, Color renderColor)
        {
            if (matrixCount <= 0)
                return;

            m_propertyBlock.Clear();
            m_propertyBlock.SetColor("_BaseColor", renderColor);
            m_propertyBlock.SetColor("_Color", renderColor);

            const int batchSize = 1023;
            int drawn = 0;
            while (drawn < matrixCount)
            {
                int count = Mathf.Min(batchSize, matrixCount - drawn);
                Graphics.DrawMeshInstanced(
                    m_pointMesh,
                    0,
                    material,
                    matrices,
                    count,
                    m_propertyBlock,
                    ShadowCastingMode.Off,
                    false,
                    gameObject.layer,
                    sampleCamera);
                drawn += count;
            }
        }

        private float SampleLinearDepth(Vector2Int texCoord, int eyeIndex)
        {
            if (!m_depthPixels.IsCreated)
                return 0f;

            int clampedEye = Mathf.Clamp(eyeIndex, 0, NumEyes - 1);
            int index = texCoord.x + texCoord.y * CopyTextureSize + CopyTextureSize * CopyTextureSize * clampedEye;
            if (index < 0 || index >= m_depthPixels.Length)
                return 0f;
            return m_depthPixels[index];
        }

        private static Vector2Int ViewportToDepthCoord(float viewportX, float viewportY)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(viewportX) * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt((1f - Mathf.Clamp01(viewportY)) * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            return new Vector2Int(x, y);
        }

        private static bool TryReconstructWorld(Vector2Int texCoord, float linearDepth, Matrix4x4 inverseMatrix, out Vector3 worldPos)
        {
            worldPos = Vector3.zero;
            if (linearDepth <= 0f)
                return false;

            Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
            float clipDepth = zParams.x / linearDepth - zParams.y;
            float oneOverSize = 1f / CopyTextureSize;
            Vector4 clipPos = new(
                texCoord.x * oneOverSize * 2f - 1f,
                texCoord.y * oneOverSize * 2f - 1f,
                clipDepth,
                1f);

            Vector4 worldH = inverseMatrix * clipPos;
            if (Mathf.Abs(worldH.w) <= 1e-5f || !IsFinite(worldH))
                return false;

            Vector4 resolvedWorld = worldH / worldH.w;
            worldPos = new Vector3(resolvedWorld.x, resolvedWorld.y, resolvedWorld.z);
            return IsFinite(worldPos);
        }

        private Vector3 ReconstructNormal(Vector2Int texCoord, int eyeIndex, Matrix4x4 inverseMatrix)
        {
            float centerDepth = SampleLinearDepth(texCoord, eyeIndex);
            if (!TryReconstructWorld(texCoord, centerDepth, inverseMatrix, out Vector3 center))
                return Vector3.up;

            Vector2Int rightCoord = new(Mathf.Min(CopyTextureSize - 1, texCoord.x + 1), texCoord.y);
            Vector2Int upCoord = new(texCoord.x, Mathf.Min(CopyTextureSize - 1, texCoord.y + 1));
            if (!TryReconstructWorld(rightCoord, SampleLinearDepth(rightCoord, eyeIndex), inverseMatrix, out Vector3 right))
                return Vector3.up;
            if (!TryReconstructWorld(upCoord, SampleLinearDepth(upCoord, eyeIndex), inverseMatrix, out Vector3 up))
                return Vector3.up;

            Vector3 horizontal = right - center;
            Vector3 vertical = up - center;
            if (horizontal.sqrMagnitude <= 1e-6f || vertical.sqrMagnitude <= 1e-6f)
                return Vector3.up;

            Vector3 normal = -Vector3.Cross(horizontal, vertical).normalized;
            return normal.sqrMagnitude > 1e-5f ? normal : Vector3.up;
        }

        private float ComputeObservedMemoryQuality(int stableHitsValue, float confidenceValue, float displayDeltaMeters)
        {
            float stableScore = Mathf.Clamp01(stableHitsValue / Mathf.Max(1f, retainedMemoryMinStableHits));
            float confidenceScore = Mathf.Clamp01(confidenceValue);
            float deltaLimit = Mathf.Max(0.02f, Mathf.Max(worldCellReconnectDistanceMeters, displayUnlockDistanceMeters));
            float consistencyScore = 1f - Mathf.Clamp01(displayDeltaMeters / deltaLimit);
            return stableScore * 0.45f + confidenceScore * 0.35f + consistencyScore * 0.20f;
        }

        private void PruneWorldCellCooldowns(float now)
        {
            m_cleanupKeys.Clear();
            foreach (KeyValuePair<Vector3Int, float> pair in m_worldCellCooldownUntil)
            {
                if (now >= pair.Value)
                    m_cleanupKeys.Add(pair.Key);
            }

            for (int i = 0; i < m_cleanupKeys.Count; i++)
                m_worldCellCooldownUntil.Remove(m_cleanupKeys[i]);
            m_cleanupKeys.Clear();
        }

        private bool IsWorldCellCoolingDown(Vector3Int worldCellId, float now)
        {
            return m_worldCellCooldownUntil.TryGetValue(worldCellId, out float untilTime) && now < untilTime;
        }

        private void TryApplyRetainedMemoryObservation(ref CellState cell, Vector3 worldPos, Vector3 normal, float confidence, float now)
        {
            Vector3 observedNormal = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up;
            Vector3 existingNormal = cell.normal.sqrMagnitude > 1e-6f ? cell.normal.normalized : Vector3.up;
            float normalDot = Mathf.Abs(Vector3.Dot(existingNormal, observedNormal));
            float positionDeltaMeters = Vector3.Distance(cell.worldPos, worldPos);

            bool directAccept =
                positionDeltaMeters <= Mathf.Max(0.01f, retainedMemoryOverwriteDistanceMeters) &&
                normalDot >= Mathf.Clamp(retainedMemoryOverwriteNormalDot, 0f, 0.9999f);

            if (!directAccept)
            {
                bool supportsPending =
                    cell.pendingReconnectFrames > 0 &&
                    Vector3.Distance(cell.pendingReconnectWorldPos, worldPos) <= Mathf.Max(0.01f, retainedMemoryOverwriteDistanceMeters) &&
                    Mathf.Abs(Vector3.Dot(
                        cell.pendingReconnectNormal.sqrMagnitude > 1e-6f ? cell.pendingReconnectNormal.normalized : Vector3.up,
                        observedNormal)) >= Mathf.Clamp(retainedMemoryOverwriteNormalDot, 0f, 0.9999f);

                if (!supportsPending)
                {
                    cell.pendingReconnectFrames = 1;
                    cell.pendingReconnectWorldPos = worldPos;
                    cell.pendingReconnectNormal = observedNormal;
                    return;
                }

                cell.pendingReconnectFrames = Mathf.Min(999, cell.pendingReconnectFrames + 1);
                cell.pendingReconnectWorldPos = Vector3.Lerp(cell.pendingReconnectWorldPos, worldPos, 0.35f);
                Vector3 blendedPendingNormal = Vector3.Lerp(cell.pendingReconnectNormal, observedNormal, 0.35f);
                cell.pendingReconnectNormal = blendedPendingNormal.sqrMagnitude > 1e-6f ? blendedPendingNormal.normalized : Vector3.up;

                if (cell.pendingReconnectFrames < Mathf.Max(1, retainedMemoryReconnectSupportFrames))
                    return;

                worldPos = cell.pendingReconnectWorldPos;
                observedNormal = cell.pendingReconnectNormal;
            }

            float blend = positionDeltaMeters <= Mathf.Max(0.001f, worldCellMergeDistanceMeters)
                ? Mathf.Clamp01(positionBlend * 0.7f)
                : 0.08f;

            cell.stableHits = Mathf.Min(999, cell.stableHits + 1);
            cell.lastSeenTime = now;
            cell.worldPos = Vector3.Lerp(cell.worldPos, worldPos, blend);

            Vector3 blendedNormal = Vector3.Lerp(cell.normal, observedNormal, Mathf.Clamp01(normalBlend * 0.8f));
            cell.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
            cell.confidence = Mathf.Lerp(cell.confidence, confidence, 0.12f);
            cell.pendingReconnectFrames = 0;
            cell.pendingReconnectWorldPos = cell.worldPos;
            cell.pendingReconnectNormal = cell.normal;
        }

        private bool ShouldRetainCellMemory(CellState cell)
        {
            if (!cell.valid)
                return false;
            if (cell.stableHits < Mathf.Max(minStableHits, retainedMemoryMinStableHits))
                return false;
            if (cell.confidence < retainedMemoryMinConfidence)
                return false;
            return cell.memoryQualityScore >= retainedMemoryMinQualityScore;
        }

        private bool ShouldInvalidateRetainedMemory(CellState cell, float now)
        {
            if (!cell.retainedMemory)
                return false;

            float unseenSeconds = Mathf.Max(0f, now - cell.lastSeenTime);
            if (unseenSeconds > Mathf.Max(stableCellHoldSeconds, retainedMemoryMaxUnseenSeconds))
                return true;

            if (cell.memoryQualityScore < retainedMemoryMinQualityScore * 0.75f &&
                unseenSeconds > Mathf.Max(holdSeconds, stableCellHoldSeconds))
            {
                return true;
            }

            return false;
        }

        private bool IsCellActiveSupport(CellState cell, float now)
        {
            if (!cell.valid || cell.stableHits < minStableHits)
                return false;

            float unseenSeconds = Mathf.Max(0f, now - cell.lastSeenTime);
            if (!cell.retainedMemory)
                return unseenSeconds <= Mathf.Max(0.05f, holdSeconds);

            if (cell.memoryQualityScore < retainedMemoryMinQualityScore)
                return false;

            return unseenSeconds <= Mathf.Max(0.05f, activeRetainedMaxUnseenSeconds);
        }

        private bool IsCellCandidateSupport(CellState cell, float now)
        {
            if (!cell.valid || cell.stableHits < minStableHits)
                return false;

            if (IsCellActiveSupport(cell, now))
                return true;

            if (!cell.retainedMemory)
                return false;
            if (cell.memoryQualityScore < retainedMemoryMinQualityScore)
                return false;

            float unseenSeconds = Mathf.Max(0f, now - cell.lastSeenTime);
            return unseenSeconds <= Mathf.Max(activeRetainedMaxUnseenSeconds, warmStaleCandidateMaxUnseenSeconds);
        }

        private string GetCellSupportState(CellState cell, float now)
        {
            if (!cell.valid)
                return "Invalid";
            if (!cell.retainedMemory)
                return IsCellActiveSupport(cell, now) ? "Recent" : "Transient";
            if (IsCellActiveSupport(cell, now))
                return "Active";
            return IsCellCandidateSupport(cell, now) ? "WarmStale" : "Stale";
        }

        private bool IsLowSupportCell(CellState cell, float now)
        {
            if (!cell.valid)
                return false;

            int signals = 0;
            if (cell.confidence <= lowSupportCellMaxConfidence)
                signals++;
            if (cell.memoryQualityScore > 0f && cell.memoryQualityScore <= lowSupportCellMaxQualityScore)
                signals++;
            if (cell.stableHits <= Mathf.Max(minStableHits, lowSupportCellMaxStableHits))
                signals++;

            float unseenSeconds = Mathf.Max(0f, now - cell.lastSeenTime);
            if (unseenSeconds >= lowSupportCellMinUnseenSeconds)
                signals++;

            string supportState = GetCellSupportState(cell, now);
            if (supportState == "WarmStale" || supportState == "Stale" || supportState == "Transient")
                signals++;

            if (cell.pendingReconnectFrames > 0)
                signals++;

            return signals >= 2;
        }

        private bool IsLowSupportCandidate(PersistentCandidateState candidate, float now)
        {
            int signals = 0;
            if (candidate.memberCount <= lowSupportCandidateMaxMemberCount)
                signals++;
            if (candidate.fitResidualMeters >= lowSupportCandidateMinResidualMeters)
                signals++;
            if (candidate.axisAlignmentScore <= lowSupportCandidateMaxAxisAlignmentScore)
                signals++;

            float deltaMeters = Vector3.Distance(candidate.displayCenter, candidate.observedCenter);
            if (deltaMeters >= lowSupportCandidateMinDeltaMeters)
                signals++;

            float unseenSeconds = Mathf.Max(0f, now - candidate.lastSeenTime);
            if (unseenSeconds >= lowSupportCandidateMinUnseenSeconds)
                signals++;

            if (candidate.displayLockState == CellState.DisplayLockState.Relocating)
                signals++;

            return signals >= 2;
        }

        private Vector3Int ResolveObservedWorldCellId(Vector3 worldPos, Vector3 normal, float now)
        {
            Vector3Int quantizedId = WorldToCellId(worldPos);
            if (m_worldCells.ContainsKey(quantizedId))
                return quantizedId;

            float reconnectDistance = Mathf.Max(worldCellSizeMeters * 0.5f, worldCellReconnectDistanceMeters);
            float reconnectDistanceSqr = reconnectDistance * reconnectDistance;
            float minNormalDot = Mathf.Clamp(worldCellReconnectNormalDot, 0f, 0.9999f);
            Vector3 observedNormal = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up;
            int neighborRange = Mathf.Max(1, Mathf.CeilToInt(reconnectDistance / Mathf.Max(0.01f, worldCellSizeMeters)));

            float bestScore = float.MaxValue;
            Vector3Int bestId = quantizedId;
            bool found = false;

            for (int dz = -neighborRange; dz <= neighborRange; dz++)
            {
                for (int dy = -neighborRange; dy <= neighborRange; dy++)
                {
                    for (int dx = -neighborRange; dx <= neighborRange; dx++)
                    {
                        Vector3Int candidateId = new(quantizedId.x + dx, quantizedId.y + dy, quantizedId.z + dz);
                        if (!m_worldCells.TryGetValue(candidateId, out CellState existing) || !existing.valid)
                            continue;

                        float age = Mathf.Max(0f, now - existing.lastSeenTime);
                        float maxReconnectAge = existing.retainedMemory
                            ? Mathf.Max(stableCellHoldSeconds, retainedMemoryMaxUnseenSeconds)
                            : (existing.stableHits >= minStableHits
                                ? Mathf.Max(holdSeconds, stableCellHoldSeconds)
                                : holdSeconds);
                        if (age > maxReconnectAge)
                            continue;

                        float candidateReconnectDistance = existing.retainedMemory
                            ? Mathf.Max(worldCellReconnectDistanceMeters, retainedMemoryReconnectDistanceMeters)
                            : Mathf.Max(worldCellSizeMeters * 0.5f, worldCellReconnectDistanceMeters);
                        float maxDistanceSqr = candidateReconnectDistance * candidateReconnectDistance;
                        float distanceSqr = (existing.worldPos - worldPos).sqrMagnitude;
                        if (distanceSqr > maxDistanceSqr)
                            continue;

                        Vector3 existingNormal = existing.normal.sqrMagnitude > 1e-6f ? existing.normal.normalized : Vector3.up;
                        float normalDot = Mathf.Abs(Vector3.Dot(existingNormal, observedNormal));
                        float minReconnectNormalDot = existing.retainedMemory
                            ? Mathf.Clamp(retainedMemoryReconnectNormalDot, 0f, 0.9999f)
                            : minNormalDot;
                        if (normalDot < minReconnectNormalDot)
                            continue;

                        if (existing.retainedMemory)
                        {
                            if (existing.memoryQualityScore < retainedMemoryMinQualityScore)
                                continue;
                            if (existing.confidence < retainedMemoryMinConfidence)
                                continue;
                            if (existing.stableHits < Mathf.Max(minStableHits, retainedMemoryMinStableHits))
                                continue;
                        }

                        float distanceScore = Mathf.Sqrt(distanceSqr) / Mathf.Max(0.001f, candidateReconnectDistance);
                        float normalScore = 1f - normalDot;
                        float ageScore = age / Mathf.Max(0.001f, maxReconnectAge);
                        float qualityScore = existing.retainedMemory ? (1f - existing.memoryQualityScore) * 0.5f : 0f;
                        float score = distanceScore + normalScore * 0.75f + ageScore * 0.35f + qualityScore;

                        if (score >= bestScore)
                            continue;

                        bestScore = score;
                        bestId = candidateId;
                        found = true;
                    }
                }
            }

            if (found)
                return bestId;

            if (IsWorldCellCoolingDown(quantizedId, now))
                return InvalidWorldCellId;

            return quantizedId;
        }

        private int CountStableWorldCells()
        {
            int stableCount = 0;
            foreach (CellState cell in m_worldCells.Values)
            {
                if (cell.valid && cell.stableHits >= minStableHits)
                    stableCount++;
            }

            return stableCount;
        }

        private int CountGridpointSurfaceMaskPoints(float now)
        {
            HashSet<SurfaceGridpointKey> normalKeys = new();
            HashSet<SurfaceGridpointKey> lowSupportKeys = new();
            CollectGridpointSurfaceMaskKeys(now, normalKeys, lowSupportKeys);
            return normalKeys.Count + lowSupportKeys.Count;
        }

        private int BuildStableWorldCellMatrices()
        {
            return BuildStableWorldCellMatrices(Time.unscaledTime, false, m_batchMatrices);
        }

        private int BuildStableWorldCellMatrices(float now, bool lowSupportOnly, Matrix4x4[] targetMatrices)
        {
            int matrixCount = 0;
            foreach (CellState cell in m_worldCells.Values)
            {
                if (!cell.valid || cell.stableHits < minStableHits)
                    continue;

                bool isLowSupport = IsLowSupportCell(cell, now);
                if (lowSupportOnly != isLowSupport)
                    continue;

                Vector3 renderNormal = cell.displayNormal.sqrMagnitude > 1e-6f ? cell.displayNormal.normalized : Vector3.up;
                Vector3 position = cell.displayWorldPos + renderNormal * surfaceOffsetMeters;
                Vector3 scale = Vector3.one * Mathf.Max(0.003f, pointScaleMeters);
                targetMatrices[matrixCount++] = Matrix4x4.TRS(position, Quaternion.identity, scale);
            }

            return matrixCount;
        }

        private int BuildGridpointSurfaceMaskMatrices()
        {
            return BuildGridpointSurfaceMaskMatrices(Time.unscaledTime, false, m_batchMatrices);
        }

        private int BuildGridpointSurfaceMaskMatrices(float now, bool lowSupportOnly, Matrix4x4[] targetMatrices)
        {
            HashSet<SurfaceGridpointKey> normalKeys = new();
            HashSet<SurfaceGridpointKey> lowSupportKeys = new();
            CollectGridpointSurfaceMaskKeys(now, normalKeys, lowSupportKeys);

            HashSet<SurfaceGridpointKey> source = lowSupportOnly ? lowSupportKeys : normalKeys;
            int matrixCount = 0;
            float pointScale = Mathf.Max(0.003f, Mathf.Max(worldCellSizeMeters, gridpointSurfaceSpacingMeters) * Mathf.Clamp01(gridpointSurfacePointScaleRatio));
            foreach (SurfaceGridpointKey key in source)
            {
                Vector3 position = GridpointSurfaceKeyToWorldPosition(key);
                targetMatrices[matrixCount++] = Matrix4x4.TRS(position, Quaternion.identity, Vector3.one * pointScale);
            }

            return matrixCount;
        }

        private void CollectGridpointSurfaceMaskKeys(float now, HashSet<SurfaceGridpointKey> normalKeys, HashSet<SurfaceGridpointKey> lowSupportKeys)
        {
            normalKeys.Clear();
            lowSupportKeys.Clear();

            foreach (CellState cell in m_worldCells.Values)
            {
                if (!cell.valid || cell.stableHits < minStableHits)
                    continue;
                if (cell.displayNormal.sqrMagnitude <= 1e-6f)
                    continue;

                SurfaceGridpointKey key = WorldCellToSurfaceGridpointKey(cell);
                bool isLowSupport = IsLowSupportCell(cell, now);
                if (!isLowSupport)
                {
                    normalKeys.Add(key);
                    lowSupportKeys.Remove(key);
                }
                else if (!normalKeys.Contains(key))
                {
                    lowSupportKeys.Add(key);
                }
            }
        }

        private static readonly Vector3Int[] SurfaceFaceDirections =
        {
            new(1, 0, 0),
            new(-1, 0, 0),
            new(0, 1, 0),
            new(0, -1, 0),
            new(0, 0, 1),
            new(0, 0, -1),
        };

        private int CountSurfaceLatticeWorldCellFaces()
        {
            int count = 0;
            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                if (!IsRenderableSurfaceLatticeCell(pair.Value))
                    continue;

                Vector3Int cellId = pair.Key;
                for (int i = 0; i < SurfaceFaceDirections.Length; i++)
                {
                    Vector3Int neighborId = cellId + SurfaceFaceDirections[i];
                    if (!m_worldCells.TryGetValue(neighborId, out CellState neighbor) || !IsRenderableSurfaceLatticeCell(neighbor))
                        count++;
                }
            }

            return count;
        }

        private int BuildSurfaceLatticeWorldCellMatrices()
        {
            return BuildSurfaceLatticeWorldCellMatrices(Time.unscaledTime, false, m_batchMatrices);
        }

        private int BuildSurfaceLatticeWorldCellMatrices(float now, bool lowSupportOnly, Matrix4x4[] targetMatrices)
        {
            int matrixCount = 0;
            float cellSize = Mathf.Max(0.01f, worldCellSizeMeters);
            float inPlaneScale = Mathf.Max(0.003f, cellSize * Mathf.Clamp01(surfaceLatticePointScaleRatio));
            float thickness = Mathf.Max(0.0015f, surfaceLatticeThicknessMeters);

            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                CellState cell = pair.Value;
                if (!IsRenderableSurfaceLatticeCell(cell))
                    continue;

                bool isLowSupport = IsLowSupportCell(cell, now);
                if (lowSupportOnly != isLowSupport)
                    continue;

                Vector3 center = CellIdToWorldCenter(pair.Key);
                for (int i = 0; i < SurfaceFaceDirections.Length; i++)
                {
                    Vector3Int directionId = SurfaceFaceDirections[i];
                    Vector3Int neighborId = pair.Key + directionId;
                    if (m_worldCells.TryGetValue(neighborId, out CellState neighbor) && IsRenderableSurfaceLatticeCell(neighbor))
                        continue;

                    Vector3 normal = new(directionId.x, directionId.y, directionId.z);
                    Vector3 position = center + normal * (cellSize * 0.5f + thickness * 0.5f + surfaceOffsetMeters);
                    Vector3 scale = directionId.x != 0
                        ? new Vector3(thickness, inPlaneScale, inPlaneScale)
                        : directionId.y != 0
                            ? new Vector3(inPlaneScale, thickness, inPlaneScale)
                            : new Vector3(inPlaneScale, inPlaneScale, thickness);
                    targetMatrices[matrixCount++] = Matrix4x4.TRS(position, Quaternion.identity, scale);
                }
            }

            return matrixCount;
        }

        private int CountWorldCellLinks()
        {
            int count = 0;
            float maxDistance = Mathf.Max(worldCellSizeMeters * 1.8f, worldCellLinkMaxDistanceMeters);
            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                if (!IsRenderableWorldCell(pair.Value))
                    continue;

                Vector3Int cellId = pair.Key;
                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0 && dz == 0)
                                continue;
                            if (!ShouldUseForwardNeighbor(dx, dy, dz))
                                continue;

                            Vector3Int neighborId = new(cellId.x + dx, cellId.y + dy, cellId.z + dz);
                            if (!m_worldCells.TryGetValue(neighborId, out CellState neighbor) || !IsRenderableWorldCell(neighbor))
                                continue;

                            if (Vector3.Distance(pair.Value.displayWorldPos, neighbor.displayWorldPos) <= maxDistance)
                                count++;
                        }
                    }
                }
            }

            return count;
        }

        private int BuildWorldCellLinkMatrices()
        {
            int matrixCount = 0;
            float maxDistance = Mathf.Max(worldCellSizeMeters * 1.8f, worldCellLinkMaxDistanceMeters);
            float thickness = Mathf.Max(0.002f, worldCellLinkThicknessMeters);

            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                if (!IsRenderableWorldCell(pair.Value))
                    continue;

                Vector3Int cellId = pair.Key;
                Vector3 from = pair.Value.displayWorldPos;
                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0 && dz == 0)
                                continue;
                            if (!ShouldUseForwardNeighbor(dx, dy, dz))
                                continue;

                            Vector3Int neighborId = new(cellId.x + dx, cellId.y + dy, cellId.z + dz);
                            if (!m_worldCells.TryGetValue(neighborId, out CellState neighbor) || !IsRenderableWorldCell(neighbor))
                                continue;

                            Vector3 to = neighbor.displayWorldPos;
                            Vector3 delta = to - from;
                            float distance = delta.magnitude;
                            if (distance <= 1e-4f || distance > maxDistance)
                                continue;

                            Quaternion rotation = Quaternion.FromToRotation(Vector3.up, delta / distance);
                            Vector3 midpoint = (from + to) * 0.5f;
                            Vector3 scale = new(thickness, distance, thickness);
                            m_batchMatrices[matrixCount++] = Matrix4x4.TRS(midpoint, rotation, scale);
                        }
                    }
                }
            }

            return matrixCount;
        }

        private int BuildCandidateCenterMatrices(float now)
        {
            return BuildCandidateCenterMatrices(now, false, m_batchMatrices);
        }

        private int BuildCandidateCenterMatrices(float now, bool lowSupportOnly, Matrix4x4[] targetMatrices)
        {
            int matrixCount = 0;
            float scaleMeters = Mathf.Max(0.004f, candidateCenterScaleMeters);
            foreach (PersistentCandidateState candidate in m_persistentCandidates.Values)
            {
                if (!IsCandidateVisible(candidate, now))
                    continue;

                bool isLowSupport = IsLowSupportCandidate(candidate, now);
                if (lowSupportOnly != isLowSupport)
                    continue;

                Vector3 renderNormal = candidate.displayNormal.sqrMagnitude > 1e-6f ? candidate.displayNormal.normalized : Vector3.up;
                Vector3 position = candidate.displayCenter + renderNormal * surfaceOffsetMeters;
                targetMatrices[matrixCount++] = Matrix4x4.TRS(position, Quaternion.identity, Vector3.one * scaleMeters);
            }

            return matrixCount;
        }

        private bool IsRenderableWorldCell(CellState cell)
        {
            return cell.valid;
        }

        private bool IsRenderableSurfaceLatticeCell(CellState cell)
        {
            return cell.valid && cell.stableHits >= minStableHits;
        }

        private static bool ShouldUseForwardNeighbor(int dx, int dy, int dz)
        {
            if (dz > 0)
                return true;
            if (dz < 0)
                return false;
            if (dy > 0)
                return true;
            if (dy < 0)
                return false;
            return dx > 0;
        }

        private int BuildConfirmedStructureMatrices(float now)
        {
            int matrixCount = 0;
            float scaleMeters = Mathf.Max(0.004f, confirmedCenterScaleMeters);
            CollectVisibleConfirmedStructureIds(now, false);
            for (int i = 0; i < m_visibleConfirmedIds.Count; i++)
            {
                if (!m_confirmedStructures.TryGetValue(m_visibleConfirmedIds[i], out ConfirmedStructureState structure))
                    continue;

                Vector3 renderNormal = structure.displayNormal.sqrMagnitude > 1e-6f ? structure.displayNormal.normalized : Vector3.up;
                Vector3 position = structure.displayCenter + renderNormal * surfaceOffsetMeters;
                m_batchMatrices[matrixCount++] = Matrix4x4.TRS(position, Quaternion.identity, Vector3.one * scaleMeters);
            }

            return matrixCount;
        }

        private int CountVisiblePersistentCandidates(float now)
        {
            int count = 0;
            foreach (PersistentCandidateState candidate in m_persistentCandidates.Values)
            {
                if (IsCandidateVisible(candidate, now))
                    count++;
            }

            return count;
        }

        private int CountVisibleConfirmedStructures(float now)
        {
            CollectVisibleConfirmedStructureIds(now, false);
            return m_visibleConfirmedIds.Count;
        }

        private int CountConfirmedStructuresInViewport()
        {
            CollectVisibleConfirmedStructureIds(Time.unscaledTime, true);
            return m_visibleConfirmedIds.Count;
        }

        private bool IsWorldPointInViewport(Vector3 worldPoint, float padding = 0f)
        {
            if (sampleCamera == null)
                return false;

            Vector3 viewport = sampleCamera.WorldToViewportPoint(worldPoint);
            if (viewport.z <= 0f)
                return false;

            float min = 0f - Mathf.Max(0f, padding);
            float max = 1f + Mathf.Max(0f, padding);
            return viewport.x >= min && viewport.x <= max && viewport.y >= min && viewport.y <= max;
        }

        private void UpdateResponseMonitor(float now)
        {
            if (sampleCamera == null)
                return;

            Transform cameraTransform = sampleCamera.transform;
            Vector3 currentPosition = cameraTransform.position;
            Quaternion currentRotation = cameraTransform.rotation;

            if (!m_hasResponseReferencePose)
            {
                m_hasResponseReferencePose = true;
                m_responseReferenceCameraPosition = currentPosition;
                m_responseReferenceCameraRotation = currentRotation;
                BeginResponseSession(now);
            }

            float positionDelta = Vector3.Distance(currentPosition, m_responseReferenceCameraPosition);
            float angularDelta = Quaternion.Angle(currentRotation, m_responseReferenceCameraRotation);
            bool canRetrigger = now - m_responseLastTriggerTime >= Mathf.Max(0.1f, responseRetriggerCooldownSeconds);
            if (canRetrigger && (positionDelta >= Mathf.Max(0.001f, responseTriggerPositionDeltaMeters) ||
                                 angularDelta >= Mathf.Max(0.1f, responseTriggerAngularDeltaDegrees)))
            {
                m_responseReferenceCameraPosition = currentPosition;
                m_responseReferenceCameraRotation = currentRotation;
                BeginResponseSession(now);
            }

            int viewportConfirmed = CountConfirmedStructuresInViewport();
            m_responseCurrentViewportConfirmedCount = viewportConfirmed;
            if (!m_responseSessionActive)
                return;

            if (viewportConfirmed > m_responsePeakViewportConfirmedCount)
            {
                m_responsePeakViewportConfirmedCount = viewportConfirmed;
                m_responsePeakConfirmedLatency = now - m_responseSessionStartTime;
            }

            if (m_responseFirstConfirmedLatency < 0f &&
                viewportConfirmed >= Mathf.Max(1, responseMinViewportConfirmedCount))
            {
                m_responseFirstConfirmedLatency = now - m_responseSessionStartTime;
            }
        }

        private void BeginResponseSession(float now)
        {
            m_responseSessionActive = true;
            m_responseSessionStartTime = now;
            m_responseLastTriggerTime = now;
            m_responseFirstConfirmedLatency = -1f;
            m_responsePeakConfirmedLatency = -1f;
            m_responsePeakViewportConfirmedCount = 0;
            m_responseCurrentViewportConfirmedCount = 0;
        }

        private bool IsCandidateVisible(PersistentCandidateState candidate, float now)
        {
            if (candidate.stableFrames < Mathf.Max(1, minPersistentCandidateFrames))
                return false;

            return now - candidate.lastSeenTime <= Mathf.Max(0.05f, visibleCandidateGraceSeconds);
        }

        private static bool IsConfirmedStructureVisible(ConfirmedStructureState structure)
        {
            return structure.status == ConfirmedStructureStatus.Confirmed;
        }

        private bool ShouldRenderConfirmedStructure(ConfirmedStructureState structure, float now)
        {
            if (!IsConfirmedStructureVisible(structure))
                return false;

            bool inCenter = IsWorldPointInViewportCenter(structure.displayCenter, centerDisplayPriorityViewportRadius);
            bool inViewport = IsWorldPointInViewport(structure.displayCenter, confirmedDisplayViewportPadding);
            if (!IsDisplayQualityConfirmedStructure(structure, now, inCenter || inViewport))
                return false;

            if (inCenter)
                return true;

            if (inViewport)
            {
                if (HasRecentCenterSupport(structure, now))
                    return now - structure.lastObservedTime <= Mathf.Max(0.1f, centerConfirmedViewportDisplaySeconds);
                return now - structure.lastObservedTime <= Mathf.Max(0.1f, confirmedPeripheralRecentDisplaySeconds);
            }

            return now - structure.lastObservedTime <= Mathf.Max(0.1f, confirmedRecentDisplaySeconds);
        }

        private void CollectVisibleConfirmedStructureIds(float now, bool viewportOnly)
        {
            m_visibleConfirmedIds.Clear();
            m_visibleConfirmedCenterIds.Clear();
            m_visibleConfirmedViewportIds.Clear();
            m_visibleConfirmedRecentIds.Clear();

            foreach (KeyValuePair<int, ConfirmedStructureState> pair in m_confirmedStructures)
            {
                ConfirmedStructureState structure = pair.Value;
                if (!ShouldRenderConfirmedStructure(structure, now))
                    continue;

                bool inCenter = IsWorldPointInViewportCenter(structure.displayCenter, centerDisplayPriorityViewportRadius);
                bool inViewport = IsWorldPointInViewport(structure.displayCenter, confirmedDisplayViewportPadding);
                if (viewportOnly && !inViewport)
                    continue;

                if (inCenter)
                {
                    m_visibleConfirmedCenterIds.Add(pair.Key);
                }
                else if (inViewport)
                {
                    m_visibleConfirmedViewportIds.Add(pair.Key);
                }
                else
                {
                    m_visibleConfirmedRecentIds.Add(pair.Key);
                }
            }

            m_visibleConfirmedCenterIds.Sort((a, b) => CompareConfirmedVisibilityPriority(a, b, now));
            m_visibleConfirmedViewportIds.Sort((a, b) => CompareConfirmedVisibilityPriority(a, b, now));
            m_visibleConfirmedRecentIds.Sort((a, b) => CompareConfirmedVisibilityPriority(a, b, now));

            int maxTotal = Mathf.Max(1, confirmedMaxVisibleStructures);
            int centerBudget = Mathf.Clamp(confirmedCenterVisibleBudget, 0, maxTotal);
            int viewportBudget = Mathf.Clamp(confirmedViewportVisibleBudget, 0, maxTotal);
            int recentBudget = Mathf.Clamp(confirmedRecentVisibleBudget, 0, maxTotal);

            AppendVisibleConfirmedIds(m_visibleConfirmedCenterIds, centerBudget, maxTotal);
            AppendVisibleConfirmedIds(m_visibleConfirmedViewportIds, viewportBudget, maxTotal);
            AppendVisibleConfirmedIds(m_visibleConfirmedRecentIds, recentBudget, maxTotal);

            if (m_visibleConfirmedIds.Count >= maxTotal)
                return;

            AppendVisibleConfirmedIds(m_visibleConfirmedCenterIds, maxTotal, maxTotal);
            AppendVisibleConfirmedIds(m_visibleConfirmedViewportIds, maxTotal, maxTotal);
            AppendVisibleConfirmedIds(m_visibleConfirmedRecentIds, maxTotal, maxTotal);
        }

        private void AppendVisibleConfirmedIds(List<int> source, int limit, int maxTotal)
        {
            int added = 0;
            for (int i = 0; i < source.Count; i++)
            {
                if (m_visibleConfirmedIds.Count >= maxTotal || added >= limit)
                    break;

                int id = source[i];
                if (m_visibleConfirmedIds.Contains(id))
                    continue;

                m_visibleConfirmedIds.Add(id);
                added++;
            }
        }

        private int CompareConfirmedVisibilityPriority(int leftId, int rightId, float now)
        {
            if (!m_confirmedStructures.TryGetValue(leftId, out ConfirmedStructureState left))
                return 1;
            if (!m_confirmedStructures.TryGetValue(rightId, out ConfirmedStructureState right))
                return -1;

            float leftScore = ScoreConfirmedVisibility(left, now);
            float rightScore = ScoreConfirmedVisibility(right, now);
            return rightScore.CompareTo(leftScore);
        }

        private float ScoreConfirmedVisibility(ConfirmedStructureState structure, float now)
        {
            float recency = 1f - Mathf.Clamp01((now - structure.lastObservedTime) / Mathf.Max(0.1f, confirmedRecentDisplaySeconds));
            float centerRecency = HasRecentCenterSupport(structure, now)
                ? 1f - Mathf.Clamp01((now - structure.lastCenterObservedTime) / Mathf.Max(0.1f, centerConfirmedRecentSupportSeconds))
                : 0f;
            float centerCarry = Mathf.Clamp01(structure.centerSupportScore / 12f);
            float support = Mathf.Clamp01(structure.supportScore / Mathf.Max(1f, confirmedDisplayMinSupportScore));
            float residual = 1f - Mathf.Clamp01(structure.fitResidualMeters / Mathf.Max(0.001f, confirmedDisplayMaxResidualMeters));
            float align = Mathf.Clamp01(structure.axisAlignmentScore);

            return centerRecency * 3.0f
                   + centerCarry * 2.0f
                   + recency * 1.6f
                   + support * 1.1f
                   + residual * 0.8f
                   + align * 0.7f;
        }

        private bool IsDisplayQualityConfirmedStructure(ConfirmedStructureState structure, float now, bool inPriorityViewport)
        {
            if (structure.supportScore >= confirmedDisplayMinSupportScore &&
                structure.fitResidualMeters <= confirmedDisplayMaxResidualMeters &&
                structure.axisAlignmentScore >= confirmedDisplayMinAxisAlignmentScore)
            {
                return true;
            }

            if (!inPriorityViewport)
                return false;

            return now - structure.lastObservedTime <= Mathf.Max(0.1f, confirmedDisplayLowQualityGraceSeconds);
        }

        private bool IsWorldPointInViewportCenter(Vector3 worldPoint, float radius)
        {
            if (sampleCamera == null)
                return false;

            Vector3 viewport = sampleCamera.WorldToViewportPoint(worldPoint);
            if (viewport.z <= 0f)
                return false;

            float dx = viewport.x - 0.5f;
            float dy = viewport.y - 0.5f;
            radius = Mathf.Max(0.02f, radius);
            return dx * dx + dy * dy <= radius * radius;
        }

        private bool HasRecentCenterSupport(ConfirmedStructureState structure, float now)
        {
            return now - structure.lastCenterObservedTime <= Mathf.Max(0.1f, centerConfirmedRecentSupportSeconds) &&
                   structure.centerSupportScore > 0.1f;
        }

        private void RebuildLocalPlaneCandidates(float now)
        {
            m_localPlaneCandidates.Clear();
            m_candidateVisited.Clear();
            m_candidateQueue.Clear();

            int nextCandidateId = 1;
            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                Vector3Int seedId = pair.Key;
                CellState seed = pair.Value;
                if (!CanSeedCandidate(seed, now) || m_candidateVisited.Contains(seedId))
                    continue;

                if (TryBuildLocalPlaneCandidate(seedId, now, ref nextCandidateId, out LocalPlaneCandidate candidate))
                    m_localPlaneCandidates.Add(candidate);
            }
        }

        private bool CanSeedCandidate(CellState cell, float now)
        {
            if (!cell.valid || cell.stableHits < Mathf.Max(1, minCandidateStableHits))
                return false;
            if (cell.confidence < Mathf.Max(0.05f, confidenceFloor * 0.25f))
                return false;
            if (!IsCellCandidateSupport(cell, now))
                return false;
            return cell.displayNormal.sqrMagnitude > 1e-6f;
        }

        private bool TryBuildLocalPlaneCandidate(Vector3Int seedId, float now, ref int nextCandidateId, out LocalPlaneCandidate candidate)
        {
            candidate = default;
            if (!m_worldCells.TryGetValue(seedId, out CellState seed) || !CanSeedCandidate(seed, now))
                return false;

            m_candidateMembers.Clear();
            m_candidateQueue.Enqueue(seedId);
            m_candidateVisited.Add(seedId);

            float maxNeighborDistance = Mathf.Max(worldCellSizeMeters * 1.5f, maxCandidateNeighborOffsetMeters);
            float minNormalDot = Mathf.Clamp(minCandidateNormalDot, 0.0f, 0.9999f);
            Vector3 seedNormal = seed.displayNormal.sqrMagnitude > 1e-6f ? seed.displayNormal.normalized : Vector3.up;
            Vector3 seedDominantAxis = ResolveDominantAxis(seedNormal);
            int seedAxisIndex = ResolveAxisIndex(seedDominantAxis);

            while (m_candidateQueue.Count > 0)
            {
                Vector3Int cellId = m_candidateQueue.Dequeue();
                if (!m_worldCells.TryGetValue(cellId, out CellState cell) || !CanSeedCandidate(cell, now))
                    continue;

                m_candidateMembers.Add(cell);

                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0 && dz == 0)
                                continue;

                            Vector3Int neighborId = new(cellId.x + dx, cellId.y + dy, cellId.z + dz);

                            if (m_candidateVisited.Contains(neighborId))
                                continue;
                            if (!m_worldCells.TryGetValue(neighborId, out CellState neighbor) || !CanSeedCandidate(neighbor, now))
                                continue;

                            Vector3 neighborNormal = neighbor.displayNormal.sqrMagnitude > 1e-6f ? neighbor.displayNormal.normalized : Vector3.up;
                            Vector3 neighborDominantAxis = ResolveDominantAxis(neighborNormal);
                            int neighborAxisIndex = ResolveAxisIndex(neighborDominantAxis);
                            if (neighborAxisIndex != seedAxisIndex)
                                continue;

                            float normalDot = Mathf.Abs(Vector3.Dot(seedNormal, neighborNormal));
                            if (normalDot < minNormalDot)
                                continue;

                            float worldDistance = Vector3.Distance(cell.displayWorldPos, neighbor.displayWorldPos);
                            if (worldDistance > maxNeighborDistance)
                                continue;

                            m_candidateVisited.Add(neighborId);
                            m_candidateQueue.Enqueue(neighborId);
                        }
                    }
                }
            }

            if (m_candidateMembers.Count < Mathf.Max(2, minCandidateCells))
                return false;

            return TryFinalizeLocalPlaneCandidate(ref nextCandidateId, out candidate);
        }

        private bool TryFinalizeLocalPlaneCandidate(ref int nextCandidateId, out LocalPlaneCandidate candidate)
        {
            candidate = default;
            if (m_candidateMembers.Count <= 0)
                return false;

            Vector3 center = Vector3.zero;
            Vector3 summedNormal = Vector3.zero;
            for (int i = 0; i < m_candidateMembers.Count; i++)
            {
                CellState member = m_candidateMembers[i];
                center += member.displayWorldPos;
                summedNormal += member.displayNormal.sqrMagnitude > 1e-6f ? member.displayNormal.normalized : Vector3.up;
            }

            center /= m_candidateMembers.Count;
            Vector3 averagedNormal = summedNormal.sqrMagnitude > 1e-6f ? summedNormal.normalized : Vector3.up;

            float residualSum = 0f;
            for (int i = 0; i < m_candidateMembers.Count; i++)
            {
                Vector3 offset = m_candidateMembers[i].displayWorldPos - center;
                residualSum += Mathf.Abs(Vector3.Dot(offset, averagedNormal));
            }

            float residual = residualSum / Mathf.Max(1, m_candidateMembers.Count);
            float axisAlignment = Mathf.Max(
                Mathf.Abs(Vector3.Dot(averagedNormal, Vector3.up)),
                Mathf.Abs(Vector3.Dot(averagedNormal, Vector3.right)),
                Mathf.Abs(Vector3.Dot(averagedNormal, Vector3.forward)));

            if (residual > Mathf.Max(0.002f, maxCandidatePlaneResidualMeters))
                return false;
            if (axisAlignment < Mathf.Clamp01(minCandidateAxisAlignmentScore))
                return false;

            Vector3 dominantAxis = ResolveDominantAxis(averagedNormal);

            candidate = new LocalPlaneCandidate
            {
                candidateId = nextCandidateId++,
                memberCount = m_candidateMembers.Count,
                center = center,
                normal = dominantAxis,
                fitResidualMeters = residual,
                axisAlignmentScore = axisAlignment,
                axisIndex = ResolveAxisIndex(dominantAxis),
            };
            return true;
        }

        private void UpdatePersistentCandidates(float now)
        {
            HashSet<int> matchedIds = new();
            for (int i = 0; i < m_localPlaneCandidates.Count; i++)
            {
                LocalPlaneCandidate candidate = m_localPlaneCandidates[i];
                int matchedId = FindBestPersistentCandidate(candidate, matchedIds);
                if (matchedId == 0 || !m_persistentCandidates.TryGetValue(matchedId, out PersistentCandidateState state))
                {
                    state = new PersistentCandidateState
                    {
                        persistentId = m_nextPersistentCandidateId++,
                        axisIndex = candidate.axisIndex,
                        displayLockState = CellState.DisplayLockState.Unlocked,
                        stableFrames = 1,
                        unlockOutlierFrames = 0,
                        relockStableFrames = 0,
                        lastSeenTime = now,
                        memberCount = candidate.memberCount,
                        fitResidualMeters = candidate.fitResidualMeters,
                        axisAlignmentScore = candidate.axisAlignmentScore,
                        observedCenter = candidate.center,
                        observedNormal = candidate.normal,
                        displayCenter = candidate.center,
                        displayNormal = candidate.normal,
                        relocatingCenter = candidate.center,
                        relocatingNormal = candidate.normal,
                    };
                    matchedId = state.persistentId;
                }
                else
                {
                    state.stableFrames = Mathf.Min(999, state.stableFrames + 1);
                    state.lastSeenTime = now;
                    state.axisIndex = candidate.axisIndex;
                    state.memberCount = candidate.memberCount;
                    state.fitResidualMeters = Mathf.Lerp(state.fitResidualMeters, candidate.fitResidualMeters, 0.3f);
                    state.axisAlignmentScore = Mathf.Lerp(state.axisAlignmentScore, candidate.axisAlignmentScore, 0.3f);
                    state.observedCenter = Vector3.Lerp(state.observedCenter, candidate.center, Mathf.Clamp01(candidateCenterBlend));
                    Vector3 blendedObservedNormal = Vector3.Lerp(state.observedNormal, candidate.normal, Mathf.Clamp01(candidateNormalBlend));
                    state.observedNormal = blendedObservedNormal.sqrMagnitude > 1e-6f ? blendedObservedNormal.normalized : Vector3.up;
                }

                UpdatePersistentCandidateDisplay(ref state);
                m_persistentCandidates[matchedId] = state;
                matchedIds.Add(matchedId);
            }

            m_candidateCleanupKeys.Clear();
            foreach (KeyValuePair<int, PersistentCandidateState> pair in m_persistentCandidates)
            {
                if (now - pair.Value.lastSeenTime > candidateHoldSeconds)
                    m_candidateCleanupKeys.Add(pair.Key);
            }

            for (int i = 0; i < m_candidateCleanupKeys.Count; i++)
                m_persistentCandidates.Remove(m_candidateCleanupKeys[i]);
        }

        private void UpdateConfirmedStructures(float now)
        {
            HashSet<int> matchedConfirmedIds = new();
            foreach (PersistentCandidateState candidate in m_persistentCandidates.Values)
            {
                if (!ShouldPromoteCandidateToConfirmed(candidate, now))
                    continue;

                int matchedId = FindBestConfirmedStructure(candidate, matchedConfirmedIds);
                if (matchedId == 0 || !m_confirmedStructures.TryGetValue(matchedId, out ConfirmedStructureState structure))
                {
                    bool centerPriority = IsCandidateInViewportCenter(candidate, centerPromotionViewportRadius);
                    structure = new ConfirmedStructureState
                    {
                        confirmedId = m_nextConfirmedStructureId++,
                        status = ConfirmedStructureStatus.Confirmed,
                        axisIndex = candidate.axisIndex,
                        sourcePersistentCandidateId = candidate.persistentId,
                        displayLockState = CellState.DisplayLockState.Unlocked,
                        unlockOutlierFrames = 0,
                        relockStableFrames = 0,
                        memberCount = candidate.memberCount,
                        stableFrames = 1,
                        lastObservedTime = now,
                        supportScore = 1f,
                        conflictScore = 0f,
                        replacementScore = 0f,
                        fitResidualMeters = candidate.fitResidualMeters,
                        axisAlignmentScore = candidate.axisAlignmentScore,
                        lastCenterObservedTime = centerPriority ? now : float.NegativeInfinity,
                        centerSupportScore = centerPriority ? centerConfirmedSupportIncrement : 0f,
                        observedCenter = candidate.displayCenter,
                        observedNormal = candidate.displayNormal.sqrMagnitude > 1e-6f ? candidate.displayNormal.normalized : Vector3.up,
                        displayCenter = candidate.displayCenter,
                        displayNormal = candidate.displayNormal.sqrMagnitude > 1e-6f ? candidate.displayNormal.normalized : Vector3.up,
                        relocatingCenter = candidate.displayCenter,
                        relocatingNormal = candidate.displayNormal.sqrMagnitude > 1e-6f ? candidate.displayNormal.normalized : Vector3.up,
                    };
                    matchedId = structure.confirmedId;
                }
                else
                {
                    bool centerPriority = IsCandidateInViewportCenter(candidate, centerPromotionViewportRadius);
                    structure.status = ConfirmedStructureStatus.Confirmed;
                    structure.axisIndex = candidate.axisIndex;
                    structure.sourcePersistentCandidateId = candidate.persistentId;
                    structure.memberCount = candidate.memberCount;
                    structure.stableFrames = Mathf.Min(999, structure.stableFrames + 1);
                    structure.lastObservedTime = now;
                    structure.supportScore = Mathf.Min(999f, structure.supportScore + 1f);
                    structure.fitResidualMeters = Mathf.Lerp(structure.fitResidualMeters, candidate.fitResidualMeters, 0.25f);
                    structure.axisAlignmentScore = Mathf.Lerp(structure.axisAlignmentScore, candidate.axisAlignmentScore, 0.25f);
                    structure.observedCenter = Vector3.Lerp(
                        structure.observedCenter,
                        candidate.displayCenter,
                        Mathf.Clamp01(confirmedCenterBlend));
                    Vector3 blendedObservedNormal = Vector3.Lerp(
                        structure.observedNormal,
                        candidate.displayNormal,
                        Mathf.Clamp01(confirmedNormalBlend));
                    structure.observedNormal = blendedObservedNormal.sqrMagnitude > 1e-6f ? blendedObservedNormal.normalized : Vector3.up;
                    if (centerPriority)
                    {
                        structure.lastCenterObservedTime = now;
                        structure.centerSupportScore = Mathf.Min(999f, structure.centerSupportScore + centerConfirmedSupportIncrement);
                    }
                    else
                    {
                        structure.centerSupportScore = Mathf.Max(0f, structure.centerSupportScore - 0.05f);
                    }
                }

                UpdateConfirmedStructureDisplay(ref structure);
                m_confirmedStructures[matchedId] = structure;
                matchedConfirmedIds.Add(matchedId);
            }

            MergeConfirmedStructures();
            CleanupConfirmedStructures(now);
        }

        private bool ShouldPromoteCandidateToConfirmed(PersistentCandidateState candidate, float now)
        {
            if (!IsCandidateVisible(candidate, now))
                return false;

            bool viewPriority = IsWorldPointInViewport(candidate.displayCenter, confirmedViewportPadding);
            bool centerPriority = IsCandidateInViewportCenter(candidate, centerPromotionViewportRadius);
            bool ceilingPriority = IsCeilingPriorityCandidate(candidate);
            int requiredFrames = Mathf.Max(
                minPersistentCandidateFrames,
                minConfirmedCandidateFrames
                - (viewPriority ? Mathf.Max(0, viewPriorityConfirmedFrameReduction) : 0)
                - (centerPriority ? Mathf.Max(0, centerPriorityConfirmedFrameReduction) : 0)
                - (ceilingPriority ? Mathf.Max(0, ceilingPriorityConfirmedFrameReduction) : 0));
            int requiredMembers = Mathf.Max(
                minCandidateCells,
                minConfirmedCandidateMembers
                - (viewPriority ? Mathf.Max(0, viewPriorityConfirmedMemberReduction) : 0)
                - (centerPriority ? Mathf.Max(0, centerPriorityConfirmedMemberReduction) : 0)
                - (ceilingPriority ? Mathf.Max(0, ceilingPriorityConfirmedMemberReduction) : 0));
            float maxResidual = Mathf.Max(
                0.002f,
                maxConfirmedCandidateResidualMeters
                * (viewPriority ? Mathf.Max(1f, viewPriorityConfirmedResidualScale) : 1f)
                * (centerPriority ? Mathf.Max(1f, centerPriorityConfirmedResidualScale) : 1f)
                * (ceilingPriority ? Mathf.Max(1f, ceilingPriorityConfirmedResidualScale) : 1f));
            float minAxisAlignment = Mathf.Clamp01(
                minConfirmedAxisAlignmentScore
                - (viewPriority ? Mathf.Max(0f, viewPriorityConfirmedAxisAlignmentReduction) : 0f)
                - (centerPriority ? Mathf.Max(0f, centerPriorityConfirmedAxisAlignmentReduction) : 0f));

            if (candidate.stableFrames < requiredFrames)
                return false;
            if (candidate.memberCount < requiredMembers)
                return false;
            if (candidate.fitResidualMeters > maxResidual)
                return false;
            if (candidate.axisAlignmentScore < minAxisAlignment)
                return false;
            return true;
        }

        private bool IsCandidateInViewportCenter(PersistentCandidateState candidate, float radius)
        {
            if (sampleCamera == null)
                return false;

            Vector3 viewport = sampleCamera.WorldToViewportPoint(candidate.displayCenter);
            if (viewport.z <= 0f)
                return false;

            float dx = viewport.x - 0.5f;
            float dy = viewport.y - 0.5f;
            radius = Mathf.Max(0.02f, radius);
            return dx * dx + dy * dy <= radius * radius;
        }

        private bool IsCeilingPriorityCandidate(PersistentCandidateState candidate)
        {
            if (sampleCamera == null)
                return false;

            Vector3 forward = sampleCamera.transform.forward;
            if (forward.y < ceilingPriorityMinForwardY)
                return false;

            Vector3 candidateNormal = candidate.displayNormal.sqrMagnitude > 1e-6f ? candidate.displayNormal.normalized : Vector3.up;
            if (candidateNormal.y > -0.55f)
                return false;

            Vector3 viewport = sampleCamera.WorldToViewportPoint(candidate.displayCenter);
            if (viewport.z <= 0f)
                return false;
            if (viewport.y < ceilingPriorityViewportMinY)
                return false;

            float heightOffset = candidate.displayCenter.y - sampleCamera.transform.position.y;
            return heightOffset >= ceilingPriorityMinHeightOffsetMeters;
        }

        private int FindBestConfirmedStructure(PersistentCandidateState candidate, HashSet<int> matchedConfirmedIds)
        {
            float minNormalDot = Mathf.Clamp(minConfirmedMatchNormalDot, 0f, 0.9999f);
            float bestScore = float.MaxValue;
            int bestId = 0;
            Vector3 candidateNormal = candidate.displayNormal.sqrMagnitude > 1e-6f ? candidate.displayNormal.normalized : Vector3.up;
            bool centerPriority = IsCandidateInViewportCenter(candidate, centerPromotionViewportRadius);
            float now = Time.unscaledTime;

            foreach (KeyValuePair<int, ConfirmedStructureState> pair in m_confirmedStructures)
            {
                if (matchedConfirmedIds.Contains(pair.Key))
                    continue;

                ConfirmedStructureState structure = pair.Value;
                if (structure.status == ConfirmedStructureStatus.Invalidated || structure.status == ConfirmedStructureStatus.Replaced)
                    continue;
                if (structure.axisIndex != candidate.axisIndex)
                    continue;

                float maxDistance = Mathf.Max(candidateMatchDistanceMeters, confirmedMatchDistanceMeters);
                if (structure.stableFrames >= Mathf.Max(2, minConfirmedCandidateFrames) &&
                    structure.displayLockState != CellState.DisplayLockState.Unlocked)
                {
                    maxDistance = Mathf.Min(
                        maxDistance,
                        Mathf.Max(worldCellSizeMeters, confirmedMaxReassociationDistanceMeters));
                }

                float distance = Vector3.Distance(structure.displayCenter, candidate.displayCenter);
                if (distance > maxDistance)
                    continue;

                Vector3 structureNormal = structure.displayNormal.sqrMagnitude > 1e-6f ? structure.displayNormal.normalized : Vector3.up;
                float normalDot = Mathf.Abs(Vector3.Dot(structureNormal, candidateNormal));
                if (normalDot < minNormalDot)
                    continue;

                float score = distance / Mathf.Max(0.001f, maxDistance) + (1f - normalDot);
                if (centerPriority)
                {
                    float centerRecencyScore = HasRecentCenterSupport(structure, now)
                        ? 1f - Mathf.Clamp01((now - structure.lastCenterObservedTime) / Mathf.Max(0.1f, centerConfirmedRecentSupportSeconds))
                        : 0f;
                    float centerCarry = Mathf.Clamp01(structure.centerSupportScore / 8f);
                    score -= Mathf.Max(0f, centerConfirmedMatchBias) * (centerRecencyScore * 0.7f + centerCarry * 0.3f);
                }
                if (score >= bestScore)
                    continue;

                bestScore = score;
                bestId = pair.Key;
            }

            return bestId;
        }

        private void UpdateConfirmedStructureDisplay(ref ConfirmedStructureState structure)
        {
            if (structure.displayNormal.sqrMagnitude <= 1e-6f)
                structure.displayNormal = structure.observedNormal.sqrMagnitude > 1e-6f ? structure.observedNormal : Vector3.up;
            if (structure.relocatingNormal.sqrMagnitude <= 1e-6f)
                structure.relocatingNormal = structure.observedNormal.sqrMagnitude > 1e-6f ? structure.observedNormal : Vector3.up;

            float unlockDistance = Mathf.Max(0.001f, confirmedUnlockDistanceMeters);
            float displayDelta = Vector3.Distance(structure.displayCenter, structure.observedCenter);

            switch (structure.displayLockState)
            {
                case CellState.DisplayLockState.Unlocked:
                    structure.displayCenter = structure.observedCenter;
                    structure.displayNormal = structure.observedNormal.sqrMagnitude > 1e-6f ? structure.observedNormal.normalized : Vector3.up;
                    structure.relocatingCenter = structure.displayCenter;
                    structure.relocatingNormal = structure.displayNormal;
                    structure.displayLockState = CellState.DisplayLockState.Locked;
                    structure.unlockOutlierFrames = 0;
                    structure.relockStableFrames = 0;
                    break;

                case CellState.DisplayLockState.Locked:
                    if (displayDelta > unlockDistance)
                    {
                        structure.unlockOutlierFrames++;
                        if (structure.unlockOutlierFrames >= Mathf.Max(1, confirmedUnlockAfterOutlierFrames))
                        {
                            structure.displayLockState = CellState.DisplayLockState.Relocating;
                            structure.unlockOutlierFrames = 0;
                            structure.relockStableFrames = 0;
                            structure.relocatingCenter = structure.observedCenter;
                            structure.relocatingNormal = structure.observedNormal.sqrMagnitude > 1e-6f ? structure.observedNormal.normalized : Vector3.up;
                        }
                    }
                    else
                    {
                        structure.unlockOutlierFrames = 0;
                    }
                    break;

                case CellState.DisplayLockState.Relocating:
                    structure.relocatingCenter = Vector3.Lerp(
                        structure.relocatingCenter,
                        structure.observedCenter,
                        Mathf.Clamp01(confirmedRelocatingCenterBlend));
                    Vector3 relocatingNormal = Vector3.Lerp(
                        structure.relocatingNormal,
                        structure.observedNormal,
                        Mathf.Clamp01(confirmedRelocatingNormalBlend));
                    structure.relocatingNormal = relocatingNormal.sqrMagnitude > 1e-6f ? relocatingNormal.normalized : Vector3.up;

                    float relocationDelta = Vector3.Distance(structure.relocatingCenter, structure.observedCenter);
                    if (relocationDelta <= unlockDistance * 0.35f)
                    {
                        structure.relockStableFrames++;
                        if (structure.relockStableFrames >= Mathf.Max(1, confirmedRelockAfterStableFrames))
                        {
                            structure.displayCenter = structure.relocatingCenter;
                            structure.displayNormal = structure.relocatingNormal;
                            structure.displayLockState = CellState.DisplayLockState.Locked;
                            structure.relockStableFrames = 0;
                            structure.unlockOutlierFrames = 0;
                        }
                    }
                    else
                    {
                        structure.relockStableFrames = 0;
                    }
                    break;
            }
        }

        private void MergeConfirmedStructures()
        {
            List<int> ids = new(m_confirmedStructures.Keys);
            float maxDistance = Mathf.Max(worldCellSizeMeters * 1.5f, confirmedMergeDistanceMeters);
            float minNormalDot = Mathf.Clamp(confirmedMergeNormalDot, 0f, 0.9999f);

            for (int i = 0; i < ids.Count; i++)
            {
                if (!m_confirmedStructures.TryGetValue(ids[i], out ConfirmedStructureState a))
                    continue;
                if (a.status != ConfirmedStructureStatus.Confirmed)
                    continue;

                for (int j = i + 1; j < ids.Count; j++)
                {
                    if (!m_confirmedStructures.TryGetValue(ids[j], out ConfirmedStructureState b))
                        continue;
                    if (b.status != ConfirmedStructureStatus.Confirmed)
                        continue;
                    if (a.axisIndex != b.axisIndex)
                        continue;

                    float distance = Vector3.Distance(a.displayCenter, b.displayCenter);
                    if (distance > maxDistance)
                        continue;

                    Vector3 aNormal = a.displayNormal.sqrMagnitude > 1e-6f ? a.displayNormal.normalized : Vector3.up;
                    Vector3 bNormal = b.displayNormal.sqrMagnitude > 1e-6f ? b.displayNormal.normalized : Vector3.up;
                    float normalDot = Mathf.Abs(Vector3.Dot(aNormal, bNormal));
                    if (normalDot < minNormalDot)
                        continue;

                    float aScore = ScoreConfirmedStructure(a);
                    float bScore = ScoreConfirmedStructure(b);
                    bool keepA = aScore >= bScore;
                    ConfirmedStructureState winner = keepA ? a : b;
                    ConfirmedStructureState loser = keepA ? b : a;

                    float winnerWeight = Mathf.Max(1f, winner.supportScore);
                    float loserWeight = Mathf.Max(1f, loser.supportScore);
                    float totalWeight = winnerWeight + loserWeight;

                    winner.observedCenter = Vector3.Lerp(loser.observedCenter, winner.observedCenter, winnerWeight / totalWeight);
                    winner.displayCenter = Vector3.Lerp(loser.displayCenter, winner.displayCenter, winnerWeight / totalWeight);
                    winner.relocatingCenter = Vector3.Lerp(loser.relocatingCenter, winner.relocatingCenter, winnerWeight / totalWeight);

                    Vector3 mergedObservedNormal = Vector3.Lerp(loser.observedNormal, winner.observedNormal, winnerWeight / totalWeight);
                    winner.observedNormal = mergedObservedNormal.sqrMagnitude > 1e-6f ? mergedObservedNormal.normalized : winner.observedNormal;
                    Vector3 mergedDisplayNormal = Vector3.Lerp(loser.displayNormal, winner.displayNormal, winnerWeight / totalWeight);
                    winner.displayNormal = mergedDisplayNormal.sqrMagnitude > 1e-6f ? mergedDisplayNormal.normalized : winner.displayNormal;
                    Vector3 mergedRelocatingNormal = Vector3.Lerp(loser.relocatingNormal, winner.relocatingNormal, winnerWeight / totalWeight);
                    winner.relocatingNormal = mergedRelocatingNormal.sqrMagnitude > 1e-6f ? mergedRelocatingNormal.normalized : winner.relocatingNormal;

                    winner.memberCount = Mathf.Max(winner.memberCount, loser.memberCount);
                    winner.stableFrames = Mathf.Max(winner.stableFrames, loser.stableFrames);
                    winner.supportScore = Mathf.Min(999f, winner.supportScore + loser.supportScore * 0.5f);
                    winner.fitResidualMeters = Mathf.Min(winner.fitResidualMeters, loser.fitResidualMeters);
                    winner.axisAlignmentScore = Mathf.Max(winner.axisAlignmentScore, loser.axisAlignmentScore);
                    winner.lastObservedTime = Mathf.Max(winner.lastObservedTime, loser.lastObservedTime);
                    winner.lastCenterObservedTime = Mathf.Max(winner.lastCenterObservedTime, loser.lastCenterObservedTime);
                    winner.centerSupportScore = Mathf.Min(999f, winner.centerSupportScore + loser.centerSupportScore * 0.5f);

                    loser.status = ConfirmedStructureStatus.Replaced;
                    loser.replacementScore = Mathf.Min(999f, loser.replacementScore + 1f);

                    if (keepA)
                    {
                        a = winner;
                        b = loser;
                    }
                    else
                    {
                        b = winner;
                        a = loser;
                    }

                    m_confirmedStructures[a.confirmedId] = a;
                    m_confirmedStructures[b.confirmedId] = b;
                }
            }
        }

        private void CleanupConfirmedStructures(float now)
        {
            m_confirmedCleanupKeys.Clear();
            m_confirmedInvalidateKeys.Clear();
            foreach (KeyValuePair<int, ConfirmedStructureState> pair in m_confirmedStructures)
            {
                ConfirmedStructureState structure = pair.Value;
                if (structure.status == ConfirmedStructureStatus.Confirmed &&
                    IsLowQualityConfirmedStructure(structure) &&
                    now - structure.lastObservedTime > Mathf.Max(0.1f, lowQualityConfirmedCleanupSeconds))
                {
                    m_confirmedInvalidateKeys.Add(pair.Key);
                    continue;
                }

                if (structure.status != ConfirmedStructureStatus.Replaced &&
                    structure.status != ConfirmedStructureStatus.Invalidated)
                    continue;

                if (now - structure.lastObservedTime <= Mathf.Max(0.1f, replacedConfirmedCleanupSeconds))
                    continue;

                m_confirmedCleanupKeys.Add(pair.Key);
            }

            for (int i = 0; i < m_confirmedInvalidateKeys.Count; i++)
            {
                int id = m_confirmedInvalidateKeys[i];
                if (!m_confirmedStructures.TryGetValue(id, out ConfirmedStructureState structure))
                    continue;

                structure.status = ConfirmedStructureStatus.Invalidated;
                m_confirmedStructures[id] = structure;
            }

            for (int i = 0; i < m_confirmedCleanupKeys.Count; i++)
                m_confirmedStructures.Remove(m_confirmedCleanupKeys[i]);
        }

        private bool IsLowQualityConfirmedStructure(ConfirmedStructureState structure)
        {
            if (structure.supportScore < lowQualityConfirmedMinSupportScore)
                return true;
            if (structure.fitResidualMeters > lowQualityConfirmedMaxResidualMeters)
                return true;
            if (structure.axisAlignmentScore < lowQualityConfirmedMinAxisAlignmentScore)
                return true;
            return false;
        }

        private static float ScoreConfirmedStructure(ConfirmedStructureState structure)
        {
            return structure.supportScore * 0.5f
                + structure.stableFrames * 0.35f
                + structure.memberCount * 0.2f
                + structure.axisAlignmentScore * 2f
                - structure.fitResidualMeters * 20f;
        }

        private void TryWriteDesktopMonitor(float now)
        {
            if (!exportDesktopMonitor)
                return;
            if (now < m_nextMonitorWriteTime)
                return;

            m_nextMonitorWriteTime = now + Mathf.Max(0.1f, desktopMonitorWriteIntervalSeconds);

            try
            {
                string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
                if (string.IsNullOrWhiteSpace(desktopPath))
                    return;

                string fileName = string.IsNullOrWhiteSpace(desktopMonitorFileName)
                    ? "DepthEffectsStableIdGridMonitor.md"
                    : desktopMonitorFileName.Trim();
                string outputPath = Path.Combine(desktopPath, fileName);

                StringBuilder builder = new();
                builder.AppendLine("# DepthEffects Stable Id Grid Monitor");
                builder.AppendLine();
                builder.AppendLine($"- Time: `{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}`");
                builder.AppendLine($"- Display Mode: `{displayMode}`");
                builder.AppendLine($"- World Cells: `{m_worldCells.Count}`");
                builder.AppendLine($"- Raw Local Plane Candidates: `{m_localPlaneCandidates.Count}`");
                builder.AppendLine($"- Persistent Candidates: `{m_persistentCandidates.Count}`");
                builder.AppendLine($"- Confirmed Structures: `{m_confirmedStructures.Count}`");
                builder.AppendLine();

                AppendPoseMonitor(builder);
                builder.AppendLine();
                AppendResponseMonitor(builder, now);
                builder.AppendLine();
                AppendVisibleCandidateTable(builder, now);
                builder.AppendLine();
                AppendConfirmedStructureTable(builder, now);
                builder.AppendLine();
                AppendStableWorldCellTable(builder, now);

                File.WriteAllText(outputPath, builder.ToString(), Encoding.UTF8);

                if (sampleCamera != null)
                {
                    m_lastMonitorCameraPosition = sampleCamera.transform.position;
                    m_lastMonitorCameraRotation = sampleCamera.transform.rotation;
                    m_hasLastMonitorPose = true;
                }
            }
            catch (Exception ex)
            {
                if (debugLog)
                    Debug.LogWarning($"[DepthEffectsStableIdGridRenderer] Failed to write desktop monitor: {ex.Message}");
            }
        }

        private void AppendPoseMonitor(StringBuilder builder)
        {
            builder.AppendLine("## Headset Pose");
            builder.AppendLine();

            if (sampleCamera == null)
            {
                builder.AppendLine("- Camera: `_none_`");
                return;
            }

            Transform cameraTransform = sampleCamera.transform;
            Vector3 currentPosition = cameraTransform.position;
            Quaternion currentRotation = cameraTransform.rotation;
            Vector3 currentEuler = currentRotation.eulerAngles;
            Vector3 currentForward = cameraTransform.forward;
            Vector3 currentUp = cameraTransform.up;

            float positionDeltaMeters = m_hasLastMonitorPose
                ? Vector3.Distance(currentPosition, m_lastMonitorCameraPosition)
                : 0f;
            float angularDeltaDegrees = m_hasLastMonitorPose
                ? Quaternion.Angle(currentRotation, m_lastMonitorCameraRotation)
                : 0f;

            builder.AppendLine($"- Position: `{FormatVector(currentPosition)}`");
            builder.AppendLine($"- Euler XYZ (deg): `{FormatVector(currentEuler)}`");
            builder.AppendLine($"- Forward: `{FormatVector(currentForward)}`");
            builder.AppendLine($"- Up: `{FormatVector(currentUp)}`");
            builder.AppendLine($"- Position Delta Since Last Write (m): `{FormatFloat(positionDeltaMeters)}`");
            builder.AppendLine($"- Angular Delta Since Last Write (deg): `{FormatFloat(angularDeltaDegrees)}`");
        }

        private void AppendResponseMonitor(StringBuilder builder, float now)
        {
            builder.AppendLine("## Confirmed Response");
            builder.AppendLine();
            builder.AppendLine($"- Current Viewport Confirmed Count: `{m_responseCurrentViewportConfirmedCount}`");
            builder.AppendLine($"- Peak Viewport Confirmed Count Since Last View Shift: `{m_responsePeakViewportConfirmedCount}`");
            builder.AppendLine($"- Response Session Active: `{m_responseSessionActive}`");
            builder.AppendLine($"- Seconds Since Last View Shift: `{FormatFloat(m_responseSessionActive ? now - m_responseSessionStartTime : 0f)}`");
            builder.AppendLine($"- First Confirmed Latency Since Last View Shift (s): `{FormatOptionalFloat(m_responseFirstConfirmedLatency)}`");
            builder.AppendLine($"- Peak Confirmed Count Latency Since Last View Shift (s): `{FormatOptionalFloat(m_responsePeakConfirmedLatency)}`");
            builder.AppendLine($"- View Shift Trigger Position Delta (m): `{FormatFloat(responseTriggerPositionDeltaMeters)}`");
            builder.AppendLine($"- View Shift Trigger Angular Delta (deg): `{FormatFloat(responseTriggerAngularDeltaDegrees)}`");
            builder.AppendLine($"- Minimum Viewport Confirmed Count To Mark Response: `{Mathf.Max(1, responseMinViewportConfirmedCount)}`");
        }

        private void AppendVisibleCandidateTable(StringBuilder builder, float now)
        {
            builder.AppendLine("## Visible Persistent Candidate Centers");
            builder.AppendLine();
            builder.AppendLine("| id | state | lowSupport | stableFrames | axis | members | residual(m) | align | unseen(s) | displayCenter | observedCenter | delta(m) | displayNormal |");
            builder.AppendLine("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |");

            bool hasRows = false;
            foreach (PersistentCandidateState candidate in m_persistentCandidates.Values)
            {
                if (!IsCandidateVisible(candidate, now))
                    continue;

                hasRows = true;
                float unseenSeconds = Mathf.Max(0f, now - candidate.lastSeenTime);
                float deltaMeters = Vector3.Distance(candidate.displayCenter, candidate.observedCenter);
                builder.Append("| ")
                    .Append(candidate.persistentId).Append(" | ")
                    .Append(candidate.displayLockState).Append(" | ")
                    .Append(IsLowSupportCandidate(candidate, now) ? "yes" : "no").Append(" | ")
                    .Append(candidate.stableFrames).Append(" | ")
                    .Append(candidate.axisIndex).Append(" | ")
                    .Append(candidate.memberCount).Append(" | ")
                    .Append(FormatFloat(candidate.fitResidualMeters)).Append(" | ")
                    .Append(FormatFloat(candidate.axisAlignmentScore)).Append(" | ")
                    .Append(FormatFloat(unseenSeconds)).Append(" | ")
                    .Append(FormatVector(candidate.displayCenter)).Append(" | ")
                    .Append(FormatVector(candidate.observedCenter)).Append(" | ")
                    .Append(FormatFloat(deltaMeters)).Append(" | ")
                    .Append(FormatVector(candidate.displayNormal)).AppendLine(" |");
            }

            if (!hasRows)
                builder.AppendLine("| _none_ |  |  |  |  |  |  |  |  |  |  |  |  |");
        }

        private void AppendStableWorldCellTable(StringBuilder builder, float now)
        {
            builder.AppendLine("## Stable World Cells");
            builder.AppendLine();
            builder.AppendLine("| cellId | state | support | lowSupport | stableHits | retained | quality | pendingReconnect | confidence | unseen(s) | displayPos | observedPos | delta(m) | displayNormal |");
            builder.AppendLine("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |");

            bool hasRows = false;
            foreach (KeyValuePair<Vector3Int, CellState> pair in m_worldCells)
            {
                CellState cell = pair.Value;
                if (!cell.valid || cell.stableHits < minStableHits)
                    continue;

                hasRows = true;
                float unseenSeconds = Mathf.Max(0f, now - cell.lastSeenTime);
                float deltaMeters = Vector3.Distance(cell.displayWorldPos, cell.worldPos);
                builder.Append("| ")
                    .Append(FormatCellId(cell.worldCellId)).Append(" | ")
                    .Append(cell.displayLockState).Append(" | ")
                    .Append(GetCellSupportState(cell, now)).Append(" | ")
                    .Append(IsLowSupportCell(cell, now) ? "yes" : "no").Append(" | ")
                    .Append(cell.stableHits).Append(" | ")
                    .Append(cell.retainedMemory ? "yes" : "no").Append(" | ")
                    .Append(FormatFloat(cell.memoryQualityScore)).Append(" | ")
                    .Append(cell.pendingReconnectFrames).Append(" | ")
                    .Append(FormatFloat(cell.confidence)).Append(" | ")
                    .Append(FormatFloat(unseenSeconds)).Append(" | ")
                    .Append(FormatVector(cell.displayWorldPos)).Append(" | ")
                    .Append(FormatVector(cell.worldPos)).Append(" | ")
                    .Append(FormatFloat(deltaMeters)).Append(" | ")
                    .Append(FormatVector(cell.displayNormal)).AppendLine(" |");
            }

            if (!hasRows)
                builder.AppendLine("| _none_ |  |  |  |  |  |  |  |  |  |  |  |  |  |");
        }

        private void AppendConfirmedStructureTable(StringBuilder builder, float now)
        {
            builder.AppendLine("## Confirmed Structures");
            builder.AppendLine();
            builder.AppendLine("| id | status | sourceCandidateId | stableFrames | support | conflict | replacement | axis | members | residual(m) | align | unseen(s) | displayCenter | observedCenter | displayNormal |");
            builder.AppendLine("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |");

            bool hasRows = false;
            foreach (ConfirmedStructureState structure in m_confirmedStructures.Values)
            {
                hasRows = true;
                float unseenSeconds = Mathf.Max(0f, now - structure.lastObservedTime);
                builder.Append("| ")
                    .Append(structure.confirmedId).Append(" | ")
                    .Append(structure.status).Append(" | ")
                    .Append(structure.sourcePersistentCandidateId).Append(" | ")
                    .Append(structure.stableFrames).Append(" | ")
                    .Append(FormatFloat(structure.supportScore)).Append(" | ")
                    .Append(FormatFloat(structure.conflictScore)).Append(" | ")
                    .Append(FormatFloat(structure.replacementScore)).Append(" | ")
                    .Append(structure.axisIndex).Append(" | ")
                    .Append(structure.memberCount).Append(" | ")
                    .Append(FormatFloat(structure.fitResidualMeters)).Append(" | ")
                    .Append(FormatFloat(structure.axisAlignmentScore)).Append(" | ")
                    .Append(FormatFloat(unseenSeconds)).Append(" | ")
                    .Append(FormatVector(structure.displayCenter)).Append(" | ")
                    .Append(FormatVector(structure.observedCenter)).Append(" | ")
                    .Append(FormatVector(structure.displayNormal)).AppendLine(" |");
            }

            if (!hasRows)
                builder.AppendLine("| _none_ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |");
        }

        private static string FormatFloat(float value)
        {
            return value.ToString("F3", CultureInfo.InvariantCulture);
        }

        private static string FormatOptionalFloat(float value)
        {
            return value < 0f ? "_pending_" : FormatFloat(value);
        }

        private static string FormatVector(Vector3 value)
        {
            return $"({FormatFloat(value.x)}, {FormatFloat(value.y)}, {FormatFloat(value.z)})";
        }

        private static string FormatCellId(Vector3Int value)
        {
            return $"({value.x}, {value.y}, {value.z})";
        }

        private int FindBestPersistentCandidate(LocalPlaneCandidate candidate, HashSet<int> matchedIds)
        {
            float minNormalDot = Mathf.Clamp(minCandidateMatchNormalDot, 0f, 0.9999f);
            int maxMemberDelta = Mathf.Max(1, maxCandidateMemberCountDelta);
            float bestScore = float.MaxValue;
            int bestId = 0;
            Vector3 candidateNormal = candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : Vector3.up;

            foreach (KeyValuePair<int, PersistentCandidateState> pair in m_persistentCandidates)
            {
                if (matchedIds.Contains(pair.Key))
                    continue;

                PersistentCandidateState state = pair.Value;
                if (state.axisIndex != candidate.axisIndex)
                    continue;

                float matchDistance = Mathf.Max(worldCellSizeMeters * 1.5f, candidateMatchDistanceMeters);
                if (state.stableFrames >= Mathf.Max(2, freezeCandidateAfterFrames) &&
                    state.displayLockState != CellState.DisplayLockState.Unlocked)
                {
                    matchDistance = Mathf.Min(
                        matchDistance,
                        Mathf.Max(worldCellSizeMeters, persistentCandidateMaxReassociationDistanceMeters));

                    if (state.stableFrames >= Mathf.Max(freezeCandidateAfterFrames + 2, minPersistentCandidateFrames + 2))
                    {
                        matchDistance = Mathf.Min(
                            matchDistance,
                            Mathf.Max(worldCellSizeMeters * 0.75f, matureCandidateMaxReassociationDistanceMeters));
                    }

                    if (state.fitResidualMeters <= 0.03f && state.axisAlignmentScore >= 0.95f)
                    {
                        matchDistance = Mathf.Min(
                            matchDistance,
                            Mathf.Max(worldCellSizeMeters * 0.6f, matureCandidateMaxReassociationDistanceMeters * 0.85f));
                    }
                }

                float distance = Vector3.Distance(state.displayCenter, candidate.center);
                if (distance > matchDistance)
                    continue;

                Vector3 stateNormal = state.displayNormal.sqrMagnitude > 1e-6f ? state.displayNormal.normalized : Vector3.up;
                float normalDot = Mathf.Abs(Vector3.Dot(stateNormal, candidateNormal));
                if (normalDot < minNormalDot)
                    continue;

                int memberDelta = Mathf.Abs(state.memberCount - candidate.memberCount);
                if (memberDelta > maxMemberDelta)
                    continue;

                float distanceScore = distance / Mathf.Max(0.001f, matchDistance);
                float normalScore = 1f - normalDot;
                float memberScore = memberDelta / (float)Mathf.Max(1, maxMemberDelta);
                float residualScore = Mathf.Abs(state.fitResidualMeters - candidate.fitResidualMeters) /
                    Mathf.Max(0.001f, maxCandidatePlaneResidualMeters);

                float score =
                    distanceScore * Mathf.Max(0.001f, candidateMatchDistanceWeight) +
                    normalScore * Mathf.Max(0.001f, candidateMatchNormalWeight) +
                    memberScore * Mathf.Max(0.001f, candidateMatchMemberWeight) +
                    residualScore * Mathf.Max(0.001f, candidateMatchResidualWeight);

                if (score >= bestScore)
                    continue;

                bestScore = score;
                bestId = pair.Key;
            }

            return bestId;
        }

        private void UpdatePersistentCandidateDisplay(ref PersistentCandidateState state)
        {
            if (state.displayNormal.sqrMagnitude <= 1e-6f)
                state.displayNormal = state.observedNormal.sqrMagnitude > 1e-6f ? state.observedNormal : Vector3.up;
            if (state.relocatingNormal.sqrMagnitude <= 1e-6f)
                state.relocatingNormal = state.observedNormal.sqrMagnitude > 1e-6f ? state.observedNormal : Vector3.up;

            int freezeThreshold = Mathf.Max(1, freezeCandidateAfterFrames);
            float unlockDistance = Mathf.Max(0.001f, candidateUnlockDistanceMeters);
            float mediumRelocationDistance = Mathf.Max(unlockDistance, candidateMediumRelocationDistanceMeters);
            float hardRelocationDistance = Mathf.Max(unlockDistance * 1.5f, candidateHardRelocationDistanceMeters);
            float displayDelta = Vector3.Distance(state.displayCenter, state.observedCenter);

            if (state.stableFrames < freezeThreshold)
            {
                state.displayLockState = CellState.DisplayLockState.Unlocked;
                state.unlockOutlierFrames = 0;
                state.relockStableFrames = 0;
                state.displayCenter = Vector3.Lerp(state.displayCenter, state.observedCenter, Mathf.Clamp01(candidateCenterBlend));
                Vector3 blendedNormal = Vector3.Lerp(state.displayNormal, state.observedNormal, Mathf.Clamp01(candidateNormalBlend));
                state.displayNormal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
                state.relocatingCenter = state.displayCenter;
                state.relocatingNormal = state.displayNormal;
                return;
            }

            if (displayDelta > hardRelocationDistance)
            {
                state.displayCenter = state.observedCenter;
                state.displayNormal = state.observedNormal.sqrMagnitude > 1e-6f ? state.observedNormal.normalized : Vector3.up;
                state.relocatingCenter = state.displayCenter;
                state.relocatingNormal = state.displayNormal;
                state.displayLockState = CellState.DisplayLockState.Locked;
                state.unlockOutlierFrames = 0;
                state.relockStableFrames = 0;
                state.stableFrames = Mathf.Min(state.stableFrames, Mathf.Max(0, candidateHardRelocationResetStableFrames));
                return;
            }

            if (displayDelta > mediumRelocationDistance)
            {
                state.displayCenter = state.observedCenter;
                state.displayNormal = state.observedNormal.sqrMagnitude > 1e-6f ? state.observedNormal.normalized : Vector3.up;
                state.relocatingCenter = state.displayCenter;
                state.relocatingNormal = state.displayNormal;
                state.displayLockState = CellState.DisplayLockState.Locked;
                state.unlockOutlierFrames = 0;
                state.relockStableFrames = 0;
                state.stableFrames = Mathf.Min(state.stableFrames, Mathf.Max(0, candidateMediumRelocationResetStableFrames));
                return;
            }

            switch (state.displayLockState)
            {
                case CellState.DisplayLockState.Unlocked:
                    state.displayCenter = state.observedCenter;
                    state.displayNormal = state.observedNormal.sqrMagnitude > 1e-6f ? state.observedNormal.normalized : Vector3.up;
                    state.relocatingCenter = state.displayCenter;
                    state.relocatingNormal = state.displayNormal;
                    state.displayLockState = CellState.DisplayLockState.Locked;
                    state.unlockOutlierFrames = 0;
                    state.relockStableFrames = 0;
                    break;

                case CellState.DisplayLockState.Locked:
                    if (displayDelta > unlockDistance)
                    {
                        state.unlockOutlierFrames++;
                        if (state.unlockOutlierFrames >= Mathf.Max(1, candidateUnlockAfterOutlierFrames))
                        {
                            state.displayLockState = CellState.DisplayLockState.Relocating;
                            state.unlockOutlierFrames = 0;
                            state.relockStableFrames = 0;
                            state.relocatingCenter = state.observedCenter;
                            state.relocatingNormal = state.observedNormal.sqrMagnitude > 1e-6f ? state.observedNormal.normalized : Vector3.up;
                        }
                    }
                    else
                    {
                        state.unlockOutlierFrames = 0;
                    }
                    break;

                case CellState.DisplayLockState.Relocating:
                    state.relocatingCenter = Vector3.Lerp(
                        state.relocatingCenter,
                        state.observedCenter,
                        Mathf.Clamp01(candidateRelocatingCenterBlend));
                    Vector3 relocatingNormal = Vector3.Lerp(
                        state.relocatingNormal,
                        state.observedNormal,
                        Mathf.Clamp01(candidateRelocatingNormalBlend));
                    state.relocatingNormal = relocatingNormal.sqrMagnitude > 1e-6f ? relocatingNormal.normalized : Vector3.up;

                    float relocationDelta = Vector3.Distance(state.relocatingCenter, state.observedCenter);
                    if (relocationDelta <= unlockDistance * 0.35f)
                    {
                        state.relockStableFrames++;
                        if (state.relockStableFrames >= Mathf.Max(1, candidateRelockAfterStableFrames))
                        {
                            state.displayCenter = state.relocatingCenter;
                            state.displayNormal = state.relocatingNormal;
                            state.displayLockState = CellState.DisplayLockState.Locked;
                            state.relockStableFrames = 0;
                            state.unlockOutlierFrames = 0;
                        }
                    }
                    else
                    {
                        state.relockStableFrames = 0;
                    }
                    break;
            }
        }

        private static Vector3 ResolveDominantAxis(Vector3 normal)
        {
            float upDot = Mathf.Abs(Vector3.Dot(normal, Vector3.up));
            float rightDot = Mathf.Abs(Vector3.Dot(normal, Vector3.right));
            float forwardDot = Mathf.Abs(Vector3.Dot(normal, Vector3.forward));

            if (upDot >= rightDot && upDot >= forwardDot)
                return Vector3.Dot(normal, Vector3.up) >= 0f ? Vector3.up : Vector3.down;
            if (rightDot >= forwardDot)
                return Vector3.Dot(normal, Vector3.right) >= 0f ? Vector3.right : Vector3.left;
            return Vector3.Dot(normal, Vector3.forward) >= 0f ? Vector3.forward : Vector3.back;
        }

        private static int ResolveAxisIndex(Vector3 normal)
        {
            float upDot = Mathf.Abs(Vector3.Dot(normal, Vector3.up));
            float rightDot = Mathf.Abs(Vector3.Dot(normal, Vector3.right));
            float forwardDot = Mathf.Abs(Vector3.Dot(normal, Vector3.forward));

            if (upDot >= rightDot && upDot >= forwardDot)
                return 0;
            return rightDot >= forwardDot ? 1 : 2;
        }

        private static bool IsFinite(Vector4 value)
        {
            return IsFinite((Vector3)value) && !float.IsNaN(value.w) && !float.IsInfinity(value.w);
        }

        private static bool IsFinite(Vector3 value)
        {
            return
                !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
                !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
                !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        }

        private void DisposeResources()
        {
            if (m_computeBuffer != null)
            {
                m_computeBuffer.Dispose();
                m_computeBuffer = null;
            }

            if (m_depthPixels.IsCreated)
                m_depthPixels.Dispose();
            if (m_gpuReadbackBuffer.IsCreated)
                m_gpuReadbackBuffer.Dispose();

            if (m_runtimeMaterial != null)
            {
                Destroy(m_runtimeMaterial);
                m_runtimeMaterial = null;
            }
        }

        private static Mesh ResolvePrimitiveMesh(PrimitiveType primitiveType)
        {
            string builtinName = primitiveType switch
            {
                PrimitiveType.Sphere => "Sphere.fbx",
                PrimitiveType.Capsule => "Capsule.fbx",
                PrimitiveType.Cylinder => "Cylinder.fbx",
                PrimitiveType.Plane => "Plane.fbx",
                PrimitiveType.Quad => "Quad.fbx",
                _ => "Cube.fbx"
            };

            Mesh mesh = Resources.GetBuiltinResource<Mesh>(builtinName);
            if (mesh != null)
                return mesh;

            GameObject temp = GameObject.CreatePrimitive(primitiveType);
            try
            {
                MeshFilter filter = temp.GetComponent<MeshFilter>();
                return filter != null ? filter.sharedMesh : null;
            }
            finally
            {
                if (Application.isPlaying)
                    Destroy(temp);
                else
                    DestroyImmediate(temp);
            }
        }

        private Vector3Int WorldToCellId(Vector3 worldPos)
        {
            float cellSize = Mathf.Max(0.01f, worldCellSizeMeters);
            return new Vector3Int(
                Mathf.RoundToInt(worldPos.x / cellSize),
                Mathf.RoundToInt(worldPos.y / cellSize),
                Mathf.RoundToInt(worldPos.z / cellSize));
        }

        private Vector3 CellIdToWorldCenter(Vector3Int cellId)
        {
            float cellSize = Mathf.Max(0.01f, worldCellSizeMeters);
            return new Vector3(
                cellId.x * cellSize,
                cellId.y * cellSize,
                cellId.z * cellSize);
        }

        private SurfaceGridpointKey WorldCellToSurfaceGridpointKey(CellState cell)
        {
            float spacing = Mathf.Max(0.01f, gridpointSurfaceSpacingMeters);
            Vector3 position = cell.displayWorldPos;
            Vector3 normal = cell.displayNormal.sqrMagnitude > 1e-6f ? cell.displayNormal.normalized : Vector3.up;
            int axisIndex = ResolveAxisIndex(normal);

            return axisIndex switch
            {
                0 => new SurfaceGridpointKey(
                    axisIndex,
                    Mathf.RoundToInt(position.y / Mathf.Max(0.01f, worldCellSizeMeters)),
                    Mathf.RoundToInt(position.x / spacing),
                    Mathf.RoundToInt(position.z / spacing)),
                1 => new SurfaceGridpointKey(
                    axisIndex,
                    Mathf.RoundToInt(position.x / Mathf.Max(0.01f, worldCellSizeMeters)),
                    Mathf.RoundToInt(position.z / spacing),
                    Mathf.RoundToInt(position.y / spacing)),
                _ => new SurfaceGridpointKey(
                    axisIndex,
                    Mathf.RoundToInt(position.z / Mathf.Max(0.01f, worldCellSizeMeters)),
                    Mathf.RoundToInt(position.x / spacing),
                    Mathf.RoundToInt(position.y / spacing)),
            };
        }

        private Vector3 GridpointSurfaceKeyToWorldPosition(SurfaceGridpointKey key)
        {
            float spacing = Mathf.Max(0.01f, gridpointSurfaceSpacingMeters);
            float cellSize = Mathf.Max(0.01f, worldCellSizeMeters);
            return key.axisIndex switch
            {
                0 => new Vector3(key.uIndex * spacing, key.sliceIndex * cellSize, key.vIndex * spacing),
                1 => new Vector3(key.sliceIndex * cellSize, key.vIndex * spacing, key.uIndex * spacing),
                _ => new Vector3(key.uIndex * spacing, key.vIndex * spacing, key.sliceIndex * cellSize),
            };
        }

        private void UpdateDisplayState(ref CellState cell)
        {
            if (!cell.valid)
                return;

            if (cell.displayNormal.sqrMagnitude <= 1e-6f)
                cell.displayNormal = cell.normal.sqrMagnitude > 1e-6f ? cell.normal : Vector3.up;
            if (cell.relocatingNormal.sqrMagnitude <= 1e-6f)
                cell.relocatingNormal = cell.normal.sqrMagnitude > 1e-6f ? cell.normal : Vector3.up;

            int freezeThreshold = Mathf.Max(minStableHits, freezeAfterStableHits);
            float unlockDistance = Mathf.Max(0.001f, displayUnlockDistanceMeters);
            float displayDelta = Vector3.Distance(cell.displayWorldPos, cell.worldPos);

            if (cell.stableHits < freezeThreshold)
            {
                cell.displayLockState = CellState.DisplayLockState.Unlocked;
                cell.unlockOutlierFrames = 0;
                cell.relockStableFrames = 0;
                cell.displayWorldPos = Vector3.Lerp(cell.displayWorldPos, cell.worldPos, Mathf.Clamp01(positionBlend));
                Vector3 blendedNormal = Vector3.Lerp(cell.displayNormal, cell.normal, Mathf.Clamp01(normalBlend));
                cell.displayNormal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
                cell.relocatingWorldPos = cell.displayWorldPos;
                cell.relocatingNormal = cell.displayNormal;
                return;
            }

            switch (cell.displayLockState)
            {
                case CellState.DisplayLockState.Unlocked:
                    cell.displayWorldPos = cell.worldPos;
                    cell.displayNormal = cell.normal.sqrMagnitude > 1e-6f ? cell.normal.normalized : Vector3.up;
                    cell.relocatingWorldPos = cell.displayWorldPos;
                    cell.relocatingNormal = cell.displayNormal;
                    cell.displayLockState = CellState.DisplayLockState.Locked;
                    cell.unlockOutlierFrames = 0;
                    cell.relockStableFrames = 0;
                    break;

                case CellState.DisplayLockState.Locked:
                    if (displayDelta > unlockDistance)
                    {
                        cell.unlockOutlierFrames++;
                        if (cell.unlockOutlierFrames >= Mathf.Max(1, unlockAfterOutlierFrames))
                        {
                            cell.displayLockState = CellState.DisplayLockState.Relocating;
                            cell.unlockOutlierFrames = 0;
                            cell.relockStableFrames = 0;
                            cell.relocatingWorldPos = cell.worldPos;
                            cell.relocatingNormal = cell.normal.sqrMagnitude > 1e-6f ? cell.normal.normalized : Vector3.up;
                        }
                    }
                    else
                    {
                        cell.unlockOutlierFrames = 0;
                    }
                    break;

                case CellState.DisplayLockState.Relocating:
                    cell.relocatingWorldPos = Vector3.Lerp(
                        cell.relocatingWorldPos,
                        cell.worldPos,
                        Mathf.Clamp01(relocatingDisplayPositionBlend));
                    Vector3 relocatingNormal = Vector3.Lerp(
                        cell.relocatingNormal,
                        cell.normal,
                        Mathf.Clamp01(relocatingDisplayNormalBlend));
                    cell.relocatingNormal = relocatingNormal.sqrMagnitude > 1e-6f ? relocatingNormal.normalized : Vector3.up;

                    float relocationDelta = Vector3.Distance(cell.relocatingWorldPos, cell.worldPos);
                    if (relocationDelta <= unlockDistance * 0.35f)
                    {
                        cell.relockStableFrames++;
                        if (cell.relockStableFrames >= Mathf.Max(1, relockAfterStableFrames))
                        {
                            cell.displayWorldPos = cell.relocatingWorldPos;
                            cell.displayNormal = cell.relocatingNormal;
                            cell.displayLockState = CellState.DisplayLockState.Locked;
                            cell.relockStableFrames = 0;
                            cell.unlockOutlierFrames = 0;
                        }
                    }
                    else
                    {
                        cell.relockStableFrames = 0;
                    }
                    break;
            }
        }
    }
}

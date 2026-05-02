using System.Collections.Generic;
using Meta.XR.EnvironmentDepth;
using Meta.XR.Samples;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using System.Linq;
using System.Reflection;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsDepthProbeRowRenderer : MonoBehaviour
    {
        private const int CopyTextureSize = 128;
        private const int NumEyes = 2;
        private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        private static readonly int EnvironmentDepthTextureSizeId = Shader.PropertyToID("_EnvironmentDepthTextureSize");
        private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
        private static readonly int EnvironmentDepthInverseReprojectionMatricesId =
            Shader.PropertyToID("_EnvironmentDepthInverseReprojectionMatrices");
        private static readonly int EnvironmentDepthReprojectionMatricesId =
            Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
        private static readonly int CopiedDepthTextureId = Shader.PropertyToID("_CopiedDepthTexture");
        private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
        private static readonly int ColorId = Shader.PropertyToID("_Color");

        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private Camera sampleCamera;

        [Header("Row Sampling")]
        [SerializeField]
        private int sampleCount = 24;

        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportY = 0.56f;

        [SerializeField]
        private float viewportWidth = 0.82f;

        [SerializeField]
        private float sampleIntervalSeconds = 0.04f;

        [Header("Depth Snapshot")]
        [SerializeField]
        private bool useDepthSnapshot = true;

        [SerializeField]
        private bool useScanCoverSnapshot = true;

        [SerializeField]
        private bool showScanCoverDepthGridSnapshot = true;

        [SerializeField]
        private bool showDepthProbeRowPreview = true;

        [SerializeField]
        private bool showDepthProbeRowNodes = true;

        [SerializeField]
        private bool syncVisibilityToScanCoverPreview = true;

        [SerializeField]
        private ComputeShader scanCoverPreprocessorComputeShader;

        [SerializeField]
        private bool scanCoverUseRowMatching = true;

        [SerializeField]
        private float scanCoverRayMatchDistanceMeters = 0.18f;

        [SerializeField]
        private bool enforceRowForwardContinuity = true;

        [SerializeField]
        private float maxRowForwardStepMeters = 0.34f;

        [SerializeField]
        private bool clearMemoryWhenScanCoverFrameChanges = false;

        [SerializeField]
        private bool scanCoverTargetsStartAnchored = true;

        [SerializeField]
        private bool hideHitsWithoutCurrentSnapshotTarget = true;

        [SerializeField]
        private bool manualSnapshotControl = true;

        [SerializeField]
        private bool freezeSnapshotUpdates = true;

        [SerializeField]
        private bool captureSnapshotNow;

        [SerializeField]
        private float snapshotRefreshIntervalSeconds = 0.18f;

        [SerializeField]
        private float snapshotStableMinSeconds = 0.12f;

        [SerializeField]
        private float minLinearDepthMeters = 0.2f;

        [SerializeField]
        private float maxLinearDepthMeters = 3.5f;

        [Header("Display")]
        [SerializeField]
        private float cubeScaleMeters = 0.03f;

        [SerializeField]
        private bool showConnections = true;

        [SerializeField]
        private float connectionThicknessMeters = 0.008f;

        [SerializeField]
        private float maxHorizontalConnectionDistanceMeters = 0.28f;

        [SerializeField]
        private float maxVerticalConnectionDistanceMeters = 0.24f;

        [SerializeField]
        private float maxDiagonalConnectionDistanceMeters = 0.32f;

        [SerializeField]
        private float maxVisibleTargetGapMeters = 0.22f;

        [SerializeField]
        private float targetCorrectionBlend = 0.35f;

        [SerializeField]
        private float committedFreezeSeconds = 0.9f;

        [SerializeField]
        private float challengeDistanceMeters = 0.12f;

        [SerializeField]
        private int challengeAfterFrames = 3;

        [SerializeField]
        private int targetCommitAfterFrames = 3;

        [SerializeField]
        private float targetCommitConsistencyMeters = 0.08f;

        [SerializeField]
        private float maxTargetCorrectionDistanceMeters = 0.18f;

        [SerializeField]
        private float maxRetainedTargetGapMeters = 0.35f;

        [SerializeField]
        private int targetMismatchDropAfterFrames = 3;

        [SerializeField]
        private bool showDiagnostics = true;

        [SerializeField]
        private bool diagnosticTargetsOnly = true;

        [SerializeField]
        private float diagnosticTargetScaleMeters = 0.018f;

        [SerializeField]
        private float diagnosticLinkThicknessMeters = 0.004f;

        [SerializeField]
        private bool overrideWorldHeight = false;

        [SerializeField]
        private float worldHeightY = 1.15f;

        [Header("Measured Row Height")]
        [SerializeField]
        private bool useMeasuredRowHeight = false;

        [SerializeField]
        private float measuredRowHeightViewportX = 0.5f;

        [SerializeField]
        private float measuredRowHeightViewportY = 0.5f;

        [SerializeField]
        private float measuredRowHeightMatchRadiusMeters = 0.24f;

        [SerializeField, Range(0f, 1f)]
        private float measuredRowHeightBlend = 1f;

        [Header("Height Slice Contour")]
        [SerializeField]
        private bool showHeightSliceContour = false;

        [SerializeField]
        private float heightSliceHalfMeters = 0.08f;

        [SerializeField]
        private float heightSliceContourSpacingMeters = 0.12f;

        [SerializeField]
        private float heightSliceMaxLinkDistanceMeters = 0.28f;

        [SerializeField]
        private float heightSliceMaxMeshEdgeMeters = 0.65f;

        [SerializeField]
        private float heightSliceContourThicknessMeters = 0.006f;

        [SerializeField]
        private Color heightSliceContourColor = new(0.08f, 0.85f, 1f, 0.95f);

        [SerializeField]
        private bool preserveHeightSliceContourSnapshots = true;

        [SerializeField, Min(1)]
        private int heightSliceMaxContourPoints = 96;

        [Header("Vertical Plane Grid")]
        [SerializeField]
        private bool showVerticalPlaneGrid = false;

        [SerializeField, Min(2)]
        private int verticalPlaneGridColumns = 10;

        [SerializeField, Min(2)]
        private int verticalPlaneGridRows = 20;

        [SerializeField, Min(0.02f)]
        private float verticalPlaneGridCellSizeMeters = 0.12f;

        [SerializeField, Min(0.1f)]
        private float verticalPlaneGridDistanceMeters = 3.0f;

        [SerializeField]
        private float verticalPlaneGridCenterOffsetXMeters = 0f;

        [SerializeField]
        private float verticalPlaneGridCenterOffsetYMeters = 0f;

        [SerializeField]
        private Color verticalPlaneGridColor = new(0.20f, 1.0f, 0.85f, 1f);

        [SerializeField]
        private Color verticalPlaneGridBaseYColor = new(1.0f, 0.85f, 0.15f, 1f);

        [SerializeField]
        private bool showBaseYPlane = false;

        [SerializeField, Min(0.1f)]
        private float baseYPlaneWidthMeters = 3.0f;

        [SerializeField, Min(0.1f)]
        private float baseYPlaneDepthMeters = 3.0f;

        [SerializeField]
        private float baseYPlaneCenterOffsetForwardMeters = 1.5f;

        [SerializeField]
        private Color baseYPlaneColor = new(1.0f, 0.85f, 0.15f, 0.18f);

        [SerializeField]
        private bool lockBaseYPlaneToResolvedHeight = false;

        [SerializeField]
        private bool followViewPitchForDisplayHeight = true;

        [SerializeField]
        private int additionalLowerRows = 2;

        [SerializeField]
        private float lowerRowStepMeters = 0.18f;

        [SerializeField]
        private float horizontalSpacingMeters = 0.12f;

        [SerializeField]
        private bool snapDisplayToHorizontalSlots = true;

        [SerializeField]
        private float forwardBandSpacingMeters = 0.18f;

        [SerializeField]
        private float verticalBandSpacingMeters = 0.10f;

        [SerializeField]
        private float holdSeconds = 12f;

        [SerializeField]
        private float anchoredHoldSeconds = 90f;

        [SerializeField]
        private float observedPositionBlend = 0.28f;

        [SerializeField]
        private int anchorAfterHits = 4;

        [SerializeField]
        private float anchorUnlockDistanceMeters = 0.10f;

        [SerializeField]
        private int anchorUnlockAfterOutlierFrames = 3;

        [SerializeField]
        private float anchoredCorrectionBlend = 0.18f;

        [SerializeField]
        private float anchoredSettleDistanceMeters = 0.03f;

        [SerializeField]
        private float anchoredSettleBlend = 0.28f;

        [SerializeField]
        private float unsettledAnchoredCleanupSeconds = 6f;

        [Header("Pose Stabilization")]
        [SerializeField]
        private float motionFreezePositionMeters = 0.025f;

        [SerializeField]
        private float motionFreezeAngleDegrees = 2.5f;

        [SerializeField]
        private float poseStableDelaySeconds = 0.2f;

        private Color cubeColor = new(0.82f, 0.9f, 0.99f, 0.95f);
        private Color diagnosticTargetColor = new(1f, 0.45f, 0.2f, 0.95f);
        private Color diagnosticLinkColor = new(0.2f, 1f, 0.85f, 0.9f);

        [SerializeField]
        private bool debugLog;

        private struct ProbeKey
        {
            public int rowLayerIndex;
            public int horizontalIndex;
            public int forwardIndex;
            public int heightIndex;
        }

        private struct VisibleSlotKey
        {
            public int rowLayerIndex;
            public int horizontalIndex;
        }

        private struct SnapshotHit
        {
            public ProbeKey key;
            public Vector3 position;
        }

        private struct RowCandidate
        {
            public bool valid;
            public Vector3 worldPos;
            public float alongForward;
        }

        private struct HeightSliceContourSegment
        {
            public Vector3 from;
            public Vector3 to;
            public float centerAlongRight;
        }

        private struct HeightSliceSurfaceTriangle
        {
            public Vector3 a;
            public Vector3 b;
            public Vector3 c;
        }

        private enum ProbeState
        {
            Targeting,
            Committed,
            Frozen,
            Challenged
        }

        private struct ProbeHit
        {
            public bool valid;
            public ProbeKey key;
            public Vector3 observedPos;
            public Vector3 anchoredPos;
            public Vector3 pendingTargetPos;
            public float lastSeenTime;
            public int supportCount;
            public bool anchored;
            public ProbeState state;
            public float frozenUntilTime;
            public int challengedFrames;
            public int outlierFrames;
            public int pendingTargetFrames;
            public int targetMismatchFrames;
        }

        private readonly List<GameObject> m_nodeObjects = new();
        private readonly List<GameObject> m_connectionObjects = new();
        private readonly List<GameObject> m_heightSliceContourObjects = new();
        private readonly List<GameObject> m_heightSliceContourMeshObjects = new();
        private readonly List<Mesh> m_heightSliceContourMeshes = new();
        private readonly List<GameObject> m_diagnosticTargetObjects = new();
        private readonly List<GameObject> m_diagnosticLinkObjects = new();
        private readonly List<ProbeHit> m_hits = new();
        private readonly List<int> m_visibleHitIndices = new();
        private readonly Dictionary<VisibleSlotKey, Vector3> m_currentSlotTargets = new();
        private readonly Dictionary<VisibleSlotKey, int> m_currentSlotTargetCounts = new();
        private readonly Dictionary<VisibleSlotKey, int> m_selectedHitBySlot = new();
        private readonly Dictionary<VisibleSlotKey, int> m_visibleHitIndexBySlot = new();
        private readonly List<(int from, int to)> m_connectionPairs = new();
        private readonly List<HeightSliceContourSegment> m_heightSliceContourSegments = new();
        private readonly List<HeightSliceSurfaceTriangle> m_heightSliceSurfaceTriangles = new();
        private readonly List<SnapshotHit> m_rawSnapshotHits = new();
        private readonly List<SnapshotHit> m_snapshotHits = new();
        private bool m_heightSliceContourObjectsDirty;
        private ComputeShader m_copyShader;
        private ComputeBuffer m_computeBuffer;
        private NativeArray<float> m_depthPixels;
        private NativeArray<float> m_gpuReadbackBuffer;
        private AsyncGPUReadbackRequest m_pendingReadback;
        private bool m_hasPendingReadback;
        private float m_nextSampleTime;
        private Material m_runtimeMaterial;
        private Material m_diagnosticTargetMaterial;
        private Material m_diagnosticLinkMaterial;
        private Material m_heightSliceContourMaterial;
        private bool m_hasRowFrame;
        private Vector3 m_rowOrigin;
        private Vector3 m_rowRight;
        private Vector3 m_rowForward;
        private bool m_hasLastPose;
        private Vector3 m_lastCameraPosition;
        private Quaternion m_lastCameraRotation;
        private float m_lastMotionTime;
        private float m_lastSnapshotTime = float.NegativeInfinity;
        private bool m_snapshotBootstrapPending;
        private ScanCoverDepthPreprocessor m_scanCoverPreprocessor;
        private ScanCoverDepthGridPointCloud m_scanCoverDepthGridPointCloud;
        private readonly List<SnapshotHit> m_scanCoverSnapshotHits = new();
        private float m_nextScanCoverCaptureTime;
        private int m_lastAppliedScanCoverFrameIndex = -1;
        private bool m_scanCoverCaptureRequested;
        private bool m_scanCoverCaptureApplyPending;
        private int m_scanCoverCaptureRequestFrameIndex = -1;
        private bool m_hasMeasuredRowHeight;
        private float m_measuredRowHeightY;
        private GameObject m_verticalPlaneGridObject;
        private Mesh m_verticalPlaneGridMesh;
        private Material m_verticalPlaneGridMaterial;
        private GameObject m_baseYPlaneObject;
        private Mesh m_baseYPlaneMesh;
        private Material m_baseYPlaneMaterial;

        private void OnEnable()
        {
            ResolveRefs();
            EnsureResources();
            EnsureRowFrame();
            CaptureCurrentPose(Time.unscaledTime);
            m_snapshotBootstrapPending = !manualSnapshotControl;
            m_scanCoverCaptureApplyPending = false;
            m_scanCoverCaptureRequestFrameIndex = -1;
            m_nextSampleTime = Time.unscaledTime;
            m_nextScanCoverCaptureTime = Time.unscaledTime;
        }

        private void OnDisable()
        {
            DisposeResources();
            SetNodeObjectsActive(false);
            SetConnectionObjectsActive(false);
            SetHeightSliceContourObjectsActive(false);
            SetDiagnosticObjectsActive(false);
            ClearVerticalPlaneGridObject();
        }

        private void OnDestroy()
        {
            DisposeResources();
            ClearHeightSliceContourMeshObjects();
            ClearVerticalPlaneGridObject();
        }

        private void Update()
        {
            ResolveRefs();
            UpdatePoseMotionState();
            if (useScanCoverSnapshot)
                UpdateScanCoverSnapshotPipeline();
            else
                UpdatePendingReadback();

            SyncVerticalPlaneGrid();

            if (!ShouldRenderProbePreview())
            {
                SetNodeObjectsActive(false);
                SetConnectionObjectsActive(false);
                SetHeightSliceContourObjectsActive(false);
                SetDiagnosticObjectsActive(false);
                SetVerticalPlaneGridActive(false);
                m_heightSliceContourObjectsDirty = true;
                if (useScanCoverSnapshot)
                    return;
            }

            if (sampleCamera == null || environmentDepthManager == null)
                return;

            if (useScanCoverSnapshot)
            {
                if (ShouldUseStaticScanCoverContourOnlyMode())
                {
                    SetNodeObjectsActive(false);
                    SetConnectionObjectsActive(false);
                    SetDiagnosticObjectsActive(false);
                    SyncHeightSliceContourObjectsIfDirty();
                    return;
                }

                ProcessCurrentSnapshotHits(Time.unscaledTime, IsPoseStable(Time.unscaledTime));
                return;
            }

            if (Time.unscaledTime < m_nextSampleTime || m_hasPendingReadback)
                return;

            m_nextSampleTime = Time.unscaledTime + Mathf.Max(0.02f, sampleIntervalSeconds);
            RequestDepthCopy();
        }

        private void ResolveRefs()
        {
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);
            if (!useScanCoverSnapshot)
                return;

            if (m_scanCoverPreprocessor == null)
                m_scanCoverPreprocessor = GetComponent<ScanCoverDepthPreprocessor>();
            if (m_scanCoverPreprocessor == null)
                m_scanCoverPreprocessor = FindAnyObjectByType<ScanCoverDepthPreprocessor>(FindObjectsInactive.Include);
            if (m_scanCoverPreprocessor == null)
                m_scanCoverPreprocessor = gameObject.AddComponent<ScanCoverDepthPreprocessor>();

            if (m_scanCoverDepthGridPointCloud == null)
                m_scanCoverDepthGridPointCloud = GetComponent<ScanCoverDepthGridPointCloud>();
            if (m_scanCoverDepthGridPointCloud == null)
                m_scanCoverDepthGridPointCloud = FindAnyObjectByType<ScanCoverDepthGridPointCloud>(FindObjectsInactive.Include);
            if (m_scanCoverDepthGridPointCloud == null)
                m_scanCoverDepthGridPointCloud = gameObject.AddComponent<ScanCoverDepthGridPointCloud>();

            if (scanCoverPreprocessorComputeShader == null)
                scanCoverPreprocessorComputeShader = Resources.Load<ComputeShader>("ScanCoverDepthPreprocessor");

            ConfigureScanCoverComponents();
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

            if (m_runtimeMaterial == null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                    shader = Shader.Find("Unlit/Color");
                if (shader != null)
                {
                    m_runtimeMaterial = new Material(shader)
                    {
                        name = "DepthEffectsDepthProbeRowRenderer_Runtime"
                    };
                    if (m_runtimeMaterial.HasProperty("_BaseColor"))
                        m_runtimeMaterial.SetColor("_BaseColor", cubeColor);
                    if (m_runtimeMaterial.HasProperty("_Color"))
                        m_runtimeMaterial.SetColor("_Color", cubeColor);
                    if (m_runtimeMaterial.HasProperty("_Cull"))
                        m_runtimeMaterial.SetFloat("_Cull", (float)CullMode.Off);
                }
            }

            if (m_diagnosticTargetMaterial == null && m_runtimeMaterial != null)
            {
                m_diagnosticTargetMaterial = new Material(m_runtimeMaterial)
                {
                    name = "DepthEffectsDepthProbeRowRenderer_DiagnosticTarget"
                };
                if (m_diagnosticTargetMaterial.HasProperty("_BaseColor"))
                    m_diagnosticTargetMaterial.SetColor("_BaseColor", diagnosticTargetColor);
                if (m_diagnosticTargetMaterial.HasProperty("_Color"))
                    m_diagnosticTargetMaterial.SetColor("_Color", diagnosticTargetColor);
            }

            if (m_diagnosticLinkMaterial == null && m_runtimeMaterial != null)
            {
                m_diagnosticLinkMaterial = new Material(m_runtimeMaterial)
                {
                    name = "DepthEffectsDepthProbeRowRenderer_DiagnosticLink"
                };
                if (m_diagnosticLinkMaterial.HasProperty("_BaseColor"))
                    m_diagnosticLinkMaterial.SetColor("_BaseColor", diagnosticLinkColor);
                if (m_diagnosticLinkMaterial.HasProperty("_Color"))
                    m_diagnosticLinkMaterial.SetColor("_Color", diagnosticLinkColor);
            }

            if (m_heightSliceContourMaterial == null)
            {
                Shader shader = Resources.Load<Shader>("DepthEffectsSolidContour");
                if (shader == null)
                    shader = Shader.Find("MRMotifs/DepthEffects/SolidContour");
                if (shader == null)
                    shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                    shader = Shader.Find("Unlit/Color");
                if (shader == null)
                    return;

                m_heightSliceContourMaterial = new Material(shader)
                {
                    name = "DepthEffectsDepthProbeRowRenderer_HeightSliceContour"
                };
                m_heightSliceContourMaterial.renderQueue = 3050;
                ApplyContourMaterialColor();
            }

            if (m_verticalPlaneGridMaterial == null && m_runtimeMaterial != null)
            {
                m_verticalPlaneGridMaterial = new Material(m_runtimeMaterial)
                {
                    name = "DepthEffectsDepthProbeRowRenderer_VerticalPlaneGrid"
                };
                if (m_verticalPlaneGridMaterial.HasProperty("_BaseColor"))
                    m_verticalPlaneGridMaterial.SetColor("_BaseColor", verticalPlaneGridColor);
                if (m_verticalPlaneGridMaterial.HasProperty("_Color"))
                    m_verticalPlaneGridMaterial.SetColor("_Color", verticalPlaneGridColor);
            }

            if (m_baseYPlaneMaterial == null && m_runtimeMaterial != null)
            {
                m_baseYPlaneMaterial = new Material(m_runtimeMaterial)
                {
                    name = "DepthEffectsDepthProbeRowRenderer_BaseYPlane"
                };
                m_baseYPlaneMaterial.renderQueue = 3000;
                if (m_baseYPlaneMaterial.HasProperty("_BaseColor"))
                    m_baseYPlaneMaterial.SetColor("_BaseColor", baseYPlaneColor);
                if (m_baseYPlaneMaterial.HasProperty("_Color"))
                    m_baseYPlaneMaterial.SetColor("_Color", baseYPlaneColor);
                if (m_baseYPlaneMaterial.HasProperty("_Cull"))
                    m_baseYPlaneMaterial.SetFloat("_Cull", (float)CullMode.Off);
            }
        }

        private void ApplyContourMaterialColor()
        {
            if (m_heightSliceContourMaterial == null)
                return;

            Color solid = heightSliceContourColor;
            solid.a = 1f;
            if (m_heightSliceContourMaterial.HasProperty(BaseColorId))
                m_heightSliceContourMaterial.SetColor(BaseColorId, solid);
            if (m_heightSliceContourMaterial.HasProperty(ColorId))
                m_heightSliceContourMaterial.SetColor(ColorId, solid);
        }

        private void SyncVerticalPlaneGrid()
        {
            if (!showVerticalPlaneGrid || sampleCamera == null)
            {
                SetVerticalPlaneGridActive(false);
                return;
            }

            EnsureResources();

            Vector3 right = sampleCamera.transform.right;
            right.y = 0f;
            if (right.sqrMagnitude <= 1e-6f)
                right = Vector3.right;
            right.Normalize();

            Vector3 forward = sampleCamera.transform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude <= 1e-6f)
                forward = Vector3.forward;
            forward.Normalize();

            float halfWidth = Mathf.Max(0.01f, (Mathf.Max(2, verticalPlaneGridColumns) - 1) * verticalPlaneGridCellSizeMeters * 0.5f);
            float verticalHalf = Mathf.Max(0.01f, (Mathf.Max(2, verticalPlaneGridRows) - 1) * verticalPlaneGridCellSizeMeters * 0.5f);
            bool hasSharedDisplayHeight = TryGetSharedDisplayHeight(
                sampleCamera.transform.position,
                sampleCamera.transform.forward,
                out float sharedDisplayHeight);

            Vector3 center = sampleCamera.transform.position +
                             forward * Mathf.Max(0.1f, verticalPlaneGridDistanceMeters) +
                             right * verticalPlaneGridCenterOffsetXMeters;
            if (hasSharedDisplayHeight)
                center.y = sharedDisplayHeight;
            else
                center += Vector3.up * verticalPlaneGridCenterOffsetYMeters;

            Mesh mesh = BuildVerticalPlaneGridMesh(
                center,
                right,
                Vector3.up,
                halfWidth,
                verticalHalf,
                Mathf.Max(2, verticalPlaneGridColumns),
                Mathf.Max(2, verticalPlaneGridRows));

            if (mesh == null)
            {
                SetVerticalPlaneGridActive(false);
                return;
            }

            EnsureVerticalPlaneGridObject();

            MeshFilter filter = m_verticalPlaneGridObject.GetComponent<MeshFilter>();
            if (filter != null)
                filter.sharedMesh = mesh;

            if (m_verticalPlaneGridMesh != null)
            {
                if (Application.isPlaying)
                    Destroy(m_verticalPlaneGridMesh);
                else
                    DestroyImmediate(m_verticalPlaneGridMesh);
            }

            m_verticalPlaneGridMesh = mesh;
            SetVerticalPlaneGridActive(true);

            if (hasSharedDisplayHeight || TryGetBaseYVisualHeight(out sharedDisplayHeight))
                SyncBaseYPlane(sampleCamera.transform.position, sampleCamera.transform.forward, sharedDisplayHeight);
            else
                SetBaseYPlaneActive(false);
        }

        private Mesh BuildVerticalPlaneGridMesh(
            Vector3 center,
            Vector3 right,
            Vector3 upAxis,
            float halfWidth,
            float verticalHalf,
            int columns,
            int rows)
        {
            var vertices = new List<Vector3>((columns + rows) * 2);
            var indices = new List<int>((columns + rows) * 2);

            for (int column = 0; column < columns; column++)
            {
                float t = columns <= 1 ? 0.5f : column / (float)(columns - 1);
                Vector3 x = right * Mathf.Lerp(-halfWidth, halfWidth, t);
                AddVerticalPlaneGridLine(vertices, indices, center + x - upAxis * verticalHalf, center + x + upAxis * verticalHalf);
            }

            for (int row = 0; row < rows; row++)
            {
                float t = rows <= 1 ? 0.5f : row / (float)(rows - 1);
                Vector3 y = upAxis * Mathf.Lerp(-verticalHalf, verticalHalf, t);
                AddVerticalPlaneGridLine(vertices, indices, center - right * halfWidth + y, center + right * halfWidth + y);
            }

            if (vertices.Count <= 0 || indices.Count <= 0)
                return null;

            Mesh mesh = new Mesh
            {
                name = "DepthProbeVerticalPlaneGridMesh"
            };
            if (vertices.Count > 65535)
                mesh.indexFormat = IndexFormat.UInt32;
            mesh.SetVertices(vertices);
            mesh.SetIndices(indices, MeshTopology.Lines, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

        private void AddVerticalPlaneGridLine(List<Vector3> vertices, List<int> indices, Vector3 from, Vector3 to)
        {
            int start = vertices.Count;
            vertices.Add(transform.InverseTransformPoint(from));
            vertices.Add(transform.InverseTransformPoint(to));
            indices.Add(start);
            indices.Add(start + 1);
        }

        private void EnsureVerticalPlaneGridObject()
        {
            if (m_verticalPlaneGridObject != null)
                return;

            m_verticalPlaneGridObject = new GameObject("DepthProbeVerticalPlaneGrid");
            m_verticalPlaneGridObject.transform.SetParent(transform, worldPositionStays: false);
            m_verticalPlaneGridObject.transform.localPosition = Vector3.zero;
            m_verticalPlaneGridObject.transform.localRotation = Quaternion.identity;
            m_verticalPlaneGridObject.transform.localScale = Vector3.one;

            MeshFilter filter = m_verticalPlaneGridObject.AddComponent<MeshFilter>();
            filter.sharedMesh = null;

            MeshRenderer renderer = m_verticalPlaneGridObject.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = m_verticalPlaneGridMaterial != null ? m_verticalPlaneGridMaterial : m_runtimeMaterial;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        }

        private void SyncBaseYPlane(Vector3 cameraPosition, Vector3 forward, float baseHeight)
        {
            if (!showBaseYPlane)
            {
                SetBaseYPlaneActive(false);
                return;
            }

            Vector3 horizontalForward = forward;
            horizontalForward.y = 0f;
            if (horizontalForward.sqrMagnitude <= 1e-6f)
                horizontalForward = Vector3.forward;
            horizontalForward.Normalize();

            Vector3 center = new Vector3(cameraPosition.x, baseHeight, cameraPosition.z) +
                             horizontalForward * Mathf.Max(0f, baseYPlaneCenterOffsetForwardMeters);
            Mesh mesh = BuildBaseYPlaneMesh(center, baseYPlaneWidthMeters, baseYPlaneDepthMeters);
            if (mesh == null)
            {
                SetBaseYPlaneActive(false);
                return;
            }

            EnsureBaseYPlaneObject();

            MeshFilter filter = m_baseYPlaneObject.GetComponent<MeshFilter>();
            if (filter != null)
                filter.sharedMesh = mesh;

            if (m_baseYPlaneMesh != null)
            {
                if (Application.isPlaying)
                    Destroy(m_baseYPlaneMesh);
                else
                    DestroyImmediate(m_baseYPlaneMesh);
            }

            m_baseYPlaneMesh = mesh;
            SetBaseYPlaneActive(true);
        }

        private bool TryGetBaseYVisualHeight(out float baseHeight)
        {
            if (sampleCamera != null &&
                TryGetSharedDisplayHeight(
                    sampleCamera.transform.position,
                    sampleCamera.transform.forward,
                    out baseHeight))
            {
                return true;
            }

            if (!lockBaseYPlaneToResolvedHeight)
            {
                baseHeight = worldHeightY;
                return true;
            }

            if (TryGetRowBaseHeight(out baseHeight))
                return true;

            baseHeight = worldHeightY;
            return true;
        }

        private bool TryGetSharedDisplayHeight(
            Vector3 cameraPosition,
            Vector3 cameraForward,
            out float displayHeight)
        {
            return TryGetDisplayFollowHeight(
                cameraPosition,
                cameraForward,
                verticalPlaneGridDistanceMeters,
                verticalPlaneGridCenterOffsetYMeters,
                out displayHeight);
        }

        private bool TryGetDisplayFollowHeight(
            Vector3 cameraPosition,
            Vector3 cameraForward,
            float referenceDistance,
            float yOffset,
            out float displayHeight)
        {
            if (!followViewPitchForDisplayHeight)
            {
                displayHeight = 0f;
                return false;
            }

            Vector3 forward = cameraForward;
            if (forward.sqrMagnitude <= 1e-6f)
                forward = Vector3.forward;
            forward.Normalize();

            displayHeight = cameraPosition.y + forward.y * Mathf.Max(0.1f, referenceDistance) + yOffset;
            return true;
        }

        private Mesh BuildBaseYPlaneMesh(Vector3 center, float width, float depth)
        {
            float halfWidth = Mathf.Max(0.05f, width * 0.5f);
            float halfDepth = Mathf.Max(0.05f, depth * 0.5f);
            var vertices = new List<Vector3>(4)
            {
                transform.InverseTransformPoint(center + new Vector3(-halfWidth, 0f, -halfDepth)),
                transform.InverseTransformPoint(center + new Vector3(halfWidth, 0f, -halfDepth)),
                transform.InverseTransformPoint(center + new Vector3(-halfWidth, 0f, halfDepth)),
                transform.InverseTransformPoint(center + new Vector3(halfWidth, 0f, halfDepth))
            };
            var indices = new List<int>(6) { 0, 2, 1, 2, 3, 1 };

            Mesh mesh = new Mesh
            {
                name = "DepthProbeBaseYPlaneMesh"
            };
            mesh.SetVertices(vertices);
            mesh.SetIndices(indices, MeshTopology.Triangles, 0);
            mesh.RecalculateBounds();
            mesh.RecalculateNormals();
            return mesh;
        }

        private void EnsureBaseYPlaneObject()
        {
            if (m_baseYPlaneObject != null)
                return;

            m_baseYPlaneObject = new GameObject("DepthProbeBaseYPlane");
            m_baseYPlaneObject.transform.SetParent(transform, worldPositionStays: false);
            m_baseYPlaneObject.transform.localPosition = Vector3.zero;
            m_baseYPlaneObject.transform.localRotation = Quaternion.identity;
            m_baseYPlaneObject.transform.localScale = Vector3.one;

            MeshFilter filter = m_baseYPlaneObject.AddComponent<MeshFilter>();
            filter.sharedMesh = null;

            MeshRenderer renderer = m_baseYPlaneObject.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = m_baseYPlaneMaterial != null ? m_baseYPlaneMaterial : m_verticalPlaneGridMaterial;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        }

        private void SetVerticalPlaneGridActive(bool active)
        {
            if (m_verticalPlaneGridObject != null)
                m_verticalPlaneGridObject.SetActive(active);
            if (!active)
                SetBaseYPlaneActive(false);
        }

        private void SetBaseYPlaneActive(bool active)
        {
            if (m_baseYPlaneObject != null)
                m_baseYPlaneObject.SetActive(active);
        }

        private void ClearVerticalPlaneGridObject()
        {
            if (m_verticalPlaneGridMesh != null)
            {
                if (Application.isPlaying)
                    Destroy(m_verticalPlaneGridMesh);
                else
                    DestroyImmediate(m_verticalPlaneGridMesh);
                m_verticalPlaneGridMesh = null;
            }

            if (m_verticalPlaneGridObject != null)
            {
                if (Application.isPlaying)
                    Destroy(m_verticalPlaneGridObject);
                else
                    DestroyImmediate(m_verticalPlaneGridObject);
                m_verticalPlaneGridObject = null;
            }

            if (m_verticalPlaneGridMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(m_verticalPlaneGridMaterial);
                else
                    DestroyImmediate(m_verticalPlaneGridMaterial);
                m_verticalPlaneGridMaterial = null;
            }

            if (m_baseYPlaneMesh != null)
            {
                if (Application.isPlaying)
                    Destroy(m_baseYPlaneMesh);
                else
                    DestroyImmediate(m_baseYPlaneMesh);
                m_baseYPlaneMesh = null;
            }

            if (m_baseYPlaneObject != null)
            {
                if (Application.isPlaying)
                    Destroy(m_baseYPlaneObject);
                else
                    DestroyImmediate(m_baseYPlaneObject);
                m_baseYPlaneObject = null;
            }

            if (m_baseYPlaneMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(m_baseYPlaneMaterial);
                else
                    DestroyImmediate(m_baseYPlaneMaterial);
                m_baseYPlaneMaterial = null;
            }
        }

        private void EnsureRowFrame()
        {
            if (m_hasRowFrame || sampleCamera == null)
                return;

            Vector3 right = sampleCamera.transform.right;
            right.y = 0f;
            if (right.sqrMagnitude <= 1e-6f)
                right = Vector3.right;
            m_rowRight = right.normalized;

            Vector3 forward = sampleCamera.transform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude <= 1e-6f)
                forward = Vector3.forward;
            m_rowForward = forward.normalized;

            m_rowOrigin = sampleCamera.transform.position;
            if (overrideWorldHeight)
                m_rowOrigin.y = worldHeightY;

            m_hasRowFrame = true;
        }

        private void CaptureCurrentPose(float now)
        {
            if (sampleCamera == null)
                return;

            m_hasLastPose = true;
            m_lastCameraPosition = sampleCamera.transform.position;
            m_lastCameraRotation = sampleCamera.transform.rotation;
            m_lastMotionTime = now;
        }

        private void UpdatePoseMotionState()
        {
            if (sampleCamera == null)
                return;

            float now = Time.unscaledTime;
            Vector3 currentPosition = sampleCamera.transform.position;
            Quaternion currentRotation = sampleCamera.transform.rotation;
            if (!m_hasLastPose)
            {
                CaptureCurrentPose(now);
                return;
            }

            float movedMeters = Vector3.Distance(currentPosition, m_lastCameraPosition);
            float rotatedDegrees = Quaternion.Angle(currentRotation, m_lastCameraRotation);
            if (movedMeters > Mathf.Max(0.001f, motionFreezePositionMeters) ||
                rotatedDegrees > Mathf.Max(0.1f, motionFreezeAngleDegrees))
            {
                m_lastMotionTime = now;
            }

            m_lastCameraPosition = currentPosition;
            m_lastCameraRotation = currentRotation;
        }

        private bool IsPoseStable(float now)
        {
            return (now - m_lastMotionTime) >= Mathf.Max(0.01f, poseStableDelaySeconds);
        }

        private bool ShouldRenderProbePreview()
        {
            if (!useScanCoverSnapshot)
                return true;
            if (!showDepthProbeRowPreview)
                return false;
            if (!syncVisibilityToScanCoverPreview)
                return true;

            return m_scanCoverDepthGridPointCloud == null || m_scanCoverDepthGridPointCloud.PreviewVisible;
        }

        private void DisposeResources()
        {
            if (!CanReleaseDepthCopyResources())
                return;

            if (m_computeBuffer != null)
            {
                m_computeBuffer.Dispose();
                m_computeBuffer = null;
            }

            if (m_depthPixels.IsCreated)
                m_depthPixels.Dispose();
            if (m_gpuReadbackBuffer.IsCreated)
                m_gpuReadbackBuffer.Dispose();

            m_hasPendingReadback = false;
        }

        private bool CanReleaseDepthCopyResources()
        {
            if (!m_hasPendingReadback)
                return true;

            if (!m_pendingReadback.done)
                return false;

            m_hasPendingReadback = false;
            return true;
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
                return;

            (m_depthPixels, m_gpuReadbackBuffer) = (m_gpuReadbackBuffer, m_depthPixels);
            ConsumeDepthSamples();
        }

        private void ConsumeDepthSamples()
        {
            EnsureRowFrame();
            m_rawSnapshotHits.Clear();

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

            Matrix4x4 inverseMatrix = inverseMatrices[0];
            int validCount = 0;
            float now = Time.unscaledTime;
            bool poseStable = IsPoseStable(now);
            int probeCount = Mathf.Max(1, sampleCount);

            for (int column = 0; column < probeCount; column++)
            {
                float u = probeCount > 1 ? (float)column / (probeCount - 1) : 0.5f;
                float viewportX = viewportCenterX + (u - 0.5f) * viewportWidth;
                Vector2Int texCoord = ViewportToDepthCoord(viewportX, viewportY);
                float linearDepth = SampleLinearDepth(texCoord, 0);
                if (linearDepth < minLinearDepthMeters || linearDepth > maxLinearDepthMeters)
                    continue;

                if (!TryReconstructWorld(texCoord, linearDepth, inverseMatrix, out Vector3 worldPos))
                    continue;

                int totalRows = Mathf.Max(0, additionalLowerRows) + 1;
                for (int rowLayerIndex = 0; rowLayerIndex < totalRows; rowLayerIndex++)
                {
                    if (TryBuildSnapshotHit(worldPos, rowLayerIndex, out SnapshotHit snapshotHit))
                        m_rawSnapshotHits.Add(snapshotHit);
                }
                validCount++;
            }

            UpdateDepthSnapshot(now, poseStable);
            ProcessCurrentSnapshotHits(now, poseStable);

            if (debugLog)
                Debug.Log($"[DepthEffectsDepthProbeRowRenderer] valid probes={validCount}/{probeCount}, hits={m_hits.Count}");
        }

        private void ProcessCurrentSnapshotHits(float now, bool poseStable)
        {
            m_currentSlotTargets.Clear();
            m_currentSlotTargetCounts.Clear();

            for (int i = 0; i < m_snapshotHits.Count; i++)
            {
                SnapshotHit snapshotHit = m_snapshotHits[i];
                RegisterCurrentSlotTarget(MakeVisibleSlotKey(snapshotHit.key), snapshotHit.position);
                UpsertHit(snapshotHit, now, poseStable);
            }

            if (!diagnosticTargetsOnly)
                ReconcileCurrentTargets(now, poseStable);
            UpdateVisibleHits(now, poseStable);

            if (!ShouldRenderProbePreview())
            {
                SetNodeObjectsActive(false);
                SetConnectionObjectsActive(false);
                SetHeightSliceContourObjectsActive(false);
                SetDiagnosticObjectsActive(false);
                return;
            }

            if (showDepthProbeRowNodes)
            {
                SyncNodeObjects();
                SyncConnectionObjects();
            }
            else
            {
                SetNodeObjectsActive(false);
                SetConnectionObjectsActive(false);
            }
            SyncHeightSliceContourObjectsIfDirty();
            SyncDiagnosticObjects(now);
        }

        private void UpdateDepthSnapshot(float now, bool poseStable)
        {
            bool hasSnapshot = m_snapshotHits.Count > 0;
            bool hasFreshRaw = m_rawSnapshotHits.Count > 0;

            if (!useDepthSnapshot)
            {
                m_snapshotHits.Clear();
                m_snapshotHits.AddRange(m_rawSnapshotHits);
                m_lastSnapshotTime = now;
                return;
            }

            if (manualSnapshotControl)
            {
                bool requestCapture = m_snapshotBootstrapPending || captureSnapshotNow;
                if (captureSnapshotNow)
                    captureSnapshotNow = false;

                if (requestCapture && hasFreshRaw)
                {
                    m_snapshotHits.Clear();
                    m_snapshotHits.AddRange(m_rawSnapshotHits);
                    m_lastSnapshotTime = now;
                    m_snapshotBootstrapPending = false;
                    return;
                }

                if (freezeSnapshotUpdates)
                    return;
            }

            bool allowRefresh = poseStable &&
                (now - m_lastMotionTime) >= Mathf.Max(0.01f, snapshotStableMinSeconds) &&
                (now - m_lastSnapshotTime) >= Mathf.Max(0.02f, snapshotRefreshIntervalSeconds);

            bool shouldBootstrap = !hasSnapshot && hasFreshRaw;
            if (!hasFreshRaw && hasSnapshot)
                return;

            if (!shouldBootstrap && !allowRefresh)
                return;

            m_snapshotHits.Clear();
            m_snapshotHits.AddRange(m_rawSnapshotHits);
            m_lastSnapshotTime = now;
            m_snapshotBootstrapPending = false;
        }

        private void UpdateScanCoverSnapshotPipeline()
        {
            if (m_scanCoverPreprocessor == null || m_scanCoverDepthGridPointCloud == null)
                return;

            float now = Time.unscaledTime;
            bool poseStable = IsPoseStable(now);
            bool requestCapture = false;

            if (manualSnapshotControl)
            {
                if (m_snapshotBootstrapPending || m_scanCoverCaptureRequested || captureSnapshotNow)
                    requestCapture = true;
            }
            else if (!freezeSnapshotUpdates && poseStable && now >= m_nextScanCoverCaptureTime)
            {
                requestCapture = true;
            }

            if (captureSnapshotNow)
                captureSnapshotNow = false;
            if (m_scanCoverCaptureRequested)
                m_scanCoverCaptureRequested = false;

            if (requestCapture)
            {
                m_scanCoverCaptureApplyPending = true;
                m_scanCoverCaptureRequestFrameIndex = m_scanCoverDepthGridPointCloud.FrameIndex;
                m_scanCoverPreprocessor.RefreshNow();
                m_scanCoverDepthGridPointCloud.RefreshNow();
                m_nextScanCoverCaptureTime = now + Mathf.Max(0.02f, snapshotRefreshIntervalSeconds);
            }

            if (manualSnapshotControl && !m_scanCoverCaptureApplyPending)
                return;

            if (!m_scanCoverDepthGridPointCloud.TryGetCurrentGridState(out ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot))
                return;
            if (snapshot == null || snapshot.entries == null || snapshot.entries.Length <= 0)
                return;
            if (manualSnapshotControl &&
                m_scanCoverCaptureApplyPending &&
                snapshot.frameIndex <= m_scanCoverCaptureRequestFrameIndex)
            {
                return;
            }
            if (snapshot.frameIndex == m_lastAppliedScanCoverFrameIndex)
                return;

            if (clearMemoryWhenScanCoverFrameChanges)
                ClearProbeMemory(clearHeightSliceContours: !ShouldPreserveHeightSliceContoursOnSnapshot());

            m_lastAppliedScanCoverFrameIndex = snapshot.frameIndex;
            bool meshOnlyMode = !showDepthProbeRowPreview && !showHeightSliceContour;
            if (!meshOnlyMode)
            {
                UpdateMeasuredRowHeight(snapshot);
                BuildHeightSliceContour(snapshot);
                m_scanCoverSnapshotHits.Clear();
                if (scanCoverUseRowMatching && sampleCamera != null)
                    BuildScanCoverMatchedRowHits(snapshot);
                else
                    BuildScanCoverFullSnapshotHits(snapshot);

                m_snapshotHits.Clear();
                m_snapshotHits.AddRange(m_scanCoverSnapshotHits);
            }
            else
            {
                m_scanCoverSnapshotHits.Clear();
                m_snapshotHits.Clear();
            }

            m_lastSnapshotTime = now;
            m_snapshotBootstrapPending = false;
            m_scanCoverCaptureApplyPending = false;
            m_scanCoverCaptureRequestFrameIndex = -1;

            if (debugLog)
                Debug.Log($"[DepthEffectsDepthProbeRowRenderer] Applied ScanCover snapshot frame={snapshot.frameIndex}, hits={m_snapshotHits.Count}, meshOnly={meshOnlyMode}");
        }

        private void BuildScanCoverFullSnapshotHits(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
        {
            for (int i = 0; i < snapshot.entries.Length; i++)
            {
                ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
                if (!entry.valid)
                    continue;

                int totalRows = Mathf.Max(0, additionalLowerRows) + 1;
                for (int rowLayerIndex = 0; rowLayerIndex < totalRows; rowLayerIndex++)
                {
                    if (TryBuildSnapshotHit(entry.worldPos, rowLayerIndex, out SnapshotHit hit))
                        m_scanCoverSnapshotHits.Add(hit);
                }
            }
        }

        private void BuildScanCoverMatchedRowHits(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
        {
            if (sampleCamera == null || snapshot.entries == null || snapshot.entries.Length <= 0)
                return;

            int probeCount = Mathf.Max(1, sampleCount);
            int totalRows = Mathf.Max(0, additionalLowerRows) + 1;
            float maxDepth = Mathf.Max(minLinearDepthMeters, maxLinearDepthMeters);
            float matchRadius = Mathf.Max(0.02f, scanCoverRayMatchDistanceMeters);
            RowCandidate[] candidates = new RowCandidate[probeCount];

            for (int column = 0; column < probeCount; column++)
            {
                float u = probeCount > 1 ? (float)column / (probeCount - 1) : 0.5f;
                float viewportX = viewportCenterX + (u - 0.5f) * viewportWidth;
                Ray sampleRay = sampleCamera.ViewportPointToRay(new Vector3(viewportX, viewportY, 0f));
                Vector3 rayDir = sampleRay.direction.normalized;
                Vector3 rayOrigin = sampleRay.origin;

                bool found = false;
                float bestScore = float.PositiveInfinity;
                Vector3 bestWorldPos = default;

                for (int i = 0; i < snapshot.entries.Length; i++)
                {
                    ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
                    if (!entry.valid)
                        continue;

                    Vector3 toPoint = entry.worldPos - rayOrigin;
                    float along = Vector3.Dot(toPoint, rayDir);
                    if (along < minLinearDepthMeters || along > maxDepth)
                        continue;

                    Vector3 closest = rayOrigin + rayDir * along;
                    float offRay = Vector3.Distance(entry.worldPos, closest);
                    if (offRay > matchRadius)
                        continue;

                    float score = offRay * 4f + along * 0.01f - Mathf.Clamp01(entry.confidence) * 0.05f;
                    if (!found || score < bestScore)
                    {
                        found = true;
                        bestScore = score;
                        bestWorldPos = entry.worldPos;
                    }
                }

                if (!found)
                    continue;

                candidates[column] = new RowCandidate
                {
                    valid = true,
                    worldPos = bestWorldPos,
                    alongForward = Vector3.Dot(bestWorldPos - m_rowOrigin, m_rowForward)
                };
            }

            if (enforceRowForwardContinuity)
                ApplyRowForwardContinuity(candidates);

            for (int column = 0; column < candidates.Length; column++)
            {
                if (!candidates[column].valid)
                    continue;

                for (int rowLayerIndex = 0; rowLayerIndex < totalRows; rowLayerIndex++)
                {
                    if (TryBuildSnapshotHit(candidates[column].worldPos, rowLayerIndex, out SnapshotHit hit))
                        m_scanCoverSnapshotHits.Add(hit);
                }
            }
        }

        private void ApplyRowForwardContinuity(RowCandidate[] candidates)
        {
            if (candidates == null || candidates.Length <= 2)
                return;

            int seed = FindContinuitySeedColumn(candidates);
            if (seed < 0)
                return;

            float maxStep = Mathf.Max(0.02f, maxRowForwardStepMeters);
            float previousForward = candidates[seed].alongForward;
            for (int column = seed - 1; column >= 0; column--)
            {
                if (!candidates[column].valid)
                    continue;
                if (Mathf.Abs(candidates[column].alongForward - previousForward) > maxStep)
                {
                    candidates[column].valid = false;
                    continue;
                }

                previousForward = candidates[column].alongForward;
            }

            previousForward = candidates[seed].alongForward;
            for (int column = seed + 1; column < candidates.Length; column++)
            {
                if (!candidates[column].valid)
                    continue;
                if (Mathf.Abs(candidates[column].alongForward - previousForward) > maxStep)
                {
                    candidates[column].valid = false;
                    continue;
                }

                previousForward = candidates[column].alongForward;
            }
        }

        private static int FindContinuitySeedColumn(RowCandidate[] candidates)
        {
            int center = candidates.Length / 2;
            if (candidates[center].valid)
                return center;

            for (int offset = 1; offset < candidates.Length; offset++)
            {
                int left = center - offset;
                if (left >= 0 && candidates[left].valid)
                    return left;

                int right = center + offset;
                if (right < candidates.Length && candidates[right].valid)
                    return right;
            }

            return -1;
        }

        private void UpdateMeasuredRowHeight(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
        {
            if (!useMeasuredRowHeight || sampleCamera == null || snapshot.entries == null || snapshot.entries.Length <= 0)
                return;

            Ray probeRay = sampleCamera.ViewportPointToRay(new Vector3(
                Mathf.Clamp01(measuredRowHeightViewportX),
                Mathf.Clamp01(measuredRowHeightViewportY),
                0f));
            Vector3 rayOrigin = probeRay.origin;
            Vector3 rayDir = probeRay.direction.normalized;
            float maxDepth = Mathf.Max(minLinearDepthMeters, maxLinearDepthMeters);
            float matchRadius = Mathf.Max(0.02f, measuredRowHeightMatchRadiusMeters);

            bool found = false;
            float bestScore = float.PositiveInfinity;
            float bestY = 0f;

            for (int i = 0; i < snapshot.entries.Length; i++)
            {
                ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
                if (!entry.valid || !IsFinite(entry.worldPos))
                    continue;

                Vector3 toPoint = entry.worldPos - rayOrigin;
                float along = Vector3.Dot(toPoint, rayDir);
                if (along < minLinearDepthMeters || along > maxDepth)
                    continue;

                Vector3 closest = rayOrigin + rayDir * along;
                float offRay = Vector3.Distance(entry.worldPos, closest);
                if (offRay > matchRadius)
                    continue;

                float score = offRay * 4f + along * 0.01f - Mathf.Clamp01(entry.confidence) * 0.05f;
                if (!found || score < bestScore)
                {
                    found = true;
                    bestScore = score;
                    bestY = entry.worldPos.y;
                }
            }

            if (!found)
                return;

            m_measuredRowHeightY = m_hasMeasuredRowHeight
                ? Mathf.Lerp(m_measuredRowHeightY, bestY, Mathf.Clamp01(measuredRowHeightBlend))
                : bestY;
            m_hasMeasuredRowHeight = true;
        }

        private void BuildHeightSliceContour(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
        {
            m_heightSliceContourSegments.Clear();
            m_heightSliceSurfaceTriangles.Clear();
            m_heightSliceContourObjectsDirty = true;
            if (!showHeightSliceContour || snapshot.entries == null || snapshot.entries.Length <= 0)
                return;
            if (!TryGetRowBaseHeight(out float baseHeight))
                return;

            int maxSegments = Mathf.Max(1, heightSliceMaxContourPoints);
            float maxSegmentLength = Mathf.Max(0.01f, heightSliceMaxLinkDistanceMeters);
            float maxMeshEdgeLength = Mathf.Max(maxSegmentLength, heightSliceMaxMeshEdgeMeters);
            BuildHeightSliceSurfaceMesh(snapshot, maxMeshEdgeLength);
            if (m_heightSliceSurfaceTriangles.Count <= 0)
                return;

            var intersections = new List<Vector3>(3);
            for (int i = 0; i < m_heightSliceSurfaceTriangles.Count; i++)
            {
                HeightSliceSurfaceTriangle triangle = m_heightSliceSurfaceTriangles[i];
                intersections.Clear();
                TryAddHeightSliceIntersection(triangle.a, triangle.b, baseHeight, intersections);
                TryAddHeightSliceIntersection(triangle.b, triangle.c, baseHeight, intersections);
                TryAddHeightSliceIntersection(triangle.c, triangle.a, baseHeight, intersections);

                if (intersections.Count == 2)
                    AddHeightSliceContourSegment(intersections[0], intersections[1], maxSegmentLength);
            }

            if (m_heightSliceContourSegments.Count > maxSegments)
            {
                m_heightSliceContourSegments.Sort((a, b) =>
                    Mathf.Abs(a.centerAlongRight).CompareTo(Mathf.Abs(b.centerAlongRight)));
                m_heightSliceContourSegments.RemoveRange(maxSegments, m_heightSliceContourSegments.Count - maxSegments);
            }
        }

        private void BuildHeightSliceSurfaceMesh(
            ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot,
            float maxMeshEdgeLength)
        {
            m_heightSliceSurfaceTriangles.Clear();
            var entriesByCell = new Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry>();
            int minGroup = int.MaxValue;
            int maxGroup = int.MinValue;
            int minRow = int.MaxValue;
            int maxRow = int.MinValue;
            int minCol = int.MaxValue;
            int maxCol = int.MinValue;

            for (int i = 0; i < snapshot.entries.Length; i++)
            {
                ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
                if (!entry.valid || !IsFinite(entry.worldPos))
                    continue;

                entriesByCell[MakeGridCellKey(entry.group, entry.row, entry.col)] = entry;
                minGroup = Mathf.Min(minGroup, entry.group);
                maxGroup = Mathf.Max(maxGroup, entry.group);
                minRow = Mathf.Min(minRow, entry.row);
                maxRow = Mathf.Max(maxRow, entry.row);
                minCol = Mathf.Min(minCol, entry.col);
                maxCol = Mathf.Max(maxCol, entry.col);
            }

            if (entriesByCell.Count <= 0 || minGroup > maxGroup || minRow >= maxRow || minCol >= maxCol)
                return;

            for (int group = minGroup; group <= maxGroup; group++)
            {
                for (int row = minRow; row < maxRow; row++)
                {
                    for (int col = minCol; col < maxCol; col++)
                    {
                        if (!entriesByCell.TryGetValue(MakeGridCellKey(group, row, col), out ScanCoverDepthGridPointCloud.GridStateEntry e00) ||
                            !entriesByCell.TryGetValue(MakeGridCellKey(group, row, col + 1), out ScanCoverDepthGridPointCloud.GridStateEntry e10) ||
                            !entriesByCell.TryGetValue(MakeGridCellKey(group, row + 1, col + 1), out ScanCoverDepthGridPointCloud.GridStateEntry e11) ||
                            !entriesByCell.TryGetValue(MakeGridCellKey(group, row + 1, col), out ScanCoverDepthGridPointCloud.GridStateEntry e01))
                        {
                            continue;
                        }

                        AddHeightSliceSurfaceTriangle(e00.worldPos, e10.worldPos, e11.worldPos, maxMeshEdgeLength);
                        AddHeightSliceSurfaceTriangle(e00.worldPos, e11.worldPos, e01.worldPos, maxMeshEdgeLength);
                    }
                }
            }
        }

        private static long MakeGridCellKey(int group, int row, int col)
        {
            unchecked
            {
                long key = group;
                key = key * 73856093L ^ row * 19349663L ^ col * 83492791L;
                return key;
            }
        }

        private void AddHeightSliceSurfaceTriangle(Vector3 a, Vector3 b, Vector3 c, float maxMeshEdgeLength)
        {
            if (!IsFinite(a) || !IsFinite(b) || !IsFinite(c))
                return;
            if (Vector3.Distance(a, b) > maxMeshEdgeLength ||
                Vector3.Distance(b, c) > maxMeshEdgeLength ||
                Vector3.Distance(c, a) > maxMeshEdgeLength)
            {
                return;
            }

            Vector3 normal = Vector3.Cross(b - a, c - a);
            if (normal.sqrMagnitude <= 0.000001f)
                return;

            m_heightSliceSurfaceTriangles.Add(new HeightSliceSurfaceTriangle
            {
                a = a,
                b = b,
                c = c
            });
        }

        private void TryAddHeightSliceIntersection(
            Vector3 from,
            Vector3 to,
            float baseHeight,
            List<Vector3> intersections)
        {
            if (!IsFinite(from) || !IsFinite(to))
                return;

            float epsilon = Mathf.Max(0.002f, heightSliceHalfMeters * 0.05f);
            float fromDelta = from.y - baseHeight;
            float toDelta = to.y - baseHeight;
            bool fromOnPlane = Mathf.Abs(fromDelta) <= epsilon;
            bool toOnPlane = Mathf.Abs(toDelta) <= epsilon;
            if (fromOnPlane && toOnPlane)
                return;
            if (!fromOnPlane && !toOnPlane && Mathf.Sign(fromDelta) == Mathf.Sign(toDelta))
                return;

            Vector3 point;
            if (fromOnPlane)
            {
                point = from;
            }
            else if (toOnPlane)
            {
                point = to;
            }
            else
            {
                float t = Mathf.Clamp01((baseHeight - from.y) / (to.y - from.y));
                point = Vector3.Lerp(from, to, t);
            }

            point.y = baseHeight;
            AddUniqueHeightSliceIntersection(intersections, point);
        }

        private static void AddUniqueHeightSliceIntersection(List<Vector3> intersections, Vector3 point)
        {
            const float MinDistanceSqr = 0.000001f;
            for (int i = 0; i < intersections.Count; i++)
            {
                if ((intersections[i] - point).sqrMagnitude <= MinDistanceSqr)
                    return;
            }

            intersections.Add(point);
        }

        private void AddHeightSliceContourSegment(Vector3 from, Vector3 to, float maxEdgeLength)
        {
            Vector3 delta = to - from;
            float length = delta.magnitude;
            if (length <= 1e-4f || length > maxEdgeLength)
                return;

            Vector3 center = (from + to) * 0.5f;
            m_heightSliceContourSegments.Add(new HeightSliceContourSegment
            {
                from = from,
                to = to,
                centerAlongRight = Vector3.Dot(center - m_rowOrigin, m_rowRight)
            });
        }

        [ContextMenu("Capture Snapshot Now")]
        public void RequestSnapshotCapture()
        {
            captureSnapshotNow = true;
            if (useScanCoverSnapshot)
            {
                m_scanCoverCaptureRequested = true;
                m_scanCoverCaptureApplyPending = true;
                m_scanCoverCaptureRequestFrameIndex = m_scanCoverDepthGridPointCloud != null
                    ? m_scanCoverDepthGridPointCloud.FrameIndex
                    : -1;
            }
        }

        public bool SnapshotUpdatesFrozen => freezeSnapshotUpdates;

        public void SetSnapshotUpdatesFrozen(bool frozen)
        {
            freezeSnapshotUpdates = frozen;
        }

        public void ToggleSnapshotUpdatesFrozen()
        {
            freezeSnapshotUpdates = !freezeSnapshotUpdates;
        }

        [ContextMenu("Clear Probe Row Runtime State")]
        public void ClearRuntimeState()
        {
            ClearRuntimeState(requestFreshSnapshot: false);
        }

        public void ClearRuntimeState(bool requestFreshSnapshot)
        {
            float now = Time.unscaledTime;
            captureSnapshotNow = false;
            m_scanCoverCaptureRequested = requestFreshSnapshot && useScanCoverSnapshot;
            m_snapshotBootstrapPending = requestFreshSnapshot;
            m_scanCoverCaptureApplyPending = requestFreshSnapshot && useScanCoverSnapshot;
            m_scanCoverCaptureRequestFrameIndex = m_scanCoverDepthGridPointCloud != null
                ? m_scanCoverDepthGridPointCloud.FrameIndex
                : -1;
            m_lastSnapshotTime = float.NegativeInfinity;
            m_lastAppliedScanCoverFrameIndex = requestFreshSnapshot || m_scanCoverDepthGridPointCloud == null
                ? -1
                : m_scanCoverDepthGridPointCloud.FrameIndex;
            m_hasPendingReadback = false;
            m_hasMeasuredRowHeight = false;
            m_rawSnapshotHits.Clear();
            m_snapshotHits.Clear();
            m_scanCoverSnapshotHits.Clear();
            m_hasRowFrame = false;
            EnsureRowFrame();
            CaptureCurrentPose(now);
            ClearProbeMemory(clearHeightSliceContours: true);
        }

        private void ConfigureScanCoverComponents()
        {
            if (m_scanCoverPreprocessor != null)
            {
                SetPrivateFieldIfNull(m_scanCoverPreprocessor, "environmentDepthManager", environmentDepthManager);
                if (scanCoverPreprocessorComputeShader != null)
                    SetPrivateFieldIfNull(m_scanCoverPreprocessor, "computeShader", scanCoverPreprocessorComputeShader);
                m_scanCoverPreprocessor.SetRefreshEveryFrame(false);
            }

            if (m_scanCoverDepthGridPointCloud != null)
            {
                SetPrivateFieldIfNull(m_scanCoverDepthGridPointCloud, "preprocessor", m_scanCoverPreprocessor);
                m_scanCoverDepthGridPointCloud.SetUpdateEveryFrame(false);
                m_scanCoverDepthGridPointCloud.SetPreviewDisplayVisible(showScanCoverDepthGridSnapshot);
            }
        }

        private static void SetPrivateField(object target, string fieldName, object value)
        {
            if (target == null)
                return;

            FieldInfo field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null)
                return;

            field.SetValue(target, value);
        }

        private static void SetPrivateFieldIfNull(object target, string fieldName, object value)
        {
            if (target == null || value == null)
                return;

            FieldInfo field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null)
                return;

            object currentValue = field.GetValue(target);
            if (currentValue == null || (currentValue is UnityEngine.Object unityObj && unityObj == null))
                field.SetValue(target, value);
        }

        private bool TryGetRowBaseHeight(out float baseHeight)
        {
            if (overrideWorldHeight)
            {
                baseHeight = worldHeightY;
                return true;
            }

            if (useScanCoverSnapshot && useMeasuredRowHeight && m_hasMeasuredRowHeight)
            {
                baseHeight = m_measuredRowHeightY;
                return true;
            }

            baseHeight = 0f;
            return false;
        }

        private bool TryBuildSnapshotHit(Vector3 worldPos, int rowLayerIndex, out SnapshotHit snapshotHit)
        {
            snapshotHit = default;
            float rowYOffset = Mathf.Max(0.01f, lowerRowStepMeters) * rowLayerIndex;
            Vector3 displayPos = worldPos;
            bool lockRowHeight = TryGetRowBaseHeight(out float rowBaseHeight);
            if (lockRowHeight)
                displayPos.y = rowBaseHeight - rowYOffset;
            else
                displayPos.y -= rowYOffset;

            float spacing = Mathf.Max(0.02f, horizontalSpacingMeters);
            float forwardSpacing = Mathf.Max(0.02f, forwardBandSpacingMeters);
            float heightSpacing = Mathf.Max(0.02f, verticalBandSpacingMeters);
            Vector3 offset = displayPos - m_rowOrigin;
            float alongRight = Vector3.Dot(offset, m_rowRight);
            float snappedAlongRight = Mathf.Round(alongRight / spacing) * spacing;
            int horizontalIndex = Mathf.RoundToInt(snappedAlongRight / spacing);
            float alongForward = Vector3.Dot(offset, m_rowForward);

            if (snapDisplayToHorizontalSlots)
            {
                float targetHeight = displayPos.y;
                displayPos = m_rowOrigin + m_rowRight * snappedAlongRight + m_rowForward * alongForward;
                displayPos.y = targetHeight;
            }

            int forwardIndex = Mathf.RoundToInt(alongForward / forwardSpacing);
            int heightIndex = lockRowHeight
                ? 0
                : Mathf.RoundToInt((displayPos.y - m_rowOrigin.y) / heightSpacing);

            ProbeKey key = new ProbeKey
            {
                rowLayerIndex = rowLayerIndex,
                horizontalIndex = horizontalIndex,
                forwardIndex = forwardIndex,
                heightIndex = heightIndex
            };

            snapshotHit = new SnapshotHit
            {
                key = key,
                position = displayPos
            };
            return true;
        }

        private void UpsertHit(SnapshotHit snapshotHit, float now, bool poseStable)
        {
            ProbeKey key = snapshotHit.key;
            Vector3 displayPos = snapshotHit.position;
            int hitIndex = FindHitIndex(key);
            if (hitIndex < 0)
            {
                bool startAnchored = useScanCoverSnapshot && scanCoverTargetsStartAnchored;
                m_hits.Add(new ProbeHit
                {
                    valid = true,
                    key = key,
                    observedPos = displayPos,
                    anchoredPos = displayPos,
                    pendingTargetPos = displayPos,
                    lastSeenTime = now,
                    supportCount = 1,
                    anchored = startAnchored,
                    state = startAnchored ? ProbeState.Frozen : ProbeState.Targeting,
                    frozenUntilTime = startAnchored ? now + Mathf.Max(0.05f, committedFreezeSeconds) : 0f,
                    challengedFrames = 0,
                    outlierFrames = 0,
                    pendingTargetFrames = 0,
                    targetMismatchFrames = 0
                });
                return;
            }

            ProbeHit hit = m_hits[hitIndex];
            float observationBlend = useScanCoverSnapshot
                ? 1f
                : Mathf.Clamp01(observedPositionBlend);
            hit.observedPos = Vector3.Lerp(hit.observedPos, displayPos, observationBlend);
            hit.lastSeenTime = now;
            hit.supportCount++;
            hit.targetMismatchFrames = 0;

            if (!hit.anchored)
            {
                if (poseStable)
                {
                    hit.anchoredPos = Vector3.Lerp(hit.anchoredPos, hit.observedPos, Mathf.Clamp01(observedPositionBlend));
                    if (hit.supportCount >= Mathf.Max(1, anchorAfterHits))
                    {
                        hit.anchored = true;
                        hit.state = ProbeState.Frozen;
                        hit.frozenUntilTime = now + Mathf.Max(0.05f, committedFreezeSeconds);
                        hit.pendingTargetPos = hit.anchoredPos;
                        hit.pendingTargetFrames = 0;
                        hit.challengedFrames = 0;
                    }
                }
            }
            else
            {
                float anchorDelta = Vector3.Distance(hit.anchoredPos, hit.observedPos);
                if (useScanCoverSnapshot && scanCoverTargetsStartAnchored && poseStable)
                {
                    hit.anchoredPos = hit.observedPos;
                    hit.pendingTargetPos = hit.anchoredPos;
                    hit.pendingTargetFrames = 0;
                    hit.challengedFrames = 0;
                    hit.outlierFrames = 0;
                    hit.state = ProbeState.Frozen;
                    hit.frozenUntilTime = now + Mathf.Max(0.05f, committedFreezeSeconds);
                }
                else
                if (!poseStable)
                {
                    hit.outlierFrames = 0;
                }
                else if (anchorDelta > Mathf.Max(0.01f, anchorUnlockDistanceMeters))
                {
                    hit.outlierFrames++;
                    if (hit.outlierFrames >= Mathf.Max(1, anchorUnlockAfterOutlierFrames))
                    {
                        hit.pendingTargetPos = hit.observedPos;
                        hit.pendingTargetFrames = Mathf.Max(hit.pendingTargetFrames, 1);
                        hit.outlierFrames = 0;
                    }
                }
                else
                {
                    hit.outlierFrames = 0;
                    if (hit.state == ProbeState.Challenged)
                        hit.state = now < hit.frozenUntilTime ? ProbeState.Frozen : ProbeState.Committed;
                }
            }

            if (TryGetRowBaseHeight(out float rowBaseHeight))
            {
                float rowYOffset = Mathf.Max(0.01f, lowerRowStepMeters) * key.rowLayerIndex;
                float clampedHeight = rowBaseHeight - rowYOffset;
                hit.observedPos.y = clampedHeight;
                hit.anchoredPos.y = clampedHeight;
            }

            m_hits[hitIndex] = hit;
        }

        private void RegisterCurrentSlotTarget(VisibleSlotKey slotKey, Vector3 position)
        {
            if (m_currentSlotTargets.TryGetValue(slotKey, out Vector3 accumulated))
            {
                int count = m_currentSlotTargetCounts[slotKey] + 1;
                m_currentSlotTargets[slotKey] = accumulated + position;
                m_currentSlotTargetCounts[slotKey] = count;
            }
            else
            {
                m_currentSlotTargets[slotKey] = position;
                m_currentSlotTargetCounts[slotKey] = 1;
            }
        }

        private void ReconcileCurrentTargets(float now, bool poseStable)
        {
            foreach (var pair in m_currentSlotTargets)
            {
                VisibleSlotKey slotKey = pair.Key;
                Vector3 target = pair.Value / Mathf.Max(1, m_currentSlotTargetCounts[slotKey]);

                for (int i = 0; i < m_hits.Count; i++)
                {
                    ProbeHit hit = m_hits[i];
                    if (!hit.valid || MakeVisibleSlotKey(hit.key).rowLayerIndex != slotKey.rowLayerIndex || MakeVisibleSlotKey(hit.key).horizontalIndex != slotKey.horizontalIndex)
                        continue;

                    Vector3 displayPos = hit.anchored ? hit.anchoredPos : hit.observedPos;
                    float targetDistance = Vector3.Distance(displayPos, target);

                    bool frozen = hit.anchored && now < hit.frozenUntilTime;

                    if (hit.anchored && frozen)
                    {
                        if (targetDistance > Mathf.Max(0.01f, challengeDistanceMeters))
                        {
                            hit.challengedFrames++;
                            if (hit.challengedFrames >= Mathf.Max(1, challengeAfterFrames))
                            {
                                hit.state = ProbeState.Challenged;
                                hit.pendingTargetPos = target;
                                hit.pendingTargetFrames = 1;
                                hit.challengedFrames = 0;
                            }
                        }
                        else
                        {
                            hit.challengedFrames = 0;
                        }
                    }
                    else if (hit.anchored && poseStable && targetDistance <= Mathf.Max(anchorUnlockDistanceMeters, maxTargetCorrectionDistanceMeters))
                    {
                        float consistencyDistance = Mathf.Max(0.01f, targetCommitConsistencyMeters);
                        if (hit.pendingTargetFrames <= 0 ||
                            Vector3.Distance(hit.pendingTargetPos, target) <= consistencyDistance)
                        {
                            hit.pendingTargetPos = hit.pendingTargetFrames <= 0
                                ? target
                                : Vector3.Lerp(hit.pendingTargetPos, target, 0.5f);
                            hit.pendingTargetFrames++;
                        }
                        else
                        {
                            hit.pendingTargetPos = target;
                            hit.pendingTargetFrames = 1;
                        }

                        if (hit.pendingTargetFrames >= Mathf.Max(1, targetCommitAfterFrames))
                        {
                            hit.anchoredPos = Vector3.Lerp(
                                hit.anchoredPos,
                                hit.pendingTargetPos,
                                Mathf.Clamp01(targetCorrectionBlend));
                            hit.state = ProbeState.Frozen;
                            hit.frozenUntilTime = now + Mathf.Max(0.05f, committedFreezeSeconds);
                            hit.pendingTargetFrames = 0;
                            hit.pendingTargetPos = hit.anchoredPos;
                            hit.challengedFrames = 0;
                        }
                    }
                    else if (hit.anchored && targetDistance <= Mathf.Max(0.005f, anchoredSettleDistanceMeters))
                    {
                        hit.pendingTargetFrames = 0;
                        hit.pendingTargetPos = hit.anchoredPos;
                        hit.challengedFrames = 0;
                        hit.state = frozen ? ProbeState.Frozen : ProbeState.Committed;
                    }

                    if (targetDistance > Mathf.Max(maxVisibleTargetGapMeters, maxRetainedTargetGapMeters))
                    {
                        hit.targetMismatchFrames++;
                        if (hit.targetMismatchFrames >= Mathf.Max(1, targetMismatchDropAfterFrames))
                        {
                            hit.valid = false;
                        }
                    }
                    else
                    {
                        hit.targetMismatchFrames = 0;
                    }

                    m_hits[i] = hit;
                }
            }

            for (int i = 0; i < m_hits.Count; i++)
            {
                ProbeHit hit = m_hits[i];
                if (!hit.valid)
                    continue;

                VisibleSlotKey slotKey = MakeVisibleSlotKey(hit.key);
                if (m_currentSlotTargets.ContainsKey(slotKey))
                    continue;

                hit.pendingTargetFrames = 0;
                hit.pendingTargetPos = hit.anchored ? hit.anchoredPos : hit.observedPos;
                hit.challengedFrames = 0;
                if (hit.anchored)
                    hit.state = now < hit.frozenUntilTime ? ProbeState.Frozen : ProbeState.Committed;
                m_hits[i] = hit;
            }
        }

        private void UpdateVisibleHits(float now, bool poseStable)
        {
            m_visibleHitIndices.Clear();
            m_visibleHitIndexBySlot.Clear();
            var bestByHorizontal = new Dictionary<VisibleSlotKey, int>();

            for (int i = 0; i < m_hits.Count; i++)
            {
                ProbeHit hit = m_hits[i];
                if (!IsHitRetained(hit, now))
                    continue;

                VisibleSlotKey slotKey = MakeVisibleSlotKey(hit.key);
                if (useScanCoverSnapshot &&
                    hideHitsWithoutCurrentSnapshotTarget &&
                    !m_currentSlotTargets.ContainsKey(slotKey))
                    continue;

                if (!bestByHorizontal.TryGetValue(slotKey, out int bestIndex))
                {
                    bestByHorizontal[slotKey] = i;
                    continue;
                }

                if (ShouldReplaceVisibleWinner(bestIndex, i, now))
                    bestByHorizontal[slotKey] = i;
            }

            if (poseStable)
            {
                m_selectedHitBySlot.Clear();
                foreach (var pair in bestByHorizontal)
                    m_selectedHitBySlot[pair.Key] = pair.Value;
            }
            else
            {
                foreach (var pair in bestByHorizontal)
                {
                    if (!m_selectedHitBySlot.ContainsKey(pair.Key))
                        m_selectedHitBySlot[pair.Key] = pair.Value;
                }
            }

            var staleSlots = new List<VisibleSlotKey>();
            foreach (var pair in m_selectedHitBySlot)
            {
                if (pair.Value < 0 ||
                    pair.Value >= m_hits.Count ||
                    !IsHitRetained(m_hits[pair.Value], now) ||
                    (useScanCoverSnapshot && hideHitsWithoutCurrentSnapshotTarget && !m_currentSlotTargets.ContainsKey(pair.Key)))
                    staleSlots.Add(pair.Key);
            }

            foreach (VisibleSlotKey staleSlot in staleSlots)
                m_selectedHitBySlot.Remove(staleSlot);

            foreach (int visibleIndex in m_selectedHitBySlot.Values.OrderBy(index => m_hits[index].key.rowLayerIndex)
                         .ThenBy(index => m_hits[index].key.horizontalIndex))
            {
                ProbeHit hit = m_hits[visibleIndex];
                VisibleSlotKey slotKey = MakeVisibleSlotKey(hit.key);
                if (!diagnosticTargetsOnly &&
                    m_currentSlotTargets.TryGetValue(slotKey, out Vector3 accumulatedTarget) &&
                    m_currentSlotTargetCounts.TryGetValue(slotKey, out int count) &&
                    count > 0)
                {
                    Vector3 target = accumulatedTarget / count;
                    Vector3 visiblePos = hit.anchored ? hit.anchoredPos : hit.observedPos;
                    if (Vector3.Distance(visiblePos, target) > Mathf.Max(0.02f, maxVisibleTargetGapMeters))
                        continue;
                }

                m_visibleHitIndices.Add(visibleIndex);
                m_visibleHitIndexBySlot[MakeVisibleSlotKey(hit.key)] = visibleIndex;
            }
        }

        private float ComputeDisplayScore(ProbeHit hit, float now)
        {
            Vector3 displayPos = hit.anchored ? hit.anchoredPos : hit.observedPos;
            float agePenalty = now - hit.lastSeenTime;
            float stateBonus = hit.state switch
            {
                ProbeState.Frozen => 18f,
                ProbeState.Committed => 12f,
                ProbeState.Challenged => 6f,
                _ => 0f
            };
            float score = hit.supportCount * 3f - agePenalty * 6f + (hit.anchored ? 8f : 0f) + stateBonus;

            VisibleSlotKey slotKey = MakeVisibleSlotKey(hit.key);
            if (!diagnosticTargetsOnly &&
                m_currentSlotTargets.TryGetValue(slotKey, out Vector3 accumulatedTarget) &&
                m_currentSlotTargetCounts.TryGetValue(slotKey, out int count) &&
                count > 0)
            {
                Vector3 target = accumulatedTarget / count;
                float targetDistance = Vector3.Distance(displayPos, target);
                score += 32f - targetDistance * 80f;
            }
            else
            {
                score += hit.anchored ? 10f : 0f;
                score -= Vector3.Distance(hit.anchored ? hit.anchoredPos : hit.observedPos, m_rowOrigin) * 1.5f;
            }

            return score;
        }

        private bool ShouldReplaceVisibleWinner(int currentIndex, int candidateIndex, float now)
        {
            ProbeHit current = m_hits[currentIndex];
            ProbeHit candidate = m_hits[candidateIndex];

            if (current.state == ProbeState.Frozen && candidate.state != ProbeState.Frozen)
                return false;

            if (candidate.state == ProbeState.Frozen && current.state != ProbeState.Frozen)
                return true;

            if (candidate.anchored && !current.anchored)
                return true;

            if (!candidate.anchored && current.anchored)
                return false;

            return ComputeDisplayScore(candidate, now) > ComputeDisplayScore(current, now);
        }

        private bool IsHitRetained(ProbeHit hit, float now)
        {
            bool unsettledAnchored = hit.anchored &&
                Vector3.Distance(hit.anchoredPos, hit.observedPos) > Mathf.Max(0.01f, anchorUnlockDistanceMeters);
            float retention = hit.anchored
                ? Mathf.Max(holdSeconds, anchoredHoldSeconds)
                : Mathf.Max(0.05f, holdSeconds);
            if (unsettledAnchored)
                retention = Mathf.Min(retention, Mathf.Max(0.25f, unsettledAnchoredCleanupSeconds));

            return hit.valid && (now - hit.lastSeenTime) <= retention;
        }

        private int FindHitIndex(ProbeKey key)
        {
            for (int i = 0; i < m_hits.Count; i++)
            {
                if (m_hits[i].valid &&
                    m_hits[i].key.rowLayerIndex == key.rowLayerIndex &&
                    m_hits[i].key.horizontalIndex == key.horizontalIndex &&
                    m_hits[i].key.forwardIndex == key.forwardIndex &&
                    m_hits[i].key.heightIndex == key.heightIndex)
                    return i;
            }

            return -1;
        }

        private void ClearProbeMemory()
        {
            ClearProbeMemory(clearHeightSliceContours: true);
        }

        private void ClearProbeMemory(bool clearHeightSliceContours)
        {
            m_hits.Clear();
            m_visibleHitIndices.Clear();
            m_visibleHitIndexBySlot.Clear();
            m_selectedHitBySlot.Clear();
            m_currentSlotTargets.Clear();
            m_currentSlotTargetCounts.Clear();
            m_connectionPairs.Clear();
            SetNodeObjectsActive(false);
            SetConnectionObjectsActive(false);
            m_heightSliceSurfaceTriangles.Clear();
            if (clearHeightSliceContours)
            {
                m_heightSliceContourSegments.Clear();
                SetHeightSliceContourObjectsActive(false);
                ClearHeightSliceContourMeshObjects();
                m_heightSliceContourObjectsDirty = true;
            }
            SetDiagnosticObjectsActive(false);
        }

        private VisibleSlotKey MakeVisibleSlotKey(ProbeKey key)
        {
            return new VisibleSlotKey
            {
                rowLayerIndex = key.rowLayerIndex,
                horizontalIndex = key.horizontalIndex
            };
        }

        private void SyncNodeObjects()
        {
            while (m_nodeObjects.Count < m_visibleHitIndices.Count)
                m_nodeObjects.Add(CreateNodeObject(m_nodeObjects.Count));

            Vector3 scale = Vector3.one * Mathf.Max(0.005f, cubeScaleMeters);
            for (int i = 0; i < m_nodeObjects.Count; i++)
            {
                GameObject node = m_nodeObjects[i];
                bool active = i < m_visibleHitIndices.Count;
                if (node.activeSelf != active)
                    node.SetActive(active);
                if (!active)
                    continue;

                ProbeHit hit = m_hits[m_visibleHitIndices[i]];
                Transform nodeTransform = node.transform;
                nodeTransform.position = hit.anchored ? hit.anchoredPos : hit.observedPos;
                nodeTransform.rotation = Quaternion.identity;
                nodeTransform.localScale = scale;
            }
        }

        private void SyncConnectionObjects()
        {
            if (!showConnections)
            {
                SetConnectionObjectsActive(false);
                return;
            }

            BuildConnectionPairs();

            while (m_connectionObjects.Count < m_connectionPairs.Count)
                m_connectionObjects.Add(CreateConnectionObject(m_connectionObjects.Count));

            for (int i = 0; i < m_connectionObjects.Count; i++)
            {
                GameObject connection = m_connectionObjects[i];
                bool active = i < m_connectionPairs.Count;
                if (connection.activeSelf != active)
                    connection.SetActive(active);
                if (!active)
                    continue;

                (int fromIndex, int toIndex) = m_connectionPairs[i];
                ProbeHit fromHit = m_hits[fromIndex];
                ProbeHit toHit = m_hits[toIndex];
                Vector3 fromPos = fromHit.anchored ? fromHit.anchoredPos : fromHit.observedPos;
                Vector3 toPos = toHit.anchored ? toHit.anchoredPos : toHit.observedPos;
                Vector3 delta = toPos - fromPos;
                float length = delta.magnitude;
                if (length <= 1e-4f)
                {
                    connection.SetActive(false);
                    continue;
                }

                Transform connectionTransform = connection.transform;
                connectionTransform.position = (fromPos + toPos) * 0.5f;
                connectionTransform.rotation = Quaternion.LookRotation(delta.normalized, Vector3.up);
                connectionTransform.localScale = new Vector3(
                    Mathf.Max(0.002f, connectionThicknessMeters),
                    Mathf.Max(0.002f, connectionThicknessMeters),
                    length);
            }
        }

        private bool ShouldUseStaticScanCoverContourOnlyMode()
        {
            return useScanCoverSnapshot &&
                   showHeightSliceContour &&
                   !showDepthProbeRowNodes &&
                   !showDiagnostics;
        }

        private bool ShouldPreserveHeightSliceContoursOnSnapshot()
        {
            return preserveHeightSliceContourSnapshots && ShouldUseStaticScanCoverContourOnlyMode();
        }

        private void SyncHeightSliceContourObjectsIfDirty()
        {
            if (!m_heightSliceContourObjectsDirty)
                return;

            SyncHeightSliceContourObjects();
        }

        private void SyncHeightSliceContourObjects()
        {
            if (!showHeightSliceContour || m_heightSliceContourSegments.Count <= 0)
            {
                SetHeightSliceContourObjectsActive(false);
                if (!ShouldPreserveHeightSliceContoursOnSnapshot())
                    ClearHeightSliceContourMeshObjects();
                m_heightSliceContourObjectsDirty = false;
                return;
            }

            bool preserveExisting = ShouldPreserveHeightSliceContoursOnSnapshot();
            if (!preserveExisting)
            {
                ClearHeightSliceContourMeshObjects();
                SetHeightSliceContourObjectsActive(false);
            }
            else
            {
                SetHeightSliceContourMeshObjectsActive(true);
                SetLegacyHeightSliceContourObjectsActive(false);
            }

            CreateHeightSliceContourMeshObject();

            m_heightSliceContourObjectsDirty = false;
        }

        private void CreateHeightSliceContourMeshObject()
        {
            Mesh mesh = BuildHeightSliceContourMesh();
            if (mesh == null || mesh.vertexCount <= 0)
                return;

            GameObject obj = new GameObject($"DepthProbeHeightSliceContourMesh_{m_heightSliceContourMeshObjects.Count}");
            obj.transform.SetParent(transform, worldPositionStays: false);
            obj.transform.localPosition = Vector3.zero;
            obj.transform.localRotation = Quaternion.identity;
            obj.transform.localScale = Vector3.one;

            MeshFilter filter = obj.AddComponent<MeshFilter>();
            filter.sharedMesh = mesh;

            MeshRenderer renderer = obj.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = m_heightSliceContourMaterial != null ? m_heightSliceContourMaterial : m_diagnosticLinkMaterial;
            ApplyContourRendererColor(renderer);
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;

            m_heightSliceContourMeshObjects.Add(obj);
            m_heightSliceContourMeshes.Add(mesh);
        }

        private void ApplyContourRendererColor(Renderer renderer)
        {
            if (renderer == null)
                return;

            Color solid = heightSliceContourColor;
            solid.a = 1f;
            var block = new MaterialPropertyBlock();
            renderer.GetPropertyBlock(block);
            block.SetColor(BaseColorId, solid);
            block.SetColor(ColorId, solid);
            renderer.SetPropertyBlock(block);
        }

        private Mesh BuildHeightSliceContourMesh()
        {
            int segmentCount = m_heightSliceContourSegments.Count;
            if (segmentCount <= 0)
                return null;

            float halfThickness = Mathf.Max(0.001f, heightSliceContourThicknessMeters) * 0.5f;
            var vertices = new List<Vector3>(segmentCount * 8);
            var triangles = new List<int>(segmentCount * 36);

            for (int i = 0; i < segmentCount; i++)
            {
                Vector3 from = m_heightSliceContourSegments[i].from;
                Vector3 to = m_heightSliceContourSegments[i].to;
                Vector3 delta = to - from;
                float length = delta.magnitude;
                if (length <= 1e-4f)
                    continue;

                Vector3 forward = delta / length;
                Vector3 side = Vector3.Cross(Vector3.up, forward);
                if (side.sqrMagnitude <= 1e-6f)
                    side = Vector3.right;
                side.Normalize();
                Vector3 up = Vector3.up;

                Vector3 s = side * halfThickness;
                Vector3 u = up * halfThickness;

                int start = vertices.Count;
                vertices.Add(transform.InverseTransformPoint(from - s - u));
                vertices.Add(transform.InverseTransformPoint(from + s - u));
                vertices.Add(transform.InverseTransformPoint(to + s - u));
                vertices.Add(transform.InverseTransformPoint(to - s - u));
                vertices.Add(transform.InverseTransformPoint(from - s + u));
                vertices.Add(transform.InverseTransformPoint(from + s + u));
                vertices.Add(transform.InverseTransformPoint(to + s + u));
                vertices.Add(transform.InverseTransformPoint(to - s + u));

                AddQuad(triangles, start + 4, start + 5, start + 6, start + 7);
                AddQuad(triangles, start + 0, start + 3, start + 2, start + 1);
                AddQuad(triangles, start + 0, start + 4, start + 7, start + 3);
                AddQuad(triangles, start + 1, start + 2, start + 6, start + 5);
                AddQuad(triangles, start + 0, start + 1, start + 5, start + 4);
                AddQuad(triangles, start + 3, start + 7, start + 6, start + 2);
            }

            if (vertices.Count <= 0)
                return null;

            Mesh mesh = new Mesh
            {
                name = $"DepthProbeHeightSliceContourMesh_{m_heightSliceContourMeshes.Count}"
            };
            if (vertices.Count > 65535)
                mesh.indexFormat = IndexFormat.UInt32;
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateBounds();
            mesh.RecalculateNormals();
            return mesh;
        }

        private static void AddQuad(List<int> triangles, int a, int b, int c, int d)
        {
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
            triangles.Add(a);
            triangles.Add(c);
            triangles.Add(d);
        }

        private void SyncDiagnosticObjects(float now)
        {
            if (!showDiagnostics)
            {
                SetDiagnosticObjectsActive(false);
                return;
            }

            var orderedTargets = m_currentSlotTargets.Keys
                .OrderBy(key => key.rowLayerIndex)
                .ThenBy(key => key.horizontalIndex)
                .ToList();

            while (m_diagnosticTargetObjects.Count < orderedTargets.Count)
                m_diagnosticTargetObjects.Add(CreateDiagnosticObject(
                    $"DepthProbeRowDiagnosticTarget_{m_diagnosticTargetObjects.Count}",
                    m_diagnosticTargetMaterial));

            int requiredLinks = 0;
            for (int i = 0; i < orderedTargets.Count; i++)
            {
                VisibleSlotKey key = orderedTargets[i];
                Vector3 target = m_currentSlotTargets[key] / Mathf.Max(1, m_currentSlotTargetCounts[key]);
                GameObject targetObject = m_diagnosticTargetObjects[i];
                if (!targetObject.activeSelf)
                    targetObject.SetActive(true);

                Transform targetTransform = targetObject.transform;
                targetTransform.position = target;
                targetTransform.rotation = Quaternion.identity;
                targetTransform.localScale = Vector3.one * Mathf.Max(0.003f, diagnosticTargetScaleMeters);

                if (m_selectedHitBySlot.TryGetValue(key, out int visibleIndex) &&
                    visibleIndex >= 0 &&
                    visibleIndex < m_hits.Count &&
                    IsHitRetained(m_hits[visibleIndex], now))
                {
                    ProbeHit hit = m_hits[visibleIndex];
                    Vector3 visiblePos = hit.anchored ? hit.anchoredPos : hit.observedPos;
                    if (Vector3.Distance(target, visiblePos) > 0.002f)
                        requiredLinks++;
                }
            }

            for (int i = orderedTargets.Count; i < m_diagnosticTargetObjects.Count; i++)
            {
                if (m_diagnosticTargetObjects[i].activeSelf)
                    m_diagnosticTargetObjects[i].SetActive(false);
            }

            while (m_diagnosticLinkObjects.Count < requiredLinks)
                m_diagnosticLinkObjects.Add(CreateDiagnosticObject(
                    $"DepthProbeRowDiagnosticLink_{m_diagnosticLinkObjects.Count}",
                    m_diagnosticLinkMaterial));

            int linkCursor = 0;
            for (int i = 0; i < orderedTargets.Count; i++)
            {
                VisibleSlotKey key = orderedTargets[i];
                if (!m_selectedHitBySlot.TryGetValue(key, out int visibleIndex) ||
                    visibleIndex < 0 ||
                    visibleIndex >= m_hits.Count ||
                    !IsHitRetained(m_hits[visibleIndex], now))
                    continue;

                Vector3 target = m_currentSlotTargets[key] / Mathf.Max(1, m_currentSlotTargetCounts[key]);
                ProbeHit hit = m_hits[visibleIndex];
                Vector3 visiblePos = hit.anchored ? hit.anchoredPos : hit.observedPos;
                Vector3 delta = visiblePos - target;
                float length = delta.magnitude;
                if (length <= 0.002f)
                    continue;

                GameObject linkObject = m_diagnosticLinkObjects[linkCursor++];
                if (!linkObject.activeSelf)
                    linkObject.SetActive(true);

                Transform linkTransform = linkObject.transform;
                linkTransform.position = (target + visiblePos) * 0.5f;
                linkTransform.rotation = Quaternion.LookRotation(delta.normalized, Vector3.up);
                linkTransform.localScale = new Vector3(
                    Mathf.Max(0.001f, diagnosticLinkThicknessMeters),
                    Mathf.Max(0.001f, diagnosticLinkThicknessMeters),
                    length);
            }

            for (int i = linkCursor; i < m_diagnosticLinkObjects.Count; i++)
            {
                if (m_diagnosticLinkObjects[i].activeSelf)
                    m_diagnosticLinkObjects[i].SetActive(false);
            }
        }

        private void BuildConnectionPairs()
        {
            m_connectionPairs.Clear();
            var edgeSet = new HashSet<ulong>();

            foreach (var pair in m_visibleHitIndexBySlot)
            {
                VisibleSlotKey slotKey = pair.Key;
                int currentIndex = pair.Value;
                TryAddConnection(slotKey, slotKey.rowLayerIndex, slotKey.horizontalIndex + 1, currentIndex, maxHorizontalConnectionDistanceMeters, edgeSet);
                TryAddConnection(slotKey, slotKey.rowLayerIndex + 1, slotKey.horizontalIndex, currentIndex, maxVerticalConnectionDistanceMeters, edgeSet);
                TryAddConnection(slotKey, slotKey.rowLayerIndex + 1, slotKey.horizontalIndex + 1, currentIndex, maxDiagonalConnectionDistanceMeters, edgeSet);
                TryAddConnection(slotKey, slotKey.rowLayerIndex + 1, slotKey.horizontalIndex - 1, currentIndex, maxDiagonalConnectionDistanceMeters, edgeSet);
            }
        }

        private void TryAddConnection(VisibleSlotKey currentSlot, int targetRowLayer, int targetHorizontalIndex, int currentIndex, float maxDistance, HashSet<ulong> edgeSet)
        {
            VisibleSlotKey targetSlot = new VisibleSlotKey
            {
                rowLayerIndex = targetRowLayer,
                horizontalIndex = targetHorizontalIndex
            };
            if (!m_visibleHitIndexBySlot.TryGetValue(targetSlot, out int targetIndex))
                return;

            ProbeHit currentHit = m_hits[currentIndex];
            ProbeHit targetHit = m_hits[targetIndex];
            Vector3 currentPos = currentHit.anchored ? currentHit.anchoredPos : currentHit.observedPos;
            Vector3 targetPos = targetHit.anchored ? targetHit.anchoredPos : targetHit.observedPos;
            if (Vector3.Distance(currentPos, targetPos) > Mathf.Max(0.01f, maxDistance))
                return;

            int a = Mathf.Min(currentIndex, targetIndex);
            int b = Mathf.Max(currentIndex, targetIndex);
            ulong edgeKey = ((ulong)(uint)a << 32) | (uint)b;
            if (!edgeSet.Add(edgeKey))
                return;

            m_connectionPairs.Add((currentIndex, targetIndex));
        }

        private GameObject CreateNodeObject(int index)
        {
            GameObject node = GameObject.CreatePrimitive(PrimitiveType.Cube);
            node.name = $"DepthProbeRowNode_{index}";
            node.transform.SetParent(transform, worldPositionStays: false);

            Collider collider = node.GetComponent<Collider>();
            if (collider != null)
            {
                if (Application.isPlaying)
                    Destroy(collider);
                else
                    DestroyImmediate(collider);
            }

            MeshRenderer renderer = node.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = m_runtimeMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            }

            node.SetActive(false);
            return node;
        }

        private GameObject CreateConnectionObject(int index)
        {
            GameObject connection = GameObject.CreatePrimitive(PrimitiveType.Cube);
            connection.name = $"DepthProbeRowConnection_{index}";
            connection.transform.SetParent(transform, worldPositionStays: false);

            Collider collider = connection.GetComponent<Collider>();
            if (collider != null)
            {
                if (Application.isPlaying)
                    Destroy(collider);
                else
                    DestroyImmediate(collider);
            }

            MeshRenderer renderer = connection.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = m_runtimeMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            }

            connection.SetActive(false);
            return connection;
        }

        private GameObject CreateDiagnosticObject(string objectName, Material material)
        {
            GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obj.name = objectName;
            obj.transform.SetParent(transform, worldPositionStays: false);

            Collider collider = obj.GetComponent<Collider>();
            if (collider != null)
            {
                if (Application.isPlaying)
                    Destroy(collider);
                else
                    DestroyImmediate(collider);
            }

            MeshRenderer renderer = obj.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = material;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            }

            obj.SetActive(false);
            return obj;
        }

        private void SetNodeObjectsActive(bool active)
        {
            for (int i = 0; i < m_nodeObjects.Count; i++)
            {
                if (m_nodeObjects[i] != null)
                    m_nodeObjects[i].SetActive(active);
            }
        }

        private void SetConnectionObjectsActive(bool active)
        {
            for (int i = 0; i < m_connectionObjects.Count; i++)
            {
                if (m_connectionObjects[i] != null)
                    m_connectionObjects[i].SetActive(active);
            }
        }

        private void SetHeightSliceContourObjectsActive(bool active)
        {
            SetLegacyHeightSliceContourObjectsActive(active);
            SetHeightSliceContourMeshObjectsActive(active);
        }

        private void SetLegacyHeightSliceContourObjectsActive(bool active)
        {
            for (int i = 0; i < m_heightSliceContourObjects.Count; i++)
            {
                if (m_heightSliceContourObjects[i] != null)
                    m_heightSliceContourObjects[i].SetActive(active);
            }
        }

        private void SetHeightSliceContourMeshObjectsActive(bool active)
        {
            for (int i = 0; i < m_heightSliceContourMeshObjects.Count; i++)
            {
                if (m_heightSliceContourMeshObjects[i] != null)
                    m_heightSliceContourMeshObjects[i].SetActive(active);
            }
        }

        private void ClearHeightSliceContourMeshObjects()
        {
            for (int i = 0; i < m_heightSliceContourMeshObjects.Count; i++)
            {
                GameObject obj = m_heightSliceContourMeshObjects[i];
                if (obj == null)
                    continue;
                MeshFilter filter = obj.GetComponent<MeshFilter>();
                if (filter != null)
                    filter.sharedMesh = null;
                if (Application.isPlaying)
                    Destroy(obj);
                else
                    DestroyImmediate(obj);
            }

            for (int i = 0; i < m_heightSliceContourMeshes.Count; i++)
            {
                Mesh mesh = m_heightSliceContourMeshes[i];
                if (mesh == null)
                    continue;
                if (Application.isPlaying)
                    Destroy(mesh);
                else
                    DestroyImmediate(mesh);
            }

            m_heightSliceContourMeshObjects.Clear();
            m_heightSliceContourMeshes.Clear();
        }

        private void SetDiagnosticObjectsActive(bool active)
        {
            for (int i = 0; i < m_diagnosticTargetObjects.Count; i++)
            {
                if (m_diagnosticTargetObjects[i] != null)
                    m_diagnosticTargetObjects[i].SetActive(active);
            }

            for (int i = 0; i < m_diagnosticLinkObjects.Count; i++)
            {
                if (m_diagnosticLinkObjects[i] != null)
                    m_diagnosticLinkObjects[i].SetActive(active);
            }
        }

        private float SampleLinearDepth(Vector2Int texCoord, int eyeIndex)
        {
            int pixelIndex = texCoord.x + texCoord.y * CopyTextureSize + eyeIndex * CopyTextureSize * CopyTextureSize;
            if (!m_depthPixels.IsCreated || pixelIndex < 0 || pixelIndex >= m_depthPixels.Length)
                return 0f;
            return m_depthPixels[pixelIndex];
        }

        private static bool TryReconstructWorld(Vector2Int texCoord, float linearDepth, Matrix4x4 inverseMatrix, out Vector3 worldPos)
        {
            worldPos = Vector3.zero;
            if (linearDepth <= 0f)
                return false;

            Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
            float clipDepth = zParams.x / linearDepth - zParams.y;
            float oneOverSize = 1f / CopyTextureSize;
            Vector4 clipPos = new Vector4(
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

        private static Vector2Int ViewportToDepthCoord(float viewportX, float viewportY)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(viewportX * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt(viewportY * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            return new Vector2Int(x, y);
        }

        private static bool IsFinite(Vector4 value)
        {
            return float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z) && float.IsFinite(value.w);
        }

        private static bool IsFinite(Vector3 value)
        {
            return float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);
        }
    }
}

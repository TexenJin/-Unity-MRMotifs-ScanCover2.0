using System;
using Unity.XR.CoreUtils.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace Genesis.RoomScan
{
    /// <summary>
    /// Captures stereo depth from the AR occlusion subsystem, computes world-space normals,
    /// runs optional bilateral filtering guided by the passthrough RGB feed, and produces
    /// dilated depth textures consumed by <see cref="VolumeIntegrator"/> for TSDF integration.
    /// </summary>
    [DefaultExecutionOrder(-40)]
    public class DepthCapture : MonoBehaviour
    {
        public static DepthCapture Instance { get; private set; }

        [SerializeField] private ComputeShader depthNormalCompute;
        [SerializeField] private ComputeShader depthDilationCompute;
        [SerializeField] private ComputeShader bilateralFilterCompute;
        [SerializeField] private ComputeShader depthEdgeCleanCompute;

        [Header("Bilateral Depth Filter")]
        [Tooltip("Edge-preserving depth denoising guided by passthrough RGB. Smooths flat surfaces while keeping object boundaries sharp.")]
        [SerializeField] private bool enableBilateralFilter = true;
        [SerializeField, Range(1f, 8f)] private float sigmaSpatial = 3.0f;
        [SerializeField, Range(0.01f, 0.5f)] private float sigmaColor = 0.1f;
        [SerializeField, Range(0.001f, 0.1f)] private float sigmaDepth = 0.02f;
        [SerializeField, Range(1, 5)] private int filterRadius = 2;

        [Header("Depth Edge Clean（v0.2 边缘识别 GPU 移植）")]
        [Tooltip("三段式深度边缘检测（邻域范围→局部平面拟合拒绝→双聚类验证），判为边缘的像素写 0：" +
                 "融合侧近距闸/截断带自然拦下=弃权不投票（绝不投远票误伤缝隙后墙）。" +
                 "治折角/缝隙的假网格桥接，TSDF 零先验区恢复系统级 45° 自然倒角。融合 shader 零改动。" +
                 "验证姿势：开后看点阵/深度预览的边缘切口是否只出现在真折角，斜面地面不应出切口。 (default true)")]
        [SerializeField] private bool enableDepthEdgeClean = true;
        [SerializeField, Range(1, 2)] private int edgeProbeRadiusPixels = 1;
        [SerializeField, Min(0.001f)] private float edgeJumpBaseMeters = 0.035f;
        [SerializeField, Min(0f)] private float edgeJumpDistanceScale = 0.018f;
        [SerializeField] private bool edgeUseLocalPlaneReject = true;
        [SerializeField, Min(0.001f)] private float edgePlaneResidualBaseMeters = 0.012f;
        [SerializeField, Min(0f)] private float edgePlaneResidualDistanceScale = 0.006f;
        [SerializeField, Range(0.1f, 1f)] private float edgePlaneMinInlierRatio = 0.65f;
        [SerializeField, Min(3)] private int edgePlaneMinNeighborPoints = 5;
        [SerializeField, Range(0.1f, 0.9f)] private float edgeTwoClusterMinFraction = 0.25f;
        [Tooltip("前景遮蔽否决（③b）：邻域有平坦前景群+真断崖间隙且中心远在间隙后=裙边飞点，杀。" +
                 "治斜视时深度在边缘拖出的渐变裙边（能通过平面拟合的斜坡桥）。连续斜坡（地面）无断崖不触发。 (default true)")]
        [SerializeField] private bool edgeSkirtVetoEnabled = true;
        [SerializeField, Min(2)] private int edgeSkirtMinNearProbes = 2;
        [Tooltip("掠射面片否决（③c）：局部拟合平面几乎侧对视线（法线·视线 < 阈值）= 沿视线拉出的飞点帘，杀。" +
                 "治连续斜坡状裙边桥（无断崖间隙、能过平面拟合的那种）。真实面片二维铺开、法线必朝向视线一侧，不触发。" +
                 "阈值对齐融合法线闸 MIN_DOT 0.3——只弃权融合反正不会用的掠射样本。 (default true)")]
        [SerializeField] private bool edgeGrazingVetoEnabled = true;
        [SerializeField, Range(0.05f, 0.9f)] private float edgeGrazingMinFacingDot = 0.3f;
        [Tooltip("③c 低跨度入口（缓坡帘）：①号断崖门槛抓不到的长缓裙边，用此更低门槛单独喂给朝向否决。" +
                 "治小掠射角下的假桥（跨度只有几厘米、能过融合法线闸的那种）。误杀远处噪面=调大。 (default 0.02)")]
        [SerializeField, Min(0.001f)] private float edgeGrazingRangeBaseMeters = 0.02f;
        [SerializeField, Min(0f)] private float edgeGrazingRangeScale = 0.008f;
        [Tooltip("双眼交叉证伪（③e，最后关卡）：把像素世界点投到另一眼，若另一眼在同一视线上看到更远表面" +
                 "（该点浮在已知空域）且面片也朝向另一眼 = 飞行像素，杀。飞点帘绕每只眼各自视点生成，" +
                 "真实表面两眼一致；背向另一眼的面受朝向闸保护。治斜视缓坡帘（单帧无签名的那种）。 (default true)")]
        [SerializeField] private bool edgeCrossEyeEnabled = true;
        [SerializeField, Min(0.005f)] private float edgeCrossEyeMarginBaseMeters = 0.05f;
        [SerializeField, Min(0f)] private float edgeCrossEyeMarginScale = 0.03f;
        [SerializeField, Range(0.05f, 0.9f)] private float edgeCrossEyeMinOtherFacing = 0.2f;

        [Header("Dilation")]
        [SerializeField] private int dilationSteps = 8;
        [SerializeField] private float voxelDistance = 0.2f;
        [SerializeField] private float voxelSize = 0.05f;

        private readonly Matrix4x4[] _proj = new Matrix4x4[2];
        private readonly Matrix4x4[] _projInv = new Matrix4x4[2];
        private readonly Matrix4x4[] _view = new Matrix4x4[2];
        private readonly Matrix4x4[] _viewInv = new Matrix4x4[2];
        private Vector2 _planes;

        /// <summary>Per-eye projection matrices derived from the depth frame's FOV and near/far planes.</summary>
        public Matrix4x4[] Proj => _proj;
        /// <summary>Inverse projection matrices (per-eye).</summary>
        public Matrix4x4[] ProjInv => _projInv;
        /// <summary>Per-eye view matrices (tracking-space to depth-camera-space).</summary>
        public Matrix4x4[] View => _view;
        /// <summary>Inverse view matrices (per-eye), mapping depth-camera-space back to tracking-space.</summary>
        public Matrix4x4[] ViewInv => _viewInv;
        /// <summary>Near and far clip distances (x = near, y = far) for the current depth frame.</summary>
        public Vector2 Planes => _planes;

        // Shader property IDs
        public static readonly int DepthTexID = Shader.PropertyToID("gsDepthTex");
        public static readonly int DepthTexRWID = Shader.PropertyToID("gsDepthTexRW");
        public static readonly int TexSizeID = Shader.PropertyToID("gsDepthTexSize");
        public static readonly int NormTexID = Shader.PropertyToID("gsDepthNormalTex");
        public static readonly int NormTexRWID = Shader.PropertyToID("gsDepthNormalTexRW");
        public static readonly int ZParamsID = Shader.PropertyToID("gsDepthZParams");
        public static readonly int ProjID = Shader.PropertyToID("gsDepthProj");
        public static readonly int ProjInvID = Shader.PropertyToID("gsDepthProjInv");
        public static readonly int ViewID = Shader.PropertyToID("gsDepthView");
        public static readonly int ViewInvID = Shader.PropertyToID("gsDepthViewInv");
        public static readonly int InputRawMonoDepthID = Shader.PropertyToID("gsInputRawMonoDepth");
        public static readonly int DilateSrcID = Shader.PropertyToID("gsDilateSrc");
        public static readonly int DilateDestID = Shader.PropertyToID("gsDilateDest");
        public static readonly int DilateStepSizeID = Shader.PropertyToID("gsDilateStepSize");
        public static readonly int DilatedDepthTexID = Shader.PropertyToID("gsDilatedDepth");
        public static readonly int EdgeReasonTexID = Shader.PropertyToID("gsEdgeReasonTex");
        public static readonly int VoxDistID = Shader.PropertyToID("gsVoxDist");
        public static readonly int VoxSizeShaderID = Shader.PropertyToID("gsVoxSize");

        // Bilateral filter property IDs
        private static readonly int BilSrcDepthID = Shader.PropertyToID("_SrcDepth");
        private static readonly int BilRGBGuideID = Shader.PropertyToID("_RGBGuide");
        private static readonly int BilDstDepthID = Shader.PropertyToID("_DstDepth");
        private static readonly int BilDepthWID = Shader.PropertyToID("_DepthW");
        private static readonly int BilDepthHID = Shader.PropertyToID("_DepthH");
        private static readonly int BilSigmaSpatialID = Shader.PropertyToID("_SigmaSpatial");
        private static readonly int BilSigmaColorID = Shader.PropertyToID("_SigmaColor");
        private static readonly int BilSigmaDepthID = Shader.PropertyToID("_SigmaDepth");
        private static readonly int BilFilterRadiusID = Shader.PropertyToID("_FilterRadius");

        // Depth edge clean property IDs（_SrcDepth/_DstDepth/_DepthW/_DepthH 与双边同名同 ID，复用）
        private static readonly int EdgeStatsID = Shader.PropertyToID("_EdgeStats");
        private static readonly int EdgeReasonRWID = Shader.PropertyToID("_EdgeReason");
        private static readonly int EdgeProjInvID = Shader.PropertyToID("_DepthProjInv");
        private static readonly int EdgeViewInvID = Shader.PropertyToID("_DepthViewInv");
        private static readonly int EdgeLinearizeABID = Shader.PropertyToID("_LinearizeAB");
        private static readonly int EdgeProbeRadiusID = Shader.PropertyToID("_ProbeRadius");
        private static readonly int EdgeJumpBaseID = Shader.PropertyToID("_EdgeJumpBase");
        private static readonly int EdgeJumpDistScaleID = Shader.PropertyToID("_EdgeJumpDistScale");
        private static readonly int EdgeUsePlaneRejectID = Shader.PropertyToID("_UsePlaneReject");
        private static readonly int EdgePlaneResidualBaseID = Shader.PropertyToID("_PlaneResidualBase");
        private static readonly int EdgePlaneResidualDistScaleID = Shader.PropertyToID("_PlaneResidualDistScale");
        private static readonly int EdgePlaneMinInlierRatioID = Shader.PropertyToID("_PlaneMinInlierRatio");
        private static readonly int EdgePlaneMinNeighborsID = Shader.PropertyToID("_PlaneMinNeighbors");
        private static readonly int EdgeTwoClusterMinFractionID = Shader.PropertyToID("_TwoClusterMinFraction");
        private static readonly int SkirtVetoEnabledID = Shader.PropertyToID("_SkirtVetoEnabled");
        private static readonly int SkirtMinNearProbesID = Shader.PropertyToID("_SkirtMinNearProbes");
        private static readonly int GrazingVetoEnabledID = Shader.PropertyToID("_GrazingVetoEnabled");
        private static readonly int GrazingMinFacingDotID = Shader.PropertyToID("_GrazingMinFacingDot");
        private static readonly int GrazingRangeBaseID = Shader.PropertyToID("_GrazingRangeBase");
        private static readonly int GrazingRangeScaleID = Shader.PropertyToID("_GrazingRangeScale");
        private static readonly int CrossEyeEnabledID = Shader.PropertyToID("_CrossEyeEnabled");
        private static readonly int CrossEyeMarginBaseID = Shader.PropertyToID("_CrossEyeMarginBase");
        private static readonly int CrossEyeMarginScaleID = Shader.PropertyToID("_CrossEyeMarginScale");
        private static readonly int CrossEyeMinOtherFacingID = Shader.PropertyToID("_CrossEyeMinOtherFacing");
        private static readonly int EdgeProjFwdID = Shader.PropertyToID("_DepthProj");
        private static readonly int EdgeViewFwdID = Shader.PropertyToID("_DepthView");

        /// <summary>True once a valid depth frame has been received from the AR occlusion subsystem.</summary>
        public static bool DepthAvailable { get; private set; }

        /// <summary>已接收的深度帧总数（诊断"帧是否还在流"：盯着不动时此数仍应持续增长）。</summary>
        public int FrameCount => _frameCount;

        /// <summary>
        /// True after USE_SCENE permission is confirmed and the initial subsystem check passes.
        /// Until this is set, <see cref="StartDepthCapture"/> is a no-op.
        /// </summary>
        private bool _permissionReady;

        /// <summary>
        /// Tracks whether the caller (RoomScanner) wants depth capture active.
        /// Persists across app pause/resume so the subsystem is re-enabled correctly.
        /// </summary>
        private bool _captureActive;

        private ComputeKernelHelper _normKernel;
        private ComputeKernelHelper _monoConvertKernel;
        private ComputeKernelHelper _initDilateKernel;
        private ComputeKernelHelper _dilateStepKernel;
        private ComputeKernelHelper _bilateralKernel;
        private bool _hasBilateralKernel;
        private ComputeKernelHelper _edgeCleanKernel;
        private bool _hasEdgeCleanKernel;

        private Texture _depthTex;
        /// <summary>The current depth texture (raw or bilateral-filtered), as a stereo Tex2DArray.</summary>
        public Texture DepthTex => _depthTex;

        private RenderTexture _normTex;
        /// <summary>World-space normals computed from the depth texture via the DepthNorm compute shader.</summary>
        public RenderTexture NormTex => _normTex;

        private RenderTexture _dilationA, _dilationB;
        private RenderTexture _dilatedDepth;
        /// <summary>Depth texture after jump-flood dilation, used by the integrator to fill holes near voxel boundaries.</summary>
        public RenderTexture DilatedDepthTex => _dilatedDepth;

        private RenderTexture _simulatedDepthTex;
        private RenderTexture _filteredDepthTex;
        private RenderTexture _edgeCleanedDepthTex;
        private RenderTexture _edgeReasonTex;
        public RenderTexture EdgeReasonTex => _edgeReasonTex;
        private ComputeBuffer _edgeStats;
        private bool _edgeStatsReadbackPending;
        private int _edgeCleansSinceStats;
        private readonly Vector4[] _linearizeAB = new Vector4[2];
        private static readonly uint[] ZeroEdgeStats = new uint[2];
        private int _dilationMaxStep;

        /// <summary>本统计周期被判为深度边缘并作废的像素数（HUD"缘:"读数）。</summary>
        public uint LastEdgeCleanCount { get; private set; }
        /// <summary>其中由③c 掠射否决击杀的像素数（HUD"掠:"读数）——定责：桥区此数高=清洗在打但桥不在深度里；低=没抓到。</summary>
        public uint LastGrazingKillCount { get; private set; }
        /// <summary>边缘清洗统计是否已有首批读数。</summary>
        public bool HasEdgeCleanStats { get; private set; }

        private Texture _rgbGuide;

        private AROcclusionManager _arOcclusionManager;
        private Unity.XR.CoreUtils.XROrigin _xrOrigin;
        private Transform _trackingSpaceTransform;
        private Camera _mainCam;
        private bool _started;
        private bool _dilationDirty;
        private int _frameCount;
        private float _lastLogTime;

        private const string ScenePermission = "com.oculus.permission.USE_SCENE";

        /// <summary>Raised after each depth frame is processed (filtering, normals computed, globals set).</summary>
        public event Action Updated;

        /// <summary>
        /// Provide an RGB texture as edge guide for bilateral depth filtering.
        /// Call each frame from RoomScanner with the passthrough camera frame.
        /// </summary>
        public void SetRGBGuide(Texture tex) => _rgbGuide = tex;

        private static readonly Vector3 ScaleFlipZ = new(1, 1, -1);

        /// <summary>
        /// Convert a pose from XR tracking space to Unity world space.
        /// Required because MRUK's world-lock may offset TrackingSpace from the XROrigin root.
        /// </summary>
        public Pose TrackingToWorld(Pose trackingPose)
        {
            if (_trackingSpaceTransform == null) return trackingPose;
            return new Pose(
                _trackingSpaceTransform.TransformPoint(trackingPose.position),
                _trackingSpaceTransform.rotation * trackingPose.rotation);
        }

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            // Editor + no XR loader = AR subsystems will all be null and any
            // toggle of AROcclusionManager.enabled blows up DestroyTextures.
            // Build/run on Quest (or via Link) to actually scan.
            if (!XRRuntimeGuard.IsXRActive)
            {
                Logger.Warning("DepthCapture: " + XRRuntimeGuard.EditorDisabledMessage);
                enabled = false;
                return;
            }

            EnsureARSession();

            _arOcclusionManager = FindFirstObjectByType<AROcclusionManager>();
            if (!_arOcclusionManager)
                throw new Exception("[RoomScan] AROcclusionManager not found in scene");

            _xrOrigin = FindFirstObjectByType<Unity.XR.CoreUtils.XROrigin>();
            CacheTrackingSpaceTransform();

            _normKernel = new ComputeKernelHelper(depthNormalCompute, "DepthNorm");
            _monoConvertKernel = new ComputeKernelHelper(depthNormalCompute, "MonoRawDepthToStereo");
            _initDilateKernel = new ComputeKernelHelper(depthDilationCompute, "InitDepthDilation");
            _dilateStepKernel = new ComputeKernelHelper(depthDilationCompute, "DilateDepthStep");
            if (bilateralFilterCompute != null)
            {
                _bilateralKernel = new ComputeKernelHelper(bilateralFilterCompute, "BilateralFilter");
                _hasBilateralKernel = true;
            }
            if (depthEdgeCleanCompute != null)
            {
                _edgeCleanKernel = new ComputeKernelHelper(depthEdgeCleanCompute, "DepthEdgeClean");
                _hasEdgeCleanKernel = true;
            }

            _dilationMaxStep = 1;
            for (int i = 0; i < dilationSteps; i++)
                _dilationMaxStep *= 2;

            // Disable occlusion manager initially, enable after permission is confirmed
            _arOcclusionManager.enabled = false;
            CheckPermissionAndEnable();

            _started = true;
        }

        /// <summary>
        /// Resolves the TrackingSpace transform — the parent of the XR cameras that
        /// MRUK world-lock can reposition each frame. Using this instead of the XROrigin
        /// root ensures depth-to-world conversion includes the world-lock offset.
        /// </summary>
        private void CacheTrackingSpaceTransform()
        {
            // Prefer OVRCameraRig.trackingSpace (most reliable on Meta devices)
            var ovrRig = FindFirstObjectByType<OVRCameraRig>();
            if (ovrRig != null && ovrRig.trackingSpace != null)
            {
                _trackingSpaceTransform = ovrRig.trackingSpace;
                Logger.Info($"DepthCapture: using OVRCameraRig.trackingSpace '{_trackingSpaceTransform.name}'");
                return;
            }

            // Fallback: XROrigin.CameraFloorOffsetObject
            if (_xrOrigin != null && _xrOrigin.CameraFloorOffsetObject != null)
            {
                _trackingSpaceTransform = _xrOrigin.CameraFloorOffsetObject.transform;
                Logger.Info($"DepthCapture: using XROrigin.CameraFloorOffsetObject '{_trackingSpaceTransform.name}'");
                return;
            }

            // Last resort: XROrigin root (pre-fix behaviour)
            _trackingSpaceTransform = _xrOrigin != null ? _xrOrigin.transform : null;
            Logger.Warning("DepthCapture: no TrackingSpace found, falling back to XROrigin root");
        }

        private void EnsureARSession()
        {
            if (FindFirstObjectByType<ARSession>() == null)
            {
                var go = new GameObject("[AR Session]");
                go.AddComponent<ARSession>();
                Logger.Info("Created ARSession (was missing from scene)");
            }
        }

        private void CheckPermissionAndEnable()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (Permission.HasUserAuthorizedPermission(ScenePermission))
            {
                EnableOcclusion();
            }
            else
            {
                Logger.Info("Requesting USE_SCENE permission...");
                var callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += _ => EnableOcclusion();
                callbacks.PermissionDenied += _ => Logger.Error("USE_SCENE permission denied — depth will not work");
                Permission.RequestUserPermission(ScenePermission, callbacks);
            }
#else
            EnableOcclusion();
#endif
        }

        private bool _subscribed;

        private async void EnableOcclusion()
        {
            if (_arOcclusionManager == null) return;

            Logger.Info("Verifying AROcclusionManager subsystem...");

            _arOcclusionManager.frameReceived -= OnDepthFrame;
            _arOcclusionManager.enabled = false;

            await Awaitable.NextFrameAsync();
            await Awaitable.NextFrameAsync();

            if (_arOcclusionManager == null) return;

            // Briefly enable to verify the subsystem is functional
            _arOcclusionManager.enabled = true;

            await Awaitable.NextFrameAsync();
            await Awaitable.NextFrameAsync();

            if (_arOcclusionManager == null) return;
            var sub = _arOcclusionManager.subsystem;
            Logger.Info($"Occlusion subsystem: {(sub != null ? sub.GetType().Name : "null")}, running={sub?.running}");

            _permissionReady = true;

            if (_captureActive)
            {
                _arOcclusionManager.frameReceived += OnDepthFrame;
                _subscribed = true;
                Logger.Info("DepthCapture: subsystem left running (scan already active)");
            }
            else
            {
                _arOcclusionManager.enabled = false;
                Logger.Info("DepthCapture: subsystem disabled (no active scan)");
            }
        }

        /// <summary>
        /// Enables the AROcclusionManager and subscribes to depth frames.
        /// Called by RoomScanner when scanning starts.
        /// </summary>
        public void StartDepthCapture()
        {
            _captureActive = true;
            if (!_permissionReady || _arOcclusionManager == null) return;
            if (!_arOcclusionManager.enabled)
                _arOcclusionManager.enabled = true;
            if (!_subscribed)
            {
                _arOcclusionManager.frameReceived += OnDepthFrame;
                _subscribed = true;
            }
            Logger.Info("DepthCapture: subsystem started");
        }

        /// <summary>
        /// Unsubscribes from depth frames and disables the AROcclusionManager,
        /// stopping the depth sensor and neural inference pipeline on Quest.
        /// Called by RoomScanner when scanning stops.
        /// </summary>
        public void StopDepthCapture()
        {
            _captureActive = false;
            if (_arOcclusionManager != null)
            {
                if (_subscribed)
                {
                    _arOcclusionManager.frameReceived -= OnDepthFrame;
                    _subscribed = false;
                }
                _arOcclusionManager.enabled = false;
            }
            DepthAvailable = false;
        }

        private void OnApplicationPause(bool paused)
        {
            if (!_started) return;

            if (paused)
            {
                if (_arOcclusionManager != null)
                {
                    _arOcclusionManager.frameReceived -= OnDepthFrame;
                    _arOcclusionManager.enabled = false;
                    _subscribed = false;
                }
                DepthAvailable = false;
            }
            else if (_captureActive)
            {
                CheckPermissionAndEnable();
            }
        }

        private void OnDisable()
        {
            if (_arOcclusionManager != null && _subscribed)
            {
                _arOcclusionManager.frameReceived -= OnDepthFrame;
                _subscribed = false;
            }
        }

        private void OnDestroy()
        {
            ReleaseResources();
        }

        /// <summary>
        /// Destroys GPU textures (normals, dilation, filtered depth) to free memory.
        /// Textures are lazily recreated when the next depth frame arrives.
        /// </summary>
        public void ReleaseResources()
        {
            if (_normTex) { Destroy(_normTex); _normTex = null; }
            if (_dilationA) { Destroy(_dilationA); _dilationA = null; }
            if (_dilationB) { Destroy(_dilationB); _dilationB = null; }
            if (_simulatedDepthTex) { Destroy(_simulatedDepthTex); _simulatedDepthTex = null; }
            if (_filteredDepthTex) { Destroy(_filteredDepthTex); _filteredDepthTex = null; }
            if (_edgeCleanedDepthTex) { Destroy(_edgeCleanedDepthTex); _edgeCleanedDepthTex = null; }
            if (_edgeReasonTex) { Destroy(_edgeReasonTex); _edgeReasonTex = null; }
            _edgeStats?.Release();
            _edgeStats = null;
            _edgeStatsReadbackPending = false;
            HasEdgeCleanStats = false;
            _dilatedDepth = null;
            Logger.Info("DepthCapture: GPU resources released");
        }

        private void Update()
        {
            float t = Time.unscaledTime;
            if (t - _lastLogTime >= 5f)
            {
                _lastLogTime = t;
                var sub = _arOcclusionManager != null ? _arOcclusionManager.subsystem : null;
                Logger.Info($"DepthCapture: frames={_frameCount}, depthAvail={DepthAvailable}, " +
                          $"occMgr.enabled={_arOcclusionManager?.enabled}, sub={sub?.GetType().Name ?? "null"}, " +
                          $"running={sub?.running}");
            }
        }

        private void OnDepthFrame(AROcclusionFrameEventArgs args)
        {
            _frameCount++;
            if (_frameCount <= 3 || _frameCount % 100 == 0)
                Logger.Info($"OnDepthFrame #{_frameCount}, textures={args.externalTextures.Count}");

            if (Application.isEditor)
                HandleEditorSimulation(args);
            else
                HandleDeviceDepth(args);

            if (!DepthAvailable) return;

            ApplyBilateralFilter();
            ApplyDepthEdgeClean();
            SetGlobalShaderProperties();
            ComputeNormals();
            _dilationDirty = true;

            Updated?.Invoke();
        }

        /// <summary>
        /// Run dilation if depth has been updated since last call.
        /// Called by VolumeIntegrator before integration (not every frame).
        /// </summary>
        public void UpdateDilationIfNeeded()
        {
            if (!_dilationDirty || !DepthAvailable) return;
            ComputeDilation();
            _dilationDirty = false;
        }

        private void HandleEditorSimulation(AROcclusionFrameEventArgs args)
        {
            Texture rawDepth = args.externalTextures[0].texture;
            DepthAvailable = rawDepth != null;
            if (!DepthAvailable) return;

            if (_simulatedDepthTex == null ||
                _simulatedDepthTex.width != rawDepth.width ||
                _simulatedDepthTex.height != rawDepth.height)
            {
                if (_simulatedDepthTex) Destroy(_simulatedDepthTex);
                _simulatedDepthTex = new RenderTexture(rawDepth.width, rawDepth.height, 0,
                    GraphicsFormat.R16_UNorm, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    enableRandomWrite = true
                };
            }

            _monoConvertKernel.Set(DepthTexRWID, _simulatedDepthTex);
            _monoConvertKernel.Set(InputRawMonoDepthID, rawDepth);
            _monoConvertKernel.DispatchFit(rawDepth.width, rawDepth.height);
            _depthTex = _simulatedDepthTex;

            if (!_mainCam) _mainCam = Camera.main;
            if (!_mainCam) return;

            Matrix4x4 p = _mainCam.projectionMatrix;
            Matrix4x4 pi = p.inverse;
            Transform ct = _mainCam.transform;
            Matrix4x4 vi = Matrix4x4.TRS(ct.position, ct.rotation, ScaleFlipZ);
            Matrix4x4 v = vi.inverse;

            for (int i = 0; i < 2; i++)
            {
                _proj[i] = p;
                _projInv[i] = pi;
                _view[i] = v;
                _viewInv[i] = vi;
            }

            _planes = new Vector2(_mainCam.nearClipPlane, _mainCam.farClipPlane);
        }

        private void HandleDeviceDepth(AROcclusionFrameEventArgs args)
        {
            _depthTex = args.externalTextures[0].texture;

            ReadOnlyList<XRFov> fovs = default;
            ReadOnlyList<Pose> poses = default;
            XRNearFarPlanes depthPlanes = default;

            DepthAvailable = _depthTex != null &&
                             args.TryGetFovs(out fovs) &&
                             args.TryGetPoses(out poses) &&
                             args.TryGetNearFarPlanes(out depthPlanes);

            if (!DepthAvailable) return;

            for (int i = 0; i < 2; i++)
            {
                _proj[i] = CalculateProjectionMatrix(fovs[i], depthPlanes);
                _projInv[i] = Matrix4x4.Inverse(_proj[i]);

                Pose pose = poses[i];
                Matrix4x4 depthFrameMat = Matrix4x4.TRS(pose.position, pose.rotation, ScaleFlipZ);

                Matrix4x4 worldToTracking = _trackingSpaceTransform != null
                    ? _trackingSpaceTransform.worldToLocalMatrix
                    : Matrix4x4.identity;

                _view[i] = depthFrameMat.inverse * worldToTracking;
                _viewInv[i] = Matrix4x4.Inverse(_view[i]);
            }

            _planes = new Vector2(depthPlanes.nearZ, depthPlanes.farZ);
        }

        private bool _loggedBilateralSkip;
        private void ApplyBilateralFilter()
        {
            if (!enableBilateralFilter || !_hasBilateralKernel || _rgbGuide == null || _depthTex == null)
            {
                if (!_loggedBilateralSkip && enableBilateralFilter && _hasBilateralKernel && _rgbGuide == null)
                {
                    _loggedBilateralSkip = true;
                    Logger.Info("Bilateral depth filter skipped — no RGB guide (camera unavailable). " +
                              "Depth will be noisier at edges.");
                }
                return;
            }

            int w = _depthTex.width;
            int h = _depthTex.height;

            if (_filteredDepthTex == null || _filteredDepthTex.width != w || _filteredDepthTex.height != h)
            {
                if (_filteredDepthTex) Destroy(_filteredDepthTex);
                _filteredDepthTex = new RenderTexture(w, h, 0, GraphicsFormat.R16_UNorm, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                _filteredDepthTex.Create();
            }

            var cs = bilateralFilterCompute;
            _bilateralKernel.Set(BilSrcDepthID, _depthTex);
            _bilateralKernel.Set(BilRGBGuideID, _rgbGuide);
            _bilateralKernel.Set(BilDstDepthID, _filteredDepthTex);
            cs.SetInt(BilDepthWID, w);
            cs.SetInt(BilDepthHID, h);
            cs.SetFloat(BilSigmaSpatialID, sigmaSpatial);
            cs.SetFloat(BilSigmaColorID, sigmaColor);
            cs.SetFloat(BilSigmaDepthID, sigmaDepth);
            cs.SetInt(BilFilterRadiusID, filterRadius);

            _bilateralKernel.DispatchFit(w, h, 2);

            _depthTex = _filteredDepthTex;
        }

        /// <summary>
        /// 深度边缘清洗（v0.2 IsDepthEdgeTexCoord 的 GPU 移植）：三段式判决找断崖像素并写 0。
        /// 夹在双边滤波与法线/膨胀之间，原位替换 _depthTex（与双边同模式），融合 shader 零改动。
        /// 边缘像素 NDC 0 = 近端，融合侧近距闸/截断带双保险拦下 = 弃权，不投"远=空"票。
        /// </summary>
        private void ApplyDepthEdgeClean()
        {
            if (!enableDepthEdgeClean || !_hasEdgeCleanKernel || _depthTex == null) return;

            int w = _depthTex.width;
            int h = _depthTex.height;

            if (_edgeCleanedDepthTex == null || _edgeCleanedDepthTex.width != w || _edgeCleanedDepthTex.height != h)
            {
                if (_edgeCleanedDepthTex) Destroy(_edgeCleanedDepthTex);
                _edgeCleanedDepthTex = new RenderTexture(w, h, 0, GraphicsFormat.R16_UNorm, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                _edgeCleanedDepthTex.Create();
            }

            if (_edgeReasonTex == null || _edgeReasonTex.width != w || _edgeReasonTex.height != h)
            {
                if (_edgeReasonTex) Destroy(_edgeReasonTex);
                _edgeReasonTex = new RenderTexture(w, h, 0, GraphicsFormat.R32_UInt, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                _edgeReasonTex.Create();
            }

            if (_edgeStats == null)
            {
                _edgeStats = new ComputeBuffer(2, sizeof(uint));
                _edgeStats.SetData(ZeroEdgeStats);
            }

            var cs = depthEdgeCleanCompute;
            _edgeCleanKernel.Set(BilSrcDepthID, _depthTex);
            _edgeCleanKernel.Set(BilDstDepthID, _edgeCleanedDepthTex);
            _edgeCleanKernel.Set(EdgeReasonRWID, _edgeReasonTex);
            _edgeCleanKernel.Set(EdgeStatsID, _edgeStats);
            cs.SetMatrixArray(EdgeProjInvID, _projInv);
            cs.SetMatrixArray(EdgeViewInvID, _viewInv);
            _linearizeAB[0] = new Vector4(_proj[0][2, 2], _proj[0][2, 3], 0f, 0f);
            _linearizeAB[1] = new Vector4(_proj[1][2, 2], _proj[1][2, 3], 0f, 0f);
            cs.SetVectorArray(EdgeLinearizeABID, _linearizeAB);
            cs.SetInt(BilDepthWID, w);
            cs.SetInt(BilDepthHID, h);
            cs.SetInt(EdgeProbeRadiusID, edgeProbeRadiusPixels);
            cs.SetFloat(EdgeJumpBaseID, edgeJumpBaseMeters);
            cs.SetFloat(EdgeJumpDistScaleID, edgeJumpDistanceScale);
            cs.SetInt(EdgeUsePlaneRejectID, edgeUseLocalPlaneReject ? 1 : 0);
            cs.SetFloat(EdgePlaneResidualBaseID, edgePlaneResidualBaseMeters);
            cs.SetFloat(EdgePlaneResidualDistScaleID, edgePlaneResidualDistanceScale);
            cs.SetFloat(EdgePlaneMinInlierRatioID, edgePlaneMinInlierRatio);
            cs.SetInt(EdgePlaneMinNeighborsID, edgePlaneMinNeighborPoints);
            cs.SetFloat(EdgeTwoClusterMinFractionID, edgeTwoClusterMinFraction);
            cs.SetInt(SkirtVetoEnabledID, edgeSkirtVetoEnabled ? 1 : 0);
            cs.SetInt(SkirtMinNearProbesID, edgeSkirtMinNearProbes);
            cs.SetInt(GrazingVetoEnabledID, edgeGrazingVetoEnabled ? 1 : 0);
            cs.SetFloat(GrazingMinFacingDotID, edgeGrazingMinFacingDot);
            cs.SetFloat(GrazingRangeBaseID, edgeGrazingRangeBaseMeters);
            cs.SetFloat(GrazingRangeScaleID, edgeGrazingRangeScale);
            cs.SetInt(CrossEyeEnabledID, edgeCrossEyeEnabled ? 1 : 0);
            cs.SetFloat(CrossEyeMarginBaseID, edgeCrossEyeMarginBaseMeters);
            cs.SetFloat(CrossEyeMarginScaleID, edgeCrossEyeMarginScale);
            cs.SetFloat(CrossEyeMinOtherFacingID, edgeCrossEyeMinOtherFacing);
            cs.SetMatrixArray(EdgeProjFwdID, _proj);
            cs.SetMatrixArray(EdgeViewFwdID, _view);

            _edgeCleanKernel.DispatchFit(w, h, 2);

            _depthTex = _edgeCleanedDepthTex;

            // 边缘计数读数（HUD"缘:"）：每 15 次清洗结算一次，读回即清零开新周期
            _edgeCleansSinceStats++;
            if (_edgeCleansSinceStats >= 15 && !_edgeStatsReadbackPending)
            {
                _edgeCleansSinceStats = 0;
                _edgeStatsReadbackPending = true;
                AsyncGPUReadback.Request(_edgeStats, OnEdgeStatsReadback);
            }
        }

        private void OnEdgeStatsReadback(AsyncGPUReadbackRequest request)
        {
            _edgeStatsReadbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < 2) return;
            LastEdgeCleanCount = data[0];
            LastGrazingKillCount = data[1];
            HasEdgeCleanStats = true;
            _edgeStats?.SetData(ZeroEdgeStats);
        }

        private void SetGlobalShaderProperties()
        {
            Shader.SetGlobalMatrixArray(ProjID, _proj);
            Shader.SetGlobalMatrixArray(ProjInvID, _projInv);
            Shader.SetGlobalMatrixArray(ViewID, _view);
            Shader.SetGlobalMatrixArray(ViewInvID, _viewInv);
            Shader.SetGlobalVector(ZParamsID, _planes);
            Shader.SetGlobalVector(TexSizeID, new Vector2(_depthTex.width, _depthTex.height));
            Shader.SetGlobalTexture(DepthTexID, _depthTex);
        }

        private void ComputeNormals()
        {
            if (_normTex == null || _normTex.width != _depthTex.width || _normTex.height != _depthTex.height)
            {
                if (_normTex) Destroy(_normTex);
                _normTex = new RenderTexture(_depthTex.width, _depthTex.height, 0,
                    GraphicsFormat.R8G8B8A8_SNorm, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    useMipMap = false,
                    enableRandomWrite = true
                };
            }

            _normKernel.Set(DepthTexID, _depthTex);
            _normKernel.Set(NormTexRWID, _normTex);
            _normKernel.DispatchFit(_normTex);
            Shader.SetGlobalTexture(NormTexID, _normTex);
        }

        private void ComputeDilation()
        {
            if (_dilationA == null || _dilationA.width != _depthTex.width || _dilationA.height != _depthTex.height)
            {
                if (_dilationA) Destroy(_dilationA);
                if (_dilationB) Destroy(_dilationB);

                var desc = new RenderTextureDescriptor
                {
                    width = _depthTex.width,
                    height = _depthTex.height,
                    volumeDepth = 1,
                    dimension = TextureDimension.Tex2D,
                    autoGenerateMips = false,
                    enableRandomWrite = true,
                    graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat,
                    msaaSamples = 1
                };

                _dilationA = new RenderTexture(desc);
                _dilationB = new RenderTexture(desc);
            }

            depthDilationCompute.SetFloat(VoxDistID, voxelDistance);
            depthDilationCompute.SetFloat(VoxSizeShaderID, voxelSize);

            _initDilateKernel.Set(DepthTexID, _depthTex);
            _initDilateKernel.Set(DilateSrcID, _dilationA);
            _initDilateKernel.Set(DilateDestID, _dilationB);
            _initDilateKernel.DispatchFit(_dilationA.width, _dilationA.height);

            int stepSize = _dilationMaxStep;
            for (int i = 0; i < dilationSteps; i++)
            {
                _dilateStepKernel.Set(DilateSrcID, _dilationA);
                _dilateStepKernel.Set(DilateDestID, _dilationB);
                depthDilationCompute.SetInt(DilateStepSizeID, stepSize);
                _dilateStepKernel.DispatchFit(_dilationA.width, _dilationA.height);

                stepSize /= 2;
                (_dilationA, _dilationB) = (_dilationB, _dilationA);
            }

            _dilatedDepth = _dilationA;
            Shader.SetGlobalTexture(DilatedDepthTexID, _dilatedDepth);
        }

        private static Matrix4x4 CalculateProjectionMatrix(XRFov fov, XRNearFarPlanes planes)
        {
            float left = Mathf.Tan(fov.angleLeft);
            float right = Mathf.Tan(fov.angleRight);
            float bottom = Mathf.Tan(fov.angleDown);
            float top = Mathf.Tan(fov.angleUp);

            float near = planes.nearZ;
            float far = planes.farZ;

            float x = 2.0f / (right - left);
            float y = 2.0f / (top - bottom);
            float a = (right + left) / (right - left);
            float b = (top + bottom) / (top - bottom);

            float c, d;
            if (float.IsInfinity(far))
            {
                c = -1.0f;
                d = -2.0f * near;
            }
            else
            {
                c = -(far + near) / (far - near);
                d = -(2.0f * far * near) / (far - near);
            }

            return new Matrix4x4
            {
                m00 = x,  m01 = 0, m02 = a,  m03 = 0,
                m10 = 0,  m11 = y, m12 = b,  m13 = 0,
                m20 = 0,  m21 = 0, m22 = c,  m23 = d,
                m30 = 0,  m31 = 0, m32 = -1, m33 = 0
            };
        }

        /// <summary>
        /// Update voxel parameters used by dilation (called by VolumeIntegrator when its values change).
        /// </summary>
        public void SetVoxelParams(float voxDist, float voxSize)
        {
            voxelDistance = voxDist;
            voxelSize = voxSize;
        }
    }
}

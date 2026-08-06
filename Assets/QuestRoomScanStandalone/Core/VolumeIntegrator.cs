using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Manages the GPU TSDF + color volume and dispatches compute-shader integration, pruning,
    /// and freeze/unfreeze passes. Voxels are integrated from depth and optional camera color
    /// each frame, with configurable convergence, exclusion zones, and warmup clearing.
    /// </summary>
    public class VolumeIntegrator : MonoBehaviour
    {
        public static VolumeIntegrator Instance { get; private set; }

        [SerializeField] private ComputeShader compute;

        [Header("Volume")]
        [SerializeField] private int3 voxelCount = new(256, 256, 256);
        [SerializeField] private float voxelSize = 0.05f;
        [SerializeField] private float voxelDistance = 0.15f;
        [SerializeField] private float voxelMin = 0.1f;

        [Header("Integration")]
        [SerializeField] private float depthDisparityThreshold = 0.5f;
        [SerializeField] private float maxUpdateDist = 5f;
        [Tooltip("最近更新距离：视锥内距相机小于此值的体素不积分。原 0.5m 会在头周留下覆盖空洞，0.15 贴脸也覆盖（近距深度噪声大，如出现贴脸飞面可调回 0.3）")]
        [SerializeField] private float minUpdateDist = 0.15f;
        [Tooltip("头部排除区半径（仅 StandaloneRoomScanner 开启排除区时生效）。0.6 = QRS 原版（防手入网但头周覆盖空洞大），0.35 = 折中")]
        [SerializeField, Range(0.1f, 1f)] private float exclusionRadius = 0.35f;
        [SerializeField] private int maxFrustumPositions = 1000000;

        [Header("Convergence")]
        [Tooltip("Blend strength. Higher = faster convergence and correction. (default 0.8)")]
        [SerializeField, Range(0.1f, 2f)] private float blendRate = 0.8f;
        [Tooltip("Weight resistance to blending. Lower = faster corrections but less stable. (default 2.5)")]
        [SerializeField, Range(0.5f, 10f)] private float stability = 2.5f;
        [Tooltip("How fast weight accumulates per frame. Lower = bad data builds less confidence. (default 0.025)")]
        [SerializeField, Range(0.005f, 0.1f)] private float weightGrowth = 0.025f;
        [Tooltip("Maximum weight any voxel can reach. Lower = all areas correct equally fast. (default 0.5)")]
        [SerializeField, Range(0.1f, 1f)] private float maxWeight = 0.5f;
        [Tooltip("矛盾扣减率（投票制反对票）。新观测与存量矛盾（更空 >3cm）时每帧扣 q²·carveGain；weight 跌破 minMeshWeight 网格消失、跌破 0.05 被 Prune 回收。0 = 关闭，恢复 QRS 原版纯驻留行为。 (default 0.075)")]
        [SerializeField, Range(0f, 0.3f)] private float carveGain = 0.075f;
        [Tooltip("排除区不对称：开=排除区内只拦写（播种/增长）不拦抹（矛盾扣减放行，治头周圆柱内幽灵冻结永驻）；关=QRS 原版双向封锁。 (default true)")]
        [SerializeField] private bool carveInsideExclusion = true;
        [Tooltip("近距采样闸：测量距离 < minUpdateDist 的深度样本整票作废（挡胸口/手臂近距观测对身前幽灵的反向加固），与视锥体素闸语义对齐。 (default true)")]
        [SerializeField] private bool rejectNearSamples = true;
        [Tooltip("膨胀闸不对称：被近表面膨胀足迹罩住的体素只拦写（播种/增长）不拦抹——当帧原始深度确实看穿时矛盾扣减绕行。" +
                 "治缝隙区（柜边↔侧墙）幽灵一旦种入永不收反对票、假桥只长不消。关=恢复遮挡闸双向封锁。 (default true)")]
        [SerializeField] private bool carveBypassDilation = true;
        [Tooltip("法线闸不对称：掠射观测只拦写不拦抹——'射线沿途为空'的证据与表面朝向无关。" +
                 "治斜视时矛盾票被法线闸全拦（普查法拦暴涨）、桥只有正视能消。此通道用 carveBypassMargin 强裕量防误抹真面。 (default true)")]
        [SerializeField] private bool carveBypassNormal = true;
        [Tooltip("法绕抹的矛盾裕量（带宽单位，1=截断带宽 0.15m）：掠射深度噪声大，新观测要比存量空出这么多才扣减。" +
                 "默认 0.6≈9cm；实机按距离放大——1.5m 内不变，之外乘 voxEyeDist/1.5（远距掠射噪声 5~10cm 防误啃真地板）。误抹真面=调大，桥消不动=调小。 (default 0.6)")]
        [SerializeField, Range(0.2f, 1.5f)] private float carveBypassMargin = 0.6f;
        [Tooltip("绕行扣减（胀绕/法绕）独立倍率：绕行通道自带双重保险（被闸拦+强裕量才触发），加力只加速清尸体，" +
                 "不动 carveGain 主通道护真面。尸体消得慢=调大；真面被误扣=先查 margin 再调回 1。 (default 2)")]
        [SerializeField, Range(1f, 4f)] private float carveBypassBoost = 2f;
        [Tooltip("弃权邻域写闸：边缘清洗弃权像素周边 1px 内只拦写（播种/增长）不拦抹（矛盾扣减照常）。" +
                 "治帘锚点旁漏网裙边零星复种导致桥闪断闪连。真折角切口旁晚一两帧种上，代价可忽略。 (default true)")]
        [SerializeField] private bool abstainSeedGuard = true;

        [Header("运动闸")]
        [Tooltip("头显角速度超过此值（°/s）时整帧停笔：不写入、不增长、不矛盾扣减（预览/提取照常）。根治位姿-深度不同帧欠账下转头把表面抹出搓衣板褶皱、错位矛盾票啃真表面。消幽灵的正确姿势变为\"停下来正对看一秒\"。0 = 关闭。 (default 90)")]
        [SerializeField, Range(0f, 360f)] private float motionGateDegPerSec = 90f;

        [Header("Meshing")]
        [Tooltip("Min voxel confidence weight for Surface Nets to generate mesh. Higher = fewer phantom surfaces. (default 0.08)")]
        [SerializeField, Range(0.01f, 0.5f)] private float minMeshWeight = 0.08f;
        public float MinMeshWeight => minMeshWeight;

        [Header("Camera Color")]
        [Tooltip("Exposure boost for camera texture. Quest 3 passthrough cameras produce dim images. (default 3.0)")]
        [SerializeField, Range(1f, 10f)] private float cameraExposure = 3f;

        private RenderTexture _volume;
        private RenderTexture _colorVolume;

        /// <summary>3D RenderTexture (R8G8_SNorm) storing the truncated signed distance field.</summary>
        public RenderTexture Volume => _volume;
        /// <summary>3D RenderTexture (RGBA8_UNorm) storing per-voxel accumulated color.</summary>
        public RenderTexture ColorVolume => _colorVolume;
        public int3 VoxelCount => voxelCount;
        public float VoxelSize => voxelSize;
        public float VoxelDistance => voxelDistance;

        private static readonly int VolumeRWID = Shader.PropertyToID("gsVolumeRW");
        private static readonly int VolumeID = Shader.PropertyToID("gsVolume");
        private static readonly int ColorVolumeRWID = Shader.PropertyToID("gsColorVolumeRW");
        private static readonly int ColorVolumeID = Shader.PropertyToID("gsColorVolume");
        private static readonly int VoxCountID = Shader.PropertyToID("gsVoxCount");
        private static readonly int VoxSizeID = Shader.PropertyToID("gsVoxSize");
        private static readonly int VoxMinID = Shader.PropertyToID("gsVoxMin");
        private static readonly int VoxDistID = Shader.PropertyToID("gsVoxDist");
        private static readonly int FrustumVolumeID = Shader.PropertyToID("gsFrustumVolume");
        private static readonly int DepthDispThreshID = Shader.PropertyToID("gsDepthDispThresh");
        private static readonly int NumExclusionsID = Shader.PropertyToID("gsNumExclusions");
        private static readonly int ExclusionHeadsID = Shader.PropertyToID("gsExclusionHeads");
        private static readonly int ExclusionRadiusID = Shader.PropertyToID("gsExclusionRadius");
        private static readonly int MaxUpdateDistID = Shader.PropertyToID("gsMaxUpdateDist");
        private static readonly int BlendRateID = Shader.PropertyToID("gsBlendRate");
        private static readonly int StabilityID = Shader.PropertyToID("gsStability");
        private static readonly int WeightGrowthID = Shader.PropertyToID("gsWeightGrowth");
        private static readonly int MaxWeightID = Shader.PropertyToID("gsMaxWeight");
        private static readonly int CarveGainID = Shader.PropertyToID("gsCarveGain");
        private static readonly int MinUpdateDistID = Shader.PropertyToID("gsMinUpdateDist");
        private static readonly int CarveInsideExclusionID = Shader.PropertyToID("gsCarveInsideExclusion");
        private static readonly int CarveBypassDilationID = Shader.PropertyToID("gsCarveBypassDilation");
        private static readonly int CarveBypassNormalID = Shader.PropertyToID("gsCarveBypassNormal");
        private static readonly int CarveBypassMarginID = Shader.PropertyToID("gsCarveBypassMargin");
        private static readonly int CarveBypassBoostID = Shader.PropertyToID("gsCarveBypassBoost");
        private static readonly int AbstainSeedGuardID = Shader.PropertyToID("gsAbstainSeedGuard");
        private static readonly int CamRGBID = Shader.PropertyToID("gsCamRGB");
        private static readonly int CamAvailableID = Shader.PropertyToID("gsCamAvailable");
        private static readonly int CamPosID = Shader.PropertyToID("gsCamPos");
        private static readonly int CamInvRotID = Shader.PropertyToID("gsCamInvRot");
        private static readonly int CamFocalLenID = Shader.PropertyToID("gsCamFocalLen");
        private static readonly int CamPrincipalPtID = Shader.PropertyToID("gsCamPrincipalPt");
        private static readonly int CamSensorResID = Shader.PropertyToID("gsCamSensorRes");
        private static readonly int CamCurrentResID = Shader.PropertyToID("gsCamCurrentRes");
        private static readonly int CamExposureID = Shader.PropertyToID("gsCamExposure");

        public float CameraExposure => cameraExposure;

        [Header("Warmup")]
        [Tooltip("Clear the volume after this many integrations to discard sensor startup noise. 0 = disabled.")]
        [SerializeField] private int warmupIntegrations = 3;

        [Header("Pruning")]
        [SerializeField] private float pruneIntervalSeconds = 3f;

        private ComputeKernelHelper _clearKernel;
        private ComputeKernelHelper _integrateKernel;
        private ComputeKernelHelper _pruneKernel;
        private ComputeKernelHelper _freezeKernel;
        private ComputeKernelHelper _unfreezeKernel;

        private ComputeBuffer _frustumVolume;
        private bool _frustumReady;
        private float _lastPruneTime;

        // Coverage metrics
        private ComputeKernelHelper _coverageKernel;
        private ComputeBuffer _coverageCounters;
        private int _integrationsSinceCoverage;
        private bool _coverageReadbackPending;
        private static readonly int CoverageCountersID = Shader.PropertyToID("_CoverageCounters");
        private static readonly int ColorVolumeReadID = Shader.PropertyToID("gsColorVolumeRead");

        // 矛盾票普查：诊断"幽灵抹不掉"——反对票到底投没投出、被哪道门禁拦住
        private ComputeBuffer _carveStats;
        private bool _carveStatsReadbackPending;
        private static readonly uint[] ZeroCarveStats = new uint[8];
        /// <summary>最近一个统计周期的矛盾票计数：0票投出 1排除区拦 2法线闸拦 3遮挡闸拦 4带外拦 5排内抹（不对称放行实际扣减）。</summary>
        public readonly uint[] LastCarveStats = new uint[8];
        /// <summary>是否已有至少一轮矛盾票读回。</summary>
        public bool HasCarveStats { get; private set; }
        private static readonly int CarveStatsID = Shader.PropertyToID("_CarveStats");

        [Header("Coverage Metrics")]
        [Tooltip("Dispatch coverage count every N integrations (0 = disabled). Higher = less GPU overhead.")]
        [SerializeField] private int coverageUpdateInterval = 30;

        /// <summary>Number of voxels near the zero-crossing with sufficient weight (surface voxels).</summary>
        public int SurfaceVoxelCount { get; private set; }
        /// <summary>Number of surface voxels that are frozen (user-confirmed done).</summary>
        public int FrozenSurfaceCount { get; private set; }
        /// <summary>Number of surface voxels with camera color data (alpha &gt; 0.1).</summary>
        public int ColoredSurfaceCount { get; private set; }

        /// <summary>
        /// Transforms whose positions define spherical exclusion zones; voxels near these are skipped during integration.
        /// </summary>
        public readonly List<Transform> ExclusionZones = new();
        private readonly Vector4[] _exclusionPositions = new Vector4[64];

        /// <summary>Total number of integration passes dispatched since startup or the last clear.</summary>
        public int IntegrationCount { get; private set; }
        public int WarmupIntegrations => warmupIntegrations;

        /// <summary>Raised after each integration compute dispatch (before pruning).</summary>
        public event Action Integrated;
        /// <summary>Raised after the volume is cleared.</summary>
        public event Action Cleared;

        private Texture _pendingCamFrame;
        private Vector3 _pendingCamPos;
        private Quaternion _pendingCamRot;
        private Vector2 _pendingFocalLen;
        private Vector2 _pendingPrincipalPt;
        private Vector2 _pendingSensorRes;
        private Vector2 _pendingCurrentRes;
        private RenderTexture _camFrameCopy;
        private Texture2D _dummyCamTex;

        // 运动闸：积分位姿角速度（EMA 平滑）。深度位姿按帧阶跃更新，瞬时速度含 0/2× 交替噪声，EMA 收敛到真实转速。
        private Quaternion _lastIntegrateRot;
        private float _lastIntegrateTime;
        private bool _hasLastIntegrateRot;
        private float _smoothedAngSpeed;
        private int _motionGatedSinceStats;
        private int _lastMotionGatedCount;
        /// <summary>平滑后的积分位姿角速度（°/s），调试用。</summary>
        public float SmoothedAngularSpeed => _smoothedAngSpeed;

        private void Awake()
        {
            Instance = this;
            // GPU resources allocate lazily on the first scan / save / full-load
            // path via ReallocateVolumes(). The lightweight LoadRefinedOnlyAsync
            // path (returning-player and editor-sim) never touches them, so a
            // pure replay session avoids the ~150 MB TSDF+color RT footprint.
        }

        private void Start()
        {
            // Intentionally empty — see Awake().
            //
            // Historic note: kernel helpers + 3D RTs used to be constructed
            // here unconditionally. They're now created on demand inside
            // ReallocateVolumes(), which is called by:
            //   * RoomScanner.StartScanning() (every scan begin)
            //   * RoomScanPersistence.SaveToNewPackageAsync (defensive)
            //   * RoomScanPersistence.LoadPackageAsync (full TSDF reload)
        }

        /// <summary>
        /// Build all compute-kernel helpers and bind them to the current
        /// <see cref="_volume"/> / <see cref="_colorVolume"/>. Idempotent —
        /// the first <see cref="ReallocateVolumes"/> call constructs them;
        /// subsequent allocations only need <see cref="RebindKernelTextures"/>.
        /// </summary>
        private void InitKernels()
        {
            _clearKernel = new ComputeKernelHelper(compute, "Clear");
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);

            _integrateKernel = new ComputeKernelHelper(compute, "Integrate");
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);

            _pruneKernel = new ComputeKernelHelper(compute, "Prune");
            _pruneKernel.Set(VolumeRWID, _volume);
            _pruneKernel.Set(ColorVolumeRWID, _colorVolume);

            _freezeKernel = new ComputeKernelHelper(compute, "FreezeInFrustum");
            _freezeKernel.Set(VolumeRWID, _volume);

            _unfreezeKernel = new ComputeKernelHelper(compute, "UnfreezeInFrustum");
            _unfreezeKernel.Set(VolumeRWID, _volume);

            _coverageKernel = new ComputeKernelHelper(compute, "CountSurfaceCoverage");
            _coverageKernel.Set(VolumeRWID, _volume);
            _coverageCounters = new ComputeBuffer(3, sizeof(uint));
            _coverageKernel.Set(CoverageCountersID, _coverageCounters);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);

            _carveStats = new ComputeBuffer(8, sizeof(uint));
            _carveStats.SetData(ZeroCarveStats);
            _integrateKernel.Set(CarveStatsID, _carveStats);

            _dummyCamTex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _dummyCamTex.SetPixel(0, 0, Color.black);
            _dummyCamTex.Apply(false, true);
        }

        private void OnDestroy()
        {
            ReleaseVolumes();
            _coverageCounters?.Release();
            _coverageCounters = null;
            _carveStats?.Release();
            _carveStats = null;
            if (_camFrameCopy) Destroy(_camFrameCopy);
            if (_dummyCamTex) Destroy(_dummyCamTex);
        }

        /// <summary>
        /// Destroys the TSDF + color volume RenderTextures and the frustum buffer to free GPU memory.
        /// The component stays alive; calling <see cref="CreateVolume"/> + <see cref="SetShaderConstants"/>
        /// re-allocates everything (handled transparently by the integration path).
        /// </summary>
        public void ReleaseVolumes()
        {
            _frustumVolume?.Release();
            _frustumVolume = null;
            _frustumReady = false;
            if (_volume) { Destroy(_volume); _volume = null; }
            if (_colorVolume) { Destroy(_colorVolume); _colorVolume = null; }
            IntegrationCount = 0;
            Logger.Info("VolumeIntegrator: GPU volumes released");
        }

        /// <summary>True when volumes have been released and need re-allocation before integration.</summary>
        public bool VolumesReleased => _volume == null;

        /// <summary>
        /// Allocate (or re-allocate) TSDF + color volumes and bring kernels +
        /// shader constants up to date. Idempotent — early-returns if volumes
        /// already exist. Handles three scenarios:
        /// <list type="bullet">
        ///   <item><description><b>First-ever scan</b>: builds compute kernels
        ///   from scratch (deferred from the old eager <c>Awake</c>/<c>Start</c>
        ///   path), allocates RTs, sets globals.</description></item>
        ///   <item><description><b>Resume after <see cref="ReleaseVolumes"/></b>:
        ///   re-allocates RTs and rebinds existing kernels via
        ///   <see cref="RebindKernelTextures"/>.</description></item>
        ///   <item><description><b>Already allocated</b>: no-op.</description></item>
        /// </list>
        /// Called by <see cref="RoomScanner.StartScanningAsync"/> and the heavy
        /// <c>RoomScanPersistence</c> save/full-load paths. The lightweight
        /// <c>LoadRefinedOnlyAsync</c> path intentionally skips this.
        /// </summary>
        public void ReallocateVolumes()
        {
            if (_volume != null) return;

            // ComputeKernelHelper is a struct — use its readonly Shader
            // backing field as the "never initialized" sentinel.
            bool firstAlloc = (_clearKernel.Shader == null);

            CreateVolume();

            if (firstAlloc) InitKernels();
            else            RebindKernelTextures();

            SetShaderConstants();
            Clear();

            if (DepthCapture.Instance != null)
                DepthCapture.Instance.SetVoxelParams(voxelDistance, voxelSize);

            Logger.Info(firstAlloc
                ? "VolumeIntegrator: GPU resources allocated lazily on first scan/save/full-load."
                : "VolumeIntegrator: GPU volumes re-allocated after release.");
        }

        private void RebindKernelTextures()
        {
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _pruneKernel.Set(VolumeRWID, _volume);
            _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
            _freezeKernel.Set(VolumeRWID, _volume);
            _unfreezeKernel.Set(VolumeRWID, _volume);
            _coverageKernel.Set(VolumeRWID, _volume);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);
        }

        private void DispatchCoverageCount()
        {
            if (_volume == null || _coverageCounters == null) return;
            _coverageReadbackPending = true;

            uint[] zeros = { 0, 0, 0 };
            _coverageCounters.SetData(zeros);
            _coverageKernel.DispatchFit(_volume);

            AsyncGPUReadback.Request(_coverageCounters, OnCoverageReadback);
        }

        private void OnCoverageReadback(AsyncGPUReadbackRequest request)
        {
            _coverageReadbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < 3) return;
            SurfaceVoxelCount = (int)data[0];
            FrozenSurfaceCount = (int)data[1];
            ColoredSurfaceCount = (int)data[2];
        }

        private void RequestCarveStatsReadback()
        {
            if (_carveStats == null || _carveStatsReadbackPending) return;
            _carveStatsReadbackPending = true;
            AsyncGPUReadback.Request(_carveStats, OnCarveStatsReadback);
        }

        private void OnCarveStatsReadback(AsyncGPUReadbackRequest request)
        {
            _carveStatsReadbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < 6) return;
            for (int i = 0; i < 8; i++) LastCarveStats[i] = data[i];
            HasCarveStats = true;
            _carveStats.SetData(ZeroCarveStats); // 数据已落袋，清零开新周期
            _lastMotionGatedCount = _motionGatedSinceStats; // 运动闸同节奏结算
            _motionGatedSinceStats = 0;
        }

        private static string FormatCarveCount(uint v)
        {
            return v >= 10000u ? (v / 10000f).ToString("0.0") + "万" : v.ToString();
        }

        /// <summary>矛盾票普查的一行中文摘要（HUD 用）。</summary>
        public string GetCarveStatsCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"投{FormatCarveCount(LastCarveStats[0])} 排抹{FormatCarveCount(LastCarveStats[5])} " +
                   $"胀绕{FormatCarveCount(LastCarveStats[6])} 法绕{FormatCarveCount(LastCarveStats[7])} " +
                   $"排拦{FormatCarveCount(LastCarveStats[1])} 法拦{FormatCarveCount(LastCarveStats[2])} " +
                   $"遮拦{FormatCarveCount(LastCarveStats[3])} 带拦{FormatCarveCount(LastCarveStats[4])}" +
                   (_lastMotionGatedCount > 0 ? $" 动闸{_lastMotionGatedCount}" : "");
        }

        private void CreateVolume()
        {
            long tsdfBytes = (long)voxelCount.x * voxelCount.y * voxelCount.z * 2;
            long colorBytes = (long)voxelCount.x * voxelCount.y * voxelCount.z * 4;
            Logger.Info($"TSDF volume: {voxelCount} RG8_SNorm = {tsdfBytes / (1024 * 1024)}MB");
            Logger.Info($"Color volume: {voxelCount} RGBA8_UNorm = {colorBytes / (1024 * 1024)}MB");

            _volume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8G8_SNorm, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _volume.Create();

            _colorVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8G8B8A8_UNorm, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _colorVolume.Create();
        }

        private void SetShaderConstants()
        {
            int3 s = voxelCount;
            compute.SetInts(VoxCountID, s.x, s.y, s.z);
            Shader.SetGlobalVector(VoxCountID, new Vector4(s.x, s.y, s.z, 0));

            compute.SetFloat(VoxSizeID, voxelSize);
            Shader.SetGlobalFloat(VoxSizeID, voxelSize);

            compute.SetFloat(VoxMinID, voxelMin);
            compute.SetFloat(VoxDistID, voxelDistance);
            Shader.SetGlobalFloat(VoxDistID, voxelDistance);

            compute.SetFloat(DepthDispThreshID, depthDisparityThreshold);
            compute.SetFloat(MaxUpdateDistID, maxUpdateDist);
            compute.SetFloat(BlendRateID, blendRate);
            compute.SetFloat(StabilityID, stability);
            compute.SetFloat(WeightGrowthID, weightGrowth);
            compute.SetFloat(MaxWeightID, maxWeight);
            compute.SetFloat(CarveGainID, carveGain);
            compute.SetFloat(MinUpdateDistID, rejectNearSamples ? minUpdateDist : 0f);
            compute.SetFloat(CarveInsideExclusionID, carveInsideExclusion ? 1f : 0f);
            compute.SetFloat(CarveBypassDilationID, carveBypassDilation ? 1f : 0f);
            compute.SetFloat(CarveBypassNormalID, carveBypassNormal ? 1f : 0f);
            compute.SetFloat(CarveBypassMarginID, carveBypassMargin);
            compute.SetFloat(CarveBypassBoostID, carveBypassBoost);
            compute.SetFloat(AbstainSeedGuardID, abstainSeedGuard ? 1f : 0f);

            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);
        }

        /// <summary>
        /// Zeros the TSDF and color volumes on the GPU. No-op if volumes
        /// haven't been allocated yet (lazy alloc — see
        /// <see cref="ReallocateVolumes"/>).
        /// </summary>
        public void Clear()
        {
            if (_volume == null || _clearKernel.Shader == null) return;
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.DispatchFit(_volume);
            _carveStats?.SetData(ZeroCarveStats);
            HasCarveStats = false;
            Cleared?.Invoke();
        }

        /// <summary>
        /// Resample TSDF + color from the current (relocated) grid into a new identity grid.
        /// After this call the volume data lives in the current tracking/world frame.
        /// </summary>
        public void BakeRelocation(Matrix4x4 relocationMatrix)
        {
            if (_volume == null || _colorVolume == null || compute == null)
                return;

            Matrix4x4 invRelocation = relocationMatrix.inverse;
            int3 vc = voxelCount;

            var dstTsdf = new RenderTexture(vc.x, vc.y, 0, _volume.graphicsFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            dstTsdf.Create();

            var dstColor = new RenderTexture(vc.x, vc.y, 0, _colorVolume.graphicsFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            dstColor.Create();

            int kernel = compute.FindKernel("BakeRelocation");
            compute.SetInts(Shader.PropertyToID("gsVoxCount"), vc.x, vc.y, vc.z);
            compute.SetFloat(Shader.PropertyToID("gsVoxSize"), voxelSize);
            compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcTsdf"), _volume);
            compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcColor"), _colorVolume);
            compute.SetTexture(kernel, VolumeRWID, dstTsdf);
            compute.SetTexture(kernel, ColorVolumeRWID, dstColor);
            compute.SetMatrix(Shader.PropertyToID("gsBakeInvRelocation"), invRelocation);

            int tx = Mathf.CeilToInt(vc.x / 4f);
            int ty = Mathf.CeilToInt(vc.y / 4f);
            int tz = Mathf.CeilToInt(vc.z / 4f);
            compute.Dispatch(kernel, tx, ty, tz);
            GL.Flush();

            // Swap volumes: destroy old, adopt baked textures.
            // Avoids Graphics.CopyTexture on 3D RTs which can silently fail on Vulkan/Quest.
            Destroy(_volume);
            Destroy(_colorVolume);
            _volume = dstTsdf;
            _colorVolume = dstColor;

            // Rebind global texture references (used by render shader for freeze tint etc.)
            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);

            // Rebind per-kernel UAV references so subsequent integrations/clears use new textures
            RebindVolumeTextures();

            Logger.Info($"BakeRelocation complete — resampled {vc} voxels, " +
                      $"reloc row0={relocationMatrix.GetRow(0)}, inv row0={invRelocation.GetRow(0)}");
        }

        private void RebindVolumeTextures()
        {
            if (_clearKernel.Shader == null) return;
            RebindKernelTextures();
        }

        /// <summary>
        /// Freeze all voxels currently visible in the camera frustum.
        /// Frozen voxels are encoded as negative weight and skip integration.
        /// Requires camera data to have been provided via SetCameraData.
        /// </summary>
        public void FreezeInView(Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (_volume == null || _freezeKernel.Shader == null)
            {
                Logger.Warning("FreezeInView called before GPU resources allocated; ignored.");
                return;
            }
            SetFrustumCameraUniforms(_freezeKernel, camPos, camRot,
                focalLen, principalPt, sensorRes, currentRes);
            _freezeKernel.Set(VolumeRWID, _volume);
            _freezeKernel.DispatchFit(_volume);
            Logger.Info("FreezeInView dispatched");
        }

        /// <summary>
        /// Unfreeze all frozen voxels currently visible in the camera frustum.
        /// </summary>
        public void UnfreezeInView(Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (_volume == null || _unfreezeKernel.Shader == null)
            {
                Logger.Warning("UnfreezeInView called before GPU resources allocated; ignored.");
                return;
            }
            SetFrustumCameraUniforms(_unfreezeKernel, camPos, camRot,
                focalLen, principalPt, sensorRes, currentRes);
            _unfreezeKernel.Set(VolumeRWID, _volume);
            _unfreezeKernel.DispatchFit(_volume);
            Logger.Info("UnfreezeInView dispatched");
        }

        private void SetFrustumCameraUniforms(ComputeKernelHelper kernel, Vector3 camPos,
            Quaternion camRot, Vector2 focalLen, Vector2 principalPt,
            Vector2 sensorRes, Vector2 currentRes)
        {
            compute.SetVector(CamPosID, camPos);
            compute.SetMatrix(CamInvRotID, Matrix4x4.Rotate(Quaternion.Inverse(camRot)));
            compute.SetVector(CamFocalLenID, focalLen);
            compute.SetVector(CamPrincipalPtID, principalPt);
            compute.SetVector(CamSensorResID, sensorRes);
            compute.SetVector(CamCurrentResID, currentRes);
        }

        /// <summary>
        /// Provide a camera frame and intrinsics for color integration this tick.
        /// Uses direct pinhole projection (matching Meta PCA samples) instead of VP matrix.
        /// Call before Integrate() each frame. Pass null frame to skip color.
        /// </summary>
        public void SetCameraData(Texture frame, Vector3 camPos, Quaternion camRot,
            Vector2 focalLength, Vector2 principalPoint, Vector2 sensorRes, Vector2 currentRes)
        {
            _pendingCamFrame = frame;
            _pendingCamPos = camPos;
            _pendingCamRot = camRot;
            _pendingFocalLen = focalLength;
            _pendingPrincipalPt = principalPoint;
            _pendingSensorRes = sensorRes;
            _pendingCurrentRes = currentRes;
        }

        /// <summary>
        /// Ensures _camFrameCopy exists and blits the pending frame to it.
        /// Called internally before Integrate() uses it for compute shader color integration.
        /// </summary>
        private void EnsureCamFrameCopy()
        {
            if (_pendingCamFrame == null) return;
            int w = _pendingCamFrame.width;
            int h = _pendingCamFrame.height;
            if (_camFrameCopy == null || _camFrameCopy.width != w || _camFrameCopy.height != h)
            {
                if (_camFrameCopy) Destroy(_camFrameCopy);
                _camFrameCopy = new RenderTexture(w, h, 0, GraphicsFormat.R8G8B8A8_SRGB, 0)
                {
                    enableRandomWrite = false,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                _camFrameCopy.Create();
            }
            Graphics.Blit(_pendingCamFrame, _camFrameCopy);
        }

        /// <summary>
        /// Builds the frustum sample positions buffer used by the Integrate kernel.
        /// Called lazily on first integration or after a volume clear/load.
        /// </summary>
        public void SetupFrustumVolume()
        {
            if (!DepthCapture.DepthAvailable) return;

            Matrix4x4 depthProj = Shader.GetGlobalMatrixArray(DepthCapture.ProjID)[0];
            FrustumPlanes frustum = depthProj.decomposeProjection;
            frustum.zFar = maxUpdateDist;

            var positions = new List<Vector3>(Mathf.Min(maxFrustumPositions, 200000));

            float ls = frustum.left / frustum.zNear;
            float rs = frustum.right / frustum.zNear;
            float ts = frustum.top / frustum.zNear;
            float bs = frustum.bottom / frustum.zNear;

            float step = voxelSize;
            bool capped = false;

            for (float z = frustum.zNear; z < frustum.zFar && !capped; z += step)
            {
                float xMin = ls * z + step;
                float xMax = rs * z - step;
                float yMin = bs * z + step;
                float yMax = ts * z - step;

                for (float x = xMin; x < xMax && !capped; x += step)
                for (float y = yMin; y < yMax; y += step)
                {
                    var v = new Vector3(x, y, -z);
                    float mag = v.magnitude;
                    if (mag > minUpdateDist && mag < maxUpdateDist)
                    {
                        positions.Add(v);
                        if (positions.Count >= maxFrustumPositions)
                        {
                            capped = true;
                            break;
                        }
                    }
                }
            }

            if (positions.Count == 0) return;

            Logger.Info($"Frustum volume: {positions.Count} positions ({positions.Count * 12 / 1024}KB)");

            _frustumVolume?.Release();
            _frustumVolume = new ComputeBuffer(positions.Count, sizeof(float) * 3);
            _frustumVolume.SetData(positions);
            _integrateKernel.Set(FrustumVolumeID, _frustumVolume);
            _frustumReady = true;
        }

        /// <summary>
        /// Dispatches one TSDF + color integration pass from the current depth frame.
        /// Handles frustum setup, exclusion zones, warmup clearing, and periodic pruning.
        /// </summary>
        public void Integrate()
        {
            var dc = DepthCapture.Instance;
            if (dc == null || !DepthCapture.DepthAvailable || dc.DepthTex == null) return;
            // Defensive: with lazy GPU alloc a stray Integrate() before
            // ReallocateVolumes can land here. RoomScanner.StartScanning()
            // always calls ReallocateVolumes first, so this is just a
            // safety net.
            if (_volume == null || _integrateKernel.Shader == null) return;
            if (!_frustumReady) SetupFrustumVolume();
            if (!_frustumReady) return;

            // 运动闸：转头时积分位姿与深度帧存在帧差，写入会切向涂抹成搓衣板褶皱、
            // 矛盾票也会按错位投影啃到真表面。超阈值整帧停笔（不集成、不扣减），
            // 停下来正对目标时票照投——消幽灵的姿势是"停住看"，不是"转着磨"。
            if (motionGateDegPerSec > 0f)
            {
                var viewInv = dc.ViewInv;
                if (viewInv != null && viewInv.Length > 0)
                {
                    Quaternion curRot = viewInv[0].rotation;
                    float dt = Time.unscaledTime - _lastIntegrateTime;
                    if (_hasLastIntegrateRot && dt > 1e-5f)
                    {
                        float inst = Quaternion.Angle(_lastIntegrateRot, curRot) / dt;
                        _smoothedAngSpeed = Mathf.Lerp(_smoothedAngSpeed, inst, 0.35f);
                    }
                    _lastIntegrateRot = curRot;
                    _lastIntegrateTime = Time.unscaledTime;
                    _hasLastIntegrateRot = true;
                    if (_smoothedAngSpeed > motionGateDegPerSec)
                    {
                        _motionGatedSinceStats++;
                        _pendingCamFrame = null; // 丢弃过期颜色帧，防与下一帧位姿错配
                        return;
                    }
                }
            }

            dc.UpdateDilationIfNeeded();

            compute.SetMatrixArray(DepthCapture.ViewID, dc.View);
            compute.SetMatrixArray(DepthCapture.ProjID, dc.Proj);
            compute.SetMatrixArray(DepthCapture.ViewInvID, dc.ViewInv);
            compute.SetMatrixArray(DepthCapture.ProjInvID, dc.ProjInv);

            int numExclusions = Mathf.Min(ExclusionZones.Count, 64);
            for (int i = 0; i < numExclusions; i++)
            {
                if (ExclusionZones[i] != null)
                    _exclusionPositions[i] = ExclusionZones[i].position;
            }
            compute.SetInt(NumExclusionsID, numExclusions);
            compute.SetVectorArray(ExclusionHeadsID, _exclusionPositions);
            compute.SetFloat(ExclusionRadiusID, exclusionRadius);

            compute.SetFloat(BlendRateID, blendRate);
            compute.SetFloat(StabilityID, stability);
            compute.SetFloat(WeightGrowthID, weightGrowth);
            compute.SetFloat(MaxWeightID, maxWeight);
            compute.SetFloat(CarveGainID, carveGain);
            compute.SetFloat(MinUpdateDistID, rejectNearSamples ? minUpdateDist : 0f);
            compute.SetFloat(CarveInsideExclusionID, carveInsideExclusion ? 1f : 0f);
            compute.SetFloat(CarveBypassDilationID, carveBypassDilation ? 1f : 0f);
            compute.SetFloat(CarveBypassNormalID, carveBypassNormal ? 1f : 0f);
            compute.SetFloat(CarveBypassMarginID, carveBypassMargin);
            compute.SetFloat(CarveBypassBoostID, carveBypassBoost);
            compute.SetFloat(AbstainSeedGuardID, abstainSeedGuard ? 1f : 0f);

            EnsureCamFrameCopy();
            if (_pendingCamFrame != null && _camFrameCopy != null)
            {
                compute.SetTexture(_integrateKernel.KernelIndex, CamRGBID, _camFrameCopy);
                compute.SetInt(CamAvailableID, 1);
                compute.SetVector(CamPosID, _pendingCamPos);
                compute.SetMatrix(CamInvRotID, Matrix4x4.Rotate(Quaternion.Inverse(_pendingCamRot)));
                compute.SetVector(CamFocalLenID, _pendingFocalLen);
                compute.SetVector(CamPrincipalPtID, _pendingPrincipalPt);
                compute.SetVector(CamSensorResID, _pendingSensorRes);
                compute.SetVector(CamCurrentResID, _pendingCurrentRes);
                compute.SetFloat(CamExposureID, cameraExposure);
            }
            else
            {
                compute.SetTexture(_integrateKernel.KernelIndex, CamRGBID, _dummyCamTex);
                compute.SetInt(CamAvailableID, 0);
            }

            _integrateKernel.Set(DepthCapture.DepthTexID, dc.DepthTex);
            _integrateKernel.Set(DepthCapture.NormTexID, dc.NormTex);
            _integrateKernel.Set(DepthCapture.DilatedDepthTexID, dc.DilatedDepthTex);

            _integrateKernel.DispatchFit(_frustumVolume.count, 1);

            IntegrationCount++;
            _pendingCamFrame = null;

            if (warmupIntegrations > 0 && IntegrationCount == warmupIntegrations)
            {
                Logger.Info($"Warmup complete ({warmupIntegrations} frames), clearing volume to discard sensor startup noise");
                Clear();
            }

            float t = Time.time;
            if (t - _lastPruneTime >= pruneIntervalSeconds)
            {
                _lastPruneTime = t;
                _pruneKernel.Set(VolumeRWID, _volume);
                _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
                _pruneKernel.DispatchFit(_volume);
            }

            if (coverageUpdateInterval > 0 && !_coverageReadbackPending)
            {
                _integrationsSinceCoverage++;
                if (_integrationsSinceCoverage >= coverageUpdateInterval)
                {
                    _integrationsSinceCoverage = 0;
                    DispatchCoverageCount();
                    RequestCarveStatsReadback();
                }
            }

            Integrated?.Invoke();
        }

        /// <summary>
        /// Uploads CPU TSDF/color blobs into the 3D RenderTextures.
        /// Uses <see cref="GraphicsFormat"/> matching the volume RTs so
        /// <see cref="Graphics.CopyTexture"/> is valid on Metal/Vulkan (RG16 Texture3D ≠ R8G8_SNorm layout).
        /// </summary>
        public bool LoadVolumes(byte[] tsdfBytes, byte[] colorBytes, int integrationCount)
        {
            if (_volume == null || _colorVolume == null)
            {
                Logger.Error("Cannot load volumes: textures not created");
                return false;
            }

            int3 s = voxelCount;
            int expectedTsdf = s.x * s.y * s.z * 2;
            int expectedColor = s.x * s.y * s.z * 4;

            if (tsdfBytes.Length != expectedTsdf)
            {
                Logger.Error($"TSDF size mismatch: got {tsdfBytes.Length}, expected {expectedTsdf}");
                return false;
            }
            if (colorBytes.Length != expectedColor)
            {
                Logger.Error($"Color volume size mismatch: got {colorBytes.Length}, expected {expectedColor}");
                return false;
            }

            // Must match CreateVolume(): R8G8_SNorm TSDF + RGBA8_UNorm color
            var tsdfTex = new Texture3D(s.x, s.y, s.z, GraphicsFormat.R8G8_SNorm, TextureCreationFlags.None);
            tsdfTex.SetPixelData(tsdfBytes, 0);
            tsdfTex.Apply(false, false);
            Graphics.CopyTexture(tsdfTex, _volume);
            Destroy(tsdfTex);

            var colorTex = new Texture3D(s.x, s.y, s.z, GraphicsFormat.R8G8B8A8_UNorm, TextureCreationFlags.None);
            colorTex.SetPixelData(colorBytes, 0);
            colorTex.Apply(false, false);
            Graphics.CopyTexture(colorTex, _colorVolume);
            Destroy(colorTex);

            GL.Flush();

            IntegrationCount = integrationCount;
            _frustumReady = false;

            Logger.Info($"Volumes loaded: {s}, integrationCount={integrationCount}");
            return true;
        }
    }
}

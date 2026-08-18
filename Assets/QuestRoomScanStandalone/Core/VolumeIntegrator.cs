using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
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
        [SerializeField] private int3 voxelCount = new(192, 128, 192);
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
        [Tooltip("矛盾扣减率（投票制反对票）。新观测与存量矛盾（更空 >3cm）时每帧扣 q²·carveGain；weight 跌破 minMeshWeight 网格消失、跌破 PRUNE_WEIGHT(08-18 起 0.03) 被 Prune 回收。0 = 关闭，恢复 QRS 原版纯驻留行为。 (default 0.075)")]
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
        [Tooltip("自由空间对称扣减倍率：只乘主矛盾通道（正式反对票），新鲜正式面 2~3 票扣死（混合斜坡伪造品的多帧一致性反而加速其死亡），" +
                 "成熟面权重基数大且有持续正视赞成票补给、天然免疫。不动 carveGain 本体、不动绕行通道（它们用 carveBypassBoost）。1 = 关回旧行为。 (default 2)")]
        [SerializeField, Range(1f, 4f)] private float freeSpaceCarveBoost = 2f;
        [Tooltip("救援样本播种资格审查改用距离分：救援标记=上游边缘清洗已用同平面连贯性背书，掠射本地法线测不准，播种闸不再重复罚角度。" +
                 "治天棚远场掠射面质量分永远够不到 0.25 播种线、点阵集体卡黄。增长/扣减仍按真实质量分，防假桥三道门不动。关=恢复按真实质量审查。 (default true)")]
        [SerializeField] private bool rescueSeedDistOnly = true;
        [Tooltip("弃权邻域写闸：边缘清洗弃权像素周边 1px 内只拦写（播种/增长）不拦抹（矛盾扣减照常）。" +
                 "治帘锚点旁漏网裙边零星复种导致桥闪断闪连。真折角切口旁晚一两帧种上，代价可忽略。 (default true)")]
        [SerializeField] private bool abstainSeedGuard = true;
        [Tooltip("平面外推闸：投影 SDF 声称接近表面时，还要求未投影的沿视线距离仍在双向 TSDF 窄带内。只拦播种/增长，不拦矛盾扣减。 (default true)")]
        [SerializeField] private bool rawSeedGate = true;
        [Tooltip("双向原始沿视线带宽，单位为 voxelDistance。1.0 表示标准的一个 TSDF 截断带。 (default 1.0)")]
        [SerializeField, Range(0.5f, 4f)] private float rawSeedBand = 1.0f;

        [Header("深度断层与膨胀路径闸")]
        [Tooltip("复用 DepthEdgeClean 的距离自适应断层阈值下限。用于诊断，也用于判断膨胀供体到接收点之间是否跨越深度断层。")]
        [SerializeField, Min(0.001f)] private float diagnosticDepthGapBaseMeters = 0.035f;
        [Tooltip("断层阈值随深度增长系数，阈值=max(下限, 深度×系数)。")]
        [SerializeField, Min(0f)] private float diagnosticDepthGapDistanceScale = 0.018f;
        [Tooltip("把已有的膨胀路径分类接入正式写入：跨已杀边缘、无效带或深度跳变的供料只拦播种/增长，" +
                 "自由空间矛盾扣减仍然放行。 (default true)")]
        [SerializeField] private bool dilationProductionGate = true;
        [Tooltip("两跳及以上的 Jump Flood 接力供料不得播种或增长正式 TSDF；一跳且路径干净的局部供料仍允许。 (default true)")]
        [SerializeField] private bool dilationBlockRelayWrites = true;
        [Tooltip("供体距离过长、固定探针无法密集覆盖整条路径时，不允许其播种或增长正式 TSDF。 (default true)")]
        [SerializeField] private bool dilationBlockSparseWrites = true;

        [Header("Provisional TSDF 生命周期")]
        [Tooltip("新体素首次写入时的暂存权重，运行时会被限制在 prune 阈值之上、正式出网阈值之下。" +
                 "重复一致观测使其越过 minMeshWeight 后才进入网格。 (default 0.06)")]
        [SerializeField, Range(0.031f, 0.079f)] private float provisionalSeedWeight = 0.06f;

        [Header("运动闸")]
        [Tooltip("头显角速度超过此值（°/s）时整帧停笔：不写入、不增长、不矛盾扣减（预览/提取照常）。根治位姿-深度不同帧欠账下转头把表面抹出搓衣板褶皱、错位矛盾票啃真表面。消幽灵的正确姿势变为\"停下来正对看一秒\"。0 = 关闭。 (default 90)")]
        [SerializeField, Range(0f, 360f)] private float motionGateDegPerSec = 90f;
        [Tooltip("运动种子闸（°/s）：角速度超阈的帧只长不种——假面 77% 出生于 60~90°/s 放行带，本闸只断假面出生口，增长/矛盾扣减照常。" +
                 "与整帧闸分工：整帧闸 90 防涂抹，种子闸 60 保覆盖生产力（天棚慢扫帧多在 60 以下）。0 = 关闭。 (default 60)")]
        [SerializeField, Range(0f, 360f)] private float motionSeedBlockDegPerSec = 60f;

        [Header("逐块可逆冻结")]
        [Tooltip("逐块可逆冻结总开关：开=冻结体素记穿越票+成熟度普查可用；关=票/普查全停（掩码冻结/解冻 API 仍可用）。" +
                 "冻结=weight 翻符号不销毁 TSDF，提取层 abs 透明，解冻=翻回。 (default true)")]
        [SerializeField] private bool frozenBlockEnable = true;
        [Tooltip("穿越票最低质量分：冻结体素只接受质量分达标的观测投票（防低质掠射噪声买票解冻）。与播种线 0.25 对齐。 (default 0.25)")]
        [SerializeField, Range(0.05f, 0.6f)] private float frozenVoteQualityMin = 0.25f;
        [Tooltip("穿越票矛盾杆（归一化单位，1=截断距离）：shader 内按 margin×max(dist/1.5,1) 随距离放大。" +
                 "固定 0.2≈3cm 在 2m 外被深度噪声日常踩破——纯噪声把票箱刷热导致冻-解振荡（实机：解=冻的 2~3 倍）。" +
                 "家具级真矛盾 ≥30cm，杆随距离抬到 ~10cm 也不误伤修复。 (default 0.2)")]
        [SerializeField, Range(0.1f, 1f)] private float frozenVoteMargin = 0.2f;
        [Tooltip("冻结块边长（体素），独立于提取块 64。32=1.6m——64³(3.2m) 全场只有 18 块，天棚和墙会同块被冻（实机：身周整片锁死）；32³=144 块，天棚单独成块。 (default 32)")]
        [SerializeField, Min(8)] private int frozenChunkSize = 32;
        [Tooltip("长熟体素权重门槛：成熟度普查只数权重大于此值的体素。必须远高于种子出生 0.06——否则毛坯脚手架与长熟墙计数无差别，块在毛坯期就被冻死（实机：天棚冻在 0.06 卡黄、确认红定格）。速冻 0.15≈连续观测 3 帧（红线：不得逼近出网阈值 0.08，观感红偏多就退回 0.2）。 (default 0.15)")]
        [SerializeField, Range(0.08f, 0.5f)] private float frozenMatureWeight = 0.15f;

        [Header("Meshing")]
        [Tooltip("Min voxel confidence weight for Surface Nets to generate mesh. Higher = fewer phantom surfaces. (default 0.08)")]
        [SerializeField, Range(0.01f, 0.5f)] private float minMeshWeight = 0.08f;
        public float MinMeshWeight => minMeshWeight;

        [Header("Incremental Meshing")]
        [Tooltip("Persistent extraction chunk edge length in voxels. This only schedules remeshing; it does not change TSDF admission.")]
        [SerializeField, Min(8)] private int extractionChunkSize = 64;
        [Tooltip("Minimum normalized TSDF movement near the zero crossing before a stable chunk is remeshed.")]
        [SerializeField, Range(0.002f, 0.25f)] private float dirtyTsdfThreshold = 0.02f;
        [Tooltip("Only changes inside this normalized zero-crossing band can dirty an already observed surface.")]
        [SerializeField, Range(0.25f, 1f)] private float dirtySurfaceBand = 1f;

        [Header("Projective TSDF A/B")]
        [SerializeField, Tooltip("建立一份只读 KinectFusion 式 raw-projective TSDF 影子体。生产体仍使用现有 raw×法向余弦；影子体只用于对照统计和手动切换显示。")]
        private bool enableProjectiveShadow = false;

        [Header("Camera Color")]
        [Tooltip("Exposure boost for camera texture. Quest 3 passthrough cameras produce dim images. (default 3.0)")]
        [SerializeField, Range(1f, 10f)] private float cameraExposure = 3f;

        private RenderTexture _volume;
        private RenderTexture _colorVolume;
        private RenderTexture _projectiveShadowVolume;
        private RenderTexture _admissionTraceVolume;

        /// <summary>3D RenderTexture (R8G8_SNorm) storing the truncated signed distance field.</summary>
        public RenderTexture Volume => _volume;
        /// <summary>只读 A/B 影子体：使用 raw projective SDF，但不写颜色、不替换生产体。</summary>
        public RenderTexture ProjectiveShadowVolume => _projectiveShadowVolume;
        public bool ProjectiveShadowEnabled => enableProjectiveShadow && _projectiveShadowVolume != null;
        /// <summary>3D RenderTexture (RGBA8_UNorm) storing per-voxel accumulated color.</summary>
        public RenderTexture ColorVolume => _colorVolume;
        /// <summary>Read-only provenance sidecar for dilation admission. Never changes TSDF production decisions.</summary>
        public RenderTexture AdmissionTraceVolume => _admissionTraceVolume;
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
        private static readonly int FreeSpaceCarveBoostID = Shader.PropertyToID("gsFreeSpaceCarveBoost");
        private static readonly int RescueSeedDistOnlyID = Shader.PropertyToID("gsRescueSeedDistOnly");
        private static readonly int MotionSeedBlockID = Shader.PropertyToID("gsMotionSeedBlock");
        private static readonly int AbstainSeedGuardID = Shader.PropertyToID("gsAbstainSeedGuard");
        private static readonly int RawSeedGateID = Shader.PropertyToID("gsRawSeedGate");
        private static readonly int RawSeedBandID = Shader.PropertyToID("gsRawSeedBand");
        private static readonly int DiagDepthGapBaseID = Shader.PropertyToID("gsDiagDepthGapBase");
        private static readonly int DiagDepthGapScaleID = Shader.PropertyToID("gsDiagDepthGapScale");
        private static readonly int DilationProductionGateID = Shader.PropertyToID("gsDilationProductionGate");
        private static readonly int DilationBlockRelayID = Shader.PropertyToID("gsDilationBlockRelay");
        private static readonly int DilationBlockSparseID = Shader.PropertyToID("gsDilationBlockSparse");
        private static readonly int ProvisionalSeedWeightID = Shader.PropertyToID("gsProvisionalSeedWeight");
        private static readonly int FormalSurfaceWeightID = Shader.PropertyToID("gsFormalSurfaceWeight");
        private static readonly int DiagnosticAngularSpeedID = Shader.PropertyToID("gsDiagnosticAngularSpeed");
        private static readonly int CamRGBID = Shader.PropertyToID("gsCamRGB");
        private static readonly int CamAvailableID = Shader.PropertyToID("gsCamAvailable");
        private static readonly int CamPosID = Shader.PropertyToID("gsCamPos");
        private static readonly int CamInvRotID = Shader.PropertyToID("gsCamInvRot");
        private static readonly int CamFocalLenID = Shader.PropertyToID("gsCamFocalLen");
        private static readonly int CamPrincipalPtID = Shader.PropertyToID("gsCamPrincipalPt");
        private static readonly int CamSensorResID = Shader.PropertyToID("gsCamSensorRes");
        private static readonly int CamCurrentResID = Shader.PropertyToID("gsCamCurrentRes");
        private static readonly int CamExposureID = Shader.PropertyToID("gsCamExposure");
        private static readonly int UseRawProjectiveSdfID = Shader.PropertyToID("gsUseRawProjectiveSdf");
        private static readonly int WriteColorID = Shader.PropertyToID("gsWriteColor");
        private static readonly int AdmissionTraceRWID = Shader.PropertyToID("gsAdmissionTraceRW");
        private static readonly int WriteAdmissionTraceID = Shader.PropertyToID("gsWriteAdmissionTrace");
        private static readonly int BakeSrcAdmissionTraceID = Shader.PropertyToID("gsBakeSrcAdmissionTrace");
        private static readonly int PruneZOffsetID = Shader.PropertyToID("gsPruneZOffset");
        private static readonly int PruneZCountID = Shader.PropertyToID("gsPruneZCount");
        private static readonly int DirtyChunkEpochsID = Shader.PropertyToID("_DirtyChunkEpochs");
        private static readonly int DirtyBoundaryEpochsID = Shader.PropertyToID("_DirtyBoundaryEpochs");
        private static readonly int DirtyChunkCountID = Shader.PropertyToID("gsDirtyChunkCount");
        private static readonly int DirtyChunkSizeID = Shader.PropertyToID("gsDirtyChunkSize");
        private static readonly int TrackDirtyChunksID = Shader.PropertyToID("gsTrackDirtyChunks");
        private static readonly int DirtyChunkEpochID = Shader.PropertyToID("gsDirtyChunkEpoch");
        private static readonly int DirtyBoundaryHaloID = Shader.PropertyToID("gsDirtyBoundaryHalo");
        private static readonly int DirtyTsdfThresholdID = Shader.PropertyToID("gsDirtyTsdfThreshold");
        private static readonly int DirtySurfaceBandID = Shader.PropertyToID("gsDirtySurfaceBand");
        private static readonly int DirtyMinWeightID = Shader.PropertyToID("gsDirtyMinWeight");
        private static readonly int ChunkFreezeSetMaskID = Shader.PropertyToID("_ChunkFreezeSetMask");
        private static readonly int ChunkFreezeClearMaskID = Shader.PropertyToID("_ChunkFreezeClearMask");
        private static readonly int FrozenChunkVotesID = Shader.PropertyToID("_FrozenChunkVotes");
        private static readonly int ChunkMaturityID = Shader.PropertyToID("_ChunkMaturity");
        private static readonly int FrozenBlockEnableID = Shader.PropertyToID("gsFrozenBlockEnable");
        private static readonly int FrozenVoteQualityMinID = Shader.PropertyToID("gsFrozenVoteQualityMin");
        private static readonly int FrozenVoteMarginID = Shader.PropertyToID("gsFrozenVoteMargin");
        private static readonly int FrozenChunkCountID = Shader.PropertyToID("gsFrozenChunkCount");
        private static readonly int FrozenChunkSizeID = Shader.PropertyToID("gsFrozenChunkSize");
        private static readonly int FrozenMatureWeightID = Shader.PropertyToID("gsFrozenMatureWeight");

        public float CameraExposure => cameraExposure;

        [Header("Warmup")]
        [Tooltip("Clear the volume after this many integrations to discard sensor startup noise. 0 = disabled.")]
        [SerializeField] private int warmupIntegrations = 3;

        [Header("Pruning")]
        [SerializeField] private float pruneIntervalSeconds = 3f;
        [SerializeField, Min(1), Tooltip("Number of Z slices pruned after each integration while a prune cycle is active.")]
        private int pruneSlicesPerIntegration = 8;

        private ComputeKernelHelper _clearKernel;
        private ComputeKernelHelper _integrateKernel;
        private ComputeKernelHelper _pruneKernel;
        private ComputeKernelHelper _freezeKernel;
        private ComputeKernelHelper _unfreezeKernel;
        private ComputeKernelHelper _applyFreezeMaskKernel;
        private ComputeKernelHelper _clearVotesKernel;
        private ComputeKernelHelper _maturityKernel;

        private ComputeBuffer _frustumVolume;
        private ComputeBuffer _dirtyChunkEpochs;
        private ComputeBuffer _dirtyBoundaryEpochs;
        private ComputeBuffer _chunkFreezeSetMask;
        private ComputeBuffer _chunkFreezeClearMask;
        private ComputeBuffer _frozenChunkVotes;
        private ComputeBuffer _chunkMaturity;
        private uint[] _voteZeros;
        private uint[] _maturityZeros;
        private int3 _frozenChunkCount;
        private int3 _dirtyChunkCount;
        private int _dirtyBoundaryHaloVoxels = 2;
        private uint _dirtyEpoch = 1;
        private bool _frustumReady;
        private float _lastPruneTime;
        private bool _pruneCycleActive;
        private int _nextPruneSlice;

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
        private ComputeBuffer _projectiveShadowCarveStats;
        private bool _projectiveShadowCarveStatsReadbackPending;
        // 0..27: existing contradiction/supply/edge ledgers.
        // 28..44: read-only dilation-donor provenance ledger.
        // 45..49: read-only donor-path classification ledger.
        // 50..59: direct-fill versus relay provenance ledger.
        // 60..65: production dilation-gate reason counters.
        // 66..67: actual seed/growth writes blocked by that gate.
        // 68..70: provisional seeds, promotions and formal-surface demotions.
        // 71..89: read-only lifecycle forensics: seed source/risk, promotion
        // mechanism, promotion-time risk and immutable birth source.
        private const int CarveStatsCount = 91;
        private static readonly uint[] ZeroCarveStats = new uint[CarveStatsCount];
        /// <summary>最近一个统计周期的矛盾票计数：0票投出 1排除区拦 2法线闸拦 3遮挡闸拦 4带外拦 5排内抹（不对称放行实际扣减）。</summary>
        public readonly uint[] LastCarveStats = new uint[CarveStatsCount];
        public readonly uint[] LastProjectiveShadowCarveStats = new uint[CarveStatsCount];
        public readonly ulong[] CumulativeCarveStats = new ulong[CarveStatsCount];
        /// <summary>是否已有至少一轮矛盾票读回。</summary>
        public bool HasCarveStats { get; private set; }
        public bool HasProjectiveShadowCarveStats { get; private set; }
        private static readonly int CarveStatsID = Shader.PropertyToID("_CarveStats");

        [Header("Coverage Metrics")]
        [Tooltip("Dispatch coverage count every N integrations (0 = disabled). Higher = less GPU overhead.")]
        [SerializeField] private int coverageUpdateInterval = 120;

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
        /// <summary>Raised when relocation/load invalidates every persistent mesh chunk.</summary>
        public event Action TopologyInvalidated;

        public ComputeBuffer DirtyChunkEpochs => _dirtyChunkEpochs;

        /// <summary>最新脏块 epoch 快照（CPU 侧，实时轨内容闸用；异步回读，帧级新鲜）。</summary>
        public uint[] LatestDirtyChunkEpochs { get; private set; }
        private bool _dirtyEpochReadbackPending;

        /// <summary>
        /// 发起脏块 epoch 回读（3×2×3=18 个 uint，72B，实时轨每巡视一次；在途时自动合并）。
        /// 内容闸信号源（T1a）：脏账由 MarkDirtyChunk 在几何级变化（新生/穿越/位移）时
        /// InterlockedMax 推进，权重纯积累不记账——比 2s 普查快照新鲜两个数量级。
        /// </summary>
        public void RequestDirtyChunkEpochs()
        {
            if (_dirtyEpochReadbackPending || _dirtyChunkEpochs == null) return;
            _dirtyEpochReadbackPending = true;
            AsyncGPUReadback.Request(_dirtyChunkEpochs, req =>
            {
                _dirtyEpochReadbackPending = false;
                if (req.hasError) return;
                var data = req.GetData<uint>();
                if (LatestDirtyChunkEpochs == null || LatestDirtyChunkEpochs.Length != data.Length)
                    LatestDirtyChunkEpochs = new uint[data.Length];
                data.CopyTo(LatestDirtyChunkEpochs);
            });
        }
        public ComputeBuffer DirtyBoundaryEpochs => _dirtyBoundaryEpochs;
        public int3 DirtyChunkCount => _dirtyChunkCount;
        public int ExtractionChunkSize => Mathf.Max(8, extractionChunkSize);
        public uint DirtyEpoch => _dirtyEpoch;

        private Texture _pendingCamFrame;
        private Vector3 _pendingCamPos;
        private Quaternion _pendingCamRot;
        private Vector2 _pendingFocalLen;
        private Vector2 _pendingPrincipalPt;
        private Vector2 _pendingSensorRes;
        private Vector2 _pendingCurrentRes;
        private RenderTexture _camFrameCopy;
        private Texture2D _dummyCamTex;

        // 运动闸：角速度镜像自 DepthCapture.SmoothedDepthAngularSpeed（深度帧事件内、
        // 原始 Pose 四元数、真实帧间隔计算），供 HUD 读数与运动闸共用。
        private float _smoothedAngSpeed;
        private int _motionGatedSinceStats;
        private int _lastMotionGatedCount;
        /// <summary>平滑后的深度位姿角速度（°/s），调试用。</summary>
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
            _clearKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);

            _integrateKernel = new ComputeKernelHelper(compute, "Integrate");
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _integrateKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
            _integrateKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);

            _pruneKernel = new ComputeKernelHelper(compute, "Prune");
            _pruneKernel.Set(VolumeRWID, _volume);
            _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
            _pruneKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _pruneKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
            _pruneKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);

            _freezeKernel = new ComputeKernelHelper(compute, "FreezeInFrustum");
            _freezeKernel.Set(VolumeRWID, _volume);

            _unfreezeKernel = new ComputeKernelHelper(compute, "UnfreezeInFrustum");
            _unfreezeKernel.Set(VolumeRWID, _volume);

            _applyFreezeMaskKernel = new ComputeKernelHelper(compute, "ApplyChunkFreezeMask");
            _applyFreezeMaskKernel.Set(VolumeRWID, _volume);
            _applyFreezeMaskKernel.Set(ChunkFreezeSetMaskID, _chunkFreezeSetMask);
            _applyFreezeMaskKernel.Set(ChunkFreezeClearMaskID, _chunkFreezeClearMask);

            _clearVotesKernel = new ComputeKernelHelper(compute, "ClearFrozenChunkVotes");
            _clearVotesKernel.Set(ChunkFreezeClearMaskID, _chunkFreezeClearMask);
            _clearVotesKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);

            _maturityKernel = new ComputeKernelHelper(compute, "CountChunkMaturity");
            _maturityKernel.Set(VolumeRWID, _volume);
            _maturityKernel.Set(ChunkMaturityID, _chunkMaturity);

            _integrateKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);

            _coverageKernel = new ComputeKernelHelper(compute, "CountSurfaceCoverage");
            _coverageKernel.Set(VolumeRWID, _volume);
            _coverageCounters = new ComputeBuffer(3, sizeof(uint));
            _coverageKernel.Set(CoverageCountersID, _coverageCounters);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);

            _carveStats = new ComputeBuffer(CarveStatsCount, sizeof(uint));
            _carveStats.SetData(ZeroCarveStats);
            _integrateKernel.Set(CarveStatsID, _carveStats);

            if (enableProjectiveShadow)
            {
                _projectiveShadowCarveStats = new ComputeBuffer(CarveStatsCount, sizeof(uint));
                _projectiveShadowCarveStats.SetData(ZeroCarveStats);
            }

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
            _projectiveShadowCarveStats?.Release();
            _projectiveShadowCarveStats = null;
            ReleaseFrozenBlockBuffers();
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
            _dirtyChunkEpochs?.Release();
            _dirtyChunkEpochs = null;
            _dirtyBoundaryEpochs?.Release();
            _dirtyBoundaryEpochs = null;
            _dirtyChunkCount = int3.zero;
            if (_volume) { Destroy(_volume); _volume = null; }
            if (_colorVolume) { Destroy(_colorVolume); _colorVolume = null; }
            if (_projectiveShadowVolume) { Destroy(_projectiveShadowVolume); _projectiveShadowVolume = null; }
            if (_admissionTraceVolume) { Destroy(_admissionTraceVolume); _admissionTraceVolume = null; }
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
            EnsureDirtyChunkBuffer();

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
            EnsureDirtyChunkBuffer();
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _integrateKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
            _integrateKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);
            _pruneKernel.Set(VolumeRWID, _volume);
            _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
            _pruneKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _pruneKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
            _pruneKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);
            _freezeKernel.Set(VolumeRWID, _volume);
            _unfreezeKernel.Set(VolumeRWID, _volume);
            _applyFreezeMaskKernel.Set(VolumeRWID, _volume);
            _applyFreezeMaskKernel.Set(ChunkFreezeSetMaskID, _chunkFreezeSetMask);
            _applyFreezeMaskKernel.Set(ChunkFreezeClearMaskID, _chunkFreezeClearMask);
            _clearVotesKernel.Set(ChunkFreezeClearMaskID, _chunkFreezeClearMask);
            _clearVotesKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);
            _maturityKernel.Set(VolumeRWID, _volume);
            _maturityKernel.Set(ChunkMaturityID, _chunkMaturity);
            _integrateKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);
            _coverageKernel.Set(VolumeRWID, _volume);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);
        }

        private void EnsureDirtyChunkBuffer()
        {
            int chunkSize = ExtractionChunkSize;
            int3 required = new int3(
                Mathf.CeilToInt(voxelCount.x / (float)chunkSize),
                Mathf.CeilToInt(voxelCount.y / (float)chunkSize),
                Mathf.CeilToInt(voxelCount.z / (float)chunkSize));
            int requiredCount = required.x * required.y * required.z;
            int boundaryCount = Mathf.Max(1, requiredCount * 6);
            int fChunkSize = Mathf.Max(8, frozenChunkSize);
            int3 frozenRequired = new int3(
                Mathf.CeilToInt(voxelCount.x / (float)fChunkSize),
                Mathf.CeilToInt(voxelCount.y / (float)fChunkSize),
                Mathf.CeilToInt(voxelCount.z / (float)fChunkSize));
            int frozenRequiredCount = frozenRequired.x * frozenRequired.y * frozenRequired.z;
            if (_dirtyChunkEpochs != null && _dirtyChunkEpochs.count == requiredCount &&
                _dirtyBoundaryEpochs != null && _dirtyBoundaryEpochs.count == boundaryCount &&
                _frozenChunkVotes != null && _frozenChunkVotes.count == Mathf.Max(1, frozenRequiredCount))
            {
                _dirtyChunkCount = required;
                _frozenChunkCount = frozenRequired;
                return;
            }

            _dirtyChunkEpochs?.Release();
            _dirtyBoundaryEpochs?.Release();
            _dirtyChunkEpochs = new ComputeBuffer(Mathf.Max(1, requiredCount), sizeof(uint));
            _dirtyBoundaryEpochs = new ComputeBuffer(boundaryCount, sizeof(uint));
            _dirtyChunkCount = required;
            _dirtyChunkEpochs.SetData(new uint[Mathf.Max(1, requiredCount)]);
            _dirtyBoundaryEpochs.SetData(new uint[boundaryCount]);

            ReleaseFrozenBlockBuffers();
            _frozenChunkCount = frozenRequired;
            int frozenCount = Mathf.Max(1, frozenRequiredCount);
            int maskWords = Mathf.Max(1, (frozenRequiredCount + 31) / 32);
            _chunkFreezeSetMask = new ComputeBuffer(maskWords, sizeof(uint));
            _chunkFreezeClearMask = new ComputeBuffer(maskWords, sizeof(uint));
            _frozenChunkVotes = new ComputeBuffer(frozenCount, sizeof(uint) * 2);
            _chunkMaturity = new ComputeBuffer(frozenCount, sizeof(uint) * 4);
            _voteZeros = new uint[frozenCount * 2];
            _maturityZeros = new uint[frozenCount * 4];
            _chunkFreezeSetMask.SetData(new uint[maskWords]);
            _chunkFreezeClearMask.SetData(new uint[maskWords]);
            _frozenChunkVotes.SetData(_voteZeros);
            _chunkMaturity.SetData(_maturityZeros);
        }

        private void ReleaseFrozenBlockBuffers()
        {
            _chunkFreezeSetMask?.Release();
            _chunkFreezeSetMask = null;
            _chunkFreezeClearMask?.Release();
            _chunkFreezeClearMask = null;
            _frozenChunkVotes?.Release();
            _frozenChunkVotes = null;
            _chunkMaturity?.Release();
            _chunkMaturity = null;
            _voteZeros = null;
            _maturityZeros = null;
        }

        public void SetDirtyBoundaryHalo(int haloVoxels)
        {
            _dirtyBoundaryHaloVoxels = Mathf.Clamp(haloVoxels, 1, Mathf.Max(1, ExtractionChunkSize / 2));
            if (compute != null)
                compute.SetInt(DirtyBoundaryHaloID, _dirtyBoundaryHaloVoxels);
        }

        private void ConfigureDirtyTracking(bool enabled)
        {
            compute.SetInts(DirtyChunkCountID, _dirtyChunkCount.x, _dirtyChunkCount.y, _dirtyChunkCount.z);
            compute.SetInt(DirtyChunkSizeID, ExtractionChunkSize);
            compute.SetInt(TrackDirtyChunksID, enabled && _dirtyChunkEpochs != null ? 1 : 0);
            compute.SetInt(DirtyChunkEpochID, unchecked((int)_dirtyEpoch));
            compute.SetInt(DirtyBoundaryHaloID, _dirtyBoundaryHaloVoxels);
            compute.SetFloat(DirtyTsdfThresholdID, dirtyTsdfThreshold);
            compute.SetFloat(DirtySurfaceBandID, dirtySurfaceBand);
            compute.SetFloat(DirtyMinWeightID, minMeshWeight);
            compute.SetInts(FrozenChunkCountID, _frozenChunkCount.x, _frozenChunkCount.y, _frozenChunkCount.z);
            compute.SetInt(FrozenChunkSizeID, Mathf.Max(8, frozenChunkSize));
            compute.SetFloat(FrozenMatureWeightID, frozenMatureWeight);
        }

        private void BeginDirtyEpoch()
        {
            _dirtyEpoch++;
            if (_dirtyEpoch == 0)
            {
                _dirtyEpoch = 1;
                if (_dirtyChunkEpochs != null)
                    _dirtyChunkEpochs.SetData(new uint[_dirtyChunkEpochs.count]);
                if (_dirtyBoundaryEpochs != null)
                    _dirtyBoundaryEpochs.SetData(new uint[_dirtyBoundaryEpochs.count]);
            }
            ConfigureDirtyTracking(true);
        }

        public void MarkAllChunksDirty()
        {
            EnsureDirtyChunkBuffer();
            _dirtyEpoch++;
            if (_dirtyEpoch == 0) _dirtyEpoch = 1;
            var all = new uint[_dirtyChunkEpochs.count];
            for (int i = 0; i < all.Length; i++) all[i] = _dirtyEpoch;
            _dirtyChunkEpochs.SetData(all);
            // A global invalidation already covers every owner.  Old face-halo
            // epochs must not survive a clear/load and manufacture neighbour
            // work in the following scan session.
            if (_dirtyBoundaryEpochs != null)
                _dirtyBoundaryEpochs.SetData(new uint[_dirtyBoundaryEpochs.count]);
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
            if (data.Length < CarveStatsCount) return;
            for (int i = 0; i < CarveStatsCount; i++)
            {
                LastCarveStats[i] = data[i];
                CumulativeCarveStats[i] += data[i];
            }
            HasCarveStats = true;
            _carveStats.SetData(ZeroCarveStats); // 数据已落袋，清零开新周期
            _lastMotionGatedCount = _motionGatedSinceStats; // 运动闸同节奏结算
            _motionGatedSinceStats = 0;
        }

        private void RequestProjectiveShadowCarveStatsReadback()
        {
            if (_projectiveShadowCarveStats == null || _projectiveShadowCarveStatsReadbackPending) return;
            _projectiveShadowCarveStatsReadbackPending = true;
            AsyncGPUReadback.Request(_projectiveShadowCarveStats, OnProjectiveShadowCarveStatsReadback);
        }

        private void OnProjectiveShadowCarveStatsReadback(AsyncGPUReadbackRequest request)
        {
            _projectiveShadowCarveStatsReadbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < CarveStatsCount) return;
            for (int i = 0; i < CarveStatsCount; i++) LastProjectiveShadowCarveStats[i] = data[i];
            HasProjectiveShadowCarveStats = true;
            _projectiveShadowCarveStats.SetData(ZeroCarveStats);
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
                   $"遮拦{FormatCarveCount(LastCarveStats[3])} 带拦{FormatCarveCount(LastCarveStats[4])} " +
                   $"前延{FormatCarveCount(LastCarveStats[8])} 后延{FormatCarveCount(LastCarveStats[9])}" +
                   (_lastMotionGatedCount > 0 ? $" 动闸{_lastMotionGatedCount}" : "");
        }

        /// <summary>只读近零供料分账；不参与融合、裁剪或提取。</summary>
        public string GetSupplyLedgerCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"内{FormatCarveCount(LastCarveStats[10])} 外{FormatCarveCount(LastCarveStats[11])} " +
                   $"邻差≤5:{FormatCarveCount(LastCarveStats[12])} " +
                   $"5~15:{FormatCarveCount(LastCarveStats[13])} " +
                   $"15~30:{FormatCarveCount(LastCarveStats[14])} " +
                   $">30:{FormatCarveCount(LastCarveStats[15])}";
        }

        /// <summary>只读自适应深度断层闸账；过=影子放行，掠/斜/正=按入射角拆分的影子拦截。</summary>
        public string GetAdaptiveGapShadowCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"过{FormatCarveCount(LastCarveStats[16])} " +
                   $"拦掠{FormatCarveCount(LastCarveStats[17])} " +
                   $"拦斜{FormatCarveCount(LastCarveStats[18])} " +
                   $"拦正{FormatCarveCount(LastCarveStats[19])}";
        }

        /// <summary>只读：把被融合端接受的近零供料追溯到边缘清洗器的具体判定来源。</summary>
        public string GetEdgeSourceLedgerCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"净{FormatCarveCount(LastCarveStats[20])} " +
                   $"平保{FormatCarveCount(LastCarveStats[21])} " +
                   $"裙{FormatCarveCount(LastCarveStats[22])} " +
                   $"双簇{FormatCarveCount(LastCarveStats[23])} " +
                   $"掠{FormatCarveCount(LastCarveStats[24])} " +
                   $"跨眼{FormatCarveCount(LastCarveStats[25])} " +
                   $"杀漏{FormatCarveCount(LastCarveStats[26])} " +
                   $"余疑{FormatCarveCount(LastCarveStats[27])}";
        }

        /// <summary>
        /// 只读：追溯膨胀深度实际从哪个像素传播而来。第一行按传播像素距离分桶；
        /// 第二行仅统计发生传播的供料，并按深度差与供体边缘原因拆分。
        /// </summary>
        public string GetDilationDonorLedgerCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"距自{FormatCarveCount(LastCarveStats[28])} ≤2:{FormatCarveCount(LastCarveStats[29])} " +
                   $"2~4:{FormatCarveCount(LastCarveStats[30])} 4~8:{FormatCarveCount(LastCarveStats[31])} " +
                   $">8:{FormatCarveCount(LastCarveStats[32])}\n" +
                   $"差≤5:{FormatCarveCount(LastCarveStats[33])} 5~15:{FormatCarveCount(LastCarveStats[34])} " +
                   $"15~30:{FormatCarveCount(LastCarveStats[35])} >30:{FormatCarveCount(LastCarveStats[36])} " +
                   $"源净{FormatCarveCount(LastCarveStats[37])} 平{FormatCarveCount(LastCarveStats[38])} " +
                   $"疑{FormatCarveCount(LastCarveStats[39])} 裙{FormatCarveCount(LastCarveStats[40])} " +
                   $"掠{FormatCarveCount(LastCarveStats[41])} 双{FormatCarveCount(LastCarveStats[42])} " +
                   $"跨{FormatCarveCount(LastCarveStats[43])} 杀{FormatCarveCount(LastCarveStats[44])}";
        }

        /// <summary>Dilation path ledger: clean, edge, invalid, jump, sparse.</summary>
        public string GetDilationPathLedgerCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"净{FormatCarveCount(LastCarveStats[45])} " +
                   $"边{FormatCarveCount(LastCarveStats[46])} " +
                   $"空{FormatCarveCount(LastCarveStats[47])} " +
                   $"跳{FormatCarveCount(LastCarveStats[48])} " +
                   $"稀{FormatCarveCount(LastCarveStats[49])}";
        }

        /// <summary>
        /// Jump-flood provenance ledger. 直补 is a one-hop raw-depth
        /// hole-fill candidate; 接力 reused a dilated result for two or more hops.
        /// Each side is split into clean / barrier / sparse path evidence.
        /// </summary>
        public string GetDilationRelayLedgerCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"直补{FormatCarveCount(LastCarveStats[50])}" +
                   $"(净{FormatCarveCount(LastCarveStats[51])}/险{FormatCarveCount(LastCarveStats[52])}/稀{FormatCarveCount(LastCarveStats[53])}) " +
                   $"接力{FormatCarveCount(LastCarveStats[54])}" +
                   $"(净{FormatCarveCount(LastCarveStats[55])}/险{FormatCarveCount(LastCarveStats[56])}/稀{FormatCarveCount(LastCarveStats[57])}) " +
                   $"二跳{FormatCarveCount(LastCarveStats[58])}/多跳{FormatCarveCount(LastCarveStats[59])}";
        }

        /// <summary>正式膨胀路径闸：命中原因与最终实际拦下的播种/增长次数。</summary>
        public string GetDilationProductionGateCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"命中{FormatCarveCount(LastCarveStats[60])} " +
                   $"边{FormatCarveCount(LastCarveStats[61])}/空{FormatCarveCount(LastCarveStats[62])}/" +
                   $"跳{FormatCarveCount(LastCarveStats[63])}/接{FormatCarveCount(LastCarveStats[64])}/" +
                   $"稀{FormatCarveCount(LastCarveStats[65])} " +
                   $"拦种{FormatCarveCount(LastCarveStats[66])}/拦长{FormatCarveCount(LastCarveStats[67])}";
        }

        /// <summary>暂存 TSDF 的出生、晋升为正式表面及被反证降回阈值以下的次数。</summary>
        public string GetProvisionalLifecycleCompact()
        {
            if (!HasCarveStats) return "统计中";
            return $"暂生{FormatCarveCount(LastCarveStats[68])} " +
                   $"转正{FormatCarveCount(LastCarveStats[69])} " +
                   $"降级{FormatCarveCount(LastCarveStats[70])}";
        }

        /// <summary>
        /// Request the final partial integration period before a frozen replay.
        /// The callback folds the GPU period into the cumulative CPU ledger;
        /// this method never stalls the render thread or changes fusion rules.
        /// </summary>
        public void FlushForensicLedger()
        {
            RequestCarveStatsReadback();
        }

        /// <summary>Append the cumulative, read-only integration causal ledger.</summary>
        public void AppendForensicLedgerReport(StringBuilder sb)
        {
            if (sb == null) return;
            static string U(ulong value) => value.ToString(CultureInfo.InvariantCulture);

            ulong seedSourceSum = CumulativeCarveStats[71] +
                                  CumulativeCarveStats[72] +
                                  CumulativeCarveStats[73];
            ulong promotionMechanismSum = CumulativeCarveStats[77] +
                                          CumulativeCarveStats[78];
            ulong promotionBirthSum = CumulativeCarveStats[82] +
                                      CumulativeCarveStats[83] +
                                      CumulativeCarveStats[84] +
                                      CumulativeCarveStats[89];

            sb.AppendLine();
            sb.AppendLine("integration_forensic_ledger:");
            sb.AppendLine("scope=cumulative_production_tsdf_since_last_clear;diagnostic_only=true");
            sb.AppendLine($"flush_pending={_carveStatsReadbackPending.ToString().ToLowerInvariant()}");
            sb.AppendLine($"seed_total={U(CumulativeCarveStats[68])}");
            sb.AppendLine($"seed_source_self={U(CumulativeCarveStats[71])}");
            sb.AppendLine($"seed_source_direct_fill={U(CumulativeCarveStats[72])}");
            sb.AppendLine($"seed_source_relay_fill={U(CumulativeCarveStats[73])}");
            sb.AppendLine($"seed_plane_rescue_overlap={U(CumulativeCarveStats[74])}");
            sb.AppendLine($"seed_near_abstain_overlap={U(CumulativeCarveStats[75])}");
            sb.AppendLine($"seed_motion_gt_60_overlap={U(CumulativeCarveStats[76])}");
            sb.AppendLine($"seed_motion_gate_block={U(CumulativeCarveStats[90])}");
            sb.AppendLine($"seed_source_reconcile_delta={(long)CumulativeCarveStats[68] - (long)seedSourceSum}");
            sb.AppendLine($"promotion_total={U(CumulativeCarveStats[69])}");
            sb.AppendLine($"promotion_natural_growth={U(CumulativeCarveStats[77])}");
            sb.AppendLine($"promotion_fast_second_raw={U(CumulativeCarveStats[78])}");
            sb.AppendLine($"promotion_current_plane_rescue_overlap={U(CumulativeCarveStats[79])}");
            sb.AppendLine($"promotion_current_near_abstain_overlap={U(CumulativeCarveStats[80])}");
            sb.AppendLine($"promotion_current_motion_gt_60_overlap={U(CumulativeCarveStats[81])}");
            sb.AppendLine($"promotion_birth_self={U(CumulativeCarveStats[82])}");
            sb.AppendLine($"promotion_birth_direct_fill={U(CumulativeCarveStats[83])}");
            sb.AppendLine($"promotion_birth_relay_fill={U(CumulativeCarveStats[84])}");
            sb.AppendLine($"promotion_birth_unknown={U(CumulativeCarveStats[89])}");
            sb.AppendLine($"promotion_combo_plane_near_abstain={U(CumulativeCarveStats[85])}");
            sb.AppendLine($"promotion_combo_plane_motion_gt_60={U(CumulativeCarveStats[86])}");
            sb.AppendLine($"promotion_combo_near_abstain_motion_gt_60={U(CumulativeCarveStats[87])}");
            sb.AppendLine($"promotion_fast_with_any_current_risk={U(CumulativeCarveStats[88])}");
            sb.AppendLine($"promotion_mechanism_reconcile_delta={(long)CumulativeCarveStats[69] - (long)promotionMechanismSum}");
            sb.AppendLine($"promotion_birth_reconcile_delta={(long)CumulativeCarveStats[69] - (long)promotionBirthSum}");
            sb.AppendLine($"formal_demotion_total={U(CumulativeCarveStats[70])}");
        }

        /// <summary>KinectFusion raw-projective 影子体的独立矛盾票摘要。</summary>
        public string GetProjectiveShadowStatsCompact()
        {
            if (!ProjectiveShadowEnabled) return "关闭";
            if (!HasProjectiveShadowCarveStats) return "统计中";
            return $"投{FormatCarveCount(LastProjectiveShadowCarveStats[0])} " +
                   $"排抹{FormatCarveCount(LastProjectiveShadowCarveStats[5])} " +
                   $"胀绕{FormatCarveCount(LastProjectiveShadowCarveStats[6])} " +
                   $"法绕{FormatCarveCount(LastProjectiveShadowCarveStats[7])} " +
                   $"排拦{FormatCarveCount(LastProjectiveShadowCarveStats[1])} " +
                   $"法拦{FormatCarveCount(LastProjectiveShadowCarveStats[2])} " +
                   $"遮拦{FormatCarveCount(LastProjectiveShadowCarveStats[3])} " +
                   $"带拦{FormatCarveCount(LastProjectiveShadowCarveStats[4])} " +
                   $"前延{FormatCarveCount(LastProjectiveShadowCarveStats[8])} " +
                   $"后延{FormatCarveCount(LastProjectiveShadowCarveStats[9])}";
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

            if (enableProjectiveShadow)
            {
                _projectiveShadowVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8G8_SNorm, 0)
                {
                    dimension = TextureDimension.Tex3D,
                    volumeDepth = voxelCount.z,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "ProjectiveTSDFShadow"
                };
                _projectiveShadowVolume.Create();
                Logger.Info($"Projective TSDF A/B shadow: {voxelCount} RG8_SNorm = {tsdfBytes / (1024 * 1024)}MB");
            }

            _colorVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8G8B8A8_UNorm, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _colorVolume.Create();

            GraphicsFormat traceFormat = SystemInfo.IsFormatSupported(GraphicsFormat.R8_UNorm, FormatUsage.LoadStore)
                ? GraphicsFormat.R8_UNorm
                : GraphicsFormat.R16_SFloat;
            _admissionTraceVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, traceFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "DilationAdmissionTrace"
            };
            _admissionTraceVolume.Create();
            long traceBytesPerVoxel = traceFormat == GraphicsFormat.R8_UNorm ? 1L : 2L;
            Logger.Info($"Dilation admission trace: {voxelCount} {traceFormat} = " +
                        $"{(traceBytesPerVoxel * voxelCount.x * voxelCount.y * voxelCount.z) / (1024 * 1024)}MB");
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
            compute.SetFloat(FreeSpaceCarveBoostID, freeSpaceCarveBoost);
            compute.SetFloat(RescueSeedDistOnlyID, rescueSeedDistOnly ? 1f : 0f);
            compute.SetFloat(MotionSeedBlockID, motionSeedBlockDegPerSec);
            compute.SetFloat(AbstainSeedGuardID, abstainSeedGuard ? 1f : 0f);
            compute.SetFloat(RawSeedGateID, rawSeedGate ? 1f : 0f);
            compute.SetFloat(RawSeedBandID, rawSeedBand);
            compute.SetFloat(DiagDepthGapBaseID, diagnosticDepthGapBaseMeters);
            compute.SetFloat(DiagDepthGapScaleID, diagnosticDepthGapDistanceScale);
            compute.SetFloat(DilationProductionGateID, dilationProductionGate ? 1f : 0f);
            compute.SetFloat(DilationBlockRelayID, dilationBlockRelayWrites ? 1f : 0f);
            compute.SetFloat(DilationBlockSparseID, dilationBlockSparseWrites ? 1f : 0f);
            compute.SetFloat(ProvisionalSeedWeightID, provisionalSeedWeight);
            compute.SetFloat(FormalSurfaceWeightID, minMeshWeight);
            compute.SetFloat(DiagnosticAngularSpeedID, 0f);
            compute.SetFloat(FrozenBlockEnableID, frozenBlockEnable ? 1f : 0f);
            compute.SetFloat(FrozenVoteQualityMinID, frozenVoteQualityMin);
            compute.SetFloat(FrozenVoteMarginID, frozenVoteMargin);
            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            ConfigureDirtyTracking(false);

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

            _pruneCycleActive = false;
            _nextPruneSlice = 0;

            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _clearKernel.DispatchFit(_volume);

            if (_projectiveShadowVolume != null)
            {
                compute.SetFloat(UseRawProjectiveSdfID, 1f);
                compute.SetFloat(WriteColorID, 0f);
                compute.SetFloat(WriteAdmissionTraceID, 0f);
                _clearKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _clearKernel.Set(ColorVolumeRWID, _colorVolume); // bound but guarded from writes
                _clearKernel.DispatchFit(_projectiveShadowVolume);
            }

            // Restore production defaults for every unrelated kernel/caller.
            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            _clearKernel.Set(VolumeRWID, _volume);
            _carveStats?.SetData(ZeroCarveStats);
            _projectiveShadowCarveStats?.SetData(ZeroCarveStats);
            // 清卷同时清冻结账：旧票箱/成熟度指向已销毁内容，必须同步归零。
            if (_frozenChunkVotes != null) _frozenChunkVotes.SetData(_voteZeros);
            if (_chunkMaturity != null) _chunkMaturity.SetData(_maturityZeros);
            Array.Clear(CumulativeCarveStats, 0, CumulativeCarveStats.Length);
            HasCarveStats = false;
            HasProjectiveShadowCarveStats = false;
            MarkAllChunksDirty();
            Cleared?.Invoke();
        }

        /// <summary>
        /// Reset counters that belong to a user-visible scan session. Kept out
        /// of <see cref="Clear"/> because the warm-up path also clears textures
        /// and must not restart its own IntegrationCount threshold forever.
        /// </summary>
        public void ResetSessionCounters()
        {
            IntegrationCount = 0;
            _integrationsSinceCoverage = 0;
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

            var dstAdmissionTrace = new RenderTexture(vc.x, vc.y, 0, _admissionTraceVolume.graphicsFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "DilationAdmissionTrace"
            };
            dstAdmissionTrace.Create();

            RenderTexture dstProjectiveShadow = null;
            if (_projectiveShadowVolume != null)
            {
                dstProjectiveShadow = new RenderTexture(vc.x, vc.y, 0, _projectiveShadowVolume.graphicsFormat, 0)
                {
                    dimension = TextureDimension.Tex3D,
                    volumeDepth = vc.z,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "ProjectiveTSDFShadow"
                };
                dstProjectiveShadow.Create();
            }

            int kernel = compute.FindKernel("BakeRelocation");
            compute.SetInts(Shader.PropertyToID("gsVoxCount"), vc.x, vc.y, vc.z);
            compute.SetFloat(Shader.PropertyToID("gsVoxSize"), voxelSize);
            compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcTsdf"), _volume);
            compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcColor"), _colorVolume);
            compute.SetTexture(kernel, BakeSrcAdmissionTraceID, _admissionTraceVolume);
            compute.SetTexture(kernel, VolumeRWID, dstTsdf);
            compute.SetTexture(kernel, ColorVolumeRWID, dstColor);
            compute.SetTexture(kernel, AdmissionTraceRWID, dstAdmissionTrace);
            compute.SetMatrix(Shader.PropertyToID("gsBakeInvRelocation"), invRelocation);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);

            int tx = Mathf.CeilToInt(vc.x / 4f);
            int ty = Mathf.CeilToInt(vc.y / 4f);
            int tz = Mathf.CeilToInt(vc.z / 4f);
            compute.Dispatch(kernel, tx, ty, tz);

            if (dstProjectiveShadow != null)
            {
                compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcTsdf"), _projectiveShadowVolume);
                compute.SetTexture(kernel, VolumeRWID, dstProjectiveShadow);
                compute.SetTexture(kernel, ColorVolumeRWID, dstColor); // write-guarded
                compute.SetFloat(WriteColorID, 0f);
                compute.SetFloat(WriteAdmissionTraceID, 0f);
                compute.Dispatch(kernel, tx, ty, tz);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
            }
            GL.Flush();

            // Swap volumes: destroy old, adopt baked textures.
            // Avoids Graphics.CopyTexture on 3D RTs which can silently fail on Vulkan/Quest.
            Destroy(_volume);
            Destroy(_colorVolume);
            if (_projectiveShadowVolume) Destroy(_projectiveShadowVolume);
            if (_admissionTraceVolume) Destroy(_admissionTraceVolume);
            _volume = dstTsdf;
            _colorVolume = dstColor;
            _projectiveShadowVolume = dstProjectiveShadow;
            _admissionTraceVolume = dstAdmissionTrace;

            // Rebind global texture references (used by render shader for freeze tint etc.)
            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);

            // Rebind per-kernel UAV references so subsequent integrations/clears use new textures
            RebindVolumeTextures();
            MarkAllChunksDirty();
            TopologyInvalidated?.Invoke();

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
            if (_projectiveShadowVolume != null)
            {
                _freezeKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _freezeKernel.DispatchFit(_projectiveShadowVolume);
                _freezeKernel.Set(VolumeRWID, _volume);
            }
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
            if (_projectiveShadowVolume != null)
            {
                _unfreezeKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _unfreezeKernel.DispatchFit(_projectiveShadowVolume);
                _unfreezeKernel.Set(VolumeRWID, _volume);
            }
            Logger.Info("UnfreezeInView dispatched");
        }

        // ── 逐块可逆冻结公共 API ─────────────────────────────────────
        /// <summary>冻结票箱（每块 uint2：x=自由空间票 y=遮挡票），调度器周期回读。</summary>
        public ComputeBuffer FrozenChunkVotes => _frozenChunkVotes;
        /// <summary>成熟度账（每块 uint4：x=surface体素数 y=其中已冻结），调度器周期回读。</summary>
        public ComputeBuffer ChunkMaturity => _chunkMaturity;
        /// <summary>冻结 API 是否可用（GPU 资源已惰性分配）。</summary>
        public bool FrozenBlockReady => _volume != null && _frozenChunkVotes != null &&
                                        _applyFreezeMaskKernel.Shader != null;
        /// <summary>冻结单元块总数（独立于脏账本网格，frozenChunkSize 边长）。</summary>
        public int FrozenBlockCount => _frozenChunkCount.x * _frozenChunkCount.y * _frozenChunkCount.z;
        /// <summary>冻结块网格维度（供调度器解码块坐标）。</summary>
        public int3 FrozenChunkCount => _frozenChunkCount;

        /// <summary>
        /// 上传 set/clear 块位图并翻符号：set=冻结（weight 正→负），clear=解冻（负→正），
        /// 解冻块票箱同批清零。主卷与影子卷各执行一次。掩码一次性消费——返回时两个
        /// 传入数组已被清零，GPU 侧掩码同步复位，残留位不会误伤下一批。
        /// 提取层 abs(weight) 对符号透明：本调用不标脏、不触发重提网格。
        /// </summary>
        public void ApplyChunkFreezeMasks(uint[] setMask, uint[] clearMask)
        {
            if (!FrozenBlockReady || setMask == null || clearMask == null) return;
            _chunkFreezeSetMask.SetData(setMask);
            _chunkFreezeClearMask.SetData(clearMask);
            _applyFreezeMaskKernel.DispatchFit(_volume);
            if (_projectiveShadowVolume != null)
            {
                _applyFreezeMaskKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _applyFreezeMaskKernel.DispatchFit(_projectiveShadowVolume);
                _applyFreezeMaskKernel.Set(VolumeRWID, _volume);
            }
            _clearVotesKernel.DispatchFit(_frozenChunkVotes.count, 1, 1);
            System.Array.Clear(setMask, 0, setMask.Length);
            System.Array.Clear(clearMask, 0, clearMask.Length);
            _chunkFreezeSetMask.SetData(setMask);
            _chunkFreezeClearMask.SetData(clearMask);
        }

        /// <summary>清零成熟度账并重新普查一遍（调度器每窗调用，回读 ChunkMaturity）。</summary>
        public void RefreshChunkMaturity()
        {
            if (!FrozenBlockReady) return;
            _chunkMaturity.SetData(_maturityZeros);
            _maturityKernel.DispatchFit(_volume);
        }

        /// <summary>清零全部冻结票箱（调度器每窗回读后调用，进入下一统计窗）。</summary>
        public void ClearAllFrozenVotes()
        {
            if (!FrozenBlockReady) return;
            _frozenChunkVotes.SetData(_voteZeros);
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
            // 角速度由 DepthCapture 在深度帧到达时用原始 Pose 四元数计算
            // （dt=真实深度帧间隔），这里只消费。旧实现用积分间隔 ÷ 矩阵.rotation
            // 增量：深度帧率低于积分率时系统性放大，且 ScaleFlipZ 负行列式矩阵的
            // 四元数提取有分支不连续风险——曾致诊断运动位 100% 饱和失效。
            if (dc.ViewInv != null && dc.ViewInv.Length > 0)
            {
                _smoothedAngSpeed = dc.SmoothedDepthAngularSpeed;
                if (motionGateDegPerSec > 0f && _smoothedAngSpeed > motionGateDegPerSec)
                {
                    _motionGatedSinceStats++;
                    _pendingCamFrame = null; // 丢弃过期颜色帧，防与下一帧位姿错配
                    return;
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
            compute.SetFloat(FreeSpaceCarveBoostID, freeSpaceCarveBoost);
            compute.SetFloat(RescueSeedDistOnlyID, rescueSeedDistOnly ? 1f : 0f);
            compute.SetFloat(MotionSeedBlockID, motionSeedBlockDegPerSec);
            compute.SetFloat(AbstainSeedGuardID, abstainSeedGuard ? 1f : 0f);
            compute.SetFloat(RawSeedGateID, rawSeedGate ? 1f : 0f);
            compute.SetFloat(RawSeedBandID, rawSeedBand);
            compute.SetFloat(DiagDepthGapBaseID, diagnosticDepthGapBaseMeters);
            compute.SetFloat(DiagDepthGapScaleID, diagnosticDepthGapDistanceScale);
            compute.SetFloat(DilationProductionGateID, dilationProductionGate ? 1f : 0f);
            compute.SetFloat(DilationBlockRelayID, dilationBlockRelayWrites ? 1f : 0f);
            compute.SetFloat(DilationBlockSparseID, dilationBlockSparseWrites ? 1f : 0f);
            compute.SetFloat(ProvisionalSeedWeightID, provisionalSeedWeight);
            compute.SetFloat(FormalSurfaceWeightID, minMeshWeight);
            compute.SetFloat(DiagnosticAngularSpeedID, _smoothedAngSpeed);
            compute.SetFloat(FrozenBlockEnableID, frozenBlockEnable ? 1f : 0f);
            compute.SetFloat(FrozenVoteQualityMinID, frozenVoteQualityMin);
            compute.SetFloat(FrozenVoteMarginID, frozenVoteMargin);

            EnsureCamFrameCopy();
            bool productionCamAvailable = _pendingCamFrame != null && _camFrameCopy != null;
            if (productionCamAvailable)
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
            _integrateKernel.Set(DepthCapture.EdgeReasonTexID, dc.EdgeReasonTex);

            // A: unchanged production path (projective difference scaled by normal cosine).
            BeginDirtyEpoch();
            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(CarveStatsID, _carveStats);
            _integrateKernel.DispatchFit(_frustumVolume.count, 1);

            // B: read-only KinectFusion-style projective TSDF shadow. It receives the
            // exact same depth, pose, gates, quality and carve settings, but stores the
            // unscaled ray-depth difference. No color/global production state is written.
            if (_projectiveShadowVolume != null && _projectiveShadowCarveStats != null)
            {
                compute.SetFloat(UseRawProjectiveSdfID, 1f);
                compute.SetFloat(WriteColorID, 0f);
                compute.SetFloat(WriteAdmissionTraceID, 0f);
                ConfigureDirtyTracking(false);
                compute.SetInt(CamAvailableID, 0);
                _integrateKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _integrateKernel.Set(ColorVolumeRWID, _colorVolume); // write-guarded
                _integrateKernel.Set(CarveStatsID, _projectiveShadowCarveStats);
                _integrateKernel.DispatchFit(_frustumVolume.count, 1);

                // Restore production bindings so external callers never inherit B state.
                compute.SetFloat(UseRawProjectiveSdfID, 0f);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
                ConfigureDirtyTracking(true);
                compute.SetInt(CamAvailableID, productionCamAvailable ? 1 : 0);
                _integrateKernel.Set(VolumeRWID, _volume);
                _integrateKernel.Set(CarveStatsID, _carveStats);
            }

            IntegrationCount++;
            _pendingCamFrame = null;

            if (warmupIntegrations > 0 && IntegrationCount == warmupIntegrations)
            {
                Logger.Info($"Warmup complete ({warmupIntegrations} frames), clearing volume to discard sensor startup noise");
                Clear();
            }

            float t = Time.time;
            if (!_pruneCycleActive && t - _lastPruneTime >= pruneIntervalSeconds)
            {
                _lastPruneTime = t;
                _nextPruneSlice = 0;
                _pruneCycleActive = true;
            }

            // Pruning used to scan the entire 3D volume in one dispatch.  Keep the
            // exact same voxel rule, but amortize it over integrations to avoid a
            // periodic full-volume GPU spike on Quest.
            if (_pruneCycleActive)
            {
                int sliceCount = Mathf.Min(Mathf.Max(1, pruneSlicesPerIntegration),
                    voxelCount.z - _nextPruneSlice);
                compute.SetInt(PruneZOffsetID, _nextPruneSlice);
                compute.SetInt(PruneZCountID, sliceCount);

                _pruneKernel.Set(VolumeRWID, _volume);
                _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
                ConfigureDirtyTracking(true);
                _pruneKernel.DispatchFit(voxelCount.x, voxelCount.y, sliceCount);

                if (_projectiveShadowVolume != null)
                {
                    compute.SetFloat(WriteColorID, 0f);
                    compute.SetFloat(WriteAdmissionTraceID, 0f);
                    ConfigureDirtyTracking(false);
                    _pruneKernel.Set(VolumeRWID, _projectiveShadowVolume);
                    _pruneKernel.Set(ColorVolumeRWID, _colorVolume); // write-guarded
                    _pruneKernel.DispatchFit(voxelCount.x, voxelCount.y, sliceCount);
                    compute.SetFloat(WriteColorID, 1f);
                    compute.SetFloat(WriteAdmissionTraceID, 1f);
                    ConfigureDirtyTracking(true);
                    _pruneKernel.Set(VolumeRWID, _volume);
                }

                _nextPruneSlice += sliceCount;
                if (_nextPruneSlice >= voxelCount.z)
                {
                    _nextPruneSlice = 0;
                    _pruneCycleActive = false;
                }
            }

            if (coverageUpdateInterval > 0 && !_coverageReadbackPending)
            {
                _integrationsSinceCoverage++;
                if (_integrationsSinceCoverage >= coverageUpdateInterval)
                {
                    _integrationsSinceCoverage = 0;
                    DispatchCoverageCount();
                    RequestCarveStatsReadback();
                    RequestProjectiveShadowCarveStatsReadback();
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

            // A saved production TSDF cannot be converted into the raw-projective
            // counterfactual. Start B empty; it will accumulate only subsequent live frames.
            if (_projectiveShadowVolume != null)
            {
                compute.SetFloat(WriteColorID, 0f);
                compute.SetFloat(WriteAdmissionTraceID, 0f);
                _clearKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _clearKernel.Set(ColorVolumeRWID, _colorVolume);
                _clearKernel.DispatchFit(_projectiveShadowVolume);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
                _clearKernel.Set(VolumeRWID, _volume);
                _projectiveShadowCarveStats?.SetData(ZeroCarveStats);
                HasProjectiveShadowCarveStats = false;
            }

            GL.Flush();

            IntegrationCount = integrationCount;
            _frustumReady = false;
            MarkAllChunksDirty();
            TopologyInvalidated?.Invoke();

            Logger.Info($"Volumes loaded: {s}, integrationCount={integrationCount}");
            return true;
        }
    }
}

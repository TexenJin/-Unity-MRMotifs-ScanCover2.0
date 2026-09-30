using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
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
        // Native 10 cm A/B profile.  Halving every lattice dimension preserves
        // the original 9.6 x 6.4 x 9.6 metre world extent; this is one coarse
        // TSDF from integration through extraction, not a 5 cm mesh decimator.
        [SerializeField] private int3 voxelCount = new(96, 64, 96);
        [SerializeField] private float voxelSize = 0.10f;
        // Preserve the v1.3 three-voxel truncation band in voxel units.
        [SerializeField] private float voxelDistance = 0.30f;
        [SerializeField] private float voxelMin = 0.1f;

        [Header("Integration")]
        [SerializeField] private float depthDisparityThreshold = 0.5f;
        [SerializeField] private float maxUpdateDist = 5f;
        [Tooltip("最近更新距离：视锥内距相机小于此值的体素不积分。原 0.5m 会在头周留下覆盖空洞，0.15 贴脸也覆盖（近距深度噪声大，如出现贴脸飞面可调回 0.3）")]
        [SerializeField] private float minUpdateDist = 0.15f;
        [Tooltip("头部排除区半径（仅 StandaloneRoomScanner 开启排除区时生效）。0.6 = QRS 原版（防手入网但头周覆盖空洞大），0.35 = 折中")]
        [SerializeField, Range(0.1f, 1f)] private float exclusionRadius = 0.35f;
        [SerializeField] private int maxFrustumPositions = 1000000;
        [SerializeField, Tooltip("在不增加每帧样本数的前提下，让视锥采样格逐帧走完8个半体素相位。只改变哪些世界体素获得同一套已准入深度证据；不放宽深度、法线、边缘、运动或枪胶门禁。用于消除相机格与世界5cm格对齐造成的漏写、弱权重和八邻域缺样。")]
        private bool enableFrustumPhaseCoverage = true;

        [Header("校枪—分层凝胶（GPU 证据层）")]
        [Tooltip("同吃生产预处理深度/法线/姿态，估计融合专用小修正并维护 K<=3 局部候选。本层自身不写 TSDF；下方受保护实验可选择消费结果。")]
        [SerializeField] private bool enableGunGelEvidenceShadow = true;
        [Tooltip("影子层采样步长。8 与离线验证口径一致，优先控制 Quest GPU/回读负担。")]
        [SerializeField, Range(2, 16)] private int gunGelPixelStride = 8;
        [Tooltip("候选空间哈希格尺寸（米）。它只是索引邻域，不是把房间切成冻结管理块。")]
        [SerializeField, Range(0.05f, 0.25f)] private float gunGelCellSize = 0.10f;
        [Tooltip("候选分类账刷新间隔（融合帧）。校枪仍逐帧异步计算。")]
        [SerializeField, Range(10, 120)] private int gunGelReportInterval = 30;

        [Header("枪胶受保护融合实验")]
        [Tooltip("开=同帧预处理深度先留在三槽流水线：校枪可靠时应用小修正；欠秩、少配、缺证或修正解越界时撤销校正权并以原始位姿进入普通生产门；" +
                 "只有明确快转/快移仍整帧停笔。逐点枪胶同样只拦双证冲突、已有反对票和跨面错配。关=原始生产融合。必须从空 TSDF 开始对比。")]
        [SerializeField] private bool enableGunGelGuardedFusionExperiment = true;
        [Tooltip("仅保留同帧 GunGel 身份/候选证据；关闭时也不允许退回旧裁判平面改写 TSDF。生产 TSDF 只吃通过前置健康筛查的原始深度。")]
        [SerializeField] private bool enableGunGelTsdfAdmission = false;
        [Tooltip("旧实验：把 GunGel 的位姿小修正应用到生产深度。新产品化链默认关闭；GunGel 只负责身份，不改距离或位姿。")]
        [SerializeField] private bool enableGunGelPoseCorrection = false;
        [Tooltip("旧的 TSDF 前裁判平面约束。已被 TSDF 后 SurfaceProductizer 取代，默认且启动时强制关闭。")]
        [SerializeField] private bool enableFinalCourtAdmissionExperiment = false;
        [Tooltip("校枪参与生产融合所需的最少点面对应数。实机健康帧约1500；低于此值说明可见稳定凝胶不足。")]
        [SerializeField, Range(128, 4096)] private int gunGelFusionMinCorrespondences = 800;
        [Tooltip("允许写入 TSDF 的最大校枪平移（毫米）。撞30mm求解上限的帧永远拒绝。")]
        [SerializeField, Range(1f, 30f)] private float gunGelFusionMaxTranslationMm = 20f;
        [Tooltip("允许写入 TSDF 的最大校枪转角（度）。")]
        [SerializeField, Range(0.1f, 1.5f)] private float gunGelFusionMaxRotationDeg = 1f;
        [Tooltip("枪胶实验的角速度硬闸（°/s）。203438实机在48°/s撞30mm上限，默认35提前停笔。")]
        [SerializeField, Range(5f, 120f)] private float gunGelFusionMaxAngularSpeed = 35f;
        [Tooltip("枪胶实验的线速度硬闸（m/s）。平移同样会触发深度/姿态时延错位。")]
        [SerializeField, Range(0.05f, 1f)] private float gunGelFusionMaxLinearSpeed = 0.35f;

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

        [Header("噪声模型加权（造炮）")]
        [Tooltip("总开关：融合权重 w∝1/σ²(d,θ,ω)——距离/掠射角/运动状态三项降权。08-21 用户拍板，" +
                 "定罪链：定点实验 O≈0（偏移脱罪）+ R 随距离掠射爆炸（方差定罪）+ 动态 O 一动掉负静止养回、平移也掉负（时延族实锤）。" +
                 "纯数据层通用机制，符合阻尼封顶。 (default true)")]
        [SerializeField] private bool noiseMotionWeightEnable = true;
        [Tooltip("运动角速度参考值（°/s）：达到此速度时运动质量分折到地板。必须远低于整帧闸 90——90 以上停笔，参考值~90 之间是降权慢写带。 (default 30)")]
        [SerializeField, Range(5f, 120f)] private float noiseMotionAngRefDegPerSec = 30f;
        [Tooltip("运动线速度参考值（m/s）：平移也触发时延（实锤），与角速度各归一取大。正常扫房步速 ~0.3-0.5m/s。 (default 0.3)")]
        [SerializeField, Range(0.05f, 2f)] private float noiseMotionLinRefMps = 0.3f;
        [Tooltip("运动质量地板：再快也保留此比例权重，防断粮、防与整帧闸之间出死区。 (default 0.15)")]
        [SerializeField, Range(0.01f, 1f)] private float noiseMotionFloor = 0.15f;
        [Tooltip("运动证据取得正式发布/纠偏权限的最低质量。低于此值仍可留下暂存候选，但不能转正、改写正式面、投冻结票或触发纸皮替换。 (default 0.70)")]
        [SerializeField, Range(0.15f, 1f)] private float motionConfirmQualityMin = 0.70f;
        [Tooltip("距离噪声指数：>1 让远距观测降权更陡（深度 σ 随距离平方增长）。1=保持现有线性不动。 (default 1)")]
        [SerializeField, Range(0.5f, 3f)] private float noiseDistExponent = 1f;
        [Tooltip("掠射角噪声指数：>1 让掠射观测降权更陡。1=保持现有线性不动。 (default 1)")]
        [SerializeField, Range(0.5f, 3f)] private float noiseAngleExponent = 1f;

        [Header("M1 成熟面观测折让")]
        [Tooltip("总开关：已长熟的面对小偏差新观测打折接收——几何大幅少跟、权重慢涨、分歧按一致记账；超过小偏差带的真变化仍走原有纠错通道。它只作用于已达最大成熟门槛的存量面，不阻止空点/弱点继续补覆盖。 (default true)")]
        [SerializeField] private bool enableMatureSurfaceObsDiscount = true;
        [Tooltip("折让判据的成熟门槛：weight≥此值的面才享折让。拍板口径 0.5=长熟面——不复用 frozenMatureWeight（实为 0.15=刚转正，折让会把爬坡期增长拖慢 20 倍）。 (default 0.5)")]
        [SerializeField, Range(0.1f, 1f)] private float matureSurfaceObsWeightMin = 0.5f;
        [Tooltip("小矛盾带半宽（归一化 sd，0.15≈2.2cm）：|新观测-存量|≤此值才折让；超带=真变化全速放行。必须小于矛盾杆 0.2 才有意义。 (default 0.15)")]
        [SerializeField, Range(0.05f, 0.4f)] private float matureSurfaceObsMargin = 0.15f;
        [Tooltip("折让力度：带内观测的积分权重/扣减乘此系数。0.05=1/20 速度=真变化的申诉通道慢但不设硬顶。 (default 0.05)")]
        [SerializeField, Range(0.01f, 0.5f)] private float matureSurfaceObsDiscount = 0.05f;

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

        [Header("InfiniTAM architecture baseline")]
        [SerializeField, Tooltip("独立 InfiniTAM V1.3 行为基线：Quest 世界位姿下直接写入唯一 raw-projective TSDF；模型 raycast 与残差只读旁证。GunGel、裁判、种面标尺、冻结和制品链全部旁路。")]
        private bool enableInfiniTamBaseline = true;
        [SerializeField, Tooltip("历史 V1.6-V1.10 实验：让未完整移植的 ICP 跟踪器拥有 TSDF 写入、位姿修正和网格发布权。V1.3 恢复基线必须保持关闭；待完整移植 TAM 金字塔、阻尼回退、质量状态和重定位后才能重新启用。")]
        private bool enableInfiniTamTrackingAuthority = false;
        [SerializeField, Min(1), Tooltip("只允许互不重复且处于静止窗口内的 Quest 深度帧建立初始 TSDF；达到该融合帧数后，所有后续帧必须先完成同帧 model-to-frame tracking。")]
        private int infiniTamTrackingBootstrapFrames = 6;
        [SerializeField, Range(1, 12), Tooltip("首张种子写入前必须连续出现的稳定、互不重复深度帧数。")]
        private int infiniTamBootstrapStillFrames = 4;
        [SerializeField, Range(1f, 45f), Tooltip("InfiniTAM 启动/复核期允许建底的最大角速度（度/秒）。正式建图后仍使用普通运动闸。")]
        private float infiniTamBootstrapMaxAngularDegPerSec = 15f;
        [SerializeField, Range(0.01f, 0.5f), Tooltip("InfiniTAM 启动/复核期允许建底的最大线速度（米/秒）。")]
        private float infiniTamBootstrapMaxLinearMps = 0.08f;
        [SerializeField, Range(1, 8), Tooltip("初始 TSDF 在连续多少张严格通过跟踪门后才可作为正式模型并显示网格。")]
        private int infiniTamBootstrapConfirmFrames = 3;
        [SerializeField, Range(1, 8), Tooltip("尚未转正的初始模型连续跟踪失败多少次后清卷重建，防止坏底图成为永久答案。")]
        private int infiniTamBootstrapRejectsBeforeReseed = 3;
        [SerializeField, Range(1, 8), Tooltip("正式模型失跟后，必须连续多少张同帧跟踪重新通过才恢复写入。重锁期保留旧网格但不写 TSDF。")]
        private int infiniTamRecoveryConfirmFrames = 3;
        [SerializeField, Min(64), Tooltip("InfiniTAM 同帧跟踪允许融合所需的最少点面对应数。")]
        private int infiniTamTrackingMinCorrespondences = 256;
        [SerializeField, Range(3, 6), Tooltip("历史跟踪实验的最低有效秩；V1.3 只读旁证不消费此门槛。")]
        private int infiniTamTrackingMinRank = 6;
        [SerializeField, Range(5f, 60f), Tooltip("单帧模型跟踪允许写入 TSDF 的最大平移修正（毫米）。")]
        private float infiniTamTrackingMaxTranslationMm = 30f;
        [SerializeField, Range(0.1f, 3f), Tooltip("单帧模型跟踪允许写入 TSDF 的最大旋转修正（度）。")]
        private float infiniTamTrackingMaxRotationDeg = 1.5f;
        [SerializeField, Range(0f, 5f), Tooltip("点面平均残差至少改善该毫米数；没有变好就停笔。")]
        private float infiniTamTrackingMinImprovementMm = 0.5f;
        [SerializeField, Range(25f, 95f), Tooltip("修正后位于 3cm 内的点面对应比例下限。")]
        private float infiniTamTrackingMinWithin30Percent = 60f;

        [Header("Projective TSDF A/B")]
        [SerializeField, Tooltip("建立一份只读 KinectFusion 式 raw-projective TSDF 影子体。生产体仍使用现有 raw×法向余弦；影子体只用于对照统计和手动切换显示。")]
        private bool enableProjectiveShadow = false;

        [Header("置信度通道 v1（只读影子）")]
        [SerializeField, Tooltip("分歧 EMA 基准速率，再乘观测质量 q（低质观测少说话）。0=不更新，1=逐帧覆盖。")]
        private float confidenceRate = 0.15f;
        [SerializeField, Tooltip("高/中置信分界（归一化 sd 的逐帧分歧 EMA，0.1≈1.5cm 抖动）。")]
        private float confidenceMidMax = 0.1f;
        [SerializeField, Tooltip("中/低置信分界（0.25≈3.8cm，超过=几何在打架）。")]
        private float confidenceLowMin = 0.25f;

        [Header("Camera Color")]
        [Tooltip("Exposure boost for camera texture. Quest 3 passthrough cameras produce dim images. (default 3.0)")]
        [SerializeField, Range(1f, 10f)] private float cameraExposure = 3f;

        private RenderTexture _volume;
        private RenderTexture _colorVolume;
        private RenderTexture _projectiveShadowVolume;
        private RenderTexture _admissionTraceVolume;
        private RenderTexture _confidenceVolume;
        private RenderTexture _coherenceVolume;
        // Baseline-only canonical TSDF vote count.  The legacy/product route
        // receives only a 1x1 descriptor placeholder and never reads or writes it.
        private RenderTexture _infiniTamVoteWeightVolume;
        // Baseline-only prediction of the sole TSDF.  The TSDF binding and
        // prediction textures stay read-only; its exact-frame tracking decision
        // is consumed only by the quality gate below, before fusion.
        private InfiniTamModelRaycastAudit _infiniTamModelRaycast;
        private RenderTexture _tsdfResponsibilityVolume;
        private RenderTexture _tsdfSupportResponsibilityVolume;
        private bool _tsdfResponsibilityCaptureEnabled;
        private bool _tsdfResponsibilityExportPending;
        internal string LastTsdfResponsibilityExportError { get; private set; } = string.Empty;

        /// <summary>3D RenderTexture (R8G8_SNorm) storing the truncated signed distance field.</summary>
        public RenderTexture Volume => _volume;
        /// <summary>互斥的 InfiniTAM 风格第一版基线是否拥有生产融合与提面路径。</summary>
        public bool InfiniTamBaselineEnabled => enableInfiniTamBaseline;
        /// <summary>
        /// V1.3 production consumes only the cleaned depth texture plus the
        /// Quest pose. The dormant tracking experiment still needs normals,
        /// dilation and edge evidence and therefore must keep the full input set.
        /// </summary>
        public bool InfiniTamCompactDepthOnly => enableInfiniTamBaseline &&
                                                 !enableInfiniTamTrackingAuthority;
        /// <summary>
        /// V1.3 publishes after the sole TSDF has received its first real
        /// observation. The dormant V1.6+ experiment retains its old tracked
        /// confirmation boundary only when explicitly re-enabled.
        /// </summary>
        public bool InfiniTamMeshPublicationReady => !enableInfiniTamBaseline ||
            (!enableInfiniTamTrackingAuthority
                ? IntegrationCount > 0
                : _infiniTamTrackingInitialised);
        /// <summary>只读模型预测距离图（米，审计降采样分辨率）；不接生产 shader。</summary>
        public RenderTexture InfiniTamModelDepth => _infiniTamModelRaycast?.ModelDepth;
        /// <summary>只读模型预测减当前观测的残差图（毫米）；负值表示模型在前。</summary>
        public RenderTexture InfiniTamModelResidualMm => _infiniTamModelRaycast?.ModelResidualMm;
        /// <summary>只读 A/B 影子体：使用 raw projective SDF，但不写颜色、不替换生产体。</summary>
        public RenderTexture ProjectiveShadowVolume => _projectiveShadowVolume;
        public bool ProjectiveShadowEnabled => !enableInfiniTamBaseline &&
                                               enableProjectiveShadow &&
                                               _projectiveShadowVolume != null;
        /// <summary>3D RenderTexture (RGBA8_UNorm) storing per-voxel accumulated color.</summary>
        public RenderTexture ColorVolume => _colorVolume;
        /// <summary>Read-only provenance sidecar for dilation admission. Never changes TSDF production decisions.</summary>
        public RenderTexture AdmissionTraceVolume => _admissionTraceVolume;
        /// <summary>置信度通道 v1：分歧 EMA 体（R8，0=逐帧观测一致 1=完全矛盾）。只读影子，生产路径一律不读。</summary>
        public RenderTexture ConfidenceVolume => _confidenceVolume;
        /// <summary>v2 相干通道：有符号分歧 EMA（0.5=中性）。噪声回中性，纠错/真变化偏两端。只读影子。</summary>
        public RenderTexture CoherenceVolume => _coherenceVolume;
        /// <summary>Read-only audit sidecar; never sampled by production fusion or extraction.</summary>
        public RenderTexture TsdfResponsibilityVolume => _tsdfResponsibilityVolume;
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
        private static readonly int FrustumPhaseOffsetID = Shader.PropertyToID("gsFrustumPhaseOffset");
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
        private static readonly int MatureObsEnableID = Shader.PropertyToID("gsMatureObsEnable");
        private static readonly int MatureObsWeightMinID = Shader.PropertyToID("gsMatureObsWeightMin");
        private static readonly int MatureObsMarginID = Shader.PropertyToID("gsMatureObsMargin");
        private static readonly int MatureObsDiscountID = Shader.PropertyToID("gsMatureObsDiscount");
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
        private static readonly int NoiseMotionQualityID = Shader.PropertyToID("gsNoiseMotionQuality");
        private static readonly int MotionAuthorityQualityID = Shader.PropertyToID("gsMotionAuthorityQuality");
        private static readonly int MotionConfirmQualityMinID = Shader.PropertyToID("gsMotionConfirmQualityMin");
        private static readonly int NoiseDistExpID = Shader.PropertyToID("gsNoiseDistExp");
        private static readonly int NoiseAngExpID = Shader.PropertyToID("gsNoiseAngExp");
        private static readonly int CamRGBID = Shader.PropertyToID("gsCamRGB");
        private static readonly int CamAvailableID = Shader.PropertyToID("gsCamAvailable");
        private static readonly int CamPosID = Shader.PropertyToID("gsCamPos");
        private static readonly int CamInvRotID = Shader.PropertyToID("gsCamInvRot");
        private static readonly int CamFocalLenID = Shader.PropertyToID("gsCamFocalLen");
        private static readonly int CamPrincipalPtID = Shader.PropertyToID("gsCamPrincipalPt");
        private static readonly int CamSensorResID = Shader.PropertyToID("gsCamSensorRes");
        private static readonly int CamCurrentResID = Shader.PropertyToID("gsCamCurrentRes");
        private static readonly int CamExposureID = Shader.PropertyToID("gsCamExposure");
        private static readonly int FusionCorrectionID = Shader.PropertyToID("gsFusionCorrection");
        private static readonly int GunGelObservationsID = Shader.PropertyToID("gsGunGelObservations");
        private static readonly int GunGelCorrespondencesID = Shader.PropertyToID("gsGunGelCorrespondences");
        private static readonly int GunGelCorrespondenceIdentityID =
            Shader.PropertyToID("gsGunGelCorrespondenceIdentity");
        private static readonly int GunGelObservationGridID = Shader.PropertyToID("gsGunGelObservationGrid");
        private static readonly int GunGelPixelStrideID = Shader.PropertyToID("gsGunGelPixelStride");
        private static readonly int GunGelAdmissionEnableID = Shader.PropertyToID("gsGunGelAdmissionEnable");
        private static readonly int FinalCourtVerdictsID =
            Shader.PropertyToID("gsFinalCourtVerdicts");
        private static readonly int FinalCourtPlanesID =
            Shader.PropertyToID("gsFinalCourtPlanes");
        private static readonly int FinalCourtGenerationsID =
            Shader.PropertyToID("gsFinalCourtGenerations");
        private static readonly int FinalCourtVerdictCapacityID =
            Shader.PropertyToID("gsFinalCourtVerdictCapacity");
        private static readonly int FinalCourtAdmissionEnableID =
            Shader.PropertyToID("gsFinalCourtAdmissionEnable");
        private static readonly int GunGelSuccessionInvalidationArgsID =
            Shader.PropertyToID("gsGunGelSuccessionInvalidationArgs");
        private static readonly int GunGelSuccessionInvalidationRegionsID =
            Shader.PropertyToID("gsGunGelSuccessionInvalidationRegions");
        private static readonly int GunGelSuccessionInvalidationCapacityID =
            Shader.PropertyToID("gsGunGelSuccessionInvalidationCapacity");
        private static readonly int ShellWitnessEpochsID = Shader.PropertyToID("gsShellWitnessEpochs");
        private static readonly int ShellWitnessCellCountID = Shader.PropertyToID("gsShellWitnessCellCount");
        private static readonly int ShellWitnessStrideID = Shader.PropertyToID("gsShellWitnessStride");
        private static readonly int ShellWitnessEpochID = Shader.PropertyToID("gsShellWitnessEpoch");
        private static readonly int ShellWitnessMaxAgeID = Shader.PropertyToID("gsShellWitnessMaxAge");
        private static readonly int ShellWitnessEnableID = Shader.PropertyToID("gsShellWitnessEnable");
        private static readonly int UseRawProjectiveSdfID = Shader.PropertyToID("gsUseRawProjectiveSdf");
        private static readonly int InfiniTamBaselineID = Shader.PropertyToID("gsInfiniTamBaseline");
        private static readonly int InfiniTamVoteWeightRWID =
            Shader.PropertyToID("gsInfiniTamVoteWeightRW");
        private static readonly int InfiniTamTicketStatsID =
            Shader.PropertyToID("_InfiniTamTicketStats");
        private static readonly int InfiniTamTicketEnabledID =
            Shader.PropertyToID("gsInfiniTamTicketEnabled");
        private static readonly int WriteColorID = Shader.PropertyToID("gsWriteColor");
        private static readonly int AdmissionTraceRWID = Shader.PropertyToID("gsAdmissionTraceRW");
        private static readonly int WriteAdmissionTraceID = Shader.PropertyToID("gsWriteAdmissionTrace");
        private static readonly int TsdfResponsibilityRWID = Shader.PropertyToID("gsTsdfResponsibilityRW");
        private static readonly int TsdfSupportResponsibilityRWID = Shader.PropertyToID("gsTsdfSupportResponsibilityRW");
        private static readonly int TsdfResponsibilityAvailableID = Shader.PropertyToID("gsTsdfResponsibilityAvailable");
        private static readonly int TsdfResponsibilityWriteID = Shader.PropertyToID("gsTsdfResponsibilityWrite");
        private static readonly int TsdfResponsibilityIntegrationID = Shader.PropertyToID("gsTsdfResponsibilityIntegration");
        private static readonly int TsdfResponsibilityExportSliceID =
            Shader.PropertyToID("gsTsdfResponsibilityExportSlice");
        private static readonly int TsdfSupportResponsibilityExportSliceID =
            Shader.PropertyToID("gsTsdfSupportResponsibilityExportSlice");
        private static readonly int TsdfResponsibilityExportZID =
            Shader.PropertyToID("gsTsdfResponsibilityExportZ");
        private static readonly int ConfidenceRWID = Shader.PropertyToID("gsConfidenceRW");
        private static readonly int CoherenceRWID = Shader.PropertyToID("gsCoherenceRW");
        private static readonly int ConfidenceWriteID = Shader.PropertyToID("gsConfidenceWrite");
        private static readonly int ConfidenceRateID = Shader.PropertyToID("gsConfidenceRate");
        private static readonly int ConfidenceMidMaxID = Shader.PropertyToID("gsConfidenceMidMax");
        private static readonly int ConfidenceLowMinID = Shader.PropertyToID("gsConfidenceLowMin");
        private static readonly int ConfidenceStatsID = Shader.PropertyToID("_ConfidenceStats");
        private static readonly int ConfidenceGlobalTexID = Shader.PropertyToID("gsConfidence");
        private static readonly int TemporalReasonAvailableID = Shader.PropertyToID("gsTemporalReasonAvailable");
        private static readonly int BakeSrcAdmissionTraceID = Shader.PropertyToID("gsBakeSrcAdmissionTrace");
        private static readonly int BakeSrcTsdfResponsibilityID = Shader.PropertyToID("gsBakeSrcTsdfResponsibility");
        private static readonly int BakeSrcTsdfSupportResponsibilityID = Shader.PropertyToID("gsBakeSrcTsdfSupportResponsibility");
        private static readonly int PruneZOffsetID = Shader.PropertyToID("gsPruneZOffset");
        private static readonly int PruneZCountID = Shader.PropertyToID("gsPruneZCount");
        private static readonly int DirtyChunkEpochsID = Shader.PropertyToID("_DirtyChunkEpochs");
        private static readonly int DirtyBoundaryEpochsID = Shader.PropertyToID("_DirtyBoundaryEpochs");
        private static readonly int ActivePageEpochsID = Shader.PropertyToID("_ActivePageEpochs");
        private static readonly int ActivePageObservedEpochsID = Shader.PropertyToID("_ActivePageObservedEpochs");
        private static readonly int ActivePageBoundaryEpochsID = Shader.PropertyToID("_ActivePageBoundaryEpochs");
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
        private static readonly int FrozenChunkBitsID = Shader.PropertyToID("_FrozenChunkBits");
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
        private ComputeKernelHelper _clearInfiniTamVotesKernel;
        private ComputeKernelHelper _invalidateGunGelSuccessionsKernel;
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
        private ComputeBuffer _activePageEpochs;
        private ComputeBuffer _activePageObservedEpochs;
        private ComputeBuffer _activePageBoundaryEpochs;
        private ComputeBuffer _chunkFreezeSetMask;
        private ComputeBuffer _chunkFreezeClearMask;
        private ComputeBuffer _frozenChunkVotes;
        private ComputeBuffer _frozenChunkBits; // T2：当前已冻块位图（1 位/块），补洞票的块冻结态判据
        private ComputeBuffer _chunkMaturity;
        private ComputeBuffer _dummyShellWitnessEpochs;
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

        // 置信度通道 v1 统计（分歧 EMA 三档普查）
        private ComputeKernelHelper _confidenceKernel;
        private ComputeBuffer _confidenceStats;
        private bool _confidenceReadbackPending;
        /// <summary>最近一轮置信度普查：有数据体素数（abs(weight)≥出网门槛，冻结 abs 后参与）。</summary>
        public int ConfidenceVoxelCount { get; private set; }
        /// <summary>高置信（分歧 EMA &lt; confidenceMidMax）。</summary>
        public int ConfidenceHighCount { get; private set; }
        /// <summary>中置信。</summary>
        public int ConfidenceMidCount { get; private set; }
        /// <summary>低置信（≥ confidenceLowMin，几何在打架）。</summary>
        public int ConfidenceLowCount { get; private set; }
        /// <summary>低置信中相干者（签名一致=真错/真变化嫌疑，相干闸该放行）。</summary>
        public int ConfidenceLowCoherentCount { get; private set; }
        /// <summary>低置信中纯噪声（方向横跳，相干闸拦得住的那类）。</summary>
        public int ConfidenceLowNoiseCount { get; private set; }

        // InfiniTAM baseline receipt. This is a throttled observation of the
        // production write, not another gate or authority path.
        private const int InfiniTamTicketStatCount = 4;
        private const float InfiniTamTicketIntervalSeconds = 1f;
        private static readonly uint[] ZeroInfiniTamTicketStats =
            new uint[InfiniTamTicketStatCount];
        private ComputeBuffer _infiniTamTicketStats;
        private bool _infiniTamTicketReadbackPending;
        private float _nextInfiniTamTicketTime;
        private int _infiniTamTicketGeneration;
        private bool _hasInfiniTamTicket;
        private uint _infiniTamTicketNew;
        private uint _infiniTamTicketContinuing;
        private uint _infiniTamTicketMature;
        private uint _infiniTamTicketVoteSum;
        private uint _infiniTamTicketSurfaceSamples;
        private ulong _infiniTamTicketCumulativeNew;
        private ulong _infiniTamTicketCumulativeContinuing;
        private ulong _infiniTamTicketCumulativeMature;
        private ulong _infiniTamTicketCumulativeVoteSum;
        private ulong _infiniTamTicketCumulativeSurfaceSamples;
        private int _infiniTamTicketSampleCount;
        private int _infiniTamAttemptedFrames;
        private int _infiniTamFusedFrames;

        // 矛盾票普查：诊断"幽灵抹不掉"——反对票到底投没投出、被哪道门禁拦住
        private ComputeBuffer _carveStats;
        private bool _carveStatsReadbackPending;
        private readonly List<Action<bool>> _carveStatsFlushCallbacks =
            new List<Action<bool>>(4);
        private ComputeBuffer _projectiveShadowCarveStats;
        private bool _projectiveShadowCarveStatsReadbackPending;
        private readonly List<Action<bool>> _projectiveShadowFlushCallbacks =
            new List<Action<bool>>(4);
        // 0..27: existing contradiction/supply/edge ledgers.
        // 28..44: read-only dilation-donor provenance ledger.
        // 45..49: read-only donor-path classification ledger.
        // 50..59: direct-fill versus relay provenance ledger.
        // 60..65: production dilation-gate reason counters.
        // 66..67: actual seed/growth writes blocked by that gate.
        // 68..70: provisional seeds, promotions and formal-surface demotions.
        // 71..89: read-only lifecycle forensics: seed source/risk, promotion
        // mechanism, promotion-time risk and immutable birth source.
        private const int CarveStatsCount = 199; // 180..197=写入生命周期；198=未成熟冻结恢复（重叠账）
        private static readonly uint[] ZeroCarveStats = new uint[CarveStatsCount];
        /// <summary>最近一个统计周期的矛盾票计数：0票投出 1排除区拦 2法线闸拦 3遮挡闸拦 4带外拦 5排内抹（不对称放行实际扣减）。</summary>
        public readonly uint[] LastCarveStats = new uint[CarveStatsCount];
        public readonly uint[] LastProjectiveShadowCarveStats = new uint[CarveStatsCount];
        public readonly ulong[] CumulativeProjectiveShadowCarveStats =
            new ulong[CarveStatsCount];
        public readonly ulong[] CumulativeCarveStats = new ulong[CarveStatsCount];
        private sealed class FovLedgerPeriod
        {
            public int Index;
            public DateTime Utc;
            public float ElapsedSeconds;
            public uint[] Counters;
        }
        private readonly List<FovLedgerPeriod> _fovLedgerPeriods = new List<FovLedgerPeriod>(256);
        private float _fovLedgerStartedRealtime;
        private int _fovLedgerPeriodIndex;
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
        // Eight centred half-voxel phases.  The per-frame budget is unchanged;
        // over eight accepted integrations the camera-local lattice no longer
        // keeps selecting the same subset of the world-aligned TSDF lattice.
        private static readonly Vector3[] FrustumCoveragePhases =
        {
            new Vector3(-0.25f, -0.25f, -0.25f),
            new Vector3( 0.25f,  0.25f, -0.25f),
            new Vector3( 0.25f, -0.25f,  0.25f),
            new Vector3(-0.25f,  0.25f,  0.25f),
            new Vector3( 0.25f, -0.25f, -0.25f),
            new Vector3(-0.25f,  0.25f, -0.25f),
            new Vector3(-0.25f, -0.25f,  0.25f),
            new Vector3( 0.25f,  0.25f,  0.25f)
        };
        public int WarmupIntegrations => warmupIntegrations;

        /// <summary>Raised after each integration compute dispatch (before pruning).</summary>
        public event Action Integrated;
        /// <summary>Raised after the volume is cleared.</summary>
        public event Action Cleared;
        /// <summary>Raised when relocation/load invalidates every persistent mesh chunk.</summary>
        public event Action TopologyInvalidated;

        public ComputeBuffer DirtyChunkEpochs => _dirtyChunkEpochs;

        internal bool TryGetGunGelPaperAuthority(
            out GunGelEvidenceShadow.PaperAuthorityBuffers authority)
        {
            authority = default;
            return enableGunGelEvidenceShadow && _gunGelEvidenceShadow != null &&
                _gunGelEvidenceShadow.TryGetPaperAuthority(out authority);
        }

        /// <summary>枪胶裁决海面的只读 GPU 缓冲；显示层不能写回生产链。</summary>
        internal bool TryGetGunGelCourtBuffers(
            out GunGelEvidenceShadow.CourtBuffers court)
        {
            court = default;
            return enableGunGelEvidenceShadow && _gunGelEvidenceShadow != null &&
                _gunGelEvidenceShadow.TryGetCourtBuffers(out court);
        }

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

        /// <summary>Latest exact 32^3 HERA page owner/boundary epochs.</summary>
        public uint[] LatestActivePageEpochs { get; private set; }
        public uint[] LatestActivePageObservedEpochs { get; private set; }
        public uint[] LatestActivePageBoundaryEpochs { get; private set; }
        private bool _activePageEpochReadbackPending;
        private bool _activePageObservedEpochReadbackPending;
        private bool _activePageBoundaryReadbackPending;

        /// <summary>
        /// Refresh the small 32^3 active-page liveness ledgers.  These buffers
        /// drive mesh scheduling only; they never participate in TSDF admission,
        /// freezing or surface classification.
        /// </summary>
        public void RequestActivePageEpochs()
        {
            if (!_activePageEpochReadbackPending && _activePageEpochs != null)
            {
                _activePageEpochReadbackPending = true;
                AsyncGPUReadback.Request(_activePageEpochs, req =>
                {
                    _activePageEpochReadbackPending = false;
                    if (req.hasError) return;
                    var data = req.GetData<uint>();
                    if (LatestActivePageEpochs == null || LatestActivePageEpochs.Length != data.Length)
                        LatestActivePageEpochs = new uint[data.Length];
                    data.CopyTo(LatestActivePageEpochs);
                });
            }
            if (!_activePageObservedEpochReadbackPending && _activePageObservedEpochs != null)
            {
                _activePageObservedEpochReadbackPending = true;
                AsyncGPUReadback.Request(_activePageObservedEpochs, req =>
                {
                    _activePageObservedEpochReadbackPending = false;
                    if (req.hasError) return;
                    var data = req.GetData<uint>();
                    if (LatestActivePageObservedEpochs == null || LatestActivePageObservedEpochs.Length != data.Length)
                        LatestActivePageObservedEpochs = new uint[data.Length];
                    data.CopyTo(LatestActivePageObservedEpochs);
                });
            }
            if (!_activePageBoundaryReadbackPending && _activePageBoundaryEpochs != null)
            {
                _activePageBoundaryReadbackPending = true;
                AsyncGPUReadback.Request(_activePageBoundaryEpochs, req =>
                {
                    _activePageBoundaryReadbackPending = false;
                    if (req.hasError) return;
                    var data = req.GetData<uint>();
                    if (LatestActivePageBoundaryEpochs == null || LatestActivePageBoundaryEpochs.Length != data.Length)
                        LatestActivePageBoundaryEpochs = new uint[data.Length];
                    data.CopyTo(LatestActivePageBoundaryEpochs);
                });
            }
        }

        private Texture _pendingCamFrame;
        private Vector3 _pendingCamPos;
        private Quaternion _pendingCamRot;
        private Vector2 _pendingFocalLen;
        private Vector2 _pendingPrincipalPt;
        private Vector2 _pendingSensorRes;
        private Vector2 _pendingCurrentRes;
        private RenderTexture _camFrameCopy;
        private Texture2D _dummyCamTex;

        private const int GunGelDeferredSlotCount = 3;

        private sealed class InfiniTamDeferredFrame
        {
            public RenderTexture Depth;
            public RenderTexture Normal;
            public RenderTexture DilatedDepth;
            public RenderTexture EdgeReason;
            public RenderTexture TemporalReason;
            public Matrix4x4[] View;
            public Matrix4x4[] Projection;
            public Matrix4x4[] ViewInverse;
            public Matrix4x4[] ProjectionInverse;
            public readonly Vector4[] ExclusionPositions = new Vector4[64];
            public int ExclusionCount;
            public int PlatformFrame;
            public int Generation;
            public float AngularSpeed;
            public float LinearSpeed;
            public float MotionQuality;
            public bool Pending;
            public bool Ready;
            public InfiniTamModelRaycastAudit.TrackingDecision Decision;
        }

        private readonly InfiniTamDeferredFrame _infiniTamDeferredFrame = new();
        private enum InfiniTamStartupPhase
        {
            AwaitingStillness,
            Seeding,
            Verifying,
            Tracking,
            TrackingLost
        }

        private int _infiniTamDeferredGeneration;
        private int _infiniTamLastQueuedPlatformFrame = -1;
        private int _infiniTamLastStartupPlatformFrame = -1;
        private int _infiniTamBootstrapStableFrames;
        private int _infiniTamBootstrapConfirmedFrames;
        private int _infiniTamRecoveryConfirmedFrames;
        private int _infiniTamConsecutiveTrackingRejects;
        private int _infiniTamBootstrapReseedCount;
        private bool _infiniTamTrackingInitialised;
        private InfiniTamStartupPhase _infiniTamStartupPhase =
            InfiniTamStartupPhase.AwaitingStillness;
        private int _infiniTamTrackingAccepted;
        private int _infiniTamTrackingRejected;
        private int _infiniTamTrackingQueueAbstained;
        private string _infiniTamLastTrackingDecision = "建模";

        private sealed class GunGelDeferredFrame
        {
            public RenderTexture RawDepth;
            public RenderTexture Depth;
            public RenderTexture Normal;
            public RenderTexture DilatedDepth;
            public RenderTexture EdgeReason;
            public RenderTexture TemporalReason;
            public Matrix4x4[] View;
            public Matrix4x4[] Projection;
            public Matrix4x4[] ViewInverse;
            public Matrix4x4[] ProjectionInverse;
            public readonly Vector4[] ExclusionPositions = new Vector4[64];
            public int ExclusionCount;
            public int FrameIndex;
            public int PlatformFrame;
            public int Generation;
            public float AngularSpeed;
            public float LinearSpeed;
            public float MotionQuality;
            public bool Pending;
            public bool Ready;
            public GunGelEvidenceShadow.FrameDecision Decision;
        }

        /// <summary>
        /// One lightweight, in-memory evidence transfer for the post-TSDF
        /// product court.  It reads only the correspondence and stable-id
        /// buffers already produced by GunGel; no depth/color texture or file
        /// capture is involved.
        /// </summary>
        private sealed class ProductCourtReadback
        {
            public int Generation;
            public int GunGelFrame;
            public int SourceFrame;
            public int AttemptIndex;
            public Matrix4x4 ViewInverse;
            public Matrix4x4 FusionCorrection;
            public float AngularSpeed;
            public float LinearSpeed;
            public float MotionQuality;
            public Vector3 HeadEuler;
            public GunGelEvidenceShadow.Correspondence[] Correspondences;
            public uint4[] Identities;
            public bool CorrespondencesDone;
            public bool IdentitiesDone;
            public bool Failed;
        }

        // 运动闸：角速度镜像自 DepthCapture.SmoothedDepthAngularSpeed（深度帧事件内、
        // 原始 Pose 四元数、真实帧间隔计算），供 HUD 读数与运动闸共用。
        private float _smoothedAngSpeed;
        private int _motionGatedSinceStats;
        private int _lastMotionGatedCount;
        /// <summary>平滑后的深度位姿角速度（°/s），调试用。</summary>
        public float SmoothedAngularSpeed => _smoothedAngSpeed;
        /// <summary>当前运动质量分（1=静止满权，地板=运动降权到底），HUD 回显用。</summary>
        public float MotionQuality => _motionQuality;
        private float _motionQuality = 1f;
        private GunGelEvidenceShadow _gunGelEvidenceShadow;
        private readonly RuntimeFinalSurfaceCourt _productSurfaceCourt =
            new RuntimeFinalSurfaceCourt();
        private ProductCourtReadback _productCourtReadback;
        private int _productCourtGeneration;
        private ComputeBuffer _gunGelDummyObservations;
        private ComputeBuffer _gunGelDummyCorrespondences;
        private ComputeBuffer _gunGelDummyCorrespondenceIdentity;
        private ComputeBuffer _finalCourtDummyVerdicts;
        private ComputeBuffer _finalCourtDummyPlanes;
        private ComputeBuffer _finalCourtDummyGenerations;
        private const int FinalCourtInvalidationCapacity = 64;
        private ComputeBuffer _finalCourtInvalidationArgs;
        private ComputeBuffer _finalCourtInvalidationRegions;
        private readonly Vector4[] _finalCourtInvalidationStaging =
            new Vector4[FinalCourtInvalidationCapacity];
        private readonly uint[] _finalCourtInvalidationCount = new uint[1];
        private bool _gunGelRuntimeFailureReported;
        private readonly GunGelDeferredFrame[] _gunGelDeferredFrames =
            new GunGelDeferredFrame[GunGelDeferredSlotCount];
        private int _gunGelDeferredGeneration;
        private int _gunGelCaptureFrameIndex;
        private int _gunGelFusionAccepted;
        private int _gunGelFusionRejected;
        private int _gunGelFusionRawFallback;
        private int _gunGelFusionQueueAbstained;
        private float _gunGelLastAppliedMm;
        private string _gunGelLastFusionDecision = "预热";
        private bool _gunGelGuardedFusionRuntimeHalted;
        private int _replayFusionAttemptIndex;

        /// <summary>
        /// 当前唯一生产路线仍依赖枪胶上游。只读暴露给扫描器，用于把
        /// “裁冻”身份钉进 HUD 与导出；不参与融合判决。
        /// </summary>
        public bool GunGelGuardedFusionExperimentEnabled =>
            enableGunGelGuardedFusionExperiment;
        public bool FinalCourtAdmissionExperimentEnabled =>
            enableFinalCourtAdmissionExperiment;
        public bool GunGelIdentityOnlyMode =>
            enableGunGelGuardedFusionExperiment &&
            !enableGunGelTsdfAdmission &&
            !enableGunGelPoseCorrection &&
            !enableFinalCourtAdmissionExperiment;

        internal int CopyProductSurfacePlanes(Bounds worldBounds,
            RuntimeFinalSurfaceCourt.ProductPlane[] destination) =>
            _productSurfaceCourt.CopyProductPlanes(worldBounds, destination);

        internal int DrainProductSurfaceChanges(Vector4[] destination) =>
            _productSurfaceCourt.DrainInvalidationRegions(destination);

        public string GetGunGelEvidenceShadowCompact()
        {
            if (!enableGunGelEvidenceShadow) return "关";
            if (_gunGelRuntimeFailureReported)
                return enableFinalCourtAdmissionExperiment
                    ? "已熔断（裁冻停笔）"
                    : "已熔断（生产融合正常）";
            string fusion;
            if (!enableGunGelGuardedFusionExperiment)
                fusion = "融基线";
            else if (enableFinalCourtAdmissionExperiment)
            {
                ScanReplaySessionPackage session = ScanReplaySessionPackage.Active;
                VirtualProbeShadowAdjudicator.VerdictCounts counts = session != null
                    ? session.FinalCourtAdmissionCounts : default;
                fusion = (_gunGelGuardedFusionRuntimeHalted
                        ? "裁入熔断停笔 "
                        : "裁入 ") +
                         $"待{counts.Hold}准{counts.Accept}弃{counts.Reject}" +
                         $" 版{(session != null ? session.FinalCourtAdmissionRevision : -1)}" +
                         $" 队{CountGunGelDeferredFrames()}末{_gunGelLastFusionDecision}";
            }
            else if (_gunGelGuardedFusionRuntimeHalted)
                fusion = "融试熔断→基线";
            else
                fusion = $"融试校{_gunGelFusionAccepted}拒校{_gunGelFusionRejected}" +
                         $"原{_gunGelFusionRawFallback}" +
                         $"队{CountGunGelDeferredFrames()}失{_gunGelFusionQueueAbstained}" +
                         $"末{_gunGelLastFusionDecision}{_gunGelLastAppliedMm:F1}mm";
            string shadow = _gunGelEvidenceShadow != null
                ? _gunGelEvidenceShadow.GetCompact()
                : "未就绪";
            return fusion + " · " + shadow;
        }

        /// <summary>
        /// 在扫描冻结点只读封存枪胶候选账本；不等待 Unity 编辑器，也不改变
        /// TSDF、候选裁决或显示状态。返回 false 表示影子层未就绪/已有导出在途。
        /// </summary>
        public bool RequestGunGelCandidateAuditExport(string reason,
            Action<string> completed = null)
        {
            if (_gunGelEvidenceShadow == null || !enableGunGelEvidenceShadow ||
                _gunGelEvidenceShadow.AuditExportPending)
                return false;
            StartCoroutine(_gunGelEvidenceShadow.ExportAuditAsync(reason, completed));
            return true;
        }

        /// <summary>
        /// Enables the full-output-ledger TSDF transaction sidecar before the first
        /// integration.  The sidecar is never read by production code.
        /// </summary>
        internal bool BeginTsdfResponsibilityCapture()
        {
            if (_volume == null || _tsdfResponsibilityVolume == null ||
                !_tsdfResponsibilityVolume.IsCreated() ||
                _tsdfSupportResponsibilityVolume == null ||
                !_tsdfSupportResponsibilityVolume.IsCreated() || IntegrationCount != 0)
                return false;
            _tsdfResponsibilityCaptureEnabled = true;
            compute.SetFloat(TsdfResponsibilityWriteID, 1f);
            return true;
        }

        /// <summary>
        /// One-shot stop-time export. The normalized TSDF is read one Z slice per
        /// texture request. Integer responsibility textures first copy one slice
        /// through a structured buffer because Quest/Vulkan rejects their direct
        /// 3D readback. Each completed lane is persisted before the next begins.
        /// </summary>
        internal bool RequestTsdfResponsibilityAuditExport(
            string reason, string destinationDirectory, Action<string> completed)
        {
            if (!_tsdfResponsibilityCaptureEnabled || _volume == null ||
                _tsdfResponsibilityVolume == null ||
                _tsdfSupportResponsibilityVolume == null ||
                string.IsNullOrEmpty(destinationDirectory) ||
                _tsdfResponsibilityExportPending)
                return false;
            _tsdfResponsibilityExportPending = true;
            LastTsdfResponsibilityExportError = string.Empty;
            StartCoroutine(ExportTsdfResponsibilitySlabbed(
                reason ?? string.Empty, destinationDirectory, completed));
            return true;
        }

        private IEnumerator ExportTsdfResponsibilitySlabbed(
            string reason, string destinationDirectory, Action<string> completed)
        {
            string failure = string.Empty;
            var completedResources = new List<string>(3);
            try { Directory.CreateDirectory(destinationDirectory); }
            catch (Exception e) { failure = "directory:" + e.Message; }

            if (string.IsNullOrEmpty(failure))
            {
                byte[] tsdf = null;
                yield return ReadTextureSlabs(_volume, 2, "final_tsdf",
                    data => tsdf = data, error => failure = error);
                if (string.IsNullOrEmpty(failure))
                    yield return WriteResponsibilityBytes(
                        Path.Combine(destinationDirectory, "final_tsdf.rg8_snorm.bin"),
                        tsdf, "final_tsdf", error => failure = error);
                if (string.IsNullOrEmpty(failure))
                {
                    completedResources.Add("final_tsdf");
                    failure = WriteResponsibilityCheckpoint(destinationDirectory,
                        completedResources, false, string.Empty);
                }
            }

            if (string.IsNullOrEmpty(failure))
            {
                byte[] responsibility = null;
                yield return ReadIntegerResponsibilitySlices(
                    _tsdfResponsibilityVolume, 2,
                    "geometry_block_responsibility", data => responsibility = data,
                    error => failure = error);
                if (string.IsNullOrEmpty(failure))
                    yield return WriteResponsibilityBytes(
                        Path.Combine(destinationDirectory,
                            "responsibility.rg32_uint.bin"), responsibility,
                        "geometry_block_responsibility", error => failure = error);
                if (string.IsNullOrEmpty(failure))
                {
                    completedResources.Add("geometry_block_responsibility");
                    failure = WriteResponsibilityCheckpoint(destinationDirectory,
                        completedResources, false, string.Empty);
                }
            }

            if (string.IsNullOrEmpty(failure))
            {
                byte[] support = null;
                yield return ReadIntegerResponsibilitySlices(
                    _tsdfSupportResponsibilityVolume, 1,
                    "support_responsibility", data => support = data,
                    error => failure = error);
                if (string.IsNullOrEmpty(failure))
                    yield return WriteResponsibilityBytes(
                        Path.Combine(destinationDirectory,
                            "support_responsibility.r32_uint.bin"), support,
                        "support_responsibility", error => failure = error);
                if (string.IsNullOrEmpty(failure))
                {
                    completedResources.Add("support_responsibility");
                    failure = WriteResponsibilityCheckpoint(destinationDirectory,
                        completedResources, false, string.Empty);
                }
            }

            if (string.IsNullOrEmpty(failure))
            {
                int3 capturedVoxelCount = voxelCount;
                float capturedVoxelSize = voxelSize;
                float capturedTruncationDistance = voxelDistance;
                Task schemaTask = null;
                try
                {
                    schemaTask = Task.Run(() => WriteTsdfResponsibilitySchema(
                        reason, destinationDirectory, capturedVoxelCount,
                        capturedVoxelSize, capturedTruncationDistance));
                }
                catch (Exception e) { failure = "schema:request:" + e.Message; }
                if (schemaTask != null)
                {
                    while (!schemaTask.IsCompleted) yield return null;
                    if (schemaTask.IsFaulted)
                        failure = "schema:write:" +
                            (schemaTask.Exception?.GetBaseException().Message ?? "unknown");
                    else if (schemaTask.IsCanceled)
                        failure = "schema:write:canceled";
                }
                if (string.IsNullOrEmpty(failure))
                    failure = WriteResponsibilityCheckpoint(destinationDirectory,
                        completedResources, true, string.Empty);
            }

            LastTsdfResponsibilityExportError = failure;
            if (!string.IsNullOrEmpty(failure))
            {
                try
                {
                    Directory.CreateDirectory(destinationDirectory);
                    File.WriteAllText(Path.Combine(destinationDirectory, "failure.json"),
                        "{\n  \"schema\": \"scancover.tsdf_responsibility_failure.v1\",\n" +
                        "  \"issue\": \"" + JsonEscape(failure) + "\",\n" +
                        "  \"completedResources\": [" +
                        JoinJsonStrings(completedResources) + "]\n}\n",
                        new UTF8Encoding(false));
                    WriteResponsibilityCheckpoint(destinationDirectory,
                        completedResources, false, failure);
                }
                catch { }
                Logger.Warning("TSDF责任链导出失败：" + failure);
            }

            _tsdfResponsibilityExportPending = false;
            try { completed?.Invoke(string.IsNullOrEmpty(failure) ? destinationDirectory : string.Empty); }
            catch (Exception e) { Logger.Warning("TSDF责任链回调失败：" + e.Message); }
        }

        private IEnumerator ReadTextureSlabs(RenderTexture texture, int bytesPerVoxel,
            string label, Action<byte[]> completed, Action<string> failed)
        {
            // Quest/Vulkan returns width*height only for this 3D RenderTexture
            // region overload even when depth > 1.  Keep every request at one
            // slice; the previous depth=8 assumption produced exactly 1/8 of
            // the expected bytes and made every otherwise-valid session seal
            // incomplete.
            const int slabDepth = 1;
            int rowBytes = voxelCount.x * voxelCount.y * bytesPerVoxel;
            byte[] output = new byte[rowBytes * voxelCount.z];
            for (int z = 0; z < voxelCount.z; z += slabDepth)
            {
                int depth = Mathf.Min(slabDepth, voxelCount.z - z);
                AsyncGPUReadbackRequest request = default;
                string requestFailure = string.Empty;
                try
                {
                    request = AsyncGPUReadback.Request(texture, 0,
                        0, voxelCount.x, 0, voxelCount.y, z, depth);
                }
                catch (Exception e)
                {
                    requestFailure = $"{label}:request:z={z}:depth={depth}:{e.Message}";
                }
                if (!string.IsNullOrEmpty(requestFailure))
                {
                    failed?.Invoke(requestFailure);
                    yield break;
                }
                while (!request.done) yield return null;
                if (request.hasError)
                {
                    failed?.Invoke($"{label}:gpu:z={z}:depth={depth}");
                    yield break;
                }
                string copyFailure = string.Empty;
                try
                {
                    byte[] slab = request.GetData<byte>().ToArray();
                    int expected = rowBytes * depth;
                    if (slab.Length != expected)
                        copyFailure = $"{label}:length:z={z}:expected={expected}:actual={slab.Length}";
                    else
                        Buffer.BlockCopy(slab, 0, output, rowBytes * z, expected);
                }
                catch (Exception e)
                {
                    copyFailure = $"{label}:copy:z={z}:depth={depth}:{e.Message}";
                }
                if (!string.IsNullOrEmpty(copyFailure))
                {
                    failed?.Invoke(copyFailure);
                    yield break;
                }
                // Request completion is already the retirement barrier.  Do not
                // add another unconditional frame for each of the 576 typical
                // stop-time slices.
            }
            completed?.Invoke(output);
        }

        private IEnumerator ReadIntegerResponsibilitySlices(
            RenderTexture texture, int uintLanes, string label,
            Action<byte[]> completed, Action<string> failed)
        {
            if (uintLanes != 1 && uintLanes != 2)
            {
                failed?.Invoke($"{label}:invalid_uint_lanes:{uintLanes}");
                yield break;
            }

            int sliceValues = voxelCount.x * voxelCount.y;
            int sliceBytes = sliceValues * sizeof(uint) * uintLanes;
            byte[] output = new byte[sliceBytes * voxelCount.z];
            ComputeBuffer staging = null;
            int kernel = -1;
            string setupFailure = string.Empty;
            try
            {
                staging = new ComputeBuffer(sliceValues,
                    sizeof(uint) * uintLanes, ComputeBufferType.Structured);
                kernel = compute.FindKernel(uintLanes == 2
                    ? "CopyTsdfResponsibilityExportSlice"
                    : "CopyTsdfSupportResponsibilityExportSlice");
                compute.SetInts(VoxCountID, voxelCount.x, voxelCount.y, voxelCount.z);
                compute.SetTexture(kernel,
                    uintLanes == 2 ? TsdfResponsibilityRWID :
                        TsdfSupportResponsibilityRWID, texture);
                compute.SetBuffer(kernel,
                    uintLanes == 2 ? TsdfResponsibilityExportSliceID :
                        TsdfSupportResponsibilityExportSliceID, staging);
            }
            catch (Exception e)
            {
                setupFailure = $"{label}:staging_setup:{e.Message}";
            }
            if (!string.IsNullOrEmpty(setupFailure))
            {
                staging?.Release();
                failed?.Invoke(setupFailure);
                yield break;
            }

            for (int z = 0; z < voxelCount.z; z++)
            {
                string dispatchFailure = string.Empty;
                try
                {
                    compute.SetInt(TsdfResponsibilityExportZID, z);
                    compute.Dispatch(kernel,
                        Mathf.CeilToInt(voxelCount.x / 8f),
                        Mathf.CeilToInt(voxelCount.y / 8f), 1);
                }
                catch (Exception e)
                {
                    dispatchFailure = $"{label}:staging_dispatch:z={z}:{e.Message}";
                }
                if (!string.IsNullOrEmpty(dispatchFailure))
                {
                    staging.Release();
                    failed?.Invoke(dispatchFailure);
                    yield break;
                }

                AsyncGPUReadbackRequest request = default;
                string requestFailure = string.Empty;
                try { request = AsyncGPUReadback.Request(staging); }
                catch (Exception e)
                {
                    requestFailure = $"{label}:staging_request:z={z}:{e.Message}";
                }
                if (!string.IsNullOrEmpty(requestFailure))
                {
                    staging.Release();
                    failed?.Invoke(requestFailure);
                    yield break;
                }

                while (!request.done) yield return null;
                if (request.hasError)
                {
                    staging.Release();
                    failed?.Invoke($"{label}:staging_gpu:z={z}");
                    yield break;
                }

                string copyFailure = string.Empty;
                try
                {
                    byte[] slice = request.GetData<byte>().ToArray();
                    if (slice.Length != sliceBytes)
                        copyFailure = $"{label}:staging_length:z={z}:expected={sliceBytes}:actual={slice.Length}";
                    else
                        Buffer.BlockCopy(slice, 0, output, sliceBytes * z, sliceBytes);
                }
                catch (Exception e)
                {
                    copyFailure = $"{label}:staging_copy:z={z}:{e.Message}";
                }
                if (!string.IsNullOrEmpty(copyFailure))
                {
                    staging.Release();
                    failed?.Invoke(copyFailure);
                    yield break;
                }
            }

            staging.Release();
            completed?.Invoke(output);
        }

        private IEnumerator WriteResponsibilityBytes(string path, byte[] data,
            string label, Action<string> failed)
        {
            if (data == null)
            {
                failed?.Invoke(label + ":write:null_data");
                yield break;
            }
            Task task = null;
            string requestFailure = string.Empty;
            try { task = Task.Run(() => File.WriteAllBytes(path, data)); }
            catch (Exception e) { requestFailure = label + ":write_request:" + e.Message; }
            if (!string.IsNullOrEmpty(requestFailure))
            {
                failed?.Invoke(requestFailure);
                yield break;
            }
            while (!task.IsCompleted) yield return null;
            if (task.IsFaulted)
                failed?.Invoke(label + ":write:" +
                    (task.Exception?.GetBaseException().Message ?? "unknown"));
            else if (task.IsCanceled)
                failed?.Invoke(label + ":write:canceled");
        }

        private static string WriteResponsibilityCheckpoint(string directory,
            List<string> completedResources, bool complete, string issue)
        {
            try
            {
                File.WriteAllText(Path.Combine(directory, "partial_status.json"),
                    "{\n  \"schema\": \"scancover.tsdf_responsibility_partial.v1\",\n" +
                    "  \"complete\": " + (complete ? "true" : "false") + ",\n" +
                    "  \"completedResources\": [" +
                    JoinJsonStrings(completedResources) + "],\n" +
                    "  \"issue\": \"" + JsonEscape(issue) + "\"\n}\n",
                    new UTF8Encoding(false));
                return string.Empty;
            }
            catch (Exception e)
            {
                return "checkpoint:write:" + e.Message;
            }
        }

        private static string JoinJsonStrings(List<string> values)
        {
            if (values == null || values.Count == 0) return string.Empty;
            var builder = new StringBuilder(values.Count * 32);
            for (int i = 0; i < values.Count; i++)
            {
                if (i > 0) builder.Append(',');
                builder.Append('"').Append(JsonEscape(values[i])).Append('"');
            }
            return builder.ToString();
        }

        private static void WriteTsdfResponsibilitySchema(string reason,
            string destinationDirectory,
            int3 capturedVoxelCount, float capturedVoxelSize,
            float capturedTruncationDistance)
        {
            Directory.CreateDirectory(destinationDirectory);
            string schema = "{\n" +
                "  \"schema\": \"scancover.tsdf_responsibility.v3\",\n" +
                "  \"reason\": \"" + JsonEscape(reason) + "\",\n" +
                "  \"voxelCount\": [" + capturedVoxelCount.x + "," + capturedVoxelCount.y + "," + capturedVoxelCount.z + "],\n" +
                "  \"voxelSizeMetres\": " + capturedVoxelSize.ToString("R", CultureInfo.InvariantCulture) + ",\n" +
                "  \"truncationDistanceMetres\": " + capturedTruncationDistance.ToString("R", CultureInfo.InvariantCulture) + ",\n" +
                "  \"readback\": \"final TSDF uses single-Z texture requests; integer responsibility lanes use compute-to-structured-buffer staging per Z slice\",\n" +
                "  \"checkpoint\": \"each completed resource is written immediately and listed in partial_status.json\",\n" +
                "  \"indexOrder\": \"x-fastest, then y, then z\",\n" +
                "  \"tsdf\": \"interleaved signed R8 distance and signed R8 weight\",\n" +
                "  \"responsibilityLaneX\": \"seedIntegration12,lastGeometryIntegration12,lastOperation4,birthFlags4\",\n" +
                "  \"responsibilityLaneY\": \"strongestBlockIntegration12,blockReason5,signedMagnitude7,lastPreTsdf8\",\n" +
                "  \"supportResponsibility\": \"integration12,operation4,preWeightSNorm8,postWeightSNorm8\",\n" +
                "  \"integrationIdRange\": \"1..4094 exact; 4095 means saturated at or beyond integration 4095\",\n" +
                "  \"operationCodes\": {\"1\":\"seed\",\"2\":\"positive_blend\",\"3\":\"lifetime_reset\"},\n" +
                "  \"supportOperationCodes\": {\"1\":\"seed\",\"2\":\"positive_growth\",\"3\":\"carve\",\"4\":\"freeze\",\"5\":\"unfreeze\",\"6\":\"reset\"},\n" +
                "  \"blockReasonCodes\": {\"1\":\"gungel\",\"2\":\"exclusion\",\"3\":\"normal\",\"4\":\"dilation_occlusion\",\"5\":\"truncation_band\",\"6\":\"abstain_neighbour\",\"9\":\"raw_projective_gate\",\"10\":\"dilation_supply\",\"11\":\"fov_motion_authority\",\"12\":\"frozen\",\"13\":\"blend_quantized\",\"14\":\"mature_discount\",\"15\":\"near_distance\",\"16\":\"invalid_depth\",\"17\":\"seed_quality\",\"18\":\"motion_seed\",\"19\":\"behind_camera\",\"20\":\"outside_fov\"},\n" +
                "  \"join\": \"packed integration ids join runtime_timeline/integration_dispatches.csv integrationCount\",\n" +
                "  \"authority\": \"diagnostic only; never sampled by production\"\n" +
                "}\n";
            File.WriteAllText(Path.Combine(destinationDirectory, "schema.json"),
                schema, new UTF8Encoding(false));
        }

        private static string JsonEscape(string value)
        {
            return (value ?? string.Empty).Replace("\\", "\\\\")
                .Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
        }

        /// <summary>
        /// A 键请求枪胶裁决层停止收新帧并等待已提交裁决排空。完成后稳定候选
        /// 与锁存浪头保留在原 GPU 缓冲，直到 B 清卷或销毁。
        /// </summary>
        public bool RequestGunGelCourtSeal(Action<bool> completed = null)
        {
            if (_gunGelEvidenceShadow == null || !enableGunGelEvidenceShadow ||
                _gunGelEvidenceShadow.CourtSealPending)
                return false;
            StartCoroutine(_gunGelEvidenceShadow.SealCourtAsync(completed));
            return true;
        }

        public string ToggleGunGelGuardedFusionExperiment()
        {
            if (IntegrationCount > 0 || CountGunGelDeferredFrames() > 0)
                return "融合模式:需空卷/重启后切";
            // 新生产制度只有一个 TSDF。GunGel 保持运行以提供 stableId 与
            // 候选证词，但不得修正位姿、逐点否决 TSDF 或用裁判平面改零面。
            enableGunGelGuardedFusionExperiment = true;
            enableGunGelTsdfAdmission = false;
            enableGunGelPoseCorrection = false;
            enableFinalCourtAdmissionExperiment = false;
            _gunGelGuardedFusionRuntimeHalted = false;
            _gunGelLastFusionDecision = "预热";
            return "单TSDF+网格后裁决:唯一生产路线";
        }

        private void Awake()
        {
            Instance = this;
            // Scene/Prefab 中遗留的裁冻值不得重新把产品裁判接回 TSDF。
            // GunGel 流水线只生产同帧身份/候选证据；距离、位姿和唯一 TSDF
            // 保持原始生产含义，裁判平面只在候选网格产品化阶段消费。
            enableGunGelGuardedFusionExperiment = !enableInfiniTamBaseline;
            enableGunGelTsdfAdmission = false;
            enableGunGelPoseCorrection = false;
            enableFinalCourtAdmissionExperiment = false;
            if (!enableInfiniTamBaseline)
                _productSurfaceCourt.BeginInMemory();
            for (int i = 0; i < _gunGelDeferredFrames.Length; i++)
                _gunGelDeferredFrames[i] = new GunGelDeferredFrame();
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
            _clearKernel = new ComputeKernelHelper(compute,
                enableInfiniTamBaseline ? "ClearInfiniTamVolume" : "Clear");
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            if (!enableInfiniTamBaseline)
            {
                _clearKernel.Set(TsdfResponsibilityRWID,
                    _tsdfResponsibilityVolume);
                _clearKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
            }

            _clearInfiniTamVotesKernel = new ComputeKernelHelper(compute,
                "ClearInfiniTamVotes");
            _clearInfiniTamVotesKernel.Set(InfiniTamVoteWeightRWID,
                _infiniTamVoteWeightVolume);

            if (!enableInfiniTamBaseline)
            {
                _invalidateGunGelSuccessionsKernel = new ComputeKernelHelper(compute,
                    "InvalidateGunGelSuccessions");
                _invalidateGunGelSuccessionsKernel.Set(VolumeRWID, _volume);
                _invalidateGunGelSuccessionsKernel.Set(ColorVolumeRWID, _colorVolume);
                _invalidateGunGelSuccessionsKernel.Set(AdmissionTraceRWID,
                    _admissionTraceVolume);
                _invalidateGunGelSuccessionsKernel.Set(TsdfResponsibilityRWID,
                    _tsdfResponsibilityVolume);
                _invalidateGunGelSuccessionsKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _invalidateGunGelSuccessionsKernel.Set(ConfidenceRWID, _confidenceVolume);
                _invalidateGunGelSuccessionsKernel.Set(CoherenceRWID, _coherenceVolume);
                _invalidateGunGelSuccessionsKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
                _invalidateGunGelSuccessionsKernel.Set(DirtyBoundaryEpochsID,
                    _dirtyBoundaryEpochs);
                _invalidateGunGelSuccessionsKernel.Set(ActivePageEpochsID,
                    _activePageEpochs);
                _invalidateGunGelSuccessionsKernel.Set(ActivePageObservedEpochsID,
                    _activePageObservedEpochs);
                _invalidateGunGelSuccessionsKernel.Set(ActivePageBoundaryEpochsID,
                    _activePageBoundaryEpochs);
            }

            _integrateKernel = new ComputeKernelHelper(compute,
                enableInfiniTamBaseline ? "IntegrateInfiniTam" : "Integrate");
            if (enableInfiniTamBaseline)
                Logger.Info("VolumeIntegrator: compact InfiniTAM fusion active " +
                            "(6 UAV, depth-only input, no camera RGB blit).");
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(InfiniTamVoteWeightRWID, _infiniTamVoteWeightVolume);
            _integrateKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
            _integrateKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);
            if (!enableInfiniTamBaseline)
            {
                _integrateKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
                _integrateKernel.Set(TsdfResponsibilityRWID, _tsdfResponsibilityVolume);
                _integrateKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _integrateKernel.Set(ActivePageEpochsID, _activePageEpochs);
                _integrateKernel.Set(ActivePageObservedEpochsID,
                    _activePageObservedEpochs);
                _integrateKernel.Set(ActivePageBoundaryEpochsID,
                    _activePageBoundaryEpochs);

                _pruneKernel = new ComputeKernelHelper(compute, "Prune");
                _pruneKernel.Set(VolumeRWID, _volume);
                _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
                _pruneKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
                _pruneKernel.Set(TsdfResponsibilityRWID, _tsdfResponsibilityVolume);
                _pruneKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _pruneKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
                _pruneKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);
                _pruneKernel.Set(ActivePageEpochsID, _activePageEpochs);
                _pruneKernel.Set(ActivePageObservedEpochsID,
                    _activePageObservedEpochs);
                _pruneKernel.Set(ActivePageBoundaryEpochsID,
                    _activePageBoundaryEpochs);
            }

            if (!enableInfiniTamBaseline)
            {
                _freezeKernel = new ComputeKernelHelper(compute,
                    "FreezeInFrustum");
                _freezeKernel.Set(VolumeRWID, _volume);
                _freezeKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);

                _unfreezeKernel = new ComputeKernelHelper(compute,
                    "UnfreezeInFrustum");
                _unfreezeKernel.Set(VolumeRWID, _volume);
                _unfreezeKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);

                _applyFreezeMaskKernel = new ComputeKernelHelper(compute,
                    "ApplyChunkFreezeMask");
                _applyFreezeMaskKernel.Set(VolumeRWID, _volume);
                _applyFreezeMaskKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _applyFreezeMaskKernel.Set(ChunkFreezeSetMaskID,
                    _chunkFreezeSetMask);
                _applyFreezeMaskKernel.Set(ChunkFreezeClearMaskID,
                    _chunkFreezeClearMask);

                _clearVotesKernel = new ComputeKernelHelper(compute,
                    "ClearFrozenChunkVotes");
                _clearVotesKernel.Set(ChunkFreezeClearMaskID,
                    _chunkFreezeClearMask);
                _clearVotesKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);

                _maturityKernel = new ComputeKernelHelper(compute,
                    "CountChunkMaturity");
                _maturityKernel.Set(VolumeRWID, _volume);
                _maturityKernel.Set(ChunkMaturityID, _chunkMaturity);
                _maturityKernel.Set(ConfidenceRWID, _confidenceVolume);

                _integrateKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);
                _integrateKernel.Set(FrozenChunkBitsID, _frozenChunkBits);
            }

            _coverageKernel = new ComputeKernelHelper(compute, "CountSurfaceCoverage");
            _coverageKernel.Set(VolumeRWID, _volume);
            _coverageCounters = new ComputeBuffer(3, sizeof(uint));
            _coverageKernel.Set(CoverageCountersID, _coverageCounters);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);

            // 置信度通道 v1：分歧 EMA 体绑进 Integrate（写）与 Clear（清零），
            // CountConfidence 内核只读它做三档普查。缓冲建后先清零——新缓冲首帧
            // 按垃圾计数写会 GPU 挂死（粗皮缓冲同款老陷阱）。
            if (!enableInfiniTamBaseline)
            {
                _integrateKernel.Set(ConfidenceRWID, _confidenceVolume);
                _integrateKernel.Set(CoherenceRWID, _coherenceVolume);
            }
            _clearKernel.Set(ConfidenceRWID, _confidenceVolume);
            _clearKernel.Set(CoherenceRWID, _coherenceVolume);
            _confidenceKernel = new ComputeKernelHelper(compute, "CountConfidence");
            _confidenceKernel.Set(VolumeRWID, _volume);
            _confidenceKernel.Set(ConfidenceRWID, _confidenceVolume);
            _confidenceKernel.Set(CoherenceRWID, _coherenceVolume);
            _confidenceStats = new ComputeBuffer(6, sizeof(uint));
            _confidenceStats.SetData(new uint[6]);
            _confidenceKernel.Set(ConfidenceStatsID, _confidenceStats);

            _carveStats = new ComputeBuffer(CarveStatsCount, sizeof(uint));
            _carveStats.SetData(ZeroCarveStats);
            if (!enableInfiniTamBaseline)
                _integrateKernel.Set(CarveStatsID, _carveStats);

            // Vulkan requires a valid descriptor even when the legacy route
            // keeps the receipt switch at zero. The buffer is tiny (16 bytes)
            // and only the isolated baseline writes it.
            _infiniTamTicketStats = new ComputeBuffer(
                InfiniTamTicketStatCount, sizeof(uint));
            _infiniTamTicketStats.SetData(ZeroInfiniTamTicketStats);
            _integrateKernel.Set(InfiniTamTicketStatsID, _infiniTamTicketStats);
            compute.SetFloat(InfiniTamTicketEnabledID, 0f);

            // Integrate 内核始终声明枪胶 SRV；基线/B 影子虽然关闭准入，Vulkan
            // 描述符仍需有效绑定。四个 1 元素零缓冲只负责占位，不参与裁决。
            _gunGelDummyObservations = new ComputeBuffer(1, sizeof(float) * 12);
            _gunGelDummyCorrespondences = new ComputeBuffer(1, sizeof(float) * 12);
            _gunGelDummyCorrespondenceIdentity = new ComputeBuffer(1, sizeof(uint) * 4);
            _finalCourtDummyVerdicts = new ComputeBuffer(1, sizeof(uint));
            _finalCourtDummyPlanes = new ComputeBuffer(1, sizeof(float) * 4);
            _finalCourtDummyGenerations = new ComputeBuffer(1, sizeof(uint));
            _finalCourtInvalidationArgs = new ComputeBuffer(1, sizeof(uint));
            _finalCourtInvalidationRegions = new ComputeBuffer(
                FinalCourtInvalidationCapacity, sizeof(float) * 4);
            _gunGelDummyObservations.SetData(new float[12]);
            _gunGelDummyCorrespondences.SetData(new float[12]);
            _gunGelDummyCorrespondenceIdentity.SetData(new uint[4]);
            _finalCourtDummyVerdicts.SetData(new uint[1]);
            _finalCourtDummyPlanes.SetData(new Vector4[1]);
            _finalCourtDummyGenerations.SetData(new uint[1]);
            _finalCourtInvalidationArgs.SetData(new uint[1]);
            _finalCourtInvalidationRegions.SetData(
                new Vector4[FinalCourtInvalidationCapacity]);
            _dummyShellWitnessEpochs = new ComputeBuffer(1, sizeof(uint),
                ComputeBufferType.Structured);
            _dummyShellWitnessEpochs.SetData(new uint[1]);
            if (!enableInfiniTamBaseline)
                _integrateKernel.Set(ShellWitnessEpochsID, _dummyShellWitnessEpochs);

            if (enableProjectiveShadow && !enableInfiniTamBaseline)
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
            _productCourtGeneration++;
            _productCourtReadback = null;
            _productSurfaceCourt.Dispose();
            _coverageCounters?.Release();
            _coverageCounters = null;
            _carveStats?.Release();
            _carveStats = null;
            _projectiveShadowCarveStats?.Release();
            _projectiveShadowCarveStats = null;
            _confidenceStats?.Release();
            _confidenceStats = null;
            _infiniTamTicketGeneration++;
            _infiniTamTicketReadbackPending = false;
            _infiniTamTicketStats?.Release();
            _infiniTamTicketStats = null;
            _gunGelDummyObservations?.Release();
            _gunGelDummyObservations = null;
            _gunGelDummyCorrespondences?.Release();
            _gunGelDummyCorrespondences = null;
            _gunGelDummyCorrespondenceIdentity?.Release();
            _gunGelDummyCorrespondenceIdentity = null;
            _finalCourtDummyVerdicts?.Release();
            _finalCourtDummyVerdicts = null;
            _finalCourtDummyPlanes?.Release();
            _finalCourtDummyPlanes = null;
            _finalCourtDummyGenerations?.Release();
            _finalCourtDummyGenerations = null;
            _finalCourtInvalidationArgs?.Release();
            _finalCourtInvalidationArgs = null;
            _finalCourtInvalidationRegions?.Release();
            _finalCourtInvalidationRegions = null;
            _dummyShellWitnessEpochs?.Release();
            _dummyShellWitnessEpochs = null;
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
            _tsdfResponsibilityExportPending = false;
            _infiniTamModelRaycast?.Dispose();
            _infiniTamModelRaycast = null;
            ResetInfiniTamDeferredFrame(true);
            ResetProductSurfaceCourt();
            ResetGunGelDeferredFrames(true);
            _gunGelEvidenceShadow?.Dispose();
            _gunGelEvidenceShadow = null;
            _frustumVolume?.Release();
            _frustumVolume = null;
            _frustumReady = false;
            _dirtyChunkEpochs?.Release();
            _dirtyChunkEpochs = null;
            _dirtyBoundaryEpochs?.Release();
            _dirtyBoundaryEpochs = null;
            _activePageEpochs?.Release();
            _activePageEpochs = null;
            _activePageObservedEpochs?.Release();
            _activePageObservedEpochs = null;
            _activePageBoundaryEpochs?.Release();
            _activePageBoundaryEpochs = null;
            LatestActivePageEpochs = null;
            LatestActivePageObservedEpochs = null;
            LatestActivePageBoundaryEpochs = null;
            _dirtyChunkCount = int3.zero;
            if (_volume) { Destroy(_volume); _volume = null; }
            if (_colorVolume) { Destroy(_colorVolume); _colorVolume = null; }
            if (_projectiveShadowVolume) { Destroy(_projectiveShadowVolume); _projectiveShadowVolume = null; }
            if (_admissionTraceVolume) { Destroy(_admissionTraceVolume); _admissionTraceVolume = null; }
            if (_confidenceVolume) { Destroy(_confidenceVolume); _confidenceVolume = null; }
            if (_coherenceVolume) { Destroy(_coherenceVolume); _coherenceVolume = null; }
            if (_infiniTamVoteWeightVolume)
            {
                Destroy(_infiniTamVoteWeightVolume);
                _infiniTamVoteWeightVolume = null;
            }
            if (_tsdfResponsibilityVolume) { Destroy(_tsdfResponsibilityVolume); _tsdfResponsibilityVolume = null; }
            if (_tsdfSupportResponsibilityVolume) { Destroy(_tsdfSupportResponsibilityVolume); _tsdfSupportResponsibilityVolume = null; }
            _tsdfResponsibilityCaptureEnabled = false;
            IntegrationCount = 0;
            ResetInfiniTamTicket();
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
            EnsureGunGelEvidenceShadow();
            Clear();

            if (DepthCapture.Instance != null)
                DepthCapture.Instance.SetVoxelParams(voxelDistance, voxelSize);

            Logger.Info(firstAlloc
                ? "VolumeIntegrator: GPU resources allocated lazily on first scan/save/full-load."
                : "VolumeIntegrator: GPU volumes re-allocated after release.");
        }

        private void EnsureGunGelEvidenceShadow()
        {
            if (enableInfiniTamBaseline) return;
            if (!enableGunGelEvidenceShadow || _gunGelEvidenceShadow != null) return;
            try
            {
                _gunGelRuntimeFailureReported = false;
                _gunGelEvidenceShadow = new GunGelEvidenceShadow(
                    gunGelPixelStride, gunGelCellSize, gunGelReportInterval);
                Logger.Info($"枪胶 GPU 层已启用：stride={gunGelPixelStride}, " +
                            $"cell={gunGelCellSize:F2}m, K<=3, report={gunGelReportInterval}帧；" +
                            $"融合模式={(enableFinalCourtAdmissionExperiment ? "裁决准入" : enableGunGelGuardedFusionExperiment ? "枪胶实验" : "基线")}。");
            }
            catch (Exception ex)
            {
                Logger.Warning(enableFinalCourtAdmissionExperiment
                    ? $"枪胶 GPU 层初始化失败，裁决准入将严格停笔：{ex.Message}"
                    : $"枪胶 GPU 影子初始化失败，生产融合继续：{ex.Message}");
                _gunGelEvidenceShadow?.Dispose();
                _gunGelEvidenceShadow = null;
            }
        }

        private void DispatchGunGelEvidenceShadow(DepthCapture depth)
        {
            if (_gunGelEvidenceShadow == null) return;
            try
            {
                _gunGelEvidenceShadow.Dispatch(depth, _motionQuality, _gunGelCaptureFrameIndex++);
            }
            catch (Exception ex)
            {
                // 普通模式可整段拔除影子；裁决准入模式则由 Integrate 严格停笔，
                // 不能把 runtime failure 偷换成原始生产融合。
                if (!_gunGelRuntimeFailureReported)
                {
                    Logger.Warning(enableFinalCourtAdmissionExperiment
                        ? $"枪胶 GPU 层运行失败，裁决准入已熔断并停笔：{ex.Message}"
                        : $"枪胶 GPU 影子运行失败，已熔断但生产融合继续：{ex.Message}");
                    _gunGelRuntimeFailureReported = true;
                }
                _gunGelEvidenceShadow.Dispose();
                _gunGelEvidenceShadow = null;
                _gunGelGuardedFusionRuntimeHalted = true;
            }
        }

        private int CountGunGelDeferredFrames()
        {
            int count = 0;
            for (int i = 0; i < _gunGelDeferredFrames.Length; i++)
            {
                GunGelDeferredFrame frame = _gunGelDeferredFrames[i];
                if (frame != null && (frame.Pending || frame.Ready)) count++;
            }
            return count;
        }

        private void ResetGunGelDeferredFrames(bool releaseTextures)
        {
            _gunGelDeferredGeneration++;
            for (int i = 0; i < _gunGelDeferredFrames.Length; i++)
            {
                GunGelDeferredFrame frame = _gunGelDeferredFrames[i];
                if (frame == null) continue;
                _gunGelEvidenceShadow?.ReleaseFrameDecision(frame.Decision);
                frame.Decision = default;
                frame.Pending = false;
                frame.Ready = false;
                frame.Generation = _gunGelDeferredGeneration;
                if (!releaseTextures) continue;
                if (frame.RawDepth) Destroy(frame.RawDepth);
                if (frame.Depth) Destroy(frame.Depth);
                if (frame.Normal) Destroy(frame.Normal);
                if (frame.DilatedDepth) Destroy(frame.DilatedDepth);
                if (frame.EdgeReason) Destroy(frame.EdgeReason);
                if (frame.TemporalReason) Destroy(frame.TemporalReason);
                frame.RawDepth = null;
                frame.Depth = null;
                frame.Normal = null;
                frame.DilatedDepth = null;
                frame.EdgeReason = null;
                frame.TemporalReason = null;
            }
        }

        private static RenderTexture EnsureGunGelFrameCopy(RenderTexture target,
            Texture source, string name)
        {
            if (source == null) return null;
            int volumeDepth = 1;
            if (source is RenderTexture sourceRt) volumeDepth = sourceRt.volumeDepth;
            else if (source is Texture2DArray sourceArray) volumeDepth = sourceArray.depth;
            bool recreate = target == null || target.width != source.width ||
                            target.height != source.height ||
                            target.graphicsFormat != source.graphicsFormat ||
                            target.dimension != source.dimension ||
                            target.volumeDepth != volumeDepth;
            if (recreate)
            {
                if (target) Destroy(target);
                var descriptor = new RenderTextureDescriptor(source.width, source.height)
                {
                    graphicsFormat = source.graphicsFormat,
                    depthBufferBits = 0,
                    msaaSamples = 1,
                    mipCount = 1,
                    dimension = source.dimension,
                    volumeDepth = volumeDepth,
                    enableRandomWrite = false,
                    useMipMap = false,
                    autoGenerateMips = false
                };
                target = new RenderTexture(descriptor)
                {
                    name = name,
                    filterMode = source.filterMode,
                    wrapMode = source.wrapMode
                };
                target.Create();
            }
            Graphics.CopyTexture(source, target);
            return target;
        }

        private void ResetInfiniTamDeferredFrame(bool releaseTextures)
        {
            _infiniTamDeferredGeneration++;
            InfiniTamDeferredFrame frame = _infiniTamDeferredFrame;
            frame.Pending = false;
            frame.Ready = false;
            frame.Generation = _infiniTamDeferredGeneration;
            frame.Decision = default;
            _infiniTamLastQueuedPlatformFrame = -1;
            if (!releaseTextures) return;
            if (frame.Depth) Destroy(frame.Depth);
            if (frame.Normal) Destroy(frame.Normal);
            if (frame.DilatedDepth) Destroy(frame.DilatedDepth);
            if (frame.EdgeReason) Destroy(frame.EdgeReason);
            if (frame.TemporalReason) Destroy(frame.TemporalReason);
            frame.Depth = null;
            frame.Normal = null;
            frame.DilatedDepth = null;
            frame.EdgeReason = null;
            frame.TemporalReason = null;
        }

        private void ResetInfiniTamStartupState(bool preserveReseedCount)
        {
            int reseedCount = preserveReseedCount
                ? _infiniTamBootstrapReseedCount : 0;
            _infiniTamTrackingInitialised = false;
            _infiniTamStartupPhase = InfiniTamStartupPhase.AwaitingStillness;
            _infiniTamLastStartupPlatformFrame = -1;
            _infiniTamBootstrapStableFrames = 0;
            _infiniTamBootstrapConfirmedFrames = 0;
            _infiniTamRecoveryConfirmedFrames = 0;
            _infiniTamConsecutiveTrackingRejects = 0;
            _infiniTamBootstrapReseedCount = reseedCount;
        }

        private bool IsInfiniTamBootstrapMotionSafe(float angularSpeed,
            float linearSpeed)
        {
            return angularSpeed <= Mathf.Max(0.1f,
                       infiniTamBootstrapMaxAngularDegPerSec) &&
                   linearSpeed <= Mathf.Max(0.001f,
                       infiniTamBootstrapMaxLinearMps);
        }

        private string GetInfiniTamStartupPhaseLabel()
        {
            if (enableInfiniTamBaseline && !enableInfiniTamTrackingAuthority)
                return IntegrationCount > 0 ? "V1.3直融" : "待首帧";
            return _infiniTamStartupPhase switch
            {
                InfiniTamStartupPhase.AwaitingStillness =>
                    $"等稳{_infiniTamBootstrapStableFrames}/" +
                    Mathf.Max(1, infiniTamBootstrapStillFrames),
                InfiniTamStartupPhase.Seeding =>
                    $"建底{IntegrationCount}/" +
                    Mathf.Max(1, infiniTamTrackingBootstrapFrames),
                InfiniTamStartupPhase.Verifying =>
                    $"复核{_infiniTamBootstrapConfirmedFrames}/" +
                    Mathf.Max(1, infiniTamBootstrapConfirmFrames),
                InfiniTamStartupPhase.TrackingLost =>
                    _infiniTamRecoveryConfirmedFrames > 0
                        ? $"重锁{_infiniTamRecoveryConfirmedFrames}/" +
                          Mathf.Max(1, infiniTamRecoveryConfirmFrames)
                        : $"失跟{_infiniTamConsecutiveTrackingRejects}",
                _ => "正式"
            };
        }

        /// <summary>
        /// A bootstrap map has no user-visible authority. If it repeatedly
        /// fails the same frame-to-model quality gate that should validate it,
        /// discard it and acquire a new stable seed. A confirmed map is never
        /// destroyed here: mature InfiniTAM freezes fusion and waits for
        /// tracking/relocalisation instead.
        /// </summary>
        private bool TryReseedUnconfirmedInfiniTam(string reason)
        {
            if (_infiniTamTrackingInitialised ||
                _infiniTamConsecutiveTrackingRejects <
                Mathf.Max(1, infiniTamBootstrapRejectsBeforeReseed))
                return false;

            int reseedCount = _infiniTamBootstrapReseedCount + 1;
            Logger.Warning("InfiniTAM 启动模型连续复核失败；清除未发布坏底并重新建底：" +
                           reason);
            ClearInternal(preserveGunGelEvidence: false);
            IntegrationCount = 0;
            _integrationsSinceCoverage = 0;
            _infiniTamBootstrapReseedCount = reseedCount;
            _infiniTamLastTrackingDecision = "重建" + reason;
            return true;
        }

        private bool QueueInfiniTamTrackedFrame(DepthCapture depth,
            float angularSpeed, float linearSpeed, float motionQuality)
        {
            InfiniTamDeferredFrame frame = _infiniTamDeferredFrame;
            if (frame.Pending || frame.Ready || depth == null ||
                depth.DepthTex == null || depth.NormTex == null ||
                depth.DilatedDepthTex == null || depth.EdgeReasonTex == null ||
                depth.CurrentPlatformFrame == _infiniTamLastQueuedPlatformFrame)
                return false;

            int generation = _infiniTamDeferredGeneration;
            int sourceFrame = depth.CurrentPlatformFrame;
            try
            {
                frame.Depth = EnsureGunGelFrameCopy(frame.Depth, depth.DepthTex,
                    $"InfiniTamTrackDepth_{sourceFrame}");
                frame.Normal = EnsureGunGelFrameCopy(frame.Normal, depth.NormTex,
                    $"InfiniTamTrackNormal_{sourceFrame}");
                frame.DilatedDepth = EnsureGunGelFrameCopy(frame.DilatedDepth,
                    depth.DilatedDepthTex, $"InfiniTamTrackDilated_{sourceFrame}");
                frame.EdgeReason = EnsureGunGelFrameCopy(frame.EdgeReason,
                    depth.EdgeReasonTex, $"InfiniTamTrackEdge_{sourceFrame}");
                if (depth.TemporalReasonTex != null)
                    frame.TemporalReason = EnsureGunGelFrameCopy(
                        frame.TemporalReason, depth.TemporalReasonTex,
                        $"InfiniTamTrackTemporal_{sourceFrame}");
                else if (frame.TemporalReason)
                {
                    Destroy(frame.TemporalReason);
                    frame.TemporalReason = null;
                }
                frame.View = (Matrix4x4[])depth.View.Clone();
                frame.Projection = (Matrix4x4[])depth.Proj.Clone();
                frame.ViewInverse = (Matrix4x4[])depth.ViewInv.Clone();
                frame.ProjectionInverse = (Matrix4x4[])depth.ProjInv.Clone();
                frame.ExclusionCount = Mathf.Min(ExclusionZones.Count, 64);
                Array.Clear(frame.ExclusionPositions, 0,
                    frame.ExclusionPositions.Length);
                for (int i = 0; i < frame.ExclusionCount; i++)
                    if (ExclusionZones[i] != null)
                        frame.ExclusionPositions[i] = ExclusionZones[i].position;
                frame.PlatformFrame = sourceFrame;
                frame.Generation = generation;
                frame.AngularSpeed = angularSpeed;
                frame.LinearSpeed = linearSpeed;
                frame.MotionQuality = motionQuality;
                frame.Pending = true;
                frame.Ready = false;
                _infiniTamLastQueuedPlatformFrame = sourceFrame;
                _infiniTamModelRaycast ??= new InfiniTamModelRaycastAudit(8);
                bool dispatched = _infiniTamModelRaycast.TryDispatchProductionTracking(
                    frame.Depth, frame.Projection, frame.View,
                    frame.ProjectionInverse, frame.ViewInverse,
                    _volume, _infiniTamVoteWeightVolume,
                    voxelCount, voxelSize, voxelDistance, minMeshWeight,
                    rejectNearSamples ? minUpdateDist : 0f, maxUpdateDist,
                    sourceFrame, decision =>
                    {
                        if (frame.Generation != _infiniTamDeferredGeneration ||
                            frame.PlatformFrame != decision.SourceFrame)
                            return;
                        frame.Decision = decision;
                        frame.Pending = false;
                        frame.Ready = true;
                    });
                if (dispatched)
                {
                    // One retained source frame is one tracking attempt.  Do not
                    // count the later Integrate() polls while its async readback
                    // is pending; otherwise the HUD "stopped" total describes
                    // scheduler latency rather than rejected observations.
                    _infiniTamAttemptedFrames++;
                    _infiniTamLastTrackingDecision = "求解";
                    return true;
                }
                frame.Pending = false;
                _infiniTamTrackingQueueAbstained++;
                _infiniTamLastTrackingDecision = "跟踪忙";
                return false;
            }
            catch (Exception e)
            {
                frame.Pending = false;
                frame.Ready = false;
                _infiniTamTrackingQueueAbstained++;
                _infiniTamLastTrackingDecision = "留帧失败";
                Logger.Warning("InfiniTAM 同帧留帧失败；本帧停笔：" + e.Message);
                return false;
            }
        }

        private bool AcceptInfiniTamTrackedFrame(InfiniTamDeferredFrame frame,
            out string reason)
        {
            InfiniTamModelRaycastAudit.TrackingDecision decision = frame.Decision;
            if (!decision.ReadbackSucceeded) { reason = "回读"; return false; }
            if (decision.CorrespondenceCount < infiniTamTrackingMinCorrespondences)
            { reason = "少配"; return false; }
            // A large planar wall or ceiling is a valid reconstruction target
            // but point-to-plane ICP cannot independently observe all six pose
            // axes from one plane. Quest already supplies the retained frame's
            // world pose; the damped solve leaves unobservable correction axes
            // at that prior. Require only the three plane-observable axes here,
            // then let residual support and correction bounds judge the frame.
            // Requiring rank 6 made bootstrap publication impossible whenever
            // the user began by looking at an ordinary wall or ceiling.
            if (decision.EffectiveRank < infiniTamTrackingMinRank)
            { reason = "欠秩"; return false; }
            // A per-pass clamp is the ICP trust-region step limiter, not a
            // frame-level verdict.  The tracker has already re-raycast and
            // completed its coarse-to-fine passes; judge the accumulated
            // correction below instead of rejecting any frame that needed a
            // bounded intermediate step.
            if (decision.TranslationMm > infiniTamTrackingMaxTranslationMm)
            { reason = "位大"; return false; }
            if (decision.RotationDeg > infiniTamTrackingMaxRotationDeg)
            { reason = "转大"; return false; }
            // InfiniTAM judges both residual and inlier support. A frame that
            // arrived already aligned should not be rejected merely because
            // ICP correctly produced a near-zero update.
            bool alreadyAligned = decision.MeanBeforeMm <= 15f &&
                                  decision.Within30BeforePercent >=
                                  infiniTamTrackingMinWithin30Percent;
            if (!alreadyAligned &&
                decision.MeanBeforeMm - decision.MeanAfterMm <
                infiniTamTrackingMinImprovementMm)
            { reason = "无益"; return false; }
            if (decision.Within30AfterPercent <
                infiniTamTrackingMinWithin30Percent)
            { reason = "残大"; return false; }
            reason = "准";
            return true;
        }

        private static void ApplyInfiniTamTrackingCorrection(
            InfiniTamDeferredFrame frame)
        {
            Matrix4x4 correction = frame.Decision.Correction;
            for (int i = 0; i < frame.ViewInverse.Length; i++)
            {
                frame.ViewInverse[i] = correction * frame.ViewInverse[i];
                frame.View[i] = frame.ViewInverse[i].inverse;
            }
        }

        private void ReleaseInfiniTamDeferredFrame()
        {
            _infiniTamDeferredFrame.Pending = false;
            _infiniTamDeferredFrame.Ready = false;
            _infiniTamDeferredFrame.Decision = default;
        }

        private bool QueueGunGelGuardedFrame(DepthCapture depth,
            float angularSpeed, float linearSpeed, float motionQuality)
        {
            if (_gunGelEvidenceShadow == null || depth == null ||
                depth.DepthTex == null || depth.NormTex == null ||
                depth.DilatedDepthTex == null || depth.EdgeReasonTex == null)
            {
                _gunGelFusionQueueAbstained++;
                _gunGelLastFusionDecision = "缺纹";
                return false;
            }

            GunGelDeferredFrame frame = null;
            for (int i = 0; i < _gunGelDeferredFrames.Length; i++)
            {
                if (!_gunGelDeferredFrames[i].Pending && !_gunGelDeferredFrames[i].Ready)
                {
                    frame = _gunGelDeferredFrames[i];
                    break;
                }
            }
            if (frame == null)
            {
                _gunGelFusionQueueAbstained++;
                _gunGelLastFusionDecision = "队满";
                return false;
            }

            int frameIndex = _gunGelCaptureFrameIndex++;
            int generation = _gunGelDeferredGeneration;
            try
            {
                if (depth.PlatformDepthWitnessTex != null)
                {
                    frame.RawDepth = EnsureGunGelFrameCopy(frame.RawDepth,
                        depth.PlatformDepthWitnessTex, $"GunGelRawDepth_{frameIndex}");
                }
                else if (frame.RawDepth)
                {
                    Destroy(frame.RawDepth);
                    frame.RawDepth = null;
                }
                frame.Depth = EnsureGunGelFrameCopy(frame.Depth, depth.DepthTex,
                    $"GunGelDepth_{frameIndex}");
                frame.Normal = EnsureGunGelFrameCopy(frame.Normal, depth.NormTex,
                    $"GunGelNormal_{frameIndex}");
                frame.DilatedDepth = EnsureGunGelFrameCopy(frame.DilatedDepth,
                    depth.DilatedDepthTex, $"GunGelDilated_{frameIndex}");
                frame.EdgeReason = EnsureGunGelFrameCopy(frame.EdgeReason,
                    depth.EdgeReasonTex, $"GunGelEdge_{frameIndex}");
                if (depth.TemporalReasonTex != null)
                {
                    frame.TemporalReason = EnsureGunGelFrameCopy(frame.TemporalReason,
                        depth.TemporalReasonTex, $"GunGelTemporalReason_{frameIndex}");
                }
                else if (frame.TemporalReason)
                {
                    Destroy(frame.TemporalReason);
                    frame.TemporalReason = null;
                }
                frame.View = (Matrix4x4[])depth.View.Clone();
                frame.Projection = (Matrix4x4[])depth.Proj.Clone();
                frame.ViewInverse = (Matrix4x4[])depth.ViewInv.Clone();
                frame.ProjectionInverse = (Matrix4x4[])depth.ProjInv.Clone();
                frame.ExclusionCount = Mathf.Min(ExclusionZones.Count, 64);
                Array.Clear(frame.ExclusionPositions, 0, frame.ExclusionPositions.Length);
                for (int i = 0; i < frame.ExclusionCount; i++)
                    if (ExclusionZones[i] != null)
                        frame.ExclusionPositions[i] = ExclusionZones[i].position;
                frame.FrameIndex = frameIndex;
                frame.PlatformFrame = depth.CurrentPlatformFrame;
                frame.Generation = generation;
                frame.AngularSpeed = angularSpeed;
                frame.LinearSpeed = linearSpeed;
                frame.MotionQuality = motionQuality;
                frame.Pending = true;
                frame.Ready = false;

                bool dispatched = _gunGelEvidenceShadow.Dispatch(
                    frame.RawDepth, frame.Depth, frame.Normal,
                    frame.EdgeReason, frame.TemporalReason,
                    depth.DepthWidth, depth.DepthHeight,
                    frame.ProjectionInverse, frame.ViewInverse,
                    motionQuality, angularSpeed, linearSpeed,
                    frame.PlatformFrame, frameIndex,
                    decision =>
                    {
                        if (frame.Generation != _gunGelDeferredGeneration ||
                            decision.FrameIndex != frame.FrameIndex)
                        {
                            _gunGelEvidenceShadow?.ReleaseFrameDecision(decision);
                            return;
                        }
                        frame.Decision = decision;
                        frame.Pending = false;
                        frame.Ready = true;
                    });
                if (dispatched) return true;
                frame.Pending = false;
                _gunGelFusionQueueAbstained++;
                _gunGelLastFusionDecision = "影忙";
                return false;
            }
            catch (Exception ex)
            {
                frame.Pending = false;
                frame.Ready = false;
                _gunGelGuardedFusionRuntimeHalted = true;
                _gunGelLastFusionDecision = "熔断";
                if (!_gunGelRuntimeFailureReported)
                {
                    Logger.Warning(enableFinalCourtAdmissionExperiment
                        ? $"枪胶留帧失败，裁决准入已熔断并停笔：{ex.Message}"
                        : $"枪胶受保护融合留帧失败，回到基线融合：{ex.Message}");
                    _gunGelRuntimeFailureReported = true;
                }
                return false;
            }
        }

        private bool TryGetOldestResolvedGunGelFrame(out GunGelDeferredFrame result)
        {
            result = null;
            GunGelDeferredFrame oldest = null;
            for (int i = 0; i < _gunGelDeferredFrames.Length; i++)
            {
                GunGelDeferredFrame frame = _gunGelDeferredFrames[i];
                if (!frame.Pending && !frame.Ready) continue;
                if (oldest == null || frame.FrameIndex < oldest.FrameIndex) oldest = frame;
            }
            if (oldest == null || oldest.Pending) return false;
            result = oldest;
            return true;
        }

        private bool AcceptGunGelFrame(GunGelDeferredFrame frame, out string reason)
        {
            GunGelEvidenceShadow.FrameDecision decision = frame.Decision;
            if (!decision.ReadbackSucceeded) { reason = "回读"; return false; }
            if (!decision.HasFusionAdmissionBuffers) { reason = "缺证"; return false; }
            if (frame.AngularSpeed > gunGelFusionMaxAngularSpeed) { reason = "快角"; return false; }
            if (frame.LinearSpeed > gunGelFusionMaxLinearSpeed) { reason = "快移"; return false; }
            if (decision.EffectiveRank < 6) { reason = "欠秩"; return false; }
            if (decision.CorrespondenceCount < gunGelFusionMinCorrespondences) { reason = "少配"; return false; }
            if (decision.TranslationClamped || decision.RotationClamped) { reason = "撞顶"; return false; }
            if (decision.TranslationMm > gunGelFusionMaxTranslationMm) { reason = "位大"; return false; }
            if (decision.RotationDeg > gunGelFusionMaxRotationDeg) { reason = "转大"; return false; }
            reason = "校";
            return true;
        }

        private static void ApplyGunGelCorrection(GunGelDeferredFrame frame)
        {
            Matrix4x4 correction = frame.Decision.Correction;
            for (int i = 0; i < frame.ViewInverse.Length; i++)
            {
                frame.ViewInverse[i] = correction * frame.ViewInverse[i];
                frame.View[i] = frame.ViewInverse[i].inverse;
            }
        }

        private void ReleaseGunGelDeferredFrame(GunGelDeferredFrame frame)
        {
            if (frame == null) return;
            _gunGelEvidenceShadow?.ReleaseFrameDecision(frame.Decision);
            frame.Decision = default;
            frame.Pending = false;
            frame.Ready = false;
        }

        private void QueueProductSurfaceCourtReadback(GunGelDeferredFrame frame,
            int attemptIndex, Matrix4x4 fusionCorrection)
        {
            if (frame == null || _productCourtReadback != null ||
                !frame.Decision.HasFusionAdmissionBuffers ||
                frame.Decision.FusionCorrespondences == null ||
                frame.Decision.FusionCorrespondenceIdentity == null)
                return;

            int eye = DepthCapture.FusionEyeIndex;
            Matrix4x4 viewInverse = frame.ViewInverse != null &&
                frame.ViewInverse.Length > eye
                ? frame.ViewInverse[eye]
                : Matrix4x4.identity;
            Camera head = Camera.main;
            var pending = new ProductCourtReadback
            {
                Generation = _productCourtGeneration,
                GunGelFrame = frame.FrameIndex,
                SourceFrame = frame.PlatformFrame,
                AttemptIndex = attemptIndex,
                ViewInverse = viewInverse,
                FusionCorrection = fusionCorrection,
                AngularSpeed = frame.AngularSpeed,
                LinearSpeed = frame.LinearSpeed,
                MotionQuality = frame.MotionQuality,
                HeadEuler = head != null ? head.transform.eulerAngles : Vector3.zero
            };
            _productCourtReadback = pending;

            try
            {
                AsyncGPUReadback.Request(frame.Decision.FusionCorrespondences,
                    request =>
                    {
                        try
                        {
                            if (request.hasError)
                                pending.Failed = true;
                            else
                                pending.Correspondences = request
                                    .GetData<GunGelEvidenceShadow.Correspondence>()
                                    .ToArray();
                        }
                        catch
                        {
                            pending.Failed = true;
                        }
                        finally
                        {
                            pending.CorrespondencesDone = true;
                            TryCompleteProductSurfaceCourtReadback(pending);
                        }
                    });
                AsyncGPUReadback.Request(
                    frame.Decision.FusionCorrespondenceIdentity, request =>
                    {
                        try
                        {
                            if (request.hasError)
                                pending.Failed = true;
                            else
                                pending.Identities = request.GetData<uint4>().ToArray();
                        }
                        catch
                        {
                            pending.Failed = true;
                        }
                        finally
                        {
                            pending.IdentitiesDone = true;
                            TryCompleteProductSurfaceCourtReadback(pending);
                        }
                    });
            }
            catch
            {
                pending.Failed = true;
                pending.CorrespondencesDone = true;
                pending.IdentitiesDone = true;
                TryCompleteProductSurfaceCourtReadback(pending);
            }
        }

        private void TryCompleteProductSurfaceCourtReadback(
            ProductCourtReadback pending)
        {
            if (pending == null || !pending.CorrespondencesDone ||
                !pending.IdentitiesDone)
                return;
            if (ReferenceEquals(_productCourtReadback, pending))
                _productCourtReadback = null;
            if (pending.Failed || pending.Generation != _productCourtGeneration)
                return;

            _productSurfaceCourt.Record(pending.GunGelFrame,
                pending.SourceFrame, pending.AttemptIndex,
                pending.Correspondences, pending.Identities,
                pending.ViewInverse, pending.FusionCorrection,
                pending.AngularSpeed, pending.LinearSpeed,
                pending.MotionQuality, pending.HeadEuler);
        }

        private void ResetProductSurfaceCourt()
        {
            _productCourtGeneration++;
            _productCourtReadback = null;
            _productSurfaceCourt.BeginInMemory();
        }

        private void RebindKernelTextures()
        {
            EnsureDirtyChunkBuffer();
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            if (!enableInfiniTamBaseline)
            {
                _clearKernel.Set(TsdfResponsibilityRWID,
                    _tsdfResponsibilityVolume);
                _clearKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
            }
            _clearInfiniTamVotesKernel.Set(InfiniTamVoteWeightRWID,
                _infiniTamVoteWeightVolume);
            if (!enableInfiniTamBaseline)
            {
                _invalidateGunGelSuccessionsKernel.Set(VolumeRWID, _volume);
                _invalidateGunGelSuccessionsKernel.Set(ColorVolumeRWID, _colorVolume);
                _invalidateGunGelSuccessionsKernel.Set(AdmissionTraceRWID,
                    _admissionTraceVolume);
                _invalidateGunGelSuccessionsKernel.Set(TsdfResponsibilityRWID,
                    _tsdfResponsibilityVolume);
                _invalidateGunGelSuccessionsKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _invalidateGunGelSuccessionsKernel.Set(ConfidenceRWID, _confidenceVolume);
                _invalidateGunGelSuccessionsKernel.Set(CoherenceRWID, _coherenceVolume);
                _invalidateGunGelSuccessionsKernel.Set(DirtyChunkEpochsID,
                    _dirtyChunkEpochs);
                _invalidateGunGelSuccessionsKernel.Set(DirtyBoundaryEpochsID,
                    _dirtyBoundaryEpochs);
                _invalidateGunGelSuccessionsKernel.Set(ActivePageEpochsID,
                    _activePageEpochs);
                _invalidateGunGelSuccessionsKernel.Set(ActivePageObservedEpochsID,
                    _activePageObservedEpochs);
                _invalidateGunGelSuccessionsKernel.Set(ActivePageBoundaryEpochsID,
                    _activePageBoundaryEpochs);
            }
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(InfiniTamVoteWeightRWID, _infiniTamVoteWeightVolume);
            _integrateKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
            _integrateKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);
            if (!enableInfiniTamBaseline)
            {
                _integrateKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
                _integrateKernel.Set(TsdfResponsibilityRWID, _tsdfResponsibilityVolume);
                _integrateKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _integrateKernel.Set(ConfidenceRWID, _confidenceVolume);
                _integrateKernel.Set(ActivePageEpochsID, _activePageEpochs);
                _integrateKernel.Set(ActivePageObservedEpochsID,
                    _activePageObservedEpochs);
                _integrateKernel.Set(ActivePageBoundaryEpochsID,
                    _activePageBoundaryEpochs);
                _pruneKernel.Set(VolumeRWID, _volume);
                _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
                _pruneKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
                _pruneKernel.Set(TsdfResponsibilityRWID, _tsdfResponsibilityVolume);
                _pruneKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _pruneKernel.Set(DirtyChunkEpochsID, _dirtyChunkEpochs);
                _pruneKernel.Set(DirtyBoundaryEpochsID, _dirtyBoundaryEpochs);
                _pruneKernel.Set(ActivePageEpochsID, _activePageEpochs);
                _pruneKernel.Set(ActivePageObservedEpochsID,
                    _activePageObservedEpochs);
                _pruneKernel.Set(ActivePageBoundaryEpochsID,
                    _activePageBoundaryEpochs);
            }
            if (!enableInfiniTamBaseline)
            {
                _freezeKernel.Set(VolumeRWID, _volume);
                _freezeKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _unfreezeKernel.Set(VolumeRWID, _volume);
                _unfreezeKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _applyFreezeMaskKernel.Set(VolumeRWID, _volume);
                _applyFreezeMaskKernel.Set(TsdfSupportResponsibilityRWID,
                    _tsdfSupportResponsibilityVolume);
                _applyFreezeMaskKernel.Set(ChunkFreezeSetMaskID,
                    _chunkFreezeSetMask);
                _applyFreezeMaskKernel.Set(ChunkFreezeClearMaskID,
                    _chunkFreezeClearMask);
                _clearVotesKernel.Set(ChunkFreezeClearMaskID,
                    _chunkFreezeClearMask);
                _clearVotesKernel.Set(FrozenChunkVotesID,
                    _frozenChunkVotes);
                _maturityKernel.Set(VolumeRWID, _volume);
                _maturityKernel.Set(ChunkMaturityID, _chunkMaturity);
                _maturityKernel.Set(ConfidenceRWID, _confidenceVolume);
                _integrateKernel.Set(FrozenChunkVotesID, _frozenChunkVotes);
                _integrateKernel.Set(FrozenChunkBitsID, _frozenChunkBits);
            }
            _coverageKernel.Set(VolumeRWID, _volume);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);
            _clearKernel.Set(ConfidenceRWID, _confidenceVolume);
            _clearKernel.Set(CoherenceRWID, _coherenceVolume);
            if (!enableInfiniTamBaseline)
                _integrateKernel.Set(CoherenceRWID, _coherenceVolume);
            _confidenceKernel.Set(VolumeRWID, _volume);
            _confidenceKernel.Set(ConfidenceRWID, _confidenceVolume);
            _confidenceKernel.Set(CoherenceRWID, _coherenceVolume);
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
            if (enableInfiniTamBaseline)
            {
                if (_dirtyChunkEpochs != null &&
                    _dirtyChunkEpochs.count == Mathf.Max(1, requiredCount) &&
                    _dirtyBoundaryEpochs != null &&
                    _dirtyBoundaryEpochs.count == boundaryCount &&
                    _activePageEpochs == null &&
                    _activePageObservedEpochs == null &&
                    _activePageBoundaryEpochs == null &&
                    _frozenChunkVotes == null)
                {
                    _dirtyChunkCount = required;
                    _frozenChunkCount = int3.zero;
                    return;
                }

                _dirtyChunkEpochs?.Release();
                _dirtyBoundaryEpochs?.Release();
                _activePageEpochs?.Release();
                _activePageEpochs = null;
                _activePageObservedEpochs?.Release();
                _activePageObservedEpochs = null;
                _activePageBoundaryEpochs?.Release();
                _activePageBoundaryEpochs = null;
                ReleaseFrozenBlockBuffers();

                _dirtyChunkEpochs = new ComputeBuffer(
                    Mathf.Max(1, requiredCount), sizeof(uint));
                _dirtyBoundaryEpochs = new ComputeBuffer(
                    boundaryCount, sizeof(uint));
                _dirtyChunkEpochs.SetData(
                    new uint[Mathf.Max(1, requiredCount)]);
                _dirtyBoundaryEpochs.SetData(new uint[boundaryCount]);
                _dirtyChunkCount = required;
                _frozenChunkCount = int3.zero;
                return;
            }

            int fChunkSize = Mathf.Max(8, frozenChunkSize);
            int3 frozenRequired = new int3(
                Mathf.CeilToInt(voxelCount.x / (float)fChunkSize),
                Mathf.CeilToInt(voxelCount.y / (float)fChunkSize),
                Mathf.CeilToInt(voxelCount.z / (float)fChunkSize));
            int frozenRequiredCount = frozenRequired.x * frozenRequired.y * frozenRequired.z;
            int activeBoundaryCount = Mathf.Max(1, frozenRequiredCount * 6);
            if (_dirtyChunkEpochs != null && _dirtyChunkEpochs.count == requiredCount &&
                _dirtyBoundaryEpochs != null && _dirtyBoundaryEpochs.count == boundaryCount &&
                _activePageEpochs != null && _activePageEpochs.count == Mathf.Max(1, frozenRequiredCount) &&
                _activePageObservedEpochs != null && _activePageObservedEpochs.count == Mathf.Max(1, frozenRequiredCount) &&
                _activePageBoundaryEpochs != null && _activePageBoundaryEpochs.count == activeBoundaryCount &&
                _frozenChunkVotes != null && _frozenChunkVotes.count == Mathf.Max(1, frozenRequiredCount))
            {
                _dirtyChunkCount = required;
                _frozenChunkCount = frozenRequired;
                return;
            }

            _dirtyChunkEpochs?.Release();
            _dirtyBoundaryEpochs?.Release();
            _activePageEpochs?.Release();
            _activePageObservedEpochs?.Release();
            _activePageBoundaryEpochs?.Release();
            _dirtyChunkEpochs = new ComputeBuffer(Mathf.Max(1, requiredCount), sizeof(uint));
            _dirtyBoundaryEpochs = new ComputeBuffer(boundaryCount, sizeof(uint));
            _activePageEpochs = new ComputeBuffer(Mathf.Max(1, frozenRequiredCount), sizeof(uint));
            _activePageObservedEpochs = new ComputeBuffer(Mathf.Max(1, frozenRequiredCount), sizeof(uint));
            _activePageBoundaryEpochs = new ComputeBuffer(activeBoundaryCount, sizeof(uint));
            _dirtyChunkCount = required;
            _dirtyChunkEpochs.SetData(new uint[Mathf.Max(1, requiredCount)]);
            _dirtyBoundaryEpochs.SetData(new uint[boundaryCount]);
            _activePageEpochs.SetData(new uint[Mathf.Max(1, frozenRequiredCount)]);
            _activePageObservedEpochs.SetData(new uint[Mathf.Max(1, frozenRequiredCount)]);
            _activePageBoundaryEpochs.SetData(new uint[activeBoundaryCount]);

            ReleaseFrozenBlockBuffers();
            _frozenChunkCount = frozenRequired;
            int frozenCount = Mathf.Max(1, frozenRequiredCount);
            int maskWords = Mathf.Max(1, (frozenRequiredCount + 31) / 32);
            _chunkFreezeSetMask = new ComputeBuffer(maskWords, sizeof(uint));
            _chunkFreezeClearMask = new ComputeBuffer(maskWords, sizeof(uint));
            _frozenChunkVotes = new ComputeBuffer(frozenCount, sizeof(uint) * 4); // T2：uint2→uint4（z=补洞票）
            _chunkMaturity = new ComputeBuffer(frozenCount, sizeof(uint) * 4);
            _frozenChunkBits = new ComputeBuffer(maskWords, sizeof(uint));
            _voteZeros = new uint[frozenCount * 4];
            _maturityZeros = new uint[frozenCount * 4];
            _chunkFreezeSetMask.SetData(new uint[maskWords]);
            _chunkFreezeClearMask.SetData(new uint[maskWords]);
            _frozenChunkVotes.SetData(_voteZeros);
            _chunkMaturity.SetData(_maturityZeros);
            _frozenChunkBits.SetData(new uint[maskWords]);
        }

        private void ReleaseFrozenBlockBuffers()
        {
            _chunkFreezeSetMask?.Release();
            _chunkFreezeSetMask = null;
            _chunkFreezeClearMask?.Release();
            _chunkFreezeClearMask = null;
            _frozenChunkVotes?.Release();
            _frozenChunkVotes = null;
            _frozenChunkBits?.Release();
            _frozenChunkBits = null;
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
                if (_activePageEpochs != null)
                    _activePageEpochs.SetData(new uint[_activePageEpochs.count]);
                if (_activePageObservedEpochs != null)
                    _activePageObservedEpochs.SetData(new uint[_activePageObservedEpochs.count]);
                if (_activePageBoundaryEpochs != null)
                    _activePageBoundaryEpochs.SetData(new uint[_activePageBoundaryEpochs.count]);
            }
            ConfigureDirtyTracking(true);
        }

        /// <summary>
        /// 消费当前候选事务产生的一次性局部拆旧票。候选只指出哪段旧历史已经
        /// 失去身份。本批正常 Integrate 完成后才清除，避免按旧账本冻结的本批
        /// 深度立刻把旧轨种回来；下一批只能用新账本认可的全分辨率深度播种。
        /// </summary>
        private void InvalidateGunGelSucceededRegions()
        {
            if (_gunGelEvidenceShadow == null ||
                !_gunGelEvidenceShadow.TryGetSuccessionInvalidations(
                    out GunGelEvidenceShadow.SuccessionInvalidationBuffers invalidations) ||
                !invalidations.IsValid)
                return;

            _invalidateGunGelSuccessionsKernel.Set(
                GunGelSuccessionInvalidationArgsID, invalidations.Args);
            _invalidateGunGelSuccessionsKernel.Set(
                GunGelSuccessionInvalidationRegionsID, invalidations.Regions);
            compute.SetInt(GunGelSuccessionInvalidationCapacityID,
                invalidations.Capacity);
            compute.DispatchIndirect(
                _invalidateGunGelSuccessionsKernel.KernelIndex, invalidations.Args);
        }

        /// <summary>
        /// Atomically retires the narrow TSDF neighbourhood of a court
        /// generation that has been replaced or conclusively revoked. The same
        /// accepted batch may then repopulate it only where current raw rays
        /// match the new independently published plane.
        /// </summary>
        private void InvalidateFinalCourtReplacements(
            ScanReplaySessionPackage session)
        {
            if (session == null || _finalCourtInvalidationArgs == null ||
                _finalCourtInvalidationRegions == null)
                return;
            int count = session.DrainFinalCourtInvalidationRegions(
                _finalCourtInvalidationStaging);
            if (count <= 0) return;

            _finalCourtInvalidationRegions.SetData(
                _finalCourtInvalidationStaging, 0, 0, count);
            _finalCourtInvalidationCount[0] = (uint)count;
            _finalCourtInvalidationArgs.SetData(_finalCourtInvalidationCount);
            _invalidateGunGelSuccessionsKernel.Set(
                GunGelSuccessionInvalidationArgsID,
                _finalCourtInvalidationArgs);
            _invalidateGunGelSuccessionsKernel.Set(
                GunGelSuccessionInvalidationRegionsID,
                _finalCourtInvalidationRegions);
            compute.SetInt(GunGelSuccessionInvalidationCapacityID, count);
            compute.Dispatch(_invalidateGunGelSuccessionsKernel.KernelIndex,
                count, 1, 1);
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
            // Active-page epochs are event ledgers, not a full-replay invalidation
            // mask. Marking every 32^3 page here would make a fresh scan enqueue
            // all empty room pages before any surface exists. Existing loaded
            // surfaces are discovered by the low-rate census fallback; new live
            // surfaces advance this ledger directly in MarkDirtyChunk.
            if (_activePageEpochs != null)
                _activePageEpochs.SetData(new uint[_activePageEpochs.count]);
            if (_activePageObservedEpochs != null)
                _activePageObservedEpochs.SetData(new uint[_activePageObservedEpochs.count]);
            if (_activePageBoundaryEpochs != null)
                _activePageBoundaryEpochs.SetData(new uint[_activePageBoundaryEpochs.count]);
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

        /// <summary>
        /// 清零置信度统计缓冲并重新普查一遍（分歧 EMA 三档：高/中/低），
        /// 异步回读到 ConfidenceVoxelCount/High/Mid/Low。HUD 定期调用。
        /// </summary>
        public void RefreshConfidenceStats()
        {
            if (_volume == null || _confidenceStats == null) return;
            if (_confidenceReadbackPending) return;
            _confidenceReadbackPending = true;
            _confidenceStats.SetData(new uint[6]);
            _confidenceKernel.DispatchFit(_volume);
            AsyncGPUReadback.Request(_confidenceStats, OnConfidenceReadback);
        }

        private void OnConfidenceReadback(AsyncGPUReadbackRequest request)
        {
            _confidenceReadbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < 6) return;
            ConfidenceVoxelCount = (int)data[0];
            ConfidenceHighCount = (int)data[1];
            ConfidenceMidCount = (int)data[2];
            ConfidenceLowCount = (int)data[3];
            ConfidenceLowCoherentCount = (int)data[4];
            ConfidenceLowNoiseCount = (int)data[5];
        }

        private bool PrepareInfiniTamTicketSample()
        {
            if (!enableInfiniTamBaseline || _infiniTamTicketStats == null ||
                _infiniTamTicketReadbackPending ||
                Time.unscaledTime < _nextInfiniTamTicketTime)
                return false;

            _infiniTamTicketStats.SetData(ZeroInfiniTamTicketStats);
            compute.SetFloat(InfiniTamTicketEnabledID, 1f);
            return true;
        }

        private void RequestInfiniTamTicketReadback()
        {
            if (_infiniTamTicketStats == null) return;
            _infiniTamTicketReadbackPending = true;
            _nextInfiniTamTicketTime = Time.unscaledTime +
                                       InfiniTamTicketIntervalSeconds;
            int generation = _infiniTamTicketGeneration;
            AsyncGPUReadback.Request(_infiniTamTicketStats,
                request => OnInfiniTamTicketReadback(request, generation));
        }

        private void OnInfiniTamTicketReadback(
            AsyncGPUReadbackRequest request, int generation)
        {
            // A clear/release invalidates the meaning of an outstanding GPU
            // receipt. Do not let its callback repopulate a fresh session.
            if (generation != _infiniTamTicketGeneration) return;
            _infiniTamTicketReadbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < InfiniTamTicketStatCount) return;

            _infiniTamTicketNew = data[0];
            _infiniTamTicketContinuing = data[1];
            _infiniTamTicketMature = data[2];
            _infiniTamTicketVoteSum = data[3];
            _infiniTamTicketSurfaceSamples = data[0] + data[1] + data[2];
            _infiniTamTicketCumulativeNew += data[0];
            _infiniTamTicketCumulativeContinuing += data[1];
            _infiniTamTicketCumulativeMature += data[2];
            _infiniTamTicketCumulativeVoteSum += data[3];
            _infiniTamTicketCumulativeSurfaceSamples +=
                (ulong)data[0] + data[1] + data[2];
            _infiniTamTicketSampleCount++;
            _hasInfiniTamTicket = true;
        }

        private void ResetInfiniTamTicket()
        {
            _infiniTamTicketGeneration++;
            _infiniTamTicketReadbackPending = false;
            _nextInfiniTamTicketTime = 0f;
            _hasInfiniTamTicket = false;
            _infiniTamTicketNew = 0;
            _infiniTamTicketContinuing = 0;
            _infiniTamTicketMature = 0;
            _infiniTamTicketVoteSum = 0;
            _infiniTamTicketSurfaceSamples = 0;
            _infiniTamTicketCumulativeNew = 0;
            _infiniTamTicketCumulativeContinuing = 0;
            _infiniTamTicketCumulativeMature = 0;
            _infiniTamTicketCumulativeVoteSum = 0;
            _infiniTamTicketCumulativeSurfaceSamples = 0;
            _infiniTamTicketSampleCount = 0;
            _infiniTamAttemptedFrames = 0;
            _infiniTamFusedFrames = 0;
            _infiniTamTicketStats?.SetData(ZeroInfiniTamTicketStats);
            if (compute != null)
                compute.SetFloat(InfiniTamTicketEnabledID, 0f);
        }

        public string GetInfiniTamTicketCompact()
        {
            if (!enableInfiniTamBaseline) return string.Empty;
            int stopped = Mathf.Max(0,
                _infiniTamAttemptedFrames - _infiniTamFusedFrames);
            if (!_hasInfiniTamTicket || _infiniTamTicketSurfaceSamples == 0)
                return $"票[统计中] 融{_infiniTamFusedFrames} 停{stopped} " +
                       $"启[{GetInfiniTamStartupPhaseLabel()}]";

            float averageVotes = (float)_infiniTamTicketVoteSum /
                                 _infiniTamTicketSurfaceSamples;
            float singleFrameAuthority = 100f / (averageVotes + 1f);
            return $"票 新{FormatCarveCount(_infiniTamTicketNew)} " +
                   $"续{FormatCarveCount(_infiniTamTicketContinuing)} " +
                   $"熟{FormatCarveCount(_infiniTamTicketMature)} " +
                   $"均{averageVotes:F1} 权{singleFrameAuthority:F1}% " +
                   $"融{_infiniTamFusedFrames} 停{stopped} " +
                   $"启[{GetInfiniTamStartupPhaseLabel()}]";
        }

        public string GetInfiniTamModelRaycastCompact()
        {
            if (!enableInfiniTamBaseline) return string.Empty;
            string model = _infiniTamModelRaycast?.GetCompact() ?? "模[待启动]";
            if (!enableInfiniTamTrackingAuthority)
                return model + "\n旁证[只读] Quest位姿直融 不改姿/不停写";
            string state = _infiniTamDeferredFrame.Pending ? "求解" :
                           _infiniTamDeferredFrame.Ready ? "待融" :
                           _infiniTamLastTrackingDecision;
            return model + $"\n跟闸 准{_infiniTamTrackingAccepted} " +
                   $"拒{_infiniTamTrackingRejected} 忙{_infiniTamTrackingQueueAbstained} " +
                   $"态{state} 重{_infiniTamBootstrapReseedCount}";
        }

        private void RequestCarveStatsReadback()
        {
            if (_carveStats == null)
            {
                CompleteCarveStatsFlushCallbacks(false);
                return;
            }
            if (_carveStatsReadbackPending) return;
            _carveStatsReadbackPending = true;
            AsyncGPUReadback.Request(_carveStats, OnCarveStatsReadback);
        }

        private void OnCarveStatsReadback(AsyncGPUReadbackRequest request)
        {
            _carveStatsReadbackPending = false;
            if (request.hasError)
            {
                CompleteCarveStatsFlushCallbacks(false);
                return;
            }
            var data = request.GetData<uint>();
            if (data.Length < CarveStatsCount)
            {
                CompleteCarveStatsFlushCallbacks(false);
                return;
            }
            for (int i = 0; i < CarveStatsCount; i++)
            {
                LastCarveStats[i] = data[i];
                CumulativeCarveStats[i] += data[i];
            }
            var periodCounters = new uint[CarveStatsCount - 93];
            for (int i = 93; i < CarveStatsCount; i++)
                periodCounters[i - 93] = data[i];
            _fovLedgerPeriods.Add(new FovLedgerPeriod
            {
                Index = _fovLedgerPeriodIndex++,
                Utc = DateTime.UtcNow,
                ElapsedSeconds = Mathf.Max(0f, Time.realtimeSinceStartup - _fovLedgerStartedRealtime),
                Counters = periodCounters
            });
            HasCarveStats = true;
            _carveStats.SetData(ZeroCarveStats); // 数据已落袋，清零开新周期
            _lastMotionGatedCount = _motionGatedSinceStats; // 运动闸同节奏结算
            _motionGatedSinceStats = 0;
            CompleteCarveStatsFlushCallbacks(true);
        }

        private void CompleteCarveStatsFlushCallbacks(bool success)
        {
            if (_carveStatsFlushCallbacks.Count == 0) return;
            Action<bool>[] callbacks = _carveStatsFlushCallbacks.ToArray();
            _carveStatsFlushCallbacks.Clear();
            for (int i = 0; i < callbacks.Length; i++)
            {
                try { callbacks[i]?.Invoke(success); }
                catch (Exception e)
                {
                    Logger.Warning("融合终账回调失败：" + e.Message);
                }
            }
        }

        private void RequestProjectiveShadowCarveStatsReadback()
        {
            if (_projectiveShadowCarveStats == null)
            {
                CompleteProjectiveShadowFlushCallbacks(false);
                return;
            }
            if (_projectiveShadowCarveStatsReadbackPending) return;
            _projectiveShadowCarveStatsReadbackPending = true;
            AsyncGPUReadback.Request(_projectiveShadowCarveStats, OnProjectiveShadowCarveStatsReadback);
        }

        private void OnProjectiveShadowCarveStatsReadback(AsyncGPUReadbackRequest request)
        {
            _projectiveShadowCarveStatsReadbackPending = false;
            if (request.hasError)
            {
                CompleteProjectiveShadowFlushCallbacks(false);
                return;
            }
            var data = request.GetData<uint>();
            if (data.Length < CarveStatsCount)
            {
                CompleteProjectiveShadowFlushCallbacks(false);
                return;
            }
            for (int i = 0; i < CarveStatsCount; i++)
            {
                LastProjectiveShadowCarveStats[i] = data[i];
                CumulativeProjectiveShadowCarveStats[i] += data[i];
            }
            HasProjectiveShadowCarveStats = true;
            _projectiveShadowCarveStats.SetData(ZeroCarveStats);
            CompleteProjectiveShadowFlushCallbacks(true);
        }

        private void CompleteProjectiveShadowFlushCallbacks(bool success)
        {
            if (_projectiveShadowFlushCallbacks.Count == 0) return;
            Action<bool>[] callbacks = _projectiveShadowFlushCallbacks.ToArray();
            _projectiveShadowFlushCallbacks.Clear();
            for (int i = 0; i < callbacks.Length; i++)
            {
                try { callbacks[i]?.Invoke(success); }
                catch (Exception e)
                {
                    Logger.Warning("投影影子终账回调失败：" + e.Message);
                }
            }
        }

        private static string FormatCarveCount(uint v)
        {
            return v >= 10000u ? (v / 10000f).ToString("0.0") + "万" : v.ToString();
        }

        /// <summary>
        /// 即时壳→TSDF 的生产写入闭环。180 是近表面机会；其余槽只记录既有
        /// 生产分支的实际去向，不参与融合、权重或提取。HUD 在即时壳档直接显示。
        /// </summary>
        public string GetFusionAdmissionDiagnosticsCompact()
        {
            if (!HasCarveStats || LastCarveStats.Length < CarveStatsCount) return "融写 统计中";
            uint opportunities = LastCarveStats[180];
            uint accepted = LastCarveStats[192] + LastCarveStats[193] + LastCarveStats[194];
            uint gunGelRejected = enableFinalCourtAdmissionExperiment
                ? (opportunities >= LastCarveStats[169]
                    ? opportunities - LastCarveStats[169] : 0u)
                : LastCarveStats[181];
            int acceptedPercent = opportunities > 0u
                ? Mathf.Clamp(Mathf.RoundToInt(accepted * 100f / opportunities), 0, 100)
                : 0;
            string court = enableFinalCourtAdmissionExperiment
                ? $"裁GPU 过{FormatCarveCount(LastCarveStats[169])} " +
                  $"待/无{FormatCarveCount(LastCarveStats[173])} " +
                  $"弃/反{FormatCarveCount(LastCarveStats[175])} " +
                  $"层/配{FormatCarveCount(LastCarveStats[176])}\n"
                : string.Empty;
            return court + $"融写 机{FormatCarveCount(opportunities)} 成{acceptedPercent:000}% " +
                   $"胶{FormatCarveCount(gunGelRejected)} 排{FormatCarveCount(LastCarveStats[182])} " +
                   $"法{FormatCarveCount(LastCarveStats[183])} 胀{FormatCarveCount(LastCarveStats[184])} " +
                   $"带{FormatCarveCount(LastCarveStats[185])}\n" +
                   $"融阻 弃{FormatCarveCount(LastCarveStats[186])} 弱{FormatCarveCount(LastCarveStats[187])} " +
                   $"动{FormatCarveCount(LastCarveStats[188])} 射{FormatCarveCount(LastCarveStats[189])} " +
                   $"借{FormatCarveCount(LastCarveStats[190])} 权{FormatCarveCount(LastCarveStats[191])} " +
                   $"缓{FormatCarveCount(LastCarveStats[195])} 吞{FormatCarveCount(LastCarveStats[196])} " +
                   $"冻{FormatCarveCount(LastCarveStats[197])}\n" +
                   $"冻修 未熟恢复{FormatCarveCount(LastCarveStats[198])}（非补齐）";
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
                   (LastCarveStats[91] > 0 ? $" 噪拦{FormatCarveCount(LastCarveStats[91])}" : "") +
                   (LastCarveStats[92] > 0 ? $" 折{FormatCarveCount(LastCarveStats[92])}" : "") +
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
        public void FlushForensicLedger(Action<bool> completed = null)
        {
            if (completed != null)
                _carveStatsFlushCallbacks.Add(completed);
            RequestCarveStatsReadback();
        }

        /// <summary>
        /// Flush both the production and raw-projective shadow counter periods.
        /// The callback runs only after every requested GPU readback settles.
        /// </summary>
        public void FlushAllForensicLedgers(Action<bool> completed)
        {
            int pending = ProjectiveShadowEnabled ? 2 : 1;
            bool success = true;
            void FinishOne(bool ok)
            {
                success &= ok;
                pending--;
                if (pending == 0) completed?.Invoke(success);
            }

            FlushForensicLedger(FinishOne);
            if (ProjectiveShadowEnabled)
            {
                _projectiveShadowFlushCallbacks.Add(FinishOne);
                RequestProjectiveShadowCarveStatsReadback();
            }
        }

        public void AppendAllForensicCountersCsv(StringBuilder sb, string session)
        {
            if (sb == null) return;
            sb.AppendLine("session,source,enabled,counter_index,value");
            for (int i = 0; i < CarveStatsCount; i++)
            {
                sb.Append(session).Append(",production,1,").Append(i).Append(',')
                    .Append(CumulativeCarveStats[i]).AppendLine();
                sb.Append(session).Append(",raw_projective_shadow,")
                    .Append(ProjectiveShadowEnabled ? 1 : 0).Append(',').Append(i).Append(',')
                    .Append(CumulativeProjectiveShadowCarveStats[i]).AppendLine();
            }
        }

        /// <summary>Append the cumulative, read-only integration causal ledger.</summary>
        public void AppendForensicLedgerReport(StringBuilder sb)
        {
            if (sb == null) return;
            static string U(ulong value) => value.ToString(CultureInfo.InvariantCulture);

            int infiniTamStoppedFrames = Mathf.Max(0,
                _infiniTamAttemptedFrames - _infiniTamFusedFrames);
            double lastAverageVotes = _infiniTamTicketSurfaceSamples > 0
                ? (double)_infiniTamTicketVoteSum / _infiniTamTicketSurfaceSamples
                : 0.0;
            double lastSingleFrameAuthority = _infiniTamTicketSurfaceSamples > 0
                ? 100.0 / (lastAverageVotes + 1.0)
                : 0.0;
            double cumulativeAverageVotes = _infiniTamTicketCumulativeSurfaceSamples > 0
                ? (double)_infiniTamTicketCumulativeVoteSum /
                  _infiniTamTicketCumulativeSurfaceSamples
                : 0.0;
            sb.AppendLine();
            sb.AppendLine("infinitam_baseline_ticket:");
            sb.AppendLine("scope=sampled_near_zero_surface_writes;diagnostic_only=true;sample_interval_s=1");
            sb.AppendLine($"enabled={enableInfiniTamBaseline.ToString().ToLowerInvariant()}");
            sb.AppendLine("active_contract=v1.3_external_quest_pose_direct_fusion_read_only_raycast");
            sb.AppendLine($"tracking_authority_enabled={enableInfiniTamTrackingAuthority.ToString().ToLowerInvariant()}");
            sb.AppendLine($"sample_ready={_hasInfiniTamTicket.ToString().ToLowerInvariant()}");
            sb.AppendLine($"readback_pending={_infiniTamTicketReadbackPending.ToString().ToLowerInvariant()}");
            sb.AppendLine($"sample_count={_infiniTamTicketSampleCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"attempted_frames={_infiniTamAttemptedFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"fused_frames={_infiniTamFusedFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"stopped_frames={infiniTamStoppedFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_new={_infiniTamTicketNew.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_continuing={_infiniTamTicketContinuing.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_mature={_infiniTamTicketMature.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_surface_samples={_infiniTamTicketSurfaceSamples.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_average_votes={lastAverageVotes.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_single_frame_authority_percent={lastSingleFrameAuthority.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"cumulative_new={U(_infiniTamTicketCumulativeNew)}");
            sb.AppendLine($"cumulative_continuing={U(_infiniTamTicketCumulativeContinuing)}");
            sb.AppendLine($"cumulative_mature={U(_infiniTamTicketCumulativeMature)}");
            sb.AppendLine($"cumulative_surface_samples={U(_infiniTamTicketCumulativeSurfaceSamples)}");
            sb.AppendLine($"cumulative_average_votes={cumulativeAverageVotes.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_bootstrap_frames={infiniTamTrackingBootstrapFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_required_still_frames={infiniTamBootstrapStillFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_max_angular_deg_per_s={infiniTamBootstrapMaxAngularDegPerSec.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_max_linear_m_per_s={infiniTamBootstrapMaxLinearMps.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_confirm_frames={infiniTamBootstrapConfirmFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_rejects_before_reseed={infiniTamBootstrapRejectsBeforeReseed.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"recovery_confirm_frames={infiniTamRecoveryConfirmFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_phase={GetInfiniTamStartupPhaseLabel()}");
            sb.AppendLine($"bootstrap_publication_ready={InfiniTamMeshPublicationReady.ToString().ToLowerInvariant()}");
            sb.AppendLine($"bootstrap_stable_frames={_infiniTamBootstrapStableFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_confirmed_frames={_infiniTamBootstrapConfirmedFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"recovery_confirmed_frames={_infiniTamRecoveryConfirmedFrames.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_consecutive_rejects={_infiniTamConsecutiveTrackingRejects.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"bootstrap_reseed_count={_infiniTamBootstrapReseedCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_min_correspondences={infiniTamTrackingMinCorrespondences.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_min_rank={infiniTamTrackingMinRank.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_max_translation_mm={infiniTamTrackingMaxTranslationMm.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_max_rotation_deg={infiniTamTrackingMaxRotationDeg.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_min_improvement_mm={infiniTamTrackingMinImprovementMm.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_min_within_30mm_percent={infiniTamTrackingMinWithin30Percent.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_accepted_frames={_infiniTamTrackingAccepted.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_rejected_frames={_infiniTamTrackingRejected.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_queue_abstained={_infiniTamTrackingQueueAbstained.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_last_decision={_infiniTamLastTrackingDecision}");
            _infiniTamModelRaycast?.AppendReport(sb);

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
            sb.AppendLine($"shell_witness_promotion_candidates={U(CumulativeCarveStats[177])}");
            sb.AppendLine($"shell_witness_fresh_hits={U(CumulativeCarveStats[178])}");
            sb.AppendLine($"shell_witness_promotions={U(CumulativeCarveStats[179])}");
            sb.AppendLine("relay_fusion_write_lifecycle:");
            sb.AppendLine("gungel_gpu_admission_status:");
            sb.AppendLine("note=status8_final_pending_merges_with_4_missing_stable;status9_final_abstain_merges_with_6_opposed;status10_final_layer_mismatch_merges_with_7_spatial_mismatch");
            sb.AppendLine($"evaluated={U(CumulativeCarveStats[168])}");
            sb.AppendLine($"passed={U(CumulativeCarveStats[169])}");
            sb.AppendLine($"invalid_observation={U(CumulativeCarveStats[170])}");
            sb.AppendLine($"raw_missing={U(CumulativeCarveStats[171])}");
            sb.AppendLine($"dual_conflict={U(CumulativeCarveStats[172])}");
            sb.AppendLine($"stable_missing_or_final_pending={U(CumulativeCarveStats[173])}");
            sb.AppendLine($"stable_dual_immature={U(CumulativeCarveStats[174])}");
            sb.AppendLine($"opposed_or_final_abstain={U(CumulativeCarveStats[175])}");
            sb.AppendLine($"spatial_or_final_layer_mismatch={U(CumulativeCarveStats[176])}");
            sb.AppendLine($"opportunities_near_surface={U(CumulativeCarveStats[180])}");
            ulong reconciledGunGelReject = enableFinalCourtAdmissionExperiment &&
                CumulativeCarveStats[180] >= CumulativeCarveStats[169]
                ? CumulativeCarveStats[180] - CumulativeCarveStats[169]
                : CumulativeCarveStats[181];
            sb.AppendLine($"reject_gungel={U(reconciledGunGelReject)}");
            sb.AppendLine($"reject_gungel_recorded_atomic={U(CumulativeCarveStats[181])}");
            sb.AppendLine($"reject_gungel_atomic_delta=" +
                $"{(long)CumulativeCarveStats[181] - (long)reconciledGunGelReject}");
            sb.AppendLine($"reject_exclusion={U(CumulativeCarveStats[182])}");
            sb.AppendLine($"reject_normal={U(CumulativeCarveStats[183])}");
            sb.AppendLine($"reject_dilation={U(CumulativeCarveStats[184])}");
            sb.AppendLine($"reject_truncation_band={U(CumulativeCarveStats[185])}");
            sb.AppendLine($"abstain_neighbour={U(CumulativeCarveStats[186])}");
            sb.AppendLine($"seed_support_low={U(CumulativeCarveStats[187])}");
            sb.AppendLine($"seed_motion_block={U(CumulativeCarveStats[188])}");
            sb.AppendLine($"raw_projective_block={U(CumulativeCarveStats[189])}");
            sb.AppendLine($"borrowed_dilation_block={U(CumulativeCarveStats[190])}");
            sb.AppendLine($"fov_motion_authority_abstain={U(CumulativeCarveStats[191])}");
            sb.AppendLine($"accepted_seed={U(CumulativeCarveStats[192])}");
            sb.AppendLine($"accepted_positive_update={U(CumulativeCarveStats[193])}");
            sb.AppendLine($"accepted_contradiction_carve={U(CumulativeCarveStats[194])}");
            sb.AppendLine($"accepted_still_provisional={U(CumulativeCarveStats[195])}");
            sb.AppendLine($"quantized_no_write={U(CumulativeCarveStats[196])}");
            sb.AppendLine($"frozen_observation_only={U(CumulativeCarveStats[197])}");
            sb.AppendLine($"immature_frozen_reopened_overlap={U(CumulativeCarveStats[198])}");
            AppendFovSampleLedgerSummary(sb);
        }

        private static readonly string[] FovLedgerRingNames = { "center", "mid", "outer" };
        private static readonly string[] TemporalLedgerReasonNames =
        {
            "disabled_or_unset", "first_frame", "current_invalid", "previous_fov_miss",
            "history_invalid", "history_hit_changed", "history_hit_stable"
        };

        private void AppendFovSampleLedgerSummary(StringBuilder sb)
        {
            static string U(ulong value) => value.ToString(CultureInfo.InvariantCulture);
            sb.AppendLine();
            sb.AppendLine("fov_temporal_sample_ledger:");
            sb.AppendLine("scope=production_near_surface_voxel_sample_opportunities;diagnostic_only=true");
            sb.AppendLine("aggregation=completed_gpu_readback_periods;the_last_partial_period_may_follow_in_the_later_frozen_report");
            sb.AppendLine("ring_definition=center<0.50;mid=0.50..0.84;outer>=0.84;radius=max(abs(2u-1),abs(2v-1))");
            for (int ring = 0; ring < 3; ring++)
            {
                sb.AppendLine($"examined_ring_{FovLedgerRingNames[ring]}={U(CumulativeCarveStats[100 + ring])}");
                sb.AppendLine($"positive_write_ring_{FovLedgerRingNames[ring]}={U(CumulativeCarveStats[103 + ring])}");
                sb.AppendLine($"applied_carve_ring_{FovLedgerRingNames[ring]}={U(CumulativeCarveStats[106 + ring])}");
                sb.AppendLine($"promotion_ring_{FovLedgerRingNames[ring]}={U(CumulativeCarveStats[109 + ring])}");
                sb.AppendLine($"demotion_ring_{FovLedgerRingNames[ring]}={U(CumulativeCarveStats[112 + ring])}");
            }
            for (int reason = 0; reason < 7; reason++)
            {
                string name = TemporalLedgerReasonNames[reason];
                sb.AppendLine($"examined_reason_{name}={U(CumulativeCarveStats[93 + reason])}");
                sb.AppendLine($"positive_write_reason_{name}={U(CumulativeCarveStats[115 + reason])}");
                sb.AppendLine($"applied_carve_reason_{name}={U(CumulativeCarveStats[122 + reason])}");
                sb.AppendLine($"outer_examined_reason_{name}={U(CumulativeCarveStats[129 + reason])}");
                sb.AppendLine($"outer_positive_write_reason_{name}={U(CumulativeCarveStats[136 + reason])}");
                sb.AppendLine($"outer_applied_carve_reason_{name}={U(CumulativeCarveStats[143 + reason])}");
            }
            for (int ring = 0; ring < 3; ring++)
            {
                string name = FovLedgerRingNames[ring];
                sb.AppendLine($"grazing_examined_ring_{name}={U(CumulativeCarveStats[150 + ring])}");
                sb.AppendLine($"grazing_applied_carve_ring_{name}={U(CumulativeCarveStats[153 + ring])}");
                sb.AppendLine($"grazing_positive_write_ring_{name}={U(CumulativeCarveStats[156 + ring])}");
                sb.AppendLine($"front_examined_ring_{name}={U(CumulativeCarveStats[159 + ring])}");
                sb.AppendLine($"front_applied_carve_ring_{name}={U(CumulativeCarveStats[162 + ring])}");
                sb.AppendLine($"front_positive_write_ring_{name}={U(CumulativeCarveStats[165 + ring])}");
            }
            double centerContradictionRate = LedgerRate(CumulativeCarveStats[106], CumulativeCarveStats[100]);
            double outerContradictionRate = LedgerRate(CumulativeCarveStats[108], CumulativeCarveStats[102]);
            double frontCenterContradictionRate = LedgerRate(CumulativeCarveStats[162], CumulativeCarveStats[159]);
            double frontOuterContradictionRate = LedgerRate(CumulativeCarveStats[164], CumulativeCarveStats[161]);
            double outerPrevFovMissRate = LedgerRate(CumulativeCarveStats[146], CumulativeCarveStats[132]);
            double outerStableRate = LedgerRate(CumulativeCarveStats[149], CumulativeCarveStats[135]);
            double outerCenterRatio = LedgerRatio(outerContradictionRate, centerContradictionRate);
            double frontOuterCenterRatio = LedgerRatio(frontOuterContradictionRate, frontCenterContradictionRate);
            double missStableRatio = LedgerRatio(outerPrevFovMissRate, outerStableRate);
            sb.AppendLine($"center_applied_carve_rate={centerContradictionRate.ToString("F6", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"outer_applied_carve_rate={outerContradictionRate.ToString("F6", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"outer_vs_center_applied_carve_ratio={LedgerNumber(outerCenterRatio)}");
            sb.AppendLine($"front_outer_vs_center_applied_carve_ratio={LedgerNumber(frontOuterCenterRatio)}");
            sb.AppendLine($"outer_prev_fov_miss_vs_stable_applied_carve_ratio={LedgerNumber(missStableRatio)}");
            sb.AppendLine($"edge_hypothesis_verdict={FovLedgerVerdict(outerCenterRatio, frontOuterCenterRatio, missStableRatio)}");
            sb.AppendLine("edge_hypothesis_note=association_only;front_facing_and_temporal_controls_reduce_grazing_confound;does_not_prove_display_lens_distortion");
            sb.AppendLine($"period_count={_fovLedgerPeriods.Count.ToString(CultureInfo.InvariantCulture)}");
        }

        private static double LedgerRate(ulong numerator, ulong denominator)
        {
            return denominator > 0 ? (double)numerator / denominator : double.NaN;
        }

        private static double LedgerRatio(double numeratorRate, double denominatorRate)
        {
            if (double.IsNaN(numeratorRate) || double.IsNaN(denominatorRate))
                return double.NaN;
            if (denominatorRate > 0d)
                return numeratorRate / denominatorRate;
            return numeratorRate > 0d ? double.PositiveInfinity : double.NaN;
        }

        private static string LedgerNumber(double value)
        {
            if (double.IsNaN(value)) return "insufficient";
            if (double.IsPositiveInfinity(value)) return "infinite";
            return value.ToString("F3", CultureInfo.InvariantCulture);
        }

        private string FovLedgerVerdict(double outerCenter, double frontOuterCenter, double missStable)
        {
            if (CumulativeCarveStats[100] < 1000 || CumulativeCarveStats[102] < 1000)
                return "insufficient_samples";
            bool outerElevated = !double.IsNaN(outerCenter) && outerCenter >= 1.5;
            bool survivesFacingControl = !double.IsNaN(frontOuterCenter) && frontOuterCenter >= 1.5;
            bool temporalMissElevated = !double.IsNaN(missStable) && missStable >= 1.5;
            if (outerElevated && (survivesFacingControl || temporalMissElevated))
                return "edge_association_supported";
            if (outerElevated)
                return "edge_association_grazing_confound_not_closed";
            return "edge_association_not_supported";
        }

        public void AppendFovSampleLedgerCsv(StringBuilder sb, string session)
        {
            if (sb == null) return;
            sb.AppendLine("session,period,utc,elapsed_s,scope,outcome,ring,temporal_reason,count");
            AppendFovSampleLedgerCsvRows(sb, session, -1, "", -1f, CumulativeCarveStats);
            for (int p = 0; p < _fovLedgerPeriods.Count; p++)
            {
                FovLedgerPeriod period = _fovLedgerPeriods[p];
                var expanded = new ulong[CarveStatsCount];
                for (int i = 93; i < CarveStatsCount; i++)
                    expanded[i] = period.Counters[i - 93];
                AppendFovSampleLedgerCsvRows(
                    sb, session, period.Index,
                    period.Utc.ToString("O", CultureInfo.InvariantCulture),
                    period.ElapsedSeconds, expanded);
            }
        }

        private static void AppendFovSampleLedgerCsvRows(
            StringBuilder sb, string session, int period, string utc,
            float elapsedSeconds, IReadOnlyList<ulong> counters)
        {
            string elapsed = elapsedSeconds < 0f
                ? ""
                : elapsedSeconds.ToString("F3", CultureInfo.InvariantCulture);
            void Row(string scope, string outcome, string ring, string reason, ulong count)
            {
                sb.Append(session).Append(',').Append(period).Append(',').Append(utc).Append(',').Append(elapsed).Append(',')
                  .Append(scope).Append(',').Append(outcome).Append(',').Append(ring).Append(',').Append(reason).Append(',')
                  .Append(count).AppendLine();
            }
            for (int ring = 0; ring < 3; ring++)
            {
                string name = FovLedgerRingNames[ring];
                Row("ring", "examined", name, "all", counters[100 + ring]);
                Row("ring", "positive_write", name, "all", counters[103 + ring]);
                Row("ring", "applied_carve", name, "all", counters[106 + ring]);
                Row("ring", "promotion", name, "all", counters[109 + ring]);
                Row("ring", "demotion", name, "all", counters[112 + ring]);
                Row("grazing", "examined", name, "all", counters[150 + ring]);
                Row("grazing", "applied_carve", name, "all", counters[153 + ring]);
                Row("grazing", "positive_write", name, "all", counters[156 + ring]);
                Row("front", "examined", name, "all", counters[159 + ring]);
                Row("front", "applied_carve", name, "all", counters[162 + ring]);
                Row("front", "positive_write", name, "all", counters[165 + ring]);
            }
            for (int reason = 0; reason < 7; reason++)
            {
                string name = TemporalLedgerReasonNames[reason];
                Row("reason", "examined", "all", name, counters[93 + reason]);
                Row("reason", "positive_write", "all", name, counters[115 + reason]);
                Row("reason", "applied_carve", "all", name, counters[122 + reason]);
                Row("outer_cross", "examined", "outer", name, counters[129 + reason]);
                Row("outer_cross", "positive_write", "outer", name, counters[136 + reason]);
                Row("outer_cross", "applied_carve", "outer", name, counters[143 + reason]);
            }
            Row("gungel_admission", "examined", "all", "all", counters[168]);
            Row("gungel_admission", "witness_accepted", "all", "all", counters[169]);
            Row("gungel_admission", "abstained_allow_raw", "all", "observation_invalid", counters[170]);
            Row("gungel_admission", "abstained_allow_raw", "all", "raw_unavailable", counters[171]);
            Row("gungel_admission", "blocked_explicit_conflict", "all", "dual_disagree", counters[172]);
            Row("gungel_admission", "abstained_allow_raw", "all", "stable_candidate_missing", counters[173]);
            Row("gungel_admission", "abstained_allow_raw", "all", "stable_dual_immature", counters[174]);
            Row("gungel_admission", "blocked_explicit_conflict", "all", "candidate_opposed", counters[175]);
            Row("gungel_admission", "blocked_explicit_conflict", "all", "pixel_cell_mismatch", counters[176]);
            Row("shell_witness", "examined", "all", "rescued_provisional", counters[177]);
            Row("shell_witness", "fresh_hit", "all", "independent_verified", counters[178]);
            Row("shell_witness", "promotion", "all", "second_raw_confirmed", counters[179]);
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

            if (enableProjectiveShadow && !enableInfiniTamBaseline)
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

            // The baseline must not reuse the product route's R8G8_SNorm .g
            // lane as both "may extract" and accumulated observation weight.
            // Keep a real (0..100) vote count in a baseline-private float volume.
            // The old route receives a 1x1 placeholder solely because Vulkan
            // requires every UAV declared by Integrate to have a descriptor.
            GraphicsFormat infiniTamVoteFormat = SystemInfo.IsFormatSupported(
                GraphicsFormat.R16_SFloat, FormatUsage.LoadStore)
                ? GraphicsFormat.R16_SFloat
                : GraphicsFormat.R32_SFloat;
            int voteWidth = enableInfiniTamBaseline ? voxelCount.x : 1;
            int voteHeight = enableInfiniTamBaseline ? voxelCount.y : 1;
            int voteDepth = enableInfiniTamBaseline ? voxelCount.z : 1;
            _infiniTamVoteWeightVolume = new RenderTexture(
                voteWidth, voteHeight, 0, infiniTamVoteFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voteDepth,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = enableInfiniTamBaseline
                    ? "InfiniTamAccumulationWeight"
                    : "InfiniTamAccumulationWeightDummy"
            };
            _infiniTamVoteWeightVolume.Create();
            if (enableInfiniTamBaseline)
            {
                long voteBytesPerVoxel = infiniTamVoteFormat == GraphicsFormat.R16_SFloat
                    ? 2L
                    : 4L;
                Logger.Info($"InfiniTAM private vote volume: {voxelCount} " +
                            $"{infiniTamVoteFormat} = " +
                            $"{(voteBytesPerVoxel * voxelCount.x * voxelCount.y * voxelCount.z) / (1024 * 1024)}MB");
            }

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

            // 置信度通道 v1：与 admission trace 同格式（R8 归一化存分歧 EMA，
            // 不支持时退 R16_SFloat）。只读影子，生产路径一律不读它。
            _confidenceVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, traceFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "ConfidenceDivergence"
            };
            _confidenceVolume.Create();

            // v2 相干通道：同格式第二张 R8（0.5 偏置有符号分歧 EMA）。
            _coherenceVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, traceFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "ConfidenceCoherence"
            };
            _coherenceVolume.Create();

            if (!enableInfiniTamBaseline)
            {
                // Two packed uint lanes are sufficient to connect a final
                // zero-crossing endpoint to its current-lifetime seed, last
                // sd-moving write and strongest blocked correction. These
                // diagnostics belong only to the legacy product route.
                _tsdfResponsibilityVolume = new RenderTexture(
                    voxelCount.x, voxelCount.y, 0,
                    GraphicsFormat.R32G32_UInt, 0)
                {
                    dimension = TextureDimension.Tex3D,
                    volumeDepth = voxelCount.z,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "TsdfResponsibility"
                };
                _tsdfResponsibilityVolume.Create();
                Logger.Info($"TSDF responsibility sidecar: {voxelCount} " +
                            $"R32G32_UInt = " +
                            $"{(8L * voxelCount.x * voxelCount.y * voxelCount.z) / (1024 * 1024)}MB");

                _tsdfSupportResponsibilityVolume = new RenderTexture(
                    voxelCount.x, voxelCount.y, 0,
                    GraphicsFormat.R32_UInt, 0)
                {
                    dimension = TextureDimension.Tex3D,
                    volumeDepth = voxelCount.z,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "TsdfSupportResponsibility"
                };
                _tsdfSupportResponsibilityVolume.Create();
                Logger.Info($"TSDF support responsibility sidecar: " +
                            $"{voxelCount} R32_UInt = " +
                            $"{(4L * voxelCount.x * voxelCount.y * voxelCount.z) / (1024 * 1024)}MB");
            }
            else
            {
                Logger.Info("InfiniTAM compact resources: skipped legacy " +
                            "responsibility sidecars and freeze/page ledgers.");
            }
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
            Vector3 frustumPhase = enableFrustumPhaseCoverage
                ? FrustumCoveragePhases[IntegrationCount & 7] * voxelSize
                : Vector3.zero;
            compute.SetVector(FrustumPhaseOffsetID, frustumPhase);
            compute.SetFloat(DiagnosticAngularSpeedID, 0f);
            compute.SetFloat(NoiseMotionQualityID, 1f);
            compute.SetFloat(MotionAuthorityQualityID, 1f);
            compute.SetFloat(MotionConfirmQualityMinID, motionConfirmQualityMin);
            compute.SetFloat(NoiseDistExpID, noiseDistExponent);
            compute.SetFloat(NoiseAngExpID, noiseAngleExponent);
            compute.SetFloat(MatureObsEnableID, enableMatureSurfaceObsDiscount ? 1f : 0f);
            compute.SetFloat(MatureObsWeightMinID, matureSurfaceObsWeightMin);
            compute.SetFloat(MatureObsMarginID, matureSurfaceObsMargin);
            compute.SetFloat(MatureObsDiscountID, matureSurfaceObsDiscount);
            compute.SetFloat(FrozenBlockEnableID, frozenBlockEnable ? 1f : 0f);
            compute.SetFloat(FrozenVoteQualityMinID, frozenVoteQualityMin);
            compute.SetFloat(FrozenVoteMarginID, frozenVoteMargin);
            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(InfiniTamBaselineID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            compute.SetFloat(TsdfResponsibilityAvailableID,
                _tsdfResponsibilityVolume != null && _tsdfResponsibilityVolume.IsCreated() ? 1f : 0f);
            compute.SetFloat(TsdfResponsibilityWriteID,
                _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
            compute.SetInt(TsdfResponsibilityIntegrationID, 0);
            compute.SetFloat(ConfidenceWriteID, 1f);
            compute.SetFloat(ConfidenceRateID, confidenceRate);
            compute.SetFloat(ConfidenceMidMaxID, confidenceMidMax);
            compute.SetFloat(ConfidenceLowMinID, confidenceLowMin);
            ConfigureDirtyTracking(false);

            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);
            Shader.SetGlobalTexture(ConfidenceGlobalTexID, _confidenceVolume);
            Shader.SetGlobalFloat(ConfidenceMidMaxID, confidenceMidMax);
            Shader.SetGlobalFloat(ConfidenceLowMinID, confidenceLowMin);
        }

        /// <summary>
        /// Zeros the TSDF and color volumes on the GPU. No-op if volumes
        /// haven't been allocated yet (lazy alloc — see
        /// <see cref="ReallocateVolumes"/>).
        /// </summary>
        public void Clear()
        {
            ClearInternal(preserveGunGelEvidence: false);
        }

        private void ClearInternal(bool preserveGunGelEvidence)
        {
            if (_volume == null || _clearKernel.Shader == null) return;

            _pruneCycleActive = false;
            _nextPruneSlice = 0;

            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            compute.SetFloat(ConfidenceWriteID, 1f);
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.Set(AdmissionTraceRWID, _admissionTraceVolume);
            _clearKernel.DispatchFit(_volume);
            ResetInfiniTamTicket();
            _infiniTamModelRaycast?.Reset();
            ResetInfiniTamDeferredFrame(false);
            if (enableInfiniTamBaseline)
                ResetInfiniTamStartupState(preserveReseedCount: true);
            if (enableInfiniTamBaseline && _infiniTamVoteWeightVolume != null)
            {
                _clearInfiniTamVotesKernel.Set(InfiniTamVoteWeightRWID,
                    _infiniTamVoteWeightVolume);
                _clearInfiniTamVotesKernel.DispatchFit(_infiniTamVoteWeightVolume);
            }

            if (_projectiveShadowVolume != null)
            {
                compute.SetFloat(UseRawProjectiveSdfID, 1f);
                compute.SetFloat(WriteColorID, 0f);
                compute.SetFloat(WriteAdmissionTraceID, 0f);
                compute.SetFloat(ConfidenceWriteID, 0f);
                _clearKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _clearKernel.Set(ColorVolumeRWID, _colorVolume); // bound but guarded from writes
                _clearKernel.DispatchFit(_projectiveShadowVolume);
            }

            // Restore production defaults for every unrelated kernel/caller.
            compute.SetFloat(UseRawProjectiveSdfID, 0f);
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            compute.SetFloat(ConfidenceWriteID, 1f);
            _clearKernel.Set(VolumeRWID, _volume);
            _carveStats?.SetData(ZeroCarveStats);
            _projectiveShadowCarveStats?.SetData(ZeroCarveStats);
            // 清卷同时清冻结账：旧票箱/成熟度指向已销毁内容，必须同步归零。
            if (_frozenChunkVotes != null) _frozenChunkVotes.SetData(_voteZeros);
            if (_chunkMaturity != null) _chunkMaturity.SetData(_maturityZeros);
            Array.Clear(CumulativeCarveStats, 0, CumulativeCarveStats.Length);
            Array.Clear(CumulativeProjectiveShadowCarveStats, 0,
                CumulativeProjectiveShadowCarveStats.Length);
            _fovLedgerPeriods.Clear();
            _fovLedgerPeriodIndex = 0;
            _fovLedgerStartedRealtime = Time.realtimeSinceStartup;
            HasCarveStats = false;
            HasProjectiveShadowCarveStats = false;
            // 用户清卷/重定位时证据与体素一起失效；暖机结束只丢传感器启动期
            // TSDF，不得把刚养成的枪胶候选同时抹掉，否则会制造第二次冷启动
            // 并再次让欠秩/少配拖住覆盖。
            if (!preserveGunGelEvidence)
            {
                _gunGelEvidenceShadow?.Clear();
                ResetProductSurfaceCourt();
            }
            ResetGunGelDeferredFrames(false);
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
            ResetInfiniTamTicket();
            _infiniTamModelRaycast?.Reset();
            ResetInfiniTamDeferredFrame(false);
            ResetInfiniTamStartupState(preserveReseedCount: false);
            _infiniTamTrackingAccepted = 0;
            _infiniTamTrackingRejected = 0;
            _infiniTamTrackingQueueAbstained = 0;
            _infiniTamLastTrackingDecision = "建模";
            _gunGelCaptureFrameIndex = 0;
            _gunGelFusionAccepted = 0;
            _gunGelFusionRejected = 0;
            _gunGelFusionRawFallback = 0;
            _gunGelFusionQueueAbstained = 0;
            _gunGelLastAppliedMm = 0f;
            _gunGelLastFusionDecision = "预热";
            _gunGelGuardedFusionRuntimeHalted = false;
            _fovLedgerPeriods.Clear();
            _fovLedgerPeriodIndex = 0;
            _fovLedgerStartedRealtime = Time.realtimeSinceStartup;
            ResetGunGelDeferredFrames(false);
        }

        /// <summary>
        /// Resample TSDF + color from the current (relocated) grid into a new identity grid.
        /// After this call the volume data lives in the current tracking/world frame.
        /// </summary>
        public void BakeRelocation(Matrix4x4 relocationMatrix)
        {
            if (_volume == null || _colorVolume == null || compute == null)
                return;
            // The active V1.3 baseline fuses directly in Quest world pose and
            // never relocates its volume. The legacy relocation kernel also
            // owns responsibility sidecars, so it must not recreate that
            // dormant route behind the compact baseline's back.
            if (enableInfiniTamBaseline)
            {
                Logger.Warning("BakeRelocation ignored: compact InfiniTAM " +
                               "uses direct Quest-world fusion.");
                return;
            }

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

            var dstResponsibility = new RenderTexture(
                vc.x, vc.y, 0, GraphicsFormat.R32G32_UInt, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "TsdfResponsibility"
            };
            dstResponsibility.Create();

            var dstSupportResponsibility = new RenderTexture(
                vc.x, vc.y, 0, GraphicsFormat.R32_UInt, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "TsdfSupportResponsibility"
            };
            dstSupportResponsibility.Create();

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
            compute.SetTexture(kernel, BakeSrcTsdfResponsibilityID,
                _tsdfResponsibilityVolume);
            compute.SetTexture(kernel, BakeSrcTsdfSupportResponsibilityID,
                _tsdfSupportResponsibilityVolume);
            compute.SetTexture(kernel, VolumeRWID, dstTsdf);
            compute.SetTexture(kernel, ColorVolumeRWID, dstColor);
            compute.SetTexture(kernel, AdmissionTraceRWID, dstAdmissionTrace);
            compute.SetTexture(kernel, TsdfResponsibilityRWID, dstResponsibility);
            compute.SetTexture(kernel, TsdfSupportResponsibilityRWID,
                dstSupportResponsibility);
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
                compute.SetFloat(TsdfResponsibilityAvailableID, 0f);
                compute.Dispatch(kernel, tx, ty, tz);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
                compute.SetFloat(TsdfResponsibilityAvailableID, 1f);
            }
            GL.Flush();

            // Swap volumes: destroy old, adopt baked textures.
            // Avoids Graphics.CopyTexture on 3D RTs which can silently fail on Vulkan/Quest.
            Destroy(_volume);
            Destroy(_colorVolume);
            if (_projectiveShadowVolume) Destroy(_projectiveShadowVolume);
            if (_admissionTraceVolume) Destroy(_admissionTraceVolume);
            if (_tsdfResponsibilityVolume) Destroy(_tsdfResponsibilityVolume);
            if (_tsdfSupportResponsibilityVolume) Destroy(_tsdfSupportResponsibilityVolume);
            _volume = dstTsdf;
            _colorVolume = dstColor;
            _projectiveShadowVolume = dstProjectiveShadow;
            _admissionTraceVolume = dstAdmissionTrace;
            _tsdfResponsibilityVolume = dstResponsibility;
            _tsdfSupportResponsibilityVolume = dstSupportResponsibility;

            // Rebind global texture references (used by render shader for freeze tint etc.)
            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);

            // Rebind per-kernel UAV references so subsequent integrations/clears use new textures
            RebindVolumeTextures();
            // 枪胶候选使用世界坐标稀疏哈希；重定位后旧键不可复用，宁可重新预热也不混坐标系。
            _gunGelEvidenceShadow?.Clear();
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
        /// Freeze mature voxels currently visible in the camera frustum.
        /// Immature supports remain writable; mature frozen weights are negative.
        /// Requires camera data to have been provided via SetCameraData.
        /// </summary>
        public void FreezeInView(Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (enableInfiniTamBaseline) return;
            if (_volume == null || _freezeKernel.Shader == null)
            {
                Logger.Warning("FreezeInView called before GPU resources allocated; ignored.");
                return;
            }
            SetFrustumCameraUniforms(_freezeKernel, camPos, camRot,
                focalLen, principalPt, sensorRes, currentRes);
            ConfigureFreezeSupportWeight();
            _freezeKernel.Set(VolumeRWID, _volume);
            _freezeKernel.DispatchFit(_volume);
            if (_projectiveShadowVolume != null)
            {
                compute.SetFloat(TsdfResponsibilityWriteID, 0f);
                _freezeKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _freezeKernel.DispatchFit(_projectiveShadowVolume);
                _freezeKernel.Set(VolumeRWID, _volume);
                compute.SetFloat(TsdfResponsibilityWriteID,
                    _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
            }
            Logger.Info("FreezeInView dispatched");
        }

        /// <summary>
        /// Unfreeze all frozen voxels currently visible in the camera frustum.
        /// </summary>
        public void UnfreezeInView(Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (enableInfiniTamBaseline) return;
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
                compute.SetFloat(TsdfResponsibilityWriteID, 0f);
                _unfreezeKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _unfreezeKernel.DispatchFit(_projectiveShadowVolume);
                _unfreezeKernel.Set(VolumeRWID, _volume);
                compute.SetFloat(TsdfResponsibilityWriteID,
                    _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
            }
            Logger.Info("UnfreezeInView dispatched");
        }

        // ── 逐块可逆冻结公共 API ─────────────────────────────────────
        /// <summary>冻结票箱（每块 uint4：x=自由空间票 y=遮挡票 z=补洞票 w=保留），调度器周期回读。</summary>
        public ComputeBuffer FrozenChunkVotes => _frozenChunkVotes;

        /// <summary>
        /// T2：下发当前已冻块全量位图（1 位/块，调度器在每次冻结/解冻裁决后调用）。
        /// 补洞票的块冻结态判据——冻块内空体素继续积分，种子出生只在该位图置位的
        /// 块里记票。与 set/clear 一次性掩码不同，本位图是持续状态、全量覆盖。
        /// </summary>
        public void SetFrozenChunkBits(uint[] bits)
        {
            if (!FrozenBlockReady || bits == null || _frozenChunkBits == null) return;
            if (bits.Length != _frozenChunkBits.count) return;
            _frozenChunkBits.SetData(bits);
        }
        /// <summary>成熟度账（每块 uint4：x=surface体素数 y=其中已冻结），调度器周期回读。</summary>
        public ComputeBuffer ChunkMaturity => _chunkMaturity;
        /// <summary>冻结 API 是否可用（GPU 资源已惰性分配）。</summary>
        public bool FrozenBlockReady => !enableInfiniTamBaseline &&
                                        _volume != null && _frozenChunkVotes != null &&
                                        _applyFreezeMaskKernel.Shader != null;
        /// <summary>冻结单元块总数（独立于脏账本网格，frozenChunkSize 边长）。</summary>
        public int FrozenBlockCount => _frozenChunkCount.x * _frozenChunkCount.y * _frozenChunkCount.z;
        /// <summary>冻结块网格维度（供调度器解码块坐标）。</summary>
        public int3 FrozenChunkCount => _frozenChunkCount;

        /// <summary>
        /// 上传 set/clear 块位图并翻符号：set=只冻达到逐体素门槛的正权重，clear=负→正，
        /// 解冻块票箱同批清零。主卷与影子卷各执行一次。掩码一次性消费——返回时两个
        /// 传入数组已被清零，GPU 侧掩码同步复位，残留位不会误伤下一批。
        /// 提取层 abs(weight) 对符号透明：本调用不标脏、不触发重提网格。
        /// </summary>
        public void ApplyChunkFreezeMasks(uint[] setMask, uint[] clearMask)
        {
            if (!FrozenBlockReady || setMask == null || clearMask == null) return;
            ConfigureFreezeSupportWeight();
            _chunkFreezeSetMask.SetData(setMask);
            _chunkFreezeClearMask.SetData(clearMask);
            _applyFreezeMaskKernel.DispatchFit(_volume);
            if (_projectiveShadowVolume != null)
            {
                compute.SetFloat(TsdfResponsibilityWriteID, 0f);
                _applyFreezeMaskKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _applyFreezeMaskKernel.DispatchFit(_projectiveShadowVolume);
                _applyFreezeMaskKernel.Set(VolumeRWID, _volume);
                compute.SetFloat(TsdfResponsibilityWriteID,
                    _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
            }
            _clearVotesKernel.DispatchFit(_frozenChunkVotes.count, 1, 1);
            System.Array.Clear(setMask, 0, setMask.Length);
            System.Array.Clear(clearMask, 0, clearMask.Length);
            _chunkFreezeSetMask.SetData(setMask);
            _chunkFreezeClearMask.SetData(clearMask);
        }

        private void ConfigureFreezeSupportWeight()
        {
            compute.SetFloat(FrozenMatureWeightID, frozenMatureWeight);
            compute.SetFloat(DirtyMinWeightID, minMeshWeight);
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

            Matrix4x4 depthProj = Shader.GetGlobalMatrixArray(DepthCapture.ProjID)[DepthCapture.FusionEyeIndex];
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
        public bool Integrate()
        {
            var dc = DepthCapture.Instance;
            if (dc == null || !DepthCapture.DepthAvailable || dc.DepthTex == null) return false;
            // The observation view still shows raw source geometry. Do not
            // write those pre-seed frames while it is open. Once a fitted seed
            // is staged, an existing roll must be cleared before the shared
            // GunGel -> court -> TSDF path may consume the corrected input.
            if (!enableInfiniTamBaseline &&
                (dc.SeedPlanePreviewActive || dc.SeedPlaneAwaitClear ||
                 !dc.SeedPlaneAppliedToCurrentFrame)) return false;
            // Defensive: with lazy GPU alloc a stray Integrate() before
            // ReallocateVolumes can land here. RoomScanner.StartScanning()
            // always calls ReallocateVolumes first, so this is just a
            // safety net.
            if (_volume == null || _integrateKernel.Shader == null) return false;
            if (!_frustumReady) SetupFrustumVolume();
            if (!_frustumReady) return false;
            int replayAttemptIndex = ++_replayFusionAttemptIndex;

            Texture fusionDepth = dc.DepthTex;
            Texture fusionNormal = dc.NormTex;
            Texture fusionDilatedDepth = dc.DilatedDepthTex;
            Texture fusionEdgeReason = dc.EdgeReasonTex;
            Texture fusionTemporalReason = dc.TemporalReasonTex;
            bool fusionTemporalReasonAvailable = fusionTemporalReason != null;
            Matrix4x4[] fusionView = dc.View;
            Matrix4x4[] fusionProjection = dc.Proj;
            Matrix4x4[] fusionViewInverse = dc.ViewInv;
            Matrix4x4[] fusionProjectionInverse = dc.ProjInv;
            GunGelDeferredFrame deferredFrame = null;
            InfiniTamDeferredFrame infiniTamFrame = null;
            bool usingGuardedFrame = false;
            bool usingInfiniTamTrackedFrame = false;
            bool gunGelCorrectionApplied = false;
            bool infiniTamCorrectionApplied = false;
            bool gunGelIdentityAccepted = false;
            string acceptedInputReason = "baseline_accept";

            // 运动闸：转头时积分位姿与深度帧存在帧差，写入会切向涂抹成搓衣板褶皱、
            // 矛盾票也会按错位投影啃到真表面。超阈值整帧停笔（不集成、不扣减），
            // 停下来正对目标时票照投——消幽灵的姿势是"停住看"，不是"转着磨"。
            // 角速度由 DepthCapture 在深度帧到达时用原始 Pose 四元数计算
            // （dt=真实深度帧间隔），这里只消费。旧实现用积分间隔 ÷ 矩阵.rotation
            // 增量：深度帧率低于积分率时系统性放大，且 ScaleFlipZ 负行列式矩阵的
            // 四元数提取有分支不连续风险——曾致诊断运动位 100% 饱和失效。
            float currentAngularSpeed = dc.SmoothedDepthAngularSpeed;
            float currentLinearSpeed = dc.SmoothedDepthLinearSpeed;
            _smoothedAngSpeed = currentAngularSpeed;
            // 运动质量同时是融合降权和正式发布权限的共同事实源。即使关闭
            // “噪声模型加权”，保护罩仍必须知道当前运动状态，不能退化为满权。
            float motionAngularRatio = noiseMotionAngRefDegPerSec > 0f
                ? currentAngularSpeed / noiseMotionAngRefDegPerSec : 0f;
            float motionLinearRatio = noiseMotionLinRefMps > 0f
                ? currentLinearSpeed / noiseMotionLinRefMps : 0f;
            _motionQuality = Mathf.Lerp(1f, noiseMotionFloor,
                Mathf.Clamp01(Mathf.Max(motionAngularRatio, motionLinearRatio)));

            // Restored V1.3 boundary: Quest owns pose and each platform depth
            // frame may update the sole TSDF at most once. Raycast remains a
            // read-only witness below; it cannot retain the frame, modify pose,
            // gate fusion, clear the model or delay mesh publication.
            if (enableInfiniTamBaseline && !enableInfiniTamTrackingAuthority)
            {
                int platformFrame = dc.CurrentPlatformFrame;
                if (platformFrame >= 0 &&
                    platformFrame == _infiniTamLastStartupPlatformFrame)
                    return false;
                if (platformFrame >= 0)
                    _infiniTamLastStartupPlatformFrame = platformFrame;
                acceptedInputReason = "infinitam_v13_external_pose";
                _infiniTamLastTrackingDecision = "只读旁证";
            }

            bool currentHardGated = motionGateDegPerSec > 0f &&
                                    currentAngularSpeed > motionGateDegPerSec;
            bool infiniTamStartupActive = enableInfiniTamBaseline &&
                                          enableInfiniTamTrackingAuthority &&
                                          !_infiniTamTrackingInitialised;
            bool startupNeedsLiveCandidate = infiniTamStartupActive &&
                (IntegrationCount < Mathf.Max(1,
                     infiniTamTrackingBootstrapFrames) ||
                 (!_infiniTamDeferredFrame.Pending &&
                  !_infiniTamDeferredFrame.Ready));
            if (startupNeedsLiveCandidate)
            {
                // A depth texture can remain current across several Unity
                // updates. Counting those polls as independent bootstrap
                // evidence used to fill the seed quota with one frozen frame.
                if (dc.CurrentPlatformFrame ==
                    _infiniTamLastStartupPlatformFrame)
                    return false;
                _infiniTamLastStartupPlatformFrame = dc.CurrentPlatformFrame;

                if (!IsInfiniTamBootstrapMotionSafe(currentAngularSpeed,
                        currentLinearSpeed))
                {
                    _infiniTamBootstrapStableFrames = 0;
                    _infiniTamStartupPhase =
                        InfiniTamStartupPhase.AwaitingStillness;
                    _infiniTamLastTrackingDecision = "等稳";
                    _motionGatedSinceStats++;
                    _pendingCamFrame = null;
                    ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                        replayAttemptIndex, dc.CurrentPlatformFrame, false,
                        "infinitam_bootstrap_motion_hold", false, -1, 0f, 0f,
                        currentAngularSpeed, currentLinearSpeed, _motionQuality);
                    return false;
                }

                _infiniTamBootstrapStableFrames++;
                if (_infiniTamBootstrapStableFrames <
                    Mathf.Max(1, infiniTamBootstrapStillFrames))
                {
                    _infiniTamStartupPhase =
                        InfiniTamStartupPhase.AwaitingStillness;
                    _infiniTamLastTrackingDecision = "等稳";
                    _pendingCamFrame = null;
                    return false;
                }

                if (IntegrationCount < Mathf.Max(1,
                        infiniTamTrackingBootstrapFrames))
                {
                    _infiniTamStartupPhase = InfiniTamStartupPhase.Seeding;
                    acceptedInputReason = "infinitam_stable_unique_bootstrap";
                }
            }
            bool guardedExperimentActive = !enableInfiniTamBaseline &&
                                           enableGunGelGuardedFusionExperiment &&
                                           !_gunGelGuardedFusionRuntimeHalted &&
                                           _gunGelEvidenceShadow != null;
            bool finalCourtAdmissionActive = guardedExperimentActive &&
                                             enableFinalCourtAdmissionExperiment;

            if (!currentHardGated)
            {
                bool compactDepthOnly = InfiniTamCompactDepthOnly;
                if (!compactDepthOnly)
                    dc.UpdateDilationIfNeeded();
                // The legacy/tracked routes create dilation lazily. Refresh the
                // complete input set after that creation. V1.3's compact kernel
                // samples only fusionDepth, so do not make unused side textures
                // a cold-start dependency or dispatch their producers.
                fusionDepth = dc.DepthTex;
                fusionNormal = dc.NormTex;
                fusionDilatedDepth = dc.DilatedDepthTex;
                fusionEdgeReason = dc.EdgeReasonTex;
                fusionTemporalReason = dc.TemporalReasonTex;
                fusionTemporalReasonAvailable = fusionTemporalReason != null;
                bool missingFullInput = !compactDepthOnly &&
                    (fusionNormal == null || fusionDilatedDepth == null ||
                     fusionEdgeReason == null);
                if (fusionDepth == null || missingFullInput)
                {
                    _pendingCamFrame = null;
                    return false;
                }
                if (guardedExperimentActive)
                {
                    QueueGunGelGuardedFrame(dc, currentAngularSpeed,
                        currentLinearSpeed, _motionQuality);
                    _pendingCamFrame = null; // 实验暂不延迟相机色帧，防跨帧贴错色。
                }
                else if (!enableInfiniTamBaseline)
                {
                    DispatchGunGelEvidenceShadow(dc);
                }
            }
            else
            {
                _motionGatedSinceStats++;
                _pendingCamFrame = null;
            }

            // 受保护实验必须消费“同一帧”的深度、法线、姿态和解算结果。
            // 最老帧尚在回读时宁可短暂停笔，绝不用上一帧校正硬套当前帧。
            bool infiniTamTrackingRequired = enableInfiniTamBaseline &&
                enableInfiniTamTrackingAuthority &&
                IntegrationCount >= Mathf.Max(1, infiniTamTrackingBootstrapFrames);
            if (infiniTamTrackingRequired)
            {
                if (!_infiniTamTrackingInitialised)
                    _infiniTamStartupPhase = InfiniTamStartupPhase.Verifying;
                infiniTamFrame = _infiniTamDeferredFrame;
                if (infiniTamFrame.Pending) return false;
                if (!infiniTamFrame.Ready)
                {
                    if (currentHardGated)
                    {
                        ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                            replayAttemptIndex, dc.CurrentPlatformFrame, false,
                            "infinitam_motion_gate", false, -1, 0f, 0f,
                            currentAngularSpeed, currentLinearSpeed, _motionQuality);
                        return false;
                    }
                    QueueInfiniTamTrackedFrame(dc, currentAngularSpeed,
                        currentLinearSpeed, _motionQuality);
                    _pendingCamFrame = null;
                    return false;
                }

                if (!AcceptInfiniTamTrackedFrame(infiniTamFrame,
                    out string trackingRejectReason))
                {
                    _infiniTamTrackingRejected++;
                    _infiniTamConsecutiveTrackingRejects++;
                    _infiniTamBootstrapConfirmedFrames = 0;
                    _infiniTamRecoveryConfirmedFrames = 0;
                    _infiniTamStartupPhase = _infiniTamTrackingInitialised
                        ? InfiniTamStartupPhase.TrackingLost
                        : InfiniTamStartupPhase.Verifying;
                    _infiniTamLastTrackingDecision = "拒" + trackingRejectReason;
                    ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                        replayAttemptIndex, infiniTamFrame.PlatformFrame, false,
                        "infinitam_tracking_reject:" + trackingRejectReason,
                        false, -1, infiniTamFrame.Decision.TranslationMm,
                        infiniTamFrame.Decision.RotationDeg,
                        infiniTamFrame.AngularSpeed, infiniTamFrame.LinearSpeed,
                        infiniTamFrame.MotionQuality);
                    ReleaseInfiniTamDeferredFrame();
                    TryReseedUnconfirmedInfiniTam(trackingRejectReason);
                    return false;
                }

                bool recoveringPublishedModel = _infiniTamTrackingInitialised &&
                    _infiniTamStartupPhase == InfiniTamStartupPhase.TrackingLost;
                if (recoveringPublishedModel)
                {
                    _infiniTamRecoveryConfirmedFrames++;
                    _infiniTamConsecutiveTrackingRejects = 0;
                    if (_infiniTamRecoveryConfirmedFrames <
                        Mathf.Max(1, infiniTamRecoveryConfirmFrames))
                    {
                        _infiniTamLastTrackingDecision = "重锁";
                        ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                            replayAttemptIndex, infiniTamFrame.PlatformFrame,
                            false, "infinitam_recovery_probation", false, -1,
                            infiniTamFrame.Decision.TranslationMm,
                            infiniTamFrame.Decision.RotationDeg,
                            infiniTamFrame.AngularSpeed,
                            infiniTamFrame.LinearSpeed,
                            infiniTamFrame.MotionQuality);
                        ReleaseInfiniTamDeferredFrame();
                        return false;
                    }
                }

                ApplyInfiniTamTrackingCorrection(infiniTamFrame);
                usingInfiniTamTrackedFrame = true;
                infiniTamCorrectionApplied = true;
                fusionDepth = infiniTamFrame.Depth;
                fusionNormal = infiniTamFrame.Normal;
                fusionDilatedDepth = infiniTamFrame.DilatedDepth;
                fusionEdgeReason = infiniTamFrame.EdgeReason;
                fusionTemporalReason = infiniTamFrame.TemporalReason;
                fusionTemporalReasonAvailable = fusionTemporalReason != null;
                fusionView = infiniTamFrame.View;
                fusionProjection = infiniTamFrame.Projection;
                fusionViewInverse = infiniTamFrame.ViewInverse;
                fusionProjectionInverse = infiniTamFrame.ProjectionInverse;
                _smoothedAngSpeed = infiniTamFrame.AngularSpeed;
                _motionQuality = infiniTamFrame.MotionQuality;
                acceptedInputReason = "infinitam_tracked_same_frame";
                _infiniTamTrackingAccepted++;
                _infiniTamConsecutiveTrackingRejects = 0;
                _infiniTamRecoveryConfirmedFrames = 0;
                if (!_infiniTamTrackingInitialised)
                {
                    _infiniTamBootstrapConfirmedFrames++;
                    _infiniTamStartupPhase = InfiniTamStartupPhase.Verifying;
                    if (_infiniTamBootstrapConfirmedFrames >=
                        Mathf.Max(1, infiniTamBootstrapConfirmFrames))
                    {
                        _infiniTamTrackingInitialised = true;
                        _infiniTamStartupPhase = InfiniTamStartupPhase.Tracking;
                        Logger.Info("InfiniTAM 启动模型已通过连续跟踪复核；正式网格现在可以发布。");
                    }
                }
                else
                {
                    _infiniTamStartupPhase = InfiniTamStartupPhase.Tracking;
                }
                _infiniTamLastTrackingDecision = "准";
            }
            else if (guardedExperimentActive && !_gunGelGuardedFusionRuntimeHalted)
            {
                if (!TryGetOldestResolvedGunGelFrame(out deferredFrame)) return false;
                if (!AcceptGunGelFrame(deferredFrame, out string rejectReason))
                {
                    // 欠秩、少配、回读/证据缺失和超出校正解算范围，只能说明
                    // 枪胶无法安全给出修正，不能证明同帧平台深度无效。除明确的
                    // 快速运动外，这些帧退回恒等位姿，由普通深度生产门继续裁决。
                    bool bootstrapObservation = rejectReason == "欠秩" ||
                                                rejectReason == "少配";
                    if (bootstrapObservation)
                        _gunGelEvidenceShadow?.BootstrapFrameDecision(
                            deferredFrame.Decision);
                    else
                        _gunGelEvidenceShadow?.AdjudicateFrameDecision(
                            deferredFrame.Decision, false);
                    _gunGelFusionRejected++;
                    bool explicitMotionConflict = rejectReason == "快角" ||
                                                  rejectReason == "快移";
                    _gunGelLastFusionDecision = explicitMotionConflict
                        ? rejectReason : "原" + rejectReason;
                    _gunGelLastAppliedMm = deferredFrame.Decision.TranslationMm;
                    if (explicitMotionConflict)
                    {
                        ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                            replayAttemptIndex, deferredFrame.PlatformFrame, false,
                            "gungel_reject:" + rejectReason, true, deferredFrame.FrameIndex,
                            deferredFrame.Decision.TranslationMm,
                            deferredFrame.Decision.RotationDeg,
                            deferredFrame.AngularSpeed, deferredFrame.LinearSpeed,
                            deferredFrame.MotionQuality);
                        ReleaseGunGelDeferredFrame(deferredFrame);
                        return false;
                    }

                    // 裁决准入是严格的“无准证不落笔”。欠秩/少配等帧仍送给
                    // GunGel 建候选，但它们没有最终 correspondence 身份，不能像
                    // 普通胶冻那样回退为原始 TSDF 写入，否则裁判会被旁路。
                    if (finalCourtAdmissionActive)
                    {
                        ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                            replayAttemptIndex, deferredFrame.PlatformFrame, false,
                            "final_court_pending:" + rejectReason, true,
                            deferredFrame.FrameIndex,
                            deferredFrame.Decision.TranslationMm,
                            deferredFrame.Decision.RotationDeg,
                            deferredFrame.AngularSpeed, deferredFrame.LinearSpeed,
                            deferredFrame.MotionQuality);
                        _gunGelLastFusionDecision = "裁待" + rejectReason;
                        ReleaseGunGelDeferredFrame(deferredFrame);
                        return false;
                    }

                    _gunGelFusionRawFallback++;
                    acceptedInputReason = "gungel_raw_fallback:" + rejectReason;
                }
                else if (_gunGelEvidenceShadow == null ||
                         !_gunGelEvidenceShadow.AdjudicateFrameDecision(
                             deferredFrame.Decision, true))
                {
                    // 候选事务失败同样只撤销枪胶修正权。原始延迟帧仍可走普通
                    // TSDF门禁；否则一个旁路账本错误会把整个生产融合一并熔断。
                    _gunGelFusionRejected++;
                    _gunGelLastFusionDecision = "原事务";
                    _gunGelLastAppliedMm = deferredFrame.Decision.TranslationMm;
                    acceptedInputReason = "gungel_raw_fallback:事务";
                    if (finalCourtAdmissionActive)
                    {
                        ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                            replayAttemptIndex, deferredFrame.PlatformFrame, false,
                            "final_court_pending:事务", true,
                            deferredFrame.FrameIndex,
                            deferredFrame.Decision.TranslationMm,
                            deferredFrame.Decision.RotationDeg,
                            deferredFrame.AngularSpeed, deferredFrame.LinearSpeed,
                            deferredFrame.MotionQuality);
                        _gunGelLastFusionDecision = "裁待事务";
                        ReleaseGunGelDeferredFrame(deferredFrame);
                        return false;
                    }
                    _gunGelFusionRawFallback++;
                }
                else
                {
                    gunGelIdentityAccepted =
                        deferredFrame.Decision.HasFusionAdmissionBuffers;
                    if (enableGunGelPoseCorrection)
                    {
                        ApplyGunGelCorrection(deferredFrame);
                        gunGelCorrectionApplied = true;
                        acceptedInputReason = "gungel_identity_pose_corrected";
                    }
                    else
                    {
                        acceptedInputReason = "gungel_identity_raw_depth";
                    }
                    _gunGelFusionAccepted++;
                    _gunGelLastFusionDecision = enableGunGelPoseCorrection
                        ? "身份+校姿" : "身份";
                    _gunGelLastAppliedMm = enableGunGelPoseCorrection
                        ? deferredFrame.Decision.TranslationMm : 0f;
                }

                // 无论使用校正还是原始位姿，都消费同一张延迟帧，禁止把这一帧的
                // 枪胶结果套到当前相机帧。二者的唯一区别是修正矩阵及逐点反证权。
                usingGuardedFrame = true;
                fusionDepth = deferredFrame.Depth;
                fusionNormal = deferredFrame.Normal;
                fusionDilatedDepth = deferredFrame.DilatedDepth;
                fusionEdgeReason = deferredFrame.EdgeReason;
                fusionTemporalReason = deferredFrame.TemporalReason;
                fusionTemporalReasonAvailable = fusionTemporalReason != null;
                fusionView = deferredFrame.View;
                fusionProjection = deferredFrame.Projection;
                fusionViewInverse = deferredFrame.ViewInverse;
                fusionProjectionInverse = deferredFrame.ProjectionInverse;
                _smoothedAngSpeed = deferredFrame.AngularSpeed;
                _motionQuality = deferredFrame.MotionQuality;
            }
            else if (currentHardGated)
            {
                ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                    replayAttemptIndex, dc.CurrentPlatformFrame, false,
                    "motion_hard_gate", false, -1, 0f, 0f,
                    currentAngularSpeed, currentLinearSpeed, _motionQuality);
                return false;
            }

            // 裁决档绝不继承普通胶冻的“旁路账本坏了、生产继续”容错语义。
            // 如果 GunGel 初始化/运行已熔断，此时没有可核验的 stableId，唯一
            // 安全结果就是整帧停笔；否则名义上的裁判档会悄悄变成原冻。
            if (enableFinalCourtAdmissionExperiment &&
                (!guardedExperimentActive || _gunGelGuardedFusionRuntimeHalted))
            {
                ScanReplaySessionPackage.Active?.RecordDecisionOnly(
                    replayAttemptIndex, dc.CurrentPlatformFrame, false,
                    "final_court_unavailable", false, -1, 0f, 0f,
                    currentAngularSpeed, currentLinearSpeed, _motionQuality);
                _gunGelLastFusionDecision = "裁不可用";
                return false;
            }

            compute.SetMatrixArray(DepthCapture.ViewID, fusionView);
            compute.SetMatrixArray(DepthCapture.ProjID, fusionProjection);
            compute.SetMatrixArray(DepthCapture.ViewInvID, fusionViewInverse);
            compute.SetMatrixArray(DepthCapture.ProjInvID, fusionProjectionInverse);
            compute.SetMatrix(FusionCorrectionID,
                infiniTamCorrectionApplied
                    ? infiniTamFrame.Decision.Correction
                    : gunGelCorrectionApplied
                    ? deferredFrame.Decision.Correction
                    : Matrix4x4.identity);

            int numExclusions;
            if (usingInfiniTamTrackedFrame)
            {
                numExclusions = infiniTamFrame.ExclusionCount;
                Array.Copy(infiniTamFrame.ExclusionPositions,
                    _exclusionPositions, _exclusionPositions.Length);
            }
            else if (usingGuardedFrame)
            {
                numExclusions = deferredFrame.ExclusionCount;
                Array.Copy(deferredFrame.ExclusionPositions, _exclusionPositions,
                    _exclusionPositions.Length);
            }
            else
            {
                numExclusions = Mathf.Min(ExclusionZones.Count, 64);
                for (int i = 0; i < numExclusions; i++)
                {
                    if (ExclusionZones[i] != null)
                        _exclusionPositions[i] = ExclusionZones[i].position;
                }
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
            compute.SetFloat(MatureObsEnableID, enableMatureSurfaceObsDiscount ? 1f : 0f);
            compute.SetFloat(MatureObsWeightMinID, matureSurfaceObsWeightMin);
            compute.SetFloat(MatureObsMarginID, matureSurfaceObsMargin);
            compute.SetFloat(MatureObsDiscountID, matureSurfaceObsDiscount);
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
            // Advance the camera-local lattice only on frames that actually
            // reach integration. SetShaderConstants establishes the initial
            // value; this accepted-frame cycle supplies the coverage benefit.
            Vector3 frustumPhase = enableFrustumPhaseCoverage
                ? FrustumCoveragePhases[IntegrationCount & 7] * voxelSize
                : Vector3.zero;
            compute.SetVector(FrustumPhaseOffsetID, frustumPhase);
            compute.SetFloat(DiagnosticAngularSpeedID, _smoothedAngSpeed);
            compute.SetFloat(NoiseMotionQualityID, noiseMotionWeightEnable ? _motionQuality : 1f);
            compute.SetFloat(MotionAuthorityQualityID, _motionQuality);
            compute.SetFloat(MotionConfirmQualityMinID, motionConfirmQualityMin);
            compute.SetFloat(NoiseDistExpID, noiseDistExponent);
            compute.SetFloat(NoiseAngExpID, noiseAngleExponent);
            compute.SetFloat(FrozenBlockEnableID, frozenBlockEnable ? 1f : 0f);
            compute.SetFloat(FrozenVoteQualityMinID, frozenVoteQualityMin);
            compute.SetFloat(FrozenVoteMarginID, frozenVoteMargin);

            // The InfiniTAM product is geometry-only and writes the established
            // white colour contract directly in its compact kernel. Do not blit
            // a passthrough RGB frame that no baseline shader invocation reads.
            bool productionCamAvailable = false;
            if (!enableInfiniTamBaseline)
            {
                if (!usingGuardedFrame && !usingInfiniTamTrackedFrame)
                    EnsureCamFrameCopy();
                productionCamAvailable = !usingGuardedFrame &&
                                         !usingInfiniTamTrackedFrame &&
                                         _pendingCamFrame != null &&
                                         _camFrameCopy != null;
                if (productionCamAvailable)
                {
                    compute.SetTexture(_integrateKernel.KernelIndex, CamRGBID,
                        _camFrameCopy);
                    compute.SetInt(CamAvailableID, 1);
                    compute.SetVector(CamPosID, _pendingCamPos);
                    compute.SetMatrix(CamInvRotID,
                        Matrix4x4.Rotate(Quaternion.Inverse(_pendingCamRot)));
                    compute.SetVector(CamFocalLenID, _pendingFocalLen);
                    compute.SetVector(CamPrincipalPtID, _pendingPrincipalPt);
                    compute.SetVector(CamSensorResID, _pendingSensorRes);
                    compute.SetVector(CamCurrentResID, _pendingCurrentRes);
                    compute.SetFloat(CamExposureID, cameraExposure);
                }
                else
                {
                    compute.SetTexture(_integrateKernel.KernelIndex, CamRGBID,
                        _dummyCamTex);
                    compute.SetInt(CamAvailableID, 0);
                }
            }

            _integrateKernel.Set(DepthCapture.DepthTexID, fusionDepth);
            Texture temporalReason = fusionTemporalReason != null
                ? fusionTemporalReason
                : fusionEdgeReason; // texture binding must remain valid even when the diagnostic is unavailable
            if (!enableInfiniTamBaseline)
            {
                _integrateKernel.Set(DepthCapture.NormTexID, fusionNormal);
                _integrateKernel.Set(DepthCapture.DilatedDepthTexID,
                    fusionDilatedDepth);
                _integrateKernel.Set(DepthCapture.EdgeReasonTexID,
                    fusionEdgeReason);
                _integrateKernel.Set(DepthCapture.TemporalReasonTexID,
                    temporalReason);
                compute.SetInt(TemporalReasonAvailableID,
                    fusionTemporalReasonAvailable ? 1 : 0);
            }

            // 枪胶不再只修整帧位姿：生产 A 直接消费“同一延迟帧”的逐点证据。
            // Correspondence.w 编码稳定候选、平台/预处理双证词与反对票状态；
            // Integrate 再按当前体素投影像素做局部一致性复核。裁决模式额外读取
            // 同一 correspondence 的 stableId 准证；当前完整深度像素先做成员
            // 资格检查，通过后由胜出平面约束生产 sDist。基线与 B 影子绑定零
            // 缓冲且关闭开关。
            bool gunGelIdentityAvailable = gunGelIdentityAccepted &&
                                           deferredFrame != null &&
                                           deferredFrame.Decision.HasFusionAdmissionBuffers;
            bool gunGelAdmissionActive = gunGelIdentityAvailable &&
                                         enableGunGelTsdfAdmission;
            ComputeBuffer gunGelObservations = gunGelIdentityAvailable
                ? deferredFrame.Decision.FusionObservations
                : _gunGelDummyObservations;
            ComputeBuffer gunGelCorrespondences = gunGelIdentityAvailable
                ? deferredFrame.Decision.FusionCorrespondences
                : _gunGelDummyCorrespondences;
            ComputeBuffer gunGelCorrespondenceIdentity = gunGelIdentityAvailable &&
                deferredFrame.Decision.FusionCorrespondenceIdentity != null
                ? deferredFrame.Decision.FusionCorrespondenceIdentity
                : _gunGelDummyCorrespondenceIdentity;
            ScanReplaySessionPackage courtSession = ScanReplaySessionPackage.Active;
            ComputeBuffer finalCourtVerdicts = null;
            ComputeBuffer finalCourtPlanes = null;
            ComputeBuffer finalCourtGenerations = null;
            bool hasFinalCourtVerdicts = courtSession != null &&
                courtSession.TryGetFinalCourtAdmissionBuffers(
                    out finalCourtVerdicts, out finalCourtPlanes,
                    out finalCourtGenerations);
            if (!hasFinalCourtVerdicts)
            {
                finalCourtVerdicts = _finalCourtDummyVerdicts;
                finalCourtPlanes = _finalCourtDummyPlanes;
                finalCourtGenerations = _finalCourtDummyGenerations;
            }
            // 即使账本意外缺席也保持 court enable=1，让 0 准证严格停笔；绝不因
            // 诊断会话启动失败而静默旁路回普通 TSDF 写入。
            bool finalCourtGateActive = finalCourtAdmissionActive &&
                                        gunGelAdmissionActive;
            if (!enableInfiniTamBaseline)
            {
                _integrateKernel.Set(GunGelObservationsID, gunGelObservations);
                _integrateKernel.Set(GunGelCorrespondencesID,
                    gunGelCorrespondences);
                _integrateKernel.Set(GunGelCorrespondenceIdentityID,
                    gunGelCorrespondenceIdentity);
                _integrateKernel.Set(FinalCourtVerdictsID, finalCourtVerdicts);
                _integrateKernel.Set(FinalCourtPlanesID, finalCourtPlanes);
                _integrateKernel.Set(FinalCourtGenerationsID,
                    finalCourtGenerations);
            }
            compute.SetInts(GunGelObservationGridID,
                gunGelIdentityAvailable ? deferredFrame.Decision.FusionObservationGridX : 1,
                gunGelIdentityAvailable ? deferredFrame.Decision.FusionObservationGridY : 1);
            compute.SetInt(GunGelPixelStrideID,
                gunGelIdentityAvailable ? deferredFrame.Decision.FusionPixelStride : 1);
            compute.SetFloat(GunGelAdmissionEnableID, gunGelAdmissionActive ? 1f : 0f);
            compute.SetInt(FinalCourtVerdictCapacityID,
                hasFinalCourtVerdicts
                    ? ScanReplaySessionPackage.FinalCourtVerdictCapacity : 1);
            compute.SetFloat(FinalCourtAdmissionEnableID,
                finalCourtGateActive ? 1f : 0f);

            // 旧历史探针的生产证词通路已停权。保留下面的空绑定是为了让
            // compute 参数布局与黑匣子计数保持兼容，但生产 Integrate 永远收到
            // enable=0；即时壳与三段接力诊断仍是只读观察者。
            ComputeBuffer shellWitnessEpochs = null;
            int3 shellWitnessCellCount = new int3(1, 1, 1);
            int shellWitnessStride = 1;
            uint shellWitnessEpoch = 0u;
            uint shellWitnessMaxAge = 0u;
            bool shellWitnessActive = false;
            if (!enableInfiniTamBaseline)
                _integrateKernel.Set(ShellWitnessEpochsID,
                    shellWitnessActive ? shellWitnessEpochs : _dummyShellWitnessEpochs);
            compute.SetInts(ShellWitnessCellCountID,
                shellWitnessActive ? shellWitnessCellCount.x : 1,
                shellWitnessActive ? shellWitnessCellCount.y : 1,
                shellWitnessActive ? shellWitnessCellCount.z : 1);
            compute.SetInt(ShellWitnessStrideID, shellWitnessActive ? shellWitnessStride : 1);
            compute.SetInt(ShellWitnessEpochID,
                shellWitnessActive ? unchecked((int)shellWitnessEpoch) : 0);
            compute.SetInt(ShellWitnessMaxAgeID,
                shellWitnessActive ? unchecked((int)shellWitnessMaxAge) : 0);
            compute.SetFloat(ShellWitnessEnableID, shellWitnessActive ? 1f : 0f);

            // 独立会话记录的是生产 Integrate 此刻真正绑定的完整输入：不仅是深度，
            // 还包括逐点枪胶准入、实际 RGB 副本及其针孔内外参。普通模式只旁路
            // 记账；裁决档复用最终 correspondence 回读更新“后续帧”的准证表，
            // 不追写当前帧，也不改变当前已冻结的 compute 绑定。
            int acceptedSourceFrame = usingInfiniTamTrackedFrame
                ? infiniTamFrame.PlatformFrame
                : usingGuardedFrame
                    ? deferredFrame.PlatformFrame
                    : dc.CurrentPlatformFrame;
            ScanReplaySessionPackage.Active?.RecordAcceptedInput(
                replayAttemptIndex,
                acceptedSourceFrame,
                acceptedInputReason,
                usingGuardedFrame,
                usingGuardedFrame ? deferredFrame.FrameIndex : -1,
                gunGelCorrectionApplied ? deferredFrame.Decision.TranslationMm : 0f,
                gunGelCorrectionApplied ? deferredFrame.Decision.RotationDeg : 0f,
                _smoothedAngSpeed,
                usingGuardedFrame ? deferredFrame.LinearSpeed : currentLinearSpeed,
                _motionQuality,
                fusionDepth, fusionNormal, fusionDilatedDepth,
                fusionEdgeReason, fusionTemporalReason,
                fusionProjection, fusionView, fusionProjectionInverse, fusionViewInverse,
                gunGelIdentityAvailable,
                gunGelIdentityAvailable ? gunGelObservations : null,
                gunGelIdentityAvailable ? gunGelCorrespondences : null,
                gunGelIdentityAvailable
                    ? deferredFrame.Decision.FusionPreTransactionCorrespondenceIdentity
                    : null,
                gunGelIdentityAvailable
                    ? deferredFrame.Decision.FusionCorrespondenceIdentity
                    : null,
                gunGelIdentityAvailable ? deferredFrame.Decision.FusionObservationGridX : 0,
                gunGelIdentityAvailable ? deferredFrame.Decision.FusionObservationGridY : 0,
                gunGelIdentityAvailable ? deferredFrame.Decision.FusionPixelStride : 0,
                productionCamAvailable,
                productionCamAvailable ? _camFrameCopy : null,
                _pendingCamPos, _pendingCamRot, _pendingFocalLen, _pendingPrincipalPt,
                _pendingSensorRes, _pendingCurrentRes,
                infiniTamCorrectionApplied
                    ? infiniTamFrame.Decision.Correction
                    : gunGelCorrectionApplied
                    ? deferredFrame.Decision.Correction
                    : Matrix4x4.identity,
                _exclusionPositions, numExclusions);

            // During bootstrap, render the old model for diagnostics before the
            // raw-pose frame writes.  After bootstrap this dispatch already ran
            // against the retained exact frame and its gated correction is now
            // in fusionView/fusionViewInverse; never dispatch it a second time.
            if (enableInfiniTamBaseline && !usingInfiniTamTrackedFrame)
            {
                _infiniTamModelRaycast ??= new InfiniTamModelRaycastAudit(8);
                _infiniTamModelRaycast.Dispatch(
                    fusionDepth, fusionProjection, fusionView,
                    fusionProjectionInverse, fusionViewInverse,
                    _volume, _infiniTamVoteWeightVolume,
                    voxelCount, voxelSize, voxelDistance,
                    minMeshWeight,
                    rejectNearSamples ? minUpdateDist : 0f,
                    maxUpdateDist, acceptedSourceFrame);
            }

            // The product court is a production service, not a side effect of
            // pressing A to start a diagnostic session.  Feed it the same
            // accepted, same-frame GunGel identities now; its verdict can only
            // constrain the extracted candidate and can never write this TSDF.
            if (gunGelIdentityAvailable && usingGuardedFrame)
            {
                QueueProductSurfaceCourtReadback(deferredFrame,
                    replayAttemptIndex,
                    gunGelCorrectionApplied
                        ? deferredFrame.Decision.Correction
                        : Matrix4x4.identity);
            }

            // A: production path (projective difference scaled by normal cosine), now
            // guarded per projected voxel by the exact same-frame GunGel evidence above.
            // Dirty epochs are extraction scheduling metadata only.  Baseline
            // fusion still writes the same sole TSDF; enabling this ledger lets
            // its private block front rebuild only regions whose zero crossing
            // changed instead of repeatedly extracting the full 256³ volume.
            BeginDirtyEpoch();
            // Replacement is a single-volume transaction: retire the previous
            // generation before the first batch carrying the new plane writes.
            // The persistent chunk pipeline continues showing its old front
            // buffer until a complete replacement extraction is committed.
            if (finalCourtAdmissionActive)
                InvalidateFinalCourtReplacements(courtSession);
            compute.SetFloat(UseRawProjectiveSdfID, enableInfiniTamBaseline ? 1f : 0f);
            compute.SetFloat(InfiniTamBaselineID, enableInfiniTamBaseline ? 1f : 0f);
            if (enableInfiniTamBaseline)
            {
                // This is a mutually-exclusive reconstruction baseline, not a
                // third adjudication office.  The sole production volume receives
                // the raw projective signed distance and the shader exits through
                // its compact weighted-average path before every GunGel/court/
                // freeze/product rule.
                compute.SetFloat(GunGelAdmissionEnableID, 0f);
                compute.SetFloat(FinalCourtAdmissionEnableID, 0f);
                compute.SetFloat(ShellWitnessEnableID, 0f);
            }
            compute.SetFloat(WriteColorID, 1f);
            compute.SetFloat(WriteAdmissionTraceID, 1f);
            compute.SetFloat(TsdfResponsibilityWriteID,
                _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
            compute.SetInt(TsdfResponsibilityIntegrationID,
                Mathf.Min(IntegrationCount + 1, 4095));
            compute.SetFloat(ConfidenceWriteID, 1f); // 主卷记置信度账
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            if (!enableInfiniTamBaseline)
                _integrateKernel.Set(CarveStatsID, _carveStats);
            // Bootstrap observations do not pass through the async tracker, so
            // count them here at the actual production dispatch.  Tracked
            // observations were counted once when their exact frame was queued.
            if (enableInfiniTamBaseline && !usingInfiniTamTrackedFrame)
                _infiniTamAttemptedFrames++;
            bool sampledInfiniTamTicket = PrepareInfiniTamTicketSample();
            _integrateKernel.DispatchFit(_frustumVolume.count, 1);
            compute.SetFloat(InfiniTamTicketEnabledID, 0f);
            if (enableInfiniTamBaseline)
            {
                _infiniTamFusedFrames++;
            }
            if (sampledInfiniTamTicket)
                RequestInfiniTamTicketReadback();
            // GunGel 自身的候选换轨仍在本批之后拆旧；最终裁判的 generation
            // 替换已在上方先拆后写，不与这个上游身份事务混为一谈。
            if (!enableInfiniTamBaseline && gunGelAdmissionActive)
                InvalidateGunGelSucceededRegions();

            // B: read-only KinectFusion-style projective TSDF shadow. It receives the
            // exact same depth, pose, gates, quality and carve settings, but stores the
            // unscaled ray-depth difference. No color/global production state is written.
            if (!enableInfiniTamBaseline &&
                _projectiveShadowVolume != null && _projectiveShadowCarveStats != null)
            {
                compute.SetFloat(GunGelAdmissionEnableID, 0f);
                compute.SetFloat(FinalCourtAdmissionEnableID, 0f);
                compute.SetFloat(ShellWitnessEnableID, 0f);
                compute.SetFloat(UseRawProjectiveSdfID, 1f);
                compute.SetFloat(WriteColorID, 0f);
                compute.SetFloat(WriteAdmissionTraceID, 0f);
                compute.SetFloat(TsdfResponsibilityWriteID, 0f);
                compute.SetFloat(ConfidenceWriteID, 0f); // 影子卷不记置信度账（防 A/B 双跑污染）
                ConfigureDirtyTracking(false);
                compute.SetInt(CamAvailableID, 0);
                _integrateKernel.Set(VolumeRWID, _projectiveShadowVolume);
                _integrateKernel.Set(ColorVolumeRWID, _colorVolume); // write-guarded
                _integrateKernel.Set(CarveStatsID, _projectiveShadowCarveStats);
                _integrateKernel.DispatchFit(_frustumVolume.count, 1);

                // Restore production bindings so external callers never inherit B state.
                compute.SetFloat(UseRawProjectiveSdfID, 0f);
                compute.SetFloat(InfiniTamBaselineID, 0f);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
                compute.SetFloat(TsdfResponsibilityWriteID,
                    _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
                compute.SetFloat(ConfidenceWriteID, 1f);
                compute.SetFloat(GunGelAdmissionEnableID, gunGelAdmissionActive ? 1f : 0f);
                compute.SetFloat(FinalCourtAdmissionEnableID,
                    finalCourtGateActive ? 1f : 0f);
                compute.SetFloat(ShellWitnessEnableID, shellWitnessActive ? 1f : 0f);
                ConfigureDirtyTracking(true);
                compute.SetInt(CamAvailableID, productionCamAvailable ? 1 : 0);
                _integrateKernel.Set(VolumeRWID, _volume);
                _integrateKernel.Set(CarveStatsID, _carveStats);
            }

            IntegrationCount++;
            // Exact bridge from the accepted fusion row to the dirty epoch that
            // extraction/replacement later consumes.  This is a dispatch fact,
            // not a claim that asynchronous GPU work has already completed.
            ScanReplaySessionPackage.Active?.RecordIntegrationDispatch(
                replayAttemptIndex, acceptedSourceFrame, IntegrationCount, _dirtyEpoch);
            _pendingCamFrame = null;
            ReleaseGunGelDeferredFrame(deferredFrame);
            if (usingInfiniTamTrackedFrame)
                ReleaseInfiniTamDeferredFrame();

            if (!enableInfiniTamBaseline && warmupIntegrations > 0 &&
                IntegrationCount == warmupIntegrations)
            {
                Logger.Info($"Warmup complete ({warmupIntegrations} frames), clearing volume to discard sensor startup noise");
                ClearInternal(preserveGunGelEvidence: true);
            }

            float t = Time.time;
            if (!enableInfiniTamBaseline && !_pruneCycleActive &&
                t - _lastPruneTime >= pruneIntervalSeconds)
            {
                _lastPruneTime = t;
                _nextPruneSlice = 0;
                _pruneCycleActive = true;
            }

            // Pruning used to scan the entire 3D volume in one dispatch.  Keep the
            // exact same voxel rule, but amortize it over integrations to avoid a
            // periodic full-volume GPU spike on Quest.
            if (!enableInfiniTamBaseline && _pruneCycleActive)
            {
                int sliceCount = Mathf.Min(Mathf.Max(1, pruneSlicesPerIntegration),
                    voxelCount.z - _nextPruneSlice);
                compute.SetInt(PruneZOffsetID, _nextPruneSlice);
                compute.SetInt(PruneZCountID, sliceCount);

                _pruneKernel.Set(VolumeRWID, _volume);
                _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
                compute.SetFloat(WriteColorID, 1f);
                compute.SetFloat(WriteAdmissionTraceID, 1f);
                compute.SetFloat(TsdfResponsibilityWriteID,
                    _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
                ConfigureDirtyTracking(true);
                _pruneKernel.DispatchFit(voxelCount.x, voxelCount.y, sliceCount);

                if (_projectiveShadowVolume != null)
                {
                    compute.SetFloat(WriteColorID, 0f);
                    compute.SetFloat(WriteAdmissionTraceID, 0f);
                    compute.SetFloat(TsdfResponsibilityWriteID, 0f);
                    ConfigureDirtyTracking(false);
                    _pruneKernel.Set(VolumeRWID, _projectiveShadowVolume);
                    _pruneKernel.Set(ColorVolumeRWID, _colorVolume); // write-guarded
                    _pruneKernel.DispatchFit(voxelCount.x, voxelCount.y, sliceCount);
                    compute.SetFloat(WriteColorID, 1f);
                    compute.SetFloat(WriteAdmissionTraceID, 1f);
                    compute.SetFloat(TsdfResponsibilityWriteID,
                        _tsdfResponsibilityCaptureEnabled ? 1f : 0f);
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
            return true;
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

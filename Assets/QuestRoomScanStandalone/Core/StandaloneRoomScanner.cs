using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// QRS 房间扫描链的独立测试编排器（裁剪版 RoomScanner）。
    /// 只保留：深度采集 → TSDF 融合 → GPU Surface Nets 提取 → 顶点色网格渲染。
    /// 砍掉：持久化、锚点、纹理精炼、三平面、关键帧、GSplat、房间理解、调试菜单。
    ///
    /// 所有兄弟组件挂在同一个 GameObject 上自动解析。
    /// 输入由 <see cref="StandaloneScanInput"/> 处理：扳机采集，A 冻结共享
    /// TSDF，Y 切 64/32/16，B 仅导出并清当前档，右摇杆切显示。
    /// </summary>
    [RequireComponent(typeof(DepthCapture), typeof(VolumeIntegrator), typeof(MeshExtractor))]
    public class StandaloneRoomScanner : MonoBehaviour
    {
        public static StandaloneRoomScanner Instance { get; private set; }

        [Header("扫描频率")]
        [SerializeField, Tooltip("TSDF 融合频率。Quest InfiniTAM 生产档实机验证 10Hz；网格生长由有界帧预算调度保证，不靠恢复 20Hz 硬顶。")]
        private float integrationHz = 10f;
        [SerializeField, Range(1f, 15f), Tooltip("完整体积网格提取频率。融合保持高频，网格沿用上次结果直到下一次提取。08-18 帧率手术后 8→12（HERA 节拍 16→24/s），帧率跌破 50 退回 8。")]
        private float meshExtractionHz = 12f;

        [Header("InfiniTAM 帧预算调度")]
        [SerializeField, Tooltip("开=只在近期帧节拍健康时放行普通融合/提取；超过最长等待仍强制放行一次，防止网格停长。")]
        private bool enableInfiniTamFrameBudget = true;
        [SerializeField, Range(55f, 71f), Tooltip("恢复普通重任务前需达到的近期帧率。68fps 给 72Hz 合成器留少量余量。")]
        private float infiniTamRecoveryFps = 68f;
        [SerializeField, Range(1, 8), Tooltip("融合/提取后需连续多少个健康帧才放行下一个普通重任务。")]
        private int infiniTamHealthyFramesBeforeWork = 3;
        [SerializeField, Range(0.015f, 0.08f), Tooltip("任意两个融合/提取提交之间的最小间隔（秒）；普通提取只在两个融合时点之间的中段放行，避免10Hz融合与5Hz提取前后贴车。")]
        private float infiniTamHeavyWorkMinSpacingSeconds = 0.035f;
        [SerializeField, Range(1.5f, 4f), Tooltip("帧压力持续时，融合最多延期几个名义周期后必须放行一次。")]
        private float infiniTamFusionMaxDelayIntervals = 2.5f;
        [SerializeField, Range(2f, 6f), Tooltip("帧压力持续时，网格提取最多延期几个名义周期后必须放行一次。")]
        private float infiniTamMeshMaxDelayIntervals = 3f;

        [Header("渲染")]
        [SerializeField, Tooltip("主显示形态：开=线框（QRS Wireframe，重心坐标边缘检测）；关=顶点色实体（QRS Vertex）")]
        private bool wireframeMode = true;
        [SerializeField, Range(0.2f, 5f), Tooltip("线框模式的线条粗细倍率（1.0 对齐原 SC 工程细线观感，可按需调）")]
        private float wireThickness = 1.0f;
        [SerializeField, Range(1, 6), Tooltip("条带抽稀（顶点侧按体素格丢三角形，任一轴对齐即保留）。08-19 实机判定观感碎、做不出 Meta 粗网，已让世界格线画法取代，默认 1=关闭，仅留作帧率应急杠杆")]
        private int meshDisplayStride = 1;
        [SerializeField, Range(0.1f, 1.0f), Tooltip("纸主三角网的世界空间间距（米）。0.12m 保留旧细网对局部结构的可读性；后续视觉定稿可再放大。")]
        private float meshGridSpacing = 0.12f;
        [SerializeField, Tooltip("置信度通道 v1 可视化（诊断开关，默认关）：开=按体素分歧 EMA 给网格着色——高置信=原色 / 中=黄 / 低=蓝紫（几何在打架）/ 无数据=灰。只读着色，不碰任何生产逻辑")]
        private bool confidenceViz = false;
        [SerializeField, Tooltip("第一阶段纯几何观察：开=所有生产/HERA网格统一白色，绕开置信、冻结和路由着色。仅显示层，不改变融合、提取或页面调度。")]
        private bool geometryTruthView = true;
        [SerializeField, Tooltip("32³融合/冻结管理块运行时线框：淡青=稳定冻结，黄=本窗自由空间票热，红=连续两窗热，洋红=已解冻待复冻。只读显示层。")]
        // Production starts clean.  The 32^3 management boxes are a diagnostic
        // overlay and must be requested explicitly; otherwise a cold start can
        // already be on the product route while the cyan block cages remain on
        // top and look like part of the delivered surface.
        private bool showManagementBlockWireOverlay = false;

        [Header("覆盖范围")]
        [SerializeField, Tooltip("头部排除区（QRS 原版防自扫）：开=头周圆柱内永不生成网格（半径在 VolumeIntegrator.exclusionRadius 调）；关=周围近距也能覆盖网格")]
        private bool enableHeadExclusion = true;

        [Header("冻结 TSDF 切块 A/B")]
        [SerializeField] private bool enableFrozenChunkAbExperiment = true;
        [SerializeField, Tooltip("在同一冻结 TSDF 上运行 HERA：64³只记账，32³分流，16³仅精修问题页；关闭时回退旧三档A/B。")]
        private bool enableHeraHierarchicalReplay = true;
        [SerializeField, Range(1, 8), Tooltip("每帧最多回放的网格页数；三档共用同一份冻结 TSDF")]
        private int abMaxChunksPerTick = 3;
        [SerializeField, Tooltip("导出分层账后保留回放显示：戴着头显走到红三角簇旁指认物理实体（贯通账坐标需要画面对照）。关=导出即清空派生网格（旧行为）。")]
        private bool keepFrozenReplayAfterExport = true;

        [Header("逐块可逆冻结")]
        [SerializeField, Tooltip("成熟 64³ 块自动冻结（weight 翻符号不销毁 TSDF），扫描不停；穿越票双门槛解冻修复再冻。" +
                 "冻结块停止积分写入=停止重提抖动+锁住已收敛几何；解冻=翻回符号，修复由正常积分驱动。 (default true)")]
        private bool enableFrozenBlockSupervisor = true;
        [SerializeField, Tooltip("枪胶净室追责：保留枪胶受保护融合和实时页面生产，但整卷停用逐块冻结/解冻监督。" +
                 "用于隔离 32³ 整块速冻是否导致转角留不住、孔洞和旧页台阶；仅允许空卷切换。主对照默认关闭。 (default false)")]
        private bool enableGunGelCleanRoomExperiment = false;
        [SerializeField, Min(0.5f), Tooltip("穿越票/成熟度统计窗（秒）：每窗回读票箱+普查成熟度。1s=新区首现/启动空窗减半（T1b；冻结时机由首达标满 1s 守卫兜底不提前）。开自适应普查后此值=快窗基准，安静期自动放慢到 2/4s。 (default 1，08-18 从 2 收紧；退回值 2)")]
        private float frozenBlockWindowSeconds = 1f;
        [SerializeField, Min(100), Tooltip("成熟判定：块内长熟体素（权重≥frozenMatureWeight）数下限（32³ 块含一面墙约 1~3k；空块/毛坯块永不冻结）。 (default 600)")]
        private int frozenBlockMinSurfaceVoxels = 600;
        [SerializeField, Min(0), Tooltip("成熟判定：相邻两窗长熟体素数允许波动（边界体素抖动容差；超过=块仍在生长/被啃，不冻）。 (default 24)")]
        private int frozenBlockStabilityTolerance = 24;
        [SerializeField, Min(1), Tooltip("解冻门槛：冻结块单窗穿越票（自由空间+遮挡）达到此数记一个热窗，连续两窗达标才解冻（防抖动）。32³ 块比 64³ 小，阈值同比例降。 (default 200)")]
        private int frozenBlockVoteThreshold = 200;
        [SerializeField, Range(1f, 8f), Tooltip("棘轮解冻：每解冻过一次的块，票阈×倍率^次数（首解保持灵敏，振荡成本指数升，真变化永留申诉通道）。1=固定阈（退回旧行为）。 (default 3，08-19 深夜校准：×8 止血档曾致纠错名义化（手难纠错），×2 复活纠错成功但放出噪声荒漠区”解-活-噪-杀“循环（实机：解23/帧32，自愈靠棘轮升档但路太长）；×3=首解 200 不变保灵敏度，惯犯升档陡一档 200/600/1800/5400/16200 让脏区更快锁死。振荡区票荒双保险（拍平截票+噪拦）不变，复发再回调)")]
        private float frozenBlockVoteRatchet = 3f;
        [SerializeField, Min(1), Tooltip("T2 补洞门槛：冻结块单窗补洞票（冻块内空体素种子出生数）达到此数记一个热窗，连续两窗达标→重排该块页面重提（不解冻）。种子每体素一生只出生一次，票天然有界，阈值可与解冻阈同档。 (default 200)")]
        private int frozenBlockHoleVoteThreshold = 200;
        [SerializeField, Tooltip("冻结需相邻两窗稳定（旧行为：冻结延迟 4-8s）。关=一窗达标即冻（速冻，2-4s），配合实时轨先看后冻。 (default false)")]
        private bool frozenBlockRequireStability = false;
        [SerializeField, Tooltip("冻结资格门（置信度消费 v2 第一刀，08-20 用户拍板主刀）：开=冻结裁决时查块内低置信体素占比，" +
                 "超阈拒冻——噪声荒漠/几何打架的块没资格稳定，永不冻结保持 v2.2 式活代谢（幻影随生随杀）；" +
                 "拍平压稳折角→观测一致→置信升→自然获得冻结资格，振荡断根不依赖棘轮高度。" +
                 "只闸普查冻结路径，不碰复冻兜底（振荡最后防线，防止闸门误伤重开振荡战）。 (default true)")]
        private bool enableFreezeConfidenceGate = true;
        [SerializeField, Range(0.02f, 0.5f), Tooltip("资格门低置信占比上限：块内可出网体素中分歧 EMA 低置信（几何在打架）占比超过此值则拒冻。0.2=五分之一体素在打架就不配冻。 (default 0.2)")]
        private float freezeGateLowConfMaxFrac = 0.2f;
        [SerializeField, Range(0.5f, 4f), Tooltip("冻结热身期（秒）：块首次报满成熟下限后须等满此时间才许冻——给分歧 EMA 留积累窗，" +
                 "堵冷启动 fail-open 洞（录屏 025342 判决：冻4 在 7.8s 已发生，EMA 全 0=高置信期谁申请都批，赃物由此进琥珀）。 (default 2，08-20 从 1 上调)")]
        private float frozenBlockMatureWarmupSeconds = 2f;
        [SerializeField, Range(0.1f, 0.8f), Tooltip("资格年审降级阈：已冻块低置信占比超过此值→主动降级解冻（赃出琥珀恢复活代谢）。" +
                 "必须高于拒冻阈=滞回防翻烙饼（冻<0.2 才批、审>0.35 才降，中间带=既往不咎）。降级不记 thawCounts=不吃棘轮误罚。 (default 0.35)")]
        private float freezeGateDemoteLowConfFrac = 0.35f;
        [SerializeField, Tooltip("自适应普查：连续安静窗（无冻/解/生长/穿越票）后普查窗按 1→2→4s 阶梯放慢，任一活动立即打回快窗。静止场景省掉空转普查的全体积 dispatch+双回读。 (default true，08-18 晚帧率预算手术)")]
        private bool enableAdaptiveCensus = true;

        [Header("活跃页调度（看哪出哪）")]
        [SerializeField, Tooltip("32³只保留空间容器身份。任何产生有效几何变化的页面均可出网；" +
                 "当前视野优先，首次变脏时刻提供防饿死期限，冻结状态不再拥有出网否决权。 (default true)")]
        private bool enableLiveTrack = true;
        [SerializeField, Min(0.1f), Tooltip("实时轨巡视间隔（秒）。 (default 0.15，帧率手术后 08-18 从 0.25 放松)")]
        private float liveTrackSweepSeconds = 0.15f;
        [SerializeField, Min(0.2f), Tooltip("同一块实时重提节流（秒）：实时页缺肉后长肉的刷新节奏。 (default 0.5，帧率手术后 08-18 从 1 放松)")]
        private float liveTrackBlockCooldownSeconds = 0.5f;
        [SerializeField, Range(1, 20), Tooltip("实时轨全局峰值（页/秒）硬顶：Quest 帧率保护闸，超限本巡视直接收工。 (default 10，帧率手术后 08-18 从 5 放松；帧率跌破 50 退回 5)")]
        private int liveTrackMaxPagesPerSecond = 10;
        [SerializeField, Min(0), Tooltip("实时轨块内容下限：普查可出网体素（≥minMeshWeight 0.08，T3 从长熟 0.15 降档——0.08~0.15 带可出网不该被当空块）低于此数的块不排。普查按 1s 窗更新，全新区域首次出网最多延迟一个窗。 (default 16)")]
        private int liveTrackMinSurfaceVoxels = 16;
        [SerializeField, Range(0f, 0.5f), Tooltip("视锥预热外扩：0.18 表示屏幕四周再扩 18%，转头前相邻页先排队。")]
        private float liveTrackViewportMargin = 0.18f;
        [SerializeField, Min(0.1f), Tooltip("当前视野脏页最长等待（秒）。持续变化不得重置这个期限。")]
        private float liveTrackVisibleDeadlineSeconds = 0.35f;
        [SerializeField, Min(0.5f), Tooltip("视野外脏页最长等待（秒）。用于后台最终一致，防止必须转头才刷新。")]
        private float liveTrackBackgroundDeadlineSeconds = 2f;
        [SerializeField, Range(1, 8), Tooltip("一次巡视最多提交的32³页面数；全局页/秒硬顶仍负责GPU保护。")]
        private int liveTrackMaxPagesPerSweep = 5;

        [Header("增量精修（两段合一）")]
        [SerializeField, Tooltip("成熟冻结块就地精修上屏：采集段不出粗网，冻哪块出哪块的 HERA 红绿网格；点阵默认隐藏（X 呼出当判官）。" +
                 "关=旧两段式（采集只点阵，A 冻结才出网格）。需同时开启 HERA 分层回放。 (default true)")]
        private bool enableIncrementalHeraRefine = true;
        [SerializeField, Min(0f), Tooltip("复冻重提冷却（秒）：振荡块在冷却内复冻不立刻重提，到点由维护时钟补提（最终一致）。" +
                 "首冻不受限。旧父页始终驻留，冷却只合并重复任务，不再撤页或强制复冻。 (default 15)")]
        private float frozenPageRequeueCooldownSeconds = 15f;

        [Header("平面拍平（B1 影子，只读验证）")]
        [SerializeField, Tooltip("B1 影子平面拟合：对视线落点块的当帧高质量观测做 PCA 平面拟合（只读不写 TSDF），" +
                 "HUD 拍行输出残差（拟合优度）/存量偏移（拍平幅度预估）/点数/主轴。验证'观测共识面干净+存量偏离'假设，成立才开 B2 写入。 (default true)")]
        private bool enablePlaneFitShadow = true;
        [SerializeField] private ComputeShader planeFitShadowCompute;

        [Header("平面拍平（B2 已退出生产）")]
        [SerializeField, Tooltip("B2 会绕过枪胶逐点准入，直接用预处理深度改写 TSDF；这会把近平行的错误深度压进几何，并抹平床/枕头等真实层次。" +
                 "生产默认关闭；B1 影子拟合仍保留只读诊断。")]
        private bool enablePlaneFlatten = false;
        [SerializeField, Range(0.1f, 0.6f), Tooltip("显著区阈值（sd 归一，1=截断带 15cm）：|观测-存量| 超此才压。0.3≈4.5cm；平墙实测 O-12mm 天然免疫")]
        private float planeFlattenMinDelta = 0.3f;
        [SerializeField, Range(0.05f, 0.6f), Tooltip("持久矛盾门槛（分歧 EMA，0-1）：EMA 超此才认定持续矛盾。0.25≈3.75cm；折角实测 ~0.67")]
        private float planeFlattenMinConf = 0.25f;
        [SerializeField, Range(0.2f, 2f), Tooltip("拍平节拍（秒）。冻块不自愈全靠本通道，0.5s 足够")]
        private float planeFlattenIntervalSec = 0.5f;
        [SerializeField, Range(0.2f, 0.9f), Tooltip("相干门槛（08-19 三面角伺服实锤）：|相干-0.5|/分歧 超此才压。" +
                 "观测方向逐帧翻=抖动观测（相干回中性）不配写几何；方向固定=系统偏差/真变化照压。与冻票噪拦同杆")]
        private float planeFlattenMinCoherence = 0.5f;
        [SerializeField, Range(1, 32), Tooltip("每拍最多重提几页（冻块拍平后的显示刷新，重排不解冻）。防一拍灌爆提取队列")]
        private int planeFlattenRepageMaxPerTick = 8;

        [Header("M3/B3 平面先验普查（只读影子）")]
        [SerializeField, Tooltip("M3/B3 第一刀：对冻结块做平面资格普查（只读 TSDF/置信度，不写任何生产状态）。" +
                 "HUD 报 合格平面块数/面旁无数据候选面积/拒因分布——先量出值不值得，再决定接不接显示层补全皮。 (default true)")]
        private bool enablePlanePriorCensus = true;
        [SerializeField, Range(0.5f, 10f), Tooltip("普查节拍（秒）。冻结块集合低频变化，2s 足够；一块一线程，开销可忽略")]
        private float planePriorCensusIntervalSec = 2f;
        [SerializeField, Range(0.03f, 0.2f), Tooltip("正式面 weight 门槛（与出网门槛同杆 0.08）：低于此不算面证据")]
        private float planePriorMinWeight = 0.08f;
        [SerializeField, Range(0.1f, 0.6f), Tooltip("低置信判线（分歧 EMA）：超过此值记为脏体素（与拒冻门 0.2 同杆）")]
        private float planePriorLowConfMin = 0.2f;
        [SerializeField, Range(0.05f, 0.6f), Tooltip("低置信占比上限：块内脏体素比例超此拒认平面（与拒冻门 w/z>0.2 同杆）")]
        private float planePriorLowConfMaxFrac = 0.2f;
        [SerializeField, Range(0.005f, 0.08f), Tooltip("平面残差上限（米 RMS）：块内 TSDF 面点 sd 子体素拟合残差超此=非平面。" +
                 "校准锚（08-20 205043 实机）：好墙 B1 真值带 23-29mm、干净冻墙 EMA 后应 10-15mm、垃圾搅局块 ~110mm；30mm 干净分离。 (default 0.03)")]
        private float planePriorMaxResidualMeters = 0.03f;
        [SerializeField, Range(0.03f, 0.3f), Tooltip("补全带半宽（米）：合格平面两侧此距离内的无数据体素记为 B3 候选面积")]
        private float planePriorFillBandMeters = 0.12f;
        [SerializeField, Min(50), Tooltip("面体素数下限：块内正式面体素少于此=证据不足拒认（300≈32³块的 1%）")]
        private int planePriorMinSurfaceVoxels = 300;
        [SerializeField, Range(1, 256), Tooltip("每拍最多普查几个冻结块（一块一线程；超出下拍再轮，低频普查不追一拍全量）")]
        private int planePriorMaxBlocksPerTick = 64;

        // ── 逐块可逆冻结调度器状态 ──
        private readonly HashSet<int> _frozenBlocks = new HashSet<int>();
        private readonly HashSet<int> _hotBlocksPrevWindow = new HashSet<int>();
        private readonly HashSet<int> _managementHotBlocks = new HashSet<int>();
        private readonly HashSet<int> _managementConfirmedHotBlocks = new HashSet<int>();
        private readonly HashSet<int> _managementThawedBlocks = new HashSet<int>();
        private FrozenBlockWireOverlay _managementBlockWireOverlay;
        private int[] _maturityPrevSurface;
        private uint[] _freezeSetMask;
        private uint[] _freezeClearMask;
        private float _frozenBlockWindowStart = -1f;
        private bool _frozenBlockReadbackPending;
        private float _frozenBlockReadbackPendingSince;
        // 自适应普查（08-18 晚）：全场普查每次=全体积成熟度 dispatch+双回读，
        // 1s 固定窗在静止场景纯属空转烧钱（实机：帧预算叠加超支的嫌疑人之一）。
        // 规则：上窗有冻/解/生长/穿越票任一活动→保持快窗；连续安静窗→阶梯放慢
        // （活动定义见 OnChunkMaturityReadback 尾部）。安静期解冻响应最多少延迟
        // 一个慢窗——票在 GPU 票箱里持续累积，下一窗必被看见并立即打回快窗。
        private int _censusQuietStreak;
        private float _censusCurrentWindow = 1f;
        private int _lastVotesHotCount;
        private int _frozenBlockUnfreezeEvents;
        // 资格门账：因低置信占比超阈被拒冻的块次（累计，HUD"资N"）。
        private int _freezeGateRejected;
        // 资格年审账：已冻块低置信超降级阈被收回资格的块次（累计，HUD"审N"）+
        // 本批降级名单（解冻记账环节对它跳过 thawCounts++，降级不吃棘轮）。
        private int _freezeGateDemoted;
        private readonly HashSet<int> _demotedThisBatch = new HashSet<int>();
        // 解冻诊断账（HUD"诊"，代替 logcat——实机看不了日志）：热窗总数 /
        // 解冻时的最大累计解冻序号（同块反复解则爬升，多块轮流则停在 1-2）/
        // 最近一次解冻的本窗票数与当时阈值（票随阈涨=棘轮被票量追平）。
        private int _diagHotWindows;
        private int _diagUnfreezeMaxThaws;
        private uint _diagLastVotes;
        private uint _diagLastThreshold;
        private int _diagLastThaws;
        // 票峰：历次解冻中的最高票数及当时解冻序号——校准棘轮倍率用
        // （峰远高于当前档阈=倍率还不够；峰贴着阈=振荡已到天花板附近）。
        private uint _diagPeakVotes;
        private int _diagPeakThaws;
        // ── B1 影子平面拟合状态（只读）──
        private ComputeKernelHelper _planeFitAccumKernel;
        private ComputeKernelHelper _planeOffsetKernel;
        private ComputeBuffer _planeFitStats;      // 16 int，与 shader 槽位约定一致
        private static readonly int[] PlaneFitZero = new int[16];
        private float _planeFitLastTick = -10f;
        private bool _planeFitPending;
        private bool _planeOffsetPending;
        private float _planePendingSince;
        private Unity.Mathematics.int3 _planeBlockMinVox;
        private bool _planeInit;                   // 平面 EMA 是否已初始化
        private Vector3 _planePointLocal;          // 平面 EMA（体积局部，米）
        private Vector3 _planeNormalLocal;
        // HUD 读数
        private bool _planeHudValid;
        private string _planeHudNote = "";
        // 空括号悬案探针：T=tick 入口计数（不涨=tick 根本没被执行/上游异常截断
        // Update）；F=拟合回读回调计数（T 涨 F 不涨=dispatch 后回读丢失）。
        private int _planeTicks;
        private int _planeFitCb;
        // ── B2 拍平写入状态（生产：写 TSDF sd，不动 weight）──
        private ComputeKernelHelper _planeFlattenKernel;
        private ComputeBuffer _planeFlatStats;       // 16 int（复用槽位约定：13=压下总数 14=其中冻块）
        private float _planeFlattenLastTick = -10f;
        private bool _planeFlatPending;
        private float _planeFlatPendingSince;
        private string _flatHudNote = "";
        private int _flatSnapTotal;                  // HUD 累计：压下体素总数
        private int _flatSnapFrozen;                 // HUD 累计：其中冻块体素
        private int _flatSnapCohBlocked;             // HUD 累计：相干闸拦下（抖动观测不配写几何）
        private int _flatSnapRepage;                 // HUD 累计：拍平后重提页数（显示刷新）
        private int _flatSnapRepageDeferred;         // HUD 累计：冷却内转延迟补提（稳态下重排全走这条路，只数排=读数误导）
        private ComputeBuffer _planeFlatBlocks;      // 64 int，本拍压到的冻块线性索引（刀3）
        private int _planeFlatBlocksExpected;        // 链式回读条数（stats[12]，钳 64）
        private readonly HashSet<int> _flatUniqueBlocks = new HashSet<int>();
        // ── M3/B3 平面先验普查状态（只读影子：只读 TSDF/置信度，不写生产状态）──
        private ComputeKernelHelper _planeCensusKernel;
        private ComputeBuffer _planeCensusBlocks;    // planePriorMaxBlocksPerTick int，本拍冻结块清单
        private ComputeBuffer _planeCensusStats;     // 16 int（槽位约定见 PlaneFitShadow.compute）
        private float _planeCensusLastTick = -10f;
        private bool _planeCensusPending;
        private float _planeCensusPendingSince;
        private readonly List<int> _planeCensusBlockList = new List<int>(64);
        // 冻前探头（引导值）：同 kernel 同闸，单块视线落点块，独立 stats buffer
        private ComputeKernelHelper _planeProbeKernel;
        private ComputeBuffer _planeProbeBlocks;    // 1 int
        private ComputeBuffer _planeProbeStats;     // 16 int，同槽位约定
        private bool _planeProbePending;
        private float _planeProbePendingSince;
        private bool _probeGazeFrozen;      // 本拍探头块是否已冻（回读切 冻/预 前缀）
        private readonly int[] _probeBlockArr = new int[1];
        private string _probeHud = "";
        // HUD 读数：查=扫描块数 合=平面资格合格 缺=面旁无数据候选体素
        // 残=算出残差块（合格+残差拒）的均残差 mm（诊断"差多少被拒"）
        // 拒因：残（残差/特征分解）浊（低置信占比）少（面体素不足）。
        private string _censusHudNote = "";
        private int _censusHudScanned;
        private int _censusHudQualified;
        private int _censusHudFill;
        private int _censusHudAvgResidualMm;
        private int _censusHudRejEigen;
        private int _censusHudRejLowConf;
        private int _censusHudRejFew;
        private float _planeResidualMm;
        private float _planeOffsetMm;
        private int _planePointCount;
        private char _planeAxis = '?';
        private int _supervisorWatchdogResets;
        // 重提冷却账：块号 → 上次实际排队时刻 / 冷却内复冻的延迟补提时刻。
        private readonly Dictionary<int, float> _lastPageQueueTime = new Dictionary<int, float>();
        private readonly Dictionary<int, float> _deferredPageRequeue = new Dictionary<int, float>();
        // 棘轮解冻账：块号 → 累计解冻次数（票阈=基数×倍率^次数）。
        private readonly Dictionary<int, int> _thawCounts = new Dictionary<int, int>();
        // ── T2 补洞票状态 ──
        // 补洞热窗账（双门槛用）/ 补洞棘轮账（块号→累计重排次数，与解冻棘轮互相独立：
        // 重排不动积分权，不该抬高解冻申诉成本，反之亦然）/ 全量冻结位图（随裁决下发 GPU）。
        private readonly HashSet<int> _hotHoleBlocksPrevWindow = new HashSet<int>();
        private readonly Dictionary<int, int> _holeRequeueCounts = new Dictionary<int, int>();
        private uint[] _frozenChunkBitsArr;
        private int _holeRequeueEvents;
        // 实时轨状态：巡视时钟 / 块级节流账 / 全局速率窗。
        private float _liveTrackLastSweep = -1f;
        private readonly Dictionary<int, float> _livePageQueueTime = new Dictionary<int, float>();
        // 尝试账（T1a）：块号 → 上次排队尝试时的页面 epoch。它只证明请求发出过，
        // 不证明页面 Built，更不证明已有非空前台；覆盖债必须由 HERA 的逐页发布产物
        // 查询单独结清。脏块 epoch 超过它仍表示已有页发生了新几何变化。
        private readonly Dictionary<int, uint> _liveQueuedEpoch = new Dictionary<int, uint>();
        // 缺产品页专用观察账：记录上次排队时已消费到哪一次“融合仍看到表面”。
        // 它不替代几何 epoch，只让空产品重试脱离冻结成熟度普查的慢时钟。
        private readonly Dictionary<int, uint> _liveQueuedObservedEpoch = new Dictionary<int, uint>();
        // 首次脏时刻只记一次，后续变化不能重置，避免繁忙转角永久饿死。
        private readonly Dictionary<int, float> _liveDirtySince = new Dictionary<int, float>();
        // 每个32³页面六个面的边界债已观察 epoch；债务同时唤醒共享面的两页。
        private uint[] _liveBoundaryEpochConsumed;
        // 网格资格账（普查 z 槽）：块号 → 可出网（≥minMeshWeight）体素数。空闸判据（T3）。
        private int[] _meshablePrevSurface;
        // 首达标时刻账：块号 → 普查首次报满成熟下限的时刻（T1b 防冻结随普查窗缩短提前）。
        private readonly Dictionary<int, float> _firstMatureTime = new Dictionary<int, float>();
        private float _liveRateWindowStart = -1f;
        private int _liveRateWindowCount;
        private int _liveTrackQueuedTotal;
        // 活闸普查：上轮巡视 slab 候选被各道闸拦截的次数（HUD 判读用，每轮巡视重计）。
        // 排=成功排队 / 冻=定稿轨地盘 / 空=长熟数不足 / 冷=块级节流 / 内=内容闸 / 途=在途 / 速=速率硬顶。
        private int _liveGateQueued, _liveGateFrozen, _liveGateEmpty, _liveGateCool,
                    _liveGateContent, _liveGateFlight, _liveGateRate;
        private int _liveDirtyCount, _liveVisibleDirtyCount, _liveOverdueCount, _liveBoundaryDebtCount;
        // 计时账（EMA α=0.25）：融合/提取 CPU 耗时 + 提取实际节拍（拍/s）。
        private float _emaIntegrateMs = -1f;
        private float _emaHeraTickMs = -1f;
        private int _heraTicksThisWindow;
        private float _heraTickWindowStart = -1f;
        private float _heraTicksPerSec;

        private static readonly int[] ChunkAbSizes = { 64, 32, 16 };
        private int _chunkAbGearIndex;
        private bool _chunkAbFrozen;
        private bool _chunkAbDiagnosticColoring = true;
        private ObservationCoverageOverlay _coverageOverlay;
        private DepthPointCloudOverlay _depthPointCloudOverlay;
        private GunGelCourtOverlay _gunGelCourtOverlay;
        private InstantDepthShellOverlay _instantDepthShellOverlay;
        private bool _bbPresentationCaptured;
        private bool _bbRestoreMeshVisible;
        private bool _bbRestoreCoarseSkinVisible;
        private bool _bbRestoreManagementBlocks;
        // X 路线中“壳纸合流后的仅纸对照”是一个独立观察态。
        // 此时即时壳不上屏、但仍在后台采证，且不得被误判为
        // 整条 X 诊断链已退出。
        private bool _shellPaperOnlyView;
        // 仅纸档的绘制级纠错隔离：稳定 Reject 单元不上屏；恢复
        // 两个独立安全支撑视角后，仍须等实际纸面贴合新目标。
        // 不修改 TSDF 或候选账本。
        private const int PaperCorrectionHashProbeCount = 24;
        private readonly List<VirtualProbeShadowAdjudicator.PaperCorrectionCell>
            _paperCorrectionCells =
                new List<VirtualProbeShadowAdjudicator.PaperCorrectionCell>(128);
        private ScanReplaySessionPackage _paperCorrectionSession;
        private ComputeBuffer _paperCorrectionKeyHash;
        private ComputeBuffer _paperCorrectionTargetHash;
        private Vector4[] _paperCorrectionKeyEntries;
        private Vector4[] _paperCorrectionTargetEntries;
        private int _paperCorrectionRevision = int.MinValue;
        private int _paperCorrectionCellCount;
        private int _paperCorrectionHashMask = -1;
        private float _heraFreezeStartedAt = -1f;
        private float _heraLastProgressAt = -1f;
        private string _heraLastProgressSignature = "";

        public bool IsChunkAbExperimentEnabled => enableFrozenChunkAbExperiment;
        public bool IsChunkAbFrozen => _chunkAbFrozen;
        public bool IsGunGelCleanRoomEnabled => enableGunGelCleanRoomExperiment;
        public int ActiveChunkAbSize => ChunkAbSizes[Mathf.Clamp(_chunkAbGearIndex, 0, ChunkAbSizes.Length - 1)];

        /// <summary>当前是否线框显示。</summary>
        public bool IsWireframe => wireframeMode;
        private bool FrozenBlockSupervisorEffective =>
            enableFrozenBlockSupervisor && !enableGunGelCleanRoomExperiment;

        private bool GunGelGuardedFusionEnabled =>
            _volumeIntegrator != null && _volumeIntegrator.GunGelGuardedFusionExperimentEnabled;
        private bool FinalCourtAdmissionEnabled =>
            _volumeIntegrator != null && _volumeIntegrator.FinalCourtAdmissionExperimentEnabled;

        /// <summary>
        /// 当前唯一生产身份。GunGel 只保留 stableId 与候选证词；原始健康
        /// 深度进入唯一 TSDF，裁判平面在提取后的块产品化阶段才有修改权。
        /// </summary>
        private string CaptureModeLabel =>
            _volumeIntegrator != null && _volumeIntegrator.GunGelIdentityOnlyMode
            ? "制品"
            : FinalCourtAdmissionEnabled
            ? (enableGunGelCleanRoomExperiment ? "裁活" : "裁冻")
            : enableGunGelCleanRoomExperiment
                ? (GunGelGuardedFusionEnabled ? "胶活" : "原活")
                : (GunGelGuardedFusionEnabled ? "胶冻" : "原冻");

        private string CaptureModeToken =>
            _volumeIntegrator != null && _volumeIntegrator.GunGelIdentityOnlyMode
            ? "single_tsdf_productized"
            : FinalCourtAdmissionEnabled
            ? (enableGunGelCleanRoomExperiment ? "court_live" : "court_freeze")
            : enableGunGelCleanRoomExperiment
                ? (GunGelGuardedFusionEnabled ? "gel_live" : "base_live")
                : (GunGelGuardedFusionEnabled ? "gel_freeze" : "base_freeze");

        private void SyncCaptureModeIdentity()
        {
            _meshExtractor?.SetCaptureModeIdentity(CaptureModeLabel, CaptureModeToken);
        }

        /// <summary>在线框 / 顶点色实体之间切换（QRS SetRenderMode 的二态精简版）。</summary>
        public void ToggleWireframe()
        {
            if (_meshExtractor != null && _meshExtractor.IsPaperFineHybridVisible)
            {
                bool paperGrid = _meshExtractor.TogglePaperOwnedGrid();
                NotifyInput(paperGrid ? "纸网：v2.3纸拓扑" : "纸网：HERA旧网格");
                RefreshStatusBadge();
                return;
            }

            wireframeMode = !wireframeMode;
            ApplyDisplayMode();
            bool supportTruthOnly = _meshExtractor != null &&
                                    _meshExtractor.IsSupportTruthOnlyVisible;
            _meshExtractor?.SetSupportTruthAudit(wireframeMode);
            if (supportTruthOnly)
                NotifyInput(wireframeMode ? "支撑：圆点原料" : "支撑：纸拓扑独显");
            else
                NotifyInput(wireframeMode ? "切到线框" : "切到实体");
        }

        /// <summary>第一阶段白网 A/B：只切最终着色，所有后台数据与调度继续运行。</summary>
        public void ToggleGeometryTruthView()
        {
            geometryTruthView = !geometryTruthView;
            ApplyDisplayMode();
            NotifyInput(geometryTruthView ? "纯白几何：开" : "纯白几何：关");
            RefreshStatusBadge();
        }

        /// <summary>
        /// Performance A/B switch. Only the QRS draw call changes; the complete
        /// depth, fusion, extraction, admission and ledger pipeline keeps running.
        /// </summary>
        public void ToggleProductionMeshRendering()
        {
            if (_meshExtractor == null) return;
            bool visible = _meshExtractor.ToggleProductionMeshVisible();
            NotifyInput(visible ? "网格显示：开" : "网格显示：关（后台继续）");
            RefreshStatusBadge();
        }

        /// <summary>
        /// Display-only GPU A/B switch.  The visible wire pass and the entire
        /// acquisition/fusion/extraction pipeline stay unchanged; only the
        /// preceding full-surface depth-only draw is admitted or skipped.
        /// </summary>
        public void ToggleRearWireDepthPrepass()
        {
            bool enabled = GPUMeshRenderer.ToggleRearWireDepthPrepass();
            string state = enabled ? "ON" : "OFF";
            Logger.Info($"QRS rear-wire depth prepass: {state} (mesh remains visible)");
            NotifyInput(enabled
                ? "网格深度预绘：开"
                : "网格深度预绘：关（网格仍显示）");
            RefreshStatusBadge();
        }

        /// <summary>
        /// Display-only A/B switch between hardware line primitives and the
        /// legacy full-triangle fragment-discard wireframe. Reconstruction,
        /// mesh buffers, HERA ownership and scheduling are unchanged.
        /// </summary>
        public void ToggleTrueLinePrimitives()
        {
            bool enabled = GPUMeshRenderer.ToggleTrueLinePrimitives();
            string state = enabled ? "TRUE_LINES" : "PSEUDO_WIRE";
            Logger.Info($"QRS wire renderer: {state} (mesh data unchanged)");
            NotifyInput(enabled
                ? "线法：真线段"
                : "线法：挖空三角");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 路线验证总闸（右摇杆直接按下）：切"当前真正在画网格的那条路径"。
        /// A/B 实验旗下开机即 PrepareForChunkAbAcquisition，renderProductionMesh 恒
        /// false、满屏网全来自增量 HERA——所以扫描中必须切增量 HERA，切旧字段是空转。
        /// 支撑真值/三角粗皮下暂停 HERA；纸网合流同时保留稳定底纸和增量 HERA。
        /// </summary>
        public void ToggleMeshDisplay()
        {
            if (_meshExtractor == null) return;
            // 点诊断和纸/HERA共用一个前景席位。若 X 档仍保持 Visible，Update()
            // 会在下一帧再次把刚切出的纸面压回隐藏。切网格档时先明确退出点诊断，
            // 但不恢复旧快照；下面的路线切换/总闸负责只打开用户刚选中的新档。
            if ((_gunGelCourtOverlay != null && _gunGelCourtOverlay.Visible) ||
                (_depthPointCloudOverlay != null && _depthPointCloudOverlay.Visible) ||
                (_instantDepthShellOverlay != null && _instantDepthShellOverlay.Visible) ||
                _shellPaperOnlyView)
            {
                _gunGelCourtOverlay?.SetVisible(false);
                _depthPointCloudOverlay?.SetVisible(false);
                _depthPointCloudOverlay?.SetAcquiring(false);
                _instantDepthShellOverlay?.SetVisible(false);
                _instantDepthShellOverlay?.SetAcquiring(IsScanning);
                _bbPresentationCaptured = false;
                _shellPaperOnlyView = false;
                ApplyDisplayMode();
            }
            if (_meshExtractor.IsRouteValidationActive)
            {
                string route = _meshExtractor.CycleRouteValidationView();
                NotifyInput($"显示：{route}");
                RefreshStatusBadge();
                return;
            }
            bool visible = _meshExtractor.HasIncrementalHera
                ? _meshExtractor.ToggleIncrementalHeraVisible()
                : _meshExtractor.ToggleProductionMeshVisible();
            NotifyInput(visible ? "网格显示：开" : "网格显示：关（后台继续）");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 明确进入生产观察档：只让真实 TSDF 的公共纸皮拥有前景。
        /// 红绿点、BB 点云、粗皮、HERA 对照与管理框全部退出；后台生产不变。
        /// </summary>
        public void ShowProductionPaperView()
        {
            _gunGelCourtOverlay?.SetVisible(false);
            _depthPointCloudOverlay?.SetVisible(false);
            _depthPointCloudOverlay?.SetAcquiring(false);
            _instantDepthShellOverlay?.SetVisible(false);
            _instantDepthShellOverlay?.SetAcquiring(IsScanning);
            _coverageOverlay?.SetMarkersVisible(false);
            showManagementBlockWireOverlay = false;
            _managementBlockWireOverlay?.SetVisible(false);
            _bbPresentationCaptured = false;
            _shellPaperOnlyView = false;
            ApplyDisplayMode();

            string route = _meshExtractor != null
                ? _meshExtractor.ShowProductionPaperView()
                : "无显示";
            NotifyInput($"生产观察：{route}");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 性能二分热键（右摇杆上）：实时轨开关。关闭后已排实时页保留，
        /// 但不再排新页——用于隔离"实时轨提取 churn"对帧率的贡献。
        /// </summary>
        public void ToggleLiveTrack()
        {
            enableLiveTrack = !enableLiveTrack;
            NotifyInput(enableLiveTrack ? "实时轨:开" : "实时轨:关");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 性能二分热键（右摇杆左）：冻结调度器开关。关闭即 Tick 早退，
        /// 停 2s 全体积普查 + 票箱回读；在途回读自然完成一次无害。
        /// 注意：已冻结块保持冻结态不再解冻——纯二分诊断用，勿当常态。
        /// </summary>
        public void ToggleFrozenBlockSupervisor()
        {
            enableFrozenBlockSupervisor = !enableFrozenBlockSupervisor;
            NotifyInput(enableFrozenBlockSupervisor ? "冻结调度:开" : "冻结调度:关");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 第一轮旧机制追责：在同一 APK 中切换“枪胶净室”与“旧冻结监督”。
        /// 必须在空卷、尚未开始扫描时切换，避免两套规则污染同一份 TSDF。
        /// 净室只隔离逐块冻结/解冻；枪胶守门、实时页生产和扣减规则保持不变。
        /// </summary>
        public void ToggleGunGelCleanRoomExperiment()
        {
            bool volumeHasData = _volumeIntegrator != null && _volumeIntegrator.IntegrationCount > 0;
            if (IsScanning || HasStarted || _chunkAbFrozen || volumeHasData || _frozenBlocks.Count > 0)
            {
                NotifyInput("净室切换:需重启空卷");
                RefreshStatusBadge();
                return;
            }

            enableGunGelCleanRoomExperiment = !enableGunGelCleanRoomExperiment;
            ResetFrozenBlockSupervisor();
            NotifyInput(enableGunGelCleanRoomExperiment
                ? "追责:净室(逐块冻停)"
                : "追责:旧冻机制");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 性能二分热键（右摇杆下）：融合频率 20Hz ↔ 10Hz 切换。
        /// </summary>
        public void ToggleIntegrationRate()
        {
            integrationHz = integrationHz > 15f ? 10f : 20f;
            NotifyInput($"融合:{integrationHz:0}Hz");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 性能二分热键（右摇杆右）：深度预处理三态循环（全开→半(只缘洗)→全关）。
        /// 实机 24→72 已证明预处理链是帧率主猪；三态用于定位链内双边/缘洗谁贵，
        /// "半"档保留缘洗=保住幽灵桥防护（v2.0 否决链质量收益），是候选生产档。
        /// </summary>
        public void ToggleDepthPreprocessing()
        {
            if (_depthCapture == null) return;
            int mode = _depthCapture.CycleDepthPreprocessingMode();
            NotifyInput(mode == 0 ? "深滤:全开" : mode == 1 ? "深滤:半(只缘洗)" : "深滤:全关");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 性能二分热键（左摇杆上）：源头时序滤波开关。盯墙养绿 A/B 用——
        /// 同墙同段实时切换（关闭侧=pre-时序滤波基线），滤波内部自门控清历史，
        /// 重开不吃隔夜残影。HUD 闸行 时开/时关 回显当前档位。
        /// </summary>
        public void ToggleTemporalFilter()
        {
            if (_depthCapture == null) return;
            NotifyInput(_depthCapture.ToggleTemporalFilter() ? "时滤:开" : "时滤:关");
            RefreshStatusBadge();
        }

        /// <summary>左摇杆右+按：32³融合管理块线框只读层。</summary>
        public void ToggleManagementBlockWireOverlay()
        {
            showManagementBlockWireOverlay = !showManagementBlockWireOverlay;
            EnsureManagementBlockWireOverlay();
            _managementBlockWireOverlay?.SetVisible(showManagementBlockWireOverlay);
            NotifyInput(showManagementBlockWireOverlay ? "32块框:开" : "32块框:关");
            RefreshStatusBadge();
        }

        /// <summary>
        /// 本轮验收锁定裁冻。按键仍保留，但只会重申唯一生产路线；原冻和
        /// 普通胶冻不能再因 Scene 遗留值或误触重新取得 TSDF 写入权。
        /// </summary>
        public void ToggleGunGelGuardedFusionExperiment()
        {
            if (_volumeIntegrator == null)
            {
                NotifyInput("枪胶融:未就绪");
                return;
            }

            bool volumeHasData = _volumeIntegrator.IntegrationCount > 0;
            if (IsScanning || HasStarted || _chunkAbFrozen || volumeHasData || _frozenBlocks.Count > 0)
            {
                NotifyInput($"{CaptureModeLabel}已锁定:需重启空卷");
                RefreshStatusBadge();
                return;
            }

            // 显示、冻结监督和生产纸皮保持原链；只锁定融合准入来源。
            enableGunGelCleanRoomExperiment = false;
            enableFrozenBlockSupervisor = true;
            ResetFrozenBlockSupervisor();
            _volumeIntegrator.ToggleGunGelGuardedFusionExperiment();
            SyncCaptureModeIdentity();
            NotifyInput($"对照模式:{CaptureModeLabel}");
            RefreshStatusBadge();
        }

        [Header("日志")]
        [SerializeField] private LogLevel logLevel = LogLevel.Info;

        private DepthCapture _depthCapture;
        private VolumeIntegrator _volumeIntegrator;
        private MeshExtractor _meshExtractor;
        private PassthroughCameraProvider _cameraProvider;

        [Header("Persistent Primary Status Badge")]
        [SerializeField, Tooltip("开启后恢复冻结票、置信度、页面债、计时账等完整诊断 HUD；默认隐藏，后台统计不受影响。")]
        private bool showDetailedRuntimeStatus = false;
        // Current GPU acceptance build: keep the headset plate limited to the
        // four switches that define the rendering A/B sample. This is runtime
        // state only; all ledgers and logging continue in the background.
        private bool _renderAbHudOnly = true;
        private UnityEngine.UI.Text _statusBadgeText;
        private UnityEngine.UI.Text _statusBadgeHeaderText;
        private UnityEngine.UI.Text _statusBadgeRightText;
        private GameObject _statusBadgeRoot;
        private bool _statusBadgeCreationPending;
        private bool _meshTailSealHudForcedVisible;
        private string _meshTailSealHudState = "未开始";
        private const float DiagnosticHudWidth = 2100f;
        private const float DiagnosticHudHeaderHeight = 104f;
        private const float DiagnosticHudPadding = 24f;
        private GameObject _probeReticleRoot;
        private RectTransform _probeTargetMarkerRect;
        private UnityEngine.UI.Text _probeReticleLabel;
        private ProbeEvidenceGraphic _probeEvidenceGraphic;
        private readonly VirtualProbeShadowAdjudicator.GuidanceCellVisual[]
            _probeEvidenceCells =
                new VirtualProbeShadowAdjudicator.GuidanceCellVisual[96];
        private readonly List<UnityEngine.UI.Graphic> _probeFrameGraphics =
            new List<UnityEngine.UI.Graphic>(4);
        private readonly List<UnityEngine.UI.Graphic> _probeMarkerGraphics =
            new List<UnityEngine.UI.Graphic>(5);

        /// <summary>正在融合（未暂停）。</summary>
        public bool IsScanning { get; private set; }
        /// <summary>本次运行期间曾经开始过扫描（暂停后为 true，清空后归 false）。</summary>
        public bool HasStarted { get; private set; }
        public bool IsSaveAndClearInProgress { get; private set; }

        public event Action ScanStarted;
        public event Action ScanStopped;
        /// <summary>每次深度融合后触发。</summary>
        public event Action Integrated;
        /// <summary>每次网格提取后触发。</summary>
        public event Action MeshExtracted;

        public VolumeIntegrator VolumeIntegrator => _volumeIntegrator;
        public DepthCapture DepthCapture => _depthCapture;
        public MeshExtractor MeshExtractor => _meshExtractor;

        private float _lastIntegrationTime;
        private float _lastMeshTime;
        private float _lastInfiniTamHeavyWorkTime = -1000f;
        private float _infiniTamFrameSecondsEma = 1f / 72f;
        private int _infiniTamHealthyFrameStreak;
        // Fresh-volume startup only. The first whole-volume bootstrap remains
        // intact, but cannot share the volume bring-up window. Once one real
        // fusion has been submitted, leave one ordinary mesh interval before
        // the first extraction; after that extraction this gate is gone for the
        // rest of the scan and the accepted 10 Hz / 5 Hz scheduler is untouched.
        private bool _infiniTamStartupFirstMeshPending;
        private bool _infiniTamStartupFirstFusionSubmitted;
        private float _infiniTamStartupFirstMeshNotBefore;
        private float _lastScannerLog;
        private int _integrateCount;

        // Isolated production-tail validation.  While active, the authoritative
        // TSDF receives no writes; only the existing InfiniTAM dirty ledger,
        // extraction queue and immutable-front commits are allowed to drain.
        private const float MeshTailValidationTimeoutSeconds = 20f;
        private const int MeshTailValidationStableTicks = 4;
        private bool _meshTailValidationActive;
        private float _meshTailValidationStartedAt;
        private DateTime _meshTailValidationStartedUtc;
        private int _meshTailValidationStableTicks;
        private int _meshTailStartIntegrationCount;
        private uint _meshTailStartDirtyEpoch;
        private long _meshTailStartLedgerApplyCount;
        private long _meshTailStartAcceptedCommitCount;
        private long _meshTailStartCompletedBatchCount;
        private long _meshTailStartPublishedBatchCount;
        private long _meshTailStartStaleDiscardCount;
        private long _meshTailStartVertexCount;
        private long _meshTailStartIndexCount;
        private int _meshTailStartVisibleBlocks;
        private string _meshTailValidationOutputPath = string.Empty;
        private bool _meshTailReasonLedgerSealRequested;
        private string _meshTailReasonLedgerSessionDirectory = string.Empty;
        private readonly StringBuilder _meshTailValidationSamples = new StringBuilder(4096);

        private float IntegrationInterval => 1f / Mathf.Max(1f, integrationHz);
        private float MeshInterval => 1f / Mathf.Max(1f, meshExtractionHz);

        // ─────────────────────────────────────────────────────────────
        //  生命周期
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            Instance = this;
            // Every fresh app run starts from the production rendering path;
            // the operator then performs an explicit same-run A/B toggle.
            GPUMeshRenderer.SetRearWireDepthPrepassEnabled(true);
            GPUMeshRenderer.SetTrueLinePrimitivesEnabled(true);
            // This performance A/B build keeps only the compact operator badge
            // visible so the headset can confirm mesh/prepass/surface state.
            // The large forensic HUD and ROI frame remain locked off.
            showOperatorHud = true;
            showDebugHud = false;
            showDiagnosticRoiFrame = false;
            Logger.Level = logLevel;
            _depthCapture = GetComponent<DepthCapture>();
            _volumeIntegrator = GetComponent<VolumeIntegrator>();
            _meshExtractor = GetComponent<MeshExtractor>();
            _cameraProvider = GetComponent<PassthroughCameraProvider>();
            if (_volumeIntegrator != null && _volumeIntegrator.InfiniTamBaselineEnabled)
            {
                // The architecture baseline owns one reconstruction and one
                // renderer.  Do not let a serialized A/B flag tear it down and
                // replace it with HERA/freeze acquisition during StartScanning.
                enableFrozenChunkAbExperiment = false;
                // Device A/B: 20Hz kept FPS in the 50s even with solid rendering;
                // 10Hz recovered the 72Hz envelope. Lock the production cadence
                // here so a serialized scene cannot silently restore 20Hz.
                integrationHz = 10f;
                _depthCapture?.ApplyProductionHalfPreprocessing();
                // One block at 5 Hz matches the dirty-ledger cadence: the mesh
                // gets a predictable small slot instead of two-block bursts
                // after fusion has starved it for 1.5 intervals.
                meshExtractionHz = 5f;
            }
            SyncCaptureModeIdentity();
            EnsureManagementBlockWireOverlay();
            _volumeIntegrator.Cleared += ResetFrozenBlockSupervisor;
            SetSafeShaderDefaults();
        }

        private void Start()
        {
            if (!XRRuntimeGuard.IsXRActive)
            {
                Logger.Warning("StandaloneRoomScanner: " + XRRuntimeGuard.EditorDisabledMessage);
                enabled = false;
                return;
            }

            if (enableHeadExclusion) SetupHeadExclusion();
            if (enableFrozenChunkAbExperiment)
            {
                _coverageOverlay = GetComponent<ObservationCoverageOverlay>();
                if (_coverageOverlay == null)
                    _coverageOverlay = gameObject.AddComponent<ObservationCoverageOverlay>();
                _meshExtractor.PrepareForChunkAbAcquisition();
            }
            _depthPointCloudOverlay = GetComponent<DepthPointCloudOverlay>();
            if (_depthPointCloudOverlay == null)
                _depthPointCloudOverlay = gameObject.AddComponent<DepthPointCloudOverlay>();
            _depthPointCloudOverlay.SetVisible(false);
            _gunGelCourtOverlay = GetComponent<GunGelCourtOverlay>();
            if (_gunGelCourtOverlay == null)
                _gunGelCourtOverlay = gameObject.AddComponent<GunGelCourtOverlay>();
            _gunGelCourtOverlay.SetVisible(false);
            _instantDepthShellOverlay = GetComponent<InstantDepthShellOverlay>();
            if (_instantDepthShellOverlay == null)
                _instantDepthShellOverlay = gameObject.AddComponent<InstantDepthShellOverlay>();
            _instantDepthShellOverlay.SetVisible(false);
            StartCoroutine(ConfigureCameraForPassthrough());
            // Current performance A/B builds keep the compact operator badge
            // visible; the large diagnostic body remains disabled.
            if (showOperatorHud)
                StartCoroutine(CreateStatusBadgeWhenCameraReady());
            if (showDebugHud)
                StartCoroutine(CreateHudWhenCameraReady());
            Application.logMessageReceived += OnLogMessage;
            Logger.Info(enableFrozenChunkAbExperiment
                ? (enableHeraHierarchicalReplay
                    ? (enableIncrementalHeraRefine
                        ? "增量精修就绪 — 扳机采集，X 裁决海/BB，A 冻结全场回放"
                        : "HERA 就绪 — 扳机采集，A 冻结并自动 32→16，B 导出并清派生网格")
                    : "切块 A/B 就绪 — 扳机采集，A 冻结，Y 换 64/32/16，B 导出并清当前档")
                : "QRS 独立链就绪 — 右手柄扳机开始扫描，A 暂停，B 停止清空");
        }

        private System.Collections.IEnumerator CreateStatusBadgeWhenCameraReady()
        {
            if (_statusBadgeRoot != null)
            {
                _statusBadgeRoot.SetActive(true);
                yield break;
            }
            if (_statusBadgeCreationPending)
                yield break;
            _statusBadgeCreationPending = true;

            while (Camera.main == null)
                yield return null;

            if (_statusBadgeRoot != null)
            {
                _statusBadgeRoot.SetActive(true);
                _statusBadgeCreationPending = false;
                yield break;
            }

            Font font = null;
            try { font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } catch { }
            if (font == null)
            {
                try { font = Resources.GetBuiltinResource<Font>("Arial.ttf"); } catch { }
            }

            var root = new GameObject("[QRS] Minimal Status Badge");
            _statusBadgeRoot = root;
            root.transform.SetParent(Camera.main.transform, false);
            // Keep the compact operator prompt close to the optical centre.
            // A centred pivot prevents its two-line height from pushing the
            // whole plate downward as the prompt changes.
            root.transform.localPosition = new Vector3(0f, -0.05f, 0.9f);
            root.transform.localRotation = Quaternion.identity;
            root.transform.localScale = Vector3.one * 0.00036f;

            var canvas = root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            // This canvas owns the diagnostic text only, not the performance graphs.
            canvas.sortingOrder = 32760;
            var rootRect = root.GetComponent<RectTransform>();
            rootRect.pivot = new Vector2(0.5f, 0.5f);
            rootRect.sizeDelta = new Vector2(DiagnosticHudWidth, 640f);

            Material badgeMaterial = null;
            Material badgeBackgroundMaterial = null;
            var badgeShader = Resources.Load<Shader>("HUDAlwaysOnTop");
            if (badgeShader != null)
            {
                badgeBackgroundMaterial = new Material(badgeShader)
                {
                    renderQueue = 4990
                };
                badgeMaterial = new Material(badgeShader)
                {
                    renderQueue = 4991
                };
            }

            // Retired acquisition square and oversized yellow phase counters.
            // Do not create their canvas; FPS/GPU overlays are independent.
            if (_probeReticleRoot != null) _probeReticleRoot.SetActive(false);

            var bg = new GameObject("Bg");
            bg.transform.SetParent(root.transform, false);
            var image = bg.AddComponent<UnityEngine.UI.Image>();
            // This is an occluding plate, not a translucent tint.  A dedicated
            // render queue places it after every scan/diagnostic wire material;
            // text uses the next queue so the plate can never cover its owner.
            image.color = new Color(0.008f, 0.012f, 0.018f, 1f);
            if (badgeBackgroundMaterial != null)
                image.material = badgeBackgroundMaterial;
            var bgRect = bg.GetComponent<RectTransform>();
            bgRect.anchorMin = Vector2.zero;
            bgRect.anchorMax = Vector2.one;
            bgRect.offsetMin = Vector2.zero;
            bgRect.offsetMax = Vector2.zero;

            _statusBadgeHeaderText = CreateDiagnosticHudText(root.transform, "Header", font,
                badgeMaterial, 0f, 1f, DiagnosticHudPadding);
            _statusBadgeText = CreateDiagnosticHudText(root.transform, "Relay", font,
                badgeMaterial, 0f, 0.5f, DiagnosticHudHeaderHeight);
            _statusBadgeRightText = CreateDiagnosticHudText(root.transform, "Attribution", font,
                badgeMaterial, 0.5f, 1f, DiagnosticHudHeaderHeight);
            _statusBadgeCreationPending = false;
            RefreshStatusBadge();
        }

        private void ShowMeshTailSealHud(string tailState)
        {
            _meshTailSealHudForcedVisible = true;
            _meshTailSealHudState = string.IsNullOrEmpty(tailState) ? "处理中" : tailState;
            if (_statusBadgeRoot != null)
            {
                _statusBadgeRoot.SetActive(true);
                RefreshStatusBadge();
                return;
            }

            if (!_statusBadgeCreationPending)
                StartCoroutine(CreateStatusBadgeWhenCameraReady());
        }

        private static UnityEngine.UI.Text CreateDiagnosticHudText(Transform root, string name,
            Font font, Material material, float left, float right, float top)
        {
            var textObject = new GameObject(name);
            textObject.transform.SetParent(root, false);
            var text = textObject.AddComponent<UnityEngine.UI.Text>();
            text.font = font;
            text.fontSize = 30;
            text.alignment = TextAnchor.UpperCenter;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            text.raycastTarget = false;
            if (material != null) text.material = material;
            var outline = textObject.AddComponent<UnityEngine.UI.Outline>();
            outline.effectColor = new Color(0f, 0f, 0f, 0.95f);
            outline.effectDistance = new Vector2(2f, -2f);
            var textRect = textObject.GetComponent<RectTransform>();
            textRect.anchorMin = new Vector2(left, 0f);
            textRect.anchorMax = new Vector2(right, 1f);
            textRect.offsetMin = new Vector2(DiagnosticHudPadding, DiagnosticHudPadding);
            textRect.offsetMax = new Vector2(-DiagnosticHudPadding, -top);
            return text;
        }

        private void CreateProbeTargetReticle(Camera cam, Font font, Material material)
        {
            if (cam == null || _probeReticleRoot != null) return;

            var go = new GameObject("[QRS] Reject Acquisition Reticle");
            _probeReticleRoot = go;
            go.transform.SetParent(cam.transform, false);

            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 32767;

            const float distance = 1f;
            const float canvasWidthPx = 1000f;
            float viewHeight = 2f * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * distance;
            float viewWidth = viewHeight * Mathf.Max(0.1f, cam.aspect);
            float canvasHeightPx = canvasWidthPx * viewHeight / Mathf.Max(0.001f, viewWidth);
            var root = go.GetComponent<RectTransform>();
            root.sizeDelta = new Vector2(canvasWidthPx, canvasHeightPx);
            go.transform.localPosition = Vector3.forward * distance;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one * (viewWidth / canvasWidthPx);

            const float halfAngleDeg = 7f;
            float halfY = 0.5f * Mathf.Tan(halfAngleDeg * Mathf.Deg2Rad) /
                          Mathf.Max(0.001f, Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad));
            float halfX = halfY / Mathf.Max(0.1f, cam.aspect);
            float x0 = Mathf.Clamp01(0.5f - halfX);
            float x1 = Mathf.Clamp01(0.5f + halfX);
            float y0 = Mathf.Clamp01(0.5f - halfY);
            float y1 = Mathf.Clamp01(0.5f + halfY);
            Color waiting = new Color(1f, 0.76f, 0.16f, 0.95f);

            _probeFrameGraphics.Add(AddProbeReticleLine(go.transform, "框左",
                new Vector2(x0, y0), new Vector2(x0, y1), new Vector2(3f, 0f), waiting, material));
            _probeFrameGraphics.Add(AddProbeReticleLine(go.transform, "框右",
                new Vector2(x1, y0), new Vector2(x1, y1), new Vector2(3f, 0f), waiting, material));
            _probeFrameGraphics.Add(AddProbeReticleLine(go.transform, "框下",
                new Vector2(x0, y0), new Vector2(x1, y0), new Vector2(0f, 3f), waiting, material));
            _probeFrameGraphics.Add(AddProbeReticleLine(go.transform, "框上",
                new Vector2(x0, y1), new Vector2(x1, y1), new Vector2(0f, 3f), waiting, material));

            var evidenceGo = new GameObject("框内证据状态");
            evidenceGo.transform.SetParent(go.transform, false);
            _probeEvidenceGraphic = evidenceGo.AddComponent<ProbeEvidenceGraphic>();
            _probeEvidenceGraphic.raycastTarget = false;
            _probeEvidenceGraphic.color = Color.white;
            if (material != null) _probeEvidenceGraphic.material = material;
            var evidenceRect = evidenceGo.GetComponent<RectTransform>();
            evidenceRect.anchorMin = Vector2.zero;
            evidenceRect.anchorMax = Vector2.one;
            evidenceRect.offsetMin = Vector2.zero;
            evidenceRect.offsetMax = Vector2.zero;

            var labelGo = new GameObject("取景状态");
            labelGo.transform.SetParent(go.transform, false);
            _probeReticleLabel = labelGo.AddComponent<UnityEngine.UI.Text>();
            _probeReticleLabel.text = "等待双采";
            _probeReticleLabel.font = font;
            _probeReticleLabel.fontSize = 23;
            _probeReticleLabel.fontStyle = FontStyle.Bold;
            _probeReticleLabel.alignment = TextAnchor.UpperCenter;
            _probeReticleLabel.color = waiting;
            _probeReticleLabel.raycastTarget = false;
            if (material != null) _probeReticleLabel.material = material;
            var labelRect = labelGo.GetComponent<RectTransform>();
            labelRect.anchorMin = new Vector2(0.5f, y1);
            labelRect.anchorMax = new Vector2(0.5f, y1);
            labelRect.pivot = new Vector2(0.5f, 0f);
            labelRect.anchoredPosition = new Vector2(0f, 10f);
            labelRect.sizeDelta = new Vector2(720f, 104f);

            var markerGo = new GameObject("Reject真实位置");
            markerGo.transform.SetParent(go.transform, false);
            _probeTargetMarkerRect = markerGo.AddComponent<RectTransform>();
            _probeTargetMarkerRect.anchorMin = new Vector2(0.5f, 0.5f);
            _probeTargetMarkerRect.anchorMax = new Vector2(0.5f, 0.5f);
            _probeTargetMarkerRect.pivot = new Vector2(0.5f, 0.5f);
            _probeTargetMarkerRect.sizeDelta = new Vector2(48f, 48f);
            _probeMarkerGraphics.Add(AddProbeMarkerBar(markerGo.transform, "左", new Vector2(-18f, 0f), new Vector2(14f, 3f), waiting, material));
            _probeMarkerGraphics.Add(AddProbeMarkerBar(markerGo.transform, "右", new Vector2(18f, 0f), new Vector2(14f, 3f), waiting, material));
            _probeMarkerGraphics.Add(AddProbeMarkerBar(markerGo.transform, "上", new Vector2(0f, 18f), new Vector2(3f, 14f), waiting, material));
            _probeMarkerGraphics.Add(AddProbeMarkerBar(markerGo.transform, "下", new Vector2(0f, -18f), new Vector2(3f, 14f), waiting, material));
            _probeMarkerGraphics.Add(AddProbeMarkerBar(markerGo.transform, "中心", Vector2.zero, new Vector2(5f, 5f), waiting, material));
            markerGo.SetActive(false);
        }

        private static UnityEngine.UI.Image AddProbeReticleLine(
            Transform parent, string name, Vector2 anchorMin, Vector2 anchorMax,
            Vector2 sizeDelta, Color color, Material material)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<UnityEngine.UI.Image>();
            image.color = color;
            image.raycastTarget = false;
            if (material != null) image.material = material;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.anchoredPosition = Vector2.zero;
            rt.sizeDelta = sizeDelta;
            return image;
        }

        private static UnityEngine.UI.Image AddProbeMarkerBar(
            Transform parent, string name, Vector2 position, Vector2 size,
            Color color, Material material)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<UnityEngine.UI.Image>();
            image.color = color;
            image.raycastTarget = false;
            if (material != null) image.material = material;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.anchoredPosition = position;
            rt.sizeDelta = size;
            return image;
        }

        private void UpdateProbeTargetReticle()
        {
            if (_probeReticleRoot == null || _probeReticleLabel == null) return;
            ScanReplaySessionPackage session = ScanReplaySessionPackage.Active;
            var frame = session != null
                ? session.ProbeFollowupGuidanceFrame
                : default;
            int evidenceCount = session != null
                ? session.CopyProbeFollowupGuidanceVisuals(_probeEvidenceCells)
                : 0;
            _probeEvidenceGraphic?.SetEvidence(Camera.main, _probeEvidenceCells,
                evidenceCount);
            int phaseProvisional = 0, phaseStable = 0, phaseVote1 = 0,
                phaseVote2 = 0, phaseReject = 0, phaseCleared = 0;
            for (int i = 0; i < evidenceCount; i++)
            {
                switch (_probeEvidenceCells[i].Phase)
                {
                    case 0: phaseProvisional++; break;
                    case 1: phaseStable++; break;
                    case 2: phaseVote1++; break;
                    case 3: phaseVote2++; break;
                    case 4: phaseReject++; break;
                    case 5: break;
                    case 6: break;
                    case 7: phaseCleared++; break;
                    case 8: phaseCleared++; break;
                }
            }

            Color color;
            if (session == null)
            {
                color = new Color(0.68f, 0.72f, 0.76f, 0.9f);
            }
            else if (!frame.Active)
            {
                color = new Color(1f, 0.76f, 0.16f, 0.95f);
            }
            else if (frame.Pending > 0 && frame.ObservedRecently > 0)
            {
                color = new Color(0.20f, 1f, 0.38f, 0.98f);
            }
            else if (frame.Pending > 0)
            {
                color = new Color(1f, 0.56f, 0.08f, 0.98f);
            }
            else if (frame.ResolvedFree + frame.ResolvedSupport > 0)
            {
                color = new Color(0.20f, 1f, 0.38f, 0.98f);
            }
            else if (frame.Expired > 0)
            {
                color = new Color(0.70f, 0.72f, 0.76f, 0.95f);
            }
            else
            {
                color = new Color(0.28f, 0.86f, 1f, 0.95f);
            }

            for (int i = 0; i < _probeFrameGraphics.Count; i++)
                if (_probeFrameGraphics[i] != null) _probeFrameGraphics[i].color = color;
            for (int i = 0; i < _probeMarkerGraphics.Count; i++)
                if (_probeMarkerGraphics[i] != null) _probeMarkerGraphics[i].color = color;
            _probeReticleLabel.color = color;
            _probeReticleLabel.text =
                $"自动 ID{frame.RegisteredFingerprints:0000} " +
                $"反1{frame.ChallengeOne:000} 反2{frame.ChallengeTwo:000} " +
                $"拒{frame.Pending:000}\n" +
                $"证空{frame.ResolvedFree:000} 面返{frame.ResolvedSupport:000} " +
                $"撤{frame.ChallengesClearedBySupport:000} 命{frame.ObservedRecently:000}\n" +
                $"窗 灰{phaseProvisional:00}绿{phaseStable:00} " +
                $"黄{phaseVote1:00}橙{phaseVote2:00}红{phaseReject:00}" +
                $"撤迹{phaseCleared:00}";

            if (_probeTargetMarkerRect != null)
                _probeTargetMarkerRect.gameObject.SetActive(false);
        }

        private void RefreshStatusBadge()
        {
            if (_statusBadgeText == null || _statusBadgeHeaderText == null) return;

            if (_renderAbHudOnly)
            {
                RefreshRenderAbHud();
                return;
            }

            string runState;
            if (IsSaveAndClearInProgress)
            {
                runState = "保存";
                _statusBadgeText.color = new Color(1f, 0.78f, 0.2f, 1f);
            }
            else if (_chunkAbFrozen)
            {
                runState = "冻结";
                _statusBadgeText.color = new Color(0.25f, 0.95f, 1f, 1f);
            }
            else if (IsScanning)
            {
                runState = "采集";
                _statusBadgeText.color = new Color(0.25f, 1f, 0.45f, 1f);
            }
            else if (HasStarted)
            {
                runState = "暂停";
                _statusBadgeText.color = new Color(1f, 0.78f, 0.2f, 1f);
            }
            else
            {
                runState = "待机";
                _statusBadgeText.color = new Color(0.75f, 0.8f, 0.85f, 1f);
            }

            string viewState;
            if (_shellPaperOnlyView)
                viewState = "合流仅纸";
            else if (_instantDepthShellOverlay != null && _instantDepthShellOverlay.Visible)
                viewState = _instantDepthShellOverlay.SeedPreviewVisible
                    ? "种子平面"
                    : _instantDepthShellOverlay.DiagnosticFreezeActive
                    ? "即时壳定格"
                    : _instantDepthShellOverlay.CompositeWithProduction
                        ? "壳纸合流"
                        : "即时外壳";
            else if (_gunGelCourtOverlay != null && _gunGelCourtOverlay.Visible)
                viewState = "裁决海";
            else if (_depthPointCloudOverlay != null && _depthPointCloudOverlay.Visible)
                viewState = "BB点云";
            else if (_meshExtractor == null || !_meshExtractor.IsAnyMeshVisible)
                viewState = "显示关闭";
            else
                viewState = _meshExtractor.RouteValidationLabel;

            string productionAdmission =
                _volumeIntegrator != null && _volumeIntegrator.InfiniTamBaselineEnabled
                ? "InfiniTAM V1.3直融"
                : _volumeIntegrator != null && _volumeIntegrator.GunGelIdentityOnlyMode
                ? "原深度TSDF→块裁决"
                : FinalCourtAdmissionEnabled
                ? "裁判平面"
                : GunGelGuardedFusionEnabled
                ? "邻证同面"
                : "原始旁路";

            string probeHud = ScanReplaySessionPackage.Active != null
                ? ScanReplaySessionPackage.Active.ProbeFollowupGuidanceHudFixed
                : ScanReplaySessionPackage.ProbeFollowupGuidanceHudEmpty;
            ScanReplaySessionPackage replayPackage = ScanReplaySessionPackage.Active ??
                                                     ScanReplaySessionPackage.Latest;
            string sealHud = replayPackage != null
                ? replayPackage.SealHudFixed
                : "封包[未开始  ] 校验000% 安全退出[否]";

            float quality = _volumeIntegrator != null
                ? Mathf.Clamp(_volumeIntegrator.MotionQuality, 0f, 9.99f)
                : 0f;
            int angular = Mathf.Clamp(Mathf.RoundToInt(
                _volumeIntegrator != null
                    ? _volumeIntegrator.SmoothedAngularSpeed
                    : 0f), 0, 999);
            float linear = Mathf.Clamp(
                _depthCapture != null ? _depthCapture.SmoothedDepthLinearSpeed : 0f,
                0f, 9.99f);
            int fps = Mathf.Clamp(Mathf.RoundToInt(
                1f / Mathf.Max(0.001f, Time.smoothDeltaTime)), 0, 999);

            string pairedState = _depthCapture == null
                ? "无"
                : _depthCapture.PairedFrameCaptureActive
                    ? "开"
                    : _depthCapture.PairedFrameCapturePending > 0 ? "写" : "关";
            int pairedPending = _depthCapture != null
                ? Mathf.Clamp(_depthCapture.PairedFrameCapturePending, 0, 9999)
                : 0;
            int pairedDropped = _depthCapture != null
                ? Mathf.Clamp(_depthCapture.PairedFrameCaptureDropped, 0, 9999)
                : 0;
            int replayDropped = _depthCapture != null
                ? Mathf.Clamp(_depthCapture.ReplayFusionCaptureDropped, 0, 9999)
                : 0;
            int viewCoverage = Mathf.Clamp(Mathf.RoundToInt(
                _coverageOverlay != null ? _coverageOverlay.CoveragePercent : 0f),
                0, 100);

            // Fixed semantic columns in every standard view. The explicit
            // court-paper inspection gear uses its compact acceptance panel.
            string seedState = _depthCapture != null
                ? _depthCapture.SeedPlaneProductionStatus
                : "无";
            string primaryPrompt = BuildPrimaryHudPrompt(viewState, seedState);
            string baselineProgress = _volumeIntegrator != null &&
                                      _volumeIntegrator.InfiniTamBaselineEnabled
                ? $" 顶{(_meshExtractor != null ? _meshExtractor.LastVertexCount : 0)} " +
                  $"面{(_meshExtractor != null ? _meshExtractor.LastIndexCount / 3 : 0)} " +
                  $"画{(_meshExtractor != null ? _meshExtractor.LastSubmittedDrawVertexCount : 0)}"
                : string.Empty;
            string baselineTicket = _volumeIntegrator != null &&
                                    _volumeIntegrator.InfiniTamBaselineEnabled
                ? "\n" + _volumeIntegrator.GetInfiniTamTicketCompact() +
                  "\n" + _volumeIntegrator.GetInfiniTamModelRaycastCompact() +
                  "\n" + (_meshExtractor != null
                      ? _meshExtractor.InfiniTamBlockStatsCompact
                      : "块前台未就绪")
                : string.Empty;
            string meshDisplayState = _meshExtractor != null &&
                                      _meshExtractor.IsAnyMeshVisible
                ? "开"
                : "关";
            string depthPrepassState = GPUMeshRenderer.RearWireDepthPrepassEnabled
                ? "开"
                : "关";
            string surfaceState = wireframeMode ? "线框" : "实体";
            string wireMethodState = GPUMeshRenderer.TrueLinePrimitivesEnabled
                ? "真线"
                : "挖空";
            string renderAbState =
                $"A/B 网格[{meshDisplayState}] 预绘[{depthPrepassState}] " +
                $"表面[{surfaceState}] 线法[{wireMethodState}]";
            _statusBadgeHeaderText.color = _statusBadgeText.color;
            _statusBadgeRightText.color = _statusBadgeText.color;
            _statusBadgeHeaderText.fontSize = 50;
            _statusBadgeHeaderText.fontStyle = FontStyle.Bold;
            _statusBadgeHeaderText.alignment = TextAnchor.UpperCenter;
            if (_meshTailSealHudForcedVisible)
            {
                string reasonSealHud = _meshTailReasonLedgerSealRequested
                    ? sealHud
                    : "封包[无活动账] 校验000% 安全退出[无账]";
                _statusBadgeHeaderText.text =
                    "▶ A键：双账封口\n" +
                    $"原因账　{reasonSealHud}\n" +
                    $"尾随账　[{_meshTailSealHudState}]";
            }
            else
            {
                _statusBadgeHeaderText.text =
                    $"{renderAbState}\n" +
                    $"▶ {primaryPrompt}\n" +
                    $"状态[{runState}] 视图[{viewState}] 种面[{seedState}] " +
                    $"准入[{productionAdmission}]{baselineProgress}{baselineTicket}";
            }

            // The evidence ledger keeps running, but defaults to hidden.  The
            // headset operator sees only the next action and the few states
            // needed to verify it; the existing inspector switch can restore
            // both forensic columns without changing acquisition semantics.
            _statusBadgeText.gameObject.SetActive(showDetailedRuntimeStatus);
            _statusBadgeRightText.gameObject.SetActive(showDetailedRuntimeStatus);
            if (showDetailedRuntimeStatus)
            {
                _statusBadgeText.fontSize = 30;
                _statusBadgeRightText.fontSize = 30;
                _statusBadgeText.text =
                    (_instantDepthShellOverlay != null ? _instantDepthShellOverlay.RelayDiagnosticsFixed :
                        "[接力·整批采样]\n审计未就绪\n[覆盖与冻结]\n暂无数据") +
                    "\n[口径]\n残=均值/峰值(mm)；无样本不算通过\n读/找=本段命中率；齐=命中内对齐率\n后段只验前段对齐样本，非全场覆盖率";
                _statusBadgeRightText.text =
                    (_instantDepthShellOverlay != null ? _instantDepthShellOverlay.AttributionDiagnosticsFixed :
                        "[错位归因]\n暂无数据\n[融合写入]\n暂无数据") + "\n[运行与采集]\n" +
                    probeHud + "\n" +
                    $"输入 质{quality:0.00} 角{angular:000}°/s " +
                    $"线{linear:0.00}m/s 帧{fps:000}\n" +
                    $"采集 双采[{HudFixedSlot(pairedState, 2)}] 待{pairedPending:0000} " +
                    $"丢{pairedDropped:0000}/{replayDropped:0000} 视域覆{viewCoverage:000}%\n" +
                    sealHud;
            }
            else
            {
                _statusBadgeText.text = string.Empty;
                _statusBadgeRightText.text = string.Empty;
            }

            // Grow below the performance graphs, with no paging, scrolling, or view gates.
            var badgeRect = _statusBadgeText.transform.parent as RectTransform;
            if (badgeRect != null)
            {
                float bodyTop = Mathf.Max(DiagnosticHudHeaderHeight,
                    _statusBadgeHeaderText.preferredHeight + DiagnosticHudPadding * 2f);
                _statusBadgeText.rectTransform.offsetMax = new Vector2(-DiagnosticHudPadding, -bodyTop);
                _statusBadgeRightText.rectTransform.offsetMax = new Vector2(-DiagnosticHudPadding, -bodyTop);
                float badgeHeight = showDetailedRuntimeStatus
                    ? bodyTop + DiagnosticHudPadding + Mathf.Max(
                        _statusBadgeText.preferredHeight, _statusBadgeRightText.preferredHeight)
                    : bodyTop + DiagnosticHudPadding;
                badgeRect.sizeDelta = new Vector2(DiagnosticHudWidth,
                    Mathf.Max(showDetailedRuntimeStatus ? 640f : 190f, badgeHeight));
            }
        }

        private void RefreshRenderAbHud()
        {
            string meshState = _meshExtractor != null &&
                               _meshExtractor.IsAnyMeshVisible
                ? "开"
                : "关";
            string surfaceState = wireframeMode ? "线框" : "实体";
            string prepassState = GPUMeshRenderer.RearWireDepthPrepassEnabled
                ? "开"
                : "关";
            string lineMethodState = GPUMeshRenderer.TrueLinePrimitivesEnabled
                ? "真线"
                : "挖空";

            _statusBadgeHeaderText.gameObject.SetActive(true);
            _statusBadgeHeaderText.color = new Color(0.92f, 0.98f, 1f, 1f);
            _statusBadgeHeaderText.fontSize = 54;
            _statusBadgeHeaderText.fontStyle = FontStyle.Bold;
            _statusBadgeHeaderText.alignment = TextAnchor.MiddleCenter;
            _statusBadgeHeaderText.text =
                $"网格[{meshState}]　表面[{surfaceState}]\n" +
                $"预绘[{prepassState}]　线法[{lineMethodState}]";

            // Keep the old forensic text objects alive for source compatibility,
            // but never submit them to the canvas in this acceptance build.
            _statusBadgeText.text = string.Empty;
            _statusBadgeRightText.text = string.Empty;
            _statusBadgeText.gameObject.SetActive(false);
            _statusBadgeRightText.gameObject.SetActive(false);

            var badgeRect = _statusBadgeHeaderText.transform.parent as RectTransform;
            if (badgeRect != null)
            {
                float height = Mathf.Max(
                    180f,
                    _statusBadgeHeaderText.preferredHeight +
                    DiagnosticHudPadding * 2f);
                badgeRect.sizeDelta = new Vector2(1500f, height);
            }
        }

        private string BuildPrimaryHudPrompt(string viewState, string seedState)
        {
            if (IsSaveAndClearInProgress)
                return "请等待：正在保存并清卷";

            if (_depthCapture == null)
                return "等待深度入口就绪";

            if (_depthCapture.SeedPlaneAwaitClear)
                return "基底已取好：按 B 保存并清卷";

            if (_instantDepthShellOverlay != null &&
                _instantDepthShellOverlay.AutomaticSeedCaptureActive)
                return _instantDepthShellOverlay.AutomaticSeedAttemptCount == 0
                    ? "自动取融合标尺：请正视一块墙面或天棚"
                    : $"自动寻找合格基底（第{_instantDepthShellOverlay.AutomaticSeedAttemptCount}次）";

            bool seedPreview = _instantDepthShellOverlay != null &&
                               _instantDepthShellOverlay.SeedPreviewVisible;
            if (seedPreview)
            {
                string preview = _instantDepthShellOverlay.SeedPreviewStatus ?? string.Empty;
                if (preview.Contains("拟合中"))
                    return "保持头部稳定：正在拟合绿色基底";
                if (preview.Contains("不成板") || preview.Contains("失败") ||
                    preview.Contains("不可用") || preview.Contains("不符"))
                    return "本帧未成板：换正视角后按 A 重拍";
                if (preview.Contains("已送入口"))
                    return "基底已取好：按 X 进入全流程";
                return "观察绿色基底；不满意按 A 重拍";
            }

            if (seedState == "未取样" || seedState == "无" || seedState == "关")
                return IsScanning
                    ? "按 X 切到“种子平面”取基底"
                    : "按扳机开始扫描";

            if (!IsScanning)
                return HasStarted
                    ? "按扳机继续基底全流程扫描"
                    : "按扳机开始基底全流程扫描";

            if (_depthCapture.SeedPlaneProductionReady)
            {
                if (seedState.StartsWith("备", StringComparison.Ordinal))
                    return "保持扫描：等待种面进入生产链";
                if (_shellPaperOnlyView)
                    return "正在验收：观察平整度、贴合度和台地";
                if (_instantDepthShellOverlay != null &&
                    _instantDepthShellOverlay.CompositeWithProduction)
                    return "按 X 查看“合流仅纸”";
                return "种面已入链：按 X 查看壳纸合流";
            }

            return $"等待种面就绪；当前视图[{viewState}]";
        }

        private static string HudFixedSlot(string value, int width)
        {
            value ??= string.Empty;
            if (value.Length > width) return value.Substring(0, width);
            return value.PadRight(width, ' ');
        }

        // 仅保留旧实现供历史排查；实机主 HUD 不再进入这些按阶段换版的分支。
        private void RefreshLegacyStatusBadge()
        {
            if (_statusBadgeText == null) return;

            if (enableFrozenChunkAbExperiment)
            {
                if (_chunkAbFrozen)
                {
                    if (enableHeraHierarchicalReplay)
                    {
                        RefreshFrozenHeraStatusBadge();
                        return;
                    }

                    int built = _meshExtractor != null ? _meshExtractor.FrozenChunkReplayBuilt : 0;
                    int total = _meshExtractor != null ? _meshExtractor.FrozenChunkReplayTotal : 0;
                    bool hasReplay = _meshExtractor != null && _meshExtractor.HasFrozenChunkReplay;
                    string pageStats = _meshExtractor != null ? _meshExtractor.FrozenChunkReplayStats : "";
                    _statusBadgeText.color = new Color(0.25f, 0.95f, 1f, 1f);
                    if (enableHeraHierarchicalReplay)
                    {
                        _statusBadgeText.text = hasReplay
                            ? $"■ HERA 冻结回放 · {built}/{total}个32页\n{pageStats} · B导出清空 · 右摇杆{(_chunkAbDiagnosticColoring ? "状态色" : "单色")}"
                            : "■ HERA 已导出并清空 · 冻结 TSDF 仍保留";
                    }
                    else
                    {
                        _statusBadgeText.text = hasReplay
                            ? $"■ 冻结 · {ActiveChunkAbSize}³ · {built}/{total}页 · {pageStats}\nY换档 B导出清档 右摇杆{(_chunkAbDiagnosticColoring ? "状态色" : "单色")}"
                            : $"■ TSDF 已冻结 · {ActiveChunkAbSize}³ 当前档已清\nY 换档 · B 仅导出当前档 · 右摇杆切显示";
                    }
                    return;
                }

                if (IsScanning)
                {
                    // HUD 防抖（08-21 拍板"让它们老实点"）：扫描中遥测 0.4s 节流——
                    // 原随提取节拍 ~18/s 重排文本，数字全在蹦迪根本没法盯。暂停/回放等
                    // 事件分支不节流（状态切换要即时）。行为层读数（角/质/探头）钉行首
                    // 固定位，热账尾巴随便跳不影响盯读。
                    if (Time.unscaledTime < _badgeNextRefresh) return;
                    _badgeNextRefresh = Time.unscaledTime + 0.4f;
                    float coverage = _coverageOverlay != null ? _coverageOverlay.CoveragePercent : 0f;
                    _statusBadgeText.color = new Color(0.25f, 1f, 0.45f, 1f);
                    if (!showDetailedRuntimeStatus)
                    {
                        string compactMode = CaptureModeLabel;
                        string compactFreeze = !FrozenBlockSupervisorEffective ? "停" : "开";
                        int compactPages = _meshExtractor != null
                            ? _meshExtractor.IncrementalHeraPagesCommitted
                            : 0;
                        float compactFps = 1f / Mathf.Max(0.001f, Time.smoothDeltaTime);
                        float compactQuality = _volumeIntegrator != null
                            ? _volumeIntegrator.MotionQuality
                            : 1f;
                        string compactGunGel = _volumeIntegrator != null
                            ? _volumeIntegrator.GetGunGelEvidenceShadowCompact()
                            : "无";
                        string compactPairedCapture = _depthCapture == null ? ""
                            : _depthCapture.PairedFrameCaptureActive
                                ? $" · 双采开(待{_depthCapture.PairedFrameCapturePending}丢{_depthCapture.PairedFrameCaptureDropped}/{_depthCapture.ReplayFusionCaptureDropped})"
                                : _depthCapture.PairedFrameCapturePending > 0
                                    ? $" · 双采写(待{_depthCapture.PairedFrameCapturePending}丢{_depthCapture.PairedFrameCaptureDropped}/{_depthCapture.ReplayFusionCaptureDropped})"
                                    : "";
                        _statusBadgeText.text =
                            $"● 采集·{compactMode}  覆{coverage,3:0}%  质{compactQuality:F2}  帧{compactFps:0}\n" +
                            $"枪胶:{compactGunGel}\n" +
                            $"{(_meshExtractor != null ? _meshExtractor.RouteValidationLabel : "无显示")} " +
                            $"{(_meshExtractor != null ? _meshExtractor.SupportTruthStatsCompact : "纸无")} · " +
                            $"冻{compactFreeze}  实{(enableLiveTrack ? "开" : "停")}  页{compactPages}  融{integrationHz:0}Hz · A冻结{compactPairedCapture}";
                        return;
                    }
                    string frozenTail = _frozenBlockUnfreezeEvents > 0
                        ? $" · 冻{_frozenBlocks.Count}解{_frozenBlockUnfreezeEvents}"
                        : (_frozenBlocks.Count > 0 ? $" · 冻{_frozenBlocks.Count}" : "");
                    if (_liveTrackQueuedTotal > 0)
                        frozenTail += $"实{_liveTrackQueuedTotal}";
                    if (_holeRequeueEvents > 0)
                        frozenTail += $"补{_holeRequeueEvents}";
                    // 资格门活度：资N=因低置信占比超阈被拒冻的块次（累计）。资涨+冻涨慢=
                    // 噪声荒漠/打架块被挡在冻结门外保持活代谢（v2.2 优势回归）；折角拍稳后
                    // 置信升→资停涨+该块入冻=资格门按设计闭环。资0=没有块撞门（或门被关）。
                    if (_freezeGateRejected > 0)
                        frozenTail += $"资{_freezeGateRejected}";
                    // 年审活度：审N=已冻块被收回资格的累计块次（解N 含审N）。
                    // 审后该块紫区回落+重新入冻=降级-修复-回冻闭环成立；
                    // 审反复涨同一块=降级阈/拒冻阈滞回带太窄或该区置信永久低（=本来就不该冻）。
                    if (_freezeGateDemoted > 0)
                        frozenTail += $"审{_freezeGateDemoted}";
                    // 解冻诊断（代替 logcat）：热=热窗总数；解K=解冻时的最大累计解冻序号
                    // （同块反复解则爬升，多块轮流则停在 1-2）；票V/T=最近一次解冻的
                    // 本窗票数/当时阈值（票随阈一起涨=棘轮被票量追平→调棘轮倍率）。
                    if (_diagHotWindows > 0)
                        frozenTail += $"诊热{_diagHotWindows}解{_diagUnfreezeMaxThaws}票{_diagLastVotes}/{_diagLastThreshold}峰{_diagPeakVotes}@{_diagPeakThaws}";
                    // B1 影子平面拟合读数（ASCII 输出：动态字体图集+置顶材质有缺字前科）：
                    // R=残差 RMS mm（小=共识面干净） O=存量偏移 mm（拍平幅度预估）
                    // N=观测点数 AX=平面主轴。"拍4"=版本戳：无 4=构建缓存旧包。
                    // 探针：T=tick 入口计数 F=回读回调计数（T0=tick 未执行；T涨F0=回读丢失）。
                    if (enablePlaneFitShadow)
                        frozenTail += _planeHudValid
                            ? $"拍4[R{_planeResidualMm:0} O{_planeOffsetMm:+0;-#;0} N{_planePointCount / 1000f:0.0}k AX{_planeAxis}]"
                            : $"拍4[T{_planeTicks}F{_planeFitCb}{_planeHudNote}]";
                    // B2 拍平写入活度：压=累计压下体素 冻=其中冻块体素 拦=相干闸拦下
                    // （三面角抖动观测，拦涨压缩=刀2 生效）排=立即重提 延=冷却内转
                    // 延迟补提（稳态下排≈0 是常态，延才是刀3 真活度）。
                    // 压涨+解/冻比降=振荡断根生效；压0=无持续矛盾（或闸太严）。!备注=分闸/异常。
                    if (enablePlaneFlatten)
                        frozenTail += _flatHudNote.Length > 0
                            ? $"压[!{_flatHudNote}]"
                            : $"压{_flatSnapTotal}冻{_flatSnapFrozen}拦{_flatSnapCohBlocked}排{_flatSnapRepage}延{_flatSnapRepageDeferred}";
                    // M3/B3 平面先验普查读数（只读影子）：查=本拍扫描冻结块 合=平面资格合格
                    // 缺=合格面旁无数据候选体素（未来 B3 补全面积，k=千）
                    // 残=算出残差块（合格+残差拒）的均残差 mm（sd 子体素改尺后=几何真值，
                    // 真平墙应远小于门槛 20；贴近/超过=折角或真不平；全拒时也可读=诊断盲区已补）
                    // 拒=残差/浊(低置信)/少(面不足) 三拒因。判读：合多缺多=B3 值得上生产；
                    // 拒残高+残大=折角块占比大（正常）；拒残高+残小=门槛还卡着尺子噪声（再查）。
                    if (enablePlanePriorCensus)
                        frozenTail += _censusHudNote.Length > 0
                            ? $"普[!{_censusHudNote}]"
                            : $"普查{_censusHudScanned}合{_censusHudQualified}缺{_censusHudFill / 1000f:0.0}k残{_censusHudAvgResidualMm}拒{_censusHudRejEigen}/{_censusHudRejLowConf}/{_censusHudRejFew}";
                    // 冻前探头+冻块就地判决（08-20/08-21 拍板"纯引导值"）：视线落点块单块过
                    // 普查闸。预=未冻块预言"冻了合不合"（预可→冻→合涨=链路通；预可合不涨=锅在
                    // 冻结通道；预从不可=锅在数据层）；冻=已冻块就地报成色（冻合R/冻残/冻低/
                    // 冻少）=质量地图，与普查行聚合值互相印证。
                    // 08-21 防抖：探头读数挪到行首固定位（watch 串），不再挂 frozenTail 随波逐流。
                    // 父页队列拥塞读数：队=排队待提取，途=已派发待回读。
                    // 队高途高=回读延迟瓶颈；队高途低=派发被预算/节拍限流。
                    if (_meshExtractor != null && _meshExtractor.HasIncrementalHera)
                    {
                        int qd = _meshExtractor.IncrementalHeraQueueDepth;
                        int inf = _meshExtractor.IncrementalHeraCommitInFlight;
                        if (qd > 0 || inf > 0)
                            frozenTail += $"队{qd}途{inf}";
                    }
                    // 看门狗活度：调度（6s 窗回读丢失）+ 提交（10s 页回读丢失）。
                    // 频繁增长=Quest 在静默丢 GPU 回读，是"出网不灵敏"的负载信号。
                    int commitWatchdogs = _meshExtractor != null ? _meshExtractor.IncrementalHeraWatchdogResets : 0;
                    if (_supervisorWatchdogResets > 0 || commitWatchdogs > 0)
                        frozenTail += $" · 狗{_supervisorWatchdogResets}+{commitWatchdogs}";
                    // 活闸普查（上轮实时轨巡视）：排=入队 冻=定稿地盘 空=没内容 冷=节流
                    // 内=内容闸（T1a 起=脏账显示几何零变化）途=在途 速=速率顶。判读：排0空多=
                    // 视线路径可出网体素不足（深色面稀疏/真没扫到）；内多=完工墙静止（正常）；
                    // 冷/速多=节流过狠；途多=回读跟不上。
                    string liveGate = enableLiveTrack
                        ? $" · 出[排{_liveGateQueued}待{_liveDirtyCount}视{_liveVisibleDirtyCount}期{_liveOverdueCount}缝{_liveBoundaryDebtCount}途{_liveGateFlight}]"
                        : "";
                    // 置信度通道普查回读（分歧 EMA 三档+低置信相干拆分）：
                    // 高=逐帧观测一致；低=几何在打架，拆 干=签名一致(纠错嫌疑/闸放行)
                    // 噪=方向横跳(纯噪声/闸拦得住)。噪拦=v2 相干闸拦下的冻票数。
                    string confTail = "";
                    if (_volumeIntegrator != null && _volumeIntegrator.ConfidenceVoxelCount > 0)
                    {
                        float ct = _volumeIntegrator.ConfidenceVoxelCount;
                        confTail = $" · 信[高{100f * _volumeIntegrator.ConfidenceHighCount / ct:0} 中{100f * _volumeIntegrator.ConfidenceMidCount / ct:0} 低{100f * _volumeIntegrator.ConfidenceLowCount / ct:0}" +
                                   $"(干{100f * _volumeIntegrator.ConfidenceLowCoherentCount / ct:0} 噪{100f * _volumeIntegrator.ConfidenceLowNoiseCount / ct:0})]";
                        if (_volumeIntegrator.HasCarveStats && _volumeIntegrator.LastCarveStats[91] > 0)
                            confTail += $"噪拦{_volumeIntegrator.LastCarveStats[91]}";
                        // M1 折让命中读数（92 槽）：>0=折让在跑；盯熟墙时持续增长=预期。
                        if (_volumeIntegrator.HasCarveStats && _volumeIntegrator.LastCarveStats[92] > 0)
                            confTail += $"折{_volumeIntegrator.LastCarveStats[92]}";
                    }
                    string secondLine = _meshExtractor != null && _meshExtractor.HasIncrementalHera
                        ? $"精修 {_meshExtractor.IncrementalHeraPagesCommitted} 页 · 视线:{GazeBlockStatus()} · X 裁决海/BB · A 冻结{liveGate}{confTail}"
                        : "黄=待成网 绿=已可出网 · A 冻结";
                    // 计时账第三行：拍=提取实际节拍(/s) 落=入队→落地 回=派发→回读
                    // 融/提=各自 CPU 耗时(ms)。判读：拍远低于16=被融合帧挤占；
                    // 回高=GPU 回读瓶颈；落高回低=排队/派发堵；融+提>帧预算=CPU 侧顶满。
                    // 第四行：四个性能二分热键（右摇杆上/左/下/右）+网格显示开关（右摇杆直接按下）
                    // 当前状态，防"以为切了其实没切"。显关=只藏绘制，融合/提取/精修后台照跑（帧率二分用）。
                    string filterLabel = _depthCapture == null ? "—"
                        : (_depthCapture.BilateralEnabled ? (_depthCapture.EdgeCleanEnabled ? "开" : "双")
                        : (_depthCapture.EdgeCleanEnabled ? "半" : "关"));
                    string accountabilityLabel = CaptureModeLabel;
                    string frozenLabel = !FrozenBlockSupervisorEffective ? "停"
                        : enableAdaptiveCensus ? $"开{_censusCurrentWindow:0}s"  // 自适应普查窗回显：1s=有活动 2/4s=安静期
                        : "开";
                    // 手部语义剔除读数（08-20）：剔=平台原生手部剔除（开=生效，关=未支持/未开）；
                    // 手=本拍打码球数（裸手 5/手，持柄时手追暂停只剩柄球）；
                    // 罩=上统计周期被球罩住的深度像素数（手入镜应显著>0，空手≈0）。
                    string handLabel = _depthCapture == null || !_depthCapture.HandMaskEnabled
                        ? ""
                        : $" 剔{(_depthCapture.NativeHandRemovalActive ? "开" : "关")}" +
                          $"手{_depthCapture.HandSphereCount}罩{_depthCapture.LastHandMaskedPixels}";
                    // 源头时序滤波读数：稳=重投影带内做 EMA；变=真变化/换轨全速放行。
                    // 静止盯墙稳应占主导；挥手/搬物变应立刻冲高（不吃残影）。
                    // 时关/时开 档位回显（左摇杆上 A/B 热键，录屏判读要知道当前在哪组）。
                    string temporalLabel = _depthCapture == null ? ""
                        : !_depthCapture.TemporalFilterEnabled ? " 时关"
                        : !_depthCapture.HasTemporalStats ? " 时开"
                        : $" 时稳{_depthCapture.LastTemporalStablePixels}变{_depthCapture.LastTemporalChangedPixels}";
                    string pairedCaptureLabel = _depthCapture == null ? ""
                        : _depthCapture.PairedFrameCaptureActive
                            ? $" 双采开(待{_depthCapture.PairedFrameCapturePending}丢{_depthCapture.PairedFrameCaptureDropped}/{_depthCapture.ReplayFusionCaptureDropped})"
                            : _depthCapture.PairedFrameCapturePending > 0
                                ? $" 双采写(待{_depthCapture.PairedFrameCapturePending}丢{_depthCapture.PairedFrameCaptureDropped}/{_depthCapture.ReplayFusionCaptureDropped})"
                                : "";
                    string toggleLine = $"闸[责{accountabilityLabel} 实{(enableLiveTrack ? "开" : "关")} 冻{frozenLabel} " +
                                        $"融{integrationHz:0} 滤{filterLabel} " +
                                        $"显{(_meshExtractor != null && _meshExtractor.IsAnyMeshVisible ? "开" : "关")}" +
                                        $"皮{(_meshExtractor == null || !_meshExtractor.HasCoarseSkin ? "无" : (_meshExtractor.IsCoarseSkinVisible ? "开" : "关"))}" +
                                        $" 框{(showManagementBlockWireOverlay ? "开" : "关")}" +
                                        $"{(meshDisplayStride > 1 ? $" 网{meshDisplayStride}" : "")}]{handLabel}{temporalLabel}{pairedCaptureLabel}";
                    // 行为层读数钉行首固定位（防抖）：角/质定宽 + 探头判决紧随其后，
                    // 前缀长度恒定（覆盖%定宽 3 位），热账 frozenTail 怎么跳都不动这里。
                    string watch = $" 角{(_volumeIntegrator != null ? _volumeIntegrator.SmoothedAngularSpeed : 0f),3:F0}°/s" +
                                   $" 质{(_volumeIntegrator != null ? _volumeIntegrator.MotionQuality : 1f):F2}" +
                                   (_probeHud.Length > 0 ? $" {_probeHud}" : "");
                    // 枪胶影必须钉在采集态精简 HUD：详细 HUD 在增量 HERA 采集时不会显示，
                    // 只接 UpdateHud 会导致实机无论截图或录像都看不到校枪/K3 读数。
                    string gunGelLine = $"枪胶影:{(_volumeIntegrator != null ? _volumeIntegrator.GetGunGelEvidenceShadowCompact() : "无")}";
                    string rejectGuideLine = "Reject复核:" +
                        (ScanReplaySessionPackage.Active != null
                            ? ScanReplaySessionPackage.Active.ProbeFollowupGuidanceCompact
                            : "未记录·启用双采后显示");
                    if (_meshExtractor != null && _meshExtractor.HasIncrementalHera)
                    {
                        string geometryLine = $"{_meshExtractor.GetGeometryStabilityStatsCompact()} · " +
                                              $"64{_meshExtractor.ProductionSyncDebtCompact} " +
                                              $"32{_meshExtractor.IncrementalHeraSyncDebtCompact} · " +
                                              $"视{(geometryTruthView ? "白" : "色")}";
                        string timingLine = $"帧{1f / Mathf.Max(0.001f, Time.smoothDeltaTime):0} 拍{_heraTicksPerSec:0}/s 落{_meshExtractor.IncrementalHeraAvgQueueToCommitMs:0}ms " +
                                            $"回{_meshExtractor.IncrementalHeraAvgDispatchToCallbackMs:0}ms " +
                                            $"融{_emaIntegrateMs:0.0}提{_emaHeraTickMs:0.0}ms " +
                                            $"温{OVRPlugin.batteryTemperature:0}°";
                        _statusBadgeText.text = $"● 采集中·胶0822·{CaptureModeLabel} · 覆盖{coverage,3:0}%{watch}{frozenTail}\n{secondLine}\n{geometryLine}\n{timingLine}\n{toggleLine}\n{gunGelLine}\n{rejectGuideLine}";
                    }
                    else
                        _statusBadgeText.text = $"● 采集中·胶0822·{CaptureModeLabel} · 覆盖{coverage,3:0}%{watch}{frozenTail}\n{secondLine}\n{toggleLine}\n{gunGelLine}\n{rejectGuideLine}";
                    return;
                }

                _statusBadgeText.color = new Color(0.75f, 0.8f, 0.85f, 1f);
                _statusBadgeText.text = HasStarted
                    ? $"Ⅱ 采集已暂停·{CaptureModeLabel} · A冻结 / 扳机继续"
                    : $"○ 待采集·{CaptureModeLabel} · 扳机开始\n融合路线已锁定：裁冻";
                return;
            }

            bool meshVisible = _meshExtractor != null && _meshExtractor.IsProductionMeshVisible;
            if (IsSaveAndClearInProgress)
            {
                _statusBadgeText.color = new Color(1f, 0.78f, 0.2f, 1f);
                _statusBadgeText.text = "… 正在保存，请稍候";
            }
            else if (IsScanning)
            {
                _statusBadgeText.color = meshVisible
                    ? new Color(0.25f, 1f, 0.45f, 1f)
                    : new Color(0.2f, 0.9f, 1f, 1f);
                _statusBadgeText.text = meshVisible
                    ? "● 扫描中 · 网格显示"
                    : "● 后台扫描中 · 网格隐藏";
            }
            else if (HasStarted)
            {
                _statusBadgeText.color = new Color(1f, 0.78f, 0.2f, 1f);
                _statusBadgeText.text = "Ⅱ 已暂停";
            }
            else
            {
                _statusBadgeText.color = new Color(0.75f, 0.8f, 0.85f, 1f);
                _statusBadgeText.text = "○ 待启动";
            }
        }

        private void RefreshFrozenHeraStatusBadge()
        {
            if (_statusBadgeText == null) return;

            if (_meshExtractor == null || !_meshExtractor.HasFrozenHeraReplay)
            {
                _statusBadgeText.color = new Color(0.75f, 0.8f, 0.85f, 1f);
                _statusBadgeText.text =
                    "HERA 冻结回放｜当前没有派生账\n" +
                    "冻结 TSDF 仍保留；请勿把这一状态当成完整结果。";
                return;
            }

            int parentBuilt = _meshExtractor.FrozenHeraParentBuilt;
            int parentTotal = _meshExtractor.FrozenHeraParentTotal;
            int childBuilt = _meshExtractor.FrozenHeraChildBuilt;
            int childQueued = _meshExtractor.FrozenHeraChildQueued;
            int childPending = _meshExtractor.FrozenHeraChildrenPending;
            int familyFinalized = _meshExtractor.FrozenHeraFamiliesFinalized;
            int familyQueued = _meshExtractor.FrozenHeraFamiliesQueued;
            int familyPending = _meshExtractor.FrozenHeraFamiliesPending;
            int familySwapped = _meshExtractor.FrozenHeraFamiliesSwapped;
            int familyBlocked = _meshExtractor.FrozenHeraFamiliesBlocked;

            string signature = $"{parentBuilt}:{parentTotal}:{childBuilt}:{childQueued}:{familyFinalized}:{familyQueued}";
            float now = Time.realtimeSinceStartup;
            if (!string.Equals(signature, _heraLastProgressSignature, StringComparison.Ordinal))
            {
                _heraLastProgressSignature = signature;
                _heraLastProgressAt = now;
            }

            float elapsed = _heraFreezeStartedAt >= 0f ? Mathf.Max(0f, now - _heraFreezeStartedAt) : 0f;
            float idle = _heraLastProgressAt >= 0f ? Mathf.Max(0f, now - _heraLastProgressAt) : 0f;
            bool failed = _meshExtractor.FrozenHeraReplayFailed;
            bool complete = _meshExtractor.FrozenHeraReplayComplete;

            string headline;
            if (failed)
            {
                headline = "HERA 回放失败｜B 不可导出";
                _statusBadgeText.color = new Color(1f, 0.32f, 0.28f, 1f);
            }
            else if (complete)
            {
                headline = "HERA 完整账已闭环｜B 可导出";
                _statusBadgeText.color = new Color(0.25f, 1f, 0.45f, 1f);
            }
            else
            {
                headline = "HERA 后台处理中｜B 暂不可用";
                _statusBadgeText.color = new Color(0.25f, 0.95f, 1f, 1f);
            }

            string failureLine = failed
                ? $"\n失败原因：{_meshExtractor.FrozenHeraReplayFailureReason}"
                : "";
            if (!showDetailedRuntimeStatus)
            {
                _statusBadgeText.text =
                    $"{CaptureModeLabel} · {headline}\n" +
                    $"页 父{parentBuilt}/{parentTotal} 子{childBuilt}/{childQueued} 裁{familyFinalized}/{familyQueued}\n" +
                    $"耗时{elapsed:0.0}s · {(complete ? "B可导出" : "等待完成")}" + failureLine;
                return;
            }
            _statusBadgeText.text =
                $"{headline}\n" +
                $"32³ 父页　{parentBuilt}/{parentTotal}\n" +
                $"16³ 子页　{childBuilt}/{childQueued}　待处理 {childPending}\n" +
                $"家族裁决　{familyFinalized}/{familyQueued}　待裁决 {familyPending}\n" +
                $"结果　替换 {familySwapped}｜保留 {familyBlocked}\n" +
                $"红网格归因　{_meshExtractor.FrozenHeraRedCauseStats}\n" +
                $"耗时 {elapsed:0.0}s｜距最近进展 {idle:0.0}s｜右摇杆 {(_chunkAbDiagnosticColoring ? "状态色" : "单色")}" +
                failureLine;
        }

        /// <summary>
        /// OVRCameraRig 运行时自动创建的中眼相机默认是 Skybox 清除 + 不透明背景，
        /// 会挡住底层透视画面。等 Camera.main 出现后改成纯色透明黑清除。
        /// </summary>
        private System.Collections.IEnumerator ConfigureCameraForPassthrough()
        {
            while (Camera.main == null)
                yield return null;

            var cam = Camera.main;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0f, 0f, 0f, 0f);
            cam.nearClipPlane = 0.1f;
            Logger.Info($"相机已配置为透视底层模式: {cam.gameObject.name}");
        }

        // ─────────────────────────────────────────────────────────────
        //  头显内中文状态面板（世界空间 UI，不依赖 adb）
        // ─────────────────────────────────────────────────────────────

        [Header("调试面板")]
        [SerializeField, Tooltip("显示中央最小操作状态牌。当前性能 A/B 包在运行时固定开启；只影响显示，不影响扫描、融合、提取或按键。")]
        private bool showOperatorHud = true;
        [SerializeField] private bool showDebugHud = false;
        [SerializeField, Tooltip("在面板右上角开一个当前深度实时预览小窗（青=近 绿=中 红=远 暗=无效）。" +
            "用途：盯着幽灵网格时看深度画面里那个斑块还在不在——在=深度自洽幻觉（Meta侧时序锁定）；转头后斑块从预览消失=深度刷新")]
        private bool showDepthPreview = true;
        [SerializeField, Tooltip("显示只读断崖样本框：左侧近面、中间边缘、右侧远景。只影响诊断计数与导出。")]
        private bool showDiagnosticRoiFrame = false;

        private UnityEngine.UI.Text _hudText;
        private RectTransform _hudRect;
        private GameObject _diagnosticRoiFrame;
        private string _hudStatus = "就绪";
        private string _hudLastError = "";
        private string _hudLastInput = "无";
        private float _hudRefreshTimer;
        private float _badgeNextRefresh; // 徽标 HUD 防抖节流（扫描中 0.4s），08-21 拍板"让它们老实点"

        /// <summary>输入处理器回显：最近一次识别到的按键（用于区分"输入没到达"与"启动失败"）。</summary>
        public void NotifyInput(string what)
        {
            _hudLastInput = $"{what} ({Time.time:F0}s)";
        }

        private System.Collections.IEnumerator CreateHudWhenCameraReady()
        {
            while (Camera.main == null)
                yield return null;

            Font font = null;
            try { font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } catch { }
            if (font == null)
            {
                try { font = Resources.GetBuiltinResource<Font>("Arial.ttf"); } catch { }
            }

            var canvasGo = new GameObject("[QRS] 状态面板");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            _hudRect = canvasGo.GetComponent<RectTransform>();
            // Keep approximately the same physical width as the old panel, but add
            // vertical room for the diagnostic ledgers so wrapped rows remain visible.
            _hudRect.sizeDelta = new Vector2(1080f, 650f);
            canvasGo.transform.localScale = Vector3.one * 0.00072f;

            // 置顶材质：WorldSpace Canvas 走 UI/Default 时 ZTest=LEqual 会被网格/墙面挡住，
            // 换 QRS/HUDAlwaysOnTop（ZTest Always + Overlay 队列）。Resources 加载防裁剪，找不到静默回退。
            Material hudMat = null;
            Material hudBackgroundMat = null;
            var hudShader = Resources.Load<Shader>("HUDAlwaysOnTop");
            if (hudShader != null)
            {
                hudBackgroundMat = new Material(hudShader) { renderQueue = 4990 };
                hudMat = new Material(hudShader) { renderQueue = 4991 };
            }

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(canvasGo.transform, false);
            _hudText = textGo.AddComponent<UnityEngine.UI.Text>();
            _hudText.font = font;
            _hudText.fontSize = 32;
            _hudText.lineSpacing = 0.88f;
            _hudText.color = Color.white;
            _hudText.alignment = TextAnchor.UpperLeft;
            if (hudMat != null) _hudText.material = hudMat;
            // Wrap：开预览时文字区收窄到左侧，长行折行而不是溢到小窗底下
            _hudText.horizontalOverflow = HorizontalWrapMode.Wrap;
            _hudText.verticalOverflow = VerticalWrapMode.Overflow;
            var rt = textGo.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(18f, 12f);
            // 开深度预览时右侧留出 320px 给小窗，文字不压图
            rt.offsetMax = new Vector2(showDepthPreview ? -258f : -18f, -12f);

            // Opaque high-contrast plate; render after all room wireframes.
            var bgGo = new GameObject("Bg");
            bgGo.transform.SetParent(canvasGo.transform, false);
            bgGo.transform.SetAsFirstSibling();
            var img = bgGo.AddComponent<UnityEngine.UI.Image>();
            img.color = new Color(0.008f, 0.012f, 0.018f, 1f);
            if (hudBackgroundMat != null) img.material = hudBackgroundMat;
            var brt = bgGo.GetComponent<RectTransform>();
            brt.anchorMin = Vector2.zero;
            brt.anchorMax = Vector2.one;
            brt.offsetMin = Vector2.zero;
            brt.offsetMax = Vector2.zero;

            if (showDepthPreview) CreateDepthPreview(canvasGo.transform);
            if (showDiagnosticRoiFrame && _meshExtractor != null && _meshExtractor.DiagnosticRoiEnabled)
                CreateDiagnosticRoiFrame(Camera.main, font, hudMat);

            Logger.Info("调试面板已创建");
        }

        /// <summary>
        /// 深度实时预览小窗：RawImage 挂 QRS/DepthPreview 材质，
        /// shader 直接采样全局 gsDepthTex，零 C# 侧每帧拷贝，零额外接线。
        /// shader 在 Resources 下（防打包裁剪），找不到就静默跳过不影响面板。
        /// </summary>
        /// <summary>
        /// 只读断崖样本框。框与计算着色器使用同一组归一化范围：
        /// 左栏放近面，中间窄栏压住断崖边，右栏放远景。
        /// 该 Canvas 不参与射线、融合、提取或准入，只给操作者对准诊断 ROI。
        /// </summary>
        private void CreateDiagnosticRoiFrame(Camera cam, Font font, Material hudMat)
        {
            if (cam == null || _meshExtractor == null) return;

            var go = new GameObject("[QRS] 断崖诊断框");
            _diagnosticRoiFrame = go;
            go.transform.SetParent(cam.transform, false);

            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 32760;

            const float distance = 1f;
            const float canvasWidthPx = 1000f;
            float viewHeight = 2f * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * distance;
            float viewWidth = viewHeight * Mathf.Max(0.1f, cam.aspect);
            float canvasHeightPx = canvasWidthPx * viewHeight / Mathf.Max(0.001f, viewWidth);

            var root = go.GetComponent<RectTransform>();
            root.sizeDelta = new Vector2(canvasWidthPx, canvasHeightPx);
            go.transform.localPosition = Vector3.forward * distance;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one * (viewWidth / canvasWidthPx);

            Vector4 r = _meshExtractor.DiagnosticRoiRect;
            float x0 = Mathf.Clamp01(Mathf.Min(r.x, r.z));
            float y0 = Mathf.Clamp01(Mathf.Min(r.y, r.w));
            float x1 = Mathf.Clamp01(Mathf.Max(r.x, r.z));
            float y1 = Mathf.Clamp01(Mathf.Max(r.y, r.w));
            Vector2 splits = _meshExtractor.DiagnosticRoiSplitX;
            float sx0 = Mathf.Clamp(splits.x, x0, x1);
            float sx1 = Mathf.Clamp(splits.y, sx0, x1);

            Color outer = new Color(1f, 1f, 1f, 0.85f);
            Color divider = new Color(1f, 0.82f, 0.18f, 0.9f);
            const float linePx = 3f;

            AddDiagnosticRoiLine(go.transform, "左边", new Vector2(x0, y0), new Vector2(x0, y1),
                new Vector2(linePx, 0f), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "右边", new Vector2(x1, y0), new Vector2(x1, y1),
                new Vector2(linePx, 0f), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "下边", new Vector2(x0, y0), new Vector2(x1, y0),
                new Vector2(0f, linePx), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "上边", new Vector2(x0, y1), new Vector2(x1, y1),
                new Vector2(0f, linePx), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "近面边界", new Vector2(sx0, y0), new Vector2(sx0, y1),
                new Vector2(linePx, 0f), divider, hudMat);
            AddDiagnosticRoiLine(go.transform, "远景边界", new Vector2(sx1, y0), new Vector2(sx1, y1),
                new Vector2(linePx, 0f), divider, hudMat);

            AddDiagnosticRoiLabel(go.transform, "近面", (x0 + sx0) * 0.5f, y1, font, hudMat);
            AddDiagnosticRoiLabel(go.transform, "边缘", (sx0 + sx1) * 0.5f, y1, font, hudMat);
            AddDiagnosticRoiLabel(go.transform, "远景", (sx1 + x1) * 0.5f, y1, font, hudMat);
        }

        private static void AddDiagnosticRoiLine(
            Transform parent, string name, Vector2 anchorMin, Vector2 anchorMax,
            Vector2 sizeDelta, Color color, Material material)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<UnityEngine.UI.Image>();
            image.color = color;
            image.raycastTarget = false;
            if (material != null) image.material = material;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.anchoredPosition = Vector2.zero;
            rt.sizeDelta = sizeDelta;
        }

        private static void AddDiagnosticRoiLabel(
            Transform parent, string label, float x, float y, Font font, Material material)
        {
            var go = new GameObject(label);
            go.transform.SetParent(parent, false);
            var text = go.AddComponent<UnityEngine.UI.Text>();
            text.text = label;
            text.font = font;
            text.fontSize = 28;
            text.fontStyle = FontStyle.Bold;
            text.alignment = TextAnchor.UpperCenter;
            text.color = new Color(1f, 0.9f, 0.35f, 0.95f);
            text.raycastTarget = false;
            if (material != null) text.material = material;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(x, y);
            rt.anchorMax = new Vector2(x, y);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = new Vector2(0f, -8f);
            rt.sizeDelta = new Vector2(130f, 42f);
        }

        private void CreateDepthPreview(Transform parent)
        {
            var shader = Resources.Load<Shader>("DepthPreview");
            if (shader == null)
            {
                Logger.Warning("深度预览 shader 未找到（Resources/DepthPreview），小窗跳过");
                return;
            }

            var go = new GameObject("DepthPreview");
            go.transform.SetParent(parent, false);
            var raw = go.AddComponent<UnityEngine.UI.RawImage>();
            raw.texture = Texture2D.whiteTexture; // 防空纹理剔除，shader 实际采样全局 gsDepthTex
            raw.material = new Material(shader);
            raw.raycastTarget = false;
            var prt = go.GetComponent<RectTransform>();
            // 右上角贴边，文字区已收窄折行，互不遮挡
            prt.anchorMin = new Vector2(1f, 1f);
            prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.anchoredPosition = new Vector2(-12f, -12f);
            prt.sizeDelta = new Vector2(228f, 128f);
        }

        private void UpdateHud()
        {
            if (_hudText == null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                var ct = cam.transform;
                Vector3 targetPos = ct.position + ct.forward * 1.1f + Vector3.down * 0.18f;
                _hudRect.position = Vector3.Lerp(_hudRect.position, targetPos, Time.deltaTime * 4f);
                _hudRect.rotation = Quaternion.LookRotation(
                    _hudRect.position - ct.position, Vector3.up);
            }

            _hudRefreshTimer += Time.deltaTime;
            if (_hudRefreshTimer < 0.25f) return;
            _hudRefreshTimer = 0f;

            int verts = _meshExtractor != null ? _meshExtractor.LastVertexCount : 0;
            int tris = _meshExtractor != null ? _meshExtractor.LastIndexCount / 3 : 0;
            int integrated = _volumeIntegrator != null ? _volumeIntegrator.IntegrationCount : 0;
            bool camPlaying = _cameraProvider != null && _cameraProvider.IsPlaying;

            string edgeStat = "—";
            if (_depthCapture != null && _depthCapture.HasEdgeCleanStats)
            {
                uint ec = _depthCapture.LastEdgeCleanCount;
                uint gz = _depthCapture.LastGrazingKillCount;
                uint gs = _depthCapture.LastGrazingPlaneSupportedCount;
                uint gr = _depthCapture.LastGrazingPlaneRescuedCount;
                edgeStat = (ec >= 10000u ? (ec / 10000f).ToString("0.0") + "万" : ec.ToString()) +
                           " 掠杀:" + (gz >= 10000u ? (gz / 10000f).ToString("0.0") + "万" : gz.ToString()) +
                           " 平证:" + (gs >= 10000u ? (gs / 10000f).ToString("0.0") + "万" : gs.ToString()) +
                           " 救:" + (gr >= 10000u ? (gr / 10000f).ToString("0.0") + "万" : gr.ToString());
            }

            _hudText.text =
                $"【QRS独立链·炮0821】{_hudStatus}\n" +
                $"深度:{(DepthCapture.DepthAvailable ? "可用" : "无")}#{(_depthCapture != null ? _depthCapture.FrameCount : 0)}  相机:{(camPlaying ? "运行" : "未运行")}  融合:{integrated}帧" +
                $" 角速:{(_volumeIntegrator != null ? _volumeIntegrator.SmoothedAngularSpeed : 0f):F0}°/s" +
                $" 动质:{(_volumeIntegrator != null ? _volumeIntegrator.MotionQuality : 1f):F2}\n" +
                $"顶点:{verts}  三角:{tris}\n" +
                $"矛盾票:{(_volumeIntegrator != null ? _volumeIntegrator.GetCarveStatsCompact() : "无")}" +
                $" 缘:{edgeStat}\n" +
                $"供料账:{(_volumeIntegrator != null ? _volumeIntegrator.GetSupplyLedgerCompact() : "无")}\n" +
                $"断层影:{(_volumeIntegrator != null ? _volumeIntegrator.GetAdaptiveGapShadowCompact() : "无")}\n" +
                $"枪胶影:{(_volumeIntegrator != null ? _volumeIntegrator.GetGunGelEvidenceShadowCompact() : "无")}\n" +
                $"Reject复核:{(ScanReplaySessionPackage.Active != null ? ScanReplaySessionPackage.Active.ProbeFollowupGuidanceCompact : "未记录·启用双采后显示")}\n" +
                $"边缘源:{(_volumeIntegrator != null ? _volumeIntegrator.GetEdgeSourceLedgerCompact() : "无")}\n" +
                $"供体证:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationDonorLedgerCompact() : "无")}\n" +
                $"路径证:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationPathLedgerCompact() : "无")}\n" +
                $"接力账:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationRelayLedgerCompact() : "无")}\n" +
                $"供料闸:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationProductionGateCompact() : "无")}\n" +
                $"暂存账:{(_volumeIntegrator != null ? _volumeIntegrator.GetProvisionalLifecycleCompact() : "无")}\n" +
                $"准入来源:{(_meshExtractor != null ? _meshExtractor.GetAdmissionSourceStatsCompact() : "无")}\n" +
                $"真实确认:{(_meshExtractor != null ? _meshExtractor.GetRealConfirmationStatsCompact() : "无")}\n" +
                $"双轨视图:{(_meshExtractor != null ? _meshExtractor.GetJointDiagnosticStatsCompact() : "无")}\n" +
                $"累计账:{(_meshExtractor != null ? _meshExtractor.GetLedgerSessionStatsCompact() : "无")}\n" +
                $"局部替换:{(_meshExtractor != null ? _meshExtractor.GetLocalReplacementStatsCompact() : "无")}\n" +
                $"框内三栏:{(_meshExtractor != null ? _meshExtractor.GetDiagnosticRoiStatsCompact() : "无")}\n" +
                $"提取:生产  严格:{(_meshExtractor != null ? _meshExtractor.GetStrictObservedStatsCompact() : "无")}\n" +
                $"最近按键:{_hudLastInput}\n" +
                BuildInputDiagLine() +
                $"显示:{(wireframeMode ? "线框" : "实体")}  扳机=开始/继续  A=暂停  B=保存并清空\n" +
                $"摇杆按=线框 左摇杆左=裁冻锁定  Y=生产A/候选B" +
                (_hudLastError.Length > 0 ? $"\n<color=#FF6060>错误:{_hudLastError}</color>" : "");
        }

        /// <summary>
        /// 输入链路体检行：OVRManager 是否存活、系统认为接着什么控制器、扳机模拟量实时值。
        /// OVRInput 只有在 OVRManager 存活时才会每帧更新，GetDown/Get 才有意义。
        /// </summary>
        private static string BuildInputDiagLine()
        {
            bool mgrAlive = OVRManager.instance != null;
            var connected = OVRInput.GetConnectedControllers();
            float trig = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
            bool touchTracked = OVRInput.GetControllerPositionTracked(OVRInput.Controller.RTouch);
            bool btnA = OVRInput.Get(OVRInput.Button.One, OVRInput.Controller.RTouch);
            bool btnB = OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.RTouch);
            return $"输入:管理器{(mgrAlive ? "存活" : "缺失")} 连接:{connected} 右手追踪:{(touchTracked ? "有" : "无")} 扳机量:{trig:F2} A:{(btnA ? "1" : "0")} B:{(btnB ? "1" : "0")}\n";
        }

        private void OnLogMessage(string condition, string stackTrace, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error)
                _hudLastError = condition.Length > 80 ? condition.Substring(0, 80) : condition;
        }

        private void OnDestroy()
        {
            Application.logMessageReceived -= OnLogMessage;
            if (_volumeIntegrator != null)
                _volumeIntegrator.Cleared -= ResetFrozenBlockSupervisor;
            if (_diagnosticRoiFrame != null)
                Destroy(_diagnosticRoiFrame);
            if (_probeReticleRoot != null)
                Destroy(_probeReticleRoot);
            _planeFitStats?.Release();
            _planeFitStats = null;
            _planeFlatStats?.Release();
            _planeFlatStats = null;
            _planeFlatBlocks?.Release();
            _planeFlatBlocks = null;
            _planeCensusStats?.Release();
            _planeCensusStats = null;
            _planeCensusBlocks?.Release();
            _planeCensusBlocks = null;
            _planeProbeStats?.Release();
            _planeProbeStats = null;
            _planeProbeBlocks?.Release();
            _planeProbeBlocks = null;
            Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);
            _paperCorrectionKeyHash?.Release();
            _paperCorrectionKeyHash = null;
            _paperCorrectionTargetHash?.Release();
            _paperCorrectionTargetHash = null;
            _paperCorrectionKeyEntries = null;
            _paperCorrectionTargetEntries = null;
        }

        private void OnDisable()
        {
            _coverageOverlay?.SetAcquiring(false);
            _depthPointCloudOverlay?.SetAcquiring(false);
            _gunGelCourtOverlay?.SetVisible(false);
            _instantDepthShellOverlay?.SetAcquiring(false);
            _instantDepthShellOverlay?.SetVisible(false);
            if (IsScanning) PauseScanning();
        }

        private void Update()
        {
            UpdateHud();
            if (_probeReticleRoot != null && _probeReticleRoot.activeSelf)
                _probeReticleRoot.SetActive(false);

            bool probeSessionActive = ScanReplaySessionPackage.Active != null;
            // Repeat the idempotent request so a support renderer created after
            // session start still receives the observation alpha contract.
            _meshExtractor?.SetPaperObservationTransparency(probeSessionActive);
            // The legacy Reject quarantine stays dormant; keep the hash path disabled.
            Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);

            // 主 HUD 由自己的稳定时钟刷新，不再依赖网格提取、页面提交或冻结回放
            // 恰好发生。这样 Reject 的倒计时/方位/换位资格最多约 0.25s 延迟。
            if (_statusBadgeText != null && Time.unscaledTime >= _badgeNextRefresh)
            {
                _badgeNextRefresh = Time.unscaledTime + 0.25f;
                RefreshStatusBadge();
            }

            // 点诊断态只允许当前点层上屏。异步新建的 HERA/粗皮页也会被
            // 下一帧重新压回隐藏，但其后台融合、提取和提交继续运行。
            bool pointDiagnosticVisible =
                (_depthPointCloudOverlay != null && _depthPointCloudOverlay.Visible) ||
                (_gunGelCourtOverlay != null && _gunGelCourtOverlay.Visible) ||
                (_instantDepthShellOverlay != null && _instantDepthShellOverlay.Visible &&
                 !_instantDepthShellOverlay.CompositeWithProduction);
            if (pointDiagnosticVisible)
            {
                _coverageOverlay?.SetMarkersVisible(false);
                if (_meshExtractor != null && _meshExtractor.IsAnyMeshVisible)
                    _meshExtractor.SetCurrentMeshDisplayVisible(false);
                if (_meshExtractor != null && _meshExtractor.IsCoarseSkinVisible)
                    _meshExtractor.SetCoarseSkinVisible(false);
                if (showManagementBlockWireOverlay)
                    showManagementBlockWireOverlay = false;
                _managementBlockWireOverlay?.SetVisible(false);
            }

            // 置信度可视化开关支持运行时改值（编辑器内拖勾即生效，重发全局量）。
            if (confidenceViz != _confidenceVizApplied)
            {
                _confidenceVizApplied = confidenceViz;
                ApplyDisplayMode();
            }

            // Tail validation deliberately bypasses the ordinary fusion branch.
            // Depth capture may remain alive during the short observation window,
            // but no frame is allowed to reach VolumeIntegrator.Integrate().
            if (_meshTailValidationActive)
            {
                TickMeshTailValidation();
                return;
            }

            if (enableFrozenChunkAbExperiment && _chunkAbFrozen)
            {
                float replayTime = Time.time;
                if (replayTime - _lastMeshTime >= MeshInterval)
                {
                    _lastMeshTime = replayTime;
                    if (enableHeraHierarchicalReplay)
                        _meshExtractor.TickFrozenHeraReplay();
                    else
                        _meshExtractor.TickFrozenChunkReplay();
                    MeshExtracted?.Invoke();
                    RefreshStatusBadge();
                }
                return;
            }

            if (!IsScanning || !DepthCapture.DepthAvailable) return;

            float t = Time.time;
            UpdateInfiniTamFrameBudget();
            TickFrozenBlockSupervisor(t);
            bool directTruthRoute = _meshExtractor != null &&
                                    _meshExtractor.RouteValidationPausesHera;
            if (!directTruthRoute)
            {
                TickDeferredPageRequeues(t);
                TickLiveTrack(t);
            }
            TickPlaneFitShadow(t);
            TickPlaneFlatten(t);
            TickPlanePriorCensus(t);

            // 置信度通道 v1 普查：2s 节流，异步回读后 HUD 回显高/中/低占比。
            if (t - _lastConfidenceStatsTime >= 2f)
            {
                _lastConfidenceStatsTime = t;
                _volumeIntegrator.RefreshConfidenceStats();
            }

            bool integrationDue = t - _lastIntegrationTime >= IntegrationInterval;
            bool startupFirstMeshReady = !_infiniTamStartupFirstMeshPending ||
                (_infiniTamStartupFirstFusionSubmitted &&
                 t >= _infiniTamStartupFirstMeshNotBefore);
            bool meshDue = startupFirstMeshReady &&
                           t - _lastMeshTime >= MeshInterval;

            // The A/B acquisition phase owns one shared TSDF only.  It never
            // extracts a production mesh; cheap depth-aligned tiles are the
            // sole progress visualization until A freezes the volume.
            if (enableFrozenChunkAbExperiment)
            {
                // 增量精修：提取节拍驱动逐块 HERA 排队/提交。沿用"融合与提取
                // 不堆同帧"纪律。防饿死（移植主链 meshStarved 先例）：帧率跌破
                // 融合频率后"融合每帧到期"自锁、提取永无空帧（实机：拍 0/s、
                // 队10途0、落 20s）——提取超时 3 个节拍即让融合让路一帧，
                // 扫描降频保命、网格出网不断流。刀A（同帧叠加）已炸毁退役。
                float heraInterval = enableLiveTrack ? MeshInterval * 0.5f : MeshInterval;
                // A direct truth view pauses HERA itself, not the shared
                // extraction-only clock: the uniform 10 cm foundation consumes
                // that slot and must keep following dirty TSDF chunks.
                bool heraDue = t - _lastMeshTime >= heraInterval;
                bool tickStarved = heraDue && t - _lastMeshTime >= heraInterval * 3f &&
                                   _meshExtractor.HasIncrementalHera;
                bool integrateNow = integrationDue && !tickStarved;
                if (integrateNow)
                {
                    _lastIntegrationTime = t;
                    ProvideColorFrame();
                    _depthCapture?.PreprocessLatestFrame(
                        _volumeIntegrator != null &&
                        _volumeIntegrator.InfiniTamCompactDepthOnly);
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    bool dispatched = _volumeIntegrator.Integrate();
                    sw.Stop();
                    float iMs = (float)sw.Elapsed.TotalMilliseconds;
                    _emaIntegrateMs = _emaIntegrateMs < 0f ? iMs : Mathf.Lerp(_emaIntegrateMs, iMs, 0.25f);
                    if (dispatched)
                    {
                        Integrated?.Invoke();
                        _integrateCount++;
                    }
                    RefreshStatusBadge();
                }
                if (!integrateNow && heraDue && _meshExtractor.HasIncrementalHera)
                {
                    _lastMeshTime = t;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    _meshExtractor.TickFrozenHeraReplay();
                    sw.Stop();
                    float tMs = (float)sw.Elapsed.TotalMilliseconds;
                    _emaHeraTickMs = _emaHeraTickMs < 0f ? tMs : Mathf.Lerp(_emaHeraTickMs, tMs, 0.25f);
                    // 实际节拍统计：名义 16 拍/s，被融合帧挤占后的真实出拍率。
                    _heraTicksThisWindow++;
                    if (_heraTickWindowStart < 0f) _heraTickWindowStart = t;
                    else if (t - _heraTickWindowStart >= 1f)
                    {
                        _heraTicksPerSec = _heraTicksThisWindow / (t - _heraTickWindowStart);
                        _heraTicksThisWindow = 0;
                        _heraTickWindowStart = t;
                    }
                    MeshExtracted?.Invoke();
                }
                return;
            }

            // Never stack fusion and extraction on the same frame. InfiniTAM
            // additionally admits ordinary work only after a short run of
            // healthy rendered frames. Missed frames therefore create idle
            // recovery space instead of a fusion/mesh catch-up pair on adjacent
            // frames. Both paths retain a maximum deferral, so overload reduces
            // cadence without ever turning mesh growth or fusion off.
            bool infiniTamFixedMeshSlot = _volumeIntegrator != null &&
                                          _volumeIntegrator.InfiniTamBaselineEnabled;
            bool meshOwnsFrame;
            bool integrateThisFrame;
            if (infiniTamFixedMeshSlot && enableInfiniTamFrameBudget)
            {
                float integrationAge = t - _lastIntegrationTime;
                float meshAge = t - _lastMeshTime;
                // At 10 Hz fusion and 5 Hz extraction their nominal deadlines
                // meet every 200 ms. A due mesh is admitted only inside the
                // middle band of the current fusion interval; when it reaches
                // the late band, the next fusion goes first and opens a fresh
                // middle slot. Forced work still retains the existing maximum
                // delay escape hatch, so this phase rule cannot stop growth.
                float minimumHeavySeparation =
                    infiniTamHeavyWorkMinSpacingSeconds;
                float meshPhaseMargin = Mathf.Min(
                    minimumHeavySeparation,
                    IntegrationInterval * 0.45f);
                bool meshMidWindow = meshDue && !integrationDue &&
                    integrationAge >= meshPhaseMargin &&
                    integrationAge <= IntegrationInterval - meshPhaseMargin;
                bool spacingReady = t - _lastInfiniTamHeavyWorkTime >=
                                    minimumHeavySeparation;
                bool recoveryReady = _infiniTamHealthyFrameStreak >=
                                     infiniTamHealthyFramesBeforeWork;
                bool integrationForced = integrationDue && integrationAge >=
                    IntegrationInterval * infiniTamFusionMaxDelayIntervals;
                bool meshForced = meshDue && meshAge >=
                    MeshInterval * infiniTamMeshMaxDelayIntervals;
                bool admitHeavyWork = spacingReady &&
                    (recoveryReady || integrationForced || meshForced);

                meshOwnsFrame = false;
                integrateThisFrame = false;
                if (admitHeavyWork && (integrationDue || meshDue))
                {
                    float integrationDebt = integrationDue
                        ? integrationAge / IntegrationInterval : -1f;
                    float meshDebt = meshDue ? meshAge / MeshInterval : -1f;
                    // A sole forced path owns this slot so its maximum wait is a
                    // real bound. If both have reached their bounds, normalized
                    // age remains the final fairness rule. Ordinary mesh work,
                    // however, only owns the centred phase slot; a shared or late
                    // deadline belongs to fusion first.
                    meshOwnsFrame = meshForced != integrationForced
                        ? meshForced
                        : meshForced && integrationForced
                            ? meshDebt > integrationDebt
                            : meshMidWindow;
                    integrateThisFrame = integrationDue && !meshOwnsFrame;
                }
            }
            else
            {
                meshOwnsFrame = meshDue && (infiniTamFixedMeshSlot ||
                    t - _lastMeshTime >= MeshInterval * 1.5f);
                integrateThisFrame = integrationDue && !meshOwnsFrame;
            }

            if (integrateThisFrame)
            {
                _lastIntegrationTime = t;
                MarkInfiniTamHeavyWork(t, infiniTamFixedMeshSlot);

                ProvideColorFrame();
                _depthCapture?.PreprocessLatestFrame(
                    _volumeIntegrator != null &&
                    _volumeIntegrator.InfiniTamCompactDepthOnly);
                bool dispatched = _volumeIntegrator.Integrate();
                if (dispatched)
                {
                    if (_infiniTamStartupFirstMeshPending &&
                        !_infiniTamStartupFirstFusionSubmitted)
                    {
                        // Anchor the one-shot separation to the first real TSDF
                        // fusion dispatch, not to wall time before depth became
                        // available.
                        // This guarantees that volume bring-up/fusion and the
                        // whole-volume bootstrap cannot collapse into one GPU
                        // startup window. Reuse the accepted mesh interval; no
                        // new production tuning value is introduced.
                        _infiniTamStartupFirstFusionSubmitted = true;
                        _infiniTamStartupFirstMeshNotBefore = t + MeshInterval;
                        _lastMeshTime = t;
                    }
                    Integrated?.Invoke();
                    _integrateCount++;
                }
            }

            if (meshOwnsFrame)
            {
                _lastMeshTime = t;
                MarkInfiniTamHeavyWork(t, infiniTamFixedMeshSlot);
                _meshExtractor.Extract();
                if (_infiniTamStartupFirstMeshPending)
                    _infiniTamStartupFirstMeshPending = false;
                MeshExtracted?.Invoke();
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Logger.Verbose($"扫描中: 融合次数={_integrateCount}, 深度可用={DepthCapture.DepthAvailable}");
            }
        }

        private void UpdateInfiniTamFrameBudget()
        {
            if (_volumeIntegrator == null ||
                !_volumeIntegrator.InfiniTamBaselineEnabled ||
                !enableInfiniTamFrameBudget)
                return;

            float sample = Mathf.Clamp(Time.unscaledDeltaTime, 1f / 144f, 0.05f);
            _infiniTamFrameSecondsEma = Mathf.Lerp(
                _infiniTamFrameSecondsEma, sample, 0.2f);
            float healthyFrameSeconds = 1f / Mathf.Max(1f, infiniTamRecoveryFps);
            if (sample <= healthyFrameSeconds &&
                _infiniTamFrameSecondsEma <= healthyFrameSeconds)
            {
                _infiniTamHealthyFrameStreak = Mathf.Min(
                    _infiniTamHealthyFrameStreak + 1,
                    infiniTamHealthyFramesBeforeWork);
            }
            else
            {
                _infiniTamHealthyFrameStreak = 0;
            }
        }

        private void MarkInfiniTamHeavyWork(float now, bool infiniTamActive)
        {
            if (!infiniTamActive || !enableInfiniTamFrameBudget)
                return;
            _lastInfiniTamHeavyWorkTime = now;
            _infiniTamHealthyFrameStreak = 0;
        }

        // ─────────────────────────────────────────────────────────────
        //  逐块可逆冻结调度器
        //  成熟 = 块内 surface 体素够多 且 相邻两窗数量稳定（不读脏 epoch，
        //  用"普查结果不动"直接度量"这里不再被写"）；解冻 = 穿越票双门槛
        //  （单窗票达阈记热窗，连续两热窗才解冻，防噪声抖动）。冻结/解冻 =
        //  weight 翻符号，不销毁 TSDF、不标脏、不重提网格；解冻块的修复由
        //  后续正常积分自然驱动，成熟后会被本调度器再次定住。
        // ─────────────────────────────────────────────────────────────

        private void TickFrozenBlockSupervisor(float now)
        {
            if (!FrozenBlockSupervisorEffective)
            {
                // 净室只停冻结裁决，不能停实时轨的内容普查。实时轨用
                // _meshablePrevSurface 判断 32³ 块是否已有可出网表面；旧实现把这份
                // 只读账寄生在冻结监督回读链上，净室早退后会永久断供，只剩 HUD。
                if (enableGunGelCleanRoomExperiment)
                    TickCleanRoomLiveCensus(now);
                return;
            }
            // 看门狗：Quest 上 GPU 回读会静默丢弃（回调永不到达），没有超时复位的
            // pending 标志会把冻结调度器永久锁死——实机症状=几个块区后不再冻新块、
            // 不再出新网格页。超时强解：迟到的回调至多重复一次无害的窗口处理。
            if (_frozenBlockReadbackPending)
            {
                if (now - _frozenBlockReadbackPendingSince > 6f)
                {
                    _frozenBlockReadbackPending = false;
                    _supervisorWatchdogResets++;
                    Logger.Warning("逐块冻结：票箱/普查回读超时（>6s 无回调），看门狗复位调度器");
                }
                else return;
            }
            if (_volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady) return;
            if (_frozenBlockWindowStart < 0f) { _frozenBlockWindowStart = now; return; }
            float censusWindow = enableAdaptiveCensus ? _censusCurrentWindow : frozenBlockWindowSeconds;
            if (now - _frozenBlockWindowStart < censusWindow) return;
            _frozenBlockWindowStart = now;
            _frozenBlockReadbackPending = true;
            _frozenBlockReadbackPendingSince = now;
            AsyncGPUReadback.Request(_volumeIntegrator.FrozenChunkVotes, OnFrozenVotesReadback);
        }

        /// <summary>
        /// 净室专用只读普查：只更新实时轨的块内容账，不读取票箱、不生成冻结掩码，
        /// 也不触发冻结、解冻、补洞重排或复冻。这样净室与旧冻只差冻结监督一个变量。
        /// </summary>
        private void TickCleanRoomLiveCensus(float now)
        {
            if (!enableLiveTrack) return;
            if (_volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady) return;

            if (_frozenBlockReadbackPending)
            {
                if (now - _frozenBlockReadbackPendingSince > 6f)
                {
                    _frozenBlockReadbackPending = false;
                    Logger.Warning("枪胶净室：只读内容普查回读超时（>6s），已复位");
                }
                else return;
            }

            if (_frozenBlockWindowStart < 0f)
            {
                _frozenBlockWindowStart = now;
                return;
            }
            // 内容账只决定实时页能否首发；固定 1s 上限，不能继承旧冻安静期 2/4s
            // 自适应慢窗，否则净室转头看新区时仍会出现数秒空白。
            float censusWindow = Mathf.Clamp(frozenBlockWindowSeconds, 0.25f, 1f);
            if (now - _frozenBlockWindowStart < censusWindow) return;

            _frozenBlockWindowStart = now;
            _frozenBlockReadbackPending = true;
            _frozenBlockReadbackPendingSince = now;
            _volumeIntegrator.RefreshChunkMaturity();
            AsyncGPUReadback.Request(
                _volumeIntegrator.ChunkMaturity,
                OnCleanRoomLiveCensusReadback);
        }

        private void OnCleanRoomLiveCensusReadback(AsyncGPUReadbackRequest request)
        {
            _frozenBlockReadbackPending = false;
            if (request.hasError || !IsScanning || _chunkAbFrozen ||
                !enableGunGelCleanRoomExperiment)
                return;

            try
            {
                var maturity = request.GetData<uint>();
                int blockCount = _volumeIntegrator.FrozenBlockCount;
                bool firstCensus = _maturityPrevSurface == null ||
                                   _maturityPrevSurface.Length != blockCount;
                if (firstCensus)
                {
                    _maturityPrevSurface = new int[blockCount];
                    _meshablePrevSurface = new int[blockCount];
                }

                for (int b = 0; b < blockCount; b++)
                {
                    _maturityPrevSurface[b] = (int)maturity[b * 4];
                    _meshablePrevSurface[b] = (int)maturity[b * 4 + 2];
                }

                if (firstCensus)
                    Logger.Info("枪胶净室：实时轨只读内容普查已建档（冻结位保持全空）");
            }
            catch (Exception e)
            {
                Logger.Warning($"枪胶净室：只读内容普查异常，本窗跳过（{e.Message}）");
            }
        }

        private void OnFrozenVotesReadback(AsyncGPUReadbackRequest request)
        {
            // 回读完成时扫描可能已暂停/冻结回放中：TSDF 进入只读期，掩码一律不应用。
            if (request.hasError || !IsScanning || _chunkAbFrozen)
            {
                _frozenBlockReadbackPending = false;
                return;
            }
            try
            {
                var votes = request.GetData<uint>();
                int blockCount = _volumeIntegrator.FrozenBlockCount;
                EnsureFreezeMaskArrays();
                Array.Clear(_freezeClearMask, 0, _freezeClearMask.Length);
                var hotNow = new HashSet<int>();
                var hotHoleNow = new HashSet<int>();
                for (int b = 0; b < blockCount; b++)
                {
                    if (!_frozenBlocks.Contains(b)) continue;
                    // 只计自由空间票（x）：它=存量真面被搬空，是解冻修复的唯一硬需求。
                    // 遮挡票（y）=更近的新表面——新家具落在从未冻结的空体素上、自己
                    // 就能积分成网，不需要解冻背后的墙；其大头是身体路过（30cm+ 矛盾，
                    // 任何距离杆都拦不住），计入只会制造冻-解振荡（实机：解=冻的 2 倍）。
                    uint total = votes[b * 4];
                    // 棘轮：解冻过 n 次的块票阈=基数×倍率^n——首解灵敏，
                    // 振荡块申诉成本指数升，真搬走迟早跨过（不设硬顶）。
                    int thaws = _thawCounts.TryGetValue(b, out int tc) ? tc : 0;
                    uint threshold = (uint)Mathf.Max(1f,
                        frozenBlockVoteThreshold * Mathf.Pow(frozenBlockVoteRatchet, thaws));
                    if (total >= threshold)
                    {
                        hotNow.Add(b);
                        // 双门槛：上窗也热才解冻（先查旧窗，循环后再换窗）。
                        bool unfreezing = _hotBlocksPrevWindow.Contains(b);
                        if (unfreezing)
                            _freezeClearMask[b >> 5] |= 1u << (b & 31);
                        // 解冻诊断记账（HUD 版，实机看不了 logcat）：判读——
                        // 同块反复热且"解K"爬升、票随阈涨=棘轮被票量追上（调倍率）；
                        // 多块轮流热=解K 停在 1-2，振荡摊在区块群上，性质不同另议。
                        _diagHotWindows++;
                        if (unfreezing)
                        {
                            _diagLastVotes = total;
                            _diagLastThreshold = threshold;
                            _diagLastThaws = thaws + 1;
                            if (_diagLastThaws > _diagUnfreezeMaxThaws)
                                _diagUnfreezeMaxThaws = _diagLastThaws;
                            if (total > _diagPeakVotes)
                            {
                                _diagPeakVotes = total;
                                _diagPeakThaws = _diagLastThaws;
                            }
                        }
                        Logger.Info($"解冻诊断：块 {FormatBlockCoords(new List<int> { b })} 热窗 票{total}≥阈{threshold}（已解{thaws}次{(unfreezing ? "，双窗达标→解冻" : "，首窗")}）");
                    }

                    // ── T2 补洞票（z）：冻块内空体素种子出生=早冻留下的洞在长数据。
                    // 实时轨第一闸硬跳冻块，这些数据永远等不到重提 → 过阈后把块重排
                    // 进定稿轨重提（页原子顶替），不解冻、不动积分权。独立阈值+独立
                    // 双窗+独立棘轮：重排与解冻是两种后果，账分开记。
                    uint holeVotes = votes[b * 4 + 2];
                    int requeues = _holeRequeueCounts.TryGetValue(b, out int rc) ? rc : 0;
                    uint holeThreshold = (uint)Mathf.Max(1f,
                        frozenBlockHoleVoteThreshold * Mathf.Pow(frozenBlockVoteRatchet, requeues));
                    if (holeVotes < holeThreshold) continue;
                    hotHoleNow.Add(b);
                    if (_hotHoleBlocksPrevWindow.Contains(b) &&
                        _meshExtractor != null && _meshExtractor.HasIncrementalHera)
                    {
                        QueueIncrementalPage(b);
                        _holeRequeueCounts[b] = requeues + 1;
                        _holeRequeueEvents++;
                        Logger.Info($"T2 补洞：块重排重提（补洞票 {holeVotes}≥{holeThreshold} 双窗达标，第 {requeues + 1} 次）：{FormatBlockCoords(new List<int> { b })}");
                    }
                }
                _managementHotBlocks.Clear();
                _managementHotBlocks.UnionWith(hotNow);
                _managementConfirmedHotBlocks.Clear();
                foreach (int b in hotNow)
                    if (_hotBlocksPrevWindow.Contains(b)) _managementConfirmedHotBlocks.Add(b);
                _hotBlocksPrevWindow.Clear();
                _hotBlocksPrevWindow.UnionWith(hotNow);
                _hotHoleBlocksPrevWindow.Clear();
                _hotHoleBlocksPrevWindow.UnionWith(hotHoleNow);
                RefreshManagementBlockWireOverlay();
                _lastVotesHotCount = hotNow.Count; // 自适应普查活动账（本窗有无解冻需求）
                _volumeIntegrator.ClearAllFrozenVotes();
                _volumeIntegrator.RefreshChunkMaturity();
                AsyncGPUReadback.Request(_volumeIntegrator.ChunkMaturity, OnChunkMaturityReadback);
            }
            catch (Exception e)
            {
                _frozenBlockReadbackPending = false;
                Logger.Warning($"逐块冻结：票箱回读处理异常，本窗跳过（{e.Message}）");
            }
        }

        private void OnChunkMaturityReadback(AsyncGPUReadbackRequest request)
        {
            _frozenBlockReadbackPending = false;
            if (request.hasError || !IsScanning || _chunkAbFrozen) return;
            try
            {
                var maturity = request.GetData<uint>();
                int blockCount = _volumeIntegrator.FrozenBlockCount;
                EnsureFreezeMaskArrays();
                Array.Clear(_freezeSetMask, 0, _freezeSetMask.Length);

                bool firstCensus = _maturityPrevSurface == null || _maturityPrevSurface.Length != blockCount;
                if (firstCensus)
                {
                    _maturityPrevSurface = new int[blockCount];
                    _meshablePrevSurface = new int[blockCount];
                }
                // 速冻（一窗达标即冻）下首轮普查即可冻结；要稳定判据时首轮只建档。
                if (!firstCensus || !frozenBlockRequireStability)
                {
                    for (int b = 0; b < blockCount; b++)
                    {
                        if (_frozenBlocks.Contains(b)) continue;
                        int surface = (int)maturity[b * 4];
                        int prev = _maturityPrevSurface[b];
                        if (surface < frozenBlockMinSurfaceVoxels) continue;
                        // T1b 守卫：普查窗 2s→1s 后冻结不得随之提前——首次报满
                        // 下限只记账，满热身期才许冻。08-20 起热身 1s→2s（可配）：
                        // 兼给分歧 EMA 留积累窗，堵冷启动 fail-open 洞（EMA 全 0=
                        // 高置信期资格门必然放行，赃物由此进琥珀）。
                        if (!_firstMatureTime.ContainsKey(b))
                        {
                            _firstMatureTime[b] = Time.time;
                            continue;
                        }
                        if (Time.time - _firstMatureTime[b] < frozenBlockMatureWarmupSeconds) continue;
                        // 冻结资格门（置信度消费 v2 第一刀）：低置信占比超阈=块内几何
                        // 在打架，没资格稳定 → 拒冻保持活代谢（v2.2 优势：幻影随生随杀）。
                        // 分母=z 可出网体素数，与 w 低置信账同族群同拍。置信体未写过时
                        // w=0 天然放行（fail-open）。折角被拍平压稳后置信升→自动过门，
                        // 振荡断根不依赖棘轮高度。
                        if (enableFreezeConfidenceGate)
                        {
                            int meshable = (int)maturity[b * 4 + 2];
                            int lowConf = (int)maturity[b * 4 + 3];
                            if (meshable > 0 && lowConf > freezeGateLowConfMaxFrac * meshable)
                            {
                                _freezeGateRejected++;
                                continue;
                            }
                        }
                        if (!frozenBlockRequireStability ||
                            (prev >= frozenBlockMinSurfaceVoxels &&
                             Math.Abs(surface - prev) <= frozenBlockStabilityTolerance))
                            _freezeSetMask[b >> 5] |= 1u << (b & 31);
                    }
                    // 资格年审（置信度消费 v2 第二刀，08-20 录屏 025342 拍板）：已冻块
                    // 低置信占比超降级阈→主动降级解冻。赃在琥珀里拒冻门管不到（实锤：
                    // 压冻占 91%=冻块内几何持续打架），年审放它出来恢复活代谢。
                    // 滞回防翻烙饼：降级阈 0.35 > 拒冻阈 0.2，中间带既往不咎。
                    // 降级记 _demotedThisBatch——记账环节对它跳过 thawCounts++
                    // （年审不是票解，吃棘轮误罚会把被动降级块推向复冻兜底惯犯池）。
                    if (enableFreezeConfidenceGate)
                    {
                        _demotedThisBatch.Clear();
                        for (int b = 0; b < blockCount; b++)
                        {
                            if (!_frozenBlocks.Contains(b)) continue;
                            int meshable = (int)maturity[b * 4 + 2];
                            int lowConf = (int)maturity[b * 4 + 3];
                            if (meshable > 0 && lowConf > freezeGateDemoteLowConfFrac * meshable)
                            {
                                _freezeClearMask[b >> 5] |= 1u << (b & 31);
                                _demotedThisBatch.Add(b);
                                _freezeGateDemoted++;
                            }
                        }
                    }
                }
                int changedBlocks = 0;
                for (int b = 0; b < blockCount; b++)
                {
                    int surface = (int)maturity[b * 4];
                    if (!firstCensus && Math.Abs(surface - _maturityPrevSurface[b]) > frozenBlockStabilityTolerance)
                        changedBlocks++;
                    _maturityPrevSurface[b] = surface;
                    _meshablePrevSurface[b] = (int)maturity[b * 4 + 2]; // T3：可出网档
                }
                if (firstCensus && frozenBlockRequireStability) return; // 稳定模式首轮只建档，不冻

                // 先记账再应用：ApplyChunkFreezeMasks 返回时掩码数组已被清零。
                var setBlocks = CollectMaskBlocks(_freezeSetMask, blockCount);
                var clearBlocks = CollectMaskBlocks(_freezeClearMask, blockCount);
                // 自适应普查：冻/解/生长（成熟度变化）/穿越票任一活动→打回快窗；
                // 连续安静窗→1→2→4s 阶梯放慢。必须放在"全空早退"之前——安静正是要记账的情形。
                if (enableAdaptiveCensus)
                {
                    bool active = setBlocks.Count > 0 || clearBlocks.Count > 0 ||
                                  _lastVotesHotCount > 0 || changedBlocks > 0;
                    _censusQuietStreak = active ? 0 : _censusQuietStreak + 1;
                    _censusCurrentWindow = _censusQuietStreak <= 0 ? frozenBlockWindowSeconds
                        : _censusQuietStreak < 3 ? Mathf.Max(frozenBlockWindowSeconds, 2f)
                        : Mathf.Max(frozenBlockWindowSeconds, 4f);
                }
                if (setBlocks.Count == 0 && clearBlocks.Count == 0) return;

                _volumeIntegrator.ApplyChunkFreezeMasks(_freezeSetMask, _freezeClearMask);
                foreach (int b in setBlocks) _frozenBlocks.Add(b);
                int demotedCount = 0;
                foreach (int b in clearBlocks)
                {
                    _frozenBlocks.Remove(b);
                    _managementThawedBlocks.Add(b);
                    // 年审降级不吃棘轮（不是票解，是资格门收回冻结资格）。
                    if (_demotedThisBatch.Remove(b)) { demotedCount++; continue; }
                    // 棘轮记账：本块解冻次数+1，下窗起票阈×倍率。
                    _thawCounts[b] = (_thawCounts.TryGetValue(b, out int tc) ? tc : 0) + 1;
                }
                // T2：全量冻结位图同步本批裁决并下发 GPU（补洞票的块冻结态判据）。
                foreach (int b in setBlocks) _frozenChunkBitsArr[b >> 5] |= 1u << (b & 31);
                foreach (int b in clearBlocks) _frozenChunkBitsArr[b >> 5] &= ~(1u << (b & 31));
                foreach (int b in setBlocks) _managementThawedBlocks.Remove(b);
                _volumeIntegrator.SetFrozenChunkBits(_frozenChunkBitsArr);
                // 增量精修挂接：新冻块排队精修上屏；解冻块撤页（重冻后自动重建）。
                SyncIncrementalHeraBlocks(setBlocks, clearBlocks);
                RefreshManagementBlockWireOverlay();
                if (clearBlocks.Count > 0)
                {
                    _frozenBlockUnfreezeEvents += clearBlocks.Count;
                    Logger.Info($"逐块冻结：解冻 {clearBlocks.Count} 块（票解 {clearBlocks.Count - demotedCount} + 年审降级 {demotedCount}）：{FormatBlockCoords(clearBlocks)}");
                }
                if (setBlocks.Count > 0)
                    Logger.Info($"逐块冻结：新冻 {setBlocks.Count} 块（累计 {_frozenBlocks.Count}）：{FormatBlockCoords(setBlocks)}");
                RefreshStatusBadge();
            }
            catch (Exception e)
            {
                Logger.Warning($"逐块冻结：成熟度回读处理异常，本窗跳过（{e.Message}）");
            }
        }

        private static List<int> CollectMaskBlocks(uint[] mask, int blockCount)
        {
            var list = new List<int>();
            for (int b = 0; b < blockCount; b++)
                if ((mask[b >> 5] & (1u << (b & 31))) != 0u) list.Add(b);
            return list;
        }

        private string FormatBlockCoords(List<int> blocks)
        {
            var grid = _volumeIntegrator.FrozenChunkCount;
            var sb = new System.Text.StringBuilder();
            int show = Math.Min(blocks.Count, 6);
            for (int i = 0; i < show; i++)
            {
                int b = blocks[i];
                int x = b % grid.x, y = (b / grid.x) % grid.y, z = b / (grid.x * grid.y);
                if (i > 0) sb.Append(' ');
                sb.Append($"({x},{y},{z})");
            }
            if (blocks.Count > show) sb.Append($" …共{blocks.Count}");
            return sb.ToString();
        }

        private void EnsureFreezeMaskArrays()
        {
            int words = Mathf.Max(1, (_volumeIntegrator.FrozenBlockCount + 31) / 32);
            if (_freezeSetMask == null || _freezeSetMask.Length != words)
            {
                _freezeSetMask = new uint[words];
                _freezeClearMask = new uint[words];
                // T2 全量冻结位图随掩码同尺寸重建（重建=网格维度变了，冻结态本应已重置）
                _frozenChunkBitsArr = new uint[words];
            }
        }

        private void ResetFrozenBlockSupervisor()
        {
            _frozenBlocks.Clear();
            _hotBlocksPrevWindow.Clear();
            _managementHotBlocks.Clear();
            _managementConfirmedHotBlocks.Clear();
            _managementThawedBlocks.Clear();
            _managementBlockWireOverlay?.Clear();
            // T2 补洞账与位图随扫描重置清零
            _hotHoleBlocksPrevWindow.Clear();
            _holeRequeueCounts.Clear();
            _holeRequeueEvents = 0;
            if (_frozenChunkBitsArr != null)
            {
                Array.Clear(_frozenChunkBitsArr, 0, _frozenChunkBitsArr.Length);
                if (_volumeIntegrator != null && _volumeIntegrator.FrozenBlockReady)
                    _volumeIntegrator.SetFrozenChunkBits(_frozenChunkBitsArr);
            }
            _maturityPrevSurface = null;
            _frozenBlockWindowStart = -1f;
            _frozenBlockReadbackPending = false;
            _frozenBlockReadbackPendingSince = 0f;
            _frozenBlockUnfreezeEvents = 0;
            _diagHotWindows = 0;
            _diagUnfreezeMaxThaws = 0;
            _diagLastVotes = 0;
            _diagLastThreshold = 0;
            _diagLastThaws = 0;
            _diagPeakVotes = 0;
            _diagPeakThaws = 0;
            _planeInit = false;
            _planeHudValid = false;
            _planeHudNote = "";
            _planeTicks = 0;
            _planeFitCb = 0;
            _flatSnapTotal = 0;
            _flatSnapFrozen = 0;
            _flatSnapCohBlocked = 0;
            _flatSnapRepage = 0;
            _flatSnapRepageDeferred = 0;
            _flatHudNote = "";
            _planeFlatPending = false;
            _planeCensusPending = false;
            _planeCensusLastTick = -10f;
            _censusHudNote = "";
            _censusHudScanned = 0;
            _censusHudQualified = 0;
            _censusHudFill = 0;
            _censusHudAvgResidualMm = 0;
            _censusHudRejEigen = 0;
            _censusHudRejLowConf = 0;
            _censusHudRejFew = 0;
            _planeProbePending = false;
            _probeHud = "";
            _planeResidualMm = 0f;
            _planeOffsetMm = 0f;
            _planePointCount = 0;
            _supervisorWatchdogResets = 0;
            _censusQuietStreak = 0;
            _censusCurrentWindow = frozenBlockWindowSeconds;
            _lastVotesHotCount = 0;
            _lastPageQueueTime.Clear();
            _deferredPageRequeue.Clear();
            _thawCounts.Clear();
            _liveTrackLastSweep = -1f;
            _livePageQueueTime.Clear();
            _liveQueuedEpoch.Clear();
            _liveQueuedObservedEpoch.Clear();
            _liveDirtySince.Clear();
            _liveBoundaryEpochConsumed = null;
            _meshablePrevSurface = null;
            _firstMatureTime.Clear();
            _liveRateWindowStart = -1f;
            _liveRateWindowCount = 0;
            _liveTrackQueuedTotal = 0;
            _liveGateQueued = _liveGateFrozen = _liveGateEmpty = _liveGateCool = 0;
            _liveGateContent = _liveGateFlight = _liveGateRate = 0;
            _liveDirtyCount = _liveVisibleDirtyCount = _liveOverdueCount = _liveBoundaryDebtCount = 0;
            _meshExtractor?.ResetIncrementalHeraState();
        }

        // ── 增量精修桥接（冻结调度器 ↔ HERA 增量模式）──

        private void SyncIncrementalHeraBlocks(List<int> setBlocks, List<int> clearBlocks)
        {
            if (_meshExtractor == null || !_meshExtractor.HasIncrementalHera) return;
            foreach (int b in setBlocks)
            {
                // 冻结只把该页升级为“可派生16³救援”的定稿候选。首次冻结立刻
                // 排队；重复冻结按冷却合并，旧父页在此期间始终保持可见。
                if (_lastPageQueueTime.TryGetValue(b, out float last))
                {
                    if (Time.time - last >= frozenPageRequeueCooldownSeconds)
                        QueueIncrementalPage(b);
                    else if (!_deferredPageRequeue.ContainsKey(b))
                        _deferredPageRequeue[b] = last + frozenPageRequeueCooldownSeconds;
                }
                else QueueIncrementalPage(b);
            }
            foreach (int b in clearBlocks)
            {
                // 管理块削权：解冻只改变数据层保护，不再撤页。父页保持最后一次
                // 成功快照；16³救援影子收起，后续有效变化由活跃页调度重提交。
                _meshExtractor.IncrementalKeepParentBlockLive(FrozenBlockCoord3(b));
            }
        }

        /// <summary>实际排队提取一页，并记账（冷却/补提两表）。</summary>
        private void QueueIncrementalPage(int b)
        {
            // 路线验证必须是真隔离：支撑真值/三角粗皮直接消费 TSDF 时，HERA 不在
            // 后台积攒页任务。切回 HERA 后由活跃页 epoch 按原调度重新唤醒。
            if (_meshExtractor != null && _meshExtractor.RouteValidationPausesHera)
                return;

            _lastPageQueueTime[b] = Time.time;
            _deferredPageRequeue.Remove(b);
            var coordinate = FrozenBlockCoord3(b);
            if (_meshExtractor.IncrementalQueueParentBlock(coordinate))
            {
                _livePageQueueTime[b] = Time.time;
                uint[] epochs = _volumeIntegrator != null ? _volumeIntegrator.LatestActivePageEpochs : null;
                if (epochs != null && b >= 0 && b < epochs.Length)
                    _liveQueuedEpoch[b] = epochs[b];
                else if (!_liveQueuedEpoch.ContainsKey(b))
                    _liveQueuedEpoch[b] = 0u;
                uint[] observed = _volumeIntegrator != null
                    ? _volumeIntegrator.LatestActivePageObservedEpochs
                    : null;
                _liveQueuedObservedEpoch[b] = observed != null && b >= 0 && b < observed.Length
                    ? observed[b]
                    : 0u;

                // A queue request is not a page product.  Keep first-coverage
                // debt alive across empty commits and deduplicated/no-op queue
                // requests; the live scheduler will retry under its normal
                // cooldown until a parent or child-rescue front is published.
                if (_meshExtractor.IncrementalParentPageHasPublishedProduct(coordinate))
                    _liveDirtySince.Remove(b);
                else
                    MarkActivePageDirty(b, Time.time);
            }
        }

        /// <summary>冷却补提：延迟队列到点且块仍在冻结态的，补一次重提（最终一致）。</summary>
        private void TickDeferredPageRequeues(float now)
        {
            if (_deferredPageRequeue.Count == 0) return;
            if (_meshExtractor == null || !_meshExtractor.HasIncrementalHera)
            {
                _deferredPageRequeue.Clear();
                return;
            }
            List<int> due = null;
            foreach (KeyValuePair<int, float> kv in _deferredPageRequeue)
            {
                if (now >= kv.Value) (due ??= new List<int>()).Add(kv.Key);
            }
            if (due == null) return;
            foreach (int b in due)
            {
                _deferredPageRequeue.Remove(b);
                // 又解冻了的块不补提：它的下次复冻会重新评估。
                if (_frozenBlocks.Contains(b)) QueueIncrementalPage(b);
            }
        }

        private void EnsureManagementBlockWireOverlay()
        {
            if (_managementBlockWireOverlay == null)
                _managementBlockWireOverlay = GetComponent<FrozenBlockWireOverlay>() ??
                                              gameObject.AddComponent<FrozenBlockWireOverlay>();
            _managementBlockWireOverlay.SetVisible(showManagementBlockWireOverlay);
        }

        private void RefreshManagementBlockWireOverlay()
        {
            EnsureManagementBlockWireOverlay();
            if (_volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady)
            {
                _managementBlockWireOverlay.Clear();
                return;
            }
            _managementBlockWireOverlay.Rebuild(
                _volumeIntegrator.VoxelCount,
                _volumeIntegrator.VoxelSize,
                _volumeIntegrator.FrozenChunkCount,
                _frozenBlocks,
                _managementHotBlocks,
                _managementConfirmedHotBlocks,
                _managementThawedBlocks);
        }

        // ─────────────────────────────────────────────────────────────
        //  B1 影子平面拟合（只读不写 TSDF）：视线落点块的观测共识面拟合。
        //  两拍流水线：①FitAccum 全帧深度归约→回读→CPU Jacobi 特征分解得
        //  平面（λ_min=拟合优度）→EMA 多帧共识；②OffsetAccum 扫块内正式面
        //  体素到平面距离→存量偏移（拍平幅度预估）。HUD"拍"行三指标。
        //  验证假设：折角/天棚观测共识面干净（残差小）+存量面偏离（偏移大）
        //  →成立才开 B2 拍平写入；若残差大=观测面本身不干净，拍平无的放矢。
        // ─────────────────────────────────────────────────────────────

        private static readonly int PlaneFitStatsID = Shader.PropertyToID("_PlaneFitStats");
        private static readonly int FitWorldToLocalID = Shader.PropertyToID("_FitWorldToLocal");
        private static readonly int FitBlockCenterVoxID = Shader.PropertyToID("_FitBlockCenterVox");
        private static readonly int FitHalfSizeVoxID = Shader.PropertyToID("_FitHalfSizeVox");
        private static readonly int FitVoxelSizeID = Shader.PropertyToID("_FitVoxelSize");
        private static readonly int FitVoxCountXID = Shader.PropertyToID("_FitVoxCountX");
        private static readonly int FitVoxCountYID = Shader.PropertyToID("_FitVoxCountY");
        private static readonly int FitVoxCountZID = Shader.PropertyToID("_FitVoxCountZ");
        private static readonly int FitEyeID = Shader.PropertyToID("_FitEye");
        private static readonly int FitTsdfVolumeID = Shader.PropertyToID("_FitTsdfVolume");
        private static readonly int FitPlanePointLocalID = Shader.PropertyToID("_FitPlanePointLocal");
        private static readonly int FitPlaneNormalLocalID = Shader.PropertyToID("_FitPlaneNormalLocal");
        private static readonly int FitBlockMinVoxIID = Shader.PropertyToID("_FitBlockMinVoxI");
        private static readonly int FlatVolumeRWID = Shader.PropertyToID("gsVolumeRW");
        private static readonly int FlatConfidenceRWID = Shader.PropertyToID("gsConfidenceRW");
        private static readonly int FlatMinDeltaID = Shader.PropertyToID("_FlatMinDelta");
        private static readonly int FlatMinConfID = Shader.PropertyToID("_FlatMinConf");
        private static readonly int FlatCoherenceRWID = Shader.PropertyToID("gsCoherenceRW");
        private static readonly int FlatBlocksRWID = Shader.PropertyToID("_FlatBlocksRW");
        private static readonly int FlatMinCohID = Shader.PropertyToID("_FlatMinCoh");
        private static readonly int FlatChunkSizeID = Shader.PropertyToID("_FlatChunkSize");
        private static readonly int FlatChunkCountID = Shader.PropertyToID("_FlatChunkCount");
        private static readonly int PlaneCensusBlocksID = Shader.PropertyToID("_PlaneCensusBlocks");
        private static readonly int PlaneCensusStatsID = Shader.PropertyToID("_PlaneCensusStats");
        private static readonly int PlaneConfidenceID = Shader.PropertyToID("_PlaneConfidence");
        private static readonly int PlaneBlockCountID = Shader.PropertyToID("_PlaneBlockCount");
        private static readonly int PlaneChunkCountID = Shader.PropertyToID("_PlaneChunkCount");
        private static readonly int PlaneChunkSizeID = Shader.PropertyToID("_PlaneChunkSize");
        private static readonly int PlaneMinWeightID = Shader.PropertyToID("_PlaneMinWeight");
        private static readonly int PlaneLowConfMinID = Shader.PropertyToID("_PlaneLowConfMin");
        private static readonly int PlaneLowConfMaxFracID = Shader.PropertyToID("_PlaneLowConfMaxFrac");
        private static readonly int PlaneMaxResidualID = Shader.PropertyToID("_PlaneMaxResidualMeters");
        private static readonly int PlaneFillBandID = Shader.PropertyToID("_PlaneFillBandMeters");
        private static readonly int PlaneMinSurfaceID = Shader.PropertyToID("_PlaneMinSurfaceVoxels");

        // ── M3/B3 平面先验普查（只读影子）────────────────────────────────────
        // 保险丝与 B1/B2 同款：tick 体内任何异常会截断 Update 后半截=饿死融合，
        // 吞掉留名 EXC<类型>。看门狗 3s 未归=GPU 回读丢失，复位重试。
        private void TickPlanePriorCensus(float now)
        {
            if (!enablePlanePriorCensus) return;
            try { TickPlanePriorCensusInner(now); }
            catch (System.Exception ex)
            {
                string nm = "EXC" + ex.GetType().Name;
                _censusHudNote = nm.Length > 16 ? nm.Substring(0, 16) : nm;
            }
        }

        private void TickPlanePriorCensusInner(float now)
        {
            if (planeFitShadowCompute == null) { _censusHudNote = "NOPARAM"; return; }
            if (_planeCensusPending || _planeProbePending)
            {
                // 看门狗必须各自闸自己的 pending（08-21 预STUCK 冤案实锤）：
                // 探头未 dispatch 的拍（预已冻/预界外）pendingSince 是陈旧值，
                // 无闸判超时会每拍覆盖合法读数。
                if (_planeCensusPending && now - _planeCensusPendingSince > 3f)
                { _planeCensusPending = false; _censusHudNote = "STUCK"; }
                if (_planeProbePending && now - _planeProbePendingSince > 3f)
                { _planeProbePending = false; _probeHud = "预STUCK"; }
                return;
            }
            if (now - _planeCensusLastTick < planePriorCensusIntervalSec) return;
            _planeCensusLastTick = now;
            if (_volumeIntegrator == null || _volumeIntegrator.Volume == null)
            { _censusHudNote = "NORDY"; return; }
            if (_volumeIntegrator.ConfidenceVolume == null) { _censusHudNote = "NOCONF"; return; }
            if (_planeCensusStats == null)
            {
                // 内核存在闸（08-19 漏 #pragma kernel 实锤的疫苗）：空资产时
                // FindKernel 抛 NRE——先探明，留名 NOKRN 而不是炸异常。
                if (!planeFitShadowCompute.HasKernel("PlaneCensus"))
                { _censusHudNote = "NOKRN"; return; }
                _planeCensusStats = new ComputeBuffer(16, sizeof(int));
                _planeCensusBlocks = new ComputeBuffer(planePriorMaxBlocksPerTick, sizeof(int));
                _planeCensusKernel = new ComputeKernelHelper(planeFitShadowCompute, "PlaneCensus");
                _planeProbeStats = new ComputeBuffer(16, sizeof(int));
                _planeProbeBlocks = new ComputeBuffer(1, sizeof(int));
                _planeProbeKernel = new ComputeKernelHelper(planeFitShadowCompute, "PlaneCensus");
            }
            var fgrid = _volumeIntegrator.FrozenChunkCount;
            if (fgrid.x <= 0) { _censusHudNote = "NOGRID"; return; }
            var vox = _volumeIntegrator.VoxelCount;
            var cs = planeFitShadowCompute;
            cs.SetInt(FitVoxCountXID, vox.x);
            cs.SetInt(FitVoxCountYID, vox.y);
            cs.SetInt(FitVoxCountZID, vox.z);
            cs.SetFloat(FitVoxelSizeID, _volumeIntegrator.VoxelSize);
            cs.SetInts(PlaneChunkCountID, fgrid.x, fgrid.y, fgrid.z);
            cs.SetInt(PlaneChunkSizeID, Mathf.Max(1, vox.x / fgrid.x));
            cs.SetFloat(PlaneMinWeightID, planePriorMinWeight);
            cs.SetFloat(PlaneLowConfMinID, planePriorLowConfMin);
            cs.SetFloat(PlaneLowConfMaxFracID, planePriorLowConfMaxFrac);
            cs.SetFloat(PlaneMaxResidualID, planePriorMaxResidualMeters);
            cs.SetFloat(PlaneFillBandID, planePriorFillBandMeters);
            cs.SetInt(PlaneMinSurfaceID, planePriorMinSurfaceVoxels);
            // ① 冻前探头先行（哪怕冻结集还空着也要跑——引导值在首冻之前就该在线）：
            // 单块 dispatch，用的是与普查完全相同的闸/尺子，预言"此块此刻冻了合不合"。
            TickGazeProbeInner(now);
            // ② 冻结块普查：清单（线性索引），封顶 planePriorMaxBlocksPerTick——
            // 先来的先查，多出的下拍再轮（低频普查不追一拍全量）。
            if (_frozenBlocks.Count == 0) { _censusHudNote = "NOFRZ"; return; }
            _planeCensusBlockList.Clear();
            foreach (int b in _frozenBlocks)
            {
                if (_planeCensusBlockList.Count >= planePriorMaxBlocksPerTick) break;
                _planeCensusBlockList.Add(b);
            }
            cs.SetInt(PlaneBlockCountID, _planeCensusBlockList.Count);
            _planeCensusBlocks.SetData(_planeCensusBlockList);
            _planeCensusStats.SetData(PlaneFitZero);
            // 同 kernel 共享绑定：探头刚把 stats/blocks 绑到自己 buffer 上，
            // 普查 dispatch 前必须重绑回自己的（两 Set 一拍一次，代价可忽略）。
            _planeCensusKernel.Set(PlaneCensusStatsID, _planeCensusStats);
            _planeCensusKernel.Set(PlaneCensusBlocksID, _planeCensusBlocks);
            _planeCensusKernel.Set(FitTsdfVolumeID, _volumeIntegrator.Volume);
            _planeCensusKernel.Set(PlaneConfidenceID, _volumeIntegrator.ConfidenceVolume);
            _planeCensusKernel.DispatchFit(_planeCensusBlockList.Count, 1, 1);
            _planeCensusPending = true;
            _planeCensusPendingSince = now;
            _censusHudNote = "";
            AsyncGPUReadback.Request(_planeCensusStats, OnPlaneCensusReadback);
        }

        private void OnPlaneCensusReadback(AsyncGPUReadbackRequest request)
        {
            _planeCensusPending = false;
            if (request.hasError || _planeCensusStats == null)
            { _censusHudNote = "RBERR"; return; }
            var d = request.GetData<int>();
            if (d.Length < 16) { _censusHudNote = "LEN"; return; }
            _censusHudScanned = d[0];
            _censusHudQualified = d[1];
            _censusHudFill = d[4];
            // 残=所有算出残差的块（合格+残差拒）的均残差 mm——诊断"门槛 vs 实测差多少"。
            // 全拒时也能读出"差多少被拒"（08-20 量化地板案的诊断盲区补洞）。
            int evaluated = d[1] + d[6];
            _censusHudAvgResidualMm = evaluated > 0 ? d[9] / evaluated : 0;
            _censusHudRejEigen = d[6] + d[10]; // 残差拒+特征分解失败合并显示（后者罕见）
            _censusHudRejLowConf = d[7];
            _censusHudRejFew = d[8];
        }

        // ── 冻前探头 + 冻块就地判决（引导值）─────────────────────────
        // 视线落点块单块过普查闸：同 kernel 同尺子。未冻=预言"此块此刻冻了合不合"（预）；
        // 已冻=就地报该块成色与拒因（冻，08-21 拍板：死显示变质量地图）。
        // 只读 TSDF/置信度，不写任何生产状态（与普查同款只读影子纪律）。
        private void TickGazeProbeInner(float now)
        {
            if (_meshExtractor == null || !TryGetGazeBlockIndex(out int gb))
            { _probeHud = "预界外"; return; }
            _probeGazeFrozen = _frozenBlocks.Contains(gb);
            var cs = planeFitShadowCompute;
            cs.SetInt(PlaneBlockCountID, 1);
            _probeBlockArr[0] = gb;
            _planeProbeBlocks.SetData(_probeBlockArr);
            _planeProbeStats.SetData(PlaneFitZero);
            _planeProbeKernel.Set(PlaneCensusStatsID, _planeProbeStats);
            _planeProbeKernel.Set(PlaneCensusBlocksID, _planeProbeBlocks);
            _planeProbeKernel.Set(FitTsdfVolumeID, _volumeIntegrator.Volume);
            _planeProbeKernel.Set(PlaneConfidenceID, _volumeIntegrator.ConfidenceVolume);
            _planeProbeKernel.DispatchFit(1, 1, 1);
            _planeProbePending = true;
            _planeProbePendingSince = now;
            AsyncGPUReadback.Request(_planeProbeStats, OnPlaneProbeReadback);
        }

        private void OnPlaneProbeReadback(AsyncGPUReadbackRequest request)
        {
            _planeProbePending = false;
            if (request.hasError || _planeProbeStats == null) { _probeHud = "预RBERR"; return; }
            var d = request.GetData<int>();
            if (d.Length < 16) { _probeHud = "预LEN"; return; }
            if (d[0] <= 0) { _probeHud = "预空"; return; }
            // 槽位：[1]合格 [5]合格残差mm [6]残差拒 [9]算出残差块残差mm和
            //       [7]低置信拒 [8]面不足拒 [10]特征分解失败 [2]面体素 [3]低置信体素
            string p = _probeGazeFrozen ? "冻" : "预";
            int lowPct = d[2] > 0 ? (int)(100L * d[3] / d[2]) : 0;
            if (d[1] > 0)
            {
                // [4]=该块合格面旁补全带无数据体素（未来 B3 候选面积）——合格同时报缺，
                // 引导值一鱼两吃：R/低 管"能不能合"，缺管"值不值得补"。
                string verdict = _probeGazeFrozen ? "冻合" : "预可";
                _probeHud = d[4] > 0
                    ? $"{verdict}R{d[5]}低{lowPct}缺{d[4] / 1000f:0.0}k"
                    : $"{verdict}R{d[5]}低{lowPct}";
                return;
            }
            if (d[6] > 0) { _probeHud = $"{p}残{d[9]}"; return; }
            if (d[7] > 0) { _probeHud = $"{p}低{lowPct}"; return; }
            if (d[8] > 0) { _probeHud = $"{p}少"; return; }
            if (d[10] > 0) { _probeHud = $"{p}分"; return; }
            _probeHud = $"{p}?";
        }

        private void TickPlaneFitShadow(float now)
        {
            if (!enablePlaneFitShadow) return;
            _planeTicks++;
            // 空括号悬案保险丝：tick 体内任何异常都会截断 Update 后半截（融合饿死）
            // 且不留备注——吞掉并留名 EX<类型>。若实机出现 EX=悬案告破且融合回血。
            try { TickPlaneFitShadowInner(now); }
            catch (System.Exception ex)
            {
                _planeHudValid = false;
                string nm = "EX" + ex.GetType().Name;
                _planeHudNote = nm.Length > 16 ? nm.Substring(0, 16) : nm;
            }
        }

        private void TickPlaneFitShadowInner(float now)
        {
            if (!enablePlaneFitShadow) return;
            // 分闸备注：拍行空括号无法定位，每个早退点都留名（实机无 logcat）。
            if (planeFitShadowCompute == null) { _planeHudValid = false; _planeHudNote = "NOPARAM"; return; }
            // 看门狗：回读 pending 超 3s 未归=回调丢失（GPU 挂死/请求丢失），
            // 复位重试。备注全 ASCII：动态字体图集+置顶材质有缺字前科（豆腐块实锤），
            // 诊断备注不赌字体覆盖。
            if (_planeFitPending || _planeOffsetPending)
            {
                if (now - _planePendingSince > 3f)
                {
                    _planeFitPending = false;
                    _planeOffsetPending = false;
                    _planeHudValid = false; _planeHudNote = "STUCK";
                }
                return;
            }
            if (now - _planeFitLastTick < 1f) return;
            _planeFitLastTick = now;
            var cam = Camera.main;
            if (cam == null) { _planeHudValid = false; _planeHudNote = "NOCAM"; return; }
            if (_volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady)
            { _planeHudValid = false; _planeHudNote = "NORDY"; return; }
            if (_depthCapture == null || _depthCapture.DepthWidth <= 0)
            { _planeHudValid = false; _planeHudNote = "NODEPTH"; return; }
            Vector3 gaze = cam.transform.position + cam.transform.forward * GetGazeDistance();
            if (!TryWorldToFrozenBlock(gaze, out _, out Unity.Mathematics.int3 bc))
            {
                _planeHudValid = false; _planeHudNote = "OUT";
                return;
            }
            if (_planeFitStats == null)
            {
                // 内核存在闸：shader 资产导入失败（空资产）时 FindKernel 抛 NRE
                // （08-19 导入中毒实锤）——先探明，留名 NOKRN 而不是炸异常。
                if (!planeFitShadowCompute.HasKernel("FitAccum") ||
                    !planeFitShadowCompute.HasKernel("OffsetAccum"))
                { _planeHudValid = false; _planeHudNote = "NOKRN"; return; }
                _planeFitStats = new ComputeBuffer(16, sizeof(int));
                _planeFitAccumKernel = new ComputeKernelHelper(planeFitShadowCompute, "FitAccum");
                _planeOffsetKernel = new ComputeKernelHelper(planeFitShadowCompute, "OffsetAccum");
                _planeFitAccumKernel.Set(PlaneFitStatsID, _planeFitStats);
                _planeOffsetKernel.Set(PlaneFitStatsID, _planeFitStats);
            }
            var vox = _volumeIntegrator.VoxelCount;
            float vs = _volumeIntegrator.VoxelSize;
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            _planeBlockMinVox = bc * blockSize;
            var cs = planeFitShadowCompute;
            cs.SetMatrix(FitWorldToLocalID, _meshExtractor.transform.worldToLocalMatrix);
            cs.SetVector(FitBlockCenterVoxID, new Vector3(
                _planeBlockMinVox.x + blockSize * 0.5f,
                _planeBlockMinVox.y + blockSize * 0.5f,
                _planeBlockMinVox.z + blockSize * 0.5f));
            cs.SetFloat(FitHalfSizeVoxID, blockSize * 0.5f);
            cs.SetFloat(FitVoxelSizeID, vs);
            cs.SetInt(FitVoxCountXID, vox.x);
            cs.SetInt(FitVoxCountYID, vox.y);
            cs.SetInt(FitVoxCountZID, vox.z);
            cs.SetInt(FitEyeID, DepthCapture.FusionEyeIndex); // 与生产融合一致使用右眼片
            _planeFitStats.SetData(PlaneFitZero);
            _planeFitAccumKernel.DispatchFit(_depthCapture.DepthWidth, _depthCapture.DepthHeight, 1);
            _planeFitPending = true;
            _planePendingSince = now;
            AsyncGPUReadback.Request(_planeFitStats, OnPlaneFitReadback);
        }

        private void OnPlaneFitReadback(AsyncGPUReadbackRequest request)
        {
            _planeFitPending = false;
            _planeFitCb++;
            if (request.hasError || _planeFitStats == null)
            { _planeHudValid = false; _planeHudNote = "RBERR"; return; }
            var d = request.GetData<int>();
            if (d.Length < 16) { _planeHudValid = false; _planeHudNote = "LEN"; return; }
            int n = d[0];
            if (n < 800)
            {
                _planeHudValid = false; _planeHudNote = $"FEW{n}";
                return;
            }
            // 质心（体素单位，块中心为原点；量化 ×4 还原）
            double cx = d[1] / (4.0 * n), cy = d[2] / (4.0 * n), cz = d[3] / (4.0 * n);
            // 协方差 = Σpp/(n×16) − ccᵀ（体素²）
            double xx = d[4] / (16.0 * n) - cx * cx, xy = d[5] / (16.0 * n) - cx * cy, xz = d[6] / (16.0 * n) - cx * cz;
            double yy = d[7] / (16.0 * n) - cy * cy, yz = d[8] / (16.0 * n) - cy * cz, zz = d[9] / (16.0 * n) - cz * cz;
            Vector3 normal = SmallestEigenVector3x3(xx, xy, xz, yy, yz, zz, out double lambdaMin);
            float vs = _volumeIntegrator.VoxelSize;
            var vox = _volumeIntegrator.VoxelCount;
            // 平面点（体积局部，米）：块中心体素 + 质心偏移 → 体素坐标 → 局部米
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            Vector3 planeVox = new Vector3(
                _planeBlockMinVox.x + blockSize * 0.5f + (float)cx,
                _planeBlockMinVox.y + blockSize * 0.5f + (float)cy,
                _planeBlockMinVox.z + blockSize * 0.5f + (float)cz);
            Vector3 planeLocal = new Vector3(
                (planeVox.x - vox.x * 0.5f) * vs,
                (planeVox.y - vox.y * 0.5f) * vs,
                (planeVox.z - vox.z * 0.5f) * vs);
            // 法向半球归一（朝相机），防 EMA 符号翻跳。体素空间方向=局部空间方向（均匀缩放）。
            var cam = Camera.main;
            Vector3 camLocal = _meshExtractor.transform.InverseTransformPoint(cam.transform.position);
            if (Vector3.Dot(normal, camLocal - planeLocal) < 0f) normal = -normal;
            if (!_planeInit)
            {
                _planeInit = true;
                _planePointLocal = planeLocal;
                _planeNormalLocal = normal;
            }
            else
            {
                _planePointLocal = Vector3.Lerp(_planePointLocal, planeLocal, 0.35f);
                _planeNormalLocal = Vector3.Lerp(_planeNormalLocal, normal, 0.35f).normalized;
            }
            _planeResidualMm = (float)Math.Sqrt(Math.Max(lambdaMin, 0.0)) * vs * 1000f;
            _planePointCount = n;
            Vector3 an = new Vector3(Math.Abs(_planeNormalLocal.x), Math.Abs(_planeNormalLocal.y), Math.Abs(_planeNormalLocal.z));
            _planeAxis = an.x >= an.y && an.x >= an.z ? 'X' : (an.y >= an.z ? 'Y' : 'Z');
            _planeHudValid = true;
            // 第二拍：存量偏移归约
            _planeFitStats.SetData(PlaneFitZero);
            var cs = planeFitShadowCompute;
            cs.SetVector(FitPlanePointLocalID, _planePointLocal);
            cs.SetVector(FitPlaneNormalLocalID, _planeNormalLocal);
            cs.SetVector(FitBlockMinVoxIID, new Vector3(_planeBlockMinVox.x, _planeBlockMinVox.y, _planeBlockMinVox.z));
            _planeOffsetKernel.Set(FitTsdfVolumeID, _volumeIntegrator.Volume);
            _planeOffsetKernel.DispatchFit(blockSize, blockSize, blockSize);
            _planeOffsetPending = true;
            _planePendingSince = Time.time;
            AsyncGPUReadback.Request(_planeFitStats, OnPlaneOffsetReadback);
        }

        private void OnPlaneOffsetReadback(AsyncGPUReadbackRequest request)
        {
            _planeOffsetPending = false;
            if (request.hasError || _planeFitStats == null) return;
            var d = request.GetData<int>();
            if (d.Length < 13) return;
            int cnt = d[11];
            _planeOffsetMm = cnt > 0 ? (float)d[10] / cnt : 0f;
        }

        // ── B2 拍平写入（生产路径：写 TSDF sd，不动 weight）────────────────
        // 保险丝与 B1 同款：异常截断 Update 后半截=饿死融合（08-19 实锤），吞掉留名。
        private void TickPlaneFlatten(float now)
        {
            if (!enablePlaneFlatten) return;
            try { TickPlaneFlattenInner(now); }
            catch (System.Exception ex)
            {
                string nm = "EXF" + ex.GetType().Name;
                _flatHudNote = nm.Length > 16 ? nm.Substring(0, 16) : nm;
            }
        }

        private void TickPlaneFlattenInner(float now)
        {
            if (planeFitShadowCompute == null) { _flatHudNote = "NOPARAM"; return; }
            // 看门狗：回读 3s 未归=请求丢失，复位重试
            if (_planeFlatPending)
            {
                if (now - _planeFlatPendingSince > 3f)
                { _planeFlatPending = false; _flatHudNote = "STUCK"; }
                return;
            }
            if (now - _planeFlattenLastTick < planeFlattenIntervalSec) return;
            _planeFlattenLastTick = now;
            if (_volumeIntegrator == null || _volumeIntegrator.Volume == null)
            { _flatHudNote = "NORDY"; return; }
            if (_volumeIntegrator.ConfidenceVolume == null ||
                _volumeIntegrator.CoherenceVolume == null) { _flatHudNote = "NOCONF"; return; }
            if (_depthCapture == null || _depthCapture.DepthWidth <= 0)
            { _flatHudNote = "NODEPTH"; return; }
            if (_planeFlatStats == null)
            {
                // 内核存在闸（08-19 漏 #pragma kernel 实锤的疫苗）
                if (!planeFitShadowCompute.HasKernel("FlattenSplat"))
                { _flatHudNote = "NOKRN"; return; }
                _planeFlatStats = new ComputeBuffer(16, sizeof(int));
                _planeFlatBlocks = new ComputeBuffer(64, sizeof(int));
                _planeFlattenKernel = new ComputeKernelHelper(planeFitShadowCompute, "FlattenSplat");
                _planeFlattenKernel.Set(PlaneFitStatsID, _planeFlatStats);
                _planeFlattenKernel.Set(FlatBlocksRWID, _planeFlatBlocks);
            }
            var vox = _volumeIntegrator.VoxelCount;
            var cs = planeFitShadowCompute;
            cs.SetMatrix(FitWorldToLocalID, _meshExtractor.transform.worldToLocalMatrix);
            cs.SetFloat(FitVoxelSizeID, _volumeIntegrator.VoxelSize);
            cs.SetInt(FitVoxCountXID, vox.x);
            cs.SetInt(FitVoxCountYID, vox.y);
            cs.SetInt(FitVoxCountZID, vox.z);
            cs.SetInt(FitEyeID, DepthCapture.FusionEyeIndex); // 与生产融合一致使用右眼片
            cs.SetFloat(FlatMinDeltaID, planeFlattenMinDelta);
            cs.SetFloat(FlatMinConfID, planeFlattenMinConf);
            cs.SetFloat(FlatMinCohID, planeFlattenMinCoherence);
            // 冻结块网格逐内核下发（gsFrozenChunkSize/Count 在 VolumeIntegration 是
            // 按 compute SetInt 的非全局量，本 shader 读不到）：0=冻结关=不上报块。
            var fgrid = _volumeIntegrator.FrozenChunkCount;
            cs.SetInt(FlatChunkSizeID, fgrid.x > 0 ? Mathf.Max(1, vox.x / fgrid.x) : 0);
            cs.SetInts(FlatChunkCountID, fgrid.x, fgrid.y, fgrid.z);
            _planeFlattenKernel.Set(FlatVolumeRWID, _volumeIntegrator.Volume);
            _planeFlattenKernel.Set(FlatConfidenceRWID, _volumeIntegrator.ConfidenceVolume);
            _planeFlattenKernel.Set(FlatCoherenceRWID, _volumeIntegrator.CoherenceVolume);
            _planeFlatStats.SetData(PlaneFitZero);
            _planeFlattenKernel.DispatchFit(_depthCapture.DepthWidth, _depthCapture.DepthHeight, 1);
            _planeFlatPending = true;
            _planeFlatPendingSince = now;
            _flatHudNote = "";
            AsyncGPUReadback.Request(_planeFlatStats, OnPlaneFlatReadback);
        }

        private void OnPlaneFlatReadback(AsyncGPUReadbackRequest request)
        {
            if (request.hasError || _planeFlatStats == null)
            { _planeFlatPending = false; _flatHudNote = "RBERR"; return; }
            var d = request.GetData<int>();
            if (d.Length < 16) { _planeFlatPending = false; _flatHudNote = "LEN"; return; }
            _flatSnapTotal += d[13];
            _flatSnapFrozen += d[14];
            _flatSnapCohBlocked += d[15];
            // 刀3 链式回读：本拍压到冻块→再取块索引清单。挂起保持到清单归还
            // （看门狗重新计 3s）；清单回调里做重排队。
            _planeFlatBlocksExpected = Mathf.Min(d[12], 64);
            if (_planeFlatBlocksExpected > 0 && _planeFlatBlocks != null)
            {
                _planeFlatPendingSince = Time.time;
                AsyncGPUReadback.Request(_planeFlatBlocks, OnPlaneFlatBlocksReadback);
                return; // _planeFlatPending 保持 true
            }
            _planeFlatPending = false;
        }

        // 刀3 重排钩：拍平改了冻块 TSDF 但直写不标脏不重排，不补这一钩页面永远
        // 显示旧几何（A 类显示陈旧=纠了看不见）。重排不解冻（补洞票同款），
        // 冷却/延迟补提与复冻共用一套账，每拍上限防灌爆提取队列。
        private void OnPlaneFlatBlocksReadback(AsyncGPUReadbackRequest request)
        {
            _planeFlatPending = false;
            if (request.hasError || _planeFlatBlocks == null) { _flatHudNote = "RBERR"; return; }
            var d = request.GetData<int>();
            int n = Mathf.Min(_planeFlatBlocksExpected, d.Length);
            _flatUniqueBlocks.Clear();
            for (int i = 0; i < n; i++) _flatUniqueBlocks.Add(d[i]);
            int repaged = 0;
            foreach (int b in _flatUniqueBlocks)
            {
                if (repaged >= planeFlattenRepageMaxPerTick) break;
                if (!_frozenBlocks.Contains(b)) continue; // 压后已解冻=正常流程接管
                if (_lastPageQueueTime.TryGetValue(b, out float last))
                {
                    if (Time.time - last >= frozenPageRequeueCooldownSeconds)
                    { QueueIncrementalPage(b); repaged++; }
                    else if (!_deferredPageRequeue.ContainsKey(b))
                    {
                        _deferredPageRequeue[b] = last + frozenPageRequeueCooldownSeconds;
                        _flatSnapRepageDeferred++; // 延迟补提也计数：稳态下排≈0 是常态，延才是真活度
                    }
                }
                else { QueueIncrementalPage(b); repaged++; }
            }
            _flatSnapRepage += repaged;
        }

        /// <summary>对称 3×3 最小特征向量（Jacobi 旋转，~12 扫收敛）。λ_min=沿法向方差=拟合优度。</summary>
        private static Vector3 SmallestEigenVector3x3(
            double xx, double xy, double xz, double yy, double yz, double zz, out double lambdaMin)
        {
            double[,] a = { { xx, xy, xz }, { xy, yy, yz }, { xz, yz, zz } };
            double[,] v = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            for (int sweep = 0; sweep < 16; sweep++)
            {
                double off = Math.Abs(a[0, 1]) + Math.Abs(a[0, 2]) + Math.Abs(a[1, 2]);
                if (off < 1e-12) break;
                for (int p = 0; p < 3; p++)
                    for (int q = p + 1; q < 3; q++)
                    {
                        double apq = a[p, q];
                        if (Math.Abs(apq) < 1e-15) continue;
                        double theta = 0.5 * Math.Atan2(2.0 * apq, a[q, q] - a[p, p]);
                        double c = Math.Cos(theta), s = Math.Sin(theta);
                        for (int k = 0; k < 3; k++)
                        {
                            double akp = a[k, p], akq = a[k, q];
                            a[k, p] = c * akp - s * akq;
                            a[k, q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++)
                        {
                            double apk = a[p, k], aqk = a[q, k];
                            a[p, k] = c * apk - s * aqk;
                            a[q, k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; k++)
                        {
                            double vkp = v[k, p], vkq = v[k, q];
                            v[k, p] = c * vkp - s * vkq;
                            v[k, q] = s * vkp + c * vkq;
                        }
                    }
            }
            int minIdx = 0;
            if (a[1, 1] < a[minIdx, minIdx]) minIdx = 1;
            if (a[2, 2] < a[minIdx, minIdx]) minIdx = 2;
            lambdaMin = a[minIdx, minIdx];
            return new Vector3((float)v[0, minIdx], (float)v[1, minIdx], (float)v[2, minIdx]).normalized;
        }


        // 32³ 活跃页调度。页面的空间地址仍沿用冻结管理网格，但出网权只看：
        // 精确页脏 epoch / 共享面债务 / 首次脏时间 / 视锥优先级。冻结仅决定本次
        // 提交是否带 16³ 定稿救援，不再决定页面能否生产或旧页是否被撤下。
        // ─────────────────────────────────────────────────────────────

        private void TickLiveTrack(float now)
        {
            if (!enableLiveTrack) return;
            if (_meshExtractor == null || !_meshExtractor.HasIncrementalHera) return;
            if (_volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady) return;
            var cam = Camera.main;
            if (cam == null) return;
            if (_liveTrackLastSweep >= 0f && now - _liveTrackLastSweep < liveTrackSweepSeconds) return;
            _liveTrackLastSweep = now;

            // 全局速率硬顶：1s 滑窗。
            if (_liveRateWindowStart < 0f || now - _liveRateWindowStart >= 1f)
            {
                _liveRateWindowStart = now;
                _liveRateWindowCount = 0;
            }
            // 中心深度只负责把精确 gaze 页钉到真实表面；全部候选仍按页体积扫描视锥，
            // 不再让单条射线拥有其它可见页的生杀权。
            _depthCapture?.RequestCenterDepthSample();
            _volumeIntegrator.RequestActivePageEpochs();
            uint[] pageEpochs = _volumeIntegrator.LatestActivePageEpochs;
            uint[] observedEpochs = _volumeIntegrator.LatestActivePageObservedEpochs;
            uint[] boundaryEpochs = _volumeIntegrator.LatestActivePageBoundaryEpochs;
            int pageCount = _volumeIntegrator.FrozenBlockCount;

            // 边界债不等冻结：某页共享面附近发生有效几何变化，双方都进入脏账。
            RefreshActiveBoundaryDebt(now, pageCount, boundaryEpochs);

            // 精确32³ owner epoch 建立首次脏时刻。后续 epoch 推进只累计，不重置。
            for (int b = 0; b < pageCount; b++)
            {
                var coordinate = FrozenBlockCoord3(b);
                uint pageEpoch = pageEpochs != null && b < pageEpochs.Length ? pageEpochs[b] : 0u;
                bool hasPublishedProduct =
                    _meshExtractor.IncrementalParentPageHasPublishedProduct(coordinate);
                bool hasSurface = _meshablePrevSurface != null && b < _meshablePrevSurface.Length &&
                                  _meshablePrevSurface[b] >= liveTrackMinSurfaceVoxels;
                uint queuedEpoch = _liveQueuedEpoch.TryGetValue(b, out uint qe) ? qe : 0u;
                uint observedEpoch = observedEpochs != null && b < observedEpochs.Length ? observedEpochs[b] : 0u;
                uint queuedObservedEpoch = _liveQueuedObservedEpoch.TryGetValue(b, out uint qoe) ? qoe : 0u;
                bool changed = pageEpoch > queuedEpoch;
                // GPU epoch 本身只在“可提取表面出生/消失/穿越/显著位移”时推进，
                // 因此它就是首次出网许可，不再等 1~4s 普查。普查只负责页账建立前
                // 已经存在的表面兜底；已建页变为空也会由 changed 触发清空旧前台。
                bool observedSinceAttempt = observedEpoch > queuedObservedEpoch;
                if (changed || (!hasPublishedProduct && (observedSinceAttempt || hasSurface)))
                    MarkActivePageDirty(b, now);
            }

            // 构造全部脏页候选：视锥中心优先，视野外仍由首次脏期限兜底。
            _liveCandidates.Clear();
            _liveVisibleDirtyCount = 0;
            _liveOverdueCount = 0;
            GeometryUtility.CalculateFrustumPlanes(cam, _liveFrustumPlanes);
            bool hasGazePage = TryGetGazeBlockIndex(out int gazePage);
            foreach (KeyValuePair<int, float> kv in _liveDirtySince)
            {
                int b = kv.Key;
                var coordinate = FrozenBlockCoord3(b);
                bool hasPublishedProduct =
                    _meshExtractor.IncrementalParentPageHasPublishedProduct(coordinate);
                bool hasSurface = _meshablePrevSurface != null && b < _meshablePrevSurface.Length &&
                                  _meshablePrevSurface[b] >= liveTrackMinSurfaceVoxels;
                uint pageEpoch = pageEpochs != null && b < pageEpochs.Length ? pageEpochs[b] : 0u;
                uint queuedEpoch = _liveQueuedEpoch.TryGetValue(b, out uint qe) ? qe : 0u;
                uint observedEpoch = observedEpochs != null && b < observedEpochs.Length ? observedEpochs[b] : 0u;
                uint queuedObservedEpoch = _liveQueuedObservedEpoch.TryGetValue(b, out uint qoe) ? qoe : 0u;
                // A fresh geometry event gets one early attempt before the
                // asynchronous surface census catches up.  After that attempt,
                // a still-empty page retries only while current meshable
                // surface evidence exists; a historical sticky epoch cannot
                // create permanent work for a surface that has disappeared.
                if (!hasSurface && !hasPublishedProduct && pageEpoch <= queuedEpoch &&
                    observedEpoch <= queuedObservedEpoch)
                    continue;

                Bounds pageBounds = FrozenBlockWorldBounds(b);
                GetPageScreenPriority(cam, pageBounds, liveTrackViewportMargin,
                    out float centerDistanceSq, out float cameraDepth, out bool overlapsPreheatViewport);
                bool visible = GeometryUtility.TestPlanesAABB(_liveFrustumPlanes, pageBounds) ||
                               overlapsPreheatViewport;
                float age = Mathf.Max(0f, now - kv.Value);
                float deadline = visible ? liveTrackVisibleDeadlineSeconds : liveTrackBackgroundDeadlineSeconds;
                bool overdue = age >= Mathf.Max(0.05f, deadline);
                _liveCandidates.Add(new LiveCandidate
                {
                    Block = b,
                    Visible = visible,
                    // The exact depth-anchored gaze page and the small central
                    // viewport neighbourhood pre-empt FIFO background debt.
                    // This is visual scheduling only; it changes no TSDF or
                    // admission decision.
                    Urgent = (hasGazePage && b == gazePage) ||
                             (visible && centerDistanceSq <= 0.04f),
                    Overdue = overdue,
                    Age = age,
                    CenterDistanceSq = centerDistanceSq,
                    CameraDepth = cameraDepth
                });
                if (visible) _liveVisibleDirtyCount++;
                if (overdue) _liveOverdueCount++;
            }
            _liveDirtyCount = _liveCandidates.Count;
            _liveCandidates.Sort(CompareLiveCandidates);

            _liveGateQueued = _liveGateFrozen = _liveGateEmpty = _liveGateCool = 0;
            _liveGateContent = _liveGateFlight = _liveGateRate = 0;

            int queued = 0;
            for (int i = 0; i < _liveCandidates.Count && queued < liveTrackMaxPagesPerSweep; i++)
            {
                LiveCandidate candidate = _liveCandidates[i];
                int b = candidate.Block;
                if (_livePageQueueTime.TryGetValue(b, out float last) &&
                    now - last < liveTrackBlockCooldownSeconds)
                { _liveGateCool++; continue; }
                var coord = FrozenBlockCoord3(b);
                // 在途时不丢脏时刻；回调完成后本页仍在字典中，下一巡视自动补跑。
                if (_meshExtractor.IncrementalParentBlockInFlight(coord))
                { _liveGateFlight++; continue; }
                if (_liveRateWindowCount >= liveTrackMaxPagesPerSecond)
                { _liveGateRate++; break; }

                // 冻结只选择“定稿+16救援”或“实时粗页”，不再拥有出网否决权。
                bool queuedOk = _frozenBlocks.Contains(b)
                    ? _meshExtractor.IncrementalQueueParentBlock(coord, candidate.Urgent)
                    : _meshExtractor.IncrementalQueueLiveParentBlock(coord, candidate.Urgent);
                if (!queuedOk) continue;
                _livePageQueueTime[b] = now;
                _liveQueuedEpoch[b] = pageEpochs != null && b < pageEpochs.Length
                    ? pageEpochs[b]
                    : 0u;
                _liveQueuedObservedEpoch[b] = observedEpochs != null && b < observedEpochs.Length
                    ? observedEpochs[b]
                    : 0u;

                bool hasPublishedProduct =
                    _meshExtractor.IncrementalParentPageHasPublishedProduct(coord);
                bool workInFlight = _meshExtractor.IncrementalParentBlockInFlight(coord);
                if (hasPublishedProduct)
                    _liveDirtySince.Remove(b);
                else
                    MarkActivePageDirty(b, now);

                // QueueStaticReplayChunk accepts identity/dedup requests too.
                // Only an actual queued/in-flight job consumes the production
                // rate budget.  A no-op without a product remains coverage debt
                // and is retried after the same block cooldown.
                if (!workInFlight)
                {
                    if (!hasPublishedProduct) _liveGateContent++;
                    continue;
                }

                _liveRateWindowCount++;
                _liveTrackQueuedTotal++;
                _liveGateQueued++;
                queued++;
            }
        }

        private void MarkActivePageDirty(int block, float now)
        {
            if (!_liveDirtySince.ContainsKey(block))
                _liveDirtySince.Add(block, now);
        }

        private void RefreshActiveBoundaryDebt(float now, int pageCount, uint[] boundaryEpochs)
        {
            _liveBoundaryDebtCount = 0;
            if (boundaryEpochs == null || boundaryEpochs.Length != pageCount * 6)
                return;
            if (_liveBoundaryEpochConsumed == null ||
                _liveBoundaryEpochConsumed.Length != boundaryEpochs.Length)
                _liveBoundaryEpochConsumed = new uint[boundaryEpochs.Length];

            var grid = _volumeIntegrator.FrozenChunkCount;
            for (int b = 0; b < pageCount; b++)
            {
                var c = FrozenBlockCoord3(b);
                for (int face = 0; face < 6; face++)
                {
                    int slot = b * 6 + face;
                    uint epoch = boundaryEpochs[slot];
                    if (epoch == 0u || epoch <= _liveBoundaryEpochConsumed[slot]) continue;
                    _liveBoundaryEpochConsumed[slot] = epoch;
                    _liveBoundaryDebtCount++;
                    MarkActivePageDirty(b, now);
                    var n = c + ActivePageFaceNeighbours[face];
                    if (n.x < 0 || n.y < 0 || n.z < 0 ||
                        n.x >= grid.x || n.y >= grid.y || n.z >= grid.z)
                        continue;
                    int neighbour = n.x + grid.x * (n.y + grid.y * n.z);
                    MarkActivePageDirty(neighbour, now);
                }
            }
        }

        private Vector3 FrozenBlockWorldCenter(int block)
        {
            var coord = FrozenBlockCoord3(block);
            var vox = _volumeIntegrator.VoxelCount;
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            Vector3 local = new Vector3(
                (coord.x * blockSize + blockSize * 0.5f - vox.x * 0.5f) * _volumeIntegrator.VoxelSize,
                (coord.y * blockSize + blockSize * 0.5f - vox.y * 0.5f) * _volumeIntegrator.VoxelSize,
                (coord.z * blockSize + blockSize * 0.5f - vox.z * 0.5f) * _volumeIntegrator.VoxelSize);
            return _meshExtractor.transform.TransformPoint(local);
        }

        private readonly Plane[] _liveFrustumPlanes = new Plane[6];
        private readonly Vector3[] _livePageCorners = new Vector3[8];

        private Bounds FrozenBlockWorldBounds(int block)
        {
            var coord = FrozenBlockCoord3(block);
            var vox = _volumeIntegrator.VoxelCount;
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            float vs = _volumeIntegrator.VoxelSize;
            Vector3 localMin = new Vector3(
                (coord.x * blockSize - vox.x * 0.5f) * vs,
                (coord.y * blockSize - vox.y * 0.5f) * vs,
                (coord.z * blockSize - vox.z * 0.5f) * vs);
            Vector3 localMax = new Vector3(
                (Mathf.Min((coord.x + 1) * blockSize, vox.x) - vox.x * 0.5f) * vs,
                (Mathf.Min((coord.y + 1) * blockSize, vox.y) - vox.y * 0.5f) * vs,
                (Mathf.Min((coord.z + 1) * blockSize, vox.z) - vox.z * 0.5f) * vs);
            Transform root = _meshExtractor.transform;
            Bounds bounds = new Bounds(root.TransformPoint(localMin), Vector3.zero);
            for (int i = 0; i < 8; i++)
            {
                Vector3 local = new Vector3(
                    (i & 1) == 0 ? localMin.x : localMax.x,
                    (i & 2) == 0 ? localMin.y : localMax.y,
                    (i & 4) == 0 ? localMin.z : localMax.z);
                Vector3 world = root.TransformPoint(local);
                _livePageCorners[i] = world;
                bounds.Encapsulate(world);
            }
            return bounds;
        }

        private void GetPageScreenPriority(Camera cam, Bounds bounds, float viewportMargin,
            out float centerDistanceSq, out float cameraDepth, out bool overlapsViewport)
        {
            if (bounds.Contains(cam.transform.position))
            {
                centerDistanceSq = 0f;
                cameraDepth = 0f;
                overlapsViewport = true;
                return;
            }

            float minX = float.PositiveInfinity, minY = float.PositiveInfinity;
            float maxX = float.NegativeInfinity, maxY = float.NegativeInfinity;
            cameraDepth = float.PositiveInfinity;
            bool anyInFront = false;
            for (int i = 0; i < 8; i++)
            {
                Vector3 viewport = cam.WorldToViewportPoint(_livePageCorners[i]);
                if (viewport.z <= 0f) continue;
                anyInFront = true;
                minX = Mathf.Min(minX, viewport.x);
                maxX = Mathf.Max(maxX, viewport.x);
                minY = Mathf.Min(minY, viewport.y);
                maxY = Mathf.Max(maxY, viewport.y);
                cameraDepth = Mathf.Min(cameraDepth, viewport.z);
            }
            if (!anyInFront)
            {
                Vector3 center = cam.WorldToViewportPoint(bounds.center);
                float dx = center.x - 0.5f;
                float dy = center.y - 0.5f;
                centerDistanceSq = dx * dx + dy * dy;
                cameraDepth = Mathf.Abs(center.z);
                overlapsViewport = false;
                return;
            }
            float margin = Mathf.Max(0f, viewportMargin);
            overlapsViewport = maxX >= -margin && minX <= 1f + margin &&
                               maxY >= -margin && minY <= 1f + margin;
            float nearestX = Mathf.Clamp(0.5f, minX, maxX);
            float nearestY = Mathf.Clamp(0.5f, minY, maxY);
            float screenDx = nearestX - 0.5f;
            float screenDy = nearestY - 0.5f;
            centerDistanceSq = screenDx * screenDx + screenDy * screenDy;
        }

        private static int CompareLiveCandidates(LiveCandidate a, LiveCandidate b)
        {
            int urgent = b.Urgent.CompareTo(a.Urgent);
            if (urgent != 0) return urgent;
            int aBand = (a.Overdue ? 2 : 0) + (a.Visible ? 1 : 0);
            int bBand = (b.Overdue ? 2 : 0) + (b.Visible ? 1 : 0);
            int band = bBand.CompareTo(aBand);
            if (band != 0) return band;
            int age = b.Age.CompareTo(a.Age);
            if (age != 0) return age;
            int center = a.CenterDistanceSq.CompareTo(b.CenterDistanceSq);
            return center != 0 ? center : a.CameraDepth.CompareTo(b.CameraDepth);
        }

        private static readonly Unity.Mathematics.int3[] ActivePageFaceNeighbours =
        {
            new Unity.Mathematics.int3(-1, 0, 0), new Unity.Mathematics.int3(1, 0, 0),
            new Unity.Mathematics.int3(0, -1, 0), new Unity.Mathematics.int3(0, 1, 0),
            new Unity.Mathematics.int3(0, 0, -1), new Unity.Mathematics.int3(0, 0, 1)
        };

        private struct LiveCandidate
        {
            public int Block;
            public bool Visible;
            public bool Urgent;
            public bool Overdue;
            public float Age;
            public float CenterDistanceSq;
            public float CameraDepth;
        }
        private readonly List<LiveCandidate> _liveCandidates = new List<LiveCandidate>(144);

        /// <summary>
        /// 实时轨视线落点距离（米）：深度锚定优先——中心 8×8 深度中位数 +0.15m
        /// （落点没入墙面块内部，确保 slab 盖住墙块）；无有效样本时退回 1.5m。
        /// 08-18 视频定案：写死 1.5m 时站 2.5m 外看墙 slab 全落空块，正眼区断供。
        /// </summary>
        private float GetGazeDistance()
        {
            if (_depthCapture != null)
            {
                float d = _depthCapture.LastCenterDepthMeters;
                if (d > 0.3f) return Mathf.Clamp(d + 0.15f, 0.6f, 6f);
            }
            return 1.5f;
        }

        /// <summary>世界坐标 → 冻结块（体积界外返回 false）。与 GazeBlockStatus 同换算。</summary>
        private bool TryWorldToFrozenBlock(Vector3 worldPoint, out int block, out Unity.Mathematics.int3 coord)
        {
            block = -1;
            coord = default;
            Vector3 local = _meshExtractor.transform.InverseTransformPoint(worldPoint);
            var vox = _volumeIntegrator.VoxelCount;
            float vs = _volumeIntegrator.VoxelSize;
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            int vx = Mathf.FloorToInt(local.x / vs + vox.x * 0.5f);
            int vy = Mathf.FloorToInt(local.y / vs + vox.y * 0.5f);
            int vz = Mathf.FloorToInt(local.z / vs + vox.z * 0.5f);
            if (vx < 0 || vy < 0 || vz < 0 || vx >= vox.x || vy >= vox.y || vz >= vox.z)
                return false;
            int bx = Mathf.Min(vx / blockSize, grid.x - 1);
            int by = Mathf.Min(vy / blockSize, grid.y - 1);
            int bz = Mathf.Min(vz / blockSize, grid.z - 1);
            coord = new Unity.Mathematics.int3(bx, by, bz);
            block = bx + grid.x * (by + grid.y * bz);
            return true;
        }

        /// <summary>冻结块线性索引 → 块坐标（冻结网格与 HERA 父页 32³ 同构，直译）。</summary>
        private Unity.Mathematics.int3 FrozenBlockCoord3(int b)
        {
            var grid = _volumeIntegrator.FrozenChunkCount;
            return new Unity.Mathematics.int3(
                b % grid.x, (b / grid.x) % grid.y, b / (grid.x * grid.y));
        }

        /// <summary>HUD 视线块状态：头显正前方深度锚定落点所在冻结块的定稿阶段+红占比。</summary>
        /// <summary>
        /// 视线落点块线性索引（GazeBlockStatus 同源数学，冻前探头用）：
        /// 无相机/未就绪/界外=false。调用方需保证 _meshExtractor 非空。
        /// </summary>
        private bool TryGetGazeBlockIndex(out int blockIndex)
        {
            blockIndex = -1;
            var cam = Camera.main;
            if (cam == null || _volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady)
                return false;
            float gazeDist = GetGazeDistance();
            Vector3 point = cam.transform.position + cam.transform.forward * gazeDist;
            // 体积以自身变换原点为中心：local = (voxel + 0.5 - count/2) * voxSize。
            Vector3 local = _meshExtractor.transform.InverseTransformPoint(point);
            var vox = _volumeIntegrator.VoxelCount;
            float vs = _volumeIntegrator.VoxelSize;
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            int vx = Mathf.FloorToInt(local.x / vs + vox.x * 0.5f);
            int vy = Mathf.FloorToInt(local.y / vs + vox.y * 0.5f);
            int vz = Mathf.FloorToInt(local.z / vs + vox.z * 0.5f);
            if (vx < 0 || vy < 0 || vz < 0 || vx >= vox.x || vy >= vox.y || vz >= vox.z)
                return false;
            int bx = Mathf.Min(vx / blockSize, grid.x - 1);
            int by = Mathf.Min(vy / blockSize, grid.y - 1);
            int bz = Mathf.Min(vz / blockSize, grid.z - 1);
            blockIndex = bx + grid.x * (by + grid.y * bz);
            return true;
        }

        private string GazeBlockStatus()
        {
            var cam = Camera.main;
            if (cam == null || _volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady)
                return "—";
            float gazeDist = GetGazeDistance();
            Vector3 point = cam.transform.position + cam.transform.forward * gazeDist;
            // 体积以自身变换原点为中心：local = (voxel + 0.5 - count/2) * voxSize。
            Vector3 local = _meshExtractor.transform.InverseTransformPoint(point);
            var vox = _volumeIntegrator.VoxelCount;
            float vs = _volumeIntegrator.VoxelSize;
            var grid = _volumeIntegrator.FrozenChunkCount;
            int blockSize = Mathf.Max(1, vox.x / grid.x);
            int vx = Mathf.FloorToInt(local.x / vs + vox.x * 0.5f);
            int vy = Mathf.FloorToInt(local.y / vs + vox.y * 0.5f);
            int vz = Mathf.FloorToInt(local.z / vs + vox.z * 0.5f);
            if (vx < 0 || vy < 0 || vz < 0 || vx >= vox.x || vy >= vox.y || vz >= vox.z)
                return "界外";
            int bx = Mathf.Min(vx / blockSize, grid.x - 1);
            int by = Mathf.Min(vy / blockSize, grid.y - 1);
            int bz = Mathf.Min(vz / blockSize, grid.z - 1);
            int b = bx + grid.x * (by + grid.y * bz);
            string suffix = $"@{gazeDist:0.0}m";
            if (!_frozenBlocks.Contains(b))
                return (_maturityPrevSurface != null && b < _maturityPrevSurface.Length &&
                       _maturityPrevSurface[b] >= frozenBlockMinSurfaceVoxels
                    ? "近熟待稳"
                    : "成长中") + suffix;
            var coord = new Unity.Mathematics.int3(bx, by, bz);
            if (_meshExtractor.TryGetIncrementalPageTally(coord, out long red, out long total) && total > 0)
                return $"已定稿·红{100f * red / total:0}%{suffix}";
            return (_meshExtractor.IncrementalParentBlockInFlight(coord) ? "精修中" : "排队中") + suffix;
        }

        // ─────────────────────────────────────────────────────────────
        //  公开 API（输入处理器调用）
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Freeze TSDF production and let only the InfiniTAM block extractor
        /// drain.  This is a read/observe experiment: it changes no fusion,
        /// extraction, crease or publication threshold.
        /// </summary>
        public bool TryBeginInfiniTamMeshTailValidation()
        {
            if (_meshTailValidationActive)
                return true;
            if (!IsScanning || _volumeIntegrator == null || _meshExtractor == null ||
                !_volumeIntegrator.InfiniTamBaselineEnabled)
                return false;

            // A is the single production endpoint.  Stop the full replay first
            // so its last accepted depth/fusion rows describe the exact TSDF
            // that is frozen below.  ScanReplaySessionPackage.End drains and
            // seals asynchronously, while this method independently drains the
            // mesh queue; neither path may admit another source frame.
            _meshTailReasonLedgerSessionDirectory = _depthCapture != null
                ? _depthCapture.PairedFrameCaptureDirectory
                : string.Empty;
            _meshTailReasonLedgerSealRequested = _depthCapture != null &&
                                                 _depthCapture.PairedFrameCaptureActive;
            if (_meshTailReasonLedgerSealRequested)
            {
                _depthCapture.TogglePairedFrameCapture();
                Logger.Info("A键：完整原因账已停止采样并开始排空封包：" +
                            _meshTailReasonLedgerSessionDirectory);
            }
            else
            {
                Logger.Warning("A键：没有活动的完整原因账；仍继续冻结并检查网格尾随");
            }
            ShowMeshTailSealHud("准备排空");

            if (!_meshExtractor.PrepareInfiniTamTailProbe())
            {
                _meshTailSealHudState = "未启动：分块网格未就绪";
                WriteRejectedMeshTailReceipt("infinitam_block_pipeline_not_ready");
                // A remains a hard production endpoint even when the tail
                // extractor cannot start: the reason ledger is already sealed,
                // so accepting more TSDF writes here would split the two books.
                if (IsScanning)
                    PauseScanning();
                NotifyInput(_meshTailReasonLedgerSealRequested
                    ? "原因账封口中·尾随未启动：分块网格尚未就绪"
                    : "尾随验证未启动：分块网格尚未就绪");
                Logger.Warning("网格尾随验证未启动：InfiniTAM 分块提取器尚未就绪");
                RefreshStatusBadge();
                return true;
            }

            _meshTailValidationActive = true;
            _meshTailSealHudState = "排空中";
            _meshTailValidationStartedAt = Time.realtimeSinceStartup;
            _meshTailValidationStartedUtc = DateTime.UtcNow;
            _meshTailValidationStableTicks = 0;
            _meshTailStartIntegrationCount = _volumeIntegrator.IntegrationCount;
            _meshTailStartDirtyEpoch = _volumeIntegrator.DirtyEpoch;
            _meshTailStartLedgerApplyCount = _meshExtractor.InfiniTamDirtyLedgerApplyCount;
            _meshTailStartAcceptedCommitCount = _meshExtractor.InfiniTamAcceptedCommitCount;
            _meshTailStartCompletedBatchCount = _meshExtractor.InfiniTamCompletedBatchCount;
            _meshTailStartPublishedBatchCount = _meshExtractor.InfiniTamPublishedBatchCount;
            _meshTailStartStaleDiscardCount =
                _meshExtractor.InfiniTamStaleCandidateDiscardCount;
            _meshTailStartVertexCount = _meshExtractor.InfiniTamCommittedVertexCount;
            _meshTailStartIndexCount = _meshExtractor.InfiniTamCommittedIndexCount;
            _meshTailStartVisibleBlocks = _meshExtractor.InfiniTamVisibleBlockCount;
            _meshTailValidationSamples.Clear();
            _meshTailValidationSamples.AppendLine(
                "seconds,integration_count,dirty_epoch,ledger_applies,accepted_commits," +
                "completed_batches,published_batches,batch_in_flight,stale_discards," +
                "queued,in_flight,outstanding,epoch_debt,visible_blocks,vertices,indices");
            AppendMeshTailValidationSample(0f);

            // Create the receipt at entry, not only at successful completion.
            // A missing file therefore means the A route never entered this
            // experiment; a RUNNING file means it entered but did not seal.
            _meshTailValidationOutputPath = string.Empty;
            try
            {
                string directory = Path.Combine(Application.persistentDataPath,
                    "ScanCoverDiagnostics");
                Directory.CreateDirectory(directory);
                string stamp = _meshTailValidationStartedUtc.ToString(
                    "yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
                _meshTailValidationOutputPath = Path.Combine(directory,
                    $"mesh_tail_{stamp}.txt");
                var runningReceipt = new StringBuilder(512);
                runningReceipt.AppendLine("schema=mesh_tail_validation_v4");
                runningReceipt.AppendLine("state=RUNNING");
                runningReceipt.AppendLine("authority=diagnostic_only_no_production_threshold_changes");
                runningReceipt.AppendLine("acceptance=tsdf_frozen_and_drained_and_published_batches_after_freeze_lte_1");
                runningReceipt.AppendLine($"started_utc={_meshTailValidationStartedUtc:O}");
                runningReceipt.AppendLine($"integration_count_start={_meshTailStartIntegrationCount}");
                runningReceipt.AppendLine($"dirty_epoch_start={_meshTailStartDirtyEpoch}");
                runningReceipt.AppendLine($"completed_batches_start={_meshTailStartCompletedBatchCount}");
                runningReceipt.AppendLine($"published_batches_start={_meshTailStartPublishedBatchCount}");
                runningReceipt.AppendLine($"stale_discards_start={_meshTailStartStaleDiscardCount}");
                runningReceipt.AppendLine($"reason_ledger_seal_requested={_meshTailReasonLedgerSealRequested.ToString().ToLowerInvariant()}");
                runningReceipt.AppendLine($"reason_ledger_session={_meshTailReasonLedgerSessionDirectory}");
                runningReceipt.AppendLine($"pipeline_start={_meshExtractor.InfiniTamBlockStatsCompact}");
                File.WriteAllText(_meshTailValidationOutputPath,
                    runningReceipt.ToString(), new UTF8Encoding(false));
            }
            catch (Exception ex)
            {
                Logger.Error($"网格尾随验证入口小票写入失败: {ex.Message}");
            }

            // Force the first post-freeze dirty-ledger observation instead of
            // waiting for the ordinary 5 Hz deadline left by the live scan.
            _meshExtractor.RequestImmediateInfiniTamDirtyLedgerRefresh();
            _lastMeshTime = 0f;
            _hudStatus = _meshTailReasonLedgerSealRequested
                ? "原因账封口中·TSDF冻结·排空网格尾随"
                : "无活动原因账·TSDF冻结·排空网格尾随";
            NotifyInput(_meshTailReasonLedgerSealRequested
                ? "A键：原因账封口＋尾随排空已开始"
                : "A键：尾随排空已开始（无活动原因账）");
            Logger.Info($"网格尾随验证开始: 融合={_meshTailStartIntegrationCount}, " +
                        $"脏纪元={_meshTailStartDirtyEpoch}, " +
                        $"提交={_meshTailStartAcceptedCommitCount}, " +
                        $"发布批={_meshTailStartPublishedBatchCount}, " +
                        $"{_meshExtractor.InfiniTamBlockStatsCompact}");
            RefreshStatusBadge();
            return true;
        }

        private void TickMeshTailValidation()
        {
            if (!_meshTailValidationActive || _meshExtractor == null ||
                _volumeIntegrator == null)
                return;

            float now = Time.realtimeSinceStartup;
            float elapsed = now - _meshTailValidationStartedAt;
            if (now - _lastMeshTime >= MeshInterval)
            {
                _lastMeshTime = now;
                _meshExtractor.Extract();
                MeshExtracted?.Invoke();
                AppendMeshTailValidationSample(elapsed);

                bool fusionFrozen =
                    _volumeIntegrator.IntegrationCount == _meshTailStartIntegrationCount &&
                    _volumeIntegrator.DirtyEpoch == _meshTailStartDirtyEpoch;
                bool ledgerObservedAfterFreeze =
                    _meshExtractor.InfiniTamDirtyLedgerApplyCount >
                    _meshTailStartLedgerApplyCount;
                bool drained = ledgerObservedAfterFreeze &&
                    _meshExtractor.InfiniTamPendingBlockWork == 0 &&
                    _meshExtractor.InfiniTamOutstandingBlockCount == 0 &&
                    _meshExtractor.InfiniTamEpochDebt == 0ul;

                _meshTailValidationStableTicks = fusionFrozen && drained
                    ? _meshTailValidationStableTicks + 1
                    : 0;

                if (!fusionFrozen)
                {
                    FinishMeshTailValidation("INVALID_TSDF_CHANGED", false);
                    return;
                }

                if (elapsed >= 1f &&
                    _meshTailValidationStableTicks >= MeshTailValidationStableTicks)
                {
                    FinishMeshTailValidation("DRAINED", false);
                    return;
                }
            }

            if (elapsed >= MeshTailValidationTimeoutSeconds)
                FinishMeshTailValidation("TIMEOUT_WITH_BACKLOG", true);
        }

        private void AppendMeshTailValidationSample(float elapsed)
        {
            if (_meshExtractor == null || _volumeIntegrator == null)
                return;
            _meshTailValidationSamples
                .Append(elapsed.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                .Append(_volumeIntegrator.IntegrationCount).Append(',')
                .Append(_volumeIntegrator.DirtyEpoch).Append(',')
                .Append(_meshExtractor.InfiniTamDirtyLedgerApplyCount).Append(',')
                .Append(_meshExtractor.InfiniTamAcceptedCommitCount).Append(',')
                .Append(_meshExtractor.InfiniTamCompletedBatchCount).Append(',')
                .Append(_meshExtractor.InfiniTamPublishedBatchCount).Append(',')
                .Append(_meshExtractor.InfiniTamBatchInFlight ? 1 : 0).Append(',')
                .Append(_meshExtractor.InfiniTamStaleCandidateDiscardCount).Append(',')
                .Append(_meshExtractor.InfiniTamQueuedBlockCount).Append(',')
                .Append(_meshExtractor.InfiniTamInFlightCommitCount).Append(',')
                .Append(_meshExtractor.InfiniTamOutstandingBlockCount).Append(',')
                .Append(_meshExtractor.InfiniTamEpochDebt).Append(',')
                .Append(_meshExtractor.InfiniTamVisibleBlockCount).Append(',')
                .Append(_meshExtractor.InfiniTamCommittedVertexCount).Append(',')
                .Append(_meshExtractor.InfiniTamCommittedIndexCount).AppendLine();
        }

        private void FinishMeshTailValidation(string drainState, bool timedOut)
        {
            if (!_meshTailValidationActive)
                return;
            _meshTailValidationActive = false;

            float elapsed = Time.realtimeSinceStartup - _meshTailValidationStartedAt;
            AppendMeshTailValidationSample(elapsed);
            int endIntegrations = _volumeIntegrator != null
                ? _volumeIntegrator.IntegrationCount : -1;
            uint endDirtyEpoch = _volumeIntegrator != null
                ? _volumeIntegrator.DirtyEpoch : 0u;
            long endCommits = _meshExtractor != null
                ? _meshExtractor.InfiniTamAcceptedCommitCount : -1L;
            long endCompletedBatches = _meshExtractor != null
                ? _meshExtractor.InfiniTamCompletedBatchCount : -1L;
            long endPublishedBatches = _meshExtractor != null
                ? _meshExtractor.InfiniTamPublishedBatchCount : -1L;
            long endStaleDiscards = _meshExtractor != null
                ? _meshExtractor.InfiniTamStaleCandidateDiscardCount : -1L;
            long endVertices = _meshExtractor != null
                ? _meshExtractor.InfiniTamCommittedVertexCount : -1L;
            long endIndices = _meshExtractor != null
                ? _meshExtractor.InfiniTamCommittedIndexCount : -1L;
            int endVisibleBlocks = _meshExtractor != null
                ? _meshExtractor.InfiniTamVisibleBlockCount : -1;
            long commitDelta = Math.Max(0L, endCommits - _meshTailStartAcceptedCommitCount);
            long completedBatchDelta = Math.Max(0L,
                endCompletedBatches - _meshTailStartCompletedBatchCount);
            long publishedBatchDelta = Math.Max(0L,
                endPublishedBatches - _meshTailStartPublishedBatchCount);
            long staleDiscardDelta = Math.Max(0L,
                endStaleDiscards - _meshTailStartStaleDiscardCount);
            bool fusionFrozen = endIntegrations == _meshTailStartIntegrationCount &&
                                endDirtyEpoch == _meshTailStartDirtyEpoch;
            string verdict = !fusionFrozen
                ? "INVALID_TSDF_CHANGED"
                : timedOut
                    ? "TAIL_BACKLOG_TIMEOUT"
                    : publishedBatchDelta == 0
                        ? "NO_TAIL_OBSERVED_IN_THIS_WINDOW"
                        : publishedBatchDelta == 1
                            ? "BOUNDED_SINGLE_BATCH_TAIL"
                            : "MULTI_BATCH_TAIL";

            string outputPath = string.Empty;
            try
            {
                outputPath = _meshTailValidationOutputPath;
                if (string.IsNullOrEmpty(outputPath))
                {
                    string directory = Path.Combine(Application.persistentDataPath,
                        "ScanCoverDiagnostics");
                    Directory.CreateDirectory(directory);
                    string stamp = _meshTailValidationStartedUtc.ToString(
                        "yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
                    outputPath = Path.Combine(directory, $"mesh_tail_{stamp}.txt");
                }
                var report = new StringBuilder(8192);
                report.AppendLine("schema=mesh_tail_validation_v4");
                report.AppendLine("state=SEALED");
                report.AppendLine("authority=diagnostic_only_no_production_threshold_changes");
                report.AppendLine("acceptance=tsdf_frozen_and_drained_and_published_batches_after_freeze_lte_1");
                report.AppendLine($"verdict={verdict}");
                report.AppendLine($"drain_state={drainState}");
                report.AppendLine($"elapsed_seconds={elapsed.ToString("F3", CultureInfo.InvariantCulture)}");
                report.AppendLine($"tsdf_frozen={fusionFrozen.ToString().ToLowerInvariant()}");
                report.AppendLine($"integration_count={_meshTailStartIntegrationCount}->{endIntegrations}");
                report.AppendLine($"dirty_epoch={_meshTailStartDirtyEpoch}->{endDirtyEpoch}");
                report.AppendLine($"accepted_block_commits_after_freeze={commitDelta}");
                report.AppendLine($"completed_mesh_batches_after_freeze={completedBatchDelta}");
                report.AppendLine($"published_mesh_batches_after_freeze={publishedBatchDelta}");
                report.AppendLine($"stale_candidate_discards_after_freeze={staleDiscardDelta}");
                report.AppendLine($"visible_blocks={_meshTailStartVisibleBlocks}->{endVisibleBlocks}");
                report.AppendLine($"committed_vertices={_meshTailStartVertexCount}->{endVertices}");
                report.AppendLine($"committed_indices={_meshTailStartIndexCount}->{endIndices}");
                report.AppendLine($"reason_ledger_seal_requested={_meshTailReasonLedgerSealRequested.ToString().ToLowerInvariant()}");
                report.AppendLine($"reason_ledger_session={_meshTailReasonLedgerSessionDirectory}");
                report.AppendLine($"final_pipeline={_meshExtractor?.InfiniTamBlockStatsCompact ?? "missing"}");
                report.AppendLine();
                report.Append(_meshTailValidationSamples);
                File.WriteAllText(outputPath, report.ToString(), new UTF8Encoding(false));
            }
            catch (Exception ex)
            {
                Logger.Error($"网格尾随验证小票写入失败: {ex.Message}");
            }

            if (IsScanning)
                PauseScanning();
            string shortVerdict = !fusionFrozen
                ? "验证无效：TSDF发生写入"
                : timedOut
                    ? $"尾随未排空：提交{commitDelta}块后仍超时"
                    : publishedBatchDelta == 0
                        ? "无生产尾随"
                        : publishedBatchDelta == 1
                            ? $"尾随受控：仅1批（{commitDelta}块）"
                            : $"尾随未收束：{publishedBatchDelta}批（{commitDelta}块）";
            _meshTailSealHudState = !fusionFrozen
                ? "无效：TSDF发生写入"
                : timedOut
                    ? "未排空：超时"
                    : "已排空";
            _hudStatus = shortVerdict;
            NotifyInput(shortVerdict);
            Logger.Info($"网格尾随验证结束: {shortVerdict}, {elapsed:F2}s, " +
                        $"小票={outputPath}");
            RefreshStatusBadge();
        }

        private void WriteRejectedMeshTailReceipt(string reason)
        {
            try
            {
                DateTime now = DateTime.UtcNow;
                string directory = Path.Combine(Application.persistentDataPath,
                    "ScanCoverDiagnostics");
                Directory.CreateDirectory(directory);
                string path = Path.Combine(directory,
                    $"mesh_tail_{now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture)}.txt");
                var receipt = new StringBuilder(384);
                receipt.AppendLine("schema=mesh_tail_validation_v4");
                receipt.AppendLine("state=REJECTED");
                receipt.AppendLine($"reason={reason}");
                receipt.AppendLine($"created_utc={now:O}");
                receipt.AppendLine($"is_scanning={IsScanning.ToString().ToLowerInvariant()}");
                receipt.AppendLine($"infinitam_baseline={(_volumeIntegrator != null && _volumeIntegrator.InfiniTamBaselineEnabled).ToString().ToLowerInvariant()}");
                receipt.AppendLine($"reason_ledger_seal_requested={_meshTailReasonLedgerSealRequested.ToString().ToLowerInvariant()}");
                receipt.AppendLine($"reason_ledger_session={_meshTailReasonLedgerSessionDirectory}");
                receipt.AppendLine($"pipeline={_meshExtractor?.InfiniTamBlockStatsCompact ?? "missing"}");
                File.WriteAllText(path, receipt.ToString(), new UTF8Encoding(false));
            }
            catch (Exception ex)
            {
                Logger.Error($"网格尾随验证拒绝小票写入失败: {ex.Message}");
            }
        }

        /// <summary>
        /// 开始（或暂停后继续）深度融合与网格提取。
        /// 保留 QRS 原版的异步分段启动：先提交 GPU 体积资源和首提缓冲，
        /// 跨帧后再启用相机与深度。新空卷的整卷首提还会等待第一帧有效
        /// 融合及一个既有网格周期；只错开冷启动，不改变后续持续调度。
        /// 避免 PCA 硬件缓冲队列握手与计算调度同帧竞争导致 Vulkan 卡死。
        /// 继续扫描时 ReallocateVolumes / EnsureInitialized 均为幂等 no-op，
        /// 已有体积数据保持不变。
        /// </summary>
        public async Task StartScanningAsync()
        {
            if (IsScanning || IsSaveAndClearInProgress) return;
            if (enableFrozenChunkAbExperiment && _chunkAbFrozen)
            {
                NotifyInput("TSDF 已冻结；B 只清当前档，Y 可切换档位");
                return;
            }
            IsScanning = true;
            try
            {
                bool resuming = HasStarted;
                bool armInfiniTamStartupMeshGate = !resuming &&
                    !enableFrozenChunkAbExperiment &&
                    _volumeIntegrator != null &&
                    _volumeIntegrator.InfiniTamBaselineEnabled;
                if (armInfiniTamStartupMeshGate)
                {
                    _infiniTamStartupFirstMeshPending = true;
                    _infiniTamStartupFirstFusionSubmitted = false;
                    _infiniTamStartupFirstMeshNotBefore = float.PositiveInfinity;
                }

                // 开扫前最后一次固化身份；后续空卷开关由 HasStarted 锁死。
                SyncCaptureModeIdentity();
                // 阶段 1：GPU 体积 bring-up
                _volumeIntegrator.ReallocateVolumes();
                await Task.Yield();
                await Task.Yield();

                // A/B 采集只保留共享 TSDF。旧生产提取器在整个实验中退出。
                if (enableFrozenChunkAbExperiment)
                    _meshExtractor.PrepareForChunkAbAcquisition();
                else
                    _meshExtractor.EnsureInitialized();
                await Task.Yield();
                await Task.Yield();

                float t = Time.time;
                _lastIntegrationTime = t;
                _lastMeshTime = t;
                _cameraAvailable = false;

                if (!resuming)
                {
                    // A-only seal HUD survives long enough for the operator to
                    // read 安全退出[是], then disappears with the next fresh roll.
                    _meshTailSealHudForcedVisible = false;
                    _meshTailSealHudState = "未开始";
                    if (_statusBadgeRoot != null && !showOperatorHud)
                        _statusBadgeRoot.SetActive(false);
                    _instantDepthShellOverlay?.ResetProductionWitnessLedger();
                    if (!(_volumeIntegrator != null &&
                          _volumeIntegrator.InfiniTamBaselineEnabled) &&
                        _instantDepthShellOverlay != null &&
                        !_instantDepthShellOverlay.BeginAutomaticSeedCapture())
                        Logger.Warning("自动基底未启动：种子平面生产入口未启用");
                    // Establish the paper ledger before camera/depth can publish
                    // the first candidate.  A/B acquisition uses the same native
                    // 5 cm paper and must not leave its session id as "未开始".
                    _meshExtractor.BeginLedgerSession();
                }

                // 阶段 3：相机 + 深度（此时启动安全）
                _cameraProvider?.StartCapture();
                _depthCapture.StartDepthCapture();

                // 完整回放会话会逐帧回读原始/处理深度和融合输入，
                // 属于昂贵诊断，不得再跟随普通生产扫描自动开启。
                // 需要取证时，操作者必须在空卷、按扣机之前用左握把+X
                // 显式开启；StartCapture 仍会拒绝中途开账，保持可复现契约。
                if (!resuming && !_depthCapture.PairedFrameCaptureActive)
                    Logger.Info("生产扫描：完整回放采集未自动开启（空卷左握把+X可手动取证）");
                if (enableFrozenChunkAbExperiment)
                {
                    if (!resuming) _coverageOverlay?.ResetCoverage();
                    _coverageOverlay?.SetAcquiring(true);
                    // 增量精修：点阵默认隐藏（X 呼出当判官）；旧两段式保持默认可见。
                    bool incremental = enableIncrementalHeraRefine && enableHeraHierarchicalReplay;
                    _coverageOverlay?.SetMarkersVisible(!incremental);
                    if (incremental)
                    {
                        _meshExtractor.BeginIncrementalHera(abMaxChunksPerTick);
                        if (!resuming)
                            _meshExtractor.ShowProductionPaperView();
                    }
                }
                _depthPointCloudOverlay?.SetAcquiring(true);
                _instantDepthShellOverlay?.SetAcquiring(true);
                if (!resuming) _gunGelCourtOverlay?.SetSealed(false);
                HasStarted = true;
                _hudStatus = resuming ? "扫描中(继续)" : "扫描中";
                _hudLastError = "";
                Logger.Info($"开始扫描 — 继续上次={resuming}, 已累计融合={_volumeIntegrator.IntegrationCount}");
                RefreshStatusBadge();
                ScanStarted?.Invoke();
            }
            catch (Exception e)
            {
                // 重置重入保护，允许用户重试
                IsScanning = false;
                if (!HasStarted)
                {
                    _infiniTamStartupFirstMeshPending = false;
                    _infiniTamStartupFirstFusionSubmitted = false;
                    _infiniTamStartupFirstMeshNotBefore = 0f;
                }
                _hudStatus = "启动失败";
                _hudLastError = e.Message;
                RefreshStatusBadge();
                throw;
            }
        }

        /// <summary>暂停融合与相机，体积与网格数据保留，可再次开始继续。</summary>
        public void PauseScanning()
        {
            if (!IsScanning) return;
            IsScanning = false;

            _depthPointCloudOverlay?.SetAcquiring(false);
            _instantDepthShellOverlay?.SetAcquiring(false);

            _cameraProvider?.StopCapture();
            _depthCapture.StopDepthCapture();

            Logger.Info("已暂停 — 扳机继续，B 保存累计账并清空");
            _hudStatus = "已暂停";
            RefreshStatusBadge();
            ScanStopped?.Invoke();
        }

        /// <summary>
        /// Freeze the shared TSDF and seal the visible incremental HERA front.
        /// Coverage tiles are hidden first; already-published HERA snapshots
        /// stay resident while never-built pages finish their bookkeeping.
        /// </summary>
        public void FreezeChunkAbTsdf()
        {
            if (!enableFrozenChunkAbExperiment)
            {
                PauseScanning();
                SealGunGelCourtViewIfVisible();
                return;
            }
            if (_chunkAbFrozen || !HasStarted) return;

            _coverageOverlay?.SetAcquiring(false);
            PauseScanning();
            SealGunGelCourtViewIfVisible();
            // A 是本次扫描的自然终点：若独立回放会话仍在采集，先停止接收新帧，
            // 让会话封装器接管几何/纸/枪胶/MRUK伴随物并等待全部回读写盘后原子封口。
            bool replayPackageFinalizing = _depthCapture != null &&
                                           _depthCapture.PairedFrameCaptureActive;
            if (replayPackageFinalizing)
            {
                _depthCapture.TogglePairedFrameCapture();
                // 完整输出账以“停止接收并排空封包”为 A 键终点。不要再切换到
                // 冻结/HERA 回放：那会替换当前生产管线状态，令停止瞬间之后的
                // 队列、纸皮和显示证据混入另一种运行模式。当前已提交纸皮保持
                // 原样可见，所有在途只由会话封装器排空并校验。
                _hudStatus = "完整账封口中";
                RefreshStatusBadge();
                Logger.Info("A键：完整输出账停止采样并排空封包；未进入冻结回放");
                return;
            }
            // 必须在 BeginFrozenHeraReplay 替换增量 32³ 管线之前落账，
            // 否则扫描期的块同步债会被冻结回放状态覆盖。
            string geometrySnapshot = !replayPackageFinalizing && _meshExtractor != null
                ? _meshExtractor.ExportFirstStageGeometrySnapshot("A键冻结前")
                : "";
            if (!string.IsNullOrEmpty(geometrySnapshot))
                Logger.Info($"A键已封存第一阶段诊断: {geometrySnapshot}");
            // 枪胶候选层在淘汰前保留稳定 ID / 双证词 / 反对票 / 空间位置；
            // 与纸层并行落盘，供 hold=4/6 之后按世界位置追责数据源。
            if (!replayPackageFinalizing && _volumeIntegrator != null &&
                !_volumeIntegrator.RequestGunGelCandidateAuditExport("A键冻结前",
                    path =>
                    {
                        if (!string.IsNullOrEmpty(path))
                            Logger.Info($"A键已封存枪胶候选黑匣子: {path}");
                    }))
                Logger.Warning("A键枪胶候选黑匣子未启动（影子层未就绪或已有导出在途）");
            _chunkAbFrozen = true;
            if (enableHeraHierarchicalReplay)
            {
                _meshExtractor.BeginFrozenHeraReplay(abMaxChunksPerTick);
                _heraFreezeStartedAt = Time.realtimeSinceStartup;
                _heraLastProgressAt = _heraFreezeStartedAt;
                _heraLastProgressSignature = "";
            }
            else
                _meshExtractor.BeginFrozenChunkReplay(ActiveChunkAbSize, abMaxChunksPerTick);
            _chunkAbDiagnosticColoring = true;
            RefreshStatusBadge();
            Logger.Info(enableHeraHierarchicalReplay
                ? "共享 TSDF 已冻结；当前 HERA 前台已封存，仅补未建页"
                : $"共享 TSDF 已冻结；开始 {ActiveChunkAbSize}³ 只读切块回放");
        }

        private void SealGunGelCourtViewIfVisible()
        {
            if (_gunGelCourtOverlay == null || !_gunGelCourtOverlay.Visible) return;
            if (_volumeIntegrator == null ||
                !_volumeIntegrator.RequestGunGelCourtSeal(success =>
                {
                    _gunGelCourtOverlay?.SetSealed(success);
                    NotifyInput(success
                        ? "裁决海已封存：绿静海/红浪头"
                        : "裁决海封存失败：账本已变化");
                }))
                NotifyInput("裁决海封存未启动（枪胶层未就绪或正在封存）");
            else
                NotifyInput("裁决海封存中：等待在途枪弹排空");
        }

        /// <summary>Y: cycle 64³/32³/16³ against the same frozen TSDF.</summary>
        public void CycleChunkAbGear()
        {
            if (!enableFrozenChunkAbExperiment || !_chunkAbFrozen) return;
            if (enableHeraHierarchicalReplay)
            {
                NotifyInput("HERA 自动分层：64记账 / 32分流 / 16精修");
                return;
            }
            _chunkAbGearIndex = (_chunkAbGearIndex + 1) % ChunkAbSizes.Length;
            _meshExtractor.BeginFrozenChunkReplay(ActiveChunkAbSize, abMaxChunksPerTick);
            _chunkAbDiagnosticColoring = true;
            RefreshStatusBadge();
            NotifyInput($"切到 {ActiveChunkAbSize}³");
        }

        /// <summary>
        /// B: export and release the active derived replay only.  The frozen
        /// TSDF and the other two future gears remain untouched.
        /// </summary>
        public void ExportAndClearActiveChunkAbGear()
        {
            if (!enableFrozenChunkAbExperiment || !_chunkAbFrozen) return;
            if (enableHeraHierarchicalReplay && _meshExtractor != null && _meshExtractor.HasFrozenHeraReplay &&
                !_meshExtractor.FrozenHeraReplayComplete)
            {
                string message = _meshExtractor.FrozenHeraReplayFailed
                    ? $"HERA 回放失败，禁止导出：{_meshExtractor.FrozenHeraReplayFailureReason}"
                    : $"账簿未闭环：还差 {_meshExtractor.FrozenHeraParentFinalizationPending} 个父页、" +
                      $"{_meshExtractor.FrozenHeraChildrenPending} 个子页、" +
                      $"{_meshExtractor.FrozenHeraFamiliesPending} 个家族裁决";
                NotifyInput(message);
                RefreshStatusBadge();
                return;
            }
            string path = enableHeraHierarchicalReplay
                ? (keepFrozenReplayAfterExport
                    ? _meshExtractor.ExportFrozenHeraReplayKeepVisible()
                    : _meshExtractor.ExportAndClearFrozenHeraReplay())
                : _meshExtractor.ExportAndClearFrozenChunkReplay();
            RefreshStatusBadge();
            NotifyInput(string.IsNullOrEmpty(path)
                ? "当前无可导出结果"
                : enableHeraHierarchicalReplay
                    ? (keepFrozenReplayAfterExport ? "已导出，回放保留（B键清场）" : "已导出 HERA 分层账并清空")
                    : $"已导出并清 {ActiveChunkAbSize}³");
        }

        /// <summary>Right thumbstick: single-color wireframe / state coloring.</summary>
        public void ToggleChunkAbDisplayMode()
        {
            if (!enableFrozenChunkAbExperiment || !_chunkAbFrozen) return;
            _chunkAbDiagnosticColoring = enableHeraHierarchicalReplay
                ? _meshExtractor.ToggleFrozenHeraReplayColoring()
                : _meshExtractor.ToggleFrozenChunkReplayColoring();
            RefreshStatusBadge();
            NotifyInput(_chunkAbDiagnosticColoring ? "状态着色" : "单色线框");
        }

        /// <summary>
        /// 将本次累计账导出后，原子式清空 TSDF、网格、融合计数与记账会话。
        /// A 暂停不会触发这里；下一次扳机将开始全新扫描和全新账簿。
        /// </summary>
        public void StopAndClearScan()
        {
            if (IsSaveAndClearInProgress) return;
            StartCoroutine(StopAndClearScanRoutine());
        }

        private System.Collections.IEnumerator StopAndClearScanRoutine()
        {
            IsSaveAndClearInProgress = true;
            bool wasActive = IsScanning || HasStarted;
            PauseScanning();
            _hudStatus = "正在保存";
            RefreshStatusBadge();

            string ledgerPath = "";
            if (_meshExtractor != null)
                yield return _meshExtractor.FinalizeAndExportLedgerAsync(
                    "B键保存并清空", path => ledgerPath = path);

            // B is a save-then-clear transaction.  If persistence failed, keep
            // the volume, mesh and open ledger intact so the user can retry.
            if (wasActive && string.IsNullOrEmpty(ledgerPath))
            {
                _hudStatus = "账簿保存失败，未清空";
                Logger.Error("累计账簿保存失败：已中止清空，扫描数据仍保留");
                IsSaveAndClearInProgress = false;
                RefreshStatusBadge();
                yield break;
            }

            // Invalidate outstanding GPU readbacks before touching the volume;
            // otherwise an old callback can repopulate a freshly cleared HUD.
            _meshExtractor?.ResetLedgerSessionAfterClear();
            HasStarted = false;
            _integrateCount = 0;
            _infiniTamStartupFirstMeshPending = false;
            _infiniTamStartupFirstFusionSubmitted = false;
            _infiniTamStartupFirstMeshNotBefore = 0f;

            _volumeIntegrator.Clear();
            _volumeIntegrator.ResetSessionCounters();
            _instantDepthShellOverlay?.ResetProductionWitnessLedger();
            yield return null;
            if (_meshExtractor.IsInitialized)
                _meshExtractor.DisposeOnly(); // 下一次扳机分帧重建，避免 B 键同帧释放+重分配

            if (wasActive)
                Logger.Info($"已保存并清空 — 账簿={ledgerPath}；扳机开始全新扫描");
            _hudStatus = ledgerPath.Length > 0 ? "已保存并清空" : "已清空(账簿为空)";
            IsSaveAndClearInProgress = false;
            RefreshStatusBadge();
        }

        /// <summary>
        /// X：关 → 枪胶裁决海 → BB 反投影 → 即时外壳独显 → 单帧种子平面 → 壳纸合流
        /// → 合流仅纸 → 关。“合流仅纸”保留同一次生产纸状态，只隐藏即时壳。
        /// 即时外壳只读当前清洗深度，短寿命三角膜不写 TSDF；通过独立视角
        /// 复核的三角仅发布存在证词，供后续原始深度辅助已有 provisional 转正；
        /// 在该档按 A 只钉住诊断快照，生产扫描继续。
        /// </summary>
        public void ToggleCoverageMarkers()
        {
            if (_depthPointCloudOverlay == null || _gunGelCourtOverlay == null ||
                _instantDepthShellOverlay == null) return;
            bool courtWasVisible = _gunGelCourtOverlay.Visible;
            bool bbWasVisible = _depthPointCloudOverlay.Visible;
            bool shellWasVisible = _instantDepthShellOverlay.Visible;
            bool seedWasVisible = shellWasVisible &&
                                  _instantDepthShellOverlay.SeedPreviewVisible;
            bool shellWasComposite = shellWasVisible &&
                                     _instantDepthShellOverlay.CompositeWithProduction;
            bool paperOnlyWasVisible = _shellPaperOnlyView;
            bool enteringDiagnostic = !paperOnlyWasVisible &&
                                      !courtWasVisible && !bbWasVisible && !shellWasVisible;

            // 旧 TSDF/HERA/纸面会污染融合前层的观察；诊断期间保持隐藏，
            // 视角覆盖账本及生产计算仍在后台继续。
            _coverageOverlay?.SetMarkersVisible(false);
            if (enteringDiagnostic)
            {
                _bbPresentationCaptured = true;
                _bbRestoreMeshVisible = _meshExtractor != null && _meshExtractor.IsAnyMeshVisible;
                _bbRestoreCoarseSkinVisible = _meshExtractor != null && _meshExtractor.IsCoarseSkinVisible;
                _bbRestoreManagementBlocks = showManagementBlockWireOverlay;
                _meshExtractor?.SetCurrentMeshDisplayVisible(false);
                _meshExtractor?.SetCoarseSkinVisible(false);
                showManagementBlockWireOverlay = false;
                _managementBlockWireOverlay?.SetVisible(false);
            }

            if (enteringDiagnostic)
            {
                _shellPaperOnlyView = false;
                _depthPointCloudOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetAcquiring(IsScanning);
                _gunGelCourtOverlay.SetVisible(true);
                NotifyInput("枪胶裁决海：红点=浪头≠Reject；框内红十字才是Reject");
            }
            else if (courtWasVisible)
            {
                _shellPaperOnlyView = false;
                _gunGelCourtOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetAcquiring(IsScanning);
                _depthPointCloudOverlay.SetVisible(true);
                _depthPointCloudOverlay.SetAcquiring(IsScanning);
                NotifyInput("BB反投影：白正视/洋红掠射");
            }
            else if (bbWasVisible)
            {
                _shellPaperOnlyView = false;
                _gunGelCourtOverlay.SetVisible(false);
                _depthPointCloudOverlay.SetVisible(false);
                _depthPointCloudOverlay.SetAcquiring(false);
                _instantDepthShellOverlay.SetVisible(true);
                _instantDepthShellOverlay.SetAcquiring(IsScanning);
                NotifyInput("即时外壳中央60%·线框粗格×2：按A定格；壳纸合流保持原样");
            }
            else if (shellWasVisible && !shellWasComposite && !seedWasVisible)
            {
                if (_depthCapture != null && _depthCapture.SeedPlaneProductionReady)
                {
                    // A staged seed surviving B-clear is already feeding this
                    // new empty roll. Do not pass through the auto-capture view
                    // again: that would replace the tested base and demand a
                    // second clear before the user can even reach the paper.
                    string route = _meshExtractor != null
                        ? _meshExtractor.ShowProductionPaperView()
                        : "无纸皮";
                    _meshExtractor?.SetCoarseSkinVisible(false);
                    showManagementBlockWireOverlay = false;
                    _managementBlockWireOverlay?.SetVisible(false);
                    _instantDepthShellOverlay.SetCompositeWithProduction(true);
                    _shellPaperOnlyView = false;
                    NotifyInput($"基底全流程：{route}；已跳过重复取样，当前纸皮来自种面→GunGel→裁决→TSDF");
                }
                else
                {
                    // 中央圆先观察原料；取到合格局部面后暂存为生产深度
                    // 基底。离开预览并从空卷扫描才会写入完整链路。
                    _instantDepthShellOverlay.SetSeedPreviewVisible(true);
                    NotifyInput("种子平面：等下一帧定格；绿=局部拟合，橙=原深度；取样后清卷再扫描验纸");
                }
            }
            else if (seedWasVisible)
            {
                // 不重新 SetVisible，避免切合流时清空刚看到的即时壳。纸皮明确
                // 回到真实 TSDF 生产档，外壳只作为短寿命前景叠加。
                _instantDepthShellOverlay.SetSeedPreviewVisible(false);
                string route = _meshExtractor != null
                    ? _meshExtractor.ShowProductionPaperView()
                    : "无纸皮";
                _meshExtractor?.SetCoarseSkinVisible(false);
                showManagementBlockWireOverlay = false;
                _managementBlockWireOverlay?.SetVisible(false);
                _instantDepthShellOverlay.SetCompositeWithProduction(true);
                _shellPaperOnlyView = false;
                NotifyInput($"壳纸合流：{route}近共面优先；更近即时壳保留前景；拒绝片不参与");
            }
            else if (shellWasComposite)
            {
                // 不离开 X 诊断链：只撤掉即时壳的上屏，保留已打开的
                // 10cm 大格生产纸皮和后台壳采证，不重选、不重启网格来源。
                // 其底层证据仍是同一份原生 5cm TSDF。
                _instantDepthShellOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetAcquiring(IsScanning);
                _shellPaperOnlyView = true;
                NotifyInput("合流仅纸：壳已隐藏，只看10cm大格纸皮（底层仍是同一5cm TSDF）");
            }
            else if (paperOnlyWasVisible)
            {
                _shellPaperOnlyView = false;
                _depthPointCloudOverlay.SetVisible(false);
                _depthPointCloudOverlay.SetAcquiring(false);
                _gunGelCourtOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetVisible(false);
                _instantDepthShellOverlay.SetAcquiring(IsScanning);
                NotifyInput("点诊断：关");
            }

            bool diagnosticStillVisible = _gunGelCourtOverlay.Visible ||
                                          _depthPointCloudOverlay.Visible ||
                                          _instantDepthShellOverlay.Visible ||
                                          _shellPaperOnlyView;
            if (!diagnosticStillVisible && _bbPresentationCaptured)
            {
                _meshExtractor?.SetCurrentMeshDisplayVisible(_bbRestoreMeshVisible);
                _meshExtractor?.SetCoarseSkinVisible(_bbRestoreCoarseSkinVisible);
                showManagementBlockWireOverlay = _bbRestoreManagementBlocks;
                _managementBlockWireOverlay?.SetVisible(_bbRestoreManagementBlocks);
                _bbPresentationCaptured = false;
            }
            ApplyDisplayMode();
            RefreshStatusBadge();
        }

        /// <summary>
        /// A 在即时外壳档的专用语义：定格/解除当前单层壳作世界空间对照，
        /// 不触碰生产冻结。
        /// 返回 true 表示本次 A 已被影子层消费。
        /// </summary>
        public bool TryToggleInstantShellFreeze()
        {
            if (_instantDepthShellOverlay == null || !_instantDepthShellOverlay.Visible)
                return false;
            if (_instantDepthShellOverlay.SeedPreviewVisible)
            {
                bool requested = _instantDepthShellOverlay.RequestSeedPreviewCapture();
                NotifyInput(requested
                    ? "种子平面：重取下一帧；新基底须从空卷进入后续链路"
                    : "种子平面：重取失败，请确认深度正在采集");
                RefreshStatusBadge();
                return true;
            }
            bool success = _instantDepthShellOverlay.ToggleDiagnosticFreeze(out bool frozen);
            NotifyInput(success
                ? frozen
                    ? "即时壳已定格：请横移20-40cm观察视差；后台仍采集"
                    : "即时壳已解除定格，恢复实时显示"
                : "即时壳定格未就绪；请先扳机采集");
            RefreshStatusBadge();
            return true;
        }

        /// <summary>
        /// 左握把+X：切换平台前处理/本工程后处理双路逐帧采集。
        /// 只增加诊断副本，不改变深度预处理、融合或显示。
        /// </summary>
        public void TogglePairedDepthFrameCapture()
        {
            if (_depthCapture == null)
            {
                NotifyInput("双深采集：DepthCapture 不可用");
                return;
            }

            bool wasActive = _depthCapture.PairedFrameCaptureActive;
            bool active = _depthCapture.TogglePairedFrameCapture();
            if (active)
            {
                NotifyInput("双深采集：开始（左握把+X停止）");
            }
            else if (wasActive)
            {
                NotifyInput($"双深采集：停止，待写{_depthCapture.PairedFrameCapturePending} " +
                            $"丢{_depthCapture.PairedFrameCaptureDropped}");
            }
            else
            {
                NotifyInput("双深采集：启动失败（查设备日志）");
            }
            RefreshStatusBadge();
        }

        /// <summary>严格生产网格已锁定；保留入口仅为旧输入/场景兼容。</summary>
        public void ToggleJointDiagnosticDisplay()
        {
            _meshExtractor?.ToggleJointDiagnosticDisplay();
        }

        // ─────────────────────────────────────────────────────────────
        //  内部
        // ─────────────────────────────────────────────────────────────

        private void SetupHeadExclusion()
        {
            if (_volumeIntegrator == null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                _volumeIntegrator.ExclusionZones.Add(cam.transform);
                Logger.Info($"头部排除区已添加: {cam.gameObject.name}");
            }
            else
            {
                Logger.Warning("未找到主相机，头部排除区未设置");
            }
        }

        private bool _cameraAvailable;
        private int _colorFrameLog;

        /// <summary>
        /// 每帧把透视相机帧喂给深度双边滤波（RGB 引导）与体积颜色融合。
        /// 与 QRS 原版一致：相机不可用时开启法线回退渲染。
        /// </summary>
        private void ProvideColorFrame()
        {
            ICameraProvider provider = _cameraProvider;

            bool cameraPlaying = provider != null && provider.IsPlaying;

            if (cameraPlaying && !_cameraAvailable)
            {
                _cameraAvailable = true;
                Shader.SetGlobalFloat(NormalFallbackID, 0f);
                Logger.Info("相机已运行 — 关闭法线回退渲染");
            }
            else if (!cameraPlaying && (_cameraAvailable || _colorFrameLog == 0))
            {
                _cameraAvailable = false;
                Shader.SetGlobalFloat(NormalFallbackID, 1f);
                Logger.Info("相机未运行 — 开启法线回退渲染（顶点色将为灰度）");
            }

            if (provider != null && provider.IsReady)
            {
                Texture frame = provider.CurrentFrame;
                if (frame != null)
                {
                    _depthCapture?.SetRGBGuide(frame);

                    Pose pose = provider.CameraPose;
                    if (_depthCapture != null)
                        pose = _depthCapture.TrackingToWorld(pose);

                    _volumeIntegrator.SetCameraData(
                        frame, pose.position, pose.rotation,
                        provider.FocalLength, provider.PrincipalPoint,
                        provider.SensorResolution, provider.CurrentResolution);

                    _colorFrameLog++;
                    if (_colorFrameLog <= 3 || _colorFrameLog % 50 == 0)
                        Logger.Verbose($"彩色帧 #{_colorFrameLog}: {frame.width}x{frame.height}");
                    return;
                }
            }

            _colorFrameLog++;
            _volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private static readonly int NormalFallbackID = Shader.PropertyToID("_RSNormalFallback");
        private static readonly int NoFreezeTintID = Shader.PropertyToID("_RSNoFreezeTint");
        private static readonly int TriAvailableID = Shader.PropertyToID("_RSTriAvailable");
        private static readonly int WireframeID = Shader.PropertyToID("_RSWireframe");
        private static readonly int WireThicknessID = Shader.PropertyToID("_RSWireThickness");
        private static readonly int MeshStrideID = Shader.PropertyToID("_RSMeshStride");
        private static readonly int GridSpacingID = Shader.PropertyToID("_RSGridSpacing");
        private static readonly int PaperGridModeID = Shader.PropertyToID("_RSPaperGridMode");
        private static readonly int ConfidenceVizID = Shader.PropertyToID("_RSConfidenceViz");
        private static readonly int GeometryTruthViewID = Shader.PropertyToID("_RSGeometryTruthView");
        private static readonly int PaperCorrectionHideActiveID =
            Shader.PropertyToID("_RSPaperCorrectionHideActive");
        private static readonly int PaperCorrectionKeyHashID =
            Shader.PropertyToID("_RSPaperCorrectionKeyHash");
        private static readonly int PaperCorrectionTargetHashID =
            Shader.PropertyToID("_RSPaperCorrectionTargetHash");
        private static readonly int PaperCorrectionHashMaskID =
            Shader.PropertyToID("_RSPaperCorrectionHashMask");
        private bool _confidenceVizApplied;
        private float _lastConfidenceStatsTime = -10f;

        private void RefreshPaperCorrectionMask()
        {
            // Retained only as a compatibility seam for the replay ledger.
            // Rendering must stay on the last visible native-5-cm baseline.
            Shader.SetGlobalInt(PaperCorrectionHashMaskID, -1);
            Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);
        }

        private void RebuildPaperCorrectionHash()
        {
            _paperCorrectionCellCount = _paperCorrectionCells.Count;
            if (_paperCorrectionCellCount == 0)
            {
                _paperCorrectionHashMask = -1;
                Shader.SetGlobalInt(PaperCorrectionHashMaskID, -1);
                return;
            }

            int capacity = 64;
            while (capacity < _paperCorrectionCellCount * 4)
                capacity <<= 1;

            while (true)
            {
                if (_paperCorrectionKeyEntries == null ||
                    _paperCorrectionKeyEntries.Length != capacity)
                {
                    _paperCorrectionKeyEntries = new Vector4[capacity];
                    _paperCorrectionTargetEntries = new Vector4[capacity];
                }
                else
                {
                    Array.Clear(_paperCorrectionKeyEntries, 0,
                        _paperCorrectionKeyEntries.Length);
                    Array.Clear(_paperCorrectionTargetEntries, 0,
                        _paperCorrectionTargetEntries.Length);
                }

                if (TryPopulatePaperCorrectionHash(capacity)) break;
                capacity <<= 1;
            }

            if (_paperCorrectionKeyHash == null ||
                _paperCorrectionKeyHash.count != capacity)
            {
                _paperCorrectionKeyHash?.Release();
                _paperCorrectionTargetHash?.Release();
                _paperCorrectionKeyHash = new ComputeBuffer(
                    capacity, sizeof(float) * 4, ComputeBufferType.Structured);
                _paperCorrectionTargetHash = new ComputeBuffer(
                    capacity, sizeof(float) * 4, ComputeBufferType.Structured);
            }
            _paperCorrectionKeyHash.SetData(_paperCorrectionKeyEntries);
            _paperCorrectionTargetHash.SetData(_paperCorrectionTargetEntries);
            _paperCorrectionHashMask = capacity - 1;
            Shader.SetGlobalBuffer(PaperCorrectionKeyHashID,
                _paperCorrectionKeyHash);
            Shader.SetGlobalBuffer(PaperCorrectionTargetHashID,
                _paperCorrectionTargetHash);
            Shader.SetGlobalInt(PaperCorrectionHashMaskID,
                _paperCorrectionHashMask);
        }

        private bool TryPopulatePaperCorrectionHash(int capacity)
        {
            int mask = capacity - 1;
            for (int i = 0; i < _paperCorrectionCells.Count; i++)
            {
                VirtualProbeShadowAdjudicator.PaperCorrectionCell cell =
                    _paperCorrectionCells[i];
                if (cell.Axis < 0 || cell.Axis > 5) continue;
                int axisFamily = cell.Axis / 2;
                int start = (int)(PaperCorrectionHash(
                    cell.X, cell.Y, cell.Z, axisFamily) & (uint)mask);
                bool inserted = false;
                for (int probe = 0; probe < PaperCorrectionHashProbeCount; probe++)
                {
                    int slot = (start + probe) & mask;
                    Vector4 entry = _paperCorrectionKeyEntries[slot];
                    if (entry.w == 0f)
                    {
                        _paperCorrectionKeyEntries[slot] = new Vector4(
                            cell.X, cell.Y, cell.Z,
                            (cell.Mode << 4) | (axisFamily + 1));
                        _paperCorrectionTargetEntries[slot] = new Vector4(
                            cell.Target.x, cell.Target.y, cell.Target.z, 0f);
                        inserted = true;
                        break;
                    }
                    if (Mathf.RoundToInt(entry.x) == cell.X &&
                        Mathf.RoundToInt(entry.y) == cell.Y &&
                        Mathf.RoundToInt(entry.z) == cell.Z &&
                        (Mathf.RoundToInt(entry.w) & 15) == axisFamily + 1)
                    {
                        int storedMode = Mathf.RoundToInt(entry.w) >> 4;
                        if (storedMode == 1 || cell.Mode == 1)
                        {
                            entry.w = (1 << 4) | (axisFamily + 1);
                            _paperCorrectionKeyEntries[slot] = entry;
                        }
                        else if (Vector3.Distance(
                                     _paperCorrectionTargetEntries[slot],
                                     cell.Target) > 0.015f)
                        {
                            // Two recovered targets disagree inside one oriented
                            // cell: keep it hidden instead of choosing one.
                            entry.w = (1 << 4) | (axisFamily + 1);
                            _paperCorrectionKeyEntries[slot] = entry;
                        }
                        inserted = true;
                        break;
                    }
                }
                if (!inserted) return false;
            }
            return true;
        }

        private static uint PaperCorrectionHash(int x, int y, int z,
            int axisFamily)
        {
            unchecked
            {
                uint hash = (uint)x * 73856093u;
                hash ^= (uint)y * 19349663u;
                hash ^= (uint)z * 83492791u;
                hash ^= (uint)axisFamily * 2654435761u;
                hash ^= hash >> 16;
                return hash;
            }
        }

        private void SetSafeShaderDefaults()
        {
            Shader.SetGlobalFloat(TriAvailableID, 0f);
            Shader.SetGlobalFloat(NormalFallbackID, 0f);
            Shader.SetGlobalFloat(PaperGridModeID, 0f);
            ApplyDisplayMode();
            Shader.SetGlobalFloat(NoFreezeTintID, 0f);
        }

        /// <summary>把当前显示形态写入 shader 全局量（QRS ApplyRenderMode 的线框部分）。</summary>
        private void ApplyDisplayMode()
        {
            Shader.SetGlobalFloat(WireframeID, wireframeMode ? 1f : 0f);
            Shader.SetGlobalFloat(WireThicknessID, wireThickness);
            Shader.SetGlobalFloat(MeshStrideID, meshDisplayStride);
            Shader.SetGlobalFloat(GridSpacingID, meshGridSpacing);
            Shader.SetGlobalFloat(ConfidenceVizID, confidenceViz ? 1f : 0f);
            Shader.SetGlobalFloat(GeometryTruthViewID, geometryTruthView ? 1f : 0f);
            // Keep the unverified display quarantine de-authorized even when a
            // stale replay session still owns correction candidates.
            Shader.SetGlobalInt(PaperCorrectionHashMaskID, -1);
            Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);
        }
    }
}

using System;
using System.Collections.Generic;
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
        [SerializeField, Tooltip("TSDF 融合频率=GPU 最大单项负载。30Hz 实机被 ASW 压到 24fps  quarter 档（帧24→融合每帧到期→节拍饿死）；20Hz 泄压保帧率，扫描质量由运动闸兜底。 (scene 20)")]
        private float integrationHz = 20f;
        [SerializeField, Range(1f, 15f), Tooltip("完整体积网格提取频率。融合保持高频，网格沿用上次结果直到下一次提取。08-18 帧率手术后 8→12（HERA 节拍 16→24/s），帧率跌破 50 退回 8。")]
        private float meshExtractionHz = 12f;

        [Header("渲染")]
        [SerializeField, Tooltip("主显示形态：开=线框（QRS Wireframe，重心坐标边缘检测）；关=顶点色实体（QRS Vertex）")]
        private bool wireframeMode = true;
        [SerializeField, Range(0.2f, 5f), Tooltip("线框模式的线条粗细倍率（1.0 对齐原 SC 工程细线观感，可按需调）")]
        private float wireThickness = 1.0f;
        [SerializeField, Range(1, 6), Tooltip("条带抽稀（顶点侧按体素格丢三角形，任一轴对齐即保留）。08-19 实机判定观感碎、做不出 Meta 粗网，已让世界格线画法取代，默认 1=关闭，仅留作帧率应急杠杆")]
        private int meshDisplayStride = 1;
        [SerializeField, Range(0.1f, 1.0f), Tooltip("世界格线网眼间距（米）：片元级在网格表面直接画经纬线，间距=网眼，观感对标 Meta 系统网格。0.3=30cm 网眼（推荐）；嫌密调大，嫌疏调小")]
        private float meshGridSpacing = 0.3f;
        [SerializeField, Tooltip("置信度通道 v1 可视化（诊断开关，默认关）：开=按体素分歧 EMA 给网格着色——高置信=原色 / 中=黄 / 低=蓝紫（几何在打架）/ 无数据=灰。只读着色，不碰任何生产逻辑")]
        private bool confidenceViz = false;

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

        [Header("实时轨（看哪出哪）")]
        [SerializeField, Tooltip("实时轨：视线落点周边未冻块即时出网（红绿粗页，不定稿不建家族省 8 倍子页负载），冻结后由定稿轨原子接管。关=纯定稿轨（旧行为）。 (default true)")]
        private bool enableLiveTrack = true;
        [SerializeField, Min(0.1f), Tooltip("实时轨巡视间隔（秒）。 (default 0.15，帧率手术后 08-18 从 0.25 放松)")]
        private float liveTrackSweepSeconds = 0.15f;
        [SerializeField, Min(0.2f), Tooltip("同一块实时重提节流（秒）：实时页缺肉后长肉的刷新节奏。 (default 0.5，帧率手术后 08-18 从 1 放松)")]
        private float liveTrackBlockCooldownSeconds = 0.5f;
        [SerializeField, Range(1, 20), Tooltip("实时轨全局峰值（页/秒）硬顶：Quest 帧率保护闸，超限本巡视直接收工。 (default 10，帧率手术后 08-18 从 5 放松；帧率跌破 50 退回 5)")]
        private int liveTrackMaxPagesPerSecond = 10;
        [SerializeField, Min(0), Tooltip("实时轨块内容下限：普查可出网体素（≥minMeshWeight 0.08，T3 从长熟 0.15 降档——0.08~0.15 带可出网不该被当空块）低于此数的块不排。普查按 1s 窗更新，全新区域首次出网最多延迟一个窗。 (default 16)")]
        private int liveTrackMinSurfaceVoxels = 16;

        [Header("增量精修（两段合一）")]
        [SerializeField, Tooltip("成熟冻结块就地精修上屏：采集段不出粗网，冻哪块出哪块的 HERA 红绿网格；点阵默认隐藏（X 呼出当判官）。" +
                 "关=旧两段式（采集只点阵，A 冻结才出网格）。需同时开启 HERA 分层回放。 (default true)")]
        private bool enableIncrementalHeraRefine = true;
        [SerializeField, Min(1f), Tooltip("解冻后旧网格页保留宽限（秒）：宽限内复冻则原子换新、画面不闪空；超期未复冻（家具真搬走等）才撤页露洞。" +
                 "解冻源多为手/身体短暂遮挡的穿越票，没有宽限会冻-解-冻振荡=整块闪烁。 (default 8)")]
        private float frozenPageInvalidateGraceSeconds = 8f;
        [SerializeField, Min(0f), Tooltip("复冻重提冷却（秒）：振荡块在冷却内复冻不立刻重提，到点由维护时钟补提（最终一致）。" +
                 "首冻/被撤页过的块不受限。防冻-解振荡的重提洪流灌满提取队列、饿死新块出网（实机：后期解升温+新页不再出现）。 (default 15)")]
        private float frozenPageRequeueCooldownSeconds = 15f;
        [SerializeField, Min(1), Tooltip("复冻兜底：宽限超期时解冻次数≥此值的振荡惯犯块不撤页、强制冻回静态几何求安分；" +
                 "真变化由穿越票走解冻棘轮硬闯（按不死，只慢一两窗）。初犯块照走撤页快路径（家具真搬走不受拖累）。" +
                 "阻尼封顶原则：这是调度层最后一个阻尼器，之后的'不安分'只准去数据层治。 (default 2)")]
        private int frozenRefreezeMinThaws = 2;

        [Header("平面拍平（B1 影子，只读验证）")]
        [SerializeField, Tooltip("B1 影子平面拟合：对视线落点块的当帧高质量观测做 PCA 平面拟合（只读不写 TSDF），" +
                 "HUD 拍行输出残差（拟合优度）/存量偏移（拍平幅度预估）/点数/主轴。验证'观测共识面干净+存量偏离'假设，成立才开 B2 写入。 (default true)")]
        private bool enablePlaneFitShadow = true;
        [SerializeField] private ComputeShader planeFitShadowCompute;

        [Header("平面拍平（B2 写入，振荡断根）")]
        [SerializeField, Tooltip("B2 逐像素泼溅拍平：观测面附近'持续矛盾'的体素 sd 直接改写为观测值" +
                 "——只写 sd 不动 weight（冻块负 weight 保留=不解冻修冻块几何），穿越票失根=振荡断根。" +
                 "持久闸=置信度分歧 EMA：瞬态手/行人进冻块不会被烙进几何；压完观测一致 EMA 衰减自然停手。 (default true)")]
        private bool enablePlaneFlatten = true;
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
        // 复冻兜底账：宽限超期被强制冻回的振荡惯犯块次数（HUD"回N"）。
        private int _forceRefreezeEvents;
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
        // 解冻块的宽限撤页表：块号 → 到期时刻（Time.time）。复冻即取消。
        private readonly Dictionary<int, float> _pendingPageInvalidate = new Dictionary<int, float>();
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
        // 内容闸账（T1a）：块号 → 上次排队时的全局融合 epoch。脏块 epoch（64³，
        // MarkDirtyChunk 只在几何级变化时记账）超过它=该块真变过才重提——帧级新鲜，
        // 取代旧"普查长熟数没变"判据（2s 相位=出网脉冲与"看着不涨"的主谋）。
        private readonly Dictionary<int, uint> _liveQueuedEpoch = new Dictionary<int, uint>();
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
        private float _heraFreezeStartedAt = -1f;
        private float _heraLastProgressAt = -1f;
        private string _heraLastProgressSignature = "";

        public bool IsChunkAbExperimentEnabled => enableFrozenChunkAbExperiment;
        public bool IsChunkAbFrozen => _chunkAbFrozen;
        public int ActiveChunkAbSize => ChunkAbSizes[Mathf.Clamp(_chunkAbGearIndex, 0, ChunkAbSizes.Length - 1)];

        /// <summary>当前是否线框显示。</summary>
        public bool IsWireframe => wireframeMode;

        /// <summary>在线框 / 顶点色实体之间切换（QRS SetRenderMode 的二态精简版）。</summary>
        public void ToggleWireframe()
        {
            wireframeMode = !wireframeMode;
            ApplyDisplayMode();
            NotifyInput(wireframeMode ? "切到线框" : "切到实体");
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
        /// 帧率二分总闸（右摇杆直接按下）：切"当前真正在画网格的那条路径"。
        /// A/B 实验旗下开机即 PrepareForChunkAbAcquisition，renderProductionMesh 恒
        /// false、满屏网全来自增量 HERA——所以扫描中必须切增量 HERA，切旧字段是空转。
        /// 只藏绘制；融合/提取/精修/记账后台照跑。
        /// </summary>
        public void ToggleMeshDisplay()
        {
            if (_meshExtractor == null) return;
            bool visible = _meshExtractor.HasIncrementalHera
                ? _meshExtractor.ToggleIncrementalHeraVisible()
                : _meshExtractor.ToggleProductionMeshVisible();
            NotifyInput(visible ? "网格显示：开" : "网格显示：关（后台继续）");
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

        [Header("日志")]
        [SerializeField] private LogLevel logLevel = LogLevel.Info;

        private DepthCapture _depthCapture;
        private VolumeIntegrator _volumeIntegrator;
        private MeshExtractor _meshExtractor;
        private PassthroughCameraProvider _cameraProvider;

        [Header("Minimal Status Badge")]
        [SerializeField] private bool showStatusBadge = true;
        private UnityEngine.UI.Text _statusBadgeText;

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
        private float _lastScannerLog;
        private int _integrateCount;

        private float IntegrationInterval => 1f / Mathf.Max(1f, integrationHz);
        private float MeshInterval => 1f / Mathf.Max(1f, meshExtractionHz);

        // ─────────────────────────────────────────────────────────────
        //  生命周期
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            Instance = this;
            // Runtime lock: serialized scene values cannot accidentally revive
            // the in-headset HUD, diagnostic ROI frame, or depth point cloud.
            showDebugHud = false;
            showDiagnosticRoiFrame = false;
            showDepthPointCloud = false;
            Logger.Level = logLevel;
            _depthCapture = GetComponent<DepthCapture>();
            _volumeIntegrator = GetComponent<VolumeIntegrator>();
            _meshExtractor = GetComponent<MeshExtractor>();
            _cameraProvider = GetComponent<PassthroughCameraProvider>();
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
            if (showDepthPointCloud) gameObject.AddComponent<DepthPointCloudOverlay>();
            StartCoroutine(ConfigureCameraForPassthrough());
            if (showStatusBadge)
                StartCoroutine(CreateStatusBadgeWhenCameraReady());
            if (showDebugHud)
                StartCoroutine(CreateHudWhenCameraReady());
            Application.logMessageReceived += OnLogMessage;
            Logger.Info(enableFrozenChunkAbExperiment
                ? (enableHeraHierarchicalReplay
                    ? (enableIncrementalHeraRefine
                        ? "增量精修就绪 — 扳机采集，成熟块自动定稿上屏，X 点阵判官，A 冻结全场回放"
                        : "HERA 就绪 — 扳机采集，A 冻结并自动 32→16，B 导出并清派生网格")
                    : "切块 A/B 就绪 — 扳机采集，A 冻结，Y 换 64/32/16，B 导出并清当前档")
                : "QRS 独立链就绪 — 右手柄扳机开始扫描，A 暂停，B 停止清空");
        }

        private System.Collections.IEnumerator CreateStatusBadgeWhenCameraReady()
        {
            while (Camera.main == null)
                yield return null;

            Font font = null;
            try { font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } catch { }
            if (font == null)
            {
                try { font = Resources.GetBuiltinResource<Font>("Arial.ttf"); } catch { }
            }

            var root = new GameObject("[QRS] Minimal Status Badge");
            root.transform.SetParent(Camera.main.transform, false);
            root.transform.localPosition = new Vector3(0f, -0.24f, 0.9f);
            root.transform.localRotation = Quaternion.identity;
            root.transform.localScale = Vector3.one * 0.00048f;

            var canvas = root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 32767;
            var rootRect = root.GetComponent<RectTransform>();
            rootRect.sizeDelta = new Vector2(900f, 330f);

            Material badgeMaterial = null;
            var badgeShader = Resources.Load<Shader>("HUDAlwaysOnTop");
            if (badgeShader != null)
                badgeMaterial = new Material(badgeShader);

            var bg = new GameObject("Bg");
            bg.transform.SetParent(root.transform, false);
            var image = bg.AddComponent<UnityEngine.UI.Image>();
            image.color = new Color(0f, 0f, 0f, 0.68f);
            if (badgeMaterial != null) image.material = badgeMaterial;
            var bgRect = bg.GetComponent<RectTransform>();
            bgRect.anchorMin = Vector2.zero;
            bgRect.anchorMax = Vector2.one;
            bgRect.offsetMin = Vector2.zero;
            bgRect.offsetMax = Vector2.zero;

            var textObject = new GameObject("Text");
            textObject.transform.SetParent(root.transform, false);
            _statusBadgeText = textObject.AddComponent<UnityEngine.UI.Text>();
            _statusBadgeText.font = font;
            _statusBadgeText.fontSize = 36;
            _statusBadgeText.alignment = TextAnchor.MiddleLeft;
            _statusBadgeText.horizontalOverflow = HorizontalWrapMode.Wrap;
            _statusBadgeText.verticalOverflow = VerticalWrapMode.Overflow;
            if (badgeMaterial != null) _statusBadgeText.material = badgeMaterial;
            var outline = textObject.AddComponent<UnityEngine.UI.Outline>();
            outline.effectColor = new Color(0f, 0f, 0f, 0.95f);
            outline.effectDistance = new Vector2(2f, -2f);
            var textRect = textObject.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(28f, 20f);
            textRect.offsetMax = new Vector2(-28f, -20f);

            RefreshStatusBadge();
        }

        private void RefreshStatusBadge()
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
                    string frozenTail = _frozenBlockUnfreezeEvents > 0
                        ? $" · 冻{_frozenBlocks.Count}解{_frozenBlockUnfreezeEvents}"
                        : (_frozenBlocks.Count > 0 ? $" · 冻{_frozenBlocks.Count}" : "");
                    if (_liveTrackQueuedTotal > 0)
                        frozenTail += $"实{_liveTrackQueuedTotal}";
                    if (_holeRequeueEvents > 0)
                        frozenTail += $"补{_holeRequeueEvents}";
                    // 复冻兜底活度：回N=振荡惯犯块被强制冻回次数。回N 涨+解/冻比降=兜底在灭火；
                    // 回N 涨但折角仍闪=该块的穿越票在棘轮下仍能硬闯，说明是真变化信号而非振荡。
                    if (_forceRefreezeEvents > 0)
                        frozenTail += $"回{_forceRefreezeEvents}";
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
                    if (_pendingPageInvalidate.Count > 0)
                        frozenTail += $"缓{_pendingPageInvalidate.Count}";
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
                        ? $" · 活[排{_liveGateQueued}冻{_liveGateFrozen}空{_liveGateEmpty}冷{_liveGateCool}内{_liveGateContent}途{_liveGateFlight}速{_liveGateRate}]"
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
                        ? $"精修 {_meshExtractor.IncrementalHeraPagesCommitted} 页 · 视线:{GazeBlockStatus()} · X 点阵 · A 冻结{liveGate}{confTail}"
                        : "黄=待成网 绿=已可出网 · A 冻结";
                    // 计时账第三行：拍=提取实际节拍(/s) 落=入队→落地 回=派发→回读
                    // 融/提=各自 CPU 耗时(ms)。判读：拍远低于16=被融合帧挤占；
                    // 回高=GPU 回读瓶颈；落高回低=排队/派发堵；融+提>帧预算=CPU 侧顶满。
                    // 第四行：四个性能二分热键（右摇杆上/左/下/右）+网格显示开关（右摇杆直接按下）
                    // 当前状态，防"以为切了其实没切"。显关=只藏绘制，融合/提取/精修后台照跑（帧率二分用）。
                    string filterLabel = _depthCapture == null ? "—"
                        : (_depthCapture.BilateralEnabled ? (_depthCapture.EdgeCleanEnabled ? "开" : "双")
                        : (_depthCapture.EdgeCleanEnabled ? "半" : "关"));
                    string frozenLabel = !enableFrozenBlockSupervisor ? "关"
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
                    string toggleLine = $"闸[实{(enableLiveTrack ? "开" : "关")} 冻{frozenLabel} " +
                                        $"融{integrationHz:0} 滤{filterLabel} " +
                                        $"显{(_meshExtractor != null && _meshExtractor.IsAnyMeshVisible ? "开" : "关")}" +
                                        $"皮{(_meshExtractor == null || !_meshExtractor.HasCoarseSkin ? "无" : (_meshExtractor.IsCoarseSkinVisible ? "开" : "关"))}" +
                                        $"{(meshDisplayStride > 1 ? $" 网{meshDisplayStride}" : "")}]{handLabel}{temporalLabel}";
                    // 行为层读数钉行首固定位（防抖）：角/质定宽 + 探头判决紧随其后，
                    // 前缀长度恒定（覆盖%定宽 3 位），热账 frozenTail 怎么跳都不动这里。
                    string watch = $" 角{(_volumeIntegrator != null ? _volumeIntegrator.SmoothedAngularSpeed : 0f),3:F0}°/s" +
                                   $" 质{(_volumeIntegrator != null ? _volumeIntegrator.MotionQuality : 1f):F2}" +
                                   (_probeHud.Length > 0 ? $" {_probeHud}" : "");
                    if (_meshExtractor != null && _meshExtractor.HasIncrementalHera)
                    {
                        string timingLine = $"帧{1f / Mathf.Max(0.001f, Time.smoothDeltaTime):0} 拍{_heraTicksPerSec:0}/s 落{_meshExtractor.IncrementalHeraAvgQueueToCommitMs:0}ms " +
                                            $"回{_meshExtractor.IncrementalHeraAvgDispatchToCallbackMs:0}ms " +
                                            $"融{_emaIntegrateMs:0.0}提{_emaHeraTickMs:0.0}ms " +
                                            $"温{OVRPlugin.batteryTemperature:0}°";
                        _statusBadgeText.text = $"● 采集中·炮0821 · 覆盖{coverage,3:0}%{watch}{frozenTail}\n{secondLine}\n{timingLine}\n{toggleLine}";
                    }
                    else
                        _statusBadgeText.text = $"● 采集中·炮0821 · 覆盖{coverage,3:0}%{watch}{frozenTail}\n{secondLine}\n{toggleLine}";
                    return;
                }

                _statusBadgeText.color = new Color(0.75f, 0.8f, 0.85f, 1f);
                _statusBadgeText.text = HasStarted
                    ? "Ⅱ 采集已暂停 · A 冻结 / 扳机继续"
                    : "○ 待采集 · 扳机开始";
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
        [SerializeField] private bool showDebugHud = false;
        [SerializeField, Tooltip("在面板右上角开一个当前深度实时预览小窗（青=近 绿=中 红=远 暗=无效）。" +
            "用途：盯着幽灵网格时看深度画面里那个斑块还在不在——在=深度自洽幻觉（Meta侧时序锁定）；转头后斑块从预览消失=深度刷新")]
        private bool showDepthPreview = true;
        [SerializeField, Tooltip("世界空间实时深度点云叠加层：当前深度以 3D 点云叠在网格上同屏对照。" +
            "幽灵位置有点云覆盖=深度自洽幻觉；空空如也却有网格=矛盾在我们侧")]
        private bool showDepthPointCloud = true;
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
            var hudShader = Resources.Load<Shader>("HUDAlwaysOnTop");
            if (hudShader != null) hudMat = new Material(hudShader);

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

            // 半透明黑底，保证在透视画面上可读
            var bgGo = new GameObject("Bg");
            bgGo.transform.SetParent(canvasGo.transform, false);
            bgGo.transform.SetAsFirstSibling();
            var img = bgGo.AddComponent<UnityEngine.UI.Image>();
            img.color = new Color(0f, 0f, 0f, 0.55f);
            if (hudMat != null) img.material = hudMat;
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
                $"摇杆按=线框  Y=生产A/候选B" +
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
        }

        private void OnDisable()
        {
            _coverageOverlay?.SetAcquiring(false);
            if (IsScanning) PauseScanning();
        }

        private void Update()
        {
            UpdateHud();

            // 置信度可视化开关支持运行时改值（编辑器内拖勾即生效，重发全局量）。
            if (confidenceViz != _confidenceVizApplied)
            {
                _confidenceVizApplied = confidenceViz;
                ApplyDisplayMode();
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
            TickFrozenBlockSupervisor(t);
            TickPendingPageInvalidates(t);
            TickDeferredPageRequeues(t);
            TickLiveTrack(t);
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
            bool meshDue = t - _lastMeshTime >= MeshInterval;

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
                bool heraDue = t - _lastMeshTime >= heraInterval;
                bool tickStarved = heraDue && t - _lastMeshTime >= heraInterval * 3f &&
                                   _meshExtractor.HasIncrementalHera;
                bool integrateNow = integrationDue && !tickStarved;
                if (integrateNow)
                {
                    _lastIntegrationTime = t;
                    ProvideColorFrame();
                    _depthCapture?.PreprocessLatestFrame();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    _volumeIntegrator.Integrate();
                    sw.Stop();
                    float iMs = (float)sw.Elapsed.TotalMilliseconds;
                    _emaIntegrateMs = _emaIntegrateMs < 0f ? iMs : Mathf.Lerp(_emaIntegrateMs, iMs, 0.25f);
                    Integrated?.Invoke();
                    _integrateCount++;
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

            // A full 256^3 extraction is substantially heavier than one fusion
            // pass on Quest. Never stack both on the same frame during normal
            // operation. If frame rate is already below target, reserve an
            // extraction-only frame after 1.5 mesh intervals so the visible
            // mesh keeps advancing instead of starving behind integration.
            bool meshStarved = meshDue &&
                               t - _lastMeshTime >= MeshInterval * 1.5f;
            bool integrateThisFrame = integrationDue && !meshStarved;

            if (integrateThisFrame)
            {
                _lastIntegrationTime = t;

                ProvideColorFrame();
                _depthCapture?.PreprocessLatestFrame();
                _volumeIntegrator.Integrate();
                Integrated?.Invoke();
                _integrateCount++;
            }

            if (meshDue && !integrateThisFrame)
            {
                _lastMeshTime = t;
                _meshExtractor.Extract();
                MeshExtracted?.Invoke();
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Logger.Verbose($"扫描中: 融合次数={_integrateCount}, 深度可用={DepthCapture.DepthAvailable}");
            }
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
            if (!enableFrozenBlockSupervisor) return;
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
                _hotBlocksPrevWindow.Clear();
                _hotBlocksPrevWindow.UnionWith(hotNow);
                _hotHoleBlocksPrevWindow.Clear();
                _hotHoleBlocksPrevWindow.UnionWith(hotHoleNow);
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
                    // 年审降级不吃棘轮（不是票解，是资格门收回冻结资格）。
                    if (_demotedThisBatch.Remove(b)) { demotedCount++; continue; }
                    // 棘轮记账：本块解冻次数+1，下窗起票阈×倍率。
                    _thawCounts[b] = (_thawCounts.TryGetValue(b, out int tc) ? tc : 0) + 1;
                }
                // T2：全量冻结位图同步本批裁决并下发 GPU（补洞票的块冻结态判据）。
                foreach (int b in setBlocks) _frozenChunkBitsArr[b >> 5] |= 1u << (b & 31);
                foreach (int b in clearBlocks) _frozenChunkBitsArr[b >> 5] &= ~(1u << (b & 31));
                _volumeIntegrator.SetFrozenChunkBits(_frozenChunkBitsArr);
                // 增量精修挂接：新冻块排队精修上屏；解冻块撤页（重冻后自动重建）。
                SyncIncrementalHeraBlocks(setBlocks, clearBlocks);
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
            _forceRefreezeEvents = 0;
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
            _pendingPageInvalidate.Clear();
            _lastPageQueueTime.Clear();
            _deferredPageRequeue.Clear();
            _thawCounts.Clear();
            _liveTrackLastSweep = -1f;
            _livePageQueueTime.Clear();
            _liveQueuedEpoch.Clear();
            _meshablePrevSurface = null;
            _firstMatureTime.Clear();
            _liveRateWindowStart = -1f;
            _liveRateWindowCount = 0;
            _liveTrackQueuedTotal = 0;
            _liveGateQueued = _liveGateFrozen = _liveGateEmpty = _liveGateCool = 0;
            _liveGateContent = _liveGateFlight = _liveGateRate = 0;
            _meshExtractor?.ResetIncrementalHeraState();
        }

        // ── 增量精修桥接（冻结调度器 ↔ HERA 增量模式）──

        private void SyncIncrementalHeraBlocks(List<int> setBlocks, List<int> clearBlocks)
        {
            if (_meshExtractor == null || !_meshExtractor.HasIncrementalHera) return;
            foreach (int b in setBlocks)
            {
                // 复冻：取消宽限撤页。首冻/被撤页过的块（无排队记录）立刻排队；
                // 其余按冷却节流——振荡块几何几乎没变，旧页继续显示即可，
                // 立刻重提只会灌满队列饿死新块；冷却到点由维护时钟补提。
                _pendingPageInvalidate.Remove(b);
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
                // 解冻不立刻撤页：旧页=上一版定稿，继续显示；宽限内复冻→原子替换。
                // 只有超期未复冻（家具真搬走等真变化）才由 TickPendingPageInvalidates 撤页露洞。
                _pendingPageInvalidate[b] = Time.time + frozenPageInvalidateGraceSeconds;
            }
        }

        /// <summary>实际排队提取一页，并记账（冷却/补提两表）。</summary>
        private void QueueIncrementalPage(int b)
        {
            _lastPageQueueTime[b] = Time.time;
            _deferredPageRequeue.Remove(b);
            _meshExtractor.IncrementalQueueParentBlock(FrozenBlockCoord3(b));
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

        /// <summary>宽限撤页到期处理：解冻块超过宽限仍未复冻，旧页内容已不可信，撤下露真洞。</summary>
        private void TickPendingPageInvalidates(float now)
        {
            if (_pendingPageInvalidate.Count == 0) return;
            if (_meshExtractor == null || !_meshExtractor.HasIncrementalHera)
            {
                _pendingPageInvalidate.Clear();
                return;
            }
            List<int> expired = null;
            foreach (KeyValuePair<int, float> kv in _pendingPageInvalidate)
            {
                if (now >= kv.Value) (expired ??= new List<int>()).Add(kv.Key);
            }
            if (expired == null) return;
            List<int> refrozen = null;
            List<int> invalidated = null;
            foreach (int b in expired)
            {
                _pendingPageInvalidate.Remove(b);
                // 复冻兜底（阻尼封顶，调度层最后一道阻尼器）：振荡惯犯块（累计解冻≥阈值次）
                // 超期未复冻 = 冻-解振荡不收敛（系统性偏差纠不完，折角案例实锤）。
                // 撤页只会让它悬在未冻态被实时轨反复重提（"整体换掉"观感引擎），此路
                // 无自限。改为强制冻回静态几何求安分；真变化由穿越票走解冻棘轮硬闯
                // （按不死，只慢一两窗）。初犯块照走下方撤页快路径——家具真搬走
                // 不该被兜底拖累。
                bool habitual = _thawCounts.TryGetValue(b, out int thaws) && thaws >= frozenRefreezeMinThaws;
                if (habitual && !_frozenBlocks.Contains(b) &&
                    _volumeIntegrator != null && _volumeIntegrator.FrozenBlockReady)
                {
                    EnsureFreezeMaskArrays();
                    Array.Clear(_freezeSetMask, 0, _freezeSetMask.Length);
                    Array.Clear(_freezeClearMask, 0, _freezeClearMask.Length);
                    _freezeSetMask[b >> 5] |= 1u << (b & 31);
                    _volumeIntegrator.ApplyChunkFreezeMasks(_freezeSetMask, _freezeClearMask);
                    _frozenBlocks.Add(b);
                    _frozenChunkBitsArr[b >> 5] |= 1u << (b & 31);
                    _volumeIntegrator.SetFrozenChunkBits(_frozenChunkBitsArr);
                    _forceRefreezeEvents++;
                    (refrozen ??= new List<int>()).Add(b);
                    continue;
                }
                // 撤页过的块视为"无页"：清掉冷却账，下次复冻必立刻重提（不吃冷却）。
                _lastPageQueueTime.Remove(b);
                _deferredPageRequeue.Remove(b);
                _meshExtractor.IncrementalInvalidateParentBlock(FrozenBlockCoord3(b));
                (invalidated ??= new List<int>()).Add(b);
            }
            if (invalidated != null)
                Logger.Info($"增量精修：{invalidated.Count} 块解冻超 {frozenPageInvalidateGraceSeconds:0}s 未复冻，撤下旧页：{FormatBlockCoords(invalidated)}");
            if (refrozen != null)
            {
                // 强制复冻后走正常复冻挂接：旧页继续显示，按冷却节流原子换新。
                SyncIncrementalHeraBlocks(refrozen, new List<int>(0));
                Logger.Info($"复冻兜底：{refrozen.Count} 个振荡惯犯块（解冻≥{frozenRefreezeMinThaws}次）超期未复冻，强制冻回求安分（真变化走解冻棘轮硬闯）：{FormatBlockCoords(refrozen)}");
                RefreshStatusBadge();
            }
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
            cs.SetInt(FitEyeID, 0); // 左眼片：逐眼交替清洗，旧一拍无妨（影子实验）
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
            cs.SetInt(FitEyeID, 0); // 左眼片（与 B1 同：旧一拍无妨）
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


        //  网格页（不建家族省 8 倍子页负载，边界三角定稿才补）。四道闸：
        //  冻结跳过（定稿轨接管）/ 在途跳过 / 块级 1s 节流（缺肉长肉节奏）
        //  / 全局 5 页/s 硬顶（帧率保护）。解冻块也会被实时轨拾起=修复
        //  过程直播（拾起即取消宽限撤页：旧页被实况页原子顶替）。
        // ─────────────────────────────────────────────────────────────

        private void TickLiveTrack(float now)
        {
            if (!enableLiveTrack) return;
            if (_meshExtractor == null || !_meshExtractor.HasIncrementalHera) return;
            if (_volumeIntegrator == null || !_volumeIntegrator.FrozenBlockReady) return;
            // 内容下限依赖普查账（2s 窗更新）；首轮普查未建档前不排实时页。
            if (_maturityPrevSurface == null) return;
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
            if (_liveRateWindowCount >= liveTrackMaxPagesPerSecond) return;

            // 落点锚定采样：每次巡视发一次中心深度回读（结果下次巡视生效，一拍延迟无感）。
            // 锚定值仅供 HUD @距离 回显，不再决定供给——供给改为下方的视线射线撒点。
            _depthCapture?.RequestCenterDepthSample();
            var grid = _volumeIntegrator.FrozenChunkCount;
            float blockWorld = _volumeIntegrator.VoxelSize *
                               Mathf.Max(1, _volumeIntegrator.VoxelCount.x / grid.x);

            // 视线射线撒候选（08-18 用户拍板：取消距离限制，"看到哪哪出网格"）。
            // 单落点制的死穴：锚定失败/出体积时整轨断供——深色床=深度稀疏无样本→
            // 退 1.5m 落在床上方空气；远处=钳 6m 出体积→同样落空。改为沿视线从
            // 0.5m 走到 6m（出体积即收工），步长 0.75 块，途经块去重、近者优先。
            // "空块不排"闸天然拦掉空气段，无需知道真实墙距；所有闸门不变。
            _liveCandidates.Clear();
            Vector3 rayOrigin = cam.transform.position;
            Vector3 rayDir = cam.transform.forward;
            float step = Mathf.Max(0.1f, blockWorld * 0.75f);
            bool enteredVolume = false;
            for (float t = 0.5f; t <= 6f && _liveCandidates.Count < 32; t += step)
            {
                if (!TryWorldToFrozenBlock(rayOrigin + rayDir * t, out int b, out _))
                {
                    if (enteredVolume) break; // 已进过体积再出界=射线穿出，收工
                    continue; // 还没进体积（贴边界站位），继续往前探
                }
                enteredVolume = true;
                if (_liveCandidates.Count > 0 && _liveCandidates[_liveCandidates.Count - 1].Block == b)
                    continue; // 连续采样打在同一块
                bool dup = false;
                for (int k = 0; k < _liveCandidates.Count; k++)
                    if (_liveCandidates[k].Block == b) { dup = true; break; }
                if (dup) continue;
                _liveCandidates.Add(new LiveCandidate { Block = b, Dist = t });
            }
            if (_liveCandidates.Count == 0) return; // 整条射线都在体积外

            // 活闸普查重计（只统计本轮走完 slab 循环的闸门分布；早退轮保留上轮读数）。
            _liveGateQueued = _liveGateFrozen = _liveGateEmpty = _liveGateCool = 0;
            _liveGateContent = _liveGateFlight = _liveGateRate = 0;

            // 内容闸信号源回读（T1a）：每巡视发一次脏块 epoch 回读（72B），
            // 结果下轮巡视生效——帧级新鲜，取代 2s 普查相位。
            _volumeIntegrator.RequestDirtyChunkEpochs();

            int queued = 0;
            for (int i = 0; i < _liveCandidates.Count && queued < 5; i++)
            {
                int b = _liveCandidates[i].Block;
                if (_frozenBlocks.Contains(b)) { _liveGateFrozen++; continue; } // 定稿轨的地盘
                if (_meshablePrevSurface[b] < liveTrackMinSurfaceVoxels) { _liveGateEmpty++; continue; } // 空块不排（T3：可出网档）
                if (_livePageQueueTime.TryGetValue(b, out float last) &&
                    now - last < liveTrackBlockCooldownSeconds) { _liveGateCool++; continue; } // 块级节流
                var coord = FrozenBlockCoord3(b);
                // 内容闸（T1a）：块所属 64³ 脏块 epoch 没超过上次排队时的全局 epoch
                // =几何级零变化（MarkDirtyChunk 只记新生/穿越/位移，权重纯积累不记账），
                // 重提纯属烧队列。快照未建档前放行（每块至少排一次）。
                var epochs = _volumeIntegrator.LatestDirtyChunkEpochs;
                var dc = _volumeIntegrator.DirtyChunkCount;
                if (epochs != null && epochs.Length == dc.x * dc.y * dc.z &&
                    _liveQueuedEpoch.TryGetValue(b, out uint qe))
                {
                    int di = coord.x / 2 + dc.x * (coord.y / 2 + dc.y * (coord.z / 2));
                    if (epochs[di] <= qe) { _liveGateContent++; continue; }
                }
                if (_meshExtractor.IncrementalParentBlockInFlight(coord)) { _liveGateFlight++; continue; } // 在途
                if (_liveRateWindowCount >= liveTrackMaxPagesPerSecond) { _liveGateRate++; break; } // 速率硬顶
                if (!_meshExtractor.IncrementalQueueLiveParentBlock(coord)) continue;
                _livePageQueueTime[b] = now;
                _liveQueuedEpoch[b] = _volumeIntegrator.DirtyEpoch;
                _liveRateWindowCount++;
                _liveTrackQueuedTotal++;
                _liveGateQueued++;
                queued++;
                // 解冻块被实时轨拾起=修复直播：旧页将被实况页原子顶替，
                // 宽限撤页失去意义（撤了反而闪空）。
                _pendingPageInvalidate.Remove(b);
            }
        }

        private struct LiveCandidate
        {
            public int Block;
            public float Dist;
        }
        private readonly List<LiveCandidate> _liveCandidates = new List<LiveCandidate>(32);

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
        /// 开始（或暂停后继续）深度融合与网格提取。
        /// 保留 QRS 原版的异步分段启动：先提交 GPU 大资源（TSDF ~150MB +
        /// Surface Nets ~480MB），跨帧 yield 两次后再启用透视相机与深度，
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

                // 阶段 3：相机 + 深度（此时启动安全）
                _cameraProvider?.StartCapture();
                _depthCapture.StartDepthCapture();

                bool resuming = HasStarted;
                if (!resuming && !enableFrozenChunkAbExperiment)
                    _meshExtractor.BeginLedgerSession();
                if (enableFrozenChunkAbExperiment)
                {
                    if (!resuming) _coverageOverlay?.ResetCoverage();
                    _coverageOverlay?.SetAcquiring(true);
                    // 增量精修：点阵默认隐藏（X 呼出当判官）；旧两段式保持默认可见。
                    bool incremental = enableIncrementalHeraRefine && enableHeraHierarchicalReplay;
                    _coverageOverlay?.SetMarkersVisible(!incremental);
                    if (incremental)
                        _meshExtractor.BeginIncrementalHera(abMaxChunksPerTick);
                }
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

            _cameraProvider?.StopCapture();
            _depthCapture.StopDepthCapture();

            Logger.Info("已暂停 — 扳机继续，B 保存累计账并清空");
            _hudStatus = "已暂停";
            RefreshStatusBadge();
            ScanStopped?.Invoke();
        }

        /// <summary>
        /// Freeze the shared TSDF and begin a read-only replay of the selected
        /// admission granularity.  Coverage tiles are hidden first, before a
        /// replay renderer is allowed to appear, so the two cannot overlap.
        /// </summary>
        public void FreezeChunkAbTsdf()
        {
            if (!enableFrozenChunkAbExperiment)
            {
                PauseScanning();
                return;
            }
            if (_chunkAbFrozen || !HasStarted) return;

            _coverageOverlay?.SetAcquiring(false);
            PauseScanning();
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
                ? "共享 TSDF 已冻结；开始 HERA 只读分层回放"
                : $"共享 TSDF 已冻结；开始 {ActiveChunkAbSize}³ 只读切块回放");
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
                    : $"账簿未闭环：还差 {_meshExtractor.FrozenHeraChildrenPending} 个子页、" +
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

            _volumeIntegrator.Clear();
            _volumeIntegrator.ResetSessionCounters();
            yield return null;
            if (_meshExtractor.IsInitialized)
                _meshExtractor.DisposeOnly(); // 下一次扳机分帧重建，避免 B 键同帧释放+重分配

            if (wasActive)
                Logger.Info($"已保存并清空 — 账簿={ledgerPath}；扳机开始全新扫描");
            _hudStatus = ledgerPath.Length > 0 ? "已保存并清空" : "已清空(账簿为空)";
            IsSaveAndClearInProgress = false;
            RefreshStatusBadge();
        }

        /// <summary>X：呼出/收起采集点阵（单帧真值判官对照层；数据账不受显示影响）。</summary>
        public void ToggleCoverageMarkers()
        {
            if (_coverageOverlay == null) return;
            bool visible = !_coverageOverlay.MarkersVisible;
            _coverageOverlay.SetMarkersVisible(visible);
            NotifyInput(visible ? "点阵判官：开" : "点阵判官：关");
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
        private static readonly int ConfidenceVizID = Shader.PropertyToID("_RSConfidenceViz");
        private bool _confidenceVizApplied;
        private float _lastConfidenceStatsTime = -10f;

        private void SetSafeShaderDefaults()
        {
            Shader.SetGlobalFloat(TriAvailableID, 0f);
            Shader.SetGlobalFloat(NormalFallbackID, 0f);
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
        }
    }
}

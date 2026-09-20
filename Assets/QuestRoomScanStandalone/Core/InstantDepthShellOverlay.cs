using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Unity.Mathematics;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 融合前“即时外壳”影子层：把最近数帧清洗深度各自连接成短寿命、视图局部的
    /// 三角膜。它只读 gsDepthTex / gsDepthNormalTex 和采集当刻矩阵，不写 TSDF、
    /// 不创建枪胶候选。通过独立视角复核的壳三角只发布低分辨率“存在证词”；
    /// 生产融合仍须用后续原始深度命中同一 provisional 体素，才能借证词转正。
    ///
    /// 每个小三角会同时检查三个端点。深度断裂过大时整片丢弃；接近断裂或掠射的
    /// 可疑片保留为黄/红边，便于直接观察“原料膜”从哪里开始失真。历史片很快退场，
    /// A 键只临时锁住一层世界空间壳，再按解除；不导出、不入历史，
    /// 生产扫描继续运行。
    /// </summary>
    public sealed class InstantDepthShellOverlay : MonoBehaviour
    {
        private const int ReviewCounterCount = 74; // 66..73: same-position freeze/support cross-audit
        [SerializeField, Range(64, 192), Tooltip("即时壳网格最大边长；只影响影子显示密度")]
        private int maxGridDim = 128;

        [SerializeField, Range(2, 6), Tooltip("同时保留的短寿命外壳层数")]
        private int liveLayerCount = 4;

        [SerializeField, Range(0.05f, 0.3f), Tooltip("即时壳采样间隔（秒）")]
        private float captureInterval = 0.12f;

        [SerializeField, Range(0.25f, 1.5f), Tooltip("未覆盖更新时的最长可见寿命（秒）")]
        private float liveLifetime = 0.65f;

        [SerializeField, Tooltip("与生产融合保持一致，读取右眼清洗深度")]
        private bool useRightEye = true;

        [Header("独立视角复核")]
        [SerializeField, Range(8, 24), Tooltip("自动保留的历史壳数量")]
        private int reviewHistoryCount = 16;

        [SerializeField, Range(0.2f, 0.8f), Tooltip("历史参照壳入库间隔（秒）")]
        private float reviewHistoryInterval = 0.35f;

        [SerializeField, Range(0.02f, 0.12f), Tooltip("相机位置至少分开这么多，才有资格成为独立视角候选")]
        private float minIndependentBaseline = 0.04f;

        [SerializeField, Range(0.15f, 1.0f), Tooltip("超过此距离的历史视角不用于即时邻域复核")]
        private float maxReferenceBaseline = 0.45f;

        [SerializeField, Range(10f, 70f), Tooltip("历史壳与当前朝向相差过大时，视为没有有效重叠")]
        private float maxReferenceForwardAngle = 40f;

        [SerializeField, Range(0.5f, 4f), Tooltip("同一世界点的两条观察射线至少分开该角度，才算独立证词")]
        private float minIndependentRayAngle = 1.25f;

        // 旧“历史探针 -> provisional 转正”生产授权已永久停权。
        // 历史壳仍可服务即时壳自身的只读观察，但不得发布或向融合暴露证词账。
        private static readonly bool PublishVerifiedWitnesses = false;

        [SerializeField, Range(1, 4), Tooltip("证词账单元边长（体素数）。默认 1×5cm，防止前景物与背后墙面共用一张存在票。")]
        private int witnessStride = 1;

        [SerializeField, Range(4, 64), Tooltip("证词在多少次即时壳采集后过期；防旧视角永久给后来几何背书。")]
        private int witnessMaxAgeCaptures = 24;

        [SerializeField, Range(0.002f, 0.02f), Tooltip("壳纸合流时的深度容差。纸与壳近乎共面时纸优先；壳明显更靠近相机时仍显示真实前景结构。")]
        private float compositeDepthToleranceMeters = 0.008f;

        [Header("接力验收（只读）")]
        [SerializeField, Range(0.005f, 0.05f), Tooltip("即时壳落到TSDF后，距零值小于该距离计为对齐；只影响HUD统计。")]
        private float relayTsdfToleranceMeters = 0.025f;

        [SerializeField, Range(0.005f, 0.06f), Tooltip("即时壳与同射线最近TSDF零交叉小于该距离计为对齐；只影响HUD统计。")]
        private float relayZeroToleranceMeters = 0.03f;

        [SerializeField, Range(0.01f, 0.08f), Tooltip("零交叉与已提交生产纸皮采样点小于该距离计为对齐；只影响HUD统计。")]
        private float relayPaperToleranceMeters = 0.04f;

        [SerializeField, Range(0.04f, 0.15f), Tooltip("在即时壳附近搜索已提交生产纸皮的最大半径；找到与对齐分开记账。")]
        private float relayPaperSearchMeters = 0.10f;

        [Header("接力自动验收（只读）")]
        [SerializeField, Range(4, 16), Tooltip("连续多少批约2Hz接力审计共同形成自动结论；避免单帧和单视角偶然值左右判断。")]
        private int relayAutoWindowBatches = 8;

        [SerializeField, Range(256, 8192), Tooltip("窗口内至少需要多少个卷内即时壳样本，未达到时只显示采样中。")]
        private int relayAutoMinSamples = 1024;

        [SerializeField, Range(0, 20), Tooltip("卷内即时壳样本中，空/弱/邻/其它不可读合计允许的最大百分比。只影响自动诊断结论。")]
        private int relayAutoMaxUnreadablePercent = 2;

        [SerializeField, Range(0, 30), Tooltip("TSDF可读样本中，深度残差超限样本允许的最大百分比。只影响自动诊断结论。")]
        private int relayAutoMaxMisalignedPercent = 5;

        [SerializeField, Range(0, 50), Tooltip("不可读样本中，持续至少8次审计仍未恢复者允许的最大百分比。只影响自动诊断结论。")]
        private int relayAutoMaxPersistentPercent = 10;

        private sealed class ShellLayer
        {
            public MeshRenderer Renderer;
            public Material Material;
            public RenderTexture Snapshot;
            public RenderTexture ReviewState;
            public RenderTexture ReviewResidualMm;
            public RenderTexture ConnectivityState;
            public RenderTexture PlaneConnectivityState;
            public ComputeBuffer ReviewCounters;
            public readonly uint[] ReviewCounterSeed = new uint[ReviewCounterCount];
            public uint AuditBatchToken;
            public bool CounterReadbackPending;
            public float CapturedAt;
            public bool Valid;
        }

        private sealed class ReviewReference
        {
            public RenderTexture Snapshot;
            public Matrix4x4 Proj;
            public Matrix4x4 ProjInv;
            public Matrix4x4 View;
            public Matrix4x4 ViewInv;
            public Vector3 CameraPosition;
            public Vector3 CameraForward;
            public float CapturedAt;
            public bool Valid;
        }

        private Mesh _mesh;
        private Material _snapshotPackMaterial;
        private ComputeShader _reviewCompute;
        private int _reviewKernel = -1;
        private int _connectivityKernel = -1;
        private int _publishWitnessKernel = -1;
        private int _clearPaperAuditKernel = -1;
        private int _rasterizePaperAuditKernel = -1;
        private int _relayAuditKernel = -1;
        private ComputeBuffer _verifiedWitnessEpochs;
        private ComputeBuffer _paperAuditCells;
        private ComputeBuffer _relayMaturityCells;
        private ComputeBuffer _paperAuditTriangles;
        private int _paperAuditTriangleCapacity;
        private int3 _paperAuditCellCount;
        private int _paperAuditStride;
        private int _paperAuditCellTotal;
        private int3 _witnessVoxelCount;
        private int3 _witnessCellCount;
        private float _witnessVoxelSize;
        private uint _witnessEpoch;
        private uint _relayAuditEpoch;
        private ShellLayer[] _liveLayers;
        private ShellLayer _frozenLayer;
        private bool _diagnosticFreezeActive;
        private InstantSeedPlanePreview _seedPreview;
        private bool _seedPreviewVisible;
        private bool _seedPreviewCapturePending;
        private bool _automaticSeedCaptureActive;
        private float _nextAutomaticSeedAttemptAt;
        private int _automaticSeedAttemptCount;
        private ReviewReference[] _reviewHistory;
        private int _nextReviewHistory;
        private float _nextReviewHistoryAt;
        private bool _built;
        private bool _visible;
        private bool _acquiring;
        private int _nextLayer;
        private float _nextCaptureAt;
        private float _lastCaptureAt = -1f;
        private int _sourceWidth;
        private int _sourceHeight;
        private int _pixelStep;
        private int _reviewGridWidth;
        private int _reviewGridHeight;
        private bool _reviewReadbackPending;
        private bool _paperOccupancyExportPending;
        private int _reviewEpoch;
        private float _nextReviewReadbackAt;
        private uint _reviewBatchSequence;
        private uint _auditLedgerBatchToken;
        // HUD 账本错码：1批次令牌，2旧分区和，4救回超深拒，
        // 8新成面和，16深拒去向和，32空读回。
        private int _auditLedgerFaultMask;
        private bool _auditLedgerHasSnapshot;
        private float _lastRelayResultAt = -1f;
        private int _reviewMissingPercent;
        private int _reviewNoWitnessPercent;
        private int _reviewSameViewPercent;
        private int _reviewIndependentPercent;
        private int _reviewSameDepthConflictPercent;
        private int _reviewSameNormalConflictPercent;
        private int _reviewIndependentDepthConflictPercent;
        private int _reviewIndependentNormalConflictPercent;
        private int _reviewIndependentDepthResidualMeanMm;
        private int _reviewIndependentDepthResidualMaxMm;
        private int _connectivityEmittedPercent;
        private int _connectivityMissingPercent;
        private int _connectivityDepthRejectPercent;
        private int _connectivityNormalRejectPercent;
        private int _connectivityBothRejectPercent;
        private int _planeConnectivityAcceptedPercent;
        private int _planeConnectivityRescuedPercent;
        private int _planeConnectivityGuardedPercent;
        private int _planeConnectivityUndecidedPercent;
        private int _witnessPublishedPercent;
        private int _relayShellTsdfSamples;
        private int _relayShellTsdfReadablePercent;
        private int _relayShellTsdfAlignedPercent;
        private int _relayShellTsdfResidualMeanMm;
        private int _relayShellTsdfResidualMaxMm;
        private int _relayTsdfZeroSamples;
        private int _relayTsdfZeroFoundPercent;
        private int _relayTsdfZeroAlignedPercent;
        private int _relayTsdfZeroResidualMeanMm;
        private int _relayTsdfZeroResidualMaxMm;
        private int _relayZeroPaperSamples;
        private int _relayZeroPaperFoundPercent;
        private int _relayZeroPaperAlignedPercent;
        private int _relayZeroPaperResidualMeanMm;
        private int _relayZeroPaperResidualMaxMm;
        private int _relayUnreadOutsidePercent;
        private int _relayUnreadUnwrittenPercent;
        private int _relayUnreadLowWeightPercent;
        private int _relayUnreadNeighbourPercent;
        private int _relayUnreadOtherPercent;
        private int _relayBadObliquePercent;
        private int _relayBadDepthEdgePercent;
        private int _relayBadTemporalPosePercent;
        private int _relayBadMultiLayerPercent;
        private int _relayBadUnattributedPercent;
        private int _relayGoodObliquePercent;
        private int _relayGoodDepthEdgePercent;
        private int _relayGoodTemporalPosePercent;
        private int _relayGoodMultiLayerPercent;
        private ulong _relayMaturityTrackedTotal;
        private ulong _relayMaturityClosedTotal;
        private ulong _relayMaturityAuditSum;
        private ulong _relayMaturityViewChangedTotal;
        private int _relayMaturityStuckSamples;
        private int _relayCornerBadPercent;
        private int _relayCornerBadMixedPercent;
        private int _relayCornerGoodPercent;
        private int _relayCornerGoodMixedPercent;
        private int _relayTemporalBadSamePercent;
        private int _relayTemporalBadBaselinePercent;
        private int _relayTemporalGoodSamePercent;
        private int _relayTemporalGoodBaselinePercent;
        private int _relayFrameAgeMs;
        private int _relayFrameIntervalMs;
        private int _relayPoseDeltaMm;
        private int _relayPoseDeltaTenthsDeg;
        private uint _relayTimestampBacksteps;
        private uint _relayTimestampRepeats;
        private readonly uint[] _relayFreezeCounts = new uint[8];
        private readonly uint[] _relayFreezeTotals = new uint[8];

        private const int MaxRelayAutoWindowBatches = 16;

        private struct RelayAutoBatch
        {
            public uint ShellSamples;
            public uint Outside;
            public uint Unwritten;
            public uint LowWeight;
            public uint Neighbour;
            public uint Other;
            public uint Readable;
            public uint Aligned;
            public uint Closed;
            public uint Stuck;
        }

        private readonly RelayAutoBatch[] _relayAutoWindow =
            new RelayAutoBatch[MaxRelayAutoWindowBatches];
        private int _relayAutoWindowLimit;
        private int _relayAutoWindowCount;
        private int _relayAutoWindowCursor;
        private int _relayAutoGapPercent;
        private int _relayAutoMisalignedPercent;
        private int _relayAutoPersistentPercent;
        private int _relayAutoUnwrittenPercent;
        private int _relayAutoLowWeightPercent;
        private int _relayAutoNeighbourPercent;
        private int _relayAutoOtherPercent;
        private ulong _relayAutoSamples;
        private ulong _relayAutoClosed;
        private string _relayAutoVerdict = "采样中";
        private string _relayAutoCause = "待";

        // Exact point-to-triangle relay audit. Eighteen bits of the spatial
        // cell payload identify a triangle, so this diagnostic-only buffer is
        // capped at 2^18 entries (about 9 MiB at nine floats per triangle).
        private const int MinPaperAuditTriangleCapacity = 4096;
        private const int MaxPaperAuditTriangleCapacity = 1 << 18;

        // 0总 1有效 2缺 3无 4同 5独 6同深 7同法 8独深 9独法
        // 10独深残差毫米和 11独深最大残差毫米；
        // 12三角总 13成面 14点缺 15深拒 16法拒 17双拒；
        // 18共面影子成面 19救回 20跨面守住 21邻域不足待定；
        // 22/23批次令牌；24经验证证词三角 25证词越卷；
        // 26即时壳样本 27 TSDF可读 28 TSDF对齐 29/30 TSDF残差和/峰；
        // 31零交叉找到 32零交叉对齐 33/34零残差和/峰；
        // 35生产纸皮找到 36纸皮对齐 37/38纸皮残差和/峰；
        // 39体积外 40中心未写 41中心低权 42八邻域缺样 43其它不可读；
        // 44..48错位样本的斜视/边缘/时姿/多层/无归因；
        // 49..52对齐样本的斜视/边缘/时姿/多层基线；
        // 53..57覆盖成熟账；58..61转角/梯度分裂；62..65同视时序/独立基线拆分。
        private static readonly uint[] ZeroReviewCounters = new uint[ReviewCounterCount];

        private static readonly int TexSizeID = Shader.PropertyToID("gsDepthTexSize");
        private static readonly int EyeIndexID = Shader.PropertyToID("_EyeIndex");
        private static readonly int SnapshotID = Shader.PropertyToID("_DepthNormalSnapshot");
        private static readonly int ProjID = Shader.PropertyToID("_SnapshotProj");
        private static readonly int ProjInvID = Shader.PropertyToID("_SnapshotProjInv");
        private static readonly int ViewInvID = Shader.PropertyToID("_SnapshotViewInv");
        private static readonly int SnapshotTexSizeID = Shader.PropertyToID("_SnapshotTexSize");
        private static readonly int AgeID = Shader.PropertyToID("_Age01");
        private static readonly int FrozenID = Shader.PropertyToID("_FrozenSnapshot");
        private static readonly int ReviewTexID = Shader.PropertyToID("_ShellReviewTex");
        private static readonly int CurrentSnapshotID = Shader.PropertyToID("_CurrentSnapshot");
        private static readonly int ReferenceSnapshotID = Shader.PropertyToID("_ReferenceSnapshot");
        private static readonly int ReviewStateID = Shader.PropertyToID("_ReviewState");
        private static readonly int ReviewResidualMmID = Shader.PropertyToID("_ReviewResidualMm");
        private static readonly int ConnectivityStateID = Shader.PropertyToID("_ConnectivityState");
        private static readonly int PlaneConnectivityStateID = Shader.PropertyToID("_PlaneConnectivityState");
        private static readonly int ShowRejectedTrianglesID = Shader.PropertyToID("_ShowRejectedTriangles");
        private static readonly int CompositeWithProductionID = Shader.PropertyToID("_CompositeWithProduction");
        private static readonly int CompositeDepthBiasMetersID = Shader.PropertyToID("_CompositeDepthBiasMeters");
        private static readonly int ReviewCountersID = Shader.PropertyToID("_ReviewCounters");
        private static readonly int VerifiedWitnessEpochsID = Shader.PropertyToID("_VerifiedWitnessEpochs");
        private static readonly int TsdfVolumeID = Shader.PropertyToID("_TsdfVolume");
        private static readonly int TsdfVoxelCountID = Shader.PropertyToID("_TsdfVoxelCount");
        private static readonly int TsdfVoxelSizeID = Shader.PropertyToID("_TsdfVoxelSize");
        private static readonly int TsdfTruncationDistanceID = Shader.PropertyToID("_TsdfTruncationDistance");
        private static readonly int RelayTsdfToleranceMetersID = Shader.PropertyToID("_RelayTsdfToleranceMeters");
        private static readonly int RelayZeroToleranceMetersID = Shader.PropertyToID("_RelayZeroToleranceMeters");
        private static readonly int RelayMinTsdfWeightID = Shader.PropertyToID("_RelayMinTsdfWeight");
        private static readonly int RelayPaperToleranceMetersID = Shader.PropertyToID("_RelayPaperToleranceMeters");
        private static readonly int RelayPaperSearchMetersID = Shader.PropertyToID("_RelayPaperSearchMeters");
        private static readonly int PaperAuditCellsID = Shader.PropertyToID("_PaperAuditCells");
        private static readonly int RelayMaturityCellsID = Shader.PropertyToID("_RelayMaturityCells");
        private static readonly int RelayAuditEpochID = Shader.PropertyToID("_RelayAuditEpoch");
        private static readonly int RelayTimingRiskID = Shader.PropertyToID("_RelayTimingRisk");
        private static readonly int RelayPoseRiskID = Shader.PropertyToID("_RelayPoseRisk");
        private static readonly int PaperAuditCellCountID = Shader.PropertyToID("_PaperAuditCellCount");
        private static readonly int PaperAuditStrideID = Shader.PropertyToID("_PaperAuditStride");
        private static readonly int PaperAuditCellTotalID = Shader.PropertyToID("_PaperAuditCellTotal");
        private static readonly int PaperAuditClearOffsetID = Shader.PropertyToID("_PaperAuditClearOffset");
        internal static readonly int RelayPaperVerticesID = Shader.PropertyToID("_RelayPaperVertices");
        internal static readonly int RelayPaperIndicesID = Shader.PropertyToID("_RelayPaperIndices");
        internal static readonly int RelayPaperVertexCountID = Shader.PropertyToID("_RelayPaperVertexCount");
        internal static readonly int RelayPaperIndexCountID = Shader.PropertyToID("_RelayPaperIndexCount");
        internal static readonly int RelayPaperLocalToWorldID = Shader.PropertyToID("_RelayPaperLocalToWorld");
        private static readonly int RelayPaperTrianglesID = Shader.PropertyToID("_RelayPaperTriangles");
        internal static readonly int RelayPaperTriangleOffsetID = Shader.PropertyToID("_RelayPaperTriangleOffset");
        internal static readonly int RelayPaperTriangleCapacityID = Shader.PropertyToID("_RelayPaperTriangleCapacity");
        private static readonly int RelayPaperTriangleCountID = Shader.PropertyToID("_RelayPaperTriangleCount");
        private static readonly int CurrentProjID = Shader.PropertyToID("_CurrentProj");
        private static readonly int CurrentProjInvID = Shader.PropertyToID("_CurrentProjInv");
        private static readonly int CurrentViewInvID = Shader.PropertyToID("_CurrentViewInv");
        private static readonly int ReferenceProjID = Shader.PropertyToID("_ReferenceProj");
        private static readonly int ReferenceProjInvID = Shader.PropertyToID("_ReferenceProjInv");
        private static readonly int ReferenceViewID = Shader.PropertyToID("_ReferenceView");
        private static readonly int ReferenceViewInvID = Shader.PropertyToID("_ReferenceViewInv");
        private static readonly int SourceSizeID = Shader.PropertyToID("_SourceSize");
        private static readonly int ReviewGridSizeID = Shader.PropertyToID("_ReviewGridSize");
        private static readonly int PixelStepID = Shader.PropertyToID("_PixelStep");
        private static readonly int HasReferenceID = Shader.PropertyToID("_HasReference");
        private static readonly int ReferencePoseIndependentID = Shader.PropertyToID("_ReferencePoseIndependent");
        private static readonly int CurrentCameraWorldID = Shader.PropertyToID("_CurrentCameraWorld");
        private static readonly int ReferenceCameraWorldID = Shader.PropertyToID("_ReferenceCameraWorld");
        private static readonly int MinIndependentRayCosID = Shader.PropertyToID("_MinIndependentRayCos");
        private static readonly int WitnessVoxelCountID = Shader.PropertyToID("_WitnessVoxelCount");
        private static readonly int WitnessCellCountID = Shader.PropertyToID("_WitnessCellCount");
        private static readonly int WitnessStrideID = Shader.PropertyToID("_WitnessStride");
        private static readonly int WitnessVoxelSizeID = Shader.PropertyToID("_WitnessVoxelSize");
        private static readonly int WitnessEpochID = Shader.PropertyToID("_WitnessEpoch");

        public bool Visible => _visible;
        /// <summary>
        /// false=即时壳独占前景，便于查原料；true=即时壳叠在稳定纸皮之上，
        /// 只验证生产观感。此标志不改变任何深度或融合数据。
        /// </summary>
        public bool CompositeWithProduction { get; private set; }
        public bool HasFrozenSnapshot => _diagnosticFreezeActive &&
                                         _frozenLayer != null && _frozenLayer.Valid;
        public bool DiagnosticFreezeActive => HasFrozenSnapshot;
        public bool SeedPreviewVisible => _visible && _seedPreviewVisible;
        public string SeedPreviewStatus => _seedPreview != null
            ? _seedPreview.Status : "种子观察未就绪";
        public bool AutomaticSeedCaptureActive => _automaticSeedCaptureActive;
        public int AutomaticSeedAttemptCount => _automaticSeedAttemptCount;
        public int WitnessMaxAgeCaptures => Mathf.Max(1, witnessMaxAgeCaptures);

        /// <summary>
        /// 旧生产接口保留仅为避免破坏调用方 ABI；探针停权后恒定返回 false，
        /// 且不再向调用方暴露历史证词缓冲。
        /// </summary>
        public bool TryGetVerifiedWitnessLedger(out ComputeBuffer epochs, out int3 cellCount,
            out int stride, out uint epoch, out uint maxAge)
        {
            epochs = null;
            cellCount = new int3(1, 1, 1);
            stride = 1;
            epoch = 0u;
            maxAge = 0u;
            return false;
        }

        public string StatusFixed => RelayDiagnosticsFixed + "\n" + AttributionDiagnosticsFixed;

        public string RelaySamplingStatus
        {
            get
            {
                string state = !enabled ? "停用" : !_acquiring ? "暂停·保留" :
                    !DepthCapture.DepthAvailable ? "等待深度" : !_auditLedgerHasSnapshot ? "等待首批" :
                    Time.unscaledTime - _lastRelayResultAt > 2f ? "更新滞后" : "采样中";
                string age = _lastRelayResultAt < 0f ? "--" :
                    Mathf.Max(0f, Time.unscaledTime - _lastRelayResultAt).ToString("0.0");
                return $"审计[{state}] 回读距今{age}s（全档约2Hz）";
            }
        }

        public string RelayDiagnosticsFixed
        {
            get
            {
                int live = 0;
                if (_liveLayers != null)
                    for (int i = 0; i < _liveLayers.Length; i++)
                        if (_liveLayers[i].Valid) live++;
                ulong maturityMeanAudits = _relayMaturityClosedTotal > 0
                    ? _relayMaturityAuditSum / _relayMaturityClosedTotal
                    : 0;
                int maturityViewPercent = _relayMaturityClosedTotal > 0
                    ? Mathf.Clamp(Mathf.RoundToInt((float)_relayMaturityViewChangedTotal * 100f /
                                                   _relayMaturityClosedTotal), 0, 100)
                    : 0;
                return "[接力·整批采样，非局部缺口]\n" + RelaySamplingStatus + "\n" +
                       $"自动 壳→TSDF[{_relayAutoVerdict}] 窗{_relayAutoWindowCount:00}/{Mathf.Max(4, relayAutoWindowBatches):00} " +
                       $"样{HudCount(_relayAutoSamples)} 缺{_relayAutoGapPercent:000}% 错{_relayAutoMisalignedPercent:000}%\n" +
                       $"自动 因[{_relayAutoCause}] 空{_relayAutoUnwrittenPercent:000}% " +
                       $"弱{_relayAutoLowWeightPercent:000}% 邻{_relayAutoNeighbourPercent:000}% " +
                       $"余{_relayAutoOtherPercent:000}% 久{_relayAutoPersistentPercent:000}%\n" +
                       $"自动 补{HudCount(_relayAutoClosed)} 门缺≤{relayAutoMaxUnreadablePercent:00}%/" +
                       $"错≤{relayAutoMaxMisalignedPercent:00}%/久≤{relayAutoMaxPersistentPercent:00}%\n" +
                       $"接力 壳→TSDF 样{HudCount((ulong)_relayShellTsdfSamples)} " +
                       $"读{_relayShellTsdfReadablePercent:000}% 齐{_relayShellTsdfAlignedPercent:000}% " +
                       $"残{_relayShellTsdfResidualMeanMm:000}/{_relayShellTsdfResidualMaxMm:000}mm\n" +
                       $"接力 TSDF→零 样{HudCount((ulong)_relayTsdfZeroSamples)} " +
                       $"找{_relayTsdfZeroFoundPercent:000}% 齐{_relayTsdfZeroAlignedPercent:000}% " +
                       $"残{_relayTsdfZeroResidualMeanMm:000}/{_relayTsdfZeroResidualMaxMm:000}mm\n" +
                       $"接力 零→纸皮 样{HudCount((ulong)_relayZeroPaperSamples)} " +
                       $"找{_relayZeroPaperFoundPercent:000}% 齐{_relayZeroPaperAlignedPercent:000}% " +
                       $"残{_relayZeroPaperResidualMeanMm:000}/{_relayZeroPaperResidualMaxMm:000}mm\n" +
                       "[覆盖与冻结]\n" +
                       $"不可读 界{_relayUnreadOutsidePercent:000}% 空{_relayUnreadUnwrittenPercent:000}% " +
                       $"弱{_relayUnreadLowWeightPercent:000}% 邻{_relayUnreadNeighbourPercent:000}% " +
                       $"余{_relayUnreadOtherPercent:000}%\n" +
                       $"成熟 追{HudCount(_relayMaturityTrackedTotal)} 补{HudCount(_relayMaturityClosedTotal)} " +
                       $"均约{HudCount(maturityMeanAudits / 2)}s\n" +
                       $"滞留 卡{HudCount((ulong)_relayMaturityStuckSamples)} 换视{maturityViewPercent:000}%\n" +
                       $"冻交 不读邻冻{FreezePercent(0)} 空旁冻{FreezePercent(3)}\n" +
                       $"冻缺 弱心冻{FreezePercent(1)} 邻缺冻{FreezePercent(2)}\n" +
                       $"冻卡 缺支冻{FreezePercent(7)} 卡支冻{FreezePercent(6)}\n" +
                       $"冻对 对齐邻冻{FreezePercent(4)} 错齐邻冻{FreezePercent(5)}\n" +
                       $"壳 活{live:00}/{Mathf.Clamp(liveLayerCount, 0, 99):00} " +
                       $"缺{_reviewMissingPercent:000}% 成{_planeConnectivityAcceptedPercent:000}% " +
                       $"账{(!_auditLedgerHasSnapshot ? '待' : _auditLedgerFaultMask == 0 ? '正' : '错')}" +
                       $"{_auditLedgerBatchToken % 1000u:000}码{_auditLedgerFaultMask:00} " +
                       $"冻{(HasFrozenSnapshot ? 1 : 0)}";
            }
        }

        public string AttributionDiagnosticsFixed =>
            "[错位归因·相关特征，非定罪]\n" +
            $"错征 斜{_relayBadObliquePercent:000}% 边{_relayBadDepthEdgePercent:000}% " +
            $"时{_relayBadTemporalPosePercent:000}% 层{_relayBadMultiLayerPercent:000}% 无{_relayBadUnattributedPercent:000}%\n" +
            $"对征 斜{_relayGoodObliquePercent:000}% 边{_relayGoodDepthEdgePercent:000}% " +
            $"时{_relayGoodTemporalPosePercent:000}% 层{_relayGoodMultiLayerPercent:000}%\n" +
            $"转角 错候{_relayCornerBadPercent:000}% 混疑{_relayCornerBadMixedPercent:000}% " +
            $"对候{_relayCornerGoodPercent:000}% 对混{_relayCornerGoodMixedPercent:000}%\n" +
            $"时姿 错时{_relayTemporalBadSamePercent:000}% 错位{_relayTemporalBadBaselinePercent:000}% " +
            $"对时{_relayTemporalGoodSamePercent:000}% 对位{_relayTemporalGoodBaselinePercent:000}%\n" +
            $"时基 龄{_relayFrameAgeMs:000}ms 间{_relayFrameIntervalMs:000}ms " +
            $"位{_relayPoseDeltaMm:000}mm/{_relayPoseDeltaTenthsDeg / 10f:0.0}°\n" +
            $"时戳 重{HudCount(_relayTimestampRepeats)} 倒{HudCount(_relayTimestampBacksteps)}\n" +
            "[融合写入·独立回读]\n" +
            (VolumeIntegrator.Instance != null ? VolumeIntegrator.Instance.GetFusionAdmissionDiagnosticsCompact() : "融写 无卷");

        private void Start()
        {
            Shader packShader = Resources.Load<Shader>("DepthPointSnapshotPack");
            Shader shellShader = Resources.Load<Shader>("InstantDepthShell");
            _reviewCompute = Resources.Load<ComputeShader>("InstantShellReview");
            if (packShader == null || shellShader == null || _reviewCompute == null)
            {
                Logger.Warning("即时外壳资源缺失（DepthPointSnapshotPack/InstantDepthShell/InstantShellReview），影子层跳过");
                enabled = false;
                return;
            }

            _snapshotPackMaterial = new Material(packShader) { name = "QRS Instant Shell Snapshot Pack" };
            _reviewKernel = _reviewCompute.FindKernel("ReviewShells");
            _connectivityKernel = _reviewCompute.FindKernel("AuditConnectivity");
            _publishWitnessKernel = _reviewCompute.FindKernel("PublishVerifiedWitnesses");
            _clearPaperAuditKernel = _reviewCompute.FindKernel("ClearPaperAuditCells");
            _rasterizePaperAuditKernel = _reviewCompute.FindKernel("RasterizePaperAuditTriangles");
            _relayAuditKernel = _reviewCompute.FindKernel("AuditRelay");
            if (DepthCapture.Instance != null)
                DepthCapture.Instance.Preprocessed += OnDepthPreprocessed;
            SetVisible(false);
        }

        public void SetVisible(bool visible)
        {
            _visible = visible && enabled;
            if (!_visible)
            {
                _seedPreviewVisible = false;
                if (!_automaticSeedCaptureActive)
                    _seedPreviewCapturePending = false;
                ApplySeedCaptureState();
            }
            // 定格只属于当次即时壳观察。离开该显示档即作废，
            // 避免绕一圈 X 档后又看到上一次的旧定格层。
            if (!_visible && _diagnosticFreezeActive)
            {
                _diagnosticFreezeActive = false;
                if (_frozenLayer != null) _frozenLayer.Valid = false;
            }
            if (_visible)
            {
                CompositeWithProduction = false;
                SetRejectedTriangleVisibility(true);
                SetDepthAwareOwnership(false);
                SetAllRenderers(true);
            }
            else
            {
                CompositeWithProduction = false;
                SetAllRenderers(false);
            }
        }

        public void SetCompositeWithProduction(bool composite)
        {
            if (composite) SetSeedPreviewVisible(false);
            CompositeWithProduction = _visible && composite;
            SetRejectedTriangleVisibility(!CompositeWithProduction);
            SetDepthAwareOwnership(CompositeWithProduction);
        }

        public void SetAcquiring(bool acquiring)
        {
            if (acquiring && !_acquiring) _nextCaptureAt = 0f;
            _acquiring = acquiring;
        }

        /// <summary>新扫描/B 清卷时清空扫描内历史和生产证词；切换显示档不会清空。</summary>
        public void ResetProductionWitnessLedger()
        {
            ClearAll();
            _witnessEpoch = 0u;
            _witnessPublishedPercent = 0;
            ResetRelayHud();
            if (_verifiedWitnessEpochs != null && _verifiedWitnessEpochs.IsValid())
                _verifiedWitnessEpochs.SetData(new uint[_verifiedWitnessEpochs.count]);
        }

        public void ClearAll()
        {
            _automaticSeedCaptureActive = false;
            _automaticSeedAttemptCount = 0;
            _nextAutomaticSeedAttemptAt = 0f;
            _seedPreviewCapturePending = false;
            _seedPreviewVisible = false;
            // ClearAll is also called by B-save/clear and by the first start of
            // a new roll.  Leaving this latch set would make Integrate reject
            // every frame forever even though the preview is no longer shown.
            DepthCapture.Instance?.SetSeedPlanePreviewActive(false);
            _seedPreview?.SetCaptureState(false, false);
            _nextLayer = 0;
            _lastCaptureAt = -1f;
            _nextReviewHistory = 0;
            _nextReviewHistoryAt = 0f;
            _diagnosticFreezeActive = false;
            _reviewEpoch++;
            _reviewMissingPercent = 0;
            _reviewNoWitnessPercent = 0;
            _reviewSameViewPercent = 0;
            _reviewIndependentPercent = 0;
            _reviewSameDepthConflictPercent = 0;
            _reviewSameNormalConflictPercent = 0;
            _reviewIndependentDepthConflictPercent = 0;
            _reviewIndependentNormalConflictPercent = 0;
            _reviewIndependentDepthResidualMeanMm = 0;
            _reviewIndependentDepthResidualMaxMm = 0;
            _connectivityEmittedPercent = 0;
            _connectivityMissingPercent = 0;
            _connectivityDepthRejectPercent = 0;
            _connectivityNormalRejectPercent = 0;
            _connectivityBothRejectPercent = 0;
            _planeConnectivityAcceptedPercent = 0;
            _planeConnectivityRescuedPercent = 0;
            _planeConnectivityGuardedPercent = 0;
            _planeConnectivityUndecidedPercent = 0;
            _witnessPublishedPercent = 0;
            _auditLedgerBatchToken = 0u;
            _auditLedgerFaultMask = 0;
            _auditLedgerHasSnapshot = false;
            _lastRelayResultAt = -1f;
            if (_reviewHistory != null)
                for (int i = 0; i < _reviewHistory.Length; i++)
                    _reviewHistory[i].Valid = false;
            if (_liveLayers != null)
            {
                for (int i = 0; i < _liveLayers.Length; i++)
                {
                    _liveLayers[i].Valid = false;
                    if (_liveLayers[i].Renderer != null) _liveLayers[i].Renderer.enabled = false;
                }
            }
            if (_frozenLayer != null)
            {
                _frozenLayer.Valid = false;
                if (_frozenLayer.Renderer != null) _frozenLayer.Renderer.enabled = false;
            }
        }

        /// <summary>
        /// A 键视差实验：首次按下复制当前最新的一层即时壳，
        /// 只显示这层世界锁定表面；再按解除。生产采集、GunGel 和 TSDF
        /// 全程不停，定格层也永不向生产发布数据。
        /// </summary>
        public bool ToggleDiagnosticFreeze(out bool frozen)
        {
            frozen = _diagnosticFreezeActive;
            if (!_visible || _seedPreviewVisible || !_built || _frozenLayer == null)
                return false;

            if (_diagnosticFreezeActive)
            {
                _diagnosticFreezeActive = false;
                _frozenLayer.Valid = false;
                ApplyDiagnosticFreezeVisibility();
                frozen = false;
                return true;
            }

            if (!DepthCapture.DepthAvailable)
                return false;

            float now = Time.unscaledTime;
            bool captured = CopyLatestLiveLayerIntoFrozen(now) ||
                            CaptureInto(_frozenLayer, now, true);
            if (!captured)
                return false;

            _diagnosticFreezeActive = true;
            ApplyDiagnosticFreezeVisibility();
            frozen = true;
            return true;
        }

        private bool CopyLatestLiveLayerIntoFrozen(float now)
        {
            if (_liveLayers == null || _frozenLayer == null)
                return false;

            ShellLayer latest = null;
            for (int i = 0; i < _liveLayers.Length; i++)
            {
                ShellLayer candidate = _liveLayers[i];
                if (candidate == null || !candidate.Valid || candidate.Snapshot == null)
                    continue;
                if (latest == null || candidate.CapturedAt > latest.CapturedAt)
                    latest = candidate;
            }
            if (latest == null || latest.Material == null || _frozenLayer.Material == null)
                return false;

            // 复制正在显示的已封装 GPU 内容，而不是在按键 Update
            // 里重读一次可能已被下一帧复用的全局深度纹理。
            Graphics.CopyTexture(latest.Snapshot, _frozenLayer.Snapshot);
            Graphics.CopyTexture(latest.ReviewState, _frozenLayer.ReviewState);
            Graphics.CopyTexture(latest.ReviewResidualMm, _frozenLayer.ReviewResidualMm);
            Graphics.CopyTexture(latest.ConnectivityState, _frozenLayer.ConnectivityState);
            Graphics.CopyTexture(latest.PlaneConnectivityState,
                _frozenLayer.PlaneConnectivityState);
            _frozenLayer.Material.SetMatrix(ProjID, latest.Material.GetMatrix(ProjID));
            _frozenLayer.Material.SetMatrix(ProjInvID, latest.Material.GetMatrix(ProjInvID));
            _frozenLayer.Material.SetMatrix(ViewInvID, latest.Material.GetMatrix(ViewInvID));
            _frozenLayer.Material.SetFloat(AgeID, 0f);
            _frozenLayer.Material.SetFloat(FrozenID, 1f);
            _frozenLayer.CapturedAt = now;
            _frozenLayer.Valid = true;
            return true;
        }

        private void ApplyDiagnosticFreezeVisibility()
        {
            if (_liveLayers != null)
                for (int i = 0; i < _liveLayers.Length; i++)
                    if (_liveLayers[i]?.Renderer != null)
                        _liveLayers[i].Renderer.enabled = _visible &&
                            !_seedPreviewVisible && !_diagnosticFreezeActive &&
                            _liveLayers[i].Valid;
            if (_frozenLayer?.Renderer != null)
                _frozenLayer.Renderer.enabled = _visible &&
                    !_seedPreviewVisible && _diagnosticFreezeActive &&
                    _frozenLayer.Valid;
        }

        /// <summary>
        /// 单帧中央圆拟合并暂存生产基底；观察期间不写卷，离开观察且卷为空时
        /// 才由共享深度入口送入 GunGel/裁决/TSDF。
        /// </summary>
        public void SetSeedPreviewVisible(bool visible)
        {
            _seedPreviewVisible = visible && _visible;
            if (_seedPreviewVisible) _seedPreviewCapturePending = true;
            ApplySeedCaptureState();
            if (_seedPreviewVisible)
            {
                _diagnosticFreezeActive = false;
                if (_frozenLayer != null) _frozenLayer.Valid = false;
                RequestSeedPreviewCapture();
            }
            ApplyDiagnosticFreezeVisibility();
        }

        /// <summary>A 键重新取一张单帧；之前结果保持可见直到新结果完成。</summary>
        public bool RequestSeedPreviewCapture()
        {
            if ((!_seedPreviewVisible && !_automaticSeedCaptureActive) ||
                _seedPreview == null) return false;
            // The first capture after entering this view must be a fresh frame,
            // not a shell already displayed before the X press.
            _seedPreviewCapturePending = true;
            return true;
        }

        private bool CaptureSeedPreviewFrom(ShellLayer source)
        {
            if ((!_seedPreviewVisible && !_automaticSeedCaptureActive) ||
                _seedPreview == null || source == null ||
                source.Material == null)
                return false;
            return _seedPreview.Capture(source.Snapshot,
                source.Material.GetMatrix(ProjInvID),
                source.Material.GetMatrix(ViewInvID), _pixelStep);
        }

        /// <summary>
        /// On a new empty roll, acquire the first qualified centre-disc ruler
        /// before the only TSDF is allowed to write.  This is deliberately
        /// hidden: it changes the production input, not the selected view.
        /// </summary>
        public bool BeginAutomaticSeedCapture()
        {
            DepthCapture depth = DepthCapture.Instance;
            if (depth == null || !depth.SeedPlaneExperimentEnabled)
                return false;

            depth.ClearSeedPlanePatches();
            _automaticSeedCaptureActive = true;
            _automaticSeedAttemptCount = 0;
            _nextAutomaticSeedAttemptAt = 0f;
            _seedPreviewCapturePending = true;
            ApplySeedCaptureState();
            Logger.Info("新空卷：融合前自动抓取中央60%合格基底；成功前TSDF保持关闭");
            return true;
        }

        private void ApplySeedCaptureState()
        {
            bool active = _seedPreviewVisible || _automaticSeedCaptureActive;
            DepthCapture.Instance?.SetSeedPlanePreviewActive(active);
            _seedPreview?.SetCaptureState(active, _seedPreviewVisible);
        }

        private void CompleteAutomaticSeedCapture()
        {
            if (!_automaticSeedCaptureActive) return;
            _automaticSeedCaptureActive = false;
            _seedPreviewCapturePending = false;
            ApplySeedCaptureState();
            Logger.Info($"融合前基底已自动抓取并放行唯一TSDF；尝试{_automaticSeedAttemptCount}次");
        }

        private void Update()
        {
            if (!_built)
            {
                Vector2 size = Shader.GetGlobalVector(TexSizeID);
                if (size.x >= 8f && size.y >= 8f)
                    Build((int)size.x, (int)size.y);
            }

            if (!_built) return;
            float now = Time.unscaledTime;
            if (_liveLayers != null)
            {
                for (int i = 0; i < _liveLayers.Length; i++)
                {
                    ShellLayer layer = _liveLayers[i];
                    if (!layer.Valid) continue;
                    float age = now - layer.CapturedAt;
                    // 壳纸合流的退场权属于“同深度纸皮已接管”，不属于
                    // 墙钟。合流期若仍按年龄淡出，纸皮即使仍在物体背后也会
                    // 在 0.65s 后露出，视觉上就是空调/柜体被墙面抹平。
                    if (!CompositeWithProduction && age >= liveLifetime)
                    {
                        layer.Valid = false;
                        layer.Renderer.enabled = false;
                        continue;
                    }
                    layer.Material.SetFloat(AgeID, CompositeWithProduction
                        ? 0f
                        : Mathf.Clamp01(age / Mathf.Max(0.1f, liveLifetime)));
                    layer.Renderer.enabled = _visible && !_seedPreviewVisible &&
                                             !_diagnosticFreezeActive;
                }
            }
            if (_frozenLayer != null && _frozenLayer.Valid)
                _frozenLayer.Renderer.enabled = _visible && !_seedPreviewVisible &&
                                                _diagnosticFreezeActive;
        }

        private void OnDepthPreprocessed()
        {
            if (!_acquiring || !DepthCapture.DepthAvailable) return;
            if (!_built)
            {
                Vector2 size = Shader.GetGlobalVector(TexSizeID);
                if (size.x < 8f || size.y < 8f) return;
                Build((int)size.x, (int)size.y);
            }

            float now = Time.unscaledTime;
            DepthCapture depth = DepthCapture.Instance;
            if (_automaticSeedCaptureActive && depth != null)
            {
                if (depth.HasStagedSeedPlanes)
                    CompleteAutomaticSeedCapture();
                else if (!_seedPreviewCapturePending &&
                         (_seedPreview == null || !_seedPreview.IsPending) &&
                         now >= _nextAutomaticSeedAttemptAt)
                    _seedPreviewCapturePending = true;
            }
            if (now < _nextCaptureAt || _liveLayers == null || _liveLayers.Length == 0) return;
            _nextCaptureAt = now + captureInterval;
            ShellLayer layer = null;
            int selectedLayer = -1;
            for (int offset = 0; offset < _liveLayers.Length; offset++)
            {
                int candidateIndex = (_nextLayer + offset) % _liveLayers.Length;
                ShellLayer candidate = _liveLayers[candidateIndex];
                if (candidate == null || candidate.CounterReadbackPending) continue;
                layer = candidate;
                selectedLayer = candidateIndex;
                break;
            }
            // 异步读回期间不得清零或复用同一计数缓冲；
            // 极端情况下宁可少捕一层，也不生成混批账本。
            if (layer == null) return;
            if (CaptureInto(layer, now, false))
            {
                _lastCaptureAt = now;
                _nextLayer = (selectedLayer + 1) % _liveLayers.Length;
                if (_seedPreviewCapturePending)
                {
                    bool started = CaptureSeedPreviewFrom(layer);
                    if (started)
                    {
                        _seedPreviewCapturePending = false;
                        if (_automaticSeedCaptureActive)
                        {
                            _automaticSeedAttemptCount++;
                            _nextAutomaticSeedAttemptAt = now + 0.5f;
                        }
                    }
                    else if (_automaticSeedCaptureActive)
                    {
                        _seedPreviewCapturePending = false;
                        _nextAutomaticSeedAttemptAt = now + 0.5f;
                    }
                }
            }
        }

        private void Build(int width, int height)
        {
            _sourceWidth = width;
            _sourceHeight = height;
            int step = Mathf.Max(1, Mathf.CeilToInt((float)Mathf.Max(width, height) / maxGridDim));
            _pixelStep = step;
            int cellsX = Mathf.Max(1, (width - 1) / step);
            int cellsY = Mathf.Max(1, (height - 1) / step);
            _reviewGridWidth = cellsX + 1;
            _reviewGridHeight = cellsY + 1;
            int vertexCount = cellsX * cellsY * 6;
            var vertices = new Vector3[vertexCount];
            var uv = new Vector2[vertexCount];
            var uv2 = new Vector2[vertexCount];
            var colors = new Color[vertexCount];
            var indices = new int[vertexCount];
            Color b0 = new Color(1f, 0f, 0f, 1f);
            Color b1 = new Color(0f, 1f, 0f, 1f);
            Color b2 = new Color(0f, 0f, 1f, 1f);

            int n = 0;
            for (int y = 0; y < cellsY; y++)
            for (int x = 0; x < cellsX; x++)
            {
                float px = x * step;
                float py = y * step;
                Vector2 cell = new Vector2(px, py);
                WriteVertex(vertices, uv, uv2, colors, indices, ref n, px, py, 0f, step, cell, b0);
                WriteVertex(vertices, uv, uv2, colors, indices, ref n, px + step, py, 0f, step, cell, b1);
                WriteVertex(vertices, uv, uv2, colors, indices, ref n, px, py + step, 0f, step, cell, b2);
                WriteVertex(vertices, uv, uv2, colors, indices, ref n, px + step, py, 1f, step, cell, b0);
                WriteVertex(vertices, uv, uv2, colors, indices, ref n, px + step, py + step, 1f, step, cell, b1);
                WriteVertex(vertices, uv, uv2, colors, indices, ref n, px, py + step, 1f, step, cell, b2);
            }

            _mesh = new Mesh
            {
                name = "QRS Instant Depth Shell Grid",
                indexFormat = IndexFormat.UInt32,
                vertices = vertices,
                uv = uv,
                uv2 = uv2,
                colors = colors,
                bounds = new Bounds(Vector3.zero, Vector3.one * 10000f)
            };
            _mesh.SetIndices(indices, MeshTopology.Triangles, 0, false);

            Shader shellShader = Resources.Load<Shader>("InstantDepthShell");
            _liveLayers = new ShellLayer[liveLayerCount];
            for (int i = 0; i < _liveLayers.Length; i++)
                _liveLayers[i] = CreateLayer($"[QRS] Instant Shell Live {i:00}", shellShader, false);
            _frozenLayer = CreateLayer("[QRS] Instant Shell Frozen", shellShader, true);
            _seedPreview = GetComponent<InstantSeedPlanePreview>();
            if (_seedPreview == null)
                _seedPreview = gameObject.AddComponent<InstantSeedPlanePreview>();
            ApplySeedCaptureState();
            _reviewHistory = new ReviewReference[Mathf.Clamp(reviewHistoryCount, 1, 24)];
            for (int i = 0; i < _reviewHistory.Length; i++)
            {
                _reviewHistory[i] = new ReviewReference
                {
                    Snapshot = CreateSnapshot($"[QRS] Instant Shell Review History {i:00}")
                };
            }
            AllocateVerifiedWitnessLedger();
            _built = true;
            Logger.Info($"即时外壳影子层已建: {cellsX}x{cellsY} 格 × {_liveLayers.Length} 活层，" +
                        $"历史参照{_reviewHistory.Length}层，步长{step}px，寿命{liveLifetime:0.00}s");
        }

        private static void WriteVertex(Vector3[] vertices, Vector2[] uv, Vector2[] uv2, Color[] colors,
            int[] indices, ref int n, float px, float py, float triangle, float step,
            Vector2 cell, Color barycentric)
        {
            vertices[n] = new Vector3(px, py, triangle);
            uv[n] = cell;
            // uv2.x 让同一三角的三个顶点共享完全相同的源像素步长，
            // 防止连通裁决在三角内部自相矛盾。
            uv2[n] = new Vector2(step, 0f);
            colors[n] = barycentric;
            indices[n] = n;
            n++;
        }

        private ShellLayer CreateLayer(string objectName, Shader shader, bool frozen)
        {
            RenderTexture snapshot = CreateSnapshot(objectName + " Snapshot");
            var reviewDescriptor = new RenderTextureDescriptor(_sourceWidth, _sourceHeight,
                GraphicsFormat.R32_UInt, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                msaaSamples = 1,
                useMipMap = false,
                autoGenerateMips = false,
                enableRandomWrite = true,
                sRGB = false
            };
            var reviewState = new RenderTexture(reviewDescriptor)
            {
                name = objectName + " Review State",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            reviewState.Create();
            var reviewResidualMm = new RenderTexture(reviewDescriptor)
            {
                name = objectName + " Review Residual Mm",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            reviewResidualMm.Create();
            var connectivityState = new RenderTexture(reviewDescriptor)
            {
                name = objectName + " Connectivity State",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            connectivityState.Create();
            var planeConnectivityState = new RenderTexture(reviewDescriptor)
            {
                name = objectName + " Plane Connectivity State",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            planeConnectivityState.Create();
            var counters = new ComputeBuffer(ZeroReviewCounters.Length, sizeof(uint),
                ComputeBufferType.Structured);
            counters.SetData(ZeroReviewCounters);

            var child = new GameObject(objectName);
            child.transform.SetParent(transform, false);
            child.AddComponent<MeshFilter>().sharedMesh = _mesh;
            MeshRenderer renderer = child.AddComponent<MeshRenderer>();
            var material = new Material(shader) { name = objectName + " Material" };
            material.SetTexture(SnapshotID, snapshot);
            material.SetTexture(ReviewTexID, reviewState);
            material.SetTexture(ConnectivityStateID, connectivityState);
            material.SetTexture(PlaneConnectivityStateID, planeConnectivityState);
            material.SetVector(SnapshotTexSizeID, new Vector4(_sourceWidth, _sourceHeight, 0f, 0f));
            material.SetFloat(FrozenID, frozen ? 1f : 0f);
            material.SetFloat(ShowRejectedTrianglesID, CompositeWithProduction ? 0f : 1f);
            material.SetFloat(CompositeWithProductionID, CompositeWithProduction ? 1f : 0f);
            material.SetFloat(CompositeDepthBiasMetersID, compositeDepthToleranceMeters);
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.enabled = false;
            return new ShellLayer
            {
                Renderer = renderer,
                Material = material,
                Snapshot = snapshot,
                ReviewState = reviewState,
                ReviewResidualMm = reviewResidualMm,
                ConnectivityState = connectivityState,
                PlaneConnectivityState = planeConnectivityState,
                ReviewCounters = counters
            };
        }

        private RenderTexture CreateSnapshot(string textureName)
        {
            var descriptor = new RenderTextureDescriptor(_sourceWidth, _sourceHeight,
                GraphicsFormat.R16G16B16A16_SFloat, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                msaaSamples = 1,
                useMipMap = false,
                autoGenerateMips = false,
                sRGB = false
            };
            var snapshot = new RenderTexture(descriptor)
            {
                name = textureName,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            snapshot.Create();
            return snapshot;
        }

        private void AllocateVerifiedWitnessLedger()
        {
            _verifiedWitnessEpochs?.Release();
            _verifiedWitnessEpochs = null;
            _paperAuditCells?.Release();
            _paperAuditCells = null;
            _relayMaturityCells?.Release();
            _relayMaturityCells = null;
            _paperAuditTriangles?.Release();
            _paperAuditTriangles = null;
            _paperAuditTriangleCapacity = 0;
            _paperAuditCellCount = new int3(0, 0, 0);
            _paperAuditCellTotal = 0;
            VolumeIntegrator volume = VolumeIntegrator.Instance;
            if (volume == null) return;

            _witnessVoxelCount = volume.VoxelCount;
            _witnessVoxelSize = volume.VoxelSize;

            // The paper relay map is rebuilt from committed production chunks
            // only for the low-rate readback batch. Keep every axis <=128 so
            // the diagnostic remains bounded on Quest even if the TSDF grows.
            int maxVoxelAxis = Mathf.Max(_witnessVoxelCount.x,
                Mathf.Max(_witnessVoxelCount.y, _witnessVoxelCount.z));
            _paperAuditStride = Mathf.Max(2, Mathf.CeilToInt(maxVoxelAxis / 128f));
            _paperAuditCellCount = new int3(
                Mathf.Max(1, (_witnessVoxelCount.x + _paperAuditStride - 1) /
                             _paperAuditStride),
                Mathf.Max(1, (_witnessVoxelCount.y + _paperAuditStride - 1) /
                             _paperAuditStride),
                Mathf.Max(1, (_witnessVoxelCount.z + _paperAuditStride - 1) /
                             _paperAuditStride));
            _paperAuditCellTotal = _paperAuditCellCount.x *
                                   _paperAuditCellCount.y *
                                   _paperAuditCellCount.z;
            _paperAuditCells = new ComputeBuffer(_paperAuditCellTotal, sizeof(uint),
                ComputeBufferType.Structured);
            _relayMaturityCells = new ComputeBuffer(_paperAuditCellTotal, sizeof(uint),
                ComputeBufferType.Structured);
            _relayMaturityCells.SetData(new uint[_paperAuditCellTotal]);
            _relayAuditEpoch = 0u;

            if (!PublishVerifiedWitnesses) return;
            int stride = Mathf.Max(1, witnessStride);
            _witnessCellCount = new int3(
                Mathf.Max(1, (_witnessVoxelCount.x + stride - 1) / stride),
                Mathf.Max(1, (_witnessVoxelCount.y + stride - 1) / stride),
                Mathf.Max(1, (_witnessVoxelCount.z + stride - 1) / stride));
            int cellCount = _witnessCellCount.x * _witnessCellCount.y * _witnessCellCount.z;
            _verifiedWitnessEpochs = new ComputeBuffer(cellCount, sizeof(uint),
                ComputeBufferType.Structured);
            _verifiedWitnessEpochs.SetData(new uint[cellCount]);
            _witnessEpoch = 0u;
        }

        private void EnsurePaperAuditTriangleCapacity(int requiredTriangles)
        {
            // Keep a small valid UAV even before the first production paper is
            // committed; Vulkan requires every resource used by AuditRelay to
            // be bound even when the runtime triangle count is zero.
            int boundedRequired = Mathf.Clamp(Mathf.Max(1, requiredTriangles),
                MinPaperAuditTriangleCapacity, MaxPaperAuditTriangleCapacity);
            int targetCapacity = Mathf.Clamp(
                Mathf.NextPowerOfTwo(boundedRequired),
                MinPaperAuditTriangleCapacity,
                MaxPaperAuditTriangleCapacity);
            if (_paperAuditTriangles != null && _paperAuditTriangles.IsValid() &&
                _paperAuditTriangleCapacity >= targetCapacity)
                return;

            _paperAuditTriangles?.Release();
            _paperAuditTriangles = new ComputeBuffer(targetCapacity,
                sizeof(float) * 9, ComputeBufferType.Structured);
            _paperAuditTriangleCapacity = targetCapacity;
        }

        private bool CaptureInto(ShellLayer layer, float now, bool frozen)
        {
            DepthCapture capture = DepthCapture.Instance;
            if (capture == null || layer == null) return false;
            if (!frozen && layer.CounterReadbackPending) return false;
            int eye = useRightEye ? DepthCapture.FusionEyeIndex : 0;
            if (capture.Proj == null || capture.ProjInv == null || capture.View == null ||
                capture.ViewInv == null || capture.Proj.Length <= eye ||
                capture.ProjInv.Length <= eye || capture.View.Length <= eye ||
                capture.ViewInv.Length <= eye)
                return false;

            _snapshotPackMaterial.SetFloat(EyeIndexID, eye);
            Graphics.Blit(null, layer.Snapshot, _snapshotPackMaterial);
            Matrix4x4 currentProj = capture.Proj[eye];
            Matrix4x4 currentProjInv = capture.ProjInv[eye];
            Matrix4x4 currentView = capture.View[eye];
            Matrix4x4 currentViewInv = capture.ViewInv[eye];
            ReviewReference reference = SelectReviewReference(currentViewInv, now,
                out bool referencePoseIndependent);
            DispatchReview(layer, currentProj, currentProjInv, currentViewInv,
                reference, referencePoseIndependent, now, frozen);
            layer.Material.SetMatrix(ProjID, currentProj);
            layer.Material.SetMatrix(ProjInvID, currentProjInv);
            layer.Material.SetMatrix(ViewInvID, currentViewInv);
            layer.Material.SetFloat(AgeID, 0f);
            layer.Material.SetFloat(FrozenID, frozen ? 1f : 0f);
            layer.CapturedAt = now;
            layer.Valid = true;
            layer.Renderer.enabled = _visible &&
                !_seedPreviewVisible &&
                (frozen ? _diagnosticFreezeActive : !_diagnosticFreezeActive);
            if (!frozen)
                RecordReviewReference(layer.Snapshot, currentProj, currentProjInv,
                    currentView, currentViewInv, now);
            return true;
        }

        private void DispatchReview(ShellLayer layer, Matrix4x4 currentProj,
            Matrix4x4 currentProjInv, Matrix4x4 currentViewInv,
            ReviewReference reference, bool referencePoseIndependent,
            float now, bool frozen)
        {
            if (_reviewCompute == null || _reviewKernel < 0 || layer.ReviewState == null ||
                _connectivityKernel < 0 || layer.ReviewResidualMm == null ||
                layer.ConnectivityState == null || layer.PlaneConnectivityState == null ||
                layer.ReviewCounters == null) return;

            uint batchToken = ++_reviewBatchSequence;
            if (batchToken == 0u) batchToken = ++_reviewBatchSequence;
            for (int i = 0; i < layer.ReviewCounterSeed.Length; i++)
                layer.ReviewCounterSeed[i] = 0u;
            layer.ReviewCounterSeed[22] = batchToken;
            layer.ReviewCounterSeed[23] = ~batchToken;
            layer.AuditBatchToken = batchToken;
            layer.ReviewCounters.SetData(layer.ReviewCounterSeed);
            // The expensive TSDF relay walk is paired with the existing 2 Hz
            // counter readback, not every 0.12 s shell capture. This keeps the
            // diagnostic from becoming a new Quest GPU pressure source.
            // View selection changes drawing only; audit the same production buffers in every view.
            bool relayAuditThisBatch = _acquiring && !frozen && !_reviewReadbackPending &&
                                       now >= _nextReviewReadbackAt;
            _reviewCompute.SetTexture(_reviewKernel, CurrentSnapshotID, layer.Snapshot);
            _reviewCompute.SetTexture(_reviewKernel, ReferenceSnapshotID,
                reference != null ? reference.Snapshot : layer.Snapshot);
            _reviewCompute.SetTexture(_reviewKernel, ReviewStateID, layer.ReviewState);
            _reviewCompute.SetTexture(_reviewKernel, ReviewResidualMmID, layer.ReviewResidualMm);
            _reviewCompute.SetBuffer(_reviewKernel, ReviewCountersID, layer.ReviewCounters);
            _reviewCompute.SetMatrix(CurrentProjID, currentProj);
            _reviewCompute.SetMatrix(CurrentProjInvID, currentProjInv);
            _reviewCompute.SetMatrix(CurrentViewInvID, currentViewInv);
            _reviewCompute.SetMatrix(ReferenceProjID,
                reference != null ? reference.Proj : currentProj);
            _reviewCompute.SetMatrix(ReferenceProjInvID,
                reference != null ? reference.ProjInv : currentProjInv);
            _reviewCompute.SetMatrix(ReferenceViewID,
                reference != null ? reference.View : Matrix4x4.Inverse(currentViewInv));
            _reviewCompute.SetMatrix(ReferenceViewInvID,
                reference != null ? reference.ViewInv : currentViewInv);
            _reviewCompute.SetInts(SourceSizeID, _sourceWidth, _sourceHeight);
            _reviewCompute.SetInts(ReviewGridSizeID, _reviewGridWidth, _reviewGridHeight);
            _reviewCompute.SetInt(PixelStepID, _pixelStep);
            _reviewCompute.SetInt(HasReferenceID, reference != null ? 1 : 0);
            _reviewCompute.SetInt(ReferencePoseIndependentID, referencePoseIndependent ? 1 : 0);
            Vector3 currentCamera = currentViewInv.MultiplyPoint3x4(Vector3.zero);
            Vector3 referenceCamera = reference != null ? reference.CameraPosition : currentCamera;
            _reviewCompute.SetVector(CurrentCameraWorldID,
                new Vector4(currentCamera.x, currentCamera.y, currentCamera.z, 1f));
            _reviewCompute.SetVector(ReferenceCameraWorldID,
                new Vector4(referenceCamera.x, referenceCamera.y, referenceCamera.z, 1f));
            _reviewCompute.SetFloat(MinIndependentRayCosID,
                Mathf.Cos(Mathf.Clamp(minIndependentRayAngle, 0.1f, 10f) * Mathf.Deg2Rad));
            DepthCapture timingCapture = DepthCapture.Instance;
            _relayFrameAgeMs = timingCapture != null
                ? Mathf.Clamp(Mathf.RoundToInt(timingCapture.LastPreprocessAgeMs), 0, 999)
                : 0;
            _relayFrameIntervalMs = timingCapture != null
                ? Mathf.Clamp(Mathf.RoundToInt(timingCapture.LastDepthFrameIntervalMs), 0, 999)
                : 0;
            _relayTimestampBacksteps = timingCapture?.DepthTimestampBackstepCount ?? 0u;
            _relayTimestampRepeats = timingCapture?.DepthTimestampRepeatCount ?? 0u;
            Camera renderCamera = Camera.main;
            if (renderCamera != null)
            {
                _relayPoseDeltaMm = Mathf.Clamp(Mathf.RoundToInt(
                    Vector3.Distance(currentCamera, renderCamera.transform.position) * 1000f), 0, 999);
                Vector3 capturedForward = currentViewInv.MultiplyVector(Vector3.forward).normalized;
                float forwardAngle = Vector3.Angle(capturedForward, renderCamera.transform.forward);
                forwardAngle = Mathf.Min(forwardAngle, 180f - forwardAngle);
                _relayPoseDeltaTenthsDeg = Mathf.Clamp(Mathf.RoundToInt(forwardAngle * 10f), 0, 999);
            }
            else
            {
                _relayPoseDeltaMm = 0;
                _relayPoseDeltaTenthsDeg = 0;
            }
            _reviewCompute.Dispatch(_reviewKernel,
                Mathf.CeilToInt(_reviewGridWidth / 8f),
                Mathf.CeilToInt(_reviewGridHeight / 8f), 1);

            // Direct relay audit: the current instantaneous shell is the ruler.
            // Historical reprojection never enters an acceptance denominator;
            // it is bound only as a read-only temporal/pose correlation flag.
            // The older witness-publication path below remains a separate
            // production policy and is not changed by this diagnostic patch.
            VolumeIntegrator relayVolume = VolumeIntegrator.Instance;
            if (relayAuditThisBatch && _relayAuditKernel >= 0 && relayVolume != null &&
                relayVolume.Volume != null && _paperAuditCells != null &&
                _paperAuditCells.IsValid())
            {
                int3 relayVoxelCount = relayVolume.VoxelCount;
                if (_paperAuditCells != null && _paperAuditCells.IsValid() &&
                    _clearPaperAuditKernel >= 0 && _rasterizePaperAuditKernel >= 0)
                {
                    MeshExtractor paperExtractor = MeshExtractor.Instance;
                    EnsurePaperAuditTriangleCapacity(
                        paperExtractor != null
                            ? paperExtractor.ProductionPaperCommittedTriangleCount
                            : 0);
                    _reviewCompute.SetBuffer(_clearPaperAuditKernel, PaperAuditCellsID,
                        _paperAuditCells);
                    _reviewCompute.SetInt(PaperAuditCellTotalID, _paperAuditCellTotal);
                    _reviewCompute.SetInt(PaperAuditClearOffsetID, 0);
                    _reviewCompute.Dispatch(_clearPaperAuditKernel,
                        Mathf.CeilToInt(_paperAuditCellTotal / 64f), 1, 1);

                    _reviewCompute.SetBuffer(_rasterizePaperAuditKernel, PaperAuditCellsID,
                        _paperAuditCells);
                    _reviewCompute.SetInts(TsdfVoxelCountID, relayVoxelCount.x,
                        relayVoxelCount.y, relayVoxelCount.z);
                    _reviewCompute.SetFloat(TsdfVoxelSizeID, relayVolume.VoxelSize);
                    _reviewCompute.SetInts(PaperAuditCellCountID, _paperAuditCellCount.x,
                        _paperAuditCellCount.y, _paperAuditCellCount.z);
                    _reviewCompute.SetInt(PaperAuditStrideID, _paperAuditStride);
                    _reviewCompute.SetInt(PaperAuditCellTotalID, _paperAuditCellTotal);
                    int paperTriangleCount = 0;
                    if (_paperAuditTriangles != null &&
                        _paperAuditTriangles.IsValid())
                    {
                        _reviewCompute.SetBuffer(_rasterizePaperAuditKernel,
                            RelayPaperTrianglesID, _paperAuditTriangles);
                        paperTriangleCount = paperExtractor?.DispatchProductionPaperRelayRaster(
                            _reviewCompute, _rasterizePaperAuditKernel,
                            RelayPaperVerticesID, RelayPaperIndicesID,
                            RelayPaperVertexCountID, RelayPaperIndexCountID,
                            RelayPaperLocalToWorldID,
                            RelayPaperTriangleOffsetID,
                            RelayPaperTriangleCapacityID,
                            _paperAuditTriangleCapacity) ?? 0;
                    }
                    _reviewCompute.SetInt(RelayPaperTriangleCountID,
                        paperTriangleCount);
                }

                _reviewCompute.SetTexture(_relayAuditKernel, CurrentSnapshotID, layer.Snapshot);
                _reviewCompute.SetTexture(_relayAuditKernel, ReferenceSnapshotID,
                    reference != null ? reference.Snapshot : layer.Snapshot);
                _reviewCompute.SetTexture(_relayAuditKernel, TsdfVolumeID, relayVolume.Volume);
                _reviewCompute.SetBuffer(_relayAuditKernel, ReviewCountersID,
                    layer.ReviewCounters);
                if (_paperAuditCells != null && _paperAuditCells.IsValid())
                    _reviewCompute.SetBuffer(_relayAuditKernel, PaperAuditCellsID,
                        _paperAuditCells);
                if (_relayMaturityCells != null && _relayMaturityCells.IsValid())
                    _reviewCompute.SetBuffer(_relayAuditKernel, RelayMaturityCellsID,
                        _relayMaturityCells);
                if (_paperAuditTriangles != null && _paperAuditTriangles.IsValid())
                    _reviewCompute.SetBuffer(_relayAuditKernel, RelayPaperTrianglesID,
                        _paperAuditTriangles);
                _reviewCompute.SetMatrix(CurrentProjID, currentProj);
                _reviewCompute.SetMatrix(CurrentProjInvID, currentProjInv);
                _reviewCompute.SetMatrix(CurrentViewInvID, currentViewInv);
                _reviewCompute.SetMatrix(ReferenceProjID,
                    reference != null ? reference.Proj : currentProj);
                _reviewCompute.SetMatrix(ReferenceProjInvID,
                    reference != null ? reference.ProjInv : currentProjInv);
                _reviewCompute.SetMatrix(ReferenceViewID,
                    reference != null ? reference.View : Matrix4x4.Inverse(currentViewInv));
                _reviewCompute.SetMatrix(ReferenceViewInvID,
                    reference != null ? reference.ViewInv : currentViewInv);
                _reviewCompute.SetInts(SourceSizeID, _sourceWidth, _sourceHeight);
                _reviewCompute.SetInts(ReviewGridSizeID, _reviewGridWidth, _reviewGridHeight);
                _reviewCompute.SetInt(PixelStepID, _pixelStep);
                _reviewCompute.SetInt(HasReferenceID, reference != null ? 1 : 0);
                _reviewCompute.SetInt(ReferencePoseIndependentID,
                    referencePoseIndependent ? 1 : 0);
                _reviewCompute.SetVector(CurrentCameraWorldID,
                    new Vector4(currentCamera.x, currentCamera.y, currentCamera.z, 1f));
                _reviewCompute.SetVector(ReferenceCameraWorldID,
                    new Vector4(referenceCamera.x, referenceCamera.y, referenceCamera.z, 1f));
                _reviewCompute.SetInts(TsdfVoxelCountID, relayVoxelCount.x,
                    relayVoxelCount.y, relayVoxelCount.z);
                _reviewCompute.SetFloat(TsdfVoxelSizeID, relayVolume.VoxelSize);
                _reviewCompute.SetFloat(TsdfTruncationDistanceID,
                    relayVolume.VoxelDistance);
                _reviewCompute.SetFloat(RelayMinTsdfWeightID,
                    Mathf.Max(0.001f, relayVolume.MinMeshWeight));
                _reviewCompute.SetFloat(RelayTsdfToleranceMetersID,
                    Mathf.Max(0.001f, relayTsdfToleranceMeters));
                _reviewCompute.SetFloat(RelayZeroToleranceMetersID,
                    Mathf.Max(0.001f, relayZeroToleranceMeters));
                _reviewCompute.SetFloat(RelayPaperToleranceMetersID,
                    Mathf.Max(0.001f, relayPaperToleranceMeters));
                _reviewCompute.SetFloat(RelayPaperSearchMetersID,
                    Mathf.Max(relayPaperToleranceMeters, relayPaperSearchMeters));
                _reviewCompute.SetInts(PaperAuditCellCountID, _paperAuditCellCount.x,
                    _paperAuditCellCount.y, _paperAuditCellCount.z);
                _reviewCompute.SetInt(PaperAuditStrideID, _paperAuditStride);
                _reviewCompute.SetInt(PaperAuditCellTotalID, _paperAuditCellTotal);
                _relayAuditEpoch++;
                if (_relayAuditEpoch == 0u) _relayAuditEpoch++;
                _reviewCompute.SetInt(RelayAuditEpochID, unchecked((int)_relayAuditEpoch));
                bool timingRisk = _relayFrameAgeMs > 50 || _relayFrameIntervalMs > 45 ||
                                  (timingCapture != null && timingCapture.LastDepthTimestampBackstepped);
                bool poseRisk = _relayPoseDeltaMm > 20 || _relayPoseDeltaTenthsDeg > 15;
                _reviewCompute.SetInt(RelayTimingRiskID, timingRisk ? 1 : 0);
                _reviewCompute.SetInt(RelayPoseRiskID, poseRisk ? 1 : 0);
                _reviewCompute.Dispatch(_relayAuditKernel,
                    Mathf.CeilToInt(_reviewGridWidth / 8f),
                    Mathf.CeilToInt(_reviewGridHeight / 8f), 1);
            }

            _reviewCompute.SetTexture(_connectivityKernel, CurrentSnapshotID, layer.Snapshot);
            _reviewCompute.SetTexture(_connectivityKernel, ConnectivityStateID, layer.ConnectivityState);
            _reviewCompute.SetTexture(_connectivityKernel, PlaneConnectivityStateID,
                layer.PlaneConnectivityState);
            _reviewCompute.SetBuffer(_connectivityKernel, ReviewCountersID, layer.ReviewCounters);
            _reviewCompute.SetMatrix(CurrentProjID, currentProj);
            _reviewCompute.SetMatrix(CurrentProjInvID, currentProjInv);
            _reviewCompute.SetMatrix(CurrentViewInvID, currentViewInv);
            _reviewCompute.SetInts(SourceSizeID, _sourceWidth, _sourceHeight);
            _reviewCompute.SetInts(ReviewGridSizeID, _reviewGridWidth, _reviewGridHeight);
            _reviewCompute.SetInt(PixelStepID, _pixelStep);
            _reviewCompute.Dispatch(_connectivityKernel,
                Mathf.CeilToInt(Mathf.Max(1, _reviewGridWidth - 1) / 8f),
                Mathf.CeilToInt(Mathf.Max(1, _reviewGridHeight - 1) / 8f), 1);

            // 证词发布严格晚于“独立视角复核 + 连通审计”。只有原生
            // 连通且三顶点全部独立确认的面才会在 5cm 账本单元写最近
            // epoch；四点共面救回面只留在诊断显示。账本不携带深度，也不能
            // 直接创建 TSDF。
            if (!frozen && PublishVerifiedWitnesses && _publishWitnessKernel >= 0 &&
                _verifiedWitnessEpochs != null && _verifiedWitnessEpochs.IsValid())
            {
                _witnessEpoch++;
                if (_witnessEpoch == 0u) _witnessEpoch++;
                _reviewCompute.SetTexture(_publishWitnessKernel, CurrentSnapshotID, layer.Snapshot);
                _reviewCompute.SetTexture(_publishWitnessKernel, ReviewStateID, layer.ReviewState);
                _reviewCompute.SetTexture(_publishWitnessKernel, PlaneConnectivityStateID,
                    layer.PlaneConnectivityState);
                _reviewCompute.SetBuffer(_publishWitnessKernel, ReviewCountersID,
                    layer.ReviewCounters);
                _reviewCompute.SetBuffer(_publishWitnessKernel, VerifiedWitnessEpochsID,
                    _verifiedWitnessEpochs);
                _reviewCompute.SetMatrix(CurrentProjInvID, currentProjInv);
                _reviewCompute.SetMatrix(CurrentViewInvID, currentViewInv);
                _reviewCompute.SetInts(SourceSizeID, _sourceWidth, _sourceHeight);
                _reviewCompute.SetInts(ReviewGridSizeID, _reviewGridWidth, _reviewGridHeight);
                _reviewCompute.SetInt(PixelStepID, _pixelStep);
                _reviewCompute.SetInts(WitnessVoxelCountID, _witnessVoxelCount.x,
                    _witnessVoxelCount.y, _witnessVoxelCount.z);
                _reviewCompute.SetInts(WitnessCellCountID, _witnessCellCount.x,
                    _witnessCellCount.y, _witnessCellCount.z);
                _reviewCompute.SetInt(WitnessStrideID, Mathf.Max(1, witnessStride));
                _reviewCompute.SetFloat(WitnessVoxelSizeID, _witnessVoxelSize);
                _reviewCompute.SetInt(WitnessEpochID, unchecked((int)_witnessEpoch));
                _reviewCompute.Dispatch(_publishWitnessKernel,
                    Mathf.CeilToInt(Mathf.Max(1, _reviewGridWidth - 1) / 8f),
                    Mathf.CeilToInt(Mathf.Max(1, _reviewGridHeight - 1) / 8f), 1);
            }

            if (relayAuditThisBatch)
            {
                _nextReviewReadbackAt = now + 0.5f;
                _reviewReadbackPending = true;
                layer.CounterReadbackPending = true;
                int requestEpoch = _reviewEpoch;
                uint requestBatchToken = batchToken;
                AsyncGPUReadback.Request(layer.ReviewCounters, request =>
                {
                    _reviewReadbackPending = false;
                    layer.CounterReadbackPending = false;
                    if (request.hasError || this == null || requestEpoch != _reviewEpoch) return;
                    var data = request.GetData<uint>();
                    if (data.Length < ZeroReviewCounters.Length ||
                        layer.AuditBatchToken != requestBatchToken) return;
                    _lastRelayResultAt = Time.unscaledTime;
                    int ledgerFaultMask = 0;
                    if (data[22] != requestBatchToken || data[23] != ~requestBatchToken)
                        ledgerFaultMask |= 1;
                    uint total = data[0];
                    uint valid = data[1];
                    if (total == 0u)
                    {
                        _auditLedgerBatchToken = requestBatchToken;
                        _auditLedgerFaultMask = ledgerFaultMask | 32;
                        _auditLedgerHasSnapshot = true;
                        ResetRelayHud();
                        return;
                    }
                    _reviewMissingPercent = Mathf.Clamp(Mathf.RoundToInt(data[2] * 100f / total), 0, 100);
                    if (valid == 0u)
                    {
                        _reviewNoWitnessPercent = 0;
                        _reviewSameViewPercent = 0;
                        _reviewIndependentPercent = 0;
                        _reviewSameDepthConflictPercent = 0;
                        _reviewSameNormalConflictPercent = 0;
                        _reviewIndependentDepthConflictPercent = 0;
                        _reviewIndependentNormalConflictPercent = 0;
                        _reviewIndependentDepthResidualMeanMm = 0;
                        _reviewIndependentDepthResidualMaxMm = 0;
                    }
                    else
                    {
                        _reviewNoWitnessPercent = Percent(data[3], valid);
                        _reviewSameViewPercent = Percent(data[4], valid);
                        _reviewIndependentPercent = Percent(data[5], valid);
                        _reviewSameDepthConflictPercent = Percent(data[6], valid);
                        _reviewSameNormalConflictPercent = Percent(data[7], valid);
                        _reviewIndependentDepthConflictPercent = Percent(data[8], valid);
                        _reviewIndependentNormalConflictPercent = Percent(data[9], valid);
                        _reviewIndependentDepthResidualMeanMm = data[8] > 0u
                            ? Mathf.Clamp(Mathf.RoundToInt((float)data[10] / data[8]), 0, 999)
                            : 0;
                        _reviewIndependentDepthResidualMaxMm = Mathf.Clamp((int)data[11], 0, 999);
                    }
                    uint triangleTotal = data[12];
                    uint oldPartition = data[13] + data[14] + data[15] + data[16] + data[17];
                    if (oldPartition != triangleTotal) ledgerFaultMask |= 2;
                    if (data[19] > data[15]) ledgerFaultMask |= 4;
                    if (data[18] != data[13] + data[19]) ledgerFaultMask |= 8;
                    if (data[15] != data[19] + data[20] + data[21]) ledgerFaultMask |= 16;
                    _auditLedgerBatchToken = requestBatchToken;
                    _auditLedgerFaultMask = ledgerFaultMask;
                    _auditLedgerHasSnapshot = true;
                    if (triangleTotal > 0u)
                    {
                        _connectivityEmittedPercent = Percent(data[13], triangleTotal);
                        _connectivityMissingPercent = Percent(data[14], triangleTotal);
                        _connectivityDepthRejectPercent = Percent(data[15], triangleTotal);
                        _connectivityNormalRejectPercent = Percent(data[16], triangleTotal);
                        _connectivityBothRejectPercent = Percent(data[17], triangleTotal);
                    }
                    else
                    {
                        _connectivityEmittedPercent = 0;
                        _connectivityMissingPercent = 0;
                        _connectivityDepthRejectPercent = 0;
                        _connectivityNormalRejectPercent = 0;
                        _connectivityBothRejectPercent = 0;
                    }
                    if (triangleTotal > 0u)
                    {
                        _planeConnectivityAcceptedPercent = Percent(data[18], triangleTotal);
                        _planeConnectivityRescuedPercent = Percent(data[19], triangleTotal);
                        _planeConnectivityGuardedPercent = Percent(data[20], triangleTotal);
                        _planeConnectivityUndecidedPercent = Percent(data[21], triangleTotal);
                        _witnessPublishedPercent = Percent(data[24], triangleTotal);
                    }
                    else
                    {
                        _planeConnectivityAcceptedPercent = 0;
                        _planeConnectivityRescuedPercent = 0;
                        _planeConnectivityGuardedPercent = 0;
                        _planeConnectivityUndecidedPercent = 0;
                        _witnessPublishedPercent = 0;
                    }

                    uint shellSamples = data[26];
                    uint tsdfReadable = data[27];
                    uint tsdfAligned = data[28];
                    uint unreadable = shellSamples >= tsdfReadable
                        ? shellSamples - tsdfReadable
                        : 0u;
                    uint unreadablePartition = data[39] + data[40] + data[41] +
                                               data[42] + data[43];
                    if (unreadablePartition != unreadable) ledgerFaultMask |= 64;
                    _auditLedgerFaultMask = ledgerFaultMask;
                    _relayShellTsdfSamples = HudSamples(shellSamples);
                    _relayShellTsdfReadablePercent = Percent(tsdfReadable, shellSamples);
                    // Alignment is conditional on readability. A missing TSDF
                    // is not silently re-labelled as bad geometry.
                    _relayShellTsdfAlignedPercent = Percent(tsdfAligned, tsdfReadable);
                    _relayShellTsdfResidualMeanMm = MeanMm(data[29], tsdfReadable);
                    _relayShellTsdfResidualMaxMm = HudMm(data[30]);
                    _relayUnreadOutsidePercent = Percent(data[39], unreadable);
                    _relayUnreadUnwrittenPercent = Percent(data[40], unreadable);
                    _relayUnreadLowWeightPercent = Percent(data[41], unreadable);
                    _relayUnreadNeighbourPercent = Percent(data[42], unreadable);
                    _relayUnreadOtherPercent = Percent(data[43], unreadable);

                    uint tsdfMisaligned = tsdfReadable >= tsdfAligned
                        ? tsdfReadable - tsdfAligned
                        : 0u;
                    uint[] freezeTotals = { unreadable, data[41], data[42], data[40],
                                            tsdfAligned, tsdfMisaligned, data[56], unreadable };
                    for (int freezeSlot = 0; freezeSlot < 8; freezeSlot++)
                    {
                        _relayFreezeCounts[freezeSlot] = data[66 + freezeSlot];
                        _relayFreezeTotals[freezeSlot] = freezeTotals[freezeSlot];
                        if (_relayFreezeCounts[freezeSlot] > freezeTotals[freezeSlot])
                            _auditLedgerFaultMask |= 128;
                    }
                    _relayBadObliquePercent = Percent(data[44], tsdfMisaligned);
                    _relayBadDepthEdgePercent = Percent(data[45], tsdfMisaligned);
                    _relayBadTemporalPosePercent = Percent(data[46], tsdfMisaligned);
                    _relayBadMultiLayerPercent = Percent(data[47], tsdfMisaligned);
                    _relayBadUnattributedPercent = Percent(data[48], tsdfMisaligned);
                    _relayGoodObliquePercent = Percent(data[49], tsdfAligned);
                    _relayGoodDepthEdgePercent = Percent(data[50], tsdfAligned);
                    _relayGoodTemporalPosePercent = Percent(data[51], tsdfAligned);
                    _relayGoodMultiLayerPercent = Percent(data[52], tsdfAligned);
                    _relayMaturityTrackedTotal += data[53];
                    _relayMaturityClosedTotal += data[54];
                    _relayMaturityAuditSum += data[55];
                    _relayMaturityStuckSamples = HudSamples(data[56]);
                    _relayMaturityViewChangedTotal += data[57];
                    _relayCornerBadPercent = Percent(data[58], tsdfMisaligned);
                    _relayCornerBadMixedPercent = Percent(data[59], data[58]);
                    _relayCornerGoodPercent = Percent(data[60], tsdfAligned);
                    _relayCornerGoodMixedPercent = Percent(data[61], data[60]);
                    _relayTemporalBadSamePercent = Percent(data[62], data[46]);
                    _relayTemporalBadBaselinePercent = Percent(data[63], data[46]);
                    _relayTemporalGoodSamePercent = Percent(data[64], data[51]);
                    _relayTemporalGoodBaselinePercent = Percent(data[65], data[51]);

                    uint zeroFound = data[31];
                    uint zeroAligned = data[32];
                    _relayTsdfZeroSamples = HudSamples(tsdfAligned);
                    _relayTsdfZeroFoundPercent = Percent(zeroFound, tsdfAligned);
                    _relayTsdfZeroAlignedPercent = Percent(zeroAligned, zeroFound);
                    _relayTsdfZeroResidualMeanMm = MeanMm(data[33], zeroFound);
                    _relayTsdfZeroResidualMaxMm = HudMm(data[34]);

                    uint paperFound = data[35];
                    uint paperAligned = data[36];
                    _relayZeroPaperSamples = HudSamples(zeroAligned);
                    _relayZeroPaperFoundPercent = Percent(paperFound, zeroAligned);
                    _relayZeroPaperAlignedPercent = Percent(paperAligned, paperFound);
                    _relayZeroPaperResidualMeanMm = MeanMm(data[37], paperFound);
                    _relayZeroPaperResidualMaxMm = HudMm(data[38]);

                    PushRelayAutoBatch(shellSamples, data[39], data[40], data[41],
                        data[42], data[43], tsdfReadable, tsdfAligned,
                        data[54], data[56]);
                });
            }
        }

        /// <summary>
        /// 自动挑一张仍与当前视野大致重叠、但相机位置已经分开的历史壳。
        /// 若暂时没有独立视角，则退回最近历史壳，只用于记录“同视重复”；
        /// 它绝不会被算成独立稳定。
        /// </summary>
        private ReviewReference SelectReviewReference(Matrix4x4 currentViewInv, float now,
            out bool poseIndependent)
        {
            poseIndependent = false;
            if (_reviewHistory == null || _reviewHistory.Length == 0) return null;

            Vector3 currentPosition = currentViewInv.MultiplyPoint3x4(Vector3.zero);
            Vector3 currentForward = currentViewInv.MultiplyVector(Vector3.forward).normalized;
            ReviewReference independent = null;
            float independentScore = float.NegativeInfinity;
            ReviewReference recent = null;
            float recentTime = float.NegativeInfinity;
            float minBaseline = Mathf.Max(0.001f, minIndependentBaseline);
            float maxBaseline = Mathf.Max(minBaseline, maxReferenceBaseline);

            for (int i = 0; i < _reviewHistory.Length; i++)
            {
                ReviewReference candidate = _reviewHistory[i];
                if (candidate == null || !candidate.Valid) continue;
                float age = now - candidate.CapturedAt;
                if (age < 0.05f || age > 12f) continue;

                float forwardAngle = Vector3.Angle(currentForward, candidate.CameraForward);
                if (forwardAngle <= maxReferenceForwardAngle && candidate.CapturedAt > recentTime)
                {
                    recent = candidate;
                    recentTime = candidate.CapturedAt;
                }

                float baseline = Vector3.Distance(currentPosition, candidate.CameraPosition);
                if (baseline < minBaseline || baseline > maxBaseline ||
                    forwardAngle > maxReferenceForwardAngle)
                    continue;

                // 约12cm的横向基线通常兼顾独立性和视野重叠；朝向差与陈旧度只做轻度扣分。
                float baselineScore = 1f - Mathf.Abs(baseline - 0.12f) / 0.12f;
                float angleScore = 1f - forwardAngle / Mathf.Max(1f, maxReferenceForwardAngle);
                float recencyScore = 1f - Mathf.Clamp01(age / 12f);
                float score = baselineScore * 0.55f + angleScore * 0.30f + recencyScore * 0.15f;
                if (score <= independentScore) continue;
                independent = candidate;
                independentScore = score;
            }

            if (independent != null)
            {
                poseIndependent = true;
                return independent;
            }
            return recent;
        }

        private void RecordReviewReference(RenderTexture source, Matrix4x4 proj,
            Matrix4x4 projInv, Matrix4x4 view, Matrix4x4 viewInv, float now)
        {
            if (_reviewHistory == null || _reviewHistory.Length == 0 ||
                source == null || now < _nextReviewHistoryAt)
                return;

            ReviewReference target = _reviewHistory[_nextReviewHistory];
            Graphics.CopyTexture(source, target.Snapshot);
            target.Proj = proj;
            target.ProjInv = projInv;
            target.View = view;
            target.ViewInv = viewInv;
            target.CameraPosition = viewInv.MultiplyPoint3x4(Vector3.zero);
            target.CameraForward = viewInv.MultiplyVector(Vector3.forward).normalized;
            target.CapturedAt = now;
            target.Valid = true;
            _nextReviewHistory = (_nextReviewHistory + 1) % _reviewHistory.Length;
            _nextReviewHistoryAt = now + Mathf.Max(0.05f, reviewHistoryInterval);
        }

        private void SetAllRenderers(bool visible)
        {
            if (_liveLayers != null)
                for (int i = 0; i < _liveLayers.Length; i++)
                    if (_liveLayers[i].Renderer != null)
                        _liveLayers[i].Renderer.enabled = visible &&
                            !_seedPreviewVisible && !_diagnosticFreezeActive &&
                            _liveLayers[i].Valid;
            if (_frozenLayer != null && _frozenLayer.Renderer != null)
                _frozenLayer.Renderer.enabled = visible &&
                    !_seedPreviewVisible && _diagnosticFreezeActive &&
                    _frozenLayer.Valid;
        }

        private void SetRejectedTriangleVisibility(bool visible)
        {
            if (_liveLayers != null)
                for (int i = 0; i < _liveLayers.Length; i++)
                    _liveLayers[i]?.Material?.SetFloat(ShowRejectedTrianglesID, visible ? 1f : 0f);
            _frozenLayer?.Material?.SetFloat(ShowRejectedTrianglesID, visible ? 1f : 0f);
        }

        private void SetDepthAwareOwnership(bool composite)
        {
            if (_liveLayers != null)
                for (int i = 0; i < _liveLayers.Length; i++)
                {
                    _liveLayers[i]?.Material?.SetFloat(CompositeWithProductionID,
                        composite ? 1f : 0f);
                    _liveLayers[i]?.Material?.SetFloat(CompositeDepthBiasMetersID,
                        compositeDepthToleranceMeters);
                }
            _frozenLayer?.Material?.SetFloat(CompositeWithProductionID, composite ? 1f : 0f);
            _frozenLayer?.Material?.SetFloat(CompositeDepthBiasMetersID,
                compositeDepthToleranceMeters);
        }

        private static int Percent(uint value, uint total)
        {
            return total > 0u
                ? Mathf.Clamp(Mathf.RoundToInt(value * 100f / total), 0, 100)
                : 0;
        }

        private static int MeanMm(uint sum, uint count)
        {
            return count > 0u
                ? Mathf.Clamp(Mathf.RoundToInt((float)sum / count), 0, 999)
                : 0;
        }

        private static int HudMm(uint value) => Mathf.Clamp((int)value, 0, 999);

        private static int HudSamples(uint value) => (int)System.Math.Min(value, (uint)int.MaxValue);

        // Preserve raw counters; mark display saturation rather than reporting a false exact 999.
        private static string HudCount(ulong value) => value > 999ul ? "999+" : value.ToString("000");

        private static int Percent(ulong value, ulong total)
        {
            if (total == 0ul) return 0;
            double percent = value * 100.0 / total;
            return Mathf.Clamp(Mathf.RoundToInt((float)percent), 0, 100);
        }

        private void PushRelayAutoBatch(uint shellSamples, uint outside,
            uint unwritten, uint lowWeight, uint neighbour, uint other,
            uint readable, uint aligned, uint closed, uint stuck)
        {
            int limit = Mathf.Clamp(relayAutoWindowBatches, 4,
                MaxRelayAutoWindowBatches);
            if (_relayAutoWindowLimit != limit)
            {
                System.Array.Clear(_relayAutoWindow, 0, _relayAutoWindow.Length);
                _relayAutoWindowLimit = limit;
                _relayAutoWindowCount = 0;
                _relayAutoWindowCursor = 0;
            }

            _relayAutoWindow[_relayAutoWindowCursor] = new RelayAutoBatch
            {
                ShellSamples = shellSamples,
                Outside = outside,
                Unwritten = unwritten,
                LowWeight = lowWeight,
                Neighbour = neighbour,
                Other = other,
                Readable = readable,
                Aligned = aligned,
                Closed = closed,
                Stuck = stuck
            };
            _relayAutoWindowCursor = (_relayAutoWindowCursor + 1) % limit;
            _relayAutoWindowCount = Mathf.Min(_relayAutoWindowCount + 1, limit);
            RecomputeRelayAutoVerdict();
        }

        private void RecomputeRelayAutoVerdict()
        {
            ulong shell = 0ul;
            ulong outside = 0ul;
            ulong unwritten = 0ul;
            ulong lowWeight = 0ul;
            ulong neighbour = 0ul;
            ulong other = 0ul;
            ulong readable = 0ul;
            ulong aligned = 0ul;
            ulong closed = 0ul;
            ulong stuck = 0ul;
            for (int i = 0; i < _relayAutoWindowCount; i++)
            {
                RelayAutoBatch batch = _relayAutoWindow[i];
                shell += batch.ShellSamples;
                outside += batch.Outside;
                unwritten += batch.Unwritten;
                lowWeight += batch.LowWeight;
                neighbour += batch.Neighbour;
                other += batch.Other;
                readable += batch.Readable;
                aligned += batch.Aligned;
                closed += batch.Closed;
                stuck += batch.Stuck;
            }

            ulong inVolumeSamples = shell >= outside ? shell - outside : 0ul;
            ulong unreadable = unwritten + lowWeight + neighbour + other;
            ulong misaligned = readable >= aligned ? readable - aligned : 0ul;
            _relayAutoSamples = inVolumeSamples;
            _relayAutoClosed = closed;
            _relayAutoGapPercent = Percent(unreadable, inVolumeSamples);
            _relayAutoMisalignedPercent = Percent(misaligned, readable);
            _relayAutoPersistentPercent = Percent(stuck, unreadable);
            _relayAutoUnwrittenPercent = Percent(unwritten, unreadable);
            _relayAutoLowWeightPercent = Percent(lowWeight, unreadable);
            _relayAutoNeighbourPercent = Percent(neighbour, unreadable);
            _relayAutoOtherPercent = Percent(other, unreadable);
            _relayAutoCause = DominantRelayCause(unreadable, unwritten,
                lowWeight, neighbour, other);

            bool enoughEvidence = _relayAutoWindowCount >= _relayAutoWindowLimit &&
                                  inVolumeSamples >= (ulong)Mathf.Max(1,
                                      relayAutoMinSamples);
            if (!enoughEvidence)
            {
                _relayAutoVerdict = "采样中";
                return;
            }
            if (_auditLedgerFaultMask != 0)
            {
                _relayAutoVerdict = "账本异常";
                return;
            }

            bool coverageBad = _relayAutoGapPercent > relayAutoMaxUnreadablePercent;
            bool depthBad = _relayAutoMisalignedPercent > relayAutoMaxMisalignedPercent;
            bool persistentBad = _relayAutoPersistentPercent > relayAutoMaxPersistentPercent;
            if (!coverageBad && !depthBad)
                _relayAutoVerdict = "通过";
            else if (coverageBad && persistentBad)
                _relayAutoVerdict = depthBad ? "久缺+深错" : "持续缺失";
            else if (coverageBad && closed > 0ul)
                _relayAutoVerdict = depthBad ? "慢补+深错" : "补得慢";
            else if (coverageBad)
                _relayAutoVerdict = depthBad ? "缺失+深错" : "覆盖不足";
            else
                _relayAutoVerdict = "深度异常";
        }

        private static string DominantRelayCause(ulong unreadable,
            ulong unwritten, ulong lowWeight, ulong neighbour, ulong other)
        {
            if (unreadable == 0ul) return "无";
            ulong top = unwritten;
            string cause = "空";
            if (lowWeight > top) { top = lowWeight; cause = "弱"; }
            if (neighbour > top) { top = neighbour; cause = "邻"; }
            if (other > top) { top = other; cause = "余"; }
            return top * 2ul >= unreadable ? cause : "混合";
        }

        private string FreezePercent(int slot) => _relayFreezeTotals[slot] == 0u
            ? "--" : $"{Percent(_relayFreezeCounts[slot], _relayFreezeTotals[slot]):000}%";

        private void ResetRelayHud()
        {
            System.Array.Clear(_relayFreezeCounts, 0, _relayFreezeCounts.Length);
            System.Array.Clear(_relayFreezeTotals, 0, _relayFreezeTotals.Length);
            _relayTimestampRepeats = 0;
            _relayTimestampBacksteps = 0;
            System.Array.Clear(_relayAutoWindow, 0, _relayAutoWindow.Length);
            _relayAutoWindowLimit = 0;
            _relayAutoWindowCount = 0;
            _relayAutoWindowCursor = 0;
            _relayAutoGapPercent = 0;
            _relayAutoMisalignedPercent = 0;
            _relayAutoPersistentPercent = 0;
            _relayAutoUnwrittenPercent = 0;
            _relayAutoLowWeightPercent = 0;
            _relayAutoNeighbourPercent = 0;
            _relayAutoOtherPercent = 0;
            _relayAutoSamples = 0ul;
            _relayAutoClosed = 0ul;
            _relayAutoVerdict = "采样中";
            _relayAutoCause = "待";
            _relayShellTsdfSamples = 0;
            _relayShellTsdfReadablePercent = 0;
            _relayShellTsdfAlignedPercent = 0;
            _relayShellTsdfResidualMeanMm = 0;
            _relayShellTsdfResidualMaxMm = 0;
            _relayTsdfZeroSamples = 0;
            _relayTsdfZeroFoundPercent = 0;
            _relayTsdfZeroAlignedPercent = 0;
            _relayTsdfZeroResidualMeanMm = 0;
            _relayTsdfZeroResidualMaxMm = 0;
            _relayZeroPaperSamples = 0;
            _relayZeroPaperFoundPercent = 0;
            _relayZeroPaperAlignedPercent = 0;
            _relayZeroPaperResidualMeanMm = 0;
            _relayZeroPaperResidualMaxMm = 0;
            _relayUnreadOutsidePercent = 0;
            _relayUnreadUnwrittenPercent = 0;
            _relayUnreadLowWeightPercent = 0;
            _relayUnreadNeighbourPercent = 0;
            _relayUnreadOtherPercent = 0;
            _relayBadObliquePercent = 0;
            _relayBadDepthEdgePercent = 0;
            _relayBadTemporalPosePercent = 0;
            _relayBadMultiLayerPercent = 0;
            _relayBadUnattributedPercent = 0;
            _relayGoodObliquePercent = 0;
            _relayGoodDepthEdgePercent = 0;
            _relayGoodTemporalPosePercent = 0;
            _relayGoodMultiLayerPercent = 0;
            _relayMaturityTrackedTotal = 0;
            _relayMaturityClosedTotal = 0;
            _relayMaturityAuditSum = 0;
            _relayMaturityViewChangedTotal = 0;
            _relayMaturityStuckSamples = 0;
            _relayCornerBadPercent = 0;
            _relayCornerBadMixedPercent = 0;
            _relayCornerGoodPercent = 0;
            _relayCornerGoodMixedPercent = 0;
            _relayTemporalBadSamePercent = 0;
            _relayTemporalBadBaselinePercent = 0;
            _relayTemporalGoodSamePercent = 0;
            _relayTemporalGoodBaselinePercent = 0;
            if (_relayMaturityCells != null && _relayMaturityCells.IsValid())
                _relayMaturityCells.SetData(new uint[_relayMaturityCells.count]);
        }

        /// <summary>
        /// Rasterize the exact committed native-5cm production paper into the
        /// TSDF cell lattice.  This stop-time audit is independent of shell or
        /// SupportTruth visibility and never feeds production.
        /// </summary>
        internal bool RequestProductionPaperOccupancyAuditExport(
            string reason, MeshExtractor extractor, Action<string> completed = null)
        {
            if (_paperOccupancyExportPending || _reviewCompute == null ||
                _clearPaperAuditKernel < 0 || _rasterizePaperAuditKernel < 0 ||
                extractor == null || VolumeIntegrator.Instance == null ||
                VolumeIntegrator.Instance.Volume == null)
                return false;

            int triangleCount = extractor.ProductionPaperCommittedTriangleCount;
            if (triangleCount <= 0 || triangleCount > MaxPaperAuditTriangleCapacity)
                return false;

            _paperOccupancyExportPending = true;
            StartCoroutine(ExportProductionPaperOccupancy(
                reason ?? string.Empty, extractor, triangleCount, completed));
            return true;
        }

        private IEnumerator ExportProductionPaperOccupancy(
            string reason, MeshExtractor extractor, int expectedTriangles,
            Action<string> completed)
        {
            string outputPath = string.Empty;
            Exception failure = null;
            ComputeBuffer cells = null;
            uint[] occupancy = null;
            int3 voxelCount = int3.zero;
            int3 cellCount = int3.zero;
            float voxelSize = 0f;
            int rasterizedTriangles = 0;

            try
            {
                VolumeIntegrator volume = VolumeIntegrator.Instance;
                voxelCount = volume.VoxelCount;
                voxelSize = volume.VoxelSize;
                cellCount = new int3(
                    Mathf.Max(1, voxelCount.x - 1),
                    Mathf.Max(1, voxelCount.y - 1),
                    Mathf.Max(1, voxelCount.z - 1));
                int total = checked(cellCount.x * cellCount.y * cellCount.z);
                cells = new ComputeBuffer(total, sizeof(uint), ComputeBufferType.Structured);
                occupancy = new uint[total];

                _reviewCompute.SetBuffer(_clearPaperAuditKernel, PaperAuditCellsID, cells);
                _reviewCompute.SetInt(PaperAuditCellTotalID, total);
                const int maxGroups = 65535;
                const int threadsPerGroup = 64;
                int clearOffset = 0;
                while (clearOffset < total)
                {
                    int batchCells = Mathf.Min(total - clearOffset,
                        maxGroups * threadsPerGroup);
                    _reviewCompute.SetInt(PaperAuditClearOffsetID, clearOffset);
                    _reviewCompute.Dispatch(_clearPaperAuditKernel,
                        Mathf.CeilToInt(batchCells / (float)threadsPerGroup), 1, 1);
                    clearOffset += batchCells;
                }

                EnsurePaperAuditTriangleCapacity(expectedTriangles);
                if (_paperAuditTriangles == null || !_paperAuditTriangles.IsValid())
                    throw new InvalidOperationException("production paper triangle audit buffer unavailable");

                _reviewCompute.SetBuffer(_rasterizePaperAuditKernel, PaperAuditCellsID, cells);
                _reviewCompute.SetBuffer(_rasterizePaperAuditKernel,
                    RelayPaperTrianglesID, _paperAuditTriangles);
                _reviewCompute.SetInts(TsdfVoxelCountID,
                    voxelCount.x, voxelCount.y, voxelCount.z);
                _reviewCompute.SetFloat(TsdfVoxelSizeID, voxelSize);
                _reviewCompute.SetInts(PaperAuditCellCountID,
                    cellCount.x, cellCount.y, cellCount.z);
                _reviewCompute.SetInt(PaperAuditStrideID, 1);
                _reviewCompute.SetInt(PaperAuditCellTotalID, total);
                rasterizedTriangles = extractor.DispatchProductionPaperRelayRaster(
                    _reviewCompute, _rasterizePaperAuditKernel,
                    RelayPaperVerticesID, RelayPaperIndicesID,
                    RelayPaperVertexCountID, RelayPaperIndexCountID,
                    RelayPaperLocalToWorldID, RelayPaperTriangleOffsetID,
                    RelayPaperTriangleCapacityID, _paperAuditTriangleCapacity);
                _reviewCompute.SetInt(RelayPaperTriangleCountID, rasterizedTriangles);
                if (rasterizedTriangles != expectedTriangles)
                    throw new InvalidDataException(
                        $"production paper raster incomplete: expected {expectedTriangles}, got {rasterizedTriangles}");
            }
            catch (Exception e)
            {
                failure = e;
            }

            if (failure == null)
            {
                const int valuesPerReadback = (1 << 20) / sizeof(uint);
                int copied = 0;
                while (copied < occupancy.Length && failure == null)
                {
                    int valueCount = Mathf.Min(valuesPerReadback, occupancy.Length - copied);
                    AsyncGPUReadbackRequest request = default;
                    try
                    {
                        request = AsyncGPUReadback.Request(cells,
                            valueCount * sizeof(uint), copied * sizeof(uint));
                    }
                    catch (Exception e)
                    {
                        failure = new IOException(
                            $"production paper occupancy readback request failed at cell {copied}", e);
                    }
                    if (failure != null) break;
                    while (!request.done) yield return null;
                    if (request.hasError)
                    {
                        failure = new IOException(
                            $"production paper occupancy GPU readback failed at cell {copied}");
                        break;
                    }

                    try
                    {
                        var source = request.GetData<uint>();
                        if (source.Length != valueCount)
                            throw new InvalidDataException(
                                $"occupancy readback length mismatch at cell {copied}: {source.Length}/{valueCount}");
                        for (int i = 0; i < valueCount; i++) occupancy[copied + i] = source[i];
                        copied += valueCount;
                    }
                    catch (Exception e)
                    {
                        failure = e;
                    }
                    yield return null;
                }
            }

            Task<string> writeTask = null;
            if (failure == null)
            {
                int3 capturedVoxelCount = voxelCount;
                int3 capturedCellCount = cellCount;
                float capturedVoxelSize = voxelSize;
                int capturedRasterizedTriangles = rasterizedTriangles;
                string directory = Path.Combine(Application.persistentDataPath,
                    "ScanCoverDiagnostics", "paper_hole_audit",
                    DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture));
                try
                {
                    writeTask = Task.Run(() => WriteProductionPaperOccupancy(
                        directory, reason, occupancy, capturedVoxelCount,
                        capturedCellCount, capturedVoxelSize, expectedTriangles,
                        capturedRasterizedTriangles));
                }
                catch (Exception e)
                {
                    failure = e;
                }
            }

            if (writeTask != null)
            {
                while (!writeTask.IsCompleted) yield return null;
                if (writeTask.IsFaulted)
                    failure = writeTask.Exception?.GetBaseException() ??
                        new IOException("production paper occupancy write failed");
                else if (writeTask.IsCanceled)
                    failure = new TaskCanceledException(
                        "production paper occupancy write canceled");
                else
                    outputPath = writeTask.Result;
            }

            cells?.Release();
            _paperOccupancyExportPending = false;
            if (failure == null)
                Logger.Info("原生5cm正式纸皮占用账已保存: " + outputPath);
            else
                Logger.Error("原生5cm正式纸皮占用账导出失败: " +
                    failure.GetType().Name + ": " + failure.Message);
            try { completed?.Invoke(outputPath); }
            catch (Exception callbackError)
            {
                Logger.Error("原生5cm正式纸皮占用账回调失败: " + callbackError.Message);
            }
        }

        private static string WriteProductionPaperOccupancy(
            string directory, string reason, uint[] occupancy,
            int3 voxelCount, int3 cellCount, float voxelSize,
            int productionTriangles, int rasterizedTriangles)
        {
            Directory.CreateDirectory(directory);
            long occupiedCells = 0;
            string binaryPath = Path.Combine(directory,
                "production_paper_occupancy.r32_uint.bin");
            using (var stream = new FileStream(binaryPath, FileMode.Create,
                FileAccess.Write, FileShare.None, 1 << 20))
            using (var writer = new BinaryWriter(stream))
            {
                for (int i = 0; i < occupancy.Length; i++)
                {
                    uint value = occupancy[i];
                    writer.Write(value);
                    if (value != 0xffffffffu) occupiedCells++;
                }
            }
            if (productionTriangles > 0 && occupiedCells == 0)
                throw new InvalidDataException(
                    "production paper contains triangles but occupancy is empty");

            string schema = "{\n" +
                "  \"schema\": \"scancover.production_paper_occupancy.v1\",\n" +
                "  \"reason\": \"" + JsonEscape(reason) + "\",\n" +
                "  \"voxelCount\": [" + voxelCount.x + "," + voxelCount.y + "," + voxelCount.z + "],\n" +
                "  \"cellCount\": [" + cellCount.x + "," + cellCount.y + "," + cellCount.z + "],\n" +
                "  \"paperStrideVoxels\": 1,\n" +
                "  \"voxelSizeMetres\": " + voxelSize.ToString("R", CultureInfo.InvariantCulture) + ",\n" +
                "  \"productionTriangles\": " + productionTriangles + ",\n" +
                "  \"rasterizedTriangles\": " + rasterizedTriangles + ",\n" +
                "  \"occupiedCells\": " + occupiedCells + ",\n" +
                "  \"emptyValue\": 4294967295,\n" +
                "  \"packing\": \"distanceMm14 in bits18..31; nearest triangle id18 in bits0..17\",\n" +
                "  \"indexOrder\": \"x-fastest, then y, then z\",\n" +
                "  \"source\": \"exact committed native-5cm production paper snapshots\",\n" +
                "  \"authority\": \"diagnostic only; never sampled by production\"\n" +
                "}\n";
            File.WriteAllText(Path.Combine(directory,
                "production_paper_occupancy_schema.json"), schema,
                new UTF8Encoding(false));
            return directory;
        }

        private static string JsonEscape(string value)
        {
            return (value ?? string.Empty).Replace("\\", "\\\\")
                .Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
        }

        private void OnDestroy()
        {
            if (DepthCapture.Instance != null)
                DepthCapture.Instance.Preprocessed -= OnDepthPreprocessed;
            ReleaseLayerArray(_liveLayers);
            ReleaseLayer(_frozenLayer);
            ReleaseReviewHistory();
            _verifiedWitnessEpochs?.Release();
            _verifiedWitnessEpochs = null;
            _paperAuditCells?.Release();
            _paperAuditCells = null;
            _relayMaturityCells?.Release();
            _relayMaturityCells = null;
            _paperAuditTriangles?.Release();
            _paperAuditTriangles = null;
            _paperAuditTriangleCapacity = 0;
            if (_mesh != null) Destroy(_mesh);
            if (_snapshotPackMaterial != null) Destroy(_snapshotPackMaterial);
        }

        private void ReleaseLayerArray(ShellLayer[] layers)
        {
            if (layers == null) return;
            for (int i = 0; i < layers.Length; i++) ReleaseLayer(layers[i]);
        }

        private void ReleaseLayer(ShellLayer layer)
        {
            if (layer == null) return;
            if (layer.Material != null) Destroy(layer.Material);
            if (layer.Snapshot != null)
            {
                layer.Snapshot.Release();
                Destroy(layer.Snapshot);
            }
            if (layer.ReviewState != null)
            {
                layer.ReviewState.Release();
                Destroy(layer.ReviewState);
            }
            if (layer.ReviewResidualMm != null)
            {
                layer.ReviewResidualMm.Release();
                Destroy(layer.ReviewResidualMm);
            }
            if (layer.ConnectivityState != null)
            {
                layer.ConnectivityState.Release();
                Destroy(layer.ConnectivityState);
            }
            if (layer.PlaneConnectivityState != null)
            {
                layer.PlaneConnectivityState.Release();
                Destroy(layer.PlaneConnectivityState);
            }
            layer.ReviewCounters?.Release();
        }

        private void ReleaseReviewHistory()
        {
            if (_reviewHistory == null) return;
            for (int i = 0; i < _reviewHistory.Length; i++)
            {
                RenderTexture snapshot = _reviewHistory[i]?.Snapshot;
                if (snapshot == null) continue;
                snapshot.Release();
                Destroy(snapshot);
            }
        }
    }
}

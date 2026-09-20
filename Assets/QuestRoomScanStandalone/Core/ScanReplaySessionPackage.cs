using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Meta.XR.MRUtilityKit;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.XR.CoreUtils;

namespace Genesis.RoomScan
{
    /// <summary>
    /// “可独立复现本次扫描”会话的总封装器。它只旁路复制生产链已经消费的输入、
    /// 裁决与结果。通常它没有生产权限；裁决准入实验仅复用最终 correspondence
    /// 的既有回读，发布 stableId 准证及获胜层射线归属描述，仍不生成深度、
    /// 距离场或网格。
    /// </summary>
    [DisallowMultipleComponent]
    internal sealed class ScanReplaySessionPackage : MonoBehaviour
    {
        internal const string Schema = "scancover.replay_session.v1";
        private const int RecordedEye = DepthCapture.FusionEyeIndex;
        private const int MaxOutstandingFusionFrames = 12;
        internal const int FinalCourtVerdictCapacity = 262144;
        private const float SystemRoomMeshStartupWaitSeconds = 2f;
        private const float SystemRoomMeshReloadWaitSeconds = 12f;
        private const int SealIdle = 0;
        private const int SealCapturing = 1;
        private const int SealDraining = 2;
        private const int SealHashing = 3;
        private const int SealComplete = 4;
        private const int SealIncomplete = 5;
        private const int SealFailed = 6;

        private readonly object _fileLock = new object();
        private readonly ConcurrentDictionary<string, string> _immutableFileHashes =
            new ConcurrentDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<int, ProbeStageCounts> _preProbeStages =
            new ConcurrentDictionary<int, ProbeStageCounts>();
        private string _sessionDirectory = string.Empty;
        private string _fusionDirectory = string.Empty;
        private string _fusionManifest = string.Empty;
        private string _probeConversionManifest = string.Empty;
        private string _runtimeTimelineManifest = string.Empty;
        private string _integrationDispatchManifest = string.Empty;
        private readonly StringBuilder _runtimeTimelineBuffer = new StringBuilder(128 * 1024);
        private readonly StringBuilder _integrationDispatchBuffer = new StringBuilder(64 * 1024);
        private const int RuntimeBufferFlushChars = 64 * 1024;
        private bool _active;
        private bool _endRequested;
        private bool _depthStreamsComplete;
        private bool _finalized;
        private Task _finalizeTask;
        private bool _finalizeClean;
        private int _fusionSequence;
        private int _fusionReadbacks;
        private int _fusionWrites;
        private int _fusionDropped;
        private int _fusionReadbackErrors;
        private int _fusionWriteErrors;
        private int _depthPairDrops;
        private int _depthPairReadbackErrors;
        private int _depthPairWriteErrors;
        private int _artifactRequests;
        private int _artifactFailures;
        private int _evidenceLineageBuildStarted;
        private int _surfaceFeatureArchiveBuildStarted;
        private int _probeConversionRows;
        private int _probeConversionMissingPreRows;
        private int _probeConversionWriteErrors;
        private int _runtimeLedgerWriteErrors;
        private bool _tsdfResponsibilityStarted;
        private VirtualProbeShadowLedger _probeShadowLedger;
        private VirtualProbeShadowLedger _probeFinalBufferLedger;
        private RuntimeFinalSurfaceCourt _finalSurfaceCourt;
        private readonly uint[] _finalCourtVerdictTable =
            new uint[FinalCourtVerdictCapacity];
        private readonly Vector4[] _finalCourtPlaneTable =
            new Vector4[FinalCourtVerdictCapacity];
        private readonly uint[] _finalCourtGenerationTable =
            new uint[FinalCourtVerdictCapacity];
        private ComputeBuffer _finalCourtVerdictBuffer;
        private ComputeBuffer _finalCourtPlaneBuffer;
        private ComputeBuffer _finalCourtGenerationBuffer;
        private int _finalCourtUploadedRevision = -1;
        private int _finalCourtUploadedPrefix = 1;
        private DirectProjectionChallengeShadowLedger _directProjectionShadowLedger;
        private string _startedUtc = string.Empty;
        private string _stoppedUtc = string.Empty;
        private int _sealPhase = SealIdle;
        private int _checksumFilesProcessed;
        private int _checksumFilesTotal;
        private string _finalizeIssue = string.Empty;

        internal static ScanReplaySessionPackage Active { get; private set; }
        internal static ScanReplaySessionPackage Latest { get; private set; }
        internal static bool FullOutputLedgerActive =>
            Active != null ||
            (Latest != null && Latest._endRequested && !Latest._finalized);
        internal bool IsActive => _active;
        internal int FinalCourtAdmissionRevision => _finalSurfaceCourt?.Revision ?? -1;
        internal VirtualProbeShadowAdjudicator.VerdictCounts FinalCourtAdmissionCounts =>
            _finalSurfaceCourt != null ? _finalSurfaceCourt.GetCounts() : default;

        internal int CopyFinalCourtProductPlanes(Bounds worldBounds,
            RuntimeFinalSurfaceCourt.ProductPlane[] destination)
        {
            return _active && _finalSurfaceCourt != null
                ? _finalSurfaceCourt.CopyProductPlanes(worldBounds, destination)
                : 0;
        }

        internal bool TryGetFinalCourtAdmissionBuffers(out ComputeBuffer verdicts,
            out ComputeBuffer planes, out ComputeBuffer generations)
        {
            // GPU admission is production state. Court testimony may arrive from
            // an AsyncGPUReadback callback, but ComputeBuffer.SetData must have
            // one deterministic owner on the production/main thread immediately
            // before the integrator binds these buffers.
            UploadFinalCourtAdmissionOnProductionThread();
            verdicts = _finalCourtVerdictBuffer;
            planes = _finalCourtPlaneBuffer;
            generations = _finalCourtGenerationBuffer;
            return _active && verdicts != null && planes != null &&
                   generations != null;
        }

        private void UploadFinalCourtAdmissionOnProductionThread()
        {
            if (!_active || _finalSurfaceCourt == null ||
                _finalCourtVerdictBuffer == null ||
                _finalCourtPlaneBuffer == null ||
                _finalCourtGenerationBuffer == null)
                return;
            int revision = _finalSurfaceCourt.Revision;
            if (revision == _finalCourtUploadedRevision) return;

            int usedPrefix = _finalSurfaceCourt.CopyAdmissionTables(
                _finalCourtVerdictTable, _finalCourtPlaneTable,
                _finalCourtGenerationTable, _finalCourtUploadedPrefix);
            int uploadCount = Mathf.Max(_finalCourtUploadedPrefix, usedPrefix);
            // Publish descriptor and generation first. The permit bit is last,
            // so an approved verdict can never expose a half-updated plane.
            _finalCourtPlaneBuffer.SetData(_finalCourtPlaneTable,
                0, 0, uploadCount);
            _finalCourtGenerationBuffer.SetData(_finalCourtGenerationTable,
                0, 0, uploadCount);
            _finalCourtVerdictBuffer.SetData(_finalCourtVerdictTable,
                0, 0, uploadCount);
            _finalCourtUploadedPrefix = usedPrefix;
            _finalCourtUploadedRevision = revision;
        }

        internal int DrainFinalCourtInvalidationRegions(Vector4[] destination)
        {
            return _active && _finalSurfaceCourt != null
                ? _finalSurfaceCourt.DrainInvalidationRegions(destination)
                : 0;
        }
        internal int OutstandingCount => Volatile.Read(ref _fusionReadbacks) +
                                         Volatile.Read(ref _fusionWrites) +
                                         Volatile.Read(ref _artifactRequests);
        internal int DroppedFusionFrames => Volatile.Read(ref _fusionDropped);
        internal bool IsSafelySealed
        {
            get
            {
                int phase = Volatile.Read(ref _sealPhase);
                return phase == SealComplete || phase == SealIncomplete;
            }
        }
        internal string SealHudFixed
        {
            get
            {
                int phase = Volatile.Read(ref _sealPhase);
                int total = Mathf.Max(0, Volatile.Read(ref _checksumFilesTotal));
                int processed = Mathf.Clamp(Volatile.Read(ref _checksumFilesProcessed), 0,
                    Mathf.Max(0, total));
                int percent = phase == SealComplete || phase == SealIncomplete
                    ? 100
                    : total > 0 ? Mathf.Clamp(Mathf.RoundToInt(100f * processed / total), 0, 99) : 0;
                string label = phase switch
                {
                    SealCapturing => "采集中",
                    SealDraining => "排空中",
                    SealHashing => "校验中",
                    SealComplete => "已封口",
                    SealIncomplete => "有损封口",
                    SealFailed => "封口失败",
                    _ => "未开始"
                };
                return $"封包[{label.PadRight(4, ' ')}] 校验{percent:000}% 安全退出[{(IsSafelySealed ? "是" : "否")}]";
            }
        }
        internal string ProbeFollowupGuidanceCompact => _probeShadowLedger != null
            ? (_directProjectionShadowLedger != null
                ? "足迹自由空间影子"
                : _probeShadowLedger.AdjudicatorGuidanceCompact)
            : "未就绪";
        internal string ProbeFollowupGuidanceHudFixed => _probeShadowLedger != null
            ? (_directProjectionShadowLedger != null
                ? _directProjectionShadowLedger.HudFixed
                : _probeShadowLedger.AdjudicatorGuidanceHudFixed)
            : ProbeFollowupGuidanceHudEmpty;
        internal VirtualProbeShadowAdjudicator.GuidanceTargetSnapshot
            ProbeFollowupGuidanceTarget => _probeShadowLedger != null
                ? (_directProjectionShadowLedger != null
                    ? default
                    : _probeShadowLedger.AdjudicatorGuidanceTarget)
                : default;
        internal VirtualProbeShadowAdjudicator.GuidanceFrameSnapshot
            ProbeFollowupGuidanceFrame => _probeShadowLedger != null
                ? (_directProjectionShadowLedger != null
                    ? default
                    : _probeShadowLedger.AdjudicatorGuidanceFrame)
                : default;
        internal int CopyProbeFollowupGuidanceVisuals(
            VirtualProbeShadowAdjudicator.GuidanceCellVisual[] destination)
        {
            return _probeShadowLedger != null && _directProjectionShadowLedger == null
                ? _probeShadowLedger.CopyAdjudicatorGuidanceVisuals(destination)
                : 0;
        }
        internal int PaperCorrectionRevision => _probeShadowLedger != null
            ? _probeShadowLedger.CorrectionRevision
            : -1;
        internal int CopyPaperCorrectionCells(
            List<VirtualProbeShadowAdjudicator.PaperCorrectionCell> destination)
        {
            return _probeShadowLedger != null
                ? _probeShadowLedger.CopyPaperCorrectionCells(destination)
                : 0;
        }

        internal const string ProbeFollowupGuidanceHudEmpty =
            "采样 总待---- 空证---- 面证---- 过期----\n" +
            "框内 待补---- 命中---- 已收---- 失效----\n" +
            "区域[等待双采] 保持物体在框内采集";

        internal void Begin(string sessionDirectory)
        {
            _sessionDirectory = sessionDirectory ?? string.Empty;
            _fusionDirectory = Path.Combine(_sessionDirectory, "fusion_inputs", "frames");
            _fusionManifest = Path.Combine(_sessionDirectory, "fusion_inputs", "manifest.csv");
            _probeConversionManifest = Path.Combine(_sessionDirectory, "probe_shadow",
                "final_buffer", "conversion_frames.csv");
            string timelineDirectory = Path.Combine(_sessionDirectory, "runtime_timeline");
            _runtimeTimelineManifest = Path.Combine(timelineDirectory, "frames.csv");
            _integrationDispatchManifest = Path.Combine(timelineDirectory,
                "integration_dispatches.csv");
            Directory.CreateDirectory(_fusionDirectory);
            Directory.CreateDirectory(timelineDirectory);
            Directory.CreateDirectory(Path.Combine(_sessionDirectory, "artifacts"));
            Directory.CreateDirectory(Path.Combine(_sessionDirectory, "system_reference"));
            _probeShadowLedger ??= new VirtualProbeShadowLedger();
            _probeShadowLedger.Begin(_sessionDirectory);
            _directProjectionShadowLedger ??=
                new DirectProjectionChallengeShadowLedger();
            _directProjectionShadowLedger.Begin(_sessionDirectory);
            _probeFinalBufferLedger ??= new VirtualProbeShadowLedger();
            _probeFinalBufferLedger.Begin(_sessionDirectory,
                Path.Combine("probe_shadow", "final_buffer"),
                "post-candidate-transaction correspondence buffer actually bound to guarded TSDF integration",
                "final_buffer_shadow_only");
            _finalSurfaceCourt ??= new RuntimeFinalSurfaceCourt();
            _finalSurfaceCourt.Begin(Path.Combine(_sessionDirectory,
                "probe_shadow", "final_buffer"));
            if (_finalCourtVerdictBuffer == null ||
                _finalCourtVerdictBuffer.count != FinalCourtVerdictCapacity)
            {
                _finalCourtVerdictBuffer?.Release();
                _finalCourtVerdictBuffer = new ComputeBuffer(
                    FinalCourtVerdictCapacity, sizeof(uint));
            }
            if (_finalCourtPlaneBuffer == null ||
                _finalCourtPlaneBuffer.count != FinalCourtVerdictCapacity)
            {
                _finalCourtPlaneBuffer?.Release();
                _finalCourtPlaneBuffer = new ComputeBuffer(
                    FinalCourtVerdictCapacity, sizeof(float) * 4);
            }
            if (_finalCourtGenerationBuffer == null ||
                _finalCourtGenerationBuffer.count != FinalCourtVerdictCapacity)
            {
                _finalCourtGenerationBuffer?.Release();
                _finalCourtGenerationBuffer = new ComputeBuffer(
                    FinalCourtVerdictCapacity, sizeof(uint));
            }
            Array.Clear(_finalCourtVerdictTable, 0, _finalCourtVerdictTable.Length);
            _finalCourtVerdictBuffer.SetData(_finalCourtVerdictTable);
            Array.Clear(_finalCourtPlaneTable, 0, _finalCourtPlaneTable.Length);
            Array.Clear(_finalCourtGenerationTable, 0,
                _finalCourtGenerationTable.Length);
            _finalCourtPlaneBuffer.SetData(_finalCourtPlaneTable, 0, 0, 1);
            _finalCourtGenerationBuffer.SetData(_finalCourtGenerationTable, 0, 0, 1);
            _finalCourtUploadedRevision = _finalSurfaceCourt.Revision;
            _finalCourtUploadedPrefix = 1;
            File.WriteAllText(_probeConversionManifest,
                "gunGelFrame,sourceFrame,preValid,postValid,preRaw,postRaw,preDual,postDual,preStableFound,postStableFound,preResidualPass,postResidualPass,preNormalPass,postNormalPass,preStableMatch,postStableMatch,preStableDual,postStableDual,preUnopposed,postUnopposed,preAuthority,postAuthority,deltaStableFound,deltaStableMatch,deltaAuthority,status\n",
                new UTF8Encoding(false));

            _startedUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
            _stoppedUtc = string.Empty;
            _fusionSequence = 0;
            _fusionDropped = 0;
            _fusionReadbackErrors = 0;
            _fusionWriteErrors = 0;
            _depthPairDrops = 0;
            _depthPairReadbackErrors = 0;
            _depthPairWriteErrors = 0;
            _endRequested = false;
            _depthStreamsComplete = false;
            _finalized = false;
            _finalizeTask = null;
            _finalizeClean = false;
            Volatile.Write(ref _sealPhase, SealCapturing);
            Volatile.Write(ref _checksumFilesProcessed, 0);
            Volatile.Write(ref _checksumFilesTotal, 0);
            _finalizeIssue = string.Empty;
            _immutableFileHashes.Clear();
            _evidenceLineageBuildStarted = 0;
            _surfaceFeatureArchiveBuildStarted = 0;
            _probeConversionRows = 0;
            _probeConversionMissingPreRows = 0;
            _probeConversionWriteErrors = 0;
            _runtimeLedgerWriteErrors = 0;
            _tsdfResponsibilityStarted = VolumeIntegrator.Instance != null &&
                                         VolumeIntegrator.Instance.BeginTsdfResponsibilityCapture();
            _artifactFailures = 0;
            _preProbeStages.Clear();
            _active = true;
            Active = this;
            Latest = this;

            File.WriteAllText(_fusionManifest,
                "sequence,attemptIndex,sourceFrame,unityFrame,scaledTime,unscaledTime,accepted,decision,guarded,gunGelAdmissionActive,gunGelFrame,translationMm,rotationDeg,angularDegPerSec,linearMps,motionQuality,fusionHeadAvailable,fusionHeadX,fusionHeadY,fusionHeadZ,fusionHeadQx,fusionHeadQy,fusionHeadQz,fusionHeadQw,fusionHeadPitchDeg,fusionHeadYawDeg,fusionHeadRollDeg,fusionEyeAvailable,fusionEyeX,fusionEyeY,fusionEyeZ,fusionEyeForwardX,fusionEyeForwardY,fusionEyeForwardZ,headToFusionEyeMm,cameraAvailable,cameraPitchDeg,cameraYawDeg,cameraRollDeg,depthFile,normalFile,dilatedFile,edgeReasonFile,temporalReasonFile,cameraFile,gunGelObservationsFile,gunGelCorrespondencesFile,metaFile,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(_runtimeTimelineManifest,
                "unityFrame,scaledTime,unscaledTime,deltaMs,fps,headAvailable,headX,headY,headZ,headQx,headQy,headQz,headQw,headPitchDeg,headYawDeg,headRollDeg,platformFrame,angularDegPerSec,linearMps,integrationCount,dirtyEpoch,paperVisible,paperTriangles,paperBuiltChunks,paperTotalChunks,paperQueuedChunks,paperCommitPending,paperQueueToCommitMs,paperDispatchToCallbackMs,fusionOutstanding,fusionDropped,sealPhase\n",
                new UTF8Encoding(false));
            File.WriteAllText(_integrationDispatchManifest,
                "attemptIndex,sourceFrame,integrationCount,dirtyEpoch,unityFrame,scaledTime,unscaledTime,headAvailable,headX,headY,headZ,headQx,headQy,headQz,headQw,headPitchDeg,headYawDeg,headRollDeg,angularDegPerSec,linearMps\n",
                new UTF8Encoding(false));
            _runtimeTimelineBuffer.Clear();
            _integrationDispatchBuffer.Clear();
            File.WriteAllText(Path.Combine(_sessionDirectory, "session_schema.json"),
                BuildSessionSchemaJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "replay_contract.json"),
                BuildReplayContractJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "full_output_index.json"),
                BuildFullOutputIndexJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "README.txt"),
                "ScanCover 独立回放会话\n" +
                "1. capture_complete.json 不存在时，本会话尚未封口，不得作为确定性样本。\n" +
                "2. HUD 必须显示‘已封口/安全退出[是]’后才能退出应用；hashing 仍是封口过程。\n" +
                "3. 先运行 Tools/ScanCoverReplaySession.py 校验全部 SHA-256。\n" +
                "4. depth_pairs 保存平台前处理前/QRS 后处理后同帧右眼深度，并给出有效率、距离分布、前后差及 callback→preprocess 位姿差。\n" +
                "5. fusion_inputs 保存实际接纳帧的深度、法线、膨胀供体、判因纹理、枪胶逐点缓冲及实际 RGB/相机参数；同一观察索引可连接事务前 A 身份与最终 B 身份。\n" +
                "6. probe_shadow 保存事务前 A 账；final_buffer 保存事务后 B 账；direct45 保存候选不确定足迹对同帧 raw/post 深度的保守复核。生产稳定候选达到旧三票时只进入待退场：0.25度内相反证词互相失效；两张相隔至少1度的支撑取消退场，两张相隔至少1度的全足迹证空才执行删除，冲突或不足均保留。gungel_candidate_audit/candidates.csv 保存仍存活的待退场证词；retirements.csv 为每次真实退场封存当时的支撑/证空帧号、两票夹角与裁决身份；candidate_summary.json 对进入、支撑取消、接班取消、身份重开、证空删除和封包仍待定做守恒对账。旧16帧、8帧幽灵及0.5/1/2度赛道仍只做零权限事后审计。相机位移不超过10毫米且视线变化不超过0.25度的样本只用于估计静止重复稳定性，不代表绝对精度。\n" +
                "7. artifacts/surface_feature_archive 是按候选世代建立的隔离档案；坐标只作会话内追溯，跨会话只能按连续特征归纳。\n" +
                "8. production_config 与 coordinate_contract 是回放契约；禁止用默认参数替代。\n" +
                "9. runtime_timeline/frames.csv 连续记录帧率、头姿、距离位移、深度帧号、融合 epoch、纸皮队列/提交/显示状态；integration_dispatches.csv 将融合 attempt/sourceFrame 精确接到 dirtyEpoch。\n" +
                "10. artifacts/paper_audit/production_paper_replacements.csv 记录最终生产纸皮每次整块替换时的姿态、距离、融合状态与旧新占用变化；surface ledger 给出平面残差及相邻页接缝；production_paper_occupancy 是不依赖显示档位的正式5cm纸皮占用。\n" +
                "11. system_reference/status.json 说明外部系统网格交接状态；旧 Scene 导出的 OBJ 仅作对照，不是生产真值。\n" +
                "12. artifacts/tsdf_responsibility 保存最终 TSDF 与逐体素当前生命周期的出生、最后几何改写、最后支撑改写、最强受阻纠正；integrationCount 可反查到实际 sourceFrame，并把无纸皮格追到 TSDF 前/内、零交叉、拓扑或发布边界。\n" +
                "13. final_surface_court/frame_gate_summary.csv 记录每帧各拒绝关卡汇总；independent_testimonies.csv 记录真正入档的独立证词；decision_checks.csv 记录每次判决的假设、残差、票份额、滞回和冷却结果。它们不逐像素写日志。\n" +
                "14. production_paper_quality_timeline.csv 记录每次正式块提交时各4x4x4小区的 crossing/raw/final 平整度，区分入场已凹凸、后续抚平、长期残留和恶化。\n" +
                "15. 将已封口 session 复制到电脑后运行 Tools/AnalyzeFullOutputLedger.ps1 -SessionDirectory <session>；它会生成 artifacts/responsibility_ledger 四张总账，并联合角度、距离、姿态、运动、候选、裁判、TSDF与纸皮，禁止单因素冒名定罪。\n",
                new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "production_config.json"),
                BuildProductionConfigJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "coordinate_contract.json"),
                BuildCoordinateContractJson("start"), new UTF8Encoding(false));
            WriteState("capturing");
        }

        internal void End()
        {
            if (!_active || _endRequested) return;
            _active = false;
            _endRequested = true;
            Volatile.Write(ref _sealPhase, SealDraining);
            _stoppedUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
            FlushRuntimeLedgers();
            if (Active == this) Active = null;
            _probeShadowLedger?.End();
            _directProjectionShadowLedger?.End();
            RequestSurfaceFeatureArchiveBuild();

            try
            {
                File.WriteAllText(Path.Combine(_sessionDirectory, "coordinate_contract_stop.json"),
                    BuildCoordinateContractJson("stop"), new UTF8Encoding(false));
            }
            catch (Exception e)
            {
                WriteArtifactStatus("coordinate_stop", "failed", e.Message, string.Empty);
            }

            CaptureOnlineArtifacts();
            // 系统房间网格由独立的 Meta Scene Mesh 场景导出。它不是本次生产会话
            // 的必要组成，也不得再卡住 depth/fusion/artifacts 已经排空后的封口。
            WriteSystemReferenceStatus(
                "external_required",
                "Export separately from Assets/MRMotifs/ScanCover/Scene/Meta Scene Mesh.unity",
                0, 0, 0);
            WriteState("draining");
        }

        private void OnDestroy()
        {
            _finalSurfaceCourt?.Dispose();
            _finalSurfaceCourt = null;
            _finalCourtVerdictBuffer?.Release();
            _finalCourtVerdictBuffer = null;
            _finalCourtPlaneBuffer?.Release();
            _finalCourtPlaneBuffer = null;
            _finalCourtGenerationBuffer?.Release();
            _finalCourtGenerationBuffer = null;
        }

        internal void NotifyDepthStreamsComplete(
            int droppedPairs, int readbackErrors, int writeErrors)
        {
            _depthPairDrops = Mathf.Max(0, droppedPairs);
            _depthPairReadbackErrors = Mathf.Max(0, readbackErrors);
            _depthPairWriteErrors = Mathf.Max(0, writeErrors);
            _depthStreamsComplete = true;
            TryFinalize();
        }

        private void Update()
        {
            if (_active || (_endRequested && !_finalized && _finalizeTask == null))
                RecordRuntimeFrame();
            if (_finalizeTask != null && _finalizeTask.IsCompleted)
            {
                Task completed = _finalizeTask;
                _finalizeTask = null;
                _finalized = true;
                if (completed.IsFaulted)
                {
                    string issue = completed.Exception?.GetBaseException().Message ?? "unknown finalize failure";
                    Logger.Warning("独立回放会话封口失败：" + issue);
                }
                else
                {
                    Logger.Info((_finalizeClean ? "独立回放会话已封口：" : "独立回放会话已排空但存在采集损失：") +
                                _sessionDirectory);
                }
            }
            if (_endRequested && !_finalized)
                TryFinalize();
        }

        internal void RecordIntegrationDispatch(
            int attemptIndex, int sourceFrame, int integrationCount, uint dirtyEpoch)
        {
            if (!_active) return;
            Camera head = Camera.main;
            DepthCapture depth = DepthCapture.Instance;
            Vector3 position = head != null ? head.transform.position : Vector3.zero;
            Quaternion rotation = head != null ? head.transform.rotation : Quaternion.identity;
            Vector3 euler = head != null ? head.transform.eulerAngles : Vector3.zero;
            _integrationDispatchBuffer
                .Append(attemptIndex).Append(',').Append(sourceFrame).Append(',')
                .Append(integrationCount).Append(',').Append(dirtyEpoch).Append(',')
                .Append(Time.frameCount).Append(',').Append(D(Time.timeAsDouble)).Append(',')
                .Append(D(Time.unscaledTimeAsDouble)).Append(',').Append(head != null ? 1 : 0).Append(',')
                .Append(F(position.x)).Append(',').Append(F(position.y)).Append(',').Append(F(position.z)).Append(',')
                .Append(F(rotation.x)).Append(',').Append(F(rotation.y)).Append(',')
                .Append(F(rotation.z)).Append(',').Append(F(rotation.w)).Append(',')
                .Append(F(euler.x)).Append(',').Append(F(euler.y)).Append(',').Append(F(euler.z)).Append(',')
                .Append(F(depth != null ? depth.SmoothedDepthAngularSpeed : 0f)).Append(',')
                .Append(F(depth != null ? depth.SmoothedDepthLinearSpeed : 0f)).AppendLine();
            FlushRuntimeLedgersIfNeeded();
        }

        private void RecordRuntimeFrame()
        {
            Camera head = Camera.main;
            DepthCapture depth = DepthCapture.Instance;
            VolumeIntegrator volume = VolumeIntegrator.Instance;
            MeshExtractor mesh = MeshExtractor.Instance;
            Vector3 position = head != null ? head.transform.position : Vector3.zero;
            Quaternion rotation = head != null ? head.transform.rotation : Quaternion.identity;
            Vector3 euler = head != null ? head.transform.eulerAngles : Vector3.zero;
            float delta = Time.unscaledDeltaTime;
            _runtimeTimelineBuffer
                .Append(Time.frameCount).Append(',').Append(D(Time.timeAsDouble)).Append(',')
                .Append(D(Time.unscaledTimeAsDouble)).Append(',').Append(F(delta * 1000f)).Append(',')
                .Append(F(delta > 1e-6f ? 1f / delta : 0f)).Append(',')
                .Append(head != null ? 1 : 0).Append(',')
                .Append(F(position.x)).Append(',').Append(F(position.y)).Append(',').Append(F(position.z)).Append(',')
                .Append(F(rotation.x)).Append(',').Append(F(rotation.y)).Append(',')
                .Append(F(rotation.z)).Append(',').Append(F(rotation.w)).Append(',')
                .Append(F(euler.x)).Append(',').Append(F(euler.y)).Append(',').Append(F(euler.z)).Append(',')
                .Append(depth != null ? depth.CurrentPlatformFrame : -1).Append(',')
                .Append(F(depth != null ? depth.SmoothedDepthAngularSpeed : 0f)).Append(',')
                .Append(F(depth != null ? depth.SmoothedDepthLinearSpeed : 0f)).Append(',')
                .Append(volume != null ? volume.IntegrationCount : -1).Append(',')
                .Append(volume != null ? volume.DirtyEpoch : 0u).Append(',')
                .Append(mesh != null && mesh.ProductionPaperVisible ? 1 : 0).Append(',')
                .Append(mesh != null ? mesh.ProductionPaperCommittedTriangleCount : 0).Append(',')
                .Append(mesh != null ? mesh.ProductionPaperBuiltChunkCount : 0).Append(',')
                .Append(mesh != null ? mesh.ProductionPaperChunkCount : 0).Append(',')
                .Append(mesh != null ? mesh.ProductionPaperPendingChunkCount : 0).Append(',')
                .Append(mesh != null ? mesh.ProductionPaperCommitPendingCount : 0).Append(',')
                .Append(F(mesh != null ? mesh.ProductionPaperAvgQueueToCommitMs : -1f)).Append(',')
                .Append(F(mesh != null ? mesh.ProductionPaperAvgDispatchToCallbackMs : -1f)).Append(',')
                .Append(OutstandingCount).Append(',').Append(DroppedFusionFrames).Append(',')
                .Append(Volatile.Read(ref _sealPhase)).AppendLine();
            FlushRuntimeLedgersIfNeeded();
        }

        private void FlushRuntimeLedgersIfNeeded()
        {
            if (_runtimeTimelineBuffer.Length < RuntimeBufferFlushChars &&
                _integrationDispatchBuffer.Length < RuntimeBufferFlushChars)
                return;
            FlushRuntimeLedgers();
        }

        private void FlushRuntimeLedgers()
        {
            lock (_fileLock)
            {
                if (_runtimeTimelineBuffer.Length > 0)
                {
                    try
                    {
                        File.AppendAllText(_runtimeTimelineManifest,
                            _runtimeTimelineBuffer.ToString(), new UTF8Encoding(false));
                    }
                    catch
                    {
                        Interlocked.Increment(ref _runtimeLedgerWriteErrors);
                    }
                    _runtimeTimelineBuffer.Clear();
                }
                if (_integrationDispatchBuffer.Length > 0)
                {
                    try
                    {
                        File.AppendAllText(_integrationDispatchManifest,
                            _integrationDispatchBuffer.ToString(), new UTF8Encoding(false));
                    }
                    catch
                    {
                        Interlocked.Increment(ref _runtimeLedgerWriteErrors);
                    }
                    _integrationDispatchBuffer.Clear();
                }
            }
        }

        internal void RecordDecisionOnly(
            int attemptIndex,
            int sourceFrame,
            bool accepted,
            string decision,
            bool guarded,
            int gunGelFrame,
            float translationMm,
            float rotationDeg,
            float angularSpeed,
            float linearSpeed,
            float motionQuality)
        {
            if (!_active) return;
            int sequence = Interlocked.Increment(ref _fusionSequence);
            var record = new FusionRecord(sequence, attemptIndex, sourceFrame, accepted,
                decision, guarded, gunGelFrame, translationMm, rotationDeg,
                angularSpeed, linearSpeed, motionQuality, null, null, null, null, null,
                false, false, Vector3.zero, Quaternion.identity, Vector2.zero,
                Vector2.zero, Vector2.zero, Vector2.zero, Matrix4x4.identity, null, 0);
            AppendFusionManifest(record, string.Empty, string.Empty, string.Empty,
                string.Empty, string.Empty, string.Empty, string.Empty, string.Empty, string.Empty,
                "decision_only");
        }

        /// <summary>
        /// 记录枪胶 CPU 校枪回读上可直接观察到的虚拟探针连续量。调用方仍持有
        /// 原始 NativeArray，本方法同步汇总后立即返回；账本不拥有也不修改缓冲。
        /// </summary>
        internal void RecordVirtualProbeShadow(
            int gunGelFrame,
            int platformFrame,
            NativeArray<GunGelEvidenceShadow.Correspondence> correspondences,
            int gridX,
            int gridY,
            int pixelStride,
            int depthWidth,
            int depthHeight,
            Matrix4x4 sourceProjectionInverse,
            Matrix4x4 sourceViewInverse,
            float angularSpeed,
            float linearSpeed,
            float motionQuality)
        {
            if (!_active || _probeShadowLedger == null) return;
            _preProbeStages[gunGelFrame] = CountProbeStages(correspondences);
            _probeShadowLedger.Record(gunGelFrame, platformFrame, correspondences,
                gridX, gridY, pixelStride, depthWidth, depthHeight,
                sourceProjectionInverse, sourceViewInverse,
                angularSpeed, linearSpeed, motionQuality);
        }

        /// <summary>
        /// 记录候选不确定足迹直投到同帧 raw/post 深度后的 GPU 影子结果。输入只同步
        /// 解码、记账，绝不回写枪胶、TSDF、纸皮或显示缓冲。
        /// </summary>
        internal void RecordDirectProjectionShadow(int gunGelFrame,
            int platformFrame, NativeArray<Unity.Mathematics.uint4> rows,
            float angularSpeed, float linearSpeed, float motionQuality)
        {
            if (!_active || _directProjectionShadowLedger == null) return;
            _directProjectionShadowLedger.Record(gunGelFrame, platformFrame,
                rows, angularSpeed, linearSpeed, motionQuality);
        }

        internal void RecordDirectProjectionShadowReadbackError()
        {
            if (!_active || _directProjectionShadowLedger == null) return;
            _directProjectionShadowLedger.RecordReadbackError();
        }

        internal void RecordAcceptedInput(
            int attemptIndex,
            int sourceFrame,
            string decision,
            bool guarded,
            int gunGelFrame,
            float translationMm,
            float rotationDeg,
            float angularSpeed,
            float linearSpeed,
            float motionQuality,
            Texture depth,
            Texture normal,
            Texture dilated,
            Texture edgeReason,
            Texture temporalReason,
            Matrix4x4[] projection,
            Matrix4x4[] view,
            Matrix4x4[] projectionInverse,
            Matrix4x4[] viewInverse,
            bool gunGelAdmissionActive,
            ComputeBuffer gunGelObservations,
            ComputeBuffer gunGelCorrespondences,
            ComputeBuffer gunGelPreTransactionCorrespondenceIdentity,
            ComputeBuffer gunGelCorrespondenceIdentity,
            int gunGelGridX,
            int gunGelGridY,
            int gunGelPixelStride,
            bool cameraAvailable,
            Texture cameraRgb,
            Vector3 cameraPosition,
            Quaternion cameraRotation,
            Vector2 cameraFocalLength,
            Vector2 cameraPrincipalPoint,
            Vector2 cameraSensorResolution,
            Vector2 cameraCurrentResolution,
            Matrix4x4 fusionCorrection,
            Vector4[] exclusionPositions,
            int exclusionCount)
        {
            if (!_active) return;
            if (OutstandingCount >= MaxOutstandingFusionFrames * 8)
            {
                Interlocked.Increment(ref _fusionDropped);
                RecordDecisionOnly(attemptIndex, sourceFrame, false,
                    "capture_queue_full:" + decision, guarded, gunGelFrame,
                    translationMm, rotationDeg, angularSpeed, linearSpeed, motionQuality);
                return;
            }

            int sequence = Interlocked.Increment(ref _fusionSequence);
            var record = new FusionRecord(sequence, attemptIndex, sourceFrame, true,
                decision, guarded, gunGelFrame, translationMm, rotationDeg,
                angularSpeed, linearSpeed, motionQuality,
                CloneMatrices(projection), CloneMatrices(view),
                CloneMatrices(projectionInverse), CloneMatrices(viewInverse),
                new PendingTextureSet(), gunGelAdmissionActive, cameraAvailable,
                cameraPosition, cameraRotation, cameraFocalLength,
                cameraPrincipalPoint, cameraSensorResolution, cameraCurrentResolution,
                fusionCorrection, CloneVectors(exclusionPositions, exclusionCount),
                Mathf.Clamp(exclusionCount, 0, 64),
                gunGelGridX, gunGelGridY, gunGelPixelStride);

            RequestTexture(record, "depth", depth, RecordedEye);
            RequestTexture(record, "normal", normal, RecordedEye);
            RequestTexture(record, "dilated", dilated, 0);
            RequestTexture(record, "edge", edgeReason, RecordedEye);
            if (temporalReason != null)
                RequestTexture(record, "temporal", temporalReason, RecordedEye);
            else
                record.Textures.temporalDone = true;
            if (cameraAvailable && cameraRgb != null)
                RequestTexture(record, "camera", cameraRgb, 0);
            else
                record.Textures.cameraDone = true;
            if (gunGelAdmissionActive && gunGelObservations != null &&
                gunGelCorrespondences != null)
            {
                RequestBuffer(record, "gungel_observations", gunGelObservations);
                RequestBuffer(record, "gungel_correspondences", gunGelCorrespondences);
                if (gunGelPreTransactionCorrespondenceIdentity != null)
                    RequestBuffer(record, "gungel_pre_transaction_identity",
                        gunGelPreTransactionCorrespondenceIdentity);
                else
                    record.Textures.gunGelPreTransactionCorrespondenceIdentityDone = true;
                if (gunGelCorrespondenceIdentity != null)
                    RequestBuffer(record, "gungel_correspondence_identity",
                        gunGelCorrespondenceIdentity);
                else
                    record.Textures.gunGelCorrespondenceIdentityDone = true;
            }
            else
            {
                record.Textures.gunGelObservationsDone = true;
                record.Textures.gunGelCorrespondencesDone = true;
                record.Textures.gunGelPreTransactionCorrespondenceIdentityDone = true;
                record.Textures.gunGelCorrespondenceIdentityDone = true;
            }
            TryCompleteFusion(record);
        }

        private void RequestBuffer(FusionRecord record, string role, ComputeBuffer buffer)
        {
            record.Textures.SetBufferDescriptor(role, buffer.count, buffer.stride);
            Interlocked.Increment(ref _fusionReadbacks);
            try
            {
                AsyncGPUReadback.Request(buffer, request =>
                {
                    try
                    {
                        if (request.hasError)
                            record.Textures.MarkError(role);
                        else
                        {
                            if (role == "gungel_correspondences")
                            {
                                record.Textures.gunGelCorrespondenceRows =
                                    request.GetData<GunGelEvidenceShadow.Correspondence>()
                                        .ToArray();
                            }
                            else if (role == "gungel_correspondence_identity")
                            {
                                record.Textures.gunGelCorrespondenceIdentityRows =
                                    request.GetData<Unity.Mathematics.uint4>().ToArray();
                            }
                            record.Textures.SetBytes(role, request.GetData<byte>().ToArray());
                        }
                    }
                    catch
                    {
                        record.Textures.MarkError(role);
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _fusionReadbacks);
                        TryCompleteFusion(record);
                    }
                });
            }
            catch
            {
                Interlocked.Decrement(ref _fusionReadbacks);
                record.Textures.MarkError(role);
                TryCompleteFusion(record);
            }
        }

        private void RequestTexture(FusionRecord record, string role, Texture texture, int requestedSlice)
        {
            if (texture == null)
            {
                record.Textures.MarkMissing(role);
                return;
            }

            int slice = texture.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray
                ? Mathf.Clamp(requestedSlice, 0, Mathf.Max(0, GetTextureDepth(texture) - 1))
                : 0;
            var descriptor = new TextureDescriptor(texture.width, texture.height,
                texture.graphicsFormat.ToString(), texture.dimension.ToString(), slice);
            record.Textures.SetDescriptor(role, descriptor);
            Interlocked.Increment(ref _fusionReadbacks);
            try
            {
                AsyncGPUReadback.Request(texture, 0,
                    0, texture.width, 0, texture.height, slice, 1,
                    request =>
                    {
                        try
                        {
                            if (request.hasError)
                                record.Textures.MarkError(role);
                            else
                                record.Textures.SetBytes(role, request.GetData<byte>().ToArray());
                        }
                        catch
                        {
                            record.Textures.MarkError(role);
                        }
                        finally
                        {
                            Interlocked.Decrement(ref _fusionReadbacks);
                            TryCompleteFusion(record);
                        }
                    });
            }
            catch
            {
                Interlocked.Decrement(ref _fusionReadbacks);
                record.Textures.MarkError(role);
                TryCompleteFusion(record);
            }
        }

        private static int GetTextureDepth(Texture texture)
        {
            if (texture is RenderTexture renderTexture) return renderTexture.volumeDepth;
            if (texture is Texture2DArray array) return array.depth;
            return 1;
        }

        private void TryCompleteFusion(FusionRecord record)
        {
            if (!record.Textures.AllDone || record.CompletionClaimed) return;
            record.CompletionClaimed = true;
            RecordFinalBufferProbe(record);
            Interlocked.Increment(ref _fusionWrites);
            _ = Task.Run(() => WriteFusionRecord(record));
        }

        private void RecordFinalBufferProbe(FusionRecord record)
        {
            if (_probeFinalBufferLedger == null || !record.GunGelAdmissionActive ||
                record.Textures.gunGelCorrespondenceRows == null)
                return;
            GunGelEvidenceShadow.Correspondence[] rows =
                record.Textures.gunGelCorrespondenceRows;
            var nativeRows = new NativeArray<GunGelEvidenceShadow.Correspondence>(
                rows, Allocator.Temp);
            try
            {
                Matrix4x4 projectionInverse = record.ProjectionInverse != null &&
                    record.ProjectionInverse.Length > RecordedEye
                    ? record.ProjectionInverse[RecordedEye]
                    : Matrix4x4.identity;
                Matrix4x4 viewInverse = record.ViewInverse != null &&
                    record.ViewInverse.Length > RecordedEye
                    ? record.ViewInverse[RecordedEye]
                    : Matrix4x4.identity;
                _probeFinalBufferLedger.Record(record.GunGelFrame,
                    record.SourceFrame, nativeRows,
                    record.GunGelGridX, record.GunGelGridY,
                    record.GunGelPixelStride,
                    record.Textures.depthDescriptor.width,
                    record.Textures.depthDescriptor.height,
                    projectionInverse, viewInverse,
                    record.AngularSpeed, record.LinearSpeed,
                    record.MotionQuality);
                if (record.Textures.gunGelCorrespondenceIdentityRows != null)
                {
                    _finalSurfaceCourt?.Record(record.GunGelFrame,
                        record.SourceFrame, record.AttemptIndex, rows,
                        record.Textures.gunGelCorrespondenceIdentityRows,
                        viewInverse, record.FusionCorrection,
                        record.AngularSpeed, record.LinearSpeed,
                        record.MotionQuality, record.FusionHeadEuler);
                }
                RecordProbeConversion(record, nativeRows);
            }
            finally
            {
                nativeRows.Dispose();
            }
        }

        private void WriteFusionRecord(FusionRecord record)
        {
            string stem = "fusion_" + record.Sequence.ToString("D6", CultureInfo.InvariantCulture);
            string depthName = string.Empty;
            string normalName = string.Empty;
            string dilatedName = string.Empty;
            string edgeName = string.Empty;
            string temporalName = string.Empty;
            string cameraName = string.Empty;
            string observationsName = string.Empty;
            string correspondencesName = string.Empty;
            string preTransactionIdentityName = string.Empty;
            string correspondenceIdentityName = string.Empty;
            string metaName = stem + "_meta.json";
            string status = record.Textures.HasError ? "partial_readback_error" : "ok";
            if (record.Textures.HasError)
                Interlocked.Increment(ref _fusionReadbackErrors);
            try
            {
                depthName = WriteTextureFile(stem, "depth", record.Textures.depth);
                normalName = WriteTextureFile(stem, "normal", record.Textures.normal);
                dilatedName = WriteTextureFile(stem, "dilated", record.Textures.dilated);
                edgeName = WriteTextureFile(stem, "edge_reason", record.Textures.edge);
                temporalName = WriteTextureFile(stem, "temporal_reason", record.Textures.temporal);
                cameraName = WriteTextureFile(stem, "camera_rgb", record.Textures.camera);
                observationsName = WriteTextureFile(stem, "gungel_observations", record.Textures.gunGelObservations);
                correspondencesName = WriteTextureFile(stem, "gungel_correspondences", record.Textures.gunGelCorrespondences);
                preTransactionIdentityName = WriteTextureFile(stem,
                    "gungel_pre_transaction_identity",
                    record.Textures.gunGelPreTransactionCorrespondenceIdentity);
                correspondenceIdentityName = WriteTextureFile(stem,
                    "gungel_correspondence_identity",
                    record.Textures.gunGelCorrespondenceIdentity);
                WriteImmutableText(Path.Combine(_fusionDirectory, metaName),
                    BuildFusionMetaJson(record, depthName, normalName, dilatedName,
                        edgeName, temporalName, cameraName, observationsName,
                        correspondencesName, preTransactionIdentityName,
                        correspondenceIdentityName,
                        status));
                AppendFusionManifest(record, depthName, normalName, dilatedName,
                    edgeName, temporalName, cameraName, observationsName, correspondencesName,
                    metaName, status);
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _fusionWriteErrors);
                try
                {
                    AppendFusionManifest(record, depthName, normalName, dilatedName,
                        edgeName, temporalName, cameraName, observationsName, correspondencesName,
                        string.Empty, "write_error:" + Csv(e.GetType().Name));
                }
                catch { }
            }
            finally
            {
                Interlocked.Decrement(ref _fusionWrites);
            }
        }

        private string WriteTextureFile(string stem, string role, byte[] bytes)
        {
            if (bytes == null || bytes.Length == 0) return string.Empty;
            string name = stem + "_" + role + ".bin";
            WriteImmutableBytes(Path.Combine(_fusionDirectory, name), bytes);
            return name;
        }

        /// <summary>
        /// 写入之后不会再变化的大文件在内存中顺手计算 SHA-256。封口时直接复用，
        /// 避免 Quest 再把整套逐帧深度和融合纹理从存储读一遍。
        /// </summary>
        internal void WriteImmutableBytes(string path, byte[] bytes)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));
            File.WriteAllBytes(path, bytes);
            RegisterImmutableHash(path, bytes);
        }

        internal void WriteImmutableText(string path, string text)
        {
            byte[] bytes = new UTF8Encoding(false).GetBytes(text ?? string.Empty);
            WriteImmutableBytes(path, bytes);
        }

        private void RegisterImmutableHash(string path, byte[] bytes)
        {
            using SHA256 sha = SHA256.Create();
            _immutableFileHashes[Path.GetFullPath(path)] = Hex(sha.ComputeHash(bytes));
        }

        private void AppendFusionManifest(FusionRecord r, string depth, string normal,
            string dilated, string edge, string temporal, string camera, string observations,
            string correspondences, string meta, string status)
        {
            string row = string.Join(",",
                r.Sequence.ToString(CultureInfo.InvariantCulture),
                r.AttemptIndex.ToString(CultureInfo.InvariantCulture),
                r.SourceFrame.ToString(CultureInfo.InvariantCulture),
                r.UnityFrame.ToString(CultureInfo.InvariantCulture),
                D(r.ScaledTime), D(r.UnscaledTime),
                r.Accepted ? "1" : "0",
                Csv(r.Decision),
                r.Guarded ? "1" : "0",
                r.GunGelAdmissionActive ? "1" : "0",
                r.GunGelFrame.ToString(CultureInfo.InvariantCulture),
                F(r.TranslationMm), F(r.RotationDeg), F(r.AngularSpeed),
                F(r.LinearSpeed), F(r.MotionQuality), r.FusionHeadAvailable ? "1" : "0",
                F(r.FusionHeadPosition.x), F(r.FusionHeadPosition.y),
                F(r.FusionHeadPosition.z), F(r.FusionHeadRotation.x),
                F(r.FusionHeadRotation.y), F(r.FusionHeadRotation.z),
                F(r.FusionHeadRotation.w), F(r.FusionHeadEuler.x),
                F(r.FusionHeadEuler.y), F(r.FusionHeadEuler.z),
                r.FusionEyeAvailable ? "1" : "0",
                F(r.FusionEyePosition.x), F(r.FusionEyePosition.y),
                F(r.FusionEyePosition.z), F(r.FusionEyeForward.x),
                F(r.FusionEyeForward.y), F(r.FusionEyeForward.z),
                F(r.HeadToFusionEyeMm), r.CameraAvailable ? "1" : "0",
                F(r.CameraEuler.x), F(r.CameraEuler.y), F(r.CameraEuler.z),
                depth, normal, dilated, edge, temporal, camera, observations,
                correspondences, meta, Csv(status)) + "\n";
            lock (_fileLock)
                File.AppendAllText(_fusionManifest, row, new UTF8Encoding(false));
        }

        private void CaptureOnlineArtifacts()
        {
            try
            {
                MeshExtractor mesh = MeshExtractor.Instance;
                if (mesh != null)
                {
                    string geometry = mesh.ExportFirstStageGeometrySnapshot("独立回放会话停止");
                    if (!string.IsNullOrEmpty(geometry))
                    {
                        Interlocked.Increment(ref _artifactRequests);
                        Task.Run(() =>
                        {
                            try { CopyGeometryFamily(geometry, Path.Combine(_sessionDirectory, "artifacts", "geometry_stage1")); }
                            finally { Interlocked.Decrement(ref _artifactRequests); }
                        });
                    }

                    Interlocked.Increment(ref _artifactRequests);
                    bool requested = mesh.RequestPaperAuditExport("独立回放会话停止", path =>
                    {
                        Task.Run(() =>
                        {
                            try { CopyArtifact(path, "paper_audit"); }
                            finally { Interlocked.Decrement(ref _artifactRequests); }
                        });
                    });
                    if (!requested)
                    {
                        Interlocked.Decrement(ref _artifactRequests);
                        WriteArtifactStatus("paper_audit", "unavailable", "paper audit not ready", string.Empty);
                        WriteArtifactStatus("evidence_lineage", "unavailable",
                            "paper audit dependency unavailable", string.Empty);
                    }
                }
                else
                {
                    WriteArtifactStatus("geometry_stage1", "unavailable", "MeshExtractor missing", string.Empty);
                    WriteArtifactStatus("paper_audit", "unavailable", "MeshExtractor missing", string.Empty);
                    WriteArtifactStatus("evidence_lineage", "unavailable",
                        "paper audit dependency unavailable", string.Empty);
                }

                VolumeIntegrator volume = VolumeIntegrator.Instance;
                if (volume != null)
                {
                    Interlocked.Increment(ref _artifactRequests);
                    bool requested = volume.RequestGunGelCandidateAuditExport("独立回放会话停止", path =>
                    {
                        Task.Run(() =>
                        {
                            try { CopyArtifact(path, "gungel_candidate_audit"); }
                            finally { Interlocked.Decrement(ref _artifactRequests); }
                        });
                    });
                    if (!requested)
                    {
                        Interlocked.Decrement(ref _artifactRequests);
                        WriteArtifactStatus("gungel_candidate_audit", "unavailable", "GunGel audit not ready", string.Empty);
                        WriteArtifactStatus("evidence_lineage", "unavailable",
                            "GunGel audit dependency unavailable", string.Empty);
                    }

                    Interlocked.Increment(ref _artifactRequests);
                    string responsibilityDestination = Path.Combine(
                        _sessionDirectory, "artifacts", "tsdf_responsibility");
                    bool responsibilityRequested = _tsdfResponsibilityStarted &&
                        volume.RequestTsdfResponsibilityAuditExport(
                            "独立回放会话停止", responsibilityDestination, path =>
                            {
                                try
                                {
                                     WriteArtifactStatus("tsdf_responsibility",
                                         string.IsNullOrEmpty(path) ? "failed" : "copied",
                                         string.IsNullOrEmpty(path)
                                             ? volume.LastTsdfResponsibilityExportError
                                             : string.Empty,
                                        string.IsNullOrEmpty(path) ? string.Empty :
                                            "artifacts/tsdf_responsibility");
                                }
                                finally { Interlocked.Decrement(ref _artifactRequests); }
                            });
                    if (!responsibilityRequested)
                    {
                        Interlocked.Decrement(ref _artifactRequests);
                        WriteArtifactStatus("tsdf_responsibility", "unavailable",
                            "responsibility capture not started or export already unavailable", string.Empty);
                    }

                }
                else
                {
                    WriteArtifactStatus("gungel_candidate_audit", "unavailable", "VolumeIntegrator missing", string.Empty);
                    WriteArtifactStatus("tsdf_responsibility", "unavailable", "VolumeIntegrator missing", string.Empty);
                    WriteArtifactStatus("evidence_lineage", "unavailable",
                        "GunGel audit dependency unavailable", string.Empty);
                }
            }
            catch (Exception e)
            {
                WriteArtifactStatus("online_artifacts", "failed", e.Message, string.Empty);
            }
        }

        private void RequestSurfaceFeatureArchiveBuild()
        {
            if (string.IsNullOrEmpty(_sessionDirectory) ||
                Interlocked.CompareExchange(ref _surfaceFeatureArchiveBuildStarted, 1, 0) != 0)
                return;
            Interlocked.Increment(ref _artifactRequests);
            Task.Run(() =>
            {
                try
                {
                    string destination = Path.Combine(_sessionDirectory, "artifacts",
                        "surface_feature_archive");
                    SurfaceFeatureArchiveBuilder.Build(_sessionDirectory, destination);
                    WriteArtifactStatus("surface_feature_archive", "built", string.Empty,
                        "artifacts/surface_feature_archive");
                }
                catch (Exception e)
                {
                    WriteArtifactStatus("surface_feature_archive", "failed",
                        e.GetType().Name + ": " + e.Message, string.Empty);
                }
                finally { Interlocked.Decrement(ref _artifactRequests); }
            });
        }

        private void CopyGeometryFamily(string summaryPath, string destination)
        {
            string sourceDirectory = Path.GetDirectoryName(summaryPath) ?? string.Empty;
            string name = Path.GetFileNameWithoutExtension(summaryPath);
            if (name.EndsWith("_summary", StringComparison.OrdinalIgnoreCase))
                name = name.Substring(0, name.Length - "_summary".Length);
            Directory.CreateDirectory(destination);
            foreach (string file in Directory.GetFiles(sourceDirectory, name + "*", SearchOption.TopDirectoryOnly))
                File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), true);
            WriteArtifactStatus("geometry_stage1", "copied", string.Empty, "artifacts/geometry_stage1");
        }

        private void CopyArtifact(string sourcePath, string destinationName)
        {
            if (string.IsNullOrEmpty(sourcePath))
            {
                WriteArtifactStatus(destinationName, "unavailable", "export returned empty path", string.Empty);
                return;
            }
            string source = Directory.Exists(sourcePath)
                ? sourcePath
                : Path.GetDirectoryName(sourcePath) ?? string.Empty;
            if (!Directory.Exists(source))
            {
                WriteArtifactStatus(destinationName, "failed", "source path missing: " + sourcePath, string.Empty);
                return;
            }
            string destination = Path.Combine(_sessionDirectory, "artifacts", destinationName);
            CopyDirectory(source, destination);
            var paperIssues = new List<string>();
            string paperContract = Path.Combine(destination, "full_output_join_contract.txt");
            if (destinationName == "paper_audit" && File.Exists(paperContract))
            {
                string contract = File.ReadAllText(paperContract);
                if (contract.Contains("fusion_final_partial_period_flush=failed"))
                    paperIssues.Add("final fusion counter period readback failed");
                if (contract.Contains("paper_queue_drain=timeout"))
                    paperIssues.Add("production paper final drain timed out");
                if (!contract.Contains("replacement_event_drops=0"))
                    paperIssues.Add("production paper replacement events were dropped");
                if (!contract.Contains("quality_timeline_bin_drops=0"))
                    paperIssues.Add("production paper quality timeline bins were dropped");
                if (!File.Exists(Path.Combine(destination,
                        "production_paper_quality_timeline.csv")))
                    paperIssues.Add("production paper quality timeline is missing");
                if (contract.Contains("visual_quality_pages=0"))
                    paperIssues.Add("production paper visual quality pages are missing");
                string occupancyPath = Path.Combine(destination,
                    "production_paper_occupancy.r32_uint.bin");
                string occupancySchemaPath = Path.Combine(destination,
                    "production_paper_occupancy_schema.json");
                if (!File.Exists(occupancyPath) || !File.Exists(occupancySchemaPath))
                {
                    paperIssues.Add("exact production paper occupancy ledger is missing");
                }
                else
                {
                    string occupancySchema = File.ReadAllText(occupancySchemaPath);
                    long productionTriangles = ReadJsonInteger(
                        occupancySchema, "productionTriangles");
                    long rasterizedTriangles = ReadJsonInteger(
                        occupancySchema, "rasterizedTriangles");
                    long occupiedCells = ReadJsonInteger(
                        occupancySchema, "occupiedCells");
                    if (productionTriangles <= 0 ||
                        rasterizedTriangles != productionTriangles ||
                        occupiedCells <= 0 || new FileInfo(occupancyPath).Length <= 0)
                        paperIssues.Add("exact production paper occupancy ledger failed self-check");
                }
                string surfaceLedger = Path.Combine(destination,
                    "production_paper_surface_ledger.txt");
                if (!File.Exists(surfaceLedger))
                {
                    paperIssues.Add("production paper surface ledger is missing");
                }
                else
                {
                    string surface = File.ReadAllText(surfaceLedger);
                    if (!surface.Contains("stage_responsibility_spatial_csv:"))
                        paperIssues.Add("same-epoch surface stage responsibility ledger is missing");
                    if (surface.Contains("stage_responsibility_comparable_bins=0"))
                        paperIssues.Add("same-epoch surface stages have no comparable spatial bins");
                }
            }
            bool incompletePaperFlush = paperIssues.Count > 0;
            WriteArtifactStatus(destinationName,
                incompletePaperFlush ? "incomplete" : "copied",
                incompletePaperFlush ? string.Join("; ", paperIssues) : string.Empty,
                "artifacts/" + destinationName);
            TryBuildEvidenceLineage();
        }

        private void TryBuildEvidenceLineage()
        {
            string artifacts = Path.Combine(_sessionDirectory, "artifacts");
            string paper = Path.Combine(artifacts, "paper_audit");
            string gunGel = Path.Combine(artifacts, "gungel_candidate_audit");
            if (!File.Exists(Path.Combine(gunGel, "candidates.csv")) ||
                !File.Exists(Path.Combine(gunGel, "retirements.csv")) ||
                !File.Exists(Path.Combine(gunGel, "court_waves.csv")))
                return;
            if (!File.Exists(Path.Combine(paper, "paper_hole_cells.csv")))
            {
                if (File.Exists(Path.Combine(paper,
                    "production_paper_occupancy_schema.json")))
                    WriteArtifactStatus("evidence_lineage", "deferred_offline",
                        "exact blank-paper lineage requires the final TSDF responsibility join",
                        "artifacts/tsdf_responsibility/paper_hole_responsibility.csv");
                return;
            }
            if (Interlocked.CompareExchange(ref _evidenceLineageBuildStarted, 1, 0) != 0)
                return;

            try
            {
                string destination = Path.Combine(artifacts, "evidence_lineage");
                PaperEvidenceLineageBuilder.Build(paper, gunGel, destination);
                WriteArtifactStatus("evidence_lineage", "copied", string.Empty,
                    "artifacts/evidence_lineage");
            }
            catch (Exception e)
            {
                WriteArtifactStatus("evidence_lineage", "failed",
                    e.GetType().Name + ": " + e.Message, string.Empty);
            }
        }

        private static long ReadJsonInteger(string json, string property)
        {
            string marker = "\"" + property + "\"";
            int start = json.IndexOf(marker, StringComparison.Ordinal);
            if (start < 0) return -1;
            start = json.IndexOf(':', start + marker.Length);
            if (start < 0) return -1;
            start++;
            while (start < json.Length && char.IsWhiteSpace(json[start])) start++;
            int end = start;
            while (end < json.Length && char.IsDigit(json[end])) end++;
            return end > start && long.TryParse(json.Substring(start, end - start),
                NumberStyles.None, CultureInfo.InvariantCulture, out long value)
                ? value : -1;
        }

        private IEnumerator ExportSystemRoomMeshAsync()
        {
            // StartCoroutine executes synchronously until the first yield.  Yield first so the
            // A-button transaction can return before touching MRUK or allocating mesh arrays.
            yield return null;
            string directory = Path.Combine(_sessionDirectory, "system_reference");
            Directory.CreateDirectory(directory);
            try
            {
                MRUK mruk = null;
                Exception lookupFailure = null;
                try
                {
                    mruk = MRUK.Instance != null
                        ? MRUK.Instance
                        : FindAnyObjectByType<MRUK>(FindObjectsInactive.Include);
                }
                catch (Exception e) { lookupFailure = e; }
                if (lookupFailure != null)
                {
                    WriteSystemReferenceStatus("failed",
                        lookupFailure.GetType().Name + ": " + lookupFailure.Message, 0, 0, 0);
                    yield break;
                }
                if (mruk == null)
                {
                    WriteSystemReferenceStatus("unavailable", "MRUK component unavailable in production scene", 0, 0, 0);
                    yield break;
                }

                // The MRUK object is intentionally active from scene start, but USE_SCENE
                // permission and device discovery complete asynchronously.  A fast A press
                // must not turn that ordinary startup race into a permanent empty reference.
                float startupDeadline = Time.realtimeSinceStartup + SystemRoomMeshStartupWaitSeconds;
                while ((mruk.Rooms == null || mruk.Rooms.Count == 0) &&
                       Time.realtimeSinceStartup < startupDeadline)
                    yield return null;

                string reloadResult = "not_needed";
                if (mruk.Rooms == null || mruk.Rooms.Count == 0)
                {
                    Task<MRUK.LoadDeviceResult> reloadTask = null;
                    try
                    {
                        // Never open Space Setup from a freeze transaction.  We only query
                        // the room the user already created in Quest settings.  V2FallbackV1
                        // accepts both high-fidelity and ordinary system room captures.
                        reloadTask = mruk.LoadSceneFromDevice(
                            requestSceneCaptureIfNoDataFound: false,
                            removeMissingRooms: true,
                            sceneModel: MRUK.SceneModel.V2FallbackV1);
                    }
                    catch (Exception e)
                    {
                        reloadResult = e.GetType().Name + ": " + e.Message;
                    }

                    if (reloadTask != null)
                    {
                        float reloadDeadline = Time.realtimeSinceStartup + SystemRoomMeshReloadWaitSeconds;
                        while (!reloadTask.IsCompleted && Time.realtimeSinceStartup < reloadDeadline)
                            yield return null;

                        if (!reloadTask.IsCompleted)
                            reloadResult = "timeout";
                        else if (reloadTask.IsFaulted)
                        {
                            Exception failure = reloadTask.Exception?.GetBaseException();
                            reloadResult = (failure?.GetType().Name ?? "load_failed") + ": " +
                                           (failure?.Message ?? "unknown MRUK load failure");
                        }
                        else
                        {
                            reloadResult = reloadTask.Result.ToString();
                            // DiscoveryOngoing means MRUK's startup request owns the query.
                            // Success can also precede Unity-side room instantiation by a frame.
                            if (reloadTask.Result == MRUK.LoadDeviceResult.DiscoveryOngoing ||
                                reloadTask.Result == MRUK.LoadDeviceResult.Success)
                            {
                                while ((mruk.Rooms == null || mruk.Rooms.Count == 0) &&
                                       Time.realtimeSinceStartup < reloadDeadline)
                                    yield return null;
                            }
                        }
                    }
                }

                if (mruk.Rooms == null || mruk.Rooms.Count == 0)
                {
                    WriteSystemReferenceStatus("unavailable",
                        "MRUK active but no rooms loaded; initialized=" + mruk.IsInitialized +
                        ", reloadResult=" + reloadResult, 0, 0, 0);
                    yield break;
                }

                var snapshots = new List<SystemMeshSnapshot>(mruk.Rooms.Count);
                for (int roomIndex = 0; roomIndex < mruk.Rooms.Count; roomIndex++)
                {
                    MRUKRoom room = mruk.Rooms[roomIndex];
                    MRUKAnchor anchor = room != null ? room.GlobalMeshAnchor : null;
                    if (anchor == null) continue;
                    Mesh mesh;
                    try { mesh = anchor.GlobalMesh; }
                    catch { continue; }
                    if (mesh == null || mesh.vertexCount == 0) continue;

                    // Unity mesh objects stay on the main thread.  Only immutable managed arrays
                    // cross to the worker that performs the expensive OBJ formatting and I/O.
                    Vector3[] vertices = mesh.vertices;
                    int[] triangles = mesh.triangles;
                    if (triangles == null || triangles.Length < 3) continue;
                    snapshots.Add(new SystemMeshSnapshot(roomIndex, anchor.name,
                        anchor.transform.localToWorldMatrix, vertices, triangles));
                    yield return null;
                }

                Task writeTask = Task.Run(() => WriteSystemRoomMeshSnapshots(directory, snapshots));
                while (!writeTask.IsCompleted)
                    yield return null;
                if (writeTask.IsFaulted)
                {
                    Exception failure = writeTask.Exception?.GetBaseException() ??
                                        new IOException("system mesh export failed");
                    WriteSystemReferenceStatus("failed",
                        failure.GetType().Name + ": " + failure.Message, 0, 0, 0);
                }
            }
            finally
            {
                Interlocked.Decrement(ref _artifactRequests);
            }
        }

        private void WriteSystemRoomMeshSnapshots(string directory, List<SystemMeshSnapshot> snapshots)
        {
            string objPath = Path.Combine(directory, "quest_system_room_mesh_world.obj");
            string provenancePath = Path.Combine(directory, "quest_system_room_mesh_provenance.json");
            int vertexCount = 0;
            int triangleCount = 0;
            var provenance = new StringBuilder(8192);
            provenance.Append("{\n  \"schema\": \"scancover.system_room_mesh_reference.v1\",\n  \"meshes\": [\n");
            using (var writer = new StreamWriter(objPath, false, new UTF8Encoding(false)))
            {
                writer.WriteLine("# Quest MRUK GlobalMeshAnchor meshes baked to Unity world space");
                int offset = 0;
                for (int meshIndex = 0; meshIndex < snapshots.Count; meshIndex++)
                {
                    SystemMeshSnapshot snapshot = snapshots[meshIndex];
                    writer.WriteLine("o room_" + snapshot.RoomIndex.ToString("D2", CultureInfo.InvariantCulture));
                    for (int i = 0; i < snapshot.Vertices.Length; i++)
                    {
                        Vector3 v = snapshot.LocalToWorld.MultiplyPoint3x4(snapshot.Vertices[i]);
                        writer.WriteLine("v " + F(v.x) + " " + F(v.y) + " " + F(v.z));
                    }
                    for (int i = 0; i + 2 < snapshot.Triangles.Length; i += 3)
                    {
                        writer.WriteLine("f " + (snapshot.Triangles[i] + 1 + offset) + " " +
                                         (snapshot.Triangles[i + 1] + 1 + offset) + " " +
                                         (snapshot.Triangles[i + 2] + 1 + offset));
                    }

                    if (meshIndex > 0) provenance.Append(",\n");
                    provenance.Append("    {\"roomIndex\":").Append(snapshot.RoomIndex)
                        .Append(",\"anchorName\":\"").Append(Json(snapshot.AnchorName)).Append("\"")
                        .Append(",\"vertexCount\":").Append(snapshot.Vertices.Length)
                        .Append(",\"triangleCount\":").Append(snapshot.Triangles.Length / 3)
                        .Append(",\"localToWorld\":");
                    AppendMatrix(provenance, snapshot.LocalToWorld);
                    provenance.Append('}');
                    offset += snapshot.Vertices.Length;
                    vertexCount += snapshot.Vertices.Length;
                    triangleCount += snapshot.Triangles.Length / 3;
                }
            }
            provenance.Append("\n  ]\n}\n");
            File.WriteAllText(provenancePath, provenance.ToString(), new UTF8Encoding(false));
            WriteSystemReferenceStatus(snapshots.Count > 0 ? "exported" : "unavailable",
                snapshots.Count > 0 ? string.Empty : "No GlobalMeshAnchor contained triangles",
                snapshots.Count, vertexCount, triangleCount);
        }

        private void WriteSystemReferenceStatus(string status, string issue,
            int meshes, int vertices, int triangles)
        {
            bool external = string.Equals(status, "external_required", StringComparison.Ordinal);
            string json = "{\n" +
                          "  \"schema\": \"scancover.system_room_mesh_reference.v1\",\n" +
                          "  \"status\": \"" + Json(status) + "\",\n" +
                          "  \"issue\": \"" + Json(issue) + "\",\n" +
                          "  \"source\": \"" + Json(external
                              ? "separate Meta Scene Mesh scene / OVRTriangleMesh"
                              : "MRUK.Rooms[].GlobalMeshAnchor.GlobalMesh") + "\",\n" +
                          "  \"coordinateSpace\": \"" + Json(external
                              ? "external Unity world, metres; offline registration required"
                              : "Unity world, metres, baked at session stop") + "\",\n" +
                          "  \"meshCount\": " + meshes + ",\n" +
                          "  \"vertexCount\": " + vertices + ",\n" +
                          "  \"triangleCount\": " + triangles + "\n}\n";
            File.WriteAllText(Path.Combine(_sessionDirectory, "system_reference", "status.json"),
                json, new UTF8Encoding(false));
        }

        private void TryFinalize()
        {
            if (_finalized || _finalizeTask != null || !_endRequested ||
                !_depthStreamsComplete || OutstandingCount > 0)
                return;
            // B 账来自已接纳融合缓冲的异步回读，必须等回读排空后再封账；
            // 否则用户按下停止键时仍在路上的最后几帧会被静默漏掉。
            FlushRuntimeLedgers();
            _probeFinalBufferLedger?.End();
            _finalSurfaceCourt?.End();
            _finalizeClean = _depthPairDrops == 0 && _depthPairReadbackErrors == 0 &&
                             _depthPairWriteErrors == 0 && _fusionDropped == 0 &&
                             _fusionReadbackErrors == 0 && _fusionWriteErrors == 0 &&
                             (_probeShadowLedger?.WriteErrors ?? 0) == 0 &&
                             (_probeShadowLedger?.AdjudicatorWriteErrors ?? 0) == 0 &&
                             (_probeShadowLedger?.GraduationRaceWriteErrors ?? 0) == 0 &&
                             (_directProjectionShadowLedger?.WriteErrors ?? 0) == 0 &&
                             (_directProjectionShadowLedger?.ReadbackErrors ?? 0) == 0 &&
                             (_directProjectionShadowLedger?.OverflowRecords ?? 0) == 0 &&
                             (_directProjectionShadowLedger?.Frames ?? 0) > 0 &&
                             (_probeFinalBufferLedger?.WriteErrors ?? 0) == 0 &&
                             (_probeFinalBufferLedger?.AdjudicatorWriteErrors ?? 0) == 0 &&
                             (_probeFinalBufferLedger?.GraduationRaceWriteErrors ?? 0) == 0 &&
                             (_finalSurfaceCourt?.WriteErrors ?? 0) == 0 &&
                             _probeConversionWriteErrors == 0 &&
                             _runtimeLedgerWriteErrors == 0 &&
                             _artifactFailures == 0;
            Volatile.Write(ref _sealPhase, SealHashing);
            Volatile.Write(ref _checksumFilesProcessed, 0);
            Volatile.Write(ref _checksumFilesTotal, 0);
            WriteState("hashing", 0);
            Interlocked.Increment(ref _artifactRequests);
            _finalizeTask = Task.Run(() =>
            {
                try
                {
                    WriteChecksumsAndCompleteMarker();
                    Volatile.Write(ref _sealPhase,
                        _finalizeClean ? SealComplete : SealIncomplete);
                    WriteState(_finalizeClean ? "complete" : "incomplete_with_capture_loss", 0);
                }
                catch (Exception e)
                {
                    _finalizeIssue = e.GetType().Name + ": " + e.Message;
                    Volatile.Write(ref _sealPhase, SealFailed);
                    try { WriteState("finalize_failed:" + e.GetType().Name, 0); }
                    catch { }
                    throw;
                }
                finally { Interlocked.Decrement(ref _artifactRequests); }
            });
        }

        private void WriteChecksumsAndCompleteMarker()
        {
            string manifestPath = Path.Combine(_sessionDirectory, "checksums.sha256");
            string manifestTempPath = manifestPath + ".tmp";
            string completePath = Path.Combine(_sessionDirectory, "capture_complete.json");
            string incompletePath = Path.Combine(_sessionDirectory, "capture_incomplete.json");
            string completeTempPath = completePath + ".tmp";
            string incompleteTempPath = incompletePath + ".tmp";
            string[] files = Directory.GetFiles(_sessionDirectory, "*", SearchOption.AllDirectories);
            Array.Sort(files, StringComparer.Ordinal);
            string statusPath = Path.Combine(_sessionDirectory, "session_status.json");
            var hashFiles = new List<string>(files.Length);
            foreach (string file in files)
            {
                if (string.Equals(file, manifestPath, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(file, manifestTempPath, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(file, completePath, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(file, completeTempPath, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(file, incompletePath, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(file, incompleteTempPath, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(file, statusPath, StringComparison.OrdinalIgnoreCase))
                    continue;
                hashFiles.Add(file);
            }
            Volatile.Write(ref _checksumFilesTotal, hashFiles.Count);
            Volatile.Write(ref _checksumFilesProcessed, 0);
            var hashes = new StringBuilder(hashFiles.Count * 96);
            using (SHA256 sha = SHA256.Create())
            {
                foreach (string file in hashFiles)
                {
                    string fullPath = Path.GetFullPath(file);
                    string hashText;
                    if (!_immutableFileHashes.TryGetValue(fullPath, out hashText))
                    {
                        using FileStream stream = File.OpenRead(file);
                        hashText = Hex(sha.ComputeHash(stream));
                    }
                    hashes.Append(hashText).Append("  ")
                        .Append(Path.GetRelativePath(_sessionDirectory, file).Replace('\\', '/'))
                        .AppendLine();
                    Interlocked.Increment(ref _checksumFilesProcessed);
                }
            }
            File.WriteAllText(manifestTempPath, hashes.ToString(), new UTF8Encoding(false));
            if (File.Exists(manifestPath)) File.Delete(manifestPath);
            File.Move(manifestTempPath, manifestPath);
            string manifestHash;
            using (SHA256 sha = SHA256.Create())
            using (FileStream stream = File.OpenRead(manifestPath))
                manifestHash = Hex(sha.ComputeHash(stream));
            bool clean = _depthPairDrops == 0 && _depthPairReadbackErrors == 0 &&
                         _depthPairWriteErrors == 0 && _fusionDropped == 0 &&
                         _fusionReadbackErrors == 0 && _fusionWriteErrors == 0 &&
                         (_probeShadowLedger?.WriteErrors ?? 0) == 0 &&
                         (_probeShadowLedger?.AdjudicatorWriteErrors ?? 0) == 0 &&
                         (_probeShadowLedger?.GraduationRaceWriteErrors ?? 0) == 0 &&
                         (_directProjectionShadowLedger?.WriteErrors ?? 0) == 0 &&
                         (_directProjectionShadowLedger?.ReadbackErrors ?? 0) == 0 &&
                         (_directProjectionShadowLedger?.OverflowRecords ?? 0) == 0 &&
                         (_directProjectionShadowLedger?.Frames ?? 0) > 0 &&
                          (_probeFinalBufferLedger?.WriteErrors ?? 0) == 0 &&
                          (_probeFinalBufferLedger?.AdjudicatorWriteErrors ?? 0) == 0 &&
                          (_probeFinalBufferLedger?.GraduationRaceWriteErrors ?? 0) == 0 &&
                          (_finalSurfaceCourt?.WriteErrors ?? 0) == 0 &&
                          _probeConversionWriteErrors == 0 &&
                         _runtimeLedgerWriteErrors == 0 &&
                         _artifactFailures == 0;
            string complete = "{\n" +
                              "  \"schema\": \"" + Schema + "\",\n" +
                              "  \"state\": \"" + (clean ? "complete" : "incomplete_with_capture_loss") + "\",\n" +
                              "  \"startedUtc\": \"" + Json(_startedUtc) + "\",\n" +
                              "  \"stoppedUtc\": \"" + Json(_stoppedUtc) + "\",\n" +
                              "  \"depthRecordedEyeIndex\": " + RecordedEye + ",\n" +
                              "  \"fusionRecords\": " + _fusionSequence + ",\n" +
                              "  \"fusionCaptureDrops\": " + _fusionDropped + ",\n" +
                              "  \"fusionReadbackErrors\": " + _fusionReadbackErrors + ",\n" +
                              "  \"fusionWriteErrors\": " + _fusionWriteErrors + ",\n" +
                              "  \"runtimeLedgerWriteErrors\": " + _runtimeLedgerWriteErrors + ",\n" +
                              "  \"artifactFailures\": " + _artifactFailures + ",\n" +
                              "  \"probeShadowRows\": " + (_probeShadowLedger?.Rows ?? 0) + ",\n" +
                              "  \"probeShadowWriteErrors\": " + (_probeShadowLedger?.WriteErrors ?? 0) + ",\n" +
                               "  \"probeAdjudicatorFrameRows\": " + (_probeShadowLedger?.AdjudicatorFrameRows ?? 0) + ",\n" +
                               "  \"probeAdjudicatorEventRows\": " + (_probeShadowLedger?.AdjudicatorEventRows ?? 0) + ",\n" +
                               "  \"probeAdjudicatorFollowupRows\": " + (_probeShadowLedger?.AdjudicatorFollowupRows ?? 0) + ",\n" +
                               "  \"probeAdjudicatorWriteErrors\": " + (_probeShadowLedger?.AdjudicatorWriteErrors ?? 0) + ",\n" +
                               "  \"graduationRaceCandidateRows\": " + (_probeShadowLedger?.GraduationRaceCandidateRows ?? 0) + ",\n" +
                               "  \"graduationRaceSummaryRows\": " + (_probeShadowLedger?.GraduationRaceSummaryRows ?? 0) + ",\n" +
                               "  \"graduationRaceWitnessEventRows\": " + (_probeShadowLedger?.GraduationRaceWitnessEventRows ?? 0) + ",\n" +
                               "  \"graduationRaceWriteErrors\": " + (_probeShadowLedger?.GraduationRaceWriteErrors ?? 0) + ",\n" +
                               "  \"direct45Frames\": " + (_directProjectionShadowLedger?.Frames ?? 0) + ",\n" +
                               "  \"direct45Rows\": " + (_directProjectionShadowLedger?.Rows ?? 0) + ",\n" +
                               "  \"direct45EventRows\": " + (_directProjectionShadowLedger?.EventRows ?? 0) + ",\n" +
                               "  \"direct45GuardDecisionRows\": " + (_directProjectionShadowLedger?.GuardDecisionRows ?? 0) + ",\n" +
                               "  \"direct45GuardAllowTransitions\": " + (_directProjectionShadowLedger?.GuardAllowTransitions ?? 0) + ",\n" +
                               "  \"direct45GuardVetoTransitions\": " + (_directProjectionShadowLedger?.GuardVetoTransitions ?? 0) + ",\n" +
                               "  \"direct45GuardPendingAtSeal\": " + (_directProjectionShadowLedger?.GuardPendingAtSeal ?? 0) + ",\n" +
                               "  \"direct45OverflowRecords\": " + (_directProjectionShadowLedger?.OverflowRecords ?? 0) + ",\n" +
                              "  \"direct45ReadbackErrors\": " + (_directProjectionShadowLedger?.ReadbackErrors ?? 0) + ",\n" +
                              "  \"direct45WriteErrors\": " + (_directProjectionShadowLedger?.WriteErrors ?? 0) + ",\n" +
                              "  \"probeFinalBufferRows\": " + (_probeFinalBufferLedger?.Rows ?? 0) + ",\n" +
                              "  \"probeFinalBufferWriteErrors\": " + (_probeFinalBufferLedger?.WriteErrors ?? 0) + ",\n" +
                              "  \"probeFinalBufferAdjudicatorFrameRows\": " + (_probeFinalBufferLedger?.AdjudicatorFrameRows ?? 0) + ",\n" +
                              "  \"probeFinalBufferAdjudicatorEventRows\": " + (_probeFinalBufferLedger?.AdjudicatorEventRows ?? 0) + ",\n" +
                               "  \"probeFinalBufferGraduationWitnessRows\": " + (_probeFinalBufferLedger?.GraduationRaceWitnessEventRows ?? 0) + ",\n" +
                               "  \"finalSurfaceCourtWriteErrors\": " + (_finalSurfaceCourt?.WriteErrors ?? 0) + ",\n" +
                               "  \"probeConversionRows\": " + _probeConversionRows + ",\n" +
                              "  \"probeConversionMissingPreRows\": " + _probeConversionMissingPreRows + ",\n" +
                              "  \"probeConversionWriteErrors\": " + _probeConversionWriteErrors + ",\n" +
                              "  \"depthPairDrops\": " + _depthPairDrops + ",\n" +
                              "  \"depthPairReadbackErrors\": " + _depthPairReadbackErrors + ",\n" +
                              "  \"depthPairWriteErrors\": " + _depthPairWriteErrors + ",\n" +
                              "  \"checksumsManifestSha256\": \"" + manifestHash + "\"\n" +
                              "}\n";
            string markerPath = clean ? completePath : incompletePath;
            string markerTempPath = markerPath + ".tmp";
            File.WriteAllText(markerTempPath, complete, new UTF8Encoding(false));
            if (File.Exists(markerPath)) File.Delete(markerPath);
            File.Move(markerTempPath, markerPath);
        }

        private void WriteState(string state, int? outstandingOverride = null)
        {
            if (string.IsNullOrEmpty(_sessionDirectory)) return;
            string json = "{\n" +
                          "  \"schema\": \"" + Schema + "\",\n" +
                          "  \"state\": \"" + Json(state) + "\",\n" +
                          "  \"depthStreamsComplete\": " + (_depthStreamsComplete ? "true" : "false") + ",\n" +
                          "  \"fusionRecords\": " + _fusionSequence + ",\n" +
                          "  \"fusionCaptureDrops\": " + _fusionDropped + ",\n" +
                          "  \"fusionReadbackErrors\": " + _fusionReadbackErrors + ",\n" +
                          "  \"fusionWriteErrors\": " + _fusionWriteErrors + ",\n" +
                          "  \"runtimeLedgerWriteErrors\": " + _runtimeLedgerWriteErrors + ",\n" +
                          "  \"artifactFailures\": " + _artifactFailures + ",\n" +
                          "  \"probeShadowRows\": " + (_probeShadowLedger?.Rows ?? 0) + ",\n" +
                          "  \"probeShadowWriteErrors\": " + (_probeShadowLedger?.WriteErrors ?? 0) + ",\n" +
                           "  \"probeAdjudicatorFrameRows\": " + (_probeShadowLedger?.AdjudicatorFrameRows ?? 0) + ",\n" +
                           "  \"probeAdjudicatorEventRows\": " + (_probeShadowLedger?.AdjudicatorEventRows ?? 0) + ",\n" +
                           "  \"probeAdjudicatorFollowupRows\": " + (_probeShadowLedger?.AdjudicatorFollowupRows ?? 0) + ",\n" +
                           "  \"probeAdjudicatorWriteErrors\": " + (_probeShadowLedger?.AdjudicatorWriteErrors ?? 0) + ",\n" +
                           "  \"graduationRaceCandidateRows\": " + (_probeShadowLedger?.GraduationRaceCandidateRows ?? 0) + ",\n" +
                           "  \"graduationRaceSummaryRows\": " + (_probeShadowLedger?.GraduationRaceSummaryRows ?? 0) + ",\n" +
                           "  \"graduationRaceWitnessEventRows\": " + (_probeShadowLedger?.GraduationRaceWitnessEventRows ?? 0) + ",\n" +
                           "  \"graduationRaceWriteErrors\": " + (_probeShadowLedger?.GraduationRaceWriteErrors ?? 0) + ",\n" +
                           "  \"direct45Frames\": " + (_directProjectionShadowLedger?.Frames ?? 0) + ",\n" +
                           "  \"direct45Rows\": " + (_directProjectionShadowLedger?.Rows ?? 0) + ",\n" +
                           "  \"direct45EventRows\": " + (_directProjectionShadowLedger?.EventRows ?? 0) + ",\n" +
                           "  \"direct45GuardDecisionRows\": " + (_directProjectionShadowLedger?.GuardDecisionRows ?? 0) + ",\n" +
                           "  \"direct45GuardAllowTransitions\": " + (_directProjectionShadowLedger?.GuardAllowTransitions ?? 0) + ",\n" +
                           "  \"direct45GuardVetoTransitions\": " + (_directProjectionShadowLedger?.GuardVetoTransitions ?? 0) + ",\n" +
                           "  \"direct45GuardPendingAtSeal\": " + (_directProjectionShadowLedger?.GuardPendingAtSeal ?? 0) + ",\n" +
                           "  \"direct45OverflowRecords\": " + (_directProjectionShadowLedger?.OverflowRecords ?? 0) + ",\n" +
                          "  \"direct45ReadbackErrors\": " + (_directProjectionShadowLedger?.ReadbackErrors ?? 0) + ",\n" +
                          "  \"direct45WriteErrors\": " + (_directProjectionShadowLedger?.WriteErrors ?? 0) + ",\n" +
                          "  \"probeFinalBufferRows\": " + (_probeFinalBufferLedger?.Rows ?? 0) + ",\n" +
                          "  \"probeFinalBufferWriteErrors\": " + (_probeFinalBufferLedger?.WriteErrors ?? 0) + ",\n" +
                          "  \"probeFinalBufferAdjudicatorFrameRows\": " + (_probeFinalBufferLedger?.AdjudicatorFrameRows ?? 0) + ",\n" +
                          "  \"probeFinalBufferAdjudicatorEventRows\": " + (_probeFinalBufferLedger?.AdjudicatorEventRows ?? 0) + ",\n" +
                           "  \"probeFinalBufferGraduationWitnessRows\": " + (_probeFinalBufferLedger?.GraduationRaceWitnessEventRows ?? 0) + ",\n" +
                           "  \"finalSurfaceCourtWriteErrors\": " + (_finalSurfaceCourt?.WriteErrors ?? 0) + ",\n" +
                           "  \"probeConversionRows\": " + _probeConversionRows + ",\n" +
                          "  \"probeConversionMissingPreRows\": " + _probeConversionMissingPreRows + ",\n" +
                          "  \"probeConversionWriteErrors\": " + _probeConversionWriteErrors + ",\n" +
                          "  \"depthPairDrops\": " + _depthPairDrops + ",\n" +
                          "  \"depthPairReadbackErrors\": " + _depthPairReadbackErrors + ",\n" +
                          "  \"depthPairWriteErrors\": " + _depthPairWriteErrors + ",\n" +
                          "  \"sealPhase\": \"" + Json(SealPhaseName(Volatile.Read(ref _sealPhase))) + "\",\n" +
                          "  \"checksumFilesProcessed\": " + Volatile.Read(ref _checksumFilesProcessed) + ",\n" +
                          "  \"checksumFilesTotal\": " + Volatile.Read(ref _checksumFilesTotal) + ",\n" +
                          "  \"safeToExit\": " + (IsSafelySealed ? "true" : "false") + ",\n" +
                          "  \"finalizeIssue\": \"" + Json(_finalizeIssue) + "\",\n" +
                          "  \"outstanding\": " + (outstandingOverride ?? OutstandingCount) + "\n}\n";
            lock (_fileLock)
                File.WriteAllText(Path.Combine(_sessionDirectory, "session_status.json"),
                    json, new UTF8Encoding(false));
        }

        private static string SealPhaseName(int phase)
        {
            return phase switch
            {
                SealCapturing => "capturing",
                SealDraining => "draining",
                SealHashing => "hashing",
                SealComplete => "sealed_complete",
                SealIncomplete => "sealed_incomplete",
                SealFailed => "failed",
                _ => "idle"
            };
        }

        private void WriteArtifactStatus(string name, string status, string issue, string relativePath)
        {
            try
            {
                if (string.Equals(status, "failed", StringComparison.Ordinal) ||
                    string.Equals(status, "unavailable", StringComparison.Ordinal) ||
                    string.Equals(status, "incomplete", StringComparison.Ordinal))
                    Interlocked.Increment(ref _artifactFailures);
                string directory = Path.Combine(_sessionDirectory, "artifacts");
                Directory.CreateDirectory(directory);
                string json = "{\n" +
                              "  \"name\": \"" + Json(name) + "\",\n" +
                              "  \"status\": \"" + Json(status) + "\",\n" +
                              "  \"issue\": \"" + Json(issue) + "\",\n" +
                              "  \"relativePath\": \"" + Json(relativePath) + "\"\n}\n";
                File.WriteAllText(Path.Combine(directory, name + "_status.json"),
                    json, new UTF8Encoding(false));
            }
            catch { }
        }

        private static string BuildSessionSchemaJson()
        {
            return "{\n" +
                   "  \"schema\": \"" + Schema + "\",\n" +
                   "  \"purpose\": \"deterministic offline replay and attribution of one production scan\",\n" +
                   "  \"depth_pairs\": \"platform pre-QRS and post-QRS depth for every preprocessed source frame\",\n" +
                   "  \"fusion_inputs\": \"exact accepted depth, admission, RGB and camera inputs plus rejected decision events; guarded frames include independent pre-transaction A and final-buffer B identity snapshots\",\n" +
                   "  \"probe_shadow\": \"pre-transaction A ledger, final_buffer post-transaction B ledger, direct45 conservative candidate-footprint raw/post reprojection, a production pending-retirement gate using 0.25-degree same-view invalidation and 1-degree cross-view support/free decisions, plus legacy zero-authority 16-frame, 8-frame ghost and 0.5/1/2-degree audit lanes\",\n" +
                   "  \"runtime_timeline\": \"every Unity frame head pose, pitch/yaw/roll, platform depth frame, motion, fusion epoch and committed production-paper queue/draw state; integration_dispatches joins fusion attemptIndex/sourceFrame to dirtyEpoch\",\n" +
                   "  \"artifacts\": \"GunGel candidate audit, final production-paper replacement/roughness/seam/hole ledgers, per-voxel TSDF geometry/support/block responsibility, their read-only evidence lineage join, surface feature archive and online geometry ledgers\",\n" +
                   "  \"system_reference\": \"handoff status for a separately exported Quest system room mesh; comparison only\",\n" +
                   "  \"completeRule\": \"capture_complete.json exists only after production depth, fusion and online artifacts drain and checksums finish; HUD safeToExit becomes true only after the marker is atomically published; external system mesh is not a seal dependency\"\n" +
                   "}\n";
        }

        private static string BuildFullOutputIndexJson()
        {
            return "{\n" +
                   "  \"schema\": \"scancover.full_output_index.v4\",\n" +
                   "  \"principle\": \"independent stage facts joined after capture; no diagnostic output has production authority\",\n" +
                   "  \"primaryJoin\": \"depth_pairs.platformFrame = fusion_inputs.sourceFrame = runtime_timeline.integration_dispatches.sourceFrame\",\n" +
                   "  \"epochJoin\": \"production_paper_replacements.candidate_epoch -> integration_dispatches.dirtyEpoch -> integration_dispatches.sourceFrame; queued paper commits may intentionally lag\",\n" +
                   "  \"courtJoin\": \"final_surface_court sourceFrame+attemptIndex+stableId -> fusion frame finalIdentity -> TSDF endpoint sourceFrame+attemptIndex; paper site association remains bounded in the declared volume coordinate contract\",\n" +
                   "  \"paperDecisionContextWarning\": \"observed_platform_frame_at_decision is asynchronous commit-time context and must never be used as candidate provenance\",\n" +
                   "  \"timeJoin\": \"unityFrame plus unscaledTime; use pose deltas rather than assuming equal callback/preprocess/fusion time\",\n" +
                   "  \"lanes\": {\n" +
                   "    \"rawAndProcessedDepth\": \"depth_pairs/manifest.csv and depth_pairs/frames\",\n" +
                   "    \"fusionAdmissionAndExactInput\": \"fusion_inputs/manifest.csv and fusion_inputs/frames\",\n" +
                   "    \"continuousPoseMotionDistanceQueue\": \"runtime_timeline/frames.csv\",\n" +
                   "    \"fusionToExtractionEpoch\": \"runtime_timeline/integration_dispatches.csv\",\n" +
                   "    \"candidateIdentityAndRetirement\": \"probe_shadow plus artifacts/gungel_candidate_audit\",\n" +
                   "    \"finalCourtResponsibility\": \"probe_shadow/final_buffer/final_surface_court frame gates, independent testimonies, decision checks and publication events\",\n" +
                   "    \"committedPaperReplacement\": \"artifacts/paper_audit/production_paper_replacements.csv\",\n" +
                   "    \"committedPaperQualityTimeline\": \"artifacts/paper_audit/production_paper_quality_timeline.csv records every accepted commit at 4x4x4-bin grain\",\n" +
                   "    \"sameEpochSurfaceStageResponsibility\": \"artifacts/paper_audit/production_paper_surface_ledger.txt sections stage_responsibility_contract and stage_responsibility_spatial_csv\",\n" +
                   "    \"committedPaperRoughnessAndSeams\": \"artifacts/paper_audit/production_paper_surface_ledger.txt\",\n" +
                    "    \"fusionVoxelAndFovPeriods\": \"artifacts/paper_audit/fusion_forensic_ledger.txt, fusion_fov_periods.csv and fusion_all_counters.csv for production/raw-projective lanes\",\n" +
                    "    \"tsdfTransactionResponsibility\": \"artifacts/tsdf_responsibility joins each final zero-crossing endpoint to current-lifetime seed, last geometry writer, last support writer and strongest blocked correction\",\n" +
                     "    \"blankPaperResponsibility\": \"artifacts/paper_audit/production_paper_occupancy plus final TSDF responsibility are joined offline to locate missing paper at TSDF support/crossing, extraction/candidate or publication boundaries\"\n" +
                   "  },\n" +
                   "  \"antiMaskingRule\": \"attribute the earliest stage where joined records diverge; do not infer one cause from a later mixed surface alone\",\n" +
                   "  \"stopRule\": \"A stops sampling and drains/seals this ledger without entering frozen replay\"\n" +
                   "}\n";
        }

        private static string BuildReplayContractJson()
        {
            return "{\n" +
                   "  \"schema\": \"scancover.deterministic_replay_contract.v1\",\n" +
                   "  \"initialState\": \"empty TSDF, color, confidence, admission and candidate volumes; recorder refuses a mid-scan start\",\n" +
                   "  \"routeA_preprocess\": {\"orderBy\":\"pairIndex\",\"input\":\"depth_pairs/*platform_pre_qrs.f32\",\"reference\":\"depth_pairs/*post_qrs.f32\"},\n" +
                   "  \"routeB_productionFusion\": {\"orderBy\":\"sequence\",\"acceptedRows\":\"apply exact recorded textures, matrices, exclusions, GunGel admission and RGB camera payload\",\"rejectedRows\":\"advance decision history without integration\",\"timeSource\":\"scaledTime and unscaledTime columns\"},\n" +
                   "  \"routeC_systemReference\": {\"input\":\"separate ScanCover_MetaSceneMeshAudit_*/meta_scene_mesh_aligned_all.obj\",\"alignment\":\"offline registration required across app launches\",\"role\":\"comparison only; never production truth\"},\n" +
                   "  \"configuration\": \"production_config.json is authoritative; do not substitute inspector defaults\",\n" +
                   "  \"direct45\": {\"input\":\"live stable candidate variance footprint plus complete same-frame raw/post right-eye depth and edge/temporal reasons\",\"rule\":\"universal footprint free-space proof or coherent support; production retirement opens only after three opposition votes, same-view opposite receipts within 0.25 degrees invalidate each other, two support receipts separated by at least 1 degree cancel retirement, two universal-free receipts separated by at least 1 degree authorize deletion, otherwise retain pending\"},\n" +
                   "  \"productionRetirementGate\": {\"authority\":\"candidate retirement only\",\"sameViewConeDeg\":0.25,\"crossViewThresholdDeg\":1.0,\"directProjectionStrideFrames\":4,\"rule\":\"all receipts must come from accepted frames, be post-pending, identity-matched and no newer than the candidate transaction frame\"},\n" +
                   "  \"legacyRetirementGuard16Shadow\": {\"input\":\"every actual production stable contradiction retirement plus the newest of that candidate's two identity-stamped direct45 support receipts whose frame is not newer than retirement\",\"rule\":\"support within the prior 16 fusion frames records veto_recent_support; identity mismatch and future-only receipts are explicit invalid states; zero production authority\"},\n" +
                   "  \"retirementGhostShadow\": {\"input\":\"actual stable contradiction retirements with valid recent support copied before candidate clear\",\"rule\":\"retain the legacy result from up to eight later valid visible frames, while independently recording up to 32 classified observations with camera geometry, motion, valid footprint samples and candidate-relative signed/absolute/RMS/max depth residuals; a <=10mm and <=0.25deg bin measures repeat precision rather than absolute accuracy\"},\n" +
                   "  \"retirementContractShadow\": {\"sameViewTemporalConeDeg\":0.25,\"lanesDeg\":[0.5,1.0,2.0],\"rule\":\"support/free observations inside the same-view cone invalidate each other for voting; surviving support and free each require two later same-class observations mutually separated by the lane angle; support wins retain, free wins delete, both conflict-defer, neither insufficient-defer; zero production authority\"},\n" +
                   "  \"coordinates\": \"coordinate_contract.json plus coordinate_contract_stop.json\",\n" +
                   "  \"integrity\": \"sort manifests by numeric keys and require capture_complete.json plus checksums.sha256; session_status.json is mutable operational state and intentionally excluded from the checksum manifest\",\n" +
                   "  \"earliestDepthBoundary\": \"Meta platform environment depth before QRS preprocessing, not inaccessible sensor raw\"\n" +
                   "}\n";
        }

        private static string BuildProductionConfigJson()
        {
            var sb = new StringBuilder(32768);
            sb.Append("{\n  \"schema\": \"scancover.production_config_snapshot.v1\",\n")
                .Append("  \"capturedUtc\": \"").Append(DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)).Append("\",\n")
                .Append("  \"application\": {\"identifier\":\"").Append(Json(Application.identifier))
                .Append("\",\"version\":\"").Append(Json(Application.version))
                .Append("\",\"buildGuid\":\"").Append(Json(Application.buildGUID))
                .Append("\",\"unityVersion\":\"").Append(Json(Application.unityVersion))
                .Append("\",\"platform\":\"").Append(Json(Application.platform.ToString())).Append("\"},\n")
                .Append("  \"runtime\": {\"deviceModel\":\"").Append(Json(SystemInfo.deviceModel))
                .Append("\",\"operatingSystem\":\"").Append(Json(SystemInfo.operatingSystem))
                .Append("\",\"graphicsDeviceName\":\"").Append(Json(SystemInfo.graphicsDeviceName))
                .Append("\",\"graphicsDeviceVendor\":\"").Append(Json(SystemInfo.graphicsDeviceVendor))
                .Append("\",\"graphicsDeviceVersion\":\"").Append(Json(SystemInfo.graphicsDeviceVersion))
                .Append("\",\"graphicsDeviceType\":\"").Append(Json(SystemInfo.graphicsDeviceType.ToString()))
                .Append("\"},\n")
                .Append("  \"components\": {\n");
            Component[] components =
            {
                DepthCapture.Instance,
                VolumeIntegrator.Instance,
                MeshExtractor.Instance,
                FindAnyObjectByType<StandaloneRoomScanner>(FindObjectsInactive.Include)
            };
            bool first = true;
            foreach (Component component in components)
            {
                if (component == null) continue;
                if (!first) sb.Append(",\n");
                first = false;
                sb.Append("    \"").Append(Json(component.GetType().FullName)).Append("\": ");
                AppendSerializedFields(sb, component);
            }
            sb.Append("\n  }\n}\n");
            return sb.ToString();
        }

        private static void AppendSerializedFields(StringBuilder sb, Component component)
        {
            sb.Append('{');
            bool first = true;
            for (Type type = component.GetType(); type != null && type != typeof(MonoBehaviour); type = type.BaseType)
            {
                FieldInfo[] fields = type.GetFields(BindingFlags.Instance | BindingFlags.Public |
                                                    BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
                Array.Sort(fields, (a, b) => string.CompareOrdinal(a.Name, b.Name));
                foreach (FieldInfo field in fields)
                {
                    bool serializable = field.IsPublic || field.GetCustomAttribute<SerializeField>() != null;
                    if (!serializable || field.IsStatic || field.IsNotSerialized) continue;
                    object value;
                    try { value = field.GetValue(component); }
                    catch { continue; }
                    if (!first) sb.Append(',');
                    first = false;
                    sb.Append("\"").Append(Json(field.Name)).Append("\":");
                    AppendConfigValue(sb, value, 0);
                }
            }
            sb.Append('}');
        }

        private static void AppendConfigValue(StringBuilder sb, object value, int depth)
        {
            if (value == null) { sb.Append("null"); return; }
            if (depth > 3) { sb.Append("\"").Append(Json(value.ToString())).Append("\""); return; }
            switch (value)
            {
                case bool b: sb.Append(b ? "true" : "false"); return;
                case string s: sb.Append('"').Append(Json(s)).Append('"'); return;
                case Enum e: sb.Append('"').Append(Json(e.ToString())).Append('"'); return;
                case UnityEngine.Object o:
                    sb.Append("{\"name\":\"").Append(Json(o.name)).Append("\",\"type\":\"")
                        .Append(Json(o.GetType().FullName)).Append("\"}"); return;
                case Vector2 v2: sb.Append('[').Append(F(v2.x)).Append(',').Append(F(v2.y)).Append(']'); return;
                case Vector3 v3: sb.Append('[').Append(F(v3.x)).Append(',').Append(F(v3.y)).Append(',').Append(F(v3.z)).Append(']'); return;
                case Vector4 v4: sb.Append('[').Append(F(v4.x)).Append(',').Append(F(v4.y)).Append(',').Append(F(v4.z)).Append(',').Append(F(v4.w)).Append(']'); return;
                case Color c: sb.Append('[').Append(F(c.r)).Append(',').Append(F(c.g)).Append(',').Append(F(c.b)).Append(',').Append(F(c.a)).Append(']'); return;
                case Rect r: sb.Append('[').Append(F(r.x)).Append(',').Append(F(r.y)).Append(',').Append(F(r.width)).Append(',').Append(F(r.height)).Append(']'); return;
            }
            if (value is IFormattable formattable && (value.GetType().IsPrimitive || value is decimal))
            {
                sb.Append(formattable.ToString(null, CultureInfo.InvariantCulture));
                return;
            }
            if (value is IEnumerable enumerable)
            {
                sb.Append('[');
                int count = 0;
                foreach (object item in enumerable)
                {
                    if (count > 0) sb.Append(',');
                    if (count++ >= 128) { sb.Append("\"...truncated...\""); break; }
                    AppendConfigValue(sb, item, depth + 1);
                }
                sb.Append(']');
                return;
            }
            sb.Append('"').Append(Json(value.ToString())).Append('"');
        }

        private static string BuildCoordinateContractJson(string phase)
        {
            XROrigin origin = FindAnyObjectByType<XROrigin>(FindObjectsInactive.Include);
            Transform tracking = origin != null && origin.CameraFloorOffsetObject != null
                ? origin.CameraFloorOffsetObject.transform
                : origin != null ? origin.transform : null;
            Camera camera = Camera.main;
            VolumeIntegrator volume = VolumeIntegrator.Instance;
            MeshExtractor extractor = MeshExtractor.Instance;
            var sb = new StringBuilder(4096);
            sb.Append("{\n  \"schema\": \"scancover.coordinate_contract.v1\",\n")
                .Append("  \"phase\": \"").Append(Json(phase)).Append("\",\n")
                .Append("  \"units\": \"metres\",\n")
                .Append("  \"handedness\": \"Unity left-handed world; projection payload retained exactly as supplied by XR\",\n")
                .Append("  \"recordedEyeIndex\": ").Append(RecordedEye).Append(",\n")
                .Append("  \"depthBinaryOrder\": \"x-fastest then y, one right-eye slice\",\n")
                .Append("  \"trackingToWorld\": ");
            AppendMatrix(sb, tracking != null ? tracking.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"xrOriginLocalToWorld\": ");
            AppendMatrix(sb, origin != null ? origin.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"mainCameraLocalToWorld\": ");
            AppendMatrix(sb, camera != null ? camera.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"volumeIntegratorLocalToWorld\": ");
            AppendMatrix(sb, volume != null ? volume.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"meshExtractorLocalToWorld\": ");
            AppendMatrix(sb, extractor != null ? extractor.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append("\n}\n");
            return sb.ToString();
        }

        private static string BuildFusionMetaJson(FusionRecord r, string depth, string normal,
            string dilated, string edge, string temporal, string camera, string observations,
            string correspondences, string preTransactionIdentity,
            string correspondenceIdentity, string status)
        {
            var sb = new StringBuilder(4096);
            sb.Append("{\n  \"schema\": \"scancover.fusion_input.v2\",\n")
                .Append("  \"sequence\": ").Append(r.Sequence).Append(",\n")
                .Append("  \"attemptIndex\": ").Append(r.AttemptIndex).Append(",\n")
                .Append("  \"sourceFrame\": ").Append(r.SourceFrame).Append(",\n")
                .Append("  \"unityFrame\": ").Append(r.UnityFrame).Append(",\n")
                .Append("  \"scaledTime\": ").Append(D(r.ScaledTime)).Append(",\n")
                .Append("  \"unscaledTime\": ").Append(D(r.UnscaledTime)).Append(",\n")
                .Append("  \"accepted\": ").Append(r.Accepted ? "true" : "false").Append(",\n")
                .Append("  \"decision\": \"").Append(Json(r.Decision)).Append("\",\n")
                .Append("  \"guardedGunGel\": ").Append(r.Guarded ? "true" : "false").Append(",\n")
                .Append("  \"gunGelAdmissionActive\": ").Append(r.GunGelAdmissionActive ? "true" : "false").Append(",\n")
                .Append("  \"gunGelFrame\": ").Append(r.GunGelFrame).Append(",\n")
                .Append("  \"translationMm\": ").Append(F(r.TranslationMm)).Append(",\n")
                .Append("  \"rotationDeg\": ").Append(F(r.RotationDeg)).Append(",\n")
                .Append("  \"angularDegPerSec\": ").Append(F(r.AngularSpeed)).Append(",\n")
                .Append("  \"linearMps\": ").Append(F(r.LinearSpeed)).Append(",\n")
                .Append("  \"motionQuality\": ").Append(F(r.MotionQuality)).Append(",\n")
                .Append("  \"fusionHead\": {\"available\":")
                .Append(r.FusionHeadAvailable ? "true" : "false")
                .Append(",\"position\":");
            AppendVector3(sb, r.FusionHeadPosition);
            sb.Append(",\"rotationQuaternion\":");
            AppendQuaternion(sb, r.FusionHeadRotation);
            sb.Append(",\"eulerDegrees\":");
            AppendVector3(sb, r.FusionHeadEuler);
            sb.Append("},\n")
                .Append("  \"fusionEyeFromViewInverse\": {\"available\":")
                .Append(r.FusionEyeAvailable ? "true" : "false")
                .Append(",\"position\":");
            AppendVector3(sb, r.FusionEyePosition);
            sb.Append(",\"forward\":");
            AppendVector3(sb, r.FusionEyeForward);
            sb.Append(",\"headToEyeMm\":").Append(F(r.HeadToFusionEyeMm)).Append("},\n")
                .Append("  \"recordedEyeIndex\": ").Append(RecordedEye).Append(",\n")
                .Append("  \"status\": \"").Append(Json(status)).Append("\",\n")
                .Append("  \"textures\": {");
            AppendTextureMeta(sb, "depth", depth, r.Textures.depthDescriptor, true);
            AppendTextureMeta(sb, "normal", normal, r.Textures.normalDescriptor, true);
            AppendTextureMeta(sb, "dilated", dilated, r.Textures.dilatedDescriptor, true);
            AppendTextureMeta(sb, "edgeReason", edge, r.Textures.edgeDescriptor, true);
            AppendTextureMeta(sb, "temporalReason", temporal, r.Textures.temporalDescriptor, true);
            AppendTextureMeta(sb, "cameraRgb", camera, r.Textures.cameraDescriptor, false);
            sb.Append("\n  },\n  \"cameraColor\": {\"available\":")
                .Append(r.CameraAvailable ? "true" : "false")
                .Append(",\"position\":");
            AppendVector3(sb, r.CameraPosition);
            sb.Append(",\"rotationQuaternion\":");
            AppendQuaternion(sb, r.CameraRotation);
            sb.Append(",\"eulerDegrees\":");
            AppendVector3(sb, r.CameraEuler);
            sb.Append(",\"focalLength\":"); AppendVector2(sb, r.CameraFocalLength);
            sb.Append(",\"principalPoint\":"); AppendVector2(sb, r.CameraPrincipalPoint);
            sb.Append(",\"sensorResolution\":"); AppendVector2(sb, r.CameraSensorResolution);
            sb.Append(",\"currentResolution\":"); AppendVector2(sb, r.CameraCurrentResolution);
            sb.Append("},\n  \"fusionCorrection\": ");
            AppendMatrix(sb, r.FusionCorrection);
            sb.Append(",\n  \"exclusions\": {\"count\":").Append(r.ExclusionCount)
                .Append(",\"positions\":[");
            for (int i = 0; i < r.ExclusionCount; i++)
            {
                if (i > 0) sb.Append(',');
                Vector4 position = r.ExclusionPositions[i];
                sb.Append('[').Append(F(position.x)).Append(',').Append(F(position.y))
                    .Append(',').Append(F(position.z)).Append(',').Append(F(position.w)).Append(']');
            }
            sb.Append("]},\n  \"gunGelAdmission\": {\"active\":")
                .Append(r.GunGelAdmissionActive ? "true" : "false")
                .Append(",\"observationFile\":\"")
                .Append(Json(observations)).Append("\",\"correspondenceFile\":\"")
                .Append(Json(correspondences)).Append("\",\"observationCount\":")
                .Append(r.Textures.gunGelObservationCount).Append(",\"observationStrideBytes\":")
                .Append(r.Textures.gunGelObservationStride)
                .Append(",\"observationLayout\":\"float4 positionSigma; float4 normalQuality; float4 rawPositionDelta; uint4 sourceReason(packedPixel,edgeReason,temporalReason,platformFrame)\",\"correspondenceCount\":")
                .Append(r.Textures.gunGelCorrespondenceCount).Append(",\"correspondenceStrideBytes\":")
                .Append(r.Textures.gunGelCorrespondenceStride)
                .Append(",\"correspondenceEvidenceSchema\":\"v2_low6_authority_high3_stable_found_residual_normal\",\"preTransactionIdentityFile\":\"")
                .Append(Json(preTransactionIdentity))
                .Append("\",\"preTransactionIdentityCount\":")
                .Append(r.Textures.gunGelPreTransactionCorrespondenceIdentityCount)
                .Append(",\"preTransactionIdentityStrideBytes\":")
                .Append(r.Textures.gunGelPreTransactionCorrespondenceIdentityStride)
                .Append(",\"finalIdentityFile\":\"")
                .Append(Json(correspondenceIdentity))
                .Append("\",\"finalIdentityCount\":")
                .Append(r.Textures.gunGelCorrespondenceIdentityCount)
                .Append(",\"finalIdentityStrideBytes\":")
                .Append(r.Textures.gunGelCorrespondenceIdentityStride)
                .Append(",\"identityFile\":\"")
                .Append(Json(correspondenceIdentity))
                .Append("\",\"identityCount\":")
                .Append(r.Textures.gunGelCorrespondenceIdentityCount)
                .Append(",\"identityStrideBytes\":")
                .Append(r.Textures.gunGelCorrespondenceIdentityStride)
                .Append(",\"identityLayout\":\"uint4(candidateIndexPlusOne,stableId,dualSupportFrames,oppositionVotes)\",\"identityTransitionJoin\":\"same observation index: preTransactionIdentity -> finalIdentity\",\"identityAuthority\":\"none_diagnostic_only\",\"gridX\":")
                .Append(r.GunGelGridX).Append(",\"gridY\":").Append(r.GunGelGridY)
                .Append(",\"pixelStride\":").Append(r.GunGelPixelStride).Append("},\n  \"projection\": ");
            AppendMatrixArray(sb, r.Projection);
            sb.Append(",\n  \"view\": "); AppendMatrixArray(sb, r.View);
            sb.Append(",\n  \"projectionInverse\": "); AppendMatrixArray(sb, r.ProjectionInverse);
            sb.Append(",\n  \"viewInverse\": "); AppendMatrixArray(sb, r.ViewInverse);
            sb.Append("\n}\n");
            return sb.ToString();
        }

        private static void AppendTextureMeta(StringBuilder sb, string role, string file,
            TextureDescriptor descriptor, bool comma)
        {
            sb.Append("\n    \"").Append(role).Append("\": {\"file\":\"").Append(Json(file))
                .Append("\",\"width\":").Append(descriptor.width)
                .Append(",\"height\":").Append(descriptor.height)
                .Append(",\"graphicsFormat\":\"").Append(Json(descriptor.format))
                .Append("\",\"sourceDimension\":\"").Append(Json(descriptor.dimension))
                .Append("\",\"sourceSlice\":").Append(descriptor.slice).Append('}');
            if (comma) sb.Append(',');
        }

        private static void AppendMatrixArray(StringBuilder sb, Matrix4x4[] matrices)
        {
            sb.Append('[');
            for (int i = 0; i < 2; i++)
            {
                if (i > 0) sb.Append(',');
                AppendMatrix(sb, matrices != null && matrices.Length > i ? matrices[i] : Matrix4x4.identity);
            }
            sb.Append(']');
        }

        private static void AppendVector2(StringBuilder sb, Vector2 value) =>
            sb.Append('[').Append(F(value.x)).Append(',').Append(F(value.y)).Append(']');

        private static void AppendVector3(StringBuilder sb, Vector3 value) =>
            sb.Append('[').Append(F(value.x)).Append(',').Append(F(value.y)).Append(',')
                .Append(F(value.z)).Append(']');

        private static void AppendQuaternion(StringBuilder sb, Quaternion value) =>
            sb.Append('[').Append(F(value.x)).Append(',').Append(F(value.y)).Append(',')
                .Append(F(value.z)).Append(',').Append(F(value.w)).Append(']');

        private static void AppendMatrix(StringBuilder sb, Matrix4x4 matrix)
        {
            sb.Append('[');
            for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                if (row != 0 || col != 0) sb.Append(',');
                sb.Append(F(matrix[row, col]));
            }
            sb.Append(']');
        }

        private static Matrix4x4[] CloneMatrices(Matrix4x4[] source)
        {
            return source != null && source.Length >= 2
                ? new[] { source[0], source[1] }
                : new[] { Matrix4x4.identity, Matrix4x4.identity };
        }

        private static Vector4[] CloneVectors(Vector4[] source, int count)
        {
            int length = Mathf.Clamp(count, 0, source != null ? source.Length : 0);
            var result = new Vector4[length];
            if (length > 0) Array.Copy(source, result, length);
            return result;
        }

        private void RecordProbeConversion(FusionRecord record,
            NativeArray<GunGelEvidenceShadow.Correspondence> finalCorrespondences)
        {
            if (string.IsNullOrEmpty(_probeConversionManifest)) return;
            try
            {
                ProbeStageCounts post = CountProbeStages(finalCorrespondences);
                bool hasPre = _preProbeStages.TryRemove(record.GunGelFrame,
                    out ProbeStageCounts pre);
                if (!hasPre) Interlocked.Increment(ref _probeConversionMissingPreRows);
                string row = string.Join(",",
                    record.GunGelFrame.ToString(CultureInfo.InvariantCulture),
                    record.SourceFrame.ToString(CultureInfo.InvariantCulture),
                    pre.Valid, post.Valid, pre.Raw, post.Raw, pre.Dual, post.Dual,
                    pre.StableFound, post.StableFound,
                    pre.ResidualPass, post.ResidualPass,
                    pre.NormalPass, post.NormalPass,
                    pre.StableMatch, post.StableMatch,
                    pre.StableDual, post.StableDual,
                    pre.Unopposed, post.Unopposed,
                    pre.Authority, post.Authority,
                    post.StableFound - pre.StableFound,
                    post.StableMatch - pre.StableMatch,
                    post.Authority - pre.Authority,
                    hasPre ? "paired" : "missing_pre") + "\n";
                lock (_fileLock)
                    File.AppendAllText(_probeConversionManifest, row,
                        new UTF8Encoding(false));
                Interlocked.Increment(ref _probeConversionRows);
            }
            catch
            {
                Interlocked.Increment(ref _probeConversionWriteErrors);
                // 转换诊断失败不得污染最终 correspondence 的原始回放保存。
            }
        }

        private static ProbeStageCounts CountProbeStages(
            NativeArray<GunGelEvidenceShadow.Correspondence> correspondences)
        {
            ProbeStageCounts counts = default;
            for (int i = 0; i < correspondences.Length; i++)
            {
                uint flags = (uint)Mathf.Max(0,
                    Mathf.RoundToInt(correspondences[i].SourceValid.w));
                if ((flags & (1u << 0)) != 0u) counts.Valid++;
                if ((flags & (1u << 1)) != 0u) counts.Raw++;
                if ((flags & (1u << 2)) != 0u) counts.Dual++;
                if ((flags & (1u << 6)) != 0u ||
                    (flags & (1u << 3)) != 0u) counts.StableFound++;
                if ((flags & (1u << 7)) != 0u) counts.ResidualPass++;
                if ((flags & (1u << 8)) != 0u) counts.NormalPass++;
                if ((flags & (1u << 3)) != 0u) counts.StableMatch++;
                if ((flags & (1u << 4)) != 0u) counts.StableDual++;
                if ((flags & (1u << 5)) != 0u) counts.Unopposed++;
                if ((flags & 0x3fu) == 0x3fu) counts.Authority++;
            }
            return counts;
        }

        private static void CopyDirectory(string source, string destination)
        {
            Directory.CreateDirectory(destination);
            foreach (string directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(Path.Combine(destination, Path.GetRelativePath(source, directory)));
            foreach (string file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            {
                string target = Path.Combine(destination, Path.GetRelativePath(source, file));
                Directory.CreateDirectory(Path.GetDirectoryName(target) ?? destination);
                File.Copy(file, target, true);
            }
        }

        private static string Hex(byte[] bytes) => BitConverter.ToString(bytes).Replace("-", string.Empty).ToLowerInvariant();
        private static string F(float value) => float.IsNaN(value) || float.IsInfinity(value)
            ? "null" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string D(double value) => double.IsNaN(value) || double.IsInfinity(value)
            ? "null" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string Csv(string value) => (value ?? string.Empty).Replace(',', ';').Replace('\n', ' ').Replace('\r', ' ');
        private static string Json(string value) => (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");

        private sealed class SystemMeshSnapshot
        {
            internal readonly int RoomIndex;
            internal readonly string AnchorName;
            internal readonly Matrix4x4 LocalToWorld;
            internal readonly Vector3[] Vertices;
            internal readonly int[] Triangles;

            internal SystemMeshSnapshot(int roomIndex, string anchorName,
                Matrix4x4 localToWorld, Vector3[] vertices, int[] triangles)
            {
                RoomIndex = roomIndex;
                AnchorName = anchorName ?? string.Empty;
                LocalToWorld = localToWorld;
                Vertices = vertices ?? Array.Empty<Vector3>();
                Triangles = triangles ?? Array.Empty<int>();
            }
        }

        private struct ProbeStageCounts
        {
            internal int Valid;
            internal int Raw;
            internal int Dual;
            internal int StableFound;
            internal int ResidualPass;
            internal int NormalPass;
            internal int StableMatch;
            internal int StableDual;
            internal int Unopposed;
            internal int Authority;
        }

        private readonly struct TextureDescriptor
        {
            internal readonly int width;
            internal readonly int height;
            internal readonly string format;
            internal readonly string dimension;
            internal readonly int slice;
            internal TextureDescriptor(int width, int height, string format, string dimension, int slice)
            {
                this.width = width; this.height = height; this.format = format;
                this.dimension = dimension; this.slice = slice;
            }
        }

        private sealed class PendingTextureSet
        {
            internal byte[] depth, normal, dilated, edge, temporal, camera,
                gunGelObservations, gunGelCorrespondences,
                gunGelPreTransactionCorrespondenceIdentity,
                gunGelCorrespondenceIdentity;
            internal GunGelEvidenceShadow.Correspondence[]
                gunGelCorrespondenceRows;
            internal Unity.Mathematics.uint4[] gunGelCorrespondenceIdentityRows;
            internal bool depthDone, normalDone, dilatedDone, edgeDone, temporalDone, cameraDone,
                gunGelObservationsDone, gunGelCorrespondencesDone,
                gunGelPreTransactionCorrespondenceIdentityDone,
                gunGelCorrespondenceIdentityDone;
            internal bool HasError;
            internal TextureDescriptor depthDescriptor, normalDescriptor, dilatedDescriptor,
                edgeDescriptor, temporalDescriptor, cameraDescriptor;
            internal int gunGelObservationCount, gunGelObservationStride,
                gunGelCorrespondenceCount, gunGelCorrespondenceStride,
                gunGelPreTransactionCorrespondenceIdentityCount,
                gunGelPreTransactionCorrespondenceIdentityStride,
                gunGelCorrespondenceIdentityCount, gunGelCorrespondenceIdentityStride;
            internal bool AllDone => depthDone && normalDone && dilatedDone && edgeDone &&
                                     temporalDone && cameraDone &&
                                     gunGelObservationsDone && gunGelCorrespondencesDone &&
                                     gunGelPreTransactionCorrespondenceIdentityDone &&
                                     gunGelCorrespondenceIdentityDone;

            internal void SetDescriptor(string role, TextureDescriptor value)
            {
                switch (role)
                {
                    case "depth": depthDescriptor = value; break;
                    case "normal": normalDescriptor = value; break;
                    case "dilated": dilatedDescriptor = value; break;
                    case "edge": edgeDescriptor = value; break;
                    case "temporal": temporalDescriptor = value; break;
                    case "camera": cameraDescriptor = value; break;
                }
            }
            internal void SetBufferDescriptor(string role, int count, int stride)
            {
                if (role == "gungel_observations")
                {
                    gunGelObservationCount = count; gunGelObservationStride = stride;
                }
                else if (role == "gungel_correspondences")
                {
                    gunGelCorrespondenceCount = count; gunGelCorrespondenceStride = stride;
                }
                else if (role == "gungel_correspondence_identity")
                {
                    gunGelCorrespondenceIdentityCount = count;
                    gunGelCorrespondenceIdentityStride = stride;
                }
                else if (role == "gungel_pre_transaction_identity")
                {
                    gunGelPreTransactionCorrespondenceIdentityCount = count;
                    gunGelPreTransactionCorrespondenceIdentityStride = stride;
                }
            }
            internal void SetBytes(string role, byte[] value)
            {
                switch (role)
                {
                    case "depth": depth = value; depthDone = true; break;
                    case "normal": normal = value; normalDone = true; break;
                    case "dilated": dilated = value; dilatedDone = true; break;
                    case "edge": edge = value; edgeDone = true; break;
                    case "temporal": temporal = value; temporalDone = true; break;
                    case "camera": camera = value; cameraDone = true; break;
                    case "gungel_observations": gunGelObservations = value; gunGelObservationsDone = true; break;
                    case "gungel_correspondences": gunGelCorrespondences = value; gunGelCorrespondencesDone = true; break;
                    case "gungel_pre_transaction_identity": gunGelPreTransactionCorrespondenceIdentity = value; gunGelPreTransactionCorrespondenceIdentityDone = true; break;
                    case "gungel_correspondence_identity": gunGelCorrespondenceIdentity = value; gunGelCorrespondenceIdentityDone = true; break;
                }
            }
            internal void MarkMissing(string role) { MarkDone(role); }
            internal void MarkError(string role) { HasError = true; MarkDone(role); }
            private void MarkDone(string role)
            {
                switch (role)
                {
                    case "depth": depthDone = true; break;
                    case "normal": normalDone = true; break;
                    case "dilated": dilatedDone = true; break;
                    case "edge": edgeDone = true; break;
                    case "temporal": temporalDone = true; break;
                    case "camera": cameraDone = true; break;
                    case "gungel_observations": gunGelObservationsDone = true; break;
                    case "gungel_correspondences": gunGelCorrespondencesDone = true; break;
                    case "gungel_pre_transaction_identity": gunGelPreTransactionCorrespondenceIdentityDone = true; break;
                    case "gungel_correspondence_identity": gunGelCorrespondenceIdentityDone = true; break;
                }
            }
        }

        private sealed class FusionRecord
        {
            internal readonly int Sequence, AttemptIndex, SourceFrame, GunGelFrame;
            internal readonly int UnityFrame;
            internal readonly double ScaledTime, UnscaledTime;
            internal readonly bool Accepted, Guarded, GunGelAdmissionActive, CameraAvailable;
            internal readonly bool FusionHeadAvailable;
            internal readonly string Decision;
            internal readonly float TranslationMm, RotationDeg, AngularSpeed, LinearSpeed, MotionQuality;
            internal readonly Matrix4x4[] Projection, View, ProjectionInverse, ViewInverse;
            internal readonly PendingTextureSet Textures;
            internal readonly int GunGelGridX, GunGelGridY, GunGelPixelStride;
            internal readonly Vector3 CameraPosition;
            internal readonly Quaternion CameraRotation;
            internal readonly Vector3 FusionHeadPosition;
            internal readonly Quaternion FusionHeadRotation;
            internal readonly Vector3 FusionHeadEuler;
            internal readonly bool FusionEyeAvailable;
            internal readonly Vector3 FusionEyePosition;
            internal readonly Vector3 FusionEyeForward;
            internal readonly float HeadToFusionEyeMm;
            internal readonly Vector3 CameraEuler;
            internal readonly Vector2 CameraFocalLength, CameraPrincipalPoint,
                CameraSensorResolution, CameraCurrentResolution;
            internal readonly Matrix4x4 FusionCorrection;
            internal readonly Vector4[] ExclusionPositions;
            internal readonly int ExclusionCount;
            internal bool CompletionClaimed;
            internal FusionRecord(int sequence, int attemptIndex, int sourceFrame, bool accepted,
                string decision, bool guarded, int gunGelFrame, float translationMm,
                float rotationDeg, float angularSpeed, float linearSpeed, float motionQuality,
                Matrix4x4[] projection, Matrix4x4[] view, Matrix4x4[] projectionInverse,
                Matrix4x4[] viewInverse, PendingTextureSet textures,
                bool gunGelAdmissionActive, bool cameraAvailable,
                Vector3 cameraPosition, Quaternion cameraRotation,
                Vector2 cameraFocalLength, Vector2 cameraPrincipalPoint,
                Vector2 cameraSensorResolution, Vector2 cameraCurrentResolution,
                Matrix4x4 fusionCorrection, Vector4[] exclusionPositions,
                int exclusionCount,
                int gunGelGridX = 0, int gunGelGridY = 0, int gunGelPixelStride = 0)
            {
                Sequence = sequence; AttemptIndex = attemptIndex; SourceFrame = sourceFrame;
                UnityFrame = Time.frameCount; ScaledTime = Time.timeAsDouble;
                UnscaledTime = Time.unscaledTimeAsDouble;
                Camera fusionHead = Camera.main;
                FusionHeadAvailable = fusionHead != null;
                FusionHeadPosition = fusionHead != null
                    ? fusionHead.transform.position : Vector3.zero;
                FusionHeadRotation = fusionHead != null
                    ? fusionHead.transform.rotation : Quaternion.identity;
                FusionHeadEuler = fusionHead != null
                    ? fusionHead.transform.eulerAngles : Vector3.zero;
                FusionEyeAvailable = viewInverse != null && viewInverse.Length > RecordedEye;
                Matrix4x4 fusionEye = FusionEyeAvailable
                    ? viewInverse[RecordedEye]
                    : Matrix4x4.identity;
                FusionEyePosition = FusionEyeAvailable
                    ? new Vector3(fusionEye.m03, fusionEye.m13, fusionEye.m23)
                    : Vector3.zero;
                FusionEyeForward = FusionEyeAvailable
                    // Depth matrices include ScaleFlipZ; local -Z is the
                    // physical viewing direction in this exact view inverse.
                    ? fusionEye.MultiplyVector(Vector3.back).normalized : Vector3.zero;
                HeadToFusionEyeMm = FusionHeadAvailable && FusionEyeAvailable
                    ? Vector3.Distance(FusionHeadPosition, FusionEyePosition) * 1000f
                    : float.NaN;
                Accepted = accepted; Decision = decision ?? string.Empty; Guarded = guarded;
                GunGelFrame = gunGelFrame; TranslationMm = translationMm; RotationDeg = rotationDeg;
                AngularSpeed = angularSpeed; LinearSpeed = linearSpeed; MotionQuality = motionQuality;
                Projection = projection; View = view; ProjectionInverse = projectionInverse;
                ViewInverse = viewInverse; Textures = textures;
                GunGelAdmissionActive = gunGelAdmissionActive;
                CameraAvailable = cameraAvailable; CameraPosition = cameraPosition;
                CameraRotation = cameraRotation; CameraFocalLength = cameraFocalLength;
                CameraEuler = cameraRotation.eulerAngles;
                CameraPrincipalPoint = cameraPrincipalPoint;
                CameraSensorResolution = cameraSensorResolution;
                CameraCurrentResolution = cameraCurrentResolution;
                FusionCorrection = fusionCorrection;
                ExclusionPositions = exclusionPositions ?? Array.Empty<Vector4>();
                ExclusionCount = Mathf.Clamp(exclusionCount, 0, ExclusionPositions.Length);
                GunGelGridX = gunGelGridX; GunGelGridY = gunGelGridY;
                GunGelPixelStride = gunGelPixelStride;
            }
        }
    }
}

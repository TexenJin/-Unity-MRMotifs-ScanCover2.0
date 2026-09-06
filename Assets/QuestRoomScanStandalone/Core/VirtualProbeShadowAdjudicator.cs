using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 虚拟探针的持续裁决器。它逐单元维护“暂缓/放行/反对”状态。
    /// 生产候选、TSDF 和网格仍不读取本类；“合流仅纸”可只读纠错
    /// 快照撤销局部显示权，恢复后仍要等实际纸面贴合新目标才重新显示。
    /// </summary>
    internal sealed class VirtualProbeShadowAdjudicator : IDisposable
    {
        // 第一批 Quest 3 房间样本只用来校准量级。结构门槛刻意采用可解释的
        // 独立视角/连续反证票，而不是为某个房间拟合一串表面类别阈值。
        private const float CellMetres = 0.05f;
        private const float IndependentBaselineMetres =
            IndependentViewWitnessPolicy.BaselineMetres;
        private const float IndependentAngleDeg =
            IndependentViewWitnessPolicy.AngleDeg;
        private const int IndependentFrameGap =
            IndependentViewWitnessPolicy.FrameGap;
        private const float SafeMotionQuality = 0.70f;
        private const float AuthorityOuterRadius = 0.84f;
        private const int RequiredSupportFrames = 2;
        private const int RequiredSupportViews = 2;
        private const float ReliableFreeGapMetres = 0.05f;
        private const int RequiredChallengeVotes = 3;
        private const int RequiredChallengeViews = 2;
        private const int RequiredRecoveryViews = 2;
        private const int FreeSpaceRayStride = 8;
        private const int SeenWitnessCapacity = 4;
        private const int SupportWitnessCapacity = 4;
        private const int ChallengeWitnessCapacity = 4;
        private const int RecoveryWitnessCapacity = 2;
        private const int FingerprintPatchCapacity = 12;
        private const int RecoveryPatchCapacity = 4;
        // GunGel 在当前设备上约 16 Hz；480 帧约等于 30 秒。它现在只是
        // 离线报告的最小观察时域，不再让空间任务失效。未结案 Reject 会
        // 保持到本次会话结束，或被自由空间/实体支持证据明确结案。
        private const int PostVerdictWindowFrames = 480;
        private const int RequiredFollowupFreeViews = 2;
        private const int FollowupWitnessCapacity = 4;
        // The fixed frame is a read-only local window, deliberately wider
        // than a single 5 cm cell. Shadow evidence is acquired passively over
        // the trusted depth field; this angle only controls what is drawn.
        // The adjudicator remains diagnostic-only and cannot change geometry.
        private const float GuidanceFrameHalfAngleDeg = 7f;
        private const float GuidanceLockReleaseHalfAngleDeg = 13f;
        private const int GuidanceLockReleaseFrames = 12;
        // Background markers are selected only from their immutable world
        // cell key.  Unlike camera-relative angular bins, this sample never
        // changes when the operator turns or translates the headset.
        private const int GuidanceBackgroundHashMask = 15; // about 1 / 16
        private const int GuidanceClearedTraceFrames = 120;

        private const uint ObservationValid = 1u << 0;
        private const uint RawAvailable = 1u << 1;
        private const uint DualAgree = 1u << 2;
        private const uint StableMatch = 1u << 3;
        private const uint Unopposed = 1u << 5;
        private const uint StableFound = 1u << 6;
        private const uint ResidualPass = 1u << 7;
        private const uint NormalPass = 1u << 8;

        internal readonly struct FrameResult
        {
            internal readonly int NewIndependentViewCells;
            internal readonly int CorrelatedViewMatches;
            internal readonly float MaxBaselineMetres;
            internal readonly float MaxViewSpreadDeg;
            internal readonly int FreeSpaceChallenges;
            internal readonly int FreeGapLt20;
            internal readonly int FreeGap20To50;
            internal readonly int FreeGap50To100;
            internal readonly int FreeGapGe100;
            internal readonly int IndependentFreeSpaceChallenges;
            internal readonly float MaxFreeSpaceGapMetres;
            internal readonly int FingerprintAssociationTests;
            internal readonly int FingerprintAssociationHits;
            internal readonly int FingerprintAssociationMisses;

            internal FrameResult(FrameCounters counters)
            {
                NewIndependentViewCells = counters.NewIndependentViewCells;
                CorrelatedViewMatches = counters.CorrelatedViewMatches;
                MaxBaselineMetres = counters.MaxBaselineMetres;
                MaxViewSpreadDeg = counters.MaxViewSpreadDeg;
                FreeSpaceChallenges = counters.FreeSpaceChallenges;
                FreeGapLt20 = counters.FreeGapLt20;
                FreeGap20To50 = counters.FreeGap20To50;
                FreeGap50To100 = counters.FreeGap50To100;
                FreeGapGe100 = counters.FreeGapGe100;
                IndependentFreeSpaceChallenges = counters.IndependentFreeSpaceChallenges;
                MaxFreeSpaceGapMetres = counters.MaxFreeSpaceGapMetres;
                FingerprintAssociationTests = counters.FingerprintAssociationTests;
                FingerprintAssociationHits = counters.FingerprintAssociationHits;
                FingerprintAssociationMisses = counters.FingerprintAssociationMisses;
            }
        }

        internal readonly struct GuidanceTargetSnapshot
        {
            internal readonly bool HasTarget;
            internal readonly Vector3 WorldPosition;
            internal readonly bool InAcquisitionFrame;
            internal readonly bool ObservedRecently;
            internal readonly bool PoseReady;
            // 0=pending, 1=Reject confirmed by free space,
            // 2=Reject was a false rejection and support returned, 3=expired.
            internal readonly int Resolution;
            internal readonly float DistanceMetres;
            internal readonly int FreeWitnessCount;
            internal readonly int SupportWitnessCount;

            internal GuidanceTargetSnapshot(
                Vector3 worldPosition,
                bool inAcquisitionFrame,
                bool observedRecently,
                bool poseReady,
                int resolution,
                float distanceMetres,
                int freeWitnessCount,
                int supportWitnessCount)
            {
                HasTarget = true;
                WorldPosition = worldPosition;
                InAcquisitionFrame = inAcquisitionFrame;
                ObservedRecently = observedRecently;
                PoseReady = poseReady;
                Resolution = resolution;
                DistanceMetres = distanceMetres;
                FreeWitnessCount = freeWitnessCount;
                SupportWitnessCount = supportWitnessCount;
            }
        }

        internal readonly struct GuidanceFrameSnapshot
        {
            internal readonly bool Active;
            internal readonly int RegisteredFingerprints;
            internal readonly int ChallengeOne;
            internal readonly int ChallengeTwo;
            internal readonly int Pending;
            internal readonly int ObservedRecently;
            internal readonly int ResolvedFree;
            internal readonly int ResolvedSupport;
            internal readonly int Expired;
            internal readonly int ChallengesClearedBySupport;

            internal GuidanceFrameSnapshot(int registeredFingerprints,
                int challengeOne, int challengeTwo, int pending,
                int observedRecently, int resolvedFree, int resolvedSupport,
                int expired, int challengesClearedBySupport)
            {
                Active = true;
                RegisteredFingerprints = registeredFingerprints;
                ChallengeOne = challengeOne;
                ChallengeTwo = challengeTwo;
                Pending = pending;
                ObservedRecently = observedRecently;
                ResolvedFree = resolvedFree;
                ResolvedSupport = resolvedSupport;
                Expired = expired;
                ChallengesClearedBySupport = challengesClearedBySupport;
            }
        }

        // Operator-visible provenance for the fixed acquisition frame.
        // 0=provisional support, 1=stable fingerprint, 2=one free-space vote,
        // 3=two votes, 4=Reject, 5=Reject reconfirmed by free space,
        // 6=Reject reopened by independent surface support,
        // 7/8=one/two challenge votes later cleared by safe support (trace).
        internal readonly struct GuidanceCellVisual
        {
            internal readonly Vector3 WorldPosition;
            internal readonly int Phase;
            internal readonly float Strength;

            internal GuidanceCellVisual(Vector3 worldPosition, int phase,
                float strength)
            {
                WorldPosition = worldPosition;
                Phase = phase;
                Strength = Mathf.Clamp01(strength);
            }
        }

        /// <summary>给纸皮最终绘制端的持续纠错快照。坐标和法向轴
        /// 与内部 5 cm CellKey 完全同源，不携带单帧颜色判定。</summary>
        internal readonly struct PaperCorrectionCell
        {
            internal readonly int X;
            internal readonly int Y;
            internal readonly int Z;
            internal readonly int Axis;
            // 1=confirmed Reject: hide the complete local paper;
            // 2=recovered support: publish only paper already aligned to Target.
            internal readonly int Mode;
            internal readonly Vector3 Target;

            internal PaperCorrectionCell(int x, int y, int z, int axis,
                int mode, Vector3 target)
            {
                X = x;
                Y = y;
                Z = z;
                Axis = axis;
                Mode = mode;
                Target = target;
            }
        }

        private enum Verdict
        {
            Hold = 0,
            Accept = 1,
            Reject = 2
        }

        private struct ViewWitness
        {
            internal Vector3 Camera;
            internal Vector3 Direction;
            internal int Frame;
        }

        private struct SurfacePatch
        {
            internal Vector3 Center;
            internal Vector3 Normal;
            internal float RadiusMetres;
            internal int Frame;
        }

        private struct FingerprintRecord
        {
            internal CellKey Key;
            internal int FingerprintId;
            internal int Generation;
            internal int AcceptedFrame;
            internal int PatchIndex;
            internal SurfacePatch Patch;
        }

        private sealed class PostVerdictRecord
        {
            internal CellKey Key;
            internal int FingerprintId;
            internal int FingerprintGeneration;
            internal Vector3 FingerprintCenter;
            internal int RejectFrame;
            internal int RejectPlatformFrame;
            internal int FirstFollowupFrame = -1;
            internal int LastFollowupFrame = -1;
            internal int ObservedFrames;
            internal int LastObservedFrame = -1;
            internal int ContactFrames;
            internal int LastContactFrame = -1;
            internal int RayIntersectionFrames;
            internal int LastRayIntersectionFrame = -1;
            internal int ReliableFreeSpaceFrames;
            internal int LastReliableFreeSpaceFrame = -1;
            internal ViewWitness[] FreeSpaceWitnesses;
            internal int FreeSpaceWitnessCount;
            internal int SafeSupportFrames;
            internal int LastSafeSupportFrame = -1;
            internal ViewWitness[] SupportWitnesses;
            internal int SupportWitnessCount;
            internal bool FreeSpaceReconfirmed;
            internal int FreeSpaceReconfirmedFrame = -1;
            internal bool Reopened;
            internal int ReopenedFrame = -1;
            internal int ReopenedFingerprintId;
            internal int ReopenedGeneration;
            internal float MaxFreeGapMetres;
            internal float MaxBaselineMetres;
            internal float MaxViewSpreadDeg;
        }

        private sealed class CellState
        {
            internal Vector3 Center;
            internal Vector3 DisplayAnchor;
            internal Vector3 Normal;
            internal int FirstFrame;
            internal int LastSeenFrame;
            internal int LastSafeSupportFrame = -1;
            internal Vector3 FirstSeenCamera;
            internal Vector3 FirstSeenDirection;
            internal bool HasSafeSupport;
            internal ViewWitness[] SeenWitnesses;
            internal int SeenWitnessCount;
            internal ViewWitness[] SupportWitnesses;
            internal int SupportWitnessCount;
            internal ViewWitness[] ChallengeWitnesses;
            internal int ChallengeWitnessCount;
            internal ViewWitness[] RecoveryWitnesses;
            internal int RecoveryWitnessCount;
            internal int SafeSupportFrames;
            internal int IndependentSupportViews;
            internal int ConsecutiveChallengeVotes;
            internal int LastChallengePeakVotes;
            internal int ChallengeClearedFrame = -1;
            internal int IndependentChallengeViews;
            internal int RecoverySupportViews;
            internal float MaxBaselineMetres;
            internal float MaxViewSpreadDeg;
            internal float MaxFreeGapMetres;
            internal int FingerprintId;
            internal int FingerprintGeneration;
            internal int FingerprintAcceptedFrame = -1;
            internal Vector3 FingerprintCenter;
            internal Vector3 FingerprintNormal;
            internal SurfacePatch[] FingerprintPatches;
            internal int FingerprintPatchCount;
            internal SurfacePatch[] RecoveryPatches;
            internal int RecoveryPatchCount;
            // Once a stable surface is rejected, its paper cell remains under
            // the corrected-surface publication gate for the whole session.
            internal bool PaperCorrectionGateActive;
            internal Verdict Verdict = Verdict.Hold;
        }

        internal struct FrameCounters
        {
            internal int Valid;
            internal int HoldRawMissing;
            internal int HoldDualDisagree;
            internal int HoldNoStable;
            internal int HoldResidual;
            internal int HoldNormal;
            internal int HoldOpposed;
            internal int HoldMotion;
            internal int HoldOuter;
            internal int EligibleSupport;
            internal int IndependentSupport;
            internal int ObservationAccept;
            internal int ObservationHold;
            internal int ObservationReject;
            internal int AcceptTransitions;
            internal int RejectTransitions;
            internal int ReopenTransitions;
            internal int NewIndependentViewCells;
            internal int CorrelatedViewMatches;
            internal float MaxBaselineMetres;
            internal float MaxViewSpreadDeg;
            internal int FreeSpaceChallenges;
            internal int FreeGapLt20;
            internal int FreeGap20To50;
            internal int FreeGap50To100;
            internal int FreeGapGe100;
            internal int IndependentFreeSpaceChallenges;
            internal float MaxFreeSpaceGapMetres;
            internal int FingerprintAssociationTests;
            internal int FingerprintAssociationHits;
            internal int FingerprintAssociationMisses;
        }

        private readonly Dictionary<CellKey, CellState> _cells =
            new Dictionary<CellKey, CellState>(8192);
        private readonly List<FingerprintRecord> _fingerprintHistory =
            new List<FingerprintRecord>(8192);
        private readonly List<PostVerdictRecord> _postVerdictHistory =
            new List<PostVerdictRecord>(256);
        private readonly Dictionary<CellKey, PostVerdictRecord> _activePostVerdicts =
            new Dictionary<CellKey, PostVerdictRecord>(256);
        private readonly GraduationContractShadowRace _graduationRace =
            new GraduationContractShadowRace();
        private readonly object _stateLock = new object();
        private readonly StringBuilder _pendingFrames = new StringBuilder(65536);
        private readonly StringBuilder _pendingEvents = new StringBuilder(65536);
        private string _directory = string.Empty;
        private string _framePath = string.Empty;
        private string _eventPath = string.Empty;
        private bool _active;
        private int _frameRows;
        private int _eventRows;
        private int _writeErrors;
        private int _lastEvaluatedFrame = -1;
        private int _heldCells;
        private int _acceptedCells;
        private int _rejectedCells;
        private int _challengesClearedBySupport;
        private int _nextFingerprintId;
        private int _correctionRevision;
        private string _guidanceCompact = "暂无Reject·范围0.15-5m";
        private string _guidanceHudFixed =
            "复核 待0000 证空0000 误拒0000 过期0000\n" +
            "目标 --.-m 约余---s 空-/2 面-/2 方[等待目标  ]\n" +
            "换位 ---cm 转角---° 判定[暂无目标    ]";
        private GuidanceTargetSnapshot _guidanceTargetSnapshot;
        private GuidanceFrameSnapshot _guidanceFrameSnapshot;
        private const int GuidanceVisualCapacity = 96;
        private readonly GuidanceCellVisual[] _guidanceVisuals =
            new GuidanceCellVisual[GuidanceVisualCapacity];
        private readonly HashSet<CellKey> _guidancePostCells =
            new HashSet<CellKey>();
        private int _guidanceVisualCount;
        private int _guidanceLockedFingerprintId = -1;
        private int _guidanceLockedFingerprintGeneration = -1;
        private int _guidanceLockOutsideSinceFrame = -1;

        internal int FrameRows => _frameRows;
        internal int EventRows => _eventRows;
        internal int FollowupRows => _postVerdictHistory.Count;
        internal int WriteErrors => _writeErrors;
        internal int GraduationRaceCandidateRows => _graduationRace.CandidateRows;
        internal int GraduationRaceSummaryRows => _graduationRace.SummaryRows;
        internal int GraduationRaceWitnessEventRows =>
            _graduationRace.WitnessEventRows;
        internal int GraduationRaceWriteErrors => _graduationRace.WriteErrors;

        internal string GetGuidanceCompact()
        {
            lock (_stateLock)
                return _guidanceCompact;
        }

        internal string GetGuidanceHudFixed()
        {
            lock (_stateLock)
                return _guidanceHudFixed;
        }

        internal GuidanceTargetSnapshot GetGuidanceTargetSnapshot()
        {
            lock (_stateLock)
                return _guidanceTargetSnapshot;
        }

        internal GuidanceFrameSnapshot GetGuidanceFrameSnapshot()
        {
            lock (_stateLock)
                return _guidanceFrameSnapshot;
        }

        internal int CopyGuidanceCellVisuals(GuidanceCellVisual[] destination)
        {
            if (destination == null || destination.Length == 0) return 0;
            lock (_stateLock)
            {
                int count = Mathf.Min(destination.Length, _guidanceVisualCount);
                Array.Copy(_guidanceVisuals, destination, count);
                return count;
            }
        }

        internal int CorrectionRevision
        {
            get
            {
                lock (_stateLock)
                    return _correctionRevision;
            }
        }

        internal int CopyPaperCorrectionCells(List<PaperCorrectionCell> destination)
        {
            if (destination == null) return 0;
            lock (_stateLock)
            {
                destination.Clear();
                foreach (KeyValuePair<CellKey, CellState> pair in _cells)
                {
                    CellState state = pair.Value;
                    if (!state.PaperCorrectionGateActive) continue;
                    CellKey key = pair.Key;
                    int mode = state.Verdict == Verdict.Reject ? 1 : 2;
                    destination.Add(new PaperCorrectionCell(
                        key.X, key.Y, key.Z, key.Axis, mode,
                        state.FingerprintCenter));
                }
                return destination.Count;
            }
        }

        internal void Begin(string probeDirectory)
        {
            End();
            if (string.IsNullOrEmpty(probeDirectory)) return;
            _directory = probeDirectory;
            Directory.CreateDirectory(_directory);
            _framePath = Path.Combine(_directory, "verdict_frames.csv");
            _eventPath = Path.Combine(_directory, "verdict_events.csv");
            File.WriteAllText(_framePath,
                "gunGelFrame,platformFrame,unityFrame,unscaledTime,valid,holdRawMissing,holdDualDisagree,holdNoStable,holdResidual,holdNormal,holdOpposed,holdMotion,holdOuter,eligibleSupport,independentSupport,observationAccept,observationHold,observationReject,acceptTransitions,rejectTransitions,reopenTransitions,newIndependentViewCells,correlatedViewMatches,maxBaselineMm,maxViewSpreadDeg,freeSpaceChallenges,freeGapLt20Mm,freeGap20To50Mm,freeGap50To100Mm,freeGapGe100Mm,independentFreeSpaceChallenges,maxFreeSpaceGapMm,fingerprintAssociationTests,fingerprintAssociationHits,fingerprintAssociationMisses,heldCells,acceptedCells,rejectedCells,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(_eventPath,
                "gunGelFrame,platformFrame,event,cellX,cellY,cellZ,axis,previousVerdict,newVerdict,reason,sourceX,sourceY,sourceZ,targetX,targetY,targetZ,residualMm,angleDeg,gapMm,viewRadius,motionQuality,baselineMm,spreadDeg,safeSupportFrames,independentSupportViews,challengeVotes,independentChallengeViews,recoverySupportViews,decisionStrength,fingerprintId,fingerprintGeneration,fingerprintAcceptedFrame,fingerprintPatchCount,fingerprintPatchIndex,intersectionX,intersectionY,intersectionZ,intersectionRangeMm,footprintRadiusMm,footprintOffsetMm,association\n",
                new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_directory, "verdict_schema.json"),
                BuildSchemaJson(), new UTF8Encoding(false));
            _cells.Clear();
            _fingerprintHistory.Clear();
            _postVerdictHistory.Clear();
            _activePostVerdicts.Clear();
            _pendingFrames.Clear();
            _pendingEvents.Clear();
            _frameRows = 0;
            _eventRows = 0;
            _writeErrors = 0;
            _heldCells = 0;
            _acceptedCells = 0;
            _rejectedCells = 0;
            _challengesClearedBySupport = 0;
            _nextFingerprintId = 0;
            _correctionRevision++;
            _lastEvaluatedFrame = -1;
            _guidanceCompact = "暂无Reject·范围0.15-5m";
            _guidanceHudFixed =
                "复核 待0000 证空0000 误拒0000 过期0000\n" +
                "目标 --.-m 约余---s 空-/2 面-/2 方[等待目标  ]\n" +
                "换位 ---cm 转角---° 判定[暂无目标    ]";
            _guidanceTargetSnapshot = default;
            _guidanceFrameSnapshot = default;
            _guidanceVisualCount = 0;
            _guidanceLockedFingerprintId = -1;
            _guidanceLockedFingerprintGeneration = -1;
            _guidanceLockOutsideSinceFrame = -1;
            _graduationRace.Begin(_directory);
            _active = true;
        }

        internal FrameResult Evaluate(
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
            float motionQuality)
        {
            if (!_active || correspondences.Length == 0)
                return default;
            lock (_stateLock)
            {
                try
                {
                    Vector3 camera = sourceViewInverse.MultiplyPoint3x4(Vector3.zero);
                    _lastEvaluatedFrame = Mathf.Max(_lastEvaluatedFrame, gunGelFrame);
                    var counters = new FrameCounters();
                    var challengedThisFrame = new HashSet<CellKey>();
                    for (int i = 0; i < correspondences.Length; i++)
                    {
                        GunGelEvidenceShadow.Correspondence item = correspondences[i];
                        uint flags = (uint)Mathf.Max(0, Mathf.RoundToInt(item.SourceValid.w));
                        if ((flags & ObservationValid) == 0u) continue;
                        counters.Valid++;

                        bool hasRaw = (flags & RawAvailable) != 0u;
                        bool agrees = (flags & DualAgree) != 0u;
                        bool found = (flags & StableFound) != 0u ||
                                     (flags & StableMatch) != 0u;
                        bool residualOk = (flags & ResidualPass) != 0u;
                        bool normalOk = (flags & NormalPass) != 0u;
                        bool matched = (flags & StableMatch) != 0u;
                        bool noOpposition = (flags & Unopposed) != 0u;
                        Vector3 source = new Vector3(item.SourceValid.x,
                            item.SourceValid.y, item.SourceValid.z);
                        float viewRadius = ViewRadius(i, gridX, pixelStride,
                            depthWidth, depthHeight);

                        if (!hasRaw) counters.HoldRawMissing++;
                        if (hasRaw && !agrees) counters.HoldDualDisagree++;
                        if (agrees && !found) counters.HoldNoStable++;
                        if (found && !residualOk) counters.HoldResidual++;
                        if (found && !normalOk) counters.HoldNormal++;
                        if (found && !noOpposition) counters.HoldOpposed++;
                        if (motionQuality < SafeMotionQuality) counters.HoldMotion++;
                        if (viewRadius >= AuthorityOuterRadius) counters.HoldOuter++;

                        CellState state = null;
                        bool safeSupport = false;
                        Vector3 target = Vector3.zero;
                        Vector3 normal = new Vector3(item.NormalAngle.x,
                            item.NormalAngle.y, item.NormalAngle.z);
                        float normalLength = normal.magnitude;
                        float residualMetres = 0f;
                        float angleDeg = Mathf.Max(0f, item.NormalAngle.w);
                        float footprintRadius = 0f;
                        if (found && normalLength > 1e-5f)
                        {
                            normal /= normalLength;
                            target = new Vector3(item.TargetSigma.x,
                                item.TargetSigma.y, item.TargetSigma.z);
                            residualMetres = Mathf.Abs(Vector3.Dot(source - target, normal));
                            footprintRadius = EstimateFootprintRadius(source, normal, i,
                                gridX, pixelStride, depthWidth, depthHeight,
                                sourceProjectionInverse, sourceViewInverse);
                            CellKey key = CellKey.From(target, normal);
                            state = GetOrCreate(key, target, normal, camera, gunGelFrame);
                            TrackSeenView(state, camera, gunGelFrame, ref counters);

                            safeSupport = hasRaw && agrees && matched && residualOk &&
                                          normalOk && noOpposition &&
                                          motionQuality >= SafeMotionQuality &&
                                          viewRadius < AuthorityOuterRadius;
                            if (state.Verdict == Verdict.Reject)
                            {
                                ObservePostVerdictContact(key, state, camera,
                                    gunGelFrame, safeSupport);
                            }
                            if (safeSupport)
                            {
                                counters.EligibleSupport++;
                                bool independent = ApplySafeSupport(key, state, camera,
                                    source, target, residualMetres, angleDeg, viewRadius,
                                    footprintRadius, motionQuality, gunGelFrame,
                                    platformFrame, ref counters);
                                if (independent) counters.IndependentSupport++;
                            }
                        }

                        bool observationAccepted = state != null && safeSupport &&
                            state.Verdict == Verdict.Accept;
                        if (observationAccepted) counters.ObservationAccept++;
                        else if (state != null && !safeSupport &&
                                 state.Verdict == Verdict.Reject)
                            counters.ObservationReject++;
                        else counters.ObservationHold++;

                        if (hasRaw && agrees && motionQuality >= SafeMotionQuality &&
                            viewRadius < AuthorityOuterRadius &&
                            (i % FreeSpaceRayStride) == 0)
                        {
                            ChallengeFreeSpace(camera, source, gunGelFrame, platformFrame,
                                viewRadius, motionQuality, challengedThisFrame,
                                ref counters);
                        }
                    }

                    UpdateGuidance(camera, sourceViewInverse.rotation, gunGelFrame);
                    AppendFrame(gunGelFrame, platformFrame, counters);
                    return new FrameResult(counters);
                }
                catch
                {
                    _writeErrors++;
                    return default;
                }
            }
        }

        internal void End()
        {
            lock (_stateLock)
            {
                if (_active)
                {
                    WriteCellSnapshot();
                    WriteFingerprintSnapshot();
                    WritePostVerdictSnapshot();
                    WriteGraduationRaceOutcomes();
                    _graduationRace.End(_lastEvaluatedFrame);
                }
                FlushLocked();
                _active = false;
            }
        }

        public void Dispose()
        {
            End();
            _cells.Clear();
            _fingerprintHistory.Clear();
            _postVerdictHistory.Clear();
            _activePostVerdicts.Clear();
            _graduationRace.Dispose();
        }

        private void WriteGraduationRaceOutcomes()
        {
            for (int i = 0; i < _postVerdictHistory.Count; i++)
            {
                PostVerdictRecord record = _postVerdictHistory[i];
                int windowEndFrame = record.RejectFrame + PostVerdictWindowFrames;
                bool windowComplete = _lastEvaluatedFrame >= windowEndFrame;
                _graduationRace.ObserveOutcome(record.Key.X, record.Key.Y,
                    record.Key.Z, record.Key.Axis, record.FingerprintGeneration,
                    record.RejectFrame, PostVerdictOutcome(record, windowComplete),
                    windowComplete);
            }
        }

        private void StartPostVerdict(CellKey key, CellState state,
            int frame, int platformFrame)
        {
            // 一个活动记录只对应被拒的那一代真实指纹。若同一单元后来重建，
            // 新世代再次被拒时会产生新记录，绝不覆盖上一代的结局。
            var record = new PostVerdictRecord
            {
                Key = key,
                FingerprintId = state.FingerprintId,
                FingerprintGeneration = state.FingerprintGeneration,
                FingerprintCenter = state.FingerprintCenter,
                RejectFrame = frame,
                RejectPlatformFrame = platformFrame,
                FreeSpaceWitnesses = new ViewWitness[FollowupWitnessCapacity],
                SupportWitnesses = new ViewWitness[FollowupWitnessCapacity]
            };
            _postVerdictHistory.Add(record);
            _activePostVerdicts[key] = record;
        }

        private void UpdateGuidance(Vector3 camera, Quaternion cameraRotation,
            int frame)
        {
            int pending = 0;
            int confirmedFalse = 0;
            int reopened = 0;
            int expired = 0;
            PostVerdictRecord target = null;
            float targetScore = float.PositiveInfinity;
            bool targetObservedRecently = false;
            PostVerdictRecord lockedTarget = null;
            bool lockedTargetObservedRecently = false;
            bool lockedTargetInsideReleaseFrame = false;
            int lockedTargetResolution = 0;
            int targetResolution = 0;
            int globalObserved = 0;
            Quaternion worldToCamera = Quaternion.Inverse(cameraRotation);
            BuildGuidanceVisuals(camera, worldToCamera, frame);

            for (int i = 0; i < _postVerdictHistory.Count; i++)
            {
                PostVerdictRecord record = _postVerdictHistory[i];
                bool matchesGuidanceLock =
                    record.FingerprintId == _guidanceLockedFingerprintId &&
                    record.FingerprintGeneration == _guidanceLockedFingerprintGeneration;
                bool recordInFrame = IsInsideGuidanceAngle(
                    record.FingerprintCenter, camera, worldToCamera,
                    GuidanceFrameHalfAngleDeg);
                int recordObservedAge = record.LastObservedFrame >= 0
                    ? frame - record.LastObservedFrame
                    : int.MaxValue;
                bool recordObservedRecently = recordObservedAge >= 0 &&
                    recordObservedAge <= 4;
                if (record.Reopened)
                {
                    reopened++;
                    if (matchesGuidanceLock)
                    {
                        lockedTarget = record;
                        lockedTargetResolution = 2;
                        lockedTargetObservedRecently = record.LastObservedFrame >= 0 &&
                            frame - record.LastObservedFrame <= 4;
                        lockedTargetInsideReleaseFrame = IsInsideGuidanceAngle(
                            record.FingerprintCenter, camera, worldToCamera,
                            GuidanceLockReleaseHalfAngleDeg);
                    }
                    continue;
                }
                if (record.FreeSpaceReconfirmed)
                {
                    confirmedFalse++;
                    if (matchesGuidanceLock)
                    {
                        lockedTarget = record;
                        lockedTargetResolution = 1;
                        lockedTargetObservedRecently = record.LastObservedFrame >= 0 &&
                            frame - record.LastObservedFrame <= 4;
                        lockedTargetInsideReleaseFrame = IsInsideGuidanceAngle(
                            record.FingerprintCenter, camera, worldToCamera,
                            GuidanceLockReleaseHalfAngleDeg);
                    }
                    continue;
                }
                pending++;
                if (recordObservedRecently) globalObserved++;
                Vector3 local = worldToCamera *
                                (record.FingerprintCenter - camera);
                float distance = local.magnitude;
                bool forward = local.z > 0.001f;
                float yaw = forward
                    ? Mathf.Abs(Mathf.Atan2(local.x, local.z) * Mathf.Rad2Deg)
                    : 180f;
                float pitch = forward
                    ? Mathf.Abs(Mathf.Atan2(local.y,
                        Mathf.Sqrt(local.x * local.x + local.z * local.z)) *
                        Mathf.Rad2Deg)
                    : 90f;
                bool usableRange = distance >= 0.15f && distance <= 5f;
                // “正在看”必须由数据射线本身命中该 Reject 单元证明，不能用
                // 小格中心是否压进取景框代替。允许 4 个 GunGel 帧的 HUD
                // 刷新滞后，避免 4Hz HUD 与约 16Hz 证据流错拍。
                bool observedRecently = recordObservedRecently;
                bool inAcquisitionFrame = recordInFrame;
                if (matchesGuidanceLock)
                {
                    lockedTarget = record;
                    lockedTargetObservedRecently = observedRecently;
                    lockedTargetInsideReleaseFrame = forward &&
                        yaw <= GuidanceLockReleaseHalfAngleDeg &&
                        pitch <= GuidanceLockReleaseHalfAngleDeg;
                }

                // An unlocked target may only be acquired inside the visible
                // frame. Never invent an off-screen target that drags the
                // operator around the room. Recent evidence only breaks ties.
                if (!inAcquisitionFrame) continue;
                float score = (observedRecently ? -2000f + recordObservedAge : 0f) +
                              (usableRange ? 0f : 1000f) +
                              Mathf.Max(yaw, pitch) +
                              distance * 0.01f;
                if (score < targetScore)
                {
                    targetScore = score;
                    target = record;
                    targetObservedRecently = observedRecently;
                }
            }

            if (lockedTarget != null)
            {
                if (lockedTargetResolution != 0)
                {
                    if (lockedTargetInsideReleaseFrame)
                    {
                        target = lockedTarget;
                        targetObservedRecently = lockedTargetObservedRecently;
                        targetResolution = lockedTargetResolution;
                        _guidanceLockOutsideSinceFrame = -1;
                    }
                    else
                    {
                        if (_guidanceLockOutsideSinceFrame < 0)
                            _guidanceLockOutsideSinceFrame = frame;
                        if (frame - _guidanceLockOutsideSinceFrame <
                            GuidanceLockReleaseFrames)
                        {
                            target = lockedTarget;
                            targetObservedRecently = lockedTargetObservedRecently;
                            targetResolution = lockedTargetResolution;
                        }
                        else
                        {
                            _guidanceLockedFingerprintId = -1;
                            _guidanceLockedFingerprintGeneration = -1;
                            _guidanceLockOutsideSinceFrame = -1;
                        }
                    }
                }
                else if (lockedTargetInsideReleaseFrame)
                {
                    _guidanceLockOutsideSinceFrame = -1;
                    target = lockedTarget;
                    targetObservedRecently = lockedTargetObservedRecently;
                }
                else
                {
                    if (_guidanceLockOutsideSinceFrame < 0)
                        _guidanceLockOutsideSinceFrame = frame;
                    if (frame - _guidanceLockOutsideSinceFrame < GuidanceLockReleaseFrames)
                    {
                        target = lockedTarget;
                        targetObservedRecently = lockedTargetObservedRecently;
                    }
                    else
                    {
                        _guidanceLockedFingerprintId = -1;
                        _guidanceLockedFingerprintGeneration = -1;
                        _guidanceLockOutsideSinceFrame = -1;
                    }
                }
            }
            else
            {
                _guidanceLockedFingerprintId = -1;
                _guidanceLockedFingerprintGeneration = -1;
                _guidanceLockOutsideSinceFrame = -1;
            }

            if (_guidanceLockedFingerprintId < 0 && target != null &&
                targetResolution == 0)
            {
                _guidanceLockedFingerprintId = target.FingerprintId;
                _guidanceLockedFingerprintGeneration = target.FingerprintGeneration;
                _guidanceLockOutsideSinceFrame = -1;
            }

            if (target == null)
            {
                _guidanceTargetSnapshot = default;
                SetFrameGuidance(pending, confirmedFalse, reopened, expired,
                    globalObserved);
                return;
            }

            Vector3 toTarget = target.FingerprintCenter - camera;
            Vector3 targetLocal = worldToCamera * toTarget;
            float targetDistance = toTarget.magnitude;
            bool targetForward = targetLocal.z > 0.001f;
            float yawDeg = targetForward
                ? Mathf.Atan2(targetLocal.x, targetLocal.z) * Mathf.Rad2Deg
                : 180f;
            float pitchDeg = targetForward
                ? Mathf.Atan2(targetLocal.y,
                    Mathf.Sqrt(targetLocal.x * targetLocal.x +
                               targetLocal.z * targetLocal.z)) * Mathf.Rad2Deg
                : 0f;
            bool targetInAcquisitionFrame = targetForward &&
                Mathf.Abs(yawDeg) <= GuidanceFrameHalfAngleDeg &&
                Mathf.Abs(pitchDeg) <= GuidanceFrameHalfAngleDeg;
            int remainingFrames = Mathf.Max(0,
                target.RejectFrame + PostVerdictWindowFrames - frame);
            float remainingSeconds = remainingFrames / 16f;

            string aim;
            if (targetResolution != 0)
                aim = "结案保留";
            else if (targetInAcquisitionFrame && targetObservedRecently)
                aim = "视野命中";
            else if (!targetForward)
                aim = "身后·转向";
            else if (Mathf.Abs(yawDeg) >= GuidanceFrameHalfAngleDeg)
                aim = $"{(yawDeg > 0f ? "右" : "左")}{Mathf.Abs(yawDeg):0}°";
            else if (Mathf.Abs(pitchDeg) >= GuidanceFrameHalfAngleDeg)
                aim = $"{(pitchDeg > 0f ? "上" : "下")}{Mathf.Abs(pitchDeg):0}°";
            else
                aim = "已对准";

            string range = targetDistance < 0.15f
                ? $"{targetDistance:0.00}m太近"
                : targetDistance > 5f
                    ? $"{targetDistance:0.0}m超5m"
                    : $"{targetDistance:0.0}m";

            ViewWitness[] witnesses;
            int witnessCount;
            string witnessLabel;
            if (target.FreeSpaceWitnessCount > 0)
            {
                witnesses = target.FreeSpaceWitnesses;
                witnessCount = target.FreeSpaceWitnessCount;
                witnessLabel = $"空证{witnessCount}/{RequiredFollowupFreeViews}";
            }
            else if (target.SupportWitnessCount > 0)
            {
                witnesses = target.SupportWitnesses;
                witnessCount = target.SupportWitnessCount;
                witnessLabel = $"面证{witnessCount}/{RequiredRecoveryViews}";
            }
            else
            {
                witnesses = null;
                witnessCount = 0;
                witnessLabel = "框内等首证";
            }

            string poseGuide = string.Empty;
            float guideBaseline = 0f;
            float guideSpread = 0f;
            bool guidePoseReady = false;
            if (witnessCount > 0 && witnesses != null)
            {
                Vector3 direction = SafeDirection(camera - target.FingerprintCenter);
                bool poseReady = true;
                float closestScore = float.PositiveInfinity;
                float nearestBaseline = 0f;
                float nearestSpread = 0f;
                int available = Mathf.Min(witnessCount, witnesses.Length);
                for (int i = 0; i < available; i++)
                {
                    float baseline = Vector3.Distance(witnesses[i].Camera, camera);
                    float spread = Vector3.Angle(witnesses[i].Direction, direction);
                    float score = Mathf.Max(
                        baseline / IndependentBaselineMetres,
                        spread / IndependentAngleDeg);
                    if (score < closestScore)
                    {
                        closestScore = score;
                        nearestBaseline = baseline;
                        nearestSpread = spread;
                    }
                    if (baseline < IndependentBaselineMetres &&
                        spread < IndependentAngleDeg)
                        poseReady = false;
                }
                poseGuide = poseReady
                    ? " 位置够·对准取证"
                    : $" 原地无效({nearestBaseline * 100f:0}cm/" +
                      $"{nearestSpread:0.0}°)";
                guideBaseline = nearestBaseline;
                guideSpread = nearestSpread;
                guidePoseReady = poseReady;
            }

            _guidanceCompact =
                $"待{pending} 证空{confirmedFalse} 误拒{reopened} 过期{expired}｜" +
                $"目标{range} 约余{remainingSeconds:0}s {aim}｜" +
                witnessLabel + poseGuide;

            string fixedAim;
            if (targetResolution != 0)
                fixedAim = "结案保留";
            else if (targetInAcquisitionFrame && targetObservedRecently)
                fixedAim = "视野命中";
            else if (!targetForward)
                fixedAim = "身后转向";
            else if (Mathf.Abs(yawDeg) >= GuidanceFrameHalfAngleDeg)
                fixedAim = $"{(yawDeg > 0f ? "右" : "左")}" +
                           $"{Mathf.Clamp(Mathf.RoundToInt(Mathf.Abs(yawDeg)), 0, 999):000}°";
            else if (Mathf.Abs(pitchDeg) >= GuidanceFrameHalfAngleDeg)
                fixedAim = $"{(pitchDeg > 0f ? "上" : "下")}" +
                           $"{Mathf.Clamp(Mathf.RoundToInt(Mathf.Abs(pitchDeg)), 0, 999):000}°";
            else
                fixedAim = "已经对准";

            string decision;
            if (targetResolution == 1)
                decision = "空证确认Reject";
            else if (targetResolution == 2)
                decision = "面证确认误拒";
            else if (targetResolution == 3)
                decision = "复核窗口过期";
            else if (targetDistance < 0.15f)
                decision = "距离太近";
            else if (targetDistance > 5f)
                decision = "距离超限";
            else if (!targetInAcquisitionFrame)
                decision = "目标不在框内";
            else if (!targetObservedRecently)
                decision = "框中但未命中";
            else if (witnessCount <= 0)
                decision = "命中但证据不足";
            else
                decision = guidePoseReady ? "位置够可取证" : "原地无效需换位";

            _guidanceTargetSnapshot = new GuidanceTargetSnapshot(
                target.FingerprintCenter,
                targetInAcquisitionFrame,
                targetObservedRecently,
                guidePoseReady,
                targetResolution,
                targetDistance,
                target.FreeSpaceWitnessCount,
                target.SupportWitnessCount);

            _guidanceHudFixed =
                $"复核 待{HudCount(pending)} 证空{HudCount(confirmedFalse)} " +
                $"误拒{HudCount(reopened)} 过期{HudCount(expired)}\n" +
                $"目标 {Mathf.Clamp(targetDistance, 0f, 99.9f),4:0.0}m " +
                $"约余{Mathf.Clamp(Mathf.RoundToInt(remainingSeconds), 0, 999):000}s " +
                $"空{Mathf.Clamp(target.FreeSpaceWitnessCount, 0, 9)}/2 " +
                $"面{Mathf.Clamp(target.SupportWitnessCount, 0, 9)}/2 " +
                $"方[{HudSlot(fixedAim, 6)}]\n" +
                $"换位 {Mathf.Clamp(Mathf.RoundToInt(guideBaseline * 100f), 0, 999):000}cm " +
                $"转角{Mathf.Clamp(Mathf.RoundToInt(guideSpread), 0, 999):000}° " +
                $"判定[{HudSlot(decision, 8)}]";

            // Operator HUD reports the passive global queue. The single-target
            // text above stays available to diagnostics, but never instructs
            // the operator or gates evidence acquisition.
            SetFrameGuidance(pending, confirmedFalse, reopened, expired,
                globalObserved);
        }

        private void SetFrameGuidance(
            int pending,
            int confirmedFalse,
            int reopened,
            int expired,
            int globalObserved)
        {
            // The operator no longer owns an acquisition ROI.  These are
            // global automatic-queue counts; the fixed frame below is only a
            // read-only window into spatial evidence currently in view.
            int registeredFingerprints = 0;
            int challengeOne = 0;
            int challengeTwo = 0;
            foreach (KeyValuePair<CellKey, CellState> pair in _cells)
            {
                CellState cell = pair.Value;
                if (cell.FingerprintId <= 0) continue;
                registeredFingerprints++;
                if (cell.Verdict != Verdict.Accept) continue;
                if (cell.ConsecutiveChallengeVotes == 1) challengeOne++;
                else if (cell.ConsecutiveChallengeVotes >= 2) challengeTwo++;
            }
            _guidanceFrameSnapshot = new GuidanceFrameSnapshot(
                registeredFingerprints, challengeOne, challengeTwo, pending,
                globalObserved, confirmedFalse, reopened, expired,
                _challengesClearedBySupport);
            int resolved = confirmedFalse + reopened;
            string state = globalObserved > 0
                ? "自动取证中"
                : pending > 0
                    ? "队列等待重见"
                    : resolved > 0
                        ? "自动样本已结案"
                        : "自动巡检中";
            _guidanceCompact =
                $"空间ID{registeredFingerprints} 反1{challengeOne} 反2{challengeTwo} " +
                $"拒{pending} 撤{_challengesClearedBySupport}｜" +
                $"空证{confirmedFalse} 面证{reopened}｜" +
                $"自动命中{globalObserved} 已收{resolved}｜{state}";
            _guidanceHudFixed =
                $"空间 ID{HudCount(registeredFingerprints)} 反1{HudCount(challengeOne)} " +
                $"反2{HudCount(challengeTwo)} 拒{HudCount(pending)}\n" +
                $"结案 空{HudCount(confirmedFalse)} 面{HudCount(reopened)} " +
                $"撤票{HudCount(_challengesClearedBySupport)}\n" +
                $"队列 命中{HudCount(globalObserved)} 已收{HudCount(resolved)} " +
                $"状态[{HudSlot(state, 8)}]";
        }

        private static string HudCount(int value)
        {
            return Mathf.Clamp(value, 0, 9999).ToString("0000",
                CultureInfo.InvariantCulture);
        }

        private static bool IsInsideGuidanceAngle(
            Vector3 worldPosition,
            Vector3 camera,
            Quaternion worldToCamera,
            float halfAngleDeg)
        {
            Vector3 local = worldToCamera * (worldPosition - camera);
            if (local.z <= 0.001f) return false;
            float yaw = Mathf.Abs(Mathf.Atan2(local.x, local.z) * Mathf.Rad2Deg);
            float pitch = Mathf.Abs(Mathf.Atan2(local.y,
                Mathf.Sqrt(local.x * local.x + local.z * local.z)) * Mathf.Rad2Deg);
            return yaw <= halfAngleDeg && pitch <= halfAngleDeg;
        }

        private static bool ShouldShowGuidanceBackground(CellKey key)
        {
            // Include the normal-axis suffix so each oriented world cell has
            // one immutable sampling decision. The integer mix is fully
            // deterministic for the lifetime of that cell.
            unchecked
            {
                int hash = key.X * 73856093 ^ key.Y * 19349663 ^
                           key.Z * 83492791 ^ key.Axis * 265443576;
                return (hash & GuidanceBackgroundHashMask) == 0;
            }
        }

        private static string HudSlot(string value, int width)
        {
            value ??= string.Empty;
            if (value.Length > width) return value.Substring(0, width);
            return value.PadRight(width, ' ');
        }

        private void BuildGuidanceVisuals(Vector3 camera,
            Quaternion worldToCamera, int frame)
        {
            _guidanceVisualCount = 0;
            _guidancePostCells.Clear();

            // Highest-value evidence goes in first, so a dense stable wall can
            // never evict the few red/purple/cyan cells from the fixed pool.
            for (int i = _postVerdictHistory.Count - 1;
                 i >= 0 && _guidanceVisualCount < GuidanceVisualCapacity; i--)
            {
                PostVerdictRecord record = _postVerdictHistory[i];
                if (!IsInsideGuidanceAngle(record.FingerprintCenter, camera,
                        worldToCamera, GuidanceFrameHalfAngleDeg))
                    continue;

                int phase;
                int resolutionFrame;
                if (record.Reopened)
                {
                    phase = 6;
                    resolutionFrame = record.ReopenedFrame;
                }
                else if (record.FreeSpaceReconfirmed)
                {
                    phase = 5;
                    resolutionFrame = record.FreeSpaceReconfirmedFrame;
                }
                else
                {
                    // Unresolved Rejects are session-long spatial tasks. Time
                    // passing never makes their fixed-world marker vanish.
                    phase = 4;
                    resolutionFrame = frame;
                }

                // Resolved cells leave a short, fixed-world trail instead of
                // blinking out on the exact verdict frame.
                if ((phase == 5 || phase == 6) &&
                    frame - resolutionFrame > 120)
                    continue;
                float strength = phase == 4
                    ? Mathf.Max(record.FreeSpaceWitnessCount,
                        record.SupportWitnessCount) /
                      (float)RequiredFollowupFreeViews
                    : 1f;
                _guidancePostCells.Add(record.Key);
                Vector3 displayPosition = record.FingerprintCenter;
                if (_cells.TryGetValue(record.Key, out CellState displayState))
                    displayPosition = displayState.DisplayAnchor;
                AddGuidanceVisual(displayPosition, phase, strength);
            }

            // Show the approach to Reject explicitly: green -> yellow ->
            // orange. These are registered fingerprint cells, not raw pixels.
            foreach (KeyValuePair<CellKey, CellState> pair in _cells)
            {
                if (_guidanceVisualCount >= GuidanceVisualCapacity) break;
                CellState state = pair.Value;
                if (state.FingerprintId <= 0 || state.Verdict != Verdict.Accept ||
                    state.ConsecutiveChallengeVotes <= 0 ||
                    !IsInsideGuidanceAngle(state.FingerprintCenter, camera,
                        worldToCamera, GuidanceFrameHalfAngleDeg))
                    continue;
                int phase = state.ConsecutiveChallengeVotes >= 2 ? 3 : 2;
                AddGuidanceVisual(state.DisplayAnchor, phase,
                    state.ConsecutiveChallengeVotes /
                    (float)RequiredChallengeVotes);
            }

            // A cleared yellow/orange vote used to disappear before the 4 Hz
            // HUD could show it. Preserve a dim fixed-world trace briefly so
            // the operator can see that the challenge existed and was reset.
            foreach (KeyValuePair<CellKey, CellState> pair in _cells)
            {
                if (_guidanceVisualCount >= GuidanceVisualCapacity) break;
                CellState state = pair.Value;
                if (state.ConsecutiveChallengeVotes > 0 ||
                    state.ChallengeClearedFrame < 0 ||
                    frame - state.ChallengeClearedFrame >
                    GuidanceClearedTraceFrames ||
                    !IsInsideGuidanceAngle(state.DisplayAnchor, camera,
                        worldToCamera, GuidanceFrameHalfAngleDeg))
                    continue;
                int phase = state.LastChallengePeakVotes >= 2 ? 8 : 7;
                AddGuidanceVisual(state.DisplayAnchor, phase, 1f);
            }

            // Provisional and stable background references share the same
            // immutable display anchor. Their world-cell hash decides once
            // whether they are visible; camera motion never chooses a new
            // representative. A state transition therefore changes only the
            // colour of an existing cross.
            foreach (KeyValuePair<CellKey, CellState> pair in _cells)
            {
                if (_guidanceVisualCount >= GuidanceVisualCapacity) break;
                CellState state = pair.Value;
                if (!ShouldShowGuidanceBackground(pair.Key) ||
                    state.Verdict == Verdict.Reject ||
                    state.ConsecutiveChallengeVotes > 0 ||
                    _activePostVerdicts.ContainsKey(pair.Key) ||
                    _guidancePostCells.Contains(pair.Key) ||
                    (state.ChallengeClearedFrame >= 0 &&
                     frame - state.ChallengeClearedFrame <=
                     GuidanceClearedTraceFrames) ||
                    !IsInsideGuidanceAngle(state.DisplayAnchor, camera,
                        worldToCamera, GuidanceFrameHalfAngleDeg))
                    continue;
                int phase = state.FingerprintId > 0 &&
                            state.Verdict == Verdict.Accept ? 1 : 0;
                float strength = phase == 1
                    ? state.IndependentSupportViews /
                      (float)RequiredSupportViews
                    : state.SeenWitnessCount /
                      (float)RequiredSupportViews;
                AddGuidanceVisual(state.DisplayAnchor, phase, strength);
            }
        }

        private void AddGuidanceVisual(Vector3 worldPosition, int phase,
            float strength)
        {
            if (_guidanceVisualCount >= GuidanceVisualCapacity) return;
            _guidanceVisuals[_guidanceVisualCount++] =
                new GuidanceCellVisual(worldPosition, phase, strength);
        }

        private void ObservePostVerdictContact(CellKey key, CellState state,
            Vector3 camera, int frame, bool safeSupport)
        {
            if (!_activePostVerdicts.TryGetValue(key, out PostVerdictRecord record) ||
                frame <= record.RejectFrame)
                return;
            TouchPostVerdict(record, frame);
            if (record.LastContactFrame != frame)
            {
                record.LastContactFrame = frame;
                record.ContactFrames++;
            }
            if (!safeSupport || record.LastSafeSupportFrame == frame) return;
            record.LastSafeSupportFrame = frame;
            record.SafeSupportFrames++;
            Vector3 direction = SafeDirection(camera - state.FingerprintCenter);
            TryAddIndependentWitness(ref record.SupportWitnesses,
                ref record.SupportWitnessCount, FollowupWitnessCapacity,
                camera, direction, frame, out float baseline, out float spread);
            record.MaxBaselineMetres = Mathf.Max(record.MaxBaselineMetres, baseline);
            record.MaxViewSpreadDeg = Mathf.Max(record.MaxViewSpreadDeg, spread);
        }

        private void ObservePostVerdictFreeSpace(CellKey key, CellState state,
            Vector3 camera, int frame, float gap)
        {
            if (!_activePostVerdicts.TryGetValue(key, out PostVerdictRecord record) ||
                frame <= record.RejectFrame)
                return;
            TouchPostVerdict(record, frame);
            if (record.LastRayIntersectionFrame != frame)
            {
                record.LastRayIntersectionFrame = frame;
                record.RayIntersectionFrames++;
            }
            record.MaxFreeGapMetres = Mathf.Max(record.MaxFreeGapMetres, gap);
            if (gap < ReliableFreeGapMetres ||
                record.LastReliableFreeSpaceFrame == frame)
                return;
            record.LastReliableFreeSpaceFrame = frame;
            record.ReliableFreeSpaceFrames++;
            Vector3 direction = SafeDirection(camera - state.FingerprintCenter);
            TryAddIndependentWitness(ref record.FreeSpaceWitnesses,
                ref record.FreeSpaceWitnessCount, FollowupWitnessCapacity,
                camera, direction, frame, out float baseline, out float spread);
            record.MaxBaselineMetres = Mathf.Max(record.MaxBaselineMetres, baseline);
            record.MaxViewSpreadDeg = Mathf.Max(record.MaxViewSpreadDeg, spread);
            if (!record.FreeSpaceReconfirmed &&
                record.FreeSpaceWitnessCount >= RequiredFollowupFreeViews)
            {
                record.FreeSpaceReconfirmed = true;
                record.FreeSpaceReconfirmedFrame = frame;
            }
        }

        private static void TouchPostVerdict(PostVerdictRecord record, int frame)
        {
            if (record.FirstFollowupFrame < 0) record.FirstFollowupFrame = frame;
            record.LastFollowupFrame = Mathf.Max(record.LastFollowupFrame, frame);
            if (record.LastObservedFrame == frame) return;
            record.LastObservedFrame = frame;
            record.ObservedFrames++;
        }

        private void MarkPostVerdictReopened(CellKey key, CellState state, int frame)
        {
            if (!_activePostVerdicts.TryGetValue(key, out PostVerdictRecord record))
                return;
            record.Reopened = true;
            record.ReopenedFrame = frame;
            record.ReopenedFingerprintId = state.FingerprintId;
            record.ReopenedGeneration = state.FingerprintGeneration;
            _activePostVerdicts.Remove(key);
        }

        private CellState GetOrCreate(CellKey key, Vector3 target, Vector3 normal,
            Vector3 camera, int frame)
        {
            if (_cells.TryGetValue(key, out CellState state))
            {
                state.Center = target;
                state.Normal = normal;
                state.LastSeenFrame = frame;
                return state;
            }
            Vector3 direction = SafeDirection(camera - target);
            state = new CellState
            {
                Center = target,
                // The visual anchor belongs to the world cell, not to the
                // latest noisy surface estimate.  It never moves afterwards.
                DisplayAnchor = new Vector3(
                    (key.X + 0.5f) * CellMetres,
                    (key.Y + 0.5f) * CellMetres,
                    (key.Z + 0.5f) * CellMetres),
                Normal = normal,
                FirstFrame = frame,
                LastSeenFrame = frame,
                FirstSeenCamera = camera,
                FirstSeenDirection = direction
            };
            SeedWitness(ref state.SeenWitnesses, ref state.SeenWitnessCount,
                SeenWitnessCapacity, camera, direction, frame);
            _cells.Add(key, state);
            _heldCells++;
            return state;
        }

        private static void TrackSeenView(CellState state, Vector3 camera, int frame,
            ref FrameCounters counters)
        {
            Vector3 direction = SafeDirection(camera - state.Center);
            float baseline = Vector3.Distance(state.FirstSeenCamera, camera);
            float spread = Vector3.Angle(state.FirstSeenDirection, direction);
            state.MaxBaselineMetres = Mathf.Max(state.MaxBaselineMetres, baseline);
            state.MaxViewSpreadDeg = Mathf.Max(state.MaxViewSpreadDeg, spread);
            counters.MaxBaselineMetres = Mathf.Max(counters.MaxBaselineMetres, baseline);
            counters.MaxViewSpreadDeg = Mathf.Max(counters.MaxViewSpreadDeg, spread);
            if (TryAddIndependentWitness(ref state.SeenWitnesses,
                    ref state.SeenWitnessCount, SeenWitnessCapacity,
                    camera, direction, frame, out _, out _))
            {
                counters.NewIndependentViewCells++;
            }
            else
            {
                counters.CorrelatedViewMatches++;
            }
        }

        private bool ApplySafeSupport(CellKey key, CellState state, Vector3 camera,
            Vector3 source, Vector3 target, float residualMetres, float angleDeg,
            float viewRadius, float footprintRadius, float motionQuality, int frame,
            int platformFrame, ref FrameCounters counters)
        {
            if (state.LastSafeSupportFrame != frame)
            {
                state.SafeSupportFrames++;
                state.LastSafeSupportFrame = frame;
            }
            Vector3 direction = SafeDirection(camera - state.Center);
            bool supportIndependent;
            IndependentWitnessDecision supportDecision;
            int supportWitnessCountBefore = state.SupportWitnessCount;
            float supportBaseline;
            float supportSpread;
            if (!state.HasSafeSupport)
            {
                state.HasSafeSupport = true;
                SeedWitness(ref state.SupportWitnesses,
                    ref state.SupportWitnessCount, SupportWitnessCapacity,
                    camera, direction, frame);
                state.IndependentSupportViews = state.SupportWitnessCount;
                supportIndependent = true;
                supportDecision = IndependentWitnessDecision.Accepted;
                supportBaseline = 0f;
                supportSpread = 0f;
            }
            else
            {
                supportIndependent = TryAddIndependentWitness(
                    ref state.SupportWitnesses, ref state.SupportWitnessCount,
                    SupportWitnessCapacity, camera, direction, frame,
                    out supportBaseline, out supportSpread,
                    out supportDecision);
                state.IndependentSupportViews = state.SupportWitnessCount;
            }

            // 任一安全接触都中断连续自由空间反证；“连续三票”不能跨过一张
            // 明确支持票继续累计。撤票必须先单独记账，即使这张支持票与
            // 旧视角相关、随后不能晋升/翻案，也不允许再无声消失。
            int clearedChallengeVotes = state.ConsecutiveChallengeVotes;
            if (clearedChallengeVotes > 0)
            {
                _challengesClearedBySupport++;
                state.LastChallengePeakVotes = Mathf.Max(
                    state.LastChallengePeakVotes, clearedChallengeVotes);
                state.ChallengeClearedFrame = frame;
                AppendEvent(frame, platformFrame,
                    "challenge_cleared_by_safe_support", key,
                    state.Verdict, state.Verdict,
                    $"safe_support_cleared_{clearedChallengeVotes}_votes",
                    source, target, residualMetres * 1000f, angleDeg, 0f,
                    viewRadius, motionQuality, 0f, 0f, state);
            }
            state.ConsecutiveChallengeVotes = 0;
            state.IndependentChallengeViews = 0;
            state.ChallengeWitnessCount = 0;

            Verdict before = state.Verdict;
            bool decisionWitness;
            float eventBaseline;
            float eventSpread;
            string eventName;
            string reason;
            if (before == Verdict.Reject)
            {
                AddSurfacePatch(ref state.RecoveryPatches,
                    ref state.RecoveryPatchCount, RecoveryPatchCapacity,
                    source, state.Normal, footprintRadius, frame);
                int productionWitnessCountBefore = state.RecoveryWitnessCount;
                decisionWitness = TryAddIndependentWitness(
                    ref state.RecoveryWitnesses, ref state.RecoveryWitnessCount,
                    RecoveryWitnessCapacity, camera, direction, frame,
                    out eventBaseline, out eventSpread,
                    out IndependentWitnessDecision productionDecision);
                state.RecoverySupportViews = state.RecoveryWitnessCount;
                _graduationRace.ObserveSafeSupport(key.X, key.Y, key.Z, key.Axis,
                    state.FingerprintGeneration + 1, frame,
                    Time.unscaledTimeAsDouble, camera, target, residualMetres,
                    angleDeg, viewRadius, motionQuality, decisionWitness,
                    "recovery", productionDecision,
                    productionWitnessCountBefore, state.RecoveryWitnessCount,
                    RecoveryWitnessCapacity, eventBaseline, eventSpread);
                if (!decisionWitness) return false;
                eventName = "recovery_independent";
                reason = "independent_recovery_support";
                if (state.RecoverySupportViews >= RequiredRecoveryViews)
                {
                    Transition(state, Verdict.Accept);
                    MintFingerprint(key, state, target, state.Normal, frame, true);
                    MarkPostVerdictReopened(key, state, frame);
                    counters.ReopenTransitions++;
                    reason = "reopened_by_independent_safe_support";
                }
            }
            else
            {
                if (state.FingerprintId == 0)
                {
                    AddSurfacePatch(ref state.FingerprintPatches,
                        ref state.FingerprintPatchCount, FingerprintPatchCapacity,
                        source, state.Normal, footprintRadius, frame);
                }
                state.RecoveryWitnessCount = 0;
                state.RecoverySupportViews = 0;
                state.RecoveryPatchCount = 0;
                decisionWitness = supportIndependent;
                eventBaseline = supportBaseline;
                eventSpread = supportSpread;
                _graduationRace.ObserveSafeSupport(key.X, key.Y, key.Z, key.Axis,
                    Mathf.Max(1, state.FingerprintGeneration), frame,
                    Time.unscaledTimeAsDouble, camera, target, residualMetres,
                    angleDeg, viewRadius, motionQuality, decisionWitness,
                    "support", supportDecision,
                    supportWitnessCountBefore, state.SupportWitnessCount,
                    SupportWitnessCapacity, supportBaseline, supportSpread);
                if (!decisionWitness) return false;
                eventName = "support_independent";
                reason = "independent_safe_support";
                if (state.SafeSupportFrames >= RequiredSupportFrames &&
                    state.IndependentSupportViews >= RequiredSupportViews &&
                    before != Verdict.Accept)
                {
                    Transition(state, Verdict.Accept);
                    MintFingerprint(key, state, target, state.Normal, frame, false);
                    counters.AcceptTransitions++;
                    reason = "accepted_by_independent_safe_support";
                }
            }
            AppendEvent(frame, platformFrame, eventName, key, before,
                state.Verdict, reason, source, target, residualMetres * 1000f,
                angleDeg, 0f, viewRadius, motionQuality, eventBaseline * 1000f,
                eventSpread, state);
            return true;
        }

        private void MintFingerprint(CellKey key, CellState state, Vector3 center,
            Vector3 normal, int frame, bool fromRecovery)
        {
            if (fromRecovery)
            {
                state.FingerprintPatchCount = 0;
                int available = state.RecoveryPatches == null
                    ? 0
                    : Mathf.Min(state.RecoveryPatchCount, state.RecoveryPatches.Length);
                for (int i = 0; i < available; i++)
                {
                    SurfacePatch patch = state.RecoveryPatches[i];
                    AddSurfacePatch(ref state.FingerprintPatches,
                        ref state.FingerprintPatchCount, FingerprintPatchCapacity,
                        patch.Center, patch.Normal, patch.RadiusMetres, patch.Frame);
                }
                state.RecoveryPatchCount = 0;
            }
            state.FingerprintId = ++_nextFingerprintId;
            state.FingerprintGeneration++;
            state.FingerprintAcceptedFrame = frame;
            state.FingerprintCenter = center;
            state.FingerprintNormal = SafeDirection(normal);
            _graduationRace.BindFingerprint(key.X, key.Y, key.Z, key.Axis,
                state.FingerprintGeneration, state.FingerprintId,
                state.FingerprintAcceptedFrame);
            int patchCount = state.FingerprintPatches == null
                ? 0
                : Mathf.Min(state.FingerprintPatchCount,
                    state.FingerprintPatches.Length);
            for (int i = 0; i < patchCount; i++)
            {
                _fingerprintHistory.Add(new FingerprintRecord
                {
                    Key = key,
                    FingerprintId = state.FingerprintId,
                    Generation = state.FingerprintGeneration,
                    AcceptedFrame = state.FingerprintAcceptedFrame,
                    PatchIndex = i,
                    Patch = state.FingerprintPatches[i]
                });
            }
        }

        private static void AddSurfacePatch(ref SurfacePatch[] patches, ref int count,
            int capacity, Vector3 center, Vector3 normal, float radiusMetres, int frame)
        {
            if (capacity <= 0 || normal.sqrMagnitude <= 1e-10f) return;
            if (patches == null || patches.Length < capacity)
                patches = new SurfacePatch[capacity];
            float radius = Mathf.Max(0.001f, radiusMetres);
            int available = Mathf.Min(count, patches.Length);
            for (int i = 0; i < available; i++)
            {
                SurfacePatch existing = patches[i];
                float mergeDistance = Mathf.Max(existing.RadiusMetres, radius) * 0.35f;
                if (Vector3.Distance(existing.Center, center) > mergeDistance ||
                    Vector3.Angle(existing.Normal, normal) > 3f)
                    continue;
                if (radius > existing.RadiusMetres)
                {
                    existing.RadiusMetres = radius;
                    patches[i] = existing;
                }
                return;
            }
            if (count >= capacity) return;
            patches[count] = new SurfacePatch
            {
                Center = center,
                Normal = SafeDirection(normal),
                RadiusMetres = radius,
                Frame = frame
            };
            count++;
        }

        private static bool TryIntersectFingerprint(Vector3 camera,
            Vector3 rayDirection, float hitRange, CellState state,
            out Vector3 intersection, out float intersectionRange,
            out int patchIndex, out float patchRadius, out float patchOffset)
        {
            intersection = Vector3.zero;
            intersectionRange = 0f;
            patchIndex = -1;
            patchRadius = 0f;
            patchOffset = 0f;
            if (state.FingerprintPatches == null || state.FingerprintPatchCount <= 0)
                return false;

            // 若一条射线命中同一指纹的多个小面片，取离新深度最近的交点。
            // 这给出最小自由空间间隔，是对删除最保守的证词。
            int available = Mathf.Min(state.FingerprintPatchCount,
                state.FingerprintPatches.Length);
            for (int i = 0; i < available; i++)
            {
                SurfacePatch patch = state.FingerprintPatches[i];
                float denominator = Vector3.Dot(rayDirection, patch.Normal);
                if (Mathf.Abs(denominator) <= 1e-5f) continue;
                float range = Vector3.Dot(patch.Center - camera, patch.Normal) /
                              denominator;
                if (range <= 0f || range >= hitRange) continue;
                Vector3 point = camera + rayDirection * range;
                float offset = Vector3.Distance(point, patch.Center);
                if (offset > patch.RadiusMetres) continue;
                if (patchIndex >= 0 && range <= intersectionRange) continue;
                intersection = point;
                intersectionRange = range;
                patchIndex = i;
                patchRadius = patch.RadiusMetres;
                patchOffset = offset;
            }
            return patchIndex >= 0;
        }

        private static float EstimateFootprintRadius(Vector3 source, Vector3 normal,
            int observationIndex, int gridX, int pixelStride, int depthWidth,
            int depthHeight, Matrix4x4 projectionInverse, Matrix4x4 viewInverse)
        {
            if (depthWidth <= 0 || depthHeight <= 0) return 0.001f;
            Matrix4x4 projection = projectionInverse.inverse;
            Matrix4x4 view = viewInverse.inverse;
            Vector4 clip = projection * (view * new Vector4(
                source.x, source.y, source.z, 1f));
            if (Mathf.Abs(clip.w) <= 1e-6f) return 0.001f;
            float ndcZ = clip.z / clip.w;
            int gx = gridX > 0 ? observationIndex % gridX : 0;
            int gy = gridX > 0 ? observationIndex / gridX : 0;
            float px = Mathf.Min(gx * Mathf.Max(1, pixelStride), depthWidth - 1);
            float py = Mathf.Min(gy * Mathf.Max(1, pixelStride), depthHeight - 1);
            float halfStride = Mathf.Max(1f, pixelStride * 0.5f);
            float neighbourX = px + halfStride <= depthWidth - 1
                ? px + halfStride
                : Mathf.Max(0f, px - halfStride);
            float neighbourY = py + halfStride <= depthHeight - 1
                ? py + halfStride
                : Mathf.Max(0f, py - halfStride);
            Vector3 xPoint = ReconstructAtPixel(neighbourX, py, ndcZ,
                depthWidth, depthHeight, projectionInverse, viewInverse);
            Vector3 yPoint = ReconstructAtPixel(px, neighbourY, ndcZ,
                depthWidth, depthHeight, projectionInverse, viewInverse);
            Vector3 n = SafeDirection(normal);
            Vector3 dx = xPoint - source;
            Vector3 dy = yPoint - source;
            dx -= n * Vector3.Dot(dx, n);
            dy -= n * Vector3.Dot(dy, n);
            return Mathf.Max(0.001f,
                Mathf.Sqrt(dx.sqrMagnitude + dy.sqrMagnitude));
        }

        private static Vector3 ReconstructAtPixel(float px, float py, float ndcZ,
            int width, int height, Matrix4x4 projectionInverse,
            Matrix4x4 viewInverse)
        {
            Vector4 hcs = new Vector4(
                px / width * 2f - 1f,
                py / height * 2f - 1f,
                ndcZ,
                1f);
            Vector4 world = viewInverse * (projectionInverse * hcs);
            if (Mathf.Abs(world.w) <= 1e-6f) return Vector3.zero;
            return new Vector3(world.x / world.w, world.y / world.w,
                world.z / world.w);
        }

        private void ChallengeFreeSpace(Vector3 camera, Vector3 hit, int frame,
            int platformFrame, float viewRadius, float motionQuality,
            HashSet<CellKey> challengedThisFrame, ref FrameCounters counters)
        {
            Vector3 ray = hit - camera;
            float hitRange = ray.magnitude;
            if (hitRange <= CellMetres) return;
            Vector3 rayDirection = ray / hitRange;
            int steps = Mathf.Min(128, Mathf.CeilToInt(hitRange / CellMetres));
            for (int step = 1; step < steps; step++)
            {
                float distance = step * CellMetres;
                if (distance >= hitRange) break;
                Vector3 sample = camera + rayDirection * distance;
                int x = Mathf.FloorToInt(sample.x / CellMetres);
                int y = Mathf.FloorToInt(sample.y / CellMetres);
                int z = Mathf.FloorToInt(sample.z / CellMetres);
                for (int axis = 0; axis < 6; axis++)
                {
                    CellKey key = CellKey.FromCoordinates(x, y, z, axis);
                    if (challengedThisFrame.Contains(key) ||
                        !_cells.TryGetValue(key, out CellState state) ||
                        state.FingerprintId <= 0 ||
                        state.FingerprintPatchCount <= 0)
                        continue;
                    bool auditingRejectedFingerprint =
                        state.Verdict == Verdict.Reject &&
                        _activePostVerdicts.ContainsKey(key);
                    if (state.Verdict != Verdict.Accept &&
                        !auditingRejectedFingerprint)
                        continue;
                    if (!auditingRejectedFingerprint)
                        counters.FingerprintAssociationTests++;
                    if (!TryIntersectFingerprint(camera, rayDirection, hitRange,
                            state, out Vector3 intersection,
                            out float intersectionRange, out int patchIndex,
                            out float patchRadius, out float patchOffset))
                    {
                        if (!auditingRejectedFingerprint)
                            counters.FingerprintAssociationMisses++;
                        continue;
                    }
                    if (!auditingRejectedFingerprint)
                        counters.FingerprintAssociationHits++;
                    float gap = hitRange - intersectionRange;
                    if (gap <= 0f) continue;

                    challengedThisFrame.Add(key);
                    if (auditingRejectedFingerprint)
                    {
                        ObservePostVerdictFreeSpace(key, state, camera, frame, gap);
                        continue;
                    }
                    counters.FreeSpaceChallenges++;
                    counters.MaxFreeSpaceGapMetres = Mathf.Max(
                        counters.MaxFreeSpaceGapMetres, gap);
                    state.MaxFreeGapMetres = Mathf.Max(state.MaxFreeGapMetres, gap);
                    if (gap < 0.020f) counters.FreeGapLt20++;
                    else if (gap < 0.050f) counters.FreeGap20To50++;
                    else if (gap < 0.100f) counters.FreeGap50To100++;
                    else counters.FreeGapGe100++;
                    if (gap < ReliableFreeGapMetres) continue;

                    Vector3 direction = SafeDirection(camera - state.FingerprintCenter);
                    bool independent = TryAddIndependentWitness(
                        ref state.ChallengeWitnesses,
                        ref state.ChallengeWitnessCount,
                        ChallengeWitnessCapacity, camera, direction, frame,
                        out float baseline, out float spread);
                    state.ConsecutiveChallengeVotes++;
                    state.LastChallengePeakVotes = Mathf.Max(
                        state.LastChallengePeakVotes,
                        state.ConsecutiveChallengeVotes);
                    state.ChallengeClearedFrame = -1;
                    state.IndependentChallengeViews = state.ChallengeWitnessCount;
                    if (independent)
                        counters.IndependentFreeSpaceChallenges++;
                    _graduationRace.ObserveReliableChallenge(key.X, key.Y, key.Z,
                        key.Axis, state.FingerprintGeneration, frame, gap,
                        independent);
                    state.RecoveryWitnessCount = 0;
                    state.RecoverySupportViews = 0;
                    state.RecoveryPatchCount = 0;
                    Verdict before = state.Verdict;
                    string reason = independent
                        ? "independent_free_space_vote"
                        : "correlated_free_space_vote";
                    if (before == Verdict.Accept &&
                        state.ConsecutiveChallengeVotes >= RequiredChallengeVotes &&
                        state.IndependentChallengeViews >= RequiredChallengeViews)
                    {
                        Transition(state, Verdict.Reject);
                        StartPostVerdict(key, state, frame, platformFrame);
                        counters.RejectTransitions++;
                        reason = "rejected_by_three_free_space_votes";
                    }
                    AppendEvent(frame, platformFrame, independent
                            ? "challenge_independent" : "challenge_correlated", key,
                        before, state.Verdict, reason, hit, state.FingerprintCenter,
                        0f, 0f,
                        gap * 1000f, viewRadius, motionQuality, baseline * 1000f,
                        spread, state, patchIndex, intersection,
                        intersectionRange * 1000f, patchRadius * 1000f,
                        patchOffset * 1000f, "fingerprint_patch_hit");
                }
            }
        }

        private void Transition(CellState state, Verdict next)
        {
            if (state.Verdict == next) return;
            bool correctionChanged = state.Verdict == Verdict.Reject ||
                                     next == Verdict.Reject;
            if (next == Verdict.Reject)
                state.PaperCorrectionGateActive = true;
            if (state.Verdict == Verdict.Hold) _heldCells--;
            else if (state.Verdict == Verdict.Accept) _acceptedCells--;
            else _rejectedCells--;
            state.Verdict = next;
            if (next == Verdict.Hold) _heldCells++;
            else if (next == Verdict.Accept) _acceptedCells++;
            else _rejectedCells++;
            if (correctionChanged) _correctionRevision++;
        }

        private void AppendFrame(int frame, int platformFrame, FrameCounters c)
        {
            _pendingFrames.Append(I(frame)).Append(',').Append(I(platformFrame))
                .Append(',').Append(I(Time.frameCount)).Append(',')
                .Append(D(Time.unscaledTimeAsDouble)).Append(',').Append(I(c.Valid))
                .Append(',').Append(I(c.HoldRawMissing)).Append(',')
                .Append(I(c.HoldDualDisagree)).Append(',').Append(I(c.HoldNoStable))
                .Append(',').Append(I(c.HoldResidual)).Append(',').Append(I(c.HoldNormal))
                .Append(',').Append(I(c.HoldOpposed)).Append(',').Append(I(c.HoldMotion))
                .Append(',').Append(I(c.HoldOuter)).Append(',').Append(I(c.EligibleSupport))
                .Append(',').Append(I(c.IndependentSupport)).Append(',')
                .Append(I(c.ObservationAccept)).Append(',').Append(I(c.ObservationHold))
                .Append(',').Append(I(c.ObservationReject)).Append(',')
                .Append(I(c.AcceptTransitions)).Append(',').Append(I(c.RejectTransitions))
                .Append(',').Append(I(c.ReopenTransitions)).Append(',')
                .Append(I(c.NewIndependentViewCells)).Append(',')
                .Append(I(c.CorrelatedViewMatches)).Append(',')
                .Append(F(c.MaxBaselineMetres * 1000f)).Append(',')
                .Append(F(c.MaxViewSpreadDeg)).Append(',').Append(I(c.FreeSpaceChallenges))
                .Append(',').Append(I(c.FreeGapLt20)).Append(',')
                .Append(I(c.FreeGap20To50)).Append(',').Append(I(c.FreeGap50To100))
                .Append(',').Append(I(c.FreeGapGe100)).Append(',')
                .Append(I(c.IndependentFreeSpaceChallenges)).Append(',')
                .Append(F(c.MaxFreeSpaceGapMetres * 1000f)).Append(',')
                .Append(I(c.FingerprintAssociationTests)).Append(',')
                .Append(I(c.FingerprintAssociationHits)).Append(',')
                .Append(I(c.FingerprintAssociationMisses)).Append(',')
                .Append(I(_heldCells)).Append(',').Append(I(_acceptedCells)).Append(',')
                .Append(I(_rejectedCells)).Append(",shadow_only\n");
            _frameRows++;
            if (_pendingFrames.Length + _pendingEvents.Length >= 131072)
                FlushLocked();
        }

        private void AppendEvent(int frame, int platformFrame, string eventName,
            CellKey key, Verdict before, Verdict after, string reason,
            Vector3 source, Vector3 target, float residualMm, float angleDeg,
            float gapMm, float viewRadius, float motionQuality, float baselineMm,
            float spreadDeg, CellState state, int fingerprintPatchIndex = -1,
            Vector3 fingerprintIntersection = default(Vector3),
            float intersectionRangeMm = 0f, float footprintRadiusMm = 0f,
            float footprintOffsetMm = 0f, string association = "")
        {
            float strength = after == Verdict.Reject
                ? Mathf.Clamp01(state.ConsecutiveChallengeVotes /
                                (float)RequiredChallengeVotes)
                : Mathf.Clamp01(state.IndependentSupportViews /
                                (float)RequiredSupportViews);
            _pendingEvents.Append(I(frame)).Append(',').Append(I(platformFrame))
                .Append(',').Append(eventName).Append(',').Append(I(key.X)).Append(',')
                .Append(I(key.Y)).Append(',').Append(I(key.Z)).Append(',')
                .Append(I(key.Axis)).Append(',').Append(VerdictName(before)).Append(',')
                .Append(VerdictName(after)).Append(',').Append(reason).Append(',')
                .Append(F(source.x)).Append(',').Append(F(source.y)).Append(',')
                .Append(F(source.z)).Append(',').Append(F(target.x)).Append(',')
                .Append(F(target.y)).Append(',').Append(F(target.z)).Append(',')
                .Append(F(residualMm)).Append(',').Append(F(angleDeg)).Append(',')
                .Append(F(gapMm)).Append(',').Append(F(viewRadius)).Append(',')
                .Append(F(motionQuality)).Append(',').Append(F(baselineMm)).Append(',')
                .Append(F(spreadDeg)).Append(',').Append(I(state.SafeSupportFrames))
                .Append(',').Append(I(state.IndependentSupportViews)).Append(',')
                .Append(I(state.ConsecutiveChallengeVotes)).Append(',')
                .Append(I(state.IndependentChallengeViews)).Append(',')
                .Append(I(state.RecoverySupportViews)).Append(',')
                .Append(F(strength)).Append(',').Append(I(state.FingerprintId))
                .Append(',').Append(I(state.FingerprintGeneration)).Append(',')
                .Append(I(state.FingerprintAcceptedFrame)).Append(',')
                .Append(I(state.FingerprintPatchCount)).Append(',')
                .Append(I(fingerprintPatchIndex)).Append(',')
                .Append(F(fingerprintIntersection.x)).Append(',')
                .Append(F(fingerprintIntersection.y)).Append(',')
                .Append(F(fingerprintIntersection.z)).Append(',')
                .Append(F(intersectionRangeMm)).Append(',')
                .Append(F(footprintRadiusMm)).Append(',')
                .Append(F(footprintOffsetMm)).Append(',')
                .Append(association).Append('\n');
            _eventRows++;
        }

        private void WriteCellSnapshot()
        {
            if (string.IsNullOrEmpty(_directory)) return;
            try
            {
                var rows = new List<KeyValuePair<CellKey, CellState>>(_cells);
                rows.Sort((a, b) => a.Key.CompareTo(b.Key));
                var sb = new StringBuilder(Mathf.Max(4096, rows.Count * 160));
                sb.Append("cellX,cellY,cellZ,axis,verdict,centerX,centerY,centerZ,normalX,normalY,normalZ,firstFrame,lastSeenFrame,safeSupportFrames,independentSupportViews,challengeVotes,independentChallengeViews,recoverySupportViews,maxBaselineMm,maxViewSpreadDeg,maxFreeGapMm,decisionStrength,fingerprintId,fingerprintGeneration,fingerprintAcceptedFrame,fingerprintCenterX,fingerprintCenterY,fingerprintCenterZ,fingerprintNormalX,fingerprintNormalY,fingerprintNormalZ,fingerprintPatchCount\n");
                foreach (KeyValuePair<CellKey, CellState> pair in rows)
                {
                    CellState s = pair.Value;
                    float strength = s.Verdict == Verdict.Reject
                        ? Mathf.Clamp01(s.ConsecutiveChallengeVotes /
                                        (float)RequiredChallengeVotes)
                        : Mathf.Clamp01(s.IndependentSupportViews /
                                        (float)RequiredSupportViews);
                    sb.Append(I(pair.Key.X)).Append(',').Append(I(pair.Key.Y)).Append(',')
                        .Append(I(pair.Key.Z)).Append(',').Append(I(pair.Key.Axis)).Append(',')
                        .Append(VerdictName(s.Verdict)).Append(',').Append(F(s.Center.x))
                        .Append(',').Append(F(s.Center.y)).Append(',').Append(F(s.Center.z))
                        .Append(',').Append(F(s.Normal.x)).Append(',').Append(F(s.Normal.y))
                        .Append(',').Append(F(s.Normal.z)).Append(',').Append(I(s.FirstFrame))
                        .Append(',').Append(I(s.LastSeenFrame)).Append(',')
                        .Append(I(s.SafeSupportFrames)).Append(',')
                        .Append(I(s.IndependentSupportViews)).Append(',')
                        .Append(I(s.ConsecutiveChallengeVotes)).Append(',')
                        .Append(I(s.IndependentChallengeViews)).Append(',')
                        .Append(I(s.RecoverySupportViews)).Append(',')
                        .Append(F(s.MaxBaselineMetres * 1000f)).Append(',')
                        .Append(F(s.MaxViewSpreadDeg)).Append(',')
                        .Append(F(s.MaxFreeGapMetres * 1000f)).Append(',')
                        .Append(F(strength)).Append(',').Append(I(s.FingerprintId))
                        .Append(',').Append(I(s.FingerprintGeneration)).Append(',')
                        .Append(I(s.FingerprintAcceptedFrame)).Append(',')
                        .Append(F(s.FingerprintCenter.x)).Append(',')
                        .Append(F(s.FingerprintCenter.y)).Append(',')
                        .Append(F(s.FingerprintCenter.z)).Append(',')
                        .Append(F(s.FingerprintNormal.x)).Append(',')
                        .Append(F(s.FingerprintNormal.y)).Append(',')
                        .Append(F(s.FingerprintNormal.z)).Append(',')
                        .Append(I(s.FingerprintPatchCount)).Append('\n');
                }
                File.WriteAllText(Path.Combine(_directory, "verdict_cells.csv"),
                    sb.ToString(), new UTF8Encoding(false));
            }
            catch
            {
                _writeErrors++;
            }
        }

        private void WriteFingerprintSnapshot()
        {
            if (string.IsNullOrEmpty(_directory)) return;
            try
            {
                var rows = new List<FingerprintRecord>(_fingerprintHistory);
                rows.Sort((a, b) =>
                {
                    int value = a.FingerprintId.CompareTo(b.FingerprintId);
                    return value != 0 ? value : a.PatchIndex.CompareTo(b.PatchIndex);
                });
                var sb = new StringBuilder(Mathf.Max(4096, rows.Count * 160));
                sb.Append("cellX,cellY,cellZ,axis,fingerprintId,fingerprintGeneration,fingerprintAcceptedFrame,patchIndex,patchFrame,centerX,centerY,centerZ,normalX,normalY,normalZ,radiusMm\n");
                foreach (FingerprintRecord row in rows)
                {
                    SurfacePatch patch = row.Patch;
                    sb.Append(I(row.Key.X)).Append(',').Append(I(row.Key.Y))
                        .Append(',').Append(I(row.Key.Z)).Append(',')
                        .Append(I(row.Key.Axis)).Append(',')
                        .Append(I(row.FingerprintId)).Append(',')
                        .Append(I(row.Generation)).Append(',')
                        .Append(I(row.AcceptedFrame)).Append(',')
                        .Append(I(row.PatchIndex)).Append(',')
                        .Append(I(patch.Frame)).Append(',')
                        .Append(F(patch.Center.x)).Append(',')
                        .Append(F(patch.Center.y)).Append(',')
                        .Append(F(patch.Center.z)).Append(',')
                        .Append(F(patch.Normal.x)).Append(',')
                        .Append(F(patch.Normal.y)).Append(',')
                        .Append(F(patch.Normal.z)).Append(',')
                        .Append(F(patch.RadiusMetres * 1000f)).Append('\n');
                }
                File.WriteAllText(Path.Combine(_directory, "verdict_fingerprints.csv"),
                    sb.ToString(), new UTF8Encoding(false));
            }
            catch
            {
                _writeErrors++;
            }
        }

        private void WritePostVerdictSnapshot()
        {
            if (string.IsNullOrEmpty(_directory)) return;
            try
            {
                var rows = new List<PostVerdictRecord>(_postVerdictHistory);
                rows.Sort((a, b) =>
                {
                    int value = a.RejectFrame.CompareTo(b.RejectFrame);
                    if (value != 0) return value;
                    value = a.FingerprintId.CompareTo(b.FingerprintId);
                    return value != 0
                        ? value
                        : a.FingerprintGeneration.CompareTo(b.FingerprintGeneration);
                });
                var sb = new StringBuilder(Mathf.Max(4096, rows.Count * 240));
                sb.Append("cellX,cellY,cellZ,axis,rejectedFingerprintId,rejectedFingerprintGeneration,rejectFrame,rejectPlatformFrame,windowEndFrame,lastSessionFrame,windowComplete,finalOutcome,firstFollowupFrame,lastFollowupFrame,followupSpanFrames,observedFrames,contactFrames,rayIntersectionFrames,reliableFreeSpaceFrames,independentFreeSpaceViews,safeSupportFrames,independentSupportViews,freeSpaceReconfirmed,freeSpaceReconfirmedFrame,reopened,reopenedFrame,reopenedFingerprintId,reopenedFingerprintGeneration,maxFreeGapMm,maxBaselineMm,maxViewSpreadDeg,authority\n");
                foreach (PostVerdictRecord row in rows)
                {
                    int windowEndFrame = row.RejectFrame + PostVerdictWindowFrames;
                    bool windowComplete = _lastEvaluatedFrame >= windowEndFrame;
                    int followupSpan = row.LastFollowupFrame >= row.FirstFollowupFrame &&
                                       row.FirstFollowupFrame >= 0
                        ? row.LastFollowupFrame - row.FirstFollowupFrame
                        : 0;
                    sb.Append(I(row.Key.X)).Append(',').Append(I(row.Key.Y))
                        .Append(',').Append(I(row.Key.Z)).Append(',')
                        .Append(I(row.Key.Axis)).Append(',')
                        .Append(I(row.FingerprintId)).Append(',')
                        .Append(I(row.FingerprintGeneration)).Append(',')
                        .Append(I(row.RejectFrame)).Append(',')
                        .Append(I(row.RejectPlatformFrame)).Append(',')
                        .Append(I(windowEndFrame)).Append(',')
                        .Append(I(_lastEvaluatedFrame)).Append(',')
                        .Append(windowComplete ? "1" : "0").Append(',')
                        .Append(PostVerdictOutcome(row, windowComplete)).Append(',')
                        .Append(I(row.FirstFollowupFrame)).Append(',')
                        .Append(I(row.LastFollowupFrame)).Append(',')
                        .Append(I(followupSpan)).Append(',')
                        .Append(I(row.ObservedFrames)).Append(',')
                        .Append(I(row.ContactFrames)).Append(',')
                        .Append(I(row.RayIntersectionFrames)).Append(',')
                        .Append(I(row.ReliableFreeSpaceFrames)).Append(',')
                        .Append(I(row.FreeSpaceWitnessCount)).Append(',')
                        .Append(I(row.SafeSupportFrames)).Append(',')
                        .Append(I(row.SupportWitnessCount)).Append(',')
                        .Append(row.FreeSpaceReconfirmed ? "1" : "0").Append(',')
                        .Append(I(row.FreeSpaceReconfirmedFrame)).Append(',')
                        .Append(row.Reopened ? "1" : "0").Append(',')
                        .Append(I(row.ReopenedFrame)).Append(',')
                        .Append(I(row.ReopenedFingerprintId)).Append(',')
                        .Append(I(row.ReopenedGeneration)).Append(',')
                        .Append(F(row.MaxFreeGapMetres * 1000f)).Append(',')
                        .Append(F(row.MaxBaselineMetres * 1000f)).Append(',')
                        .Append(F(row.MaxViewSpreadDeg)).Append(',')
                        .Append("shadow_only_no_production_authority\n");
                }
                File.WriteAllText(Path.Combine(_directory, "verdict_followups.csv"),
                    sb.ToString(), new UTF8Encoding(false));
            }
            catch
            {
                _writeErrors++;
            }
        }

        private static string PostVerdictOutcome(PostVerdictRecord record,
            bool windowComplete)
        {
            if (record.Reopened)
            {
                return record.FreeSpaceReconfirmed
                    ? "support_reestablished_after_conflict"
                    : "support_reestablished";
            }
            if (record.FreeSpaceReconfirmed) return "confirmed_false_surface";
            if (record.ObservedFrames <= 0)
            {
                return windowComplete
                    ? "unknown_not_reobserved"
                    : "unknown_censored_session_end";
            }
            if (record.SafeSupportFrames > 0 || record.SupportWitnessCount > 0)
                return "unknown_insufficient_support";
            return "unknown_insufficient_followup";
        }

        private void FlushLocked()
        {
            try
            {
                if (_pendingFrames.Length > 0 && !string.IsNullOrEmpty(_framePath))
                {
                    File.AppendAllText(_framePath, _pendingFrames.ToString(),
                        new UTF8Encoding(false));
                    _pendingFrames.Clear();
                }
                if (_pendingEvents.Length > 0 && !string.IsNullOrEmpty(_eventPath))
                {
                    File.AppendAllText(_eventPath, _pendingEvents.ToString(),
                        new UTF8Encoding(false));
                    _pendingEvents.Clear();
                }
            }
            catch
            {
                _writeErrors++;
            }
        }

        private static float ViewRadius(int index, int gridX, int pixelStride,
            int depthWidth, int depthHeight)
        {
            int gx = gridX > 0 ? index % gridX : 0;
            int gy = gridX > 0 ? index / gridX : 0;
            float px = Mathf.Min(gx * Mathf.Max(1, pixelStride),
                Mathf.Max(0, depthWidth - 1));
            float py = Mathf.Min(gy * Mathf.Max(1, pixelStride),
                Mathf.Max(0, depthHeight - 1));
            float nx = depthWidth > 0 ? px / depthWidth * 2f - 1f : 0f;
            float ny = depthHeight > 0 ? py / depthHeight * 2f - 1f : 0f;
            return Mathf.Max(Mathf.Abs(nx), Mathf.Abs(ny));
        }

        private static Vector3 SafeDirection(Vector3 value) =>
            value.sqrMagnitude > 1e-10f ? value.normalized : Vector3.forward;

        private static void SeedWitness(ref ViewWitness[] witnesses, ref int count,
            int capacity, Vector3 camera, Vector3 direction, int frame)
        {
            if (capacity <= 0) return;
            if (witnesses == null || witnesses.Length < capacity)
                witnesses = new ViewWitness[capacity];
            witnesses[0] = new ViewWitness
            {
                Camera = camera,
                Direction = direction,
                Frame = frame
            };
            count = 1;
        }

        private static bool TryAddIndependentWitness(
            ref ViewWitness[] witnesses, ref int count, int capacity,
            Vector3 camera, Vector3 direction, int frame,
            out float nearestBaseline, out float nearestSpread)
        {
            return TryAddIndependentWitness(ref witnesses, ref count, capacity,
                camera, direction, frame, out nearestBaseline,
                out nearestSpread, out _);
        }

        private static bool TryAddIndependentWitness(
            ref ViewWitness[] witnesses, ref int count, int capacity,
            Vector3 camera, Vector3 direction, int frame,
            out float nearestBaseline, out float nearestSpread,
            out IndependentWitnessDecision decision)
        {
            nearestBaseline = 0f;
            nearestSpread = 0f;
            decision = IndependentWitnessDecision.InvalidCapacity;
            if (capacity <= 0) return false;
            if (count <= 0 || witnesses == null)
            {
                SeedWitness(ref witnesses, ref count, capacity,
                    camera, direction, frame);
                bool seeded = count > 0;
                decision = seeded
                    ? IndependentWitnessDecision.Accepted
                    : IndependentWitnessDecision.InvalidCapacity;
                return seeded;
            }

            int available = Mathf.Min(count, witnesses.Length);
            bool blockedByTime = false;
            bool blockedByPose = false;
            float closestScore = float.PositiveInfinity;
            for (int i = 0; i < available; i++)
            {
                ViewWitness witness = witnesses[i];
                IndependentViewWitnessPolicy.ComparePair(
                    witness.Camera, witness.Direction, witness.Frame,
                    camera, direction, frame, out float baseline,
                    out float spread, out bool timeSeparated,
                    out bool poseSeparated);
                float score = Mathf.Max(
                    baseline / IndependentBaselineMetres,
                    spread / IndependentAngleDeg);
                if (score < closestScore)
                {
                    closestScore = score;
                    nearestBaseline = baseline;
                    nearestSpread = spread;
                }
                if (!timeSeparated) blockedByTime = true;
                if (!poseSeparated) blockedByPose = true;
            }
            decision = IndependentViewWitnessPolicy.ClassifyBlocked(
                blockedByTime, blockedByPose);
            if (decision != IndependentWitnessDecision.Accepted) return false;
            if (count >= capacity)
            {
                decision = IndependentWitnessDecision.CapacityFull;
                return false;
            }

            witnesses[count] = new ViewWitness
            {
                Camera = camera,
                Direction = direction,
                Frame = frame
            };
            count++;
            decision = IndependentWitnessDecision.Accepted;
            return true;
        }

        private static string VerdictName(Verdict verdict)
        {
            if (verdict == Verdict.Accept) return "accept";
            if (verdict == Verdict.Reject) return "reject";
            return "hold";
        }

        private static string BuildSchemaJson()
        {
            return "{\n" +
                   "  \"schema\": \"scancover.virtual_probe_shadow_verdict.v8\",\n" +
                   "  \"authority\": \"none; outputs are write-only diagnostics and are never consumed by GunGel, TSDF, paper or mesh\",\n" +
                   "  \"profile\": \"quest3_room_candidate_fingerprint_shadow_20260828\",\n" +
                   "  \"verdicts\": {\"hold\":\"evidence exists but cannot appoint or delete geometry\",\"accept\":\"two safe independent support witnesses\",\"reject\":\"three consecutive reliable free-space votes with at least two independent witness poses\"},\n" +
                   "  \"recovery\": \"a rejected cell is reopened by two independent safe support views\",\n" +
                   "  \"postVerdictAudit\": \"every rejected fingerprint generation becomes a session-long spatial task; continuing independent free-space evidence confirms a false surface, a newly minted supported generation records support reestablishment, and elapsed time alone never removes an unresolved task\",\n" +
                   "  \"postVerdictOutcomes\": {\"confirmed_false_surface\":\"two independent reliable free-space views after Reject\",\"support_reestablished\":\"two independent safe-support views minted a new generation\",\"support_reestablished_after_conflict\":\"both follow-up free-space and later support were observed\",\"unknown_*\":\"follow-up was absent, censored or insufficient and must not train a rule\"},\n" +
                   "  \"independence\": \"new poses are compared pairwise with stored real camera witnesses; no angular direction bins are used\",\n" +
                   "  \"fingerprint\": \"each accepted generation seals a stable id plus the real safe-support surfel footprints present at acceptance; a free-space vote is attributed only when its ray intersects one of those patches before the new depth hit\",\n" +
                   "  \"footprint\": \"each patch radius is reconstructed from half the actual depth sampling stride at that observation depth and projected into the candidate tangent plane\",\n" +
                   "  \"acquisitionRoi\": \"automatic passive acquisition across the trusted depth field; admission still requires dual agreement, motion quality and view-radius gates. The fixed plus/minus 7 degree frame is read-only visualization and has no verdict authority; production remains untouched\",\n" +
                   "  \"challengeRetraction\": \"any safe support interrupts consecutive free-space votes, but every such retraction is emitted as challenge_cleared_by_safe_support before counters are reset\",\n" +
                   "  \"taskLifetime\": \"accepted fingerprints keep stable id, generation and world centre for the session; unresolved Reject follow-up remains active until resolved or session export; postVerdictWindowFrames is an offline minimum observation horizon, not an expiry gate\",\n" +
                   "  \"graduationRace\": \"parallel shadow-only graduation contracts consume the same safe-support stream and are labelled only by later follow-up outcomes; every callback exports raw pose plus production/race independence decisions; revisit timing is an explicit proxy, not proof of leaving the target\",\n" +
                   "  \"visualization\": \"provisional and accepted background crosses are deterministically sampled by immutable 5 cm world-cell key and use a creation-time display anchor; state transitions change colour without changing position; cleared challenge votes retain a short dim trace\",\n" +
                   "  \"thresholds\": {\"cellMetres\":0.05,\"readOnlyGuidanceFrameHalfAngleDeg\":7.0,\"motionQualityMin\":0.70,\"outerRadiusMax\":0.84,\"independentBaselineMetres\":0.08,\"independentAngleDeg\":3.0,\"independentFrameGap\":2,\"supportFrames\":2,\"supportViews\":2,\"reliableFreeGapMetres\":0.05,\"challengeVotes\":3,\"challengeViews\":2,\"recoveryViews\":2,\"postVerdictWindowFrames\":480,\"postVerdictFreeViews\":2,\"seenWitnessCapacity\":4,\"supportWitnessCapacity\":4,\"challengeWitnessCapacity\":4,\"recoveryWitnessCapacity\":2,\"fingerprintPatchCapacity\":12,\"recoveryPatchCapacity\":4},\n" +
                   "  \"decisionStrength\": \"evidence count saturation only; it is not a calibrated probability\",\n" +
                   "  \"files\": {\"frames\":\"verdict_frames.csv\",\"events\":\"verdict_events.csv\",\"cells\":\"verdict_cells.csv\",\"fingerprints\":\"verdict_fingerprints.csv\",\"followups\":\"verdict_followups.csv\",\"graduationRaceCandidates\":\"graduation_race_candidates.csv\",\"graduationRaceSummary\":\"graduation_race_summary.csv\",\"graduationRaceWitnessEvents\":\"graduation_race_witness_events.csv\",\"graduationRaceSchema\":\"graduation_race_schema.json\"}\n" +
                   "}\n";
        }

        private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
        private static string F(float value) => float.IsNaN(value) || float.IsInfinity(value)
            ? "" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string D(double value) => double.IsNaN(value) || double.IsInfinity(value)
            ? "" : value.ToString("R", CultureInfo.InvariantCulture);

        private readonly struct CellKey : IEquatable<CellKey>, IComparable<CellKey>
        {
            internal readonly int X;
            internal readonly int Y;
            internal readonly int Z;
            internal readonly int Axis;

            private CellKey(int x, int y, int z, int axis)
            {
                X = x;
                Y = y;
                Z = z;
                Axis = axis;
            }

            internal static CellKey From(Vector3 position, Vector3 normal)
            {
                Vector3 abs = new Vector3(Mathf.Abs(normal.x), Mathf.Abs(normal.y),
                    Mathf.Abs(normal.z));
                int axis;
                if (abs.x >= abs.y && abs.x >= abs.z) axis = normal.x >= 0f ? 0 : 1;
                else if (abs.y >= abs.z) axis = normal.y >= 0f ? 2 : 3;
                else axis = normal.z >= 0f ? 4 : 5;
                return FromCoordinates(
                    Mathf.FloorToInt(position.x / CellMetres),
                    Mathf.FloorToInt(position.y / CellMetres),
                    Mathf.FloorToInt(position.z / CellMetres), axis);
            }

            internal static CellKey FromCoordinates(int x, int y, int z, int axis) =>
                new CellKey(x, y, z, axis);

            public bool Equals(CellKey other) => X == other.X && Y == other.Y &&
                Z == other.Z && Axis == other.Axis;
            public override bool Equals(object obj) => obj is CellKey other && Equals(other);
            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = X;
                    hash = hash * 397 ^ Y;
                    hash = hash * 397 ^ Z;
                    return hash * 397 ^ Axis;
                }
            }

            public int CompareTo(CellKey other)
            {
                int value = X.CompareTo(other.X);
                if (value != 0) return value;
                value = Y.CompareTo(other.Y);
                if (value != 0) return value;
                value = Z.CompareTo(other.Z);
                return value != 0 ? value : Axis.CompareTo(other.Axis);
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 候选不确定足迹直投 raw/post 深度的只读影子账本。GPU 只提交“足迹内
    /// 存在可靠表面”或“整个足迹前方存在 45 mm 可靠自由空间”的证词；
    /// 本类按稳定 ID、几何世代和独立视角记账。
    /// 它不写候选、TSDF、纸皮和显示层，也没有任何生产去留权限。
    /// </summary>
    internal sealed class DirectProjectionChallengeShadowLedger : IDisposable
    {
        private const int HeaderRows = 3;
        private const int RecordRows = 9;
        private const int MaxWitnessesPerClass = 8;
        private const float GeometryResetDistance = 0.02f;
        private const float GeometryResetAngleDeg = 15f;
        private const int GuardSupportWindowFrames = 16;
        private const int GuardSettleFrames = 8;
        private const int GuardDecisionNone = 0;
        private const int GuardDecisionAllow = 1;
        private const int GuardDecisionRecentSupportVeto = 2;
        private const int GuardDecisionConflictVeto = 3;
        private const int GuardDecisionPendingAtSeal = 4;
        private const int GuardDecisionGenerationResetCancelled = 5;

        private sealed class Witness
        {
            internal int Frame;
            internal Vector3 Camera;
            internal Vector3 Direction;
        }

        private sealed class CandidateState
        {
            internal uint StableId;
            internal uint CandidateIndex;
            internal int Generation;
            internal Vector3 Center;
            internal Vector3 Normal;
            internal int FirstFrame;
            internal int LastFrame;
            internal int SupportEvents;
            internal int FreeEvents;
            internal int MixedEvents;
            internal int IncompleteEvents;
            internal int NeutralEvents;
            internal bool SupportConfirmed;
            internal bool FreeConfirmed;
            internal bool Conflict;
            internal int LastSupportFrame = -1;
            internal bool GuardPending;
            internal int GuardFreeConfirmedFrame = -1;
            internal int GuardDueFrame = -1;
            internal int GuardLastSupportFrame = -1;
            internal int GuardDecision;
            internal int GuardDecisionFrame = -1;
            internal readonly List<Witness> Support = new List<Witness>(MaxWitnessesPerClass);
            internal readonly List<Witness> Free = new List<Witness>(MaxWitnessesPerClass);
        }

        private readonly Dictionary<uint, CandidateState> _candidates =
            new Dictionary<uint, CandidateState>(2048);
        private readonly StringBuilder _frameRows = new StringBuilder(32768);
        private readonly StringBuilder _eventRows = new StringBuilder(65536);
        private readonly StringBuilder _guardRows = new StringBuilder(16384);
        private readonly List<uint> _guardPendingIds = new List<uint>(256);
        private string _directory = string.Empty;
        private string _framesPath = string.Empty;
        private string _eventsPath = string.Empty;
        private string _guardPath = string.Empty;
        private bool _active;
        private int _frames;
        private int _records;
        private int _supportEvents;
        private int _freeEvents;
        private int _mixedEvents;
        private int _incompleteEvents;
        private int _neutralEvents;
        private int _eventRowsWritten;
        private int _supportConfirmedTransitions;
        private int _freeConfirmedTransitions;
        private int _conflicts;
        private int _overflowRecords;
        private int _readbackErrors;
        private int _writeErrors;
        private int _guardDecisionRowsWritten;
        private int _guardAllowTransitions;
        private int _guardRecentSupportVetoTransitions;
        private int _guardConflictVetoTransitions;
        private int _guardPendingAtSeal;
        private int _guardGenerationResetCancelled;

        internal int Frames => _frames;
        internal int Rows => _records;
        internal int EventRows => _eventRowsWritten;
        internal int WriteErrors => _writeErrors;
        internal int ReadbackErrors => _readbackErrors;
        internal int OverflowRecords => _overflowRecords;
        internal int GuardDecisionRows => _guardDecisionRowsWritten;
        internal int GuardAllowTransitions => _guardAllowTransitions;
        internal int GuardVetoTransitions =>
            _guardRecentSupportVetoTransitions + _guardConflictVetoTransitions;
        internal int GuardPendingAtSeal => _guardPendingAtSeal;
        internal string HudFixed =>
            $"足迹 帧{_frames:0000} 证{_records:0000} 自{_freeConfirmedTransitions:0000} 保{_supportConfirmedTransitions:0000}\n" +
            $"拟退{_guardAllowTransitions:0000} 拟拦{GuardVetoTransitions:0000} 待{_guardPendingIds.Count:0000} 溢{_overflowRecords:0000}\n" +
            $"窗{GuardSupportWindowFrames:00} 延{GuardSettleFrames:00} 冲{_conflicts:0000} 删除[关闭]";

        internal void Begin(string sessionDirectory)
        {
            End();
            if (string.IsNullOrEmpty(sessionDirectory)) return;
            _directory = Path.Combine(sessionDirectory, "probe_shadow", "direct45");
            Directory.CreateDirectory(_directory);
            _framesPath = Path.Combine(_directory, "frames.csv");
            _eventsPath = Path.Combine(_directory, "events.csv");
            _guardPath = Path.Combine(_directory, "guarded_decisions.csv");
            File.WriteAllText(_framesPath,
                "gunGelFrame,platformFrame,unityFrame,stableEvaluated,visible,footprintEvaluated,supportRecords,freeRecords,mixedRecords,incompleteRecords,neutralRecords,radiusClamped,writtenRecords,overflow,angularDegPerSec,linearMps,motionQuality,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(_eventsPath,
                "gunGelFrame,platformFrame,stableId,candidateIndex,generation,class,independent,independentCount,supportConfirmed,freeConfirmed,conflict,centerX,centerY,centerZ,varianceM2,normalX,normalY,normalZ,cameraX,cameraY,cameraZ,candidateRangeM,minGapM,maxGapM,meanGapM,viewRadius,incidence,totalSamples,validSamples,supportSamples,freeSamples,foregroundSamples,neutralSamples,invalidSamples,pixelRadius,hardEdgeSamples,temporalRiskSamples,dualDisagreeSamples,radiusClamped,maxRawPostDifferenceM,lastSafeFrame,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(_guardPath,
                "stableId,candidateIndex,generation,freeConfirmedFrame,dueFrame,decisionFrame,lastSupportFrame,supportAgeAtDecisionFrames,supportWindowFrames,settleFrames,decision,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_directory, "schema.json"),
                "{\n" +
                "  \"schema\": \"scancover.direct_projection_footprint_shadow.v3\",\n" +
                "  \"authority\": \"shadow_only_zero_production_authority\",\n" +
                "  \"source\": \"stable GunGel candidate variance footprint centered by projection into the same-frame complete right-eye platform-pre-QRS and post-QRS depth textures\",\n" +
                "  \"footprint\": \"projected two-sigma candidate variance, minimum one pixel and maximum five pixels; a clamped footprint must abstain\",\n" +
                "  \"freeProof\": \"every footprint pixel has valid dual-agree raw/post depth at least 0.045 m behind the candidate, with no hard-edge or temporal risk\",\n" +
                "  \"supportProof\": \"at least one valid footprint pixel lies within 0.018 m of candidate range and no footprint pixel is free or foreground\",\n" +
                "  \"abstention\": \"mixed foreground/background, invalid depth, dual disagreement, hard-edge, temporal risk, neutral band or radius clamp\",\n" +
                "  \"independentView\": \"frame gap >= 2 and baseline >= 0.08 m or viewing-direction spread >= 3 deg\",\n" +
                "  \"confirmation\": \"two independent witnesses of the same class for the same stable ID and geometry generation; support/free conflict is order-independent\",\n" +
                "  \"geometryReset\": \"new generation when center moves > 0.02 m or unsigned normal changes > 15 deg\",\n" +
                "  \"samplingStride\": 4,\n" +
                "  \"guardedDecisionShadow\": {\"supportLookbackFrames\": 16, \"settleFrames\": 8, \"rule\": \"after free confirmation, wait eight GunGel frames; veto on any coherent support from the prior sixteen frames through the due frame, or on an order-independent support/free conflict; otherwise record would-allow\", \"productionAuthority\": \"none\"}\n" +
                "}\n", new UTF8Encoding(false));
            _candidates.Clear();
            _frameRows.Clear();
            _eventRows.Clear();
            _guardRows.Clear();
            _guardPendingIds.Clear();
            _frames = _records = _supportEvents = _freeEvents = 0;
            _mixedEvents = _incompleteEvents = _neutralEvents = 0;
            _eventRowsWritten = 0;
            _supportConfirmedTransitions = _freeConfirmedTransitions = 0;
            _conflicts = _overflowRecords = 0;
            _readbackErrors = _writeErrors = 0;
            _guardDecisionRowsWritten = 0;
            _guardAllowTransitions = 0;
            _guardRecentSupportVetoTransitions = 0;
            _guardConflictVetoTransitions = 0;
            _guardPendingAtSeal = 0;
            _guardGenerationResetCancelled = 0;
            _active = true;
        }

        internal void RecordReadbackError()
        {
            if (!_active) return;
            _readbackErrors++;
        }

        internal void Record(int gunGelFrame, int platformFrame,
            NativeArray<uint4> rows, float angularSpeed, float linearSpeed,
            float motionQuality)
        {
            if (!_active || rows.Length < HeaderRows) return;
            try
            {
                uint4 header0 = rows[0];
                uint4 header1 = rows[1];
                uint4 header2 = rows[2];
                int written = Mathf.Min((int)header0.x,
                    Mathf.Max(0, (rows.Length - HeaderRows) / RecordRows));
                _frames++;
                _overflowRecords += (int)header1.w;
                _frameRows.Append(gunGelFrame).Append(',').Append(platformFrame).Append(',')
                    .Append(Time.frameCount).Append(',').Append(header0.y).Append(',')
                    .Append(header0.z).Append(',').Append(header0.w).Append(',')
                    .Append(header1.x).Append(',').Append(header1.y).Append(',')
                    .Append(header1.z).Append(',').Append(header2.x).Append(',')
                    .Append(header2.y).Append(',').Append(header2.z).Append(',')
                    .Append(written).Append(',').Append(header1.w).Append(',')
                    .Append(F(angularSpeed)).Append(',')
                    .Append(F(linearSpeed)).Append(',').Append(F(motionQuality))
                    .Append(",shadow_only\n");

                for (int record = 0; record < written; record++)
                    ConsumeRecord(gunGelFrame, platformFrame, rows,
                        HeaderRows + record * RecordRows);
                EvaluatePendingGuards(gunGelFrame);
                FlushPending(false);
            }
            catch (Exception e)
            {
                _writeErrors++;
                Logger.Warning("候选足迹影子账本记录失败：" + e.Message);
            }
        }

        private void ConsumeRecord(int gunGelFrame, int platformFrame,
            NativeArray<uint4> rows, int offset)
        {
            uint4 identity = rows[offset];
            uint4 centerVariance = rows[offset + 1];
            uint4 normalRange = rows[offset + 2];
            uint4 cameraView = rows[offset + 3];
            uint4 gapStats = rows[offset + 4];
            uint4 sampleCounts = rows[offset + 5];
            uint4 otherCounts = rows[offset + 6];
            uint4 riskCounts = rows[offset + 7];
            uint4 rawPostStats = rows[offset + 8];
            uint stableId = identity.x;
            if (stableId == 0u) return;
            int classification = (int)identity.w;
            Vector3 center = V(centerVariance);
            Vector3 normal = V(normalRange).normalized;
            Vector3 camera = V(cameraView);
            Vector3 direction = (center - camera).normalized;
            float variance = math.asfloat(centerVariance.w);
            float candidateRange = math.asfloat(normalRange.w);
            float viewRadius = math.asfloat(cameraView.w);
            float minGap = math.asfloat(gapStats.x);
            float maxGap = math.asfloat(gapStats.y);
            float meanGap = math.asfloat(gapStats.z);
            float incidence = math.asfloat(gapStats.w);
            float maxRawPostDifference = math.asfloat(rawPostStats.x);

            if (!_candidates.TryGetValue(stableId, out CandidateState state))
            {
                state = new CandidateState
                {
                    StableId = stableId,
                    CandidateIndex = identity.y,
                    Center = center,
                    Normal = normal,
                    FirstFrame = gunGelFrame,
                    LastFrame = gunGelFrame
                };
                _candidates.Add(stableId, state);
            }
            else if (Vector3.Distance(state.Center, center) > GeometryResetDistance ||
                     UnsignedNormalAngle(state.Normal, normal) > GeometryResetAngleDeg)
            {
                if (state.GuardPending)
                {
                    FinalizeGuardDecision(state,
                        GuardDecisionGenerationResetCancelled, gunGelFrame);
                    _guardPendingIds.Remove(stableId);
                }
                state.Generation++;
                state.Center = center;
                state.Normal = normal;
                state.FirstFrame = gunGelFrame;
                state.SupportEvents = 0;
                state.FreeEvents = 0;
                state.MixedEvents = 0;
                state.IncompleteEvents = 0;
                state.NeutralEvents = 0;
                state.Support.Clear();
                state.Free.Clear();
                state.SupportConfirmed = false;
                state.FreeConfirmed = false;
                state.Conflict = false;
                state.LastSupportFrame = -1;
                state.GuardPending = false;
                state.GuardFreeConfirmedFrame = -1;
                state.GuardDueFrame = -1;
                state.GuardLastSupportFrame = -1;
                state.GuardDecision = GuardDecisionNone;
                state.GuardDecisionFrame = -1;
            }
            state.CandidateIndex = identity.y;
            state.LastFrame = gunGelFrame;

            List<Witness> witnesses = null;
            bool independent = false;
            if (classification == 1)
            {
                state.LastSupportFrame = gunGelFrame;
                if (state.GuardPending && gunGelFrame <= state.GuardDueFrame)
                    state.GuardLastSupportFrame = gunGelFrame;
                witnesses = state.Support;
                independent = TryAddIndependentWitness(witnesses,
                    gunGelFrame, camera, direction);
                state.SupportEvents++;
                _supportEvents++;
            }
            else if (classification == 2)
            {
                witnesses = state.Free;
                independent = TryAddIndependentWitness(witnesses,
                    gunGelFrame, camera, direction);
                state.FreeEvents++;
                _freeEvents++;
            }
            else if (classification == 3)
            {
                state.MixedEvents++;
                _mixedEvents++;
            }
            else if (classification == 4)
            {
                state.IncompleteEvents++;
                _incompleteEvents++;
            }
            else
            {
                state.NeutralEvents++;
                _neutralEvents++;
            }

            bool wasSupportConfirmed = state.SupportConfirmed;
            bool wasFreeConfirmed = state.FreeConfirmed;
            bool wasConflict = state.Conflict;
            state.SupportConfirmed = state.Support.Count >= 2;
            state.FreeConfirmed = state.Free.Count >= 2;
            state.Conflict = state.SupportConfirmed && state.FreeConfirmed;
            if (!wasSupportConfirmed && state.SupportConfirmed)
                _supportConfirmedTransitions++;
            if (!wasFreeConfirmed && state.FreeConfirmed)
            {
                _freeConfirmedTransitions++;
                BeginGuardedDecision(state, gunGelFrame);
            }
            if (!wasConflict && state.Conflict)
                _conflicts++;
            _records++;

            // 常规支持会非常密集；完整数量已在 frames/candidates 中保留。逐事件表
            // 只保存全部自由票、独立支持和对已受自由质疑候选的支持，避免巨型包。
            bool writeEvent = classification == 2 ||
                              (classification == 1 &&
                               (independent || state.FreeEvents > 0));
            if (!writeEvent) return;
            _eventRowsWritten++;
            _eventRows.Append(gunGelFrame).Append(',').Append(platformFrame).Append(',')
                .Append(stableId).Append(',').Append(identity.y).Append(',')
                .Append(state.Generation).Append(',')
                .Append(classification == 2 ? "free" : "support").Append(',')
                .Append(independent ? 1 : 0).Append(',')
                .Append(witnesses != null ? witnesses.Count : 0).Append(',')
                .Append(state.SupportConfirmed ? 1 : 0).Append(',')
                .Append(state.FreeConfirmed ? 1 : 0).Append(',')
                .Append(state.Conflict ? 1 : 0).Append(',')
                .Append(F(center.x)).Append(',').Append(F(center.y)).Append(',')
                .Append(F(center.z)).Append(',').Append(F(variance)).Append(',')
                .Append(F(normal.x)).Append(',')
                .Append(F(normal.y)).Append(',').Append(F(normal.z)).Append(',')
                .Append(F(camera.x)).Append(',').Append(F(camera.y)).Append(',')
                .Append(F(camera.z)).Append(',').Append(F(candidateRange)).Append(',')
                .Append(F(minGap)).Append(',').Append(F(maxGap)).Append(',')
                .Append(F(meanGap)).Append(',').Append(F(viewRadius)).Append(',')
                .Append(F(incidence)).Append(',').Append(sampleCounts.x).Append(',')
                .Append(sampleCounts.y).Append(',').Append(sampleCounts.z).Append(',')
                .Append(sampleCounts.w).Append(',').Append(otherCounts.x).Append(',')
                .Append(otherCounts.y).Append(',').Append(otherCounts.z).Append(',')
                .Append(otherCounts.w).Append(',').Append(riskCounts.x).Append(',')
                .Append(riskCounts.y).Append(',').Append(riskCounts.z).Append(',')
                .Append(riskCounts.w).Append(',').Append(F(maxRawPostDifference)).Append(',')
                .Append(identity.z).Append(",shadow_only\n");
        }

        private void BeginGuardedDecision(CandidateState state, int gunGelFrame)
        {
            if (state.GuardPending || state.GuardDecision != GuardDecisionNone)
                return;
            state.GuardPending = true;
            state.GuardFreeConfirmedFrame = gunGelFrame;
            state.GuardDueFrame = gunGelFrame + GuardSettleFrames;
            state.GuardLastSupportFrame =
                state.LastSupportFrame >= gunGelFrame - GuardSupportWindowFrames
                    ? state.LastSupportFrame
                    : -1;
            _guardPendingIds.Add(state.StableId);
        }

        private void EvaluatePendingGuards(int gunGelFrame)
        {
            for (int i = _guardPendingIds.Count - 1; i >= 0; i--)
            {
                uint stableId = _guardPendingIds[i];
                if (!_candidates.TryGetValue(stableId, out CandidateState state) ||
                    !state.GuardPending)
                {
                    _guardPendingIds.RemoveAt(i);
                    continue;
                }
                if (gunGelFrame < state.GuardDueFrame) continue;

                int decision;
                if (state.Conflict)
                {
                    decision = GuardDecisionConflictVeto;
                }
                else if (state.GuardLastSupportFrame >=
                         state.GuardFreeConfirmedFrame - GuardSupportWindowFrames &&
                         state.GuardLastSupportFrame <= state.GuardDueFrame)
                {
                    decision = GuardDecisionRecentSupportVeto;
                }
                else
                {
                    decision = GuardDecisionAllow;
                }
                FinalizeGuardDecision(state, decision, gunGelFrame);
                _guardPendingIds.RemoveAt(i);
            }
        }

        private void FinalizeGuardDecision(CandidateState state, int decision,
            int decisionFrame)
        {
            if (!state.GuardPending) return;
            state.GuardPending = false;
            state.GuardDecision = decision;
            state.GuardDecisionFrame = decisionFrame;
            _guardDecisionRowsWritten++;
            switch (decision)
            {
                case GuardDecisionAllow:
                    _guardAllowTransitions++;
                    break;
                case GuardDecisionRecentSupportVeto:
                    _guardRecentSupportVetoTransitions++;
                    break;
                case GuardDecisionConflictVeto:
                    _guardConflictVetoTransitions++;
                    break;
                case GuardDecisionPendingAtSeal:
                    _guardPendingAtSeal++;
                    break;
                case GuardDecisionGenerationResetCancelled:
                    _guardGenerationResetCancelled++;
                    break;
            }

            _guardRows.Append(state.StableId).Append(',')
                .Append(state.CandidateIndex).Append(',').Append(state.Generation)
                .Append(',').Append(state.GuardFreeConfirmedFrame).Append(',')
                .Append(state.GuardDueFrame).Append(',').Append(decisionFrame).Append(',')
                .Append(state.GuardLastSupportFrame).Append(',');
            if (state.GuardLastSupportFrame >= 0)
                _guardRows.Append(decisionFrame -
                                  state.GuardLastSupportFrame);
            _guardRows.Append(',').Append(GuardSupportWindowFrames).Append(',')
                .Append(GuardSettleFrames).Append(',')
                .Append(GuardDecisionName(decision)).Append(",shadow_only\n");
        }

        private static string GuardDecisionName(int decision)
        {
            switch (decision)
            {
                case GuardDecisionAllow: return "would_allow";
                case GuardDecisionRecentSupportVeto: return "recent_support_veto";
                case GuardDecisionConflictVeto: return "conflict_veto";
                case GuardDecisionPendingAtSeal: return "pending_at_seal";
                case GuardDecisionGenerationResetCancelled:
                    return "generation_reset_cancelled";
                default: return "none";
            }
        }

        private static bool TryAddIndependentWitness(List<Witness> witnesses,
            int frame, Vector3 camera, Vector3 direction)
        {
            if (witnesses.Count == 0)
            {
                witnesses.Add(new Witness { Frame = frame, Camera = camera,
                    Direction = direction });
                return true;
            }
            bool independentOfAny = false;
            for (int i = 0; i < witnesses.Count; i++)
            {
                Witness previous = witnesses[i];
                IndependentViewWitnessPolicy.ComparePair(previous.Camera,
                    previous.Direction, previous.Frame, camera, direction, frame,
                    out _, out _, out bool timeSeparated, out bool poseSeparated);
                if (timeSeparated && poseSeparated)
                    independentOfAny = true;
            }
            if (!independentOfAny || witnesses.Count >= MaxWitnessesPerClass)
                return false;
            witnesses.Add(new Witness { Frame = frame, Camera = camera,
                Direction = direction });
            return true;
        }

        private static float UnsignedNormalAngle(Vector3 a, Vector3 b)
        {
            if (a.sqrMagnitude < 1e-8f || b.sqrMagnitude < 1e-8f) return 180f;
            return Mathf.Min(Vector3.Angle(a, b), Vector3.Angle(a, -b));
        }

        private static Vector3 V(uint4 row)
        {
            return new Vector3(math.asfloat(row.x), math.asfloat(row.y),
                math.asfloat(row.z));
        }

        private static string F(float value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        private void FlushPending(bool force)
        {
            if (!force && _frameRows.Length < 16384 && _eventRows.Length < 32768 &&
                _guardRows.Length < 8192)
                return;
            try
            {
                if (_frameRows.Length > 0)
                {
                    File.AppendAllText(_framesPath, _frameRows.ToString(),
                        new UTF8Encoding(false));
                    _frameRows.Clear();
                }
                if (_eventRows.Length > 0)
                {
                    File.AppendAllText(_eventsPath, _eventRows.ToString(),
                        new UTF8Encoding(false));
                    _eventRows.Clear();
                }
                if (_guardRows.Length > 0)
                {
                    File.AppendAllText(_guardPath, _guardRows.ToString(),
                        new UTF8Encoding(false));
                    _guardRows.Clear();
                }
            }
            catch (Exception e)
            {
                _writeErrors++;
                Logger.Warning("候选足迹影子账本写盘失败：" + e.Message);
            }
        }

        internal void End()
        {
            if (!_active) return;
            var pendingAtSeal = new List<uint>(_guardPendingIds);
            pendingAtSeal.Sort();
            foreach (uint stableId in pendingAtSeal)
            {
                if (_candidates.TryGetValue(stableId, out CandidateState state) &&
                    state.GuardPending)
                    FinalizeGuardDecision(state, GuardDecisionPendingAtSeal,
                        state.LastFrame);
            }
            _guardPendingIds.Clear();
            FlushPending(true);
            try
            {
                var candidates = new StringBuilder(Mathf.Max(1024,
                    _candidates.Count * 96));
                candidates.Append("stableId,candidateIndex,generation,firstFrame,lastFrame,supportEvents,independentSupport,supportConfirmed,freeEvents,independentFree,freeConfirmed,mixedEvents,incompleteEvents,neutralEvents,conflict,lastSupportFrame,guardFreeConfirmedFrame,guardDueFrame,guardDecisionFrame,guardDecision,centerX,centerY,centerZ,normalX,normalY,normalZ,status\n");
                var stableIds = new List<uint>(_candidates.Keys);
                stableIds.Sort();
                foreach (uint stableId in stableIds)
                {
                    CandidateState state = _candidates[stableId];
                    candidates.Append(state.StableId).Append(',')
                        .Append(state.CandidateIndex).Append(',').Append(state.Generation)
                        .Append(',').Append(state.FirstFrame).Append(',').Append(state.LastFrame)
                        .Append(',').Append(state.SupportEvents).Append(',')
                        .Append(state.Support.Count).Append(',')
                        .Append(state.SupportConfirmed ? 1 : 0).Append(',')
                        .Append(state.FreeEvents).Append(',').Append(state.Free.Count)
                        .Append(',').Append(state.FreeConfirmed ? 1 : 0).Append(',')
                        .Append(state.MixedEvents).Append(',')
                        .Append(state.IncompleteEvents).Append(',')
                        .Append(state.NeutralEvents).Append(',')
                        .Append(state.Conflict ? 1 : 0).Append(',')
                        .Append(state.LastSupportFrame).Append(',')
                        .Append(state.GuardFreeConfirmedFrame).Append(',')
                        .Append(state.GuardDueFrame).Append(',')
                        .Append(state.GuardDecisionFrame).Append(',')
                        .Append(GuardDecisionName(state.GuardDecision)).Append(',')
                        .Append(F(state.Center.x)).Append(',').Append(F(state.Center.y))
                        .Append(',').Append(F(state.Center.z)).Append(',')
                        .Append(F(state.Normal.x)).Append(',').Append(F(state.Normal.y))
                        .Append(',').Append(F(state.Normal.z)).Append(",shadow_only\n");
                }
                File.WriteAllText(Path.Combine(_directory, "candidates.csv"),
                    candidates.ToString(), new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(_directory, "summary.json"),
                    "{\n" +
                    "  \"authority\": \"shadow_only_zero_production_authority\",\n" +
                    "  \"frames\": " + _frames + ",\n" +
                    "  \"records\": " + _records + ",\n" +
                    "  \"supportEvents\": " + _supportEvents + ",\n" +
                    "  \"freeEvents\": " + _freeEvents + ",\n" +
                    "  \"mixedEvents\": " + _mixedEvents + ",\n" +
                    "  \"incompleteEvents\": " + _incompleteEvents + ",\n" +
                    "  \"neutralEvents\": " + _neutralEvents + ",\n" +
                    "  \"eventRowsWritten\": " + _eventRowsWritten + ",\n" +
                    "  \"supportConfirmedTransitions\": " + _supportConfirmedTransitions + ",\n" +
                    "  \"freeConfirmedTransitions\": " + _freeConfirmedTransitions + ",\n" +
                    "  \"conflicts\": " + _conflicts + ",\n" +
                    "  \"guardSupportWindowFrames\": " + GuardSupportWindowFrames + ",\n" +
                    "  \"guardSettleFrames\": " + GuardSettleFrames + ",\n" +
                    "  \"guardDecisionRowsWritten\": " + _guardDecisionRowsWritten + ",\n" +
                    "  \"guardAllowTransitions\": " + _guardAllowTransitions + ",\n" +
                    "  \"guardRecentSupportVetoTransitions\": " + _guardRecentSupportVetoTransitions + ",\n" +
                    "  \"guardConflictVetoTransitions\": " + _guardConflictVetoTransitions + ",\n" +
                    "  \"guardPendingAtSeal\": " + _guardPendingAtSeal + ",\n" +
                    "  \"guardGenerationResetCancelled\": " + _guardGenerationResetCancelled + ",\n" +
                    "  \"overflowRecords\": " + _overflowRecords + ",\n" +
                    "  \"readbackErrors\": " + _readbackErrors + ",\n" +
                    "  \"writeErrors\": " + _writeErrors + "\n" +
                    "}\n", new UTF8Encoding(false));
            }
            catch (Exception e)
            {
                _writeErrors++;
                Logger.Warning("候选足迹影子账本封口失败：" + e.Message);
            }
            _active = false;
        }

        public void Dispose()
        {
            End();
        }
    }
}

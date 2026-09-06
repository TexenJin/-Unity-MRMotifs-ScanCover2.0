using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 多套“稳定面何时毕业”契约的只读赛马。它只旁听虚拟探针已经判定为
    /// safe support / reliable free-space 的事件并写诊断文件；任何结果都没有
    /// GunGel、TSDF、纸皮或网格生产权限。
    /// </summary>
    internal sealed class GraduationContractShadowRace : IDisposable
    {
        private const string Authority = "shadow_only_no_production_authority";
        private const float IndependentBaselineMetres =
            IndependentViewWitnessPolicy.BaselineMetres;
        private const float IndependentAngleDeg =
            IndependentViewWitnessPolicy.AngleDeg;
        private const int IndependentFrameGap =
            IndependentViewWitnessPolicy.FrameGap;
        private const int WitnessFlushChars = 262144;

        // 这些是并行实验臂，不是生产阈值。revisit_* 先使用两个独立支持证词
        // 之间的真实经过时间作为“离开再回来”的代理，同时把连续量全部导出，
        // 供离线审计；不能据此声称头显确实离开过目标区域。
        private static readonly Contract[] Contracts =
        {
            new Contract("baseline_2view", 2, 0.0),
            new Contract("support_3view", 3, 0.0),
            new Contract("support_4view", 4, 0.0),
            new Contract("revisit_1s", 2, 1.0),
            new Contract("revisit_2s", 2, 2.0),
            new Contract("revisit_4s", 2, 4.0),
            new Contract("support_3view_revisit_2s", 3, 2.0)
        };

        private readonly Dictionary<TrialKey, Trial> _trials =
            new Dictionary<TrialKey, Trial>(8192);
        private string _directory = string.Empty;
        private bool _active;
        private int _candidateRows;
        private int _summaryRows;
        private int _witnessEventRows;
        private int _writeErrors;
        private string _witnessEventPath = string.Empty;
        private readonly StringBuilder _pendingWitnessEvents =
            new StringBuilder(WitnessFlushChars);

        internal int CandidateRows => _candidateRows;
        internal int SummaryRows => _summaryRows;
        internal int WitnessEventRows => _witnessEventRows;
        internal int WriteErrors => _writeErrors;

        internal void Begin(string probeDirectory)
        {
            End(-1);
            _trials.Clear();
            _candidateRows = 0;
            _summaryRows = 0;
            _witnessEventRows = 0;
            _writeErrors = 0;
            _pendingWitnessEvents.Clear();
            _directory = probeDirectory ?? string.Empty;
            if (string.IsNullOrEmpty(_directory)) return;
            Directory.CreateDirectory(_directory);
            try
            {
                _witnessEventPath = Path.Combine(_directory,
                    "graduation_race_witness_events.csv");
                File.WriteAllText(_witnessEventPath,
                    "frame,timeSeconds,cellX,cellY,cellZ,axis,generation," +
                    "productionStream,cameraX,cameraY,cameraZ,centerX,centerY," +
                    "centerZ,residualMm,angleDeg,viewRadius,motionQuality," +
                    "productionIndependent,productionDecision," +
                    "productionCountBefore,productionCountAfter," +
                    "productionCapacity,productionNearestBaselineMm," +
                    "productionNearestSpreadDeg,raceIndependent,raceDecision," +
                    "raceCountBefore,raceCountAfter,raceCapacity," +
                    "nearestRaceBaselineMm,nearestRaceSpreadDeg,authority\n",
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(_directory,
                        "graduation_race_schema.json"), BuildSchemaJson(),
                    new UTF8Encoding(false));
                _active = true;
            }
            catch
            {
                _writeErrors++;
                _active = false;
            }
        }

        internal void ObserveSafeSupport(int x, int y, int z, int axis,
            int generation, int frame, double timeSeconds, Vector3 camera,
            Vector3 target, float residualMetres, float angleDeg,
            float viewRadius, float motionQuality, bool independent,
            string productionStream,
            IndependentWitnessDecision productionDecision,
            int productionCountBefore, int productionCountAfter,
            int productionCapacity, float productionNearestBaseline,
            float productionNearestSpread)
        {
            if (!_active || generation <= 0) return;
            try
            {
                Trial trial = GetOrCreate(x, y, z, axis, generation, frame,
                    timeSeconds, target);
                if (trial.LastSafeSupportFrame != frame)
                {
                    trial.SafeSupportFrames++;
                    trial.LastSafeSupportFrame = frame;
                }
                trial.LastSupportTimeSeconds = Math.Max(
                    trial.LastSupportTimeSeconds, timeSeconds);
                trial.MaxResidualMetres = Mathf.Max(trial.MaxResidualMetres,
                    residualMetres);
                trial.MaxAngleDeg = Mathf.Max(trial.MaxAngleDeg, angleDeg);
                trial.MaxViewRadius = Mathf.Max(trial.MaxViewRadius, viewRadius);
                trial.MinMotionQuality = Mathf.Min(trial.MinMotionQuality,
                    motionQuality);
                if (independent) trial.AdjudicatorIndependentEvents++;
                int raceCountBefore = trial.Witnesses == null
                    ? 0 : trial.Witnesses.Count;
                bool raceIndependent = TryAddIndependentWitness(trial, camera,
                    frame, out IndependentWitnessDecision raceDecision,
                    out float nearestRaceBaseline,
                    out float nearestRaceSpread);
                int raceCountAfter = trial.Witnesses == null
                    ? 0 : trial.Witnesses.Count;
                if (raceIndependent != independent)
                    trial.IndependenceDisagreements++;
                AppendWitnessEvent(frame, timeSeconds, x, y, z, axis,
                    generation, productionStream, camera, target,
                    residualMetres, angleDeg, viewRadius, motionQuality,
                    independent, productionDecision, productionCountBefore,
                    productionCountAfter, productionCapacity, raceIndependent,
                    productionNearestBaseline, productionNearestSpread,
                    raceDecision, raceCountBefore, raceCountAfter,
                    nearestRaceBaseline, nearestRaceSpread);
                if (!raceIndependent) return;

                if (trial.IndependentSupportViews > 0)
                {
                    int gapFrames = Mathf.Max(0,
                        frame - trial.LastIndependentFrame);
                    double gapSeconds = Math.Max(0.0,
                        timeSeconds - trial.LastIndependentTimeSeconds);
                    float baseline = Vector3.Distance(
                        trial.LastIndependentCamera, camera);
                    float spread = Vector3.Angle(
                        SafeDirection(trial.LastIndependentCamera - trial.Center),
                        SafeDirection(camera - trial.Center));
                    trial.MaxIndependentGapFrames = Mathf.Max(
                        trial.MaxIndependentGapFrames, gapFrames);
                    trial.MaxIndependentGapSeconds = Math.Max(
                        trial.MaxIndependentGapSeconds, gapSeconds);
                    trial.MaxIndependentBaselineMetres = Mathf.Max(
                        trial.MaxIndependentBaselineMetres, baseline);
                    trial.MaxIndependentSpreadDeg = Mathf.Max(
                        trial.MaxIndependentSpreadDeg, spread);
                }

                trial.IndependentSupportViews++;
                trial.LastIndependentFrame = frame;
                trial.LastIndependentTimeSeconds = timeSeconds;
                trial.LastIndependentCamera = camera;
                for (int i = 0; i < Contracts.Length; i++)
                {
                    LaneResult lane = trial.Lanes[i];
                    if (lane.Graduated) continue;
                    Contract contract = Contracts[i];
                    if (trial.IndependentSupportViews < contract.RequiredViews)
                        continue;
                    if (contract.RequiredRevisitSeconds > 0.0 &&
                        trial.MaxIndependentGapSeconds + 1e-6 <
                        contract.RequiredRevisitSeconds)
                        continue;
                    lane.Graduated = true;
                    lane.GraduatedFrame = frame;
                    lane.GraduatedTimeSeconds = timeSeconds;
                    lane.ViewsAtGraduation = trial.IndependentSupportViews;
                    lane.MaxGapSecondsAtGraduation =
                        trial.MaxIndependentGapSeconds;
                }
            }
            catch { _writeErrors++; }
        }

        internal void BindFingerprint(int x, int y, int z, int axis,
            int generation, int fingerprintId, int acceptedFrame)
        {
            if (!_active || generation <= 0) return;
            try
            {
                Trial trial = GetOrCreate(x, y, z, axis, generation,
                    acceptedFrame, 0.0, Vector3.zero);
                trial.FingerprintId = fingerprintId;
                trial.BaseAcceptedFrame = acceptedFrame;
            }
            catch { _writeErrors++; }
        }

        internal void ObserveReliableChallenge(int x, int y, int z, int axis,
            int generation, int frame, float gapMetres, bool independent)
        {
            if (!_active || generation <= 0) return;
            try
            {
                TrialKey key = new TrialKey(x, y, z, axis, generation);
                if (!_trials.TryGetValue(key, out Trial trial)) return;
                if (trial.LastChallengeFrame != frame)
                {
                    trial.ReliableChallengeFrames++;
                    trial.LastChallengeFrame = frame;
                }
                if (independent) trial.IndependentChallengeViews++;
                trial.MaxFreeGapMetres = Mathf.Max(trial.MaxFreeGapMetres,
                    gapMetres);
            }
            catch { _writeErrors++; }
        }

        internal void ObserveOutcome(int x, int y, int z, int axis,
            int generation, int rejectFrame, string outcome,
            bool observationWindowComplete)
        {
            if (!_active || generation <= 0) return;
            try
            {
                TrialKey key = new TrialKey(x, y, z, axis, generation);
                if (!_trials.TryGetValue(key, out Trial trial)) return;
                trial.BaseRejectFrame = rejectFrame;
                trial.FinalOutcome = string.IsNullOrEmpty(outcome)
                    ? "unknown_unlabelled" : outcome;
                trial.OutcomeWindowComplete = observationWindowComplete;
            }
            catch { _writeErrors++; }
        }

        internal void End(int lastFrame)
        {
            if (!_active) return;
            try
            {
                FlushWitnessEvents();
                WriteCandidates(lastFrame);
                WriteSummary();
            }
            catch
            {
                _writeErrors++;
            }
            _active = false;
        }

        public void Dispose()
        {
            End(-1);
            _trials.Clear();
        }

        private Trial GetOrCreate(int x, int y, int z, int axis,
            int generation, int frame, double timeSeconds, Vector3 center)
        {
            var key = new TrialKey(x, y, z, axis, generation);
            if (_trials.TryGetValue(key, out Trial trial))
            {
                if (center.sqrMagnitude > 1e-12f) trial.Center = center;
                return trial;
            }
            trial = new Trial
            {
                Key = key,
                Center = center,
                FirstSupportFrame = frame,
                FirstSupportTimeSeconds = timeSeconds,
                LastSupportTimeSeconds = timeSeconds,
                MinMotionQuality = 1f,
                Lanes = CreateLaneResults()
            };
            _trials.Add(key, trial);
            return trial;
        }

        private void WriteCandidates(int lastFrame)
        {
            var rows = new List<Trial>(_trials.Values);
            rows.Sort((a, b) => a.Key.CompareTo(b.Key));
            var sb = new StringBuilder(Mathf.Max(4096,
                rows.Count * Contracts.Length * 240));
            sb.Append("cellX,cellY,cellZ,axis,candidateGeneration,fingerprintId,")
                .Append("contract,requiredIndependentViews,requiredRevisitSeconds,")
                .Append("graduated,graduatedFrame,graduatedTimeSeconds,")
                .Append("delayFromFirstSupportFrames,delayFromFirstSupportSeconds,")
                .Append("delayFromBaselineFrames,viewsAtGraduation,")
                .Append("revisitGapAtGraduationSeconds,")
                .Append("firstSupportFrame,lastSupportFrame,")
                .Append("firstSupportTimeSeconds,lastSupportTimeSeconds,")
                .Append("safeSupportFrames,independentSupportViews,")
                .Append("adjudicatorIndependentEvents,independenceDisagreements,")
                .Append("maxIndependentGapFrames,maxIndependentGapSeconds,")
                .Append("maxIndependentBaselineMm,maxIndependentSpreadDeg,")
                .Append("maxResidualMm,maxAngleDeg,maxViewRadius,minMotionQuality,")
                .Append("reliableChallengeFrames,independentChallengeViews,")
                .Append("maxFreeGapMm,baseAcceptedFrame,baseRejectFrame,")
                .Append("finalOutcome,outcomeWindowComplete,lastSessionFrame,")
                .Append("labelClass,authority\n");

            foreach (Trial trial in rows)
            {
                int baselineFrame = trial.Lanes[0].Graduated
                    ? trial.Lanes[0].GraduatedFrame : -1;
                for (int i = 0; i < Contracts.Length; i++)
                {
                    Contract contract = Contracts[i];
                    LaneResult lane = trial.Lanes[i];
                    int delayFrames = lane.Graduated
                        ? lane.GraduatedFrame - trial.FirstSupportFrame : -1;
                    double delaySeconds = lane.Graduated
                        ? lane.GraduatedTimeSeconds - trial.FirstSupportTimeSeconds
                        : -1.0;
                    int baselineDelay = lane.Graduated && baselineFrame >= 0
                        ? lane.GraduatedFrame - baselineFrame : -1;
                    sb.Append(I(trial.Key.X)).Append(',')
                        .Append(I(trial.Key.Y)).Append(',')
                        .Append(I(trial.Key.Z)).Append(',')
                        .Append(I(trial.Key.Axis)).Append(',')
                        .Append(I(trial.Key.Generation)).Append(',')
                        .Append(I(trial.FingerprintId)).Append(',')
                        .Append(contract.Name).Append(',')
                        .Append(I(contract.RequiredViews)).Append(',')
                        .Append(D(contract.RequiredRevisitSeconds)).Append(',')
                        .Append(lane.Graduated ? "1" : "0").Append(',')
                        .Append(I(lane.GraduatedFrame)).Append(',')
                        .Append(lane.Graduated ? D(lane.GraduatedTimeSeconds) : "")
                        .Append(',').Append(I(delayFrames)).Append(',')
                        .Append(lane.Graduated ? D(delaySeconds) : "").Append(',')
                        .Append(I(baselineDelay)).Append(',')
                        .Append(I(lane.ViewsAtGraduation)).Append(',')
                        .Append(D(lane.MaxGapSecondsAtGraduation)).Append(',')
                        .Append(I(trial.FirstSupportFrame)).Append(',')
                        .Append(I(trial.LastSafeSupportFrame)).Append(',')
                        .Append(D(trial.FirstSupportTimeSeconds)).Append(',')
                        .Append(D(trial.LastSupportTimeSeconds)).Append(',')
                        .Append(I(trial.SafeSupportFrames)).Append(',')
                        .Append(I(trial.IndependentSupportViews)).Append(',')
                        .Append(I(trial.AdjudicatorIndependentEvents)).Append(',')
                        .Append(I(trial.IndependenceDisagreements)).Append(',')
                        .Append(I(trial.MaxIndependentGapFrames)).Append(',')
                        .Append(D(trial.MaxIndependentGapSeconds)).Append(',')
                        .Append(F(trial.MaxIndependentBaselineMetres * 1000f))
                        .Append(',').Append(F(trial.MaxIndependentSpreadDeg))
                        .Append(',').Append(F(trial.MaxResidualMetres * 1000f))
                        .Append(',').Append(F(trial.MaxAngleDeg)).Append(',')
                        .Append(F(trial.MaxViewRadius)).Append(',')
                        .Append(F(trial.MinMotionQuality)).Append(',')
                        .Append(I(trial.ReliableChallengeFrames)).Append(',')
                        .Append(I(trial.IndependentChallengeViews)).Append(',')
                        .Append(F(trial.MaxFreeGapMetres * 1000f)).Append(',')
                        .Append(I(trial.BaseAcceptedFrame)).Append(',')
                        .Append(I(trial.BaseRejectFrame)).Append(',')
                        .Append(trial.FinalOutcome).Append(',')
                        .Append(trial.OutcomeWindowComplete ? "1" : "0")
                        .Append(',').Append(I(lastFrame)).Append(',')
                        .Append(LabelClass(trial, lane)).Append(',')
                        .Append(Authority).Append('\n');
                    _candidateRows++;
                }
            }
            File.WriteAllText(Path.Combine(_directory,
                    "graduation_race_candidates.csv"), sb.ToString(),
                new UTF8Encoding(false));
        }

        private void WriteSummary()
        {
            var sb = new StringBuilder(4096);
            sb.Append("contract,requiredIndependentViews,requiredRevisitSeconds,")
                .Append("trials,graduated,notGraduated,knownFalseTrials,")
                .Append("knownFalseGraduated,knownFalsePrevented,")
                .Append("supportReturnedTrials,supportReturnedGraduated,")
                .Append("unknownTrials,unlabelledTrials,meanDelayFromBaselineFrames,")
                .Append("meanDelayFromBaselineSeconds,authority\n");
            for (int i = 0; i < Contracts.Length; i++)
            {
                Contract contract = Contracts[i];
                int trials = 0, graduated = 0, knownFalse = 0;
                int knownFalseGraduated = 0, returned = 0;
                int returnedGraduated = 0, unknown = 0, unlabelled = 0;
                long delaySum = 0;
                double delaySecondsSum = 0.0;
                int delayCount = 0;
                foreach (Trial trial in _trials.Values)
                {
                    trials++;
                    LaneResult lane = trial.Lanes[i];
                    if (lane.Graduated) graduated++;
                    if (trial.FinalOutcome == "confirmed_false_surface")
                    {
                        knownFalse++;
                        if (lane.Graduated) knownFalseGraduated++;
                    }
                    else if (trial.FinalOutcome == "support_reestablished" ||
                             trial.FinalOutcome ==
                             "support_reestablished_after_conflict")
                    {
                        returned++;
                        if (lane.Graduated) returnedGraduated++;
                    }
                    else if (trial.FinalOutcome.StartsWith("unknown_",
                                 StringComparison.Ordinal))
                        unknown++;
                    else unlabelled++;

                    LaneResult baseline = trial.Lanes[0];
                    if (lane.Graduated && baseline.Graduated)
                    {
                        delaySum += lane.GraduatedFrame - baseline.GraduatedFrame;
                        delaySecondsSum += lane.GraduatedTimeSeconds -
                                           baseline.GraduatedTimeSeconds;
                        delayCount++;
                    }
                }
                double meanDelay = delayCount > 0
                    ? delaySum / (double)delayCount : 0.0;
                double meanDelaySeconds = delayCount > 0
                    ? delaySecondsSum / delayCount : 0.0;
                sb.Append(contract.Name).Append(',')
                    .Append(I(contract.RequiredViews)).Append(',')
                    .Append(D(contract.RequiredRevisitSeconds)).Append(',')
                    .Append(I(trials)).Append(',').Append(I(graduated)).Append(',')
                    .Append(I(trials - graduated)).Append(',')
                    .Append(I(knownFalse)).Append(',')
                    .Append(I(knownFalseGraduated)).Append(',')
                    .Append(I(knownFalse - knownFalseGraduated)).Append(',')
                    .Append(I(returned)).Append(',')
                    .Append(I(returnedGraduated)).Append(',')
                    .Append(I(unknown)).Append(',').Append(I(unlabelled)).Append(',')
                    .Append(D(meanDelay)).Append(',').Append(D(meanDelaySeconds))
                    .Append(',').Append(Authority).Append('\n');
                _summaryRows++;
            }
            File.WriteAllText(Path.Combine(_directory,
                    "graduation_race_summary.csv"), sb.ToString(),
                new UTF8Encoding(false));
        }

        private static string LabelClass(Trial trial, LaneResult lane)
        {
            if (trial.FinalOutcome == "confirmed_false_surface")
                return lane.Graduated
                    ? "known_false_admitted" : "known_false_not_graduated";
            if (trial.FinalOutcome == "support_reestablished" ||
                trial.FinalOutcome == "support_reestablished_after_conflict")
                return lane.Graduated
                    ? "support_returned_and_graduated"
                    : "support_returned_not_graduated";
            if (trial.FinalOutcome.StartsWith("unknown_",
                    StringComparison.Ordinal))
                return "unknown_do_not_train";
            return "unlabelled_no_reject_followup";
        }

        private static LaneResult[] CreateLaneResults()
        {
            var results = new LaneResult[Contracts.Length];
            for (int i = 0; i < results.Length; i++) results[i] = new LaneResult();
            return results;
        }

        private static bool TryAddIndependentWitness(Trial trial, Vector3 camera,
            int frame, out IndependentWitnessDecision decision,
            out float nearestBaseline, out float nearestSpread)
        {
            if (trial.Witnesses == null)
                trial.Witnesses = new List<ViewWitness>(8);
            Vector3 direction = SafeDirection(camera - trial.Center);
            bool blockedByTime = false;
            bool blockedByPose = false;
            nearestBaseline = 0f;
            nearestSpread = 0f;
            float closestScore = float.PositiveInfinity;
            for (int i = 0; i < trial.Witnesses.Count; i++)
            {
                ViewWitness witness = trial.Witnesses[i];
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
            trial.Witnesses.Add(new ViewWitness(camera, direction, frame));
            decision = IndependentWitnessDecision.Accepted;
            return true;
        }

        private void AppendWitnessEvent(int frame, double timeSeconds,
            int x, int y, int z, int axis, int generation,
            string productionStream, Vector3 camera, Vector3 center,
            float residualMetres, float angleDeg, float viewRadius,
            float motionQuality, bool productionIndependent,
            IndependentWitnessDecision productionDecision,
            int productionCountBefore, int productionCountAfter,
            int productionCapacity, bool raceIndependent,
            float productionNearestBaseline, float productionNearestSpread,
            IndependentWitnessDecision raceDecision, int raceCountBefore,
            int raceCountAfter, float nearestRaceBaseline,
            float nearestRaceSpread)
        {
            _pendingWitnessEvents.Append(I(frame)).Append(',')
                .Append(D(timeSeconds)).Append(',').Append(I(x)).Append(',')
                .Append(I(y)).Append(',').Append(I(z)).Append(',')
                .Append(I(axis)).Append(',').Append(I(generation)).Append(',')
                .Append(string.IsNullOrEmpty(productionStream)
                    ? "unknown" : productionStream).Append(',')
                .Append(F(camera.x)).Append(',').Append(F(camera.y)).Append(',')
                .Append(F(camera.z)).Append(',').Append(F(center.x)).Append(',')
                .Append(F(center.y)).Append(',').Append(F(center.z)).Append(',')
                .Append(F(residualMetres * 1000f)).Append(',')
                .Append(F(angleDeg)).Append(',').Append(F(viewRadius)).Append(',')
                .Append(F(motionQuality)).Append(',')
                .Append(productionIndependent ? "1" : "0").Append(',')
                .Append(IndependentViewWitnessPolicy.Name(productionDecision))
                .Append(',').Append(I(productionCountBefore)).Append(',')
                .Append(I(productionCountAfter)).Append(',')
                .Append(I(productionCapacity)).Append(',')
                .Append(F(productionNearestBaseline * 1000f)).Append(',')
                .Append(F(productionNearestSpread)).Append(',')
                .Append(raceIndependent ? "1" : "0").Append(',')
                .Append(IndependentViewWitnessPolicy.Name(raceDecision))
                .Append(',').Append(I(raceCountBefore)).Append(',')
                .Append(I(raceCountAfter)).Append(',')
                .Append("unbounded").Append(',')
                .Append(F(nearestRaceBaseline * 1000f)).Append(',')
                .Append(F(nearestRaceSpread)).Append(',')
                .Append(Authority).Append('\n');
            _witnessEventRows++;
            if (_pendingWitnessEvents.Length >= WitnessFlushChars)
                FlushWitnessEvents();
        }

        private void FlushWitnessEvents()
        {
            if (_pendingWitnessEvents.Length == 0 ||
                string.IsNullOrEmpty(_witnessEventPath)) return;
            File.AppendAllText(_witnessEventPath,
                _pendingWitnessEvents.ToString(), new UTF8Encoding(false));
            _pendingWitnessEvents.Clear();
        }

        private static Vector3 SafeDirection(Vector3 value)
        {
            float magnitude = value.magnitude;
            return magnitude > 1e-6f ? value / magnitude : Vector3.forward;
        }

        private static string BuildSchemaJson()
        {
            return "{\n" +
                   "  \"schema\": \"scancover.graduation_contract_shadow_race.v1\",\n" +
                   "  \"authority\": \"none; write-only diagnostics, never consumed by GunGel, TSDF, paper or mesh\",\n" +
                   "  \"question\": \"which graduation contract avoids admitting surfaces later confirmed false without needlessly delaying surfaces whose support returns\",\n" +
                   "  \"sharedInput\": \"all lanes receive the same safe-support and reliable free-space events from VirtualProbeShadowAdjudicator; the race keeps generation-local witnesses so recovery generations are not contaminated by the preceding generation\",\n" +
                   "  \"groundTruthBoundary\": \"confirmed_false_surface and support_reestablished are follow-up outcomes, not universal geometric truth; unknown and unlabelled rows must not train a rule\",\n" +
                   "  \"revisitProxy\": \"revisit lanes use elapsed real time between independent safe-support witnesses; this is an explicit proxy and does not prove that the headset left the target region\",\n" +
                   "  \"independenceMirror\": {\"baselineMetres\":0.08,\"angleDeg\":3.0,\"frameGap\":2,\"scope\":\"shared IndependentViewWitnessPolicy; production capacity and unbounded race archive remain explicitly separate\"},\n" +
                   "  \"witnessEventLog\": \"every safe-support callback records raw camera and candidate center plus production/race decisions, counts and capacity reason so offline replay does not infer from aggregate counters\",\n" +
                   "  \"antiOverfit\": \"lanes vary evidence structure and revisit duration; continuous gap, baseline, spread, residual, view-radius and motion values are exported for offline analysis instead of silently baking a room-specific optimum\",\n" +
                   "  \"contracts\": [\"baseline_2view\",\"support_3view\",\"support_4view\",\"revisit_1s\",\"revisit_2s\",\"revisit_4s\",\"support_3view_revisit_2s\"],\n" +
                   "  \"files\": {\"candidates\":\"graduation_race_candidates.csv\",\"summary\":\"graduation_race_summary.csv\",\"witnessEvents\":\"graduation_race_witness_events.csv\"}\n" +
                   "}\n";
        }

        private static string I(int value) =>
            value.ToString(CultureInfo.InvariantCulture);
        private static string F(float value) =>
            float.IsNaN(value) || float.IsInfinity(value)
                ? "" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string D(double value) =>
            double.IsNaN(value) || double.IsInfinity(value)
                ? "" : value.ToString("R", CultureInfo.InvariantCulture);

        private readonly struct Contract
        {
            internal readonly string Name;
            internal readonly int RequiredViews;
            internal readonly double RequiredRevisitSeconds;

            internal Contract(string name, int requiredViews,
                double requiredRevisitSeconds)
            {
                Name = name;
                RequiredViews = requiredViews;
                RequiredRevisitSeconds = requiredRevisitSeconds;
            }
        }

        private sealed class LaneResult
        {
            internal bool Graduated;
            internal int GraduatedFrame = -1;
            internal double GraduatedTimeSeconds;
            internal int ViewsAtGraduation;
            internal double MaxGapSecondsAtGraduation;
        }

        private sealed class Trial
        {
            internal TrialKey Key;
            internal Vector3 Center;
            internal int FingerprintId;
            internal int FirstSupportFrame;
            internal double FirstSupportTimeSeconds;
            internal double LastSupportTimeSeconds;
            internal int LastSafeSupportFrame = -1;
            internal int SafeSupportFrames;
            internal int IndependentSupportViews;
            internal int AdjudicatorIndependentEvents;
            internal int IndependenceDisagreements;
            internal List<ViewWitness> Witnesses;
            internal int LastIndependentFrame = -1;
            internal double LastIndependentTimeSeconds;
            internal Vector3 LastIndependentCamera;
            internal int MaxIndependentGapFrames;
            internal double MaxIndependentGapSeconds;
            internal float MaxIndependentBaselineMetres;
            internal float MaxIndependentSpreadDeg;
            internal float MaxResidualMetres;
            internal float MaxAngleDeg;
            internal float MaxViewRadius;
            internal float MinMotionQuality;
            internal int ReliableChallengeFrames;
            internal int LastChallengeFrame = -1;
            internal int IndependentChallengeViews;
            internal float MaxFreeGapMetres;
            internal int BaseAcceptedFrame = -1;
            internal int BaseRejectFrame = -1;
            internal string FinalOutcome = "unlabelled_no_reject_followup";
            internal bool OutcomeWindowComplete;
            internal LaneResult[] Lanes;
        }

        private readonly struct ViewWitness
        {
            internal readonly Vector3 Camera;
            internal readonly Vector3 Direction;
            internal readonly int Frame;

            internal ViewWitness(Vector3 camera, Vector3 direction, int frame)
            {
                Camera = camera;
                Direction = direction;
                Frame = frame;
            }
        }

        private readonly struct TrialKey : IEquatable<TrialKey>,
            IComparable<TrialKey>
        {
            internal readonly int X;
            internal readonly int Y;
            internal readonly int Z;
            internal readonly int Axis;
            internal readonly int Generation;

            internal TrialKey(int x, int y, int z, int axis, int generation)
            {
                X = x;
                Y = y;
                Z = z;
                Axis = axis;
                Generation = generation;
            }

            public bool Equals(TrialKey other) => X == other.X && Y == other.Y &&
                Z == other.Z && Axis == other.Axis && Generation == other.Generation;
            public override bool Equals(object obj) =>
                obj is TrialKey other && Equals(other);
            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = X;
                    hash = hash * 397 ^ Y;
                    hash = hash * 397 ^ Z;
                    hash = hash * 397 ^ Axis;
                    return hash * 397 ^ Generation;
                }
            }

            public int CompareTo(TrialKey other)
            {
                int value = X.CompareTo(other.X);
                if (value != 0) return value;
                value = Y.CompareTo(other.Y);
                if (value != 0) return value;
                value = Z.CompareTo(other.Z);
                if (value != 0) return value;
                value = Axis.CompareTo(other.Axis);
                return value != 0 ? value : Generation.CompareTo(other.Generation);
            }
        }
    }
}

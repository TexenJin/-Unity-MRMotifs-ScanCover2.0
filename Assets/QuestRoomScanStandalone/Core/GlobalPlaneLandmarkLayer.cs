using System;
using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Short-lived local plane evidence entering the global landmark layer.
    /// This DTO deliberately contains no TSDF or renderer reference: callers
    /// may remove the complete landmark layer without touching reconstruction.
    /// </summary>
    internal sealed class GlobalPlaneObservation
    {
        internal int SourceId;
        internal Vector3 Normal;
        internal float Offset;
        internal Vector3 Center;
        internal bool Horizontal;
        internal float SupportArea;
        internal float AngleSigmaDegrees;
        internal float OffsetSigmaMeters;
        internal readonly List<GlobalPlanePatch> Patches =
            new List<GlobalPlanePatch>(24);
    }

    internal struct GlobalPlanePatch
    {
        internal Vector3 Center;
        internal float HalfExtent;

        internal GlobalPlanePatch(Vector3 center, float halfExtent)
        {
            Center = center;
            HalfExtent = halfExtent;
        }
    }

    internal enum GlobalPlaneLandmarkState : byte
    {
        Tentative,
        Confirmed,
        Dormant,
        Rejected,
    }

    internal sealed class GlobalPlaneLandmarkCell
    {
        internal int Votes;
        internal int LastEpoch;
        internal float LastSeen;
    }

    /// <summary>
    /// One canonical, globally persistent plane identity. Coverage cells are
    /// evidence on the plane, not independent identities.
    /// </summary>
    internal sealed class GlobalPlaneLandmark
    {
        internal int Id;
        internal Vector3 Normal;
        internal float Offset;
        internal Vector3 U;
        internal Vector3 V;
        internal bool Horizontal;
        internal GlobalPlaneLandmarkState State;
        internal int ObservationEpochs;
        internal int LastObservationEpoch;
        internal float FirstSeen;
        internal float LastSeen;
        internal float SupportArea;
        internal float AngleVariance;
        internal float OffsetVariance;
        internal int EvidenceScore;
        internal int Contradictions;
        internal int ConsecutiveMisses;
        internal readonly Dictionary<long, GlobalPlaneLandmarkCell> Cells =
            new Dictionary<long, GlobalPlaneLandmarkCell>(128);
    }

    /// <summary>
    /// Lightweight PlanarSLAM/S-Graphs-style canonical plane landmark layer.
    /// Local observations are associated one-way into at most sixteen global
    /// identities. It owns no GPU resource and never writes reconstruction.
    /// </summary>
    internal sealed class GlobalPlaneLandmarkLayer : IDisposable
    {
        internal const float CellSizeMeters = 0.20f;
        internal const int DisplayCellVotes = 4;
        internal const int DurableCellVotes = 12;

        private const int MaxLandmarks = 16;
        private const int MaxPendingSeeds = 24;
        private const float AssociationAngleDegrees = 8f;
        private const float ConfirmedAssociationAngleDegrees = 6f;
        private const float StrongAngleDegrees = 4f;
        private const float MaximumOffsetGateMeters = 0.14f;
        private const float ConfirmedOffsetGateMeters = 0.12f;
        private const float StrongOffsetMeters = 0.06f;
        private const float MaximumCoverageGapMeters = 0.80f;
        private const float CompetitionCoverageGapMeters = 0.24f;
        private const float AmbiguousCoverageGapMeters = 0.30f;
        private const float AmbiguousPlaneDistanceMeters = 0.10f;
        private const float CornerSeparationAngleDegrees = 12f;
        private const float ObservationPatchResidualMeters = 0.08f;
        private const float MinimumCoherentPatchFraction = 0.70f;
        private const float SeedMaximumAngleSigmaDegrees = 5.5f;
        private const float SeedMaximumOffsetSigmaMeters = 0.06f;
        private const float StrictMergeAngleDegrees = 3f;
        private const float StrictMergeOffsetMeters = 0.05f;
        private const float RelaxedMergeAngleDegrees = 7f;
        private const float RelaxedMergeOffsetMeters = 0.10f;
        private const float RelaxedMergeCoverageGapMeters = 0.65f;
        private const float DormantAfterSeconds = 3.0f;
        private const float TentativeLifetimeSeconds = 5.0f;
        private const float DormantLifetimeSeconds = 30.0f;
        private const float PendingCellLifetimeSeconds = 6.0f;
        private const float WeakCellLifetimeSeconds = 18.0f;
        private const int ConfirmObservationEpochs = 6;
        private const int ConfirmCellCount = 6;
        private const int StructuralObservationEpochs = 16;
        private const int StructuralCellCount = 20;
        private const int MinimumEvidence = 0;
        private const int ConfirmEvidence = 12;
        private const int MaximumEvidence = 30;
        private const int DemoteContradictions = 5;
        private const int RejectContradictions = 9;

        private sealed class PendingSeed
        {
            internal Vector3 Normal;
            internal float Offset;
            internal Vector3 Center;
            internal bool Horizontal;
            internal float SupportArea;
            internal float AngleSigmaDegrees;
            internal float OffsetSigmaMeters;
            internal int Hits;
            internal int LastEpoch;
            internal float LastSeen;
        }

        private readonly Dictionary<int, GlobalPlaneLandmark> _landmarks =
            new Dictionary<int, GlobalPlaneLandmark>(MaxLandmarks);
        private readonly Dictionary<int, int> _assignmentBySource =
            new Dictionary<int, int>(32);
        private readonly List<PendingSeed> _seeds = new List<PendingSeed>(16);
        private readonly HashSet<int> _contradictedThisEpoch =
            new HashSet<int>();
        private int _nextId = 1;
        private int _epoch;
        private int _lastObservationCount;
        private int _rejectedThisEpoch;
        private bool _disposed;

        internal IEnumerable<GlobalPlaneLandmark> Landmarks => _landmarks.Values;
        internal int RetainedCount => _landmarks.Count;
        internal int LastObservationCount => _lastObservationCount;
        internal int RejectedThisEpoch => _rejectedThisEpoch;
        internal int SeedCount => _seeds.Count;

        internal void Update(IList<GlobalPlaneObservation> input, float now)
        {
            if (_disposed) return;
            _epoch++;
            _assignmentBySource.Clear();
            _contradictedThisEpoch.Clear();
            _rejectedThisEpoch = 0;
            _lastObservationCount = input != null ? input.Count : 0;

            var observations = input != null
                ? new List<GlobalPlaneObservation>(input)
                : new List<GlobalPlaneObservation>();
            observations.Sort((a, b) => b.SupportArea.CompareTo(a.SupportArea));

            var seen = new HashSet<int>();
            for (int i = 0; i < observations.Count; i++)
            {
                GlobalPlaneObservation observation = observations[i];
                NormalizeObservation(observation);
                // A family spanning two surfaces has patch centres on both
                // sides of its fitted equation.  Do not let that blended fit
                // vote for either plane or become a third plane at the edge.
                if (!IsCoherentObservation(observation)) continue;
                GlobalPlaneLandmark winner = FindBestLandmark(observation);
                if (winner == null)
                {
                    // An incompatible claim occupying the already confirmed
                    // coverage is normally a fold, curtain ripple or rough
                    // zero-crossing, not a new room plane.  A genuine adjacent
                    // wall becomes eligible once its support grows away from
                    // the old plane instead of being born on the corner.
                    if (!CanSeedObservation(observation) ||
                        IsAmbiguousNearConfirmedStructure(observation))
                        continue;
                    winner = TryPromoteSeed(observation, now);
                    if (winner == null) continue;
                }
                else
                    RemoveCompatibleSeeds(observation);

                RegisterContradictions(observation, winner);
                UpdateLandmark(winner, observation, now);
                seen.Add(winner.Id);
                _assignmentBySource[observation.SourceId] = winner.Id;
            }

            ApplyContradictions(seen, now);
            UpdateMissingAndLifecycle(seen, now);
            CullStaleSeeds(now);
            MergeCanonicalDuplicates(now);
            EnforceCapacity(now);
        }

        internal bool TryGetAssignment(int sourceId, out int landmarkId)
        {
            return _assignmentBySource.TryGetValue(sourceId, out landmarkId);
        }

        internal bool IsRecentlyVisible(GlobalPlaneLandmark landmark, float now)
        {
            return landmark != null &&
                   landmark.State != GlobalPlaneLandmarkState.Rejected &&
                   now - landmark.LastSeen <= DormantAfterSeconds &&
                   HasCellsWithVotes(landmark, 4);
        }

        internal bool IsStructural(GlobalPlaneLandmark landmark)
        {
            return landmark != null &&
                   (landmark.State == GlobalPlaneLandmarkState.Confirmed ||
                    landmark.State == GlobalPlaneLandmarkState.Dormant) &&
                   landmark.ObservationEpochs >= StructuralObservationEpochs &&
                   CountCellsWithVotes(landmark, DurableCellVotes) >=
                   StructuralCellCount;
        }

        internal bool IsDisplayable(GlobalPlaneLandmark landmark)
        {
            return landmark != null &&
                   (landmark.State == GlobalPlaneLandmarkState.Confirmed ||
                    landmark.State == GlobalPlaneLandmarkState.Dormant) &&
                   landmark.ObservationEpochs >= ConfirmObservationEpochs &&
                   CountCellsWithVotes(landmark, DisplayCellVotes) >=
                   ConfirmCellCount;
        }

        internal int CountCellsWithVotes(GlobalPlaneLandmark landmark,
            int minimumVotes)
        {
            int count = 0;
            foreach (GlobalPlaneLandmarkCell cell in landmark.Cells.Values)
                if (cell.Votes >= minimumVotes) count++;
            return count;
        }

        internal Vector3 CellCenter(GlobalPlaneLandmark landmark, long key)
        {
            DecodeCellKey(key, out int u, out int v);
            return landmark.Normal * landmark.Offset +
                   landmark.U * ((u + 0.5f) * CellSizeMeters) +
                   landmark.V * ((v + 0.5f) * CellSizeMeters);
        }

        internal void Reset()
        {
            _landmarks.Clear();
            _assignmentBySource.Clear();
            _seeds.Clear();
            _contradictedThisEpoch.Clear();
            _nextId = 1;
            _epoch = 0;
            _lastObservationCount = 0;
            _rejectedThisEpoch = 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Reset();
        }

        private GlobalPlaneLandmark FindBestLandmark(
            GlobalPlaneObservation observation)
        {
            GlobalPlaneLandmark best = null;
            float bestCost = float.MaxValue;
            foreach (GlobalPlaneLandmark landmark in _landmarks.Values)
            {
                if (landmark.State == GlobalPlaneLandmarkState.Rejected ||
                    landmark.Horizontal != observation.Horizontal) continue;

                PlaneDifference(landmark, observation, out float angle,
                    out float offsetDelta, out float signedOffset);
                bool established = landmark.State ==
                    GlobalPlaneLandmarkState.Confirmed ||
                    landmark.State == GlobalPlaneLandmarkState.Dormant;
                float angleGate = established
                    ? ConfirmedAssociationAngleDegrees
                    : AssociationAngleDegrees;
                float offsetGate = Mathf.Clamp(
                    0.04f + 2f * Mathf.Sqrt(
                        Mathf.Max(0.000001f, landmark.OffsetVariance) +
                        observation.OffsetSigmaMeters *
                        observation.OffsetSigmaMeters),
                    StrongOffsetMeters, established
                        ? ConfirmedOffsetGateMeters
                        : MaximumOffsetGateMeters);
                if (angle > angleGate || offsetDelta > offsetGate) continue;

                float gap = DistanceObservationToCoverage(landmark, observation);
                bool strongEquation = angle <= StrongAngleDegrees &&
                    offsetDelta <= StrongOffsetMeters;
                if (!strongEquation && gap > MaximumCoverageGapMeters) continue;

                float angleVariance = Mathf.Max(1f,
                    landmark.AngleVariance +
                    observation.AngleSigmaDegrees *
                    observation.AngleSigmaDegrees);
                float offsetVariance = Mathf.Max(0.0004f,
                    landmark.OffsetVariance +
                    observation.OffsetSigmaMeters *
                    observation.OffsetSigmaMeters);
                float mahalanobis = angle * angle / angleVariance +
                    offsetDelta * offsetDelta / offsetVariance;
                // The online variances are useful for ranking already-valid
                // candidates, but they are not calibrated sensor covariance.
                // They therefore have no admission/veto authority here.
                float stateBias = landmark.State ==
                    GlobalPlaneLandmarkState.Confirmed ? -0.30f : 0f;
                float cost = mahalanobis +
                    Mathf.Min(gap, MaximumCoverageGapMeters) * 0.20f +
                    stateBias + Mathf.Abs(signedOffset - landmark.Offset) * 0.01f;
                if (cost >= bestCost) continue;
                best = landmark;
                bestCost = cost;
            }
            return best;
        }

        private GlobalPlaneLandmark CreateLandmark(
            GlobalPlaneObservation observation, float now)
        {
            if (_landmarks.Count >= MaxLandmarks && !EvictWeakest(now))
                return null;

            BuildBasis(observation.Normal, observation.Horizontal,
                out Vector3 u, out Vector3 v);
            var landmark = new GlobalPlaneLandmark
            {
                Id = _nextId++,
                Normal = observation.Normal,
                Offset = observation.Offset,
                U = u,
                V = v,
                Horizontal = observation.Horizontal,
                State = GlobalPlaneLandmarkState.Tentative,
                FirstSeen = now,
                LastSeen = now,
                SupportArea = observation.SupportArea,
                AngleVariance = Mathf.Max(1f,
                    observation.AngleSigmaDegrees *
                    observation.AngleSigmaDegrees),
                OffsetVariance = Mathf.Max(0.0004f,
                    observation.OffsetSigmaMeters *
                    observation.OffsetSigmaMeters),
                EvidenceScore = 3,
            };
            _landmarks.Add(landmark.Id, landmark);
            return landmark;
        }

        private GlobalPlaneLandmark TryPromoteSeed(
            GlobalPlaneObservation observation, float now)
        {
            PendingSeed best = null;
            float bestCost = float.MaxValue;
            for (int i = 0; i < _seeds.Count; i++)
            {
                PendingSeed seed = _seeds[i];
                if (seed.Horizontal != observation.Horizontal) continue;
                float alignment = Vector3.Dot(seed.Normal, observation.Normal);
                float sign = alignment < 0f ? -1f : 1f;
                float angle = Vector3.Angle(seed.Normal,
                    observation.Normal * sign);
                float offset = Mathf.Abs(seed.Offset -
                    observation.Offset * sign);
                if (angle > RelaxedMergeAngleDegrees ||
                    offset > MaximumOffsetGateMeters) continue;
                float gap = Vector3.Distance(seed.Center, observation.Center);
                bool strongEquation = angle <= StrongAngleDegrees &&
                    offset <= StrongOffsetMeters;
                if (!strongEquation && gap > MaximumCoverageGapMeters)
                    continue;
                float cost = angle / AssociationAngleDegrees +
                    offset / MaximumOffsetGateMeters +
                    Mathf.Min(gap, MaximumCoverageGapMeters) * 0.10f;
                if (cost >= bestCost) continue;
                best = seed;
                bestCost = cost;
            }

            if (best == null)
            {
                MakeRoomForSeed(now);
                best = new PendingSeed
                {
                    Normal = observation.Normal,
                    Offset = observation.Offset,
                    Center = observation.Center,
                    Horizontal = observation.Horizontal,
                    SupportArea = observation.SupportArea,
                    AngleSigmaDegrees = observation.AngleSigmaDegrees,
                    OffsetSigmaMeters = observation.OffsetSigmaMeters,
                    Hits = 1,
                    LastEpoch = _epoch,
                    LastSeen = now,
                };
                _seeds.Add(best);
                return null;
            }

            float signToSeed = Vector3.Dot(best.Normal, observation.Normal) < 0f
                ? -1f : 1f;
            best.Normal = Vector3.Slerp(best.Normal,
                observation.Normal * signToSeed, 0.20f).normalized;
            best.Offset = Mathf.Lerp(best.Offset,
                observation.Offset * signToSeed, 0.20f);
            best.Center = Vector3.Lerp(best.Center, observation.Center, 0.20f);
            best.SupportArea = Mathf.Max(best.SupportArea,
                observation.SupportArea);
            best.AngleSigmaDegrees = Mathf.Lerp(best.AngleSigmaDegrees,
                observation.AngleSigmaDegrees, 0.20f);
            best.OffsetSigmaMeters = Mathf.Lerp(best.OffsetSigmaMeters,
                observation.OffsetSigmaMeters, 0.20f);
            if (best.LastEpoch != _epoch)
            {
                best.Hits = Mathf.Min(255, best.Hits + 1);
                best.LastEpoch = _epoch;
            }
            best.LastSeen = now;
            if (best.Hits < RequiredSeedHits(best.SupportArea)) return null;

            observation.Normal = best.Normal;
            observation.Offset = best.Offset;
            observation.Center = best.Center;
            observation.AngleSigmaDegrees = best.AngleSigmaDegrees;
            observation.OffsetSigmaMeters = best.OffsetSigmaMeters;
            _seeds.Remove(best);
            return CreateLandmark(observation, now);
        }

        private static int RequiredSeedHits(float supportArea)
        {
            if (supportArea >= 0.25f) return 2;
            if (supportArea >= 0.12f) return 3;
            return 5;
        }

        private void MakeRoomForSeed(float now)
        {
            if (_seeds.Count < MaxPendingSeeds) return;
            int weakest = 0;
            float weakestScore = float.MaxValue;
            for (int i = 0; i < _seeds.Count; i++)
            {
                PendingSeed seed = _seeds[i];
                float score = seed.Hits * 10f +
                    Mathf.Clamp(seed.SupportArea, 0f, 1f) * 5f -
                    Mathf.Min(TentativeLifetimeSeconds,
                        now - seed.LastSeen);
                if (score >= weakestScore) continue;
                weakest = i;
                weakestScore = score;
            }
            _seeds.RemoveAt(weakest);
        }

        private void RemoveCompatibleSeeds(GlobalPlaneObservation observation)
        {
            for (int i = _seeds.Count - 1; i >= 0; i--)
            {
                PendingSeed seed = _seeds[i];
                if (seed.Horizontal != observation.Horizontal) continue;
                float alignment = Vector3.Dot(seed.Normal, observation.Normal);
                float sign = alignment < 0f ? -1f : 1f;
                if (Vector3.Angle(seed.Normal,
                        observation.Normal * sign) <= StrongAngleDegrees &&
                    Mathf.Abs(seed.Offset - observation.Offset * sign) <=
                    StrongOffsetMeters)
                    _seeds.RemoveAt(i);
            }
        }

        private void CullStaleSeeds(float now)
        {
            for (int i = _seeds.Count - 1; i >= 0; i--)
                if (now - _seeds[i].LastSeen > TentativeLifetimeSeconds)
                    _seeds.RemoveAt(i);
        }

        private static bool CanSeedObservation(
            GlobalPlaneObservation observation)
        {
            return observation.AngleSigmaDegrees <=
                       SeedMaximumAngleSigmaDegrees &&
                   observation.OffsetSigmaMeters <=
                       SeedMaximumOffsetSigmaMeters &&
                   observation.SupportArea >= 0.02f;
        }

        private static bool IsCoherentObservation(
            GlobalPlaneObservation observation)
        {
            if (observation.Patches.Count < 3) return true;
            float residualGate = Mathf.Clamp(
                observation.OffsetSigmaMeters * 2f, 0.05f,
                ObservationPatchResidualMeters);
            int coherent = 0;
            for (int i = 0; i < observation.Patches.Count; i++)
            {
                float residual = Mathf.Abs(
                    Vector3.Dot(observation.Normal,
                        observation.Patches[i].Center) -
                    observation.Offset);
                if (residual <= residualGate) coherent++;
            }
            return coherent >= Mathf.CeilToInt(
                observation.Patches.Count * MinimumCoherentPatchFraction);
        }

        private bool IsAmbiguousNearConfirmedStructure(
            GlobalPlaneObservation observation)
        {
            var conflicts = new List<GlobalPlaneLandmark>(4);
            foreach (GlobalPlaneLandmark landmark in _landmarks.Values)
            {
                if (landmark.State != GlobalPlaneLandmarkState.Confirmed &&
                    landmark.State != GlobalPlaneLandmarkState.Dormant)
                    continue;
                PlaneDifference(landmark, observation, out float angle,
                    out _, out _);
                if (angle < CornerSeparationAngleDegrees) continue;
                if (DistanceObservationToCoverage(landmark, observation) >
                    AmbiguousCoverageGapMeters) continue;
                conflicts.Add(landmark);
            }
            if (conflicts.Count == 0) return false;

            int sampleCount = Mathf.Max(1, observation.Patches.Count);
            int occupied = 0;
            for (int i = 0; i < sampleCount; i++)
            {
                Vector3 point = observation.Patches.Count > 0
                    ? observation.Patches[i].Center
                    : observation.Center;
                bool claimed = false;
                for (int c = 0; c < conflicts.Count; c++)
                {
                    GlobalPlaneLandmark plane = conflicts[c];
                    float planeDistance = Mathf.Abs(
                        Vector3.Dot(plane.Normal, point) - plane.Offset);
                    if (planeDistance > AmbiguousPlaneDistanceMeters ||
                        DistanceToCoverage(plane, point) >
                        AmbiguousCoverageGapMeters)
                        continue;
                    claimed = true;
                    break;
                }
                if (claimed) occupied++;
            }
            // If most of a differently oriented claim already occupies the
            // evidence footprint of a confirmed plane, quarantine it.  A true
            // adjacent plane is admitted once most of its support extends away
            // from the old surface, so corners remain discoverable but cannot
            // spawn a fan of blended hypotheses.
            return occupied >= Mathf.CeilToInt(sampleCount * 0.65f);
        }

        private void UpdateLandmark(GlobalPlaneLandmark landmark,
            GlobalPlaneObservation observation, float now)
        {
            float alignment = Vector3.Dot(landmark.Normal, observation.Normal);
            float sign = alignment < 0f ? -1f : 1f;
            Vector3 observedNormal = observation.Normal * sign;
            float observedOffset = observation.Offset * sign;

            float observationAngleVariance = Mathf.Max(1f,
                observation.AngleSigmaDegrees * observation.AngleSigmaDegrees);
            float observationOffsetVariance = Mathf.Max(0.0004f,
                observation.OffsetSigmaMeters * observation.OffsetSigmaMeters);
            float supportWeight = Mathf.Clamp(observation.SupportArea / 0.25f,
                0.25f, 4f);
            float angleGain = Mathf.Clamp(
                landmark.AngleVariance /
                (landmark.AngleVariance + observationAngleVariance /
                 supportWeight), 0.04f, 0.22f);
            float offsetGain = Mathf.Clamp(
                landmark.OffsetVariance /
                (landmark.OffsetVariance + observationOffsetVariance /
                 supportWeight), 0.04f, 0.22f);

            Vector3 oldNormal = landmark.Normal;
            float oldOffset = landmark.Offset;
            Vector3 oldU = landmark.U;
            Vector3 oldV = landmark.V;
            landmark.Normal = Vector3.Slerp(landmark.Normal, observedNormal,
                angleGain).normalized;
            CanonicalizeNormal(ref landmark.Normal, ref observedOffset);
            landmark.Offset = Mathf.Lerp(landmark.Offset, observedOffset,
                offsetGain);
            landmark.AngleVariance = Mathf.Lerp(landmark.AngleVariance,
                observationAngleVariance, angleGain);
            landmark.OffsetVariance = Mathf.Lerp(landmark.OffsetVariance,
                observationOffsetVariance, offsetGain);
            landmark.SupportArea = Mathf.Lerp(landmark.SupportArea,
                observation.SupportArea, 0.20f);

            if (Vector3.Angle(oldNormal, landmark.Normal) > 0.10f ||
                Mathf.Abs(oldOffset - landmark.Offset) > 0.002f)
                ReprojectCells(landmark, oldNormal, oldOffset, oldU, oldV);
            else
                BuildBasis(landmark.Normal, landmark.Horizontal,
                    out landmark.U, out landmark.V);

            if (landmark.LastObservationEpoch != _epoch)
            {
                landmark.ObservationEpochs = Mathf.Min(65535,
                    landmark.ObservationEpochs + 1);
                landmark.LastObservationEpoch = _epoch;
            }
            landmark.LastSeen = now;
            landmark.ConsecutiveMisses = 0;
            landmark.EvidenceScore = Mathf.Min(MaximumEvidence,
                landmark.EvidenceScore + 3);
            landmark.Contradictions = Mathf.Max(0,
                landmark.Contradictions - 1);
            StampObservation(landmark, observation, now);

            if (landmark.State == GlobalPlaneLandmarkState.Dormant)
                landmark.State = GlobalPlaneLandmarkState.Confirmed;
            if (landmark.State == GlobalPlaneLandmarkState.Tentative &&
                landmark.ObservationEpochs >= ConfirmObservationEpochs &&
                landmark.Cells.Count >= ConfirmCellCount &&
                landmark.EvidenceScore >= ConfirmEvidence)
                landmark.State = GlobalPlaneLandmarkState.Confirmed;
        }

        private void RegisterContradictions(GlobalPlaneObservation observation,
            GlobalPlaneLandmark winner)
        {
            foreach (GlobalPlaneLandmark competitor in _landmarks.Values)
            {
                if (competitor.Id == winner.Id ||
                    competitor.State == GlobalPlaneLandmarkState.Rejected ||
                    competitor.Horizontal != observation.Horizontal) continue;
                if (DistanceObservationToCoverage(competitor, observation) >
                    CompetitionCoverageGapMeters) continue;

                PlaneDifference(competitor, observation, out float angle,
                    out float offsetDelta, out _);
                // Perpendicular observations are legitimate corners. Compete
                // only when two near-parallel hypotheses claim the same space.
                if (angle > 20f || offsetDelta <= StrongOffsetMeters) continue;
                _contradictedThisEpoch.Add(competitor.Id);
            }
        }

        private void ApplyContradictions(HashSet<int> seen, float now)
        {
            foreach (int id in _contradictedThisEpoch)
            {
                // If the old plane also received direct support this epoch,
                // this is a legitimate edge/two-surface situation, not a loss.
                if (seen.Contains(id) ||
                    !_landmarks.TryGetValue(id,
                        out GlobalPlaneLandmark competitor)) continue;
                competitor.Contradictions = Mathf.Min(255,
                    competitor.Contradictions + 1);
                competitor.EvidenceScore = Mathf.Max(MinimumEvidence,
                    competitor.EvidenceScore - 4);
                if (competitor.Contradictions >= DemoteContradictions &&
                    competitor.EvidenceScore <= ConfirmEvidence &&
                    competitor.State == GlobalPlaneLandmarkState.Confirmed)
                    competitor.State = GlobalPlaneLandmarkState.Tentative;
                if (competitor.Contradictions >= RejectContradictions &&
                    competitor.EvidenceScore <= MinimumEvidence)
                {
                    competitor.State = GlobalPlaneLandmarkState.Rejected;
                    competitor.LastSeen = now;
                }
            }
        }

        private void UpdateMissingAndLifecycle(HashSet<int> seen, float now)
        {
            var remove = new List<int>();
            foreach (GlobalPlaneLandmark landmark in _landmarks.Values)
            {
                if (!seen.Contains(landmark.Id))
                {
                    landmark.ConsecutiveMisses = Mathf.Min(255,
                        landmark.ConsecutiveMisses + 1);
                    if (landmark.State == GlobalPlaneLandmarkState.Tentative)
                        landmark.EvidenceScore = Mathf.Max(MinimumEvidence,
                            landmark.EvidenceScore - 1);
                    else if (landmark.State == GlobalPlaneLandmarkState.Confirmed &&
                             now - landmark.LastSeen > DormantAfterSeconds)
                        landmark.State = GlobalPlaneLandmarkState.Dormant;
                }

                CullWeakCells(landmark, now);
                float age = now - landmark.LastSeen;
                if (landmark.State == GlobalPlaneLandmarkState.Rejected ||
                    (landmark.State == GlobalPlaneLandmarkState.Tentative &&
                     (age > TentativeLifetimeSeconds ||
                      landmark.EvidenceScore <= MinimumEvidence)) ||
                    (landmark.State == GlobalPlaneLandmarkState.Dormant &&
                     age > DormantLifetimeSeconds && !IsStructuralHistory(landmark)))
                    remove.Add(landmark.Id);
            }
            for (int i = 0; i < remove.Count; i++)
            {
                if (_landmarks.TryGetValue(remove[i], out GlobalPlaneLandmark old) &&
                    old.State == GlobalPlaneLandmarkState.Rejected)
                    _rejectedThisEpoch++;
                _landmarks.Remove(remove[i]);
            }
        }

        private void MergeCanonicalDuplicates(float now)
        {
            bool merged;
            do
            {
                merged = false;
                var planes = new List<GlobalPlaneLandmark>(_landmarks.Values);
                planes.Sort((a, b) => LandmarkStrength(b, now).CompareTo(
                    LandmarkStrength(a, now)));
                for (int i = 0; i < planes.Count && !merged; i++)
                {
                    GlobalPlaneLandmark keep = planes[i];
                    if (!_landmarks.ContainsKey(keep.Id)) continue;
                    for (int j = i + 1; j < planes.Count; j++)
                    {
                        GlobalPlaneLandmark remove = planes[j];
                        if (!_landmarks.ContainsKey(remove.Id) ||
                            keep.Horizontal != remove.Horizontal) continue;
                        PlaneDifference(keep, remove, out float angle,
                            out float offsetDelta);
                        bool strictEquation =
                            angle <= StrictMergeAngleDegrees &&
                            offsetDelta <= StrictMergeOffsetMeters;
                        if (!strictEquation)
                        {
                            if (angle > RelaxedMergeAngleDegrees ||
                                offsetDelta > RelaxedMergeOffsetMeters)
                                continue;
                            // A modest equation disagreement may be TSDF fit
                            // noise on two fragments of one wall, but it is not
                            // enough to merge remote parallel structures.  The
                            // relaxed path therefore requires their evidence
                            // extents to touch or nearly touch on the plane.
                            if (CoverageGapBetween(keep, remove) >
                                RelaxedMergeCoverageGapMeters)
                                continue;
                        }
                        MergeInto(keep, remove);
                        merged = true;
                        break;
                    }
                }
            } while (merged);
        }

        private void MergeInto(GlobalPlaneLandmark keep,
            GlobalPlaneLandmark remove)
        {
            foreach (KeyValuePair<long, GlobalPlaneLandmarkCell> pair in
                     remove.Cells)
            {
                Vector3 point = CellCenter(remove, pair.Key);
                long targetKey = CellKeyOnPlane(keep, point);
                if (!keep.Cells.TryGetValue(targetKey,
                        out GlobalPlaneLandmarkCell target))
                {
                    target = new GlobalPlaneLandmarkCell();
                    keep.Cells.Add(targetKey, target);
                }
                GlobalPlaneLandmarkCell source = pair.Value;
                target.Votes = Mathf.Max(target.Votes, source.Votes);
                target.LastEpoch = Mathf.Max(target.LastEpoch, source.LastEpoch);
                target.LastSeen = Mathf.Max(target.LastSeen, source.LastSeen);
            }
            keep.ObservationEpochs = Mathf.Max(keep.ObservationEpochs,
                remove.ObservationEpochs);
            keep.LastSeen = Mathf.Max(keep.LastSeen, remove.LastSeen);
            keep.FirstSeen = Mathf.Min(keep.FirstSeen, remove.FirstSeen);
            keep.EvidenceScore = Mathf.Max(keep.EvidenceScore,
                remove.EvidenceScore);
            keep.Contradictions = Mathf.Min(keep.Contradictions,
                remove.Contradictions);
            if (remove.State == GlobalPlaneLandmarkState.Confirmed)
                keep.State = GlobalPlaneLandmarkState.Confirmed;

            var sources = new List<int>();
            foreach (KeyValuePair<int, int> pair in _assignmentBySource)
                if (pair.Value == remove.Id) sources.Add(pair.Key);
            for (int i = 0; i < sources.Count; i++)
                _assignmentBySource[sources[i]] = keep.Id;
            _landmarks.Remove(remove.Id);
        }

        private void EnforceCapacity(float now)
        {
            while (_landmarks.Count > MaxLandmarks && EvictWeakest(now)) { }
        }

        private bool EvictWeakest(float now)
        {
            GlobalPlaneLandmark weakest = null;
            float weakestStrength = float.MaxValue;
            foreach (GlobalPlaneLandmark landmark in _landmarks.Values)
            {
                if (IsStructuralHistory(landmark)) continue;
                float strength = LandmarkStrength(landmark, now);
                if (strength >= weakestStrength) continue;
                weakest = landmark;
                weakestStrength = strength;
            }
            if (weakest == null) return false;
            _landmarks.Remove(weakest.Id);
            return true;
        }

        private static float LandmarkStrength(GlobalPlaneLandmark landmark,
            float now)
        {
            float state = landmark.State == GlobalPlaneLandmarkState.Confirmed
                ? 100f
                : landmark.State == GlobalPlaneLandmarkState.Dormant ? 50f : 0f;
            return state + landmark.EvidenceScore * 2f +
                   landmark.ObservationEpochs * 0.25f +
                   landmark.Cells.Count * 0.10f -
                   landmark.Contradictions * 3f -
                   Mathf.Min(30f, now - landmark.LastSeen);
        }

        private void StampObservation(GlobalPlaneLandmark landmark,
            GlobalPlaneObservation observation, float now)
        {
            float residualGate = Mathf.Clamp(
                observation.OffsetSigmaMeters * 2f, 0.05f,
                ObservationPatchResidualMeters);
            for (int i = 0; i < observation.Patches.Count; i++)
            {
                GlobalPlanePatch patch = observation.Patches[i];
                // Even a coherent family can contain a few edge witnesses.
                // They may support the identity statistically, but must not
                // extend its visible coverage across a physical turn.
                if (Mathf.Abs(Vector3.Dot(landmark.Normal, patch.Center) -
                              landmark.Offset) > residualGate)
                    continue;
                Vector3 projected = patch.Center - landmark.Normal *
                    (Vector3.Dot(landmark.Normal, patch.Center) -
                     landmark.Offset);
                float centerU = Vector3.Dot(projected, landmark.U);
                float centerV = Vector3.Dot(projected, landmark.V);
                float half = Mathf.Clamp(patch.HalfExtent,
                    CellSizeMeters * 0.20f, CellSizeMeters * 1.5f);
                int minU = Mathf.FloorToInt((centerU - half) / CellSizeMeters);
                int maxU = Mathf.FloorToInt((centerU + half - 0.001f) /
                                            CellSizeMeters);
                int minV = Mathf.FloorToInt((centerV - half) / CellSizeMeters);
                int maxV = Mathf.FloorToInt((centerV + half - 0.001f) /
                                            CellSizeMeters);
                for (int v = minV; v <= maxV; v++)
                {
                    for (int u = minU; u <= maxU; u++)
                    {
                        long key = CellKey(u, v);
                        if (!landmark.Cells.TryGetValue(key,
                                out GlobalPlaneLandmarkCell cell))
                        {
                            cell = new GlobalPlaneLandmarkCell();
                            landmark.Cells.Add(key, cell);
                        }
                        if (cell.LastEpoch != _epoch)
                        {
                            cell.Votes = Mathf.Min(255, cell.Votes + 1);
                            cell.LastEpoch = _epoch;
                        }
                        cell.LastSeen = now;
                    }
                }
            }
        }

        private static void CullWeakCells(GlobalPlaneLandmark landmark, float now)
        {
            var remove = new List<long>();
            foreach (KeyValuePair<long, GlobalPlaneLandmarkCell> pair in
                     landmark.Cells)
            {
                GlobalPlaneLandmarkCell cell = pair.Value;
                bool persistentIdentity =
                    landmark.State == GlobalPlaneLandmarkState.Confirmed ||
                    landmark.State == GlobalPlaneLandmarkState.Dormant;
                float lifetime = cell.Votes >= DurableCellVotes ||
                                 (persistentIdentity &&
                                  cell.Votes >= DisplayCellVotes)
                    ? float.PositiveInfinity
                    : cell.Votes >= DisplayCellVotes
                        ? WeakCellLifetimeSeconds
                        : PendingCellLifetimeSeconds;
                if (now - cell.LastSeen > lifetime) remove.Add(pair.Key);
            }
            for (int i = 0; i < remove.Count; i++)
                landmark.Cells.Remove(remove[i]);
        }

        private void ReprojectCells(GlobalPlaneLandmark landmark,
            Vector3 oldNormal, float oldOffset, Vector3 oldU, Vector3 oldV)
        {
            BuildBasis(landmark.Normal, landmark.Horizontal,
                out landmark.U, out landmark.V);
            if (landmark.Cells.Count == 0) return;
            var oldCells = new List<KeyValuePair<long, GlobalPlaneLandmarkCell>>(
                landmark.Cells);
            landmark.Cells.Clear();
            for (int i = 0; i < oldCells.Count; i++)
            {
                DecodeCellKey(oldCells[i].Key, out int u, out int v);
                Vector3 point = oldNormal * oldOffset +
                    oldU * ((u + 0.5f) * CellSizeMeters) +
                    oldV * ((v + 0.5f) * CellSizeMeters);
                long key = CellKeyOnPlane(landmark, point);
                if (!landmark.Cells.TryGetValue(key,
                        out GlobalPlaneLandmarkCell target))
                {
                    target = new GlobalPlaneLandmarkCell();
                    landmark.Cells.Add(key, target);
                }
                GlobalPlaneLandmarkCell source = oldCells[i].Value;
                target.Votes = Mathf.Max(target.Votes, source.Votes);
                target.LastEpoch = Mathf.Max(target.LastEpoch, source.LastEpoch);
                target.LastSeen = Mathf.Max(target.LastSeen, source.LastSeen);
            }
        }

        private static float DistanceObservationToCoverage(
            GlobalPlaneLandmark landmark, GlobalPlaneObservation observation)
        {
            if (landmark.Cells.Count == 0) return 0f;
            float best = DistanceToCoverage(landmark, observation.Center);
            for (int i = 0; i < observation.Patches.Count; i++)
                best = Mathf.Min(best, DistanceToCoverage(landmark,
                    observation.Patches[i].Center));
            return best;
        }

        private static float CoverageGapBetween(GlobalPlaneLandmark reference,
            GlobalPlaneLandmark sample)
        {
            if (reference.Cells.Count == 0 || sample.Cells.Count == 0)
                return float.PositiveInfinity;
            CoverageBounds(reference, reference, out float aMinU,
                out float aMaxU, out float aMinV, out float aMaxV);
            CoverageBounds(reference, sample, out float bMinU,
                out float bMaxU, out float bMinV, out float bMaxV);
            float gapU = bMinU > aMaxU ? bMinU - aMaxU :
                aMinU > bMaxU ? aMinU - bMaxU : 0f;
            float gapV = bMinV > aMaxV ? bMinV - aMaxV :
                aMinV > bMaxV ? aMinV - bMaxV : 0f;
            return Mathf.Sqrt(gapU * gapU + gapV * gapV);
        }

        private static void CoverageBounds(GlobalPlaneLandmark reference,
            GlobalPlaneLandmark sample, out float minU, out float maxU,
            out float minV, out float maxV)
        {
            minU = minV = float.PositiveInfinity;
            maxU = maxV = float.NegativeInfinity;
            float half = CellSizeMeters * 0.5f;
            foreach (long key in sample.Cells.Keys)
            {
                Vector3 center = CellCenterStatic(sample, key);
                float u = Vector3.Dot(center, reference.U);
                float v = Vector3.Dot(center, reference.V);
                minU = Mathf.Min(minU, u - half);
                maxU = Mathf.Max(maxU, u + half);
                minV = Mathf.Min(minV, v - half);
                maxV = Mathf.Max(maxV, v + half);
            }
        }

        private static Vector3 CellCenterStatic(GlobalPlaneLandmark landmark,
            long key)
        {
            DecodeCellKey(key, out int u, out int v);
            return landmark.Normal * landmark.Offset +
                   landmark.U * ((u + 0.5f) * CellSizeMeters) +
                   landmark.V * ((v + 0.5f) * CellSizeMeters);
        }

        private static float DistanceToCoverage(GlobalPlaneLandmark landmark,
            Vector3 point)
        {
            float u = Vector3.Dot(point, landmark.U) / CellSizeMeters;
            float v = Vector3.Dot(point, landmark.V) / CellSizeMeters;
            float bestSq = float.MaxValue;
            foreach (long key in landmark.Cells.Keys)
            {
                DecodeCellKey(key, out int cellU, out int cellV);
                float du = u < cellU ? cellU - u :
                    (u > cellU + 1 ? u - (cellU + 1) : 0f);
                float dv = v < cellV ? cellV - v :
                    (v > cellV + 1 ? v - (cellV + 1) : 0f);
                bestSq = Mathf.Min(bestSq, du * du + dv * dv);
            }
            return Mathf.Sqrt(bestSq) * CellSizeMeters;
        }

        private static bool HasCellsWithVotes(GlobalPlaneLandmark landmark,
            int votes)
        {
            foreach (GlobalPlaneLandmarkCell cell in landmark.Cells.Values)
                if (cell.Votes >= votes) return true;
            return false;
        }

        private static bool IsStructuralHistory(GlobalPlaneLandmark landmark)
        {
            return landmark.ObservationEpochs >= StructuralObservationEpochs &&
                   CountCellsStatic(landmark, DurableCellVotes) >=
                   StructuralCellCount;
        }

        private static int CountCellsStatic(GlobalPlaneLandmark landmark,
            int votes)
        {
            int count = 0;
            foreach (GlobalPlaneLandmarkCell cell in landmark.Cells.Values)
                if (cell.Votes >= votes) count++;
            return count;
        }

        private static void PlaneDifference(GlobalPlaneLandmark landmark,
            GlobalPlaneObservation observation, out float angle,
            out float offsetDelta, out float signedOffset)
        {
            float alignment = Vector3.Dot(landmark.Normal, observation.Normal);
            float sign = alignment < 0f ? -1f : 1f;
            angle = Vector3.Angle(landmark.Normal, observation.Normal * sign);
            signedOffset = observation.Offset * sign;
            float equationDelta = Mathf.Abs(landmark.Offset - signedOffset);
            float centerDelta = Mathf.Abs(
                Vector3.Dot(landmark.Normal, observation.Center) -
                landmark.Offset);
            offsetDelta = Mathf.Max(equationDelta, centerDelta);
        }

        private static void PlaneDifference(GlobalPlaneLandmark a,
            GlobalPlaneLandmark b, out float angle, out float offsetDelta)
        {
            float alignment = Vector3.Dot(a.Normal, b.Normal);
            float sign = alignment < 0f ? -1f : 1f;
            angle = Vector3.Angle(a.Normal, b.Normal * sign);
            offsetDelta = Mathf.Abs(a.Offset - b.Offset * sign);
        }

        private static void NormalizeObservation(
            GlobalPlaneObservation observation)
        {
            if (observation.Normal.sqrMagnitude < 0.5f)
                observation.Normal = Vector3.up;
            observation.Normal.Normalize();
            CanonicalizeNormal(ref observation.Normal, ref observation.Offset);
            observation.AngleSigmaDegrees = Mathf.Clamp(
                observation.AngleSigmaDegrees, 1f, 12f);
            observation.OffsetSigmaMeters = Mathf.Clamp(
                observation.OffsetSigmaMeters, 0.01f, 0.10f);
        }

        private static void CanonicalizeNormal(ref Vector3 normal,
            ref float offset)
        {
            float dominant = Mathf.Abs(normal.x) >= Mathf.Abs(normal.y)
                ? (Mathf.Abs(normal.x) >= Mathf.Abs(normal.z)
                    ? normal.x : normal.z)
                : (Mathf.Abs(normal.y) >= Mathf.Abs(normal.z)
                    ? normal.y : normal.z);
            if (dominant >= 0f) return;
            normal = -normal;
            offset = -offset;
        }

        private static void BuildBasis(Vector3 normal, bool horizontal,
            out Vector3 u, out Vector3 v)
        {
            u = horizontal ? Vector3.right :
                Vector3.Cross(Vector3.up, normal).normalized;
            if (u.sqrMagnitude < 0.5f)
                u = Vector3.Cross(Vector3.forward, normal).normalized;
            if (u.sqrMagnitude < 0.5f) u = Vector3.right;
            v = Vector3.Cross(normal, u).normalized;
            u = Vector3.Cross(v, normal).normalized;
        }

        private static long CellKeyOnPlane(GlobalPlaneLandmark landmark,
            Vector3 point)
        {
            int u = Mathf.FloorToInt(Vector3.Dot(point, landmark.U) /
                                     CellSizeMeters);
            int v = Mathf.FloorToInt(Vector3.Dot(point, landmark.V) /
                                     CellSizeMeters);
            return CellKey(u, v);
        }

        internal static long CellKey(int u, int v)
        {
            return ((long)(uint)u << 32) | (uint)v;
        }

        internal static void DecodeCellKey(long key, out int u, out int v)
        {
            u = unchecked((int)(uint)(key >> 32));
            v = unchecked((int)(uint)key);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

/// <summary>
/// Persistent, Meta-depth-driven plane registry used by the read-only plane shadow.
/// It never writes the production TSDF or changes the production triangle buffer.
/// </summary>
public sealed class ScanCoverStablePlaneRegistry
{
    public struct PlaneConstraint
    {
        public int Id;
        public Vector3 Center;
        public Vector3 Normal;
        public float Offset;
        public float Radius;
        public float Rms;
        public int TotalInliers;
        public int ConsecutiveBatches;
    }

    public struct Settings
    {
        public int MaxSamples;
        public int MaxPlanes;
        public int RansacIterations;
        public int MinInliers;
        public float FitDistanceMeters;
        public float MinSampleNormalDot;
        public float MatchNormalDot;
        public float MatchPlaneDistanceMeters;
        public float MatchTangentialPaddingMeters;
        public int MatureBatches;
        public int RetireMissingBatches;
        public int MatureHoldMissingBatches;
        public int MatureDemoteBadRmsBatches;
        public float VertexNormalDot;
        public float MaxVertexPlaneDistanceMeters;
        public float PatchPaddingMeters;
        public float MaxVertexMoveMeters;
        public float MinTriangleAreaRatio;
        public float MinTriangleNormalDot;
        public float CanonicalMergeNormalDot;
        public float CanonicalMergePlaneDistanceMeters;
        public float CanonicalMergeMinOverlapRatio;
        public float ParallelDuplicateMaxDistanceMeters;
        public float ParallelDuplicateMaxSigma;
        public float ParallelDuplicateMinOverlapRatio;
    }

    public struct Result
    {
        public int InputSamples;
        public int ExtractionAttempts;
        public int MaxActiveSamples;
        public int RansacValidHypotheses;
        public int MaxBestDistanceInliers;
        public int MaxBestNormalInliers;
        public int RejectedInsufficientActiveSamples;
        public int RejectedNoValidHypothesis;
        public int RejectedMinInliers;
        public int RejectedByNormalConsistency;
        public int RejectedRefinedMinInliers;
        public int RejectedRms;
        public int MaturityBlockedRms;
        public int ProposalCount;
        public int CandidatePlaneCount;
        public int MaturePlaneCount;
        public int AssignedVertexCount;
        public int MovedVertexCount;
        public int RevertedVertexCount;
        public float MeanResidualBeforeMeters;
        public float MeanResidualAfterMeters;
        public float MaxDisplacementMeters;
        public int CanonicalMergeCount;
        public int CanonicalMergedTrackCount;
        public float CanonicalMergeMaxPlaneDistanceMeters;
        public float CanonicalMergeMinNormalDot;
        public int ParallelDuplicateSuppressedCount;
    }

    private struct Sample
    {
        public Vector3 Point;
        public Vector3 Normal;
        public bool Active;
    }

    private struct Proposal
    {
        public Vector3 Center;
        public Vector3 Normal;
        public float Offset;
        public float Radius;
        public float Rms;
        public int Inliers;
    }

    private sealed class Track
    {
        public int Id;
        public Vector3 Center;
        public Vector3 Normal;
        public float Offset;
        public float Radius;
        public float Rms;
        public int TotalInliers;
        public int ConsecutiveBatches;
        public int MissingBatches;
        public bool Mature;
        public bool Matched;
        public int BadRmsBatches;
        public int SuppressedById;
        public readonly List<int> AliasIds = new List<int>();
    }

    private readonly List<Track> _tracks = new List<Track>(12);
    private int _nextId = 1;
    private int _matureHoldMissingBatches = 1;
    private int[] _lastVertexPlaneIds = Array.Empty<int>();

    public void Clear()
    {
        _tracks.Clear();
        _nextId = 1;
        _lastVertexPlaneIds = Array.Empty<int>();
    }

    public List<PlaneConstraint> GetMatureConstraints()
    {
        List<PlaneConstraint> constraints = new List<PlaneConstraint>(_tracks.Count);
        for (int i = 0; i < _tracks.Count; i++)
        {
            Track track = _tracks[i];
            if (!IsUsableMatureTrack(track) || !Finite(track.Center) || !Finite(track.Normal))
                continue;
            constraints.Add(new PlaneConstraint
            {
                Id = track.Id,
                Center = track.Center,
                Normal = track.Normal,
                Offset = track.Offset,
                Radius = track.Radius,
                Rms = track.Rms,
                TotalInliers = track.TotalInliers,
                ConsecutiveBatches = track.ConsecutiveBatches
            });
        }
        return constraints;
    }

    public List<PlaneConstraint> GetCurrentCandidateConstraints()
    {
        List<PlaneConstraint> constraints = new List<PlaneConstraint>(_tracks.Count);
        for (int i = 0; i < _tracks.Count; i++)
        {
            Track track = _tracks[i];
            if (track == null || track.Mature || track.MissingBatches > 0 ||
                track.SuppressedById != 0 ||
                !Finite(track.Center) || !Finite(track.Normal))
                continue;
            constraints.Add(new PlaneConstraint
            {
                Id = track.Id,
                Center = track.Center,
                Normal = track.Normal,
                Offset = track.Offset,
                Radius = track.Radius,
                Rms = track.Rms,
                TotalInliers = track.TotalInliers,
                ConsecutiveBatches = track.ConsecutiveBatches
            });
        }
        return constraints;
    }

    public int GetLastVertexPlaneId(int vertexIndex)
    {
        return vertexIndex >= 0 && vertexIndex < _lastVertexPlaneIds.Length
            ? _lastVertexPlaneIds[vertexIndex]
            : -1;
    }

    public bool IsLastVertexConstrained(int vertexIndex)
    {
        return GetLastVertexPlaneId(vertexIndex) >= 0;
    }

    public StringBuilder BuildCsv()
    {
        StringBuilder csv = new StringBuilder(2048);
        csv.AppendLine("plane_id,mature,usable,consecutive_batches,missing_batches,bad_rms_batches,suppressed_by,total_inliers,center_x,center_y,center_z,normal_x,normal_y,normal_z,offset_m,radius_m,rms_m,alias_count,alias_ids");
        for (int i = 0; i < _tracks.Count; i++)
        {
            Track track = _tracks[i];
            csv.Append(track.Id).Append(',')
                .Append(track.Mature ? 1 : 0).Append(',')
                .Append(IsUsableMatureTrack(track) ? 1 : 0).Append(',')
                .Append(track.ConsecutiveBatches).Append(',')
                .Append(track.MissingBatches).Append(',')
                .Append(track.BadRmsBatches).Append(',')
                .Append(track.SuppressedById).Append(',')
                .Append(track.TotalInliers).Append(',')
                .Append(track.Center.x.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Center.y.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Center.z.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Normal.x.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Normal.y.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Normal.z.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Offset.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Radius.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.Rms.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(track.AliasIds.Count).Append(',')
                .Append('"').Append(string.Join(";", track.AliasIds)).Append('"').AppendLine();
        }
        return csv;
    }

    public bool UpdateAndBuildShadow(
        List<Vector3> samplePoints,
        List<Vector3> sampleNormals,
        List<Vector3> sourceVertices,
        List<int> sourceTriangles,
        Settings settings,
        bool allowCandidatePlanes,
        out List<Vector3> shadowVertices,
        out Result result)
    {
        shadowVertices = sourceVertices != null ? new List<Vector3>(sourceVertices) : new List<Vector3>();
        result = default;
        if (samplePoints == null || sampleNormals == null || sourceVertices == null || sourceTriangles == null ||
            samplePoints.Count != sampleNormals.Count || samplePoints.Count < 3 ||
            sourceVertices.Count < 3 || sourceTriangles.Count < 3)
            return false;

        SanitizeSettings(ref settings);
        _matureHoldMissingBatches = settings.MatureHoldMissingBatches;
        List<Sample> samples = BuildSamples(samplePoints, sampleNormals, settings.MaxSamples);
        result.InputSamples = samples.Count;
        List<Proposal> proposals = ExtractProposals(samples, settings, ref result);
        result.ProposalCount = proposals.Count;
        UpdateTracks(proposals, settings, ref result);

        for (int i = 0; i < _tracks.Count; i++)
        {
            if (IsUsableMatureTrack(_tracks[i]))
                result.MaturePlaneCount++;
            else
                result.CandidatePlaneCount++;
        }

        return BuildSnappedVertices(sourceVertices, sourceTriangles, settings, allowCandidatePlanes,
            shadowVertices, ref result);
    }

    /// <summary>
    /// Updates only the persistent plane tracks. This is the lightweight bootstrap
    /// entry used by the incremental production path, where block-local geometry is
    /// snapped and published later by the authoritative block pipeline.
    /// </summary>
    public bool UpdateTracksOnly(
        List<Vector3> samplePoints,
        List<Vector3> sampleNormals,
        Settings settings,
        out Result result)
    {
        result = default;
        if (samplePoints == null || sampleNormals == null ||
            samplePoints.Count != sampleNormals.Count || samplePoints.Count < 3)
            return false;

        SanitizeSettings(ref settings);
        _matureHoldMissingBatches = settings.MatureHoldMissingBatches;
        List<Sample> samples = BuildSamples(samplePoints, sampleNormals, settings.MaxSamples);
        result.InputSamples = samples.Count;
        if (samples.Count < settings.MinInliers)
            return false;

        List<Proposal> proposals = ExtractProposals(samples, settings, ref result);
        result.ProposalCount = proposals.Count;
        UpdateTracks(proposals, settings, ref result);

        for (int i = 0; i < _tracks.Count; i++)
        {
            if (IsUsableMatureTrack(_tracks[i]))
                result.MaturePlaneCount++;
            else
                result.CandidatePlaneCount++;
        }
        return proposals.Count > 0;
    }

    private static void SanitizeSettings(ref Settings settings)
    {
        settings.MaxSamples = Mathf.Clamp(settings.MaxSamples, 256, 12000);
        settings.MaxPlanes = Mathf.Clamp(settings.MaxPlanes, 1, 12);
        settings.RansacIterations = Mathf.Clamp(settings.RansacIterations, 16, 512);
        settings.MinInliers = Mathf.Clamp(settings.MinInliers, 16, settings.MaxSamples);
        settings.FitDistanceMeters = Mathf.Clamp(settings.FitDistanceMeters, 0.005f, 0.15f);
        settings.MinSampleNormalDot = Mathf.Clamp01(settings.MinSampleNormalDot);
        settings.MatchNormalDot = Mathf.Clamp(settings.MatchNormalDot, 0.5f, 1f);
        settings.MatchPlaneDistanceMeters = Mathf.Clamp(settings.MatchPlaneDistanceMeters, 0.005f, 0.3f);
        settings.MatchTangentialPaddingMeters = Mathf.Clamp(settings.MatchTangentialPaddingMeters, 0.01f, 1f);
        settings.MatureBatches = Mathf.Clamp(settings.MatureBatches, 1, 6);
        settings.RetireMissingBatches = Mathf.Clamp(settings.RetireMissingBatches, 1, 12);
        settings.MatureHoldMissingBatches = Mathf.Clamp(
            settings.MatureHoldMissingBatches, 0, settings.RetireMissingBatches);
        settings.MatureDemoteBadRmsBatches = Mathf.Clamp(settings.MatureDemoteBadRmsBatches, 1, 6);
        settings.VertexNormalDot = Mathf.Clamp(settings.VertexNormalDot, 0.2f, 1f);
        settings.MaxVertexPlaneDistanceMeters = Mathf.Clamp(settings.MaxVertexPlaneDistanceMeters, 0.005f, 0.2f);
        settings.PatchPaddingMeters = Mathf.Clamp(settings.PatchPaddingMeters, 0f, 1f);
        settings.MaxVertexMoveMeters = Mathf.Clamp(settings.MaxVertexMoveMeters, 0.001f, 0.2f);
        settings.MinTriangleAreaRatio = Mathf.Clamp(settings.MinTriangleAreaRatio, 0.05f, 0.95f);
        settings.MinTriangleNormalDot = Mathf.Clamp(settings.MinTriangleNormalDot, -0.5f, 0.999f);
        settings.CanonicalMergeNormalDot = Mathf.Clamp(settings.CanonicalMergeNormalDot, 0.95f, 1f);
        settings.CanonicalMergePlaneDistanceMeters = Mathf.Clamp(
            settings.CanonicalMergePlaneDistanceMeters, 0.005f, 0.1f);
        settings.CanonicalMergeMinOverlapRatio = Mathf.Clamp01(settings.CanonicalMergeMinOverlapRatio);
        settings.ParallelDuplicateMaxDistanceMeters = Mathf.Clamp(
            settings.ParallelDuplicateMaxDistanceMeters, settings.CanonicalMergePlaneDistanceMeters, 0.25f);
        settings.ParallelDuplicateMaxSigma = Mathf.Clamp(settings.ParallelDuplicateMaxSigma, 1f, 8f);
        settings.ParallelDuplicateMinOverlapRatio = Mathf.Clamp01(settings.ParallelDuplicateMinOverlapRatio);
    }

    private bool IsUsableMatureTrack(Track track)
    {
        return track != null && track.Mature &&
               track.MissingBatches <= _matureHoldMissingBatches &&
               track.SuppressedById == 0;
    }

    private static List<Sample> BuildSamples(List<Vector3> points, List<Vector3> normals, int maxSamples)
    {
        int step = Mathf.Max(1, Mathf.CeilToInt(points.Count / (float)maxSamples));
        List<Sample> samples = new List<Sample>(Mathf.Min(points.Count, maxSamples));
        for (int i = 0; i < points.Count; i += step)
        {
            Vector3 point = points[i];
            Vector3 normal = normals[i];
            if (!Finite(point) || !Finite(normal) || normal.sqrMagnitude < 0.000001f)
                continue;
            samples.Add(new Sample { Point = point, Normal = normal.normalized, Active = true });
        }
        return samples;
    }

    private static List<Proposal> ExtractProposals(
        List<Sample> samples,
        Settings settings,
        ref Result result)
    {
        List<Proposal> proposals = new List<Proposal>(settings.MaxPlanes);
        for (int planeIndex = 0; planeIndex < settings.MaxPlanes; planeIndex++)
        {
            if (!TryExtractProposal(samples, settings, planeIndex, ref result, out Proposal proposal))
                break;
            proposals.Add(proposal);
        }
        return proposals;
    }

    private static bool TryExtractProposal(
        List<Sample> samples,
        Settings settings,
        int planeIndex,
        ref Result result,
        out Proposal proposal)
    {
        proposal = default;
        result.ExtractionAttempts++;
        List<int> active = new List<int>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            if (samples[i].Active)
                active.Add(i);
        }
        result.MaxActiveSamples = Mathf.Max(result.MaxActiveSamples, active.Count);
        if (active.Count < settings.MinInliers)
        {
            result.RejectedInsufficientActiveSamples++;
            return false;
        }

        System.Random random = new System.Random(486187739 ^ active.Count * 16777619 ^ planeIndex * 73856093);
        int bestCount = -1;
        int bestDistanceCount = 0;
        Vector3 bestPoint = Vector3.zero;
        Vector3 bestNormal = Vector3.zero;
        for (int iteration = 0; iteration < settings.RansacIterations; iteration++)
        {
            int ia = active[random.Next(active.Count)];
            int ib = active[random.Next(active.Count)];
            int ic = active[random.Next(active.Count)];
            if (ia == ib || ia == ic || ib == ic)
                continue;
            Vector3 a = samples[ia].Point;
            Vector3 normal = Vector3.Cross(samples[ib].Point - a, samples[ic].Point - a);
            if (!Finite(normal) || normal.sqrMagnitude < 0.000001f)
                continue;
            normal.Normalize();
            result.RansacValidHypotheses++;

            int count = 0;
            int distanceCount = 0;
            for (int i = 0; i < active.Count; i++)
            {
                Sample sample = samples[active[i]];
                if (Mathf.Abs(Vector3.Dot(normal, sample.Point - a)) > settings.FitDistanceMeters)
                    continue;
                distanceCount++;
                if (Mathf.Abs(Vector3.Dot(normal, sample.Normal)) >= settings.MinSampleNormalDot)
                    count++;
            }
            if (count < bestCount || (count == bestCount && distanceCount <= bestDistanceCount))
                continue;
            bestCount = count;
            bestDistanceCount = distanceCount;
            bestPoint = a;
            bestNormal = normal;
        }
        result.MaxBestDistanceInliers = Mathf.Max(result.MaxBestDistanceInliers, bestDistanceCount);
        result.MaxBestNormalInliers = Mathf.Max(result.MaxBestNormalInliers, bestCount);
        if (bestNormal.sqrMagnitude < 0.000001f)
        {
            result.RejectedNoValidHypothesis++;
            return false;
        }
        if (bestCount < settings.MinInliers)
        {
            result.RejectedMinInliers++;
            if (bestDistanceCount >= settings.MinInliers)
                result.RejectedByNormalConsistency++;
            return false;
        }

        Vector3 center = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        List<int> inliers = new List<int>(bestCount);
        for (int i = 0; i < active.Count; i++)
        {
            int index = active[i];
            Sample sample = samples[index];
            if (Mathf.Abs(Vector3.Dot(bestNormal, sample.Point - bestPoint)) > settings.FitDistanceMeters ||
                Mathf.Abs(Vector3.Dot(bestNormal, sample.Normal)) < settings.MinSampleNormalDot)
                continue;
            center += sample.Point;
            normalSum += Vector3.Dot(sample.Normal, bestNormal) >= 0f ? sample.Normal : -sample.Normal;
            inliers.Add(index);
        }
        if (inliers.Count < settings.MinInliers)
        {
            result.RejectedRefinedMinInliers++;
            return false;
        }

        center /= inliers.Count;
        Vector3 sampleNormal = normalSum.sqrMagnitude > 0.000001f ? normalSum.normalized : bestNormal;
        float bestNormalSquaredResidual = 0f;
        float sampleNormalSquaredResidual = 0f;
        for (int i = 0; i < inliers.Count; i++)
        {
            Vector3 delta = samples[inliers[i]].Point - center;
            float bestSigned = Vector3.Dot(bestNormal, delta);
            float sampleSigned = Vector3.Dot(sampleNormal, delta);
            bestNormalSquaredResidual += bestSigned * bestSigned;
            sampleNormalSquaredResidual += sampleSigned * sampleSigned;
        }
        Vector3 refinedNormal = sampleNormalSquaredResidual <= bestNormalSquaredResidual
            ? sampleNormal
            : bestNormal;
        float offset = -Vector3.Dot(refinedNormal, center);
        float squaredResidual = 0f;
        float radius = 0f;
        for (int i = 0; i < inliers.Count; i++)
        {
            int index = inliers[i];
            Sample sample = samples[index];
            float signed = Vector3.Dot(refinedNormal, sample.Point) + offset;
            squaredResidual += signed * signed;
            Vector3 tangent = sample.Point - center - refinedNormal * signed;
            radius = Mathf.Max(radius, tangent.magnitude);
        }

        float rms = Mathf.Sqrt(squaredResidual / Mathf.Max(1, inliers.Count));
        if (rms > settings.FitDistanceMeters)
        {
            result.RejectedRms++;
            return false;
        }
        for (int i = 0; i < inliers.Count; i++)
        {
            int index = inliers[i];
            Sample inactive = samples[index];
            inactive.Active = false;
            samples[index] = inactive;
        }

        proposal = new Proposal
        {
            Center = center,
            Normal = refinedNormal,
            Offset = offset,
            Radius = radius,
            Rms = rms,
            Inliers = inliers.Count
        };
        return true;
    }

    private void UpdateTracks(List<Proposal> proposals, Settings settings, ref Result result)
    {
        for (int i = 0; i < _tracks.Count; i++)
            _tracks[i].Matched = false;

        for (int proposalIndex = 0; proposalIndex < proposals.Count; proposalIndex++)
        {
            Proposal proposal = proposals[proposalIndex];
            Track best = null;
            float bestScore = float.PositiveInfinity;
            for (int trackIndex = 0; trackIndex < _tracks.Count; trackIndex++)
            {
                Track track = _tracks[trackIndex];
                if (track.Matched)
                    continue;
                float dot = Vector3.Dot(proposal.Normal, track.Normal);
                float signedOffset = proposal.Offset;
                Vector3 signedNormal = proposal.Normal;
                if (dot < 0f)
                {
                    dot = -dot;
                    signedOffset = -signedOffset;
                    signedNormal = -signedNormal;
                }
                if (dot < settings.MatchNormalDot ||
                    Mathf.Abs(signedOffset - track.Offset) > settings.MatchPlaneDistanceMeters)
                    continue;
                Vector3 delta = proposal.Center - track.Center;
                float normalDistance = Mathf.Abs(Vector3.Dot(delta, track.Normal));
                float tangentDistance = (delta - track.Normal * Vector3.Dot(delta, track.Normal)).magnitude;
                if (normalDistance > settings.MatchPlaneDistanceMeters ||
                    tangentDistance > proposal.Radius + track.Radius + settings.MatchTangentialPaddingMeters)
                    continue;
                float score = (1f - dot) * 4f + normalDistance / settings.MatchPlaneDistanceMeters +
                              tangentDistance / Mathf.Max(0.01f, proposal.Radius + track.Radius) +
                              (track.Mature ? 0f : 0.25f) +
                              track.MissingBatches * 0.05f;
                if (score >= bestScore)
                    continue;
                bestScore = score;
                best = track;
                proposal.Normal = signedNormal;
                proposal.Offset = signedOffset;
            }

            if (best == null)
            {
                _tracks.Add(new Track
                {
                    Id = _nextId++,
                    Center = proposal.Center,
                    Normal = proposal.Normal,
                    Offset = proposal.Offset,
                    Radius = proposal.Radius,
                    Rms = proposal.Rms,
                    TotalInliers = proposal.Inliers,
                    ConsecutiveBatches = 1,
                    MissingBatches = 0,
                    Mature = settings.MatureBatches <= 1 && proposal.Rms <= settings.FitDistanceMeters,
                    Matched = true,
                    BadRmsBatches = proposal.Rms > settings.FitDistanceMeters ? 1 : 0
                });
                continue;
            }

            float oldWeight = Mathf.Max(1, best.TotalInliers);
            float newWeight = Mathf.Max(1, proposal.Inliers);
            float total = oldWeight + newWeight;
            best.Center = (best.Center * oldWeight + proposal.Center * newWeight) / total;
            Vector3 mergedNormal = best.Normal * oldWeight + proposal.Normal * newWeight;
            best.Normal = mergedNormal.sqrMagnitude > 0.000001f ? mergedNormal.normalized : best.Normal;
            best.Offset = -Vector3.Dot(best.Normal, best.Center);
            best.Radius = Mathf.Max(best.Radius, proposal.Radius);
            best.Rms = (best.Rms * oldWeight + proposal.Rms * newWeight) / total;
            best.TotalInliers += proposal.Inliers;
            best.ConsecutiveBatches++;
            best.MissingBatches = 0;
            best.BadRmsBatches = proposal.Rms > settings.FitDistanceMeters
                ? best.BadRmsBatches + 1
                : 0;
            if (!best.Mature)
            {
                best.Mature = best.ConsecutiveBatches >= settings.MatureBatches &&
                              best.Rms <= settings.FitDistanceMeters;
            }
            else if (best.BadRmsBatches >= settings.MatureDemoteBadRmsBatches)
            {
                best.Mature = false;
            }
            if (best.ConsecutiveBatches >= settings.MatureBatches &&
                best.Rms > settings.FitDistanceMeters)
                result.MaturityBlockedRms++;
            best.Matched = true;
        }

        for (int i = _tracks.Count - 1; i >= 0; i--)
        {
            Track track = _tracks[i];
            if (track.Matched)
                continue;
            track.MissingBatches++;
            if (track.MissingBatches > settings.RetireMissingBatches)
                _tracks.RemoveAt(i);
        }
        MergeCanonicalDuplicateTracks(settings, ref result);
        SuppressParallelDuplicateTracks(settings, ref result);
    }

    private void MergeCanonicalDuplicateTracks(Settings settings, ref Result result)
    {
        bool merged;
        do
        {
            merged = false;
            for (int i = 0; i < _tracks.Count && !merged; i++)
            {
                for (int j = i + 1; j < _tracks.Count; j++)
                {
                    Track a = _tracks[i];
                    Track b = _tracks[j];
                    if (!TryGetCanonicalMergeEvidence(a, b, settings,
                            out float normalDot, out float planeDistance, out float overlapRatio))
                        continue;
                    Track survivor;
                    Track absorbed;
                    int absorbedIndex;
                    if (PreferCanonicalTrack(a, b))
                    {
                        survivor = a;
                        absorbed = b;
                        absorbedIndex = j;
                    }
                    else
                    {
                        survivor = b;
                        absorbed = a;
                        absorbedIndex = i;
                    }
                    MergeTrackIntoCanonical(survivor, absorbed, settings);
                    _tracks.RemoveAt(absorbedIndex);
                    result.CanonicalMergeCount++;
                    result.CanonicalMergedTrackCount++;
                    result.CanonicalMergeMaxPlaneDistanceMeters = Mathf.Max(
                        result.CanonicalMergeMaxPlaneDistanceMeters, planeDistance);
                    result.CanonicalMergeMinNormalDot = result.CanonicalMergeCount == 1
                        ? normalDot
                        : Mathf.Min(result.CanonicalMergeMinNormalDot, normalDot);
                    merged = true;
                    break;
                }
            }
        } while (merged);
    }

    private static bool TryGetCanonicalMergeEvidence(
        Track a,
        Track b,
        Settings settings,
        out float normalDot,
        out float planeDistance,
        out float overlapRatio)
    {
        float signedDot = Vector3.Dot(a.Normal, b.Normal);
        normalDot = Mathf.Abs(signedDot);
        float alignedOffsetB = signedDot >= 0f ? b.Offset : -b.Offset;
        planeDistance = Mathf.Abs(a.Offset - alignedOffsetB);
        Vector3 delta = b.Center - a.Center;
        Vector3 tangent = delta - a.Normal * Vector3.Dot(delta, a.Normal);
        float tangentDistance = tangent.magnitude;
        float minRadius = Mathf.Max(0.01f, Mathf.Min(a.Radius, b.Radius));
        float overlapDepth = a.Radius + b.Radius - tangentDistance;
        overlapRatio = overlapDepth / minRadius;
        return normalDot >= settings.CanonicalMergeNormalDot &&
               planeDistance <= settings.CanonicalMergePlaneDistanceMeters &&
               overlapRatio >= settings.CanonicalMergeMinOverlapRatio;
    }

    private static bool PreferCanonicalTrack(Track a, Track b)
    {
        if (a.Mature != b.Mature)
            return a.Mature;
        if (a.ConsecutiveBatches != b.ConsecutiveBatches)
            return a.ConsecutiveBatches > b.ConsecutiveBatches;
        if (a.MissingBatches != b.MissingBatches)
            return a.MissingBatches < b.MissingBatches;
        if (a.TotalInliers != b.TotalInliers)
            return a.TotalInliers > b.TotalInliers;
        return a.Id < b.Id;
    }

    private static void MergeTrackIntoCanonical(Track survivor, Track absorbed, Settings settings)
    {
        bool preserveMature = survivor.Mature || absorbed.Mature;
        float oldWeight = Mathf.Max(1, survivor.TotalInliers);
        float absorbedWeight = Mathf.Max(1, absorbed.TotalInliers);
        float totalWeight = oldWeight + absorbedWeight;
        Vector3 absorbedNormal = absorbed.Normal;
        if (Vector3.Dot(survivor.Normal, absorbedNormal) < 0f)
            absorbedNormal = -absorbedNormal;
        Vector3 mergedNormal = survivor.Normal * oldWeight + absorbedNormal * absorbedWeight;
        if (mergedNormal.sqrMagnitude > 0.000001f)
            mergedNormal.Normalize();
        else
            mergedNormal = survivor.Normal;
        Vector3 mergedCenter = (survivor.Center * oldWeight + absorbed.Center * absorbedWeight) / totalWeight;
        float mergedOffset = -Vector3.Dot(mergedNormal, mergedCenter);
        float survivorPlaneError = Mathf.Abs(Vector3.Dot(mergedNormal, survivor.Center) + mergedOffset);
        float absorbedPlaneError = Mathf.Abs(Vector3.Dot(mergedNormal, absorbed.Center) + mergedOffset);
        float mergedVariance =
            oldWeight * (survivor.Rms * survivor.Rms + survivorPlaneError * survivorPlaneError) +
            absorbedWeight * (absorbed.Rms * absorbed.Rms + absorbedPlaneError * absorbedPlaneError);
        mergedVariance /= totalWeight;
        Vector3 survivorDelta = survivor.Center - mergedCenter;
        Vector3 absorbedDelta = absorbed.Center - mergedCenter;
        float survivorTangent = (survivorDelta - mergedNormal * Vector3.Dot(survivorDelta, mergedNormal)).magnitude;
        float absorbedTangent = (absorbedDelta - mergedNormal * Vector3.Dot(absorbedDelta, mergedNormal)).magnitude;

        survivor.Center = mergedCenter;
        survivor.Normal = mergedNormal;
        survivor.Offset = mergedOffset;
        survivor.Radius = Mathf.Max(survivor.Radius + survivorTangent, absorbed.Radius + absorbedTangent);
        survivor.Rms = Mathf.Sqrt(Mathf.Max(0f, mergedVariance));
        survivor.TotalInliers += absorbed.TotalInliers;
        survivor.ConsecutiveBatches = Mathf.Max(survivor.ConsecutiveBatches, absorbed.ConsecutiveBatches);
        survivor.MissingBatches = Mathf.Min(survivor.MissingBatches, absorbed.MissingBatches);
        survivor.BadRmsBatches = Mathf.Min(survivor.BadRmsBatches, absorbed.BadRmsBatches);
        survivor.Matched |= absorbed.Matched;
        survivor.Mature = preserveMature
            ? survivor.BadRmsBatches < settings.MatureDemoteBadRmsBatches
            : survivor.ConsecutiveBatches >= settings.MatureBatches &&
              survivor.Rms <= settings.FitDistanceMeters;
        if (!survivor.AliasIds.Contains(absorbed.Id))
            survivor.AliasIds.Add(absorbed.Id);
        for (int i = 0; i < absorbed.AliasIds.Count; i++)
        {
            if (!survivor.AliasIds.Contains(absorbed.AliasIds[i]))
                survivor.AliasIds.Add(absorbed.AliasIds[i]);
        }
        survivor.AliasIds.Sort();
    }

    private void SuppressParallelDuplicateTracks(Settings settings, ref Result result)
    {
        for (int i = 0; i < _tracks.Count; i++)
            _tracks[i].SuppressedById = 0;

        for (int i = 0; i < _tracks.Count; i++)
        {
            Track a = _tracks[i];
            if (a.SuppressedById != 0)
                continue;
            for (int j = i + 1; j < _tracks.Count; j++)
            {
                Track b = _tracks[j];
                if (b.SuppressedById != 0)
                    continue;
                float signedDot = Vector3.Dot(a.Normal, b.Normal);
                float normalDot = Mathf.Abs(signedDot);
                if (normalDot < settings.CanonicalMergeNormalDot)
                    continue;
                float alignedOffsetB = signedDot >= 0f ? b.Offset : -b.Offset;
                float planeDistance = Mathf.Abs(a.Offset - alignedOffsetB);
                if (planeDistance <= settings.CanonicalMergePlaneDistanceMeters ||
                    planeDistance > settings.ParallelDuplicateMaxDistanceMeters)
                    continue;
                Vector3 delta = b.Center - a.Center;
                Vector3 tangent = delta - a.Normal * Vector3.Dot(delta, a.Normal);
                float overlapDepth = a.Radius + b.Radius - tangent.magnitude;
                float overlapRatio = overlapDepth / Mathf.Max(0.01f, Mathf.Min(a.Radius, b.Radius));
                if (overlapRatio < settings.ParallelDuplicateMinOverlapRatio)
                    continue;
                float combinedSigma = Mathf.Sqrt(a.Rms * a.Rms + b.Rms * b.Rms);
                if (planeDistance > settings.ParallelDuplicateMaxSigma * Mathf.Max(0.001f, combinedSigma))
                    continue;

                Track survivor = PreferSuppressionTrack(a, b) ? a : b;
                Track suppressed = survivor == a ? b : a;
                suppressed.SuppressedById = survivor.Id;
                result.ParallelDuplicateSuppressedCount++;
                if (suppressed == a)
                    break;
            }
        }
    }

    private bool PreferSuppressionTrack(Track a, Track b)
    {
        bool aUsable = a.Mature && a.MissingBatches <= _matureHoldMissingBatches;
        bool bUsable = b.Mature && b.MissingBatches <= _matureHoldMissingBatches;
        if (aUsable != bUsable)
            return aUsable;
        if (a.MissingBatches != b.MissingBatches)
            return a.MissingBatches < b.MissingBatches;
        if (a.ConsecutiveBatches != b.ConsecutiveBatches)
            return a.ConsecutiveBatches > b.ConsecutiveBatches;
        if (a.TotalInliers != b.TotalInliers)
            return a.TotalInliers > b.TotalInliers;
        if (!Mathf.Approximately(a.Rms, b.Rms))
            return a.Rms < b.Rms;
        return a.Id < b.Id;
    }

    private bool BuildSnappedVertices(
        List<Vector3> vertices,
        List<int> triangles,
        Settings settings,
        bool allowCandidates,
        List<Vector3> output,
        ref Result result)
    {
        _lastVertexPlaneIds = new int[vertices.Count];
        for (int i = 0; i < _lastVertexPlaneIds.Length; i++)
            _lastVertexPlaneIds[i] = -1;
        Vector3[] vertexNormals = BuildVertexNormals(vertices, triangles);
        int[] assignments = new int[vertices.Count];
        float[] residuals = new float[vertices.Count];
        bool[] moved = new bool[vertices.Count];
        for (int i = 0; i < assignments.Length; i++)
            assignments[i] = -1;

        float beforeSum = 0f;
        for (int vertexIndex = 0; vertexIndex < vertices.Count; vertexIndex++)
        {
            Vector3 point = vertices[vertexIndex];
            Vector3 normal = vertexNormals[vertexIndex];
            int bestTrack = -1;
            float bestScore = float.PositiveInfinity;
            float bestSigned = 0f;
            for (int trackIndex = 0; trackIndex < _tracks.Count; trackIndex++)
            {
                Track track = _tracks[trackIndex];
                if (track.SuppressedById != 0)
                    continue;
                bool usableMature = IsUsableMatureTrack(track);
                if (!usableMature && !allowCandidates)
                    continue;
                if (normal.sqrMagnitude < 0.000001f ||
                    Mathf.Abs(Vector3.Dot(normal, track.Normal)) < settings.VertexNormalDot)
                    continue;
                float signed = Vector3.Dot(track.Normal, point) + track.Offset;
                float distance = Mathf.Abs(signed);
                if (distance > settings.MaxVertexPlaneDistanceMeters)
                    continue;
                Vector3 delta = point - track.Center;
                Vector3 tangent = delta - track.Normal * Vector3.Dot(delta, track.Normal);
                if (tangent.magnitude > track.Radius + settings.PatchPaddingMeters)
                    continue;
                float score = distance + tangent.magnitude * 0.001f + (usableMature ? 0f : 0.01f);
                if (score >= bestScore)
                    continue;
                bestScore = score;
                bestTrack = trackIndex;
                bestSigned = signed;
            }
            if (bestTrack < 0)
                continue;

            Vector3 displacement = -_tracks[bestTrack].Normal * bestSigned;
            if (displacement.magnitude > settings.MaxVertexMoveMeters)
                continue;
            assignments[vertexIndex] = bestTrack;
            residuals[vertexIndex] = Mathf.Abs(bestSigned);
            output[vertexIndex] = point + displacement;
            moved[vertexIndex] = displacement.sqrMagnitude > 0.000000000001f;
            result.AssignedVertexCount++;
            beforeSum += Mathf.Abs(bestSigned);
            result.MaxDisplacementMeters = Mathf.Max(result.MaxDisplacementMeters, displacement.magnitude);
        }

        bool[] rejected = new bool[vertices.Count];
        for (int pass = 0; pass < 2; pass++)
        {
            bool found = false;
            for (int i = 0; i + 2 < triangles.Count; i += 3)
            {
                int a = triangles[i];
                int b = triangles[i + 1];
                int c = triangles[i + 2];
                if (!ValidTriangle(a, b, c, vertices.Count))
                    continue;
                Vector3 oldCross = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
                Vector3 newCross = Vector3.Cross(output[b] - output[a], output[c] - output[a]);
                float oldArea = oldCross.magnitude;
                float newArea = newCross.magnitude;
                float normalDot = oldArea > 0.000001f && newArea > 0.000001f
                    ? Vector3.Dot(oldCross / oldArea, newCross / newArea)
                    : -1f;
                if (newArea >= oldArea * settings.MinTriangleAreaRatio &&
                    normalDot >= settings.MinTriangleNormalDot)
                    continue;
                found = true;
                RejectVertex(a, vertices, output, moved, rejected);
                RejectVertex(b, vertices, output, moved, rejected);
                RejectVertex(c, vertices, output, moved, rejected);
            }
            if (!found)
                break;
        }

        float afterSum = 0f;
        for (int i = 0; i < vertices.Count; i++)
        {
            if (rejected[i])
            {
                if (assignments[i] >= 0)
                    result.RevertedVertexCount++;
                continue;
            }
            if (moved[i])
                result.MovedVertexCount++;
            if (assignments[i] >= 0)
            {
                Track track = _tracks[assignments[i]];
                afterSum += Mathf.Abs(Vector3.Dot(track.Normal, output[i]) + track.Offset);
                _lastVertexPlaneIds[i] = track.Id;
            }
        }
        result.MeanResidualBeforeMeters = result.AssignedVertexCount > 0
            ? beforeSum / result.AssignedVertexCount
            : 0f;
        int retainedAssigned = Mathf.Max(1, result.AssignedVertexCount - result.RevertedVertexCount);
        result.MeanResidualAfterMeters = afterSum / retainedAssigned;
        // A recognized plane is still useful to visualize even when the source mesh
        // already lies close enough to it that no vertex needs a measurable move.
        return result.AssignedVertexCount > 0;
    }

    private static Vector3[] BuildVertexNormals(List<Vector3> vertices, List<int> triangles)
    {
        Vector3[] normals = new Vector3[vertices.Count];
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int a = triangles[i];
            int b = triangles[i + 1];
            int c = triangles[i + 2];
            if (!ValidTriangle(a, b, c, vertices.Count))
                continue;
            Vector3 cross = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
            AccumulateAlignedNormal(normals, a, cross);
            AccumulateAlignedNormal(normals, b, cross);
            AccumulateAlignedNormal(normals, c, cross);
        }
        for (int i = 0; i < normals.Length; i++)
        {
            if (normals[i].sqrMagnitude > 0.000001f)
                normals[i].Normalize();
        }
        return normals;
    }

    private static void AccumulateAlignedNormal(Vector3[] normals, int index, Vector3 contribution)
    {
        if (contribution.sqrMagnitude <= 0.000000000001f)
            return;
        if (normals[index].sqrMagnitude > 0.000000000001f &&
            Vector3.Dot(normals[index], contribution) < 0f)
            contribution = -contribution;
        normals[index] += contribution;
    }

    private static void RejectVertex(
        int index, List<Vector3> original, List<Vector3> output, bool[] moved, bool[] rejected)
    {
        if (index < 0 || index >= output.Count || !moved[index])
            return;
        output[index] = original[index];
        rejected[index] = true;
    }

    private static bool ValidTriangle(int a, int b, int c, int count)
    {
        return a >= 0 && b >= 0 && c >= 0 && a < count && b < count && c < count &&
               a != b && b != c && c != a;
    }

    private static bool Finite(Vector3 value)
    {
        return !(float.IsNaN(value.x) || float.IsNaN(value.y) || float.IsNaN(value.z) ||
                 float.IsInfinity(value.x) || float.IsInfinity(value.y) || float.IsInfinity(value.z));
    }
}

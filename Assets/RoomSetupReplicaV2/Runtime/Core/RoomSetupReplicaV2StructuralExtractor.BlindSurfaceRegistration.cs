using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

/// <summary>
/// Blind final-surface registration for the already selected rectilinear room
/// topology.  This pass never reads a previous simplified-room outline, labels
/// or user regions: it consumes only the frozen final mesh, room axes and the
/// contour-stack polygon produced in the current generation request.
/// </summary>
public sealed partial class RoomSetupReplicaV2StructuralExtractor
{
    private const float RuntimeSurfaceHeightBand = 0.10f;
    private const float RuntimeSurfaceSearchRadius = 0.15f;
    private const float RuntimeSurfaceSearchStep = 0.005f;
    private const float RuntimeSurfaceKernelSigma = 0.018f;
    private const float RuntimeSurfaceIndexBinSize = 0.05f;

    private struct RuntimeSurfaceObservation
    {
        public float cross;
        public float alongMinimum;
        public float alongMaximum;
        public float heightMinimum;
        public float heightMaximum;
        public float weight;
        public int heightFold;
        public int sourceOrder;
    }

    private sealed class RuntimeSurfaceObservationIndex
    {
        public readonly Dictionary<int, List<RuntimeSurfaceObservation>>
            constantU = new();
        public readonly Dictionary<int, List<RuntimeSurfaceObservation>>
            constantV = new();
    }

    private sealed class RuntimeSurfaceEdgeEvidence
    {
        public bool vertical;
        public float original;
        public float target;
        public float length;
        public bool accepted;
    }

    private struct RuntimeSurfaceProfile
    {
        public bool valid;
        public float coordinate;
        public float heightSupport;
        public float alongSupport;
        public float sharpness;
    }

    private struct RuntimeSurfaceRegistrationStats
    {
        public int edgeCount;
        public int acceptedEdges;
        public float medianShift;
        public float maximumShift;
    }

    private bool TryRegisterRuntimeBlindWallSurfaces(
        Vector3[] vertices, List<Vector2> topology,
        float floor, float ceiling,
        CancellationToken token,
        out List<Vector2> registered,
        out RuntimeSurfaceRegistrationStats stats)
    {
        registered = new List<Vector2>(topology);
        stats = new RuntimeSurfaceRegistrationStats
        {
            edgeCount = topology != null ? topology.Count : 0
        };
        if (vertices == null || topology == null || topology.Count < 4 ||
            ceiling - floor < RuntimeMinimumRoomHeight)
            return false;

        RuntimeSurfaceObservationIndex index =
            BuildRuntimeSurfaceObservationIndex(
                vertices, floor, ceiling, token);
        List<RuntimeSurfaceEdgeEvidence> edges = new(topology.Count);
        List<float> acceptedShifts = new();
        for (int i = 0; i < topology.Count; i++)
        {
            token.ThrowIfCancellationRequested();
            RuntimeSurfaceEdgeEvidence edge =
                FitRuntimeBlindSurfaceEdge(
                    index, topology[i], topology[(i + 1) % topology.Count],
                    floor, ceiling);
            edges.Add(edge);
            if (!edge.accepted) continue;
            float shift = Mathf.Abs(edge.target - edge.original);
            acceptedShifts.Add(shift);
            stats.maximumShift = Mathf.Max(stats.maximumShift, shift);
        }

        stats.acceptedEdges = acceptedShifts.Count;
        if (acceptedShifts.Count == 0)
            return false;
        acceptedShifts.Sort();
        int middle = acceptedShifts.Count / 2;
        stats.medianShift = (acceptedShifts.Count & 1) != 0
            ? acceptedShifts[middle]
            : (acceptedShifts[middle - 1] + acceptedShifts[middle]) * 0.5f;

        List<Vector2> candidate = new(topology.Count);
        for (int i = 0; i < topology.Count; i++)
        {
            RuntimeSurfaceEdgeEvidence previous =
                edges[(i - 1 + edges.Count) % edges.Count];
            RuntimeSurfaceEdgeEvidence current = edges[i];
            if (previous.vertical == current.vertical)
            {
                // A malformed/non-rectilinear junction is not repaired here.
                candidate.Add(topology[i]);
                continue;
            }
            candidate.Add(previous.vertical
                ? new Vector2(previous.target, current.target)
                : new Vector2(current.target, previous.target));
        }

        float beforeArea = Mathf.Abs(RuntimePolygonArea(topology));
        float afterArea = Mathf.Abs(RuntimePolygonArea(candidate));
        bool preservesOrientation =
            RuntimePolygonArea(topology) * RuntimePolygonArea(candidate) > 0f;
        bool safe = preservesOrientation &&
                    !RuntimePolygonHasSelfIntersections(candidate) &&
                    afterArea >= beforeArea * 0.90f &&
                    afterArea <= beforeArea * 1.10f;
        if (!safe)
            return false;

        registered = candidate;
        return true;
    }

    private RuntimeSurfaceEdgeEvidence FitRuntimeBlindSurfaceEdge(
        RuntimeSurfaceObservationIndex index, Vector2 a, Vector2 b,
        float floor, float ceiling)
    {
        Vector2 delta = b - a;
        bool vertical = Mathf.Abs(delta.x) < Mathf.Abs(delta.y);
        float original = vertical
            ? (a.x + b.x) * 0.5f
            : (a.y + b.y) * 0.5f;
        float alongMinimum = vertical
            ? Mathf.Min(a.y, b.y) : Mathf.Min(a.x, b.x);
        float alongMaximum = vertical
            ? Mathf.Max(a.y, b.y) : Mathf.Max(a.x, b.x);
        RuntimeSurfaceEdgeEvidence result = new()
        {
            vertical = vertical,
            original = original,
            target = original,
            length = alongMaximum - alongMinimum,
            accepted = false
        };
        if (result.length < 0.05f)
            return result;

        List<RuntimeSurfaceObservation> observations =
            GatherRuntimeSurfaceObservations(
                index, vertical, original,
                alongMinimum, alongMaximum, floor, ceiling);
        RuntimeSurfaceProfile full = FitRuntimeSurfaceProfile(
            observations, -1, original, alongMinimum, alongMaximum,
            floor, ceiling);
        RuntimeSurfaceProfile fold0 = FitRuntimeSurfaceProfile(
            observations, 0, original, alongMinimum, alongMaximum,
            floor, ceiling);
        RuntimeSurfaceProfile fold1 = FitRuntimeSurfaceProfile(
            observations, 1, original, alongMinimum, alongMaximum,
            floor, ceiling);
        if (!full.valid || !fold0.valid || !fold1.valid)
            return result;
        if (Mathf.Abs(fold0.coordinate - fold1.coordinate) >
            runtimeSurfaceFoldAgreement)
            return result;

        float requiredHeight = result.length >= 0.35f ? 0.32f : 0.48f;
        float requiredAlong = result.length >= 0.35f ? 0.32f : 0.55f;
        if (full.heightSupport < requiredHeight ||
            full.alongSupport < requiredAlong || full.sharpness < 1.18f)
            return result;

        float shift = full.coordinate - original;
        if (Mathf.Abs(shift) > runtimeSurfaceMaximumShift)
            return result;
        result.target = full.coordinate;
        result.accepted = true;
        return result;
    }

    private RuntimeSurfaceObservationIndex BuildRuntimeSurfaceObservationIndex(
        Vector3[] vertices, float floor, float ceiling,
        CancellationToken token)
    {
        RuntimeSurfaceObservationIndex index = new();
        int count = vertices.Length - vertices.Length % 3;
        for (int i = 0; i < count; i += 3)
        {
            if (i % 768 == 0)
                token.ThrowIfCancellationRequested();
            Vector3 a = vertices[i];
            Vector3 b = vertices[i + 1];
            Vector3 c = vertices[i + 2];
            if (!RuntimeFinite(a) || !RuntimeFinite(b) || !RuntimeFinite(c))
                continue;
            Vector3 cross = Vector3.Cross(b - a, c - a);
            float doubleArea = cross.magnitude;
            if (doubleArea <= 0.000001f)
                continue;
            Vector3 normal = cross / doubleArea;
            if (Mathf.Abs(normal.y) > 0.55f)
                continue;
            float heightMinimum = Mathf.Min(a.y, Mathf.Min(b.y, c.y));
            float heightMaximum = Mathf.Max(a.y, Mathf.Max(b.y, c.y));
            if (heightMaximum < floor + 0.20f ||
                heightMinimum > ceiling - 0.08f)
                continue;
            Vector3 center = (a + b + c) / 3f;
            AddRuntimeSurfaceObservationToIndex(
                index.constantU, a, b, c, center, normal, doubleArea,
                _lastAxisA, _lastAxisB, heightMinimum, heightMaximum,
                floor, i / 3);
            AddRuntimeSurfaceObservationToIndex(
                index.constantV, a, b, c, center, normal, doubleArea,
                _lastAxisB, _lastAxisA, heightMinimum, heightMaximum,
                floor, i / 3);
        }
        return index;
    }

    private static void AddRuntimeSurfaceObservationToIndex(
        Dictionary<int, List<RuntimeSurfaceObservation>> buckets,
        Vector3 a, Vector3 b, Vector3 c, Vector3 center,
        Vector3 normal, float doubleArea,
        Vector3 normalAxis, Vector3 tangentAxis,
        float heightMinimum, float heightMaximum,
        float floor, int sourceOrder)
    {
        float alignment = Mathf.Abs(Vector3.Dot(normal, normalAxis));
        if (alignment < 0.78f)
            return;
        float crossCoordinate = Vector3.Dot(center, normalAxis);
        float ta = Vector3.Dot(a, tangentAxis);
        float tb = Vector3.Dot(b, tangentAxis);
        float tc = Vector3.Dot(c, tangentAxis);
        RuntimeSurfaceObservation observation = new()
        {
            cross = crossCoordinate,
            alongMinimum = Mathf.Min(ta, Mathf.Min(tb, tc)),
            alongMaximum = Mathf.Max(ta, Mathf.Max(tb, tc)),
            heightMinimum = heightMinimum,
            heightMaximum = heightMaximum,
            weight = doubleArea * 0.5f * alignment,
            heightFold = Mathf.FloorToInt(
                (center.y - floor) / RuntimeSurfaceHeightBand) & 1,
            sourceOrder = sourceOrder
        };
        int bin = Mathf.FloorToInt(
            crossCoordinate / RuntimeSurfaceIndexBinSize);
        if (!buckets.TryGetValue(bin,
                out List<RuntimeSurfaceObservation> values))
        {
            values = new List<RuntimeSurfaceObservation>();
            buckets.Add(bin, values);
        }
        values.Add(observation);
    }

    private static List<RuntimeSurfaceObservation>
        GatherRuntimeSurfaceObservations(
        RuntimeSurfaceObservationIndex index,
        bool constantU, float coordinate,
        float alongMinimum, float alongMaximum,
        float floor, float ceiling)
    {
        List<RuntimeSurfaceObservation> observations = new();
        Dictionary<int, List<RuntimeSurfaceObservation>> buckets =
            constantU ? index.constantU : index.constantV;
        int firstBin = Mathf.FloorToInt(
            (coordinate - 0.18f) / RuntimeSurfaceIndexBinSize);
        int lastBin = Mathf.FloorToInt(
            (coordinate + 0.18f) / RuntimeSurfaceIndexBinSize);
        for (int bin = firstBin; bin <= lastBin; bin++)
        {
            if (!buckets.TryGetValue(bin,
                    out List<RuntimeSurfaceObservation> values))
                continue;
            for (int i = 0; i < values.Count; i++)
            {
                RuntimeSurfaceObservation observation = values[i];
                if (Mathf.Abs(observation.cross - coordinate) > 0.18f ||
                    observation.alongMaximum < alongMinimum - 0.04f ||
                    observation.alongMinimum > alongMaximum + 0.04f ||
                    observation.heightMaximum < floor + 0.20f ||
                    observation.heightMinimum > ceiling - 0.08f)
                    continue;
                observations.Add(observation);
            }
        }
        // Restore source triangle order so floating-point accumulation and
        // therefore the selected registration coordinate stay equivalent to
        // the former full scan.
        observations.Sort((left, right) =>
            left.sourceOrder.CompareTo(right.sourceOrder));
        return observations;
    }

    private static RuntimeSurfaceProfile FitRuntimeSurfaceProfile(
        List<RuntimeSurfaceObservation> observations, int requestedFold,
        float original, float alongMinimum, float alongMaximum,
        float floor, float ceiling)
    {
        int selectedCount = 0;
        for (int i = 0; i < observations.Count; i++)
            if (requestedFold < 0 ||
                observations[i].heightFold == requestedFold)
                selectedCount++;
        if (selectedCount < 6)
            return default;

        int candidateCount = Mathf.RoundToInt(
            RuntimeSurfaceSearchRadius * 2f /
            RuntimeSurfaceSearchStep) + 1;
        float[] scores = new float[candidateCount];
        float[] heightSupport = new float[candidateCount];
        float[] alongSupport = new float[candidateCount];
        int heightBinCount = Mathf.Max(1, Mathf.CeilToInt(
            (ceiling - floor) / RuntimeSurfaceHeightBand));
        int alongBinCount = Mathf.Max(1, Mathf.CeilToInt(
            (alongMaximum - alongMinimum) / RuntimeSurfaceHeightBand));
        float sigmaSquared = RuntimeSurfaceKernelSigma *
                             RuntimeSurfaceKernelSigma;
        bool[] occupiedHeight = new bool[heightBinCount];
        bool[] occupiedAlong = new bool[alongBinCount];

        for (int candidateIndex = 0;
             candidateIndex < candidateCount; candidateIndex++)
        {
            float candidate = original - RuntimeSurfaceSearchRadius +
                              candidateIndex * RuntimeSurfaceSearchStep;
            float density = 0f;
            Array.Clear(occupiedHeight, 0, occupiedHeight.Length);
            Array.Clear(occupiedAlong, 0, occupiedAlong.Length);
            for (int i = 0; i < observations.Count; i++)
            {
                RuntimeSurfaceObservation observation = observations[i];
                if (requestedFold >= 0 &&
                    observation.heightFold != requestedFold)
                    continue;
                float residual = observation.cross - candidate;
                density += observation.weight * Mathf.Exp(
                    -0.5f * residual * residual / sigmaSquared);
                if (Mathf.Abs(residual) > 0.032f)
                    continue;
                int firstHeightBin = Mathf.Clamp(Mathf.FloorToInt(
                    (observation.heightMinimum - floor) /
                    RuntimeSurfaceHeightBand), 0, heightBinCount - 1);
                int lastHeightBin = Mathf.Clamp(Mathf.FloorToInt(
                    (observation.heightMaximum - floor) /
                    RuntimeSurfaceHeightBand), 0, heightBinCount - 1);
                for (int bin = firstHeightBin; bin <= lastHeightBin; bin++)
                    occupiedHeight[bin] = true;
                int firstAlongBin = Mathf.Clamp(Mathf.FloorToInt(
                    (observation.alongMinimum - alongMinimum) /
                    RuntimeSurfaceHeightBand), 0, alongBinCount - 1);
                int lastAlongBin = Mathf.Clamp(Mathf.FloorToInt(
                    (observation.alongMaximum - alongMinimum) /
                    RuntimeSurfaceHeightBand), 0, alongBinCount - 1);
                for (int bin = firstAlongBin; bin <= lastAlongBin; bin++)
                    occupiedAlong[bin] = true;
            }
            int occupiedHeightCount = 0;
            for (int i = 0; i < occupiedHeight.Length; i++)
                if (occupiedHeight[i]) occupiedHeightCount++;
            int occupiedAlongCount = 0;
            for (int i = 0; i < occupiedAlong.Length; i++)
                if (occupiedAlong[i]) occupiedAlongCount++;
            heightSupport[candidateIndex] =
                occupiedHeightCount / (float)heightBinCount;
            alongSupport[candidateIndex] =
                occupiedAlongCount / (float)alongBinCount;
            scores[candidateIndex] = density *
                (0.25f + 0.75f * heightSupport[candidateIndex]) *
                (0.25f + 0.75f * alongSupport[candidateIndex]);
        }

        int peak = 0;
        List<float> positiveScores = new();
        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] > scores[peak]) peak = i;
            if (scores[i] > 0f) positiveScores.Add(scores[i]);
        }
        if (positiveScores.Count == 0 || scores[peak] <= 0f)
            return default;
        positiveScores.Sort();
        int middle = positiveScores.Count / 2;
        float background = (positiveScores.Count & 1) != 0
            ? positiveScores[middle]
            : (positiveScores[middle - 1] + positiveScores[middle]) * 0.5f;
        return new RuntimeSurfaceProfile
        {
            valid = true,
            coordinate = original - RuntimeSurfaceSearchRadius +
                         peak * RuntimeSurfaceSearchStep,
            heightSupport = heightSupport[peak],
            alongSupport = alongSupport[peak],
            sharpness = scores[peak] / Mathf.Max(background, 0.000000001f)
        };
    }
}

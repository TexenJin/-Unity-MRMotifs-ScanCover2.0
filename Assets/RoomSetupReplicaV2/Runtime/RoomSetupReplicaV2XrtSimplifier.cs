using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Deterministic clean-room implementation of the recovered XRT simplifier
/// envelope. The wrapper constants, directed-edge sampling, batch size,
/// topology/normal rejection and stop conditions come from libmrengine. The
/// metric below is an independently expressed equivalent of the recovered
/// physical-error terms; it intentionally keeps surviving vertices at their
/// measured positions instead of inventing smoothed points.
/// </summary>
internal static class RoomSetupReplicaV2XrtSimplifier
{
    internal const int MinimumTriangles = 100;
    internal const int TargetVertexDivisor = 20;
    internal const int CandidateBatchSize = 16;
    internal const int MaximumFailedBatches = 200;
    internal const uint RandomSeed = 13;
    internal const float MaximumErrorMeters = 0.075f;
    // Constructor 0x60dc20 loads the dword bit patterns 0x80 and 0x20 from
    // libmrengine VA 0x120060 into the two XRT multiplier fields. They are
    // floats, not the decimal values 0.1 and 0.5 found at a different address.
    // Multiplication by the 0.075 m base therefore underflows both optional
    // regularizers to near zero; the active metric is overwhelmingly the
    // accumulated point-to-plane displacement recovered at 0x697180.
    internal const int AreaErrorFactorBits = 0x00000080;
    internal const int ShapeErrorFactorBits = 0x00000020;
    internal const float AreaErrorFactor =
        1.793662034335766e-43f;
    internal const float ShapeErrorFactor =
        4.484155085839415e-44f;
    internal const float AreaErrorMeters =
        MaximumErrorMeters * AreaErrorFactor;
    internal const float ShapeErrorMeters =
        MaximumErrorMeters * ShapeErrorFactor;
    internal const float MinimumNormalDot = 0.1f;

    internal sealed class Input
    {
        public Vector3[] vertices = Array.Empty<Vector3>();
        public int[] triangles = Array.Empty<int>();
        public int[] triangleOwners = Array.Empty<int>();
        public Vector3Int[] ownerChunkIds = Array.Empty<Vector3Int>();
        public Vector3Int[] publicationChunkIds = Array.Empty<Vector3Int>();
        public int resetGeneration;
        public int serial;
    }

    internal sealed class Result
    {
        public Vector3[] vertices = Array.Empty<Vector3>();
        public int[] triangles = Array.Empty<int>();
        public int[] triangleOwners = Array.Empty<int>();
        public Vector3Int[] ownerChunkIds = Array.Empty<Vector3Int>();
        public Vector3Int[] publicationChunkIds = Array.Empty<Vector3Int>();
        public int resetGeneration;
        public int serial;
        public bool applied;
        public int inputVertices;
        public int inputTriangles;
        public int targetVertices;
        public int outputVertices;
        public int outputTriangles;
        public int acceptedCollapses;
        public int failedBatches;
        public int topologyRejectedCandidates;
        public int normalRejectedCandidates;
        public int errorRejectedCandidates;
        public string stopReason = string.Empty;
        public double elapsedMilliseconds;
    }

    private struct Face
    {
        public int a;
        public int b;
        public int c;
        public int owner;
        public bool active;

        public int Vertex(int corner)
        {
            return corner == 0 ? a : corner == 1 ? b : c;
        }

        public bool Contains(int vertex)
        {
            return a == vertex || b == vertex || c == vertex;
        }

        public void Replace(int from, int to)
        {
            if (a == from) a = to;
            if (b == from) b = to;
            if (c == from) c = to;
        }
    }

    private enum Rejection : byte
    {
        None,
        Topology,
        Normal,
        Error
    }

    private sealed class TopologyScratch
    {
        private readonly int[] _marks;
        private int _stamp;

        public TopologyScratch(int vertexCount)
        {
            _marks = new int[vertexCount];
        }

        public int CountCommonNeighbours(
            int remove,
            int keep,
            Face[] faces,
            HashSet<int>[] incident)
        {
            _stamp++;
            if (_stamp == int.MaxValue)
            {
                Array.Clear(_marks, 0, _marks.Length);
                _stamp = 1;
            }
            foreach (int faceIndex in incident[remove])
            {
                Face face = faces[faceIndex];
                if (!face.active) continue;
                Mark(face.a, remove, keep);
                Mark(face.b, remove, keep);
                Mark(face.c, remove, keep);
            }

            int common = 0;
            foreach (int faceIndex in incident[keep])
            {
                Face face = faces[faceIndex];
                if (!face.active) continue;
                common += CountOnce(face.a, remove, keep);
                common += CountOnce(face.b, remove, keep);
                common += CountOnce(face.c, remove, keep);
            }
            return common;
        }

        private void Mark(int vertex, int remove, int keep)
        {
            if (vertex != remove && vertex != keep)
                _marks[vertex] = _stamp;
        }

        private int CountOnce(int vertex, int remove, int keep)
        {
            if (vertex == remove || vertex == keep ||
                _marks[vertex] != _stamp)
                return 0;
            _marks[vertex] = -_stamp;
            return 1;
        }
    }

    internal static Result Run(Input input)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var result = new Result
        {
            resetGeneration = input.resetGeneration,
            serial = input.serial,
            ownerChunkIds = input.ownerChunkIds,
            publicationChunkIds = input.publicationChunkIds,
            inputVertices = input.vertices.Length,
            inputTriangles = input.triangles.Length / 3,
            targetVertices = Math.Max(4,
                input.vertices.Length / TargetVertexDivisor)
        };

        if (result.inputTriangles < MinimumTriangles ||
            input.vertices.Length < 4)
        {
            CopyInput(input, result);
            result.stopReason = "BelowMinimumInput";
            watch.Stop();
            result.elapsedMilliseconds = watch.Elapsed.TotalMilliseconds;
            return result;
        }

        int faceCount = result.inputTriangles;
        var faces = new Face[faceCount];
        var incident = new HashSet<int>[input.vertices.Length];
        var vertexActive = new bool[input.vertices.Length];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int index = faceIndex * 3;
            int a = input.triangles[index];
            int b = input.triangles[index + 1];
            int c = input.triangles[index + 2];
            if (!ValidVertex(a, input.vertices.Length) ||
                !ValidVertex(b, input.vertices.Length) ||
                !ValidVertex(c, input.vertices.Length) ||
                a == b || b == c || c == a)
                continue;
            faces[faceIndex] = new Face
            {
                a = a,
                b = b,
                c = c,
                owner = faceIndex < input.triangleOwners.Length
                    ? input.triangleOwners[faceIndex]
                    : 0,
                active = true
            };
            AddIncident(incident, a, faceIndex);
            AddIncident(incident, b, faceIndex);
            AddIncident(incident, c, faceIndex);
            vertexActive[a] = true;
            vertexActive[b] = true;
            vertexActive[c] = true;
        }

        int activeVertices = 0;
        for (int i = 0; i < vertexActive.Length; i++)
            if (vertexActive[i]) activeVertices++;
        result.targetVertices = Math.Max(4,
            activeVertices / TargetVertexDivisor);

        var random = new Mt19937(RandomSeed);
        var topologyScratch = new TopologyScratch(input.vertices.Length);
        var sampledCorners = new int[CandidateBatchSize];
        int cornerCount = faceCount * 3;
        int failedBatches = 0;
        int accepted = 0;

        while (activeVertices > result.targetVertices &&
               failedBatches < MaximumFailedBatches)
        {
            int sampled = 0;
            int attempts = 0;
            int maximumAttempts = Math.Max(64, CandidateBatchSize * 64);
            while (sampled < CandidateBatchSize &&
                   attempts++ < maximumAttempts)
            {
                int corner = random.NextInt(cornerCount);
                int faceIndex = corner / 3;
                if (!faces[faceIndex].active ||
                    Contains(sampledCorners, sampled, corner))
                    continue;
                sampledCorners[sampled++] = corner;
            }

            if (sampled == 0)
            {
                failedBatches++;
                continue;
            }

            int bestSlot = -1;
            float bestCost = float.MaxValue;
            for (int slot = 0; slot < sampled; slot++)
            {
                int corner = sampledCorners[slot];
                Face face = faces[corner / 3];
                int localCorner = corner % 3;
                int remove = face.Vertex(localCorner);
                int keep = face.Vertex((localCorner + 1) % 3);
                float cost = ComputeCollapseCost(
                    remove, keep, input.vertices, faces, incident,
                    vertexActive, topologyScratch,
                    out Rejection rejection);
                if (rejection == Rejection.Topology)
                    result.topologyRejectedCandidates++;
                else if (rejection == Rejection.Normal)
                    result.normalRejectedCandidates++;
                else if (rejection == Rejection.Error)
                    result.errorRejectedCandidates++;
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestSlot = slot;
                }
            }

            if (bestSlot < 0 || !(bestCost < MaximumErrorMeters))
            {
                failedBatches++;
                continue;
            }

            int bestCorner = sampledCorners[bestSlot];
            Face bestFace = faces[bestCorner / 3];
            int bestLocal = bestCorner % 3;
            int removeVertex = bestFace.Vertex(bestLocal);
            int keepVertex = bestFace.Vertex((bestLocal + 1) % 3);
            if (!ApplyCollapse(
                    removeVertex, keepVertex, faces, incident,
                    vertexActive))
            {
                failedBatches++;
                result.topologyRejectedCandidates++;
                continue;
            }
            activeVertices--;
            accepted++;
        }

        result.applied = true;
        result.acceptedCollapses = accepted;
        result.failedBatches = failedBatches;
        result.stopReason = activeVertices <= result.targetVertices
            ? "TargetVertexCount"
            : "MaximumFailedBatches";
        Compact(input.vertices, faces, vertexActive, result);
        // A short, nearly planar tape can make every directed collapse look
        // geometrically free. Never replace a non-empty cleaned Front with an
        // empty result while the exact native terminal topology rule remains
        // unresolved. This is a publication fail-safe, not a metric change.
        if (result.inputTriangles > 0 && result.outputTriangles == 0)
        {
            CopyInput(input, result);
            result.applied = false;
            result.stopReason = "EmptyOutputRejected";
        }
        watch.Stop();
        result.elapsedMilliseconds = watch.Elapsed.TotalMilliseconds;
        return result;
    }

    private static float ComputeCollapseCost(
        int remove,
        int keep,
        Vector3[] vertices,
        Face[] faces,
        HashSet<int>[] incident,
        bool[] vertexActive,
        TopologyScratch topologyScratch,
        out Rejection rejection)
    {
        rejection = Rejection.Topology;
        if (remove == keep ||
            !ValidVertex(remove, vertices.Length) ||
            !ValidVertex(keep, vertices.Length) ||
            !vertexActive[remove] || !vertexActive[keep] ||
            incident[remove] == null || incident[keep] == null)
            return float.MaxValue;

        int sharedFaces = 0;
        foreach (int faceIndex in incident[remove])
        {
            Face face = faces[faceIndex];
            if (face.active && face.Contains(keep))
                sharedFaces++;
        }
        if (sharedFaces < 1 || sharedFaces > 2 ||
            topologyScratch.CountCommonNeighbours(
                remove, keep, faces, incident) != sharedFaces)
            return float.MaxValue;

        float cost = 0f;
        Vector3 displacement = vertices[keep] - vertices[remove];
        foreach (int faceIndex in incident[remove])
        {
            Face face = faces[faceIndex];
            if (!face.active || face.Contains(keep))
                continue;

            Vector3 oldA = vertices[face.a];
            Vector3 oldB = vertices[face.b];
            Vector3 oldC = vertices[face.c];
            Vector3 newA = face.a == remove ? vertices[keep] : oldA;
            Vector3 newB = face.b == remove ? vertices[keep] : oldB;
            Vector3 newC = face.c == remove ? vertices[keep] : oldC;
            Vector3 oldCross = Vector3.Cross(oldB - oldA, oldC - oldA);
            Vector3 newCross = Vector3.Cross(newB - newA, newC - newA);
            float oldLength = oldCross.magnitude;
            float newLength = newCross.magnitude;
            if (!(oldLength > 0.0000001f) ||
                !(newLength > 0.0000001f))
            {
                rejection = Rejection.Normal;
                return float.MaxValue;
            }

            float normalDot = Vector3.Dot(oldCross, newCross) /
                              (oldLength * newLength);
            if (!(normalDot >= MinimumNormalDot))
            {
                rejection = Rejection.Normal;
                return float.MaxValue;
            }

            Vector3 oldNormal = oldCross / oldLength;
            float pointPlane = Mathf.Abs(
                Vector3.Dot(displacement, oldNormal));
            float oldEdgeEnergy = EdgeEnergy(oldA, oldB, oldC);
            float newEdgeEnergy = EdgeEnergy(newA, newB, newC);
            if (!(oldEdgeEnergy > 0.0000001f) ||
                !(newEdgeEnergy > 0.0000001f))
            {
                rejection = Rejection.Normal;
                return float.MaxValue;
            }

            float oldQuality = (oldLength * 0.5f) / oldEdgeEnergy;
            float newQuality = (newLength * 0.5f) / newEdgeEnergy;
            float qualityRatio = oldQuality /
                                 Mathf.Max(0.0000001f, newQuality);
            cost += pointPlane +
                    newLength * 0.5f * AreaErrorMeters +
                    qualityRatio * ShapeErrorMeters;
            if (!(cost < MaximumErrorMeters))
            {
                rejection = Rejection.Error;
                return cost;
            }
        }

        // Native 0x797180 adds a terminal distance for every open fan side:
        // the surviving endpoint must remain close to the infinite line of
        // each boundary edge incident to the removed vertex. This term is
        // essential on open room surfaces; without it, coplanar boundary
        // vertices have zero point-to-plane cost and collapse into very long
        // triangles even though the surface plane itself is unchanged.
        foreach (int faceIndex in incident[remove])
        {
            Face face = faces[faceIndex];
            if (!face.active)
                continue;
            for (int corner = 0; corner < 3; corner++)
            {
                int neighbour = face.Vertex(corner);
                if (neighbour == remove)
                    continue;
                int edgeFaces = 0;
                foreach (int adjacentIndex in incident[remove])
                {
                    Face adjacent = faces[adjacentIndex];
                    if (adjacent.active && adjacent.Contains(neighbour) &&
                        ++edgeFaces > 1)
                        break;
                }
                if (edgeFaces != 1)
                    continue;
                Vector3 boundary = vertices[neighbour] - vertices[remove];
                float boundaryLength = boundary.magnitude;
                if (!(boundaryLength > 0.0000001f))
                {
                    rejection = Rejection.Normal;
                    return float.MaxValue;
                }
                Vector3 toKeep = vertices[keep] - vertices[remove];
                cost += Vector3.Cross(toKeep, boundary).magnitude /
                        boundaryLength;
                if (!(cost < MaximumErrorMeters))
                {
                    rejection = Rejection.Error;
                    return cost;
                }
            }
        }

        rejection = cost < MaximumErrorMeters
            ? Rejection.None
            : Rejection.Error;
        return cost;
    }

    private static bool ApplyCollapse(
        int remove,
        int keep,
        Face[] faces,
        HashSet<int>[] incident,
        bool[] vertexActive)
    {
        if (incident[remove] == null || incident[keep] == null)
            return false;
        int[] affected = new int[incident[remove].Count];
        incident[remove].CopyTo(affected);
        for (int i = 0; i < affected.Length; i++)
        {
            int faceIndex = affected[i];
            Face face = faces[faceIndex];
            if (!face.active) continue;
            RemoveIncident(incident, face.a, faceIndex);
            RemoveIncident(incident, face.b, faceIndex);
            RemoveIncident(incident, face.c, faceIndex);
            if (face.Contains(keep))
            {
                face.active = false;
                faces[faceIndex] = face;
                continue;
            }
            face.Replace(remove, keep);
            if (face.a == face.b || face.b == face.c || face.c == face.a)
            {
                face.active = false;
                faces[faceIndex] = face;
                continue;
            }
            faces[faceIndex] = face;
            AddIncident(incident, face.a, faceIndex);
            AddIncident(incident, face.b, faceIndex);
            AddIncident(incident, face.c, faceIndex);
        }
        incident[remove].Clear();
        vertexActive[remove] = false;
        return true;
    }

    private static void Compact(
        Vector3[] sourceVertices,
        Face[] faces,
        bool[] vertexActive,
        Result result)
    {
        var remap = new int[sourceVertices.Length];
        for (int i = 0; i < remap.Length; i++)
            remap[i] = -1;
        var vertices = new List<Vector3>(sourceVertices.Length);
        var triangles = new List<int>(faces.Length * 3);
        var owners = new List<int>(faces.Length);
        for (int faceIndex = 0; faceIndex < faces.Length; faceIndex++)
        {
            Face face = faces[faceIndex];
            if (!face.active) continue;
            int a = RemapVertex(face.a, sourceVertices, remap, vertices);
            int b = RemapVertex(face.b, sourceVertices, remap, vertices);
            int c = RemapVertex(face.c, sourceVertices, remap, vertices);
            if (a == b || b == c || c == a) continue;
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
            owners.Add(face.owner);
        }
        result.vertices = vertices.ToArray();
        result.triangles = triangles.ToArray();
        result.triangleOwners = owners.ToArray();
        result.outputVertices = result.vertices.Length;
        result.outputTriangles = result.triangles.Length / 3;
    }

    private static int RemapVertex(
        int source,
        Vector3[] sourceVertices,
        int[] remap,
        List<Vector3> vertices)
    {
        int mapped = remap[source];
        if (mapped >= 0) return mapped;
        mapped = vertices.Count;
        remap[source] = mapped;
        vertices.Add(sourceVertices[source]);
        return mapped;
    }

    private static void CopyInput(Input input, Result result)
    {
        result.vertices = (Vector3[])input.vertices.Clone();
        result.triangles = (int[])input.triangles.Clone();
        result.triangleOwners = (int[])input.triangleOwners.Clone();
        result.outputVertices = result.vertices.Length;
        result.outputTriangles = result.triangles.Length / 3;
    }

    private static float EdgeEnergy(Vector3 a, Vector3 b, Vector3 c)
    {
        return (a - b).sqrMagnitude +
               (b - c).sqrMagnitude +
               (c - a).sqrMagnitude;
    }

    private static bool Contains(int[] values, int count, int value)
    {
        for (int i = 0; i < count; i++)
            if (values[i] == value) return true;
        return false;
    }

    private static bool ValidVertex(int value, int count)
    {
        return value >= 0 && value < count;
    }

    private static void AddIncident(
        HashSet<int>[] incident, int vertex, int face)
    {
        HashSet<int> set = incident[vertex];
        if (set == null)
        {
            set = new HashSet<int>();
            incident[vertex] = set;
        }
        set.Add(face);
    }

    private static void RemoveIncident(
        HashSet<int>[] incident, int vertex, int face)
    {
        incident[vertex]?.Remove(face);
    }

    /// <summary>Standard 32-bit MT19937 with the recovered seed.</summary>
    private sealed class Mt19937
    {
        private const int StateSize = 624;
        private const int MiddleWord = 397;
        private readonly uint[] _state = new uint[StateSize];
        private int _index = StateSize;

        public Mt19937(uint seed)
        {
            _state[0] = seed;
            for (uint i = 1; i < StateSize; i++)
                _state[i] = unchecked(
                    0x6c078965u *
                    (_state[i - 1] ^ (_state[i - 1] >> 30)) + i);
        }

        public int NextInt(int exclusiveMaximum)
        {
            if (exclusiveMaximum <= 1) return 0;
            uint range = (uint)exclusiveMaximum;
            uint limit = uint.MaxValue - uint.MaxValue % range;
            uint value;
            do value = NextUInt(); while (value >= limit);
            return (int)(value % range);
        }

        private uint NextUInt()
        {
            if (_index >= StateSize) Twist();
            uint value = _state[_index++];
            value ^= value >> 11;
            value ^= (value << 7) & 0x9d2c5680u;
            value ^= (value << 15) & 0xefc60000u;
            value ^= value >> 18;
            return value;
        }

        private void Twist()
        {
            for (int i = 0; i < StateSize; i++)
            {
                uint combined = (_state[i] & 0x80000000u) |
                                (_state[(i + 1) % StateSize] & 0x7fffffffu);
                uint next = _state[(i + MiddleWord) % StateSize] ^
                            (combined >> 1);
                if ((combined & 1u) != 0) next ^= 0x9908b0dfu;
                _state[i] = next;
            }
            _index = 0;
        }
    }
}

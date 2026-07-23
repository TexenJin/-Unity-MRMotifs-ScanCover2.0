using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Conservative display-mesh regularization inspired by feature-constrained isotropic remeshing.
/// It deliberately keeps the triangle index buffer unchanged: only planar interior vertices may move.
/// Boundary, non-manifold and crease vertices are treated as hard constraints.
/// </summary>
public static class ScanCoverFeaturePreservingMeshOptimizer
{
    public struct Settings
    {
        public int Iterations;
        public float MinPlanarNormalDot;
        public float FeatureAngleDegrees;
        public float TangentialRelaxation;
        public float PlaneProjectionStrength;
        public float MaxVertexMoveMeters;
        public float MinTriangleAreaRatio;
        public float MinTriangleNormalDot;
        public float MaxMeanTriangleQualityLoss;
    }

    public struct Result
    {
        public bool Applied;
        public bool AtomicRollback;
        public int UniqueTriangleCount;
        public int BoundaryVertexCount;
        public int CreaseVertexCount;
        public int EligibleVertexCount;
        public int MovedVertexCount;
        public int RevertedVertexCount;
        public float MeanTriangleQualityBefore;
        public float MeanTriangleQualityAfter;
        public float MeanPlaneResidualBeforeMeters;
        public float MeanPlaneResidualAfterMeters;
        public float RmsDisplacementMeters;
        public float MaxDisplacementMeters;
    }

    private struct Face
    {
        public int A;
        public int B;
        public int C;
        public Vector3 Normal;
        public float DoubleArea;
    }

    private sealed class EdgeFaces
    {
        public int First = -1;
        public int Second = -1;
        public int Count;
    }

    public static bool Optimize(
        List<Vector3> vertices,
        List<int> triangles,
        Settings settings,
        out Result result)
    {
        result = new Result();
        if (vertices == null || triangles == null || vertices.Count < 4 || triangles.Count < 6)
            return false;

        Vector3[] original = vertices.ToArray();
        settings.Iterations = Mathf.Clamp(settings.Iterations, 1, 4);
        settings.MinPlanarNormalDot = Mathf.Clamp(settings.MinPlanarNormalDot, 0.5f, 0.9999f);
        settings.FeatureAngleDegrees = Mathf.Clamp(settings.FeatureAngleDegrees, 5f, 85f);
        settings.TangentialRelaxation = Mathf.Clamp01(settings.TangentialRelaxation);
        settings.PlaneProjectionStrength = Mathf.Clamp01(settings.PlaneProjectionStrength);
        settings.MaxVertexMoveMeters = Mathf.Max(0.0001f, settings.MaxVertexMoveMeters);
        settings.MinTriangleAreaRatio = Mathf.Clamp(settings.MinTriangleAreaRatio, 0.1f, 0.95f);
        settings.MinTriangleNormalDot = Mathf.Clamp(settings.MinTriangleNormalDot, -0.5f, 0.999f);
        settings.MaxMeanTriangleQualityLoss = Mathf.Clamp(settings.MaxMeanTriangleQualityLoss, 0f, 0.1f);

        bool qualityInitialized = false;
        float residualBeforeSum = 0f;
        float residualAfterSum = 0f;
        int residualCount = 0;
        int totalEligible = 0;
        int totalReverted = 0;

        for (int iteration = 0; iteration < settings.Iterations; iteration++)
        {
            Result iterationResult;
            bool applied = OptimizeIteration(vertices, triangles, settings, out iterationResult);
            result.UniqueTriangleCount = iterationResult.UniqueTriangleCount;
            result.BoundaryVertexCount = iterationResult.BoundaryVertexCount;
            result.CreaseVertexCount = iterationResult.CreaseVertexCount;
            totalEligible += iterationResult.EligibleVertexCount;
            totalReverted += iterationResult.RevertedVertexCount;
            residualBeforeSum += iterationResult.MeanPlaneResidualBeforeMeters * iterationResult.EligibleVertexCount;
            residualAfterSum += iterationResult.MeanPlaneResidualAfterMeters * iterationResult.EligibleVertexCount;
            residualCount += iterationResult.EligibleVertexCount;

            if (!qualityInitialized)
            {
                result.MeanTriangleQualityBefore = iterationResult.MeanTriangleQualityBefore;
                qualityInitialized = true;
            }
            result.MeanTriangleQualityAfter = iterationResult.MeanTriangleQualityAfter;

            if (iterationResult.AtomicRollback)
            {
                result.AtomicRollback = true;
                break;
            }
            if (!applied)
                break;
        }

        float displacementSquaredSum = 0f;
        float maxDisplacement = 0f;
        int finalMovedCount = 0;
        for (int i = 0; i < vertices.Count && i < original.Length; i++)
        {
            float displacement = Vector3.Distance(original[i], vertices[i]);
            if (displacement <= 0.000001f)
                continue;
            finalMovedCount++;
            displacementSquaredSum += displacement * displacement;
            maxDisplacement = Mathf.Max(maxDisplacement, displacement);
        }

        result.EligibleVertexCount = totalEligible;
        result.MovedVertexCount = finalMovedCount;
        result.RevertedVertexCount = totalReverted;
        result.MeanPlaneResidualBeforeMeters = residualCount > 0 ? residualBeforeSum / residualCount : 0f;
        result.MeanPlaneResidualAfterMeters = residualCount > 0 ? residualAfterSum / residualCount : 0f;
        result.RmsDisplacementMeters = finalMovedCount > 0
            ? Mathf.Sqrt(displacementSquaredSum / finalMovedCount)
            : 0f;
        result.MaxDisplacementMeters = maxDisplacement;
        result.Applied = finalMovedCount > 0;
        return result.Applied;
    }

    private static bool OptimizeIteration(
        List<Vector3> vertices,
        List<int> triangles,
        Settings settings,
        out Result result)
    {
        result = new Result();
        List<Face> faces = BuildUniqueFaces(vertices, triangles);
        result.UniqueTriangleCount = faces.Count;
        if (faces.Count < 2)
            return false;

        int vertexCount = vertices.Count;
        HashSet<int>[] neighbors = new HashSet<int>[vertexCount];
        List<int>[] incidentFaces = new List<int>[vertexCount];
        Dictionary<ulong, EdgeFaces> edges = new Dictionary<ulong, EdgeFaces>(faces.Count * 2);
        for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
        {
            Face face = faces[faceIndex];
            AddNeighbor(neighbors, face.A, face.B);
            AddNeighbor(neighbors, face.B, face.C);
            AddNeighbor(neighbors, face.C, face.A);
            AddIncidentFace(incidentFaces, face.A, faceIndex);
            AddIncidentFace(incidentFaces, face.B, faceIndex);
            AddIncidentFace(incidentFaces, face.C, faceIndex);
            AddEdge(edges, face.A, face.B, faceIndex);
            AddEdge(edges, face.B, face.C, faceIndex);
            AddEdge(edges, face.C, face.A, faceIndex);
        }

        bool[] constrained = new bool[vertexCount];
        bool[] boundary = new bool[vertexCount];
        bool[] crease = new bool[vertexCount];
        float creaseNormalDot = Mathf.Cos(settings.FeatureAngleDegrees * Mathf.Deg2Rad);
        foreach (KeyValuePair<ulong, EdgeFaces> pair in edges)
        {
            DecodeEdge(pair.Key, out int a, out int b);
            EdgeFaces edge = pair.Value;
            if (edge.Count != 2 || edge.First < 0 || edge.Second < 0)
            {
                constrained[a] = true;
                constrained[b] = true;
                boundary[a] = true;
                boundary[b] = true;
                continue;
            }

            float normalDot = Mathf.Abs(Vector3.Dot(faces[edge.First].Normal, faces[edge.Second].Normal));
            if (normalDot < creaseNormalDot)
            {
                constrained[a] = true;
                constrained[b] = true;
                crease[a] = true;
                crease[b] = true;
            }
        }

        for (int i = 0; i < vertexCount; i++)
        {
            if (boundary[i])
                result.BoundaryVertexCount++;
            if (crease[i])
                result.CreaseVertexCount++;
        }

        Vector3[] before = vertices.ToArray();
        Vector3[] candidate = vertices.ToArray();
        bool[] moved = new bool[vertexCount];
        float residualBeforeSum = 0f;
        float residualAfterSum = 0f;

        for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
        {
            if (constrained[vertexIndex] || neighbors[vertexIndex] == null || neighbors[vertexIndex].Count < 3 ||
                incidentFaces[vertexIndex] == null || incidentFaces[vertexIndex].Count < 3)
            {
                continue;
            }

            List<int> localFaces = incidentFaces[vertexIndex];
            Vector3 referenceNormal = faces[localFaces[0]].Normal;
            Vector3 normalSum = Vector3.zero;
            Vector3 planeCenterSum = Vector3.zero;
            float areaSum = 0f;
            for (int i = 0; i < localFaces.Count; i++)
            {
                Face face = faces[localFaces[i]];
                Vector3 alignedNormal = Vector3.Dot(referenceNormal, face.Normal) < 0f ? -face.Normal : face.Normal;
                float areaWeight = Mathf.Max(0.0000001f, face.DoubleArea);
                normalSum += alignedNormal * areaWeight;
                planeCenterSum += ((before[face.A] + before[face.B] + before[face.C]) / 3f) * areaWeight;
                areaSum += areaWeight;
            }

            if (normalSum.sqrMagnitude < 0.00000001f || areaSum <= 0f)
                continue;
            Vector3 planeNormal = normalSum.normalized;
            bool planar = true;
            for (int i = 0; i < localFaces.Count; i++)
            {
                if (Mathf.Abs(Vector3.Dot(planeNormal, faces[localFaces[i]].Normal)) < settings.MinPlanarNormalDot)
                {
                    planar = false;
                    break;
                }
            }
            if (!planar)
                continue;

            Vector3 planeCenter = planeCenterSum / areaSum;
            Vector3 neighborCenter = Vector3.zero;
            foreach (int neighbor in neighbors[vertexIndex])
                neighborCenter += before[neighbor];
            neighborCenter /= neighbors[vertexIndex].Count;

            Vector3 tangentialDelta = neighborCenter - before[vertexIndex];
            tangentialDelta -= planeNormal * Vector3.Dot(tangentialDelta, planeNormal);
            float planeResidual = Vector3.Dot(before[vertexIndex] - planeCenter, planeNormal);
            Vector3 planeDelta = -planeNormal * planeResidual;
            Vector3 displacement = tangentialDelta * settings.TangentialRelaxation +
                                   planeDelta * settings.PlaneProjectionStrength;
            displacement = Vector3.ClampMagnitude(displacement, settings.MaxVertexMoveMeters);
            if (displacement.sqrMagnitude <= 0.000000000001f)
                continue;

            candidate[vertexIndex] = before[vertexIndex] + displacement;
            moved[vertexIndex] = true;
            result.EligibleVertexCount++;
            residualBeforeSum += Mathf.Abs(planeResidual);
            residualAfterSum += Mathf.Abs(Vector3.Dot(candidate[vertexIndex] - planeCenter, planeNormal));
        }

        if (result.EligibleVertexCount <= 0)
        {
            result.MeanTriangleQualityBefore = MeanTriangleQuality(before, faces);
            result.MeanTriangleQualityAfter = result.MeanTriangleQualityBefore;
            return false;
        }

        bool[] rejected = new bool[vertexCount];
        for (int pass = 0; pass < 2; pass++)
        {
            bool foundInvalid = false;
            for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
            {
                Face face = faces[faceIndex];
                Vector3 newCross = Vector3.Cross(
                    candidate[face.B] - candidate[face.A],
                    candidate[face.C] - candidate[face.A]);
                float newDoubleArea = newCross.magnitude;
                float normalDot = newDoubleArea > 0.0000001f
                    ? Vector3.Dot(face.Normal, newCross / newDoubleArea)
                    : -1f;
                if (newDoubleArea >= face.DoubleArea * settings.MinTriangleAreaRatio &&
                    normalDot >= settings.MinTriangleNormalDot)
                {
                    continue;
                }

                foundInvalid = true;
                RejectMovedVertex(face.A, moved, rejected);
                RejectMovedVertex(face.B, moved, rejected);
                RejectMovedVertex(face.C, moved, rejected);
            }

            if (!foundInvalid)
                break;
            for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
            {
                if (rejected[vertexIndex])
                    candidate[vertexIndex] = before[vertexIndex];
            }
        }

        int movedAfterValidation = 0;
        for (int i = 0; i < vertexCount; i++)
        {
            if (moved[i] && rejected[i])
                result.RevertedVertexCount++;
            if (moved[i] && !rejected[i])
                movedAfterValidation++;
        }

        result.MeanTriangleQualityBefore = MeanTriangleQuality(before, faces);
        result.MeanTriangleQualityAfter = MeanTriangleQuality(candidate, faces);
        result.MeanPlaneResidualBeforeMeters = residualBeforeSum / Mathf.Max(1, result.EligibleVertexCount);
        result.MeanPlaneResidualAfterMeters = residualAfterSum / Mathf.Max(1, result.EligibleVertexCount);
        if (movedAfterValidation <= 0)
            return false;

        if (result.MeanTriangleQualityAfter + settings.MaxMeanTriangleQualityLoss < result.MeanTriangleQualityBefore)
        {
            result.AtomicRollback = true;
            result.MeanTriangleQualityAfter = result.MeanTriangleQualityBefore;
            return false;
        }

        for (int i = 0; i < vertexCount; i++)
        {
            if (moved[i] && !rejected[i])
                vertices[i] = candidate[i];
        }
        result.MovedVertexCount = movedAfterValidation;
        result.Applied = true;
        return true;
    }

    private static List<Face> BuildUniqueFaces(List<Vector3> vertices, List<int> triangles)
    {
        List<Face> faces = new List<Face>(triangles.Count / 6);
        HashSet<Vector3Int> uniqueKeys = new HashSet<Vector3Int>();
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int a = triangles[i];
            int b = triangles[i + 1];
            int c = triangles[i + 2];
            if (a < 0 || b < 0 || c < 0 || a >= vertices.Count || b >= vertices.Count || c >= vertices.Count ||
                a == b || b == c || c == a)
            {
                continue;
            }

            Vector3Int key = SortedTriangleKey(a, b, c);
            if (!uniqueKeys.Add(key))
                continue;
            Vector3 cross = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
            float doubleArea = cross.magnitude;
            if (doubleArea <= 0.0000001f)
                continue;
            faces.Add(new Face
            {
                A = a,
                B = b,
                C = c,
                Normal = cross / doubleArea,
                DoubleArea = doubleArea
            });
        }
        return faces;
    }

    private static float MeanTriangleQuality(IList<Vector3> vertices, List<Face> faces)
    {
        if (faces == null || faces.Count == 0)
            return 0f;
        float qualitySum = 0f;
        int count = 0;
        for (int i = 0; i < faces.Count; i++)
        {
            Face face = faces[i];
            Vector3 ab = vertices[face.B] - vertices[face.A];
            Vector3 bc = vertices[face.C] - vertices[face.B];
            Vector3 ca = vertices[face.A] - vertices[face.C];
            float denominator = ab.sqrMagnitude + bc.sqrMagnitude + ca.sqrMagnitude;
            if (denominator <= 0.0000001f)
                continue;
            float doubleArea = Vector3.Cross(ab, vertices[face.C] - vertices[face.A]).magnitude;
            qualitySum += 2f * Mathf.Sqrt(3f) * doubleArea / denominator;
            count++;
        }
        return count > 0 ? qualitySum / count : 0f;
    }

    private static void AddNeighbor(HashSet<int>[] neighbors, int a, int b)
    {
        if (neighbors[a] == null)
            neighbors[a] = new HashSet<int>();
        if (neighbors[b] == null)
            neighbors[b] = new HashSet<int>();
        neighbors[a].Add(b);
        neighbors[b].Add(a);
    }

    private static void AddIncidentFace(List<int>[] incidentFaces, int vertex, int face)
    {
        if (incidentFaces[vertex] == null)
            incidentFaces[vertex] = new List<int>(6);
        incidentFaces[vertex].Add(face);
    }

    private static void AddEdge(Dictionary<ulong, EdgeFaces> edges, int a, int b, int faceIndex)
    {
        ulong key = EdgeKey(a, b);
        if (!edges.TryGetValue(key, out EdgeFaces edge))
        {
            edge = new EdgeFaces();
            edges[key] = edge;
        }
        if (edge.Count == 0)
            edge.First = faceIndex;
        else if (edge.Count == 1)
            edge.Second = faceIndex;
        edge.Count++;
    }

    private static void RejectMovedVertex(int vertex, bool[] moved, bool[] rejected)
    {
        if (vertex >= 0 && vertex < moved.Length && moved[vertex])
            rejected[vertex] = true;
    }

    private static Vector3Int SortedTriangleKey(int a, int b, int c)
    {
        if (a > b)
        {
            int swap = a;
            a = b;
            b = swap;
        }
        if (b > c)
        {
            int swap = b;
            b = c;
            c = swap;
        }
        if (a > b)
        {
            int swap = a;
            a = b;
            b = swap;
        }
        return new Vector3Int(a, b, c);
    }

    private static ulong EdgeKey(int a, int b)
    {
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        return ((ulong)min << 32) | max;
    }

    private static void DecodeEdge(ulong key, out int a, out int b)
    {
        a = (int)(key >> 32);
        b = (int)(key & 0xffffffffu);
    }
}

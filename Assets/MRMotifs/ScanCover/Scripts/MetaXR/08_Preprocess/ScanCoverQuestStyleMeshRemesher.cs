using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Display-layer room-mesh remeshing. It collapses only short, planar, manifold interior edges,
/// then flips planar diagonals when the worst triangle quality improves. The source TSDF and the
/// stable-quad lifecycle are never modified.
/// </summary>
public static class ScanCoverQuestStyleMeshRemesher
{
    public struct Settings
    {
        public float TargetEdgeLengthMeters;
        public float CollapseBelowTargetScale;
        public float MinPlanarNormalDot;
        public float FeatureAngleDegrees;
        public float MaxCollapseFraction;
        public int MaxEdgeCollapses;
        public int MaxEdgeFlips;
        public float MinFlipQualityGain;
        public float MaxFlipEdgeTargetScale;
        public float MaxCollapseProjectionMeters;
        public float MinTriangleAreaMetersSquared;
        public float MaxMeanQualityLoss;
    }

    public struct Result
    {
        public bool Applied;
        public bool AtomicRollback;
        public int UniqueTrianglesBefore;
        public int UniqueTrianglesAfter;
        public int RenderTrianglesBefore;
        public int RenderTrianglesAfter;
        public int CandidateCollapseEdges;
        public int CollapsedEdges;
        public int RejectedCollapseEligibility;
        public int RejectedCollapseGeometry;
        public int RejectedCollapseTopology;
        public int RejectedCollapseQuality;
        public int DegenerateTrianglesRemoved;
        public int DuplicateTrianglesRemoved;
        public int CandidateFlipEdges;
        public int FlippedEdges;
        public int BoundaryEdgesBefore;
        public int BoundaryEdgesAfter;
        public int NonManifoldEdgesAfter;
        public int ProtectedBoundaryVertices;
        public int ProtectedCreaseVertices;
        public float MeanQualityBefore;
        public float MeanQualityAfter;
        public float MeanEdgeLengthBeforeMeters;
        public float MeanEdgeLengthAfterMeters;
    }

    private struct Tri
    {
        public int A;
        public int B;
        public int C;

        public Tri(int a, int b, int c)
        {
            A = a;
            B = b;
            C = c;
        }
    }

    private sealed class EdgeInfo
    {
        public int FirstTriangle = -1;
        public int SecondTriangle = -1;
        public int Count;
    }

    private struct EdgeCandidate
    {
        public int A;
        public int B;
        public float Length;
    }

    private enum CollapseRejectReason
    {
        None,
        Eligibility,
        Geometry,
        Topology,
        Quality
    }

    public static bool Remesh(
        List<Vector3> vertices,
        List<int> triangles,
        List<Color> colors,
        bool doubleSided,
        Settings settings,
        out Result result)
    {
        result = new Result();
        if (vertices == null || triangles == null || vertices.Count < 4 || triangles.Count < 6)
            return false;

        settings.TargetEdgeLengthMeters = Mathf.Max(0.001f, settings.TargetEdgeLengthMeters);
        settings.CollapseBelowTargetScale = Mathf.Clamp(settings.CollapseBelowTargetScale, 0.25f, 1.1f);
        settings.MinPlanarNormalDot = Mathf.Clamp(settings.MinPlanarNormalDot, 0.5f, 0.9999f);
        settings.FeatureAngleDegrees = Mathf.Clamp(settings.FeatureAngleDegrees, 5f, 85f);
        settings.MaxCollapseFraction = Mathf.Clamp(settings.MaxCollapseFraction, 0f, 0.45f);
        settings.MaxEdgeCollapses = Mathf.Max(0, settings.MaxEdgeCollapses);
        settings.MaxEdgeFlips = Mathf.Max(0, settings.MaxEdgeFlips);
        settings.MinFlipQualityGain = Mathf.Clamp(settings.MinFlipQualityGain, 0f, 0.5f);
        settings.MaxFlipEdgeTargetScale = Mathf.Clamp(settings.MaxFlipEdgeTargetScale, 1f, 3f);
        settings.MaxCollapseProjectionMeters = Mathf.Max(0f, settings.MaxCollapseProjectionMeters);
        settings.MinTriangleAreaMetersSquared = Mathf.Max(0.0000001f, settings.MinTriangleAreaMetersSquared);
        settings.MaxMeanQualityLoss = Mathf.Clamp(settings.MaxMeanQualityLoss, 0f, 0.1f);

        result.RenderTrianglesBefore = triangles.Count / 3;
        List<Tri> sourceFaces = BuildUniqueTriangles(vertices, triangles, out int sourceDuplicates);
        result.UniqueTrianglesBefore = sourceFaces.Count;
        result.DuplicateTrianglesRemoved = sourceDuplicates;
        if (sourceFaces.Count < 2)
            return false;

        Vector3[] workingVertices = vertices.ToArray();
        Color[] workingColors = BuildColorArray(vertices.Count, colors);
        Dictionary<ulong, EdgeInfo> sourceEdges = BuildEdges(sourceFaces);
        CountTopology(sourceEdges, out result.BoundaryEdgesBefore, out int sourceNonManifold);
        result.MeanQualityBefore = MeanTriangleQuality(workingVertices, sourceFaces);
        result.MeanEdgeLengthBeforeMeters = MeanUniqueEdgeLength(workingVertices, sourceEdges);
        result.UniqueTrianglesAfter = result.UniqueTrianglesBefore;
        result.RenderTrianglesAfter = result.RenderTrianglesBefore;
        result.BoundaryEdgesAfter = result.BoundaryEdgesBefore;
        result.MeanQualityAfter = result.MeanQualityBefore;
        result.MeanEdgeLengthAfterMeters = result.MeanEdgeLengthBeforeMeters;
        if (sourceNonManifold > 0)
        {
            result.AtomicRollback = true;
            return false;
        }

        BuildVertexNeighborhoods(
            workingVertices.Length,
            sourceFaces,
            sourceEdges,
            workingVertices,
            settings.FeatureAngleDegrees,
            out HashSet<int>[] neighbors,
            out List<int>[] incidentFaces,
            out bool[] protectedVertices,
            out bool[] boundaryVertices,
            out bool[] creaseVertices);
        CountProtected(boundaryVertices, creaseVertices, ref result);

        List<EdgeCandidate> candidates = new List<EdgeCandidate>();
        float collapseLength = settings.TargetEdgeLengthMeters * settings.CollapseBelowTargetScale;
        foreach (KeyValuePair<ulong, EdgeInfo> pair in sourceEdges)
        {
            EdgeInfo edge = pair.Value;
            if (edge.Count != 2 || edge.FirstTriangle < 0 || edge.SecondTriangle < 0)
                continue;
            DecodeEdge(pair.Key, out int a, out int b);
            if (protectedVertices[a] || protectedVertices[b])
                continue;
            float length = Vector3.Distance(workingVertices[a], workingVertices[b]);
            if (length >= collapseLength)
                continue;
            if (!FacesArePlanar(sourceFaces, workingVertices, edge.FirstTriangle, edge.SecondTriangle, settings.MinPlanarNormalDot))
                continue;
            if (!VertexFanIsPlanar(a, sourceFaces, workingVertices, incidentFaces, settings.MinPlanarNormalDot) ||
                !VertexFanIsPlanar(b, sourceFaces, workingVertices, incidentFaces, settings.MinPlanarNormalDot))
            {
                continue;
            }
            if (!PassLinkCondition(a, b, sourceFaces, edge, neighbors))
                continue;
            candidates.Add(new EdgeCandidate { A = a, B = b, Length = length });
        }
        candidates.Sort((left, right) => left.Length.CompareTo(right.Length));
        result.CandidateCollapseEdges = candidates.Count;

        int collapseBudget = Mathf.Min(
            settings.MaxEdgeCollapses,
            Mathf.FloorToInt(workingVertices.Length * settings.MaxCollapseFraction));
        bool[] collapseVertexUsed = new bool[workingVertices.Length];
        List<Tri> collapsedFaces = sourceFaces;
        int collapsedBoundary = result.BoundaryEdgesBefore;
        float collapsedQuality = result.MeanQualityBefore;

        for (int candidateIndex = 0;
             candidateIndex < candidates.Count && result.CollapsedEdges < collapseBudget;
             candidateIndex++)
        {
            EdgeCandidate candidate = candidates[candidateIndex];
            if (collapseVertexUsed[candidate.A] || collapseVertexUsed[candidate.B])
                continue;

            if (!TryCollapseEdgeAtomic(
                candidate.A,
                candidate.B,
                workingVertices,
                workingColors,
                collapsedFaces,
                settings,
                result.MeanQualityBefore,
                collapsedBoundary,
                out List<Tri> candidateFaces,
                out int candidateBoundary,
                out float candidateQuality,
                out int removedDegenerate,
                out HashSet<int>[] currentNeighbors,
                out CollapseRejectReason rejectReason))
            {
                CountCollapseReject(rejectReason, ref result);
                continue;
            }

            collapsedFaces = candidateFaces;
            collapsedBoundary = candidateBoundary;
            collapsedQuality = candidateQuality;
            result.DegenerateTrianglesRemoved += removedDegenerate;
            result.CollapsedEdges++;
            MarkCollapseNeighborhood(candidate.A, currentNeighbors, collapseVertexUsed);
            MarkCollapseNeighborhood(candidate.B, currentNeighbors, collapseVertexUsed);
        }

        Dictionary<ulong, EdgeInfo> collapsedEdges = BuildEdges(collapsedFaces);
        CountTopology(collapsedEdges, out collapsedBoundary, out int collapsedNonManifold);
        collapsedQuality = MeanTriangleQuality(workingVertices, collapsedFaces);
        result.UniqueTrianglesAfter = collapsedFaces.Count;
        result.RenderTrianglesAfter = collapsedFaces.Count * (doubleSided ? 2 : 1);
        result.BoundaryEdgesAfter = collapsedBoundary;
        result.NonManifoldEdgesAfter = collapsedNonManifold;
        result.MeanQualityAfter = collapsedQuality;
        result.MeanEdgeLengthAfterMeters = MeanUniqueEdgeLength(workingVertices, collapsedEdges);
        if (collapsedFaces.Count < 2 || collapsedBoundary > result.BoundaryEdgesBefore || collapsedNonManifold > 0 ||
            collapsedQuality + settings.MaxMeanQualityLoss < result.MeanQualityBefore)
        {
            result.AtomicRollback = true;
            return false;
        }

        CompactMesh(workingVertices, workingColors, collapsedFaces,
            out List<Vector3> compactVertices,
            out List<Color> compactColors,
            out List<Tri> compactFaces);

        List<Tri> flippedFaces = TryFlipPlanarEdges(
            compactVertices,
            compactFaces,
            settings,
            ref result);
        Dictionary<ulong, EdgeInfo> finalEdges = BuildEdges(flippedFaces);
        CountTopology(finalEdges, out result.BoundaryEdgesAfter, out result.NonManifoldEdgesAfter);
        result.MeanQualityAfter = MeanTriangleQuality(compactVertices, flippedFaces);
        result.MeanEdgeLengthAfterMeters = MeanUniqueEdgeLength(compactVertices, finalEdges);
        result.UniqueTrianglesAfter = flippedFaces.Count;
        result.RenderTrianglesAfter = flippedFaces.Count * (doubleSided ? 2 : 1);
        if (result.BoundaryEdgesAfter > result.BoundaryEdgesBefore || result.NonManifoldEdgesAfter > 0 ||
            result.MeanQualityAfter + settings.MaxMeanQualityLoss < result.MeanQualityBefore)
        {
            result.AtomicRollback = true;
            return false;
        }

        bool changed = result.CollapsedEdges > 0 || result.FlippedEdges > 0;
        if (!changed)
            return false;

        vertices.Clear();
        vertices.AddRange(compactVertices);
        colors.Clear();
        colors.AddRange(compactColors);
        triangles.Clear();
        for (int faceIndex = 0; faceIndex < flippedFaces.Count; faceIndex++)
        {
            Tri face = flippedFaces[faceIndex];
            triangles.Add(face.A);
            triangles.Add(face.B);
            triangles.Add(face.C);
            if (doubleSided)
            {
                triangles.Add(face.A);
                triangles.Add(face.C);
                triangles.Add(face.B);
            }
        }

        result.Applied = true;
        return true;
    }

    private static bool TryCollapseEdgeAtomic(
        int a,
        int b,
        Vector3[] vertices,
        Color[] colors,
        List<Tri> faces,
        Settings settings,
        float baselineQuality,
        int expectedBoundary,
        out List<Tri> collapsedFaces,
        out int collapsedBoundary,
        out float collapsedQuality,
        out int removedDegenerate,
        out HashSet<int>[] currentNeighbors,
        out CollapseRejectReason rejectReason)
    {
        collapsedFaces = null;
        collapsedBoundary = expectedBoundary;
        collapsedQuality = baselineQuality;
        removedDegenerate = 0;
        currentNeighbors = null;
        rejectReason = CollapseRejectReason.Eligibility;

        if (a < 0 || b < 0 || a >= vertices.Length || b >= vertices.Length || a == b)
            return false;

        Dictionary<ulong, EdgeInfo> edges = BuildEdges(faces);
        if (!edges.TryGetValue(EdgeKey(a, b), out EdgeInfo edge) ||
            edge.Count != 2 || edge.FirstTriangle < 0 || edge.SecondTriangle < 0)
        {
            return false;
        }

        BuildVertexNeighborhoods(
            vertices.Length,
            faces,
            edges,
            vertices,
            settings.FeatureAngleDegrees,
            out currentNeighbors,
            out List<int>[] incidentFaces,
            out bool[] protectedVertices,
            out _,
            out _);

        float collapseLength = settings.TargetEdgeLengthMeters * settings.CollapseBelowTargetScale;
        if (protectedVertices[a] || protectedVertices[b] ||
            Vector3.Distance(vertices[a], vertices[b]) >= collapseLength ||
            !FacesArePlanar(faces, vertices, edge.FirstTriangle, edge.SecondTriangle, settings.MinPlanarNormalDot) ||
            !VertexFanIsPlanar(a, faces, vertices, incidentFaces, settings.MinPlanarNormalDot) ||
            !VertexFanIsPlanar(b, faces, vertices, incidentFaces, settings.MinPlanarNormalDot) ||
            !PassLinkCondition(a, b, faces, edge, currentNeighbors))
        {
            return false;
        }

        int keep = Mathf.Min(a, b);
        int drop = Mathf.Max(a, b);
        Vector3 oldKeep = vertices[keep];
        Vector3 target = ComputePlanarCollapseTarget(
            a,
            b,
            vertices,
            faces,
            incidentFaces,
            settings.MaxCollapseProjectionMeters);
        vertices[keep] = target;

        List<Tri> candidateFaces = new List<Tri>(Mathf.Max(0, faces.Count - 2));
        HashSet<Vector3Int> candidateKeys = new HashSet<Vector3Int>();
        int removed = 0;
        for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
        {
            Tri before = faces[faceIndex];
            int mappedA = before.A == drop ? keep : before.A;
            int mappedB = before.B == drop ? keep : before.B;
            int mappedC = before.C == drop ? keep : before.C;
            bool touchedByCollapse = ContainsVertex(before, keep) || ContainsVertex(before, drop);
            if (mappedA == mappedB || mappedB == mappedC || mappedC == mappedA)
            {
                if (!ContainsVertex(before, a) || !ContainsVertex(before, b))
                {
                    vertices[keep] = oldKeep;
                    rejectReason = CollapseRejectReason.Topology;
                    return false;
                }
                removed++;
                continue;
            }

            Tri after = new Tri(mappedA, mappedB, mappedC);
            if (!touchedByCollapse)
            {
                if (!candidateKeys.Add(SortedTriangleKey(after.A, after.B, after.C)))
                {
                    vertices[keep] = oldKeep;
                    rejectReason = CollapseRejectReason.Topology;
                    return false;
                }
                candidateFaces.Add(after);
                continue;
            }

            Vector3 cross = Vector3.Cross(
                vertices[after.B] - vertices[after.A],
                vertices[after.C] - vertices[after.A]);
            if (cross.magnitude * 0.5f < settings.MinTriangleAreaMetersSquared)
            {
                vertices[keep] = oldKeep;
                rejectReason = CollapseRejectReason.Geometry;
                return false;
            }

            Vector3 beforeCross = TriangleCrossWithVertexOverride(before, vertices, keep, oldKeep);
            if (beforeCross.sqrMagnitude <= 0.000000000001f ||
                Vector3.Dot(beforeCross.normalized, cross.normalized) < 0.5f)
            {
                vertices[keep] = oldKeep;
                rejectReason = CollapseRejectReason.Geometry;
                return false;
            }

            if (!candidateKeys.Add(SortedTriangleKey(after.A, after.B, after.C)))
            {
                vertices[keep] = oldKeep;
                rejectReason = CollapseRejectReason.Topology;
                return false;
            }
            candidateFaces.Add(after);
        }

        if (removed != 2 || candidateFaces.Count < 2)
        {
            vertices[keep] = oldKeep;
            rejectReason = CollapseRejectReason.Topology;
            return false;
        }

        Dictionary<ulong, EdgeInfo> candidateEdges = BuildEdges(candidateFaces);
        CountTopology(candidateEdges, out int boundary, out int nonManifold);
        if (boundary != expectedBoundary || nonManifold > 0)
        {
            vertices[keep] = oldKeep;
            rejectReason = CollapseRejectReason.Topology;
            return false;
        }

        float quality = MeanTriangleQuality(vertices, candidateFaces);
        if (quality + settings.MaxMeanQualityLoss < baselineQuality)
        {
            vertices[keep] = oldKeep;
            rejectReason = CollapseRejectReason.Quality;
            return false;
        }

        colors[keep] = Color.Lerp(colors[a], colors[b], 0.5f);
        collapsedFaces = candidateFaces;
        collapsedBoundary = boundary;
        collapsedQuality = quality;
        removedDegenerate = removed;
        rejectReason = CollapseRejectReason.None;
        return true;
    }

    private static Vector3 TriangleCrossWithVertexOverride(
        Tri face,
        Vector3[] vertices,
        int overrideIndex,
        Vector3 overridePosition)
    {
        Vector3 a = face.A == overrideIndex ? overridePosition : vertices[face.A];
        Vector3 b = face.B == overrideIndex ? overridePosition : vertices[face.B];
        Vector3 c = face.C == overrideIndex ? overridePosition : vertices[face.C];
        return Vector3.Cross(b - a, c - a);
    }

    private static bool ContainsVertex(Tri face, int vertex)
    {
        return face.A == vertex || face.B == vertex || face.C == vertex;
    }

    private static void CountCollapseReject(CollapseRejectReason reason, ref Result result)
    {
        switch (reason)
        {
            case CollapseRejectReason.Geometry:
                result.RejectedCollapseGeometry++;
                break;
            case CollapseRejectReason.Topology:
                result.RejectedCollapseTopology++;
                break;
            case CollapseRejectReason.Quality:
                result.RejectedCollapseQuality++;
                break;
            default:
                result.RejectedCollapseEligibility++;
                break;
        }
    }

    private static List<Tri> TryFlipPlanarEdges(
        List<Vector3> vertices,
        List<Tri> faces,
        Settings settings,
        ref Result result)
    {
        if (settings.MaxEdgeFlips <= 0 || faces.Count < 2)
            return faces;

        Dictionary<ulong, EdgeInfo> edges = BuildEdges(faces);
        BuildVertexNeighborhoods(
            vertices.Count,
            faces,
            edges,
            vertices,
            settings.FeatureAngleDegrees,
            out _,
            out _,
            out bool[] protectedVertices,
            out _,
            out _);
        bool[] faceUsed = new bool[faces.Count];
        HashSet<ulong> plannedDiagonals = new HashSet<ulong>();
        List<Tri> output = new List<Tri>(faces);

        foreach (KeyValuePair<ulong, EdgeInfo> pair in edges)
        {
            if (result.FlippedEdges >= settings.MaxEdgeFlips)
                break;
            EdgeInfo edge = pair.Value;
            if (edge.Count != 2 || edge.FirstTriangle < 0 || edge.SecondTriangle < 0 ||
                faceUsed[edge.FirstTriangle] || faceUsed[edge.SecondTriangle])
            {
                continue;
            }
            DecodeEdge(pair.Key, out int a, out int b);
            Tri first = faces[edge.FirstTriangle];
            Tri second = faces[edge.SecondTriangle];
            if (!HasDirectedEdge(first, a, b))
            {
                int swap = a;
                a = b;
                b = swap;
            }
            if (!HasDirectedEdge(first, a, b) || !HasDirectedEdge(second, b, a))
                continue;
            int c = ThirdVertex(first, a, b);
            int d = ThirdVertex(second, a, b);
            if (c < 0 || d < 0 || c == d || protectedVertices[a] || protectedVertices[b] ||
                protectedVertices[c] || protectedVertices[d])
            {
                continue;
            }
            result.CandidateFlipEdges++;
            ulong newDiagonal = EdgeKey(c, d);
            if (edges.ContainsKey(newDiagonal) || !plannedDiagonals.Add(newDiagonal))
                continue;
            if (!FacesArePlanar(faces, vertices, edge.FirstTriangle, edge.SecondTriangle, settings.MinPlanarNormalDot))
                continue;
            if (Vector3.Distance(vertices[c], vertices[d]) > settings.TargetEdgeLengthMeters * settings.MaxFlipEdgeTargetScale)
                continue;

            Tri proposedFirst = new Tri(c, a, d);
            Tri proposedSecond = new Tri(c, d, b);
            float beforeWorst = Mathf.Min(TriangleQuality(vertices, first), TriangleQuality(vertices, second));
            float afterWorst = Mathf.Min(TriangleQuality(vertices, proposedFirst), TriangleQuality(vertices, proposedSecond));
            if (afterWorst < beforeWorst + settings.MinFlipQualityGain)
                continue;
            if (!PreservesTriangleOrientation(vertices, first, proposedFirst) ||
                !PreservesTriangleOrientation(vertices, second, proposedSecond))
            {
                continue;
            }

            output[edge.FirstTriangle] = proposedFirst;
            output[edge.SecondTriangle] = proposedSecond;
            faceUsed[edge.FirstTriangle] = true;
            faceUsed[edge.SecondTriangle] = true;
            result.FlippedEdges++;
        }

        Dictionary<ulong, EdgeInfo> outputEdges = BuildEdges(output);
        CountTopology(outputEdges, out int outputBoundary, out int outputNonManifold);
        CountTopology(edges, out int inputBoundary, out int inputNonManifold);
        if (inputNonManifold > 0 || outputNonManifold > 0 || outputBoundary != inputBoundary)
        {
            result.FlippedEdges = 0;
            return faces;
        }
        return output;
    }

    private static List<Tri> BuildUniqueTriangles(List<Vector3> vertices, List<int> triangles, out int duplicates)
    {
        duplicates = 0;
        List<Tri> faces = new List<Tri>(triangles.Count / 6);
        HashSet<Vector3Int> keys = new HashSet<Vector3Int>();
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
            if (!keys.Add(key))
            {
                duplicates++;
                continue;
            }
            if (Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]).sqrMagnitude <= 0.000000000001f)
                continue;
            faces.Add(new Tri(a, b, c));
        }
        return faces;
    }

    private static void BuildVertexNeighborhoods(
        int vertexCount,
        List<Tri> faces,
        Dictionary<ulong, EdgeInfo> edges,
        IList<Vector3> positions,
        float featureAngleDegrees,
        out HashSet<int>[] neighbors,
        out List<int>[] incidentFaces,
        out bool[] protectedVertices,
        out bool[] boundaryVertices,
        out bool[] creaseVertices)
    {
        neighbors = new HashSet<int>[vertexCount];
        incidentFaces = new List<int>[vertexCount];
        protectedVertices = new bool[vertexCount];
        boundaryVertices = new bool[vertexCount];
        creaseVertices = new bool[vertexCount];
        for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
        {
            Tri face = faces[faceIndex];
            AddNeighbor(neighbors, face.A, face.B);
            AddNeighbor(neighbors, face.B, face.C);
            AddNeighbor(neighbors, face.C, face.A);
            AddIncidentFace(incidentFaces, face.A, faceIndex);
            AddIncidentFace(incidentFaces, face.B, faceIndex);
            AddIncidentFace(incidentFaces, face.C, faceIndex);
        }
        float creaseDot = Mathf.Cos(featureAngleDegrees * Mathf.Deg2Rad);
        foreach (KeyValuePair<ulong, EdgeInfo> pair in edges)
        {
            DecodeEdge(pair.Key, out int a, out int b);
            EdgeInfo edge = pair.Value;
            if (edge.Count != 2)
            {
                protectedVertices[a] = true;
                protectedVertices[b] = true;
                boundaryVertices[a] = true;
                boundaryVertices[b] = true;
                continue;
            }
            Vector3 firstNormal = TriangleNormal(faces[edge.FirstTriangle], positions);
            Vector3 secondNormal = TriangleNormal(faces[edge.SecondTriangle], positions);
            if (firstNormal != Vector3.zero && secondNormal != Vector3.zero &&
                Mathf.Abs(Vector3.Dot(firstNormal, secondNormal)) < creaseDot)
            {
                protectedVertices[a] = true;
                protectedVertices[b] = true;
                creaseVertices[a] = true;
                creaseVertices[b] = true;
            }
        }
    }

    private static void CountProtected(bool[] boundary, bool[] crease, ref Result result)
    {
        for (int i = 0; i < boundary.Length; i++)
        {
            if (boundary[i])
                result.ProtectedBoundaryVertices++;
            if (crease[i])
                result.ProtectedCreaseVertices++;
        }
    }

    private static bool FacesArePlanar(
        List<Tri> faces,
        IList<Vector3> vertices,
        int first,
        int second,
        float minDot)
    {
        Vector3 firstNormal = TriangleNormal(faces[first], vertices);
        Vector3 secondNormal = TriangleNormal(faces[second], vertices);
        return firstNormal != Vector3.zero && secondNormal != Vector3.zero &&
               Mathf.Abs(Vector3.Dot(firstNormal, secondNormal)) >= minDot;
    }

    private static bool VertexFanIsPlanar(
        int vertex,
        List<Tri> faces,
        IList<Vector3> vertices,
        List<int>[] incidentFaces,
        float minDot)
    {
        if (incidentFaces[vertex] == null || incidentFaces[vertex].Count < 3)
            return false;
        Vector3 reference = TriangleNormal(faces[incidentFaces[vertex][0]], vertices);
        if (reference == Vector3.zero)
            return false;
        for (int i = 1; i < incidentFaces[vertex].Count; i++)
        {
            Vector3 normal = TriangleNormal(faces[incidentFaces[vertex][i]], vertices);
            if (normal == Vector3.zero || Mathf.Abs(Vector3.Dot(reference, normal)) < minDot)
                return false;
        }
        return true;
    }

    private static bool PassLinkCondition(
        int a,
        int b,
        List<Tri> faces,
        EdgeInfo edge,
        HashSet<int>[] neighbors)
    {
        if (neighbors[a] == null || neighbors[b] == null)
            return false;
        int firstOpposite = ThirdVertex(faces[edge.FirstTriangle], a, b);
        int secondOpposite = ThirdVertex(faces[edge.SecondTriangle], a, b);
        if (firstOpposite < 0 || secondOpposite < 0 || firstOpposite == secondOpposite)
            return false;
        int commonCount = 0;
        bool foundFirst = false;
        bool foundSecond = false;
        HashSet<int> smaller = neighbors[a].Count <= neighbors[b].Count ? neighbors[a] : neighbors[b];
        HashSet<int> larger = smaller == neighbors[a] ? neighbors[b] : neighbors[a];
        foreach (int neighbor in smaller)
        {
            if (!larger.Contains(neighbor))
                continue;
            commonCount++;
            foundFirst |= neighbor == firstOpposite;
            foundSecond |= neighbor == secondOpposite;
        }
        return commonCount == 2 && foundFirst && foundSecond;
    }

    private static Vector3 ComputePlanarCollapseTarget(
        int a,
        int b,
        Vector3[] vertices,
        List<Tri> faces,
        List<int>[] incidentFaces,
        float maxProjection)
    {
        Vector3 midpoint = (vertices[a] + vertices[b]) * 0.5f;
        Vector3 normalSum = Vector3.zero;
        Vector3 centerSum = Vector3.zero;
        int count = 0;
        Vector3 reference = Vector3.zero;
        AccumulateCollapsePlane(a, vertices, faces, incidentFaces, ref reference, ref normalSum, ref centerSum, ref count);
        AccumulateCollapsePlane(b, vertices, faces, incidentFaces, ref reference, ref normalSum, ref centerSum, ref count);
        if (count <= 0 || normalSum.sqrMagnitude < 0.0000001f)
            return midpoint;
        Vector3 normal = normalSum.normalized;
        Vector3 center = centerSum / count;
        Vector3 projected = midpoint - normal * Vector3.Dot(midpoint - center, normal);
        return midpoint + Vector3.ClampMagnitude(projected - midpoint, maxProjection);
    }

    private static void AccumulateCollapsePlane(
        int vertex,
        Vector3[] vertices,
        List<Tri> faces,
        List<int>[] incidentFaces,
        ref Vector3 reference,
        ref Vector3 normalSum,
        ref Vector3 centerSum,
        ref int count)
    {
        if (incidentFaces[vertex] == null)
            return;
        for (int i = 0; i < incidentFaces[vertex].Count; i++)
        {
            Tri face = faces[incidentFaces[vertex][i]];
            Vector3 normal = TriangleNormal(face, vertices);
            if (normal == Vector3.zero)
                continue;
            if (reference == Vector3.zero)
                reference = normal;
            if (Vector3.Dot(reference, normal) < 0f)
                normal = -normal;
            normalSum += normal;
            centerSum += (vertices[face.A] + vertices[face.B] + vertices[face.C]) / 3f;
            count++;
        }
    }

    private static void CompactMesh(
        Vector3[] vertices,
        Color[] colors,
        List<Tri> faces,
        out List<Vector3> compactVertices,
        out List<Color> compactColors,
        out List<Tri> compactFaces)
    {
        int[] remap = new int[vertices.Length];
        for (int i = 0; i < remap.Length; i++)
            remap[i] = -1;
        compactVertices = new List<Vector3>();
        compactColors = new List<Color>();
        compactFaces = new List<Tri>(faces.Count);
        for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
        {
            Tri face = faces[faceIndex];
            int a = CompactVertex(face.A, vertices, colors, remap, compactVertices, compactColors);
            int b = CompactVertex(face.B, vertices, colors, remap, compactVertices, compactColors);
            int c = CompactVertex(face.C, vertices, colors, remap, compactVertices, compactColors);
            compactFaces.Add(new Tri(a, b, c));
        }
    }

    private static int CompactVertex(
        int source,
        Vector3[] vertices,
        Color[] colors,
        int[] remap,
        List<Vector3> compactVertices,
        List<Color> compactColors)
    {
        if (remap[source] >= 0)
            return remap[source];
        int target = compactVertices.Count;
        remap[source] = target;
        compactVertices.Add(vertices[source]);
        compactColors.Add(colors[source]);
        return target;
    }

    private static Dictionary<ulong, EdgeInfo> BuildEdges(List<Tri> faces)
    {
        Dictionary<ulong, EdgeInfo> edges = new Dictionary<ulong, EdgeInfo>(faces.Count * 2);
        for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
        {
            Tri face = faces[faceIndex];
            AddEdge(edges, face.A, face.B, faceIndex);
            AddEdge(edges, face.B, face.C, faceIndex);
            AddEdge(edges, face.C, face.A, faceIndex);
        }
        return edges;
    }

    private static void AddEdge(Dictionary<ulong, EdgeInfo> edges, int a, int b, int face)
    {
        ulong key = EdgeKey(a, b);
        if (!edges.TryGetValue(key, out EdgeInfo info))
        {
            info = new EdgeInfo();
            edges[key] = info;
        }
        if (info.Count == 0)
            info.FirstTriangle = face;
        else if (info.Count == 1)
            info.SecondTriangle = face;
        info.Count++;
    }

    private static void CountTopology(Dictionary<ulong, EdgeInfo> edges, out int boundary, out int nonManifold)
    {
        boundary = 0;
        nonManifold = 0;
        foreach (EdgeInfo edge in edges.Values)
        {
            if (edge.Count == 1)
                boundary++;
            else if (edge.Count > 2)
                nonManifold++;
        }
    }

    private static float MeanUniqueEdgeLength(IList<Vector3> vertices, Dictionary<ulong, EdgeInfo> edges)
    {
        if (edges.Count == 0)
            return 0f;
        float sum = 0f;
        foreach (ulong key in edges.Keys)
        {
            DecodeEdge(key, out int a, out int b);
            sum += Vector3.Distance(vertices[a], vertices[b]);
        }
        return sum / edges.Count;
    }

    private static float MeanTriangleQuality(IList<Vector3> vertices, List<Tri> faces)
    {
        if (faces.Count == 0)
            return 0f;
        float sum = 0f;
        for (int i = 0; i < faces.Count; i++)
            sum += TriangleQuality(vertices, faces[i]);
        return sum / faces.Count;
    }

    private static float TriangleQuality(IList<Vector3> vertices, Tri face)
    {
        Vector3 ab = vertices[face.B] - vertices[face.A];
        Vector3 bc = vertices[face.C] - vertices[face.B];
        Vector3 ca = vertices[face.A] - vertices[face.C];
        float denominator = ab.sqrMagnitude + bc.sqrMagnitude + ca.sqrMagnitude;
        if (denominator <= 0.0000001f)
            return 0f;
        float doubleArea = Vector3.Cross(ab, vertices[face.C] - vertices[face.A]).magnitude;
        return 2f * Mathf.Sqrt(3f) * doubleArea / denominator;
    }

    private static bool PreservesTriangleOrientation(IList<Vector3> vertices, Tri before, Tri after)
    {
        Vector3 beforeNormal = TriangleNormal(before, vertices);
        Vector3 afterNormal = TriangleNormal(after, vertices);
        return beforeNormal != Vector3.zero && afterNormal != Vector3.zero &&
               Vector3.Dot(beforeNormal, afterNormal) >= 0.5f;
    }

    private static Vector3 TriangleNormal(Tri face, IList<Vector3> vertices)
    {
        if (vertices == null)
            return Vector3.zero;
        Vector3 cross = Vector3.Cross(vertices[face.B] - vertices[face.A], vertices[face.C] - vertices[face.A]);
        return cross.sqrMagnitude > 0.000000000001f ? cross.normalized : Vector3.zero;
    }

    private static Color[] BuildColorArray(int vertexCount, List<Color> colors)
    {
        Color[] result = new Color[vertexCount];
        for (int i = 0; i < vertexCount; i++)
            result[i] = colors != null && i < colors.Count ? colors[i] : Color.white;
        return result;
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

    private static void MarkCollapseNeighborhood(int vertex, HashSet<int>[] neighbors, bool[] blocked)
    {
        blocked[vertex] = true;
        if (neighbors[vertex] == null)
            return;
        foreach (int neighbor in neighbors[vertex])
            blocked[neighbor] = true;
    }

    private static bool HasDirectedEdge(Tri face, int a, int b)
    {
        return (face.A == a && face.B == b) ||
               (face.B == a && face.C == b) ||
               (face.C == a && face.A == b);
    }

    private static int ThirdVertex(Tri face, int a, int b)
    {
        if (face.A != a && face.A != b)
            return face.A;
        if (face.B != a && face.B != b)
            return face.B;
        if (face.C != a && face.C != b)
            return face.C;
        return -1;
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

using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Read-only, whole-volume TSDF iso-surface reference used to compare the legacy
/// face-lifecycle mesher with a disposable volumetric mesh.  A fixed six-tetrahedra
/// cube decomposition avoids lookup-table ambiguity and a global lattice-edge cache
/// makes adjacent cells share the exact same vertices.
/// </summary>
public static class ScanCoverReferenceVolumetricMesher
{
    public sealed class Result
    {
        public readonly struct BoundarySegment
        {
            public readonly int A;
            public readonly int B;

            public BoundarySegment(int a, int b)
            {
                A = a;
                B = b;
            }
        }

        public readonly List<Vector3> Vertices = new List<Vector3>(32768);
        public readonly List<int> Triangles = new List<int>(98304);
        public readonly List<BoundarySegment> BoundarySegments = new List<BoundarySegment>(4096);
        public int ScannedCells;
        public int SupportedTetrahedra;
        public int BoundaryEdges;
        public int NonManifoldEdges;
        public int DuplicateTriangles;
        public int WeldedVertices;
        public int WeldRemovedDegenerateTriangles;
        public int WeldRemovedDuplicateTriangles;
        public bool Truncated;
    }

    // Corner order matches ScanCoverTsdfSingleShellPrototype.
    private static readonly int[] CornerX = { 0, 1, 1, 0, 0, 1, 1, 0 };
    private static readonly int[] CornerY = { 0, 0, 1, 1, 0, 0, 1, 1 };
    private static readonly int[] CornerZ = { 0, 0, 0, 0, 1, 1, 1, 1 };

    // All cubes use the same body diagonal (0, 6).  The induced face diagonals
    // therefore agree between neighboring cubes.
    private static readonly int[,] Tetrahedra =
    {
        { 0, 5, 1, 6 },
        { 0, 1, 2, 6 },
        { 0, 2, 3, 6 },
        { 0, 3, 7, 6 },
        { 0, 7, 4, 6 },
        { 0, 4, 5, 6 }
    };

    public static Result Build(
        float[] tsdf,
        byte[] weights,
        int dimX,
        int dimY,
        int dimZ,
        Vector3 volumeOriginWorld,
        float voxelSize,
        int minimumWeight,
        int maximumTriangles)
    {
        return BuildCore(tsdf, weights, null, dimX, dimY, dimZ, volumeOriginWorld,
            voxelSize, Mathf.Max(1, minimumWeight), maximumTriangles);
    }

    public static Result Build(
        float[] tsdf,
        float[] weights,
        int dimX,
        int dimY,
        int dimZ,
        Vector3 volumeOriginWorld,
        float voxelSize,
        float minimumWeight,
        int maximumTriangles)
    {
        return BuildCore(tsdf, null, weights, dimX, dimY, dimZ, volumeOriginWorld,
            voxelSize, Mathf.Max(0.0001f, minimumWeight), maximumTriangles);
    }

    private static Result BuildCore(
        float[] tsdf,
        byte[] byteWeights,
        float[] floatWeights,
        int dimX,
        int dimY,
        int dimZ,
        Vector3 volumeOriginWorld,
        float voxelSize,
        float minimumWeight,
        int maximumTriangles)
    {
        Result result = new Result();
        int weightLength = byteWeights != null ? byteWeights.Length : floatWeights != null ? floatWeights.Length : 0;
        if (tsdf == null || weightLength <= 0 || dimX < 2 || dimY < 2 || dimZ < 2 ||
            tsdf.Length != weightLength || tsdf.Length < dimX * dimY * dimZ)
            return result;

        int triangleLimit = Mathf.Max(1, maximumTriangles);
        if (!TryFindObservedBounds(byteWeights, floatWeights, dimX, dimY, dimZ, minimumWeight,
                out Vector3Int minimum, out Vector3Int maximum))
            return result;

        int[] cornerIndices = new int[8];
        float[] cornerValues = new float[8];
        Vector3[] cornerPositions = new Vector3[8];
        int[] tetraCorners = new int[4];
        int[] tetraInside = new int[4];
        int[] tetraOutside = new int[4];
        Dictionary<ulong, int> edgeVertices = new Dictionary<ulong, int>(65536);

        int minX = Mathf.Max(0, minimum.x - 1);
        int minY = Mathf.Max(0, minimum.y - 1);
        int minZ = Mathf.Max(0, minimum.z - 1);
        int maxX = Mathf.Min(dimX - 2, maximum.x);
        int maxY = Mathf.Min(dimY - 2, maximum.y);
        int maxZ = Mathf.Min(dimZ - 2, maximum.z);

        for (int z = minZ; z <= maxZ && !result.Truncated; z++)
        for (int y = minY; y <= maxY && !result.Truncated; y++)
        for (int x = minX; x <= maxX && !result.Truncated; x++)
        {
            result.ScannedCells++;
            for (int corner = 0; corner < 8; corner++)
            {
                int vx = x + CornerX[corner];
                int vy = y + CornerY[corner];
                int vz = z + CornerZ[corner];
                int index = Index(vx, vy, vz, dimX, dimY);
                cornerIndices[corner] = index;
                cornerValues[corner] = tsdf[index];
                cornerPositions[corner] = volumeOriginWorld +
                    new Vector3(vx * voxelSize, vy * voxelSize, vz * voxelSize);
            }

            for (int tetra = 0; tetra < 6; tetra++)
            {
                int a = Tetrahedra[tetra, 0];
                int b = Tetrahedra[tetra, 1];
                int c = Tetrahedra[tetra, 2];
                int d = Tetrahedra[tetra, 3];
                if (GetWeight(byteWeights, floatWeights, cornerIndices[a]) < minimumWeight ||
                    GetWeight(byteWeights, floatWeights, cornerIndices[b]) < minimumWeight ||
                    GetWeight(byteWeights, floatWeights, cornerIndices[c]) < minimumWeight ||
                    GetWeight(byteWeights, floatWeights, cornerIndices[d]) < minimumWeight)
                    continue;
                if (!Finite(cornerValues[a]) || !Finite(cornerValues[b]) ||
                    !Finite(cornerValues[c]) || !Finite(cornerValues[d]))
                    continue;

                result.SupportedTetrahedra++;
                PolygonizeTetrahedron(
                    a, b, c, d,
                    cornerIndices, cornerValues, cornerPositions,
                    tetraCorners, tetraInside, tetraOutside,
                    edgeVertices, result, triangleLimit);
                if (result.Triangles.Count / 3 >= triangleLimit)
                    result.Truncated = true;
            }
        }

        WeldNearCoincidentVertices(result, Mathf.Max(0.000001f, voxelSize * 0.0025f));
        AuditTopology(result);
        return result;
    }

    private static void WeldNearCoincidentVertices(Result result, float tolerance)
    {
        if (result.Vertices.Count <= 0 || result.Triangles.Count <= 0 || tolerance <= 0f)
            return;

        float inverseCell = 1f / tolerance;
        float toleranceSq = tolerance * tolerance;
        Dictionary<Vector3Int, List<int>> buckets = new Dictionary<Vector3Int, List<int>>(result.Vertices.Count);
        List<Vector3> welded = new List<Vector3>(result.Vertices.Count);
        int[] remap = new int[result.Vertices.Count];
        for (int oldIndex = 0; oldIndex < result.Vertices.Count; oldIndex++)
        {
            Vector3 point = result.Vertices[oldIndex];
            Vector3Int cell = new Vector3Int(
                Mathf.FloorToInt(point.x * inverseCell),
                Mathf.FloorToInt(point.y * inverseCell),
                Mathf.FloorToInt(point.z * inverseCell));
            int matched = -1;
            for (int z = -1; z <= 1 && matched < 0; z++)
            for (int y = -1; y <= 1 && matched < 0; y++)
            for (int x = -1; x <= 1 && matched < 0; x++)
            {
                if (!buckets.TryGetValue(cell + new Vector3Int(x, y, z), out List<int> candidates))
                    continue;
                for (int i = 0; i < candidates.Count; i++)
                {
                    int candidate = candidates[i];
                    if ((welded[candidate] - point).sqrMagnitude > toleranceSq)
                        continue;
                    matched = candidate;
                    break;
                }
            }
            if (matched < 0)
            {
                matched = welded.Count;
                welded.Add(point);
                if (!buckets.TryGetValue(cell, out List<int> cellVertices))
                {
                    cellVertices = new List<int>(4);
                    buckets[cell] = cellVertices;
                }
                cellVertices.Add(matched);
            }
            else
            {
                result.WeldedVertices++;
            }
            remap[oldIndex] = matched;
        }

        List<int> triangles = new List<int>(result.Triangles.Count);
        HashSet<TriangleKey> unique = new HashSet<TriangleKey>();
        for (int i = 0; i + 2 < result.Triangles.Count; i += 3)
        {
            int a = remap[result.Triangles[i]];
            int b = remap[result.Triangles[i + 1]];
            int c = remap[result.Triangles[i + 2]];
            if (a == b || b == c || c == a)
            {
                result.WeldRemovedDegenerateTriangles++;
                continue;
            }
            if (!unique.Add(new TriangleKey(a, b, c)))
            {
                result.WeldRemovedDuplicateTriangles++;
                continue;
            }
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
        }
        result.Vertices.Clear();
        result.Vertices.AddRange(welded);
        result.Triangles.Clear();
        result.Triangles.AddRange(triangles);
    }

    private static void PolygonizeTetrahedron(
        int a, int b, int c, int d,
        int[] indices, float[] values, Vector3[] positions,
        int[] tetra, int[] inside, int[] outside,
        Dictionary<ulong, int> edgeVertices, Result result, int triangleLimit)
    {
        tetra[0] = a;
        tetra[1] = b;
        tetra[2] = c;
        tetra[3] = d;
        int insideCount = 0;
        int outsideCount = 0;
        for (int i = 0; i < 4; i++)
        {
            if (values[tetra[i]] < 0f)
                inside[insideCount++] = tetra[i];
            else
                outside[outsideCount++] = tetra[i];
        }

        if (insideCount == 0 || insideCount == 4 || result.Triangles.Count / 3 >= triangleLimit)
            return;

        Vector3 outward = Vector3.zero;
        for (int i = 0; i < outsideCount; i++) outward += positions[outside[i]];
        for (int i = 0; i < insideCount; i++) outward -= positions[inside[i]];

        if (insideCount == 1 || insideCount == 3)
        {
            bool invert = insideCount == 3;
            int pivot = invert ? outside[0] : inside[0];
            int q0 = invert ? inside[0] : outside[0];
            int q1 = invert ? inside[1] : outside[1];
            int q2 = invert ? inside[2] : outside[2];
            int v0 = EdgeVertex(pivot, q0, indices, values, positions, edgeVertices, result.Vertices);
            int v1 = EdgeVertex(pivot, q1, indices, values, positions, edgeVertices, result.Vertices);
            int v2 = EdgeVertex(pivot, q2, indices, values, positions, edgeVertices, result.Vertices);
            AddOrientedTriangle(v0, v1, v2, outward, result);
            return;
        }

        // Two negative and two positive vertices form a quad.  The diagonal is
        // deterministic because tetrahedron vertex order is fixed globally.
        int i0 = inside[0];
        int i1 = inside[1];
        int o0 = outside[0];
        int o1 = outside[1];
        int v00 = EdgeVertex(i0, o0, indices, values, positions, edgeVertices, result.Vertices);
        int v01 = EdgeVertex(i0, o1, indices, values, positions, edgeVertices, result.Vertices);
        int v11 = EdgeVertex(i1, o1, indices, values, positions, edgeVertices, result.Vertices);
        int v10 = EdgeVertex(i1, o0, indices, values, positions, edgeVertices, result.Vertices);
        AddOrientedTriangle(v00, v01, v11, outward, result);
        if (result.Triangles.Count / 3 < triangleLimit)
            AddOrientedTriangle(v00, v11, v10, outward, result);
    }

    private static int EdgeVertex(
        int localA, int localB,
        int[] indices, float[] values, Vector3[] positions,
        Dictionary<ulong, int> edgeVertices, List<Vector3> vertices)
    {
        int indexA = indices[localA];
        int indexB = indices[localB];
        ulong key = UndirectedPairKey(indexA, indexB);
        if (edgeVertices.TryGetValue(key, out int vertexIndex))
            return vertexIndex;

        float valueA = values[localA];
        float valueB = values[localB];
        float denominator = valueA - valueB;
        float t = Mathf.Abs(denominator) > 0.000001f
            ? Mathf.Clamp01(valueA / denominator)
            : 0.5f;
        vertexIndex = vertices.Count;
        vertices.Add(Vector3.Lerp(positions[localA], positions[localB], t));
        edgeVertices[key] = vertexIndex;
        return vertexIndex;
    }

    private static void AddOrientedTriangle(int a, int b, int c, Vector3 outward, Result result)
    {
        if (a == b || b == c || c == a)
            return;
        Vector3 va = result.Vertices[a];
        Vector3 vb = result.Vertices[b];
        Vector3 vc = result.Vertices[c];
        Vector3 cross = Vector3.Cross(vb - va, vc - va);
        if (cross.sqrMagnitude <= 0.0000000001f)
            return;
        if (Vector3.Dot(cross, outward) < 0f)
        {
            int swap = b;
            b = c;
            c = swap;
        }
        result.Triangles.Add(a);
        result.Triangles.Add(b);
        result.Triangles.Add(c);
    }

    private static void AuditTopology(Result result)
    {
        Dictionary<ulong, int> edgeUse = new Dictionary<ulong, int>(result.Triangles.Count);
        HashSet<TriangleKey> triangles = new HashSet<TriangleKey>();
        for (int i = 0; i + 2 < result.Triangles.Count; i += 3)
        {
            int a = result.Triangles[i];
            int b = result.Triangles[i + 1];
            int c = result.Triangles[i + 2];
            AddEdgeUse(a, b, edgeUse);
            AddEdgeUse(b, c, edgeUse);
            AddEdgeUse(c, a, edgeUse);
            if (!triangles.Add(new TriangleKey(a, b, c)))
                result.DuplicateTriangles++;
        }
        foreach (KeyValuePair<ulong, int> pair in edgeUse)
        {
            if (pair.Value == 1)
            {
                result.BoundaryEdges++;
                result.BoundarySegments.Add(new Result.BoundarySegment(
                    (int)(pair.Key >> 32), (int)(pair.Key & 0xffffffffUL)));
            }
            else if (pair.Value > 2) result.NonManifoldEdges++;
        }
    }

    private static void AddEdgeUse(int a, int b, Dictionary<ulong, int> edgeUse)
    {
        ulong key = UndirectedPairKey(a, b);
        edgeUse.TryGetValue(key, out int count);
        edgeUse[key] = count + 1;
    }

    private static bool TryFindObservedBounds(
        byte[] byteWeights, float[] floatWeights, int dimX, int dimY, int dimZ, float minimumWeight,
        out Vector3Int minimum, out Vector3Int maximum)
    {
        minimum = new Vector3Int(dimX, dimY, dimZ);
        maximum = new Vector3Int(-1, -1, -1);
        int plane = dimX * dimY;
        int length = byteWeights != null ? byteWeights.Length : floatWeights.Length;
        for (int index = 0; index < length; index++)
        {
            if (GetWeight(byteWeights, floatWeights, index) < minimumWeight)
                continue;
            int z = index / plane;
            int remainder = index - z * plane;
            int y = remainder / dimX;
            int x = remainder - y * dimX;
            minimum.x = Mathf.Min(minimum.x, x);
            minimum.y = Mathf.Min(minimum.y, y);
            minimum.z = Mathf.Min(minimum.z, z);
            maximum.x = Mathf.Max(maximum.x, x);
            maximum.y = Mathf.Max(maximum.y, y);
            maximum.z = Mathf.Max(maximum.z, z);
        }
        return maximum.x >= minimum.x && maximum.y >= minimum.y && maximum.z >= minimum.z;
    }

    private static float GetWeight(byte[] byteWeights, float[] floatWeights, int index)
    {
        return byteWeights != null ? byteWeights[index] : floatWeights[index];
    }

    private static int Index(int x, int y, int z, int dimX, int dimY)
    {
        return x + dimX * (y + dimY * z);
    }

    private static ulong UndirectedPairKey(int a, int b)
    {
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        return ((ulong)min << 32) | max;
    }

    private static bool Finite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private readonly struct TriangleKey : System.IEquatable<TriangleKey>
    {
        private readonly int _a;
        private readonly int _b;
        private readonly int _c;

        public TriangleKey(int a, int b, int c)
        {
            if (a > b) { int t = a; a = b; b = t; }
            if (b > c) { int t = b; b = c; c = t; }
            if (a > b) { int t = a; a = b; b = t; }
            _a = a;
            _b = b;
            _c = c;
        }

        public bool Equals(TriangleKey other)
        {
            return _a == other._a && _b == other._b && _c == other._c;
        }

        public override bool Equals(object obj)
        {
            return obj is TriangleKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = _a;
                hash = hash * 397 ^ _b;
                return hash * 397 ^ _c;
            }
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

public static class ScanCoverTsdfMesherUtil
{
    private static readonly Vector3Int[] CornerOffsets =
    {
        new Vector3Int(0, 0, 0),
        new Vector3Int(1, 0, 0),
        new Vector3Int(1, 0, 1),
        new Vector3Int(0, 0, 1),
        new Vector3Int(0, 1, 0),
        new Vector3Int(1, 1, 0),
        new Vector3Int(1, 1, 1),
        new Vector3Int(0, 1, 1),
    };

    private static readonly int[] EdgeCornerA = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
    private static readonly int[] EdgeCornerB = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

    private static readonly Vector3Int AxisX = new Vector3Int(1, 0, 0);
    private static readonly Vector3Int AxisY = new Vector3Int(0, 1, 0);
    private static readonly Vector3Int AxisZ = new Vector3Int(0, 0, 1);

    public static bool BuildMesh(
        float[] tsdf,
        float[] weights,
        Vector3Int volumeSize,
        float metersPerVoxel,
        float minWeight,
        Mesh mesh,
        out int triangleCount,
        out string issue)
    {
        return BuildMesh(tsdf, weights, volumeSize, metersPerVoxel, minWeight, mesh, out triangleCount, out issue, null, null);
    }

    public static bool BuildMesh(
        float[] tsdf,
        float[] weights,
        Vector3Int volumeSize,
        float metersPerVoxel,
        float minWeight,
        Mesh mesh,
        out int triangleCount,
        out string issue,
        List<int> cellIndicesOut,
        List<Vector3> normalsOut = null)
    {
        triangleCount = 0;
        issue = null;

        if (tsdf == null || weights == null || mesh == null)
        {
            if (mesh != null)
                mesh.Clear();
            issue = "TSDF mesh inputs are missing.";
            return false;
        }

        if (volumeSize.x < 2 || volumeSize.y < 2 || volumeSize.z < 2)
        {
            mesh.Clear();
            issue = "TSDF volume size is too small.";
            return false;
        }

        int expectedVoxelCount;
        try
        {
            expectedVoxelCount = checked(volumeSize.x * volumeSize.y * volumeSize.z);
        }
        catch (System.OverflowException)
        {
            mesh.Clear();
            issue = "TSDF volume size overflowed expected voxel count.";
            return false;
        }

        if (tsdf.Length < expectedVoxelCount || weights.Length < expectedVoxelCount)
        {
            mesh.Clear();
            issue = $"TSDF readback size mismatch. expected={expectedVoxelCount}, tsdf={tsdf.Length}, weights={weights.Length}.";
            return false;
        }

        int cellSizeX = volumeSize.x - 1;
        int cellSizeY = volumeSize.y - 1;
        int cellSizeZ = volumeSize.z - 1;
        int cellCount = cellSizeX * cellSizeY * cellSizeZ;
        var cellVertexIndices = new int[cellCount];
        for (int i = 0; i < cellVertexIndices.Length; i++)
            cellVertexIndices[i] = -1;

        var vertices = new List<Vector3>(Mathf.Max(1024, cellCount / 8));
        var triangles = new List<int>(Mathf.Max(2048, cellCount / 2));
        Vector3 halfVolume = 0.5f * new Vector3(volumeSize.x, volumeSize.y, volumeSize.z);
        float[] values = new float[8];
        bool[] observedFlags = new bool[8];
        Vector3[] positions = new Vector3[8];

        for (int z = 0; z < cellSizeZ; z++)
        {
            for (int y = 0; y < cellSizeY; y++)
            {
                for (int x = 0; x < cellSizeX; x++)
                {
                    Vector3Int cell = new Vector3Int(x, y, z);
                    bool hasNegative = false;
                    bool hasPositive = false;
                    bool hasObserved = false;

                    for (int c = 0; c < 8; c++)
                    {
                        Vector3Int coord = cell + CornerOffsets[c];
                        int dataIndex = GridIndex(coord, volumeSize);
                        if (dataIndex < 0 || dataIndex >= expectedVoxelCount)
                        {
                            mesh.Clear();
                            issue = $"TSDF grid index out of range at cell={cell}, corner={coord}, index={dataIndex}, expected={expectedVoxelCount}.";
                            return false;
                        }
                        bool observed = weights[dataIndex] >= minWeight;
                        // QuestRoomScan semantics: unobserved corner = TSDF 0
                        // (exactly at surface), NOT free space (+1).  This
                        // prevents phantom zero-crossings at the boundary of
                        // observed space — the root cause of mesh "skirts"
                        // and corner gaps.
                        float value = observed ? tsdf[dataIndex] : 0f;

                        values[c] = value;
                        observedFlags[c] = observed;
                        positions[c] = coord;
                        hasObserved |= observed;
                        hasNegative |= value < 0f;
                        hasPositive |= value >= 0f;
                    }

                    if (!hasObserved || !hasNegative || !hasPositive)
                        continue;

                    Vector3 vertexPosition = Vector3.zero;
                    Vector3 gradientDirection = Vector3.zero;
                    int crossingCount = 0;
                    int badCrossingCount = 0;

                    for (int edge = 0; edge < 12; edge++)
                    {
                        int a = EdgeCornerA[edge];
                        int b = EdgeCornerB[edge];
                        float va = values[a];
                        float vb = values[b];

                        // QuestRoomScan ClassifyAndEmit: the gradient accumulates
                        // over ALL 12 edges (not just crossings) and becomes the
                        // extraction normal used by the normal-aware smoother.
                        gradientDirection += (positions[a] - positions[b]) * (va - vb);

                        if ((va < 0f) == (vb < 0f))
                            continue;

                        float denom = va - vb;
                        if (Mathf.Abs(denom) < 1e-6f)
                            continue;

                        if (!observedFlags[a] || !observedFlags[b])
                            badCrossingCount++;

                        float t = Mathf.Clamp01(va / denom);
                        vertexPosition += Vector3.LerpUnclamped(positions[a], positions[b], t);
                        crossingCount++;
                    }

                    // QuestRoomScan extraction gates:
                    //   1. numCrossings >= 3 — at least 3 edge crossings to
                    //      form a reliable surface vertex (filters isolated
                    //      single-edge noise).
                    //   2. numCrossings != numBadCrossings — at least one
                    //      crossing between two confirmed voxels (prevents
                    //      surfaces from forming at the observed/unobserved
                    //      boundary).
                    if (crossingCount < 3 || crossingCount == badCrossingCount)
                        continue;

                    vertexPosition /= crossingCount;
                    vertexPosition = (vertexPosition - halfVolume) * metersPerVoxel;

                    int vertexIndex = vertices.Count;
                    vertices.Add(vertexPosition);
                    cellVertexIndices[CellIndex(cell, cellSizeX, cellSizeY)] = vertexIndex;
                    cellIndicesOut?.Add(CellIndex(cell, cellSizeX, cellSizeY));
                    if (normalsOut != null)
                    {
                        normalsOut.Add(gradientDirection.sqrMagnitude > 1e-12f
                            ? gradientDirection.normalized
                            : Vector3.up);
                    }
                }
            }
        }

        for (int z = 1; z < cellSizeZ; z++)
        {
            for (int y = 1; y < cellSizeY; y++)
            {
                for (int x = 1; x < cellSizeX; x++)
                {
                    Vector3Int cell = new Vector3Int(x, y, z);
                    AddFace(tsdf, weights, volumeSize, minWeight, cellVertexIndices, cellSizeX, cellSizeY, cell, AxisX, AxisZ, AxisY, triangles);
                    AddFace(tsdf, weights, volumeSize, minWeight, cellVertexIndices, cellSizeX, cellSizeY, cell, AxisY, AxisX, AxisZ, triangles);
                    AddFace(tsdf, weights, volumeSize, minWeight, cellVertexIndices, cellSizeX, cellSizeY, cell, AxisZ, AxisY, AxisX, triangles);
                }
            }
        }

        mesh.Clear();
        if (triangles.Count <= 0)
            return false;

        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0, true);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        triangleCount = triangles.Count / 3;
        return triangleCount > 0;
    }

    /// <summary>
    /// HC-Laplacian smoothing, faithful port of QuestRoomScan's
    /// SmoothVertices kernel:
    ///   - neighbors are the 6 grid-adjacent cells (via the cell-vertex map),
    ///     NOT triangle-edge adjacency (which adds diagonal neighbors);
    ///   - each neighbor is weighted by max(0, dot(n_i, n_j)) so corners
    ///     (perpendicular normals => ~0 weight) are preserved;
    ///   - ping-pong buffers (all vertices update simultaneously, no
    ///     in-place Gauss-Seidel bias).
    /// QuestRoomScan defaults: lambda=0.33, beta=0.5, 1 iteration.
    /// </summary>
    public static void SmoothMesh(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<int> cellIndices,
        Vector3Int volumeSize,
        float lambda,
        float beta,
        int iterations)
    {
        if (vertices == null || normals == null || cellIndices == null ||
            vertices.Count == 0 || iterations <= 0 ||
            vertices.Count != normals.Count || vertices.Count != cellIndices.Count)
            return;

        int cellSizeX = volumeSize.x - 1;
        int cellSizeY = volumeSize.y - 1;
        int cellSizeZ = volumeSize.z - 1;
        if (cellSizeX < 1 || cellSizeY < 1 || cellSizeZ < 1)
            return;

        // cell flat index -> vertex index (-1 = no vertex)
        int cellCount = cellSizeX * cellSizeY * cellSizeZ;
        var cellVertexMap = new int[cellCount];
        for (int i = 0; i < cellVertexMap.Length; i++)
            cellVertexMap[i] = -1;
        for (int i = 0; i < cellIndices.Count; i++)
        {
            int cellIdx = cellIndices[i];
            if (cellIdx >= 0 && cellIdx < cellCount)
                cellVertexMap[cellIdx] = i;
        }

        var original = vertices.ToArray();
        var bufferA = vertices.ToArray();
        var bufferB = new Vector3[vertices.Count];

        for (int iter = 0; iter < iterations; iter++)
        {
            for (int i = 0; i < bufferA.Length; i++)
            {
                int cellIdx = cellIndices[i];
                if (cellIdx < 0 || cellIdx >= cellCount)
                {
                    bufferB[i] = bufferA[i];
                    continue;
                }

                Vector3Int coord = UnflattenCell(cellIdx, cellSizeX, cellSizeY);
                Vector3 normal = normals[i];

                Vector3 laplacian = Vector3.zero;
                float totalWeight = 0f;

                for (int axis = 0; axis < 3; axis++)
                {
                    for (int dir = -1; dir <= 1; dir += 2)
                    {
                        Vector3Int nc = coord;
                        nc[axis] += dir;
                        if (nc.x < 0 || nc.y < 0 || nc.z < 0 ||
                            nc.x >= cellSizeX || nc.y >= cellSizeY || nc.z >= cellSizeZ)
                            continue;

                        int nbrVert = cellVertexMap[CellIndex(nc, cellSizeX, cellSizeY)];
                        if (nbrVert < 0)
                            continue;

                        float nw = Mathf.Max(0f, Vector3.Dot(normal, normals[nbrVert]));
                        laplacian += bufferA[nbrVert] * nw;
                        totalWeight += nw;
                    }
                }

                if (totalWeight < 0.001f)
                {
                    bufferB[i] = bufferA[i];
                    continue;
                }

                laplacian /= totalWeight;
                Vector3 q = Vector3.Lerp(bufferA[i], laplacian, lambda);
                // HC correction: pull back toward the original position to
                // prevent shrinkage.
                bufferB[i] = q - beta * (q - original[i]);
            }

            (bufferA, bufferB) = (bufferB, bufferA);
        }

        for (int i = 0; i < vertices.Count; i++)
            vertices[i] = bufferA[i];
    }

    private static Vector3Int UnflattenCell(int cellIdx, int cellSizeX, int cellSizeY)
    {
        int sliceXY = cellSizeX * cellSizeY;
        int z = cellIdx / sliceXY;
        int rem = cellIdx - z * sliceXY;
        int y = rem / cellSizeX;
        int x = rem - y * cellSizeX;
        return new Vector3Int(x, y, z);
    }

    private static void AddFace(
        float[] tsdf,
        float[] weights,
        Vector3Int volumeSize,
        float minWeight,
        int[] cellVertexIndices,
        int cellSizeX,
        int cellSizeY,
        Vector3Int cell,
        Vector3Int axis,
        Vector3Int d1,
        Vector3Int d2,
        List<int> triangles)
    {
        int ia = GridIndex(cell, volumeSize);
        int ib = GridIndex(cell + axis, volumeSize);
        // QuestRoomScan SampleSDF: low-weight or unobserved = 0 (surface),
        // not free space (+1).  This prevents quad generation from using
        // phantom crossings at the observed/unobserved boundary.
        float va = weights[ia] >= minWeight ? tsdf[ia] : 0f;
        float vb = weights[ib] >= minWeight ? tsdf[ib] : 0f;

        bool negA = va < 0f;
        bool negB = vb < 0f;
        if (negA == negB)
            return;

        int a = GetCellVertex(cellVertexIndices, cell, cellSizeX, cellSizeY);
        int b = GetCellVertex(cellVertexIndices, cell - d1, cellSizeX, cellSizeY);
        int c = GetCellVertex(cellVertexIndices, cell - d1 - d2, cellSizeX, cellSizeY);
        int d = GetCellVertex(cellVertexIndices, cell - d2, cellSizeX, cellSizeY);
        if (a < 0 || b < 0 || c < 0 || d < 0)
            return;

        if (negA)
        {
            triangles.Add(c);
            triangles.Add(b);
            triangles.Add(a);
            triangles.Add(d);
            triangles.Add(c);
            triangles.Add(a);
        }
        else
        {
            triangles.Add(a);
            triangles.Add(c);
            triangles.Add(d);
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
        }
    }

    private static int GetCellVertex(int[] cellVertexIndices, Vector3Int cell, int cellSizeX, int cellSizeY)
    {
        if (cell.x < 0 || cell.y < 0 || cell.z < 0)
            return -1;

        int cellSizeZ = cellVertexIndices.Length / (cellSizeX * cellSizeY);
        if (cell.x >= cellSizeX || cell.y >= cellSizeY || cell.z >= cellSizeZ)
            return -1;

        return cellVertexIndices[CellIndex(cell, cellSizeX, cellSizeY)];
    }

    private static int GridIndex(Vector3Int coord, Vector3Int size)
    {
        return coord.x + coord.y * size.x + coord.z * size.x * size.y;
    }

    private static int CellIndex(Vector3Int coord, int cellSizeX, int cellSizeY)
    {
        return coord.x + coord.y * cellSizeX + coord.z * cellSizeX * cellSizeY;
    }
}

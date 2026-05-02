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
                        float value = observed ? tsdf[dataIndex] : 1f;

                        values[c] = value;
                        positions[c] = coord;
                        hasObserved |= observed;
                        hasNegative |= value < 0f;
                        hasPositive |= value >= 0f;
                    }

                    if (!hasObserved || !hasNegative || !hasPositive)
                        continue;

                    Vector3 vertexPosition = Vector3.zero;
                    int crossingCount = 0;

                    for (int edge = 0; edge < 12; edge++)
                    {
                        int a = EdgeCornerA[edge];
                        int b = EdgeCornerB[edge];
                        float va = values[a];
                        float vb = values[b];
                        if ((va < 0f) == (vb < 0f))
                            continue;

                        float denom = va - vb;
                        if (Mathf.Abs(denom) < 1e-6f)
                            continue;

                        float t = Mathf.Clamp01(va / denom);
                        vertexPosition += Vector3.LerpUnclamped(positions[a], positions[b], t);
                        crossingCount++;
                    }

                    if (crossingCount <= 0)
                        continue;

                    vertexPosition /= crossingCount;
                    vertexPosition = (vertexPosition - halfVolume) * metersPerVoxel;

                    int vertexIndex = vertices.Count;
                    vertices.Add(vertexPosition);
                    cellVertexIndices[CellIndex(cell, cellSizeX, cellSizeY)] = vertexIndex;
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
        float va = weights[ia] >= minWeight ? tsdf[ia] : 1f;
        float vb = weights[ib] >= minWeight ? tsdf[ib] : 1f;

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

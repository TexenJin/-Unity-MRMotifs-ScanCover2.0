using System.Globalization;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

/// <summary>Spatial audit of the independent classic TSDF mesh boundary. Diagnostic only.</summary>
public static class ScanCoverReferenceBoundaryDiagnostics
{
    public sealed class Result
    {
        public sealed class BoundaryRecord
        {
            public int A;
            public int B;
            public Vector3 Midpoint;
            public string Classification;
            public int InternalComponent = -1;
        }

        public readonly List<Vector3> LineVertices = new List<Vector3>(8192);
        public readonly List<Color> LineColors = new List<Color>(8192);
        public readonly List<int> LineIndices = new List<int>(8192);
        public readonly StringBuilder Csv = new StringBuilder(65536);
        public readonly List<BoundaryRecord> Records = new List<BoundaryRecord>(4096);
        public int TotalEdges;
        public int VolumeLimitEdges;
        public int EnvelopeEdges;
        public int InternalEdges;
        public int InternalUnobservedEdges;
        public int InternalLowWeightEdges;
        public int InternalSupportedEdges;
        public int InternalComponentCount;
        public float TotalLength;
        public float InternalLength;
    }

    public static Result Build(
        ScanCoverReferenceVolumetricMesher.Result mesh,
        float[] weights,
        int dimX, int dimY, int dimZ,
        Vector3 origin, float voxelSize, float minimumWeight)
    {
        Result result = new Result();
        result.Csv.AppendLine("edge,a,b,ax,ay,az,bx,by,bz,mx,my,mz,length_m,class,eligible_neighbors,weak_neighbors,unobserved_neighbors,near_observed_envelope");
        if (mesh == null || weights == null || mesh.Vertices == null ||
            weights.Length < dimX * dimY * dimZ || dimX < 2 || dimY < 2 || dimZ < 2)
            return result;

        FindObservedBounds(weights, dimX, dimY, dimZ, minimumWeight,
            out Vector3Int observedMinimum, out Vector3Int observedMaximum);
        float voxel = Mathf.Max(0.0001f, voxelSize);
        for (int edgeIndex = 0; edgeIndex < mesh.BoundarySegments.Count; edgeIndex++)
        {
            ScanCoverReferenceVolumetricMesher.Result.BoundarySegment edge = mesh.BoundarySegments[edgeIndex];
            if (edge.A < 0 || edge.B < 0 || edge.A >= mesh.Vertices.Count || edge.B >= mesh.Vertices.Count)
                continue;
            Vector3 a = mesh.Vertices[edge.A];
            Vector3 b = mesh.Vertices[edge.B];
            Vector3 midpoint = (a + b) * 0.5f;
            Vector3 grid = (midpoint - origin) / voxel;
            int gx = Mathf.Clamp(Mathf.RoundToInt(grid.x), 0, dimX - 1);
            int gy = Mathf.Clamp(Mathf.RoundToInt(grid.y), 0, dimY - 1);
            int gz = Mathf.Clamp(Mathf.RoundToInt(grid.z), 0, dimZ - 1);

            int eligible = 0;
            int weak = 0;
            int unobserved = 0;
            for (int z = Mathf.Max(0, gz - 1); z <= Mathf.Min(dimZ - 1, gz + 1); z++)
            for (int y = Mathf.Max(0, gy - 1); y <= Mathf.Min(dimY - 1, gy + 1); y++)
            for (int x = Mathf.Max(0, gx - 1); x <= Mathf.Min(dimX - 1, gx + 1); x++)
            {
                float weight = weights[x + dimX * (y + dimY * z)];
                if (weight >= minimumWeight) eligible++;
                else if (weight > 0.0001f) weak++;
                else unobserved++;
            }

            bool volumeLimit = gx <= 1 || gy <= 1 || gz <= 1 ||
                               gx >= dimX - 2 || gy >= dimY - 2 || gz >= dimZ - 2;
            bool nearEnvelope = gx <= observedMinimum.x + 1 || gy <= observedMinimum.y + 1 || gz <= observedMinimum.z + 1 ||
                                gx >= observedMaximum.x - 1 || gy >= observedMaximum.y - 1 || gz >= observedMaximum.z - 1;
            string classification;
            Color color;
            if (volumeLimit)
            {
                classification = "volume_limit";
                color = new Color(0.55f, 0.55f, 0.55f, 1f);
                result.VolumeLimitEdges++;
            }
            else if (nearEnvelope)
            {
                classification = "observed_envelope";
                color = new Color(0.1f, 0.9f, 1f, 1f);
                result.EnvelopeEdges++;
            }
            else if (weak > 0)
            {
                classification = "internal_low_weight_gap";
                color = new Color(1f, 0.55f, 0.05f, 1f);
                result.InternalEdges++;
                result.InternalLowWeightEdges++;
            }
            else if (unobserved > 0)
            {
                classification = "internal_unobserved_gap";
                color = new Color(1f, 0.08f, 0.08f, 1f);
                result.InternalEdges++;
                result.InternalUnobservedEdges++;
            }
            else
            {
                classification = "internal_supported_open_edge";
                color = new Color(1f, 0.05f, 1f, 1f);
                result.InternalEdges++;
                result.InternalSupportedEdges++;
            }

            float length = Vector3.Distance(a, b);
            result.TotalEdges++;
            result.TotalLength += length;
            if (!volumeLimit && !nearEnvelope)
                result.InternalLength += length;
            int lineStart = result.LineVertices.Count;
            result.LineVertices.Add(a);
            result.LineVertices.Add(b);
            result.LineColors.Add(color);
            result.LineColors.Add(color);
            result.LineIndices.Add(lineStart);
            result.LineIndices.Add(lineStart + 1);
            result.Records.Add(new Result.BoundaryRecord
            {
                A = edge.A,
                B = edge.B,
                Midpoint = midpoint,
                Classification = classification
            });
            AppendRow(result.Csv, edgeIndex, edge.A, edge.B, a, b, midpoint, length,
                classification, eligible, weak, unobserved, nearEnvelope);
        }
        AssignInternalComponents(result);
        return result;
    }

    private static void AssignInternalComponents(Result result)
    {
        Dictionary<int, List<int>> edgesByVertex = new Dictionary<int, List<int>>();
        for (int edgeIndex = 0; edgeIndex < result.Records.Count; edgeIndex++)
        {
            Result.BoundaryRecord edge = result.Records[edgeIndex];
            if (!IsInternal(edge.Classification))
                continue;
            AddEdgeAtVertex(edgesByVertex, edge.A, edgeIndex);
            AddEdgeAtVertex(edgesByVertex, edge.B, edgeIndex);
        }

        Queue<int> queue = new Queue<int>();
        int component = 0;
        for (int edgeIndex = 0; edgeIndex < result.Records.Count; edgeIndex++)
        {
            Result.BoundaryRecord seed = result.Records[edgeIndex];
            if (!IsInternal(seed.Classification) || seed.InternalComponent >= 0)
                continue;
            seed.InternalComponent = component;
            queue.Enqueue(edgeIndex);
            while (queue.Count > 0)
            {
                Result.BoundaryRecord current = result.Records[queue.Dequeue()];
                AssignConnectedEdges(result, edgesByVertex, current.A, component, queue);
                AssignConnectedEdges(result, edgesByVertex, current.B, component, queue);
            }
            component++;
        }
        result.InternalComponentCount = component;
    }

    private static void AddEdgeAtVertex(Dictionary<int, List<int>> edgesByVertex, int vertex, int edge)
    {
        if (!edgesByVertex.TryGetValue(vertex, out List<int> edges))
        {
            edges = new List<int>(4);
            edgesByVertex[vertex] = edges;
        }
        edges.Add(edge);
    }

    private static void AssignConnectedEdges(
        Result result,
        Dictionary<int, List<int>> edgesByVertex,
        int vertex,
        int component,
        Queue<int> queue)
    {
        if (!edgesByVertex.TryGetValue(vertex, out List<int> edges))
            return;
        for (int i = 0; i < edges.Count; i++)
        {
            Result.BoundaryRecord neighbor = result.Records[edges[i]];
            if (neighbor.InternalComponent >= 0)
                continue;
            neighbor.InternalComponent = component;
            queue.Enqueue(edges[i]);
        }
    }

    private static bool IsInternal(string classification)
    {
        return classification == "internal_low_weight_gap" ||
               classification == "internal_unobserved_gap" ||
               classification == "internal_supported_open_edge";
    }

    private static void FindObservedBounds(float[] weights, int dimX, int dimY, int dimZ, float minimumWeight,
        out Vector3Int minimum, out Vector3Int maximum)
    {
        minimum = new Vector3Int(dimX, dimY, dimZ);
        maximum = new Vector3Int(-1, -1, -1);
        int plane = dimX * dimY;
        for (int index = 0; index < weights.Length; index++)
        {
            if (weights[index] < minimumWeight) continue;
            int z = index / plane;
            int remainder = index - z * plane;
            int y = remainder / dimX;
            int x = remainder - y * dimX;
            minimum = Vector3Int.Min(minimum, new Vector3Int(x, y, z));
            maximum = Vector3Int.Max(maximum, new Vector3Int(x, y, z));
        }
    }

    private static void AppendRow(StringBuilder csv, int edge, int ai, int bi,
        Vector3 a, Vector3 b, Vector3 midpoint, float length, string classification,
        int eligible, int weak, int unobserved, bool nearEnvelope)
    {
        CultureInfo c = CultureInfo.InvariantCulture;
        csv.Append(edge).Append(',').Append(ai).Append(',').Append(bi).Append(',')
            .Append(a.x.ToString("R", c)).Append(',').Append(a.y.ToString("R", c)).Append(',').Append(a.z.ToString("R", c)).Append(',')
            .Append(b.x.ToString("R", c)).Append(',').Append(b.y.ToString("R", c)).Append(',').Append(b.z.ToString("R", c)).Append(',')
            .Append(midpoint.x.ToString("R", c)).Append(',').Append(midpoint.y.ToString("R", c)).Append(',').Append(midpoint.z.ToString("R", c)).Append(',')
            .Append(length.ToString("R", c)).Append(',').Append(classification).Append(',')
            .Append(eligible).Append(',').Append(weak).Append(',').Append(unobserved).Append(',')
            .Append(nearEnvelope ? 1 : 0).AppendLine();
    }
}

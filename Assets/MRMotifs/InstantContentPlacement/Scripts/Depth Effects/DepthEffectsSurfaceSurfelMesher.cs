using System;
using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsSurfaceSurfelMesher : MonoBehaviour
    {
        private readonly struct TriangleKey : IEquatable<TriangleKey>
        {
            private readonly int a;
            private readonly int b;
            private readonly int c;

            public TriangleKey(int i0, int i1, int i2)
            {
                if (i0 > i1) (i0, i1) = (i1, i0);
                if (i1 > i2) (i1, i2) = (i2, i1);
                if (i0 > i1) (i0, i1) = (i1, i0);
                a = i0;
                b = i1;
                c = i2;
            }

            public bool Equals(TriangleKey other) => a == other.a && b == other.b && c == other.c;
            public override bool Equals(object obj) => obj is TriangleKey other && Equals(other);
            public override int GetHashCode() => HashCode.Combine(a, b, c);
        }

        private readonly struct EdgeKey : IEquatable<EdgeKey>
        {
            private readonly int a;
            private readonly int b;

            public EdgeKey(int i0, int i1)
            {
                if (i0 > i1) (i0, i1) = (i1, i0);
                a = i0;
                b = i1;
            }

            public bool Equals(EdgeKey other) => a == other.a && b == other.b;
            public override bool Equals(object obj) => obj is EdgeKey other && Equals(other);
            public override int GetHashCode() => HashCode.Combine(a, b);
        }

        [SerializeField]
        private DepthEffectsSurfaceSurfelAccumulator surfelAccumulator;

        [SerializeField]
        private Camera sampleCamera;

        [Header("Meshing")]
        [SerializeField]
        private float rebuildIntervalSeconds = 0.12f;

        [SerializeField]
        private int minSupportCount = 3;

        [SerializeField]
        private float minConfidence = 0.26f;

        [SerializeField]
        private float neighborRadiusMeters = 0.28f;

        [SerializeField]
        private float maxTriangleEdgeMeters = 0.34f;

        [SerializeField]
        private float minTriangleNormalDot = 0.74f;

        [SerializeField]
        private int maxNeighborsPerSurfel = 10;

        [SerializeField]
        private int maxSurfelsForMeshing = 1800;

        [SerializeField]
        private float maxTriangleAspectRatio = 2.8f;

        [SerializeField]
        private float minTriangleAreaMeters2 = 0.0025f;

        [Header("Surface Clustering")]
        [SerializeField]
        private int minClusterSurfels = 4;

        [SerializeField]
        private float clusterNeighborRadiusMeters = 0.32f;

        [SerializeField]
        private float clusterPlaneThicknessMeters = 0.14f;

        [SerializeField]
        private float uvNeighborRadiusMeters = 0.26f;

        [SerializeField]
        private int maxUvNeighborsPerSurfel = 8;

        [Header("Regularization")]
        [SerializeField]
        private float regularizedPointSpacingMeters = 0.16f;

        [SerializeField]
        private float regularizedPointBlend = 0.72f;

        [SerializeField]
        private int maxEdgesPerVertex = 4;

        private readonly List<DepthEffectsSurfaceSurfelAccumulator.SurfelData> m_surfels = new();
        private readonly List<Vector3> m_vertices = new();
        private readonly List<int> m_lineIndices = new();
        private readonly List<Vector3> m_nodePositions = new();
        private readonly List<int> m_neighborIndices = new();
        private readonly List<(int index, float angle)> m_sortedNeighbors = new();
        private readonly List<Vector2> m_clusterUvs = new();
        private readonly List<Vector3> m_clusterWorldPoints = new();
        private readonly List<int> m_clusterIndices = new();
        private readonly List<int> m_clusterQueue = new();
        private readonly HashSet<int> m_clusterVisited = new();
        private readonly HashSet<TriangleKey> m_triangleKeys = new();
        private readonly HashSet<EdgeKey> m_edgeKeys = new();
        private readonly Dictionary<int, int> m_vertexDegrees = new();
        private float m_nextRebuildTime;

        private struct FittedPlaneFrame
        {
            public Vector3 origin;
            public Vector3 normal;
            public Vector3 tangent;
            public Vector3 bitangent;
        }

        public int CopyWireframe(List<Vector3> vertices, List<int> lineIndices)
        {
            vertices.Clear();
            lineIndices.Clear();
            vertices.AddRange(m_vertices);
            lineIndices.AddRange(m_lineIndices);
            return m_lineIndices.Count;
        }

        public int CopyNodes(List<Vector3> positions)
        {
            positions.Clear();
            positions.AddRange(m_nodePositions);
            return m_nodePositions.Count;
        }

        public void Configure(DepthEffectsSurfaceSurfelAccumulator accumulator, Camera camera)
        {
            surfelAccumulator = accumulator;
            sampleCamera = camera;
        }

        private void LateUpdate()
        {
            ResolveRefs();
            if (surfelAccumulator == null || sampleCamera == null)
                return;

            if (Time.unscaledTime < m_nextRebuildTime)
                return;

            m_nextRebuildTime = Time.unscaledTime + Mathf.Max(0.04f, rebuildIntervalSeconds);
            RebuildWireframe(Time.unscaledTime);
        }

        private void ResolveRefs()
        {
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (surfelAccumulator == null)
                surfelAccumulator = GetComponent<DepthEffectsSurfaceSurfelAccumulator>();
        }

        private void RebuildWireframe(float now)
        {
            surfelAccumulator.FillSurfels(m_surfels, now);
            m_vertices.Clear();
            m_lineIndices.Clear();
            m_nodePositions.Clear();
            m_triangleKeys.Clear();
            m_edgeKeys.Clear();
            m_vertexDegrees.Clear();

            if (m_surfels.Count <= 0)
                return;

            int surfelCount = Mathf.Min(m_surfels.Count, Mathf.Max(32, maxSurfelsForMeshing));
            for (int i = 0; i < surfelCount; i++)
                m_vertices.Add(Vector3.zero);

            m_clusterVisited.Clear();
            for (int i = 0; i < surfelCount; i++)
            {
                DepthEffectsSurfaceSurfelAccumulator.SurfelData center = m_surfels[i];
                if (!IsMeshingCandidate(center))
                    continue;
                if (m_clusterVisited.Contains(i))
                    continue;

                BuildCluster(i, surfelCount, m_clusterIndices);
                if (m_clusterIndices.Count < Mathf.Max(3, minClusterSurfels))
                    continue;

                TriangulateClusterInPlane(m_clusterIndices);
            }
        }

        private void BuildCluster(int seedIndex, int surfelCount, List<int> clusterIndices)
        {
            clusterIndices.Clear();
            m_clusterQueue.Clear();

            DepthEffectsSurfaceSurfelAccumulator.SurfelData seed = m_surfels[seedIndex];
            int dominantAxis = ResolveDominantAxis(seed.normal);
            float seedSlice = GetAxisValue(seed.position, dominantAxis);
            Vector3 seedNormal = seed.normal.sqrMagnitude > 1e-6f ? seed.normal.normalized : Vector3.up;

            m_clusterVisited.Add(seedIndex);
            m_clusterQueue.Add(seedIndex);

            for (int queueIndex = 0; queueIndex < m_clusterQueue.Count; queueIndex++)
            {
                int current = m_clusterQueue[queueIndex];
                clusterIndices.Add(current);
                DepthEffectsSurfaceSurfelAccumulator.SurfelData currentSurfel = m_surfels[current];

                for (int j = 0; j < surfelCount; j++)
                {
                    if (m_clusterVisited.Contains(j))
                        continue;

                    DepthEffectsSurfaceSurfelAccumulator.SurfelData candidate = m_surfels[j];
                    if (!IsMeshingCandidate(candidate))
                        continue;
                    if (ResolveDominantAxis(candidate.normal) != dominantAxis)
                        continue;

                    float sliceDistance = Mathf.Abs(GetAxisValue(candidate.position, dominantAxis) - seedSlice);
                    if (sliceDistance > Mathf.Max(0.02f, clusterPlaneThicknessMeters))
                        continue;

                    float distance = Vector3.Distance(currentSurfel.position, candidate.position);
                    if (distance > Mathf.Max(0.05f, clusterNeighborRadiusMeters))
                        continue;

                    float normalDot = Mathf.Abs(Vector3.Dot(
                        candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : Vector3.up,
                        seedNormal));
                    if (normalDot < Mathf.Clamp(minTriangleNormalDot, 0f, 0.9999f))
                        continue;

                    m_clusterVisited.Add(j);
                    m_clusterQueue.Add(j);
                }
            }
        }

        private void TriangulateClusterInPlane(List<int> clusterIndices)
        {
            m_clusterUvs.Clear();
            m_clusterWorldPoints.Clear();
            if (clusterIndices.Count <= 0)
                return;

            FittedPlaneFrame frame = FitClusterPlane(clusterIndices);
            for (int i = 0; i < clusterIndices.Count; i++)
            {
                Vector3 worldPos = m_surfels[clusterIndices[i]].position;
                Vector2 uv = ProjectToPlaneUv(worldPos, frame);
                Vector2 snappedUv = SnapUvToRegularGrid(uv);
                Vector2 regularizedUv = Vector2.Lerp(uv, snappedUv, Mathf.Clamp01(regularizedPointBlend));
                m_clusterUvs.Add(regularizedUv);
                m_clusterWorldPoints.Add(ExpandPlaneUvToWorld(regularizedUv, frame, worldPos));
            }

            for (int i = 0; i < clusterIndices.Count; i++)
            {
                m_vertices[clusterIndices[i]] = m_clusterWorldPoints[i];
                m_nodePositions.Add(m_clusterWorldPoints[i]);
            }

            for (int localIndex = 0; localIndex < clusterIndices.Count; localIndex++)
            {
                int centerIndex = clusterIndices[localIndex];
                Vector3 centerWorld = m_clusterWorldPoints[localIndex];

                m_neighborIndices.Clear();
                for (int otherLocal = 0; otherLocal < clusterIndices.Count; otherLocal++)
                {
                    if (otherLocal == localIndex)
                        continue;

                    int neighborIndex = clusterIndices[otherLocal];
                    float distance3D = Vector3.Distance(centerWorld, m_clusterWorldPoints[otherLocal]);
                    if (distance3D > Mathf.Max(0.05f, maxTriangleEdgeMeters))
                        continue;

                    float distance2D = Vector2.Distance(m_clusterUvs[localIndex], m_clusterUvs[otherLocal]);
                    if (distance2D > Mathf.Max(0.03f, uvNeighborRadiusMeters))
                        continue;

                    m_neighborIndices.Add(otherLocal);
                }

                if (m_neighborIndices.Count < 2)
                    continue;

                m_sortedNeighbors.Clear();
                BuildSortedNeighborAnglesInPlane(localIndex, m_neighborIndices, m_sortedNeighbors);
                if (m_sortedNeighbors.Count > maxUvNeighborsPerSurfel)
                    m_sortedNeighbors.RemoveRange(maxUvNeighborsPerSurfel, m_sortedNeighbors.Count - maxUvNeighborsPerSurfel);

                for (int n = 0; n < m_sortedNeighbors.Count; n++)
                {
                    int leftLocal = m_sortedNeighbors[n].index;
                    int rightLocal = m_sortedNeighbors[(n + 1) % m_sortedNeighbors.Count].index;
                    int leftIndex = clusterIndices[leftLocal];
                    int rightIndex = clusterIndices[rightLocal];

                    if (!IsValidTriangle(centerIndex, leftIndex, rightIndex))
                        continue;

                    TriangleKey triangleKey = new(centerIndex, leftIndex, rightIndex);
                    if (!m_triangleKeys.Add(triangleKey))
                        continue;

                    AddEdge(centerIndex, leftIndex);
                    AddEdge(leftIndex, rightIndex);
                    AddEdge(rightIndex, centerIndex);
                }
            }
        }

        private void BuildSortedNeighborAnglesInPlane(int centerLocalIndex, List<int> neighborLocals, List<(int index, float angle)> output)
        {
            Vector2 centerUv = m_clusterUvs[centerLocalIndex];
            for (int i = 0; i < neighborLocals.Count; i++)
            {
                int neighborLocal = neighborLocals[i];
                Vector2 dir = m_clusterUvs[neighborLocal] - centerUv;
                if (dir.sqrMagnitude <= 1e-6f)
                    continue;

                float angle = Mathf.Atan2(dir.y, dir.x);
                output.Add((neighborLocal, angle));
            }

            output.Sort((a, b) => a.angle.CompareTo(b.angle));
        }

        private bool IsValidTriangle(int i0, int i1, int i2)
        {
            Vector3 p0 = m_surfels[i0].position;
            Vector3 p1 = m_surfels[i1].position;
            Vector3 p2 = m_surfels[i2].position;

            float e01 = Vector3.Distance(p0, p1);
            float e12 = Vector3.Distance(p1, p2);
            float e20 = Vector3.Distance(p2, p0);
            float maxEdge = Mathf.Max(0.05f, maxTriangleEdgeMeters);
            if (e01 > maxEdge || e12 > maxEdge || e20 > maxEdge)
                return false;

            float shortestEdge = Mathf.Max(1e-4f, Mathf.Min(e01, Mathf.Min(e12, e20)));
            float longestEdge = Mathf.Max(e01, Mathf.Max(e12, e20));
            if (longestEdge / shortestEdge > Mathf.Max(1.1f, maxTriangleAspectRatio))
                return false;

            Vector3 triNormal = Vector3.Cross(p1 - p0, p2 - p0);
            if (triNormal.sqrMagnitude <= 1e-6f)
                return false;

            float area = triNormal.magnitude * 0.5f;
            if (area < Mathf.Max(0.0005f, minTriangleAreaMeters2))
                return false;

            triNormal.Normalize();
            float n0 = Mathf.Abs(Vector3.Dot(triNormal, m_surfels[i0].normal.normalized));
            float n1 = Mathf.Abs(Vector3.Dot(triNormal, m_surfels[i1].normal.normalized));
            float n2 = Mathf.Abs(Vector3.Dot(triNormal, m_surfels[i2].normal.normalized));
            return n0 >= minTriangleNormalDot && n1 >= minTriangleNormalDot && n2 >= minTriangleNormalDot;
        }

        private void AddEdge(int a, int b)
        {
            EdgeKey edge = new(a, b);
            if (!m_edgeKeys.Add(edge))
                return;
            if (!CanAddEdgeForVertex(a) || !CanAddEdgeForVertex(b))
            {
                m_edgeKeys.Remove(edge);
                return;
            }

            m_lineIndices.Add(a);
            m_lineIndices.Add(b);
            IncrementVertexDegree(a);
            IncrementVertexDegree(b);
        }

        private bool IsMeshingCandidate(DepthEffectsSurfaceSurfelAccumulator.SurfelData surfel)
        {
            if (surfel.supportCount < minSupportCount)
                return false;
            if (surfel.confidence < minConfidence)
                return false;
            return surfel.activeSupport || surfel.retained;
        }

        private static float GetAxisValue(Vector3 value, int axis)
        {
            return axis switch
            {
                0 => value.x,
                1 => value.y,
                _ => value.z
            };
        }

        private static Vector2 ProjectToPlaneUv(Vector3 worldPos, FittedPlaneFrame frame)
        {
            Vector3 offset = worldPos - frame.origin;
            return new Vector2(Vector3.Dot(offset, frame.tangent), Vector3.Dot(offset, frame.bitangent));
        }

        private Vector2 SnapUvToRegularGrid(Vector2 uv)
        {
            float spacing = Mathf.Max(0.02f, regularizedPointSpacingMeters);
            return new Vector2(
                Mathf.Round(uv.x / spacing) * spacing,
                Mathf.Round(uv.y / spacing) * spacing);
        }

        private static Vector3 ExpandPlaneUvToWorld(Vector2 uv, FittedPlaneFrame frame, Vector3 sourceWorldPos)
        {
            float planeOffset = Vector3.Dot(sourceWorldPos - frame.origin, frame.normal);
            return frame.origin + frame.tangent * uv.x + frame.bitangent * uv.y + frame.normal * planeOffset;
        }

        private FittedPlaneFrame FitClusterPlane(List<int> clusterIndices)
        {
            Vector3 origin = Vector3.zero;
            Vector3 normalSum = Vector3.zero;
            for (int i = 0; i < clusterIndices.Count; i++)
            {
                DepthEffectsSurfaceSurfelAccumulator.SurfelData surfel = m_surfels[clusterIndices[i]];
                origin += surfel.position;
                Vector3 surfelNormal = surfel.normal.sqrMagnitude > 1e-6f ? surfel.normal.normalized : Vector3.up;
                normalSum += surfelNormal;
            }

            origin /= Mathf.Max(1, clusterIndices.Count);
            Vector3 normal = normalSum.sqrMagnitude > 1e-6f ? normalSum.normalized : Vector3.up;

            // Estimate the dominant in-plane direction from the cluster spread, projected onto the fitted plane.
            Vector3 tangent = Vector3.zero;
            float bestSpread = 0f;
            for (int i = 0; i < clusterIndices.Count; i++)
            {
                Vector3 offset = m_surfels[clusterIndices[i]].position - origin;
                offset -= Vector3.Dot(offset, normal) * normal;
                float spread = offset.sqrMagnitude;
                if (spread > bestSpread)
                {
                    bestSpread = spread;
                    tangent = offset;
                }
            }

            if (tangent.sqrMagnitude <= 1e-6f)
            {
                tangent = Vector3.Cross(normal, Vector3.up);
                if (tangent.sqrMagnitude <= 1e-6f)
                    tangent = Vector3.Cross(normal, Vector3.right);
            }

            tangent.Normalize();
            Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;
            tangent = Vector3.Cross(bitangent, normal).normalized;

            return new FittedPlaneFrame
            {
                origin = origin,
                normal = normal,
                tangent = tangent,
                bitangent = bitangent
            };
        }

        private bool CanAddEdgeForVertex(int vertexIndex)
        {
            return !m_vertexDegrees.TryGetValue(vertexIndex, out int degree) || degree < Mathf.Max(1, maxEdgesPerVertex);
        }

        private void IncrementVertexDegree(int vertexIndex)
        {
            m_vertexDegrees.TryGetValue(vertexIndex, out int degree);
            m_vertexDegrees[vertexIndex] = degree + 1;
        }

        private static int ResolveDominantAxis(Vector3 normal)
        {
            Vector3 absNormal = new(Mathf.Abs(normal.x), Mathf.Abs(normal.y), Mathf.Abs(normal.z));
            if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
                return 1;
            return absNormal.x >= absNormal.z ? 0 : 2;
        }
    }
}

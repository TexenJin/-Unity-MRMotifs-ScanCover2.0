// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [Meta.XR.Samples.MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsStablePlaneRenderer : MonoBehaviour
    {
        private const int MaxInstancesPerDraw = 1023;

        private struct PlaneBucketKey : IEquatable<PlaneBucketKey>
        {
            public int axisIndex;
            public int distanceBucket;

            public bool Equals(PlaneBucketKey other)
            {
                return axisIndex == other.axisIndex && distanceBucket == other.distanceBucket;
            }

            public override bool Equals(object obj)
            {
                return obj is PlaneBucketKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    return (axisIndex * 397) ^ distanceBucket;
                }
            }
        }

        private struct PlaneGridKey : IEquatable<PlaneGridKey>
        {
            public int u;
            public int v;

            public bool Equals(PlaneGridKey other)
            {
                return u == other.u && v == other.v;
            }

            public override bool Equals(object obj)
            {
                return obj is PlaneGridKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    return (u * 397) ^ v;
                }
            }
        }

        private sealed class PlaneComponentAccumulator
        {
            public float minU = float.PositiveInfinity;
            public float maxU = float.NegativeInfinity;
            public float minV = float.PositiveInfinity;
            public float maxV = float.NegativeInfinity;
            public int pointCount;
            public int occupiedCount;
        }

        private sealed class PlaneCellAggregate
        {
            public float minU = float.PositiveInfinity;
            public float maxU = float.NegativeInfinity;
            public float minV = float.PositiveInfinity;
            public float maxV = float.NegativeInfinity;
            public int pointCount;
        }

        private sealed class PlaneBucketData
        {
            public int axisIndex;
            public float planeDistance;
            public Vector3 normal;
            public Vector3 tangent;
            public Vector3 bitangent;
            public readonly List<ScanCoverVoxelPointAccumulator.StablePoint> points =
                new List<ScanCoverVoxelPointAccumulator.StablePoint>(64);
        }

        private struct PlaneCandidate
        {
            public Vector3 center;
            public Vector3 normal;
            public Vector3 tangent;
            public Vector3 bitangent;
            public float width;
            public float height;
            public int pointCount;
            public int occupiedCells;
            public float score;
        }

        [Header("Refs")]
        [SerializeField]
        private ScanCoverVoxelPointAccumulator voxelPointAccumulator;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private Material planeMaterialOverride;

        [Header("Display")]
        [SerializeField]
        private bool showStablePlanes = true;

        [SerializeField]
        private Color planeColor = new Color(0.92f, 0.96f, 0.99f, 0.96f);

        [SerializeField]
        private float displayRefreshIntervalSeconds = 0.12f;

        [SerializeField]
        private float surfaceOffsetMeters = 0.01f;

        [SerializeField]
        private float frameThicknessRatio = 0.08f;

        [SerializeField]
        private int maxDisplayPlanes = 24;

        [Header("Plane Extraction")]
        [SerializeField]
        private float planeDistanceQuantizationMeters = 0.2f;

        [SerializeField]
        private float occupancyCellSizeMeters = 0.22f;

        [SerializeField]
        private int minPlanePointCount = 6;

        [SerializeField]
        private int minOccupiedCells = 3;

        [SerializeField]
        private float minPlaneWidthMeters = 0.25f;

        [SerializeField]
        private float minPlaneHeightMeters = 0.25f;

        [SerializeField]
        private float maxDisplayDistanceMeters = 4.5f;

        [Header("Debug")]
        [SerializeField]
        private bool debugLog = true;

        public int LastStablePointCount { get; private set; }
        public int LastPlaneBucketCount { get; private set; }
        public int LastCandidatePlaneCount { get; private set; }
        public int LastPlaneCount { get; private set; }
        public int LastDrawBatchCount { get; private set; }

        private readonly List<ScanCoverVoxelPointAccumulator.StablePoint> m_stablePoints =
            new List<ScanCoverVoxelPointAccumulator.StablePoint>(4096);
        private readonly Dictionary<PlaneBucketKey, PlaneBucketData> m_planeBuckets =
            new Dictionary<PlaneBucketKey, PlaneBucketData>(128);
        private readonly HashSet<PlaneGridKey> m_occupiedCells =
            new HashSet<PlaneGridKey>();
        private readonly HashSet<PlaneGridKey> m_visitedCells =
            new HashSet<PlaneGridKey>();
        private readonly Queue<PlaneGridKey> m_componentQueue =
            new Queue<PlaneGridKey>();
        private readonly List<PlaneCandidate> m_planeCandidates =
            new List<PlaneCandidate>(64);
        private readonly List<Matrix4x4[]> m_cachedDrawBatches =
            new List<Matrix4x4[]>(8);
        private readonly List<int> m_cachedDrawCounts =
            new List<int>(8);

        private MaterialPropertyBlock m_propertyBlock;
        private Material m_runtimeMaterial;
        private Mesh m_frameMesh;
        private int m_lastRenderedRevision = -1;
        private float m_nextAllowedRefreshTime;
        private bool m_loggedMaterialFailure;
        private float m_nextDebugLogTime;

        public void Configure(ScanCoverVoxelPointAccumulator accumulator, Camera camera, Material materialOverride)
        {
            voxelPointAccumulator = accumulator;
            sampleCamera = camera;
            planeMaterialOverride = materialOverride;
        }

        private void Awake()
        {
            ResolveRefs();
            Debug.LogWarning("[DepthEffectsStablePlaneRenderer] Awake");
        }

        private void OnEnable()
        {
            Debug.LogWarning("[DepthEffectsStablePlaneRenderer] OnEnable");
        }

        private void LateUpdate()
        {
            ResolveRefs();
            if (!showStablePlanes || voxelPointAccumulator == null)
            {
                LastStablePointCount = 0;
                LastPlaneBucketCount = 0;
                LastCandidatePlaneCount = 0;
                LastPlaneCount = 0;
                LastDrawBatchCount = 0;
                return;
            }

            RebuildDisplayCacheIfNeeded();
            if (m_cachedDrawBatches.Count <= 0)
            {
                LastPlaneCount = 0;
                LastDrawBatchCount = 0;
                MaybeLogDiagnostics("No cached draw batches");
                return;
            }

            Mesh mesh = ResolveFrameMesh();
            Material material = ResolveMaterial();
            if (mesh == null || material == null)
            {
                MaybeLogDiagnostics(mesh == null ? "Frame mesh missing" : "Plane material missing");
                return;
            }

            EnsurePropertyBlock();
            m_propertyBlock.SetColor("_BaseColor", planeColor);
            m_propertyBlock.SetColor("_Color", planeColor);
            LastDrawBatchCount = m_cachedDrawBatches.Count;

            for (int i = 0; i < m_cachedDrawBatches.Count; i++)
            {
                Graphics.DrawMeshInstanced(
                    mesh,
                    0,
                    material,
                    m_cachedDrawBatches[i],
                    m_cachedDrawCounts[i],
                    m_propertyBlock,
                    ShadowCastingMode.Off,
                    false,
                    gameObject.layer);
            }
        }

        private void ResolveRefs()
        {
            if (voxelPointAccumulator == null)
                voxelPointAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();
            if (sampleCamera == null)
                sampleCamera = Camera.main;
        }

        private void RebuildDisplayCacheIfNeeded()
        {
            int revision = voxelPointAccumulator != null ? voxelPointAccumulator.Revision : -1;
            float now = Time.unscaledTime;
            if (revision == m_lastRenderedRevision && now < m_nextAllowedRefreshTime)
                return;

            voxelPointAccumulator.GetStablePointsNonAlloc(m_stablePoints);
            LastStablePointCount = m_stablePoints.Count;
            BuildPlaneCandidates();
            BuildDrawBatches();
            LastPlaneCount = m_planeCandidates.Count;
            m_lastRenderedRevision = revision;
            m_nextAllowedRefreshTime = now + Mathf.Max(0.05f, displayRefreshIntervalSeconds);
            MaybeLogDiagnostics("Rebuilt display cache");
        }

        private void BuildPlaneCandidates()
        {
            m_planeCandidates.Clear();
            m_planeBuckets.Clear();
            LastPlaneBucketCount = 0;
            LastCandidatePlaneCount = 0;
            if (m_stablePoints.Count <= 0)
                return;

            float distanceQuantization = Mathf.Max(0.05f, planeDistanceQuantizationMeters);
            float maxDistanceSq = Mathf.Max(0.25f, maxDisplayDistanceMeters) * Mathf.Max(0.25f, maxDisplayDistanceMeters);
            Vector3 cameraPos = sampleCamera != null ? sampleCamera.transform.position : Vector3.zero;

            for (int i = 0; i < m_stablePoints.Count; i++)
            {
                ScanCoverVoxelPointAccumulator.StablePoint point = m_stablePoints[i];
                if (sampleCamera != null && (point.worldPos - cameraPos).sqrMagnitude > maxDistanceSq)
                    continue;

                int axisIndex = ResolveAxisIndex(point.normal);
                Vector3 axisNormal = ResolveAxisNormal(axisIndex);
                float planeDistance = Vector3.Dot(axisNormal, point.worldPos);
                PlaneBucketKey key = new PlaneBucketKey
                {
                    axisIndex = axisIndex,
                    distanceBucket = Mathf.RoundToInt(planeDistance / distanceQuantization),
                };

                if (!m_planeBuckets.TryGetValue(key, out PlaneBucketData bucket))
                {
                    ResolvePlaneBasis(axisIndex, out Vector3 tangent, out Vector3 bitangent);
                    bucket = new PlaneBucketData
                    {
                        axisIndex = axisIndex,
                        planeDistance = key.distanceBucket * distanceQuantization,
                        normal = axisNormal,
                        tangent = tangent,
                        bitangent = bitangent,
                    };
                    m_planeBuckets.Add(key, bucket);
                }

                bucket.points.Add(point);
            }

            foreach (KeyValuePair<PlaneBucketKey, PlaneBucketData> pair in m_planeBuckets)
                TryCreatePlaneCandidate(pair.Value);

            LastPlaneBucketCount = m_planeBuckets.Count;
            LastCandidatePlaneCount = m_planeCandidates.Count;
            m_planeCandidates.Sort((a, b) => b.score.CompareTo(a.score));
            int maxPlanes = Mathf.Max(1, maxDisplayPlanes);
            if (m_planeCandidates.Count > maxPlanes)
                m_planeCandidates.RemoveRange(maxPlanes, m_planeCandidates.Count - maxPlanes);
        }

        private void TryCreatePlaneCandidate(PlaneBucketData bucket)
        {
            if (bucket.points.Count < Mathf.Max(4, minPlanePointCount))
                return;

            float spacing = Mathf.Max(0.05f, occupancyCellSizeMeters);
            m_occupiedCells.Clear();
            m_visitedCells.Clear();
            Dictionary<PlaneGridKey, PlaneCellAggregate> cellAggregates =
                new Dictionary<PlaneGridKey, PlaneCellAggregate>(bucket.points.Count);

            for (int i = 0; i < bucket.points.Count; i++)
            {
                Vector3 pos = bucket.points[i].worldPos;
                float u = Vector3.Dot(pos, bucket.tangent);
                float v = Vector3.Dot(pos, bucket.bitangent);
                PlaneGridKey key = new PlaneGridKey
                {
                    u = Mathf.RoundToInt(u / spacing),
                    v = Mathf.RoundToInt(v / spacing),
                };

                m_occupiedCells.Add(key);
                if (!cellAggregates.TryGetValue(key, out PlaneCellAggregate aggregate))
                {
                    aggregate = new PlaneCellAggregate();
                    cellAggregates.Add(key, aggregate);
                }

                aggregate.minU = Mathf.Min(aggregate.minU, u);
                aggregate.maxU = Mathf.Max(aggregate.maxU, u);
                aggregate.minV = Mathf.Min(aggregate.minV, v);
                aggregate.maxV = Mathf.Max(aggregate.maxV, v);
                aggregate.pointCount++;
            }

            foreach (PlaneGridKey origin in m_occupiedCells)
            {
                if (m_visitedCells.Contains(origin))
                    continue;

                PlaneComponentAccumulator component = ExtractConnectedComponent(origin, cellAggregates);
                TryCreateCandidateFromComponent(bucket, component);
            }
        }

        private PlaneComponentAccumulator ExtractConnectedComponent(
            PlaneGridKey origin,
            Dictionary<PlaneGridKey, PlaneCellAggregate> cellAggregates)
        {
            PlaneComponentAccumulator component = new PlaneComponentAccumulator();
            m_componentQueue.Clear();
            m_componentQueue.Enqueue(origin);
            m_visitedCells.Add(origin);

            while (m_componentQueue.Count > 0)
            {
                PlaneGridKey key = m_componentQueue.Dequeue();
                if (!cellAggregates.TryGetValue(key, out PlaneCellAggregate aggregate))
                    continue;

                component.minU = Mathf.Min(component.minU, aggregate.minU);
                component.maxU = Mathf.Max(component.maxU, aggregate.maxU);
                component.minV = Mathf.Min(component.minV, aggregate.minV);
                component.maxV = Mathf.Max(component.maxV, aggregate.maxV);
                component.pointCount += aggregate.pointCount;
                component.occupiedCount++;

                EnqueueNeighbor(new PlaneGridKey { u = key.u + 1, v = key.v }, m_occupiedCells);
                EnqueueNeighbor(new PlaneGridKey { u = key.u - 1, v = key.v }, m_occupiedCells);
                EnqueueNeighbor(new PlaneGridKey { u = key.u, v = key.v + 1 }, m_occupiedCells);
                EnqueueNeighbor(new PlaneGridKey { u = key.u, v = key.v - 1 }, m_occupiedCells);
            }

            return component;
        }

        private void EnqueueNeighbor(PlaneGridKey key, HashSet<PlaneGridKey> occupiedCells)
        {
            if (!occupiedCells.Contains(key) || m_visitedCells.Contains(key))
                return;

            m_visitedCells.Add(key);
            m_componentQueue.Enqueue(key);
        }

        private void TryCreateCandidateFromComponent(PlaneBucketData bucket, PlaneComponentAccumulator component)
        {
            float width = component.maxU - component.minU;
            float height = component.maxV - component.minV;
            if (width < Mathf.Max(0.1f, minPlaneWidthMeters) || height < Mathf.Max(0.1f, minPlaneHeightMeters))
                return;

            if (component.occupiedCount < Mathf.Max(1, minOccupiedCells))
                return;

            if (component.pointCount < Mathf.Max(4, minPlanePointCount))
                return;

            float centerU = (component.minU + component.maxU) * 0.5f;
            float centerV = (component.minV + component.maxV) * 0.5f;
            Vector3 center = bucket.normal * bucket.planeDistance + bucket.tangent * centerU + bucket.bitangent * centerV;
            float score = width * height + component.occupiedCount * 0.3f + component.pointCount * 0.05f;

            m_planeCandidates.Add(new PlaneCandidate
            {
                center = center,
                normal = bucket.normal,
                tangent = bucket.tangent,
                bitangent = bucket.bitangent,
                width = width,
                height = height,
                pointCount = component.pointCount,
                occupiedCells = component.occupiedCount,
                score = score,
            });
        }

        private void BuildDrawBatches()
        {
            m_cachedDrawBatches.Clear();
            m_cachedDrawCounts.Clear();
            if (m_planeCandidates.Count <= 0)
                return;

            Matrix4x4[] batch = new Matrix4x4[MaxInstancesPerDraw];
            int batchCount = 0;

            for (int i = 0; i < m_planeCandidates.Count; i++)
            {
                PlaneCandidate plane = m_planeCandidates[i];
                Quaternion rotation = Quaternion.LookRotation(plane.normal, plane.bitangent);
                Vector3 position = plane.center + plane.normal * Mathf.Max(0f, surfaceOffsetMeters);
                Vector3 scale = new Vector3(Mathf.Max(0.05f, plane.width), Mathf.Max(0.05f, plane.height), 1f);
                batch[batchCount++] = Matrix4x4.TRS(position, rotation, scale);

                if (batchCount >= MaxInstancesPerDraw)
                {
                    CommitBatch(batch, batchCount);
                    batch = new Matrix4x4[MaxInstancesPerDraw];
                    batchCount = 0;
                }
            }

            if (batchCount > 0)
                CommitBatch(batch, batchCount);
        }

        private void CommitBatch(Matrix4x4[] batch, int count)
        {
            m_cachedDrawBatches.Add(batch);
            m_cachedDrawCounts.Add(count);
        }

        private void MaybeLogDiagnostics(string reason)
        {
            if (!debugLog)
                return;

            float now = Time.unscaledTime;
            if (now < m_nextDebugLogTime)
                return;

            m_nextDebugLogTime = now + 0.5f;
            Debug.Log(
                $"[DepthEffectsStablePlaneRenderer] {reason} | stable={LastStablePointCount}, buckets={LastPlaneBucketCount}, candidates={LastCandidatePlaneCount}, displayed={LastPlaneCount}, batches={LastDrawBatchCount}");
        }

        private Mesh ResolveFrameMesh()
        {
            if (m_frameMesh != null)
                return m_frameMesh;

            float half = 0.5f;
            float thickness = Mathf.Clamp01(frameThicknessRatio);
            float inner = half - half * Mathf.Clamp(thickness, 0.02f, 0.45f);

            Vector3[] vertices =
            {
                new Vector3(-half, -half, 0f), new Vector3(half, -half, 0f), new Vector3(half, -inner, 0f), new Vector3(-half, -inner, 0f),
                new Vector3(-half, inner, 0f), new Vector3(half, inner, 0f), new Vector3(half, half, 0f), new Vector3(-half, half, 0f),
                new Vector3(-half, -inner, 0f), new Vector3(-inner, -inner, 0f), new Vector3(-inner, inner, 0f), new Vector3(-half, inner, 0f),
                new Vector3(inner, -inner, 0f), new Vector3(half, -inner, 0f), new Vector3(half, inner, 0f), new Vector3(inner, inner, 0f),
            };

            int[] triangles =
            {
                0, 1, 2, 0, 2, 3,
                4, 5, 6, 4, 6, 7,
                8, 9, 10, 8, 10, 11,
                12, 13, 14, 12, 14, 15,
            };

            Vector2[] uvs = new Vector2[vertices.Length];
            for (int i = 0; i < uvs.Length; i++)
                uvs[i] = new Vector2(vertices[i].x + 0.5f, vertices[i].y + 0.5f);

            m_frameMesh = new Mesh
            {
                name = "DepthEffects_StablePlaneFrame"
            };
            m_frameMesh.SetVertices(vertices);
            m_frameMesh.SetTriangles(triangles, 0);
            m_frameMesh.SetUVs(0, uvs);
            m_frameMesh.RecalculateNormals();
            m_frameMesh.RecalculateBounds();
            return m_frameMesh;
        }

        private Material ResolveMaterial()
        {
            Material material = planeMaterialOverride != null ? planeMaterialOverride : _GetOrCreateRuntimeMaterial();
            if (material == null)
            {
                if (!m_loggedMaterialFailure)
                {
                    Debug.LogWarning("[DepthEffectsStablePlaneRenderer] Unable to resolve plane material.");
                    m_loggedMaterialFailure = true;
                }
                return null;
            }

            material.enableInstancing = true;
            if (material.HasProperty("_BaseColor"))
                material.SetColor("_BaseColor", planeColor);
            if (material.HasProperty("_Color"))
                material.SetColor("_Color", planeColor);
            if (material.HasProperty("_Surface"))
                material.SetFloat("_Surface", 0f);
            if (material.HasProperty("_ZWrite"))
                material.SetFloat("_ZWrite", 1f);
            if (material.HasProperty("_Cull"))
                material.SetFloat("_Cull", (float)CullMode.Off);

            return material;
        }

        private Material _GetOrCreateRuntimeMaterial()
        {
            if (m_runtimeMaterial != null)
                return m_runtimeMaterial;

            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                return null;

            m_runtimeMaterial = new Material(shader)
            {
                name = "DepthEffectsStablePlaneRenderer_Runtime"
            };
            return m_runtimeMaterial;
        }

        private void EnsurePropertyBlock()
        {
            if (m_propertyBlock == null)
                m_propertyBlock = new MaterialPropertyBlock();
        }

        private static int ResolveAxisIndex(Vector3 normal)
        {
            Vector3 n = normal.sqrMagnitude > 1e-5f ? normal.normalized : Vector3.up;
            float ax = Mathf.Abs(n.x);
            float ay = Mathf.Abs(n.y);
            float az = Mathf.Abs(n.z);

            if (ax >= ay && ax >= az)
                return n.x >= 0f ? 0 : 1;
            if (ay >= ax && ay >= az)
                return n.y >= 0f ? 2 : 3;
            return n.z >= 0f ? 4 : 5;
        }

        private static Vector3 ResolveAxisNormal(int axisIndex)
        {
            switch (axisIndex)
            {
                case 0: return Vector3.right;
                case 1: return Vector3.left;
                case 2: return Vector3.up;
                case 3: return Vector3.down;
                case 4: return Vector3.forward;
                default: return Vector3.back;
            }
        }

        private static void ResolvePlaneBasis(int axisIndex, out Vector3 tangent, out Vector3 bitangent)
        {
            Vector3 normal = ResolveAxisNormal(axisIndex);
            Vector3 referenceUp = Mathf.Abs(Vector3.Dot(normal, Vector3.up)) > 0.85f ? Vector3.forward : Vector3.up;
            tangent = Vector3.ProjectOnPlane(referenceUp, normal).normalized;
            if (tangent.sqrMagnitude <= 1e-5f)
                tangent = Vector3.right;
            bitangent = Vector3.Cross(normal, tangent).normalized;
            tangent = Vector3.Cross(bitangent, normal).normalized;
        }
    }
}

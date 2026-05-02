// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsSurfacePatchGridRenderer : MonoBehaviour
    {
        private const int MaxInstancesPerDraw = 1023;

        private struct DisplayCellBucketKey
        {
            public int x;
            public int y;
            public int z;
            public int normalAxis;

            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = x;
                    hash = (hash * 397) ^ y;
                    hash = (hash * 397) ^ z;
                    hash = (hash * 397) ^ normalAxis;
                    return hash;
                }
            }
        }

        [Header("Refs")]
        [SerializeField]
        private DepthEffectsGridGuidedCoverageSampler coverageSampler;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private Material surfaceCellMaterialOverride;

        [Header("Display")]
        [SerializeField]
        private bool showSurfacePatchGrid = true;

        [SerializeField]
        private float cellFillRatio = 0.22f;

        [SerializeField]
        private float frameThicknessRatio = 0.12f;

        [SerializeField]
        private float surfaceOffsetMeters = 0.01f;

        [SerializeField]
        private float minDisplayDistanceMeters = 0.38f;

        [SerializeField]
        private float maxDisplayDistanceMeters = 4.2f;

        [SerializeField]
        private int maxDisplayCells = 2048;

        [SerializeField]
        private float displayRefreshIntervalSeconds = 0.06f;

        [SerializeField]
        private Color surfaceCellColor = new Color(0.92f, 0.96f, 0.99f, 0.92f);

        [Header("Debug")]
        [SerializeField]
        private bool debugLog;

        public int LastSourceCellCount { get; private set; }
        public int LastDisplayCellCount { get; private set; }
        public int LastDrawBatchCount { get; private set; }

        private readonly List<DepthEffectsGridGuidedCoverageSampler.DisplayCell> m_sourceCells =
            new List<DepthEffectsGridGuidedCoverageSampler.DisplayCell>(4096);
        private readonly List<DepthEffectsGridGuidedCoverageSampler.DisplayCell> m_displayCells =
            new List<DepthEffectsGridGuidedCoverageSampler.DisplayCell>(4096);
        private readonly Dictionary<DisplayCellBucketKey, DepthEffectsGridGuidedCoverageSampler.DisplayCell> m_displayBuckets =
            new Dictionary<DisplayCellBucketKey, DepthEffectsGridGuidedCoverageSampler.DisplayCell>(4096);
        private readonly List<Matrix4x4[]> m_cachedDrawBatches = new List<Matrix4x4[]>(8);
        private readonly List<int> m_cachedDrawCounts = new List<int>(8);

        private Mesh m_frameMesh;
        private Material m_runtimeMaterial;
        private MaterialPropertyBlock m_propertyBlock;
        private bool m_loggedMaterialFailure;
        private float m_lastSamplerTickTime = -1f;
        private float m_nextAllowedRefreshTime;

        private void Awake()
        {
            ResolveRefs();
        }

        private void LateUpdate()
        {
            ResolveRefs();
            if (!showSurfacePatchGrid || coverageSampler == null)
            {
                LastSourceCellCount = 0;
                LastDisplayCellCount = 0;
                LastDrawBatchCount = 0;
                return;
            }

            RebuildDisplayCacheIfNeeded();
            if (m_cachedDrawBatches.Count <= 0)
            {
                LastDrawBatchCount = 0;
                return;
            }

            Mesh mesh = ResolveFrameMesh();
            Material material = ResolveMaterial();
            if (mesh == null || material == null)
                return;

            EnsurePropertyBlock();
            m_propertyBlock.SetColor("_BaseColor", surfaceCellColor);
            m_propertyBlock.SetColor("_Color", surfaceCellColor);

            LastDrawBatchCount = 0;
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
                LastDrawBatchCount++;
            }
        }

        public void Configure(DepthEffectsGridGuidedCoverageSampler sampler, Camera camera, Material materialOverride)
        {
            coverageSampler = sampler;
            sampleCamera = camera;
            surfaceCellMaterialOverride = materialOverride;
        }

        private void ResolveRefs()
        {
            if (coverageSampler == null)
                coverageSampler = GetComponent<DepthEffectsGridGuidedCoverageSampler>();
            if (sampleCamera == null)
                sampleCamera = Camera.main;
        }

        private void RebuildDisplayCacheIfNeeded()
        {
            float samplerTickTime = coverageSampler != null ? coverageSampler.LastTickTime : -1f;
            float now = Time.unscaledTime;
            if (Mathf.Approximately(samplerTickTime, m_lastSamplerTickTime) &&
                now < m_nextAllowedRefreshTime)
                return;

            coverageSampler.GetDisplayCellsNonAlloc(m_sourceCells);
            LastSourceCellCount = m_sourceCells.Count;
            BuildDisplayCells();
            BuildDrawBatches();
            LastDisplayCellCount = m_displayCells.Count;
            m_lastSamplerTickTime = samplerTickTime;
            m_nextAllowedRefreshTime = now + Mathf.Max(0.01f, displayRefreshIntervalSeconds);
        }

        private void BuildDisplayCells()
        {
            m_displayCells.Clear();
            if (m_sourceCells.Count <= 0)
                return;

            m_displayBuckets.Clear();
            Transform cameraTransform = sampleCamera != null ? sampleCamera.transform : null;
            float minDistanceSq = Mathf.Max(0f, minDisplayDistanceMeters);
            minDistanceSq *= minDistanceSq;
            float maxDistance = Mathf.Max(minDisplayDistanceMeters, maxDisplayDistanceMeters);
            float maxDistanceSq = maxDistance > 0f ? maxDistance * maxDistance : float.PositiveInfinity;
            float spacing = coverageSampler != null ? coverageSampler.TargetSurfaceSpacingMeters : 0.22f;
            float bucketSpacing = Mathf.Max(0.04f, spacing * 0.75f);

            for (int i = 0; i < m_sourceCells.Count; i++)
            {
                DepthEffectsGridGuidedCoverageSampler.DisplayCell cell = m_sourceCells[i];
                if (cameraTransform != null)
                {
                    float distanceSq = (cell.worldPos - cameraTransform.position).sqrMagnitude;
                    if (distanceSq < minDistanceSq || distanceSq > maxDistanceSq)
                        continue;
                }

                DisplayCellBucketKey key = new DisplayCellBucketKey
                {
                    x = Mathf.RoundToInt(cell.worldPos.x / bucketSpacing),
                    y = Mathf.RoundToInt(cell.worldPos.y / bucketSpacing),
                    z = Mathf.RoundToInt(cell.worldPos.z / bucketSpacing),
                    normalAxis = ResolveDominantNormalAxis(cell.normal),
                };

                if (m_displayBuckets.TryGetValue(key, out DepthEffectsGridGuidedCoverageSampler.DisplayCell existing))
                {
                    if (ResolveCellScore(cell) <= ResolveCellScore(existing))
                        continue;
                }

                m_displayBuckets[key] = cell;
            }

            foreach (KeyValuePair<DisplayCellBucketKey, DepthEffectsGridGuidedCoverageSampler.DisplayCell> pair in m_displayBuckets)
                m_displayCells.Add(pair.Value);

            m_displayCells.Sort((a, b) =>
            {
                return ResolveCellScore(b).CompareTo(ResolveCellScore(a));
            });

            int maxCount = Mathf.Max(64, maxDisplayCells);
            if (m_displayCells.Count > maxCount)
                m_displayCells.RemoveRange(maxCount, m_displayCells.Count - maxCount);
        }

        private void BuildDrawBatches()
        {
            m_cachedDrawCounts.Clear();
            for (int i = 0; i < m_cachedDrawBatches.Count; i++)
                System.Array.Clear(m_cachedDrawBatches[i], 0, m_cachedDrawBatches[i].Length);

            if (m_displayCells.Count <= 0)
                return;

            float spacing = coverageSampler != null ? coverageSampler.TargetSurfaceSpacingMeters : 0.22f;
            float cellScale = Mathf.Max(0.02f, spacing * Mathf.Clamp(cellFillRatio, 0.05f, 1f));

            int cellIndex = 0;
            while (cellIndex < m_displayCells.Count)
            {
                int batchIndex = m_cachedDrawCounts.Count;
                EnsureBatchCapacity(batchIndex);
                Matrix4x4[] batch = m_cachedDrawBatches[batchIndex];
                int drawCount = Mathf.Min(MaxInstancesPerDraw, m_displayCells.Count - cellIndex);
                for (int i = 0; i < drawCount; i++)
                {
                    DepthEffectsGridGuidedCoverageSampler.DisplayCell cell = m_displayCells[cellIndex + i];
                    Vector3 worldPos = cell.worldPos + cell.normal * Mathf.Max(0f, surfaceOffsetMeters);
                    Quaternion rotation = Quaternion.LookRotation(cell.normal, cell.bitangent);
                    batch[i] = Matrix4x4.TRS(worldPos, rotation, new Vector3(cellScale, cellScale, 1f));
                }

                m_cachedDrawCounts.Add(drawCount);
                cellIndex += drawCount;
            }
        }

        private void EnsureBatchCapacity(int batchIndex)
        {
            while (m_cachedDrawBatches.Count <= batchIndex)
                m_cachedDrawBatches.Add(new Matrix4x4[MaxInstancesPerDraw]);
        }

        private void EnsurePropertyBlock()
        {
            if (m_propertyBlock == null)
                m_propertyBlock = new MaterialPropertyBlock();
        }

        private Mesh ResolveFrameMesh()
        {
            if (m_frameMesh != null)
                return m_frameMesh;

            float t = Mathf.Clamp(frameThicknessRatio, 0.02f, 0.45f) * 0.5f;
            float leftOuter = -0.5f;
            float rightOuter = 0.5f;
            float bottomOuter = -0.5f;
            float topOuter = 0.5f;
            float leftInner = leftOuter + t;
            float rightInner = rightOuter - t;
            float bottomInner = bottomOuter + t;
            float topInner = topOuter - t;

            Vector3[] vertices = new[]
            {
                new Vector3(leftOuter, bottomOuter, 0f), new Vector3(rightOuter, bottomOuter, 0f), new Vector3(rightOuter, bottomInner, 0f), new Vector3(leftOuter, bottomInner, 0f),
                new Vector3(leftOuter, topInner, 0f), new Vector3(rightOuter, topInner, 0f), new Vector3(rightOuter, topOuter, 0f), new Vector3(leftOuter, topOuter, 0f),
                new Vector3(leftOuter, bottomInner, 0f), new Vector3(leftInner, bottomInner, 0f), new Vector3(leftInner, topInner, 0f), new Vector3(leftOuter, topInner, 0f),
                new Vector3(rightInner, bottomInner, 0f), new Vector3(rightOuter, bottomInner, 0f), new Vector3(rightOuter, topInner, 0f), new Vector3(rightInner, topInner, 0f),
            };

            int[] triangles = new[]
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
                name = "DepthEffects_SurfacePatchFrame"
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
            if (surfaceCellMaterialOverride != null)
            {
                surfaceCellMaterialOverride.enableInstancing = true;
                if (surfaceCellMaterialOverride.HasProperty("_BaseColor"))
                    surfaceCellMaterialOverride.SetColor("_BaseColor", surfaceCellColor);
                if (surfaceCellMaterialOverride.HasProperty("_Color"))
                    surfaceCellMaterialOverride.SetColor("_Color", surfaceCellColor);
                if (surfaceCellMaterialOverride.HasProperty("_Surface"))
                    surfaceCellMaterialOverride.SetFloat("_Surface", 0f);
                if (surfaceCellMaterialOverride.HasProperty("_ZWrite"))
                    surfaceCellMaterialOverride.SetFloat("_ZWrite", 1f);
                if (surfaceCellMaterialOverride.HasProperty("_Cull"))
                    surfaceCellMaterialOverride.SetFloat("_Cull", (float)CullMode.Off);
                return surfaceCellMaterialOverride;
            }

            if (m_runtimeMaterial != null)
                return m_runtimeMaterial;

            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                shader = Shader.Find("Standard");

            if (shader == null)
            {
                if (!m_loggedMaterialFailure)
                {
                    Debug.LogWarning("[DepthEffectsSurfacePatchGridRenderer] Could not resolve a patch grid shader.");
                    m_loggedMaterialFailure = true;
                }
                return null;
            }

            m_runtimeMaterial = new Material(shader)
            {
                name = "DepthEffects_SurfacePatchGrid_Runtime"
            };
            m_runtimeMaterial.enableInstancing = true;
            m_runtimeMaterial.SetColor("_BaseColor", surfaceCellColor);
            m_runtimeMaterial.SetColor("_Color", surfaceCellColor);
            if (m_runtimeMaterial.HasProperty("_Surface"))
                m_runtimeMaterial.SetFloat("_Surface", 0f);
            if (m_runtimeMaterial.HasProperty("_ZWrite"))
                m_runtimeMaterial.SetFloat("_ZWrite", 1f);
            if (m_runtimeMaterial.HasProperty("_Cull"))
                m_runtimeMaterial.SetFloat("_Cull", (float)CullMode.Off);
            return m_runtimeMaterial;
        }

        private void OnDestroy()
        {
            if (m_runtimeMaterial != null)
                Destroy(m_runtimeMaterial);
            if (m_frameMesh != null)
                Destroy(m_frameMesh);
        }

        private static float ResolveCellScore(DepthEffectsGridGuidedCoverageSampler.DisplayCell cell)
        {
            return cell.patchHits + cell.lastSeenTime * 0.01f;
        }

        private static int ResolveDominantNormalAxis(Vector3 normal)
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
    }
}

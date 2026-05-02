// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsPointCloudDebugRenderer : MonoBehaviour
    {
        [SerializeField]
        private ScanCoverVoxelPointAccumulator voxelPointAccumulator;

        [SerializeField]
        private Material pointMaterialOverride;

        [SerializeField]
        private bool showDebugPoints = true;

        [SerializeField]
        private int maxVisiblePoints = 512;

        [SerializeField]
        private float pointScaleMeters = 0.012f;

        [SerializeField]
        private float maxDisplayDistanceMeters = 4.5f;

        [SerializeField]
        private bool sortByDistanceToCamera = true;

        [SerializeField]
        private float surfaceOffsetMeters = 0.01f;

        [SerializeField]
        private Color pointColor = new Color(1f, 1f, 1f, 1f);

        [SerializeField]
        private bool debugLog;

        private readonly List<ScanCoverVoxelPointAccumulator.StablePoint> m_stablePoints =
            new List<ScanCoverVoxelPointAccumulator.StablePoint>(256);
        private readonly List<ScanCoverVoxelPointAccumulator.StablePoint> m_filteredPoints =
            new List<ScanCoverVoxelPointAccumulator.StablePoint>(256);

        private readonly List<GameObject> m_pointObjects = new List<GameObject>(128);
        private Transform m_root;

        private void Awake()
        {
            EnsureRefs();
            EnsureRoot();
        }

        private void LateUpdate()
        {
            EnsureRefs();
            EnsureRoot();

            if (!showDebugPoints || voxelPointAccumulator == null)
            {
                SetVisibleCount(0);
                return;
            }

            voxelPointAccumulator.GetStablePointsNonAlloc(m_stablePoints);
            BuildFilteredPoints();
            int visibleCount = Mathf.Min(Mathf.Max(0, maxVisiblePoints), m_filteredPoints.Count);
            EnsurePointObjects(visibleCount);

            for (int i = 0; i < visibleCount; i++)
            {
                ScanCoverVoxelPointAccumulator.StablePoint point = m_filteredPoints[i];
                GameObject pointObject = m_pointObjects[i];
                Vector3 worldPos = point.worldPos + point.normal * Mathf.Max(0f, surfaceOffsetMeters);
                pointObject.transform.SetPositionAndRotation(worldPos, Quaternion.identity);
                pointObject.transform.localScale = Vector3.one * Mathf.Max(0.001f, pointScaleMeters);
                if (!pointObject.activeSelf)
                    pointObject.SetActive(true);
            }

            SetVisibleCount(visibleCount);

            if (debugLog && Random.value < 0.01f)
                Debug.Log($"[DepthEffectsPointCloudDebugRenderer] visible={visibleCount}, filtered={m_filteredPoints.Count}, stable={m_stablePoints.Count}");
        }

        private void OnDestroy()
        {
            for (int i = 0; i < m_pointObjects.Count; i++)
            {
                if (m_pointObjects[i] != null)
                    Destroy(m_pointObjects[i]);
            }

            if (m_root != null)
                Destroy(m_root.gameObject);
        }

        public void Configure(ScanCoverVoxelPointAccumulator accumulator, Material material)
        {
            voxelPointAccumulator = accumulator;
            pointMaterialOverride = material;
        }

        public void SetShowDebugPoints(bool show)
        {
            showDebugPoints = show;
            if (!showDebugPoints)
                SetVisibleCount(0);
        }

        private void BuildFilteredPoints()
        {
            m_filteredPoints.Clear();
            if (m_stablePoints.Count <= 0)
                return;

            Camera camera = Camera.main;
            Vector3 cameraPos = camera != null ? camera.transform.position : Vector3.zero;
            float maxDistanceSq = Mathf.Max(0.1f, maxDisplayDistanceMeters) * Mathf.Max(0.1f, maxDisplayDistanceMeters);

            for (int i = 0; i < m_stablePoints.Count; i++)
            {
                ScanCoverVoxelPointAccumulator.StablePoint point = m_stablePoints[i];
                if (camera != null && (point.worldPos - cameraPos).sqrMagnitude > maxDistanceSq)
                    continue;

                m_filteredPoints.Add(point);
            }

            if (!sortByDistanceToCamera || camera == null)
                return;

            m_filteredPoints.Sort((a, b) =>
            {
                float da = (a.worldPos - cameraPos).sqrMagnitude;
                float db = (b.worldPos - cameraPos).sqrMagnitude;
                return da.CompareTo(db);
            });
        }

        private void EnsureRefs()
        {
            if (voxelPointAccumulator == null)
                voxelPointAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();

            if (pointMaterialOverride == null)
                pointMaterialOverride = Resources.Load<Material>("DepthEffectsPointCloudUnlit");
        }

        private void EnsureRoot()
        {
            if (m_root != null)
                return;

            GameObject root = new GameObject("DepthEffectsPointCloudDebugRoot");
            root.transform.SetParent(transform, false);
            m_root = root.transform;
        }

        private void EnsurePointObjects(int count)
        {
            while (m_pointObjects.Count < count)
            {
                GameObject primitive = GameObject.CreatePrimitive(PrimitiveType.Cube);
                primitive.name = $"DebugPoint_{m_pointObjects.Count:D3}";
                primitive.transform.SetParent(m_root, false);
                Collider collider = primitive.GetComponent<Collider>();
                if (collider != null)
                    Destroy(collider);

                MeshRenderer renderer = primitive.GetComponent<MeshRenderer>();
                if (renderer != null && pointMaterialOverride != null)
                {
                    renderer.sharedMaterial = pointMaterialOverride;
                    if (renderer.sharedMaterial.HasProperty("_BaseColor"))
                        renderer.sharedMaterial.SetColor("_BaseColor", pointColor);
                    if (renderer.sharedMaterial.HasProperty("_Color"))
                        renderer.sharedMaterial.SetColor("_Color", pointColor);
                }

                primitive.SetActive(false);
                m_pointObjects.Add(primitive);
            }
        }

        private void SetVisibleCount(int visibleCount)
        {
            for (int i = visibleCount; i < m_pointObjects.Count; i++)
            {
                if (m_pointObjects[i] != null && m_pointObjects[i].activeSelf)
                    m_pointObjects[i].SetActive(false);
            }
        }
    }
}

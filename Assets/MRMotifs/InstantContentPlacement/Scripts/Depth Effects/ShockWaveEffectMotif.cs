// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using Meta.XR;
using Meta.XR.Samples;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    /// <summary>
    /// Expands the scan wave effect over a set duration, then destroys the object.
    /// </summary>
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class ShockWaveEffectMotif : MonoBehaviour
    {
        [Header("Scan Wave Settings")]
        [Tooltip("The maximum scale the scan wave will reach.")]
        [SerializeField]
        private float endScale = 25.0f;

        [Tooltip("The duration over which the scan wave expands.")]
        [SerializeField]
        private float duration = 3.5f;

        [Tooltip("How long the scan wave keeps expanding before freezing at its current size.")]
        [SerializeField]
        private float lifetime = 0.7f;

        [Tooltip("Controls the growth rate of the scan wave expansion.")]
        [SerializeField]
        private float growthRate = 2.0f;

        [Header("Point Cloud Mapping")]
        [Tooltip("Whether the expanding shock wave writes sparse observations into a persistent point cloud map.")]
        [SerializeField]
        private bool drivePointCloudMapping = false;

        [Tooltip("Maximum world-space scan radius used for point cloud mapping.")]
        [SerializeField]
        private float mappingMaxRadiusMeters = 2.8f;

        [Tooltip("Thickness of the scan band that writes new observations.")]
        [SerializeField]
        private float mappingBandThicknessMeters = 0.18f;

        [Tooltip("Sparse viewport sampling columns for each scan write step.")]
        [SerializeField]
        private int mappingSampleColumns = 20;

        [Tooltip("Sparse viewport sampling rows for each scan write step.")]
        [SerializeField]
        private int mappingSampleRows = 12;

        [Tooltip("How often the shock wave writes new observations into the point cloud map.")]
        [SerializeField]
        private float mappingSampleIntervalSeconds = 0.08f;

        [Tooltip("Viewport margin left unused to avoid low-confidence edge samples.")]
        [SerializeField]
        private float mappingViewportMargin = 0.08f;

        [Tooltip("Maximum environment raycast distance used while sampling the scan band.")]
        [SerializeField]
        private float mappingRayMaxDistanceMeters = 5.0f;

        [Tooltip("Minimum normal confidence required to write a point into the map.")]
        [SerializeField]
        private float mappingMinNormalConfidence = 0.2f;

        private EnvironmentRaycastManager m_environmentRaycastManager;
        private Camera m_sampleCamera;
        private DepthEffectsPointCloudMapMotif m_pointCloudMap;
        private float m_nextMappingSampleTime;

        private const float START_SCALE = 0.0f;
        private float m_currentTimer;

        private void Awake()
        {
            transform.localScale = Vector3.one * START_SCALE;
            StartCoroutine(ExpandAndDestroy());
        }

        public void Initialize(EnvironmentRaycastManager environmentRaycastManager, Camera sampleCamera, DepthEffectsPointCloudMapMotif pointCloudMap)
        {
            m_environmentRaycastManager = environmentRaycastManager;
            m_sampleCamera = sampleCamera;
            m_pointCloudMap = pointCloudMap;
            m_nextMappingSampleTime = Time.unscaledTime;
        }

        private IEnumerator ExpandAndDestroy()
        {
            while (m_currentTimer <= lifetime)
            {
                m_currentTimer += Time.deltaTime;
                var normalizedTime = duration > 0f ? Mathf.Clamp01(m_currentTimer / duration) : 1f;
                var scale = Mathf.Lerp(START_SCALE, endScale, Mathf.Pow(normalizedTime, growthRate));
                transform.localScale = Vector3.one * scale;

                if (drivePointCloudMapping && Time.unscaledTime >= m_nextMappingSampleTime)
                {
                    m_nextMappingSampleTime = Time.unscaledTime + Mathf.Max(0.02f, mappingSampleIntervalSeconds);
                    SampleMappingBand(normalizedTime);
                }

                yield return null;
            }
        }

        private void SampleMappingBand(float normalizedTime)
        {
            if (m_environmentRaycastManager == null || m_pointCloudMap == null)
                return;

            Camera sampleCamera = m_sampleCamera != null ? m_sampleCamera : Camera.main;
            if (sampleCamera == null)
                return;

            float currentRadius = Mathf.Lerp(0f, Mathf.Max(0.1f, mappingMaxRadiusMeters), Mathf.Pow(normalizedTime, growthRate));
            float halfBand = Mathf.Max(0.01f, mappingBandThicknessMeters) * 0.5f;
            float margin = Mathf.Clamp(mappingViewportMargin, 0f, 0.45f);
            float viewportExtent = Mathf.Max(0.1f, 1f - margin * 2f);
            int columns = Mathf.Max(4, mappingSampleColumns);
            int rows = Mathf.Max(4, mappingSampleRows);

            for (int row = 0; row < rows; row++)
            {
                float v = margin + viewportExtent * ((row + 0.5f) / rows);
                for (int column = 0; column < columns; column++)
                {
                    float u = margin + viewportExtent * ((column + 0.5f) / columns);
                    Ray ray = sampleCamera.ViewportPointToRay(new Vector3(u, v, 0f));
                    bool didHit = m_environmentRaycastManager.Raycast(ray, out var hit, Mathf.Max(0.25f, mappingRayMaxDistanceMeters));
                    if (!didHit &&
                        hit.status != EnvironmentRaycastHitStatus.HitPointOccluded)
                        continue;

                    if (hit.status != EnvironmentRaycastHitStatus.Hit &&
                        hit.status != EnvironmentRaycastHitStatus.HitPointOccluded)
                        continue;

                    float confidence = hit.status == EnvironmentRaycastHitStatus.Hit
                        ? hit.normalConfidence
                        : Mathf.Min(hit.normalConfidence, 0.5f);
                    if (confidence < Mathf.Clamp01(mappingMinNormalConfidence))
                        continue;

                    float distanceToCenter = Vector3.Distance(hit.point, transform.position);
                    if (currentRadius > halfBand && Mathf.Abs(distanceToCenter - currentRadius) > halfBand)
                        continue;

                    Vector3 normal = hit.normal.sqrMagnitude > 1e-5f ? hit.normal.normalized : -ray.direction;
                    m_pointCloudMap.AddObservation(hit.point, normal, confidence);
                }
            }
        }
    }
}

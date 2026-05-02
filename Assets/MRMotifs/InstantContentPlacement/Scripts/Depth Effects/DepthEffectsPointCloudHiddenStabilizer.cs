// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using System.Text;
using Meta.XR.Samples;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsPointCloudHiddenStabilizer : MonoBehaviour
    {
        [SerializeField]
        private bool enableStabilizer = true;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private ScanCoverVoxelPointAccumulator voxelPointAccumulator;

        [SerializeField]
        private ScanCoverInstancedPointCloudRenderer pointRenderer;

        [SerializeField]
        private DepthEffectsGridGuidedCoverageSampler coverageSampler;

        [SerializeField]
        private float refreshIntervalSeconds = 0.2f;

        private readonly List<ScanCoverVoxelPointAccumulator.StablePoint> m_stablePoints =
            new List<ScanCoverVoxelPointAccumulator.StablePoint>(4096);

        private readonly StringBuilder m_builder = new StringBuilder(512);

        private float m_nextRefreshTime;

        public void Configure(
            Camera camera,
            ScanCoverVoxelPointAccumulator accumulator,
            ScanCoverInstancedPointCloudRenderer renderer,
            DepthEffectsGridGuidedCoverageSampler sampler)
        {
            sampleCamera = camera;
            voxelPointAccumulator = accumulator;
            pointRenderer = renderer;
            coverageSampler = sampler;
        }

        private void LateUpdate()
        {
            EnsureRefs();
            if (!enableStabilizer || Time.unscaledTime < m_nextRefreshTime)
                return;

            m_nextRefreshTime = Time.unscaledTime + Mathf.Max(0.05f, refreshIntervalSeconds);
            RunStabilizationPass();
        }

        private void EnsureRefs()
        {
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (voxelPointAccumulator == null)
                voxelPointAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();
            if (pointRenderer == null)
                pointRenderer = GetComponent<ScanCoverInstancedPointCloudRenderer>();
            if (coverageSampler == null)
                coverageSampler = GetComponent<DepthEffectsGridGuidedCoverageSampler>();
        }

        private void RunStabilizationPass()
        {
            int voxelCount = voxelPointAccumulator != null ? voxelPointAccumulator.TotalVoxelCount : 0;
            int stableCount = 0;
            int inFrontCount = 0;
            int inFrustumCount = 0;
            int recentCount = 0;

            if (voxelPointAccumulator != null)
            {
                voxelPointAccumulator.GetStablePointsNonAlloc(m_stablePoints);
                stableCount = m_stablePoints.Count;
                ComputeViewMetrics(out inFrontCount, out inFrustumCount, out recentCount);
            }

            int rendererStableCount = pointRenderer != null ? pointRenderer.LastSourceStablePointCount : 0;
            int rendererDisplayCount = pointRenderer != null ? pointRenderer.LastDisplayPointCount : 0;
            int rendererBatchCount = pointRenderer != null ? pointRenderer.LastDrawBatchCount : 0;

            int sampleAttempts = coverageSampler != null ? coverageSampler.LastTickAttemptCount : 0;
            int sampleScheduled = coverageSampler != null ? coverageSampler.LastTickScheduledCount : 0;
            int sampleAccepted = coverageSampler != null ? coverageSampler.LastTickAcceptedHitCount : 0;
            int sampleOccluded = coverageSampler != null ? coverageSampler.LastTickAcceptedOccludedCount : 0;
            int sampleRejectedConfidence = coverageSampler != null ? coverageSampler.LastTickRejectedConfidenceCount : 0;

            m_builder.Clear();
            m_builder.Append("voxels=").Append(voxelCount)
                .Append(" stable=").Append(stableCount)
                .Append(" recent<1s=").Append(recentCount).AppendLine();
            m_builder.Append("inFront=").Append(inFrontCount)
                .Append(" inFrustum=").Append(inFrustumCount).AppendLine();
            m_builder.Append("renderer stable=").Append(rendererStableCount)
                .Append(" display=").Append(rendererDisplayCount)
                .Append(" batches=").Append(rendererBatchCount).AppendLine();
            m_builder.Append("sampler attempts=").Append(sampleAttempts)
                .Append(" scheduled=").Append(sampleScheduled)
                .Append(" accepted=").Append(sampleAccepted).AppendLine();
            m_builder.Append("sampler occluded=").Append(sampleOccluded)
                .Append(" rejectConf=").Append(sampleRejectedConfidence).AppendLine();
            m_builder.Append(BuildDiagnosis(stableCount, inFrustumCount, rendererDisplayCount, recentCount));
        }

        private void ComputeViewMetrics(out int inFrontCount, out int inFrustumCount, out int recentCount)
        {
            inFrontCount = 0;
            inFrustumCount = 0;
            recentCount = 0;

            if (sampleCamera == null)
                return;

            Transform cam = sampleCamera.transform;
            float now = Time.unscaledTime;

            for (int i = 0; i < m_stablePoints.Count; i++)
            {
                ScanCoverVoxelPointAccumulator.StablePoint point = m_stablePoints[i];
                Vector3 toPoint = point.worldPos - cam.position;
                if (Vector3.Dot(cam.forward, toPoint) > 0f)
                    inFrontCount++;

                Vector3 viewport = sampleCamera.WorldToViewportPoint(point.worldPos);
                if (viewport.z > 0f &&
                    viewport.x >= 0f && viewport.x <= 1f &&
                    viewport.y >= 0f && viewport.y <= 1f)
                {
                    inFrustumCount++;
                }

                if (now - point.lastSeenTime < 1.0f)
                    recentCount++;
            }
        }

        private static string BuildDiagnosis(int stableCount, int inFrustumCount, int rendererDisplayCount, int recentCount)
        {
            if (stableCount <= 0)
                return "diag: map empty";

            if (inFrustumCount <= 0)
                return "diag: current frustum has no stable points";

            if (rendererDisplayCount <= 0)
                return "diag: renderer got no display set";

            if (recentCount <= 0)
                return "diag: map exists but sampling paused / stale";

            return "diag: data alive; if cubes vanish, suspect display path jitter";
        }
    }
}

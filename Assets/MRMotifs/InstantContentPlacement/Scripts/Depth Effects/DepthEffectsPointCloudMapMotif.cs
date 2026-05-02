// Copyright (c) Meta Platforms, Inc. and affiliates.

using Meta.XR;
using Meta.XR.EnvironmentDepth;
using Meta.XR.Samples;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsPointCloudMapMotif : MonoBehaviour
    {
        public enum PerformanceSliceMode
        {
            Baseline = 0,
            SamplerHalf = 1,
            DisplayHalf = 2,
            BothHalf = 3,
            SamplerQuarter = 4,
            DisplayQuarter = 5,
            NoDisplay = 6,
            NoSamplingAndDisplay = 7,
        }

        [Header("Refs")]
        [SerializeField]
        private EnvironmentRaycastManager environmentRaycastManager;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private Material pointMaterialOverride;

        [SerializeField]
        private Material instancedPointMaterialOverride;

        [SerializeField]
        private DepthEffectsGridGuidedCoverageSampler coverageSampler;

        [SerializeField]
        private DepthEffectsSparseDepthTextureSampler depthTextureSampler;

        [SerializeField]
        private DepthEffectsPointCloudDebugRenderer debugPointRenderer;

        [SerializeField]
        private DepthEffectsPointCloudDiagnosticsHUD diagnosticsHud;

        [SerializeField]
        private DepthEffectsPointCloudHiddenStabilizer hiddenStabilizer;

        [SerializeField]
        private DepthEffectsPointCloudInvisibleForegroundStabilizer invisibleForegroundStabilizer;

        [SerializeField]
        private DepthEffectsSurfacePatchGridRenderer surfacePatchGridRenderer;

        [SerializeField]
        private DepthEffectsStablePlaneRenderer stablePlaneRenderer;

        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private DepthEffectsEnvironmentDepthGridRenderer environmentDepthGridRenderer;

        [SerializeField]
        private DepthEffectsStableIdGridRenderer stableIdGridRenderer;

        [SerializeField]
        private DepthEffectsViewportAttachedGridRenderer viewportAttachedGridRenderer;

        [Header("Voxel")]
        [SerializeField]
        private float cellSizeMeters = 0.03f;

        [SerializeField]
        private int minStableHits = 1;

        [SerializeField]
        private float holdSeconds = 12.0f;

        [SerializeField]
        private int maxStablePoints = 16000;

        [Header("Neighbor Merge")]
        [SerializeField]
        private bool enableNeighborMerge = true;

        [SerializeField]
        private float neighborMergeDistanceMeters = 0.06f;

        [SerializeField]
        private float neighborMergeNormalDot = 0.86f;

        [Header("Display")]
        [SerializeField]
        private bool showInstancedPointCloud = false;

        [SerializeField]
        private bool showStablePlanes = false;

        [SerializeField]
        private bool showGpuDepthGrid = false;

        [SerializeField]
        private bool showStableIdGrid = false;

        [SerializeField]
        private bool showViewportAttachedGrid = true;

        [SerializeField]
        private bool showDebugFallbackPoints = false;

        [SerializeField]
        private bool showDiagnosticsHud = false;

        [SerializeField]
        private bool enableHiddenStabilizer = false;

        [SerializeField]
        private bool enableInvisibleForegroundStabilizer = false;

        [SerializeField]
        private float pointScaleMeters = 0.0065f;

        [SerializeField]
        private float minPointScaleMeters = 0.0045f;

        [SerializeField]
        private float maxPointScaleMeters = 0.0085f;

        [SerializeField]
        private float surfaceOffsetMeters = 0.008f;

        [SerializeField]
        private Color pointColor = new Color(0.92f, 0.96f, 0.99f, 1.0f);

        [SerializeField]
        private Color planeColor = new Color(0.92f, 0.96f, 0.99f, 0.96f);

        [SerializeField]
        private bool debugLog;

        [Header("Performance Slice")]
        [SerializeField]
        private PerformanceSliceMode performanceSliceMode = PerformanceSliceMode.Baseline;

        private ScanCoverVoxelPointAccumulator m_voxelAccumulator;
        private ScanCoverInstancedPointCloudRenderer m_pointRenderer;

        private void Awake()
        {
            Debug.LogWarning("[DepthEffectsPointCloudMapMotif] Awake");
            EnsureComponents();
            ApplyConfiguration();
        }

        public void Configure(EnvironmentRaycastManager raycastManager, Camera camera)
        {
            environmentRaycastManager = raycastManager;
            sampleCamera = camera;
            EnsureComponents();
            ApplyConfiguration();
        }

        public void SetPerformanceSliceMode(PerformanceSliceMode mode)
        {
            performanceSliceMode = mode;
            EnsureComponents();
            ApplyConfiguration();
        }

        public void AddObservation(Vector3 point, Vector3 normal, float confidence)
        {
            EnsureComponents();
            m_voxelAccumulator.AddObservation(point, normal, confidence, Time.unscaledTime);
        }

        public void ClearMap()
        {
            EnsureComponents();
            m_voxelAccumulator.ClearAll();
        }

        private void EnsureComponents()
        {
            bool addedStablePlaneRenderer = false;
            if (m_voxelAccumulator == null)
                m_voxelAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();
            if (m_voxelAccumulator == null)
                m_voxelAccumulator = gameObject.AddComponent<ScanCoverVoxelPointAccumulator>();

            if (m_pointRenderer == null)
                m_pointRenderer = GetComponent<ScanCoverInstancedPointCloudRenderer>();
            if (m_pointRenderer == null)
                m_pointRenderer = gameObject.AddComponent<ScanCoverInstancedPointCloudRenderer>();

            if (coverageSampler == null)
                coverageSampler = GetComponent<DepthEffectsGridGuidedCoverageSampler>();
            if (coverageSampler == null)
                coverageSampler = gameObject.AddComponent<DepthEffectsGridGuidedCoverageSampler>();

            if (depthTextureSampler == null)
                depthTextureSampler = GetComponent<DepthEffectsSparseDepthTextureSampler>();
            if (depthTextureSampler == null)
                depthTextureSampler = gameObject.AddComponent<DepthEffectsSparseDepthTextureSampler>();

            if (debugPointRenderer == null)
                debugPointRenderer = GetComponent<DepthEffectsPointCloudDebugRenderer>();
            if (debugPointRenderer == null)
                debugPointRenderer = gameObject.AddComponent<DepthEffectsPointCloudDebugRenderer>();

            if (diagnosticsHud == null)
                diagnosticsHud = GetComponent<DepthEffectsPointCloudDiagnosticsHUD>();
            if (diagnosticsHud == null)
                diagnosticsHud = gameObject.AddComponent<DepthEffectsPointCloudDiagnosticsHUD>();

            if (hiddenStabilizer == null)
                hiddenStabilizer = GetComponent<DepthEffectsPointCloudHiddenStabilizer>();
            if (hiddenStabilizer == null)
                hiddenStabilizer = gameObject.AddComponent<DepthEffectsPointCloudHiddenStabilizer>();

            if (invisibleForegroundStabilizer == null)
                invisibleForegroundStabilizer = GetComponent<DepthEffectsPointCloudInvisibleForegroundStabilizer>();
            if (invisibleForegroundStabilizer == null)
                invisibleForegroundStabilizer = gameObject.AddComponent<DepthEffectsPointCloudInvisibleForegroundStabilizer>();

            if (surfacePatchGridRenderer == null)
                surfacePatchGridRenderer = GetComponent<DepthEffectsSurfacePatchGridRenderer>();
            if (surfacePatchGridRenderer == null)
                surfacePatchGridRenderer = gameObject.AddComponent<DepthEffectsSurfacePatchGridRenderer>();

            if (stablePlaneRenderer == null)
                stablePlaneRenderer = GetComponent<DepthEffectsStablePlaneRenderer>();
            if (stablePlaneRenderer == null)
            {
                stablePlaneRenderer = gameObject.AddComponent<DepthEffectsStablePlaneRenderer>();
                addedStablePlaneRenderer = true;
            }

            if (environmentDepthGridRenderer == null)
                environmentDepthGridRenderer = GetComponent<DepthEffectsEnvironmentDepthGridRenderer>();
            if (environmentDepthGridRenderer == null)
                environmentDepthGridRenderer = gameObject.AddComponent<DepthEffectsEnvironmentDepthGridRenderer>();

            if (stableIdGridRenderer == null)
                stableIdGridRenderer = GetComponent<DepthEffectsStableIdGridRenderer>();
            if (stableIdGridRenderer == null)
                stableIdGridRenderer = gameObject.AddComponent<DepthEffectsStableIdGridRenderer>();

            if (viewportAttachedGridRenderer == null)
                viewportAttachedGridRenderer = GetComponent<DepthEffectsViewportAttachedGridRenderer>();
            if (viewportAttachedGridRenderer == null)
                viewportAttachedGridRenderer = gameObject.AddComponent<DepthEffectsViewportAttachedGridRenderer>();

            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);

            if (sampleCamera == null)
                sampleCamera = Camera.main;

            if (pointMaterialOverride == null)
                pointMaterialOverride = Resources.Load<Material>("DepthEffectsPointCloudUnlit");
            if (instancedPointMaterialOverride == null)
                instancedPointMaterialOverride = Resources.Load<Material>("DepthEffectsPointCloudLitInstanced");

            if (addedStablePlaneRenderer)
                Debug.LogWarning("[DepthEffectsPointCloudMapMotif] Added DepthEffectsStablePlaneRenderer");
        }

        private void ApplyConfiguration()
        {
            bool effectiveShowInstancedPointCloud = showInstancedPointCloud;
            bool effectiveShowStablePlanes = showStablePlanes;
            bool effectiveShowGpuDepthGrid = showGpuDepthGrid;
            bool effectiveShowStableIdGrid = showStableIdGrid;
            bool effectiveShowViewportAttachedGrid = showViewportAttachedGrid;
            bool effectiveShowDebugFallbackPoints = showDebugFallbackPoints;
            bool effectiveShowDiagnosticsHud = showDiagnosticsHud;
            bool effectiveEnableHiddenStabilizer = enableHiddenStabilizer;
            bool effectiveEnableInvisibleForegroundStabilizer = enableInvisibleForegroundStabilizer;

            if (performanceSliceMode == PerformanceSliceMode.NoDisplay)
            {
                effectiveShowInstancedPointCloud = false;
                effectiveShowStablePlanes = false;
                effectiveShowGpuDepthGrid = false;
                effectiveShowStableIdGrid = false;
                effectiveShowViewportAttachedGrid = false;
                effectiveShowDebugFallbackPoints = false;
                effectiveShowDiagnosticsHud = false;
                effectiveEnableHiddenStabilizer = false;
                effectiveEnableInvisibleForegroundStabilizer = false;
            }
            else if (performanceSliceMode == PerformanceSliceMode.NoSamplingAndDisplay)
            {
                effectiveShowInstancedPointCloud = false;
                effectiveShowStablePlanes = false;
                effectiveShowGpuDepthGrid = false;
                effectiveShowStableIdGrid = false;
                effectiveShowViewportAttachedGrid = false;
                effectiveShowDebugFallbackPoints = false;
                effectiveShowDiagnosticsHud = false;
                effectiveEnableHiddenStabilizer = false;
                effectiveEnableInvisibleForegroundStabilizer = false;
            }

            bool useViewportAttachedGridMainline = effectiveShowViewportAttachedGrid;
            m_voxelAccumulator.cellSizeMeters = Mathf.Max(0.005f, cellSizeMeters);
            m_voxelAccumulator.minStableHits = Mathf.Max(1, minStableHits);
            m_voxelAccumulator.holdSeconds = Mathf.Max(0.05f, holdSeconds);
            m_voxelAccumulator.maxStablePoints = Mathf.Max(64, maxStablePoints);
            m_voxelAccumulator.enableNeighborMerge = enableNeighborMerge;
            m_voxelAccumulator.neighborMergeDistanceMeters = Mathf.Max(0.001f, neighborMergeDistanceMeters);
            m_voxelAccumulator.neighborMergeNormalDot = Mathf.Clamp(neighborMergeNormalDot, -1f, 1f);
            m_voxelAccumulator.debugLog = debugLog;
            m_voxelAccumulator.enabled =
                performanceSliceMode != PerformanceSliceMode.NoSamplingAndDisplay &&
                !effectiveShowGpuDepthGrid &&
                !effectiveShowStableIdGrid &&
                !useViewportAttachedGridMainline;

            m_pointRenderer.voxelPointAccumulator = m_voxelAccumulator;
            m_pointRenderer.sampleCamera = sampleCamera;
            m_pointRenderer.showPointCloud = effectiveShowInstancedPointCloud;
            m_pointRenderer.enabled = effectiveShowInstancedPointCloud;
            m_pointRenderer.pointPrimitive = PrimitiveType.Cube;
            m_pointRenderer.billboardPointQuads = false;
            m_pointRenderer.scalePointByConfidence = true;
            m_pointRenderer.regularizeDisplayPoints = true;
            m_pointRenderer.displayRegularizationSpacingMeters = Mathf.Max(cellSizeMeters * 2.25f, 0.04f);
            m_pointRenderer.clampDisplayPointCount = true;
            m_pointRenderer.maxDisplayPoints = 2048;
            m_pointRenderer.minDisplayDistanceMeters = 0.38f;
            m_pointRenderer.maxDisplayDistanceMeters = 4.2f;
            m_pointRenderer.pointOrientationMode = ScanCoverInstancedPointCloudRenderer.PointOrientationMode.AxisAligned;
            m_pointRenderer.pointScaleMeters = Mathf.Max(0.003f, pointScaleMeters);
            m_pointRenderer.minPointScaleMeters = Mathf.Max(0.003f, minPointScaleMeters);
            m_pointRenderer.maxPointScaleMeters = Mathf.Max(m_pointRenderer.minPointScaleMeters, maxPointScaleMeters);
            m_pointRenderer.surfaceOffsetMeters = Mathf.Max(0f, surfaceOffsetMeters);
            m_pointRenderer.distanceSurfaceOffsetFactor = 0.0035f;
            m_pointRenderer.maxSurfaceOffsetMeters = 0.035f;
            m_pointRenderer.displayRefreshIntervalSeconds = 0.06f;
            m_pointRenderer.pointColor = pointColor;
            m_pointRenderer.pointMaterialOverride = instancedPointMaterialOverride != null
                ? instancedPointMaterialOverride
                : pointMaterialOverride;
            m_pointRenderer.debugLog = debugLog;

            stablePlaneRenderer.Configure(
                m_voxelAccumulator,
                sampleCamera,
                instancedPointMaterialOverride != null ? instancedPointMaterialOverride : pointMaterialOverride);
            stablePlaneRenderer.enabled = effectiveShowStablePlanes;

            depthTextureSampler.Configure(environmentDepthManager, sampleCamera, m_voxelAccumulator);
            ApplyPerformanceSliceMode();
            depthTextureSampler.enabled =
                performanceSliceMode != PerformanceSliceMode.NoSamplingAndDisplay &&
                !effectiveShowGpuDepthGrid &&
                !effectiveShowStableIdGrid &&
                !useViewportAttachedGridMainline;
            coverageSampler.enabled = false;
            surfacePatchGridRenderer.Configure(
                coverageSampler,
                sampleCamera,
                instancedPointMaterialOverride != null ? instancedPointMaterialOverride : pointMaterialOverride);
            surfacePatchGridRenderer.enabled = false;
            environmentDepthGridRenderer.Configure(sampleCamera, environmentDepthManager, null);
            environmentDepthGridRenderer.enabled = effectiveShowGpuDepthGrid;
            stableIdGridRenderer.Configure(sampleCamera, environmentDepthManager, instancedPointMaterialOverride != null
                ? instancedPointMaterialOverride
                : pointMaterialOverride);
            stableIdGridRenderer.enabled = effectiveShowStableIdGrid;
            viewportAttachedGridRenderer.Configure(sampleCamera, environmentDepthManager, instancedPointMaterialOverride != null
                ? instancedPointMaterialOverride
                : pointMaterialOverride);
            viewportAttachedGridRenderer.enabled = effectiveShowViewportAttachedGrid;
            debugPointRenderer.Configure(m_voxelAccumulator, pointMaterialOverride);
            debugPointRenderer.SetShowDebugPoints(effectiveShowDebugFallbackPoints);
            debugPointRenderer.enabled = effectiveShowDebugFallbackPoints;
            diagnosticsHud.Configure(sampleCamera, m_voxelAccumulator, m_pointRenderer, coverageSampler);
            diagnosticsHud.enabled = effectiveShowDiagnosticsHud;
            hiddenStabilizer.Configure(sampleCamera, m_voxelAccumulator, m_pointRenderer, coverageSampler);
            hiddenStabilizer.enabled = effectiveEnableHiddenStabilizer;
            invisibleForegroundStabilizer.Configure(sampleCamera, m_voxelAccumulator, m_pointRenderer, coverageSampler);
            invisibleForegroundStabilizer.enabled = effectiveEnableInvisibleForegroundStabilizer;

            if (debugLog)
            {
                string cameraName = sampleCamera != null ? sampleCamera.name : "<missing>";
                string raycastName = environmentRaycastManager != null ? environmentRaycastManager.name : "<missing>";
                Debug.Log($"[DepthEffectsPointCloudMapMotif] camera={cameraName}, raycast={raycastName}, perfSlice={performanceSliceMode}");
            }
        }

        private void ApplyPerformanceSliceMode()
        {
            float seedScale = 1f;
            float patchScale = 1f;
            float intervalScale = 1f;
            float rayDistanceScale = 1f;
            float displayPointScale = 1f;
            float displayRefreshScale = 1f;
            bool coalesceRevisionChanges = false;

            switch (performanceSliceMode)
            {
                case PerformanceSliceMode.SamplerHalf:
                    seedScale = 0.5f;
                    patchScale = 0.5f;
                    intervalScale = 2f;
                    break;
                case PerformanceSliceMode.DisplayHalf:
                    displayPointScale = 0.5f;
                    displayRefreshScale = 2f;
                    coalesceRevisionChanges = true;
                    break;
                case PerformanceSliceMode.BothHalf:
                    seedScale = 0.5f;
                    patchScale = 0.5f;
                    intervalScale = 2f;
                    displayPointScale = 0.5f;
                    displayRefreshScale = 2f;
                    coalesceRevisionChanges = true;
                    break;
                case PerformanceSliceMode.SamplerQuarter:
                    seedScale = 0.25f;
                    patchScale = 0.25f;
                    intervalScale = 4f;
                    break;
                case PerformanceSliceMode.DisplayQuarter:
                    displayPointScale = 0.25f;
                    displayRefreshScale = 4f;
                    coalesceRevisionChanges = true;
                    break;
                case PerformanceSliceMode.NoDisplay:
                    break;
                case PerformanceSliceMode.NoSamplingAndDisplay:
                    break;
            }

            if (depthTextureSampler != null)
                depthTextureSampler.ApplyPerformanceSlice(seedScale, intervalScale, rayDistanceScale);
            m_pointRenderer.ApplyPerformanceSlice(displayPointScale, displayRefreshScale, coalesceRevisionChanges);
        }
    }
}

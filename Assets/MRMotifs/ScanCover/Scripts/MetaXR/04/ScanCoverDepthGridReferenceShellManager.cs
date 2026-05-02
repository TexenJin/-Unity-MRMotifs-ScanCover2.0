using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverDepthGridReferenceShellManager : MonoBehaviour
{
    [Header("Refs")]
    public ScanCoverDepthGridPointCloud depthGridPointCloud;
    public ScanCoverTsdfBranch tsdfBranch;

    [Header("Window Accumulation")]
    public bool captureWindowWhilePreviewRuns = true;
    [Min(1)] public int accumulationFrameWindow = 16;
    [Min(0.01f)] public float accumulationIntervalSeconds = 0.03f;
    [Range(0f, 1f)] public float accumulationAverageBlend = 0.35f;
    [Range(0f, 1f)] public float recencyWeightFalloff = 0.12f;
    [Range(0f, 4f)] public float tsdfTrendWeightBias = 0.35f;
    [Min(0f)] public float tsdfTrendMinWeight = 0.50f;

    [Header("Integration")]
    [Min(0.001f)] public float duplicateDistanceMeters = 0.015f;
    [Min(0.001f)] public float insideRejectDistanceMeters = 0.012f;
    [Min(0.001f)] public float outwardReplaceDistanceMeters = 0.025f;
    [Range(-1f, 1f)] public float minNormalDot = 0.2f;
    [Range(0f, 1f)] public float replaceConfidenceBias = 0.05f;
    [Min(0.001f)] public float clusterCompressionRadiusMeters = 0.02f;
    [Min(0.001f)] public float clusterCompressionTangentMeters = 0.015f;
    [Min(0.001f)] public float clusterCompressionNormalMeters = 0.015f;

    [Header("Shell Thinning")]
    public bool thinReferenceLayers = true;
    [Range(-1f, 1f)] public float shellThinMinNormalDot = 0.65f;
    [Min(0.001f)] public float shellThinTangentMeters = 0.02f;
    [Min(0.001f)] public float shellThinNormalMeters = 0.03f;
    [Range(0f, 1f)] public float shellThinConfidenceTolerance = 0.15f;
    [Range(-1f, 1f)] public float shellThinMinObservationDot = 0.75f;
    [Min(0.001f)] public float shellThinObservationDepthMeters = 0.10f;

    [Header("Visibility Culling")]
    public bool cullOccludedLayers = true;
    [Range(-1f, 1f)] public float occlusionMinRayDot = 0.992f;
    [Min(0.001f)] public float occlusionMinDepthDeltaMeters = 0.05f;
    [Min(0.001f)] public float occlusionMaxTangentMeters = 0.025f;
    [Range(0f, 1f)] public float occlusionConfidenceTolerance = 0.10f;

    [Header("Incremental Freeze Gate")]
    public bool enforceNarrowBandFreezeGate = true;
    [Min(1)] public int minOverlapPointCount = 64;
    [Range(0f, 1f)] public float minOverlapRatio = 0.10f;
    [Range(0f, 1f)] public float maxOverlapRatio = 0.42f;
    [Min(1)] public int minNovelPointCount = 72;
    [Range(0f, 1f)] public float minNovelRatio = 0.18f;
    [Range(0f, 4f)] public float minNovelToOverlapRatio = 0.60f;

    [Header("Fused Point Cloud Display")]
    public bool showPointCloud = true;
    public PrimitiveType pointPrimitive = PrimitiveType.Quad;
    public bool billboardPointQuads = true;
    public bool scalePointByConfidence = true;
    public bool regularizeDisplayPoints = true;
    [Min(0.001f)] public float displayPointSpacingMeters = 0.0095f;
    [Min(0.001f)] public float displayPatchPlaneBandMeters = 0.06f;
    [Min(1f)] public float displayPatchNormalQuantizationSteps = 2f;
    [Range(-1f, 1f)] public float displayPatchMergeNormalDot = 0.965f;
    [Min(0.001f)] public float displayPatchMergeDepthMeters = 0.10f;
    [Min(0.001f)] public float pointScaleMeters = 0.0022f;
    [Min(0.001f)] public float minPointScaleMeters = 0.0016f;
    [Min(0.001f)] public float maxPointScaleMeters = 0.0030f;
    public bool colorizePointsByBatch = true;
    public Material pointMaterialOverride;
    public Color pointColor = new Color(0.95f, 0.95f, 0.97f, 1f);

    [Header("OBJ Export")]
    public string exportDirectoryOverride = "";
    public bool exportNormals = true;

    [Header("Debug")]
    public bool debugLog = false;

    public bool HasReferenceState => _referenceClusters.Count > 0;
    public int WindowSampleCount => _windowSnapshots.Count;
    public string LastIssue { get; private set; }
    public string LastExportPath { get; private set; }
    public float LastOverlapRatio { get; private set; }
    public float LastNovelRatio { get; private set; }
    public int LastOverlapPointCount { get; private set; }
    public int LastNovelPointCount { get; private set; }
    public int LastDisplayPointCount { get; private set; }
    public int FusedClusterCount => _referenceClusters.Count;

    private GameObject _shellRoot;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private GameObject _pointRoot;
    private Mesh _mesh;
    private Material _runtimePointMaterial;
    private MaterialPropertyBlock _pointPropertyBlock;
    private ScanCoverDepthGridPointCloud.GridStateSnapshot _referenceSnapshot;
    private readonly List<ScanCoverDepthGridPointCloud.GridStateSnapshot> _windowSnapshots = new List<ScanCoverDepthGridPointCloud.GridStateSnapshot>(8);
    private readonly List<GameObject> _points = new List<GameObject>(2048);
    private float _nextAccumulationSampleTime;
    private int _lastCapturedFrameIndex = -1;
    private int _nextClusterId;
    private int _currentIntegrationBatch;

    private readonly List<Vector3> _verts = new List<Vector3>(4096);
    private readonly List<Vector3> _normals = new List<Vector3>(4096);
    private readonly List<float> _confidences = new List<float>(4096);
    private readonly List<int> _tris = new List<int>(8192);
    private readonly List<ReferenceClusterState> _referenceClusters = new List<ReferenceClusterState>(4096);

    private struct AccumulationCluster
    {
        public ScanCoverDepthGridPointCloud.GridStateEntry representative;
        public float totalScore;
        public float totalConfidence;
        public int sampleCount;
        public bool initialized;
    }

    private struct ReferenceClusterState
    {
        public int id;
        public int batchIndex;
        public Vector3 worldPos;
        public Vector3 normal;
        public Vector3 observationDir;
        public float confidence;
        public int sampleCount;
    }

    private struct DisplayPointState
    {
        public Vector3 worldPos;
        public float confidence;
        public int batchIndex;
    }

    private struct DisplayGridCoord
    {
        public float u;
        public float v;
        public float w;
    }

    private struct DisplayBucketKey
    {
        public int patch;
        public int u;
        public int v;
        public int w;
    }

    private struct DisplayPatchKey
    {
        public int nx;
        public int ny;
        public int nz;
        public int w;
    }

    private struct DisplayPatchState
    {
        public Vector3 normal;
        public Vector3 centroid;
        public float planeDepth;
        public Vector3 tangent;
        public Vector3 bitangent;
        public float totalWeight;
        public int sampleCount;
    }

    private struct DisplayPatchFamilyState
    {
        public Vector3 normal;
        public Vector3 centroid;
        public float planeDepth;
        public Vector3 tangent;
        public Vector3 bitangent;
        public float totalWeight;
        public int sampleCount;
    }

    public void EnsureInitialized()
    {
        ResolveRefs();
        EnsureShellObjects();
    }

    private void Update()
    {
        UpdatePointBillboards();

        if (captureWindowWhilePreviewRuns)
            CaptureCurrentStateIntoWindow();
    }

    public bool IntegrateCurrentGridState()
    {
        ResolveRefs();
        if (depthGridPointCloud == null)
            return SetIssue("DepthGridPointCloud is missing.");

        if (!TryGetIncomingSnapshot(out ScanCoverDepthGridPointCloud.GridStateSnapshot incoming))
            return false;

        if (!ValidateIncrementalFreeze(incoming))
            return false;

        _currentIntegrationBatch++;
        IntegrateSnapshot(incoming, out int added, out int replaced, out int rejected);
        RebuildFusedPointCloudDisplay();
        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridReferenceShellManager] integrate window={_windowSnapshots.Count}, visible={incoming.visibleCount}, added={added}, replaced={replaced}, rejected={rejected}, fusedCount={_referenceClusters.Count}");
        return true;
    }

    public bool ExportReferenceStateJson(out string exportPath)
    {
        exportPath = null;
        if (!HasReferenceState)
            return SetIssue("Fused point cloud state is empty.");

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        exportPath = Path.Combine(exportDirectory, $"FusedPointCloudState_{System.DateTime.Now:yyyyMMdd_HHmmss}.json");
        File.WriteAllText(exportPath, JsonUtility.ToJson(_referenceSnapshot, true), Encoding.UTF8);
        LastIssue = null;
        LastExportPath = exportPath;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridReferenceShellManager] Exported FPC state => {exportPath}");
        return true;
    }

    public bool ExportReferenceShellAsObj(out string exportPath)
    {
        exportPath = null;
        return SetIssue("OBJ export is disabled in fused point cloud display mode.");
    }

    public void ClearAll()
    {
        _referenceSnapshot = null;
        _referenceClusters.Clear();
        _windowSnapshots.Clear();
        _nextAccumulationSampleTime = 0f;
        _lastCapturedFrameIndex = -1;
        _nextClusterId = 0;
        _currentIntegrationBatch = 0;
        LastIssue = null;
        LastExportPath = null;
        LastOverlapRatio = 0f;
        LastNovelRatio = 0f;
        LastOverlapPointCount = 0;
        LastNovelPointCount = 0;
        LastDisplayPointCount = 0;
        if (_mesh != null)
            _mesh.Clear();
        ClearPoints();
        if (_shellRoot != null)
            _shellRoot.SetActive(false);
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_runtimePointMaterial != null)
            Destroy(_runtimePointMaterial);
        ClearPoints();
        if (_shellRoot != null)
            Destroy(_shellRoot);
    }

    private void IntegrateSnapshot(ScanCoverDepthGridPointCloud.GridStateSnapshot incoming, out int added, out int replaced, out int rejected)
    {
        added = 0;
        replaced = 0;
        rejected = 0;
        Vector3 observationOrigin = depthGridPointCloud != null ? depthGridPointCloud.CurrentObservationOrigin : transform.position;

        for (int i = 0; i < incoming.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry candidate = incoming.entries[i];
            if (!candidate.valid)
                continue;

            Vector3 candidateObservationDir = ResolveObservationDirection(observationOrigin, candidate.worldPos, candidate.normal);
            int clusterIndex = FindBestClusterIndex(candidate, candidateObservationDir);
            if (clusterIndex < 0)
            {
                _referenceClusters.Add(new ReferenceClusterState
                {
                    id = _nextClusterId++,
                    batchIndex = _currentIntegrationBatch,
                    worldPos = candidate.worldPos,
                    normal = candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : Vector3.up,
                    observationDir = candidateObservationDir,
                    confidence = Mathf.Clamp01(candidate.confidence),
                    sampleCount = 1
                });
                added++;
                continue;
            }

            ReferenceClusterState cluster = _referenceClusters[clusterIndex];
            Vector3 clusterNormal = cluster.normal.sqrMagnitude > 1e-6f ? cluster.normal.normalized : Vector3.up;
            Vector3 candidateNormal = candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : clusterNormal;
            float normalDot = Vector3.Dot(clusterNormal, candidateNormal);
            float distance = Vector3.Distance(cluster.worldPos, candidate.worldPos);
            float signedOffset = Vector3.Dot(candidate.worldPos - cluster.worldPos, clusterNormal);

            bool overlap = normalDot >= minNormalDot && distance <= duplicateDistanceMeters;
            bool inside = normalDot >= minNormalDot && signedOffset < -insideRejectDistanceMeters;
            bool replace = overlap
                ? candidate.confidence > cluster.confidence + replaceConfidenceBias
                : !inside && (candidate.confidence > cluster.confidence + replaceConfidenceBias ||
                              (signedOffset > outwardReplaceDistanceMeters && candidate.confidence >= cluster.confidence * 0.85f));

            if (replace)
            {
                cluster.worldPos = candidate.worldPos;
                cluster.normal = candidateNormal;
                cluster.observationDir = BlendDirection(cluster.observationDir, candidateObservationDir, 0.65f);
                cluster.batchIndex = _currentIntegrationBatch;
                cluster.confidence = Mathf.Clamp01(Mathf.Max(cluster.confidence, candidate.confidence));
                cluster.sampleCount += 1;
                _referenceClusters[clusterIndex] = cluster;
                replaced++;
                continue;
            }

            if (inside || overlap)
            {
                float blend = Mathf.Clamp01(accumulationAverageBlend / Mathf.Max(1f, cluster.sampleCount));
                cluster.worldPos = Vector3.Lerp(cluster.worldPos, candidate.worldPos, blend);
                Vector3 blendedNormal = Vector3.Lerp(clusterNormal, candidateNormal, blend);
                cluster.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : clusterNormal;
                cluster.observationDir = BlendDirection(cluster.observationDir, candidateObservationDir, blend);
                cluster.batchIndex = _currentIntegrationBatch;
                cluster.confidence = Mathf.Clamp01(Mathf.Max(cluster.confidence, Mathf.Lerp(cluster.confidence, candidate.confidence, blend)));
                cluster.sampleCount += 1;
                _referenceClusters[clusterIndex] = cluster;
                continue;
            }

            _referenceClusters.Add(new ReferenceClusterState
            {
                id = _nextClusterId++,
                batchIndex = _currentIntegrationBatch,
                worldPos = candidate.worldPos,
                normal = candidateNormal,
                observationDir = candidateObservationDir,
                confidence = Mathf.Clamp01(candidate.confidence),
                sampleCount = 1
            });
            added++;
        }

        CompactReferenceClusters();
        ThinReferenceLayers();
        CullOccludedLayersFromObservation(observationOrigin);
        RebuildReferenceSnapshotFromClusters(incoming);
    }

    private bool TryGetIncomingSnapshot(out ScanCoverDepthGridPointCloud.GridStateSnapshot incoming)
    {
        incoming = null;
        CaptureCurrentStateIntoWindow(force: true);

        if (TryBuildAccumulatedSnapshot(out incoming))
            return true;

        if (!depthGridPointCloud.TryGetCurrentGridState(out incoming))
            return SetIssue(depthGridPointCloud.LastIssue ?? "Grid state is unavailable.");

        return true;
    }

    private bool ValidateIncrementalFreeze(ScanCoverDepthGridPointCloud.GridStateSnapshot incoming)
    {
        LastOverlapRatio = 0f;
        LastNovelRatio = 0f;
        LastOverlapPointCount = 0;
        LastNovelPointCount = 0;

        if (!enforceNarrowBandFreezeGate || !HasReferenceState || _referenceClusters.Count <= 0)
            return true;

        if (incoming == null || incoming.entries == null || incoming.entries.Length <= 0 || incoming.visibleCount <= 0)
            return SetIssue("Freeze rejected: incoming grid state is empty.");

        Vector3 observationOrigin = depthGridPointCloud != null ? depthGridPointCloud.CurrentObservationOrigin : transform.position;
        int visibleCount = 0;
        int overlapCount = 0;
        int novelCount = 0;
        List<Vector3> overlapPositions = new List<Vector3>(128);

        for (int i = 0; i < incoming.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry candidate = incoming.entries[i];
            if (!candidate.valid)
                continue;

            visibleCount++;
            Vector3 candidateObservationDir = ResolveObservationDirection(observationOrigin, candidate.worldPos, candidate.normal);
            int clusterIndex = FindBestClusterIndex(candidate, candidateObservationDir);
            if (clusterIndex >= 0)
            {
                overlapCount++;
                overlapPositions.Add(candidate.worldPos);
            }
            else
                novelCount++;
        }

        if (visibleCount <= 0)
            return SetIssue("Freeze rejected: incoming grid state has no visible points.");

        LastOverlapPointCount = overlapCount;
        LastNovelPointCount = novelCount;
        LastOverlapRatio = (float)overlapCount / visibleCount;
        LastNovelRatio = (float)novelCount / visibleCount;

        if (overlapCount < Mathf.Max(1, minOverlapPointCount) || LastOverlapRatio < Mathf.Clamp01(minOverlapRatio))
            return SetIssue($"Freeze rejected: overlap too small ({overlapCount}/{visibleCount}, ratio={LastOverlapRatio:F2}).");

        if (LastOverlapRatio > Mathf.Clamp01(maxOverlapRatio))
            return SetIssue($"Freeze rejected: overlap too large ({overlapCount}/{visibleCount}, ratio={LastOverlapRatio:F2}).");

        if (novelCount < Mathf.Max(1, minNovelPointCount) || LastNovelRatio < Mathf.Clamp01(minNovelRatio))
            return SetIssue($"Freeze rejected: novel region too small ({novelCount}/{visibleCount}, ratio={LastNovelRatio:F2}).");

        float novelToOverlapRatio = overlapCount > 0 ? (float)novelCount / overlapCount : float.PositiveInfinity;
        if (novelToOverlapRatio < Mathf.Max(0f, minNovelToOverlapRatio))
            return SetIssue($"Freeze rejected: overlap dominates novel region (novel/overlap={novelToOverlapRatio:F2}).");

        return true;
    }

    private void RebuildFusedPointCloudDisplay()
    {
        EnsureShellObjects();
        if (!HasReferenceState)
        {
            ClearPoints();
            if (_shellRoot != null)
                _shellRoot.SetActive(false);
            return;
        }

        if (_mesh != null)
            _mesh.Clear();
        if (_meshRenderer != null)
            _meshRenderer.enabled = false;

        RebuildPoints();
        _shellRoot.SetActive(showPointCloud && _points.Count > 0);
    }

    private void RebuildPoints()
    {
        ClearPoints();
        if (!showPointCloud || !HasReferenceState || _pointRoot == null)
            return;

        Material pointMaterial = pointMaterialOverride != null ? pointMaterialOverride : _runtimePointMaterial;
        if (pointMaterial == null)
            return;

        if (_pointPropertyBlock == null)
            _pointPropertyBlock = new MaterialPropertyBlock();

        List<DisplayPointState> displayPoints = BuildDisplayPoints();
        LastDisplayPointCount = displayPoints.Count;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridReferenceShellManager] displayPoints={LastDisplayPointCount}, fusedClusters={_referenceClusters.Count}");

        for (int i = 0; i < displayPoints.Count; i++)
        {
            DisplayPointState pointState = displayPoints[i];

            GameObject point = GameObject.CreatePrimitive(pointPrimitive);
            point.name = $"Point_B{pointState.batchIndex:D2}_{i:D4}";
            point.transform.SetParent(_pointRoot.transform, false);
            point.transform.position = pointState.worldPos;
            point.transform.rotation = Quaternion.identity;

            float displayScale = Mathf.Max(0.001f, pointScaleMeters);
            if (scalePointByConfidence)
            {
                float confidence01 = Mathf.Clamp01(pointState.confidence);
                displayScale = Mathf.Lerp(Mathf.Max(0.001f, minPointScaleMeters), Mathf.Max(minPointScaleMeters, maxPointScaleMeters), confidence01);
            }

            point.transform.localScale = Vector3.one * displayScale;

            Collider collider = point.GetComponent<Collider>();
            if (collider != null)
                Destroy(collider);

            Renderer renderer = point.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = pointMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                ApplyPointBatchColor(renderer, pointState.batchIndex);
            }

            _points.Add(point);
        }

        _pointRoot.SetActive(_points.Count > 0);
        UpdatePointBillboards();
    }

    private List<DisplayPointState> BuildDisplayPoints()
    {
        if (!regularizeDisplayPoints || _referenceClusters.Count <= 1)
        {
            List<DisplayPointState> passthrough = new List<DisplayPointState>(_referenceClusters.Count);
            for (int i = 0; i < _referenceClusters.Count; i++)
            {
                ReferenceClusterState cluster = _referenceClusters[i];
                passthrough.Add(new DisplayPointState
                {
                    worldPos = cluster.worldPos,
                    confidence = cluster.confidence,
                    batchIndex = cluster.batchIndex
                });
            }

            return passthrough;
        }

        float spacing = Mathf.Max(0.001f, displayPointSpacingMeters);
        float normalBand = Mathf.Max(0.001f, displayPatchPlaneBandMeters);
        float normalQuantizationSteps = Mathf.Max(1f, displayPatchNormalQuantizationSteps);
        Dictionary<DisplayPatchKey, int> patchKeyToIndex = new Dictionary<DisplayPatchKey, int>(_referenceClusters.Count);
        List<DisplayPatchState> patches = new List<DisplayPatchState>(_referenceClusters.Count);
        int[] clusterToPatch = new int[_referenceClusters.Count];

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            ReferenceClusterState cluster = _referenceClusters[i];
            Vector3 patchNormal = ResolveDisplayPatchNormal(cluster.normal, normalQuantizationSteps);
            DisplayPatchKey patchKey = BuildDisplayPatchKey(cluster, patchNormal, normalBand);
            float clusterWeight = ResolveClusterScore(cluster);
            float clusterPlaneDepth = Vector3.Dot(cluster.worldPos, patchNormal);

            if (!patchKeyToIndex.TryGetValue(patchKey, out int patchIndex))
            {
                patchIndex = patches.Count;
                patchKeyToIndex.Add(patchKey, patchIndex);
                BuildStablePatchBasis(patchNormal, out Vector3 tangent, out Vector3 bitangent);
                patches.Add(new DisplayPatchState
                {
                    normal = patchNormal,
                    centroid = cluster.worldPos,
                    planeDepth = clusterPlaneDepth,
                    tangent = tangent,
                    bitangent = bitangent,
                    totalWeight = clusterWeight,
                    sampleCount = 1
                });
            }
            else
            {
                DisplayPatchState patch = patches[patchIndex];
                float totalWeight = patch.totalWeight + clusterWeight;
                if (totalWeight <= 1e-6f)
                    totalWeight = 1f;

                Vector3 blendedNormal = patch.normal * patch.totalWeight + patchNormal * clusterWeight;
                patch.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : patch.normal;
                patch.centroid = (patch.centroid * patch.totalWeight + cluster.worldPos * clusterWeight) / totalWeight;
                patch.planeDepth = (patch.planeDepth * patch.totalWeight + clusterPlaneDepth * clusterWeight) / totalWeight;
                BuildStablePatchBasis(patch.normal, out patch.tangent, out patch.bitangent);
                patch.totalWeight = totalWeight;
                patch.sampleCount += 1;
                patches[patchIndex] = patch;
            }

            clusterToPatch[i] = patchIndex;
        }

        Dictionary<DisplayBucketKey, int> bucketToIndex = new Dictionary<DisplayBucketKey, int>(_referenceClusters.Count);
        List<DisplayPointState> regularized = new List<DisplayPointState>(_referenceClusters.Count);
        List<float> bucketWeights = new List<float>(_referenceClusters.Count);
        List<int> bucketPatchIndices = new List<int>(_referenceClusters.Count);
        List<DisplayBucketKey> bucketKeys = new List<DisplayBucketKey>(_referenceClusters.Count);
        int[] patchToFamily = BuildDisplayPatchFamilies(patches);

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            ReferenceClusterState cluster = _referenceClusters[i];
            int familyIndex = patchToFamily[clusterToPatch[i]];
            DisplayPatchFamilyState family = BuildDisplayPatchFamilyState(familyIndex, patches, patchToFamily);
            DisplayGridCoord coord = ResolveDisplayGridCoord(family, cluster.worldPos, spacing, normalBand);
            DisplayBucketKey bucket = new DisplayBucketKey
            {
                patch = familyIndex,
                u = Mathf.RoundToInt(coord.u),
                v = Mathf.RoundToInt(coord.v),
                w = Mathf.RoundToInt(coord.w)
            };

            AccumulateDisplayBucket(
                familyIndex,
                family,
                bucket,
                ResolveClusterScore(cluster),
                cluster,
                spacing,
                normalBand,
                bucketToIndex,
                regularized,
                bucketWeights,
                bucketPatchIndices,
                bucketKeys,
                patches,
                patchToFamily);
        }

        return regularized;
    }

    private static DisplayPatchKey BuildDisplayPatchKey(ReferenceClusterState cluster, Vector3 patchNormal, float normalBand)
    {
        float planeCoord = Vector3.Dot(cluster.worldPos, patchNormal) / normalBand;

        return new DisplayPatchKey
        {
            nx = Mathf.RoundToInt(patchNormal.x * 100f),
            ny = Mathf.RoundToInt(patchNormal.y * 100f),
            nz = Mathf.RoundToInt(patchNormal.z * 100f),
            w = Mathf.RoundToInt(planeCoord)
        };
    }

    private static Vector3 ResolveDisplayPatchNormal(Vector3 normal, float quantizationSteps)
    {
        Vector3 n = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up;
        float step = 1f / Mathf.Max(1f, quantizationSteps);
        Vector3 snapped = new Vector3(
            Mathf.Round(n.x / step) * step,
            Mathf.Round(n.y / step) * step,
            Mathf.Round(n.z / step) * step);
        if (snapped.sqrMagnitude <= 1e-6f)
            return n;
        return snapped.normalized;
    }

    private static void BuildStablePatchBasis(Vector3 patchNormal, out Vector3 tangent, out Vector3 bitangent)
    {
        Vector3 n = patchNormal.sqrMagnitude > 1e-6f ? patchNormal.normalized : Vector3.up;
        float upDot = Mathf.Abs(Vector3.Dot(n, Vector3.up));
        if (upDot < 0.35f)
        {
            bitangent = Vector3.up;
            tangent = Vector3.Cross(bitangent, n).normalized;
            if (NeedsAxisFlip(tangent))
                tangent = -tangent;
            bitangent = Vector3.Cross(n, tangent).normalized;
            return;
        }

        if (upDot > 0.85f)
        {
            tangent = Vector3.right - n * Vector3.Dot(Vector3.right, n);
            if (tangent.sqrMagnitude <= 1e-6f)
                tangent = Vector3.forward - n * Vector3.Dot(Vector3.forward, n);
            tangent.Normalize();
            if (NeedsAxisFlip(tangent))
                tangent = -tangent;
            bitangent = Vector3.Cross(n, tangent).normalized;
            if (NeedsAxisFlip(bitangent))
                bitangent = -bitangent;
            return;
        }

        Vector3 tangentSeed = ResolveStableTangentSeed(n);
        tangent = tangentSeed - n * Vector3.Dot(tangentSeed, n);
        if (tangent.sqrMagnitude <= 1e-6f)
            tangent = Vector3.right - n * Vector3.Dot(Vector3.right, n);
        tangent.Normalize();
        if (NeedsAxisFlip(tangent))
            tangent = -tangent;
        bitangent = Vector3.Cross(n, tangent).normalized;
        if (NeedsAxisFlip(bitangent))
        {
            bitangent = -bitangent;
            tangent = Vector3.Cross(bitangent, n).normalized;
        }
    }

    private static DisplayGridCoord ResolveDisplayGridCoord(DisplayPatchFamilyState patch, Vector3 position, float spacing, float normalBand)
    {
        float uCoord = Vector3.Dot(position, patch.tangent) / spacing;
        float vCoord = Vector3.Dot(position, patch.bitangent) / spacing;
        float wCoord = (Vector3.Dot(position, patch.normal) - patch.planeDepth) / normalBand;

        return new DisplayGridCoord
        {
            u = uCoord,
            v = vCoord,
            w = wCoord
        };
    }

    private static Vector3 ResolveDisplayPointPosition(DisplayPatchFamilyState patch, DisplayBucketKey bucket, float spacing, float normalBand)
    {
        return patch.tangent * (bucket.u * spacing)
            + patch.bitangent * (bucket.v * spacing)
            + patch.normal * (patch.planeDepth + bucket.w * normalBand);
    }

    private static void AccumulateDisplayBucket(
        int familyIndex,
        DisplayPatchFamilyState family,
        DisplayBucketKey bucket,
        float clusterWeight,
        ReferenceClusterState cluster,
        float spacing,
        float normalBand,
        Dictionary<DisplayBucketKey, int> bucketToIndex,
        List<DisplayPointState> regularized,
        List<float> bucketWeights,
        List<int> bucketPatchIndices,
        List<DisplayBucketKey> bucketKeys,
        List<DisplayPatchState> patches,
        int[] patchToFamily)
    {
        if (clusterWeight <= 1e-5f)
            return;

        if (bucketToIndex.TryGetValue(bucket, out int displayIndex))
        {
            float accumulatedWeight = bucketWeights[displayIndex];
            float totalWeight = accumulatedWeight + clusterWeight;
            if (totalWeight <= 1e-6f)
                totalWeight = 1f;

            DisplayPointState merged = regularized[displayIndex];
            merged.worldPos = ResolveDisplayPointPosition(BuildDisplayPatchFamilyState(bucketPatchIndices[displayIndex], patches, patchToFamily), bucketKeys[displayIndex], spacing, normalBand);
            merged.confidence = Mathf.Clamp01((merged.confidence * accumulatedWeight + cluster.confidence * clusterWeight) / totalWeight);
            if (clusterWeight >= accumulatedWeight * 0.75f)
                merged.batchIndex = cluster.batchIndex;
            regularized[displayIndex] = merged;
            bucketWeights[displayIndex] = totalWeight;
            return;
        }

        bucketToIndex.Add(bucket, regularized.Count);
        bucketPatchIndices.Add(familyIndex);
        bucketKeys.Add(bucket);
        regularized.Add(new DisplayPointState
        {
            worldPos = ResolveDisplayPointPosition(family, bucket, spacing, normalBand),
            confidence = cluster.confidence,
            batchIndex = cluster.batchIndex
        });
        bucketWeights.Add(clusterWeight);
    }

    private int[] BuildDisplayPatchFamilies(List<DisplayPatchState> patches)
    {
        int[] patchToFamily = new int[patches.Count];
        List<DisplayPatchFamilyState> families = new List<DisplayPatchFamilyState>(patches.Count);
        float mergeNormalDot = Mathf.Clamp(displayPatchMergeNormalDot, -1f, 1f);
        float mergeDepthMeters = Mathf.Max(0.001f, displayPatchMergeDepthMeters);

        for (int i = 0; i < patches.Count; i++)
        {
            DisplayPatchState patch = patches[i];
            int bestFamily = -1;
            float bestScore = float.NegativeInfinity;

            for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
            {
                DisplayPatchFamilyState family = families[familyIndex];
                float normalDot = Vector3.Dot(family.normal, patch.normal);
                if (normalDot < mergeNormalDot)
                    continue;

                float depthDelta = Mathf.Abs(family.planeDepth - patch.planeDepth);
                if (depthDelta > mergeDepthMeters)
                    continue;

                float score = normalDot - depthDelta;
                if (score > bestScore)
                {
                    bestScore = score;
                    bestFamily = familyIndex;
                }
            }

            if (bestFamily < 0)
            {
                bestFamily = families.Count;
                families.Add(new DisplayPatchFamilyState
                {
                    normal = patch.normal,
                    centroid = patch.centroid,
                    planeDepth = patch.planeDepth,
                    tangent = patch.tangent,
                    bitangent = patch.bitangent,
                    totalWeight = patch.totalWeight,
                    sampleCount = patch.sampleCount
                });
            }
            else
            {
                DisplayPatchFamilyState family = families[bestFamily];
                float totalWeight = family.totalWeight + patch.totalWeight;
                if (totalWeight <= 1e-6f)
                    totalWeight = 1f;

                Vector3 blendedNormal = family.normal * family.totalWeight + patch.normal * patch.totalWeight;
                family.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : family.normal;
                family.centroid = (family.centroid * family.totalWeight + patch.centroid * patch.totalWeight) / totalWeight;
                family.planeDepth = (family.planeDepth * family.totalWeight + patch.planeDepth * patch.totalWeight) / totalWeight;
                BuildStablePatchBasis(family.normal, out family.tangent, out family.bitangent);
                family.totalWeight = totalWeight;
                family.sampleCount += patch.sampleCount;
                families[bestFamily] = family;
            }

            patchToFamily[i] = bestFamily;
        }

        return patchToFamily;
    }

    private static DisplayPatchFamilyState BuildDisplayPatchFamilyState(int familyIndex, List<DisplayPatchState> patches, int[] patchToFamily)
    {
        DisplayPatchFamilyState family = default;
        bool initialized = false;

        for (int i = 0; i < patches.Count; i++)
        {
            if (patchToFamily[i] != familyIndex)
                continue;

            DisplayPatchState patch = patches[i];
            if (!initialized)
            {
                family = new DisplayPatchFamilyState
                {
                    normal = patch.normal,
                    centroid = patch.centroid,
                    planeDepth = patch.planeDepth,
                    tangent = patch.tangent,
                    bitangent = patch.bitangent,
                    totalWeight = patch.totalWeight,
                    sampleCount = patch.sampleCount
                };
                initialized = true;
                continue;
            }

            float totalWeight = family.totalWeight + patch.totalWeight;
            if (totalWeight <= 1e-6f)
                totalWeight = 1f;

            Vector3 blendedNormal = family.normal * family.totalWeight + patch.normal * patch.totalWeight;
            family.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : family.normal;
            family.centroid = (family.centroid * family.totalWeight + patch.centroid * patch.totalWeight) / totalWeight;
            family.planeDepth = (family.planeDepth * family.totalWeight + patch.planeDepth * patch.totalWeight) / totalWeight;
            BuildStablePatchBasis(family.normal, out family.tangent, out family.bitangent);
            family.totalWeight = totalWeight;
            family.sampleCount += patch.sampleCount;
        }

        return family;
    }

    private static Vector3 ResolveStableTangentSeed(Vector3 normal)
    {
        Vector3[] candidates = { Vector3.right, Vector3.up, Vector3.forward };
        Vector3 best = Vector3.right;
        float bestScore = float.PositiveInfinity;
        for (int i = 0; i < candidates.Length; i++)
        {
            float score = Mathf.Abs(Vector3.Dot(normal, candidates[i]));
            if (score < bestScore)
            {
                bestScore = score;
                best = candidates[i];
            }
        }

        return best;
    }

    private static bool NeedsAxisFlip(Vector3 axis)
    {
        Vector3 abs = new Vector3(Mathf.Abs(axis.x), Mathf.Abs(axis.y), Mathf.Abs(axis.z));
        if (abs.x >= abs.y && abs.x >= abs.z)
            return axis.x < 0f;
        if (abs.y >= abs.x && abs.y >= abs.z)
            return axis.y < 0f;
        return axis.z < 0f;
    }

    private void ClearPoints()
    {
        for (int i = 0; i < _points.Count; i++)
        {
            if (_points[i] != null)
                Destroy(_points[i]);
        }
        _points.Clear();
        if (_pointRoot != null)
            _pointRoot.SetActive(false);
    }

    private int FindBestClusterIndex(ScanCoverDepthGridPointCloud.GridStateEntry candidate, Vector3 candidateObservationDir)
    {
        Vector3 candidateNormal = candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : Vector3.up;
        float searchRadius = Mathf.Max(duplicateDistanceMeters, outwardReplaceDistanceMeters, insideRejectDistanceMeters);
        float bestDistance = float.PositiveInfinity;
        int bestIndex = -1;

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            ReferenceClusterState cluster = _referenceClusters[i];
            Vector3 clusterNormal = cluster.normal.sqrMagnitude > 1e-6f ? cluster.normal.normalized : Vector3.up;
            float normalDot = Vector3.Dot(clusterNormal, candidateNormal);
            if (normalDot < minNormalDot)
                continue;

            Vector3 clusterObservationDir = cluster.observationDir.sqrMagnitude > 1e-6f ? cluster.observationDir.normalized : candidateObservationDir;
            if (candidateObservationDir.sqrMagnitude > 1e-6f && clusterObservationDir.sqrMagnitude > 1e-6f)
            {
                float observationDot = Vector3.Dot(clusterObservationDir, candidateObservationDir);
                if (observationDot < -0.15f)
                    continue;
            }

            float distance = Vector3.Distance(cluster.worldPos, candidate.worldPos);
            if (distance > searchRadius)
                continue;

            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private void RebuildReferenceSnapshotFromClusters(ScanCoverDepthGridPointCloud.GridStateSnapshot incoming)
    {
        _referenceClusters.Sort((a, b) => a.id.CompareTo(b.id));
        ScanCoverDepthGridPointCloud.GridStateEntry[] entries = new ScanCoverDepthGridPointCloud.GridStateEntry[_referenceClusters.Count];

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            ReferenceClusterState cluster = _referenceClusters[i];
            entries[i] = new ScanCoverDepthGridPointCloud.GridStateEntry
            {
                index = i,
                row = i,
                col = 0,
                valid = true,
                worldPos = cluster.worldPos,
                normal = cluster.normal,
                confidence = cluster.confidence
            };
        }

        _referenceSnapshot = new ScanCoverDepthGridPointCloud.GridStateSnapshot
        {
            componentName = incoming.componentName,
            samplingMode = incoming.samplingMode + "_FusedPointCloud",
            frameIndex = incoming.frameIndex,
            resolutionWidth = incoming.resolutionWidth,
            resolutionHeight = incoming.resolutionHeight,
            cellCount = entries.Length,
            visibleCount = entries.Length,
            entries = entries
        };
    }

    private void CompactReferenceClusters()
    {
        if (_referenceClusters.Count <= 1)
            return;

        float radius = Mathf.Max(0.001f, clusterCompressionRadiusMeters);
        float tangentLimit = Mathf.Max(0.001f, clusterCompressionTangentMeters);
        float normalLimit = Mathf.Max(0.001f, clusterCompressionNormalMeters);
        bool[] removed = new bool[_referenceClusters.Count];

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (removed[i])
                continue;

            ReferenceClusterState a = _referenceClusters[i];
            Vector3 normalA = a.normal.sqrMagnitude > 1e-6f ? a.normal.normalized : Vector3.up;

            for (int j = i + 1; j < _referenceClusters.Count; j++)
            {
                if (removed[j])
                    continue;

                ReferenceClusterState b = _referenceClusters[j];
                Vector3 normalB = b.normal.sqrMagnitude > 1e-6f ? b.normal.normalized : normalA;
                float normalDot = Vector3.Dot(normalA, normalB);
                if (normalDot < minNormalDot)
                    continue;

                Vector3 delta = b.worldPos - a.worldPos;
                float distance = delta.magnitude;
                if (distance > radius)
                    continue;

                Vector3 avgNormal = (normalA + normalB);
                if (avgNormal.sqrMagnitude <= 1e-6f)
                    avgNormal = normalA;
                else
                    avgNormal.Normalize();

                float signedOffset = Vector3.Dot(delta, avgNormal);
                float normalOffset = Mathf.Abs(signedOffset);
                Vector3 tangentDelta = delta - avgNormal * signedOffset;
                float tangentDistance = tangentDelta.magnitude;

                if (tangentDistance > tangentLimit || normalOffset > normalLimit)
                    continue;

                bool preferB = false;
                if (signedOffset > outwardReplaceDistanceMeters * 0.5f)
                    preferB = true;
                else if (signedOffset < -outwardReplaceDistanceMeters * 0.5f)
                    preferB = false;
                else
                    preferB = b.confidence > a.confidence + replaceConfidenceBias;

                if (preferB)
                {
                    b.worldPos = Vector3.Lerp(b.worldPos, a.worldPos, Mathf.Clamp01(accumulationAverageBlend * 0.35f));
                    Vector3 blendedNormal = (normalB + normalA).normalized;
                    b.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal : normalB;
                    b.confidence = Mathf.Max(a.confidence, b.confidence);
                    b.sampleCount = Mathf.Max(a.sampleCount, b.sampleCount) + 1;
                    _referenceClusters[j] = b;
                    removed[i] = true;
                    break;
                }

                a.worldPos = Vector3.Lerp(a.worldPos, b.worldPos, Mathf.Clamp01(accumulationAverageBlend * 0.35f));
                Vector3 mergedNormal = (normalA + normalB).normalized;
                a.normal = mergedNormal.sqrMagnitude > 1e-6f ? mergedNormal : normalA;
                a.confidence = Mathf.Max(a.confidence, b.confidence);
                a.sampleCount = Mathf.Max(a.sampleCount, b.sampleCount) + 1;
                _referenceClusters[i] = a;
                normalA = a.normal.sqrMagnitude > 1e-6f ? a.normal.normalized : normalA;
                removed[j] = true;
            }
        }

        if (_referenceClusters.Count == 0)
            return;

        List<ReferenceClusterState> compacted = new List<ReferenceClusterState>(_referenceClusters.Count);
        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (!removed[i])
                compacted.Add(_referenceClusters[i]);
        }

        _referenceClusters.Clear();
        _referenceClusters.AddRange(compacted);
    }

    private void ThinReferenceLayers()
    {
        if (!thinReferenceLayers || _referenceClusters.Count <= 1)
            return;

        float tangentLimit = Mathf.Max(0.001f, shellThinTangentMeters);
        float normalLimit = Mathf.Max(0.001f, shellThinNormalMeters);
        float normalDotLimit = Mathf.Max(minNormalDot, shellThinMinNormalDot);
        float observationDotLimit = Mathf.Clamp(shellThinMinObservationDot, -1f, 1f);
        float observationDepthLimit = Mathf.Max(0.001f, shellThinObservationDepthMeters);
        bool[] removed = new bool[_referenceClusters.Count];

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (removed[i])
                continue;

            ReferenceClusterState a = _referenceClusters[i];
            Vector3 normalA = a.normal.sqrMagnitude > 1e-6f ? a.normal.normalized : Vector3.up;

            for (int j = i + 1; j < _referenceClusters.Count; j++)
            {
                if (removed[j])
                    continue;

                ReferenceClusterState b = _referenceClusters[j];
                Vector3 normalB = b.normal.sqrMagnitude > 1e-6f ? b.normal.normalized : normalA;
                float normalDot = Vector3.Dot(normalA, normalB);
                if (normalDot < normalDotLimit)
                    continue;

                Vector3 observationA = a.observationDir.sqrMagnitude > 1e-6f ? a.observationDir.normalized : -normalA;
                Vector3 observationB = b.observationDir.sqrMagnitude > 1e-6f ? b.observationDir.normalized : -normalB;
                float observationDot = Vector3.Dot(observationA, observationB);
                if (observationDot < observationDotLimit)
                    continue;

                Vector3 avgNormal = normalA + normalB;
                if (avgNormal.sqrMagnitude <= 1e-6f)
                    avgNormal = normalA;
                else
                    avgNormal.Normalize();

                Vector3 delta = b.worldPos - a.worldPos;
                float signedOffset = Vector3.Dot(delta, avgNormal);
                float normalOffset = Mathf.Abs(signedOffset);
                if (normalOffset > normalLimit)
                    continue;

                Vector3 tangentDelta = delta - avgNormal * signedOffset;
                float tangentDistance = tangentDelta.magnitude;
                if (tangentDistance > tangentLimit)
                    continue;

                Vector3 avgObservation = observationA + observationB;
                if (avgObservation.sqrMagnitude <= 1e-6f)
                    avgObservation = observationA;
                else
                    avgObservation.Normalize();

                float observationDepthOffset = Vector3.Dot(delta, avgObservation);
                int winnerIndex = ChooseShellLayerWinner(a, b, signedOffset, observationDepthOffset, observationDepthLimit);
                if (winnerIndex == 0)
                {
                    removed[j] = true;
                    continue;
                }

                removed[i] = true;
                break;
            }
        }

        if (_referenceClusters.Count == 0)
            return;

        List<ReferenceClusterState> thinned = new List<ReferenceClusterState>(_referenceClusters.Count);
        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (!removed[i])
                thinned.Add(_referenceClusters[i]);
        }

        _referenceClusters.Clear();
        _referenceClusters.AddRange(thinned);
    }

    private int ChooseShellLayerWinner(ReferenceClusterState a, ReferenceClusterState b, float signedOffset, float observationDepthOffset, float observationDepthLimit)
    {
        float confidenceTolerance = Mathf.Clamp01(shellThinConfidenceTolerance);
        float sampleA = Mathf.Max(1f, a.sampleCount);
        float sampleB = Mathf.Max(1f, b.sampleCount);
        float scoreA = a.confidence + Mathf.Min(1f, sampleA / 6f) * 0.25f;
        float scoreB = b.confidence + Mathf.Min(1f, sampleB / 6f) * 0.25f;

        if (observationDepthOffset > observationDepthLimit && scoreA + confidenceTolerance >= scoreB)
            return 0;

        if (observationDepthOffset < -observationDepthLimit && scoreB + confidenceTolerance >= scoreA)
            return 1;

        if (signedOffset > insideRejectDistanceMeters && scoreB + confidenceTolerance >= scoreA)
            return 1;

        if (signedOffset < -insideRejectDistanceMeters && scoreA + confidenceTolerance >= scoreB)
            return 0;

        return scoreB > scoreA ? 1 : 0;
    }

    private void CullOccludedLayersFromObservation(Vector3 observationOrigin)
    {
        if (!cullOccludedLayers || _referenceClusters.Count <= 1)
            return;

        float minRayDot = Mathf.Clamp(occlusionMinRayDot, -1f, 1f);
        float minDepthDelta = Mathf.Max(0.001f, occlusionMinDepthDeltaMeters);
        float maxTangent = Mathf.Max(0.001f, occlusionMaxTangentMeters);
        float confidenceTolerance = Mathf.Clamp01(occlusionConfidenceTolerance);
        bool[] removed = new bool[_referenceClusters.Count];

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            ReferenceClusterState occluder = _referenceClusters[i];
            if (occluder.batchIndex != _currentIntegrationBatch)
                continue;

            Vector3 occluderOffset = occluder.worldPos - observationOrigin;
            float occluderDepth = occluderOffset.magnitude;
            if (occluderDepth <= 1e-6f)
                continue;

            Vector3 occluderRay = occluderOffset / occluderDepth;
            float occluderScore = ResolveClusterScore(occluder);

            for (int j = 0; j < _referenceClusters.Count; j++)
            {
                if (i == j || removed[j])
                    continue;

                ReferenceClusterState target = _referenceClusters[j];
                if (target.batchIndex == _currentIntegrationBatch)
                    continue;

                Vector3 targetOffset = target.worldPos - observationOrigin;
                float targetDepth = targetOffset.magnitude;
                if (targetDepth <= occluderDepth + minDepthDelta)
                    continue;

                Vector3 targetRay = targetOffset / targetDepth;
                float rayDot = Vector3.Dot(occluderRay, targetRay);
                if (rayDot < minRayDot)
                    continue;

                Vector3 rejection = targetOffset - occluderRay * Vector3.Dot(targetOffset, occluderRay);
                if (rejection.magnitude > maxTangent)
                    continue;

                float targetScore = ResolveClusterScore(target);
                if (occluderScore + confidenceTolerance < targetScore)
                    continue;

                removed[j] = true;
            }
        }

        if (_referenceClusters.Count == 0)
            return;

        List<ReferenceClusterState> visible = new List<ReferenceClusterState>(_referenceClusters.Count);
        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (!removed[i])
                visible.Add(_referenceClusters[i]);
        }

        _referenceClusters.Clear();
        _referenceClusters.AddRange(visible);
    }

    private static Vector3 ResolveObservationDirection(Vector3 observationOrigin, Vector3 worldPos, Vector3 fallbackNormal)
    {
        Vector3 direction = worldPos - observationOrigin;
        if (direction.sqrMagnitude > 1e-6f)
            return direction.normalized;

        if (fallbackNormal.sqrMagnitude > 1e-6f)
            return (-fallbackNormal).normalized;

        return Vector3.forward;
    }

    private static Vector3 BlendDirection(Vector3 current, Vector3 incoming, float blend)
    {
        Vector3 a = current.sqrMagnitude > 1e-6f ? current.normalized : Vector3.zero;
        Vector3 b = incoming.sqrMagnitude > 1e-6f ? incoming.normalized : a;
        Vector3 mixed = Vector3.Lerp(a, b, Mathf.Clamp01(blend));
        if (mixed.sqrMagnitude > 1e-6f)
            return mixed.normalized;
        if (b.sqrMagnitude > 1e-6f)
            return b.normalized;
        if (a.sqrMagnitude > 1e-6f)
            return a.normalized;
        return Vector3.forward;
    }

    private static float ResolveClusterScore(ReferenceClusterState cluster)
    {
        float sampleTerm = Mathf.Min(1f, Mathf.Max(1f, cluster.sampleCount) / 6f) * 0.25f;
        return Mathf.Clamp01(cluster.confidence) + sampleTerm;
    }

    private void ApplyPointBatchColor(Renderer renderer, int batchIndex)
    {
        if (renderer == null)
            return;

        Color color = colorizePointsByBatch ? EvaluateBatchColor(batchIndex) : pointColor;
        _pointPropertyBlock.Clear();
        if (renderer.sharedMaterial != null)
        {
            if (renderer.sharedMaterial.HasProperty("_BaseColor"))
                _pointPropertyBlock.SetColor("_BaseColor", color);
            if (renderer.sharedMaterial.HasProperty("_Color"))
                _pointPropertyBlock.SetColor("_Color", color);
        }

        renderer.SetPropertyBlock(_pointPropertyBlock);
    }

    private Color EvaluateBatchColor(int batchIndex)
    {
        if (batchIndex <= 0)
            return pointColor;

        float hue = Mathf.Repeat(0.13f + batchIndex * 0.19f, 1f);
        Color color = Color.HSVToRGB(hue, 0.65f, 1f);
        color.a = pointColor.a;
        return color;
    }

    private void ResolveRefs()
    {
        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (tsdfBranch == null)
            tsdfBranch = GetComponent<ScanCoverTsdfBranch>();
    }

    private void CaptureCurrentStateIntoWindow(bool force = false)
    {
        ResolveRefs();
        if (depthGridPointCloud == null)
            return;

        if (!force && Time.time < _nextAccumulationSampleTime)
            return;

        if (!depthGridPointCloud.TryGetCurrentGridState(out ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot))
            return;

        if (!force && snapshot.frameIndex == _lastCapturedFrameIndex)
            return;

        _lastCapturedFrameIndex = snapshot.frameIndex;
        _nextAccumulationSampleTime = Time.time + Mathf.Max(0.01f, accumulationIntervalSeconds);

        if (snapshot.visibleCount <= 0 || snapshot.entries == null || snapshot.entries.Length <= 0)
            return;

        _windowSnapshots.Add(CloneSnapshot(snapshot));
        int maxCount = Mathf.Max(1, accumulationFrameWindow);
        while (_windowSnapshots.Count > maxCount)
            _windowSnapshots.RemoveAt(0);
    }

    private bool TryBuildAccumulatedSnapshot(out ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
    {
        snapshot = null;
        if (_windowSnapshots.Count <= 0)
            return false;

        ScanCoverDepthGridPointCloud.GridStateSnapshot latest = _windowSnapshots[_windowSnapshots.Count - 1];
        if (latest.entries == null || latest.entries.Length <= 0)
            return false;

        int entryCount = latest.entries.Length;
        ScanCoverDepthGridPointCloud.GridStateEntry[] accumulated = new ScanCoverDepthGridPointCloud.GridStateEntry[entryCount];
        int visible = 0;

        for (int i = 0; i < entryCount; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry baseline = latest.entries[i];
            ScanCoverDepthGridPointCloud.GridStateEntry result = baseline;
            List<AccumulationCluster> clusters = new List<AccumulationCluster>(3);

            for (int s = 0; s < _windowSnapshots.Count; s++)
            {
                ScanCoverDepthGridPointCloud.GridStateSnapshot sample = _windowSnapshots[s];
                if (sample.entries == null || i >= sample.entries.Length)
                    continue;

                ScanCoverDepthGridPointCloud.GridStateEntry entry = sample.entries[i];
                if (!entry.valid)
                    continue;

                float recency01 = _windowSnapshots.Count <= 1 ? 1f : (float)s / (_windowSnapshots.Count - 1);
                float recencyWeight = Mathf.Lerp(1f - Mathf.Clamp01(recencyWeightFalloff), 1f, recency01);
                float tsdfWeight = ResolveTsdfTrendWeight(entry.worldPos);
                float score = Mathf.Max(0.001f, entry.confidence * recencyWeight * tsdfWeight);
                AddToClusters(clusters, entry, score);
            }

            if (TrySelectBestCluster(clusters, out AccumulationCluster bestCluster))
            {
                result.valid = true;
                result.worldPos = bestCluster.representative.worldPos;
                result.normal = bestCluster.representative.normal;
                result.confidence = Mathf.Clamp01(bestCluster.sampleCount > 0 ? bestCluster.totalConfidence / bestCluster.sampleCount : bestCluster.representative.confidence);
                visible++;
            }
            else
            {
                result.valid = false;
                result.worldPos = Vector3.zero;
                result.normal = Vector3.zero;
                result.confidence = 0f;
            }

            accumulated[i] = result;
        }

        snapshot = new ScanCoverDepthGridPointCloud.GridStateSnapshot
        {
            componentName = latest.componentName,
            samplingMode = latest.samplingMode + "_Accumulated",
            frameIndex = latest.frameIndex,
            resolutionWidth = latest.resolutionWidth,
            resolutionHeight = latest.resolutionHeight,
            cellCount = accumulated.Length,
            visibleCount = visible,
            entries = accumulated
        };
        return true;
    }

    private void AddToClusters(List<AccumulationCluster> clusters, ScanCoverDepthGridPointCloud.GridStateEntry entry, float score)
    {
        for (int i = 0; i < clusters.Count; i++)
        {
            if (!BelongsToCluster(clusters[i].representative, entry))
                continue;

            AccumulationCluster cluster = clusters[i];
            cluster.totalScore += score;
            cluster.totalConfidence += entry.confidence;
            cluster.sampleCount++;

            float representativeScore = cluster.representative.confidence;
            float candidateScore = entry.confidence + score;
            if (candidateScore > representativeScore)
                cluster.representative = entry;

            clusters[i] = cluster;
            return;
        }

        clusters.Add(new AccumulationCluster
        {
            representative = entry,
            totalScore = score,
            totalConfidence = entry.confidence,
            sampleCount = 1,
            initialized = true
        });
    }

    private bool TrySelectBestCluster(List<AccumulationCluster> clusters, out AccumulationCluster bestCluster)
    {
        bestCluster = default;
        bool hasBest = false;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < clusters.Count; i++)
        {
            AccumulationCluster cluster = clusters[i];
            if (!cluster.initialized || cluster.sampleCount <= 0)
                continue;

            float clusterScore = cluster.totalScore + cluster.sampleCount * 0.5f + cluster.totalConfidence;
            if (!hasBest || clusterScore > bestScore)
            {
                bestScore = clusterScore;
                bestCluster = cluster;
                hasBest = true;
            }
        }

        return hasBest;
    }

    private bool BelongsToCluster(ScanCoverDepthGridPointCloud.GridStateEntry representative, ScanCoverDepthGridPointCloud.GridStateEntry candidate)
    {
        Vector3 representativeNormal = representative.normal.sqrMagnitude > 1e-6f ? representative.normal.normalized : Vector3.up;
        Vector3 candidateNormal = candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : representativeNormal;
        float normalDot = Vector3.Dot(representativeNormal, candidateNormal);
        if (normalDot < minNormalDot)
            return false;

        return Vector3.Distance(representative.worldPos, candidate.worldPos) <= Mathf.Max(0.001f, duplicateDistanceMeters);
    }

    private float ResolveTsdfTrendWeight(Vector3 worldPos)
    {
        if (tsdfBranch == null)
            return 1f;

        Transform referenceFrame = tsdfBranch.referenceFrame != null ? tsdfBranch.referenceFrame : tsdfBranch.transform;
        if (referenceFrame == null)
            return 1f;

        Vector3 referenceLocal = referenceFrame.InverseTransformPoint(worldPos);
        if (!tsdfBranch.TryGetWeightAtReferenceLocalPosition(referenceLocal, out float weight))
            return 1f;

        if (weight < Mathf.Max(0f, tsdfTrendMinWeight))
            return 1f;

        return 1f + Mathf.Max(0f, tsdfTrendWeightBias);
    }

    private void EnsureShellObjects()
    {
        if (_shellRoot == null)
        {
            _shellRoot = new GameObject("[ScanCover] Fused Point Cloud");
            _shellRoot.transform.SetParent(null, false);
            _meshFilter = _shellRoot.AddComponent<MeshFilter>();
            _meshRenderer = _shellRoot.AddComponent<MeshRenderer>();

            _pointRoot = new GameObject("FusedPoints");
            _pointRoot.transform.SetParent(_shellRoot.transform, false);
        }

        if (_mesh == null)
        {
            _mesh = new Mesh { name = "ScanCover_FusedPointCloud_DisplayMesh" };
            _mesh.MarkDynamic();
        }

        if (pointMaterialOverride != null)
        {
            if (_runtimePointMaterial != null)
            {
                Destroy(_runtimePointMaterial);
                _runtimePointMaterial = null;
            }
        }
        else if (_runtimePointMaterial == null)
        {
            Shader pointShader = Shader.Find("Universal Render Pipeline/Unlit");
            if (pointShader == null)
                pointShader = Shader.Find("Unlit/Color");
            if (pointShader == null)
                pointShader = Shader.Find("Standard");
            _runtimePointMaterial = new Material(pointShader) { name = "ScanCover_FusedPointCloud_PointMat" };
        }

        _meshFilter.sharedMesh = _mesh;
        _meshRenderer.sharedMaterial = null;
        _meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _meshRenderer.receiveShadows = false;
        _meshRenderer.enabled = false;

        Material pointTarget = pointMaterialOverride != null ? pointMaterialOverride : _runtimePointMaterial;
        if (pointTarget != null)
        {
            if (pointTarget.HasProperty("_BaseColor")) pointTarget.SetColor("_BaseColor", pointColor);
            if (pointTarget.HasProperty("_Color")) pointTarget.SetColor("_Color", pointColor);
            if (pointTarget.HasProperty("_Cull")) pointTarget.SetFloat("_Cull", (float)CullMode.Off);
            if (pointTarget.HasProperty("_Glossiness")) pointTarget.SetFloat("_Glossiness", 0f);
            if (pointTarget.HasProperty("_Smoothness")) pointTarget.SetFloat("_Smoothness", 0f);
        }
    }

    private void UpdatePointBillboards()
    {
        if (!billboardPointQuads || pointPrimitive != PrimitiveType.Quad || _points.Count <= 0)
            return;

        Camera activeCamera = Camera.main;
        if (activeCamera == null)
            return;

        Quaternion cameraRotation = activeCamera.transform.rotation;
        for (int i = 0; i < _points.Count; i++)
        {
            GameObject point = _points[i];
            if (point != null)
                point.transform.rotation = cameraRotation;
        }
    }

    private string ResolveExportDirectory()
    {
        if (!string.IsNullOrWhiteSpace(exportDirectoryOverride))
            return Path.GetFullPath(exportDirectoryOverride);

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName ?? Application.dataPath;
        return Path.Combine(projectRoot, "ScanCoverExports");
    }

    private bool TryGetValidEntry(Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> byKey, int row, int col, out ScanCoverDepthGridPointCloud.GridStateEntry entry)
    {
        if (byKey.TryGetValue(ComposeKey(row, col), out entry) && entry.valid)
            return true;
        entry = default;
        return false;
    }

    private static ScanCoverDepthGridPointCloud.GridStateSnapshot CloneSnapshot(ScanCoverDepthGridPointCloud.GridStateSnapshot source)
    {
        ScanCoverDepthGridPointCloud.GridStateEntry[] entries = new ScanCoverDepthGridPointCloud.GridStateEntry[source.entries.Length];
        System.Array.Copy(source.entries, entries, source.entries.Length);
        return new ScanCoverDepthGridPointCloud.GridStateSnapshot
        {
            componentName = source.componentName,
            samplingMode = source.samplingMode,
            frameIndex = source.frameIndex,
            resolutionWidth = source.resolutionWidth,
            resolutionHeight = source.resolutionHeight,
            cellCount = source.cellCount,
            visibleCount = source.visibleCount,
            entries = entries
        };
    }

    private static int CountVisible(List<ScanCoverDepthGridPointCloud.GridStateEntry> entries)
    {
        int count = 0;
        for (int i = 0; i < entries.Count; i++)
            if (entries[i].valid)
                count++;
        return count;
    }

    private static long ComposeKey(int row, int col)
        => ((long)row << 32) | (uint)col;

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog && !string.IsNullOrEmpty(issue))
            Debug.LogWarning($"[ScanCoverDepthGridReferenceShellManager] {issue}");
        return false;
    }
}

using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public sealed class ScanCoverFusedPointCloudManager : MonoBehaviour
{
    [System.Serializable]
    public struct FusedClusterState
    {
        public int id;
        public int batchIndex;
        public Vector3 worldPos;
        public Vector3 normal;
        public Vector3 observationDir;
        public float confidence;
        public int sampleCount;
    }

    private struct AccumulationCluster
    {
        public ScanCoverDepthGridPointCloud.GridStateEntry representative;
        public float totalScore;
        public float totalConfidence;
        public int sampleCount;
        public bool initialized;
    }

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

    [Header("Debug")]
    public bool debugLog = false;

    public bool HasFusedState => _referenceClusters.Count > 0;
    public int WindowSampleCount => _windowSnapshots.Count;
    public string LastIssue { get; private set; }
    public float LastOverlapRatio { get; private set; }
    public float LastNovelRatio { get; private set; }
    public int LastOverlapPointCount { get; private set; }
    public int LastNovelPointCount { get; private set; }
    public int FusedClusterCount => _referenceClusters.Count;
    public int StateRevision { get; private set; }

    private readonly List<ScanCoverDepthGridPointCloud.GridStateSnapshot> _windowSnapshots = new List<ScanCoverDepthGridPointCloud.GridStateSnapshot>(8);
    private readonly List<FusedClusterState> _referenceClusters = new List<FusedClusterState>(4096);
    private float _nextAccumulationSampleTime;
    private int _lastCapturedFrameIndex = -1;
    private int _nextClusterId;
    private int _currentIntegrationBatch;

    private void Awake() => ResolveRefs();

    private void Update()
    {
        if (captureWindowWhilePreviewRuns)
            CaptureCurrentStateIntoWindow();
    }

    public void EnsureInitialized() => ResolveRefs();

    public void GetClustersNonAlloc(List<FusedClusterState> destination)
    {
        destination.Clear();
        destination.AddRange(_referenceClusters);
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
        StateRevision++;
        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverFusedPointCloudManager] integrate window={_windowSnapshots.Count}, visible={incoming.visibleCount}, added={added}, replaced={replaced}, rejected={rejected}, fusedCount={_referenceClusters.Count}");
        return true;
    }

    public void ClearAll()
    {
        _referenceClusters.Clear();
        _windowSnapshots.Clear();
        _nextAccumulationSampleTime = 0f;
        _lastCapturedFrameIndex = -1;
        _nextClusterId = 0;
        _currentIntegrationBatch = 0;
        LastIssue = null;
        LastOverlapRatio = 0f;
        LastNovelRatio = 0f;
        LastOverlapPointCount = 0;
        LastNovelPointCount = 0;
        StateRevision++;
    }

    public static float ResolveClusterScore(FusedClusterState cluster)
    {
        float sampleTerm = Mathf.Min(1f, Mathf.Max(1f, cluster.sampleCount) / 6f) * 0.25f;
        return Mathf.Clamp01(cluster.confidence) + sampleTerm;
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
                _referenceClusters.Add(new FusedClusterState
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

            FusedClusterState cluster = _referenceClusters[clusterIndex];
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

            _referenceClusters.Add(new FusedClusterState
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

        if (!enforceNarrowBandFreezeGate || !HasFusedState || _referenceClusters.Count <= 0)
            return true;

        if (incoming == null || incoming.entries == null || incoming.entries.Length <= 0 || incoming.visibleCount <= 0)
            return SetIssue("Freeze rejected: incoming grid state is empty.");

        Vector3 observationOrigin = depthGridPointCloud != null ? depthGridPointCloud.CurrentObservationOrigin : transform.position;
        int visibleCount = 0;
        int overlapCount = 0;
        int novelCount = 0;

        for (int i = 0; i < incoming.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry candidate = incoming.entries[i];
            if (!candidate.valid)
                continue;

            visibleCount++;
            Vector3 candidateObservationDir = ResolveObservationDirection(observationOrigin, candidate.worldPos, candidate.normal);
            int clusterIndex = FindBestClusterIndex(candidate, candidateObservationDir);
            if (clusterIndex >= 0)
                overlapCount++;
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

    private int FindBestClusterIndex(ScanCoverDepthGridPointCloud.GridStateEntry candidate, Vector3 candidateObservationDir)
    {
        Vector3 candidateNormal = candidate.normal.sqrMagnitude > 1e-6f ? candidate.normal.normalized : Vector3.up;
        float searchRadius = Mathf.Max(duplicateDistanceMeters, outwardReplaceDistanceMeters, insideRejectDistanceMeters);
        float bestDistance = float.PositiveInfinity;
        int bestIndex = -1;

        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            FusedClusterState cluster = _referenceClusters[i];
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

            FusedClusterState a = _referenceClusters[i];
            Vector3 normalA = a.normal.sqrMagnitude > 1e-6f ? a.normal.normalized : Vector3.up;

            for (int j = i + 1; j < _referenceClusters.Count; j++)
            {
                if (removed[j])
                    continue;

                FusedClusterState b = _referenceClusters[j];
                Vector3 normalB = b.normal.sqrMagnitude > 1e-6f ? b.normal.normalized : normalA;
                float normalDot = Vector3.Dot(normalA, normalB);
                if (normalDot < minNormalDot)
                    continue;

                Vector3 delta = b.worldPos - a.worldPos;
                float distance = delta.magnitude;
                if (distance > radius)
                    continue;

                Vector3 avgNormal = normalA + normalB;
                avgNormal = avgNormal.sqrMagnitude > 1e-6f ? avgNormal.normalized : normalA;
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

        List<FusedClusterState> compacted = new List<FusedClusterState>(_referenceClusters.Count);
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

            FusedClusterState a = _referenceClusters[i];
            Vector3 normalA = a.normal.sqrMagnitude > 1e-6f ? a.normal.normalized : Vector3.up;

            for (int j = i + 1; j < _referenceClusters.Count; j++)
            {
                if (removed[j])
                    continue;

                FusedClusterState b = _referenceClusters[j];
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
                avgNormal = avgNormal.sqrMagnitude > 1e-6f ? avgNormal.normalized : normalA;

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
                avgObservation = avgObservation.sqrMagnitude > 1e-6f ? avgObservation.normalized : observationA;
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

        List<FusedClusterState> thinned = new List<FusedClusterState>(_referenceClusters.Count);
        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (!removed[i])
                thinned.Add(_referenceClusters[i]);
        }

        _referenceClusters.Clear();
        _referenceClusters.AddRange(thinned);
    }

    private int ChooseShellLayerWinner(FusedClusterState a, FusedClusterState b, float signedOffset, float observationDepthOffset, float observationDepthLimit)
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
            FusedClusterState occluder = _referenceClusters[i];
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

                FusedClusterState target = _referenceClusters[j];
                if (target.batchIndex == _currentIntegrationBatch)
                    continue;

                Vector3 targetOffset = target.worldPos - observationOrigin;
                float targetDepth = targetOffset.magnitude;
                if (targetDepth <= occluderDepth + minDepthDelta)
                    continue;

                Vector3 targetRay = targetOffset / targetDepth;
                if (Vector3.Dot(occluderRay, targetRay) < minRayDot)
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

        List<FusedClusterState> visible = new List<FusedClusterState>(_referenceClusters.Count);
        for (int i = 0; i < _referenceClusters.Count; i++)
        {
            if (!removed[i])
                visible.Add(_referenceClusters[i]);
        }

        _referenceClusters.Clear();
        _referenceClusters.AddRange(visible);
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
            ScanCoverDepthGridPointCloud.GridStateEntry result = latest.entries[i];
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
        if (Vector3.Dot(representativeNormal, candidateNormal) < minNormalDot)
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

    private void ResolveRefs()
    {
        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (tsdfBranch == null)
            tsdfBranch = GetComponent<ScanCoverTsdfBranch>();
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

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog && !string.IsNullOrEmpty(issue))
            Debug.LogWarning($"[ScanCoverFusedPointCloudManager] {issue}");
        return false;
    }
}

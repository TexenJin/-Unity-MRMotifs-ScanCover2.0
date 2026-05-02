using System;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-43)]
[DisallowMultipleComponent]
public sealed class ScanCoverStableObservationCloudAccumulator : MonoBehaviour
{
    [Serializable]
    private struct StableNode
    {
        public bool occupied;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public float confidence;
        public float linearDepth;
        public float lastSeenTime;
        public int consecutiveHits;
        public int frameIndex;
        public ScanCoverDepthObservationGridProvider.ObservationSupportLayer supportLayer;
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverDepthObservationGridProvider provider;
    [SerializeField] private ScanCoverBinocularFusedObservationProvider fusedProvider;

    [Header("Grid")]
    [SerializeField, Min(0.001f)] private float cellSizeMeters = 0.05f;
    [SerializeField] private bool useProviderFrameIndex = true;

    [Header("Stability")]
    [SerializeField, Range(0f, 1f)] private float positionBlend = 0.3f;
    [SerializeField, Range(0f, 1f)] private float normalBlend = 0.3f;
    [SerializeField, Range(0f, 1f)] private float confidenceBlend = 0.35f;
    [SerializeField, Min(1)] private int minConsecutiveHits = 2;
    [SerializeField, Min(0f)] private float holdMissingSeconds = 0.35f;
    [SerializeField, Min(0f)] private float minConfidence = 0.2f;

    [Header("Consolidation")]
    [SerializeField] private bool enableNeighborConsolidation = true;
    [SerializeField, Min(0.001f)] private float consolidationRadiusMeters = 0.06f;
    [SerializeField, Range(-1f, 1f)] private float consolidationMinNormalDot = 0.65f;

    [Header("Debug")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField] private bool debugLog;

    public IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> CurrentObservations => _consolidatedObservations;
    public int StableCount => _consolidatedObservations.Count;
    public string LastIssue { get; private set; }

    private readonly Dictionary<Vector3Int, StableNode> _nodes = new Dictionary<Vector3Int, StableNode>(4096);
    private readonly List<Vector3Int> _keys = new List<Vector3Int>(4096);
    private readonly List<ScanCoverDepthObservationGridProvider.Observation> _stableObservations =
        new List<ScanCoverDepthObservationGridProvider.Observation>(4096);
    private readonly List<ScanCoverDepthObservationGridProvider.Observation> _consolidatedObservations =
        new List<ScanCoverDepthObservationGridProvider.Observation>(4096);
    private int _lastRefreshFrame = -1;

    private void Awake()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();
    }

    private void Update()
    {
        if (!updateEveryFrame)
            return;

        RefreshNow();
    }

    [ContextMenu("Refresh Stable Observation Cloud")]
    public bool RefreshNow()
    {
        if (_lastRefreshFrame == Time.frameCount)
            return _consolidatedObservations.Count > 0;

        ResolveRefs();
        if (provider == null && fusedProvider == null)
            return SetIssue("Observation source is missing.");

        if (provider != null && !provider.HasPendingReadback)
            provider.RefreshNow();
        if (fusedProvider != null)
            fusedProvider.RefreshNow();

        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> observations =
            fusedProvider != null ? fusedProvider.CurrentObservations : provider.CurrentObservations;
        if (observations == null || observations.Count == 0)
        {
            int emptyFrameIndex = useProviderFrameIndex
                ? (fusedProvider != null ? fusedProvider.ObservationFrameIndex : provider.ObservationFrameIndex)
                : Time.frameCount;
            RebuildStableObservationList(Time.time, emptyFrameIndex);
            return SetIssue("Observation list is empty.");
        }

        float now = Time.time;
        int frameIndex = useProviderFrameIndex
            ? (fusedProvider != null ? fusedProvider.ObservationFrameIndex : provider.ObservationFrameIndex)
            : Time.frameCount;
        float cellSize = Mathf.Max(0.001f, cellSizeMeters);

        for (int i = 0; i < observations.Count; i++)
        {
            var observation = observations[i];
            if (!observation.valid || observation.confidence < minConfidence)
                continue;

            Vector3Int key = new Vector3Int(
                Mathf.RoundToInt(observation.worldPos.x / cellSize),
                Mathf.RoundToInt(observation.worldPos.y / cellSize),
                Mathf.RoundToInt(observation.worldPos.z / cellSize));

            if (_nodes.TryGetValue(key, out StableNode node))
            {
                node.occupied = true;
                node.worldPos = Vector3.Lerp(node.worldPos, observation.worldPos, positionBlend);

                Vector3 oldNormal = node.worldNormal.sqrMagnitude > 1e-6f ? node.worldNormal.normalized : Vector3.up;
                Vector3 newNormal = observation.worldNormal.sqrMagnitude > 1e-6f ? observation.worldNormal.normalized : oldNormal;
                Vector3 blendedNormal = Vector3.Lerp(oldNormal, newNormal, normalBlend);
                node.worldNormal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : newNormal;

                node.confidence = Mathf.Lerp(node.confidence, observation.confidence, confidenceBlend);
                node.linearDepth = Mathf.Lerp(node.linearDepth, observation.linearDepth, positionBlend);
                node.lastSeenTime = now;
                node.consecutiveHits += 1;
                node.frameIndex = frameIndex;
                node.supportLayer = observation.supportLayer;
                _nodes[key] = node;
            }
            else
            {
                _nodes.Add(key, new StableNode
                {
                    occupied = true,
                    worldPos = observation.worldPos,
                    worldNormal = observation.worldNormal.sqrMagnitude > 1e-6f ? observation.worldNormal.normalized : Vector3.up,
                    confidence = observation.confidence,
                    linearDepth = observation.linearDepth,
                    lastSeenTime = now,
                    consecutiveHits = 1,
                    frameIndex = frameIndex,
                    supportLayer = observation.supportLayer,
                });
            }
        }

        RebuildStableObservationList(now, frameIndex);
        ConsolidateStableObservations();
        _lastRefreshFrame = Time.frameCount;
        LastIssue = null;

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverStableObservationCloudAccumulator] stable={_stableObservations.Count}, " +
                $"consolidated={_consolidatedObservations.Count}, " +
                $"nodes={_nodes.Count}, frame={frameIndex}");
        }

        return _consolidatedObservations.Count > 0;
    }

    private void RebuildStableObservationList(float now, int frameIndex)
    {
        _stableObservations.Clear();
        _keys.Clear();
        foreach (Vector3Int key in _nodes.Keys)
            _keys.Add(key);

        for (int i = 0; i < _keys.Count; i++)
        {
            Vector3Int key = _keys[i];
            StableNode node = _nodes[key];
            if (now - node.lastSeenTime > holdMissingSeconds)
            {
                _nodes.Remove(key);
                continue;
            }

            if (node.consecutiveHits < minConsecutiveHits)
                continue;

            if (node.confidence < minConfidence)
                continue;

            _stableObservations.Add(new ScanCoverDepthObservationGridProvider.Observation
            {
                valid = true,
                worldPos = node.worldPos,
                worldNormal = node.worldNormal,
                linearDepth = node.linearDepth,
                confidence = node.confidence,
                frameIndex = frameIndex,
                sourcePixel = new Vector2Int(key.x, key.y),
                supportLayer = node.supportLayer,
            });
        }
    }

    private void ConsolidateStableObservations()
    {
        _consolidatedObservations.Clear();

        if (!enableNeighborConsolidation)
        {
            _consolidatedObservations.AddRange(_stableObservations);
            return;
        }

        float radius = Mathf.Max(0.001f, consolidationRadiusMeters);
        float radiusSqr = radius * radius;
        bool[] consumed = new bool[_stableObservations.Count];

        for (int i = 0; i < _stableObservations.Count; i++)
        {
            if (consumed[i])
                continue;

            var seed = _stableObservations[i];
            Vector3 accumPos = seed.worldPos * Mathf.Max(0.001f, seed.confidence);
            Vector3 accumNormal = seed.worldNormal.normalized * Mathf.Max(0.001f, seed.confidence);
            float accumConfidence = Mathf.Max(0.001f, seed.confidence);
            float bestConfidence = seed.confidence;
            float minDepth = seed.linearDepth;
            int frameIndex = seed.frameIndex;
            Vector2Int sourcePixel = seed.sourcePixel;
            var supportLayer = seed.supportLayer;

            consumed[i] = true;

            for (int j = i + 1; j < _stableObservations.Count; j++)
            {
                if (consumed[j])
                    continue;

                var candidate = _stableObservations[j];
                if ((candidate.worldPos - seed.worldPos).sqrMagnitude > radiusSqr)
                    continue;

                Vector3 seedNormal = seed.worldNormal.sqrMagnitude > 1e-6f ? seed.worldNormal.normalized : Vector3.up;
                Vector3 candidateNormal = candidate.worldNormal.sqrMagnitude > 1e-6f ? candidate.worldNormal.normalized : seedNormal;
                if (Vector3.Dot(seedNormal, candidateNormal) < consolidationMinNormalDot)
                    continue;

                float weight = Mathf.Max(0.001f, candidate.confidence);
                accumPos += candidate.worldPos * weight;
                accumNormal += candidateNormal * weight;
                accumConfidence += weight;
                minDepth = Mathf.Min(minDepth, candidate.linearDepth);
                if (candidate.confidence > bestConfidence)
                {
                    bestConfidence = candidate.confidence;
                    sourcePixel = candidate.sourcePixel;
                    supportLayer = candidate.supportLayer;
                }

                consumed[j] = true;
            }

            Vector3 mergedNormal = accumNormal.sqrMagnitude > 1e-6f ? accumNormal.normalized : seed.worldNormal;
            _consolidatedObservations.Add(new ScanCoverDepthObservationGridProvider.Observation
            {
                valid = true,
                worldPos = accumPos / accumConfidence,
                worldNormal = mergedNormal,
                linearDepth = minDepth,
                confidence = Mathf.Clamp01(bestConfidence),
                frameIndex = frameIndex,
                sourcePixel = sourcePixel,
                supportLayer = supportLayer,
            });
        }
    }

    private void ResolveRefs()
    {
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverDepthObservationGridProvider>();
        if (fusedProvider == null)
            fusedProvider = FindAnyObjectByType<ScanCoverBinocularFusedObservationProvider>();
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverStableObservationCloudAccumulator] {issue}");
        return false;
    }
}

using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-43)]
[DisallowMultipleComponent]
public sealed class ScanCoverBinocularFusedObservationProvider : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverDepthObservationGridProvider leftProvider;
    [SerializeField] private ScanCoverDepthObservationGridProvider rightProvider;

    [Header("Pairing")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(0.001f)] private float pairMaxDistanceMeters = 0.04f;
    [SerializeField, Min(0f)] private float pairMaxDepthDeltaMeters = 0.03f;
    [SerializeField, Range(-1f, 1f)] private float pairMinNormalDot = 0.75f;
    [SerializeField] private bool requireMutualBest = true;
    [SerializeField, Min(0.1f)] private float maxCameraDistanceMeters = 8f;

    [Header("Fallback")]
    [SerializeField] private bool allowMonocularFallback = true;
    [SerializeField, Range(0f, 1f)] private float monocularFallbackConfidenceScale = 0.45f;
    [SerializeField, Range(0f, 1f)] private float centerStrictRegionNormalized = 0.6f;
    [SerializeField] private bool dropUnpairedInsideCenterRegion = true;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> CurrentObservations => _currentObservations;
    public int ObservationFrameIndex { get; private set; }
    public string LastIssue { get; private set; }

    private readonly List<ScanCoverDepthObservationGridProvider.Observation> _currentObservations =
        new List<ScanCoverDepthObservationGridProvider.Observation>(4096);
    private readonly List<int> _leftBest = new List<int>(4096);
    private readonly List<int> _rightBest = new List<int>(4096);
    private readonly List<bool> _leftUsed = new List<bool>(4096);
    private readonly List<bool> _rightUsed = new List<bool>(4096);
    private readonly Dictionary<Vector3Int, List<int>> _leftBuckets = new Dictionary<Vector3Int, List<int>>(2048);
    private readonly Dictionary<Vector3Int, List<int>> _rightBuckets = new Dictionary<Vector3Int, List<int>>(2048);
    private readonly List<Vector3Int> _neighborCells = new List<Vector3Int>(27);
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

    [ContextMenu("Refresh Binocular Fused Observations")]
    public bool RefreshNow()
    {
        if (_lastRefreshFrame == Time.frameCount)
            return _currentObservations.Count > 0;

        ResolveRefs();
        if (leftProvider == null || rightProvider == null)
            return SetIssue("Left or right provider is missing.");

        if (!leftProvider.HasPendingReadback)
            leftProvider.RefreshNow();
        if (!rightProvider.HasPendingReadback)
            rightProvider.RefreshNow();

        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> left = leftProvider.CurrentObservations;
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> right = rightProvider.CurrentObservations;
        if (left == null || right == null || left.Count == 0 || right.Count == 0)
        {
            _currentObservations.Clear();
            return SetIssue("Left or right observations are empty.");
        }

        BuildSpatialBuckets(left, _leftBuckets);
        BuildSpatialBuckets(right, _rightBuckets);
        BuildBestMatchLists(left, right);
        BuildFusedObservations(left, right);
        ObservationFrameIndex++;
        _lastRefreshFrame = Time.frameCount;
        LastIssue = null;

        if (debugLog)
        {
            int confirmed = 0;
            int fallback = 0;
            for (int i = 0; i < _currentObservations.Count; i++)
            {
                switch (_currentObservations[i].supportLayer)
                {
                    case ScanCoverDepthObservationGridProvider.ObservationSupportLayer.BinocularConfirmed:
                        confirmed++;
                        break;
                    case ScanCoverDepthObservationGridProvider.ObservationSupportLayer.MonocularFallback:
                        fallback++;
                        break;
                }
            }

            Debug.Log(
                $"[ScanCoverBinocularFusedObservationProvider] fused={_currentObservations.Count}, " +
                $"confirmed={confirmed}, fallback={fallback}");
        }

        return _currentObservations.Count > 0;
    }

    private void BuildBestMatchLists(
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> left,
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> right)
    {
        EnsureIntListSize(_leftBest, left.Count, -1);
        EnsureIntListSize(_rightBest, right.Count, -1);
        EnsureBoolListSize(_leftUsed, left.Count, false);
        EnsureBoolListSize(_rightUsed, right.Count, false);

        for (int i = 0; i < left.Count; i++)
        {
            _leftBest[i] = FindBestCandidate(left[i], right, _rightBuckets);
            _leftUsed[i] = false;
        }

        for (int i = 0; i < right.Count; i++)
        {
            _rightBest[i] = FindBestCandidate(right[i], left, _leftBuckets);
            _rightUsed[i] = false;
        }
    }

    private void BuildFusedObservations(
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> left,
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> right)
    {
        _currentObservations.Clear();

        for (int i = 0; i < left.Count; i++)
        {
            int candidateIndex = _leftBest[i];
            if (candidateIndex < 0 || candidateIndex >= right.Count)
                continue;

            if (requireMutualBest && _rightBest[candidateIndex] != i)
                continue;

            if (_leftUsed[i] || _rightUsed[candidateIndex])
                continue;

            _leftUsed[i] = true;
            _rightUsed[candidateIndex] = true;
            _currentObservations.Add(FusePair(left[i], right[candidateIndex]));
        }

        if (!allowMonocularFallback)
            return;

        for (int i = 0; i < left.Count; i++)
        {
            if (_leftUsed[i])
                continue;
            TryAddFallback(left[i], leftProvider.CurrentResolution);
        }

        for (int i = 0; i < right.Count; i++)
        {
            if (_rightUsed[i])
                continue;
            TryAddFallback(right[i], rightProvider.CurrentResolution);
        }
    }

    private int FindBestCandidate(
        ScanCoverDepthObservationGridProvider.Observation source,
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> candidates,
        Dictionary<Vector3Int, List<int>> candidateBuckets)
    {
        if (!IsUsableObservation(source))
            return -1;

        int bestIndex = -1;
        float bestScore = float.PositiveInfinity;
        float maxDistanceSqr = pairMaxDistanceMeters * pairMaxDistanceMeters;
        Vector3 sourceNormal = source.worldNormal.sqrMagnitude > 1e-6f ? source.worldNormal.normalized : Vector3.up;
        float bucketSize = Mathf.Max(0.001f, pairMaxDistanceMeters);
        Vector3Int baseKey = Quantize(source.worldPos, bucketSize);
        FillNeighborCells(baseKey);

        for (int c = 0; c < _neighborCells.Count; c++)
        {
            if (!candidateBuckets.TryGetValue(_neighborCells[c], out List<int> bucket))
                continue;

            for (int bi = 0; bi < bucket.Count; bi++)
            {
                int i = bucket[bi];
                var candidate = candidates[i];
                if (!IsUsableObservation(candidate))
                    continue;

                float distanceSqr = (candidate.worldPos - source.worldPos).sqrMagnitude;
                if (distanceSqr > maxDistanceSqr)
                    continue;

                if (Mathf.Abs(candidate.linearDepth - source.linearDepth) > pairMaxDepthDeltaMeters)
                    continue;

                Vector3 candidateNormal = candidate.worldNormal.sqrMagnitude > 1e-6f ? candidate.worldNormal.normalized : sourceNormal;
                float normalDot = Vector3.Dot(sourceNormal, candidateNormal);
                if (normalDot < pairMinNormalDot)
                    continue;

                float score = distanceSqr + (1f - normalDot) * 0.01f;
                if (score < bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }
        }

        return bestIndex;
    }

    private ScanCoverDepthObservationGridProvider.Observation FusePair(
        ScanCoverDepthObservationGridProvider.Observation left,
        ScanCoverDepthObservationGridProvider.Observation right)
    {
        float leftWeight = Mathf.Max(0.001f, left.confidence);
        float rightWeight = Mathf.Max(0.001f, right.confidence);
        float totalWeight = leftWeight + rightWeight;

        Vector3 fusedPos = (left.worldPos * leftWeight + right.worldPos * rightWeight) / totalWeight;
        Vector3 fusedNormal = left.worldNormal * leftWeight + right.worldNormal * rightWeight;
        if (fusedNormal.sqrMagnitude > 1e-8f)
            fusedNormal.Normalize();

        return new ScanCoverDepthObservationGridProvider.Observation
        {
            valid = true,
            worldPos = fusedPos,
            worldNormal = fusedNormal.sqrMagnitude > 1e-8f ? fusedNormal : Vector3.up,
            linearDepth = Mathf.Min(left.linearDepth, right.linearDepth),
            confidence = Mathf.Clamp01(((left.confidence + right.confidence) * 0.5f) * 1.1f),
            frameIndex = Mathf.Max(left.frameIndex, right.frameIndex),
            sourcePixel = left.confidence >= right.confidence ? left.sourcePixel : right.sourcePixel,
            supportLayer = ScanCoverDepthObservationGridProvider.ObservationSupportLayer.BinocularConfirmed,
        };
    }

    private void TryAddFallback(
        ScanCoverDepthObservationGridProvider.Observation observation,
        Vector2Int resolution)
    {
        if (!IsUsableObservation(observation))
            return;

        if (dropUnpairedInsideCenterRegion && IsInsideCenterStrictRegion(observation.sourcePixel, resolution))
            return;

        observation.confidence = Mathf.Clamp01(observation.confidence * monocularFallbackConfidenceScale);
        observation.supportLayer = ScanCoverDepthObservationGridProvider.ObservationSupportLayer.MonocularFallback;
        _currentObservations.Add(observation);
    }

    private bool IsInsideCenterStrictRegion(Vector2Int pixel, Vector2Int resolution)
    {
        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        float nx = (pixel.x + 0.5f) / width;
        float ny = (pixel.y + 0.5f) / height;
        float half = Mathf.Clamp01(centerStrictRegionNormalized) * 0.5f;
        return Mathf.Abs(nx - 0.5f) <= half && Mathf.Abs(ny - 0.5f) <= half;
    }

    private void ResolveRefs()
    {
        if (leftProvider == null || rightProvider == null)
        {
            ScanCoverDepthObservationGridProvider[] providers =
                FindObjectsByType<ScanCoverDepthObservationGridProvider>(FindObjectsSortMode.None);
            for (int i = 0; i < providers.Length; i++)
            {
                ScanCoverDepthObservationGridProvider candidate = providers[i];
                if (candidate == null)
                    continue;

                if (leftProvider == null)
                {
                    leftProvider = candidate;
                    continue;
                }

                if (rightProvider == null && candidate != leftProvider)
                {
                    rightProvider = candidate;
                    break;
                }
            }
        }
    }

    private void BuildSpatialBuckets(
        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> observations,
        Dictionary<Vector3Int, List<int>> buckets)
    {
        foreach (List<int> bucket in buckets.Values)
            bucket.Clear();

        float bucketSize = Mathf.Max(0.001f, pairMaxDistanceMeters);
        for (int i = 0; i < observations.Count; i++)
        {
            if (!IsUsableObservation(observations[i]))
                continue;

            Vector3Int key = Quantize(observations[i].worldPos, bucketSize);
            if (!buckets.TryGetValue(key, out List<int> bucket))
            {
                bucket = new List<int>(8);
                buckets.Add(key, bucket);
            }
            bucket.Add(i);
        }
    }

    private void FillNeighborCells(Vector3Int center)
    {
        _neighborCells.Clear();
        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                    _neighborCells.Add(new Vector3Int(center.x + x, center.y + y, center.z + z));
            }
        }
    }

    private bool IsUsableObservation(ScanCoverDepthObservationGridProvider.Observation observation)
    {
        if (!observation.valid)
            return false;
        if (!IsFinite(observation.worldPos))
            return false;

        Camera mainCamera = Camera.main;
        if (mainCamera != null && Vector3.Distance(mainCamera.transform.position, observation.worldPos) > maxCameraDistanceMeters)
            return false;

        return true;
    }

    private static Vector3Int Quantize(Vector3 worldPos, float cellSize)
    {
        return new Vector3Int(
            Mathf.RoundToInt(worldPos.x / cellSize),
            Mathf.RoundToInt(worldPos.y / cellSize),
            Mathf.RoundToInt(worldPos.z / cellSize));
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverBinocularFusedObservationProvider] {issue}");
        return false;
    }

    private static void EnsureIntListSize(List<int> list, int count, int fillValue)
    {
        while (list.Count < count)
            list.Add(fillValue);
        while (list.Count > count)
            list.RemoveAt(list.Count - 1);
        for (int i = 0; i < list.Count; i++)
            list[i] = fillValue;
    }

    private static void EnsureBoolListSize(List<bool> list, int count, bool fillValue)
    {
        while (list.Count < count)
            list.Add(fillValue);
        while (list.Count > count)
            list.RemoveAt(list.Count - 1);
        for (int i = 0; i < list.Count; i++)
            list[i] = fillValue;
    }

    private static bool IsFinite(Vector3 value)
    {
        return
            !float.IsNaN(value.x) && !float.IsNaN(value.y) && !float.IsNaN(value.z) &&
            !float.IsInfinity(value.x) && !float.IsInfinity(value.y) && !float.IsInfinity(value.z);
    }
}

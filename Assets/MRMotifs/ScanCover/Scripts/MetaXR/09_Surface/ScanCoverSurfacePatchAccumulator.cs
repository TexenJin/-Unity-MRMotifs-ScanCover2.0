using System;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-43)]
[DisallowMultipleComponent]
public sealed class ScanCoverSurfacePatchAccumulator : MonoBehaviour
{
    [Serializable]
    private struct CachedPatch
    {
        public bool occupied;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public Quaternion rotation;
        public Vector2 sizeMeters;
        public float confidence;
        public float lastSeenTime;
        public int frameIndex;
        public Vector2Int tileCoord;
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverSurfacePatchCandidateProvider provider;

    [Header("Grouping")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(0.01f)] private float positionCellMeters = 0.12f;
    [SerializeField, Range(-1f, 1f)] private float mergeMinNormalDot = 0.88f;
    [SerializeField, Min(0.001f)] private float mergeMaxDistanceMeters = 0.12f;

    [Header("Temporal")]
    [SerializeField, Range(0f, 1f)] private float positionBlend = 0.35f;
    [SerializeField, Range(0f, 1f)] private float normalBlend = 0.35f;
    [SerializeField, Range(0f, 1f)] private float sizeBlend = 0.3f;
    [SerializeField, Range(0f, 1f)] private float confidenceBlend = 0.4f;
    [SerializeField, Min(0f)] private float holdMissingSeconds = 0.35f;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0.25f;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public IReadOnlyList<ScanCoverSurfacePatchCandidateProvider.PatchCandidate> CurrentPatches => _currentPatches;
    public string LastIssue { get; private set; }

    private readonly Dictionary<PatchKey, CachedPatch> _patches = new Dictionary<PatchKey, CachedPatch>(512);
    private readonly List<PatchKey> _keys = new List<PatchKey>(512);
    private readonly List<ScanCoverSurfacePatchCandidateProvider.PatchCandidate> _currentPatches =
        new List<ScanCoverSurfacePatchCandidateProvider.PatchCandidate>(512);
    private int _lastRefreshFrame = -1;

    private struct PatchKey : IEquatable<PatchKey>
    {
        public Vector3Int positionCell;
        public Vector3Int normalBucket;

        public bool Equals(PatchKey other)
        {
            return positionCell.Equals(other.positionCell) && normalBucket.Equals(other.normalBucket);
        }

        public override bool Equals(object obj)
        {
            return obj is PatchKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (positionCell.GetHashCode() * 397) ^ normalBucket.GetHashCode();
            }
        }
    }

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

    [ContextMenu("Refresh Surface Patch Accumulator")]
    public bool RefreshNow()
    {
        if (_lastRefreshFrame == Time.frameCount)
            return _currentPatches.Count > 0;

        ResolveRefs();
        if (provider == null)
            return SetIssue("ScanCoverSurfacePatchCandidateProvider is missing.");

        if (provider.HasPendingReadback)
        {
            LastIssue = "Waiting for patch candidate readback.";
            return false;
        }

        IReadOnlyList<ScanCoverSurfacePatchCandidateProvider.PatchCandidate> patches = provider.CurrentPatches;
        float now = Time.time;
        if (patches == null || patches.Count == 0)
        {
            RebuildCurrentPatches(now);
            _lastRefreshFrame = Time.frameCount;
            return SetIssue("Patch candidate list is empty.");
        }

        for (int i = 0; i < patches.Count; i++)
        {
            ScanCoverSurfacePatchCandidateProvider.PatchCandidate patch = patches[i];
            if (!patch.valid || patch.confidence < minConfidence)
                continue;

            PatchKey key = BuildKey(patch);
            if (_patches.TryGetValue(key, out CachedPatch cached))
            {
                Vector3 oldNormal = cached.worldNormal.sqrMagnitude > 1e-6f ? cached.worldNormal.normalized : Vector3.up;
                Vector3 newNormal = patch.worldNormal.sqrMagnitude > 1e-6f ? patch.worldNormal.normalized : oldNormal;
                if (Vector3.Dot(oldNormal, newNormal) < mergeMinNormalDot)
                    continue;

                float distance = Vector3.Distance(cached.worldPos, patch.worldPos);
                if (distance > mergeMaxDistanceMeters)
                    continue;

                cached.worldPos = Vector3.Lerp(cached.worldPos, patch.worldPos, positionBlend);
                Vector3 blendedNormal = Vector3.Lerp(oldNormal, newNormal, normalBlend);
                cached.worldNormal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : newNormal;
                cached.rotation = Quaternion.Slerp(cached.rotation, patch.rotation, normalBlend);
                cached.sizeMeters = Vector2.Lerp(cached.sizeMeters, patch.sizeMeters, sizeBlend);
                cached.confidence = Mathf.Lerp(cached.confidence, patch.confidence, confidenceBlend);
                cached.lastSeenTime = now;
                cached.frameIndex = provider.PatchFrameIndex;
                cached.tileCoord = patch.tileCoord;
                cached.occupied = true;
                _patches[key] = cached;
            }
            else
            {
                _patches.Add(key, new CachedPatch
                {
                    occupied = true,
                    worldPos = patch.worldPos,
                    worldNormal = patch.worldNormal.sqrMagnitude > 1e-6f ? patch.worldNormal.normalized : Vector3.up,
                    rotation = patch.rotation,
                    sizeMeters = patch.sizeMeters,
                    confidence = patch.confidence,
                    lastSeenTime = now,
                    frameIndex = provider.PatchFrameIndex,
                    tileCoord = patch.tileCoord,
                });
            }
        }

        RebuildCurrentPatches(now);
        _lastRefreshFrame = Time.frameCount;
        LastIssue = null;

        if (!provider.HasPendingReadback)
            provider.RefreshNow();

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverSurfacePatchAccumulator] stable={_currentPatches.Count}, cached={_patches.Count}, frame={provider.PatchFrameIndex}");
        }

        return _currentPatches.Count > 0;
    }

    private void RebuildCurrentPatches(float now)
    {
        _currentPatches.Clear();
        _keys.Clear();
        foreach (PatchKey key in _patches.Keys)
            _keys.Add(key);

        for (int i = 0; i < _keys.Count; i++)
        {
            PatchKey key = _keys[i];
            CachedPatch cached = _patches[key];
            if (now - cached.lastSeenTime > holdMissingSeconds)
            {
                _patches.Remove(key);
                continue;
            }

            if (cached.confidence < minConfidence)
                continue;

            _currentPatches.Add(new ScanCoverSurfacePatchCandidateProvider.PatchCandidate
            {
                valid = true,
                worldPos = cached.worldPos,
                worldNormal = cached.worldNormal,
                rotation = cached.rotation,
                sizeMeters = cached.sizeMeters,
                confidence = cached.confidence,
                sampleCount = 1,
                tileCoord = cached.tileCoord,
            });
        }
    }

    private PatchKey BuildKey(ScanCoverSurfacePatchCandidateProvider.PatchCandidate patch)
    {
        float cell = Mathf.Max(0.01f, positionCellMeters);
        Vector3 n = patch.worldNormal.sqrMagnitude > 1e-6f ? patch.worldNormal.normalized : Vector3.up;
        return new PatchKey
        {
            positionCell = new Vector3Int(
                Mathf.RoundToInt(patch.worldPos.x / cell),
                Mathf.RoundToInt(patch.worldPos.y / cell),
                Mathf.RoundToInt(patch.worldPos.z / cell)),
            normalBucket = new Vector3Int(
                Mathf.RoundToInt(n.x * 2f),
                Mathf.RoundToInt(n.y * 2f),
                Mathf.RoundToInt(n.z * 2f)),
        };
    }

    private void ResolveRefs()
    {
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverSurfacePatchCandidateProvider>();
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverSurfacePatchAccumulator] {issue}");
        return false;
    }
}

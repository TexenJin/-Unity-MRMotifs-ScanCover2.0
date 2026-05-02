using System;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-42)]
[DisallowMultipleComponent]
public sealed class ScanCoverPatchLocalLatticeProvider : MonoBehaviour
{
    [Serializable]
    public struct LatticeNode
    {
        public bool valid;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public float confidence;
        public int patchIndex;
        public Vector2Int localCoord;
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverSurfacePatchAccumulator accumulator;
    [SerializeField] private ScanCoverSurfacePatchCandidateProvider fallbackProvider;

    [Header("Lattice")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(0.005f)] private float nodeSpacingMeters = 0.035f;
    [SerializeField, Min(0f)] private float edgeInsetMeters = 0.01f;
    [SerializeField, Range(0f, 1f)] private float minPatchConfidence = 0.25f;
    [SerializeField, Min(1)] private int maxNodesPerAxis = 16;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public IReadOnlyList<LatticeNode> CurrentNodes => _currentNodes;
    public string LastIssue { get; private set; }

    private readonly List<LatticeNode> _currentNodes = new List<LatticeNode>(4096);
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

    [ContextMenu("Refresh Patch Local Lattice")]
    public bool RefreshNow()
    {
        if (_lastRefreshFrame == Time.frameCount)
            return _currentNodes.Count > 0;

        ResolveRefs();
        if (accumulator == null && fallbackProvider == null)
            return SetIssue("Patch source is missing.");

        if (accumulator != null)
            accumulator.RefreshNow();

        IReadOnlyList<ScanCoverSurfacePatchCandidateProvider.PatchCandidate> patches =
            accumulator != null ? accumulator.CurrentPatches : fallbackProvider.CurrentPatches;

        _currentNodes.Clear();
        if (patches == null || patches.Count == 0)
        {
            _lastRefreshFrame = Time.frameCount;
            return SetIssue("Patch list is empty.");
        }

        float spacing = Mathf.Max(0.005f, nodeSpacingMeters);
        float inset = Mathf.Max(0f, edgeInsetMeters);
        int maxAxis = Mathf.Max(1, maxNodesPerAxis);

        for (int patchIndex = 0; patchIndex < patches.Count; patchIndex++)
        {
            var patch = patches[patchIndex];
            if (!patch.valid || patch.confidence < minPatchConfidence)
                continue;

            float usableWidth = Mathf.Max(0.001f, patch.sizeMeters.x - inset * 2f);
            float usableHeight = Mathf.Max(0.001f, patch.sizeMeters.y - inset * 2f);
            int countX = Mathf.Clamp(Mathf.Max(1, Mathf.RoundToInt(usableWidth / spacing)), 1, maxAxis);
            int countY = Mathf.Clamp(Mathf.Max(1, Mathf.RoundToInt(usableHeight / spacing)), 1, maxAxis);

            Vector3 axisU = patch.rotation * Vector3.right;
            Vector3 axisV = patch.rotation * Vector3.forward;
            Vector3 normal = patch.worldNormal.sqrMagnitude > 1e-6f ? patch.worldNormal.normalized : Vector3.up;

            float spanX = countX <= 1 ? 0f : usableWidth;
            float spanY = countY <= 1 ? 0f : usableHeight;

            for (int y = 0; y < countY; y++)
            {
                float ty = countY <= 1 ? 0.5f : y / (float)(countY - 1);
                float offsetV = (ty - 0.5f) * spanY;

                for (int x = 0; x < countX; x++)
                {
                    float tx = countX <= 1 ? 0.5f : x / (float)(countX - 1);
                    float offsetU = (tx - 0.5f) * spanX;

                    Vector3 pos = patch.worldPos + axisU * offsetU + axisV * offsetV;
                    _currentNodes.Add(new LatticeNode
                    {
                        valid = true,
                        worldPos = pos,
                        worldNormal = normal,
                        confidence = patch.confidence,
                        patchIndex = patchIndex,
                        localCoord = new Vector2Int(x, y),
                    });
                }
            }
        }

        _lastRefreshFrame = Time.frameCount;
        LastIssue = null;

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverPatchLocalLatticeProvider] nodes={_currentNodes.Count}, patches={patches.Count}, spacing={spacing:0.###}");
        }

        return _currentNodes.Count > 0;
    }

    private void ResolveRefs()
    {
        if (accumulator == null)
            accumulator = FindAnyObjectByType<ScanCoverSurfacePatchAccumulator>();
        if (fallbackProvider == null)
            fallbackProvider = FindAnyObjectByType<ScanCoverSurfacePatchCandidateProvider>();
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverPatchLocalLatticeProvider] {issue}");
        return false;
    }
}

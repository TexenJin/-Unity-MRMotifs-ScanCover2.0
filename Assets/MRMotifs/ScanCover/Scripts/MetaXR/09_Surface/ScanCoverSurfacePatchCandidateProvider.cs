using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-44)]
[DisallowMultipleComponent]
public sealed class ScanCoverSurfacePatchCandidateProvider : MonoBehaviour
{
    [Serializable]
    public struct PatchCandidate
    {
        public bool valid;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public Quaternion rotation;
        public Vector2 sizeMeters;
        public float confidence;
        public int sampleCount;
        public Vector2Int tileCoord;
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverDepthPreprocessor preprocessor;

    [Header("Tiling")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(2)] private int tileSizePixels = 12;
    [SerializeField, Min(1)] private int sampleStridePixels = 2;

    [Header("Acceptance")]
    [SerializeField, Range(0f, 1f)] private float minValidRatio = 0.45f;
    [SerializeField, Range(0f, 1f)] private float minMeanConfidence = 0.25f;
    [SerializeField, Range(-1f, 1f)] private float minNormalDot = 0.82f;
    [SerializeField, Min(0f)] private float maxPlaneDeviationMeters = 0.025f;
    [SerializeField, Min(0.1f)] private float maxCameraDistanceMeters = 4f;
    [SerializeField, Min(0.005f)] private float minPatchExtentMeters = 0.02f;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public IReadOnlyList<PatchCandidate> CurrentPatches => _currentPatches;
    public bool HasPendingReadback => _hasPendingReadback;
    public int PatchFrameIndex => _patchFrameIndex;
    public string LastIssue { get; private set; }

    private readonly List<PatchCandidate> _currentPatches = new List<PatchCandidate>(512);
    private AsyncGPUReadbackRequest _worldPositionRequest;
    private AsyncGPUReadbackRequest _worldNormalRequest;
    private AsyncGPUReadbackRequest _observationMetaRequest;
    private bool _hasPendingReadback;
    private Vector2Int _pendingResolution;
    private int _patchFrameIndex;

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
        if (_hasPendingReadback)
        {
            UpdatePendingReadback();
            return;
        }

        if (!updateEveryFrame)
            return;

        RefreshNow();
    }

    [ContextMenu("Refresh Surface Patch Candidates")]
    public bool RefreshNow()
    {
        ResolveRefs();
        if (preprocessor == null)
            return SetIssue("ScanCoverDepthPreprocessor is missing.");

        if (_hasPendingReadback)
            return false;

        if (!preprocessor.TryGetOutputs(
                out RenderTexture worldPositionTexture,
                out RenderTexture worldNormalTexture,
                out RenderTexture observationMetaTexture))
        {
            return SetIssue(preprocessor.LastIssue ?? "Preprocessor outputs are unavailable.");
        }

        _pendingResolution = preprocessor.OutputResolution;
        _worldPositionRequest = AsyncGPUReadback.Request(worldPositionTexture);
        _worldNormalRequest = AsyncGPUReadback.Request(worldNormalTexture);
        _observationMetaRequest = AsyncGPUReadback.Request(observationMetaTexture);
        _hasPendingReadback = true;
        LastIssue = null;
        return true;
    }

    private void UpdatePendingReadback()
    {
        if (!_worldPositionRequest.done || !_worldNormalRequest.done || !_observationMetaRequest.done)
            return;

        _hasPendingReadback = false;
        if (_worldPositionRequest.hasError || _worldNormalRequest.hasError || _observationMetaRequest.hasError)
        {
            SetIssue("AsyncGPUReadback failed.");
            return;
        }

        BuildPatchList(
            _worldPositionRequest.GetData<Color>(),
            _worldNormalRequest.GetData<Color>(),
            _observationMetaRequest.GetData<Color>(),
            _pendingResolution);

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverSurfacePatchCandidateProvider] patches={_currentPatches.Count}, " +
                $"resolution={_pendingResolution.x}x{_pendingResolution.y}, tile={tileSizePixels}, sampleStride={sampleStridePixels}");
        }
    }

    private void BuildPatchList(
        NativeArray<Color> worldPositions,
        NativeArray<Color> worldNormals,
        NativeArray<Color> observationMeta,
        Vector2Int resolution)
    {
        _currentPatches.Clear();
        _patchFrameIndex++;

        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int tile = Mathf.Max(2, tileSizePixels);
        int step = Mathf.Max(1, sampleStridePixels);
        Camera mainCamera = Camera.main;
        Vector3 cameraPos = mainCamera ? mainCamera.transform.position : Vector3.zero;

        for (int tileY = 0; tileY < height; tileY += tile)
        {
            for (int tileX = 0; tileX < width; tileX += tile)
            {
                int maxX = Mathf.Min(width, tileX + tile);
                int maxY = Mathf.Min(height, tileY + tile);
                int totalSamples = 0;
                int validSamples = 0;
                Vector3 sumPos = Vector3.zero;
                Vector3 sumNormal = Vector3.zero;
                float sumConfidence = 0f;

                for (int y = tileY; y < maxY; y += step)
                {
                    for (int x = tileX; x < maxX; x += step)
                    {
                        totalSamples++;
                        int index = x + y * width;
                        Color meta = observationMeta[index];
                        if (meta.r < 0.5f || meta.g < minMeanConfidence)
                            continue;

                        Vector3 worldPos = new Vector3(worldPositions[index].r, worldPositions[index].g, worldPositions[index].b);
                        if (!IsFinite(worldPos))
                            continue;

                        if (mainCamera != null && Vector3.Distance(cameraPos, worldPos) > maxCameraDistanceMeters)
                            continue;

                        Color normalColor = worldNormals[index];
                        if (normalColor.a < 0.5f)
                            continue;

                        Vector3 worldNormal = new Vector3(normalColor.r, normalColor.g, normalColor.b);
                        if (worldNormal.sqrMagnitude <= 1e-6f)
                            continue;

                        validSamples++;
                        sumPos += worldPos;
                        sumNormal += worldNormal.normalized;
                        sumConfidence += meta.g;
                    }
                }

                if (totalSamples <= 0 || validSamples <= 0)
                    continue;

                float validRatio = validSamples / (float)totalSamples;
                if (validRatio < minValidRatio)
                    continue;

                float meanConfidence = sumConfidence / validSamples;
                if (meanConfidence < minMeanConfidence)
                    continue;

                Vector3 centroid = sumPos / validSamples;
                Vector3 avgNormal = sumNormal.sqrMagnitude > 1e-6f ? sumNormal.normalized : Vector3.up;

                int coherentCount = 0;
                float maxPlaneDeviation = 0f;
                Vector3 tangent = Vector3.Cross(
                    Mathf.Abs(Vector3.Dot(avgNormal, Vector3.up)) > 0.9f ? Vector3.right : Vector3.up,
                    avgNormal).normalized;
                Vector3 bitangent = Vector3.Cross(avgNormal, tangent).normalized;
                float minU = float.PositiveInfinity;
                float maxU = float.NegativeInfinity;
                float minV = float.PositiveInfinity;
                float maxV = float.NegativeInfinity;

                for (int y = tileY; y < maxY; y += step)
                {
                    for (int x = tileX; x < maxX; x += step)
                    {
                        int index = x + y * width;
                        Color meta = observationMeta[index];
                        if (meta.r < 0.5f || meta.g < minMeanConfidence)
                            continue;

                        Vector3 worldPos = new Vector3(worldPositions[index].r, worldPositions[index].g, worldPositions[index].b);
                        if (!IsFinite(worldPos))
                            continue;

                        Color normalColor = worldNormals[index];
                        if (normalColor.a < 0.5f)
                            continue;

                        Vector3 worldNormal = new Vector3(normalColor.r, normalColor.g, normalColor.b);
                        if (worldNormal.sqrMagnitude <= 1e-6f)
                            continue;
                        worldNormal.Normalize();

                        float normalDot = Vector3.Dot(avgNormal, worldNormal);
                        if (normalDot < minNormalDot)
                            continue;

                        Vector3 delta = worldPos - centroid;
                        float planeDeviation = Mathf.Abs(Vector3.Dot(delta, avgNormal));
                        if (planeDeviation > maxPlaneDeviationMeters)
                            continue;

                        coherentCount++;
                        maxPlaneDeviation = Mathf.Max(maxPlaneDeviation, planeDeviation);
                        float u = Vector3.Dot(delta, tangent);
                        float v = Vector3.Dot(delta, bitangent);
                        minU = Mathf.Min(minU, u);
                        maxU = Mathf.Max(maxU, u);
                        minV = Mathf.Min(minV, v);
                        maxV = Mathf.Max(maxV, v);
                    }
                }

                if (coherentCount <= 0)
                    continue;

                float coherentRatio = coherentCount / (float)validSamples;
                if (coherentRatio < 0.65f)
                    continue;

                Vector2 size = new Vector2(
                    Mathf.Max(minPatchExtentMeters, maxU - minU),
                    Mathf.Max(minPatchExtentMeters, maxV - minV));

                _currentPatches.Add(new PatchCandidate
                {
                    valid = true,
                    worldPos = centroid,
                    worldNormal = avgNormal,
                    rotation = Quaternion.LookRotation(bitangent, avgNormal),
                    sizeMeters = size,
                    confidence = Mathf.Clamp01(meanConfidence * validRatio * coherentRatio),
                    sampleCount = coherentCount,
                    tileCoord = new Vector2Int(tileX / tile, tileY / tile),
                });
            }
        }

        LastIssue = null;
    }

    private void ResolveRefs()
    {
        if (preprocessor == null)
            preprocessor = FindAnyObjectByType<ScanCoverDepthPreprocessor>();
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverSurfacePatchCandidateProvider] {issue}");
        return false;
    }

    private static bool IsFinite(Vector3 value)
    {
        return
            !float.IsNaN(value.x) && !float.IsNaN(value.y) && !float.IsNaN(value.z) &&
            !float.IsInfinity(value.x) && !float.IsInfinity(value.y) && !float.IsInfinity(value.z);
    }
}

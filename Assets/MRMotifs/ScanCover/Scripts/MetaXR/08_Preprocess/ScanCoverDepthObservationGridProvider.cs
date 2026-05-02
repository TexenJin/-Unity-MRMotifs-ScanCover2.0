using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-44)]
[DisallowMultipleComponent]
public sealed class ScanCoverDepthObservationGridProvider : MonoBehaviour
{
    public enum InputMode
    {
        PrimaryOnly = 0,
        BinocularUnion = 1,
    }

    public enum ObservationSupportLayer
    {
        None = 0,
        MonoSupported = 1,
        BinocularConfirmed = 2,
        MonocularFallback = 3,
    }

    [Serializable]
    public struct Observation
    {
        public bool valid;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public float linearDepth;
        public float confidence;
        public int frameIndex;
        public Vector2Int sourcePixel;
        public ObservationSupportLayer supportLayer;
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverDepthPreprocessor preprocessor;
    [SerializeField] private ScanCoverDepthPreprocessor secondaryPreprocessor;

    [Header("Sampling")]
    [SerializeField] private InputMode inputMode = InputMode.PrimaryOnly;
    [SerializeField, Min(1)] private int stride = 4;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0.15f;
    [SerializeField] private bool requireValidNormal = true;
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField] private bool dedupeBinocularOverlap = true;
    [SerializeField, Min(0.001f)] private float binocularMergeCellMeters = 0.05f;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public IReadOnlyList<Observation> CurrentObservations => _currentObservations;
    public int ObservationFrameIndex => _observationFrameIndex;
    public bool HasPendingReadback => _hasPendingReadback;
    public int Stride => Mathf.Max(1, stride);
    public Vector2Int CurrentResolution => _currentResolution;
    public string LastIssue { get; private set; }

    private readonly List<Observation> _currentObservations = new List<Observation>(4096);
    private AsyncGPUReadbackRequest _worldPositionRequest;
    private AsyncGPUReadbackRequest _worldNormalRequest;
    private AsyncGPUReadbackRequest _observationMetaRequest;
    private AsyncGPUReadbackRequest _secondaryWorldPositionRequest;
    private AsyncGPUReadbackRequest _secondaryWorldNormalRequest;
    private AsyncGPUReadbackRequest _secondaryObservationMetaRequest;
    private bool _hasPendingReadback;
    private Vector2Int _pendingResolution;
    private Vector2Int _pendingSecondaryResolution;
    private Vector2Int _currentResolution;
    private int _observationFrameIndex;

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

    [ContextMenu("Refresh Observation Grid")]
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

        if (inputMode == InputMode.BinocularUnion &&
            secondaryPreprocessor != null &&
            secondaryPreprocessor.TryGetOutputs(
                out RenderTexture secondaryWorldPositionTexture,
                out RenderTexture secondaryWorldNormalTexture,
                out RenderTexture secondaryObservationMetaTexture))
        {
            _pendingSecondaryResolution = secondaryPreprocessor.OutputResolution;
            _secondaryWorldPositionRequest = AsyncGPUReadback.Request(secondaryWorldPositionTexture);
            _secondaryWorldNormalRequest = AsyncGPUReadback.Request(secondaryWorldNormalTexture);
            _secondaryObservationMetaRequest = AsyncGPUReadback.Request(secondaryObservationMetaTexture);
        }
        else
        {
            _pendingSecondaryResolution = Vector2Int.zero;
            _secondaryWorldPositionRequest = default;
            _secondaryWorldNormalRequest = default;
            _secondaryObservationMetaRequest = default;
        }

        _hasPendingReadback = true;
        LastIssue = null;
        return true;
    }

    private void UpdatePendingReadback()
    {
        if (!_worldPositionRequest.done || !_worldNormalRequest.done || !_observationMetaRequest.done)
            return;

        bool hasSecondary = _pendingSecondaryResolution != Vector2Int.zero;
        if (hasSecondary &&
            (!_secondaryWorldPositionRequest.done || !_secondaryWorldNormalRequest.done || !_secondaryObservationMetaRequest.done))
        {
            return;
        }

        _hasPendingReadback = false;

        if (_worldPositionRequest.hasError || _worldNormalRequest.hasError || _observationMetaRequest.hasError)
        {
            SetIssue("AsyncGPUReadback failed.");
            return;
        }

        BuildPrimaryObservationList(
            _worldPositionRequest.GetData<Color>(),
            _worldNormalRequest.GetData<Color>(),
            _observationMetaRequest.GetData<Color>(),
            _pendingResolution);

        if (hasSecondary)
        {
            if (_secondaryWorldPositionRequest.hasError || _secondaryWorldNormalRequest.hasError || _secondaryObservationMetaRequest.hasError)
            {
                SetIssue("Secondary AsyncGPUReadback failed.");
                return;
            }

            AppendObservationList(
                _currentObservations,
                _secondaryWorldPositionRequest.GetData<Color>(),
                _secondaryWorldNormalRequest.GetData<Color>(),
                _secondaryObservationMetaRequest.GetData<Color>(),
                _pendingSecondaryResolution,
                _pendingResolution.x + Mathf.Max(1, stride));
        }

        if (inputMode == InputMode.BinocularUnion && dedupeBinocularOverlap)
            DedupeBinocularObservations();

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverDepthObservationGridProvider] observations={_currentObservations.Count}, " +
                $"resolution={_currentResolution.x}x{_currentResolution.y}, stride={Mathf.Max(1, stride)}, mode={inputMode}");
        }
    }

    private void BuildPrimaryObservationList(
        NativeArray<Color> worldPositions,
        NativeArray<Color> worldNormals,
        NativeArray<Color> observationMeta,
        Vector2Int resolution)
    {
        _currentObservations.Clear();
        _observationFrameIndex++;
        _currentResolution = resolution;

        AppendObservationList(_currentObservations, worldPositions, worldNormals, observationMeta, resolution, 0);
        LastIssue = null;
    }

    private void AppendObservationList(
        List<Observation> destination,
        NativeArray<Color> worldPositions,
        NativeArray<Color> worldNormals,
        NativeArray<Color> observationMeta,
        Vector2Int resolution,
        int pixelXOffset)
    {
        if (destination == null)
            return;

        int width = Mathf.Max(1, resolution.x);
        int height = Mathf.Max(1, resolution.y);
        int step = Mathf.Max(1, stride);

        for (int y = 0; y < height; y += step)
        {
            for (int x = 0; x < width; x += step)
            {
                int index = x + y * width;
                Color meta = observationMeta[index];
                if (meta.r < 0.5f)
                    continue;

                float confidence = meta.g;
                if (confidence < minConfidence)
                    continue;

                Color pos = worldPositions[index];
                Vector3 worldPos = new Vector3(pos.r, pos.g, pos.b);
                if (!IsFinite(worldPos))
                    continue;

                Color normal = worldNormals[index];
                float normalValid = normal.a;
                if (requireValidNormal && normalValid < 0.5f)
                    continue;

                Vector3 worldNormal = new Vector3(normal.r, normal.g, normal.b);
                if (worldNormal.sqrMagnitude > 1e-8f)
                    worldNormal.Normalize();
                else if (requireValidNormal)
                    continue;

                destination.Add(new Observation
                {
                    valid = true,
                    worldPos = worldPos,
                    worldNormal = worldNormal,
                    linearDepth = meta.b,
                    confidence = confidence,
                    frameIndex = _observationFrameIndex,
                    sourcePixel = new Vector2Int(x + pixelXOffset, y),
                    supportLayer = ObservationSupportLayer.MonoSupported,
                });
            }
        }
    }

    private void ResolveRefs()
    {
        if (preprocessor == null)
            preprocessor = FindAnyObjectByType<ScanCoverDepthPreprocessor>();
        if (secondaryPreprocessor == null && inputMode == InputMode.BinocularUnion)
        {
            ScanCoverDepthPreprocessor[] preprocessors = FindObjectsByType<ScanCoverDepthPreprocessor>(FindObjectsSortMode.None);
            for (int i = 0; i < preprocessors.Length; i++)
            {
                ScanCoverDepthPreprocessor candidate = preprocessors[i];
                if (candidate == null || candidate == preprocessor)
                    continue;
                secondaryPreprocessor = candidate;
                break;
            }
        }
    }

    private void DedupeBinocularObservations()
    {
        float cellSize = Mathf.Max(0.001f, binocularMergeCellMeters);
        var merged = new Dictionary<Vector3Int, Observation>(_currentObservations.Count);

        for (int i = 0; i < _currentObservations.Count; i++)
        {
            Observation observation = _currentObservations[i];
            Vector3Int key = new Vector3Int(
                Mathf.RoundToInt(observation.worldPos.x / cellSize),
                Mathf.RoundToInt(observation.worldPos.y / cellSize),
                Mathf.RoundToInt(observation.worldPos.z / cellSize));

            if (!merged.TryGetValue(key, out Observation existing))
            {
                merged.Add(key, observation);
                continue;
            }

            float existingWeight = Mathf.Max(0.001f, existing.confidence);
            float incomingWeight = Mathf.Max(0.001f, observation.confidence);
            float totalWeight = existingWeight + incomingWeight;

            existing.worldPos = Vector3.Lerp(existing.worldPos, observation.worldPos, incomingWeight / totalWeight);

            Vector3 mergedNormal = existing.worldNormal * existingWeight + observation.worldNormal * incomingWeight;
            if (mergedNormal.sqrMagnitude > 1e-8f)
                mergedNormal.Normalize();
            existing.worldNormal = mergedNormal;

            existing.confidence = Mathf.Clamp01(Mathf.Max(existing.confidence, observation.confidence));
            existing.linearDepth = Mathf.Min(existing.linearDepth, observation.linearDepth);
            if (observation.sourcePixel.x < existing.sourcePixel.x)
                existing.sourcePixel = observation.sourcePixel;

            merged[key] = existing;
        }

        _currentObservations.Clear();
        foreach (Observation observation in merged.Values)
            _currentObservations.Add(observation);
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverDepthObservationGridProvider] {issue}");
        return false;
    }

    private static bool IsFinite(Vector3 value)
    {
        return
            !float.IsNaN(value.x) && !float.IsNaN(value.y) && !float.IsNaN(value.z) &&
            !float.IsInfinity(value.x) && !float.IsInfinity(value.y) && !float.IsInfinity(value.z);
    }
}

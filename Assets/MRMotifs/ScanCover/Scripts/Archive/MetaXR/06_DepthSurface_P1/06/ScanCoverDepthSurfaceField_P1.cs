using System;
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverDepthSurfaceField_P1 : MonoBehaviour
{
    private struct ProjectedPatchCellKey : IEquatable<ProjectedPatchCellKey>
    {
        public int x;
        public int y;

        public ProjectedPatchCellKey(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        public bool Equals(ProjectedPatchCellKey other) => x == other.x && y == other.y;
        public override bool Equals(object obj) => obj is ProjectedPatchCellKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                return (x * 397) ^ y;
            }
        }
    }

    [Serializable]
    public struct SurfaceSampleKey : IEquatable<SurfaceSampleKey>
    {
        public int x;
        public int y;
        public int z;

        public SurfaceSampleKey(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public bool Equals(SurfaceSampleKey other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is SurfaceSampleKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + x;
                h = h * 31 + y;
                h = h * 31 + z;
                return h;
            }
        }

        public override string ToString() => $"({x},{y},{z})";
    }

    [Serializable]
    public struct SurfacePatchKey : IEquatable<SurfacePatchKey>
    {
        public int x;
        public int y;
        public int z;

        public SurfacePatchKey(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public bool Equals(SurfacePatchKey other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is SurfacePatchKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + x;
                h = h * 31 + y;
                h = h * 31 + z;
                return h;
            }
        }

        public override string ToString() => $"({x},{y},{z})";
    }

    [Serializable]
    public struct SurfaceSampleInfo
    {
        public SurfaceSampleKey key;
        public SurfacePatchKey patchKey;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public float confidence;
        public float stability;
        public int hitCount;
        public float firstSeenTime;
        public float lastSeenTime;
        public ScanCoverSurfaceSupportLayer supportLayer;
    }

    [Serializable]
    public struct SurfacePatchInfo
    {
        public SurfacePatchKey key;
        public Vector3 centerWS;
        public Vector3 meanNormalWS;
        public Vector3 tangentWS;
        public Vector3 bitangentWS;
        public Vector2 footprintMeters;
        public float supportRadiusMeters;
        public int sampleCount;
        public int stableSampleCount;
        public float stableRatio;
        public int occupiedProjectedCells;
        public int projectedCellSpan;
        public float coverageRatio;
        public float meanHeightDeviationMeters;
        public float meanConfidence;
        public float meanStability;
        public float lastSeenTime;
    }

    private sealed class SurfaceSample
    {
        public Vector3 positionRef;
        public Vector3 normalRef;
        public float confidence;
        public float stability;
        public int hitCount;
        public float firstSeen;
        public float lastSeen;
        public ScanCoverSurfaceSupportLayer supportLayer;
    }

    private sealed class PatchAccumulator
    {
        public Vector3 centerSumRef;
        public Vector3 normalSumRef;
        public float confidenceSum;
        public float stabilitySum;
        public int sampleCount;
        public int stableSampleCount;
        public float lastSeen;
        public Vector3 centerRef;
        public Vector3 normalRef;
        public Vector3 tangentRef;
        public Vector3 bitangentRef;
        public Vector2 footprintMeters;
        public float supportRadiusMeters;
        public int occupiedProjectedCells;
        public int projectedCellSpan;
        public float coverageRatio;
        public float meanHeightDeviationMeters;
    }

    private sealed class InferenceCellAccumulator
    {
        public Vector2 planeSum;
        public float heightSum;
        public float confidenceSum;
        public float stabilitySum;
        public int sampleCount;
    }

    [Header("Refs")]
    public ScanCoverSkeletonBuilder_A builder;
    public ScanCoverSkeletonSessionController sessionController;
    public Transform referenceFrame;
    public EnvironmentRaycastManager environmentRaycast;
    public Camera sampleCamera;
    public ScanCoverDepthSurfaceProvider_P1 surfaceProvider;
    public ScanCoverDepthObservationProvider_07 observationProvider07;

    [Header("Sampling State")]
    public bool sampleWhileScanning = true;
    public bool sampleWhenFrozen = false;
    public bool clearOnEnterScanning = true;

    [Header("Sampling")]
    [Min(1)] public int samplesPerFrame = 48;
    [Min(0.2f)] public float maxRayDistance = 6.0f;
    public bool acceptHitPointOccluded = true;
    [Range(0f, 0.2f)] public float viewportMargin = 0.03f;

    [Header("Surface Quantization")]
    [Min(0.01f)] public float surfaceCellSizeMeters = 0.04f;
    [Min(0.05f)] public float patchSizeMeters = 0.40f;

    [Header("Patch Aggregation")]
    [Min(0.01f)] public float patchProjectionCellMeters = 0.04f;

    [Header("Fusion")]
    [Range(0f, 0.999f)] public float normalDotMin = 0.45f;
    [Min(0f)] public float maxMergeDistanceMeters = 0.10f;
    [Range(0f, 1f)] public float stablePositionBlend = 0.20f;
    [Range(0f, 1f)] public float correctionPositionBlend = 0.08f;
    [Range(0f, 1f)] public float normalBlend = 0.18f;
    [Min(0f)] public float stableDistanceMeters = 0.025f;
    [Range(0f, 1f)] public float stableNormalDot = 0.92f;
    [Range(0f, 1f)] public float confidenceGain = 0.16f;
    [Range(0f, 1f)] public float confidenceDecay = 0.08f;
    [Range(0f, 1f)] public float stabilityLerp = 0.22f;

    [Header("Hit Filters")]
    public bool inheritHitFiltersFromBuilder = true;
    [Min(0f)] public float minHitDistanceMeters = 0.35f;
    public bool enableSelfExclusion = true;
    public Transform[] selfExcludeTransforms;
    [Min(0f)] public float selfExcludeRadiusMeters = 0.25f;

    [Header("Sampling Bounds")]
    public bool inheritSamplingBoundsFromBuilder = true;
    public bool enableSamplingBoundsFilter = false;
    public Vector2 samplingViewportCenter = new Vector2(0.5f, 0.5f);
    public Vector2 samplingViewportSize = new Vector2(0.40f, 0.50f);
    [Range(0.2f, 1f)] public float samplingViewportMaxSize = 0.90f;

    [Header("Prune")]
    [Min(0f)] public float staleSeconds = 20f;
    [Min(0.1f)] public float pruneIntervalSeconds = 1.0f;
    [Min(100)] public int softMaxSamples = 16000;

    [Header("Temporal Inference")]
    public bool enableTemporallyInferredSamples = true;
    [Min(1)] public int temporallyInferredNeighborRadiusCells = 1;
    [Min(1)] public int temporallyInferredMinNeighborCells = 4;
    [Range(0f, 1f)] public float temporallyInferredConfidenceScale = 0.45f;
    [Range(0f, 1f)] public float temporallyInferredStabilityScale = 0.85f;
    [Min(0f)] public float temporallyInferredMaxHeightDeviationMeters = 0.035f;
    [Min(0)] public int maxTemporallyInferredSamples = 4096;
    public bool enablePlanarBridgeFill = true;
    [Min(1)] public int planarBridgeMaxGapCells = 8;
    [Min(1)] public int planarBridgeMinRunLength = 2;
    [Range(0f, 1f)] public float planarBridgeConfidenceScale = 0.35f;
    [Range(0f, 1f)] public float planarBridgeStabilityScale = 0.80f;

    [Header("Debug")]
    public bool debugLog = false;

    public int SampleCount => _samples.Count;
    public int InferredSampleCount
    {
        get
        {
            RebuildPatchCacheIfNeeded();
            return _inferredSampleSnapshotScratch.Count;
        }
    }
    public int PatchCount
    {
        get
        {
            RebuildPatchCacheIfNeeded();
            return _patchCache.Count;
        }
    }

    private readonly Dictionary<SurfaceSampleKey, SurfaceSample> _samples = new Dictionary<SurfaceSampleKey, SurfaceSample>(4096);
    private readonly Dictionary<SurfacePatchKey, PatchAccumulator> _patchCache = new Dictionary<SurfacePatchKey, PatchAccumulator>(256);
    private readonly List<SurfaceSampleKey> _removeKeys = new List<SurfaceSampleKey>(2048);
    private readonly List<SurfaceSampleInfo> _sampleSnapshotScratch = new List<SurfaceSampleInfo>(8192);
    private readonly List<SurfaceSampleInfo> _inferredSampleSnapshotScratch = new List<SurfaceSampleInfo>(4096);
    private readonly List<ScanCoverDepthSurfaceProvider_P1.SurfaceObservation> _observationScratch = new List<ScanCoverDepthSurfaceProvider_P1.SurfaceObservation>(256);
    private readonly List<ScanCoverSurfaceObservation> _observation07Scratch = new List<ScanCoverSurfaceObservation>(4096);
    private int _sampleIndex = 1;
    private float _nextPruneTime;
    private bool _patchCacheDirty = true;
    private ScanCoverSkeletonSessionController.SessionState _lastState = ScanCoverSkeletonSessionController.SessionState.Idle;

    private void Awake()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();
        _nextPruneTime = 0f;
        if (sessionController != null)
            _lastState = sessionController.State;
    }

    private void Update()
    {
        ResolveRefs();
        SyncBuilderDefaults();

        var state = sessionController != null ? sessionController.State : ScanCoverSkeletonSessionController.SessionState.Idle;
        if (clearOnEnterScanning && state != _lastState && state == ScanCoverSkeletonSessionController.SessionState.Scanning)
            ClearAllSamples();
        _lastState = state;

        if (!ShouldSample(state))
            return;

        float now = Time.time;
        if (observationProvider07 != null && observationProvider07.IsReady)
        {
            _observation07Scratch.Clear();
            observationProvider07.CollectObservations(_observation07Scratch);
            for (int i = 0; i < _observation07Scratch.Count; i++)
            {
                var obs = _observation07Scratch[i];
                AddObservation(obs, now);
            }
        }
        else if (surfaceProvider != null)
        {
            _observationScratch.Clear();
            surfaceProvider.CollectObservations(_observationScratch, Mathf.Max(1, samplesPerFrame));
            for (int i = 0; i < _observationScratch.Count; i++)
            {
                var obs = _observationScratch[i];
                AddObservation(obs.worldPos, obs.worldNormal, now, obs.confidence);
            }
        }
        else
        {
            if (!environmentRaycast || !EnvironmentRaycastManager.IsSupported)
                return;

            if (!sampleCamera)
                sampleCamera = Camera.main;
            if (!sampleCamera)
                return;

            int stepCount = Mathf.Max(1, samplesPerFrame);
            for (int i = 0; i < stepCount; i++)
            {
                Vector2 uv = NextHalton2D();
                Ray ray = sampleCamera.ViewportPointToRay(new Vector3(uv.x, uv.y, 0f));

                bool hitSuccess = environmentRaycast.Raycast(ray, out var hit, maxDistance: maxRayDistance);
                bool usable = hitSuccess || (acceptHitPointOccluded && hit.status == EnvironmentRaycastHitStatus.HitPointOccluded);
                if (!usable)
                    continue;

                Vector3 hitPos = hit.point;
                Vector3 hitNormal = hit.normal.sqrMagnitude > 1e-6f ? hit.normal.normalized : (-ray.direction).normalized;

                float hitDist = Vector3.Distance(ray.origin, hitPos);
                if (minHitDistanceMeters > 0f && hitDist < minHitDistanceMeters)
                    continue;
                if (enableSelfExclusion && selfExcludeRadiusMeters > 0f && IsSelfExcluded(hitPos))
                    continue;
                if (enableSamplingBoundsFilter && !IsInsideSamplingBounds(hitPos))
                    continue;

                AddObservation(hitPos, hitNormal, now, hitSuccess ? 1f : 0.6f);
            }
        }

        if (staleSeconds > 0f && now >= _nextPruneTime)
        {
            PruneStale(now);
            _nextPruneTime = now + Mathf.Max(0.1f, pruneIntervalSeconds);
        }

        if (softMaxSamples > 0 && _samples.Count > softMaxSamples)
            PruneAggressive();
    }

    public void ClearAllSamples()
    {
        _samples.Clear();
        _patchCache.Clear();
        _sampleSnapshotScratch.Clear();
        _removeKeys.Clear();
        _sampleIndex = 1;
        _patchCacheDirty = false;
        if (debugLog) Debug.Log("[ScanCoverDepthSurfaceField_P1] ClearAllSamples");
    }

    public void GetSamplesSnapshot(List<SurfaceSampleInfo> outList)
    {
        if (outList == null)
            return;

        outList.Clear();
        outList.Capacity = Mathf.Max(outList.Capacity, _samples.Count);
        foreach (var kv in _samples)
        {
            SurfaceSample sample = kv.Value;
            Vector3 normalRef = sample.normalRef.sqrMagnitude > 1e-6f ? sample.normalRef.normalized : Vector3.up;
            Vector3 worldPos = referenceFrame.TransformPoint(sample.positionRef);
            outList.Add(new SurfaceSampleInfo
            {
                key = kv.Key,
                patchKey = RefToPatchKey(sample.positionRef),
                worldPos = worldPos,
                worldNormal = referenceFrame.TransformDirection(normalRef).normalized,
                confidence = sample.confidence,
                stability = sample.stability,
                hitCount = sample.hitCount,
                firstSeenTime = sample.firstSeen,
                lastSeenTime = sample.lastSeen,
                supportLayer = sample.supportLayer,
            });
        }

        RebuildPatchCacheIfNeeded();
        if (_inferredSampleSnapshotScratch.Count > 0)
            outList.AddRange(_inferredSampleSnapshotScratch);
    }

    public void GetPatchSnapshot(List<SurfacePatchInfo> outList)
    {
        if (outList == null)
            return;

        RebuildPatchCacheIfNeeded();
        outList.Clear();
        outList.Capacity = Mathf.Max(outList.Capacity, _patchCache.Count);

        foreach (var kv in _patchCache)
        {
            PatchAccumulator patch = kv.Value;
            if (patch.sampleCount <= 0)
                continue;

            outList.Add(new SurfacePatchInfo
            {
                key = kv.Key,
                centerWS = referenceFrame.TransformPoint(patch.centerRef),
                meanNormalWS = referenceFrame.TransformDirection(patch.normalRef).normalized,
                tangentWS = referenceFrame.TransformDirection(patch.tangentRef).normalized,
                bitangentWS = referenceFrame.TransformDirection(patch.bitangentRef).normalized,
                footprintMeters = patch.footprintMeters,
                supportRadiusMeters = patch.supportRadiusMeters,
                sampleCount = patch.sampleCount,
                stableSampleCount = patch.stableSampleCount,
                stableRatio = patch.sampleCount > 0 ? (float)patch.stableSampleCount / patch.sampleCount : 0f,
                occupiedProjectedCells = patch.occupiedProjectedCells,
                projectedCellSpan = patch.projectedCellSpan,
                coverageRatio = patch.coverageRatio,
                meanHeightDeviationMeters = patch.meanHeightDeviationMeters,
                meanConfidence = patch.confidenceSum / patch.sampleCount,
                meanStability = patch.stabilitySum / patch.sampleCount,
                lastSeenTime = patch.lastSeen,
            });
        }
    }

    private void ResolveRefs()
    {
        if (!builder) builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (!sessionController) sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        if (!referenceFrame && builder) referenceFrame = builder.referenceFrame;
        if (!referenceFrame) referenceFrame = transform;
        if (!environmentRaycast && builder) environmentRaycast = builder.environmentRaycast;
        if (!sampleCamera && builder) sampleCamera = builder.sampleCamera;
        if (!sampleCamera) sampleCamera = Camera.main;
        if (!surfaceProvider) surfaceProvider = GetComponent<ScanCoverDepthSurfaceProvider_P1>();
        if (!observationProvider07) observationProvider07 = GetComponent<ScanCoverDepthObservationProvider_07>();
    }

    private void SyncBuilderDefaults()
    {
        if (builder == null)
            return;

        if (!referenceFrame) referenceFrame = builder.referenceFrame ? builder.referenceFrame : transform;
        if (!environmentRaycast) environmentRaycast = builder.environmentRaycast;
        if (!sampleCamera) sampleCamera = builder.sampleCamera ? builder.sampleCamera : Camera.main;

        if (inheritHitFiltersFromBuilder)
        {
            minHitDistanceMeters = builder.minHitDistanceMeters;
            enableSelfExclusion = builder.enableSelfExclusion;
            selfExcludeTransforms = builder.selfExcludeTransforms;
            selfExcludeRadiusMeters = builder.selfExcludeRadiusMeters;
            acceptHitPointOccluded = builder.acceptHitPointOccluded;
            maxRayDistance = builder.maxRayDistance;
        }

        if (inheritSamplingBoundsFromBuilder)
        {
            enableSamplingBoundsFilter = builder.enableSamplingBoundsFilter;
            samplingViewportCenter = builder.samplingViewportCenter;
            samplingViewportSize = builder.samplingViewportSize;
            samplingViewportMaxSize = builder.samplingViewportMaxSize;
        }
    }

    private bool ShouldSample(ScanCoverSkeletonSessionController.SessionState state)
    {
        if (state == ScanCoverSkeletonSessionController.SessionState.Scanning)
            return sampleWhileScanning;
        if (state == ScanCoverSkeletonSessionController.SessionState.Frozen)
            return sampleWhenFrozen;
        return sessionController == null && sampleWhileScanning;
    }

    private void AddObservation(Vector3 worldPos, Vector3 worldNormal, float timeNow, float observationConfidence)
    {
        Vector3 posRef = referenceFrame.InverseTransformPoint(worldPos);
        Vector3 normalRef = referenceFrame.InverseTransformDirection(worldNormal).normalized;
        SurfaceSampleKey key = RefToSampleKey(posRef);
        float baseConfidence = Mathf.Clamp01(observationConfidence);

        if (!_samples.TryGetValue(key, out var sample))
        {
            sample = new SurfaceSample
            {
                positionRef = posRef,
                normalRef = normalRef,
                confidence = Mathf.Clamp01(Mathf.Max(confidenceGain, baseConfidence)),
                stability = 0f,
                hitCount = 1,
                firstSeen = timeNow,
                lastSeen = timeNow,
                supportLayer = ScanCoverSurfaceSupportLayer.MonoSupported,
            };
            _samples.Add(key, sample);
            _patchCacheDirty = true;
            return;
        }

        Vector3 currentNormal = sample.normalRef.sqrMagnitude > 1e-6f ? sample.normalRef.normalized : normalRef;
        float posDelta = Vector3.Distance(sample.positionRef, posRef);
        float normalDot = Vector3.Dot(currentNormal, normalRef);
        bool isCompatible = normalDot >= normalDotMin && posDelta <= maxMergeDistanceMeters;
        bool isStable = posDelta <= stableDistanceMeters && normalDot >= stableNormalDot;

        float posBlend = isCompatible ? stablePositionBlend : correctionPositionBlend;
        sample.positionRef = Vector3.Lerp(sample.positionRef, posRef, Mathf.Clamp01(posBlend));
        sample.normalRef = Vector3.Slerp(currentNormal, normalRef, Mathf.Clamp01(normalBlend));
        if (sample.normalRef.sqrMagnitude > 1e-6f)
            sample.normalRef.Normalize();

        sample.hitCount++;
        sample.lastSeen = timeNow;
        float confidenceStep = isCompatible ? confidenceGain * Mathf.Lerp(0.5f, 1f, baseConfidence) : -confidenceDecay;
        sample.confidence = Mathf.Clamp01(sample.confidence + confidenceStep);
        float targetStability = isStable ? 1f : (isCompatible ? 0.55f : 0f);
        sample.stability = Mathf.Lerp(sample.stability, targetStability, Mathf.Clamp01(stabilityLerp));
        if (sample.supportLayer != ScanCoverSurfaceSupportLayer.StereoConfirmed)
            sample.supportLayer = ScanCoverSurfaceSupportLayer.MonoSupported;
        _patchCacheDirty = true;
    }

    private void AddObservation(ScanCoverSurfaceObservation observation, float timeNow)
    {
        if (!observation.valid)
            return;

        float layerBoost = 1f;
        switch (observation.supportLayer)
        {
            case ScanCoverSurfaceSupportLayer.StereoConfirmed:
                layerBoost = 1f;
                break;
            case ScanCoverSurfaceSupportLayer.MonoSupported:
                layerBoost = 0.75f;
                break;
            case ScanCoverSurfaceSupportLayer.TemporallyInferred:
                layerBoost = 0.5f;
                break;
            default:
                layerBoost = 0f;
                break;
        }

        if (layerBoost <= 0f)
            return;

        Vector3 posRef = referenceFrame.InverseTransformPoint(observation.worldPos);
        Vector3 normalRef = referenceFrame.InverseTransformDirection(observation.worldNormal).normalized;
        SurfaceSampleKey key = RefToSampleKey(posRef);
        float baseConfidence = Mathf.Clamp01(observation.confidence * layerBoost);

        if (!_samples.TryGetValue(key, out var sample))
        {
            sample = new SurfaceSample
            {
                positionRef = posRef,
                normalRef = normalRef,
                confidence = Mathf.Clamp01(Mathf.Max(confidenceGain, baseConfidence)),
                stability = 0f,
                hitCount = 1,
                firstSeen = timeNow,
                lastSeen = timeNow,
                supportLayer = observation.supportLayer,
            };
            _samples.Add(key, sample);
            _patchCacheDirty = true;
            return;
        }

        Vector3 currentNormal = sample.normalRef.sqrMagnitude > 1e-6f ? sample.normalRef.normalized : normalRef;
        float posDelta = Vector3.Distance(sample.positionRef, posRef);
        float normalDot = Vector3.Dot(currentNormal, normalRef);
        bool isCompatible = normalDot >= normalDotMin && posDelta <= maxMergeDistanceMeters;
        bool isStable = posDelta <= stableDistanceMeters && normalDot >= stableNormalDot;

        float posBlend = isCompatible ? stablePositionBlend : correctionPositionBlend;
        sample.positionRef = Vector3.Lerp(sample.positionRef, posRef, Mathf.Clamp01(posBlend));
        sample.normalRef = Vector3.Slerp(currentNormal, normalRef, Mathf.Clamp01(normalBlend));
        if (sample.normalRef.sqrMagnitude > 1e-6f)
            sample.normalRef.Normalize();

        sample.hitCount++;
        sample.lastSeen = timeNow;
        float confidenceStep = isCompatible ? confidenceGain * Mathf.Lerp(0.5f, 1f, baseConfidence) : -confidenceDecay;
        sample.confidence = Mathf.Clamp01(sample.confidence + confidenceStep);
        float targetStability = isStable ? 1f : (isCompatible ? 0.55f : 0f);
        sample.stability = Mathf.Lerp(sample.stability, targetStability, Mathf.Clamp01(stabilityLerp));
        if (observation.supportLayer == ScanCoverSurfaceSupportLayer.StereoConfirmed ||
            sample.supportLayer == ScanCoverSurfaceSupportLayer.StereoConfirmed)
        {
            sample.supportLayer = ScanCoverSurfaceSupportLayer.StereoConfirmed;
        }
        else
        {
            sample.supportLayer = observation.supportLayer;
        }
        _patchCacheDirty = true;
    }

    private void RebuildPatchCacheIfNeeded()
    {
        if (!_patchCacheDirty)
            return;

        _patchCache.Clear();
        _inferredSampleSnapshotScratch.Clear();
        foreach (var kv in _samples)
        {
            SurfaceSample sample = kv.Value;
            SurfacePatchKey patchKey = RefToPatchKey(sample.positionRef);
            if (!_patchCache.TryGetValue(patchKey, out var patch))
            {
                patch = new PatchAccumulator();
                _patchCache.Add(patchKey, patch);
            }

            patch.centerSumRef += sample.positionRef;
            patch.normalSumRef += sample.normalRef;
            patch.confidenceSum += sample.confidence;
            patch.stabilitySum += sample.stability;
            patch.sampleCount++;
            if (sample.stability >= 0.75f)
                patch.stableSampleCount++;
            if (sample.lastSeen > patch.lastSeen)
                patch.lastSeen = sample.lastSeen;
        }

        foreach (var kv in _patchCache)
        {
            PatchAccumulator patch = kv.Value;
            if (patch.sampleCount <= 0)
                continue;

            patch.centerRef = patch.centerSumRef / patch.sampleCount;
            patch.normalRef = patch.normalSumRef.sqrMagnitude > 1e-6f ? patch.normalSumRef.normalized : Vector3.up;
            BuildPatchBasis(patch.normalRef, out patch.tangentRef, out patch.bitangentRef);
            patch.footprintMeters = Vector2.zero;
            patch.supportRadiusMeters = 0f;
            patch.occupiedProjectedCells = 0;
            patch.projectedCellSpan = 0;
            patch.coverageRatio = 0f;
            patch.meanHeightDeviationMeters = 0f;
        }

        float projCellSize = Mathf.Max(0.01f, patchProjectionCellMeters);
        var patchProjectedCells = new Dictionary<SurfacePatchKey, HashSet<ProjectedPatchCellKey>>(_patchCache.Count);
        var patchBounds = new Dictionary<SurfacePatchKey, Vector4>(_patchCache.Count);
        var patchHeightDev = new Dictionary<SurfacePatchKey, Vector2>(_patchCache.Count);

        foreach (var kv in _samples)
        {
            SurfaceSample sample = kv.Value;
            SurfacePatchKey patchKey = RefToPatchKey(sample.positionRef);
            if (!_patchCache.TryGetValue(patchKey, out var patch) || patch.sampleCount <= 0)
                continue;

            Vector3 delta = sample.positionRef - patch.centerRef;
            float tx = Mathf.Abs(Vector3.Dot(delta, patch.tangentRef));
            float ty = Mathf.Abs(Vector3.Dot(delta, patch.bitangentRef));
            if (tx * 2f > patch.footprintMeters.x)
                patch.footprintMeters.x = tx * 2f;
            if (ty * 2f > patch.footprintMeters.y)
                patch.footprintMeters.y = ty * 2f;

            float radial = Mathf.Sqrt(tx * tx + ty * ty);
            if (radial > patch.supportRadiusMeters)
                patch.supportRadiusMeters = radial;

            int cx = Mathf.RoundToInt(Vector3.Dot(delta, patch.tangentRef) / projCellSize);
            int cy = Mathf.RoundToInt(Vector3.Dot(delta, patch.bitangentRef) / projCellSize);
            if (!patchProjectedCells.TryGetValue(patchKey, out var cells))
            {
                cells = new HashSet<ProjectedPatchCellKey>();
                patchProjectedCells.Add(patchKey, cells);
            }
            cells.Add(new ProjectedPatchCellKey(cx, cy));

            if (!patchBounds.TryGetValue(patchKey, out var bounds))
                bounds = new Vector4(cx, cy, cx, cy);
            else
                bounds = new Vector4(
                    Mathf.Min(bounds.x, cx),
                    Mathf.Min(bounds.y, cy),
                    Mathf.Max(bounds.z, cx),
                    Mathf.Max(bounds.w, cy));
            patchBounds[patchKey] = bounds;

            float hz = Mathf.Abs(Vector3.Dot(delta, patch.normalRef));
            if (!patchHeightDev.TryGetValue(patchKey, out var heightStats))
                heightStats = Vector2.zero;
            heightStats.x += hz;
            heightStats.y += 1f;
            patchHeightDev[patchKey] = heightStats;
        }

        foreach (var kv in _patchCache)
        {
            SurfacePatchKey patchKey = kv.Key;
            PatchAccumulator patch = kv.Value;
            if (patch.sampleCount <= 0)
                continue;

            if (patchProjectedCells.TryGetValue(patchKey, out var cells))
            {
                patch.occupiedProjectedCells = cells.Count;
                if (patchBounds.TryGetValue(patchKey, out var bounds))
                {
                    int spanX = Mathf.Max(1, Mathf.RoundToInt(bounds.z - bounds.x) + 1);
                    int spanY = Mathf.Max(1, Mathf.RoundToInt(bounds.w - bounds.y) + 1);
                    patch.projectedCellSpan = spanX * spanY;
                    patch.coverageRatio = patch.projectedCellSpan > 0 ? (float)patch.occupiedProjectedCells / patch.projectedCellSpan : 0f;

                    // Use occupied projected bounds rather than raw max-sample outliers.
                    patch.footprintMeters = new Vector2(spanX * projCellSize, spanY * projCellSize);
                }
            }

            if (patchHeightDev.TryGetValue(patchKey, out var heightStats2) && heightStats2.y > 0f)
                patch.meanHeightDeviationMeters = heightStats2.x / heightStats2.y;
        }

        if (enableTemporallyInferredSamples)
            RebuildTemporallyInferredSamples(projCellSize, patchBounds);

        _patchCacheDirty = false;
    }

    private void RebuildTemporallyInferredSamples(float projCellSize, Dictionary<SurfacePatchKey, Vector4> patchBounds)
    {
        if (_patchCache.Count <= 0 || _samples.Count <= 0 || maxTemporallyInferredSamples <= 0)
            return;

        var patchCells = new Dictionary<SurfacePatchKey, Dictionary<ProjectedPatchCellKey, InferenceCellAccumulator>>(_patchCache.Count);
        foreach (var kv in _samples)
        {
            SurfaceSample sample = kv.Value;
            SurfacePatchKey patchKey = RefToPatchKey(sample.positionRef);
            if (!_patchCache.TryGetValue(patchKey, out var patch) || patch.sampleCount <= 0)
                continue;
            if (!patchBounds.ContainsKey(patchKey))
                continue;

            Vector3 delta = sample.positionRef - patch.centerRef;
            int cx = Mathf.RoundToInt(Vector3.Dot(delta, patch.tangentRef) / projCellSize);
            int cy = Mathf.RoundToInt(Vector3.Dot(delta, patch.bitangentRef) / projCellSize);
            float hz = Vector3.Dot(delta, patch.normalRef);

            if (!patchCells.TryGetValue(patchKey, out var cells))
            {
                cells = new Dictionary<ProjectedPatchCellKey, InferenceCellAccumulator>();
                patchCells.Add(patchKey, cells);
            }

            var cellKey = new ProjectedPatchCellKey(cx, cy);
            if (!cells.TryGetValue(cellKey, out var cell))
            {
                cell = new InferenceCellAccumulator();
                cells.Add(cellKey, cell);
            }

            cell.planeSum += new Vector2(cx * projCellSize, cy * projCellSize);
            cell.heightSum += hz;
            cell.confidenceSum += sample.confidence;
            cell.stabilitySum += sample.stability;
            cell.sampleCount++;
        }

        int radius = Mathf.Max(1, temporallyInferredNeighborRadiusCells);
        int minNeighbors = Mathf.Max(1, temporallyInferredMinNeighborCells);
        int budget = maxTemporallyInferredSamples;
        var emittedCells = new Dictionary<SurfacePatchKey, HashSet<ProjectedPatchCellKey>>(_patchCache.Count);

        foreach (var patchPair in _patchCache)
        {
            if (budget <= 0)
                break;

            SurfacePatchKey patchKey = patchPair.Key;
            PatchAccumulator patch = patchPair.Value;
            if (!patchBounds.TryGetValue(patchKey, out var bounds))
                continue;
            if (patch.meanHeightDeviationMeters > temporallyInferredMaxHeightDeviationMeters)
                continue;
            if (!patchCells.TryGetValue(patchKey, out var cells))
                continue;

            int minX = Mathf.RoundToInt(bounds.x);
            int minY = Mathf.RoundToInt(bounds.y);
            int maxX = Mathf.RoundToInt(bounds.z);
            int maxY = Mathf.RoundToInt(bounds.w);

            for (int cy = minY; cy <= maxY && budget > 0; cy++)
            {
                for (int cx = minX; cx <= maxX && budget > 0; cx++)
                {
                    var targetKey = new ProjectedPatchCellKey(cx, cy);
                    if (cells.ContainsKey(targetKey))
                        continue;

                    int neighborCount = 0;
                    Vector2 planeSum = Vector2.zero;
                    float heightSum = 0f;
                    float confidenceSum = 0f;
                    float stabilitySum = 0f;

                    for (int ny = cy - radius; ny <= cy + radius; ny++)
                    {
                        for (int nx = cx - radius; nx <= cx + radius; nx++)
                        {
                            if (nx == cx && ny == cy)
                                continue;

                            if (!cells.TryGetValue(new ProjectedPatchCellKey(nx, ny), out var neighbor))
                                continue;
                            if (neighbor.sampleCount <= 0)
                                continue;

                            neighborCount++;
                            planeSum += neighbor.planeSum / neighbor.sampleCount;
                            heightSum += neighbor.heightSum / neighbor.sampleCount;
                            confidenceSum += neighbor.confidenceSum / neighbor.sampleCount;
                            stabilitySum += neighbor.stabilitySum / neighbor.sampleCount;
                        }
                    }

                    if (neighborCount < minNeighbors)
                        continue;

                    Vector2 planeAvg = planeSum / neighborCount;
                    float heightAvg = heightSum / neighborCount;
                    float confidenceAvg = (confidenceSum / neighborCount) * Mathf.Clamp01(temporallyInferredConfidenceScale);
                    float stabilityAvg = (stabilitySum / neighborCount) * Mathf.Clamp01(temporallyInferredStabilityScale);

                    if (TryEmitInferredSample(
                            patchKey,
                            targetKey,
                            patch,
                            planeAvg,
                            heightAvg,
                            confidenceAvg,
                            stabilityAvg,
                            ref budget,
                            emittedCells))
                    {
                        continue;
                    }
                }
            }

            if (budget <= 0 || !enablePlanarBridgeFill)
                continue;

            ApplyPlanarBridgeFill(
                patchKey,
                patch,
                cells,
                bounds,
                projCellSize,
                ref budget,
                emittedCells);
        }
    }

    private void ApplyPlanarBridgeFill(
        SurfacePatchKey patchKey,
        PatchAccumulator patch,
        Dictionary<ProjectedPatchCellKey, InferenceCellAccumulator> cells,
        Vector4 bounds,
        float projCellSize,
        ref int budget,
        Dictionary<SurfacePatchKey, HashSet<ProjectedPatchCellKey>> emittedCells)
    {
        if (budget <= 0 || cells == null || cells.Count <= 0)
            return;
        if (patch.meanHeightDeviationMeters > temporallyInferredMaxHeightDeviationMeters)
            return;

        int minX = Mathf.RoundToInt(bounds.x);
        int minY = Mathf.RoundToInt(bounds.y);
        int maxX = Mathf.RoundToInt(bounds.z);
        int maxY = Mathf.RoundToInt(bounds.w);
        int maxGap = Mathf.Max(1, planarBridgeMaxGapCells);
        int minRun = Mathf.Max(1, planarBridgeMinRunLength);

        for (int cy = minY; cy <= maxY && budget > 0; cy++)
        {
            int segmentStart = int.MaxValue;
            int segmentEnd = int.MinValue;
            int segmentCount = 0;

            for (int cx = minX; cx <= maxX; cx++)
            {
                if (!cells.ContainsKey(new ProjectedPatchCellKey(cx, cy)))
                    continue;

                if (segmentStart == int.MaxValue)
                    segmentStart = cx;
                segmentEnd = cx;
                segmentCount++;
            }

            if (segmentCount < minRun || segmentStart == int.MaxValue)
                continue;

            FillBridgeCellsOnLine(
                patchKey,
                patch,
                cells,
                emittedCells,
                projCellSize,
                true,
                cy,
                segmentStart,
                segmentEnd,
                maxGap,
                ref budget);
        }

        for (int cx = minX; cx <= maxX && budget > 0; cx++)
        {
            int segmentStart = int.MaxValue;
            int segmentEnd = int.MinValue;
            int segmentCount = 0;

            for (int cy = minY; cy <= maxY; cy++)
            {
                if (!cells.ContainsKey(new ProjectedPatchCellKey(cx, cy)))
                    continue;

                if (segmentStart == int.MaxValue)
                    segmentStart = cy;
                segmentEnd = cy;
                segmentCount++;
            }

            if (segmentCount < minRun || segmentStart == int.MaxValue)
                continue;

            FillBridgeCellsOnLine(
                patchKey,
                patch,
                cells,
                emittedCells,
                projCellSize,
                false,
                cx,
                segmentStart,
                segmentEnd,
                maxGap,
                ref budget);
        }
    }

    private void FillBridgeCellsOnLine(
        SurfacePatchKey patchKey,
        PatchAccumulator patch,
        Dictionary<ProjectedPatchCellKey, InferenceCellAccumulator> cells,
        Dictionary<SurfacePatchKey, HashSet<ProjectedPatchCellKey>> emittedCells,
        float projCellSize,
        bool horizontal,
        int fixedCoord,
        int lineStart,
        int lineEnd,
        int maxGap,
        ref int budget)
    {
        int cursor = lineStart;
        while (cursor <= lineEnd && budget > 0)
        {
            ProjectedPatchCellKey currentKey = horizontal
                ? new ProjectedPatchCellKey(cursor, fixedCoord)
                : new ProjectedPatchCellKey(fixedCoord, cursor);

            if (cells.ContainsKey(currentKey))
            {
                cursor++;
                continue;
            }

            int gapStart = cursor;
            while (cursor <= lineEnd)
            {
                ProjectedPatchCellKey gapKey = horizontal
                    ? new ProjectedPatchCellKey(cursor, fixedCoord)
                    : new ProjectedPatchCellKey(fixedCoord, cursor);
                if (cells.ContainsKey(gapKey))
                    break;
                cursor++;
            }

            int gapEnd = cursor - 1;
            int gapLength = gapEnd - gapStart + 1;
            if (gapLength <= 0 || gapLength > maxGap)
                continue;

            ProjectedPatchCellKey leftKey = horizontal
                ? new ProjectedPatchCellKey(gapStart - 1, fixedCoord)
                : new ProjectedPatchCellKey(fixedCoord, gapStart - 1);
            ProjectedPatchCellKey rightKey = horizontal
                ? new ProjectedPatchCellKey(gapEnd + 1, fixedCoord)
                : new ProjectedPatchCellKey(fixedCoord, gapEnd + 1);

            if (!cells.TryGetValue(leftKey, out var leftCell) || !cells.TryGetValue(rightKey, out var rightCell))
                continue;
            if (leftCell.sampleCount <= 0 || rightCell.sampleCount <= 0)
                continue;

            Vector2 leftPlane = leftCell.planeSum / leftCell.sampleCount;
            Vector2 rightPlane = rightCell.planeSum / rightCell.sampleCount;
            float leftHeight = leftCell.heightSum / leftCell.sampleCount;
            float rightHeight = rightCell.heightSum / rightCell.sampleCount;
            float leftConfidence = leftCell.confidenceSum / leftCell.sampleCount;
            float rightConfidence = rightCell.confidenceSum / rightCell.sampleCount;
            float leftStability = leftCell.stabilitySum / leftCell.sampleCount;
            float rightStability = rightCell.stabilitySum / rightCell.sampleCount;

            for (int i = 0; i < gapLength && budget > 0; i++)
            {
                float t = (i + 1f) / (gapLength + 1f);
                int gapCoord = gapStart + i;
                ProjectedPatchCellKey targetKey = horizontal
                    ? new ProjectedPatchCellKey(gapCoord, fixedCoord)
                    : new ProjectedPatchCellKey(fixedCoord, gapCoord);
                Vector2 planeAvg = Vector2.Lerp(leftPlane, rightPlane, t);
                float heightAvg = Mathf.Lerp(leftHeight, rightHeight, t);
                float confidenceAvg = Mathf.Lerp(leftConfidence, rightConfidence, t) * Mathf.Clamp01(planarBridgeConfidenceScale);
                float stabilityAvg = Mathf.Lerp(leftStability, rightStability, t) * Mathf.Clamp01(planarBridgeStabilityScale);

                TryEmitInferredSample(
                    patchKey,
                    targetKey,
                    patch,
                    planeAvg,
                    heightAvg,
                    confidenceAvg,
                    stabilityAvg,
                    ref budget,
                    emittedCells);
            }
        }
    }

    private bool TryEmitInferredSample(
        SurfacePatchKey patchKey,
        ProjectedPatchCellKey targetKey,
        PatchAccumulator patch,
        Vector2 planeAvg,
        float heightAvg,
        float confidenceAvg,
        float stabilityAvg,
        ref int budget,
        Dictionary<SurfacePatchKey, HashSet<ProjectedPatchCellKey>> emittedCells)
    {
        if (budget <= 0)
            return false;

        if (!emittedCells.TryGetValue(patchKey, out var emitted))
        {
            emitted = new HashSet<ProjectedPatchCellKey>();
            emittedCells.Add(patchKey, emitted);
        }

        if (!emitted.Add(targetKey))
            return false;

        Vector3 inferredRef = patch.centerRef
            + patch.tangentRef * planeAvg.x
            + patch.bitangentRef * planeAvg.y
            + patch.normalRef * heightAvg;

        _inferredSampleSnapshotScratch.Add(new SurfaceSampleInfo
        {
            key = new SurfaceSampleKey(int.MinValue + _inferredSampleSnapshotScratch.Count, patchKey.x, patchKey.y),
            patchKey = patchKey,
            worldPos = referenceFrame.TransformPoint(inferredRef),
            worldNormal = referenceFrame.TransformDirection(patch.normalRef).normalized,
            confidence = confidenceAvg,
            stability = stabilityAvg,
            hitCount = 0,
            firstSeenTime = 0f,
            lastSeenTime = patch.lastSeen,
            supportLayer = ScanCoverSurfaceSupportLayer.TemporallyInferred,
        });
        budget--;
        return true;
    }

    private void PruneStale(float timeNow)
    {
        float threshold = timeNow - Mathf.Max(0f, staleSeconds);
        _removeKeys.Clear();
        foreach (var kv in _samples)
        {
            if (kv.Value.lastSeen < threshold)
                _removeKeys.Add(kv.Key);
        }

        if (_removeKeys.Count <= 0)
            return;

        for (int i = 0; i < _removeKeys.Count; i++)
            _samples.Remove(_removeKeys[i]);
        _patchCacheDirty = true;
    }

    private void PruneAggressive()
    {
        _sampleSnapshotScratch.Clear();
        GetSamplesSnapshot(_sampleSnapshotScratch);
        _sampleSnapshotScratch.Sort((a, b) => a.lastSeenTime.CompareTo(b.lastSeenTime));
        int removeCount = Mathf.Max(1, _samples.Count - softMaxSamples);
        for (int i = 0; i < removeCount && i < _sampleSnapshotScratch.Count; i++)
            _samples.Remove(_sampleSnapshotScratch[i].key);
        _patchCacheDirty = true;
    }

    private SurfaceSampleKey RefToSampleKey(Vector3 posRef)
    {
        float inv = 1f / Mathf.Max(1e-4f, surfaceCellSizeMeters);
        return new SurfaceSampleKey(
            Mathf.FloorToInt(posRef.x * inv),
            Mathf.FloorToInt(posRef.y * inv),
            Mathf.FloorToInt(posRef.z * inv));
    }

    private SurfacePatchKey RefToPatchKey(Vector3 posRef)
    {
        float inv = 1f / Mathf.Max(1e-4f, patchSizeMeters);
        return new SurfacePatchKey(
            Mathf.FloorToInt(posRef.x * inv),
            Mathf.FloorToInt(posRef.y * inv),
            Mathf.FloorToInt(posRef.z * inv));
    }

    private Vector2 NextHalton2D()
    {
        float u = Halton(_sampleIndex, 2);
        float v = Halton(_sampleIndex, 3);
        _sampleIndex++;
        if (_sampleIndex > 1_000_000)
            _sampleIndex = 1;

        float margin = Mathf.Clamp(viewportMargin, 0f, 0.2f);
        u = Mathf.Lerp(margin, 1f - margin, u);
        v = Mathf.Lerp(margin, 1f - margin, v);
        return new Vector2(u, v);
    }

    private static float Halton(int index, int b)
    {
        float f = 1f;
        float r = 0f;
        int i = index;
        while (i > 0)
        {
            f /= b;
            r += f * (i % b);
            i /= b;
        }
        return r;
    }

    private static void BuildPatchBasis(Vector3 normalRef, out Vector3 tangentRef, out Vector3 bitangentRef)
    {
        Vector3 n = normalRef.sqrMagnitude > 1e-6f ? normalRef.normalized : Vector3.up;
        Vector3 seed = Mathf.Abs(Vector3.Dot(n, Vector3.up)) < 0.9f ? Vector3.up : Vector3.right;
        tangentRef = Vector3.Cross(seed, n);
        if (tangentRef.sqrMagnitude <= 1e-6f)
            tangentRef = Vector3.Cross(Vector3.forward, n);
        tangentRef.Normalize();
        bitangentRef = Vector3.Cross(n, tangentRef);
        if (bitangentRef.sqrMagnitude <= 1e-6f)
            bitangentRef = Vector3.Cross(n, Vector3.right);
        bitangentRef.Normalize();
    }

    private bool IsSelfExcluded(Vector3 worldPos)
    {
        float radius = selfExcludeRadiusMeters;
        if (radius <= 0f)
            return false;

        float radiusSq = radius * radius;
        if (selfExcludeTransforms != null && selfExcludeTransforms.Length > 0)
        {
            for (int i = 0; i < selfExcludeTransforms.Length; i++)
            {
                Transform tr = selfExcludeTransforms[i];
                if (!tr)
                    continue;
                if ((worldPos - tr.position).sqrMagnitude <= radiusSq)
                    return true;
            }
            return false;
        }

        return sampleCamera && (worldPos - sampleCamera.transform.position).sqrMagnitude <= radiusSq;
    }

    private bool IsInsideSamplingBounds(Vector3 worldPos)
    {
        Camera cam = sampleCamera ? sampleCamera : Camera.main;
        if (!cam)
            return true;

        Vector3 vp = cam.WorldToViewportPoint(worldPos);
        if (vp.z <= 0f)
            return false;

        Vector2 center = new Vector2(
            Mathf.Clamp01(samplingViewportCenter.x),
            Mathf.Clamp01(samplingViewportCenter.y));
        float maxSize = Mathf.Clamp(samplingViewportMaxSize, 0.2f, 1f);
        Vector2 size = new Vector2(
            Mathf.Clamp(samplingViewportSize.x, 0.01f, maxSize),
            Mathf.Clamp(samplingViewportSize.y, 0.01f, maxSize));
        Vector2 half = size * 0.5f;
        return Mathf.Abs(vp.x - center.x) <= half.x && Mathf.Abs(vp.y - center.y) <= half.y;
    }
}

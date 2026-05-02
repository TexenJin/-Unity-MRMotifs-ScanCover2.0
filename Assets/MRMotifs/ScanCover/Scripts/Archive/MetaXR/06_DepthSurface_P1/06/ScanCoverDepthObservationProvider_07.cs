using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverDepthObservationProvider_07 : MonoBehaviour
{
    [Header("Refs")]
    public DepthGridPointCloud depthGridPointCloud;

    [Header("Filtering")]
    public bool includeStereoConfirmed = true;
    public bool includeMonoSupported = true;
    public bool includeTemporallyInferred = true;
    [Range(0f, 1f)] public float minConfidence = 0f;

    [Header("Debug")]
    public bool debugLog = false;

    public int LastFrameIndex { get; private set; } = -1;
    public int LastCollectedCount { get; private set; }

    private readonly List<ScanCoverSurfaceObservation> _cachedObservations = new List<ScanCoverSurfaceObservation>(4096);

    public bool IsReady => depthGridPointCloud != null;

    public IReadOnlyList<ScanCoverSurfaceObservation> CachedObservations => _cachedObservations;

    private void Awake()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();
    }

    private void OnValidate()
    {
        ResolveRefs();
    }

    public int CollectObservations(List<ScanCoverSurfaceObservation> outList)
    {
        ResolveRefs();

        if (outList == null)
            return 0;

        outList.Clear();
        _cachedObservations.Clear();
        LastCollectedCount = 0;

        if (!depthGridPointCloud)
            return 0;

        IReadOnlyList<ScanCoverSurfaceObservation> source = depthGridPointCloud.CurrentObservations;
        LastFrameIndex = depthGridPointCloud.ObservationFrameIndex;
        if (source == null || source.Count == 0)
            return 0;

        for (int i = 0; i < source.Count; i++)
        {
            ScanCoverSurfaceObservation observation = source[i];
            if (!observation.valid)
                continue;
            if (observation.confidence < minConfidence)
                continue;
            if (!IsLayerEnabled(observation.supportLayer))
                continue;

            outList.Add(observation);
            _cachedObservations.Add(observation);
        }

        LastCollectedCount = outList.Count;

        if (debugLog)
            Debug.Log($"[ScanCoverDepthObservationProvider_07] frame={LastFrameIndex} collected={LastCollectedCount}");

        return LastCollectedCount;
    }

    private bool IsLayerEnabled(ScanCoverSurfaceSupportLayer layer)
    {
        switch (layer)
        {
            case ScanCoverSurfaceSupportLayer.StereoConfirmed:
                return includeStereoConfirmed;
            case ScanCoverSurfaceSupportLayer.MonoSupported:
                return includeMonoSupported;
            case ScanCoverSurfaceSupportLayer.TemporallyInferred:
                return includeTemporallyInferred;
            default:
                return false;
        }
    }

    private void ResolveRefs()
    {
        if (!depthGridPointCloud)
            depthGridPointCloud = GetComponent<DepthGridPointCloud>();
    }
}

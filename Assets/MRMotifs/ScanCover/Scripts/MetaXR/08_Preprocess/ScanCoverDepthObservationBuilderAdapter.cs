using UnityEngine;

[DefaultExecutionOrder(-42)]
[DisallowMultipleComponent]
public sealed class ScanCoverDepthObservationBuilderAdapter : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverSkeletonBuilder_A builder;
    [SerializeField] private ScanCoverDepthObservationGridProvider provider;
    [SerializeField] private ScanCoverBinocularFusedObservationProvider fusedProvider;

    [Header("Flow")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField] private bool disableBuilderInternalRaycastSampling = true;
    [SerializeField, Min(1)] private int maxObservationsPerFrame = 128;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0.2f;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    private int _cursor;

    private void Awake()
    {
        ResolveRefs();
        ApplyBuilderMode();
    }

    private void OnEnable()
    {
        ResolveRefs();
        ApplyBuilderMode();
    }

    private void Update()
    {
        if (!updateEveryFrame)
            return;

        RefreshNow();
    }

    [ContextMenu("Refresh Builder Adapter")]
    public int RefreshNow()
    {
        ResolveRefs();
        ApplyBuilderMode();

        if (builder == null || (provider == null && fusedProvider == null))
            return 0;

        if (provider != null && !provider.HasPendingReadback)
            provider.RefreshNow();
        if (fusedProvider != null)
            fusedProvider.RefreshNow();

        var observations = fusedProvider != null ? fusedProvider.CurrentObservations : provider.CurrentObservations;
        int total = observations.Count;
        if (total <= 0)
            return 0;

        int budget = Mathf.Min(Mathf.Max(1, maxObservationsPerFrame), total);
        int added = 0;
        float now = Time.time;

        for (int i = 0; i < total && added < budget; i++)
        {
            int index = (_cursor + i) % total;
            var observation = observations[index];
            if (!observation.valid || observation.confidence < minConfidence)
                continue;

            if (builder.TryAddExternalObservation(observation.worldPos, observation.worldNormal, now))
                added++;
        }

        _cursor = (_cursor + budget) % Mathf.Max(1, total);

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverDepthObservationBuilderAdapter] added={added}, total={total}, " +
                $"budget={budget}, cursor={_cursor}");
        }

        return added;
    }

    private void ResolveRefs()
    {
        if (builder == null)
            builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (provider == null)
            provider = GetComponent<ScanCoverDepthObservationGridProvider>();
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverDepthObservationGridProvider>();
        if (fusedProvider == null)
            fusedProvider = GetComponent<ScanCoverBinocularFusedObservationProvider>();
        if (fusedProvider == null)
            fusedProvider = FindAnyObjectByType<ScanCoverBinocularFusedObservationProvider>();
    }

    private void ApplyBuilderMode()
    {
        if (builder != null && disableBuilderInternalRaycastSampling)
            builder.useInternalRaycastSampling = false;
    }
}

using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-43)]
[DisallowMultipleComponent]
public sealed class ScanCoverDepthObservationDebugPoints : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverDepthObservationGridProvider provider;
    [SerializeField] private ScanCoverBinocularFusedObservationProvider fusedProvider;
    [SerializeField] private ScanCoverStableObservationCloudAccumulator accumulator;
    [SerializeField] private GameObject markerPrefab;
    [SerializeField] private Transform markerParent;

    [Header("Display")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(0f)] private float surfaceBiasMeters = 0.0015f;
    [SerializeField] private bool orientToNormal = false;
    [SerializeField, Min(0.001f)] private float fallbackMarkerScaleMeters = 0.05f;
    [SerializeField] private Color debugPointUniformColor = new Color(0.55f, 0.55f, 0.55f, 1f);

    [Header("Sanity")]
    [SerializeField] private bool forceCameraRelativePreview;
    [SerializeField, Min(1)] private int cameraRelativePreviewCount = 64;
    [SerializeField, Min(0.1f)] private float cameraRelativeDistanceMeters = 0.8f;
    [SerializeField, Min(0.01f)] private float cameraRelativeSpacingMeters = 0.05f;

    [Header("Debug")]
    [SerializeField] private bool debugLog;
    [SerializeField] private bool logObservationBounds = true;

    private readonly List<GameObject> _pool = new List<GameObject>(2048);
    private readonly List<Renderer[]> _rendererCache = new List<Renderer[]>(2048);
    private MaterialPropertyBlock _propertyBlock;
    private Material _fallbackMaterial;

    private void Reset()
    {
        debugPointUniformColor = new Color(0.55f, 0.55f, 0.55f, 1f);
    }

    private void Awake()
    {
        ResolveRefs();
        _propertyBlock = new MaterialPropertyBlock();
        RefreshNow();
    }

    private void OnEnable()
    {
        ResolveRefs();
        if (_propertyBlock == null)
            _propertyBlock = new MaterialPropertyBlock();
        RefreshNow();
    }

    private void Update()
    {
        if (!updateEveryFrame)
            return;

        RefreshNow();
    }

    private void OnDestroy()
    {
        if (_fallbackMaterial != null)
        {
            Object.Destroy(_fallbackMaterial);
            _fallbackMaterial = null;
        }
    }

    [ContextMenu("Refresh Observation Debug Points")]
    public void RefreshNow()
    {
        ResolveRefs();
        if (provider == null && fusedProvider == null && accumulator == null)
            return;

        if (provider != null && !provider.HasPendingReadback)
            provider.RefreshNow();
        if (fusedProvider != null)
            fusedProvider.RefreshNow();
        if (accumulator != null)
            accumulator.RefreshNow();

        IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> observations =
            accumulator != null ? accumulator.CurrentObservations :
            fusedProvider != null ? fusedProvider.CurrentObservations :
            provider.CurrentObservations;
        EnsurePoolSize(observations.Count);

        Vector3 min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        Vector3 max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

        for (int i = 0; i < observations.Count; i++)
        {
            var observation = observations[i];
            GameObject marker = _pool[i];
            if (marker == null)
                continue;

            Vector3 position = observation.worldPos;
            if (surfaceBiasMeters > 0f && observation.worldNormal.sqrMagnitude > 1e-6f)
                position += observation.worldNormal.normalized * surfaceBiasMeters;

            if (forceCameraRelativePreview && i < cameraRelativePreviewCount && Camera.main != null)
            {
                int columns = Mathf.Max(1, Mathf.CeilToInt(Mathf.Sqrt(cameraRelativePreviewCount)));
                int row = i / columns;
                int col = i % columns;
                float offsetX = (col - (columns - 1) * 0.5f) * cameraRelativeSpacingMeters;
                float offsetY = -(row * cameraRelativeSpacingMeters);
                position =
                    Camera.main.transform.position +
                    Camera.main.transform.forward * cameraRelativeDistanceMeters +
                    Camera.main.transform.right * offsetX +
                    Camera.main.transform.up * offsetY;
            }

            min = Vector3.Min(min, position);
            max = Vector3.Max(max, position);

            marker.transform.position = position;
            if (orientToNormal && observation.worldNormal.sqrMagnitude > 1e-6f)
                marker.transform.rotation = Quaternion.LookRotation(observation.worldNormal.normalized, Vector3.up);

            ApplyColor(i, debugPointUniformColor);
            if (!marker.activeSelf)
                marker.SetActive(true);
        }

        for (int i = observations.Count; i < _pool.Count; i++)
        {
            if (_pool[i] != null && _pool[i].activeSelf)
                _pool[i].SetActive(false);
        }

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverDepthObservationDebugPoints] visible={observations.Count}, " +
                $"source={(accumulator != null ? "StableAccumulator" : fusedProvider != null ? "FusedProvider" : "GridProvider")}");
            if (observations.Count > 0 && logObservationBounds)
            {
                Vector3 first = observations[0].worldPos;
                Debug.Log(
                    $"[ScanCoverDepthObservationDebugPoints] first={first}, " +
                    $"boundsMin={min}, boundsMax={max}, markerPrefab={(markerPrefab ? markerPrefab.name : "<fallback-sphere>")}");
            }
        }
    }

    private void ResolveRefs()
    {
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverDepthObservationGridProvider>();
        if (fusedProvider == null)
            fusedProvider = FindAnyObjectByType<ScanCoverBinocularFusedObservationProvider>();
        if (accumulator == null)
            accumulator = FindAnyObjectByType<ScanCoverStableObservationCloudAccumulator>();
    }

    private void EnsurePoolSize(int count)
    {
        while (_pool.Count < count)
        {
            GameObject instance = markerPrefab != null
                ? Instantiate(markerPrefab, markerParent ? markerParent : transform)
                : CreateFallbackMarker();
            instance.SetActive(false);
            _pool.Add(instance);
            _rendererCache.Add(instance.GetComponentsInChildren<Renderer>(true));
        }
    }

    private GameObject CreateFallbackMarker()
    {
        GameObject instance = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        instance.name = "ScanCoverObservationMarker";
        instance.transform.SetParent(markerParent ? markerParent : transform, false);
        instance.transform.localScale = Vector3.one * Mathf.Max(0.001f, fallbackMarkerScaleMeters);
        Collider collider = instance.GetComponent<Collider>();
        if (collider != null)
            Object.Destroy(collider);

        Renderer renderer = instance.GetComponent<Renderer>();
        if (renderer != null)
        {
            if (_fallbackMaterial == null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                    shader = Shader.Find("Unlit/Color");
                _fallbackMaterial = new Material(shader)
                {
                    name = "ScanCoverObservationFallback_Mat"
                };
                _fallbackMaterial.SetColor("_BaseColor", Color.cyan);
                _fallbackMaterial.SetColor("_Color", Color.cyan);
            }

            renderer.sharedMaterial = _fallbackMaterial;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }

        return instance;
    }

    private void ApplyColor(int index, Color color)
    {
        Renderer[] renderers = _rendererCache[index];
        if (renderers == null)
            return;

        _propertyBlock.SetColor("_BaseColor", color);
        _propertyBlock.SetColor("_Color", color);

        foreach (Renderer renderer in renderers)
        {
            if (renderer == null)
                continue;
            renderer.SetPropertyBlock(_propertyBlock);
        }
    }
}

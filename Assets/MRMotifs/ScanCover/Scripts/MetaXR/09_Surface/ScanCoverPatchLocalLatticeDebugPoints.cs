using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-41)]
[DisallowMultipleComponent]
public sealed class ScanCoverPatchLocalLatticeDebugPoints : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverPatchLocalLatticeProvider provider;
    [SerializeField] private GameObject markerPrefab;
    [SerializeField] private Transform markerParent;

    [Header("Display")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(0.001f)] private float markerScaleMeters = 0.012f;
    [SerializeField, Min(0f)] private float surfaceBiasMeters = 0.001f;
    [SerializeField] private Gradient confidenceGradient;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    private readonly List<GameObject> _pool = new List<GameObject>(2048);
    private readonly List<Renderer[]> _rendererCache = new List<Renderer[]>(2048);
    private MaterialPropertyBlock _propertyBlock;
    private Material _fallbackMaterial;

    private void Reset()
    {
        if (confidenceGradient == null || confidenceGradient.colorKeys == null || confidenceGradient.colorKeys.Length == 0)
        {
            confidenceGradient = new Gradient
            {
                colorKeys = new[]
                {
                    new GradientColorKey(Color.green, 0f),
                    new GradientColorKey(Color.cyan, 0.5f),
                    new GradientColorKey(Color.white, 1f),
                },
                alphaKeys = new[]
                {
                    new GradientAlphaKey(1f, 0f),
                    new GradientAlphaKey(1f, 1f),
                }
            };
        }
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

    [ContextMenu("Refresh Patch Local Lattice Debug Points")]
    public void RefreshNow()
    {
        ResolveRefs();
        if (provider == null)
            return;

        provider.RefreshNow();
        IReadOnlyList<ScanCoverPatchLocalLatticeProvider.LatticeNode> nodes = provider.CurrentNodes;
        EnsurePoolSize(nodes.Count);

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];
            GameObject marker = _pool[i];
            if (marker == null)
                continue;

            Vector3 pos = node.worldPos;
            if (surfaceBiasMeters > 0f && node.worldNormal.sqrMagnitude > 1e-6f)
                pos += node.worldNormal.normalized * surfaceBiasMeters;

            marker.transform.position = pos;
            ApplyColor(i, confidenceGradient.Evaluate(Mathf.Clamp01(node.confidence)));
            if (!marker.activeSelf)
                marker.SetActive(true);
        }

        for (int i = nodes.Count; i < _pool.Count; i++)
        {
            if (_pool[i] != null && _pool[i].activeSelf)
                _pool[i].SetActive(false);
        }

        if (debugLog)
            Debug.Log($"[ScanCoverPatchLocalLatticeDebugPoints] visible={nodes.Count}");
    }

    private void ResolveRefs()
    {
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverPatchLocalLatticeProvider>();
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
        instance.name = "ScanCoverPatchLatticeMarker";
        instance.transform.SetParent(markerParent ? markerParent : transform, false);
        instance.transform.localScale = Vector3.one * Mathf.Max(0.001f, markerScaleMeters);

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
                    name = "ScanCoverPatchLatticeMarker_Mat"
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

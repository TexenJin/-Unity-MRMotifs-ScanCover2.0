using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-43)]
[DisallowMultipleComponent]
public sealed class ScanCoverSurfacePatchDebugQuads : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverSurfacePatchCandidateProvider provider;
    [SerializeField] private ScanCoverSurfacePatchAccumulator accumulator;
    [SerializeField] private Transform patchParent;

    [Header("Display")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Min(0.001f)] private float thicknessMeters = 0.003f;
    [SerializeField, Range(0.1f, 1f)] private float patchSizeScale = 0.7f;
    [SerializeField] private Gradient confidenceGradient;
    [SerializeField] private bool orientToPatch = true;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    private readonly List<GameObject> _pool = new List<GameObject>(512);
    private readonly List<Renderer> _renderers = new List<Renderer>(512);
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
                    new GradientColorKey(Color.red, 0f),
                    new GradientColorKey(Color.yellow, 0.5f),
                    new GradientColorKey(Color.cyan, 1f),
                },
                alphaKeys = new[]
                {
                    new GradientAlphaKey(0.5f, 0f),
                    new GradientAlphaKey(0.85f, 1f),
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

    private void OnDestroy()
    {
        if (_fallbackMaterial != null)
        {
            Object.Destroy(_fallbackMaterial);
            _fallbackMaterial = null;
        }
    }

    private void Update()
    {
        if (!updateEveryFrame)
            return;

        RefreshNow();
    }

    [ContextMenu("Refresh Patch Debug Quads")]
    public void RefreshNow()
    {
        ResolveRefs();
        if (provider == null && accumulator == null)
            return;

        // When an accumulator is assigned, let it drive the candidate provider lifecycle.
        // Refreshing the provider first can leave it in pending-readback state and starve the accumulator.
        if (accumulator != null)
            accumulator.RefreshNow();
        else if (provider != null && !provider.HasPendingReadback)
            provider.RefreshNow();

        IReadOnlyList<ScanCoverSurfacePatchCandidateProvider.PatchCandidate> patches =
            accumulator != null ? accumulator.CurrentPatches : provider.CurrentPatches;
        EnsurePoolSize(patches.Count);

        for (int i = 0; i < patches.Count; i++)
        {
            var patch = patches[i];
            GameObject quad = _pool[i];
            if (quad == null)
                continue;

            quad.transform.position = patch.worldPos;
            if (orientToPatch)
                quad.transform.rotation = patch.rotation;

            quad.transform.localScale = new Vector3(
                Mathf.Max(0.001f, patch.sizeMeters.x * patchSizeScale),
                Mathf.Max(0.001f, thicknessMeters),
                Mathf.Max(0.001f, patch.sizeMeters.y * patchSizeScale));

            ApplyColor(i, confidenceGradient.Evaluate(Mathf.Clamp01(patch.confidence)));
            if (!quad.activeSelf)
                quad.SetActive(true);
        }

        for (int i = patches.Count; i < _pool.Count; i++)
        {
            if (_pool[i] != null && _pool[i].activeSelf)
                _pool[i].SetActive(false);
        }

        if (debugLog)
            Debug.Log(
                $"[ScanCoverSurfacePatchDebugQuads] visible={patches.Count}, " +
                $"source={(accumulator != null ? "PatchAccumulator" : "PatchCandidates")}");
    }

    private void ResolveRefs()
    {
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverSurfacePatchCandidateProvider>();
        if (accumulator == null)
            accumulator = FindAnyObjectByType<ScanCoverSurfacePatchAccumulator>();
        if (patchParent == null)
            patchParent = transform;
    }

    private void EnsurePoolSize(int count)
    {
        while (_pool.Count < count)
        {
            GameObject instance = GameObject.CreatePrimitive(PrimitiveType.Cube);
            instance.name = "ScanCoverPatchQuad";
            instance.transform.SetParent(patchParent ? patchParent : transform, false);
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
                        name = "ScanCoverPatchCandidate_Mat"
                    };
                    _fallbackMaterial.SetColor("_BaseColor", Color.cyan);
                    _fallbackMaterial.SetColor("_Color", Color.cyan);
                }

                renderer.sharedMaterial = _fallbackMaterial;
                renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }

            instance.SetActive(false);
            _pool.Add(instance);
            _renderers.Add(renderer);
        }
    }

    private void ApplyColor(int index, Color color)
    {
        Renderer renderer = _renderers[index];
        if (renderer == null)
            return;

        _propertyBlock.SetColor("_BaseColor", color);
        _propertyBlock.SetColor("_Color", color);
        renderer.SetPropertyBlock(_propertyBlock);
    }
}

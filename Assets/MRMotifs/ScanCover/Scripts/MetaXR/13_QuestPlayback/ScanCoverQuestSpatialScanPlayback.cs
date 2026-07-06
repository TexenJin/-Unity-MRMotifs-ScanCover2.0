using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-35)]
[DisallowMultipleComponent]
public sealed class ScanCoverQuestSpatialScanPlayback : MonoBehaviour
{
    private const string DefaultResourcePath = "QuestSpatialScanDemo/office0_observer_scan_lite";

    private enum ScanGrowthMode
    {
        PresetPlayback = 0,
        HeadsetDriven = 1,
    }

    [Header("Data")]
    [SerializeField] private TextAsset scanLineAsset;
    [SerializeField] private string fallbackResourcePath = DefaultResourcePath;

    [Header("Playback")]
    [SerializeField] private ScanGrowthMode growthMode = ScanGrowthMode.HeadsetDriven;
    [SerializeField] private bool playOnStart = true;
    [SerializeField] private bool loop = true;
    [SerializeField, Min(0.5f)] private float playbackSeconds = 10.5f;
    [SerializeField, Range(1, 35)] private int maxVisibleStep = 35;

    [Header("Headset Driven")]
    [SerializeField, Range(20f, 120f)] private float headsetHorizontalFovDegrees = 76f;
    [SerializeField, Range(20f, 100f)] private float headsetVerticalFovDegrees = 58f;
    [SerializeField, Min(0.1f)] private float headsetRevealMaxDistance = 5.5f;
    [SerializeField, Min(1)] private int headsetRevealBudgetPerFrame = 180;
    [SerializeField, Range(0f, 1f)] private float headsetForwardBias = 0.15f;

    [Header("Placement")]
    [SerializeField] private bool alignRoomCenterToInitialHeadset = true;
    [SerializeField] private Vector3 roomCenterOffsetFromInitialHeadset = Vector3.zero;
    [SerializeField] private Vector3 localOffset = Vector3.zero;
    [SerializeField, Min(0.05f)] private float localScale = 1.0f;
    [SerializeField] private bool keepFloorAtParentY = true;

    [Header("Display")]
    [SerializeField] private Color baseLineColor = new Color(0.06f, 0.85f, 0.95f, 0.38f);
    [SerializeField] private Color hotLineColor = new Color(0.75f, 1f, 0.95f, 0.95f);
    [SerializeField] private bool showObserver = true;
    [SerializeField] private Color observerColor = new Color(1f, 0.83f, 0.32f, 1f);
    [SerializeField, Min(0.005f)] private float observerMarkerSize = 0.045f;

    [Header("Safety")]
    [SerializeField, Min(500)] private int maxRuntimeEdges = 24000;
    [SerializeField] private bool showPerfHud = true;
    [SerializeField] private bool debugLog;

    private readonly List<MeshRenderer> _stepRenderers = new List<MeshRenderer>(35);
    private readonly List<ObserverSample> _observerSamples = new List<ObserverSample>(128);
    private readonly List<LineRecord>[] _linesByStep = new List<LineRecord>[35];
    private readonly List<LineRecord> _allLines = new List<LineRecord>(16384);
    private readonly List<int> _revealedLineIndices = new List<int>(16384);

    private GameObject _contentRoot;
    private GameObject _observerMarker;
    private Mesh _headsetRevealMesh;
    private MeshRenderer _headsetRevealRenderer;
    private Material _lineMaterial;
    private Material _observerMaterial;
    private MaterialPropertyBlock _linePropertyBlock;
    private TextMesh _perfHud;
    private Transform _perfHudTransform;
    private float _perfHudTimer;
    private float _smoothedDeltaTime;
    private int[] _cumulativeEdgesByStep = Array.Empty<int>();
    private float _playTime;
    private int _frameCount = 35;
    private int _currentStep = -1;
    private bool[] _headsetRevealed;
    private bool _alignedToInitialHeadset;
    private bool _isReady;
    private bool _isPlaying;

    private struct LineRecord
    {
        public Vector3 a;
        public Vector3 b;
    }

    private struct ObserverSample
    {
        public Vector3 position;
        public Vector3 forward;
    }

    private void Awake()
    {
        LoadAndBuild();
    }

    private void Start()
    {
        if (playOnStart)
            Play();
        else
            SetInitialVisibleState();
    }

    private void OnEnable()
    {
        if (_isReady && playOnStart)
            Play();
    }

    private void Update()
    {
        if (!_isReady || !_isPlaying)
            return;

        TryAlignToInitialHeadset();

        if (growthMode == ScanGrowthMode.HeadsetDriven)
        {
            UpdateHeadsetDrivenReveal();
            UpdatePerfHud(HeadsetDrivenStepForHud());
            return;
        }

        _playTime += Time.deltaTime;
        float duration = Mathf.Max(0.1f, playbackSeconds);
        if (_playTime > duration)
        {
            if (loop)
                _playTime %= duration;
            else
            {
                _playTime = duration;
                _isPlaying = false;
            }
        }

        int stepLimit = Mathf.Clamp(maxVisibleStep, 1, _frameCount);
        int step = Mathf.Clamp(Mathf.FloorToInt((_playTime / duration) * stepLimit), 0, stepLimit - 1);
        SetStep(step);
        UpdatePerfHud(step);
    }

    private void OnDestroy()
    {
        DestroyRuntimeObject(_lineMaterial);
        DestroyRuntimeObject(_observerMaterial);
        DestroyRuntimeObject(_headsetRevealMesh);
    }

    [ContextMenu("Play")]
    public void Play()
    {
        _isPlaying = true;
        if (_playTime >= playbackSeconds)
            _playTime = 0f;
    }

    [ContextMenu("Restart Playback")]
    public void RestartPlayback()
    {
        _playTime = 0f;
        _isPlaying = true;
        if (growthMode == ScanGrowthMode.HeadsetDriven)
            ResetHeadsetReveal();
        else
            SetStep(0);
    }

    [ContextMenu("Show Complete Scan")]
    public void ShowCompleteScan()
    {
        _isPlaying = false;
        _playTime = playbackSeconds;
        if (growthMode == ScanGrowthMode.HeadsetDriven)
            RevealAllHeadsetLines();
        else
            SetStep(Mathf.Clamp(maxVisibleStep, 1, _frameCount) - 1);
    }

    private void LoadAndBuild()
    {
        for (int i = 0; i < _linesByStep.Length; i++)
            _linesByStep[i] = new List<LineRecord>(512);

        TextAsset asset = scanLineAsset;
        if (asset == null && !string.IsNullOrEmpty(fallbackResourcePath))
            asset = Resources.Load<TextAsset>(fallbackResourcePath);

        if (asset == null)
        {
            Debug.LogWarning($"[ScanCoverQuestSpatialScanPlayback] Missing scan asset: {fallbackResourcePath}", this);
            return;
        }

        ParseAsset(asset.text);
        EnsureMaterials();
        EnsureContentRoot();
        if (growthMode == ScanGrowthMode.HeadsetDriven)
            BuildHeadsetDrivenMesh();
        else
        {
            BuildStepMeshes();
            BuildObserverMarker();
        }
        BuildCumulativeCounts();
        BuildPerfHud();
        SetInitialVisibleState();
        _isReady = true;

        if (debugLog)
            Debug.Log($"[ScanCoverQuestSpatialScanPlayback] ready frames={_frameCount}, edges={CountLoadedEdges()}, observers={_observerSamples.Count}", this);
    }

    private void ParseAsset(string text)
    {
        int loadedEdges = 0;
        using (StringReader reader = new StringReader(text))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line) || line[0] == '#')
                    continue;

                string[] parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0)
                    continue;

                if (parts[0] == "frames" && parts.Length >= 2)
                {
                    _frameCount = Mathf.Clamp(ParseInt(parts[1], 35), 1, 35);
                    continue;
                }

                if (parts[0] == "observer" && parts.Length >= 7)
                {
                    _observerSamples.Add(new ObserverSample
                    {
                        position = TransformLocal(ParseVector(parts, 1)),
                        forward = ParseVector(parts, 4).normalized
                    });
                    continue;
                }

                if (!char.IsDigit(parts[0][0]) && parts[0][0] != '-')
                    continue;

                if (parts.Length < 7 || loadedEdges >= maxRuntimeEdges)
                    continue;

                int step = Mathf.Clamp(ParseInt(parts[0], 0), 0, _linesByStep.Length - 1);
                var record = new LineRecord
                {
                    a = TransformLocal(ParseVector(parts, 1)),
                    b = TransformLocal(ParseVector(parts, 4))
                };
                _linesByStep[step].Add(record);
                _allLines.Add(record);
                loadedEdges++;
            }
        }
    }

    private Vector3 TransformLocal(Vector3 p)
    {
        if (keepFloorAtParentY)
            p.y = Mathf.Max(0f, p.y);
        return localOffset + p * localScale;
    }

    private static Vector3 ParseVector(string[] parts, int index)
    {
        return new Vector3(
            ParseFloat(parts[index]),
            ParseFloat(parts[index + 1]),
            ParseFloat(parts[index + 2]));
    }

    private static float ParseFloat(string value)
    {
        return float.Parse(value, CultureInfo.InvariantCulture);
    }

    private static int ParseInt(string value, int fallback)
    {
        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result) ? result : fallback;
    }

    private void EnsureContentRoot()
    {
        if (_contentRoot != null)
            return;

        _contentRoot = new GameObject("ScanCoverQuestSpatialScanPlayback_Runtime");
        _contentRoot.transform.SetParent(transform, false);
        _contentRoot.transform.localPosition = Vector3.zero;
        _contentRoot.transform.localRotation = Quaternion.identity;
        _contentRoot.transform.localScale = Vector3.one;
    }

    private void EnsureMaterials()
    {
        if (_lineMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            _lineMaterial = new Material(shader)
            {
                name = "ScanCoverQuestSpatialScanPlayback_Line_Mat"
            };
            ApplyMaterialColor(_lineMaterial, baseLineColor);
        }

        if (_observerMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            _observerMaterial = new Material(shader)
            {
                name = "ScanCoverQuestSpatialScanPlayback_Observer_Mat"
            };
            ApplyMaterialColor(_observerMaterial, observerColor);
        }

        if (_linePropertyBlock == null)
            _linePropertyBlock = new MaterialPropertyBlock();
    }

    private static void ApplyMaterialColor(Material material, Color color)
    {
        if (material == null)
            return;
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        if (material.HasProperty("_Surface"))
            material.SetFloat("_Surface", 1f);
        if (material.HasProperty("_SrcBlend"))
            material.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (material.HasProperty("_DstBlend"))
            material.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        if (material.HasProperty("_ZWrite"))
            material.SetFloat("_ZWrite", 0f);
        material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.renderQueue = (int)RenderQueue.Transparent;
    }

    private void BuildStepMeshes()
    {
        _stepRenderers.Clear();
        for (int step = 0; step < _frameCount; step++)
        {
            List<LineRecord> lines = _linesByStep[step];
            GameObject child = new GameObject($"ScanCoverSpatialScanStep_{step:00}");
            child.transform.SetParent(_contentRoot.transform, false);

            Mesh mesh = new Mesh
            {
                name = $"ScanCoverSpatialScanStep_{step:00}_Mesh"
            };
            mesh.MarkDynamic();

            Vector3[] vertices = new Vector3[lines.Count * 2];
            int[] indices = new int[lines.Count * 2];
            for (int i = 0; i < lines.Count; i++)
            {
                int vi = i * 2;
                vertices[vi] = lines[i].a;
                vertices[vi + 1] = lines[i].b;
                indices[vi] = vi;
                indices[vi + 1] = vi + 1;
            }

            mesh.vertices = vertices;
            mesh.SetIndices(indices, MeshTopology.Lines, 0, true);
            mesh.RecalculateBounds();

            MeshFilter filter = child.AddComponent<MeshFilter>();
            filter.sharedMesh = mesh;
            MeshRenderer renderer = child.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = _lineMaterial;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            ApplyRendererColor(renderer, baseLineColor);
            _stepRenderers.Add(renderer);
        }
    }

    private void BuildObserverMarker()
    {
        if (!showObserver)
            return;

        _observerMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        _observerMarker.name = "ScanCoverSpatialScanObserver";
        _observerMarker.transform.SetParent(_contentRoot.transform, false);
        _observerMarker.transform.localScale = Vector3.one * observerMarkerSize;

        Collider collider = _observerMarker.GetComponent<Collider>();
        if (collider != null)
            DestroyRuntimeObject(collider);

        MeshRenderer renderer = _observerMarker.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = _observerMaterial;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }
    }

    private void BuildHeadsetDrivenMesh()
    {
        _headsetRevealed = new bool[_allLines.Count];
        _revealedLineIndices.Clear();

        GameObject child = new GameObject("ScanCoverSpatialScanHeadsetDrivenLines");
        child.transform.SetParent(_contentRoot.transform, false);

        _headsetRevealMesh = new Mesh
        {
            name = "ScanCoverSpatialScanHeadsetDrivenLines_Mesh",
            indexFormat = IndexFormat.UInt32
        };
        _headsetRevealMesh.MarkDynamic();

        MeshFilter filter = child.AddComponent<MeshFilter>();
        filter.sharedMesh = _headsetRevealMesh;
        _headsetRevealRenderer = child.AddComponent<MeshRenderer>();
        _headsetRevealRenderer.sharedMaterial = _lineMaterial;
        _headsetRevealRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _headsetRevealRenderer.receiveShadows = false;
        ApplyRendererColor(_headsetRevealRenderer, hotLineColor);
        RebuildHeadsetRevealMesh();
    }

    private void SetInitialVisibleState()
    {
        if (growthMode == ScanGrowthMode.HeadsetDriven)
        {
            ResetHeadsetReveal();
            return;
        }

        SetStep(0);
    }

    private void TryAlignToInitialHeadset()
    {
        if (!alignRoomCenterToInitialHeadset || _alignedToInitialHeadset)
            return;

        Camera camera = Camera.main;
        if (camera == null)
            return;

        Vector3 cameraPosition = camera.transform.position;
        transform.position = new Vector3(cameraPosition.x, 0f, cameraPosition.z) + roomCenterOffsetFromInitialHeadset;
        _alignedToInitialHeadset = true;
    }

    private void UpdateHeadsetDrivenReveal()
    {
        Camera camera = Camera.main;
        if (camera == null || _headsetRevealed == null || _allLines.Count == 0)
            return;

        Transform cameraTransform = camera.transform;
        float tanH = Mathf.Tan(headsetHorizontalFovDegrees * 0.5f * Mathf.Deg2Rad);
        float tanV = Mathf.Tan(headsetVerticalFovDegrees * 0.5f * Mathf.Deg2Rad);
        float maxDistance = Mathf.Max(0.1f, headsetRevealMaxDistance);
        int budget = Mathf.Max(1, headsetRevealBudgetPerFrame);
        int added = 0;

        for (int i = 0; i < _allLines.Count && added < budget; i++)
        {
            if (_headsetRevealed[i])
                continue;

            LineRecord line = _allLines[i];
            Vector3 midLocal = (line.a + line.b) * 0.5f;
            Vector3 midWorld = _contentRoot.transform.TransformPoint(midLocal);
            if (!IsVisibleFromCamera(cameraTransform, midWorld, tanH, tanV, maxDistance))
                continue;

            _headsetRevealed[i] = true;
            _revealedLineIndices.Add(i);
            added++;
        }

        if (added > 0)
            RebuildHeadsetRevealMesh();
    }

    private bool IsVisibleFromCamera(Transform cameraTransform, Vector3 worldPoint, float tanH, float tanV, float maxDistance)
    {
        Vector3 cameraLocal = cameraTransform.InverseTransformPoint(worldPoint);
        if (cameraLocal.z <= 0.05f || cameraLocal.z > maxDistance)
            return false;

        float forwardNormalized = Mathf.Clamp01(cameraLocal.z / maxDistance);
        float edgeAllowance = Mathf.Lerp(0.65f, 1f, forwardNormalized + headsetForwardBias);
        float xLimit = cameraLocal.z * tanH * edgeAllowance;
        float yLimit = cameraLocal.z * tanV * edgeAllowance;
        return Mathf.Abs(cameraLocal.x) <= xLimit && Mathf.Abs(cameraLocal.y) <= yLimit;
    }

    private void RebuildHeadsetRevealMesh()
    {
        if (_headsetRevealMesh == null)
            return;

        int lineCount = _revealedLineIndices.Count;
        Vector3[] vertices = new Vector3[lineCount * 2];
        int[] indices = new int[lineCount * 2];
        for (int i = 0; i < lineCount; i++)
        {
            LineRecord line = _allLines[_revealedLineIndices[i]];
            int vi = i * 2;
            vertices[vi] = line.a;
            vertices[vi + 1] = line.b;
            indices[vi] = vi;
            indices[vi + 1] = vi + 1;
        }

        _headsetRevealMesh.Clear();
        _headsetRevealMesh.vertices = vertices;
        _headsetRevealMesh.SetIndices(indices, MeshTopology.Lines, 0, true);
        _headsetRevealMesh.RecalculateBounds();
    }

    private void ResetHeadsetReveal()
    {
        if (_headsetRevealed == null)
            _headsetRevealed = new bool[_allLines.Count];
        else
            Array.Clear(_headsetRevealed, 0, _headsetRevealed.Length);

        _revealedLineIndices.Clear();
        RebuildHeadsetRevealMesh();
        _currentStep = -1;
    }

    private void RevealAllHeadsetLines()
    {
        if (_headsetRevealed == null)
            _headsetRevealed = new bool[_allLines.Count];
        _revealedLineIndices.Clear();
        for (int i = 0; i < _allLines.Count; i++)
        {
            _headsetRevealed[i] = true;
            _revealedLineIndices.Add(i);
        }

        RebuildHeadsetRevealMesh();
    }

    private int HeadsetDrivenStepForHud()
    {
        if (_allLines.Count <= 0)
            return 0;
        float t = _revealedLineIndices.Count / Mathf.Max(1f, _allLines.Count);
        return Mathf.Clamp(Mathf.RoundToInt(t * Mathf.Max(0, _frameCount - 1)), 0, Mathf.Max(0, _frameCount - 1));
    }

    private void SetStep(int step)
    {
        if (step == _currentStep)
            return;

        _currentStep = step;
        for (int i = 0; i < _stepRenderers.Count; i++)
        {
            MeshRenderer renderer = _stepRenderers[i];
            if (renderer != null)
            {
                renderer.enabled = i <= step;
                ApplyRendererColor(renderer, i == step ? hotLineColor : baseLineColor);
            }
        }

        UpdateObserver(step);
    }

    private void UpdateObserver(int step)
    {
        if (_observerMarker == null || _observerSamples.Count == 0)
            return;

        int index = Mathf.Clamp(
            Mathf.RoundToInt((step / Mathf.Max(1f, _frameCount - 1f)) * (_observerSamples.Count - 1)),
            0,
            _observerSamples.Count - 1);
        ObserverSample sample = _observerSamples[index];
        _observerMarker.transform.localPosition = sample.position;
        if (sample.forward.sqrMagnitude > 1e-5f)
            _observerMarker.transform.localRotation = Quaternion.LookRotation(sample.forward, Vector3.up);
    }

    private void ApplyRendererColor(Renderer renderer, Color color)
    {
        if (renderer == null)
            return;
        if (_linePropertyBlock == null)
            _linePropertyBlock = new MaterialPropertyBlock();
        renderer.GetPropertyBlock(_linePropertyBlock);
        _linePropertyBlock.SetColor("_BaseColor", color);
        _linePropertyBlock.SetColor("_Color", color);
        renderer.SetPropertyBlock(_linePropertyBlock);
    }

    private void BuildCumulativeCounts()
    {
        _cumulativeEdgesByStep = new int[Mathf.Max(1, _frameCount)];
        int total = 0;
        for (int i = 0; i < _cumulativeEdgesByStep.Length; i++)
        {
            if (i < _linesByStep.Length && _linesByStep[i] != null)
                total += _linesByStep[i].Count;
            _cumulativeEdgesByStep[i] = total;
        }
    }

    private void BuildPerfHud()
    {
        if (!showPerfHud)
            return;

        GameObject hudObject = new GameObject("ScanCoverSpatialScanPlaybackHud");
        hudObject.transform.SetParent(_contentRoot != null ? _contentRoot.transform : transform, false);
        hudObject.transform.localPosition = new Vector3(-1.05f, 1.35f, 0.55f);
        hudObject.transform.localRotation = Quaternion.identity;
        hudObject.transform.localScale = Vector3.one * 0.035f;

        _perfHud = hudObject.AddComponent<TextMesh>();
        _perfHudTransform = hudObject.transform;
        _perfHud.anchor = TextAnchor.UpperLeft;
        _perfHud.alignment = TextAlignment.Left;
        _perfHud.fontSize = 42;
        _perfHud.characterSize = 0.1f;
        _perfHud.color = new Color(0.82f, 1f, 1f, 0.9f);
        _perfHud.text = "ScanCover Spatial Scan";
    }

    private void UpdatePerfHud(int step)
    {
        _smoothedDeltaTime = Mathf.Lerp(
            _smoothedDeltaTime <= 0f ? Time.unscaledDeltaTime : _smoothedDeltaTime,
            Time.unscaledDeltaTime,
            0.08f);

        if (_perfHud == null || !showPerfHud)
            return;

        _perfHudTimer += Time.unscaledDeltaTime;
        if (_perfHudTimer < 0.2f)
            return;

        _perfHudTimer = 0f;
        int safeStep = Mathf.Clamp(step, 0, Mathf.Max(0, _frameCount - 1));
        int visibleEdges = growthMode == ScanGrowthMode.HeadsetDriven
            ? _revealedLineIndices.Count
            : (_cumulativeEdgesByStep != null && _cumulativeEdgesByStep.Length > 0
                ? _cumulativeEdgesByStep[Mathf.Clamp(safeStep, 0, _cumulativeEdgesByStep.Length - 1)]
                : CountLoadedEdges());
        float fps = _smoothedDeltaTime > 0.0001f ? 1f / _smoothedDeltaTime : 0f;
        _perfHud.text =
            $"ScanCover Spatial Scan\n" +
            $"{growthMode}\n" +
            $"Step {safeStep + 1:00}/{_frameCount:00}\n" +
            $"Edges {visibleEdges}/{CountLoadedEdges()}\n" +
            $"FPS {fps:0}";

        Camera camera = Camera.main;
        if (camera != null && _perfHudTransform != null)
            _perfHudTransform.rotation = Quaternion.LookRotation(_perfHudTransform.position - camera.transform.position, Vector3.up);
    }

    private int CountLoadedEdges()
    {
        if (growthMode == ScanGrowthMode.HeadsetDriven && _allLines.Count > 0)
            return _allLines.Count;

        int count = 0;
        for (int i = 0; i < _linesByStep.Length; i++)
            count += _linesByStep[i] != null ? _linesByStep[i].Count : 0;
        return count;
    }

    private static void DestroyRuntimeObject(UnityEngine.Object obj)
    {
        if (obj == null)
            return;
        if (Application.isPlaying)
            Destroy(obj);
        else
            DestroyImmediate(obj);
    }
}

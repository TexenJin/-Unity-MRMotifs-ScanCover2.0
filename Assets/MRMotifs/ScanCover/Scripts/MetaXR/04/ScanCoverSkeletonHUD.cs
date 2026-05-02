using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class ScanCoverSkeletonHUD : MonoBehaviour
{
    [Header("Refs")]
    public ScanCoverSkeletonBuilder_A builder;
    public ScanCoverSkeletonMesher_B mesher;
    public ScanCoverSkeletonSessionController session;
    public Camera hudCamera;

    [Header("HUD Placement")]
    public bool createWorldSpaceHUD = true;
    public Vector3 localPosition = new Vector3(0f, -0.12f, 0.7f);
    public Vector3 localEuler = Vector3.zero;
    public Vector3 localScale = new Vector3(0.0008f, 0.0008f, 0.0008f);
    public int panelWidth = 560;
    public int panelHeight = 300;

    [Header("HUD Refresh")]
    [Min(0.05f)] public float refreshInterval = 0.20f;
    public bool showInputHelp = true;
    [Tooltip("极简模式：仅显示关键统计，减少占视野")]
    public bool minimalMode = true;

    [Header("Sampling Bounds UI")]
    [Tooltip("Show projected sampling bounds on HUD")]
    public bool showSamplingBoundsBox = true;
    [Tooltip("Show bounds box even when filter is disabled")]
    public bool showBoundsWhenFilterDisabled = true;
    public Color samplingBoundsBoxColor = new Color(0.2f, 0.85f, 1f, 0f);
    public Color samplingBoundsBoxBorderColor = new Color(0.2f, 0.85f, 1f, 0.8f);
    [Min(1f)] public float minBoundsBoxPixels = 18f;
    [Min(1f)] public float crosshairThickness = 2.0f;
    [Range(0.3f, 1f)] public float maxViewportFill = 0.90f;

    private Canvas _canvas;
    private GameObject _root;
    private RectTransform _panel;
    private Text _text;
    private Image _panelImage;
    private RectTransform _boundsRect;
    private Image _boundsImage;
    private readonly RectTransform[] _boundsEdges = new RectTransform[4];
    private float _nextRefresh;

    private void Awake()
    {
        if (!builder) builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (!mesher) mesher = GetComponent<ScanCoverSkeletonMesher_B>();
        if (!session) session = GetComponent<ScanCoverSkeletonSessionController>();
        if (!hudCamera && builder && builder.sampleCamera) hudCamera = builder.sampleCamera;
        if (!hudCamera && Camera.main) hudCamera = Camera.main;
        CleanupOrphanedHUDRoots();
    }

    private void OnEnable()
    {
        CleanupOrphanedHUDRoots();
        EnsureHUD();
        SetHUDActive(true);
        _nextRefresh = 0f;
    }

    private void OnDisable()
    {
        SetHUDActive(false);
    }

    private void OnDestroy()
    {
        if (_root != null)
        {
            if (Application.isPlaying)
                Destroy(_root);
            else
                DestroyImmediate(_root);
        }
    }

    private void LateUpdate()
    {
        if (!_canvas || !_text) EnsureHUD();
        if (_canvas && createWorldSpaceHUD && hudCamera)
        {
            Transform t = _canvas.transform;
            t.SetParent(hudCamera.transform, false);
            t.localPosition = localPosition;
            t.localEulerAngles = localEuler;
            t.localScale = localScale;
        }

        UpdateSamplingBoundsBoxUI();

        if (Time.time < _nextRefresh) return;
        _nextRefresh = Time.time + Mathf.Max(0.05f, refreshInterval);
        RefreshText();
    }

    private void EnsureHUD()
    {
        if (_canvas) return;

        GameObject root = new GameObject("[ScanCover] SkeletonHUD", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler));
        root.transform.SetParent(transform, false);
        root.hideFlags = HideFlags.DontSave;
        _root = root;
        _canvas = root.GetComponent<Canvas>();
        _canvas.renderMode = RenderMode.WorldSpace;
        _canvas.worldCamera = hudCamera;

        var scaler = root.GetComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;
        scaler.referencePixelsPerUnit = 100f;

        GameObject panelGO = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelGO.transform.SetParent(root.transform, false);
        _panel = panelGO.GetComponent<RectTransform>();
        _panel.sizeDelta = new Vector2(panelWidth, panelHeight);
        _panel.anchorMin = new Vector2(0.5f, 0.5f);
        _panel.anchorMax = new Vector2(0.5f, 0.5f);
        _panel.pivot = new Vector2(0.5f, 0.5f);
        _panelImage = panelGO.GetComponent<Image>();
        _panelImage.color = new Color(0f, 0f, 0f, 0.65f);
        _panelImage.raycastTarget = false;

        GameObject textGO = new GameObject("Text", typeof(RectTransform), typeof(Text));
        textGO.transform.SetParent(panelGO.transform, false);
        var rt = textGO.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0f, 0f);
        rt.anchorMax = new Vector2(1f, 1f);
        rt.offsetMin = new Vector2(20f, 20f);
        rt.offsetMax = new Vector2(-20f, -20f);

        _text = textGO.GetComponent<Text>();
        _text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (_text.font == null)
            _text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        _text.fontSize = 28;
        _text.alignment = TextAnchor.UpperLeft;
        _text.horizontalOverflow = HorizontalWrapMode.Wrap;
        _text.verticalOverflow = VerticalWrapMode.Overflow;
        _text.color = new Color(0.82f, 1f, 1f, 1f);
        _text.text = "Skeleton HUD";
        _text.raycastTarget = false;

        if (showSamplingBoundsBox)
        {
            GameObject boundsGO = new GameObject("SamplingBoundsBox", typeof(RectTransform), typeof(Image));
            boundsGO.transform.SetParent(panelGO.transform, false);
            _boundsRect = boundsGO.GetComponent<RectTransform>();
            _boundsRect.anchorMin = new Vector2(0.5f, 0.5f);
            _boundsRect.anchorMax = new Vector2(0.5f, 0.5f);
            _boundsRect.pivot = new Vector2(0.5f, 0.5f);
            _boundsRect.sizeDelta = new Vector2(140f, 100f);
            _boundsImage = boundsGO.GetComponent<Image>();
            _boundsImage.color = samplingBoundsBoxColor;
            _boundsImage.raycastTarget = false;
            for (int i = 0; i < 4; i++)
            {
                GameObject edgeGO = new GameObject("Edge" + i, typeof(RectTransform), typeof(Image));
                edgeGO.transform.SetParent(boundsGO.transform, false);
                RectTransform edge = edgeGO.GetComponent<RectTransform>();
                edge.anchorMin = new Vector2(0.5f, 0.5f);
                edge.anchorMax = new Vector2(0.5f, 0.5f);
                edge.pivot = new Vector2(0.5f, 0.5f);
                Image edgeImage = edgeGO.GetComponent<Image>();
                edgeImage.color = samplingBoundsBoxBorderColor;
                edgeImage.raycastTarget = false;
                _boundsEdges[i] = edge;
            }

            boundsGO.SetActive(false);
        }
    }

    private void SetHUDActive(bool active)
    {
        if (_root != null && _root.activeSelf != active)
            _root.SetActive(active);
    }

    private void CleanupOrphanedHUDRoots()
    {
        CleanupNamedChildren(transform, "[ScanCover] SkeletonHUD");
        if (hudCamera != null)
            CleanupNamedChildren(hudCamera.transform, "[ScanCover] SkeletonHUD");
    }

    private void CleanupNamedChildren(Transform parent, string childName)
    {
        if (parent == null) return;

        for (int i = parent.childCount - 1; i >= 0; i--)
        {
            Transform child = parent.GetChild(i);
            if (child == null || child.gameObject == null) continue;
            if (!string.Equals(child.name, childName, System.StringComparison.Ordinal)) continue;
            if (_root != null && child.gameObject == _root) continue;

            if (Application.isPlaying)
                Destroy(child.gameObject);
            else
                DestroyImmediate(child.gameObject);
        }
    }

    private void RefreshText()
    {
        if (!_text) return;

        string state = session ? session.GetStateLabel() : (builder ? (builder.scanEnabled ? "Scanning" : "Frozen") : "N/A");
        ScanCoverSkeletonBuilder_A.SummaryStats stats = builder ? builder.GetSummaryStats() : default;
        int chunkTotal = 0, chunkRendered = 0, chunkColliders = 0;
        if (mesher) mesher.GetChunkStats(out chunkTotal, out chunkRendered, out chunkColliders);

        bool ready = session ? session.CurrentTileReady : stats.ready;
        string readyText = ready ? "READY" : "NOT READY";
        _text.color = ready ? new Color(0.45f, 1f, 0.55f, 1f) : new Color(1f, 0.82f, 0.45f, 1f);

        float summaryGrowthRatio = (stats.confirmedCells > 0) ? (100f * stats.recentConfirmedCells / Mathf.Max(1, stats.confirmedCells)) : 0f;
        float coverageScore = CalcCoverageScore(stats);
        float stabilityScore = CalcStabilityScore(stats);
        float holeFillScore = CalcHoleFillScore(stats);
        float meshScore = CalcMeshScore(chunkTotal, chunkRendered, chunkColliders);
        float currentTileGrowthPct = session ? (session.CurrentTileRecentGrowthRatio * 100f) : 0f;

        if (minimalMode)
        {
            _text.text =
                "ScanCover 4-Score\n" +
                $"Coverage  : {coverageScore:0}\n" +
                $"Stability : {stabilityScore:0}\n" +
                $"Hole Fill : {holeFillScore:0}\n" +
                $"Mesh      : {meshScore:0}\n" +
                $"State {state} / {readyText}";

            if (showInputHelp)
                _text.text += "\n1 Start  2 Freeze  3 Clear  4 Toggle";
        }
        else
        {
            _text.text =
                "ScanCover Auto-Commit HUD\n" +
                "------------------------------\n" +
                $"State                : {state}\n" +
                $"Current Tile Ready   : {readyText}\n" +
                "\n" +
                $"Session Cells Total  : {stats.totalCells}\n" +
                $"Session Confirmed    : {stats.confirmedCells}\n" +
                $"Session New Confirm. : {stats.recentConfirmedCells} ({summaryGrowthRatio:0.0}%)\n" +
                $"Coverage Score       : {coverageScore:0}\n" +
                $"Stability Score      : {stabilityScore:0}\n" +
                $"HoleFill Score       : {holeFillScore:0}\n" +
                $"Mesh Score           : {meshScore:0}\n" +
                "\n" +
                $"Current Tile         : {(session && session.HasCurrentTile ? session.CurrentTile.ToString() : "N/A")}\n" +
                $"Current Tile Cells   : {(session ? session.CurrentTileTotalCells : 0)}\n" +
                $"Current Tile Confirm.: {(session ? session.CurrentTileConfirmedCells : 0)}\n" +
                $"Current Tile New Conf: {(session ? session.CurrentTileRecentConfirmedCells : 0)} ({currentTileGrowthPct:0.0}%)\n" +
                $"Stable Uncommitted   : {(session ? session.StableUncommittedTileCount : 0)}\n" +
                $"Committed Tiles      : {(session ? session.CommittedTileCount : 0)}\n" +
                "\n" +
                $"Frozen Chunks        : {chunkTotal}\n" +
                $"Renderers On         : {chunkRendered}\n" +
                $"Colliders On         : {chunkColliders}\n" +
                "\n" +
                $"Scan Enabled         : {(builder && builder.scanEnabled ? "Yes" : "No")}\n" +
                $"Live Build           : {(mesher && mesher.liveBuild ? "On" : "Off")}\n";

            if (showInputHelp)
            {
                _text.text +=
                    "\n" +
                    "Keys\n" +
                    "1 : Start New Scan\n" +
                    "2 : Freeze + Build Stable Skeleton\n" +
                    "3 : Clear All\n" +
                    "4 : Toggle Frozen Renderer\n";
            }
        }

        if (_panelImage)
            _panelImage.color = new Color(0f, 0f, 0f, 0.58f);
    }

    private void UpdateSamplingBoundsBoxUI()
    {
        if (_boundsRect == null || _panel == null || builder == null)
            return;

        bool filterOn = builder.enableSamplingBoundsFilter;
        bool shouldShow = showSamplingBoundsBox && (filterOn || showBoundsWhenFilterDisabled);
        if (!shouldShow)
        {
            if (_boundsRect.gameObject.activeSelf) _boundsRect.gameObject.SetActive(false);
            return;
        }

        Vector2 center = new Vector2(
            Mathf.Clamp01(builder.samplingViewportCenter.x),
            Mathf.Clamp01(builder.samplingViewportCenter.y));
        float fillLimit = Mathf.Clamp(maxViewportFill, 0.3f, 1f);
        if (builder != null)
            fillLimit = Mathf.Min(fillLimit, Mathf.Clamp(builder.samplingViewportMaxSize, 0.2f, 1f));
        Vector2 size = new Vector2(
            Mathf.Clamp(builder.samplingViewportSize.x, 0.01f, fillLimit),
            Mathf.Clamp(builder.samplingViewportSize.y, 0.01f, fillLimit));

        float panelW = Mathf.Max(1f, _panel.rect.width);
        float panelH = Mathf.Max(1f, _panel.rect.height);
        float w = Mathf.Max(minBoundsBoxPixels, size.x * panelW);
        float h = Mathf.Max(minBoundsBoxPixels, size.y * panelH);
        float cx = (center.x - 0.5f) * panelW;
        float cy = (center.y - 0.5f) * panelH;

        _boundsRect.anchoredPosition = new Vector2(cx, cy);
        _boundsRect.sizeDelta = new Vector2(w, h);
        if (!_boundsRect.gameObject.activeSelf) _boundsRect.gameObject.SetActive(true);

        if (_boundsImage != null)
            _boundsImage.color = samplingBoundsBoxColor;

        float t = Mathf.Max(1f, crosshairThickness);
        UpdateEdge(0, cx, cy + (h * 0.5f), w, t); // Top
        UpdateEdge(1, cx, cy - (h * 0.5f), w, t); // Bottom
        UpdateEdge(2, cx - (w * 0.5f), cy, t, h); // Left
        UpdateEdge(3, cx + (w * 0.5f), cy, t, h); // Right
    }

    private void UpdateEdge(int idx, float x, float y, float w, float h)
    {
        if (idx < 0 || idx >= _boundsEdges.Length) return;
        RectTransform rt = _boundsEdges[idx];
        if (rt == null) return;
        rt.anchoredPosition = new Vector2(x, y);
        rt.sizeDelta = new Vector2(w, h);
        rt.localRotation = Quaternion.identity;

        Image img = rt.GetComponent<Image>();
        if (img != null) img.color = samplingBoundsBoxBorderColor;
    }

    private float CalcCoverageScore(ScanCoverSkeletonBuilder_A.SummaryStats stats)
    {
        int target = (builder != null) ? Mathf.Max(1, builder.readinessMinConfirmedCells) : 200;
        return Mathf.Clamp01((float)stats.confirmedCells / target) * 100f;
    }

    private float CalcStabilityScore(ScanCoverSkeletonBuilder_A.SummaryStats stats)
    {
        float growthRatio = (stats.confirmedCells > 0) ? ((float)stats.recentConfirmedCells / Mathf.Max(1, stats.confirmedCells)) : 1f;
        float threshold = (builder != null) ? Mathf.Max(0.01f, builder.readinessMaxRecentGrowthRatio) : 0.12f;
        float normalized = Mathf.Clamp01(growthRatio / threshold);
        return (1f - normalized) * 100f;
    }

    private float CalcHoleFillScore(ScanCoverSkeletonBuilder_A.SummaryStats stats)
    {
        bool fillEnabled = mesher != null && mesher.enableHoleFill;
        float baseScore = fillEnabled ? 70f : 40f;
        float growthRatio = (stats.confirmedCells > 0) ? ((float)stats.recentConfirmedCells / Mathf.Max(1, stats.confirmedCells)) : 1f;
        float stabilityBonus = (1f - Mathf.Clamp01(growthRatio / 0.25f)) * 30f;
        return Mathf.Clamp(baseScore + stabilityBonus, 0f, 100f);
    }

    private float CalcMeshScore(int chunkTotal, int chunkRendered, int chunkColliders)
    {
        if (chunkTotal <= 0)
            return 0f;
        float renderRatio = Mathf.Clamp01((float)chunkRendered / chunkTotal);
        float colliderRatio = (mesher != null && !mesher.buildCollider) ? 1f : Mathf.Clamp01((float)chunkColliders / chunkTotal);
        return renderRatio * colliderRatio * 100f;
    }
}

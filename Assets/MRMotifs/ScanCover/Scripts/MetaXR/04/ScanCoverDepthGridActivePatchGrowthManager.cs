using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverDepthGridActivePatchGrowthManager : MonoBehaviour
{
    public enum GrowthSide
    {
        Left,
        Right,
        Top,
        Bottom
    }

    private sealed class BandVisual
    {
        public GrowthSide side;
        public ScanCoverDepthGridPointCloud.GridStateEntry[] entries;
        public GameObject root;
        public Mesh snapshotMesh;
        public MeshFilter snapshotMeshFilter;
        public MeshRenderer snapshotMeshRenderer;
        public Mesh lineMesh;
        public MeshFilter lineFilter;
        public MeshRenderer lineRenderer;
        public readonly List<GameObject> markers = new List<GameObject>(128);
    }

    [Header("Refs")]
    public ScanCoverDepthGridPointCloud depthGridPointCloud;

    [Header("Flow")]
    public bool showBaseGrid = true;
    public bool showGrowthBands = true;
    [Min(0.001f)] public float markerScaleMeters = 0.012f;
    [Min(0.0005f)] public float lineWidthMeters = 0.004f;
    public bool debugLog = false;

    [Header("Display")]
    public Material baseMaterialOverride;
    public Material growthMaterialOverride;
    public Material snapshotMeshMaterialOverride;
    public Color baseColor = new Color(0.18f, 0.95f, 0.98f, 1f);
    public Color growthColorA = new Color(0.98f, 0.68f, 0.18f, 1f);
    public Color growthColorB = new Color(0.94f, 0.25f, 0.78f, 1f);
    public Color growthColorC = new Color(0.35f, 0.96f, 0.44f, 1f);
    public Color growthColorD = new Color(0.95f, 0.94f, 0.25f, 1f);

    public bool HasBaseShell => _baseSnapshot != null && _baseSnapshot.entries != null && _baseSnapshot.entries.Length > 0;
    public string LastIssue { get; private set; }

    private GameObject _root;
    private GameObject _baseRoot;
    private Mesh _baseSnapshotMesh;
    private MeshFilter _baseSnapshotMeshFilter;
    private MeshRenderer _baseSnapshotMeshRenderer;
    private Mesh _baseLineMesh;
    private MeshFilter _baseLineFilter;
    private MeshRenderer _baseLineRenderer;
    private readonly List<GameObject> _baseMarkers = new List<GameObject>(2048);

    private readonly List<BandVisual> _bands = new List<BandVisual>(32);
    private readonly Dictionary<GrowthSide, ScanCoverDepthGridPointCloud.GridStateEntry[]> _frontierBySide =
        new Dictionary<GrowthSide, ScanCoverDepthGridPointCloud.GridStateEntry[]>();
    private readonly Dictionary<GrowthSide, ScanCoverDepthGridPointCloud.GridStateEntry[]> _supportBySide =
        new Dictionary<GrowthSide, ScanCoverDepthGridPointCloud.GridStateEntry[]>();

    private Material _runtimeBaseMaterial;
    private Material _runtimeGrowthMaterial;
    private Material _runtimeSnapshotMeshMaterial;
    private ScanCoverDepthGridPointCloud.GridStateSnapshot _baseSnapshot;
    private int _bandCount;

    private readonly List<Vector3> _lineVerts = new List<Vector3>(4096);
    private readonly List<int> _lineIndices = new List<int>(8192);

    public void EnsureInitialized()
    {
        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (depthGridPointCloud != null)
        {
            depthGridPointCloud.SetKeepSurfaceMeshAvailableWhenHidden(true);
            depthGridPointCloud.SetPreviewSurfaceMeshVisible(false);
        }
        EnsureObjects();
    }

    public bool CaptureOrUpdateFromCurrentGrid()
    {
        EnsureInitialized();
        if (depthGridPointCloud == null)
            return SetIssue("DepthGridPointCloud is missing.");
        if (!depthGridPointCloud.TryGetCurrentGridState(out ScanCoverDepthGridPointCloud.GridStateSnapshot current))
            return SetIssue(depthGridPointCloud.LastIssue ?? "Current grid state is unavailable.");

        if (!HasBaseShell)
        {
            _baseSnapshot = CloneSnapshot(current);
            depthGridPointCloud.TryGetPreviewSurfaceData(out Mesh basePreviewSurfaceMesh, out Transform basePreviewSurfaceTransform);
            BuildBaseVisual(basePreviewSurfaceMesh, basePreviewSurfaceTransform, depthGridPointCloud != null ? depthGridPointCloud.GetPreviewSurfaceMaterial() : null);
            InitializeFrontiers(_baseSnapshot);
            _bandCount = 0;
            LastIssue = null;
            if (debugLog)
                Debug.Log($"[ScanCoverDepthGridActivePatchGrowthManager] Seed base grid visible={_baseSnapshot.visibleCount}");
            return true;
        }

        GrowthSide side = ResolvePreferredGrowthSide(current);
        ScanCoverDepthGridPointCloud.GridStateEntry[] newStrip = ExtractSideStrip(current, side);
        if (newStrip.Length < 2)
            return SetIssue($"Current preview strip on {side} has insufficient valid points.");

        if (!_frontierBySide.TryGetValue(side, out ScanCoverDepthGridPointCloud.GridStateEntry[] previousStrip) || previousStrip == null || previousStrip.Length < 2)
            return SetIssue($"Current frontier on {side} is unavailable.");

        depthGridPointCloud.TryGetPreviewSurfaceData(out Mesh previewSurfaceMesh, out Transform previewSurfaceTransform);
        Material previewSurfaceMaterial = depthGridPointCloud != null ? depthGridPointCloud.GetPreviewSurfaceMaterial() : null;
        _supportBySide.TryGetValue(side, out ScanCoverDepthGridPointCloud.GridStateEntry[] supportStrip);
        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> bandStrips = BuildIntermediateBandStrips(
            side,
            supportStrip,
            previousStrip,
            newStrip,
            previewSurfaceMesh,
            previewSurfaceTransform,
            out float averageRatio,
            out int interiorStripCount);

        Color bandColor = ResolveGrowthBandColor(_bandCount);
        CreateBandVisual(side, previousStrip, bandStrips, bandColor, previewSurfaceMesh, previewSurfaceTransform, previewSurfaceMaterial);
        _supportBySide[side] = CloneEntries(previousStrip);
        _frontierBySide[side] = CloneEntries(newStrip);
        _bandCount++;
        LastIssue = null;

        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridActivePatchGrowthManager] Grow side={side}, strip={newStrip.Length}, interior={interiorStripCount}, ratio={averageRatio:F2}, bandIndex={_bandCount}");

        return true;
    }

    public void ClearAll()
    {
        _baseSnapshot = null;
        _frontierBySide.Clear();
        _supportBySide.Clear();
        _bandCount = 0;
        LastIssue = null;

        ClearMarkerList(_baseMarkers);
        if (_baseLineMesh != null)
            _baseLineMesh.Clear();
        if (_baseRoot != null)
            _baseRoot.SetActive(false);

        for (int i = 0; i < _bands.Count; i++)
            DestroyBandVisual(_bands[i]);
        _bands.Clear();

        if (_root != null)
            _root.SetActive(false);
    }

    private void EnsureObjects()
    {
        if (_root == null)
        {
            _root = new GameObject("[ScanCover] Frozen Grid Growth");
            _root.transform.position = Vector3.zero;
            _root.transform.rotation = Quaternion.identity;
            _root.transform.localScale = Vector3.one;
        }

        if (_baseRoot == null)
        {
            _baseRoot = new GameObject("BaseShell");
            _baseRoot.transform.SetParent(_root.transform, false);

            GameObject snapshotMesh = new GameObject("SnapshotMesh");
            snapshotMesh.transform.SetParent(_baseRoot.transform, false);
            _baseSnapshotMeshFilter = snapshotMesh.AddComponent<MeshFilter>();
            _baseSnapshotMeshRenderer = snapshotMesh.AddComponent<MeshRenderer>();
            _baseSnapshotMesh = new Mesh { name = "ScanCover_BaseSnapshotMesh" };
            _baseSnapshotMeshFilter.sharedMesh = _baseSnapshotMesh;

            GameObject lines = new GameObject("Lines");
            lines.transform.SetParent(_baseRoot.transform, false);
            _baseLineFilter = lines.AddComponent<MeshFilter>();
            _baseLineRenderer = lines.AddComponent<MeshRenderer>();
            _baseLineMesh = new Mesh { name = "ScanCover_BaseGrid_Lines" };
            _baseLineFilter.sharedMesh = _baseLineMesh;

            new GameObject("Markers").transform.SetParent(_baseRoot.transform, false);
        }

        if (_runtimeBaseMaterial == null)
            _runtimeBaseMaterial = CreateRuntimeMaterial("ScanCover_BaseGrid_Mat", baseColor);
        if (_runtimeGrowthMaterial == null)
            _runtimeGrowthMaterial = CreateRuntimeMaterial("ScanCover_GrowthBand_Mat", growthColorA);
        if (_runtimeSnapshotMeshMaterial == null)
            _runtimeSnapshotMeshMaterial = CreateRuntimeMaterial("ScanCover_SnapshotMesh_Mat", Color.white);

        ApplyRendererMaterial(_baseLineRenderer, baseMaterialOverride != null ? baseMaterialOverride : _runtimeBaseMaterial, baseColor);
    }

    private void BuildBaseVisual(Mesh previewSurfaceMesh, Transform previewSurfaceTransform, Material previewSurfaceMaterial)
    {
        EnsureObjects();
        _root.SetActive(true);
        _baseRoot.SetActive(showBaseGrid);
        ApplySnapshotMeshVisual(_baseSnapshotMesh, _baseSnapshotMeshRenderer, _baseSnapshotMeshFilter, previewSurfaceMesh, previewSurfaceTransform, previewSurfaceMaterial);
        BuildGridLineMesh(_baseSnapshot, _baseLineMesh);
        RebuildMarkers(_baseMarkers, FindChild(_baseRoot.transform, "Markers"), _baseSnapshot.entries, baseColor);
    }

    private void InitializeFrontiers(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot)
    {
        _frontierBySide.Clear();
        _supportBySide.Clear();
        InitializeSideFrontier(snapshot, GrowthSide.Left);
        InitializeSideFrontier(snapshot, GrowthSide.Right);
        InitializeSideFrontier(snapshot, GrowthSide.Top);
        InitializeSideFrontier(snapshot, GrowthSide.Bottom);
    }

    private void InitializeSideFrontier(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot, GrowthSide side)
    {
        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> orderedStrips = ExtractOrderedSideStrips(snapshot, side);
        if (orderedStrips.Count <= 0)
        {
            _frontierBySide[side] = Array.Empty<ScanCoverDepthGridPointCloud.GridStateEntry>();
            _supportBySide[side] = Array.Empty<ScanCoverDepthGridPointCloud.GridStateEntry>();
            return;
        }

        _frontierBySide[side] = CloneEntries(orderedStrips[0]);
        _supportBySide[side] = CloneEntries(orderedStrips.Count > 1 ? orderedStrips[1] : orderedStrips[0]);
    }

    private void CreateBandVisual(
        GrowthSide side,
        ScanCoverDepthGridPointCloud.GridStateEntry[] previousStrip,
        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> bandStrips,
        Color color,
        Mesh previewSurfaceMesh,
        Transform previewSurfaceTransform,
        Material previewSurfaceMaterial)
    {
        EnsureObjects();
        ScanCoverDepthGridPointCloud.GridStateEntry[] flattened = FlattenStrips(bandStrips);

        BandVisual band = new BandVisual
        {
            side = side,
            entries = CloneEntries(flattened),
            root = new GameObject($"Band_{_bandCount + 1:000}_{side}")
        };
        band.root.transform.SetParent(_root.transform, false);

        GameObject snapshotMesh = new GameObject("SnapshotMesh");
        snapshotMesh.transform.SetParent(band.root.transform, false);
        band.snapshotMeshFilter = snapshotMesh.AddComponent<MeshFilter>();
        band.snapshotMeshRenderer = snapshotMesh.AddComponent<MeshRenderer>();
        band.snapshotMesh = new Mesh { name = $"ScanCover_GrowthBand_{_bandCount + 1:000}_SnapshotMesh" };
        band.snapshotMeshFilter.sharedMesh = band.snapshotMesh;

        GameObject lines = new GameObject("Lines");
        lines.transform.SetParent(band.root.transform, false);
        band.lineFilter = lines.AddComponent<MeshFilter>();
        band.lineRenderer = lines.AddComponent<MeshRenderer>();
        band.lineMesh = new Mesh { name = $"ScanCover_GrowthBand_{_bandCount + 1:000}_Lines" };
        band.lineFilter.sharedMesh = band.lineMesh;

        ApplySnapshotMeshVisual(band.snapshotMesh, band.snapshotMeshRenderer, band.snapshotMeshFilter, previewSurfaceMesh, previewSurfaceTransform, previewSurfaceMaterial);
        ApplyRendererMaterial(band.lineRenderer, growthMaterialOverride != null ? growthMaterialOverride : _runtimeGrowthMaterial, color);
        BuildBandLineMesh(side, previousStrip, bandStrips, band.lineMesh);

        Transform markerRoot = new GameObject("Markers").transform;
        markerRoot.SetParent(band.root.transform, false);
        RebuildMarkers(band.markers, markerRoot, flattened, color);

        band.root.SetActive(showGrowthBands);
        _bands.Add(band);
    }

    private void ApplySnapshotMeshVisual(
        Mesh targetMesh,
        MeshRenderer targetRenderer,
        MeshFilter targetFilter,
        Mesh sourceMesh,
        Transform sourceTransform,
        Material sourceMaterial)
    {
        if (targetMesh == null || targetRenderer == null || targetFilter == null)
            return;

        targetMesh.Clear();
        if (sourceMesh == null || sourceTransform == null || sourceMesh.vertexCount <= 0)
        {
            targetRenderer.enabled = false;
            return;
        }

        targetMesh.vertices = sourceMesh.vertices;
        targetMesh.normals = sourceMesh.normals;
        targetMesh.uv = sourceMesh.uv;
        targetMesh.triangles = sourceMesh.triangles;
        targetMesh.RecalculateBounds();
        targetFilter.sharedMesh = targetMesh;

        Transform meshTransform = targetFilter.transform;
        meshTransform.position = sourceTransform.position;
        meshTransform.rotation = sourceTransform.rotation;
        meshTransform.localScale = sourceTransform.lossyScale;

        Material material = snapshotMeshMaterialOverride != null
            ? snapshotMeshMaterialOverride
            : (sourceMaterial != null ? sourceMaterial : _runtimeSnapshotMeshMaterial);
        targetRenderer.sharedMaterial = material;
        targetRenderer.shadowCastingMode = ShadowCastingMode.Off;
        targetRenderer.receiveShadows = true;
        targetRenderer.enabled = false;
    }

    private void BuildGridLineMesh(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot, Mesh targetMesh)
    {
        _lineVerts.Clear();
        _lineIndices.Clear();

        Dictionary<long, int> vertexByKey = new Dictionary<long, int>(snapshot.entries.Length);
        for (int i = 0; i < snapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
            if (!entry.valid)
                continue;

            int vertexIndex = _lineVerts.Count;
            _lineVerts.Add(entry.worldPos);
            vertexByKey[ComposeKey(entry.row, entry.col)] = vertexIndex;
        }

        foreach (KeyValuePair<long, int> kv in vertexByKey)
        {
            SplitKey(kv.Key, out int row, out int col);
            TryAddGridEdge(vertexByKey, row, col, row, col + 1);
            TryAddGridEdge(vertexByKey, row, col, row + 1, col);
        }

        targetMesh.Clear();
        if (_lineVerts.Count <= 0 || _lineIndices.Count <= 0)
            return;
        targetMesh.SetVertices(_lineVerts);
        targetMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);
        targetMesh.RecalculateBounds();
    }

    private void BuildBandLineMesh(
        GrowthSide side,
        ScanCoverDepthGridPointCloud.GridStateEntry[] previousStrip,
        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> bandStrips,
        Mesh targetMesh)
    {
        _lineVerts.Clear();
        _lineIndices.Clear();

        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> previousByAxis = BuildAxisMap(previousStrip, side);

        for (int stripIndex = 0; stripIndex < bandStrips.Count; stripIndex++)
        {
            List<ScanCoverDepthGridPointCloud.GridStateEntry> sorted = new List<ScanCoverDepthGridPointCloud.GridStateEntry>(bandStrips[stripIndex].Length);
            for (int i = 0; i < bandStrips[stripIndex].Length; i++)
            {
                if (bandStrips[stripIndex][i].valid)
                    sorted.Add(bandStrips[stripIndex][i]);
            }

            SortStrip(sorted, side);

            for (int i = 0; i < sorted.Count - 1; i++)
            {
                if (!AreAdjacentOnStrip(side, sorted[i], sorted[i + 1]))
                    continue;
                AddLineSegment(sorted[i].worldPos, sorted[i + 1].worldPos);
            }

            Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> sourceByAxis =
                stripIndex == 0
                    ? previousByAxis
                    : BuildAxisMap(bandStrips[stripIndex - 1], side);

            for (int i = 0; i < sorted.Count; i++)
            {
                long axisKey = ComposeAxisKey(side, sorted[i]);
                if (!sourceByAxis.TryGetValue(axisKey, out ScanCoverDepthGridPointCloud.GridStateEntry previous))
                    continue;
                AddLineSegment(previous.worldPos, sorted[i].worldPos);
            }
        }

        targetMesh.Clear();
        if (_lineVerts.Count <= 0 || _lineIndices.Count <= 0)
            return;
        targetMesh.SetVertices(_lineVerts);
        targetMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);
        targetMesh.RecalculateBounds();
    }

    private static bool AreAdjacentOnStrip(GrowthSide side, ScanCoverDepthGridPointCloud.GridStateEntry a, ScanCoverDepthGridPointCloud.GridStateEntry b)
    {
        return side == GrowthSide.Left || side == GrowthSide.Right
            ? Mathf.Abs(a.row - b.row) == 1
            : Mathf.Abs(a.col - b.col) == 1;
    }

    private static void SortStrip(List<ScanCoverDepthGridPointCloud.GridStateEntry> entries, GrowthSide side)
    {
        if (side == GrowthSide.Left || side == GrowthSide.Right)
            entries.Sort((a, b) => a.row.CompareTo(b.row));
        else
            entries.Sort((a, b) => a.col.CompareTo(b.col));
    }

    private void AddLineSegment(Vector3 a, Vector3 b)
    {
        int start = _lineVerts.Count;
        _lineVerts.Add(a);
        _lineVerts.Add(b);
        _lineIndices.Add(start);
        _lineIndices.Add(start + 1);
    }

    private void TryAddGridEdge(Dictionary<long, int> vertexByKey, int rowA, int colA, int rowB, int colB)
    {
        if (!vertexByKey.TryGetValue(ComposeKey(rowA, colA), out int a))
            return;
        if (!vertexByKey.TryGetValue(ComposeKey(rowB, colB), out int b))
            return;
        _lineIndices.Add(a);
        _lineIndices.Add(b);
    }

    private GrowthSide ResolvePreferredGrowthSide(ScanCoverDepthGridPointCloud.GridStateSnapshot current)
    {
        Vector3 shellCenter = ComputeShellCenter();
        Vector3 previewCenter = ComputeSnapshotCenter(current.entries);
        Vector3 delta = previewCenter - shellCenter;

        float bestScore = float.NegativeInfinity;
        GrowthSide bestSide = GrowthSide.Right;

        foreach (KeyValuePair<GrowthSide, ScanCoverDepthGridPointCloud.GridStateEntry[]> kv in _frontierBySide)
        {
            Vector3 frontierCenter = ComputeSnapshotCenter(kv.Value);
            Vector3 direction = frontierCenter - shellCenter;
            if (direction.sqrMagnitude < 1e-6f)
                continue;
            float score = Vector3.Dot(delta.normalized, direction.normalized);
            if (score > bestScore)
            {
                bestScore = score;
                bestSide = kv.Key;
            }
        }

        return bestSide;
    }

    private Vector3 ComputeShellCenter()
    {
        List<Vector3> points = new List<Vector3>(4096);
        if (_baseSnapshot != null && _baseSnapshot.entries != null)
        {
            for (int i = 0; i < _baseSnapshot.entries.Length; i++)
            {
                if (_baseSnapshot.entries[i].valid)
                    points.Add(_baseSnapshot.entries[i].worldPos);
            }
        }

        for (int i = 0; i < _bands.Count; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry[] entries = _bands[i].entries;
            for (int j = 0; j < entries.Length; j++)
            {
                if (entries[j].valid)
                    points.Add(entries[j].worldPos);
            }
        }

        return ComputeSnapshotCenter(points);
    }

    private static Vector3 ComputeSnapshotCenter(IReadOnlyList<Vector3> points)
    {
        if (points == null || points.Count <= 0)
            return Vector3.zero;
        Vector3 sum = Vector3.zero;
        int count = 0;
        for (int i = 0; i < points.Count; i++)
        {
            sum += points[i];
            count++;
        }
        return count > 0 ? sum / count : Vector3.zero;
    }

    private static Vector3 ComputeSnapshotCenter(ScanCoverDepthGridPointCloud.GridStateEntry[] entries)
    {
        if (entries == null || entries.Length <= 0)
            return Vector3.zero;
        Vector3 sum = Vector3.zero;
        int count = 0;
        for (int i = 0; i < entries.Length; i++)
        {
            if (!entries[i].valid)
                continue;
            sum += entries[i].worldPos;
            count++;
        }
        return count > 0 ? sum / count : Vector3.zero;
    }

    private static ScanCoverDepthGridPointCloud.GridStateEntry[] ExtractSideStrip(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot, GrowthSide side)
    {
        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> orderedStrips = ExtractOrderedSideStrips(snapshot, side);
        return orderedStrips.Count > 0 ? orderedStrips[0] : Array.Empty<ScanCoverDepthGridPointCloud.GridStateEntry>();
    }

    private static List<ScanCoverDepthGridPointCloud.GridStateEntry[]> ExtractOrderedSideStrips(
        ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot,
        GrowthSide side)
    {
        Dictionary<int, List<ScanCoverDepthGridPointCloud.GridStateEntry>> stripsByAxis =
            new Dictionary<int, List<ScanCoverDepthGridPointCloud.GridStateEntry>>(64);

        for (int i = 0; i < snapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
            if (!entry.valid)
                continue;

            int axis = side == GrowthSide.Left || side == GrowthSide.Right ? entry.col : entry.row;
            if (!stripsByAxis.TryGetValue(axis, out List<ScanCoverDepthGridPointCloud.GridStateEntry> strip))
            {
                strip = new List<ScanCoverDepthGridPointCloud.GridStateEntry>(64);
                stripsByAxis.Add(axis, strip);
            }

            strip.Add(entry);
        }

        List<int> axes = new List<int>(stripsByAxis.Keys);
        axes.Sort((a, b) => CompareAxesOuterFirst(side, a, b));

        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> ordered =
            new List<ScanCoverDepthGridPointCloud.GridStateEntry[]>(axes.Count);

        for (int i = 0; i < axes.Count; i++)
        {
            List<ScanCoverDepthGridPointCloud.GridStateEntry> strip = stripsByAxis[axes[i]];
            SortStrip(strip, side);
            ordered.Add(strip.ToArray());
        }

        return ordered;
    }

    private static int CompareAxesOuterFirst(GrowthSide side, int a, int b)
    {
        return side switch
        {
            GrowthSide.Left => a.CompareTo(b),
            GrowthSide.Top => a.CompareTo(b),
            GrowthSide.Right => b.CompareTo(a),
            _ => b.CompareTo(a)
        };
    }

    private static List<ScanCoverDepthGridPointCloud.GridStateEntry[]> BuildIntermediateBandStrips(
        GrowthSide side,
        ScanCoverDepthGridPointCloud.GridStateEntry[] supportStrip,
        ScanCoverDepthGridPointCloud.GridStateEntry[] previousStrip,
        ScanCoverDepthGridPointCloud.GridStateEntry[] newStrip,
        Mesh previewSurfaceMesh,
        Transform previewSurfaceTransform,
        out float averageRatio,
        out int interiorStripCount)
    {
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> previousByAxis = BuildAxisMap(previousStrip, side);
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> newByAxis = BuildAxisMap(newStrip, side);
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> supportByAxis = BuildAxisMap(supportStrip, side);

        float ratioSum = 0f;
        int ratioCount = 0;
        foreach (KeyValuePair<long, ScanCoverDepthGridPointCloud.GridStateEntry> kv in previousByAxis)
        {
            if (!newByAxis.TryGetValue(kv.Key, out ScanCoverDepthGridPointCloud.GridStateEntry target))
                continue;
            if (!supportByAxis.TryGetValue(kv.Key, out ScanCoverDepthGridPointCloud.GridStateEntry support))
                continue;

            Vector3 step = kv.Value.worldPos - support.worldPos;
            float stepLength = step.magnitude;
            if (stepLength <= 1e-5f)
                continue;

            float gapLength = Vector3.Distance(kv.Value.worldPos, target.worldPos);
            ratioSum += gapLength / stepLength;
            ratioCount++;
        }

        averageRatio = ratioCount > 0 ? ratioSum / ratioCount : 1f;
        int segments = Mathf.Max(1, Mathf.FloorToInt(averageRatio + 1e-4f));
        interiorStripCount = Mathf.Max(0, segments - 1);

        List<ScanCoverDepthGridPointCloud.GridStateEntry[]> strips =
            new List<ScanCoverDepthGridPointCloud.GridStateEntry[]>(interiorStripCount + 1);

        for (int segmentIndex = 1; segmentIndex < segments; segmentIndex++)
        {
            float t = segmentIndex / (float)segments;
            List<ScanCoverDepthGridPointCloud.GridStateEntry> generated =
                new List<ScanCoverDepthGridPointCloud.GridStateEntry>(previousStrip.Length);

            for (int i = 0; i < previousStrip.Length; i++)
            {
                ScanCoverDepthGridPointCloud.GridStateEntry previous = previousStrip[i];
                if (!previous.valid)
                    continue;

                if (!newByAxis.TryGetValue(ComposeAxisKey(side, previous), out ScanCoverDepthGridPointCloud.GridStateEntry target))
                    continue;

                ScanCoverDepthGridPointCloud.GridStateEntry grown = previous;
                Vector3 rulePosition = Vector3.Lerp(previous.worldPos, target.worldPos, t);
                Vector3 ruleNormal = Vector3.Lerp(previous.normal, target.normal, t);
                if (ruleNormal.sqrMagnitude > 1e-6f)
                    ruleNormal.Normalize();
                else
                    ruleNormal = previous.normal.sqrMagnitude > 1e-6f ? previous.normal.normalized : Vector3.forward;

                grown.worldPos = rulePosition;
                grown.normal = ruleNormal;
                grown.confidence = Mathf.Lerp(previous.confidence, target.confidence, t);
                grown.valid = true;
                generated.Add(grown);
            }

            SortStrip(generated, side);
            if (generated.Count > 0)
                strips.Add(generated.ToArray());
        }

        strips.Add(CloneEntries(newStrip));
        return strips;
    }

    private static bool TryProjectPointOntoPreviewSurface(
        Vector3 worldPoint,
        Vector3 preferredDirectionWorld,
        Transform previewSurfaceTransform,
        Vector3[] previewVertices,
        int[] previewTriangles,
        Vector3[] previewNormals,
        float maxSnapDistance,
        out Vector3 projectedWorldPoint,
        out Vector3 projectedWorldNormal)
    {
        projectedWorldPoint = worldPoint;
        projectedWorldNormal = Vector3.forward;

        if (previewSurfaceTransform == null ||
            previewVertices == null || previewTriangles == null ||
            previewVertices.Length <= 0 || previewTriangles.Length < 3)
        {
            return false;
        }

        Vector3 localPoint = previewSurfaceTransform.InverseTransformPoint(worldPoint);
        Vector3 preferredDirectionLocal = previewSurfaceTransform.InverseTransformDirection(preferredDirectionWorld);
        if (preferredDirectionLocal.sqrMagnitude > 1e-6f)
            preferredDirectionLocal.Normalize();
        else
            preferredDirectionLocal = Vector3.forward;
        float bestSqrDistance = maxSnapDistance * maxSnapDistance;
        bool hasHit = false;
        Vector3 bestLocalPoint = localPoint;
        Vector3 bestLocalNormal = Vector3.forward;

        for (int i = 0; i <= previewTriangles.Length - 3; i += 3)
        {
            int ia = previewTriangles[i];
            int ib = previewTriangles[i + 1];
            int ic = previewTriangles[i + 2];
            if (ia < 0 || ib < 0 || ic < 0 ||
                ia >= previewVertices.Length || ib >= previewVertices.Length || ic >= previewVertices.Length)
            {
                continue;
            }

            Vector3 a = previewVertices[ia];
            Vector3 b = previewVertices[ib];
            Vector3 c = previewVertices[ic];
            Vector3 candidate = ClosestPointOnTriangle(localPoint, a, b, c);
            Vector3 candidateOffset = candidate - localPoint;
            float candidateOffsetLength = candidateOffset.magnitude;
            if (candidateOffsetLength > 1e-5f)
            {
                float forwardDot = Vector3.Dot(candidateOffset / candidateOffsetLength, preferredDirectionLocal);
                if (forwardDot < 0.45f)
                    continue;
            }
            float sqrDistance = (candidate - localPoint).sqrMagnitude;
            if (sqrDistance > bestSqrDistance)
                continue;

            bestSqrDistance = sqrDistance;
            bestLocalPoint = candidate;
            bestLocalNormal = ResolveTriangleNormal(previewNormals, ia, ib, ic, a, b, c);
            hasHit = true;
        }

        if (!hasHit)
            return false;

        projectedWorldPoint = previewSurfaceTransform.TransformPoint(bestLocalPoint);
        projectedWorldNormal = previewSurfaceTransform.TransformDirection(bestLocalNormal.normalized);
        if (projectedWorldNormal.sqrMagnitude <= 1e-6f)
            projectedWorldNormal = Vector3.forward;
        else
            projectedWorldNormal.Normalize();
        return true;
    }

    private static Vector3 ResolveTriangleNormal(
        Vector3[] previewNormals,
        int ia,
        int ib,
        int ic,
        Vector3 a,
        Vector3 b,
        Vector3 c)
    {
        if (previewNormals != null &&
            ia < previewNormals.Length && ib < previewNormals.Length && ic < previewNormals.Length)
        {
            Vector3 blended = previewNormals[ia] + previewNormals[ib] + previewNormals[ic];
            if (blended.sqrMagnitude > 1e-6f)
                return blended.normalized;
        }

        Vector3 cross = Vector3.Cross(b - a, c - a);
        return cross.sqrMagnitude > 1e-6f ? cross.normalized : Vector3.forward;
    }

    private static Vector3 ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;
        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0f && d2 <= 0f)
            return a;

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0f && d4 <= d3)
            return b;

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0f && d1 >= 0f && d3 <= 0f)
        {
            float v = d1 / (d1 - d3);
            return a + v * ab;
        }

        Vector3 cp = p - c;
        float d5 = Vector3.Dot(ab, cp);
        float d6 = Vector3.Dot(ac, cp);
        if (d6 >= 0f && d5 <= d6)
            return c;

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0f && d2 >= 0f && d6 <= 0f)
        {
            float w = d2 / (d2 - d6);
            return a + w * ac;
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b);
        }

        float denom = 1f / (va + vb + vc);
        float baryV = vb * denom;
        float baryW = vc * denom;
        return a + ab * baryV + ac * baryW;
    }

    private static ScanCoverDepthGridPointCloud.GridStateEntry[] FlattenStrips(List<ScanCoverDepthGridPointCloud.GridStateEntry[]> strips)
    {
        int totalCount = 0;
        for (int i = 0; i < strips.Count; i++)
            totalCount += strips[i]?.Length ?? 0;

        ScanCoverDepthGridPointCloud.GridStateEntry[] flattened =
            new ScanCoverDepthGridPointCloud.GridStateEntry[totalCount];

        int writeIndex = 0;
        for (int i = 0; i < strips.Count; i++)
        {
            if (strips[i] == null || strips[i].Length <= 0)
                continue;

            Array.Copy(strips[i], 0, flattened, writeIndex, strips[i].Length);
            writeIndex += strips[i].Length;
        }

        return flattened;
    }

    private static Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> BuildAxisMap(
        ScanCoverDepthGridPointCloud.GridStateEntry[] entries,
        GrowthSide side)
    {
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> map =
            new Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry>(entries?.Length ?? 0);

        if (entries == null)
            return map;

        for (int i = 0; i < entries.Length; i++)
        {
            if (!entries[i].valid)
                continue;
            map[ComposeAxisKey(side, entries[i])] = entries[i];
        }

        return map;
    }

    private void RebuildMarkers(List<GameObject> markers, Transform parent, ScanCoverDepthGridPointCloud.GridStateEntry[] entries, Color color)
    {
        ClearMarkerList(markers);
        if (entries == null || parent == null)
            return;

        Material markerMaterial = CreateRuntimeMaterial($"ScanCover_FrozenGridMarkers_{color.r:F2}_{color.g:F2}_{color.b:F2}", color);
        for (int i = 0; i < entries.Length; i++)
        {
            if (!entries[i].valid)
                continue;

            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.name = $"Marker_{entries[i].row}_{entries[i].col}";
            marker.transform.SetParent(parent, false);
            marker.transform.position = entries[i].worldPos;
            marker.transform.rotation = Quaternion.identity;
            marker.transform.localScale = Vector3.one * Mathf.Max(0.001f, markerScaleMeters);
            Collider collider = marker.GetComponent<Collider>();
            if (collider != null)
                Destroy(collider);
            Renderer renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = markerMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
            markers.Add(marker);
        }
    }

    private static Transform FindChild(Transform parent, string childName)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.name == childName)
                return child;
        }
        return null;
    }

    private Material CreateRuntimeMaterial(string name, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Standard");

        Material material = new Material(shader) { name = name };
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        if (material.HasProperty("_Surface"))
            material.SetFloat("_Surface", 0f);
        if (material.HasProperty("_Cull"))
            material.SetFloat("_Cull", (float)CullMode.Off);
        return material;
    }

    private void ApplyRendererMaterial(Renderer renderer, Material material, Color color)
    {
        if (renderer == null || material == null)
            return;
        if (renderer.sharedMaterial != material)
            renderer.sharedMaterial = material;
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
    }

    private static ScanCoverDepthGridPointCloud.GridStateSnapshot CloneSnapshot(ScanCoverDepthGridPointCloud.GridStateSnapshot source)
    {
        return new ScanCoverDepthGridPointCloud.GridStateSnapshot
        {
            componentName = source.componentName,
            samplingMode = source.samplingMode,
            frameIndex = source.frameIndex,
            resolutionWidth = source.resolutionWidth,
            resolutionHeight = source.resolutionHeight,
            cellCount = source.cellCount,
            visibleCount = source.visibleCount,
            entries = CloneEntries(source.entries)
        };
    }

    private static ScanCoverDepthGridPointCloud.GridStateEntry[] CloneEntries(ScanCoverDepthGridPointCloud.GridStateEntry[] entries)
    {
        if (entries == null)
            return Array.Empty<ScanCoverDepthGridPointCloud.GridStateEntry>();
        ScanCoverDepthGridPointCloud.GridStateEntry[] clone = new ScanCoverDepthGridPointCloud.GridStateEntry[entries.Length];
        Array.Copy(entries, clone, entries.Length);
        return clone;
    }

    private static long ComposeKey(int row, int col)
    {
        return ((long)row << 32) ^ (uint)col;
    }

    private static void SplitKey(long key, out int row, out int col)
    {
        row = (int)(key >> 32);
        col = (int)(key & 0xffffffff);
    }

    private static long ComposeAxisKey(GrowthSide side, ScanCoverDepthGridPointCloud.GridStateEntry entry)
    {
        return side == GrowthSide.Left || side == GrowthSide.Right ? entry.row : entry.col;
    }

    private Color ResolveGrowthBandColor(int bandIndex)
    {
        return (bandIndex % 4) switch
        {
            0 => growthColorA,
            1 => growthColorB,
            2 => growthColorC,
            _ => growthColorD
        };
    }

    private void ClearMarkerList(List<GameObject> markers)
    {
        for (int i = 0; i < markers.Count; i++)
        {
            if (markers[i] != null)
                Destroy(markers[i]);
        }
        markers.Clear();
    }

    private void DestroyBandVisual(BandVisual band)
    {
        if (band == null)
            return;
        ClearMarkerList(band.markers);
        if (band.lineMesh != null)
            Destroy(band.lineMesh);
        if (band.snapshotMesh != null)
            Destroy(band.snapshotMesh);
        if (band.root != null)
            Destroy(band.root);
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog && !string.IsNullOrEmpty(issue))
            Debug.LogWarning($"[ScanCoverDepthGridActivePatchGrowthManager] {issue}");
        return false;
    }

    private void OnDestroy()
    {
        ClearAll();
        if (_runtimeBaseMaterial != null)
            Destroy(_runtimeBaseMaterial);
        if (_runtimeGrowthMaterial != null)
            Destroy(_runtimeGrowthMaterial);
        if (_runtimeSnapshotMeshMaterial != null)
            Destroy(_runtimeSnapshotMeshMaterial);
        if (_baseLineMesh != null)
            Destroy(_baseLineMesh);
        if (_baseSnapshotMesh != null)
            Destroy(_baseSnapshotMesh);
        if (_root != null)
            Destroy(_root);
    }
}

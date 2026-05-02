using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverDepthGridLineConflictManager : MonoBehaviour
{
    [Serializable]
    public struct SegmentConflictRecord
    {
        public int otherShellId;
        public int currentRow;
        public int currentCol;
        public bool currentHorizontal;
        public int otherRow;
        public int otherCol;
        public bool otherHorizontal;
        public float distance;
        public float directionDot;
        public float normalOffset;
        public string kind;
    }

    [Serializable]
    public struct CellConflictRecord
    {
        public int index;
        public int row;
        public int col;
        public bool valid;
        public int overlapConflictCount;
        public int behindNearConflictCount;
        public int crossingConflictCount;
        public int totalConflictCount;
        public int uniqueShellConflictCount;
        public bool overlapHorizontalBand;
        public bool overlapVerticalBand;
        public int overlapNeighborhoodCellCount;
        public int occupancyNeighborhoodCellCount;
        public int crossingNeighborhoodCellCount;
        public bool horizontalConflict;
        public bool verticalConflict;
    }

    [Serializable]
    public sealed class LineConflictStateSnapshot
    {
        public string componentName;
        public int shellId;
        public int historyShellCount;
        public int frameIndex;
        public int resolutionWidth;
        public int resolutionHeight;
        public int cellCount;
        public int visibleCount;
        public int segmentCount;
        public int overlapConflictCount;
        public int behindNearConflictCount;
        public int crossingConflictCount;
        public int acceptedVisibleCount;
        public int acceptedSegmentCount;
        public int culledCellCount;
        public int exportedConflictRecordCount;
        public int omittedConflictRecordCount;
        public CellConflictRecord[] cells;
        public SegmentConflictRecord[] segmentConflicts;
    }

    private sealed class ShellRecord
    {
        public int shellId;
        public ScanCoverDepthGridPointCloud.GridStateSnapshot grid;
        public List<SegmentRecord> segments;
    }

    private struct SegmentRecord
    {
        public int shellId;
        public int row;
        public int col;
        public bool horizontal;
        public Vector3 a;
        public Vector3 b;
        public Vector3 midpoint;
        public Vector3 direction;
        public Vector3 averageNormal;
    }

    [Header("Refs")]
    public ScanCoverDepthGridPointCloud depthGridPointCloud;

    [Header("Conflict Detection")]
    [Min(0.001f)] public float segmentNearDistanceMeters = 0.03f;
    [Min(0.001f)] public float overlapMaxNormalOffsetMeters = 0.02f;
    [Range(0f, 1f)] public float overlapMinParallelDot = 0.92f;
    [Min(0.001f)] public float behindNearDistanceMeters = 0.48f;
    [Min(0.001f)] public float behindNearMaxNormalOffsetMeters = 0.72f;
    [Range(0f, 1f)] public float behindNearMinParallelDot = 0.80f;
    [Range(0f, 1f)] public float crossingMaxParallelDot = 0.65f;
    [Min(0.01f)] public float indexCellSizeMeters = 0.08f;
    [Min(1)] public int maxExportConflictRecords = 2048;
    public bool cullCurrentCellsOnOverlap = true;
    [Min(1)] public int overlapCullMinConflictCount = 1;
    public bool overlapCullRequiresBidirectionalBand = true;
    [Min(1)] public int occupancyBandSearchRadius = 2;
    [Min(1)] public int occupancyBandMinSeedCells = 2;
    public bool cullCurrentCellsOnCrossing = false;
    [Min(1)] public int crossingCullMinConflictCount = 3;
    [Min(1)] public int crossingCullMinUniqueShells = 2;

    [Header("Export")]
    public string exportDirectoryOverride = "";

    [Header("Surface")]
    public bool showAcceptedShellMeshInScene = false;
    public Material surfaceMaterialOverride;
    public bool surfaceDoubleSided = true;
    public Color surfaceColor = new Color(0.95f, 0.95f, 0.97f, 1f);

    [Header("Accepted Point Snapshots")]
    public bool showAcceptedPointSnapshotsInScene = true;
    [Min(0.001f)] public float acceptedPointScaleMeters = 0.014f;
    [Min(0.0005f)] public float acceptedLineWidthMeters = 0.004f;
    public Color acceptedPointColor = new Color(0.98f, 0.72f, 0.18f, 1f);

    [Header("Debug")]
    public bool debugLog = false;

    public int HistoryShellCount => _history.Count;
    public LineConflictStateSnapshot LatestConflictState => _latestState;
    public ScanCoverDepthGridPointCloud.GridStateSnapshot LatestAcceptedSnapshot => _latestAcceptedSnapshot;
    public string LastIssue { get; private set; }
    public string LastExportPath { get; private set; }

    private readonly List<ShellRecord> _history = new List<ShellRecord>(16);
    private readonly List<SegmentRecord> _allSegments = new List<SegmentRecord>(8192);
    private readonly Dictionary<Vector3Int, List<int>> _segmentIndex = new Dictionary<Vector3Int, List<int>>(4096);
    private readonly List<SegmentConflictRecord> _latestConflictRecords = new List<SegmentConflictRecord>(2048);
    private readonly List<Vector3> _meshVerts = new List<Vector3>(4096);
    private readonly List<Vector3> _meshNormals = new List<Vector3>(4096);
    private readonly List<int> _meshTriangles = new List<int>(8192);
    private LineConflictStateSnapshot _latestState;
    private ScanCoverDepthGridPointCloud.GridStateSnapshot _latestAcceptedSnapshot;
    private int _nextShellId = 1;
    private GameObject _shellRoot;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;
    private Material _runtimeMaterial;
    private GameObject _pointSnapshotRoot;
    private int _pointSnapshotCounter = 1;

    public void EnsureInitialized()
    {
        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        EnsureShellObjects();
    }

    public bool AnalyzeCurrentGridState()
    {
        EnsureInitialized();
        if (depthGridPointCloud == null)
            return SetIssue("DepthGridPointCloud is missing.");
        if (!depthGridPointCloud.TryGetCurrentGridState(out ScanCoverDepthGridPointCloud.GridStateSnapshot incoming))
            return SetIssue(depthGridPointCloud.LastIssue ?? "Grid state is unavailable.");

        ShellRecord shell = new ShellRecord
        {
            shellId = _nextShellId++,
            grid = CloneSnapshot(incoming),
        };
        shell.segments = BuildSegments(shell.grid, shell.shellId);
        AnalyzeAgainstHistory(shell, out CellConflictRecord[] cellRecords, out int overlapCount, out int behindNearCount, out int crossingCount, out int omittedCount);
        _latestAcceptedSnapshot = BuildAcceptedSnapshot(shell.grid, cellRecords, out int culledCellCount);

        ShellRecord acceptedShell = new ShellRecord
        {
            shellId = shell.shellId,
            grid = _latestAcceptedSnapshot,
        };
        acceptedShell.segments = BuildSegments(acceptedShell.grid, acceptedShell.shellId);

        _latestState = new LineConflictStateSnapshot
        {
            componentName = incoming.componentName,
            shellId = shell.shellId,
            historyShellCount = _history.Count,
            frameIndex = incoming.frameIndex,
            resolutionWidth = incoming.resolutionWidth,
            resolutionHeight = incoming.resolutionHeight,
            cellCount = incoming.cellCount,
            visibleCount = incoming.visibleCount,
            segmentCount = shell.segments.Count,
            overlapConflictCount = overlapCount,
            behindNearConflictCount = behindNearCount,
            crossingConflictCount = crossingCount,
            acceptedVisibleCount = _latestAcceptedSnapshot != null ? _latestAcceptedSnapshot.visibleCount : 0,
            acceptedSegmentCount = acceptedShell.segments.Count,
            culledCellCount = culledCellCount,
            exportedConflictRecordCount = _latestConflictRecords.Count,
            omittedConflictRecordCount = omittedCount,
            cells = cellRecords,
            segmentConflicts = _latestConflictRecords.ToArray(),
        };

        RebuildAcceptedShellMesh();
        CreateAcceptedPointSnapshot();
        AddShellToHistory(acceptedShell);
        LastIssue = null;
        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverDepthGridLineConflictManager] shell={shell.shellId}, historyBefore={_latestState.historyShellCount}, " +
                $"segments={shell.segments.Count}, overlap={overlapCount}, behindNear={behindNearCount}, crossing={crossingCount}, culledCells={culledCellCount}, acceptedVisible={_latestState.acceptedVisibleCount}, exportedRecords={_latestState.exportedConflictRecordCount}, omittedRecords={omittedCount}");
        }
        return true;
    }

    public bool ExportLatestConflictStateJson(out string exportPath)
    {
        exportPath = null;
        if (_latestState == null)
            return SetIssue("Latest line conflict state is empty.");

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        exportPath = Path.Combine(exportDirectory, $"LineConflictState_{DateTime.Now:yyyyMMdd_HHmmss}.json");
        File.WriteAllText(exportPath, JsonUtility.ToJson(_latestState, true), Encoding.UTF8);
        LastExportPath = exportPath;
        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridLineConflictManager] Exported conflict state => {exportPath}");
        return true;
    }

    public bool ExportLatestAcceptedGridStateJson(out string exportPath)
    {
        exportPath = null;
        if (_latestAcceptedSnapshot == null)
            return SetIssue("Latest accepted grid state is empty.");

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        exportPath = Path.Combine(exportDirectory, $"LineAcceptedGridState_{DateTime.Now:yyyyMMdd_HHmmss}.json");
        File.WriteAllText(exportPath, JsonUtility.ToJson(_latestAcceptedSnapshot, true), Encoding.UTF8);
        LastExportPath = exportPath;
        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridLineConflictManager] Exported accepted grid state => {exportPath}");
        return true;
    }

    public bool ExportLatestAcceptedShellAsObj(out string exportPath)
    {
        exportPath = null;
        if (_latestAcceptedSnapshot == null)
            return SetIssue("Latest accepted shell is empty.");

        List<Vector3> vertices = new List<Vector3>(_latestAcceptedSnapshot.visibleCount);
        List<Vector3> normals = new List<Vector3>(_latestAcceptedSnapshot.visibleCount);
        List<int> triangles = new List<int>(_latestAcceptedSnapshot.visibleCount * 6);
        BuildMeshDataFromSnapshot(_latestAcceptedSnapshot, vertices, normals, triangles);
        if (vertices.Count <= 0 || triangles.Count < 3)
            return SetIssue("Latest accepted shell has no exportable triangles.");

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        exportPath = Path.Combine(exportDirectory, $"LineAcceptedShell_{DateTime.Now:yyyyMMdd_HHmmss}.obj");

        StringBuilder builder = new StringBuilder(1024 * 64);
        builder.AppendLine("# ScanCover Line Accepted Shell OBJ");
        builder.AppendLine("o LineAcceptedShell");
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            builder.Append("v ")
                .Append(v.x.ToString("R", CultureInfo.InvariantCulture)).Append(' ')
                .Append(v.y.ToString("R", CultureInfo.InvariantCulture)).Append(' ')
                .Append(v.z.ToString("R", CultureInfo.InvariantCulture)).AppendLine();
        }
        for (int i = 0; i < normals.Count; i++)
        {
            Vector3 n = normals[i].sqrMagnitude > 1e-6f ? normals[i].normalized : Vector3.up;
            builder.Append("vn ")
                .Append(n.x.ToString("R", CultureInfo.InvariantCulture)).Append(' ')
                .Append(n.y.ToString("R", CultureInfo.InvariantCulture)).Append(' ')
                .Append(n.z.ToString("R", CultureInfo.InvariantCulture)).AppendLine();
        }
        for (int i = 0; i <= triangles.Count - 3; i += 3)
        {
            int a = triangles[i] + 1;
            int b = triangles[i + 1] + 1;
            int c = triangles[i + 2] + 1;
            builder.Append("f ")
                .Append(a).Append("//").Append(a).Append(' ')
                .Append(b).Append("//").Append(b).Append(' ')
                .Append(c).Append("//").Append(c).AppendLine();
        }

        File.WriteAllText(exportPath, builder.ToString(), Encoding.UTF8);
        LastExportPath = exportPath;
        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridLineConflictManager] Exported accepted shell => {exportPath}");
        return true;
    }

    public void ClearAll()
    {
        _history.Clear();
        _allSegments.Clear();
        _segmentIndex.Clear();
        _latestConflictRecords.Clear();
        _latestState = null;
        _latestAcceptedSnapshot = null;
        _nextShellId = 1;
        LastIssue = null;
        LastExportPath = null;
        if (_mesh != null)
            _mesh.Clear();
        if (_shellRoot != null)
            _shellRoot.SetActive(false);
        if (_pointSnapshotRoot != null)
        {
            Destroy(_pointSnapshotRoot);
            _pointSnapshotRoot = null;
        }
        _pointSnapshotCounter = 1;
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_runtimeMaterial != null)
            Destroy(_runtimeMaterial);
        if (_shellRoot != null)
            Destroy(_shellRoot);
        if (_pointSnapshotRoot != null)
            Destroy(_pointSnapshotRoot);
    }

    private void AnalyzeAgainstHistory(
        ShellRecord shell,
        out CellConflictRecord[] cellRecords,
        out int overlapCount,
        out int behindNearCount,
        out int crossingCount,
        out int omittedCount)
    {
        int cellCount = shell.grid.entries != null ? shell.grid.entries.Length : 0;
        overlapCount = 0;
        behindNearCount = 0;
        crossingCount = 0;
        omittedCount = 0;
        _latestConflictRecords.Clear();

        CellConflictRecord[] cells = new CellConflictRecord[cellCount];
        HashSet<int>[] shellHits = new HashSet<int>[cellCount];
        for (int i = 0; i < cellCount; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = shell.grid.entries[i];
            cells[i] = new CellConflictRecord
            {
                index = entry.index,
                row = entry.row,
                col = entry.col,
                valid = entry.valid,
            };
        }

        if (_history.Count <= 0 || shell.segments.Count <= 0 || _allSegments.Count <= 0)
        {
            cellRecords = cells;
            return;
        }

        float queryDistance = Mathf.Max(0.001f, Mathf.Max(segmentNearDistanceMeters, behindNearDistanceMeters));
        float queryDistanceSqr = queryDistance * queryDistance;
        HashSet<long> dedupe = new HashSet<long>();

        for (int i = 0; i < shell.segments.Count; i++)
        {
            SegmentRecord current = shell.segments[i];
            Vector3Int centerKey = Quantize(current.midpoint, Mathf.Max(0.01f, indexCellSizeMeters));
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        Vector3Int key = centerKey + new Vector3Int(x, y, z);
                        if (!_segmentIndex.TryGetValue(key, out List<int> bucket))
                            continue;

                        for (int j = 0; j < bucket.Count; j++)
                        {
                            SegmentRecord other = _allSegments[bucket[j]];
                            float midDistanceSqr = (current.midpoint - other.midpoint).sqrMagnitude;
                            if (midDistanceSqr > queryDistanceSqr * 9f)
                                continue;

                            float segmentDistance = SegmentDistance(current.a, current.b, other.a, other.b);
                            if (segmentDistance > queryDistance)
                                continue;

                            float directionDot = Mathf.Abs(Vector3.Dot(current.direction, other.direction));
                            float normalOffset = ResolveNormalOffset(current, other);
                            string kind = ClassifyConflictKind(segmentDistance, directionDot, normalOffset);
                            if (kind == "skew")
                                continue;

                            long pairKey = ComposeConflictKey(current.row, current.col, current.horizontal, other.shellId, other.row, other.col, other.horizontal, kind);
                            if (!dedupe.Add(pairKey))
                                continue;

                            if (kind == "overlap") overlapCount++;
                            else if (kind == "behind-near") behindNearCount++;
                            else if (kind == "crossing") crossingCount++;

                            MarkConflict(ref cells, shellHits, current.row, current.col, current.horizontal, other.shellId, kind);
                            if (_latestConflictRecords.Count < Mathf.Max(1, maxExportConflictRecords))
                            {
                                _latestConflictRecords.Add(new SegmentConflictRecord
                                {
                                    otherShellId = other.shellId,
                                    currentRow = current.row,
                                    currentCol = current.col,
                                    currentHorizontal = current.horizontal,
                                    otherRow = other.row,
                                    otherCol = other.col,
                                    otherHorizontal = other.horizontal,
                                    distance = segmentDistance,
                                    directionDot = directionDot,
                                    normalOffset = normalOffset,
                                    kind = kind,
                                });
                            }
                            else
                            {
                                omittedCount++;
                            }
                        }
                    }
                }
            }
        }

        for (int i = 0; i < cells.Length; i++)
            cells[i].uniqueShellConflictCount = shellHits[i] != null ? shellHits[i].Count : 0;

        PopulateNeighborhoodConflictCounts(ref cells);

        cellRecords = cells;
    }

    private ScanCoverDepthGridPointCloud.GridStateSnapshot BuildAcceptedSnapshot(
        ScanCoverDepthGridPointCloud.GridStateSnapshot source,
        CellConflictRecord[] cellRecords,
        out int culledCellCount)
    {
        ScanCoverDepthGridPointCloud.GridStateSnapshot accepted = CloneSnapshot(source);
        culledCellCount = 0;
        if (accepted.entries == null || cellRecords == null)
        {
            accepted.visibleCount = accepted.entries != null ? CountVisible(accepted.entries) : 0;
            return accepted;
        }

        Dictionary<long, CellConflictRecord> conflictByCell = new Dictionary<long, CellConflictRecord>(cellRecords.Length);
        for (int i = 0; i < cellRecords.Length; i++)
            conflictByCell[ComposeCellKey(cellRecords[i].row, cellRecords[i].col)] = cellRecords[i];

        for (int i = 0; i < accepted.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = accepted.entries[i];
            if (!entry.valid)
                continue;

            if (!conflictByCell.TryGetValue(ComposeCellKey(entry.row, entry.col), out CellConflictRecord conflict))
                continue;

            int occupancyConflictCount = conflict.overlapConflictCount + conflict.behindNearConflictCount;
            bool cull = false;
            if (cullCurrentCellsOnOverlap &&
                occupancyConflictCount >= Mathf.Max(1, overlapCullMinConflictCount) &&
                conflict.occupancyNeighborhoodCellCount >= Mathf.Max(1, occupancyBandMinSeedCells) &&
                (!overlapCullRequiresBidirectionalBand || (conflict.overlapHorizontalBand && conflict.overlapVerticalBand)))
                cull = true;
            if (!cull &&
                cullCurrentCellsOnCrossing &&
                conflict.crossingConflictCount >= Mathf.Max(1, crossingCullMinConflictCount) &&
                conflict.uniqueShellConflictCount >= Mathf.Max(1, crossingCullMinUniqueShells))
                cull = true;
            if (!cull)
                continue;

            entry.valid = false;
            entry.worldPos = Vector3.zero;
            entry.normal = Vector3.zero;
            entry.confidence = 0f;
            accepted.entries[i] = entry;
            culledCellCount++;
        }

        accepted.visibleCount = CountVisible(accepted.entries);
        return accepted;
    }

    private void PopulateNeighborhoodConflictCounts(ref CellConflictRecord[] cells)
    {
        Dictionary<long, int> indexByCell = new Dictionary<long, int>(cells.Length);
        for (int i = 0; i < cells.Length; i++)
            indexByCell[ComposeCellKey(cells[i].row, cells[i].col)] = i;

        for (int i = 0; i < cells.Length; i++)
        {
            int overlapNeighbors = 0;
            int crossingNeighbors = 0;
            int row = cells[i].row;
            int col = cells[i].col;
            for (int dr = -1; dr <= 1; dr++)
            {
                for (int dc = -1; dc <= 1; dc++)
                {
                    int nr = row + dr;
                    int nc = col + dc;
                    if (!indexByCell.TryGetValue(ComposeCellKey(nr, nc), out int ni))
                        continue;

                    if (cells[ni].overlapConflictCount > 0)
                        overlapNeighbors++;
                    if (cells[ni].overlapConflictCount > 0 || cells[ni].behindNearConflictCount > 0)
                        cells[i].occupancyNeighborhoodCellCount++;
                    if (cells[ni].crossingConflictCount > 0)
                        crossingNeighbors++;
                }
            }

            cells[i].overlapNeighborhoodCellCount = overlapNeighbors;
            cells[i].crossingNeighborhoodCellCount = crossingNeighbors;
        }

        int searchRadius = Mathf.Max(1, occupancyBandSearchRadius);
        int minSeeds = Mathf.Max(1, occupancyBandMinSeedCells);
        for (int i = 0; i < cells.Length; i++)
        {
            if (cells[i].overlapConflictCount <= 0 && cells[i].behindNearConflictCount <= 0)
                continue;

            int horizontalSeeds = 0;
            int verticalSeeds = 0;
            for (int step = -searchRadius; step <= searchRadius; step++)
            {
                if (step == 0)
                    continue;

                if (indexByCell.TryGetValue(ComposeCellKey(cells[i].row, cells[i].col + step), out int h))
                {
                    if (cells[h].overlapConflictCount > 0 || cells[h].behindNearConflictCount > 0)
                        horizontalSeeds++;
                }

                if (indexByCell.TryGetValue(ComposeCellKey(cells[i].row + step, cells[i].col), out int v))
                {
                    if (cells[v].overlapConflictCount > 0 || cells[v].behindNearConflictCount > 0)
                        verticalSeeds++;
                }
            }

            if (horizontalSeeds >= minSeeds)
                cells[i].overlapHorizontalBand = true;
            if (verticalSeeds >= minSeeds)
                cells[i].overlapVerticalBand = true;
        }
    }

    private static void MarkConflict(
        ref CellConflictRecord[] cells,
        HashSet<int>[] shellHits,
        int row,
        int col,
        bool horizontal,
        int otherShellId,
        string kind)
    {
        MarkCell(ref cells, shellHits, row, col, horizontal, otherShellId, kind);
        if (horizontal)
            MarkCell(ref cells, shellHits, row, col + 1, horizontal, otherShellId, kind);
        else
            MarkCell(ref cells, shellHits, row + 1, col, horizontal, otherShellId, kind);
    }

    private static void MarkCell(
        ref CellConflictRecord[] cells,
        HashSet<int>[] shellHits,
        int row,
        int col,
        bool horizontal,
        int otherShellId,
        string kind)
    {
        for (int i = 0; i < cells.Length; i++)
        {
            if (cells[i].row != row || cells[i].col != col)
                continue;

            if (kind == "overlap")
            {
                cells[i].overlapConflictCount++;
                cells[i].overlapHorizontalBand |= horizontal;
                cells[i].overlapVerticalBand |= !horizontal;
            }
            else if (kind == "behind-near")
            {
                cells[i].behindNearConflictCount++;
                cells[i].overlapHorizontalBand |= horizontal;
                cells[i].overlapVerticalBand |= !horizontal;
            }
            else
            {
                cells[i].crossingConflictCount++;
            }
            cells[i].totalConflictCount++;
            cells[i].horizontalConflict |= horizontal;
            cells[i].verticalConflict |= !horizontal;
            shellHits[i] ??= new HashSet<int>();
            shellHits[i].Add(otherShellId);
            return;
        }
    }

    private void AddShellToHistory(ShellRecord shell)
    {
        _history.Add(shell);
        float cellSize = Mathf.Max(0.01f, indexCellSizeMeters);
        for (int i = 0; i < shell.segments.Count; i++)
        {
            int segmentIndex = _allSegments.Count;
            _allSegments.Add(shell.segments[i]);
            Vector3Int key = Quantize(shell.segments[i].midpoint, cellSize);
            if (!_segmentIndex.TryGetValue(key, out List<int> bucket))
            {
                bucket = new List<int>(8);
                _segmentIndex.Add(key, bucket);
            }
            bucket.Add(segmentIndex);
        }
    }

    private static List<SegmentRecord> BuildSegments(ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot, int shellId)
    {
        List<SegmentRecord> segments = new List<SegmentRecord>(4096);
        if (snapshot == null || snapshot.entries == null || snapshot.entries.Length <= 0)
            return segments;

        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> byKey = new Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry>(snapshot.entries.Length);
        int maxRow = -1;
        int maxCol = -1;
        for (int i = 0; i < snapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
            byKey[ComposeCellKey(entry.row, entry.col)] = entry;
            if (entry.row > maxRow) maxRow = entry.row;
            if (entry.col > maxCol) maxCol = entry.col;
        }

        for (int row = 0; row <= maxRow; row++)
        {
            for (int col = 0; col <= maxCol; col++)
            {
                if (TryGetValidEntry(byKey, row, col, out ScanCoverDepthGridPointCloud.GridStateEntry current))
                {
                    if (TryGetValidEntry(byKey, row, col + 1, out ScanCoverDepthGridPointCloud.GridStateEntry right))
                        AddSegment(segments, shellId, current, right, true);
                    if (TryGetValidEntry(byKey, row + 1, col, out ScanCoverDepthGridPointCloud.GridStateEntry down))
                        AddSegment(segments, shellId, current, down, false);
                }
            }
        }
        return segments;
    }

    private static void AddSegment(
        List<SegmentRecord> target,
        int shellId,
        ScanCoverDepthGridPointCloud.GridStateEntry a,
        ScanCoverDepthGridPointCloud.GridStateEntry b,
        bool horizontal)
    {
        Vector3 delta = b.worldPos - a.worldPos;
        float length = delta.magnitude;
        if (length <= 1e-5f)
            return;

        target.Add(new SegmentRecord
        {
            shellId = shellId,
            row = a.row,
            col = a.col,
            horizontal = horizontal,
            a = a.worldPos,
            b = b.worldPos,
            midpoint = (a.worldPos + b.worldPos) * 0.5f,
            direction = delta / length,
            averageNormal = ResolveAverageNormal(a.normal, b.normal),
        });
    }

    private string ClassifyConflictKind(float segmentDistance, float directionDot, float normalOffset)
    {
        float overlapDistance = Mathf.Max(0.001f, segmentNearDistanceMeters);
        float overlapNormalOffset = Mathf.Max(0.0005f, overlapMaxNormalOffsetMeters);
        float behindDistance = Mathf.Max(overlapDistance, behindNearDistanceMeters);
        float behindOffset = Mathf.Max(overlapNormalOffset, behindNearMaxNormalOffsetMeters);

        if (directionDot >= overlapMinParallelDot &&
            segmentDistance <= overlapDistance &&
            normalOffset <= overlapNormalOffset)
            return "overlap";

        if (directionDot >= behindNearMinParallelDot &&
            segmentDistance <= behindDistance &&
            normalOffset > overlapNormalOffset &&
            normalOffset <= behindOffset)
            return "behind-near";

        if (directionDot <= crossingMaxParallelDot)
            return "crossing";

        return "skew";
    }

    private static float ResolveNormalOffset(SegmentRecord current, SegmentRecord other)
    {
        Vector3 normal = current.averageNormal.sqrMagnitude > 1e-6f ? current.averageNormal.normalized : Vector3.zero;
        if (normal == Vector3.zero)
            return Vector3.Distance(current.midpoint, other.midpoint);

        return Mathf.Abs(Vector3.Dot(other.midpoint - current.midpoint, normal));
    }

    private static Vector3 ResolveAverageNormal(Vector3 a, Vector3 b)
    {
        Vector3 sum = Vector3.zero;
        if (a.sqrMagnitude > 1e-6f)
            sum += a.normalized;
        if (b.sqrMagnitude > 1e-6f)
            sum += b.normalized;
        return sum.sqrMagnitude > 1e-6f ? sum.normalized : Vector3.up;
    }

    private static bool TryGetValidEntry(
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> byKey,
        int row,
        int col,
        out ScanCoverDepthGridPointCloud.GridStateEntry entry)
    {
        if (byKey.TryGetValue(ComposeCellKey(row, col), out entry) && entry.valid)
            return true;
        entry = default;
        return false;
    }

    private static ScanCoverDepthGridPointCloud.GridStateSnapshot CloneSnapshot(ScanCoverDepthGridPointCloud.GridStateSnapshot source)
    {
        ScanCoverDepthGridPointCloud.GridStateEntry[] entries = new ScanCoverDepthGridPointCloud.GridStateEntry[source.entries.Length];
        Array.Copy(source.entries, entries, source.entries.Length);
        return new ScanCoverDepthGridPointCloud.GridStateSnapshot
        {
            componentName = source.componentName,
            samplingMode = source.samplingMode,
            frameIndex = source.frameIndex,
            resolutionWidth = source.resolutionWidth,
            resolutionHeight = source.resolutionHeight,
            cellCount = source.cellCount,
            visibleCount = source.visibleCount,
            entries = entries,
        };
    }

    private static int CountVisible(ScanCoverDepthGridPointCloud.GridStateEntry[] entries)
    {
        if (entries == null)
            return 0;

        int count = 0;
        for (int i = 0; i < entries.Length; i++)
            if (entries[i].valid)
                count++;
        return count;
    }

    private static void BuildMeshDataFromSnapshot(
        ScanCoverDepthGridPointCloud.GridStateSnapshot snapshot,
        List<Vector3> vertices,
        List<Vector3> normals,
        List<int> triangles)
    {
        if (snapshot == null || snapshot.entries == null)
            return;

        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> byKey = new Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry>(snapshot.entries.Length);
        Dictionary<long, int> vertexIndices = new Dictionary<long, int>(snapshot.visibleCount);
        int maxRow = -1;
        int maxCol = -1;
        for (int i = 0; i < snapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = snapshot.entries[i];
            byKey[ComposeCellKey(entry.row, entry.col)] = entry;
            if (entry.row > maxRow) maxRow = entry.row;
            if (entry.col > maxCol) maxCol = entry.col;
            if (!entry.valid)
                continue;

            long key = ComposeCellKey(entry.row, entry.col);
            vertexIndices[key] = vertices.Count;
            vertices.Add(entry.worldPos);
            normals.Add(entry.normal.sqrMagnitude > 1e-6f ? entry.normal.normalized : Vector3.up);
        }

        for (int row = 0; row < maxRow; row++)
        {
            for (int col = 0; col < maxCol; col++)
            {
                bool v00 = TryGetValidEntry(byKey, row, col, out ScanCoverDepthGridPointCloud.GridStateEntry e00);
                bool v10 = TryGetValidEntry(byKey, row, col + 1, out ScanCoverDepthGridPointCloud.GridStateEntry e10);
                bool v01 = TryGetValidEntry(byKey, row + 1, col, out ScanCoverDepthGridPointCloud.GridStateEntry e01);
                bool v11 = TryGetValidEntry(byKey, row + 1, col + 1, out ScanCoverDepthGridPointCloud.GridStateEntry e11);

                int validCount = (v00 ? 1 : 0) + (v10 ? 1 : 0) + (v01 ? 1 : 0) + (v11 ? 1 : 0);
                if (validCount < 3)
                    continue;

                if (validCount == 4)
                {
                    TryAddTriangle(triangles, vertexIndices, e00, e10, e11);
                    TryAddTriangle(triangles, vertexIndices, e00, e11, e01);
                    continue;
                }

                if (v00 && v10 && v11) TryAddTriangle(triangles, vertexIndices, e00, e10, e11);
                if (v00 && v11 && v01) TryAddTriangle(triangles, vertexIndices, e00, e11, e01);
                if (v00 && v10 && v01) TryAddTriangle(triangles, vertexIndices, e00, e10, e01);
                if (v10 && v11 && v01) TryAddTriangle(triangles, vertexIndices, e10, e11, e01);
            }
        }
    }

    private void RebuildAcceptedShellMesh()
    {
        EnsureShellObjects();
        if (_mesh == null || _meshFilter == null || _meshRenderer == null)
            return;

        if (_latestAcceptedSnapshot == null || _latestAcceptedSnapshot.entries == null || _latestAcceptedSnapshot.visibleCount <= 0)
        {
            _mesh.Clear();
            if (_shellRoot != null)
                _shellRoot.SetActive(false);
            return;
        }

        _meshVerts.Clear();
        _meshNormals.Clear();
        _meshTriangles.Clear();
        BuildMeshDataFromSnapshot(_latestAcceptedSnapshot, _meshVerts, _meshNormals, _meshTriangles);
        if (_meshVerts.Count <= 0 || _meshTriangles.Count < 3)
        {
            _mesh.Clear();
            if (_shellRoot != null)
                _shellRoot.SetActive(false);
            return;
        }

        _mesh.Clear();
        _mesh.SetVertices(_meshVerts);
        _mesh.SetNormals(_meshNormals);
        _mesh.SetTriangles(_meshTriangles, 0, true);
        _mesh.RecalculateBounds();
        _meshFilter.sharedMesh = _mesh;
        _shellRoot.SetActive(showAcceptedShellMeshInScene);
    }

    private void EnsureShellObjects()
    {
        if (_shellRoot == null)
        {
            _shellRoot = new GameObject("[ScanCover] Line Accepted Shell");
            _shellRoot.transform.SetParent(null, false);
            _meshFilter = _shellRoot.AddComponent<MeshFilter>();
            _meshRenderer = _shellRoot.AddComponent<MeshRenderer>();
        }

        if (_mesh == null)
        {
            _mesh = new Mesh { name = "ScanCover_LineAcceptedShell" };
            _mesh.MarkDynamic();
        }

        if (surfaceMaterialOverride != null)
        {
            if (_runtimeMaterial != null)
            {
                Destroy(_runtimeMaterial);
                _runtimeMaterial = null;
            }
        }
        else if (_runtimeMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard");
            _runtimeMaterial = new Material(shader) { name = "ScanCover_LineAcceptedShell_Mat" };
        }

        Material target = surfaceMaterialOverride != null ? surfaceMaterialOverride : _runtimeMaterial;
        if (target != null)
        {
            if (target.HasProperty("_BaseColor")) target.SetColor("_BaseColor", surfaceColor);
            if (target.HasProperty("_Color")) target.SetColor("_Color", surfaceColor);
            if (target.HasProperty("_Cull")) target.SetFloat("_Cull", surfaceDoubleSided ? (float)CullMode.Off : (float)CullMode.Back);
            if (target.HasProperty("_Surface")) target.SetFloat("_Surface", 0f);
            if (target.HasProperty("_Blend")) target.SetFloat("_Blend", 0f);
            if (target.HasProperty("_ZWrite")) target.SetFloat("_ZWrite", 1f);
        }

        _meshFilter.sharedMesh = _mesh;
        _meshRenderer.sharedMaterial = target;
        _meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _meshRenderer.receiveShadows = true;
        _shellRoot.SetActive(showAcceptedShellMeshInScene && _mesh != null && _mesh.vertexCount > 0);
    }

    private void CreateAcceptedPointSnapshot()
    {
        if (!showAcceptedPointSnapshotsInScene || _latestAcceptedSnapshot == null || _latestAcceptedSnapshot.entries == null || _latestAcceptedSnapshot.visibleCount <= 0)
            return;

        EnsurePointSnapshotObjects();
        if (_pointSnapshotRoot == null)
            return;

        int snapshotIndex = _pointSnapshotCounter;
        GameObject snapshotRoot = new GameObject($"Snapshot_{snapshotIndex:D3}_{_latestAcceptedSnapshot.visibleCount}pts");
        _pointSnapshotCounter++;
        snapshotRoot.transform.SetParent(_pointSnapshotRoot.transform, false);
        snapshotRoot.transform.position = Vector3.zero;
        snapshotRoot.transform.rotation = Quaternion.identity;
        snapshotRoot.transform.localScale = Vector3.one;

        Vector3 scale = Vector3.one * Mathf.Max(0.001f, acceptedPointScaleMeters);
        Material snapshotMaterial = CreatePointSnapshotMaterial(snapshotIndex);
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> byCell = new Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry>(_latestAcceptedSnapshot.entries.Length);
        int maxRow = -1;
        int maxCol = -1;
        for (int i = 0; i < _latestAcceptedSnapshot.entries.Length; i++)
        {
            ScanCoverDepthGridPointCloud.GridStateEntry entry = _latestAcceptedSnapshot.entries[i];
            if (!entry.valid)
                continue;

            byCell[ComposeCellKey(entry.row, entry.col)] = entry;
            if (entry.row > maxRow) maxRow = entry.row;
            if (entry.col > maxCol) maxCol = entry.col;

            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.name = $"Pt_r{entry.row}_c{entry.col}";
            point.transform.SetParent(snapshotRoot.transform, false);
            point.transform.position = entry.worldPos;
            point.transform.rotation = Quaternion.identity;
            point.transform.localScale = scale;

            Collider collider = point.GetComponent<Collider>();
            if (collider != null)
                Destroy(collider);

            MeshRenderer renderer = point.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = snapshotMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
        }

        CreateAcceptedLineSnapshot(snapshotRoot.transform, snapshotMaterial, byCell, maxRow, maxCol);
    }

    private void EnsurePointSnapshotObjects()
    {
        if (_pointSnapshotRoot == null)
        {
            _pointSnapshotRoot = new GameObject("[ScanCover] Line Accepted Point Snapshots");
            _pointSnapshotRoot.transform.SetParent(null, false);
        }
    }

    private Material CreatePointSnapshotMaterial(int snapshotIndex)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
            shader = Shader.Find("Standard");

        Material material = new Material(shader)
        {
            name = $"ScanCover_LineAcceptedPoint_Mat_{snapshotIndex:D3}"
        };

        Color color = ResolveSnapshotColor(snapshotIndex);
        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        if (material.HasProperty("_Cull")) material.SetFloat("_Cull", (float)CullMode.Back);
        if (material.HasProperty("_Surface")) material.SetFloat("_Surface", 0f);
        if (material.HasProperty("_Blend")) material.SetFloat("_Blend", 0f);
        if (material.HasProperty("_ZWrite")) material.SetFloat("_ZWrite", 1f);
        return material;
    }

    private void CreateAcceptedLineSnapshot(
        Transform snapshotRoot,
        Material snapshotMaterial,
        Dictionary<long, ScanCoverDepthGridPointCloud.GridStateEntry> byCell,
        int maxRow,
        int maxCol)
    {
        if (snapshotRoot == null || snapshotMaterial == null || byCell == null || byCell.Count <= 1)
            return;

        float width = Mathf.Max(0.0005f, acceptedLineWidthMeters);
        for (int row = 0; row <= maxRow; row++)
        {
            for (int col = 0; col <= maxCol; col++)
            {
                if (!byCell.TryGetValue(ComposeCellKey(row, col), out ScanCoverDepthGridPointCloud.GridStateEntry current))
                    continue;

                if (byCell.TryGetValue(ComposeCellKey(row, col + 1), out ScanCoverDepthGridPointCloud.GridStateEntry right))
                    CreateLineObject(snapshotRoot, snapshotMaterial, $"H_r{row}_c{col}", current.worldPos, right.worldPos, width);

                if (byCell.TryGetValue(ComposeCellKey(row + 1, col), out ScanCoverDepthGridPointCloud.GridStateEntry down))
                    CreateLineObject(snapshotRoot, snapshotMaterial, $"V_r{row}_c{col}", current.worldPos, down.worldPos, width);
            }
        }
    }

    private static void CreateLineObject(Transform parent, Material material, string name, Vector3 a, Vector3 b, float width)
    {
        if ((b - a).sqrMagnitude <= 1e-8f)
            return;

        GameObject lineObject = new GameObject(name);
        lineObject.transform.SetParent(parent, false);
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.useWorldSpace = true;
        line.positionCount = 2;
        line.SetPosition(0, a);
        line.SetPosition(1, b);
        line.startWidth = width;
        line.endWidth = width;
        line.sharedMaterial = material;
        line.shadowCastingMode = ShadowCastingMode.Off;
        line.receiveShadows = false;
        line.textureMode = LineTextureMode.Stretch;
        line.alignment = LineAlignment.View;
        line.numCapVertices = 0;
        line.numCornerVertices = 0;
    }

    private Color ResolveSnapshotColor(int snapshotIndex)
    {
        if (snapshotIndex <= 1)
            return acceptedPointColor;

        float hue = Mathf.Repeat((snapshotIndex - 1) * 0.173f, 1f);
        Color color = Color.HSVToRGB(hue, 0.78f, 0.98f);
        color.a = 1f;
        return color;
    }

    private static void TryAddTriangle(
        List<int> triangles,
        Dictionary<long, int> vertexIndices,
        ScanCoverDepthGridPointCloud.GridStateEntry a,
        ScanCoverDepthGridPointCloud.GridStateEntry b,
        ScanCoverDepthGridPointCloud.GridStateEntry c)
    {
        int ia = vertexIndices[ComposeCellKey(a.row, a.col)];
        int ib = vertexIndices[ComposeCellKey(b.row, b.col)];
        int ic = vertexIndices[ComposeCellKey(c.row, c.col)];
        if (Vector3.Cross(b.worldPos - a.worldPos, c.worldPos - a.worldPos).sqrMagnitude <= 1e-8f)
            return;
        triangles.Add(ia);
        triangles.Add(ib);
        triangles.Add(ic);
    }

    private string ResolveExportDirectory()
    {
        if (!string.IsNullOrWhiteSpace(exportDirectoryOverride))
            return Path.GetFullPath(exportDirectoryOverride);

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName ?? Application.dataPath;
        return Path.Combine(projectRoot, "ScanCoverExports");
    }

    private static Vector3Int Quantize(Vector3 position, float cellSize)
    {
        float inv = 1f / Mathf.Max(0.001f, cellSize);
        return new Vector3Int(
            Mathf.RoundToInt(position.x * inv),
            Mathf.RoundToInt(position.y * inv),
            Mathf.RoundToInt(position.z * inv));
    }

    private static long ComposeCellKey(int row, int col)
        => ((long)row << 32) | (uint)col;

    private static long ComposeConflictKey(int row, int col, bool horizontal, int otherShellId, int otherRow, int otherCol, bool otherHorizontal, string kind)
    {
        unchecked
        {
            long key = row;
            key = (key * 397) ^ col;
            key = (key * 397) ^ (horizontal ? 1 : 0);
            key = (key * 397) ^ otherShellId;
            key = (key * 397) ^ otherRow;
            key = (key * 397) ^ otherCol;
            key = (key * 397) ^ (otherHorizontal ? 1 : 0);
            key = (key * 397) ^ (kind == "overlap" ? 1 : kind == "crossing" ? 2 : kind == "behind-near" ? 3 : 4);
            return key;
        }
    }

    private static float SegmentDistance(Vector3 p1, Vector3 q1, Vector3 p2, Vector3 q2)
    {
        Vector3 d1 = q1 - p1;
        Vector3 d2 = q2 - p2;
        Vector3 r = p1 - p2;
        float a = Vector3.Dot(d1, d1);
        float e = Vector3.Dot(d2, d2);
        float f = Vector3.Dot(d2, r);

        float s;
        float t;

        if (a <= 1e-8f && e <= 1e-8f)
            return Vector3.Distance(p1, p2);

        if (a <= 1e-8f)
        {
            s = 0f;
            t = Mathf.Clamp01(f / e);
        }
        else
        {
            float c = Vector3.Dot(d1, r);
            if (e <= 1e-8f)
            {
                t = 0f;
                s = Mathf.Clamp01(-c / a);
            }
            else
            {
                float b = Vector3.Dot(d1, d2);
                float denom = a * e - b * b;
                s = denom != 0f ? Mathf.Clamp01((b * f - c * e) / denom) : 0f;
                t = (b * s + f) / e;

                if (t < 0f)
                {
                    t = 0f;
                    s = Mathf.Clamp01(-c / a);
                }
                else if (t > 1f)
                {
                    t = 1f;
                    s = Mathf.Clamp01((b - c) / a);
                }
            }
        }

        Vector3 c1 = p1 + d1 * s;
        Vector3 c2 = p2 + d2 * t;
        return Vector3.Distance(c1, c2);
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog && !string.IsNullOrEmpty(issue))
            Debug.LogWarning($"[ScanCoverDepthGridLineConflictManager] {issue}");
        return false;
    }
}

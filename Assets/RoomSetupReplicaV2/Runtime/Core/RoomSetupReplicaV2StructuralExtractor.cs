using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Reduces the final V2 Environment Depth mesh to supported structural sheets.
/// Direction fitting and extent generation are separate: directions are locked,
/// while sheets are emitted only where a 2D support grid is occupied.
/// </summary>
[DefaultExecutionOrder(20)]
[DisallowMultipleComponent]
[RequireComponent(typeof(RoomSetupReplicaV2Reconstruction))]
public sealed partial class RoomSetupReplicaV2StructuralExtractor : MonoBehaviour
{
    private const float DegreesToRadians = Mathf.PI / 180f;
    private const int MaximumGridDimension = 256;

    [Header("Extraction")]
    [SerializeField] private bool extractAutomatically = true;
    [SerializeField, Min(4)] private int minimumIntegrationCount = 12;
    [SerializeField, Range(0.5f, 10f)]
    private float extractionIntervalSeconds = 2f;
    [SerializeField, Range(5f, 30f)]
    private float axisToleranceDegrees = 15f;
    [SerializeField, Range(0.05f, 0.30f)]
    private float coplanarMergeDistance = 0.20f;
    [SerializeField, Range(0.06f, 0.20f)]
    private float supportCellSize = 0.10f;
    [SerializeField, Range(0f, 0.12f)]
    private float sheetPadding = 0.03f;
    [SerializeField, Range(0.05f, 1f)]
    private float minimumVerticalPlaneArea = 0.12f;
    [SerializeField, Range(0.1f, 3f)]
    private float minimumHorizontalArea = 0.70f;
    [SerializeField, Range(0.03f, 1f)]
    private float minimumSupportSurfaceArea = 0.08f;
    [SerializeField, Range(4, 48)]
    private int maximumRectanglesPerPlane = 24;
    [SerializeField, Range(0.10f, 0.50f)]
    private float wallHeightMergeTolerance = 0.35f;
    [SerializeField, Range(0, 4)]
    private int coplanarGapCloseCells = 4;
    [SerializeField, Range(1, 10)]
    private int horizontalOcclusionGapCells = 6;
    [SerializeField, Range(0.25f, 6f)]
    private float maximumHorizontalOcclusionArea = 3f;
    [SerializeField, Range(0.15f, 1.20f)]
    private float orthogonalConnectionDistance = 0.75f;

    [Header("Preview")]
    [SerializeField] private bool showStructuralPlanes = true;
    [SerializeField] private Color wallColor =
        new Color(1f, 0.55f, 0.10f, 0.22f);
    [SerializeField] private Color floorColor =
        new Color(0.10f, 0.85f, 1f, 0.18f);
    [SerializeField] private Color ceilingColor =
        new Color(0.72f, 0.35f, 1f, 0.18f);
    [SerializeField] private Color supportSurfaceColor =
        new Color(0.18f, 1f, 0.38f, 0.28f);
    [SerializeField] private Color outlineColor =
        new Color(1f, 0.78f, 0.22f, 0.95f);
    [SerializeField, Range(0.002f, 0.04f)]
    private float outlineWidthMeters = 0.012f;

    private readonly List<StructuralSample> _samples = new();
    private readonly List<PlaneReference> _wallFamilyA = new();
    private readonly List<PlaneReference> _wallFamilyB = new();
    private readonly List<PlaneReference> _horizontal = new();
    private readonly List<ExtractedPlane> _planes = new();
    private readonly List<ExtractedPlane> _planesBeforeTopology = new();
    private readonly List<LowObjectBox> _lowObjectBoxes = new();

    private RoomSetupReplicaV2Reconstruction _reconstruction;
    private Transform _previewRoot;
    private Material _wallMaterial;
    private Material _floorMaterial;
    private Material _ceilingMaterial;
    private Material _supportMaterial;
    private Material _outlineMaterial;
    private bool _requestPending;
    private float _nextExtractionTime;
    private int _orthogonalConnectionCount;
    private Vector3 _lastAxisA = Vector3.right;
    private Vector3 _lastAxisB = Vector3.forward;
    private float _lastFloorHeight;

    public string LastStatus { get; private set; } = "waiting";
    public int StructuralSampleCount { get; private set; }
    public int StructuralPlaneCount { get; private set; }

    private struct StructuralSample
    {
        public Vector3 a;
        public Vector3 b;
        public Vector3 c;
        public Vector3 center;
        public Vector3 normal;
        public float area;
    }

    private struct PlaneReference
    {
        public float coordinate;
        public int sampleIndex;
    }

    private sealed class PlaneCluster
    {
        public float coordinate;
        public float weight;
        public readonly List<int> sampleIndices = new();

        public void Add(float plane, float area, int sampleIndex)
        {
            float safeArea = Mathf.Max(area, 0.0001f);
            coordinate = weight > 0f
                ? (coordinate * weight + plane * safeArea) /
                  (weight + safeArea)
                : plane;
            weight += safeArea;
            sampleIndices.Add(sampleIndex);
        }
    }

    private sealed class GridData
    {
        public bool[,] cells;
        public float minU;
        public float minV;
        public float cellSize;
    }

    private enum StructuralKind
    {
        Floor,
        Ceiling,
        Wall,
        SupportSurface
    }

    private struct LowObjectBox
    {
        public Vector3 center;
        public Vector3 axisU;
        public Vector3 axisV;
        public float halfU;
        public float halfV;
        public float bottom;
        public float top;
    }

    private struct ExtractedPlane
    {
        public StructuralKind kind;
        public Vector3 p0;
        public Vector3 p1;
        public Vector3 p2;
        public Vector3 p3;
    }

    private struct HorizontalEnvelope
    {
        public bool hasFloor;
        public bool hasCeiling;
        public float floorHeight;
        public float ceilingHeight;
    }

    private void Awake()
    {
        _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
    }

    private void Update()
    {
        if (!extractAutomatically || _requestPending ||
            _reconstruction == null ||
            _reconstruction.IntegrationCount < minimumIntegrationCount ||
            Time.unscaledTime < _nextExtractionTime)
            return;
        RequestExtraction();
    }

    [ContextMenu("Extract Structural Planes Now")]
    public void RequestExtraction()
    {
        if (_requestPending || _reconstruction == null)
            return;
        _nextExtractionTime = Time.unscaledTime + extractionIntervalSeconds;
        _requestPending = _reconstruction.TryRequestStructuralSamples(
            OnStructuralSamplesReady);
        if (!_requestPending)
            LastStatus = "GPU structural readback not ready";
    }

    public void SetPreviewVisible(bool visible)
    {
        showStructuralPlanes = visible;
        if (_previewRoot != null)
            _previewRoot.gameObject.SetActive(visible);
    }

    public void SetAutomaticExtraction(bool enabled)
    {
        extractAutomatically = enabled;
    }

    public void ClearExtractedPlanes()
    {
        _samples.Clear();
        _planes.Clear();
        _planesBeforeTopology.Clear();
        StructuralSampleCount = 0;
        StructuralPlaneCount = 0;
        LastStatus = "cleared";
        DestroyPreview();
        ClearSimplifiedRoom();
    }

    public bool TryRefreshForExport(Action<bool> completed)
    {
        if (_requestPending || _reconstruction == null)
            return false;
        _requestPending = _reconstruction.TryRequestStructuralSamples(
            packed =>
            {
                bool valid = packed != null && packed.Length >= 4;
                OnStructuralSamplesReady(packed);
                completed?.Invoke(valid && _samples.Count >= 16);
            });
        return _requestPending;
    }

    private void OnStructuralSamplesReady(Vector4[] packed)
    {
        _requestPending = false;
        if (this == null || !isActiveAndEnabled)
            return;
        if (packed == null || packed.Length < 4)
        {
            LastStatus = "GPU structural readback failed";
            return;
        }
        ParseSamples(packed);
        if (_samples.Count < 16)
        {
            LastStatus = "not enough final-mesh samples";
            return;
        }
        ExtractPlanes();
        RebuildPreview();
        int walls = 0;
        int floors = 0;
        int ceilings = 0;
        for (int i = 0; i < _planes.Count; i++)
        {
            if (_planes[i].kind == StructuralKind.Wall) walls++;
            else if (_planes[i].kind == StructuralKind.Floor) floors++;
            else if (_planes[i].kind == StructuralKind.Ceiling) ceilings++;
        }
        LastStatus = $"samples={StructuralSampleCount} " +
                     $"walls={walls} floorSheets={floors} " +
                     $"ceilingSheets={ceilings} " +
                     $"orthogonalConnections={_orthogonalConnectionCount} " +
                     $"furniturePass=off";
        Debug.Log("[RoomSetupReplicaV2] structural extraction " +
                  LastStatus);
    }

    private void ParseSamples(Vector4[] packed)
    {
        _samples.Clear();
        int sampleCount = packed.Length / 4;
        for (int i = 0; i < sampleCount; i++)
        {
            Vector4 av = packed[i * 4];
            Vector4 bv = packed[i * 4 + 1];
            Vector4 cv = packed[i * 4 + 2];
            Vector4 nv = packed[i * 4 + 3];
            if (!IsFinite(av) || !IsFinite(bv) || !IsFinite(cv) ||
                !IsFinite(nv) || av.w <= 0f)
                continue;
            Vector3 normal = new Vector3(nv.x, nv.y, nv.z);
            if (normal.sqrMagnitude < 0.25f)
                continue;
            Vector3 a = new Vector3(av.x, av.y, av.z);
            Vector3 b = new Vector3(bv.x, bv.y, bv.z);
            Vector3 c = new Vector3(cv.x, cv.y, cv.z);
            _samples.Add(new StructuralSample
            {
                a = a,
                b = b,
                c = c,
                center = (a + b + c) / 3f,
                normal = normal.normalized,
                area = av.w
            });
        }
        StructuralSampleCount = _samples.Count;
    }

    private void ExtractPlanes()
    {
        _planes.Clear();
        _planesBeforeTopology.Clear();
        _wallFamilyA.Clear();
        _wallFamilyB.Clear();
        _horizontal.Clear();
        _lowObjectBoxes.Clear();
        _orthogonalConnectionCount = 0;
        if (!TryFindRoomAxes(out Vector3 axisA, out Vector3 axisB))
        {
            LastStatus = "no stable vertical wall direction";
            StructuralPlaneCount = 0;
            return;
        }
        _lastAxisA = axisA;
        _lastAxisB = axisB;
        float tolerance = Mathf.Cos(axisToleranceDegrees * DegreesToRadians);
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            if (Mathf.Abs(sample.normal.y) >= tolerance)
            {
                _horizontal.Add(new PlaneReference
                {
                    coordinate = sample.center.y,
                    sampleIndex = i
                });
                continue;
            }
            Vector3 horizontalNormal = Vector3.ProjectOnPlane(
                sample.normal, Vector3.up);
            if (horizontalNormal.sqrMagnitude < 0.25f)
                continue;
            horizontalNormal.Normalize();
            float alignmentA = Mathf.Abs(Vector3.Dot(
                horizontalNormal, axisA));
            float alignmentB = Mathf.Abs(Vector3.Dot(
                horizontalNormal, axisB));
            if (Mathf.Max(alignmentA, alignmentB) < tolerance)
                continue;
            Vector3 normalAxis = alignmentA >= alignmentB ? axisA : axisB;
            var reference = new PlaneReference
            {
                coordinate = Vector3.Dot(sample.center, normalAxis),
                sampleIndex = i
            };
            if (alignmentA >= alignmentB)
                _wallFamilyA.Add(reference);
            else
                _wallFamilyB.Add(reference);
        }
        HorizontalEnvelope envelope =
            AddHorizontalClusters(axisA, axisB);
        _lastFloorHeight = envelope.floorHeight;
        AddWallClusters(_wallFamilyA, axisA, envelope);
        AddWallClusters(_wallFamilyB, axisB, envelope);
        _planesBeforeTopology.AddRange(_planes);
        ConnectOrthogonalWalls();
        StructuralPlaneCount = _planes.Count;
    }

    private bool TryFindRoomAxes(out Vector3 axisA, out Vector3 axisB)
    {
        double sumCos = 0.0;
        double sumSin = 0.0;
        double totalWeight = 0.0;
        float maximumNormalY = Mathf.Sin(
            axisToleranceDegrees * DegreesToRadians);
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            if (Mathf.Abs(sample.normal.y) > maximumNormalY)
                continue;
            Vector3 horizontal = Vector3.ProjectOnPlane(
                sample.normal, Vector3.up);
            if (horizontal.sqrMagnitude < 0.25f)
                continue;
            horizontal.Normalize();
            double angle = Math.Atan2(horizontal.z, horizontal.x);
            double weight = Math.Max(sample.area, 0.0001f);
            sumCos += Math.Cos(4.0 * angle) * weight;
            sumSin += Math.Sin(4.0 * angle) * weight;
            totalWeight += weight;
        }
        if (totalWeight < 0.25)
        {
            axisA = Vector3.right;
            axisB = Vector3.forward;
            return false;
        }
        float result = (float)(0.25 * Math.Atan2(sumSin, sumCos));
        axisA = new Vector3(
            Mathf.Cos(result), 0f, Mathf.Sin(result)).normalized;
        axisB = Vector3.Cross(Vector3.up, axisA).normalized;
        return true;
    }

    private List<PlaneCluster> BuildClusters(List<PlaneReference> references)
    {
        references.Sort((left, right) =>
            left.coordinate.CompareTo(right.coordinate));
        List<PlaneCluster> clusters = new();
        PlaneCluster cluster = null;
        for (int i = 0; i < references.Count; i++)
        {
            PlaneReference item = references[i];
            if (cluster == null ||
                Mathf.Abs(item.coordinate - cluster.coordinate) >
                coplanarMergeDistance)
            {
                cluster = new PlaneCluster();
                clusters.Add(cluster);
            }
            cluster.Add(item.coordinate,
                _samples[item.sampleIndex].area, item.sampleIndex);
        }
        return clusters;
    }

    private void AddWallClusters(List<PlaneReference> references,
                                 Vector3 normalAxis,
                                 HorizontalEnvelope envelope)
    {
        if (references.Count == 0)
            return;
        List<PlaneCluster> clusters = BuildClusters(references);
        Vector3 tangentAxis = Vector3.Cross(Vector3.up, normalAxis);
        for (int i = 0; i < clusters.Count; i++)
        {
            PlaneCluster cluster = clusters[i];
            if (cluster.weight < minimumVerticalPlaneArea)
                continue;
            GridData grid = BuildSupportGrid(
                cluster.sampleIndices, tangentAxis, Vector3.up);
            if (grid == null)
                continue;
            AddGroundedWallSheets(grid,
                normalAxis * cluster.coordinate, tangentAxis,
                envelope.floorHeight);
        }
    }

    private HorizontalEnvelope AddHorizontalClusters(Vector3 roomAxisA,
                                                     Vector3 roomAxisB)
    {
        HorizontalEnvelope envelope = new HorizontalEnvelope();
        if (_horizontal.Count == 0)
        {
            envelope.floorHeight =
                ResolveObserverHeight(new List<PlaneCluster>()) - 1.55f;
            return envelope;
        }
        List<PlaneCluster> clusters = BuildClusters(_horizontal);
        float observerHeight = ResolveObserverHeight(clusters);
        int floorIndex = -1;
        int ceilingIndex = -1;
        float floorWeight = 0f;
        float ceilingWeight = 0f;
        for (int i = 0; i < clusters.Count; i++)
        {
            if (clusters[i].weight < minimumHorizontalArea ||
                clusters[i].coordinate > observerHeight - 0.35f)
                continue;
            if (clusters[i].weight > floorWeight)
            {
                floorIndex = i;
                floorWeight = clusters[i].weight;
            }
        }
        for (int i = 0; i < clusters.Count; i++)
        {
            if (clusters[i].coordinate >= observerHeight + 0.30f &&
                clusters[i].weight >= minimumHorizontalArea &&
                clusters[i].weight > ceilingWeight)
            {
                ceilingIndex = i;
                ceilingWeight = clusters[i].weight;
            }
        }

        if (floorIndex >= 0)
        {
            envelope.hasFloor = true;
            envelope.floorHeight = clusters[floorIndex].coordinate;
        }
        else
        {
            envelope.floorHeight = observerHeight - 1.55f;
        }
        if (ceilingIndex >= 0)
        {
            envelope.hasCeiling = true;
            envelope.ceilingHeight = clusters[ceilingIndex].coordinate;
        }

        if (floorIndex >= 0)
        {
            PlaneCluster floor = clusters[floorIndex];
            GridData grid = BuildSupportGrid(
                floor.sampleIndices, roomAxisA, roomAxisB);
            if (grid != null)
            {
                ConsolidateHorizontalGrid(grid);
                AddGridRectangles(grid, StructuralKind.Floor,
                    Vector3.up * floor.coordinate,
                    roomAxisA, roomAxisB, 0.35f, 0.35f);
            }
        }
        if (ceilingIndex >= 0)
        {
            PlaneCluster ceiling = clusters[ceilingIndex];
            GridData grid = BuildSupportGrid(
                ceiling.sampleIndices, roomAxisA, roomAxisB);
            if (grid != null)
            {
                ConsolidateHorizontalGrid(grid);
                AddGridRectangles(grid, StructuralKind.Ceiling,
                    Vector3.up * ceiling.coordinate,
                    roomAxisA, roomAxisB, 0.35f, 0.35f);
            }
        }
        return envelope;
    }

    private void AddGroundedWallSheets(GridData grid,
                                       Vector3 origin,
                                       Vector3 tangentAxis,
                                       float floorHeight)
    {
        int width = grid.cells.GetLength(0);
        int height = grid.cells.GetLength(1);
        float[] tops = new float[width];
        for (int x = 0; x < width; x++)
        {
            tops[x] = float.NaN;
            for (int y = height - 1; y >= 0; y--)
            {
                if (!grid.cells[x, y])
                    continue;
                tops[x] = grid.minV + (y + 1) * grid.cellSize;
                break;
            }
        }
        CloseColumnGaps(tops, coplanarGapCloseCells);

        int cursor = 0;
        float padding = Mathf.Min(sheetPadding, grid.cellSize * 0.25f);
        while (cursor < width)
        {
            while (cursor < width && !float.IsFinite(tops[cursor]))
                cursor++;
            if (cursor >= width)
                break;
            int start = cursor;
            float minimumTop = tops[cursor];
            float maximumTop = tops[cursor];
            cursor++;
            while (cursor < width && float.IsFinite(tops[cursor]))
            {
                float nextMinimum = Mathf.Min(minimumTop, tops[cursor]);
                float nextMaximum = Mathf.Max(maximumTop, tops[cursor]);
                if (nextMaximum - nextMinimum >
                    wallHeightMergeTolerance)
                    break;
                minimumTop = nextMinimum;
                maximumTop = nextMaximum;
                cursor++;
            }
            int end = cursor;
            float runWidth = (end - start) * grid.cellSize;
            if (runWidth < 0.20f || maximumTop <= floorHeight + 0.08f)
                continue;
            float minU = grid.minU + start * grid.cellSize - padding;
            float maxU = grid.minU + end * grid.cellSize + padding;
            _planes.Add(new ExtractedPlane
            {
                kind = StructuralKind.Wall,
                p0 = origin + tangentAxis * minU +
                     Vector3.up * floorHeight,
                p1 = origin + tangentAxis * maxU +
                     Vector3.up * floorHeight,
                p2 = origin + tangentAxis * maxU +
                     Vector3.up * maximumTop,
                p3 = origin + tangentAxis * minU +
                     Vector3.up * maximumTop
            });
        }
    }

    private static void CloseColumnGaps(float[] tops, int maximumGap)
    {
        if (maximumGap <= 0)
            return;
        int index = 0;
        while (index < tops.Length)
        {
            if (float.IsFinite(tops[index]))
            {
                index++;
                continue;
            }
            int start = index;
            while (index < tops.Length && !float.IsFinite(tops[index]))
                index++;
            int gap = index - start;
            if (gap > maximumGap || start == 0 || index >= tops.Length ||
                !float.IsFinite(tops[start - 1]) ||
                !float.IsFinite(tops[index]))
                continue;
            float fill = Mathf.Min(tops[start - 1], tops[index]);
            for (int x = start; x < index; x++)
                tops[x] = fill;
        }
    }

    private void ConnectOrthogonalWalls()
    {
        int planeCount = _planes.Count;
        Vector3[] startTargets = new Vector3[planeCount];
        Vector3[] endTargets = new Vector3[planeCount];
        float[] startDistances = new float[planeCount];
        float[] endDistances = new float[planeCount];
        bool[] hasStartTarget = new bool[planeCount];
        bool[] hasEndTarget = new bool[planeCount];
        for (int i = 0; i < planeCount; i++)
        {
            startDistances[i] = float.PositiveInfinity;
            endDistances[i] = float.PositiveInfinity;
        }

        for (int i = 0; i < planeCount; i++)
        {
            if (_planes[i].kind != StructuralKind.Wall)
                continue;
            ExtractedPlane first = _planes[i];
            Vector2 firstStart = Horizontal(first.p0);
            Vector2 firstEnd = Horizontal(first.p1);
            Vector2 firstDelta = firstEnd - firstStart;
            float firstLength = firstDelta.magnitude;
            if (firstLength < 0.05f)
                continue;

            for (int j = i + 1; j < planeCount; j++)
            {
                if (_planes[j].kind != StructuralKind.Wall)
                    continue;
                ExtractedPlane second = _planes[j];
                Vector2 secondStart = Horizontal(second.p0);
                Vector2 secondEnd = Horizontal(second.p1);
                Vector2 secondDelta = secondEnd - secondStart;
                float secondLength = secondDelta.magnitude;
                if (secondLength < 0.05f ||
                    Mathf.Abs(Vector2.Dot(firstDelta / firstLength,
                                           secondDelta / secondLength)) > 0.25f)
                    continue;
                if (!TryIntersectLines(firstStart, firstDelta,
                                       secondStart, secondDelta,
                                       out Vector2 intersection,
                                       out float firstParameter,
                                       out float secondParameter))
                    continue;
                float firstReach = orthogonalConnectionDistance / firstLength;
                float secondReach = orthogonalConnectionDistance / secondLength;
                if (firstParameter < -firstReach ||
                    firstParameter > 1f + firstReach ||
                    secondParameter < -secondReach ||
                    secondParameter > 1f + secondReach)
                    continue;

                Vector3 worldIntersection = new Vector3(
                    intersection.x, first.p0.y, intersection.y);
                ConsiderWallEndpoint(i, firstParameter, firstLength,
                    worldIntersection, startTargets, endTargets,
                    startDistances, endDistances,
                    hasStartTarget, hasEndTarget);
                ConsiderWallEndpoint(j, secondParameter, secondLength,
                    worldIntersection, startTargets, endTargets,
                    startDistances, endDistances,
                    hasStartTarget, hasEndTarget);
            }
        }

        for (int i = 0; i < planeCount; i++)
        {
            if (_planes[i].kind != StructuralKind.Wall)
                continue;
            ExtractedPlane plane = _planes[i];
            bool changed = false;
            if (hasStartTarget[i])
            {
                SetHorizontal(ref plane.p0, startTargets[i]);
                SetHorizontal(ref plane.p3, startTargets[i]);
                changed = true;
            }
            if (hasEndTarget[i])
            {
                SetHorizontal(ref plane.p1, endTargets[i]);
                SetHorizontal(ref plane.p2, endTargets[i]);
                changed = true;
            }
            if (!changed)
                continue;
            _planes[i] = plane;
            _orthogonalConnectionCount +=
                (hasStartTarget[i] ? 1 : 0) +
                (hasEndTarget[i] ? 1 : 0);
        }
    }

    private void ConsiderWallEndpoint(
        int planeIndex, float parameter, float length,
        Vector3 intersection,
        Vector3[] startTargets, Vector3[] endTargets,
        float[] startDistances, float[] endDistances,
        bool[] hasStartTarget, bool[] hasEndTarget)
    {
        float startDistance = Mathf.Abs(parameter) * length;
        float endDistance = Mathf.Abs(parameter - 1f) * length;
        if (startDistance <= endDistance &&
            startDistance <= orthogonalConnectionDistance &&
            startDistance < startDistances[planeIndex])
        {
            startDistances[planeIndex] = startDistance;
            startTargets[planeIndex] = intersection;
            hasStartTarget[planeIndex] = true;
        }
        else if (endDistance <= orthogonalConnectionDistance &&
                 endDistance < endDistances[planeIndex])
        {
            endDistances[planeIndex] = endDistance;
            endTargets[planeIndex] = intersection;
            hasEndTarget[planeIndex] = true;
        }
    }

    private static bool TryIntersectLines(
        Vector2 firstOrigin, Vector2 firstDirection,
        Vector2 secondOrigin, Vector2 secondDirection,
        out Vector2 intersection,
        out float firstParameter, out float secondParameter)
    {
        float denominator = Cross(firstDirection, secondDirection);
        if (Mathf.Abs(denominator) < 0.0001f)
        {
            intersection = default;
            firstParameter = 0f;
            secondParameter = 0f;
            return false;
        }
        Vector2 offset = secondOrigin - firstOrigin;
        firstParameter = Cross(offset, secondDirection) / denominator;
        secondParameter = Cross(offset, firstDirection) / denominator;
        intersection = firstOrigin + firstDirection * firstParameter;
        return true;
    }

    private static Vector2 Horizontal(Vector3 point)
    {
        return new Vector2(point.x, point.z);
    }

    private static void SetHorizontal(ref Vector3 point, Vector3 target)
    {
        point.x = target.x;
        point.z = target.z;
    }

    private void AddSupportSurfaceBoxes(PlaneCluster cluster,
                                        float floorHeight,
                                        Vector3 roomAxisA,
                                        Vector3 roomAxisB)
    {
        GridData grid = BuildSupportGrid(
            cluster.sampleIndices, roomAxisA, roomAxisB);
        if (grid == null)
            return;
        List<List<Vector2Int>> components = ExtractComponents(grid.cells);
        int minimumCells = Mathf.Max(3, Mathf.CeilToInt(
            minimumSupportSurfaceArea /
            (grid.cellSize * grid.cellSize) * 0.35f));
        for (int i = 0; i < components.Count; i++)
        {
            List<Vector2Int> component = components[i];
            if (component.Count < minimumCells)
                continue;
            List<Vector2> worldPoints = new(component.Count);
            for (int j = 0; j < component.Count; j++)
            {
                Vector2Int cell = component[j];
                float u = grid.minU + (cell.x + 0.5f) * grid.cellSize;
                float v = grid.minV + (cell.y + 0.5f) * grid.cellSize;
                Vector3 world = roomAxisA * u + roomAxisB * v;
                worldPoints.Add(new Vector2(world.x, world.z));
            }
            if (!TryFitMinimumAreaBox(worldPoints, roomAxisA, roomAxisB,
                    grid.cellSize, out LowObjectBox box))
                continue;
            box.bottom = floorHeight;
            box.top = cluster.coordinate;
            if (box.top < box.bottom + 0.25f ||
                box.halfU * 2f < 0.20f || box.halfV * 2f < 0.20f)
                continue;
            _lowObjectBoxes.Add(box);
            AddLowObjectBoxSheets(box);
        }
    }

    private static List<List<Vector2Int>> ExtractComponents(bool[,] cells)
    {
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        bool[,] visited = new bool[width, height];
        List<List<Vector2Int>> result = new();
        int[] dx = { -1, 1, 0, 0 };
        int[] dy = { 0, 0, -1, 1 };
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            if (!cells[x, y] || visited[x, y])
                continue;
            List<Vector2Int> component = new();
            Queue<Vector2Int> queue = new();
            queue.Enqueue(new Vector2Int(x, y));
            visited[x, y] = true;
            while (queue.Count > 0)
            {
                Vector2Int cell = queue.Dequeue();
                component.Add(cell);
                for (int direction = 0; direction < 4; direction++)
                {
                    int nx = cell.x + dx[direction];
                    int ny = cell.y + dy[direction];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height ||
                        visited[nx, ny] || !cells[nx, ny])
                        continue;
                    visited[nx, ny] = true;
                    queue.Enqueue(new Vector2Int(nx, ny));
                }
            }
            result.Add(component);
        }
        return result;
    }

    private static bool TryFitMinimumAreaBox(
        List<Vector2> points,
        Vector3 roomAxisA,
        Vector3 roomAxisB,
        float cellSize,
        out LowObjectBox box)
    {
        box = new LowObjectBox();
        if (points.Count < 3)
            return false;
        List<Vector2> hull = BuildConvexHull(points);
        if (hull.Count < 2)
            return false;
        float bestArea = float.PositiveInfinity;
        Vector2 bestU = Vector2.right;
        Vector2 bestV = Vector2.up;
        float bestMinU = 0f;
        float bestMaxU = 0f;
        float bestMinV = 0f;
        float bestMaxV = 0f;
        for (int i = 0; i < hull.Count; i++)
        {
            Vector2 edge = hull[(i + 1) % hull.Count] - hull[i];
            if (edge.sqrMagnitude < 0.000001f)
                continue;
            Vector2 axisU = edge.normalized;
            Vector2 axisV = new Vector2(-axisU.y, axisU.x);
            FitAtAxes(points, axisU, axisV,
                out float minU, out float maxU,
                out float minV, out float maxV);
            float area = (maxU - minU) * (maxV - minV);
            if (area >= bestArea)
                continue;
            bestArea = area;
            bestU = axisU;
            bestV = axisV;
            bestMinU = minU;
            bestMaxU = maxU;
            bestMinV = minV;
            bestMaxV = maxV;
        }
        float lengthU = bestMaxU - bestMinU;
        float lengthV = bestMaxV - bestMinV;
        if (Mathf.Max(lengthU, lengthV) /
            Mathf.Max(0.001f, Mathf.Min(lengthU, lengthV)) < 1.15f)
        {
            bestU = new Vector2(roomAxisA.x, roomAxisA.z).normalized;
            bestV = new Vector2(roomAxisB.x, roomAxisB.z).normalized;
            FitAtAxes(points, bestU, bestV,
                out bestMinU, out bestMaxU,
                out bestMinV, out bestMaxV);
        }
        Vector2 center2 = bestU * ((bestMinU + bestMaxU) * 0.5f) +
                          bestV * ((bestMinV + bestMaxV) * 0.5f);
        box.center = new Vector3(center2.x, 0f, center2.y);
        box.axisU = new Vector3(bestU.x, 0f, bestU.y).normalized;
        box.axisV = new Vector3(bestV.x, 0f, bestV.y).normalized;
        box.halfU = (bestMaxU - bestMinU) * 0.5f + cellSize * 0.5f;
        box.halfV = (bestMaxV - bestMinV) * 0.5f + cellSize * 0.5f;
        return float.IsFinite(bestArea);
    }

    private static void FitAtAxes(List<Vector2> points,
                                  Vector2 axisU,
                                  Vector2 axisV,
                                  out float minU,
                                  out float maxU,
                                  out float minV,
                                  out float maxV)
    {
        minU = minV = float.PositiveInfinity;
        maxU = maxV = float.NegativeInfinity;
        for (int i = 0; i < points.Count; i++)
        {
            float u = Vector2.Dot(points[i], axisU);
            float v = Vector2.Dot(points[i], axisV);
            minU = Mathf.Min(minU, u);
            maxU = Mathf.Max(maxU, u);
            minV = Mathf.Min(minV, v);
            maxV = Mathf.Max(maxV, v);
        }
    }

    private static List<Vector2> BuildConvexHull(List<Vector2> points)
    {
        List<Vector2> sorted = new(points);
        sorted.Sort((a, b) =>
        {
            int x = a.x.CompareTo(b.x);
            return x != 0 ? x : a.y.CompareTo(b.y);
        });
        if (sorted.Count <= 2)
            return sorted;
        List<Vector2> hull = new();
        for (int i = 0; i < sorted.Count; i++)
        {
            while (hull.Count >= 2 && Cross(
                       hull[hull.Count - 1] - hull[hull.Count - 2],
                       sorted[i] - hull[hull.Count - 1]) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(sorted[i]);
        }
        int lowerCount = hull.Count;
        for (int i = sorted.Count - 2; i >= 0; i--)
        {
            while (hull.Count > lowerCount && Cross(
                       hull[hull.Count - 1] - hull[hull.Count - 2],
                       sorted[i] - hull[hull.Count - 1]) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(sorted[i]);
        }
        if (hull.Count > 1)
            hull.RemoveAt(hull.Count - 1);
        return hull;
    }

    private static float Cross(Vector2 a, Vector2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    private void AddLowObjectBoxSheets(LowObjectBox box)
    {
        Vector3 u = box.axisU * box.halfU;
        Vector3 v = box.axisV * box.halfV;
        Vector3 top = box.center + Vector3.up * box.top;
        Vector3 bottom = box.center + Vector3.up * box.bottom;
        Vector3 t0 = top - u - v;
        Vector3 t1 = top + u - v;
        Vector3 t2 = top + u + v;
        Vector3 t3 = top - u + v;
        Vector3 b0 = bottom - u - v;
        Vector3 b1 = bottom + u - v;
        Vector3 b2 = bottom + u + v;
        Vector3 b3 = bottom - u + v;
        AddPlane(StructuralKind.SupportSurface, t0, t1, t2, t3);
        AddPlane(StructuralKind.SupportSurface, b0, b1, t1, t0);
        AddPlane(StructuralKind.SupportSurface, b1, b2, t2, t1);
        AddPlane(StructuralKind.SupportSurface, b2, b3, t3, t2);
        AddPlane(StructuralKind.SupportSurface, b3, b0, t0, t3);
    }

    private void AddPlane(StructuralKind kind,
                          Vector3 p0, Vector3 p1,
                          Vector3 p2, Vector3 p3)
    {
        _planes.Add(new ExtractedPlane
        {
            kind = kind,
            p0 = p0,
            p1 = p1,
            p2 = p2,
            p3 = p3
        });
    }

    private void RemoveLowObjectWallSamples(List<PlaneReference> references)
    {
        for (int i = references.Count - 1; i >= 0; i--)
        {
            Vector3 point = _samples[references[i].sampleIndex].center;
            if (IsInsideLowObjectBox(point))
                references.RemoveAt(i);
        }
    }

    private bool IsInsideLowObjectBox(Vector3 point)
    {
        for (int i = 0; i < _lowObjectBoxes.Count; i++)
        {
            LowObjectBox box = _lowObjectBoxes[i];
            if (point.y < box.bottom - 0.10f || point.y > box.top + 0.10f)
                continue;
            Vector3 delta = point - box.center;
            if (Mathf.Abs(Vector3.Dot(delta, box.axisU)) <=
                    box.halfU + 0.08f &&
                Mathf.Abs(Vector3.Dot(delta, box.axisV)) <=
                    box.halfV + 0.08f)
                return true;
        }
        return false;
    }

    private GridData BuildSupportGrid(List<int> indices,
                                      Vector3 axisU,
                                      Vector3 axisV)
    {
        if (indices.Count == 0)
            return null;
        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < indices.Count; i++)
        {
            StructuralSample sample = _samples[indices[i]];
            IncludePoint(sample.a, axisU, axisV,
                ref minU, ref maxU, ref minV, ref maxV);
            IncludePoint(sample.b, axisU, axisV,
                ref minU, ref maxU, ref minV, ref maxV);
            IncludePoint(sample.c, axisU, axisV,
                ref minU, ref maxU, ref minV, ref maxV);
        }
        if (!float.IsFinite(minU) || maxU <= minU || maxV <= minV)
            return null;
        float cell = Mathf.Max(supportCellSize,
            Mathf.Max((maxU - minU) / MaximumGridDimension,
                      (maxV - minV) / MaximumGridDimension));
        int width = Mathf.Clamp(
            Mathf.CeilToInt((maxU - minU) / cell) + 1,
            1, MaximumGridDimension);
        int height = Mathf.Clamp(
            Mathf.CeilToInt((maxV - minV) / cell) + 1,
            1, MaximumGridDimension);
        bool[,] cells = new bool[width, height];
        for (int i = 0; i < indices.Count; i++)
        {
            StructuralSample sample = _samples[indices[i]];
            MarkTriangleBounds(cells, minU, minV, cell,
                Project(sample.a, axisU, axisV),
                Project(sample.b, axisU, axisV),
                Project(sample.c, axisU, axisV));
        }
        FillNarrowGaps(cells);
        return new GridData
        {
            cells = cells,
            minU = minU,
            minV = minV,
            cellSize = cell
        };
    }

    private static void IncludePoint(Vector3 point,
                                     Vector3 axisU,
                                     Vector3 axisV,
                                     ref float minU,
                                     ref float maxU,
                                     ref float minV,
                                     ref float maxV)
    {
        float u = Vector3.Dot(point, axisU);
        float v = Vector3.Dot(point, axisV);
        minU = Mathf.Min(minU, u);
        maxU = Mathf.Max(maxU, u);
        minV = Mathf.Min(minV, v);
        maxV = Mathf.Max(maxV, v);
    }

    private static Vector2 Project(Vector3 point,
                                   Vector3 axisU,
                                   Vector3 axisV)
    {
        return new Vector2(Vector3.Dot(point, axisU),
                           Vector3.Dot(point, axisV));
    }

    private static void MarkTriangleBounds(bool[,] cells,
                                           float minU,
                                           float minV,
                                           float cell,
                                           Vector2 a,
                                           Vector2 b,
                                           Vector2 c)
    {
        float lowU = Mathf.Min(a.x, Mathf.Min(b.x, c.x));
        float highU = Mathf.Max(a.x, Mathf.Max(b.x, c.x));
        float lowV = Mathf.Min(a.y, Mathf.Min(b.y, c.y));
        float highV = Mathf.Max(a.y, Mathf.Max(b.y, c.y));
        int x0 = Mathf.Clamp(Mathf.FloorToInt((lowU - minU) / cell),
                             0, cells.GetLength(0) - 1);
        int x1 = Mathf.Clamp(Mathf.FloorToInt((highU - minU) / cell),
                             0, cells.GetLength(0) - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt((lowV - minV) / cell),
                             0, cells.GetLength(1) - 1);
        int y1 = Mathf.Clamp(Mathf.FloorToInt((highV - minV) / cell),
                             0, cells.GetLength(1) - 1);
        for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++)
            cells[x, y] = true;
    }

    private static void FillNarrowGaps(bool[,] cells)
    {
        bool[,] source = (bool[,])cells.Clone();
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        for (int y = 1; y < height - 1; y++)
        for (int x = 1; x < width - 1; x++)
        {
            if (source[x, y])
                continue;
            int neighbours = 0;
            for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++)
                if ((ox != 0 || oy != 0) && source[x + ox, y + oy])
                    neighbours++;
            if ((source[x - 1, y] && source[x + 1, y]) ||
                (source[x, y - 1] && source[x, y + 1]) ||
                neighbours >= 5)
                cells[x, y] = true;
        }
    }

    private void ConsolidateHorizontalGrid(GridData grid)
    {
        bool[,] cells = grid.cells;
        FillLinearGaps(cells, horizontalOcclusionGapCells, true);
        FillLinearGaps(cells, horizontalOcclusionGapCells, false);
        int maximumHoleCells = Mathf.CeilToInt(
            maximumHorizontalOcclusionArea /
            (grid.cellSize * grid.cellSize));
        FillEnclosedHoles(cells, maximumHoleCells);
        FillNarrowGaps(cells);
    }

    private static void FillLinearGaps(bool[,] cells,
                                       int maximumGap,
                                       bool alongRows)
    {
        int primaryCount = alongRows
            ? cells.GetLength(1) : cells.GetLength(0);
        int secondaryCount = alongRows
            ? cells.GetLength(0) : cells.GetLength(1);
        for (int primary = 0; primary < primaryCount; primary++)
        {
            int secondary = 0;
            while (secondary < secondaryCount)
            {
                bool occupied = alongRows
                    ? cells[secondary, primary]
                    : cells[primary, secondary];
                if (occupied)
                {
                    secondary++;
                    continue;
                }
                int start = secondary;
                while (secondary < secondaryCount)
                {
                    bool empty = alongRows
                        ? !cells[secondary, primary]
                        : !cells[primary, secondary];
                    if (!empty)
                        break;
                    secondary++;
                }
                int gap = secondary - start;
                if (gap > maximumGap || start == 0 ||
                    secondary >= secondaryCount)
                    continue;
                for (int fill = start; fill < secondary; fill++)
                {
                    if (alongRows) cells[fill, primary] = true;
                    else cells[primary, fill] = true;
                }
            }
        }
    }

    private static void FillEnclosedHoles(bool[,] cells,
                                          int maximumHoleCells)
    {
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        bool[,] exterior = new bool[width, height];
        Queue<Vector2Int> queue = new();
        for (int x = 0; x < width; x++)
        {
            EnqueueExterior(x, 0, cells, exterior, queue);
            EnqueueExterior(x, height - 1, cells, exterior, queue);
        }
        for (int y = 0; y < height; y++)
        {
            EnqueueExterior(0, y, cells, exterior, queue);
            EnqueueExterior(width - 1, y, cells, exterior, queue);
        }
        int[] dx = { -1, 1, 0, 0 };
        int[] dy = { 0, 0, -1, 1 };
        while (queue.Count > 0)
        {
            Vector2Int cell = queue.Dequeue();
            for (int direction = 0; direction < 4; direction++)
            {
                int nx = cell.x + dx[direction];
                int ny = cell.y + dy[direction];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                EnqueueExterior(nx, ny, cells, exterior, queue);
            }
        }
        bool[,] handled = new bool[width, height];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            if (cells[x, y] || exterior[x, y] || handled[x, y])
                continue;
            List<Vector2Int> hole = new();
            Queue<Vector2Int> holeQueue = new();
            holeQueue.Enqueue(new Vector2Int(x, y));
            handled[x, y] = true;
            while (holeQueue.Count > 0)
            {
                Vector2Int cell = holeQueue.Dequeue();
                hole.Add(cell);
                for (int direction = 0; direction < 4; direction++)
                {
                    int nx = cell.x + dx[direction];
                    int ny = cell.y + dy[direction];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height ||
                        cells[nx, ny] || exterior[nx, ny] ||
                        handled[nx, ny])
                        continue;
                    handled[nx, ny] = true;
                    holeQueue.Enqueue(new Vector2Int(nx, ny));
                }
            }
            if (hole.Count > maximumHoleCells)
                continue;
            for (int i = 0; i < hole.Count; i++)
                cells[hole[i].x, hole[i].y] = true;
        }
    }

    private static void EnqueueExterior(int x, int y,
                                        bool[,] cells,
                                        bool[,] exterior,
                                        Queue<Vector2Int> queue)
    {
        if (cells[x, y] || exterior[x, y])
            return;
        exterior[x, y] = true;
        queue.Enqueue(new Vector2Int(x, y));
    }

    private void AddGridRectangles(GridData grid,
                                   StructuralKind kind,
                                   Vector3 origin,
                                   Vector3 axisU,
                                   Vector3 axisV,
                                   float minimumWidth,
                                   float minimumHeight)
    {
        bool[,] remaining = (bool[,])grid.cells.Clone();
        float padding = Mathf.Min(sheetPadding, grid.cellSize * 0.25f);
        for (int iteration = 0;
             iteration < maximumRectanglesPerPlane;
             iteration++)
        {
            RectInt rectangle = FindLargestRectangle(remaining);
            if (rectangle.width == 0 || rectangle.height == 0 ||
                rectangle.width * rectangle.height < 3)
                break;
            for (int y = rectangle.yMin; y < rectangle.yMax; y++)
            for (int x = rectangle.xMin; x < rectangle.xMax; x++)
                remaining[x, y] = false;
            float width = rectangle.width * grid.cellSize;
            float height = rectangle.height * grid.cellSize;
            if (width < minimumWidth || height < minimumHeight)
                continue;
            float minU = grid.minU + rectangle.xMin * grid.cellSize -
                         padding;
            float maxU = grid.minU + rectangle.xMax * grid.cellSize +
                         padding;
            float minV = grid.minV + rectangle.yMin * grid.cellSize -
                         padding;
            float maxV = grid.minV + rectangle.yMax * grid.cellSize +
                         padding;
            _planes.Add(new ExtractedPlane
            {
                kind = kind,
                p0 = origin + axisU * minU + axisV * minV,
                p1 = origin + axisU * maxU + axisV * minV,
                p2 = origin + axisU * maxU + axisV * maxV,
                p3 = origin + axisU * minU + axisV * maxV
            });
        }
    }

    private static RectInt FindLargestRectangle(bool[,] cells)
    {
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        int[] bars = new int[width];
        int[] stack = new int[width + 1];
        int bestArea = 0;
        RectInt best = new RectInt();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
                bars[x] = cells[x, y] ? bars[x] + 1 : 0;
            int stackCount = 0;
            for (int x = 0; x <= width; x++)
            {
                int current = x < width ? bars[x] : 0;
                while (stackCount > 0 &&
                       bars[stack[stackCount - 1]] > current)
                {
                    int index = stack[--stackCount];
                    int barHeight = bars[index];
                    int start = stackCount == 0
                        ? 0 : stack[stackCount - 1] + 1;
                    int rectWidth = x - start;
                    int area = rectWidth * barHeight;
                    if (area <= bestArea)
                        continue;
                    bestArea = area;
                    best = new RectInt(start, y - barHeight + 1,
                                       rectWidth, barHeight);
                }
                if (x < width)
                    stack[stackCount++] = x;
            }
        }
        return best;
    }

    private static float ResolveObserverHeight(List<PlaneCluster> clusters)
    {
        Camera camera = Camera.main;
        if (camera == null)
            camera = FindAnyObjectByType<Camera>();
        if (camera != null)
            return camera.transform.position.y;
        if (clusters.Count == 0)
            return 1.5f;
        return (clusters[0].coordinate +
                clusters[clusters.Count - 1].coordinate) * 0.5f;
    }

    private void RebuildPreview()
    {
        EnsureMaterials();
        DestroyPreview();
        var root = new GameObject("V2 Extracted Structural Planes");
        root.transform.SetParent(transform, false);
        _previewRoot = root.transform;
        root.SetActive(showStructuralPlanes);
        int walls = 0;
        int floors = 0;
        int ceilings = 0;
        int supports = 0;
        for (int i = 0; i < _planes.Count; i++)
        {
            ExtractedPlane plane = _planes[i];
            string name;
            Material material;
            switch (plane.kind)
            {
                case StructuralKind.Floor:
                    name = $"Floor_{floors++:00}";
                    material = _floorMaterial;
                    break;
                case StructuralKind.Ceiling:
                    name = $"Ceiling_{ceilings++:00}";
                    material = _ceilingMaterial;
                    break;
                case StructuralKind.SupportSurface:
                    name = $"SupportSurface_{supports++:00}";
                    material = _supportMaterial;
                    break;
                default:
                    name = $"Wall_{walls++:00}";
                    material = _wallMaterial;
                    break;
            }
            CreatePlaneObject(name, plane, material);
        }
    }

    private void CreatePlaneObject(string name,
                                   ExtractedPlane plane,
                                   Material material)
    {
        var planeObject = new GameObject(name);
        planeObject.transform.SetParent(_previewRoot, false);
        var filter = planeObject.AddComponent<MeshFilter>();
        var renderer = planeObject.AddComponent<MeshRenderer>();
        var mesh = new Mesh { name = name + " Mesh" };
        mesh.vertices = new[]
        {
            transform.InverseTransformPoint(plane.p0),
            transform.InverseTransformPoint(plane.p1),
            transform.InverseTransformPoint(plane.p2),
            transform.InverseTransformPoint(plane.p3)
        };
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        filter.sharedMesh = mesh;
        renderer.sharedMaterial = material;
        renderer.shadowCastingMode =
            UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        var outline = planeObject.AddComponent<LineRenderer>();
        outline.useWorldSpace = false;
        outline.loop = true;
        outline.positionCount = 4;
        outline.SetPositions(mesh.vertices);
        outline.startWidth = outlineWidthMeters;
        outline.endWidth = outlineWidthMeters;
        outline.sharedMaterial = _outlineMaterial;
        outline.shadowCastingMode =
            UnityEngine.Rendering.ShadowCastingMode.Off;
        outline.receiveShadows = false;
    }

    private void EnsureMaterials()
    {
        if (_wallMaterial != null)
            return;
        Shader shader = Resources.Load<Shader>(
            "RoomSetupReplicaV2StructuralPlane");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        _wallMaterial = CreateMaterial(
            shader, "V2 Structural Wall", wallColor);
        _floorMaterial = CreateMaterial(
            shader, "V2 Structural Floor", floorColor);
        _ceilingMaterial = CreateMaterial(
            shader, "V2 Structural Ceiling", ceilingColor);
        _supportMaterial = CreateMaterial(
            shader, "V2 Structural Support Surface",
            supportSurfaceColor);
        _outlineMaterial = CreateMaterial(
            shader, "V2 Structural Outline", outlineColor);
    }

    private static Material CreateMaterial(Shader shader,
                                           string name,
                                           Color color)
    {
        var material = new Material(shader) { name = name };
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        material.renderQueue = 3000;
        return material;
    }

    private static bool IsFinite(Vector4 value)
    {
        return float.IsFinite(value.x) && float.IsFinite(value.y) &&
               float.IsFinite(value.z) && float.IsFinite(value.w);
    }

    private void DestroyPreview()
    {
        if (_previewRoot == null)
            return;
        MeshFilter[] filters = _previewRoot.GetComponentsInChildren<
            MeshFilter>(true);
        for (int i = 0; i < filters.Length; i++)
            if (filters[i].sharedMesh != null)
                ReleaseObject(filters[i].sharedMesh);
        ReleaseObject(_previewRoot.gameObject);
        _previewRoot = null;
    }

    private static void ReleaseObject(UnityEngine.Object value)
    {
        if (value == null)
            return;
        if (Application.isPlaying)
            Destroy(value);
        else
            DestroyImmediate(value);
    }

    private void OnDestroy()
    {
        _requestPending = false;
        CancelRuntimeRoomComputation();
        DestroyPreview();
        ReleaseObject(_wallMaterial);
        ReleaseObject(_floorMaterial);
        ReleaseObject(_ceilingMaterial);
        ReleaseObject(_supportMaterial);
        ReleaseObject(_outlineMaterial);
    }
}

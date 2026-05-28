using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

[DefaultExecutionOrder(-17)]
[DisallowMultipleComponent]
public sealed class ScanCoverRawDepthProjectedPointCloud : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverDepthPreprocessor preprocessor;
    [SerializeField] private ScanCoverSkeletonSessionController sessionController;
    [SerializeField] private Shader pointShader;

    [Header("Capture")]
    [SerializeField] private bool showPointCloud = true;
    [SerializeField] private bool updateUntilSnapshot = false;
    [SerializeField] private KeyCode captureKey = KeyCode.Space;
    [SerializeField] private bool captureWhenSessionFrozen = true;
    [SerializeField, Min(1)] private int pixelStride = 1;
    [SerializeField, Min(512)] private int maxPoints = 120000;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0.01f;

    [Header("Point Style")]
    [SerializeField, Min(1f)] private float pointSizePixels = 18f;
    [SerializeField, Min(0.002f)] private float pointSizeMeters = 0.006f;
    [SerializeField, Range(0f, 1f)] private float pointAlpha = 1f;
    [SerializeField, Min(0.1f)] private float pointBrightness = 1.8f;
    [SerializeField] private ColorMode colorMode = ColorMode.StructureDiagnostics;
    [SerializeField, Min(0.05f)] private float colorDepthScaleMeters = 4f;
    [SerializeField] private Color pointColor = new Color(0f, 1f, 1f, 1f);

    [Header("Structure Diagnostics")]
    [SerializeField, Min(0.001f)] private float stableDepthDeltaMeters = 0.015f;
    [SerializeField, Min(0.001f)] private float edgeDepthDeltaMeters = 0.06f;
    [SerializeField, Min(0.001f)] private float stableWorldDistanceMeters = 0.025f;
    [SerializeField, Min(0.001f)] private float edgeWorldDistanceMeters = 0.08f;
    [SerializeField] private Color stablePointColor = new Color(0f, 0.95f, 1f, 1f);
    [SerializeField] private Color transitionPointColor = new Color(1f, 0.85f, 0f, 1f);
    [SerializeField] private Color edgePointColor = new Color(1f, 0f, 0.05f, 1f);
    [SerializeField] private Color isolatedPointColor = new Color(0.45f, 0.45f, 0.45f, 1f);

    [Header("Local Patch Grid")]
    [SerializeField] private bool showLocalPatchGrid = false;
    [SerializeField, Min(1)] private int localPatchMaxChunks = 6;
    [SerializeField, Min(16)] private int localPatchMinStablePoints = 90;
    [SerializeField, Min(0.05f)] private float localPatchRadiusMeters = 0.5f;
    [SerializeField, Min(0.005f)] private float localPatchMaxPlaneDistanceMeters = 0.035f;
    [SerializeField, Min(0.01f)] private float localPatchGridSpacingMeters = 0.06f;
    [SerializeField, Min(1)] private int localPatchMinPointsPerCell = 2;
    [SerializeField, Min(0.001f)] private float localPatchLineThicknessMeters = 0.008f;
    [SerializeField, Min(0.001f)] private float localPatchMaxNeighborDistanceMeters = 0.055f;
    [SerializeField] private Color localPatchGridColor = new Color(1f, 1f, 1f, 0.82f);

    [Header("Outer Boundary Ring")]
    [SerializeField] private bool showOuterBoundaryRing = true;
    [SerializeField, Min(0.001f)] private float outerBoundaryLineThicknessMeters = 0.006f;
    [SerializeField, Min(0.001f)] private float outerBoundaryMaxLinkDistanceMeters = 0.08f;
    [SerializeField, Min(1)] private int outerBoundaryLinkSearchPixels = 3;
    [SerializeField] private Color outerBoundaryColor = new Color(1f, 1f, 1f, 0.95f);
    [SerializeField] private bool exportOuterBoundaryCsvToDesktop = true;

    [Header("Debug")]
    [SerializeField] private bool debugLog = true;
    [SerializeField] private bool exportSnapshotCsvToDesktop = true;
    [SerializeField] private string csvFilePrefix = "ScanCover_RawDepthProjectedPoints";

    private static readonly int PointSizePixelsId = Shader.PropertyToID("_PointSizePixels");
    private static readonly int AlphaId = Shader.PropertyToID("_Alpha");
    private static readonly int BrightnessId = Shader.PropertyToID("_Brightness");

    private GameObject _root;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;
    private Material _material;
    private GameObject _patchRoot;
    private MeshFilter _patchMeshFilter;
    private MeshRenderer _patchMeshRenderer;
    private Mesh _patchMesh;
    private Material _patchMaterial;
    private AsyncGPUReadbackRequest _worldPositionRequest;
    private AsyncGPUReadbackRequest _observationMetaRequest;
    private bool _hasPendingReadback;
    private bool _freezeRequested;
    private bool _frozen;
    private ScanCoverSkeletonSessionController.SessionState _lastSessionState;
    private readonly List<Vector3> _vertices = new List<Vector3>(65536);
    private readonly List<Color> _colors = new List<Color>(65536);
    private readonly List<int> _indices = new List<int>(65536);
    private readonly List<Vector3> _renderVertices = new List<Vector3>(262144);
    private readonly List<Color> _renderColors = new List<Color>(262144);
    private readonly List<int> _renderIndices = new List<int>(393216);
    private readonly List<Vector3> _patchVertices = new List<Vector3>(4096);
    private readonly List<Color> _patchColors = new List<Color>(4096);
    private readonly List<int> _patchIndices = new List<int>(8192);
    private readonly List<PointRecord> _records = new List<PointRecord>(65536);

    private enum ColorMode
    {
        StructureDiagnostics,
        LinearDepth,
        Solid
    }

    private enum EdgeClass
    {
        Stable,
        Transition,
        Edge,
        Isolated
    }

    private struct PointRecord
    {
        public int PixelX;
        public int PixelY;
        public Vector3 World;
        public float Depth;
        public float Confidence;
        public float RightDepthDelta;
        public float DownDepthDelta;
        public float RightWorldDistance;
        public float DownWorldDistance;
        public float EdgeScore;
        public EdgeClass EdgeClass;
        public bool IsOuterBoundary;
    }

    private readonly struct PatchSeed
    {
        public PatchSeed(int index, float score)
        {
            Index = index;
            Score = score;
        }

        public readonly int Index;
        public readonly float Score;
    }

    private void Awake()
    {
        ResolveRefs();
        EnsureObjects();
        _lastSessionState = sessionController != null
            ? sessionController.State
            : ScanCoverSkeletonSessionController.SessionState.Idle;
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureObjects();
        _frozen = false;
        _freezeRequested = false;
    }

    private void Update()
    {
        ResolveRefs();
        EnsureObjects();
        UpdateMaterial();
        UpdateVisibility();

        bool snapshotRequested = SessionFreezeTriggered() || CaptureKeyPressed();
        if (snapshotRequested)
        {
            _freezeRequested = true;
            _frozen = false;
            ClearPointMesh();
        }

        if (_hasPendingReadback)
        {
            CompleteReadbackIfReady();
            return;
        }

        if (_frozen)
            return;

        if (_freezeRequested || updateUntilSnapshot)
            RequestCurrentFrame();
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_material != null)
            Destroy(_material);
        if (_patchMaterial != null)
            Destroy(_patchMaterial);
        if (_root != null)
            Destroy(_root);
        if (_patchRoot != null)
            Destroy(_patchRoot);
    }

    [ContextMenu("Capture And Freeze Raw Depth Point Cloud")]
    public void CaptureAndFreeze()
    {
        _freezeRequested = true;
        _frozen = false;
        ClearPointMesh();
        if (!_hasPendingReadback)
            RequestCurrentFrame();
    }

    [ContextMenu("Clear Raw Depth Point Cloud Snapshot")]
    public void ClearSnapshot()
    {
        _frozen = false;
        _freezeRequested = false;
        ClearPointMesh();
    }

    private void ClearPointMesh()
    {
        _vertices.Clear();
        _colors.Clear();
        _indices.Clear();
        _renderVertices.Clear();
        _renderColors.Clear();
        _renderIndices.Clear();
        _patchVertices.Clear();
        _patchColors.Clear();
        _patchIndices.Clear();
        _records.Clear();
        if (_mesh != null)
            _mesh.Clear();
        if (_patchMesh != null)
            _patchMesh.Clear();
    }

    private void RequestCurrentFrame()
    {
        if (_hasPendingReadback || preprocessor == null)
            return;

        if (!preprocessor.RefreshNow())
        {
            LogIssue($"preprocessor refresh failed: {preprocessor.LastIssue}");
            return;
        }

        if (!preprocessor.TryGetOutputs(
                out RenderTexture worldPositionTexture,
                out _,
                out RenderTexture observationMetaTexture))
        {
            LogIssue("preprocessor outputs are not ready.");
            return;
        }

        _worldPositionRequest = AsyncGPUReadback.Request(worldPositionTexture);
        _observationMetaRequest = AsyncGPUReadback.Request(observationMetaTexture);
        _hasPendingReadback = true;
    }

    private void CompleteReadbackIfReady()
    {
        if (!_worldPositionRequest.done || !_observationMetaRequest.done)
            return;

        _hasPendingReadback = false;
        if (_worldPositionRequest.hasError || _observationMetaRequest.hasError)
        {
            LogIssue("GPU readback failed.");
            return;
        }

        BuildPointMesh(
            _worldPositionRequest.GetData<Color>(),
            _observationMetaRequest.GetData<Color>(),
            preprocessor != null ? preprocessor.OutputResolution : Vector2Int.zero);

        if (_freezeRequested)
        {
            _frozen = true;
            _freezeRequested = false;
            if (debugLog)
                Debug.Log("[ScanCoverRawDepthProjectedPointCloud] snapshot frozen.", this);
        }
    }

    private void BuildPointMesh(NativeArray<Color> worldPositions, NativeArray<Color> observationMeta, Vector2Int resolution)
    {
        if (_mesh == null || resolution.x <= 0 || resolution.y <= 0 || worldPositions.Length <= 0)
            return;

        _vertices.Clear();
        _colors.Clear();
        _indices.Clear();
        _renderVertices.Clear();
        _renderColors.Clear();
        _renderIndices.Clear();
        _records.Clear();

        int width = resolution.x;
        int height = resolution.y;
        int stride = Mathf.Max(1, pixelStride);
        int estimated = Mathf.CeilToInt(width / (float)stride) * Mathf.CeilToInt(height / (float)stride);
        while (estimated > maxPoints && stride < Mathf.Max(width, height))
        {
            stride++;
            estimated = Mathf.CeilToInt(width / (float)stride) * Mathf.CeilToInt(height / (float)stride);
        }

        int pixelCount = width * height;
        int[] vertexByPixel = new int[pixelCount];
        for (int i = 0; i < vertexByPixel.Length; i++)
            vertexByPixel[i] = -1;

        for (int y = 0; y < height; y += stride)
        {
            for (int x = 0; x < width; x += stride)
            {
                int index = x + y * width;
                if (index < 0 || index >= worldPositions.Length || index >= observationMeta.Length)
                    continue;

                Color meta = observationMeta[index];
                if (meta.r <= 0.5f || meta.g < minConfidence)
                    continue;

                Color rawPosition = worldPositions[index];
                Vector3 point = new Vector3(rawPosition.r, rawPosition.g, rawPosition.b);
                if (!IsFinite(point))
                    continue;

                int vertexIndex = _vertices.Count;
                _vertices.Add(point);
                _colors.Add(pointColor);
                _indices.Add(vertexIndex);
                vertexByPixel[index] = vertexIndex;
                _records.Add(new PointRecord
                {
                    PixelX = x,
                    PixelY = y,
                    World = point,
                    Depth = meta.b,
                    Confidence = meta.g
                });
            }
        }

        ApplyDiagnostics(width, height, stride, vertexByPixel);
        int outerBoundaryCount = MarkStableTransitionOuterBoundary(width, height, stride);

        BuildQuadGlyphMesh();
        BuildLocalPatchGridMesh();
        BuildOuterBoundaryRingMesh(width, height, stride, vertexByPixel, outerBoundaryCount);

        _mesh.Clear();
        _mesh.indexFormat = _renderVertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _mesh.SetVertices(_renderVertices);
        _mesh.SetColors(_renderColors);
        _mesh.SetIndices(_renderIndices, MeshTopology.Triangles, 0, true);
        _mesh.RecalculateBounds();
        UpdateVisibility();

        if (debugLog)
            LogSnapshotStats(width, height, stride);

        if (_freezeRequested && exportSnapshotCsvToDesktop)
            ExportRecordsToDesktop(width, height, stride);
        if (_freezeRequested && exportOuterBoundaryCsvToDesktop)
            ExportOuterBoundaryRecordsToDesktop(width, height, stride, outerBoundaryCount);
    }

    private void BuildQuadGlyphMesh()
    {
        _renderVertices.Clear();
        _renderColors.Clear();
        _renderIndices.Clear();

        Vector3 right = Vector3.right;
        Vector3 up = Vector3.up;
        Camera sourceCamera = Camera.main != null ? Camera.main : FindAnyObjectByType<Camera>();
        if (sourceCamera != null)
        {
            right = sourceCamera.transform.right;
            up = sourceCamera.transform.up;
        }

        float halfSize = Mathf.Max(0.001f, pointSizeMeters) * 0.5f;
        Vector3 xOffset = right * halfSize;
        Vector3 yOffset = up * halfSize;

        for (int i = 0; i < _vertices.Count; i++)
        {
            Vector3 point = _vertices[i];
            Color color = _colors[i];
            int start = _renderVertices.Count;
            _renderVertices.Add(point - xOffset - yOffset);
            _renderVertices.Add(point - xOffset + yOffset);
            _renderVertices.Add(point + xOffset + yOffset);
            _renderVertices.Add(point + xOffset - yOffset);
            _renderColors.Add(color);
            _renderColors.Add(color);
            _renderColors.Add(color);
            _renderColors.Add(color);
            _renderIndices.Add(start);
            _renderIndices.Add(start + 1);
            _renderIndices.Add(start + 2);
            _renderIndices.Add(start);
            _renderIndices.Add(start + 2);
            _renderIndices.Add(start + 3);
        }
    }

    private void BuildLocalPatchGridMesh()
    {
        _patchVertices.Clear();
        _patchColors.Clear();
        _patchIndices.Clear();
        if (_patchMesh != null)
            _patchMesh.Clear();

        if (!showLocalPatchGrid || _records.Count < localPatchMinStablePoints)
        {
            UpdateVisibility();
            return;
        }

        Camera camera = Camera.main != null ? Camera.main : FindAnyObjectByType<Camera>();
        if (camera == null)
        {
            UpdateVisibility();
            return;
        }

        var recordByPixel = BuildRecordByPixelLookup();
        var seeds = CollectPatchSeeds(camera);
        var visited = new HashSet<int>();
        int chunks = 0;
        int totalCells = 0;
        int totalInliers = 0;
        int maxChunks = Mathf.Max(1, localPatchMaxChunks);
        for (int i = 0; i < seeds.Count && chunks < maxChunks; i++)
        {
            int seedIndex = seeds[i].Index;
            if (visited.Contains(seedIndex))
                continue;

            List<Vector3> candidates = GatherConnectedStablePatch(seedIndex, recordByPixel, visited);
            if (candidates.Count < localPatchMinStablePoints)
                continue;

            if (TryAppendLocalPatchGrid(candidates, camera.transform, out int occupiedCells, out int inlierCount))
            {
                chunks++;
                totalCells += occupiedCells;
                totalInliers += inlierCount;
            }
        }

        if (totalCells <= 0)
        {
            if (debugLog)
                Debug.Log($"[ScanCoverRawDepthProjectedPointCloud] local patch skipped: no accepted chunks. seeds={seeds.Count}.", this);
            UpdateVisibility();
            return;
        }

        if (debugLog)
            Debug.Log($"[ScanCoverRawDepthProjectedPointCloud] local patch grid chunks={chunks} inliers={totalInliers} cells={totalCells} spacing={localPatchGridSpacingMeters:0.###}m.", this);

        UploadPatchMesh();
    }

    private int MarkStableTransitionOuterBoundary(int width, int height, int stride)
    {
        int gridWidth = Mathf.CeilToInt(width / (float)stride);
        int gridHeight = Mathf.CeilToInt(height / (float)stride);
        bool[] occupied = new bool[gridWidth * gridHeight];

        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            int gx = record.PixelX / stride;
            int gy = record.PixelY / stride;
            if (IsOuterBoundaryMaskPoint(record) && gx >= 0 && gx < gridWidth && gy >= 0 && gy < gridHeight)
                occupied[gx + gy * gridWidth] = true;
        }

        bool[] outside = new bool[gridWidth * gridHeight];
        var queue = new Queue<Vector2Int>();
        for (int x = 0; x < gridWidth; x++)
        {
            TryQueueOutsideCell(x, 0, gridWidth, gridHeight, occupied, outside, queue);
            TryQueueOutsideCell(x, gridHeight - 1, gridWidth, gridHeight, occupied, outside, queue);
        }

        for (int y = 0; y < gridHeight; y++)
        {
            TryQueueOutsideCell(0, y, gridWidth, gridHeight, occupied, outside, queue);
            TryQueueOutsideCell(gridWidth - 1, y, gridWidth, gridHeight, occupied, outside, queue);
        }

        while (queue.Count > 0)
        {
            Vector2Int cell = queue.Dequeue();
            TryQueueOutsideCell(cell.x + 1, cell.y, gridWidth, gridHeight, occupied, outside, queue);
            TryQueueOutsideCell(cell.x - 1, cell.y, gridWidth, gridHeight, occupied, outside, queue);
            TryQueueOutsideCell(cell.x, cell.y + 1, gridWidth, gridHeight, occupied, outside, queue);
            TryQueueOutsideCell(cell.x, cell.y - 1, gridWidth, gridHeight, occupied, outside, queue);
        }

        int boundaryCount = 0;
        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            int gx = record.PixelX / stride;
            int gy = record.PixelY / stride;
            bool isBoundary = IsOuterBoundaryMaskPoint(record) && (IsOutsideNeighbor(gx + 1, gy, gridWidth, gridHeight, outside)
                || IsOutsideNeighbor(gx - 1, gy, gridWidth, gridHeight, outside)
                || IsOutsideNeighbor(gx, gy + 1, gridWidth, gridHeight, outside)
                || IsOutsideNeighbor(gx, gy - 1, gridWidth, gridHeight, outside));
            record.IsOuterBoundary = isBoundary;
            _records[i] = record;
            if (isBoundary)
                boundaryCount++;
        }

        return boundaryCount;
    }

    private static bool IsOuterBoundaryMaskPoint(PointRecord record)
    {
        return record.EdgeClass == EdgeClass.Stable || record.EdgeClass == EdgeClass.Transition;
    }

    private static void TryQueueOutsideCell(
        int x,
        int y,
        int width,
        int height,
        bool[] occupied,
        bool[] outside,
        Queue<Vector2Int> queue)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return;

        int index = x + y * width;
        if (occupied[index] || outside[index])
            return;

        outside[index] = true;
        queue.Enqueue(new Vector2Int(x, y));
    }

    private static bool IsOutsideNeighbor(int x, int y, int width, int height, bool[] outside)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return true;
        return outside[x + y * width];
    }

    private void BuildOuterBoundaryRingMesh(int width, int height, int stride, int[] vertexByPixel, int outerBoundaryCount)
    {
        if (!showOuterBoundaryRing || outerBoundaryCount <= 0)
        {
            UploadPatchMesh();
            return;
        }

        Camera camera = Camera.main != null ? Camera.main : FindAnyObjectByType<Camera>();
        Vector3 ribbonNormal = camera != null ? camera.transform.forward : Vector3.forward;
        int links = 0;
        float maxLinkDistance = Mathf.Max(0.001f, outerBoundaryMaxLinkDistanceMeters);
        int searchPixels = Mathf.Max(1, outerBoundaryLinkSearchPixels);
        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            if (!record.IsOuterBoundary)
                continue;

            links += TryAddOuterBoundaryLink(record, stride, 0, width, height, stride, vertexByPixel, maxLinkDistance, searchPixels, ribbonNormal);
            links += TryAddOuterBoundaryLink(record, 0, stride, width, height, stride, vertexByPixel, maxLinkDistance, searchPixels, ribbonNormal);
            links += TryAddOuterBoundaryLink(record, stride, stride, width, height, stride, vertexByPixel, maxLinkDistance, searchPixels, ribbonNormal);
            links += TryAddOuterBoundaryLink(record, -stride, stride, width, height, stride, vertexByPixel, maxLinkDistance, searchPixels, ribbonNormal);
        }

        UploadPatchMesh();

        if (debugLog)
            Debug.Log($"[ScanCoverRawDepthProjectedPointCloud] outer boundary ring points={outerBoundaryCount} links={links}.", this);
    }

    private int TryAddOuterBoundaryLink(
        PointRecord record,
        int deltaX,
        int deltaY,
        int width,
        int height,
        int stride,
        int[] vertexByPixel,
        float maxLinkDistance,
        int searchPixels,
        Vector3 ribbonNormal)
    {
        for (int step = 1; step <= searchPixels; step++)
        {
            int neighborPixelX = record.PixelX + deltaX * step;
            int neighborPixelY = record.PixelY + deltaY * step;
            if (neighborPixelX < 0 || neighborPixelX >= width || neighborPixelY < 0 || neighborPixelY >= height)
                return 0;

            int neighborIndex = vertexByPixel[neighborPixelX + neighborPixelY * width];
            if (neighborIndex < 0 || neighborIndex >= _records.Count)
                continue;

            PointRecord neighbor = _records[neighborIndex];
            if (!neighbor.IsOuterBoundary)
                continue;

            float distance = Vector3.Distance(record.World, neighbor.World);
            float scaledMaxLinkDistance = maxLinkDistance * DepthThresholdScale((record.Depth + neighbor.Depth) * 0.5f) * step;
            if (distance > scaledMaxLinkDistance)
                return 0;

            AddRibbon(record.World, neighbor.World, ribbonNormal, outerBoundaryLineThicknessMeters, outerBoundaryColor);
            return 1;
        }

        return 0;
    }

    private void UploadPatchMesh()
    {
        if (_patchMesh == null)
            return;

        _patchMesh.Clear();
        if (_patchVertices.Count <= 0 || _patchIndices.Count <= 0)
        {
            UpdateVisibility();
            return;
        }

        _patchMesh.indexFormat = _patchVertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _patchMesh.SetVertices(_patchVertices);
        _patchMesh.SetColors(_patchColors);
        _patchMesh.SetIndices(_patchIndices, MeshTopology.Triangles, 0, true);
        _patchMesh.RecalculateBounds();
        UpdateVisibility();
    }

    private Dictionary<int, int> BuildRecordByPixelLookup()
    {
        var recordByPixel = new Dictionary<int, int>(_records.Count);
        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            recordByPixel[PixelKey(record.PixelX, record.PixelY)] = i;
        }

        return recordByPixel;
    }

    private List<PatchSeed> CollectPatchSeeds(Camera camera)
    {
        var seeds = new List<PatchSeed>(_records.Count);
        Ray gaze = new Ray(camera.transform.position, camera.transform.forward);
        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            if (record.EdgeClass != EdgeClass.Stable)
                continue;

            Vector3 fromOrigin = record.World - gaze.origin;
            float along = Vector3.Dot(fromOrigin, gaze.direction);
            if (along <= 0f)
                continue;

            Vector3 closest = gaze.origin + gaze.direction * along;
            float lateral = Vector3.Distance(record.World, closest);
            float score = lateral + along * 0.015f;
            seeds.Add(new PatchSeed(i, score));
        }

        seeds.Sort((a, b) => a.Score.CompareTo(b.Score));
        return seeds;
    }

    private List<Vector3> GatherConnectedStablePatch(
        int seedIndex,
        Dictionary<int, int> recordByPixel,
        HashSet<int> visited)
    {
        var result = new List<Vector3>(512);
        if (seedIndex < 0 || seedIndex >= _records.Count)
            return result;

        PointRecord seed = _records[seedIndex];
        float radiusSqr = localPatchRadiusMeters * localPatchRadiusMeters;
        float maxNeighborDistance = localPatchMaxNeighborDistanceMeters * DepthThresholdScale(seed.Depth);
        int stride = Mathf.Max(1, pixelStride);
        var queue = new Queue<int>();
        visited.Add(seedIndex);
        queue.Enqueue(seedIndex);

        while (queue.Count > 0)
        {
            int index = queue.Dequeue();
            PointRecord current = _records[index];
            result.Add(current.World);

            TryQueuePatchNeighbor(current.PixelX + stride, current.PixelY, current, seed.World, radiusSqr, maxNeighborDistance, recordByPixel, visited, queue);
            TryQueuePatchNeighbor(current.PixelX - stride, current.PixelY, current, seed.World, radiusSqr, maxNeighborDistance, recordByPixel, visited, queue);
            TryQueuePatchNeighbor(current.PixelX, current.PixelY + stride, current, seed.World, radiusSqr, maxNeighborDistance, recordByPixel, visited, queue);
            TryQueuePatchNeighbor(current.PixelX, current.PixelY - stride, current, seed.World, radiusSqr, maxNeighborDistance, recordByPixel, visited, queue);
        }

        return result;
    }

    private bool TryAppendLocalPatchGrid(
        List<Vector3> candidates,
        Transform cameraTransform,
        out int occupiedCells,
        out int inlierCount)
    {
        occupiedCells = 0;
        inlierCount = 0;
        if (!TryFitPlane(candidates, out Vector3 center, out Vector3 normal, out Vector3 axisU))
            return false;

        var inliers = new List<Vector3>(candidates.Count);
        for (int pass = 0; pass < 2; pass++)
        {
            inliers.Clear();
            for (int i = 0; i < candidates.Count; i++)
            {
                float distance = Mathf.Abs(Vector3.Dot(candidates[i] - center, normal));
                if (distance <= localPatchMaxPlaneDistanceMeters)
                    inliers.Add(candidates[i]);
            }

            if (inliers.Count < localPatchMinStablePoints || pass == 1)
                break;

            TryFitPlane(inliers, out center, out normal, out axisU);
        }

        if (inliers.Count < localPatchMinStablePoints)
            return false;

        ChoosePatchAxes(normal, cameraTransform, ref axisU, out Vector3 axisV);

        float spacing = Mathf.Max(0.01f, localPatchGridSpacingMeters);
        var cellCounts = new Dictionary<Vector2Int, int>(inliers.Count);
        for (int i = 0; i < inliers.Count; i++)
        {
            Vector3 delta = inliers[i] - center;
            int u = Mathf.FloorToInt(Vector3.Dot(delta, axisU) / spacing);
            int v = Mathf.FloorToInt(Vector3.Dot(delta, axisV) / spacing);
            Vector2Int key = new Vector2Int(u, v);
            cellCounts.TryGetValue(key, out int count);
            cellCounts[key] = count + 1;
        }

        foreach (KeyValuePair<Vector2Int, int> pair in cellCounts)
        {
            if (pair.Value < localPatchMinPointsPerCell)
                continue;

            occupiedCells++;
            Vector2Int c = pair.Key;
            Vector3 p00 = center + axisU * (c.x * spacing) + axisV * (c.y * spacing);
            Vector3 p10 = p00 + axisU * spacing;
            Vector3 p11 = p10 + axisV * spacing;
            Vector3 p01 = p00 + axisV * spacing;
            AddPatchRibbon(p00, p10, normal);
            AddPatchRibbon(p10, p11, normal);
            AddPatchRibbon(p11, p01, normal);
            AddPatchRibbon(p01, p00, normal);
        }

        inlierCount = inliers.Count;
        return occupiedCells > 0;
    }

    private void TryQueuePatchNeighbor(
        int pixelX,
        int pixelY,
        PointRecord current,
        Vector3 seedWorld,
        float radiusSqr,
        float maxNeighborDistance,
        Dictionary<int, int> recordByPixel,
        HashSet<int> visited,
        Queue<int> queue)
    {
        if (!recordByPixel.TryGetValue(PixelKey(pixelX, pixelY), out int index) || visited.Contains(index))
            return;

        PointRecord neighbor = _records[index];
        if (neighbor.EdgeClass != EdgeClass.Stable)
            return;
        if ((neighbor.World - seedWorld).sqrMagnitude > radiusSqr)
            return;
        if (Vector3.Distance(neighbor.World, current.World) > maxNeighborDistance)
            return;

        visited.Add(index);
        queue.Enqueue(index);
    }

    private static int PixelKey(int x, int y)
    {
        return (y << 16) ^ (x & 0xffff);
    }

    private static void ChoosePatchAxes(Vector3 normal, Transform cameraTransform, ref Vector3 axisU, out Vector3 axisV)
    {
        Vector3 projectedUp = Vector3.ProjectOnPlane(Vector3.up, normal);
        if (projectedUp.sqrMagnitude > 0.25f)
        {
            axisV = projectedUp.normalized;
            axisU = Vector3.Cross(axisV, normal).normalized;
            return;
        }

        axisU = Vector3.ProjectOnPlane(cameraTransform.right, normal);
        if (axisU.sqrMagnitude < 0.25f)
            axisU = Vector3.ProjectOnPlane(Vector3.forward, normal);
        if (axisU.sqrMagnitude < 0.25f)
            axisU = Vector3.Cross(normal, Vector3.up);

        axisU.Normalize();
        axisV = Vector3.Cross(normal, axisU).normalized;
    }

    private bool TryFitPlane(IReadOnlyList<Vector3> points, out Vector3 center, out Vector3 normal, out Vector3 majorAxis)
    {
        center = Vector3.zero;
        normal = Vector3.up;
        majorAxis = Vector3.right;
        if (points.Count < 3)
            return false;

        for (int i = 0; i < points.Count; i++)
            center += points[i];
        center /= points.Count;

        float xx = 0f, xy = 0f, xz = 0f, yy = 0f, yz = 0f, zz = 0f;
        for (int i = 0; i < points.Count; i++)
        {
            Vector3 d = points[i] - center;
            xx += d.x * d.x;
            xy += d.x * d.y;
            xz += d.x * d.z;
            yy += d.y * d.y;
            yz += d.y * d.z;
            zz += d.z * d.z;
        }

        Matrix3x3 covariance = new Matrix3x3(xx, xy, xz, xy, yy, yz, xz, yz, zz);
        majorAxis = PowerIterate(covariance, new Vector3(1f, 0.37f, 0.19f));
        float majorValue = Vector3.Dot(majorAxis, covariance.Multiply(majorAxis));
        Matrix3x3 deflated = covariance.SubtractOuter(majorAxis, majorValue);
        Vector3 secondAxis = PowerIterate(deflated, new Vector3(0.23f, 1f, 0.41f));
        normal = Vector3.Cross(majorAxis, secondAxis).normalized;
        if (normal.sqrMagnitude < 0.5f)
            return false;

        Camera camera = Camera.main;
        if (camera != null && Vector3.Dot(normal, camera.transform.position - center) < 0f)
            normal = -normal;

        return true;
    }

    private static Vector3 PowerIterate(Matrix3x3 matrix, Vector3 seed)
    {
        Vector3 v = seed.normalized;
        for (int i = 0; i < 10; i++)
        {
            v = matrix.Multiply(v);
            if (v.sqrMagnitude < 1e-8f)
                return seed.normalized;
            v.Normalize();
        }
        return v;
    }

    private void AddPatchRibbon(Vector3 a, Vector3 b, Vector3 normal)
    {
        AddRibbon(a, b, normal, localPatchLineThicknessMeters, localPatchGridColor);
    }

    private void AddRibbon(Vector3 a, Vector3 b, Vector3 normal, float thicknessMeters, Color color)
    {
        Vector3 direction = b - a;
        if (direction.sqrMagnitude < 1e-8f)
            return;

        direction.Normalize();
        Vector3 side = Vector3.Cross(normal, direction).normalized * (Mathf.Max(0.001f, thicknessMeters) * 0.5f);
        int start = _patchVertices.Count;
        _patchVertices.Add(a - side);
        _patchVertices.Add(a + side);
        _patchVertices.Add(b + side);
        _patchVertices.Add(b - side);
        for (int i = 0; i < 4; i++)
            _patchColors.Add(color);
        _patchIndices.Add(start);
        _patchIndices.Add(start + 1);
        _patchIndices.Add(start + 2);
        _patchIndices.Add(start);
        _patchIndices.Add(start + 2);
        _patchIndices.Add(start + 3);
    }

    private readonly struct Matrix3x3
    {
        private readonly float m00, m01, m02, m10, m11, m12, m20, m21, m22;

        public Matrix3x3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
        {
            this.m00 = m00;
            this.m01 = m01;
            this.m02 = m02;
            this.m10 = m10;
            this.m11 = m11;
            this.m12 = m12;
            this.m20 = m20;
            this.m21 = m21;
            this.m22 = m22;
        }

        public Vector3 Multiply(Vector3 v)
        {
            return new Vector3(
                m00 * v.x + m01 * v.y + m02 * v.z,
                m10 * v.x + m11 * v.y + m12 * v.z,
                m20 * v.x + m21 * v.y + m22 * v.z);
        }

        public Matrix3x3 SubtractOuter(Vector3 v, float scale)
        {
            return new Matrix3x3(
                m00 - scale * v.x * v.x,
                m01 - scale * v.x * v.y,
                m02 - scale * v.x * v.z,
                m10 - scale * v.y * v.x,
                m11 - scale * v.y * v.y,
                m12 - scale * v.y * v.z,
                m20 - scale * v.z * v.x,
                m21 - scale * v.z * v.y,
                m22 - scale * v.z * v.z);
        }
    }

    private void LogSnapshotStats(int width, int height, int stride)
    {
        int stable = 0;
        int transition = 0;
        int edge = 0;
        int isolated = 0;
        for (int i = 0; i < _records.Count; i++)
        {
            switch (_records[i].EdgeClass)
            {
                case EdgeClass.Stable:
                    stable++;
                    break;
                case EdgeClass.Transition:
                    transition++;
                    break;
                case EdgeClass.Edge:
                    edge++;
                    break;
                default:
                    isolated++;
                    break;
            }
        }

        Debug.Log(
            $"[ScanCoverRawDepthProjectedPointCloud] points={_vertices.Count} stable={stable} transition={transition} edge={edge} isolated={isolated} stride={stride} source={width}x{height} frozen={_frozen}",
            this);
    }

    private void ExportRecordsToDesktop(int width, int height, int stride)
    {
        if (_records.Count <= 0)
            return;

        try
        {
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            if (string.IsNullOrEmpty(desktop))
            {
                LogIssue("desktop path is unavailable; CSV export skipped.");
                return;
            }

            string safePrefix = string.IsNullOrWhiteSpace(csvFilePrefix)
                ? "ScanCover_RawDepthProjectedPoints"
                : string.Join("_", csvFilePrefix.Split(Path.GetInvalidFileNameChars()));
            string fileName = $"{safePrefix}_{DateTime.Now:yyyyMMdd_HHmmss_fff}_{width}x{height}_stride{stride}.csv";
            string path = Path.Combine(desktop, fileName);

            using (var writer = new StreamWriter(path, false, new UTF8Encoding(false)))
            {
                writer.WriteLine("pixel_x,pixel_y,world_x_m,world_y_m,world_z_m,linear_depth_m,confidence,right_depth_delta_m,down_depth_delta_m,right_world_distance_m,down_world_distance_m,edge_score,edge_class,is_outer_boundary");
                for (int i = 0; i < _records.Count; i++)
                {
                    PointRecord record = _records[i];
                    writer.Write(record.PixelX.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.PixelY.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.World.x.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.World.y.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.World.z.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.Depth.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.Confidence.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    WriteOptionalFloat(writer, record.RightDepthDelta);
                    writer.Write(',');
                    WriteOptionalFloat(writer, record.DownDepthDelta);
                    writer.Write(',');
                    WriteOptionalFloat(writer, record.RightWorldDistance);
                    writer.Write(',');
                    WriteOptionalFloat(writer, record.DownWorldDistance);
                    writer.Write(',');
                    writer.Write(record.EdgeScore.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.EdgeClass.ToString());
                    writer.Write(',');
                    writer.WriteLine(record.IsOuterBoundary ? "1" : "0");
                }
            }

            if (debugLog)
                Debug.Log($"[ScanCoverRawDepthProjectedPointCloud] exported {_records.Count} points to {path}", this);
        }
        catch (Exception exception)
        {
            LogIssue($"CSV export failed: {exception.Message}");
        }
    }

    private void ExportOuterBoundaryRecordsToDesktop(int width, int height, int stride, int outerBoundaryCount)
    {
        if (_records.Count <= 0 || outerBoundaryCount <= 0)
            return;

        try
        {
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            if (string.IsNullOrEmpty(desktop))
            {
                LogIssue("desktop path is unavailable; outer boundary CSV export skipped.");
                return;
            }

            string safePrefix = string.IsNullOrWhiteSpace(csvFilePrefix)
                ? "ScanCover_RawDepthProjectedPoints"
                : string.Join("_", csvFilePrefix.Split(Path.GetInvalidFileNameChars()));
            string fileName = $"{safePrefix}_OuterBoundary_{DateTime.Now:yyyyMMdd_HHmmss_fff}_{width}x{height}_stride{stride}.csv";
            string path = Path.Combine(desktop, fileName);

            using (var writer = new StreamWriter(path, false, new UTF8Encoding(false)))
            {
                writer.WriteLine("boundary_index,pixel_x,pixel_y,world_x_m,world_y_m,world_z_m,linear_depth_m,confidence,edge_score,edge_class");
                int boundaryIndex = 0;
                for (int i = 0; i < _records.Count; i++)
                {
                    PointRecord record = _records[i];
                    if (!record.IsOuterBoundary)
                        continue;

                    writer.Write(boundaryIndex.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.PixelX.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.PixelY.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.World.x.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.World.y.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.World.z.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.Depth.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.Confidence.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(record.EdgeScore.ToString("R", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.WriteLine(record.EdgeClass.ToString());
                    boundaryIndex++;
                }
            }

            if (debugLog)
                Debug.Log($"[ScanCoverRawDepthProjectedPointCloud] exported {outerBoundaryCount} outer boundary points to {path}", this);
        }
        catch (Exception exception)
        {
            LogIssue($"Outer boundary CSV export failed: {exception.Message}");
        }
    }

    private void ApplyDiagnostics(int width, int height, int stride, int[] vertexByPixel)
    {
        float[] candidateScores = new float[_records.Count];
        int[] strongNeighborCounts = new int[_records.Count];
        int[] transitionNeighborCounts = new int[_records.Count];
        bool[] hasNeighbors = new bool[_records.Count];

        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            int pixelIndex = record.PixelX + record.PixelY * width;
            int vertexIndex = vertexByPixel[pixelIndex];
            if (vertexIndex < 0)
                continue;

            float maxScore = 0f;
            int strongCount = 0;
            int transitionCount = 0;
            bool hasNeighbor = false;

            int rightVertex = -1;
            if (record.PixelX + stride < width)
                rightVertex = vertexByPixel[record.PixelX + stride + record.PixelY * width];
            if (rightVertex >= 0)
            {
                PointRecord neighbor = _records[rightVertex];
                IncludeNeighborScore(record, neighbor, ref maxScore, ref strongCount, ref transitionCount, ref hasNeighbor);
                MeasureNeighbor(record, neighbor, out record.RightDepthDelta, out record.RightWorldDistance);
            }
            else
            {
                record.RightDepthDelta = float.NaN;
                record.RightWorldDistance = float.NaN;
            }

            int downVertex = -1;
            if (record.PixelY + stride < height)
                downVertex = vertexByPixel[record.PixelX + (record.PixelY + stride) * width];
            if (downVertex >= 0)
            {
                PointRecord neighbor = _records[downVertex];
                IncludeNeighborScore(record, neighbor, ref maxScore, ref strongCount, ref transitionCount, ref hasNeighbor);
                MeasureNeighbor(record, neighbor, out record.DownDepthDelta, out record.DownWorldDistance);
            }
            else
            {
                record.DownDepthDelta = float.NaN;
                record.DownWorldDistance = float.NaN;
            }

            int leftVertex = -1;
            if (record.PixelX - stride >= 0)
                leftVertex = vertexByPixel[record.PixelX - stride + record.PixelY * width];
            if (leftVertex >= 0)
                IncludeNeighborScore(record, _records[leftVertex], ref maxScore, ref strongCount, ref transitionCount, ref hasNeighbor);

            int upVertex = -1;
            if (record.PixelY - stride >= 0)
                upVertex = vertexByPixel[record.PixelX + (record.PixelY - stride) * width];
            if (upVertex >= 0)
                IncludeNeighborScore(record, _records[upVertex], ref maxScore, ref strongCount, ref transitionCount, ref hasNeighbor);

            if (!hasNeighbor)
            {
                record.EdgeScore = 1f;
                record.EdgeClass = EdgeClass.Isolated;
            }
            else
            {
                record.EdgeScore = maxScore;
                record.EdgeClass = EdgeClass.Stable;
            }

            candidateScores[i] = record.EdgeScore;
            strongNeighborCounts[i] = strongCount;
            transitionNeighborCounts[i] = transitionCount;
            hasNeighbors[i] = hasNeighbor;
            _records[i] = record;
        }

        for (int i = 0; i < _records.Count; i++)
        {
            PointRecord record = _records[i];
            int pixelIndex = record.PixelX + record.PixelY * width;
            int vertexIndex = vertexByPixel[pixelIndex];
            if (vertexIndex < 0)
                continue;

            if (!hasNeighbors[i])
            {
                record.EdgeClass = EdgeClass.Isolated;
            }
            else if (candidateScores[i] >= 0.9f)
            {
                bool confirmedByShape = strongNeighborCounts[i] >= 2 || HasAdjacentStrongCandidate(record, width, height, stride, vertexByPixel, candidateScores);
                record.EdgeClass = confirmedByShape ? EdgeClass.Edge : EdgeClass.Transition;
            }
            else if (candidateScores[i] >= 0.35f || transitionNeighborCounts[i] >= 2)
            {
                record.EdgeClass = EdgeClass.Transition;
            }
            else
            {
                record.EdgeClass = EdgeClass.Stable;
            }

            _records[i] = record;
            _colors[vertexIndex] = PointDiagnosticColor(record);
        }
    }

    private void IncludeNeighborScore(
        PointRecord record,
        PointRecord neighbor,
        ref float maxScore,
        ref int strongCount,
        ref int transitionCount,
        ref bool hasNeighbor)
    {
        MeasureNeighbor(record, neighbor, out float depthDelta, out float worldDistance);
        float scale = DepthThresholdScale((record.Depth + neighbor.Depth) * 0.5f);
        float depthScore = Mathf.InverseLerp(stableDepthDeltaMeters * scale, edgeDepthDeltaMeters * scale, depthDelta);
        float worldScore = Mathf.InverseLerp(stableWorldDistanceMeters * scale, edgeWorldDistanceMeters * scale, worldDistance);
        float score = Mathf.Clamp01(Mathf.Max(depthScore, worldScore));
        maxScore = Mathf.Max(maxScore, score);
        if (score >= 0.9f)
            strongCount++;
        else if (score >= 0.35f)
            transitionCount++;
        hasNeighbor = true;
    }

    private static void MeasureNeighbor(PointRecord record, PointRecord neighbor, out float depthDelta, out float worldDistance)
    {
        depthDelta = Mathf.Abs(record.Depth - neighbor.Depth);
        worldDistance = Vector3.Distance(record.World, neighbor.World);
    }

    private static float DepthThresholdScale(float depthMeters)
    {
        return Mathf.Lerp(1f, 1.75f, Mathf.InverseLerp(1f, 2.2f, depthMeters));
    }

    private static bool HasAdjacentStrongCandidate(
        PointRecord record,
        int width,
        int height,
        int stride,
        int[] vertexByPixel,
        float[] candidateScores)
    {
        return IsStrongCandidate(record.PixelX + stride, record.PixelY, width, height, vertexByPixel, candidateScores)
            || IsStrongCandidate(record.PixelX - stride, record.PixelY, width, height, vertexByPixel, candidateScores)
            || IsStrongCandidate(record.PixelX, record.PixelY + stride, width, height, vertexByPixel, candidateScores)
            || IsStrongCandidate(record.PixelX, record.PixelY - stride, width, height, vertexByPixel, candidateScores);
    }

    private static bool IsStrongCandidate(int x, int y, int width, int height, int[] vertexByPixel, float[] candidateScores)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return false;

        int vertex = vertexByPixel[x + y * width];
        return vertex >= 0 && candidateScores[vertex] >= 0.9f;
    }

    private Color PointDiagnosticColor(PointRecord record)
    {
        if (colorMode == ColorMode.LinearDepth)
            return DepthColor(record.Depth);
        if (colorMode == ColorMode.Solid)
            return pointColor;

        Color color = record.EdgeClass switch
        {
            EdgeClass.Stable => stablePointColor,
            EdgeClass.Transition => transitionPointColor,
            EdgeClass.Edge => edgePointColor,
            _ => isolatedPointColor
        };
        color.a = pointAlpha;
        return color;
    }

    private static void WriteOptionalFloat(StreamWriter writer, float value)
    {
        if (float.IsNaN(value) || float.IsInfinity(value))
            return;
        writer.Write(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private bool CaptureKeyPressed()
    {
        if (captureKey == KeyCode.None)
            return false;

#if ENABLE_INPUT_SYSTEM
        if (captureKey == KeyCode.Space)
            return Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame;

        return false;
#elif ENABLE_LEGACY_INPUT_MANAGER
        return Input.GetKeyDown(captureKey);
#else
        return false;
#endif
    }

    private bool SessionFreezeTriggered()
    {
        if (!captureWhenSessionFrozen)
            return false;

        if (sessionController == null)
            return false;

        var state = sessionController.State;
        bool triggered = state == ScanCoverSkeletonSessionController.SessionState.Frozen &&
                         _lastSessionState != ScanCoverSkeletonSessionController.SessionState.Frozen;
        _lastSessionState = state;
        return triggered;
    }

    private Color DepthColor(float depth)
    {
        float value = Mathf.Pow(Mathf.Clamp01(depth / Mathf.Max(0.05f, colorDepthScaleMeters)), 1.25f);
        Color blue = new Color(0.06f, 0.18f, 1f, pointAlpha);
        Color cyan = new Color(0f, 0.95f, 1f, pointAlpha);
        Color green = new Color(0f, 1f, 0.35f, pointAlpha);
        Color yellow = new Color(1f, 0.92f, 0.05f, pointAlpha);
        Color red = new Color(1f, 0.08f, 0.02f, pointAlpha);

        if (value < 0.25f) return Color.Lerp(blue, cyan, value / 0.25f);
        if (value < 0.5f) return Color.Lerp(cyan, green, (value - 0.25f) / 0.25f);
        if (value < 0.72f) return Color.Lerp(green, yellow, (value - 0.5f) / 0.22f);
        return Color.Lerp(yellow, red, (value - 0.72f) / 0.28f);
    }

    private void ResolveRefs()
    {
        if (preprocessor == null)
            preprocessor = FindAnyObjectByType<ScanCoverDepthPreprocessor>();
        if (sessionController == null)
            sessionController = FindAnyObjectByType<ScanCoverSkeletonSessionController>();
        if (pointShader == null)
            pointShader = Shader.Find("Hidden/ScanCover/RawDepthProjectedPointCloud");
    }

    private void EnsureObjects()
    {
        if (_material == null && pointShader != null)
            _material = new Material(pointShader) { name = "ScanCover Raw Depth Projected Point Cloud Material" };
        if (_patchMaterial == null && pointShader != null)
            _patchMaterial = new Material(pointShader) { name = "ScanCover Local Patch Grid Material" };

        if (_root == null)
        {
            _root = new GameObject("[ScanCover] Raw Depth Projected Point Cloud");
            _meshFilter = _root.AddComponent<MeshFilter>();
            _meshRenderer = _root.AddComponent<MeshRenderer>();
            _mesh = new Mesh { name = "ScanCover_RawDepthProjectedPointCloud" };
            _mesh.MarkDynamic();
            _meshFilter.sharedMesh = _mesh;
        }

        if (_patchRoot == null)
        {
            _patchRoot = new GameObject("[ScanCover] Local Patch Grid");
            _patchMeshFilter = _patchRoot.AddComponent<MeshFilter>();
            _patchMeshRenderer = _patchRoot.AddComponent<MeshRenderer>();
            _patchMesh = new Mesh { name = "ScanCover_LocalPatchGrid" };
            _patchMesh.MarkDynamic();
            _patchMeshFilter.sharedMesh = _patchMesh;
        }

        if (_meshRenderer != null && _material != null)
            _meshRenderer.sharedMaterial = _material;
        if (_patchMeshRenderer != null && _patchMaterial != null)
            _patchMeshRenderer.sharedMaterial = _patchMaterial;
    }

    private void UpdateMaterial()
    {
        if (_material == null)
            return;

        _material.SetFloat(PointSizePixelsId, Mathf.Max(1f, pointSizePixels));
        _material.SetFloat(AlphaId, pointAlpha);
        _material.SetFloat(BrightnessId, Mathf.Max(0.1f, pointBrightness));
        if (_patchMaterial != null)
        {
            _patchMaterial.SetFloat(PointSizePixelsId, 1f);
            _patchMaterial.SetFloat(AlphaId, 1f);
            _patchMaterial.SetFloat(BrightnessId, 1.3f);
        }
    }

    private void UpdateVisibility()
    {
        if (_root != null && _root.activeSelf != showPointCloud)
            _root.SetActive(showPointCloud);
        bool showPatch = showPointCloud
            && (showLocalPatchGrid || showOuterBoundaryRing)
            && _patchMesh != null
            && _patchMesh.vertexCount > 0;
        if (_patchRoot != null && _patchRoot.activeSelf != showPatch)
            _patchRoot.SetActive(showPatch);
    }

    private void LogIssue(string issue)
    {
        if (debugLog)
            Debug.LogWarning($"[ScanCoverRawDepthProjectedPointCloud] {issue}", this);
    }

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);
}

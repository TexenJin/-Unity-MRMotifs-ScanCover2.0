using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverRawSnapshotCompressedGrid : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverDepthGridPointCloud rawDepthSource;
    [SerializeField] private Transform displayRoot;
    [SerializeField] private bool useWorldSpaceOutput = true;

    [Header("Raw Snapshot Refresh")]
    [SerializeField] private bool hideSourcePreview = true;
    [SerializeField] private bool forceRawRefreshOnSnapshot = true;
    [SerializeField, Min(0.05f)] private float waitForRawFrameTimeoutSeconds = 0.85f;

    [Header("Compression")]
    [SerializeField, Min(2)] private int sampleStridePixels = 16;
    [SerializeField, Min(1)] private int minValidSamplesPerBlock = 12;
    [SerializeField, Min(0.005f)] private float maxBlockDeviationMeters = 0.08f;
    [SerializeField, Min(0.02f)] private float maxNeighborDistanceMeters = 0.45f;
    [SerializeField, Range(-1f, 1f)] private float minNeighborNormalDot = 0.35f;
    [SerializeField, Min(128)] private int maxLineSegments = 4000;
    [SerializeField, Min(0f)] private float surfaceOffsetMeters = 0.006f;

    [Header("Display")]
    [SerializeField] private Color lineColor = new Color(0.96f, 1f, 0.86f, 0.92f);
    [SerializeField] private Material lineMaterialOverride;
    [SerializeField] private bool debugLog = true;

    private GameObject _lineObject;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;
    private Material _lineMaterial;
    private Coroutine _buildRoutine;

    private struct BlockNode
    {
        public bool valid;
        public int support;
        public Vector3 position;
        public Vector3 normal;
    }

    public int LastLineSegmentCount { get; private set; }
    public int LastSamplePointCount { get; private set; }
    public int LastRawFrameIndex { get; private set; } = -1;

    private void Awake()
    {
        ResolveRefs();
        EnsureLineObjects();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureLineObjects();
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_lineMaterial != null)
            Destroy(_lineMaterial);
        if (_lineObject != null)
            Destroy(_lineObject);
    }

    [ContextMenu("Prepare Raw Snapshot Route")]
    public void PrepareRoute()
    {
        ResolveRefs();
        if (rawDepthSource == null)
            return;

        rawDepthSource.enabled = true;
        SuppressSourcePreview();
    }

    [ContextMenu("Build From Latest Raw Snapshot")]
    public bool BuildFromLatestRawSnapshot()
    {
        ResolveRefs();
        if (rawDepthSource == null)
            return Warn("Raw depth source is missing.");

        if (!rawDepthSource.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot))
            return Warn("No latest raw depth frame snapshot is available.");

        bool built = BuildMesh(snapshot);
        if (built && debugLog)
        {
            Debug.Log(
                $"[ScanCoverRawSnapshotCompressedGrid] Built compressed raw grid: frame={snapshot.frameIndex} resolution={snapshot.resolutionWidth}x{snapshot.resolutionHeight} stride={sampleStridePixels} samples={LastSamplePointCount} lines={LastLineSegmentCount}",
                this);
        }

        return built;
    }

    [ContextMenu("Capture Raw Snapshot And Build")]
    public void CaptureRawSnapshotAndBuild()
    {
        ResolveRefs();
        if (_buildRoutine != null)
            StopCoroutine(_buildRoutine);
        _buildRoutine = StartCoroutine(CaptureRawSnapshotAndBuildRoutine());
    }

    public void ClearDisplay()
    {
        LastLineSegmentCount = 0;
        LastSamplePointCount = 0;
        LastRawFrameIndex = -1;
        if (_mesh != null)
            _mesh.Clear();
    }

    private IEnumerator CaptureRawSnapshotAndBuildRoutine()
    {
        ResolveRefs();
        EnsureLineObjects();

        if (rawDepthSource == null)
        {
            Warn("Raw depth source is missing.");
            _buildRoutine = null;
            yield break;
        }

        rawDepthSource.enabled = true;
        SuppressSourcePreview();

        int previousFrame = rawDepthSource.FrameIndex;
        if (forceRawRefreshOnSnapshot)
            rawDepthSource.RefreshNow(forcePreprocessorRefresh: true);

        float start = Time.unscaledTime;
        while (Time.unscaledTime - start < waitForRawFrameTimeoutSeconds)
        {
            if (!rawDepthSource.HasPendingReadback &&
                rawDepthSource.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot ready) &&
                ready.worldPositions != null &&
                ready.worldPositions.Length > 0 &&
                ready.frameIndex >= previousFrame)
            {
                BuildMesh(ready);
                if (debugLog)
                {
                    Debug.Log(
                        $"[ScanCoverRawSnapshotCompressedGrid] Capture+build complete: frame={ready.frameIndex} resolution={ready.resolutionWidth}x{ready.resolutionHeight} stride={sampleStridePixels} samples={LastSamplePointCount} lines={LastLineSegmentCount}",
                        this);
                }
                _buildRoutine = null;
                yield break;
            }

            yield return null;
        }

        BuildFromLatestRawSnapshot();
        _buildRoutine = null;
    }

    private bool BuildMesh(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot)
    {
        EnsureLineObjects();
        LastLineSegmentCount = 0;
        LastSamplePointCount = 0;
        LastRawFrameIndex = snapshot != null ? snapshot.frameIndex : -1;

        if (snapshot == null ||
            snapshot.worldPositions == null ||
            snapshot.resolutionWidth <= 1 ||
            snapshot.resolutionHeight <= 1)
            return Warn("Raw depth snapshot is empty.");

        int width = snapshot.resolutionWidth;
        int height = snapshot.resolutionHeight;
        Vector3[] positions = snapshot.worldPositions;
        Vector3[] normals = snapshot.worldNormals;
        int expected = width * height;
        if (positions.Length < expected)
            return Warn($"Raw depth snapshot has insufficient positions: {positions.Length}/{expected}.");

        int stride = Mathf.Max(2, sampleStridePixels);
        int maxSegments = Mathf.Max(128, maxLineSegments);
        List<Vector3> vertices = new List<Vector3>(Mathf.Min(maxSegments * 2, 65534));
        List<int> indices = new List<int>(Mathf.Min(maxSegments * 2, 65534));
        int cols = Mathf.CeilToInt(width / (float)stride);
        int rows = Mathf.CeilToInt(height / (float)stride);
        BlockNode[] nodes = new BlockNode[cols * rows];

        for (int row = 0; row < rows; row++)
        {
            int y0 = row * stride;
            int y1 = Mathf.Min(height, y0 + stride);
            for (int col = 0; col < cols; col++)
            {
                int x0 = col * stride;
                int x1 = Mathf.Min(width, x0 + stride);
                BlockNode node = BuildBlockNode(x0, x1, y0, y1, width, positions, normals);
                nodes[row * cols + col] = node;
                if (node.valid)
                    LastSamplePointCount++;
            }
        }

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                BlockNode node = nodes[row * cols + col];
                if (!node.valid)
                    continue;

                if (col + 1 < cols)
                    TryAddBlockSegment(node, nodes[row * cols + col + 1], vertices, indices, maxSegments);
                if (row + 1 < rows)
                    TryAddBlockSegment(node, nodes[(row + 1) * cols + col], vertices, indices, maxSegments);

                if (LastLineSegmentCount >= maxSegments)
                    break;
            }

            if (LastLineSegmentCount >= maxSegments)
                break;
        }

        _mesh.Clear();
        if (vertices.Count <= 0)
        {
            _meshRenderer.enabled = false;
            return Warn("No compressed grid lines survived filtering.");
        }

        _mesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _mesh.SetVertices(vertices);
        _mesh.SetIndices(indices, MeshTopology.Lines, 0, calculateBounds: true);
        _mesh.UploadMeshData(false);
        _meshRenderer.enabled = true;
        return true;
    }

    private BlockNode BuildBlockNode(
        int x0,
        int x1,
        int y0,
        int y1,
        int width,
        Vector3[] positions,
        Vector3[] normals)
    {
        Vector3 sum = Vector3.zero;
        Vector3 normalSum = Vector3.zero;
        int count = 0;

        for (int y = y0; y < y1; y++)
        {
            int rowOffset = y * width;
            for (int x = x0; x < x1; x++)
            {
                int index = rowOffset + x;
                Vector3 position = positions[index];
                if (!IsUsablePosition(position))
                    continue;

                sum += position;
                if (normals != null && index < normals.Length && IsUsableNormal(normals[index]))
                    normalSum += normals[index].normalized;
                count++;
            }
        }

        if (count < Mathf.Max(1, minValidSamplesPerBlock))
            return default;

        Vector3 center = sum / count;
        int stableCount = 0;
        float maxDeviation = Mathf.Max(0.005f, maxBlockDeviationMeters);
        for (int y = y0; y < y1; y++)
        {
            int rowOffset = y * width;
            for (int x = x0; x < x1; x++)
            {
                Vector3 position = positions[rowOffset + x];
                if (IsUsablePosition(position) && Vector3.Distance(position, center) <= maxDeviation)
                    stableCount++;
            }
        }

        if (stableCount < Mathf.Max(1, minValidSamplesPerBlock))
            return default;

        Vector3 normal = normalSum.sqrMagnitude > 0.0001f ? normalSum.normalized : Vector3.zero;
        return new BlockNode
        {
            valid = true,
            support = stableCount,
            position = center,
            normal = normal
        };
    }

    private void TryAddBlockSegment(
        BlockNode a,
        BlockNode b,
        List<Vector3> vertices,
        List<int> indices,
        int maxSegments)
    {
        if (LastLineSegmentCount >= maxSegments || !a.valid || !b.valid)
            return;

        float distance = Vector3.Distance(a.position, b.position);
        if (distance <= 0.0001f || distance > maxNeighborDistanceMeters)
            return;

        if (IsUsableNormal(a.normal) &&
            IsUsableNormal(b.normal) &&
            Vector3.Dot(a.normal, b.normal) < minNeighborNormalDot)
            return;

        int baseVertex = vertices.Count;
        vertices.Add(a.position + a.normal * surfaceOffsetMeters);
        vertices.Add(b.position + b.normal * surfaceOffsetMeters);
        indices.Add(baseVertex);
        indices.Add(baseVertex + 1);
        LastLineSegmentCount++;
    }

    private static bool IsUsablePosition(Vector3 position)
    {
        return IsFinite(position) && position.sqrMagnitude > 0.000001f;
    }

    private static bool IsUsableNormal(Vector3 normal)
    {
        return IsFinite(normal) && normal.sqrMagnitude > 0.0001f;
    }

    private static bool IsFinite(Vector3 value)
    {
        return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
               !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
               !float.IsNaN(value.z) && !float.IsInfinity(value.z);
    }

    private void ResolveRefs()
    {
        if (rawDepthSource == null)
            rawDepthSource = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (rawDepthSource == null)
            rawDepthSource = FindAnyObjectByType<ScanCoverDepthGridPointCloud>(FindObjectsInactive.Include);
    }

    private void SuppressSourcePreview()
    {
        if (rawDepthSource == null)
            return;

        rawDepthSource.ApplyRuleHardeningProfileV01();
        rawDepthSource.SetUpdateEveryFrame(false);
        if (hideSourcePreview)
        {
            rawDepthSource.SetPreviewDisplayVisible(false);
            rawDepthSource.SetPreviewVisible(false);
        }
    }

    private void EnsureLineObjects()
    {
        if (_lineObject == null)
        {
            _lineObject = new GameObject("ScanCover Raw Snapshot Compressed Grid");
            Transform parent = useWorldSpaceOutput ? null : (displayRoot != null ? displayRoot : transform);
            _lineObject.transform.SetParent(parent, worldPositionStays: false);
            _lineObject.transform.position = Vector3.zero;
            _lineObject.transform.rotation = Quaternion.identity;
            _lineObject.transform.localScale = Vector3.one;
            _meshFilter = _lineObject.AddComponent<MeshFilter>();
            _meshRenderer = _lineObject.AddComponent<MeshRenderer>();
        }

        if (useWorldSpaceOutput)
        {
            _lineObject.transform.SetParent(null, worldPositionStays: false);
            _lineObject.transform.position = Vector3.zero;
            _lineObject.transform.rotation = Quaternion.identity;
            _lineObject.transform.localScale = Vector3.one;
        }

        if (_mesh == null)
        {
            _mesh = new Mesh
            {
                name = "ScanCover Raw Snapshot Compressed Grid Mesh",
                indexFormat = IndexFormat.UInt32
            };
        }

        if (_meshFilter == null)
            _meshFilter = _lineObject.GetComponent<MeshFilter>();
        if (_meshRenderer == null)
            _meshRenderer = _lineObject.GetComponent<MeshRenderer>();

        _meshFilter.sharedMesh = _mesh;
        _meshRenderer.sharedMaterial = ResolveLineMaterial();
        _meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _meshRenderer.receiveShadows = false;
        _meshRenderer.enabled = _mesh != null && LastLineSegmentCount > 0;
    }

    private Material ResolveLineMaterial()
    {
        if (lineMaterialOverride != null)
            return lineMaterialOverride;
        if (_lineMaterial != null)
            return _lineMaterial;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Sprites/Default");

        _lineMaterial = new Material(shader)
        {
            name = "ScanCover Raw Snapshot Compressed Grid Material"
        };
        if (_lineMaterial.HasProperty("_BaseColor"))
            _lineMaterial.SetColor("_BaseColor", lineColor);
        if (_lineMaterial.HasProperty("_Color"))
            _lineMaterial.SetColor("_Color", lineColor);
        _lineMaterial.renderQueue = (int)RenderQueue.Transparent;
        return _lineMaterial;
    }

    private bool Warn(string message)
    {
        if (debugLog)
            Debug.LogWarning($"[ScanCoverRawSnapshotCompressedGrid] {message}", this);
        return false;
    }
}

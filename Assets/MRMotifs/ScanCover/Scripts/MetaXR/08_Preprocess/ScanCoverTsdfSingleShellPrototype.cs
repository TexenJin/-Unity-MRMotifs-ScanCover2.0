using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverTsdfSingleShellPrototype : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private ScanCoverDepthGridPointCloud rawDepthSource;
    [SerializeField] private Transform volumeAnchor;
    [SerializeField] private Transform displayRoot;
    [SerializeField] private bool forceWorldSpaceDisplay = true;

    [Header("Raw Snapshot Refresh")]
    [SerializeField] private bool hideSourcePreview = true;
    [SerializeField] private bool forceRawRefreshOnSnapshot = true;
    [SerializeField, Min(0.05f)] private float waitForRawFrameTimeoutSeconds = 0.85f;
    [SerializeField, Min(0.05f)] private float fusionBurstDurationSeconds = 2.0f;
    [SerializeField, Min(1)] private int maxFusionFramesPerTrigger = 24;
    [SerializeField] private bool rebuildMeshAfterFusionBurstOnly = false;

    [Header("Distilled TSDF Shell Parameters")]
    [SerializeField, Min(0.02f)] private float voxelSizeMeters = 0.065f;
    [SerializeField, Min(0.02f)] private float truncationMeters = 0.195f;
    [SerializeField] private Vector3 volumeSizeMeters = new Vector3(6.5f, 3.2f, 6.5f);
    [SerializeField, Range(1, 64)] private int maxFusionWeight = 12;
    [SerializeField, Range(1, 8)] private int minSurfaceCornerWeight = 1;

    [Header("Depth Filtering")]
    [SerializeField, Min(1)] private int sampleStridePixels = 2;
    [SerializeField, Min(0f)] private float minDepthMeters = 0.35f;
    [SerializeField, Min(0f)] private float maxDepthMeters = 5.5f;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0f;
    [SerializeField] private bool requireValidNormal = false;
    [SerializeField, Range(-1f, 1f)] private float minNormalFacingCameraDot = -0.25f;

    [Header("Surface Extraction")]
    [SerializeField] private bool rebuildMeshAfterEachCapture = true;
    [SerializeField, Min(0)] private int maxSurfaceVertices = 90000;
    [SerializeField] private bool doubleSidedTriangles = true;
    [SerializeField] private bool showFallbackSurfaceSamples = true;
    [SerializeField, Min(256)] private int maxFallbackSurfaceSamples = 48000;
    [SerializeField, Min(0.002f)] private float fallbackSurfaceSampleSizeMeters = 0.018f;
    [SerializeField] private Color shellColor = new Color(0.12f, 0.9f, 1f, 1f);
    [SerializeField] private Material shellMaterialOverride;
    [SerializeField] private bool debugLog = true;

    private GameObject _meshObject;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;
    private Material _runtimeMaterial;
    private Coroutine _captureRoutine;

    private float[] _tsdf;
    private byte[] _weights;
    private int[] _cellVertexIndices;
    private readonly List<Vector3> _fallbackSurfacePoints = new List<Vector3>(12000);
    private readonly List<Vector3> _fallbackSurfaceNormals = new List<Vector3>(12000);
    private readonly HashSet<int> _fallbackSurfaceVoxels = new HashSet<int>();
    private int _dimX;
    private int _dimY;
    private int _dimZ;
    private Vector3 _volumeOriginWorld;
    private bool _volumeInitialized;

    private static readonly int[] CornerOffsetX = { 0, 1, 1, 0, 0, 1, 1, 0 };
    private static readonly int[] CornerOffsetY = { 0, 0, 1, 1, 0, 0, 1, 1 };
    private static readonly int[] CornerOffsetZ = { 0, 0, 0, 0, 1, 1, 1, 1 };

    private static readonly int[,] CubeEdges =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };

    public int IntegratedFrameCount { get; private set; }
    public int LastRawFrameIndex { get; private set; } = -1;
    public int LastInputSampleCount { get; private set; }
    public int LastIntegratedSampleCount { get; private set; }
    public int LastUpdatedVoxelCount { get; private set; }
    public int LastMeshVertexCount { get; private set; }
    public int LastMeshTriangleCount { get; private set; }

    private void Awake()
    {
        ResolveRefs();
        EnsureObjects();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureObjects();
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_runtimeMaterial != null)
            Destroy(_runtimeMaterial);
        if (_meshObject != null)
            Destroy(_meshObject);
    }

    [ContextMenu("Prepare TSDF Single Shell Route")]
    public void PrepareRoute()
    {
        ResolveRefs();
        EnsureObjects();
        if (rawDepthSource == null)
            return;

        rawDepthSource.enabled = true;
        SuppressSourcePreview();
    }

    [ContextMenu("Capture Raw Snapshot And Integrate")]
    public void CaptureRawSnapshotAndIntegrate()
    {
        ResolveRefs();
        if (_captureRoutine != null)
            StopCoroutine(_captureRoutine);
        _captureRoutine = StartCoroutine(CaptureRawSnapshotAndIntegrateRoutine());
    }

    [ContextMenu("Integrate Latest Raw Snapshot")]
    public bool IntegrateLatestRawSnapshot()
    {
        ResolveRefs();
        if (rawDepthSource == null)
            return Warn("Raw depth source is missing.");

        if (!rawDepthSource.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot))
            return Warn("No latest raw depth frame snapshot is available.");

        bool integrated = IntegrateSnapshot(snapshot);
        if (integrated && rebuildMeshAfterEachCapture)
            RebuildMesh();
        return integrated;
    }

    [ContextMenu("Clear TSDF Shell")]
    public void ClearShell()
    {
        if (_captureRoutine != null)
        {
            StopCoroutine(_captureRoutine);
            _captureRoutine = null;
        }

        _volumeInitialized = false;
        _tsdf = null;
        _weights = null;
        _cellVertexIndices = null;
        _fallbackSurfacePoints.Clear();
        _fallbackSurfaceNormals.Clear();
        _fallbackSurfaceVoxels.Clear();
        IntegratedFrameCount = 0;
        LastRawFrameIndex = -1;
        LastInputSampleCount = 0;
        LastIntegratedSampleCount = 0;
        LastUpdatedVoxelCount = 0;
        LastMeshVertexCount = 0;
        LastMeshTriangleCount = 0;

        if (_mesh != null)
            _mesh.Clear();
    }

    [ContextMenu("Rebuild Mesh From TSDF")]
    public bool RebuildMesh()
    {
        EnsureObjects();
        LastMeshVertexCount = 0;
        LastMeshTriangleCount = 0;

        if (!_volumeInitialized || _tsdf == null || _weights == null)
            return Warn("TSDF volume is empty.");

        int cellX = Mathf.Max(0, _dimX - 1);
        int cellY = Mathf.Max(0, _dimY - 1);
        int cellZ = Mathf.Max(0, _dimZ - 1);
        if (cellX <= 0 || cellY <= 0 || cellZ <= 0)
            return Warn("TSDF volume dimensions are invalid.");

        int cellCount = cellX * cellY * cellZ;
        if (_cellVertexIndices == null || _cellVertexIndices.Length != cellCount)
            _cellVertexIndices = new int[cellCount];
        for (int i = 0; i < _cellVertexIndices.Length; i++)
            _cellVertexIndices[i] = -1;

        List<Vector3> vertices = new List<Vector3>(Mathf.Min(maxSurfaceVertices, cellCount));
        List<int> triangles = new List<int>(Mathf.Min(maxSurfaceVertices * 3, cellCount * 6));

        Vector3[] cornerPositions = new Vector3[8];
        float[] cornerValues = new float[8];
        byte[] cornerWeights = new byte[8];

        for (int z = 0; z < cellZ; z++)
        {
            for (int y = 0; y < cellY; y++)
            {
                for (int x = 0; x < cellX; x++)
                {
                    if (vertices.Count >= maxSurfaceVertices)
                        break;

                    if (!TryBuildCellVertex(x, y, z, cornerPositions, cornerValues, cornerWeights, out Vector3 vertex))
                        continue;

                    int vertexIndex = vertices.Count;
                    vertices.Add(vertex);
                    _cellVertexIndices[CellIndex(x, y, z, cellX, cellY)] = vertexIndex;
                }
            }
        }

        for (int z = 0; z < _dimZ; z++)
        {
            for (int y = 0; y < _dimY; y++)
            {
                for (int x = 0; x < _dimX; x++)
                {
                    if (x + 1 < _dimX && IsSignChange(Index(x, y, z), Index(x + 1, y, z)))
                        AddQuadAroundXEdge(x, y, z, cellX, cellY, triangles);
                    if (y + 1 < _dimY && IsSignChange(Index(x, y, z), Index(x, y + 1, z)))
                        AddQuadAroundYEdge(x, y, z, cellX, cellY, triangles);
                    if (z + 1 < _dimZ && IsSignChange(Index(x, y, z), Index(x, y, z + 1)))
                        AddQuadAroundZEdge(x, y, z, cellX, cellY, triangles);
                }
            }
        }

        if (vertices.Count == 0 || triangles.Count == 0)
            BuildFallbackSurfaceSampleMesh(vertices, triangles);

        _mesh.Clear();
        _mesh.indexFormat = vertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        _mesh.SetVertices(vertices);
        _mesh.SetTriangles(triangles, 0, true);
        _mesh.RecalculateNormals();
        _mesh.RecalculateBounds();

        LastMeshVertexCount = vertices.Count;
        LastMeshTriangleCount = triangles.Count / 3;

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverTsdfSingleShellPrototype] Mesh rebuilt: vertices={LastMeshVertexCount} triangles={LastMeshTriangleCount} fallbackSamples={_fallbackSurfacePoints.Count} frames={IntegratedFrameCount} voxel={voxelSizeMeters:F3} trunc={truncationMeters:F3}",
                this);
        }

        return vertices.Count > 0;
    }

    private IEnumerator CaptureRawSnapshotAndIntegrateRoutine()
    {
        ResolveRefs();
        EnsureObjects();

        if (rawDepthSource == null)
        {
            Warn("Raw depth source is missing.");
            _captureRoutine = null;
            yield break;
        }

        rawDepthSource.enabled = true;
        SuppressSourcePreview();

        int lastAcceptedFrame = int.MinValue;
        int integratedFrames = 0;
        float burstStart = Time.unscaledTime;
        float burstDuration = Mathf.Max(0.05f, fusionBurstDurationSeconds);
        int maxFrames = Mathf.Max(1, maxFusionFramesPerTrigger);

        while (Time.unscaledTime - burstStart < burstDuration && integratedFrames < maxFrames)
        {
            if (forceRawRefreshOnSnapshot && !rawDepthSource.HasPendingReadback)
                rawDepthSource.RefreshNow(forcePreprocessorRefresh: true);

            float frameStart = Time.unscaledTime;
            bool integratedThisFrame = false;
            while (Time.unscaledTime - frameStart < waitForRawFrameTimeoutSeconds)
            {
                if (!rawDepthSource.HasPendingReadback &&
                    rawDepthSource.TryGetLatestRawDepthFrameSnapshot(out ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot ready) &&
                    ready.worldPositions != null &&
                    ready.worldPositions.Length > 0 &&
                    ready.frameIndex != lastAcceptedFrame)
                {
                    integratedThisFrame = IntegrateSnapshot(ready);
                    if (integratedThisFrame)
                    {
                        lastAcceptedFrame = ready.frameIndex;
                        integratedFrames++;
                    }
                    if (integratedThisFrame && rebuildMeshAfterEachCapture && !rebuildMeshAfterFusionBurstOnly)
                        RebuildMesh();
                    break;
                }

                yield return null;
            }

            if (!integratedThisFrame)
                yield return null;
        }

        if (integratedFrames <= 0)
        {
            bool integratedLatest = IntegrateLatestRawSnapshot();
            if (integratedLatest)
                integratedFrames = 1;
        }

        if (rebuildMeshAfterEachCapture && (rebuildMeshAfterFusionBurstOnly || integratedFrames <= 0))
            RebuildMesh();

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverTsdfSingleShellPrototype] Fusion trigger complete: integratedFrames={integratedFrames} totalFrames={IntegratedFrameCount} fallbackSamples={_fallbackSurfacePoints.Count} vertices={LastMeshVertexCount} triangles={LastMeshTriangleCount}",
                this);
        }

        _captureRoutine = null;
    }

    private bool IntegrateSnapshot(ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot)
    {
        if (snapshot == null ||
            snapshot.worldPositions == null ||
            snapshot.worldPositions.Length <= 0 ||
            snapshot.resolutionWidth <= 1 ||
            snapshot.resolutionHeight <= 1)
            return Warn("Raw depth snapshot is empty.");

        EnsureVolumeInitialized();

        Vector3 cameraPosition = GetCameraPosition();
        Vector3[] positions = snapshot.worldPositions;
        Vector3[] normals = snapshot.worldNormals;
        Color[] meta = snapshot.observationMeta;
        int width = snapshot.resolutionWidth;
        int height = snapshot.resolutionHeight;
        int expected = Mathf.Min(width * height, positions.Length);
        int stride = Mathf.Max(1, sampleStridePixels);
        int halfBandSteps = Mathf.Max(1, Mathf.CeilToInt(truncationMeters / voxelSizeMeters));

        LastRawFrameIndex = snapshot.frameIndex;
        LastInputSampleCount = 0;
        LastIntegratedSampleCount = 0;
        LastUpdatedVoxelCount = 0;

        for (int y = 0; y < height; y += stride)
        {
            for (int x = 0; x < width; x += stride)
            {
                int index = y * width + x;
                if (index < 0 || index >= expected)
                    continue;

                LastInputSampleCount++;

                if (!TryGetUsableSample(index, positions, normals, meta, cameraPosition, out Vector3 point, out Vector3 normal, out float sampleWeight))
                    continue;

                LastIntegratedSampleCount++;
                AddFallbackSurfaceSample(point, normal);
                for (int step = -halfBandSteps; step <= halfBandSteps; step++)
                {
                    float signedDistance = step * voxelSizeMeters;
                    Vector3 voxelWorld = point + normal * signedDistance;
                    if (!TryWorldToVoxel(voxelWorld, out int vx, out int vy, out int vz))
                        continue;

                    float sampleTsdf = Mathf.Clamp(signedDistance / truncationMeters, -1f, 1f);
                    IntegrateVoxel(vx, vy, vz, sampleTsdf, sampleWeight);
                }
            }
        }

        IntegratedFrameCount++;
        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverTsdfSingleShellPrototype] Integrated raw frame={snapshot.frameIndex} resolution={width}x{height} stride={stride} input={LastInputSampleCount} accepted={LastIntegratedSampleCount} voxelUpdates={LastUpdatedVoxelCount} volume={_dimX}x{_dimY}x{_dimZ}",
                this);
        }

        return LastIntegratedSampleCount > 0;
    }

    private bool TryGetUsableSample(
        int index,
        Vector3[] positions,
        Vector3[] normals,
        Color[] meta,
        Vector3 cameraPosition,
        out Vector3 point,
        out Vector3 normal,
        out float sampleWeight)
    {
        point = Vector3.zero;
        normal = Vector3.up;
        sampleWeight = 1f;

        if (index < 0 || index >= positions.Length)
            return false;

        point = positions[index];
        if (!Finite(point))
            return false;

        Vector3 toPoint = point - cameraPosition;
        float depth = toPoint.magnitude;
        if (depth < minDepthMeters || depth > maxDepthMeters)
            return false;

        bool hasMeta = meta != null && index < meta.Length;
        Vector3 viewDirection = toPoint.sqrMagnitude > 0.0001f ? toPoint.normalized : Vector3.forward;
        bool normalValid = false;
        if (normals != null && index < normals.Length)
        {
            if (!hasMeta || meta[index].a >= 0.5f)
            {
                normal = normals[index];
                normalValid = Finite(normal) && normal.sqrMagnitude > 0.01f;
            }
        }

        if (!normalValid && !requireValidNormal)
        {
            normal = -viewDirection;
            normalValid = true;
        }

        if (!normalValid)
            return false;

        normal.Normalize();
        if (Vector3.Dot(normal, viewDirection) > 0f)
            normal = -normal;
        if (Vector3.Dot(normal, -viewDirection) < minNormalFacingCameraDot)
            return false;

        float confidence = hasMeta ? meta[index].g : 1f;
        if (confidence < minConfidence)
            return false;

        sampleWeight = Mathf.Clamp01(confidence);
        return true;
    }

    private void IntegrateVoxel(int x, int y, int z, float sampleTsdf, float sampleWeight)
    {
        int index = Index(x, y, z);
        int oldWeight = _weights[index];
        int cappedWeight = Mathf.Min(maxFusionWeight, oldWeight + Mathf.Max(1, Mathf.RoundToInt(sampleWeight)));
        float weightedOld = _tsdf[index] * oldWeight;
        float weightedNew = sampleTsdf * sampleWeight;
        float denominator = Mathf.Max(0.0001f, oldWeight + sampleWeight);
        _tsdf[index] = Mathf.Clamp(weightedOld + weightedNew, -maxFusionWeight, maxFusionWeight) / Mathf.Min(denominator, maxFusionWeight);
        _weights[index] = (byte)cappedWeight;
        LastUpdatedVoxelCount++;
    }

    private void AddFallbackSurfaceSample(Vector3 point, Vector3 normal)
    {
        if (!showFallbackSurfaceSamples || _fallbackSurfacePoints.Count >= maxFallbackSurfaceSamples)
            return;
        if (!TryWorldToVoxel(point, out int x, out int y, out int z))
            return;

        int voxelIndex = Index(x, y, z);
        if (!_fallbackSurfaceVoxels.Add(voxelIndex))
            return;

        _fallbackSurfacePoints.Add(point);
        _fallbackSurfaceNormals.Add(normal.sqrMagnitude > 0.0001f ? normal.normalized : Vector3.up);
    }

    private void BuildFallbackSurfaceSampleMesh(List<Vector3> vertices, List<int> triangles)
    {
        if (!showFallbackSurfaceSamples || _fallbackSurfacePoints.Count <= 0)
            return;

        float halfSize = Mathf.Max(0.001f, fallbackSurfaceSampleSizeMeters) * 0.5f;
        int count = Mathf.Min(_fallbackSurfacePoints.Count, Mathf.Max(1, maxFallbackSurfaceSamples));
        for (int i = 0; i < count; i++)
        {
            Vector3 normal = i < _fallbackSurfaceNormals.Count ? _fallbackSurfaceNormals[i] : Vector3.up;
            if (!Finite(normal) || normal.sqrMagnitude < 0.0001f)
                normal = Vector3.up;
            normal.Normalize();

            Vector3 tangent = Vector3.Cross(normal, Vector3.up);
            if (tangent.sqrMagnitude < 0.0001f)
                tangent = Vector3.Cross(normal, Vector3.right);
            tangent.Normalize();
            Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;
            Vector3 center = _fallbackSurfacePoints[i] + normal * 0.004f;

            int baseIndex = vertices.Count;
            vertices.Add(center - tangent * halfSize - bitangent * halfSize);
            vertices.Add(center + tangent * halfSize - bitangent * halfSize);
            vertices.Add(center + tangent * halfSize + bitangent * halfSize);
            vertices.Add(center - tangent * halfSize + bitangent * halfSize);

            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 1);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex + 3);

            if (!doubleSidedTriangles)
                continue;

            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex + 1);
            triangles.Add(baseIndex);
            triangles.Add(baseIndex + 3);
            triangles.Add(baseIndex + 2);
            triangles.Add(baseIndex);
        }
    }

    private void EnsureVolumeInitialized()
    {
        if (_volumeInitialized && _tsdf != null && _weights != null)
            return;

        float voxel = Mathf.Max(0.02f, voxelSizeMeters);
        _dimX = Mathf.Max(4, Mathf.CeilToInt(Mathf.Max(voxel, volumeSizeMeters.x) / voxel) + 1);
        _dimY = Mathf.Max(4, Mathf.CeilToInt(Mathf.Max(voxel, volumeSizeMeters.y) / voxel) + 1);
        _dimZ = Mathf.Max(4, Mathf.CeilToInt(Mathf.Max(voxel, volumeSizeMeters.z) / voxel) + 1);

        Vector3 center = volumeAnchor != null ? volumeAnchor.position : GetCameraPosition();
        _volumeOriginWorld = center - new Vector3((_dimX - 1) * voxel, (_dimY - 1) * voxel, (_dimZ - 1) * voxel) * 0.5f;

        int count = _dimX * _dimY * _dimZ;
        _tsdf = new float[count];
        _weights = new byte[count];
        for (int i = 0; i < count; i++)
            _tsdf[i] = 1f;

        _volumeInitialized = true;
    }

    private bool TryBuildCellVertex(
        int x,
        int y,
        int z,
        Vector3[] cornerPositions,
        float[] cornerValues,
        byte[] cornerWeights,
        out Vector3 vertex)
    {
        vertex = Vector3.zero;
        bool hasPositive = false;
        bool hasNegative = false;
        int supportedCorners = 0;

        for (int i = 0; i < 8; i++)
        {
            int vx = x + CornerOffsetX[i];
            int vy = y + CornerOffsetY[i];
            int vz = z + CornerOffsetZ[i];
            int index = Index(vx, vy, vz);
            cornerPositions[i] = VoxelCenter(vx, vy, vz);
            cornerValues[i] = _tsdf[index];
            cornerWeights[i] = _weights[index];

            if (cornerWeights[i] >= minSurfaceCornerWeight)
            {
                supportedCorners++;
                if (cornerValues[i] < 0f) hasNegative = true;
                if (cornerValues[i] > 0f) hasPositive = true;
            }
        }

        if (supportedCorners < 2 || !hasPositive || !hasNegative)
            return false;

        Vector3 sum = Vector3.zero;
        int count = 0;
        for (int edge = 0; edge < 12; edge++)
        {
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            if (cornerWeights[a] < minSurfaceCornerWeight || cornerWeights[b] < minSurfaceCornerWeight)
                continue;

            float va = cornerValues[a];
            float vb = cornerValues[b];
            if ((va < 0f && vb < 0f) || (va > 0f && vb > 0f) || Mathf.Approximately(va, vb))
                continue;

            float t = Mathf.Clamp01(va / (va - vb));
            sum += Vector3.Lerp(cornerPositions[a], cornerPositions[b], t);
            count++;
        }

        if (count <= 0)
            return false;

        vertex = sum / count;
        return Finite(vertex);
    }

    private bool IsSignChange(int a, int b)
    {
        if (a < 0 || b < 0 || a >= _tsdf.Length || b >= _tsdf.Length)
            return false;
        if (_weights[a] < minSurfaceCornerWeight || _weights[b] < minSurfaceCornerWeight)
            return false;
        float va = _tsdf[a];
        float vb = _tsdf[b];
        return (va < 0f && vb > 0f) || (va > 0f && vb < 0f);
    }

    private void AddQuadAroundXEdge(int x, int y, int z, int cellX, int cellY, List<int> triangles)
    {
        if (y <= 0 || z <= 0 || x >= cellX)
            return;
        AddQuad(
            TryGetCellVertex(x, y, z, cellX, cellY),
            TryGetCellVertex(x, y - 1, z, cellX, cellY),
            TryGetCellVertex(x, y - 1, z - 1, cellX, cellY),
            TryGetCellVertex(x, y, z - 1, cellX, cellY),
            triangles);
    }

    private void AddQuadAroundYEdge(int x, int y, int z, int cellX, int cellY, List<int> triangles)
    {
        if (x <= 0 || z <= 0 || y >= cellY)
            return;
        AddQuad(
            TryGetCellVertex(x, y, z, cellX, cellY),
            TryGetCellVertex(x, y, z - 1, cellX, cellY),
            TryGetCellVertex(x - 1, y, z - 1, cellX, cellY),
            TryGetCellVertex(x - 1, y, z, cellX, cellY),
            triangles);
    }

    private void AddQuadAroundZEdge(int x, int y, int z, int cellX, int cellY, List<int> triangles)
    {
        if (x <= 0 || y <= 0 || z >= _dimZ - 1)
            return;
        AddQuad(
            TryGetCellVertex(x, y, z, cellX, cellY),
            TryGetCellVertex(x - 1, y, z, cellX, cellY),
            TryGetCellVertex(x - 1, y - 1, z, cellX, cellY),
            TryGetCellVertex(x, y - 1, z, cellX, cellY),
            triangles);
    }

    private int TryGetCellVertex(int x, int y, int z, int cellX, int cellY)
    {
        int cellZ = _dimZ - 1;
        if (x < 0 || y < 0 || z < 0 || x >= cellX || y >= cellY || z >= cellZ)
            return -1;
        return _cellVertexIndices[CellIndex(x, y, z, cellX, cellY)];
    }

    private void AddQuad(int a, int b, int c, int d, List<int> triangles)
    {
        if (a < 0 || b < 0 || c < 0 || d < 0)
            return;

        triangles.Add(a);
        triangles.Add(b);
        triangles.Add(c);
        triangles.Add(a);
        triangles.Add(c);
        triangles.Add(d);

        if (!doubleSidedTriangles)
            return;

        triangles.Add(c);
        triangles.Add(b);
        triangles.Add(a);
        triangles.Add(d);
        triangles.Add(c);
        triangles.Add(a);
    }

    private bool TryWorldToVoxel(Vector3 world, out int x, out int y, out int z)
    {
        Vector3 local = world - _volumeOriginWorld;
        float invVoxel = 1f / Mathf.Max(0.0001f, voxelSizeMeters);
        x = Mathf.RoundToInt(local.x * invVoxel);
        y = Mathf.RoundToInt(local.y * invVoxel);
        z = Mathf.RoundToInt(local.z * invVoxel);
        return x >= 0 && y >= 0 && z >= 0 && x < _dimX && y < _dimY && z < _dimZ;
    }

    private Vector3 VoxelCenter(int x, int y, int z)
    {
        return _volumeOriginWorld + new Vector3(x * voxelSizeMeters, y * voxelSizeMeters, z * voxelSizeMeters);
    }

    private int Index(int x, int y, int z)
    {
        return x + _dimX * (y + _dimY * z);
    }

    private static int CellIndex(int x, int y, int z, int cellX, int cellY)
    {
        return x + cellX * (y + cellY * z);
    }

    private Vector3 GetCameraPosition()
    {
        if (volumeAnchor != null)
            return volumeAnchor.position;
        Camera camera = Camera.main;
        return camera != null ? camera.transform.position : transform.position;
    }

    private void ResolveRefs()
    {
        if (rawDepthSource == null)
            rawDepthSource = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (rawDepthSource == null)
            rawDepthSource = FindAnyObjectByType<ScanCoverDepthGridPointCloud>(FindObjectsInactive.Include);
        if (volumeAnchor == null && Camera.main != null)
            volumeAnchor = Camera.main.transform;
    }

    private void EnsureObjects()
    {
        if (displayRoot == null)
            displayRoot = transform;

        if (_meshObject == null)
        {
            _meshObject = new GameObject("[ScanCover] TSDF Single Shell");
        }

        if (forceWorldSpaceDisplay)
        {
            _meshObject.transform.SetParent(null, true);
            _meshObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _meshObject.transform.localScale = Vector3.one;
        }
        else if (_meshObject.transform.parent != displayRoot)
        {
            _meshObject.transform.SetParent(displayRoot, false);
        }

        if (_meshFilter == null)
            _meshFilter = _meshObject.GetComponent<MeshFilter>();
        if (_meshFilter == null)
            _meshFilter = _meshObject.AddComponent<MeshFilter>();

        if (_meshRenderer == null)
            _meshRenderer = _meshObject.GetComponent<MeshRenderer>();
        if (_meshRenderer == null)
            _meshRenderer = _meshObject.AddComponent<MeshRenderer>();

        if (_mesh == null)
        {
            _mesh = new Mesh { name = "ScanCover_TSDF_SingleShell" };
            _mesh.indexFormat = IndexFormat.UInt32;
        }

        _meshFilter.sharedMesh = _mesh;
        _meshRenderer.sharedMaterial = ResolveMaterial();
    }

    private Material ResolveMaterial()
    {
        if (shellMaterialOverride != null)
            return shellMaterialOverride;

        if (_runtimeMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader == null)
                shader = Shader.Find("Sprites/Default");

            _runtimeMaterial = new Material(shader);
            _runtimeMaterial.name = "ScanCover_TSDF_SingleShell_Material";
        }

        if (_runtimeMaterial.HasProperty("_BaseColor"))
            _runtimeMaterial.SetColor("_BaseColor", shellColor);
        if (_runtimeMaterial.HasProperty("_Color"))
            _runtimeMaterial.SetColor("_Color", shellColor);
        if (_runtimeMaterial.HasProperty("_Cull"))
            _runtimeMaterial.SetFloat("_Cull", (float)CullMode.Off);
        _runtimeMaterial.renderQueue = (int)RenderQueue.Geometry;
        return _runtimeMaterial;
    }

    private void SuppressSourcePreview()
    {
        if (!hideSourcePreview || rawDepthSource == null)
            return;

        rawDepthSource.SetPreviewDisplayVisible(false);
    }

    private bool Warn(string message)
    {
        if (debugLog)
            Debug.LogWarning($"[ScanCoverTsdfSingleShellPrototype] {message}", this);
        return false;
    }

    private static bool Finite(Vector3 value)
    {
        return !(float.IsNaN(value.x) || float.IsNaN(value.y) || float.IsNaN(value.z) ||
                 float.IsInfinity(value.x) || float.IsInfinity(value.y) || float.IsInfinity(value.z));
    }
}

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverHybridSurfaceReceiver : MonoBehaviour
{
    private struct EdgeKey
    {
        public readonly int a;
        public readonly int b;

        public EdgeKey(int i0, int i1)
        {
            if (i0 <= i1)
            {
                a = i0;
                b = i1;
            }
            else
            {
                a = i1;
                b = i0;
            }
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (a * 397) ^ b;
            }
        }

        public override bool Equals(object obj)
        {
            return obj is EdgeKey other && other.a == a && other.b == b;
        }
    }

    [Header("Refs")]
    public Transform referenceFrame;
    public ScanCoverSkeletonBuilder_A builder;
    public ScanCoverTsdfBranch tsdfBranch;

    [Header("Voxel Support")]
    public bool allowVoxelOnlyFallback = false;
    [Min(1)] public int voxelMinHits = 2;
    [Min(0f)] public float skipVoxelIfTsdfWeightAtLeast = 1.5f;

    [Header("Hole Fill")]
    public bool enableHoleFill = true;
    public bool requireVoxelSupportForHoleFill = true;
    [Min(0f)] public float holeFillVoxelSupportRadiusMeters = 0.26f;
    [Min(3)] public int maxHoleBoundaryVertices = 40;
    [Min(0f)] public float maxHolePerimeterMeters = 5.5f;
    [Min(0f)] public float maxHoleSpanMeters = 2.4f;
    [Min(0f)] public float maxHoleAreaM2 = 2.5f;
    [Min(0f)] public float maxHolePlaneDeviationMeters = 0.08f;

    [Header("Cleanup")]
    public bool enableUnsupportedTriangleCull = true;
    public bool enableFloatingIslandCull = true;
    [Min(0f)] public float triangleSupportRadiusMeters = 0.18f;
    [Min(0f)] public float minTsdfWeightToKeepTriangle = 0.85f;
    [Min(0f)] public float maxUnsupportedTriangleAreaM2 = 0.03f;
    [Min(1)] public int minSupportedIslandTriangles = 48;
    [Min(0f)] public float minSupportedIslandAreaM2 = 0.08f;

    [Header("Display")]
    public bool showWhileScanning = false;
    public bool showWhenFrozen = true;
    public bool buildCollider = false;
    [Min(0)] public int minTrianglesToShow = 24;
    public Color surfaceColor = new Color(0.18f, 0.16f, 0.08f, 0.025f);
    public Color surfaceGridColor = new Color(1.0f, 0.88f, 0.18f, 0.84f);
    public Color surfaceFresnelColor = new Color(1.0f, 0.96f, 0.68f, 0.10f);
    [Range(0f, 1f)] public float surfaceBaseAlpha = 0.006f;
    [Min(0.1f)] public float surfaceGridScale = 4.0f;
    [Range(0.001f, 0.2f)] public float surfaceGridThickness = 0.006f;
    [Range(0f, 3f)] public float surfaceGridIntensity = 1.35f;
    [Range(0.1f, 8f)] public float surfaceFresnelPower = 2.1f;
    [Range(0f, 3f)] public float surfaceFresnelStrength = 0.25f;
    public bool doubleSided = false;

    [Header("Debug")]
    public bool debugLog = false;

    public string LastIssue { get; private set; }
    public int TriangleCount { get; private set; }

    private GameObject _surfaceRoot;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private MeshCollider _meshCollider;
    private Mesh _mesh;
    private Material _runtimeMaterial;

    private readonly List<ScanCoverSkeletonBuilder_A.CellInfo> _builderSnapshot = new List<ScanCoverSkeletonBuilder_A.CellInfo>(16384);
    private readonly HashSet<ScanCoverSkeletonBuilder_A.VoxelKey> _confirmedVoxels = new HashSet<ScanCoverSkeletonBuilder_A.VoxelKey>();
    private readonly List<Vector3> _tsdfVertices = new List<Vector3>(65536);
    private readonly List<int> _tsdfTriangles = new List<int>(131072);
    private readonly List<Vector3> _combinedVertices = new List<Vector3>(65536);
    private readonly List<int> _combinedTriangles = new List<int>(131072);
    private readonly Dictionary<EdgeKey, int> _edgeUseCounts = new Dictionary<EdgeKey, int>(131072);
    private readonly Dictionary<int, List<int>> _boundaryAdjacency = new Dictionary<int, List<int>>(4096);
    private readonly HashSet<EdgeKey> _visitedBoundaryEdges = new HashSet<EdgeKey>();
    private readonly HashSet<int> _loopVisitedVertices = new HashSet<int>();
    private readonly List<int> _boundaryLoop = new List<int>(128);
    private readonly List<int> _boundaryVertices = new List<int>(4096);
    private readonly List<Vector2> _projectedLoop = new List<Vector2>(128);
    private readonly List<int> _earClipIndices = new List<int>(128);
    private readonly List<int> _workingTriangles = new List<int>(131072);
    private readonly List<Vector3> _workingVertices = new List<Vector3>(65536);
    private readonly Dictionary<int, int> _vertexRemap = new Dictionary<int, int>(65536);
    private readonly Dictionary<int, List<int>> _vertexTriangleAdjacency = new Dictionary<int, List<int>>(65536);
    private readonly Queue<int> _triangleQueue = new Queue<int>(4096);
    private readonly List<int> _componentTriangleIds = new List<int>(4096);

    private void Awake()
    {
        ResolveRefs();
        EnsureSurface();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureSurface();
        Subscribe();
        UpdateVisibility();
    }

    private void OnDisable()
    {
        Unsubscribe();
    }

    private void OnDestroy()
    {
        ReleaseSurface();
    }

    private void Update()
    {
        UpdateVisibility();
    }

    public void EnsureInitialized()
    {
        ResolveRefs();
        EnsureSurface();
        Subscribe();
    }

    public void ClearAll()
    {
        LastIssue = null;
        TriangleCount = 0;
        _combinedVertices.Clear();
        _combinedTriangles.Clear();
        _confirmedVoxels.Clear();
        _tsdfVertices.Clear();
        _tsdfTriangles.Clear();
        if (_mesh != null)
            _mesh.Clear();
        if (_meshCollider != null)
            _meshCollider.sharedMesh = null;
        SetVisible(false);
    }

    public void RebuildSurface()
    {
        ResolveRefs();
        EnsureSurface();

        _combinedVertices.Clear();
        _combinedTriangles.Clear();
        _tsdfVertices.Clear();
        _tsdfTriangles.Clear();
        _confirmedVoxels.Clear();

        if (builder != null)
        {
            _builderSnapshot.Clear();
            builder.GetCellsSnapshot(_builderSnapshot);
            int minHits = Mathf.Max(1, voxelMinHits);
            for (int i = 0; i < _builderSnapshot.Count; i++)
            {
                if (_builderSnapshot[i].count >= minHits)
                    _confirmedVoxels.Add(_builderSnapshot[i].key);
            }
        }

        bool hasTsdf = tsdfBranch != null && tsdfBranch.TryCopySurfaceSnapshot(_tsdfVertices, _tsdfTriangles);
        int holeFillTriangles = 0;
        if (hasTsdf)
        {
            Vector3 tsdfOffset = tsdfBranch.volumeCenterLocal;
            for (int i = 0; i < _tsdfVertices.Count; i++)
                _combinedVertices.Add(_tsdfVertices[i] + tsdfOffset);
            _combinedTriangles.AddRange(_tsdfTriangles);

            if (enableHoleFill)
                holeFillTriangles = AppendHoleFillOnTsdfBoundary();
        }

        int voxelTriangles = 0;
        if (_confirmedVoxels.Count > 0 && !hasTsdf && allowVoxelOnlyFallback)
            voxelTriangles = AppendVoxelFallbackSurface();

        int removedTriangles = CleanupSurfaceGeometry();
        TriangleCount = _combinedTriangles.Count / 3;
        bool visible = TriangleCount >= Mathf.Max(0, minTrianglesToShow);

        _mesh.Clear();
        if (visible)
        {
            _mesh.SetVertices(_combinedVertices);
            _mesh.SetTriangles(_combinedTriangles, 0, true);
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
        }

        if (_meshCollider != null)
        {
            _meshCollider.sharedMesh = null;
            if (buildCollider && visible)
                _meshCollider.sharedMesh = _mesh;
        }

        LastIssue = visible ? null : (_confirmedVoxels.Count > 0 || hasTsdf ? "Hybrid surface generated too few triangles." : "Hybrid surface has no source data.");
        SetVisible(visible);

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverHybridSurfaceReceiver] build tsdf={hasTsdf}, " +
                $"holeFillTriangles={holeFillTriangles}, confirmed={_confirmedVoxels.Count}, " +
                $"voxelFallbackTriangles={voxelTriangles}, removedTriangles={removedTriangles}, " +
                $"triangles={TriangleCount}, visible={visible}");
        }
    }

    private void ResolveRefs()
    {
        if (builder == null)
            builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (tsdfBranch == null)
            tsdfBranch = GetComponent<ScanCoverTsdfBranch>();
        if (referenceFrame == null)
        {
            if (builder != null && builder.referenceFrame != null)
                referenceFrame = builder.referenceFrame;
            else
                referenceFrame = transform;
        }
    }

    private void Subscribe()
    {
        if (tsdfBranch != null)
        {
            tsdfBranch.SurfaceDataUpdated -= HandleTsdfSurfaceDataUpdated;
            tsdfBranch.SurfaceDataUpdated += HandleTsdfSurfaceDataUpdated;
        }
    }

    private void Unsubscribe()
    {
        if (tsdfBranch != null)
            tsdfBranch.SurfaceDataUpdated -= HandleTsdfSurfaceDataUpdated;
    }

    private void HandleTsdfSurfaceDataUpdated(ScanCoverTsdfBranch _)
    {
        RebuildSurface();
    }

    private int AppendHoleFillOnTsdfBoundary()
    {
        int triangleCountBefore = _combinedTriangles.Count / 3;
        BuildBoundaryAdjacency(_tsdfTriangles);

        _boundaryVertices.Clear();
        foreach (KeyValuePair<int, List<int>> pair in _boundaryAdjacency)
            _boundaryVertices.Add(pair.Key);

        for (int i = 0; i < _boundaryVertices.Count; i++)
        {
            int start = _boundaryVertices[i];
            if (!_boundaryAdjacency.TryGetValue(start, out List<int> startNeighbors) || startNeighbors == null || startNeighbors.Count != 2)
                continue;

            for (int n = 0; n < startNeighbors.Count; n++)
            {
                int next = startNeighbors[n];
                EdgeKey edge = new EdgeKey(start, next);
                if (_visitedBoundaryEdges.Contains(edge))
                    continue;

                if (!TryBuildBoundaryLoop(start, next))
                    continue;

                TryTriangulateBoundaryLoop();
            }
        }

        return (_combinedTriangles.Count / 3) - triangleCountBefore;
    }

    private void BuildBoundaryAdjacency(List<int> triangles)
    {
        _edgeUseCounts.Clear();
        _boundaryAdjacency.Clear();
        _visitedBoundaryEdges.Clear();

        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            CountEdge(triangles[i + 0], triangles[i + 1]);
            CountEdge(triangles[i + 1], triangles[i + 2]);
            CountEdge(triangles[i + 2], triangles[i + 0]);
        }

        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            AddBoundaryAdjacency(triangles[i + 0], triangles[i + 1]);
            AddBoundaryAdjacency(triangles[i + 1], triangles[i + 2]);
            AddBoundaryAdjacency(triangles[i + 2], triangles[i + 0]);
        }
    }

    private void CountEdge(int a, int b)
    {
        EdgeKey key = new EdgeKey(a, b);
        _edgeUseCounts.TryGetValue(key, out int count);
        _edgeUseCounts[key] = count + 1;
    }

    private void AddBoundaryAdjacency(int a, int b)
    {
        EdgeKey key = new EdgeKey(a, b);
        if (!_edgeUseCounts.TryGetValue(key, out int count) || count != 1)
            return;

        AddNeighbor(a, b);
        AddNeighbor(b, a);
    }

    private void AddNeighbor(int from, int to)
    {
        if (!_boundaryAdjacency.TryGetValue(from, out List<int> neighbors))
        {
            neighbors = new List<int>(2);
            _boundaryAdjacency[from] = neighbors;
        }

        if (!neighbors.Contains(to))
            neighbors.Add(to);
    }

    private bool TryBuildBoundaryLoop(int start, int next)
    {
        _boundaryLoop.Clear();
        _loopVisitedVertices.Clear();

        int previous = start;
        int current = next;

        _boundaryLoop.Add(start);
        _loopVisitedVertices.Add(start);
        _visitedBoundaryEdges.Add(new EdgeKey(start, next));

        int guard = _boundaryAdjacency.Count + 2;
        while (guard-- > 0)
        {
            if (!_boundaryAdjacency.TryGetValue(current, out List<int> neighbors) || neighbors == null || neighbors.Count != 2)
                return false;

            if (!_loopVisitedVertices.Add(current))
                return false;

            _boundaryLoop.Add(current);

            int candidateA = neighbors[0];
            int candidateB = neighbors[1];
            int candidate = candidateA == previous ? candidateB : candidateA;
            EdgeKey nextEdge = new EdgeKey(current, candidate);

            if (candidate == start)
            {
                _visitedBoundaryEdges.Add(nextEdge);
                return _boundaryLoop.Count >= 3;
            }

            if (_visitedBoundaryEdges.Contains(nextEdge))
                return false;

            _visitedBoundaryEdges.Add(nextEdge);
            previous = current;
            current = candidate;
        }

        return false;
    }

    private void TryTriangulateBoundaryLoop()
    {
        int vertexCount = _boundaryLoop.Count;
        if (vertexCount < 3 || vertexCount > Mathf.Max(3, maxHoleBoundaryVertices))
            return;

        Vector3 centroid = Vector3.zero;
        float perimeter = 0f;
        float maxSpan = 0f;
        for (int i = 0; i < vertexCount; i++)
            centroid += _combinedVertices[_boundaryLoop[i]];
        centroid /= vertexCount;

        if (requireVoxelSupportForHoleFill && builder != null && !HasVoxelSupportNearPosition(centroid))
            return;

        for (int i = 0; i < vertexCount; i++)
        {
            Vector3 a = _combinedVertices[_boundaryLoop[i]];
            Vector3 b = _combinedVertices[_boundaryLoop[(i + 1) % vertexCount]];
            perimeter += Vector3.Distance(a, b);
        }

        if (maxHolePerimeterMeters > 0f && perimeter > maxHolePerimeterMeters)
            return;

        Vector3 normal = ComputeLoopNormal(_boundaryLoop);
        if (normal.sqrMagnitude <= 1e-8f)
            return;
        normal.Normalize();

        for (int i = 0; i < vertexCount; i++)
        {
            Vector3 vi = _combinedVertices[_boundaryLoop[i]];
            float deviation = Mathf.Abs(Vector3.Dot(normal, vi - centroid));
            if (maxHolePlaneDeviationMeters > 0f && deviation > maxHolePlaneDeviationMeters)
                return;

            for (int j = i + 1; j < vertexCount; j++)
            {
                float span = Vector3.Distance(vi, _combinedVertices[_boundaryLoop[j]]);
                if (span > maxSpan)
                    maxSpan = span;
            }
        }

        if (maxHoleSpanMeters > 0f && maxSpan > maxHoleSpanMeters)
            return;

        Vector3 tangent = Vector3.zero;
        for (int i = 0; i < vertexCount; i++)
        {
            Vector3 edge = _combinedVertices[_boundaryLoop[(i + 1) % vertexCount]] - _combinedVertices[_boundaryLoop[i]];
            if (edge.sqrMagnitude > 1e-8f)
            {
                tangent = Vector3.ProjectOnPlane(edge, normal);
                if (tangent.sqrMagnitude > 1e-8f)
                    break;
            }
        }

        if (tangent.sqrMagnitude <= 1e-8f)
            return;

        tangent.Normalize();
        Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;
        if (bitangent.sqrMagnitude <= 1e-8f)
            return;

        _projectedLoop.Clear();
        for (int i = 0; i < vertexCount; i++)
        {
            Vector3 delta = _combinedVertices[_boundaryLoop[i]] - centroid;
            _projectedLoop.Add(new Vector2(Vector3.Dot(delta, tangent), Vector3.Dot(delta, bitangent)));
        }

        float signedArea = ComputeSignedArea(_projectedLoop);
        if (Mathf.Abs(signedArea) <= 1e-6f)
            return;

        if (maxHoleAreaM2 > 0f && Mathf.Abs(signedArea) > maxHoleAreaM2)
            return;

        if (signedArea < 0f)
        {
            _boundaryLoop.Reverse();
            _projectedLoop.Reverse();
        }

        if (!TriangulateProjectedLoop())
            return;
    }

    private Vector3 ComputeLoopNormal(List<int> loop)
    {
        Vector3 normal = Vector3.zero;
        int count = loop.Count;
        for (int i = 0; i < count; i++)
        {
            Vector3 current = _combinedVertices[loop[i]];
            Vector3 next = _combinedVertices[loop[(i + 1) % count]];
            normal.x += (current.y - next.y) * (current.z + next.z);
            normal.y += (current.z - next.z) * (current.x + next.x);
            normal.z += (current.x - next.x) * (current.y + next.y);
        }
        return normal;
    }

    private float ComputeSignedArea(List<Vector2> polygon)
    {
        float area = 0f;
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            area += a.x * b.y - b.x * a.y;
        }
        return area * 0.5f;
    }

    private bool TriangulateProjectedLoop()
    {
        int count = _boundaryLoop.Count;
        if (count < 3)
            return false;

        _earClipIndices.Clear();
        for (int i = 0; i < count; i++)
            _earClipIndices.Add(i);

        int guard = count * count;
        while (_earClipIndices.Count > 3 && guard-- > 0)
        {
            bool earFound = false;
            for (int i = 0; i < _earClipIndices.Count; i++)
            {
                int prevLocal = _earClipIndices[(i - 1 + _earClipIndices.Count) % _earClipIndices.Count];
                int currentLocal = _earClipIndices[i];
                int nextLocal = _earClipIndices[(i + 1) % _earClipIndices.Count];

                Vector2 a = _projectedLoop[prevLocal];
                Vector2 b = _projectedLoop[currentLocal];
                Vector2 c = _projectedLoop[nextLocal];
                if (SignedArea2(a, b, c) <= 1e-6f)
                    continue;

                bool containsPoint = false;
                for (int j = 0; j < _earClipIndices.Count; j++)
                {
                    int testLocal = _earClipIndices[j];
                    if (testLocal == prevLocal || testLocal == currentLocal || testLocal == nextLocal)
                        continue;

                    if (PointInTriangle(_projectedLoop[testLocal], a, b, c))
                    {
                        containsPoint = true;
                        break;
                    }
                }

                if (containsPoint)
                    continue;

                _combinedTriangles.Add(_boundaryLoop[prevLocal]);
                _combinedTriangles.Add(_boundaryLoop[currentLocal]);
                _combinedTriangles.Add(_boundaryLoop[nextLocal]);
                _earClipIndices.RemoveAt(i);
                earFound = true;
                break;
            }

            if (!earFound)
                return false;
        }

        if (_earClipIndices.Count == 3)
        {
            _combinedTriangles.Add(_boundaryLoop[_earClipIndices[0]]);
            _combinedTriangles.Add(_boundaryLoop[_earClipIndices[1]]);
            _combinedTriangles.Add(_boundaryLoop[_earClipIndices[2]]);
            return true;
        }

        return false;
    }

    private float SignedArea2(Vector2 a, Vector2 b, Vector2 c)
    {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }

    private bool PointInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
    {
        float area0 = SignedArea2(a, b, p);
        float area1 = SignedArea2(b, c, p);
        float area2 = SignedArea2(c, a, p);
        return area0 >= -1e-6f && area1 >= -1e-6f && area2 >= -1e-6f;
    }

    private int AppendVoxelFallbackSurface()
    {
        if (builder == null)
            return 0;

        float cellSize = Mathf.Max(1e-4f, builder.cellSizeMeters);
        int trianglesBefore = _combinedTriangles.Count;

        foreach (ScanCoverSkeletonBuilder_A.VoxelKey key in _confirmedVoxels)
        {
            if (ShouldSkipVoxelByTsdf(key, cellSize))
                continue;

            AddVoxelFaces(key, cellSize);
        }

        return (_combinedTriangles.Count - trianglesBefore) / 3;
    }

    private int CleanupSurfaceGeometry()
    {
        int removed = 0;

        if (enableUnsupportedTriangleCull)
            removed += CullUnsupportedTriangles();

        if (enableFloatingIslandCull)
            removed += CullFloatingIslands();

        return removed;
    }

    private int CullUnsupportedTriangles()
    {
        if (_combinedTriangles.Count <= 0)
            return 0;

        _workingTriangles.Clear();
        int removed = 0;
        for (int i = 0; i + 2 < _combinedTriangles.Count; i += 3)
        {
            int ia = _combinedTriangles[i + 0];
            int ib = _combinedTriangles[i + 1];
            int ic = _combinedTriangles[i + 2];

            Vector3 a = _combinedVertices[ia];
            Vector3 b = _combinedVertices[ib];
            Vector3 c = _combinedVertices[ic];
            float area = ComputeTriangleArea(a, b, c);
            if (area <= 1e-7f)
            {
                removed++;
                continue;
            }

            if (area < maxUnsupportedTriangleAreaM2 && !HasTriangleSupport((a + b + c) / 3f))
            {
                removed++;
                continue;
            }

            _workingTriangles.Add(ia);
            _workingTriangles.Add(ib);
            _workingTriangles.Add(ic);
        }

        if (removed > 0)
            CompactWorkingGeometry();

        return removed;
    }

    private int CullFloatingIslands()
    {
        int triangleCount = _combinedTriangles.Count / 3;
        if (triangleCount <= 0)
            return 0;

        _vertexTriangleAdjacency.Clear();
        for (int tri = 0; tri < triangleCount; tri++)
        {
            int baseIndex = tri * 3;
            AddTriangleAdjacency(_combinedTriangles[baseIndex + 0], tri);
            AddTriangleAdjacency(_combinedTriangles[baseIndex + 1], tri);
            AddTriangleAdjacency(_combinedTriangles[baseIndex + 2], tri);
        }

        bool[] visited = new bool[triangleCount];
        bool[] keepTriangle = new bool[triangleCount];
        int removed = 0;

        for (int tri = 0; tri < triangleCount; tri++)
        {
            if (visited[tri])
                continue;

            _componentTriangleIds.Clear();
            _triangleQueue.Clear();
            _triangleQueue.Enqueue(tri);
            visited[tri] = true;

            float totalArea = 0f;
            int supportedTriangleCount = 0;

            while (_triangleQueue.Count > 0)
            {
                int current = _triangleQueue.Dequeue();
                _componentTriangleIds.Add(current);

                int baseIndex = current * 3;
                Vector3 a = _combinedVertices[_combinedTriangles[baseIndex + 0]];
                Vector3 b = _combinedVertices[_combinedTriangles[baseIndex + 1]];
                Vector3 c = _combinedVertices[_combinedTriangles[baseIndex + 2]];
                totalArea += ComputeTriangleArea(a, b, c);
                if (HasTriangleSupport((a + b + c) / 3f))
                    supportedTriangleCount++;

                EnqueueAdjacentTriangles(_combinedTriangles[baseIndex + 0], visited);
                EnqueueAdjacentTriangles(_combinedTriangles[baseIndex + 1], visited);
                EnqueueAdjacentTriangles(_combinedTriangles[baseIndex + 2], visited);
            }

            bool keepComponent =
                supportedTriangleCount > 0 ||
                _componentTriangleIds.Count >= Mathf.Max(1, minSupportedIslandTriangles) ||
                totalArea >= Mathf.Max(0f, minSupportedIslandAreaM2);

            for (int i = 0; i < _componentTriangleIds.Count; i++)
            {
                int triangleId = _componentTriangleIds[i];
                keepTriangle[triangleId] = keepComponent;
                if (!keepComponent)
                    removed++;
            }
        }

        if (removed <= 0)
            return 0;

        _workingTriangles.Clear();
        for (int tri = 0; tri < triangleCount; tri++)
        {
            if (!keepTriangle[tri])
                continue;

            int baseIndex = tri * 3;
            _workingTriangles.Add(_combinedTriangles[baseIndex + 0]);
            _workingTriangles.Add(_combinedTriangles[baseIndex + 1]);
            _workingTriangles.Add(_combinedTriangles[baseIndex + 2]);
        }

        CompactWorkingGeometry();
        return removed;
    }

    private void AddTriangleAdjacency(int vertexIndex, int triangleId)
    {
        if (!_vertexTriangleAdjacency.TryGetValue(vertexIndex, out List<int> triangles))
        {
            triangles = new List<int>(4);
            _vertexTriangleAdjacency[vertexIndex] = triangles;
        }

        triangles.Add(triangleId);
    }

    private void EnqueueAdjacentTriangles(int vertexIndex, bool[] visited)
    {
        if (!_vertexTriangleAdjacency.TryGetValue(vertexIndex, out List<int> triangles))
            return;

        for (int i = 0; i < triangles.Count; i++)
        {
            int triangleId = triangles[i];
            if (visited[triangleId])
                continue;

            visited[triangleId] = true;
            _triangleQueue.Enqueue(triangleId);
        }
    }

    private void CompactWorkingGeometry()
    {
        _vertexRemap.Clear();
        _workingVertices.Clear();

        for (int i = 0; i < _workingTriangles.Count; i++)
        {
            int src = _workingTriangles[i];
            if (!_vertexRemap.TryGetValue(src, out int dst))
            {
                dst = _workingVertices.Count;
                _vertexRemap[src] = dst;
                _workingVertices.Add(_combinedVertices[src]);
            }

            _workingTriangles[i] = dst;
        }

        _combinedVertices.Clear();
        _combinedVertices.AddRange(_workingVertices);
        _combinedTriangles.Clear();
        _combinedTriangles.AddRange(_workingTriangles);
    }

    private bool HasTriangleSupport(Vector3 centroidReferenceLocal)
    {
        if (HasVoxelSupportNearPosition(centroidReferenceLocal, triangleSupportRadiusMeters))
            return true;

        return tsdfBranch != null &&
            tsdfBranch.TryGetWeightAtReferenceLocalPosition(centroidReferenceLocal, out float weight) &&
            weight >= Mathf.Max(0f, minTsdfWeightToKeepTriangle);
    }

    private float ComputeTriangleArea(Vector3 a, Vector3 b, Vector3 c)
    {
        return Vector3.Cross(b - a, c - a).magnitude * 0.5f;
    }

    private bool HasVoxelSupportNearPosition(Vector3 referenceLocalPosition, float supportRadiusMeters = -1f)
    {
        if (builder == null || _confirmedVoxels.Count <= 0)
            return false;

        float cellSize = Mathf.Max(1e-4f, builder.cellSizeMeters);
        float requestedRadius = supportRadiusMeters >= 0f ? supportRadiusMeters : holeFillVoxelSupportRadiusMeters;
        float radius = Mathf.Max(cellSize * 1.5f, requestedRadius);
        float radiusSq = radius * radius;

        foreach (ScanCoverSkeletonBuilder_A.VoxelKey key in _confirmedVoxels)
        {
            Vector3 center = new Vector3(
                (key.x + 0.5f) * cellSize,
                (key.y + 0.5f) * cellSize,
                (key.z + 0.5f) * cellSize);

            if ((center - referenceLocalPosition).sqrMagnitude <= radiusSq)
                return true;
        }

        return false;
    }

    private bool ShouldSkipVoxelByTsdf(ScanCoverSkeletonBuilder_A.VoxelKey key, float cellSize)
    {
        if (tsdfBranch == null)
            return false;

        Vector3 refLocalCenter = new Vector3(
            (key.x + 0.5f) * cellSize,
            (key.y + 0.5f) * cellSize,
            (key.z + 0.5f) * cellSize);

        return tsdfBranch.TryGetWeightAtReferenceLocalPosition(refLocalCenter, out float weight) &&
            weight >= Mathf.Max(0f, skipVoxelIfTsdfWeightAtLeast);
    }

    private void AddVoxelFaces(ScanCoverSkeletonBuilder_A.VoxelKey key, float cellSize)
    {
        float x0 = key.x * cellSize;
        float y0 = key.y * cellSize;
        float z0 = key.z * cellSize;
        float x1 = x0 + cellSize;
        float y1 = y0 + cellSize;
        float z1 = z0 + cellSize;

        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x + 1, key.y, key.z)))
            AddQuad(new Vector3(x1, y0, z0), new Vector3(x1, y0, z1), new Vector3(x1, y1, z1), new Vector3(x1, y1, z0));
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x - 1, key.y, key.z)))
            AddQuad(new Vector3(x0, y0, z1), new Vector3(x0, y0, z0), new Vector3(x0, y1, z0), new Vector3(x0, y1, z1));
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y + 1, key.z)))
            AddQuad(new Vector3(x0, y1, z0), new Vector3(x1, y1, z0), new Vector3(x1, y1, z1), new Vector3(x0, y1, z1));
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y - 1, key.z)))
            AddQuad(new Vector3(x0, y0, z1), new Vector3(x1, y0, z1), new Vector3(x1, y0, z0), new Vector3(x0, y0, z0));
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y, key.z + 1)))
            AddQuad(new Vector3(x1, y0, z1), new Vector3(x0, y0, z1), new Vector3(x0, y1, z1), new Vector3(x1, y1, z1));
        if (!_confirmedVoxels.Contains(new ScanCoverSkeletonBuilder_A.VoxelKey(key.x, key.y, key.z - 1)))
            AddQuad(new Vector3(x0, y0, z0), new Vector3(x1, y0, z0), new Vector3(x1, y1, z0), new Vector3(x0, y1, z0));
    }

    private void AddQuad(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
    {
        int baseIndex = _combinedVertices.Count;
        _combinedVertices.Add(a);
        _combinedVertices.Add(b);
        _combinedVertices.Add(c);
        _combinedVertices.Add(d);
        _combinedTriangles.Add(baseIndex + 0);
        _combinedTriangles.Add(baseIndex + 1);
        _combinedTriangles.Add(baseIndex + 2);
        _combinedTriangles.Add(baseIndex + 0);
        _combinedTriangles.Add(baseIndex + 2);
        _combinedTriangles.Add(baseIndex + 3);
    }

    private void EnsureSurface()
    {
        if (_surfaceRoot == null)
        {
            _surfaceRoot = new GameObject("[ScanCover] Hybrid Surface");
            Transform parent = referenceFrame != null ? referenceFrame : transform;
            _surfaceRoot.transform.SetParent(parent, false);
            _surfaceRoot.transform.localPosition = Vector3.zero;
            _surfaceRoot.transform.localRotation = Quaternion.identity;
            _surfaceRoot.transform.localScale = Vector3.one;
            _meshFilter = _surfaceRoot.AddComponent<MeshFilter>();
            _meshRenderer = _surfaceRoot.AddComponent<MeshRenderer>();
            _meshCollider = _surfaceRoot.AddComponent<MeshCollider>();
        }

        if (_mesh == null)
        {
            _mesh = new Mesh
            {
                name = "ScanCover_HybridSurfaceMesh",
                indexFormat = IndexFormat.UInt32,
            };
        }

        if (_meshFilter != null)
            _meshFilter.sharedMesh = _mesh;

        if (_runtimeMaterial == null)
        {
            Shader shader = Shader.Find("MRMotifs/ScanCover/ObservationSurface");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            if (shader != null)
            {
                _runtimeMaterial = new Material(shader)
                {
                    name = "ScanCover_HybridSurfaceMaterial"
                };
            }
        }

        ApplyMaterialSettings();

        if (_meshRenderer != null && _runtimeMaterial != null)
            _meshRenderer.sharedMaterial = _runtimeMaterial;
    }

    private void ApplyMaterialSettings()
    {
        if (_runtimeMaterial == null)
            return;

        if (_runtimeMaterial.HasProperty("_BaseColor"))
            _runtimeMaterial.SetColor("_BaseColor", surfaceColor);
        if (_runtimeMaterial.HasProperty("_Color"))
            _runtimeMaterial.SetColor("_Color", surfaceColor);
        if (_runtimeMaterial.HasProperty("_FresnelColor"))
            _runtimeMaterial.SetColor("_FresnelColor", surfaceFresnelColor);
        if (_runtimeMaterial.HasProperty("_GridColor"))
            _runtimeMaterial.SetColor("_GridColor", surfaceGridColor);
        if (_runtimeMaterial.HasProperty("_BaseAlpha"))
            _runtimeMaterial.SetFloat("_BaseAlpha", surfaceBaseAlpha);
        if (_runtimeMaterial.HasProperty("_FresnelPower"))
            _runtimeMaterial.SetFloat("_FresnelPower", surfaceFresnelPower);
        if (_runtimeMaterial.HasProperty("_FresnelStrength"))
            _runtimeMaterial.SetFloat("_FresnelStrength", surfaceFresnelStrength);
        if (_runtimeMaterial.HasProperty("_GridScale"))
            _runtimeMaterial.SetFloat("_GridScale", surfaceGridScale);
        if (_runtimeMaterial.HasProperty("_GridThickness"))
            _runtimeMaterial.SetFloat("_GridThickness", surfaceGridThickness);
        if (_runtimeMaterial.HasProperty("_GridIntensity"))
            _runtimeMaterial.SetFloat("_GridIntensity", surfaceGridIntensity);
        if (_runtimeMaterial.HasProperty("_Cull"))
            _runtimeMaterial.SetFloat("_Cull", doubleSided ? (float)CullMode.Off : (float)CullMode.Back);
        if (_runtimeMaterial.HasProperty("_Surface"))
            _runtimeMaterial.SetFloat("_Surface", 1f);
        if (_runtimeMaterial.HasProperty("_Blend"))
            _runtimeMaterial.SetFloat("_Blend", 0f);
        if (_runtimeMaterial.HasProperty("_ZWrite"))
            _runtimeMaterial.SetFloat("_ZWrite", 0f);
        if (_runtimeMaterial.HasProperty("_SrcBlend"))
            _runtimeMaterial.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (_runtimeMaterial.HasProperty("_DstBlend"))
            _runtimeMaterial.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        _runtimeMaterial.renderQueue = (int)RenderQueue.Transparent;
    }

    private void ReleaseSurface()
    {
        if (_mesh != null)
            Destroy(_mesh);
        _mesh = null;

        if (_runtimeMaterial != null)
            Destroy(_runtimeMaterial);
        _runtimeMaterial = null;

        if (_surfaceRoot != null)
            Destroy(_surfaceRoot);

        _surfaceRoot = null;
        _meshFilter = null;
        _meshRenderer = null;
        _meshCollider = null;
    }

    private void UpdateVisibility()
    {
        bool shouldShow = _surfaceRoot != null &&
            _mesh != null &&
            TriangleCount >= Mathf.Max(0, minTrianglesToShow) &&
            (showWhileScanning || builder == null || builder.IsFrozen) &&
            (showWhenFrozen || builder == null || !builder.IsFrozen);

        SetVisible(shouldShow);
    }

    private void SetVisible(bool visible)
    {
        if (_surfaceRoot != null && _surfaceRoot.activeSelf != visible)
            _surfaceRoot.SetActive(visible);
    }
}

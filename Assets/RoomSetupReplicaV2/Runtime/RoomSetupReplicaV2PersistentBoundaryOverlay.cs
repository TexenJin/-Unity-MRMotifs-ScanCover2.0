using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Dormant experimental TSDF-independent world-space boundary state and
/// optional admission gate. The reconstruction's rollback baseline does not
/// call Publish, so this component cannot alter production admission masks
/// unless its compile-time route switch is explicitly re-enabled.
/// StepSplit supplies only a region-of-interest seed. Accepted points around
/// that seed are separated into robust local plane patches, with a
/// gravity-constrained fallback for a thin closed-step face. Two
/// stereo-supported intersections may provisionally quarantine a bridge;
/// three consecutive stereo epochs make their world boundary persistent.
/// Continuous points that bridge the parallel layers away from the supported
/// face are removed before TSDF integration.
/// </summary>
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2PersistentBoundaryOverlay : MonoBehaviour
{
    [Header("Local surface fitting")]
    [SerializeField, Range(4, 32)] private int regionPaddingPixels = 18;
    [SerializeField, Range(8, 64)] private int minimumSeedPixels = 10;
    [SerializeField, Range(12, 128)] private int minimumPlaneInliers = 18;
    [SerializeField, Range(16, 160)] private int ransacIterations = 64;
    [SerializeField, Range(2, 5)] private int maximumPlanesPerRegion = 4;
    [SerializeField, Range(0.008f, 0.06f)] private float planeInlierMeters = 0.025f;
    [SerializeField, Range(256, 8192)] private int maximumRegionSamples = 3072;
    [SerializeField, Range(10f, 80f)] private float minimumPlaneAngleDegrees = 24f;
    [SerializeField, Min(0.05f)] private float minimumBoundarySpanMeters = 0.14f;
    [SerializeField, Range(0.10f, 0.60f)] private float maximumPatchToLineMeters = 0.36f;

    [Header("World persistence")]
    [SerializeField, Range(3f, 30f)] private float associationAngleDegrees = 12f;
    [SerializeField, Range(0.02f, 0.20f)] private float associationDistanceMeters = 0.09f;
    [SerializeField, Range(0.05f, 0.40f)] private float associationGapMeters = 0.22f;
    [SerializeField, Range(0.05f, 0.65f)] private float observationBlend = 0.18f;
    [SerializeField, Min(0.25f)] private float tentativeLifetimeSeconds = 1.5f;
    [SerializeField, Range(2, 8)] private int minimumStereoEpochsForGuard = 3;

    [Header("Gravity-constrained closed-step fallback")]
    [SerializeField, Range(0.01f, 0.04f)] private float layerHistogramBinMeters = 0.02f;
    [SerializeField, Range(0.015f, 0.06f)] private float layerSupportHalfThicknessMeters = 0.03f;
    [SerializeField, Range(0.02f, 0.08f)] private float faceInlierMeters = 0.04f;
    [SerializeField, Range(0.03f, 0.08f)] private float maximumFaceRmsMeters = 0.05f;
    [SerializeField, Range(0.10f, 0.60f)] private float minimumFaceHeightFraction = 0.40f;
    [SerializeField, Range(8, 64)] private int minimumFaceInliers = 12;

    [Header("Production bridge gate")]
    [SerializeField, Range(0.90f, 0.999f)]
    private float minimumParallelLayerNormalDot = 0.94f;
    [SerializeField, Range(0.05f, 0.45f)]
    private float maximumFaceLayerNormalDot = 0.45f;
    [SerializeField, Range(0.08f, 0.30f)]
    private float minimumLayerSeparationMeters = 0.12f;
    [SerializeField, Range(0.20f, 0.80f)]
    private float maximumLayerSeparationMeters = 0.55f;
    [SerializeField, Range(0.015f, 0.10f)]
    private float retainedFaceHalfThicknessMeters = 0.055f;
    [SerializeField, Range(0.10f, 0.50f)]
    private float maximumBridgeReachMeters = 0.42f;
    [SerializeField, Range(0.01f, 0.10f)]
    private float retainedLayerHalfThicknessMeters = 0.035f;
    [SerializeField, Range(0.02f, 0.20f)]
    private float boundaryEndMarginMeters = 0.20f;

    [Header("Display")]
    [SerializeField, Range(0.001f, 0.02f)] private float lineHalfWidthMeters = 0.005f;
    [SerializeField] private Color confirmedColor = new(1f, 0.05f, 0.72f, 0.96f);
    [SerializeField, Range(0.02f, 1f)] private float refreshInterval = 0.12f;

    private struct FittedSegment
    {
        public Vector3 A;
        public Vector3 B;
        public int EyeMask;
        public Vector3 PlaneANormal;
        public float PlaneAOffset;
        public Vector3 PlaneBNormal;
        public float PlaneBOffset;
    }

    private struct GuardedBand
    {
        public Vector3 FaceNormal;
        public float FaceOffset;
        public Vector3 LayerNormal;
        public float LayerMinimum;
        public float LayerMaximum;
        public Vector3 Axis;
        public float AxisMinimum;
        public float AxisMaximum;
    }

    private sealed class PlanePatch
    {
        public Vector3 Center;
        public Vector3 Normal;
        public float Offset;
        public readonly List<Vector3> Points = new(256);
    }

    private sealed class BoundaryTrack
    {
        public Vector3 A;
        public Vector3 B;
        public int LastEpoch;
        public int EyeMaskThisEpoch;
        public float LastSeen;
        public bool Confirmed;
        public int StereoEpochCount;
        public int LastStereoEpoch;
        public Vector3 PlaneANormal;
        public float PlaneAOffset;
        public Vector3 PlaneBNormal;
        public float PlaneBOffset;
    }

    private readonly List<BoundaryTrack> _tracks = new(64);
    private readonly List<int> _componentIndices = new(2048);
    private readonly List<Vector3> _regionPoints = new(3072);
    private readonly List<int> _activeIndices = new(3072);
    private readonly List<int> _bestInliers = new(3072);
    private readonly List<int> _finalInliers = new(3072);
    private readonly List<int> _gravityLowIndices = new(3072);
    private readonly List<int> _gravityHighIndices = new(3072);
    private readonly List<int> _gravityInteriorIndices = new(3072);
    private readonly List<int> _gravityBestInliers = new(3072);
    private readonly List<int> _gravityTrialInliers = new(3072);
    private readonly List<int> _heightPeakIndices = new(8);
    private readonly List<PlanePatch> _planes = new(4);
    private readonly List<Vector3> _vertices = new(2048);
    private readonly List<Color> _colors = new(2048);
    private readonly List<int> _triangles = new(3072);
    private readonly List<GuardedBand> _guardedBands = new(32);
    private readonly List<BoundaryTrack> _displayedTracks = new(64);

    private byte[] _seedMask;
    private byte[] _visited;
    private int[] _queue;
    private int[] _heightCounts;
    private int[] _heightSmooth;
    private Mesh _mesh;
    private GameObject _renderObject;
    private MeshRenderer _renderer;
    private Material _material;
    private bool _visible;
    private bool _frozen;
    private float _nextRefreshAt;
    private int _epoch;

    public bool Visible => _visible;
    public int GuardedBandCount => _guardedBands.Count;
    public int LastRejectedBridgePointCount { get; private set; }
    public long RejectedBridgePointCount { get; private set; }
    public int ConfirmedBoundaryCount
    {
        get
        {
            int count = 0;
            for (int i = 0; i < _tracks.Count; i++)
                if (_tracks[i].Confirmed) count++;
            return count;
        }
    }

    private void Awake()
    {
        Shader shader = Resources.Load<Shader>(
            "RoomSetupReplicaV2PersistentBoundary");
        if (shader == null)
            shader = Shader.Find(
                "Genesis/RoomSetupReplicaV2PersistentBoundary");
        if (shader == null)
        {
            // Rendering is optional. The production admission guard must not
            // disappear merely because its diagnostic shader is unavailable.
            return;
        }

        _mesh = new Mesh
        {
            name = "Room Setup Replica V2 Persistent World Boundaries",
            indexFormat = IndexFormat.UInt32,
            bounds = new Bounds(Vector3.zero, Vector3.one * 10000f)
        };
        _mesh.MarkDynamic();
        _renderObject = new GameObject(
            "Room Setup Replica V2 Persistent Boundary Display");
        _renderObject.transform.SetParent(transform, false);
        _renderObject.AddComponent<MeshFilter>().sharedMesh = _mesh;
        _renderer = _renderObject.AddComponent<MeshRenderer>();
        _renderer.shadowCastingMode = ShadowCastingMode.Off;
        _renderer.receiveShadows = false;
        _material = new Material(shader)
        {
            name = "Room Setup Replica V2 Persistent World Boundaries"
        };
        _renderer.sharedMaterial = _material;
        _renderer.enabled = false;
    }

    public void SetVisible(bool visible)
    {
        _visible = visible;
        if (_renderer != null)
            _renderer.enabled = visible && _mesh != null &&
                                _mesh.vertexCount > 0;
        if (visible)
        {
            _nextRefreshAt = 0f;
            RebuildMesh();
        }
    }

    public void SetFrozen(bool frozen)
    {
        _frozen = frozen;
    }

    public void ClearBoundaries()
    {
        _tracks.Clear();
        _guardedBands.Clear();
        _displayedTracks.Clear();
        _epoch = 0;
        LastRejectedBridgePointCount = 0;
        RejectedBridgePointCount = 0;
        _mesh?.Clear(false);
        if (_renderer != null)
            _renderer.enabled = false;
    }

    public void Publish(
        NativeArray<Color> leftPixels,
        float[] leftDepths,
        byte[] leftAccepted,
        byte[] leftStepQualified,
        NativeArray<Color> rightPixels,
        float[] rightDepths,
        byte[] rightAccepted,
        byte[] rightStepQualified,
        int width,
        int height)
    {
        LastRejectedBridgePointCount = 0;
        if (width <= 2 || height <= 2)
            return;
        int expected = width * height;
        if (!ValidEye(leftPixels, leftDepths, leftAccepted,
                leftStepQualified, expected) ||
            !ValidEye(rightPixels, rightDepths, rightAccepted,
                rightStepQualified, expected))
            return;

        float now = Time.unscaledTime;
        if (!_frozen && now >= _nextRefreshAt)
        {
            _nextRefreshAt = now + Mathf.Max(0.01f, refreshInterval);
            _epoch++;
            EnsureScratch(expected);
            ExtractEye(leftPixels, leftAccepted, leftStepQualified,
                width, height, 1);
            ExtractEye(rightPixels, rightAccepted, rightStepQualified,
                width, height, 2);
            CullTentativeTracks(now);
            RebuildGuardedBands();
            RebuildMesh();
        }

        // The display is deliberately downstream of this mutation. The
        // admission-point overlay and IntegrateChunkSlice therefore see the
        // same post-gate mask rather than a diagnostic-only approximation.
        ApplyGuard(leftPixels, leftAccepted, expected);
        ApplyGuard(rightPixels, rightAccepted, expected);
    }

    private static bool ValidEye(
        NativeArray<Color> pixels,
        float[] depths,
        byte[] accepted,
        byte[] qualified,
        int expected)
    {
        return pixels.IsCreated && pixels.Length >= expected &&
               depths != null && depths.Length >= expected &&
               accepted != null && accepted.Length >= expected &&
               qualified != null && qualified.Length >= expected;
    }

    private void EnsureScratch(int expected)
    {
        if (_seedMask == null || _seedMask.Length != expected)
        {
            _seedMask = new byte[expected];
            _visited = new byte[expected];
            _queue = new int[expected];
        }
        else
        {
            System.Array.Clear(_seedMask, 0, _seedMask.Length);
            System.Array.Clear(_visited, 0, _visited.Length);
        }
    }

    private void ExtractEye(
        NativeArray<Color> pixels,
        byte[] accepted,
        byte[] stepQualified,
        int width,
        int height,
        int eyeMask)
    {
        int expected = width * height;
        // Pixel indices are eye-local. Reusing the left-eye visited mask for
        // the right eye silently suppresses right-eye components at the same
        // image coordinates and corrupts stereo confirmation.
        System.Array.Clear(_seedMask, 0, expected);
        System.Array.Clear(_visited, 0, expected);
        for (int index = 0; index < expected; index++)
            if (accepted[index] != 0 && stepQualified[index] != 0)
                _seedMask[index] = 1;

        for (int index = 0; index < expected; index++)
        {
            if (_seedMask[index] == 0 || _visited[index] != 0)
                continue;
            CollectComponent(index, width, height,
                out int minX, out int maxX, out int minY, out int maxY);
            if (_componentIndices.Count < minimumSeedPixels)
                continue;

            minX = Mathf.Max(0, minX - regionPaddingPixels);
            maxX = Mathf.Min(width - 1, maxX + regionPaddingPixels);
            minY = Mathf.Max(0, minY - regionPaddingPixels);
            maxY = Mathf.Min(height - 1, maxY + regionPaddingPixels);
            CollectRegionPoints(pixels, accepted, width,
                minX, maxX, minY, maxY);
            if (_regionPoints.Count < minimumPlaneInliers * 2)
                continue;

            FitPlanePatches(index + eyeMask * 7919);
            AppendPatchIntersections(eyeMask);
            AppendGravityStepIntersections(
                index + eyeMask * 7919, eyeMask);
            ClearPlanePatches();
        }
    }

    private void CollectComponent(
        int seed,
        int width,
        int height,
        out int minX,
        out int maxX,
        out int minY,
        out int maxY)
    {
        _componentIndices.Clear();
        int seedX = seed % width;
        int seedY = seed / width;
        minX = maxX = seedX;
        minY = maxY = seedY;
        int head = 0;
        int tail = 0;
        _queue[tail++] = seed;
        _visited[seed] = 1;
        while (head < tail)
        {
            int index = _queue[head++];
            _componentIndices.Add(index);
            int x = index % width;
            int y = index / width;
            minX = Mathf.Min(minX, x);
            maxX = Mathf.Max(maxX, x);
            minY = Mathf.Min(minY, y);
            maxY = Mathf.Max(maxY, y);
            for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++)
            {
                if (ox == 0 && oy == 0)
                    continue;
                int nx = x + ox;
                int ny = y + oy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                int neighbor = nx + ny * width;
                if (_seedMask[neighbor] == 0 || _visited[neighbor] != 0)
                    continue;
                _visited[neighbor] = 1;
                _queue[tail++] = neighbor;
            }
        }
    }

    private void CollectRegionPoints(
        NativeArray<Color> pixels,
        byte[] accepted,
        int width,
        int minX,
        int maxX,
        int minY,
        int maxY)
    {
        _regionPoints.Clear();
        int area = (maxX - minX + 1) * (maxY - minY + 1);
        int stride = Mathf.Max(1,
            Mathf.CeilToInt(area /
                            (float)Mathf.Max(256, maximumRegionSamples)));
        int ordinal = 0;
        for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            int index = x + y * width;
            if (accepted[index] == 0 || ordinal++ % stride != 0)
                continue;
            Color encoded = pixels[index];
            Vector3 world = new(encoded.r, encoded.g, encoded.b);
            if (Finite(world))
                _regionPoints.Add(world);
        }
    }

    private void FitPlanePatches(int randomSeed)
    {
        ClearPlanePatches();
        _activeIndices.Clear();
        for (int i = 0; i < _regionPoints.Count; i++)
            _activeIndices.Add(i);

        uint state = unchecked((uint)randomSeed) | 1u;
        for (int planeIndex = 0;
             planeIndex < maximumPlanesPerRegion;
             planeIndex++)
        {
            if (_activeIndices.Count < minimumPlaneInliers)
                break;
            _bestInliers.Clear();
            for (int iteration = 0; iteration < ransacIterations; iteration++)
            {
                int a = NextIndex(ref state, _activeIndices.Count);
                int b = NextIndex(ref state, _activeIndices.Count);
                int c = NextIndex(ref state, _activeIndices.Count);
                if (a == b || a == c || b == c)
                    continue;
                Vector3 p0 = _regionPoints[_activeIndices[a]];
                Vector3 p1 = _regionPoints[_activeIndices[b]];
                Vector3 p2 = _regionPoints[_activeIndices[c]];
                Vector3 hypothesisNormal =
                    Vector3.Cross(p1 - p0, p2 - p0);
                if (hypothesisNormal.sqrMagnitude < 0.0000001f)
                    continue;
                hypothesisNormal.Normalize();
                float offset = Vector3.Dot(hypothesisNormal, p0);
                _finalInliers.Clear();
                for (int i = 0; i < _activeIndices.Count; i++)
                {
                    int pointIndex = _activeIndices[i];
                    if (Mathf.Abs(Vector3.Dot(hypothesisNormal,
                            _regionPoints[pointIndex]) - offset) <=
                        planeInlierMeters)
                        _finalInliers.Add(pointIndex);
                }
                if (_finalInliers.Count <= _bestInliers.Count)
                    continue;
                _bestInliers.Clear();
                _bestInliers.AddRange(_finalInliers);
            }

            if (_bestInliers.Count < minimumPlaneInliers ||
                !TryFitPlanePca(_bestInliers, out Vector3 center,
                    out Vector3 normal, out float rms,
                    out float middleVariance) ||
                rms > planeInlierMeters * 1.25f ||
                middleVariance < 0.00002f)
                break;

            float offsetRefined = Vector3.Dot(normal, center);
            _finalInliers.Clear();
            for (int i = 0; i < _activeIndices.Count; i++)
            {
                int pointIndex = _activeIndices[i];
                if (Mathf.Abs(Vector3.Dot(normal,
                        _regionPoints[pointIndex]) - offsetRefined) <=
                    planeInlierMeters * 1.25f)
                    _finalInliers.Add(pointIndex);
            }
            if (_finalInliers.Count < minimumPlaneInliers)
                break;

            var patch = new PlanePatch
            {
                Center = center,
                Normal = normal,
                Offset = offsetRefined
            };
            for (int i = 0; i < _finalInliers.Count; i++)
                patch.Points.Add(_regionPoints[_finalInliers[i]]);
            _planes.Add(patch);

            var retained = new List<int>(_activeIndices.Count);
            for (int i = 0; i < _activeIndices.Count; i++)
            {
                int pointIndex = _activeIndices[i];
                if (Mathf.Abs(Vector3.Dot(normal,
                        _regionPoints[pointIndex]) - offsetRefined) >
                    planeInlierMeters * 1.25f)
                    retained.Add(pointIndex);
            }
            _activeIndices.Clear();
            _activeIndices.AddRange(retained);
        }
    }

    private static int NextIndex(ref uint state, int count)
    {
        state = state * 1664525u + 1013904223u;
        return count > 0 ? (int)(state % (uint)count) : 0;
    }

    private bool TryFitPlanePca(
        List<int> indices,
        out Vector3 center,
        out Vector3 normal,
        out float rms,
        out float middleVariance)
    {
        center = Vector3.zero;
        normal = Vector3.up;
        rms = float.PositiveInfinity;
        middleVariance = 0f;
        if (indices == null || indices.Count < 3)
            return false;
        for (int i = 0; i < indices.Count; i++)
            center += _regionPoints[indices[i]];
        center /= indices.Count;

        float[,] matrix = new float[3, 3];
        for (int i = 0; i < indices.Count; i++)
        {
            Vector3 d = _regionPoints[indices[i]] - center;
            matrix[0, 0] += d.x * d.x;
            matrix[0, 1] += d.x * d.y;
            matrix[0, 2] += d.x * d.z;
            matrix[1, 1] += d.y * d.y;
            matrix[1, 2] += d.y * d.z;
            matrix[2, 2] += d.z * d.z;
        }
        matrix[1, 0] = matrix[0, 1];
        matrix[2, 0] = matrix[0, 2];
        matrix[2, 1] = matrix[1, 2];
        float[,] vectors =
        {
            { 1f, 0f, 0f },
            { 0f, 1f, 0f },
            { 0f, 0f, 1f }
        };
        for (int sweep = 0; sweep < 10; sweep++)
        {
            JacobiRotate(matrix, vectors, 0, 1);
            JacobiRotate(matrix, vectors, 0, 2);
            JacobiRotate(matrix, vectors, 1, 2);
        }
        int smallest = matrix[1, 1] < matrix[0, 0] ? 1 : 0;
        if (matrix[2, 2] < matrix[smallest, smallest]) smallest = 2;
        normal = new Vector3(vectors[0, smallest],
            vectors[1, smallest], vectors[2, smallest]).normalized;
        if (!Finite(normal) || normal.sqrMagnitude < 0.5f)
            return false;
        float[] eigen = { matrix[0, 0], matrix[1, 1], matrix[2, 2] };
        System.Array.Sort(eigen);
        middleVariance = Mathf.Max(0f, eigen[1]) /
                         Mathf.Max(1, indices.Count);
        float offset = Vector3.Dot(normal, center);
        float sumSquared = 0f;
        for (int i = 0; i < indices.Count; i++)
        {
            float residual = Vector3.Dot(normal,
                _regionPoints[indices[i]]) - offset;
            sumSquared += residual * residual;
        }
        rms = Mathf.Sqrt(sumSquared / Mathf.Max(1, indices.Count));
        return float.IsFinite(rms);
    }

    private static void JacobiRotate(
        float[,] matrix,
        float[,] vectors,
        int p,
        int q)
    {
        float apq = matrix[p, q];
        if (Mathf.Abs(apq) < 0.000000001f)
            return;
        float app = matrix[p, p];
        float aqq = matrix[q, q];
        float tau = (aqq - app) / (2f * apq);
        float t = (tau >= 0f ? 1f : -1f) /
                  (Mathf.Abs(tau) + Mathf.Sqrt(1f + tau * tau));
        float c = 1f / Mathf.Sqrt(1f + t * t);
        float s = t * c;
        for (int k = 0; k < 3; k++)
        {
            if (k == p || k == q)
                continue;
            float akp = matrix[k, p];
            float akq = matrix[k, q];
            float rotatedP = c * akp - s * akq;
            float rotatedQ = s * akp + c * akq;
            matrix[k, p] = matrix[p, k] = rotatedP;
            matrix[k, q] = matrix[q, k] = rotatedQ;
        }
        matrix[p, p] = app - t * apq;
        matrix[q, q] = aqq + t * apq;
        matrix[p, q] = matrix[q, p] = 0f;
        for (int k = 0; k < 3; k++)
        {
            float vkp = vectors[k, p];
            float vkq = vectors[k, q];
            vectors[k, p] = c * vkp - s * vkq;
            vectors[k, q] = s * vkp + c * vkq;
        }
    }

    private void AppendPatchIntersections(int eyeMask)
    {
        float maximumNormalDot = Mathf.Cos(
            minimumPlaneAngleDegrees * Mathf.Deg2Rad);
        for (int first = 0; first < _planes.Count; first++)
        for (int second = first + 1; second < _planes.Count; second++)
        {
            PlanePatch a = _planes[first];
            PlanePatch b = _planes[second];
            if (Mathf.Abs(Vector3.Dot(a.Normal, b.Normal)) >
                maximumNormalDot)
                continue;
            Vector3 unnormalizedDirection = Vector3.Cross(
                a.Normal, b.Normal);
            float denominator = unnormalizedDirection.sqrMagnitude;
            if (denominator < 0.000001f)
                continue;
            Vector3 linePoint = (
                a.Offset * Vector3.Cross(
                    b.Normal, unnormalizedDirection) +
                b.Offset * Vector3.Cross(
                    unnormalizedDirection, a.Normal)) /
                denominator;
            Vector3 direction = unnormalizedDirection /
                                Mathf.Sqrt(denominator);
            if (!Finite(linePoint))
                continue;
            if (DistancePointToLine(a.Center, linePoint, direction) >
                    maximumPatchToLineMeters ||
                DistancePointToLine(b.Center, linePoint, direction) >
                    maximumPatchToLineMeters)
                continue;

            ProjectInterval(a.Points, direction,
                out float aMin, out float aMax);
            ProjectInterval(b.Points, direction,
                out float bMin, out float bMax);
            float minimum = Mathf.Max(aMin, bMin);
            float maximum = Mathf.Min(aMax, bMax);
            if (maximum - minimum < minimumBoundarySpanMeters)
                continue;
            float lineProjection = Vector3.Dot(linePoint, direction);
            Vector3 start = linePoint + direction *
                (minimum - lineProjection);
            Vector3 end = linePoint + direction *
                (maximum - lineProjection);
            Accumulate(new FittedSegment
            {
                A = start,
                B = end,
                EyeMask = eyeMask,
                PlaneANormal = a.Normal,
                PlaneAOffset = a.Offset,
                PlaneBNormal = b.Normal,
                PlaneBOffset = b.Offset
            });
        }
    }

    private void AppendGravityStepIntersections(int randomSeed, int eyeMask)
    {
        if (_regionPoints.Count < Mathf.Max(80, minimumPlaneInliers * 2))
            return;

        float minimumY = float.PositiveInfinity;
        float maximumY = float.NegativeInfinity;
        for (int i = 0; i < _regionPoints.Count; i++)
        {
            minimumY = Mathf.Min(minimumY, _regionPoints[i].y);
            maximumY = Mathf.Max(maximumY, _regionPoints[i].y);
        }
        float binSize = Mathf.Max(0.005f, layerHistogramBinMeters);
        int binCount = Mathf.CeilToInt((maximumY - minimumY) / binSize) + 1;
        if (binCount < 4 || binCount > 512)
            return;
        EnsureHeightScratch(binCount);
        System.Array.Clear(_heightCounts, 0, binCount);
        System.Array.Clear(_heightSmooth, 0, binCount);
        for (int i = 0; i < _regionPoints.Count; i++)
        {
            int bin = Mathf.Clamp(
                Mathf.FloorToInt((_regionPoints[i].y - minimumY) / binSize),
                0, binCount - 1);
            _heightCounts[bin]++;
        }
        for (int i = 0; i < binCount; i++)
        {
            _heightSmooth[i] = _heightCounts[i] * 2;
            if (i > 0) _heightSmooth[i] += _heightCounts[i - 1];
            if (i + 1 < binCount)
                _heightSmooth[i] += _heightCounts[i + 1];
        }
        _heightPeakIndices.Clear();
        for (int i = 0; i < binCount; i++)
        {
            int previous = i > 0 ? _heightSmooth[i - 1] : -1;
            int following = i + 1 < binCount ? _heightSmooth[i + 1] : -1;
            if (_heightSmooth[i] >= previous &&
                _heightSmooth[i] > following &&
                _heightCounts[i] >= minimumPlaneInliers)
                _heightPeakIndices.Add(i);
        }
        _heightPeakIndices.Sort((a, b) =>
            _heightSmooth[b].CompareTo(_heightSmooth[a]));
        if (_heightPeakIndices.Count > 4)
            _heightPeakIndices.RemoveRange(4, _heightPeakIndices.Count - 4);

        bool found = false;
        float bestScore = float.NegativeInfinity;
        FittedSegment bestLow = default;
        FittedSegment bestHigh = default;
        uint state = unchecked((uint)(randomSeed ^ 0x71A9)) | 1u;
        for (int first = 0; first < _heightPeakIndices.Count; first++)
        for (int second = first + 1; second < _heightPeakIndices.Count; second++)
        {
            float firstHeight = minimumY +
                (_heightPeakIndices[first] + 0.5f) * binSize;
            float secondHeight = minimumY +
                (_heightPeakIndices[second] + 0.5f) * binSize;
            float low = Mathf.Min(firstHeight, secondHeight);
            float high = Mathf.Max(firstHeight, secondHeight);
            float separation = high - low;
            if (separation < minimumLayerSeparationMeters ||
                separation > maximumLayerSeparationMeters)
                continue;

            _gravityLowIndices.Clear();
            _gravityHighIndices.Clear();
            _gravityInteriorIndices.Clear();
            float support = Mathf.Max(0.005f,
                layerSupportHalfThicknessMeters);
            float margin = Mathf.Min(0.06f, separation * 0.22f);
            for (int i = 0; i < _regionPoints.Count; i++)
            {
                float y = _regionPoints[i].y;
                if (Mathf.Abs(y - low) <= support)
                    _gravityLowIndices.Add(i);
                if (Mathf.Abs(y - high) <= support)
                    _gravityHighIndices.Add(i);
                if (y >= low + margin && y <= high - margin)
                    _gravityInteriorIndices.Add(i);
            }
            if (_gravityLowIndices.Count < minimumPlaneInliers ||
                _gravityHighIndices.Count < minimumPlaneInliers ||
                _gravityInteriorIndices.Count < minimumPlaneInliers)
                continue;

            _gravityBestInliers.Clear();
            int iterations = Mathf.Max(16, ransacIterations);
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                int a = NextIndex(ref state, _gravityInteriorIndices.Count);
                int b = NextIndex(ref state, _gravityInteriorIndices.Count);
                int c = NextIndex(ref state, _gravityInteriorIndices.Count);
                if (a == b || a == c || b == c)
                    continue;
                Vector3 p0 = _regionPoints[_gravityInteriorIndices[a]];
                Vector3 p1 = _regionPoints[_gravityInteriorIndices[b]];
                Vector3 p2 = _regionPoints[_gravityInteriorIndices[c]];
                Vector3 normal = Vector3.Cross(p1 - p0, p2 - p0);
                if (normal.sqrMagnitude < 0.0000001f)
                    continue;
                normal.Normalize();
                if (Mathf.Abs(normal.y) > 0.30f)
                    continue;
                float offset = Vector3.Dot(normal, p0);
                _gravityTrialInliers.Clear();
                float minimumInlierY = float.PositiveInfinity;
                float maximumInlierY = float.NegativeInfinity;
                for (int i = 0; i < _gravityInteriorIndices.Count; i++)
                {
                    int pointIndex = _gravityInteriorIndices[i];
                    Vector3 point = _regionPoints[pointIndex];
                    if (Mathf.Abs(Vector3.Dot(normal, point) - offset) >
                        faceInlierMeters)
                        continue;
                    _gravityTrialInliers.Add(pointIndex);
                    minimumInlierY = Mathf.Min(minimumInlierY, point.y);
                    maximumInlierY = Mathf.Max(maximumInlierY, point.y);
                }
                if (maximumInlierY - minimumInlierY <
                    separation * minimumFaceHeightFraction ||
                    _gravityTrialInliers.Count <= _gravityBestInliers.Count)
                    continue;
                _gravityBestInliers.Clear();
                _gravityBestInliers.AddRange(_gravityTrialInliers);
            }
            if (_gravityBestInliers.Count < minimumFaceInliers ||
                !TryFitGravityVerticalPlane(_gravityBestInliers,
                    out Vector3 faceCenter, out Vector3 faceNormal,
                    out float faceRms, out float alongVariance) ||
                faceRms > maximumFaceRmsMeters ||
                alongVariance < 0.00001f)
                continue;

            float refinedFaceOffset = Vector3.Dot(faceNormal, faceCenter);
            float faceMinimumY = float.PositiveInfinity;
            float faceMaximumY = float.NegativeInfinity;
            for (int i = 0; i < _gravityBestInliers.Count; i++)
            {
                float y = _regionPoints[_gravityBestInliers[i]].y;
                faceMinimumY = Mathf.Min(faceMinimumY, y);
                faceMaximumY = Mathf.Max(faceMaximumY, y);
            }
            if (faceMaximumY - faceMinimumY <
                separation * minimumFaceHeightFraction ||
                !TryBuildGravityBoundarySegment(low, _gravityLowIndices,
                    faceNormal, refinedFaceOffset, eyeMask,
                    out FittedSegment lowSegment) ||
                !TryBuildGravityBoundarySegment(high, _gravityHighIndices,
                    faceNormal, refinedFaceOffset, eyeMask,
                    out FittedSegment highSegment))
                continue;

            float score = _gravityBestInliers.Count + 0.05f *
                Mathf.Min(_gravityLowIndices.Count,
                    _gravityHighIndices.Count);
            if (score <= bestScore)
                continue;
            bestScore = score;
            bestLow = lowSegment;
            bestHigh = highSegment;
            found = true;
        }

        if (!found)
            return;
        Accumulate(bestLow);
        Accumulate(bestHigh);
    }

    private void EnsureHeightScratch(int count)
    {
        if (_heightCounts != null && _heightCounts.Length >= count)
            return;
        int capacity = Mathf.NextPowerOfTwo(Mathf.Max(16, count));
        _heightCounts = new int[capacity];
        _heightSmooth = new int[capacity];
    }

    private bool TryFitGravityVerticalPlane(
        List<int> indices,
        out Vector3 center,
        out Vector3 normal,
        out float rms,
        out float alongVariance)
    {
        center = Vector3.zero;
        normal = Vector3.forward;
        rms = float.PositiveInfinity;
        alongVariance = 0f;
        if (indices == null || indices.Count < 3)
            return false;
        for (int i = 0; i < indices.Count; i++)
            center += _regionPoints[indices[i]];
        center /= indices.Count;

        float xx = 0f;
        float xz = 0f;
        float zz = 0f;
        for (int i = 0; i < indices.Count; i++)
        {
            Vector3 delta = _regionPoints[indices[i]] - center;
            xx += delta.x * delta.x;
            xz += delta.x * delta.z;
            zz += delta.z * delta.z;
        }
        float discriminant = Mathf.Sqrt(
            Mathf.Max(0f, (xx - zz) * (xx - zz) + 4f * xz * xz));
        float smallest = (xx + zz - discriminant) * 0.5f;
        float largest = (xx + zz + discriminant) * 0.5f;
        Vector2 horizontalNormal;
        if (Mathf.Abs(xz) > 0.0000001f)
            horizontalNormal = new Vector2(xz, smallest - xx).normalized;
        else
            horizontalNormal = xx <= zz ? Vector2.right : Vector2.up;
        normal = new Vector3(
            horizontalNormal.x, 0f, horizontalNormal.y).normalized;
        float offset = Vector3.Dot(normal, center);
        float sumSquared = 0f;
        for (int i = 0; i < indices.Count; i++)
        {
            float residual = Vector3.Dot(normal,
                _regionPoints[indices[i]]) - offset;
            sumSquared += residual * residual;
        }
        rms = Mathf.Sqrt(sumSquared / Mathf.Max(1, indices.Count));
        alongVariance = Mathf.Max(0f, largest) /
                        Mathf.Max(1, indices.Count);
        return Finite(normal) && float.IsFinite(rms);
    }

    private bool TryBuildGravityBoundarySegment(
        float height,
        List<int> layerIndices,
        Vector3 faceNormal,
        float faceOffset,
        int eyeMask,
        out FittedSegment segment)
    {
        segment = default;
        Vector3 axisUnnormalized = Vector3.Cross(Vector3.up, faceNormal);
        float denominator = axisUnnormalized.sqrMagnitude;
        if (denominator < 0.000001f)
            return false;
        Vector3 linePoint = (
            height * Vector3.Cross(faceNormal, axisUnnormalized) +
            faceOffset * Vector3.Cross(axisUnnormalized, Vector3.up)) /
            denominator;
        Vector3 axis = axisUnnormalized / Mathf.Sqrt(denominator);
        float faceMinimum = float.PositiveInfinity;
        float faceMaximum = float.NegativeInfinity;
        for (int i = 0; i < _gravityBestInliers.Count; i++)
        {
            float projection = Vector3.Dot(
                _regionPoints[_gravityBestInliers[i]], axis);
            faceMinimum = Mathf.Min(faceMinimum, projection);
            faceMaximum = Mathf.Max(faceMaximum, projection);
        }
        float layerMinimum = float.PositiveInfinity;
        float layerMaximum = float.NegativeInfinity;
        for (int i = 0; i < layerIndices.Count; i++)
        {
            float projection = Vector3.Dot(
                _regionPoints[layerIndices[i]], axis);
            layerMinimum = Mathf.Min(layerMinimum, projection);
            layerMaximum = Mathf.Max(layerMaximum, projection);
        }
        float minimum = Mathf.Max(faceMinimum, layerMinimum);
        float maximum = Mathf.Min(faceMaximum, layerMaximum);
        if (maximum - minimum < minimumBoundarySpanMeters)
            return false;
        float lineProjection = Vector3.Dot(linePoint, axis);
        segment = new FittedSegment
        {
            A = linePoint + axis * (minimum - lineProjection),
            B = linePoint + axis * (maximum - lineProjection),
            EyeMask = eyeMask,
            PlaneANormal = Vector3.up,
            PlaneAOffset = height,
            PlaneBNormal = faceNormal,
            PlaneBOffset = faceOffset
        };
        return true;
    }

    private static void ProjectInterval(
        List<Vector3> points,
        Vector3 axis,
        out float minimum,
        out float maximum)
    {
        minimum = float.PositiveInfinity;
        maximum = float.NegativeInfinity;
        for (int i = 0; i < points.Count; i++)
        {
            float projection = Vector3.Dot(points[i], axis);
            minimum = Mathf.Min(minimum, projection);
            maximum = Mathf.Max(maximum, projection);
        }
    }

    private static float DistancePointToLine(
        Vector3 point,
        Vector3 linePoint,
        Vector3 lineDirection)
    {
        Vector3 delta = point - linePoint;
        return (delta - lineDirection *
                Vector3.Dot(delta, lineDirection)).magnitude;
    }

    private void ClearPlanePatches()
    {
        for (int i = 0; i < _planes.Count; i++)
            _planes[i].Points.Clear();
        _planes.Clear();
    }

    private void Accumulate(FittedSegment observation)
    {
        Vector3 observedDirection = observation.B - observation.A;
        float observedLength = observedDirection.magnitude;
        if (observedLength < minimumBoundarySpanMeters)
            return;
        observedDirection /= observedLength;
        Vector3 observedCenter = (observation.A + observation.B) * 0.5f;
        BoundaryTrack best = null;
        float bestScore = float.PositiveInfinity;
        float minimumDot = Mathf.Cos(
            associationAngleDegrees * Mathf.Deg2Rad);
        for (int i = 0; i < _tracks.Count; i++)
        {
            BoundaryTrack track = _tracks[i];
            if (!PlanePairsCompatible(track, observation))
                continue;
            Vector3 trackDirection = (track.B - track.A).normalized;
            float alignment = Mathf.Abs(Vector3.Dot(
                observedDirection, trackDirection));
            if (alignment < minimumDot)
                continue;
            Vector3 trackCenter = (track.A + track.B) * 0.5f;
            float perpendicular = DistancePointToLine(
                observedCenter, trackCenter, trackDirection);
            if (perpendicular > associationDistanceMeters)
                continue;
            float gap = SegmentGap(track.A, track.B,
                observation.A, observation.B, trackDirection);
            if (gap > associationGapMeters)
                continue;
            float score = perpendicular + gap * 0.5f +
                          (1f - alignment) * 0.25f;
            if (score >= bestScore)
                continue;
            best = track;
            bestScore = score;
        }

        if (best == null)
        {
            _tracks.Add(new BoundaryTrack
            {
                A = observation.A,
                B = observation.B,
                LastEpoch = _epoch,
                EyeMaskThisEpoch = observation.EyeMask,
                LastSeen = Time.unscaledTime,
                PlaneANormal = observation.PlaneANormal,
                PlaneAOffset = observation.PlaneAOffset,
                PlaneBNormal = observation.PlaneBNormal,
                PlaneBOffset = observation.PlaneBOffset
            });
            return;
        }

        Vector3 oldDirection = (best.B - best.A).normalized;
        if (Vector3.Dot(oldDirection, observedDirection) < 0f)
        {
            Vector3 swap = observation.A;
            observation.A = observation.B;
            observation.B = swap;
            observedDirection = -observedDirection;
        }
        if (best.LastEpoch != _epoch)
        {
            best.LastEpoch = _epoch;
            best.EyeMaskThisEpoch = observation.EyeMask;
        }
        else
        {
            best.EyeMaskThisEpoch |= observation.EyeMask;
        }
        if (best.EyeMaskThisEpoch == 3 &&
            best.LastStereoEpoch != _epoch)
        {
            best.LastStereoEpoch = _epoch;
            best.StereoEpochCount++;
            if (best.StereoEpochCount >=
                Mathf.Max(2, minimumStereoEpochsForGuard))
                best.Confirmed = true;
        }

        Vector3 oldCenter = (best.A + best.B) * 0.5f;
        float oldHalfLength = Vector3.Distance(best.A, best.B) * 0.5f;
        float newHalfLength = Vector3.Distance(
            observation.A, observation.B) * 0.5f;
        float alpha = Mathf.Clamp01(observationBlend);
        AlignObservationPlanes(best, ref observation);
        Vector3 center = Vector3.Lerp(oldCenter, observedCenter, alpha);
        Vector3 direction = Vector3.Slerp(
            oldDirection, observedDirection, alpha).normalized;
        float halfLength = Mathf.Max(oldHalfLength,
            Mathf.Lerp(oldHalfLength, newHalfLength, alpha));
        best.A = center - direction * halfLength;
        best.B = center + direction * halfLength;
        BlendPlane(ref best.PlaneANormal, ref best.PlaneAOffset,
            observation.PlaneANormal, observation.PlaneAOffset, alpha);
        BlendPlane(ref best.PlaneBNormal, ref best.PlaneBOffset,
            observation.PlaneBNormal, observation.PlaneBOffset, alpha);
        best.LastSeen = Time.unscaledTime;
    }

    private bool PlanePairsCompatible(
        BoundaryTrack track, FittedSegment observation)
    {
        AlignObservationPlanes(track, ref observation);
        float minimumDot = Mathf.Cos(
            associationAngleDegrees * Mathf.Deg2Rad);
        return Vector3.Dot(track.PlaneANormal,
                   observation.PlaneANormal) >= minimumDot &&
               Vector3.Dot(track.PlaneBNormal,
                   observation.PlaneBNormal) >= minimumDot &&
               Mathf.Abs(track.PlaneAOffset -
                   observation.PlaneAOffset) <= associationDistanceMeters &&
               Mathf.Abs(track.PlaneBOffset -
                   observation.PlaneBOffset) <= associationDistanceMeters;
    }

    private static void AlignObservationPlanes(
        BoundaryTrack track, ref FittedSegment observation)
    {
        float direct = Mathf.Abs(Vector3.Dot(
                           track.PlaneANormal,
                           observation.PlaneANormal)) +
                       Mathf.Abs(Vector3.Dot(
                           track.PlaneBNormal,
                           observation.PlaneBNormal));
        float swapped = Mathf.Abs(Vector3.Dot(
                            track.PlaneANormal,
                            observation.PlaneBNormal)) +
                        Mathf.Abs(Vector3.Dot(
                            track.PlaneBNormal,
                            observation.PlaneANormal));
        if (swapped > direct)
        {
            Vector3 normal = observation.PlaneANormal;
            float offset = observation.PlaneAOffset;
            observation.PlaneANormal = observation.PlaneBNormal;
            observation.PlaneAOffset = observation.PlaneBOffset;
            observation.PlaneBNormal = normal;
            observation.PlaneBOffset = offset;
        }
        AlignPlane(track.PlaneANormal,
            ref observation.PlaneANormal,
            ref observation.PlaneAOffset);
        AlignPlane(track.PlaneBNormal,
            ref observation.PlaneBNormal,
            ref observation.PlaneBOffset);
    }

    private static void BlendPlane(
        ref Vector3 normal,
        ref float offset,
        Vector3 observedNormal,
        float observedOffset,
        float alpha)
    {
        AlignPlane(normal, ref observedNormal, ref observedOffset);
        normal = Vector3.Slerp(normal, observedNormal, alpha).normalized;
        offset = Mathf.Lerp(offset, observedOffset, alpha);
    }

    private static void AlignPlane(
        Vector3 reference,
        ref Vector3 normal,
        ref float offset)
    {
        if (Vector3.Dot(reference, normal) >= 0f)
            return;
        normal = -normal;
        offset = -offset;
    }

    private void RebuildGuardedBands()
    {
        _guardedBands.Clear();
        _displayedTracks.Clear();
        for (int first = 0; first < _tracks.Count; first++)
        for (int second = first + 1; second < _tracks.Count; second++)
        {
            BoundaryTrack a = _tracks[first];
            BoundaryTrack b = _tracks[second];
            if (!TryBuildGuardedBand(a, b, out GuardedBand band))
                continue;
            _guardedBands.Add(band);
            if (a.Confirmed && b.Confirmed)
            {
                if (!_displayedTracks.Contains(a)) _displayedTracks.Add(a);
                if (!_displayedTracks.Contains(b)) _displayedTracks.Add(b);
            }
        }
    }

    private bool TryBuildGuardedBand(
        BoundaryTrack first,
        BoundaryTrack second,
        out GuardedBand band)
    {
        band = default;
        // Quarantine a stereo-supported bridge immediately so an uncertain
        // first observation cannot become irreversible TSDF history.  If it
        // does not persist, the tentative tracks expire and later frames can
        // fill the temporary hole.  Three stereo epochs are still required
        // before the world boundary itself becomes persistent/confirmed.
        if (first.StereoEpochCount < 1 || second.StereoEpochCount < 1)
            return false;
        float minimumSharedDot = Mathf.Cos(
            associationAngleDegrees * Mathf.Deg2Rad);
        for (int firstShared = 0; firstShared < 2; firstShared++)
        for (int secondShared = 0; secondShared < 2; secondShared++)
        {
            GetPlane(first, firstShared,
                out Vector3 firstFace, out float firstFaceOffset);
            GetPlane(second, secondShared,
                out Vector3 secondFace, out float secondFaceOffset);
            AlignPlane(firstFace, ref secondFace, ref secondFaceOffset);
            if (Vector3.Dot(firstFace, secondFace) < minimumSharedDot ||
                Mathf.Abs(firstFaceOffset - secondFaceOffset) >
                    associationDistanceMeters)
                continue;

            GetPlane(first, 1 - firstShared,
                out Vector3 firstLayer, out float firstLayerOffset);
            GetPlane(second, 1 - secondShared,
                out Vector3 secondLayer, out float secondLayerOffset);
            AlignPlane(firstLayer, ref secondLayer,
                ref secondLayerOffset);
            if (Vector3.Dot(firstLayer, secondLayer) <
                    minimumParallelLayerNormalDot)
                continue;

            Vector3 faceNormal = Vector3.Slerp(
                firstFace, secondFace, 0.5f).normalized;
            float faceOffset = (firstFaceOffset + secondFaceOffset) * 0.5f;
            Vector3 layerNormal = Vector3.Slerp(
                firstLayer, secondLayer, 0.5f).normalized;
            if (Mathf.Abs(Vector3.Dot(faceNormal, layerNormal)) >
                    maximumFaceLayerNormalDot)
                continue;
            float layerMinimum = Mathf.Min(
                firstLayerOffset, secondLayerOffset);
            float layerMaximum = Mathf.Max(
                firstLayerOffset, secondLayerOffset);
            float separation = layerMaximum - layerMinimum;
            if (separation < minimumLayerSeparationMeters ||
                separation > maximumLayerSeparationMeters)
                continue;

            Vector3 axis = Vector3.Cross(faceNormal, layerNormal);
            if (axis.sqrMagnitude < 0.000001f)
                continue;
            axis.Normalize();
            ProjectSegment(first.A, first.B, axis,
                out float firstMin, out float firstMax);
            ProjectSegment(second.A, second.B, axis,
                out float secondMin, out float secondMax);
            float axisMinimum = Mathf.Max(firstMin, secondMin);
            float axisMaximum = Mathf.Min(firstMax, secondMax);
            if (axisMaximum - axisMinimum < minimumBoundarySpanMeters)
                continue;

            band = new GuardedBand
            {
                FaceNormal = faceNormal,
                FaceOffset = faceOffset,
                LayerNormal = layerNormal,
                LayerMinimum = layerMinimum,
                LayerMaximum = layerMaximum,
                Axis = axis,
                AxisMinimum = axisMinimum,
                AxisMaximum = axisMaximum
            };
            return true;
        }
        return false;
    }

    private static void GetPlane(
        BoundaryTrack track,
        int index,
        out Vector3 normal,
        out float offset)
    {
        if (index == 0)
        {
            normal = track.PlaneANormal;
            offset = track.PlaneAOffset;
        }
        else
        {
            normal = track.PlaneBNormal;
            offset = track.PlaneBOffset;
        }
    }

    private static void ProjectSegment(
        Vector3 a,
        Vector3 b,
        Vector3 axis,
        out float minimum,
        out float maximum)
    {
        float first = Vector3.Dot(a, axis);
        float second = Vector3.Dot(b, axis);
        minimum = Mathf.Min(first, second);
        maximum = Mathf.Max(first, second);
    }

    private void ApplyGuard(
        NativeArray<Color> pixels,
        byte[] accepted,
        int expected)
    {
        if (_guardedBands.Count == 0)
            return;
        int count = Mathf.Min(expected,
            Mathf.Min(pixels.Length, accepted.Length));
        for (int index = 0; index < count; index++)
        {
            if (accepted[index] == 0)
                continue;
            Color encoded = pixels[index];
            Vector3 world = new Vector3(encoded.r, encoded.g, encoded.b);
            if (!Finite(world))
                continue;
            for (int guard = 0; guard < _guardedBands.Count; guard++)
            {
                GuardedBand band = _guardedBands[guard];
                float axis = Vector3.Dot(world, band.Axis);
                if (axis < band.AxisMinimum - boundaryEndMarginMeters ||
                    axis > band.AxisMaximum + boundaryEndMarginMeters)
                    continue;
                float layer = Vector3.Dot(world, band.LayerNormal);
                if (layer <= band.LayerMinimum +
                        retainedLayerHalfThicknessMeters ||
                    layer >= band.LayerMaximum -
                        retainedLayerHalfThicknessMeters)
                    continue;
                float faceResidual = Mathf.Abs(
                    Vector3.Dot(world, band.FaceNormal) -
                    band.FaceOffset);
                if (faceResidual <= retainedFaceHalfThicknessMeters ||
                    faceResidual > maximumBridgeReachMeters)
                    continue;
                accepted[index] = 0;
                LastRejectedBridgePointCount++;
                RejectedBridgePointCount++;
                break;
            }
        }
    }

    private static float SegmentGap(
        Vector3 a0,
        Vector3 a1,
        Vector3 b0,
        Vector3 b1,
        Vector3 axis)
    {
        float aMin = Mathf.Min(Vector3.Dot(a0, axis),
                               Vector3.Dot(a1, axis));
        float aMax = Mathf.Max(Vector3.Dot(a0, axis),
                               Vector3.Dot(a1, axis));
        float bMin = Mathf.Min(Vector3.Dot(b0, axis),
                               Vector3.Dot(b1, axis));
        float bMax = Mathf.Max(Vector3.Dot(b0, axis),
                               Vector3.Dot(b1, axis));
        if (bMin > aMax) return bMin - aMax;
        if (aMin > bMax) return aMin - bMax;
        return 0f;
    }

    private void CullTentativeTracks(float now)
    {
        for (int i = _tracks.Count - 1; i >= 0; i--)
            if (!_tracks[i].Confirmed &&
                now - _tracks[i].LastSeen > tentativeLifetimeSeconds)
                _tracks.RemoveAt(i);
    }

    private void RebuildMesh()
    {
        if (_mesh == null)
            return;
        _vertices.Clear();
        _colors.Clear();
        _triangles.Clear();
        for (int i = 0; i < _displayedTracks.Count; i++)
        {
            BoundaryTrack track = _displayedTracks[i];
            AppendCrossRibbon(track.A, track.B, confirmedColor);
        }
        _mesh.Clear(false);
        if (_vertices.Count > 0)
        {
            _mesh.SetVertices(_vertices);
            _mesh.SetColors(_colors);
            _mesh.SetTriangles(_triangles, 0, false);
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000f);
        }
        if (_renderer != null)
            _renderer.enabled = _visible && _vertices.Count > 0;
    }

    private void AppendCrossRibbon(Vector3 a, Vector3 b, Color color)
    {
        Vector3 direction = b - a;
        if (direction.sqrMagnitude < 0.000001f)
            return;
        direction.Normalize();
        Vector3 reference = Mathf.Abs(Vector3.Dot(direction, Vector3.up)) < 0.92f
            ? Vector3.up
            : Vector3.right;
        Vector3 sideA = Vector3.Cross(direction, reference).normalized *
                        lineHalfWidthMeters;
        Vector3 sideB = Vector3.Cross(direction, sideA).normalized *
                        lineHalfWidthMeters;
        AppendRibbon(a, b, sideA, color);
        AppendRibbon(a, b, sideB, color);
    }

    private void AppendRibbon(
        Vector3 a,
        Vector3 b,
        Vector3 side,
        Color color)
    {
        int first = _vertices.Count;
        _vertices.Add(a - side);
        _vertices.Add(a + side);
        _vertices.Add(b + side);
        _vertices.Add(b - side);
        for (int i = 0; i < 4; i++)
            _colors.Add(color);
        _triangles.Add(first);
        _triangles.Add(first + 1);
        _triangles.Add(first + 2);
        _triangles.Add(first);
        _triangles.Add(first + 2);
        _triangles.Add(first + 3);
    }

    private static bool Finite(Vector3 value)
    {
        return float.IsFinite(value.x) && float.IsFinite(value.y) &&
               float.IsFinite(value.z);
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_material != null)
            Destroy(_material);
        if (_renderObject != null)
            Destroy(_renderObject);
    }
}

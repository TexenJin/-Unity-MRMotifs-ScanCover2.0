using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Quest-side, post-freeze room-shell generation.  It reads the complete
/// non-indexed vertex buffer selected by the production draw path, then runs
/// the same multi-candidate evidence fitting used by the offline generic
/// extractor.  It never asks MRUK/Room Setup for a room model.
/// </summary>
public sealed partial class RoomSetupReplicaV2StructuralExtractor
{
    private const int RuntimeMaximumRasterDimension = 192;
    private const float RuntimeMinimumRoomHeight = 1.55f;

    [Header("Post-freeze simplified room")]
    [SerializeField, Range(0.04f, 0.10f)]
    private float runtimeFootprintCellSize = 0.05f;
    [SerializeField, Range(0.08f, 0.35f)]
    private float runtimeFootprintCloseDistance = 0.15f;
    [SerializeField, Range(0.06f, 0.30f)]
    private float runtimeWallLineMergeDistance = 0.12f;
    [SerializeField, Range(0.05f, 0.40f)]
    private float runtimeWallSnapDistance = 0.22f;
    [SerializeField]
    private bool runtimeUseBlindContourStack = true;
    [SerializeField]
    private bool runtimeRequireEndToEndBlindRoom = true;
    [SerializeField]
    private bool runtimeRegisterBlindWallSurfaces = true;
    [SerializeField]
    private bool runtimeComputeRoomInBackground = true;
    [SerializeField, Range(1, 4)]
    private int runtimeGraphCutWorkerCount = 2;
    [SerializeField, Range(0.02f, 0.08f)]
    private float runtimeSurfaceFoldAgreement = 0.04f;
    [SerializeField, Range(0.05f, 0.20f)]
    private float runtimeSurfaceMaximumShift = 0.15f;
    [SerializeField] private Color runtimeRoomColor =
        new Color(0.05f, 0.82f, 1f, 0.10f);
    [SerializeField] private Color runtimeFloorColor =
        new Color(0.06f, 0.72f, 0.96f, 0.08f);
    [SerializeField] private Color runtimeCeilingColor =
        new Color(0.64f, 0.42f, 1f, 0.08f);
    [SerializeField] private Color runtimeRoomEdgeColor =
        new Color(0.02f, 0.95f, 1f, 0.95f);

    private bool _runtimeRoomBuildPending;
    private CancellationTokenSource _runtimeRoomComputationCancellation;
    private int _runtimeRoomComputationGeneration;
    private Transform _runtimeRoomRoot;
    private Material _runtimeWallMaterial;
    private Material _runtimeFloorMaterial;
    private Material _runtimeCeilingMaterial;
    private Material _runtimeRoomEdgeMaterial;
    private readonly List<Vector3> _runtimeRoomOutline = new();
    private Vector3 _runtimeRoomAxisA = Vector3.right;
    private Vector3 _runtimeRoomAxisB = Vector3.forward;

    private sealed class RuntimeRoomComputation
    {
        public bool success;
        public string issue;
        public List<Vector2> polygon;
        public float floor;
        public float ceiling;
        public Vector3 axisA;
        public Vector3 axisB;
        public int sourceTriangleCount;
        public int candidateCount;
        public int appliedFeatureCount;
        public int strongEdges;
        public int moderateEdges;
        public int weakEdges;
        public string method;
        public string parameter;
        public float selectionScore;
        public float maskIoU;
        public long computeMilliseconds;
    }

    public string SimplifiedRoomStatus { get; private set; } = "not built";
    public int SimplifiedRoomWallCount { get; private set; }
    public int SimplifiedRoomAttachedPierCount { get; private set; }
    public int SimplifiedRoomSourceTriangleCount { get; private set; }
    public int SimplifiedRoomCandidateCount { get; private set; }
    public string SimplifiedRoomMethod { get; private set; } = "";
    public string SimplifiedRoomParameter { get; private set; } = "";
    public float SimplifiedRoomSelectionScore { get; private set; }
    public float SimplifiedRoomMaskIoU { get; private set; }
    public float SimplifiedRoomFloorHeight { get; private set; }
    public float SimplifiedRoomCeilingHeight { get; private set; }
    public IReadOnlyList<Vector3> SimplifiedRoomOutline =>
        _runtimeRoomOutline;

    private sealed class RuntimeEvidenceGrid
    {
        public bool[,] cells;
        public float minU;
        public float minV;
        public float cellSize;
    }

    private struct RuntimeLineEvidence
    {
        public float coordinate;
        public float weight;
    }

    private struct RuntimeRawWallObservation
    {
        public bool constantU;
        public float coordinate;
        public float tangentMinimum;
        public float tangentMaximum;
        public float minimumHeight;
        public float maximumHeight;
        public float area;
    }

    private sealed class RuntimeRawWallCluster
    {
        public bool constantU;
        public float coordinate;
        public float weight;
        public readonly List<RuntimeRawWallObservation> observations = new();
    }

    private struct RuntimeRawWallFace
    {
        public Vector2 a;
        public Vector2 b;
        public float bottom;
        public float top;
        public float area;
        public int verticalBins;
    }

    private struct RuntimeBoundaryFeature
    {
        public int edgeIndex;
        public float startParameter;
        public float endParameter;
        public Vector2 host0;
        public Vector2 front0;
        public Vector2 front1;
        public Vector2 host1;
        public bool completeU;
        public float score;
    }

    private readonly struct RuntimeGridPoint : IEquatable<RuntimeGridPoint>
    {
        public readonly int x;
        public readonly int y;

        public RuntimeGridPoint(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        public bool Equals(RuntimeGridPoint other) =>
            x == other.x && y == other.y;

        public override bool Equals(object obj) =>
            obj is RuntimeGridPoint other && Equals(other);

        public override int GetHashCode() => (x * 397) ^ y;
    }

    private readonly struct RuntimeDirectedEdge :
        IEquatable<RuntimeDirectedEdge>
    {
        public readonly RuntimeGridPoint a;
        public readonly RuntimeGridPoint b;

        public RuntimeDirectedEdge(RuntimeGridPoint a, RuntimeGridPoint b)
        {
            this.a = a;
            this.b = b;
        }

        public bool Equals(RuntimeDirectedEdge other) =>
            a.Equals(other.a) && b.Equals(other.b);

        public override bool Equals(object obj) =>
            obj is RuntimeDirectedEdge other && Equals(other);

        public override int GetHashCode() => (a.GetHashCode() * 397) ^
                                             b.GetHashCode();
    }

    /// <summary>
    /// Called by the right-thumbstick freeze path. Structural samples still
    /// establish gravity-relative room axes and the floor/ceiling pair. The
    /// frozen full MC buffer is then read once for a 5 cm multi-height stack;
    /// it never becomes a second live reconstruction path.
    /// </summary>
    public bool RequestSimplifiedRoomGeneration()
    {
        if (_reconstruction == null)
            _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
        if (_reconstruction == null)
        {
            SimplifiedRoomStatus = "no reconstruction owner";
            return false;
        }

        if (_runtimeRoomBuildPending)
            return true;
        _runtimeRoomBuildPending = true;
        SimplifiedRoomStatus = "reading frozen V2 structural evidence";
        if (_reconstruction.TryRequestStructuralSamples(
                OnRuntimeStructuralSamplesReady))
            return true;

        _runtimeRoomBuildPending = false;
        SimplifiedRoomStatus = "GPU structural readback not ready";
        return false;
    }

    private void OnRuntimeStructuralSamplesReady(Vector4[] packed)
    {
        if (!_runtimeRoomBuildPending || this == null || !isActiveAndEnabled)
            return;
        if (packed == null || packed.Length < 4)
        {
            _runtimeRoomBuildPending = false;
            SimplifiedRoomStatus = "frozen structural evidence is empty";
            Debug.LogWarning("[RoomSetupReplicaV2] simplified room failed: " +
                             SimplifiedRoomStatus, this);
            return;
        }

        ParseSamples(packed);
        SimplifiedRoomSourceTriangleCount = _samples.Count;
        if (_samples.Count < 16)
        {
            _runtimeRoomBuildPending = false;
            SimplifiedRoomStatus =
                "frozen structural evidence has too few valid samples";
            return;
        }
        ExtractPlanes();
        if (_planes.Count == 0)
        {
            _runtimeRoomBuildPending = false;
            SimplifiedRoomStatus = "frozen mesh has no usable structure";
            return;
        }

        SimplifiedRoomStatus = "reading frozen V2 mesh for adaptive slice";
        if (_reconstruction.TryRequestMeshSnapshot(
                OnRuntimeMeshSnapshotReady))
            return;

        // A concurrent export may temporarily own the full-mesh readback.
        // End-to-end blind mode must never replace the missing current mesh
        // with the legacy floor/ceiling footprint.  Doing so would produce a
        // plausible room from a different algorithm and falsely label the
        // result as equivalent.  Keep the previous room absent and let the
        // user retry once the readback owner releases it.
        _runtimeRoomBuildPending = false;
        if (runtimeRequireEndToEndBlindRoom)
        {
            SimplifiedRoomStatus =
                "blind end-to-end waiting for frozen full-mesh readback; retry";
            Debug.LogWarning("[RoomSetupReplicaV2] simplified room not " +
                             "generated: " + SimplifiedRoomStatus, this);
            return;
        }
        if (!TryBuildRuntimeRoom(null, out string fallbackIssue))
        {
            SimplifiedRoomStatus = fallbackIssue;
            Debug.LogWarning("[RoomSetupReplicaV2] simplified room failed: " +
                             fallbackIssue, this);
            return;
        }
        SimplifiedRoomMethod = "fallback_no_full_mesh+" +
                               SimplifiedRoomMethod;
        SimplifiedRoomStatus += " fullMeshSlice=fallback-readback-busy";
        Debug.Log("[RoomSetupReplicaV2] simplified room generated on Quest; " +
                  SimplifiedRoomStatus, this);
    }

    private void OnRuntimeMeshSnapshotReady(
        Vector3[] vertices, float[] stability, uint[] counters)
    {
        if (!_runtimeRoomBuildPending || this == null || !isActiveAndEnabled)
            return;
        if (vertices == null || vertices.Length < 12)
        {
            _runtimeRoomBuildPending = false;
            if (runtimeRequireEndToEndBlindRoom)
            {
                SimplifiedRoomStatus =
                    "blind end-to-end requires a non-empty frozen full mesh";
                Debug.LogWarning("[RoomSetupReplicaV2] simplified room not " +
                                 "generated: " + SimplifiedRoomStatus, this);
                return;
            }
            if (!TryBuildRuntimeRoom(null, out string fallbackIssue))
            {
                SimplifiedRoomStatus = fallbackIssue;
                return;
            }
            SimplifiedRoomMethod = "fallback_empty_full_mesh+" +
                                   SimplifiedRoomMethod;
            SimplifiedRoomStatus += " fullMeshSlice=fallback-empty";
            return;
        }

        SimplifiedRoomSourceTriangleCount = vertices.Length / 3;
        if (runtimeComputeRoomInBackground)
        {
            BeginRuntimeRoomComputation(vertices);
            return;
        }

        _runtimeRoomBuildPending = false;
        if (!TryBuildRuntimeRoom(vertices, out string issue))
        {
            SimplifiedRoomStatus = issue;
            Debug.LogWarning("[RoomSetupReplicaV2] simplified room failed: " +
                             issue, this);
            return;
        }

        Debug.Log("[RoomSetupReplicaV2] simplified room generated on Quest; " +
                  SimplifiedRoomStatus, this);
    }

    public void ClearSimplifiedRoom()
    {
        CancelRuntimeRoomComputation();
        ClearSimplifiedRoomInternal(true);
    }

    private void ClearSimplifiedRoomInternal(bool resetPending)
    {
        if (resetPending)
            _runtimeRoomBuildPending = false;
        _runtimeRoomOutline.Clear();
        SimplifiedRoomWallCount = 0;
        SimplifiedRoomAttachedPierCount = 0;
        SimplifiedRoomSourceTriangleCount = 0;
        SimplifiedRoomCandidateCount = 0;
        SimplifiedRoomMethod = "";
        SimplifiedRoomParameter = "";
        SimplifiedRoomSelectionScore = 0f;
        SimplifiedRoomMaskIoU = 0f;
        SimplifiedRoomFloorHeight = 0f;
        SimplifiedRoomCeilingHeight = 0f;
        _runtimeRoomAxisA = Vector3.right;
        _runtimeRoomAxisB = Vector3.forward;
        SimplifiedRoomStatus = "not built";
        if (_runtimeRoomRoot != null)
        {
            MeshFilter oldFilter =
                _runtimeRoomRoot.GetComponent<MeshFilter>();
            if (oldFilter != null && oldFilter.sharedMesh != null)
                Destroy(oldFilter.sharedMesh);
            Destroy(_runtimeRoomRoot.gameObject);
            _runtimeRoomRoot = null;
        }
        ReleaseRuntimeMaterial(ref _runtimeWallMaterial);
        ReleaseRuntimeMaterial(ref _runtimeFloorMaterial);
        ReleaseRuntimeMaterial(ref _runtimeCeilingMaterial);
        if (_runtimeRoomEdgeMaterial != null)
        {
            Destroy(_runtimeRoomEdgeMaterial);
            _runtimeRoomEdgeMaterial = null;
        }
    }

    private void CancelRuntimeRoomComputation()
    {
        _runtimeRoomComputationGeneration++;
        CancellationTokenSource cancellation =
            _runtimeRoomComputationCancellation;
        _runtimeRoomComputationCancellation = null;
        if (cancellation == null)
            return;
        cancellation.Cancel();
        cancellation.Dispose();
    }

    private async void BeginRuntimeRoomComputation(Vector3[] frozenMesh)
    {
        CancelRuntimeRoomComputation();
        int generation = _runtimeRoomComputationGeneration;
        CancellationTokenSource cancellation = new();
        CancellationToken token = cancellation.Token;
        _runtimeRoomComputationCancellation = cancellation;
        int sourceTriangleCount = frozenMesh.Length / 3;

        // Unity objects are removed and recreated only on the main thread.
        // The frozen Vector3 array and structural evidence are read-only for
        // the lifetime of this request.
        ClearSimplifiedRoomInternal(false);
        SimplifiedRoomSourceTriangleCount = sourceTriangleCount;
        SimplifiedRoomStatus = "computing simplified room off main thread";

        RuntimeRoomComputation computation;
        try
        {
            computation = await Task.Run(
                () => ComputeRuntimeRoomData(
                    frozenMesh, sourceTriangleCount, token), token);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        catch (Exception exception)
        {
            if (generation != _runtimeRoomComputationGeneration ||
                token.IsCancellationRequested)
                return;
            _runtimeRoomBuildPending = false;
            SimplifiedRoomStatus =
                "background simplified-room computation failed: " +
                exception.Message;
            Debug.LogException(exception, this);
            return;
        }

        if (this == null || !isActiveAndEnabled ||
            generation != _runtimeRoomComputationGeneration ||
            token.IsCancellationRequested)
            return;

        if (ReferenceEquals(_runtimeRoomComputationCancellation,
                            cancellation))
        {
            _runtimeRoomComputationCancellation = null;
            cancellation.Dispose();
        }
        _runtimeRoomBuildPending = false;
        if (!computation.success)
        {
            SimplifiedRoomStatus = computation.issue;
            Debug.LogWarning("[RoomSetupReplicaV2] simplified room failed: " +
                             computation.issue, this);
            return;
        }
        if (!TryApplyRuntimeRoomData(computation, out string issue))
        {
            SimplifiedRoomStatus = issue;
            Debug.LogWarning("[RoomSetupReplicaV2] simplified room failed: " +
                             issue, this);
            return;
        }
        Debug.Log("[RoomSetupReplicaV2] simplified room generated on Quest; " +
                  SimplifiedRoomStatus, this);
    }

    private bool TryBuildRuntimeRoom(Vector3[] frozenMesh,
                                     out string issue)
    {
        int sourceTriangleCount = SimplifiedRoomSourceTriangleCount;
        ClearSimplifiedRoomInternal(true);
        SimplifiedRoomSourceTriangleCount = sourceTriangleCount;
        RuntimeRoomComputation computation = ComputeRuntimeRoomData(
            frozenMesh, sourceTriangleCount, CancellationToken.None);
        if (!computation.success)
        {
            issue = computation.issue;
            return false;
        }
        return TryApplyRuntimeRoomData(computation, out issue);
    }

    private RuntimeRoomComputation ComputeRuntimeRoomData(
        Vector3[] frozenMesh, int sourceTriangleCount,
        CancellationToken token)
    {
        System.Diagnostics.Stopwatch stopwatch =
            System.Diagnostics.Stopwatch.StartNew();
        RuntimeRoomComputation result = new()
        {
            sourceTriangleCount = sourceTriangleCount,
            issue = "simplified-room computation did not complete"
        };
        token.ThrowIfCancellationRequested();
        if (!TryResolveRuntimeEnvelope(out float floor,
                                       out float ceiling))
        {
            result.issue = "main floor/ceiling pair was not supported";
            return result;
        }

        RuntimeEvidenceGrid raster = null;
        float sliceMinimumHeight = 0f;
        float sliceMaximumHeight = 0f;
        float sliceScore = 0f;
        int validSliceLayers = 0;
        bool usedBlindContourStack = false;
        string blindContourIssue = "blind contour stack was not attempted";
        RuntimeFootprintSelection selection = null;
        if (runtimeUseBlindContourStack &&
            frozenMesh != null && frozenMesh.Length >= 12)
        {
            usedBlindContourStack = TryBuildRuntimeBlindContourFootprint(
                frozenMesh, floor, ceiling, token, out selection,
                out blindContourIssue);
        }
        token.ThrowIfCancellationRequested();
        if (runtimeRequireEndToEndBlindRoom && !usedBlindContourStack)
        {
            result.issue = !runtimeUseBlindContourStack
                ? "blind end-to-end is required but contour stack is disabled"
                : frozenMesh == null || frozenMesh.Length < 12
                ? "blind end-to-end requires the current frozen full mesh"
                : "blind end-to-end contour stage failed: " +
                  blindContourIssue;
            return result;
        }
        if (!usedBlindContourStack &&
            frozenMesh != null && frozenMesh.Length >= 12)
        {
            raster = BuildRuntimeMultiHeightSliceEvidence(
                frozenMesh, floor, ceiling,
                out sliceMinimumHeight, out sliceMaximumHeight,
                out sliceScore, out validSliceLayers);
        }
        bool usedMultiHeightSlice = !usedBlindContourStack && raster != null;
        if (!usedBlindContourStack && raster == null)
            raster = BuildRuntimeFootprintEvidence(floor, ceiling);
        if (!usedBlindContourStack && raster == null)
        {
            result.issue =
                "multi-height stack and floor/ceiling evidence were empty";
            return result;
        }

        if (!usedBlindContourStack)
        {
            bool[,] closed = CloseRuntimeMask(raster.cells,
                Mathf.Max(1, Mathf.CeilToInt(
                    runtimeFootprintCloseDistance / raster.cellSize)));
            closed = KeepLargestRuntimeComponent(closed);
            FillRuntimeMaskHoles(closed);

            List<float> uLines = BuildRuntimeCandidateLines(
                closed, raster, true, floor, ceiling);
            List<float> vLines = BuildRuntimeCandidateLines(
                closed, raster, false, floor, ceiling);
            if (uLines.Count < 2 || vLines.Count < 2)
            {
                result.issue = "not enough supported wall coordinates";
                return result;
            }

            if (!TrySelectRuntimeFootprint(closed, raster, uLines, vLines,
                    floor, ceiling, out selection,
                    out string selectionIssue))
            {
                result.issue = selectionIssue;
                return result;
            }
        }
        token.ThrowIfCancellationRequested();
        List<Vector2> polygon = new(selection.polygon);
        string method = usedBlindContourStack
            ? "fully_blind_end_to_end_current_mesh+" + selection.method
            : usedMultiHeightSlice
            ? "multi_height_5cm_stack+" + selection.method
            : "floor_ceiling_fallback+" + selection.method;
        string parameter = usedBlindContourStack
            ? selection.parameter
            : usedMultiHeightSlice
            ? $"heightRange={sliceMinimumHeight:F3}-" +
              $"{sliceMaximumHeight:F3},step=0.050," +
              $"stackScore={sliceScore:F3}," +
              $"validLayers={validSliceLayers};{selection.parameter}"
            : selection.parameter;
        CleanupRuntimePolygon(polygon);
        if (polygon.Count < 4 || Mathf.Abs(RuntimePolygonArea(polygon)) < 1f)
        {
            result.issue =
                "outer footprint was too small after simplification";
            return result;
        }
        if (RuntimePolygonArea(polygon) < 0f)
            polygon.Reverse();

        // The main candidate deliberately stays conservative and simple.
        // Reinsert only wall-attached architectural returns that retain one
        // vertically connected face through height.  This protects real
        // piers after footprint smoothing without allowing a low cabinet and
        // a hanging object to supplement one another in top projection.
        int appliedFeatureCount = 0;
        if (!usedBlindContourStack)
        {
            List<RuntimeRawWallFace> rawWallFaces =
                BuildRuntimeRawWallFaces(floor, ceiling);
            List<RuntimeBoundaryFeature> boundaryFeatures =
                FindRuntimeBoundaryFeatures(
                    polygon, rawWallFaces, floor, ceiling);
            polygon = ApplyRuntimeBoundaryFeatures(
                polygon, boundaryFeatures, out appliedFeatureCount);
        }
        token.ThrowIfCancellationRequested();
        CleanupRuntimePolygon(polygon);
        if (RuntimePolygonArea(polygon) < 0f)
            polygon.Reverse();

        // The contour stack decides topology.  A separate blind registration
        // pass may then translate each mother-wall line along its own normal,
        // but only when alternating 10 cm height bands independently select
        // the same final-mesh surface.  Rejected or ambiguous walls remain at
        // their contour-stack coordinate, so this stage cannot invent a new
        // recess, pier or wall count.
        if (usedBlindContourStack && runtimeRegisterBlindWallSurfaces &&
            frozenMesh != null && frozenMesh.Length >= 12)
        {
            bool registrationApplied = TryRegisterRuntimeBlindWallSurfaces(
                frozenMesh, polygon, floor, ceiling, token,
                out List<Vector2> registeredPolygon,
                out RuntimeSurfaceRegistrationStats registration);
            if (registrationApplied)
                polygon = registeredPolygon;
            method += "+cross_height_surface_registration";
            parameter +=
                $";surfaceRegistered={registration.acceptedEdges}/" +
                $"{registration.edgeCount}," +
                $"surfaceApplied={(registrationApplied ? 1 : 0)}," +
                $"medianShift={registration.medianShift:F3}," +
                $"maximumShift={registration.maximumShift:F3}," +
                $"foldAgreement={runtimeSurfaceFoldAgreement:F3}," +
                $"shiftLimit={runtimeSurfaceMaximumShift:F3}";
        }
        token.ThrowIfCancellationRequested();
        stopwatch.Stop();
        result.success = true;
        result.issue = "";
        result.polygon = new List<Vector2>(polygon);
        result.floor = floor;
        result.ceiling = ceiling;
        result.axisA = _lastAxisA;
        result.axisB = _lastAxisB;
        result.candidateCount = selection.candidateCount;
        result.appliedFeatureCount = appliedFeatureCount;
        result.strongEdges = selection.strongEdges;
        result.moderateEdges = selection.moderateEdges;
        result.weakEdges = selection.weakEdges;
        result.method = method;
        result.parameter = parameter;
        result.selectionScore = selection.score;
        result.maskIoU = selection.maskIoU;
        result.computeMilliseconds = stopwatch.ElapsedMilliseconds;
        return result;
    }

    private bool TryApplyRuntimeRoomData(RuntimeRoomComputation computation,
                                         out string issue)
    {
        if (computation == null || !computation.success ||
            computation.polygon == null)
        {
            issue = computation != null ? computation.issue :
                "simplified-room result was empty";
            return false;
        }
        List<Vector2> polygon = computation.polygon;
        float floor = computation.floor;
        float ceiling = computation.ceiling;
        SimplifiedRoomSourceTriangleCount = computation.sourceTriangleCount;
        SimplifiedRoomCandidateCount = computation.candidateCount;
        SimplifiedRoomMethod = computation.method;
        SimplifiedRoomParameter = computation.parameter;
        SimplifiedRoomSelectionScore = computation.selectionScore;
        SimplifiedRoomMaskIoU = computation.maskIoU;
        SimplifiedRoomFloorHeight = floor;
        SimplifiedRoomCeilingHeight = ceiling;
        _runtimeRoomAxisA = computation.axisA;
        _runtimeRoomAxisB = computation.axisB;

        Mesh roomMesh = BuildRuntimeRoomMesh(polygon, floor, ceiling);
        if (roomMesh == null || roomMesh.vertexCount == 0)
        {
            issue = "room shell triangulation failed";
            return false;
        }

        CreateRuntimeRoomObjects(roomMesh, polygon, floor, ceiling);
        _runtimeRoomOutline.Clear();
        for (int i = 0; i < polygon.Count; i++)
            _runtimeRoomOutline.Add(RuntimeWorld(polygon[i], floor));
        SimplifiedRoomWallCount = polygon.Count;
        SimplifiedRoomAttachedPierCount = computation.appliedFeatureCount;
        SimplifiedRoomStatus =
            $"ready method={SimplifiedRoomMethod} " +
            $"parameter={SimplifiedRoomParameter} candidates=" +
            $"{computation.candidateCount} samples=" +
            $"{SimplifiedRoomSourceTriangleCount} walls={polygon.Count} " +
            $"strong={computation.strongEdges} " +
            $"moderate={computation.moderateEdges} " +
            $"weak={computation.weakEdges} " +
            $"score={computation.selectionScore:F4} " +
            $"iou={computation.maskIoU:F4} " +
            $"boundaryFeatures={SimplifiedRoomAttachedPierCount} " +
            $"floor={floor:F2}m ceiling={ceiling:F2}m " +
            $"computeMs={computation.computeMilliseconds}";
        issue = "";
        return true;
    }

    private bool TryResolveRuntimeEnvelope(out float floor,
                                           out float ceiling)
    {
        floor = _lastFloorHeight;
        List<float> ceilingHeights = new();
        for (int i = 0; i < _planes.Count; i++)
        {
            if (_planes[i].kind == StructuralKind.Ceiling)
                ceilingHeights.Add(_planes[i].p0.y);
        }
        if (ceilingHeights.Count == 0)
        {
            ceiling = 0f;
            return false;
        }
        ceilingHeights.Sort();
        ceiling = ceilingHeights[ceilingHeights.Count / 2];
        return float.IsFinite(floor) && float.IsFinite(ceiling) &&
               ceiling - floor >= RuntimeMinimumRoomHeight;
    }

    private RuntimeEvidenceGrid BuildRuntimeMultiHeightSliceEvidence(
        Vector3[] vertices, float floor, float ceiling,
        out float usedMinimumHeight, out float usedMaximumHeight,
        out float stackScore, out int validLayerCount)
    {
        usedMinimumHeight = 0f;
        usedMaximumHeight = 0f;
        stackScore = 0f;
        validLayerCount = 0;
        RuntimeEvidenceGrid template = BuildRuntimeSliceGridTemplate(
            vertices, floor, ceiling);
        if (template == null)
            return null;

        const float layerStep = 0.05f;
        float roomHeight = ceiling - floor;
        float firstHeight = floor + 0.25f;
        float lastHeight = ceiling - Mathf.Max(0.08f, roomHeight * 0.03f);
        if (lastHeight < firstHeight + layerStep * 2f)
            return null;

        int layerCount = Mathf.FloorToInt(
            (lastHeight - firstHeight) / layerStep) + 1;
        int width = template.cells.GetLength(0);
        int height = template.cells.GetLength(1);
        bool[][,] rawLayers = new bool[layerCount][,];
        for (int layer = 0; layer < layerCount; layer++)
            rawLayers[layer] = new bool[width, height];

        // Bucket every vertical-ish triangle into every 5 cm plane that it
        // crosses. This is one mesh pass rather than one full pass per layer.
        int vertexCount = vertices.Length - vertices.Length % 3;
        for (int i = 0; i < vertexCount; i += 3)
        {
            Vector3 a = vertices[i];
            Vector3 b = vertices[i + 1];
            Vector3 c = vertices[i + 2];
            if (!RuntimeFinite(a) || !RuntimeFinite(b) || !RuntimeFinite(c))
                continue;
            Vector3 cross = Vector3.Cross(b - a, c - a);
            float magnitude = cross.magnitude;
            if (magnitude <= 0.000001f ||
                Mathf.Abs(cross.y) / magnitude > 0.78f)
                continue;

            float minimumY = Mathf.Min(a.y, Mathf.Min(b.y, c.y));
            float maximumY = Mathf.Max(a.y, Mathf.Max(b.y, c.y));
            int firstLayer = Mathf.Clamp(Mathf.CeilToInt(
                (minimumY - firstHeight - 0.0005f) / layerStep),
                0, layerCount - 1);
            int lastLayer = Mathf.Clamp(Mathf.FloorToInt(
                (maximumY - firstHeight + 0.0005f) / layerStep),
                0, layerCount - 1);
            if (lastLayer < firstLayer || maximumY < firstHeight ||
                minimumY > lastHeight)
                continue;
            for (int layer = firstLayer; layer <= lastLayer; layer++)
            {
                float sliceHeight = firstHeight + layer * layerStep;
                if (!TryRuntimeSliceTriangle(a, b, c, sliceHeight,
                        out Vector2 first, out Vector2 second))
                    continue;
                RasterRuntimeSegment(rawLayers[layer], template.minU,
                    template.minV, template.cellSize, first, second);
            }
        }

        int[,] totalSupport = new int[width, height];
        int[,] maximumRun = new int[width, height];
        int[,] currentRun = new int[width, height];
        bool[,] rawUnion = new bool[width, height];
        int firstValidLayer = -1;
        int lastValidLayer = -1;
        int closeRadius = Mathf.Max(1, Mathf.CeilToInt(
            runtimeFootprintCloseDistance / template.cellSize));
        for (int layer = 0; layer < layerCount; layer++)
        {
            bool[,] raw = rawLayers[layer];
            bool[,] valid = BuildRuntimeValidSliceLayer(
                raw, template, closeRadius, out int acceptedComponents);
            bool hasEvidence = acceptedComponents > 0;
            if (hasEvidence)
            {
                validLayerCount++;
                if (firstValidLayer < 0) firstValidLayer = layer;
                lastValidLayer = layer;
            }
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                rawUnion[x, y] |= raw[x, y];
                if (valid[x, y])
                {
                    totalSupport[x, y]++;
                    currentRun[x, y]++;
                    maximumRun[x, y] = Mathf.Max(
                        maximumRun[x, y], currentRun[x, y]);
                }
                else
                {
                    currentRun[x, y] = 0;
                }
            }
        }
        if (validLayerCount < 3 || firstValidLayer < 0)
            return null;

        // A wall/room region must recur through height. A single accidental
        // closed contour cannot enter the footprint, while lower annexes can
        // survive without reaching the ceiling.
        bool[,] persistent = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            persistent[x, y] = totalSupport[x, y] >= 3 &&
                               maximumRun[x, y] >= 2;

        int stackCloseRadius = Mathf.Max(closeRadius, Mathf.CeilToInt(
            0.20f / template.cellSize));
        bool[,] finalMask = CloseRuntimeMask(
            persistent, stackCloseRadius);
        // Component selection intentionally happens only after all heights
        // have voted, so a lower room section is not discarded per slice.
        finalMask = KeepLargestRuntimeComponent(finalMask);
        FillRuntimeMaskHoles(finalMask);
        int filledCells = CountRuntimeMaskCells(finalMask);
        float area = filledCells * template.cellSize * template.cellSize;
        float perimeter = RuntimeMaskPerimeter(finalMask, template.cellSize);
        float boundarySupport = RuntimeSliceBoundarySupport(
            finalMask, rawUnion, Mathf.Max(1, Mathf.CeilToInt(
                0.15f / template.cellSize)));
        if (filledCells == 0 || area < 1f || perimeter < 4f ||
            boundarySupport < 0.24f)
            return null;

        usedMinimumHeight = firstHeight + firstValidLayer * layerStep;
        usedMaximumHeight = firstHeight + lastValidLayer * layerStep;
        float layerCoverage = validLayerCount / (float)layerCount;
        stackScore = perimeter *
            (0.70f + 0.30f * boundarySupport) + 0.15f * area +
            2.50f * layerCoverage;
        return new RuntimeEvidenceGrid
        {
            cells = finalMask,
            minU = template.minU,
            minV = template.minV,
            cellSize = template.cellSize
        };
    }

    private bool[,] BuildRuntimeValidSliceLayer(
        bool[,] raw, RuntimeEvidenceGrid raster, int closeRadius,
        out int acceptedComponentCount)
    {
        acceptedComponentCount = 0;
        int width = raw.GetLength(0);
        int height = raw.GetLength(1);
        bool[,] accepted = new bool[width, height];
        if (CountRuntimeMaskCells(raw) < 12)
            return accepted;

        bool[,] closed = CloseRuntimeMask(raw, closeRadius);
        bool[,] visited = new bool[width, height];
        Queue<RuntimeGridPoint> queue = new();
        List<RuntimeGridPoint> component = new();
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };
        int supportRadius = Mathf.Max(1, Mathf.CeilToInt(
            0.12f / raster.cellSize));
        for (int sx = 0; sx < width; sx++)
        for (int sy = 0; sy < height; sy++)
        {
            if (!closed[sx, sy] || visited[sx, sy])
                continue;
            component.Clear();
            visited[sx, sy] = true;
            queue.Enqueue(new RuntimeGridPoint(sx, sy));
            while (queue.Count > 0)
            {
                RuntimeGridPoint point = queue.Dequeue();
                component.Add(point);
                for (int direction = 0; direction < 4; direction++)
                {
                    int x = point.x + dx[direction];
                    int y = point.y + dy[direction];
                    if (x < 0 || y < 0 || x >= width || y >= height ||
                        visited[x, y] || !closed[x, y])
                        continue;
                    visited[x, y] = true;
                    queue.Enqueue(new RuntimeGridPoint(x, y));
                }
            }
            if (component.Count < 8)
                continue;

            bool[,] candidate = new bool[width, height];
            for (int i = 0; i < component.Count; i++)
                candidate[component[i].x, component[i].y] = true;
            FillRuntimeMaskHoles(candidate);
            int filledCells = CountRuntimeMaskCells(candidate);
            float area = filledCells * raster.cellSize * raster.cellSize;
            float perimeter = RuntimeMaskPerimeter(
                candidate, raster.cellSize);
            float enclosureRatio = filledCells /
                (float)Mathf.Max(component.Count, 1);
            float boundarySupport = RuntimeSliceBoundarySupport(
                candidate, raw, supportRadius);
            if (area < 0.45f || perimeter < 2.5f ||
                enclosureRatio < 1.60f || boundarySupport < 0.28f)
                continue;

            acceptedComponentCount++;
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                accepted[x, y] |= candidate[x, y];
        }
        return accepted;
    }

    private RuntimeEvidenceGrid BuildRuntimeSliceGridTemplate(
        Vector3[] vertices, float floor, float ceiling)
    {
        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        int valid = 0;
        float yMinimum = floor - 0.20f;
        float yMaximum = ceiling + 0.20f;
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 point = vertices[i];
            if (!RuntimeFinite(point) || point.y < yMinimum ||
                point.y > yMaximum)
                continue;
            Vector2 uv = RuntimeUv(point);
            minU = Mathf.Min(minU, uv.x);
            maxU = Mathf.Max(maxU, uv.x);
            minV = Mathf.Min(minV, uv.y);
            maxV = Mathf.Max(maxV, uv.y);
            valid++;
        }
        if (valid < 12 || !float.IsFinite(minU) ||
            maxU - minU < 1f || maxV - minV < 1f)
            return null;

        float cell = Mathf.Max(0.05f, runtimeFootprintCellSize);
        float margin = Mathf.Max(0.45f, cell * 9f);
        minU = Mathf.Floor((minU - margin) / cell) * cell;
        minV = Mathf.Floor((minV - margin) / cell) * cell;
        maxU = Mathf.Ceil((maxU + margin) / cell) * cell;
        maxV = Mathf.Ceil((maxV + margin) / cell) * cell;
        int width = Mathf.RoundToInt((maxU - minU) / cell) + 1;
        int height = Mathf.RoundToInt((maxV - minV) / cell) + 1;
        int maximum = Mathf.Max(width, height);
        if (maximum > RuntimeMaximumRasterDimension)
        {
            cell *= maximum / (float)RuntimeMaximumRasterDimension;
            minU = Mathf.Floor(minU / cell) * cell;
            minV = Mathf.Floor(minV / cell) * cell;
            maxU = Mathf.Ceil(maxU / cell) * cell;
            maxV = Mathf.Ceil(maxV / cell) * cell;
            width = Mathf.RoundToInt((maxU - minU) / cell) + 1;
            height = Mathf.RoundToInt((maxV - minV) / cell) + 1;
        }
        width = Mathf.Clamp(width, 1, RuntimeMaximumRasterDimension);
        height = Mathf.Clamp(height, 1, RuntimeMaximumRasterDimension);
        return new RuntimeEvidenceGrid
        {
            cells = new bool[width, height],
            minU = minU,
            minV = minV,
            cellSize = cell
        };
    }

    private bool TryRuntimeSliceTriangle(
        Vector3 a, Vector3 b, Vector3 c, float height,
        out Vector2 first, out Vector2 second)
    {
        Vector2 p0 = default;
        Vector2 p1 = default;
        Vector2 p2 = default;
        int count = 0;
        CollectRuntimeSliceEdge(a, b, height,
            ref p0, ref p1, ref p2, ref count);
        CollectRuntimeSliceEdge(b, c, height,
            ref p0, ref p1, ref p2, ref count);
        CollectRuntimeSliceEdge(c, a, height,
            ref p0, ref p1, ref p2, ref count);
        first = default;
        second = default;
        if (count < 2)
            return false;
        first = p0;
        second = p1;
        if (count >= 3)
        {
            float d01 = (p1 - p0).sqrMagnitude;
            float d02 = (p2 - p0).sqrMagnitude;
            float d12 = (p2 - p1).sqrMagnitude;
            if (d02 >= d01 && d02 >= d12)
                second = p2;
            else if (d12 >= d01)
            {
                first = p1;
                second = p2;
            }
        }
        return (second - first).sqrMagnitude > 0.000004f;
    }

    private void CollectRuntimeSliceEdge(
        Vector3 a, Vector3 b, float height,
        ref Vector2 p0, ref Vector2 p1, ref Vector2 p2, ref int count)
    {
        const float epsilon = 0.0005f;
        float da = a.y - height;
        float db = b.y - height;
        if (Mathf.Abs(da) <= epsilon && Mathf.Abs(db) <= epsilon)
        {
            AddRuntimeSlicePoint(RuntimeUv(a),
                ref p0, ref p1, ref p2, ref count);
            AddRuntimeSlicePoint(RuntimeUv(b),
                ref p0, ref p1, ref p2, ref count);
            return;
        }
        if (Mathf.Abs(da) <= epsilon)
        {
            AddRuntimeSlicePoint(RuntimeUv(a),
                ref p0, ref p1, ref p2, ref count);
            return;
        }
        if (Mathf.Abs(db) <= epsilon)
        {
            AddRuntimeSlicePoint(RuntimeUv(b),
                ref p0, ref p1, ref p2, ref count);
            return;
        }
        if ((da < 0f) == (db < 0f))
            return;
        float t = Mathf.Clamp01(da / (da - db));
        AddRuntimeSlicePoint(RuntimeUv(Vector3.LerpUnclamped(a, b, t)),
            ref p0, ref p1, ref p2, ref count);
    }

    private static void AddRuntimeSlicePoint(
        Vector2 point, ref Vector2 p0, ref Vector2 p1,
        ref Vector2 p2, ref int count)
    {
        if ((count > 0 && (point - p0).sqrMagnitude < 0.000004f) ||
            (count > 1 && (point - p1).sqrMagnitude < 0.000004f) ||
            (count > 2 && (point - p2).sqrMagnitude < 0.000004f))
            return;
        if (count == 0) p0 = point;
        else if (count == 1) p1 = point;
        else if (count == 2) p2 = point;
        if (count < 3) count++;
    }

    private static void RasterRuntimeSegment(
        bool[,] cells, float minU, float minV, float cell,
        Vector2 a, Vector2 b)
    {
        float length = Vector2.Distance(a, b);
        int steps = Mathf.Max(1, Mathf.CeilToInt(
            length / Mathf.Max(cell * 0.45f, 0.005f)));
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        for (int i = 0; i <= steps; i++)
        {
            Vector2 point = Vector2.Lerp(a, b, i / (float)steps);
            int x = Mathf.RoundToInt((point.x - minU) / cell);
            int y = Mathf.RoundToInt((point.y - minV) / cell);
            if (x >= 0 && y >= 0 && x < width && y < height)
                cells[x, y] = true;
        }
    }

    private static int CountRuntimeMaskCells(bool[,] cells)
    {
        int count = 0;
        for (int x = 0; x < cells.GetLength(0); x++)
        for (int y = 0; y < cells.GetLength(1); y++)
            if (cells[x, y]) count++;
        return count;
    }

    private static float RuntimeMaskPerimeter(bool[,] cells, float cell)
    {
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        int exposedEdges = 0;
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            if (!cells[x, y]) continue;
            if (x == 0 || !cells[x - 1, y]) exposedEdges++;
            if (x == width - 1 || !cells[x + 1, y]) exposedEdges++;
            if (y == 0 || !cells[x, y - 1]) exposedEdges++;
            if (y == height - 1 || !cells[x, y + 1]) exposedEdges++;
        }
        return exposedEdges * cell;
    }

    private static float RuntimeSliceBoundarySupport(
        bool[,] filled, bool[,] raw, int radius)
    {
        int width = filled.GetLength(0);
        int height = filled.GetLength(1);
        int boundary = 0;
        int supported = 0;
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            if (!filled[x, y]) continue;
            bool isBoundary = x == 0 || y == 0 ||
                x == width - 1 || y == height - 1 ||
                !filled[Mathf.Max(0, x - 1), y] ||
                !filled[Mathf.Min(width - 1, x + 1), y] ||
                !filled[x, Mathf.Max(0, y - 1)] ||
                !filled[x, Mathf.Min(height - 1, y + 1)];
            if (!isBoundary) continue;
            boundary++;
            bool found = false;
            for (int dx = -radius; dx <= radius && !found; dx++)
            for (int dy = -radius; dy <= radius; dy++)
            {
                if (dx * dx + dy * dy > radius * radius)
                    continue;
                int qx = x + dx;
                int qy = y + dy;
                if (qx >= 0 && qy >= 0 && qx < width && qy < height &&
                    raw[qx, qy])
                {
                    found = true;
                    break;
                }
            }
            if (found) supported++;
        }
        return boundary > 0 ? supported / (float)boundary : 0f;
    }

    private static bool RuntimeFinite(Vector3 value) =>
        float.IsFinite(value.x) && float.IsFinite(value.y) &&
        float.IsFinite(value.z);

    private RuntimeEvidenceGrid BuildRuntimeFootprintEvidence(
        float floor, float ceiling)
    {
        float minU = float.PositiveInfinity;
        float maxU = float.NegativeInfinity;
        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;
        List<int> horizontal = new();
        HashSet<int> selectedHorizontalClusters = new();
        List<PlaneCluster> horizontalClusters = BuildClusters(_horizontal);
        if (horizontalClusters.Count > 0)
        {
            int floorIndex = 0;
            int ceilingIndex = 0;
            float floorDistance = float.PositiveInfinity;
            float ceilingDistance = float.PositiveInfinity;
            for (int i = 0; i < horizontalClusters.Count; i++)
            {
                float df = Mathf.Abs(horizontalClusters[i].coordinate - floor);
                if (df < floorDistance)
                {
                    floorDistance = df;
                    floorIndex = i;
                }
                float dc = Mathf.Abs(horizontalClusters[i].coordinate - ceiling);
                if (dc < ceilingDistance)
                {
                    ceilingDistance = dc;
                    ceilingIndex = i;
                }
            }
            selectedHorizontalClusters.UnionWith(
                horizontalClusters[floorIndex].sampleIndices);
            selectedHorizontalClusters.UnionWith(
                horizontalClusters[ceilingIndex].sampleIndices);
        }
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            bool selectedCluster = selectedHorizontalClusters.Contains(i);
            bool heightBand = Mathf.Abs(sample.normal.y) >= 0.90f &&
                Mathf.Min(Mathf.Abs(sample.center.y - floor),
                          Mathf.Abs(sample.center.y - ceiling)) <= 0.18f;
            if (!selectedCluster && !heightBand)
                continue;
            horizontal.Add(i);
            // Match generic_room_structure_extractor.py: the raster extent is
            // owned only by the selected floor/ceiling clusters.  Height-band
            // evidence may fill that raster but must not enlarge it.
            if (selectedCluster)
            {
                RuntimeExpandBounds(sample.a, ref minU, ref maxU,
                                    ref minV, ref maxV);
                RuntimeExpandBounds(sample.b, ref minU, ref maxU,
                                    ref minV, ref maxV);
                RuntimeExpandBounds(sample.c, ref minU, ref maxU,
                                    ref minV, ref maxV);
            }
        }
        if (horizontal.Count < 8 || !float.IsFinite(minU) ||
            maxU - minU < 1f || maxV - minV < 1f)
            return null;

        float cell = Mathf.Max(0.05f, runtimeFootprintCellSize);
        // Match the offline grid: morphology must have empty border space or
        // an outer wall touching the raster edge becomes an open fragment.
        float margin = Mathf.Max(0.45f, cell * 9f);
        // Match the offline raster coordinates exactly: grid samples live on
        // cell-spaced nodes, and the padded bounds are snapped to that grid.
        minU = Mathf.Floor((minU - margin) / cell) * cell;
        minV = Mathf.Floor((minV - margin) / cell) * cell;
        maxU = Mathf.Ceil((maxU + margin) / cell) * cell;
        maxV = Mathf.Ceil((maxV + margin) / cell) * cell;
        int width = Mathf.RoundToInt((maxU - minU) / cell) + 1;
        int height = Mathf.RoundToInt((maxV - minV) / cell) + 1;
        int maximum = Mathf.Max(width, height);
        if (maximum > RuntimeMaximumRasterDimension)
        {
            cell *= maximum / (float)RuntimeMaximumRasterDimension;
            minU = Mathf.Floor(minU / cell) * cell;
            minV = Mathf.Floor(minV / cell) * cell;
            maxU = Mathf.Ceil(maxU / cell) * cell;
            maxV = Mathf.Ceil(maxV / cell) * cell;
            width = Mathf.RoundToInt((maxU - minU) / cell) + 1;
            height = Mathf.RoundToInt((maxV - minV) / cell) + 1;
        }
        width = Mathf.Clamp(width, 1, RuntimeMaximumRasterDimension);
        height = Mathf.Clamp(height, 1, RuntimeMaximumRasterDimension);

        bool[,] cells = new bool[width, height];
        for (int i = 0; i < horizontal.Count; i++)
        {
            StructuralSample sample = _samples[horizontal[i]];
            RasterRuntimeTriangle(cells, minU, minV, cell,
                RuntimeUv(sample.a), RuntimeUv(sample.b),
                RuntimeUv(sample.c));
        }
        cells = DilateRuntimeMask(cells,
            Mathf.Max(1, Mathf.RoundToInt(0.08f / cell)));
        return new RuntimeEvidenceGrid
        {
            cells = cells,
            minU = minU,
            minV = minV,
            cellSize = cell
        };
    }

    private void RuntimeExpandBounds(Vector3 point,
        ref float minU, ref float maxU, ref float minV, ref float maxV)
    {
        Vector2 uv = RuntimeUv(point);
        minU = Mathf.Min(minU, uv.x);
        maxU = Mathf.Max(maxU, uv.x);
        minV = Mathf.Min(minV, uv.y);
        maxV = Mathf.Max(maxV, uv.y);
    }

    private Vector2 RuntimeUv(Vector3 point) =>
        new Vector2(Vector3.Dot(point, _lastAxisA),
                    Vector3.Dot(point, _lastAxisB));

    private Vector3 RuntimeWorld(Vector2 uv, float height) =>
        _lastAxisA * uv.x + _lastAxisB * uv.y + Vector3.up * height;

    private static void RasterRuntimeTriangle(bool[,] cells,
        float minU, float minV, float cell, Vector2 a, Vector2 b, Vector2 c)
    {
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        int x0 = Mathf.Clamp(Mathf.FloorToInt(
            (Mathf.Min(a.x, Mathf.Min(b.x, c.x)) - minU) / cell),
            0, width - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(
            (Mathf.Max(a.x, Mathf.Max(b.x, c.x)) - minU) / cell),
            0, width - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(
            (Mathf.Min(a.y, Mathf.Min(b.y, c.y)) - minV) / cell),
            0, height - 1);
        int y1 = Mathf.Clamp(Mathf.CeilToInt(
            (Mathf.Max(a.y, Mathf.Max(b.y, c.y)) - minV) / cell),
            0, height - 1);
        for (int x = x0; x <= x1; x++)
        for (int y = y0; y <= y1; y++)
        {
            Vector2 p = new Vector2(
                minU + x * cell,
                minV + y * cell);
            if (RuntimePointInTriangle(p, a, b, c))
                cells[x, y] = true;
        }
    }

    private static bool RuntimePointInTriangle(
        Vector2 p, Vector2 a, Vector2 b, Vector2 c)
    {
        float d1 = RuntimeCross(b - a, p - a);
        float d2 = RuntimeCross(c - b, p - b);
        float d3 = RuntimeCross(a - c, p - c);
        bool negative = d1 < 0f || d2 < 0f || d3 < 0f;
        bool positive = d1 > 0f || d2 > 0f || d3 > 0f;
        return !(negative && positive);
    }

    private static bool RuntimeTriangleTouchesCell(
        Vector2 a, Vector2 b, Vector2 c, Vector2 center, float radius)
    {
        return RuntimePointSegmentDistance(center, a, b) <= radius ||
               RuntimePointSegmentDistance(center, b, c) <= radius ||
               RuntimePointSegmentDistance(center, c, a) <= radius;
    }

    private static bool[,] CloseRuntimeMask(bool[,] source, int radius)
    {
        return ErodeRuntimeMask(DilateRuntimeMask(source, radius), radius);
    }

    private static bool[,] DilateRuntimeMask(bool[,] source, int radius)
    {
        int width = source.GetLength(0);
        int height = source.GetLength(1);
        bool[,] result = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            bool value = false;
            for (int dx = -radius; dx <= radius && !value; dx++)
            for (int dy = -radius; dy <= radius; dy++)
            {
                if (dx * dx + dy * dy > radius * radius)
                    continue;
                int qx = x + dx;
                int qy = y + dy;
                if (qx >= 0 && qy >= 0 && qx < width && qy < height &&
                    source[qx, qy])
                {
                    value = true;
                    break;
                }
            }
            result[x, y] = value;
        }
        return result;
    }

    private static bool[,] ErodeRuntimeMask(bool[,] source, int radius)
    {
        int width = source.GetLength(0);
        int height = source.GetLength(1);
        bool[,] result = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            bool value = true;
            for (int dx = -radius; dx <= radius && value; dx++)
            for (int dy = -radius; dy <= radius; dy++)
            {
                if (dx * dx + dy * dy > radius * radius)
                    continue;
                int qx = x + dx;
                int qy = y + dy;
                if (qx < 0 || qy < 0 || qx >= width || qy >= height ||
                    !source[qx, qy])
                {
                    value = false;
                    break;
                }
            }
            result[x, y] = value;
        }
        return result;
    }

    private static bool[,] KeepLargestRuntimeComponent(bool[,] source)
    {
        int width = source.GetLength(0);
        int height = source.GetLength(1);
        int[,] labels = new int[width, height];
        int nextLabel = 0;
        int bestLabel = 0;
        int bestSize = 0;
        Queue<RuntimeGridPoint> queue = new();
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };
        for (int sx = 0; sx < width; sx++)
        for (int sy = 0; sy < height; sy++)
        {
            if (!source[sx, sy] || labels[sx, sy] != 0)
                continue;
            nextLabel++;
            int size = 0;
            labels[sx, sy] = nextLabel;
            queue.Enqueue(new RuntimeGridPoint(sx, sy));
            while (queue.Count > 0)
            {
                RuntimeGridPoint p = queue.Dequeue();
                size++;
                for (int n = 0; n < 4; n++)
                {
                    int x = p.x + dx[n];
                    int y = p.y + dy[n];
                    if (x < 0 || y < 0 || x >= width || y >= height ||
                        !source[x, y] || labels[x, y] != 0)
                        continue;
                    labels[x, y] = nextLabel;
                    queue.Enqueue(new RuntimeGridPoint(x, y));
                }
            }
            if (size > bestSize)
            {
                bestSize = size;
                bestLabel = nextLabel;
            }
        }

        bool[,] result = new bool[width, height];
        if (bestLabel == 0)
            return result;
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            result[x, y] = labels[x, y] == bestLabel;
        return result;
    }

    private static void FillRuntimeMaskHoles(bool[,] cells)
    {
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        bool[,] outside = new bool[width, height];
        Queue<RuntimeGridPoint> queue = new();
        void Seed(int x, int y)
        {
            if (x < 0 || y < 0 || x >= width || y >= height ||
                cells[x, y] || outside[x, y])
                return;
            outside[x, y] = true;
            queue.Enqueue(new RuntimeGridPoint(x, y));
        }
        for (int x = 0; x < width; x++)
        {
            Seed(x, 0);
            Seed(x, height - 1);
        }
        for (int y = 0; y < height; y++)
        {
            Seed(0, y);
            Seed(width - 1, y);
        }
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };
        while (queue.Count > 0)
        {
            RuntimeGridPoint p = queue.Dequeue();
            for (int n = 0; n < 4; n++)
                Seed(p.x + dx[n], p.y + dy[n]);
        }
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            if (!outside[x, y]) cells[x, y] = true;
    }

    private List<float> BuildRuntimeCandidateLines(bool[,] mask,
        RuntimeEvidenceGrid raster, bool alongU,
        float floor, float ceiling)
    {
        // Strict counterpart of generic_room_structure_extractor.py
        // candidate_lines(): use raw wall-family cluster coordinates only.
        // Grounded/extended topology sheets must not shift these coordinates.
        List<PlaneCluster> rawClusters = BuildClusters(
            alongU ? _wallFamilyA : _wallFamilyB);
        List<float> coordinates = new();
        for (int i = 0; i < rawClusters.Count; i++)
            coordinates.Add(rawClusters[i].coordinate);

        if (!TryRuntimeMaskExtent(mask, raster, alongU,
                                  out float minimum, out float maximum))
            return new List<float>();
        float extentTolerance = Mathf.Max(0.08f,
            1.6f * raster.cellSize);
        if (!RuntimeCoordinateNear(
                coordinates, minimum, extentTolerance))
        {
            coordinates.Add(Mathf.Round(
                (minimum + 0.5f * raster.cellSize) /
                raster.cellSize) * raster.cellSize);
        }
        if (!RuntimeCoordinateNear(
                coordinates, maximum, extentTolerance))
        {
            coordinates.Add(Mathf.Round(
                (maximum - 0.5f * raster.cellSize) /
                raster.cellSize) * raster.cellSize);
        }

        List<float> result = new();
        for (int i = 0; i < coordinates.Count; i++)
        {
            float coordinate = coordinates[i];
            if (coordinate >= minimum - 0.15f &&
                coordinate <= maximum + 0.15f)
                result.Add(coordinate);
        }
        result.Sort();
        return result;
    }

    private static bool RuntimeCoordinateNear(
        List<float> coordinates, float value, float tolerance)
    {
        for (int i = 0; i < coordinates.Count; i++)
        {
            if (Mathf.Abs(coordinates[i] - value) <= tolerance)
                return true;
        }
        return false;
    }
    private static bool TryRuntimeMaskExtent(bool[,] mask,
        RuntimeEvidenceGrid raster, bool alongU,
        out float minimum, out float maximum)
    {
        minimum = float.PositiveInfinity;
        maximum = float.NegativeInfinity;
        int width = mask.GetLength(0);
        int height = mask.GetLength(1);
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            if (!mask[x, y]) continue;
            float center = alongU
                ? raster.minU + x * raster.cellSize
                : raster.minV + y * raster.cellSize;
            float lo = center - 0.5f * raster.cellSize;
            float hi = center + 0.5f * raster.cellSize;
            minimum = Mathf.Min(minimum, lo);
            maximum = Mathf.Max(maximum, hi);
        }
        return float.IsFinite(minimum) && float.IsFinite(maximum);
    }

    private static float SnapRuntimeExtent(float value,
        List<RuntimeLineEvidence> evidence, float maximumDistance)
    {
        float best = value;
        float distance = maximumDistance;
        for (int i = 0; i < evidence.Count; i++)
        {
            float d = Mathf.Abs(evidence[i].coordinate - value);
            if (d >= distance) continue;
            distance = d;
            best = evidence[i].coordinate;
        }
        return best;
    }

    private static void MergeRuntimeCoordinates(List<float> values,
                                                 float tolerance)
    {
        int index = 1;
        while (index < values.Count)
        {
            if (values[index] - values[index - 1] >= tolerance)
            {
                index++;
                continue;
            }
            values[index - 1] = (values[index - 1] + values[index]) * 0.5f;
            values.RemoveAt(index);
        }
    }

    private static bool[,] BuildRuntimeArrangement(bool[,] mask,
        RuntimeEvidenceGrid raster, List<float> uLines, List<float> vLines)
    {
        int nx = uLines.Count - 1;
        int ny = vLines.Count - 1;
        bool[,] cells = new bool[nx, ny];
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
            int x0 = Mathf.Clamp(Mathf.FloorToInt(
                (uLines[i] - raster.minU) / raster.cellSize),
                0, mask.GetLength(0) - 1);
            int x1 = Mathf.Clamp(Mathf.CeilToInt(
                (uLines[i + 1] - raster.minU) / raster.cellSize),
                x0 + 1, mask.GetLength(0));
            int y0 = Mathf.Clamp(Mathf.FloorToInt(
                (vLines[j] - raster.minV) / raster.cellSize),
                0, mask.GetLength(1) - 1);
            int y1 = Mathf.Clamp(Mathf.CeilToInt(
                (vLines[j + 1] - raster.minV) / raster.cellSize),
                y0 + 1, mask.GetLength(1));
            int count = 0;
            int total = 0;
            for (int x = x0; x < x1; x++)
            for (int y = y0; y < y1; y++)
            {
                total++;
                if (mask[x, y]) count++;
            }
            cells[i, j] = total > 0 &&
                          (count / (float)total >= 0.12f || count >= 3);
        }
        return cells;
    }

    private static List<Vector2> TraceRuntimeArrangementBoundary(
        bool[,] cells, List<float> xs, List<float> ys)
    {
        HashSet<RuntimeDirectedEdge> edges = new();
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            if (!cells[x, y]) continue;
            if (y == 0 || !cells[x, y - 1])
                edges.Add(new RuntimeDirectedEdge(
                    new RuntimeGridPoint(x, y),
                    new RuntimeGridPoint(x + 1, y)));
            if (x == width - 1 || !cells[x + 1, y])
                edges.Add(new RuntimeDirectedEdge(
                    new RuntimeGridPoint(x + 1, y),
                    new RuntimeGridPoint(x + 1, y + 1)));
            if (y == height - 1 || !cells[x, y + 1])
                edges.Add(new RuntimeDirectedEdge(
                    new RuntimeGridPoint(x + 1, y + 1),
                    new RuntimeGridPoint(x, y + 1)));
            if (x == 0 || !cells[x - 1, y])
                edges.Add(new RuntimeDirectedEdge(
                    new RuntimeGridPoint(x, y + 1),
                    new RuntimeGridPoint(x, y)));
        }

        List<Vector2> best = new();
        float bestArea = 0f;
        while (edges.Count > 0)
        {
            RuntimeDirectedEdge first = default;
            foreach (RuntimeDirectedEdge edge in edges)
            {
                first = edge;
                break;
            }
            List<RuntimeGridPoint> loop = new() { first.a };
            RuntimeDirectedEdge current = first;
            edges.Remove(current);
            int guard = 0;
            while (!current.b.Equals(first.a) && guard++ < 10000)
            {
                loop.Add(current.b);
                if (!TryTakeRuntimeNextEdge(current, edges,
                                            out RuntimeDirectedEdge next))
                    break;
                current = next;
            }
            if (!current.b.Equals(first.a) || loop.Count < 4)
                continue;
            List<Vector2> polygon = new(loop.Count);
            for (int i = 0; i < loop.Count; i++)
                polygon.Add(new Vector2(xs[loop[i].x], ys[loop[i].y]));
            float area = Mathf.Abs(RuntimePolygonArea(polygon));
            if (area > bestArea)
            {
                bestArea = area;
                best = polygon;
            }
        }
        return best;
    }

    private static bool TryTakeRuntimeNextEdge(
        RuntimeDirectedEdge current,
        HashSet<RuntimeDirectedEdge> remaining,
        out RuntimeDirectedEdge next)
    {
        RuntimeDirectedEdge fallback = default;
        bool found = false;
        float bestTurn = float.NegativeInfinity;
        Vector2 incoming = new Vector2(
            current.b.x - current.a.x, current.b.y - current.a.y);
        foreach (RuntimeDirectedEdge candidate in remaining)
        {
            if (!candidate.a.Equals(current.b)) continue;
            Vector2 outgoing = new Vector2(
                candidate.b.x - candidate.a.x,
                candidate.b.y - candidate.a.y);
            float turn = RuntimeCross(incoming, outgoing) * 2f +
                         Vector2.Dot(incoming, outgoing);
            if (found && turn <= bestTurn) continue;
            bestTurn = turn;
            fallback = candidate;
            found = true;
        }
        next = fallback;
        if (found) remaining.Remove(next);
        return found;
    }

    private static void CleanupRuntimePolygon(List<Vector2> polygon)
    {
        bool changed = true;
        while (changed && polygon.Count >= 4)
        {
            changed = false;
            for (int i = 0; i < polygon.Count; i++)
            {
                Vector2 previous = polygon[(i - 1 + polygon.Count) %
                                           polygon.Count];
                Vector2 current = polygon[i];
                Vector2 next = polygon[(i + 1) % polygon.Count];
                bool duplicate = Vector2.Distance(previous, current) < 0.03f;
                bool collinear =
                    (Mathf.Abs(previous.x - current.x) < 0.001f &&
                     Mathf.Abs(current.x - next.x) < 0.001f) ||
                    (Mathf.Abs(previous.y - current.y) < 0.001f &&
                     Mathf.Abs(current.y - next.y) < 0.001f);
                if (!duplicate && !collinear) continue;
                polygon.RemoveAt(i);
                changed = true;
                break;
            }
        }
    }

    private List<RuntimeRawWallFace> BuildRuntimeRawWallFaces(
        float floor, float ceiling)
    {
        List<RuntimeRawWallObservation> observations = new();
        float alignmentThreshold = Mathf.Cos(
            axisToleranceDegrees * DegreesToRadians);
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            Vector3 horizontalNormal = Vector3.ProjectOnPlane(
                sample.normal, Vector3.up);
            if (horizontalNormal.sqrMagnitude < 0.35f)
                continue;
            horizontalNormal.Normalize();
            float alignmentA = Mathf.Abs(Vector3.Dot(
                horizontalNormal, _lastAxisA));
            float alignmentB = Mathf.Abs(Vector3.Dot(
                horizontalNormal, _lastAxisB));
            if (Mathf.Max(alignmentA, alignmentB) < alignmentThreshold)
                continue;

            bool constantU = alignmentA >= alignmentB;
            Vector3 normalAxis = constantU ? _lastAxisA : _lastAxisB;
            Vector3 tangentAxis = constantU ? _lastAxisB : _lastAxisA;
            float ta = Vector3.Dot(sample.a, tangentAxis);
            float tb = Vector3.Dot(sample.b, tangentAxis);
            float tc = Vector3.Dot(sample.c, tangentAxis);
            observations.Add(new RuntimeRawWallObservation
            {
                constantU = constantU,
                coordinate = Vector3.Dot(sample.center, normalAxis),
                tangentMinimum = Mathf.Min(ta, Mathf.Min(tb, tc)),
                tangentMaximum = Mathf.Max(ta, Mathf.Max(tb, tc)),
                minimumHeight = Mathf.Min(sample.a.y,
                    Mathf.Min(sample.b.y, sample.c.y)),
                maximumHeight = Mathf.Max(sample.a.y,
                    Mathf.Max(sample.b.y, sample.c.y)),
                area = sample.area
            });
        }
        observations.Sort((a, b) =>
        {
            int family = a.constantU.CompareTo(b.constantU);
            return family != 0 ? family :
                a.coordinate.CompareTo(b.coordinate);
        });

        List<RuntimeRawWallCluster> clusters = new();
        const float planeClusterDistance = 0.07f;
        for (int i = 0; i < observations.Count; i++)
        {
            RuntimeRawWallObservation observation = observations[i];
            RuntimeRawWallCluster cluster = clusters.Count > 0
                ? clusters[clusters.Count - 1] : null;
            if (cluster == null ||
                cluster.constantU != observation.constantU ||
                Mathf.Abs(cluster.coordinate - observation.coordinate) >
                planeClusterDistance)
            {
                cluster = new RuntimeRawWallCluster
                {
                    constantU = observation.constantU,
                    coordinate = observation.coordinate,
                    weight = Mathf.Max(observation.area, 0.0001f)
                };
                clusters.Add(cluster);
            }
            else
            {
                float weight = Mathf.Max(observation.area, 0.0001f);
                cluster.coordinate =
                    (cluster.coordinate * cluster.weight +
                     observation.coordinate * weight) /
                    (cluster.weight + weight);
                cluster.weight += weight;
            }
            cluster.observations.Add(observation);
        }

        List<RuntimeRawWallFace> result = new();
        for (int clusterIndex = 0;
             clusterIndex < clusters.Count; clusterIndex++)
        {
            RuntimeRawWallCluster cluster = clusters[clusterIndex];
            cluster.observations.Sort((a, b) =>
                a.tangentMinimum.CompareTo(b.tangentMinimum));
            int start = 0;
            while (start < cluster.observations.Count)
            {
                int end = start + 1;
                float runMaximum =
                    cluster.observations[start].tangentMaximum;
                while (end < cluster.observations.Count &&
                       cluster.observations[end].tangentMinimum <=
                       runMaximum + 0.10f)
                {
                    runMaximum = Mathf.Max(runMaximum,
                        cluster.observations[end].tangentMaximum);
                    end++;
                }
                AddRuntimeRawWallFace(result, cluster,
                    start, end, floor, ceiling);
                start = end;
            }
        }
        return result;
    }

    private static void AddRuntimeRawWallFace(
        List<RuntimeRawWallFace> result,
        RuntimeRawWallCluster cluster, int start, int end,
        float floor, float ceiling)
    {
        const float heightBinSize = 0.05f;
        int heightBinCount = Mathf.Max(1,
            Mathf.CeilToInt((ceiling - floor) / heightBinSize));
        bool[] heightBins = new bool[heightBinCount];
        float tangentMinimum = float.PositiveInfinity;
        float tangentMaximum = float.NegativeInfinity;
        float minimumHeight = float.PositiveInfinity;
        float maximumHeight = float.NegativeInfinity;
        float area = 0f;
        float roomHeight = ceiling - floor;
        for (int i = start; i < end; i++)
        {
            RuntimeRawWallObservation observation =
                cluster.observations[i];
            tangentMinimum = Mathf.Min(tangentMinimum,
                observation.tangentMinimum);
            tangentMaximum = Mathf.Max(tangentMaximum,
                observation.tangentMaximum);
            minimumHeight = Mathf.Min(minimumHeight,
                observation.minimumHeight);
            maximumHeight = Mathf.Max(maximumHeight,
                observation.maximumHeight);
            area += observation.area;
            for (int bin = 0; bin < heightBinCount; bin++)
            {
                float binBottom = floor + bin * heightBinSize;
                float binTop = Mathf.Min(
                    ceiling, binBottom + heightBinSize);
                if (observation.maximumHeight >= binBottom + 0.01f &&
                    observation.minimumHeight <= binTop - 0.01f)
                    heightBins[bin] = true;
            }
        }
        int supportedBins = 0;
        int longestRun = 0;
        int currentRun = 0;
        int gapBudget = 1;
        int gapUsed = 0;
        for (int i = 0; i < heightBins.Length; i++)
        {
            if (heightBins[i])
            {
                supportedBins++;
                currentRun++;
                longestRun = Mathf.Max(longestRun, currentRun);
                gapUsed = 0;
            }
            else if (currentRun > 0 && gapUsed < gapBudget)
            {
                currentRun++;
                gapUsed++;
            }
            else
            {
                currentRun = 0;
                gapUsed = 0;
            }
        }
        float longestConnectedHeight = longestRun * heightBinSize;
        float length = tangentMaximum - tangentMinimum;
        if (length < 0.075f || area < 0.025f ||
            longestConnectedHeight < Mathf.Max(
                0.70f, roomHeight * 0.28f) ||
            supportedBins < Mathf.Max(6,
                Mathf.CeilToInt(longestRun * 0.72f)))
            return;

        Vector2 a = cluster.constantU
            ? new Vector2(cluster.coordinate, tangentMinimum)
            : new Vector2(tangentMinimum, cluster.coordinate);
        Vector2 b = cluster.constantU
            ? new Vector2(cluster.coordinate, tangentMaximum)
            : new Vector2(tangentMaximum, cluster.coordinate);
        result.Add(new RuntimeRawWallFace
        {
            a = a,
            b = b,
            bottom = minimumHeight,
            top = maximumHeight,
            area = area,
            verticalBins = Mathf.Clamp(Mathf.RoundToInt(
                6f * longestConnectedHeight /
                Mathf.Max(roomHeight, 0.0001f)), 0, 6)
        });
    }

    private List<RuntimeBoundaryFeature> FindRuntimeBoundaryFeatures(
        List<Vector2> polygon, List<RuntimeRawWallFace> faces,
        float floor, float ceiling)
    {
        List<RuntimeBoundaryFeature> result = new();
        if (polygon == null || polygon.Count < 4 || faces.Count == 0)
            return result;
        bool counterClockwise = RuntimePolygonArea(polygon) > 0f;
        const int maximumCandidates = 64;
        for (int edgeIndex = 0; edgeIndex < polygon.Count; edgeIndex++)
        {
            Vector2 hostA = polygon[edgeIndex];
            Vector2 hostB = polygon[(edgeIndex + 1) % polygon.Count];
            Vector2 hostDelta = hostB - hostA;
            float hostLength = hostDelta.magnitude;
            if (hostLength < 0.24f) continue;
            Vector2 tangent = hostDelta / hostLength;
            Vector2 inward = counterClockwise
                ? new Vector2(-tangent.y, tangent.x)
                : new Vector2(tangent.y, -tangent.x);

            for (int faceIndex = 0; faceIndex < faces.Count; faceIndex++)
            {
                RuntimeRawWallFace front = faces[faceIndex];
                Vector2 frontDelta = front.b - front.a;
                float frontLength = frontDelta.magnitude;
                // A narrow pier has a short front, while an offset wall step
                // can expose one very short return beside a long parallel
                // wall section.  Keep both here; the partial-step path below
                // requires the unsupported end to terminate at a host corner.
                if (frontLength < 0.075f ||
                    frontLength > Mathf.Min(3.50f, hostLength + 0.16f) ||
                    Mathf.Abs(Vector2.Dot(
                        frontDelta / frontLength, tangent)) < 0.94f)
                    continue;

                Vector2 front0 = front.a;
                Vector2 front1 = front.b;
                if (Vector2.Dot(front1 - front0, tangent) < 0f)
                {
                    Vector2 swap = front0;
                    front0 = front1;
                    front1 = swap;
                }
                float startParameter = Vector2.Dot(
                    front0 - hostA, tangent);
                float endParameter = Vector2.Dot(
                    front1 - hostA, tangent);
                if (startParameter < -0.08f ||
                    endParameter > hostLength + 0.08f ||
                    endParameter - startParameter < 0.075f)
                    continue;
                startParameter = Mathf.Clamp(startParameter, 0f, hostLength);
                endParameter = Mathf.Clamp(endParameter, 0f, hostLength);
                Vector2 host0 = hostA + tangent * startParameter;
                Vector2 host1 = hostA + tangent * endParameter;
                float depth0 = Vector2.Dot(front0 - host0, inward);
                float depth1 = Vector2.Dot(front1 - host1, inward);
                float depth = (depth0 + depth1) * 0.5f;
                if (depth < 0.06f || depth > 0.75f ||
                    Mathf.Abs(depth0 - depth1) > 0.10f ||
                    !RuntimePointInsidePolygon(
                        (front0 + front1) * 0.5f, polygon))
                    continue;

                bool hasReturn0 = TryFindRuntimeRawReturn(faces,
                    host0, front0, tangent, inward, out _);
                bool hasReturn1 = TryFindRuntimeRawReturn(faces,
                    host1, front1, tangent, inward, out _);
                bool completeU = hasReturn0 && hasReturn1 &&
                    frontLength <= 1.60f;
                bool reachesHostCorner = startParameter <= 0.12f ||
                    hostLength - endParameter <= 0.12f;
                bool partialNarrowU = hasReturn0 != hasReturn1 &&
                    (frontLength <= 0.48f || reachesHostCorner) &&
                    depth <= 0.45f &&
                    front.verticalBins >= 5;
                if (!completeU && !partialNarrowU)
                    continue;
                if (RuntimeHasFurnitureTopEvidence(
                        host0, front0, front1, host1, floor, ceiling))
                    continue;

                RuntimeBoundaryFeature feature = new RuntimeBoundaryFeature
                {
                    edgeIndex = edgeIndex,
                    startParameter = startParameter,
                    endParameter = endParameter,
                    host0 = host0,
                    front0 = front0,
                    front1 = front1,
                    host1 = host1,
                    completeU = completeU,
                    score = (completeU ? 2f : 1f) +
                            front.verticalBins * 0.15f +
                            Mathf.Min(front.area, 0.40f)
                };
                if (!RuntimeBoundaryFeatureAlreadyPresent(result, feature))
                    result.Add(feature);
                if (result.Count >= maximumCandidates)
                    return result;
            }
        }
        return result;
    }

    private static bool TryFindRuntimeRawReturn(
        List<RuntimeRawWallFace> faces,
        Vector2 hostPoint, Vector2 frontPoint,
        Vector2 tangent, Vector2 inward,
        out RuntimeRawWallFace best)
    {
        best = default;
        bool found = false;
        float bestError = float.PositiveInfinity;
        for (int i = 0; i < faces.Count; i++)
        {
            RuntimeRawWallFace face = faces[i];
            Vector2 delta = face.b - face.a;
            float length = delta.magnitude;
            if (length < 0.06f || face.verticalBins < 4)
                continue;
            Vector2 direction = delta / length;
            if (Mathf.Abs(Vector2.Dot(direction, inward)) < 0.90f ||
                Mathf.Abs(Vector2.Dot(direction, tangent)) > 0.44f)
                continue;
            float hostError = RuntimePointSegmentDistance(
                hostPoint, face.a, face.b);
            float frontError = RuntimePointSegmentDistance(
                frontPoint, face.a, face.b);
            float maximumError = Mathf.Max(hostError, frontError);
            float error = hostError + frontError;
            if (maximumError > 0.09f || error >= bestError)
                continue;
            bestError = error;
            best = face;
            found = true;
        }
        return found;
    }

    private bool RuntimeHasFurnitureTopEvidence(
        Vector2 host0, Vector2 front0, Vector2 front1, Vector2 host1,
        float floor, float ceiling)
    {
        List<Vector2> footprint = new()
        {
            host0, front0, front1, host1
        };
        float area = 0f;
        int count = 0;
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            if (Mathf.Abs(sample.normal.y) < 0.88f ||
                sample.center.y <= floor + 0.20f ||
                sample.center.y >= ceiling - 0.14f)
                continue;
            Vector2 center = RuntimeUv(sample.center);
            if (!RuntimePointInsidePolygon(center, footprint) &&
                RuntimeBoundaryDistance(center, footprint) > 0.04f)
                continue;
            area += sample.area;
            count++;
            if (count >= 2 && area >= 0.025f)
                return true;
        }
        return false;
    }

    private static bool RuntimeBoundaryFeatureAlreadyPresent(
        List<RuntimeBoundaryFeature> existing,
        RuntimeBoundaryFeature candidate)
    {
        Vector2 center = (candidate.front0 + candidate.front1) * 0.5f;
        for (int i = 0; i < existing.Count; i++)
        {
            RuntimeBoundaryFeature item = existing[i];
            if (item.edgeIndex != candidate.edgeIndex)
                continue;
            Vector2 itemCenter = (item.front0 + item.front1) * 0.5f;
            if (Vector2.Distance(center, itemCenter) < 0.12f)
                return true;
        }
        return false;
    }

    private static List<Vector2> ApplyRuntimeBoundaryFeatures(
        List<Vector2> polygon, List<RuntimeBoundaryFeature> features,
        out int appliedCount)
    {
        appliedCount = 0;
        if (features == null || features.Count == 0)
            return new List<Vector2>(polygon);

        features.Sort((a, b) => b.score.CompareTo(a.score));
        List<RuntimeBoundaryFeature> selected = new();
        for (int i = 0; i < features.Count; i++)
        {
            if (selected.Count >= 12)
                break;
            RuntimeBoundaryFeature candidate = features[i];
            bool overlaps = false;
            for (int j = 0; j < selected.Count; j++)
            {
                RuntimeBoundaryFeature item = selected[j];
                if (item.edgeIndex != candidate.edgeIndex)
                    continue;
                float overlap = Mathf.Min(item.endParameter,
                    candidate.endParameter) - Mathf.Max(item.startParameter,
                    candidate.startParameter);
                if (overlap > 0.04f)
                {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) selected.Add(candidate);
        }
        selected.Sort((a, b) =>
        {
            int edge = a.edgeIndex.CompareTo(b.edgeIndex);
            return edge != 0 ? edge :
                a.startParameter.CompareTo(b.startParameter);
        });

        List<Vector2> result = new();
        int featureIndex = 0;
        for (int edgeIndex = 0; edgeIndex < polygon.Count; edgeIndex++)
        {
            AddRuntimePolygonPoint(result, polygon[edgeIndex]);
            while (featureIndex < selected.Count &&
                   selected[featureIndex].edgeIndex == edgeIndex)
            {
                RuntimeBoundaryFeature feature = selected[featureIndex++];
                AddRuntimePolygonPoint(result, feature.host0);
                AddRuntimePolygonPoint(result, feature.front0);
                AddRuntimePolygonPoint(result, feature.front1);
                AddRuntimePolygonPoint(result, feature.host1);
                appliedCount++;
            }
        }
        if (result.Count > 1 &&
            Vector2.Distance(result[0], result[result.Count - 1]) < 0.02f)
            result.RemoveAt(result.Count - 1);

        float originalArea = Mathf.Abs(RuntimePolygonArea(polygon));
        float resultArea = Mathf.Abs(RuntimePolygonArea(result));
        if (result.Count < 4 || resultArea < originalArea * 0.55f ||
            resultArea > originalArea + 0.05f ||
            RuntimePolygonHasSelfIntersections(result))
        {
            appliedCount = 0;
            return new List<Vector2>(polygon);
        }
        return result;
    }

    private static void AddRuntimePolygonPoint(
        List<Vector2> polygon, Vector2 point)
    {
        if (polygon.Count == 0 ||
            Vector2.Distance(polygon[polygon.Count - 1], point) >= 0.02f)
            polygon.Add(point);
    }

    private static bool RuntimePolygonHasSelfIntersections(
        List<Vector2> polygon)
    {
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            for (int j = i + 1; j < polygon.Count; j++)
            {
                if (j == i || j == (i + 1) % polygon.Count ||
                    (j + 1) % polygon.Count == i)
                    continue;
                Vector2 c = polygon[j];
                Vector2 d = polygon[(j + 1) % polygon.Count];
                if (Vector2.Distance(a, c) < 0.015f ||
                    Vector2.Distance(a, d) < 0.015f ||
                    Vector2.Distance(b, c) < 0.015f ||
                    Vector2.Distance(b, d) < 0.015f)
                    continue;
                float abC = RuntimeCross(b - a, c - a);
                float abD = RuntimeCross(b - a, d - a);
                float cdA = RuntimeCross(d - c, a - c);
                float cdB = RuntimeCross(d - c, b - c);
                if (abC * abD < -0.000001f &&
                    cdA * cdB < -0.000001f)
                    return true;
            }
        }
        return false;
    }

    private static bool RuntimePointInsidePolygon(
        Vector2 point, List<Vector2> polygon)
    {
        bool inside = false;
        for (int i = 0, j = polygon.Count - 1;
             i < polygon.Count; j = i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[j];
            bool crosses = (a.y > point.y) != (b.y > point.y) &&
                point.x < (b.x - a.x) * (point.y - a.y) /
                (b.y - a.y) + a.x;
            if (crosses) inside = !inside;
        }
        return inside;
    }

    private Mesh BuildRuntimeRoomMesh(List<Vector2> polygon,
        float floor, float ceiling)
    {
        List<int> cap = TriangulateRuntimePolygon(polygon);
        if (cap.Count < 3) return null;
        List<Vector3> vertices = new();
        List<int> floorTriangles = new();
        List<int> ceilingTriangles = new();
        List<int> wallTriangles = new();
        int count = polygon.Count;
        for (int i = 0; i < count; i++)
            vertices.Add(RuntimeWorld(polygon[i], floor));
        for (int i = 0; i < count; i++)
            vertices.Add(RuntimeWorld(polygon[i], ceiling));
        for (int i = 0; i < cap.Count; i += 3)
        {
            floorTriangles.Add(cap[i + 2]);
            floorTriangles.Add(cap[i + 1]);
            floorTriangles.Add(cap[i]);
            ceilingTriangles.Add(count + cap[i]);
            ceilingTriangles.Add(count + cap[i + 1]);
            ceilingTriangles.Add(count + cap[i + 2]);
        }
        for (int i = 0; i < count; i++)
        {
            int next = (i + 1) % count;
            wallTriangles.Add(i);
            wallTriangles.Add(count + next);
            wallTriangles.Add(next);
            wallTriangles.Add(i);
            wallTriangles.Add(count + i);
            wallTriangles.Add(count + next);
        }
        Mesh mesh = new Mesh
        {
            name = "V2 Runtime Simplified Room",
            indexFormat = vertices.Count > 65535
                ? IndexFormat.UInt32 : IndexFormat.UInt16
        };
        mesh.SetVertices(vertices);
        mesh.subMeshCount = 3;
        mesh.SetTriangles(floorTriangles, 0, false);
        mesh.SetTriangles(ceilingTriangles, 1, false);
        mesh.SetTriangles(wallTriangles, 2, true);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }
    private static List<int> TriangulateRuntimePolygon(
        List<Vector2> polygon)
    {
        List<int> remaining = new();
        for (int i = 0; i < polygon.Count; i++) remaining.Add(i);
        List<int> triangles = new();
        int guard = 0;
        while (remaining.Count > 3 && guard++ < polygon.Count * polygon.Count)
        {
            bool clipped = false;
            for (int i = 0; i < remaining.Count; i++)
            {
                int previous = remaining[(i - 1 + remaining.Count) %
                                         remaining.Count];
                int current = remaining[i];
                int next = remaining[(i + 1) % remaining.Count];
                Vector2 a = polygon[previous];
                Vector2 b = polygon[current];
                Vector2 c = polygon[next];
                if (RuntimeCross(b - a, c - b) <= 0.0001f)
                    continue;
                bool contains = false;
                for (int q = 0; q < remaining.Count; q++)
                {
                    int candidate = remaining[q];
                    if (candidate == previous || candidate == current ||
                        candidate == next)
                        continue;
                    if (RuntimePointInTriangle(
                            polygon[candidate], a, b, c))
                    {
                        contains = true;
                        break;
                    }
                }
                if (contains) continue;
                triangles.Add(previous);
                triangles.Add(current);
                triangles.Add(next);
                remaining.RemoveAt(i);
                clipped = true;
                break;
            }
            if (!clipped) break;
        }
        if (remaining.Count == 3)
        {
            triangles.Add(remaining[0]);
            triangles.Add(remaining[1]);
            triangles.Add(remaining[2]);
        }
        return triangles;
    }

    private void CreateRuntimeRoomObjects(Mesh roomMesh,
        List<Vector2> polygon, float floor, float ceiling)
    {
        GameObject root = new GameObject("V2 Runtime Simplified Room");
        root.transform.SetParent(transform, false);
        _runtimeRoomRoot = root.transform;
        MeshFilter filter = root.AddComponent<MeshFilter>();
        filter.sharedMesh = roomMesh;
        MeshRenderer renderer = root.AddComponent<MeshRenderer>();
        renderer.sharedMaterials = new[]
        {
            CreateRuntimeRoomMaterial(ref _runtimeFloorMaterial,
                runtimeFloorColor, "Floor"),
            CreateRuntimeRoomMaterial(ref _runtimeCeilingMaterial,
                runtimeCeilingColor, "Ceiling"),
            CreateRuntimeRoomMaterial(ref _runtimeWallMaterial,
                runtimeRoomColor, "Walls")
        };
        MeshCollider collider = root.AddComponent<MeshCollider>();
        collider.sharedMesh = roomMesh;

        CreateRuntimeLoop("Floor outline", polygon, floor, true);
        CreateRuntimeLoop("Ceiling outline", polygon, ceiling, true);
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 p = polygon[i];
            CreateRuntimeLine("Corner " + i,
                new[] { RuntimeWorld(p, floor), RuntimeWorld(p, ceiling) });
        }
    }

    private Material CreateRuntimeRoomMaterial(ref Material material,
        Color color, string surfaceName)
    {
        if (material != null) return material;
        Shader shader = Resources.Load<Shader>(
                            "RoomSetupReplicaV2SimplifiedRoom") ??
                        Shader.Find(
                            "RoomSetupReplicaV2/SimplifiedRoomTransparent");
        if (shader == null)
            throw new InvalidOperationException(
                "Simplified-room transparent shader is unavailable.");
        material = new Material(shader)
        {
            name = "V2 Runtime Simplified Room " + surfaceName,
            renderQueue = 3000
        };
        material.SetColor("_BaseColor", color);
        return material;
    }

    private static void ReleaseRuntimeMaterial(ref Material material)
    {
        if (material == null) return;
        Destroy(material);
        material = null;
    }

    private Material CreateRuntimeEdgeMaterial()
    {
        if (_runtimeRoomEdgeMaterial != null)
            return _runtimeRoomEdgeMaterial;
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ??
                        Shader.Find("Unlit/Color") ??
                        Shader.Find("Sprites/Default");
        _runtimeRoomEdgeMaterial = new Material(shader)
        {
            name = "V2 Runtime Simplified Room Edge Material",
            color = runtimeRoomEdgeColor
        };
        if (_runtimeRoomEdgeMaterial.HasProperty("_BaseColor"))
            _runtimeRoomEdgeMaterial.SetColor(
                "_BaseColor", runtimeRoomEdgeColor);
        return _runtimeRoomEdgeMaterial;
    }

    private void CreateRuntimeLoop(string objectName,
        List<Vector2> polygon, float height, bool close)
    {
        Vector3[] points = new Vector3[polygon.Count];
        for (int i = 0; i < polygon.Count; i++)
            points[i] = RuntimeWorld(polygon[i], height);
        CreateRuntimeLine(objectName, points, close);
    }

    private void CreateRuntimeLine(string objectName, Vector3[] points,
                                   bool close = false)
    {
        GameObject lineObject = new GameObject(objectName);
        lineObject.transform.SetParent(_runtimeRoomRoot, false);
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.sharedMaterial = CreateRuntimeEdgeMaterial();
        line.useWorldSpace = true;
        line.loop = close;
        line.widthMultiplier = Mathf.Max(0.006f, outlineWidthMeters);
        line.positionCount = points.Length;
        line.SetPositions(points);
        line.startColor = runtimeRoomEdgeColor;
        line.endColor = runtimeRoomEdgeColor;
        line.numCornerVertices = 1;
        line.numCapVertices = 1;
    }

    private static float RuntimePolygonArea(List<Vector2> polygon)
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

    private static float RuntimeBoundaryDistance(
        Vector2 point, List<Vector2> polygon)
    {
        float best = float.PositiveInfinity;
        for (int i = 0; i < polygon.Count; i++)
            best = Mathf.Min(best, RuntimePointSegmentDistance(
                point, polygon[i], polygon[(i + 1) % polygon.Count]));
        return best;
    }

    private static float RuntimePointSegmentDistance(
        Vector2 point, Vector2 a, Vector2 b)
    {
        Vector2 delta = b - a;
        float denominator = delta.sqrMagnitude;
        float t = denominator > 0.000001f
            ? Mathf.Clamp01(Vector2.Dot(point - a, delta) / denominator)
            : 0f;
        return Vector2.Distance(point, a + delta * t);
    }

    private static float RuntimeCross(Vector2 a, Vector2 b) =>
        a.x * b.y - a.y * b.x;
}

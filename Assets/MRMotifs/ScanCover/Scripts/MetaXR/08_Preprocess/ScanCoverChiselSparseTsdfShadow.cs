using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Isolated CHISEL-route proof: the same admitted stereo observations are fused
/// into an independent spatially-hashed, chunked TSDF.  It never reads or writes
/// the production TSDF, extraction buffers, visibility, or publication state.
///
/// The managed adapter follows OpenChisel's depth-only ProjectionIntegrator:
/// voxel centres are projected into the paired depth image and metric signed
/// distances are fused in spatially hashed chunks. Quest-specific safeguards
/// add discontinuity rejection, paired-frame confirmed free-space correction,
/// bounded weights, and time slicing. It remains an isolated Unity adapter,
/// not a claim of native or bit-for-bit parity.
/// </summary>
[DefaultExecutionOrder(-20)]
[DisallowMultipleComponent]
public sealed class ScanCoverChiselSparseTsdfShadow : MonoBehaviour
{
    private const int ChunkSize = 16;
    private const int VoxelsPerChunk = ChunkSize * ChunkSize * ChunkSize;
    private const int MeshSampleSize = ChunkSize + 1;
    private const int MeshSampleCount =
        MeshSampleSize * MeshSampleSize * MeshSampleSize;

    private enum ObservationPreparationStage : byte
    {
        None,
        DecodeLeft,
        ContinuityLeft,
        EvidenceLeft,
        CandidatesLeft,
        FreeSpaceLeft,
        DecodeRight,
        ContinuityRight,
        EvidenceRight,
        CandidatesRight,
        FreeSpaceRight,
        Finalize,
        ExpireCandidates
    }

    private struct Voxel
    {
        public float distance;
        public float weight;
        public Vector3 direction;
        public float directionWeight;
        public int lastSupportPair;
        public int lastConflictPair;
        public int pendingConflictPair;
        public byte pendingConflictEyes;
        public byte freeSpaceConflictStreak;
    }

    private sealed class SurfaceCandidate
    {
        public int pendingPair;
        public byte pendingEyes;
        public Vector3 leftPositionSum;
        public Vector3 rightPositionSum;
        public Vector3 leftNormalSum;
        public Vector3 rightNormalSum;
        public int leftCount;
        public int rightCount;
        public int leftNormalCount;
        public int rightNormalCount;
        public int lastConfirmedPair;
        public int lastSeenPair;
        public byte confirmedPairs;
        public Vector3 referencePosition;
        public Vector3 referenceNormal;
        public Vector3 firstViewPosition;
        public bool promoted;
    }

    private readonly struct SurfaceCandidateKey : IEquatable<SurfaceCandidateKey>
    {
        public readonly Vector3Int cell;
        public readonly byte directionAxis;

        public SurfaceCandidateKey(Vector3Int cell, byte directionAxis)
        {
            this.cell = cell;
            this.directionAxis = directionAxis;
        }

        public bool Equals(SurfaceCandidateKey other)
        {
            return cell == other.cell && directionAxis == other.directionAxis;
        }

        public override bool Equals(object obj)
        {
            return obj is SurfaceCandidateKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (cell.GetHashCode() * 397) ^ directionAxis;
            }
        }
    }

    private readonly struct VoxelAddress : IEquatable<VoxelAddress>
    {
        public readonly Vector3Int chunkId;
        public readonly int localIndex;

        public VoxelAddress(Vector3Int chunkId, int localIndex)
        {
            this.chunkId = chunkId;
            this.localIndex = localIndex;
        }

        public bool Equals(VoxelAddress other)
        {
            return chunkId == other.chunkId && localIndex == other.localIndex;
        }

        public override bool Equals(object obj)
        {
            return obj is VoxelAddress other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (chunkId.GetHashCode() * 397) ^ localIndex;
            }
        }
    }

    private sealed class Chunk
    {
        public readonly Voxel[] voxels = new Voxel[VoxelsPerChunk];
        // Snapshot of the voxel state that the currently published meshes were
        // built from. Fusion can write a stable voxel many times without
        // changing visible topology; compare against this snapshot instead of
        // treating every TSDF write as mesh work.
        public float[] meshedDistances;
        public byte[] meshedObserved;
        public bool meshStateInitialized;
        public int lastTouchedFrame;
        public GameObject previewObject;
        public MeshFilter previewFilter;
        public Mesh previewMesh;
        public Mesh previewStagingMesh;
        public int previewTriangleCount;
    }

    [Header("Independent sparse TSDF")]
    [SerializeField] private bool routeEnabled = true;
    [SerializeField, Range(0.03f, 0.10f)] private float voxelSizeMeters = 0.05f;
    [SerializeField, Range(0.05f, 0.30f)] private float truncationMeters = 0.15f;
    [SerializeField, Range(0.1f, 3f)] private float captureIntervalSeconds = 0.5f;
    [SerializeField, Range(0.2f, 2f)] private float minimumDepthMeters = 0.30f;
    [SerializeField, Range(1f, 8f)] private float maximumDepthMeters = 6.0f;
    [SerializeField, Range(1f, 8f)] private float minimumMeshWeight = 2f;

    [Header("Depth discontinuity protection")]
    [SerializeField] private bool protectDepthDiscontinuities = true;
    [Tooltip("A neighbouring depth must agree by this absolute/relative tolerance to support fusion.")]
    [SerializeField, Range(0.02f, 0.30f)] private float depthEdgeAbsoluteMeters = 0.06f;
    [SerializeField, Range(0.005f, 0.15f)] private float depthEdgeRelativeFraction = 0.02f;
    [SerializeField, Range(1, 8)] private int minimumConsistentDepthNeighbours = 4;

    [Header("Write-time observation association")]
    [Tooltip("Keep the legacy wide candidate search, but only fuse evidence inside the true TSDF truncation band.")]
    [SerializeField] private bool strictSurfaceEvidenceBand = true;
    [Tooltip("Compare new evidence with an already mature voxel before it can change the TSDF.")]
    [SerializeField] private bool enableObservationAssociation = true;
    [SerializeField, Range(1f, 16f)] private float matureAssociationWeight = 4f;
    [SerializeField, Range(0.02f, 0.20f)] private float associationResidualMeters = 0.075f;
    [SerializeField] private bool useViewWeightedObservations = true;
    [SerializeField, Range(0.01f, 0.50f)] private float minimumObservationWeight = 0.05f;
    [SerializeField, Range(0.10f, 1f)] private float immatureObservationScale = 1.0f;

    [Header("Novel-surface confirmation")]
    [Tooltip("New space is quarantined until paired multi-view evidence is stable. Existing TSDF surfaces remain immediately updateable.")]
    [SerializeField] private bool confirmNovelSurfacesBeforeFusion = true;
    [SerializeField, Range(0.05f, 0.20f)] private float novelSurfaceCellMeters = 0.10f;
    [SerializeField, Range(2, 6)] private int novelSurfaceConfirmationPairs = 3;
    [SerializeField, Range(0.03f, 0.20f)] private float novelSurfacePositionToleranceMeters = 0.075f;
    [SerializeField, Range(0.02f, 0.30f)] private float novelSurfaceMinimumViewBaselineMeters = 0.08f;
    [SerializeField, Range(2, 20)] private int novelSurfaceExpiryPairs = 6;
    [Tooltip("Once an adjacent cell of the same plane direction is proven, let the connected frontier spread without requiring another camera translation.")]
    [SerializeField] private bool fastTrackMatureSurfaceFrontier = true;
    [SerializeField, Range(2, 4)] private int frontierConfirmationPairs = 2;
    [SerializeField, Range(1, 2)] private int frontierNeighbourRadiusCells = 1;

    [Header("Surface-direction ownership")]
    [Tooltip("Prevents perpendicular/opposite observations from being averaged into an already stable scalar TSDF surface.")]
    [SerializeField] private bool protectSurfaceDirection = true;
    [SerializeField, Range(20f, 85f)] private float maximumDirectionChangeDegrees = 55f;
    [SerializeField, Range(1f, 16f)] private float matureDirectionWeight = 3f;

    [Header("Bounded TSDF and confirmed free space")]
    [SerializeField] private bool enableConfirmedFreeSpace = true;
    [SerializeField, Range(2f, 64f)] private float maximumVoxelWeight = 16f;
    [Tooltip("A voxel this far in front of the observed surface is contradictory free space.")]
    [SerializeField, Range(0.03f, 0.30f)] private float freeSpaceConflictMeters = 0.10f;
    [Tooltip("Stereo-confirmed contradictions also repair the residual gap below the full free-space threshold. This closes the former 7.5-10 cm dead zone without accepting a single-frame carve.")]
    [SerializeField, Range(0.03f, 0.10f)] private float repairConflictMeters = 0.06f;
    [SerializeField, Range(2, 8)] private int freeSpaceConfirmationPairs = 3;
    [SerializeField, Range(0.25f, 4f)] private float freeSpaceCorrectionWeight = 2f;
    [Tooltip("After stereo-confirmed conflicts, carve the stale surface directly instead of slowly averaging it away.")]
    [SerializeField] private bool directConfirmedFreeSpaceCarving = true;
    [Tooltip("Only selects already allocated chunks along stable rays; it never allocates free-space chunks.")]
    [SerializeField, Range(2, 16)] private int freeSpaceRayStridePixels = 4;

    [Header("Time-sliced integration")]
    [Tooltip("Maximum CHSL projection-integration time spent on the main thread per rendered frame.")]
    [SerializeField, Range(0.5f, 12f)] private float integrationBudgetMilliseconds = 4f;
    [SerializeField, Range(1, 64)] private int maximumIntegrationChunksPerFrame = 24;

    [Header("Independent preview")]
    [Tooltip("Off by default so the bypass cannot disguise or replace production geometry.")]
    [SerializeField] private bool showPreview;
    [SerializeField] private Color previewColor = new Color(0.05f, 0.85f, 1f, 0.72f);
    [Tooltip("Upper bound on FIFO mesh work per frame. Dirty chunks are never dropped.")]
    [SerializeField, Range(1, 24)] private int previewChunksPerFrame = 8;
    [Tooltip("Main-thread Marching Cubes budget. At least one dirty chunk is completed so the queue cannot starve.")]
    [SerializeField, Range(0.5f, 12f)] private float previewMeshingBudgetMilliseconds = 4f;
    [Tooltip("Accumulated TSDF motion relative to the last published mesh required before unchanged-sign geometry is rebuilt. Topology and observation-threshold changes always rebuild immediately.")]
    [SerializeField, Range(0.001f, 0.025f)] private float previewGeometryChangeMeters = 0.005f;

    public int AcceptedStereoPairCount { get; private set; }
    public int DroppedBusyPairCount { get; private set; }
    public int ReadbackFailureCount { get; private set; }
    public int AllocationDropCount { get; private set; }
    public int ActiveChunkCount => _chunks.Count;
    public int WeightedVoxelCount { get; private set; }
    public int SurfaceVoxelCount { get; private set; }
    public int LastValidSampleCount { get; private set; }
    public int LastVoxelWriteCount { get; private set; }
    public int LastCandidateChunkCount { get; private set; }
    public int LastUpdatedChunkCount { get; private set; }
    public int LastDepthEdgeRejectedCount { get; private set; }
    public int LastFreeSpaceConflictCount { get; private set; }
    public int LastFreeSpaceCorrectionCount { get; private set; }
    public int LastAssociationRejectedCount { get; private set; }
    public int LastDirectionRejectedCount { get; private set; }
    public int LastNovelSurfaceHeldCount { get; private set; }
    public int LastNovelSurfacePromotedCount { get; private set; }
    public int LastNovelSurfaceOcclusionRejectedCount { get; private set; }
    public int ActiveNovelSurfaceCandidateCount => _surfaceCandidates.Count;
    public int EmptyChunkCollectionCount { get; private set; }
    public int LastPreviewTriangleCount { get; private set; }
    public int MeshedChunkCount { get; private set; }
    public int PendingMeshChunkCount => _queuedMeshChunks.Count;
    public float LastIntegrationMilliseconds { get; private set; }
    public float LastIntegrationTotalMilliseconds { get; private set; }
    public float LastMeshingMilliseconds { get; private set; }
    public bool PreviewVisible => showPreview;
    public bool HasPendingReadback =>
        _readbackPending || _observationPreparationPending || _integrationPending;
    public float MinimumDepthMeters => Mathf.Max(0.01f, minimumDepthMeters);
    public float MaximumDepthMeters => Mathf.Max(MinimumDepthMeters, maximumDepthMeters);

    public string CompactStatus =>
        $"CHSL pair={AcceptedStereoPairCount} block={ActiveChunkCount} " +
        $"vox={WeightedVoxelCount} surf={SurfaceVoxelCount} " +
        $"int={LastIntegrationTotalMilliseconds:0.0}/{LastIntegrationMilliseconds:0.0}ms\n" +
        $"CHSL guard edge={LastDepthEdgeRejectedCount} " +
        $"free={LastFreeSpaceCorrectionCount}/{LastFreeSpaceConflictCount} " +
        $"assoc={LastAssociationRejectedCount} dir={LastDirectionRejectedCount} " +
        $"new={LastNovelSurfacePromotedCount}/{LastNovelSurfaceHeldCount}/{ActiveNovelSurfaceCandidateCount} " +
        $"occ={LastNovelSurfaceOcclusionRejectedCount} " +
        $"proj={LastUpdatedChunkCount}/{LastCandidateChunkCount} " +
        $"gc={EmptyChunkCollectionCount} " +
        $"band={(strictSurfaceEvidenceBand ? "15cm" : "LEGACY")}";

    private readonly Dictionary<Vector3Int, Chunk> _chunks =
        new Dictionary<Vector3Int, Chunk>(256);
    private readonly HashSet<Vector3Int> _touchedChunks =
        new HashSet<Vector3Int>();
    // Bits 0/1/2 mean the changed source voxel lies on the chunk's minimum
    // X/Y/Z boundary. Bit 3 means this chunk itself changed. Only the relevant
    // negative-direction cell owners need invalidation.
    private readonly Dictionary<Vector3Int, byte> _touchedChunkBoundaryMasks =
        new Dictionary<Vector3Int, byte>(64);
    private readonly HashSet<Vector3Int> _queuedMeshChunks =
        new HashSet<Vector3Int>();
    private readonly Queue<Vector3Int> _meshChunkQueue =
        new Queue<Vector3Int>();
    private readonly HashSet<Vector3Int> _candidateChunks =
        new HashSet<Vector3Int>();
    private readonly List<Vector3Int> _leftPendingChunks =
        new List<Vector3Int>(256);
    private readonly List<Vector3Int> _rightPendingChunks =
        new List<Vector3Int>(256);
    private readonly List<VoxelAddress> _freeSpaceEvidenceVoxels =
        new List<VoxelAddress>(4096);
    private readonly HashSet<VoxelAddress> _freeSpaceEvidenceSet =
        new HashSet<VoxelAddress>();
    private readonly Dictionary<SurfaceCandidateKey, SurfaceCandidate> _surfaceCandidates =
        new Dictionary<SurfaceCandidateKey, SurfaceCandidate>(2048);
    private readonly HashSet<SurfaceCandidateKey> _promotedSurfaceCells =
        new HashSet<SurfaceCandidateKey>();
    private readonly HashSet<SurfaceCandidateKey> _surfaceCandidatesTouched =
        new HashSet<SurfaceCandidateKey>();
    private readonly List<SurfaceCandidateKey> _surfaceCandidateCleanupKeys =
        new List<SurfaceCandidateKey>(512);
    private readonly List<SurfaceCandidateKey> _surfaceCandidateFinalizeKeys =
        new List<SurfaceCandidateKey>(2048);
    private readonly HashSet<Vector3Int> _leftNovelOcclusionChecked =
        new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _rightNovelOcclusionChecked =
        new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _leftOccludedNovelCells =
        new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _rightOccludedNovelCells =
        new HashSet<Vector3Int>();
    private readonly Stack<Chunk> _emptyChunkPool = new Stack<Chunk>(64);
    private AsyncGPUReadbackRequest _leftPositionRequest;
    private AsyncGPUReadbackRequest _leftMetaRequest;
    private AsyncGPUReadbackRequest _rightPositionRequest;
    private AsyncGPUReadbackRequest _rightMetaRequest;
    private bool _readbackPending;
    private NativeArray<Color> _leftPendingPositions;
    private NativeArray<Color> _leftPendingMeta;
    private NativeArray<Color> _rightPendingPositions;
    private NativeArray<Color> _rightPendingMeta;
    private bool _observationPreparationPending;
    private ObservationPreparationStage _observationPreparationStage;
    private int _observationPreparationCursor;
    private int _resetGeneration;
    private int _pendingGeneration;
    private int _pendingFrameOrdinal;
    private int _pendingLeftWidth;
    private int _pendingLeftHeight;
    private int _pendingRightWidth;
    private int _pendingRightHeight;
    private Vector3 _pendingLeftEye;
    private Vector3 _pendingRightEye;
    private Matrix4x4 _pendingLeftWorldToDepthClip;
    private Matrix4x4 _pendingRightWorldToDepthClip;
    private float[] _leftObservedDepth;
    private float[] _rightObservedDepth;
    private float[] _leftObservationWeights;
    private float[] _rightObservationWeights;
    private byte[] _leftDepthContinuity;
    private byte[] _rightDepthContinuity;
    private Vector3[] _leftObservationNormals;
    private Vector3[] _rightObservationNormals;
    private byte[] _leftSurfaceAdmission;
    private byte[] _rightSurfaceAdmission;
    private bool _integrationPending;
    private int _integrationPairSequence;
    private int _leftPendingChunkIndex;
    private int _rightPendingChunkIndex;
    private int _leftPendingExpected;
    private int _rightPendingExpected;
    private Matrix4x4 _leftPendingWorldToClip;
    private Matrix4x4 _rightPendingWorldToClip;
    private double _integrationTotalMilliseconds;
    private float _nextCaptureTime;
    private GameObject _previewObject;
    private Material _previewMaterial;
    // A 16^3 chunk owns 16^3 cells but samples a 17^3 voxel halo. Cache that
    // halo once, instead of performing eight dictionary lookups for every cell.
    // These buffers are reused because preview meshing is main-thread serial.
    private readonly Chunk[] _meshNeighbourChunks = new Chunk[8];
    private readonly float[] _meshSampleDistances =
        new float[MeshSampleCount];
    private readonly byte[] _meshSampleObserved =
        new byte[MeshSampleCount];
    private readonly List<Vector3> _meshVertices =
        new List<Vector3>(4096);
    private readonly List<int> _meshLineIndices =
        new List<int>(8192);
    // Marching Cubes cells share grid edges. Cache the emitted vertex for each
    // local X/Y/Z edge so adjacent cells reference the same zero crossing
    // instead of uploading three private vertices for every triangle.
    private readonly int[] _meshXEdgeVertexIndices =
        new int[ChunkSize * MeshSampleSize * MeshSampleSize];
    private readonly int[] _meshYEdgeVertexIndices =
        new int[MeshSampleSize * ChunkSize * MeshSampleSize];
    private readonly int[] _meshZEdgeVertexIndices =
        new int[MeshSampleSize * MeshSampleSize * ChunkSize];
    private readonly int[] _meshCellEdgeVertexIndices = new int[12];
    // The preview is a line mesh. Two neighbouring triangles otherwise submit
    // their shared edge twice, increasing both index traffic and overdraw.
    private readonly HashSet<ulong> _meshUniqueLineKeys =
        new HashSet<ulong>(8192);
    private readonly Vector3[] _meshCornerPositions = new Vector3[8];
    private readonly float[] _meshCornerDistances = new float[8];

    // OpenChisel MarchingCubes::edgeIndexPairs, kept in the same order.
    private static readonly int[,] EdgeIndexPairs =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };

    private static readonly Vector3Int[] CubeOffsets =
    {
        new Vector3Int(0, 0, 0), new Vector3Int(1, 0, 0),
        new Vector3Int(1, 1, 0), new Vector3Int(0, 1, 0),
        new Vector3Int(0, 0, 1), new Vector3Int(1, 0, 1),
        new Vector3Int(1, 1, 1), new Vector3Int(0, 1, 1)
    };

    public void ObserveAcceptedStereoPair(
        int frameOrdinal,
        RenderTexture leftWorldPosition,
        RenderTexture leftObservationMeta,
        Vector3 leftEye,
        Matrix4x4 leftWorldToDepthClip,
        RenderTexture rightWorldPosition,
        RenderTexture rightObservationMeta,
        Vector3 rightEye,
        Matrix4x4 rightWorldToDepthClip)
    {
        if (!routeEnabled || !isActiveAndEnabled)
            return;
        if (_readbackPending || _observationPreparationPending ||
            _integrationPending)
        {
            DroppedBusyPairCount++;
            return;
        }
        if (Time.unscaledTime < _nextCaptureTime)
            return;
        if (leftWorldPosition == null || leftObservationMeta == null ||
            rightWorldPosition == null || rightObservationMeta == null ||
            !Finite(leftEye) || !Finite(rightEye) ||
            !Finite(leftWorldToDepthClip) ||
            !Finite(rightWorldToDepthClip))
            return;

        _pendingFrameOrdinal = frameOrdinal;
        _pendingLeftWidth = leftWorldPosition.width;
        _pendingLeftHeight = leftWorldPosition.height;
        _pendingRightWidth = rightWorldPosition.width;
        _pendingRightHeight = rightWorldPosition.height;
        _pendingLeftEye = leftEye;
        _pendingRightEye = rightEye;
        _pendingLeftWorldToDepthClip = leftWorldToDepthClip;
        _pendingRightWorldToDepthClip = rightWorldToDepthClip;
        _leftPositionRequest = AsyncGPUReadback.Request(leftWorldPosition);
        _leftMetaRequest = AsyncGPUReadback.Request(leftObservationMeta);
        _rightPositionRequest = AsyncGPUReadback.Request(rightWorldPosition);
        _rightMetaRequest = AsyncGPUReadback.Request(rightObservationMeta);
        _pendingGeneration = _resetGeneration;
        _readbackPending = true;
        _nextCaptureTime = Time.unscaledTime + Mathf.Max(0.1f, captureIntervalSeconds);
    }

    private void Update()
    {
        bool startedPreparationThisFrame = false;
        if (_readbackPending)
        {
            CompletePendingReadbackWhenReady();
            startedPreparationThisFrame =
                !_readbackPending && _observationPreparationPending;
        }

        if (_observationPreparationPending && !startedPreparationThisFrame)
            ProcessObservationPreparationSlice();
        else if (_integrationPending)
            ProcessIntegrationSlice();

        EnsurePreviewObjects();
        if (_previewObject != null)
            _previewObject.SetActive(showPreview);
        // Meshing is an independent bounded queue. Keeping it moving while the
        // next stereo pair is prepared/fused prevents visible publication from
        // stalling behind observation work.
        if (showPreview && _meshChunkQueue.Count > 0)
            RebuildPreviewMesh();
    }

    private void CompletePendingReadbackWhenReady()
    {
        if (!_leftPositionRequest.done || !_leftMetaRequest.done ||
            !_rightPositionRequest.done || !_rightMetaRequest.done)
            return;

        if (_pendingGeneration != _resetGeneration)
        {
            _readbackPending = false;
            return;
        }
        if (_leftPositionRequest.hasError || _leftMetaRequest.hasError ||
            _rightPositionRequest.hasError || _rightMetaRequest.hasError)
        {
            _readbackPending = false;
            ReadbackFailureCount++;
            return;
        }

        double start = Time.realtimeSinceStartupAsDouble;
        NativeArray<Color> leftPositions =
            _leftPositionRequest.GetData<Color>();
        NativeArray<Color> leftMeta = _leftMetaRequest.GetData<Color>();
        NativeArray<Color> rightPositions =
            _rightPositionRequest.GetData<Color>();
        NativeArray<Color> rightMeta = _rightMetaRequest.GetData<Color>();
        EnsureColorCache(ref _leftPendingPositions, leftPositions.Length);
        EnsureColorCache(ref _leftPendingMeta, leftMeta.Length);
        EnsureColorCache(ref _rightPendingPositions, rightPositions.Length);
        EnsureColorCache(ref _rightPendingMeta, rightMeta.Length);
        NativeArray<Color>.Copy(
            leftPositions, _leftPendingPositions, leftPositions.Length);
        NativeArray<Color>.Copy(
            leftMeta, _leftPendingMeta, leftMeta.Length);
        NativeArray<Color>.Copy(
            rightPositions, _rightPendingPositions, rightPositions.Length);
        NativeArray<Color>.Copy(
            rightMeta, _rightPendingMeta, rightMeta.Length);

        _leftPendingExpected = Mathf.Min(
            Mathf.Min(leftPositions.Length, leftMeta.Length),
            Mathf.Max(0, _pendingLeftWidth * _pendingLeftHeight));
        _rightPendingExpected = Mathf.Min(
            Mathf.Min(rightPositions.Length, rightMeta.Length),
            Mathf.Max(0, _pendingRightWidth * _pendingRightHeight));
        _leftPendingWorldToClip = _pendingLeftWorldToDepthClip;
        _rightPendingWorldToClip = _pendingRightWorldToDepthClip;
        EnsureDepthCache(ref _leftObservedDepth, _leftPendingExpected);
        EnsureDepthCache(ref _leftObservationWeights, _leftPendingExpected);
        EnsureByteCache(ref _leftDepthContinuity, _leftPendingExpected);
        EnsureVectorCache(ref _leftObservationNormals, _leftPendingExpected);
        EnsureByteCache(ref _leftSurfaceAdmission, _leftPendingExpected);
        EnsureDepthCache(ref _rightObservedDepth, _rightPendingExpected);
        EnsureDepthCache(ref _rightObservationWeights, _rightPendingExpected);
        EnsureByteCache(ref _rightDepthContinuity, _rightPendingExpected);
        EnsureVectorCache(ref _rightObservationNormals, _rightPendingExpected);
        EnsureByteCache(ref _rightSurfaceAdmission, _rightPendingExpected);

        LastValidSampleCount = 0;
        LastVoxelWriteCount = 0;
        LastCandidateChunkCount = 0;
        LastUpdatedChunkCount = 0;
        LastDepthEdgeRejectedCount = 0;
        LastFreeSpaceConflictCount = 0;
        LastFreeSpaceCorrectionCount = 0;
        LastAssociationRejectedCount = 0;
        LastDirectionRejectedCount = 0;
        LastNovelSurfaceHeldCount = 0;
        LastNovelSurfacePromotedCount = 0;
        LastNovelSurfaceOcclusionRejectedCount = 0;
        LastIntegrationMilliseconds = 0f;
        LastIntegrationTotalMilliseconds = 0f;
        _touchedChunks.Clear();
        _touchedChunkBoundaryMasks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
        _surfaceCandidatesTouched.Clear();
        _leftNovelOcclusionChecked.Clear();
        _rightNovelOcclusionChecked.Clear();
        _leftOccludedNovelCells.Clear();
        _rightOccludedNovelCells.Clear();
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _integrationPairSequence++;
        _observationPreparationStage =
            ObservationPreparationStage.DecodeLeft;
        _observationPreparationCursor = 0;
        _integrationTotalMilliseconds =
            (Time.realtimeSinceStartupAsDouble - start) * 1000d;
        LastIntegrationMilliseconds = (float)_integrationTotalMilliseconds;
        _readbackPending = false;
        _observationPreparationPending = true;
        _integrationPending = false;
    }

    private void ProcessObservationPreparationSlice()
    {
        double start = Time.realtimeSinceStartupAsDouble;
        double deadline = start + Mathf.Max(
            0.0005f, integrationBudgetMilliseconds * 0.001f);

        while (_observationPreparationPending)
        {
            switch (_observationPreparationStage)
            {
                case ObservationPreparationStage.DecodeLeft:
                    if (PrepareDecodedObservationSlice(
                        _leftPendingPositions, _leftPendingMeta,
                        _leftPendingExpected, _leftPendingWorldToClip,
                        _leftObservedDepth, _leftObservationWeights,
                        _leftDepthContinuity, _leftObservationNormals,
                        _leftSurfaceAdmission, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.ContinuityLeft);
                    break;

                case ObservationPreparationStage.ContinuityLeft:
                    if (PrepareContinuitySlice(
                        _leftObservedDepth, _leftDepthContinuity,
                        _pendingLeftWidth, _pendingLeftHeight,
                        _leftPendingExpected, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.EvidenceLeft);
                    break;

                case ObservationPreparationStage.EvidenceLeft:
                    if (PrepareEvidenceSlice(
                        _leftPendingPositions, _leftObservedDepth,
                        _leftDepthContinuity, _pendingLeftWidth,
                        _pendingLeftHeight, _leftPendingExpected,
                        _pendingLeftEye, 1, _leftObservationNormals,
                        _leftSurfaceAdmission, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.CandidatesLeft);
                    break;

                case ObservationPreparationStage.CandidatesLeft:
                    if (PrepareCandidateChunksSlice(
                        _leftPendingPositions, _leftObservedDepth,
                        _leftDepthContinuity, _leftPendingExpected,
                        _pendingLeftEye, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.FreeSpaceLeft);
                    break;

                case ObservationPreparationStage.FreeSpaceLeft:
                    if (PrepareFreeSpaceChunksSlice(
                        _leftPendingPositions, _leftObservedDepth,
                        _leftDepthContinuity, _pendingLeftWidth,
                        _pendingLeftHeight, _leftPendingExpected,
                        _pendingLeftEye, deadline))
                    {
                        CommitPreparedEyeChunks(_leftPendingChunks);
                        BeginPreparationStage(
                            ObservationPreparationStage.DecodeRight);
                    }
                    break;

                case ObservationPreparationStage.DecodeRight:
                    if (PrepareDecodedObservationSlice(
                        _rightPendingPositions, _rightPendingMeta,
                        _rightPendingExpected, _rightPendingWorldToClip,
                        _rightObservedDepth, _rightObservationWeights,
                        _rightDepthContinuity, _rightObservationNormals,
                        _rightSurfaceAdmission, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.ContinuityRight);
                    break;

                case ObservationPreparationStage.ContinuityRight:
                    if (PrepareContinuitySlice(
                        _rightObservedDepth, _rightDepthContinuity,
                        _pendingRightWidth, _pendingRightHeight,
                        _rightPendingExpected, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.EvidenceRight);
                    break;

                case ObservationPreparationStage.EvidenceRight:
                    if (PrepareEvidenceSlice(
                        _rightPendingPositions, _rightObservedDepth,
                        _rightDepthContinuity, _pendingRightWidth,
                        _pendingRightHeight, _rightPendingExpected,
                        _pendingRightEye, 2, _rightObservationNormals,
                        _rightSurfaceAdmission, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.CandidatesRight);
                    break;

                case ObservationPreparationStage.CandidatesRight:
                    if (PrepareCandidateChunksSlice(
                        _rightPendingPositions, _rightObservedDepth,
                        _rightDepthContinuity, _rightPendingExpected,
                        _pendingRightEye, deadline))
                        BeginPreparationStage(
                            ObservationPreparationStage.FreeSpaceRight);
                    break;

                case ObservationPreparationStage.FreeSpaceRight:
                    if (PrepareFreeSpaceChunksSlice(
                        _rightPendingPositions, _rightObservedDepth,
                        _rightDepthContinuity, _pendingRightWidth,
                        _pendingRightHeight, _rightPendingExpected,
                        _pendingRightEye, deadline))
                    {
                        CommitPreparedEyeChunks(_rightPendingChunks);
                        _surfaceCandidateFinalizeKeys.Clear();
                        _surfaceCandidateFinalizeKeys.AddRange(
                            _surfaceCandidatesTouched);
                        _surfaceCandidateCleanupKeys.Clear();
                        BeginPreparationStage(
                            ObservationPreparationStage.Finalize);
                    }
                    break;

                case ObservationPreparationStage.Finalize:
                    if (PrepareFinalizeCandidatesSlice(
                        (_pendingLeftEye + _pendingRightEye) * 0.5f,
                        deadline))
                    {
                        RemovePreparedSurfaceCandidateKeys();
                        _surfaceCandidateFinalizeKeys.Clear();
                        _surfaceCandidateFinalizeKeys.AddRange(
                            _surfaceCandidates.Keys);
                        BeginPreparationStage(
                            ObservationPreparationStage.ExpireCandidates);
                    }
                    break;

                case ObservationPreparationStage.ExpireCandidates:
                    if (!PrepareExpiredCandidatesSlice(deadline))
                        break;
                    RemovePreparedSurfaceCandidateKeys();
                    _surfaceCandidateFinalizeKeys.Clear();
                    _leftPendingChunkIndex = 0;
                    _rightPendingChunkIndex = 0;
                    _observationPreparationStage =
                        ObservationPreparationStage.None;
                    _observationPreparationCursor = 0;
                    _observationPreparationPending = false;
                    _integrationPending = true;
                    break;

                default:
                    _observationPreparationPending = false;
                    break;
            }

            if (Time.realtimeSinceStartupAsDouble >= deadline)
                break;
        }

        float elapsed = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
        _integrationTotalMilliseconds += elapsed;
        LastIntegrationMilliseconds = Mathf.Max(
            LastIntegrationMilliseconds, elapsed);
    }

    private void BeginPreparationStage(ObservationPreparationStage stage)
    {
        _observationPreparationStage = stage;
        _observationPreparationCursor = 0;
        if (stage == ObservationPreparationStage.CandidatesLeft ||
            stage == ObservationPreparationStage.CandidatesRight)
            _candidateChunks.Clear();
    }

    private bool PrepareDecodedObservationSlice(
        NativeArray<Color> positions,
        NativeArray<Color> meta,
        int expected,
        Matrix4x4 worldToDepthClip,
        float[] observedDepths,
        float[] observationWeights,
        byte[] continuity,
        Vector3[] observationNormals,
        byte[] surfaceAdmission,
        double deadline)
    {
        while (_observationPreparationCursor < expected)
        {
            int index = _observationPreparationCursor++;
            observedDepths[index] = 0f;
            observationWeights[index] = 0f;
            continuity[index] = 0;
            observationNormals[index] = Vector3.zero;
            surfaceAdmission[index] = 0;
            if (meta[index].r >= 0.5f)
            {
                Color encoded = positions[index];
                Vector3 surface = new Vector3(
                    encoded.r, encoded.g, encoded.b);
                if (Finite(surface))
                {
                    float observedDepth = ProjectiveDepth(
                        worldToDepthClip, surface.x, surface.y, surface.z);
                    if (float.IsFinite(observedDepth) &&
                        observedDepth >= minimumDepthMeters &&
                        observedDepth <= maximumDepthMeters)
                    {
                        observedDepths[index] = observedDepth;
                        observationWeights[index] = Mathf.Clamp(
                            meta[index].g, minimumObservationWeight, 1f);
                        LastValidSampleCount++;
                    }
                }
            }

            if ((_observationPreparationCursor & 127) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool PrepareContinuitySlice(
        float[] observedDepths,
        byte[] continuity,
        int width,
        int height,
        int expected,
        double deadline)
    {
        int requiredNeighbours = Mathf.Clamp(
            minimumConsistentDepthNeighbours, 1, 8);
        while (_observationPreparationCursor < expected)
        {
            int index = _observationPreparationCursor++;
            continuity[index] = 0;
            int x = index % width;
            int y = index / width;
            if (y < height && observedDepths[index] > 0f)
            {
                if (!protectDepthDiscontinuities)
                {
                    continuity[index] = 1;
                }
                else
                {
                    float center = observedDepths[index];
                    float tolerance = Mathf.Max(
                        depthEdgeAbsoluteMeters,
                        center * depthEdgeRelativeFraction);
                    int consistent = 0;
                    for (int oy = -1; oy <= 1; oy++)
                    for (int ox = -1; ox <= 1; ox++)
                    {
                        if ((ox == 0 && oy == 0) ||
                            x + ox < 0 || x + ox >= width ||
                            y + oy < 0 || y + oy >= height)
                            continue;
                        int neighbourIndex =
                            x + ox + (y + oy) * width;
                        if (neighbourIndex < 0 ||
                            neighbourIndex >= expected)
                            continue;
                        float neighbour = observedDepths[neighbourIndex];
                        if (neighbour > 0f &&
                            Mathf.Abs(neighbour - center) <= tolerance)
                            consistent++;
                    }

                    if (consistent >= requiredNeighbours)
                        continuity[index] = 1;
                    else
                        LastDepthEdgeRejectedCount++;
                }
            }

            if ((_observationPreparationCursor & 63) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool PrepareEvidenceSlice(
        NativeArray<Color> positions,
        float[] observedDepths,
        byte[] continuity,
        int width,
        int height,
        int expected,
        Vector3 eye,
        byte eyeBit,
        Vector3[] observationNormals,
        byte[] surfaceAdmission,
        double deadline)
    {
        HashSet<Vector3Int> checkedCells = (eyeBit & 1) != 0
            ? _leftNovelOcclusionChecked
            : _rightNovelOcclusionChecked;
        HashSet<Vector3Int> occludedCells = (eyeBit & 1) != 0
            ? _leftOccludedNovelCells
            : _rightOccludedNovelCells;
        while (_observationPreparationCursor < expected)
        {
            int index = _observationPreparationCursor++;
            observationNormals[index] = Vector3.zero;
            surfaceAdmission[index] = 0;
            int x = index % width;
            int y = index / width;
            if (y < height && observedDepths[index] > 0f &&
                continuity[index] != 0)
            {
                Color encoded = positions[index];
                Vector3 surface = new Vector3(
                    encoded.r, encoded.g, encoded.b);
                if (Finite(surface))
                {
                    Vector3 normal = EstimateObservationNormal(
                        positions, observedDepths, continuity,
                        x, y, width, height, expected, surface, eye);
                    observationNormals[index] = normal;
                    if (!confirmNovelSurfacesBeforeFusion)
                    {
                        surfaceAdmission[index] = 1;
                    }
                    else
                    {
                        Vector3Int spatialKey = NovelSurfaceCell(surface);
                        SurfaceCandidateKey key =
                            NovelSurfaceKey(surface, normal);
                        if (checkedCells.Add(spatialKey) &&
                            HasMatureOccluderAlongRay(eye, surface))
                            occludedCells.Add(spatialKey);
                        if (occludedCells.Contains(spatialKey))
                        {
                            LastNovelSurfaceOcclusionRejectedCount++;
                        }
                        else if (IsNovelSurfacePromoted(key))
                        {
                            surfaceAdmission[index] = 1;
                        }
                        else
                        {
                            RecordNovelSurfaceCandidate(
                                key, surface, normal, eyeBit);
                        }
                    }
                }
            }

            // One new spatial cell may perform a full occluder ray walk, so
            // check the budget more frequently than the arithmetic stages.
            if ((_observationPreparationCursor & 15) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool PrepareCandidateChunksSlice(
        NativeArray<Color> positions,
        float[] observedDepths,
        byte[] continuity,
        int expected,
        Vector3 eye,
        double deadline)
    {
        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        float truncation = Mathf.Max(voxel, truncationMeters);
        float candidateBand = truncation + Mathf.Sqrt(3f) * voxel;
        float chunkWorldSize = ChunkSize * voxel;
        while (_observationPreparationCursor < expected)
        {
            int index = _observationPreparationCursor++;
            if (observedDepths[index] > 0f && continuity[index] != 0)
            {
                Color encoded = positions[index];
                Vector3 surface = new Vector3(
                    encoded.r, encoded.g, encoded.b);
                float rayScale = (surface - eye).magnitude /
                    Mathf.Max(0.0001f, observedDepths[index]);
                float worldRadius =
                    candidateBand * Mathf.Max(1f, rayScale) +
                    Mathf.Sqrt(3f) * voxel;
                AddCandidateChunkRange(
                    surface, worldRadius, chunkWorldSize);
            }

            if ((_observationPreparationCursor & 63) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool PrepareFreeSpaceChunksSlice(
        NativeArray<Color> positions,
        float[] observedDepths,
        byte[] continuity,
        int width,
        int height,
        int expected,
        Vector3 eye,
        double deadline)
    {
        if (!enableConfirmedFreeSpace)
            return true;

        int stride = Mathf.Max(2, freeSpaceRayStridePixels);
        int columns = (width + stride - 1) / stride;
        int rows = (height + stride - 1) / stride;
        int sampleCount = columns * rows;
        float chunkWorldSize = ChunkSize *
            Mathf.Max(0.005f, voxelSizeMeters);
        while (_observationPreparationCursor < sampleCount)
        {
            int sample = _observationPreparationCursor++;
            int x = (sample % columns) * stride;
            int y = (sample / columns) * stride;
            int index = x + y * width;
            if (index >= 0 && index < expected &&
                observedDepths[index] > 0f && continuity[index] != 0)
            {
                Color encoded = positions[index];
                Vector3 surface = new Vector3(
                    encoded.r, encoded.g, encoded.b);
                AddExistingFreeSpaceChunksAlongRay(
                    eye, surface, (surface - eye).magnitude,
                    chunkWorldSize);
            }

            if ((_observationPreparationCursor & 15) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private void CommitPreparedEyeChunks(List<Vector3Int> pendingChunks)
    {
        LastCandidateChunkCount += _candidateChunks.Count;
        pendingChunks.AddRange(_candidateChunks);
        _candidateChunks.Clear();
    }

    private bool PrepareFinalizeCandidatesSlice(
        Vector3 viewPosition, double deadline)
    {
        if (!confirmNovelSurfacesBeforeFusion)
        {
            _observationPreparationCursor =
                _surfaceCandidateFinalizeKeys.Count;
            return true;
        }

        float positionTolerance = Mathf.Max(
            Mathf.Max(0.005f, voxelSizeMeters),
            novelSurfacePositionToleranceMeters);
        float normalAlignment = Mathf.Cos(
            Mathf.Clamp(maximumDirectionChangeDegrees, 20f, 85f) *
            Mathf.Deg2Rad);
        while (_observationPreparationCursor <
               _surfaceCandidateFinalizeKeys.Count)
        {
            SurfaceCandidateKey key = _surfaceCandidateFinalizeKeys[
                _observationPreparationCursor++];
            if (!_surfaceCandidates.TryGetValue(
                    key, out SurfaceCandidate candidate) ||
                candidate.promoted)
            {
                if ((_observationPreparationCursor & 31) == 0 &&
                    Time.realtimeSinceStartupAsDouble >= deadline)
                    return false;
                continue;
            }

            bool stereoComplete =
                candidate.pendingPair == _integrationPairSequence &&
                (candidate.pendingEyes & 3) == 3 &&
                candidate.leftCount > 0 && candidate.rightCount > 0;
            if (!stereoComplete)
            {
                candidate.confirmedPairs = 0;
            }
            else
            {
                Vector3 leftPosition =
                    candidate.leftPositionSum / candidate.leftCount;
                Vector3 rightPosition =
                    candidate.rightPositionSum / candidate.rightCount;
                if (Vector3.Distance(leftPosition, rightPosition) >
                    positionTolerance)
                {
                    candidate.confirmedPairs = 0;
                }
                else
                {
                    Vector3 pairPosition =
                        (leftPosition + rightPosition) * 0.5f;
                    Vector3 pairNormal = Vector3.zero;
                    bool normalCompatible = true;
                    if (candidate.leftNormalCount > 0 &&
                        candidate.rightNormalCount > 0)
                    {
                        Vector3 leftNormal =
                            (candidate.leftNormalSum /
                             candidate.leftNormalCount).normalized;
                        Vector3 rightNormal =
                            (candidate.rightNormalSum /
                             candidate.rightNormalCount).normalized;
                        if (Vector3.Dot(leftNormal, rightNormal) <
                            normalAlignment)
                        {
                            candidate.confirmedPairs = 0;
                            normalCompatible = false;
                        }
                        else
                        {
                            pairNormal =
                                (leftNormal + rightNormal).normalized;
                        }
                    }

                    if (normalCompatible)
                    {
                        bool consecutive = candidate.lastConfirmedPair ==
                            _integrationPairSequence - 1;
                        bool positionStable = candidate.confirmedPairs == 0 ||
                            Vector3.Distance(
                                candidate.referencePosition,
                                pairPosition) <= positionTolerance;
                        bool directionStable =
                            candidate.confirmedPairs == 0 ||
                            pairNormal.sqrMagnitude <= 0.25f ||
                            candidate.referenceNormal.sqrMagnitude <= 0.25f ||
                            Vector3.Dot(
                                candidate.referenceNormal, pairNormal) >=
                            normalAlignment;
                        if (!consecutive || !positionStable ||
                            !directionStable)
                        {
                            candidate.confirmedPairs = 1;
                            candidate.referencePosition = pairPosition;
                            candidate.referenceNormal = pairNormal;
                            candidate.firstViewPosition = viewPosition;
                        }
                        else
                        {
                            int nextCount = Mathf.Min(
                                255, candidate.confirmedPairs + 1);
                            float blend = 1f / nextCount;
                            candidate.referencePosition = Vector3.Lerp(
                                candidate.referencePosition,
                                pairPosition, blend);
                            if (pairNormal.sqrMagnitude > 0.25f)
                                candidate.referenceNormal = Vector3.Lerp(
                                    candidate.referenceNormal,
                                    pairNormal, blend).normalized;
                            candidate.confirmedPairs = (byte)nextCount;
                        }
                        candidate.lastConfirmedPair =
                            _integrationPairSequence;

                        bool connectedFrontier =
                            fastTrackMatureSurfaceFrontier &&
                            HasPromotedSurfaceNeighbour(key);
                        int requiredPairs = connectedFrontier
                            ? Mathf.Max(2, frontierConfirmationPairs)
                            : Mathf.Max(2, novelSurfaceConfirmationPairs);
                        float requiredBaseline = connectedFrontier
                            ? 0f
                            : Mathf.Max(
                                0.01f,
                                novelSurfaceMinimumViewBaselineMeters);
                        if (candidate.confirmedPairs >= requiredPairs &&
                            Vector3.Distance(
                                candidate.firstViewPosition,
                                viewPosition) >= requiredBaseline)
                        {
                            candidate.promoted = true;
                            _promotedSurfaceCells.Add(key);
                            _surfaceCandidateCleanupKeys.Add(key);
                            LastNovelSurfacePromotedCount++;
                        }
                    }
                }
            }

            if ((_observationPreparationCursor & 7) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool PrepareExpiredCandidatesSlice(double deadline)
    {
        int expiryPairs = Mathf.Max(2, novelSurfaceExpiryPairs);
        while (_observationPreparationCursor <
               _surfaceCandidateFinalizeKeys.Count)
        {
            SurfaceCandidateKey key = _surfaceCandidateFinalizeKeys[
                _observationPreparationCursor++];
            if (_surfaceCandidates.TryGetValue(
                    key, out SurfaceCandidate candidate) &&
                !candidate.promoted &&
                _integrationPairSequence - candidate.lastSeenPair >
                expiryPairs)
                _surfaceCandidateCleanupKeys.Add(key);

            if ((_observationPreparationCursor & 63) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private void RemovePreparedSurfaceCandidateKeys()
    {
        for (int i = 0; i < _surfaceCandidateCleanupKeys.Count; i++)
            _surfaceCandidates.Remove(_surfaceCandidateCleanupKeys[i]);
        _surfaceCandidateCleanupKeys.Clear();
    }

    private void PrepareEyeObservation(
        NativeArray<Color> positions,
        NativeArray<Color> meta,
        int width,
        int height,
        Vector3 eye,
        Matrix4x4 worldToDepthClip,
        ref float[] observedDepths,
        ref float[] observationWeights,
        ref byte[] continuity,
        ref Vector3[] observationNormals,
        ref byte[] surfaceAdmission,
        List<Vector3Int> pendingChunks,
        byte eyeBit,
        out int expected,
        out Matrix4x4 worldToClip)
    {
        int count = Mathf.Min(positions.Length, meta.Length);
        expected = Mathf.Min(count, Mathf.Max(0, width * height));
        worldToClip = worldToDepthClip;
        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        float truncation = Mathf.Max(voxel, truncationMeters);
        // Candidate discovery is deliberately wider than the write band so a
        // surface crossing a chunk boundary cannot be missed. The actual TSDF
        // evidence band is selected later and is 15 cm by default.
        float candidateBand = truncation + Mathf.Sqrt(3f) * voxel;
        float chunkWorldSize = ChunkSize * voxel;
        EnsureDepthCache(ref observedDepths, expected);
        EnsureDepthCache(ref observationWeights, expected);
        EnsureByteCache(ref continuity, expected);
        EnsureVectorCache(ref observationNormals, expected);
        EnsureByteCache(ref surfaceAdmission, expected);

        for (int index = 0; index < expected; index++)
        {
            observedDepths[index] = 0f;
            observationWeights[index] = 0f;
            observationNormals[index] = Vector3.zero;
            surfaceAdmission[index] = 0;
            if (meta[index].r < 0.5f)
                continue;
            Color encoded = positions[index];
            Vector3 surface = new Vector3(encoded.r, encoded.g, encoded.b);
            if (!Finite(surface))
                continue;
            // The same provider matrix generated this world position.  Its clip
            // W is the projective depth used by the TSDF and cannot drift with
            // a later Camera.main render pose.
            float observedDepth = ProjectiveDepth(
                worldToDepthClip, surface.x, surface.y, surface.z);
            if (!float.IsFinite(observedDepth) ||
                observedDepth < minimumDepthMeters ||
                observedDepth > maximumDepthMeters)
                continue;

            observedDepths[index] = observedDepth;
            observationWeights[index] = Mathf.Clamp(
                meta[index].g, minimumObservationWeight, 1f);
            LastValidSampleCount++;
        }

        BuildDepthContinuityMask(
            observedDepths, continuity, width, height, expected);
        BuildObservationNormalsAndNovelSurfaceEvidence(
            positions, observedDepths, continuity, width, height, expected,
            eye, eyeBit, observationNormals, surfaceAdmission);

        // Surface candidates may allocate. Free-space candidates below only
        // select chunks which already exist, so carving cannot fill the view
        // frustum with empty storage.
        _candidateChunks.Clear();
        for (int index = 0; index < expected; index++)
        {
            if (observedDepths[index] <= 0f || continuity[index] == 0)
                continue;
            Color encoded = positions[index];
            Vector3 surface = new Vector3(encoded.r, encoded.g, encoded.b);
            float rayScale = (surface - eye).magnitude /
                             Mathf.Max(0.0001f, observedDepths[index]);
            float worldRadius = candidateBand * Mathf.Max(1f, rayScale) +
                                Mathf.Sqrt(3f) * voxel;
            AddCandidateChunkRange(surface, worldRadius, chunkWorldSize);
        }

        if (enableConfirmedFreeSpace)
        {
            int stride = Mathf.Max(2, freeSpaceRayStridePixels);
            for (int y = 0; y < height; y += stride)
            for (int x = 0; x < width; x += stride)
            {
                int index = x + y * width;
                if (index < 0 || index >= expected ||
                    observedDepths[index] <= 0f || continuity[index] == 0)
                    continue;
                Color encoded = positions[index];
                Vector3 surface = new Vector3(encoded.r, encoded.g, encoded.b);
                AddExistingFreeSpaceChunksAlongRay(
                    eye, surface, (surface - eye).magnitude, chunkWorldSize);
            }
        }

        LastCandidateChunkCount += _candidateChunks.Count;
        pendingChunks.AddRange(_candidateChunks);
        _candidateChunks.Clear();
    }

    private void ProcessIntegrationSlice()
    {
        double start = Time.realtimeSinceStartupAsDouble;
        double budgetSeconds = Mathf.Max(0.0005f,
            integrationBudgetMilliseconds * 0.001f);
        int chunkBudget = Mathf.Max(1, maximumIntegrationChunksPerFrame);
        int processed = 0;

        while (processed < chunkBudget)
        {
            bool hasChunk;
            if (_leftPendingChunkIndex < _leftPendingChunks.Count)
            {
                Vector3Int chunkId =
                    _leftPendingChunks[_leftPendingChunkIndex++];
                ProcessPendingChunk(
                    chunkId, _leftObservedDepth, _leftObservationWeights,
                    _leftDepthContinuity, _leftObservationNormals,
                    _leftSurfaceAdmission,
                    _leftPendingExpected, _pendingLeftWidth,
                    _pendingLeftHeight, _leftPendingWorldToClip, 1);
                hasChunk = true;
            }
            else if (_rightPendingChunkIndex < _rightPendingChunks.Count)
            {
                Vector3Int chunkId =
                    _rightPendingChunks[_rightPendingChunkIndex++];
                ProcessPendingChunk(
                    chunkId, _rightObservedDepth, _rightObservationWeights,
                    _rightDepthContinuity, _rightObservationNormals,
                    _rightSurfaceAdmission,
                    _rightPendingExpected, _pendingRightWidth,
                    _pendingRightHeight, _rightPendingWorldToClip, 2);
                hasChunk = true;
            }
            else
            {
                hasChunk = false;
            }

            if (!hasChunk)
                break;
            processed++;
            if (Time.realtimeSinceStartupAsDouble - start >= budgetSeconds)
                break;
        }

        float elapsed = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
        _integrationTotalMilliseconds += elapsed;
        LastIntegrationMilliseconds = Mathf.Max(
            LastIntegrationMilliseconds, elapsed);

        if (_leftPendingChunkIndex < _leftPendingChunks.Count ||
            _rightPendingChunkIndex < _rightPendingChunks.Count)
            return;

        ResolveFreeSpaceConflicts();
        MarkTouchedChunkMeshesDirty();
        AcceptedStereoPairCount++;
        LastIntegrationTotalMilliseconds =
            (float)_integrationTotalMilliseconds;
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
        _integrationPending = false;
    }

    private void ProcessPendingChunk(
        Vector3Int chunkId,
        float[] observedDepths,
        float[] observationWeights,
        byte[] continuity,
        Vector3[] observationNormals,
        byte[] surfaceAdmission,
        int expected,
        int width,
        int height,
        Matrix4x4 worldToClip,
        byte eyeBit)
    {
        bool chunkNew = !_chunks.TryGetValue(chunkId, out Chunk chunk);
        if (chunkNew)
        {
            chunk = AcquireEmptyChunk();
            _chunks.Add(chunkId, chunk);
        }

        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        float truncation = Mathf.Max(voxel, truncationMeters);
        float legacyUpdateBand = truncation + 2f * Mathf.Sqrt(3f) * voxel;
        float surfaceEvidenceBand = strictSurfaceEvidenceBand
            ? truncation
            : legacyUpdateBand;
        bool updatedChunk = IntegrateProjectedChunk(
            chunkId, chunk, observedDepths, observationWeights, continuity,
            observationNormals, surfaceAdmission,
            expected, width, height, worldToClip,
            surfaceEvidenceBand, voxel, eyeBit,
            out byte meshDirtyMask);
        if (updatedChunk)
        {
            LastUpdatedChunkCount++;
            chunk.lastTouchedFrame = _pendingFrameOrdinal;
            if ((meshDirtyMask & 8) != 0)
                RecordMeshDirtySource(chunkId, meshDirtyMask);
        }
        else if (chunkNew)
        {
            RemoveEmptyChunk(chunkId, chunk);
        }
    }

    private bool IntegrateProjectedChunk(
        Vector3Int chunkId,
        Chunk chunk,
        float[] observedDepths,
        float[] observationWeights,
        byte[] continuity,
        Vector3[] observationNormals,
        byte[] surfaceAdmission,
        int expected,
        int width,
        int height,
        Matrix4x4 worldToClip,
        float surfaceEvidenceBand,
        float voxel,
        byte eyeBit,
        out byte meshDirtyMask)
    {
        Vector3Int chunkOrigin = chunkId * ChunkSize;
        bool updatedChunk = false;
        meshDirtyMask = 0;
        float baseX = (chunkOrigin.x + 0.5f) * voxel;
        float baseY = (chunkOrigin.y + 0.5f) * voxel;
        float baseZ = (chunkOrigin.z + 0.5f) * voxel;
        for (int z = 0; z < ChunkSize; z++)
        for (int y = 0; y < ChunkSize; y++)
        for (int x = 0; x < ChunkSize; x++)
        {
            int localIndex = x + ChunkSize * (y + ChunkSize * z);
            float worldX = baseX + x * voxel;
            float worldY = baseY + y * voxel;
            float worldZ = baseZ + z * voxel;
            float clipX = worldToClip.m00 * worldX +
                          worldToClip.m01 * worldY +
                          worldToClip.m02 * worldZ + worldToClip.m03;
            float clipY = worldToClip.m10 * worldX +
                          worldToClip.m11 * worldY +
                          worldToClip.m12 * worldZ + worldToClip.m13;
            float clipW = worldToClip.m30 * worldX +
                          worldToClip.m31 * worldY +
                          worldToClip.m32 * worldZ + worldToClip.m33;
            float voxelDepth = clipW;
            if (!float.IsFinite(voxelDepth) || voxelDepth <= 0.000001f)
                continue;
            float inverseW = 1f / clipW;
            float u = clipX * inverseW * 0.5f + 0.5f;
            float v = clipY * inverseW * 0.5f + 0.5f;
            if (u < 0f || u >= 1f || v < 0f || v >= 1f)
                continue;

            int px = Mathf.Clamp(Mathf.FloorToInt(u * width), 0, width - 1);
            int py = Mathf.Clamp(Mathf.FloorToInt(v * height), 0, height - 1);
            int index = px + py * width;
            if (index < 0 || index >= expected)
                continue;
            float observedDepth = observedDepths[index];
            if (observedDepth <= 0f || continuity[index] == 0)
                continue;

            float surfaceDistance = observedDepth - voxelDepth;
            Voxel old = chunk.voxels[localIndex];
            if (enableConfirmedFreeSpace &&
                surfaceDistance >= Mathf.Max(voxel, repairConflictMeters) &&
                old.weight > 0f &&
                old.distance <= freeSpaceConflictMeters * 0.5f)
            {
                RecordFreeSpaceConflict(
                    chunkId, chunk, localIndex, eyeBit);
                continue;
            }
            if (Mathf.Abs(surfaceDistance) >= surfaceEvidenceBand)
                continue;

            if (confirmNovelSurfacesBeforeFusion && old.weight <= 0f &&
                (surfaceAdmission == null ||
                 index >= surfaceAdmission.Length ||
                 surfaceAdmission[index] == 0))
            {
                LastNovelSurfaceHeldCount++;
                continue;
            }

            float sampleDistance = Mathf.Clamp(
                surfaceDistance, -truncationMeters, truncationMeters);
            float observationWeight = observationWeights != null &&
                                      index < observationWeights.Length
                ? Mathf.Clamp(observationWeights[index],
                    minimumObservationWeight, 1f)
                : 1f;
            if (!useViewWeightedObservations)
                observationWeight = 1f;

            // Promotion already represents several paired observations from a
            // translated viewpoint. Transfer that confidence into the fresh
            // TSDF band so distant, low-weight detail does not need many extra
            // seconds merely to reach the topology threshold.
            if (confirmNovelSurfacesBeforeFusion &&
                surfaceAdmission != null &&
                index < surfaceAdmission.Length &&
                surfaceAdmission[index] != 0 &&
                old.weight < Mathf.Max(1f, minimumMeshWeight))
            {
                observationWeight = Mathf.Max(
                    observationWeight,
                    Mathf.Max(1f, minimumMeshWeight) * 0.5f);
            }

            Vector3 observedNormal = observationNormals != null &&
                                     index < observationNormals.Length
                ? observationNormals[index]
                : Vector3.zero;

            if (protectSurfaceDirection &&
                old.directionWeight >= Mathf.Max(1f, matureDirectionWeight) &&
                observedNormal.sqrMagnitude > 0.25f)
            {
                float minimumAlignment = Mathf.Cos(
                    Mathf.Clamp(maximumDirectionChangeDegrees, 20f, 85f) *
                    Mathf.Deg2Rad);
                Vector3 existingDirection = old.direction.sqrMagnitude > 0.25f
                    ? old.direction.normalized
                    : Vector3.zero;
                if (existingDirection.sqrMagnitude > 0.25f &&
                    Vector3.Dot(existingDirection, observedNormal) <
                    minimumAlignment)
                {
                    LastDirectionRejectedCount++;
                    continue;
                }
            }

            // Mature evidence is not blindly averaged with a contradictory
            // observation. A farther observation already entered the confirmed
            // free-space path above; all other large residuals remain unwritten
            // so a single edge leak cannot build a false bridge.
            if (enableObservationAssociation &&
                old.weight >= Mathf.Max(1f, matureAssociationWeight) &&
                Mathf.Abs(sampleDistance - old.distance) >
                Mathf.Max(voxel, associationResidualMeters))
            {
                LastAssociationRejectedCount++;
                continue;
            }

            if (old.weight < Mathf.Max(1f, matureAssociationWeight))
                observationWeight *= Mathf.Clamp01(immatureObservationScale);

            RecordSurfaceSupport(chunk, localIndex);
            bool meshChanged = IntegrateVoxel(
                chunk, localIndex, sampleDistance, observationWeight,
                observedNormal);
            if (meshChanged)
                AddMeshDirtyVoxel(ref meshDirtyMask, localIndex);
            updatedChunk = true;
        }
        return updatedChunk;
    }

    private static float ProjectiveDepth(
        Matrix4x4 worldToDepthClip,
        float worldX,
        float worldY,
        float worldZ)
    {
        return worldToDepthClip.m30 * worldX +
               worldToDepthClip.m31 * worldY +
               worldToDepthClip.m32 * worldZ +
               worldToDepthClip.m33;
    }

    private void RecordSurfaceSupport(Chunk chunk, int localIndex)
    {
        Voxel voxel = chunk.voxels[localIndex];
        voxel.lastSupportPair = _integrationPairSequence;
        voxel.freeSpaceConflictStreak = 0;
        chunk.voxels[localIndex] = voxel;
    }

    private void RecordFreeSpaceConflict(
        Vector3Int chunkId, Chunk chunk, int localIndex, byte eyeBit)
    {
        Voxel voxel = chunk.voxels[localIndex];
        if (voxel.pendingConflictPair != _integrationPairSequence)
        {
            voxel.pendingConflictPair = _integrationPairSequence;
            voxel.pendingConflictEyes = 0;
        }
        voxel.pendingConflictEyes |= eyeBit;
        chunk.voxels[localIndex] = voxel;

        var address = new VoxelAddress(chunkId, localIndex);
        if (_freeSpaceEvidenceSet.Add(address))
            _freeSpaceEvidenceVoxels.Add(address);
    }

    private void ResolveFreeSpaceConflicts()
    {
        if (!enableConfirmedFreeSpace)
            return;

        int requiredPairs = Mathf.Max(2, freeSpaceConfirmationPairs);
        float correctionDistance = Mathf.Max(
            Mathf.Max(0.005f, voxelSizeMeters), truncationMeters);
        for (int i = 0; i < _freeSpaceEvidenceVoxels.Count; i++)
        {
            VoxelAddress address = _freeSpaceEvidenceVoxels[i];
            if (!_chunks.TryGetValue(address.chunkId, out Chunk chunk))
                continue;
            Voxel voxel = chunk.voxels[address.localIndex];
            if (voxel.lastSupportPair == _integrationPairSequence)
            {
                voxel.freeSpaceConflictStreak = 0;
                voxel.pendingConflictEyes = 0;
                chunk.voxels[address.localIndex] = voxel;
                continue;
            }

            // A direct carve is destructive, so it requires the same conflict
            // from both members of the accepted stereo pair.
            if (voxel.pendingConflictPair != _integrationPairSequence ||
                (voxel.pendingConflictEyes & 3) != 3)
                continue;

            voxel.freeSpaceConflictStreak =
                voxel.lastConflictPair == _integrationPairSequence - 1
                    ? (byte)Mathf.Min(255,
                        voxel.freeSpaceConflictStreak + 1)
                    : (byte)1;
            voxel.lastConflictPair = _integrationPairSequence;
            chunk.voxels[address.localIndex] = voxel;

            LastFreeSpaceConflictCount++;
            if (voxel.freeSpaceConflictStreak < requiredPairs)
                continue;

            bool meshChanged;
            if (directConfirmedFreeSpaceCarving)
                meshChanged = CarveVoxel(
                    chunk, address.localIndex, correctionDistance);
            else
                meshChanged = IntegrateVoxel(
                    chunk, address.localIndex, correctionDistance,
                    Mathf.Max(0.25f, freeSpaceCorrectionWeight),
                    Vector3.zero);
            chunk.lastTouchedFrame = _pendingFrameOrdinal;
            if (meshChanged)
            {
                byte meshDirtyMask = 0;
                AddMeshDirtyVoxel(ref meshDirtyMask, address.localIndex);
                RecordMeshDirtySource(address.chunkId, meshDirtyMask);
            }
            LastFreeSpaceCorrectionCount++;
        }
    }

    private bool IntegrateVoxel(
        Chunk chunk,
        int localIndex,
        float signedDistance,
        float observationWeight,
        Vector3 observationDirection)
    {
        Voxel old = chunk.voxels[localIndex];
        bool wasSurface = IsSurfaceVoxel(old);
        if (old.weight <= 0f)
            WeightedVoxelCount++;
        float sampleWeight = Mathf.Max(0.001f, observationWeight);
        float weightLimit = Mathf.Max(sampleWeight, maximumVoxelWeight);
        float retainedWeight = old.weight > 0f
            ? Mathf.Min(old.weight, Mathf.Max(0f, weightLimit - sampleWeight))
            : 0f;
        float newWeight = retainedWeight + sampleWeight;
        float distance = retainedWeight > 0f
            ? (old.distance * retainedWeight +
               signedDistance * sampleWeight) / newWeight
            : signedDistance;
        Voxel updated = old;
        updated.distance = distance;
        updated.weight = newWeight;
        if (observationDirection.sqrMagnitude > 0.25f)
        {
            Vector3 direction = observationDirection.normalized;
            float oldDirectionWeight = Mathf.Min(
                old.directionWeight,
                Mathf.Max(0f, weightLimit - sampleWeight));
            Vector3 directionSum = old.direction.sqrMagnitude > 0.25f
                ? old.direction.normalized * oldDirectionWeight
                : Vector3.zero;
            updated.direction = (directionSum + direction * sampleWeight)
                .normalized;
            updated.directionWeight = oldDirectionWeight + sampleWeight;
        }
        chunk.voxels[localIndex] = updated;
        bool isSurface = IsSurfaceVoxel(updated);
        if (wasSurface != isSurface)
            SurfaceVoxelCount += isSurface ? 1 : -1;
        LastVoxelWriteCount++;
        return MeshStateDiffersFromPublished(chunk, localIndex, updated);
    }

    private bool CarveVoxel(
        Chunk chunk, int localIndex, float freeSpaceDistance)
    {
        Voxel old = chunk.voxels[localIndex];
        bool wasSurface = IsSurfaceVoxel(old);
        if (old.weight <= 0f)
            return false;

        Voxel carved = old;
        carved.distance = Mathf.Max(
            Mathf.Max(0.005f, voxelSizeMeters), freeSpaceDistance);
        carved.weight = Mathf.Max(
            Mathf.Max(1f, minimumMeshWeight),
            Mathf.Min(old.weight, Mathf.Max(1f, maximumVoxelWeight)));
        carved.direction = Vector3.zero;
        carved.directionWeight = 0f;
        carved.freeSpaceConflictStreak = 0;
        carved.pendingConflictEyes = 0;
        chunk.voxels[localIndex] = carved;
        if (wasSurface)
            SurfaceVoxelCount--;
        LastVoxelWriteCount++;
        return MeshStateDiffersFromPublished(chunk, localIndex, carved);
    }

    private void BuildDepthContinuityMask(
        float[] observedDepths,
        byte[] continuity,
        int width,
        int height,
        int expected)
    {
        Array.Clear(continuity, 0, expected);
        int requiredNeighbours = Mathf.Clamp(
            minimumConsistentDepthNeighbours, 1, 8);
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;
            if (index < 0 || index >= expected || observedDepths[index] <= 0f)
                continue;
            if (!protectDepthDiscontinuities)
            {
                continuity[index] = 1;
                continue;
            }

            float center = observedDepths[index];
            float tolerance = Mathf.Max(
                depthEdgeAbsoluteMeters,
                center * depthEdgeRelativeFraction);
            int consistent = 0;
            for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++)
            {
                if ((ox == 0 && oy == 0) ||
                    x + ox < 0 || x + ox >= width ||
                    y + oy < 0 || y + oy >= height)
                    continue;
                int neighbourIndex = x + ox + (y + oy) * width;
                if (neighbourIndex < 0 || neighbourIndex >= expected)
                    continue;
                float neighbour = observedDepths[neighbourIndex];
                if (neighbour > 0f &&
                    Mathf.Abs(neighbour - center) <= tolerance)
                    consistent++;
            }

            if (consistent >= requiredNeighbours)
                continuity[index] = 1;
            else
                LastDepthEdgeRejectedCount++;
        }
    }

    private void BuildObservationNormalsAndNovelSurfaceEvidence(
        NativeArray<Color> positions,
        float[] observedDepths,
        byte[] continuity,
        int width,
        int height,
        int expected,
        Vector3 eye,
        byte eyeBit,
        Vector3[] observationNormals,
        byte[] surfaceAdmission)
    {
        Array.Clear(observationNormals, 0, expected);
        Array.Clear(surfaceAdmission, 0, expected);
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;
            if (index < 0 || index >= expected ||
                observedDepths[index] <= 0f || continuity[index] == 0)
                continue;

            Color encoded = positions[index];
            Vector3 surface = new Vector3(encoded.r, encoded.g, encoded.b);
            if (!Finite(surface))
                continue;

            Vector3 normal = EstimateObservationNormal(
                positions, observedDepths, continuity,
                x, y, width, height, expected, surface, eye);
            observationNormals[index] = normal;

            if (!confirmNovelSurfacesBeforeFusion)
            {
                surfaceAdmission[index] = 1;
                continue;
            }

            Vector3Int spatialKey = NovelSurfaceCell(surface);
            SurfaceCandidateKey key = NovelSurfaceKey(surface, normal);
            HashSet<Vector3Int> checkedCells = (eyeBit & 1) != 0
                ? _leftNovelOcclusionChecked
                : _rightNovelOcclusionChecked;
            HashSet<Vector3Int> occludedCells = (eyeBit & 1) != 0
                ? _leftOccludedNovelCells
                : _rightOccludedNovelCells;
            if (checkedCells.Add(spatialKey) &&
                HasMatureOccluderAlongRay(eye, surface))
                occludedCells.Add(spatialKey);
            if (occludedCells.Contains(spatialKey))
            {
                LastNovelSurfaceOcclusionRejectedCount++;
                continue;
            }
            if (IsNovelSurfacePromoted(key))
            {
                surfaceAdmission[index] = 1;
                continue;
            }
            RecordNovelSurfaceCandidate(key, surface, normal, eyeBit);
        }
    }

    private bool HasMatureOccluderAlongRay(Vector3 eye, Vector3 surface)
    {
        Vector3 ray = surface - eye;
        float length = ray.magnitude;
        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        float end = length - Mathf.Max(
            Mathf.Max(voxel * 2f, freeSpaceConflictMeters), 0.10f);
        float start = Mathf.Max(0.01f, minimumDepthMeters);
        if (!Finite(ray) || end <= start)
            return false;

        Vector3 direction = ray / length;
        float step = Mathf.Max(voxel * 0.75f, 0.025f);
        Vector3Int previous = new Vector3Int(int.MinValue, int.MinValue, int.MinValue);
        for (float distance = start; distance <= end; distance += step)
        {
            Vector3 point = eye + direction * distance;
            Vector3Int voxelId = new Vector3Int(
                Mathf.FloorToInt(point.x / voxel),
                Mathf.FloorToInt(point.y / voxel),
                Mathf.FloorToInt(point.z / voxel));
            if (voxelId == previous)
                continue;
            previous = voxelId;
            if (TryGetVoxelRaw(voxelId, out Voxel existing) &&
                IsSurfaceVoxel(existing))
                return true;
        }
        return false;
    }

    private Vector3 EstimateObservationNormal(
        NativeArray<Color> positions,
        float[] observedDepths,
        byte[] continuity,
        int x,
        int y,
        int width,
        int height,
        int expected,
        Vector3 center,
        Vector3 eye)
    {
        Vector3 horizontal = Vector3.zero;
        Vector3 vertical = Vector3.zero;
        if (TryObservationPoint(
                positions, observedDepths, continuity,
                x + 1, y, width, height, expected, out Vector3 right))
            horizontal = right - center;
        else if (TryObservationPoint(
                     positions, observedDepths, continuity,
                     x - 1, y, width, height, expected, out Vector3 left))
            horizontal = center - left;

        if (TryObservationPoint(
                positions, observedDepths, continuity,
                x, y + 1, width, height, expected, out Vector3 up))
            vertical = up - center;
        else if (TryObservationPoint(
                     positions, observedDepths, continuity,
                     x, y - 1, width, height, expected, out Vector3 down))
            vertical = center - down;

        Vector3 normal = Vector3.Cross(horizontal, vertical);
        if (!Finite(normal) || normal.sqrMagnitude < 0.000001f)
            return Vector3.zero;
        normal.Normalize();
        if (Vector3.Dot(normal, eye - center) < 0f)
            normal = -normal;
        return normal;
    }

    private static bool TryObservationPoint(
        NativeArray<Color> positions,
        float[] observedDepths,
        byte[] continuity,
        int x,
        int y,
        int width,
        int height,
        int expected,
        out Vector3 point)
    {
        point = Vector3.zero;
        if (x < 0 || x >= width || y < 0 || y >= height)
            return false;
        int index = x + y * width;
        if (index < 0 || index >= expected ||
            observedDepths[index] <= 0f || continuity[index] == 0)
            return false;
        Color encoded = positions[index];
        point = new Vector3(encoded.r, encoded.g, encoded.b);
        return Finite(point);
    }

    private Vector3Int NovelSurfaceCell(Vector3 surface)
    {
        float cell = Mathf.Max(
            Mathf.Max(0.005f, voxelSizeMeters), novelSurfaceCellMeters);
        return new Vector3Int(
            Mathf.FloorToInt(surface.x / cell),
            Mathf.FloorToInt(surface.y / cell),
            Mathf.FloorToInt(surface.z / cell));
    }

    private SurfaceCandidateKey NovelSurfaceKey(
        Vector3 surface, Vector3 normal)
    {
        byte axis = 3;
        if (normal.sqrMagnitude > 0.25f)
        {
            Vector3 absolute = new Vector3(
                Mathf.Abs(normal.x), Mathf.Abs(normal.y), Mathf.Abs(normal.z));
            axis = absolute.x >= absolute.y && absolute.x >= absolute.z
                ? (byte)0
                : absolute.y >= absolute.z ? (byte)1 : (byte)2;
        }
        return new SurfaceCandidateKey(NovelSurfaceCell(surface), axis);
    }

    private bool IsNovelSurfacePromoted(SurfaceCandidateKey key)
    {
        return _promotedSurfaceCells.Contains(key);
    }

    private void RecordNovelSurfaceCandidate(
        SurfaceCandidateKey key,
        Vector3 surface,
        Vector3 normal,
        byte eyeBit)
    {
        if (!_surfaceCandidates.TryGetValue(key, out SurfaceCandidate candidate))
        {
            candidate = new SurfaceCandidate();
            _surfaceCandidates.Add(key, candidate);
        }

        if (candidate.pendingPair != _integrationPairSequence)
        {
            candidate.pendingPair = _integrationPairSequence;
            candidate.pendingEyes = 0;
            candidate.leftPositionSum = Vector3.zero;
            candidate.rightPositionSum = Vector3.zero;
            candidate.leftNormalSum = Vector3.zero;
            candidate.rightNormalSum = Vector3.zero;
            candidate.leftCount = 0;
            candidate.rightCount = 0;
            candidate.leftNormalCount = 0;
            candidate.rightNormalCount = 0;
        }

        candidate.pendingEyes |= eyeBit;
        candidate.lastSeenPair = _integrationPairSequence;
        if ((eyeBit & 1) != 0)
        {
            candidate.leftPositionSum += surface;
            candidate.leftCount++;
            if (normal.sqrMagnitude > 0.25f)
            {
                candidate.leftNormalSum += normal;
                candidate.leftNormalCount++;
            }
        }
        if ((eyeBit & 2) != 0)
        {
            candidate.rightPositionSum += surface;
            candidate.rightCount++;
            if (normal.sqrMagnitude > 0.25f)
            {
                candidate.rightNormalSum += normal;
                candidate.rightNormalCount++;
            }
        }
        _surfaceCandidatesTouched.Add(key);
    }

    private void FinalizeNovelSurfaceCandidates(Vector3 viewPosition)
    {
        if (!confirmNovelSurfacesBeforeFusion)
            return;

        float positionTolerance = Mathf.Max(
            Mathf.Max(0.005f, voxelSizeMeters),
            novelSurfacePositionToleranceMeters);
        float normalAlignment = Mathf.Cos(
            Mathf.Clamp(maximumDirectionChangeDegrees, 20f, 85f) *
            Mathf.Deg2Rad);
        _surfaceCandidateCleanupKeys.Clear();

        foreach (SurfaceCandidateKey key in _surfaceCandidatesTouched)
        {
            if (!_surfaceCandidates.TryGetValue(
                    key, out SurfaceCandidate candidate) ||
                candidate.promoted)
                continue;

            bool stereoComplete =
                candidate.pendingPair == _integrationPairSequence &&
                (candidate.pendingEyes & 3) == 3 &&
                candidate.leftCount > 0 && candidate.rightCount > 0;
            if (!stereoComplete)
            {
                candidate.confirmedPairs = 0;
                continue;
            }

            Vector3 leftPosition =
                candidate.leftPositionSum / candidate.leftCount;
            Vector3 rightPosition =
                candidate.rightPositionSum / candidate.rightCount;
            if (Vector3.Distance(leftPosition, rightPosition) >
                positionTolerance)
            {
                candidate.confirmedPairs = 0;
                continue;
            }

            Vector3 pairPosition = (leftPosition + rightPosition) * 0.5f;
            Vector3 pairNormal = Vector3.zero;
            if (candidate.leftNormalCount > 0 &&
                candidate.rightNormalCount > 0)
            {
                Vector3 leftNormal =
                    (candidate.leftNormalSum / candidate.leftNormalCount)
                    .normalized;
                Vector3 rightNormal =
                    (candidate.rightNormalSum / candidate.rightNormalCount)
                    .normalized;
                if (Vector3.Dot(leftNormal, rightNormal) < normalAlignment)
                {
                    candidate.confirmedPairs = 0;
                    continue;
                }
                pairNormal = (leftNormal + rightNormal).normalized;
            }

            bool consecutive =
                candidate.lastConfirmedPair == _integrationPairSequence - 1;
            bool positionStable = candidate.confirmedPairs == 0 ||
                Vector3.Distance(candidate.referencePosition, pairPosition) <=
                positionTolerance;
            bool directionStable = candidate.confirmedPairs == 0 ||
                pairNormal.sqrMagnitude <= 0.25f ||
                candidate.referenceNormal.sqrMagnitude <= 0.25f ||
                Vector3.Dot(candidate.referenceNormal, pairNormal) >=
                normalAlignment;
            if (!consecutive || !positionStable || !directionStable)
            {
                candidate.confirmedPairs = 1;
                candidate.referencePosition = pairPosition;
                candidate.referenceNormal = pairNormal;
                candidate.firstViewPosition = viewPosition;
            }
            else
            {
                int nextCount = Mathf.Min(255, candidate.confirmedPairs + 1);
                float blend = 1f / nextCount;
                candidate.referencePosition = Vector3.Lerp(
                    candidate.referencePosition, pairPosition, blend);
                if (pairNormal.sqrMagnitude > 0.25f)
                    candidate.referenceNormal = Vector3.Lerp(
                        candidate.referenceNormal, pairNormal, blend).normalized;
                candidate.confirmedPairs = (byte)nextCount;
            }
            candidate.lastConfirmedPair = _integrationPairSequence;

            bool connectedFrontier = fastTrackMatureSurfaceFrontier &&
                HasPromotedSurfaceNeighbour(key);
            int requiredPairs = connectedFrontier
                ? Mathf.Max(2, frontierConfirmationPairs)
                : Mathf.Max(2, novelSurfaceConfirmationPairs);
            float requiredBaseline = connectedFrontier
                ? 0f
                : Mathf.Max(0.01f, novelSurfaceMinimumViewBaselineMeters);
            if (candidate.confirmedPairs >= requiredPairs &&
                Vector3.Distance(candidate.firstViewPosition, viewPosition) >=
                requiredBaseline)
            {
                candidate.promoted = true;
                _promotedSurfaceCells.Add(key);
                _surfaceCandidateCleanupKeys.Add(key);
                LastNovelSurfacePromotedCount++;
            }
        }

        for (int i = 0; i < _surfaceCandidateCleanupKeys.Count; i++)
            _surfaceCandidates.Remove(_surfaceCandidateCleanupKeys[i]);
        _surfaceCandidateCleanupKeys.Clear();

        CleanupExpiredNovelSurfaceCandidates();
    }

    private bool HasPromotedSurfaceNeighbour(SurfaceCandidateKey key)
    {
        int radius = Mathf.Clamp(frontierNeighbourRadiusCells, 1, 2);
        for (int z = -radius; z <= radius; z++)
        for (int y = -radius; y <= radius; y++)
        for (int x = -radius; x <= radius; x++)
        {
            if (x == 0 && y == 0 && z == 0)
                continue;
            var neighbour = new SurfaceCandidateKey(
                key.cell + new Vector3Int(x, y, z), key.directionAxis);
            if (_promotedSurfaceCells.Contains(neighbour))
                return true;
        }
        return false;
    }

    private void CleanupExpiredNovelSurfaceCandidates()
    {
        int expiryPairs = Mathf.Max(2, novelSurfaceExpiryPairs);
        _surfaceCandidateCleanupKeys.Clear();
        foreach (KeyValuePair<SurfaceCandidateKey, SurfaceCandidate> item in
                 _surfaceCandidates)
        {
            if (!item.Value.promoted &&
                _integrationPairSequence - item.Value.lastSeenPair > expiryPairs)
                _surfaceCandidateCleanupKeys.Add(item.Key);
        }
        for (int i = 0; i < _surfaceCandidateCleanupKeys.Count; i++)
            _surfaceCandidates.Remove(_surfaceCandidateCleanupKeys[i]);
        _surfaceCandidateCleanupKeys.Clear();
    }

    private void AddExistingFreeSpaceChunksAlongRay(
        Vector3 eye,
        Vector3 surface,
        float observedDepth,
        float chunkWorldSize)
    {
        Vector3 ray = surface - eye;
        float rayLength = ray.magnitude;
        if (!Finite(surface) || rayLength <= 0.0001f || observedDepth <= 0f)
            return;

        float endScale = Mathf.Clamp01(
            (observedDepth - Mathf.Max(0.01f, freeSpaceConflictMeters)) /
            observedDepth);
        float startScale = Mathf.Clamp01(
            Mathf.Max(0.01f, minimumDepthMeters) / observedDepth);
        if (endScale <= startScale)
            return;

        float stepScale = Mathf.Max(0.01f, chunkWorldSize * 0.65f) /
                          rayLength;
        for (float scale = startScale; scale <= endScale;
             scale += stepScale)
        {
            Vector3Int chunkId = WorldToChunk(
                eye + ray * scale, chunkWorldSize);
            if (_chunks.ContainsKey(chunkId))
                _candidateChunks.Add(chunkId);
        }
    }

    private void AddCandidateChunkRange(
        Vector3 surface, float radius, float chunkWorldSize)
    {
        Vector3Int minimum = WorldToChunk(surface - Vector3.one * radius,
                                          chunkWorldSize);
        Vector3Int maximum = WorldToChunk(surface + Vector3.one * radius,
                                          chunkWorldSize);
        for (int z = minimum.z; z <= maximum.z; z++)
        for (int y = minimum.y; y <= maximum.y; y++)
        for (int x = minimum.x; x <= maximum.x; x++)
            _candidateChunks.Add(new Vector3Int(x, y, z));
    }

    private static Vector3Int WorldToChunk(Vector3 world, float chunkWorldSize)
    {
        return new Vector3Int(
            Mathf.FloorToInt(world.x / chunkWorldSize),
            Mathf.FloorToInt(world.y / chunkWorldSize),
            Mathf.FloorToInt(world.z / chunkWorldSize));
    }

    private void RemoveEmptyChunk(Vector3Int chunkId, Chunk chunk)
    {
        DisposeChunkPreview(chunk);
        _chunks.Remove(chunkId);
        chunk.lastTouchedFrame = 0;
        chunk.meshStateInitialized = false;
        chunk.meshedDistances = null;
        chunk.meshedObserved = null;
        if (_emptyChunkPool.Count < 64)
            _emptyChunkPool.Push(chunk);
        EmptyChunkCollectionCount++;
    }

    private Chunk AcquireEmptyChunk()
    {
        return _emptyChunkPool.Count > 0 ? _emptyChunkPool.Pop() : new Chunk();
    }

    private static void EnsureColorCache(
        ref NativeArray<Color> cache, int required)
    {
        if (required < 0)
            required = 0;
        if (cache.IsCreated && cache.Length >= required)
            return;
        if (cache.IsCreated)
            cache.Dispose();
        cache = new NativeArray<Color>(
            required, Allocator.Persistent,
            NativeArrayOptions.UninitializedMemory);
    }

    private static void DisposeColorCache(ref NativeArray<Color> cache)
    {
        if (cache.IsCreated)
            cache.Dispose();
        cache = default;
    }

    private static void EnsureDepthCache(ref float[] cache, int required)
    {
        if (cache == null || cache.Length < required)
            cache = new float[required];
    }

    private static void EnsureByteCache(ref byte[] cache, int required)
    {
        if (cache == null || cache.Length < required)
            cache = new byte[required];
    }

    private static void EnsureVectorCache(ref Vector3[] cache, int required)
    {
        if (cache == null || cache.Length < required)
            cache = new Vector3[required];
    }

    private void MarkTouchedChunkMeshesDirty()
    {
        // A mesh cell owned by a chunk only reads that chunk and its positive
        // X/Y/Z neighbours. An interior source voxel therefore dirties only
        // its own owner. A negative neighbour is invalidated only when the
        // changed voxel actually lies on the corresponding minimum boundary.
        foreach (Vector3Int touched in _touchedChunks)
        {
            byte mask = _touchedChunkBoundaryMasks.TryGetValue(
                touched, out byte value) ? value : (byte)8;
            int minimumX = (mask & 1) != 0 ? -1 : 0;
            int minimumY = (mask & 2) != 0 ? -1 : 0;
            int minimumZ = (mask & 4) != 0 ? -1 : 0;
            for (int z = minimumZ; z <= 0; z++)
            for (int y = minimumY; y <= 0; y++)
            for (int x = minimumX; x <= 0; x++)
            {
                Vector3Int dirty = touched + new Vector3Int(x, y, z);
                if (_queuedMeshChunks.Add(dirty))
                    _meshChunkQueue.Enqueue(dirty);
            }
        }
        _touchedChunks.Clear();
        _touchedChunkBoundaryMasks.Clear();
    }

    private void RecordMeshDirtySource(Vector3Int chunkId, byte dirtyMask)
    {
        _touchedChunks.Add(chunkId);
        if (_touchedChunkBoundaryMasks.TryGetValue(chunkId, out byte existing))
            _touchedChunkBoundaryMasks[chunkId] = (byte)(existing | dirtyMask);
        else
            _touchedChunkBoundaryMasks.Add(chunkId, dirtyMask);
    }

    private static void AddMeshDirtyVoxel(ref byte dirtyMask, int localIndex)
    {
        int x = localIndex % ChunkSize;
        int yz = localIndex / ChunkSize;
        int y = yz % ChunkSize;
        int z = yz / ChunkSize;
        dirtyMask |= 8;
        if (x == 0)
            dirtyMask |= 1;
        if (y == 0)
            dirtyMask |= 2;
        if (z == 0)
            dirtyMask |= 4;
    }

    private bool MeshStateDiffersFromPublished(
        Chunk chunk, int localIndex, Voxel current)
    {
        float requiredWeight = Mathf.Max(1f, minimumMeshWeight);
        bool currentObserved = current.weight >= requiredWeight;
        if (!chunk.meshStateInitialized ||
            chunk.meshedDistances == null ||
            chunk.meshedObserved == null)
        {
            // Before the first extraction, sub-threshold accumulation cannot
            // form a Marching-Cubes cell. Queue the chunk as soon as a sample
            // first becomes eligible for meshing.
            return currentObserved;
        }

        bool publishedObserved = chunk.meshedObserved[localIndex] != 0;
        if (currentObserved != publishedObserved)
            return true;
        if (!currentObserved)
            return false;

        float publishedDistance = chunk.meshedDistances[localIndex];
        if ((current.distance < 0f) != (publishedDistance < 0f))
            return true;

        float threshold = Mathf.Clamp(
            previewGeometryChangeMeters,
            0.001f,
            Mathf.Max(0.001f, voxelSizeMeters * 0.5f));
        return Mathf.Abs(current.distance - publishedDistance) >= threshold;
    }

    private static void CaptureMeshedVoxelState(Chunk chunk, float requiredWeight)
    {
        if (chunk.meshedDistances == null ||
            chunk.meshedDistances.Length != VoxelsPerChunk)
            chunk.meshedDistances = new float[VoxelsPerChunk];
        if (chunk.meshedObserved == null ||
            chunk.meshedObserved.Length != VoxelsPerChunk)
            chunk.meshedObserved = new byte[VoxelsPerChunk];

        for (int i = 0; i < VoxelsPerChunk; i++)
        {
            Voxel voxel = chunk.voxels[i];
            bool observed = voxel.weight >= requiredWeight;
            chunk.meshedObserved[i] = observed ? (byte)1 : (byte)0;
            chunk.meshedDistances[i] = observed ? voxel.distance : 0f;
        }
        chunk.meshStateInitialized = true;
    }

    private bool IsSurfaceVoxel(Voxel voxel)
    {
        return voxel.weight >= Mathf.Max(1f, minimumMeshWeight) &&
               Mathf.Abs(voxel.distance) <=
               Mathf.Max(0.005f, truncationMeters) * 0.35f;
    }

    [ContextMenu("Toggle CHISEL Shadow Preview")]
    public void TogglePreview()
    {
        SetPreviewVisible(!showPreview);
    }

    public void SetPreviewVisible(bool visible)
    {
        showPreview = visible;
        if (_previewObject != null)
            _previewObject.SetActive(showPreview);
    }

    [ContextMenu("Clear CHISEL Shadow")]
    public void ClearShadow()
    {
        _resetGeneration++;
        _readbackPending = false;
        _observationPreparationPending = false;
        _observationPreparationStage = ObservationPreparationStage.None;
        _observationPreparationCursor = 0;
        _integrationPending = false;
        foreach (Chunk chunk in _chunks.Values)
            DisposeChunkPreview(chunk);
        _chunks.Clear();
        _touchedChunks.Clear();
        _touchedChunkBoundaryMasks.Clear();
        _queuedMeshChunks.Clear();
        _meshChunkQueue.Clear();
        _candidateChunks.Clear();
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
        _surfaceCandidates.Clear();
        _promotedSurfaceCells.Clear();
        _surfaceCandidatesTouched.Clear();
        _surfaceCandidateCleanupKeys.Clear();
        _surfaceCandidateFinalizeKeys.Clear();
        _leftNovelOcclusionChecked.Clear();
        _rightNovelOcclusionChecked.Clear();
        _leftOccludedNovelCells.Clear();
        _rightOccludedNovelCells.Clear();
        _emptyChunkPool.Clear();
        _leftPendingChunkIndex = 0;
        _rightPendingChunkIndex = 0;
        _leftPendingExpected = 0;
        _rightPendingExpected = 0;
        _integrationPairSequence = 0;
        _integrationTotalMilliseconds = 0d;
        AcceptedStereoPairCount = 0;
        DroppedBusyPairCount = 0;
        ReadbackFailureCount = 0;
        AllocationDropCount = 0;
        WeightedVoxelCount = 0;
        SurfaceVoxelCount = 0;
        LastValidSampleCount = 0;
        LastVoxelWriteCount = 0;
        LastCandidateChunkCount = 0;
        LastUpdatedChunkCount = 0;
        LastDepthEdgeRejectedCount = 0;
        LastFreeSpaceConflictCount = 0;
        LastFreeSpaceCorrectionCount = 0;
        LastAssociationRejectedCount = 0;
        LastDirectionRejectedCount = 0;
        LastNovelSurfaceHeldCount = 0;
        LastNovelSurfacePromotedCount = 0;
        LastNovelSurfaceOcclusionRejectedCount = 0;
        EmptyChunkCollectionCount = 0;
        LastPreviewTriangleCount = 0;
        MeshedChunkCount = 0;
        LastIntegrationMilliseconds = 0f;
        LastIntegrationTotalMilliseconds = 0f;
        LastMeshingMilliseconds = 0f;
    }

    private void EnsurePreviewObjects()
    {
        if (_previewObject != null)
            return;
        _previewObject = new GameObject("[ScanCover] CHISEL Sparse TSDF Shadow");
        // CHSL vertices are already emitted in Unity world coordinates. Keeping
        // this root below the moving session/reference transform applies the
        // viewer transform a second time and makes the mesh appear to follow the
        // headset before snapping back. Keep the preview in world space exactly
        // like the production world-space display roots.
        _previewObject.transform.SetParent(null, false);
        _previewObject.transform.SetPositionAndRotation(
            Vector3.zero, Quaternion.identity);
        _previewObject.transform.localScale = Vector3.one;
        // Built-in Unlit/Color is not single-pass-instanced stereo safe on the
        // Quest URP/Vulkan player and can become magenta or diverge per eye.
        // Load the dedicated Resources shader so player stripping cannot drop
        // the only material used by this reflection-owned optional preview.
        Shader shader = Resources.Load<Shader>("ChiselSparsePreview");
        if (shader == null)
            shader = Shader.Find("Genesis/ChiselSparsePreview");
        if (shader != null)
        {
            _previewMaterial = new Material(shader)
            {
                name = "ScanCover CHISEL Shadow Material",
                enableInstancing = true
            };
            if (_previewMaterial.HasProperty("_BaseColor"))
                _previewMaterial.SetColor("_BaseColor", previewColor);
        }
        else
        {
            // Never expose Unity's magenta error material in-headset. The
            // shadow TSDF keeps accumulating; only this optional view stays off.
            showPreview = false;
            Debug.LogError("CHSL preview shader missing; preview disabled safely.");
        }
        _previewObject.SetActive(showPreview);
    }

    private void RebuildPreviewMesh()
    {
        if (_previewObject == null || _previewMaterial == null ||
            _meshChunkQueue.Count == 0)
            return;
        double start = Time.realtimeSinceStartupAsDouble;
        double timeBudgetSeconds = Mathf.Max(
            0.0005f, previewMeshingBudgetMilliseconds * 0.001f);
        int chunkBudget = Mathf.Max(1, previewChunksPerFrame);
        int processed = 0;
        int attemptsRemaining = _meshChunkQueue.Count;
        while (processed < chunkBudget && _meshChunkQueue.Count > 0 &&
               attemptsRemaining-- > 0)
        {
            Vector3Int dirty = _meshChunkQueue.Dequeue();
            // A queued result from the previous pair may overlap a chunk now
            // being fused. Keep draining unrelated chunks, but do not publish
            // a left-eye/part-pair intermediate for the active chunk.
            if (_integrationPending && _touchedChunks.Contains(dirty))
            {
                _meshChunkQueue.Enqueue(dirty);
                continue;
            }
            _queuedMeshChunks.Remove(dirty);
            if (_chunks.TryGetValue(dirty, out Chunk chunk))
                RebuildChunkMesh(dirty, chunk);
            processed++;
            // A chunk is atomic: never publish a half-written Unity Mesh. Stop
            // after the completed chunk once this frame's budget is consumed.
            if (Time.realtimeSinceStartupAsDouble - start >= timeBudgetSeconds)
                break;
        }

        int triangleCount = 0;
        int meshCount = 0;
        foreach (Chunk chunk in _chunks.Values)
        {
            triangleCount += chunk.previewTriangleCount;
            if (chunk.previewTriangleCount > 0)
                meshCount++;
        }
        LastPreviewTriangleCount = triangleCount;
        MeshedChunkCount = meshCount;
        LastMeshingMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
    }

    private void RebuildChunkMesh(Vector3Int chunkId, Chunk chunk)
    {
        _meshVertices.Clear();
        _meshLineIndices.Clear();
        _meshUniqueLineKeys.Clear();
        Array.Fill(_meshXEdgeVertexIndices, -1);
        Array.Fill(_meshYEdgeVertexIndices, -1);
        Array.Fill(_meshZEdgeVertexIndices, -1);
        CacheMeshVoxelHalo(chunkId);
        Vector3Int chunkOrigin = chunkId * ChunkSize;
        float voxelSize = Mathf.Max(0.005f, voxelSizeMeters);
        int triangleCount = 0;

        // This is the same cell ownership as OpenChisel GenerateMesh:
        // 15^3 interior cells plus the max-X, max-Y and max-Z border cells.
        // Iterating all 16^3 minimum corners once is exactly that union.
        for (int z = 0; z < ChunkSize; z++)
        for (int y = 0; y < ChunkSize; y++)
        for (int x = 0; x < ChunkSize; x++)
        {
            Vector3Int cell = chunkOrigin + new Vector3Int(x, y, z);
            int configuration = 0;
            bool allObserved = true;
            for (int corner = 0; corner < 8; corner++)
            {
                Vector3Int offset = CubeOffsets[corner];
                int sampleIndex = MeshSampleIndex(
                    x + offset.x, y + offset.y, z + offset.z);
                if (_meshSampleObserved[sampleIndex] == 0)
                {
                    allObserved = false;
                    break;
                }
                float distance = _meshSampleDistances[sampleIndex];
                _meshCornerDistances[corner] = distance;
                _meshCornerPositions[corner] = new Vector3(
                    (cell.x + offset.x + 0.5f) * voxelSize,
                    (cell.y + offset.y + 0.5f) * voxelSize,
                    (cell.z + offset.z + 0.5f) * voxelSize);
                if (distance < 0f)
                    configuration |= 1 << corner;
            }
            if (!allObserved || configuration == 0 || configuration == 255)
                continue;

            for (int edge = 0; edge < 12; edge++)
            {
                _meshCellEdgeVertexIndices[edge] = -1;
                int a = EdgeIndexPairs[edge, 0];
                int b = EdgeIndexPairs[edge, 1];
                bool crosses =
                    (_meshCornerDistances[a] < 0f &&
                     _meshCornerDistances[b] >= 0f) ||
                    (_meshCornerDistances[a] >= 0f &&
                     _meshCornerDistances[b] < 0f);
                if (crosses)
                    _meshCellEdgeVertexIndices[edge] =
                        GetOrCreateMeshEdgeVertex(x, y, z, edge);
            }

            for (int tableColumn = 0;
                 tableColumn < ScanCoverPaperDmcTables.TriangleEntriesPerIndex;
                 tableColumn += 3)
            {
                int edge0 = ScanCoverPaperDmcTables.TriangleEdge(
                    configuration, tableColumn);
                if (edge0 < 0)
                    break;
                int edge1 = ScanCoverPaperDmcTables.TriangleEdge(
                    configuration, tableColumn + 1);
                int edge2 = ScanCoverPaperDmcTables.TriangleEdge(
                    configuration, tableColumn + 2);

                // OpenChisel uses table edges 2,1,0 for triangle winding. The
                // preview is lines, but retaining that order keeps its indexed
                // topology equivalent if filled rendering is added later.
                int vertex0 = _meshCellEdgeVertexIndices[edge2];
                int vertex1 = _meshCellEdgeVertexIndices[edge1];
                int vertex2 = _meshCellEdgeVertexIndices[edge0];
                if (vertex0 < 0 || vertex1 < 0 || vertex2 < 0)
                    continue;
                AddUniqueMeshLine(vertex0, vertex1);
                AddUniqueMeshLine(vertex1, vertex2);
                AddUniqueMeshLine(vertex2, vertex0);
                triangleCount++;
            }
        }

        if (triangleCount == 0 || _meshVertices.Count == 0)
        {
            chunk.previewTriangleCount = 0;
            if (chunk.previewObject != null)
                chunk.previewObject.SetActive(false);
            CaptureMeshedVoxelState(chunk, Mathf.Max(1f, minimumMeshWeight));
            return;
        }

        EnsureChunkPreview(chunkId, chunk);
        // Build only into the hidden back buffer. The MeshFilter keeps reading
        // the old front buffer until the replacement is fully uploaded, then a
        // single reference swap publishes the completed chunk atomically.
        Mesh staging = chunk.previewStagingMesh;
        staging.Clear(false);
        staging.SetVertices(_meshVertices);
        staging.SetIndices(
            _meshLineIndices, MeshTopology.Lines, 0, false);
        // Vertices are already in world coordinates and the preview root is
        // identity. The chunk's deterministic AABB avoids a second full scan
        // of every emitted vertex on the main thread.
        float chunkWorldSize = ChunkSize * voxelSize;
        Vector3 boundsCenter = new Vector3(
            (chunkOrigin.x + ChunkSize * 0.5f + 0.5f) * voxelSize,
            (chunkOrigin.y + ChunkSize * 0.5f + 0.5f) * voxelSize,
            (chunkOrigin.z + ChunkSize * 0.5f + 0.5f) * voxelSize);
        staging.bounds = new Bounds(
            boundsCenter, Vector3.one * (chunkWorldSize + voxelSize));
        staging.UploadMeshData(false);
        chunk.previewFilter.sharedMesh = staging;
        chunk.previewStagingMesh = chunk.previewMesh;
        chunk.previewMesh = staging;
        chunk.previewTriangleCount = triangleCount;
        CaptureMeshedVoxelState(chunk, Mathf.Max(1f, minimumMeshWeight));
        chunk.previewObject.SetActive(showPreview);
    }

    private int GetOrCreateMeshEdgeVertex(
        int cellX,
        int cellY,
        int cellZ,
        int edge)
    {
        int a = EdgeIndexPairs[edge, 0];
        int b = EdgeIndexPairs[edge, 1];
        Vector3Int offsetA = CubeOffsets[a];
        Vector3Int offsetB = CubeOffsets[b];
        int sampleX = cellX + Mathf.Min(offsetA.x, offsetB.x);
        int sampleY = cellY + Mathf.Min(offsetA.y, offsetB.y);
        int sampleZ = cellZ + Mathf.Min(offsetA.z, offsetB.z);

        int[] cache;
        int cacheIndex;
        if (offsetA.x != offsetB.x)
        {
            cache = _meshXEdgeVertexIndices;
            cacheIndex = sampleX + ChunkSize *
                (sampleY + MeshSampleSize * sampleZ);
        }
        else if (offsetA.y != offsetB.y)
        {
            cache = _meshYEdgeVertexIndices;
            cacheIndex = sampleX + MeshSampleSize *
                (sampleY + ChunkSize * sampleZ);
        }
        else
        {
            cache = _meshZEdgeVertexIndices;
            cacheIndex = sampleX + MeshSampleSize *
                (sampleY + MeshSampleSize * sampleZ);
        }

        int vertexIndex = cache[cacheIndex];
        if (vertexIndex >= 0)
            return vertexIndex;

        vertexIndex = _meshVertices.Count;
        _meshVertices.Add(Interpolate(
            a, b, _meshCornerPositions, _meshCornerDistances));
        cache[cacheIndex] = vertexIndex;
        return vertexIndex;
    }

    private void AddUniqueMeshLine(int first, int second)
    {
        if (first == second)
            return;
        uint minimum = (uint)Mathf.Min(first, second);
        uint maximum = (uint)Mathf.Max(first, second);
        ulong key = ((ulong)minimum << 32) | maximum;
        if (!_meshUniqueLineKeys.Add(key))
            return;
        _meshLineIndices.Add(first);
        _meshLineIndices.Add(second);
    }

    private void CacheMeshVoxelHalo(Vector3Int chunkId)
    {
        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            int neighbourIndex = dx + (dy << 1) + (dz << 2);
            _chunks.TryGetValue(
                chunkId + new Vector3Int(dx, dy, dz),
                out _meshNeighbourChunks[neighbourIndex]);
        }

        float requiredWeight = Mathf.Max(1f, minimumMeshWeight);
        for (int z = 0; z < MeshSampleSize; z++)
        for (int y = 0; y < MeshSampleSize; y++)
        for (int x = 0; x < MeshSampleSize; x++)
        {
            int dx = x == ChunkSize ? 1 : 0;
            int dy = y == ChunkSize ? 1 : 0;
            int dz = z == ChunkSize ? 1 : 0;
            Chunk source = _meshNeighbourChunks[dx + (dy << 1) + (dz << 2)];
            int sampleIndex = MeshSampleIndex(x, y, z);
            if (source == null)
            {
                _meshSampleObserved[sampleIndex] = 0;
                _meshSampleDistances[sampleIndex] = 0f;
                continue;
            }

            int lx = dx == 0 ? x : 0;
            int ly = dy == 0 ? y : 0;
            int lz = dz == 0 ? z : 0;
            Voxel voxel = source.voxels[
                lx + ChunkSize * (ly + ChunkSize * lz)];
            bool observed = voxel.weight >= requiredWeight;
            _meshSampleObserved[sampleIndex] = observed ? (byte)1 : (byte)0;
            _meshSampleDistances[sampleIndex] = observed ? voxel.distance : 0f;
        }
    }

    private static int MeshSampleIndex(int x, int y, int z)
    {
        return x + MeshSampleSize * (y + MeshSampleSize * z);
    }

    private void EnsureChunkPreview(Vector3Int chunkId, Chunk chunk)
    {
        if (chunk.previewObject != null)
            return;
        chunk.previewObject = new GameObject(
            $"CHSL MC Chunk {chunkId.x},{chunkId.y},{chunkId.z}");
        chunk.previewObject.transform.SetParent(_previewObject.transform, false);
        MeshFilter filter = chunk.previewObject.AddComponent<MeshFilter>();
        MeshRenderer renderer = chunk.previewObject.AddComponent<MeshRenderer>();
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.sharedMaterial = _previewMaterial;
        chunk.previewFilter = filter;
        chunk.previewMesh = CreateChunkPreviewMesh(chunkId, "front");
        chunk.previewStagingMesh = CreateChunkPreviewMesh(chunkId, "back");
        chunk.previewMesh.MarkDynamic();
        chunk.previewStagingMesh.MarkDynamic();
        filter.sharedMesh = chunk.previewMesh;
    }

    private static Mesh CreateChunkPreviewMesh(Vector3Int chunkId, string role)
    {
        return new Mesh
        {
            name = $"CHSL MC {chunkId.x},{chunkId.y},{chunkId.z} {role}",
            indexFormat = IndexFormat.UInt32
        };
    }

    private static Vector3 Interpolate(
        int a,
        int b,
        Vector3[] positions,
        float[] distances)
    {
        float denominator = distances[a] - distances[b];
        float alpha = Mathf.Abs(denominator) > 0.000001f
            ? Mathf.Clamp01(distances[a] / denominator)
            : 0.5f;
        return Vector3.Lerp(positions[a], positions[b], alpha);
    }

    private static void DisposeChunkPreview(Chunk chunk)
    {
        if (chunk.previewMesh != null)
            Destroy(chunk.previewMesh);
        if (chunk.previewStagingMesh != null)
            Destroy(chunk.previewStagingMesh);
        if (chunk.previewObject != null)
            Destroy(chunk.previewObject);
        chunk.previewFilter = null;
        chunk.previewMesh = null;
        chunk.previewStagingMesh = null;
        chunk.previewObject = null;
        chunk.previewTriangleCount = 0;
    }

    private bool TryGetVoxel(Vector3Int voxelId, out Voxel voxel)
    {
        if (!TryGetVoxelRaw(voxelId, out voxel))
            return false;
        // Weak one-frame evidence must never become visible topology. This is
        // intentionally the same threshold used by SurfaceVoxelCount.
        return voxel.weight >= Mathf.Max(1f, minimumMeshWeight);
    }

    private bool TryGetVoxelRaw(Vector3Int voxelId, out Voxel voxel)
    {
        Vector3Int chunkId = new Vector3Int(
            FloorDiv(voxelId.x, ChunkSize),
            FloorDiv(voxelId.y, ChunkSize),
            FloorDiv(voxelId.z, ChunkSize));
        if (!_chunks.TryGetValue(chunkId, out Chunk chunk))
        {
            voxel = default;
            return false;
        }
        int lx = PositiveMod(voxelId.x, ChunkSize);
        int ly = PositiveMod(voxelId.y, ChunkSize);
        int lz = PositiveMod(voxelId.z, ChunkSize);
        voxel = chunk.voxels[lx + ChunkSize * (ly + ChunkSize * lz)];
        return voxel.weight > 0f;
    }

    private Vector3 VoxelWorldPosition(Vector3Int voxelId)
    {
        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        return new Vector3(
            (voxelId.x + 0.5f) * voxel,
            (voxelId.y + 0.5f) * voxel,
            (voxelId.z + 0.5f) * voxel);
    }

    private static int FloorDiv(int value, int divisor)
    {
        int quotient = value / divisor;
        int remainder = value % divisor;
        return remainder < 0 ? quotient - 1 : quotient;
    }

    private static int PositiveMod(int value, int modulus)
    {
        int result = value % modulus;
        return result < 0 ? result + modulus : result;
    }

    private static bool Finite(Vector3 value)
    {
        return float.IsFinite(value.x) &&
               float.IsFinite(value.y) &&
               float.IsFinite(value.z);
    }

    private static bool Finite(Matrix4x4 value)
    {
        for (int row = 0; row < 4; row++)
        for (int column = 0; column < 4; column++)
        {
            if (!float.IsFinite(value[row, column]))
                return false;
        }
        return true;
    }

    private void OnDestroy()
    {
        DisposeColorCache(ref _leftPendingPositions);
        DisposeColorCache(ref _leftPendingMeta);
        DisposeColorCache(ref _rightPendingPositions);
        DisposeColorCache(ref _rightPendingMeta);
        foreach (Chunk chunk in _chunks.Values)
            DisposeChunkPreview(chunk);
        if (_previewMaterial != null)
            Destroy(_previewMaterial);
        if (_previewObject != null)
            Destroy(_previewObject);
    }

    private void OnDisable()
    {
        _resetGeneration++;
        _readbackPending = false;
        _observationPreparationPending = false;
        _observationPreparationStage = ObservationPreparationStage.None;
        _observationPreparationCursor = 0;
        _integrationPending = false;
        _candidateChunks.Clear();
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
        _surfaceCandidatesTouched.Clear();
        _surfaceCandidateCleanupKeys.Clear();
        _surfaceCandidateFinalizeKeys.Clear();
        _touchedChunks.Clear();
        _touchedChunkBoundaryMasks.Clear();
        if (_previewObject != null)
            _previewObject.SetActive(false);
    }
}

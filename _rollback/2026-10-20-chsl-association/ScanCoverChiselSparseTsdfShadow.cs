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

    private struct Voxel
    {
        public float distance;
        public float weight;
        public int lastSupportPair;
        public int lastConflictPair;
        public byte freeSpaceConflictStreak;
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
        public int lastTouchedFrame;
        public GameObject previewObject;
        public Mesh previewMesh;
        public int previewTriangleCount;
    }

    [Header("Independent sparse TSDF")]
    [SerializeField] private bool routeEnabled = true;
    [SerializeField, Range(0.03f, 0.10f)] private float voxelSizeMeters = 0.05f;
    [SerializeField, Range(0.05f, 0.30f)] private float truncationMeters = 0.15f;
    [SerializeField, Range(0.1f, 3f)] private float captureIntervalSeconds = 0.5f;
    [SerializeField, Range(0.2f, 2f)] private float minimumDepthMeters = 0.30f;
    [SerializeField, Range(1f, 8f)] private float maximumDepthMeters = 5.5f;
    [SerializeField, Range(1f, 8f)] private float minimumMeshWeight = 2f;

    [Header("Depth discontinuity protection")]
    [SerializeField] private bool protectDepthDiscontinuities = true;
    [Tooltip("A neighbouring depth must agree by this absolute/relative tolerance to support fusion.")]
    [SerializeField, Range(0.02f, 0.30f)] private float depthEdgeAbsoluteMeters = 0.10f;
    [SerializeField, Range(0.005f, 0.15f)] private float depthEdgeRelativeFraction = 0.035f;
    [SerializeField, Range(1, 8)] private int minimumConsistentDepthNeighbours = 4;

    [Header("Bounded TSDF and confirmed free space")]
    [SerializeField] private bool enableConfirmedFreeSpace = true;
    [SerializeField, Range(2f, 64f)] private float maximumVoxelWeight = 16f;
    [Tooltip("A voxel this far in front of the observed surface is contradictory free space.")]
    [SerializeField, Range(0.03f, 0.30f)] private float freeSpaceConflictMeters = 0.10f;
    [SerializeField, Range(2, 8)] private int freeSpaceConfirmationPairs = 3;
    [SerializeField, Range(0.25f, 4f)] private float freeSpaceCorrectionWeight = 2f;
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
    [Tooltip("FIFO mesh work per frame. This limits latency without dropping dirty chunks.")]
    [SerializeField, Range(1, 24)] private int previewChunksPerFrame = 8;

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
    public int EmptyChunkCollectionCount { get; private set; }
    public int LastPreviewTriangleCount { get; private set; }
    public int MeshedChunkCount { get; private set; }
    public int PendingMeshChunkCount => _queuedMeshChunks.Count;
    public float LastIntegrationMilliseconds { get; private set; }
    public float LastIntegrationTotalMilliseconds { get; private set; }
    public float LastMeshingMilliseconds { get; private set; }
    public bool PreviewVisible => showPreview;
    public bool HasPendingReadback => _readbackPending || _integrationPending;

    public string CompactStatus =>
        $"CHSL pair={AcceptedStereoPairCount} block={ActiveChunkCount} " +
        $"vox={WeightedVoxelCount} surf={SurfaceVoxelCount} " +
        $"int={LastIntegrationTotalMilliseconds:0.0}/{LastIntegrationMilliseconds:0.0}ms\n" +
        $"CHSL guard edge={LastDepthEdgeRejectedCount} " +
        $"free={LastFreeSpaceCorrectionCount}/{LastFreeSpaceConflictCount} " +
        $"proj={LastUpdatedChunkCount}/{LastCandidateChunkCount} " +
        $"gc={EmptyChunkCollectionCount}";

    private readonly Dictionary<Vector3Int, Chunk> _chunks =
        new Dictionary<Vector3Int, Chunk>(256);
    private readonly HashSet<Vector3Int> _touchedChunks =
        new HashSet<Vector3Int>();
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
    private readonly Stack<Chunk> _emptyChunkPool = new Stack<Chunk>(64);
    private AsyncGPUReadbackRequest _leftPositionRequest;
    private AsyncGPUReadbackRequest _leftMetaRequest;
    private AsyncGPUReadbackRequest _rightPositionRequest;
    private AsyncGPUReadbackRequest _rightMetaRequest;
    private bool _readbackPending;
    private int _resetGeneration;
    private int _pendingGeneration;
    private int _pendingFrameOrdinal;
    private int _pendingLeftWidth;
    private int _pendingLeftHeight;
    private int _pendingRightWidth;
    private int _pendingRightHeight;
    private Vector3 _pendingLeftEye;
    private Vector3 _pendingRightEye;
    private Matrix4x4 _pendingLeftProjection;
    private Matrix4x4 _pendingLeftView;
    private Matrix4x4 _pendingRightProjection;
    private Matrix4x4 _pendingRightView;
    private float[] _leftObservedDepth;
    private float[] _rightObservedDepth;
    private byte[] _leftDepthContinuity;
    private byte[] _rightDepthContinuity;
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
        Matrix4x4 leftProjection,
        Matrix4x4 leftView,
        RenderTexture rightWorldPosition,
        RenderTexture rightObservationMeta,
        Vector3 rightEye,
        Matrix4x4 rightProjection,
        Matrix4x4 rightView)
    {
        if (!routeEnabled || !isActiveAndEnabled)
            return;
        if (_readbackPending || _integrationPending)
        {
            DroppedBusyPairCount++;
            return;
        }
        if (Time.unscaledTime < _nextCaptureTime)
            return;
        if (leftWorldPosition == null || leftObservationMeta == null ||
            rightWorldPosition == null || rightObservationMeta == null ||
            !Finite(leftEye) || !Finite(rightEye) ||
            !Finite(leftProjection) || !Finite(leftView) ||
            !Finite(rightProjection) || !Finite(rightView))
            return;

        _pendingFrameOrdinal = frameOrdinal;
        _pendingLeftWidth = leftWorldPosition.width;
        _pendingLeftHeight = leftWorldPosition.height;
        _pendingRightWidth = rightWorldPosition.width;
        _pendingRightHeight = rightWorldPosition.height;
        _pendingLeftEye = leftEye;
        _pendingRightEye = rightEye;
        _pendingLeftProjection = leftProjection;
        _pendingLeftView = leftView;
        _pendingRightProjection = rightProjection;
        _pendingRightView = rightView;
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
        bool preparedThisFrame = false;
        if (_readbackPending)
        {
            CompletePendingReadbackWhenReady();
            preparedThisFrame = !_readbackPending && _integrationPending;
        }

        if (_integrationPending && !preparedThisFrame)
            ProcessIntegrationSlice();

        EnsurePreviewObjects();
        if (_previewObject != null)
            _previewObject.SetActive(showPreview);
        if (!_integrationPending && showPreview && _meshChunkQueue.Count > 0)
            RebuildPreviewMesh();
    }

    private void CompletePendingReadbackWhenReady()
    {
        if (!_leftPositionRequest.done || !_leftMetaRequest.done ||
            !_rightPositionRequest.done || !_rightMetaRequest.done)
            return;

        _readbackPending = false;
        if (_pendingGeneration != _resetGeneration)
            return;
        if (_leftPositionRequest.hasError || _leftMetaRequest.hasError ||
            _rightPositionRequest.hasError || _rightMetaRequest.hasError)
        {
            ReadbackFailureCount++;
            return;
        }

        double start = Time.realtimeSinceStartupAsDouble;
        LastValidSampleCount = 0;
        LastVoxelWriteCount = 0;
        LastCandidateChunkCount = 0;
        LastUpdatedChunkCount = 0;
        LastDepthEdgeRejectedCount = 0;
        LastFreeSpaceConflictCount = 0;
        LastFreeSpaceCorrectionCount = 0;
        LastIntegrationMilliseconds = 0f;
        LastIntegrationTotalMilliseconds = 0f;
        _touchedChunks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _integrationPairSequence++;
        PrepareEyeObservation(
            _leftPositionRequest.GetData<Color>(),
            _leftMetaRequest.GetData<Color>(),
            _pendingLeftWidth,
            _pendingLeftHeight,
            _pendingLeftEye,
            _pendingLeftProjection,
            _pendingLeftView,
            ref _leftObservedDepth,
            ref _leftDepthContinuity,
            _leftPendingChunks,
            out _leftPendingExpected,
            out _leftPendingWorldToClip);
        PrepareEyeObservation(
            _rightPositionRequest.GetData<Color>(),
            _rightMetaRequest.GetData<Color>(),
            _pendingRightWidth,
            _pendingRightHeight,
            _pendingRightEye,
            _pendingRightProjection,
            _pendingRightView,
            ref _rightObservedDepth,
            ref _rightDepthContinuity,
            _rightPendingChunks,
            out _rightPendingExpected,
            out _rightPendingWorldToClip);
        _leftPendingChunkIndex = 0;
        _rightPendingChunkIndex = 0;
        _integrationTotalMilliseconds =
            (Time.realtimeSinceStartupAsDouble - start) * 1000d;
        LastIntegrationMilliseconds = (float)_integrationTotalMilliseconds;
        _integrationPending = true;
    }

    private void PrepareEyeObservation(
        NativeArray<Color> positions,
        NativeArray<Color> meta,
        int width,
        int height,
        Vector3 eye,
        Matrix4x4 projection,
        Matrix4x4 view,
        ref float[] observedDepths,
        ref byte[] continuity,
        List<Vector3Int> pendingChunks,
        out int expected,
        out Matrix4x4 worldToClip)
    {
        int count = Mathf.Min(positions.Length, meta.Length);
        expected = Mathf.Min(count, Mathf.Max(0, width * height));
        worldToClip = projection * view;
        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        float truncation = Mathf.Max(voxel, truncationMeters);
        float diagonalAllowance = 2f * Mathf.Sqrt(3f) * voxel;
        float updateBand = truncation + diagonalAllowance;
        float chunkWorldSize = ChunkSize * voxel;
        EnsureDepthCache(ref observedDepths, expected);
        EnsureByteCache(ref continuity, expected);

        for (int index = 0; index < expected; index++)
        {
            observedDepths[index] = 0f;
            if (meta[index].r < 0.5f)
                continue;
            Color encoded = positions[index];
            Vector3 surface = new Vector3(encoded.r, encoded.g, encoded.b);
            if (!Finite(surface))
                continue;
            float observedDepth = -(
                view.m20 * surface.x + view.m21 * surface.y +
                view.m22 * surface.z + view.m23);
            if (!float.IsFinite(observedDepth) ||
                observedDepth < minimumDepthMeters ||
                observedDepth > maximumDepthMeters)
                continue;

            observedDepths[index] = observedDepth;
            LastValidSampleCount++;
        }

        BuildDepthContinuityMask(
            observedDepths, continuity, width, height, expected);

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
            float worldRadius = updateBand * Mathf.Max(1f, rayScale) +
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
                    eye, surface, observedDepths[index], chunkWorldSize);
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
                    chunkId, _leftObservedDepth, _leftDepthContinuity,
                    _leftPendingExpected, _pendingLeftWidth,
                    _pendingLeftHeight, _leftPendingWorldToClip,
                    _pendingLeftView);
                hasChunk = true;
            }
            else if (_rightPendingChunkIndex < _rightPendingChunks.Count)
            {
                Vector3Int chunkId =
                    _rightPendingChunks[_rightPendingChunkIndex++];
                ProcessPendingChunk(
                    chunkId, _rightObservedDepth, _rightDepthContinuity,
                    _rightPendingExpected, _pendingRightWidth,
                    _pendingRightHeight, _rightPendingWorldToClip,
                    _pendingRightView);
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
        byte[] continuity,
        int expected,
        int width,
        int height,
        Matrix4x4 worldToClip,
        Matrix4x4 view)
    {
        bool chunkNew = !_chunks.TryGetValue(chunkId, out Chunk chunk);
        if (chunkNew)
        {
            chunk = AcquireEmptyChunk();
            _chunks.Add(chunkId, chunk);
        }

        float voxel = Mathf.Max(0.005f, voxelSizeMeters);
        float truncation = Mathf.Max(voxel, truncationMeters);
        float updateBand = truncation + 2f * Mathf.Sqrt(3f) * voxel;
        bool updatedChunk = IntegrateProjectedChunk(
            chunkId, chunk, observedDepths, continuity, expected,
            width, height, worldToClip, view, updateBand, voxel);
        if (updatedChunk)
        {
            LastUpdatedChunkCount++;
            chunk.lastTouchedFrame = _pendingFrameOrdinal;
            _touchedChunks.Add(chunkId);
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
        byte[] continuity,
        int expected,
        int width,
        int height,
        Matrix4x4 worldToClip,
        Matrix4x4 view,
        float updateBand,
        float voxel)
    {
        Vector3Int chunkOrigin = chunkId * ChunkSize;
        bool updatedChunk = false;
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
            float voxelDepth = -(
                view.m20 * worldX + view.m21 * worldY +
                view.m22 * worldZ + view.m23);
            if (!float.IsFinite(voxelDepth) || voxelDepth <= 0f)
                continue;

            float clipX = worldToClip.m00 * worldX +
                          worldToClip.m01 * worldY +
                          worldToClip.m02 * worldZ + worldToClip.m03;
            float clipY = worldToClip.m10 * worldX +
                          worldToClip.m11 * worldY +
                          worldToClip.m12 * worldZ + worldToClip.m13;
            float clipW = worldToClip.m30 * worldX +
                          worldToClip.m31 * worldY +
                          worldToClip.m32 * worldZ + worldToClip.m33;
            if (!float.IsFinite(clipW) || Mathf.Abs(clipW) < 0.000001f)
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
                surfaceDistance >= Mathf.Max(voxel, freeSpaceConflictMeters) &&
                old.weight > 0f &&
                old.distance <= freeSpaceConflictMeters * 0.5f)
            {
                RecordFreeSpaceConflict(chunkId, chunk, localIndex);
                continue;
            }
            if (Mathf.Abs(surfaceDistance) >= updateBand)
                continue;

            RecordSurfaceSupport(chunk, localIndex);
            IntegrateVoxel(
                chunk, localIndex,
                Mathf.Clamp(surfaceDistance, -truncationMeters,
                    truncationMeters),
                1f);
            updatedChunk = true;
        }
        return updatedChunk;
    }

    private void RecordSurfaceSupport(Chunk chunk, int localIndex)
    {
        Voxel voxel = chunk.voxels[localIndex];
        voxel.lastSupportPair = _integrationPairSequence;
        voxel.freeSpaceConflictStreak = 0;
        chunk.voxels[localIndex] = voxel;
    }

    private void RecordFreeSpaceConflict(
        Vector3Int chunkId, Chunk chunk, int localIndex)
    {
        Voxel voxel = chunk.voxels[localIndex];
        if (voxel.lastConflictPair != _integrationPairSequence)
        {
            voxel.freeSpaceConflictStreak =
                voxel.lastConflictPair == _integrationPairSequence - 1
                    ? (byte)Mathf.Min(255,
                        voxel.freeSpaceConflictStreak + 1)
                    : (byte)1;
            voxel.lastConflictPair = _integrationPairSequence;
            chunk.voxels[localIndex] = voxel;
        }

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
                chunk.voxels[address.localIndex] = voxel;
                continue;
            }

            LastFreeSpaceConflictCount++;
            if (voxel.freeSpaceConflictStreak < requiredPairs)
                continue;

            IntegrateVoxel(
                chunk, address.localIndex, correctionDistance,
                Mathf.Max(0.25f, freeSpaceCorrectionWeight));
            chunk.lastTouchedFrame = _pendingFrameOrdinal;
            _touchedChunks.Add(address.chunkId);
            LastFreeSpaceCorrectionCount++;
        }
    }

    private void IntegrateVoxel(
        Chunk chunk,
        int localIndex,
        float signedDistance,
        float observationWeight)
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
        chunk.voxels[localIndex] = updated;
        bool isSurface = IsSurfaceVoxel(updated);
        if (wasSurface != isSurface)
            SurfaceVoxelCount += isSurface ? 1 : -1;
        LastVoxelWriteCount++;
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
        if (_emptyChunkPool.Count < 64)
            _emptyChunkPool.Push(chunk);
        EmptyChunkCollectionCount++;
    }

    private Chunk AcquireEmptyChunk()
    {
        return _emptyChunkPool.Count > 0 ? _emptyChunkPool.Pop() : new Chunk();
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

    private void MarkTouchedChunkMeshesDirty()
    {
        // A mesh cell owned by a chunk only reads that chunk and its positive
        // X/Y/Z neighbours. Therefore a changed chunk can affect its own mesh
        // and the seven negative-direction owners, not all 27 neighbours.
        foreach (Vector3Int touched in _touchedChunks)
        {
            for (int z = -1; z <= 0; z++)
            for (int y = -1; y <= 0; y++)
            for (int x = -1; x <= 0; x++)
            {
                Vector3Int dirty = touched + new Vector3Int(x, y, z);
                if (_queuedMeshChunks.Add(dirty))
                    _meshChunkQueue.Enqueue(dirty);
            }
        }
        _touchedChunks.Clear();
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
        _integrationPending = false;
        foreach (Chunk chunk in _chunks.Values)
            DisposeChunkPreview(chunk);
        _chunks.Clear();
        _touchedChunks.Clear();
        _queuedMeshChunks.Clear();
        _meshChunkQueue.Clear();
        _candidateChunks.Clear();
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
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
        _previewObject.transform.SetParent(transform, false);
        _previewObject.transform.localPosition = Vector3.zero;
        _previewObject.transform.localRotation = Quaternion.identity;
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
        int budget = Mathf.Max(1, previewChunksPerFrame);
        for (int i = 0; i < budget && _meshChunkQueue.Count > 0; i++)
        {
            Vector3Int dirty = _meshChunkQueue.Dequeue();
            _queuedMeshChunks.Remove(dirty);
            if (_chunks.TryGetValue(dirty, out Chunk chunk))
                RebuildChunkMesh(dirty, chunk);
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
        var vertices = new List<Vector3>(4096);
        var lineIndices = new List<int>(8192);
        var cornerPositions = new Vector3[8];
        var cornerDistances = new float[8];
        var edgeVertices = new Vector3[12];
        Vector3Int chunkOrigin = chunkId * ChunkSize;

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
                Vector3Int voxelId = cell + CubeOffsets[corner];
                if (!TryGetVoxel(voxelId, out Voxel voxel))
                {
                    allObserved = false;
                    break;
                }
                cornerDistances[corner] = voxel.distance;
                cornerPositions[corner] = VoxelWorldPosition(voxelId);
                if (voxel.distance < 0f)
                    configuration |= 1 << corner;
            }
            if (!allObserved || configuration == 0 || configuration == 255)
                continue;

            for (int edge = 0; edge < 12; edge++)
            {
                int a = EdgeIndexPairs[edge, 0];
                int b = EdgeIndexPairs[edge, 1];
                bool crosses =
                    (cornerDistances[a] < 0f && cornerDistances[b] >= 0f) ||
                    (cornerDistances[a] >= 0f && cornerDistances[b] < 0f);
                if (crosses)
                    edgeVertices[edge] = Interpolate(
                        a, b, cornerPositions, cornerDistances);
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

                // OpenChisel appends table edges 2,1,0 to preserve winding.
                int startIndex = vertices.Count;
                vertices.Add(edgeVertices[edge2]);
                vertices.Add(edgeVertices[edge1]);
                vertices.Add(edgeVertices[edge0]);
                lineIndices.Add(startIndex);
                lineIndices.Add(startIndex + 1);
                lineIndices.Add(startIndex + 1);
                lineIndices.Add(startIndex + 2);
                lineIndices.Add(startIndex + 2);
                lineIndices.Add(startIndex);
            }
        }

        if (vertices.Count == 0)
        {
            chunk.previewTriangleCount = 0;
            if (chunk.previewMesh != null)
                chunk.previewMesh.Clear();
            if (chunk.previewObject != null)
                chunk.previewObject.SetActive(false);
            return;
        }

        EnsureChunkPreview(chunkId, chunk);
        chunk.previewMesh.Clear();
        chunk.previewMesh.SetVertices(vertices);
        chunk.previewMesh.SetIndices(lineIndices, MeshTopology.Lines, 0, true);
        chunk.previewMesh.RecalculateBounds();
        chunk.previewTriangleCount = vertices.Count / 3;
        chunk.previewObject.SetActive(showPreview);
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
        chunk.previewMesh = new Mesh
        {
            name = $"CHSL MC {chunkId.x},{chunkId.y},{chunkId.z}",
            indexFormat = IndexFormat.UInt32
        };
        filter.sharedMesh = chunk.previewMesh;
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
        if (chunk.previewObject != null)
            Destroy(chunk.previewObject);
        chunk.previewMesh = null;
        chunk.previewObject = null;
        chunk.previewTriangleCount = 0;
    }

    private bool TryGetVoxel(Vector3Int voxelId, out Voxel voxel)
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
        foreach (Chunk chunk in _chunks.Values)
        {
            if (chunk.previewMesh != null)
                Destroy(chunk.previewMesh);
        }
        if (_previewMaterial != null)
            Destroy(_previewMaterial);
        if (_previewObject != null)
            Destroy(_previewObject);
    }

    private void OnDisable()
    {
        _resetGeneration++;
        _readbackPending = false;
        _integrationPending = false;
        _leftPendingChunks.Clear();
        _rightPendingChunks.Clear();
        _freeSpaceEvidenceVoxels.Clear();
        _freeSpaceEvidenceSet.Clear();
        _touchedChunks.Clear();
        if (_previewObject != null)
            _previewObject.SetActive(false);
    }
}

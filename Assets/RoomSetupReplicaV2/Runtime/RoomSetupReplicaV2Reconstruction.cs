using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Genesis.RoomScan;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Clean-room behavioral replica of the recovered Room Setup reconstruction
/// path. It shares only the accepted Quest depth packet; it does not read or
/// mutate ScanCover TSDF, admission, cleanup, extraction or publication state.
/// </summary>
[DefaultExecutionOrder(-10)]
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2Reconstruction : MonoBehaviour
{
    private const int ChunkSize = 16;
    private const int VoxelsPerChunk = ChunkSize * ChunkSize * ChunkSize;
    private const int SampleSize = ChunkSize + 1;
    private const int SampleCount = SampleSize * SampleSize * SampleSize;
    private const float VoxelMeters = 0.05f;
    private const float DerivedVoxelMeters = 0.10f;
    private const float ResampleRatio = VoxelMeters / DerivedVoxelMeters;
    private const int MeshStride = 2;
    private const int MeshCells = ChunkSize / MeshStride;
    private const float MinimumDepthMeters = 0.30f;
    private const float MaximumDepthMeters = 6.00f;
    private const float PositiveTruncationScale = 0.20f;
    private const float NegativeTruncationScale = 0.20f / 3f;
    private const float GradientThresholdSquared = 0.01f;
    private const float MaximumWeight = 1000f;
    private const float PlanarMetricNormalDot = 0.94f;
    private const float SeamWeldUnitsPerMeter = 10000f;
    private const int RemoveSmallTriangleThreshold = 32;
    private const int RemoveSmallTriangleGate = 40000;
    private const int MinimumRecordedPairsToCommit = 16;

    private enum WorkStage : byte
    {
        None,
        UnprojectReplayLeft,
        DecodeLeft,
        GradientLeft,
        CandidateLeft,
        UnprojectReplayRight,
        DecodeRight,
        GradientRight,
        CandidateRight,
        IntegrateLeft,
        IntegrateRight,
        GarbageCollect,
        Finalize
    }

    private struct Voxel
    {
        public float distance;
        public float weight;
    }

    private sealed class Chunk
    {
        public readonly Voxel[] voxels = new Voxel[VoxelsPerChunk];
        public GameObject previewObject;
        public MeshFilter previewFilter;
        public MeshRenderer previewRenderer;
        public Mesh frontMesh;
        public Mesh backMesh;
        public Vector3[] candidateVertices = Array.Empty<Vector3>();
        public int[] candidateTriangles = Array.Empty<int>();
        public int candidateTriangleCount;
        public int triangleCount;
    }

    private struct DirtyStamp
    {
        public double realtime;
        public int pair;
    }

    private sealed class FinalMeshMetrics
    {
        public string path = string.Empty;
        public string sha256 = string.Empty;
        public int vertices;
        public int triangles;
        public int connectedComponents;
        public int isolatedComponents;
        public int largestComponentTriangles;
        public int planarVertices;
        public float planeResidualRmsMeters;
        public float planeResidualP95Meters;
        public bool directDepthSupportMeasured;
        public int trianglesWithoutDirectDepthSupport;
        public float directDepthSupportRatio;
        public float directDepthAnalysisMilliseconds;
    }

    private readonly struct QuantizedPosition : IEquatable<QuantizedPosition>
    {
        private readonly int _x;
        private readonly int _y;
        private readonly int _z;

        public QuantizedPosition(Vector3 value)
        {
            _x = Mathf.RoundToInt(value.x * SeamWeldUnitsPerMeter);
            _y = Mathf.RoundToInt(value.y * SeamWeldUnitsPerMeter);
            _z = Mathf.RoundToInt(value.z * SeamWeldUnitsPerMeter);
        }

        public bool Equals(QuantizedPosition other)
        {
            return _x == other._x && _y == other._y && _z == other._z;
        }

        public override bool Equals(object value)
        {
            return value is QuantizedPosition other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = _x;
                hash = hash * 397 ^ _y;
                return hash * 397 ^ _z;
            }
        }
    }

    private sealed class UnionFind
    {
        private readonly int[] _parent;
        private readonly byte[] _rank;

        public UnionFind(int count)
        {
            _parent = new int[count];
            _rank = new byte[count];
            for (int i = 0; i < count; i++)
                _parent[i] = i;
        }

        public int Find(int value)
        {
            int root = value;
            while (_parent[root] != root)
                root = _parent[root];
            while (_parent[value] != value)
            {
                int next = _parent[value];
                _parent[value] = root;
                value = next;
            }
            return root;
        }

        public void Union(int first, int second)
        {
            int a = Find(first);
            int b = Find(second);
            if (a == b)
                return;
            if (_rank[a] < _rank[b])
                _parent[a] = b;
            else if (_rank[a] > _rank[b])
                _parent[b] = a;
            else
            {
                _parent[b] = a;
                _rank[a]++;
            }
        }
    }

    [Header("Execution budgets")]
    [SerializeField, Range(0.5f, 12f)]
    private float reconstructionBudgetMilliseconds = 6f;
    // This is only a runaway-work guard. The millisecond deadline below is
    // the real frame-time limiter. A value of 32 forced a 329-chunk stereo
    // pair to span at least ~22 display frames even when CPU budget remained.
    [SerializeField, Range(1, 512)]
    private int integrationChunksPerFrame = 256;
    [SerializeField, Range(0.5f, 12f)]
    private float meshingBudgetMilliseconds = 4f;
    [SerializeField, Range(1, 16)]
    private int meshChunksPerFrame = 8;

    [Header("Deterministic input tape")]
    [SerializeField, Range(10, 600)]
    private int maximumRecordedPairs = 180;
    [SerializeField, Range(1f, 30f)]
    private float replayHz = 10f;

    [Header("Replica display")]
    [SerializeField] private bool showReplica = true;
    [SerializeField] private Color lineColor =
        new Color(0.02f, 0.86f, 1f, 0.88f);

    public int AcceptedPairCount { get; private set; }
    public int DroppedBusyPairCount { get; private set; }
    public int ReadbackFailureCount { get; private set; }
    public int ActiveChunkCount => _chunks.Count;
    public int WeightedVoxelCount { get; private set; }
    public int PreviewTriangleCount { get; private set; }
    public int PendingMeshChunkCount => _queuedMeshChunks.Count;
    public float LastReconstructionMilliseconds { get; private set; }
    public float LastMeshingMilliseconds { get; private set; }
    public int LastDerivedValidSampleCount { get; private set; }
    public int LastDerivedMissingSampleCount { get; private set; }
    public int LastPreCleanupTriangleCount { get; private set; }
    public int LastPreCleanupComponentCount { get; private set; }
    public int LastRemovedSmallComponentCount { get; private set; }
    public int LastRemovedSmallTriangleCount { get; private set; }
    public int LastPostCleanupComponentCount { get; private set; }
    public float LastComponentCleanupMilliseconds { get; private set; }
    public bool LastSimplificationApplied { get; private set; }
    public int LastPreSimplificationVertexCount { get; private set; }
    public int LastPreSimplificationTriangleCount { get; private set; }
    public int LastSimplificationTargetVertexCount { get; private set; }
    public int LastPostSimplificationVertexCount { get; private set; }
    public int LastPostSimplificationTriangleCount { get; private set; }
    public int LastAcceptedCollapseCount { get; private set; }
    public int LastSimplificationFailedBatchCount { get; private set; }
    public int LastSimplificationTopologyRejectCount { get; private set; }
    public int LastSimplificationNormalRejectCount { get; private set; }
    public int LastSimplificationErrorRejectCount { get; private set; }
    public string LastSimplificationStopReason { get; private set; } =
        string.Empty;
    public float LastSimplificationMilliseconds { get; private set; }
    public bool HasPendingWork => _readbackPending || _workStage != WorkStage.None;
    public bool PreviewVisible => showReplica;
    public bool IsRecording => _tapeWriter != null;
    public bool IsReplaying => _tapeReader != null;
    public string TapePath => Path.Combine(
        Application.persistentDataPath,
        "RoomSetupReplicaV2", "last-session.rsv2");
    private string PendingTapePath => TapePath + ".recording";
    private string BackupTapePath => TapePath + ".previous";
    public string CompactStatus =>
        $"RS-V2 p={AcceptedPairCount:D6} chunk={ActiveChunkCount:D4} " +
        $"vox={WeightedVoxelCount:D7} tri={PreviewTriangleCount:D6} " +
        $"q={PendingMeshChunkCount:D4} " +
        $"rec={(IsRecording ? 1 : 0)} rep={(IsReplaying ? 1 : 0)}";

    private readonly Dictionary<Vector3Int, Chunk> _chunks =
        new Dictionary<Vector3Int, Chunk>(256);
    private readonly Stack<Chunk> _chunkPool = new Stack<Chunk>(64);
    private readonly HashSet<Vector3Int> _candidateSet =
        new HashSet<Vector3Int>();
    private readonly List<Vector3Int> _leftCandidates =
        new List<Vector3Int>(256);
    private readonly List<Vector3Int> _rightCandidates =
        new List<Vector3Int>(256);
    private readonly HashSet<Vector3Int> _garbageSet =
        new HashSet<Vector3Int>();
    private readonly List<Vector3Int> _garbageCandidates =
        new List<Vector3Int>(256);
    private readonly HashSet<Vector3Int> _touchedChunks =
        new HashSet<Vector3Int>();
    private readonly Dictionary<Vector3Int, byte> _touchedMasks =
        new Dictionary<Vector3Int, byte>();
    private readonly HashSet<Vector3Int> _queuedMeshChunks =
        new HashSet<Vector3Int>();
    private readonly Queue<Vector3Int> _meshQueue =
        new Queue<Vector3Int>();
    private readonly Dictionary<Vector3Int, DirtyStamp> _meshDirtyStamps =
        new Dictionary<Vector3Int, DirtyStamp>();
    private readonly HashSet<Vector3Int> _meshPassPublicationChunks =
        new HashSet<Vector3Int>();

    private ComputeShader _unproject;
    private int _unprojectKernel = -1;
    private RenderTexture _leftWorld;
    private RenderTexture _rightWorld;
    private AsyncGPUReadbackRequest _leftReadback;
    private AsyncGPUReadbackRequest _rightReadback;
    private NativeArray<Color> _leftPixels;
    private NativeArray<Color> _rightPixels;
    private int _width;
    private int _height;
    private int _expected;
    private int _resetGeneration;
    private int _pendingGeneration;
    private bool _readbackPending;
    private WorkStage _workStage;
    private int _cursor;
    private int _leftChunkCursor;
    private int _rightChunkCursor;
    private int _garbageCursor;
    private int _pendingFrame;
    private double _pendingCaptureTime;
    private Matrix4x4 _leftWorldToClip;
    private Matrix4x4 _rightWorldToClip;
    private Matrix4x4 _leftProjectionInverse;
    private Matrix4x4 _rightProjectionInverse;
    private Matrix4x4 _leftViewInverse;
    private Matrix4x4 _rightViewInverse;
    private Matrix4x4 _pendingStageToWorld;
    private Matrix4x4 _pendingWorldToStage;
    private bool _pendingStageValid;
    private Vector3 _leftEye;
    private Vector3 _rightEye;
    private float[] _leftDepth;
    private float[] _rightDepth;
    private float[] _leftNdc;
    private float[] _rightNdc;
    private byte[] _leftContinuous;
    private byte[] _rightContinuous;

    private RoomSetupReplicaV2Tape.Writer _tapeWriter;
    private RoomSetupReplicaV2Tape.Reader _tapeReader;
    private bool _processingReplayFrame;
    private float _nextReplayTime;
    private int _replayedFrames;
    private bool _replayUsesStageSpace;
    private bool _stageUnavailableLogged;
    private OVRCameraRig _ovrRig;
    private double _replayStartRealtime;
    private float _firstMeshLatencySeconds = -1f;
    private int _firstMeshPair = -1;
    private double _correctionDelaySecondsSum;
    private double _correctionDelaySecondsMax;
    private long _correctionPairLagSum;
    private int _correctionPairLagMax;
    private int _correctionPublicationCount;
    private Task<RoomSetupReplicaV2XrtSimplifier.Result>
        _simplificationTask;
    private RoomSetupReplicaV2XrtSimplifier.Input
        _pendingSimplificationInput;
    private int _simplificationSerial;

    private GameObject _previewRoot;
    private Material _lineMaterial;
    private Material _depthMaterial;
    private bool _meshPassActive;
    private float _nextMeshTime;
    private int _passDerivedValidSamples;
    private int _passDerivedMissingSamples;

    private readonly float[] _sampleDistances = new float[SampleCount];
    private readonly byte[] _sampleObserved = new byte[SampleCount];
    private readonly int[] _xEdgeVertices =
        new int[ChunkSize * SampleSize * SampleSize];
    private readonly int[] _yEdgeVertices =
        new int[SampleSize * ChunkSize * SampleSize];
    private readonly int[] _zEdgeVertices =
        new int[SampleSize * SampleSize * ChunkSize];
    private readonly int[] _cellEdgeVertices = new int[12];
    private readonly Vector3[] _cornerPositions = new Vector3[8];
    private readonly float[] _cornerDistances = new float[8];
    private readonly List<Vector3> _meshVertices = new List<Vector3>(4096);
    private readonly List<int> _meshTriangles = new List<int>(8192);
    private readonly List<int> _meshLines = new List<int>(8192);
    private readonly HashSet<ulong> _uniqueLines = new HashSet<ulong>();
    private readonly List<KeyValuePair<Vector3Int, Chunk>> _orderedChunks =
        new List<KeyValuePair<Vector3Int, Chunk>>(256);
    private readonly List<int> _orderedChunkVertexBases =
        new List<int>(256);
    private readonly Dictionary<int, int> _publishedVertexRemap =
        new Dictionary<int, int>(4096);
    private readonly List<Vector3> _finalVertices =
        new List<Vector3>(32768);
    private readonly List<int> _finalTriangles = new List<int>(65536);

    // A projective matrix is separable on the regular 16^3 voxel lattice.
    // Cache the three axis terms once per chunk so each voxel retains the
    // original left-to-right float additions without repeating nine scalar
    // multiplications. These arrays are single-flight scratch, like meshing.
    private readonly float[] _clipXX = new float[ChunkSize];
    private readonly float[] _clipXY = new float[ChunkSize];
    private readonly float[] _clipXZ = new float[ChunkSize];
    private readonly float[] _clipYX = new float[ChunkSize];
    private readonly float[] _clipYY = new float[ChunkSize];
    private readonly float[] _clipYZ = new float[ChunkSize];
    private readonly float[] _depthX = new float[ChunkSize];
    private readonly float[] _depthY = new float[ChunkSize];
    private readonly float[] _depthZ = new float[ChunkSize];

    private static readonly Vector3Int[] CubeOffsets =
    {
        new Vector3Int(0, 0, 0), new Vector3Int(1, 0, 0),
        new Vector3Int(1, 1, 0), new Vector3Int(0, 1, 0),
        new Vector3Int(0, 0, 1), new Vector3Int(1, 0, 1),
        new Vector3Int(1, 1, 1), new Vector3Int(0, 1, 1)
    };

    private static readonly int[,] EdgePairs =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };

    public void ObserveDepthFrame(DepthCapture depth)
    {
        if (IsReplaying || depth == null ||
            !DepthCapture.DepthAvailable || HasPendingWork)
        {
            if (HasPendingWork)
                DroppedBusyPairCount++;
            return;
        }

        Texture source = depth.ChiselSourceDepthTex;
        if (source == null || depth.ProjInv == null || depth.ProjInv.Length < 2 ||
            depth.ViewInv == null || depth.ViewInv.Length < 2 ||
            depth.Proj == null || depth.Proj.Length < 2 ||
            depth.View == null || depth.View.Length < 2 ||
            !EnsureUnproject() || !EnsureTargets(source.width, source.height))
            return;

        DispatchEye(source, depth, 0, _leftWorld);
        DispatchEye(source, depth, 1, _rightWorld);
        _leftWorldToClip = depth.Proj[0] * depth.View[0];
        _rightWorldToClip = depth.Proj[1] * depth.View[1];
        _leftProjectionInverse = depth.ProjInv[0];
        _rightProjectionInverse = depth.ProjInv[1];
        _leftViewInverse = depth.ViewInv[0];
        _rightViewInverse = depth.ViewInv[1];
        _pendingStageValid = TryGetStageTransforms(
            out _pendingStageToWorld, out _pendingWorldToStage);
        _leftEye = depth.ViewInv[0].MultiplyPoint3x4(Vector3.zero);
        _rightEye = depth.ViewInv[1].MultiplyPoint3x4(Vector3.zero);
        _pendingFrame = depth.CurrentPlatformFrame;
        _pendingCaptureTime = Time.realtimeSinceStartupAsDouble;
        _leftReadback = AsyncGPUReadback.Request(_leftWorld);
        _rightReadback = AsyncGPUReadback.Request(_rightWorld);
        _pendingGeneration = _resetGeneration;
        _readbackPending = true;
    }

    public bool ToggleRecording()
    {
        if (IsRecording)
        {
            StopRecording();
            return false;
        }
        return StartRecording();
    }

    public bool StartRecording()
    {
        if (IsRecording)
            return true;
        if (IsReplaying || _width <= 0 || _height <= 0)
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2] wait for the first live depth pair " +
                "before recording.");
            return false;
        }
        if (!TryGetStageTransforms(out _, out _))
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2] Quest Stage space is unavailable; " +
                "recording was not started.");
            return false;
        }
        try
        {
            RecoverInterruptedRecording();
            _tapeWriter = new RoomSetupReplicaV2Tape.Writer(
                PendingTapePath, _width, _height, maximumRecordedPairs);
            Debug.Log(
                "[RoomSetupReplicaV2] recording pending: " +
                PendingTapePath);
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] cannot start recording: " + exception);
            _tapeWriter = null;
            return false;
        }
    }

    public void StopRecording()
    {
        if (_tapeWriter == null)
            return;
        RoomSetupReplicaV2Tape.Writer writer = _tapeWriter;
        _tapeWriter = null;
        writer.Dispose();
        bool committed = writer.Failure == null &&
                         writer.WrittenFrames >=
                         MinimumRecordedPairsToCommit &&
                         TryCommitRecording(writer.Path);
        if (!committed)
            DeleteFileQuietly(writer.Path);
        Debug.Log(
            $"[RoomSetupReplicaV2] tape closed written={writer.WrittenFrames} " +
            $"dropped={writer.DroppedFrames} committed={committed} " +
            $"path={(committed ? TapePath : writer.Path)}");
        if (writer.Failure != null)
            Debug.LogError(
                "[RoomSetupReplicaV2] tape writer failed: " + writer.Failure);
        else if (writer.WrittenFrames < MinimumRecordedPairsToCommit)
            Debug.LogWarning(
                "[RoomSetupReplicaV2] short recording rejected; existing " +
                $"tape preserved ({writer.WrittenFrames} < " +
                $"{MinimumRecordedPairsToCommit} pairs)");
    }

    private void RecoverInterruptedRecording()
    {
        string directory = Path.GetDirectoryName(TapePath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        if (!File.Exists(TapePath) && File.Exists(BackupTapePath))
            File.Move(BackupTapePath, TapePath);
        else if (File.Exists(BackupTapePath))
            File.Delete(BackupTapePath);
        DeleteFileQuietly(PendingTapePath);
    }

    private bool TryCommitRecording(string pendingPath)
    {
        try
        {
            DeleteFileQuietly(BackupTapePath);
            bool hadPrevious = File.Exists(TapePath);
            if (hadPrevious)
                File.Move(TapePath, BackupTapePath);
            try
            {
                File.Move(pendingPath, TapePath);
            }
            catch
            {
                if (!File.Exists(TapePath) && File.Exists(BackupTapePath))
                    File.Move(BackupTapePath, TapePath);
                throw;
            }
            DeleteFileQuietly(BackupTapePath);
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] tape commit failed; previous tape " +
                "was preserved when recoverable: " + exception);
            return false;
        }
    }

    private static void DeleteFileQuietly(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch (Exception exception)
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2] could not delete temporary tape " +
                path + ": " + exception.Message);
        }
    }

    public bool BeginReplayLast()
    {
        StopRecording();
        StopReplay();
        if (!File.Exists(TapePath))
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2] no tape exists: " + TapePath);
            return false;
        }
        try
        {
            var reader = new RoomSetupReplicaV2Tape.Reader(TapePath);
            ClearReplica();
            if (!EnsureTargets(reader.Width, reader.Height))
            {
                reader.Dispose();
                return false;
            }
            if (!TryGetStageTransforms(out _, out _))
            {
                reader.Dispose();
                Debug.LogWarning(
                    "[RoomSetupReplicaV2] Quest Stage space is unavailable; " +
                    "replay was not started.");
                return false;
            }
            _replayUsesStageSpace = true;
            UpdatePreviewTransform();
            _tapeReader = reader;
            _replayedFrames = 0;
            _nextReplayTime = 0f;
            _replayStartRealtime = Time.realtimeSinceStartupAsDouble;
            _firstMeshLatencySeconds = -1f;
            _firstMeshPair = -1;
            _correctionDelaySecondsSum = 0d;
            _correctionDelaySecondsMax = 0d;
            _correctionPairLagSum = 0;
            _correctionPairLagMax = 0;
            _correctionPublicationCount = 0;
            _meshDirtyStamps.Clear();
            Debug.Log("[RoomSetupReplicaV2] replay started: " + TapePath);
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] cannot open replay: " + exception);
            StopReplay();
            return false;
        }
    }

    public void StopReplay()
    {
        if (_tapeReader == null)
            return;
        _tapeReader.Dispose();
        _tapeReader = null;
        _processingReplayFrame = false;
        Debug.Log("[RoomSetupReplicaV2] replay stopped.");
    }

    public void PrepareForLiveCapture()
    {
        if (!_replayUsesStageSpace)
            return;
        ClearReplica();
        _replayUsesStageSpace = false;
        UpdatePreviewTransform();
    }

    public void SetPreviewVisible(bool visible)
    {
        showReplica = visible;
        if (_previewRoot != null)
            _previewRoot.SetActive(visible);
    }

    [ContextMenu("Clear Room Setup Replica V2")]
    public void ClearReplica()
    {
        _resetGeneration++;
        _readbackPending = false;
        _workStage = WorkStage.None;
        foreach (Chunk chunk in _chunks.Values)
            DisposeChunk(chunk);
        _chunks.Clear();
        _chunkPool.Clear();
        _candidateSet.Clear();
        _leftCandidates.Clear();
        _rightCandidates.Clear();
        _garbageSet.Clear();
        _garbageCandidates.Clear();
        _touchedChunks.Clear();
        _touchedMasks.Clear();
        _queuedMeshChunks.Clear();
        _meshQueue.Clear();
        _meshDirtyStamps.Clear();
        _meshPassPublicationChunks.Clear();
        _meshPassActive = false;
        _pendingSimplificationInput = null;
        _nextMeshTime = 0f;
        AcceptedPairCount = 0;
        DroppedBusyPairCount = 0;
        ReadbackFailureCount = 0;
        WeightedVoxelCount = 0;
        PreviewTriangleCount = 0;
        LastDerivedValidSampleCount = 0;
        LastDerivedMissingSampleCount = 0;
        LastPreCleanupTriangleCount = 0;
        LastPreCleanupComponentCount = 0;
        LastRemovedSmallComponentCount = 0;
        LastRemovedSmallTriangleCount = 0;
        LastPostCleanupComponentCount = 0;
        LastComponentCleanupMilliseconds = 0f;
        LastSimplificationApplied = false;
        LastPreSimplificationVertexCount = 0;
        LastPreSimplificationTriangleCount = 0;
        LastSimplificationTargetVertexCount = 0;
        LastPostSimplificationVertexCount = 0;
        LastPostSimplificationTriangleCount = 0;
        LastAcceptedCollapseCount = 0;
        LastSimplificationFailedBatchCount = 0;
        LastSimplificationTopologyRejectCount = 0;
        LastSimplificationNormalRejectCount = 0;
        LastSimplificationErrorRejectCount = 0;
        LastSimplificationStopReason = string.Empty;
        LastSimplificationMilliseconds = 0f;
        LastReconstructionMilliseconds = 0f;
        LastMeshingMilliseconds = 0f;
    }

    private void Update()
    {
        UpdatePreviewTransform();
        PollSimplification();
        if (_readbackPending)
            CompleteReadbackWhenReady();
        if (_workStage != WorkStage.None)
            ProcessWorkSlice();
        // Visibility is presentation state, not a publication gate. Replay
        // must finish the same mesh work and receipt even while hidden.
        if (_meshQueue.Count > 0 && BeginMeshPass())
        {
            ProcessMeshSlice();
            if (_meshQueue.Count == 0)
                CompleteMeshPass();
        }
        if (IsReplaying && !HasPendingWork &&
            Time.unscaledTime >= _nextReplayTime)
            StartNextReplayFrame();
    }

    private bool EnsureUnproject()
    {
        if (_unproject != null && _unprojectKernel >= 0)
            return true;
        _unproject = Resources.Load<ComputeShader>("RoomSetupReplicaV2Unproject");
        if (_unproject == null)
            return false;
        try
        {
            _unprojectKernel = _unproject.FindKernel("UnprojectDepth");
            return _unprojectKernel >= 0;
        }
        catch
        {
            _unprojectKernel = -1;
            return false;
        }
    }

    private bool EnsureTargets(int width, int height)
    {
        if (_leftWorld != null && _leftWorld.IsCreated() &&
            width == _width && height == _height)
            return true;
        ReleaseTarget(ref _leftWorld);
        ReleaseTarget(ref _rightWorld);
        _width = width;
        _height = height;
        _expected = Mathf.Max(0, width * height);
        _leftWorld = CreateTarget("RoomSetup Replica Left", width, height);
        _rightWorld = CreateTarget("RoomSetup Replica Right", width, height);
        EnsurePixelArray(ref _leftPixels, _expected);
        EnsurePixelArray(ref _rightPixels, _expected);
        EnsureArray(ref _leftDepth, _expected);
        EnsureArray(ref _rightDepth, _expected);
        EnsureArray(ref _leftNdc, _expected);
        EnsureArray(ref _rightNdc, _expected);
        EnsureArray(ref _leftContinuous, _expected);
        EnsureArray(ref _rightContinuous, _expected);
        return _leftWorld != null && _rightWorld != null;
    }

    private static RenderTexture CreateTarget(string name, int width, int height)
    {
        var target = new RenderTexture(
            width, height, 0, RenderTextureFormat.ARGBFloat,
            RenderTextureReadWrite.Linear)
        {
            name = name,
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        target.Create();
        return target.IsCreated() ? target : null;
    }

    private void DispatchEye(
        Texture source, DepthCapture depth, int eye, RenderTexture output)
    {
        _unproject.SetTexture(_unprojectKernel, "_DepthTex", source);
        _unproject.SetTexture(_unprojectKernel, "_WorldPosition", output);
        _unproject.SetInts("_ImageSize", _width, _height);
        _unproject.SetInt("_Eye", eye);
        _unproject.SetFloat("_MinimumDepthMeters", MinimumDepthMeters);
        _unproject.SetFloat("_MaximumDepthMeters", MaximumDepthMeters);
        _unproject.SetMatrix("_ProjectionInverse", depth.ProjInv[eye]);
        _unproject.SetMatrix("_ViewInverse", depth.ViewInv[eye]);
        _unproject.Dispatch(
            _unprojectKernel,
            Mathf.CeilToInt(_width / 8f),
            Mathf.CeilToInt(_height / 8f), 1);
    }

    private void CompleteReadbackWhenReady()
    {
        if (!_leftReadback.done || !_rightReadback.done)
            return;
        _readbackPending = false;
        if (_pendingGeneration != _resetGeneration)
            return;
        if (_leftReadback.hasError || _rightReadback.hasError)
        {
            ReadbackFailureCount++;
            return;
        }
        NativeArray<Color> left = _leftReadback.GetData<Color>();
        NativeArray<Color> right = _rightReadback.GetData<Color>();
        if (left.Length < _expected || right.Length < _expected)
        {
            ReadbackFailureCount++;
            return;
        }
        NativeArray<Color>.Copy(left, _leftPixels, _expected);
        NativeArray<Color>.Copy(right, _rightPixels, _expected);
        Array.Clear(_leftDepth, 0, _expected);
        Array.Clear(_rightDepth, 0, _expected);
        Array.Clear(_leftNdc, 0, _expected);
        Array.Clear(_rightNdc, 0, _expected);
        Array.Clear(_leftContinuous, 0, _expected);
        Array.Clear(_rightContinuous, 0, _expected);
        _candidateSet.Clear();
        _leftCandidates.Clear();
        _rightCandidates.Clear();
        _garbageSet.Clear();
        _garbageCandidates.Clear();
        _touchedChunks.Clear();
        _touchedMasks.Clear();
        _leftChunkCursor = 0;
        _rightChunkCursor = 0;
        _garbageCursor = 0;
        BeginStage(WorkStage.DecodeLeft);
    }

    private void ProcessWorkSlice()
    {
        double start = Time.realtimeSinceStartupAsDouble;
        double deadline = start + Mathf.Max(
            0.0005f, reconstructionBudgetMilliseconds * 0.001f);
        int chunkBudget = Mathf.Max(1, integrationChunksPerFrame);
        int chunkWork = 0;
        while (_workStage != WorkStage.None &&
               Time.realtimeSinceStartupAsDouble < deadline)
        {
            switch (_workStage)
            {
                case WorkStage.UnprojectReplayLeft:
                    if (UnprojectReplaySlice(
                            _leftPixels, _leftNdc,
                            _leftProjectionInverse, _leftViewInverse,
                            deadline))
                        BeginStage(WorkStage.DecodeLeft);
                    break;
                case WorkStage.DecodeLeft:
                    if (DecodeSlice(_leftPixels, _leftDepth, _leftNdc,
                            _leftWorldToClip, deadline))
                        BeginStage(WorkStage.GradientLeft);
                    break;
                case WorkStage.GradientLeft:
                    if (GradientSlice(
                            _leftDepth, _leftContinuous, deadline))
                        BeginStage(WorkStage.CandidateLeft);
                    break;
                case WorkStage.CandidateLeft:
                    if (CandidateSlice(
                            _leftPixels, _leftDepth, _leftContinuous,
                            _leftEye, deadline))
                    {
                        _leftCandidates.AddRange(_candidateSet);
                        _candidateSet.Clear();
                        BeginStage(_processingReplayFrame
                            ? WorkStage.UnprojectReplayRight
                            : WorkStage.DecodeRight);
                    }
                    break;
                case WorkStage.UnprojectReplayRight:
                    if (UnprojectReplaySlice(
                            _rightPixels, _rightNdc,
                            _rightProjectionInverse, _rightViewInverse,
                            deadline))
                        BeginStage(WorkStage.DecodeRight);
                    break;
                case WorkStage.DecodeRight:
                    if (DecodeSlice(_rightPixels, _rightDepth, _rightNdc,
                            _rightWorldToClip, deadline))
                    {
                        if (!_processingReplayFrame)
                            RecordCurrentPair();
                        BeginStage(WorkStage.GradientRight);
                    }
                    break;
                case WorkStage.GradientRight:
                    if (GradientSlice(
                            _rightDepth, _rightContinuous, deadline))
                        BeginStage(WorkStage.CandidateRight);
                    break;
                case WorkStage.CandidateRight:
                    if (CandidateSlice(
                            _rightPixels, _rightDepth, _rightContinuous,
                            _rightEye, deadline))
                    {
                        _rightCandidates.AddRange(_candidateSet);
                        _candidateSet.Clear();
                        BeginStage(WorkStage.IntegrateLeft);
                    }
                    break;
                case WorkStage.IntegrateLeft:
                    if (chunkWork >= chunkBudget)
                        goto Done;
                    if (_leftChunkCursor < _leftCandidates.Count)
                    {
                        IntegrateChunk(
                            _leftCandidates[_leftChunkCursor++],
                            _leftDepth, _leftContinuous,
                            _leftWorldToClip);
                        chunkWork++;
                    }
                    else
                    {
                        BeginStage(WorkStage.IntegrateRight);
                    }
                    break;
                case WorkStage.IntegrateRight:
                    if (chunkWork >= chunkBudget)
                        goto Done;
                    if (_rightChunkCursor < _rightCandidates.Count)
                    {
                        IntegrateChunk(
                            _rightCandidates[_rightChunkCursor++],
                            _rightDepth, _rightContinuous,
                            _rightWorldToClip);
                        chunkWork++;
                    }
                    else
                    {
                        BeginGarbagePass();
                        BeginStage(WorkStage.GarbageCollect);
                    }
                    break;
                case WorkStage.GarbageCollect:
                    if (chunkWork >= chunkBudget)
                        goto Done;
                    if (_garbageCursor < _garbageCandidates.Count)
                    {
                        CollectGarbageChunk(
                            _garbageCandidates[_garbageCursor++]);
                        chunkWork++;
                    }
                    else
                    {
                        BeginStage(WorkStage.Finalize);
                    }
                    break;
                case WorkStage.Finalize:
                    FinalizePair();
                    break;
            }
        }

        Done:
        LastReconstructionMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
    }

    private void BeginStage(WorkStage stage)
    {
        _workStage = stage;
        _cursor = 0;
    }

    private void RecordCurrentPair()
    {
        RoomSetupReplicaV2Tape.Writer writer = _tapeWriter;
        if (writer == null)
            return;
        if (!_pendingStageValid)
        {
            if (!_stageUnavailableLogged)
            {
                _stageUnavailableLogged = true;
                Debug.LogWarning(
                    "[RoomSetupReplicaV2] skipped tape pair because Quest " +
                    "Stage space was unavailable at capture time.");
            }
            return;
        }
        _stageUnavailableLogged = false;

        // Freeze the packet in the persistent room coordinate frame rather
        // than this application's session-local Unity world frame.
        Matrix4x4 leftStageToClip =
            _leftWorldToClip * _pendingStageToWorld;
        Matrix4x4 rightStageToClip =
            _rightWorldToClip * _pendingStageToWorld;
        Matrix4x4 leftViewInverseStage =
            _pendingWorldToStage * _leftViewInverse;
        Matrix4x4 rightViewInverseStage =
            _pendingWorldToStage * _rightViewInverse;
        bool accepted = writer.TryEnqueue(
            _pendingFrame,
            _pendingCaptureTime,
            leftStageToClip,
            rightStageToClip,
            _leftProjectionInverse,
            _rightProjectionInverse,
            leftViewInverseStage,
            rightViewInverseStage,
            _leftNdc,
            _rightNdc);
        if (writer.Failure != null ||
            (!accepted && writer.AcceptedFrames >= maximumRecordedPairs) ||
            writer.AcceptedFrames >= maximumRecordedPairs)
            StopRecording();
    }

    private void StartNextReplayFrame()
    {
        RoomSetupReplicaV2Tape.Reader reader = _tapeReader;
        if (reader == null)
            return;
        RoomSetupReplicaV2Tape.Frame frame;
        try
        {
            if (!reader.TryReadNext(out frame))
            {
                if (_meshQueue.Count > 0 || _meshPassActive ||
                    _simplificationTask != null ||
                    _pendingSimplificationInput != null)
                {
                    _nextReplayTime = Time.unscaledTime + 0.05f;
                    return;
                }
                CompleteReplay();
                return;
            }
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] replay read failed: " + exception);
            CompleteReplay();
            return;
        }

        Array.Copy(frame.leftNdc, _leftNdc, _expected);
        Array.Copy(frame.rightNdc, _rightNdc, _expected);
        _pendingFrame = frame.platformFrame;
        _pendingCaptureTime = frame.captureTime;
        _leftWorldToClip = frame.leftWorldToClip;
        _rightWorldToClip = frame.rightWorldToClip;
        _leftProjectionInverse = frame.leftProjectionInverse;
        _rightProjectionInverse = frame.rightProjectionInverse;
        _leftViewInverse = frame.leftViewInverse;
        _rightViewInverse = frame.rightViewInverse;
        _leftEye = _leftViewInverse.MultiplyPoint3x4(Vector3.zero);
        _rightEye = _rightViewInverse.MultiplyPoint3x4(Vector3.zero);
        Array.Clear(_leftDepth, 0, _expected);
        Array.Clear(_rightDepth, 0, _expected);
        Array.Clear(_leftContinuous, 0, _expected);
        Array.Clear(_rightContinuous, 0, _expected);
        _candidateSet.Clear();
        _leftCandidates.Clear();
        _rightCandidates.Clear();
        _garbageSet.Clear();
        _garbageCandidates.Clear();
        _touchedChunks.Clear();
        _touchedMasks.Clear();
        _leftChunkCursor = 0;
        _rightChunkCursor = 0;
        _garbageCursor = 0;
        _processingReplayFrame = true;
        _nextReplayTime = Time.unscaledTime +
                          1f / Mathf.Max(1f, replayHz);
        BeginStage(WorkStage.UnprojectReplayLeft);
    }

    private bool UnprojectReplaySlice(
        NativeArray<Color> pixels,
        float[] ndcDepths,
        Matrix4x4 projectionInverse,
        Matrix4x4 viewInverse,
        double deadline)
    {
        Vector3 eye = viewInverse.MultiplyPoint3x4(Vector3.zero);
        while (_cursor < _expected)
        {
            int index = _cursor++;
            float ndc = ndcDepths[index];
            pixels[index] = default;
            if (ndc > 0f && ndc < 1f)
            {
                int x = index % _width;
                int y = index / _width;
                float u = (x + 0.5f) / _width;
                float v = (y + 0.5f) / _height;
                Vector4 clip = new Vector4(
                    u * 2f - 1f, v * 2f - 1f,
                    ndc * 2f - 1f, 1f);
                Vector4 viewH = projectionInverse * clip;
                Vector4 worldH = viewInverse * viewH;
                if (Mathf.Abs(viewH.w) > 0.00001f &&
                    Mathf.Abs(worldH.w) > 0.00001f)
                {
                    Vector3 world = new Vector3(
                        worldH.x / worldH.w,
                        worldH.y / worldH.w,
                        worldH.z / worldH.w);
                    float range = Vector3.Distance(world, eye);
                    if (Finite(world) &&
                        range >= MinimumDepthMeters &&
                        range <= MaximumDepthMeters)
                        pixels[index] = new Color(
                            world.x, world.y, world.z, ndc);
                }
            }
            if ((_cursor & 255) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool DecodeSlice(
        NativeArray<Color> pixels,
        float[] depths,
        float[] ndcDepths,
        Matrix4x4 worldToClip,
        double deadline)
    {
        while (_cursor < _expected)
        {
            int index = _cursor++;
            depths[index] = 0f;
            Color encoded = pixels[index];
            float ndc = encoded.a;
            ndcDepths[index] = ndc > 0f && ndc < 1f ? ndc : 0f;
            if (ndcDepths[index] > 0f)
            {
                float depth = ProjectiveDepth(
                    worldToClip, encoded.r, encoded.g, encoded.b);
                if (float.IsFinite(depth) &&
                    depth >= MinimumDepthMeters &&
                    depth <= MaximumDepthMeters)
                    depths[index] = depth;
            }
            if ((_cursor & 255) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool GradientSlice(
        float[] depths, byte[] accepted, double deadline)
    {
        while (_cursor < _expected)
        {
            int index = _cursor++;
            accepted[index] = 0;
            int x = index % _width;
            int y = index / _width;
            if (depths[index] > 0f && x > 0 && x < _width - 1 &&
                y > 0 && y < _height - 1)
            {
                float left = depths[index - 1];
                float right = depths[index + 1];
                float up = depths[index - _width];
                float down = depths[index + _width];
                if (left > 0f && right > 0f && up > 0f && down > 0f)
                {
                    float dx = right - left;
                    float dy = down - up;
                    if (dx * dx + dy * dy <= GradientThresholdSquared)
                        accepted[index] = 1;
                }
            }
            if ((_cursor & 255) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private bool CandidateSlice(
        NativeArray<Color> pixels,
        float[] depths,
        byte[] accepted,
        Vector3 eye,
        double deadline)
    {
        float chunkWorld = ChunkSize * VoxelMeters;
        while (_cursor < _expected)
        {
            int index = _cursor++;
            if (accepted[index] != 0)
            {
                Color encoded = pixels[index];
                AddRayCandidates(
                    eye, new Vector3(encoded.r, encoded.g, encoded.b),
                    depths[index], chunkWorld);
            }
            if ((_cursor & 63) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
        }
        return true;
    }

    private void AddRayCandidates(
        Vector3 eye,
        Vector3 surface,
        float observedDepth,
        float chunkWorld)
    {
        Vector3 ray = surface - eye;
        if (!Finite(surface) || !Finite(eye) || observedDepth <= 0f ||
            ray.sqrMagnitude <= 0.000001f)
            return;
        float positive = PositiveTruncationScale * (observedDepth + 1f);
        float negative = NegativeTruncationScale * (observedDepth + 1f);
        float startScale = Mathf.Max(
            0f, (observedDepth - positive) / observedDepth);
        float endScale = (observedDepth + negative) / observedDepth;
        Vector3 start = eye + ray * startScale;
        Vector3 end = eye + ray * endScale;
        float length = Vector3.Distance(start, end);
        int steps = Mathf.Max(1, Mathf.CeilToInt(
            length / Mathf.Max(VoxelMeters, chunkWorld * 0.65f)));
        float halo = Mathf.Sqrt(3f) * VoxelMeters;
        for (int step = 0; step <= steps; step++)
        {
            Vector3 point = Vector3.LerpUnclamped(
                start, end, step / (float)steps);
            AddChunkRange(point, halo, chunkWorld);
        }
    }

    private void AddChunkRange(Vector3 point, float radius, float chunkWorld)
    {
        Vector3Int minimum = WorldToChunk(
            point - Vector3.one * radius, chunkWorld);
        Vector3Int maximum = WorldToChunk(
            point + Vector3.one * radius, chunkWorld);
        for (int z = minimum.z; z <= maximum.z; z++)
        for (int y = minimum.y; y <= maximum.y; y++)
        for (int x = minimum.x; x <= maximum.x; x++)
            _candidateSet.Add(new Vector3Int(x, y, z));
    }

    private void IntegrateChunk(
        Vector3Int chunkId,
        float[] depths,
        byte[] accepted,
        Matrix4x4 worldToClip)
    {
        bool isNew = !_chunks.TryGetValue(chunkId, out Chunk chunk);
        if (isNew)
        {
            chunk = AcquireChunk();
            _chunks.Add(chunkId, chunk);
        }

        Vector3Int origin = chunkId * ChunkSize;
        PrepareProjectionTerms(origin, worldToClip);
        bool wrote = false;
        byte dirtyMask = 0;
        for (int z = 0; z < ChunkSize; z++)
        for (int y = 0; y < ChunkSize; y++)
        for (int x = 0; x < ChunkSize; x++)
        {
            int local = x + ChunkSize * (y + ChunkSize * z);
            float clipX = ((_clipXX[x] + _clipXY[y]) +
                           _clipXZ[z]) + worldToClip.m03;
            float clipY = ((_clipYX[x] + _clipYY[y]) +
                           _clipYZ[z]) + worldToClip.m13;
            float voxelDepth = ((_depthX[x] + _depthY[y]) +
                                _depthZ[z]) + worldToClip.m33;
            if (!float.IsFinite(voxelDepth) || voxelDepth <= 0.000001f)
                continue;
            float u = clipX / voxelDepth * 0.5f + 0.5f;
            float v = clipY / voxelDepth * 0.5f + 0.5f;
            if (u < 0f || u >= 1f || v < 0f || v >= 1f)
                continue;
            int px = Mathf.Clamp(Mathf.FloorToInt(u * _width), 0, _width - 1);
            int py = Mathf.Clamp(Mathf.FloorToInt(v * _height), 0, _height - 1);
            int index = px + py * _width;
            if (index < 0 || index >= _expected || accepted[index] == 0)
                continue;
            float observedDepth = depths[index];
            if (observedDepth <= 0f)
                continue;

            float factor = voxelDepth + 1f;
            float sampleDistance = Mathf.Min(
                observedDepth - voxelDepth,
                PositiveTruncationScale * factor);
            if (sampleDistance <= -NegativeTruncationScale * factor)
                continue;
            float sampleWeight = 1f /
                Mathf.Max(0.000001f, factor * factor);
            Voxel old = chunk.voxels[local];
            float retained = old.weight > 0f
                ? Mathf.Min(old.weight, MaximumWeight - sampleWeight)
                : 0f;
            retained = Mathf.Max(0f, retained);
            float newWeight = retained + sampleWeight;
            Voxel updated = new Voxel
            {
                distance = QuantizeHalf(retained > 0f
                    ? (old.distance * retained + sampleDistance * sampleWeight) /
                      newWeight
                    : sampleDistance),
                weight = QuantizeHalf(newWeight)
            };
            chunk.voxels[local] = updated;
            if (old.weight <= 0f)
                WeightedVoxelCount++;
            if (MeshRelevantChange(old, updated))
                AddDirtyVoxel(ref dirtyMask, local);
            wrote = true;
        }

        if (!wrote && isNew)
        {
            _chunks.Remove(chunkId);
            RecycleChunk(chunk);
            return;
        }
        if ((dirtyMask & 8) != 0)
            RecordTouched(chunkId, dirtyMask);
    }

    private static bool MeshRelevantChange(Voxel old, Voxel updated)
    {
        if (old.weight <= 0f)
            return true;
        // Room Setup rebuilds its derived mesh field from the current source
        // field. Do not add a ScanCover-specific geometry-change threshold.
        return old.distance != updated.distance;
    }

    private void BeginGarbagePass()
    {
        _garbageSet.Clear();
        _garbageCandidates.Clear();
        for (int i = 0; i < _leftCandidates.Count; i++)
            if (_garbageSet.Add(_leftCandidates[i]))
                _garbageCandidates.Add(_leftCandidates[i]);
        for (int i = 0; i < _rightCandidates.Count; i++)
            if (_garbageSet.Add(_rightCandidates[i]))
                _garbageCandidates.Add(_rightCandidates[i]);
        _garbageCursor = 0;
    }

    private void CollectGarbageChunk(Vector3Int chunkId)
    {
        if (!_chunks.TryGetValue(chunkId, out Chunk chunk) ||
            HasRetainedVoxel(chunkId, chunk, _leftWorldToClip))
            return;
        for (int i = 0; i < chunk.voxels.Length; i++)
            if (chunk.voxels[i].weight > 0f)
                WeightedVoxelCount--;
        WeightedVoxelCount = Mathf.Max(0, WeightedVoxelCount);
        RecordTouched(chunkId, 7);
        _chunks.Remove(chunkId);
        DisposeChunkPreview(chunk);
        RecycleChunk(chunk);
    }

    private bool HasRetainedVoxel(
        Vector3Int chunkId, Chunk chunk, Matrix4x4 worldToClip)
    {
        Vector3Int origin = chunkId * ChunkSize;
        PrepareProjectionTerms(origin, worldToClip);
        for (int z = 0; z < ChunkSize; z++)
        for (int y = 0; y < ChunkSize; y++)
        for (int x = 0; x < ChunkSize; x++)
        {
            Voxel voxel = chunk.voxels[
                x + ChunkSize * (y + ChunkSize * z)];
            if (voxel.weight <= 0f)
                continue;
            float viewDepth = ((_depthX[x] + _depthY[y]) +
                               _depthZ[z]) + worldToClip.m33;
            if (float.IsFinite(viewDepth) && viewDepth > 0f &&
                voxel.distance < 0.3f * Mathf.Clamp(viewDepth, 1f, 2f))
                return true;
        }
        return false;
    }

    private void PrepareProjectionTerms(
        Vector3Int origin, Matrix4x4 worldToClip)
    {
        // Native fuseDepthMap (0x7a5400) converts the sparse-tree integer
        // coordinates directly to float before applying its spatial scale.
        // Its +0.5 terms belong to image-pixel coordinates, not voxel-space
        // sample positions. Keep source fusion and the derived 10 cm lattice
        // on the same world-anchored integer grid.
        for (int x = 0; x < ChunkSize; x++)
        {
            float world = (origin.x + x) * VoxelMeters;
            _clipXX[x] = worldToClip.m00 * world;
            _clipYX[x] = worldToClip.m10 * world;
            _depthX[x] = worldToClip.m30 * world;
        }
        for (int y = 0; y < ChunkSize; y++)
        {
            float world = (origin.y + y) * VoxelMeters;
            _clipXY[y] = worldToClip.m01 * world;
            _clipYY[y] = worldToClip.m11 * world;
            _depthY[y] = worldToClip.m31 * world;
        }
        for (int z = 0; z < ChunkSize; z++)
        {
            float world = (origin.z + z) * VoxelMeters;
            _clipXZ[z] = worldToClip.m02 * world;
            _clipYZ[z] = worldToClip.m12 * world;
            _depthZ[z] = worldToClip.m32 * world;
        }
    }

    private void FinalizePair()
    {
        QueueTouchedMeshes();
        AcceptedPairCount++;
        if (_processingReplayFrame)
            _replayedFrames++;
        _processingReplayFrame = false;
        _leftCandidates.Clear();
        _rightCandidates.Clear();
        _garbageSet.Clear();
        _garbageCandidates.Clear();
        _workStage = WorkStage.None;
    }

    [Serializable]
    private sealed class ReplayReceipt
    {
        public string algorithm;
        public float sourceVoxelMeters;
        public float derivedVoxelMeters;
        public float resampleRatio;
        public string sourceSampleCoordinates;
        public string resampleKernel;
        public string missingSourceRule;
        public string tape;
        public string coordinateSpace;
        public int inputWidth;
        public int inputHeight;
        public int replayedPairs;
        public int activeChunks;
        public int weightedVoxels;
        public int previewTriangles;
        public bool removeSmallApplied;
        public int removeSmallTriangleThreshold;
        public int removeSmallTriangleGate;
        public int preCleanupTriangles;
        public int preCleanupComponents;
        public int removedSmallTriangles;
        public int removedSmallComponents;
        public int postCleanupComponents;
        public float componentCleanupMilliseconds;
        public bool simplificationApplied;
        public string simplificationMode;
        public int simplificationMinimumTriangles;
        public int simplificationTargetVertexDivisor;
        public int simplificationCandidateBatchSize;
        public int simplificationMaximumFailedBatches;
        public int simplificationRandomSeed;
        public float simplificationMaximumErrorMeters;
        public int simplificationAreaErrorFactorBits;
        public int simplificationShapeErrorFactorBits;
        public float simplificationAreaErrorMeters;
        public float simplificationShapeErrorMeters;
        public float simplificationMinimumNormalDot;
        public int preSimplificationVertices;
        public int preSimplificationTriangles;
        public int simplificationTargetVertices;
        public int postSimplificationVertices;
        public int postSimplificationTriangles;
        public int acceptedCollapses;
        public int simplificationFailedBatches;
        public int simplificationTopologyRejects;
        public int simplificationNormalRejects;
        public int simplificationErrorRejects;
        public string simplificationStopReason;
        public float simplificationMilliseconds;
        public int lastMeshPassDerivedValidSamples;
        public int lastMeshPassDerivedMissingSamples;
        public float lastReconstructionMilliseconds;
        public float lastMeshingMilliseconds;
        public float replayElapsedSeconds;
        public float replayPairsPerSecond;
        public float firstMeshLatencySeconds;
        public int firstMeshPair;
        public int correctionPublications;
        public float correctionDelayMeanSeconds;
        public float correctionDelayMaxSeconds;
        public float correctionPairLagMean;
        public int correctionPairLagMax;
        public string finalMeshPath;
        public string finalMeshSha256;
        public int finalMeshVertices;
        public int finalMeshTriangles;
        public int connectedComponents;
        public int isolatedComponents;
        public int largestComponentTriangles;
        public int planarVertices;
        public float planeResidualRmsMeters;
        public float planeResidualP95Meters;
        public bool directDepthSupportMeasured;
        public int trianglesWithoutDirectDepthSupport;
        public float directDepthSupportRatio;
        public float directDepthAnalysisMilliseconds;
    }

    private void CompleteReplay()
    {
        RoomSetupReplicaV2Tape.Reader reader = _tapeReader;
        _tapeReader = null;
        reader?.Dispose();
        _processingReplayFrame = false;
        float replayElapsedSeconds = (float)Math.Max(
            0d, Time.realtimeSinceStartupAsDouble - _replayStartRealtime);
        try
        {
            string directory = Path.GetDirectoryName(TapePath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            FinalMeshMetrics meshMetrics =
                ExportAndMeasureFinalMesh(directory ??
                                         Application.persistentDataPath);
            Debug.Log(
                "[RoomSetupReplicaV2] measuring final-triangle direct " +
                "depth support...");
            MeasureDirectDepthSupport(TapePath, meshMetrics);
            Debug.Log(
                $"[RoomSetupReplicaV2] direct support=" +
                $"{meshMetrics.directDepthSupportRatio:P2} " +
                $"unsupported=" +
                $"{meshMetrics.trianglesWithoutDirectDepthSupport} " +
                $"analysisMs=" +
                $"{meshMetrics.directDepthAnalysisMilliseconds:F1}");
            string receiptPath = Path.Combine(
                directory ?? Application.persistentDataPath,
                "last-replay-receipt.json");
            var receipt = new ReplayReceipt
            {
                algorithm =
                    "V2BWhiteboxDirectSampleRemoveSmall32XrtPointPlaneEmptyFallback",
                sourceVoxelMeters = VoxelMeters,
                derivedVoxelMeters = DerivedVoxelMeters,
                resampleRatio = ResampleRatio,
                sourceSampleCoordinates = "IntegerGridPoints",
                resampleKernel = "AlignedExactCopy",
                missingSourceRule = "KeepDerivedSampleUnobserved",
                tape = TapePath,
                coordinateSpace = reader != null
                    ? reader.CoordinateSpace
                    : "QuestStage",
                inputWidth = reader != null ? reader.Width : _width,
                inputHeight = reader != null ? reader.Height : _height,
                replayedPairs = _replayedFrames,
                activeChunks = ActiveChunkCount,
                weightedVoxels = WeightedVoxelCount,
                previewTriangles = PreviewTriangleCount,
                removeSmallApplied =
                    LastPreCleanupTriangleCount < RemoveSmallTriangleGate,
                removeSmallTriangleThreshold = RemoveSmallTriangleThreshold,
                removeSmallTriangleGate = RemoveSmallTriangleGate,
                preCleanupTriangles = LastPreCleanupTriangleCount,
                preCleanupComponents = LastPreCleanupComponentCount,
                removedSmallTriangles = LastRemovedSmallTriangleCount,
                removedSmallComponents = LastRemovedSmallComponentCount,
                postCleanupComponents = LastPostCleanupComponentCount,
                componentCleanupMilliseconds =
                    LastComponentCleanupMilliseconds,
                simplificationApplied = LastSimplificationApplied,
                simplificationMode = "XRTDirectedEdgeCollapse",
                simplificationMinimumTriangles =
                    RoomSetupReplicaV2XrtSimplifier.MinimumTriangles,
                simplificationTargetVertexDivisor =
                    RoomSetupReplicaV2XrtSimplifier.TargetVertexDivisor,
                simplificationCandidateBatchSize =
                    RoomSetupReplicaV2XrtSimplifier.CandidateBatchSize,
                simplificationMaximumFailedBatches =
                    RoomSetupReplicaV2XrtSimplifier.MaximumFailedBatches,
                simplificationRandomSeed =
                    (int)RoomSetupReplicaV2XrtSimplifier.RandomSeed,
                simplificationMaximumErrorMeters =
                    RoomSetupReplicaV2XrtSimplifier.MaximumErrorMeters,
                simplificationAreaErrorFactorBits =
                    RoomSetupReplicaV2XrtSimplifier.AreaErrorFactorBits,
                simplificationShapeErrorFactorBits =
                    RoomSetupReplicaV2XrtSimplifier.ShapeErrorFactorBits,
                simplificationAreaErrorMeters =
                    RoomSetupReplicaV2XrtSimplifier.AreaErrorMeters,
                simplificationShapeErrorMeters =
                    RoomSetupReplicaV2XrtSimplifier.ShapeErrorMeters,
                simplificationMinimumNormalDot =
                    RoomSetupReplicaV2XrtSimplifier.MinimumNormalDot,
                preSimplificationVertices =
                    LastPreSimplificationVertexCount,
                preSimplificationTriangles =
                    LastPreSimplificationTriangleCount,
                simplificationTargetVertices =
                    LastSimplificationTargetVertexCount,
                postSimplificationVertices =
                    LastPostSimplificationVertexCount,
                postSimplificationTriangles =
                    LastPostSimplificationTriangleCount,
                acceptedCollapses = LastAcceptedCollapseCount,
                simplificationFailedBatches =
                    LastSimplificationFailedBatchCount,
                simplificationTopologyRejects =
                    LastSimplificationTopologyRejectCount,
                simplificationNormalRejects =
                    LastSimplificationNormalRejectCount,
                simplificationErrorRejects =
                    LastSimplificationErrorRejectCount,
                simplificationStopReason =
                    LastSimplificationStopReason,
                simplificationMilliseconds =
                    LastSimplificationMilliseconds,
                lastMeshPassDerivedValidSamples =
                    LastDerivedValidSampleCount,
                lastMeshPassDerivedMissingSamples =
                    LastDerivedMissingSampleCount,
                lastReconstructionMilliseconds =
                    LastReconstructionMilliseconds,
                lastMeshingMilliseconds = LastMeshingMilliseconds,
                replayElapsedSeconds = replayElapsedSeconds,
                replayPairsPerSecond = replayElapsedSeconds > 0f
                    ? _replayedFrames / replayElapsedSeconds
                    : 0f,
                firstMeshLatencySeconds = _firstMeshLatencySeconds,
                firstMeshPair = _firstMeshPair,
                correctionPublications = _correctionPublicationCount,
                correctionDelayMeanSeconds = _correctionPublicationCount > 0
                    ? (float)(_correctionDelaySecondsSum /
                              _correctionPublicationCount)
                    : 0f,
                correctionDelayMaxSeconds =
                    (float)_correctionDelaySecondsMax,
                correctionPairLagMean = _correctionPublicationCount > 0
                    ? (float)_correctionPairLagSum /
                      _correctionPublicationCount
                    : 0f,
                correctionPairLagMax = _correctionPairLagMax,
                finalMeshPath = meshMetrics.path,
                finalMeshSha256 = meshMetrics.sha256,
                finalMeshVertices = meshMetrics.vertices,
                finalMeshTriangles = meshMetrics.triangles,
                connectedComponents = meshMetrics.connectedComponents,
                isolatedComponents = meshMetrics.isolatedComponents,
                largestComponentTriangles =
                    meshMetrics.largestComponentTriangles,
                planarVertices = meshMetrics.planarVertices,
                planeResidualRmsMeters =
                    meshMetrics.planeResidualRmsMeters,
                planeResidualP95Meters =
                    meshMetrics.planeResidualP95Meters,
                directDepthSupportMeasured =
                    meshMetrics.directDepthSupportMeasured,
                trianglesWithoutDirectDepthSupport =
                    meshMetrics.trianglesWithoutDirectDepthSupport,
                directDepthSupportRatio =
                    meshMetrics.directDepthSupportRatio,
                directDepthAnalysisMilliseconds =
                    meshMetrics.directDepthAnalysisMilliseconds
            };
            File.WriteAllText(
                receiptPath, JsonUtility.ToJson(receipt, true));
            Debug.Log(
                $"[RoomSetupReplicaV2] replay complete pairs={_replayedFrames} " +
                $"receipt={receiptPath}");
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] cannot write replay receipt: " +
                exception);
        }
    }

    private FinalMeshMetrics ExportAndMeasureFinalMesh(string directory)
    {
        _orderedChunks.Clear();
        foreach (KeyValuePair<Vector3Int, Chunk> item in _chunks)
            if (item.Value.frontMesh != null && item.Value.triangleCount > 0)
                _orderedChunks.Add(item);
        _orderedChunks.Sort(CompareChunks);

        _finalVertices.Clear();
        _finalTriangles.Clear();
        for (int chunkIndex = 0;
             chunkIndex < _orderedChunks.Count;
             chunkIndex++)
        {
            Mesh mesh = _orderedChunks[chunkIndex].Value.frontMesh;
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.GetTriangles(0);
            int vertexBase = _finalVertices.Count;
            _finalVertices.AddRange(vertices);
            for (int i = 0; i < triangles.Length; i++)
                _finalTriangles.Add(vertexBase + triangles[i]);
        }

        var metrics = new FinalMeshMetrics
        {
            vertices = _finalVertices.Count,
            triangles = _finalTriangles.Count / 3,
            path = Path.Combine(directory, "last-replay-mesh.ply")
        };
        WriteBinaryPly(metrics.path, _finalVertices, _finalTriangles);
        metrics.sha256 = ComputeFileSha256(metrics.path);
        MeasureConnectedComponents(metrics);
        MeasurePlanarResidual(metrics);
        return metrics;
    }

    private void MeasureDirectDepthSupport(
        string tapePath, FinalMeshMetrics metrics)
    {
        int triangleCount = _finalTriangles.Count / 3;
        double start = Time.realtimeSinceStartupAsDouble;
        if (triangleCount <= 0)
        {
            metrics.directDepthSupportMeasured = true;
            metrics.trianglesWithoutDirectDepthSupport = 0;
            metrics.directDepthSupportRatio = 1f;
            metrics.directDepthAnalysisMilliseconds = 0f;
            return;
        }

        var supported = new bool[triangleCount];
        int remaining = triangleCount;
        using (var reader = new RoomSetupReplicaV2Tape.Reader(tapePath))
        {
            while (remaining > 0 && reader.TryReadNext(out var frame))
            {
                for (int triangle = 0;
                     triangle < triangleCount;
                     triangle++)
                {
                    if (supported[triangle])
                        continue;
                    int index = triangle * 3;
                    Vector3 a = _finalVertices[
                        _finalTriangles[index]];
                    Vector3 b = _finalVertices[
                        _finalTriangles[index + 1]];
                    Vector3 c = _finalVertices[
                        _finalTriangles[index + 2]];
                    Vector3 centroid = (a + b + c) / 3f;
                    bool hasSupport = HasDirectDepthSupport(
                        centroid, frame, reader.Width, reader.Height);
                    if (!hasSupport)
                    {
                        int supportedVertices = 0;
                        if (HasDirectDepthSupport(
                                a, frame, reader.Width, reader.Height))
                            supportedVertices++;
                        if (HasDirectDepthSupport(
                                b, frame, reader.Width, reader.Height))
                            supportedVertices++;
                        if (supportedVertices < 2 &&
                            HasDirectDepthSupport(
                                c, frame, reader.Width, reader.Height))
                            supportedVertices++;
                        hasSupport = supportedVertices >= 2;
                    }
                    if (!hasSupport)
                        continue;
                    supported[triangle] = true;
                    remaining--;
                }
            }
        }

        metrics.directDepthSupportMeasured = true;
        metrics.trianglesWithoutDirectDepthSupport = remaining;
        metrics.directDepthSupportRatio =
            1f - remaining / (float)triangleCount;
        metrics.directDepthAnalysisMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
    }

    private static bool HasDirectDepthSupport(
        Vector3 stagePoint,
        RoomSetupReplicaV2Tape.Frame frame,
        int width,
        int height)
    {
        return HasDirectDepthSupport(
                   stagePoint, frame.leftWorldToClip,
                   frame.leftProjectionInverse, frame.leftViewInverse,
                   frame.leftNdc, width, height) ||
               HasDirectDepthSupport(
                   stagePoint, frame.rightWorldToClip,
                   frame.rightProjectionInverse, frame.rightViewInverse,
                   frame.rightNdc, width, height);
    }

    private static bool HasDirectDepthSupport(
        Vector3 stagePoint,
        Matrix4x4 stageToClip,
        Matrix4x4 projectionInverse,
        Matrix4x4 viewInverse,
        float[] ndcDepths,
        int width,
        int height)
    {
        Vector4 clip = stageToClip * new Vector4(
            stagePoint.x, stagePoint.y, stagePoint.z, 1f);
        if (!float.IsFinite(clip.w) || clip.w <= 0.00001f)
            return false;
        float inverseW = 1f / clip.w;
        float u = clip.x * inverseW * 0.5f + 0.5f;
        float v = clip.y * inverseW * 0.5f + 0.5f;
        if (!float.IsFinite(u) || !float.IsFinite(v) ||
            u < 0f || u >= 1f || v < 0f || v >= 1f)
            return false;
        int x = Mathf.Clamp((int)(u * width), 0, width - 1);
        int y = Mathf.Clamp((int)(v * height), 0, height - 1);
        float ndc = ndcDepths[x + y * width];
        if (!(ndc > 0f && ndc < 1f))
            return false;

        float sampleU = (x + 0.5f) / width;
        float sampleV = (y + 0.5f) / height;
        Vector4 viewH = projectionInverse * new Vector4(
            sampleU * 2f - 1f,
            sampleV * 2f - 1f,
            ndc * 2f - 1f,
            1f);
        Vector4 stageH = viewInverse * viewH;
        if (!float.IsFinite(stageH.w) ||
            Mathf.Abs(stageH.w) <= 0.00001f)
            return false;
        Vector3 observed = new Vector3(
            stageH.x / stageH.w,
            stageH.y / stageH.w,
            stageH.z / stageH.w);
        return Finite(observed) &&
               Vector3.Distance(stagePoint, observed) <= VoxelMeters;
    }

    private static int CompareChunks(
        KeyValuePair<Vector3Int, Chunk> first,
        KeyValuePair<Vector3Int, Chunk> second)
    {
        int compare = first.Key.x.CompareTo(second.Key.x);
        if (compare != 0) return compare;
        compare = first.Key.y.CompareTo(second.Key.y);
        return compare != 0
            ? compare
            : first.Key.z.CompareTo(second.Key.z);
    }

    private static void WriteBinaryPly(
        string path, List<Vector3> vertices, List<int> triangles)
    {
        var header = new StringBuilder(256);
        header.Append("ply\n")
              .Append("format binary_little_endian 1.0\n")
              .Append("comment coordinate_space QuestStage\n")
              .Append("element vertex ").Append(vertices.Count).Append('\n')
              .Append("property float x\n")
              .Append("property float y\n")
              .Append("property float z\n")
              .Append("element face ").Append(triangles.Count / 3)
              .Append('\n')
              .Append("property list uchar int vertex_indices\n")
              .Append("end_header\n");
        using (var writer = new BinaryWriter(new FileStream(
                   path, FileMode.Create, FileAccess.Write, FileShare.Read,
                   1 << 20, FileOptions.SequentialScan)))
        {
            writer.Write(Encoding.ASCII.GetBytes(header.ToString()));
            for (int i = 0; i < vertices.Count; i++)
            {
                Vector3 vertex = vertices[i];
                writer.Write(vertex.x);
                writer.Write(vertex.y);
                writer.Write(vertex.z);
            }
            for (int i = 0; i < triangles.Count; i += 3)
            {
                writer.Write((byte)3);
                writer.Write(triangles[i]);
                writer.Write(triangles[i + 1]);
                writer.Write(triangles[i + 2]);
            }
        }
    }

    private static string ComputeFileSha256(string path)
    {
        byte[] hash;
        using (SHA256 sha = SHA256.Create())
        using (FileStream stream = File.OpenRead(path))
            hash = sha.ComputeHash(stream);
        var text = new StringBuilder(hash.Length * 2);
        for (int i = 0; i < hash.Length; i++)
            text.Append(hash[i].ToString("X2"));
        return text.ToString();
    }

    private void MeasureConnectedComponents(FinalMeshMetrics metrics)
    {
        int vertexCount = _finalVertices.Count;
        if (vertexCount == 0 || _finalTriangles.Count == 0)
            return;
        var union = new UnionFind(vertexCount);
        for (int i = 0; i < _finalTriangles.Count; i += 3)
        {
            int a = _finalTriangles[i];
            int b = _finalTriangles[i + 1];
            int c = _finalTriangles[i + 2];
            union.Union(a, b);
            union.Union(b, c);
        }

        // Chunk meshes duplicate seam vertices. A 0.1 mm spatial weld makes
        // connectivity independent of chunk ownership without changing the
        // exported geometry or joining visibly separate surfaces.
        var seamOwners = new Dictionary<QuantizedPosition, int>(vertexCount);
        for (int i = 0; i < vertexCount; i++)
        {
            var key = new QuantizedPosition(_finalVertices[i]);
            if (seamOwners.TryGetValue(key, out int owner))
                union.Union(i, owner);
            else
                seamOwners.Add(key, i);
        }

        var trianglesByRoot = new Dictionary<int, int>();
        for (int i = 0; i < _finalTriangles.Count; i += 3)
        {
            int root = union.Find(_finalTriangles[i]);
            if (trianglesByRoot.TryGetValue(root, out int count))
                trianglesByRoot[root] = count + 1;
            else
                trianglesByRoot.Add(root, 1);
        }
        metrics.connectedComponents = trianglesByRoot.Count;
        metrics.isolatedComponents = Mathf.Max(
            0, metrics.connectedComponents - 1);
        foreach (int count in trianglesByRoot.Values)
            metrics.largestComponentTriangles = Mathf.Max(
                metrics.largestComponentTriangles, count);
    }

    private void MeasurePlanarResidual(FinalMeshMetrics metrics)
    {
        int vertexCount = _finalVertices.Count;
        if (vertexCount == 0 || _finalTriangles.Count == 0)
            return;
        var normalSums = new Vector3[vertexCount];
        var normals = new Vector3[vertexCount];
        var minimumDots = new float[vertexCount];
        var faceCounts = new int[vertexCount];
        var residualSquares = new double[vertexCount];
        var residualCounts = new int[vertexCount];
        for (int i = 0; i < vertexCount; i++)
            minimumDots[i] = 1f;

        for (int i = 0; i < _finalTriangles.Count; i += 3)
        {
            int a = _finalTriangles[i];
            int b = _finalTriangles[i + 1];
            int c = _finalTriangles[i + 2];
            Vector3 face = Vector3.Cross(
                _finalVertices[b] - _finalVertices[a],
                _finalVertices[c] - _finalVertices[a]);
            float length = face.magnitude;
            if (!float.IsFinite(length) || length <= 0.000001f)
                continue;
            face /= length;
            normalSums[a] += face;
            normalSums[b] += face;
            normalSums[c] += face;
            faceCounts[a]++;
            faceCounts[b]++;
            faceCounts[c]++;
        }
        for (int i = 0; i < vertexCount; i++)
        {
            float length = normalSums[i].magnitude;
            if (float.IsFinite(length) && length > 0.000001f)
                normals[i] = normalSums[i] / length;
        }

        for (int i = 0; i < _finalTriangles.Count; i += 3)
        {
            int a = _finalTriangles[i];
            int b = _finalTriangles[i + 1];
            int c = _finalTriangles[i + 2];
            Vector3 pa = _finalVertices[a];
            Vector3 pb = _finalVertices[b];
            Vector3 pc = _finalVertices[c];
            Vector3 face = Vector3.Cross(pb - pa, pc - pa);
            float length = face.magnitude;
            if (!float.IsFinite(length) || length <= 0.000001f)
                continue;
            face /= length;
            AccumulatePlanarMetric(
                a, pa, pb, pc, face, normals,
                minimumDots, residualSquares, residualCounts);
            AccumulatePlanarMetric(
                b, pb, pa, pc, face, normals,
                minimumDots, residualSquares, residualCounts);
            AccumulatePlanarMetric(
                c, pc, pa, pb, face, normals,
                minimumDots, residualSquares, residualCounts);
        }

        var vertexResiduals = new List<float>(vertexCount);
        double totalSquares = 0d;
        long totalSamples = 0;
        for (int i = 0; i < vertexCount; i++)
        {
            if (faceCounts[i] < 2 || residualCounts[i] <= 0 ||
                minimumDots[i] < PlanarMetricNormalDot)
                continue;
            float rms = Mathf.Sqrt((float)(
                residualSquares[i] / residualCounts[i]));
            if (!float.IsFinite(rms))
                continue;
            vertexResiduals.Add(rms);
            totalSquares += residualSquares[i];
            totalSamples += residualCounts[i];
        }
        metrics.planarVertices = vertexResiduals.Count;
        if (totalSamples <= 0 || vertexResiduals.Count == 0)
            return;
        metrics.planeResidualRmsMeters = Mathf.Sqrt(
            (float)(totalSquares / totalSamples));
        vertexResiduals.Sort();
        int p95 = Mathf.Clamp(
            Mathf.CeilToInt(vertexResiduals.Count * 0.95f) - 1,
            0, vertexResiduals.Count - 1);
        metrics.planeResidualP95Meters = vertexResiduals[p95];
    }

    private static void AccumulatePlanarMetric(
        int index,
        Vector3 center,
        Vector3 first,
        Vector3 second,
        Vector3 face,
        Vector3[] normals,
        float[] minimumDots,
        double[] residualSquares,
        int[] residualCounts)
    {
        Vector3 normal = normals[index];
        if (!Finite(normal) || normal.sqrMagnitude <= 0.000001f)
            return;
        minimumDots[index] = Mathf.Min(
            minimumDots[index],
            Mathf.Abs(Vector3.Dot(normal, face)));
        float firstDistance = Vector3.Dot(first - center, normal);
        float secondDistance = Vector3.Dot(second - center, normal);
        residualSquares[index] +=
            firstDistance * (double)firstDistance +
            secondDistance * (double)secondDistance;
        residualCounts[index] += 2;
    }

    private void RecordTouched(Vector3Int chunkId, byte mask)
    {
        _touchedChunks.Add(chunkId);
        if (_touchedMasks.TryGetValue(chunkId, out byte old))
            _touchedMasks[chunkId] = (byte)(old | mask);
        else
            _touchedMasks.Add(chunkId, mask);
    }

    private static void AddDirtyVoxel(ref byte mask, int local)
    {
        int x = local % ChunkSize;
        int yz = local / ChunkSize;
        int y = yz % ChunkSize;
        int z = yz / ChunkSize;
        mask |= 8;
        if (x == 0) mask |= 1;
        if (y == 0) mask |= 2;
        if (z == 0) mask |= 4;
    }

    private void QueueTouchedMeshes()
    {
        foreach (Vector3Int touched in _touchedChunks)
        {
            byte mask = _touchedMasks.TryGetValue(
                touched, out byte value) ? value : (byte)8;
            int minX = (mask & 1) != 0 ? -1 : 0;
            int minY = (mask & 2) != 0 ? -1 : 0;
            int minZ = (mask & 4) != 0 ? -1 : 0;
            for (int z = minZ; z <= 0; z++)
            for (int y = minY; y <= 0; y++)
            for (int x = minX; x <= 0; x++)
            {
                Vector3Int dirty = touched + new Vector3Int(x, y, z);
                if (_processingReplayFrame || IsReplaying)
                {
                    _meshDirtyStamps[dirty] = new DirtyStamp
                    {
                        realtime = Time.realtimeSinceStartupAsDouble,
                        pair = _replayedFrames + 1
                    };
                }
                if (_queuedMeshChunks.Add(dirty))
                    _meshQueue.Enqueue(dirty);
            }
        }
        _touchedChunks.Clear();
        _touchedMasks.Clear();
    }

    private bool BeginMeshPass()
    {
        if (_meshPassActive)
            return true;
        if (Time.unscaledTime < _nextMeshTime)
            return false;
        _meshPassActive = true;
        _passDerivedValidSamples = 0;
        _passDerivedMissingSamples = 0;
        _meshPassPublicationChunks.Clear();
        return true;
    }

    private void CompleteMeshPass()
    {
        PublishRemoveSmallCandidateMeshes();
        _meshPassPublicationChunks.Clear();
        _meshPassActive = false;
        LastDerivedValidSampleCount = _passDerivedValidSamples;
        LastDerivedMissingSampleCount = _passDerivedMissingSamples;
        if (IsReplaying && PreviewTriangleCount > 0 &&
            _firstMeshLatencySeconds < 0f)
        {
            _firstMeshLatencySeconds = (float)(
                Time.realtimeSinceStartupAsDouble - _replayStartRealtime);
            _firstMeshPair = _replayedFrames;
        }
        int band = Mathf.Max(0, PreviewTriangleCount) / 6000;
        _nextMeshTime = Time.unscaledTime +
                        Mathf.Min(0.9f, 0.15f + 0.25f * band);
    }

    private void ProcessMeshSlice()
    {
        EnsurePreviewResources();
        if (_previewRoot == null || _lineMaterial == null ||
            _depthMaterial == null)
            return;
        double start = Time.realtimeSinceStartupAsDouble;
        double budget = Mathf.Max(0.0005f, meshingBudgetMilliseconds * 0.001f);
        int processed = 0;
        int attempts = _meshQueue.Count;
        while (processed < Mathf.Max(1, meshChunksPerFrame) &&
               _meshQueue.Count > 0 && attempts-- > 0)
        {
            Vector3Int chunkId = _meshQueue.Dequeue();
            if (_workStage != WorkStage.None && _touchedChunks.Contains(chunkId))
            {
                _meshQueue.Enqueue(chunkId);
                continue;
            }
            _queuedMeshChunks.Remove(chunkId);
            if (_chunks.TryGetValue(chunkId, out Chunk chunk))
                RebuildChunkMesh(chunkId, chunk);
            // A removed chunk is also a published correction: its previous
            // Front has disappeared when the global candidate is committed.
            _meshPassPublicationChunks.Add(chunkId);
            processed++;
            if (Time.realtimeSinceStartupAsDouble - start >= budget)
                break;
        }
        LastMeshingMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
    }

    private void RecordCorrectionPublication(Vector3Int chunkId)
    {
        if (!_meshDirtyStamps.TryGetValue(
                chunkId, out DirtyStamp stamp))
            return;
        _meshDirtyStamps.Remove(chunkId);
        double seconds = Math.Max(
            0d, Time.realtimeSinceStartupAsDouble - stamp.realtime);
        int pairLag = Mathf.Max(0, _replayedFrames - stamp.pair);
        _correctionDelaySecondsSum += seconds;
        _correctionDelaySecondsMax = Math.Max(
            _correctionDelaySecondsMax, seconds);
        _correctionPairLagSum += pairLag;
        _correctionPairLagMax = Mathf.Max(
            _correctionPairLagMax, pairLag);
        _correctionPublicationCount++;
    }

    private void RebuildChunkMesh(Vector3Int chunkId, Chunk chunk)
    {
        CacheSamples(chunkId);
        _meshVertices.Clear();
        _meshTriangles.Clear();
        _meshLines.Clear();
        _uniqueLines.Clear();
        Array.Fill(_xEdgeVertices, -1);
        Array.Fill(_yEdgeVertices, -1);
        Array.Fill(_zEdgeVertices, -1);
        Vector3Int origin = chunkId * ChunkSize;

        int triangles = 0;
        for (int z = 0; z < MeshCells; z++)
        for (int y = 0; y < MeshCells; y++)
        for (int x = 0; x < MeshCells; x++)
        {
            int sx = x * MeshStride;
            int sy = y * MeshStride;
            int sz = z * MeshStride;
            int configuration = 0;
            bool observed = true;
            for (int corner = 0; corner < 8; corner++)
            {
                Vector3Int offset = CubeOffsets[corner] * MeshStride;
                int sample = SampleIndex(
                    sx + offset.x, sy + offset.y, sz + offset.z);
                if (_sampleObserved[sample] == 0)
                {
                    observed = false;
                    break;
                }
                float distance = _sampleDistances[sample];
                _cornerDistances[corner] = distance;
                _cornerPositions[corner] = new Vector3(
                    (origin.x + sx + offset.x) * VoxelMeters,
                    (origin.y + sy + offset.y) * VoxelMeters,
                    (origin.z + sz + offset.z) * VoxelMeters);
                if (distance < 0f)
                    configuration |= 1 << corner;
            }
            if (!observed || configuration == 0 || configuration == 255)
                continue;

            for (int edge = 0; edge < 12; edge++)
            {
                _cellEdgeVertices[edge] = -1;
                int a = EdgePairs[edge, 0];
                int b = EdgePairs[edge, 1];
                if ((_cornerDistances[a] < 0f) !=
                    (_cornerDistances[b] < 0f))
                    _cellEdgeVertices[edge] = GetEdgeVertex(
                        sx, sy, sz, edge);
            }
            for (int entry = 0;
                 entry < ScanCoverPaperDmcTables.TriangleEntriesPerIndex;
                 entry += 3)
            {
                int edge0 = ScanCoverPaperDmcTables.TriangleEdge(
                    configuration, entry);
                if (edge0 < 0)
                    break;
                int edge1 = ScanCoverPaperDmcTables.TriangleEdge(
                    configuration, entry + 1);
                int edge2 = ScanCoverPaperDmcTables.TriangleEdge(
                    configuration, entry + 2);
                int a = _cellEdgeVertices[edge2];
                int b = _cellEdgeVertices[edge1];
                int c = _cellEdgeVertices[edge0];
                if (a < 0 || b < 0 || c < 0)
                    continue;
                _meshTriangles.Add(a);
                _meshTriangles.Add(b);
                _meshTriangles.Add(c);
                triangles++;
            }
        }

        // Keep the unfiltered candidate separate from the published Front.
        // A component that is currently below 32 triangles may grow on a
        // later pass; destructively filtering this source would prevent that.
        chunk.candidateVertices = triangles > 0
            ? _meshVertices.ToArray()
            : Array.Empty<Vector3>();
        chunk.candidateTriangles = triangles > 0
            ? _meshTriangles.ToArray()
            : Array.Empty<int>();
        chunk.candidateTriangleCount = triangles;
    }

    private void PublishRemoveSmallCandidateMeshes()
    {
        double start = Time.realtimeSinceStartupAsDouble;
        _orderedChunks.Clear();
        foreach (KeyValuePair<Vector3Int, Chunk> item in _chunks)
            _orderedChunks.Add(item);
        _orderedChunks.Sort(CompareChunks);

        _orderedChunkVertexBases.Clear();
        int vertexCount = 0;
        int triangleCount = 0;
        for (int i = 0; i < _orderedChunks.Count; i++)
        {
            Chunk chunk = _orderedChunks[i].Value;
            _orderedChunkVertexBases.Add(vertexCount);
            vertexCount += chunk.candidateVertices.Length;
            triangleCount += chunk.candidateTriangleCount;
        }

        var union = new UnionFind(vertexCount);
        for (int chunkIndex = 0;
             chunkIndex < _orderedChunks.Count;
             chunkIndex++)
        {
            Chunk chunk = _orderedChunks[chunkIndex].Value;
            int vertexBase = _orderedChunkVertexBases[chunkIndex];
            int[] triangles = chunk.candidateTriangles;
            for (int i = 0; i < triangles.Length; i += 3)
            {
                int a = vertexBase + triangles[i];
                int b = vertexBase + triangles[i + 1];
                int c = vertexBase + triangles[i + 2];
                union.Union(a, b);
                union.Union(b, c);
            }
        }

        // Marching-Cubes chunks duplicate boundary vertices. Weld positions at
        // 0.1 mm only for connectivity classification; geometry is unchanged.
        var seamOwners = new Dictionary<QuantizedPosition, int>(vertexCount);
        for (int chunkIndex = 0;
             chunkIndex < _orderedChunks.Count;
             chunkIndex++)
        {
            Vector3[] vertices =
                _orderedChunks[chunkIndex].Value.candidateVertices;
            int vertexBase = _orderedChunkVertexBases[chunkIndex];
            for (int i = 0; i < vertices.Length; i++)
            {
                int global = vertexBase + i;
                var key = new QuantizedPosition(vertices[i]);
                if (seamOwners.TryGetValue(key, out int owner))
                    union.Union(global, owner);
                else
                    seamOwners.Add(key, global);
            }
        }

        var trianglesByRoot = new Dictionary<int, int>();
        for (int chunkIndex = 0;
             chunkIndex < _orderedChunks.Count;
             chunkIndex++)
        {
            Chunk chunk = _orderedChunks[chunkIndex].Value;
            int vertexBase = _orderedChunkVertexBases[chunkIndex];
            int[] triangles = chunk.candidateTriangles;
            for (int i = 0; i < triangles.Length; i += 3)
            {
                int root = union.Find(vertexBase + triangles[i]);
                if (trianglesByRoot.TryGetValue(root, out int count))
                    trianglesByRoot[root] = count + 1;
                else
                    trianglesByRoot.Add(root, 1);
            }
        }

        bool applyCleanup = triangleCount < RemoveSmallTriangleGate;
        var removedRoots = new HashSet<int>();
        int removedTriangles = 0;
        if (applyCleanup)
        {
            foreach (KeyValuePair<int, int> item in trianglesByRoot)
            {
                // Recovered native rule is strictly less than 32. A component
                // with exactly 32 triangles must survive.
                if (item.Value >= RemoveSmallTriangleThreshold)
                    continue;
                removedRoots.Add(item.Key);
                removedTriangles += item.Value;
            }
        }

        // XRT operates on the complete extracted mesh, not independently on
        // rendering chunks. Weld duplicated seam vertices into one topology
        // while retaining each triangle's original publishing chunk.
        var compactVertices = new List<Vector3>(vertexCount);
        var compactTriangles = new List<int>(triangleCount * 3);
        var compactOwners = new List<int>(triangleCount);
        var weldedVertices =
            new Dictionary<QuantizedPosition, int>(vertexCount);
        for (int chunkIndex = 0;
             chunkIndex < _orderedChunks.Count;
             chunkIndex++)
        {
            Chunk chunk = _orderedChunks[chunkIndex].Value;
            int vertexBase = _orderedChunkVertexBases[chunkIndex];
            int[] triangles = chunk.candidateTriangles;
            for (int i = 0; i < triangles.Length; i += 3)
            {
                if (removedRoots.Contains(
                        union.Find(vertexBase + triangles[i])))
                    continue;
                int a = RemapWeldedVertex(
                    chunk.candidateVertices[triangles[i]],
                    compactVertices, weldedVertices);
                int b = RemapWeldedVertex(
                    chunk.candidateVertices[triangles[i + 1]],
                    compactVertices, weldedVertices);
                int c = RemapWeldedVertex(
                    chunk.candidateVertices[triangles[i + 2]],
                    compactVertices, weldedVertices);
                if (a == b || b == c || c == a)
                    continue;
                compactTriangles.Add(a);
                compactTriangles.Add(b);
                compactTriangles.Add(c);
                compactOwners.Add(chunkIndex);
            }
        }

        LastPreCleanupTriangleCount = triangleCount;
        LastPreCleanupComponentCount = trianglesByRoot.Count;
        LastRemovedSmallComponentCount = removedRoots.Count;
        LastRemovedSmallTriangleCount = removedTriangles;
        LastPostCleanupComponentCount =
            trianglesByRoot.Count - removedRoots.Count;
        LastComponentCleanupMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
        var ownerChunkIds = new Vector3Int[_orderedChunks.Count];
        for (int i = 0; i < _orderedChunks.Count; i++)
            ownerChunkIds[i] = _orderedChunks[i].Key;
        var publicationChunkIds =
            new Vector3Int[_meshPassPublicationChunks.Count];
        _meshPassPublicationChunks.CopyTo(publicationChunkIds);
        var input = new RoomSetupReplicaV2XrtSimplifier.Input
        {
            vertices = compactVertices.ToArray(),
            triangles = compactTriangles.ToArray(),
            triangleOwners = compactOwners.ToArray(),
            ownerChunkIds = ownerChunkIds,
            publicationChunkIds = publicationChunkIds,
            resetGeneration = _resetGeneration,
            serial = ++_simplificationSerial
        };
        QueueSimplification(input);
        Debug.Log(
            $"[RoomSetupReplicaV2] RemoveSmall<32 applied={applyCleanup} " +
            $"components={LastPreCleanupComponentCount}->" +
            $"{LastPostCleanupComponentCount} triangles=" +
            $"{LastPreCleanupTriangleCount}->" +
            $"{compactTriangles.Count / 3} " +
            $"removed={LastRemovedSmallTriangleCount} " +
            $"ms={LastComponentCleanupMilliseconds:F1}");
    }

    private static int RemapWeldedVertex(
        Vector3 position,
        List<Vector3> vertices,
        Dictionary<QuantizedPosition, int> weldedVertices)
    {
        var key = new QuantizedPosition(position);
        if (weldedVertices.TryGetValue(key, out int mapped))
            return mapped;
        mapped = vertices.Count;
        weldedVertices.Add(key, mapped);
        vertices.Add(position);
        return mapped;
    }

    private void QueueSimplification(
        RoomSetupReplicaV2XrtSimplifier.Input input)
    {
        // Mirror the recovered single-flight publication behavior. Meshing
        // can continue while one immutable snapshot is simplified; only the
        // newest waiting snapshot is retained.
        if (_simplificationTask != null)
        {
            _pendingSimplificationInput = input;
            return;
        }
        StartSimplification(input);
    }

    private void StartSimplification(
        RoomSetupReplicaV2XrtSimplifier.Input input)
    {
        _simplificationTask = Task.Run(
            () => RoomSetupReplicaV2XrtSimplifier.Run(input));
    }

    private void PollSimplification()
    {
        Task<RoomSetupReplicaV2XrtSimplifier.Result> task =
            _simplificationTask;
        if (task == null || !task.IsCompleted)
            return;
        _simplificationTask = null;
        try
        {
            RoomSetupReplicaV2XrtSimplifier.Result result = task.Result;
            if (result.resetGeneration == _resetGeneration)
                PublishSimplificationResult(result);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] XRT simplification failed: " +
                exception);
        }
        RoomSetupReplicaV2XrtSimplifier.Input pending =
            _pendingSimplificationInput;
        _pendingSimplificationInput = null;
        if (pending != null && pending.resetGeneration == _resetGeneration)
            StartSimplification(pending);
    }

    private void PublishSimplificationResult(
        RoomSetupReplicaV2XrtSimplifier.Result result)
    {
        LastSimplificationApplied = result.applied;
        LastPreSimplificationVertexCount = result.inputVertices;
        LastPreSimplificationTriangleCount = result.inputTriangles;
        LastSimplificationTargetVertexCount = result.targetVertices;
        LastPostSimplificationVertexCount = result.outputVertices;
        LastPostSimplificationTriangleCount = result.outputTriangles;
        LastAcceptedCollapseCount = result.acceptedCollapses;
        LastSimplificationFailedBatchCount = result.failedBatches;
        LastSimplificationTopologyRejectCount =
            result.topologyRejectedCandidates;
        LastSimplificationNormalRejectCount =
            result.normalRejectedCandidates;
        LastSimplificationErrorRejectCount =
            result.errorRejectedCandidates;
        LastSimplificationStopReason = result.stopReason;
        LastSimplificationMilliseconds =
            (float)result.elapsedMilliseconds;

        for (int owner = 0; owner < result.ownerChunkIds.Length; owner++)
        {
            Vector3Int chunkId = result.ownerChunkIds[owner];
            if (_chunks.TryGetValue(chunkId, out Chunk chunk))
                PublishSimplifiedChunkMesh(
                    chunkId, chunk, owner, result);
        }
        PreviewTriangleCount = 0;
        foreach (Chunk chunk in _chunks.Values)
            PreviewTriangleCount += chunk.triangleCount;
        for (int i = 0; i < result.publicationChunkIds.Length; i++)
            RecordCorrectionPublication(result.publicationChunkIds[i]);
        if (IsReplaying && PreviewTriangleCount > 0 &&
            _firstMeshLatencySeconds < 0f)
        {
            _firstMeshLatencySeconds = (float)(
                Time.realtimeSinceStartupAsDouble - _replayStartRealtime);
            _firstMeshPair = _replayedFrames;
        }
        Debug.Log(
            $"[RoomSetupReplicaV2] XRT simplify applied={result.applied} " +
            $"vertices={result.inputVertices}->{result.outputVertices} " +
            $"target={result.targetVertices} triangles=" +
            $"{result.inputTriangles}->{result.outputTriangles} " +
            $"collapses={result.acceptedCollapses} " +
            $"failedBatches={result.failedBatches} " +
            $"stop={result.stopReason} ms={result.elapsedMilliseconds:F1}");
    }

    private void PublishSimplifiedChunkMesh(
        Vector3Int chunkId,
        Chunk chunk,
        int owner,
        RoomSetupReplicaV2XrtSimplifier.Result result)
    {
        _meshVertices.Clear();
        _meshTriangles.Clear();
        _meshLines.Clear();
        _uniqueLines.Clear();
        _publishedVertexRemap.Clear();

        int[] triangles = result.triangles;
        int faceCount = triangles.Length / 3;
        for (int face = 0; face < faceCount; face++)
        {
            if (face >= result.triangleOwners.Length ||
                result.triangleOwners[face] != owner)
                continue;
            int index = face * 3;
            int a = RemapPublishedVertex(
                result.vertices, triangles[index]);
            int b = RemapPublishedVertex(
                result.vertices, triangles[index + 1]);
            int c = RemapPublishedVertex(
                result.vertices, triangles[index + 2]);
            _meshTriangles.Add(a);
            _meshTriangles.Add(b);
            _meshTriangles.Add(c);
            AddLine(a, b);
            AddLine(b, c);
            AddLine(c, a);
        }

        int publishedTriangleCount = _meshTriangles.Count / 3;
        if (publishedTriangleCount <= 0)
        {
            if (chunk.previewObject != null)
            {
                Mesh empty = chunk.backMesh;
                empty.Clear(false);
                chunk.previewFilter.sharedMesh = empty;
                chunk.backMesh = chunk.frontMesh;
                chunk.frontMesh = empty;
                chunk.previewObject.SetActive(false);
            }
            chunk.triangleCount = 0;
            return;
        }

        EnsureChunkPreview(chunkId, chunk);
        Mesh staging = chunk.backMesh;
        staging.Clear(false);
        staging.SetVertices(_meshVertices);
        staging.subMeshCount = 2;
        staging.SetIndices(_meshTriangles, MeshTopology.Triangles, 0, false);
        staging.SetIndices(_meshLines, MeshTopology.Lines, 1, false);
        // A legal collapse may retain the endpoint owned by a neighbouring
        // extraction chunk. Recalculate bounds from the actual surviving
        // vertices rather than assuming every point remains inside its old
        // chunk AABB.
        staging.RecalculateBounds();
        staging.UploadMeshData(false);
        chunk.previewFilter.sharedMesh = staging;
        chunk.backMesh = chunk.frontMesh;
        chunk.frontMesh = staging;
        chunk.triangleCount = publishedTriangleCount;
        chunk.previewObject.SetActive(showReplica);
    }

    private int RemapPublishedVertex(
        Vector3[] vertices, int candidateIndex)
    {
        if (_publishedVertexRemap.TryGetValue(
                candidateIndex, out int published))
            return published;
        published = _meshVertices.Count;
        _publishedVertexRemap.Add(candidateIndex, published);
        _meshVertices.Add(vertices[candidateIndex]);
        return published;
    }

    private void CacheSamples(Vector3Int chunkId)
    {
        Array.Clear(_sampleDistances, 0, _sampleDistances.Length);
        Array.Clear(_sampleObserved, 0, _sampleObserved.Length);
        Vector3Int origin = chunkId * ChunkSize;
        for (int z = 0; z < SampleSize; z += MeshStride)
        for (int y = 0; y < SampleSize; y += MeshStride)
        for (int x = 0; x < SampleSize; x += MeshStride)
        {
            int sample = SampleIndex(x, y, z);
            Vector3Int lattice = origin + new Vector3Int(x, y, z);
            if (!TryResampleDerivedField(lattice, out float distance))
            {
                _passDerivedMissingSamples++;
                continue;
            }
            _sampleObserved[sample] = 1;
            _sampleDistances[sample] = distance;
            _passDerivedValidSamples++;
        }
    }

    private bool TryResampleDerivedField(
        Vector3Int lattice, out float distance)
    {
        // Room Setup's active Resampled mode passes ratio 0.5 to 0x78cd20.
        // Every integer derived coordinate therefore maps to an exact even
        // source coordinate (derived / 0.5), selecting the function's exact
        // half-SDF/half-weight copy branch. Its trilinear branch is retained
        // for non-aligned ratios, but is not the active 5 cm -> 10 cm kernel.
        // `lattice` is already expressed in source-grid coordinates at stride
        // two, so this lookup is the equivalent aligned source sample.
        if (!TryGetVoxel(lattice, out Voxel voxel) ||
            voxel.weight <= 0f ||
            !float.IsFinite(voxel.distance) ||
            !float.IsFinite(voxel.weight))
        {
            distance = 0f;
            return false;
        }
        distance = voxel.distance;
        return true;
    }

    private bool TryGetVoxel(Vector3Int global, out Voxel voxel)
    {
        Vector3Int chunkId = new Vector3Int(
            FloorDivide(global.x, ChunkSize),
            FloorDivide(global.y, ChunkSize),
            FloorDivide(global.z, ChunkSize));
        if (!_chunks.TryGetValue(chunkId, out Chunk chunk))
        {
            voxel = default;
            return false;
        }
        Vector3Int local = global - chunkId * ChunkSize;
        int index = local.x + ChunkSize *
                    (local.y + ChunkSize * local.z);
        voxel = chunk.voxels[index];
        return true;
    }

    private static int FloorDivide(int value, int divisor)
    {
        int quotient = value / divisor;
        int remainder = value % divisor;
        return remainder < 0 ? quotient - 1 : quotient;
    }

    private int GetEdgeVertex(int cellX, int cellY, int cellZ, int edge)
    {
        int a = EdgePairs[edge, 0];
        int b = EdgePairs[edge, 1];
        Vector3Int offsetA = CubeOffsets[a];
        Vector3Int offsetB = CubeOffsets[b];
        int sx = cellX + Mathf.Min(offsetA.x, offsetB.x) * MeshStride;
        int sy = cellY + Mathf.Min(offsetA.y, offsetB.y) * MeshStride;
        int sz = cellZ + Mathf.Min(offsetA.z, offsetB.z) * MeshStride;
        int[] cache;
        int index;
        if (offsetA.x != offsetB.x)
        {
            cache = _xEdgeVertices;
            index = sx + ChunkSize * (sy + SampleSize * sz);
        }
        else if (offsetA.y != offsetB.y)
        {
            cache = _yEdgeVertices;
            index = sx + SampleSize * (sy + ChunkSize * sz);
        }
        else
        {
            cache = _zEdgeVertices;
            index = sx + SampleSize * (sy + SampleSize * sz);
        }
        if (cache[index] >= 0)
            return cache[index];
        float da = _cornerDistances[a];
        float db = _cornerDistances[b];
        float denominator = da - db;
        float alpha = Mathf.Abs(denominator) > 0.000001f
            ? Mathf.Clamp01(da / denominator)
            : 0.5f;
        int vertex = _meshVertices.Count;
        _meshVertices.Add(Vector3.Lerp(
            _cornerPositions[a], _cornerPositions[b], alpha));
        cache[index] = vertex;
        return vertex;
    }

    private void AddLine(int first, int second)
    {
        uint low = (uint)Mathf.Min(first, second);
        uint high = (uint)Mathf.Max(first, second);
        ulong key = ((ulong)low << 32) | high;
        if (!_uniqueLines.Add(key))
            return;
        _meshLines.Add(first);
        _meshLines.Add(second);
    }

    private void EnsurePreviewResources()
    {
        if (_previewRoot == null)
        {
            _previewRoot = new GameObject("[Room Setup Replica V2] Mesh");
            _previewRoot.transform.SetPositionAndRotation(
                Vector3.zero, Quaternion.identity);
            _previewRoot.transform.localScale = Vector3.one;
            _previewRoot.SetActive(showReplica);
            UpdatePreviewTransform();
        }
        if (_lineMaterial == null)
        {
            Shader shader = Resources.Load<Shader>("RoomSetupReplicaV2Preview");
            if (shader == null)
                shader = Shader.Find("Genesis/RoomSetupReplicaV2Preview");
            if (shader != null)
            {
                _lineMaterial = new Material(shader)
                {
                    name = "Room Setup Replica V2 Lines"
                };
                _lineMaterial.SetColor("_BaseColor", lineColor);
            }
        }
        if (_depthMaterial == null)
        {
            Shader shader = Resources.Load<Shader>("RoomSetupReplicaV2Depth");
            if (shader == null)
                shader = Shader.Find("Genesis/RoomSetupReplicaV2Depth");
            if (shader != null)
                _depthMaterial = new Material(shader)
                {
                    name = "Room Setup Replica V2 Depth"
                };
        }
    }

    private void EnsureChunkPreview(Vector3Int chunkId, Chunk chunk)
    {
        if (chunk.previewObject != null)
            return;
        var go = new GameObject(
            $"RS Replica {chunkId.x},{chunkId.y},{chunkId.z}");
        go.transform.SetParent(_previewRoot.transform, false);
        chunk.previewObject = go;
        chunk.previewFilter = go.AddComponent<MeshFilter>();
        chunk.previewRenderer = go.AddComponent<MeshRenderer>();
        chunk.previewRenderer.shadowCastingMode = ShadowCastingMode.Off;
        chunk.previewRenderer.receiveShadows = false;
        chunk.previewRenderer.sharedMaterials =
            new[] { _depthMaterial, _lineMaterial };
        chunk.frontMesh = CreateMesh(chunkId, "front");
        chunk.backMesh = CreateMesh(chunkId, "back");
        chunk.previewFilter.sharedMesh = chunk.frontMesh;
    }

    private static Mesh CreateMesh(Vector3Int chunkId, string suffix)
    {
        var mesh = new Mesh
        {
            name = $"RS Replica {chunkId.x},{chunkId.y},{chunkId.z} {suffix}",
            indexFormat = IndexFormat.UInt32
        };
        mesh.MarkDynamic();
        return mesh;
    }

    private Chunk AcquireChunk()
    {
        return _chunkPool.Count > 0 ? _chunkPool.Pop() : new Chunk();
    }

    private void RecycleChunk(Chunk chunk)
    {
        Array.Clear(chunk.voxels, 0, chunk.voxels.Length);
        chunk.candidateVertices = Array.Empty<Vector3>();
        chunk.candidateTriangles = Array.Empty<int>();
        chunk.candidateTriangleCount = 0;
        chunk.triangleCount = 0;
        if (_chunkPool.Count < 64)
            _chunkPool.Push(chunk);
    }

    private static Vector3Int WorldToChunk(Vector3 world, float chunkWorld)
    {
        return new Vector3Int(
            Mathf.FloorToInt(world.x / chunkWorld),
            Mathf.FloorToInt(world.y / chunkWorld),
            Mathf.FloorToInt(world.z / chunkWorld));
    }

    private static int SampleIndex(int x, int y, int z)
    {
        return x + SampleSize * (y + SampleSize * z);
    }

    private static float ProjectiveDepth(
        Matrix4x4 matrix, float x, float y, float z)
    {
        return matrix.m30 * x + matrix.m31 * y +
               matrix.m32 * z + matrix.m33;
    }

    private static float QuantizeHalf(float value)
    {
        return Mathf.HalfToFloat(Mathf.FloatToHalf(value));
    }

    private static bool Finite(Vector3 value)
    {
        return float.IsFinite(value.x) &&
               float.IsFinite(value.y) &&
               float.IsFinite(value.z);
    }

    private bool TryGetStageTransforms(
        out Matrix4x4 stageToWorld, out Matrix4x4 worldToStage)
    {
        if (_ovrRig == null)
            _ovrRig = FindAnyObjectByType<OVRCameraRig>();
        Transform trackingSpace = _ovrRig != null
            ? _ovrRig.trackingSpace
            : null;
        if (trackingSpace == null)
        {
            stageToWorld = Matrix4x4.identity;
            worldToStage = Matrix4x4.identity;
            return false;
        }

        OVRPose stageToLocal = OVRPlugin
            .GetTrackingTransformRelativePose(OVRPlugin.TrackingOrigin.Stage)
            .ToOVRPose();
        Matrix4x4 stageToLocalMatrix = Matrix4x4.TRS(
            stageToLocal.position, stageToLocal.orientation, Vector3.one);
        stageToWorld = trackingSpace.localToWorldMatrix *
                       stageToLocalMatrix;
        float determinant = stageToWorld.determinant;
        if (!float.IsFinite(determinant) ||
            Mathf.Abs(determinant) < 0.000001f)
        {
            stageToWorld = Matrix4x4.identity;
            worldToStage = Matrix4x4.identity;
            return false;
        }
        worldToStage = stageToWorld.inverse;
        return true;
    }

    private void UpdatePreviewTransform()
    {
        if (_previewRoot == null)
            return;
        if (_replayUsesStageSpace &&
            TryGetStageTransforms(out Matrix4x4 stageToWorld, out _))
        {
            _previewRoot.transform.SetPositionAndRotation(
                stageToWorld.GetColumn(3), stageToWorld.rotation);
            _previewRoot.transform.localScale = Vector3.one;
            return;
        }
        _previewRoot.transform.SetPositionAndRotation(
            Vector3.zero, Quaternion.identity);
        _previewRoot.transform.localScale = Vector3.one;
    }

    private static void EnsureArray(ref float[] array, int length)
    {
        if (array == null || array.Length < length)
            array = new float[length];
    }

    private static void EnsureArray(ref byte[] array, int length)
    {
        if (array == null || array.Length < length)
            array = new byte[length];
    }

    private static void EnsurePixelArray(
        ref NativeArray<Color> array, int length)
    {
        if (array.IsCreated && array.Length >= length)
            return;
        if (array.IsCreated)
            array.Dispose();
        array = new NativeArray<Color>(
            length, Allocator.Persistent,
            NativeArrayOptions.UninitializedMemory);
    }

    private static void ReleaseTarget(ref RenderTexture target)
    {
        if (target == null)
            return;
        target.Release();
        Destroy(target);
        target = null;
    }

    private static void DisposeChunkPreview(Chunk chunk)
    {
        if (chunk.previewObject != null)
            Destroy(chunk.previewObject);
        if (chunk.frontMesh != null)
            Destroy(chunk.frontMesh);
        if (chunk.backMesh != null)
            Destroy(chunk.backMesh);
        chunk.previewObject = null;
        chunk.previewFilter = null;
        chunk.previewRenderer = null;
        chunk.frontMesh = null;
        chunk.backMesh = null;
    }

    private static void DisposeChunk(Chunk chunk)
    {
        DisposeChunkPreview(chunk);
        Array.Clear(chunk.voxels, 0, chunk.voxels.Length);
        chunk.candidateVertices = Array.Empty<Vector3>();
        chunk.candidateTriangles = Array.Empty<int>();
        chunk.candidateTriangleCount = 0;
        chunk.triangleCount = 0;
    }

    private void OnDestroy()
    {
        StopRecording();
        StopReplay();
        ReleaseTarget(ref _leftWorld);
        ReleaseTarget(ref _rightWorld);
        if (_leftPixels.IsCreated) _leftPixels.Dispose();
        if (_rightPixels.IsCreated) _rightPixels.Dispose();
        foreach (Chunk chunk in _chunks.Values)
            DisposeChunk(chunk);
        if (_lineMaterial != null) Destroy(_lineMaterial);
        if (_depthMaterial != null) Destroy(_depthMaterial);
        if (_previewRoot != null) Destroy(_previewRoot);
    }
}

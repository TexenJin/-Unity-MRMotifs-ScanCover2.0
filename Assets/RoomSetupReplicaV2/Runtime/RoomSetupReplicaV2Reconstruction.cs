using System;
using System.Collections.Generic;
using System.IO;
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
    private const int MeshStride = 2;
    private const int MeshCells = ChunkSize / MeshStride;
    private const float MinimumDepthMeters = 0.30f;
    private const float MaximumDepthMeters = 6.00f;
    private const float PositiveTruncationScale = 0.20f;
    private const float NegativeTruncationScale = 0.20f / 3f;
    private const float GradientThresholdSquared = 0.01f;
    private const float MaximumWeight = 1000f;

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
        public int triangleCount;
    }

    [Header("Execution budgets")]
    [SerializeField, Range(0.5f, 12f)]
    private float reconstructionBudgetMilliseconds = 6f;
    [SerializeField, Range(1, 64)]
    private int integrationChunksPerFrame = 32;
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
    public bool HasPendingWork => _readbackPending || _workStage != WorkStage.None;
    public bool PreviewVisible => showReplica;
    public bool IsRecording => _tapeWriter != null;
    public bool IsReplaying => _tapeReader != null;
    public string TapePath => Path.Combine(
        Application.persistentDataPath,
        "RoomSetupReplicaV2", "last-session.rsv2");
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
        try
        {
            _tapeWriter = new RoomSetupReplicaV2Tape.Writer(
                TapePath, _width, _height, maximumRecordedPairs);
            Debug.Log("[RoomSetupReplicaV2] recording: " + TapePath);
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
        Debug.Log(
            $"[RoomSetupReplicaV2] tape closed written={writer.WrittenFrames} " +
            $"dropped={writer.DroppedFrames} path={writer.Path}");
        if (writer.Failure != null)
            Debug.LogError(
                "[RoomSetupReplicaV2] tape writer failed: " + writer.Failure);
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
            _tapeReader = reader;
            _replayedFrames = 0;
            _nextReplayTime = 0f;
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
        _meshPassActive = false;
        _nextMeshTime = 0f;
        AcceptedPairCount = 0;
        DroppedBusyPairCount = 0;
        ReadbackFailureCount = 0;
        WeightedVoxelCount = 0;
        PreviewTriangleCount = 0;
        LastDerivedValidSampleCount = 0;
        LastDerivedMissingSampleCount = 0;
        LastReconstructionMilliseconds = 0f;
        LastMeshingMilliseconds = 0f;
    }

    private void Update()
    {
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
        bool accepted = writer.TryEnqueue(
            _pendingFrame,
            _pendingCaptureTime,
            _leftWorldToClip,
            _rightWorldToClip,
            _leftProjectionInverse,
            _rightProjectionInverse,
            _leftViewInverse,
            _rightViewInverse,
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
                if (_meshQueue.Count > 0 || _meshPassActive)
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
        bool wrote = false;
        byte dirtyMask = 0;
        for (int z = 0; z < ChunkSize; z++)
        for (int y = 0; y < ChunkSize; y++)
        for (int x = 0; x < ChunkSize; x++)
        {
            int local = x + ChunkSize * (y + ChunkSize * z);
            float worldX = (origin.x + x + 0.5f) * VoxelMeters;
            float worldY = (origin.y + y + 0.5f) * VoxelMeters;
            float worldZ = (origin.z + z + 0.5f) * VoxelMeters;
            float clipX = worldToClip.m00 * worldX +
                          worldToClip.m01 * worldY +
                          worldToClip.m02 * worldZ + worldToClip.m03;
            float clipY = worldToClip.m10 * worldX +
                          worldToClip.m11 * worldY +
                          worldToClip.m12 * worldZ + worldToClip.m13;
            float voxelDepth = ProjectiveDepth(
                worldToClip, worldX, worldY, worldZ);
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

    private static bool HasRetainedVoxel(
        Vector3Int chunkId, Chunk chunk, Matrix4x4 worldToClip)
    {
        Vector3Int origin = chunkId * ChunkSize;
        for (int z = 0; z < ChunkSize; z++)
        for (int y = 0; y < ChunkSize; y++)
        for (int x = 0; x < ChunkSize; x++)
        {
            Voxel voxel = chunk.voxels[
                x + ChunkSize * (y + ChunkSize * z)];
            if (voxel.weight <= 0f)
                continue;
            float worldX = (origin.x + x + 0.5f) * VoxelMeters;
            float worldY = (origin.y + y + 0.5f) * VoxelMeters;
            float worldZ = (origin.z + z + 0.5f) * VoxelMeters;
            float viewDepth = ProjectiveDepth(
                worldToClip, worldX, worldY, worldZ);
            if (float.IsFinite(viewDepth) && viewDepth > 0f &&
                voxel.distance < 0.3f * Mathf.Clamp(viewDepth, 1f, 2f))
                return true;
        }
        return false;
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
        public string tape;
        public int inputWidth;
        public int inputHeight;
        public int replayedPairs;
        public int activeChunks;
        public int weightedVoxels;
        public int previewTriangles;
        public int lastMeshPassDerivedValidSamples;
        public int lastMeshPassDerivedMissingSamples;
        public float lastReconstructionMilliseconds;
        public float lastMeshingMilliseconds;
    }

    private void CompleteReplay()
    {
        RoomSetupReplicaV2Tape.Reader reader = _tapeReader;
        _tapeReader = null;
        reader?.Dispose();
        _processingReplayFrame = false;
        try
        {
            string directory = Path.GetDirectoryName(TapePath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            string receiptPath = Path.Combine(
                directory ?? Application.persistentDataPath,
                "last-replay-receipt.json");
            var receipt = new ReplayReceipt
            {
                tape = TapePath,
                inputWidth = reader != null ? reader.Width : _width,
                inputHeight = reader != null ? reader.Height : _height,
                replayedPairs = _replayedFrames,
                activeChunks = ActiveChunkCount,
                weightedVoxels = WeightedVoxelCount,
                previewTriangles = PreviewTriangleCount,
                lastMeshPassDerivedValidSamples =
                    LastDerivedValidSampleCount,
                lastMeshPassDerivedMissingSamples =
                    LastDerivedMissingSampleCount,
                lastReconstructionMilliseconds =
                    LastReconstructionMilliseconds,
                lastMeshingMilliseconds = LastMeshingMilliseconds
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
        return true;
    }

    private void CompleteMeshPass()
    {
        _meshPassActive = false;
        LastDerivedValidSampleCount = _passDerivedValidSamples;
        LastDerivedMissingSampleCount = _passDerivedMissingSamples;
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
            processed++;
            if (Time.realtimeSinceStartupAsDouble - start >= budget)
                break;
        }
        PreviewTriangleCount = 0;
        foreach (Chunk chunk in _chunks.Values)
            PreviewTriangleCount += chunk.triangleCount;
        LastMeshingMilliseconds = (float)(
            (Time.realtimeSinceStartupAsDouble - start) * 1000d);
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
                AddLine(a, b);
                AddLine(b, c);
                AddLine(c, a);
                triangles++;
            }
        }

        if (triangles <= 0)
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
        float chunkWorld = ChunkSize * VoxelMeters;
        staging.bounds = new Bounds(
            new Vector3(
                (origin.x + ChunkSize * 0.5f + 0.5f) * VoxelMeters,
                (origin.y + ChunkSize * 0.5f + 0.5f) * VoxelMeters,
                (origin.z + ChunkSize * 0.5f + 0.5f) * VoxelMeters),
            Vector3.one * (chunkWorld + VoxelMeters));
        staging.UploadMeshData(false);
        chunk.previewFilter.sharedMesh = staging;
        chunk.backMesh = chunk.frontMesh;
        chunk.frontMesh = staging;
        chunk.triangleCount = triangles;
        chunk.previewObject.SetActive(showReplica);
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
        double weightedDistance = 0d;
        double weightSum = 0d;
        for (int z = -1; z <= 0; z++)
        for (int y = -1; y <= 0; y++)
        for (int x = -1; x <= 0; x++)
        {
            if (!TryGetVoxel(
                    lattice + new Vector3Int(x, y, z), out Voxel voxel) ||
                voxel.weight <= 0f ||
                !float.IsFinite(voxel.distance) ||
                !float.IsFinite(voxel.weight))
            {
                distance = 0f;
                return false;
            }
            weightedDistance += voxel.distance * (double)voxel.weight;
            weightSum += voxel.weight;
        }
        if (weightSum <= 0.000001d)
        {
            distance = 0f;
            return false;
        }
        distance = QuantizeHalf((float)(weightedDistance / weightSum));
        return float.IsFinite(distance);
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

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
    private const int GarbageLeafSize = 4;
    private const float GarbageBaseThresholdMeters = 0.30f;
    // libmrengine _INIT_413 initializes the postSdfFusion frequency config to
    // -0.15. postSdfFusion runs only when this value is positive, using a
    // strict lastTimestamp + 1/frequency < currentTimestamp gate.
    private const float NativeGarbageFrequencyDefaultHz = -0.15f;
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
    private const float DirectSupportMeters = VoxelMeters;
    private const float ScarContradictionMeters = DerivedVoxelMeters;
    private const float SameLayerMaximumSeparationMeters = 0.15f;
    private const float ContradictionMinimumDistanceMeters =
        VoxelMeters * 0.5f;
    private const byte SpatialHistoricalPairsThreshold = 3;
    private const byte ContradictionPairsToRetire = 5;
    private const byte ReplacementPairsToPublish = 3;
    private const float ContradictionMinimumMatureWeight = 1f;
    private const float ContradictionAnchorConsistencyMeters = 0.075f;
    // One 5 cm source voxel is already enough to move a derived 10 cm zero
    // crossing by a visually obvious half-cell and create the repeated wall
    // bands seen on-device. Larger thresholds missed those thin scars.
    private const float LayerShiftChallengeMeters = VoxelMeters;
    private const float LayerShiftDistanceConsistencyMeters = 0.025f;
    private const float LayerShiftAnchorConsistencyMeters = 0.15f;
    private const byte LayerShiftPairsToReplace = 2;
    private const float Q75CandidateLayerCellMeters = DerivedVoxelMeters;
    // Compile-time A/B route. Do not use serialized defaults here: Unity hot
    // reload preserves the previously loaded component value across builds.
    private const bool SameLayerSupportEnabled = true;
    private const bool ContradictionRetirementEnabled = false;
    // Production baseline restored to the 2026-10-27 device build.  Keep the
    // q75 implementation available for isolated experiments, but do not let
    // it add retirement/replacement latency to the live A route.
    private const bool ProductionQ75Enabled = false;
    // Evidence-only observer. Device receipts measured 6-9 ms per accepted
    // pair while affectedReconstruction remained false, so production never
    // creates or feeds it.
    private const bool ShadowObserverEnabled = false;
    // V2's logical reconstruction chunks remain the ownership/update unit, but
    // nearby depth-only occluders are submitted through one render page. This
    // is the active-scene counterpart of ScanCover's RESULT page batching.
    private const bool RenderPageBatchingEnabled = true;
    private const int RenderPageSpanChunks = 2;
    // Candidate chunks publish into their affected render pages immediately.
    // Whole-room seam/component classification is retained as a periodic
    // maintenance pass instead of being repeated after every small update.
    private const float GlobalTopologyCleanupIntervalSeconds = 4f;
    private const int RefutationFootprintRadiusPixels = 2;
    private const float RefutationFootprintSpreadMeters = 0.12f;
    private const int MinimumRecordedPairsToCommit = 16;
    // Recovered Room Setup input rule. The public Meta OpenXR provider used by
    // this project exposes only _EnvironmentDepthTexture, not the native
    // uncertainty image, so the rule is receipt-visible but deliberately not
    // applied to a guessed substitute.
    private const float UncertaintyRejectThreshold = 89f;
    private const float MeshIntervalMaximumSeconds = 0.9f;
    private const float MeshIntervalBaseSeconds = 0.15f;
    private const float MeshIntervalBucketSeconds = 0.25f;
    private const int MeshIntervalTriangleBucket = 6000;

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
        CommitQ75History,
        Finalize
    }

    private struct Voxel
    {
        public float distance;
        public float weight;
        // Latest directly observed surface point that supported this source
        // voxel. It is provenance, not another filtering weight.
        public Vector3 layerAnchor;
        public Vector3 contradictionAnchor;
        public Vector3Int contradictionLayerKey;
        public int lastEvidencePair;
        public byte contradictionStreak;
        public sbyte contradictionSign;
        public bool retiredOnce;
        public float replacementDistance;
        public float replacementWeight;
        public Vector3 replacementAnchor;
        public Vector3Int replacementLayerKey;
        public byte replacementStreak;
        public sbyte replacementSign;
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
        public Vector3[] publishedVertices = Array.Empty<Vector3>();
        public int[] publishedTriangles = Array.Empty<int>();
        public int candidateTriangleCount;
        public int triangleCount;
        public int pendingReplacementVoxels;
    }

    private sealed class RenderPage
    {
        public Vector3Int key;
        public GameObject gameObject;
        public MeshFilter filter;
        public MeshRenderer renderer;
        public Mesh frontMesh;
        public Mesh backMesh;
        public GameObject lineObject;
        public MeshFilter lineFilter;
        public MeshRenderer lineRenderer;
        public Mesh lineFrontMesh;
        public Mesh lineBackMesh;
        public readonly HashSet<Vector3Int> members =
            new HashSet<Vector3Int>();
        public readonly List<Vector3> vertices = new List<Vector3>();
        public readonly List<int> triangles = new List<int>();
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
        public float surfaceAreaSquareMeters;
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
        public string spatialReceiptPath = string.Empty;
        public int scarClusters;
        public int crossLayerCandidateTriangles;
        public int historicalShellCandidateTriangles;
        public int insufficientEvidenceTriangles;
    }

    [Serializable]
    private sealed class LiveMeshReceipt
    {
        public string schema;
        public string createdUtc;
        public string arm;
        public string mutation;
        public string coordinateSpace;
        public int acceptedPairs;
        public int droppedBusyPairs;
        public int readbackFailures;
        public int activeChunks;
        public int weightedVoxels;
        public int contradictionVotes;
        public int retiredVoxels;
        public int pendingReplacementVoxels;
        public int replacementPairsToPublish;
        public bool atomicChunkFrontHold;
        public bool productionQ75Enabled;
        public bool q75ModelReady;
        public float q75MaximumRisk;
        public float q75MaximumStereoAnchorSeparationMeters;
        public float q75CandidateLayerCellMeters;
        public long q75QualifiedVoxelObservations;
        public long q75DeferredVoxelObservations;
        public long layerShiftDeferredObservations;
        public long layerShiftPublishedVoxels;
        public int vertices;
        public int triangles;
        public int connectedComponents;
        public float surfaceAreaSquareMeters;
        public string meshPath;
        public string meshSha256;
        public bool renderedDuringCapture;
    }

    public readonly struct ReplayProbeHit
    {
        public readonly int triangle;
        public readonly float distanceMeters;
        public readonly Vector3 barycentric;
        public readonly Vector3 stagePoint;
        public readonly Vector3 stageNormal;
        public readonly Vector3 worldPoint;
        public readonly Vector3 worldNormal;
        public readonly Vector3 stageA;
        public readonly Vector3 stageB;
        public readonly Vector3 stageC;

        public ReplayProbeHit(
            int triangle,
            float distanceMeters,
            Vector3 barycentric,
            Vector3 stagePoint,
            Vector3 stageNormal,
            Vector3 worldPoint,
            Vector3 worldNormal,
            Vector3 stageA,
            Vector3 stageB,
            Vector3 stageC)
        {
            this.triangle = triangle;
            this.distanceMeters = distanceMeters;
            this.barycentric = barycentric;
            this.stagePoint = stagePoint;
            this.stageNormal = stageNormal;
            this.worldPoint = worldPoint;
            this.worldNormal = worldNormal;
            this.stageA = stageA;
            this.stageB = stageB;
            this.stageC = stageC;
        }
    }

    [Serializable]
    private sealed class SpatialTriangleReceipt
    {
        public int triangle;
        public int cluster;
        public Vector3 centroid;
        public Vector3 normal;
        public float areaSquareMeters;
        public int validDepthObservations;
        public int foregroundContradictions;
        public int backgroundContradictions;
        public float nearestAbsoluteResidualMeters;
        public float minimumResidualMeters;
        public float maximumResidualMeters;
        public string classification;
    }

    [Serializable]
    private sealed class SpatialClusterReceipt
    {
        public int cluster;
        public int triangles;
        public Vector3 boundsMin;
        public Vector3 boundsMax;
        public int crossLayerCandidates;
        public int historicalShellCandidates;
        public int insufficientEvidence;
    }

    [Serializable]
    private sealed class SpatialScarReceipt
    {
        public string coordinateSpace;
        public string sourceMeshSha256;
        public float directSupportMeters;
        public float contradictionMeters;
        public string classificationRule;
        public int totalTriangles;
        public int unsupportedTriangles;
        public int clusters;
        public SpatialClusterReceipt[] clusterReceipts;
        public SpatialTriangleReceipt[] triangleReceipts;
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

    [Header("White-box lifecycle")]
    [SerializeField, Tooltip(
        "Recovered postSdfFusion frequency in Hz. The current native " +
        "binary defaults to -0.15, which disables GC. Set a positive value " +
        "only when a native runtime override has been evidenced.")]
    private float garbageCollectionFrequencyHz =
        NativeGarbageFrequencyDefaultHz;

    [Header("Deterministic input tape")]
    [SerializeField, Range(10, 600)]
    private int maximumRecordedPairs = 180;
    [SerializeField, Range(1f, 30f)]
    private float replayHz = 10f;

    [Header("Replica display")]
    [SerializeField] private bool showReplica = true;
    [SerializeField] private Color lineColor =
        new Color(0.02f, 0.86f, 1f, 0.88f);
    [SerializeField, Range(0.75f, 3f)]
    private float lineWidthPixels = 1.5f;

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
    public int SameLayerRejectedCellCount { get; private set; }
    public int SameLayerComparableEdgeCount { get; private set; }
    public int SameLayerUncomparableEdgeCount { get; private set; }
    public int SameLayerIncompatibleEdgeCount { get; private set; }
    public int ContradictionVoteCount { get; private set; }
    public int RetiredContradictedVoxelCount { get; private set; }
    public long LayerShiftDeferredObservationCount { get; private set; }
    public long LayerShiftPublishedVoxelCount { get; private set; }
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
    public bool CanProbeFinalMesh { get; private set; }
    public string LastReplayMeshPath { get; private set; } = string.Empty;
    public string LastReplayMeshSha256 { get; private set; } = string.Empty;
    public int LastReplayPairCount { get; private set; }
    public int FinalReplayTriangleCount => _finalTriangles.Count / 3;
    public Transform ReplayStageTransform =>
        _previewRoot != null ? _previewRoot.transform : null;
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

    public bool ReadyForSynchronizedDepthPair => !HasPendingWork;
    public string RouteLabel => _abExperimentalArm
        ? "B-atomic-retirement-hold-v2"
        : ProductionQ75Enabled
            ? "A-q75-layer-shift-latch-atomic-v2"
            : "A-old-rules";
    private bool UsesProductionQ75 =>
        ProductionQ75Enabled && !_abExperimentalArm;
    private bool UsesAtomicRetirement =>
        _abExperimentalArm || UsesProductionQ75;

    private readonly Dictionary<Vector3Int, Chunk> _chunks =
        new Dictionary<Vector3Int, Chunk>(256);
    private readonly Stack<Chunk> _chunkPool = new Stack<Chunk>(64);
    private readonly HashSet<Vector3Int> _candidateSet =
        new HashSet<Vector3Int>();
    private readonly List<Vector3Int> _leftCandidates =
        new List<Vector3Int>(256);
    private readonly List<Vector3Int> _rightCandidates =
        new List<Vector3Int>(256);
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
    private readonly List<Vector3Int> _localPublicationChunks =
        new List<Vector3Int>(128);

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
    private bool _integrationSliceActive;
    private Vector3Int _integrationSliceChunkId;
    private int _integrationSliceEye;
    private int _integrationSliceLocal;
    private bool _integrationSliceIsNew;
    private bool _integrationSliceWrote;
    private byte _integrationSliceDirtyMask;
    private Chunk _integrationSliceChunk;
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
    private RoomSetupReplicaV2ShadowMode _shadowMode;
    private RoomSetupReplicaV2FusionGuardQ75 _fusionGuardQ75;
    private bool _abExperimentalArm;
    private string _pendingLiveExportDirectory;
    private string _pendingLiveExportFileName;
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
    private int _simplificationSerial;
    private float _nextGlobalTopologyCleanupTime;
    private bool _globalTopologyCleanupDirty;

    private GameObject _previewRoot;
    private Material _lineMaterial;
    private Material _depthMaterial;
    private GameObject _globalLineObject;
    private MeshFilter _globalLineFilter;
    private MeshRenderer _globalLineRenderer;
    private Mesh _globalLineFrontMesh;
    private Mesh _globalLineBackMesh;
    private int _globalLineIndexCount;
    private readonly Dictionary<Vector3Int, RenderPage> _renderPages =
        new Dictionary<Vector3Int, RenderPage>();
    private bool _meshPassActive;
    private float _nextMeshTime;
    private float _lastScheduledMeshIntervalSeconds;
    private bool _asyncMeshingPending;
    private int _skippedAsyncBusyCount;
    private int _meshPublicationCount;
    private int _garbagePassCount;
    private int _garbageCandidateLeafCount;
    private int _garbageReleasedLeafCount;
    private int _garbageClearedVoxelCount;
    private int _garbageReleasedChunkCount;
    private double _lastGarbageCaptureTime;
    private int _passDerivedValidSamples;
    private int _passDerivedMissingSamples;

    private readonly float[] _sampleDistances = new float[SampleCount];
    private readonly byte[] _sampleObserved = new byte[SampleCount];
    private readonly Vector3[] _sampleLayerAnchors =
        new Vector3[SampleCount];
    private readonly byte[] _sampleLayerObserved = new byte[SampleCount];
    private readonly int[] _sampleLayerPairs = new int[SampleCount];
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
    private readonly List<Vector3> _lineQuadVertices =
        new List<Vector3>(16384);
    private readonly List<Vector3> _lineQuadOtherEndpoints =
        new List<Vector3>(16384);
    private readonly List<Vector2> _lineQuadExtrusions =
        new List<Vector2>(16384);
    private readonly List<int> _meshLines = new List<int>(24576);
    private readonly HashSet<ulong> _uniqueLines = new HashSet<ulong>();
    private readonly Dictionary<Vector3Int, int> _globalLinePointIds =
        new Dictionary<Vector3Int, int>(4096);
    private int _nextGlobalLinePointId;
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
        TryObserveDepthFrame(depth);
    }

    /// <summary>
    /// Starts one callback-latched stereo pair and reports whether ownership of
    /// that exact pair was accepted. The synchronized live A/B harness uses the
    /// return value to make any accidental one-arm submission visible instead
    /// of silently treating two different input streams as an algorithm test.
    /// </summary>
    public bool TryObserveDepthFrame(DepthCapture depth)
    {
        if (IsReplaying || depth == null ||
            !DepthCapture.DepthAvailable || HasPendingWork)
        {
            if (HasPendingWork)
                DroppedBusyPairCount++;
            return false;
        }

        Texture source = depth.ChiselSourceDepthTex;
        if (source == null || depth.ProjInv == null || depth.ProjInv.Length < 2 ||
            depth.ViewInv == null || depth.ViewInv.Length < 2 ||
            depth.Proj == null || depth.Proj.Length < 2 ||
            depth.View == null || depth.View.Length < 2 ||
            !EnsureUnproject() || !EnsureTargets(source.width, source.height))
            return false;

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
        return true;
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
        {
            _shadowMode?.FlushReceipts();
            return;
        }
        RoomSetupReplicaV2Tape.Writer writer = _tapeWriter;
        _tapeWriter = null;
        writer.Dispose();
        _shadowMode?.FlushReceipts();
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
        if (_globalLineObject != null)
            _globalLineObject.SetActive(
                visible && _globalLineIndexCount > 0);
    }

    /// <summary>
    /// Copies scene-tuned execution and presentation settings before a runtime
    /// A/B arm starts processing. Reconstruction state is deliberately not
    /// copied: both arms must build independently from the same stereo pairs.
    /// </summary>
    public void CopyRuntimeConfigurationFrom(
        RoomSetupReplicaV2Reconstruction source)
    {
        if (source == null || source == this)
            return;
        reconstructionBudgetMilliseconds =
            source.reconstructionBudgetMilliseconds;
        integrationChunksPerFrame = source.integrationChunksPerFrame;
        meshingBudgetMilliseconds = source.meshingBudgetMilliseconds;
        meshChunksPerFrame = source.meshChunksPerFrame;
        garbageCollectionFrequencyHz = source.garbageCollectionFrequencyHz;
        maximumRecordedPairs = source.maximumRecordedPairs;
        replayHz = source.replayHz;
        lineColor = source.lineColor;
        lineWidthPixels = source.lineWidthPixels;
    }

    /// <summary>
    /// Selects the experimental B rules on this reconstruction instance before
    /// it receives input. In synchronized diagnostic capture the baseline A and
    /// this B instance run as isolated volumes fed by the same platform frame.
    /// </summary>
    public void ConfigureAsExperimentalB()
    {
        _abExperimentalArm = true;
    }

    public void RequestLiveExport(string directory, string fileName)
    {
        if (string.IsNullOrEmpty(directory) || string.IsNullOrEmpty(fileName))
            return;
        _pendingLiveExportDirectory = directory;
        _pendingLiveExportFileName = fileName;
        Directory.CreateDirectory(directory);
        Debug.Log(
            $"[RoomSetupReplicaV2] export queued route={RouteLabel} " +
            $"pairs={AcceptedPairCount} path={Path.Combine(directory, fileName)}");
    }

    [ContextMenu("Clear Room Setup Replica V2")]
    public void ClearReplica()
    {
        _resetGeneration++;
        _readbackPending = false;
        _workStage = WorkStage.None;
        ResetIntegrationSlice();
        ClearRenderPages();
        foreach (Chunk chunk in _chunks.Values)
            DisposeChunk(chunk);
        _chunks.Clear();
        ClearGlobalLinePreview();
        _chunkPool.Clear();
        _candidateSet.Clear();
        _leftCandidates.Clear();
        _rightCandidates.Clear();
        _garbageCandidates.Clear();
        _touchedChunks.Clear();
        _touchedMasks.Clear();
        _queuedMeshChunks.Clear();
        _meshQueue.Clear();
        _meshDirtyStamps.Clear();
        _meshPassPublicationChunks.Clear();
        _localPublicationChunks.Clear();
        _meshPassActive = false;
        _nextMeshTime = 0f;
        _nextGlobalTopologyCleanupTime = 0f;
        _globalTopologyCleanupDirty = false;
        _lastScheduledMeshIntervalSeconds = 0f;
        _asyncMeshingPending = false;
        _skippedAsyncBusyCount = 0;
        _meshPublicationCount = 0;
        _garbagePassCount = 0;
        _garbageCandidateLeafCount = 0;
        _garbageReleasedLeafCount = 0;
        _garbageClearedVoxelCount = 0;
        _garbageReleasedChunkCount = 0;
        _lastGarbageCaptureTime = 0d;
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
        SameLayerRejectedCellCount = 0;
        SameLayerComparableEdgeCount = 0;
        SameLayerUncomparableEdgeCount = 0;
        SameLayerIncompatibleEdgeCount = 0;
        ContradictionVoteCount = 0;
        RetiredContradictedVoxelCount = 0;
        LayerShiftDeferredObservationCount = 0;
        LayerShiftPublishedVoxelCount = 0;
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
        CanProbeFinalMesh = false;
        LastReplayMeshPath = string.Empty;
        LastReplayMeshSha256 = string.Empty;
        LastReplayPairCount = 0;
        _pendingLiveExportDirectory = null;
        _pendingLiveExportFileName = null;
        _finalVertices.Clear();
        _finalTriangles.Clear();
        if (!_abExperimentalArm)
        {
            if (ShadowObserverEnabled)
                EnsureShadowMode().ResetShadowSession();
            if (UsesProductionQ75)
                EnsureFusionGuardQ75().ResetSession();
        }
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
        TryCompleteLiveExport();
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
                    {
                        ObserveShadowPair();
                        BeginStage(WorkStage.CandidateRight);
                    }
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
                        if (!IntegrateChunkSlice(
                            _leftCandidates[_leftChunkCursor],
                            _leftPixels, _leftDepth, _leftContinuous,
                            _leftWorldToClip, 0, deadline))
                            goto Done;
                        _leftChunkCursor++;
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
                        if (!IntegrateChunkSlice(
                            _rightCandidates[_rightChunkCursor],
                            _rightPixels, _rightDepth, _rightContinuous,
                            _rightWorldToClip, 1, deadline))
                            goto Done;
                        _rightChunkCursor++;
                        chunkWork++;
                    }
                    else
                    {
                        if (IsGarbagePassDue())
                        {
                            BeginGarbagePass();
                            BeginStage(WorkStage.GarbageCollect);
                        }
                        else
                        {
                            BeginFinalizeStage();
                        }
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
                        BeginFinalizeStage();
                    }
                    break;
                case WorkStage.CommitQ75History:
                    if (_fusionGuardQ75 == null ||
                        _fusionGuardQ75.CommitPairSlice(deadline))
                        BeginStage(WorkStage.Finalize);
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

    private void BeginFinalizeStage()
    {
        if (UsesProductionQ75 && _fusionGuardQ75 != null)
        {
            _fusionGuardQ75.BeginCommitPair();
            BeginStage(WorkStage.CommitQ75History);
            return;
        }
        BeginStage(WorkStage.Finalize);
    }

    private RoomSetupReplicaV2ShadowMode EnsureShadowMode()
    {
        if (_shadowMode == null)
            _shadowMode = GetComponent<RoomSetupReplicaV2ShadowMode>();
        if (_shadowMode == null)
            _shadowMode = gameObject.AddComponent<RoomSetupReplicaV2ShadowMode>();
        return _shadowMode;
    }

    private RoomSetupReplicaV2FusionGuardQ75 EnsureFusionGuardQ75()
    {
        if (_fusionGuardQ75 == null)
            _fusionGuardQ75 =
                GetComponent<RoomSetupReplicaV2FusionGuardQ75>();
        if (_fusionGuardQ75 == null)
            _fusionGuardQ75 =
                gameObject.AddComponent<RoomSetupReplicaV2FusionGuardQ75>();
        return _fusionGuardQ75;
    }

    private void ObserveShadowPair()
    {
        // The B arm tests only boundary-guarded direct refutation. Running a
        // second receipt writer would race A for the same JSONL and would add
        // a different admission mechanism to this experiment.
        if (_abExperimentalArm)
            return;
        if (UsesProductionQ75)
        {
            EnsureFusionGuardQ75().BeginPair(
                _width,
                _height,
                _leftPixels,
                _rightPixels,
                _leftDepth,
                _rightDepth,
                _leftContinuous,
                _rightContinuous,
                _leftWorldToClip,
                _rightWorldToClip,
                _leftEye,
                _rightEye);
            // The old read-only shadow observer computes a second, coarser
            // stereo/temporal state machine.  Once q75 is production, running
            // both would add cost without affecting the selected A route.
            return;
        }
        if (!ShadowObserverEnabled)
            return;
        RoomSetupReplicaV2ShadowMode shadow = EnsureShadowMode();
        if (shadow == null || !shadow.ShadowModeEnabled)
            return;
        shadow.ObservePair(
            _pendingFrame,
            _pendingCaptureTime,
            _processingReplayFrame,
            _width,
            _height,
            _leftPixels,
            _rightPixels,
            _leftDepth,
            _rightDepth,
            _leftContinuous,
            _rightContinuous,
            _leftWorldToClip,
            _rightWorldToClip);
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
                    _simplificationTask != null)
                {
                    _nextReplayTime = Time.unscaledTime + 0.05f;
                    return;
                }
                if (_globalTopologyCleanupDirty)
                {
                    PublishRemoveSmallCandidateMeshes();
                    _globalTopologyCleanupDirty = false;
                    _nextGlobalTopologyCleanupTime = Time.unscaledTime +
                        GlobalTopologyCleanupIntervalSeconds;
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

    private bool IntegrateChunkSlice(
        Vector3Int chunkId,
        NativeArray<Color> pixels,
        float[] depths,
        byte[] accepted,
        Matrix4x4 worldToClip,
        int eye,
        double deadline)
    {
        if (!_integrationSliceActive)
        {
            _integrationSliceActive = true;
            _integrationSliceChunkId = chunkId;
            _integrationSliceEye = eye;
            _integrationSliceLocal = 0;
            _integrationSliceWrote = false;
            _integrationSliceDirtyMask = 0;
            _integrationSliceIsNew =
                !_chunks.TryGetValue(chunkId, out _integrationSliceChunk);
            if (_integrationSliceIsNew)
            {
                _integrationSliceChunk = AcquireChunk();
                _chunks.Add(chunkId, _integrationSliceChunk);
            }
        }
        else if (_integrationSliceChunkId != chunkId ||
                 _integrationSliceEye != eye)
        {
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 integration slice changed owner before completion");
        }

        Chunk chunk = _integrationSliceChunk;

        Vector3Int origin = chunkId * ChunkSize;
        PrepareProjectionTerms(origin, worldToClip);
        while (_integrationSliceLocal < VoxelsPerChunk)
        {
            if ((_integrationSliceLocal & 15) == 0 &&
                Time.realtimeSinceStartupAsDouble >= deadline)
                return false;
            int local = _integrationSliceLocal++;
            int x = local % ChunkSize;
            int yz = local / ChunkSize;
            int y = yz % ChunkSize;
            int z = yz / ChunkSize;
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
            int pair = _processingReplayFrame
                ? _replayedFrames + 1
                : AcceptedPairCount + 1;
            Color surface = pixels[index];
            Vector3 layerAnchor = new Vector3(
                surface.r, surface.g, surface.b);
            Vector3 voxelWorld = new Vector3(
                (origin.x + x) * VoxelMeters,
                (origin.y + y) * VoxelMeters,
                (origin.z + z) * VoxelMeters);
            if (UsesAtomicRetirement &&
                old.retiredOnce && old.weight <= 0f)
            {
                Vector3 replacementAnchor = layerAnchor;
                if (UsesProductionQ75 &&
                    !TryQualifyQ75Voxel(
                        eye, index, voxelWorld,
                        out replacementAnchor, out _))
                {
                    // Missing, conflicting, or high-risk evidence defers.  It
                    // never erases the old Front and never resets a qualified
                    // replacement streak.
                    continue;
                }
                Voxel replacement = AdvanceReplacementCandidate(
                    old, sampleDistance, sampleWeight, replacementAnchor,
                    CandidateLayerKey(replacementAnchor), pair,
                    out bool publishedReplacement);
                chunk.voxels[local] = replacement;
                if (publishedReplacement)
                {
                    chunk.pendingReplacementVoxels = Mathf.Max(
                        0, chunk.pendingReplacementVoxels - 1);
                    WeightedVoxelCount++;
                }
                if (MeshRelevantChange(old, replacement))
                    AddDirtyVoxel(ref _integrationSliceDirtyMask, local);
                _integrationSliceWrote = true;
                continue;
            }
            if (UsesProductionQ75 && old.weight > 0f &&
                !old.retiredOnce && old.replacementWeight > 0f &&
                old.lastEvidencePair == pair)
            {
                // The other eye already established this pair's stereo
                // challenger. Do not let the second integration pass clear it
                // merely because its projective TSDF residual is slightly
                // closer to the old dominant value.
                _integrationSliceWrote = true;
                continue;
            }
            byte contradictionStreak = old.contradictionStreak;
            Vector3 contradictionAnchor = old.contradictionAnchor;
            Vector3Int contradictionLayerKey =
                old.contradictionLayerKey;
            sbyte contradictionSign = old.contradictionSign;
            int lastEvidencePair = old.lastEvidencePair;
            bool footprintSupported = !_abExperimentalArm ||
                HasInteriorRefutationFootprint(depths, px, py);
            bool matureDominant = !old.retiredOnce &&
                old.weight >= ContradictionMinimumMatureWeight &&
                Mathf.Abs(old.distance) >=
                    ContradictionMinimumDistanceMeters;
            bool signOpposesDominant = matureDominant &&
                sampleDistance != 0f &&
                Mathf.Sign(old.distance) != Mathf.Sign(sampleDistance);
            bool directOpposition = signOpposesDominant &&
                Mathf.Abs(sampleDistance) >=
                    ContradictionMinimumDistanceMeters;
            bool matureLayerSurface = UsesProductionQ75 &&
                !old.retiredOnce &&
                old.weight >= ContradictionMinimumMatureWeight &&
                Finite(old.layerAnchor) &&
                old.layerAnchor.sqrMagnitude > 0.000001f;
            bool sameSignLayerShift = matureLayerSurface &&
                !signOpposesDominant &&
                Mathf.Abs(sampleDistance - old.distance) >=
                    LayerShiftChallengeMeters;
            if (sameSignLayerShift)
            {
                RoomSetupReplicaV2FusionGuardQ75 guard =
                    EnsureFusionGuardQ75();
                if (guard == null || !guard.ModelReady ||
                    !guard.TryGetStereoSurfaceAnchor(
                        eye, index, out Vector3 stereoAnchor))
                {
                    LayerShiftDeferredObservationCount++;
                    continue;
                }
                Voxel shifted = AdvanceLayerShiftCandidate(
                    old, sampleDistance, sampleWeight, stereoAnchor,
                    CandidateLayerKey(stereoAnchor), pair,
                    out bool publishedShift);
                chunk.voxels[local] = shifted;
                if (publishedShift)
                {
                    LayerShiftPublishedVoxelCount++;
                    if (MeshRelevantChange(old, shifted))
                        AddDirtyVoxel(ref _integrationSliceDirtyMask, local);
                }
                else
                {
                    LayerShiftDeferredObservationCount++;
                }
                _integrationSliceWrote = true;
                continue;
            }
            // replacement* is otherwise unused while a voxel still has a
            // mature dominant surface. Returning to that surface, or entering
            // the signed contradiction route, cancels a stale layer-shift
            // challenger without touching the dominant TSDF value.
            ClearLayerShiftCandidate(ref old);
            bool q75Qualified = false;
            Vector3 qualifiedAnchor = layerAnchor;
            sbyte qualifiedSign = sampleDistance > 0f
                ? (sbyte)1
                : (sbyte)-1;
            if (UsesProductionQ75 && directOpposition)
            {
                q75Qualified = TryQualifyQ75Voxel(
                    eye, index, voxelWorld,
                    out qualifiedAnchor, out qualifiedSign);
            }
            bool oppositeDirectObservation = directOpposition &&
                footprintSupported &&
                (!UsesProductionQ75 || q75Qualified);
            if (old.weight > 0f && pair != lastEvidencePair)
            {
                if (oppositeDirectObservation)
                {
                    sbyte sampleSign = UsesProductionQ75
                        ? qualifiedSign
                        : sampleDistance > 0f
                            ? (sbyte)1
                            : (sbyte)-1;
                    Vector3 candidateAnchor = UsesProductionQ75
                        ? qualifiedAnchor
                        : layerAnchor;
                    Vector3Int candidateLayerKey =
                        CandidateLayerKey(candidateAnchor);
                    bool consistentCandidate =
                        contradictionStreak > 0 &&
                        contradictionSign == sampleSign &&
                        (!UsesProductionQ75 ||
                         contradictionLayerKey == candidateLayerKey) &&
                        Finite(candidateAnchor) &&
                        Finite(contradictionAnchor) &&
                        Vector3.Distance(
                            candidateAnchor, contradictionAnchor) <=
                            ContradictionAnchorConsistencyMeters;
                    contradictionStreak = consistentCandidate
                        ? (byte)Mathf.Min(
                            byte.MaxValue, contradictionStreak + 1)
                        : (byte)1;
                    contradictionAnchor = candidateAnchor;
                    contradictionLayerKey = candidateLayerKey;
                    contradictionSign = sampleSign;
                    ContradictionVoteCount++;
                    lastEvidencePair = pair;
                }
                else if (!UsesProductionQ75 || !signOpposesDominant)
                {
                    contradictionStreak = 0;
                    contradictionAnchor = default;
                    contradictionLayerKey = default;
                    contradictionSign = 0;
                    lastEvidencePair = pair;
                }
            }
            float retained = old.weight > 0f
                ? Mathf.Min(old.weight, MaximumWeight - sampleWeight)
                : 0f;
            bool retiredNow = false;
            if ((ContradictionRetirementEnabled || UsesAtomicRetirement) &&
                contradictionStreak >= ContradictionPairsToRetire)
            {
                // Retirement is local and evidence driven. Nothing happens
                // to voxels that are absent or outside the direct depth view.
                retained = 0f;
                contradictionStreak = 0;
                contradictionAnchor = default;
                contradictionLayerKey = default;
                contradictionSign = 0;
                retiredNow = true;
                RetiredContradictedVoxelCount++;
            }
            retained = Mathf.Max(0f, retained);
            Voxel updated;
            if (retiredNow)
            {
                // Do not turn the fifth contradictory sample directly into a
                // new zero crossing. Quarantine it as a replacement candidate
                // and keep this chunk's previous Front until all its pending
                // replacements have survived additional direct observations.
                updated = new Voxel
                {
                    distance = old.distance,
                    weight = 0f,
                    layerAnchor = old.layerAnchor,
                    lastEvidencePair = pair,
                    retiredOnce = true,
                    replacementDistance = sampleDistance,
                    replacementWeight = sampleWeight,
                    replacementAnchor = UsesProductionQ75
                        ? qualifiedAnchor
                        : layerAnchor,
                    replacementLayerKey = CandidateLayerKey(
                        UsesProductionQ75
                            ? qualifiedAnchor
                            : layerAnchor),
                    replacementStreak = 0,
                    replacementSign = sampleDistance > 0f
                        ? (sbyte)1
                        : (sbyte)-1
                };
                chunk.pendingReplacementVoxels++;
                WeightedVoxelCount = Mathf.Max(0, WeightedVoxelCount - 1);
            }
            else if (UsesProductionQ75 && signOpposesDominant)
            {
                // This is the central production change: a mature surface is
                // not averaged toward zero by one opposing snapshot.  A q75
                // qualified observation can advance its isolated challenger;
                // everything else simply defers.
                old.contradictionAnchor = contradictionAnchor;
                old.contradictionLayerKey = contradictionLayerKey;
                old.lastEvidencePair = lastEvidencePair;
                old.contradictionStreak = contradictionStreak;
                old.contradictionSign = contradictionSign;
                chunk.voxels[local] = old;
                _integrationSliceWrote = true;
                continue;
            }
            else
            {
                float newWeight = retained + sampleWeight;
                updated = new Voxel
                {
                    distance = QuantizeHalf(retained > 0f
                        ? (old.distance * retained +
                           sampleDistance * sampleWeight) / newWeight
                        : sampleDistance),
                    weight = QuantizeHalf(newWeight),
                    layerAnchor = Finite(layerAnchor)
                        ? layerAnchor
                        : old.layerAnchor,
                    contradictionAnchor = contradictionAnchor,
                    contradictionLayerKey = contradictionLayerKey,
                    lastEvidencePair = lastEvidencePair == 0
                        ? pair
                        : lastEvidencePair,
                    contradictionStreak = contradictionStreak,
                    contradictionSign = contradictionSign,
                    retiredOnce = old.retiredOnce
                };
            }
            chunk.voxels[local] = updated;
            if (old.weight <= 0f && updated.weight > 0f)
                WeightedVoxelCount++;
            if (MeshRelevantChange(old, updated))
                AddDirtyVoxel(ref _integrationSliceDirtyMask, local);
            _integrationSliceWrote = true;

        }

        if (!_integrationSliceWrote && _integrationSliceIsNew)
        {
            _chunks.Remove(chunkId);
            RecycleChunk(chunk);
        }
        else if ((_integrationSliceDirtyMask & 8) != 0)
        {
            RecordTouched(chunkId, _integrationSliceDirtyMask);
        }
        ResetIntegrationSlice();
        return true;
    }

    private void ResetIntegrationSlice()
    {
        _integrationSliceActive = false;
        _integrationSliceChunkId = default;
        _integrationSliceEye = 0;
        _integrationSliceLocal = 0;
        _integrationSliceIsNew = false;
        _integrationSliceWrote = false;
        _integrationSliceDirtyMask = 0;
        _integrationSliceChunk = null;
    }

    private static bool MeshRelevantChange(Voxel old, Voxel updated)
    {
        bool oldObserved = old.weight > 0f;
        bool updatedObserved = updated.weight > 0f;
        if (oldObserved != updatedObserved)
            return true;
        if (!oldObserved)
            return false;
        // Room Setup rebuilds its derived mesh field from the current source
        // field. Do not add a ScanCover-specific geometry-change threshold.
        return old.distance != updated.distance;
    }

    private bool TryQualifyQ75Voxel(
        int eye,
        int pixel,
        Vector3 voxelWorld,
        out Vector3 anchor,
        out sbyte sign)
    {
        anchor = default;
        sign = 0;
        RoomSetupReplicaV2FusionGuardQ75 guard = EnsureFusionGuardQ75();
        return guard != null && guard.ModelReady &&
               guard.TryQualifyVoxelObservation(
                   eye, pixel, voxelWorld, out anchor, out sign);
    }

    private static Vector3Int CandidateLayerKey(Vector3 anchor)
    {
        return new Vector3Int(
            Mathf.RoundToInt(anchor.x / Q75CandidateLayerCellMeters),
            Mathf.RoundToInt(anchor.y / Q75CandidateLayerCellMeters),
            Mathf.RoundToInt(anchor.z / Q75CandidateLayerCellMeters));
    }

    private static Voxel AdvanceReplacementCandidate(
        Voxel old,
        float sampleDistance,
        float sampleWeight,
        Vector3 layerAnchor,
        Vector3Int layerKey,
        int pair,
        out bool published)
    {
        if (old.lastEvidencePair == pair)
        {
            published = false;
            return old;
        }
        sbyte sampleSign = sampleDistance > 0f ? (sbyte)1 : (sbyte)-1;
        bool consistent = old.replacementWeight > 0f &&
                          old.replacementSign == sampleSign &&
                          old.replacementLayerKey == layerKey &&
                          Finite(layerAnchor) &&
                          Finite(old.replacementAnchor) &&
                          Vector3.Distance(
                              layerAnchor, old.replacementAnchor) <=
                          ContradictionAnchorConsistencyMeters;
        float candidateWeight = consistent
            ? old.replacementWeight + sampleWeight
            : sampleWeight;
        float candidateDistance = consistent
            ? (old.replacementDistance * old.replacementWeight +
               sampleDistance * sampleWeight) /
              Mathf.Max(0.000001f, candidateWeight)
            : sampleDistance;
        byte candidateStreak = consistent
            ? (byte)Mathf.Min(byte.MaxValue, old.replacementStreak + 1)
            : (byte)1;
        published = candidateStreak >= ReplacementPairsToPublish;
        if (published)
        {
            return new Voxel
            {
                distance = QuantizeHalf(candidateDistance),
                weight = QuantizeHalf(candidateWeight),
                layerAnchor = layerAnchor,
                lastEvidencePair = pair,
                retiredOnce = true
            };
        }
        return new Voxel
        {
            distance = old.distance,
            weight = 0f,
            layerAnchor = old.layerAnchor,
            lastEvidencePair = pair,
            retiredOnce = true,
            replacementDistance = candidateDistance,
            replacementWeight = candidateWeight,
            replacementAnchor = layerAnchor,
            replacementLayerKey = layerKey,
            replacementStreak = candidateStreak,
            replacementSign = sampleSign
        };
    }

    private static Voxel AdvanceLayerShiftCandidate(
        Voxel old,
        float sampleDistance,
        float sampleWeight,
        Vector3 layerAnchor,
        Vector3Int layerKey,
        int pair,
        out bool published)
    {
        if (old.lastEvidencePair == pair)
        {
            published = false;
            return old;
        }
        sbyte sampleSign = sampleDistance >= 0f ? (sbyte)1 : (sbyte)-1;
        bool consistent = old.replacementWeight > 0f &&
                          old.replacementSign == sampleSign &&
                          Mathf.Abs(
                              old.replacementDistance - sampleDistance) <=
                          LayerShiftDistanceConsistencyMeters &&
                          Finite(layerAnchor) &&
                          Finite(old.replacementAnchor) &&
                          Vector3.Distance(
                              layerAnchor, old.replacementAnchor) <=
                          LayerShiftAnchorConsistencyMeters;
        float candidateWeight = consistent
            ? old.replacementWeight + sampleWeight
            : sampleWeight;
        float candidateDistance = consistent
            ? (old.replacementDistance * old.replacementWeight +
               sampleDistance * sampleWeight) /
              Mathf.Max(0.000001f, candidateWeight)
            : sampleDistance;
        byte candidateStreak = consistent
            ? (byte)Mathf.Min(byte.MaxValue, old.replacementStreak + 1)
            : (byte)1;
        published = candidateStreak >= LayerShiftPairsToReplace;
        if (published)
        {
            return new Voxel
            {
                distance = QuantizeHalf(candidateDistance),
                weight = QuantizeHalf(candidateWeight),
                layerAnchor = layerAnchor,
                lastEvidencePair = pair,
                retiredOnce = old.retiredOnce
            };
        }
        old.lastEvidencePair = pair;
        old.replacementDistance = candidateDistance;
        old.replacementWeight = candidateWeight;
        old.replacementAnchor = layerAnchor;
        old.replacementLayerKey = layerKey;
        old.replacementStreak = candidateStreak;
        old.replacementSign = sampleSign;
        return old;
    }

    private static void ClearLayerShiftCandidate(ref Voxel voxel)
    {
        if (voxel.weight <= 0f || voxel.retiredOnce ||
            voxel.replacementWeight <= 0f)
            return;
        voxel.replacementDistance = 0f;
        voxel.replacementWeight = 0f;
        voxel.replacementAnchor = default;
        voxel.replacementLayerKey = default;
        voxel.replacementStreak = 0;
        voxel.replacementSign = 0;
    }

    private bool HasInteriorRefutationFootprint(
        float[] depths, int centerX, int centerY)
    {
        if (depths == null ||
            centerX < RefutationFootprintRadiusPixels ||
            centerY < RefutationFootprintRadiusPixels ||
            centerX >= _width - RefutationFootprintRadiusPixels ||
            centerY >= _height - RefutationFootprintRadiusPixels)
            return false;
        float minimum = float.PositiveInfinity;
        float maximum = float.NegativeInfinity;
        for (int y = centerY - RefutationFootprintRadiusPixels;
             y <= centerY + RefutationFootprintRadiusPixels; y++)
        for (int x = centerX - RefutationFootprintRadiusPixels;
             x <= centerX + RefutationFootprintRadiusPixels; x++)
        {
            float value = depths[x + y * _width];
            if (!(value > 0f) || !float.IsFinite(value))
                return false;
            minimum = Mathf.Min(minimum, value);
            maximum = Mathf.Max(maximum, value);
        }
        return maximum - minimum < RefutationFootprintSpreadMeters;
    }

    private void BeginGarbagePass()
    {
        _garbageCandidates.Clear();
        // Native postSdfFusion enumerates allocated sparse leaves intersecting
        // the current camera frustum. Our storage Chunk contains 4x4x4 such
        // 4^3 leaves, so first take a conservative stereo-frustum snapshot of
        // allocated Chunks and classify their leaves below.
        foreach (Vector3Int chunkId in _chunks.Keys)
        {
            Vector3Int origin = chunkId * ChunkSize;
            if (GridBoxIntersectsFrustum(
                    origin, ChunkSize, _leftWorldToClip) ||
                GridBoxIntersectsFrustum(
                    origin, ChunkSize, _rightWorldToClip))
                _garbageCandidates.Add(chunkId);
        }
        _garbagePassCount++;
        _garbageCursor = 0;
    }

    private bool IsGarbagePassDue()
    {
        float frequency = garbageCollectionFrequencyHz;
        if (!float.IsFinite(frequency) || frequency <= 0f)
            return false;
        double captureTime = _pendingCaptureTime;
        if (!double.IsFinite(captureTime))
            return false;
        double interval = 1d / frequency;
        if (!(_lastGarbageCaptureTime + interval < captureTime))
            return false;
        // The recovered native path advances this timestamp when the due
        // postSdfFusion pass starts, not after leaf enumeration finishes.
        _lastGarbageCaptureTime = captureTime;
        return true;
    }

    private void CollectGarbageChunk(Vector3Int chunkId)
    {
        if (!_chunks.TryGetValue(chunkId, out Chunk chunk))
            return;
        Vector3Int origin = chunkId * ChunkSize;
        byte dirtyMask = 0;
        int remainingWeighted = 0;
        for (int leafZ = 0; leafZ < ChunkSize;
             leafZ += GarbageLeafSize)
        for (int leafY = 0; leafY < ChunkSize;
             leafY += GarbageLeafSize)
        for (int leafX = 0; leafX < ChunkSize;
             leafX += GarbageLeafSize)
        {
            Vector3Int leafOrigin = origin +
                new Vector3Int(leafX, leafY, leafZ);
            bool inLeft = GridBoxIntersectsFrustum(
                leafOrigin, GarbageLeafSize, _leftWorldToClip);
            bool inRight = GridBoxIntersectsFrustum(
                leafOrigin, GarbageLeafSize, _rightWorldToClip);
            int weighted = 0;
            bool retained = false;
            float threshold = GarbageRetentionThreshold(
                leafOrigin, inLeft, inRight);
            for (int z = leafZ; z < leafZ + GarbageLeafSize; z++)
            for (int y = leafY; y < leafY + GarbageLeafSize; y++)
            for (int x = leafX; x < leafX + GarbageLeafSize; x++)
            {
                int local = x + ChunkSize * (y + ChunkSize * z);
                Voxel voxel = chunk.voxels[local];
                if (voxel.weight <= 0f)
                    continue;
                weighted++;
                if ((inLeft || inRight) &&
                    float.IsFinite(voxel.distance) &&
                    voxel.distance < threshold)
                    retained = true;
            }
            if (weighted <= 0)
                continue;
            if (!inLeft && !inRight)
            {
                remainingWeighted += weighted;
                continue;
            }
            _garbageCandidateLeafCount++;
            if (retained)
            {
                remainingWeighted += weighted;
                continue;
            }
            for (int z = leafZ; z < leafZ + GarbageLeafSize; z++)
            for (int y = leafY; y < leafY + GarbageLeafSize; y++)
            for (int x = leafX; x < leafX + GarbageLeafSize; x++)
            {
                int local = x + ChunkSize * (y + ChunkSize * z);
                if (chunk.voxels[local].weight <= 0f)
                    continue;
                chunk.voxels[local] = default;
                AddDirtyVoxel(ref dirtyMask, local);
            }
            WeightedVoxelCount -= weighted;
            _garbageReleasedLeafCount++;
            _garbageClearedVoxelCount += weighted;
        }
        WeightedVoxelCount = Mathf.Max(0, WeightedVoxelCount);
        if (remainingWeighted <= 0)
        {
            RecordTouched(chunkId, 7);
            _chunks.Remove(chunkId);
            DisposeChunkPreview(chunk);
            RecycleChunk(chunk);
            _garbageReleasedChunkCount++;
            return;
        }
        if ((dirtyMask & 8) != 0)
            RecordTouched(chunkId, dirtyMask);
    }

    private float GarbageRetentionThreshold(
        Vector3Int leafOrigin, bool inLeft, bool inRight)
    {
        Vector3 world = (Vector3)leafOrigin * VoxelMeters;
        float viewDepth = 0f;
        if (inLeft)
            viewDepth = Mathf.Max(viewDepth, ProjectiveDepth(
                _leftWorldToClip, world.x, world.y, world.z));
        if (inRight)
            viewDepth = Mathf.Max(viewDepth, ProjectiveDepth(
                _rightWorldToClip, world.x, world.y, world.z));
        return GarbageBaseThresholdMeters *
               Mathf.Clamp(viewDepth, 1f, 2f);
    }

    private static bool GridBoxIntersectsFrustum(
        Vector3Int gridOrigin, int gridSize, Matrix4x4 worldToClip)
    {
        bool outsideLeft = true;
        bool outsideRight = true;
        bool outsideBottom = true;
        bool outsideTop = true;
        bool outsideNear = true;
        bool outsideFar = true;
        bool anyFiniteCorner = false;
        for (int z = 0; z <= 1; z++)
        for (int y = 0; y <= 1; y++)
        for (int x = 0; x <= 1; x++)
        {
            float worldX = (gridOrigin.x + x * gridSize) * VoxelMeters;
            float worldY = (gridOrigin.y + y * gridSize) * VoxelMeters;
            float worldZ = (gridOrigin.z + z * gridSize) * VoxelMeters;
            float clipX = worldToClip.m00 * worldX +
                          worldToClip.m01 * worldY +
                          worldToClip.m02 * worldZ + worldToClip.m03;
            float clipY = worldToClip.m10 * worldX +
                          worldToClip.m11 * worldY +
                          worldToClip.m12 * worldZ + worldToClip.m13;
            float depth = ProjectiveDepth(
                worldToClip, worldX, worldY, worldZ);
            if (!float.IsFinite(clipX) || !float.IsFinite(clipY) ||
                !float.IsFinite(depth))
                continue;
            anyFiniteCorner = true;
            outsideLeft &= clipX < -depth;
            outsideRight &= clipX > depth;
            outsideBottom &= clipY < -depth;
            outsideTop &= clipY > depth;
            outsideNear &= depth < MinimumDepthMeters;
            outsideFar &= depth > MaximumDepthMeters;
        }
        return anyFiniteCorner &&
               !(outsideLeft || outsideRight || outsideBottom ||
                 outsideTop || outsideNear || outsideFar);
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
        public bool uncertaintyRuleKnown;
        public float uncertaintyRejectThreshold;
        public string uncertaintyRule;
        public bool uncertaintyInputAvailable;
        public bool uncertaintyApplied;
        public string uncertaintyInputStatus;
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
        public bool adaptiveMeshIntervalEnabled;
        public int adaptiveMeshIntervalMaximumMilliseconds;
        public int adaptiveMeshIntervalBaseMilliseconds;
        public int adaptiveMeshIntervalBucketMilliseconds;
        public int adaptiveMeshIntervalTriangleBucket;
        public float lastScheduledMeshIntervalSeconds;
        public bool asyncMeshingSingleFlight;
        public int skippedAsyncBusy;
        public int meshPublications;
        public string publicationOrder;
        public string garbageCollectionMode;
        public string garbageCollectionCadence;
        public bool garbageCollectionEnabled;
        public float garbageCollectionConfiguredFrequencyHz;
        public float garbageCollectionNativeDefaultFrequencyHz;
        public string garbageCollectionFrequencySource;
        public int garbageLeafEdgeVoxels;
        public int garbageLeafVoxelCount;
        public float garbageBaseThresholdMeters;
        public int garbagePasses;
        public int garbageCandidateLeaves;
        public int garbageReleasedLeaves;
        public int garbageClearedVoxels;
        public int garbageReleasedChunks;
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
        public string spatialScarReceiptPath;
        public int spatialScarClusters;
        public int crossLayerCandidateTriangles;
        public int historicalShellCandidateTriangles;
        public int insufficientEvidenceTriangles;
        public bool sameLayerSupportEnabled;
        public float sameLayerMaximumSeparationMeters;
        public int sameLayerRejectedCells;
        public int sameLayerComparableEdges;
        public int sameLayerUncomparableEdges;
        public int sameLayerIncompatibleEdges;
        public string sameLayerRule;
        public bool contradictionRetirementEnabled;
        public float contradictionMinimumDistanceMeters;
        public float contradictionMinimumMatureWeight;
        public float contradictionAnchorConsistencyMeters;
        public int contradictionPairsToRetire;
        public int replacementPairsToPublish;
        public bool atomicChunkFrontHold;
        public bool productionQ75Enabled;
        public bool q75ModelReady;
        public float q75MaximumRisk;
        public float q75MaximumStereoAnchorSeparationMeters;
        public float q75CandidateLayerCellMeters;
        public long q75QualifiedVoxelObservations;
        public long q75DeferredVoxelObservations;
        public float layerShiftChallengeMeters;
        public float layerShiftDistanceConsistencyMeters;
        public int layerShiftPairsToReplace;
        public long layerShiftDeferredObservations;
        public long layerShiftPublishedVoxels;
        public bool contradictionSingleRetirementPerVoxel;
        public int contradictionVotes;
        public int retiredContradictedVoxels;
        public string contradictionRetirementRule;
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
            LastReplayMeshPath = meshMetrics.path;
            LastReplayMeshSha256 = meshMetrics.sha256;
            LastReplayPairCount = _replayedFrames;
            CanProbeFinalMesh = meshMetrics.triangles > 0;
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
                algorithm = _abExperimentalArm
                    ? "V2BAtomicRetirementHoldV2"
                    : UsesProductionQ75
                        ? "V2AProductionQ75LayerShiftLatchAtomicV2"
                        : (SameLayerSupportEnabled
                        ? "V2CSpatialReceiptSamePairCrossingEdgeLayerOnly"
                        : "V2CSpatialReceiptBaseline"),
                sourceVoxelMeters = VoxelMeters,
                derivedVoxelMeters = DerivedVoxelMeters,
                resampleRatio = ResampleRatio,
                sourceSampleCoordinates = "IntegerGridPoints",
                resampleKernel = "AlignedExactCopy",
                missingSourceRule = "KeepDerivedSampleUnobserved",
                uncertaintyRuleKnown = true,
                uncertaintyRejectThreshold = UncertaintyRejectThreshold,
                uncertaintyRule =
                    "DiscardIfUncertaintyGreaterThan89AndDepthWithinMax",
                uncertaintyInputAvailable = false,
                uncertaintyApplied = false,
                uncertaintyInputStatus =
                    "PrivateDepthCapabilityBuffer_NoExport_NoSubstitution",
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
                simplificationMode = RoomSetupReplicaV2XrtSimplifier.Enabled
                    ? "XRTDirectedEdgeCollapse"
                    : "DisabledForRegular10cmMesh",
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
                adaptiveMeshIntervalEnabled = true,
                adaptiveMeshIntervalMaximumMilliseconds =
                    Mathf.RoundToInt(MeshIntervalMaximumSeconds * 1000f),
                adaptiveMeshIntervalBaseMilliseconds =
                    Mathf.RoundToInt(MeshIntervalBaseSeconds * 1000f),
                adaptiveMeshIntervalBucketMilliseconds =
                    Mathf.RoundToInt(MeshIntervalBucketSeconds * 1000f),
                adaptiveMeshIntervalTriangleBucket =
                    MeshIntervalTriangleBucket,
                lastScheduledMeshIntervalSeconds =
                    _lastScheduledMeshIntervalSeconds,
                asyncMeshingSingleFlight = true,
                skippedAsyncBusy = _skippedAsyncBusyCount,
                meshPublications = _meshPublicationCount,
                publicationOrder =
                    "BuildBack_ReplaceFront_ThenReleaseOldFront",
                garbageCollectionMode = "StereoFrustumLeaf4x4x4",
                garbageCollectionCadence =
                    garbageCollectionFrequencyHz > 0f
                        ? "NativeTimestampGate_OneOverFrequencySeconds"
                        : "DisabledByRecoveredNativeDefaultNonPositiveFrequency",
                garbageCollectionEnabled =
                    float.IsFinite(garbageCollectionFrequencyHz) &&
                    garbageCollectionFrequencyHz > 0f,
                garbageCollectionConfiguredFrequencyHz =
                    garbageCollectionFrequencyHz,
                garbageCollectionNativeDefaultFrequencyHz =
                    NativeGarbageFrequencyDefaultHz,
                garbageCollectionFrequencySource =
                    "libmrengine_INIT_413_ConfigDefault_And_postSdfFusionGate",
                garbageLeafEdgeVoxels = GarbageLeafSize,
                garbageLeafVoxelCount = GarbageLeafSize *
                    GarbageLeafSize * GarbageLeafSize,
                garbageBaseThresholdMeters =
                    GarbageBaseThresholdMeters,
                garbagePasses = _garbagePassCount,
                garbageCandidateLeaves = _garbageCandidateLeafCount,
                garbageReleasedLeaves = _garbageReleasedLeafCount,
                garbageClearedVoxels = _garbageClearedVoxelCount,
                garbageReleasedChunks = _garbageReleasedChunkCount,
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
                    meshMetrics.directDepthAnalysisMilliseconds,
                spatialScarReceiptPath =
                    meshMetrics.spatialReceiptPath,
                spatialScarClusters = meshMetrics.scarClusters,
                crossLayerCandidateTriangles =
                    meshMetrics.crossLayerCandidateTriangles,
                historicalShellCandidateTriangles =
                    meshMetrics.historicalShellCandidateTriangles,
                insufficientEvidenceTriangles =
                    meshMetrics.insufficientEvidenceTriangles,
                sameLayerSupportEnabled = SameLayerSupportEnabled,
                sameLayerMaximumSeparationMeters =
                    SameLayerMaximumSeparationMeters,
                sameLayerRejectedCells = SameLayerRejectedCellCount,
                sameLayerComparableEdges =
                    SameLayerComparableEdgeCount,
                sameLayerUncomparableEdges =
                    SameLayerUncomparableEdgeCount,
                sameLayerIncompatibleEdges =
                    SameLayerIncompatibleEdgeCount,
                sameLayerRule =
                    "RejectOnlyIncompatibleZeroCrossingEdgeWithSamePairWitness",
                contradictionRetirementEnabled =
                    ContradictionRetirementEnabled || UsesAtomicRetirement,
                contradictionMinimumDistanceMeters =
                    ContradictionMinimumDistanceMeters,
                contradictionMinimumMatureWeight =
                    ContradictionMinimumMatureWeight,
                contradictionAnchorConsistencyMeters =
                    ContradictionAnchorConsistencyMeters,
                contradictionPairsToRetire =
                    ContradictionPairsToRetire,
                replacementPairsToPublish =
                    ReplacementPairsToPublish,
                atomicChunkFrontHold = UsesAtomicRetirement,
                productionQ75Enabled = UsesProductionQ75,
                q75ModelReady =
                    _fusionGuardQ75 != null &&
                    _fusionGuardQ75.ModelReady,
                q75MaximumRisk =
                    RoomSetupReplicaV2FusionGuardQ75.MaximumRisk,
                q75MaximumStereoAnchorSeparationMeters =
                    RoomSetupReplicaV2FusionGuardQ75
                        .MaximumStereoAnchorSeparationMeters,
                q75CandidateLayerCellMeters =
                    Q75CandidateLayerCellMeters,
                q75QualifiedVoxelObservations =
                    _fusionGuardQ75 != null
                        ? _fusionGuardQ75.QualifiedVoxelObservations
                        : 0,
                q75DeferredVoxelObservations =
                    _fusionGuardQ75 != null
                        ? _fusionGuardQ75.DeferredVoxelObservations
                        : 0,
                layerShiftChallengeMeters = LayerShiftChallengeMeters,
                layerShiftDistanceConsistencyMeters =
                    LayerShiftDistanceConsistencyMeters,
                layerShiftPairsToReplace = LayerShiftPairsToReplace,
                layerShiftDeferredObservations =
                    LayerShiftDeferredObservationCount,
                layerShiftPublishedVoxels =
                    LayerShiftPublishedVoxelCount,
                contradictionSingleRetirementPerVoxel = true,
                contradictionVotes = ContradictionVoteCount,
                retiredContradictedVoxels =
                    RetiredContradictedVoxelCount,
                contradictionRetirementRule =
                    "FiveConsistentDirectPairsOpposeMatureOldSign_" +
                    "StableAnchor_QuarantineThreeMorePairs_" +
                    "AtomicChunkFrontHold_OneLocalReset_NoDecay"
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

    private void TryCompleteLiveExport()
    {
        if (string.IsNullOrEmpty(_pendingLiveExportDirectory) ||
            string.IsNullOrEmpty(_pendingLiveExportFileName) ||
            HasPendingWork || _meshQueue.Count > 0 || _meshPassActive ||
            _simplificationTask != null)
            return;
        if (_globalTopologyCleanupDirty)
        {
            PublishRemoveSmallCandidateMeshes();
            _globalTopologyCleanupDirty = false;
            _nextGlobalTopologyCleanupTime = Time.unscaledTime +
                GlobalTopologyCleanupIntervalSeconds;
            return;
        }
        string directory = _pendingLiveExportDirectory;
        string fileName = _pendingLiveExportFileName;
        _pendingLiveExportDirectory = null;
        _pendingLiveExportFileName = null;
        try
        {
            FinalMeshMetrics metrics = ExportAndMeasureFinalMesh(
                directory, fileName);
            var receipt = new LiveMeshReceipt
            {
                schema = "roomsetup-replica-v2-live-mesh-v1",
                createdUtc = DateTime.UtcNow.ToString("O"),
                arm = RouteLabel,
                mutation = _abExperimentalArm
                    ? "5x5-valid-spread-under-12cm plus five-pair direct refutation; replacement quarantined for three more stable pairs and published with an atomic chunk Front"
                    : UsesProductionQ75
                        ? "ordinary first-surface fusion plus two-pair stereo layer-shift latch for mature same-sign TSDF jumps; signed opposition keeps the q75 dominant/challenger atomic route"
                        : "frozen old admission fusion extraction and lifecycle rules",
                coordinateSpace = "QuestStage",
                acceptedPairs = AcceptedPairCount,
                droppedBusyPairs = DroppedBusyPairCount,
                readbackFailures = ReadbackFailureCount,
                activeChunks = ActiveChunkCount,
                weightedVoxels = WeightedVoxelCount,
                contradictionVotes = ContradictionVoteCount,
                retiredVoxels = RetiredContradictedVoxelCount,
                pendingReplacementVoxels =
                    CountPendingReplacementVoxels(),
                replacementPairsToPublish = ReplacementPairsToPublish,
                atomicChunkFrontHold = UsesAtomicRetirement,
                productionQ75Enabled = UsesProductionQ75,
                q75ModelReady =
                    _fusionGuardQ75 != null &&
                    _fusionGuardQ75.ModelReady,
                q75MaximumRisk =
                    RoomSetupReplicaV2FusionGuardQ75.MaximumRisk,
                q75MaximumStereoAnchorSeparationMeters =
                    RoomSetupReplicaV2FusionGuardQ75
                        .MaximumStereoAnchorSeparationMeters,
                q75CandidateLayerCellMeters =
                    Q75CandidateLayerCellMeters,
                q75QualifiedVoxelObservations =
                    _fusionGuardQ75 != null
                        ? _fusionGuardQ75.QualifiedVoxelObservations
                        : 0,
                q75DeferredVoxelObservations =
                    _fusionGuardQ75 != null
                        ? _fusionGuardQ75.DeferredVoxelObservations
                        : 0,
                layerShiftDeferredObservations =
                    LayerShiftDeferredObservationCount,
                layerShiftPublishedVoxels =
                    LayerShiftPublishedVoxelCount,
                vertices = metrics.vertices,
                triangles = metrics.triangles,
                connectedComponents = metrics.connectedComponents,
                surfaceAreaSquareMeters = metrics.surfaceAreaSquareMeters,
                meshPath = metrics.path,
                meshSha256 = metrics.sha256,
                renderedDuringCapture = showReplica
            };
            string receiptPath = Path.Combine(
                directory,
                Path.GetFileNameWithoutExtension(fileName) + ".json");
            File.WriteAllText(
                receiptPath, JsonUtility.ToJson(receipt, true) + "\n");
            Debug.Log(
                $"[RoomSetupReplicaV2] export complete route={receipt.arm} " +
                $"pairs={receipt.acceptedPairs} tri={receipt.triangles} " +
                $"retired={receipt.retiredVoxels} path={metrics.path}");
        }
        catch (Exception exception)
        {
            Debug.LogError(
                $"[RoomSetupReplicaV2] export failed route={RouteLabel}: " +
                exception);
        }
    }

    private int CountPendingReplacementVoxels()
    {
        int total = 0;
        foreach (Chunk chunk in _chunks.Values)
            total += chunk.pendingReplacementVoxels;
        return total;
    }

    private FinalMeshMetrics ExportAndMeasureFinalMesh(
        string directory, string fileName = "last-replay-mesh.ply")
    {
        _orderedChunks.Clear();
        foreach (KeyValuePair<Vector3Int, Chunk> item in _chunks)
            if (item.Value.publishedVertices != null &&
                item.Value.publishedTriangles != null &&
                item.Value.publishedTriangles.Length > 0)
                _orderedChunks.Add(item);
        _orderedChunks.Sort(CompareChunks);

        _finalVertices.Clear();
        _finalTriangles.Clear();
        for (int chunkIndex = 0;
             chunkIndex < _orderedChunks.Count;
             chunkIndex++)
        {
            Chunk chunk = _orderedChunks[chunkIndex].Value;
            Vector3[] vertices = chunk.publishedVertices;
            int[] triangles = chunk.publishedTriangles;
            int vertexBase = _finalVertices.Count;
            _finalVertices.AddRange(vertices);
            for (int i = 0; i < triangles.Length; i++)
                _finalTriangles.Add(vertexBase + triangles[i]);
        }

        var metrics = new FinalMeshMetrics
        {
            vertices = _finalVertices.Count,
            triangles = _finalTriangles.Count / 3,
            path = Path.Combine(directory, fileName)
        };
        WriteBinaryPly(metrics.path, _finalVertices, _finalTriangles);
        metrics.sha256 = ComputeFileSha256(metrics.path);
        metrics.surfaceAreaSquareMeters = MeasureSurfaceArea(
            _finalVertices, _finalTriangles);
        MeasureConnectedComponents(metrics);
        MeasurePlanarResidual(metrics);
        return metrics;
    }

    private static float MeasureSurfaceArea(
        List<Vector3> vertices, List<int> triangles)
    {
        double area = 0d;
        for (int index = 0; index + 2 < triangles.Count; index += 3)
        {
            Vector3 a = vertices[triangles[index]];
            Vector3 b = vertices[triangles[index + 1]];
            Vector3 c = vertices[triangles[index + 2]];
            area += Vector3.Cross(b - a, c - a).magnitude * 0.5d;
        }
        return (float)area;
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
        var validObservations = new int[triangleCount];
        var foregroundContradictions = new int[triangleCount];
        var backgroundContradictions = new int[triangleCount];
        var minimumResiduals = new float[triangleCount];
        var maximumResiduals = new float[triangleCount];
        var nearestResiduals = new float[triangleCount];
        Array.Fill(minimumResiduals, float.PositiveInfinity);
        Array.Fill(maximumResiduals, float.NegativeInfinity);
        Array.Fill(nearestResiduals, float.PositiveInfinity);
        int remaining = triangleCount;
        using (var reader = new RoomSetupReplicaV2Tape.Reader(tapePath))
        {
            while (reader.TryReadNext(out var frame))
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
                    AccumulateDepthResidual(
                        centroid, frame, reader.Width, reader.Height,
                        ref validObservations[triangle],
                        ref foregroundContradictions[triangle],
                        ref backgroundContradictions[triangle],
                        ref minimumResiduals[triangle],
                        ref maximumResiduals[triangle],
                        ref nearestResiduals[triangle]);
                    if (supported[triangle])
                        continue;
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
        string directory = Path.GetDirectoryName(metrics.path);
        metrics.spatialReceiptPath = Path.Combine(
            string.IsNullOrEmpty(directory)
                ? Application.persistentDataPath
                : directory,
            "last-replay-spatial-receipt.json");
        WriteSpatialScarReceipt(
            metrics.spatialReceiptPath, metrics.sha256, supported,
            validObservations, foregroundContradictions,
            backgroundContradictions, minimumResiduals,
            maximumResiduals, nearestResiduals, metrics);
    }

    private static void AccumulateDepthResidual(
        Vector3 stagePoint,
        RoomSetupReplicaV2Tape.Frame frame,
        int width,
        int height,
        ref int valid,
        ref int foreground,
        ref int background,
        ref float minimum,
        ref float maximum,
        ref float nearest)
    {
        int frameForeground = 0;
        int frameBackground = 0;
        AccumulateDepthResidual(
            stagePoint, frame.leftWorldToClip,
            frame.leftProjectionInverse, frame.leftViewInverse,
            frame.leftNdc, width, height,
            ref valid, ref frameForeground, ref frameBackground,
            ref minimum, ref maximum, ref nearest);
        AccumulateDepthResidual(
            stagePoint, frame.rightWorldToClip,
            frame.rightProjectionInverse, frame.rightViewInverse,
            frame.rightNdc, width, height,
            ref valid, ref frameForeground, ref frameBackground,
            ref minimum, ref maximum, ref nearest);
        if (frameForeground > 0)
            foreground++;
        if (frameBackground > 0)
            background++;
    }

    private static void AccumulateDepthResidual(
        Vector3 stagePoint,
        Matrix4x4 stageToClip,
        Matrix4x4 projectionInverse,
        Matrix4x4 viewInverse,
        float[] ndcDepths,
        int width,
        int height,
        ref int valid,
        ref int foreground,
        ref int background,
        ref float minimum,
        ref float maximum,
        ref float nearest)
    {
        if (!TryDepthResidual(
                stagePoint, stageToClip, projectionInverse, viewInverse,
                ndcDepths, width, height, out float residual, out _))
            return;
        valid++;
        minimum = Mathf.Min(minimum, residual);
        maximum = Mathf.Max(maximum, residual);
        nearest = Mathf.Min(nearest, Mathf.Abs(residual));
        if (residual <= -ScarContradictionMeters)
            foreground++;
        else if (residual >= ScarContradictionMeters)
            background++;
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
        return TryDepthResidual(
                   stagePoint, stageToClip, projectionInverse, viewInverse,
                   ndcDepths, width, height, out _, out float distance) &&
               distance <= DirectSupportMeters;
    }

    private static bool TryDepthResidual(
        Vector3 stagePoint,
        Matrix4x4 stageToClip,
        Matrix4x4 projectionInverse,
        Matrix4x4 viewInverse,
        float[] ndcDepths,
        int width,
        int height,
        out float residual,
        out float pointDistance)
    {
        residual = 0f;
        pointDistance = float.PositiveInfinity;
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
        if (!Finite(observed))
            return false;
        Vector3 eye = viewInverse.MultiplyPoint3x4(Vector3.zero);
        if (!Finite(eye))
            return false;
        pointDistance = Vector3.Distance(stagePoint, observed);
        residual = Vector3.Distance(eye, observed) -
                   Vector3.Distance(eye, stagePoint);
        return float.IsFinite(residual) && float.IsFinite(pointDistance);
    }

    private void WriteSpatialScarReceipt(
        string path,
        string meshSha256,
        bool[] supported,
        int[] validObservations,
        int[] foregroundContradictions,
        int[] backgroundContradictions,
        float[] minimumResiduals,
        float[] maximumResiduals,
        float[] nearestResiduals,
        FinalMeshMetrics metrics)
    {
        int triangleCount = _finalTriangles.Count / 3;
        var union = new UnionFind(triangleCount);
        var vertexOwner = new Dictionary<QuantizedPosition, int>();
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            if (supported[triangle])
                continue;
            int index = triangle * 3;
            for (int corner = 0; corner < 3; corner++)
            {
                Vector3 vertex = _finalVertices[
                    _finalTriangles[index + corner]];
                var key = new QuantizedPosition(vertex);
                if (vertexOwner.TryGetValue(key, out int owner))
                    union.Union(triangle, owner);
                else
                    vertexOwner.Add(key, triangle);
            }
        }

        var compactClusters = new Dictionary<int, int>();
        var triangleReceipts = new List<SpatialTriangleReceipt>();
        var clusterTriangles = new List<int>();
        var clusterMins = new List<Vector3>();
        var clusterMaxs = new List<Vector3>();
        var clusterCross = new List<int>();
        var clusterShell = new List<int>();
        var clusterUnknown = new List<int>();
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            if (supported[triangle])
                continue;
            int root = union.Find(triangle);
            if (!compactClusters.TryGetValue(root, out int cluster))
            {
                cluster = compactClusters.Count;
                compactClusters.Add(root, cluster);
                clusterTriangles.Add(0);
                clusterMins.Add(new Vector3(
                    float.PositiveInfinity, float.PositiveInfinity,
                    float.PositiveInfinity));
                clusterMaxs.Add(new Vector3(
                    float.NegativeInfinity, float.NegativeInfinity,
                    float.NegativeInfinity));
                clusterCross.Add(0);
                clusterShell.Add(0);
                clusterUnknown.Add(0);
            }

            int index = triangle * 3;
            Vector3 a = _finalVertices[_finalTriangles[index]];
            Vector3 b = _finalVertices[_finalTriangles[index + 1]];
            Vector3 c = _finalVertices[_finalTriangles[index + 2]];
            Vector3 cross = Vector3.Cross(b - a, c - a);
            Vector3 normal = cross.sqrMagnitude > 0.00000001f
                ? cross.normalized
                : Vector3.zero;
            string classification;
            if (foregroundContradictions[triangle] > 0 &&
                backgroundContradictions[triangle] > 0)
            {
                classification = "CrossLayerConnectionCandidate";
                clusterCross[cluster]++;
                metrics.crossLayerCandidateTriangles++;
            }
            else if (validObservations[triangle] >=
                         SpatialHistoricalPairsThreshold &&
                     (foregroundContradictions[triangle] >=
                          SpatialHistoricalPairsThreshold ||
                      backgroundContradictions[triangle] >=
                          SpatialHistoricalPairsThreshold))
            {
                classification = "HistoricalShellCandidate";
                clusterShell[cluster]++;
                metrics.historicalShellCandidateTriangles++;
            }
            else
            {
                classification = "InsufficientEvidence";
                clusterUnknown[cluster]++;
                metrics.insufficientEvidenceTriangles++;
            }

            Vector3 centroid = (a + b + c) / 3f;
            triangleReceipts.Add(new SpatialTriangleReceipt
            {
                triangle = triangle,
                cluster = cluster,
                centroid = centroid,
                normal = normal,
                areaSquareMeters = cross.magnitude * 0.5f,
                validDepthObservations = validObservations[triangle],
                foregroundContradictions =
                    foregroundContradictions[triangle],
                backgroundContradictions =
                    backgroundContradictions[triangle],
                nearestAbsoluteResidualMeters =
                    float.IsFinite(nearestResiduals[triangle])
                        ? nearestResiduals[triangle]
                        : -1f,
                minimumResidualMeters =
                    float.IsFinite(minimumResiduals[triangle])
                        ? minimumResiduals[triangle]
                        : 0f,
                maximumResidualMeters =
                    float.IsFinite(maximumResiduals[triangle])
                        ? maximumResiduals[triangle]
                        : 0f,
                classification = classification
            });
            clusterTriangles[cluster]++;
            clusterMins[cluster] = Vector3.Min(
                clusterMins[cluster], Vector3.Min(a, Vector3.Min(b, c)));
            clusterMaxs[cluster] = Vector3.Max(
                clusterMaxs[cluster], Vector3.Max(a, Vector3.Max(b, c)));
        }

        var clusters = new SpatialClusterReceipt[compactClusters.Count];
        for (int cluster = 0; cluster < clusters.Length; cluster++)
        {
            clusters[cluster] = new SpatialClusterReceipt
            {
                cluster = cluster,
                triangles = clusterTriangles[cluster],
                boundsMin = clusterMins[cluster],
                boundsMax = clusterMaxs[cluster],
                crossLayerCandidates = clusterCross[cluster],
                historicalShellCandidates = clusterShell[cluster],
                insufficientEvidence = clusterUnknown[cluster]
            };
        }
        metrics.scarClusters = clusters.Length;
        var receipt = new SpatialScarReceipt
        {
            coordinateSpace = "QuestStage",
            sourceMeshSha256 = meshSha256,
            directSupportMeters = DirectSupportMeters,
            contradictionMeters = ScarContradictionMeters,
            classificationRule =
                "BothResidualSigns=CrossLayer;RepeatedOneSign=HistoricalShell",
            totalTriangles = triangleCount,
            unsupportedTriangles = triangleReceipts.Count,
            clusters = clusters.Length,
            clusterReceipts = clusters,
            triangleReceipts = triangleReceipts.ToArray()
        };
        File.WriteAllText(path, JsonUtility.ToJson(receipt, true));
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
        if (Time.unscaledTime <= _nextMeshTime)
            return false;
        int band = Mathf.Max(0, PreviewTriangleCount) /
                   MeshIntervalTriangleBucket;
        _lastScheduledMeshIntervalSeconds = Mathf.Min(
            MeshIntervalMaximumSeconds,
            MeshIntervalBaseSeconds + MeshIntervalBucketSeconds * band);
        // The recovered gate advances its timestamp as soon as a due trigger
        // fires, before the separate submit path discovers a busy future.
        _nextMeshTime = Time.unscaledTime +
                        _lastScheduledMeshIntervalSeconds;
        // Native scheduling is single-flight across the complete asynchronous
        // meshing job. A due trigger while the prior future is still running
        // is skipped; it does not accumulate another immutable snapshot.
        if (_simplificationTask != null)
        {
            _skippedAsyncBusyCount++;
            _asyncMeshingPending = true;
            return false;
        }
        _asyncMeshingPending = false;
        _meshPassActive = true;
        _passDerivedValidSamples = 0;
        _passDerivedMissingSamples = 0;
        _meshPassPublicationChunks.Clear();
        return true;
    }

    private void CompleteMeshPass()
    {
        bool globalCleanupDue = !RenderPageBatchingEnabled ||
            _renderPages.Count == 0 ||
            Time.unscaledTime >= _nextGlobalTopologyCleanupTime;
        if (globalCleanupDue)
        {
            PublishRemoveSmallCandidateMeshes();
            _nextGlobalTopologyCleanupTime = Time.unscaledTime +
                GlobalTopologyCleanupIntervalSeconds;
            _globalTopologyCleanupDirty = false;
        }
        else
        {
            PublishDirtyCandidateMeshes();
            _globalTopologyCleanupDirty = true;
        }
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
    }

    private void PublishDirtyCandidateMeshes()
    {
        _localPublicationChunks.Clear();
        foreach (Vector3Int chunkId in _meshPassPublicationChunks)
        {
            _localPublicationChunks.Add(chunkId);
            if (_chunks.TryGetValue(chunkId, out Chunk chunk))
                PublishCandidateChunkMesh(chunkId, chunk);
        }

        RebuildRenderPages(
            _localPublicationChunks,
            _renderPages.Count == 0);
        PreviewTriangleCount = 0;
        foreach (Chunk chunk in _chunks.Values)
            PreviewTriangleCount += chunk.triangleCount;
        for (int i = 0; i < _localPublicationChunks.Count; i++)
            RecordCorrectionPublication(_localPublicationChunks[i]);
        _meshPublicationCount++;
        LastSimplificationApplied = false;
        LastSimplificationStopReason =
            "LocalPagePublishPendingPeriodicGlobalCleanup";
    }

    private void PublishCandidateChunkMesh(Vector3Int chunkId, Chunk chunk)
    {
        Vector3[] vertices = chunk.candidateVertices ??
            Array.Empty<Vector3>();
        int[] triangles = chunk.candidateTriangles ?? Array.Empty<int>();
        int triangleCount = triangles.Length / 3;
        chunk.publishedVertices = vertices;
        chunk.publishedTriangles = triangles;
        chunk.triangleCount = triangleCount;

        if (RenderPageBatchingEnabled)
        {
            if (chunk.previewObject != null)
                chunk.previewObject.SetActive(false);
            return;
        }

        if (triangleCount <= 0)
        {
            if (chunk.previewObject != null)
                chunk.previewObject.SetActive(false);
            return;
        }
        EnsureChunkPreview(chunkId, chunk);
        Mesh staging = chunk.backMesh;
        staging.Clear(false);
        staging.vertices = vertices;
        staging.subMeshCount = 1;
        staging.SetIndices(triangles, MeshTopology.Triangles, 0, false);
        staging.RecalculateBounds();
        staging.UploadMeshData(false);
        chunk.previewFilter.sharedMesh = staging;
        chunk.backMesh = chunk.frontMesh;
        chunk.frontMesh = staging;
        chunk.previewObject.SetActive(showReplica);
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
        if (UsesAtomicRetirement &&
            HasPendingReplacementDependency(chunkId))
        {
            // Keep the last complete candidate/Front. Publishing a chunk while
            // only some retired voxels have replacement evidence recreates the
            // exact asynchronous zero-crossing ring this route is meant to
            // eliminate. A derived Chunk samples its own source Chunk plus
            // positive-boundary samples from seven neighbours, so all eight
            // dependencies participate in the hold.
            return;
        }
        CacheSamples(chunkId);
        _meshVertices.Clear();
        _meshTriangles.Clear();
        _meshLines.Clear();
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
            if (SameLayerSupportEnabled &&
                !CellHasCompatibleLayerSupport(sx, sy, sz))
            {
                SameLayerRejectedCellCount++;
                continue;
            }

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

    private bool HasPendingReplacementDependency(Vector3Int meshChunkId)
    {
        for (int z = 0; z <= 1; z++)
        for (int y = 0; y <= 1; y++)
        for (int x = 0; x <= 1; x++)
        {
            Vector3Int sourceChunkId = meshChunkId +
                new Vector3Int(x, y, z);
            if (_chunks.TryGetValue(sourceChunkId, out Chunk source) &&
                source.pendingReplacementVoxels > 0)
                return true;
        }
        return false;
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
        // The normal path cannot enter here while a prior job is active,
        // because BeginMeshPass enforces single-flight. Keep the same skip
        // semantics as a defensive invariant instead of queuing a snapshot.
        if (_simplificationTask != null)
        {
            if (!_asyncMeshingPending)
                _skippedAsyncBusyCount++;
            _asyncMeshingPending = true;
            for (int i = 0; i < input.publicationChunkIds.Length; i++)
            {
                Vector3Int chunkId = input.publicationChunkIds[i];
                if (_queuedMeshChunks.Add(chunkId))
                    _meshQueue.Enqueue(chunkId);
            }
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
    }

    private void PublishSimplificationResult(
        RoomSetupReplicaV2XrtSimplifier.Result result)
    {
        _localPublicationChunks.Clear();
        _meshPublicationCount++;
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
            {
                if (PublishSimplifiedChunkMesh(
                        chunkId, chunk, owner, result))
                    AddLocalPublicationChunk(chunkId);
            }
        }
        for (int i = 0; i < result.publicationChunkIds.Length; i++)
            AddLocalPublicationChunk(result.publicationChunkIds[i]);
        RebuildRenderPages(
            _localPublicationChunks,
            _renderPages.Count == 0);
        // The simplifier input is already welded globally at 0.1 mm. Build
        // one indexed line Front from that topology so every edge is submitted
        // once and the room uses one line Renderer instead of one per chunk.
        if (!RenderPageBatchingEnabled)
            PublishGlobalLineMesh(result.vertices, result.triangles);
        else
            ClearGlobalLinePreview();
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
            $"[RoomSetupReplicaV2] mesh post-process " +
            $"mode={(RoomSetupReplicaV2XrtSimplifier.Enabled ? "XRT" : "Regular10cm")} " +
            $"applied={result.applied} " +
            $"vertices={result.inputVertices}->{result.outputVertices} " +
            $"target={result.targetVertices} triangles=" +
            $"{result.inputTriangles}->{result.outputTriangles} " +
            $"collapses={result.acceptedCollapses} " +
            $"failedBatches={result.failedBatches} " +
            $"stop={result.stopReason} ms={result.elapsedMilliseconds:F1}");
    }

    private bool PublishSimplifiedChunkMesh(
        Vector3Int chunkId,
        Chunk chunk,
        int owner,
        RoomSetupReplicaV2XrtSimplifier.Result result)
    {
        _meshVertices.Clear();
        _meshTriangles.Clear();
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
        }

        int publishedTriangleCount = _meshTriangles.Count / 3;
        if (publishedTriangleCount <= 0)
        {
            bool changed = chunk.publishedTriangles != null &&
                chunk.publishedTriangles.Length > 0;
            chunk.publishedVertices = Array.Empty<Vector3>();
            chunk.publishedTriangles = Array.Empty<int>();
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
            return changed;
        }

        bool publicationChanged = !PublishedMeshEquals(
            chunk, _meshVertices, _meshTriangles);
        chunk.publishedVertices = _meshVertices.ToArray();
        chunk.publishedTriangles = _meshTriangles.ToArray();
        chunk.triangleCount = publishedTriangleCount;
        if (RenderPageBatchingEnabled)
        {
            if (chunk.previewObject != null)
                chunk.previewObject.SetActive(false);
            return publicationChanged;
        }

        EnsureChunkPreview(chunkId, chunk);
        Mesh staging = chunk.backMesh;
        staging.Clear(false);
        staging.SetVertices(_meshVertices);
        staging.subMeshCount = 1;
        staging.SetIndices(_meshTriangles, MeshTopology.Triangles, 0, false);
        // A legal collapse may retain the endpoint owned by a neighbouring
        // extraction chunk. Recalculate bounds from the actual surviving
        // vertices rather than assuming every point remains inside its old
        // chunk AABB.
        staging.RecalculateBounds();
        staging.UploadMeshData(false);
        if (chunk.previewFilter != null)
            chunk.previewFilter.sharedMesh = staging;
        chunk.backMesh = chunk.frontMesh;
        chunk.frontMesh = staging;
        if (chunk.previewObject != null)
            chunk.previewObject.SetActive(
                showReplica && !RenderPageBatchingEnabled);
        return publicationChanged;
    }

    private void AddLocalPublicationChunk(Vector3Int chunkId)
    {
        if (!_localPublicationChunks.Contains(chunkId))
            _localPublicationChunks.Add(chunkId);
    }

    private static bool PublishedMeshEquals(
        Chunk chunk,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<int> triangles)
    {
        if (chunk.publishedVertices == null ||
            chunk.publishedTriangles == null ||
            chunk.publishedVertices.Length != vertices.Count ||
            chunk.publishedTriangles.Length != triangles.Count)
            return false;
        for (int i = 0; i < vertices.Count; i++)
            if (!chunk.publishedVertices[i].Equals(vertices[i]))
                return false;
        for (int i = 0; i < triangles.Count; i++)
            if (chunk.publishedTriangles[i] != triangles[i])
                return false;
        return true;
    }

    private static Vector3Int RenderPageKey(Vector3Int chunkId)
    {
        return new Vector3Int(
            Mathf.FloorToInt(chunkId.x / (float)RenderPageSpanChunks),
            Mathf.FloorToInt(chunkId.y / (float)RenderPageSpanChunks),
            Mathf.FloorToInt(chunkId.z / (float)RenderPageSpanChunks));
    }

    private void RebuildRenderPages(
        IReadOnlyList<Vector3Int> dirtyChunkIds,
        bool forceAll)
    {
        if (!RenderPageBatchingEnabled)
            return;
        EnsurePreviewResources();
        if (_previewRoot == null || _depthMaterial == null)
            return;

        var dirtyPages = new HashSet<Vector3Int>();
        if (forceAll)
        {
            foreach (KeyValuePair<Vector3Int, Chunk> item in _chunks)
            {
                if (item.Value.triangleCount > 0)
                    dirtyPages.Add(RenderPageKey(item.Key));
            }
            foreach (Vector3Int existing in _renderPages.Keys)
                dirtyPages.Add(existing);
        }
        else if (dirtyChunkIds != null)
        {
            for (int i = 0; i < dirtyChunkIds.Count; i++)
                dirtyPages.Add(RenderPageKey(dirtyChunkIds[i]));
        }

        foreach (Vector3Int pageKey in dirtyPages)
        {
            RenderPage page = EnsureRenderPage(pageKey);
            page.members.Clear();
            page.vertices.Clear();
            page.triangles.Clear();
            Vector3Int first = pageKey * RenderPageSpanChunks;
            for (int z = 0; z < RenderPageSpanChunks; z++)
            for (int y = 0; y < RenderPageSpanChunks; y++)
            for (int x = 0; x < RenderPageSpanChunks; x++)
            {
                Vector3Int chunkId = first + new Vector3Int(x, y, z);
                if (!_chunks.TryGetValue(chunkId, out Chunk chunk) ||
                    chunk.publishedVertices == null ||
                    chunk.publishedTriangles == null ||
                    chunk.publishedTriangles.Length == 0)
                    continue;
                int vertexBase = page.vertices.Count;
                page.vertices.AddRange(chunk.publishedVertices);
                for (int i = 0; i < chunk.publishedTriangles.Length; i++)
                    page.triangles.Add(
                        vertexBase + chunk.publishedTriangles[i]);
                page.members.Add(chunkId);
            }

            if (page.triangles.Count == 0)
            {
                DisposeRenderPage(page);
                _renderPages.Remove(pageKey);
                continue;
            }

            Mesh staging = page.backMesh;
            staging.Clear(false);
            staging.indexFormat = page.vertices.Count > 65535
                ? IndexFormat.UInt32
                : IndexFormat.UInt16;
            staging.SetVertices(page.vertices);
            staging.SetIndices(
                page.triangles, MeshTopology.Triangles, 0, false);
            staging.RecalculateBounds();
            staging.UploadMeshData(false);
            page.filter.sharedMesh = staging;
            page.backMesh = page.frontMesh;
            page.frontMesh = staging;
            page.triangleCount = page.triangles.Count / 3;
            RebuildRenderPageLines(page);
            page.gameObject.SetActive(showReplica);
        }
    }

    private RenderPage EnsureRenderPage(Vector3Int pageKey)
    {
        if (_renderPages.TryGetValue(pageKey, out RenderPage page))
            return page;
        var go = new GameObject(
            $"RS Replica render page {pageKey.x},{pageKey.y},{pageKey.z}");
        go.transform.SetParent(_previewRoot.transform, false);
        page = new RenderPage
        {
            key = pageKey,
            gameObject = go,
            filter = go.AddComponent<MeshFilter>(),
            renderer = go.AddComponent<MeshRenderer>(),
            frontMesh = CreateRenderPageMesh(pageKey, "front"),
            backMesh = CreateRenderPageMesh(pageKey, "back")
        };
        page.renderer.shadowCastingMode = ShadowCastingMode.Off;
        page.renderer.receiveShadows = false;
        page.renderer.sharedMaterial = _depthMaterial;
        page.filter.sharedMesh = page.frontMesh;
        page.lineObject = new GameObject("Lines");
        page.lineObject.transform.SetParent(go.transform, false);
        page.lineFilter = page.lineObject.AddComponent<MeshFilter>();
        page.lineRenderer = page.lineObject.AddComponent<MeshRenderer>();
        page.lineRenderer.shadowCastingMode = ShadowCastingMode.Off;
        page.lineRenderer.receiveShadows = false;
        page.lineRenderer.sharedMaterial = _lineMaterial;
        page.lineFrontMesh = CreateRenderPageMesh(pageKey, "lines-front");
        page.lineBackMesh = CreateRenderPageMesh(pageKey, "lines-back");
        page.lineFilter.sharedMesh = page.lineFrontMesh;
        go.SetActive(false);
        _renderPages.Add(pageKey, page);
        return page;
    }

    private static Mesh CreateRenderPageMesh(
        Vector3Int pageKey, string suffix)
    {
        var mesh = new Mesh
        {
            name = $"RS Replica render page {pageKey.x},{pageKey.y},{pageKey.z} {suffix}",
            indexFormat = IndexFormat.UInt32
        };
        mesh.MarkDynamic();
        return mesh;
    }

    private void RebuildRenderPageLines(RenderPage page)
    {
        _lineQuadVertices.Clear();
        _lineQuadOtherEndpoints.Clear();
        _lineQuadExtrusions.Clear();
        _meshLines.Clear();
        _uniqueLines.Clear();
        _globalLinePointIds.Clear();
        _nextGlobalLinePointId = 0;
        for (int i = 0; i + 2 < page.triangles.Count; i += 3)
        {
            AddUniqueGlobalLine(
                page.vertices, page.triangles[i], page.triangles[i + 1]);
            AddUniqueGlobalLine(
                page.vertices, page.triangles[i + 1], page.triangles[i + 2]);
            AddUniqueGlobalLine(
                page.vertices, page.triangles[i + 2], page.triangles[i]);
        }

        Mesh staging = page.lineBackMesh;
        staging.Clear(false);
        staging.indexFormat = _lineQuadVertices.Count > 65535
            ? IndexFormat.UInt32
            : IndexFormat.UInt16;
        staging.SetVertices(_lineQuadVertices);
        staging.SetUVs(0, _lineQuadExtrusions);
        staging.SetUVs(1, _lineQuadOtherEndpoints);
        staging.SetIndices(_meshLines, MeshTopology.Triangles, 0, false);
        if (_meshLines.Count > 0)
            staging.RecalculateBounds();
        staging.UploadMeshData(false);
        page.lineFilter.sharedMesh = staging;
        page.lineBackMesh = page.lineFrontMesh;
        page.lineFrontMesh = staging;
        page.lineObject.SetActive(_meshLines.Count > 0);
    }

    private void ClearRenderPages()
    {
        foreach (RenderPage page in _renderPages.Values)
            DisposeRenderPage(page);
        _renderPages.Clear();
    }

    private static void DisposeRenderPage(RenderPage page)
    {
        if (page == null)
            return;
        if (page.gameObject != null)
            Destroy(page.gameObject);
        if (page.frontMesh != null)
            Destroy(page.frontMesh);
        if (page.backMesh != null)
            Destroy(page.backMesh);
        if (page.lineFrontMesh != null)
            Destroy(page.lineFrontMesh);
        if (page.lineBackMesh != null)
            Destroy(page.lineBackMesh);
    }

    private void PublishGlobalLineMesh(Vector3[] vertices, int[] triangles)
    {
        EnsureGlobalLinePreview();
        if (_globalLineObject == null || _globalLineBackMesh == null)
            return;

        _lineQuadVertices.Clear();
        _lineQuadOtherEndpoints.Clear();
        _lineQuadExtrusions.Clear();
        _meshLines.Clear();
        _uniqueLines.Clear();
        _globalLinePointIds.Clear();
        _nextGlobalLinePointId = 0;
        for (int i = 0; i + 2 < triangles.Length; i += 3)
        {
            AddUniqueGlobalLine(vertices, triangles[i], triangles[i + 1]);
            AddUniqueGlobalLine(vertices, triangles[i + 1], triangles[i + 2]);
            AddUniqueGlobalLine(vertices, triangles[i + 2], triangles[i]);
        }

        Mesh staging = _globalLineBackMesh;
        staging.Clear(false);
        staging.SetVertices(_lineQuadVertices);
        staging.SetUVs(0, _lineQuadExtrusions);
        staging.SetUVs(1, _lineQuadOtherEndpoints);
        staging.subMeshCount = 1;
        staging.SetIndices(_meshLines, MeshTopology.Triangles, 0, false);
        if (_meshLines.Count > 0)
            staging.RecalculateBounds();
        staging.UploadMeshData(false);
        _globalLineFilter.sharedMesh = staging;
        _globalLineBackMesh = _globalLineFrontMesh;
        _globalLineFrontMesh = staging;
        _globalLineIndexCount = _meshLines.Count;
        _globalLineObject.SetActive(
            showReplica && _globalLineIndexCount > 0);
    }

    private void AddUniqueGlobalLine(
        IReadOnlyList<Vector3> vertices, int first, int second)
    {
        if (first == second || first < 0 || second < 0 ||
            first >= vertices.Count || second >= vertices.Count)
            return;
        Vector3 a = vertices[first];
        Vector3 b = vertices[second];
        if (!Finite(a) || !Finite(b) ||
            (a - b).sqrMagnitude <= 0.00000001f)
            return;

        uint firstPoint = (uint)GetGlobalLinePointId(a);
        uint secondPoint = (uint)GetGlobalLinePointId(b);
        uint low = Math.Min(firstPoint, secondPoint);
        uint high = Math.Max(firstPoint, secondPoint);
        ulong key = ((ulong)low << 32) | high;
        if (!_uniqueLines.Add(key))
            return;

        int start = _lineQuadVertices.Count;
        _lineQuadVertices.Add(a);
        _lineQuadVertices.Add(a);
        _lineQuadVertices.Add(b);
        _lineQuadVertices.Add(b);
        _lineQuadOtherEndpoints.Add(b);
        _lineQuadOtherEndpoints.Add(b);
        _lineQuadOtherEndpoints.Add(a);
        _lineQuadOtherEndpoints.Add(a);
        _lineQuadExtrusions.Add(new Vector2(-1f, 0f));
        _lineQuadExtrusions.Add(new Vector2(1f, 0f));
        _lineQuadExtrusions.Add(new Vector2(-1f, 1f));
        _lineQuadExtrusions.Add(new Vector2(1f, 1f));
        _meshLines.Add(start);
        _meshLines.Add(start + 2);
        _meshLines.Add(start + 1);
        _meshLines.Add(start + 1);
        _meshLines.Add(start + 2);
        _meshLines.Add(start + 3);
    }

    private int GetGlobalLinePointId(Vector3 point)
    {
        // Marching-Cubes vertices generated by neighbouring chunks may be
        // separate indices even when they occupy the same geometric point.
        // A 0.1 mm key removes only duplicate display edges; it never welds
        // or otherwise changes the published reconstruction mesh.
        const float Quantization = 10000f;
        var key = new Vector3Int(
            Mathf.RoundToInt(point.x * Quantization),
            Mathf.RoundToInt(point.y * Quantization),
            Mathf.RoundToInt(point.z * Quantization));
        if (_globalLinePointIds.TryGetValue(key, out int id))
            return id;
        id = _nextGlobalLinePointId++;
        _globalLinePointIds.Add(key, id);
        return id;
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
        Array.Clear(_sampleLayerAnchors, 0, _sampleLayerAnchors.Length);
        Array.Clear(_sampleLayerObserved, 0, _sampleLayerObserved.Length);
        Array.Clear(_sampleLayerPairs, 0, _sampleLayerPairs.Length);
        Vector3Int origin = chunkId * ChunkSize;
        for (int z = 0; z < SampleSize; z += MeshStride)
        for (int y = 0; y < SampleSize; y += MeshStride)
        for (int x = 0; x < SampleSize; x += MeshStride)
        {
            int sample = SampleIndex(x, y, z);
            Vector3Int lattice = origin + new Vector3Int(x, y, z);
            if (!TryResampleDerivedField(
                    lattice, out float distance, out Voxel voxel))
            {
                _passDerivedMissingSamples++;
                continue;
            }
            _sampleObserved[sample] = 1;
            _sampleDistances[sample] = distance;
            if (voxel.lastEvidencePair > 0 && Finite(voxel.layerAnchor))
            {
                _sampleLayerObserved[sample] = 1;
                _sampleLayerAnchors[sample] = voxel.layerAnchor;
                _sampleLayerPairs[sample] = voxel.lastEvidencePair;
            }
            _passDerivedValidSamples++;
        }
    }

    private bool TryResampleDerivedField(
        Vector3Int lattice, out float distance, out Voxel voxel)
    {
        // Room Setup's active Resampled mode passes ratio 0.5 to 0x78cd20.
        // Every integer derived coordinate therefore maps to an exact even
        // source coordinate (derived / 0.5), selecting the function's exact
        // half-SDF/half-weight copy branch. Its trilinear branch is retained
        // for non-aligned ratios, but is not the active 5 cm -> 10 cm kernel.
        // `lattice` is already expressed in source-grid coordinates at stride
        // two, so this lookup is the equivalent aligned source sample.
        if (!TryGetVoxel(lattice, out voxel) ||
            voxel.weight <= 0f ||
            !float.IsFinite(voxel.distance) ||
            !float.IsFinite(voxel.weight))
        {
            distance = 0f;
            voxel = default;
            return false;
        }
        distance = voxel.distance;
        return true;
    }

    private bool CellHasCompatibleLayerSupport(int sx, int sy, int sz)
    {
        // Estimate the zero-crossing normal from the eight SDF corners. Only
        // edges that actually cross zero can create this cell's triangles.
        // Reject only when both edge endpoints carry direct witnesses from the
        // same stereo pair and those witnesses occupy incompatible layers.
        // Different-pair or missing evidence is not permission to erase a
        // surface: it remains uncomparable until a shared observation exists.
        float gx = (_cornerDistances[1] + _cornerDistances[2] +
                    _cornerDistances[5] + _cornerDistances[6] -
                    _cornerDistances[0] - _cornerDistances[3] -
                    _cornerDistances[4] - _cornerDistances[7]) * 0.25f;
        float gy = (_cornerDistances[2] + _cornerDistances[3] +
                    _cornerDistances[6] + _cornerDistances[7] -
                    _cornerDistances[0] - _cornerDistances[1] -
                    _cornerDistances[4] - _cornerDistances[5]) * 0.25f;
        float gz = (_cornerDistances[4] + _cornerDistances[5] +
                    _cornerDistances[6] + _cornerDistances[7] -
                    _cornerDistances[0] - _cornerDistances[1] -
                    _cornerDistances[2] - _cornerDistances[3]) * 0.25f;
        Vector3 normal = new Vector3(gx, gy, gz);
        if (!Finite(normal) || normal.sqrMagnitude <= 0.000001f)
            return true;
        normal.Normalize();

        for (int edge = 0; edge < 12; edge++)
        {
            int a = EdgePairs[edge, 0];
            int b = EdgePairs[edge, 1];
            if ((_cornerDistances[a] < 0f) ==
                (_cornerDistances[b] < 0f))
                continue;
            Vector3Int offsetA = CubeOffsets[a] * MeshStride;
            Vector3Int offsetB = CubeOffsets[b] * MeshStride;
            int sampleA = SampleIndex(
                sx + offsetA.x, sy + offsetA.y, sz + offsetA.z);
            int sampleB = SampleIndex(
                sx + offsetB.x, sy + offsetB.y, sz + offsetB.z);
            int pair = _sampleLayerPairs[sampleA];
            if (_sampleLayerObserved[sampleA] == 0 ||
                _sampleLayerObserved[sampleB] == 0 ||
                pair <= 0 || pair != _sampleLayerPairs[sampleB])
            {
                SameLayerUncomparableEdgeCount++;
                continue;
            }
            SameLayerComparableEdgeCount++;
            float separation = Mathf.Abs(Vector3.Dot(
                _sampleLayerAnchors[sampleA] -
                _sampleLayerAnchors[sampleB], normal));
            if (separation <= SameLayerMaximumSeparationMeters)
                continue;
            SameLayerIncompatibleEdgeCount++;
            return false;
        }
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
                _lineMaterial.SetFloat(
                    "_LineWidthPixels", lineWidthPixels);
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
        EnsureGlobalLinePreview();
    }

    private void EnsureGlobalLinePreview()
    {
        if (_globalLineObject != null || _previewRoot == null ||
            _lineMaterial == null)
            return;
        _globalLineObject = new GameObject(
            "[Room Setup Replica V2] Global Lines");
        _globalLineObject.transform.SetParent(_previewRoot.transform, false);
        _globalLineFilter = _globalLineObject.AddComponent<MeshFilter>();
        _globalLineRenderer =
            _globalLineObject.AddComponent<MeshRenderer>();
        _globalLineRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _globalLineRenderer.receiveShadows = false;
        _globalLineRenderer.sharedMaterial = _lineMaterial;
        _globalLineFrontMesh = CreateGlobalLineMesh("front");
        _globalLineBackMesh = CreateGlobalLineMesh("back");
        _globalLineFilter.sharedMesh = _globalLineFrontMesh;
        _globalLineObject.SetActive(false);
    }

    private static Mesh CreateGlobalLineMesh(string suffix)
    {
        var mesh = new Mesh
        {
            name = $"RS Replica global lines {suffix}",
            indexFormat = IndexFormat.UInt32
        };
        mesh.MarkDynamic();
        return mesh;
    }

    private void ClearGlobalLinePreview()
    {
        if (_globalLineFrontMesh != null)
            _globalLineFrontMesh.Clear(false);
        if (_globalLineBackMesh != null)
            _globalLineBackMesh.Clear(false);
        if (_globalLineObject != null)
            _globalLineObject.SetActive(false);
        _globalLineIndexCount = 0;
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
        chunk.previewRenderer.sharedMaterial = _depthMaterial;
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
        chunk.publishedVertices = Array.Empty<Vector3>();
        chunk.publishedTriangles = Array.Empty<int>();
        chunk.candidateTriangleCount = 0;
        chunk.triangleCount = 0;
        chunk.pendingReplacementVoxels = 0;
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

    public bool TryRaycastFinalMesh(
        Ray worldRay,
        float maximumDistanceMeters,
        out ReplayProbeHit hit)
    {
        hit = default;
        if (!CanProbeFinalMesh || _previewRoot == null ||
            _finalTriangles.Count < 3 ||
            maximumDistanceMeters <= 0f)
            return false;

        Transform stage = _previewRoot.transform;
        Vector3 origin = stage.InverseTransformPoint(worldRay.origin);
        Vector3 direction = stage.InverseTransformDirection(
            worldRay.direction).normalized;
        float nearest = maximumDistanceMeters;
        int nearestTriangle = -1;
        float nearestU = 0f;
        float nearestV = 0f;

        for (int index = 0; index < _finalTriangles.Count; index += 3)
        {
            Vector3 a = _finalVertices[_finalTriangles[index]];
            Vector3 b = _finalVertices[_finalTriangles[index + 1]];
            Vector3 c = _finalVertices[_finalTriangles[index + 2]];
            if (!RayTriangle(
                    origin, direction, a, b, c,
                    out float distance, out float u, out float v) ||
                distance > nearest)
                continue;
            nearest = distance;
            nearestTriangle = index / 3;
            nearestU = u;
            nearestV = v;
        }

        if (nearestTriangle < 0)
            return false;
        int triangleIndex = nearestTriangle * 3;
        Vector3 stageA = _finalVertices[_finalTriangles[triangleIndex]];
        Vector3 stageB = _finalVertices[_finalTriangles[triangleIndex + 1]];
        Vector3 stageC = _finalVertices[_finalTriangles[triangleIndex + 2]];
        Vector3 stagePoint = origin + direction * nearest;
        Vector3 stageNormal = Vector3.Cross(
            stageB - stageA, stageC - stageA).normalized;
        Vector3 worldPoint = stage.TransformPoint(stagePoint);
        Vector3 worldNormal = stage.TransformDirection(stageNormal).normalized;
        hit = new ReplayProbeHit(
            nearestTriangle,
            Vector3.Distance(worldRay.origin, worldPoint),
            new Vector3(1f - nearestU - nearestV, nearestU, nearestV),
            stagePoint,
            stageNormal,
            worldPoint,
            worldNormal,
            stageA,
            stageB,
            stageC);
        return true;
    }

    public Vector3 ReplayWorldToStagePoint(Vector3 worldPoint)
    {
        return _previewRoot != null
            ? _previewRoot.transform.InverseTransformPoint(worldPoint)
            : worldPoint;
    }

    public Vector3 ReplayWorldToStageDirection(Vector3 worldDirection)
    {
        return _previewRoot != null
            ? _previewRoot.transform.InverseTransformDirection(
                worldDirection).normalized
            : worldDirection.normalized;
    }

    private static bool RayTriangle(
        Vector3 origin,
        Vector3 direction,
        Vector3 a,
        Vector3 b,
        Vector3 c,
        out float distance,
        out float u,
        out float v)
    {
        Vector3 edge1 = b - a;
        Vector3 edge2 = c - a;
        Vector3 cross = Vector3.Cross(direction, edge2);
        float determinant = Vector3.Dot(edge1, cross);
        if (Mathf.Abs(determinant) < 0.000001f)
        {
            distance = u = v = 0f;
            return false;
        }
        float inverse = 1f / determinant;
        Vector3 fromA = origin - a;
        u = Vector3.Dot(fromA, cross) * inverse;
        if (u < 0f || u > 1f)
        {
            distance = v = 0f;
            return false;
        }
        Vector3 cross2 = Vector3.Cross(fromA, edge1);
        v = Vector3.Dot(direction, cross2) * inverse;
        if (v < 0f || u + v > 1f)
        {
            distance = 0f;
            return false;
        }
        distance = Vector3.Dot(edge2, cross2) * inverse;
        return distance >= 0f;
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
        chunk.publishedVertices = Array.Empty<Vector3>();
        chunk.publishedTriangles = Array.Empty<int>();
        chunk.candidateTriangleCount = 0;
        chunk.triangleCount = 0;
        chunk.pendingReplacementVoxels = 0;
    }

    private void DisposeGlobalLinePreview()
    {
        if (_globalLineFrontMesh != null)
            Destroy(_globalLineFrontMesh);
        if (_globalLineBackMesh != null)
            Destroy(_globalLineBackMesh);
        if (_globalLineObject != null)
            Destroy(_globalLineObject);
        _globalLineObject = null;
        _globalLineFilter = null;
        _globalLineRenderer = null;
        _globalLineFrontMesh = null;
        _globalLineBackMesh = null;
        _globalLineIndexCount = 0;
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
        ClearRenderPages();
        DisposeGlobalLinePreview();
        if (_lineMaterial != null) Destroy(_lineMaterial);
        if (_depthMaterial != null) Destroy(_depthMaterial);
        if (_previewRoot != null) Destroy(_previewRoot);
    }
}

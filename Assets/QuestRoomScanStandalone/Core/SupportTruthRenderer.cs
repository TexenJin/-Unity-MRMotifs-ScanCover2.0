using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Route-validation truth view. Each accepted TSDF zero-crossing node becomes
    /// one world-space oriented Surfel. In paper mode the same shared edge points
    /// are locally triangulated inside each support-sample cube, so the visible
    /// carrier exposes real connectivity without HERA page ownership or CPU
    /// readback. Audit mode still renders the independent Surfels. GPU-only
    /// evidence state lets separate edge observations accumulate into owned
    /// paper. Published cell topology owns a compact cell-local geometry
    /// snapshot, so a closed replacement switches connectivity and positions
    /// atomically. Complex cells may publish several separately validated local
    /// sheets; they are never collapsed into one cross-surface fan.
    /// </summary>
    public sealed class SupportTruthRenderer : MonoBehaviour
    {
        private ComputeShader _compute;
        private Material _material;
        private MaterialPropertyBlock _props;
        private sealed class SnapshotBuffers
        {
            public GraphicsBuffer Points;
            public GraphicsBuffer Normals;
            public GraphicsBuffer PointCells;
            public GraphicsBuffer Counters;
            public GraphicsBuffer DrawArgs;
            public GraphicsBuffer TopologyPoints;
            public GraphicsBuffer TopologyNormals;
            public GraphicsBuffer TopologyIndices;
            public GraphicsBuffer TopologyDrawArgs;

            public void Release()
            {
                Points?.Release();
                Normals?.Release();
                PointCells?.Release();
                Counters?.Release();
                DrawArgs?.Release();
                TopologyPoints?.Release();
                TopologyNormals?.Release();
                TopologyIndices?.Release();
                TopologyDrawArgs?.Release();
                Points = null;
                Normals = null;
                PointCells = null;
                Counters = null;
                DrawArgs = null;
                TopologyPoints = null;
                TopologyNormals = null;
                TopologyIndices = null;
                TopologyDrawArgs = null;
            }
        }

        private SnapshotBuffers _front;
        private SnapshotBuffers _back;
        private GraphicsBuffer _cellPointMap;
        private GraphicsBuffer _topologyCellStates;
        private GraphicsBuffer _topologyCellGeometry;
        private Bounds _bounds;

        private int _kClear;
        private int _kClearCells;
        private int _kEmit;
        private int _kRetain;
        private int _kClassify;
        private int _kTopology;
        private int _kArgs;
        private int3 _voxCount;
        private int3 _sampleCount;
        private int3 _topologyCellCount;
        private float _voxSize;
        private int _stride = 2;
        private float _minWeight = 0.04f;
        private float _hz = 4f;
        private int _maxPoints = 65536;
        private int _maxTopologyIndices;
        private int _maxTopologyVertices;
        private int _totalSampleCells;
        private int _totalMapEntries;
        private int _totalTopologyCells;
        private float _nextExtractTime;
        private int _holdFrames = 3;
        private bool _ready;
        private bool _visible;
        private bool _auditMode = true;
        private bool _paperGridEnabled;
        private bool _statsReadbackPending;
        private float _nextStatsReadbackTime;
        private uint _lastPointCount;
        private uint _lastTopologyTriangleCount;
        private uint _lastTopologyOverflow;
        private uint _lastTopologyVertexOverflow;
        private static bool _missingLogged;

        private static readonly Color TruthColor = new Color(1.0f, 0.52f, 0.08f, 0.92f);
        private static readonly Color PaperColor = new Color(0.11f, 0.15f, 0.18f, 1.0f);

        private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ID_VoxCount = Shader.PropertyToID("_VoxCount");
        private static readonly int ID_VoxSize = Shader.PropertyToID("_VoxSize");
        private static readonly int ID_SupportStride = Shader.PropertyToID("_SupportStride");
        private static readonly int ID_SupportMinWeight = Shader.PropertyToID("_SupportMinWeight");
        private static readonly int ID_SupportSampleCount = Shader.PropertyToID("_SupportSampleCount");
        private static readonly int ID_SupportMaxPoints = Shader.PropertyToID("_SupportMaxPoints");
        private static readonly int ID_SupportMaxTopologyIndices = Shader.PropertyToID("_SupportMaxTopologyIndices");
        private static readonly int ID_SupportTopologyCellCount = Shader.PropertyToID("_SupportTopologyCellCount");
        private static readonly int ID_SupportPoints = Shader.PropertyToID("_SupportPoints");
        private static readonly int ID_SupportNormals = Shader.PropertyToID("_SupportNormals");
        private static readonly int ID_SupportPointCells = Shader.PropertyToID("_SupportPointCells");
        private static readonly int ID_SupportCellPointMap = Shader.PropertyToID("_SupportCellPointMap");
        private static readonly int ID_SupportCounters = Shader.PropertyToID("_SupportCounters");
        private static readonly int ID_SupportDrawArgs = Shader.PropertyToID("_SupportDrawArgs");
        private static readonly int ID_SupportTopologyIndices = Shader.PropertyToID("_SupportTopologyIndices");
        private static readonly int ID_SupportTopologyPoints = Shader.PropertyToID("_SupportTopologyPoints");
        private static readonly int ID_SupportTopologyNormals = Shader.PropertyToID("_SupportTopologyNormals");
        private static readonly int ID_SupportTopologyCellStates = Shader.PropertyToID("_SupportTopologyCellStates");
        private static readonly int ID_SupportTopologyCellGeometry = Shader.PropertyToID("_SupportTopologyCellGeometry");
        private static readonly int ID_SupportMaxTopologyVertices = Shader.PropertyToID("_SupportMaxTopologyVertices");
        private static readonly int ID_SupportTopologyDrawArgs = Shader.PropertyToID("_SupportTopologyDrawArgs");
        private static readonly int ID_SupportPreviousPoints = Shader.PropertyToID("_SupportPreviousPoints");
        private static readonly int ID_SupportPreviousNormals = Shader.PropertyToID("_SupportPreviousNormals");
        private static readonly int ID_SupportPreviousPointCells = Shader.PropertyToID("_SupportPreviousPointCells");
        private static readonly int ID_SupportPreviousCounters = Shader.PropertyToID("_SupportPreviousCounters");
        private static readonly int ID_SupportHoldFrames = Shader.PropertyToID("_SupportHoldFrames");
        private static readonly int ID_SupportColor = Shader.PropertyToID("_SupportColor");
        private static readonly int ID_SupportPaperColor = Shader.PropertyToID("_SupportPaperColor");
        private static readonly int ID_SupportAuditMode = Shader.PropertyToID("_SupportAuditMode");
        private static readonly int ID_SupportPaperGrid = Shader.PropertyToID("_SupportPaperGrid");
        private static readonly int ID_SupportTopologyMesh = Shader.PropertyToID("_SupportTopologyMesh");
        private static readonly int ID_SupportSurfelRadius = Shader.PropertyToID("_SupportSurfelRadius");

        public bool Visible
        {
            get => _visible;
            set => _visible = value;
        }

        public bool IsReady => _ready;
        public string StatsCompact => !_ready
            ? "纸无"
            : $"点{_lastPointCount} 三{_lastTopologyTriangleCount}" +
              (_lastTopologyOverflow > 0 || _lastTopologyVertexOverflow > 0
                  ? $" 溢{_lastTopologyOverflow}/{_lastTopologyVertexOverflow}" : "");

        public void ResetHistory()
        {
            if (_front == null || _back == null) return;
            _front.Counters.SetData(new uint[6]);
            _front.DrawArgs.SetData(new uint[5]);
            _front.TopologyDrawArgs.SetData(new uint[5]);
            _back.Counters.SetData(new uint[6]);
            _back.DrawArgs.SetData(new uint[5]);
            _back.TopologyDrawArgs.SetData(new uint[5]);
            if (_topologyCellStates != null)
                _topologyCellStates.SetData(new uint2[_totalTopologyCells]);
            _lastPointCount = 0;
            _lastTopologyTriangleCount = 0;
            _lastTopologyOverflow = 0;
            _lastTopologyVertexOverflow = 0;
            _nextExtractTime = 0f;
        }

        public bool AuditMode
        {
            get => _auditMode;
            set
            {
                _auditMode = value;
                _props?.SetFloat(ID_SupportAuditMode, value ? 1f : 0f);
            }
        }

        public bool PaperGridEnabled
        {
            get => _paperGridEnabled;
            set
            {
                _paperGridEnabled = value;
                _props?.SetFloat(ID_SupportPaperGrid, value ? 1f : 0f);
            }
        }

        public bool Initialize(ComputeShader compute, Shader truthShader, int3 voxCount,
            float voxSize, int stride, float minWeight, float extractHz, float surfelRadius,
            float holdSeconds)
        {
            if (_ready) return true;

            Shader shader = truthShader != null
                ? truthShader
                : Shader.Find("Genesis/ScanSupportTruth");
            if (compute == null || shader == null)
            {
                if (!_missingLogged)
                {
                    Logger.Error($"SupportTruthRenderer: compute 或 shader 缺失 " +
                        $"(compute={(compute == null ? "NULL" : compute.name)}, " +
                        $"shader={(shader == null ? "NULL" : shader.name)})");
                    _missingLogged = true;
                }
                return false;
            }

            _compute = compute;
            _material = new Material(shader);
            _voxCount = voxCount;
            _voxSize = voxSize;
            _stride = Mathf.Max(1, stride);
            _minWeight = minWeight;
            _hz = Mathf.Clamp(extractHz, 0.5f, 30f);
            _holdFrames = Mathf.Clamp(Mathf.CeilToInt(Mathf.Max(0f, holdSeconds) * _hz), 0, 12);
            _sampleCount = new int3(
                Mathf.Max(1, (voxCount.x - 1) / _stride),
                Mathf.Max(1, (voxCount.y - 1) / _stride),
                Mathf.Max(1, (voxCount.z - 1) / _stride));
            _topologyCellCount = new int3(
                Mathf.Max(1, _sampleCount.x - 1),
                Mathf.Max(1, _sampleCount.y - 1),
                Mathf.Max(1, _sampleCount.z - 1));
            _totalSampleCells = _sampleCount.x * _sampleCount.y * _sampleCount.z;
            _totalMapEntries = _totalSampleCells * 3;
            _totalTopologyCells = _topologyCellCount.x *
                _topologyCellCount.y * _topologyCellCount.z;
            // Nine indices per support crossing is deliberately generous for
            // local 3..6-point polygons and remains a small GPU-only buffer.
            // Keep it divisible by three so overflow cannot expose an unwritten
            // partial triangle at the tail of the indirect draw span.
            _maxTopologyIndices = _maxPoints * 9;
            // Topology vertices are cell-local snapshots.  Two support-point
            // budgets leave ample room for duplicated cell boundaries while
            // keeping the Quest allocation bounded.
            _maxTopologyVertices = _maxPoints * 2;

            _front = CreateSnapshotBuffers();
            _back = CreateSnapshotBuffers();
            _cellPointMap = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, _totalMapEntries, 4);
            _topologyCellStates = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, _totalTopologyCells, 8);
            _topologyCellStates.SetData(new uint2[_totalTopologyCells]);
            // xyz pack twelve 8-bit edge parameters; w stores the optional
            // multi-sheet cluster labels.  This is the geometry half of the
            // paper cell's two-phase commit.
            _topologyCellGeometry = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, _totalTopologyCells, 16);

            _kClear = compute.FindKernel("SupportClear");
            _kClearCells = compute.FindKernel("SupportClearCells");
            _kEmit = compute.FindKernel("SupportEmitCrossings");
            _kRetain = compute.FindKernel("SupportRetainPrevious");
            _kClassify = compute.FindKernel("SupportClassifyTopology");
            _kTopology = compute.FindKernel("SupportBuildTopology");
            _kArgs = compute.FindKernel("SupportBuildArgs");

            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
            _compute.SetInts(ID_SupportSampleCount,
                _sampleCount.x, _sampleCount.y, _sampleCount.z);
            _compute.SetInts(ID_SupportTopologyCellCount,
                _topologyCellCount.x, _topologyCellCount.y, _topologyCellCount.z);
            _compute.SetFloat(ID_VoxSize, _voxSize);
            _compute.SetInt(ID_SupportStride, _stride);
            _compute.SetFloat(ID_SupportMinWeight, _minWeight);
            _compute.SetInt(ID_SupportMaxPoints, _maxPoints);
            _compute.SetInt(ID_SupportMaxTopologyIndices, _maxTopologyIndices);
            _compute.SetInt(ID_SupportMaxTopologyVertices, _maxTopologyVertices);
            _compute.SetFloat(ID_SupportSurfelRadius, surfelRadius);
            _compute.SetInt(ID_SupportHoldFrames, _holdFrames);

            BindBuffer(_kClearCells, ID_SupportCellPointMap, _cellPointMap);
            BindBuffer(_kEmit, ID_SupportCellPointMap, _cellPointMap);
            BindBuffer(_kRetain, ID_SupportCellPointMap, _cellPointMap);
            BindBuffer(_kClassify, ID_SupportCellPointMap, _cellPointMap);
            BindBuffer(_kTopology, ID_SupportCellPointMap, _cellPointMap);
            BindBuffer(_kTopology, ID_SupportTopologyCellStates, _topologyCellStates);
            BindBuffer(_kTopology, ID_SupportTopologyCellGeometry, _topologyCellGeometry);

            _props = new MaterialPropertyBlock();
            _props.SetColor(ID_SupportColor, TruthColor);
            _props.SetColor(ID_SupportPaperColor, PaperColor);
            _props.SetFloat(ID_SupportAuditMode, 1f);
            _props.SetFloat(ID_SupportPaperGrid, 0f);
            _props.SetFloat(ID_SupportTopologyMesh, 0f);
            _props.SetFloat(ID_SupportSurfelRadius, surfelRadius);
            SetDrawSnapshot(_front);

            Vector3 size = new Vector3(
                _voxCount.x * _voxSize, _voxCount.y * _voxSize, _voxCount.z * _voxSize);
            _bounds = new Bounds(Vector3.zero, size);
            _ready = true;
            Logger.Info($"支撑拓扑审计已初始化：采样步长={_stride * _voxSize:0.00}m，" +
                $"片半径={surfelRadius:0.00}m，格={_sampleCount}，" +
                $"上限={_maxPoints}，纸面拓扑格={_topologyCellCount}，" +
                $"提取={_hz:0.#}Hz，候选会合窗={_holdFrames / _hz:0.00}s，" +
                $"成纸后按反证撤销");
            return true;
        }

        private void Update()
        {
            if (!_ready || !_visible || Time.unscaledTime < _nextExtractTime) return;
            var volume = VolumeIntegrator.Instance != null ? VolumeIntegrator.Instance.Volume : null;
            if (volume == null || !volume.IsCreated()) return;

            _nextExtractTime = Time.unscaledTime + 1f / _hz;
            SnapshotBuffers oldFront = _front;
            _front = _back;
            _back = oldFront;
            BindFrameBuffers();
            _compute.SetTexture(_kEmit, ID_TsdfVolume, volume);
            _compute.SetTexture(_kRetain, ID_TsdfVolume, volume);
            // Compute textures are bound per kernel.  The topology adjudicator
            // now reads the eight TSDF corners for its asymptotic face decision,
            // so sharing the same shader property name is not enough: without
            // this binding SupportBuildTopology receives no volume and emits
            // zero indices while SupportEmitCrossings still reports points.
            _compute.SetTexture(_kTopology, ID_TsdfVolume, volume);
            _compute.Dispatch(_kClear, 1, 1, 1);
            _compute.Dispatch(_kClearCells, CeilDiv(_totalMapEntries, 64), 1, 1);
            _compute.Dispatch(_kEmit,
                CeilDiv(_sampleCount.x, 4),
                CeilDiv(_sampleCount.y, 4),
                CeilDiv(_sampleCount.z, 4));
            if (_holdFrames > 0)
                _compute.Dispatch(_kRetain, CeilDiv(_maxPoints, 64), 1, 1);
            _compute.Dispatch(_kClassify, CeilDiv(_maxPoints, 64), 1, 1);
            _compute.Dispatch(_kTopology,
                CeilDiv(_topologyCellCount.x, 4),
                CeilDiv(_topologyCellCount.y, 4),
                CeilDiv(_topologyCellCount.z, 4));
            _compute.Dispatch(_kArgs, 1, 1, 1);
            SetDrawSnapshot(_front);
            RequestCompactStats();
        }

        private void RequestCompactStats()
        {
            if (_statsReadbackPending || Time.unscaledTime < _nextStatsReadbackTime)
                return;
            _statsReadbackPending = true;
            _nextStatsReadbackTime = Time.unscaledTime + 0.8f;
            AsyncGPUReadback.Request(_front.Counters, request =>
            {
                _statsReadbackPending = false;
                if (this == null || request.hasError) return;
                var data = request.GetData<uint>();
                if (data.Length < 6) return;
                _lastPointCount = data[0];
                _lastTopologyTriangleCount = data[2] / 3u;
                _lastTopologyOverflow = data[3];
                _lastTopologyVertexOverflow = data[5];
            });
        }

        private void LateUpdate()
        {
            if (!_ready || !_visible) return;
            bool drawTopologyMesh = _paperGridEnabled || !_auditMode;
            _props.SetFloat(ID_SupportTopologyMesh, drawTopologyMesh ? 1f : 0f);
            var rp = new RenderParams(_material)
            {
                worldBounds = _bounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };
            Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles,
                drawTopologyMesh ? _front.TopologyDrawArgs : _front.DrawArgs, 1);
        }

        private void OnDestroy()
        {
            _front?.Release();
            _back?.Release();
            _cellPointMap?.Release();
            _topologyCellStates?.Release();
            _topologyCellGeometry?.Release();
            _front = null;
            _back = null;
            _cellPointMap = null;
            _topologyCellStates = null;
            _topologyCellGeometry = null;
            if (_material != null)
            {
                Destroy(_material);
                _material = null;
            }
            _statsReadbackPending = false;
            _ready = false;
        }

        private SnapshotBuffers CreateSnapshotBuffers()
        {
            const GraphicsBuffer.Target structuredIndirect =
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;
            var buffers = new SnapshotBuffers
            {
                Points = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxPoints, 16),
                Normals = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxPoints, 16),
                PointCells = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxPoints, 8),
                Counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 6, 4),
                DrawArgs = new GraphicsBuffer(structuredIndirect, 5, 4),
                TopologyPoints = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _maxTopologyVertices, 16),
                TopologyNormals = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _maxTopologyVertices, 16),
                TopologyIndices = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _maxTopologyIndices, 4),
                TopologyDrawArgs = new GraphicsBuffer(structuredIndirect, 5, 4)
            };
            buffers.Counters.SetData(new uint[6]);
            buffers.DrawArgs.SetData(new uint[5]);
            buffers.TopologyDrawArgs.SetData(new uint[5]);
            return buffers;
        }

        private void BindFrameBuffers()
        {
            BindBuffer(_kClear, ID_SupportCounters, _front.Counters);

            BindBuffer(_kEmit, ID_SupportPoints, _front.Points);
            BindBuffer(_kEmit, ID_SupportNormals, _front.Normals);
            BindBuffer(_kEmit, ID_SupportPointCells, _front.PointCells);
            BindBuffer(_kEmit, ID_SupportCounters, _front.Counters);

            BindBuffer(_kRetain, ID_SupportPoints, _front.Points);
            BindBuffer(_kRetain, ID_SupportNormals, _front.Normals);
            BindBuffer(_kRetain, ID_SupportPointCells, _front.PointCells);
            BindBuffer(_kRetain, ID_SupportCounters, _front.Counters);
            BindBuffer(_kRetain, ID_SupportPreviousPoints, _back.Points);
            BindBuffer(_kRetain, ID_SupportPreviousNormals, _back.Normals);
            BindBuffer(_kRetain, ID_SupportPreviousPointCells, _back.PointCells);
            BindBuffer(_kRetain, ID_SupportPreviousCounters, _back.Counters);

            BindBuffer(_kClassify, ID_SupportPoints, _front.Points);
            BindBuffer(_kClassify, ID_SupportNormals, _front.Normals);
            BindBuffer(_kClassify, ID_SupportPointCells, _front.PointCells);
            BindBuffer(_kClassify, ID_SupportCounters, _front.Counters);

            BindBuffer(_kTopology, ID_SupportPoints, _front.Points);
            BindBuffer(_kTopology, ID_SupportNormals, _front.Normals);
            // SupportBuildTopology reads the missing-frame age when a face has
            // three retained/current crossings.  Compute-buffer bindings are
            // per kernel on Unity/Quest; omitting this binding invalidates the
            // whole topology dispatch even when a particular cell does not
            // enter that branch, leaving pointCount > 0 but triangleCount = 0.
            BindBuffer(_kTopology, ID_SupportPointCells, _front.PointCells);
            BindBuffer(_kTopology, ID_SupportCounters, _front.Counters);
            BindBuffer(_kTopology, ID_SupportTopologyPoints, _front.TopologyPoints);
            BindBuffer(_kTopology, ID_SupportTopologyNormals, _front.TopologyNormals);
            BindBuffer(_kTopology, ID_SupportTopologyIndices, _front.TopologyIndices);

            BindBuffer(_kArgs, ID_SupportCounters, _front.Counters);
            BindBuffer(_kArgs, ID_SupportDrawArgs, _front.DrawArgs);
            BindBuffer(_kArgs, ID_SupportTopologyDrawArgs, _front.TopologyDrawArgs);
        }

        private void SetDrawSnapshot(SnapshotBuffers snapshot)
        {
            if (_props == null || snapshot == null) return;
            _props.SetBuffer(ID_SupportPoints, snapshot.Points);
            _props.SetBuffer(ID_SupportNormals, snapshot.Normals);
            _props.SetBuffer(ID_SupportTopologyPoints, snapshot.TopologyPoints);
            _props.SetBuffer(ID_SupportTopologyNormals, snapshot.TopologyNormals);
            _props.SetBuffer(ID_SupportTopologyIndices, snapshot.TopologyIndices);
        }

        private void BindBuffer(int kernel, int id, GraphicsBuffer buffer)
        {
            _compute.SetBuffer(kernel, id, buffer);
        }

        private static int CeilDiv(int value, int divisor)
        {
            return (value + divisor - 1) / divisor;
        }
    }
}

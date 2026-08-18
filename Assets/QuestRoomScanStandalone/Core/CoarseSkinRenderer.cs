using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 粗皮渲染器（08-19 路线A，用户"套皮"设想）：同一份 TSDF 原料，按粗晶格
    /// （默认 stride 4 = 20cm 网眼）二次 Surface Nets 提取，画出几何级大三角形
    /// 网格 = Meta 系统网格观感本尊。三角形量约为细网 1/16，光栅成本大降。
    /// 纯只读 _TsdfVolume、独立缓冲、独立绘制——不碰融合层，不继承诊断类。
    /// 定位=产品态显示层；细网（ScanMeshVertexColor）保留为诊断层。
    /// 显开/显关总闸（右摇杆直接按下）连带切它。
    /// </summary>
    internal class CoarseSkinRenderer : MonoBehaviour
    {
        private ComputeShader _compute;
        private Material _material;
        private MaterialPropertyBlock _props;
        private Bounds _bounds;

        private GraphicsBuffer _cellVertMap;
        private GraphicsBuffer _verts;
        private GraphicsBuffer _vertCells;
        private GraphicsBuffer _indices;
        private GraphicsBuffer _counters;
        private GraphicsBuffer _drawArgs;

        private int _kClear;
        private int _kEmit;
        private int _kIndices;
        private int _kArgs;

        private int3 _voxCount;
        private float _voxSize;
        private int _stride = 4;
        private float _minWeight = 0.04f;
        private float _hz = 4f;
        private int3 _cellCount;
        private int _maxVertices;
        private int _maxIndices;
        private float _nextExtractTime;

        private bool _ready;
        private bool _visible = true;
        private static bool _shaderMissingLogged;

        private static readonly Color SkinColor = new Color(0.55f, 0.75f, 0.95f, 0.85f);

        private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ID_VoxCount = Shader.PropertyToID("_VoxCount");
        private static readonly int ID_VoxSize = Shader.PropertyToID("_VoxSize");
        private static readonly int ID_SkinStride = Shader.PropertyToID("_SkinStride");
        private static readonly int ID_SkinMinWeight = Shader.PropertyToID("_SkinMinWeight");
        private static readonly int ID_SkinMaxVertices = Shader.PropertyToID("_SkinMaxVertices");
        private static readonly int ID_SkinMaxIndices = Shader.PropertyToID("_SkinMaxIndices");
        private static readonly int ID_SkinCellCount = Shader.PropertyToID("_SkinCellCount");
        private static readonly int ID_SkinCellVertMap = Shader.PropertyToID("_SkinCellVertMap");
        private static readonly int ID_SkinVerts = Shader.PropertyToID("_SkinVerts");
        private static readonly int ID_SkinVertCells = Shader.PropertyToID("_SkinVertCells");
        private static readonly int ID_SkinIndices = Shader.PropertyToID("_SkinIndices");
        private static readonly int ID_SkinCounters = Shader.PropertyToID("_SkinCounters");
        private static readonly int ID_SkinDrawArgs = Shader.PropertyToID("_SkinDrawArgs");
        private static readonly int ID_SkinColor = Shader.PropertyToID("_SkinColor");

        public bool Visible
        {
            get => _visible;
            set => _visible = value;
        }

        public bool IsReady => _ready;

        /// <summary>幂等。compute 或 shader 缺失时返回 false，皮整体缺席但不影响任何既有路径。</summary>
        public bool Initialize(ComputeShader compute, int3 voxCount, float voxSize,
            int stride, float minWeight, float extractHz)
        {
            if (_ready) return true;

            var shader = Shader.Find("Genesis/ScanCoarseSkin");
            if (compute == null || shader == null)
            {
                if (!_shaderMissingLogged)
                {
                    Logger.Error($"CoarseSkinRenderer: compute 或 shader 缺失 " +
                        $"(compute={(compute == null ? "NULL" : compute.name)}, " +
                        $"shader={(shader == null ? "NULL" : shader.name)})，粗皮缺席");
                    _shaderMissingLogged = true;
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

            // 粗格 cell 数：角点落在 cell*stride，远侧角点钳到 N-1（kernel 内处理）。
            _cellCount = new int3(
                Mathf.Max(1, (voxCount.x - 1) / _stride),
                Mathf.Max(1, (voxCount.y - 1) / _stride),
                Mathf.Max(1, (voxCount.z - 1) / _stride));
            int totalCells = _cellCount.x * _cellCount.y * _cellCount.z;

            // 192x128x192 / stride4 → 47x31x47 ≈ 6.8 万 cell。表面 cell 占比
            // 实测一般 <20%，24k 顶点上限留足裕量；缓冲合计 <2MB。
            _maxVertices = Mathf.Min(24576, totalCells);
            _maxIndices = _maxVertices * 6;

            const GraphicsBuffer.Target structuredIndirect =
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments;

            _cellVertMap = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalCells, 4);
            _verts = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, 16);
            _vertCells = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, 4);
            _indices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxIndices, 4);
            _counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 4, 4);
            _drawArgs = new GraphicsBuffer(structuredIndirect, 5, 4);
            // 新缓冲内容未定义：首帧 LateUpdate 若抢在首次提取前画，会按垃圾
            // 计数发起超大规模绘制（GPU 挂死风险）——先清零。
            _counters.SetData(new uint[4]);
            _drawArgs.SetData(new uint[5]);

            _kClear = compute.FindKernel("SkinClear");
            _kEmit = compute.FindKernel("SkinClassifyAndEmit");
            _kIndices = compute.FindKernel("SkinGenerateIndices");
            _kArgs = compute.FindKernel("SkinBuildArgs");

            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
            _compute.SetInts(ID_SkinCellCount, _cellCount.x, _cellCount.y, _cellCount.z);
            _compute.SetFloat(ID_VoxSize, _voxSize);
            _compute.SetInt(ID_SkinStride, _stride);
            _compute.SetFloat(ID_SkinMinWeight, _minWeight);
            _compute.SetInt(ID_SkinMaxVertices, _maxVertices);
            _compute.SetInt(ID_SkinMaxIndices, _maxIndices);

            BindBuffer(_kClear, ID_SkinCounters, _counters);
            BindBuffer(_kEmit, ID_SkinCellVertMap, _cellVertMap);
            BindBuffer(_kEmit, ID_SkinVerts, _verts);
            BindBuffer(_kEmit, ID_SkinVertCells, _vertCells);
            BindBuffer(_kEmit, ID_SkinCounters, _counters);
            BindBuffer(_kIndices, ID_SkinCellVertMap, _cellVertMap);
            BindBuffer(_kIndices, ID_SkinVertCells, _vertCells);
            BindBuffer(_kIndices, ID_SkinIndices, _indices);
            BindBuffer(_kIndices, ID_SkinCounters, _counters);
            BindBuffer(_kArgs, ID_SkinCounters, _counters);
            BindBuffer(_kArgs, ID_SkinDrawArgs, _drawArgs);

            _props = new MaterialPropertyBlock();
            _props.SetBuffer(ID_SkinVerts, _verts);
            _props.SetBuffer(ID_SkinIndices, _indices);
            _props.SetColor(ID_SkinColor, SkinColor);

            Vector3 size = new Vector3(
                _voxCount.x * _voxSize, _voxCount.y * _voxSize, _voxCount.z * _voxSize);
            _bounds = new Bounds(Vector3.zero, size);

            _ready = true;
            Logger.Info($"粗皮已初始化：cell={_cellCount}（网眼 {_stride * _voxSize:0.00}m），" +
                $"maxVerts={_maxVertices}，提取 {_hz:0.#}Hz");
            return true;
        }

        private void Update()
        {
            if (!_ready || !_visible) return;
            if (Time.unscaledTime < _nextExtractTime) return;

            var vol = VolumeIntegrator.Instance != null ? VolumeIntegrator.Instance.Volume : null;
            if (vol == null || !vol.IsCreated()) return;

            _nextExtractTime = Time.unscaledTime + 1f / _hz;

            _compute.SetTexture(_kEmit, ID_TsdfVolume, vol);
            _compute.SetTexture(_kIndices, ID_TsdfVolume, vol);

            _compute.Dispatch(_kClear, 1, 1, 1);
            _compute.Dispatch(_kEmit,
                CeilDiv(_cellCount.x, 4), CeilDiv(_cellCount.y, 4), CeilDiv(_cellCount.z, 4));
            // 索引内核按顶点数在 GPU 内早退；固定派满上限组数省一道间接参数。
            _compute.Dispatch(_kIndices, CeilDiv(_maxVertices, 64), 1, 1);
            _compute.Dispatch(_kArgs, 1, 1, 1);
        }

        private void LateUpdate()
        {
            if (!_ready || !_visible) return;

            var rp = new RenderParams(_material)
            {
                worldBounds = _bounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };
            Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles, _drawArgs, 1);
        }

        private void OnDestroy()
        {
            ReleaseBuffers();
            if (_material != null)
            {
                Destroy(_material);
                _material = null;
            }
        }

        private void ReleaseBuffers()
        {
            _cellVertMap?.Release();
            _verts?.Release();
            _vertCells?.Release();
            _indices?.Release();
            _counters?.Release();
            _drawArgs?.Release();
            _cellVertMap = null;
            _verts = null;
            _vertCells = null;
            _indices = null;
            _counters = null;
            _drawArgs = null;
            _ready = false;
        }

        private void BindBuffer(int kernel, int nameID, GraphicsBuffer buffer)
        {
            _compute.SetBuffer(kernel, nameID, buffer);
        }

        private static int CeilDiv(int a, int b) => (a + b - 1) / b;
    }
}

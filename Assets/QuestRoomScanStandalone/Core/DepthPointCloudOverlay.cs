using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 融合前只读“BB 弹道”探针。定时把当前深度与世界法线打包成 GPU 快照，
    /// 再用采集当刻的深度相机矩阵反投影到世界空间并短暂留痕。
    ///
    /// 这条旁路只读 gsDepthTex/gsDepthNormalTex：不回读 CPU、不写 TSDF、不参与冻结。
    /// 正视点偏白，掠射角逐渐变黄/洋红；旧快照逐渐淡出。平墙应收成薄层，
    /// 转角若存在多深度主张，会直接显成厚带、双层或楔形。
    /// </summary>
    public sealed class DepthPointCloudOverlay : MonoBehaviour
    {
        [SerializeField, Range(1f, 8f), Tooltip("BB 点尺寸（像素）")]
        private float pointSize = 3f;

        [SerializeField, Range(64, 192), Tooltip("单帧点阵最大边长；仅影响诊断显示密度")]
        private int maxGridDim = 128;

        [SerializeField, Range(4, 16), Tooltip("世界空间留痕层数")]
        private int trailLayerCount = 12;

        [SerializeField, Range(0.1f, 1f), Tooltip("深度快照间隔（秒）")]
        private float captureInterval = 0.25f;

        [SerializeField, Range(1f, 8f), Tooltip("BB 留痕寿命（秒）")]
        private float trailLifetime = 3f;

        [SerializeField, Tooltip("与生产融合一致使用右眼深度；关闭时仅用于诊断左眼视差")]
        private bool useRightEye = true;

        private sealed class TrailLayer
        {
            public GameObject GameObject;
            public MeshRenderer Renderer;
            public Material Material;
            public RenderTexture Snapshot;
            public float CapturedAt;
            public bool Valid;
        }

        private Mesh _mesh;
        private Material _snapshotPackMaterial;
        private TrailLayer[] _layers;
        private bool _built;
        private bool _visible;
        private bool _acquiring;
        private int _nextLayer;
        private float _nextCaptureAt;

        private static readonly int TexSizeID = Shader.PropertyToID("gsDepthTexSize");
        private static readonly int EyeIndexID = Shader.PropertyToID("_EyeIndex");
        private static readonly int SnapshotID = Shader.PropertyToID("_DepthNormalSnapshot");
        private static readonly int ProjID = Shader.PropertyToID("_SnapshotProj");
        private static readonly int ProjInvID = Shader.PropertyToID("_SnapshotProjInv");
        private static readonly int ViewInvID = Shader.PropertyToID("_SnapshotViewInv");
        private static readonly int SnapshotTexSizeID = Shader.PropertyToID("_SnapshotTexSize");
        private static readonly int PointSizeID = Shader.PropertyToID("_PointSize");
        private static readonly int AgeID = Shader.PropertyToID("_Age01");

        public bool Visible => _visible;

        private void Start()
        {
            Shader packShader = Resources.Load<Shader>("DepthPointSnapshotPack");
            Shader pointShader = Resources.Load<Shader>("DepthPointCloud");
            if (packShader == null || pointShader == null)
            {
                Logger.Warning("BB 深度探针 shader 缺失（DepthPointSnapshotPack/DepthPointCloud），探针跳过");
                enabled = false;
                return;
            }

            _snapshotPackMaterial = new Material(packShader) { name = "QRS BB Snapshot Pack" };
            if (DepthCapture.Instance != null)
                DepthCapture.Instance.Preprocessed += OnDepthPreprocessed;
            SetVisible(false);
        }

        public void SetVisible(bool visible)
        {
            _visible = visible;
            if (visible)
            {
                ClearTrail();
                _nextCaptureAt = 0f;
            }
            else
            {
                SetAllRenderers(false);
            }
        }

        public void SetAcquiring(bool acquiring)
        {
            _acquiring = acquiring;
            if (acquiring && _visible) _nextCaptureAt = 0f;
        }

        public void ClearTrail()
        {
            _nextLayer = 0;
            if (_layers == null) return;
            foreach (TrailLayer layer in _layers)
            {
                layer.Valid = false;
                if (layer.Renderer != null) layer.Renderer.enabled = false;
            }
        }

        private void Update()
        {
            if (!_built)
            {
                Vector2 size = Shader.GetGlobalVector(TexSizeID);
                if (size.x >= 8f && size.y >= 8f)
                    Build((int)size.x, (int)size.y);
            }

            if (!_built || !_visible) return;

            float now = Time.unscaledTime;
            UpdateLayerAges(now);
        }

        private void OnDepthPreprocessed()
        {
            if (!_visible || !_acquiring || !DepthCapture.DepthAvailable) return;

            if (!_built)
            {
                Vector2 size = Shader.GetGlobalVector(TexSizeID);
                if (size.x < 8f || size.y < 8f) return;
                Build((int)size.x, (int)size.y);
            }

            float now = Time.unscaledTime;
            if (now < _nextCaptureAt) return;
            _nextCaptureAt = now + captureInterval;
            CaptureSnapshot(now);
        }

        private void Build(int width, int height)
        {
            int step = Mathf.Max(1, Mathf.CeilToInt((float)Mathf.Max(width, height) / maxGridDim));
            int gridWidth = Mathf.Max(1, width / step);
            int gridHeight = Mathf.Max(1, height / step);
            int count = gridWidth * gridHeight;
            var vertices = new Vector3[count];
            var indices = new int[count];
            int index = 0;
            for (int y = 0; y < gridHeight; y++)
            for (int x = 0; x < gridWidth; x++, index++)
            {
                vertices[index] = new Vector3(x * step, y * step, 0f);
                indices[index] = index;
            }

            _mesh = new Mesh
            {
                name = "QRS BB Depth Pixels",
                indexFormat = IndexFormat.UInt32,
                vertices = vertices,
                bounds = new Bounds(Vector3.zero, Vector3.one * 10000f)
            };
            _mesh.SetIndices(indices, MeshTopology.Points, 0);

            Shader pointShader = Resources.Load<Shader>("DepthPointCloud");
            _layers = new TrailLayer[trailLayerCount];
            for (int i = 0; i < _layers.Length; i++)
            {
                var descriptor = new RenderTextureDescriptor(width, height,
                    GraphicsFormat.R16G16B16A16_SFloat, 0)
                {
                    dimension = TextureDimension.Tex2D,
                    volumeDepth = 1,
                    msaaSamples = 1,
                    useMipMap = false,
                    autoGenerateMips = false,
                    sRGB = false
                };
                var snapshot = new RenderTexture(descriptor)
                {
                    name = $"QRS BB Snapshot {i:00}",
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                snapshot.Create();

                var child = new GameObject($"[QRS] BB Trail {i:00}");
                child.transform.SetParent(transform, false);
                child.AddComponent<MeshFilter>().sharedMesh = _mesh;
                MeshRenderer renderer = child.AddComponent<MeshRenderer>();
                var material = new Material(pointShader) { name = $"QRS BB Trail {i:00}" };
                material.SetTexture(SnapshotID, snapshot);
                material.SetVector(SnapshotTexSizeID, new Vector4(width, height, 0f, 0f));
                renderer.sharedMaterial = material;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                renderer.enabled = false;

                _layers[i] = new TrailLayer
                {
                    GameObject = child,
                    Renderer = renderer,
                    Material = material,
                    Snapshot = snapshot
                };
            }

            _built = true;
            Logger.Info($"BB 反投影探针已建: {gridWidth}x{gridHeight} 点 × {_layers.Length} 层，右眼生产源，留痕 {trailLifetime:0.0}s");
        }

        private void CaptureSnapshot(float now)
        {
            DepthCapture capture = DepthCapture.Instance;
            if (capture == null || _layers == null || _layers.Length == 0) return;

            int eye = useRightEye ? 1 : 0;
            if (capture.Proj == null || capture.ProjInv == null || capture.ViewInv == null ||
                capture.Proj.Length <= eye || capture.ProjInv.Length <= eye || capture.ViewInv.Length <= eye)
                return;

            TrailLayer layer = _layers[_nextLayer];
            _snapshotPackMaterial.SetFloat(EyeIndexID, eye);
            Graphics.Blit(null, layer.Snapshot, _snapshotPackMaterial);

            layer.Material.SetMatrix(ProjID, capture.Proj[eye]);
            layer.Material.SetMatrix(ProjInvID, capture.ProjInv[eye]);
            layer.Material.SetMatrix(ViewInvID, capture.ViewInv[eye]);
            layer.Material.SetFloat(PointSizeID, pointSize);
            layer.Material.SetFloat(AgeID, 0f);
            layer.CapturedAt = now;
            layer.Valid = true;
            layer.Renderer.enabled = true;
            _nextLayer = (_nextLayer + 1) % _layers.Length;
        }

        private void UpdateLayerAges(float now)
        {
            if (_layers == null) return;
            foreach (TrailLayer layer in _layers)
            {
                if (!layer.Valid) continue;
                float age = now - layer.CapturedAt;
                if (age >= trailLifetime)
                {
                    layer.Valid = false;
                    layer.Renderer.enabled = false;
                    continue;
                }

                layer.Material.SetFloat(PointSizeID, pointSize);
                layer.Material.SetFloat(AgeID, Mathf.Clamp01(age / Mathf.Max(0.1f, trailLifetime)));
                layer.Renderer.enabled = true;
            }
        }

        private void SetAllRenderers(bool visible)
        {
            if (_layers == null) return;
            foreach (TrailLayer layer in _layers)
                if (layer.Renderer != null) layer.Renderer.enabled = visible && layer.Valid;
        }

        private void OnDestroy()
        {
            if (DepthCapture.Instance != null)
                DepthCapture.Instance.Preprocessed -= OnDepthPreprocessed;
            if (_layers != null)
            {
                foreach (TrailLayer layer in _layers)
                {
                    if (layer.Material != null) Destroy(layer.Material);
                    if (layer.Snapshot != null)
                    {
                        layer.Snapshot.Release();
                        Destroy(layer.Snapshot);
                    }
                }
            }
            if (_mesh != null) Destroy(_mesh);
            if (_snapshotPackMaterial != null) Destroy(_snapshotPackMaterial);
        }
    }
}

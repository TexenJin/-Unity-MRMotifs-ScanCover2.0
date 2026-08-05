using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 实时深度点云叠加层（世界空间）。建一个"顶点位置=深度图像素坐标"的 Points 网格，
    /// 顶点着色器每帧从全局 gsDepthTex 反投影出世界坐标——CPU 侧零每帧开销。
    /// 用途：把"当前深度"以 3D 点云叠在融合网格上同屏对照：
    ///   幽灵网格位置有点云覆盖 = 深度自洽幻觉（Meta 侧）；空空如也却有网格 = 矛盾在我们侧。
    /// 由 StandaloneRoomScanner 按 showDepthPointCloud 开关运行时创建，场景无需挂任何东西。
    /// </summary>
    public class DepthPointCloudOverlay : MonoBehaviour
    {
        [SerializeField, Range(1f, 12f), Tooltip("点尺寸（像素）。太小看不清、太大糊成一片")]
        private float pointSize = 4f;

        [SerializeField, Tooltip("网格最大边长（实际深度分辨率超过则均匀抽稀）")]
        private int maxGridDim = 192;

        [SerializeField, Tooltip("点云深度来源：勾选=右眼，不勾=左眼（融合用眼）。" +
            "右眼源的点云与右眼透视画面视差对齐——主要靠右眼看时勾选它。" +
            "所选眼深度无效时 shader 自动回退另一只眼，不会整片消失")]
        private bool useRightEye = true;

        [SerializeField, Range(0.1f, 1f), Tooltip("等深色带宽度（米）。越小轮廓线越密，起伏越易读")]
        private float bandWidth = 0.25f;

        private Material _mat;
        private MeshFilter _mf;
        private Mesh _mesh;
        private bool _built;

        private static readonly int PointSizeID = Shader.PropertyToID("_PointSize");
        private static readonly int EyeIndexID = Shader.PropertyToID("_EyeIndex");
        private static readonly int BandWidthID = Shader.PropertyToID("_BandWidth");
        private static readonly int TexSizeID = Shader.PropertyToID("gsDepthTexSize");

        private void Start()
        {
            var shader = Resources.Load<Shader>("DepthPointCloud");
            if (shader == null)
            {
                Logger.Warning("深度点云 shader 未找到（Resources/DepthPointCloud），叠加层跳过");
                enabled = false;
                return;
            }
            _mat = new Material(shader);

            _mf = gameObject.AddComponent<MeshFilter>();
            var mr = gameObject.AddComponent<MeshRenderer>();
            mr.sharedMaterial = _mat;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = false;
        }

        private void Update()
        {
            if (!_built)
            {
                // 等 DepthCapture 把全局 gsDepthTexSize 写出来再建网格（尺寸取自真实深度纹理）
                Vector2 sz = Shader.GetGlobalVector(TexSizeID);
                if (sz.x < 8 || sz.y < 8) return;
                BuildGrid((int)sz.x, (int)sz.y);
            }
            if (_mat != null)
            {
                _mat.SetFloat(PointSizeID, pointSize);
                _mat.SetFloat(EyeIndexID, useRightEye ? 1f : 0f);
                _mat.SetFloat(BandWidthID, bandWidth);
            }
        }

        private void BuildGrid(int w, int h)
        {
            int step = Mathf.Max(1, Mathf.CeilToInt((float)Mathf.Max(w, h) / maxGridDim));
            int gw = w / step, gh = h / step;
            int count = gw * gh;

            var verts = new Vector3[count];
            var indices = new int[count];
            int i = 0;
            for (int y = 0; y < gh; y++)
                for (int x = 0; x < gw; x++)
                {
                    verts[i] = new Vector3(x * step, y * step, 0);
                    indices[i] = i;
                    i++;
                }

            _mesh = new Mesh { indexFormat = IndexFormat.UInt32 };
            _mesh.vertices = verts;
            _mesh.SetIndices(indices, MeshTopology.Points, 0);
            // 顶点真实位置由 shader 按深度反投影，包围盒给无限大防视锥剔除
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000f);
            _mf.sharedMesh = _mesh;
            _built = true;
            Logger.Info($"深度点云已建: {gw}x{gh}（源 {w}x{h}，抽稀 {step}x）");
        }

        private void OnDestroy()
        {
            if (_mesh != null) Destroy(_mesh);
            if (_mat != null) Destroy(_mat);
        }
    }
}

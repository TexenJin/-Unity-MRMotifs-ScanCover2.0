using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 融合前“枪胶裁决海面”只读显示：绿色是已取得稳定 ID 的候选中心，
    /// 红色是相对这些候选出现过、并按稳定 ID 锁存的最严重浪头。
    /// 不复制、不筛改、不写回任何生产缓冲；A 后由 GunGelEvidenceShadow
    /// 排空在途裁决并原地封存，所以两色均固定在采集时的世界坐标。
    /// </summary>
    public sealed class GunGelCourtOverlay : MonoBehaviour
    {
        [SerializeField, Range(1f, 12f), Tooltip("绿色稳定候选点尺寸（像素）")]
        private float calmPointSize = 4f;

        [SerializeField, Range(1f, 16f), Tooltip("红色锁存浪头点尺寸（像素）")]
        private float wavePointSize = 6f;

        private static readonly int CandidateCenterSigmaID =
            Shader.PropertyToID("_CandidateCenterSigma");
        private static readonly int CandidateMetaID = Shader.PropertyToID("_CandidateMeta");
        private static readonly int CandidateEvidenceID = Shader.PropertyToID("_CandidateEvidence");
        private static readonly int WavePeakBitsID = Shader.PropertyToID("_WavePeakBits");
        private static readonly int WavePositionResidualID =
            Shader.PropertyToID("_WavePositionResidual");
        private static readonly int WaveMetaID = Shader.PropertyToID("_WaveMeta");
        private static readonly int CourtModeID = Shader.PropertyToID("_CourtMode");
        private static readonly int PointSizeID = Shader.PropertyToID("_PointSize");
        private static readonly int PointColorID = Shader.PropertyToID("_PointColor");

        private Material _material;
        private MaterialPropertyBlock _properties;
        private bool _visible;
        private bool _sealed;
        private readonly Bounds _worldBounds = new(Vector3.zero, Vector3.one * 10000f);

        public bool Visible => _visible;
        public bool Sealed => _sealed;

        private void Awake()
        {
            Shader shader = Resources.Load<Shader>("GunGelCourtPoints");
            if (shader == null)
            {
                Logger.Warning("枪胶裁决海面 shader 缺失（GunGelCourtPoints），诊断层跳过");
                enabled = false;
                return;
            }

            _material = new Material(shader) { name = "QRS GunGel Court Points" };
            _properties = new MaterialPropertyBlock();
            _visible = false;
        }

        public void SetVisible(bool visible)
        {
            _visible = visible && enabled;
        }

        public void SetSealed(bool isSealed)
        {
            _sealed = isSealed;
        }

        private void LateUpdate()
        {
            if (!_visible || _material == null || _properties == null ||
                VolumeIntegrator.Instance == null ||
                !VolumeIntegrator.Instance.TryGetGunGelCourtBuffers(out var court) ||
                !court.IsValid)
                return;

            _properties.Clear();
            _properties.SetBuffer(CandidateCenterSigmaID, court.CandidateCenterSigma);
            _properties.SetBuffer(CandidateMetaID, court.CandidateMeta);
            _properties.SetBuffer(CandidateEvidenceID, court.CandidateEvidence);
            _properties.SetBuffer(WavePeakBitsID, court.WavePeakBits);
            _properties.SetBuffer(WavePositionResidualID, court.WavePositionResidual);
            _properties.SetBuffer(WaveMetaID, court.WaveMeta);

            var renderParams = new RenderParams(_material)
            {
                worldBounds = _worldBounds,
                matProps = _properties,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            // 静海：当前存活且已经取得稳定 ID 的候选。候选被正式淘汰后会
            // 从绿色集合消失；A 封存完成后缓冲不再变化。
            _properties.SetInt(CourtModeID, 0);
            _properties.SetFloat(PointSizeID, calmPointSize);
            _properties.SetColor(PointColorID, new Color(0.05f, 1f, 0.22f, 0.88f));
            Graphics.RenderPrimitives(renderParams, MeshTopology.Points,
                court.CandidateCount, 1);

            // 浪头：每个稳定 ID 在本轮扫描中最严重的一次异常观测。视觉
            // 一律红色，放行/暂存/拒绝的具体裁决写入 court_waves.csv。
            _properties.SetInt(CourtModeID, 1);
            _properties.SetFloat(PointSizeID, wavePointSize);
            _properties.SetColor(PointColorID, new Color(1f, 0.035f, 0.02f, 0.96f));
            Graphics.RenderPrimitives(renderParams, MeshTopology.Points,
                court.WaveCapacity, 1);
        }

        private void OnDestroy()
        {
            if (_material != null) Destroy(_material);
        }
    }
}

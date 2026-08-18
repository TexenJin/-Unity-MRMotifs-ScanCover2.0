using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Cheap acquisition-only coverage markers.  The angular visit ledger is
    /// retained for the HUD, but colour is authoritative TSDF readiness: yellow
    /// means the live depth point is not yet extractable and green means a nearby
    /// production Surface-Nets cell can currently emit.  Freeze always hides the
    /// markers before any replay mesh appears.
    /// </summary>
    public sealed class ObservationCoverageOverlay : MonoBehaviour
    {
        private const int YawBins = 16;
        private const int PitchBins = 5;
        private readonly byte[] _visits = new byte[YawBins * PitchBins];

        private Mesh _mesh;
        private Material _material;
        private MeshRenderer _renderer;
        private VolumeIntegrator _volume;
        private bool _acquiring;
        private bool _markersVisible = true;
        private float _nextObservation;
        private int _visited;

        public float CoveragePercent => 100f * _visited / _visits.Length;
        public int CurrentStrength { get; private set; }
        /// <summary>点阵显示开关（判官热切）。视角覆盖账本照常统计，不受显示影响。</summary>
        public bool MarkersVisible => _markersVisible;

        public void SetMarkersVisible(bool visible)
        {
            _markersVisible = visible;
            if (_renderer != null) _renderer.enabled = _acquiring && visible;
        }

        private void Awake()
        {
            _volume = VolumeIntegrator.Instance != null
                ? VolumeIntegrator.Instance
                : GetComponent<VolumeIntegrator>();
            BuildTiles();
            SetAcquiring(false);
        }

        private void Update()
        {
            RefreshReadinessBindings();
            if (!_acquiring || Time.unscaledTime < _nextObservation) return;
            _nextObservation = Time.unscaledTime + 0.20f;
            Camera cam = Camera.main;
            if (cam == null || DepthCapture.Instance == null || !DepthCapture.DepthAvailable)
                return;

            // The third integration intentionally clears the TSDF.  Do not let
            // those discarded startup frames advance the view-coverage ledger.
            if (_volume == null || _volume.Volume == null ||
                _volume.IntegrationCount <= _volume.WarmupIntegrations)
            {
                CurrentStrength = 0;
                return;
            }

            Vector3 forward = cam.transform.forward.normalized;
            float yaw = Mathf.Atan2(forward.x, forward.z) * Mathf.Rad2Deg;
            if (yaw < 0f) yaw += 360f;
            float pitch = Mathf.Asin(Mathf.Clamp(forward.y, -1f, 1f)) * Mathf.Rad2Deg;
            int x = Mathf.Clamp(Mathf.FloorToInt(yaw / 360f * YawBins), 0, YawBins - 1);
            int y = Mathf.Clamp(Mathf.FloorToInt(Mathf.InverseLerp(-55f, 55f, pitch) * PitchBins), 0, PitchBins - 1);
            int index = y * YawBins + x;
            if (_visits[index] == 0) _visited++;
            _visits[index] = (byte)Mathf.Min(3, _visits[index] + 1);
            CurrentStrength = _visits[index];
        }

        public void ResetCoverage()
        {
            System.Array.Clear(_visits, 0, _visits.Length);
            _visited = 0;
            CurrentStrength = 0;
            RefreshReadinessBindings();
        }

        public void SetAcquiring(bool acquiring)
        {
            _acquiring = acquiring;
            if (_renderer != null) _renderer.enabled = acquiring && _markersVisible;
            RefreshReadinessBindings();
        }

        private void RefreshReadinessBindings()
        {
            if (_material == null) return;
            if (_volume == null)
                _volume = VolumeIntegrator.Instance != null
                    ? VolumeIntegrator.Instance
                    : GetComponent<VolumeIntegrator>();

            bool ready = _acquiring && _volume != null && _volume.Volume != null &&
                         _volume.IntegrationCount > _volume.WarmupIntegrations;
            _material.SetFloat("_ReadinessEnabled", ready ? 1f : 0f);
            if (!ready) return;

            var count = _volume.VoxelCount;
            _material.SetTexture("_TsdfVolume", _volume.Volume);
            _material.SetVector("_VoxCount", new Vector4(count.x, count.y, count.z, 0f));
            _material.SetFloat("_VoxSize", _volume.VoxelSize);
            _material.SetFloat("_MinWeight", _volume.MinMeshWeight);
        }

        private void BuildTiles()
        {
            const int columns = 28;
            const int rows = 18;
            int quadCount = columns * rows;
            var vertices = new Vector3[quadCount * 4];
            var uv = new Vector2[quadCount * 4];
            var indices = new int[quadCount * 6];
            int q = 0;
            for (int y = 0; y < rows; y++)
            for (int x = 0; x < columns; x++, q++)
            {
                float px = (x + 0.5f) / columns;
                float py = (y + 0.5f) / rows;
                int v = q * 4;
                vertices[v + 0] = new Vector3(px, py, 0f);
                vertices[v + 1] = new Vector3(px, py, 0f);
                vertices[v + 2] = new Vector3(px, py, 0f);
                vertices[v + 3] = new Vector3(px, py, 0f);
                uv[v + 0] = new Vector2(-1f, -1f);
                uv[v + 1] = new Vector2( 1f, -1f);
                uv[v + 2] = new Vector2( 1f,  1f);
                uv[v + 3] = new Vector2(-1f,  1f);
                int i = q * 6;
                indices[i + 0] = v + 0; indices[i + 1] = v + 1; indices[i + 2] = v + 2;
                indices[i + 3] = v + 0; indices[i + 4] = v + 2; indices[i + 5] = v + 3;
            }

            _mesh = new Mesh { name = "QRS Observation Coverage Tiles" };
            _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            _mesh.vertices = vertices;
            _mesh.uv = uv;
            _mesh.triangles = indices;
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 100000f);

            var child = new GameObject("[QRS] Acquisition Coverage Tiles");
            child.transform.SetParent(transform, false);
            child.AddComponent<MeshFilter>().sharedMesh = _mesh;
            _renderer = child.AddComponent<MeshRenderer>();
            Shader shader = Resources.Load<Shader>("ObservationCoverageTiles");
            if (shader == null)
            {
                Logger.Error("缺少 ObservationCoverageTiles shader；采集覆盖片不可见");
                child.SetActive(false);
                return;
            }
            _material = new Material(shader) { name = "QRS Acquisition Coverage Tiles" };
            _renderer.sharedMaterial = _material;
            _renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _renderer.receiveShadows = false;
            RefreshReadinessBindings();
        }

        private void OnDestroy()
        {
            if (_mesh != null) Destroy(_mesh);
            if (_material != null) Destroy(_material);
        }
    }
}

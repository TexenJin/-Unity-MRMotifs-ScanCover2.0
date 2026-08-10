using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Renders the GPU Surface Nets mesh via Graphics.RenderPrimitivesIndirect.
    /// Replaces per-chunk MeshFilter+MeshRenderer with a single indirect draw call.
    /// </summary>
    internal class GPUMeshRenderer : MonoBehaviour
    {
        [SerializeField] private Material gpuMeshMaterial;

        private GPUSurfaceNets _surfaceNets;
        private MaterialPropertyBlock _props;
        private bool _ready;
        private Bounds _bounds;

        private static readonly int ID_SurfaceVerts = Shader.PropertyToID("_SurfaceVerts");
        private static readonly int ID_SurfaceIndices = Shader.PropertyToID("_SurfaceIndices");
        private static readonly int ID_VertexAdmissionClass = Shader.PropertyToID("_VertexAdmissionClass");
        private static readonly int ID_ExtractionColor = Shader.PropertyToID("_RSExtractionColor");
        private static readonly int ID_JointDiagnostic = Shader.PropertyToID("_RSJointDiagnostic");
        private static readonly int ID_SuppressPink = Shader.PropertyToID("_RSSuppressPink");
        private static readonly int ID_TemporalIllegalActive = Shader.PropertyToID("_RSTemporalIllegalActive");

        // Display-only A/B colors. They never feed back into extraction or TSDF state.
        private static readonly Color ProductionColor = new Color(1.0f, 0.62f, 0.02f, 0.96f);
        private static readonly Color StrictObservedColor = new Color(0.10f, 1.0f, 0.25f, 0.96f);
        private Color _extractionColor = ProductionColor;
        private bool _jointDiagnosticDisplay;
        private bool _suppressPink = true;
        private bool _temporalIllegalCandidateActive;

        private bool _renderVisible = true;

        /// <summary>
        /// Toggle rendering without disabling the component (which destroys state).
        /// </summary>
        public bool RenderVisible
        {
            get => _renderVisible;
            set => _renderVisible = value;
        }

        public Material GpuMeshMaterial
        {
            get => gpuMeshMaterial;
            set => gpuMeshMaterial = value;
        }

        public void SetStrictObservedDisplay(bool strictObserved)
        {
            _extractionColor = strictObserved ? StrictObservedColor : ProductionColor;
        }

        public void SetJointDiagnosticDisplay(bool enabled)
        {
            _jointDiagnosticDisplay = enabled;
        }

        public void SetTemporalIllegalCandidateActive(bool active)
        {
            _temporalIllegalCandidateActive = active;
        }

        /// <summary>
        /// Display-only quarantine for confirmation-only mixed triangles.
        /// The TSDF, admission trace and cumulative ledger remain untouched.
        /// </summary>
        public void SetPinkIsolation(bool enabled)
        {
            _suppressPink = enabled;
        }

        internal void Initialize(GPUSurfaceNets surfaceNets, Bounds volumeBounds)
        {
            _surfaceNets = surfaceNets;
            _bounds = volumeBounds;
            _props = new MaterialPropertyBlock();
            _ready = true;
        }

        public void UpdateBounds(Bounds bounds)
        {
            _bounds = bounds;
        }

        private void LateUpdate()
        {
            if (!_ready || !_renderVisible || _surfaceNets == null || gpuMeshMaterial == null)
                return;

            var vertBuf = _surfaceNets.VertexBuffer;
            var idxBuf = _surfaceNets.IndexBuffer;
            var admissionBuf = _surfaceNets.VertexAdmissionClassBuffer;
            var argsBuf = _surfaceNets.DrawIndirectArgs;

            if (vertBuf == null || idxBuf == null || admissionBuf == null || argsBuf == null)
                return;

            _props.SetBuffer(ID_SurfaceVerts, vertBuf);
            _props.SetBuffer(ID_SurfaceIndices, idxBuf);
            _props.SetBuffer(ID_VertexAdmissionClass, admissionBuf);
            _props.SetColor(ID_ExtractionColor, _extractionColor);
            _props.SetFloat(ID_JointDiagnostic, _jointDiagnosticDisplay ? 1f : 0f);
            _props.SetFloat(ID_SuppressPink, _suppressPink ? 1f : 0f);
            _props.SetFloat(ID_TemporalIllegalActive, _temporalIllegalCandidateActive ? 1f : 0f);

            var rp = new RenderParams(gpuMeshMaterial)
            {
                worldBounds = _bounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles, argsBuf, 1);
        }

        private void OnDisable()
        {
            _ready = false;
        }
    }
}

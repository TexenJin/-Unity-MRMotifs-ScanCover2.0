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

        private IGPUMeshBufferSource _meshSource;
        private MaterialPropertyBlock _props;
        private bool _ready;
        private Bounds _bounds;

        public int LastSubmittedVertexCount { get; private set; }
        public int LastSubmittedFrame { get; private set; } = -1;

        private static readonly int ID_SurfaceVerts = Shader.PropertyToID("_SurfaceVerts");
        private static readonly int ID_SurfaceIndices = Shader.PropertyToID("_SurfaceIndices");
        private static readonly int ID_VertexAdmissionClass = Shader.PropertyToID("_VertexAdmissionClass");
        private static readonly int ID_ExtractionColor = Shader.PropertyToID("_RSExtractionColor");
        private static readonly int ID_JointDiagnostic = Shader.PropertyToID("_RSJointDiagnostic");
        private static readonly int ID_SuppressPink = Shader.PropertyToID("_RSSuppressPink");
        private static readonly int ID_TemporalIllegalActive = Shader.PropertyToID("_RSTemporalIllegalActive");
        private static readonly int ID_HeraReplayActive = Shader.PropertyToID("_RSHeraReplayActive");

        // Display-only A/B colors. They never feed back into extraction or TSDF state.
        private static readonly Color ProductionColor = new Color(1.0f, 0.62f, 0.02f, 0.96f);
        private static readonly Color StrictObservedColor = new Color(0.10f, 1.0f, 0.25f, 0.96f);
        private static readonly Color ReplayGoodColor = new Color(0.12f, 1.0f, 0.28f, 0.96f);
        private static readonly Color ReplayBadColor = new Color(1.0f, 0.18f, 0.32f, 0.96f);
        private static readonly Color HeraLocalAcceptedPatchColor = new Color(0.18f, 1.0f, 0.42f, 0.98f);
        private Color _extractionColor = ProductionColor;
        private bool _jointDiagnosticDisplay;
        private bool _suppressPink = true;
        private bool _temporalIllegalCandidateActive;
        private bool _heraReplayActive;

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

        /// <summary>
        /// Frozen replay page state. -1 = ordinary single-color, 1 = clean
        /// page, 2 = page containing questionable triangles. Empty pages have
        /// no indices and are therefore represented by absence, not a fake box.
        /// </summary>
        public void SetReplayPageState(int pageClass)
        {
            _heraReplayActive = false;
            _jointDiagnosticDisplay = false;
            _extractionColor = pageClass == 1
                ? ReplayGoodColor
                : pageClass == 2
                    ? ReplayBadColor
                    : ProductionColor;
        }

        /// <summary>
        /// HERA replay is encoded per triangle: resident accepted triangles are
        /// green and fine-tier requests red. Both routes always remain visible;
        /// colour is descriptive and has no admission authority.
        /// </summary>
        public void SetHeraReplayDisplay(bool _)
        {
            _heraReplayActive = true;
            _jointDiagnosticDisplay = false;
            _extractionColor = ReplayGoodColor;
        }

        /// <summary>
        /// Exact child16 triangles that passed the local interior false-conflict
        /// classifier while their parent32 family stayed resident.  This is a
        /// committed local patch and is always shown as accepted green.
        /// </summary>
        public void SetHeraLocalAcceptedPatchDisplay(bool _)
        {
            // Local child16 rescue triangles are accepted additions and use
            // route class 2, so the same dual-colour shader renders them green.
            _heraReplayActive = true;
            _jointDiagnosticDisplay = false;
            _extractionColor = HeraLocalAcceptedPatchColor;
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

        internal void Initialize(IGPUMeshBufferSource meshSource, Bounds volumeBounds)
        {
            _meshSource = meshSource;
            _bounds = volumeBounds;
            _props = new MaterialPropertyBlock();
            _ready = true;
        }

        internal void SetMeshSource(IGPUMeshBufferSource meshSource)
        {
            _meshSource = meshSource;
            // Chunk renderers are intentionally disabled while pages are being
            // recycled.  OnDisable clears _ready; restoring a valid immutable
            // front buffer must make the renderer drawable again.
            if (_meshSource != null && _props != null)
                _ready = true;
        }

        public void UpdateBounds(Bounds bounds)
        {
            _bounds = bounds;
        }

        private void LateUpdate()
        {
            LastSubmittedVertexCount = 0;
            if (!_ready || !_renderVisible || _meshSource == null || gpuMeshMaterial == null)
                return;

            var vertBuf = _meshSource.VertexBuffer;
            var idxBuf = _meshSource.IndexBuffer;
            var admissionBuf = _meshSource.VertexAdmissionClassBuffer;
            var argsBuf = _meshSource.DrawIndirectArgs;
            int knownDrawVertexCount = _meshSource.KnownDrawVertexCount;

            if (vertBuf == null || idxBuf == null || admissionBuf == null ||
                (knownDrawVertexCount < 0 && argsBuf == null))
                return;

            if (knownDrawVertexCount == 0)
                return;

            _props.SetBuffer(ID_SurfaceVerts, vertBuf);
            _props.SetBuffer(ID_SurfaceIndices, idxBuf);
            _props.SetBuffer(ID_VertexAdmissionClass, admissionBuf);
            _props.SetColor(ID_ExtractionColor, _extractionColor);
            _props.SetFloat(ID_JointDiagnostic, _jointDiagnosticDisplay ? 1f : 0f);
            _props.SetFloat(ID_SuppressPink, _suppressPink ? 1f : 0f);
            _props.SetFloat(ID_TemporalIllegalActive, _temporalIllegalCandidateActive ? 1f : 0f);
            _props.SetFloat(ID_HeraReplayActive, _heraReplayActive ? 1f : 0f);

            var rp = new RenderParams(gpuMeshMaterial)
            {
                worldBounds = _bounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            if (knownDrawVertexCount > 0)
            {
                // Immutable chunk/HERA snapshots already completed an async GPU
                // readback, so their exact index count is authoritative on the
                // CPU.  Draw them directly instead of reinterpreting the old
                // five-uint argument buffer as platform-specific IndirectDrawArgs.
                // This keeps parent32 visibility independent from Vulkan's
                // indirect-command layout while preserving the same SV_VertexID
                // index fetch used by the shader.
                Graphics.RenderPrimitives(rp, MeshTopology.Triangles, knownDrawVertexCount, 1);
                LastSubmittedVertexCount = knownDrawVertexCount;
            }
            else
            {
                Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles, argsBuf, 1);
                LastSubmittedVertexCount = -1;
            }
            LastSubmittedFrame = Time.frameCount;
        }

        private void OnDisable()
        {
            _ready = false;
        }

        private void OnEnable()
        {
            if (_meshSource != null && _props != null)
                _ready = true;
        }
    }
}

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
        [SerializeField, Tooltip("线框显示前先用无色深度壳遮住后层网格。只影响显示，不修改 TSDF、顶点或索引。")]
        private bool occludeRearWireframe = true;

        // Display-only A/B gate shared by every live mesh page.  HERA and the
        // block pipelines create many GPUMeshRenderer instances over time, so
        // an instance-local runtime switch would leave old/new pages in mixed
        // states and make the GPU comparison invalid.
        private static bool s_rearWireDepthPrepassEnabled = true;
        private static bool s_trueLinePrimitivesEnabled = true;

        private IGPUMeshBufferSource _meshSource;
        private MaterialPropertyBlock _props;
        private Material _rearWireOccluderMaterial;
        private Material _trueLineMaterial;
        private bool _ready;
        private Bounds _bounds;

        public int LastSubmittedVertexCount { get; private set; }
        public int LastSubmittedFrame { get; private set; } = -1;

        public static bool RearWireDepthPrepassEnabled =>
            s_rearWireDepthPrepassEnabled;

        public static bool TrueLinePrimitivesEnabled =>
            s_trueLinePrimitivesEnabled;

        public static void SetRearWireDepthPrepassEnabled(bool enabled)
        {
            s_rearWireDepthPrepassEnabled = enabled;
        }

        public static bool ToggleRearWireDepthPrepass()
        {
            s_rearWireDepthPrepassEnabled = !s_rearWireDepthPrepassEnabled;
            return s_rearWireDepthPrepassEnabled;
        }

        public static void SetTrueLinePrimitivesEnabled(bool enabled)
        {
            s_trueLinePrimitivesEnabled = enabled;
        }

        public static bool ToggleTrueLinePrimitives()
        {
            s_trueLinePrimitivesEnabled = !s_trueLinePrimitivesEnabled;
            return s_trueLinePrimitivesEnabled;
        }

        private static readonly int ID_SurfaceVerts = Shader.PropertyToID("_SurfaceVerts");
        private static readonly int ID_SurfaceIndices = Shader.PropertyToID("_SurfaceIndices");
        private static readonly int ID_VertexAdmissionClass = Shader.PropertyToID("_VertexAdmissionClass");
        private static readonly int ID_ExtractionColor = Shader.PropertyToID("_RSExtractionColor");
        private static readonly int ID_JointDiagnostic = Shader.PropertyToID("_RSJointDiagnostic");
        private static readonly int ID_SuppressPink = Shader.PropertyToID("_RSSuppressPink");
        private static readonly int ID_TemporalIllegalActive = Shader.PropertyToID("_RSTemporalIllegalActive");
        private static readonly int ID_HeraReplayActive = Shader.PropertyToID("_RSHeraReplayActive");
        private static readonly int ID_ProductGridMode = Shader.PropertyToID("_RSProductGridMode");
        private static readonly int ID_Wireframe = Shader.PropertyToID("_RSWireframe");
        private static readonly int ID_PaperGridMode = Shader.PropertyToID("_RSPaperGridMode");
        private static readonly int ID_TrueLineQuadPerimeters =
            Shader.PropertyToID("_RSTrueLineQuadPerimeters");
        private static readonly int ID_TrueLineMaxViewDistance =
            Shader.PropertyToID("_RSTrueLineMaxViewDistance");
        private static readonly int ID_DepthPrepassMaxViewDistance =
            Shader.PropertyToID("_RSDepthPrepassMaxViewDistance");

        // Keep the CPU block rejection conservative for stereo eyes and for
        // blocks touching the display-range boundary. The depth shader applies
        // the exact per-vertex range to the surviving boundary blocks.
        private const float DepthPrepassBoundsCullPadding = 0.15f;
        private static Camera s_mainCamera;

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
        private bool _productGridMode;
        private bool _trueLineQuadPerimeters;

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
        /// Draws a world-anchored 10 cm grid over the complete native candidate
        /// surface. This changes no indices and therefore cannot introduce
        /// holes, slivers or cross-page topology damage.
        /// </summary>
        public void SetProductGridDisplay(bool enabled)
        {
            _productGridMode = enabled;
        }

        /// <summary>
        /// The compact InfiniTAM front emits each native surface quad as two
        /// consecutive triangles. Draw its four perimeter edges without the
        /// internal split diagonal; immutable mesh data remains unchanged.
        /// </summary>
        public void SetTrueLineQuadPerimeters(bool enabled)
        {
            _trueLineQuadPerimeters = enabled;
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
            var lineArgsBuf = _meshSource.LineDrawIndirectArgs;
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
            _props.SetFloat(ID_ProductGridMode, _productGridMode ? 1f : 0f);
            bool useQuadPerimeters = _trueLineQuadPerimeters &&
                knownDrawVertexCount >= 6 && knownDrawVertexCount % 6 == 0;
            _props.SetFloat(ID_TrueLineQuadPerimeters,
                useQuadPerimeters ? 1f : 0f);
            var rp = new RenderParams(gpuMeshMaterial)
            {
                worldBounds = _bounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            bool lineSurfaceVisible =
                Shader.GetGlobalFloat(ID_Wireframe) > 0.5f ||
                Shader.GetGlobalFloat(ID_PaperGridMode) > 0.5f ||
                _productGridMode;
            bool ordinaryWireframe =
                Shader.GetGlobalFloat(ID_Wireframe) > 0.5f &&
                Shader.GetGlobalFloat(ID_PaperGridMode) <= 0.5f &&
                !_productGridMode;
            bool useTrueLines =
                s_trueLinePrimitivesEnabled &&
                ordinaryWireframe &&
                ((knownDrawVertexCount > 0 &&
                  knownDrawVertexCount <= int.MaxValue / 2) ||
                  (knownDrawVertexCount < 0 && lineArgsBuf != null)) &&
                EnsureTrueLineMaterial();
            float depthPrepassMaxDistance = useTrueLines
                ? Mathf.Max(0f, Shader.GetGlobalFloat(ID_TrueLineMaxViewDistance))
                : 0f;
            _props.SetFloat(ID_DepthPrepassMaxViewDistance,
                depthPrepassMaxDistance);
            if (occludeRearWireframe && s_rearWireDepthPrepassEnabled &&
                lineSurfaceVisible &&
                !IsEntirelyOutsideDepthPrepassRange(depthPrepassMaxDistance) &&
                EnsureRearWireOccluderMaterial())
            {
                // Fill only the depth buffer with the nearest complete triangle
                // surface.  The passthrough colour remains untouched, while the
                // following wire pass can no longer reveal displaced sheets and
                // long triangles behind that nearest surface through every cell.
                var depthRp = new RenderParams(_rearWireOccluderMaterial)
                {
                    worldBounds = _bounds,
                    matProps = _props,
                    receiveShadows = false,
                    shadowCastingMode = ShadowCastingMode.Off,
                    layer = gameObject.layer
                };
                Submit(depthRp, knownDrawVertexCount, argsBuf);
            }

            if (useTrueLines)
            {
                var lineRp = new RenderParams(_trueLineMaterial)
                {
                    worldBounds = _bounds,
                    matProps = _props,
                    receiveShadows = false,
                    shadowCastingMode = ShadowCastingMode.Off,
                    layer = gameObject.layer
                };

                // Ordinary snapshots expose all three triangle edges. The compact
                // InfiniTAM front has a stronger contract: every six indices are
                // one quad, so eight line vertices cover its perimeter instead of
                // twelve vertices drawing the split diagonal twice. Both routes
                // avoid rasterizing triangle interiors.
                if (knownDrawVertexCount > 0)
                {
                    int lineVertexCount = useQuadPerimeters
                        ? knownDrawVertexCount / 6 * 8
                        : knownDrawVertexCount * 2;
                    Graphics.RenderPrimitives(
                        lineRp,
                        MeshTopology.Lines,
                        lineVertexCount,
                        1);
                }
                else
                {
                    Graphics.RenderPrimitivesIndirect(
                        lineRp,
                        MeshTopology.Lines,
                        lineArgsBuf,
                        1);
                }
            }
            else
            {
                // Unknown-count indirect buffers, paper/product grids and the
                // explicit A/B fallback retain the proven triangle path.
                Submit(rp, knownDrawVertexCount, argsBuf);
            }
            LastSubmittedVertexCount = knownDrawVertexCount > 0 ? knownDrawVertexCount : -1;
            LastSubmittedFrame = Time.frameCount;
        }

        private bool IsEntirelyOutsideDepthPrepassRange(float maxDistance)
        {
            if (maxDistance <= 0f)
                return false;

            if (s_mainCamera == null)
                s_mainCamera = Camera.main;
            if (s_mainCamera == null)
                return false;

            float conservativeDistance = maxDistance +
                                         DepthPrepassBoundsCullPadding;
            return _bounds.SqrDistance(s_mainCamera.transform.position) >
                   conservativeDistance * conservativeDistance;
        }

        private bool EnsureTrueLineMaterial()
        {
            if (_trueLineMaterial != null)
                return true;

            Shader shader = Resources.Load<Shader>("ScanMeshTrueLines");
            if (shader == null)
                return false;

            _trueLineMaterial = new Material(shader)
            {
                name = "[QRS] True Mesh Lines"
            };
            return true;
        }

        private bool EnsureRearWireOccluderMaterial()
        {
            if (_rearWireOccluderMaterial != null)
                return true;

            Shader shader = Resources.Load<Shader>("ScanMeshDepthOccluder");
            if (shader == null)
                return false;

            _rearWireOccluderMaterial = new Material(shader)
            {
                name = "[QRS] Rear Wire Occluder"
            };
            return true;
        }

        private static void Submit(RenderParams rp, int knownDrawVertexCount, GraphicsBuffer argsBuf)
        {
            if (knownDrawVertexCount > 0)
            {
                // Immutable chunk/HERA snapshots already completed an async GPU
                // readback, so their exact index count is authoritative on the
                // CPU. Draw them directly instead of reinterpreting the old
                // five-uint argument buffer as platform-specific IndirectDrawArgs.
                Graphics.RenderPrimitives(rp, MeshTopology.Triangles, knownDrawVertexCount, 1);
            }
            else
            {
                Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles, argsBuf, 1);
            }
        }

        private void OnDisable()
        {
            _ready = false;
        }

        private void OnDestroy()
        {
            if (_rearWireOccluderMaterial != null)
                Destroy(_rearWireOccluderMaterial);
            _rearWireOccluderMaterial = null;
            if (_trueLineMaterial != null)
                Destroy(_trueLineMaterial);
            _trueLineMaterial = null;
        }

        private void OnEnable()
        {
            if (_meshSource != null && _props != null)
                _ready = true;
        }
    }
}

using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Manual, bounded A/B probe: a small world-anchored 5 cm TSDF is fused
    /// directly from the current depth frame and rendered with its native
    /// 5 cm connectivity in cyan over the unchanged global 10 cm reconstruction.
    /// It reuses the production compact-history/view-ticket rules, but owns no
    /// smoothing, QEF, feature labels or 10 cm parent simplification.
    /// </summary>
    public sealed class LocalFineTsdfExperiment : MonoBehaviour
    {
        public static LocalFineTsdfExperiment Instance { get; private set; }

        private const float VoxelSize = 0.05f;
        private const float Truncation = 0.15f;
        private const float MinDepth = 0.5f;
        private const float MaxDepth = 3.5f;
        private const float ContradictionDistanceVoxels = 0.5f;
        private const float IndependentViewDegrees = 12f;
        private const float ExtractInterval = 0.20f;
        private const float PreviewSampleInterval = 0.20f;
        private const float PreviewDepthFreshSeconds = 0.80f;
        private static readonly int3 VoxelCount = new int3(16, 24, 16);
        private static readonly Color PreviewColor = new Color(1f, 0.56f, 0.04f, 0.98f);
        private static readonly Color LockedColor = new Color(0.10f, 1f, 0.42f, 0.98f);
        private static readonly Color FineMeshColor = new Color(0.02f, 0.95f, 1f, 1f);
        private static readonly int[,] FrameEdgePairs =
        {
            { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 },
            { 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 },
            { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
        };

        private static readonly int FineTsdfID = Shader.PropertyToID("_FineTsdfRW");
        private static readonly int FineHistoryID = Shader.PropertyToID("_FineHistoryRW");
        private static readonly int FineCoherenceID = Shader.PropertyToID("_FineCoherenceRW");
        private static readonly int FineCandidateID = Shader.PropertyToID("_FineCandidateRW");
        private static readonly int FineColorID = Shader.PropertyToID("_FineColorRW");
        private static readonly int FineTraceID = Shader.PropertyToID("_FineTraceRW");
        private static readonly int FineVoxCountID = Shader.PropertyToID("_FineVoxCount");
        private static readonly int FineCenterID = Shader.PropertyToID("_FineCenter");
        private static readonly int FineVoxSizeID = Shader.PropertyToID("_FineVoxSize");
        private static readonly int FineTruncationID = Shader.PropertyToID("_FineTruncation");
        private static readonly int FineMinDepthID = Shader.PropertyToID("_FineMinDepth");
        private static readonly int FineMaxDepthID = Shader.PropertyToID("_FineMaxDepth");
        private static readonly int FineContradictionThresholdID =
            Shader.PropertyToID("_FineContradictionThreshold");
        private static readonly int FineIndependentViewCosID =
            Shader.PropertyToID("_FineIndependentViewCos");

        private ComputeShader _integrator;
        private int _clearKernel = -1;
        private int _clearCandidatesKernel = -1;
        private int _integrateKernel = -1;
        private int _evaluateEdgeKernel = -1;
        private int _commitEdgeKernel = -1;
        private RenderTexture _tsdf;
        private RenderTexture _history;
        private RenderTexture _coherence;
        private RenderTexture _candidate;
        private RenderTexture _color;
        private RenderTexture _trace;
        private GPUSurfaceNets _surface;
        private GPUMeshRenderer _renderer;
        private GameObject _meshObject;
        private GameObject _frameRoot;
        private LineRenderer[] _frameEdges;
        private Material _frameMaterial;
        private Material _comparisonMaterial;
        private readonly Vector3[] _frameCorners = new Vector3[8];
        private bool _initialized;
        private bool _locked;
        private Vector3 _center;
        private float _placementDistance = 1.6f;
        private bool _autoDepthReady;
        private int _lastCenterDepthRevision = -1;
        private float _lastAutoDepthTime = -100f;
        private float _nextPreviewSampleTime;
        private int _lastIntegratedPlatformFrame = -1;
        private int _acceptedFrames;
        private float _nextExtractionTime;

        public bool IsLocked => _locked;
        private bool IsOperatorFrozen => StandaloneRoomScanner.Instance != null &&
                                         StandaloneRoomScanner.Instance.IsOperatorFrozen;
        public string StatusCompact => !_initialized
            ? "5cm框:未就绪"
            : !_locked
                ? _autoDepthReady
                    ? $"5cm框:<color=#FF9A20>自动瞄准</color> {_placementDistance:F1}m 盯住目标/按右摇杆锁定"
                    : "5cm框:<color=#FFB020>等待中心深度</color> 盯住目标即可"
                : IsOperatorFrozen
                    ? $"5cm框:<color=#38DFFF>A定格</color> 帧{_acceptedFrames} 青线/历史均保持"
                    : $"5cm框:<color=#38FFC0>原生稳定采集</color> 帧{_acceptedFrames} 青线=原生5cm(视角票/16票/相干发布)";

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
                return;
            }
            Instance = this;
        }

        public bool Initialize(ComputeShader surfaceNetsCompute,
            Material meshMaterial, int layer)
        {
            if (_initialized) return true;
            if (surfaceNetsCompute == null || meshMaterial == null) return false;

            _integrator = Resources.Load<ComputeShader>("LocalFineTsdfExperiment");
            if (_integrator == null)
            {
                Debug.LogError("[QRS] LocalFineTsdfExperiment.compute not found in Resources");
                return false;
            }

            try
            {
                _clearKernel = _integrator.FindKernel("ClearFineVolume");
                _clearCandidatesKernel = _integrator.FindKernel("ClearFineCandidates");
                _integrateKernel = _integrator.FindKernel("IntegrateFineVolume");
                _evaluateEdgeKernel = _integrator.FindKernel("EvaluateFineEdgeCoherence");
                _commitEdgeKernel = _integrator.FindKernel("CommitFineEdgeCoherence");
                CreateVolumes();

                _surface = new GPUSurfaceNets(surfaceNetsCompute)
                {
                    MinMeshWeight = 0.08f,
                    SmoothIterations = 0,
                    SmoothLambda = 0f,
                    SmoothBeta = 0f,
                    TemporalAlphaMax = 1f,
                    TemporalAlphaMin = 1f,
                    TemporalDecayRate = 0f,
                    ConvergenceThreshold = 0f,
                    TemporalDeadzone = 0f,
                    StrictObservedEdges = true,
                    CandidateHistoryUpdateEnabled = false,
                    // The first validation stage must expose the actual fine
                    // surface.  No 2x2x2 parent may erase multi-sheet evidence.
                    FoundationTopologyMode = false,
                    FoundationCellStride = 1,
                    FoundationConstrainedSimplification = false,
                    FoundationParentTopologyFromFineEvidence = false,
                    FoundationParentStabilityEnabled = false,
                    FoundationSimplifyNormalDotMin = 0.9659258f,
                    FoundationSimplifyPlaneResidualVoxels = 0.2f,
                    VisualQualityDiagnosticsEnabled = false,
                    DiagnosticRoiEnabled = false
                };
                // The ROI is tiny; allocate one vertex slot per cell so a
                // cluttered manually framed region cannot silently overflow.
                _surface.EnsureBuffers(VoxelCount, 1f);

                _meshObject = new GameObject("[QRS] Local Native Stable 5cm Mesh");
                _meshObject.layer = layer;
                _meshObject.transform.SetParent(transform, false);
                _renderer = _meshObject.AddComponent<GPUMeshRenderer>();
                // Keep exact vertex positions. A slightly later transparent
                // queue makes coincident cyan lines readable without nudging
                // the comparison geometry toward the camera.
                _comparisonMaterial = new Material(meshMaterial)
                {
                    name = "[QRS] Native Stable 5cm Comparison",
                    // Draw after the production shell.  This is an observation
                    // overlay, so it must not disappear behind the material it
                    // is being compared against.
                    renderQueue = Mathf.Max(meshMaterial.renderQueue + 8,
                        (int)RenderQueue.Transparent + 80)
                };
                _comparisonMaterial.SetInt("_RSMeshZWrite", 0);
                _comparisonMaterial.SetInt("_RSMeshZTest",
                    (int)CompareFunction.Always);
                _renderer.GpuMeshMaterial = _comparisonMaterial;
                _renderer.Initialize(_surface, new Bounds(Vector3.zero,
                    (Vector3)((float3)VoxelCount * VoxelSize)));
                _renderer.SetComparisonDisplay(FineMeshColor);
                _renderer.SetJointDiagnosticDisplay(false);
                _renderer.SetTemporalIllegalCandidateActive(false);
                _renderer.SetPinkIsolation(true);
                _renderer.SetRearWireOcclusion(false);
                _renderer.RenderVisible = false;

                CreateFrame(layer);
                _frameRoot.SetActive(false);
                ClearFineVolume();
                _initialized = true;
                Debug.Log("[QRS] Manual native 5 cm ROI ready: compact history, view tickets, coherent publication, no 10 cm parent collapse");
                return true;
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[QRS] Manual 5 cm ROI initialization failed: {ex.Message}");
                DisposeResources();
                return false;
            }
        }

        public string ToggleLock()
        {
            if (!_initialized) return "5cm实验框尚未就绪";
            if (_locked)
            {
                _locked = false;
                _autoDepthReady = false;
                _lastAutoDepthTime = -100f;
                _renderer.RenderVisible = false;
                ClearFineVolume();
                if (_frameRoot != null) _frameRoot.SetActive(false);
                UpdateFrameColor();
                return "5cm实验框已取消，请重新瞄准";
            }

            RefreshAutoPreview(Camera.main, true);
            if (!_autoDepthReady)
                return "中心暂无有效深度，5cm框未锁定";

            UpdatePreviewCenter(Camera.main);
            _locked = true;
            if (_frameRoot != null) _frameRoot.SetActive(true);
            ApplyPlacement();
            ClearFineVolume();
            UpdateFrameColor();
            return $"5cm实验框已自动锁定 {_placementDistance:F1}m 并清零";
        }

        public void ResetForNewScan()
        {
            if (!_initialized) return;
            ClearFineVolume();
            _renderer.RenderVisible = false;
        }

        private void LateUpdate()
        {
            if (!_initialized) return;

            if (!_locked)
            {
                RefreshAutoPreview(Camera.main, false);
                return;
            }

            StandaloneRoomScanner scanner = StandaloneRoomScanner.Instance;
            DepthCapture depth = DepthCapture.Instance;
            if (scanner == null || !scanner.IsScanning || scanner.IsOperatorFrozen ||
                depth == null ||
                !DepthCapture.DepthAvailable || depth.DepthTex == null)
                return;

            int platformFrame = depth.CurrentPlatformFrame;
            if (platformFrame == _lastIntegratedPlatformFrame) return;
            _lastIntegratedPlatformFrame = platformFrame;

            DispatchIntegration(depth);
            _acceptedFrames++;

            if (Time.unscaledTime < _nextExtractionTime || _acceptedFrames < 2)
                return;
            _nextExtractionTime = Time.unscaledTime + ExtractInterval;

            _surface.VolumeOrigin = _center;
            _surface.Extract(_tsdf, _color, _trace, VoxelSize,
                null, null, false);
            _renderer.UpdateBounds(_surface.GetVolumeBounds(VoxelSize));
            _renderer.RenderVisible = true;
        }

        private void DispatchIntegration(DepthCapture depth)
        {
            _integrator.SetInts(FineVoxCountID,
                VoxelCount.x, VoxelCount.y, VoxelCount.z);
            _integrator.SetVector(FineCenterID, _center);
            _integrator.SetFloat(FineVoxSizeID, VoxelSize);
            _integrator.SetFloat(FineTruncationID, Truncation);
            _integrator.SetFloat(FineMinDepthID, MinDepth);
            _integrator.SetFloat(FineMaxDepthID, MaxDepth);
            _integrator.SetFloat(FineContradictionThresholdID,
                Mathf.Clamp(ContradictionDistanceVoxels * VoxelSize /
                            Mathf.Max(Truncation, 1e-5f), 1f / 127f, 1f));
            _integrator.SetFloat(FineIndependentViewCosID,
                Mathf.Cos(IndependentViewDegrees * Mathf.Deg2Rad));
            _integrator.SetInts(DepthCapture.TexSizeID,
                depth.DepthTex.width, depth.DepthTex.height);
            _integrator.SetMatrixArray(DepthCapture.ProjID,
                depth.DepthProjectionMatrices);
            _integrator.SetMatrixArray(DepthCapture.ProjInvID,
                depth.DepthProjectionInverseMatrices);
            _integrator.SetMatrixArray(DepthCapture.ViewID,
                depth.DepthViewMatrices);
            _integrator.SetMatrixArray(DepthCapture.ViewInvID,
                depth.DepthViewInverseMatrices);

            _integrator.SetTexture(_clearCandidatesKernel, FineCandidateID,
                _candidate);
            _integrator.Dispatch(_clearCandidatesKernel,
                CeilDiv(VoxelCount.x, 4), CeilDiv(VoxelCount.y, 4),
                CeilDiv(VoxelCount.z, 4));

            _integrator.SetTexture(_integrateKernel, DepthCapture.DepthTexID,
                depth.DepthTex);
            BindFineState(_integrateKernel);
            _integrator.Dispatch(_integrateKernel,
                CeilDiv(VoxelCount.x, 4), CeilDiv(VoxelCount.y, 4),
                CeilDiv(VoxelCount.z, 4));

            _integrator.SetTexture(_evaluateEdgeKernel, FineTsdfID, _tsdf);
            _integrator.SetTexture(_evaluateEdgeKernel, FineCandidateID,
                _candidate);
            _integrator.Dispatch(_evaluateEdgeKernel,
                CeilDiv(VoxelCount.x, 4), CeilDiv(VoxelCount.y, 4),
                CeilDiv(VoxelCount.z, 4));

            _integrator.SetTexture(_commitEdgeKernel, FineTsdfID, _tsdf);
            _integrator.SetTexture(_commitEdgeKernel, FineHistoryID, _history);
            _integrator.SetTexture(_commitEdgeKernel, FineCandidateID,
                _candidate);
            _integrator.Dispatch(_commitEdgeKernel,
                CeilDiv(VoxelCount.x, 4), CeilDiv(VoxelCount.y, 4),
                CeilDiv(VoxelCount.z, 4));
        }

        private void BindFineState(int kernel)
        {
            _integrator.SetTexture(kernel, FineTsdfID, _tsdf);
            _integrator.SetTexture(kernel, FineHistoryID, _history);
            _integrator.SetTexture(kernel, FineCoherenceID, _coherence);
            _integrator.SetTexture(kernel, FineCandidateID, _candidate);
        }

        private void ClearFineVolume()
        {
            if (_integrator == null || _tsdf == null) return;
            _integrator.SetInts(FineVoxCountID,
                VoxelCount.x, VoxelCount.y, VoxelCount.z);
            BindFineState(_clearKernel);
            _integrator.SetTexture(_clearKernel, FineColorID, _color);
            _integrator.SetTexture(_clearKernel, FineTraceID, _trace);
            _integrator.Dispatch(_clearKernel,
                CeilDiv(VoxelCount.x, 4), CeilDiv(VoxelCount.y, 4),
                CeilDiv(VoxelCount.z, 4));
            _surface?.ResetTemporalState();
            _lastIntegratedPlatformFrame = -1;
            _acceptedFrames = 0;
            _nextExtractionTime = 0f;
        }

        private void CreateVolumes()
        {
            _tsdf = CreateVolume("LocalFine5cmTSDF", GraphicsFormat.R8G8_SNorm,
                FilterMode.Bilinear);
            GraphicsFormat historyFormat = SystemInfo.IsFormatSupported(
                GraphicsFormat.R16G16B16A16_SFloat, FormatUsage.LoadStore)
                ? GraphicsFormat.R16G16B16A16_SFloat
                : GraphicsFormat.R32G32B32A32_SFloat;
            GraphicsFormat candidateFormat = SystemInfo.IsFormatSupported(
                GraphicsFormat.R16G16_SFloat, FormatUsage.LoadStore)
                ? GraphicsFormat.R16G16_SFloat
                : GraphicsFormat.R32G32_SFloat;
            GraphicsFormat scalarFormat = SystemInfo.IsFormatSupported(
                GraphicsFormat.R16_SFloat, FormatUsage.LoadStore)
                ? GraphicsFormat.R16_SFloat
                : GraphicsFormat.R32_SFloat;
            _history = CreateVolume("LocalFine5cmHistory", historyFormat,
                FilterMode.Point);
            _coherence = CreateVolume("LocalFine5cmCoherence", scalarFormat,
                FilterMode.Point);
            _candidate = CreateVolume("LocalFine5cmCandidate", candidateFormat,
                FilterMode.Point);
            _color = CreateVolume("LocalFine5cmColor", GraphicsFormat.R8G8B8A8_UNorm,
                FilterMode.Point);
            GraphicsFormat traceFormat = SystemInfo.IsFormatSupported(
                GraphicsFormat.R8_UNorm, FormatUsage.LoadStore)
                ? GraphicsFormat.R8_UNorm
                : GraphicsFormat.R16_SFloat;
            _trace = CreateVolume("LocalFine5cmTrace", traceFormat,
                FilterMode.Point);
        }

        private static RenderTexture CreateVolume(string name, GraphicsFormat format,
            FilterMode filter)
        {
            var texture = new RenderTexture(VoxelCount.x, VoxelCount.y, 0,
                format, 0)
            {
                name = name,
                dimension = TextureDimension.Tex3D,
                volumeDepth = VoxelCount.z,
                enableRandomWrite = true,
                filterMode = filter,
                wrapMode = TextureWrapMode.Clamp
            };
            texture.Create();
            return texture;
        }

        private void CreateFrame(int layer)
        {
            _frameRoot = new GameObject("[QRS] Manual 5cm ROI Frame");
            _frameRoot.layer = layer;
            _frameRoot.transform.SetParent(transform, false);

            Shader lineShader = Resources.Load<Shader>("LocalFineRoiLine");
            if (lineShader == null) lineShader = Shader.Find("Genesis/LocalFineRoiLine");
            if (lineShader == null)
                throw new System.InvalidOperationException("LocalFineRoiLine shader missing");
            _frameMaterial = new Material(lineShader)
            {
                name = "[QRS] Manual 5cm ROI Frame"
            };

            _frameEdges = new LineRenderer[12];
            for (int i = 0; i < _frameEdges.Length; i++)
            {
                var edgeObject = new GameObject($"Edge {i:00}");
                edgeObject.layer = layer;
                edgeObject.transform.SetParent(_frameRoot.transform, false);
                var line = edgeObject.AddComponent<LineRenderer>();
                line.useWorldSpace = true;
                line.positionCount = 2;
                line.widthMultiplier = 0.008f;
                line.numCapVertices = 2;
                line.numCornerVertices = 1;
                line.shadowCastingMode = ShadowCastingMode.Off;
                line.receiveShadows = false;
                line.sharedMaterial = _frameMaterial;
                _frameEdges[i] = line;
            }
            UpdateFrameColor();
        }

        private void UpdatePreviewCenter(Camera camera)
        {
            if (camera == null) return;
            Vector3 proposed = camera.transform.position +
                               camera.transform.forward * _placementDistance;
            _center = new Vector3(
                Mathf.Round(proposed.x / VoxelSize) * VoxelSize,
                Mathf.Round(proposed.y / VoxelSize) * VoxelSize,
                Mathf.Round(proposed.z / VoxelSize) * VoxelSize);
        }

        /// <summary>
        /// 免调距放置：低频请求视野中心 8x8 深度中位数，橙框只在深度新鲜时
        /// 跟随视线。操作者看到框贴到目标后只需按一次右摇杆。
        /// </summary>
        private void RefreshAutoPreview(Camera camera, bool requestNow)
        {
            if (camera == null) return;
            DepthCapture depth = DepthCapture.Instance;
            float now = Time.unscaledTime;

            if (depth != null && DepthCapture.DepthAvailable &&
                (requestNow || now >= _nextPreviewSampleTime))
            {
                depth.RequestCenterDepthSample();
                _nextPreviewSampleTime = now + PreviewSampleInterval;
            }

            if (depth != null &&
                depth.CenterDepthSampleRevision != _lastCenterDepthRevision)
            {
                _lastCenterDepthRevision = depth.CenterDepthSampleRevision;
                if (depth.LastCenterDepthSampleValid &&
                    depth.LastCenterDepthMeters >= MinDepth &&
                    depth.LastCenterDepthMeters <= MaxDepth)
                {
                    _placementDistance = depth.LastCenterDepthMeters;
                    _lastAutoDepthTime = now;
                }
            }

            _autoDepthReady = now - _lastAutoDepthTime <=
                              PreviewDepthFreshSeconds;
            if (_frameRoot != null) _frameRoot.SetActive(_autoDepthReady);
            if (!_autoDepthReady) return;

            UpdatePreviewCenter(camera);
            ApplyPlacement();
        }

        private void ApplyPlacement()
        {
            if (_surface != null) _surface.VolumeOrigin = _center;
            if (_renderer != null)
                _renderer.UpdateBounds(new Bounds(_center,
                    (Vector3)((float3)VoxelCount * VoxelSize)));
            UpdateFrameGeometry();
        }

        private void UpdateFrameGeometry()
        {
            if (_frameEdges == null) return;
            Vector3 half = (Vector3)((float3)VoxelCount * VoxelSize * 0.5f);
            for (int i = 0; i < 8; i++)
            {
                _frameCorners[i] = _center + new Vector3(
                    (i & 1) == 0 ? -half.x : half.x,
                    (i & 2) == 0 ? -half.y : half.y,
                    (i & 4) == 0 ? -half.z : half.z);
            }
            for (int i = 0; i < 12; i++)
            {
                _frameEdges[i].SetPosition(0,
                    _frameCorners[FrameEdgePairs[i, 0]]);
                _frameEdges[i].SetPosition(1,
                    _frameCorners[FrameEdgePairs[i, 1]]);
            }
        }

        private void UpdateFrameColor()
        {
            if (_frameEdges == null) return;
            Color color = _locked ? LockedColor : PreviewColor;
            for (int i = 0; i < _frameEdges.Length; i++)
            {
                _frameEdges[i].startColor = color;
                _frameEdges[i].endColor = color;
            }
        }

        private static int CeilDiv(int value, int divisor)
        {
            return (value + divisor - 1) / divisor;
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
            DisposeResources();
        }

        private void DisposeResources()
        {
            _surface?.Dispose();
            _surface = null;
            ReleaseVolume(ref _tsdf);
            ReleaseVolume(ref _history);
            ReleaseVolume(ref _coherence);
            ReleaseVolume(ref _candidate);
            ReleaseVolume(ref _color);
            ReleaseVolume(ref _trace);
            if (_frameMaterial != null) Destroy(_frameMaterial);
            _frameMaterial = null;
            if (_comparisonMaterial != null) Destroy(_comparisonMaterial);
            _comparisonMaterial = null;
            if (_meshObject != null) Destroy(_meshObject);
            if (_frameRoot != null) Destroy(_frameRoot);
            _meshObject = null;
            _frameRoot = null;
            _initialized = false;
        }

        private static void ReleaseVolume(ref RenderTexture texture)
        {
            if (texture == null) return;
            texture.Release();
            UnityEngine.Object.Destroy(texture);
            texture = null;
        }
    }
}

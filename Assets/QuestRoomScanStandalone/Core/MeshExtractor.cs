using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Orchestrates GPU Surface Nets mesh extraction from the TSDF volume.
    /// Dispatches GPUSurfaceNets compute shaders and manages the GPUMeshRenderer.
    /// </summary>
    public class MeshExtractor : MonoBehaviour
    {
        public static MeshExtractor Instance { get; private set; }

        [Header("Mesh Smoothing")]
        [SerializeField, Tooltip("Post-extraction vertex smoothing iterations. 0 = disabled.")]
        [Range(0, 8)] private int meshSmoothIterations = 1;
        [SerializeField, Tooltip("Laplacian blend strength per iteration.")]
        [Range(0.1f, 1f)] private float meshSmoothLambda = 0.33f;
        [SerializeField, Tooltip("HC back-projection strength to prevent volume shrinkage.")]
        [Range(0f, 1f)] private float meshSmoothBeta = 0.5f;

        [Header("Temporal Stability")]
        [SerializeField, Tooltip("Alpha for large displacements (fast convergence).")]
        [Range(0.1f, 1f)] private float temporalAlphaMax = 0.85f;
        [SerializeField, Tooltip("Alpha for long-stable vertices (strong resistance to change).")]
        [Range(0.01f, 0.5f)] private float temporalAlphaMin = 0.1f;
        [SerializeField, Tooltip("How quickly alpha decays from max to min as vertex stabilizes.")]
        [Range(0.01f, 1f)] private float temporalDecayRate = 0.15f;
        [SerializeField, Tooltip("Displacement threshold (meters) to consider a vertex still converging.")]
        [Range(0.001f, 0.02f)] private float convergenceThreshold = 0.005f;
        [SerializeField, Tooltip("Position changes below this (meters) are suppressed entirely.")]
        [Range(0f, 0.01f)] private float temporalDeadzone = 0.001f;

        [Header("Rendering")]
        [SerializeField] private Material scanMeshMaterial;

        [Header("Compute")]
        [SerializeField] public ComputeShader surfaceNetsCompute;
        [SerializeField, Tooltip("Max vertex fraction of total voxels (0.01-0.10).")]
        [Range(0.01f, 0.10f)] private float gpuVertexBudgetPercent = 0.08f;

        [Header("断崖只读诊断框")]
        [SerializeField, Tooltip("只限制诊断计数与导出，不影响融合、提取或生产网格。")]
        private bool enableDiagnosticRoi = true;
        [SerializeField, Tooltip("归一化画面范围：xMin, yMin, xMax, yMax。")]
        private Vector4 diagnosticRoiRect = new Vector4(0.2f, 0.25f, 0.8f, 0.75f);
        [SerializeField, Tooltip("归一化画面横坐标：近面/边缘与边缘/远景分界。")]
        private Vector2 diagnosticRoiSplitX = new Vector2(0.44f, 0.56f);

        [Header("只读二色时序验证")]
        [SerializeField, Min(1f), Tooltip("每个时序观察窗的长度（秒）。")]
        private float temporalDiagnosticWindowSeconds = 5f;
        [SerializeField, Range(0f, 1f), Tooltip("中栏边缘/多跳证据不足占比达到该值才计为坏窗。")]
        private float temporalDiagnosticInsufficientThreshold = 0.40f;
        [SerializeField, Range(0f, 1f), Tooltip("中栏当前深度支持占比不高于该值才计为坏窗。")]
        private float temporalDiagnosticSupportCeiling = 0.60f;
        [SerializeField, Min(1), Tooltip("连续多少个坏窗后才把局部候选标红。")]
        private int temporalDiagnosticRequiredWindows = 2;
        [SerializeField, Min(0f), Tooltip("开始扫描后忽略的预热时长（秒）。")]
        private float temporalDiagnosticWarmupSeconds = 5f;

        public bool DiagnosticRoiEnabled => enableDiagnosticRoi;
        public Vector4 DiagnosticRoiRect => diagnosticRoiRect;
        public Vector2 DiagnosticRoiSplitX => diagnosticRoiSplitX;

        private GPUSurfaceNets _gpuSurfaceNets;
        private GPUMeshRenderer _gpuRenderer;
        private int _extractCount;

        // A ledger sample is a completed GPU readback, not a frame-time guess.
        // The values are full mesh snapshots; repeated geometry is therefore
        // never summed as if it were a new unique event.
        private sealed class LedgerSample
        {
            public DateTime Utc;
            public float ElapsedSeconds;
            public long ExtractionSerial;
            public bool Strict;
            public uint[] Counters;
        }

        private static readonly string[] CounterNames = BuildCounterNames();

        private static string[] BuildCounterNames()
        {
            var names = new List<string>
            {
                "vertices", "indices", "rejected_unknown_edges", "mixed_observed_cells",
                "rejected_unknown_quads", "strict_emitted_cells",
                "source_cell_self", "source_cell_direct", "source_cell_relay",
                "source_cell_mixed", "source_cell_untracked",
                "confirm_cell_pending", "confirm_cell_confirmed", "confirm_cell_mixed",
                "confirm_cell_untracked",
                "source_tri_self", "source_tri_direct", "source_tri_relay",
                "source_tri_mixed", "source_tri_untracked",
                "confirm_tri_pending", "confirm_tri_confirmed", "confirm_tri_mixed",
                "confirm_tri_untracked"
            };
            string[] sources = { "untracked", "self", "direct", "relay", "mixed" };
            string[] confirmations = { "untracked", "pending", "confirmed", "mixed" };
            for (int s = 0; s < sources.Length; s++)
                for (int q = 0; q < confirmations.Length; q++)
                    names.Add($"joint_cell_{sources[s]}_{confirmations[q]}");
            for (int s = 0; s < sources.Length; s++)
                for (int q = 0; q < confirmations.Length; q++)
                    names.Add($"joint_tri_{sources[s]}_{confirmations[q]}");
            names.Add("white_tri_transition");
            names.Add("white_tri_pending");
            names.Add("white_tri_internal_mixed");
            names.Add("white_tri_admission_internal_mixed");
            names.Add("white_tri_confirmation_internal_mixed");
            names.Add("white_tri_both_internal_mixed");
            names.Add("white_tri_both_same_vertex");
            names.Add("white_tri_both_split_vertices");
            names.Add("white_tri_both_same_vertex_count_1");
            names.Add("white_tri_both_same_vertex_count_2");
            names.Add("white_tri_both_same_vertex_count_3");
            names.Add("double_mixed_vertex_total");
            names.Add("double_mixed_source_unknown_known");
            names.Add("double_mixed_source_self_direct");
            names.Add("double_mixed_source_self_relay");
            names.Add("double_mixed_source_direct_relay");
            names.Add("double_mixed_source_self_direct_relay");
            names.Add("double_mixed_source_other");
            names.Add("double_mixed_confirm_unknown_pending");
            names.Add("double_mixed_confirm_unknown_confirmed");
            names.Add("double_mixed_confirm_pending_confirmed");
            names.Add("double_mixed_confirm_all");
            names.Add("double_mixed_confirm_other");
            names.Add("double_mixed_assoc_self_pending_direct_confirmed");
            names.Add("double_mixed_assoc_self_confirmed_direct_pending");
            names.Add("double_mixed_assoc_self_pending_relay_confirmed");
            names.Add("double_mixed_assoc_self_confirmed_relay_pending");
            names.Add("double_mixed_assoc_direct_pending_relay_confirmed");
            names.Add("double_mixed_assoc_direct_confirmed_relay_pending");
            names.Add("double_mixed_assoc_overlap_or_multi");
            string[] residualKnownSource = { "self", "direct", "relay", "multi" };
            string[] residualConfirmation = { "unknown_pending", "unknown_confirmed", "pending_confirmed", "all_or_other" };
            for (int s = 0; s < residualKnownSource.Length; s++)
                for (int q = 0; q < residualConfirmation.Length; q++)
                    names.Add($"double_mixed_residual_unknown_{residualKnownSource[s]}_{residualConfirmation[q]}");
            names.Add("double_mixed_residual_known_unknown_confirmation");
            names.Add("double_mixed_residual_known_same_source_overlap");
            names.Add("double_mixed_residual_known_three_sources");
            names.Add("double_mixed_residual_known_missing_attribution");
            names.Add("double_mixed_residual_known_multi_assignment");
            names.Add("double_mixed_residual_invariant_violation");
            names.Add("unknown_residual_current_depth_support");
            names.Add("unknown_residual_history_only");
            names.Add("unknown_residual_current_free_space_contradiction");
            names.Add("unknown_residual_edge_or_multihop_insufficient");
            string[] roiZones = { "near", "edge", "far" };
            string[] roiEvidence = { "depth_support", "history_only", "free_space_contradiction", "edge_or_multihop_insufficient" };
            for (int zone = 0; zone < roiZones.Length; zone++)
                for (int evidence = 0; evidence < roiEvidence.Length; evidence++)
                    names.Add($"diagnostic_roi_{roiZones[zone]}_{roiEvidence[evidence]}");
            names.Add("candidate_tri_pending");
            names.Add("candidate_tri_mature_current");
            names.Add("candidate_tri_grace_held");
            names.Add("candidate_tri_edge_only");
            names.Add("candidate_tri_retired");
            names.Add("candidate_history_hash_overflow");
            return names.ToArray();
        }

        private readonly List<LedgerSample> _ledgerSamples = new List<LedgerSample>(1024);
        private string _ledgerSessionId = "未开始";
        private DateTime _ledgerStartedUtc;
        private float _ledgerStartedRealtime;
        private bool _ledgerOpen;
        private int _ledgerGeneration;
        private long _readbackSerial;
        private long _lastAppliedReadbackSerial;
        private string _lastLedgerExportPath = "";

        internal GPUSurfaceNets GpuSurfaceNets => _gpuSurfaceNets;
        public bool IsInitialized => _gpuSurfaceNets != null;

        /// <summary>Current GPU mesh vertex count (updated after each extraction via async readback).</summary>
        public int LastVertexCount { get; private set; }
        /// <summary>Current GPU mesh index count (updated after each extraction via async readback).</summary>
        public int LastIndexCount { get; private set; }
        public uint LastRejectedUnknownEdges { get; private set; }
        public uint LastMixedObservedCells { get; private set; }
        public uint LastRejectedUnknownQuads { get; private set; }
        public uint LastStrictEmittedCells { get; private set; }
        // Ledger A: admission source. This records why geometry entered the mesh;
        // it must never imply that the geometry was later confirmed by raw depth.
        public uint LastAdmissionSourceSelfCells { get; private set; }
        public uint LastAdmissionSourceDirectCells { get; private set; }
        public uint LastAdmissionSourceRelayCells { get; private set; }
        public uint LastAdmissionSourceMixedCells { get; private set; }
        public uint LastAdmissionSourceUntrackedCells { get; private set; }
        public uint LastAdmissionSourceSelfTriangles { get; private set; }
        public uint LastAdmissionSourceDirectTriangles { get; private set; }
        public uint LastAdmissionSourceRelayTriangles { get; private set; }
        public uint LastAdmissionSourceMixedTriangles { get; private set; }
        public uint LastAdmissionSourceUntrackedTriangles { get; private set; }

        // Ledger B: real confirmation. This records whether admitted geometry was
        // actually re-observed by a valid receiver-side raw-depth sample.
        public uint LastRealConfirmationPendingCells { get; private set; }
        public uint LastRealConfirmationConfirmedCells { get; private set; }
        public uint LastRealConfirmationMixedCells { get; private set; }
        public uint LastRealConfirmationUntrackedCells { get; private set; }
        public uint LastRealConfirmationPendingTriangles { get; private set; }
        public uint LastRealConfirmationConfirmedTriangles { get; private set; }
        public uint LastRealConfirmationMixedTriangles { get; private set; }
        public uint LastRealConfirmationUntrackedTriangles { get; private set; }
        private readonly uint[] _lastJointCells = new uint[20];
        private readonly uint[] _lastJointTriangles = new uint[20];
        public uint LastWhiteTransitionTriangles { get; private set; }
        public uint LastWhitePendingTriangles { get; private set; }
        public uint LastWhiteInternalMixedTriangles { get; private set; }
        public uint LastWhiteAdmissionInternalMixedTriangles { get; private set; }
        public uint LastWhiteConfirmationInternalMixedTriangles { get; private set; }
        public uint LastWhiteBothInternalMixedTriangles { get; private set; }
        public uint LastWhiteBothSameVertexTriangles { get; private set; }
        public uint LastWhiteBothSplitVerticesTriangles { get; private set; }
        public uint LastWhiteBothSameVertexCount1Triangles { get; private set; }
        public uint LastWhiteBothSameVertexCount2Triangles { get; private set; }
        public uint LastWhiteBothSameVertexCount3Triangles { get; private set; }
        public uint LastDoubleMixedVertexTotal { get; private set; }
        private readonly uint[] _lastDoubleMixedSourceSet = new uint[6];
        private readonly uint[] _lastDoubleMixedConfirmationSet = new uint[5];
        private readonly uint[] _lastDoubleMixedAssociation = new uint[7];
        private readonly uint[] _lastDoubleMixedResidualUnknownCross = new uint[16];
        private readonly uint[] _lastDoubleMixedResidualKnown = new uint[6];
        private readonly uint[] _lastUnknownResidualEvidence = new uint[4];
        private readonly uint[] _lastDiagnosticRoiEvidence = new uint[12];
        private float _temporalDiagnosticStartedRealtime = -1f;
        private float _temporalDiagnosticWindowStartedRealtime = -1f;
        private ulong _temporalMiddleSupportSum;
        private ulong _temporalMiddleHistorySum;
        private ulong _temporalMiddleContradictionSum;
        private ulong _temporalMiddleInsufficientSum;
        private int _temporalDiagnosticCompletedWindows;
        private int _temporalDiagnosticConsecutiveBadWindows;
        private float _lastTemporalSupportRatio = 1f;
        private float _lastTemporalInsufficientRatio;
        private bool _temporalIllegalCandidateActive;
        public bool UseStrictObservedExtraction { get; private set; }
        // Candidate B is now the sole foreground production mesh. Production A
        // remains backend-only for ledger/export comparison.
        public bool UseJointDiagnosticDisplay { get; private set; } = true;
        public string ActiveExtractionLabel => "生产";

        private VolumeIntegrator _volume;

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            _volume = VolumeIntegrator.Instance;
            if (_volume == null)
                throw new Exception("[RoomScan] VolumeIntegrator not found");

            if (surfaceNetsCompute == null)
                throw new Exception("[RoomScan] surfaceNetsCompute not assigned on MeshExtractor");

            // GPU Surface Nets buffers (~480 MB at the default 256³ voxel grid)
            // are allocated lazily — first scan via RoomScanner.StartScanning,
            // or full-load via RoomScanPersistence.LoadPackageAsync. Pure
            // refined-mesh-only replay paths never trigger this. Keep the
            // pipeline log here so device traces still show RP/stereo/material
            // state at scene load, but defer the heavy Init().
            var rpAsset = GraphicsSettings.currentRenderPipeline;
            Logger.Info($"MeshExtractor Start (Surface Nets buffers deferred): " +
                $"mat={scanMeshMaterial?.name ?? "NULL"}, " +
                $"shader={scanMeshMaterial?.shader?.name ?? "NULL"}, " +
                $"rp={rpAsset?.name ?? "NULL"}, " +
                $"stereoMode={UnityEngine.XR.XRSettings.stereoRenderingMode}");
        }

        /// <summary>
        /// Lazy initializer. Brings up GPU Surface Nets buffers + the renderer
        /// component if they aren't already up. Idempotent. Existing callers
        /// (<see cref="Reinitialize"/>, <see cref="RoomScanner"/>'s update loop
        /// guard, the Start of every scan) all funnel through here so the
        /// allocation cost only appears when a scan or full reload actually
        /// needs it.
        /// </summary>
        public void EnsureInitialized()
        {
            if (_gpuSurfaceNets != null) return;
            Init();
        }

        private void OnDestroy()
        {
            _gpuSurfaceNets?.Dispose();
            _gpuSurfaceNets = null;
        }

        private void Init()
        {
            _gpuSurfaceNets = new GPUSurfaceNets(surfaceNetsCompute)
            {
                MinMeshWeight = _volume.MinMeshWeight,
                SmoothIterations = meshSmoothIterations,
                SmoothLambda = meshSmoothLambda,
                SmoothBeta = meshSmoothBeta,
                TemporalAlphaMax = temporalAlphaMax,
                TemporalAlphaMin = temporalAlphaMin,
                TemporalDecayRate = temporalDecayRate,
                ConvergenceThreshold = convergenceThreshold,
                TemporalDeadzone = temporalDeadzone,
                DiagnosticRoiEnabled = enableDiagnosticRoi,
                DiagnosticRoiRect = diagnosticRoiRect,
                DiagnosticRoiSplitX = diagnosticRoiSplitX
            };

            _gpuSurfaceNets.EnsureBuffers(_volume.VoxelCount, gpuVertexBudgetPercent);

            _gpuRenderer = gameObject.AddComponent<GPUMeshRenderer>();
            _gpuRenderer.GpuMeshMaterial = scanMeshMaterial;
            _gpuRenderer.Initialize(_gpuSurfaceNets, _gpuSurfaceNets.GetVolumeBounds(_volume.VoxelSize));
            _gpuRenderer.SetStrictObservedDisplay(false);
            _gpuRenderer.SetJointDiagnosticDisplay(UseJointDiagnosticDisplay);
            _gpuRenderer.SetTemporalIllegalCandidateActive(_temporalIllegalCandidateActive);

            Logger.Info($"GPU Surface Nets initialized lazily: voxels={_volume.VoxelCount}, " +
                      $"voxSize={_volume.VoxelSize}");
        }

        /// <summary>
        /// Run one GPU mesh extraction pass from the current TSDF volume state.
        /// Called by RoomScanner at the configured mesh extraction rate.
        /// </summary>
        public void Extract()
        {
            if (_gpuSurfaceNets == null) return;

            _extractCount++;
            _gpuSurfaceNets.MinMeshWeight = _volume.MinMeshWeight;
            // The live/front pipeline is always production.  Strict extraction
            // is captured only by the paired save transaction below.
            UseStrictObservedExtraction = false;
            _gpuSurfaceNets.StrictObservedEdges = false;
            ExtractCurrentVolume();

            if (_gpuRenderer != null)
                _gpuRenderer.UpdateBounds(_gpuSurfaceNets.GetVolumeBounds(_volume.VoxelSize));

            var counters = _gpuSurfaceNets.CountersBuffer;
            if (counters != null)
            {
                int requestGeneration = _ledgerGeneration;
                long requestSerial = ++_readbackSerial;
                const bool requestWasStrict = false;
                AsyncGPUReadback.Request(counters, (req) =>
                {
                    if (req.hasError || requestGeneration != _ledgerGeneration ||
                        requestSerial <= _lastAppliedReadbackSerial)
                        return;

                    var data = req.GetData<uint>();
                    if (data.Length < CounterNames.Length) return;

                    uint[] snapshot = new uint[CounterNames.Length];
                    for (int i = 0; i < snapshot.Length; i++)
                        snapshot[i] = data[i];

                    _lastAppliedReadbackSerial = requestSerial;
                    ApplySnapshot(snapshot);
                    UpdateTemporalDiagnostic(snapshot);
                    if (_ledgerOpen)
                    {
                        _ledgerSamples.Add(new LedgerSample
                        {
                            Utc = DateTime.UtcNow,
                            ElapsedSeconds = Time.realtimeSinceStartup - _ledgerStartedRealtime,
                            ExtractionSerial = requestSerial,
                            Strict = requestWasStrict,
                            Counters = snapshot
                        });
                    }
                });
            }
        }

        private void ApplySnapshot(uint[] data)
        {
            LastVertexCount = (int)data[0];
            LastIndexCount = (int)data[1];
            LastRejectedUnknownEdges = data[2];
            LastMixedObservedCells = data[3];
            LastRejectedUnknownQuads = data[4];
            LastStrictEmittedCells = data[5];

            LastAdmissionSourceSelfCells = data[6];
            LastAdmissionSourceDirectCells = data[7];
            LastAdmissionSourceRelayCells = data[8];
            LastAdmissionSourceMixedCells = data[9];
            LastAdmissionSourceUntrackedCells = data[10];
            LastRealConfirmationPendingCells = data[11];
            LastRealConfirmationConfirmedCells = data[12];
            LastRealConfirmationMixedCells = data[13];
            LastRealConfirmationUntrackedCells = data[14];

            LastAdmissionSourceSelfTriangles = data[15];
            LastAdmissionSourceDirectTriangles = data[16];
            LastAdmissionSourceRelayTriangles = data[17];
            LastAdmissionSourceMixedTriangles = data[18];
            LastAdmissionSourceUntrackedTriangles = data[19];
            LastRealConfirmationPendingTriangles = data[20];
            LastRealConfirmationConfirmedTriangles = data[21];
            LastRealConfirmationMixedTriangles = data[22];
            LastRealConfirmationUntrackedTriangles = data[23];
            for (int i = 0; i < 20; i++)
            {
                _lastJointCells[i] = data[24 + i];
                _lastJointTriangles[i] = data[44 + i];
            }
            LastWhiteTransitionTriangles = data[64];
            LastWhitePendingTriangles = data[65];
            LastWhiteInternalMixedTriangles = data[66];
            LastWhiteAdmissionInternalMixedTriangles = data[67];
            LastWhiteConfirmationInternalMixedTriangles = data[68];
            LastWhiteBothInternalMixedTriangles = data[69];
            LastWhiteBothSameVertexTriangles = data[70];
            LastWhiteBothSplitVerticesTriangles = data[71];
            LastWhiteBothSameVertexCount1Triangles = data[72];
            LastWhiteBothSameVertexCount2Triangles = data[73];
            LastWhiteBothSameVertexCount3Triangles = data[74];
            LastDoubleMixedVertexTotal = data[75];
            for (int i = 0; i < _lastDoubleMixedSourceSet.Length; i++)
                _lastDoubleMixedSourceSet[i] = data[76 + i];
            for (int i = 0; i < _lastDoubleMixedConfirmationSet.Length; i++)
                _lastDoubleMixedConfirmationSet[i] = data[82 + i];
            for (int i = 0; i < _lastDoubleMixedAssociation.Length; i++)
                _lastDoubleMixedAssociation[i] = data[87 + i];
            for (int i = 0; i < _lastDoubleMixedResidualUnknownCross.Length; i++)
                _lastDoubleMixedResidualUnknownCross[i] = data[94 + i];
            for (int i = 0; i < _lastDoubleMixedResidualKnown.Length; i++)
                _lastDoubleMixedResidualKnown[i] = data[110 + i];
            for (int i = 0; i < _lastUnknownResidualEvidence.Length; i++)
                _lastUnknownResidualEvidence[i] = data[116 + i];
            for (int i = 0; i < _lastDiagnosticRoiEvidence.Length; i++)
                _lastDiagnosticRoiEvidence[i] = data[120 + i];
        }

        /// <summary>
        /// Dispatch extraction with the last valid stereo depth frame attached as
        /// read-only evidence. A frozen frame remains valid for paired exports.
        /// </summary>
        private void ExtractCurrentVolume()
        {
            DepthCapture depthCapture = DepthCapture.Instance;
            Texture currentDepth = depthCapture != null ? depthCapture.DepthTex : null;
            Texture currentEdgeReason = depthCapture != null ? depthCapture.EdgeReasonTex : null;
            bool currentEvidenceAvailable =
                depthCapture != null && currentDepth != null && depthCapture.FrameCount > 0;

            _gpuSurfaceNets.Extract(
                _volume.Volume,
                _volume.ColorVolume,
                _volume.AdmissionTraceVolume,
                _volume.VoxelSize,
                currentDepth,
                currentEdgeReason,
                currentEvidenceAvailable);
        }

        /// <summary>Start one cumulative diagnostic session. Pause/resume is idempotent.</summary>
        public void BeginLedgerSession()
        {
            if (_ledgerOpen) return;

            _ledgerSamples.Clear();
            _ledgerStartedUtc = DateTime.UtcNow;
            _ledgerStartedRealtime = Time.realtimeSinceStartup;
            _ledgerSessionId = _ledgerStartedUtc.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            _lastLedgerExportPath = "";
            _ledgerOpen = true;
            ResetTemporalDiagnosticState();
            Logger.Info($"累计账开始: {_ledgerSessionId}");
        }

        public string GetLedgerSessionStatsCompact()
        {
            string state = _ledgerOpen ? "记账中" : "未记账";
            string exported = _lastLedgerExportPath.Length > 0 ? " 已保存" : "";
            return $"{state} 样本{_ledgerSamples.Count}{exported}";
        }

        /// <summary>
        /// Export the accumulated sequence of complete mesh snapshots. Values are
        /// intentionally not added together because the same geometry appears in
        /// many extractions; CSV rows preserve the only honest cumulative record.
        /// </summary>
        public string FinalizeAndExportLedger(string reason)
        {
            if (!_ledgerOpen && _ledgerSamples.Count == 0)
                return "";

            try
            {
                CapturePairedExportSnapshots();

                string outputDir = Path.Combine(Application.persistentDataPath, "ScanCoverDiagnostics");
                Directory.CreateDirectory(outputDir);
                string stem = "mesh_ledger_" + _ledgerSessionId;
                string csvPath = Path.Combine(outputDir, stem + ".csv");
                string summaryPath = Path.Combine(outputDir, stem + "_summary.txt");

                var csv = new StringBuilder(4096 + _ledgerSamples.Count * 256);
                csv.Append("session_id,utc,elapsed_s,extract_serial,mode");
                for (int i = 0; i < CounterNames.Length; i++)
                    csv.Append(',').Append(CounterNames[i]);
                csv.AppendLine();

                for (int s = 0; s < _ledgerSamples.Count; s++)
                {
                    LedgerSample sample = _ledgerSamples[s];
                    csv.Append(_ledgerSessionId).Append(',')
                       .Append(sample.Utc.ToString("O", CultureInfo.InvariantCulture)).Append(',')
                       .Append(sample.ElapsedSeconds.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                       .Append(sample.ExtractionSerial).Append(',')
                       .Append(sample.Strict ? "strict" : "production");
                    for (int i = 0; i < sample.Counters.Length; i++)
                        csv.Append(',').Append(sample.Counters[i]);
                    csv.AppendLine();
                }

                File.WriteAllText(csvPath, csv.ToString(), new UTF8Encoding(true));
                File.WriteAllText(summaryPath, BuildLedgerSummary(reason), new UTF8Encoding(true));
                _lastLedgerExportPath = summaryPath;
                _ledgerOpen = false;
                Logger.Info($"累计账已保存: {summaryPath}");
                return summaryPath;
            }
            catch (Exception e)
            {
                Logger.Error($"累计账导出失败: {e.Message}");
                return "";
            }
        }

        private string BuildLedgerSummary(string reason)
        {
            var sb = new StringBuilder(4096);
            sb.AppendLine("QRS 网格累计诊断账");
            sb.AppendLine($"会话: {_ledgerSessionId}");
            sb.AppendLine($"开始(UTC): {_ledgerStartedUtc:O}");
            sb.AppendLine($"结束(UTC): {DateTime.UtcNow:O}");
            sb.AppendLine($"结束原因: {reason}");
            sb.AppendLine($"完成回读样本: {_ledgerSamples.Count}");
            sb.AppendLine("统计口径: 每行是一次完整 GPU 提取快照；同一网格会跨行重复出现，因此禁止把各行直接求和当作新增几何。");
            sb.AppendLine("准入来源: 只统计实际零交叉边端点；记录体素本生命周期首次进入时的 自身/直补/接力 来源，来源不随后续观察改写。");
            sb.AppendLine("真实确认: 当前零交叉几何同时获得有效原始接收像素、投影窄带和既有 TSDF 一致性确认；它不由准入来源推断。");
            sb.AppendLine("原始联合账: 5种准入来源×4种确认状态仍完整保留并导出，但不再直接决定 Y 键显示颜色。");
            sb.AppendLine("成对终检: 保存时从同一份冻结 TSDF 依次抓取生产与严格快照；严格快照仅用于导出对照，不进入前台显示。");
            sb.AppendLine($"边缘只读验证: 三点获当前深度支持时画完整绿色三角；仅两点受支持时只画两点间绿色边线。中栏证据不足>={temporalDiagnosticInsufficientThreshold:P0}、当前深度支持<={temporalDiagnosticSupportCeiling:P0}，连续{Mathf.Max(1, temporalDiagnosticRequiredWindows)}个约{Mathf.Max(1f, temporalDiagnosticWindowSeconds):F1}秒窗口后形成的红色候选与其余未决状态均隐藏，但继续记账。它不参与生产准入、融合、删面或拓扑。");
            sb.AppendLine($"边缘终态: {(_temporalIllegalCandidateActive ? "隐藏红色候选激活" : "绿色观察")}，最近中栏支持{_lastTemporalSupportRatio:P1}，证据不足{_lastTemporalInsufficientRatio:P1}，连续坏窗{_temporalDiagnosticConsecutiveBadWindows}/{Mathf.Max(1, temporalDiagnosticRequiredWindows)}。");

            if (_ledgerSamples.Count == 0)
                return sb.ToString();

            uint[] peak = new uint[CounterNames.Length];
            LedgerSample lastProduction = null;
            LedgerSample lastStrict = null;
            for (int s = 0; s < _ledgerSamples.Count; s++)
            {
                LedgerSample sample = _ledgerSamples[s];
                if (sample.Strict) lastStrict = sample; else lastProduction = sample;
                for (int i = 0; i < peak.Length; i++)
                    if (sample.Counters[i] > peak[i]) peak[i] = sample.Counters[i];
            }

            AppendSnapshotSummary(sb, "生产末次", lastProduction);
            AppendSnapshotSummary(sb, "严格末次", lastStrict);
            sb.AppendLine("全会话峰值:");
            for (int i = 0; i < CounterNames.Length; i++)
                sb.Append(CounterNames[i]).Append('=').Append(peak[i]).Append(i % 4 == 3 ? '\n' : ' ');
            return sb.ToString();
        }

        private static void AppendSnapshotSummary(StringBuilder sb, string label, LedgerSample sample)
        {
            if (sample == null) return;
            uint[] c = sample.Counters;
            sb.AppendLine($"{label}: t={sample.ElapsedSeconds:F1}s 顶点={c[0]} 三角={c[1] / 3}");
            sb.AppendLine($"  准入点 自{c[6]}/直{c[7]}/接{c[8]}/混{c[9]}/空{c[10]}");
            sb.AppendLine($"  确认点 待{c[11]}/真{c[12]}/混{c[13]}/空{c[14]}");
            sb.AppendLine($"  准入面 自{c[15]}/直{c[16]}/接{c[17]}/混{c[18]}/空{c[19]}");
            sb.AppendLine($"  确认面 待{c[20]}/真{c[21]}/混{c[22]}/空{c[23]}");
            AppendJointMatrix(sb, "  联合点", c, 24);
            AppendJointMatrix(sb, "  联合面", c, 44);
            sb.AppendLine($"  白因面 交界{c[64]}/待定{c[65]}/内混总{c[66]} 入{c[67]}/确{c[68]}/双{c[69]}");
            sb.AppendLine($"  框内三栏(深/史/反/边) 近{c[120]}/{c[121]}/{c[122]}/{c[123]} " +
                          $"中{c[124]}/{c[125]}/{c[126]}/{c[127]} " +
                          $"远{c[128]}/{c[129]}/{c[130]}/{c[131]}");
            sb.AppendLine($"  双混面 同点{c[70]}/异点{c[71]} 同点数1:{c[72]}/2:{c[73]}/3:{c[74]}");
        }

        private static void AppendJointMatrix(StringBuilder sb, string label, uint[] c, int offset)
        {
            if (offset == 44)
                AppendCombinationLedgerSummary(sb, c);
            string[] source = { "空", "自", "直", "接", "混" };
            sb.AppendLine(label + "(列=空/待/真/混):");
            for (int s = 0; s < 5; s++)
            {
                int i = offset + s * 4;
                sb.Append("    ").Append(source[s]).Append(':')
                  .Append(c[i]).Append('/').Append(c[i + 1]).Append('/')
                  .Append(c[i + 2]).Append('/').Append(c[i + 3]).AppendLine();
            }
        }

        private static void AppendCombinationLedgerSummary(StringBuilder sb, uint[] c)
        {
            uint sourceSum = c[76] + c[77] + c[78] + c[79] + c[80] + c[81];
            uint confirmationSum = c[82] + c[83] + c[84] + c[85] + c[86];
            uint associationSum = c[87] + c[88] + c[89] + c[90] + c[91] + c[92] + c[93];
            uint residualSum = 0;
            for (int i = 94; i <= 115; i++) residualSum += c[i];
            uint unknownResidualSum = 0;
            for (int i = 94; i <= 109; i++) unknownResidualSum += c[i];
            uint evidenceSum = c[116] + c[117] + c[118] + c[119];
            sb.AppendLine($"  双混组合点 总{c[75]}");
            sb.AppendLine($"    来源集合 未知+已知{c[76]} 自+直{c[77]} 自+接{c[78]} 直+接{c[79]} 三源{c[80]} 其他{c[81]}");
            sb.AppendLine($"    确认集合 未知+待{c[82]} 未知+真{c[83]} 待+真{c[84]} 三态{c[85]} 其他{c[86]}");
            sb.AppendLine($"    对应 自待直真{c[87]} 自真直待{c[88]} 自待接真{c[89]} 自真接待{c[90]} 直待接真{c[91]} 直真接待{c[92]} 重叠/多源{c[93]}");
            sb.AppendLine($"    闭合 来源{sourceSum}/{c[75]} 确认{confirmationSum}/{c[75]} 对应{associationSum}/{c[75]}");
            sb.AppendLine($"    兜底拆分 未知+自(未知待/未知真/待真/三态余) {c[94]}/{c[95]}/{c[96]}/{c[97]}");
            sb.AppendLine($"             未知+直(未知待/未知真/待真/三态余) {c[98]}/{c[99]}/{c[100]}/{c[101]}");
            sb.AppendLine($"             未知+接(未知待/未知真/待真/三态余) {c[102]}/{c[103]}/{c[104]}/{c[105]}");
            sb.AppendLine($"             未知+多(未知待/未知真/待真/三态余) {c[106]}/{c[107]}/{c[108]}/{c[109]}");
            sb.AppendLine($"             已知余 未知确认{c[110]} 同源跨态{c[111]} 三来源{c[112]} 缺归属{c[113]} 多重归属{c[114]} 异常{c[115]}");
            sb.AppendLine($"    兜底闭合 {residualSum}/{c[93]}");
            sb.AppendLine($"    最终四分 当前深度支持{c[116]} 历史仅支持{c[117]} 当前自由空间反证{c[118]} 边缘/多跳不足{c[119]}");
            sb.AppendLine($"    最终四分闭合 {evidenceSum}/{unknownResidualSum}");
        }

        /// <summary>
        /// Capture production and strict counters from the same frozen TSDF at
        /// save time.  The strict pass temporarily reuses the extraction buffers
        /// to avoid a second ~480 MB GPU allocation, then production is restored.
        /// </summary>
        private void CapturePairedExportSnapshots()
        {
            if (_gpuSurfaceNets == null || _volume == null)
                throw new InvalidOperationException("成对终检失败：GPU 提取器或 TSDF 尚未初始化");

            uint[] production = null;
            uint[] strict = null;
            bool previousCandidateHistoryUpdate = _gpuSurfaceNets.CandidateHistoryUpdateEnabled;
            _gpuSurfaceNets.CandidateHistoryUpdateEnabled = false;
            try
            {
                production = CaptureExtractionSnapshot(false);
                strict = CaptureExtractionSnapshot(true);
            }
            finally
            {
                try
                {
                    // Never leave the reusable buffers or renderer in strict mode.
                    // The whole save transaction is read-only for candidate B: its
                    // two consecutive confirmations must come from live extractions,
                    // not production/strict/restoration bookkeeping passes.
                    UseStrictObservedExtraction = false;
                    _gpuSurfaceNets.StrictObservedEdges = false;
                    ExtractCurrentVolume();
                    _gpuRenderer?.SetStrictObservedDisplay(false);
                    _gpuRenderer?.UpdateBounds(_gpuSurfaceNets.GetVolumeBounds(_volume.VoxelSize));
                }
                finally
                {
                    _gpuSurfaceNets.CandidateHistoryUpdateEnabled = previousCandidateHistoryUpdate;
                }
            }

            if (production == null || strict == null)
                throw new InvalidOperationException("成对终检失败：生产或严格快照为空");

            float elapsed = Time.realtimeSinceStartup - _ledgerStartedRealtime;
            long productionSerial = ++_readbackSerial;
            long strictSerial = ++_readbackSerial;

            // Invalidate any older asynchronous live readback.  Scanning is
            // paused by the save transaction, so no newer request can overtake it.
            _lastAppliedReadbackSerial = strictSerial;
            _ledgerSamples.Add(new LedgerSample
            {
                Utc = DateTime.UtcNow,
                ElapsedSeconds = elapsed,
                ExtractionSerial = productionSerial,
                Strict = false,
                Counters = production
            });
            _ledgerSamples.Add(new LedgerSample
            {
                Utc = DateTime.UtcNow,
                ElapsedSeconds = elapsed,
                ExtractionSerial = strictSerial,
                Strict = true,
                Counters = strict
            });

            ApplySnapshot(production);
            Logger.Info($"成对终检已抓取: 生产顶点={production[0]} 严格顶点={strict[0]}");
        }

        private uint[] CaptureExtractionSnapshot(bool strict)
        {
            _gpuSurfaceNets.MinMeshWeight = _volume.MinMeshWeight;
            _gpuSurfaceNets.StrictObservedEdges = strict;
            ExtractCurrentVolume();

            GraphicsBuffer counters = _gpuSurfaceNets.CountersBuffer;
            if (counters == null)
                throw new InvalidOperationException("GPU 计数缓冲不可用");

            var snapshot = new uint[CounterNames.Length];
            counters.GetData(snapshot);
            return snapshot;
        }

        /// <summary>Invalidate pending readbacks and return counters to a new-session state.</summary>
        public void ResetLedgerSessionAfterClear()
        {
            _ledgerGeneration++;
            _lastAppliedReadbackSerial = _readbackSerial;
            _ledgerOpen = false;
            _ledgerSamples.Clear();
            _ledgerSessionId = "未开始";
            ResetLastSnapshot();
            ResetTemporalDiagnosticState();
        }

        private void ResetLastSnapshot()
        {
            ApplySnapshot(new uint[CounterNames.Length]);
        }

        /// <summary>The strict candidate chain is the locked foreground production mesh.</summary>
        public void ToggleJointDiagnosticDisplay()
        {
            UseJointDiagnosticDisplay = true;
            _gpuRenderer?.SetJointDiagnosticDisplay(true);
            Logger.Info("Foreground mesh is locked to strict production B; production A is backend-only.");
        }

        public string GetJointDiagnosticStatsCompact()
        {
            return "生产B: 连续2次稳定3/3才显示；缺点/不连续/自由空间矛盾立即整三角拒绝";
        }

        private void ResetTemporalDiagnosticState()
        {
            float now = Time.realtimeSinceStartup;
            _temporalDiagnosticStartedRealtime = now;
            _temporalDiagnosticWindowStartedRealtime = now;
            _temporalMiddleSupportSum = 0;
            _temporalMiddleHistorySum = 0;
            _temporalMiddleContradictionSum = 0;
            _temporalMiddleInsufficientSum = 0;
            _temporalDiagnosticCompletedWindows = 0;
            _temporalDiagnosticConsecutiveBadWindows = 0;
            _lastTemporalSupportRatio = 1f;
            _lastTemporalInsufficientRatio = 0f;
            _temporalIllegalCandidateActive = false;
            _gpuRenderer?.SetTemporalIllegalCandidateActive(false);
        }

        private void UpdateTemporalDiagnostic(uint[] data)
        {
            if (!enableDiagnosticRoi || data == null || data.Length < 132)
                return;

            float now = Time.realtimeSinceStartup;
            if (_temporalDiagnosticStartedRealtime < 0f)
            {
                _temporalDiagnosticStartedRealtime = now;
                _temporalDiagnosticWindowStartedRealtime = now;
            }

            _temporalMiddleSupportSum += data[124];
            _temporalMiddleHistorySum += data[125];
            _temporalMiddleContradictionSum += data[126];
            _temporalMiddleInsufficientSum += data[127];
            float windowSeconds = Mathf.Max(1f, temporalDiagnosticWindowSeconds);
            if (now - _temporalDiagnosticWindowStartedRealtime < windowSeconds)
                return;

            ulong total = _temporalMiddleSupportSum + _temporalMiddleHistorySum +
                          _temporalMiddleContradictionSum + _temporalMiddleInsufficientSum;
            if (total > 0)
            {
                _lastTemporalSupportRatio = (float)_temporalMiddleSupportSum / total;
                _lastTemporalInsufficientRatio = (float)_temporalMiddleInsufficientSum / total;
            }
            else
            {
                _lastTemporalSupportRatio = 0f;
                _lastTemporalInsufficientRatio = 0f;
            }

            bool warmup = now - _temporalDiagnosticStartedRealtime < Mathf.Max(0f, temporalDiagnosticWarmupSeconds) ||
                          _temporalDiagnosticCompletedWindows == 0;
            bool badWindow = !warmup && total > 0 &&
                             _lastTemporalInsufficientRatio >= temporalDiagnosticInsufficientThreshold &&
                             _lastTemporalSupportRatio <= temporalDiagnosticSupportCeiling;

            _temporalDiagnosticConsecutiveBadWindows = badWindow
                ? _temporalDiagnosticConsecutiveBadWindows + 1
                : 0;
            _temporalIllegalCandidateActive =
                _temporalDiagnosticConsecutiveBadWindows >= Mathf.Max(1, temporalDiagnosticRequiredWindows);
            _gpuRenderer?.SetTemporalIllegalCandidateActive(_temporalIllegalCandidateActive);

            _temporalDiagnosticCompletedWindows++;
            _temporalMiddleSupportSum = 0;
            _temporalMiddleHistorySum = 0;
            _temporalMiddleContradictionSum = 0;
            _temporalMiddleInsufficientSum = 0;
            _temporalDiagnosticWindowStartedRealtime = now;
        }

        /// <summary>
        /// Read-only double-mixed vertex combination ledger. The source-set,
        /// confirmation-set, and association groups are disjoint partitions;
        /// each group must close against the same total.
        /// </summary>
        public string GetDoubleMixedCombinationStatsCompact()
        {
            uint sourceSum = 0;
            uint confirmationSum = 0;
            uint associationSum = 0;
            uint residualSum = 0;
            uint unknownResidualSum = 0;
            uint evidenceSum = 0;
            for (int i = 0; i < _lastDoubleMixedSourceSet.Length; i++) sourceSum += _lastDoubleMixedSourceSet[i];
            for (int i = 0; i < _lastDoubleMixedConfirmationSet.Length; i++) confirmationSum += _lastDoubleMixedConfirmationSet[i];
            for (int i = 0; i < _lastDoubleMixedAssociation.Length; i++) associationSum += _lastDoubleMixedAssociation[i];
            for (int i = 0; i < _lastDoubleMixedResidualUnknownCross.Length; i++)
            {
                residualSum += _lastDoubleMixedResidualUnknownCross[i];
                unknownResidualSum += _lastDoubleMixedResidualUnknownCross[i];
            }
            for (int i = 0; i < _lastDoubleMixedResidualKnown.Length; i++) residualSum += _lastDoubleMixedResidualKnown[i];
            for (int i = 0; i < _lastUnknownResidualEvidence.Length; i++) evidenceSum += _lastUnknownResidualEvidence[i];
            return $"总{LastDoubleMixedVertexTotal} " +
                   $"源 未知已知{_lastDoubleMixedSourceSet[0]}/自直{_lastDoubleMixedSourceSet[1]}/自接{_lastDoubleMixedSourceSet[2]}/直接{_lastDoubleMixedSourceSet[3]}/三源{_lastDoubleMixedSourceSet[4]}/其{_lastDoubleMixedSourceSet[5]} " +
                   $"确 未知待{_lastDoubleMixedConfirmationSet[0]}/未知真{_lastDoubleMixedConfirmationSet[1]}/待真{_lastDoubleMixedConfirmationSet[2]}/三态{_lastDoubleMixedConfirmationSet[3]}/其{_lastDoubleMixedConfirmationSet[4]} " +
                   $"配 自待直真{_lastDoubleMixedAssociation[0]}/自真直待{_lastDoubleMixedAssociation[1]}/自待接真{_lastDoubleMixedAssociation[2]}/自真接待{_lastDoubleMixedAssociation[3]}/直待接真{_lastDoubleMixedAssociation[4]}/直真接待{_lastDoubleMixedAssociation[5]}/复{_lastDoubleMixedAssociation[6]} " +
                   $"闭{sourceSum}/{confirmationSum}/{associationSum} 兜{residualSum}/{_lastDoubleMixedAssociation[6]} " +
                   $"终 深{_lastUnknownResidualEvidence[0]}/史{_lastUnknownResidualEvidence[1]}/反{_lastUnknownResidualEvidence[2]}/边{_lastUnknownResidualEvidence[3]} 闭{evidenceSum}/{unknownResidualSum}";
        }

        /// <summary>
        /// Read-only cliff sample split. Each column uses the same mutually
        /// exclusive support/history/contradiction/insufficient classes as the
        /// global final ledger; it never gates extraction.
        /// </summary>
        public string GetDiagnosticRoiStatsCompact()
        {
            if (!enableDiagnosticRoi) return "关闭";
            return $"近 深{_lastDiagnosticRoiEvidence[0]}/史{_lastDiagnosticRoiEvidence[1]}/反{_lastDiagnosticRoiEvidence[2]}/边{_lastDiagnosticRoiEvidence[3]} " +
                   $"中 深{_lastDiagnosticRoiEvidence[4]}/史{_lastDiagnosticRoiEvidence[5]}/反{_lastDiagnosticRoiEvidence[6]}/边{_lastDiagnosticRoiEvidence[7]} " +
                   $"远 深{_lastDiagnosticRoiEvidence[8]}/史{_lastDiagnosticRoiEvidence[9]}/反{_lastDiagnosticRoiEvidence[10]}/边{_lastDiagnosticRoiEvidence[11]}";
        }

        public string GetStrictObservedStatsCompact()
        {
            return "保存时成对导出";
        }

        /// <summary>
        /// Read-only immutable admission-source settlement for the endpoints of
        /// zero-crossing edges used by emitted cells and triangles.
        /// </summary>
        public string GetAdmissionSourceStatsCompact()
        {
            return $"点 自{LastAdmissionSourceSelfCells}/直{LastAdmissionSourceDirectCells}/接{LastAdmissionSourceRelayCells}/混{LastAdmissionSourceMixedCells}/空{LastAdmissionSourceUntrackedCells} " +
                   $"面 自{LastAdmissionSourceSelfTriangles}/直{LastAdmissionSourceDirectTriangles}/接{LastAdmissionSourceRelayTriangles}/混{LastAdmissionSourceMixedTriangles}/空{LastAdmissionSourceUntrackedTriangles}";
        }

        /// <summary>
        /// Read-only geometric-confirmation settlement for emitted Surface Nets
        /// cells and triangles. Admission source is intentionally absent.
        /// </summary>
        public string GetRealConfirmationStatsCompact()
        {
            return $"点 待{LastRealConfirmationPendingCells}/真{LastRealConfirmationConfirmedCells}/混{LastRealConfirmationMixedCells}/空{LastRealConfirmationUntrackedCells} " +
                   $"面 待{LastRealConfirmationPendingTriangles}/真{LastRealConfirmationConfirmedTriangles}/混{LastRealConfirmationMixedTriangles}/空{LastRealConfirmationUntrackedTriangles}";
        }

        /// <summary>
        /// Release GPU resources without re-creating them.
        /// Used by ClearAllData to avoid a heavy re-alloc while the GPU may
        /// still be referencing the old buffers from the previous frame's draw.
        /// Call <see cref="Reinitialize"/> when resources are needed again.
        /// </summary>
        public void DisposeOnly()
        {
            if (_gpuRenderer != null)
            {
                _gpuRenderer.RenderVisible = false;
                Destroy(_gpuRenderer);
                _gpuRenderer = null;
            }
            _gpuSurfaceNets?.Dispose();
            _gpuSurfaceNets = null;
        }

        /// <summary>
        /// Dispose GPU resources and reinitialize. Used after loading a saved scan.
        /// </summary>
        public void Reinitialize()
        {
            if (_gpuRenderer != null)
            {
                _gpuRenderer.RenderVisible = false;
                Destroy(_gpuRenderer);
                _gpuRenderer = null;
            }
            _gpuSurfaceNets?.Dispose();
            _gpuSurfaceNets = null;
            Init();
        }
    }
}

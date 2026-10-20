using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Read-only history recorder for the operator-locked blue frame. It samples
    /// only voxels and published Front geometry inside that world-space box.
    /// Nothing in fusion, extraction, publication or rendering reads its data.
    /// </summary>
    public sealed class LockedRoiHistoryCapture : MonoBehaviour
    {
        public static LockedRoiHistoryCapture Instance { get; private set; }

        private static readonly Vector3 FrameSizeMetres = new Vector3(0.8f, 1.2f, 0.8f);
        private static readonly Color FrameColor = new Color(0.05f, 0.90f, 1f, 0.98f);
        private static readonly int[,] EdgePairs =
        {
            { 0, 1 }, { 0, 2 }, { 0, 4 }, { 1, 3 }, { 1, 5 }, { 2, 3 },
            { 2, 6 }, { 3, 7 }, { 4, 5 }, { 4, 6 }, { 5, 7 }, { 6, 7 }
        };

        private static readonly int CellsId = Shader.PropertyToID("_LockedRoiCells");
        private static readonly int TsdfId = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ResponsibilityId = Shader.PropertyToID("_ResponsibilityVolume");
        private static readonly int RoiMinId = Shader.PropertyToID("_RoiMin");
        private static readonly int RoiCountId = Shader.PropertyToID("_RoiCount");
        private static readonly int VolumeCountId = Shader.PropertyToID("_VolumeCount");
        private static readonly int ResponsibilityAvailableId = Shader.PropertyToID("_ResponsibilityAvailable");
        private static readonly int FusionWitnessId = Shader.PropertyToID("_FusionWitness");
        private static readonly int RawDepthId = Shader.PropertyToID("_RawDepth");
        private static readonly int FusionDepthId = Shader.PropertyToID("_FusionDepth");
        private static readonly int DilatedDepthId = Shader.PropertyToID("_DilatedDepth");
        private static readonly int EdgeReasonId = Shader.PropertyToID("_EdgeReason");
        private static readonly int TemporalReasonId = Shader.PropertyToID("_TemporalReason");
        private static readonly int VoteWeightVolumeId = Shader.PropertyToID("_VoteWeightVolume");
        private static readonly int AccumulatedTsdfVolumeId = Shader.PropertyToID("_AccumulatedTsdfVolume");
        private static readonly int WriteEpochsId = Shader.PropertyToID("_WriteEpochs");
        private static readonly int ProjectionId = Shader.PropertyToID("_Projection");
        private static readonly int ProjectionInverseId = Shader.PropertyToID("_ProjectionInverse");
        private static readonly int ViewId = Shader.PropertyToID("_View");
        private static readonly int ViewInverseId = Shader.PropertyToID("_ViewInverse");
        private static readonly int FusionDepthSizeId = Shader.PropertyToID("_FusionDepthSize");
        private static readonly int VoxelSizeId = Shader.PropertyToID("_VoxelSize");
        private static readonly int VoxelDistanceId = Shader.PropertyToID("_VoxelDistance");
        private static readonly int MinUpdateDistanceId = Shader.PropertyToID("_MinUpdateDistance");
        private static readonly int MaxUpdateDistanceId = Shader.PropertyToID("_MaxUpdateDistance");
        private static readonly int ExpectedWriteEpochId = Shader.PropertyToID("_ExpectedWriteEpoch");
        private static readonly int RawDepthAvailableId = Shader.PropertyToID("_RawDepthAvailable");
        private static readonly int TemporalReasonAvailableId = Shader.PropertyToID("_TemporalReasonAvailable");

        private sealed class FilteredFront
        {
            public int Block, Generation;
            public uint PublishedEpoch;
            public byte[] Vertices, Indices, Crossings, CrossingSpans, RawSurfaceNets;
            public int VertexCount => Vertices != null ? Vertices.Length / 32 : 0;
            public int IndexCount => Indices != null ? Indices.Length / 4 : 0;
        }

        private sealed class HistorySample
        {
            public int Serial, UnityFrame, PlatformFrame, Integration;
            public uint DirtyEpoch;
            public double Time;
            public Vector3 CameraPosition;
            public Quaternion CameraRotation;
            public byte[] Cells;
            public readonly List<FilteredFront> Fronts = new List<FilteredFront>();
        }

        private sealed class FusionWitnessSample
        {
            public int Serial, Attempt, SourceFrame, Integration, UnityFrame;
            public int DepthWidth, DepthHeight;
            public uint DirtyEpochBefore, DirtyEpochAfter, WriteEpoch;
            public double Time;
            public Vector3 CameraPosition;
            public Quaternion CameraRotation;
            public bool RawAvailable, TemporalAvailable;
            public Matrix4x4[] Projection, View, ProjectionInverse, ViewInverse;
            public Task<byte[]> Readback;
            public byte[] Bytes;
        }

        private VolumeIntegrator _volume;
        private ComputeShader _captureShader;
        private int _captureKernel;
        private int _fusionPreKernel;
        private int _fusionPostKernel;
        private GraphicsBuffer _cellBuffer;
        private GraphicsBuffer _fusionWitnessBuffer;
        private int3 _roiCount;
        private int _cellCount;
        private readonly List<HistorySample> _history = new List<HistorySample>();
        private readonly List<FusionWitnessSample> _fusionWitnesses =
            new List<FusionWitnessSample>();
        private readonly Vector3[] _corners = new Vector3[8];
        private LineRenderer[] _edges;
        private GameObject _frameRoot;
        private Material _frameMaterial;
        private Vector3 _center;
        private float _placementDistance = 1.6f;
        private float _nextPlacementProbe;
        private float _nextSampleTime;
        private Task _pendingSample;
        private int _sampleSerial;
        private int _fusionWitnessSerial;
        private FusionWitnessSample _activeFusionWitness;
        private string _lastError = string.Empty;
        private bool _initialized;
        private bool _locked;
        private bool _recording;
        private bool _sessionActive;
        private bool _manualPlacement;

        private const float SampleIntervalSeconds = 0.5f;
        private const int MaximumSamples = 240;
        private const int CellStrideBytes = 32;
        private const int FusionWitnessStrideBytes = 128;
        private const int MaximumFusionWitnesses = 512;

        public bool Locked => _locked;
        public bool Recording => _recording;
        public string StatusCompact => !_initialized ? "蓝框:未就绪" :
            !_locked ? $"蓝框:右摇杆前后推拉 {_placementDistance:F1}m，按下锁定" :
            !_sessionActive ? "蓝框:已锁定，按扳机从空卷开始" :
            _recording
                ? $"蓝框:采集中 网格{_history.Count} 深证{_fusionWitnesses.Count}"
                : $"蓝框:已停，网格{_history.Count} 深证{_fusionWitnesses.Count}";

        internal void Initialize(VolumeIntegrator volume, int layer)
        {
            if (volume == null) return;
            if (_initialized)
            {
                enabled = true;
                Instance = this;
                if (_frameRoot != null) _frameRoot.SetActive(true);
                UpdatePreview(true);
                return;
            }
            Instance = this;
            _volume = volume;
            _captureShader = Resources.Load<ComputeShader>("LockedRoiHistoryCapture");
            if (_captureShader == null)
                throw new InvalidOperationException("LockedRoiHistoryCapture.compute missing");
            _captureKernel = _captureShader.FindKernel("CaptureCells");
            _fusionPreKernel = _captureShader.FindKernel("CaptureFusionPre");
            _fusionPostKernel = _captureShader.FindKernel("CaptureFusionPost");
            _roiCount = new int3(
                Mathf.Max(1, Mathf.RoundToInt(FrameSizeMetres.x / volume.VoxelSize)),
                Mathf.Max(1, Mathf.RoundToInt(FrameSizeMetres.y / volume.VoxelSize)),
                Mathf.Max(1, Mathf.RoundToInt(FrameSizeMetres.z / volume.VoxelSize)));
            _cellCount = checked(_roiCount.x * _roiCount.y * _roiCount.z);
            _cellBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _cellCount, CellStrideBytes);
            _fusionWitnessBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _cellCount, FusionWitnessStrideBytes);
            CreateFrame(layer);
            _initialized = true;
            UpdatePreview(true);
            Logger.Info($"Locked blue ROI history ready: {FrameSizeMetres} m, " +
                        $"{_roiCount.x}x{_roiCount.y}x{_roiCount.z} TSDF cells");
        }

        internal void BeginSession()
        {
            if (_pendingSample != null && !_pendingSample.IsCompleted)
                throw new InvalidOperationException("上一轮蓝框 GPU 回读尚未结束");
            _sessionActive = true;
            _recording = _locked;
            _history.Clear();
            _fusionWitnesses.Clear();
            _sampleSerial = 0;
            _fusionWitnessSerial = 0;
            _activeFusionWitness = null;
            _pendingSample = null;
            _lastError = string.Empty;
            _nextSampleTime = Time.unscaledTime + SampleIntervalSeconds;
            if (!_locked)
            {
                UpdatePreview(true);
                return;
            }

            // Capture the empty-volume baseline immediately. The GPU dispatch
            // is queued before camera/depth startup and therefore before the
            // first fusion write. Later samples can now be traced back to a
            // real integration-0 state instead of starting from an already
            // meshed volume.
            _pendingSample = CaptureOneAsync();
        }

        public void AdjustPlacement(float push, float unscaledDeltaTime)
        {
            if (!_initialized || _locked || Mathf.Abs(push) < 0.15f) return;
            _manualPlacement = true;
            _placementDistance = Mathf.Clamp(
                _placementDistance + push * Mathf.Max(0f, unscaledDeltaTime) * 0.8f,
                0.3f, 4f);
            UpdatePreview(false);
        }

        public string AdvanceCapture()
        {
            if (!_initialized) return "蓝色锁框尚未就绪";
            if (_pendingSample != null && !_pendingSample.IsCompleted)
                return "蓝框正在收当前历史帧，请稍候";
            if (!_locked)
            {
                UpdatePreview(true);
                _locked = true;
                _recording = _sessionActive;
                _history.Clear();
                _fusionWitnesses.Clear();
                _sampleSerial = 0;
                _fusionWitnessSerial = 0;
                _activeFusionWitness = null;
                _nextSampleTime = 0f;
                return _sessionActive
                    ? "蓝框已锁定：采框内同帧深度、TSDF 前后态和已发布网格；换视角即可"
                    : "蓝框已锁定；按扳机后从空卷第0次融合开始记录";
            }
            if (_recording)
            {
                _recording = false;
                return $"蓝框历史已暂停并保留 {_history.Count} 帧；再按右摇杆可重新瞄准";
            }
            _locked = false;
            _manualPlacement = false;
            UpdatePreview(true);
            return "蓝框已解锁，请重新瞄准后按右摇杆开始";
        }

        internal void StopForSeal()
        {
            _recording = false;
            _sessionActive = false;
        }

        /// <summary>
        /// Called only after the package reports a safe seal and the owning
        /// volume has been cleared. This makes the next package a new capture,
        /// not an append to stale tasks or evidence from the previous package.
        /// </summary>
        internal bool PrepareForNextPackage()
        {
            if (_pendingSample != null && !_pendingSample.IsCompleted)
            {
                Logger.Warning("蓝框仍有未完成回读，暂不能重置下一包");
                return false;
            }
            foreach (FusionWitnessSample witness in _fusionWitnesses)
            {
                if (witness.Readback != null && !witness.Readback.IsCompleted)
                {
                    Logger.Warning("蓝框仍有未完成融合证据回读，暂不能重置下一包");
                    return false;
                }
            }
            _recording = false;
            _sessionActive = false;
            _locked = false;
            _manualPlacement = false;
            _activeFusionWitness = null;
            _pendingSample = null;
            _history.Clear();
            _fusionWitnesses.Clear();
            _sampleSerial = 0;
            _fusionWitnessSerial = 0;
            _lastError = string.Empty;
            _nextSampleTime = 0f;
            if (_frameRoot != null) _frameRoot.SetActive(true);
            UpdatePreview(true);
            return true;
        }

        internal void ShutdownFeature()
        {
            StopForSeal();
            _locked = false;
            _manualPlacement = false;
            _activeFusionWitness = null;
            if (_frameRoot != null) _frameRoot.SetActive(false);
            if (Instance == this) Instance = null;
            enabled = false;
        }

        private void LateUpdate()
        {
            if (!_initialized) return;
            if (!_locked)
            {
                UpdatePreview(false);
                return;
            }
            if (!_recording || Time.unscaledTime < _nextSampleTime ||
                (_pendingSample != null && !_pendingSample.IsCompleted)) return;
            _nextSampleTime = Time.unscaledTime + SampleIntervalSeconds;
            _pendingSample = CaptureOneAsync();
        }

        private void UpdatePreview(bool requestNow)
        {
            Camera camera = Camera.main;
            if (camera == null || _volume == null) return;
            DepthCapture depth = DepthCapture.Instance;
            if (depth != null && DepthCapture.DepthAvailable &&
                (requestNow || Time.unscaledTime >= _nextPlacementProbe))
            {
                depth.RequestCenterDepthSample();
                _nextPlacementProbe = Time.unscaledTime + 0.25f;
            }
            if (!_manualPlacement && depth != null && depth.LastCenterDepthMeters >= 0.3f &&
                depth.LastCenterDepthMeters <= 4f)
                _placementDistance = depth.LastCenterDepthMeters;
            float voxel = Mathf.Max(0.001f, _volume.VoxelSize);
            Vector3 proposed = camera.transform.position +
                               camera.transform.forward * _placementDistance;
            _center = new Vector3(
                Mathf.Round(proposed.x / voxel) * voxel,
                Mathf.Round(proposed.y / voxel) * voxel,
                Mathf.Round(proposed.z / voxel) * voxel);
            ApplyFrame();
        }

        /// <summary>
        /// Queues a read-only snapshot immediately before the production
        /// InfiniTAM dispatch.  The matrices and textures are the exact objects
        /// bound to that accepted source frame, including delayed tracking
        /// frames; no current-frame substitute is allowed.
        /// </summary>
        internal void CaptureFusionPre(int attempt, int sourceFrame,
            int integration, uint dirtyEpoch, Texture rawDepth,
            Texture fusionDepth, Texture dilatedDepth, Texture edgeReason,
            Texture temporalReason, Matrix4x4[] projection, Matrix4x4[] view,
            Matrix4x4[] projectionInverse, Matrix4x4[] viewInverse,
            float minUpdateDistance, float maxUpdateDistance, uint writeEpoch)
        {
            if (!_initialized || !_recording || !_locked ||
                _fusionWitnesses.Count >= MaximumFusionWitnesses ||
                _activeFusionWitness != null) return;
            try
            {
                RenderTexture responsibility = _volume.TsdfResponsibilityVolume;
                RenderTexture votes = _volume.InfiniTamVoteWeightVolume;
                RenderTexture accumulated = _volume.InfiniTamAccumulatedTsdfVolume;
                ComputeBuffer writeEpochs = _volume.InfiniTamWriteEpochs;
                if (fusionDepth == null || dilatedDepth == null || edgeReason == null ||
                    responsibility == null || votes == null || accumulated == null ||
                    writeEpochs == null || projection == null || view == null ||
                    projectionInverse == null || viewInverse == null)
                    return;

                int3 roiMin = CalculateRoiMin();
                Texture temporalBinding = temporalReason != null
                    ? temporalReason : edgeReason;
                Texture rawBinding = rawDepth != null ? rawDepth : fusionDepth;
                _captureShader.SetTexture(_fusionPreKernel, TsdfId, _volume.Volume);
                _captureShader.SetTexture(_fusionPreKernel, ResponsibilityId, responsibility);
                _captureShader.SetTexture(_fusionPreKernel, VoteWeightVolumeId, votes);
                _captureShader.SetTexture(_fusionPreKernel, AccumulatedTsdfVolumeId, accumulated);
                _captureShader.SetTexture(_fusionPreKernel, RawDepthId, rawBinding);
                _captureShader.SetTexture(_fusionPreKernel, FusionDepthId, fusionDepth);
                _captureShader.SetTexture(_fusionPreKernel, DilatedDepthId, dilatedDepth);
                _captureShader.SetTexture(_fusionPreKernel, EdgeReasonId, edgeReason);
                _captureShader.SetTexture(_fusionPreKernel, TemporalReasonId, temporalBinding);
                _captureShader.SetBuffer(_fusionPreKernel, WriteEpochsId, writeEpochs);
                _captureShader.SetBuffer(_fusionPreKernel, FusionWitnessId,
                    _fusionWitnessBuffer);
                SetFusionWitnessCommon(roiMin, fusionDepth, projection, view,
                    projectionInverse, viewInverse, minUpdateDistance,
                    maxUpdateDistance, writeEpoch, rawDepth != null,
                    temporalReason != null);
                _captureShader.Dispatch(_fusionPreKernel,
                    CeilDiv(_roiCount.x, 4), CeilDiv(_roiCount.y, 4),
                    CeilDiv(_roiCount.z, 4));

                Camera camera = Camera.main;
                _activeFusionWitness = new FusionWitnessSample
                {
                    Serial = _fusionWitnessSerial++, Attempt = attempt,
                    SourceFrame = sourceFrame, Integration = integration,
                    UnityFrame = Time.frameCount, DirtyEpochBefore = dirtyEpoch,
                    DepthWidth = fusionDepth.width,
                    DepthHeight = fusionDepth.height,
                    WriteEpoch = writeEpoch, Time = Time.unscaledTimeAsDouble,
                    CameraPosition = camera != null
                        ? camera.transform.position : Vector3.zero,
                    CameraRotation = camera != null
                        ? camera.transform.rotation : Quaternion.identity,
                    RawAvailable = rawDepth != null,
                    TemporalAvailable = temporalReason != null,
                    Projection = (Matrix4x4[])projection.Clone(),
                    View = (Matrix4x4[])view.Clone(),
                    ProjectionInverse = (Matrix4x4[])projectionInverse.Clone(),
                    ViewInverse = (Matrix4x4[])viewInverse.Clone()
                };
            }
            catch (Exception e)
            {
                _activeFusionWitness = null;
                _lastError = e.Message;
                Logger.Warning("蓝框融合前证据失败（生产不受影响）: " + e.Message);
            }
        }

        /// <summary>Queues the matching post-write state and an async GPU copy.</summary>
        internal void CaptureFusionPost(uint dirtyEpoch)
        {
            FusionWitnessSample sample = _activeFusionWitness;
            _activeFusionWitness = null;
            if (sample == null) return;
            try
            {
                _captureShader.SetTexture(_fusionPostKernel, TsdfId, _volume.Volume);
                _captureShader.SetTexture(_fusionPostKernel, ResponsibilityId,
                    _volume.TsdfResponsibilityVolume);
                _captureShader.SetTexture(_fusionPostKernel, VoteWeightVolumeId,
                    _volume.InfiniTamVoteWeightVolume);
                _captureShader.SetTexture(_fusionPostKernel, AccumulatedTsdfVolumeId,
                    _volume.InfiniTamAccumulatedTsdfVolume);
                _captureShader.SetBuffer(_fusionPostKernel, WriteEpochsId,
                    _volume.InfiniTamWriteEpochs);
                _captureShader.SetBuffer(_fusionPostKernel, FusionWitnessId,
                    _fusionWitnessBuffer);
                _captureShader.Dispatch(_fusionPostKernel,
                    CeilDiv(_roiCount.x, 4), CeilDiv(_roiCount.y, 4),
                    CeilDiv(_roiCount.z, 4));
                sample.DirtyEpochAfter = dirtyEpoch;
                sample.Readback = CaptureAuditIO.ReadAsync(_fusionWitnessBuffer,
                    checked(_cellCount * FusionWitnessStrideBytes));
                _fusionWitnesses.Add(sample);
            }
            catch (Exception e)
            {
                _lastError = e.Message;
                Logger.Warning("蓝框融合后证据失败（生产不受影响）: " + e.Message);
            }
        }

        private void SetFusionWitnessCommon(int3 roiMin, Texture fusionDepth,
            Matrix4x4[] projection, Matrix4x4[] view,
            Matrix4x4[] projectionInverse, Matrix4x4[] viewInverse,
            float minUpdateDistance, float maxUpdateDistance, uint writeEpoch,
            bool rawAvailable, bool temporalAvailable)
        {
            _captureShader.SetInts(RoiMinId, roiMin.x, roiMin.y, roiMin.z);
            _captureShader.SetInts(RoiCountId, _roiCount.x, _roiCount.y, _roiCount.z);
            int3 volumeCount = _volume.VoxelCount;
            _captureShader.SetInts(VolumeCountId,
                volumeCount.x, volumeCount.y, volumeCount.z);
            _captureShader.SetInts(FusionDepthSizeId,
                fusionDepth.width, fusionDepth.height);
            _captureShader.SetMatrixArray(ProjectionId, projection);
            _captureShader.SetMatrixArray(ViewId, view);
            _captureShader.SetMatrixArray(ProjectionInverseId, projectionInverse);
            _captureShader.SetMatrixArray(ViewInverseId, viewInverse);
            _captureShader.SetFloat(VoxelSizeId, _volume.VoxelSize);
            _captureShader.SetFloat(VoxelDistanceId, _volume.VoxelDistance);
            _captureShader.SetFloat(MinUpdateDistanceId, minUpdateDistance);
            _captureShader.SetFloat(MaxUpdateDistanceId, maxUpdateDistance);
            _captureShader.SetInt(ExpectedWriteEpochId, unchecked((int)writeEpoch));
            _captureShader.SetInt(RawDepthAvailableId, rawAvailable ? 1 : 0);
            _captureShader.SetInt(TemporalReasonAvailableId,
                temporalAvailable ? 1 : 0);
        }

        private int3 CalculateRoiMin()
        {
            int3 volumeCount = _volume.VoxelCount;
            return new int3(
                Mathf.FloorToInt(_center.x / _volume.VoxelSize +
                                 volumeCount.x * 0.5f) - _roiCount.x / 2,
                Mathf.FloorToInt(_center.y / _volume.VoxelSize +
                                 volumeCount.y * 0.5f) - _roiCount.y / 2,
                Mathf.FloorToInt(_center.z / _volume.VoxelSize +
                                 volumeCount.z * 0.5f) - _roiCount.z / 2);
        }

        private async Task CaptureOneAsync()
        {
            HistorySample sample = null;
            try
            {
                Bounds roi = new Bounds(_center, FrameSizeMetres);
                int3 volumeCount = _volume.VoxelCount;
                int3 roiMin = CalculateRoiMin();

                _captureShader.SetTexture(_captureKernel, TsdfId, _volume.Volume);
                RenderTexture responsibility = _volume.TsdfResponsibilityVolume;
                if (responsibility == null || !responsibility.IsCreated())
                    throw new InvalidOperationException("TSDF responsibility sidecar unavailable");
                _captureShader.SetTexture(_captureKernel, ResponsibilityId, responsibility);
                _captureShader.SetBuffer(_captureKernel, CellsId, _cellBuffer);
                _captureShader.SetInts(RoiMinId, roiMin.x, roiMin.y, roiMin.z);
                _captureShader.SetInts(RoiCountId, _roiCount.x, _roiCount.y, _roiCount.z);
                _captureShader.SetInts(VolumeCountId, volumeCount.x, volumeCount.y, volumeCount.z);
                _captureShader.SetInt(ResponsibilityAvailableId, 1);
                _captureShader.Dispatch(_captureKernel,
                    CeilDiv(_roiCount.x, 4), CeilDiv(_roiCount.y, 4), CeilDiv(_roiCount.z, 4));

                sample = new HistorySample
                {
                    Serial = _sampleSerial++, UnityFrame = Time.frameCount,
                    PlatformFrame = DepthCapture.Instance != null
                        ? DepthCapture.Instance.CurrentPlatformFrame : -1,
                    Integration = _volume.IntegrationCount, DirtyEpoch = _volume.DirtyEpoch,
                    Time = Time.unscaledTimeAsDouble,
                    CameraPosition = Camera.main != null ? Camera.main.transform.position : Vector3.zero,
                    CameraRotation = Camera.main != null ? Camera.main.transform.rotation : Quaternion.identity
                };
                Task<byte[]> cellsRead = CaptureAuditIO.ReadAsync(_cellBuffer,
                    checked(_cellCount * CellStrideBytes));
                List<CaptureFront> fronts = MeshExtractor.Instance != null
                    ? MeshExtractor.Instance.CaptureLockedRoiFronts(roi)
                    : new List<CaptureFront>();
                foreach (CaptureFront front in fronts) front.RequestReads();
                sample.Cells = await cellsRead;
                foreach (CaptureFront front in fronts)
                {
                    await Task.WhenAll(front.VertexRead, front.IndexRead, front.ArgsRead,
                        front.CrossingRead, front.CrossingSpanRead, front.RawSurfaceNetsRead);
                    FilteredFront filtered = FilterFront(front, roi);
                    if (filtered != null && filtered.IndexCount > 0)
                        sample.Fronts.Add(filtered);
                }
                _history.Add(sample);
                if (_history.Count > MaximumSamples)
                {
                    // Preserve the integration=0 empty-volume anchor for the
                    // entire package; rotate the oldest later history row.
                    int remove = _history[0].Integration == 0 &&
                                 _history.Count > 1 ? 1 : 0;
                    _history.RemoveAt(remove);
                }
            }
            catch (Exception e)
            {
                _lastError = e.Message;
                Logger.Warning("蓝框历史采样失败（生产不受影响）: " + e.Message);
            }
        }

        internal async Task ExportAsync(string directory)
        {
            StopForSeal();
            if (_pendingSample != null) await _pendingSample;
            foreach (FusionWitnessSample witness in _fusionWitnesses)
            {
                if (witness.Readback == null) continue;
                witness.Bytes = await witness.Readback;
            }
            if (!_locked) throw new InvalidOperationException("蓝框尚未锁定");
            if (_history.Count == 0)
                throw new InvalidOperationException(string.IsNullOrEmpty(_lastError)
                    ? "蓝框内没有历史样本" : _lastError);
            List<HistorySample> snapshot = new List<HistorySample>(_history);
            List<FusionWitnessSample> fusionSnapshot =
                new List<FusionWitnessSample>(_fusionWitnesses);
            Vector3 center = _center;
            int3 roiCount = _roiCount;
            float voxel = _volume.VoxelSize;
            await Task.Run(() => WritePackage(directory, snapshot, fusionSnapshot,
                center, roiCount, voxel));
        }

        private static void WritePackage(string directory, List<HistorySample> samples,
            List<FusionWitnessSample> fusionWitnesses, Vector3 center,
            int3 roiCount, float voxel)
        {
            Directory.CreateDirectory(directory);
            string schema = "{\n" +
                "  \"schema\": \"scancover.locked_roi_history.v2\",\n" +
                "  \"scope\": \"same-frame projected depth evidence, voxels and published Front geometry inside the locked blue frame\",\n" +
                "  \"coordinates\": \"Unity world metres; x-fastest voxel cells\",\n" +
                "  \"center\": [" + F(center.x) + "," + F(center.y) + "," + F(center.z) + "],\n" +
                "  \"sizeMetres\": [0.8,1.2,0.8],\n" +
                "  \"voxelCount\": [" + roiCount.x + "," + roiCount.y + "," + roiCount.z + "],\n" +
                "  \"voxelSizeMetres\": " + F(voxel) + ",\n" +
                "  \"cellLayout\": \"32 bytes: int3 globalVoxel, uint valid, float tsdf, float weight, uint2 responsibility\",\n" +
                "  \"fusionWitnessLayout\": \"128 bytes: int3 voxel, uint flags, int2 projectedPixel, raw/fusion/dilated depth, edge/temporal reason, projected ranges and observed TSDF, pre/post TSDF-weight-vote-accumulated TSDF-responsibility, exact write claim, raw/dilated metric ranges and deltas\",\n" +
                "  \"fusionWitnessRule\": \"one pre/post record for every accepted integration while the blue frame is locked; raw depth belongs to the same retained source frame\",\n" +
                "  \"fusionWitnessMatrices\": \"512 bytes float32 row-major: projection eyes 0/1, view eyes 0/1, projectionInverse eyes 0/1, viewInverse eyes 0/1\",\n" +
                "  \"fusionWitnessFlags\": \"bits 0..15 = valid coordinate, in projection, raw valid, fusion valid, dilated valid, voxel range, depth range, truncation gate, pre observed, post observed, exact write claim, state changed, sign changed, edge reason, temporal available, temporal reason\",\n" +
                "  \"fusionWitnessCounting\": \"fusion_witness.csv counts voxel rows; multiple ROI voxels may project to the same depth pixel\",\n" +
                "  \"meshVertexLayout\": \"32-byte production GPUVertex; crossing/span/raw arrays stay vertex-aligned\",\n" +
                "  \"historyRule\": \"0.5 second cadence, same fixed world frame while operator changes viewpoint\",\n" +
                "  \"authority\": \"read-only observer; never consumed by production\"\n" +
                "}\n";
            File.WriteAllText(Path.Combine(directory, "schema.json"), schema,
                new UTF8Encoding(false));
            const string witnessFields =
                "offset,type,name\n" +
                "0,int3,globalVoxel\n12,uint,flags\n16,int2,projectedPixel\n" +
                "24,float,platformRawDepthNdc\n28,float,fusionDepthNdc\n" +
                "32,float,dilatedDepthNdc\n36,uint,edgeReason\n" +
                "40,uint,temporalReason\n44,float,voxelEyeDistanceMetres\n" +
                "48,float,depthEyeDistanceMetres\n52,float,observedTsdf\n" +
                "56,float,preTsdf\n60,float,preWeight\n64,float,preVoteWeight\n" +
                "68,float,preAccumulatedTsdf\n72,uint,preResponsibility0\n" +
                "76,uint,preResponsibility1\n80,float,postTsdf\n" +
                "84,float,postWeight\n88,float,postVoteWeight\n" +
                "92,float,postAccumulatedTsdf\n96,uint,postResponsibility0\n" +
                "100,uint,postResponsibility1\n104,uint,writeClaim\n" +
                "108,float,platformRawDepthEyeDistanceMetres\n" +
                "112,float,dilatedDepthEyeDistanceMetres\n" +
                "116,float,platformRawMinusFusionMillimetres\n" +
                "120,float,dilatedMinusFusionMillimetres\n124,uint,reserved\n";
            File.WriteAllText(Path.Combine(directory, "fusion_witness_fields.csv"),
                witnessFields, new UTF8Encoding(false));
            var sampleCsv = new StringBuilder("sample,unityFrame,platformFrame,time,integration,dirtyEpoch,cameraX,cameraY,cameraZ,rotX,rotY,rotZ,rotW,cellsFile,frontCount\n");
            var frontCsv = new StringBuilder("sample,block,generation,publishedEpoch,vertices,indices,verticesFile,indicesFile,crossingsFile,crossingSpansFile,rawSurfaceNetsFile\n");
            foreach (HistorySample sample in samples)
            {
                string sampleName = "sample_" + sample.Serial.ToString("D4", CultureInfo.InvariantCulture);
                string sampleDir = Path.Combine(directory, sampleName);
                Directory.CreateDirectory(sampleDir);
                File.WriteAllBytes(Path.Combine(sampleDir, "cells.bin"), sample.Cells);
                sampleCsv.Append(sample.Serial).Append(',').Append(sample.UnityFrame).Append(',')
                    .Append(sample.PlatformFrame).Append(',').Append(D(sample.Time)).Append(',')
                    .Append(sample.Integration).Append(',').Append(sample.DirtyEpoch).Append(',')
                    .Append(F(sample.CameraPosition.x)).Append(',').Append(F(sample.CameraPosition.y)).Append(',')
                    .Append(F(sample.CameraPosition.z)).Append(',').Append(F(sample.CameraRotation.x)).Append(',')
                    .Append(F(sample.CameraRotation.y)).Append(',').Append(F(sample.CameraRotation.z)).Append(',')
                    .Append(F(sample.CameraRotation.w)).Append(',').Append(sampleName).Append("/cells.bin,")
                    .Append(sample.Fronts.Count).AppendLine();
                foreach (FilteredFront front in sample.Fronts)
                {
                    string stem = sampleName + "/front_" + front.Block;
                    File.WriteAllBytes(Path.Combine(directory, stem + ".vertices.bin"), front.Vertices);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".indices.bin"), front.Indices);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".crossings.u32x4.bin"), front.Crossings);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".crossing_spans.u32x3.bin"), front.CrossingSpans);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".raw_surface_nets.f32x3.bin"), front.RawSurfaceNets);
                    frontCsv.Append(sample.Serial).Append(',').Append(front.Block).Append(',')
                        .Append(front.Generation).Append(',').Append(front.PublishedEpoch).Append(',')
                        .Append(front.VertexCount).Append(',').Append(front.IndexCount).Append(',')
                        .Append(stem).Append(".vertices.bin,").Append(stem).Append(".indices.bin,")
                        .Append(stem).Append(".crossings.u32x4.bin,")
                        .Append(stem).Append(".crossing_spans.u32x3.bin,")
                        .Append(stem).AppendLine(".raw_surface_nets.f32x3.bin");
                }
            }
            File.WriteAllText(Path.Combine(directory, "samples.csv"), sampleCsv.ToString(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(directory, "fronts.csv"), frontCsv.ToString(), new UTF8Encoding(false));

            string witnessDir = Path.Combine(directory, "fusion_witness");
            Directory.CreateDirectory(witnessDir);
            var witnessCsv = new StringBuilder(
                "witness,attempt,sourceFrame,integration,unityFrame,time,dirtyEpochBefore,dirtyEpochAfter,writeEpoch,depthWidth,depthHeight,rawAvailable,temporalAvailable,cameraX,cameraY,cameraZ,rotX,rotY,rotZ,rotW,valid,inProjection,rawValid,fusionValid,claimed,changed,signChanged,claimedEdgeReason,claimedTemporalReason,rawFusionPairs,rawFusionMeanAbsMm,rawFusionMaxAbsMm,dilatedFusionPairs,dilatedFusionMeanAbsMm,dilatedFusionMaxAbsMm,file,matricesFile\n");
            foreach (FusionWitnessSample witness in fusionWitnesses)
            {
                byte[] bytes = witness.Bytes ?? Array.Empty<byte>();
                if (bytes.Length != roiCount.x * roiCount.y * roiCount.z *
                    FusionWitnessStrideBytes)
                    throw new InvalidDataException("ROI fusion witness byte count mismatch");
                int valid = 0, projected = 0, rawValid = 0, fusionValid = 0;
                int claimed = 0, changed = 0, signChanged = 0;
                int claimedEdge = 0, claimedTemporal = 0;
                int rawPairs = 0, dilatedPairs = 0;
                double rawAbsSum = 0.0, dilatedAbsSum = 0.0;
                float rawAbsMax = 0f, dilatedAbsMax = 0f;
                for (int offset = 0; offset < bytes.Length;
                     offset += FusionWitnessStrideBytes)
                {
                    uint flags = BitConverter.ToUInt32(bytes, offset + 12);
                    if ((flags & (1u << 0)) != 0) valid++;
                    if ((flags & (1u << 1)) != 0) projected++;
                    if ((flags & (1u << 2)) != 0) rawValid++;
                    if ((flags & (1u << 3)) != 0) fusionValid++;
                    if ((flags & (1u << 10)) != 0) claimed++;
                    if ((flags & (1u << 11)) != 0) changed++;
                    if ((flags & (1u << 12)) != 0) signChanged++;
                    bool wasClaimed = (flags & (1u << 10)) != 0;
                    if (wasClaimed && (flags & (1u << 13)) != 0) claimedEdge++;
                    if (wasClaimed && (flags & (1u << 15)) != 0)
                        claimedTemporal++;
                    if (wasClaimed && (flags & (1u << 2)) != 0 &&
                        (flags & (1u << 3)) != 0)
                    {
                        float delta = Math.Abs(BitConverter.ToSingle(bytes,
                            offset + 116));
                        rawPairs++;
                        rawAbsSum += delta;
                        rawAbsMax = Math.Max(rawAbsMax, delta);
                    }
                    if (wasClaimed && (flags & (1u << 4)) != 0 &&
                        (flags & (1u << 3)) != 0)
                    {
                        float delta = Math.Abs(BitConverter.ToSingle(bytes,
                            offset + 120));
                        dilatedPairs++;
                        dilatedAbsSum += delta;
                        dilatedAbsMax = Math.Max(dilatedAbsMax, delta);
                    }
                }
                string file = "witness_" + witness.Serial.ToString("D4",
                    CultureInfo.InvariantCulture) + ".bin";
                string matricesFile = "witness_" + witness.Serial.ToString("D4",
                    CultureInfo.InvariantCulture) + ".matrices.f32";
                File.WriteAllBytes(Path.Combine(witnessDir, file), bytes);
                File.WriteAllBytes(Path.Combine(witnessDir, matricesFile),
                    MatrixBytes(witness));
                witnessCsv.Append(witness.Serial).Append(',').Append(witness.Attempt)
                    .Append(',').Append(witness.SourceFrame).Append(',')
                    .Append(witness.Integration).Append(',').Append(witness.UnityFrame)
                    .Append(',').Append(D(witness.Time)).Append(',')
                    .Append(witness.DirtyEpochBefore).Append(',')
                    .Append(witness.DirtyEpochAfter).Append(',')
                    .Append(witness.WriteEpoch).Append(',')
                    .Append(witness.DepthWidth).Append(',')
                    .Append(witness.DepthHeight).Append(',')
                    .Append(witness.RawAvailable ? 1 : 0).Append(',')
                    .Append(witness.TemporalAvailable ? 1 : 0).Append(',')
                    .Append(F(witness.CameraPosition.x)).Append(',')
                    .Append(F(witness.CameraPosition.y)).Append(',')
                    .Append(F(witness.CameraPosition.z)).Append(',')
                    .Append(F(witness.CameraRotation.x)).Append(',')
                    .Append(F(witness.CameraRotation.y)).Append(',')
                    .Append(F(witness.CameraRotation.z)).Append(',')
                    .Append(F(witness.CameraRotation.w)).Append(',')
                    .Append(valid).Append(',').Append(projected).Append(',')
                    .Append(rawValid).Append(',').Append(fusionValid).Append(',')
                    .Append(claimed).Append(',').Append(changed).Append(',')
                    .Append(signChanged).Append(',').Append(claimedEdge).Append(',')
                    .Append(claimedTemporal).Append(',').Append(rawPairs).Append(',')
                    .Append(D(rawPairs > 0 ? rawAbsSum / rawPairs : 0.0)).Append(',')
                    .Append(F(rawAbsMax)).Append(',').Append(dilatedPairs).Append(',')
                    .Append(D(dilatedPairs > 0
                        ? dilatedAbsSum / dilatedPairs : 0.0)).Append(',')
                    .Append(F(dilatedAbsMax)).Append(",fusion_witness/")
                    .Append(file).Append(",fusion_witness/")
                    .Append(matricesFile).AppendLine();
            }
            File.WriteAllText(Path.Combine(directory, "fusion_witness.csv"),
                witnessCsv.ToString(), new UTF8Encoding(false));
        }

        private static FilteredFront FilterFront(CaptureFront source, Bounds roi)
        {
            byte[] vertices = source.VertexRead.Result;
            byte[] indices = source.IndexRead.Result;
            if (vertices.Length != source.Vertices * 32 || indices.Length != source.Indices * 4)
                throw new InvalidDataException("ROI Front readback byte count mismatch");
            int[] remap = new int[source.Vertices];
            Array.Fill(remap, -1);
            var selected = new List<int>();
            Vector3 min = roi.min, max = roi.max;
            for (int i = 0; i < source.Vertices; i++)
            {
                int o = i * 32;
                float x = BitConverter.ToSingle(vertices, o);
                float y = BitConverter.ToSingle(vertices, o + 4);
                float z = BitConverter.ToSingle(vertices, o + 8);
                if (x < min.x || x > max.x || y < min.y || y > max.y || z < min.z || z > max.z)
                    continue;
                remap[i] = selected.Count;
                selected.Add(i);
            }
            var keptIndices = new List<uint>();
            for (int i = 0; i + 2 < source.Indices; i += 3)
            {
                int a = checked((int)BitConverter.ToUInt32(indices, i * 4));
                int b = checked((int)BitConverter.ToUInt32(indices, (i + 1) * 4));
                int c = checked((int)BitConverter.ToUInt32(indices, (i + 2) * 4));
                if (a < 0 || b < 0 || c < 0 || a >= remap.Length || b >= remap.Length || c >= remap.Length ||
                    remap[a] < 0 || remap[b] < 0 || remap[c] < 0) continue;
                keptIndices.Add((uint)remap[a]);
                keptIndices.Add((uint)remap[b]);
                keptIndices.Add((uint)remap[c]);
            }
            if (keptIndices.Count == 0) return null;
            return new FilteredFront
            {
                Block = source.Block, Generation = source.Generation,
                PublishedEpoch = source.Published,
                Vertices = Gather(vertices, 32, selected),
                Indices = UIntBytes(keptIndices),
                Crossings = Gather(source.CrossingRead.Result, 16, selected),
                CrossingSpans = Gather(source.CrossingSpanRead.Result, 12, selected),
                RawSurfaceNets = Gather(source.RawSurfaceNetsRead.Result, 12, selected)
            };
        }

        private static byte[] Gather(byte[] source, int stride, List<int> selected)
        {
            byte[] result = new byte[checked(selected.Count * stride)];
            for (int i = 0; i < selected.Count; i++)
                Buffer.BlockCopy(source, selected[i] * stride, result, i * stride, stride);
            return result;
        }

        private static byte[] UIntBytes(List<uint> values)
        {
            byte[] result = new byte[checked(values.Count * 4)];
            for (int i = 0; i < values.Count; i++)
                Buffer.BlockCopy(BitConverter.GetBytes(values[i]), 0, result, i * 4, 4);
            return result;
        }

        private static byte[] MatrixBytes(FusionWitnessSample witness)
        {
            Matrix4x4[][] groups =
            {
                witness.Projection, witness.View,
                witness.ProjectionInverse, witness.ViewInverse
            };
            byte[] bytes = new byte[checked(4 * 2 * 16 * sizeof(float))];
            int offset = 0;
            foreach (Matrix4x4[] group in groups)
            {
                for (int eye = 0; eye < 2; eye++)
                {
                    Matrix4x4 matrix = group != null && eye < group.Length
                        ? group[eye] : Matrix4x4.identity;
                    for (int row = 0; row < 4; row++)
                    for (int column = 0; column < 4; column++)
                    {
                        byte[] scalar = BitConverter.GetBytes(matrix[row, column]);
                        Buffer.BlockCopy(scalar, 0, bytes, offset, sizeof(float));
                        offset += sizeof(float);
                    }
                }
            }
            return bytes;
        }

        private void CreateFrame(int layer)
        {
            _frameRoot = new GameObject("[ScanCover] Locked ROI History Frame");
            _frameRoot.layer = layer;
            _frameRoot.transform.SetParent(transform, false);
            Shader shader = Resources.Load<Shader>("LocalFineRoiLine");
            if (shader == null) throw new InvalidOperationException("LocalFineRoiLine shader missing");
            _frameMaterial = new Material(shader) { name = "[ScanCover] Locked ROI History Frame" };
            _frameMaterial.renderQueue = 3990;
            _edges = new LineRenderer[12];
            for (int i = 0; i < _edges.Length; i++)
            {
                GameObject edge = new GameObject("Edge " + i.ToString("D2"));
                edge.layer = layer;
                edge.transform.SetParent(_frameRoot.transform, false);
                LineRenderer line = edge.AddComponent<LineRenderer>();
                line.useWorldSpace = true;
                line.positionCount = 2;
                line.widthMultiplier = 0.014f;
                line.numCapVertices = 2;
                line.numCornerVertices = 1;
                line.shadowCastingMode = ShadowCastingMode.Off;
                line.receiveShadows = false;
                line.sharedMaterial = _frameMaterial;
                line.startColor = FrameColor;
                line.endColor = FrameColor;
                _edges[i] = line;
            }
            ApplyFrame();
        }

        private void ApplyFrame()
        {
            if (_edges == null) return;
            Vector3 half = FrameSizeMetres * 0.5f;
            for (int i = 0; i < 8; i++)
                _corners[i] = _center + new Vector3(
                    (i & 1) == 0 ? -half.x : half.x,
                    (i & 2) == 0 ? -half.y : half.y,
                    (i & 4) == 0 ? -half.z : half.z);
            for (int i = 0; i < 12; i++)
            {
                _edges[i].SetPosition(0, _corners[EdgePairs[i, 0]]);
                _edges[i].SetPosition(1, _corners[EdgePairs[i, 1]]);
            }
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
            _cellBuffer?.Release();
            _cellBuffer = null;
            _fusionWitnessBuffer?.Release();
            _fusionWitnessBuffer = null;
            if (_frameMaterial != null) Destroy(_frameMaterial);
            _frameMaterial = null;
        }

        private void OnDisable()
        {
            StopForSeal();
            if (Instance == this) Instance = null;
        }

        private static int CeilDiv(int value, int divisor) => (value + divisor - 1) / divisor;
        private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
        private static string D(double value) => value.ToString("R", CultureInfo.InvariantCulture);
    }

    public partial class MeshExtractor
    {
        // The blue locked-ROI recorder is retained as an isolated forensic
        // module, but production never creates, starts or waits on it.
        internal bool LockedRoiHistoryCaptureEnabled => false;

        internal List<CaptureFront> CaptureLockedRoiFronts(Bounds roi)
        {
            return _infiniTamBlocks != null
                ? _infiniTamBlocks.CaptureFronts(roi)
                : new List<CaptureFront>();
        }
    }

    internal sealed partial class InfiniTamBlockMeshPipeline
    {
        internal List<CaptureFront> CaptureFronts(Bounds roi)
        {
            var result = new List<CaptureFront>();
            float voxel = _volume.VoxelSize;
            float3 halfVolume = (float3)_volume.VoxelCount * (voxel * 0.5f);
            foreach (Block block in _blocks)
            {
                if (block.Front == null || block.Front.KnownDrawVertexCount <= 0) continue;
                float3 min = (float3)block.CoreMin * voxel - halfVolume;
                float3 max = (float3)(block.CoreMax + 1) * voxel - halfVolume;
                Bounds blockBounds = new Bounds();
                blockBounds.SetMinMax((Vector3)min, (Vector3)max);
                if (!blockBounds.Intersects(roi)) continue;
                result.Add(new CaptureFront
                {
                    Block = block.Index, Generation = _generation,
                    Target = block.TargetEpoch, Candidate = block.CandidateEpoch,
                    Published = block.PublishedEpoch, Source = block.Front,
                    Vertices = block.Front.KnownVertexCount,
                    Indices = block.Front.KnownDrawVertexCount,
                    Visible = block.Renderer != null && block.Renderer.RenderVisible,
                    Pending = block.Queued || block.CommitPending || block.ReadyToPublish,
                    SubmittedIndices = block.Renderer?.LastSubmittedVertexCount ?? 0,
                    SubmittedFrame = block.Renderer?.LastSubmittedFrame ?? -1
                });
            }
            return result;
        }
    }
}



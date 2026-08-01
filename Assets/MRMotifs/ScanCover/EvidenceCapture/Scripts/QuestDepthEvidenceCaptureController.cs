using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR.EnvironmentDepth;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.UI;
using UnityEngine.XR;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace ScanCover.EvidenceCapture
{
    [DefaultExecutionOrder(500)]
    [DisallowMultipleComponent]
    public sealed class QuestDepthEvidenceCaptureController : MonoBehaviour
    {
        private const string SchemaName = "ScanCoverDepthEvidence/v3";
        private const int EyeCount = 2;

        private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        private static readonly int ReprojectionMatricesId = Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
        private static readonly int ZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
        private static readonly int LeftEyeWorldPositionId = Shader.PropertyToID("_LeftEyeWorldPosition");
        private static readonly int RightEyeWorldPositionId = Shader.PropertyToID("_RightEyeWorldPosition");

        [Header("采集依赖")]
        [SerializeField] private EnvironmentDepthManager environmentDepthManager;
        [SerializeField] private Camera sourceCamera;
        [SerializeField] private ComputeShader evidenceCompute;
        [SerializeField] private QuestSystemRoomMeshCapture systemRoomMeshCapture;

        [Header("采集节奏")]
        [SerializeField, Min(0.05f)] private float captureIntervalSeconds = 0.2f;
        [SerializeField, Min(60f)] private float sessionDurationSeconds = 600f;
        [SerializeField, Min(1)] private int maxFramesPerSession = 5000;
        [SerializeField, Range(1, 8)] private int maxGpuFramesInFlight = 3;
        [SerializeField, Range(1, 64)] private int maxWriterQueueFrames = 24;
        [SerializeField] private bool startRecordingOnEnable;
        [SerializeField, Min(0.001f)] private float neighbourDepthToleranceMetres = 0.08f;

        [Header("精简提示")]
        [SerializeField] private bool showHud = true;
        [SerializeField, Range(0.5f, 2f)] private float hudScale = 1f;
        [SerializeField] private bool remedialCaptureGuidance = true;

        private int _freezeKernel = -1;
        private int _geometryKernel = -1;
        private int _normalKernel = -1;
        private int _copyR32Kernel = -1;
        private int _copyRgba32Kernel = -1;
        private float _nextCaptureTime;
        private double _sessionStartRealtime;
        private double _sessionDeadlineRealtime;
        private double _captureStopRealtime = -1.0;
        private bool _automaticStop;
        private string _stopReason = string.Empty;
        private bool _recording;
        private bool _closing;
        private int _nextFrameIndex;
        private int _gpuInFlight;
        private int _readbackFailedFrames;
        private int _sourceUnavailableCount;
        private int _queuePauseCount;
        private int _discardedAfterFatalFrames;
        private bool _sessionFailed;
        private int _fatalFailureFrame = -1;
        private string _fatalFailureCode = string.Empty;
        private string _fatalFailureDetail = string.Empty;
        private bool _runtimeSummaryWritten;
        private bool _systemRoomMeshExportFinalized;
        private string _lastRuntimeIssueCode = string.Empty;
        private double _lastRuntimeIssueRealtime = -1000.0;
        private string _sessionDirectory = string.Empty;
        private string _lastIssue = "等待深度";
        private EvidenceSessionWriter _writer;
        private Text _hudText;
        private Canvas _hudCanvas;
        private float _lastSelfConsistentNormalCoverage;
        private float _lastEdgeRiskRatio;
        private bool _hasQualitySummary;
        private readonly HashSet<int> _nearViewSectors = new HashSet<int>();
        private readonly HashSet<int> _middleViewSectors = new HashSet<int>();
        private readonly HashSet<int> _farViewSectors = new HashSet<int>();
        private readonly HashSet<int> _grazingViewSectors = new HashSet<int>();

        private const int NearSectorGoal = 3;
        private const int MiddleSectorGoal = 8;
        private const int FarSectorGoal = 4;
        private const int GrazingSectorGoal = 4;

        private sealed class PendingGpuFrame
        {
            public EvidenceFramePayload Payload;
            public FrameSnapshot Snapshot;
            public readonly List<RenderTexture> Textures = new List<RenderTexture>(8);
            public readonly List<ComputeBuffer> ReadbackBuffers = new List<ComputeBuffer>(8);
            public int RemainingRequests;
            public int FailedRequests;
            public string ReadbackErrors = string.Empty;
            public double LastReadbackRealtime;
        }

        private sealed class FrameSnapshot
        {
            public int FrameIndex;
            public int UnityFrame;
            public long RequestUtcTicks;
            public string RequestUtcIso;
            public double RequestRealtime;
            public double RequestDspTime;
            public float TimeValue;
            public float UnscaledTime;
            public int Width;
            public int Height;
            public int SourceEyeCount;
            public string SourceType;
            public string SourceGraphicsFormat;
            public string SourceDimension;
            public int SourceMipCount;
            public FilterMode SourceFilterMode;
            public TextureWrapMode SourceWrapMode;
            public Vector4 ZBufferParams;
            public Matrix4x4[] Reprojection = new Matrix4x4[EyeCount];
            public Matrix4x4[] InverseReprojection = new Matrix4x4[EyeCount];
            public Matrix4x4[] StereoView = new Matrix4x4[EyeCount];
            public Matrix4x4[] StereoProjection = new Matrix4x4[EyeCount];
            public Vector3[] EyeWorldPositions = new Vector3[EyeCount];
            public Matrix4x4 CameraToWorld;
            public Matrix4x4 WorldToCamera;
            public Matrix4x4 Projection;
            public Matrix4x4 NonJitteredProjection;
            public Vector3 CameraPosition;
            public Quaternion CameraRotation;
            public float NearClip;
            public float FarClip;
            public float FieldOfView;
            public float Aspect;
            public int CameraPixelWidth;
            public int CameraPixelHeight;
            public TrackingSnapshot Tracking;
        }

        private struct TrackingSnapshot
        {
            public XrNodeSnapshot Head;
            public XrNodeSnapshot CenterEye;
            public XrNodeSnapshot LeftEye;
            public XrNodeSnapshot RightEye;
        }

        private struct XrNodeSnapshot
        {
            public bool DeviceValid;
            public bool IsTracked;
            public InputTrackingState TrackingState;
            public Vector3 Position;
            public Quaternion Rotation;
            public Vector3 Velocity;
            public Vector3 AngularVelocity;
        }

        public bool IsRecording => _recording;
        public string SessionDirectory => _sessionDirectory;

        public void ConfigureForScene(ComputeShader shader)
        {
            evidenceCompute = shader;
            ResolveDependencies();
            ResolveKernels();
        }

        private void Awake()
        {
            ResolveDependencies();
            ResolveKernels();
        }

        private void OnEnable()
        {
            if (startRecordingOnEnable)
            {
                StartSession();
            }
        }

        private void Update()
        {
            ResolveDependencies();
            EnsureHud();
            HandleInput();
            DrainWriterResults();

            if (_recording && remedialCaptureGuidance && RemedialProgress >= 1f && (_writer?.WrittenCount ?? 0) >= 48)
            {
                StopSessionInternal(
                    automatic: true,
                    reason: "coverage_complete",
                    issue: "补采覆盖目标已齐，正在自动存盘");
            }
            else if (_recording && Time.realtimeSinceStartupAsDouble >= _sessionDeadlineRealtime)
            {
                StopSessionInternal(
                    automatic: true,
                    reason: "safety_timeout",
                    issue: $"已到 {sessionDurationSeconds:0} 秒安全上限，正在自动存盘");
            }

            if (_recording && Time.unscaledTime >= _nextCaptureTime)
            {
                _nextCaptureTime = Time.unscaledTime + captureIntervalSeconds;
                TryCaptureFrame();
            }

            if (_recording && _nextFrameIndex >= maxFramesPerSession)
            {
                StopSessionInternal(
                    automatic: true,
                    reason: "frame_limit",
                    issue: "已达到本轮帧数上限，正在自动存盘");
            }

            if (_closing && _gpuInFlight == 0 && (_writer == null || _writer.PendingCount == 0))
            {
                _writer?.RequestStop();
                if (_writer == null || _writer.IsFinished)
                {
                    _closing = false;
                    if (!_systemRoomMeshExportFinalized)
                    {
                        systemRoomMeshCapture?.ExportForSession(_sessionDirectory);
                        _systemRoomMeshExportFinalized = true;
                    }
                    WriteRuntimeSummary();
                    _lastIssue = _sessionFailed
                        ? "本轮失败：" + _fatalFailureCode
                        : _writer != null && _writer.FailedCount > 0
                            ? "收尾完成，有写盘错误"
                            : "本轮已完整保存";
                }
            }

            UpdateHud();
        }

        private void OnDisable()
        {
            if (_recording) StopSession();
        }

        private void OnApplicationQuit()
        {
            _recording = false;
            _writer?.RequestStop();
            _writer?.Dispose();
            _writer = null;
        }

        public void StartSession()
        {
            if (_recording || _closing)
            {
                return;
            }

            ResolveDependencies();
            ResolveKernels();
            if (evidenceCompute == null || _freezeKernel < 0 || _geometryKernel < 0 || _normalKernel < 0 ||
                _copyR32Kernel < 0 || _copyRgba32Kernel < 0)
            {
                _lastIssue = "采集着色器未配置";
                return;
            }

            string root = Path.Combine(Application.persistentDataPath, "ScanCoverDepthEvidence");
            string sessionName = "Evidence_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            _sessionDirectory = Path.Combine(root, sessionName);
            Directory.CreateDirectory(_sessionDirectory);
            WriteSessionDescriptor();
            systemRoomMeshCapture?.BeginSession(_sessionDirectory);

            _writer?.Dispose();
            _writer = new EvidenceSessionWriter(_sessionDirectory);
            _nextFrameIndex = 0;
            _gpuInFlight = 0;
            _readbackFailedFrames = 0;
            _sourceUnavailableCount = 0;
            _queuePauseCount = 0;
            _discardedAfterFatalFrames = 0;
            _sessionFailed = false;
            _fatalFailureFrame = -1;
            _fatalFailureCode = string.Empty;
            _fatalFailureDetail = string.Empty;
            _runtimeSummaryWritten = false;
            _systemRoomMeshExportFinalized = false;
            _lastSelfConsistentNormalCoverage = 0f;
            _lastEdgeRiskRatio = 0f;
            _hasQualitySummary = false;
            _nearViewSectors.Clear();
            _middleViewSectors.Clear();
            _farViewSectors.Clear();
            _grazingViewSectors.Clear();
            _lastRuntimeIssueCode = string.Empty;
            _lastRuntimeIssueRealtime = -1000.0;
            _sessionStartRealtime = Time.realtimeSinceStartupAsDouble;
            _sessionDeadlineRealtime = _sessionStartRealtime + sessionDurationSeconds;
            _captureStopRealtime = -1.0;
            _automaticStop = false;
            _stopReason = string.Empty;
            _recording = true;
            _closing = false;
            _nextCaptureTime = Time.unscaledTime;
            _lastIssue = "正在记录";
            Debug.Log("[深度证据采集] 开始：" + _sessionDirectory);
        }

        public void StopSession()
        {
            StopSessionInternal(
                automatic: false,
                reason: "manual",
                issue: "正在保存剩余数据");
        }

        private void StopSessionInternal(bool automatic, string reason, string issue)
        {
            if (!_recording)
            {
                return;
            }

            _recording = false;
            _closing = true;
            _captureStopRealtime = Time.realtimeSinceStartupAsDouble;
            _automaticStop = automatic;
            _stopReason = reason ?? string.Empty;
            _lastIssue = issue ?? "正在保存剩余数据";
            Debug.Log($"[深度证据采集] 停止采样（{_stopReason}），等待异步写盘完成。");
        }

        private void FailSession(string code, string detail, int frameIndex)
        {
            if (!_sessionFailed)
            {
                _sessionFailed = true;
                _fatalFailureCode = string.IsNullOrEmpty(code) ? "UNKNOWN_FATAL_ERROR" : code;
                _fatalFailureDetail = detail ?? string.Empty;
                _fatalFailureFrame = frameIndex;
                Debug.LogError(
                    $"[深度证据采集] 本轮失败：{_fatalFailureCode}，frame={frameIndex}，{_fatalFailureDetail}");
            }

            _recording = false;
            _closing = true;
            if (_captureStopRealtime < 0.0)
            {
                _captureStopRealtime = Time.realtimeSinceStartupAsDouble;
            }
            _automaticStop = true;
            _stopReason = "fatal_error";
            _lastIssue = "本轮失败：" + _fatalFailureCode;
            _writer?.MarkSessionFailed(_fatalFailureCode, _fatalFailureDetail);
            RecordRuntimeIssue(_fatalFailureCode, _fatalFailureDetail);
        }

        private void TryCaptureFrame()
        {
            if (_gpuInFlight >= maxGpuFramesInFlight || (_writer != null && _writer.PendingCount >= maxWriterQueueFrames))
            {
                _queuePauseCount++;
                _lastIssue = "队列繁忙，已暂停采样";
                RecordRuntimeIssue("QUEUE_PAUSE", $"gpu={_gpuInFlight},writer={_writer?.PendingCount ?? 0}");
                return;
            }

            if (environmentDepthManager == null || !environmentDepthManager.enabled)
            {
                _sourceUnavailableCount++;
                _lastIssue = "环境深度组件未就绪";
                RecordRuntimeIssue("DEPTH_MANAGER_NOT_READY", "EnvironmentDepthManager missing or disabled");
                return;
            }

            Texture sourceDepth = Shader.GetGlobalTexture(EnvironmentDepthTextureId);
            Matrix4x4[] reprojection = Shader.GetGlobalMatrixArray(ReprojectionMatricesId);
            if (sourceDepth == null || reprojection == null)
            {
                _sourceUnavailableCount++;
                _lastIssue = "等待双眼原始深度";
                RecordRuntimeIssue("DEPTH_SOURCE_MISSING", "texture or reprojection matrices unavailable");
                return;
            }

            if (reprojection.Length < EyeCount)
            {
                FailSession(
                    "REPROJECTION_EYE_MISSING",
                    $"actual={reprojection.Length},expected={EyeCount}",
                    _nextFrameIndex);
                return;
            }

            if (sourceDepth.dimension != TextureDimension.Tex2DArray)
            {
                FailSession(
                    "DEPTH_SOURCE_DIMENSION",
                    $"actual={sourceDepth.dimension},expected={TextureDimension.Tex2DArray}",
                    _nextFrameIndex);
                return;
            }

            int sourceEyeCount = GetTextureArrayDepth(sourceDepth);
            if (sourceEyeCount != EyeCount)
            {
                FailSession(
                    "DEPTH_SOURCE_EYE_COUNT",
                    $"actual={sourceEyeCount},expected={EyeCount}",
                    _nextFrameIndex);
                return;
            }

            if (sourceDepth.width <= 0 || sourceDepth.height <= 0)
            {
                FailSession(
                    "DEPTH_SOURCE_SIZE",
                    $"width={sourceDepth.width},height={sourceDepth.height}",
                    _nextFrameIndex);
                return;
            }

            FrameSnapshot snapshot = CaptureSnapshot(sourceDepth, reprojection);
            var payload = new EvidenceFramePayload
            {
                FrameIndex = snapshot.FrameIndex,
                UnityFrame = snapshot.UnityFrame,
                RequestUtcTicks = snapshot.RequestUtcTicks,
                RequestUtcIso = snapshot.RequestUtcIso,
                ViewSector = QuantizeViewSector(snapshot.CameraRotation * Vector3.forward)
            };
            for (int eye = 0; eye < EyeCount; ++eye)
            {
                Vector3 eyePosition = snapshot.EyeWorldPositions[eye];
                payload.EyeWorldPositions[eye * 3] = eyePosition.x;
                payload.EyeWorldPositions[eye * 3 + 1] = eyePosition.y;
                payload.EyeWorldPositions[eye * 3 + 2] = eyePosition.z;
            }
            var pending = new PendingGpuFrame
            {
                Payload = payload,
                Snapshot = snapshot
            };

            RenderTexture frozen = CreateArrayTexture("证据_原始深度", snapshot.Width, snapshot.Height, GraphicsFormat.R32_SFloat, true);
            RenderTexture depthMetrics = CreateArrayTexture("证据_深度量", snapshot.Width, snapshot.Height, GraphicsFormat.R32G32B32A32_SFloat, true);
            RenderTexture worldRaw = CreateArrayTexture("证据_原始世界点", snapshot.Width, snapshot.Height, GraphicsFormat.R32G32B32A32_SFloat, true);
            RenderTexture worldFiltered = CreateArrayTexture("证据_邻域世界点", snapshot.Width, snapshot.Height, GraphicsFormat.R32G32B32A32_SFloat, true);
            RenderTexture normalRaw = CreateArrayTexture("证据_原始法线", snapshot.Width, snapshot.Height, GraphicsFormat.R32G32B32A32_SFloat, true);
            RenderTexture normalFiltered = CreateArrayTexture("证据_邻域法线", snapshot.Width, snapshot.Height, GraphicsFormat.R32G32B32A32_SFloat, true);
            RenderTexture diagnostics = CreateArrayTexture("证据_异常量", snapshot.Width, snapshot.Height, GraphicsFormat.R32G32B32A32_SFloat, true);
            pending.Textures.Add(frozen);
            pending.Textures.Add(depthMetrics);
            pending.Textures.Add(worldRaw);
            pending.Textures.Add(worldFiltered);
            pending.Textures.Add(normalRaw);
            pending.Textures.Add(normalFiltered);
            pending.Textures.Add(diagnostics);

            evidenceCompute.SetInt("_CaptureWidth", snapshot.Width);
            evidenceCompute.SetInt("_CaptureHeight", snapshot.Height);
            evidenceCompute.SetFloat("_NeighbourDepthToleranceMetres", neighbourDepthToleranceMetres);
            evidenceCompute.SetVector("_EnvironmentDepthZBufferParams", snapshot.ZBufferParams);
            evidenceCompute.SetMatrixArray("_InverseReprojectionMatrices", snapshot.InverseReprojection);
            evidenceCompute.SetMatrixArray("_WorldToEyeMatrices", snapshot.StereoView);
            Vector3 leftEye = snapshot.EyeWorldPositions[0];
            Vector3 rightEye = snapshot.EyeWorldPositions[1];
            // Do not use a dynamically indexed float4[2] uniform here. It produced
            // corrupt per-thread values on the Quest/Vulkan path while the same
            // frame's inverse reprojection remained correct.
            evidenceCompute.SetVector(LeftEyeWorldPositionId, new Vector4(leftEye.x, leftEye.y, leftEye.z, 1f));
            evidenceCompute.SetVector(RightEyeWorldPositionId, new Vector4(rightEye.x, rightEye.y, rightEye.z, 1f));

            evidenceCompute.SetTexture(_freezeKernel, "_EnvironmentDepthTexture", sourceDepth);
            evidenceCompute.SetTexture(_freezeKernel, "_FrozenDepthWritable", frozen);
            Dispatch(evidenceCompute, _freezeKernel, snapshot.Width, snapshot.Height, EyeCount);

            evidenceCompute.SetTexture(_geometryKernel, "_FrozenDepthTexture", frozen);
            evidenceCompute.SetTexture(_geometryKernel, "_DepthMetricsTexture", depthMetrics);
            evidenceCompute.SetTexture(_geometryKernel, "_WorldPositionRawTexture", worldRaw);
            evidenceCompute.SetTexture(_geometryKernel, "_WorldPositionFilteredTexture", worldFiltered);
            Dispatch(evidenceCompute, _geometryKernel, snapshot.Width, snapshot.Height, EyeCount);

            evidenceCompute.SetTexture(_normalKernel, "_DepthMetricsTexture", depthMetrics);
            evidenceCompute.SetTexture(_normalKernel, "_WorldPositionRawTexture", worldRaw);
            evidenceCompute.SetTexture(_normalKernel, "_WorldPositionFilteredTexture", worldFiltered);
            evidenceCompute.SetTexture(_normalKernel, "_WorldNormalRawTexture", normalRaw);
            evidenceCompute.SetTexture(_normalKernel, "_WorldNormalFilteredTexture", normalFiltered);
            evidenceCompute.SetTexture(_normalKernel, "_DiagnosticsTexture", diagnostics);
            Dispatch(evidenceCompute, _normalKernel, snapshot.Width, snapshot.Height, EyeCount);

            _gpuInFlight++;
            RequestBuffer(pending, frozen, 1, "raw_depth_r32f", "原始双眼 depth01；GPU 按左眼、右眼顺序展开为 float32");
            RequestBuffer(pending, depthMetrics, 4, "depth_metrics_rgba32f", "R=depth01,G=官方Z参数线性深度米,B=径向米,A=有效");
            RequestBuffer(pending, worldRaw, 4, "world_position_raw_rgba32f", "RGB=原始世界点米,A=有效");
            RequestBuffer(pending, worldFiltered, 4, "world_position_neighbour_rgba32f", "RGB=保守邻域均值,A=支持率");
            RequestBuffer(pending, normalRaw, 4, "world_normal_raw_rgba32f", "RGB=原始GPU法线,A=有效");
            RequestBuffer(pending, normalFiltered, 4, "world_normal_neighbour_rgba32f", "RGB=邻域GPU法线,A=有效");
            RequestBuffer(pending, diagnostics, 4, "diagnostics_rgba32f", "R=邻域最大径向跳变米,G=depth01跳变,B=邻居支持率,A=边缘风险");

            _nextFrameIndex++;
            _lastIssue = "正在记录";
        }

        private void RequestBuffer(PendingGpuFrame pending, RenderTexture texture, int componentCount, string name, string semantic)
        {
            int stride = componentCount * sizeof(float);
            if (texture == null || texture.dimension != TextureDimension.Tex2DArray ||
                texture.volumeDepth != EyeCount || texture.width != pending.Snapshot.Width ||
                texture.height != pending.Snapshot.Height)
            {
                string actual = texture == null
                    ? "null"
                    : $"{texture.width}x{texture.height}x{texture.volumeDepth},{texture.dimension}";
                FailSession(
                    "GPU_BUFFER_DESCRIPTOR",
                    $"buffer={name},actual={actual},expected={pending.Snapshot.Width}x{pending.Snapshot.Height}x{EyeCount},{TextureDimension.Tex2DArray}",
                    pending.Snapshot.FrameIndex);
                return;
            }

            int elementCount = checked(texture.width * texture.height * EyeCount);
            int expectedBytes = checked(elementCount * stride);
            var readbackBuffer = new ComputeBuffer(elementCount, stride, ComputeBufferType.Structured);
            pending.ReadbackBuffers.Add(readbackBuffer);

            int copyKernel;
            if (componentCount == 1)
            {
                copyKernel = _copyR32Kernel;
                evidenceCompute.SetTexture(copyKernel, "_ReadbackR32Texture", texture);
                evidenceCompute.SetBuffer(copyKernel, "_ReadbackR32Buffer", readbackBuffer);
            }
            else if (componentCount == 4)
            {
                copyKernel = _copyRgba32Kernel;
                evidenceCompute.SetTexture(copyKernel, "_ReadbackRGBA32Texture", texture);
                evidenceCompute.SetBuffer(copyKernel, "_ReadbackRGBA32Buffer", readbackBuffer);
            }
            else
            {
                readbackBuffer.Release();
                pending.ReadbackBuffers.Remove(readbackBuffer);
                throw new ArgumentOutOfRangeException(nameof(componentCount), componentCount, "只支持 R32 或 RGBA32F 证据缓冲");
            }

            Dispatch(evidenceCompute, copyKernel, texture.width, texture.height, EyeCount);
            pending.RemainingRequests++;
            AsyncGPUReadback.Request(readbackBuffer, request =>
            {
                if (request.hasError)
                {
                    pending.FailedRequests++;
                    pending.ReadbackErrors += (pending.ReadbackErrors.Length == 0 ? string.Empty : ";") + name;
                }
                else
                {
                    NativeArray<byte> data = request.GetData<byte>();
                    byte[] bytes = data.ToArray();
                    if (bytes.Length != expectedBytes)
                    {
                        pending.FailedRequests++;
                        pending.ReadbackErrors += (pending.ReadbackErrors.Length == 0 ? string.Empty : ";") +
                                                  $"{name}:bytes={bytes.Length},expected={expectedBytes}";
                    }
                    else
                    {
                        pending.Payload.Buffers.Add(new EvidenceBufferPayload
                        {
                            Name = name,
                            Semantic = semantic,
                            GraphicsFormat = texture.graphicsFormat.ToString(),
                            Width = texture.width,
                            Height = texture.height,
                            Depth = EyeCount,
                            Bytes = bytes
                        });
                    }
                }

                pending.LastReadbackRealtime = Time.realtimeSinceStartupAsDouble;
                pending.RemainingRequests--;
                if (pending.RemainingRequests == 0)
                {
                    CompleteGpuFrame(pending);
                }
            });
        }

        private void CompleteGpuFrame(PendingGpuFrame pending)
        {
            _gpuInFlight = Mathf.Max(0, _gpuInFlight - 1);
            bool frameComplete = ValidateCompletedFrame(pending, out string integrityError);
            if (!frameComplete)
            {
                _readbackFailedFrames++;
                FailSession(
                    "FRAME_INTEGRITY_FAILED",
                    integrityError,
                    pending.Snapshot.FrameIndex);
            }

            if (frameComplete && !_sessionFailed)
            {
                TrackingSnapshot completionTracking = CaptureTracking();
                Vector3 completionPosition = sourceCamera ? sourceCamera.transform.position : Vector3.zero;
                Quaternion completionRotation = sourceCamera ? sourceCamera.transform.rotation : Quaternion.identity;
                pending.Payload.MetadataJson = BuildFrameMetadata(
                    pending.Snapshot,
                    pending.LastReadbackRealtime,
                    completionPosition,
                    completionRotation,
                    completionTracking,
                    pending.FailedRequests,
                    pending.ReadbackErrors,
                    pending.Payload.Buffers);

                _writer?.Enqueue(pending.Payload);
            }
            else if (frameComplete)
            {
                _discardedAfterFatalFrames++;
            }

            foreach (RenderTexture texture in pending.Textures)
            {
                if (texture == null) continue;
                texture.Release();
                Destroy(texture);
            }
            foreach (ComputeBuffer buffer in pending.ReadbackBuffers)
            {
                buffer?.Release();
            }
        }

        private static bool ValidateCompletedFrame(PendingGpuFrame pending, out string error)
        {
            if (pending.FailedRequests > 0)
            {
                error = string.IsNullOrEmpty(pending.ReadbackErrors)
                    ? $"readbackFailures={pending.FailedRequests}"
                    : pending.ReadbackErrors;
                return false;
            }

            string[] names =
            {
                "raw_depth_r32f",
                "depth_metrics_rgba32f",
                "world_position_raw_rgba32f",
                "world_position_neighbour_rgba32f",
                "world_normal_raw_rgba32f",
                "world_normal_neighbour_rgba32f",
                "diagnostics_rgba32f"
            };
            int[] components = { 1, 4, 4, 4, 4, 4, 4 };
            EvidenceBufferPayload rawDepth = null;
            EvidenceBufferPayload depthMetrics = null;
            EvidenceBufferPayload worldRaw = null;
            var seen = new HashSet<string>(StringComparer.Ordinal);

            for (int expectedIndex = 0; expectedIndex < names.Length; ++expectedIndex)
            {
                EvidenceBufferPayload match = null;
                foreach (EvidenceBufferPayload buffer in pending.Payload.Buffers)
                {
                    if (buffer != null && buffer.Name == names[expectedIndex])
                    {
                        if (!seen.Add(buffer.Name))
                        {
                            error = "duplicateBuffer=" + buffer.Name;
                            return false;
                        }
                        match = buffer;
                    }
                }

                if (match == null)
                {
                    error = "missingBuffer=" + names[expectedIndex];
                    return false;
                }

                int expectedBytes;
                try
                {
                    expectedBytes = checked(
                        pending.Snapshot.Width * pending.Snapshot.Height * EyeCount * components[expectedIndex] * sizeof(float));
                }
                catch (OverflowException)
                {
                    error = $"bufferSizeOverflow={match.Name}";
                    return false;
                }

                if (match.Width != pending.Snapshot.Width || match.Height != pending.Snapshot.Height ||
                    match.Depth != EyeCount || match.Bytes == null || match.Bytes.Length != expectedBytes)
                {
                    error = $"bufferShape={match.Name}:actual={match.Width}x{match.Height}x{match.Depth},bytes={match.Bytes?.Length ?? 0}," +
                            $"expected={pending.Snapshot.Width}x{pending.Snapshot.Height}x{EyeCount},bytes={expectedBytes}";
                    return false;
                }

                if (match.Name == "raw_depth_r32f") rawDepth = match;
                else if (match.Name == "depth_metrics_rgba32f") depthMetrics = match;
                else if (match.Name == "world_position_raw_rgba32f") worldRaw = match;
            }

            int eyePixels = pending.Snapshot.Width * pending.Snapshot.Height;
            for (int eye = 0; eye < EyeCount; ++eye)
            {
                bool hasValidDepth = false;
                int firstPixel = eye * eyePixels;
                int endPixel = firstPixel + eyePixels;
                for (int pixel = firstPixel; pixel < endPixel; ++pixel)
                {
                    float value = BitConverter.ToSingle(rawDepth.Bytes, pixel * sizeof(float));
                    if (!float.IsNaN(value) && !float.IsInfinity(value) && value > 0f)
                    {
                        hasValidDepth = true;
                        break;
                    }
                }

                if (!hasValidDepth)
                {
                    error = $"eyeMissing={eye},validDepthSamples=0";
                    return false;
                }
            }

            if (!ValidateWorldRadialConsistency(pending.Snapshot, rawDepth, depthMetrics, worldRaw, out error))
            {
                return false;
            }

            error = string.Empty;
            return true;
        }

        private static bool ValidateWorldRadialConsistency(
            FrameSnapshot snapshot,
            EvidenceBufferPayload rawDepth,
            EvidenceBufferPayload depthMetrics,
            EvidenceBufferPayload worldRaw,
            out string error)
        {
            // A sparse deterministic audit is enough to catch the previous Quest
            // failure (~36% false rejects and metre-to-kilometre radial errors)
            // without scanning another 20 MB on the main thread every frame.
            int step = Mathf.Max(1, Mathf.Min(snapshot.Width, snapshot.Height) / 20);
            int expectedValid = 0;
            int falseRejected = 0;
            int comparable = 0;
            int radialMismatch = 0;
            float maxRadialError = 0f;

            int eyePixels = snapshot.Width * snapshot.Height;
            for (int eye = 0; eye < EyeCount; ++eye)
            {
                for (int y = step / 2; y < snapshot.Height; y += step)
                {
                    for (int x = step / 2; x < snapshot.Width; x += step)
                    {
                        int pixel = eye * eyePixels + y * snapshot.Width + x;
                        float depth01 = ReadFloat(rawDepth.Bytes, pixel, 1, 0);
                        if (!TryExpectedWorld(snapshot, eye, x, y, depth01, out _))
                        {
                            continue;
                        }

                        expectedValid++;
                        float metricsValid = ReadFloat(depthMetrics.Bytes, pixel, 4, 3);
                        float worldValid = ReadFloat(worldRaw.Bytes, pixel, 4, 3);
                        if (!(metricsValid > 0.5f) || !(worldValid > 0.5f))
                        {
                            falseRejected++;
                            continue;
                        }

                        Vector3 world = new Vector3(
                            ReadFloat(worldRaw.Bytes, pixel, 4, 0),
                            ReadFloat(worldRaw.Bytes, pixel, 4, 1),
                            ReadFloat(worldRaw.Bytes, pixel, 4, 2));
                        float actualRadial = ReadFloat(depthMetrics.Bytes, pixel, 4, 2);
                        float expectedRadial = Vector3.Distance(world, snapshot.EyeWorldPositions[eye]);
                        comparable++;

                        if (!IsFinite(world.x) || !IsFinite(world.y) || !IsFinite(world.z) ||
                            !IsFinite(actualRadial) || !IsFinite(expectedRadial))
                        {
                            radialMismatch++;
                            maxRadialError = float.PositiveInfinity;
                            continue;
                        }

                        float radialError = Mathf.Abs(actualRadial - expectedRadial);
                        maxRadialError = Mathf.Max(maxRadialError, radialError);
                        float tolerance = 0.005f + expectedRadial * 0.0001f;
                        if (radialError > tolerance)
                        {
                            radialMismatch++;
                        }
                    }
                }
            }

            if (expectedValid == 0 || comparable == 0)
            {
                error = $"semanticAuditSamples=0,expectedValid={expectedValid},comparable={comparable}";
                return false;
            }

            int falseRejectLimit = Mathf.Max(2, Mathf.CeilToInt(expectedValid * 0.01f));
            if (falseRejected > falseRejectLimit)
            {
                error = $"semanticFalseReject={falseRejected}/{expectedValid},limit={falseRejectLimit}";
                return false;
            }

            int radialMismatchLimit = Mathf.Max(2, Mathf.CeilToInt(comparable * 0.01f));
            if (radialMismatch > radialMismatchLimit)
            {
                error = $"semanticRadialMismatch={radialMismatch}/{comparable},limit={radialMismatchLimit}," +
                        $"maxErrorMetres={maxRadialError.ToString("R", CultureInfo.InvariantCulture)}";
                return false;
            }

            error = string.Empty;
            return true;
        }

        private static bool TryExpectedWorld(
            FrameSnapshot snapshot,
            int eye,
            int pixelX,
            int pixelY,
            float depth01,
            out Vector3 world)
        {
            world = Vector3.zero;
            if (!IsFinite(depth01) || depth01 <= 0f)
            {
                return false;
            }

            float ndcDepth = depth01 * 2f - 1f;
            float denominator = ndcDepth + snapshot.ZBufferParams.y;
            if (!IsFinite(denominator) || Mathf.Abs(denominator) < 1e-7f)
            {
                return false;
            }

            float linearDepth = snapshot.ZBufferParams.x / denominator;
            if (!IsFinite(linearDepth) || linearDepth <= 0f)
            {
                return false;
            }

            float ndcX = ((pixelX + 0.5f) / snapshot.Width) * 2f - 1f;
            float ndcY = ((pixelY + 0.5f) / snapshot.Height) * 2f - 1f;
            Vector4 homogeneousWorld = snapshot.InverseReprojection[eye] * new Vector4(ndcX, ndcY, ndcDepth, 1f);
            if (!IsFinite(homogeneousWorld.x) || !IsFinite(homogeneousWorld.y) ||
                !IsFinite(homogeneousWorld.z) || !IsFinite(homogeneousWorld.w) ||
                Mathf.Abs(homogeneousWorld.w) < 1e-7f)
            {
                return false;
            }

            world = new Vector3(homogeneousWorld.x, homogeneousWorld.y, homogeneousWorld.z) / homogeneousWorld.w;
            return IsFinite(world.x) && IsFinite(world.y) && IsFinite(world.z);
        }

        private static float ReadFloat(byte[] bytes, int pixel, int componentCount, int component)
        {
            int offset = checked((pixel * componentCount + component) * sizeof(float));
            return BitConverter.ToSingle(bytes, offset);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private FrameSnapshot CaptureSnapshot(Texture sourceDepth, Matrix4x4[] reprojection)
        {
            DateTime nowUtc = DateTime.UtcNow;
            var snapshot = new FrameSnapshot
            {
                FrameIndex = _nextFrameIndex,
                UnityFrame = Time.frameCount,
                RequestUtcTicks = nowUtc.Ticks,
                RequestUtcIso = nowUtc.ToString("O", CultureInfo.InvariantCulture),
                RequestRealtime = Time.realtimeSinceStartupAsDouble,
                RequestDspTime = AudioSettings.dspTime,
                TimeValue = Time.time,
                UnscaledTime = Time.unscaledTime,
                Width = sourceDepth.width,
                Height = sourceDepth.height,
                SourceEyeCount = GetTextureArrayDepth(sourceDepth),
                SourceType = sourceDepth.GetType().FullName,
                SourceGraphicsFormat = sourceDepth.graphicsFormat.ToString(),
                SourceDimension = sourceDepth.dimension.ToString(),
                SourceMipCount = sourceDepth.mipmapCount,
                SourceFilterMode = sourceDepth.filterMode,
                SourceWrapMode = sourceDepth.wrapMode,
                ZBufferParams = Shader.GetGlobalVector(ZBufferParamsId),
                Tracking = CaptureTracking()
            };

            for (int eye = 0; eye < EyeCount; ++eye)
            {
                snapshot.Reprojection[eye] = reprojection[eye];
                snapshot.InverseReprojection[eye] = reprojection[eye].inverse;
            }

            if (sourceCamera != null)
            {
                snapshot.CameraToWorld = sourceCamera.cameraToWorldMatrix;
                snapshot.WorldToCamera = sourceCamera.worldToCameraMatrix;
                snapshot.Projection = sourceCamera.projectionMatrix;
                snapshot.NonJitteredProjection = sourceCamera.nonJitteredProjectionMatrix;
                snapshot.CameraPosition = sourceCamera.transform.position;
                snapshot.CameraRotation = sourceCamera.transform.rotation;
                snapshot.NearClip = sourceCamera.nearClipPlane;
                snapshot.FarClip = sourceCamera.farClipPlane;
                snapshot.FieldOfView = sourceCamera.fieldOfView;
                snapshot.Aspect = sourceCamera.aspect;
                snapshot.CameraPixelWidth = sourceCamera.pixelWidth;
                snapshot.CameraPixelHeight = sourceCamera.pixelHeight;
                try
                {
                    snapshot.StereoView[0] = sourceCamera.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
                    snapshot.StereoView[1] = sourceCamera.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
                    snapshot.StereoProjection[0] = sourceCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
                    snapshot.StereoProjection[1] = sourceCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right);
                }
                catch
                {
                    snapshot.StereoView[0] = snapshot.WorldToCamera;
                    snapshot.StereoView[1] = snapshot.WorldToCamera;
                    snapshot.StereoProjection[0] = snapshot.Projection;
                    snapshot.StereoProjection[1] = snapshot.Projection;
                }
            }
            else
            {
                snapshot.StereoView[0] = Matrix4x4.identity;
                snapshot.StereoView[1] = Matrix4x4.identity;
                snapshot.StereoProjection[0] = Matrix4x4.identity;
                snapshot.StereoProjection[1] = Matrix4x4.identity;
            }

            for (int eye = 0; eye < EyeCount; ++eye)
            {
                Matrix4x4 eyeToWorld = snapshot.StereoView[eye].inverse;
                snapshot.EyeWorldPositions[eye] = eyeToWorld.GetColumn(3);
            }

            return snapshot;
        }

        private static int GetTextureArrayDepth(Texture texture)
        {
            if (texture is RenderTexture renderTexture)
            {
                return renderTexture.volumeDepth;
            }

            if (texture is Texture2DArray textureArray)
            {
                return textureArray.depth;
            }

            return 0;
        }

        private static TrackingSnapshot CaptureTracking()
        {
            return new TrackingSnapshot
            {
                Head = CaptureNode(XRNode.Head),
                CenterEye = CaptureNode(XRNode.CenterEye),
                LeftEye = CaptureNode(XRNode.LeftEye),
                RightEye = CaptureNode(XRNode.RightEye)
            };
        }

        private static XrNodeSnapshot CaptureNode(XRNode node)
        {
            UnityEngine.XR.InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            var sample = new XrNodeSnapshot { DeviceValid = device.isValid };
            if (!device.isValid) return sample;
            device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.isTracked, out sample.IsTracked);
            device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.trackingState, out sample.TrackingState);
            device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.devicePosition, out sample.Position);
            device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.deviceRotation, out sample.Rotation);
            device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.deviceVelocity, out sample.Velocity);
            device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.deviceAngularVelocity, out sample.AngularVelocity);
            return sample;
        }

        private void HandleInput()
        {
            bool toggle = false;
            try
            {
                toggle |= OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch);
            }
            catch
            {
                // Editor and non-Oculus runtimes use the keyboard fallback below.
            }

#if ENABLE_INPUT_SYSTEM
            Keyboard keyboard = Keyboard.current;
            if (keyboard != null)
            {
                toggle |= keyboard.f9Key.wasPressedThisFrame;
            }
#endif

            if (toggle)
            {
                if (_recording) StopSession(); else StartSession();
            }
        }

        private void ResolveDependencies()
        {
            if (environmentDepthManager == null)
            {
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);
            }
            if (sourceCamera == null)
            {
                sourceCamera = Camera.main;
            }
            if (systemRoomMeshCapture == null)
            {
                systemRoomMeshCapture = FindAnyObjectByType<QuestSystemRoomMeshCapture>(FindObjectsInactive.Include);
            }
        }

        private void ResolveKernels()
        {
            if (evidenceCompute == null) return;
            try
            {
                _freezeKernel = evidenceCompute.FindKernel("FreezeDepth");
                _geometryKernel = evidenceCompute.FindKernel("CaptureGeometry");
                _normalKernel = evidenceCompute.FindKernel("CaptureNormals");
                _copyR32Kernel = evidenceCompute.FindKernel("CopyR32ArrayToBuffer");
                _copyRgba32Kernel = evidenceCompute.FindKernel("CopyRGBA32ArrayToBuffer");
            }
            catch (Exception ex)
            {
                _lastIssue = "采集着色器错误：" + ex.Message;
            }
        }

        private void DrainWriterResults()
        {
            if (_writer == null) return;
            while (_writer.TryDequeueCompleted(out EvidenceWriteSummary summary))
            {
                if (!summary.Success)
                {
                    FailSession("WRITE_FAILED", summary.Error, summary.FrameIndex);
                }
                else
                {
                    _lastSelfConsistentNormalCoverage = summary.SelfConsistentNormalCoverage;
                    _lastEdgeRiskRatio = summary.EdgeRiskRatio;
                    _hasQualitySummary = true;
                    if ((summary.RemedialCategoryMask & 1) != 0) _nearViewSectors.Add(summary.ViewSector);
                    if ((summary.RemedialCategoryMask & 2) != 0) _middleViewSectors.Add(summary.ViewSector);
                    if ((summary.RemedialCategoryMask & 4) != 0) _farViewSectors.Add(summary.ViewSector);
                    if ((summary.RemedialCategoryMask & 8) != 0) _grazingViewSectors.Add(summary.ViewSector);
                }
            }
        }

        private void EnsureHud()
        {
            if (!showHud)
            {
                if (_hudCanvas != null) _hudCanvas.gameObject.SetActive(false);
                return;
            }
            if (_hudCanvas != null)
            {
                _hudCanvas.gameObject.SetActive(true);
                return;
            }
            if (sourceCamera == null) return;

            var root = new GameObject("深度证据提示");
            root.transform.SetParent(sourceCamera.transform, false);
            root.transform.localPosition = new Vector3(0f, -0.22f, 0.9f);
            root.transform.localRotation = Quaternion.identity;
            root.transform.localScale = Vector3.one * (0.00072f * hudScale);

            _hudCanvas = root.AddComponent<Canvas>();
            _hudCanvas.renderMode = RenderMode.WorldSpace;
            _hudCanvas.worldCamera = sourceCamera;
            var canvasRect = (RectTransform)_hudCanvas.transform;
            canvasRect.sizeDelta = new Vector2(760f, 178f);
            var scaler = root.AddComponent<CanvasScaler>();
            scaler.dynamicPixelsPerUnit = 2f;

            var panel = new GameObject("底板");
            panel.transform.SetParent(root.transform, false);
            var panelRect = panel.AddComponent<RectTransform>();
            panelRect.anchorMin = Vector2.zero;
            panelRect.anchorMax = Vector2.one;
            panelRect.offsetMin = Vector2.zero;
            panelRect.offsetMax = Vector2.zero;
            var image = panel.AddComponent<Image>();
            image.color = new Color(0.015f, 0.02f, 0.035f, 0.78f);
            image.raycastTarget = false;

            var textObject = new GameObject("提示文字");
            textObject.transform.SetParent(panel.transform, false);
            var textRect = textObject.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(20f, 10f);
            textRect.offsetMax = new Vector2(-20f, -10f);
            _hudText = textObject.AddComponent<Text>();
            _hudText.font = LoadBuiltinFont();
            _hudText.fontSize = 30;
            _hudText.alignment = TextAnchor.MiddleLeft;
            _hudText.color = new Color(0.35f, 0.92f, 1f, 1f);
            _hudText.horizontalOverflow = HorizontalWrapMode.Overflow;
            _hudText.verticalOverflow = VerticalWrapMode.Truncate;
            _hudText.raycastTarget = false;
        }

        private static Font LoadBuiltinFont()
        {
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (font == null) font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return font;
        }

        private void UpdateHud()
        {
            if (_hudText == null) return;
            string state = _sessionFailed ? "失败" : _recording ? "记录中" : _closing ? "保存中" : "待机";
            int written = _writer?.WrittenCount ?? 0;
            int queue = (_writer?.PendingCount ?? 0) + _gpuInFlight;
            string progress = remedialCaptureGuidance
                ? $"覆盖 {RemedialProgress * 100f:0}%"
                : _recording
                    ? $"安全剩余 {Mathf.CeilToInt((float)Math.Max(0.0, _sessionDeadlineRealtime - Time.realtimeSinceStartupAsDouble))} 秒"
                    : "待命";
            string guidance = CurrentRemedialGuidance();
            string quality = _hasQualitySummary
                ? $"法线自洽 {_lastSelfConsistentNormalCoverage * 100f:0}%　边缘风险 {_lastEdgeRiskRatio * 100f:0}%"
                : "质量账等待首帧";
            _hudText.text =
                $"深度证据｜{state}　{progress}　帧 {_nextFrameIndex}　已存 {written}　队列 {queue}\n" +
                $"{guidance}｜{quality}\n" +
                "B 开始/停止采集";
        }

        private string CurrentRemedialGuidance()
        {
            if (!remedialCaptureGuidance || !_recording)
            {
                return _lastIssue;
            }

            float near = Mathf.Clamp01((float)_nearViewSectors.Count / NearSectorGoal);
            float middle = Mathf.Clamp01((float)_middleViewSectors.Count / MiddleSectorGoal);
            float far = Mathf.Clamp01((float)_farViewSectors.Count / FarSectorGoal);
            float grazing = Mathf.Clamp01((float)_grazingViewSectors.Count / GrazingSectorGoal);
            float minimum = Mathf.Min(near, middle, far, grazing);
            if (near <= minimum)
                return "靠近约0.35–0.75米扫凹角、凸角";
            if (far <= minimum)
                return "退到约1.5–3米重扫同一结构";
            if (grazing <= minimum)
                return "从斜侧和另一侧扫边缘与折角";
            return "中距缓慢扫平面、门框和桌沿";
        }

        private float RemedialProgress
        {
            get
            {
                float near = Mathf.Clamp01((float)_nearViewSectors.Count / NearSectorGoal);
                float middle = Mathf.Clamp01((float)_middleViewSectors.Count / MiddleSectorGoal);
                float far = Mathf.Clamp01((float)_farViewSectors.Count / FarSectorGoal);
                float grazing = Mathf.Clamp01((float)_grazingViewSectors.Count / GrazingSectorGoal);
                return (near + middle + far + grazing) * 0.25f;
            }
        }

        private static int QuantizeViewSector(Vector3 forward)
        {
            if (forward.sqrMagnitude <= 1e-8f) return 0;
            forward.Normalize();
            float yaw = Mathf.Atan2(forward.x, forward.z) * Mathf.Rad2Deg;
            if (yaw < 0f) yaw += 360f;
            int yawBin = Mathf.Clamp(Mathf.FloorToInt(yaw / 30f), 0, 11);
            float pitch = Mathf.Asin(Mathf.Clamp(forward.y, -1f, 1f)) * Mathf.Rad2Deg;
            int pitchBin = pitch < -20f ? 0 : pitch > 20f ? 2 : 1;
            return pitchBin * 12 + yawBin;
        }

        private void WriteSessionDescriptor()
        {
            string path = Path.Combine(_sessionDirectory, "session.json");
            var sb = new StringBuilder(4096);
            sb.AppendLine("{");
            AppendJsonString(sb, "schema", SchemaName, true, 2);
            AppendJsonString(sb, "createdUtc", DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture), true, 2);
            AppendJsonString(sb, "createdLocal", DateTime.Now.ToString("O", CultureInfo.InvariantCulture), true, 2);
            AppendJsonString(sb, "applicationVersion", Application.version, true, 2);
            AppendJsonString(sb, "applicationIdentifier", Application.identifier, true, 2);
            AppendJsonString(sb, "buildGuid", Application.buildGUID, true, 2);
            AppendJsonString(sb, "unityVersion", Application.unityVersion, true, 2);
            AppendJsonString(sb, "platform", Application.platform.ToString(), true, 2);
            AppendJsonString(sb, "deviceModel", SystemInfo.deviceModel, true, 2);
            AppendJsonString(sb, "deviceName", SystemInfo.deviceName, true, 2);
            AppendJsonString(sb, "operatingSystem", SystemInfo.operatingSystem, true, 2);
            AppendJsonString(sb, "processorType", SystemInfo.processorType, true, 2);
            AppendJsonNumber(sb, "processorCount", SystemInfo.processorCount, true, 2);
            AppendJsonNumber(sb, "systemMemoryMb", SystemInfo.systemMemorySize, true, 2);
            AppendJsonString(sb, "graphicsDevice", SystemInfo.graphicsDeviceName, true, 2);
            AppendJsonString(sb, "graphicsVendor", SystemInfo.graphicsDeviceVendor, true, 2);
            AppendJsonString(sb, "graphicsApi", SystemInfo.graphicsDeviceType.ToString(), true, 2);
            AppendJsonNumber(sb, "graphicsMemoryMb", SystemInfo.graphicsMemorySize, true, 2);
            AppendJsonNumber(sb, "graphicsShaderLevel", SystemInfo.graphicsShaderLevel, true, 2);
            AppendJsonBool(sb, "supportsComputeShaders", SystemInfo.supportsComputeShaders, true, 2);
            AppendJsonBool(sb, "supportsAsyncGpuReadback", SystemInfo.supportsAsyncGPUReadback, true, 2);
            AppendJsonString(sb, "colorSpace", QualitySettings.activeColorSpace.ToString(), true, 2);
            AppendJsonString(sb, "qualityLevel", QualitySettings.names.Length > QualitySettings.GetQualityLevel() ? QualitySettings.names[QualitySettings.GetQualityLevel()] : QualitySettings.GetQualityLevel().ToString(CultureInfo.InvariantCulture), true, 2);
            AppendJsonBool(sb, "xrEnabled", XRSettings.enabled, true, 2);
            AppendJsonString(sb, "xrLoadedDevice", XRSettings.loadedDeviceName, true, 2);
            AppendJsonNumber(sb, "xrEyeTextureWidth", XRSettings.eyeTextureWidth, true, 2);
            AppendJsonNumber(sb, "xrEyeTextureHeight", XRSettings.eyeTextureHeight, true, 2);
            AppendJsonNumber(sb, "xrEyeTextureResolutionScale", XRSettings.eyeTextureResolutionScale, true, 2);
            AppendJsonBool(sb, "environmentDepthSupported", EnvironmentDepthManager.IsSupported, true, 2);
            AppendJsonString(sb, "environmentDepthAssemblyVersion", environmentDepthManager != null ? environmentDepthManager.GetType().Assembly.GetName().Version?.ToString() : string.Empty, true, 2);
            AppendJsonString(sb, "trackingOriginMode", CurrentTrackingOriginMode(), true, 2);
            AppendJsonNumber(sb, "captureIntervalSeconds", captureIntervalSeconds, true, 2);
            AppendJsonNumber(sb, "sessionDurationSeconds", sessionDurationSeconds, true, 2);
            AppendJsonString(sb, "stopPolicy", remedialCaptureGuidance ? "coverage_complete_or_safety_timeout" : "safety_timeout", true, 2);
            AppendJsonNumber(sb, "maxFrames", maxFramesPerSession, true, 2);
            AppendJsonNumber(sb, "maxGpuFramesInFlight", maxGpuFramesInFlight, true, 2);
            AppendJsonNumber(sb, "maxWriterQueueFrames", maxWriterQueueFrames, true, 2);
            AppendJsonNumber(sb, "neighbourDepthToleranceMetres", neighbourDepthToleranceMetres, true, 2);
            AppendJsonString(sb, "sourceTextureGlobal", "_EnvironmentDepthTexture", true, 2);
            AppendJsonString(sb, "sourceMatricesGlobal", "_EnvironmentDepthReprojectionMatrices", true, 2);
            AppendJsonBool(sb, "sensorTimestampAvailable", false, true, 2);
            AppendJsonBool(sb, "sensorFrameIdAvailable", false, true, 2);
            AppendJsonString(sb, "captureControl", "start_stop_only", true, 2);
            AppendJsonBool(sb, "humanAnnotationsEnabled", false, true, 2);
            AppendJsonBool(sb, "systemRoomMeshCompanionEnabled", systemRoomMeshCapture != null, true, 2);
            AppendJsonString(sb, "systemRoomMeshCompanionDirectory", systemRoomMeshCapture != null ? systemRoomMeshCapture.OutputFolderName : string.Empty, true, 2);
            AppendJsonBool(sb, "systemRoomMeshRenderedInCaptureScene", false, true, 2);
            AppendJsonBool(sb, "remedialCaptureGuidance", remedialCaptureGuidance, true, 2);
            AppendJsonString(sb, "normalQualityGate", "raw/filtered agreement <=20deg and not depth-edge-risk; diagnostic only", true, 2);
            AppendJsonString(sb, "classificationStage", "offline_deterministic", true, 2);
            AppendJsonNumber(sb, "eyeCount", EyeCount, true, 2);
            AppendJsonString(sb, "readbackLayout", "gpu_structured_buffer_left_then_right", true, 2);
            AppendJsonString(sb, "authority", "raw_depth_r32f + frame matrices; all other buffers are reproducible diagnostics", true, 2);
            sb.AppendLine("  \"buffers\": [");
            sb.AppendLine("    \"raw_depth_r32f\",");
            sb.AppendLine("    \"depth_metrics_rgba32f\",");
            sb.AppendLine("    \"world_position_raw_rgba32f\",");
            sb.AppendLine("    \"world_position_neighbour_rgba32f\",");
            sb.AppendLine("    \"world_normal_raw_rgba32f\",");
            sb.AppendLine("    \"world_normal_neighbour_rgba32f\",");
            sb.AppendLine("    \"diagnostics_rgba32f\"");
            sb.AppendLine("  ]");
            sb.AppendLine("}");
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
        }

        private void RecordRuntimeIssue(string code, string detail)
        {
            if (string.IsNullOrEmpty(_sessionDirectory)) return;
            double now = Time.realtimeSinceStartupAsDouble;
            if (code == _lastRuntimeIssueCode && now - _lastRuntimeIssueRealtime < 2.0) return;
            _lastRuntimeIssueCode = code;
            _lastRuntimeIssueRealtime = now;
            try
            {
                string path = Path.Combine(_sessionDirectory, "runtime_issues.csv");
                if (!File.Exists(path))
                    File.WriteAllText(path, "utc,realtime,next_frame,code,detail\n", new UTF8Encoding(false));
                string line = CsvValue(DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)) + "," +
                              now.ToString("R", CultureInfo.InvariantCulture) + "," +
                              _nextFrameIndex.ToString(CultureInfo.InvariantCulture) + "," +
                              CsvValue(code) + "," + CsvValue(detail) + "\n";
                File.AppendAllText(path, line, new UTF8Encoding(false));
            }
            catch
            {
                // Runtime summary counters remain available even when this auxiliary timeline cannot be written.
            }
        }

        private static string CurrentTrackingOriginMode()
        {
            var subsystems = new List<XRInputSubsystem>();
            SubsystemManager.GetSubsystems(subsystems);
            foreach (XRInputSubsystem subsystem in subsystems)
            {
                if (subsystem != null && subsystem.running)
                    return subsystem.GetTrackingOriginMode().ToString();
            }
            return "Unavailable";
        }

        private void WriteRuntimeSummary()
        {
            if (_runtimeSummaryWritten || string.IsNullOrEmpty(_sessionDirectory)) return;
            _runtimeSummaryWritten = true;
            try
            {
                var sb = new StringBuilder(1024);
                sb.AppendLine("{");
                AppendJsonString(sb, "schema", "ScanCoverDepthEvidenceRuntimeSummary/v1", true, 2);
                AppendJsonString(sb, "finishedUtc", DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture), true, 2);
                AppendJsonString(sb, "sessionStatus", _sessionFailed ? "failed" : "complete", true, 2);
                AppendJsonBool(sb, "sessionFailed", _sessionFailed, true, 2);
                AppendJsonString(sb, "fatalFailureCode", _fatalFailureCode, true, 2);
                AppendJsonString(sb, "fatalFailureDetail", _fatalFailureDetail, true, 2);
                AppendJsonNumber(sb, "fatalFailureFrame", _fatalFailureFrame, true, 2);
                AppendJsonNumber(sb, "requestedFrames", _nextFrameIndex, true, 2);
                AppendJsonNumber(sb, "writtenFrames", _writer?.WrittenCount ?? 0, true, 2);
                AppendJsonNumber(sb, "writerFailedFrames", _writer?.FailedCount ?? 0, true, 2);
                AppendJsonNumber(sb, "gpuReadbackFailedFrames", _readbackFailedFrames, true, 2);
                AppendJsonNumber(sb, "discardedAfterFatalFrames", _discardedAfterFatalFrames, true, 2);
                AppendJsonNumber(sb, "sourceUnavailableTicks", _sourceUnavailableCount, true, 2);
                AppendJsonNumber(sb, "queuePauseTicks", _queuePauseCount, true, 2);
                AppendJsonNumber(sb, "configuredDurationSeconds", sessionDurationSeconds, true, 2);
                double stopRealtime = _captureStopRealtime >= 0.0
                    ? _captureStopRealtime
                    : Time.realtimeSinceStartupAsDouble;
                AppendJsonNumber(sb, "recordingElapsedSeconds", Math.Max(0.0, stopRealtime - _sessionStartRealtime), true, 2);
                AppendJsonBool(sb, "automaticStop", _automaticStop, true, 2);
                AppendJsonString(sb, "stopReason", _stopReason, true, 2);
                AppendJsonString(sb, "systemRoomMeshStatus", systemRoomMeshCapture != null ? systemRoomMeshCapture.LastStatus : "disabled", true, 2);
                AppendJsonString(sb, "systemRoomMeshIssue", systemRoomMeshCapture != null ? systemRoomMeshCapture.LastIssue : string.Empty, true, 2);
                AppendJsonNumber(sb, "systemRoomMeshCount", systemRoomMeshCapture != null ? systemRoomMeshCapture.LastExportedMeshCount : 0, true, 2);
                AppendJsonNumber(sb, "systemRoomMeshVertices", systemRoomMeshCapture != null ? systemRoomMeshCapture.LastExportedVertexCount : 0, true, 2);
                AppendJsonNumber(sb, "systemRoomMeshTriangles", systemRoomMeshCapture != null ? systemRoomMeshCapture.LastExportedTriangleCount : 0, false, 2);
                sb.AppendLine("}");
                File.WriteAllText(Path.Combine(_sessionDirectory, "runtime_summary.json"), sb.ToString(), new UTF8Encoding(false));
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[深度证据采集] 运行摘要写入失败：" + ex.Message);
            }
        }

        private string BuildFrameMetadata(
            FrameSnapshot s,
            double completionRealtime,
            Vector3 completionPosition,
            Quaternion completionRotation,
            TrackingSnapshot completionTracking,
            int failedReadbacks,
            string readbackErrors,
            List<EvidenceBufferPayload> buffers)
        {
            var sb = new StringBuilder(16384);
            sb.AppendLine("{");
            AppendJsonString(sb, "schema", SchemaName, true, 2);
            AppendJsonNumber(sb, "frameIndex", s.FrameIndex, true, 2);
            AppendJsonNumber(sb, "unityFrame", s.UnityFrame, true, 2);
            AppendJsonString(sb, "requestUtc", s.RequestUtcIso, true, 2);
            AppendJsonNumber(sb, "requestUtcTicks", s.RequestUtcTicks, true, 2);
            AppendJsonNumber(sb, "requestRealtime", s.RequestRealtime, true, 2);
            AppendJsonNumber(sb, "readbackCompleteRealtime", completionRealtime, true, 2);
            AppendJsonNumber(sb, "gpuReadbackLatencySeconds", completionRealtime - s.RequestRealtime, true, 2);
            AppendJsonNumber(sb, "dspTime", s.RequestDspTime, true, 2);
            AppendJsonNumber(sb, "time", s.TimeValue, true, 2);
            AppendJsonNumber(sb, "unscaledTime", s.UnscaledTime, true, 2);
            AppendJsonNumber(sb, "readbackFailures", failedReadbacks, true, 2);
            AppendJsonString(sb, "readbackFailedBuffers", readbackErrors, true, 2);
            AppendJsonBool(sb, "qualityGateApplied", false, true, 2);
            AppendJsonBool(sb, "frameDroppedForQuality", false, true, 2);
            AppendJsonBool(sb, "humanAnnotationApplied", false, true, 2);
            AppendJsonString(sb, "classificationStage", "offline_deterministic", true, 2);
            AppendJsonBool(sb, "rawDepthPreserved", buffers.Exists(buffer => buffer.Name == "raw_depth_r32f"), true, 2);
            AppendJsonString(sb, "rawDepthEncoding", "float32 depth01, little-endian, eye slices left then right", true, 2);
            AppendJsonNumber(sb, "expectedEyeCount", EyeCount, true, 2);
            AppendJsonString(sb, "readbackLayout", "gpu_structured_buffer_left_then_right", true, 2);
            sb.AppendLine("  \"sourceTexture\": {");
            AppendJsonString(sb, "type", s.SourceType, true, 4);
            AppendJsonString(sb, "graphicsFormat", s.SourceGraphicsFormat, true, 4);
            AppendJsonString(sb, "dimension", s.SourceDimension, true, 4);
            AppendJsonNumber(sb, "width", s.Width, true, 4);
            AppendJsonNumber(sb, "height", s.Height, true, 4);
            AppendJsonNumber(sb, "eyes", s.SourceEyeCount, true, 4);
            AppendJsonNumber(sb, "mipCount", s.SourceMipCount, true, 4);
            AppendJsonString(sb, "filterMode", s.SourceFilterMode.ToString(), true, 4);
            AppendJsonString(sb, "wrapMode", s.SourceWrapMode.ToString(), false, 4);
            sb.AppendLine("  },");
            AppendVector4(sb, "zBufferParams", s.ZBufferParams, true, 2);
            AppendMatrixArray(sb, "reprojectionMatrices", s.Reprojection, true, 2);
            AppendMatrixArray(sb, "inverseReprojectionMatrices", s.InverseReprojection, true, 2);
            AppendMatrixArray(sb, "stereoViewMatrices", s.StereoView, true, 2);
            AppendMatrixArray(sb, "stereoProjectionMatrices", s.StereoProjection, true, 2);
            AppendVector3Array(sb, "eyeWorldPositions", s.EyeWorldPositions, true, 2);
            AppendMatrix(sb, "cameraToWorld", s.CameraToWorld, true, 2);
            AppendMatrix(sb, "worldToCamera", s.WorldToCamera, true, 2);
            AppendMatrix(sb, "projection", s.Projection, true, 2);
            AppendMatrix(sb, "nonJitteredProjection", s.NonJitteredProjection, true, 2);
            AppendVector3(sb, "cameraPositionAtRequest", s.CameraPosition, true, 2);
            AppendQuaternion(sb, "cameraRotationAtRequest", s.CameraRotation, true, 2);
            AppendVector3(sb, "cameraPositionAtReadback", completionPosition, true, 2);
            AppendQuaternion(sb, "cameraRotationAtReadback", completionRotation, true, 2);
            AppendJsonNumber(sb, "nearClip", s.NearClip, true, 2);
            AppendJsonNumber(sb, "farClip", s.FarClip, true, 2);
            AppendJsonNumber(sb, "fieldOfView", s.FieldOfView, true, 2);
            AppendJsonNumber(sb, "aspect", s.Aspect, true, 2);
            AppendJsonNumber(sb, "cameraPixelWidth", s.CameraPixelWidth, true, 2);
            AppendJsonNumber(sb, "cameraPixelHeight", s.CameraPixelHeight, true, 2);
            AppendTracking(sb, "trackingAtRequest", s.Tracking, true, 2);
            AppendTracking(sb, "trackingAtReadback", completionTracking, true, 2);
            sb.AppendLine("  \"buffersPresent\": [");
            for (int i = 0; i < buffers.Count; ++i)
            {
                sb.Append("    \"").Append(JsonEscape(buffers[i].Name)).Append('"');
                sb.AppendLine(i + 1 < buffers.Count ? "," : string.Empty);
            }
            sb.AppendLine("  ]");
            sb.AppendLine("}");
            return sb.ToString();
        }

        private static RenderTexture CreateArrayTexture(string name, int width, int height, GraphicsFormat format, bool randomWrite)
        {
            var descriptor = new RenderTextureDescriptor(width, height)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = EyeCount,
                graphicsFormat = format,
                depthBufferBits = 0,
                msaaSamples = 1,
                mipCount = 1,
                useMipMap = false,
                autoGenerateMips = false,
                enableRandomWrite = randomWrite,
                sRGB = false
            };
            var texture = new RenderTexture(descriptor)
            {
                name = name,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            texture.Create();
            return texture;
        }

        private static void Dispatch(ComputeShader shader, int kernel, int width, int height, int depth)
        {
            shader.GetKernelThreadGroupSizes(kernel, out uint x, out uint y, out uint z);
            shader.Dispatch(
                kernel,
                Mathf.CeilToInt(width / (float)Mathf.Max(1, (int)x)),
                Mathf.CeilToInt(height / (float)Mathf.Max(1, (int)y)),
                Mathf.CeilToInt(depth / (float)Mathf.Max(1, (int)z)));
        }

        private static void AppendTracking(StringBuilder sb, string name, TrackingSnapshot tracking, bool comma, int indent)
        {
            string pad = new string(' ', indent);
            sb.Append(pad).Append('"').Append(name).AppendLine("\": {");
            AppendNode(sb, "head", tracking.Head, true, indent + 2);
            AppendNode(sb, "centerEye", tracking.CenterEye, true, indent + 2);
            AppendNode(sb, "leftEye", tracking.LeftEye, true, indent + 2);
            AppendNode(sb, "rightEye", tracking.RightEye, false, indent + 2);
            sb.Append(pad).Append('}').AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendNode(StringBuilder sb, string name, XrNodeSnapshot node, bool comma, int indent)
        {
            string pad = new string(' ', indent);
            sb.Append(pad).Append('"').Append(name).AppendLine("\": {");
            AppendJsonBool(sb, "deviceValid", node.DeviceValid, true, indent + 2);
            AppendJsonBool(sb, "isTracked", node.IsTracked, true, indent + 2);
            AppendJsonString(sb, "trackingState", node.TrackingState.ToString(), true, indent + 2);
            AppendVector3(sb, "position", node.Position, true, indent + 2);
            AppendQuaternion(sb, "rotation", node.Rotation, true, indent + 2);
            AppendVector3(sb, "velocity", node.Velocity, true, indent + 2);
            AppendVector3(sb, "angularVelocity", node.AngularVelocity, false, indent + 2);
            sb.Append(pad).Append('}').AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendMatrixArray(StringBuilder sb, string name, Matrix4x4[] values, bool comma, int indent)
        {
            string pad = new string(' ', indent);
            sb.Append(pad).Append('"').Append(name).AppendLine("\": [");
            for (int i = 0; i < values.Length; ++i)
            {
                sb.Append(pad).Append("  ").Append(MatrixJson(values[i]));
                sb.AppendLine(i + 1 < values.Length ? "," : string.Empty);
            }
            sb.Append(pad).Append(']').AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendVector3Array(StringBuilder sb, string name, Vector3[] values, bool comma, int indent)
        {
            string pad = new string(' ', indent);
            sb.Append(pad).Append('"').Append(name).Append("\": [");
            for (int i = 0; i < values.Length; ++i)
            {
                sb.Append(Vector3Json(values[i]));
                if (i + 1 < values.Length) sb.Append(',');
            }
            sb.Append(']').AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendMatrix(StringBuilder sb, string name, Matrix4x4 value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": ").Append(MatrixJson(value)).AppendLine(comma ? "," : string.Empty);
        }

        private static string MatrixJson(Matrix4x4 value)
        {
            var sb = new StringBuilder(256);
            sb.Append('[');
            for (int i = 0; i < 16; ++i)
            {
                if (i > 0) sb.Append(',');
                sb.Append(value[i].ToString("R", CultureInfo.InvariantCulture));
            }
            sb.Append(']');
            return sb.ToString();
        }

        private static void AppendVector3(StringBuilder sb, string name, Vector3 value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": ").Append(Vector3Json(value)).AppendLine(comma ? "," : string.Empty);
        }

        private static string Vector3Json(Vector3 value)
        {
            return "[" + N(value.x) + "," + N(value.y) + "," + N(value.z) + "]";
        }

        private static void AppendVector4(StringBuilder sb, string name, Vector4 value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": [")
                .Append(N(value.x)).Append(',').Append(N(value.y)).Append(',').Append(N(value.z)).Append(',').Append(N(value.w)).Append(']')
                .AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendQuaternion(StringBuilder sb, string name, Quaternion value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": [")
                .Append(N(value.x)).Append(',').Append(N(value.y)).Append(',').Append(N(value.z)).Append(',').Append(N(value.w)).Append(']')
                .AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendJsonString(StringBuilder sb, string name, string value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": \"").Append(JsonEscape(value)).Append('"').AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendJsonNumber(StringBuilder sb, string name, double value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": ").Append(value.ToString("R", CultureInfo.InvariantCulture)).AppendLine(comma ? "," : string.Empty);
        }

        private static void AppendJsonBool(StringBuilder sb, string name, bool value, bool comma, int indent)
        {
            sb.Append(' ', indent).Append('"').Append(name).Append("\": ").Append(value ? "true" : "false").AppendLine(comma ? "," : string.Empty);
        }

        private static string N(float value) => value.ToString("R", CultureInfo.InvariantCulture);

        private static string CsvValue(string value)
        {
            return "\"" + (value ?? string.Empty).Replace("\"", "\"\"") + "\"";
        }

        private static string JsonEscape(string value)
        {
            return (value ?? string.Empty)
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n");
        }
    }
}

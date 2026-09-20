using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.XR.ARSubsystems;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 最小双路逐帧采集器。
    /// A 路是 Meta 平台环境深度进入 QRS 自有预处理之前的 GPU 副本（不是传感器 raw）；
    /// B 路是同一平台帧经过手罩/时序/双边/缘洗后、真正送往融合的深度。
    /// 不做抽点、拟合、体素化或跨帧揉合，离线端可据此裁决误差出生在哪一层。
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PairedDepthFrameRecorder : MonoBehaviour
    {
        private const int EyeCount = 2;
        private const int RecordedLayerCount = 1;
        private const int FusionEyeIndex = DepthCapture.FusionEyeIndex;
        private const int MaxOutstandingPairs = 24;
        private const string CopyShaderResource = "DepthPairCaptureCopy";

        private ComputeShader _copyShader;
        private int _copyKernel = -1;
        private RenderTexture _platformSnapshot;
        private RenderTexture _processedSnapshot;

        private bool _isCapturing;
        private bool _stagedEligible;
        private int _stagedPlatformFrame = -1;
        private FrameMetadata _stagedMetadata;
        private string _sessionDirectory = string.Empty;
        private string _framesDirectory = string.Empty;
        private string _manifestPath = string.Empty;
        private int _pairSequence;
        private int _pendingReadbackPairs;
        private int _writesInFlight;
        private int _droppedPairs;
        private int _readbackErrors;
        private int _writeErrors;
        private bool _depthDrainReported;
        private ScanReplaySessionPackage _sessionPackage;
        private DepthCapture _depthCapture;
        private readonly object _manifestLock = new object();

        private static readonly int SourceDepthId = Shader.PropertyToID("_SourceDepth");
        private static readonly int DestDepthId = Shader.PropertyToID("_DestDepth");
        private static readonly int CaptureSizeId = Shader.PropertyToID("_CaptureSize");
        private static readonly int SourceEyeIndexId = Shader.PropertyToID("_SourceEyeIndex");

        public bool IsCapturing => _isCapturing;
        public string SessionDirectory => _sessionDirectory;
        internal int LocalOutstandingPairCount => Volatile.Read(ref _pendingReadbackPairs) +
                                                  Volatile.Read(ref _writesInFlight);
        public int OutstandingPairCount => LocalOutstandingPairCount +
                                           (_sessionPackage != null ? _sessionPackage.OutstandingCount : 0);
        public int DroppedPairCount => Volatile.Read(ref _droppedPairs);
        public int DroppedFusionFrameCount => _sessionPackage != null
            ? _sessionPackage.DroppedFusionFrames
            : 0;

        public bool StartCapture()
        {
            if (_isCapturing) return true;
            VolumeIntegrator volume = VolumeIntegrator.Instance;
            if (volume != null && volume.IntegrationCount > 0)
            {
                Logger.Warning(
                    "独立复现会话必须从空体积的第一帧开始；当前体积已有融合数据，拒绝中途新开会话。" +
                    "请先完成/清空本次扫描，再开始下一次采集。");
                return false;
            }
            if (!EnsureShader()) return false;
            _depthCapture = GetComponent<DepthCapture>();

            string root = Path.Combine(
                Application.persistentDataPath,
                "ScanCoverDiagnostics",
                "replay_sessions");
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            _sessionDirectory = Path.Combine(root, "session_" + stamp);
            _framesDirectory = Path.Combine(_sessionDirectory, "depth_pairs", "frames");
            _manifestPath = Path.Combine(_sessionDirectory, "depth_pairs", "manifest.csv");

            try
            {
                Directory.CreateDirectory(_framesDirectory);
                File.WriteAllText(_manifestPath,
                    "pairIndex,platformFrame,timestampNs,timestampValid,callbackUnityFrame,callbackUnscaledTime,preprocessUnityFrame,preprocessUnscaledTime,callbackToPreprocessMs,width,height,layers,updatedProcessedEye,angularDegPerSec,linearMps,callbackHeadAvailable,callbackHeadX,callbackHeadY,callbackHeadZ,callbackHeadQx,callbackHeadQy,callbackHeadQz,callbackHeadQw,callbackPitchDeg,callbackYawDeg,callbackRollDeg,preprocessHeadAvailable,preprocessHeadX,preprocessHeadY,preprocessHeadZ,preprocessHeadQx,preprocessHeadQy,preprocessHeadQz,preprocessHeadQw,preprocessPitchDeg,preprocessYawDeg,preprocessRollDeg,callbackToPreprocessPositionMm,callbackToPreprocessRotationDeg,rawTotal,rawValid,rawInvalid,rawMinNdc,rawMaxNdc,rawMeanNdc,rawMinMetres,rawMaxMetres,rawMeanMetres,processedTotal,processedValid,processedInvalid,processedMinNdc,processedMaxNdc,processedMeanNdc,processedMinMetres,processedMaxMetres,processedMeanMetres,pairedValid,processedMinusRawMeanMetres,processedMinusRawMeanAbsMetres,processedMinusRawRmsMetres,processedMinusRawMaxAbsMetres,rawFile,processedFile,metadataFile,status\n",
                    new UTF8Encoding(false));
                File.WriteAllText(
                    Path.Combine(_sessionDirectory, "depth_pairs", "schema.json"),
                    BuildSchemaJson(),
                    new UTF8Encoding(false));
                _sessionPackage = GetComponent<ScanReplaySessionPackage>() ??
                                  gameObject.AddComponent<ScanReplaySessionPackage>();
                _sessionPackage.Begin(_sessionDirectory);
            }
            catch (Exception e)
            {
                Logger.Warning("双路深度采集无法创建目录：" + e.Message);
                _sessionDirectory = string.Empty;
                return false;
            }

            _pairSequence = 0;
            _droppedPairs = 0;
            _readbackErrors = 0;
            _writeErrors = 0;
            _stagedEligible = false;
            _stagedPlatformFrame = -1;
            _depthDrainReported = false;
            _isCapturing = true;
            Logger.Info("双路深度采集开始：" + _sessionDirectory);
            return true;
        }

        public void StopCapture()
        {
            if (!_isCapturing) return;
            _isCapturing = false;
            _stagedEligible = false;
            _sessionPackage?.End();
            WriteStatusFile("stopped");
            Logger.Info($"双路深度采集停止：帧对={_pairSequence}，丢={_droppedPairs}，" +
                        $"回读错={_readbackErrors}，待写={OutstandingPairCount}，目录={_sessionDirectory}");
        }

        public void StagePlatformFrame(
            Texture platformDepth,
            int platformFrame,
            bool timestampValid,
            long timestampNs,
            Pose pose0,
            Pose pose1,
            XRFov fov0,
            XRFov fov1,
            Vector2 nearFar,
            Matrix4x4[] projection,
            Matrix4x4[] projectionInverse,
            Matrix4x4[] view,
            Matrix4x4[] viewInverse)
        {
            _stagedEligible = false;
            _stagedPlatformFrame = platformFrame;
            if (!_isCapturing || platformDepth == null) return;

            if (LocalOutstandingPairCount >= MaxOutstandingPairs)
            {
                Interlocked.Increment(ref _droppedPairs);
                return;
            }

            if (!EnsureSnapshotTextures(platformDepth.width, platformDepth.height))
            {
                Interlocked.Increment(ref _droppedPairs);
                return;
            }

            CopyDepth(platformDepth, _platformSnapshot);
            Camera callbackCamera = Camera.main;
            _stagedMetadata = new FrameMetadata
            {
                platformFrame = platformFrame,
                timestampValid = timestampValid,
                timestampNs = timestampNs,
                unityFrame = Time.frameCount,
                unscaledTime = Time.unscaledTimeAsDouble,
                callbackUnityFrame = Time.frameCount,
                callbackUnscaledTime = Time.unscaledTimeAsDouble,
                callbackHeadAvailable = callbackCamera != null,
                callbackHeadWorldPose = callbackCamera != null
                    ? new Pose(callbackCamera.transform.position, callbackCamera.transform.rotation)
                    : new Pose(Vector3.zero, Quaternion.identity),
                callbackHeadEuler = callbackCamera != null
                    ? callbackCamera.transform.eulerAngles
                    : Vector3.zero,
                nearFar = nearFar,
                poses = new[] { pose0, pose1 },
                fovs = new[] { fov0, fov1 },
                projection = CloneMatrices(projection),
                projectionInverse = CloneMatrices(projectionInverse),
                view = CloneMatrices(view),
                viewInverse = CloneMatrices(viewInverse)
            };
            _stagedEligible = true;
        }

        public void CapturePreprocessedFrame(Texture processedDepth, int platformFrame, int updatedEye)
        {
            if (!_isCapturing || !_stagedEligible || processedDepth == null ||
                platformFrame != _stagedPlatformFrame)
                return;

            _stagedEligible = false;
            if (LocalOutstandingPairCount >= MaxOutstandingPairs)
            {
                Interlocked.Increment(ref _droppedPairs);
                return;
            }

            CopyDepth(processedDepth, _processedSnapshot);
            _stagedMetadata.updatedProcessedEye = updatedEye;
            _stagedMetadata.preprocessUnityFrame = Time.frameCount;
            _stagedMetadata.preprocessUnscaledTime = Time.unscaledTimeAsDouble;
            Camera preprocessCamera = Camera.main;
            _stagedMetadata.preprocessHeadAvailable = preprocessCamera != null;
            _stagedMetadata.preprocessHeadWorldPose = preprocessCamera != null
                ? new Pose(preprocessCamera.transform.position, preprocessCamera.transform.rotation)
                : new Pose(Vector3.zero, Quaternion.identity);
            _stagedMetadata.preprocessHeadEuler = preprocessCamera != null
                ? preprocessCamera.transform.eulerAngles
                : Vector3.zero;
            _stagedMetadata.callbackToPreprocessPositionMm =
                _stagedMetadata.callbackHeadAvailable && _stagedMetadata.preprocessHeadAvailable
                    ? Vector3.Distance(_stagedMetadata.callbackHeadWorldPose.position,
                        _stagedMetadata.preprocessHeadWorldPose.position) * 1000f
                    : float.NaN;
            _stagedMetadata.callbackToPreprocessRotationDeg =
                _stagedMetadata.callbackHeadAvailable && _stagedMetadata.preprocessHeadAvailable
                    ? Quaternion.Angle(_stagedMetadata.callbackHeadWorldPose.rotation,
                        _stagedMetadata.preprocessHeadWorldPose.rotation)
                    : float.NaN;
            if (_depthCapture != null)
            {
                _stagedMetadata.angularDegPerSec = _depthCapture.SmoothedDepthAngularSpeed;
                _stagedMetadata.linearMps = _depthCapture.SmoothedDepthLinearSpeed;
                _stagedMetadata.worldPoses = new Pose[EyeCount];
                for (int eye = 0; eye < EyeCount; eye++)
                    _stagedMetadata.worldPoses[eye] = _depthCapture.TrackingToWorld(_stagedMetadata.poses[eye]);
            }

            int pairIndex = ++_pairSequence;
            var pending = new PendingPair
            {
                pairIndex = pairIndex,
                metadata = _stagedMetadata,
                width = processedDepth.width,
                height = processedDepth.height,
                layers = RecordedLayerCount,
                sessionDirectory = _sessionDirectory,
                framesDirectory = _framesDirectory,
                manifestPath = _manifestPath
            };
            Interlocked.Increment(ref _pendingReadbackPairs);

            AsyncGPUReadback.Request(_platformSnapshot, 0, request =>
            {
                if (request.hasError)
                    pending.hasError = true;
                else
                    pending.platform = request.GetData<float>().ToArray();
                pending.platformDone = true;
                TryComplete(pending);
            });

            AsyncGPUReadback.Request(_processedSnapshot, 0, request =>
            {
                if (request.hasError)
                    pending.hasError = true;
                else
                    pending.processed = request.GetData<float>().ToArray();
                pending.processedDone = true;
                TryComplete(pending);
            });
        }

        private void TryComplete(PendingPair pending)
        {
            if (!pending.platformDone || !pending.processedDone || pending.completionClaimed)
                return;

            pending.completionClaimed = true;
            Interlocked.Decrement(ref _pendingReadbackPairs);
            if (pending.hasError || pending.platform == null || pending.processed == null)
            {
                Interlocked.Increment(ref _readbackErrors);
                AppendManifestError(pending);
                if (!_isCapturing) WriteStatusFile("stopped");
                return;
            }

            Interlocked.Increment(ref _writesInFlight);
            _ = Task.Run(() => WritePair(pending));
        }

        private void WritePair(PendingPair pending)
        {
            string stem = "frame_" + pending.pairIndex.ToString("D6", CultureInfo.InvariantCulture);
            string rawName = stem + "_platform_pre_qrs.f32";
            string processedName = stem + "_post_qrs.f32";
            string metadataName = stem + "_meta.json";
            try
            {
                byte[] rawBytes = FloatsToBytes(pending.platform);
                byte[] processedBytes = FloatsToBytes(pending.processed);
                string metadata = BuildFrameJson(pending);
                string rawPath = Path.Combine(pending.framesDirectory, rawName);
                string processedPath = Path.Combine(pending.framesDirectory, processedName);
                string metadataPath = Path.Combine(pending.framesDirectory, metadataName);
                if (_sessionPackage != null)
                {
                    _sessionPackage.WriteImmutableBytes(rawPath, rawBytes);
                    _sessionPackage.WriteImmutableBytes(processedPath, processedBytes);
                    _sessionPackage.WriteImmutableText(metadataPath, metadata);
                }
                else
                {
                    File.WriteAllBytes(rawPath, rawBytes);
                    File.WriteAllBytes(processedPath, processedBytes);
                    File.WriteAllText(metadataPath, metadata, new UTF8Encoding(false));
                }

                string row = BuildManifestRow(pending, rawName, processedName, metadataName, "ok");
                lock (_manifestLock)
                    File.AppendAllText(pending.manifestPath, row, new UTF8Encoding(false));
            }
            catch (Exception)
            {
                Interlocked.Increment(ref _writeErrors);
                AppendManifestError(pending);
            }
            finally
            {
                Interlocked.Decrement(ref _writesInFlight);
                if (!_isCapturing) WriteStatusFile("stopped");
            }
        }

        private void AppendManifestError(PendingPair pending)
        {
            try
            {
                string row = BuildManifestRow(pending, string.Empty, string.Empty, string.Empty, "error");
                lock (_manifestLock)
                    File.AppendAllText(pending.manifestPath, row, new UTF8Encoding(false));
            }
            catch
            {
                // 诊断出口失败不能反向打断生产扫描。
            }
        }

        private bool EnsureShader()
        {
            if (_copyShader != null && _copyKernel >= 0) return true;
            _copyShader = Resources.Load<ComputeShader>(CopyShaderResource);
            if (_copyShader == null)
            {
                Logger.Warning("双路深度采集缺少 Resources/DepthPairCaptureCopy.compute");
                return false;
            }
            _copyKernel = _copyShader.FindKernel("CopyDepth");
            return true;
        }

        private bool EnsureSnapshotTextures(int width, int height)
        {
            if (_platformSnapshot != null && _platformSnapshot.width == width &&
                _platformSnapshot.height == height && _processedSnapshot != null)
                return true;

            ReleaseSnapshotTextures();
            if (!SystemInfo.IsFormatSupported(GraphicsFormat.R32_SFloat, GraphicsFormatUsage.LoadStore))
            {
                Logger.Warning("双路深度采集：设备不支持 R32_SFloat UAV");
                return false;
            }

            _platformSnapshot = CreateSnapshot(width, height, "DepthPair_Platform");
            _processedSnapshot = CreateSnapshot(width, height, "DepthPair_Processed");
            return _platformSnapshot.IsCreated() && _processedSnapshot.IsCreated();
        }

        private static RenderTexture CreateSnapshot(int width, int height, string name)
        {
            var texture = new RenderTexture(width, height, 0, GraphicsFormat.R32_SFloat, 1)
            {
                name = name,
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = RecordedLayerCount,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            texture.Create();
            return texture;
        }

        private void CopyDepth(Texture source, RenderTexture destination)
        {
            _copyShader.SetTexture(_copyKernel, SourceDepthId, source);
            _copyShader.SetTexture(_copyKernel, DestDepthId, destination);
            _copyShader.SetInts(CaptureSizeId, destination.width, destination.height);
            _copyShader.SetInt(SourceEyeIndexId, FusionEyeIndex);
            _copyShader.Dispatch(
                _copyKernel,
                Mathf.CeilToInt(destination.width / 8f),
                Mathf.CeilToInt(destination.height / 8f),
                RecordedLayerCount);
        }

        private void WriteStatusFile(string state)
        {
            if (string.IsNullOrEmpty(_sessionDirectory)) return;
            try
            {
                string text = "state=" + state + "\n" +
                              "requestedPairs=" + _pairSequence + "\n" +
                              "droppedPairs=" + _droppedPairs + "\n" +
                              "readbackErrors=" + _readbackErrors + "\n" +
                              "writeErrors=" + _writeErrors + "\n" +
                              "outstandingDepthPairs=" + LocalOutstandingPairCount + "\n" +
                              "outstandingSessionWork=" + OutstandingPairCount + "\n" +
                              "droppedFusionFrames=" + DroppedFusionFrameCount + "\n";
                lock (_manifestLock)
                    File.WriteAllText(Path.Combine(_sessionDirectory, "status.txt"), text, new UTF8Encoding(false));
            }
            catch
            {
                // 同上：采集诊断绝不影响生产链。
            }
        }

        private static string BuildManifestRow(
            PendingPair pending,
            string rawName,
            string processedName,
            string metadataName,
            string status)
        {
            FrameMetadata m = pending.metadata;
            Matrix4x4 projection = m.projection != null && m.projection.Length > FusionEyeIndex
                ? m.projection[FusionEyeIndex]
                : Matrix4x4.identity;
            DepthStats raw = CalculateDepthStats(pending.platform, projection);
            DepthStats processed = CalculateDepthStats(pending.processed, projection);
            PairDeltaStats delta = CalculatePairDeltaStats(
                pending.platform, pending.processed, projection);
            return string.Join(",",
                pending.pairIndex.ToString(CultureInfo.InvariantCulture),
                m.platformFrame.ToString(CultureInfo.InvariantCulture),
                m.timestampNs.ToString(CultureInfo.InvariantCulture),
                m.timestampValid ? "1" : "0",
                m.callbackUnityFrame.ToString(CultureInfo.InvariantCulture),
                m.callbackUnscaledTime.ToString("R", CultureInfo.InvariantCulture),
                m.preprocessUnityFrame.ToString(CultureInfo.InvariantCulture),
                m.preprocessUnscaledTime.ToString("R", CultureInfo.InvariantCulture),
                ((m.preprocessUnscaledTime - m.callbackUnscaledTime) * 1000.0)
                    .ToString("R", CultureInfo.InvariantCulture),
                pending.width.ToString(CultureInfo.InvariantCulture),
                pending.height.ToString(CultureInfo.InvariantCulture),
                pending.layers.ToString(CultureInfo.InvariantCulture),
                m.updatedProcessedEye.ToString(CultureInfo.InvariantCulture),
                Format(m.angularDegPerSec),
                Format(m.linearMps),
                m.callbackHeadAvailable ? "1" : "0",
                Format(m.callbackHeadWorldPose.position.x),
                Format(m.callbackHeadWorldPose.position.y),
                Format(m.callbackHeadWorldPose.position.z),
                Format(m.callbackHeadWorldPose.rotation.x),
                Format(m.callbackHeadWorldPose.rotation.y),
                Format(m.callbackHeadWorldPose.rotation.z),
                Format(m.callbackHeadWorldPose.rotation.w),
                Format(m.callbackHeadEuler.x), Format(m.callbackHeadEuler.y), Format(m.callbackHeadEuler.z),
                m.preprocessHeadAvailable ? "1" : "0",
                Format(m.preprocessHeadWorldPose.position.x),
                Format(m.preprocessHeadWorldPose.position.y),
                Format(m.preprocessHeadWorldPose.position.z),
                Format(m.preprocessHeadWorldPose.rotation.x),
                Format(m.preprocessHeadWorldPose.rotation.y),
                Format(m.preprocessHeadWorldPose.rotation.z),
                Format(m.preprocessHeadWorldPose.rotation.w),
                Format(m.preprocessHeadEuler.x), Format(m.preprocessHeadEuler.y), Format(m.preprocessHeadEuler.z),
                Format(m.callbackToPreprocessPositionMm),
                Format(m.callbackToPreprocessRotationDeg),
                raw.Total.ToString(CultureInfo.InvariantCulture),
                raw.Valid.ToString(CultureInfo.InvariantCulture),
                raw.Invalid.ToString(CultureInfo.InvariantCulture),
                Format(raw.MinNdc), Format(raw.MaxNdc), Format(raw.MeanNdc),
                Format(raw.MinMetres), Format(raw.MaxMetres), Format(raw.MeanMetres),
                processed.Total.ToString(CultureInfo.InvariantCulture),
                processed.Valid.ToString(CultureInfo.InvariantCulture),
                processed.Invalid.ToString(CultureInfo.InvariantCulture),
                Format(processed.MinNdc), Format(processed.MaxNdc), Format(processed.MeanNdc),
                Format(processed.MinMetres), Format(processed.MaxMetres), Format(processed.MeanMetres),
                delta.Valid.ToString(CultureInfo.InvariantCulture),
                Format(delta.Mean), Format(delta.MeanAbs), Format(delta.Rms), Format(delta.MaxAbs),
                rawName,
                processedName,
                metadataName,
                status) + "\n";
        }

        private static DepthStats CalculateDepthStats(float[] values, Matrix4x4 projection)
        {
            var result = new DepthStats
            {
                Total = values?.Length ?? 0,
                MinNdc = float.PositiveInfinity,
                MaxNdc = float.NegativeInfinity,
                MinMetres = float.PositiveInfinity,
                MaxMetres = float.NegativeInfinity
            };
            if (values == null) return result.Finish();
            double ndcSum = 0.0;
            double metreSum = 0.0;
            for (int i = 0; i < values.Length; i++)
            {
                float ndc = values[i];
                if (!TryLinearizeMetres(ndc, projection, out float metres))
                {
                    result.Invalid++;
                    continue;
                }
                result.Valid++;
                result.MinNdc = Mathf.Min(result.MinNdc, ndc);
                result.MaxNdc = Mathf.Max(result.MaxNdc, ndc);
                result.MinMetres = Mathf.Min(result.MinMetres, metres);
                result.MaxMetres = Mathf.Max(result.MaxMetres, metres);
                ndcSum += ndc;
                metreSum += metres;
            }
            if (result.Valid > 0)
            {
                result.MeanNdc = (float)(ndcSum / result.Valid);
                result.MeanMetres = (float)(metreSum / result.Valid);
            }
            return result.Finish();
        }

        private static PairDeltaStats CalculatePairDeltaStats(
            float[] raw, float[] processed, Matrix4x4 projection)
        {
            var result = new PairDeltaStats
            {
                Mean = float.NaN,
                MeanAbs = float.NaN,
                Rms = float.NaN,
                MaxAbs = float.NaN
            };
            if (raw == null || processed == null) return result;
            int count = Math.Min(raw.Length, processed.Length);
            double signed = 0.0;
            double absolute = 0.0;
            double squared = 0.0;
            float max = 0f;
            for (int i = 0; i < count; i++)
            {
                if (!TryLinearizeMetres(raw[i], projection, out float a) ||
                    !TryLinearizeMetres(processed[i], projection, out float b))
                    continue;
                float d = b - a;
                float ad = Mathf.Abs(d);
                result.Valid++;
                signed += d;
                absolute += ad;
                squared += d * d;
                max = Mathf.Max(max, ad);
            }
            if (result.Valid > 0)
            {
                result.Mean = (float)(signed / result.Valid);
                result.MeanAbs = (float)(absolute / result.Valid);
                result.Rms = (float)Math.Sqrt(squared / result.Valid);
                result.MaxAbs = max;
            }
            return result;
        }

        private static bool TryLinearizeMetres(
            float ndc, Matrix4x4 projection, out float metres)
        {
            metres = 0f;
            if (float.IsNaN(ndc) || float.IsInfinity(ndc) || ndc <= 0f || ndc >= 1f)
                return false;
            float denominator = ndc * 2f - 1f + projection.m22;
            if (Mathf.Abs(denominator) < 1e-7f) return false;
            metres = Mathf.Abs(projection.m23 / denominator);
            return !float.IsNaN(metres) && !float.IsInfinity(metres) && metres > 0f;
        }

        private struct DepthStats
        {
            public int Total, Valid, Invalid;
            public float MinNdc, MaxNdc, MeanNdc;
            public float MinMetres, MaxMetres, MeanMetres;

            public DepthStats Finish()
            {
                if (Valid == 0)
                {
                    MinNdc = MaxNdc = MeanNdc = float.NaN;
                    MinMetres = MaxMetres = MeanMetres = float.NaN;
                }
                return this;
            }
        }

        private struct PairDeltaStats
        {
            public int Valid;
            public float Mean, MeanAbs, Rms, MaxAbs;
        }

        private static string BuildSchemaJson()
        {
            return "{\n" +
                   "  \"schema\": \"scancover-depth-pair-v1\",\n" +
                   "  \"platformStream\": \"Meta runtime environment depth copied before QRS preprocessing; not sensor raw\",\n" +
                   "  \"processedStream\": \"same frame after hand mask, temporal filter, bilateral filter and edge clean\",\n" +
                   "  \"binaryEncoding\": \"little-endian float32, x-fastest then y then texture-array layer\",\n" +
                   "  \"depthEncoding\": \"normalized device depth (NDC texture value), not metres\",\n" +
                   "  \"linearizeMetres\": \"z=ndc*2-1; metres=abs(projection.m23/(z+projection.m22))\",\n" +
                   "  \"matrixEncoding\": \"row-major m00..m33\",\n" +
                   "  \"pairing\": \"only frames consumed by QRS PreprocessLatestFrame are emitted\",\n" +
                   "  \"timingLineage\": \"platform callback time and current head pose, preprocessing consumption time and current head pose; fusion consumption time is joined by platformFrame/sourceFrame in fusion_inputs\",\n" +
                   "  \"recordedTextureLayers\": 1,\n" +
                   "  \"recordedEyeIndex\": 1,\n" +
                   "  \"eyeMetadata\": \"pose, FOV and matrices are retained for both runtime eyes, while both binary streams contain only fusion eye slice 1 (right)\",\n" +
                   "  \"updatedProcessedEye\": \"fixed to fusion eye slice 1 (right)\"\n" +
                   "}\n";
        }

        private void Update()
        {
            if (_isCapturing || _depthDrainReported || LocalOutstandingPairCount > 0)
                return;
            _depthDrainReported = true;
            // 先写深度流自己的最终状态，再通知会话封口；否则这个文件可能在
            // capture_complete.json 生成之后又被覆盖，破坏校验清单。
            WriteStatusFile("depth_streams_drained");
            _sessionPackage?.NotifyDepthStreamsComplete(
                _droppedPairs, _readbackErrors, _writeErrors);
        }

        private static string BuildFrameJson(PendingPair pending)
        {
            FrameMetadata m = pending.metadata;
            var sb = new StringBuilder(4096);
            sb.AppendLine("{");
            sb.Append("  \"schema\": \"scancover-depth-pair-v1\",\n");
            sb.Append("  \"pairIndex\": ").Append(pending.pairIndex).AppendLine(",");
            sb.Append("  \"platformFrame\": ").Append(m.platformFrame).AppendLine(",");
            sb.Append("  \"timestampValid\": ").Append(m.timestampValid ? "true" : "false").AppendLine(",");
            sb.Append("  \"timestampNs\": ").Append(m.timestampNs).AppendLine(",");
            sb.Append("  \"unityFrame\": ").Append(m.unityFrame).AppendLine(",");
            sb.Append("  \"unscaledTime\": ").Append(Format(m.unscaledTime)).AppendLine(",");
            sb.Append("  \"callbackUnityFrame\": ").Append(m.callbackUnityFrame).AppendLine(",");
            sb.Append("  \"callbackUnscaledTime\": ").Append(Format(m.callbackUnscaledTime)).AppendLine(",");
            sb.Append("  \"preprocessUnityFrame\": ").Append(m.preprocessUnityFrame).AppendLine(",");
            sb.Append("  \"preprocessUnscaledTime\": ").Append(Format(m.preprocessUnscaledTime)).AppendLine(",");
            sb.Append("  \"callbackToPreprocessMs\": ")
                .Append(Format((m.preprocessUnscaledTime - m.callbackUnscaledTime) * 1000.0))
                .AppendLine(",");
            sb.Append("  \"callbackHeadWorldPose\": {\"available\":")
                .Append(m.callbackHeadAvailable ? "true" : "false")
                .Append(",\"pose\":");
            AppendPoseValue(sb, m.callbackHeadWorldPose);
            sb.Append(",\"eulerDegrees\":[")
                .Append(Format(m.callbackHeadEuler.x)).Append(',')
                .Append(Format(m.callbackHeadEuler.y)).Append(',')
                .Append(Format(m.callbackHeadEuler.z)).AppendLine("]},");
            sb.Append("  \"preprocessHeadWorldPose\": {\"available\":")
                .Append(m.preprocessHeadAvailable ? "true" : "false")
                .Append(",\"pose\":");
            AppendPoseValue(sb, m.preprocessHeadWorldPose);
            sb.Append(",\"eulerDegrees\":[")
                .Append(Format(m.preprocessHeadEuler.x)).Append(',')
                .Append(Format(m.preprocessHeadEuler.y)).Append(',')
                .Append(Format(m.preprocessHeadEuler.z)).AppendLine("]},");
            sb.Append("  \"callbackToPreprocessPositionMm\": ")
                .Append(Format(m.callbackToPreprocessPositionMm)).AppendLine(",");
            sb.Append("  \"callbackToPreprocessRotationDeg\": ")
                .Append(Format(m.callbackToPreprocessRotationDeg)).AppendLine(",");
            sb.Append("  \"width\": ").Append(pending.width).AppendLine(",");
            sb.Append("  \"height\": ").Append(pending.height).AppendLine(",");
            sb.Append("  \"layers\": ").Append(pending.layers).AppendLine(",");
            sb.Append("  \"recordedEyeIndex\": ").Append(FusionEyeIndex).AppendLine(",");
            sb.Append("  \"updatedProcessedEye\": ").Append(m.updatedProcessedEye).AppendLine(",");
            sb.Append("  \"angularDegPerSec\": ").Append(Format(m.angularDegPerSec)).AppendLine(",");
            sb.Append("  \"linearMps\": ").Append(Format(m.linearMps)).AppendLine(",");
            sb.Append("  \"farIsInfinite\": ").Append(float.IsInfinity(m.nearFar.y) ? "true" : "false").AppendLine(",");
            sb.Append("  \"nearFarMetres\": [").Append(Format(m.nearFar.x)).Append(',')
                .Append(Format(m.nearFar.y)).AppendLine("],");
            AppendPoseArray(sb, m.poses);
            sb.AppendLine(",");
            sb.Append("  \"worldPoses\": ");
            AppendPoseValues(sb, m.worldPoses ?? m.poses);
            sb.AppendLine(",");
            AppendFovArray(sb, m.fovs);
            sb.AppendLine(",");
            AppendMatrixArray(sb, "projection", m.projection);
            sb.AppendLine(",");
            AppendMatrixArray(sb, "projectionInverse", m.projectionInverse);
            sb.AppendLine(",");
            AppendMatrixArray(sb, "view", m.view);
            sb.AppendLine(",");
            AppendMatrixArray(sb, "viewInverse", m.viewInverse);
            sb.AppendLine();
            sb.AppendLine("}");
            return sb.ToString();
        }

        private static void AppendPoseArray(StringBuilder sb, Pose[] poses)
        {
            sb.Append("  \"trackingPoses\": ");
            AppendPoseValues(sb, poses);
        }

        private static void AppendPoseValues(StringBuilder sb, Pose[] poses)
        {
            sb.Append('[');
            for (int i = 0; i < EyeCount; i++)
            {
                if (i > 0) sb.Append(',');
                Pose p = poses[i];
                sb.Append("{\"position\":[").Append(Format(p.position.x)).Append(',')
                    .Append(Format(p.position.y)).Append(',').Append(Format(p.position.z))
                    .Append("],\"rotation\":[").Append(Format(p.rotation.x)).Append(',')
                    .Append(Format(p.rotation.y)).Append(',').Append(Format(p.rotation.z)).Append(',')
                    .Append(Format(p.rotation.w)).Append("]}");
            }
            sb.Append(']');
        }

        private static void AppendPoseValue(StringBuilder sb, Pose pose)
        {
            sb.Append("{\"position\":[").Append(Format(pose.position.x)).Append(',')
                .Append(Format(pose.position.y)).Append(',').Append(Format(pose.position.z))
                .Append("],\"rotation\":[").Append(Format(pose.rotation.x)).Append(',')
                .Append(Format(pose.rotation.y)).Append(',').Append(Format(pose.rotation.z))
                .Append(',').Append(Format(pose.rotation.w)).Append("]}");
        }

        private static void AppendFovArray(StringBuilder sb, XRFov[] fovs)
        {
            sb.Append("  \"fovRadians\": [");
            for (int i = 0; i < EyeCount; i++)
            {
                if (i > 0) sb.Append(',');
                XRFov f = fovs[i];
                sb.Append('[').Append(Format(f.angleLeft)).Append(',')
                    .Append(Format(f.angleRight)).Append(',').Append(Format(f.angleUp)).Append(',')
                    .Append(Format(f.angleDown)).Append(']');
            }
            sb.Append(']');
        }

        private static void AppendMatrixArray(StringBuilder sb, string name, Matrix4x4[] matrices)
        {
            sb.Append("  \"").Append(name).Append("\": [");
            for (int i = 0; i < EyeCount; i++)
            {
                if (i > 0) sb.Append(',');
                Matrix4x4 m = matrices[i];
                sb.Append('[');
                for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                {
                    if (row != 0 || col != 0) sb.Append(',');
                    sb.Append(Format(m[row, col]));
                }
                sb.Append(']');
            }
            sb.Append(']');
        }

        private static Matrix4x4[] CloneMatrices(Matrix4x4[] source)
        {
            return new[] { source[0], source[1] };
        }

        private static byte[] FloatsToBytes(float[] values)
        {
            var bytes = new byte[values.Length * sizeof(float)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        private static string Format(float value) => float.IsNaN(value) || float.IsInfinity(value)
            ? "null"
            : value.ToString("R", CultureInfo.InvariantCulture);
        private static string Format(double value) => double.IsNaN(value) || double.IsInfinity(value)
            ? "null"
            : value.ToString("R", CultureInfo.InvariantCulture);

        private void OnDestroy()
        {
            if (_isCapturing) StopCapture();
            ReleaseSnapshotTextures();
        }

        private void ReleaseSnapshotTextures()
        {
            if (_platformSnapshot != null)
            {
                _platformSnapshot.Release();
                Destroy(_platformSnapshot);
                _platformSnapshot = null;
            }
            if (_processedSnapshot != null)
            {
                _processedSnapshot.Release();
                Destroy(_processedSnapshot);
                _processedSnapshot = null;
            }
        }

        private sealed class PendingPair
        {
            public int pairIndex;
            public FrameMetadata metadata;
            public int width;
            public int height;
            public int layers;
            public string sessionDirectory;
            public string framesDirectory;
            public string manifestPath;
            public float[] platform;
            public float[] processed;
            public bool platformDone;
            public bool processedDone;
            public bool completionClaimed;
            public bool hasError;
        }

        private struct FrameMetadata
        {
            public int platformFrame;
            public bool timestampValid;
            public long timestampNs;
            public int unityFrame;
            public double unscaledTime;
            public int callbackUnityFrame;
            public double callbackUnscaledTime;
            public int preprocessUnityFrame;
            public double preprocessUnscaledTime;
            public bool callbackHeadAvailable;
            public Pose callbackHeadWorldPose;
            public Vector3 callbackHeadEuler;
            public bool preprocessHeadAvailable;
            public Pose preprocessHeadWorldPose;
            public Vector3 preprocessHeadEuler;
            public float callbackToPreprocessPositionMm;
            public float callbackToPreprocessRotationDeg;
            public int updatedProcessedEye;
            public Vector2 nearFar;
            public Pose[] poses;
            public Pose[] worldPoses;
            public XRFov[] fovs;
            public Matrix4x4[] projection;
            public Matrix4x4[] projectionInverse;
            public Matrix4x4[] view;
            public Matrix4x4[] viewInverse;
            public float angularDegPerSec;
            public float linearMps;
        }
    }
}

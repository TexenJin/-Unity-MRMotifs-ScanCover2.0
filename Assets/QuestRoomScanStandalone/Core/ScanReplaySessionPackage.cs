using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Meta.XR.MRUtilityKit;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.XR.CoreUtils;

namespace Genesis.RoomScan
{
    /// <summary>
    /// “可独立复现本次扫描”会话的总封装器。它只旁路复制生产链已经消费的输入、
    /// 裁决与结果，不参与深度清洗、枪胶、TSDF 或出网判决。
    /// </summary>
    [DisallowMultipleComponent]
    internal sealed class ScanReplaySessionPackage : MonoBehaviour
    {
        internal const string Schema = "scancover.replay_session.v1";
        private const int RecordedEye = DepthCapture.FusionEyeIndex;
        private const int MaxOutstandingFusionFrames = 12;
        private const float SystemRoomMeshStartupWaitSeconds = 2f;
        private const float SystemRoomMeshReloadWaitSeconds = 12f;

        private readonly object _fileLock = new object();
        private readonly ConcurrentDictionary<string, string> _immutableFileHashes =
            new ConcurrentDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        private string _sessionDirectory = string.Empty;
        private string _fusionDirectory = string.Empty;
        private string _fusionManifest = string.Empty;
        private bool _active;
        private bool _endRequested;
        private bool _depthStreamsComplete;
        private bool _finalized;
        private Task _finalizeTask;
        private bool _finalizeClean;
        private int _fusionSequence;
        private int _fusionReadbacks;
        private int _fusionWrites;
        private int _fusionDropped;
        private int _fusionReadbackErrors;
        private int _fusionWriteErrors;
        private int _depthPairDrops;
        private int _depthPairReadbackErrors;
        private int _depthPairWriteErrors;
        private int _artifactRequests;
        private string _startedUtc = string.Empty;
        private string _stoppedUtc = string.Empty;

        internal static ScanReplaySessionPackage Active { get; private set; }
        internal bool IsActive => _active;
        internal int OutstandingCount => Volatile.Read(ref _fusionReadbacks) +
                                         Volatile.Read(ref _fusionWrites) +
                                         Volatile.Read(ref _artifactRequests);
        internal int DroppedFusionFrames => Volatile.Read(ref _fusionDropped);

        internal void Begin(string sessionDirectory)
        {
            _sessionDirectory = sessionDirectory ?? string.Empty;
            _fusionDirectory = Path.Combine(_sessionDirectory, "fusion_inputs", "frames");
            _fusionManifest = Path.Combine(_sessionDirectory, "fusion_inputs", "manifest.csv");
            Directory.CreateDirectory(_fusionDirectory);
            Directory.CreateDirectory(Path.Combine(_sessionDirectory, "artifacts"));
            Directory.CreateDirectory(Path.Combine(_sessionDirectory, "system_reference"));

            _startedUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
            _stoppedUtc = string.Empty;
            _fusionSequence = 0;
            _fusionDropped = 0;
            _fusionReadbackErrors = 0;
            _fusionWriteErrors = 0;
            _depthPairDrops = 0;
            _depthPairReadbackErrors = 0;
            _depthPairWriteErrors = 0;
            _endRequested = false;
            _depthStreamsComplete = false;
            _finalized = false;
            _finalizeTask = null;
            _finalizeClean = false;
            _immutableFileHashes.Clear();
            _active = true;
            Active = this;

            File.WriteAllText(_fusionManifest,
                "sequence,attemptIndex,sourceFrame,unityFrame,scaledTime,unscaledTime,accepted,decision,guarded,gunGelAdmissionActive,gunGelFrame,translationMm,rotationDeg,angularDegPerSec,linearMps,motionQuality,cameraAvailable,depthFile,normalFile,dilatedFile,edgeReasonFile,temporalReasonFile,cameraFile,gunGelObservationsFile,gunGelCorrespondencesFile,metaFile,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "session_schema.json"),
                BuildSessionSchemaJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "replay_contract.json"),
                BuildReplayContractJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "README.txt"),
                "ScanCover 独立回放会话\n" +
                "1. capture_complete.json 不存在时，本会话尚未封口，不得作为确定性样本。\n" +
                "2. 先运行 Tools/ScanCoverReplaySession.py 校验全部 SHA-256。\n" +
                "3. depth_pairs 保存平台前处理前/QRS 后处理后同帧右眼深度。\n" +
                "4. fusion_inputs 保存实际接纳帧的深度、法线、膨胀供体、判因纹理、枪胶逐点缓冲及实际 RGB/相机参数。\n" +
                "5. production_config 与 coordinate_contract 是回放契约；禁止用默认参数替代。\n" +
                "6. system_reference/status.json 说明外部系统网格交接状态；旧 Scene 导出的 OBJ 仅作对照，不是生产真值。\n",
                new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "production_config.json"),
                BuildProductionConfigJson(), new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "coordinate_contract.json"),
                BuildCoordinateContractJson("start"), new UTF8Encoding(false));
            WriteState("capturing");
        }

        internal void End()
        {
            if (!_active || _endRequested) return;
            _active = false;
            _endRequested = true;
            _stoppedUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
            if (Active == this) Active = null;

            try
            {
                File.WriteAllText(Path.Combine(_sessionDirectory, "coordinate_contract_stop.json"),
                    BuildCoordinateContractJson("stop"), new UTF8Encoding(false));
            }
            catch (Exception e)
            {
                WriteArtifactStatus("coordinate_stop", "failed", e.Message, string.Empty);
            }

            CaptureOnlineArtifacts();
            // 系统房间网格由独立的 Meta Scene Mesh 场景导出。它不是本次生产会话
            // 的必要组成，也不得再卡住 depth/fusion/artifacts 已经排空后的封口。
            WriteSystemReferenceStatus(
                "external_required",
                "Export separately from Assets/MRMotifs/ScanCover/Scene/Meta Scene Mesh.unity",
                0, 0, 0);
            WriteState("draining");
        }

        internal void NotifyDepthStreamsComplete(
            int droppedPairs, int readbackErrors, int writeErrors)
        {
            _depthPairDrops = Mathf.Max(0, droppedPairs);
            _depthPairReadbackErrors = Mathf.Max(0, readbackErrors);
            _depthPairWriteErrors = Mathf.Max(0, writeErrors);
            _depthStreamsComplete = true;
            TryFinalize();
        }

        private void Update()
        {
            if (_finalizeTask != null && _finalizeTask.IsCompleted)
            {
                Task completed = _finalizeTask;
                _finalizeTask = null;
                _finalized = true;
                if (completed.IsFaulted)
                {
                    string issue = completed.Exception?.GetBaseException().Message ?? "unknown finalize failure";
                    WriteState("finalize_failed:" + completed.Exception?.GetBaseException().GetType().Name);
                    Logger.Warning("独立回放会话封口失败：" + issue);
                }
                else
                {
                    Logger.Info((_finalizeClean ? "独立回放会话已封口：" : "独立回放会话已排空但存在采集损失：") +
                                _sessionDirectory);
                }
            }
            if (_endRequested && !_finalized)
                TryFinalize();
        }

        internal void RecordDecisionOnly(
            int attemptIndex,
            int sourceFrame,
            bool accepted,
            string decision,
            bool guarded,
            int gunGelFrame,
            float translationMm,
            float rotationDeg,
            float angularSpeed,
            float linearSpeed,
            float motionQuality)
        {
            if (!_active) return;
            int sequence = Interlocked.Increment(ref _fusionSequence);
            var record = new FusionRecord(sequence, attemptIndex, sourceFrame, accepted,
                decision, guarded, gunGelFrame, translationMm, rotationDeg,
                angularSpeed, linearSpeed, motionQuality, null, null, null, null, null,
                false, false, Vector3.zero, Quaternion.identity, Vector2.zero,
                Vector2.zero, Vector2.zero, Vector2.zero, Matrix4x4.identity, null, 0);
            AppendFusionManifest(record, string.Empty, string.Empty, string.Empty,
                string.Empty, string.Empty, string.Empty, string.Empty, string.Empty, string.Empty,
                "decision_only");
        }

        internal void RecordAcceptedInput(
            int attemptIndex,
            int sourceFrame,
            string decision,
            bool guarded,
            int gunGelFrame,
            float translationMm,
            float rotationDeg,
            float angularSpeed,
            float linearSpeed,
            float motionQuality,
            Texture depth,
            Texture normal,
            Texture dilated,
            Texture edgeReason,
            Texture temporalReason,
            Matrix4x4[] projection,
            Matrix4x4[] view,
            Matrix4x4[] projectionInverse,
            Matrix4x4[] viewInverse,
            bool gunGelAdmissionActive,
            ComputeBuffer gunGelObservations,
            ComputeBuffer gunGelCorrespondences,
            int gunGelGridX,
            int gunGelGridY,
            int gunGelPixelStride,
            bool cameraAvailable,
            Texture cameraRgb,
            Vector3 cameraPosition,
            Quaternion cameraRotation,
            Vector2 cameraFocalLength,
            Vector2 cameraPrincipalPoint,
            Vector2 cameraSensorResolution,
            Vector2 cameraCurrentResolution,
            Matrix4x4 fusionCorrection,
            Vector4[] exclusionPositions,
            int exclusionCount)
        {
            if (!_active) return;
            if (OutstandingCount >= MaxOutstandingFusionFrames * 8)
            {
                Interlocked.Increment(ref _fusionDropped);
                RecordDecisionOnly(attemptIndex, sourceFrame, false,
                    "capture_queue_full:" + decision, guarded, gunGelFrame,
                    translationMm, rotationDeg, angularSpeed, linearSpeed, motionQuality);
                return;
            }

            int sequence = Interlocked.Increment(ref _fusionSequence);
            var record = new FusionRecord(sequence, attemptIndex, sourceFrame, true,
                decision, guarded, gunGelFrame, translationMm, rotationDeg,
                angularSpeed, linearSpeed, motionQuality,
                CloneMatrices(projection), CloneMatrices(view),
                CloneMatrices(projectionInverse), CloneMatrices(viewInverse),
                new PendingTextureSet(), gunGelAdmissionActive, cameraAvailable,
                cameraPosition, cameraRotation, cameraFocalLength,
                cameraPrincipalPoint, cameraSensorResolution, cameraCurrentResolution,
                fusionCorrection, CloneVectors(exclusionPositions, exclusionCount),
                Mathf.Clamp(exclusionCount, 0, 64),
                gunGelGridX, gunGelGridY, gunGelPixelStride);

            RequestTexture(record, "depth", depth, RecordedEye);
            RequestTexture(record, "normal", normal, RecordedEye);
            RequestTexture(record, "dilated", dilated, 0);
            RequestTexture(record, "edge", edgeReason, RecordedEye);
            if (temporalReason != null)
                RequestTexture(record, "temporal", temporalReason, RecordedEye);
            else
                record.Textures.temporalDone = true;
            if (cameraAvailable && cameraRgb != null)
                RequestTexture(record, "camera", cameraRgb, 0);
            else
                record.Textures.cameraDone = true;
            if (gunGelAdmissionActive && gunGelObservations != null && gunGelCorrespondences != null)
            {
                RequestBuffer(record, "gungel_observations", gunGelObservations);
                RequestBuffer(record, "gungel_correspondences", gunGelCorrespondences);
            }
            else
            {
                record.Textures.gunGelObservationsDone = true;
                record.Textures.gunGelCorrespondencesDone = true;
            }
            TryCompleteFusion(record);
        }

        private void RequestBuffer(FusionRecord record, string role, ComputeBuffer buffer)
        {
            record.Textures.SetBufferDescriptor(role, buffer.count, buffer.stride);
            Interlocked.Increment(ref _fusionReadbacks);
            try
            {
                AsyncGPUReadback.Request(buffer, request =>
                {
                    try
                    {
                        if (request.hasError)
                            record.Textures.MarkError(role);
                        else
                            record.Textures.SetBytes(role, request.GetData<byte>().ToArray());
                    }
                    catch
                    {
                        record.Textures.MarkError(role);
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _fusionReadbacks);
                        TryCompleteFusion(record);
                    }
                });
            }
            catch
            {
                Interlocked.Decrement(ref _fusionReadbacks);
                record.Textures.MarkError(role);
                TryCompleteFusion(record);
            }
        }

        private void RequestTexture(FusionRecord record, string role, Texture texture, int requestedSlice)
        {
            if (texture == null)
            {
                record.Textures.MarkMissing(role);
                return;
            }

            int slice = texture.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray
                ? Mathf.Clamp(requestedSlice, 0, Mathf.Max(0, GetTextureDepth(texture) - 1))
                : 0;
            var descriptor = new TextureDescriptor(texture.width, texture.height,
                texture.graphicsFormat.ToString(), texture.dimension.ToString(), slice);
            record.Textures.SetDescriptor(role, descriptor);
            Interlocked.Increment(ref _fusionReadbacks);
            try
            {
                AsyncGPUReadback.Request(texture, 0,
                    0, texture.width, 0, texture.height, slice, 1,
                    request =>
                    {
                        try
                        {
                            if (request.hasError)
                                record.Textures.MarkError(role);
                            else
                                record.Textures.SetBytes(role, request.GetData<byte>().ToArray());
                        }
                        catch
                        {
                            record.Textures.MarkError(role);
                        }
                        finally
                        {
                            Interlocked.Decrement(ref _fusionReadbacks);
                            TryCompleteFusion(record);
                        }
                    });
            }
            catch
            {
                Interlocked.Decrement(ref _fusionReadbacks);
                record.Textures.MarkError(role);
                TryCompleteFusion(record);
            }
        }

        private static int GetTextureDepth(Texture texture)
        {
            if (texture is RenderTexture renderTexture) return renderTexture.volumeDepth;
            if (texture is Texture2DArray array) return array.depth;
            return 1;
        }

        private void TryCompleteFusion(FusionRecord record)
        {
            if (!record.Textures.AllDone || record.CompletionClaimed) return;
            record.CompletionClaimed = true;
            Interlocked.Increment(ref _fusionWrites);
            _ = Task.Run(() => WriteFusionRecord(record));
        }

        private void WriteFusionRecord(FusionRecord record)
        {
            string stem = "fusion_" + record.Sequence.ToString("D6", CultureInfo.InvariantCulture);
            string depthName = string.Empty;
            string normalName = string.Empty;
            string dilatedName = string.Empty;
            string edgeName = string.Empty;
            string temporalName = string.Empty;
            string cameraName = string.Empty;
            string observationsName = string.Empty;
            string correspondencesName = string.Empty;
            string metaName = stem + "_meta.json";
            string status = record.Textures.HasError ? "partial_readback_error" : "ok";
            if (record.Textures.HasError)
                Interlocked.Increment(ref _fusionReadbackErrors);
            try
            {
                depthName = WriteTextureFile(stem, "depth", record.Textures.depth);
                normalName = WriteTextureFile(stem, "normal", record.Textures.normal);
                dilatedName = WriteTextureFile(stem, "dilated", record.Textures.dilated);
                edgeName = WriteTextureFile(stem, "edge_reason", record.Textures.edge);
                temporalName = WriteTextureFile(stem, "temporal_reason", record.Textures.temporal);
                cameraName = WriteTextureFile(stem, "camera_rgb", record.Textures.camera);
                observationsName = WriteTextureFile(stem, "gungel_observations", record.Textures.gunGelObservations);
                correspondencesName = WriteTextureFile(stem, "gungel_correspondences", record.Textures.gunGelCorrespondences);
                WriteImmutableText(Path.Combine(_fusionDirectory, metaName),
                    BuildFusionMetaJson(record, depthName, normalName, dilatedName,
                        edgeName, temporalName, cameraName, observationsName, correspondencesName,
                        status));
                AppendFusionManifest(record, depthName, normalName, dilatedName,
                    edgeName, temporalName, cameraName, observationsName, correspondencesName,
                    metaName, status);
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _fusionWriteErrors);
                try
                {
                    AppendFusionManifest(record, depthName, normalName, dilatedName,
                        edgeName, temporalName, cameraName, observationsName, correspondencesName,
                        string.Empty, "write_error:" + Csv(e.GetType().Name));
                }
                catch { }
            }
            finally
            {
                Interlocked.Decrement(ref _fusionWrites);
            }
        }

        private string WriteTextureFile(string stem, string role, byte[] bytes)
        {
            if (bytes == null || bytes.Length == 0) return string.Empty;
            string name = stem + "_" + role + ".bin";
            WriteImmutableBytes(Path.Combine(_fusionDirectory, name), bytes);
            return name;
        }

        /// <summary>
        /// 写入之后不会再变化的大文件在内存中顺手计算 SHA-256。封口时直接复用，
        /// 避免 Quest 再把整套逐帧深度和融合纹理从存储读一遍。
        /// </summary>
        internal void WriteImmutableBytes(string path, byte[] bytes)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));
            File.WriteAllBytes(path, bytes);
            RegisterImmutableHash(path, bytes);
        }

        internal void WriteImmutableText(string path, string text)
        {
            byte[] bytes = new UTF8Encoding(false).GetBytes(text ?? string.Empty);
            WriteImmutableBytes(path, bytes);
        }

        private void RegisterImmutableHash(string path, byte[] bytes)
        {
            using SHA256 sha = SHA256.Create();
            _immutableFileHashes[Path.GetFullPath(path)] = Hex(sha.ComputeHash(bytes));
        }

        private void AppendFusionManifest(FusionRecord r, string depth, string normal,
            string dilated, string edge, string temporal, string camera, string observations,
            string correspondences, string meta, string status)
        {
            string row = string.Join(",",
                r.Sequence.ToString(CultureInfo.InvariantCulture),
                r.AttemptIndex.ToString(CultureInfo.InvariantCulture),
                r.SourceFrame.ToString(CultureInfo.InvariantCulture),
                r.UnityFrame.ToString(CultureInfo.InvariantCulture),
                D(r.ScaledTime), D(r.UnscaledTime),
                r.Accepted ? "1" : "0",
                Csv(r.Decision),
                r.Guarded ? "1" : "0",
                r.GunGelAdmissionActive ? "1" : "0",
                r.GunGelFrame.ToString(CultureInfo.InvariantCulture),
                F(r.TranslationMm), F(r.RotationDeg), F(r.AngularSpeed),
                F(r.LinearSpeed), F(r.MotionQuality), r.CameraAvailable ? "1" : "0",
                depth, normal, dilated, edge, temporal, camera, observations,
                correspondences, meta, Csv(status)) + "\n";
            lock (_fileLock)
                File.AppendAllText(_fusionManifest, row, new UTF8Encoding(false));
        }

        private void CaptureOnlineArtifacts()
        {
            try
            {
                MeshExtractor mesh = MeshExtractor.Instance;
                if (mesh != null)
                {
                    string geometry = mesh.ExportFirstStageGeometrySnapshot("独立回放会话停止");
                    if (!string.IsNullOrEmpty(geometry))
                    {
                        Interlocked.Increment(ref _artifactRequests);
                        Task.Run(() =>
                        {
                            try { CopyGeometryFamily(geometry, Path.Combine(_sessionDirectory, "artifacts", "geometry_stage1")); }
                            finally { Interlocked.Decrement(ref _artifactRequests); }
                        });
                    }

                    Interlocked.Increment(ref _artifactRequests);
                    bool requested = mesh.RequestPaperAuditExport("独立回放会话停止", path =>
                    {
                        Task.Run(() =>
                        {
                            try { CopyArtifact(path, "paper_audit"); }
                            finally { Interlocked.Decrement(ref _artifactRequests); }
                        });
                    });
                    if (!requested)
                    {
                        Interlocked.Decrement(ref _artifactRequests);
                        WriteArtifactStatus("paper_audit", "unavailable", "paper audit not ready", string.Empty);
                    }
                }
                else
                {
                    WriteArtifactStatus("geometry_stage1", "unavailable", "MeshExtractor missing", string.Empty);
                    WriteArtifactStatus("paper_audit", "unavailable", "MeshExtractor missing", string.Empty);
                }

                VolumeIntegrator volume = VolumeIntegrator.Instance;
                if (volume != null)
                {
                    Interlocked.Increment(ref _artifactRequests);
                    bool requested = volume.RequestGunGelCandidateAuditExport("独立回放会话停止", path =>
                    {
                        Task.Run(() =>
                        {
                            try { CopyArtifact(path, "gungel_candidate_audit"); }
                            finally { Interlocked.Decrement(ref _artifactRequests); }
                        });
                    });
                    if (!requested)
                    {
                        Interlocked.Decrement(ref _artifactRequests);
                        WriteArtifactStatus("gungel_candidate_audit", "unavailable", "GunGel audit not ready", string.Empty);
                    }
                }
                else
                {
                    WriteArtifactStatus("gungel_candidate_audit", "unavailable", "VolumeIntegrator missing", string.Empty);
                }
            }
            catch (Exception e)
            {
                WriteArtifactStatus("online_artifacts", "failed", e.Message, string.Empty);
            }
        }

        private void CopyGeometryFamily(string summaryPath, string destination)
        {
            string sourceDirectory = Path.GetDirectoryName(summaryPath) ?? string.Empty;
            string name = Path.GetFileNameWithoutExtension(summaryPath);
            if (name.EndsWith("_summary", StringComparison.OrdinalIgnoreCase))
                name = name.Substring(0, name.Length - "_summary".Length);
            Directory.CreateDirectory(destination);
            foreach (string file in Directory.GetFiles(sourceDirectory, name + "*", SearchOption.TopDirectoryOnly))
                File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), true);
            WriteArtifactStatus("geometry_stage1", "copied", string.Empty, "artifacts/geometry_stage1");
        }

        private void CopyArtifact(string sourcePath, string destinationName)
        {
            if (string.IsNullOrEmpty(sourcePath))
            {
                WriteArtifactStatus(destinationName, "unavailable", "export returned empty path", string.Empty);
                return;
            }
            string source = Directory.Exists(sourcePath)
                ? sourcePath
                : Path.GetDirectoryName(sourcePath) ?? string.Empty;
            if (!Directory.Exists(source))
            {
                WriteArtifactStatus(destinationName, "failed", "source path missing: " + sourcePath, string.Empty);
                return;
            }
            string destination = Path.Combine(_sessionDirectory, "artifacts", destinationName);
            CopyDirectory(source, destination);
            WriteArtifactStatus(destinationName, "copied", string.Empty,
                "artifacts/" + destinationName);
        }

        private IEnumerator ExportSystemRoomMeshAsync()
        {
            // StartCoroutine executes synchronously until the first yield.  Yield first so the
            // A-button transaction can return before touching MRUK or allocating mesh arrays.
            yield return null;
            string directory = Path.Combine(_sessionDirectory, "system_reference");
            Directory.CreateDirectory(directory);
            try
            {
                MRUK mruk = null;
                Exception lookupFailure = null;
                try
                {
                    mruk = MRUK.Instance != null
                        ? MRUK.Instance
                        : FindAnyObjectByType<MRUK>(FindObjectsInactive.Include);
                }
                catch (Exception e) { lookupFailure = e; }
                if (lookupFailure != null)
                {
                    WriteSystemReferenceStatus("failed",
                        lookupFailure.GetType().Name + ": " + lookupFailure.Message, 0, 0, 0);
                    yield break;
                }
                if (mruk == null)
                {
                    WriteSystemReferenceStatus("unavailable", "MRUK component unavailable in production scene", 0, 0, 0);
                    yield break;
                }

                // The MRUK object is intentionally active from scene start, but USE_SCENE
                // permission and device discovery complete asynchronously.  A fast A press
                // must not turn that ordinary startup race into a permanent empty reference.
                float startupDeadline = Time.realtimeSinceStartup + SystemRoomMeshStartupWaitSeconds;
                while ((mruk.Rooms == null || mruk.Rooms.Count == 0) &&
                       Time.realtimeSinceStartup < startupDeadline)
                    yield return null;

                string reloadResult = "not_needed";
                if (mruk.Rooms == null || mruk.Rooms.Count == 0)
                {
                    Task<MRUK.LoadDeviceResult> reloadTask = null;
                    try
                    {
                        // Never open Space Setup from a freeze transaction.  We only query
                        // the room the user already created in Quest settings.  V2FallbackV1
                        // accepts both high-fidelity and ordinary system room captures.
                        reloadTask = mruk.LoadSceneFromDevice(
                            requestSceneCaptureIfNoDataFound: false,
                            removeMissingRooms: true,
                            sceneModel: MRUK.SceneModel.V2FallbackV1);
                    }
                    catch (Exception e)
                    {
                        reloadResult = e.GetType().Name + ": " + e.Message;
                    }

                    if (reloadTask != null)
                    {
                        float reloadDeadline = Time.realtimeSinceStartup + SystemRoomMeshReloadWaitSeconds;
                        while (!reloadTask.IsCompleted && Time.realtimeSinceStartup < reloadDeadline)
                            yield return null;

                        if (!reloadTask.IsCompleted)
                            reloadResult = "timeout";
                        else if (reloadTask.IsFaulted)
                        {
                            Exception failure = reloadTask.Exception?.GetBaseException();
                            reloadResult = (failure?.GetType().Name ?? "load_failed") + ": " +
                                           (failure?.Message ?? "unknown MRUK load failure");
                        }
                        else
                        {
                            reloadResult = reloadTask.Result.ToString();
                            // DiscoveryOngoing means MRUK's startup request owns the query.
                            // Success can also precede Unity-side room instantiation by a frame.
                            if (reloadTask.Result == MRUK.LoadDeviceResult.DiscoveryOngoing ||
                                reloadTask.Result == MRUK.LoadDeviceResult.Success)
                            {
                                while ((mruk.Rooms == null || mruk.Rooms.Count == 0) &&
                                       Time.realtimeSinceStartup < reloadDeadline)
                                    yield return null;
                            }
                        }
                    }
                }

                if (mruk.Rooms == null || mruk.Rooms.Count == 0)
                {
                    WriteSystemReferenceStatus("unavailable",
                        "MRUK active but no rooms loaded; initialized=" + mruk.IsInitialized +
                        ", reloadResult=" + reloadResult, 0, 0, 0);
                    yield break;
                }

                var snapshots = new List<SystemMeshSnapshot>(mruk.Rooms.Count);
                for (int roomIndex = 0; roomIndex < mruk.Rooms.Count; roomIndex++)
                {
                    MRUKRoom room = mruk.Rooms[roomIndex];
                    MRUKAnchor anchor = room != null ? room.GlobalMeshAnchor : null;
                    if (anchor == null) continue;
                    Mesh mesh;
                    try { mesh = anchor.GlobalMesh; }
                    catch { continue; }
                    if (mesh == null || mesh.vertexCount == 0) continue;

                    // Unity mesh objects stay on the main thread.  Only immutable managed arrays
                    // cross to the worker that performs the expensive OBJ formatting and I/O.
                    Vector3[] vertices = mesh.vertices;
                    int[] triangles = mesh.triangles;
                    if (triangles == null || triangles.Length < 3) continue;
                    snapshots.Add(new SystemMeshSnapshot(roomIndex, anchor.name,
                        anchor.transform.localToWorldMatrix, vertices, triangles));
                    yield return null;
                }

                Task writeTask = Task.Run(() => WriteSystemRoomMeshSnapshots(directory, snapshots));
                while (!writeTask.IsCompleted)
                    yield return null;
                if (writeTask.IsFaulted)
                {
                    Exception failure = writeTask.Exception?.GetBaseException() ??
                                        new IOException("system mesh export failed");
                    WriteSystemReferenceStatus("failed",
                        failure.GetType().Name + ": " + failure.Message, 0, 0, 0);
                }
            }
            finally
            {
                Interlocked.Decrement(ref _artifactRequests);
            }
        }

        private void WriteSystemRoomMeshSnapshots(string directory, List<SystemMeshSnapshot> snapshots)
        {
            string objPath = Path.Combine(directory, "quest_system_room_mesh_world.obj");
            string provenancePath = Path.Combine(directory, "quest_system_room_mesh_provenance.json");
            int vertexCount = 0;
            int triangleCount = 0;
            var provenance = new StringBuilder(8192);
            provenance.Append("{\n  \"schema\": \"scancover.system_room_mesh_reference.v1\",\n  \"meshes\": [\n");
            using (var writer = new StreamWriter(objPath, false, new UTF8Encoding(false)))
            {
                writer.WriteLine("# Quest MRUK GlobalMeshAnchor meshes baked to Unity world space");
                int offset = 0;
                for (int meshIndex = 0; meshIndex < snapshots.Count; meshIndex++)
                {
                    SystemMeshSnapshot snapshot = snapshots[meshIndex];
                    writer.WriteLine("o room_" + snapshot.RoomIndex.ToString("D2", CultureInfo.InvariantCulture));
                    for (int i = 0; i < snapshot.Vertices.Length; i++)
                    {
                        Vector3 v = snapshot.LocalToWorld.MultiplyPoint3x4(snapshot.Vertices[i]);
                        writer.WriteLine("v " + F(v.x) + " " + F(v.y) + " " + F(v.z));
                    }
                    for (int i = 0; i + 2 < snapshot.Triangles.Length; i += 3)
                    {
                        writer.WriteLine("f " + (snapshot.Triangles[i] + 1 + offset) + " " +
                                         (snapshot.Triangles[i + 1] + 1 + offset) + " " +
                                         (snapshot.Triangles[i + 2] + 1 + offset));
                    }

                    if (meshIndex > 0) provenance.Append(",\n");
                    provenance.Append("    {\"roomIndex\":").Append(snapshot.RoomIndex)
                        .Append(",\"anchorName\":\"").Append(Json(snapshot.AnchorName)).Append("\"")
                        .Append(",\"vertexCount\":").Append(snapshot.Vertices.Length)
                        .Append(",\"triangleCount\":").Append(snapshot.Triangles.Length / 3)
                        .Append(",\"localToWorld\":");
                    AppendMatrix(provenance, snapshot.LocalToWorld);
                    provenance.Append('}');
                    offset += snapshot.Vertices.Length;
                    vertexCount += snapshot.Vertices.Length;
                    triangleCount += snapshot.Triangles.Length / 3;
                }
            }
            provenance.Append("\n  ]\n}\n");
            File.WriteAllText(provenancePath, provenance.ToString(), new UTF8Encoding(false));
            WriteSystemReferenceStatus(snapshots.Count > 0 ? "exported" : "unavailable",
                snapshots.Count > 0 ? string.Empty : "No GlobalMeshAnchor contained triangles",
                snapshots.Count, vertexCount, triangleCount);
        }

        private void WriteSystemReferenceStatus(string status, string issue,
            int meshes, int vertices, int triangles)
        {
            bool external = string.Equals(status, "external_required", StringComparison.Ordinal);
            string json = "{\n" +
                          "  \"schema\": \"scancover.system_room_mesh_reference.v1\",\n" +
                          "  \"status\": \"" + Json(status) + "\",\n" +
                          "  \"issue\": \"" + Json(issue) + "\",\n" +
                          "  \"source\": \"" + Json(external
                              ? "separate Meta Scene Mesh scene / OVRTriangleMesh"
                              : "MRUK.Rooms[].GlobalMeshAnchor.GlobalMesh") + "\",\n" +
                          "  \"coordinateSpace\": \"" + Json(external
                              ? "external Unity world, metres; offline registration required"
                              : "Unity world, metres, baked at session stop") + "\",\n" +
                          "  \"meshCount\": " + meshes + ",\n" +
                          "  \"vertexCount\": " + vertices + ",\n" +
                          "  \"triangleCount\": " + triangles + "\n}\n";
            File.WriteAllText(Path.Combine(_sessionDirectory, "system_reference", "status.json"),
                json, new UTF8Encoding(false));
        }

        private void TryFinalize()
        {
            if (_finalized || _finalizeTask != null || !_endRequested ||
                !_depthStreamsComplete || OutstandingCount > 0)
                return;
            _finalizeClean = _depthPairDrops == 0 && _depthPairReadbackErrors == 0 &&
                             _depthPairWriteErrors == 0 && _fusionDropped == 0 &&
                             _fusionReadbackErrors == 0 && _fusionWriteErrors == 0;
            Interlocked.Increment(ref _artifactRequests);
            _finalizeTask = Task.Run(() =>
            {
                try
                {
                    WriteState(_finalizeClean ? "complete" : "incomplete_with_capture_loss", 0);
                    WriteChecksumsAndCompleteMarker();
                }
                finally { Interlocked.Decrement(ref _artifactRequests); }
            });
        }

        private void WriteChecksumsAndCompleteMarker()
        {
            string manifestPath = Path.Combine(_sessionDirectory, "checksums.sha256");
            string manifestTempPath = manifestPath + ".tmp";
            string completePath = Path.Combine(_sessionDirectory, "capture_complete.json");
            string incompletePath = Path.Combine(_sessionDirectory, "capture_incomplete.json");
            string completeTempPath = completePath + ".tmp";
            string incompleteTempPath = incompletePath + ".tmp";
            string[] files = Directory.GetFiles(_sessionDirectory, "*", SearchOption.AllDirectories);
            Array.Sort(files, StringComparer.Ordinal);
            var hashes = new StringBuilder(files.Length * 96);
            using (SHA256 sha = SHA256.Create())
            {
                foreach (string file in files)
                {
                    if (string.Equals(file, manifestPath, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(file, manifestTempPath, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(file, completePath, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(file, completeTempPath, StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (string.Equals(file, incompletePath, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(file, incompleteTempPath, StringComparison.OrdinalIgnoreCase))
                        continue;
                    string fullPath = Path.GetFullPath(file);
                    string hashText;
                    if (!_immutableFileHashes.TryGetValue(fullPath, out hashText))
                    {
                        using FileStream stream = File.OpenRead(file);
                        hashText = Hex(sha.ComputeHash(stream));
                    }
                    hashes.Append(hashText).Append("  ")
                        .Append(Path.GetRelativePath(_sessionDirectory, file).Replace('\\', '/'))
                        .AppendLine();
                }
            }
            File.WriteAllText(manifestTempPath, hashes.ToString(), new UTF8Encoding(false));
            if (File.Exists(manifestPath)) File.Delete(manifestPath);
            File.Move(manifestTempPath, manifestPath);
            string manifestHash;
            using (SHA256 sha = SHA256.Create())
            using (FileStream stream = File.OpenRead(manifestPath))
                manifestHash = Hex(sha.ComputeHash(stream));
            bool clean = _depthPairDrops == 0 && _depthPairReadbackErrors == 0 &&
                         _depthPairWriteErrors == 0 && _fusionDropped == 0 &&
                         _fusionReadbackErrors == 0 && _fusionWriteErrors == 0;
            string complete = "{\n" +
                              "  \"schema\": \"" + Schema + "\",\n" +
                              "  \"state\": \"" + (clean ? "complete" : "incomplete_with_capture_loss") + "\",\n" +
                              "  \"startedUtc\": \"" + Json(_startedUtc) + "\",\n" +
                              "  \"stoppedUtc\": \"" + Json(_stoppedUtc) + "\",\n" +
                              "  \"depthRecordedEyeIndex\": " + RecordedEye + ",\n" +
                              "  \"fusionRecords\": " + _fusionSequence + ",\n" +
                              "  \"fusionCaptureDrops\": " + _fusionDropped + ",\n" +
                              "  \"fusionReadbackErrors\": " + _fusionReadbackErrors + ",\n" +
                              "  \"fusionWriteErrors\": " + _fusionWriteErrors + ",\n" +
                              "  \"depthPairDrops\": " + _depthPairDrops + ",\n" +
                              "  \"depthPairReadbackErrors\": " + _depthPairReadbackErrors + ",\n" +
                              "  \"depthPairWriteErrors\": " + _depthPairWriteErrors + ",\n" +
                              "  \"checksumsManifestSha256\": \"" + manifestHash + "\"\n" +
                              "}\n";
            string markerPath = clean ? completePath : incompletePath;
            string markerTempPath = markerPath + ".tmp";
            File.WriteAllText(markerTempPath, complete, new UTF8Encoding(false));
            if (File.Exists(markerPath)) File.Delete(markerPath);
            File.Move(markerTempPath, markerPath);
        }

        private void WriteState(string state, int? outstandingOverride = null)
        {
            if (string.IsNullOrEmpty(_sessionDirectory)) return;
            string json = "{\n" +
                          "  \"schema\": \"" + Schema + "\",\n" +
                          "  \"state\": \"" + Json(state) + "\",\n" +
                          "  \"depthStreamsComplete\": " + (_depthStreamsComplete ? "true" : "false") + ",\n" +
                          "  \"fusionRecords\": " + _fusionSequence + ",\n" +
                          "  \"fusionCaptureDrops\": " + _fusionDropped + ",\n" +
                          "  \"fusionReadbackErrors\": " + _fusionReadbackErrors + ",\n" +
                          "  \"fusionWriteErrors\": " + _fusionWriteErrors + ",\n" +
                          "  \"depthPairDrops\": " + _depthPairDrops + ",\n" +
                          "  \"depthPairReadbackErrors\": " + _depthPairReadbackErrors + ",\n" +
                          "  \"depthPairWriteErrors\": " + _depthPairWriteErrors + ",\n" +
                          "  \"outstanding\": " + (outstandingOverride ?? OutstandingCount) + "\n}\n";
            lock (_fileLock)
                File.WriteAllText(Path.Combine(_sessionDirectory, "session_status.json"),
                    json, new UTF8Encoding(false));
        }

        private void WriteArtifactStatus(string name, string status, string issue, string relativePath)
        {
            try
            {
                string directory = Path.Combine(_sessionDirectory, "artifacts");
                Directory.CreateDirectory(directory);
                string json = "{\n" +
                              "  \"name\": \"" + Json(name) + "\",\n" +
                              "  \"status\": \"" + Json(status) + "\",\n" +
                              "  \"issue\": \"" + Json(issue) + "\",\n" +
                              "  \"relativePath\": \"" + Json(relativePath) + "\"\n}\n";
                File.WriteAllText(Path.Combine(directory, name + "_status.json"),
                    json, new UTF8Encoding(false));
            }
            catch { }
        }

        private static string BuildSessionSchemaJson()
        {
            return "{\n" +
                   "  \"schema\": \"" + Schema + "\",\n" +
                   "  \"purpose\": \"deterministic offline replay and attribution of one production scan\",\n" +
                   "  \"depth_pairs\": \"platform pre-QRS and post-QRS depth for every preprocessed source frame\",\n" +
                   "  \"fusion_inputs\": \"exact accepted depth, admission, RGB and camera inputs plus rejected decision events\",\n" +
                   "  \"artifacts\": \"GunGel candidate audit, paper audit and online geometry ledgers\",\n" +
                   "  \"system_reference\": \"handoff status for a separately exported Quest system room mesh; comparison only\",\n" +
                   "  \"completeRule\": \"capture_complete.json exists only after production depth, fusion and online artifacts drain; external system mesh is not a seal dependency\"\n" +
                   "}\n";
        }

        private static string BuildReplayContractJson()
        {
            return "{\n" +
                   "  \"schema\": \"scancover.deterministic_replay_contract.v1\",\n" +
                   "  \"initialState\": \"empty TSDF, color, confidence, admission and candidate volumes; recorder refuses a mid-scan start\",\n" +
                   "  \"routeA_preprocess\": {\"orderBy\":\"pairIndex\",\"input\":\"depth_pairs/*platform_pre_qrs.f32\",\"reference\":\"depth_pairs/*post_qrs.f32\"},\n" +
                   "  \"routeB_productionFusion\": {\"orderBy\":\"sequence\",\"acceptedRows\":\"apply exact recorded textures, matrices, exclusions, GunGel admission and RGB camera payload\",\"rejectedRows\":\"advance decision history without integration\",\"timeSource\":\"scaledTime and unscaledTime columns\"},\n" +
                   "  \"routeC_systemReference\": {\"input\":\"separate ScanCover_MetaSceneMeshAudit_*/meta_scene_mesh_aligned_all.obj\",\"alignment\":\"offline registration required across app launches\",\"role\":\"comparison only; never production truth\"},\n" +
                   "  \"configuration\": \"production_config.json is authoritative; do not substitute inspector defaults\",\n" +
                   "  \"coordinates\": \"coordinate_contract.json plus coordinate_contract_stop.json\",\n" +
                   "  \"integrity\": \"sort manifests by numeric keys and require capture_complete.json plus checksums.sha256\",\n" +
                   "  \"earliestDepthBoundary\": \"Meta platform environment depth before QRS preprocessing, not inaccessible sensor raw\"\n" +
                   "}\n";
        }

        private static string BuildProductionConfigJson()
        {
            var sb = new StringBuilder(32768);
            sb.Append("{\n  \"schema\": \"scancover.production_config_snapshot.v1\",\n")
                .Append("  \"capturedUtc\": \"").Append(DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)).Append("\",\n")
                .Append("  \"application\": {\"identifier\":\"").Append(Json(Application.identifier))
                .Append("\",\"version\":\"").Append(Json(Application.version))
                .Append("\",\"buildGuid\":\"").Append(Json(Application.buildGUID))
                .Append("\",\"unityVersion\":\"").Append(Json(Application.unityVersion))
                .Append("\",\"platform\":\"").Append(Json(Application.platform.ToString())).Append("\"},\n")
                .Append("  \"runtime\": {\"deviceModel\":\"").Append(Json(SystemInfo.deviceModel))
                .Append("\",\"operatingSystem\":\"").Append(Json(SystemInfo.operatingSystem))
                .Append("\",\"graphicsDeviceName\":\"").Append(Json(SystemInfo.graphicsDeviceName))
                .Append("\",\"graphicsDeviceVendor\":\"").Append(Json(SystemInfo.graphicsDeviceVendor))
                .Append("\",\"graphicsDeviceVersion\":\"").Append(Json(SystemInfo.graphicsDeviceVersion))
                .Append("\",\"graphicsDeviceType\":\"").Append(Json(SystemInfo.graphicsDeviceType.ToString()))
                .Append("\"},\n")
                .Append("  \"components\": {\n");
            Component[] components =
            {
                DepthCapture.Instance,
                VolumeIntegrator.Instance,
                MeshExtractor.Instance,
                FindAnyObjectByType<StandaloneRoomScanner>(FindObjectsInactive.Include)
            };
            bool first = true;
            foreach (Component component in components)
            {
                if (component == null) continue;
                if (!first) sb.Append(",\n");
                first = false;
                sb.Append("    \"").Append(Json(component.GetType().FullName)).Append("\": ");
                AppendSerializedFields(sb, component);
            }
            sb.Append("\n  }\n}\n");
            return sb.ToString();
        }

        private static void AppendSerializedFields(StringBuilder sb, Component component)
        {
            sb.Append('{');
            bool first = true;
            for (Type type = component.GetType(); type != null && type != typeof(MonoBehaviour); type = type.BaseType)
            {
                FieldInfo[] fields = type.GetFields(BindingFlags.Instance | BindingFlags.Public |
                                                    BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
                Array.Sort(fields, (a, b) => string.CompareOrdinal(a.Name, b.Name));
                foreach (FieldInfo field in fields)
                {
                    bool serializable = field.IsPublic || field.GetCustomAttribute<SerializeField>() != null;
                    if (!serializable || field.IsStatic || field.IsNotSerialized) continue;
                    object value;
                    try { value = field.GetValue(component); }
                    catch { continue; }
                    if (!first) sb.Append(',');
                    first = false;
                    sb.Append("\"").Append(Json(field.Name)).Append("\":");
                    AppendConfigValue(sb, value, 0);
                }
            }
            sb.Append('}');
        }

        private static void AppendConfigValue(StringBuilder sb, object value, int depth)
        {
            if (value == null) { sb.Append("null"); return; }
            if (depth > 3) { sb.Append("\"").Append(Json(value.ToString())).Append("\""); return; }
            switch (value)
            {
                case bool b: sb.Append(b ? "true" : "false"); return;
                case string s: sb.Append('"').Append(Json(s)).Append('"'); return;
                case Enum e: sb.Append('"').Append(Json(e.ToString())).Append('"'); return;
                case UnityEngine.Object o:
                    sb.Append("{\"name\":\"").Append(Json(o.name)).Append("\",\"type\":\"")
                        .Append(Json(o.GetType().FullName)).Append("\"}"); return;
                case Vector2 v2: sb.Append('[').Append(F(v2.x)).Append(',').Append(F(v2.y)).Append(']'); return;
                case Vector3 v3: sb.Append('[').Append(F(v3.x)).Append(',').Append(F(v3.y)).Append(',').Append(F(v3.z)).Append(']'); return;
                case Vector4 v4: sb.Append('[').Append(F(v4.x)).Append(',').Append(F(v4.y)).Append(',').Append(F(v4.z)).Append(',').Append(F(v4.w)).Append(']'); return;
                case Color c: sb.Append('[').Append(F(c.r)).Append(',').Append(F(c.g)).Append(',').Append(F(c.b)).Append(',').Append(F(c.a)).Append(']'); return;
                case Rect r: sb.Append('[').Append(F(r.x)).Append(',').Append(F(r.y)).Append(',').Append(F(r.width)).Append(',').Append(F(r.height)).Append(']'); return;
            }
            if (value is IFormattable formattable && (value.GetType().IsPrimitive || value is decimal))
            {
                sb.Append(formattable.ToString(null, CultureInfo.InvariantCulture));
                return;
            }
            if (value is IEnumerable enumerable)
            {
                sb.Append('[');
                int count = 0;
                foreach (object item in enumerable)
                {
                    if (count > 0) sb.Append(',');
                    if (count++ >= 128) { sb.Append("\"...truncated...\""); break; }
                    AppendConfigValue(sb, item, depth + 1);
                }
                sb.Append(']');
                return;
            }
            sb.Append('"').Append(Json(value.ToString())).Append('"');
        }

        private static string BuildCoordinateContractJson(string phase)
        {
            XROrigin origin = FindAnyObjectByType<XROrigin>(FindObjectsInactive.Include);
            Transform tracking = origin != null && origin.CameraFloorOffsetObject != null
                ? origin.CameraFloorOffsetObject.transform
                : origin != null ? origin.transform : null;
            Camera camera = Camera.main;
            VolumeIntegrator volume = VolumeIntegrator.Instance;
            MeshExtractor extractor = MeshExtractor.Instance;
            var sb = new StringBuilder(4096);
            sb.Append("{\n  \"schema\": \"scancover.coordinate_contract.v1\",\n")
                .Append("  \"phase\": \"").Append(Json(phase)).Append("\",\n")
                .Append("  \"units\": \"metres\",\n")
                .Append("  \"handedness\": \"Unity left-handed world; projection payload retained exactly as supplied by XR\",\n")
                .Append("  \"recordedEyeIndex\": ").Append(RecordedEye).Append(",\n")
                .Append("  \"depthBinaryOrder\": \"x-fastest then y, one right-eye slice\",\n")
                .Append("  \"trackingToWorld\": ");
            AppendMatrix(sb, tracking != null ? tracking.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"xrOriginLocalToWorld\": ");
            AppendMatrix(sb, origin != null ? origin.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"mainCameraLocalToWorld\": ");
            AppendMatrix(sb, camera != null ? camera.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"volumeIntegratorLocalToWorld\": ");
            AppendMatrix(sb, volume != null ? volume.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append(",\n  \"meshExtractorLocalToWorld\": ");
            AppendMatrix(sb, extractor != null ? extractor.transform.localToWorldMatrix : Matrix4x4.identity);
            sb.Append("\n}\n");
            return sb.ToString();
        }

        private static string BuildFusionMetaJson(FusionRecord r, string depth, string normal,
            string dilated, string edge, string temporal, string camera, string observations,
            string correspondences, string status)
        {
            var sb = new StringBuilder(4096);
            sb.Append("{\n  \"schema\": \"scancover.fusion_input.v1\",\n")
                .Append("  \"sequence\": ").Append(r.Sequence).Append(",\n")
                .Append("  \"attemptIndex\": ").Append(r.AttemptIndex).Append(",\n")
                .Append("  \"sourceFrame\": ").Append(r.SourceFrame).Append(",\n")
                .Append("  \"unityFrame\": ").Append(r.UnityFrame).Append(",\n")
                .Append("  \"scaledTime\": ").Append(D(r.ScaledTime)).Append(",\n")
                .Append("  \"unscaledTime\": ").Append(D(r.UnscaledTime)).Append(",\n")
                .Append("  \"accepted\": ").Append(r.Accepted ? "true" : "false").Append(",\n")
                .Append("  \"decision\": \"").Append(Json(r.Decision)).Append("\",\n")
                .Append("  \"guardedGunGel\": ").Append(r.Guarded ? "true" : "false").Append(",\n")
                .Append("  \"gunGelAdmissionActive\": ").Append(r.GunGelAdmissionActive ? "true" : "false").Append(",\n")
                .Append("  \"gunGelFrame\": ").Append(r.GunGelFrame).Append(",\n")
                .Append("  \"translationMm\": ").Append(F(r.TranslationMm)).Append(",\n")
                .Append("  \"rotationDeg\": ").Append(F(r.RotationDeg)).Append(",\n")
                .Append("  \"angularDegPerSec\": ").Append(F(r.AngularSpeed)).Append(",\n")
                .Append("  \"linearMps\": ").Append(F(r.LinearSpeed)).Append(",\n")
                .Append("  \"motionQuality\": ").Append(F(r.MotionQuality)).Append(",\n")
                .Append("  \"recordedEyeIndex\": ").Append(RecordedEye).Append(",\n")
                .Append("  \"status\": \"").Append(Json(status)).Append("\",\n")
                .Append("  \"textures\": {");
            AppendTextureMeta(sb, "depth", depth, r.Textures.depthDescriptor, true);
            AppendTextureMeta(sb, "normal", normal, r.Textures.normalDescriptor, true);
            AppendTextureMeta(sb, "dilated", dilated, r.Textures.dilatedDescriptor, true);
            AppendTextureMeta(sb, "edgeReason", edge, r.Textures.edgeDescriptor, true);
            AppendTextureMeta(sb, "temporalReason", temporal, r.Textures.temporalDescriptor, true);
            AppendTextureMeta(sb, "cameraRgb", camera, r.Textures.cameraDescriptor, false);
            sb.Append("\n  },\n  \"cameraColor\": {\"available\":")
                .Append(r.CameraAvailable ? "true" : "false")
                .Append(",\"position\":");
            AppendVector3(sb, r.CameraPosition);
            sb.Append(",\"rotationQuaternion\":");
            AppendQuaternion(sb, r.CameraRotation);
            sb.Append(",\"focalLength\":"); AppendVector2(sb, r.CameraFocalLength);
            sb.Append(",\"principalPoint\":"); AppendVector2(sb, r.CameraPrincipalPoint);
            sb.Append(",\"sensorResolution\":"); AppendVector2(sb, r.CameraSensorResolution);
            sb.Append(",\"currentResolution\":"); AppendVector2(sb, r.CameraCurrentResolution);
            sb.Append("},\n  \"fusionCorrection\": ");
            AppendMatrix(sb, r.FusionCorrection);
            sb.Append(",\n  \"exclusions\": {\"count\":").Append(r.ExclusionCount)
                .Append(",\"positions\":[");
            for (int i = 0; i < r.ExclusionCount; i++)
            {
                if (i > 0) sb.Append(',');
                Vector4 position = r.ExclusionPositions[i];
                sb.Append('[').Append(F(position.x)).Append(',').Append(F(position.y))
                    .Append(',').Append(F(position.z)).Append(',').Append(F(position.w)).Append(']');
            }
            sb.Append("]},\n  \"gunGelAdmission\": {\"active\":")
                .Append(r.GunGelAdmissionActive ? "true" : "false")
                .Append(",\"observationFile\":\"")
                .Append(Json(observations)).Append("\",\"correspondenceFile\":\"")
                .Append(Json(correspondences)).Append("\",\"observationCount\":")
                .Append(r.Textures.gunGelObservationCount).Append(",\"observationStrideBytes\":")
                .Append(r.Textures.gunGelObservationStride)
                .Append(",\"observationLayout\":\"float4 positionSigma; float4 normalQuality; float4 rawPositionDelta; uint4 sourceReason(packedPixel,edgeReason,temporalReason,platformFrame)\",\"correspondenceCount\":")
                .Append(r.Textures.gunGelCorrespondenceCount).Append(",\"correspondenceStrideBytes\":")
                .Append(r.Textures.gunGelCorrespondenceStride).Append(",\"gridX\":")
                .Append(r.GunGelGridX).Append(",\"gridY\":").Append(r.GunGelGridY)
                .Append(",\"pixelStride\":").Append(r.GunGelPixelStride).Append("},\n  \"projection\": ");
            AppendMatrixArray(sb, r.Projection);
            sb.Append(",\n  \"view\": "); AppendMatrixArray(sb, r.View);
            sb.Append(",\n  \"projectionInverse\": "); AppendMatrixArray(sb, r.ProjectionInverse);
            sb.Append(",\n  \"viewInverse\": "); AppendMatrixArray(sb, r.ViewInverse);
            sb.Append("\n}\n");
            return sb.ToString();
        }

        private static void AppendTextureMeta(StringBuilder sb, string role, string file,
            TextureDescriptor descriptor, bool comma)
        {
            sb.Append("\n    \"").Append(role).Append("\": {\"file\":\"").Append(Json(file))
                .Append("\",\"width\":").Append(descriptor.width)
                .Append(",\"height\":").Append(descriptor.height)
                .Append(",\"graphicsFormat\":\"").Append(Json(descriptor.format))
                .Append("\",\"sourceDimension\":\"").Append(Json(descriptor.dimension))
                .Append("\",\"sourceSlice\":").Append(descriptor.slice).Append('}');
            if (comma) sb.Append(',');
        }

        private static void AppendMatrixArray(StringBuilder sb, Matrix4x4[] matrices)
        {
            sb.Append('[');
            for (int i = 0; i < 2; i++)
            {
                if (i > 0) sb.Append(',');
                AppendMatrix(sb, matrices != null && matrices.Length > i ? matrices[i] : Matrix4x4.identity);
            }
            sb.Append(']');
        }

        private static void AppendVector2(StringBuilder sb, Vector2 value) =>
            sb.Append('[').Append(F(value.x)).Append(',').Append(F(value.y)).Append(']');

        private static void AppendVector3(StringBuilder sb, Vector3 value) =>
            sb.Append('[').Append(F(value.x)).Append(',').Append(F(value.y)).Append(',')
                .Append(F(value.z)).Append(']');

        private static void AppendQuaternion(StringBuilder sb, Quaternion value) =>
            sb.Append('[').Append(F(value.x)).Append(',').Append(F(value.y)).Append(',')
                .Append(F(value.z)).Append(',').Append(F(value.w)).Append(']');

        private static void AppendMatrix(StringBuilder sb, Matrix4x4 matrix)
        {
            sb.Append('[');
            for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                if (row != 0 || col != 0) sb.Append(',');
                sb.Append(F(matrix[row, col]));
            }
            sb.Append(']');
        }

        private static Matrix4x4[] CloneMatrices(Matrix4x4[] source)
        {
            return source != null && source.Length >= 2
                ? new[] { source[0], source[1] }
                : new[] { Matrix4x4.identity, Matrix4x4.identity };
        }

        private static Vector4[] CloneVectors(Vector4[] source, int count)
        {
            int length = Mathf.Clamp(count, 0, source != null ? source.Length : 0);
            var result = new Vector4[length];
            if (length > 0) Array.Copy(source, result, length);
            return result;
        }

        private static void CopyDirectory(string source, string destination)
        {
            Directory.CreateDirectory(destination);
            foreach (string directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(Path.Combine(destination, Path.GetRelativePath(source, directory)));
            foreach (string file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            {
                string target = Path.Combine(destination, Path.GetRelativePath(source, file));
                Directory.CreateDirectory(Path.GetDirectoryName(target) ?? destination);
                File.Copy(file, target, true);
            }
        }

        private static string Hex(byte[] bytes) => BitConverter.ToString(bytes).Replace("-", string.Empty).ToLowerInvariant();
        private static string F(float value) => float.IsNaN(value) || float.IsInfinity(value)
            ? "null" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string D(double value) => double.IsNaN(value) || double.IsInfinity(value)
            ? "null" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string Csv(string value) => (value ?? string.Empty).Replace(',', ';').Replace('\n', ' ').Replace('\r', ' ');
        private static string Json(string value) => (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");

        private sealed class SystemMeshSnapshot
        {
            internal readonly int RoomIndex;
            internal readonly string AnchorName;
            internal readonly Matrix4x4 LocalToWorld;
            internal readonly Vector3[] Vertices;
            internal readonly int[] Triangles;

            internal SystemMeshSnapshot(int roomIndex, string anchorName,
                Matrix4x4 localToWorld, Vector3[] vertices, int[] triangles)
            {
                RoomIndex = roomIndex;
                AnchorName = anchorName ?? string.Empty;
                LocalToWorld = localToWorld;
                Vertices = vertices ?? Array.Empty<Vector3>();
                Triangles = triangles ?? Array.Empty<int>();
            }
        }

        private readonly struct TextureDescriptor
        {
            internal readonly int width;
            internal readonly int height;
            internal readonly string format;
            internal readonly string dimension;
            internal readonly int slice;
            internal TextureDescriptor(int width, int height, string format, string dimension, int slice)
            {
                this.width = width; this.height = height; this.format = format;
                this.dimension = dimension; this.slice = slice;
            }
        }

        private sealed class PendingTextureSet
        {
            internal byte[] depth, normal, dilated, edge, temporal, camera,
                gunGelObservations, gunGelCorrespondences;
            internal bool depthDone, normalDone, dilatedDone, edgeDone, temporalDone, cameraDone,
                gunGelObservationsDone, gunGelCorrespondencesDone;
            internal bool HasError;
            internal TextureDescriptor depthDescriptor, normalDescriptor, dilatedDescriptor,
                edgeDescriptor, temporalDescriptor, cameraDescriptor;
            internal int gunGelObservationCount, gunGelObservationStride,
                gunGelCorrespondenceCount, gunGelCorrespondenceStride;
            internal bool AllDone => depthDone && normalDone && dilatedDone && edgeDone &&
                                     temporalDone && cameraDone &&
                                     gunGelObservationsDone && gunGelCorrespondencesDone;

            internal void SetDescriptor(string role, TextureDescriptor value)
            {
                switch (role)
                {
                    case "depth": depthDescriptor = value; break;
                    case "normal": normalDescriptor = value; break;
                    case "dilated": dilatedDescriptor = value; break;
                    case "edge": edgeDescriptor = value; break;
                    case "temporal": temporalDescriptor = value; break;
                    case "camera": cameraDescriptor = value; break;
                }
            }
            internal void SetBufferDescriptor(string role, int count, int stride)
            {
                if (role == "gungel_observations")
                {
                    gunGelObservationCount = count; gunGelObservationStride = stride;
                }
                else
                {
                    gunGelCorrespondenceCount = count; gunGelCorrespondenceStride = stride;
                }
            }
            internal void SetBytes(string role, byte[] value)
            {
                switch (role)
                {
                    case "depth": depth = value; depthDone = true; break;
                    case "normal": normal = value; normalDone = true; break;
                    case "dilated": dilated = value; dilatedDone = true; break;
                    case "edge": edge = value; edgeDone = true; break;
                    case "temporal": temporal = value; temporalDone = true; break;
                    case "camera": camera = value; cameraDone = true; break;
                    case "gungel_observations": gunGelObservations = value; gunGelObservationsDone = true; break;
                    case "gungel_correspondences": gunGelCorrespondences = value; gunGelCorrespondencesDone = true; break;
                }
            }
            internal void MarkMissing(string role) { MarkDone(role); }
            internal void MarkError(string role) { HasError = true; MarkDone(role); }
            private void MarkDone(string role)
            {
                switch (role)
                {
                    case "depth": depthDone = true; break;
                    case "normal": normalDone = true; break;
                    case "dilated": dilatedDone = true; break;
                    case "edge": edgeDone = true; break;
                    case "temporal": temporalDone = true; break;
                    case "camera": cameraDone = true; break;
                    case "gungel_observations": gunGelObservationsDone = true; break;
                    case "gungel_correspondences": gunGelCorrespondencesDone = true; break;
                }
            }
        }

        private sealed class FusionRecord
        {
            internal readonly int Sequence, AttemptIndex, SourceFrame, GunGelFrame;
            internal readonly int UnityFrame;
            internal readonly double ScaledTime, UnscaledTime;
            internal readonly bool Accepted, Guarded, GunGelAdmissionActive, CameraAvailable;
            internal readonly string Decision;
            internal readonly float TranslationMm, RotationDeg, AngularSpeed, LinearSpeed, MotionQuality;
            internal readonly Matrix4x4[] Projection, View, ProjectionInverse, ViewInverse;
            internal readonly PendingTextureSet Textures;
            internal readonly int GunGelGridX, GunGelGridY, GunGelPixelStride;
            internal readonly Vector3 CameraPosition;
            internal readonly Quaternion CameraRotation;
            internal readonly Vector2 CameraFocalLength, CameraPrincipalPoint,
                CameraSensorResolution, CameraCurrentResolution;
            internal readonly Matrix4x4 FusionCorrection;
            internal readonly Vector4[] ExclusionPositions;
            internal readonly int ExclusionCount;
            internal bool CompletionClaimed;
            internal FusionRecord(int sequence, int attemptIndex, int sourceFrame, bool accepted,
                string decision, bool guarded, int gunGelFrame, float translationMm,
                float rotationDeg, float angularSpeed, float linearSpeed, float motionQuality,
                Matrix4x4[] projection, Matrix4x4[] view, Matrix4x4[] projectionInverse,
                Matrix4x4[] viewInverse, PendingTextureSet textures,
                bool gunGelAdmissionActive, bool cameraAvailable,
                Vector3 cameraPosition, Quaternion cameraRotation,
                Vector2 cameraFocalLength, Vector2 cameraPrincipalPoint,
                Vector2 cameraSensorResolution, Vector2 cameraCurrentResolution,
                Matrix4x4 fusionCorrection, Vector4[] exclusionPositions,
                int exclusionCount,
                int gunGelGridX = 0, int gunGelGridY = 0, int gunGelPixelStride = 0)
            {
                Sequence = sequence; AttemptIndex = attemptIndex; SourceFrame = sourceFrame;
                UnityFrame = Time.frameCount; ScaledTime = Time.timeAsDouble;
                UnscaledTime = Time.unscaledTimeAsDouble;
                Accepted = accepted; Decision = decision ?? string.Empty; Guarded = guarded;
                GunGelFrame = gunGelFrame; TranslationMm = translationMm; RotationDeg = rotationDeg;
                AngularSpeed = angularSpeed; LinearSpeed = linearSpeed; MotionQuality = motionQuality;
                Projection = projection; View = view; ProjectionInverse = projectionInverse;
                ViewInverse = viewInverse; Textures = textures;
                GunGelAdmissionActive = gunGelAdmissionActive;
                CameraAvailable = cameraAvailable; CameraPosition = cameraPosition;
                CameraRotation = cameraRotation; CameraFocalLength = cameraFocalLength;
                CameraPrincipalPoint = cameraPrincipalPoint;
                CameraSensorResolution = cameraSensorResolution;
                CameraCurrentResolution = cameraCurrentResolution;
                FusionCorrection = fusionCorrection;
                ExclusionPositions = exclusionPositions ?? Array.Empty<Vector4>();
                ExclusionCount = Mathf.Clamp(exclusionCount, 0, ExclusionPositions.Length);
                GunGelGridX = gunGelGridX; GunGelGridY = gunGelGridY;
                GunGelPixelStride = gunGelPixelStride;
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    internal static class CaptureAuditIO
    {
        internal static Task<byte[]> ReadAsync(ComputeBuffer buffer)
        {
            var completion = new TaskCompletionSource<byte[]>();
            try
            {
                AsyncGPUReadback.Request(buffer, request =>
                {
                    try
                    {
                        if (request.hasError) completion.TrySetException(new IOException("audit buffer readback failed"));
                        else completion.TrySetResult(request.GetData<byte>().ToArray());
                    }
                    catch (Exception e) { completion.TrySetException(e); }
                });
            }
            catch (Exception e) { completion.TrySetException(e); }
            return completion.Task;
        }

        internal static Task<byte[]> ReadAsync(GraphicsBuffer buffer, int bytes)
        {
            if (buffer == null || bytes == 0) return Task.FromResult(Array.Empty<byte>());
            var completion = new TaskCompletionSource<byte[]>();
            try
            {
                AsyncGPUReadback.Request(buffer, bytes, 0, request =>
                {
                    try
                    {
                        if (request.hasError) completion.TrySetException(new IOException("Front buffer readback failed"));
                        else completion.TrySetResult(request.GetData<byte>().ToArray());
                    }
                    catch (Exception e) { completion.TrySetException(e); }
                });
            }
            catch (Exception e) { completion.TrySetException(e); }
            return completion.Task;
        }
    }

    internal sealed class CaptureFront
    {
        internal int Block, Generation, Vertices, Indices, SubmittedIndices, SubmittedFrame;
        internal uint Target, Candidate, Published;
        internal bool Visible, Pending;
        internal GPUChunkMeshSnapshot Source;
        internal Task<byte[]> VertexRead, IndexRead, ArgsRead;
        internal Task<byte[]> CrossingRead, CrossingSpanRead, RawSurfaceNetsRead;
        internal void RequestReads()
        {
            VertexRead = CaptureAuditIO.ReadAsync(Source?.VertexBuffer, checked(Vertices * 32));
            IndexRead = CaptureAuditIO.ReadAsync(Source?.IndexBuffer, checked(Indices * 4));
            ArgsRead = CaptureAuditIO.ReadAsync(Source?.DrawIndirectArgs, Source != null ? 20 : 0);
            CrossingRead = CaptureAuditIO.ReadAsync(
                Source?.LayerCrossingEvidenceBuffer, checked(Vertices * 16));
            CrossingSpanRead = CaptureAuditIO.ReadAsync(
                Source?.LayerCrossingSpanBuffer, checked(Vertices * 12));
            RawSurfaceNetsRead = CaptureAuditIO.ReadAsync(
                Source?.LayerRawSurfaceNetsBuffer, checked(Vertices * 12));
        }
    }

    public partial class MeshExtractor
    {
        private bool CaptureUsesBlocks => IsInfiniTamBaselineActive && _infiniTamBlocks != null &&
            !_infiniTamBlocks.Failed && _infiniTamBlocks.InitialBuildComplete;
        internal string CaptureMeshRoute => !IsInfiniTamBaselineActive ? "legacy_paper" :
            CaptureUsesBlocks ? "infinitam_block_front" : "infinitam_global_front";
        private int CaptureTriangleCount => CaptureUsesBlocks
            ? (int)Math.Min(int.MaxValue, _infiniTamBlocks.CommittedIndexCount / 3)
            : (_infiniTamFront?.KnownDrawVertexCount ?? 0) / 3;
        private int CaptureBuiltBlocks => CaptureUsesBlocks ? _infiniTamBlocks.CaptureBuiltBlocks :
            (_infiniTamFront != null ? 1 : 0);
        private bool CaptureVisible => CaptureUsesBlocks ? _infiniTamBlocks.CaptureVisible :
            _gpuRenderer != null && _gpuRenderer.RenderVisible && _infiniTamFront != null;
        internal int CaptureSubmittedIndices => CaptureUsesBlocks
            ? _infiniTamBlocks.CaptureSubmittedIndices : (_gpuRenderer?.LastSubmittedVertexCount ?? 0);
        internal int CaptureSubmittedFrame => CaptureUsesBlocks
            ? _infiniTamBlocks.CaptureSubmittedFrame : (_gpuRenderer?.LastSubmittedFrame ?? -1);

        internal async Task ExportInfiniTamFrontAsync(string directory)
        {
            if (!IsInfiniTamBaselineActive) throw new InvalidOperationException("InfiniTAM is not the production source");
            int unityFrame = Time.frameCount;
            int integration = _volume.IntegrationCount;
            uint epoch = _volume.DirtyEpoch;
            string route = CaptureMeshRoute;
            List<CaptureFront> fronts;
            if (CaptureUsesBlocks) fronts = _infiniTamBlocks.CaptureFronts();
            else
            {
                if (_infiniTamFront == null) throw new InvalidOperationException("no published global Front");
                fronts = new List<CaptureFront> { new CaptureFront {
                    Block = -1, Source = _infiniTamFront,
                    Vertices = _infiniTamFront.KnownVertexCount, Indices = _infiniTamFront.KnownDrawVertexCount,
                    Visible = CaptureVisible, SubmittedIndices = CaptureSubmittedIndices,
                    SubmittedFrame = CaptureSubmittedFrame } };
            }
            // Queue ALL reads now. Never await one block and accidentally capture
            // the other blocks after a subsequent Front replacement.
            foreach (CaptureFront front in fronts) front.RequestReads();
            foreach (CaptureFront front in fronts)
                await Task.WhenAll(front.VertexRead, front.IndexRead,
                    front.ArgsRead, front.CrossingRead, front.CrossingSpanRead,
                    front.RawSurfaceNetsRead);
            if (_volume.IntegrationCount != integration || _volume.DirtyEpoch != epoch)
                throw new InvalidOperationException("fusion changed during Front snapshot; stop scanning before sealing");
            string metadata = "{\"schema\":\"scancover.actual_infinitam_front.v3\",\"route\":\"" + route +
                "\",\"unityFrame\":" + unityFrame + ",\"integrationCount\":" + integration +
                ",\"dirtyEpoch\":" + epoch + ",\"vertexStrideBytes\":32," +
                "\"voxelCount\":[" + _volume.VoxelCount.x + "," +
                _volume.VoxelCount.y + "," + _volume.VoxelCount.z + "]," +
                "\"voxelSizeMetres\":" +
                _volume.VoxelSize.ToString("R", CultureInfo.InvariantCulture) + "," +
                "\"positionOffsetBytes\":0,\"voxelFlatIndexOffsetBytes\":28,\"coordinates\":\"Unity world metres\"," +
                "\"crossingEvidence\":\"uint4 per vertex; w low12=edge mask, w bits12..23=endpoint-A sign, xyz=packed t8\"," +
                "\"crossingSpan\":\"uint3 per vertex; packed 6-bit span and 2-bit pusher class\"," +
                "\"rawSurfaceNets\":\"float3 full-volume voxel coordinate per vertex\"," +
                "\"source\":\"actual published Front buffers, never candidate Surface buffers\"}";
            await Task.Run(() =>
            {
                Directory.CreateDirectory(directory);
                var csv = new StringBuilder("block,generation,targetEpoch,candidateEpoch,publishedEpoch,vertices,indices,drawArgsIndices,visible,pending,lastSubmittedIndices,lastSubmittedFrame,verticesFile,indicesFile,argsFile,crossingsFile,crossingSpansFile,rawSurfaceNetsFile\n");
                foreach (CaptureFront front in fronts)
                {
                    string stem = "front_" + front.Block.ToString(CultureInfo.InvariantCulture);
                    byte[] args = front.ArgsRead.Result;
                    if (front.VertexRead.Result.Length != front.Vertices * 32 ||
                        front.IndexRead.Result.Length != front.Indices * 4 ||
                        front.CrossingRead.Result.Length != front.Vertices * 16 ||
                        front.CrossingSpanRead.Result.Length != front.Vertices * 12 ||
                        front.RawSurfaceNetsRead.Result.Length != front.Vertices * 12 ||
                        (front.Source != null && args.Length != 20))
                        throw new InvalidDataException("Front readback byte count mismatch");
                    uint drawn = args.Length >= 4 ? BitConverter.ToUInt32(args, 0) : 0;
                    if (drawn != (uint)front.Indices)
                        throw new InvalidDataException("Front index count disagrees with actual indirect draw args");
                    File.WriteAllBytes(Path.Combine(directory, stem + ".vertices.bin"), front.VertexRead.Result);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".indices.bin"), front.IndexRead.Result);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".args.bin"), args);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".crossings.u32x4.bin"),
                        front.CrossingRead.Result);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".crossing_spans.u32x3.bin"),
                        front.CrossingSpanRead.Result);
                    File.WriteAllBytes(Path.Combine(directory, stem + ".raw_surface_nets.f32x3.bin"),
                        front.RawSurfaceNetsRead.Result);
                    csv.Append(front.Block).Append(',').Append(front.Generation).Append(',').Append(front.Target).Append(',')
                        .Append(front.Candidate).Append(',').Append(front.Published).Append(',').Append(front.Vertices).Append(',')
                        .Append(front.Indices).Append(',').Append(drawn).Append(',').Append(front.Visible ? 1 : 0).Append(',')
                        .Append(front.Pending ? 1 : 0).Append(',').Append(front.SubmittedIndices).Append(',')
                        .Append(front.SubmittedFrame).Append(',').Append(stem).Append(".vertices.bin,")
                        .Append(stem).Append(".indices.bin,").Append(stem).Append(".args.bin,")
                        .Append(stem).Append(".crossings.u32x4.bin,")
                        .Append(stem).Append(".crossing_spans.u32x3.bin,")
                        .Append(stem).AppendLine(".raw_surface_nets.f32x3.bin");
                }
                File.WriteAllText(Path.Combine(directory, "front_blocks.csv"), csv.ToString(), new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory, "schema.json"), metadata, new UTF8Encoding(false));
            });
        }
    }

    internal sealed partial class InfiniTamBlockMeshPipeline
    {
        private double _captureQueueMs, _captureCallbackMs;
        private int _capturePublished, _captureCallbacks;
        internal float CaptureQueueMs => _capturePublished > 0 ? (float)(_captureQueueMs / _capturePublished) : -1f;
        internal float CaptureCallbackMs => _captureCallbacks > 0 ? (float)(_captureCallbackMs / _captureCallbacks) : -1f;
        internal int CapturePendingCommits => _inFlightCommits;
        internal int CaptureBuiltBlocks { get { int n = 0; foreach (Block b in _blocks) if (b.Built) n++; return n; } }
        internal bool CaptureVisible { get { foreach (Block b in _blocks) if (b.Renderer != null && b.Renderer.RenderVisible && b.IndexCount > 0) return true; return false; } }
        internal int CaptureSubmittedIndices { get { int n = 0; foreach (Block b in _blocks) if (b.Renderer != null && b.Renderer.LastSubmittedFrame >= Time.frameCount - 2) n += b.Renderer.LastSubmittedVertexCount; return n; } }
        internal int CaptureSubmittedFrame { get { int n = -1; foreach (Block b in _blocks) if (b.Renderer != null) n = Math.Max(n, b.Renderer.LastSubmittedFrame); return n; } }

        private void AuditBlockEvent(string kind, Block block)
        {
            ScanReplaySessionPackage package = ScanReplaySessionPackage.Active;
            if (package == null || !package.LockedRoiCaptureActive)
                return;
            if (kind == "callback_ready") { _captureCallbackMs += (Time.realtimeSinceStartup - block.CommitStartedAt) * 1000.0; _captureCallbacks++; }
            if (kind == "published")
            {
                _captureQueueMs +=
                    (Time.realtimeSinceStartup - block.AuditQueuedAt) * 1000.0;
                _capturePublished++;
            }
            package.RecordInfiniTamBlockEvent(kind, _generation, block.Index,
                block.TargetEpoch, block.CandidateEpoch, block.PublishedEpoch,
                block.Front?.KnownVertexCount ?? 0, block.Front?.KnownDrawVertexCount ?? 0);
        }

        internal List<CaptureFront> CaptureFronts()
        {
            var result = new List<CaptureFront>(_blocks.Count);
            foreach (Block b in _blocks)
            {
                result.Add(new CaptureFront {
                    Block = b.Index, Generation = _generation, Target = b.TargetEpoch,
                    Candidate = b.CandidateEpoch, Published = b.PublishedEpoch, Source = b.Front,
                    Vertices = b.Front?.KnownVertexCount ?? 0, Indices = b.Front?.KnownDrawVertexCount ?? 0,
                    Visible = b.Renderer != null && b.Renderer.RenderVisible,
                    Pending = b.Queued || b.CommitPending || b.ReadyToPublish,
                    SubmittedIndices = b.Renderer?.LastSubmittedVertexCount ?? 0,
                    SubmittedFrame = b.Renderer?.LastSubmittedFrame ?? -1 });
            }
            return result;
        }
    }

    internal sealed partial class ScanReplaySessionPackage
    {
        private bool _lockedRoiCaptureEnabled;
        private bool _lockedRoiCheckpointComplete;
        internal bool LockedRoiCaptureActive => _lockedRoiCaptureEnabled;
        internal bool RequiredCaptureComplete => !_lockedRoiCaptureEnabled ||
                                                 _lockedRoiCheckpointComplete;
        internal bool CaptureSealedClean => _finalized && _finalizeClean;
        private readonly StringBuilder _captureBlockEvents = new StringBuilder(4096);
        private string _captureBlockPath;

        private void BeginLockedRoiCaptureLedger()
        {
            MeshExtractor mesh = MeshExtractor.Instance;
            _lockedRoiCaptureEnabled = mesh != null &&
                mesh.LockedRoiHistoryCaptureEnabled &&
                VolumeIntegrator.Instance != null &&
                VolumeIntegrator.Instance.InfiniTamBaselineEnabled;
            _lockedRoiCheckpointComplete = false;
            _captureBlockEvents.Clear();
            _captureBlockPath = string.Empty;
            if (!_lockedRoiCaptureEnabled)
                return;
            _captureBlockPath = Path.Combine(_sessionDirectory, "runtime_timeline", "infinitam_blocks.csv");
            File.WriteAllText(_captureBlockPath,
                "event,unityFrame,unscaledTime,integrationCount,dirtyEpoch,generation,block,targetEpoch,candidateEpoch,publishedEpoch,frontVertices,frontIndices\n", new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(_sessionDirectory, "capture_route.json"),
                "{\"schema\":\"scancover.capture_route.v1\",\"lockedRoiCapture\":true," +
                "\"infinitamBaseline\":true," +
                "\"timelinePaperColumns\":\"active production route, see meshRoute column\"," +
                "\"lockedRoiHistory\":\"artifacts/locked_roi_history\"," +
                "\"lockedRoiDepthWitness\":\"artifacts/locked_roi_history/fusion_witness.csv\"," +
                "\"captureScope\":\"only the locked blue world-space frame; no whole-volume snapshot\"," +
                "\"armingRule\":\"operator positions and locks blue frame before scan trigger\"," +
                "\"initialSample\":\"integration=0 empty-volume ROI queued before camera and depth start\"," +
                "\"blockEpochs\":\"runtime_timeline/infinitam_blocks.csv\"}", new UTF8Encoding(false));
            LockedRoiHistoryCapture.Instance?.BeginSession();
        }

        internal void AbortLockedRoiCaptureBegin()
        {
            if (_lockedRoiCaptureEnabled)
                LockedRoiHistoryCapture.Instance?.StopForSeal();
            _lockedRoiCaptureEnabled = false;
            _lockedRoiCheckpointComplete = false;
            _captureBlockEvents.Clear();
            _captureBlockPath = string.Empty;
            _active = false;
            if (Active == this) Active = null;
            if (Latest == this) Latest = null;
        }

        internal void EnsureTsdfResponsibilityCapture(VolumeIntegrator volume)
        {
            if (_active && !_tsdfResponsibilityStarted && volume != null)
                _tsdfResponsibilityStarted = volume.BeginTsdfResponsibilityCapture();
        }

        internal void RecordInfiniTamBlockEvent(string kind, int generation, int block,
            uint target, uint candidate, uint published, int vertices, int indices)
        {
            if (!_active || !_lockedRoiCaptureEnabled) return;
            VolumeIntegrator v = VolumeIntegrator.Instance;
            _captureBlockEvents.Append(kind).Append(',').Append(Time.frameCount).Append(',')
                .Append(D(Time.unscaledTimeAsDouble)).Append(',').Append(v != null ? v.IntegrationCount : -1)
                .Append(',').Append(v != null ? v.DirtyEpoch : 0u).Append(',').Append(generation).Append(',')
                .Append(block).Append(',').Append(target).Append(',').Append(candidate).Append(',')
                .Append(published).Append(',').Append(vertices).Append(',').Append(indices).AppendLine();
        }

        private void FlushLockedRoiCaptureLedger()
        {
            if (_captureBlockEvents.Length == 0 || string.IsNullOrEmpty(_captureBlockPath)) return;
            try { File.AppendAllText(_captureBlockPath, _captureBlockEvents.ToString(), new UTF8Encoding(false)); _captureBlockEvents.Clear(); }
            catch (Exception e) { Interlocked.Increment(ref _runtimeLedgerWriteErrors); Logger.Warning("块发布账写入失败: " + e.Message); }
        }

        private async void CaptureLockedRoiOnlineArtifacts()
        {
            if (!_lockedRoiCaptureEnabled)
                return;
            // One package-owned async operation, with one matching decrement.
            // B may not clear the volume or dispose Fronts until it completes.
            Interlocked.Increment(ref _artifactRequests);
            try
            {
                VolumeIntegrator volume = VolumeIntegrator.Instance;
                if (volume == null) throw new InvalidOperationException("VolumeIntegrator missing");
                LockedRoiHistoryCapture capture = LockedRoiHistoryCapture.Instance;
                if (capture == null)
                    throw new InvalidOperationException("locked ROI history capture missing");
                string relative = "artifacts/locked_roi_history";
                await capture.ExportAsync(Path.Combine(_sessionDirectory, relative));
                WriteArtifactStatus("locked_roi_history", "copied", string.Empty, relative);
                _lockedRoiCheckpointComplete = true;
            }
            catch (Exception e)
            {
                WriteArtifactStatus("locked_roi_history", "failed", e.Message, string.Empty);
            }
            finally { Interlocked.Decrement(ref _artifactRequests); }
        }
    }
}



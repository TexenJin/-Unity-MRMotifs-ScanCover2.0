using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace ScanCover.EvidenceCapture
{
    internal sealed class EvidenceBufferPayload
    {
        public string Name;
        public string Semantic;
        public string GraphicsFormat;
        public int Width;
        public int Height;
        public int Depth;
        public byte[] Bytes;
    }

    internal sealed class EvidenceFramePayload
    {
        public int FrameIndex;
        public int UnityFrame;
        public long RequestUtcTicks;
        public string RequestUtcIso;
        public string MetadataJson;
        public int ViewSector;
        public readonly float[] EyeWorldPositions = new float[6];
        public readonly List<EvidenceBufferPayload> Buffers = new List<EvidenceBufferPayload>(8);
    }

    internal sealed class EvidenceWriteSummary
    {
        public int FrameIndex;
        public bool Success;
        public string RelativePath;
        public string Sha256;
        public string Error;
        public float RawNormalValidRatio;
        public float FilteredNormalValidRatio;
        public float NormalAgreementCoverage;
        public float SelfConsistentNormalCoverage;
        public float EdgeRiskRatio;
        public int RemedialCategoryMask;
        public int ViewSector;
    }

    /// <summary>
    /// Dedicated single-consumer writer. GPU callbacks only freeze byte arrays; all file I/O,
    /// hashes and statistics run here so Quest's render thread is not blocked.
    /// </summary>
    internal sealed class EvidenceSessionWriter : IDisposable
    {
        private const string Magic = "SCDEVID1";
        private const int ContainerVersion = 1;

        private readonly string _sessionDirectory;
        private readonly string _framesDirectory;
        private readonly string _manifestPath;
        private readonly string _checksumPath;
        private readonly string _auditPath;
        private readonly ConcurrentQueue<EvidenceFramePayload> _queue = new ConcurrentQueue<EvidenceFramePayload>();
        private readonly ConcurrentQueue<EvidenceWriteSummary> _completed = new ConcurrentQueue<EvidenceWriteSummary>();
        private readonly AutoResetEvent _wake = new AutoResetEvent(false);
        private readonly Thread _thread;
        private volatile bool _stopRequested;
        private volatile bool _finished;
        private volatile bool _sessionFailed;
        private int _pendingCount;
        private int _writtenCount;
        private int _failedCount;
        private string _lastError = string.Empty;
        private string _sessionFailureCode = string.Empty;
        private string _sessionFailureDetail = string.Empty;
        private string _previousRawSha = string.Empty;
        private long _totalBytes;

        public EvidenceSessionWriter(string sessionDirectory)
        {
            _sessionDirectory = sessionDirectory;
            _framesDirectory = Path.Combine(sessionDirectory, "frames");
            _manifestPath = Path.Combine(sessionDirectory, "manifest.csv");
            _checksumPath = Path.Combine(sessionDirectory, "checksums.sha256");
            _auditPath = Path.Combine(sessionDirectory, "capture_audit.json");

            Directory.CreateDirectory(_framesDirectory);
            File.WriteAllText(
                _manifestPath,
                "frame,unity_frame,request_utc,file,bytes,sha256,raw_sha256,raw_duplicate,valid_ratio,left_valid,right_valid,normal_valid,raw_normal_valid,filtered_normal_valid,normal_agreement_coverage,normal_self_consistent_coverage,edge_crossing_normal_coverage,edge_risk,max_jump_m,mean_depth_m,min_depth_m,max_depth_m,status,error\n",
                new UTF8Encoding(false));
            File.WriteAllText(_checksumPath, string.Empty, new UTF8Encoding(false));

            _thread = new Thread(WriterLoop)
            {
                IsBackground = true,
                Name = "ScanCover 深度证据写盘"
            };
            _thread.Start();
        }

        public int PendingCount => Volatile.Read(ref _pendingCount);
        public int WrittenCount => Volatile.Read(ref _writtenCount);
        public int FailedCount => Volatile.Read(ref _failedCount);
        public long TotalBytes => Interlocked.Read(ref _totalBytes);
        public bool IsFinished => _finished;
        public string LastError => _lastError;

        public void Enqueue(EvidenceFramePayload payload)
        {
            if (payload == null || _stopRequested)
            {
                return;
            }

            Interlocked.Increment(ref _pendingCount);
            _queue.Enqueue(payload);
            _wake.Set();
        }

        public bool TryDequeueCompleted(out EvidenceWriteSummary summary)
        {
            return _completed.TryDequeue(out summary);
        }

        public void RequestStop()
        {
            _stopRequested = true;
            _wake.Set();
        }

        public void MarkSessionFailed(string code, string detail)
        {
            if (_sessionFailed) return;
            _sessionFailureCode = code ?? string.Empty;
            _sessionFailureDetail = detail ?? string.Empty;
            _sessionFailed = true;
        }

        private void WriterLoop()
        {
            while (!_stopRequested || !_queue.IsEmpty)
            {
                if (!_queue.TryDequeue(out EvidenceFramePayload payload))
                {
                    _wake.WaitOne(250);
                    continue;
                }

                EvidenceWriteSummary summary = WriteFrame(payload);
                _completed.Enqueue(summary);
                Interlocked.Decrement(ref _pendingCount);
                if (summary.Success)
                {
                    Interlocked.Increment(ref _writtenCount);
                }
                else
                {
                    Interlocked.Increment(ref _failedCount);
                    _lastError = summary.Error ?? "未知写盘错误";
                }
            }

            WriteAudit();
            _finished = true;
        }

        private EvidenceWriteSummary WriteFrame(EvidenceFramePayload payload)
        {
            string fileName = $"frame_{payload.FrameIndex:D7}.scede";
            string relativePath = "frames/" + fileName;
            string finalPath = Path.Combine(_framesDirectory, fileName);
            string temporaryPath = finalPath + ".tmp";
            var summary = new EvidenceWriteSummary
            {
                FrameIndex = payload.FrameIndex,
                RelativePath = relativePath
            };

            try
            {
                byte[] metadataBytes = Encoding.UTF8.GetBytes(payload.MetadataJson ?? "{}");
                using (var stream = new FileStream(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 256, FileOptions.SequentialScan))
                using (var writer = new BinaryWriter(stream, Encoding.UTF8, false))
                {
                    writer.Write(Encoding.ASCII.GetBytes(Magic));
                    writer.Write(ContainerVersion);
                    writer.Write(metadataBytes.Length);
                    writer.Write(payload.Buffers.Count);
                    writer.Write(metadataBytes);

                    foreach (EvidenceBufferPayload buffer in payload.Buffers)
                    {
                        WriteUtf8(writer, buffer.Name);
                        WriteUtf8(writer, buffer.Semantic);
                        WriteUtf8(writer, buffer.GraphicsFormat);
                        writer.Write(buffer.Width);
                        writer.Write(buffer.Height);
                        writer.Write(buffer.Depth);
                        writer.Write(buffer.Bytes?.LongLength ?? 0L);
                        writer.Write(Crc32.Compute(buffer.Bytes));
                        if (buffer.Bytes != null)
                        {
                            writer.Write(buffer.Bytes);
                        }
                    }
                }

                if (File.Exists(finalPath))
                {
                    throw new IOException("证据帧目标文件已存在，拒绝覆盖：" + finalPath);
                }
                File.Move(temporaryPath, finalPath);

                summary.Sha256 = ComputeSha256File(finalPath);
                summary.Success = true;
                long fileLength = new FileInfo(finalPath).Length;
                Interlocked.Add(ref _totalBytes, fileLength);

                EvidenceFrameStatistics stats = EvidenceFrameStatistics.From(payload.Buffers, payload.EyeWorldPositions);
                summary.RawNormalValidRatio = stats.RawNormalValidRatio;
                summary.FilteredNormalValidRatio = stats.FilteredNormalValidRatio;
                summary.NormalAgreementCoverage = stats.NormalAgreementCoverage;
                summary.SelfConsistentNormalCoverage = stats.SelfConsistentNormalCoverage;
                summary.EdgeRiskRatio = stats.EdgeRiskRatio;
                summary.RemedialCategoryMask = stats.RemedialCategoryMask;
                summary.ViewSector = payload.ViewSector;
                string rawSha = ComputeRawSha(payload.Buffers);
                bool duplicateRaw = !string.IsNullOrEmpty(rawSha) && rawSha == _previousRawSha;
                _previousRawSha = rawSha;

                File.AppendAllText(_checksumPath, summary.Sha256 + "  " + relativePath.Replace('\\', '/') + "\n", new UTF8Encoding(false));
                File.AppendAllText(_manifestPath, BuildManifestLine(payload, summary, fileLength, rawSha, duplicateRaw, stats), new UTF8Encoding(false));
            }
            catch (Exception ex)
            {
                summary.Success = false;
                summary.Error = ex.GetType().Name + ": " + ex.Message;
                try
                {
                    if (File.Exists(temporaryPath))
                    {
                        File.Delete(temporaryPath);
                    }
                    File.AppendAllText(_manifestPath, BuildManifestLine(payload, summary, 0, string.Empty, false, default), new UTF8Encoding(false));
                }
                catch
                {
                    // Preserve the original failure; the session audit will still expose it.
                }
            }

            return summary;
        }

        private static string BuildManifestLine(
            EvidenceFramePayload payload,
            EvidenceWriteSummary summary,
            long fileLength,
            string rawSha,
            bool duplicateRaw,
            EvidenceFrameStatistics stats)
        {
            string[] fields =
            {
                payload.FrameIndex.ToString(CultureInfo.InvariantCulture),
                payload.UnityFrame.ToString(CultureInfo.InvariantCulture),
                Csv(payload.RequestUtcIso), Csv(summary.RelativePath),
                fileLength.ToString(CultureInfo.InvariantCulture), Csv(summary.Sha256), Csv(rawSha),
                duplicateRaw ? "1" : "0",
                F(stats.ValidRatio), F(stats.LeftValidRatio), F(stats.RightValidRatio), F(stats.NormalValidRatio),
                F(stats.RawNormalValidRatio), F(stats.FilteredNormalValidRatio),
                F(stats.NormalAgreementCoverage), F(stats.SelfConsistentNormalCoverage), F(stats.EdgeCrossingNormalCoverage),
                F(stats.EdgeRiskRatio), F(stats.MaxJumpMetres), F(stats.MeanDepthMetres), F(stats.MinDepthMetres), F(stats.MaxDepthMetres),
                summary.Success ? "ok" : "failed", Csv(summary.Error)
            };
            return string.Join(",", fields) + "\n";
        }

        private void WriteAudit()
        {
            try
            {
                string json = "{\n" +
                              "  \"schema\": \"ScanCoverDepthEvidenceAudit/v1\",\n" +
                              "  \"finishedUtc\": \"" + Json(DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)) + "\",\n" +
                              "  \"sessionStatus\": \"" + (_sessionFailed ? "failed" : "complete") + "\",\n" +
                              "  \"sessionFailed\": " + (_sessionFailed ? "true" : "false") + ",\n" +
                              "  \"fatalFailureCode\": \"" + Json(_sessionFailureCode) + "\",\n" +
                              "  \"fatalFailureDetail\": \"" + Json(_sessionFailureDetail) + "\",\n" +
                              "  \"writtenFrames\": " + WrittenCount.ToString(CultureInfo.InvariantCulture) + ",\n" +
                              "  \"failedFrames\": " + FailedCount.ToString(CultureInfo.InvariantCulture) + ",\n" +
                              "  \"totalBytes\": " + TotalBytes.ToString(CultureInfo.InvariantCulture) + ",\n" +
                              "  \"lastError\": \"" + Json(_lastError) + "\"\n" +
                              "}\n";
                File.WriteAllText(_auditPath, json, new UTF8Encoding(false));
            }
            catch
            {
                // The manifest and frame files remain independently recoverable.
            }
        }

        private static string ComputeRawSha(List<EvidenceBufferPayload> buffers)
        {
            foreach (EvidenceBufferPayload buffer in buffers)
            {
                if (buffer.Name == "raw_depth_r32f")
                {
                    return ComputeSha256(buffer.Bytes);
                }
            }
            return string.Empty;
        }

        private static string ComputeSha256(byte[] bytes)
        {
            if (bytes == null)
            {
                return string.Empty;
            }
            using (SHA256 sha = SHA256.Create())
            {
                return Hex(sha.ComputeHash(bytes));
            }
        }

        private static string ComputeSha256File(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 256, FileOptions.SequentialScan))
            using (SHA256 sha = SHA256.Create())
            {
                return Hex(sha.ComputeHash(stream));
            }
        }

        private static string Hex(byte[] bytes)
        {
            var builder = new StringBuilder(bytes.Length * 2);
            foreach (byte value in bytes)
            {
                builder.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            }
            return builder.ToString();
        }

        private static void WriteUtf8(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        private static string F(float value)
        {
            return value.ToString("0.######", CultureInfo.InvariantCulture);
        }

        private static string Csv(string value)
        {
            string safe = value ?? string.Empty;
            return "\"" + safe.Replace("\"", "\"\"") + "\"";
        }

        private static string Json(string value)
        {
            return (value ?? string.Empty)
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n");
        }

        public void Dispose()
        {
            RequestStop();
            if (_thread != null && _thread.IsAlive)
            {
                _thread.Join(1500);
            }
            _wake.Dispose();
        }

        private struct EvidenceFrameStatistics
        {
            public float ValidRatio;
            public float LeftValidRatio;
            public float RightValidRatio;
            public float NormalValidRatio;
            public float RawNormalValidRatio;
            public float FilteredNormalValidRatio;
            public float NormalAgreementCoverage;
            public float SelfConsistentNormalCoverage;
            public float EdgeCrossingNormalCoverage;
            public float EdgeRiskRatio;
            public float MaxJumpMetres;
            public float MeanDepthMetres;
            public float MinDepthMetres;
            public float MaxDepthMetres;
            public int RemedialCategoryMask;

            public static EvidenceFrameStatistics From(List<EvidenceBufferPayload> buffers, float[] eyeWorldPositions = null)
            {
                EvidenceBufferPayload depth = null;
                EvidenceBufferPayload rawNormal = null;
                EvidenceBufferPayload filteredNormal = null;
                EvidenceBufferPayload diagnostics = null;
                EvidenceBufferPayload world = null;
                foreach (EvidenceBufferPayload buffer in buffers)
                {
                    if (buffer.Name == "depth_metrics_rgba32f") depth = buffer;
                    else if (buffer.Name == "world_normal_raw_rgba32f") rawNormal = buffer;
                    else if (buffer.Name == "world_normal_neighbour_rgba32f") filteredNormal = buffer;
                    else if (buffer.Name == "diagnostics_rgba32f") diagnostics = buffer;
                    else if (buffer.Name == "world_position_raw_rgba32f") world = buffer;
                }

                var result = new EvidenceFrameStatistics { MinDepthMetres = float.PositiveInfinity };
                if (depth?.Bytes == null || depth.Bytes.Length < 16)
                {
                    result.MinDepthMetres = 0f;
                    return result;
                }

                int pixelCount = depth.Width * depth.Height * Math.Max(1, depth.Depth);
                int eyePixels = depth.Width * depth.Height;
                int valid = 0;
                int leftValid = 0;
                int rightValid = 0;
                double sumDepth = 0.0;
                for (int i = 0; i < pixelCount; ++i)
                {
                    int offset = i * 16;
                    if (offset + 15 >= depth.Bytes.Length) break;
                    float radial = BitConverter.ToSingle(depth.Bytes, offset + 8);
                    float validity = BitConverter.ToSingle(depth.Bytes, offset + 12);
                    if (validity > 0.5f && radial > 0f && !float.IsNaN(radial) && !float.IsInfinity(radial))
                    {
                        valid++;
                        if (i < eyePixels) leftValid++; else rightValid++;
                        sumDepth += radial;
                        result.MinDepthMetres = Math.Min(result.MinDepthMetres, radial);
                        result.MaxDepthMetres = Math.Max(result.MaxDepthMetres, radial);
                    }
                }

                result.ValidRatio = pixelCount > 0 ? (float)valid / pixelCount : 0f;
                result.LeftValidRatio = eyePixels > 0 ? (float)leftValid / eyePixels : 0f;
                result.RightValidRatio = eyePixels > 0 ? (float)rightValid / eyePixels : 0f;
                result.MeanDepthMetres = valid > 0 ? (float)(sumDepth / valid) : 0f;
                if (float.IsInfinity(result.MinDepthMetres)) result.MinDepthMetres = 0f;

                if (rawNormal?.Bytes != null)
                {
                    int count = Math.Min(pixelCount, rawNormal.Bytes.Length / 16);
                    int normalValid = 0;
                    for (int i = 0; i < count; ++i)
                    {
                        if (BitConverter.ToSingle(rawNormal.Bytes, i * 16 + 12) > 0.5f) normalValid++;
                    }
                    result.NormalValidRatio = count > 0 ? (float)normalValid / count : 0f;
                    result.RawNormalValidRatio = result.NormalValidRatio;
                }

                if (filteredNormal?.Bytes != null)
                {
                    int count = Math.Min(pixelCount, filteredNormal.Bytes.Length / 16);
                    int normalValid = 0;
                    for (int i = 0; i < count; ++i)
                    {
                        if (BitConverter.ToSingle(filteredNormal.Bytes, i * 16 + 12) > 0.5f) normalValid++;
                    }
                    result.FilteredNormalValidRatio = count > 0 ? (float)normalValid / count : 0f;
                }

                if (diagnostics?.Bytes != null)
                {
                    int count = Math.Min(pixelCount, diagnostics.Bytes.Length / 16);
                    int edges = 0;
                    for (int i = 0; i < count; ++i)
                    {
                        int offset = i * 16;
                        float jump = BitConverter.ToSingle(diagnostics.Bytes, offset);
                        float edge = BitConverter.ToSingle(diagnostics.Bytes, offset + 12);
                        if (edge > 0.5f) edges++;
                        if (!float.IsNaN(jump) && !float.IsInfinity(jump)) result.MaxJumpMetres = Math.Max(result.MaxJumpMetres, jump);
                    }
                    result.EdgeRiskRatio = count > 0 ? (float)edges / count : 0f;
                }

                if (rawNormal?.Bytes != null && filteredNormal?.Bytes != null && diagnostics?.Bytes != null)
                {
                    int count = Math.Min(
                        pixelCount,
                        Math.Min(rawNormal.Bytes.Length / 16,
                            Math.Min(filteredNormal.Bytes.Length / 16, diagnostics.Bytes.Length / 16)));
                    int agreeing = 0;
                    int reliable = 0;
                    int edgeCrossing = 0;
                    const float minimumAgreementDot = 0.9396926f; // cos(20 degrees)
                    for (int i = 0; i < count; ++i)
                    {
                        int offset = i * 16;
                        bool rawValid = BitConverter.ToSingle(rawNormal.Bytes, offset + 12) > 0.5f;
                        bool filteredValid = BitConverter.ToSingle(filteredNormal.Bytes, offset + 12) > 0.5f;
                        bool edgeRisk = BitConverter.ToSingle(diagnostics.Bytes, offset + 12) > 0.5f;
                        if (rawValid && edgeRisk) edgeCrossing++;
                        if (!rawValid || !filteredValid) continue;

                        float rx = BitConverter.ToSingle(rawNormal.Bytes, offset);
                        float ry = BitConverter.ToSingle(rawNormal.Bytes, offset + 4);
                        float rz = BitConverter.ToSingle(rawNormal.Bytes, offset + 8);
                        float fx = BitConverter.ToSingle(filteredNormal.Bytes, offset);
                        float fy = BitConverter.ToSingle(filteredNormal.Bytes, offset + 4);
                        float fz = BitConverter.ToSingle(filteredNormal.Bytes, offset + 8);
                        float rawLength = (float)Math.Sqrt(rx * rx + ry * ry + rz * rz);
                        float filteredLength = (float)Math.Sqrt(fx * fx + fy * fy + fz * fz);
                        if (rawLength <= 1e-6f || filteredLength <= 1e-6f) continue;
                        float agreement = Math.Abs((rx * fx + ry * fy + rz * fz) / (rawLength * filteredLength));
                        if (agreement < minimumAgreementDot) continue;
                        agreeing++;
                        if (!edgeRisk) reliable++;
                    }
                    result.NormalAgreementCoverage = count > 0 ? (float)agreeing / count : 0f;
                    result.SelfConsistentNormalCoverage = count > 0 ? (float)reliable / count : 0f;
                    result.EdgeCrossingNormalCoverage = count > 0 ? (float)edgeCrossing / count : 0f;
                }

                if (world?.Bytes != null && rawNormal?.Bytes != null && filteredNormal?.Bytes != null &&
                    eyeWorldPositions != null && eyeWorldPositions.Length >= 6)
                {
                    int count = Math.Min(pixelCount,
                        Math.Min(world.Bytes.Length / 16,
                            Math.Min(rawNormal.Bytes.Length / 16, filteredNormal.Bytes.Length / 16)));
                    int near = 0;
                    int middle = 0;
                    int far = 0;
                    int grazing = 0;
                    int angleUsable = 0;
                    int minimumSamples = Math.Max(128, valid / 200); // 0.5% of this frame's valid depth.
                    const float minimumAgreementDot = 0.9396926f;
                    const float grazingCosine = 0.5f; // 60 degrees or more from frontal.
                    for (int i = 0; i < count; ++i)
                    {
                        int offset = i * 16;
                        float radial = BitConverter.ToSingle(depth.Bytes, offset + 8);
                        float depthValid = BitConverter.ToSingle(depth.Bytes, offset + 12);
                        if (depthValid <= 0.5f || radial < 0.35f || radial >= 4.0f) continue;
                        if (radial < 0.75f) near++;
                        else if (radial < 1.5f) middle++;
                        else if (radial < 3.0f) far++;

                        bool rawValid = BitConverter.ToSingle(rawNormal.Bytes, offset + 12) > 0.5f;
                        bool filteredValid = BitConverter.ToSingle(filteredNormal.Bytes, offset + 12) > 0.5f;
                        if (!rawValid || !filteredValid) continue;
                        float rx = BitConverter.ToSingle(rawNormal.Bytes, offset);
                        float ry = BitConverter.ToSingle(rawNormal.Bytes, offset + 4);
                        float rz = BitConverter.ToSingle(rawNormal.Bytes, offset + 8);
                        float fx = BitConverter.ToSingle(filteredNormal.Bytes, offset);
                        float fy = BitConverter.ToSingle(filteredNormal.Bytes, offset + 4);
                        float fz = BitConverter.ToSingle(filteredNormal.Bytes, offset + 8);
                        float rawLength = (float)Math.Sqrt(rx * rx + ry * ry + rz * rz);
                        float filteredLength = (float)Math.Sqrt(fx * fx + fy * fy + fz * fz);
                        if (rawLength <= 1e-6f || filteredLength <= 1e-6f) continue;
                        float agreement = Math.Abs((rx * fx + ry * fy + rz * fz) / (rawLength * filteredLength));
                        if (agreement < minimumAgreementDot) continue;

                        float px = BitConverter.ToSingle(world.Bytes, offset);
                        float py = BitConverter.ToSingle(world.Bytes, offset + 4);
                        float pz = BitConverter.ToSingle(world.Bytes, offset + 8);
                        int eye = i < eyePixels ? 0 : 1;
                        float vx = eyeWorldPositions[eye * 3] - px;
                        float vy = eyeWorldPositions[eye * 3 + 1] - py;
                        float vz = eyeWorldPositions[eye * 3 + 2] - pz;
                        float viewLength = (float)Math.Sqrt(vx * vx + vy * vy + vz * vz);
                        if (viewLength <= 1e-6f) continue;
                        angleUsable++;
                        float incidenceCosine = Math.Abs((rx * vx + ry * vy + rz * vz) / (rawLength * viewLength));
                        if (incidenceCosine <= grazingCosine) grazing++;
                    }

                    int mask = 0;
                    if (near >= minimumSamples) mask |= 1;
                    if (middle >= minimumSamples) mask |= 2;
                    if (far >= minimumSamples) mask |= 4;
                    if (grazing >= Math.Max(64, angleUsable / 200)) mask |= 8;
                    result.RemedialCategoryMask = mask;
                }

                return result;
            }
        }
    }

    internal static class Crc32
    {
        private static readonly uint[] Table = BuildTable();

        public static uint Compute(byte[] bytes)
        {
            if (bytes == null) return 0u;
            uint crc = 0xffffffffu;
            foreach (byte value in bytes)
            {
                crc = Table[(crc ^ value) & 0xff] ^ (crc >> 8);
            }
            return crc ^ 0xffffffffu;
        }

        private static uint[] BuildTable()
        {
            var table = new uint[256];
            for (uint i = 0; i < table.Length; ++i)
            {
                uint value = i;
                for (int bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) != 0u ? 0xedb88320u ^ (value >> 1) : value >> 1;
                }
                table[i] = value;
            }
            return table;
        }
    }
}

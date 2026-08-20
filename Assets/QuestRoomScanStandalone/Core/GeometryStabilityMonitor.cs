using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Read-only comparison of consecutive production TSDF snapshots.  It measures
    /// the near-zero band directly, before confidence colouring and chunk meshing
    /// can reinterpret the result.  The production volume is never written.
    /// </summary>
    internal sealed class GeometryStabilityMonitor : MonoBehaviour
    {
        private const int StatsCount = 72;
        private const int HistogramBase = 8;
        private const int HistogramCount = 64;

        internal readonly struct Sample
        {
            public readonly float Elapsed;
            public readonly uint PreviousSurface;
            public readonly uint CurrentSurface;
            public readonly uint Retained;
            public readonly uint Lost;
            public readonly uint Added;
            public readonly uint Moving;
            public readonly float SurvivalPercent;
            public readonly float ChurnPercent;
            public readonly float P50Millimeters;
            public readonly float P95Millimeters;
            public readonly float MaxMillimeters;

            public Sample(float elapsed, uint previousSurface, uint currentSurface,
                uint retained, uint lost, uint added, uint moving,
                float survivalPercent, float churnPercent,
                float p50Millimeters, float p95Millimeters, float maxMillimeters)
            {
                Elapsed = elapsed;
                PreviousSurface = previousSurface;
                CurrentSurface = currentSurface;
                Retained = retained;
                Lost = lost;
                Added = added;
                Moving = moving;
                SurvivalPercent = survivalPercent;
                ChurnPercent = churnPercent;
                P50Millimeters = p50Millimeters;
                P95Millimeters = p95Millimeters;
                MaxMillimeters = maxMillimeters;
            }
        }

        private static readonly int CurrentVolumeId = Shader.PropertyToID("_CurrentVolume");
        private static readonly int PreviousVolumeId = Shader.PropertyToID("_PreviousVolume");
        private static readonly int PreviousVolumeRwId = Shader.PropertyToID("_PreviousVolumeRW");
        private static readonly int StatsId = Shader.PropertyToID("_GeometryStats");
        private static readonly int VoxelCountId = Shader.PropertyToID("_VoxelCount");
        private static readonly int MinMeshWeightId = Shader.PropertyToID("_MinMeshWeight");
        private static readonly int SurfaceBandId = Shader.PropertyToID("_SurfaceBand");
        private static readonly int VoxelDistanceId = Shader.PropertyToID("_VoxelDistance");
        private static readonly int VoxelSizeId = Shader.PropertyToID("_VoxelSize");

        private readonly List<Sample> _samples = new List<Sample>(1024);
        private readonly uint[] _zeros = new uint[StatsCount];
        private VolumeIntegrator _volume;
        private ComputeShader _compute;
        private ComputeBuffer _stats;
        private RenderTexture _previous;
        private int _compareKernel = -1;
        private int _copyKernel = -1;
        private bool _hasBaseline;
        private bool _readbackPending;
        private int _pendingReadbackGeneration = -1;
        private bool _initialized;
        private int _generation;
        private float _sampleHz = 1f;
        private float _surfaceBand = 0.3f;
        private float _nextSampleTime;
        private float _sessionStarted;
        private Sample? _last;

        public bool IsReady => _initialized && _previous != null && _stats != null;
        public int SampleCount => _samples.Count;

        public void Initialize(VolumeIntegrator volume, float sampleHz, float surfaceBand)
        {
            _volume = volume;
            _sampleHz = Mathf.Max(0.1f, sampleHz);
            _surfaceBand = Mathf.Clamp(surfaceBand, 0.05f, 1f);
            _compute = Resources.Load<ComputeShader>("GeometryStabilityDiagnostics");
            if (_compute == null)
            {
                Logger.Warning("几何稳定诊断未启动：Resources/GeometryStabilityDiagnostics 缺失");
                return;
            }

            _compareKernel = _compute.FindKernel("CompareVolumes");
            _copyKernel = _compute.FindKernel("CopyCurrent");
            _stats = new ComputeBuffer(StatsCount, sizeof(uint));
            _stats.SetData(_zeros);
            if (_volume != null)
            {
                _volume.Cleared += ResetBaseline;
                _volume.TopologyInvalidated += ResetBaseline;
            }
            _initialized = true;
            ResetSession();
        }

        public void ResetSession()
        {
            _samples.Clear();
            _last = null;
            _sessionStarted = Time.realtimeSinceStartup;
            ResetBaseline();
        }

        private void Update()
        {
            if (!_initialized || _volume == null || _volume.Volume == null ||
                _volume.IntegrationCount <= 0 || _readbackPending)
                return;

            EnsurePreviousVolume();
            if (_previous == null) return;

            float now = Time.realtimeSinceStartup;
            if (!_hasBaseline)
            {
                DispatchCopy();
                _hasBaseline = true;
                _nextSampleTime = now + 1f / _sampleHz;
                return;
            }
            if (now < _nextSampleTime) return;

            _nextSampleTime = now + 1f / _sampleHz;
            _stats.SetData(_zeros);
            BindCommon(_compareKernel);
            _compute.SetTexture(_compareKernel, PreviousVolumeId, _previous);
            _compute.SetBuffer(_compareKernel, StatsId, _stats);
            DispatchVolume(_compareKernel);
            // GPU command ordering guarantees that this copy runs after CompareVolumes.
            DispatchCopy();

            _readbackPending = true;
            int requestGeneration = _generation;
            _pendingReadbackGeneration = requestGeneration;
            AsyncGPUReadback.Request(_stats, request =>
            {
                if (_pendingReadbackGeneration == requestGeneration)
                {
                    _readbackPending = false;
                    _pendingReadbackGeneration = -1;
                }
                if (!_initialized || requestGeneration != _generation || request.hasError)
                    return;
                var data = request.GetData<uint>();
                if (data.Length < StatsCount) return;
                ApplySample(data);
            });
        }

        private void ApplySample(Unity.Collections.NativeArray<uint> data)
        {
            uint previousSurface = data[0];
            uint currentSurface = data[1];
            uint retained = data[2];
            uint lost = data[3];
            uint added = data[4];
            uint moving = data[5];
            float survival = previousSurface > 0 ? 100f * retained / previousSurface : 100f;
            ulong union = (ulong)retained + lost + added;
            float churn = union > 0 ? 100f * (lost + added) / union : 0f;
            uint histogramTotal = 0;
            for (int i = 0; i < HistogramCount; i++) histogramTotal += data[HistogramBase + i];
            float p50 = HistogramPercentileMillimeters(data, histogramTotal, 0.50f);
            float p95 = HistogramPercentileMillimeters(data, histogramTotal, 0.95f);
            float maxMm = data[6] / 1024f * _volume.VoxelSize * 1000f;
            var sample = new Sample(
                Time.realtimeSinceStartup - _sessionStarted,
                previousSurface, currentSurface, retained, lost, added, moving,
                survival, churn, p50, p95, maxMm);
            _last = sample;
            _samples.Add(sample);
        }

        private float HistogramPercentileMillimeters(
            Unity.Collections.NativeArray<uint> data, uint total, float percentile)
        {
            if (total == 0) return 0f;
            uint target = (uint)Mathf.CeilToInt(total * percentile);
            uint cumulative = 0;
            for (int i = 0; i < HistogramCount; i++)
            {
                cumulative += data[HistogramBase + i];
                if (cumulative >= target)
                    return (i + 0.5f) / 16f * _volume.VoxelSize * 1000f;
            }
            return (HistogramCount - 0.5f) / 16f * _volume.VoxelSize * 1000f;
        }

        public string GetCompactStats()
        {
            if (!_initialized) return "几何诊断无";
            if (!_last.HasValue) return "几何预热";
            Sample s = _last.Value;
            return $"几何[留{s.SurvivalPercent:0}% 扰{s.ChurnPercent:0.0}% P95{s.P95Millimeters:0.0}mm]";
        }

        public void AppendSummary(StringBuilder sb)
        {
            if (sb == null) return;
            sb.AppendLine();
            sb.AppendLine("纯几何稳定账（只读 TSDF 近零带，不读颜色/置信度/网格块）:");
            sb.AppendLine($"采样数={_samples.Count} 采样频率={_sampleHz.ToString("F2", CultureInfo.InvariantCulture)}Hz " +
                          $"近零带=|sd|<{_surfaceBand.ToString("F2", CultureInfo.InvariantCulture)}");
            if (!_last.HasValue) return;
            Sample s = _last.Value;
            sb.AppendLine($"末次: 留存={s.SurvivalPercent:F2}% 扰动={s.ChurnPercent:F2}% " +
                          $"P50={s.P50Millimeters:F2}mm P95={s.P95Millimeters:F2}mm 最大={s.MaxMillimeters:F2}mm " +
                          $"旧/现/同/失/新={s.PreviousSurface}/{s.CurrentSurface}/{s.Retained}/{s.Lost}/{s.Added}");
            sb.AppendLine("位移口径: 同一体素的 TSDF sd 变化×截断距离，属于表面法向位移代理；块缝与出网延迟另账统计。");
        }

        public void AppendCsv(StringBuilder sb, string sessionId)
        {
            if (sb == null) return;
            sb.AppendLine("session_id,elapsed_s,previous_surface,current_surface,retained,lost,added,moving_ge_quarter_voxel,survival_percent,churn_percent,p50_mm,p95_mm,max_mm");
            for (int i = 0; i < _samples.Count; i++)
            {
                Sample s = _samples[i];
                sb.Append(sessionId).Append(',')
                  .Append(s.Elapsed.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.PreviousSurface).Append(',').Append(s.CurrentSurface).Append(',')
                  .Append(s.Retained).Append(',').Append(s.Lost).Append(',').Append(s.Added).Append(',')
                  .Append(s.Moving).Append(',')
                  .Append(s.SurvivalPercent.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.ChurnPercent.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.P50Millimeters.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.P95Millimeters.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.MaxMillimeters.ToString("F3", CultureInfo.InvariantCulture)).AppendLine();
            }
        }

        private void EnsurePreviousVolume()
        {
            RenderTexture source = _volume.Volume;
            if (_previous != null && _previous.width == source.width &&
                _previous.height == source.height && _previous.volumeDepth == source.volumeDepth &&
                _previous.graphicsFormat == source.graphicsFormat)
                return;

            ReleasePreviousVolume();
            var descriptor = source.descriptor;
            descriptor.dimension = TextureDimension.Tex3D;
            descriptor.volumeDepth = source.volumeDepth;
            descriptor.graphicsFormat = GraphicsFormat.R8G8_SNorm;
            descriptor.enableRandomWrite = true;
            descriptor.msaaSamples = 1;
            descriptor.depthBufferBits = 0;
            _previous = new RenderTexture(descriptor)
            {
                name = "GeometryStabilityPreviousTSDF",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Point
            };
            _previous.Create();
            _hasBaseline = false;
        }

        private void BindCommon(int kernel)
        {
            var count = _volume.VoxelCount;
            _compute.SetInts(VoxelCountId, count.x, count.y, count.z);
            _compute.SetFloat(MinMeshWeightId, _volume.MinMeshWeight);
            _compute.SetFloat(SurfaceBandId, _surfaceBand);
            _compute.SetFloat(VoxelDistanceId, _volume.VoxelDistance);
            _compute.SetFloat(VoxelSizeId, _volume.VoxelSize);
            _compute.SetTexture(kernel, CurrentVolumeId, _volume.Volume);
        }

        private void DispatchCopy()
        {
            BindCommon(_copyKernel);
            _compute.SetTexture(_copyKernel, PreviousVolumeRwId, _previous);
            DispatchVolume(_copyKernel);
        }

        private void DispatchVolume(int kernel)
        {
            var count = _volume.VoxelCount;
            _compute.Dispatch(kernel,
                Mathf.CeilToInt(count.x / 8f),
                Mathf.CeilToInt(count.y / 8f),
                Mathf.CeilToInt(count.z / 8f));
        }

        private void ResetBaseline()
        {
            _generation++;
            _hasBaseline = false;
            _readbackPending = false;
            _pendingReadbackGeneration = -1;
            _nextSampleTime = 0f;
        }

        private void ReleasePreviousVolume()
        {
            if (_previous == null) return;
            _previous.Release();
            Destroy(_previous);
            _previous = null;
        }

        private void OnDestroy()
        {
            _initialized = false;
            _generation++;
            if (_volume != null)
            {
                _volume.Cleared -= ResetBaseline;
                _volume.TopologyInvalidated -= ResetBaseline;
            }
            _stats?.Release();
            _stats = null;
            ReleasePreviousVolume();
        }
    }
}

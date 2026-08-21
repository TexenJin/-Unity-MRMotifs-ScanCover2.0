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
        private const int SpatialChunkSize = 32;
        private const int SpatialStride = 8;

        internal readonly struct SpatialSample
        {
            public readonly float Elapsed;
            public readonly Vector3Int Page;
            public readonly uint PreviousSurface;
            public readonly uint CurrentSurface;
            public readonly uint Retained;
            public readonly uint Lost;
            public readonly uint Added;
            public readonly uint Moving;
            public readonly uint SevereMoving;
            public readonly float MaxMillimeters;
            public readonly Vector3Int MaxVoxel;
            public readonly Vector3 MaxLocalMeters;

            public SpatialSample(float elapsed, Vector3Int page, uint previousSurface,
                uint currentSurface, uint retained, uint lost, uint added, uint moving,
                uint severeMoving, float maxMillimeters, Vector3Int maxVoxel,
                Vector3 maxLocalMeters)
            {
                Elapsed = elapsed;
                Page = page;
                PreviousSurface = previousSurface;
                CurrentSurface = currentSurface;
                Retained = retained;
                Lost = lost;
                Added = added;
                Moving = moving;
                SevereMoving = severeMoving;
                MaxMillimeters = maxMillimeters;
                MaxVoxel = maxVoxel;
                MaxLocalMeters = maxLocalMeters;
            }

            public ulong ActivityScore => (ulong)SevereMoving * 16ul +
                (ulong)(Lost + Added) * 4ul + Moving;
        }

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
        private static readonly int SpatialChunkCountId = Shader.PropertyToID("_SpatialChunkCount");
        private static readonly int SpatialChunkSizeId = Shader.PropertyToID("_SpatialChunkSize");
        private static readonly int SpatialBaseId = Shader.PropertyToID("_SpatialBase");

        private readonly List<Sample> _samples = new List<Sample>(1024);
        private readonly List<SpatialSample> _spatialSamples = new List<SpatialSample>(8192);
        private readonly List<SpatialSample> _lastSpatialSamples = new List<SpatialSample>(256);
        private uint[] _zeros;
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
        private Vector3Int _spatialChunkCount;
        private int _totalStatsCount;

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
            var voxels = _volume != null ? _volume.VoxelCount : default;
            _spatialChunkCount = new Vector3Int(
                Mathf.Max(1, Mathf.CeilToInt(voxels.x / (float)SpatialChunkSize)),
                Mathf.Max(1, Mathf.CeilToInt(voxels.y / (float)SpatialChunkSize)),
                Mathf.Max(1, Mathf.CeilToInt(voxels.z / (float)SpatialChunkSize)));
            int spatialPages = _spatialChunkCount.x * _spatialChunkCount.y * _spatialChunkCount.z;
            _totalStatsCount = StatsCount + spatialPages * SpatialStride;
            _zeros = new uint[_totalStatsCount];
            _stats = new ComputeBuffer(_totalStatsCount, sizeof(uint));
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
            _spatialSamples.Clear();
            _lastSpatialSamples.Clear();
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
                if (data.Length < _totalStatsCount) return;
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
            float elapsed = Time.realtimeSinceStartup - _sessionStarted;
            var sample = new Sample(
                elapsed,
                previousSurface, currentSurface, retained, lost, added, moving,
                survival, churn, p50, p95, maxMm);
            _last = sample;
            _samples.Add(sample);
            ApplySpatialSample(data, elapsed);
        }

        private void ApplySpatialSample(Unity.Collections.NativeArray<uint> data, float elapsed)
        {
            _lastSpatialSamples.Clear();
            int pageCount = _spatialChunkCount.x * _spatialChunkCount.y * _spatialChunkCount.z;
            var volumeCount = _volume.VoxelCount;
            float voxelSize = _volume.VoxelSize;
            for (int pageIndex = 0; pageIndex < pageCount; pageIndex++)
            {
                int baseIndex = StatsCount + pageIndex * SpatialStride;
                uint lost = data[baseIndex + 3];
                uint added = data[baseIndex + 4];
                uint moving = data[baseIndex + 5];
                uint severe = data[baseIndex + 6];
                if (lost == 0u && added == 0u && moving == 0u && severe == 0u)
                    continue;

                int pageX = pageIndex % _spatialChunkCount.x;
                int pageY = (pageIndex / _spatialChunkCount.x) % _spatialChunkCount.y;
                int pageZ = pageIndex / (_spatialChunkCount.x * _spatialChunkCount.y);
                uint packed = data[baseIndex + 7];
                uint driftFixed = packed >> 15;
                int localFlat = (int)(packed & 0x7FFFu);
                int localX = localFlat % SpatialChunkSize;
                int localY = (localFlat / SpatialChunkSize) % SpatialChunkSize;
                int localZ = localFlat / (SpatialChunkSize * SpatialChunkSize);
                var maxVoxel = new Vector3Int(
                    Mathf.Min(pageX * SpatialChunkSize + localX, volumeCount.x - 1),
                    Mathf.Min(pageY * SpatialChunkSize + localY, volumeCount.y - 1),
                    Mathf.Min(pageZ * SpatialChunkSize + localZ, volumeCount.z - 1));
                var maxLocal = new Vector3(
                    (maxVoxel.x + 0.5f - volumeCount.x * 0.5f) * voxelSize,
                    (maxVoxel.y + 0.5f - volumeCount.y * 0.5f) * voxelSize,
                    (maxVoxel.z + 0.5f - volumeCount.z * 0.5f) * voxelSize);
                var row = new SpatialSample(
                    elapsed, new Vector3Int(pageX, pageY, pageZ),
                    data[baseIndex], data[baseIndex + 1], data[baseIndex + 2],
                    lost, added, moving, severe,
                    driftFixed / 1024f * voxelSize * 1000f, maxVoxel, maxLocal);
                _lastSpatialSamples.Add(row);
                _spatialSamples.Add(row);
            }
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
            sb.AppendLine($"空间定位: 32³页={_spatialChunkCount.x}×{_spatialChunkCount.y}×{_spatialChunkCount.z} " +
                          $"末次活跃页={_lastSpatialSamples.Count} 累计空间行={_spatialSamples.Count}；每页保留最强位移体素，不保留逐体素总档案。");
            if (_lastSpatialSamples.Count > 0)
            {
                var hottest = new List<SpatialSample>(_lastSpatialSamples);
                hottest.Sort((a, b) => b.ActivityScore.CompareTo(a.ActivityScore));
                int count = Mathf.Min(8, hottest.Count);
                sb.AppendLine("末次热点32³页:");
                for (int i = 0; i < count; i++)
                {
                    SpatialSample p = hottest[i];
                    sb.AppendLine($"  页={p.Page.x}/{p.Page.y}/{p.Page.z} 失/新/动/剧={p.Lost}/{p.Added}/{p.Moving}/{p.SevereMoving} " +
                                  $"最大={p.MaxMillimeters:F2}mm 体素={p.MaxVoxel.x}/{p.MaxVoxel.y}/{p.MaxVoxel.z} " +
                                  $"局部米={p.MaxLocalMeters.x:F3}/{p.MaxLocalMeters.y:F3}/{p.MaxLocalMeters.z:F3}");
                }
            }
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

        public void AppendSpatialCsv(StringBuilder sb, string sessionId)
        {
            if (sb == null) return;
            sb.AppendLine("session_id,elapsed_s,page_x,page_y,page_z,previous_surface,current_surface,retained,lost,added,moving_ge_quarter_voxel,moving_ge_half_voxel,max_mm,max_voxel_x,max_voxel_y,max_voxel_z,max_local_x_m,max_local_y_m,max_local_z_m");
            for (int i = 0; i < _spatialSamples.Count; i++)
            {
                SpatialSample s = _spatialSamples[i];
                sb.Append(sessionId).Append(',')
                  .Append(s.Elapsed.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.Page.x).Append(',').Append(s.Page.y).Append(',').Append(s.Page.z).Append(',')
                  .Append(s.PreviousSurface).Append(',').Append(s.CurrentSurface).Append(',')
                  .Append(s.Retained).Append(',').Append(s.Lost).Append(',').Append(s.Added).Append(',')
                  .Append(s.Moving).Append(',').Append(s.SevereMoving).Append(',')
                  .Append(s.MaxMillimeters.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.MaxVoxel.x).Append(',').Append(s.MaxVoxel.y).Append(',').Append(s.MaxVoxel.z).Append(',')
                  .Append(s.MaxLocalMeters.x.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.MaxLocalMeters.y.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
                  .Append(s.MaxLocalMeters.z.ToString("F4", CultureInfo.InvariantCulture)).AppendLine();
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
            _compute.SetInts(SpatialChunkCountId,
                _spatialChunkCount.x, _spatialChunkCount.y, _spatialChunkCount.z);
            _compute.SetInt(SpatialChunkSizeId, SpatialChunkSize);
            _compute.SetInt(SpatialBaseId, StatsCount);
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

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Read-only join between paper-cell outcomes and the GunGel candidate court.
    /// It runs after both independent GPU snapshots have reached the replay package;
    /// no value produced here is consumed by fusion, paper adjudication or rendering.
    /// </summary>
    internal static class PaperEvidenceLineageBuilder
    {
        private const float DefaultPaperCellSide = 0.10f;
        private const float DefaultGunGelCellSide = 0.06f;

        private interface IPositioned
        {
            float X { get; }
            float Y { get; }
            float Z { get; }
        }

        private sealed class Candidate : IPositioned
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
            public string Index;
            public string StableId;
            public string State;
            public string Observations;
            public string LastSeen;
            public string SigmaMm;
            public string Nx;
            public string Ny;
            public string Nz;
            public string Support;
            public string Dual;
            public string Opposition;
            public string LastChallenge;
        }

        private sealed class Retirement : IPositioned
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
            public string StableId;
            public string Reason;
            public string RetiredFrame;
            public string LastSeen;
            public string LastChallenge;
            public string SigmaMm;
            public string Observations;
            public string WasStable;
            public string Dual;
            public string Opposition;
        }

        private sealed class Wave : IPositioned
        {
            // Spatial ownership is the stable target, not the displaced observation.
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
            public float ObservationX;
            public float ObservationY;
            public float ObservationZ;
            public string StableId;
            public string SourceFrame;
            public string PlatformFrame;
            public string PixelX;
            public string PixelY;
            public string ViewRadius;
            public string Angular;
            public string Linear;
            public string MotionQuality;
            public string EdgeBits;
            public string EdgeLabels;
            public string TemporalReason;
            public string TemporalLabel;
            public string Verdict;
            public string EvidenceFlags;
            public string RawAvailable;
            public string DualAgree;
            public string StableMatch;
            public string StableDual;
            public string Unopposed;
            public string RawX;
            public string RawY;
            public string RawZ;
            public string RawDeltaMm;
            public string ResidualMm;
            public string CombinedSigmaMm;
            public string SeverityMm;
        }

        private readonly struct GridKey : IEquatable<GridKey>
        {
            public readonly int X;
            public readonly int Y;
            public readonly int Z;

            public GridKey(int x, int y, int z)
            {
                X = x;
                Y = y;
                Z = z;
            }

            public bool Equals(GridKey other) => X == other.X && Y == other.Y && Z == other.Z;
            public override bool Equals(object obj) => obj is GridKey other && Equals(other);
            public override int GetHashCode()
            {
                unchecked { return ((X * 397) ^ Y) * 397 ^ Z; }
            }
        }

        private sealed class SpatialIndex<T> where T : class, IPositioned
        {
            private readonly float _bucket;
            private readonly Dictionary<GridKey, List<T>> _cells = new();

            public SpatialIndex(float bucket)
            {
                _bucket = Math.Max(bucket, 0.02f);
            }

            public void Add(T item)
            {
                GridKey key = Key(item.X, item.Y, item.Z);
                if (!_cells.TryGetValue(key, out List<T> list))
                {
                    list = new List<T>();
                    _cells.Add(key, list);
                }
                list.Add(item);
            }

            public T Nearest(float x, float y, float z, float radius,
                out int localCount, out float nearestDistance)
            {
                localCount = 0;
                nearestDistance = float.PositiveInfinity;
                T nearest = null;
                int minX = Floor((x - radius) / _bucket);
                int maxX = Floor((x + radius) / _bucket);
                int minY = Floor((y - radius) / _bucket);
                int maxY = Floor((y + radius) / _bucket);
                int minZ = Floor((z - radius) / _bucket);
                int maxZ = Floor((z + radius) / _bucket);
                float radiusSq = radius * radius;
                for (int gz = minZ; gz <= maxZ; gz++)
                for (int gy = minY; gy <= maxY; gy++)
                for (int gx = minX; gx <= maxX; gx++)
                {
                    if (!_cells.TryGetValue(new GridKey(gx, gy, gz), out List<T> list))
                        continue;
                    foreach (T item in list)
                    {
                        float dx = item.X - x;
                        float dy = item.Y - y;
                        float dz = item.Z - z;
                        float distanceSq = dx * dx + dy * dy + dz * dz;
                        if (distanceSq > radiusSq) continue;
                        localCount++;
                        if (distanceSq >= nearestDistance * nearestDistance) continue;
                        nearestDistance = (float)Math.Sqrt(distanceSq);
                        nearest = item;
                    }
                }
                return nearest;
            }

            private GridKey Key(float x, float y, float z) =>
                new(Floor(x / _bucket), Floor(y / _bucket), Floor(z / _bucket));

            private static int Floor(float value) => (int)Math.Floor(value);
        }

        private sealed class CsvHeader
        {
            private readonly Dictionary<string, int> _indices =
                new(StringComparer.OrdinalIgnoreCase);

            public CsvHeader(string line)
            {
                string[] names = SplitCsv(line);
                for (int i = 0; i < names.Length; i++)
                    _indices[names[i]] = i;
            }

            public string Get(string[] row, string name)
            {
                return _indices.TryGetValue(name, out int index) && index < row.Length
                    ? row[index] : string.Empty;
            }
        }

        internal static string Build(string paperDirectory, string gunGelDirectory,
            string outputDirectory)
        {
            string paperCsv = Path.Combine(paperDirectory, "paper_hole_cells.csv");
            string paperSummary = Path.Combine(paperDirectory, "paper_hole_summary.json");
            string candidateCsv = Path.Combine(gunGelDirectory, "candidates.csv");
            string retirementCsv = Path.Combine(gunGelDirectory, "retirements.csv");
            string waveCsv = Path.Combine(gunGelDirectory, "court_waves.csv");
            string gunGelSummary = Path.Combine(gunGelDirectory, "candidate_summary.json");
            RequireFile(paperCsv);
            RequireFile(candidateCsv);
            RequireFile(retirementCsv);
            RequireFile(waveCsv);

            float voxelSize = JsonNumber(paperSummary, "voxelSizeM", 0.05f);
            float stride = JsonNumber(paperSummary, "supportStride", 2f);
            float paperCellSide = Math.Max(voxelSize * stride, DefaultPaperCellSide);
            float gunGelCellSide = JsonNumber(gunGelSummary, "cell_size_m",
                DefaultGunGelCellSide);
            // The table reports the exact distance. The radius is only a bounded
            // association search, never a production acceptance threshold.
            float linkRadius = Math.Max(0.20f, paperCellSide * 1.25f + gunGelCellSide);

            var stable = new SpatialIndex<Candidate>(linkRadius);
            var embryo = new SpatialIndex<Candidate>(linkRadius);
            var retired = new SpatialIndex<Retirement>(linkRadius);
            var waves = new SpatialIndex<Wave>(linkRadius);
            var wavesByStableId = new Dictionary<string, Wave>(StringComparer.Ordinal);
            LoadCandidates(candidateCsv, stable, embryo);
            LoadRetirements(retirementCsv, retired);
            LoadWaves(waveCsv, waves, wavesByStableId);

            Directory.CreateDirectory(outputDirectory);
            string lineageCsv = Path.Combine(outputDirectory, "paper_evidence_lineage.csv");
            string summaryJson = Path.Combine(outputDirectory, "evidence_lineage_summary.json");
            var causes = new SortedDictionary<string, long>(StringComparer.Ordinal);
            var classes = new SortedDictionary<string, long>(StringComparer.Ordinal);
            var resolutions = new SortedDictionary<string, long>(StringComparer.Ordinal);
            var cross = new SortedDictionary<string, SortedDictionary<string, long>>(StringComparer.Ordinal);
            long rows = 0;
            long directRows = 0;
            long directFailures = 0;
            long directWithStable = 0;
            long failureWithStable = 0;
            long failureWithRetired = 0;
            long failureWithWave = 0;
            long failureWithoutGunGel = 0;
            long frameReplayRequired = 0;

            using (var reader = new StreamReader(paperCsv, Encoding.UTF8, true, 1 << 20))
            using (var writer = new StreamWriter(lineageCsv, false,
                new UTF8Encoding(false), 1 << 20))
            {
                string sourceHeaderLine = reader.ReadLine() ??
                    throw new InvalidDataException("paper_hole_cells.csv has no header");
                var header = new CsvHeader(sourceHeaderLine);
                writer.Write(sourceHeaderLine);
                writer.WriteLine(",lineage_class,lineage_is_inference,lineage_resolution," +
                    "fusion_replay_required,fusion_replay_manifest,link_radius_m," +
                    "active_stable_local_count,active_stable_distance_m,active_stable_id," +
                    "active_stable_candidate_index,active_stable_observations,active_stable_last_seen_frame," +
                    "active_stable_center_x_m,active_stable_center_y_m,active_stable_center_z_m," +
                    "active_stable_sigma_mm,active_stable_normal_x,active_stable_normal_y,active_stable_normal_z," +
                    "active_stable_effective_support,active_stable_dual_support,active_stable_opposition_votes," +
                    "active_stable_last_challenge_frame,active_embryo_local_count,active_embryo_distance_m," +
                    "active_embryo_observations,active_embryo_dual_support,active_embryo_opposition_votes," +
                    "retirement_local_count,retirement_distance_m,retired_stable_id,retirement_reason," +
                    "retired_frame,retired_last_seen_frame,retired_last_challenge_frame,retired_was_stable," +
                    "retired_dual_support,retired_opposition_votes,wave_link_source,wave_target_local_count," +
                    "wave_target_distance_m,wave_observation_distance_m,wave_stable_id,wave_source_frame," +
                    "wave_platform_frame,wave_pixel_x,wave_pixel_y,wave_adjudication,wave_evidence_flags," +
                    "wave_raw_available,wave_dual_agree,wave_stable_match,wave_stable_dual_mature,wave_unopposed," +
                    "wave_normal_residual_mm,wave_combined_sigma_mm,wave_severity_mm,wave_view_radius," +
                    "wave_angular_deg_s,wave_linear_m_s,wave_motion_quality,wave_edge_reason_bits," +
                    "wave_edge_reason_labels,wave_temporal_reason,wave_temporal_reason_label," +
                    "wave_raw_x_m,wave_raw_y_m,wave_raw_z_m,wave_raw_processed_delta_mm," +
                    "wave_observation_x_m,wave_observation_y_m,wave_observation_z_m," +
                    "wave_candidate_x_m,wave_candidate_y_m,wave_candidate_z_m");

                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] row = SplitCsv(line);
                    float x = ParseFloat(header.Get(row, "world_x_m"));
                    float y = ParseFloat(header.Get(row, "world_y_m"));
                    float z = ParseFloat(header.Get(row, "world_z_m"));
                    string cause = header.Get(row, "cause");
                    bool halo = header.Get(row, "empty_surface_halo") == "1";
                    bool published = header.Get(row, "final_published") == "1";

                    Candidate nearestStable = stable.Nearest(x, y, z, linkRadius,
                        out int stableCount, out float stableDistance);
                    Candidate nearestEmbryo = embryo.Nearest(x, y, z, linkRadius,
                        out int embryoCount, out float embryoDistance);
                    Retirement nearestRetired = retired.Nearest(x, y, z, linkRadius,
                        out int retirementCount, out float retirementDistance);
                    bool nearestWasStable = nearestRetired != null && nearestRetired.WasStable == "1";
                    Wave nearestWave = waves.Nearest(x, y, z, linkRadius,
                        out int waveCount, out float waveTargetDistance);

                    Wave linkedWave = null;
                    string waveSource = "none";
                    if (nearestStable != null && !string.IsNullOrEmpty(nearestStable.StableId) &&
                        wavesByStableId.TryGetValue(nearestStable.StableId, out linkedWave))
                        waveSource = "active_stable_id";
                    else if (nearestWasStable && !string.IsNullOrEmpty(nearestRetired.StableId) &&
                        wavesByStableId.TryGetValue(nearestRetired.StableId, out linkedWave))
                        waveSource = "retired_stable_id";
                    else if (nearestWave != null)
                    {
                        linkedWave = nearestWave;
                        waveSource = "nearest_wave_target";
                    }

                    string lineageClass = Classify(halo, published, cause,
                        nearestStable != null, nearestWasStable, linkedWave);
                    string lineageResolution = ResolveLineage(halo, published, cause,
                        nearestStable != null, nearestWasStable, linkedWave != null);
                    bool replayRequired = lineageResolution == "requires_frame_replay";
                    Increment(causes, cause);
                    Increment(classes, lineageClass);
                    Increment(resolutions, lineageResolution);
                    if (replayRequired) frameReplayRequired++;
                    if (!cross.TryGetValue(cause, out SortedDictionary<string, long> causeCross))
                    {
                        causeCross = new SortedDictionary<string, long>(StringComparer.Ordinal);
                        cross.Add(cause, causeCross);
                    }
                    Increment(causeCross, lineageClass);
                    rows++;
                    bool failure = !halo && !published;
                    if (!halo)
                    {
                        directRows++;
                        if (nearestStable != null) directWithStable++;
                    }
                    if (failure)
                    {
                        directFailures++;
                        if (nearestStable != null) failureWithStable++;
                        if (nearestRetired != null) failureWithRetired++;
                        if (linkedWave != null) failureWithWave++;
                        if (nearestStable == null && nearestRetired == null && linkedWave == null)
                            failureWithoutGunGel++;
                    }

                    writer.Write(line);
                    WriteField(writer, lineageClass);
                    WriteField(writer, "1");
                    WriteField(writer, lineageResolution);
                    WriteField(writer, replayRequired ? "1" : "0");
                    WriteField(writer, "../../fusion_inputs/manifest.csv");
                    WriteField(writer, F(linkRadius));
                    WriteField(writer, stableCount.ToString(CultureInfo.InvariantCulture));
                    WriteField(writer, Distance(stableDistance));
                    WriteCandidate(writer, nearestStable);
                    WriteField(writer, embryoCount.ToString(CultureInfo.InvariantCulture));
                    WriteField(writer, Distance(embryoDistance));
                    WriteEmbryo(writer, nearestEmbryo);
                    WriteField(writer, retirementCount.ToString(CultureInfo.InvariantCulture));
                    WriteField(writer, Distance(retirementDistance));
                    WriteRetirement(writer, nearestRetired);
                    WriteField(writer, waveSource);
                    WriteField(writer, waveCount.ToString(CultureInfo.InvariantCulture));
                    WriteField(writer, Distance(linkedWave == nearestWave
                        ? waveTargetDistance
                        : linkedWave == null ? float.PositiveInfinity
                        : Distance3(x, y, z, linkedWave.X, linkedWave.Y, linkedWave.Z)));
                    WriteField(writer, linkedWave == null ? string.Empty : F(Distance3(
                        x, y, z, linkedWave.ObservationX, linkedWave.ObservationY,
                        linkedWave.ObservationZ)));
                    WriteWave(writer, linkedWave);
                    writer.WriteLine();
                }
            }

            WriteSummary(summaryJson, linkRadius, paperCellSide, gunGelCellSide,
                rows, directRows, directFailures, directWithStable, failureWithStable,
                failureWithRetired, failureWithWave, failureWithoutGunGel,
                frameReplayRequired, causes, classes, resolutions, cross);
            return outputDirectory;
        }

        private static void LoadCandidates(string path, SpatialIndex<Candidate> stable,
            SpatialIndex<Candidate> embryo)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            var header = new CsvHeader(reader.ReadLine() ?? string.Empty);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] row = SplitCsv(line);
                var item = new Candidate
                {
                    Index = header.Get(row, "candidate_index"),
                    StableId = header.Get(row, "stable_id"),
                    State = header.Get(row, "state"),
                    Observations = header.Get(row, "observation_count"),
                    LastSeen = header.Get(row, "last_seen_frame"),
                    X = ParseFloat(header.Get(row, "center_x_m")),
                    Y = ParseFloat(header.Get(row, "center_y_m")),
                    Z = ParseFloat(header.Get(row, "center_z_m")),
                    SigmaMm = header.Get(row, "sigma_mm"),
                    Nx = header.Get(row, "normal_x"),
                    Ny = header.Get(row, "normal_y"),
                    Nz = header.Get(row, "normal_z"),
                    Support = header.Get(row, "effective_support"),
                    Dual = header.Get(row, "dual_agree_support"),
                    Opposition = header.Get(row, "opposition_votes"),
                    LastChallenge = header.Get(row, "last_challenge_frame")
                };
                if (string.Equals(item.State, "stable", StringComparison.OrdinalIgnoreCase))
                    stable.Add(item);
                else
                    embryo.Add(item);
            }
        }

        private static void LoadRetirements(string path, SpatialIndex<Retirement> index)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            var header = new CsvHeader(reader.ReadLine() ?? string.Empty);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] row = SplitCsv(line);
                index.Add(new Retirement
                {
                    StableId = header.Get(row, "stable_id"),
                    Reason = header.Get(row, "reason"),
                    RetiredFrame = header.Get(row, "retired_frame"),
                    LastSeen = header.Get(row, "last_seen_frame"),
                    LastChallenge = header.Get(row, "last_challenge_frame"),
                    X = ParseFloat(header.Get(row, "center_x_m")),
                    Y = ParseFloat(header.Get(row, "center_y_m")),
                    Z = ParseFloat(header.Get(row, "center_z_m")),
                    SigmaMm = header.Get(row, "sigma_mm"),
                    Observations = header.Get(row, "observation_count"),
                    WasStable = header.Get(row, "was_stable"),
                    Dual = header.Get(row, "dual_agree_support"),
                    Opposition = header.Get(row, "opposition_votes")
                });
            }
        }

        private static void LoadWaves(string path, SpatialIndex<Wave> index,
            Dictionary<string, Wave> byStableId)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            var header = new CsvHeader(reader.ReadLine() ?? string.Empty);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] row = SplitCsv(line);
                var item = new Wave
                {
                    StableId = header.Get(row, "stable_id"),
                    SourceFrame = header.Get(row, "source_frame"),
                    PlatformFrame = header.Get(row, "platform_frame"),
                    PixelX = header.Get(row, "pixel_x"),
                    PixelY = header.Get(row, "pixel_y"),
                    ViewRadius = header.Get(row, "view_radius"),
                    Angular = header.Get(row, "angular_deg_s"),
                    Linear = header.Get(row, "linear_m_s"),
                    MotionQuality = header.Get(row, "motion_quality"),
                    EdgeBits = header.Get(row, "edge_reason_bits"),
                    EdgeLabels = header.Get(row, "edge_reason_labels"),
                    TemporalReason = header.Get(row, "temporal_reason"),
                    TemporalLabel = header.Get(row, "temporal_reason_label"),
                    Verdict = header.Get(row, "adjudication"),
                    EvidenceFlags = header.Get(row, "evidence_flags"),
                    RawAvailable = header.Get(row, "raw_available"),
                    DualAgree = header.Get(row, "dual_agree"),
                    StableMatch = header.Get(row, "stable_match"),
                    StableDual = header.Get(row, "stable_dual_mature"),
                    Unopposed = header.Get(row, "unopposed"),
                    RawX = header.Get(row, "raw_x_m"),
                    RawY = header.Get(row, "raw_y_m"),
                    RawZ = header.Get(row, "raw_z_m"),
                    RawDeltaMm = header.Get(row, "raw_processed_delta_mm"),
                    ObservationX = ParseFloat(header.Get(row, "observation_x_m")),
                    ObservationY = ParseFloat(header.Get(row, "observation_y_m")),
                    ObservationZ = ParseFloat(header.Get(row, "observation_z_m")),
                    X = ParseFloat(header.Get(row, "candidate_x_m")),
                    Y = ParseFloat(header.Get(row, "candidate_y_m")),
                    Z = ParseFloat(header.Get(row, "candidate_z_m")),
                    ResidualMm = header.Get(row, "normal_residual_mm"),
                    CombinedSigmaMm = header.Get(row, "combined_sigma_mm"),
                    SeverityMm = header.Get(row, "severity_mm")
                };
                index.Add(item);
                if (!string.IsNullOrEmpty(item.StableId))
                    byStableId[item.StableId] = item;
            }
        }

        private static string Classify(bool halo, bool published, string cause,
            bool activeStable, bool retiredStable, Wave wave)
        {
            if (halo) return "empty_halo_not_direct_surface";
            if (published) return "covered_control";
            if (cause == "raw_not_mature") return "paper_maturity_after_tsdf_crossing";
            if (cause == "face_parity_open" || cause == "closed_graph_failed" ||
                cause == "fallback_rejected")
                return "paper_topology_after_mature_crossing";
            if (activeStable && wave != null) return "local_stable_with_latched_wave";
            if (activeStable) return "tsdf_crossing_missing_despite_live_stable_candidate";
            if (retiredStable && wave != null) return "retired_candidate_with_latched_wave";
            if (retiredStable) return "local_candidate_retired_before_paper_snapshot";
            if (wave != null) return "latched_wave_without_live_candidate";
            return "no_local_gungel_candidate_or_wave";
        }

        private static string ResolveLineage(bool halo, bool published, string cause,
            bool activeStable, bool retiredStable, bool hasWave)
        {
            if (halo || published || cause == "raw_not_mature" ||
                cause == "face_parity_open" || cause == "closed_graph_failed" ||
                cause == "fallback_rejected")
                return "resolved_to_paper_snapshot";
            if (activeStable || retiredStable || hasWave)
                return "resolved_to_retained_gungel_snapshot";
            return "requires_frame_replay";
        }

        private static void WriteCandidate(StreamWriter writer, Candidate item)
        {
            string[] values = item == null ? new string[15] : new[]
            {
                item.StableId, item.Index, item.Observations, item.LastSeen,
                F(item.X), F(item.Y), F(item.Z), item.SigmaMm,
                item.Nx, item.Ny, item.Nz, item.Support, item.Dual,
                item.Opposition, item.LastChallenge
            };
            // Two columns were already emitted before this helper; the helper
            // owns the remaining 15 candidate columns.
            for (int i = 0; i < 15; i++) WriteField(writer, values[i]);
        }

        private static void WriteEmbryo(StreamWriter writer, Candidate item)
        {
            WriteField(writer, item?.Observations ?? string.Empty);
            WriteField(writer, item?.Dual ?? string.Empty);
            WriteField(writer, item?.Opposition ?? string.Empty);
        }

        private static void WriteRetirement(StreamWriter writer, Retirement item)
        {
            string[] values = item == null ? new string[8] : new[]
            {
                item.StableId, item.Reason, item.RetiredFrame, item.LastSeen,
                item.LastChallenge, item.WasStable, item.Dual, item.Opposition
            };
            for (int i = 0; i < 8; i++) WriteField(writer, values[i]);
        }

        private static void WriteWave(StreamWriter writer, Wave item)
        {
            string[] values = item == null ? new string[33] : new[]
            {
                item.StableId, item.SourceFrame, item.PlatformFrame, item.PixelX, item.PixelY,
                item.Verdict, item.EvidenceFlags, item.RawAvailable, item.DualAgree,
                item.StableMatch, item.StableDual, item.Unopposed, item.ResidualMm,
                item.CombinedSigmaMm, item.SeverityMm, item.ViewRadius, item.Angular,
                item.Linear, item.MotionQuality, item.EdgeBits, item.EdgeLabels,
                item.TemporalReason, item.TemporalLabel, item.RawX, item.RawY, item.RawZ,
                item.RawDeltaMm, F(item.ObservationX), F(item.ObservationY),
                F(item.ObservationZ), F(item.X), F(item.Y), F(item.Z)
            };
            for (int i = 0; i < 33; i++) WriteField(writer, values[i]);
        }

        private static void WriteSummary(string path, float linkRadius,
            float paperCellSide, float gunGelCellSide, long rows, long directRows,
            long directFailures, long directWithStable, long failureWithStable,
            long failureWithRetired, long failureWithWave, long failureWithoutGunGel,
            long frameReplayRequired, SortedDictionary<string, long> causes,
            SortedDictionary<string, long> classes,
            SortedDictionary<string, long> resolutions,
            SortedDictionary<string, SortedDictionary<string, long>> cross)
        {
            var json = new StringBuilder(8192);
            json.Append("{\n  \"schema\": \"scancover.paper_evidence_lineage.v1\",\n")
                .Append("  \"capturedUtc\": \"").Append(DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)).Append("\",\n")
                .Append("  \"readOnly\": true,\n")
                .Append("  \"classificationIsInference\": true,\n")
                .Append("  \"paperCellSideM\": ").Append(F(paperCellSide)).Append(",\n")
                .Append("  \"gunGelCellSideM\": ").Append(F(gunGelCellSide)).Append(",\n")
                .Append("  \"spatialLinkRadiusM\": ").Append(F(linkRadius)).Append(",\n")
                .Append("  \"semantics\": {\n")
                .Append("    \"join\": \"paper world-cell center to bounded nearby active candidates, retirements and wave targets; stable_id takes precedence for wave linkage\",\n")
                .Append("    \"negativeEvidence\": \"absence of a local row means no retained evidence in these bounded snapshots, not proof that no observation ever existed\",\n")
                .Append("    \"lineageResolution\": \"paper snapshot means the break is already localized after TSDF; retained GunGel snapshot means a nearby current or retired witness exists; requires frame replay means snapshot evidence cannot establish the earlier history\",\n")
                .Append("    \"fusionReplayManifest\": \"../../fusion_inputs/manifest.csv\",\n")
                .Append("    \"productionEffect\": \"none\"\n  },\n")
                .Append("  \"rows\": ").Append(rows).Append(",\n")
                .Append("  \"directSurfaceRows\": ").Append(directRows).Append(",\n")
                .Append("  \"directFailureRows\": ").Append(directFailures).Append(",\n")
                .Append("  \"directRowsWithLiveStableCandidate\": ").Append(directWithStable).Append(",\n")
                .Append("  \"failuresWithLiveStableCandidate\": ").Append(failureWithStable).Append(",\n")
                .Append("  \"failuresWithNearbyRetirement\": ").Append(failureWithRetired).Append(",\n")
                .Append("  \"failuresWithLinkedWave\": ").Append(failureWithWave).Append(",\n")
                .Append("  \"failuresWithoutRetainedLocalGunGelEvidence\": ").Append(failureWithoutGunGel).Append(",\n")
                .Append("  \"rowsRequiringFrameReplay\": ").Append(frameReplayRequired).Append(",\n")
                .Append("  \"causeCounts\": ");
            AppendCounts(json, causes, "  ");
            json.Append(",\n  \"lineageClassCounts\": ");
            AppendCounts(json, classes, "  ");
            json.Append(",\n  \"lineageResolutionCounts\": ");
            AppendCounts(json, resolutions, "  ");
            json.Append(",\n  \"causeByLineageClass\": {");
            bool firstCause = true;
            foreach (var cause in cross)
            {
                if (!firstCause) json.Append(',');
                json.Append("\n    \"").Append(Json(cause.Key)).Append("\": ");
                AppendCounts(json, cause.Value, "    ");
                firstCause = false;
            }
            json.Append("\n  }\n}\n");
            File.WriteAllText(path, json.ToString(), new UTF8Encoding(false));
        }

        private static void AppendCounts(StringBuilder json,
            SortedDictionary<string, long> counts, string indent)
        {
            json.Append('{');
            bool first = true;
            foreach (var pair in counts)
            {
                if (!first) json.Append(',');
                json.Append("\n").Append(indent).Append("  \"")
                    .Append(Json(pair.Key)).Append("\": ").Append(pair.Value);
                first = false;
            }
            json.Append("\n").Append(indent).Append('}');
        }

        private static void Increment(SortedDictionary<string, long> values, string key)
        {
            key ??= string.Empty;
            values.TryGetValue(key, out long count);
            values[key] = count + 1;
        }

        private static void WriteField(StreamWriter writer, string value)
        {
            writer.Write(',');
            writer.Write(Csv(value));
        }

        private static string Csv(string value)
        {
            value ??= string.Empty;
            if (value.IndexOfAny(new[] { ',', '"', '\r', '\n' }) < 0) return value;
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static string[] SplitCsv(string line)
        {
            var values = new List<string>();
            var value = new StringBuilder();
            bool quoted = false;
            for (int i = 0; i < (line?.Length ?? 0); i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (quoted && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        value.Append('"');
                        i++;
                    }
                    else quoted = !quoted;
                }
                else if (c == ',' && !quoted)
                {
                    values.Add(value.ToString());
                    value.Clear();
                }
                else value.Append(c);
            }
            values.Add(value.ToString());
            return values.ToArray();
        }

        private static float JsonNumber(string path, string key, float fallback)
        {
            if (!File.Exists(path)) return fallback;
            string json = File.ReadAllText(path);
            int keyIndex = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (keyIndex < 0) return fallback;
            int colon = json.IndexOf(':', keyIndex);
            if (colon < 0) return fallback;
            int start = colon + 1;
            while (start < json.Length && char.IsWhiteSpace(json[start])) start++;
            int end = start;
            while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '-' ||
                   json[end] == '+' || json[end] == '.' || json[end] == 'e' || json[end] == 'E')) end++;
            return float.TryParse(json.Substring(start, end - start), NumberStyles.Float,
                CultureInfo.InvariantCulture, out float value) ? value : fallback;
        }

        private static float ParseFloat(string value)
        {
            return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture,
                out float result) ? result : 0f;
        }

        private static float Distance3(float ax, float ay, float az,
            float bx, float by, float bz)
        {
            float dx = ax - bx;
            float dy = ay - by;
            float dz = az - bz;
            return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        private static string Distance(float value) => float.IsInfinity(value) ? string.Empty : F(value);
        private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
        private static string Json(string value) => (value ?? string.Empty)
            .Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n");

        private static void RequireFile(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("lineage input missing", path);
        }
    }
}

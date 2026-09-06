using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Builds a diagnostic-only, session-local feature archive from the sealed
    /// virtual-probe ledger. Coordinates remain available for audit, but the
    /// reusable unit is the continuous surface/evidence feature vector rather
    /// than a coordinate blacklist. Nothing produced here is read by GunGel,
    /// TSDF, paper, mesh generation or rendering.
    /// </summary>
    internal static class SurfaceFeatureArchiveBuilder
    {
        internal const string Schema = "scancover.surface_feature_archive.v2";

        private readonly struct CellKey : IEquatable<CellKey>, IComparable<CellKey>
        {
            internal readonly int X;
            internal readonly int Y;
            internal readonly int Z;
            internal readonly int Axis;

            internal CellKey(int x, int y, int z, int axis)
            {
                X = x;
                Y = y;
                Z = z;
                Axis = axis;
            }

            public bool Equals(CellKey other) => X == other.X && Y == other.Y &&
                Z == other.Z && Axis == other.Axis;
            public override bool Equals(object obj) => obj is CellKey other && Equals(other);
            public override int GetHashCode()
            {
                unchecked { return (((X * 397) ^ Y) * 397 ^ Z) * 397 ^ Axis; }
            }
            public int CompareTo(CellKey other)
            {
                int value = X.CompareTo(other.X);
                if (value != 0) return value;
                value = Y.CompareTo(other.Y);
                if (value != 0) return value;
                value = Z.CompareTo(other.Z);
                return value != 0 ? value : Axis.CompareTo(other.Axis);
            }
        }

        private readonly struct RecordKey : IEquatable<RecordKey>, IComparable<RecordKey>
        {
            internal readonly CellKey Cell;
            internal readonly int FingerprintId;
            internal readonly int Generation;

            internal RecordKey(CellKey cell, int fingerprintId, int generation)
            {
                Cell = cell;
                FingerprintId = fingerprintId;
                Generation = generation;
            }

            public bool Equals(RecordKey other) => Cell.Equals(other.Cell) &&
                FingerprintId == other.FingerprintId && Generation == other.Generation;
            public override bool Equals(object obj) => obj is RecordKey other && Equals(other);
            public override int GetHashCode()
            {
                unchecked { return (Cell.GetHashCode() * 397 ^ FingerprintId) * 397 ^ Generation; }
            }
            public int CompareTo(RecordKey other)
            {
                int value = Cell.CompareTo(other.Cell);
                if (value != 0) return value;
                value = FingerprintId.CompareTo(other.FingerprintId);
                return value != 0 ? value : Generation.CompareTo(other.Generation);
            }
        }

        private sealed class Patch
        {
            internal Vector3 Center;
            internal Vector3 Normal;
            internal float RadiusMm;
            internal int Frame;
        }

        private sealed class Record
        {
            internal RecordKey Key;
            internal string FinalVerdict = "superseded";
            internal int AcceptedFrame = -1;
            internal int FirstFrame = int.MaxValue;
            internal int LastFrame = -1;
            internal Vector3 SnapshotCenter;
            internal Vector3 SnapshotNormal;
            internal bool HasSnapshot;
            internal readonly List<Patch> Patches = new List<Patch>(12);
            internal readonly List<float> SupportResidualMm = new List<float>(8);
            internal readonly List<float> SupportAngleDeg = new List<float>(8);
            internal readonly List<float> ViewRadius = new List<float>(8);
            internal readonly List<float> MotionQuality = new List<float>(8);
            internal readonly List<float> FreeGapMm = new List<float>(8);
            internal int Samples;
            internal int SupportSamples;
            internal int IndependentSupportSamples;
            internal int RecoverySamples;
            internal int ChallengeSamples;
            internal int IndependentChallengeSamples;
            internal int CorrelatedChallengeSamples;
            internal int AcceptTransitions;
            internal int RejectTransitions;
            internal int ReopenTransitions;
            internal int FingerprintHitSamples;
            internal int FingerprintEdgeHitSamples;
            internal int MaxSupportViews;
            internal int MaxChallengeVotes;
            internal int MaxChallengeViews;
            internal int MaxRecoveryViews;
            internal float MaxBaselineMm;
            internal float MaxSpreadDeg;
            internal float MaxFreeGapMm;
            internal string PostVerdictOutcome = string.Empty;
            internal bool PostVerdictWindowComplete;
            internal int PostObservedFrames;
            internal int PostContactFrames;
            internal int PostRayFrames;
            internal int PostReliableFreeFrames;
            internal int PostFreeViews;
            internal int PostSupportFrames;
            internal int PostSupportViews;
            internal bool PostReconfirmed;
            internal bool PostReopened;
            internal int PostFollowupSpanFrames;
        }

        private sealed class Sample
        {
            internal int EventRow;
            internal Record Record;
            internal string Event;
            internal string Reason;
            internal int GunGelFrame;
            internal int PlatformFrame;
            internal Vector3 Source;
            internal Vector3 Target;
            internal float ResidualMm;
            internal float AngleDeg;
            internal float GapMm;
            internal float ViewRadius;
            internal float MotionQuality;
            internal float BaselineMm;
            internal float SpreadDeg;
            internal int PatchIndex;
            internal float PatchRadiusMm;
            internal float PatchOffsetMm;
            internal string Association;
        }

        private sealed class CsvHeader
        {
            private readonly Dictionary<string, int> _indices =
                new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            internal CsvHeader(string line)
            {
                string[] names = Split(line);
                for (int i = 0; i < names.Length; i++) _indices[names[i]] = i;
            }

            internal string Get(string[] row, string name) =>
                _indices.TryGetValue(name, out int index) && index < row.Length
                    ? row[index] : string.Empty;
        }

        internal static string Build(string sessionDirectory, string outputDirectory)
        {
            if (string.IsNullOrEmpty(sessionDirectory))
                throw new ArgumentException("session directory is empty", nameof(sessionDirectory));
            string probe = Path.Combine(sessionDirectory, "probe_shadow");
            string cellsPath = Path.Combine(probe, "verdict_cells.csv");
            string eventsPath = Path.Combine(probe, "verdict_events.csv");
            string fingerprintsPath = Path.Combine(probe, "verdict_fingerprints.csv");
            string followupsPath = Path.Combine(probe, "verdict_followups.csv");
            Require(cellsPath);
            Require(eventsPath);
            Require(fingerprintsPath);
            Directory.CreateDirectory(outputDirectory);

            var records = new Dictionary<RecordKey, Record>();
            var byCell = new Dictionary<CellKey, List<Record>>();
            LoadFingerprints(fingerprintsPath, records, byCell);
            LoadCells(cellsPath, records, byCell);
            List<Sample> samples = LoadEvents(eventsPath, records, byCell);
            if (File.Exists(followupsPath))
                LoadFollowups(followupsPath, records, byCell);

            var ordered = new List<Record>(records.Values);
            ordered.Sort((a, b) => a.Key.CompareTo(b.Key));
            string archivePath = Path.Combine(outputDirectory, "surface_feature_archive.csv");
            string samplesPath = Path.Combine(outputDirectory, "surface_feature_samples.csv");
            WriteArchive(archivePath, Path.GetFileName(sessionDirectory), ordered);
            WriteSamples(samplesPath, Path.GetFileName(sessionDirectory), samples);
            WriteSchema(Path.Combine(outputDirectory, "schema.json"));
            string summaryPath = Path.Combine(outputDirectory, "surface_feature_summary.json");
            WriteSummary(summaryPath, Path.GetFileName(sessionDirectory), ordered, samples);
            File.WriteAllText(Path.Combine(outputDirectory, "README.txt"),
                "ScanCover 表面特征隔离仓\n" +
                "- 会话内坐标和指纹只用于追溯，不得跨会话充当黑名单。\n" +
                 "- 跨会话学习单位是连续特征、证据过程和裁决结果。\n" +
                 "- Reject 后续结局单列：自由空间复核、重新获证、冲突或明确未知。\n" +
                 "- 本目录零生产权限，枪胶、TSDF、纸皮和网格均不得读取。\n",
                new UTF8Encoding(false));
            return summaryPath;
        }

        private static void LoadFingerprints(string path,
            Dictionary<RecordKey, Record> records,
            Dictionary<CellKey, List<Record>> byCell)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            string headerLine = reader.ReadLine() ??
                throw new InvalidDataException("verdict_fingerprints.csv has no header");
            var header = new CsvHeader(headerLine);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] row = Split(line);
                CellKey cell = ReadCell(header, row);
                int id = Int(header.Get(row, "fingerprintId"));
                int generation = Int(header.Get(row, "fingerprintGeneration"));
                if (id <= 0 || generation <= 0) continue;
                Record record = Get(records, byCell, new RecordKey(cell, id, generation));
                record.AcceptedFrame = Int(header.Get(row, "fingerprintAcceptedFrame"), -1);
                int patchFrame = Int(header.Get(row, "patchFrame"), -1);
                record.FirstFrame = Math.Min(record.FirstFrame,
                    record.AcceptedFrame >= 0 ? record.AcceptedFrame : patchFrame);
                record.LastFrame = Math.Max(record.LastFrame, patchFrame);
                record.Patches.Add(new Patch
                {
                    Center = Vector(header, row, "centerX", "centerY", "centerZ"),
                    Normal = SafeNormal(Vector(header, row, "normalX", "normalY", "normalZ")),
                    RadiusMm = Float(header.Get(row, "radiusMm")),
                    Frame = patchFrame
                });
            }
        }

        private static void LoadCells(string path,
            Dictionary<RecordKey, Record> records,
            Dictionary<CellKey, List<Record>> byCell)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            string headerLine = reader.ReadLine() ??
                throw new InvalidDataException("verdict_cells.csv has no header");
            var header = new CsvHeader(headerLine);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] row = Split(line);
                CellKey cell = ReadCell(header, row);
                int id = Int(header.Get(row, "fingerprintId"));
                int generation = Int(header.Get(row, "fingerprintGeneration"));
                Record record = Get(records, byCell,
                    new RecordKey(cell, Math.Max(0, id), Math.Max(0, generation)));
                record.FinalVerdict = header.Get(row, "verdict");
                record.AcceptedFrame = Int(header.Get(row, "fingerprintAcceptedFrame"),
                    record.AcceptedFrame);
                record.FirstFrame = Int(header.Get(row, "firstFrame"), record.FirstFrame);
                record.LastFrame = Int(header.Get(row, "lastSeenFrame"), record.LastFrame);
                record.SnapshotCenter = Vector(header, row, "centerX", "centerY", "centerZ");
                record.SnapshotNormal = SafeNormal(Vector(header, row,
                    "normalX", "normalY", "normalZ"));
                record.HasSnapshot = true;
                record.MaxSupportViews = Int(header.Get(row, "independentSupportViews"));
                record.MaxChallengeVotes = Int(header.Get(row, "challengeVotes"));
                record.MaxChallengeViews = Int(header.Get(row, "independentChallengeViews"));
                record.MaxRecoveryViews = Int(header.Get(row, "recoverySupportViews"));
                record.MaxBaselineMm = Float(header.Get(row, "maxBaselineMm"));
                record.MaxSpreadDeg = Float(header.Get(row, "maxViewSpreadDeg"));
                record.MaxFreeGapMm = Float(header.Get(row, "maxFreeGapMm"));
            }
        }

        private static List<Sample> LoadEvents(string path,
            Dictionary<RecordKey, Record> records,
            Dictionary<CellKey, List<Record>> byCell)
        {
            var samples = new List<Sample>(16384);
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            string headerLine = reader.ReadLine() ??
                throw new InvalidDataException("verdict_events.csv has no header");
            var header = new CsvHeader(headerLine);
            int eventRow = 0;
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                eventRow++;
                string[] row = Split(line);
                CellKey cell = ReadCell(header, row);
                int frame = Int(header.Get(row, "gunGelFrame"));
                int id = Int(header.Get(row, "fingerprintId"));
                int generation = Int(header.Get(row, "fingerprintGeneration"));
                Record record = ResolveEventRecord(records, byCell, cell, id,
                    generation, frame);
                var sample = new Sample
                {
                    EventRow = eventRow,
                    Record = record,
                    Event = header.Get(row, "event"),
                    Reason = header.Get(row, "reason"),
                    GunGelFrame = frame,
                    PlatformFrame = Int(header.Get(row, "platformFrame")),
                    Source = Vector(header, row, "sourceX", "sourceY", "sourceZ"),
                    Target = Vector(header, row, "targetX", "targetY", "targetZ"),
                    ResidualMm = Float(header.Get(row, "residualMm")),
                    AngleDeg = Float(header.Get(row, "angleDeg")),
                    GapMm = Float(header.Get(row, "gapMm")),
                    ViewRadius = Float(header.Get(row, "viewRadius")),
                    MotionQuality = Float(header.Get(row, "motionQuality")),
                    BaselineMm = Float(header.Get(row, "baselineMm")),
                    SpreadDeg = Float(header.Get(row, "spreadDeg")),
                    PatchIndex = Int(header.Get(row, "fingerprintPatchIndex"), -1),
                    PatchRadiusMm = Float(header.Get(row, "footprintRadiusMm")),
                    PatchOffsetMm = Float(header.Get(row, "footprintOffsetMm")),
                    Association = header.Get(row, "association")
                };
                samples.Add(sample);
                ApplyEvent(record, sample, header, row);
            }
            return samples;
        }

        private static void LoadFollowups(string path,
            Dictionary<RecordKey, Record> records,
            Dictionary<CellKey, List<Record>> byCell)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true, 1 << 20);
            string headerLine = reader.ReadLine() ??
                throw new InvalidDataException("verdict_followups.csv has no header");
            var header = new CsvHeader(headerLine);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] row = Split(line);
                CellKey cell = ReadCell(header, row);
                int id = Int(header.Get(row, "rejectedFingerprintId"));
                int generation = Int(header.Get(row, "rejectedFingerprintGeneration"));
                if (id <= 0 || generation <= 0) continue;
                Record record = Get(records, byCell, new RecordKey(cell, id, generation));
                record.PostVerdictOutcome = header.Get(row, "finalOutcome");
                record.PostVerdictWindowComplete = Bool(header.Get(row, "windowComplete"));
                record.PostObservedFrames = Int(header.Get(row, "observedFrames"));
                record.PostContactFrames = Int(header.Get(row, "contactFrames"));
                record.PostRayFrames = Int(header.Get(row, "rayIntersectionFrames"));
                record.PostReliableFreeFrames = Int(header.Get(row, "reliableFreeSpaceFrames"));
                record.PostFreeViews = Int(header.Get(row, "independentFreeSpaceViews"));
                record.PostSupportFrames = Int(header.Get(row, "safeSupportFrames"));
                record.PostSupportViews = Int(header.Get(row, "independentSupportViews"));
                record.PostReconfirmed = Bool(header.Get(row, "freeSpaceReconfirmed"));
                record.PostReopened = Bool(header.Get(row, "reopened"));
                record.PostFollowupSpanFrames = Int(header.Get(row, "followupSpanFrames"));
            }
        }

        private static void ApplyEvent(Record record, Sample sample,
            CsvHeader header, string[] row)
        {
            record.Samples++;
            record.FirstFrame = Math.Min(record.FirstFrame, sample.GunGelFrame);
            record.LastFrame = Math.Max(record.LastFrame, sample.GunGelFrame);
            AddFinite(record.ViewRadius, sample.ViewRadius);
            AddFinite(record.MotionQuality, sample.MotionQuality);
            record.MaxBaselineMm = Math.Max(record.MaxBaselineMm, sample.BaselineMm);
            record.MaxSpreadDeg = Math.Max(record.MaxSpreadDeg, sample.SpreadDeg);
            record.MaxSupportViews = Math.Max(record.MaxSupportViews,
                Int(header.Get(row, "independentSupportViews")));
            record.MaxChallengeVotes = Math.Max(record.MaxChallengeVotes,
                Int(header.Get(row, "challengeVotes")));
            record.MaxChallengeViews = Math.Max(record.MaxChallengeViews,
                Int(header.Get(row, "independentChallengeViews")));
            record.MaxRecoveryViews = Math.Max(record.MaxRecoveryViews,
                Int(header.Get(row, "recoverySupportViews")));

            bool support = sample.Event == "support_independent" ||
                           sample.Event.StartsWith("recovery_", StringComparison.Ordinal);
            bool challenge = sample.Event.StartsWith("challenge_", StringComparison.Ordinal);
            if (support)
            {
                record.SupportSamples++;
                if (sample.Event == "support_independent") record.IndependentSupportSamples++;
                else record.RecoverySamples++;
                AddFinite(record.SupportResidualMm, sample.ResidualMm);
                AddFinite(record.SupportAngleDeg, sample.AngleDeg);
            }
            if (challenge)
            {
                record.ChallengeSamples++;
                if (sample.Event == "challenge_independent") record.IndependentChallengeSamples++;
                else record.CorrelatedChallengeSamples++;
                AddFinite(record.FreeGapMm, sample.GapMm);
                record.MaxFreeGapMm = Math.Max(record.MaxFreeGapMm, sample.GapMm);
            }
            if (header.Get(row, "previousVerdict") != "accept" &&
                header.Get(row, "newVerdict") == "accept") record.AcceptTransitions++;
            if (header.Get(row, "previousVerdict") != "reject" &&
                header.Get(row, "newVerdict") == "reject") record.RejectTransitions++;
            if (sample.Reason == "reopened_by_independent_safe_support") record.ReopenTransitions++;
            if (sample.Association == "fingerprint_patch_hit")
            {
                record.FingerprintHitSamples++;
                if (sample.PatchRadiusMm > 0f &&
                    sample.PatchOffsetMm / sample.PatchRadiusMm >= 0.95f)
                    record.FingerprintEdgeHitSamples++;
            }
        }

        private static Record ResolveEventRecord(Dictionary<RecordKey, Record> records,
            Dictionary<CellKey, List<Record>> byCell, CellKey cell,
            int id, int generation, int frame)
        {
            if (id > 0 && generation > 0)
                return Get(records, byCell, new RecordKey(cell, id, generation));
            if (byCell.TryGetValue(cell, out List<Record> candidates))
            {
                Record nearestFuture = null;
                foreach (Record candidate in candidates)
                {
                    if (candidate.Key.FingerprintId <= 0 || candidate.AcceptedFrame < frame)
                        continue;
                    if (nearestFuture == null || candidate.AcceptedFrame < nearestFuture.AcceptedFrame)
                        nearestFuture = candidate;
                }
                if (nearestFuture != null) return nearestFuture;
            }
            return Get(records, byCell, new RecordKey(cell, 0, 0));
        }

        private static Record Get(Dictionary<RecordKey, Record> records,
            Dictionary<CellKey, List<Record>> byCell, RecordKey key)
        {
            if (records.TryGetValue(key, out Record record)) return record;
            record = new Record { Key = key };
            records.Add(key, record);
            if (!byCell.TryGetValue(key.Cell, out List<Record> list))
            {
                list = new List<Record>();
                byCell.Add(key.Cell, list);
            }
            list.Add(record);
            return record;
        }

        private static void WriteArchive(string path, string sessionId,
            List<Record> records)
        {
            using var writer = new StreamWriter(path, false, new UTF8Encoding(false), 1 << 20);
            writer.WriteLine("sessionId,archiveKey,cellX,cellY,cellZ,axis,fingerprintId," +
                "fingerprintGeneration,acceptedFrame,firstFrame,lastFrame,finalVerdict,evidenceOutcome," +
                "orientationFamily,centerX,centerY,centerZ,normalX,normalY,normalZ," +
                "patchCount,patchRadiusP50Mm,patchRadiusP95Mm,patchRadiusMaxMm," +
                "patchNormalSpreadP50Deg,patchNormalSpreadP95Deg,patchNormalSpreadMaxDeg," +
                "patchThicknessMm,patchTangentialSpanMm,samples,supportSamples," +
                "independentSupportSamples,recoverySamples,challengeSamples," +
                "independentChallengeSamples,correlatedChallengeSamples,acceptTransitions," +
                "rejectTransitions,reopenTransitions,fingerprintHitSamples," +
                "fingerprintEdgeHitSamples,maxSupportViews,maxChallengeVotes," +
                "maxChallengeViews,maxRecoveryViews,supportResidualP50Mm," +
                "supportResidualP95Mm,supportResidualMaxMm,supportAngleP50Deg," +
                "supportAngleP95Deg,supportAngleMaxDeg,viewRadiusP50,viewRadiusP95," +
                 "motionQualityP05,motionQualityP50,freeGapP50Mm,freeGapP95Mm," +
                 "maxBaselineMm,maxSpreadDeg,maxFreeGapMm,postVerdictOutcome," +
                 "postVerdictWindowComplete,postVerdictObservedFrames,postVerdictContactFrames," +
                 "postVerdictRayFrames,postVerdictReliableFreeFrames,postVerdictFreeViews," +
                 "postVerdictSupportFrames,postVerdictSupportViews,postVerdictReconfirmed," +
                 "postVerdictReopened,postVerdictFollowupSpanFrames,descriptorTags,authority");
            foreach (Record record in records)
            {
                Geometry(record, out Vector3 center, out Vector3 normal,
                    out List<float> radii, out List<float> spreads,
                    out float thicknessMm, out float tangentialSpanMm);
                string outcome = Outcome(record);
                string orientation = Orientation(normal);
                string tags = Tags(record, orientation, outcome);
                writer.WriteLine(string.Join(",",
                    Csv(sessionId), Csv(Key(record.Key)), I(record.Key.Cell.X),
                    I(record.Key.Cell.Y), I(record.Key.Cell.Z), I(record.Key.Cell.Axis),
                    I(record.Key.FingerprintId), I(record.Key.Generation), I(record.AcceptedFrame),
                    I(record.FirstFrame == int.MaxValue ? -1 : record.FirstFrame), I(record.LastFrame),
                    Csv(record.FinalVerdict), Csv(outcome), Csv(orientation),
                    F(center.x), F(center.y), F(center.z), F(normal.x), F(normal.y), F(normal.z),
                    I(record.Patches.Count), F(Q(radii, 0.50f)), F(Q(radii, 0.95f)), F(Max(radii)),
                    F(Q(spreads, 0.50f)), F(Q(spreads, 0.95f)), F(Max(spreads)),
                    F(thicknessMm), F(tangentialSpanMm), I(record.Samples), I(record.SupportSamples),
                    I(record.IndependentSupportSamples), I(record.RecoverySamples),
                    I(record.ChallengeSamples), I(record.IndependentChallengeSamples),
                    I(record.CorrelatedChallengeSamples), I(record.AcceptTransitions),
                    I(record.RejectTransitions), I(record.ReopenTransitions),
                    I(record.FingerprintHitSamples), I(record.FingerprintEdgeHitSamples),
                    I(record.MaxSupportViews), I(record.MaxChallengeVotes),
                    I(record.MaxChallengeViews), I(record.MaxRecoveryViews),
                    F(Q(record.SupportResidualMm, 0.50f)), F(Q(record.SupportResidualMm, 0.95f)),
                    F(Max(record.SupportResidualMm)), F(Q(record.SupportAngleDeg, 0.50f)),
                    F(Q(record.SupportAngleDeg, 0.95f)), F(Max(record.SupportAngleDeg)),
                    F(Q(record.ViewRadius, 0.50f)), F(Q(record.ViewRadius, 0.95f)),
                    F(Q(record.MotionQuality, 0.05f)), F(Q(record.MotionQuality, 0.50f)),
                     F(Q(record.FreeGapMm, 0.50f)), F(Q(record.FreeGapMm, 0.95f)),
                     F(record.MaxBaselineMm), F(record.MaxSpreadDeg), F(record.MaxFreeGapMm),
                     Csv(record.PostVerdictOutcome), record.PostVerdictWindowComplete ? "1" : "0",
                     I(record.PostObservedFrames), I(record.PostContactFrames),
                     I(record.PostRayFrames), I(record.PostReliableFreeFrames),
                     I(record.PostFreeViews), I(record.PostSupportFrames),
                     I(record.PostSupportViews), record.PostReconfirmed ? "1" : "0",
                     record.PostReopened ? "1" : "0", I(record.PostFollowupSpanFrames),
                     Csv(tags), "diagnostic_only"));
            }
        }

        private static void WriteSamples(string path, string sessionId,
            List<Sample> samples)
        {
            using var writer = new StreamWriter(path, false, new UTF8Encoding(false), 1 << 20);
            writer.WriteLine("sessionId,eventRow,archiveKey,gunGelFrame,platformFrame,event,reason," +
                "sourceX,sourceY,sourceZ,targetX,targetY,targetZ,residualMm,angleDeg,gapMm," +
                "viewRadius,motionQuality,baselineMm,spreadDeg,fingerprintPatchIndex," +
                "footprintRadiusMm,footprintOffsetMm,footprintOffsetRatio,association," +
                "probeEventReference,depthPairManifestReference,authority");
            foreach (Sample sample in samples)
            {
                float ratio = sample.PatchRadiusMm > 0f
                    ? sample.PatchOffsetMm / sample.PatchRadiusMm : 0f;
                writer.WriteLine(string.Join(",", Csv(sessionId), I(sample.EventRow),
                    Csv(Key(sample.Record.Key)), I(sample.GunGelFrame), I(sample.PlatformFrame),
                    Csv(sample.Event), Csv(sample.Reason), F(sample.Source.x), F(sample.Source.y),
                    F(sample.Source.z), F(sample.Target.x), F(sample.Target.y), F(sample.Target.z),
                    F(sample.ResidualMm), F(sample.AngleDeg), F(sample.GapMm), F(sample.ViewRadius),
                    F(sample.MotionQuality), F(sample.BaselineMm), F(sample.SpreadDeg),
                    I(sample.PatchIndex), F(sample.PatchRadiusMm), F(sample.PatchOffsetMm), F(ratio),
                    Csv(sample.Association), Csv("../../probe_shadow/verdict_events.csv#row=" +
                        sample.EventRow.ToString(CultureInfo.InvariantCulture)),
                    Csv("../../depth_pairs/manifest.csv#platformFrame=" +
                        sample.PlatformFrame.ToString(CultureInfo.InvariantCulture)),
                    "diagnostic_only"));
            }
        }

        private static void WriteSummary(string path, string sessionId,
            List<Record> records, List<Sample> samples)
        {
            var outcomes = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var orientations = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var postVerdictOutcomes = new SortedDictionary<string, int>(StringComparer.Ordinal);
            int fingerprintRecords = 0, holdRecords = 0, challenged = 0;
            int rejectTransitions = 0, reopenTransitions = 0;
            int postVerdictRecords = 0, postVerdictResolved = 0, postVerdictUnknown = 0;
            foreach (Record record in records)
            {
                string outcome = Outcome(record);
                string orientation = Orientation(RepresentativeNormal(record));
                Increment(outcomes, outcome);
                Increment(orientations, orientation);
                if (record.Key.FingerprintId > 0) fingerprintRecords++; else holdRecords++;
                if (record.ChallengeSamples > 0) challenged++;
                rejectTransitions += record.RejectTransitions;
                reopenTransitions += record.ReopenTransitions;
                if (!string.IsNullOrEmpty(record.PostVerdictOutcome))
                {
                    postVerdictRecords++;
                    Increment(postVerdictOutcomes, record.PostVerdictOutcome);
                    if (record.PostVerdictOutcome.StartsWith("unknown_",
                            StringComparison.Ordinal))
                        postVerdictUnknown++;
                    else
                        postVerdictResolved++;
                }
            }
            var sb = new StringBuilder(4096);
            sb.Append("{\n  \"schema\": \"").Append(Schema).Append("\",\n")
                .Append("  \"sessionId\": \"").Append(Json(sessionId)).Append("\",\n")
                .Append("  \"authority\": \"diagnostic_only\",\n")
                .Append("  \"generalizationUnit\": \"continuous surface and evidence features; never world coordinates or fingerprint ids\",\n")
                .Append("  \"records\": ").Append(records.Count).Append(",\n")
                .Append("  \"fingerprintRecords\": ").Append(fingerprintRecords).Append(",\n")
                .Append("  \"unconfirmedCellRecords\": ").Append(holdRecords).Append(",\n")
                .Append("  \"sampleRows\": ").Append(samples.Count).Append(",\n")
                .Append("  \"challengedRecords\": ").Append(challenged).Append(",\n")
                .Append("  \"rejectTransitions\": ").Append(rejectTransitions).Append(",\n")
                 .Append("  \"reopenTransitions\": ").Append(reopenTransitions).Append(",\n")
                 .Append("  \"postVerdictRecords\": ").Append(postVerdictRecords).Append(",\n")
                 .Append("  \"postVerdictResolved\": ").Append(postVerdictResolved).Append(",\n")
                 .Append("  \"postVerdictUnknown\": ").Append(postVerdictUnknown).Append(",\n")
                 .Append("  \"outcomes\": ");
            AppendCounts(sb, outcomes);
            sb.Append(",\n  \"orientations\": ");
            AppendCounts(sb, orientations);
            sb.Append(",\n  \"postVerdictOutcomes\": ");
            AppendCounts(sb, postVerdictOutcomes);
            sb.Append("\n}\n");
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
        }

        private static void WriteSchema(string path)
        {
            string json = "{\n" +
                "  \"schema\": \"" + Schema + "\",\n" +
                "  \"authority\": \"none; this archive is never consumed by production\",\n" +
                "  \"identity\": \"sessionId plus cell plus fingerprint generation is an audit key only\",\n" +
                "  \"generalization\": \"cross-session analysis groups continuous geometry, view, motion and evidence features; coordinates and ids are forbidden as reusable blacklist keys\",\n" +
                "  \"archive\": \"one row per accepted fingerprint generation plus unresolved cells\",\n" +
                 "  \"samples\": \"normalized references to decision-bearing probe events and source depth manifests; payloads are not duplicated\",\n" +
                 "  \"postVerdict\": \"optional verdict_followups rows distinguish confirmed false surfaces, support reestablishment, conflicts and explicit unknowns after Reject\",\n" +
                 "  \"descriptors\": \"orientation and evidenceOutcome are descriptive bins only, never acceptance or deletion thresholds\",\n" +
                "  \"missingValues\": \"empty numeric values mean unavailable evidence, not zero\"\n" +
                "}\n";
            File.WriteAllText(path, json, new UTF8Encoding(false));
        }

        private static void Geometry(Record record, out Vector3 center,
            out Vector3 normal, out List<float> radii, out List<float> spreads,
            out float thicknessMm, out float tangentialSpanMm)
        {
            radii = new List<float>(record.Patches.Count);
            spreads = new List<float>(record.Patches.Count);
            center = record.HasSnapshot ? record.SnapshotCenter : Vector3.zero;
            normal = record.HasSnapshot ? record.SnapshotNormal : Vector3.zero;
            if (record.Patches.Count > 0)
            {
                center = Vector3.zero;
                Vector3 normalSum = Vector3.zero;
                foreach (Patch patch in record.Patches)
                {
                    center += patch.Center;
                    normalSum += patch.Normal;
                    radii.Add(patch.RadiusMm);
                }
                center /= record.Patches.Count;
                normal = SafeNormal(normalSum);
                foreach (Patch patch in record.Patches)
                    spreads.Add(Vector3.Angle(normal, patch.Normal));
            }
            normal = SafeNormal(normal);
            float minNormal = float.PositiveInfinity, maxNormal = float.NegativeInfinity;
            float maxTangent = 0f;
            foreach (Patch patch in record.Patches)
            {
                Vector3 delta = patch.Center - center;
                float along = Vector3.Dot(delta, normal);
                minNormal = Math.Min(minNormal, along);
                maxNormal = Math.Max(maxNormal, along);
                Vector3 tangent = delta - normal * along;
                maxTangent = Math.Max(maxTangent, tangent.magnitude + patch.RadiusMm * 0.001f);
            }
            thicknessMm = record.Patches.Count > 0 ? (maxNormal - minNormal) * 1000f : 0f;
            tangentialSpanMm = maxTangent * 2000f;
        }

        private static Vector3 RepresentativeNormal(Record record)
        {
            if (record.HasSnapshot) return SafeNormal(record.SnapshotNormal);
            Vector3 sum = Vector3.zero;
            foreach (Patch patch in record.Patches) sum += patch.Normal;
            return SafeNormal(sum);
        }

        private static string Outcome(Record record)
        {
            if (record.Key.FingerprintId <= 0) return "unconfirmed_hold";
            if (record.FinalVerdict == "reject" || record.RejectTransitions > 0)
                return "free_space_refuted";
            if (record.ReopenTransitions > 0 || record.Key.Generation > 1)
                return "recovered_generation";
            if (record.FinalVerdict == "superseded") return "superseded_generation";
            if (record.ChallengeSamples > 0) return "challenged_supported";
            return "supported";
        }

        private static string Orientation(Vector3 normal)
        {
            normal = SafeNormal(normal);
            Vector3 a = new Vector3(Math.Abs(normal.x), Math.Abs(normal.y), Math.Abs(normal.z));
            if (a.y >= 0.85f) return normal.y >= 0f ? "horizontal_up" : "horizontal_down";
            if (a.x >= 0.85f) return normal.x >= 0f ? "vertical_x_positive" : "vertical_x_negative";
            if (a.z >= 0.85f) return normal.z >= 0f ? "vertical_z_positive" : "vertical_z_negative";
            return "oblique_or_structured";
        }

        private static string Tags(Record record, string orientation, string outcome)
        {
            var tags = new List<string> { orientation, outcome };
            if (record.Patches.Count == 0) tags.Add("no_fingerprint_footprint");
            else if (record.Patches.Count < 3) tags.Add("sparse_fingerprint_footprint");
            else if (record.Patches.Count >= 12) tags.Add("full_fingerprint_footprint");
            if (record.FingerprintEdgeHitSamples > 0) tags.Add("footprint_edge_challenge");
            if (record.MaxSpreadDeg >= 3f || record.MaxBaselineMm >= 80f)
                tags.Add("multi_view_observed");
            if (!string.IsNullOrEmpty(record.PostVerdictOutcome))
                tags.Add("post_" + record.PostVerdictOutcome);
            return string.Join("|", tags);
        }

        private static CellKey ReadCell(CsvHeader header, string[] row) => new CellKey(
            Int(header.Get(row, "cellX")), Int(header.Get(row, "cellY")),
            Int(header.Get(row, "cellZ")), Int(header.Get(row, "axis")));

        private static Vector3 Vector(CsvHeader h, string[] row,
            string x, string y, string z) => new Vector3(
                Float(h.Get(row, x)), Float(h.Get(row, y)), Float(h.Get(row, z)));

        private static Vector3 SafeNormal(Vector3 value) =>
            value.sqrMagnitude > 1e-10f ? value.normalized : Vector3.up;

        private static void AddFinite(List<float> values, float value)
        {
            if (!float.IsNaN(value) && !float.IsInfinity(value)) values.Add(value);
        }

        private static float Q(List<float> values, float fraction)
        {
            if (values == null || values.Count == 0) return float.NaN;
            var sorted = new List<float>(values);
            sorted.Sort();
            int index = Mathf.Clamp(Mathf.RoundToInt((sorted.Count - 1) * fraction),
                0, sorted.Count - 1);
            return sorted[index];
        }

        private static float Max(List<float> values) =>
            values == null || values.Count == 0 ? float.NaN : values.Max();

        private static void Increment(SortedDictionary<string, int> values, string key)
        {
            if (string.IsNullOrEmpty(key)) key = "unknown";
            values[key] = values.TryGetValue(key, out int count) ? count + 1 : 1;
        }

        private static void AppendCounts(StringBuilder sb,
            SortedDictionary<string, int> values)
        {
            sb.Append('{');
            bool first = true;
            foreach (KeyValuePair<string, int> pair in values)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append("\"").Append(Json(pair.Key)).Append("\":").Append(pair.Value);
            }
            sb.Append('}');
        }

        private static string Key(RecordKey key) =>
            key.Cell.X + ":" + key.Cell.Y + ":" + key.Cell.Z + ":" +
            key.Cell.Axis + ":" + key.FingerprintId + ":" + key.Generation;

        private static void Require(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("required probe file missing", path);
        }

        private static string[] Split(string line) => line.Split(',');
        private static int Int(string value, int fallback = 0) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture,
                out int parsed) ? parsed : fallback;
        private static bool Bool(string value) => value == "1" ||
            string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
        private static float Float(string value) =>
            float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture,
                out float parsed) ? parsed : 0f;
        private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
        private static string F(float value) => float.IsNaN(value) || float.IsInfinity(value)
            ? string.Empty : value.ToString("R", CultureInfo.InvariantCulture);
        private static string Csv(string value) => "\"" + (value ?? string.Empty).Replace("\"", "\"\"") + "\"";
        private static string Json(string value) => (value ?? string.Empty)
            .Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n");
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 虚拟探针的只读账本。它只消费 GunGel 已经回读到 CPU 的对应结果，
    /// 不写候选、对应、TSDF、纸皮或任何生产缓冲。稳定裁决可为
    /// “合流仅纸”提供最终绘制隔离，但仍没有几何或融合修改权。
    /// </summary>
    internal sealed class VirtualProbeShadowLedger : IDisposable
    {
        private const uint ObservationValid = 1u << 0;
        private const uint RawAvailable = 1u << 1;
        private const uint DualAgree = 1u << 2;
        private const uint StableMatch = 1u << 3;
        private const uint StableDual = 1u << 4;
        private const uint Unopposed = 1u << 5;
        private const uint StableFound = 1u << 6;
        private const uint ResidualPass = 1u << 7;
        private const uint NormalPass = 1u << 8;
        private const uint ProductionAuthorityMask = (1u << 6) - 1u;

        private readonly VirtualProbeShadowAdjudicator _adjudicator =
            new VirtualProbeShadowAdjudicator();
        private readonly object _writeLock = new object();
        private readonly StringBuilder _pendingRows = new StringBuilder(65536);
        private string _manifestPath = string.Empty;
        private string _sourceDescription =
            "the same pre-correction GunGel correspondence readback already used for frame calibration";
        private string _statusLabel = "shadow_only";
        private bool _active;
        private int _rows;
        private int _writeErrors;

        internal int Rows => _rows;
        internal int WriteErrors => _writeErrors;
        internal int AdjudicatorFrameRows => _adjudicator.FrameRows;
        internal int AdjudicatorEventRows => _adjudicator.EventRows;
        internal int AdjudicatorFollowupRows => _adjudicator.FollowupRows;
        internal int AdjudicatorWriteErrors => _adjudicator.WriteErrors;
        internal int GraduationRaceCandidateRows =>
            _adjudicator.GraduationRaceCandidateRows;
        internal int GraduationRaceSummaryRows =>
            _adjudicator.GraduationRaceSummaryRows;
        internal int GraduationRaceWitnessEventRows =>
            _adjudicator.GraduationRaceWitnessEventRows;
        internal int GraduationRaceWriteErrors =>
            _adjudicator.GraduationRaceWriteErrors;
        internal string AdjudicatorGuidanceCompact =>
            _adjudicator.GetGuidanceCompact();
        internal string AdjudicatorGuidanceHudFixed =>
            _adjudicator.GetGuidanceHudFixed();
        internal VirtualProbeShadowAdjudicator.GuidanceTargetSnapshot
            AdjudicatorGuidanceTarget => _adjudicator.GetGuidanceTargetSnapshot();
        internal VirtualProbeShadowAdjudicator.GuidanceFrameSnapshot
            AdjudicatorGuidanceFrame => _adjudicator.GetGuidanceFrameSnapshot();

        internal int CopyAdjudicatorGuidanceVisuals(
            VirtualProbeShadowAdjudicator.GuidanceCellVisual[] destination)
        {
            return _adjudicator.CopyGuidanceCellVisuals(destination);
        }

        internal int CorrectionRevision => _adjudicator.CorrectionRevision;

        internal int CopyPaperCorrectionCells(
            List<VirtualProbeShadowAdjudicator.PaperCorrectionCell> destination)
        {
            return _adjudicator.CopyPaperCorrectionCells(destination);
        }

        internal void Begin(string sessionDirectory,
            string relativeDirectory = "probe_shadow",
            string sourceDescription = null,
            string statusLabel = "shadow_only")
        {
            End();
            if (string.IsNullOrEmpty(sessionDirectory)) return;
            string directory = Path.Combine(sessionDirectory,
                string.IsNullOrEmpty(relativeDirectory) ? "probe_shadow" : relativeDirectory);
            Directory.CreateDirectory(directory);
            _sourceDescription = string.IsNullOrEmpty(sourceDescription)
                ? "the same pre-correction GunGel correspondence readback already used for frame calibration"
                : sourceDescription;
            _statusLabel = string.IsNullOrEmpty(statusLabel) ? "shadow_only" : statusLabel;
            _manifestPath = Path.Combine(directory, "frames.csv");
            File.WriteAllText(_manifestPath,
                "gunGelFrame,platformFrame,unityFrame,unscaledTime,gridX,gridY,pixelStride,depthWidth,depthHeight,observations,valid,rawAvailable,dualAgree,stableFound,residualPass,normalPass,stableMatch,stableDual,unopposed,productionAuthority,singleContact,corroboratedContact,stableGeometry,opposedOrMismatch,centerValid,middleValid,outerValid,centerAuthority,middleAuthority,outerAuthority,residualP50Mm,residualP90Mm,residualP95Mm,residualMaxMm,angleP50Deg,angleP90Deg,angleP95Deg,angleMaxDeg,rangeP50M,rangeP90M,rangeMaxM,newIndependentViewCells,correlatedViewMatches,maxBaselineMm,maxViewSpreadDeg,freeSpaceChallenges,freeGapLt20Mm,freeGap20To50Mm,freeGap50To100Mm,freeGapGe100Mm,independentFreeSpaceChallenges,maxFreeSpaceGapMm,fingerprintAssociationTests,fingerprintAssociationHits,fingerprintAssociationMisses,sourceCameraX,sourceCameraY,sourceCameraZ,sourceCameraQx,sourceCameraQy,sourceCameraQz,sourceCameraQw,angularDegPerSec,linearMps,motionQuality,status\n",
                new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(directory, "schema.json"), BuildSchemaJson(_sourceDescription),
                new UTF8Encoding(false));
            _adjudicator.Begin(directory);
            _pendingRows.Clear();
            _rows = 0;
            _writeErrors = 0;
            _active = true;
        }

        internal void Record(
            int gunGelFrame,
            int platformFrame,
            NativeArray<GunGelEvidenceShadow.Correspondence> correspondences,
            int gridX,
            int gridY,
            int pixelStride,
            int depthWidth,
            int depthHeight,
            Matrix4x4 sourceProjectionInverse,
            Matrix4x4 sourceViewInverse,
            float angularSpeed,
            float linearSpeed,
            float motionQuality)
        {
            if (!_active || string.IsNullOrEmpty(_manifestPath) ||
                correspondences.Length == 0) return;

            try
            {
                Vector3 camera = sourceViewInverse.MultiplyPoint3x4(Vector3.zero);
                Quaternion cameraRotation = sourceViewInverse.rotation;
                VirtualProbeShadowAdjudicator.FrameResult shadow =
                    _adjudicator.Evaluate(gunGelFrame, platformFrame,
                        correspondences, gridX, gridY, pixelStride,
                        depthWidth, depthHeight, sourceProjectionInverse,
                        sourceViewInverse, motionQuality);
                var residualHistogram = new int[101]; // 0..100+ mm
                var angleHistogram = new int[91];     // 0..90+ deg
                var rangeHistogram = new int[101];    // 0..5+ m, 5 cm bins
                int residualSamples = 0, angleSamples = 0, rangeSamples = 0;
                float residualMax = 0f, angleMax = 0f, rangeMax = 0f;
                int valid = 0, raw = 0, dual = 0, stableFound = 0;
                int residualPass = 0, normalPass = 0, stableMatch = 0;
                int stableDual = 0, unopposed = 0, authority = 0;
                int singleContact = 0, corroborated = 0, stableGeometry = 0;
                int opposedOrMismatch = 0;
                int centerValid = 0, middleValid = 0, outerValid = 0;
                int centerAuthority = 0, middleAuthority = 0, outerAuthority = 0;
                int independentCells = shadow.NewIndependentViewCells;
                int correlatedMatches = shadow.CorrelatedViewMatches;
                float maxBaseline = shadow.MaxBaselineMetres;
                float maxSpread = shadow.MaxViewSpreadDeg;
                int freeSpaceChallenges = shadow.FreeSpaceChallenges;
                int freeGapLt20 = shadow.FreeGapLt20;
                int freeGap20To50 = shadow.FreeGap20To50;
                int freeGap50To100 = shadow.FreeGap50To100;
                int freeGapGe100 = shadow.FreeGapGe100;
                int independentFreeSpace = shadow.IndependentFreeSpaceChallenges;
                float maxFreeSpaceGap = shadow.MaxFreeSpaceGapMetres;
                int fingerprintTests = shadow.FingerprintAssociationTests;
                int fingerprintHits = shadow.FingerprintAssociationHits;
                int fingerprintMisses = shadow.FingerprintAssociationMisses;

                for (int i = 0; i < correspondences.Length; i++)
                {
                    GunGelEvidenceShadow.Correspondence item = correspondences[i];
                    uint flags = (uint)Mathf.Max(0, Mathf.RoundToInt(item.SourceValid.w));
                    bool isValid = (flags & ObservationValid) != 0u;
                    if (!isValid) continue;
                    valid++;
                    bool hasRaw = (flags & RawAvailable) != 0u;
                    bool agrees = (flags & DualAgree) != 0u;
                    bool found = (flags & StableFound) != 0u ||
                                 (flags & StableMatch) != 0u;
                    bool residualOk = (flags & ResidualPass) != 0u;
                    bool normalOk = (flags & NormalPass) != 0u;
                    bool matched = (flags & StableMatch) != 0u;
                    bool mature = (flags & StableDual) != 0u;
                    bool noOpposition = (flags & Unopposed) != 0u;
                    bool hasAuthority = (flags & ProductionAuthorityMask) ==
                                        ProductionAuthorityMask;
                    if (hasRaw) raw++;
                    if (agrees) dual++;
                    if (found) stableFound++;
                    if (residualOk) residualPass++;
                    if (normalOk) normalPass++;
                    if (matched) stableMatch++;
                    if (mature) stableDual++;
                    if (noOpposition) unopposed++;
                    if (hasAuthority) authority++;

                    if (!agrees) singleContact++;
                    else if (!matched) corroborated++;
                    else stableGeometry++;
                    if (agrees && found && (!matched || !noOpposition))
                        opposedOrMismatch++;

                    int gx = gridX > 0 ? i % gridX : 0;
                    int gy = gridX > 0 ? i / gridX : 0;
                    float px = Mathf.Min(gx * Mathf.Max(1, pixelStride),
                        Mathf.Max(0, depthWidth - 1));
                    float py = Mathf.Min(gy * Mathf.Max(1, pixelStride),
                        Mathf.Max(0, depthHeight - 1));
                    float nx = depthWidth > 0 ? px / depthWidth * 2f - 1f : 0f;
                    float ny = depthHeight > 0 ? py / depthHeight * 2f - 1f : 0f;
                    float radius = Mathf.Max(Mathf.Abs(nx), Mathf.Abs(ny));
                    if (radius < 0.35f)
                    {
                        centerValid++;
                        if (hasAuthority) centerAuthority++;
                    }
                    else if (radius < 0.70f)
                    {
                        middleValid++;
                        if (hasAuthority) middleAuthority++;
                    }
                    else
                    {
                        outerValid++;
                        if (hasAuthority) outerAuthority++;
                    }

                    Vector3 source = new Vector3(item.SourceValid.x,
                        item.SourceValid.y, item.SourceValid.z);
                    float range = Vector3.Distance(camera, source);
                    AddHistogram(rangeHistogram, range / 0.05f);
                    rangeSamples++;
                    rangeMax = Mathf.Max(rangeMax, range);

                    if (!found) continue;
                    Vector3 target = new Vector3(item.TargetSigma.x,
                        item.TargetSigma.y, item.TargetSigma.z);
                    Vector3 normal = new Vector3(item.NormalAngle.x,
                        item.NormalAngle.y, item.NormalAngle.z);
                    float normalLength = normal.magnitude;
                    if (normalLength > 1e-5f)
                    {
                        normal /= normalLength;
                        float residual = Mathf.Abs(Vector3.Dot(source - target, normal));
                        AddHistogram(residualHistogram, residual * 1000f);
                        residualSamples++;
                        residualMax = Mathf.Max(residualMax, residual);
                    }
                    float angle = Mathf.Max(0f, item.NormalAngle.w);
                    AddHistogram(angleHistogram, angle);
                    angleSamples++;
                    angleMax = Mathf.Max(angleMax, angle);

                }

                string row = string.Join(",",
                    I(gunGelFrame), I(platformFrame), I(Time.frameCount),
                    D(Time.unscaledTimeAsDouble), I(gridX), I(gridY), I(pixelStride),
                    I(depthWidth), I(depthHeight), I(correspondences.Length), I(valid),
                    I(raw), I(dual), I(stableFound), I(residualPass), I(normalPass),
                    I(stableMatch), I(stableDual), I(unopposed), I(authority),
                    I(singleContact), I(corroborated), I(stableGeometry),
                    I(opposedOrMismatch), I(centerValid), I(middleValid), I(outerValid),
                    I(centerAuthority), I(middleAuthority), I(outerAuthority),
                    F(Percentile(residualHistogram, residualSamples, 0.50f)),
                    F(Percentile(residualHistogram, residualSamples, 0.90f)),
                    F(Percentile(residualHistogram, residualSamples, 0.95f)),
                    F(residualMax * 1000f),
                    F(Percentile(angleHistogram, angleSamples, 0.50f)),
                    F(Percentile(angleHistogram, angleSamples, 0.90f)),
                    F(Percentile(angleHistogram, angleSamples, 0.95f)),
                    F(angleMax),
                    F(Percentile(rangeHistogram, rangeSamples, 0.50f) * 0.05f),
                    F(Percentile(rangeHistogram, rangeSamples, 0.90f) * 0.05f),
                    F(rangeMax), I(independentCells), I(correlatedMatches),
                    F(maxBaseline * 1000f), F(maxSpread),
                    I(freeSpaceChallenges), I(freeGapLt20), I(freeGap20To50),
                    I(freeGap50To100), I(freeGapGe100), I(independentFreeSpace),
                    F(maxFreeSpaceGap * 1000f),
                    I(fingerprintTests), I(fingerprintHits), I(fingerprintMisses),
                    F(camera.x), F(camera.y), F(camera.z),
                    F(cameraRotation.x), F(cameraRotation.y), F(cameraRotation.z),
                    F(cameraRotation.w), F(angularSpeed), F(linearSpeed),
                    F(motionQuality), _statusLabel) + "\n";
                lock (_writeLock)
                {
                    _pendingRows.Append(row);
                    if (_pendingRows.Length >= 65536)
                        FlushRowsLocked();
                }
                _rows++;
            }
            catch
            {
                _writeErrors++;
                // 旁观账本失败不能改变生产扫描。
            }
        }

        internal void End()
        {
            _adjudicator.End();
            lock (_writeLock)
                FlushRowsLocked();
            _active = false;
        }

        public void Dispose()
        {
            End();
            _adjudicator.Dispose();
        }

        private void FlushRowsLocked()
        {
            if (_pendingRows.Length == 0 || string.IsNullOrEmpty(_manifestPath)) return;
            try
            {
                File.AppendAllText(_manifestPath, _pendingRows.ToString(),
                    new UTF8Encoding(false));
                _pendingRows.Clear();
            }
            catch
            {
                _writeErrors++;
                // 保留内存中的行，后续 End 仍可重试一次。
            }
        }

        private static string BuildSchemaJson(string sourceDescription)
        {
            return "{\n" +
                   "  \"schema\": \"scancover.virtual_probe_shadow.v2\",\n" +
                   "  \"authority\": \"none; diagnostic-only and never read by GunGel candidates, TSDF, paper or mesh\",\n" +
                   "  \"source\": \"" + Json(sourceDescription) + "\",\n" +
                   "  \"continuousEvidence\": \"contact range, candidate residual, normal angle, view radius, motion, source camera baseline and view spread are retained before offline threshold calibration\",\n" +
                   "  \"provisionalGrouping\": {\"spatialCellMetres\":0.05,\"independentBaselineMetres\":0.08,\"independentAngleDeg\":3.0,\"independentFrameGap\":2,\"freeSpaceRayStride\":8},\n" +
                   "  \"viewWitnessSemantics\": \"independence is tested pairwise against stored real camera poses; coarse direction bins have no authority\",\n" +
                   "  \"freeSpaceSemantics\": \"a dual-witness ray can challenge an accepted generation only after intersecting one of its sealed safe-support footprint patches before the new hit; gap bands, patch attribution and view independence are recorded, but no production candidate is removed online\",\n" +
                   "  \"contactSemantics\": \"a valid hit supports contact only; raw/post-QRS agreement supports existence; stable match supports a geometry hypothesis; final free-space thresholds are calibrated by deterministic offline replay\",\n" +
                   "  \"adjudicator\": \"verdict_frames/events/cells implement reversible hold/accept/reject shadow decisions; verdict_followups audits each rejected fingerprint as confirmed false surface, support reestablished, conflict, or explicit unknown; all outputs have zero production authority\",\n" +
                   "  \"graduationRace\": \"graduation_race_candidates/summary compare parallel shadow-only graduation contracts; graduation_race_witness_events records every raw support callback and both decision paths without changing production\",\n" +
                   "  \"important\": \"provisional grouping and verdicts are audit lenses, not production acceptance thresholds\"\n" +
                   "}\n";
        }

        private static string Json(string value)
        {
            return (value ?? string.Empty).Replace("\\", "\\\\")
                .Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
        }

        private static void AddHistogram(int[] histogram, float value)
        {
            int index = Mathf.Clamp(Mathf.FloorToInt(value), 0, histogram.Length - 1);
            histogram[index]++;
        }

        private static float Percentile(int[] histogram, int count, float fraction)
        {
            if (count <= 0) return 0f;
            int wanted = Mathf.Clamp(Mathf.CeilToInt(count * fraction), 1, count);
            int sum = 0;
            for (int i = 0; i < histogram.Length; i++)
            {
                sum += histogram[i];
                if (sum >= wanted) return i;
            }
            return histogram.Length - 1;
        }

        private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
        private static string F(float value) => float.IsNaN(value) || float.IsInfinity(value)
            ? "" : value.ToString("R", CultureInfo.InvariantCulture);
        private static string D(double value) => double.IsNaN(value) || double.IsInfinity(value)
            ? "" : value.ToString("R", CultureInfo.InvariantCulture);

    }
}

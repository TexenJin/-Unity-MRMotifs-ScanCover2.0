using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// Complete-mesh, room-independent candidate generation and evidence scoring.
/// This is the Quest C# equivalent of generic_room_structure_extractor.py.
/// </summary>
public sealed partial class RoomSetupReplicaV2StructuralExtractor
{
    private enum RuntimeEdgeLevel
    {
        Weak,
        Moderate,
        Strong
    }

    private sealed class RuntimeFootprintSelection
    {
        public List<Vector2> polygon;
        public string method;
        public string parameter;
        public int candidateCount;
        public int strongEdges;
        public int moderateEdges;
        public int weakEdges;
        public float score;
        public float maskIoU;
    }

    private sealed class RuntimeFootprintCandidate
    {
        public string method;
        public string parameter;
        public bool[,] cells;
        public List<Vector2> polygon;
        public List<RuntimeEdgeScore> evidence;
        public float score;
        public float iou;
    }

    private struct RuntimeEdgeScore
    {
        public float length;
        public float planeCoverage;
        public float sampleCoverage;
        public float heightSpan;
        public RuntimeEdgeLevel level;
        public bool softConnector;
    }

    private struct RuntimePlaneSegment
    {
        public bool horizontal;
        public float coordinate;
        public float lo;
        public float hi;
        public float height;
    }

    private struct RuntimeSamplePoint
    {
        public float u;
        public float v;
        public float y;
        public float minimumY;
        public float maximumY;
        public Vector3 normal;
    }

    private readonly struct RuntimeSupportKey :
        IEquatable<RuntimeSupportKey>
    {
        private readonly bool _horizontal;
        private readonly int _coordinate;
        private readonly int _lo;
        private readonly int _hi;

        public RuntimeSupportKey(bool horizontal, float coordinate,
                                 float lo, float hi)
        {
            _horizontal = horizontal;
            _coordinate = Mathf.RoundToInt(coordinate * 1000f);
            _lo = Mathf.RoundToInt(lo * 1000f);
            _hi = Mathf.RoundToInt(hi * 1000f);
        }

        public bool Equals(RuntimeSupportKey other) =>
            _horizontal == other._horizontal &&
            _coordinate == other._coordinate &&
            _lo == other._lo && _hi == other._hi;

        public override bool Equals(object obj) =>
            obj is RuntimeSupportKey other && Equals(other);

        public override int GetHashCode()
        {
            int hash = _horizontal ? 1 : 0;
            hash = hash * 397 ^ _coordinate;
            hash = hash * 397 ^ _lo;
            return hash * 397 ^ _hi;
        }
    }

    private struct RuntimeSupportValue
    {
        public float planeCoverage;
        public float sampleCoverage;
        public float heightSpan;
    }

    private readonly Dictionary<RuntimeSupportKey, RuntimeSupportValue>
        _runtimeSupportCache = new();

    private bool TrySelectRuntimeFootprint(
        bool[,] seed, RuntimeEvidenceGrid raster,
        List<float> xs, List<float> ys,
        float floor, float ceiling,
        out RuntimeFootprintSelection selection, out string issue)
    {
        selection = null;
        issue = "no supported closed footprint candidate";
        _runtimeSupportCache.Clear();
        if (xs.Count < 2 || ys.Count < 2)
            return false;

        List<RuntimePlaneSegment> planeSegments =
            BuildRuntimePlaneSegments();
        List<RuntimeSamplePoint> verticalSamples =
            BuildRuntimeVerticalSamples();
        BuildRuntimeCellOccupancy(seed, raster, xs, ys,
            out float[,] occupancy, out float[,] cellArea);

        List<RuntimeFootprintCandidate> candidates = new();
        HashSet<string> signatures = new(StringComparer.Ordinal);

        float[] sampleThresholds = { 0.50f, 0.58f, 0.66f };
        float[] envelopeMargins = { 0.30f, 0.40f, 0.52f };
        for (int s = 0; s < sampleThresholds.Length; s++)
        for (int m = 0; m < envelopeMargins.Length; m++)
        {
            bool[,] cells = BuildRuntimeSupportEnvelope(
                seed, raster, xs, ys, cellArea,
                sampleThresholds[s], envelopeMargins[m],
                planeSegments, verticalSamples);
            AddRuntimeCandidate(candidates, signatures,
                "outer_raw_support_envelope",
                $"sample={sampleThresholds[s]:F2},margin=" +
                $"{envelopeMargins[m]:F2}", cells, xs, ys,
                planeSegments, verticalSamples, ceiling - floor,
                raster.cellSize);
        }

        float[] closeRadii = { 0.10f, 0.16f, 0.22f, 0.30f, 0.40f };
        float[] enclosureThresholds = { 0.25f, 0.45f, 0.65f };
        for (int r = 0; r < closeRadii.Length; r++)
        {
            bool[,] enclosed = BuildRuntimeWallEnclosure(
                seed, raster, closeRadii[r], ceiling - floor,
                planeSegments);
            if (!RuntimeAny(enclosed)) continue;
            BuildRuntimeCellOccupancy(enclosed, raster, xs, ys,
                out float[,] enclosureOccupancy, out _);
            for (int t = 0; t < enclosureThresholds.Length; t++)
            {
                bool[,] cells = RuntimeThresholdCells(
                    enclosureOccupancy, enclosureThresholds[t]);
                cells = KeepLargestRuntimeComponentWeighted(cells, cellArea);
                AddRuntimeCandidate(candidates, signatures,
                    "wall_enclosure",
                    $"close={closeRadii[r]:F2},cell=" +
                    $"{enclosureThresholds[t]:F2}", cells, xs, ys,
                    planeSegments, verticalSamples, ceiling - floor,
                    raster.cellSize);
            }
        }

        float[] unseenPriors =
            { 0.035f, 0.08f, 0.14f, 0.22f, 0.30f, 0.40f, 0.48f, 0.54f, 0.60f };
        float[] smoothWeights = { 0.12f, 0.20f, 0.32f, 0.50f, 0.80f };
        for (int p = 0; p < unseenPriors.Length; p++)
        for (int w = 0; w < smoothWeights.Length; w++)
        {
            bool[,] cells = BuildRuntimeGraphCut(
                occupancy, cellArea, xs, ys,
                smoothWeights[w], unseenPriors[p],
                planeSegments, verticalSamples, raster.cellSize);
            cells = KeepLargestRuntimeComponentWeighted(cells, cellArea);
            AddRuntimeCandidate(candidates, signatures,
                "wall_aware_graph_cut",
                $"smooth={smoothWeights[w]:F2},prior=" +
                $"{unseenPriors[p]:F3}", cells, xs, ys,
                planeSegments, verticalSamples, ceiling - floor,
                raster.cellSize);
        }

        for (float threshold = 0.18f;
             threshold <= 0.821f; threshold += 0.04f)
        {
            bool[,] cells = RuntimeThresholdCells(occupancy, threshold);
            cells = KeepLargestRuntimeComponentWeighted(cells, cellArea);
            AddRuntimeCandidate(candidates, signatures,
                "occupancy_threshold", $"threshold={threshold:F2}",
                cells, xs, ys, planeSegments, verticalSamples,
                ceiling - floor, raster.cellSize);
        }

        if (candidates.Count == 0)
            return false;

        RuntimeFootprintCandidate best = null;
        for (int i = 0; i < candidates.Count; i++)
        {
            RuntimeFootprintCandidate candidate = candidates[i];
            candidate.iou = RuntimeMaskIoU(
                RuntimeMaskFromCells(candidate.cells, raster, xs, ys), seed);
            float perimeter = 0f;
            float weightedSupport = 0f;
            float weakLong = 0f;
            int unsupported = 0;
            int connectors = 0;
            for (int e = 0; e < candidate.evidence.Count; e++)
            {
                RuntimeEdgeScore edge = candidate.evidence[e];
                perimeter += edge.length;
                weightedSupport += edge.length * Mathf.Max(
                    edge.planeCoverage, 0.85f * edge.sampleCoverage);
                if (edge.level == RuntimeEdgeLevel.Weak)
                {
                    unsupported++;
                    if (edge.length > 0.45f) weakLong += edge.length;
                }
                if (edge.softConnector) connectors++;
            }
            weightedSupport /= Mathf.Max(perimeter, 0.0001f);
            float outerBonus = candidate.method ==
                "outer_raw_support_envelope" ? 0.05f : 0f;
            candidate.score = candidate.iou +
                0.12f * weightedSupport -
                0.0035f * candidate.polygon.Count -
                0.025f * unsupported -
                0.10f * weakLong / Mathf.Max(perimeter, 0.0001f) +
                0.005f * connectors + outerBonus;
            if (best == null || candidate.score > best.score)
                best = candidate;
        }

        if (best == null || best.polygon.Count < 4)
            return false;
        int strong = 0;
        int moderate = 0;
        int weak = 0;
        for (int i = 0; i < best.evidence.Count; i++)
        {
            switch (best.evidence[i].level)
            {
                case RuntimeEdgeLevel.Strong: strong++; break;
                case RuntimeEdgeLevel.Moderate: moderate++; break;
                default: weak++; break;
            }
        }
        selection = new RuntimeFootprintSelection
        {
            polygon = best.polygon,
            method = best.method,
            parameter = best.parameter,
            candidateCount = candidates.Count,
            strongEdges = strong,
            moderateEdges = moderate,
            weakEdges = weak,
            score = best.score,
            maskIoU = best.iou
        };
        issue = "";
        return true;
    }

    private void AddRuntimeCandidate(
        List<RuntimeFootprintCandidate> candidates,
        HashSet<string> signatures, string method, string parameter,
        bool[,] cells, List<float> xs, List<float> ys,
        List<RuntimePlaneSegment> planeSegments,
        List<RuntimeSamplePoint> verticalSamples,
        float roomHeight, float rasterCell)
    {
        if (!RuntimeAny(cells)) return;
        List<Vector2> polygon = TraceRuntimeArrangementBoundary(
            cells, xs, ys);
        CleanupRuntimePolygon(polygon);
        if (polygon.Count < 4 || Mathf.Abs(RuntimePolygonArea(polygon)) < 1f)
            return;
        if (RuntimePolygonArea(polygon) < 0f) polygon.Reverse();
        // Keep repeated candidates from different parameter combinations.
        // The offline script retains them, and their order is part of exact
        // winner/tie comparison even when the resulting polygon is identical.
        List<RuntimeEdgeScore> evidence = EvaluateRuntimeEdges(
            polygon, planeSegments, verticalSamples,
            roomHeight, rasterCell);
        candidates.Add(new RuntimeFootprintCandidate
        {
            method = method,
            parameter = parameter,
            cells = cells,
            polygon = polygon,
            evidence = evidence
        });
    }

    private List<RuntimePlaneSegment> BuildRuntimePlaneSegments()
    {
        List<RuntimePlaneSegment> result = new();
        for (int i = 0; i < _planes.Count; i++)
        {
            ExtractedPlane plane = _planes[i];
            if (plane.kind != StructuralKind.Wall) continue;
            Vector2 a = RuntimeUv(plane.p0);
            Vector2 b = RuntimeUv(plane.p1);
            bool horizontal = Mathf.Abs(b.x - a.x) >=
                              Mathf.Abs(b.y - a.y);
            result.Add(new RuntimePlaneSegment
            {
                horizontal = horizontal,
                coordinate = horizontal ? (a.y + b.y) * 0.5f
                                        : (a.x + b.x) * 0.5f,
                lo = horizontal ? Mathf.Min(a.x, b.x) : Mathf.Min(a.y, b.y),
                hi = horizontal ? Mathf.Max(a.x, b.x) : Mathf.Max(a.y, b.y),
                height = plane.p3.y - plane.p0.y
            });
        }
        return result;
    }

    private List<RuntimeSamplePoint> BuildRuntimeVerticalSamples()
    {
        List<RuntimeSamplePoint> result = new();
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            if (Mathf.Abs(sample.normal.y) > 0.45f) continue;
            Vector2 uv = RuntimeUv(sample.center);
            result.Add(new RuntimeSamplePoint
            {
                u = uv.x,
                v = uv.y,
                y = sample.center.y,
                minimumY = Mathf.Min(sample.a.y,
                    Mathf.Min(sample.b.y, sample.c.y)),
                maximumY = Mathf.Max(sample.a.y,
                    Mathf.Max(sample.b.y, sample.c.y)),
                normal = sample.normal
            });
        }
        return result;
    }

    private List<RuntimeEdgeScore> EvaluateRuntimeEdges(
        List<Vector2> polygon,
        List<RuntimePlaneSegment> planeSegments,
        List<RuntimeSamplePoint> verticalSamples,
        float roomHeight, float rasterCell)
    {
        List<RuntimeEdgeScore> result = new();
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            bool horizontal = Mathf.Abs(a.y - b.y) < 0.0001f;
            float coordinate = horizontal ? a.y : a.x;
            float lo = horizontal ? Mathf.Min(a.x, b.x) : Mathf.Min(a.y, b.y);
            float hi = horizontal ? Mathf.Max(a.x, b.x) : Mathf.Max(a.y, b.y);
            RuntimeSegmentSupport(horizontal, coordinate, lo, hi,
                planeSegments, verticalSamples, rasterCell,
                out float planeCoverage, out float sampleCoverage,
                out float heightSpan);
            float heightContinuity = heightSpan /
                Mathf.Max(roomHeight, 0.0001f);
            RuntimeEdgeLevel level =
                planeCoverage >= 0.75f && sampleCoverage >= 0.60f &&
                heightContinuity >= 0.42f
                    ? RuntimeEdgeLevel.Strong
                    : (planeCoverage >= 0.45f ||
                       sampleCoverage >= 0.45f) &&
                      heightContinuity >= 0.24f
                        ? RuntimeEdgeLevel.Moderate
                        : RuntimeEdgeLevel.Weak;
            result.Add(new RuntimeEdgeScore
            {
                length = hi - lo,
                planeCoverage = planeCoverage,
                sampleCoverage = sampleCoverage,
                heightSpan = heightSpan,
                level = level,
                softConnector = false
            });
        }

        float shortLimit = Mathf.Clamp(
            0.16f * Mathf.Sqrt(Mathf.Abs(RuntimePolygonArea(polygon))),
            0.45f, 0.65f);
        for (int i = 0; i < result.Count; i++)
        {
            RuntimeEdgeScore edge = result[i];
            RuntimeEdgeScore previous = result[
                (i - 1 + result.Count) % result.Count];
            RuntimeEdgeScore next = result[(i + 1) % result.Count];
            Vector2 a = polygon[i];
            Vector2 b = polygon[(i + 1) % polygon.Count];
            Vector2 pa = polygon[(i - 1 + polygon.Count) % polygon.Count];
            Vector2 nb = polygon[(i + 2) % polygon.Count];
            bool edgeHorizontal = Mathf.Abs(a.y - b.y) < 0.0001f;
            bool previousHorizontal = Mathf.Abs(pa.y - a.y) < 0.0001f;
            bool nextHorizontal = Mathf.Abs(b.y - nb.y) < 0.0001f;
            bool connector = edge.length <= shortLimit &&
                edge.planeCoverage < 0.45f &&
                edge.sampleCoverage >= 0.45f &&
                edge.heightSpan >= 0.45f * roomHeight &&
                previous.level == RuntimeEdgeLevel.Strong &&
                next.level == RuntimeEdgeLevel.Strong &&
                previousHorizontal == nextHorizontal &&
                previousHorizontal != edgeHorizontal;
            if (!connector) continue;
            edge.softConnector = true;
            edge.level = RuntimeEdgeLevel.Moderate;
            result[i] = edge;
        }
        return result;
    }

    private void RuntimeSegmentSupport(bool horizontal, float coordinate,
        float lo, float hi,
        List<RuntimePlaneSegment> planeSegments,
        List<RuntimeSamplePoint> verticalSamples,
        float rasterCell,
        out float planeCoverage, out float sampleCoverage,
        out float heightSpan)
    {
        RuntimeSupportKey key = new RuntimeSupportKey(
            horizontal, coordinate, lo, hi);
        if (_runtimeSupportCache.TryGetValue(
                key, out RuntimeSupportValue cached))
        {
            planeCoverage = cached.planeCoverage;
            sampleCoverage = cached.sampleCoverage;
            heightSpan = cached.heightSpan;
            return;
        }

        float lineTolerance = Mathf.Max(0.075f, 1.5f * rasterCell);
        List<Vector2> intervals = new();
        for (int i = 0; i < planeSegments.Count; i++)
        {
            RuntimePlaneSegment plane = planeSegments[i];
            if (plane.horizontal != horizontal ||
                Mathf.Abs(plane.coordinate - coordinate) > lineTolerance ||
                Mathf.Min(hi, plane.hi) <= Mathf.Max(lo, plane.lo))
                continue;
            intervals.Add(new Vector2(plane.lo, plane.hi));
        }
        planeCoverage = RuntimeIntervalCoverage(intervals, lo, hi);

        float length = Mathf.Max(hi - lo, 0.0001f);
        int binCount = Mathf.Max(1, Mathf.CeilToInt(
            length / Mathf.Max(0.10f, 2f * rasterCell)));
        bool[] bins = new bool[binCount];
        Vector3 expected = horizontal ? _lastAxisB : _lastAxisA;
        float maximumDistance = Mathf.Max(0.11f, 2.2f * rasterCell);
        float alongPadding = length <= 0.45f ? 0.12f : 0.08f;
        List<Vector2> heightIntervals = new();
        for (int i = 0; i < verticalSamples.Count; i++)
        {
            RuntimeSamplePoint sample = verticalSamples[i];
            float distance = Mathf.Abs(
                (horizontal ? sample.v : sample.u) - coordinate);
            float along = horizontal ? sample.u : sample.v;
            if (distance > maximumDistance ||
                along < lo - alongPadding || along > hi + alongPadding ||
                Mathf.Abs(Vector3.Dot(sample.normal, expected)) <
                Mathf.Cos(28f * DegreesToRadians))
                continue;
            int bin = Mathf.Clamp(Mathf.FloorToInt(
                (along - lo) / length * binCount), 0, binCount - 1);
            bins[bin] = true;
            heightIntervals.Add(new Vector2(
                sample.minimumY, sample.maximumY));
        }
        int hits = 0;
        for (int i = 0; i < bins.Length; i++)
            if (bins[i]) hits++;
        sampleCoverage = hits / (float)binCount;
        // Use the longest vertically connected run, not min-to-max.  A low
        // cabinet and a hanging object may project onto the same wall line;
        // the empty height between them must remain empty instead of turning
        // the pair into an apparently full-height architectural edge.
        heightSpan = RuntimeLongestConnectedHeight(
            heightIntervals, 0.075f);
        _runtimeSupportCache[key] = new RuntimeSupportValue
        {
            planeCoverage = planeCoverage,
            sampleCoverage = sampleCoverage,
            heightSpan = heightSpan
        };
    }

    private static float RuntimeLongestConnectedHeight(
        List<Vector2> intervals, float maximumGap)
    {
        if (intervals == null || intervals.Count == 0)
            return 0f;
        intervals.Sort((a, b) => a.x.CompareTo(b.x));
        float runStart = intervals[0].x;
        float runEnd = intervals[0].y;
        float longest = Mathf.Max(0f, runEnd - runStart);
        for (int i = 1; i < intervals.Count; i++)
        {
            float start = intervals[i].x;
            float end = intervals[i].y;
            if (start <= runEnd + maximumGap)
            {
                runEnd = Mathf.Max(runEnd, end);
            }
            else
            {
                longest = Mathf.Max(longest, runEnd - runStart);
                runStart = start;
                runEnd = end;
            }
        }
        return Mathf.Max(longest, runEnd - runStart);
    }

    private static float RuntimeIntervalCoverage(
        List<Vector2> intervals, float lo, float hi)
    {
        if (hi <= lo || intervals.Count == 0) return 0f;
        intervals.Sort((a, b) => a.x.CompareTo(b.x));
        float covered = 0f;
        float start = Mathf.Max(lo, intervals[0].x);
        float end = Mathf.Min(hi, intervals[0].y);
        for (int i = 1; i < intervals.Count; i++)
        {
            float a = Mathf.Max(lo, intervals[i].x);
            float b = Mathf.Min(hi, intervals[i].y);
            if (b <= a) continue;
            if (a > end)
            {
                covered += Mathf.Max(0f, end - start);
                start = a;
                end = b;
            }
            else
            {
                end = Mathf.Max(end, b);
            }
        }
        covered += Mathf.Max(0f, end - start);
        return Mathf.Clamp01(covered / (hi - lo));
    }

    private static void BuildRuntimeCellOccupancy(
        bool[,] seed, RuntimeEvidenceGrid raster,
        List<float> xs, List<float> ys,
        out float[,] occupancy, out float[,] cellArea)
    {
        int nx = xs.Count - 1;
        int ny = ys.Count - 1;
        occupancy = new float[nx, ny];
        cellArea = new float[nx, ny];
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
            RuntimeRasterRange(raster, seed, xs[i], xs[i + 1],
                ys[j], ys[j + 1], out int x0, out int x1,
                out int y0, out int y1);
            int total = 0;
            int hit = 0;
            for (int x = x0; x < x1; x++)
            for (int y = y0; y < y1; y++)
            {
                total++;
                if (seed[x, y]) hit++;
            }
            occupancy[i, j] = total > 0 ? hit / (float)total : 0f;
            cellArea[i, j] = (xs[i + 1] - xs[i]) *
                             (ys[j + 1] - ys[j]);
        }
    }

    private bool[,] BuildRuntimeSupportEnvelope(
        bool[,] seed, RuntimeEvidenceGrid raster,
        List<float> xs, List<float> ys, float[,] cellArea,
        float sampleThreshold, float envelopeMargin,
        List<RuntimePlaneSegment> planeSegments,
        List<RuntimeSamplePoint> verticalSamples)
    {
        bool[,] cells = new bool[xs.Count - 1, ys.Count - 1];
        List<float> allV = new();
        for (int x = 0; x < seed.GetLength(0); x++)
        for (int y = 0; y < seed.GetLength(1); y++)
            if (seed[x, y])
                allV.Add(raster.minV + (y + 0.5f) * raster.cellSize);
        if (allV.Count == 0) return cells;
        float globalCenter = RuntimeMedian(allV);

        for (int i = 0; i < xs.Count - 1; i++)
        {
            int x0 = Mathf.Clamp(Mathf.FloorToInt(
                (xs[i] - raster.minU) / raster.cellSize) - 1,
                0, seed.GetLength(0));
            int x1 = Mathf.Clamp(Mathf.CeilToInt(
                (xs[i + 1] - raster.minU) / raster.cellSize) + 2,
                0, seed.GetLength(0));
            List<float> localV = RuntimeSeedV(seed, raster, x0, x1);
            if (localV.Count == 0)
            {
                int expand = Mathf.Max(1,
                    Mathf.RoundToInt(0.18f / raster.cellSize));
                localV = RuntimeSeedV(seed, raster,
                    Mathf.Max(0, x0 - expand),
                    Mathf.Min(seed.GetLength(0), x1 + expand));
            }
            if (localV.Count == 0) continue;
            localV.Sort();
            float localMin = localV[0];
            float localMax = localV[localV.Count - 1];
            float center = localV.Count >= 4
                ? RuntimeMedianSorted(localV) : globalCenter;
            float top = float.NegativeInfinity;
            float bottom = float.PositiveInfinity;
            for (int y = 0; y < ys.Count; y++)
            {
                float coordinate = ys[y];
                RuntimeSegmentSupport(true, coordinate,
                    xs[i], xs[i + 1], planeSegments, verticalSamples,
                    raster.cellSize, out _, out float sampleCoverage, out _);
                bool shortSpan = xs[i + 1] - xs[i] <= 0.45f;
                bool accepted = sampleCoverage >= sampleThreshold ||
                    (shortSpan && sampleCoverage >= 0.45f);
                if (!accepted) continue;
                if (coordinate > center &&
                    coordinate <= localMax + envelopeMargin)
                    top = Mathf.Max(top, coordinate);
                if (coordinate < center &&
                    coordinate >= localMin - envelopeMargin)
                    bottom = Mathf.Min(bottom, coordinate);
            }
            if (!float.IsFinite(top) || !float.IsFinite(bottom)) continue;
            for (int j = 0; j < ys.Count - 1; j++)
            {
                float middle = (ys[j] + ys[j + 1]) * 0.5f;
                if (middle > bottom && middle < top)
                    cells[i, j] = true;
            }
        }
        return KeepLargestRuntimeComponentWeighted(cells, cellArea);
    }

    private static List<float> RuntimeSeedV(bool[,] seed,
        RuntimeEvidenceGrid raster, int x0, int x1)
    {
        List<float> result = new();
        for (int x = x0; x < x1; x++)
        for (int y = 0; y < seed.GetLength(1); y++)
            if (seed[x, y])
                result.Add(raster.minV + (y + 0.5f) * raster.cellSize);
        return result;
    }

    private bool[,] BuildRuntimeWallEnclosure(
        bool[,] seed, RuntimeEvidenceGrid raster, float closeRadius,
        float roomHeight, List<RuntimePlaneSegment> segments)
    {
        int width = seed.GetLength(0);
        int height = seed.GetLength(1);
        bool[,] barrier = new bool[width, height];
        for (int i = 0; i < segments.Count; i++)
        {
            RuntimePlaneSegment segment = segments[i];
            if (segment.height < 0.42f * roomHeight ||
                segment.hi - segment.lo < 0.14f)
                continue;
            Vector2 a = segment.horizontal
                ? new Vector2(segment.lo, segment.coordinate)
                : new Vector2(segment.coordinate, segment.lo);
            Vector2 b = segment.horizontal
                ? new Vector2(segment.hi, segment.coordinate)
                : new Vector2(segment.coordinate, segment.hi);
            RuntimeDrawRasterLine(barrier, raster, a, b);
        }
        barrier = DilateRuntimeMask(barrier,
            Mathf.Max(1, Mathf.RoundToInt(0.045f / raster.cellSize)));
        barrier = CloseRuntimeMask(barrier,
            Mathf.Max(1, Mathf.RoundToInt(closeRadius / raster.cellSize)));

        bool[,] outside = new bool[width, height];
        Queue<RuntimeGridPoint> queue = new();
        void SeedOutside(int x, int y)
        {
            if (x < 0 || y < 0 || x >= width || y >= height ||
                barrier[x, y] || outside[x, y]) return;
            outside[x, y] = true;
            queue.Enqueue(new RuntimeGridPoint(x, y));
        }
        for (int x = 0; x < width; x++)
        {
            SeedOutside(x, 0);
            SeedOutside(x, height - 1);
        }
        for (int y = 0; y < height; y++)
        {
            SeedOutside(0, y);
            SeedOutside(width - 1, y);
        }
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };
        while (queue.Count > 0)
        {
            RuntimeGridPoint p = queue.Dequeue();
            for (int n = 0; n < 4; n++)
                SeedOutside(p.x + dx[n], p.y + dy[n]);
        }
        bool[,] enclosed = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            enclosed[x, y] = !barrier[x, y] && !outside[x, y];

        int[,] labels = new int[width, height];
        int label = 0;
        int bestLabel = 0;
        float bestScore = 0f;
        for (int sx = 0; sx < width; sx++)
        for (int sy = 0; sy < height; sy++)
        {
            if (!enclosed[sx, sy] || labels[sx, sy] != 0) continue;
            label++;
            float score = 0f;
            labels[sx, sy] = label;
            queue.Enqueue(new RuntimeGridPoint(sx, sy));
            while (queue.Count > 0)
            {
                RuntimeGridPoint p = queue.Dequeue();
                score += (seed[p.x, p.y] ? 1f : 0f) + 0.08f;
                for (int n = 0; n < 4; n++)
                {
                    int x = p.x + dx[n];
                    int y = p.y + dy[n];
                    if (x < 0 || y < 0 || x >= width || y >= height ||
                        !enclosed[x, y] || labels[x, y] != 0) continue;
                    labels[x, y] = label;
                    queue.Enqueue(new RuntimeGridPoint(x, y));
                }
            }
            if (score > bestScore)
            {
                bestScore = score;
                bestLabel = label;
            }
        }
        bool[,] result = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            result[x, y] = labels[x, y] == bestLabel || barrier[x, y];
        return result;
    }

    private bool[,] BuildRuntimeGraphCut(
        float[,] occupancy, float[,] cellArea,
        List<float> xs, List<float> ys,
        float smoothWeight, float unseenPrior,
        List<RuntimePlaneSegment> planeSegments,
        List<RuntimeSamplePoint> verticalSamples,
        float rasterCell)
    {
        int nx = occupancy.GetLength(0);
        int ny = occupancy.GetLength(1);
        int nodeCount = nx * ny;
        int source = nodeCount;
        int sink = nodeCount + 1;
        RuntimeMaxFlow flow = new RuntimeMaxFlow(nodeCount + 2);
        int Node(int x, int y) => x * ny + y;
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
            float probability = Mathf.Clamp(
                unseenPrior + (0.975f - unseenPrior) * occupancy[i, j],
                0.025f, 0.975f);
            float exterior = -Mathf.Log(1f - probability) * cellArea[i, j];
            float interior = -Mathf.Log(probability) * cellArea[i, j];
            flow.AddDirected(source, Node(i, j), exterior);
            flow.AddDirected(Node(i, j), sink, interior);
        }
        for (int i = 1; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
            RuntimeSegmentSupport(false, xs[i], ys[j], ys[j + 1],
                planeSegments, verticalSamples, rasterCell,
                out float plane, out float sample, out _);
            float support = Mathf.Max(plane, 0.85f * sample);
            float cost = smoothWeight * (ys[j + 1] - ys[j]) *
                         (0.05f + Mathf.Pow(1f - support, 2f));
            flow.AddPair(Node(i - 1, j), Node(i, j), cost);
        }
        for (int i = 0; i < nx; i++)
        for (int j = 1; j < ny; j++)
        {
            RuntimeSegmentSupport(true, ys[j], xs[i], xs[i + 1],
                planeSegments, verticalSamples, rasterCell,
                out float plane, out float sample, out _);
            float support = Mathf.Max(plane, 0.85f * sample);
            float cost = smoothWeight * (xs[i + 1] - xs[i]) *
                         (0.05f + Mathf.Pow(1f - support, 2f));
            flow.AddPair(Node(i, j - 1), Node(i, j), cost);
        }
        for (int i = 0; i < nx; i++)
        {
            AddRuntimeExteriorCost(flow, Node(i, 0), sink, true, ys[0],
                xs[i], xs[i + 1], smoothWeight,
                planeSegments, verticalSamples, rasterCell);
            AddRuntimeExteriorCost(flow, Node(i, ny - 1), sink, true,
                ys[ys.Count - 1], xs[i], xs[i + 1], smoothWeight,
                planeSegments, verticalSamples, rasterCell);
        }
        for (int j = 0; j < ny; j++)
        {
            AddRuntimeExteriorCost(flow, Node(0, j), sink, false, xs[0],
                ys[j], ys[j + 1], smoothWeight,
                planeSegments, verticalSamples, rasterCell);
            AddRuntimeExteriorCost(flow, Node(nx - 1, j), sink, false,
                xs[xs.Count - 1], ys[j], ys[j + 1], smoothWeight,
                planeSegments, verticalSamples, rasterCell);
        }
        flow.Calculate(source, sink);
        bool[] reachable = flow.SourceReachable(source);
        bool[,] cells = new bool[nx, ny];
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            cells[i, j] = reachable[Node(i, j)];
        return cells;
    }

    private void AddRuntimeExteriorCost(RuntimeMaxFlow flow,
        int node, int sink, bool horizontal, float coordinate,
        float lo, float hi, float smoothWeight,
        List<RuntimePlaneSegment> planeSegments,
        List<RuntimeSamplePoint> verticalSamples, float rasterCell)
    {
        RuntimeSegmentSupport(horizontal, coordinate, lo, hi,
            planeSegments, verticalSamples, rasterCell,
            out float plane, out float sample, out _);
        float support = Mathf.Max(plane, 0.85f * sample);
        float cost = smoothWeight * (hi - lo) *
                     (0.05f + Mathf.Pow(1f - support, 2f));
        flow.AddDirected(node, sink, cost);
    }

    private static bool[,] RuntimeThresholdCells(float[,] values,
                                                  float threshold)
    {
        bool[,] result = new bool[values.GetLength(0), values.GetLength(1)];
        for (int x = 0; x < values.GetLength(0); x++)
        for (int y = 0; y < values.GetLength(1); y++)
            result[x, y] = values[x, y] >= threshold;
        return result;
    }

    private static bool[,] KeepLargestRuntimeComponentWeighted(
        bool[,] source, float[,] weights)
    {
        int width = source.GetLength(0);
        int height = source.GetLength(1);
        int[,] labels = new int[width, height];
        Queue<RuntimeGridPoint> queue = new();
        int label = 0;
        int bestLabel = 0;
        float bestWeight = 0f;
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };
        for (int sx = 0; sx < width; sx++)
        for (int sy = 0; sy < height; sy++)
        {
            if (!source[sx, sy] || labels[sx, sy] != 0) continue;
            label++;
            float total = 0f;
            labels[sx, sy] = label;
            queue.Enqueue(new RuntimeGridPoint(sx, sy));
            while (queue.Count > 0)
            {
                RuntimeGridPoint p = queue.Dequeue();
                total += weights[p.x, p.y];
                for (int n = 0; n < 4; n++)
                {
                    int x = p.x + dx[n];
                    int y = p.y + dy[n];
                    if (x < 0 || y < 0 || x >= width || y >= height ||
                        !source[x, y] || labels[x, y] != 0) continue;
                    labels[x, y] = label;
                    queue.Enqueue(new RuntimeGridPoint(x, y));
                }
            }
            if (total > bestWeight)
            {
                bestWeight = total;
                bestLabel = label;
            }
        }
        bool[,] result = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            result[x, y] = labels[x, y] == bestLabel && bestLabel != 0;
        return result;
    }

    private static bool[,] RuntimeMaskFromCells(bool[,] cells,
        RuntimeEvidenceGrid raster, List<float> xs, List<float> ys)
    {
        bool[,] result = new bool[
            raster.cells.GetLength(0), raster.cells.GetLength(1)];
        for (int i = 0; i < cells.GetLength(0); i++)
        for (int j = 0; j < cells.GetLength(1); j++)
        {
            if (!cells[i, j]) continue;
            RuntimeRasterRange(raster, result, xs[i], xs[i + 1],
                ys[j], ys[j + 1], out int x0, out int x1,
                out int y0, out int y1);
            for (int x = x0; x < x1; x++)
            for (int y = y0; y < y1; y++)
                result[x, y] = true;
        }
        return result;
    }

    private static float RuntimeMaskIoU(bool[,] a, bool[,] b)
    {
        int intersection = 0;
        int union = 0;
        for (int x = 0; x < a.GetLength(0); x++)
        for (int y = 0; y < a.GetLength(1); y++)
        {
            if (a[x, y] || b[x, y]) union++;
            if (a[x, y] && b[x, y]) intersection++;
        }
        return union > 0 ? intersection / (float)union : 0f;
    }

    private static void RuntimeRasterRange(RuntimeEvidenceGrid raster,
        bool[,] target, float u0, float u1, float v0, float v1,
        out int x0, out int x1, out int y0, out int y1)
    {
        x0 = Mathf.Clamp(Mathf.FloorToInt(
            (u0 - raster.minU) / raster.cellSize),
            0, target.GetLength(0) - 1);
        x1 = Mathf.Clamp(Mathf.CeilToInt(
            (u1 - raster.minU) / raster.cellSize) + 1,
            x0 + 1, target.GetLength(0));
        y0 = Mathf.Clamp(Mathf.FloorToInt(
            (v0 - raster.minV) / raster.cellSize),
            0, target.GetLength(1) - 1);
        y1 = Mathf.Clamp(Mathf.CeilToInt(
            (v1 - raster.minV) / raster.cellSize) + 1,
            y0 + 1, target.GetLength(1));
    }

    private static void RuntimeDrawRasterLine(bool[,] target,
        RuntimeEvidenceGrid raster, Vector2 a, Vector2 b)
    {
        int x0 = Mathf.RoundToInt((a.x - raster.minU) / raster.cellSize);
        int y0 = Mathf.RoundToInt((a.y - raster.minV) / raster.cellSize);
        int x1 = Mathf.RoundToInt((b.x - raster.minU) / raster.cellSize);
        int y1 = Mathf.RoundToInt((b.y - raster.minV) / raster.cellSize);
        int dx = Mathf.Abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -Mathf.Abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true)
        {
            if (x0 >= 0 && y0 >= 0 &&
                x0 < target.GetLength(0) && y0 < target.GetLength(1))
                target[x0, y0] = true;
            if (x0 == x1 && y0 == y1) break;
            int twice = 2 * error;
            if (twice >= dy) { error += dy; x0 += sx; }
            if (twice <= dx) { error += dx; y0 += sy; }
        }
    }

    private static bool RuntimeAny(bool[,] cells)
    {
        for (int x = 0; x < cells.GetLength(0); x++)
        for (int y = 0; y < cells.GetLength(1); y++)
            if (cells[x, y]) return true;
        return false;
    }

    private static float RuntimeMedian(List<float> values)
    {
        values.Sort();
        return RuntimeMedianSorted(values);
    }

    private static float RuntimeMedianSorted(List<float> values)
    {
        int middle = values.Count / 2;
        return values.Count % 2 == 0
            ? (values[middle - 1] + values[middle]) * 0.5f
            : values[middle];
    }

    private static string RuntimePolygonSignature(List<Vector2> polygon)
    {
        StringBuilder builder = new StringBuilder(polygon.Count * 18);
        for (int i = 0; i < polygon.Count; i++)
        {
            builder.Append(Mathf.RoundToInt(polygon[i].x * 1000f))
                .Append(',')
                .Append(Mathf.RoundToInt(polygon[i].y * 1000f))
                .Append(';');
        }
        return builder.ToString();
    }

    private sealed class RuntimeMaxFlow
    {
        private sealed class Edge
        {
            public int to;
            public int reverse;
            public float capacity;
        }

        private readonly List<Edge>[] _graph;
        private int[] _level;
        private int[] _next;

        public RuntimeMaxFlow(int nodes)
        {
            _graph = new List<Edge>[nodes];
            for (int i = 0; i < nodes; i++) _graph[i] = new List<Edge>();
        }

        public void AddDirected(int from, int to, float capacity)
        {
            Edge forward = new Edge
            {
                to = to,
                reverse = _graph[to].Count,
                capacity = Mathf.Max(0f, capacity)
            };
            Edge reverse = new Edge
            {
                to = from,
                reverse = _graph[from].Count,
                capacity = 0f
            };
            _graph[from].Add(forward);
            _graph[to].Add(reverse);
        }

        public void AddPair(int a, int b, float capacity)
        {
            AddDirected(a, b, capacity);
            AddDirected(b, a, capacity);
        }

        public float Calculate(int source, int sink)
        {
            float total = 0f;
            while (BuildLevels(source, sink))
            {
                _next = new int[_graph.Length];
                while (true)
                {
                    float value = Send(source, sink, float.PositiveInfinity);
                    if (value <= 0.000001f) break;
                    total += value;
                }
            }
            return total;
        }

        public bool[] SourceReachable(int source)
        {
            bool[] seen = new bool[_graph.Length];
            Queue<int> queue = new Queue<int>();
            seen[source] = true;
            queue.Enqueue(source);
            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                for (int i = 0; i < _graph[node].Count; i++)
                {
                    Edge edge = _graph[node][i];
                    if (edge.capacity <= 0.000001f || seen[edge.to]) continue;
                    seen[edge.to] = true;
                    queue.Enqueue(edge.to);
                }
            }
            return seen;
        }

        private bool BuildLevels(int source, int sink)
        {
            _level = new int[_graph.Length];
            Array.Fill(_level, -1);
            Queue<int> queue = new Queue<int>();
            _level[source] = 0;
            queue.Enqueue(source);
            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                for (int i = 0; i < _graph[node].Count; i++)
                {
                    Edge edge = _graph[node][i];
                    if (edge.capacity <= 0.000001f || _level[edge.to] >= 0)
                        continue;
                    _level[edge.to] = _level[node] + 1;
                    queue.Enqueue(edge.to);
                }
            }
            return _level[sink] >= 0;
        }

        private float Send(int node, int sink, float pushed)
        {
            if (node == sink) return pushed;
            for (; _next[node] < _graph[node].Count; _next[node]++)
            {
                Edge edge = _graph[node][_next[node]];
                if (edge.capacity <= 0.000001f ||
                    _level[edge.to] != _level[node] + 1)
                    continue;
                float value = Send(edge.to, sink,
                    Mathf.Min(pushed, edge.capacity));
                if (value <= 0.000001f) continue;
                edge.capacity -= value;
                _graph[edge.to][edge.reverse].capacity += value;
                return value;
            }
            return 0f;
        }
    }
}

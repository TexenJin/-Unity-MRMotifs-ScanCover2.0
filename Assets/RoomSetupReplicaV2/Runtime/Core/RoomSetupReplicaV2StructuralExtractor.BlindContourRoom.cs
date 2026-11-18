using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Fully blind post-freeze room footprint recovered from open 5 cm contour
/// slices plus near-floor/near-ceiling horizontal evidence.  This is the
/// device-side equivalent of the verified offline contour-stack route.
/// </summary>
public sealed partial class RoomSetupReplicaV2StructuralExtractor
{
    private const float RuntimeBlindLayerStep = 0.05f;

    private sealed class RuntimeBlindStack
    {
        public RuntimeEvidenceGrid grid;
        public bool[][,] rawLayers;
        public float[,] continuousHeight;
        public float[,] supportHeight;
        public float[,] rawFraction;
        public bool[,] floorMask;
        public bool[,] ceilingMask;
        public int layerCount;
        public float firstHeight;
        public float lastHeight;
    }

    private struct RuntimeBlindSegment
    {
        public int first;
        public int last;
        public float value;
    }

    private bool TryBuildRuntimeBlindContourFootprint(
        Vector3[] vertices, float floor, float ceiling,
        out RuntimeFootprintSelection selection, out string issue)
    {
        selection = null;
        issue = "blind contour stack was empty";
        RuntimeBlindStack stack = BuildRuntimeBlindStack(
            vertices, floor, ceiling);
        if (stack == null)
            return false;

        bool[][,] variants = new bool[4][,];
        variants[0] = SolveRuntimeBlindGraphCut(stack, 14f, 3f, 10f, 3f);
        variants[1] = SolveRuntimeBlindGraphCut(stack, 20f, 2f, 12f, 3f);
        variants[2] = SolveRuntimeBlindGraphCut(stack, 24f, 3f, 16f, 4f);
        variants[3] = SolveRuntimeBlindGraphCut(stack, 30f, 2f, 18f, 5f);

        int width = stack.grid.cells.GetLength(0);
        int height = stack.grid.cells.GetLength(1);
        bool[,] consensus = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            int votes = 0;
            for (int i = 0; i < variants.Length; i++)
                if (variants[i][x, y]) votes++;
            consensus[x, y] = votes >= 3;
        }
        consensus = KeepLargestRuntimeComponent(consensus);
        FillRuntimeMaskHoles(consensus);
        if (CountRuntimeMaskCells(consensus) *
                stack.grid.cellSize * stack.grid.cellSize < 1f)
        {
            issue = "blind contour consensus was too small";
            return false;
        }

        if (!TryBuildRuntimeBlindRectilinearPolygon(
                consensus, stack.grid, out List<Vector2> polygon))
        {
            issue = "blind contour rectilinear fit failed";
            return false;
        }
        int motherWallChanges = RefitRuntimeBlindMicroSteps(
            polygon, stack);
        CleanupRuntimePolygon(polygon);
        if (polygon.Count < 4 ||
            Mathf.Abs(RuntimePolygonArea(polygon)) < 1f ||
            RuntimePolygonHasSelfIntersections(polygon))
        {
            issue = "blind contour polygon was invalid";
            return false;
        }
        if (RuntimePolygonArea(polygon) < 0f)
            polygon.Reverse();

        float iou = RuntimeBlindPolygonMaskIoU(
            polygon, consensus, stack.grid);
        _runtimeSupportCache.Clear();
        List<RuntimeEdgeScore> edgeEvidence = EvaluateRuntimeEdges(
            polygon, BuildRuntimePlaneSegments(),
            BuildRuntimeVerticalSamples(), ceiling - floor,
            stack.grid.cellSize);
        int strong = 0;
        int moderate = 0;
        int weak = 0;
        for (int i = 0; i < edgeEvidence.Count; i++)
        {
            switch (edgeEvidence[i].level)
            {
                case RuntimeEdgeLevel.Strong: strong++; break;
                case RuntimeEdgeLevel.Moderate: moderate++; break;
                default: weak++; break;
            }
        }
        selection = new RuntimeFootprintSelection
        {
            polygon = polygon,
            method = "blind_open_5cm_contour_stack_consensus_" +
                     "rectilinear_mother_wall",
            parameter = $"heightRange={stack.firstHeight:F3}-" +
                        $"{stack.lastHeight:F3},step=0.050," +
                        $"variants=4,votes=3,rectPenalty=0.140," +
                        $"motherWallChanges={motherWallChanges}",
            candidateCount = 4,
            strongEdges = strong,
            moderateEdges = moderate,
            weakEdges = weak,
            score = iou,
            maskIoU = iou
        };
        issue = "";
        return true;
    }

    private RuntimeBlindStack BuildRuntimeBlindStack(
        Vector3[] vertices, float floor, float ceiling)
    {
        RuntimeEvidenceGrid grid = BuildRuntimeSliceGridTemplate(
            vertices, floor, ceiling);
        if (grid == null)
            return null;
        float roomHeight = ceiling - floor;
        float firstHeight = floor + 0.25f;
        float lastHeight = ceiling - Mathf.Max(0.08f, roomHeight * 0.03f);
        if (lastHeight < firstHeight + RuntimeBlindLayerStep * 2f)
            return null;

        int layerCount = Mathf.FloorToInt(
            (lastHeight - firstHeight) / RuntimeBlindLayerStep) + 1;
        int width = grid.cells.GetLength(0);
        int height = grid.cells.GetLength(1);
        bool[][,] rawLayers = new bool[layerCount][,];
        for (int layer = 0; layer < layerCount; layer++)
            rawLayers[layer] = new bool[width, height];
        bool[,] floorMask = new bool[width, height];
        bool[,] ceilingMask = new bool[width, height];

        int vertexCount = vertices.Length - vertices.Length % 3;
        for (int i = 0; i < vertexCount; i += 3)
        {
            Vector3 a = vertices[i];
            Vector3 b = vertices[i + 1];
            Vector3 c = vertices[i + 2];
            if (!RuntimeFinite(a) || !RuntimeFinite(b) || !RuntimeFinite(c))
                continue;
            Vector3 cross = Vector3.Cross(b - a, c - a);
            float magnitude = cross.magnitude;
            if (magnitude <= 0.000001f)
                continue;
            float horizontal = Mathf.Abs(cross.y) / magnitude;
            float centerY = (a.y + b.y + c.y) / 3f;
            if (horizontal >= 0.78f)
            {
                bool nearFloor = centerY >= floor - 0.10f &&
                                 centerY <= floor + 0.22f;
                bool nearCeiling = centerY >= ceiling - 0.18f &&
                                   centerY <= ceiling + 0.18f;
                if (nearFloor)
                    RasterRuntimeTriangle(floorMask, grid.minU, grid.minV,
                        grid.cellSize, RuntimeUv(a), RuntimeUv(b), RuntimeUv(c));
                if (nearCeiling)
                    RasterRuntimeTriangle(ceilingMask, grid.minU, grid.minV,
                        grid.cellSize, RuntimeUv(a), RuntimeUv(b), RuntimeUv(c));
                continue;
            }

            float minimumY = Mathf.Min(a.y, Mathf.Min(b.y, c.y));
            float maximumY = Mathf.Max(a.y, Mathf.Max(b.y, c.y));
            if (maximumY < firstHeight || minimumY > lastHeight)
                continue;
            int firstLayer = Mathf.Clamp(Mathf.CeilToInt(
                (minimumY - firstHeight - 0.0005f) / RuntimeBlindLayerStep),
                0, layerCount - 1);
            int lastLayer = Mathf.Clamp(Mathf.FloorToInt(
                (maximumY - firstHeight + 0.0005f) / RuntimeBlindLayerStep),
                0, layerCount - 1);
            for (int layer = firstLayer; layer <= lastLayer; layer++)
            {
                float sliceHeight = firstHeight + layer * RuntimeBlindLayerStep;
                if (!TryRuntimeSliceTriangle(a, b, c, sliceHeight,
                        out Vector2 first, out Vector2 second))
                    continue;
                RasterRuntimeSegment(rawLayers[layer], grid.minU,
                    grid.minV, grid.cellSize, first, second);
            }
        }

        floorMask = CloseRuntimeMask(floorMask, 2);
        ceilingMask = CloseRuntimeMask(ceilingMask, 2);
        int[,] current = new int[width, height];
        int[,] maximum = new int[width, height];
        bool[,] gapUsed = new bool[width, height];
        int[,] supportCount = new int[width, height];
        int[,] rawCount = new int[width, height];
        for (int layer = 0; layer < layerCount; layer++)
        {
            bool[,] raw = rawLayers[layer];
            bool[,] aligned = DilateRuntimeMask(raw, 1);
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                if (raw[x, y]) rawCount[x, y]++;
                if (aligned[x, y])
                {
                    supportCount[x, y]++;
                    current[x, y]++;
                    maximum[x, y] = Mathf.Max(maximum[x, y], current[x, y]);
                    gapUsed[x, y] = false;
                }
                else if (current[x, y] > 0 && !gapUsed[x, y])
                {
                    current[x, y]++;
                    gapUsed[x, y] = true;
                }
                else
                {
                    current[x, y] = 0;
                    gapUsed[x, y] = false;
                }
            }
        }

        float[,] continuousHeight = new float[width, height];
        float[,] supportHeight = new float[width, height];
        float[,] rawFraction = new float[width, height];
        int structuralCells = 0;
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            continuousHeight[x, y] = maximum[x, y] * RuntimeBlindLayerStep;
            supportHeight[x, y] = supportCount[x, y] * RuntimeBlindLayerStep;
            rawFraction[x, y] = rawCount[x, y] / (float)layerCount;
            if (continuousHeight[x, y] >= 0.70f) structuralCells++;
        }
        if (structuralCells < 20 ||
            (!RuntimeAny(floorMask) && !RuntimeAny(ceilingMask)))
            return null;

        return new RuntimeBlindStack
        {
            grid = grid,
            rawLayers = rawLayers,
            continuousHeight = continuousHeight,
            supportHeight = supportHeight,
            rawFraction = rawFraction,
            floorMask = floorMask,
            ceilingMask = ceilingMask,
            layerCount = layerCount,
            firstHeight = firstHeight,
            lastHeight = lastHeight
        };
    }

    private bool[,] SolveRuntimeBlindGraphCut(
        RuntimeBlindStack stack, float pairBase, float unionReward,
        float bothReward, float outsidePenalty)
    {
        int width = stack.grid.cells.GetLength(0);
        int height = stack.grid.cells.GetLength(1);
        int nodeCount = width * height;
        int source = nodeCount;
        int sink = nodeCount + 1;
        RuntimeMaxFlow flow = new RuntimeMaxFlow(nodeCount + 2);
        bool[,] both = new bool[width, height];
        bool[,] union = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            both[x, y] = stack.floorMask[x, y] && stack.ceilingMask[x, y];
            union[x, y] = stack.floorMask[x, y] || stack.ceilingMask[x, y];
        }
        bool[,] strongInside = ErodeRuntimeMask(both, 2);
        int Node(int x, int y) => x * height + y;
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            float costOutside = both[x, y]
                ? bothReward : union[x, y] ? unionReward : 0f;
            float costInside = union[x, y] ? 0f : outsidePenalty;
            if (strongInside[x, y]) costOutside += 80f;
            if (x < 3 || y < 3 || x >= width - 3 || y >= height - 3)
                costInside += 120f;
            flow.AddDirected(source, Node(x, y), costOutside);
            flow.AddDirected(Node(x, y), sink, costInside);
            if (x + 1 < width)
            {
                float ridge = Mathf.Max(stack.continuousHeight[x, y],
                    stack.continuousHeight[x + 1, y]);
                float cost = 1f + Mathf.Round(
                    pairBase * Mathf.Exp(-ridge / 0.42f));
                flow.AddPair(Node(x, y), Node(x + 1, y), cost);
            }
            if (y + 1 < height)
            {
                float ridge = Mathf.Max(stack.continuousHeight[x, y],
                    stack.continuousHeight[x, y + 1]);
                float cost = 1f + Mathf.Round(
                    pairBase * Mathf.Exp(-ridge / 0.42f));
                flow.AddPair(Node(x, y), Node(x, y + 1), cost);
            }
        }
        flow.Calculate(source, sink);
        bool[] reachable = flow.SourceReachable(source);
        bool[,] result = new bool[width, height];
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            result[x, y] = reachable[Node(x, y)];
        return result;
    }

    private static bool TryBuildRuntimeBlindRectilinearPolygon(
        bool[,] mask, RuntimeEvidenceGrid grid,
        out List<Vector2> polygon)
    {
        polygon = null;
        int width = mask.GetLength(0);
        int height = mask.GetLength(1);
        bool[] eligible = new bool[height];
        for (int y = 0; y < height; y++)
        {
            int count = 0;
            for (int x = 0; x < width; x++)
                if (mask[x, y]) count++;
            eligible[y] = count * grid.cellSize >= 1f;
        }
        int bestStart = -1;
        int bestEnd = -1;
        int runStart = -1;
        for (int y = 0; y <= height; y++)
        {
            bool hit = y < height && eligible[y];
            if (hit && runStart < 0) runStart = y;
            if (hit || runStart < 0) continue;
            int runEnd = y - 1;
            if (bestStart < 0 || runEnd - runStart > bestEnd - bestStart)
            {
                bestStart = runStart;
                bestEnd = runEnd;
            }
            runStart = -1;
        }
        if (bestStart < 0 || bestEnd - bestStart < 2)
            return false;
        int rowCount = bestEnd - bestStart + 1;
        float[] left = new float[rowCount];
        float[] right = new float[rowCount];
        for (int row = 0; row < rowCount; row++)
        {
            int y = bestStart + row;
            int first = -1;
            int last = -1;
            for (int x = 0; x < width; x++)
            {
                if (!mask[x, y]) continue;
                if (first < 0) first = x;
                last = x;
            }
            if (first < 0) return false;
            left[row] = grid.minU + first * grid.cellSize - grid.cellSize * 0.5f;
            right[row] = grid.minU + last * grid.cellSize + grid.cellSize * 0.5f;
        }

        List<RuntimeBlindSegment> leftSegments =
            RuntimeBlindPiecewise(left, grid.cellSize, 0.14f);
        List<RuntimeBlindSegment> rightSegments =
            RuntimeBlindPiecewise(right, grid.cellSize, 0.14f);
        if (leftSegments.Count == 0 || rightSegments.Count == 0)
            return false;

        float yBottom = grid.minV + bestStart * grid.cellSize -
                        grid.cellSize * 0.5f;
        float yTop = grid.minV + bestEnd * grid.cellSize +
                     grid.cellSize * 0.5f;
        List<Vector2> points = new List<Vector2>();
        AddRuntimeBlindPoint(points,
            new Vector2(leftSegments[0].value, yBottom));
        for (int i = 0; i < leftSegments.Count; i++)
        {
            RuntimeBlindSegment segment = leftSegments[i];
            float segmentTop = grid.minV +
                (bestStart + segment.last) * grid.cellSize +
                grid.cellSize * 0.5f;
            AddRuntimeBlindPoint(points,
                new Vector2(segment.value, segmentTop));
            if (i + 1 < leftSegments.Count)
                AddRuntimeBlindPoint(points,
                    new Vector2(leftSegments[i + 1].value, segmentTop));
        }
        AddRuntimeBlindPoint(points,
            new Vector2(rightSegments[rightSegments.Count - 1].value, yTop));
        for (int i = rightSegments.Count - 1; i >= 0; i--)
        {
            RuntimeBlindSegment segment = rightSegments[i];
            float segmentBottom = grid.minV +
                (bestStart + segment.first) * grid.cellSize -
                grid.cellSize * 0.5f;
            AddRuntimeBlindPoint(points,
                new Vector2(segment.value, segmentBottom));
            if (i > 0)
                AddRuntimeBlindPoint(points,
                    new Vector2(rightSegments[i - 1].value, segmentBottom));
        }
        AddRuntimeBlindPoint(points,
            new Vector2(leftSegments[0].value, yBottom));
        if (points.Count > 1 &&
            Vector2.Distance(points[0], points[points.Count - 1]) < 0.001f)
            points.RemoveAt(points.Count - 1);
        CleanupRuntimePolygon(points);
        polygon = points;
        return polygon.Count >= 4;
    }

    private static List<RuntimeBlindSegment> RuntimeBlindPiecewise(
        float[] values, float cell, float penalty)
    {
        int count = values.Length;
        double[] sum = new double[count + 1];
        double[] squared = new double[count + 1];
        for (int i = 0; i < count; i++)
        {
            sum[i + 1] = sum[i] + values[i];
            squared[i + 1] = squared[i] + values[i] * values[i];
        }
        double[] cost = new double[count + 1];
        int[] previous = new int[count + 1];
        for (int i = 0; i <= count; i++) cost[i] = double.PositiveInfinity;
        cost[0] = -penalty;
        for (int end = 1; end <= count; end++)
        for (int start = 0; start < end; start++)
        {
            int length = end - start;
            double total = sum[end] - sum[start];
            double sse = squared[end] - squared[start] -
                         total * total / length;
            double candidate = cost[start] + sse + penalty;
            if (candidate >= cost[end]) continue;
            cost[end] = candidate;
            previous[end] = start;
        }
        List<RuntimeBlindSegment> segments = new List<RuntimeBlindSegment>();
        int cursor = count;
        while (cursor > 0)
        {
            int start = previous[cursor];
            List<float> local = new List<float>(cursor - start);
            for (int i = start; i < cursor; i++) local.Add(values[i]);
            float median = RuntimeMedian(local);
            float snapped = Mathf.Round(median / cell) * cell;
            segments.Add(new RuntimeBlindSegment
            {
                first = start,
                last = cursor - 1,
                value = snapped
            });
            cursor = start;
        }
        segments.Reverse();
        if (segments.Count > 1 &&
            segments[0].last - segments[0].first <= 0)
        {
            RuntimeBlindSegment first = segments[0];
            RuntimeBlindSegment second = segments[1];
            second.first = first.first;
            segments[1] = second;
            segments.RemoveAt(0);
        }
        if (segments.Count > 1 &&
            segments[segments.Count - 1].last -
            segments[segments.Count - 1].first <= 0)
        {
            RuntimeBlindSegment before = segments[segments.Count - 2];
            RuntimeBlindSegment last = segments[segments.Count - 1];
            before.last = last.last;
            segments[segments.Count - 2] = before;
            segments.RemoveAt(segments.Count - 1);
        }
        return segments;
    }

    private static void AddRuntimeBlindPoint(
        List<Vector2> points, Vector2 point)
    {
        if (points.Count == 0 ||
            Vector2.Distance(points[points.Count - 1], point) > 0.000001f)
            points.Add(point);
    }

    private static int RefitRuntimeBlindMicroSteps(
        List<Vector2> polygon, RuntimeBlindStack stack)
    {
        int changes = 0;
        bool changed = true;
        while (changed && polygon.Count >= 4)
        {
            changed = false;
            int count = polygon.Count;
            for (int start = 0; start < count; start++)
            {
                Vector2 p0 = polygon[start % count];
                Vector2 p1 = polygon[(start + 1) % count];
                Vector2 p2 = polygon[(start + 2) % count];
                Vector2 p3 = polygon[(start + 3) % count];
                Vector2 e0 = p1 - p0;
                Vector2 bridge = p2 - p1;
                Vector2 e2 = p3 - p2;
                bool vertical = Mathf.Abs(e0.x) < 0.0001f &&
                                Mathf.Abs(e2.x) < 0.0001f;
                bool horizontal = Mathf.Abs(e0.y) < 0.0001f &&
                                  Mathf.Abs(e2.y) < 0.0001f;
                if ((!vertical && !horizontal) ||
                    Vector2.Dot(e0, e2) <= 0f ||
                    bridge.magnitude > 0.150001f ||
                    Mathf.Min(e0.magnitude, e2.magnitude) < 0.75f)
                    continue;

                float cross0 = vertical ? p0.x : p0.y;
                float cross1 = vertical ? p2.x : p2.y;
                float along0 = vertical
                    ? Mathf.Min(p0.y, Mathf.Min(p1.y, Mathf.Min(p2.y, p3.y)))
                    : Mathf.Min(p0.x, Mathf.Min(p1.x, Mathf.Min(p2.x, p3.x)));
                float along1 = vertical
                    ? Mathf.Max(p0.y, Mathf.Max(p1.y, Mathf.Max(p2.y, p3.y)))
                    : Mathf.Max(p0.x, Mathf.Max(p1.x, Mathf.Max(p2.x, p3.x)));
                RuntimeBlindEvidenceProfile(stack, vertical,
                    Mathf.Min(cross0, cross1), Mathf.Max(cross0, cross1),
                    along0, along1, out float[] coordinates,
                    out float[] profile);
                if (profile.Length == 0 ||
                    RuntimeBlindHasTwoSupportedModes(profile))
                    continue;
                int peak = 0;
                for (int i = 1; i < profile.Length; i++)
                    if (profile[i] > profile[peak]) peak = i;
                float mother = coordinates[peak];
                if (mother < Mathf.Min(cross0, cross1) - stack.grid.cellSize ||
                    mother > Mathf.Max(cross0, cross1) + stack.grid.cellSize)
                    continue;

                List<Vector2> rotated = new List<Vector2>(count);
                for (int i = 0; i < count; i++)
                    rotated.Add(polygon[(start + i) % count]);
                Vector2 first = rotated[0];
                Vector2 fourth = rotated[3];
                if (vertical) { first.x = mother; fourth.x = mother; }
                else { first.y = mother; fourth.y = mother; }
                List<Vector2> result = new List<Vector2>(count - 2) { first };
                result.Add(fourth);
                for (int i = 4; i < rotated.Count; i++) result.Add(rotated[i]);
                polygon.Clear();
                polygon.AddRange(result);
                changes++;
                changed = true;
                break;
            }
        }
        return changes;
    }

    private static void RuntimeBlindEvidenceProfile(
        RuntimeBlindStack stack, bool vertical,
        float crossMinimum, float crossMaximum,
        float alongMinimum, float alongMaximum,
        out float[] coordinates, out float[] profile)
    {
        float crossOrigin = vertical ? stack.grid.minU : stack.grid.minV;
        float alongOrigin = vertical ? stack.grid.minV : stack.grid.minU;
        int crossSize = vertical
            ? stack.continuousHeight.GetLength(0)
            : stack.continuousHeight.GetLength(1);
        int alongSize = vertical
            ? stack.continuousHeight.GetLength(1)
            : stack.continuousHeight.GetLength(0);
        float cell = stack.grid.cellSize;
        int first = Mathf.Max(0, Mathf.FloorToInt(
            (crossMinimum - cell - crossOrigin) / cell));
        int last = Mathf.Min(crossSize - 1, Mathf.CeilToInt(
            (crossMaximum + cell - crossOrigin) / cell));
        int alongFirst = Mathf.Max(0, Mathf.FloorToInt(
            (alongMinimum - alongOrigin) / cell));
        int alongLast = Mathf.Min(alongSize - 1, Mathf.CeilToInt(
            (alongMaximum - alongOrigin) / cell));
        if (last < first || alongLast < alongFirst)
        {
            coordinates = Array.Empty<float>();
            profile = Array.Empty<float>();
            return;
        }
        int count = last - first + 1;
        coordinates = new float[count];
        float[] raw = new float[count];
        int alongCount = alongLast - alongFirst + 1;
        for (int index = 0; index < count; index++)
        {
            int cross = first + index;
            coordinates[index] = crossOrigin + cross * cell;
            float continuous = 0f;
            float support = 0f;
            float fraction = 0f;
            for (int along = alongFirst; along <= alongLast; along++)
            {
                int x = vertical ? cross : along;
                int y = vertical ? along : cross;
                continuous += stack.continuousHeight[x, y];
                support += stack.supportHeight[x, y];
                fraction += stack.rawFraction[x, y];
            }
            raw[index] = 0.55f * continuous / alongCount +
                         0.25f * support / alongCount +
                         2.50f * fraction / alongCount;
        }
        profile = new float[count];
        const float sigma = 0.7f;
        for (int i = 0; i < count; i++)
        {
            float weighted = 0f;
            float weights = 0f;
            for (int offset = -3; offset <= 3; offset++)
            {
                int sample = Mathf.Clamp(i + offset, 0, count - 1);
                float weight = Mathf.Exp(
                    -(offset * offset) / (2f * sigma * sigma));
                weighted += raw[sample] * weight;
                weights += weight;
            }
            profile[i] = weighted / Mathf.Max(weights, 0.000001f);
        }
    }

    private static bool RuntimeBlindHasTwoSupportedModes(float[] profile)
    {
        if (profile.Length < 5)
            return false;
        float maximum = 0f;
        for (int i = 0; i < profile.Length; i++)
            maximum = Mathf.Max(maximum, profile[i]);
        if (maximum <= 0.000001f)
            return false;
        List<int> strong = new List<int>();
        for (int i = 0; i < profile.Length; i++)
        {
            bool peak = i == 0
                ? profile[0] > profile[1]
                : i == profile.Length - 1
                    ? profile[i] > profile[i - 1]
                    : profile[i] > profile[i - 1] &&
                      profile[i] >= profile[i + 1];
            if (peak && profile[i] >= 0.75f * maximum)
                strong.Add(i);
        }
        for (int a = 0; a < strong.Count; a++)
        for (int b = a + 1; b < strong.Count; b++)
        {
            if (strong[b] - strong[a] < 2) continue;
            float valley = float.PositiveInfinity;
            for (int i = strong[a]; i <= strong[b]; i++)
                valley = Mathf.Min(valley, profile[i]);
            if (valley <= 0.55f * Mathf.Min(
                    profile[strong[a]], profile[strong[b]]))
                return true;
        }
        return false;
    }

    private static float RuntimeBlindPolygonMaskIoU(
        List<Vector2> polygon, bool[,] reference, RuntimeEvidenceGrid grid)
    {
        int intersection = 0;
        int union = 0;
        for (int x = 0; x < reference.GetLength(0); x++)
        for (int y = 0; y < reference.GetLength(1); y++)
        {
            Vector2 point = new Vector2(
                grid.minU + x * grid.cellSize,
                grid.minV + y * grid.cellSize);
            bool inside = RuntimePointInsidePolygon(point, polygon);
            if (inside || reference[x, y]) union++;
            if (inside && reference[x, y]) intersection++;
        }
        return union > 0 ? intersection / (float)union : 0f;
    }
}

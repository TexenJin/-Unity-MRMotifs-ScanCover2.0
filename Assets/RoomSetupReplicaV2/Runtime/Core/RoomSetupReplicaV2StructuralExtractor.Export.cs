using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

public sealed partial class RoomSetupReplicaV2StructuralExtractor
{
    public string BuildStructuralSamplesJson()
    {
        var builder = new StringBuilder(
            Mathf.Max(1024, _samples.Count * 180));
        builder.Append("{\n  \"sampleCount\": ")
            .Append(_samples.Count).Append(",\n  \"samples\": [\n");
        for (int i = 0; i < _samples.Count; i++)
        {
            StructuralSample sample = _samples[i];
            builder.Append("    {\"a\":");
            AppendJsonVector3(builder, sample.a);
            builder.Append(",\"b\":");
            AppendJsonVector3(builder, sample.b);
            builder.Append(",\"c\":");
            AppendJsonVector3(builder, sample.c);
            builder.Append(",\"center\":");
            AppendJsonVector3(builder, sample.center);
            builder.Append(",\"normal\":");
            AppendJsonVector3(builder, sample.normal);
            builder.Append(",\"area\":");
            AppendJsonFloat(builder, sample.area);
            builder.Append(i + 1 < _samples.Count ? "},\n" : "}\n");
        }
        builder.Append("  ]\n}\n");
        return builder.ToString();
    }

    public string BuildPlaneClustersJson()
    {
        var builder = new StringBuilder(32768);
        builder.Append("{\n  \"axisA\":");
        AppendJsonVector3(builder, _lastAxisA);
        builder.Append(",\n  \"axisB\":");
        AppendJsonVector3(builder, _lastAxisB);
        builder.Append(",\n  \"families\": [\n");
        AppendClusterFamily(builder, "wallA", _wallFamilyA, false);
        builder.Append(",\n");
        AppendClusterFamily(builder, "wallB", _wallFamilyB, false);
        builder.Append(",\n");
        AppendClusterFamily(builder, "horizontal", _horizontal, true);
        builder.Append("\n  ]\n}\n");
        return builder.ToString();
    }

    public string BuildSupportGridsJson()
    {
        var builder = new StringBuilder(65536);
        builder.Append("{\n  \"encoding\": ")
            .Append("\"occupiedRuns are [row,startInclusive,endExclusive]\"")
            .Append(",\n  \"grids\": [\n");
        bool first = true;
        AppendGridFamily(builder, "wallA", _wallFamilyA,
            Vector3.Cross(Vector3.up, _lastAxisA), Vector3.up, ref first);
        AppendGridFamily(builder, "wallB", _wallFamilyB,
            Vector3.Cross(Vector3.up, _lastAxisB), Vector3.up, ref first);
        AppendGridFamily(builder, "horizontal", _horizontal,
            _lastAxisA, _lastAxisB, ref first);
        builder.Append("\n  ]\n}\n");
        return builder.ToString();
    }

    public string BuildPlanesJson(bool beforeTopology)
    {
        List<ExtractedPlane> source = beforeTopology
            ? _planesBeforeTopology : _planes;
        var builder = new StringBuilder(
            Mathf.Max(1024, source.Count * 220));
        builder.Append("{\n  \"stage\": \"")
            .Append(beforeTopology ? "before_topology" : "after_topology")
            .Append("\",\n  \"planeCount\": ")
            .Append(source.Count).Append(",\n  \"planes\": [\n");
        for (int i = 0; i < source.Count; i++)
        {
            ExtractedPlane plane = source[i];
            builder.Append("    {\"kind\":\"")
                .Append(plane.kind).Append("\",\"p0\":");
            AppendJsonVector3(builder, plane.p0);
            builder.Append(",\"p1\":");
            AppendJsonVector3(builder, plane.p1);
            builder.Append(",\"p2\":");
            AppendJsonVector3(builder, plane.p2);
            builder.Append(",\"p3\":");
            AppendJsonVector3(builder, plane.p3);
            builder.Append(i + 1 < source.Count ? "},\n" : "}\n");
        }
        builder.Append("  ]\n}\n");
        return builder.ToString();
    }

    public string BuildExtractorMetadataJson()
    {
        var builder = new StringBuilder(2048);
        builder.Append("{\n  \"status\": ")
            .Append(JsonString(LastStatus)).Append(",\n  \"axisA\": ");
        AppendJsonVector3(builder, _lastAxisA);
        builder.Append(",\n  \"axisB\": ");
        AppendJsonVector3(builder, _lastAxisB);
        builder.Append(",\n  \"estimatedFloorHeight\": ");
        AppendJsonFloat(builder, _lastFloorHeight);
        builder.Append(",\n  \"sampleCount\": ")
            .Append(_samples.Count)
            .Append(",\n  \"planesBeforeTopology\": ")
            .Append(_planesBeforeTopology.Count)
            .Append(",\n  \"planesAfterTopology\": ")
            .Append(_planes.Count)
            .Append(",\n  \"settings\": {\n")
            .Append("    \"axisToleranceDegrees\": ");
        AppendJsonFloat(builder, axisToleranceDegrees);
        builder.Append(",\n    \"coplanarMergeDistance\": ");
        AppendJsonFloat(builder, coplanarMergeDistance);
        builder.Append(",\n    \"supportCellSize\": ");
        AppendJsonFloat(builder, supportCellSize);
        builder.Append(",\n    \"minimumVerticalPlaneArea\": ");
        AppendJsonFloat(builder, minimumVerticalPlaneArea);
        builder.Append(",\n    \"minimumHorizontalArea\": ");
        AppendJsonFloat(builder, minimumHorizontalArea);
        builder.Append(",\n    \"wallHeightMergeTolerance\": ");
        AppendJsonFloat(builder, wallHeightMergeTolerance);
        builder.Append(",\n    \"coplanarGapCloseCells\": ")
            .Append(coplanarGapCloseCells)
            .Append(",\n    \"horizontalOcclusionGapCells\": ")
            .Append(horizontalOcclusionGapCells)
            .Append(",\n    \"maximumHorizontalOcclusionArea\": ");
        AppendJsonFloat(builder, maximumHorizontalOcclusionArea);
        builder.Append(",\n    \"orthogonalConnectionDistance\": ");
        AppendJsonFloat(builder, orthogonalConnectionDistance);
        builder.Append("\n  }\n}\n");
        return builder.ToString();
    }

    public string BuildSimplifiedRoomJson()
    {
        var builder = new StringBuilder(4096);
        builder.Append("{\n  \"status\": ")
            .Append(JsonString(SimplifiedRoomStatus))
            .Append(",\n  \"equivalenceMode\": ")
            .Append(JsonString(
                SimplifiedRoomMethod.StartsWith(
                    "fully_blind_end_to_end_current_mesh",
                    StringComparison.Ordinal)
                    ? "end_to_end_blind_current_frozen_mesh"
                    : "legacy_or_incomplete_pipeline"))
            .Append(",\n  \"selectedMethod\": ")
            .Append(JsonString(SimplifiedRoomMethod))
            .Append(",\n  \"selectedParameter\": ")
            .Append(JsonString(SimplifiedRoomParameter))
            .Append(",\n  \"selectionScore\": ");
        AppendJsonFloat(builder, SimplifiedRoomSelectionScore);
        builder.Append(",\n  \"maskIoU\": ");
        AppendJsonFloat(builder, SimplifiedRoomMaskIoU);
        builder.Append(",\n  \"candidateCount\": ")
            .Append(SimplifiedRoomCandidateCount)
            .Append(",\n  \"wallCount\": ")
            .Append(SimplifiedRoomWallCount)
            .Append(",\n  \"boundaryFeatureCount\": ")
            .Append(SimplifiedRoomAttachedPierCount)
            .Append(",\n  \"floorY\": ");
        AppendJsonFloat(builder, SimplifiedRoomFloorHeight);
        builder.Append(",\n  \"ceilingY\": ");
        AppendJsonFloat(builder, SimplifiedRoomCeilingHeight);
        builder.Append(",\n  \"axisA\": ");
        AppendJsonVector3(builder, _runtimeRoomAxisA);
        builder.Append(",\n  \"axisB\": ");
        AppendJsonVector3(builder, _runtimeRoomAxisB);
        builder.Append(",\n  \"outlineAxisCoordinates\": [");
        for (int i = 0; i < _runtimeRoomOutline.Count; i++)
        {
            if (i > 0) builder.Append(',');
            Vector3 point = _runtimeRoomOutline[i];
            builder.Append('[');
            AppendJsonFloat(builder, Vector3.Dot(
                point, _runtimeRoomAxisA));
            builder.Append(',');
            AppendJsonFloat(builder, Vector3.Dot(
                point, _runtimeRoomAxisB));
            builder.Append(']');
        }
        builder.Append("]\n}\n");
        return builder.ToString();
    }

    private void AppendClusterFamily(StringBuilder builder, string name,
                                     List<PlaneReference> references,
                                     bool horizontal)
    {
        List<PlaneCluster> clusters = BuildClusters(references);
        builder.Append("    {\"name\":").Append(JsonString(name))
            .Append(",\"horizontal\":")
            .Append(horizontal ? "true" : "false")
            .Append(",\"clusters\":[");
        for (int i = 0; i < clusters.Count; i++)
        {
            PlaneCluster cluster = clusters[i];
            if (i > 0) builder.Append(',');
            builder.Append("{\"coordinate\":");
            AppendJsonFloat(builder, cluster.coordinate);
            builder.Append(",\"supportArea\":");
            AppendJsonFloat(builder, cluster.weight);
            builder.Append(",\"sampleIndices\":[");
            for (int j = 0; j < cluster.sampleIndices.Count; j++)
            {
                if (j > 0) builder.Append(',');
                builder.Append(cluster.sampleIndices[j]);
            }
            builder.Append("]}");
        }
        builder.Append("]}");
    }

    private void AppendGridFamily(StringBuilder builder, string name,
                                  List<PlaneReference> references,
                                  Vector3 axisU, Vector3 axisV,
                                  ref bool first)
    {
        List<PlaneCluster> clusters = BuildClusters(references);
        for (int i = 0; i < clusters.Count; i++)
        {
            GridData grid = BuildSupportGrid(
                clusters[i].sampleIndices, axisU, axisV);
            if (grid == null) continue;
            if (!first) builder.Append(",\n");
            first = false;
            int width = grid.cells.GetLength(0);
            int height = grid.cells.GetLength(1);
            builder.Append("    {\"family\":")
                .Append(JsonString(name)).Append(",\"coordinate\":");
            AppendJsonFloat(builder, clusters[i].coordinate);
            builder.Append(",\"cellSize\":");
            AppendJsonFloat(builder, grid.cellSize);
            builder.Append(",\"minU\":");
            AppendJsonFloat(builder, grid.minU);
            builder.Append(",\"minV\":");
            AppendJsonFloat(builder, grid.minV);
            builder.Append(",\"width\":").Append(width)
                .Append(",\"height\":").Append(height)
                .Append(",\"occupiedRuns\":[");
            bool firstRun = true;
            for (int y = 0; y < height; y++)
            {
                int x = 0;
                while (x < width)
                {
                    while (x < width && !grid.cells[x, y]) x++;
                    if (x >= width) break;
                    int start = x;
                    while (x < width && grid.cells[x, y]) x++;
                    if (!firstRun) builder.Append(',');
                    firstRun = false;
                    builder.Append('[').Append(y).Append(',')
                        .Append(start).Append(',').Append(x).Append(']');
                }
            }
            builder.Append("]}");
        }
    }

    private static void AppendJsonVector3(StringBuilder builder,
                                          Vector3 value)
    {
        builder.Append('[');
        AppendJsonFloat(builder, value.x);
        builder.Append(',');
        AppendJsonFloat(builder, value.y);
        builder.Append(',');
        AppendJsonFloat(builder, value.z);
        builder.Append(']');
    }

    private static void AppendJsonFloat(StringBuilder builder, float value)
    {
        builder.Append(float.IsFinite(value)
            ? value.ToString("R", CultureInfo.InvariantCulture)
            : "null");
    }

    private static string JsonString(string value)
    {
        if (value == null) return "null";
        return "\"" + value.Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\r", "\\r")
            .Replace("\n", "\\n") + "\"";
    }
}

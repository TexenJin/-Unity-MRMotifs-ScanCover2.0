using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(ScanCoverDepthGridPointCloud))]
[CanEditMultipleObjects]
public sealed class ScanCoverDepthGridPointCloudEditor : Editor
{
    private static bool showDisplayAdvanced;
    private static bool showSurfaceAdvanced;
    private static bool showDebugAdvanced;

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        DrawScriptField();
        DrawRuntimeStatus();
        DrawRuntimeControls();
        EditorGUILayout.Space(6f);

        DrawCoreReferences();
        DrawPrimaryGrid();
        DrawVisibleOutput();
        DrawDepthAndFilter();
        DrawSnapshotFrame();
        DrawDebugMarkers();

        EditorGUILayout.Space(8f);
        DrawAdvancedSections();
        DrawArchiveNotice();

        serializedObject.ApplyModifiedProperties();
    }

    private void DrawScriptField()
    {
        using (new EditorGUI.DisabledScope(true))
        {
            MonoScript script = MonoScript.FromMonoBehaviour((ScanCoverDepthGridPointCloud)target);
            EditorGUILayout.ObjectField("Script", script, typeof(MonoScript), false);
        }
    }

    private void DrawRuntimeStatus()
    {
        if (targets.Length != 1)
            return;

        ScanCoverDepthGridPointCloud component = (ScanCoverDepthGridPointCloud)target;
        using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
        {
            EditorGUILayout.LabelField("Runtime Status", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Visible Points", component.VisibleCount.ToString());
            EditorGUILayout.LabelField("Surface Triangles", component.SurfaceTriangleCount.ToString());
            EditorGUILayout.LabelField("Frame", component.FrameIndex.ToString());
            EditorGUILayout.LabelField("Pending Readback", component.HasPendingReadback ? "Yes" : "No");
            if (!string.IsNullOrEmpty(component.LastIssue))
                EditorGUILayout.HelpBox(component.LastIssue, MessageType.Warning);
        }
    }

    private void DrawRuntimeControls()
    {
        using (new EditorGUILayout.HorizontalScope())
        {
            using (new EditorGUI.DisabledScope(!Application.isPlaying))
            {
                if (GUILayout.Button("Refresh"))
                {
                    foreach (Object item in targets)
                        ((ScanCoverDepthGridPointCloud)item).RefreshNow();
                }

                if (GUILayout.Button("Clear"))
                {
                    foreach (Object item in targets)
                        ((ScanCoverDepthGridPointCloud)item).ClearRuntimeState();
                }
            }
        }
    }

    private void DrawCoreReferences()
    {
        BeginSection("Core References");
        Draw("preprocessor");
        Draw("previewDisplayVisible");
        Draw("updateEveryFrame");
        EndSection();
    }

    private void DrawPrimaryGrid()
    {
        BeginSection("Primary Depth Grid");
        Draw("samplingMode");
        Draw("regularGridMaxColumns", "Columns");
        Draw("regularGridMaxRows", "Rows");
        Draw("stridePixels");
        Draw("stridePixelsX");
        Draw("stridePixelsY");
        Draw("centerRegularGridWindow");
        Draw("centerRegularGridWindowOnHeadsetForward");
        Draw("regularGridUseViewportCoverage");
        Draw("regularGridViewportCoverageScale");
        EndSection();
    }

    private void DrawVisibleOutput()
    {
        BeginSection("Visible Output");
        Draw("showGridLines");
        Draw("showGridOuterContourOnly");
        Draw("showRectilinearFillInsideContour");
        Draw("showGridTriangulation");
        Draw("showGridInteriorMesh");
        Draw("gridInteriorDisplayMode");
        Draw("showSurfaceMesh");
        Draw("showHeightSliceContour");
        Draw("heightSliceUseFrozenScreenCenterHeight");
        Draw("heightSliceRowCount");
        Draw("heightSliceShowPerpendicularColumns");
        Draw("heightSliceColumnCount");
        Draw("showHeightSlicePlaneFrame");
        Draw("heightSliceShowSampleColumnPlaneFrames");
        Draw("heightSliceSampleColumnPlaneFrameCount");
        Draw("showCandidatePlaneObjects");
        Draw("showMarkers");
        EditorGUILayout.Space(3f);
        Draw("gridLineColor");
        Draw("gridLineMaterialOverride");
        Draw("gridLineSurfaceOffsetMeters");
        Draw("heightSliceEpsilonMeters");
        Draw("heightSliceMaxSegmentMeters");
        Draw("heightSliceLineWidthMeters");
        Draw("heightSliceContourColor");
        Draw("surfaceColor");
        Draw("surfaceMaterialOverride");
        EndSection();
    }

    private void DrawDepthAndFilter()
    {
        BeginSection("Depth Validity");
        Draw("minConfidence");
        Draw("minLinearDepthMeters");
        Draw("maxLinearDepthMeters");
        Draw("requireValidNormal");
        Draw("neighborFill");
        Draw("neighborRadiusPixels");
        Draw("surfaceBiasMeters");
        EndSection();
    }

    private void DrawSnapshotFrame()
    {
        BeginSection("Snapshot Frame");
        Draw("useWorldSpaceDisplayRoots");
        Draw("lockUnfrozenDisplayRoll");
        Draw("lockUnfrozenDisplayPitch");
        Draw("lockUnfrozenDisplayYaw");
        Draw("compensateRegularGridRollSampling");
        EndSection();
    }

    private void DrawDebugMarkers()
    {
        BeginSection("Main Debug Markers");
        Draw("showHeadsetScreenCenterMarker");
        Draw("showRawDepthScreenCenterMarker");
        Draw("showOriginalGridCenterMarker");
        Draw("showCenterDebugMarkers");
        Draw("centerDebugMarkerScaleMeters");
        EndSection();
    }

    private void DrawAdvancedSections()
    {
        showDisplayAdvanced = DrawFoldout(showDisplayAdvanced, "Advanced Grid Lines");
        if (showDisplayAdvanced)
        {
            using (new EditorGUI.IndentLevelScope())
            {
                Draw("rectifyGridLinesAfterDepthWrap");
                Draw("rectilinearRennetStride");
                Draw("rectilinearRennetMinNormalDot");
                Draw("rectilinearRennetMaxEdgeSpanMultiplier");
                Draw("rectifiedGridLineMaxSpanMultiplier");
                Draw("rectifiedGridLineMinValidRatio");
                Draw("gridLineRequireCompleteCellSupport");
                Draw("gridLineMinCompleteCellIslandCount");
                Draw("gridLineKeepLargestCompleteCellIslands");
                Draw("gridLinesRenderBehindCandidatePatch");
                Draw("syncGridLinesToFocusedCandidate");
                Draw("focusedGridPlaneToleranceMeters");
                Draw("focusedGridExpandMeters");
            }
        }

        showSurfaceAdvanced = DrawFoldout(showSurfaceAdvanced, "Advanced Surface Mesh");
        if (showSurfaceAdvanced)
        {
            using (new EditorGUI.IndentLevelScope())
            {
                Draw("keepSurfaceMeshAvailableWhenHidden");
                Draw("useIndexConnectivity");
                Draw("maxEdgeLengthMeters");
                Draw("minNeighborNormalDot");
                Draw("surfaceDoubleSided");
                Draw("colorizeSurfaceRegions");
                Draw("showIrregularSurfaceBucket");
                Draw("irregularSurfaceBucketAlpha");
                Draw("surfaceRegionCreaseAngleDegrees");
                Draw("surfaceRegionMinQuadCount");
                Draw("surfaceNormalPatchMergeAngleDegrees");
                Draw("surfaceNormalFamilyAngleDegrees");
                Draw("surfaceNormalAxisBucketMinDot");
                Draw("surfaceLargeComponentMinTriangleCount");
                Draw("surfaceLargeComponentKeepRatio");
                Draw("surfaceRegionMaxNeighborDistanceMeters");
                Draw("surfaceRegionMaxPlaneOffsetMeters");
                Draw("surfaceRegionColorSaturation");
                Draw("surfaceRegionColorValue");
                EditorGUILayout.Space(3f);
                Draw("isolateTopCandidateSurfaces");
                Draw("rebuildLargestCandidateAsRegularGrid");
                Draw("largestCandidateUseTriangularLattice");
                Draw("largestCandidateUseOriginalGridTerrain");
                Draw("largestCandidateProjectRegularGridToMeshTerrain");
                Draw("largestCandidateShowFill");
                Draw("largestCandidateGridCellSizeMeters");
                Draw("largestCandidateGridMaxColumns");
                Draw("largestCandidateGridMaxRows");
            }
        }

        showDebugAdvanced = DrawFoldout(showDebugAdvanced, "Advanced Debug");
        if (showDebugAdvanced)
        {
            using (new EditorGUI.IndentLevelScope())
            {
                Draw("showRuntimeAxisDebug");
                Draw("runtimeAxisOffset");
                Draw("runtimeAxisCubeSize");
                Draw("runtimeAxisLength");
                Draw("runtimeAxisThickness");
                Draw("markerPrefab");
                Draw("markerParent");
                Draw("orientToNormal");
                Draw("fallbackMarkerScaleMeters");
                Draw("confidenceGradient");
                Draw("showScreenCenterDebugMarker");
                Draw("showFocusedGridCenterDebugMarker");
                Draw("showPatchCenterDebugMarker");
                Draw("centerDebugSurfaceOffsetMeters");
                Draw("screenCenterDebugColor");
                Draw("gridCenterDebugColor");
                Draw("patchCenterDebugColor");
                Draw("headsetScreenCenterMarkerDistanceMeters");
                Draw("headsetScreenCenterMarkerScaleMeters");
                Draw("headsetScreenCenterMarkerColor");
                Draw("rawDepthScreenCenterMarkerScaleMeters");
                Draw("rawDepthScreenCenterMarkerColor");
                Draw("originalGridCenterMarkerScaleMeters");
                Draw("originalGridCenterMarkerColor");
                Draw("showSurfaceNormalIndicators");
                Draw("surfaceNormalIndicatorLengthMeters");
                Draw("surfaceNormalIndicatorThicknessMeters");
                Draw("surfaceNormalIndicatorColor");
                Draw("debugLog");
                Draw("logBounds");
                Draw("dumpRosterOnceOnPlay");
                Draw("dumpOnlyValidCellsInRoster");
            }
        }
    }

    private void DrawArchiveNotice()
    {
        EditorGUILayout.Space(6f);
        EditorGUILayout.HelpBox(
            "Archived experiments are intentionally hidden from this working Inspector. " +
            "They are kept in Assets/MRMotifs/ScanCover/Extensions/ArchivedDepthGridFeatures and can be restored when needed.",
            MessageType.Info);
    }

    private static void BeginSection(string title)
    {
        EditorGUILayout.Space(5f);
        using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
        {
            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);
        }
    }

    private static void EndSection()
    {
        EditorGUILayout.Space(2f);
    }

    private bool DrawFoldout(bool value, string title)
    {
        EditorGUILayout.Space(3f);
        return EditorGUILayout.Foldout(value, title, true, EditorStyles.foldoutHeader);
    }

    private void Draw(string propertyName, string label = null)
    {
        SerializedProperty property = serializedObject.FindProperty(propertyName);
        if (property == null)
            return;

        if (string.IsNullOrEmpty(label))
            EditorGUILayout.PropertyField(property, true);
        else
            EditorGUILayout.PropertyField(property, new GUIContent(label), true);
    }
}

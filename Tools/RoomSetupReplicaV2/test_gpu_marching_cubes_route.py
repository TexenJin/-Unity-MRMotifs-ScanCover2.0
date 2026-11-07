#!/usr/bin/env python3
"""Static contracts for the isolated RoomSetup GPU Marching Cubes route."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOM = REPO / "Assets" / "RoomSetupReplicaV2"


def read(relative: str) -> str:
    return (ROOM / relative).read_text(encoding="utf-8")


def test_backend_is_dedicated_marching_cubes() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    for forbidden in (
        "GPUSurfaceNets", "GPUMeshRenderer", "SurfaceNets", "Foundation"
    ):
        assert forbidden not in backend
    assert 'Resources.Load<ComputeShader>(\n                "RoomSetupReplicaV2GpuMarchingCubes")' in backend
    assert "Graphics.RenderPrimitivesIndirect" in backend
    assert "MaximumDepthMeters = 3.00f" in backend


def test_fusion_has_surface_and_voxel_hard_range() -> None:
    fusion = read("Resources/RoomSetupReplicaV2GpuFusion.compute")
    assert "voxelRange > _MaximumDepthMeters" in fusion
    assert "surfaceRange > _MaximumDepthMeters" in fusion
    assert "LayerAnchor" not in fusion
    assert "LayerPair" not in fusion


def test_v011_truncation_gradient_and_weighted_fusion_contract() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    fusion = read("Resources/RoomSetupReplicaV2GpuFusion.compute")
    assert "PositiveTruncationScale = 0.20f" in backend
    assert "NegativeTruncationScale = 0.20f / 3f" in backend
    assert "GradientThresholdSquared = 0.01f" in backend
    assert "MaximumWeight = 1000f" in backend
    assert "float positive = _PositiveTruncationScale * factor" in fusion
    assert "float negative = _NegativeTruncationScale * factor" in fusion
    assert "dx * dx + dy * dy <= _GradientThresholdSquared" in fusion
    assert "min(accumulated.y, _MaximumWeight - observation.weight)" in fusion
    assert "FuseV011Observation(accumulated, left)" in fusion
    assert "FuseV011Observation(accumulated, right)" in fusion
    for removed in (
        "PassFivePixelDepthBand", "firstWriteRisk", "pairedCoherent",
        "_MinimumViewBaselineMeters", "_ChallengerVolume",
        "_ChallengerMetaVolume", "_MatureCorrectionVolume",
        "MatureResidualDeadband", "MatureSignHysteresis",
        "LayerAnchor", "LayerPair",
    ):
        assert removed not in fusion


def test_mc_uses_current_tsdf_without_later_vertex_history() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    fusion = read("Resources/RoomSetupReplicaV2GpuFusion.compute")
    mc = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.compute")
    for removed in (
        "UpdateEdgeHistory", "EdgeHistory", "edgeHistory", "MatureWeight",
        "VertexNormalDeadzone", "VertexTangentialAlpha",
        "LastHistoryObservationPair",
    ):
        assert removed not in backend
        assert removed not in mc
    assert "edgePositions[edge] = lerp(" in mc
    assert "positions[endpoints.x], positions[endpoints.y], alpha" in mc
    assert "MatureResidual" not in fusion


def test_world_tiles_follow_current_view_without_moving_old_tsdf() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    assert "TileSampleCount = 31" in backend
    assert "TileStepSamples = TileCellCount * MeshStride" in backend
    assert "(leftEye + rightEye) * 0.5f," in backend
    assert "MaximumDepthMeters + TileActivationPaddingMeters" in backend
    assert "foreach (GpuTile tile in _tiles.Values)" in backend
    assert '"_VolumeOrigin", tile.volumeOrigin' in backend
    assert "ReleaseAllTiles();" in backend
    voxel_metres = 0.05
    tile_cells = (31 - 1) // 2
    tile_step = tile_cells * 2 * voxel_metres
    assert tile_step == 1.5
    assert 0 * tile_step + 30 * voxel_metres == 1 * tile_step


def test_mc_reproduces_aligned_v011_cell_contract() -> None:
    mc = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.compute")
    assert "int3 sourceBase = (int3)id * 2" in mc
    assert "sourceBase + CubeOffsets[corner] * 2" in mc
    assert "if (!(tsdfSample.y > 0.0)" in mc
    assert "CellHasCompatibleLayerSupport" not in mc
    assert "SameLayerMaximumSeparation" not in mc
    assert "LayerAnchor" not in mc
    assert "LayerPair" not in mc
    assert "_CandidateTriangleVertices[outputBase] = edgePositions[edge2]" in mc
    assert ("_CandidateTriangleVertices[outputBase + 2u] = "
            "edgePositions[edge0]" in mc)


def test_global_remove_small_contract_is_gpu_resident() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    mc = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.compute")
    wire = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.shader")
    assert "RemoveSmallTriangleThreshold = 32" in backend
    assert "RemoveSmallTriangleGate = 40000" in backend
    assert "_SeamWeldUnitsPerMeter" in mc
    assert "WeldQuantizedPositions" in mc
    assert "PositionHash" in mc
    assert "UnionConnectivityRoots" in mc
    assert "_ComponentTriangleCounts[root] <" in mc
    assert "triangleCount < _RemoveSmallTriangleGate" in mc
    assert "DispatchIndirect" in backend
    assert "_RoomSetupMcCounters[6] != 0u" in wire
    assert "uint triangle =" not in mc


def test_teacher_scene_needs_no_old_gpu_asset_bindings() -> None:
    teacher = read("Scenes/RoomSetupReplicaV2.unity")
    for field in ("gpuFusionCompute", "gpuSurfaceNetsCompute", "gpuMeshMaterial"):
        assert field not in teacher


def test_sequential_environment_depth_b_is_exclusive_and_reuses_tsdf_mc() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    reconstruction = read("Runtime/RoomSetupReplicaV2Reconstruction.cs")
    scanner = read(
        "B/Runtime/RoomSetupReplicaV2EnvironmentDepthScanner.cs")
    fusion = read("Resources/RoomSetupReplicaV2GpuFusion.compute")
    builder = read("B/Editor/RoomSetupReplicaV2BBuilder.cs")
    b_scene = read(
        "B/Scenes/RoomSetupReplicaV2EnvironmentDepthTeacher.unity")
    source_scene = (REPO / "Assets" / "RoomSetupReplicaV2" / "Scenes" /
                    "RoomSetupReplicaV2.unity").read_text(encoding="utf-8")

    assert "TryObserveEnvironmentDepthGlobal(Camera camera)" in backend
    assert 'Shader.GetGlobalTexture(\n            "_EnvironmentDepthTexture")' in backend
    assert 'FindKernel("IntegrateStereo")' in backend
    assert '"IntegrateEnvironmentDepth")' in backend
    assert "_UseEnvironmentDepthGlobal" not in fusion
    assert "TryObserveEnvironmentDepthGlobal(Camera camera)" in reconstruction
    assert ("B-environment-depth-global-final-v0.1.2-gpu-mc-hard3m-v1"
            in reconstruction)
    assert "EnvironmentDepthManager" in scanner
    assert "FindObjectsByType<DepthCapture>" in scanner
    assert "_EnvironmentDepthReprojectionMatrices" in fusion
    assert "LoadEnvironmentProjectiveDepth" in fusion
    assert "DestroyAll<DepthCapture>();" in builder
    assert "DestroyAll<AROcclusionManager>();" in builder
    assert "RoomSetupReplicaV2BBuilder" in builder
    assert 'PlayerSettings.bundleVersion = "0.1.2"' in builder
    assert "BuildApk" in builder
    assert "_enableDynamicResolution: 1" in b_scene
    assert "_enableDynamicResolution: 1" in source_scene
    assert "quest3MaxDynamicResolutionScale: 1.6" in b_scene


def test_unique_edge_display_is_render_only() -> None:
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    reconstruction = read("Runtime/RoomSetupReplicaV2Reconstruction.cs")
    shader = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.shader")
    mc = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.compute")
    b_input = read(
        "B/Runtime/RoomSetupReplicaV2EnvironmentDepthInput.cs")

    assert "SetUniqueEdgeDisplay(bool enabled)" in backend
    assert "ToggleUniqueEdgeDisplay()" in reconstruction
    assert "BuildUniqueEdgeMask" in shader
    assert "ClearUniqueEdgeState" in mc
    assert "BuildUniqueEdgeArgs" in mc
    assert "ClaimUniqueEdge" in mc
    assert "CanonicalEdge" in mc
    assert "_UniqueEdgeHashOwners" in mc
    assert "_RoomSetupMcUniqueEdgeMasks" in shader
    assert "_UniqueEdgesOnly" in shader
    assert "ToggleUniqueEdgeDisplay" in b_input
    assert "RoomSetupReplicaV2GpuFusion.compute" not in shader


def test_wire_width_uses_inverse_distance_display_scale() -> None:
    shader = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.shader")
    backend = read("Runtime/RoomSetupReplicaV2GpuBackend.cs")
    fusion = read("Resources/RoomSetupReplicaV2GpuFusion.compute")
    mc = read("Resources/RoomSetupReplicaV2GpuMarchingCubes.compute")

    assert ('_LineReferenceDistance ("Full Width Distance", Float) = 0.5'
            in shader)
    assert "GetCameraPositionWS(), input.positionWS" in shader
    assert "_LineReferenceDistance /" in shader
    assert "max(distanceWS, 0.0001)" in shader
    assert "float subpixelCoverage = min(widthPixels, 1.0)" in shader
    assert "coverage *= subpixelCoverage" in shader
    for production_source in (backend, fusion, mc):
        assert "_LineReferenceDistance" not in production_source


def main() -> int:
    test_backend_is_dedicated_marching_cubes()
    test_fusion_has_surface_and_voxel_hard_range()
    test_v011_truncation_gradient_and_weighted_fusion_contract()
    test_mc_uses_current_tsdf_without_later_vertex_history()
    test_world_tiles_follow_current_view_without_moving_old_tsdf()
    test_mc_reproduces_aligned_v011_cell_contract()
    test_global_remove_small_contract_is_gpu_resident()
    test_teacher_scene_needs_no_old_gpu_asset_bindings()
    test_sequential_environment_depth_b_is_exclusive_and_reuses_tsdf_mc()
    test_unique_edge_display_is_render_only()
    test_wire_width_uses_inverse_distance_display_scale()
    print("11 isolated GPU Marching Cubes route tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

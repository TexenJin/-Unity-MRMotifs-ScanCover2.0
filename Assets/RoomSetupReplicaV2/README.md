# RoomSetup Replica V2 — v0.1.3

This folder is the complete, isolated production delivery unit. No alternate
route, replay, CPU reconstruction, Surface Nets, FusionGuard, point-cloud
probe, or offline training route remains in the production directory.

## Production route

- Input: Meta `EnvironmentDepthManager` global `_EnvironmentDepthTexture`.
- Fusion and geometry: `Runtime/Core/` and `Resources/` contain the production GPU TSDF
  and GPU Marching Cubes implementation.
- Range: hard 3 m admission boundary; unsupported space remains empty.
- Mesh cleanup: GPU-resident global `RemoveSmall<32` component cleanup.
- Display: cyan wireframe with inverse-distance stroke width and optional unique-edge display.

The GPU backend accepts only the global Environment Depth route. The previous
DepthCapture/A entry point and its compute kernel have been removed.

## Contents

- `Scenes/RoomSetupReplicaV2EnvironmentDepth.unity`: generated production scene.
- `Runtime/RoomSetupReplicaV2EnvironmentDepthScanner.cs`: depth owner and frame scheduler.
- `Runtime/RoomSetupReplicaV2EnvironmentDepthInput.cs`: controller input.
- `Runtime/Core/RoomSetupReplicaV2Reconstruction.cs`: production lifetime and controls.
- `Runtime/Core/RoomSetupReplicaV2GpuBackend.cs`: GPU TSDF, MC, cleanup and indirect rendering.
- `Runtime/Core/RoomSetupReplicaV2McTables.cs`: local Marching Cubes lookup table.
- `Resources/`: the four GPU fusion, extraction and display assets.
- `Editor/RoomSetupReplicaV2Builder.cs`: reproducible scene preparation and Android build entry.

## Build

In Unity, use:

`Codex > RoomSetup Replica V2 > Build APK`

The builder validates and builds the production scene directly. It no longer
copies or mutates an A/Teacher source scene. `ProjectSettings/EditorBuildSettings.asset`
contains this scene as the sole build scene.

## Frozen outcome

On 2026-11-08 this production route was frozen for its calmer presentation and closer resemblance to native RoomSetup. Unity packaging and final Quest acceptance remain user-owned.

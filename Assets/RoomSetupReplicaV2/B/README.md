# RoomSetupReplicaV2 B — Final v0.1.2

This folder is the isolated delivery unit for the user-selected B route.

## Production route

- Input: Meta `EnvironmentDepthManager` global `_EnvironmentDepthTexture`.
- Fusion and geometry: shared `Assets/RoomSetupReplicaV2` GPU TSDF and GPU Marching Cubes implementation.
- Range: hard 3 m admission boundary; unsupported space remains empty.
- Mesh cleanup: GPU-resident global `RemoveSmall<32` component cleanup.
- Display: cyan wireframe with inverse-distance stroke width and optional unique-edge display.

The shared TSDF/MC implementation deliberately remains outside this folder so A and B cannot silently fork their production geometry rules. B-specific scene ownership, input, controls, and build tooling live here.

## Contents

- `Scenes/RoomSetupReplicaV2EnvironmentDepthTeacher.unity`: generated, buildable B scene.
- `Runtime/RoomSetupReplicaV2EnvironmentDepthScanner.cs`: B depth owner and frame scheduler.
- `Runtime/RoomSetupReplicaV2EnvironmentDepthInput.cs`: B controller input.
- `Editor/RoomSetupReplicaV2BBuilder.cs`: reproducible scene preparation and Android build entry.

## Build

In Unity, use:

`Codex > RoomSetup Replica V2 B > Build APK`

The builder recreates the B scene from the shared RoomSetupReplicaV2 source scene before each build, then removes competing depth owners and installs the B Environment Depth route. The generated scene stays inside this folder.

## Frozen outcome

On 2026-11-08 the user selected B over A on Quest 3 for its calmer presentation and closer resemblance to native RoomSetup. Unity packaging and final Quest acceptance remain user-owned.

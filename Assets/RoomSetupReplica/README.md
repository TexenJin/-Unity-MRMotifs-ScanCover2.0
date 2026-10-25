# Room Setup Replica

This folder is intentionally independent from the existing ScanCover and CHSL
production/diagnostic routes.

Use Unity menu `Room Setup Replica > 重建独立测试场景`. The command replaces the
placeholder scene with a Meta XR Building Blocks scene, adds only `DepthCapture`,
`PassthroughCameraProvider`, `RoomSetupReplicaScanner`, and
`RoomSetupReplicaReconstruction`, then makes that scene the only build scene.

Quest controls:

- Right index trigger: pause/resume capture.
- B: clear the replica volume and mesh.
- X: show/hide the replica mesh.

The current replica implements the statically recovered active chain: 5 cm
sparse projective SDF, central-gradient rejection at 0.1 m, asymmetric
depth-scaled truncation, inverse-square weights capped at 1000, view-dependent
half-precision voxel storage, view-dependent garbage collection, 10 cm
resampled Marching Cubes, adaptive single-flight mesh cadence, and front/back
mesh publication. The unresolved native uncertainty channel is not guessed or
substituted.

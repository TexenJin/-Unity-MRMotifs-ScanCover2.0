# Room Setup Replica V2

This folder is intentionally independent from the existing ScanCover and CHSL
production/diagnostic routes.

Use Unity menu `Room Setup Replica V2 > 重建独立测试场景`. The command replaces the
placeholder scene with a Meta XR Building Blocks scene, adds only `DepthCapture`,
`PassthroughCameraProvider`, `RoomSetupReplicaV2Scanner`, and
`RoomSetupReplicaV2Reconstruction`, then makes that scene the only build scene.

Quest controls:

- Right index trigger: pause/resume capture.
- A: start/stop a bounded exact-input recording (up to 180 stereo pairs).
- Y: pause live capture, clear the current replica, and replay the last tape.
- B: clear the replica volume and mesh.
- X: show/hide the replica mesh.

V2 keeps V1 intact and changes two deliberately isolated parts:

- It records the exact 32-bit NDC depth accepted from both eyes, together with
  that packet's projection/view matrices. File writes use a bounded background
  queue; storage pressure drops a recording pair instead of stalling fusion.
- The 10 cm mesh field is no longer made by directly taking every second 5 cm
  voxel. Each lattice sample is rebuilt from all eight surrounding source
  voxels with source-confidence weights. A missing source remains unknown, so
  the mesh extractor cannot invent support across an unobserved gap.

Everything else stays on the recovered active chain: 5 cm sparse projective
SDF, central-gradient rejection at 0.1 m, asymmetric depth-scaled truncation,
inverse-square weights capped at 1000, half-precision voxel storage,
view-dependent garbage collection, 10 cm Marching Cubes, adaptive
single-flight mesh cadence, and front/back publication. The unresolved native
uncertainty channel is not guessed or substituted.

Artifacts on Quest are written below
`Application.persistentDataPath/RoomSetupReplicaV2/`:

- `last-session.rsv2`: deterministic stereo input tape.
- `last-replay-receipt.json`: replayed-pair, source-field, derived-field and
  final mesh counts after the mesh queue has drained.

This makes parameter or topology changes comparable on identical input. It
does not claim bit-identical output to Meta Room Setup, and it does not replace
Quest device A/B validation.

# Room Setup Replica V2

This folder is intentionally independent from the existing ScanCover and CHSL
production/diagnostic routes.

Use Unity menu `Room Setup Replica V2 > 重建独立测试场景`. The command replaces the
placeholder scene with a Meta XR Building Blocks scene, adds only `DepthCapture`,
`PassthroughCameraProvider`, `RoomSetupReplicaV2Scanner`, and
`RoomSetupReplicaV2Reconstruction`, then makes that scene the only build scene.

Quest controls:

- Right index trigger: start/stop coverage and its bounded exact-input
  recording together (up to 180 accepted stereo pairs).
- A: stop live coverage, clear the current replica, and replay the last tape.
- B: clear the replica volume and mesh.
- X: show/hide the replica mesh.

The application starts idle. Recording is armed by the same trigger press as
coverage and begins on the first accepted stereo depth pair, so visual coverage
cannot silently run without its matching tape. The second trigger press closes
the tape before stopping capture.

V2 keeps V1 intact and changes two deliberately isolated parts:

- It records the exact 32-bit NDC depth accepted from both eyes, together with
  that packet's projection/view matrices in Quest Stage space. File writes use
  a bounded background queue; storage pressure drops a recording pair instead
  of stalling fusion. Stage space keeps position and yaw tied to the room
  across application restarts; changing the Guardian/room boundary invalidates
  that relationship and requires a new tape.
- The 10 cm mesh field is no longer made by directly taking every second 5 cm
  voxel. Each lattice sample is rebuilt from all eight surrounding source
  voxels with source-confidence weights. A missing source remains unknown, so
  the mesh extractor cannot invent support across an unobserved gap.

The first device-backed scheduling correction raises the per-slice chunk guard
from 32 to 256 while retaining the 6 ms reconstruction deadline. The guard no
longer imposes an artificial multi-frame minimum on a stereo pair; the time
budget still bounds work on each Unity frame. Fusion, truncation, voxel values,
derived-field rules and topology are unchanged by this scheduling correction.

The next fusion optimization factors each chunk's regular 16x16x16 projective
coordinates into reusable X/Y/Z matrix terms. Per-voxel float addition order,
pixel selection, integration order and all reconstruction rules remain the
same; only repeated scalar multiplications are removed. The identical Stage
tape is the acceptance input for geometry and replay-time comparison.

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

Tape format 2 uses Quest Stage coordinates. Format-1 tapes stored a previous
Unity session's world coordinates and are rejected because their cross-session
translation and yaw cannot be reconstructed reliably; record once again after
installing this version.

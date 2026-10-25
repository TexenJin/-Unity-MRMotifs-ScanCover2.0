# Room Setup Replica V2

This folder is intentionally independent from the existing ScanCover and CHSL
production/diagnostic routes.

Use Unity menu `Room Setup Replica V2 > 重建独立测试场景`. The command replaces the
placeholder scene with a Meta XR Building Blocks scene, adds only `DepthCapture`,
`PassthroughCameraProvider`, `RoomSetupReplicaV2Scanner`, and
`RoomSetupReplicaV2Reconstruction`, then makes that scene the only build scene.

Quest controls:

- A: clear the current replica and replay the fixed last tape.
- Right index trigger, B and X are temporarily disabled in `replayOnlyMode`
  while the V2-B simplifier is compared on one immutable input.

All controller actions share a 250 ms debounce window so one physical press
cannot restart a replay or trigger two state transitions.

When recording controls are restored, capture writes to a temporary tape.
Only a clean recording containing at least 16 stereo pairs replaces
`last-session.rsv2`; a short, failed or interrupted recording preserves the
previous comparison tape.

V2 keeps V1 intact and changes two deliberately isolated parts:

- It records the exact 32-bit NDC depth accepted from both eyes, together with
  that packet's projection/view matrices in Quest Stage space. File writes use
  a bounded background queue; storage pressure drops a recording pair instead
  of stalling fusion. Stage space keeps position and yaw tied to the room
  across application restarts; changing the Guardian/room boundary invalidates
  that relationship and requires a new tape.
- V2-B follows the recovered active `voxelSizeRatio=0.5` path in native
  `0x78cd20`: each 10 cm derived lattice coordinate maps to an aligned even
  coordinate in the 5 cm source field, and the stored half SDF/weight are
  copied exactly. A missing aligned source stays unobserved. The function's
  trilinear branch supports non-aligned ratios but is not active for `0.5`.
  Source fusion and derived meshing both use world-anchored integer grid
  points; the native `+0.5` terms belong to image pixels, not voxel-space
  sample positions. See `V2B_WHITEBOX_RESAMPLING.md` for the evidence boundary.
- After Marching Cubes, the recovered `RemoveSmallConnectedComponents` stage
  operates on the global candidate mesh: while the candidate has fewer than
  40,000 triangles it removes components with strictly fewer than 32
  triangles. Components spanning Chunk seams are joined for classification;
  exactly 32 triangles are retained. The raw candidate is preserved so a
  temporarily removed fragment can grow and reappear, and the filtered meshes
  are published together only when the mesh pass completes.
- The recovered XRT production stage is retained after `<32` cleanup on the
  complete seam-welded candidate, but is currently disabled. The requested
  output is the regular 10 cm coverage mesh, not the later performance mesh.
  When re-enabled in a future phase, it follows the observed directed-edge
  collapse envelope: minimum 100 input triangles, target `vertices / 20`,
  MT19937 seed 13, 16 unique directed candidates per batch, a `0.075 m`
  point-to-plane acceptance ceiling, near-zero optional regularizers loaded
  from float bit patterns `0x80 / 0x20`, manifold and normal-flip rejection
  (`dot >= 0.1`), the recovered open-fan endpoint-to-boundary-line distance,
  and at most 200 failed batches. The
  target is deliberately not forced; the error/topology rules may stop first.
  Surviving vertices remain at measured Marching-Cubes positions, so this is
  simplification rather than an unrelated smoothing pass. It runs on an
  immutable background snapshot with one active job. A due trigger while that
  job is busy is skipped and counted rather than queued; the current Front
  remains visible until a complete result is ready. While disabled, the
  post-`<32` mesh passes through unchanged. Until
  the exact native terminal topology rule is resolved, a
  non-empty cleaned input that simplifies to zero faces is rejected and the
  cleaned input is published unchanged; this is an explicit publication
  fail-safe rather than a substitute simplification metric.

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

V2-C garbage collection uses the recovered sparse-leaf granularity: each
16-cubed storage Chunk is partitioned into 4x4x4 leaves containing 64 source
voxels. Only leaves intersecting either current depth-eye frustum are tested;
a leaf survives when any weighted voxel has
`sdf < 0.3 m * clamp(viewDepth, 1, 2)`. Otherwise that leaf is cleared and its
mesh neighbourhood is rebuilt. The native GC trigger frequency is still
unresolved, so this experiment deliberately preserves the earlier
once-per-accepted-pair cadence and labels that boundary in the receipt.

Artifacts on Quest are written below
`Application.persistentDataPath/RoomSetupReplicaV2/`:

- `last-session.rsv2`: deterministic stereo input tape.
- `last-replay-receipt.json`: replayed-pair, source-field, derived-field and
  final mesh counts after the mesh queue has drained.
- `last-replay-mesh.ply`: deterministic binary little-endian final mesh in
  Quest Stage coordinates. The receipt includes its SHA-256 hash.

The replay receipt also reports the known-but-unavailable uncertainty rule,
adaptive interval constants, single-flight busy skips, mesh publication count,
leaf-GC mode/cadence and release counts,
first-mesh latency, local dirty-to-publication delay, replay throughput,
pre/post `<32` cleanup counts and time, XRT
pre/target/post counts, accepted collapses, rejection counts, stop reason and
background processing time, seam-welded connected components, and a local
planar residual over normal-consistent vertices. Direct-depth support
reprojects every final triangle against every stereo tape packet. A triangle
is supported when its centroid agrees with either eye's depth within 5 cm, or
at least two of its vertices agree within the same packet. The receipt keeps
this offline analysis time separate from replay throughput; TSDF weights are
not used as a substitute.

This makes parameter or topology changes comparable on identical input. It
does not claim bit-identical output to Meta Room Setup, and it does not replace
Quest device A/B validation.

The accepted Quest 3 receipt for the current geometry baseline is frozen in
`V2B_REGULAR_10CM_BASELINE.md`. V2-C uncertainty and lifecycle changes must use
that receipt as their before-state.

Tape format 2 uses Quest Stage coordinates. Format-1 tapes stored a previous
Unity session's world coordinates and are rejected because their cross-session
translation and yaw cannot be reconstructed reliably; record once again after
installing this version.

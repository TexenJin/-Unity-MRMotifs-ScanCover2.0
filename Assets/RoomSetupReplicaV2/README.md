# RoomSetup Replica V2 — v0.1.4

## v0.1.4 simplified-room archive

This release archives the improved simplified-room result currently accepted
on Quest.  After the scan is frozen, the runtime reads the current complete
Marching Cubes mesh and runs the full blind route on that same snapshot:

1. open 5 cm contour-stack evidence;
2. four graph-cut variants and three-of-four consensus;
3. rectilinear topology and mother-wall micro-step refit;
4. cross-height final-surface registration;
5. the double-sided transparent wall, floor and ceiling room shell.

No previous simplified outline, offline fitted polygon, annotated region or
intermediate JSON is accepted as runtime input.  End-to-end blind mode is
required by default.  If the frozen full-mesh readback or any blind contour
stage fails, the runtime leaves the simplified room absent and reports the
failed stage instead of silently publishing the legacy floor/ceiling fallback.
Successful exports identify themselves with
`equivalenceMode=end_to_end_blind_current_frozen_mesh`.

## Current weighted production selection and comparison routes

`RoomSetupReplicaV2EnvironmentDepth.unity` currently selects
`V013Weighted`. This restores the retained weighted-average TSDF route as the
active production comparison baseline. `RobustCandidateV1` and
`RobustCandidateV2` remain available as inactive rollback/analysis modes; they
are not deleted and changing the enum followed by a cleared restart is enough
to compare them again.

When either robust comparison mode is selected, Marching Cubes, its buffers,
cleanup kernels, renderer and export format remain the existing production
path. MC still reads one published TSDF. A second GPU volume holds a private
candidate distance, bounded support, six-direction view mask and contradiction
evidence; it is never read by MC.

- observations within 1.5 cm robustly converge the published distance and add
  maturity, with retained averaging authority capped at eight votes;
- changes of at least 2.5 cm enter the candidate rather than averaging directly
  into the published surface;
- repeated observations from one view add only 0.125 support, while a new
  cardinal view direction adds a full independent vote;
- farther observations carry stronger free-space contradiction authority;
  nearer observations use a stricter promotion threshold because they may be
  real occluders;
- unobserved and occluded voxels never decay;
- a candidate reaches MC only after both support and contradiction thresholds
  pass. Until then the former published surface remains MC's input.

The live mesh remains magenta (changing), yellow (settling), or cyan (mature),
and the existing continuous recording/export supplies the before/after evidence.
Color is maturity evidence, not proof that geometry is correct. Small-component
cleanup stays disabled in validation modes so it cannot hide unstable evidence.

The reconstruction component exposes an explicit safe rollback enum:

- `V013Weighted` — current active retained v0.1.3 weighted fusion;
- `RobustCandidateV1` — retained conservative candidate comparison;
- `RobustCandidateV2` — retained more responsive candidate comparison;
- `LatestSampleProbe` — the prior latest-sample temporal probe.

Changing the enum and clearing/restarting the scan is sufficient; no MC or
renderer asset needs to be replaced.

Pulling the right index trigger to start scanning now also starts a stability
recording. While the scan runs, complete GPU mesh snapshots are recorded with a
minimum interval of four depth integrations. Pressing the **left thumbstick**
stops new integration, finishes the last snapshot, seals the recording, and
then runs the normal final-room export. If the trigger temporarily pauses the
scan, the same recording waits and resumes rather than being discarded.

The depth source stays enabled during ordinary recording, but new integration
is briefly held while each asynchronous snapshot is copied so counters,
vertices and stability cannot come from different mesh epochs. Each frame
stores the complete non-indexed MC vertices, aligned per-vertex stability,
actual integration number, GPU counters and headset pose in a compact
little-endian `RSV2SB01` binary file. Its adjacent JSON describes that frame;
`manifest.json` describes the complete sequence. Recordings are written under:

`RoomSetupReplicaV2Exports/StabilityRecordings/<timestamp>/`

`stability_recording_status.json` in the export root is the ADB-readable receipt.
Only one GPU readback and one binary write may be in flight, so slow storage
causes frames to be skipped rather than building an unbounded memory queue.

This folder is the complete, isolated production delivery unit. No alternate
route, replay, CPU reconstruction, Surface Nets, FusionGuard, point-cloud
probe, or offline training route remains in the production directory.

## Production route

- Input: Meta `EnvironmentDepthManager` global `_EnvironmentDepthTexture`.
- Fusion and geometry: `Runtime/Core/` and `Resources/` contain the production GPU TSDF
  and GPU Marching Cubes implementation.
- Range: hard 3 m admission boundary; unsupported space remains empty.
- Fusion, truncation, admission range, tile scheduling, MC cadence and display
  remain identical to the tagged `v0.1.3` production route.
- Mesh admission and cleanup are the tagged `v0.1.3` behavior: any positive
  accumulated support is admissible, with GPU-resident `RemoveSmall<32`
  component cleanup for meshes below 40,000 triangles.
- Display: cyan wireframe with inverse-distance stroke width and optional unique-edge display.
- Observation: the former head-locked stereo depth window is retired from the
  production scene and is not created by the controller input path.

The GPU backend accepts only the global Environment Depth route. The previous
DepthCapture/A entry point and its compute kernel have been removed.

## Contents

- `Scenes/RoomSetupReplicaV2EnvironmentDepth.unity`: generated production scene.
- `Runtime/RoomSetupReplicaV2EnvironmentDepthScanner.cs`: depth owner and frame scheduler.
- `Runtime/RoomSetupReplicaV2EnvironmentDepthInput.cs`: controller input.
- `Runtime/RoomSetupReplicaV2CaptureExporter.cs`: left-thumbstick V2 capture
  exporter and compact in-headset status UI.
- `Runtime/Core/RoomSetupReplicaV2Reconstruction.cs`: production lifetime and controls.
- `Runtime/Core/RoomSetupReplicaV2GpuBackend.cs`: GPU TSDF, MC, cleanup and indirect rendering.
- `Runtime/Core/RoomSetupReplicaV2McTables.cs`: local Marching Cubes lookup table.
- `Resources/`: GPU fusion, extraction, and display assets.
- `Editor/RoomSetupReplicaV2Builder.cs`: reproducible scene preparation and Android build entry.

## Build

In Unity, use:

`Codex > RoomSetup Replica V2 > Build APK`

The builder validates and builds the production scene directly. It no longer
copies or mutates an A/Teacher source scene. `ProjectSettings/EditorBuildSettings.asset`
contains this scene as the sole build scene.

## Closed-room data export

The structural simplification preview is hidden by default, but extraction
continues in the background. After completing a room scan, press the **left
thumbstick**. The exporter pauses capture, locks one final GPU mesh snapshot,
refreshes the structural samples, and writes a timestamped package under:

The binding is explicit and exclusive: the physical left thumbstick click
exports. Runtime detection reads the same control through both OVRInput and
Unity XR so OpenXR controller mapping differences cannot silently drop the
click. The exporter writes `export_status.json` as soon as the click starts an
export, including the input source used. Meanwhile,
`RThumbstick + RTouch` freezes the current scan. Scene preparation rejects
continuous-move providers and `OVRPlayerController` so locomotion cannot take
ownership of the export button.

`Android/data/com.pcaii.scancover.quest3/files/RoomSetupReplicaV2Exports/`

The package includes the final OBJ/JSON mesh, GPU counters, structural
samples, plane clusters, support grids, planes before/after topology, and
capture/extractor metadata. `export_status.json` in the export root provides
a small ADB-readable success/failure receipt.

## Frozen outcome

On 2026-11-18 the current weighted mesh route and improved end-to-end blind
simplified-room route were archived as `v0.1.4`. Unity packaging and final
Quest acceptance remain user-owned.

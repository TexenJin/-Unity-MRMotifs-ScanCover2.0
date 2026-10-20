# Depth frame closure and CHISEL shadow route

## Production closure completed here

Each accepted observation now carries one indivisible application-side contract:

1. the Meta Environment Depth texture;
2. the matching provider reprojection matrix for the selected eye;
3. the observing eye position recovered from that same matrix;
4. the generated world-position, normal and validity textures;
5. a Unity-frame/realtime generation receipt.

The TSDF no longer combines a provider depth texture/matrix with the later
`Camera.main` render pose. Both stereo eyes must be prepared successfully from
the same refresh generation before either eye may mutate the production TSDF.
Partial stereo integration is rejected.

The Meta provider remains a black box: it does not expose a raw sensor timestamp
through this API. Therefore this is the strongest closure available at the
application boundary: texture, calibration and provider pose are owned by the
same provider publication, while the receipt records when that publication was
consumed by ScanCover.

## Independent CHISEL-route proof

`ScanCoverChiselSparseTsdfShadow` receives only retained, non-warmup stereo pairs
that already passed the production admission contract. It implements a separate
spatial hash of 16 x 16 x 16 voxel chunks, 5 cm voxels and a 15 cm truncation
band. It cannot read or write the production TSDF, extraction buffers, draw
arguments, visibility state or publication state.

This is an architecture and cost proof inspired by OpenChisel, not a copied or
bit-for-bit port. It deliberately uses a managed implementation and asynchronous
GPU readback so it can answer the next question before a native dependency is
adopted: does sparse, allocated-on-observation TSDF storage preserve useful room
surface coverage within the Quest memory and update budget?

The normal HUD adds one line:

`CHSL pair=<retained pairs> block=<allocated chunks> vox=<weighted voxels> surf=<surface-band voxels> <CPU integration ms>`

The optional cyan preview is disabled by default. Turning it on only visualizes
the independent shadow mesh; it never replaces or hides the production mesh.

## Acceptance and boundaries

The bypass is worth continuing only if a device run shows all of the following:

- `pair` rises steadily after warmup, with no partial-eye admissions;
- `block` grows locally with explored space instead of allocating a dense room box;
- `surf` rises on observed surfaces without runaway allocation drops;
- integration time remains bounded and scanning cadence stays usable;
- when the optional preview is enabled for an A/B run, it covers the same room
  regions without becoming a second visible source of geometry.

This bypass does not yet contain OpenChisel's native optimization, garbage
collection, deformation/loop-closure handling, semantic room understanding or
the Plane-Priors optimization described in the referenced research. Those are
later decisions, after this isolated route produces device evidence.

## Rollback

Set `enableChiselRouteShadow` to false in
`ScanCoverQuestRoomSurfaceNetsPipeline` to stop the bypass without changing the
production path. Full removal consists of deleting
`ScanCoverChiselSparseTsdfShadow.cs`, removing its single call site and HUD line,
and leaving `ScanCoverDepthFrameContract` in place because the latter is the
production timing fix, not part of CHISEL.

Reference architecture: <https://github.com/personalrobotics/OpenChisel>

## Validation state (2026-10-19)

The complete `Assembly-CSharp` source set was compiled with Unity 6000.3.9f1's
own generated response file and Roslyn toolchain after these edits: zero C#
errors. This is source/assembly validation only. Android packaging, signing,
Quest runtime cost and visible A/B acceptance remain device-side work.


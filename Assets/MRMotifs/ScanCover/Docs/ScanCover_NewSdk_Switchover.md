# ScanCover New SDK Switchover

Date: 2026-07-04

`E:\PCAII\NEW-SCANCOVER` is now the active ScanCover working project.

The previous project remains available as a reference only:

`E:\PCAII\Unity-MRMotifs-ScanCover-main`

## Active Unity Entry

Open this scene for capture and runtime validation:

`E:\PCAII\NEW-SCANCOVER\Assets\MRMotifs\ScanCover\Scene\DepthEffects_ScanCover.unity`

The scene uses the newer Meta XR SDK stack from `NEW-SCANCOVER`. Quest Link testing on this project did not reproduce the headset tearing seen in the older project.

## Active Export Root

Editor capture/export output should land under:

`E:\PCAII\NEW-SCANCOVER\ScanCoverExports`

The scene field `editorLocalSessionExportRoot` is set to that path.

## Capture State

The active scene keeps these capture lines enabled:

- Binocular room Raw Depth snapshots
- Virtual clone input metadata beside the binocular snapshots
- Session reference frame

The temporary fixed HUD/cube objects used during tearing diagnostics are present but inactive.

## Tooling

The Python ScanCover toolset is mirrored under:

`E:\PCAII\NEW-SCANCOVER\Tools`

Tool defaults that previously pointed at the old project were switched to the new project root. Prefer running tools from this folder so shell reconstruction, growth ordering, mapping inputs, and virtual clone/Replica validation all consume the new export tree.

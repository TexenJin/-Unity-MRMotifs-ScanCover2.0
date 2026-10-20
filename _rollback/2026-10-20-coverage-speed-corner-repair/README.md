# Coverage, speed, and corner-repair rollback

This snapshot preserves the source immediately before the 2026-10-20 follow-up
for fragmented coverage, slow correction, headset-relative CHSL drift, and the
remaining corner ridge.

## Restore

Copy the two CHSL/QRS `*.before` files back to their corresponding locations under:

`Assets/MRMotifs/ScanCover/Scripts/MetaXR/08_Preprocess/`

## Changes after this snapshot

- Keep the CHSL preview root in world space because its vertices are already
  world-coordinate positions.
- In CHSL-only execution, retain exact paired-eye preparation but skip the
  hidden dense QRS volume integration, pruning, and extraction workload.
- Key new-surface evidence by 10 cm cell plus unsigned dominant normal axis so
  perpendicular room surfaces cannot promote one another in the same cell.
- Fast-track only the connected, same-direction promoted frontier (two stereo
  pairs); isolated new regions retain the three-pair and 8 cm-baseline gate.
- Extend stereo-confirmed free-space repair down to 6 cm while leaving the
  10 cm full free-space/occluder margin intact, closing the old correction gap.

Unity 6000.3.9f1 imported both changed scripts and completed a domain reload
without C# errors. Quest runtime behavior remains a device-side A/B check.

# ScanCover TSDF And Mesh Baseline Roadmap

This document defines the active engineering baseline. New changes must advance
the current stage and must not reopen completed gates without diagnostic evidence.

## Stage 01 - Exact Contribution And Geometry Diagnostics

Status: COMPLETE

- Record TSDF writes with capture, frame, source pixel, voxel, operation, TSDF,
  weight and confidence features.
- Link suspect geometry back to contributing observations and voxels.
- Separate pure geometric double layers from dirty-state labels.

Exit evidence: runtime counters, contribution ledgers and final voxel state can
be reconciled for integration, replacement, repair, retirement and clearing.

## Stage 02 - Confidence-Gated TSDF Lifecycle

Status: COMPLETE

- Classify observations as ACCEPT, PROVISIONAL or REJECT.
- Keep provisional evidence low-weight until cross-frame confirmation.
- Admit only verified observations to the formal TSDF surface.
- Maintain local continuity without globally relaxing dirty-data gates.
- Remove stale geometry with temporally confirmed clearing rays.
- Traverse clearing rays voxel by voxel with 3D DDA.
- Protect confirmed current surfaces and cancel negative evidence only with a
  trusted near-surface observation.
- Decrease weight before clearing; commit weight zero only after the free-space
  TSDF threshold is reached.

Exit evidence: repeated captures demonstrate the complete lifecycle
`new -> repeat -> apply -> clear`, with ledger zero-weight rows matching runtime
clear counters, no pure geometric double-layer recurrence and no hidden clears.

## Stage 03 - Stable Persistent Mesh

Status: CURRENT BASELINE

Stage 03 consumes the confidence-gated TSDF produced by Stage 02. It must not
compensate for mesh defects by weakening the completed TSDF safety gates.

### Stage 03A - Clean TSDF Iso-Surface

Status: IN PROGRESS

- Extract the mesh from valid weighted TSDF zero crossings.
- Exclude pending, dirty, quarantined and unverified provisional voxels.
- Preserve real corners and depth boundaries.
- Repair continuity in the TSDF field, not with unconditional triangle bridges.

Exit criterion: planar regions grow from clean TSDF into a coherent surface,
without bulges, double layers, overlapping patches or widespread new holes.

### Stage 03B - Incremental Persistent Chunks

- Rebuild only chunks touched by changed TSDF voxels.
- Preserve untouched chunks and stable chunk identities across trigger captures.
- Retire obsolete chunk geometry only after the underlying voxel state changes.
- Avoid replacing the complete visible mesh after every capture.

Exit criterion: revisiting or extending a room updates local geometry while old
clean regions remain visible and stable.

### Stage 03C - Topology, Performance And Room Validation

- Remove degenerate triangles, duplicates, invalid winding and tiny unsupported
  islands while preserving real geometric edges.
- Budget extraction, normal generation and cleanup across frames.
- Validate repeated multi-angle room scans for mesh continuity, stable memory,
  stable frame time, no bulges, no double layers and no full-mesh refresh effect.

Exit criterion: an extended room scan remains geometrically stable and usable
within the target runtime and memory budgets.

## Stage 03 Guardrails

- Do not globally lower confidence, support or TSDF weight thresholds to fill a
  local hole.
- Do not add hard geometric bridging without clean TSDF support.
- Do not let display caches become authoritative geometry.
- Keep diagnostics concise and tied to an actionable Stage 03 exit criterion.

## Later Research Route

Exact per-frame de-integration/re-integration and contaminated-band
reconstruction remain valid future work. They are not prerequisites for the
Stage 03 baseline unless new evidence shows that Stage 02 lifecycle controls are
insufficient.

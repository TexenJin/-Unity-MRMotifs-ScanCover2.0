# Phase 1 Minimal Surface Path

## Purpose

Phase 1 adds a new parallel prototype path inside ScanCover.

It does not replace the current chunk/display pipeline yet.

Its purpose is to prove that ScanCover can maintain a world-space, visible-surface sample field that is closer to a future `depth-driven surface` route than chunk occupancy is.

## Added Scripts

### `ScanCoverDepthSurfaceField_P1`

Path:
- `Assets/MRMotifs/ScanCover/Scripts/06/ScanCoverDepthSurfaceField_P1.cs`

Role:
- samples visible room-facing surface hits through the existing environment raycast path
- stores world-space surface samples in a local quantized field
- groups them into patch-local buckets
- maintains confidence and stability values per sample

Important note:
- this is the minimal Phase 1 bridge for the current project
- it uses the public environment raycast path already available in ScanCover
- it does **not** yet depend on a custom depth texture query wrapper

### `ScanCoverDepthSurfaceDebugRenderer_P1`

Path:
- `Assets/MRMotifs/ScanCover/Scripts/06/ScanCoverDepthSurfaceDebugRenderer_P1.cs`

Role:
- renders the Phase 1 sample field as lightweight instanced quads
- can optionally render patch markers
- serves as a debug/validation view only

## Intended Usage

Attach both scripts to the same object that already hosts:
- `ScanCoverSkeletonBuilder_A`
- `ScanCoverSkeletonSessionController`

Recommended first setup:

1. `ScanCoverDepthSurfaceField_P1`
- assign `builder`
- assign `sessionController`
- leave `referenceFrame`, `environmentRaycast`, and `sampleCamera` empty unless overrides are needed
- keep:
  - `sampleWhileScanning = true`
  - `sampleWhenFrozen = false`
  - `clearOnEnterScanning = true`

2. `ScanCoverDepthSurfaceDebugRenderer_P1`
- assign `surfaceField`
- keep `renderSamples = true`
- keep `renderPatches = false` for the first pass

## What This Prototype Proves

This prototype is intended to answer these questions:

1. Can ScanCover maintain visible surface samples in stable world coordinates?
2. Can those samples be accumulated in patch-local groups rather than chunk-only occupancy groups?
3. Can this run in parallel without disturbing the current chunk/display chain?

## What This Prototype Does Not Yet Prove

This is not yet:

1. the final blue mesh
2. the final shrink-wrap solver
3. the final collision pipeline
4. a replacement for `DisplaySurfaceBuilder`

It is a Phase 1 foundation only.

## Success Criteria For Phase 1

Phase 1 is considered successful if:

1. the new field remains stable in world space
2. sample accumulation is visibly tied to room-facing surfaces
3. patch-local grouping is functioning
4. the prototype runs without obvious visible hitching


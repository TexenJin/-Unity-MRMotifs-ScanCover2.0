# ScanCover Runtime Organization Guide

Date: 2026-03-14

## Current Goal

Keep ScanCover inside the current MetaXR runtime architecture, while reducing script clutter and removing experimental or frozen content from the active prefab and scene path.

## Active Runtime Layers

The current active SessionReferenceFrame runtime is intentionally reduced to:

- `MetaXR/01`
  - `ScanCoverReferenceFrameRegistry`
  - `ScanWorkCenterFollower`
  - `ScanSurfaceSnapper`
- `MetaXR/02`
  - `RevealManager`
  - `ShockwaveScanSpawnerMotif`
- `MetaXR/04`
  - `ScanCoverSkeletonBuilder_A`
  - `ScanCoverSkeletonMesher_B`
  - `ScanCoverSkeletonSessionController`
  - `ScanCoverSkeletonHUD`

These are the only scripts that should remain on the active `[ScanCover] SessionReferenceFrame` prefab for now.

## Removed From Active Prefab

The following were removed from the active prefab because they are not part of the current runtime mainline:

- inactive effect bridge child object
  - `ScanShockwaveBridge`
- inactive isolated driver child object
  - `ScanCoverDriver`
- disabled diagnostic or display components
  - `ScanCoverCollisionVerifier`
  - `ScanCoverDisplaySurfaceBuilder`
  - `ScanCoverDisplaySurfaceRenderer`
  - `ScanCoverSkeletonDebugViz_A`
  - `ScanCoverTileOverlayPrototype`
- experimental depth-surface chain
  - `ScanCoverDepthSurfaceField_P1`
  - `ScanCoverDepthSurfaceDebugRenderer_P1`
  - `ScanCoverDepthSurfaceProvider_P1`
  - `ScanCoverCustomDepthRaycaster_P1`
  - `CustomEnvironmentDepthRaycaster`
  - `DepthGridPointCloud`
  - `ScanCoverDepthObservationProvider_07`

## Archive Policy

The following content classes should not stay mixed into the active runtime folder:

- frozen experiments
- disabled scene-only debug helpers
- technical reserve implementations
- isolated bridge packages
- deprecated display prototypes

These should live under `Assets/MRMotifs/ScanCover/Archive`.

Current archive locations:

- `Assets/MRMotifs/ScanCover/Scripts/Archive/MetaXR/01_Isolated`
  - isolated driver and shockwave bridge package
- `Assets/MRMotifs/ScanCover/Scripts/Archive/MetaXR/03_DepthRevealOverlay`
  - depth-driven overlay rendering experiments
- `Assets/MRMotifs/ScanCover/Scripts/Archive/MetaXR/05_DebugAndOverlay`
  - collision verifier and tile overlay prototypes
- `Assets/MRMotifs/ScanCover/Scripts/Archive/MetaXR/06_DepthSurface_P1`
  - Phase 1 depth surface transition experiments
- `Assets/MRMotifs/ScanCover/Scripts/Archive/MetaXR/07_DepthGrid_Experimental`
  - custom depth raycaster, depth grid, and observation experiments

## Reserve Policy

OpenXR content remains reserve content, not the production-facing consumer layer.

Current rule:

- official consumer layer: `EnvironmentDepthManager`
- reserve diagnostics layer: `Scripts/OpenXR`

## Next Runtime Direction

The next active engineering task is not to restore the removed experimental components.

The old observation-driven surface branch is now downgraded to reference or diagnostic status:

- `ScanCoverBinocularFusedObservationProvider`
- `ScanCoverStableObservationCloudAccumulator`
- `ScanCoverObservationSurfaceMesher`

These scripts remain useful for:

- binocular pairing diagnostics
- point-cloud sanity checks
- historical comparison against the older point-to-surface attempt

They are no longer the preferred runtime mainline for producing ScanCover surface structure.

The next active engineering task is:

`EnvironmentDepthManager` depth texture
-> stable preprocessing layer
-> dense world position / world normal / confidence field
-> local surface patch candidates

This replaces the old `CustomEnvironmentDepthRaycaster + dual-eye agreement` path instead of expanding it.

Current preferred experimental direction:

- `ScanCoverDepthPreprocessor`
- `ScanCoverSurfacePatchCandidateProvider`
- `ScanCoverSurfacePatchDebugQuads`

Rationale:

- fused binocular points became too sparse to reliably seed a regular lattice
- dense `world normal` output already contains stronger local surface evidence
- patch candidates are a better intermediate representation for later lattice or surface growth

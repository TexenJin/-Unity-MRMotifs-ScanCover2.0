# ScanCover Phase 0 Architecture Lock

## Purpose

This document locks the architectural direction for the ScanCover branch before any Phase 1 implementation work starts.

Phase 0 changes no runtime behavior. It only freezes the engineering direction so later work does not drift back to chunk-driven visual logic.

## Locked Conclusions

1. `depth-driven surface` is the future primary geometry layer.
2. `voxel/chunk` is a support layer, not the final geometric truth.
3. The blue display mesh should ultimately visualize `depth-driven surface`, not `chunk` occupancy.
4. `chunk` remains useful for:
   - collision
   - anchoring
   - coarse occupancy
   - fallback coverage
5. `chunk` should not continue to absorb new feature work as the main visual surface.

## Why This Lock Exists

The current ScanCover chain can:
- detect occupied space
- accumulate stable cells
- build chunk meshes
- build a display shell from chunk output

But it does not yet provide true geometric shrink-wrap behavior. Previous tests showed:
- chunk color/state changes did not imply geometric correction
- chunk geometry mostly stabilized by occupancy, not by continuous surface fitting
- display-layer smoothing could improve appearance without proving that the underlying geometry had actually converged

That makes the old route insufficient for the target effect.

## Layer Model

### 1. Depth-Driven Surface

Role:
- primary surface
- geometric boundary
- shrink-wrap carrier
- future source of the blue mesh

Requirements:
- expressed in world/reference coordinates
- updated continuously from visible surface observations
- capable of local correction over time
- supports patch-local rebuilds instead of global rebuilds

### 2. Voxel Skeleton

Role:
- collision layer
- anchoring layer
- coarse occupancy support
- fallback stabilization layer

Requirements:
- low frequency updates
- coarse resolution acceptable
- must remain decoupled from the high-frequency display surface

### 3. Blue Display Mesh

Role:
- observation and presentation layer
- should expose the state of the primary surface
- should not define geometry on its own

Requirements:
- patch-local updates
- low-overhead line/shell display modes
- suitable for validating actual geometric motion

## Current Script Mapping To This Model

### Existing scripts that belong to the current support stack

- `ScanCoverSkeletonBuilder_A`
  - scan sampling
  - voxel/cell accumulation
  - wave-gated acquisition
- `ScanCoverSkeletonMesher_B`
  - chunk mesh generation from confirmed voxel cells
  - chunk collider generation
  - voxel hole fill
- `ScanCoverSkeletonSessionController`
  - scanning/freeze/session orchestration
  - auto-commit logic
- `ScanCoverDisplaySurfaceBuilder`
  - display shell generation from chunk output
  - useful as an observation-layer reference, not as the final shrink mechanism
- `ScanCoverDisplaySurfaceRenderer`
  - display visibility and rebuild coordination

### Existing scripts that are prototype-only or auxiliary

- `ScanCoverTileOverlayPrototype`
  - lightweight surface-aligned visual prototype
  - useful as an exploratory path, not yet the locked primary implementation
- `ScanCoverCollisionVerifier`
  - validation tool for collider behavior
- `ScanCoverSkeletonHUD`
  - metrics/debug UI
- `ScanCoverSkeletonDebugViz_A`
  - debug-only visualization

### Existing scripts that are integration/effect layers

- `ShockwaveScanSpawnerMotif`
- `RevealManager`
- `DepthRevealOverlayRendererFeature`
- `ScanShockwaveBridge`
- `ScanSurfaceSnapper`
- `ScanWorkCenterFollower`
- `ScanCoverEffectDriver`

These are not the first entry point for the new geometry architecture.

## Engineering Rules Locked For Phase 1+

1. Do not add more "apparent convergence" features on top of chunk color/state changes.
2. Do not treat display-layer smoothing as proof of geometry convergence.
3. Do not couple blue mesh visibility to chunk generation.
4. Do not use full-scene synchronous rebuilds as the normal update path.
5. Prefer local patch updates, fixed budgets, and decoupled frequencies.

## Definition Of Success For The New Route

The system is considered on-track only if all of the following become true:

1. A watched region shows actual geometric correction over time.
2. The correction is visible even when using a neutral wireframe/shell view.
3. The blue mesh reflects that correction rather than inventing it.
4. Collision remains available through the support layer without forcing the display layer to update at the same frequency.
5. The process avoids visible frame hitches.

## Phase 0 Exit Criteria

Phase 0 is complete when the team accepts these decisions:

1. `depth-driven surface` is the new mainline.
2. `voxel/chunk` is a support layer.
3. the blue mesh is future display of the surface layer.
4. new implementation work starts from this split rather than extending chunk-driven presentation further.


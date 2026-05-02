# ScanCover Current Script Responsibilities

This document records the current responsibility boundaries in `Assets/MRMotifs/ScanCover` so Phase 1 work can start from a clear baseline.

## Core Scan Stack

### `Scripts/04/ScanCoverSkeletonBuilder_A.cs`

Current role:
- collects scan hits into voxel/cell data
- stores summary statistics
- owns reference frame and sample camera relationships
- supports reveal-wave gated scanning
- supports viewport/sampling bounds filtering

What it is not:
- not the final display surface
- not the collision presenter
- not the blue mesh renderer

### `Scripts/04/ScanCoverSkeletonMesher_B.cs`

Current role:
- converts confirmed voxel cells into chunk meshes
- optionally builds chunk colliders
- supports simple voxel hole fill
- provides coarse frozen geometry and chunk stats

What it is not:
- not a true shrink-wrap solver
- not a continuous surface fitter
- not the future authoritative blue-mesh geometry source

### `Scripts/04/ScanCoverSkeletonSessionController.cs`

Current role:
- controls scan/freeze/clear/toggle session flow
- manages auto-commit of tiles
- coordinates builder, mesher, and display surface build
- now uses the new Input System controller path

What it is not:
- not a geometry solver
- not a visual quality layer

### `Scripts/04/ScanCoverSkeletonHUD.cs`

Current role:
- displays four score metrics
- displays state and input hints
- draws the HUD sampling bounds box/crosshair UI

What it is not:
- not a geometry source
- not a convergence proof by itself

## Display Stack

### `Scripts/04/ScanCoverDisplaySurfaceBuilder.cs`

Current role:
- merges chunk meshes into a display shell
- supports raw/smoothed compare modes
- supports line/wire/face presentation styles
- supports near-view culling and small-island culling
- supports locked display snapshots for frozen output

What it is not:
- not the true geometry correction mechanism
- not proof that underlying voxel geometry has converged

Future note:
- useful reference for future display-layer structure
- should eventually visualize `depth-driven surface` rather than chunk-derived shell output

### `Scripts/04/ScanCoverDisplaySurfaceRenderer.cs`

Current role:
- toggles display visibility according to frozen/scanning state
- coordinates display rebuild timing and source chunk hiding/restoration

What it is not:
- not the geometry owner

## Debug / Verification / Prototype Stack

### `Scripts/04/ScanCoverSkeletonDebugViz_A.cs`

Current role:
- debug visualization for low-level skeleton state

### `Scripts/05/ScanCoverCollisionVerifier.cs`

Current role:
- validates collider behavior with controller-triggered ray and spawned test objects

### `Scripts/05/ScanCoverTileOverlayPrototype.cs`

Current role:
- prototype for lightweight surface-aligned tile/patch rendering
- tests a display strategy closer to "贴瓷砖" than chunk bodies

Why it matters:
- this script is conceptually closer to future `depth-driven surface` presentation than chunk rendering is

What it is not:
- not yet the production geometry pipeline

## Effect / Integration Stack

### `Scripts/01/ScanCoverReferenceFrameRegistry.cs`
- shared reference frame registration

### `Scripts/01/ScanSurfaceSnapper.cs`
- snapping/alignment helper around scan surfaces

### `Scripts/01/ScanWorkCenterFollower.cs`
- follows scan work center / control anchor

### `Scripts/01/ScanShockwaveBridge_Package/ScanShockwaveBridge.cs`
- bridge between scan logic and shockwave effect system

### `Scripts/01/隔离/ScanCoverEffectDriver.cs`
- isolated effect driver for ScanCover-related visuals

### `Scripts/02/RevealManager.cs`
- reveal effect orchestration

### `Scripts/02/ShockwaveScanSpawnerMotif.cs`
- shockwave spawning behavior tied to ScanCover motif flow

### `Scripts/03/DepthRevealOverlayRendererFeature.cs`
- URP renderer feature for depth reveal overlay

These scripts are integration layers around the scan experience. They should remain downstream of the geometry architecture, not define it.

## Responsibility Lock Going Forward

Until the new primary geometry route is implemented:

1. `Builder_A` remains the voxel/cell acquisition layer.
2. `Mesher_B` remains the coarse chunk/collider layer.
3. `DisplaySurfaceBuilder/Renderer` remain observation/presentation layers.
4. Any future true shrink-wrap logic should be introduced as a new `depth-driven surface` layer, not forced into chunk presentation as if chunk were the final surface.

## Immediate Implication For Phase 1

Phase 1 should start by introducing a new surface-focused path rather than extending:
- chunk color-state tricks
- chunk-only convergence claims
- display-only smoothing as a substitute for true geometry correction


# ScanCover Phase 0.5 External Findings

## Purpose

This note records useful findings from the external project:

- `D:/PCA/v83PCA-Rebuild`

The goal is to identify reusable ideas and reference implementations that can inform the future `depth-driven surface` path in ScanCover.

This document does not change ScanCover runtime behavior. It is a reference input for Phase 1.

## Executive Conclusion

The most useful external value is not a ready-made shrink-wrap module, but a set of building blocks that strongly support a `depth-driven surface` architecture:

1. stable depth-to-world projection
2. reconstructed surface normals
3. event-driven depth-aligned rebuild cadence
4. motion-aware temporal stabilization
5. lightweight surface-aligned point/tile placement

These findings support the Phase 0 lock:

- `depth-driven surface` should become the primary surface layer
- `voxel/chunk` should remain the support/collision layer

## High-Value External Scripts

### 1. `CustomEnvironmentDepthRaycaster.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/核心/CustomEnvironmentDepthRaycaster.cs`

Why it matters:
- This is the strongest reusable conceptual base.
- It already provides the core surface-query interface needed by a new ScanCover surface layer.

Key capabilities observed:
- GPU-to-CPU depth availability management
- world position reconstruction from depth texels
- linear depth query in camera space
- normal reconstruction from neighboring depth samples
- depth-based raycast support
- world/frustum/camera matrix maintenance

Key exposed methods:
- `WorldPosAtDepthTexCoord02`
- `WorldPosToLinearDepth02`
- `ReconstructNormal02`
- `ReconstructNormalAtWorldPos02`
- `Raycast02`

Phase 1 relevance:
- likely input/query backbone for a new `depth-driven surface` path
- avoids rebuilding low-level depth sampling infrastructure from scratch

### 2. `RgbZMapCPUBuilder.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/棋盘校准/棋盘校准/深度数据/RgbZMapCPUBuilder.cs`

Why it matters:
- This script contains more useful architectural ideas than a simple point cloud.
- It treats the visible depth surface as a projected field rather than as occupancy.

Key capabilities observed:
- forward projection of depth-driven geometry into RGB-aligned space
- CPU z-buffer rasterization
- optional hole fill
- event-driven rebuild scheduling
- motion-aware stabilization
- adaptive EMA logic for static vs dynamic pixels
- readiness checks before build

Important concepts worth carrying forward:
- `buildOnDepthEvent`
- motion-aware EMA
- distinction between static and dynamic observations
- jump reset thresholding
- validity heuristics before rebuild

Phase 1 relevance:
- directly useful as a reference for temporal stabilization policy
- useful model for "surface state changes over time" rather than voxel occupancy locking

### 3. `DepthGridPointCloud.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/棋盘校准/实验/DepthGridPointCloud.cs`

Why it matters:
- It demonstrates a simple but important principle:
  depth samples can become world-space surface points directly.

Key capabilities observed:
- regular sampling over depth texels
- world-space point reconstruction
- neighbor-based invalid-depth fallback
- optional normal-oriented marker placement
- continuous refresh behavior

Phase 1 relevance:
- useful as the minimal conceptual model for a surface sample field
- demonstrates the difference between:
  - continuous surface points
  - voxel occupancy blocks

### 4. `RandomPlaneScatter.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/核心/RandomPlaneScatter.cs`

Why it matters:
- small script, but conceptually close to the "贴瓷砖 / surface patch placement" direction

Key capabilities observed:
- random valid depth sampling
- world placement from depth texels
- normal-aligned prefab placement

Phase 1 relevance:
- useful reference for low-cost surface-aligned patch/instance placement
- directly relevant to tile-like overlay experiments

## Secondary / Support References

### `MetaDepthTextureConvert2OCV.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/核心/MetaDepthTextureConvert2OCV.cs`

Value:
- useful as a diagnostics / offline analysis bridge
- can help inspect:
  - depth validity
  - hole distribution
  - conversion sanity
  - depth snapshots for comparison

Limitation:
- not a good candidate for the main runtime path due to CPU-access and conversion overhead

### `CameraToWorldManager.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/CameraToWorld/Scripts/CameraToWorldManager.cs`

Value:
- helpful as a camera/world alignment reference
- useful for understanding stable world-space projection and viewport ray relationships

Limitation:
- not a surface reconstruction script
- best treated as a spatial-debug reference rather than a geometry core

## What This Means For ScanCover

The external project suggests that ScanCover should not continue trying to make `chunk` behave like a continuous surface.

Instead, ScanCover should build Phase 1 around:

1. a depth-query backbone
2. a world-space surface sample representation
3. temporal stabilization
4. local patch-based display generation

This aligns with the Phase 0 architectural lock:

- `depth-driven surface` = future primary surface
- `voxel/chunk` = support / collision / anchoring

## Practical Reuse Guidance

### Reuse as Concepts

Strong candidates to reuse conceptually:
- depth texel -> world position reconstruction
- normal reconstruction from adjacent depth samples
- event-driven rebuild scheduling
- motion-aware stabilization
- light surface-aligned patch placement

### Reuse as Code Reference

Strongest code references:
- `CustomEnvironmentDepthRaycaster.cs`
- `RgbZMapCPUBuilder.cs`

These should be treated as primary external references during Phase 1 design.

### Do Not Promote Directly To Runtime Mainline Without Adaptation

Scripts that are useful but should not be copied blindly into the main ScanCover runtime:
- `MetaDepthTextureConvert2OCV.cs`
- `CameraToWorldManager.cs`
- debug and sample scripts outside the core depth path

## Phase 0.5 Exit Result

The external scan confirms that ScanCover already has a credible next direction:

1. stop using chunk logic as the main visual surface path
2. build a new `depth-driven surface` route
3. use external depth-ray and temporal-stabilization ideas as reference
4. keep chunk/voxel for support responsibilities only


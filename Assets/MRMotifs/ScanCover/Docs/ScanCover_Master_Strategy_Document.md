# ScanCover Master Strategy Document

## 1. Document Purpose

This is the consolidated planning document for the current ScanCover direction.

It merges:
- Phase 0 architecture lock
- Phase 0.5 external project findings
- overall strategy
- engineering roadmap
- staged implementation plan
- current script responsibility boundaries

This document does not modify runtime behavior. It defines the technical direction before Phase 1 implementation starts.

## 2. Problem Statement

The existing ScanCover pipeline can already:
- accumulate scan evidence
- commit stable spatial cells
- build chunk-based frozen geometry
- build a blue display shell from chunk output
- provide HUD and interaction flow

But current tests showed a clear limitation:

1. chunk-based geometry is good at representing occupancy, not continuous surface
2. chunk growth and state changes are not equivalent to actual geometric shrink-wrap
3. display smoothing can improve appearance without proving real surface convergence
4. the current route is insufficient for the target visual effect shown by Quest-like room mesh examples

The main problem is therefore:

**ScanCover currently has a support geometry pipeline, but does not yet have a true primary surface pipeline.**

## 3. Target Effect

The target effect is not just "scan coverage."

The intended result is:
- a thin, room-facing, continuous surface
- gradual geometric correction over time
- visible surface attachment to walls, floor, furniture, and other room boundaries
- a blue mesh that reflects real surface evolution
- no visible large frame hitches during interaction

More specifically:

1. early scan:
- surface appears quickly, even if rough

2. mid scan:
- surface visibly adjusts
- edges tighten
- local geometry becomes more room-aligned

3. late scan:
- surface stabilizes
- display mesh becomes coherent and presentation-ready

## 4. Locked Core Conclusion

### 4.1 Main Direction

The future ScanCover mainline should be:

- `depth-driven surface` as the primary geometry layer
- `voxel/chunk` as the support layer
- blue display mesh as the visualization of `depth-driven surface`

### 4.2 Why

Because:
- voxel/chunk logic is fundamentally occupancy-oriented
- target effect is fundamentally surface-oriented
- a support layer can remain coarse and stable
- the primary surface must remain continuous and locally correctable

### 4.3 What Must Stop

The following should no longer be treated as the main path:

1. chunk state/color change as proof of geometric convergence
2. display-layer smoothing as proof of real shrink-wrap
3. chunk-driven visual surface as the long-term architecture
4. full-scene synchronous rebuild as the standard update path

## 5. Layer Model

## 5.1 Depth-Driven Surface

Role:
- primary geometry
- boundary owner
- shrink-wrap carrier
- room-facing surface
- future source of the blue display mesh

It should be:
- world-space stable
- continuously correctable
- patch-local
- surface-first rather than occupancy-first

This is the future:
- knife
- cage
- surface itself

## 5.2 Voxel / Chunk Support Layer

Role:
- collision
- anchoring
- coarse occupancy
- fallback support

It should be:
- low frequency
- low resolution compared to the surface layer
- stable
- decoupled from display updates

This is not the final visual truth.

## 5.3 Blue Display Mesh

Role:
- observation layer
- presentation layer
- validation layer for surface evolution

It should visualize:
- the primary `depth-driven surface`
- not chunk occupancy as the final authority

It should support:
- wireframe
- thin shell
- previous vs current comparison when needed

## 5.4 HUD / Debug Layer

Role:
- reporting only
- not geometry truth

It should provide:
- coverage metrics
- stability metrics
- hole fill metrics
- mesh/display metrics
- patch/debug state as needed

## 6. Why The Previous Chunk Route Was Insufficient

Earlier observations showed:

1. chunk bodies often appeared as a large mass and then stabilized
2. red/blue state transitions did not correspond to visible shrink-wrap
3. geometry did not visibly tighten around observed surfaces
4. visual smoothing could create the impression of improvement without real underlying correction

This happened because:

1. the system confirmed occupied voxels rather than fitting continuous surfaces
2. voxel positions were largely locked to grid structure
3. commit behavior expressed stability, not iterative geometric correction
4. hole fill filled occupancy gaps but did not fit a continuous boundary

Conclusion:

**The old route was not failing due to parameter tuning alone. It was structurally the wrong primary representation for the target effect.**

## 7. External Findings From `v83PCA-Rebuild`

An external project review was done at:
- `D:/PCA/v83PCA-Rebuild`

This review strongly reinforced the new direction.

### 7.1 Highest-Value External Scripts

#### A. `CustomEnvironmentDepthRaycaster.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/核心/CustomEnvironmentDepthRaycaster.cs`

Most important value:
- ready-made depth-query backbone

Notable capabilities:
- depth texel to world position
- world position to linear depth
- normal reconstruction
- depth-based raycast
- camera/frustum matrix handling

Strategic value:
- likely direct conceptual backbone for Phase 1 surface input

#### B. `RgbZMapCPUBuilder.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/棋盘校准/棋盘校准/深度数据/RgbZMapCPUBuilder.cs`

Most important value:
- temporal stabilization strategy for surface-driven data

Notable capabilities:
- event-driven rebuild cadence
- RGB-aligned z map construction
- CPU z-buffer rasterization
- motion-aware EMA
- jump reset logic
- hole fill
- readiness heuristics

Strategic value:
- provides direct inspiration for surface patch stabilization and update budgeting

#### C. `DepthGridPointCloud.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/棋盘校准/实验/DepthGridPointCloud.cs`

Most important value:
- demonstrates a direct world-space visible-surface sample field

Strategic value:
- useful conceptual minimal model for `depth-driven surface`

#### D. `RandomPlaneScatter.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/核心/RandomPlaneScatter.cs`

Most important value:
- demonstrates low-cost normal-aligned placement on visible depth surfaces

Strategic value:
- close to "tile/patch on surface" prototype logic

### 7.2 Secondary References

#### `MetaDepthTextureConvert2OCV.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/Scripts/核心/MetaDepthTextureConvert2OCV.cs`

Value:
- debugging / offline analysis tool

#### `CameraToWorldManager.cs`

Path:
- `D:/PCA/v83PCA-Rebuild/Assets/Calib/CameraToWorld/Scripts/CameraToWorldManager.cs`

Value:
- camera/world alignment reference

### 7.3 External Review Conclusion

The external review does not provide a ready-made final solver.
It does provide strong evidence that the right route is:

1. depth query backbone
2. world-space visible surface samples
3. temporal stabilization
4. patch-local reconstruction
5. display mesh derived from that surface

## 8. Performance Reality

The target system must maintain:
- good visual quality
- no visible frame hitches

This does not mean "perfect full-scene high-quality reconstruction every frame."

It means:
- the user should not perceive obvious stutter
- quality should be prioritized in the active region
- convergence should be progressive rather than instantaneous

## 8.1 Main Performance Pressure Sources

1. primary surface update cost
2. temporal stabilization cost
3. local patch topology update cost
4. display mesh refresh cost
5. support collision layer refresh cost

The biggest cost is not the blue mesh itself.  
The biggest cost is maintaining a stable, correctable primary surface beneath it.

## 8.2 Performance Rules

1. never rebuild the whole scene every frame
2. decouple surface, display, and collider frequencies
3. only rebuild dirty patches
4. keep collider lower-frequency than display
5. budget work per frame
6. prioritize center/active area quality over global uniformity

## 8.3 Practical Standard

The real target is:
- no visible hitching
- not mathematically maximal quality at all times

This is realistic.  
A full-scene, fully dynamic, always-high-resolution reconstruction path is not realistic on-device.

## 9. Full Strategy

## 9.1 Overall Direction

1. stop extending chunk as if chunk were the final surface
2. build a new `depth-driven surface` path alongside the current support stack
3. keep voxel/chunk for:
- collision
- anchoring
- coarse occupancy
- fallback support
4. migrate blue display mesh to the surface path once the surface becomes valid

## 9.2 Visual Interpretation

During debugging, the best external manifestation of geometric correction is:

1. wireframe shell
2. thin transparent shell
3. previous vs current comparison

Chunk blocks are not a good long-term observation medium for shrink-wrap behavior.

## 9.3 Intended Representation Evolution

1. early phase:
- points / surfels / tiles

2. middle phase:
- local patch shell

3. late phase:
- blue triangle mesh shell

This staged representation can align with the target video-like appearance while controlling cost.

## 10. Engineering Roadmap

## 10.1 Data Structures

### A. Primary Surface Sample

Each surface sample or patch-local support element should eventually track:
- `positionWS`
- `normalWS`
- `confidence`
- `stability`
- `lastUpdateTime`
- source patch id
- room-facing / valid-facing state

### B. Support Voxel

Each support element only needs:
- occupancy
- hit count
- update time
- support-state / collider-state
- patch relation if needed

### C. Patch Container

Each patch should track:
- spatial bounds
- local sample set
- current display cache
- previous display cache if needed
- dirty flag
- stable flag
- priority

## 10.2 Update Pipeline

### A. Sampling Stage

Input:
- depth data
- pose
- current view/sampling region

Output:
- visible world-space samples
- local normal estimates

### B. Surface Fusion Stage

Input:
- new visible samples
- existing patch samples

Output:
- updated primary surface state

Should include:
- nearest association
- local correction
- confidence accumulation
- stability update
- outlier rejection

### C. Dirty Patch Decision Stage

Determines:
- which patches need display rebuild
- which patches need support-layer refresh

### D. Display Rebuild Stage

Builds:
- wireframe
- shell mesh
- local blue mesh output

Only for dirty patches.

### E. Support Layer Refresh Stage

Updates:
- coarse voxel/chunk/collider representation

At a lower frequency.

## 10.3 Frequency Separation

### High Frequency
- sample ingestion
- local sample fusion

### Mid Frequency
- patch stabilization
- patch priority updates

### Low Frequency
- display shell rebuild

### Lower Frequency
- voxel/collider refresh

## 10.4 What Must Never Be Reintroduced

1. full-scene synchronous display rebuild as normal path
2. collider refresh at the same rate as visible surface updates
3. chunk color-state logic presented as geometric truth
4. display smoothing treated as proof of convergence

## 11. Staged Implementation Plan

## Phase 0: Architecture Lock

Status:
- complete

Output:
- architecture lock
- script responsibility map

Acceptance:
- `depth-driven surface` accepted as mainline
- chunk/voxel accepted as support layer

## Phase 0.5: External Reference Review

Status:
- complete

Output:
- `v83PCA-Rebuild` findings document

Acceptance:
- external references identified for Phase 1 input

## Phase 1: Minimal Surface Path

Goal:
- introduce a new `depth-driven surface` prototype path without replacing the current stack yet

Required result:
- world-space visible surface samples
- patch-local update path
- no obvious hitch from basic updates

Not yet required:
- final blue mesh
- final collider integration

## Phase 2: Real Shrink-Wrap Validation

Goal:
- prove that geometry itself changes over time in watched regions

Required result:
- the same observed region visibly corrects over time
- not just color/state changes

Validation methods:
- wireframe shell
- current vs previous shell comparison

## Phase 3: Blue Surface Display

Goal:
- let blue mesh directly reflect the primary surface

Required result:
- patch-local blue mesh generation
- wireframe mode and thin-shell mode
- no full-scene rebuild dependency

## Phase 4: Support Layer Refactor

Goal:
- reframe voxel/chunk strictly as support/collision layer

Required result:
- collision remains stable
- display and collision become clearly decoupled

## Phase 5: Performance Stabilization

Goal:
- achieve no visible hitching

Required result:
- patch budgeting
- dirty-region update only
- lower-frequency collider updates
- bounded per-frame work

## Phase 6: Final Presentation Layer

Goal:
- push toward final visual quality

Required result:
- stable blue shell appearance
- reduced obstruction
- visual polish and motif integration

## Phase 7: Metrics And Acceptance

Goal:
- make visual quality and runtime metrics correlate

Candidate metrics:
- coverage
- stability
- surface change rate
- patch stable ratio
- visible mesh completeness
- average and peak frame cost

## 12. Current Script Responsibility Map

## 12.1 Core Scan Stack

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverSkeletonBuilder_A.cs`

Current role:
- scan acquisition
- voxel/cell accumulation
- summary statistics
- reveal-wave gated acquisition
- sampling bounds filtering

Not the future primary surface.

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverSkeletonMesher_B.cs`

Current role:
- chunk generation from confirmed voxel cells
- collider generation
- voxel hole fill

Support layer only.

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverSkeletonSessionController.cs`

Current role:
- scan/freeze/clear/toggle session orchestration
- auto-commit logic
- builder/mesher/display coordination
- new Input System controller input

Not a geometry solver.

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverSkeletonHUD.cs`

Current role:
- four-score HUD
- state display
- sampling bounds UI

Debug/reporting only.

## 12.2 Display Stack

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverDisplaySurfaceBuilder.cs`

Current role:
- merges chunk output into display shell
- compare modes
- wire/face display modes
- locked display snapshots
- near-view and small-island culling

Useful as an observation layer reference.  
Not the future source of true geometric convergence.

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverDisplaySurfaceRenderer.cs`

Current role:
- display visibility control
- display rebuild timing
- source chunk hide/show coordination

Presentation coordinator only.

## 12.3 Debug / Prototype Stack

### `Assets/MRMotifs/ScanCover/Scripts/04/ScanCoverSkeletonDebugViz_A.cs`
- debug visualization

### `Assets/MRMotifs/ScanCover/Scripts/05/ScanCoverCollisionVerifier.cs`
- collider validation tool

### `Assets/MRMotifs/ScanCover/Scripts/05/ScanCoverTileOverlayPrototype.cs`
- lightweight surface-aligned tile/patch prototype
- conceptually closer to future surface presentation than chunk bodies are

## 12.4 Integration / Effect Stack

### `Assets/MRMotifs/ScanCover/Scripts/01/...`
- reference frame support
- snapping/following
- shockwave bridge
- isolated effect driver

### `Assets/MRMotifs/ScanCover/Scripts/02/...`
- reveal manager
- shockwave scan spawner motif

### `Assets/MRMotifs/ScanCover/Scripts/03/...`
- URP depth reveal overlay renderer feature

These are downstream integration layers.  
They should not define the new geometry core.

## 13. Phase 1 Entry Conditions

Phase 1 should only begin under these assumptions:

1. chunk is no longer treated as the final visual truth
2. `depth-driven surface` is accepted as the new main path
3. blue mesh will eventually be driven from the new surface path
4. Phase 1 may start as a parallel prototype path inside ScanCover
5. current runtime remains intact until the new path proves itself

## 14. Immediate Next Step

The next concrete task after this document is:

1. define the minimal data structure for `depth-driven surface`
2. define the first script/module boundary for the new path
3. decide whether the first representation is:
- point/surfel
- tile/patch
- local shell mesh

The recommended starting point is:

- patch-local surface samples first
- blue mesh later

Because that is the lowest-risk way to prove real geometric correction before building final presentation.


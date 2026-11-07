# RoomSetupReplica V2 GPU production migration

## Active live route

`A-gpu-v0.1.1-fusion-mc-hard3m-v9`

The live scanner submits the callback-latched stereo depth texture and its four
camera matrices to one GPU backend. The production route is:

`stereo depth -> 3 m hard gate -> v0.1.1 GPU TSDF -> GPU Marching Cubes -> indirect draw`

There is no whole-depth `AsyncGPUReadback`, CPU TSDF loop, CPU Marching Cubes,
`Mesh.SetVertices`, or `Mesh.SetIndices` on this route. The RoomSetup backend
does not instantiate or call the prior GPU extractor, its grouping modes,
smoothing, or candidate-history state.

## Preserved admission rules

- Source voxel size: 5 cm.
- Minimum observed range: 0.30 m.
- Maximum observed surface range: 3.00 m.
- Maximum distance of every written TSDF voxel from the owning eye: 3.00 m.
- The recovered v0.1.1 central depth-gradient test is restored: all four
  neighbours must be valid and `dx^2 + dy^2 <= 0.01`.
- Positive truncation is `0.20 * (voxelDepth + 1)` and negative truncation is
  `(0.20 / 3) * (voxelDepth + 1)`, matching v0.1.1 rather than the later fixed
  15 cm bands.
- Left and right accepted observations are integrated sequentially with the
  v0.1.1 inverse-square sample weight and a maximum accumulated weight of
  1000. Either eye may first-write and every later accepted observation has
  ordinary weighted authority.
- There is no mature residual deadband, sign hysteresis, four-pair latch,
  bounded 2 mm/5-10 mm correction, single-eye mature freeze, five-pixel edge
  quarantine, or displaced-view challenger on the live GPU route.

## Recovered v0.1.1 mesh behavior

- Persistent storage is divided into world-anchored `31 x 31 x 31` sample
  tiles. Each tile spans 1.5 m of MC cells at 5 cm source spacing; adjacent
  tiles share their boundary sample plane.
- Every frame activates only tiles intersecting the current stereo view and
  the current-eye 3 m sphere. Previously created tiles retain their TSDF and
  do not move when the user walks, so coverage is no longer bounded by the
  first eye position.
- The MC source field is the 5 cm TSDF. Every cell corner is an exact even
  source coordinate, reproducing the aligned 10 cm derived lattice without
  trilinear resampling.
- A cell is emitted only when all eight source corners have positive support.
- The recovered MC triangle table, zero-crossing interpolation and `2,1,0`
  winding are used directly on the GPU.
- MC vertices are recomputed directly from the current TSDF zero crossings,
  as in v0.1.1. The later persistent edge history, missing-crossing grace,
  tangential freeze, 1 cm vertex dead-zone and 0.5 normal response are removed.
- The later same-layer anchor/pair volume and cross-layer cell rejection are
  removed. They were added after the v0.1.1 restoration and are outside this
  baseline; MC topology now follows the supported v0.1.1 TSDF corners alone.
- Triangle vertices remain in GPU buffers and are rendered with
  `Graphics.RenderPrimitivesIndirect`, including a dedicated depth-only pass
  before the cyan wire pass.
- A GPU union-find first joins the three vertices of every triangle. A GPU
  position hash then rounds every intersection to the recovered 0.1 mm
  seam-weld lattice and joins equal positions, matching the CPU connectivity
  definition without a CPU dictionary.
- When the candidate contains strictly fewer than 40,000 triangles, GPU
  component counting removes components with strictly fewer than 32
  triangles and compacts survivors into the final draw buffer. At 40,000 or
  above, cleanup is bypassed exactly like the recovered CPU rule.
- Connectivity, counts, filtering and compaction remain GPU-resident; the CPU
  only submits direct or indirect dispatches and never reads the mesh back.
- All persistent tiles emit into the same global candidate buffer before
  connectivity cleanup, so components and seams are classified across tile
  boundaries rather than independently per tile.

## Compatibility boundary

The recovered CPU reconstruction remains compiled for deterministic tape
replay and evidence tools, but live scanning returns before that path can
allocate readback targets or integrate a CPU voxel. Existing tape recording and
CPU mesh export have not yet been converted into GPU stop-time transactions;
they are not part of the normal right-trigger live coverage path.

## Acceptance state

- Static source/resource/scene-reference checks: passed.
- RoomSetupReplica Python regression scripts: 18 passed, including the new
  isolated GPU Marching Cubes route contract.
- Two optional ML-training scripts were not run because their external
  `joblib` dependency is not installed; neither exercises the Unity live path.
- Unity import and compute/shader compilation: pending user-owned Unity refresh.
- Quest frame time, memory headroom and visible geometry: pending user-owned
  device A/B. No device result is claimed by this source migration.

## Sequential Environment Depth source comparison

The separate B app uses route
`B-environment-depth-global-final-v0.1.2-gpu-mc-hard3m-v1`. Its only intended
production-path difference is the input publication contract:

`_EnvironmentDepthTexture + _EnvironmentDepthReprojectionMatrices + _EnvironmentDepthZBufferParams -> same 3 m GPU TSDF/MC`

It does not run beside A. Its generated scene removes both `DepthCapture` and
`AROcclusionManager`, then gives the single acquire ownership to
`EnvironmentDepthManager`. B reconstructs sampled depth back to world space
with the inverse published reprojection matrix, applies the same four-neighbour
0.1 m gradient rule and both 3 m surface/voxel gates, and feeds the same
persistent TSDF tiles and MC extraction buffers. Depth masks and hand removal
remain off for this first source-only A/B.

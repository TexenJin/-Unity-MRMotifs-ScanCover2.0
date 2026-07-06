# Quest Spatial Scan Playback Lite v1

Date: 2026-07-05

## Purpose

This is the first lightweight Quest-return result for the ScanCover / Replica validation line.

It does not run reconstruction on Quest. The expensive work remains offline:

- Replica `office0` quad-dominant remeshing
- observer-view growth ordering
- edge thinning for mobile runtime

Quest runtime only plays back precomputed scan-growth line chunks.

## Why This Shape

The current strategy document locks the target architecture as:

- depth-driven surface as primary geometry
- voxel/chunk as support
- blue/scan mesh as display layer
- bounded per-frame work

This playback package follows that direction as a display-layer validation artifact. It simulates the video-like effect of a user walking and continuously observing the room while a spatial scan mesh grows.

## Active Unity Files

Runtime scripts:

- `Assets/MRMotifs/ScanCover/Scripts/MetaXR/13_QuestPlayback/ScanCoverQuestSpatialScanPlayback.cs`
- `Assets/MRMotifs/ScanCover/Scripts/MetaXR/13_QuestPlayback/ScanCoverQuestSpatialScanBootstrap.cs`

Runtime data:

- `Assets/MRMotifs/ScanCover/Resources/QuestSpatialScanDemo/office0_observer_scan_lite.bytes`
- `Assets/MRMotifs/ScanCover/Resources/QuestSpatialScanDemo/office0_observer_scan_lite_manifest.json`

Offline generation scripts:

- `C:/Users/ROG/Documents/New project/make_replica_observer_growth.py`
- `C:/Users/ROG/Documents/New project/build_quest_spatial_scan_lite_asset.py`

## Data Budget

Source:

- Replica office0 quad wire
- 40,422 source quad edges
- 35 observer-growth steps
- 145 observer samples

Quest Lite asset:

- 13,906 complete medium-LOD quad edges
- 35 reveal steps
- one parse at startup
- static MeshTopology.Lines chunks
- no runtime remeshing
- no runtime Poisson / Instant Meshes / pymeshlab
- no edge thinning / no random edge sampling

## Runtime Behavior

On Android Quest builds only, the bootstrap auto-creates:

`[ScanCover] Quest Spatial Scan Playback`

The playback component:

1. loads `QuestSpatialScanDemo/office0_observer_scan_lite`
2. builds 35 static line-mesh children once
3. reveals chunks over time
4. moves a small observer marker along the simulated scan path

Editor play mode is not affected by the bootstrap because it is wrapped in:

`UNITY_ANDROID && !UNITY_EDITOR`

## Disable

To disable this Quest playback without deleting data, remove or comment out:

`ScanCoverQuestSpatialScanBootstrap.cs`

The rest of ScanCover capture/export remains independent.

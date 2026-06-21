# Stage 0-A: Meta Scene Mesh Export Audit

## Purpose

Stage 0-A makes Meta Scene Mesh a concrete reference artifact before it is used to judge ScanCover's own BL surface, TSDF, multi-frame fusion, or plane-family diagnostics.

This stage does not decide that Meta Scene Mesh is correct. It only answers:

- Can the runtime Meta Scene Mesh be captured as files?
- Which mesh objects/chunks are produced?
- What coordinate space are they in?
- Can we compare the exported mesh against ScanCover-generated geometry later?

## Exporter

Runtime component:

`Assets/MRMotifs/ScanCover/Scripts/MetaXR/12_Audit/ScanCoverMetaSceneMeshAuditExporter.cs`

Editor button:

`Assets/MRMotifs/ScanCover/Editor/ScanCoverMetaSceneMeshAuditExporterEditor.cs`

The exporter is deliberately generic. It reads runtime `MeshFilter` objects instead of depending on private or version-sensitive Meta building-block classes.

## Recommended Scene

Open:

`Assets/MRMotifs/ScanCover/Scene/Meta Scene Mesh.unity`

Add `ScanCoverMetaSceneMeshAuditExporter` to an empty GameObject, or to `[BuildingBlock] Scene Mesh`.

Recommended settings for the first pass:

- `Scene Mesh Root`: leave empty for whole-scene search, or assign the loaded Scene Mesh root if visible at runtime.
- `Include Inactive Children`: on
- `Require Renderer`: off
- `Require Renderer Enabled`: off
- `Export Local Raw Objects`: on
- `Export World Aligned Objects`: on
- `Export Combined World Obj`: on
- `Export Component Inventory`: on

Enter Play Mode on Quest/Link, wait for Meta Scene Mesh to load, then press:

`Export Meta Scene Mesh Audit Package`

## Output

Default output directory:

`ScanCoverExports/MetaSceneMeshAuditSessions/ScanCover_MetaSceneMeshAudit_*`

Expected files:

- `raw_local_meshes/*.obj`
  - Per-object mesh vertices in each object's local space.
- `aligned_world_meshes/*.obj`
  - Per-object mesh vertices baked into Unity world space.
- `meta_scene_mesh_aligned_all.obj`
  - One combined world-space OBJ for CloudCompare/Open3D comparison.
- `mesh_filters.csv`
  - Mesh object metadata: path, active state, renderer state, layer, vertex count, triangle count, world transform, world bounds.
- `component_inventory.csv`
  - Component type inventory on exported mesh objects.
- `session_info.json`
  - Scene name, source root, total mesh/vertex/triangle counts, aggregate world bounds.

## Pass Criteria

Stage 0-A is considered usable when:

- Exported mesh count is non-zero.
- `meta_scene_mesh_aligned_all.obj` opens in CloudCompare/Open3D.
- `mesh_filters.csv` identifies stable object paths or generated chunk paths.
- World bounds are plausible for the scanned room.
- The exported world-space mesh roughly aligns with ScanCover's BL surface export in the same Unity coordinate space.

## Failure Modes To Record

- No `MeshFilter` found: Scene Mesh did not load, root assignment is wrong, or Meta permissions/setup are incomplete.
- Mesh exists but all bounds are near zero: transform or generation problem.
- Mesh opens but is rotated/scaled/offset from BL export: coordinate-space issue, not yet an algorithm issue.
- Many tiny meshes with unstable names: use geometry/bounds/chunk ids rather than object names as correspondence keys.

## Next Stage

Stage 0-B compares this exported Meta Scene Mesh against:

- Current BL surface mesh export.
- Multi-frame depth-derived surface export.
- TSDF branch output.
- Plane-family diagnostic output.

The comparison must happen in a declared common comparison space, preferably Unity world space first.

# ScanCover surface-productization chain

## Production invariant

There is exactly one 5 cm TSDF. Quest depth that passes the existing frame
health gates is integrated without GunGel pose correction, per-point GunGel
distance authority, or a court plane replacing the measured zero level.

The production path is:

1. Quest depth and its matching pose.
2. Existing frame-health filtering.
3. GunGel stable-id/candidate evidence (identity only).
4. The sole native 5 cm TSDF.
5. Native 5 cm Surface-Nets candidate chunks.
6. `SurfaceProductizer`: approved court planes and the once-captured scene
   ruler enter here as post-TSDF sidecars. The ruler retains its measured U/V
   footprint; neither source may replace the depth texture or TSDF zero level.
   They may adjust only a compatible planar interior by at most 15 mm. Mixed
   evidence, features and chunk boundaries remain unchanged.
7. `ChunkQualityGate`: stale, temporarily empty and abruptly regressive
   candidates retain the previous immutable front. Real removal requires
   repeated confirmation.
8. Atomic chunk publication in `PersistentChunkMeshPipeline`.
9. The complete native topology remains intact. A world-anchored 10 cm
   triangular grid is drawn at presentation time, so visual grid size cannot
   create degenerate triangles, holes or cross-block tears.

## Ownership rules

- TSDF owns reconstruction evidence, not the delivered product.
- GunGel owns surface identity and grouping, not distance.
- The final court is an always-on in-memory production service. It reads only
  GunGel correspondence/stable-id buffers and does not depend on pressing A,
  a recorder, a replay package, or file output.
- The final court owns bounded product constraints, not TSDF writes.
- The captured scene ruler owns a finite reference footprint, not measured
  depth. It bypasses fusion and is consumed only by `SurfaceProductizer`.
- `SurfaceProductizer` owns candidate-vertex adjustment, never history.
- `ChunkQualityGate` owns publish/retain/remove decisions, never geometry.
- `PersistentChunkMeshPipeline` owns immutable-front atomicity.
- Product extraction has at most four asynchronous commits in flight. Queue
  entries are unique per block, unseen blocks outrank refreshes, and a result
  whose epoch was superseded while in flight is discarded before publication.
- A one-voxel mesh-block skirt overlaps adjacent products for display
  continuity. It is read from the same TSDF and never feeds back upstream.

## Failure behaviour

- Missing productizer shader: exact raw-candidate pass-through.
- No approved local plane: exact raw-candidate pass-through.
- Productizer dispatch exception: exact raw-candidate pass-through.
- Empty or regressive update: keep the last visible front until confirmed.
- Product-plane change during an in-flight commit: queue that block again after
  the current transaction completes.
- Diagnostic capture absent or failing: the production court and mesh product
  continue independently; diagnostics are observers, never an authority.

These fallbacks prevent a presentation feature from producing an empty scan or
silently reconnecting a second reconstruction authority.

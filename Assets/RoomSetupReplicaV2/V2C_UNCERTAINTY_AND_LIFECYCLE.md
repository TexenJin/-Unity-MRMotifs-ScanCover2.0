# V2-C uncertainty and lifecycle

V2-C starts from the frozen `room-setup-replica/v2b-regular10cm-baseline`.
It does not change the 5 cm SDF, aligned 10 cm derived field, Marching Cubes,
or `RemoveSmallConnectedComponents(<32)` geometry rules.

## Uncertainty boundary

The recovered Room Setup rule is confirmed:

```text
if uncertainty > 89 and depth <= maxDepth: depth = 0
```

That rule is not active in this replica because its input is not available
through the project's public depth path. AR Foundation 6.5 defines optional
environment-depth confidence APIs, but the Meta OpenXR occlusion provider used
by this project:

- registers `environmentDepthConfidenceImageSupportedDelegate = null`;
- publishes one swapchain texture named `_EnvironmentDepthTexture`;
- returns one texture descriptor per frame.

`DepthCapture` consequently consumes only `args.externalTextures[0]`. A generic
confidence value, temporal variance, depth gradient, normal angle, or TSDF
weight is not substituted for Room Setup's native uncertainty byte. The replay
receipt records the threshold and rule as known while reporting
`uncertaintyInputAvailable=false` and `uncertaintyApplied=false`.

This boundary should change only if an exact frame-paired native uncertainty
image is recovered and its direction and numeric range are verified.

## Lifecycle alignment

V2-C makes the confirmed scheduling state machine explicit:

```text
bucket = lastPublishedTriangleCount / 6000
interval = min(900 ms, 150 ms + 250 ms * bucket)
```

The interval therefore steps through 150, 400, 650, and 900 ms. The trigger
timestamp is advanced when a mesh pass starts, as in the recovered gate, not
after all chunk work ends.

The complete asynchronous mesh/post-process/publication path is single-flight.
If its prior task is still active when another trigger becomes due, the trigger
is skipped and counted; no second immutable mesh snapshot is queued. Fusion can
continue, and the next eligible pass reads the current field.

Publication remains ordered as follows:

1. Build the replacement in the Back mesh.
2. Point the renderer and stored Front at the completed replacement.
3. Reuse the former Front as the next Back mesh.

Thus a partial candidate never replaces the visible Front. A legitimate empty
replacement can still remove geometry after the complete global candidate and
`<32` decision are ready.

## Acceptance

The next Quest replay must use the same 156-pair Stage tape as the V2-B
baseline. Geometry acceptance is:

- regular 10 cm mesh remains visually unchanged;
- `simplificationApplied=false` and zero accepted collapses;
- `<32` cleanup remains active with the same threshold;
- the final mesh hash and counts match V2-B unless a documented lifecycle race
  in the older build caused a difference;
- the new uncertainty and lifecycle receipt fields contain the values described
  above.

### Quest 3 replay accepted on 2026-10-26

The same 156-pair Quest Stage tape was replayed after commit `a43c101`.

```text
algorithm                         V2CWhiteboxLifecycleRegular10cmMesh
activeChunks                      407
weightedVoxels                    697294
preview/final triangles           27393 / 27393
pre-cleanup components            138
removed <32 components/triangles  127 / 510
post-cleanup components           11
simplification applied/collapses  false / 0
mesh publications                 57
skipped async busy                0
first mesh                        0.2970 s at pair 1
mean correction delay             0.2675 s
mean correction pair lag          0.3741
replay elapsed                    34.7382 s
direct depth support              0.90789616
final mesh SHA-256                4E12E007F652E1EE13203B298B1E665D6687B35A42E2AC32A7C9E083EE3F5906
```

The final hash, vertex/triangle counts, component counts, plane residuals, and
direct-depth support are identical to the frozen V2-B baseline. The scheduling
change therefore altered publication timing without altering final geometry.
Mean correction delay improved from approximately 0.32 s to 0.27 s and mean
pair lag from 0.59 to 0.37. The single-run maxima moved from 3.57 s / 13 pairs
to 3.83 s / 14 pairs; this is retained as timing variance, not presented as an
improvement.

## Next isolated lifecycle step

The recovered GC operates on sparse leaves containing 64 voxels (4 x 4 x 4),
whereas the current replica applies the same retention predicate to an entire
16 x 16 x 16 storage Chunk. Leaf-granular frustum GC must therefore be tested
as a separate V2-C change. Its native trigger frequency remains unresolved and
must not be invented silently.

### Implementation prepared

The next build now partitions each storage Chunk into 4 x 4 x 4 source leaves,
tests each leaf against the union of the two current depth-eye frusta, and
applies the recovered predicate independently:

```text
retain leaf iff any voxel has
    weight > 0 and sdf < 0.3 m * clamp(viewDepth, 1, 2)
```

Released leaves clear their 64-voxel storage and dirty the corresponding mesh
neighbourhood. An empty parent Chunk is then returned to the existing pool.
The cadence remains once per accepted pair solely to preserve the V2-B
variable; the receipt calls this `Preserved_NotNativeConfirmed`. Candidate and
released leaves, cleared voxels, released parent Chunks, and pass count are all
recorded for the next same-tape A/B.

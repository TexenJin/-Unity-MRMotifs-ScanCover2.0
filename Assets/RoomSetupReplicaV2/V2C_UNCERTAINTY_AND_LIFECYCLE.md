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
16 x 16 x 16 storage Chunk. Leaf-granular frustum GC therefore had to be tested
as a separate V2-C change. At this point in the investigation its native
trigger frequency was unresolved and could not be invented silently.

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
The first test retained a once-per-accepted-pair cadence solely to preserve the
V2-B variable; its receipt called this `Preserved_NotNativeConfirmed`.
Candidate and released leaves, cleared voxels, released parent Chunks, and pass
count were recorded for the same-tape A/B.

### Leaf-GC Quest 3 replay on 2026-10-26

The same 156-pair Quest Stage tape was replayed after commit `b30e69e`.

```text
algorithm                         V2CWhiteboxLifecycleLeafGcRegular10cmMesh
GC passes                         156
candidate/released leaves         524314 / 44912
cleared weighted voxel events     2519379
released parent Chunks            68
resident weighted voxels          632011 (-9.36%)
preview/final triangles           27418 / 27418 (+25)
post-cleanup components           12 (+1)
direct depth support              0.90849078 (+0.00059462)
plane RMS                         0.00818584 m (-0.08%)
mean correction delay             0.2618 s (-2.12%)
mean correction pair lag          0.3406 (-8.96%)
replay elapsed                    35.7199 s (+2.83%)
final mesh SHA-256                2F793B32B2F9BA2D3E3C80E1588E02F116AEC5378386925D17B907E22F14BECC
```

The near-identical visible surface is expected: most released leaves contain
positive/free-space SDF rather than a final zero crossing. The counters prove
that leaf GC executed and materially reduced the resident field, but the
preserved once-per-pair cadence is not accepted as native-equivalent. A welded
topology comparison located the extra surviving component as a 45-triangle
piece near `(3.25, 1.16, 0.44) m`, split from the former local component rather
than introduced as a distant island. This is a small but real topology
regression, accompanied by repeated clear/reintegrate churn and a 2.83 percent
replay-time cost.

Keep the recovered 4 x 4 x 4 leaf rule as the V2-C candidate mechanism. Do not
freeze the once-per-pair cadence.

### Native frequency gate recovered

A whole-binary scalar-reference scan linked the `postSdfFusion` member at
`this+0x18c` to the construction path and its registered configuration object:

```text
_INIT_413 at 0x58a5f0
    DAT_012423e0 -> DAT_012425c0
    config current/default double = -0.15

constructor path at 0x6cc740
    this+0x18c = float(*(DAT_012423e0 + 0x10))

postSdfFusion at 0x788f80
    run iff frequency > 0 and
        lastTimestamp + 1e9/frequency < currentTimestamp
    advance lastTimestamp when the pass starts
```

The native binary therefore defaults this optional lifecycle pass to disabled;
a positive runtime configuration override would enable the recovered frequency
gate. The replica now uses `-0.15 Hz` as the documented native default and does
not run GC for a non-positive or non-finite value. If later device evidence
recovers a positive runtime override, setting the serialized frequency applies
the same strict `last + 1/frequency < current` gate to capture timestamps. The
receipt records the configured frequency, the native default, whether GC was
enabled, and the source of the rule.

The next same-tape replay should consequently return to the accepted 11
post-cleanup components and original V2-C geometry while reporting zero GC
passes. That replay validates the gate/default integration; it does not erase
the separately proven leaf-GC mechanism or claim that no server-side override
can exist.

### Native-default gate Quest 3 replay accepted on 2026-10-26

The same 156-pair Quest Stage tape was replayed after commit `4b0aa81`.

```text
configured/default frequency      -0.15 / -0.15 Hz
GC enabled/passes                 false / 0
candidate/released leaves         0 / 0
cleared voxels/released Chunks    0 / 0
active Chunks                     409
weighted voxels                   716842
preview/final triangles           27397 / 27397
post-cleanup components           11
largest component                 21359 triangles
direct depth support              0.90790963
plane RMS/P95                     0.00818797 / 0.01647617 m
replay elapsed                    35.5396 s
final mesh SHA-256                3C528BE8B2615E06F48038252A7F93A9134F11D657F29782C042F9700D4C4B9D
```

The recovered default and execution gate are accepted: every GC counter is
zero and the 45-triangle split introduced by once-per-pair leaf GC is gone.
Welded component sizes are identical to the pre-leaf-GC V2-C replay except for
the largest component, which contains four additional triangles (`21355 ->
21359`). All ten smaller component sizes match exactly. The different final
hash is therefore not a new island or topology regression; disabling the old
invented cadence preserves slightly more of the main surface and 19,548 more
weighted voxels than the former whole-Chunk-GC baseline.

This accepts the compiled native default, not an unevidenced claim that Meta
can never provide a positive runtime override. The positive-frequency leaf
path remains implemented behind the recovered timestamp gate for future
runtime evidence.

## Final bounded uncertainty trace and scar-correction implementation

Static tracing is now closed at the private `Depth` / `SporadicDepthCompute`
packet boundary. The float uncertainty plane reaches
`filterDepthUncertainty` with the confirmed `> 89` rejection rule, but the
packet and its owner are carried by private capability callbacks. The only
related exports register system AOSP image/event callbacks; they do not return
the reconstruction depth packet or uncertainty plane. The exact rule remains
receipt-visible and unapplied. Confidence, TSDF weight, gradients, and temporal
stability are not substituted for it.

The next V2-C implementation consists of three independently receipted rules:

1. `last-replay-spatial-receipt.json` maps every final triangle lacking direct
   tape support into welded spatial clusters. Per-triangle centroid, normal,
   area, bounds, depth residual range, and evidence counts are retained. Both
   foreground and background residual signs classify a cross-layer connection
   candidate; repeated one-sided contradiction classifies a historical-shell
   candidate. The result is an offline receipt only and does not add a HUD.
2. Every source voxel retains its latest direct surface anchor. Before
   Marching Cubes, the eight corners of a 10 cm cell are projected onto that
   cell's SDF-gradient normal. A span greater than `0.15 m`, or a corner with
   no direct layer witness, leaves the cell empty. Tangential separation on
   the same wall is allowed. This is a depth-layer compatibility rule, not an
   eight-neighbour count.
3. A weighted voxel is locally retired only after three distinct stereo pairs
   directly provide a sign-opposite sample at least half a source voxel from
   zero. Confirming evidence resets the streak. No unobserved/out-of-range
   voxel is changed, and there is no global weight decay.

Both production rules are serialized switches and default enabled. The normal
replay receipt records their thresholds, rejected-cell/vote/retirement counts,
and the spatial receipt path. Quest APK and device acceptance remain a separate
step; source implementation alone does not accept the geometry.

### Frozen-baseline spatial receipt generated

The offline tool was run against the accepted native-GC-gate mesh and the
preserved 156-pair Quest Stage tape. It independently reproduced the existing
direct-support result exactly before classifying it:

```text
final triangles                    27397
triangles without direct support    2523
direct support ratio             0.9079096251
welded scar clusters                    130
cross-layer candidates                2213
historical-shell candidates            298
insufficient evidence                   12
mesh SHA-256       3C528BE8B2615E06F48038252A7F93A9134F11D657F29782C042F9700D4C4B9D
tape SHA-256       A37036993CF429E4386DFB5BF2201EC82C48C8DD05C19BDFC7630869417C0399
```

The resulting `last-replay-spatial-receipt.json` contains exactly 2,523
triangle records and 130 cluster records. Classification is evidence for the
next A/B test, not proof that every listed triangle is visibly objectionable.

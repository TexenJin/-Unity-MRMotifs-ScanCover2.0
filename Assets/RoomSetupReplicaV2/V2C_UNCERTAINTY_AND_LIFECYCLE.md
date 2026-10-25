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

# V2-B Regular 10 cm Mesh Baseline

Frozen from Quest 3 device replay on 2026-10-26. This is the geometry and
publication baseline for V2-C. Later uncertainty or lifecycle work must be
compared against the same tape and must not silently enable XRT simplification.

## Accepted route

```text
5 cm sparse projective SDF
  -> aligned direct sample onto the 10 cm derived field
  -> 10 cm Marching Cubes
  -> RemoveSmallConnectedComponents(<32), below the 40000-face gate
  -> publish unchanged (XRT disabled)
```

The user visually confirmed that the result returned to the requested regular
Room Setup-style coverage mesh rather than the highly abstracted large-triangle
mesh produced by metric-error simplification.

## Device receipt

```text
algorithm                         V2BWhiteboxDirectSampleRemoveSmall32Regular10cmMesh
replayed stereo pairs             156
active chunks                     407
weighted voxels                   697294
pre-cleanup triangles             27909
pre-cleanup components            138
removed <32 triangles             510
removed <32 components            127
post-cleanup components           11
pre-simplification triangles      27393
post-simplification triangles     27393
accepted collapses                0
simplification applied            false
simplification stop               DisabledForRegular10cmMesh
final exported vertices           18345
final exported triangles          27393
final connected components        11
largest component triangles       21355
plane residual RMS                0.008192615 m
plane residual P95                0.016488621 m
direct-depth support ratio        0.90789616
triangles without direct support  2523
first mesh pair                   1
first mesh latency                0.2951 s
replay elapsed                    34.9643 s
replay throughput                 4.4617 pairs/s
final mesh SHA-256                4E12E007F652E1EE13203B298B1E665D6687B35A42E2AC32A7C9E083EE3F5906
```

## Acceptance boundary

- XRT metric-error simplification remains preserved in source but disabled.
- Replay-only controls remain active so the fixed tape cannot be overwritten.
- V2-C may change uncertainty admission and mesh lifecycle only in isolated,
  receipt-visible steps.
- Any regression to large, highly abstracted triangles is a route failure, not
  an acceptable performance tradeoff for this baseline.

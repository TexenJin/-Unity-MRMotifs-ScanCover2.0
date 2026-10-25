# V2-B white-box resampling contract

This phase changes only the `5 cm source field -> 10 cm derived field` rule.
Fusion, truncation, source voxel values, scheduling, Marching Cubes, rendering,
and the recorded Stage-space input remain unchanged.

## Recovered active behavior

The active Room Setup mesh mode is `Resampled` and calls native function
`0x78cd20` with `voxelSizeRatio = 0.5`.

The function supports both exact copying and trilinear interpolation:

1. It maps allocated source coordinates into the derived volume.
2. For a derived integer coordinate, it computes the source sampling
   coordinate as `derived / ratio`.
3. It selects the exact branch when each source sampling coordinate has zero
   remainder under the ratio alignment test.
4. Otherwise it reads eight source SDF values and performs trilinear
   interpolation.

At the active ratio of `0.5`, every derived integer coordinate maps to an even
integer source coordinate. The exact branch is therefore the active 5 cm to
10 cm kernel. It copies the stored half SDF and half weight. The trilinear
branch is compiled support for other ratios, not evidence that the active
Room Setup configuration interpolates eight neighbours.

The equivalent reconstruction rule is:

```text
derived[i,j,k] = source[2*i, 2*j, 2*k]

if source weight <= 0:
    derived sample remains unobserved
else:
    copy source SDF and weight exactly
```

The derived lattice uses its own 10 cm coordinate grid. It must not inherit the
source field's 2.5 cm voxel-centre offset.

## Controlled comparison

The previous strict eight-neighbour confidence-weighted candidate is retained
in Git history and in the captured V2-A artifacts. It produced useful evidence
that smoothing can reduce fragmentation, but it is no longer presented as the
recovered Room Setup resampling kernel.

V2-B now tests the white-box-aligned exact-copy rule against the same tape and
the same receipts. Room Setup's confirmed connected-component cleanup and
metric-error simplification remain later, separate production stages; their
effects must not be silently folded into the resampler.

## Static evidence

- `libmrengine.so` SHA-256:
  `0E558058C08A6E22EA6E2FB1C8B7890BE5A472B515076C84CF643C57F112BABE`
- `0x78c4b0`: active `Resampled` orchestration.
- `0x78cc60`: derived-volume reset.
- `0x78cd20`: ratio validation, aligned exact copy, and non-aligned trilinear
  fallback.
- Active configuration: source pitch `0.05 m`, ratio `0.5`, derived pitch
  `0.10 m`.

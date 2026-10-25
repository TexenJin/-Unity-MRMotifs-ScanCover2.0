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

The source field follows the same integer-grid convention. Native
`fuseDepthMap` (`0x7a5400`) converts sparse-tree integer voxel coordinates to
float and applies the spatial scale directly; it does not add `0.5` in voxel
space. The `+0.5` terms visible in that function are part of image-pixel
centre/principal-point handling. Consequently, both 5 cm fusion samples and
10 cm derived samples are anchored at integer grid points:

```text
source world position  = sourceIndex  * 0.05 m
derived world position = derivedIndex * 0.10 m
sourceIndex             = derivedIndex * 2
```

Keeping the old source `index+0.5` convention while removing the offset only
from the derived mesh creates a uniform 2.5 cm disagreement between the field
and its published surface. The first V2-B device run isolated exactly that
translation and is retained as evidence, not as the final coordinate rule.

## Controlled comparison

The previous strict eight-neighbour confidence-weighted candidate is retained
in Git history and in the captured V2-A artifacts. It produced useful evidence
that smoothing can reduce fragmentation, but it is no longer presented as the
recovered Room Setup resampling kernel.

V2-B now tests the white-box-aligned exact-copy rule against the same tape and
the same receipts. Its next recovered stage is kept separate from resampling:
after Marching Cubes, the full candidate mesh is classified into global
connected components and any component with strictly fewer than 32 triangles
is removed when the pre-cleanup mesh has fewer than 40,000 triangles. A
32-triangle component survives. Chunk-boundary duplicates are welded at
0.1 mm for connectivity classification only; vertex positions are unchanged.

The unfiltered candidate remains stored behind the published Front mesh. This
allows a fragment removed on one pass to grow past the threshold on a later
pass. Publication happens after the whole mesh pass so connectivity is never
decided independently per chunk. The replay receipt records pre/post component
and triangle counts plus cleanup time. Metric-error simplification remains a
separate production stage and is not silently folded into either the resampler
or this cleanup. It is now implemented immediately after cleanup as a global
XRT-style directed-edge collapse pass. The recovered wrapper and kernel
establish the following contract:

```text
minimum input                 100 triangles
target                        current vertices / 20
random generator              MT19937, seed 13
candidates per batch          16 unique directed triangle corners
accepted cost                 cost < 0.075 m
optional regularizer factors  float bits 0x00000080 and 0x00000020
normal compatibility          oldNormal dot newNormal >= 0.1
failed-batch stop             200 total failed batches
```

Each candidate collapses one existing endpoint onto the other. No midpoint,
plane fit, vertex smoothing, or new surface position is introduced. Standard
link-condition checks prevent a collapse that would make the local triangle
topology non-manifold, while degenerate faces and excessive normal changes are
rejected before the physical metric is considered. The target is a lower
bound request, not permission to violate the error ceiling.

The implementation preserves an explicit evidence boundary. The wrapper
constants, XRT selection, seeded 16-candidate sampling, directed endpoint
collapse, topology/normal gates, target condition and 200-failure condition
are recovered behavior. The C# expression of the per-face physical cost is a
clean-room equivalent of native `0x797180`, not a claim of instruction-level
or floating-point-order identity. The replay receipt exposes every reduction,
rejection and stop reason so the same tape can reveal a mismatched metric
instead of hiding it behind a visually pleasant result.

An initial implementation incorrectly associated the nearby `0.1 / 0.5`
constant pair with the XRT regularizer fields. Device replay exposed the
mistake: only 130 of 14,829 vertices collapsed before 200 failed batches.
Rechecking constructor assembly at actual ELF `0x60dc20` shows that it loads
eight bytes from VA `0x120060`, whose two dwords are `0x00000080` and
`0x00000020`. Interpreted by the later floating-point instructions and
multiplied by the base error, both optional terms underflow to near zero.
Consequently the active physical metric is overwhelmingly accumulated
point-to-plane displacement; `0.1 / 0.5` belongs to a different constant pair
and must not be substituted.

## Static evidence

- `libmrengine.so` SHA-256:
  `0E558058C08A6E22EA6E2FB1C8B7890BE5A472B515076C84CF643C57F112BABE`
- `0x78c4b0`: active `Resampled` orchestration; invokes
  `RemoveSmallConnectedComponents` after extraction and before simplification
  when the active triangle count is below 40,000.
- `0x78cc60`: derived-volume reset.
- `0x78cd20`: ratio validation, aligned exact copy, and non-aligned trilinear
  fallback.
- `0x7a5400`: integer sparse-tree coordinates are scaled without a voxel-space
  half-cell addition.
- `0x789ab0`: simplification wrapper, minimum input, target and three error
  arguments.
- `0x796400`: active XRT kernel, seed 13, 16-candidate batches and 200 failed
  batches.
- `0x796f70`: directed-collapse topology legality.
- `0x797180`: local physical-error and normal-compatibility evaluation.
- `0x797820`: half-edge collapse and adjacency rewiring.
- Active configuration: source pitch `0.05 m`, ratio `0.5`, derived pitch
  `0.10 m`.
- Recovered cleanup threshold: remove components with `<32` triangles.

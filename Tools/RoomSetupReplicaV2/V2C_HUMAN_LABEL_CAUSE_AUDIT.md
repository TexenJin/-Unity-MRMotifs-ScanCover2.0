# V2-C human-labelled scar cause audit

## Scope and provenance

- Frozen replay mesh: `0A022B000599E5748486A91AFEA474DE8F7B1084F552321CDA556E19A50E3B9E`
- Frozen replay tape: `A37036993CF429E4386DFB5BF2201EC82C48C8DD05C19BDFC7630869417C0399`
- Final human selection: 650 unique triangles in 28 surviving strokes.
- This report classifies the 128 selected triangles that did not satisfy the
  earlier `RepeatedMixedSignSourceVoxel` threshold. It does not treat
  unlabelled triangles as ground-truth negatives.

## Result

The 128 triangles are not evidence that the earlier source audit was wrong.
They expose a second, more specific failure mode that the repeated-mixing rule
was not designed to detect.

| Mutually exclusive class | Triangles | Evidence |
| --- | ---: | --- |
| Derived-grid zero-plane phase ambiguity | 104 | Non-axis-planar MC case and face centroid within 5 mm of the dominant 10 cm lattice plane |
| Weak temporal/layer switching below the old repeat threshold | 17 | Non-axis-planar MC case farther from the lattice plane; every face has a temporal sign switch and/or a layer-anchor jump |
| Axis-planar neighbour selected by the paint stroke | 7 | Final MC configuration is already a simple plane; current evidence does not prove that the triangle itself is malformed |

Together with the 522 repeated-mixed triangles, 643/650 human-selected
triangles now have a source/extraction explanation. The last seven are kept as
selection-boundary cases instead of being forced into a causal class.

## Dominant floor scar

Stroke 1 contains 82 triangles in 52 derived cells and spans approximately
1.75 m by 0.37 m while remaining within 14 mm of the floor plane.

The three aligned source-sample layers are:

- `y = -0.10 m`: 58/58 samples negative, median SDF `-84.6 mm`;
- `y = 0.00 m`: 26 negative and 33 positive samples, median absolute SDF
  `3.10 mm`;
- `y = +0.10 m`: 54/54 samples positive, median SDF `+90.7 mm`.

On the `y = 0` sample plane, 31 of 90 neighbouring sample edges change sign.
The physical floor therefore remains a single layer, but its SDF lies so close
to the exact derived-grid plane that millimetre-scale residual differences
create a checkerboard of signs. Adjacent cells alternately extract the surface
from the cell below or the cell above.

This changes topology even when the reconstructed vertices remain nearly
coplanar:

- 51/52 selected cells use a non-planar Marching Cubes configuration;
- only 59/296 nearby unlabelled floor cells do so;
- the selected sample-plane absolute SDF median is `2.85 mm`, versus
  `13.60 mm` in the nearby unlabelled floor comparison.

The visual result is the long irregular triangle band the user labelled. It is
not an unsupported island or a large-separation foreground/background bridge.

## Vertical-wall confirmation

The same pattern appears on the selected wall strip near `x = 0.60 m`:

- 16/17 selected comparison cells use a non-axis-planar MC configuration;
- 74/230 nearby unlabelled wall cells do so;
- every selected unresolved wall cell has a temporal sign switch and 16/17
  have a layer-anchor jump.

This rules out a floor-only special case. The common condition is a physical
surface near an aligned 10 cm sample plane plus unstable near-zero signs.

## Why the current V2 rules allow it

1. The active 5 cm to 10 cm path copies exact aligned source samples. It does
   not spatially regularize the near-zero sample plane.
2. Marching Cubes classifies each corner with the strict test
   `distance < 0`. There is no near-zero hysteresis or shared plane decision.
3. Same-layer support rejects only a zero-crossing edge whose two latest
   anchors come from the same stereo pair and differ by more than 15 cm. These
   scars are normally the same physical layer and differ by millimetres, so
   that rule must not reject them.
4. `<32` connected-component cleanup cannot remove the band because it is
   attached to the main floor or wall.
5. Contradiction retirement was disabled in the audited replay. This matters
   for the 17 weak temporal remnants, but it does not by itself solve the
   zero-plane phase ambiguity.

## Repair boundary

The production repair should have two local stages:

1. **Near-zero plane stabilization at the derived-cell layer.** When an
   observed, locally coherent surface has several shared corners within a
   small zero band, derive one consistent local plane/gradient decision for
   those shared samples. Do not apply a global SDF bias and do not smooth real
   edges.
2. **Evidence-driven retirement for weak temporal remnants.** Retire an old
   local sign only after repeated, spatially consistent opposite direct
   observations. Preserve unobserved correct surfaces and avoid global weight
   decay.

The seven axis-planar paint-boundary triangles must remain outside automatic
repair acceptance unless a denser point selection or adjacent-fold expansion
shows that their geometry is itself defective.

## Acceptance checks on the frozen input

- Human-positive explanation coverage: `643/650` without counting the seven
  boundary triangles as confirmed defects.
- Stroke-1 non-planar derived cells: reduce from `51/52` while preserving one
  continuous floor sheet.
- Nearby unlabelled floor and wall controls: do not create holes or move the
  dominant plane.
- Preserve all previously accepted `<32` cleanup and same-layer-support
  behaviour.
- Compare the same mesh hash input before and after; do not use a new scan to
  judge this repair.

## Full-mesh localization of the 104-face signature

The same rule was applied to all 26,656 triangles after reconstructing the
aligned 10 cm cell evidence for the full frozen mesh. The result is deliberately
split by evidence strength:

| Evidence level | Triangles | Interpretation |
| --- | ---: | --- |
| Human-confirmed zero-plane class | 104 | Exact controller-painted positives; base rule recall is 104/104 |
| High-confidence same-problem discoveries | 344 | All 72 shared-edge continuations plus 272 independent atlas matches |
| Probable same-problem discoveries | 294 | Independent atlas matches between the positive leave-one-out 90th and 95th percentile distances |
| Broad static-signature hits | 1,050 | Base-rule matches with insufficient atlas similarity; review-only |

The feature atlas is rotation-independent and includes distance to the 10 cm
plane, dominant-normal deviation, triangle area, MC crossing/negative-corner
structure, near-grid-layer SDF median/minimum/spread, temporal switches, layer
jumps, same-signature neighbours, and component size. Its high threshold is
the 90th percentile of positive leave-one-out nearest-neighbour distance; the
probable threshold is the 95th percentile. Leave-one-out recall is 93/104 and
98/104 respectively. Manually confirmed faces remain pinned as positives.

The 1,792 base-signature triangles form 571 shared-edge components. Sixteen
components contain at least one of the 104 confirmed faces. A legitimate
physical plane can still lie within 5 mm of the world-anchored 10 cm lattice
and use a non-axis-planar Marching Cubes case, so the remaining 1,050 broad
hits are not counted as defects. This is why the full-mesh result is a ranked
localization receipt rather than a deletion mask.

The visualization also keeps the other 546 human-confirmed faces visible as a
separate class. Those faces belong to repeated mixed-source, weak temporal
switching, or paint-boundary categories and must not be presented as evidence
for the 104-face zero-plane mechanism.

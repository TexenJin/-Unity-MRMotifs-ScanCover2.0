# V2-C scar-source audit

## Accepted inputs and boundary

This audit is read-only. It changes neither fusion nor extraction.

- Tape: the frozen 156-pair Quest Stage `last-session.rsv2`, SHA-256
  `A37036993CF429E4386DFB5BF2201EC82C48C8DD05C19BDFC7630869417C0399`.
- Mesh: the native-GC-gate regular 10 cm baseline, SHA-256
  `3C528BE8B2615E06F48038252A7F93A9134F11D657F29782C042F9700D4C4B9D`.
- Targets: all 2,523 final triangles without direct tape support from the
  frozen spatial receipt.
- Output: `last-replay-source-audit.json` beside the frozen baseline capture.

The offline audit reconstructs the 4,741 unique aligned source corners used by
the targets' 1,494 derived 10 cm cells. These are exact samples of the 5 cm
source lattice, not a second mesh approximation. All 2,523 target triangles
landed in cells whose reconstructed final corner signs crossed zero. The 100%
crossing reproduction is the acceptance boundary for the source analysis.

## Matrix, stereo-rig, and timing integrity

```text
worldToClip * viewInverse * projectionInverse max error   2.15e-6
stereo baseline mean                                     0.06484998 m
stereo baseline min/max                   0.06484945 / 0.06485040 m
non-increasing capture timestamps                                0
non-increasing platform frame numbers                            0
capture interval mean/max                       0.3256 / 0.4485 s
```

The recorded matrix chain, stereo baseline, and ordering are internally
consistent. This rules out corrupt tape matrices, a drifting recorded stereo
baseline, and timestamp reversal. It cannot rule out latency that already
existed between depth and pose before the callback-owned packet was recorded,
because the tape has no external pose ground truth.

Motion correlation narrows that remaining boundary:

```text
                                      all pairs   mixed-corner switches
linear speed median                    0.1632             0.1779 m/s
linear top-quartile share               25%                24.36%
angular speed median                    19.82              32.55 deg/s
angular top-quartile share               25%                37.51%

same-view mixed-corner switches:
linear top-quartile share                                  20.85%
angular top-quartile share                                 34.82%
```

The switches do not concentrate in fast translation, so a gross translational
pose mismatch is not the general cause. They are moderately enriched during
head rotation, which leaves rotational depth/pose latency as a contributor to
the mixed-source subset, not as an explanation for all scars. Raw depth-layer
instability remains necessary because many same-view switches also occur below
the top motion quartile.

## Source evidence

Counts overlap; a triangle may contain both a mixed source voxel and a
crossing edge whose dominant surface anchors occupy different layers.

```text
target unsupported triangles                         2523
any same-pair left/right sign opposition               90   (3.57%)
repeated same-pair opposition                            3   (0.12%)
repeated temporal sign switching                       237   (9.39%)
repeated positive and negative evidence in one
aligned 5 cm source corner                             599  (23.74%)
  includes a cross-sign observation from nearly
  the same eye position and viewing ray                391  (15.50%)
  includes view-separated cross-sign evidence          382  (15.14%)
crossing edge whose endpoints' cross-frame dominant
anchors differ by more than 0.15 m along the SDF
normal                                                 1071  (42.45%)
union of repeated source mixing or dominant-anchor
layer bridging                                        1383  (54.82%)
neither repeated mixing nor dominant-anchor bridge    1140  (45.18%)
```

Exclusive partition:

```text
no repeated mixed corner, compatible dominant anchors 1140
no repeated mixed corner, dominant-anchor bridge       784
repeated mixed corner, compatible dominant anchors     312
repeated mixed corner and dominant-anchor bridge       287
```

## Spatially matched supported controls

The same audit was run on 2,523 unique **directly supported** triangles,
matched to the targets by nearest centroid while preferring the same 0.8 m
source Chunk. Mean/P95 match distance is 0.211/0.521 m. This control falsifies
two initially plausible but non-discriminative rules:

```text
                                               unsupported   supported control
same-pair stereo opposition                             90                 285
repeated mixed-sign 5 cm source corner                 599                 809
same-view repeated mixed-sign corner                   391                 678
dominant-anchor separation >0.15 m                    1071                 459
separation >0.30 m in a welded component
of at least 8 similarly separated triangles            459                  32
```

Stereo disagreement and mixed-sign history occur more often in correct,
directly supported controls. They are environmental/input conditions, not
standalone defect detectors. By contrast, the combined high-separation and
spatial-coherence rule is present in 18.19% of unsupported targets and only
1.27% of the matched supported controls: a 14.3x enrichment. The production
detector additionally requires absent direct support, so the matched controls
fail that predicate by construction.

The high-confidence rule used by the 3D model is therefore:

```text
no direct depth support
AND dominant crossing-edge anchor separation > 0.30 m
AND welded high-separation component contains >= 8 triangles
```

It identifies 459 triangles in 11 coherent components. The wider 0.15 m rule
remains a candidate overlay only; it is not labeled as a confirmed defect.

## What is established

1. Same-pair stereo opposition is not the dominant source. It appears at least
   once in only 90 targets and repeats in only three.
2. Repeated source mixing is real but is **not** a defect classifier. It is
   more prevalent in the matched correct controls than in the unsupported
   targets. The 599 target occurrences can explain history inside a confirmed
   region, but cannot authorize fusion changes or triangle deletion alone.
3. Spatially coherent, high-separation layer bridging is the discriminative
   mechanism. The 459 high-confidence triangles have no direct support, exceed
   0.30 m endpoint-anchor separation along the SDF normal, and form components
   of at least eight such triangles. The matched-control rate is 14.3x lower.
4. The latest-pair-only rule was safe but weak because accumulated dominant
   layers usually do not coexist as comparable endpoint witnesses in the same
   final stereo pair.
5. The remaining unsupported triangles are not thereby proven scars.
   A triangle may lack a sample within 5 cm and still be a correct ordinary
   TSDF interpolation. They must not be deleted merely because they are in the
   unsupported receipt.

## Cluster differences

The causes are not uniform across space. Examples among the largest clusters:

```text
cluster  triangles  mixed-source  dominant bridge  interpretation
89             726            73              282  mostly spatial bridge/unknown
51             251           161              164  mixed fusion plus bridge
1              164             9              137  predominantly spatial bridge
54             161           109              103  mixed fusion plus bridge
60             101            24               12  mainly same-view instability/unknown
40              32            29               19  strongly mixed source evidence
```

Consequently one global decay, retirement, mixed-sign, or neighbourhood-count
threshold cannot be a faithful correction.

## General recognition signatures

### Source-voxel layer mixing clue

Use all of the following, not appearance alone:

- the same world-anchored 5 cm source sample receives at least three positive
  and three negative observations;
- its observation sign changes repeatedly, rather than once at a normal
  occlusion boundary;
- cross-sign observations recur from nearly the same eye position and ray, or
  persist after controlling for viewpoint;
- the final zero crossing has weak/no direct support and lies in the affected
  source sample's local cell.

The characteristic geometry can be doubled, corrugated, ring-like, or
wandering, but the matched controls prove that this history is also common on
correct surfaces. It is an explanatory clue only when a triangle independently
passes the unsupported, high-separation, spatial-coherence rule.

### Derived-cell layer bridge

The signature is different:

- the actual Marching-Cubes edge crosses zero;
- its endpoints may each be temporally stable;
- the endpoints' cross-frame dominant surface anchors differ by more than
  0.30 m along the cell SDF-gradient normal;
- at least eight welded neighbouring triangles pass the same separation test;
- the resulting triangle lacks direct depth support near its centroid/edges.

This produces coherent ribbons, V-folds, long false connectors, or surfaces
across a gap. A count of valid neighbours cannot identify it because all eight
samples may be numerically valid while belonging to incompatible layers. The
0.15 m rule remains useful for broad inspection, but not for automatic defect
classification.

### Evidence-insufficient interpolation

An unsupported triangle without a coherent high-separation bridge is not a
confirmed defect. Ordinary TSDF zero crossings are
interpolated and need not have a raw depth sample at the final triangle
centroid. These triangles remain visible and are colored separately in the 3D
inspection model.

## Correction route

Only the coherent high-separation bridge currently passes matched-control
validation. The repair order follows that evidence.

1. **10 cm zero-crossing edge compatibility.** Accumulate a robust dominant
   anchor per aligned source corner across frames. Inspect only the edges that
   actually cross zero. If the final triangle is unsupported, endpoint anchors
   are mature, normal separation exceeds 0.30 m, and the condition is spatially
   coherent across at least eight welded triangles, leave that local crossing
   empty or retain separate layer hypotheses; do not join the edge. Missing
   evidence is not permission to erase the cell. Keep the broad 0.15 m layer as
   receipt-only evidence. Use identical world-anchored samples on both sides of
   Chunk boundaries.
2. **Publication.** Preserve the old correct Front while a corrected local
   candidate is incomplete. Publish per local seam-connected component; do not
   hide the whole room and do not use a global display gate as a correction.
3. **Mixed-sign source history.** Do not change production fusion on this
   evidence alone. Re-evaluate it only inside residual defects after the bridge
   A/B. If a later matched-control test validates a fusion rule, keep a bounded
   challenger layer, require repeated coherent takeover, update current voxels
   only, and never decay unobserved/out-of-range geometry.
4. **Evidence-insufficient triangles.** Make no production change. They are
   controls for false-positive measurement and must remain in every A/B.

The acceptance test is not just fewer unsupported triangles. A successful A/B
must remove the colored confirmed regions, retain the evidence-insufficient
controls, keep the connected-component structure, and avoid increasing holes,
largest-component loss, correction latency, or GPU/CPU cost beyond the stated
budget.

## Remaining identification boundary

The receipt now identifies causal evidence in Quest Stage space, but the old
screen recording did not save its viewer-camera matrices. Therefore it still
cannot prove which numbered spatial cluster corresponds to each hand-circled
video scar. A one-pass viewer-pose receipt during the same replay is sufficient
for that last mapping; the room depth tape does not need to be recorded again.

Until that mapping exists, the safe implementation conclusion is limited to:

- do not add another voxel-retirement rule;
- do not reject all unsupported triangles;
- test the validated unsupported + 0.30 m + eight-connected dominant-layer
  bridge rule as a separate extraction A/B;
- keep the 0.15 m candidates and mixed-source history receipt-only until the
  high-confidence A/B exposes any residual class.

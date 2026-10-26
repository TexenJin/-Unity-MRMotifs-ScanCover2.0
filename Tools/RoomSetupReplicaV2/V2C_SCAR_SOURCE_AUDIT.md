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
  includes only view-separated cross-sign evidence     382  (15.14%)
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

## What is established

1. Same-pair stereo opposition is not the dominant source. It appears at least
   once in only 90 targets and repeats in only three.
2. There is a real source-fusion problem, but it is localized rather than
   universal. 599 targets contain a 5 cm aligned source corner with at least
   three positive and three negative pair observations. 391 targets include
   such disagreement from nearly the same viewpoint, so viewpoint-dependent
   occlusion alone cannot explain all of it. The remaining ambiguity is raw
   depth instability versus depth/pose latency already present in the packet.
3. The larger confirmed mechanism is spatial layer bridging at extraction.
   In 1,071 targets, the two endpoints of a final zero-crossing edge have
   cross-frame dominant surface anchors separated by more than 0.15 m along
   the cell SDF normal. This explains why the earlier latest-pair-only rule was
   safe but weak: most endpoints do not carry comparable witnesses from the
   same last stereo pair even though their accumulated dominant layers are
   incompatible.
4. The 1,140 remaining unsupported triangles are not thereby proven scars.
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

Consequently one global decay, retirement, or neighbourhood-count threshold
cannot be a faithful correction.

## Remaining identification boundary

The receipt now identifies causal evidence in Quest Stage space, but the old
screen recording did not save its viewer-camera matrices. Therefore it still
cannot prove which numbered spatial cluster corresponds to each hand-circled
video scar. A one-pass viewer-pose receipt during the same replay is sufficient
for that last mapping; the room depth tape does not need to be recorded again.

Until that mapping exists, the safe implementation conclusion is limited to:

- do not add another voxel-retirement rule;
- do not reject all unsupported triangles;
- test cross-frame dominant-layer compatibility at zero-crossing edges as a
  separate extraction A/B;
- treat the 599 repeated mixed-source targets separately from the 1,071
  spatial bridges instead of forcing one threshold to solve both.

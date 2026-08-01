# Directional Marching Cubes route

Goal: establish a reproducible paper-reference baseline before any ScanCover
customization. The visible yellow mesh remains an isolated, read-only shadow
and never overwrites the production mesh.

## Current phase: locked paper topology plus local sampling refinement

The coarse implementation follows Splietker et al., *Directional TSDF: Modeling
Surface Orientation for Coherent Meshes* (IROS 2019), and the authors'
`MeshHashingDTSDF` reference source:

- six directional TSDFs in the authors' order `Y+, Y-, X+, X-, Z-, Z+`;
- signed SDF-gradient compliance with threshold `sin(pi/8)`;
- the authors' exact MC-index decomposition table;
- the authors' exact per-direction compatibility table and oriented
  opposite-edge test;
- Algorithm 1 inter-directional weighted voting;
- at most two orientation-aware intersections per physical grid edge;
- at most two combined MC indices per cell using the authors'
  `MCIndexCompatible` rule;
- neighborhood majority regularization of the first combined index;
- the authors' classic MC triangle table;
- triangle-local deferral when a required edge intersection is unavailable.

Quest replay established the paper branch as the coherent baseline, while also
showing that a 0.10 m lattice cuts measured right-angle folds diagonally and
cannot correct a mature depth challenger whose identity drifts into an adjacent
voxel/direction.  The first isolated extension therefore changes evidence
resolution, not topology:

- stable orthogonal surface families or same-direction depth spread request a
  bounded 0.05 m local TSDF block after two consecutive batches;
- the local block executes the same paper direction filter, Algorithm 1,
  component table, two-index limit, edge slots and classic MC triangle table;
- missing fine corners may inherit a direction-preserving trilinear prior from
  the coarse TSDF only when at least 75% of the interpolation basis is observed
  at sufficient weight; measured fine samples always take precedence;
- the bounded active set is scheduled as one strongest persistent seed plus
  its six face-neighbor blocks, avoiding a scattered top-N set in which nearly
  every refined cell lies on an unsafe coarse/fine seam;
- a coarse cell is replaced only when the fine cell has complete evidence;
- cells at an active/inactive refinement boundary remain coarse, unless the
  adjacent block is also active;
- mature depth challengers persist by local plane identity across neighboring
  voxels and direction channels, with stricter evidence for risky sign changes.

The Quest paper-input port must feed this refinement extension explicitly.
`IntegratePaperNormalRaycast` now records the same accepted projective surface
point, neighbour normal, distance/view weighting and compatible direction
sectors used by the coarse paper TSDF.  Creases are detected from two stable,
distinct measured normal families rather than from sector fan-out (one oblique
plane may legitimately write several sectors).  A persistent two-surface paper
DMC cell is an independent refinement request.  These rules only select bounded
fine blocks; they do not change the paper MC indices, edge slots, voting or
triangle table.

The reference tables are copied byte-for-byte from author source. Diagnostics
must report:

`directional_mc_shadow_paper_reference=1`

and semantics beginning:

`paper_reference_direction_filter_index_compatibility`

## Reference-baseline gates

- the project compiles without new errors;
- all three embedded lookup tables match author source byte-for-byte;
- the yellow shadow is fed only by the paper-reference branch;
- MC33, certificate-driven topology, quarantine, synthetic corner completion,
  distance clustering, face trimming and winding repair remain unreachable
  from the paper-reference entry;
- frozen-volume desktop and Unity decisions can be compared without a hidden
  fallback to the legacy/custom DMC path;
- Quest 3 output is judged first for correct surface separation and coherent
  topology, not for matching the old six-color mesh cosmetically.

## Locked next order

1. Collect paper-reference Quest 3 output with non-zero projective refinement
   entries, buffered samples and fine DMC cells after the second batch.
2. Compare the same frozen volume at coarse 0.10 m and bounded local 0.05 m,
   using identical author-rule topology in both branches.
3. Classify remaining failures as input/fusion, paper-rule port, missing data,
   or an actual limitation of the reference method.
4. Only after that classification, add one isolated customization at a time,
   behind a separate switch and with a paper-reference A/B baseline.
5. Only after topology is stable, optimize steady-state extraction and
   publication cost.

## Prohibited before the baseline is measured

- no MC33 or custom ambiguous-face rule;
- no certificate-driven topology;
- no persistent shared-face voting or face trimming;
- no synthetic unknown-corner completion;
- no extra direction layers or arbitrary slot expansion;
- no generic smoothing, aggressive hole filling or global voxel-size reduction;
- no global half-voxel promotion or alternate fine-grid topology;
- no production-mesh replacement.

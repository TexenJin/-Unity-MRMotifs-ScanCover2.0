# V2-C edge-signature homology and source trace

## Scope

This trace combines three independent human inputs on the same frozen replay:

- the original 650 painted triangles;
- the 104-face derived-grid zero-plane subclass;
- the later 48 manually selected representative edges, expanded to their 83
  exact owner triangles.

The 83 edge-owner triangles do not overlap the earlier 650 painted triangles.
They therefore provide an independent confirmation rather than a re-count of
the same labels.

## Homology result

The earlier 650 triangles split into mutually exclusive classes:

| Class | Triangles | Share |
| --- | ---: | ---: |
| Repeated mixed source voxel | 522 | 80.31% |
| Derived 10 cm zero-plane phase ambiguity | 104 | 16.00% |
| Weak temporal/layer switching | 17 | 2.62% |
| Paint boundary or unresolved | 7 | 1.08% |

Thus 643/650, or 98.92%, have a source/extraction explanation. The new edge
atlas independently shows:

- 82/83 owner triangles with repeated mixed-sign source voxels;
- 79/83 with same-view repeated mixing;
- 76/83 with repeated temporal sign switching;
- 45/48 selected edges are ordinary two-owner manifold edges, excluding an
  unwelded Chunk seam as the general explanation.

The three user-visible descriptions are highly homologous but not synonymous:

1. **Repeated layer/sign switching** is the source-history observation.
2. **History/depth-layer competition edge chains** are its coherent geometric
   manifestation after fusion and extraction.
3. **10 cm zero-plane phase ambiguity** is an extraction amplifier. It can
   coexist with repeated mixing, as on the new long wall chain, or occur
   without meeting the old repeat threshold, as in the separate 104-face
   class.

The common failure family is therefore `unstable local source sign/layer ->
single-field TSDF compromise -> independent 10 cm sign classification ->
coherent scar edge chain`. It is not a Chunk-welding, simplification, or
small-component-cleanup failure.

## Full-mesh edge expansion

The 48 representative edges are pinned positives. A conservative full-mesh
expansion requires the learned MC configuration family, repeated mixed-source
evidence, compatible edge length/dihedral, and either an exact derived-cell
phase boundary or a connected continuation:

| Evidence class | Edges |
| --- | ---: |
| Manual positives | 48 |
| High-confidence phase-boundary discoveries | 208 |
| High-confidence connected continuations | 317 |
| Probable, review-only discoveries | 187 |

The mesh contains 23,190 edges with some repeated mixed-source signature.
22,431 of them are deliberately excluded because source mixing without the
learned geometry/topology is not a defect classifier. This prevents the source
clue from becoming a global destructive filter.

## Source chain in the replica

### 1. Input packet admits unstable raw depth

`RoomSetupReplicaV2Scanner` deliberately sends the callback-latched raw stereo
environment depth into the replica and bypasses ScanCover temporal, bilateral,
and edge filters. `RoomSetupReplicaV2Unproject.compute` rejects only invalid
NDC/range values. The reconstruction then accepts a pixel when its four-neighbour
depth-gradient magnitude is at most the fixed threshold.

The native Room Setup uncertainty rule is known (`uncertainty > 89` rejects the
depth sample), but the public Meta OpenXR path exposes no frame-paired native
uncertainty plane. The replica records `uncertaintyInputAvailable=false` and
does not replace it with a guessed confidence signal. This is the strongest
known input-stage difference that can explain why the official mesh shows the
same family less often, but its exact contribution cannot be measured without
the private uncertainty buffer.

### 2. Fusion merges competing hypotheses into one scalar

For every accepted eye sample, `IntegrateChunk` computes one projective SDF
sample and folds it into a single weighted average. Its weight depends only on
view depth. There is no uncertainty weight and no foreground/background layer
hypothesis. Opposite direct evidence is counted, but
`ContradictionRetirementEnabled` is false in the audited baseline, so mature
old evidence is not replaced after a coherent opposing takeover.

This explains how same-view depth-layer flips persist as small positive and
negative values around the same world-anchored 5 cm source sample.

### 3. Aligned 10 cm derivation preserves the unstable sign exactly

The recovered active 5 cm to 10 cm path copies `source[2*i,2*j,2*k]` exactly.
It does not average or stabilize the shared near-zero plane. This behavior is
white-box aligned and should not be replaced with an invented generic smoother.

### 4. Strict Marching-Cubes sign classification amplifies millimetres

`RebuildChunkMesh` independently classifies every derived corner with
`distance < 0`. Adjacent cells can therefore choose opposite MC phases when a
shared physical floor or wall lies within millimetres of the exact 10 cm plane.
The result is the irregular band or long edge chain even when the physical
surface is one continuous layer.

### 5. Existing guards do not target this family

- Same-layer support compares endpoint anchors only when both come from the
  same stereo pair and rejects separation above 0.15 m. Most marked scars are
  millimetre-scale same-layer phase conflicts or older-pair history, so they
  remain intentionally uncomparable.
- `<32` cleanup cannot remove an edge chain attached to the main wall/floor.
- Chunk welding cannot repair an ordinary shared edge.
- Simplification is disabled in the frozen regular-10-cm baseline and did not
  create these edges.

## What is and is not established upstream

The recorded matrix chain, 64.85 mm stereo baseline, frame ordering, and
timestamps are internally consistent. Mixed switches do not concentrate in
fast translation, excluding gross translational pose mismatch as the general
cause. They are moderately enriched during head rotation, so rotational
depth/pose latency remains a contributor. Same-pair stereo opposition is not
dominant, and many switches recur from nearly the same view, which requires raw
depth-layer instability in addition to any pose latency.

The remaining private boundary is the exact Meta uncertainty plane and any
runtime-only filtering before Room Setup fusion. Static analysis reached that
private packet boundary; it cannot assign an exact percentage to uncertainty
without inventing evidence.

## Repair implication

The causal order should be preserved in implementation and A/B testing:

1. stabilize near-zero shared derived-plane decisions locally;
2. add bounded, evidence-driven challenger/retirement for repeatedly opposed
   direct observations, without global decay;
3. retain the existing large-separation same-layer bridge guard;
4. publish corrected local components only when complete, preserving the old
   correct Front elsewhere;
5. compare the same frozen tape and the manual/high-confidence edge atlas
   before and after.

Uncertainty remains an optional future input improvement, not a prerequisite
for correcting the demonstrated fusion/extraction amplification inside the
replica.

## Independent class B confirmation

The user subsequently marked another visual category with 91 representative
edges. It is disjoint from both the class-A 48 edges and the original 650
painted triangles. Expanding the new edges produces 159 owner triangles:

- 151/159 have repeated mixed-sign source voxels;
- 150/159 have same-view repeated mixing;
- 146/159 have repeated temporal sign switching;
- 73/91 selected edges are exact derived-cell phase boundaries;
- the remaining 18 are same-cell chain segments.

Class B therefore shares the same upstream source-instability family, but it
must remain a separate geometric subtype. Its median/P95 owner-face dihedral is
`13.16/45.22 degrees`, versus `3.24/22.03 degrees` for class A, and its MC-pair
family is dominated by `136/221`, `204/221`, `136/153`, and `136/204` instead
of class A's predominantly `9/159` phase pair. The independently expanded
high-confidence sets do not overlap.

Using class B alone as the positive atlas finds 157 new high-confidence phase
edges, 228 connected continuations, and 151 probable review-only edges. The
working taxonomy is consequently:

- **Class A:** low-dihedral mixed-layer phase chain;
- **Class B:** high-dihedral mixed-layer phase chain.

They are two manifestations of the same source/fusion instability, not one
identical geometric rule. A repair may share source-history stabilization but
must retain separate geometry acceptance metrics for the two classes.

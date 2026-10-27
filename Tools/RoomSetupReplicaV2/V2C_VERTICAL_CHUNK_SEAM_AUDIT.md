# V2-C vertical chunk-seam audit

## Scope

- Frozen replay mesh: `0A022B000599E5748486A91AFEA474DE8F7B1084F552321CDA556E19A50E3B9E`
- Total mesh triangles: 26,656
- This audit targets the long, straight, visually unattractive vertical lines
  inside otherwise planar walls. It is separate from the 104-face derived-grid
  zero-plane phase class.

## Detection rule

A chain is retained only when all of the following hold:

1. the two sides do not share original vertex indices or an original edge;
2. after quantizing positions to 0.1 mm, the two sides become the same edge;
3. the adjacent face normals agree by at least 0.97;
4. each edge is predominantly vertical and the connected chain spans at least
   0.6 m;
5. the wall-normal coherence is at least 0.85;
6. the chain's horizontal tangent lies within 3 cm of a world-aligned 0.8 m
   Chunk plane and varies by no more than 18 cm.

This rejects ordinary shared triangulation edges and most real wall corners.

## Result

- 3,441 mesh vertices duplicate an existing coordinate at 0.1 mm precision.
- 18 vertical Chunk-seam chains pass the complete rule.
- They contain 262 unwelded boundary edges adjacent to 524 triangles.
- Three chains span at least 2.4 m and visually match the full-height line class
  in the marked screenshot.
- Every accepted chain lies exactly on a tangential 0.8 m Chunk coordinate in
  the frozen mesh, rather than merely falling inside the 3 cm tolerance.

The three longest chains are:

| Chain | Vertical span | Wall normal | Chunk boundary | Edges / adjacent faces |
| --- | ---: | --- | --- | ---: |
| 0 | 2.71 m | Z | X = 1.60 m | 29 / 58 |
| 1 | 2.70 m | X | Z = -0.80 m | 29 / 58 |
| 2 | 2.61 m | X | Z = -1.60 m | 29 / 58 |

## Cause and repair boundary

This is a Chunk publication/topology seam. The geometry on both sides is
spatially coincident and has coherent wall normals, but the original mesh keeps
separate vertex and edge identities. The wire topology therefore exposes a
long internal line even though it is not a room corner.

The production repair belongs at the Chunk boundary, not in TSDF smoothing:

1. derive both adjacent chunks from the same world-keyed boundary samples and
   deterministic sign/edge decisions;
2. share or weld boundary vertex identities when producing a unified mesh, or
   generate an explicit stitch strip with one owner for the internal edge;
3. publish the two sides atomically when a boundary topology changes so a
   transient single-sided seam is not exposed;
4. preserve real corners by applying the weld only to coordinate-coincident
   boundaries whose normals and source layer agree.

Do not lower the global mixed-layer threshold, smooth the wall, or delete the
524 adjacent faces. Those operations target the wrong layer and risk moving
valid geometry.

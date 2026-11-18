# Pre-v0.1.3 rollback archive

These text files preserve the production files that were present immediately
before the 2026-11-16 rollback to the complete v0.1.3 weighted fusion path.

They are deliberately stored as `.txt`, so Unity does not compile or load the
LatestSample/RobustCandidate experiments into the live room-scanning path.
They remain available for later comparison or selective recovery.

The simplified-room system is **not** archived or disabled. Its runtime
extractor, scene components, materials and exporter remain compiled in their
original locations. They now consume explicit read-only snapshots of the
restored v0.1.3 final mesh.

The one intentional live-rendering change on top of v0.1.3 is the lightweight
indirect `MeshTopology.Lines` wire pass. The depth occluder remains triangle
based; the visible cyan grid no longer shades triangle interiors.

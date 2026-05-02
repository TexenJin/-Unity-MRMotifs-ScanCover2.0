# Archived Depth Grid Features

This folder is a non-compiled archive for old `ScanCoverDepthGridPointCloud`
experiments and extension ideas.

Current baseline keeps the runtime path focused on:

- regular visual-direction depth grid sampling
- world-space snapshot display
- roll compensation
- grid line / outer contour display
- optional grid interior mesh display
- center/debug markers needed for calibration

Archived or hidden from the working Inspector:

- adaptive tile sampling
- view-locked volume sampling
- fixed-world-size / depth-hit-only experiments
- vertical depth plane experiment
- candidate surface plane objects
- largest-candidate remesh / triangular lattice experiments
- full surface region colorization controls
- old marker roster/debug dumping

`ScanCoverDepthGridPointCloud_full_legacy_snapshot.cs.txt` is a full snapshot
of the pre-cleanup script. It is intentionally saved as `.cs.txt`, so Unity
will not compile it. Restore pieces from this file only when one of the
archived features is needed again.

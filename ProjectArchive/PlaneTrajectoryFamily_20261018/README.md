# Plane trajectory / family route archive (2026-10-18)

Status: rejected after device inspection and removed from the active Unity
runtime. Files in this directory are reference snapshots only; this directory
is outside `Assets`, so Unity does not import or execute them.

The archived route included:

- persistent plane evidence (`RuntimeFinalSurfaceCourt`),
- pre-publication vertex projection (`SurfaceProductizer`),
- plane-family relabelling and display (`PlaneFamilyDisplayLayer`), and
- the integration points captured in the copied runtime, renderer and shader
  files.

The active project now bypasses this entire route. Mesh straightening is a
separate, bounded Surface Nets post-process and does not depend on this archive.

Active-tree decoupling details:

- the production block and persistent-chunk pipelines copy the raw extracted
  candidate directly and contain no productizer or family relabel queues;
- renderer interfaces, immutable fronts, shaders, HUD mode 4 and family GPU
  buffers were removed;
- `RuntimeFinalSurfaceCourt` was restored to its pre-family implementation so
  the older replay/final-court diagnostic remains independent; and
- the two empty `.cs` retirement markers under `Assets` contain no types or
  executable code. They only keep Unity's already-generated compiler response
  entries harmless; all implementation lives in this archive.

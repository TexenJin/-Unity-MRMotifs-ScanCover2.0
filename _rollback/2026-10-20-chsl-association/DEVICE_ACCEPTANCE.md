# Quest acceptance after restart

This change has passed Unity import and script/compute-shader compilation, but
it has not been called device-validated.

## HUD prerequisites

- `PAIR ... STREAM ON`
- `CHSL ... src=RAW`
- `CHSL guard ... band=15cm`
- no `CHSL fault=...`

## Same-session acceptance scenes

1. Scan one wall from near and then from 4-6 m. The far pass may be less dense,
   but it must not smear the wall into the floor or ceiling merely because of a
   global prefilter.
2. Scan a sharp wall/cabinet corner from both sides. A turn may initially have
   a gap, but a filled diagonal bridge between foreground and background is a
   failure.
3. Deliberately create a transient bridge, then observe the space through it
   from a clean angle for several accepted CHSL pairs. `free=correction/conflict`
   should advance and the obsolete island/bridge must eventually disappear.
4. Revisit a stable planar wall. `assoc` may rise at discontinuities, but the
   stable surface must not erode, flicker, or duplicate into parallel sheets.
5. Watch `int=total/peak`. The per-frame peak should remain close to the 4 ms
   slice budget; sustained visual stalls or a permanently growing mesh queue
   fail the performance gate.

## Plane Priors gate

Plane Priors remains off until all five observations pass in one uninterrupted
whole-room scan. Raw CHSL Marching Cubes is intentionally retained as evidence.
Triangle reduction for production stays in the existing 5 cm fine-fusion then
constrained-simplification path; it is not imitated inside the proof renderer.

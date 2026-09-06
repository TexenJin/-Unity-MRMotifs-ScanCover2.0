# Freeze/support relay audit — 2026-09-04

Scope: diagnostic-only. No TSDF weights, admission, freeze policy, meshing,
ownership, or extraction thresholds changed. Old probe remains disabled.
Review counter capacity is now 74; fusion counters remain 198.

## HUD (same audit batch denominators)

| Label | Numerator | Denominator | Interpretation |
|---|---|---|---|
| 不读邻冻 | unreadable sample with any frozen interpolation support (66) | all unreadable | spatial association only |
| 弱心冻 | weak nearest voxel is itself frozen (67) | weak-centre samples (41) | directly under-supported frozen centre |
| 邻缺冻 | at least one deficient interpolation support is frozen (68) | neighbour-deficit samples (42) | directly under-supported frozen neighbour |
| 空旁冻 | unwritten centre has a frozen support nearby (69) | unwritten-centre samples (40) | contextual, zero weight itself is not frozen |
| 对齐邻冻 | aligned readable sample with frozen support (70) | aligned samples (28) | healthy control |
| 错齐邻冻 | misaligned readable sample with frozen support (71) | readable minus aligned | misalignment control |
| 卡支冻 | stalled unreadable sample has deficient frozen support (72) | stalled samples (56) | first-seen unreadable cell still unclosed after eight audits |
| 缺支冻 | unreadable sample has deficient frozen support (73) | all unreadable | direct deficit overlap |

Frozen means signed TSDF weight < 0, matching production's frozen-voxel
branch. Deficient means abs(weight) < max(relayMinTsdfWeight, 0.001), matching
the existing read test. Supports are the same eight voxel centres used for
trilinear interpolation, not an enlarged search neighbourhood. No volume
bounds/support availability means no freeze evidence. Percentages can overlap;
do not sum them. Zero denominator displays `--`, not a misleading zero percent.
Numerator exceeding denominator sets ledger fault bit 128.

The counters do not establish that freezing CAUSED a hole, nor identify when a
cell was frozen or why. Normal frozen surfaces may be numerous. Prioritize
weak-centre/deficient-neighbour overlap over the global fusion `冻` count.

## Timestamps and saturation

`时戳 重` counts equal consecutive available platform timestamps; `倒` counts
strictly decreasing timestamps. Repeats no longer assert the backstep flag.
The diagnostic split does not deduplicate, discard, reorder, or replace frames.
`时基 间` retains the last positive platform interval (local-arrival fallback
when platform timestamp is unavailable); it is not a new-frame freshness clock.

Samples, tracked/completed cells, stalled samples, and timestamp counts retain
raw values and display `999+` above 999. Exact 999 remains `999`. Maturity mean
is labelled approximate: audit epochs / 2, not measured wall time. Completed
samples exclude never-completed cells; `卡` is sample count, not unique holes.

## Device acceptance

In 即时外壳, hold on the cabinet-top gap for roughly 10 seconds, then gently
translate and stop. Capture the full relay HUD. Check ledger `码00`, nonempty
freeze percentages, and whether high `卡` coincides with high `卡支冻`/`弱心冻`/
`邻缺冻`. Healthy-control freeze overlap alone must not trigger policy changes.
Compare `重` and `倒`; a large repeat count is not proof of backward time.

## Local verification

C# Android response-file compilation and all six shader entry points via
D3DCompile (default and O3) are independent checks, not full Unity Vulkan/APK
or Quest validation. Device layout, counters, and frame-time cost remain to be
validated on device. Existing unrelated compiler warnings are not repaired.

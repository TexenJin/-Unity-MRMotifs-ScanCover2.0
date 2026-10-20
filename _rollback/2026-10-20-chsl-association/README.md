# CHSL raw-depth / association rollback

This folder contains the four files exactly as they were before the
2026-10-20 CHSL observation-path change.

The new path is independently reversible in the Inspector:

- `StandaloneChiselSparseTsdfBridge.usePreSpatialFilterDepth`
- `ScanCoverChiselSparseTsdfShadow.strictSurfaceEvidenceBand`
- `ScanCoverChiselSparseTsdfShadow.enableObservationAssociation`
- `ScanCoverChiselSparseTsdfShadow.useViewWeightedObservations`
- `ScanCoverChiselSparseTsdfShadow.enableConfirmedFreeSpace`

Defaults select full-resolution exact-frame depth after hand masking, a true
15 cm surface evidence band, view/range-weighted observations, mature-voxel
association, bounded weights, and confirmed free-space correction. The
production TSDF and production mesh path are not modified by this change.

For a complete source rollback, copy the four saved files back to their
matching project paths. Re-run Unity import/compilation after restoring them.

Validation record: Unity 6000.3.9f1 batch import and C# compilation completed
successfully. APK packaging and Quest device acceptance were intentionally not
performed here.

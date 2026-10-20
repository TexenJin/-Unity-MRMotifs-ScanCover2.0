# CHSL three-class TSDF fix rollback

Created: 2026-10-20

Scope: `Assets/MRMotifs/ScanCover/Scripts/MetaXR/08_Preprocess/ScanCoverChiselSparseTsdfShadow.cs`

The preserved pre-change source is:

`ScanCoverChiselSparseTsdfShadow.cs.before`

SHA-256 before:

`2F416148E365FFA8761A057EBF3441A98C7C90C64A668239E12D4785F6AFC136`

SHA-256 after:

`9C563EE02B1C165AEB5942E287211EEAB33D7DEE68AFFC45E7EE9D64B10EE203`

The change preserves the effective 5 cm voxel, 15 cm truncation, 6 m depth range,
exact paired input, and existing depth-discontinuity protection. It adds:

- paired multi-view quarantine for novel surfaces;
- mature-surface direction ownership at wall corners;
- mature-occluder rejection along the observation ray;
- stereo-confirmed, consecutive free-space carving with bounded weights;
- mesh publication only after the minimum evidence weight.

Rollback in PowerShell (run while Unity is closed):

```powershell
Copy-Item -LiteralPath 'E:\-Unity-MRMotifs-ScanCover2.0-3.1-2\_rollback\2026-10-20-three-class-tsdf-fix\ScanCoverChiselSparseTsdfShadow.cs.before' -Destination 'E:\-Unity-MRMotifs-ScanCover2.0-3.1-2\Assets\MRMotifs\ScanCover\Scripts\MetaXR\08_Preprocess\ScanCoverChiselSparseTsdfShadow.cs' -Force
```

Validation completed before shutdown:

- Unity 6000.3.9f1 Android editor import/compile succeeded.
- `Assembly-CSharp.dll` was rebuilt and Mono reloaded successfully.
- No C# compile error was present in the final compile.
- APK packaging and Quest device A/B were intentionally not claimed here.

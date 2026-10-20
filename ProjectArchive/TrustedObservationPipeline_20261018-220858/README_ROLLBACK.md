# Trusted Observation Pipeline rollback point

Created before the 2026-10-18 trusted-observation implementation.

## Fast runtime rollback

In `VolumeIntegrator`, turn off `enableTrustedObservationPipeline`, or call:

```csharp
volumeIntegrator.SetTrustedObservationPipelineEnabled(false);
```

The next integration frame bypasses the added temporal-health gate, model
association, outlier rejection and structural protection. Pending association
confirmations are cleared by a dedicated GPU clear kernel. The existing
same-callback depth/matrix latch, 10 cm voxels, 25 cm truncation, spatial
false-bridge protection and 16/2/3 history remain unchanged.

Default production policy:

- latched packet age limit: 200 ms;
- old-model support: residual at most 5 cm;
- 5–12 cm hysteresis band: pass to the existing 16/2/3 history;
- conflict beyond 12 cm: require two same-direction accepted fusion frames;
- high-maturity, locally coherent horizontal/vertical structure: require three;
- unknown model space: remains birth-capable;
- prediction failure: counted and fail-open to the pre-change baseline.

Each layer also has its own serialized switch. Changing a rejection/structure
policy clears pending confirmation state before the new policy is used.

## Exact source rollback

Run the script below from PowerShell. It is dry-run by default:

```powershell
.\Rollback-TrustedObservationPipeline.ps1
.\Rollback-TrustedObservationPipeline.ps1 -Apply
```

`-Apply` first copies the then-current versions into a timestamped
`BeforeRollback_*` folder beside this file, then restores the exact pre-change
files. It does not delete project data.

The archive includes `DepthCapture.cs` for provenance even though this change
does not modify it: temporal pairing was already implemented there by latching
depth and matrices inside the same provider callback.

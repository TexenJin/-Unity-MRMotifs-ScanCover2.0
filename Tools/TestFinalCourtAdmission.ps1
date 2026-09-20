param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

function Read-Source([string]$relativePath) {
    Get-Content -LiteralPath (Join-Path $RepoRoot $relativePath) -Raw
}

function Assert-Contains([string]$text, [string]$needle, [string]$message) {
    if (-not $text.Contains($needle)) { throw $message }
}

function Assert-NotContains([string]$text, [string]$needle, [string]$message) {
    if ($text.Contains($needle)) { throw $message }
}

$integrator = Read-Source 'Assets/QuestRoomScanStandalone/Core/VolumeIntegrator.cs'
$shader = Read-Source 'Assets/QuestRoomScanStandalone/Shaders/VolumeIntegration.compute'
$session = Read-Source 'Assets/QuestRoomScanStandalone/Core/ScanReplaySessionPackage.cs'
$court = Read-Source 'Assets/QuestRoomScanStandalone/Core/RuntimeFinalSurfaceCourt.cs'
$scanner = Read-Source 'Assets/QuestRoomScanStandalone/Core/StandaloneRoomScanner.cs'

# Architecture: the court publishes an address verdict plus a plane constraint.
# It must not own a volume, extractor, renderer, or paper mesh.
Assert-Contains $court 'CopyAdmissionTables' 'Final court does not publish its stable-id admission tables.'
Assert-Contains $court 'never owns a mesh, volume or TSDF write' 'Final court authority boundary is missing.'
Assert-Contains $court 'Vector3.Dot(source,' 'Final court does not project testimony onto a shared coordinate axis.'
Assert-Contains $court 'state.ReferenceNormal' 'Final court compares coordinates from changing per-frame normals.'
Assert-Contains $court 'IndependentViewWitnessPolicy.FrameGap' 'Stationary temporal testimony cannot advance the final court.'
Assert-Contains $court 'Mode(audit, out float auditCenter, out int auditMembers)' 'Audit witnesses still do not participate in the verdict.'
Assert-Contains $court 'Mathf.Abs(hypotheses[winnerIndex].Center - auditCenter)' 'Audit cannot veto a mismatched proposal/selection winner.'
Assert-Contains $court 'state.Views.RemoveRange(0, Mathf.Min(3, state.Views.Count))' 'Court testimony still stops permanently at capacity or rotates witness roles.'
Assert-Contains $court 'PublishedLayerCoordinate' 'Layer replacement still lacks a scalar normal-depth coordinate.'
Assert-Contains $court 'WinnerNormal(state, winner.Center)' 'Published plane normal is still a one-frame normal.'
Assert-Contains $court 'DrainInvalidationRegions' 'Replaced court generations cannot retire their old TSDF neighbourhood.'
Assert-Contains $court 'RevocationConfirmations' 'Court revocation can still thrash on one observation.'
Assert-Contains $court 'ReplacementCooldownFrames' 'Court replacement has no anti-thrashing cooldown.'
foreach ($forbidden in @('RenderTexture', 'ComputeShader', 'SupportTruthRenderer',
        'SetVolumeOverride', 'SurfaceNets', 'MeshFilter', 'FinalCourtPaper')) {
    Assert-NotContains $court $forbidden "Final court regained forbidden geometry authority: $forbidden"
}

# Final post-transaction identity is read back by the existing replay capture
# and becomes the next-frame GPU admission table.
Assert-Contains $session 'gungel_correspondence_identity' 'Final correspondence identity is not captured.'
Assert-Contains $session 'gunGelCorrespondenceIdentityRows' 'Typed final identity rows are not retained.'
Assert-Contains $session '_finalSurfaceCourt?.Record' 'Final-buffer evidence does not reach the court.'
Assert-Contains $session '_finalCourtVerdictBuffer.SetData' 'Court verdict transitions do not reach the GPU gate.'
Assert-Contains $session '_finalCourtPlaneBuffer.SetData' 'Winning-layer membership descriptors do not reach the GPU gate.'
Assert-Contains $session '0, 0, uploadCount' 'Court still uploads the full 262144-entry table on every transition.'
Assert-Contains $session 'UploadFinalCourtAdmissionOnProductionThread();' 'Court GPU tables are not refreshed by the production thread before binding.'
Assert-Contains $session 'GPU admission is production state' 'Court upload ownership is not documented.'
Assert-Contains $session 'DrainFinalCourtInvalidationRegions' 'Court invalidation transaction is not exposed to production.'

# The only production Integrate dispatch binds identity + verdicts; absence is
# fail-closed and the projective A/B shadow explicitly disables the court.
Assert-Contains $integrator 'GunGelCorrespondenceIdentityID' 'Production integration does not bind stable identities.'
Assert-Contains $integrator 'FinalCourtVerdictsID' 'Production integration does not bind court verdicts.'
Assert-Contains $integrator 'FinalCourtPlanesID' 'Production integration does not bind winning-layer membership.'
Assert-Contains $integrator 'InvalidateFinalCourtReplacements(courtSession)' 'Court replacement does not retire old TSDF locally.'
Assert-Contains $integrator 'final_court_unavailable' 'Court runtime failure is not fail-closed.'
Assert-Contains $integrator 'final_court_pending:' 'Unresolved GunGel frames can bypass the court.'
Assert-Contains $integrator 'compute.SetFloat(FinalCourtAdmissionEnableID, 0f);' 'Projective shadow does not disable court admission.'
Assert-Contains $integrator 'return "裁冻:唯一生产路线";' 'The old three-office toggle can still regain production authority.'
Assert-NotContains $integrator 'enableFinalCourtAdmissionExperiment = false' 'Legacy mode code can still disable the final court.'
$invalidateIndex = $integrator.IndexOf('InvalidateFinalCourtReplacements(courtSession)')
$integrateIndex = $integrator.IndexOf('_integrateKernel.DispatchFit(_frustumVolume.count, 1);')
if ($invalidateIndex -lt 0 -or $integrateIndex -lt 0 -or $invalidateIndex -gt $integrateIndex) {
    throw 'Court replacement is not retired before the new plane writes.'
}

# The shader uses raw full-resolution depth for membership, then the approved
# plane—not that noisy ray—defines the production TSDF zero-level constraint.
Assert-Contains $shader 'gsGunGelCorrespondenceIdentity[index].y' 'Shader does not address verdicts by stableId.'
Assert-Contains $shader 'if (gunGelStatus == 0u)' 'Court permit is not the positive production branch.'
Assert-Contains $shader 'else if (gsFinalCourtAdmissionEnable > 0.5 ||' 'Court mode is not a strict status-zero whitelist.'
Assert-NotContains $shader 'GunGelStatusBlocksWrite(gunGelStatus)' 'GPU reinterprets an already classified court status through a second helper.'
Assert-Contains $shader 'gsFinalCourtPlanes[stableId]' 'Approved addresses do not select the winning raw-depth batch.'
Assert-Contains $shader 'dot(depthPosition, winnerNormal) - winnerPlane.w' 'Winning-layer membership does not test the original ray position.'
Assert-Contains $shader 'float rawProjectiveSdf = depthEyeDist - voxEyeDist;' 'Raw full-resolution projective distance disappeared.'
Assert-Contains $shader 'float sDist = grazingPlaneRescued' 'Production signed-distance derivation disappeared.'
Assert-Contains $shader 'uint GunGelAdmissionResult' 'Admission does not return one packed status/address permit.'
Assert-Contains $shader 'uint approvedStableId = gunGelAdmissionResult & 0xfffffu;' 'Approved stableId is not decoded beside its status.'
Assert-Contains $shader 'float4 approvedCourtPlane = gsFinalCourtPlanes[approvedStableId];' 'Production does not read the approved plane directly by its returned address.'
Assert-NotContains $shader 'out float4 approvedCourtPlane' 'Approved plane still crosses nested Vulkan calls as a separate out parameter.'
Assert-Contains $shader 'sDist = dot(vWorldPos, courtNormal) - courtOffset;' 'Approved plane does not constrain production signed distance.'
Assert-NotContains $shader 'gsFinalCourtVerdicts[stableId].xyz' 'Court verdicts are being treated as geometry.'

# This is an acquisition mode, not a new X display page. Existing merged-paper
# display remains named and the mode identity is visible as 裁冻.
Assert-Contains $scanner '"裁冻"' 'Court acquisition mode is not visible in the HUD identity.'
Assert-Contains $scanner '融合路线已锁定：裁冻' 'Final-court-only control hint is missing.'
Assert-Contains $scanner '合流仅纸' 'Existing merged-paper display gear was removed.'
Assert-NotContains $scanner '裁决仅纸' 'Court was incorrectly reintroduced as a display gear.'

Write-Output 'PASS: final court is the only fail-closed route and its independently audited plane constrains the single production TSDF.'

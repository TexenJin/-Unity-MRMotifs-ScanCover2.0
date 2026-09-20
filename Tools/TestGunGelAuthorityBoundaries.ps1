$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$core = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core'
$shaders = Join-Path $repo 'Assets/QuestRoomScanStandalone/Shaders'
$integrator = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'VolumeIntegrator.cs')
$scanner = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'StandaloneRoomScanner.cs')
$gunGel = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'GunGelEvidenceShadow.cs')
$fusion = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $shaders 'VolumeIntegration.compute')
$gunGelCompute = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $repo 'Assets/QuestRoomScanStandalone/Resources/GunGelEvidenceShadow.compute')

function Require([bool]$condition, [string]$message) {
    if (!$condition) { throw $message }
}

# Frame-level authority: GunGel uncertainty removes correction authority, not
# the platform depth frame. Only direct motion conflict remains a whole-frame veto.
Require ($integrator.Contains('public bool Integrate()')) `
    'Integrate does not report whether a production dispatch occurred.'
foreach ($reason in @('欠秩', '少配', '回读', '缺证', '撞顶', '位大', '转大')) {
    Require ($integrator.Contains('acceptedInputReason = "gungel_raw_fallback:" + rejectReason;')) `
        "Missing raw fallback path for correction abstentions ($reason)."
}
Require ($integrator.Contains('rejectReason == "快角"') -and
         $integrator.Contains('rejectReason == "快移"')) `
    'Explicit frame-motion conflicts are not isolated as the hard veto.'
Require ($integrator.Contains('bool gunGelAdmissionActive = gunGelCorrectionApplied &&')) `
    'Per-point GunGel evidence can still gate an uncorrected fallback frame.'

# Point-level authority: statuses 1/2/4/5 are lack-of-proof and statuses
# 3/6/7 are explicit contradictory evidence.
$block = [regex]::Match($fusion,
    '(?s)else if \(gsFinalCourtAdmissionEnable > 0\.5 \|\|(.*?)\)\s*\{').Groups[1].Value
Require (![string]::IsNullOrWhiteSpace($block)) 'Inline GunGel rejection branch is missing.'
foreach ($status in @(3, 6, 7)) {
    Require ($block.Contains("gunGelStatus == ${status}u")) "Explicit conflict status $status no longer blocks."
}
foreach ($status in @(1, 2, 4, 5)) {
    Require (!$block.Contains("gunGelStatus == ${status}u")) "Abstention status $status still blocks production."
}
Require ($fusion.Contains('if (gunGelStatus == 0u)')) `
    'Integration does not route approved GunGel status directly into production.'

# Candidate geometry is allowed to classify the physical-surface identity of
# the current full-resolution point, but must never become the TSDF sample.
Require ($fusion.Contains('float candidatePlaneResidual = abs(dot(depthPosition - candidatePosition,') -and
         $fusion.Contains('if (candidatePlaneResidual > branchLimit)')) `
    'Full-resolution points are not associated directly with the stable candidate plane.'
Require ($fusion.Contains('abs(dot(depthNormalUnit, candidateNormalUnit)) < 0.5')) `
    'Full-resolution candidate identity lost its normal-consistency gate.'
Require ($fusion.Contains('float sDist = grazingPlaneRescued') -and
         $fusion.Contains('rawProjectiveSdf * saturate(normDot)')) `
    'Production SDF is no longer derived from the current full-resolution depth sample.'
foreach ($forbidden in @(
    'sDist = dot(voxPos - candidatePosition',
    'sDist = dot(voxPos - adjudicatedSurfacePosition',
    'adjudicatedSurfacePosition',
    'adjudicatedSurfaceNormal')) {
    Require (!$fusion.Contains($forbidden)) `
        "Stable candidate geometry regained TSDF-shaping authority: $forbidden"
}
Require (([regex]::Matches($fusion, 'RWTexture3D<float2> gsVolumeRW;')).Count -eq 1) `
    'Full-resolution association unexpectedly added another production TSDF slot.'

# Transaction boundary: the current batch is judged against the candidate
# ledger that existed before the batch. Candidate updates may affect only the
# next batch; there must be no post-update correspondence rebuild.
$captureCall = '_captureCourtWaves.DispatchFit(slot.ObservationCount, 1, 1);'
$updateCall = '_updateCandidates.DispatchFit(slot.ObservationCount, 1, 1);'
$captureIndex = $gunGel.IndexOf($captureCall)
$updateIndex = $gunGel.IndexOf($updateCall)
Require ($captureIndex -ge 0 -and $updateIndex -gt $captureIndex) `
    'Current-batch production correspondence is not frozen before candidate update.'
Require (!$gunGel.Contains('_rebuildProductionCorrespondences')) `
    'Current batch can still rebuild its verdict after moving or replacing candidates.'
Require (!$gunGelCompute.Contains('RebuildProductionCorrespondences')) `
    'Post-update production rebuild kernel still exists and can restore self-approval.'

# A successful atomic candidate succession must invalidate only a bounded local
# region of the production TSDF. The candidate identifies what to discard, but
# cannot directly provide a replacement SDF. The frozen current batch is written
# first and then erased locally, so only a later batch judged by the new ledger
# can reseed the unknown region.
Require ($gunGelCompute.Contains('_SuccessionInvalidationCenterRadius[invalidationIndex]') -and
         $gunGelCompute.Contains('_CellSize * 0.75, _CellSize * 1.25')) `
    'Atomic succession does not emit a bounded local TSDF invalidation ticket.'
Require ($fusion.Contains('#pragma kernel InvalidateGunGelSuccessions') -and
         $fusion.Contains('[numthreads(8, 8, 8)]') -and
         $fusion.Contains('WriteTrackedTsdf(coord, float2(GS_EMPTY_VOXEL, 0));')) `
    'Production TSDF does not restore the bounded succession region to unknown.'
foreach ($forbidden in @(
    'WriteTrackedTsdf(coord, float2(candidate',
    'gsVolumeRW[coord] = float2(candidate',
    'gsVolumeRW[coord] = float2(region')) {
    Require (!$fusion.Contains($forbidden)) `
        "Candidate succession regained direct TSDF-shaping authority: $forbidden"
}
Require ($integrator.Contains('compute.DispatchIndirect(') -and
         $integrator.Contains('GunGelSuccessionInvalidationArgsID, invalidations.Args')) `
    'Bounded succession tickets are not consumed through their indirect dispatch count.'
$productionDispatch = '_integrateKernel.DispatchFit(_frustumVolume.count, 1);'
$invalidateCall = 'InvalidateGunGelSucceededRegions();'
$productionIndex = $integrator.IndexOf($productionDispatch)
$invalidateIndex = $integrator.IndexOf($invalidateCall, $productionIndex)
$shadowMarker = '// B: read-only KinectFusion-style projective TSDF shadow.'
$shadowIndex = $integrator.IndexOf($shadowMarker, $productionIndex)
Require ($productionIndex -ge 0 -and $invalidateIndex -gt $productionIndex -and
         $shadowIndex -gt $invalidateIndex) `
    'Succession invalidation must run after current production integration and before shadow integration.'

# Lifecycle and accounting: warmup keeps the witness ledger, and scanner-side
# throughput counts only actual GPU production dispatches.
Require ($integrator.Contains('ClearInternal(preserveGunGelEvidence: true);')) `
    'Warmup still clears the newly-grown GunGel candidate ledger.'
Require ($integrator.Contains('if (!preserveGunGelEvidence)') -and
         $integrator.Contains('_gunGelEvidenceShadow?.Clear();')) `
    'User clear and warmup clear do not have separate evidence lifetimes.'
Require (([regex]::Matches($scanner,
    'bool dispatched = _volumeIntegrator\.Integrate\(\);')).Count -eq 2) `
    'Scanner integration call sites do not capture actual dispatch state.'
Require (([regex]::Matches($scanner,
    'if \(dispatched\)')).Count -ge 2) `
    'Scanner still counts attempted integrations as successful dispatches.'

'PASS: GunGel freezes each batch against the pre-transaction candidate ledger; atomic succession clears only a bounded production-TSDF region after the old batch and lets later winner-aligned full-resolution depth rebuild it; candidate geometry never shapes TSDF, explicit conflicts veto, uncertainty abstains, warmup preserves evidence, and scanner counts real dispatches.'

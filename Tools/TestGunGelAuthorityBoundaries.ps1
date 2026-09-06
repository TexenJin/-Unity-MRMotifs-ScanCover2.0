$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$core = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core'
$shaders = Join-Path $repo 'Assets/QuestRoomScanStandalone/Shaders'
$integrator = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'VolumeIntegrator.cs')
$scanner = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'StandaloneRoomScanner.cs')
$fusion = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $shaders 'VolumeIntegration.compute')

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
    '(?s)bool GunGelStatusBlocksWrite\(uint status\)\s*\{(.*?)\}').Groups[1].Value
Require (![string]::IsNullOrWhiteSpace($block)) 'GunGel write-block classifier is missing.'
foreach ($status in @(3, 6, 7)) {
    Require ($block.Contains("status == ${status}u")) "Explicit conflict status $status no longer blocks."
}
foreach ($status in @(1, 2, 4, 5)) {
    Require (!$block.Contains("status == ${status}u")) "Abstention status $status still blocks production."
}
Require ($fusion.Contains('if (GunGelStatusBlocksWrite(gunGelStatus))')) `
    'Integration does not use the bounded GunGel blocker.'

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

'PASS: GunGel uncertainty abstains, explicit conflicts veto, warmup preserves evidence, and scanner counts real dispatches.'

$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$core = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core'
$shaders = Join-Path $repo 'Assets/QuestRoomScanStandalone/Shaders'
$scene = Join-Path $repo 'Assets/QuestRoomScanStandalone/Scenes/QuestRoomScanStandalone.unity'

$integrator = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'VolumeIntegrator.cs')
$fusion = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $shaders 'VolumeIntegration.compute')
$mainScene = Get-Content -Raw -Encoding UTF8 -LiteralPath $scene

function Require([bool]$condition, [string]$message) {
    if (!$condition) { throw $message }
}

# The production scene must explicitly enable the mature-surface settling path;
# relying only on a field initializer would let an older serialized scene keep
# the feature disabled.
Require ($integrator.Contains('private bool enableMatureSurfaceObsDiscount = true;')) `
    'Mature-surface convergence is not enabled by default.'
Require ($mainScene.Contains('enableMatureSurfaceObsDiscount: 1')) `
    'The Quest production scene does not enable mature-surface convergence.'
Require ($mainScene.Contains('matureSurfaceObsWeightMin: 0.5')) `
    'The production scene no longer limits settling to fully mature surfaces.'

# Small residuals on mature surfaces must reduce the actual TSDF blend, while
# weak/empty cells and larger changes retain their normal production path.
Require ($fusion.Contains('bool matureSurfaceObs = weight >= gsMatureObsWeightMin;')) `
    'Mature-surface admission is missing from fusion.'
Require ($fusion.Contains('float q2obs = (obsDiscount ? q2 * gsMatureObsDiscount : q2) *')) `
    'The mature discount no longer controls the production observation weight.'
Require ($fusion.Contains('float blend = q2obs * gsBlendRate / (1.0 + weight * gsStability);')) `
    'The discounted observation weight no longer controls TSDF geometry movement.'
Require ($fusion.Contains('float newTsdf = contradicted ? oldVal : lerp(oldVal, sDistNorm, blend);')) `
    'Contradictions no longer preserve stored geometry while removing support.'
Require ($fusion.Contains('vol.y >= FreezeSupportWeightFloor()')) `
    'Block freeze can no longer leave weak support live for coverage repair.'

'PASS: mature surfaces settle, true changes retain correction authority, and weak coverage remains writable.'

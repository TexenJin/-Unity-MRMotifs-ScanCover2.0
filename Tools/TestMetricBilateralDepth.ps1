$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$capturePath = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core/DepthCapture.cs'
$shaderPath = Join-Path $repo 'Assets/QuestRoomScanStandalone/Shaders/BilateralDepthFilter.compute'

$capture = Get-Content -Raw -Encoding UTF8 -LiteralPath $capturePath
$shader = Get-Content -Raw -Encoding UTF8 -LiteralPath $shaderPath

function Require([bool]$condition, [string]$message) {
    if (!$condition) { throw $message }
}

# The production pass must compare and accumulate physical depth, then encode
# the result back to the source texture's NDC representation.
Require ($shader.Contains('float centerLinear = ToLinearDepth(centerDepth, eye);')) `
    'Bilateral center depth is not linearized.'
Require ($shader.Contains('float depthDiff = centerLinear - neighborLinear;')) `
    'Bilateral range weight is not measured in metres.'
Require ($shader.Contains('depthSum  += neighborLinear * w;')) `
    'Bilateral output is still averaging nonlinear NDC depth.'
Require ($shader.Contains('ToNdcDepth(filteredLinear, eye)')) `
    'Filtered metric depth is not converted back to NDC.'
Require ($shader.Contains('if (nDepth <= 0.0)')) `
    'Abstention depth can still vote as a near-plane sample.'
Require (!$shader.Contains('float depthDiff = centerDepth - nDepth;')) `
    'Legacy distance-dependent NDC comparison remains active.'
Require (($capture.Split('cs.SetVectorArray(EdgeLinearizeABID, _linearizeAB);').Count - 1) -ge 2) `
    'Projection coefficients are not supplied to the compute shader.'

# Verify the shader algebra against the finite perspective matrix used by
# DepthCapture. Both near and far samples must round-trip without distance bias.
$near = 0.2
$far = 20.0
$a = -($far + $near) / ($far - $near)
$b = -(2.0 * $far * $near) / ($far - $near)
foreach ($metres in @(0.25, 0.5, 1.0, 2.0, 5.0, 15.0)) {
    $clipZ = $b / $metres - $a
    $ndc = ($clipZ + 1.0) * 0.5
    $roundTrip = [Math]::Abs($b / (($ndc * 2.0 - 1.0) + $a))
    Require ([Math]::Abs($roundTrip - $metres) -lt 1e-9) `
        "Metric/NDC round-trip failed at $metres m."
}

'PASS: bilateral depth weighting and averaging are metric, abstentions do not vote, and projection conversion round-trips.'

# Run hybrid pipeline: person_dog_park.png + snow instruction
# Usage: .\run_hybrid_park_snow.ps1 [-RealApi] [-Iterations 2] [-Candidates 2]

param(
    [switch]$RealApi,
    [int]$Iterations = 1,
    [int]$Candidates = 2,
    [string]$Image = "data/input/edit/person_dog_park.png",
    [string]$Instruction = "如果公园里正在下大雪，地面被雪覆盖会怎样？",
    [string]$Output = "data/output/hybrid/park_snow_out"
)

$cmdArgs = @(
    "-m", "reason.hybrid_pipeline",
    "--image", $Image,
    "--instruction", $Instruction,
    "--output", $Output,
    "--iterations", $Iterations,
    "--candidates", $Candidates
)

if ($RealApi) {
    $cmdArgs += "--real-api"
}

Write-Host "=== ReasonGenPilot Hybrid Pipeline ===" -ForegroundColor Cyan
Write-Host "Image      : $Image"
Write-Host "Instruction: $Instruction"
Write-Host "Output     : $Output"
Write-Host "Iterations : $Iterations"
Write-Host "Candidates : $Candidates"
Write-Host "Real API   : $($RealApi.IsPresent)"
Write-Host ""

python @cmdArgs
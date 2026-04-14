param(
    [ValidateSet("debug", "10", "20", "all")]
    [string]$Mode = "all",
    [ValidateSet("local", "kaggle")]
    [string]$Env = "local",
    [int]$NumSamples = 12,
    [string]$Split = "test"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonCandidates = @(
    "d:/DIP/.venv/Scripts/python.exe",
    "$repoRoot/.venv/Scripts/python.exe",
    "python"
)

$pythonExe = $null
foreach ($cand in $pythonCandidates) {
    if ($cand -eq "python") {
        $pythonExe = "python"
        break
    }
    if (Test-Path $cand) {
        $pythonExe = $cand
        break
    }
}

if (-not $pythonExe) {
    throw "Could not find Python executable."
}

function Get-LatestCheckpoint {
    $dir = Join-Path $repoRoot "outputs/checkpoints/resnet"
    if (-not (Test-Path $dir)) {
        return $null
    }

    $latest = Get-ChildItem -Path $dir -Filter "*_best.pth" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    return $latest
}

function Run-DebugZeroEpoch {
    Write-Host "[1/3] Debug pipeline (0 epoch, no train)" -ForegroundColor Cyan
    & $pythonExe "scripts/test_learned_landmark_kaggle.py" `
        --env $Env `
        --config "resnet" `
        --split $Split `
        --num_samples $NumSamples `
        --save_per_keypoint `
        --output_dir "outputs/learned_landmark_test/debug_0ep"
}

function Run-TrainAndCompare {
    param(
        [Parameter(Mandatory = $true)][string]$TrainConfig,
        [Parameter(Mandatory = $true)][string]$Tag
    )

    Write-Host "Train and compare for $Tag" -ForegroundColor Cyan

    & $pythonExe "scripts/train.py" --config $TrainConfig --env $Env

    $ckpt = Get-LatestCheckpoint
    if ($null -eq $ckpt) {
        throw "No checkpoint found in outputs/checkpoints/resnet after training."
    }

    Write-Host "Using checkpoint: $($ckpt.FullName)" -ForegroundColor Yellow

    & $pythonExe "scripts/test_learned_landmark_kaggle.py" `
        --env $Env `
        --config "resnet" `
        --split $Split `
        --num_samples $NumSamples `
        --save_per_keypoint `
        --checkpoint $ckpt.FullName `
        --output_dir "outputs/learned_landmark_test/compare_$Tag"
}

switch ($Mode) {
    "debug" {
        Run-DebugZeroEpoch
    }
    "10" {
        Run-TrainAndCompare -TrainConfig "resnet_landmark_10ep" -Tag "10ep"
    }
    "20" {
        Run-TrainAndCompare -TrainConfig "resnet_landmark_20ep" -Tag "20ep"
    }
    "all" {
        Run-DebugZeroEpoch
        Run-TrainAndCompare -TrainConfig "resnet_landmark_10ep" -Tag "10ep"
        Run-TrainAndCompare -TrainConfig "resnet_landmark_20ep" -Tag "20ep"
    }
}

Write-Host "Done. Check outputs/learned_landmark_test for generated maps." -ForegroundColor Green

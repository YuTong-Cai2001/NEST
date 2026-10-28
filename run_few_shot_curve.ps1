param(
    [string]$PythonExecutable = "",
    [string]$Seeds = "42,43,44",
    [string]$Shots = "5,10,15,20",
    [string]$NormalMultipliers = "1,2,3,5",
    [int]$MaxSupportAbnormal = 20,
    [int]$MaxSupportNormal = 100,
    [int]$StartYear = 2018,
    [int]$EndYear = 2022,
    [string]$CheckpointPrefix = "nest_fullseq_loss_full_seed",
    [string]$CheckpointName = "best_source_loss.pth",
    [string]$ResultsCsv = "results/reproduced/few_shot_grid_runs.csv",
    [string]$SummaryCsv = "results/reproduced/few_shot_grid_mean_std.csv",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $condaPython = if ($env:CONDA_PREFIX) {
        Join-Path $env:CONDA_PREFIX "python.exe"
    } else {
        $null
    }
    if ($condaPython -and (Test-Path -LiteralPath $condaPython)) {
        $PythonExecutable = $condaPython
    } else {
        $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
    }
}

Write-Host "Using Python: $PythonExecutable"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    if ($DryRun) {
        Write-Host "DRY RUN: $PythonExecutable $($Arguments -join ' ')"
        return
    }
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

$seedValues = @($Seeds.Split(',') | ForEach-Object { [int]$_.Trim() })
foreach ($seed in $seedValues) {
    $checkpoint = "outputs/$CheckpointPrefix$seed/checkpoints/$CheckpointName"
    if ((-not $DryRun) -and (-not (Test-Path -LiteralPath $checkpoint))) {
        throw "Checkpoint not found: $checkpoint"
    }
    Invoke-CheckedPython -Arguments @(
        "few_shot_curve.py",
        "--model_path", $checkpoint,
        "--shots", $Shots,
        "--normal_multipliers", $NormalMultipliers,
        "--max_support_abnormal", "$MaxSupportAbnormal",
        "--max_support_normal", "$MaxSupportNormal",
        "--start_year", "$StartYear",
        "--end_year", "$EndYear",
        "--seed", "$seed",
        "--results_csv", $ResultsCsv
    )
}

Invoke-CheckedPython -Arguments @(
    "summarize_few_shot_curve.py", "--input", $ResultsCsv,
    "--output", $SummaryCsv
)

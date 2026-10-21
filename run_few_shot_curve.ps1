param(
    [string]$PythonExecutable = "",
    [string]$Seeds = "42,43,44",
    [string]$Shots = "5,10,20",
    [int]$NormalMultiplier = 5,
    [int]$MaxSupportAbnormal = 20,
    [int]$MaxSupportNormal = 100,
    [int]$MinAdaptationAbnormal = 10,
    [int]$StartYear = 2018,
    [int]$EndYear = 2022,
    [string]$CheckpointPrefix = "nest_paper_full_seed",
    [string]$ResultsCsv = "results/reproduced/few_shot_curve_runs.csv",
    [string]$SummaryCsv = "results/reproduced/few_shot_curve_mean_std.csv",
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
    $checkpoint = "outputs/$CheckpointPrefix$seed/checkpoints/best_source_loss.pth"
    if ((-not $DryRun) -and (-not (Test-Path -LiteralPath $checkpoint))) {
        throw "Checkpoint not found: $checkpoint"
    }
    Invoke-CheckedPython -Arguments @(
        "few_shot_curve.py",
        "--model_path", $checkpoint,
        "--shots", $Shots,
        "--normal_multiplier", "$NormalMultiplier",
        "--max_support_abnormal", "$MaxSupportAbnormal",
        "--max_support_normal", "$MaxSupportNormal",
        "--min_adaptation_abnormal", "$MinAdaptationAbnormal",
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

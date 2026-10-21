param(
    [string]$PythonExecutable = "",
    [string]$Seeds = "42,43,44",
    [int]$StartYear = 2018,
    [int]$EndYear = 2022,
    [int]$NumAbnormal = 10,
    [int]$NumNormal = 50,
    [string]$CheckpointPrefix = "spad_ablation_no_reg_seed",
    [string]$ResultsCsv = "results/few_shot_final_runs.csv",
    [string]$SummaryCsv = "results/few_shot_mean_std.csv",
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
        "few_shot_finetune.py",
        "--model_path", $checkpoint,
        "--start_year", "$StartYear",
        "--end_year", "$EndYear",
        "--num_abnormal", "$NumAbnormal",
        "--num_normal", "$NumNormal",
        "--seed", "$seed",
        "--adaptation_mode", "score_fusion",
        "--results_csv", $ResultsCsv
    )
}

Invoke-CheckedPython -Arguments @(
    "summarize_few_shot.py", "--input", $ResultsCsv,
    "--output", $SummaryCsv
)

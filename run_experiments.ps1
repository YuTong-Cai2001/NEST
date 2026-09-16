param(
    [string]$PythonExecutable = "",
    [string]$Seeds = "42",
    [int]$Epochs = 50,
    [int]$StartYear = 2018,
    [int]$EndYear = 2022,
    [string]$RunPrefix = "spad_dipt",
    [string]$Variants = "full,no_self,no_proto,no_reg,no_transport,dual",
    [switch]$SkipFewShot,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# When this script is launched from an activated Conda CMD prompt through a
# nested powershell.exe process, plain `python` can resolve back to the base
# interpreter.  Prefer the interpreter inside the inherited Conda prefix.
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
$variantValues = @($Variants.Split(',') | ForEach-Object { $_.Trim() })
$validVariants = @("full", "no_self", "no_proto", "no_reg", "no_transport", "dual")
$unknownVariants = @($variantValues | Where-Object { $_ -notin $validVariants })
if ($unknownVariants.Count -gt 0) {
    throw "Unknown variants: $($unknownVariants -join ', ')"
}

foreach ($seed in $seedValues) {
    $common = @(
        "train_anomaly_gfm.py",
        "--mode", "dynamic",
        "--epochs", "$Epochs",
        "--batch_size", "1",
        "--sequence_length", "5",
        "--start_year", "$StartYear",
        "--end_year", "$EndYear",
        "--seed", "$seed",
        "--method_version", "dipt"
    )

    if ($variantValues -contains "full") {
        Invoke-CheckedPython -Arguments ($common + @(
            "--run_name", "$($RunPrefix)_full_seed$seed"
        ))
    }
    if ($variantValues -contains "no_self") {
        Invoke-CheckedPython -Arguments ($common + @(
            "--run_name", "$($RunPrefix)_no_self_seed$seed", "--lambda_self", "0"
        ))
    }
    if ($variantValues -contains "no_proto") {
        Invoke-CheckedPython -Arguments ($common + @(
            "--run_name", "$($RunPrefix)_no_proto_seed$seed", "--lambda_proto", "0"
        ))
    }
    if ($variantValues -contains "no_reg") {
        Invoke-CheckedPython -Arguments ($common + @(
            "--run_name", "$($RunPrefix)_no_reg_seed$seed", "--lambda_reg", "0"
        ))
    }
    if ($variantValues -contains "no_transport") {
        Invoke-CheckedPython -Arguments ($common + @(
            "--run_name", "$($RunPrefix)_no_transport_seed$seed", "--disable_transport_residual"
        ))
    }
    if ($variantValues -contains "dual") {
        Invoke-CheckedPython -Arguments ($common + @(
            "--run_name", "$($RunPrefix)_dual_seed$seed", "--prototype_mode", "dual"
        ))
    }

    if ((-not $SkipFewShot) -and ($variantValues -contains "full")) {
        $checkpoint = "outputs/$($RunPrefix)_full_seed$seed/checkpoints/best_source_loss.pth"
        Invoke-CheckedPython -Arguments @(
            "few_shot_finetune.py",
            "--model_path", $checkpoint,
            "--start_year", "$StartYear",
            "--end_year", "$EndYear",
            "--seed", "$seed"
        )
    }
}

Invoke-CheckedPython -Arguments @("plot_results.py")

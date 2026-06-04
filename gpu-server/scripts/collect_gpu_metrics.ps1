[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [int]$DurationSeconds = 0,

    [int]$IntervalSeconds = 1
)

$ErrorActionPreference = "Stop"

if ($IntervalSeconds -lt 1) {
    throw "IntervalSeconds must be 1 or greater."
}

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory -and -not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$query = "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
$header = "timestamp,utilization_gpu,utilization_memory,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c"
Set-Content -LiteralPath $OutputPath -Value $header -Encoding ascii

$startedAt = Get-Date

while ($true) {
    $sample = & nvidia-smi --query-gpu=$query --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi query failed."
    }

    foreach ($line in $sample) {
        Add-Content -LiteralPath $OutputPath -Value $line -Encoding ascii
    }

    if ($DurationSeconds -gt 0) {
        $elapsed = ((Get-Date) - $startedAt).TotalSeconds
        if ($elapsed -ge $DurationSeconds) {
            break
        }
    }

    Start-Sleep -Seconds $IntervalSeconds
}

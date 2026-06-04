[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("llamacpp", "vllm")]
    [string]$Backend,

    [Parameter(Mandatory = $true)]
    [ValidateSet("process-utterance", "final-edit", "post-pipeline")]
    [string]$Scenario,

    [string]$BaseUrl = "http://127.0.0.1:8000",

    [string]$FastApiContainer = "project-fastapi",

    [string]$ModelContainer = "",

    [string]$StoreId = "00000000-0000-0000-0000-000000000001",

    [string]$SessionIdPrefix = "loadtest",

    [string]$DraftKeys = "/ai-drafts/test-session/draft-001.png,/ai-drafts/test-session/draft-002.png",

    [string]$OutputRoot = "artifacts/load-tests",

    [int[]]$VusStages = @(1, 2, 4, 6),

    [int]$WarmupIterations = 3,

    [int]$WarmupVus = 1,

    [int]$MeasureSeconds = 60,

    [int]$CoolDownSeconds = 120,

    [int]$ThinkTimeSeconds = 1,

    [int]$RepeatCount = 3,

    [string]$RequestTimeout = "120s"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$scriptMap = @{
    "process-utterance" = Join-Path $repoRoot "ai\scripts\k6_process_utterance.js"
    "final-edit" = Join-Path $repoRoot "ai\scripts\k6_final_edit.js"
    "post-pipeline" = Join-Path $repoRoot "ai\scripts\k6_post_pipeline.js"
}
$scriptPath = $scriptMap[$Scenario]

if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "k6 script not found: $scriptPath"
}

function New-StageDirectory {
    param(
        [string]$Root,
        [int]$RepeatIndex,
        [int]$Vus
    )

    $path = Join-Path $Root ("repeat-{0}\vu-{1}" -f $RepeatIndex, $Vus)
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    return $path
}

function Start-ContainerLogCapture {
    param(
        [string]$ContainerName,
        [string]$OutputPath
    )

    if ([string]::IsNullOrWhiteSpace($ContainerName)) {
        return $null
    }

    $logCommand = "docker logs -f $ContainerName *> `"$OutputPath`""
    return Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-Command", $logCommand) `
        -WindowStyle Hidden `
        -PassThru
}

function Start-GpuCollector {
    param(
        [string]$OutputPath,
        [int]$DurationSeconds
    )

    return Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-File",
            (Join-Path $repoRoot "gpu-server\scripts\collect_gpu_metrics.ps1"),
            "-OutputPath",
            $OutputPath,
            "-DurationSeconds",
            $DurationSeconds,
            "-IntervalSeconds",
            1
        ) `
        -WindowStyle Hidden `
        -PassThru
}

function Invoke-K6 {
    param(
        [string]$ScriptPath,
        [string]$SummaryPath,
        [int]$Vus,
        [string]$Duration = "",
        [int]$Iterations = 0
    )

    $envMap = [ordered]@{
        BASE_URL = $BaseUrl
        K6_VUS = $Vus
        THINK_TIME_SECONDS = $ThinkTimeSeconds
        STORE_ID = $StoreId
        SESSION_ID_PREFIX = $SessionIdPrefix
        DRAFT_KEYS = $DraftKeys
        REQUEST_TIMEOUT = $RequestTimeout
    }

    if ($Iterations -gt 0) {
        $envMap["K6_ITERATIONS"] = $Iterations
    } elseif (-not [string]::IsNullOrWhiteSpace($Duration)) {
        $envMap["K6_DURATION"] = $Duration
    }

    $command = @(
        $envMap.GetEnumerator() | ForEach-Object {
            $escaped = [string]$_.Value -replace "'", "''"
            "`$env:{0}='{1}'" -f $_.Key, $escaped
        }
    )
    $command += "k6 run --summary-export `"$SummaryPath`" `"$ScriptPath`""

    & powershell.exe -NoProfile -Command ($command -join "; ")
    if ($LASTEXITCODE -ne 0) {
        throw "k6 command failed for $ScriptPath"
    }
}

function Get-RestartCount {
    param([string]$ContainerName)

    if ([string]::IsNullOrWhiteSpace($ContainerName)) {
        return 0
    }

    $count = & docker inspect --format "{{.RestartCount}}" $ContainerName 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($count)) {
        return 0
    }

    return [int]$count
}

function Test-Health {
    param([string]$Url)

    $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 15 -UseBasicParsing
    if ($response.StatusCode -ne 200) {
        throw "Health check failed: $Url"
    }
}

function Test-StopCriteria {
    param(
        [string]$SummaryPath,
        [string]$GpuPath,
        [int]$RestartCountBefore,
        [int]$RestartCountAfter
    )

    $summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $timeoutRateValue = 0.0
    $http5xxRateValue = 0.0

    if ($summary.metrics.timeout_rate) {
        $timeoutRateValue = [double]$summary.metrics.timeout_rate.values.rate
    }

    if ($summary.metrics.http_5xx_rate) {
        $http5xxRateValue = [double]$summary.metrics.http_5xx_rate.values.rate
    }

    if ($timeoutRateValue -ge 0.10) {
        return "timeout_rate"
    }

    if ($http5xxRateValue -ge 0.05) {
        return "http_5xx_rate"
    }

    if (Test-Path -LiteralPath $GpuPath) {
        $samples = Import-Csv -LiteralPath $GpuPath
        $highVramStreak = 0

        foreach ($sample in $samples) {
            $used = [double]$sample.memory_used_mib
            $total = [double]$sample.memory_total_mib
            if ($total -le 0) {
                continue
            }

            if (($used / $total) -ge 0.95) {
                $highVramStreak += 1
                if ($highVramStreak -ge 10) {
                    return "vram_threshold"
                }
            } else {
                $highVramStreak = 0
            }
        }
    }

    if ($RestartCountAfter -gt $RestartCountBefore) {
        return "container_restart"
    }

    return $null
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$experimentRoot = Join-Path $repoRoot (Join-Path $OutputRoot "$Backend\$Scenario\$timestamp")
New-Item -ItemType Directory -Path $experimentRoot -Force | Out-Null

$metadata = [ordered]@{
    backend = $Backend
    scenario = $Scenario
    base_url = $BaseUrl
    fastapi_container = $FastApiContainer
    model_container = $ModelContainer
    store_id = $StoreId
    session_id_prefix = $SessionIdPrefix
    draft_keys = $DraftKeys
    vus_stages = $VusStages
    warmup_iterations = $WarmupIterations
    measure_seconds = $MeasureSeconds
    cooldown_seconds = $CoolDownSeconds
    repeat_count = $RepeatCount
    created_at = (Get-Date).ToString("o")
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $experimentRoot "metadata.json") -Encoding utf8

$stopReason = $null

for ($repeatIndex = 1; $repeatIndex -le $RepeatCount; $repeatIndex += 1) {
    foreach ($vus in $VusStages) {
        Test-Health -Url "$BaseUrl/ai/health"

        $warmupSummaryPath = Join-Path $experimentRoot ("repeat-{0}\vu-{1}-warmup-summary.json" -f $repeatIndex, $vus)
        New-Item -ItemType Directory -Path (Split-Path -Parent $warmupSummaryPath) -Force | Out-Null
        Invoke-K6 -ScriptPath $scriptPath -SummaryPath $warmupSummaryPath -Vus $WarmupVus -Iterations $WarmupIterations

        $stageDir = New-StageDirectory -Root $experimentRoot -RepeatIndex $repeatIndex -Vus $vus
        $k6SummaryPath = Join-Path $stageDir "k6-summary.json"
        $gpuMetricsPath = Join-Path $stageDir "gpu-metrics.csv"
        $fastapiLogPath = Join-Path $stageDir "fastapi.log"
        $modelLogPath = Join-Path $stageDir "model.log"

        $restartBefore = Get-RestartCount -ContainerName $ModelContainer
        $fastapiLogProc = Start-ContainerLogCapture -ContainerName $FastApiContainer -OutputPath $fastapiLogPath
        $modelLogProc = Start-ContainerLogCapture -ContainerName $ModelContainer -OutputPath $modelLogPath
        $gpuProc = Start-GpuCollector -OutputPath $gpuMetricsPath -DurationSeconds $MeasureSeconds

        try {
            Invoke-K6 -ScriptPath $scriptPath -SummaryPath $k6SummaryPath -Vus $vus -Duration ("{0}s" -f $MeasureSeconds)
        } finally {
            if ($gpuProc) {
                $gpuProc.WaitForExit()
            }
            foreach ($proc in @($fastapiLogProc, $modelLogProc)) {
                if ($proc -and -not $proc.HasExited) {
                    Stop-Process -Id $proc.Id -Force
                }
            }
        }

        $restartAfter = Get-RestartCount -ContainerName $ModelContainer
        $stopReason = Test-StopCriteria `
            -SummaryPath $k6SummaryPath `
            -GpuPath $gpuMetricsPath `
            -RestartCountBefore $restartBefore `
            -RestartCountAfter $restartAfter

        if ($stopReason) {
            break
        }

        Start-Sleep -Seconds $CoolDownSeconds
    }

    if ($stopReason) {
        break
    }
}

$summaryScript = Join-Path $repoRoot "gpu-server\scripts\summarize_load_test.py"
if (Test-Path -LiteralPath $summaryScript) {
    & python $summaryScript $experimentRoot
}

if ($stopReason) {
    Write-Warning "Load test stopped early due to: $stopReason"
}

Write-Host "Artifacts stored in $experimentRoot"

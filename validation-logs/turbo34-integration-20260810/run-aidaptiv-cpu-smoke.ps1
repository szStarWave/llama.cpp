param(
    [string] $RunName = "run1",
    [int] $Port = 18081,
    [switch] $Managed,
    [string] $CachePrefix = "codext34-",
    [string] $CacheId = "doc-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$bin = Join-Path $root "build-validation-b10068\bin\Release"
$exe = Join-Path $bin "llama-server.exe"
$model = "C:\herdsman\models\Qwen2.5_1.5B-Instruct\windows\amd64\amd\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$runDir = Join-Path $PSScriptRoot "work\aidaptiv-cpu-q8-turbo3-$RunName"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$stdout = Join-Path $runDir "stdout.log"
$stderr = Join-Path $runDir "stderr.log"
$arguments = @(
    "-m", $model,
    "-dev", "none",
    "-ngl", "0",
    "-c", "1024",
    "-b", "128",
    "-ub", "128",
    "-ctk", "q8_0",
    "-ctv", "turbo3",
    "-fa", "on",
    "--ssd-kv-offload-gb", "1",
    "--kv-cache-resume-policy", "1",
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--no-webui",
    "-np", "1"
)
if ($Managed) {
    $arguments += @("--aidaptiv-cache-prefix", $CachePrefix)
}

$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $bin `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        if ($process.HasExited) {
            break
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $ready) {
        throw "server did not become ready"
    }

    $prompt = "Reliable storage preserves important data across restarts. " * 18
    $body = @{
        prompt = $prompt
        n_predict = 8
        temperature = 0
        seed = 1
        stream = $false
    }
    if ($Managed) {
        $body.aidaptiv_cache_id = $CacheId
    }
    $body = $body | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 180
    Write-Output "CONTENT=$($response.content)"
    Write-Output "CACHED=$($response.timings.cache_n)"

    $shutdown = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/shutdown" -Method Post `
        -ContentType "application/json" -Body "{}" -TimeoutSec 10
    Write-Output "SHUTDOWN=$($shutdown | ConvertTo-Json -Compress)"

    if (-not $process.WaitForExit(60000)) {
        throw "server did not exit after shutdown"
    }
    $process.WaitForExit()
    $process.Refresh()
    Write-Output "EXITED=$($process.HasExited)"
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}

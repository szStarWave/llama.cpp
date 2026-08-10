param(
    [switch] $ConfirmVulkanModelRun,
    [int] $BasePort = 18200,
    [int] $ContextSize = 4096
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmVulkanModelRun) {
    throw "Pass -ConfirmVulkanModelRun to run model-level Vulkan tests outside Codex."
}

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$bin = Join-Path $root "build-validation-b10068\bin\Release"
$serverExe = Join-Path $bin "llama-server.exe"
$model = "C:\herdsman\models\Qwen2.5_1.5B-Instruct\windows\amd64\amd\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$outRoot = Join-Path $PSScriptRoot ("external-vulkan-qwen25-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$cases = @(
    @{ Name = "f16-f16"; K = "f16"; V = "f16"; Persist = $false },
    @{ Name = "q8-q8"; K = "q8_0"; V = "q8_0"; Persist = $false },
    @{ Name = "q8-turbo3"; K = "q8_0"; V = "turbo3"; Persist = $true },
    @{ Name = "q8-turbo4"; K = "q8_0"; V = "turbo4"; Persist = $true },
    @{ Name = "turbo3-turbo3"; K = "turbo3"; V = "turbo3"; Persist = $false },
    @{ Name = "turbo4-turbo4"; K = "turbo4"; V = "turbo4"; Persist = $false },
    @{ Name = "turbo3-q8"; K = "turbo3"; V = "q8_0"; Persist = $false },
    @{ Name = "turbo4-q8"; K = "turbo4"; V = "q8_0"; Persist = $false }
)

function Invoke-ServerCase {
    param($Case, [int] $Pass, [int] $Port)

    $runDir = Join-Path $outRoot ("{0}-pass{1}" -f $Case.Name, $Pass)
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $stdout = Join-Path $runDir "stdout.log"
    $stderr = Join-Path $runDir "stderr.log"
    $args = @(
        "-m", $model,
        "-c", "$ContextSize",
        "-b", "256",
        "-ub", "256",
        "-ngl", "99",
        "-ctk", $Case.K,
        "-ctv", $Case.V,
        "-fa", "on",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--no-webui",
        "-np", "1"
    )
    if ($Case.Persist) {
        $args += @("--ssd-kv-offload-gb", "1", "--kv-cache-resume-policy", "1")
    }

    $process = Start-Process -FilePath $serverExe -ArgumentList $args -WorkingDirectory $bin `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    try {
        $ready = $false
        for ($i = 0; $i -lt 240; $i++) {
            if ($process.HasExited) { break }
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
                if ($health.status -eq "ok") { $ready = $true; break }
            } catch {}
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) { throw "server startup failed for $($Case.Name), pass $Pass" }

        $body = @{
            prompt = "Write one short sentence about reliable storage."
            n_predict = 24
            temperature = 0
            seed = 1
            stream = $false
        } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -ContentType "application/json" -Body $body -TimeoutSec 300
        if (-not $response.content) { throw "empty completion for $($Case.Name), pass $Pass" }

        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/shutdown" -Method Post `
            -ContentType "application/json" -Body "{}" -TimeoutSec 10 | Out-Null
        if (-not $process.WaitForExit(60000)) { throw "shutdown timeout for $($Case.Name), pass $Pass" }
        $process.WaitForExit()
        $process.Refresh()
        if (-not $process.HasExited) { throw "server did not exit for $($Case.Name), pass $Pass" }
        if (Select-String -Path $stderr -Pattern "access violation|fatal error|GPU timeout" -Quiet) {
            throw "fatal runtime error in $($Case.Name), pass $Pass; inspect $stderr"
        }

        [pscustomobject]@{
            case = $Case.Name
            pass = $Pass
            content = $response.content
            cache_n = $response.timings.cache_n
            exited = $process.HasExited
            log = $stderr
        }
    } finally {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
}

$results = @()
for ($index = 0; $index -lt $cases.Count; $index++) {
    $case = $cases[$index]
    $results += Invoke-ServerCase -Case $case -Pass 1 -Port ($BasePort + $index)
    if ($case.Persist) {
        $results += Invoke-ServerCase -Case $case -Pass 2 -Port ($BasePort + $index)
    }
}

$results | ConvertTo-Json -Depth 5 | Set-Content -Encoding ascii (Join-Path $outRoot "results.json")
$results | Format-Table -AutoSize
Write-Output "Logs: $outRoot"

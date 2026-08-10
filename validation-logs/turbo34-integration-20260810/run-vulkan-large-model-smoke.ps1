param(
    [switch] $ConfirmVulkanModelRun,
    [int] $BasePort = 18300
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmVulkanModelRun) {
    throw "Pass -ConfirmVulkanModelRun to run model-level Vulkan tests outside Codex."
}

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$bin = Join-Path $root "build-validation-b10068\bin\Release"
$serverExe = Join-Path $bin "llama-server.exe"
$fixture = Join-Path (Split-Path -Parent $PSScriptRoot) "b10068-nonturbo-20260809\vision-fixture.png"
$outRoot = Join-Path $PSScriptRoot ("external-vulkan-large-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$cases = @(
    @{
        Name = "qwen35-27b-hybrid"
        Model = "C:\herdsman\models\Qwen3.5-27B\windows\amd64\amd\Qwen3.5-27B-Q4_K_M.gguf"
        Mmproj = "C:\herdsman\models\Qwen3.5-27B\windows\amd64\amd\mmproj-BF16.gguf"
        Extra = @()
    },
    @{
        Name = "qwen35-35b-a3b-expert"
        Model = "C:\herdsman\models\Qwen3.5_35b_a3b\windows\amd64\amd\Qwen3.5-35B-A3B-Q4_K_M.gguf"
        Mmproj = $null
        Extra = @("--vram-experts-cached-gb", "4", "--dram-experts-cached-gb", "8")
    },
    @{
        Name = "gemma4-e2b-vision"
        Model = "C:\herdsman\models\Gemma4_E2B-IT\windows\amd64\amd\gemma-4-E2B-it-Q4_K_M.gguf"
        Mmproj = "C:\herdsman\models\Gemma4_E2B-IT\windows\amd64\amd\mmproj-BF16.gguf"
        Extra = @()
    }
)

function Invoke-LargeCase {
    param($Case, [string] $VType, [int] $Port)

    $name = "$($Case.Name)-q8-$VType"
    $runDir = Join-Path $outRoot $name
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $stdout = Join-Path $runDir "stdout.log"
    $stderr = Join-Path $runDir "stderr.log"
    $args = @(
        "-m", $Case.Model,
        "-c", "4096",
        "-b", "512",
        "-ub", "512",
        "-ngl", "99",
        "-ctk", "q8_0",
        "-ctv", $VType,
        "-fa", "on",
        "--ssd-kv-offload-gb", "4",
        "--dram-kv-offload-gb", "2",
        "--kv-cache-resume-policy", "1",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--no-webui",
        "-np", "1"
    )
    if ($Case.Mmproj) {
        $args += @("--mmproj", $Case.Mmproj, "--jinja", "--image-max-tokens", "1024")
    }
    $args += $Case.Extra

    $process = Start-Process -FilePath $serverExe -ArgumentList $args -WorkingDirectory $bin `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    try {
        $ready = $false
        for ($i = 0; $i -lt 1200; $i++) {
            if ($process.HasExited) { break }
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
                if ($health.status -eq "ok") { $ready = $true; break }
            } catch {}
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) { throw "server startup failed for $name" }

        if ($Case.Mmproj) {
            $image = [Convert]::ToBase64String([IO.File]::ReadAllBytes($fixture))
            $body = @{
                messages = @(@{
                    role = "user"
                    content = @(
                        @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$image" } },
                        @{ type = "text"; text = "Describe the image in one short sentence." }
                    )
                })
                max_tokens = 24
                temperature = 0
                seed = 1
                stream = $false
            } | ConvertTo-Json -Depth 8
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType "application/json" -Body $body -TimeoutSec 600
            $content = $response.choices[0].message.content
        } else {
            $body = @{
                prompt = "Write one short sentence about reliable storage."
                n_predict = 24
                temperature = 0
                seed = 1
                stream = $false
            } | ConvertTo-Json
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
                -ContentType "application/json" -Body $body -TimeoutSec 600
            $content = $response.content
        }
        if (-not $content) { throw "empty completion for $name" }

        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/shutdown" -Method Post `
            -ContentType "application/json" -Body "{}" -TimeoutSec 10 | Out-Null
        if (-not $process.WaitForExit(120000)) { throw "shutdown timeout for $name" }
        $process.WaitForExit()
        $process.Refresh()
        if (-not $process.HasExited) { throw "server did not exit for $name" }
        if (Select-String -Path $stderr -Pattern "access violation|fatal error|GPU timeout" -Quiet) {
            throw "fatal runtime error in $name; inspect $stderr"
        }
        [pscustomobject]@{ case = $name; content = $content; exited = $process.HasExited; log = $stderr }
    } finally {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
}

$results = @()
$portOffset = 0
foreach ($case in $cases) {
    foreach ($vType in @("turbo3", "turbo4")) {
        $results += Invoke-LargeCase -Case $case -VType $vType -Port ($BasePort + $portOffset)
        $portOffset++
    }
}

$results | ConvertTo-Json -Depth 5 | Set-Content -Encoding ascii (Join-Path $outRoot "results.json")
$results | Format-Table -AutoSize
Write-Output "Logs: $outRoot"

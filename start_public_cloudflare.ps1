param([switch]$TunnelOnly)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppUrl = "http://127.0.0.1:7860"

$PythonExe = "D:\CondaEnvs\disertatie\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$CloudflaredExe = $null
$CloudflaredCommand = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($CloudflaredCommand) {
    $CloudflaredExe = $CloudflaredCommand.Source
}
else {
    $CloudflaredCandidates = @(
        "C:\Program Files\cloudflared\cloudflared.exe",
        "C:\Program Files (x86)\cloudflared\cloudflared.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
    )
    foreach ($Candidate in $CloudflaredCandidates) {
        if (Test-Path $Candidate) {
            $CloudflaredExe = $Candidate
            break
        }
    }
}

if (-not $CloudflaredExe) {
    throw 'cloudflared was not found. Install it with: winget install Cloudflare.cloudflared. Then reopen PowerShell and retry. Any already-running local app has been left running.'
}

$AppProcess = $null
if (-not $TunnelOnly) {
    # Only the standalone launcher owns an app process. The defense launcher
    # supplies a ready app and must keep its exact model settings and lifetime.
    $env:SERVER_HOST = "127.0.0.1"
    $env:SERVER_PORT = "7860"

    if (-not $env:OPENAI_BASE_URL) {
        $env:OPENAI_BASE_URL = "http://localhost:11434/v1"
    }
    if (-not $env:OPENAI_API_KEY) {
        $env:OPENAI_API_KEY = "ollama"
    }
    if (-not $env:OPENAI_MODEL) {
        $env:OPENAI_MODEL = "qwen3:8b"
    }
    if (-not $env:OPENAI_RAG_MODEL) {
        $env:OPENAI_RAG_MODEL = "mistral-small3.2:latest"
    }
    if (-not $env:OPENAI_CALCULATION_MODEL) {
        $env:OPENAI_CALCULATION_MODEL = "qwen3:14b"
    }
    if (-not $env:OLLAMA_CALCULATION_THINK) {
        $env:OLLAMA_CALCULATION_THINK = "true"
    }
    if (-not $env:OPENAI_CALCULATION_TEMPERATURE) {
        $env:OPENAI_CALCULATION_TEMPERATURE = "0.1"
    }
    if (-not $env:OLLAMA_CALCULATION_NUM_PREDICT) {
        $env:OLLAMA_CALCULATION_NUM_PREDICT = "6000"
    }
    if (-not $env:OPENAI_SYNTHESIS_MODEL) {
        $env:OPENAI_SYNTHESIS_MODEL = "mistral-small3.2:latest"
    }
    if (-not $env:OPENAI_TIMEOUT_SECONDS) {
        $env:OPENAI_TIMEOUT_SECONDS = "180"
    }
    if (-not $env:OPENAI_MAX_TOKENS) {
        $env:OPENAI_MAX_TOKENS = "1800"
    }

    # Avoid app.py's stale-listener cleanup replacing a running application.
    $PortProbe = New-Object System.Net.Sockets.TcpClient
    $AlreadyRunning = $false
    try {
        $PortProbe.Connect('127.0.0.1', 7860)
        $AlreadyRunning = $true
    }
    catch [System.Net.Sockets.SocketException] { }
    finally { $PortProbe.Dispose() }

    if ($AlreadyRunning) {
        Write-Host 'Port 7860 is in use. Verifying the existing app before creating a tunnel.'
    }
    else {
        Write-Host "Starting the Credit Assistant at $AppUrl ..."
        $AppProcess = Start-Process -FilePath $PythonExe -ArgumentList "app.py" `
            -WorkingDirectory $ProjectDir -PassThru -WindowStyle Hidden
    }
}

try {
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 45; $Attempt++) {
        if ($AppProcess) {
            $AppProcess.Refresh()
            if ($AppProcess.HasExited) { throw 'NovaTech exited before it was ready. No tunnel was started.' }
        }
        $ConfigResponse = $null
        try {
            $ConfigResponse = Invoke-WebRequest -Uri "$AppUrl/config" -UseBasicParsing -TimeoutSec 2
        }
        catch { }
        if ($ConfigResponse) {
            # Read the raw JSON: Gradio uses empty property names unsupported
            # by ConvertFrom-Json in Windows PowerShell 5.1.
            if ($ConfigResponse.StatusCode -ne 200 -or
                $ConfigResponse.Content -notmatch '"title"\s*:\s*"NovaTech Credit Assistant"' -or
                $ConfigResponse.Content -notmatch '"elem_id"\s*:\s*"novatech-hero"') {
                throw 'Port 7860 is not serving NovaTech. No tunnel was started.'
            }
            $Ready = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $Ready) { throw "NovaTech is not ready at $AppUrl. No tunnel was started." }

    Write-Host ""
    Write-Host "Creating a temporary public link. Share the https://...trycloudflare.com URL displayed below."
    Write-Host "Anyone with this link can access the app, including /runtime-errors. Use synthetic demo data only."
    Write-Host "Keep this terminal open. Press Ctrl+C to stop the tunnel."
    Write-Host ""
    & $CloudflaredExe tunnel --url $AppUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Cloudflare exited with code $LASTEXITCODE. Check its output above and your Internet connection."
    }
}
finally {
    if ($AppProcess -and -not $AppProcess.HasExited) {
        Stop-Process -Id $AppProcess.Id -Force
    }
}

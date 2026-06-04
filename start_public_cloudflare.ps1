$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$PythonExe = "D:\CondaEnvs\disertatie\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$env:SERVER_HOST = "0.0.0.0"
$env:SERVER_PORT = "7860"

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

if (-not $env:OPENAI_BASE_URL) {
    $env:OPENAI_BASE_URL = "http://localhost:11434/v1"
}
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = "ollama"
}
if (-not $env:OPENAI_MODEL) {
    $env:OPENAI_MODEL = "qwen3:8b"
}
if (-not $env:OPENAI_TIMEOUT_SECONDS) {
    $env:OPENAI_TIMEOUT_SECONDS = "180"
}
if (-not $env:OPENAI_MAX_TOKENS) {
    $env:OPENAI_MAX_TOKENS = "1800"
}

Write-Host "Pornesc Asistentul de Creditare pe http://127.0.0.1:7860 ..."
$AppProcess = Start-Process -FilePath $PythonExe -ArgumentList "app.py" -PassThru -WindowStyle Hidden

try {
    Start-Sleep -Seconds 8

    if (-not $CloudflaredExe) {
        Write-Host ""
        Write-Host "cloudflared nu este instalat."
        Write-Host "Instaleaza-l cu:"
        Write-Host "  winget install Cloudflare.cloudflared"
        Write-Host ""
        Write-Host "Daca winget spune ca este deja instalat, inchide si redeschide PowerShell."
        Write-Host ""
        Write-Host "Dupa instalare, ruleaza din nou:"
        Write-Host "  .\start_public_cloudflare.ps1"
        Write-Host ""
        Write-Host "Serverul local ramane pornit pana inchizi aceasta fereastra."
        Wait-Process -Id $AppProcess.Id
        exit 1
    }

    Write-Host ""
    Write-Host "Creez link public temporar. Trimite URL-ul https://...trycloudflare.com afisat mai jos."
    Write-Host "Pentru oprire, apasa Ctrl+C in acest terminal."
    Write-Host ""
    & $CloudflaredExe tunnel --url "http://127.0.0.1:7860"
}
finally {
    if ($AppProcess -and -not $AppProcess.HasExited) {
        Stop-Process -Id $AppProcess.Id -Force
    }
}

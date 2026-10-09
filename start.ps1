# ###수정 원본과 분리된 Python 환경에서 CMSL GUI를 기본 8092 포트로 실행합니다.
param(
    [int]$Port = 8092,
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
# ###수정 기존 RapidOCR를 지원하는 Python 3.12 전용 실험 환경을 우선 사용합니다.
$python = Join-Path $experimentRoot '.venv312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $experimentRoot '.venv\Scripts\python.exe'
}
# ###
$appRoot = Join-Path $experimentRoot 'plotlogic_mvp'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create the experiment .venv312 and install requirements-experiment.txt first.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use; specify a different -Port."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:CMSL_PORT = [string]$Port
$logs = Join-Path $experimentRoot 'data\server'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if ($Background) {
    $arguments = @('-m', 'uvicorn', 'app.main:app', '--app-dir', ('"' + $appRoot + '"'), '--host', '127.0.0.1', '--port', [string]$Port)
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'stdout.log') -RedirectStandardError (Join-Path $logs 'stderr.log')
    $process.Id | Set-Content -LiteralPath (Join-Path $logs 'pid.txt')
    Write-Output "CMSL GUI started: http://127.0.0.1:$Port/ (PID $($process.Id))"
} else {
    & $python -m uvicorn app.main:app --app-dir $appRoot --host 127.0.0.1 --port $Port
}
# ###

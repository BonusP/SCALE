# ###수정 Python 3.12 전용 환경에 실험 의존성을 설치합니다.
param([string]$Python312 = '')
$ErrorActionPreference = 'Stop'
$environmentPath = Join-Path $PSScriptRoot '.venv312'
if (-not (Test-Path -LiteralPath (Join-Path $environmentPath 'Scripts\python.exe'))) {
    if ($Python312) {
        & $Python312 -m venv $environmentPath
    } else {
        & py -3.12 -m venv $environmentPath
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
}
& (Join-Path $environmentPath 'Scripts\python.exe') -X utf8 -m pip install -r (Join-Path $PSScriptRoot 'requirements-experiment.txt')
if ($LASTEXITCODE -ne 0) { throw 'Experiment dependency installation failed.' }
# ###

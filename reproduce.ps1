param([ValidateSet('smoke','standard','full')][string]$Preset='smoke',[int]$Workers=1)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if(-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if($LASTEXITCODE -ne 0){throw 'Virtual environment creation failed'}
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if($LASTEXITCODE -ne 0){throw 'Dependency installation failed'}
}
& '.\.venv\Scripts\python.exe' reproduce.py --preset $Preset --workers $Workers
if($LASTEXITCODE -ne 0){throw 'Reproduction failed; see command above'}

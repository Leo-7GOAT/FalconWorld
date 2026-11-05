param([switch]$VerifyOnly)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$rlPython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $rlPython)) {
    if($VerifyOnly){throw 'Create the project environment with .\setup_rl.ps1 first'}
    python -m venv .venv
    if($LASTEXITCODE -ne 0){throw 'Failed to create project environment'}
}
if(-not $VerifyOnly) {
    & $rlPython -m pip install -r requirements-rl-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
    if($LASTEXITCODE -ne 0){throw 'Dependency installation failed'}
}
@'
from importlib.metadata import version
from pathlib import Path
for line in Path('requirements-rl-lock.txt').read_text(encoding='utf-8').splitlines():
    if not line or line.startswith('#'):continue
    package,expected=line.split('==',1)
    actual=version(package)
    if actual!=expected:raise RuntimeError(f'{package}: expected {expected}, installed {actual}')
import torch,stable_baselines3,gymnasium
assert torch.version.cuda is None, 'This registered environment uses CPU Torch'
print('Pinned runtime verified:',torch.__version__,stable_baselines3.__version__,gymnasium.__version__)
'@ | & $rlPython -
if($LASTEXITCODE -ne 0){throw 'Runtime verification failed'}

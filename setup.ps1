$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $projectPython -c "from data import download; print(download())"
if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed.' }
& $projectPython test_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model checks failed.' }
Write-Host 'Setup complete. Run .venv\Scripts\python train.py --epochs 12.'

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYINSTALLER_CONFIG_DIR = Join-Path $PSScriptRoot '.build-cache'
# Keep unrelated applications' DLLs out of PyInstaller's dependency search.
$onbitPreviousPath = $env:PATH
$env:PATH = @((Join-Path $PSScriptRoot '.venv\Scripts'),
    (Join-Path $env:SystemRoot 'System32'), $env:SystemRoot) -join ';'
try {
& '.\.venv\Scripts\python.exe' build_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Icon build failed.' }
& '.\.venv\Scripts\python.exe' -m PyInstaller --clean --noconfirm Onbit.spec
if ($LASTEXITCODE -ne 0) { throw 'Executable build failed.' }
} finally {
    $env:PATH = $onbitPreviousPath
}
Write-Output (Join-Path $PSScriptRoot 'dist\Onbit\Onbit.exe')

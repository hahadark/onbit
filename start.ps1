$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$executable = Join-Path $PSScriptRoot 'dist\Onbit\Onbit.exe'
if (Test-Path -LiteralPath $executable) {
    Start-Process -FilePath $executable -WorkingDirectory (Split-Path $executable) -WindowStyle Normal
} else {
    $pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup.ps1 and build.ps1 first.' }
    Start-Process -FilePath $pythonPath -ArgumentList 'desktop.py' -WorkingDirectory $PSScriptRoot -WindowStyle Normal
}

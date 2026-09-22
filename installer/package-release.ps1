# Package both editions for a GitHub release and build the small web installer.
#
#   1. Build the editions first (PyInstaller), e.g. into build\release-gpu\Onbit and build\release-cpu\Onbit.
#   2. powershell -ExecutionPolicy Bypass -File installer\package-release.ps1 -Gpu <dir> -Cpu <dir>
#
# Output (installer\Output): OnbitSetup-<version>.exe and Onbit-<version>-*.7z. Upload all of them
# to the GitHub release v<version>; the installer downloads the archives from there.
# -BaseUrl builds a test installer that downloads from another place (e.g. a local web server).
param(
    [Parameter(Mandatory = $true)][string]$Gpu,
    [Parameter(Mandatory = $true)][string]$Cpu,
    [string]$Version = '1.0.0',
    [string]$BaseUrl = ''
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$output = Join-Path $here 'Output'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$sevenZip = @("$env:ProgramFiles\7-Zip\7z.exe", "${env:ProgramFiles(x86)}\7-Zip\7z.exe") |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $sevenZip) { throw '7-Zip is required: winget install 7zip.7zip' }
$compiler = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
              "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $compiler) { throw 'Inno Setup 6.5+ is required: winget install JRSoftware.InnoSetup' }

function Assert-Payload([string]$dir) {
    if (-not (Test-Path -LiteralPath (Join-Path $dir 'Onbit.exe'))) { throw "Onbit.exe not found in $dir" }
    # Never ship a local library, logs or exports that a test run may have left behind.
    $extra = Get-ChildItem -LiteralPath $dir -Force | Where-Object { $_.Name -notin @('Onbit.exe', '_internal') }
    if ($extra) { throw "Unexpected files in ${dir}: $($extra.Name -join ', ')" }
}

function New-Archive([string]$dir, [string]$name, [string[]]$relative) {
    $archive = Join-Path $output $name
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
    $list = Join-Path $env:TEMP "onbit-$name.txt"
    [IO.File]::WriteAllLines($list, $relative, (New-Object Text.UTF8Encoding $false))
    Push-Location -LiteralPath $dir
    try {
        # Non-solid (-ms=off): the installer extracts file by file, and a solid block would be
        # decompressed again from its start for every file.
        & $sevenZip a -t7z -mx=7 -ms=off -mmt=on -bso0 -bsp0 -scsUTF-8 $archive "@$list"
        if ($LASTEXITCODE -ne 0) { throw "7-Zip failed for $name" }
    } finally { Pop-Location; Remove-Item -LiteralPath $list }
    $bytes = (Get-Item -LiteralPath $archive).Length
    if ($bytes -ge 2GB) { throw "$name is $bytes bytes; GitHub release assets must stay below 2 GiB" }
    $extracted = 0
    foreach ($path in $relative) { $extracted += (Get-Item -LiteralPath (Join-Path $dir $path)).Length }
    [pscustomobject]@{ Name = $name; Bytes = $bytes; Extracted = $extracted
                       Sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() }
}

function Get-Files([string]$dir) {
    $root = (Resolve-Path -LiteralPath $dir).Path.TrimEnd('\') + '\'
    Get-ChildItem -LiteralPath $dir -Recurse -File -Force | ForEach-Object {
        [pscustomobject]@{ Path = $_.FullName.Substring($root.Length); Length = $_.Length }
    }
}

Assert-Payload $Gpu
Assert-Payload $Cpu

# CPU edition: one archive.
$cpuArchive = New-Archive $Cpu "Onbit-$Version-cpu.7z" @((Get-Files $Cpu).Path)

# GPU edition: the large CUDA libraries go into two balanced archives, everything else into a core archive.
$gpuFiles = @(Get-Files $Gpu)
$cuda = @($gpuFiles | Where-Object { $_.Path -like '_internal\torch\lib\*' -and $_.Length -gt 50MB } |
          Sort-Object Length -Descending)
$core = @($gpuFiles | Where-Object { $cuda.Path -notcontains $_.Path })
$parts = @((New-Object Collections.ArrayList), (New-Object Collections.ArrayList)); $totals = @(0, 0)
foreach ($file in $cuda) {
    $index = if ($totals[0] -le $totals[1]) { 0 } else { 1 }
    [void]$parts[$index].Add($file.Path); $totals[$index] += $file.Length
}
$gpuCore = New-Archive $Gpu "Onbit-$Version-gpu-core.7z" @($core.Path)
$gpuCuda1 = New-Archive $Gpu "Onbit-$Version-gpu-cuda1.7z" ([string[]]$parts[0])
$gpuCuda2 = New-Archive $Gpu "Onbit-$Version-gpu-cuda2.7z" ([string[]]$parts[1])

function Format-Size([long]$bytes) {
    if ($bytes -ge 1GB) { return ('{0:N1}GB' -f ($bytes / 1GB)) }
    return ('{0:N0}MB' -f ($bytes / 1MB))
}
$gpuDownload = $gpuCore.Bytes + $gpuCuda1.Bytes + $gpuCuda2.Bytes
$defines = @(
    "#define CpuArchive `"$($cpuArchive.Name)`"", "#define CpuSha256 `"$($cpuArchive.Sha256)`"",
    "#define CpuExtracted `"$($cpuArchive.Extracted)`"",
    "#define GpuCoreArchive `"$($gpuCore.Name)`"", "#define GpuCoreSha256 `"$($gpuCore.Sha256)`"",
    "#define GpuCoreExtracted `"$($gpuCore.Extracted)`"",
    "#define GpuCuda1Archive `"$($gpuCuda1.Name)`"", "#define GpuCuda1Sha256 `"$($gpuCuda1.Sha256)`"",
    "#define GpuCuda1Extracted `"$($gpuCuda1.Extracted)`"",
    "#define GpuCuda2Archive `"$($gpuCuda2.Name)`"", "#define GpuCuda2Sha256 `"$($gpuCuda2.Sha256)`"",
    "#define GpuCuda2Extracted `"$($gpuCuda2.Extracted)`"",
    "#define GpuDownloadText `"$(Format-Size $gpuDownload)`"", "#define CpuDownloadText `"$(Format-Size $cpuArchive.Bytes)`""
)
[IO.File]::WriteAllLines((Join-Path $output 'assets.iss'), $defines, (New-Object Text.UTF8Encoding $true))

$arguments = @("/DAppVersion=$Version")
if ($BaseUrl) { $arguments += "/DBaseUrl=$BaseUrl" }
& $compiler @arguments (Join-Path $here 'onbit.iss')
if ($LASTEXITCODE -ne 0) { throw "Installer build failed ($LASTEXITCODE)." }
$cpuArchive, $gpuCore, $gpuCuda1, $gpuCuda2 | Format-Table Name, Bytes, Extracted, Sha256 -AutoSize

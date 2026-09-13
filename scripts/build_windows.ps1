param(
    [string]$Python = "python",
    [string]$Iscc = "",
    [switch]$SkipTests,
    [switch]$SkipInstaller
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    $Python = (Get-Command $Python -ErrorAction Stop).Source
    $Version = (& $Python -c "from Toolbox.version import APP_VERSION; print(APP_VERSION)").Trim()
    if ($LASTEXITCODE -ne 0) { throw "Cannot read the application version." }
    if (-not $SkipTests) {
        & $Python -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) { throw "Regression tests failed." }
    }
    & $Python scripts/prepare_build.py
    if ($LASTEXITCODE -ne 0) { throw "Build preparation failed." }
    & $Python -m PyInstaller --noconfirm --clean --distpath dist --workpath .build/pyinstaller Toolbox/toolbox.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

    $AppDir = Join-Path $RepoRoot "dist\RemoteSensingToolbox"
    $AppExe = Join-Path $AppDir "RemoteSensingToolbox.exe"
    $Report = Join-Path $RepoRoot ".build\frozen-self-test.json"
    if (Test-Path $Report) { Remove-Item $Report }
    $ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
    $ProcessInfo.FileName = $AppExe
    $ProcessInfo.Arguments = "--self-test --report `"$Report`""
    $ProcessInfo.WorkingDirectory = Join-Path $RepoRoot ".build"
    $ProcessInfo.UseShellExecute = $false
    $ProcessInfo.EnvironmentVariables["PATH"] = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem"
    foreach ($Name in @("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "GDAL_DATA", "PROJ_LIB", "PROJ_DATA", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH")) {
        $ProcessInfo.EnvironmentVariables.Remove($Name)
    }
    $ProcessInfo.EnvironmentVariables["QT_QPA_PLATFORM"] = "offscreen"
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $ProcessInfo
    if (-not $Process.Start()) { throw "Could not start the frozen application self-test." }
    if (-not $Process.WaitForExit(180000)) {
        $Process.Kill()
        throw "The frozen application self-test timed out."
    }
    $Process.Refresh()
    if ($Process.ExitCode -ne 0 -or -not (Test-Path $Report)) { throw "The frozen application self-test failed." }
    $Result = Get-Content $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $Result.ok -or -not $Result.frozen) { throw "Frozen self-test did not pass." }

    Copy-Item THIRD_PARTY_NOTICES.md $AppDir
    $ReleaseDir = Join-Path $RepoRoot "release"
    New-Item -ItemType Directory -Force $ReleaseDir | Out-Null
    if (-not $SkipInstaller) {
        if (-not $Iscc) {
            $Candidates = @(
                (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
                "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
            )
            $Iscc = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        }
        if (-not $Iscc) { throw "Inno Setup 6 was not found. Install it or pass -Iscc with its path." }
        & $Iscc "/DAppVersion=$Version" "/DSourceDir=$AppDir" "/DReleaseDir=$ReleaseDir" installer/toolbox.iss
        if ($LASTEXITCODE -ne 0) { throw "Installer compilation failed." }
    }
    & $Python scripts/package_release.py
    if ($LASTEXITCODE -ne 0) { throw "Release archive creation failed." }
    Write-Host "Release artifacts: $ReleaseDir"
}
finally { Pop-Location }

param(
    [string]$Installer = "",
    [switch]$CheckRenderer
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VerificationDir = Join-Path $RepoRoot ".build\installer-verification"
$InstallDir = Join-Path $VerificationDir "安装 验证"
$UninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{E07C5774-6B8F-45DF-A905-A416A7D4CB9C}_is1"
if (Test-Path $UninstallKey) { throw "An existing installation is registered; do not run an isolated install/uninstall test over it." }
if (Test-Path $InstallDir) { throw "The verification install directory already exists; inspect it before testing again." }
if (-not $Installer) {
    $Installer = (Get-ChildItem (Join-Path $RepoRoot "release\*-setup.exe") | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
if (-not $Installer -or -not (Test-Path $Installer)) { throw "Installer not found." }
New-Item -ItemType Directory -Force $VerificationDir | Out-Null
$GroupName = "RemoteSensingToolbox Validation"
$MenuDir = Join-Path ([Environment]::GetFolderPath("Programs")) $GroupName
$AppExe = Join-Path $InstallDir "RemoteSensingToolbox.exe"
$Report = Join-Path $VerificationDir "installed-self-test.json"
$Installed = $false
$Results = [ordered]@{ installer = (Split-Path -Leaf $Installer); install = $false; shortcuts = $false; self_test = $false; native_launch = $false; uninstall = $false }

function New-CleanProcess([string]$Arguments, [string]$QtPlatform) {
    $Info = New-Object System.Diagnostics.ProcessStartInfo
    $Info.FileName = $AppExe
    $Info.Arguments = $Arguments
    $Info.WorkingDirectory = $VerificationDir
    $Info.UseShellExecute = $false
    $Info.EnvironmentVariables["PATH"] = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem"
    foreach ($Name in @("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "GDAL_DATA", "PROJ_LIB", "PROJ_DATA", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH")) {
        $Info.EnvironmentVariables.Remove($Name)
    }
    $Info.EnvironmentVariables["QT_QPA_PLATFORM"] = $QtPlatform
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $Info
    if (-not $Process.Start()) { throw "Could not start the installed application." }
    return $Process
}

try {
    $InstallLog = Join-Path $VerificationDir "install.log"
    $Process = Start-Process -FilePath $Installer -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", "/CURRENTUSER", "/DIR=`"$InstallDir`"", "/GROUP=`"$GroupName`"", "/LOG=`"$InstallLog`"") -Wait -PassThru
    if ($Process.ExitCode -ne 0 -or -not (Test-Path $AppExe)) { throw "Silent installation failed." }
    $Installed = $true
    $Results.install = $true
    if (-not (Test-Path $UninstallKey)) { throw "The uninstall registration was not created." }
    $Shortcuts = @(Get-ChildItem $MenuDir -Filter *.lnk)
    $Shell = New-Object -ComObject WScript.Shell
    $Targets = @($Shortcuts | ForEach-Object { $Shell.CreateShortcut($_.FullName).TargetPath })
    if ($Targets -notcontains $AppExe) { throw "The start menu application shortcut is missing." }
    $Results.shortcuts = $true

    if (Test-Path $Report) { Remove-Item $Report }
    $Arguments = "--self-test --report `"$Report`""
    if ($CheckRenderer) { $Arguments += " --check-renderer" }
    $Process = New-CleanProcess $Arguments "offscreen"
    if (-not $Process.WaitForExit(180000)) { $Process.Kill(); throw "Installed self-test timed out." }
    if ($Process.ExitCode -ne 0 -or -not (Test-Path $Report)) { throw "Installed self-test failed." }
    $SelfTest = Get-Content $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $SelfTest.ok -or -not $SelfTest.frozen) { throw "Installed self-test reported a failure." }
    $Results.self_test = $true
    $Results.checks = @($SelfTest.checks).Count
    $Results.renderer = [bool]$CheckRenderer

    $Process = New-CleanProcess "" "windows"
    $Deadline = [DateTime]::UtcNow.AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 100
        $Process.Refresh()
    } while (-not $Process.HasExited -and $Process.MainWindowHandle -eq 0 -and [DateTime]::UtcNow -lt $Deadline)
    if ($Process.HasExited -or $Process.MainWindowHandle -eq 0) {
        if (-not $Process.HasExited) { $Process.Kill() }
        throw "The native Windows application window did not appear."
    }
    $Results.window_title = $Process.MainWindowTitle
    $Process.CloseMainWindow() | Out-Null
    if (-not $Process.WaitForExit(15000)) { $Process.Kill(); throw "The application did not close cleanly." }
    if ($Process.ExitCode -ne 0) { throw "The application exited with an error." }
    $Results.native_launch = $true
}
finally {
    if ($Installed) {
        $Uninstaller = Join-Path $InstallDir "unins000.exe"
        $UninstallLog = Join-Path $VerificationDir "uninstall.log"
        $Process = Start-Process -FilePath $Uninstaller -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$UninstallLog`"") -Wait -PassThru
        if ($Process.ExitCode -eq 0 -and -not (Test-Path $AppExe) -and -not (Test-Path $UninstallKey) -and -not (Test-Path $MenuDir)) {
            $Results.uninstall = $true
        }
    }
    $Results | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $VerificationDir "result.json") -Encoding UTF8
}
if (-not $Results.uninstall) { throw "Uninstall verification failed." }
$Results | ConvertTo-Json -Depth 5

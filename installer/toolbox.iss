#ifndef AppVersion
  #define AppVersion "1.1.1"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\RemoteSensingToolbox"
#endif
#ifndef ReleaseDir
  #define ReleaseDir "..\release"
#endif

[Setup]
AppId={{E07C5774-6B8F-45DF-A905-A416A7D4CB9C}
AppName=遥感与点云综合工具箱
AppVersion={#AppVersion}
AppVerName=遥感与点云综合工具箱 {#AppVersion}
AppPublisher=yaluncoco
AppPublisherURL=https://github.com/yaluncoco/RemoteSensingToolbox
DefaultDirName={localappdata}\Programs\RemoteSensingToolbox
DefaultGroupName=遥感与点云综合工具箱
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#ReleaseDir}
OutputBaseFilename=RemoteSensingToolbox-{#AppVersion}-windows-x64-setup
SetupIconFile=..\PointCloud\images\point.ico
UninstallDisplayIcon={app}\RemoteSensingToolbox.exe
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter=RemoteSensingToolbox.exe
RestartApplications=no
SetupLogging=yes
VersionInfoVersion={#AppVersion}
VersionInfoDescription=遥感与点云综合工具箱安装程序

[Languages]
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
Type: files; Name: "{app}\使用说明.md"
Type: files; Name: "{app}\_internal\quick_start.md"
Type: files; Name: "{group}\使用说明.lnk"

[Icons]
Name: "{group}\遥感与点云综合工具箱"; Filename: "{app}\RemoteSensingToolbox.exe"; WorkingDir: "{app}"
Name: "{group}\卸载"; Filename: "{uninstallexe}"
Name: "{autodesktop}\遥感与点云综合工具箱"; Filename: "{app}\RemoteSensingToolbox.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RemoteSensingToolbox.exe"; Description: "启动遥感与点云综合工具箱"; Flags: nowait postinstall skipifsilent

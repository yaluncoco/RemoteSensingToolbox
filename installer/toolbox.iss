#ifndef AppVersion
  #define AppVersion "1.3.0"
#endif
#ifndef AppIdentity
  #define AppIdentity "{{E07C5774-6B8F-45DF-A905-A416A7D4CB9C}"
#endif
#ifndef AppDisplayName
  #define AppDisplayName "遥感与点云综合工具箱"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\RemoteSensingToolbox"
#endif
#ifndef ReleaseDir
  #define ReleaseDir "..\release"
#endif

[Setup]
AppId={#AppIdentity}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppVerName={#AppDisplayName} {#AppVersion}
AppPublisher=yaluncoco
AppPublisherURL=https://github.com/yaluncoco/RemoteSensingToolbox
DefaultDirName={localappdata}\Programs\RemoteSensingToolbox
DefaultGroupName={#AppDisplayName}
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
Filename: "{app}\RemoteSensingToolbox.exe"; Flags: nowait; Check: RestartAfterUpdate

[Code]
const
  SynchronizeAccess = $00100000;

function OpenProcess(Access: LongWord; Inherit: Boolean; ProcessId: LongWord): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function WaitForSingleObject(Handle: THandle; Milliseconds: LongWord): LongWord;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function CloseHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function RestartAfterUpdate: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RESTARTAPP|0}') = '1');
end;

function InitializeSetup: Boolean;
var
  ParentId: Integer;
  ParentHandle: THandle;
  WaitResult: LongWord;
begin
  Result := True;
  ParentId := StrToIntDef(ExpandConstant('{param:UPDATEPID|0}'), 0);
  if ParentId <= 0 then
    Exit;
  ParentHandle := OpenProcess(SynchronizeAccess, False, ParentId);
  if ParentHandle = 0 then
    Exit; { The old application already exited. }
  try
    Log('Waiting for the previous application process to exit.');
    WaitResult := WaitForSingleObject(ParentHandle, 60000);
    if WaitResult <> 0 then
    begin
      MsgBox('旧版本尚未退出，更新未开始。请关闭工具箱后重新安装。', mbError, MB_OK);
      Result := False;
    end;
  finally
    CloseHandle(ParentHandle);
  end;
end;

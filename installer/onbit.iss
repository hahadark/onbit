; Onbit web installer (Inno Setup 6.5+). Build with installer\package-release.ps1.
; The setup itself is small: it downloads only the edition the user picks (GPU or CPU)
; from the GitHub release, verifies SHA-256, and extracts it. Each release asset stays
; below GitHub's 2 GiB limit.

#define AppName "온빛 Onbit"
#define AppExe "Onbit.exe"
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef BaseUrl
  #define BaseUrl "https://github.com/hahadark/onbit/releases/download/v" + AppVersion + "/"
#endif
; Archive names, SHA-256, extracted sizes and download sizes, written by package-release.ps1.
#include "Output\assets.iss"

[Setup]
AppId={{6F3B8E2A-9C41-4D57-B0E3-2A7C5D1F8E64}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=hahadark
AppPublisherURL=https://github.com/hahadark/onbit
AppSupportURL=https://github.com/hahadark/onbit/issues
AppUpdatesURL=https://github.com/hahadark/onbit/releases
; Per-user install: no administrator rights, and the program folder stays writable
; for its portable settings (onbit-data).
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\Onbit
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
SetupIconFile=..\assets\onbit.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
InfoBeforeFile=notice-ko.txt
OutputDir=Output
OutputBaseFilename=OnbitSetup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchiveExtraction=enhanced/nopassword

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 작업:"; Flags: unchecked

[InstallDelete]
; Switching editions (GPU <-> CPU) must not leave the other edition's libraries behind.
; The user's library settings in onbit-data are kept.
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\{#AppExe}"

[Files]
Source: "{#BaseUrl}{#CpuArchive}"; DestName: "{#CpuArchive}"; DestDir: "{app}"; \
  Hash: "{#CpuSha256}"; ExternalSize: {#CpuExtracted}; Check: IsCpuEdition; \
  Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion
Source: "{#BaseUrl}{#GpuCoreArchive}"; DestName: "{#GpuCoreArchive}"; DestDir: "{app}"; \
  Hash: "{#GpuCoreSha256}"; ExternalSize: {#GpuCoreExtracted}; Check: IsGpuEdition; \
  Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion
Source: "{#BaseUrl}{#GpuCuda1Archive}"; DestName: "{#GpuCuda1Archive}"; DestDir: "{app}"; \
  Hash: "{#GpuCuda1Sha256}"; ExternalSize: {#GpuCuda1Extracted}; Check: IsGpuEdition; \
  Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion
Source: "{#BaseUrl}{#GpuCuda2Archive}"; DestName: "{#GpuCuda2Archive}"; DestDir: "{app}"; \
  Hash: "{#GpuCuda2Sha256}"; ExternalSize: {#GpuCuda2Extracted}; Check: IsGpuEdition; \
  Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "온빛 실행"; Flags: nowait postinstall skipifsilent

[Code]
var
  EditionPage: TInputOptionWizardPage;

function HasNvidiaDriver: Boolean;
begin
  // The NVIDIA display driver installs the CUDA driver library; no toolkit is needed.
  Result := FileExists(ExpandConstant('{sys}\nvcuda.dll'));
end;

function IsGpuEdition: Boolean;
var
  Forced: String;
begin
  Forced := Lowercase(ExpandConstant('{param:edition|}'));
  if Forced = 'gpu' then
    Result := True
  else if Forced = 'cpu' then
    Result := False
  else if EditionPage <> nil then
    Result := EditionPage.SelectedValueIndex = 0
  else
    Result := HasNvidiaDriver;
end;

function IsCpuEdition: Boolean;
begin
  Result := not IsGpuEdition;
end;

procedure InitializeWizard;
var
  Detected: String;
begin
  if HasNvidiaDriver then
    Detected := 'NVIDIA 그래픽 드라이버를 찾았습니다. GPU 가속 버전을 권장합니다.'
  else
    Detected := 'NVIDIA 그래픽 드라이버를 찾지 못했습니다. CPU 버전을 권장합니다.';
  EditionPage := CreateInputOptionPage(wpSelectDir,
    '설치할 버전 선택',
    '컴퓨터에 맞는 버전을 고르세요. 고른 버전만 인터넷에서 내려받습니다.',
    Detected + #13#10#13#10 +
    'GPU 가속 버전은 NVIDIA 그래픽카드로 AI 보정·피부 보정·일괄 처리를 훨씬 빠르게 합니다. ' +
    'CUDA Toolkit은 필요 없고 그래픽 드라이버만 있으면 됩니다.' + #13#10 +
    'CPU 버전은 그래픽카드와 관계없이 모든 PC에서 동작하지만 처리 속도가 느립니다. 기능은 같습니다.',
    True, False);
  EditionPage.Add('GPU 가속 버전 (NVIDIA 그래픽카드)  ·  다운로드 약 {#GpuDownloadText}');
  EditionPage.Add('CPU 버전 (모든 PC)  ·  다운로드 약 {#CpuDownloadText}');
  if IsGpuEdition then
    EditionPage.SelectedValueIndex := 0
  else
    EditionPage.SelectedValueIndex := 1;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  Edition: String;
begin
  if IsGpuEdition then
    Edition := 'GPU 가속 버전 (다운로드 약 {#GpuDownloadText})'
  else
    Edition := 'CPU 버전 (다운로드 약 {#CpuDownloadText})';
  Result := MemoDirInfo + NewLine + NewLine +
    '설치할 버전:' + NewLine + Space + Edition + NewLine + NewLine +
    '다음 단계에서 GitHub에서 파일을 내려받아 검증한 뒤 설치합니다. 인터넷 연결이 필요합니다.';
  if MemoTasksInfo <> '' then
    Result := Result + NewLine + NewLine + MemoTasksInfo;
end;

param(
  [string]$Root = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
  $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}

$svtDir = Join-Path $Root "third_party\svt"
New-Item -ItemType Directory -Path $svtDir -Force | Out-Null

$downloads = @(
  @{
    Name = "psyex-v3.0.2B-win64.zip"
    Url  = "https://github.com/BlueSwordM/svt-av1-psyex/releases/download/v3.0.2-B/windows_svt-av1-psyex-v3.0.2B_x86-64-v3.zip"
    Out  = Join-Path $svtDir "psyex-v3.0.2B-win64.zip"
    Extract = Join-Path $svtDir "psyex-v3.0.2B-win64"
  },
  @{
    Name = "ac-v2.1.2-win64.zip"
    Url  = "https://github.com/AmusementClub/SVT-AV1/releases/download/v2.1.2-AC/svtav1-win64-x86-64-clang.v2.1.2-AC.zip"
    Out  = Join-Path $svtDir "ac-v2.1.2-win64.zip"
    Extract = Join-Path $svtDir "ac-v2.1.2-win64"
  }
)

foreach ($d in $downloads) {
  if (!(Test-Path $d.Out)) {
    Write-Host "Downloading $($d.Name)..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $d.Url -OutFile $d.Out
  } else {
    Write-Host "Already downloaded: $($d.Name)" -ForegroundColor DarkGray
  }

  New-Item -ItemType Directory -Path $d.Extract -Force | Out-Null
  Expand-Archive -Path $d.Out -DestinationPath $d.Extract -Force
}

$psyExe = Join-Path $svtDir "psyex-v3.0.2B-win64\SvtAv1EncApp.exe"
$acExe = Join-Path $svtDir "ac-v2.1.2-win64\SvtAv1EncApp.exe"

if (!(Test-Path $psyExe)) { throw "Missing $psyExe" }
if (!(Test-Path $acExe)) { throw "Missing $acExe" }

Write-Host ""
Write-Host "Installed binaries:" -ForegroundColor Green
Write-Host "  $psyExe"
Write-Host "  $acExe"

Write-Host ""
Write-Host "Version checks:" -ForegroundColor Green
& $psyExe --version
& $acExe --version

Write-Host ""
Write-Host "Flag checks (PSYEX):" -ForegroundColor Green
& $psyExe --help | Select-String -Pattern "fgs-table|sharp-tx|hbd-mds|complex-hvs"


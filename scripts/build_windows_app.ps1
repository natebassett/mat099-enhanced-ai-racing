param(
    [switch]$SkipTests,
    [switch]$SkipClean
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "torcs-env\Scripts\python.exe"
$spec = Join-Path $projectRoot "packaging\windows\EnhancedAIRacing.spec"

if (-not (Test-Path -LiteralPath $python)) {
    throw "torcs-env was not found. Create the development environment before building."
}

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw (
        "PyInstaller is not installed. Run: " +
        ".\torcs-env\Scripts\python.exe -m pip install " +
        "-r packaging\windows\requirements-build.txt"
    )
}

if (-not $SkipTests) {
    & $python -m unittest tests.test_project_paths tests.test_gui_project_discovery
    if ($LASTEXITCODE -ne 0) {
        throw "Packaging preflight tests failed."
    }
}

Push-Location $projectRoot
try {
    $pyinstallerArguments = @(
        "-m",
        "PyInstaller",
        "--noconfirm"
    )
    if (-not $SkipClean) {
        $pyinstallerArguments += "--clean"
    }
    $pyinstallerArguments += @(
        "--distpath",
        (Join-Path $projectRoot "dist"),
        "--workpath",
        (Join-Path $projectRoot "build\windows"),
        $spec
    )
    & $python @pyinstallerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }
}
finally {
    Pop-Location
}

$executable = Join-Path $projectRoot "dist\Enhanced AI Racing\EnhancedAIRacing.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "Build completed without producing $executable"
}

$smokeLog = Join-Path $env:TEMP "enhanced-ai-racing-smoke-test.log"
Remove-Item -LiteralPath $smokeLog -Force -ErrorAction SilentlyContinue
$smokeTest = Start-Process `
    -FilePath $executable `
    -ArgumentList "--packaging-smoke-test" `
    -WorkingDirectory (Split-Path $executable) `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($smokeTest.ExitCode -ne 0) {
    $details = if (Test-Path -LiteralPath $smokeLog) {
        Get-Content -LiteralPath $smokeLog -Raw
    }
    else {
        "No diagnostic log was produced."
    }
    throw "Packaged runtime smoke test failed.`n$details"
}

Write-Host ""
Write-Host "Windows application created:" -ForegroundColor Green
Write-Host $executable
Write-Host "Packaged TORCS and policy smoke test passed." -ForegroundColor Green

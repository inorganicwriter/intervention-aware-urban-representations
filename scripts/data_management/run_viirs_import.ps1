param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,
    [string]$InputDir = $env:MIT_VIIRS_RAW
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($InputDir)) {
    throw "Provide -InputDir or set MIT_VIIRS_RAW."
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$runtimeDir = Join-Path $projectRoot "outputs/viirs_monthly/runtime"
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
Set-Location $projectRoot

$stdout = Join-Path $runtimeDir "import.stdout.log"
$stderr = Join-Path $runtimeDir "import.stderr.log"
$exitCodePath = Join-Path $runtimeDir "import.exitcode.txt"

& $PythonPath -u scripts/collection/process_viirs_monthly.py `
    --input-dir $InputDir `
    --require-complete `
    --output-dir "data\curated\viirs\monthly" `
    --audit-dir "outputs\viirs_monthly\partition_audits" `
    --compression-level 9 1>> $stdout 2>> $stderr

$LASTEXITCODE | Set-Content -LiteralPath $exitCodePath -Encoding ascii
exit $LASTEXITCODE

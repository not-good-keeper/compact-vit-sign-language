$python = Join-Path $PSScriptRoot "nndlvit\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "download_include_compressed.py"
$stdoutLog = Join-Path $PSScriptRoot "include_run.stdout.log"
$stderrLog = Join-Path $PSScriptRoot "include_run.stderr.log"

Start-Process `
    -FilePath $python `
    -ArgumentList "-u", $script `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

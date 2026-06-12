# scripts/offline_video.ps1
#
# PowerShell entry point for the offline multi-pass BVH extractor.
# Windows-compatible version of scripts/offline_video.sh.
#
# Usage:
#    .\scripts\offline_video.ps1 --from clip.mp4 --bvh out.bvh [--smoothing ...]
#    .\scripts\offline_video.ps1 clip.mp4 --bvh out.bvh --save vis.mp4

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location "$PSScriptRoot\.."

# Locate the input video
$FROM_SRC = ""
$OFFLINE_ARGS = @()
$SAVE_REQUESTED = $false
$SAVE_OUTPUT = ""

for ($i = 0; $i -lt $args.Count; $i++) {
    $a = $args[$i]
    if ($a -eq "--from") {
        if ($i + 1 -lt $args.Count) {
            $FROM_SRC = $args[$i + 1]
            $i++
        }
    }
    elseif ($a -eq "--save") {
        $SAVE_REQUESTED = $true
        if ($i + 1 -lt $args.Count -and -not $args[$i + 1].StartsWith("-")) {
            $SAVE_OUTPUT = $args[$i + 1]
            $i++
        }
    }
    else {
        # Positional fallback for the first argument if it doesn't start with -
        if ($i -eq 0 -and -not $a.StartsWith("-") -and $FROM_SRC -eq "") {
            $FROM_SRC = $a
        }
        else {
            $OFFLINE_ARGS += $a
        }
    }
}

if ([string]::IsNullOrEmpty($FROM_SRC)) {
    Write-Error "Usage: .\scripts\offline_video.ps1 --from VIDEO --bvh OUT.bvh [options] [--save VIS.mp4]"
    Write-Error "       (or positional: .\scripts\offline_video.ps1 VIDEO --bvh OUT.bvh ...)"
    exit 2
}

if (-not (Test-Path $FROM_SRC)) {
    Write-Error "Input video not found: $FROM_SRC"
    exit 2
}

# Locate binary
$BIN = ""
$BIN_PATHS = @(
    "build\Release\offline_sam_3dbody_render.exe",
    "build\Debug\offline_sam_3dbody_render.exe",
    "build\offline_sam_3dbody_render.exe"
)

foreach ($path in $BIN_PATHS) {
    if (Test-Path $path) {
        $BIN = $path
        break
    }
}

if ($BIN -eq "") {
    Write-Error "Could not find offline_sam_3dbody_render.exe in build directories."
    exit 1
}

$FIXED_FLAGS = @(
    "--onnx-dir", ".\onnx",
    "--gguf", ".\onnx\pipeline.gguf",
    "--yolo", ".\onnx\yolo.onnx"
)

Write-Host "Running: $BIN --from $FROM_SRC $($FIXED_FLAGS -join ' ') $($OFFLINE_ARGS -join ' ')"
& $BIN --from $FROM_SRC @FIXED_FLAGS @OFFLINE_ARGS
$OFFLINE_EXIT = $LASTEXITCODE

if ($OFFLINE_EXIT -eq $null) {
    # If binary is missing or can't be executed
    $OFFLINE_EXIT = 1
}

if ($OFFLINE_EXIT -ne 0) {
    Write-Error "Offline binary exited with code $OFFLINE_EXIT - skipping rendered-mp4 step."
    exit $OFFLINE_EXIT
}

if ($SAVE_REQUESTED) {
    Write-Host ""
    Write-Host ("-" * 66)
    Write-Host " offline BVH done - now rendering visualisation mp4 via video.ps1"
    Write-Host ("-" * 66)

    if (-not [string]::IsNullOrEmpty($SAVE_OUTPUT)) {
        & "$PSScriptRoot\video.ps1" --from "$FROM_SRC" --save "$SAVE_OUTPUT"
    } else {
        & "$PSScriptRoot\video.ps1" --from "$FROM_SRC" --save
    }
}

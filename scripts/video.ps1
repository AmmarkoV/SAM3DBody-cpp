# scripts/video.ps1
#
# PowerShell entry point for the live visualization renderer.
# Windows-compatible version of scripts/video.sh.
#
# Usage:
#    .\scripts\video.ps1 --from clip.mp4 [--save vis.mp4]

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location "$PSScriptRoot\.."

# Parse arguments
$SAVE_REQUESTED = $false
$SAVE_OUTPUT = ""
$FROM_SRC = ""
$FORWARD_ARGS = @()

for ($i = 0; $i -lt $args.Count; $i++) {
    $a = $args[$i]
    if ($a -eq "--save") {
        $SAVE_REQUESTED = $true
        if ($i + 1 -lt $args.Count -and -not $args[$i + 1].StartsWith("-")) {
            $SAVE_OUTPUT = $args[$i + 1]
            $i++
        }
    }
    elseif ($a -eq "--from") {
        if ($i + 1 -lt $args.Count) {
            $FROM_SRC = $args[$i + 1]
            $i++
        }
    }
    else {
        # Positional fallback for the first argument
        if ($i -eq 0 -and -not $a.StartsWith("-") -and $FROM_SRC -eq "") {
            $FROM_SRC = $a
        }
        else {
            $FORWARD_ARGS += $a
        }
    }
}

if ([string]::IsNullOrEmpty($FROM_SRC)) {
    Write-Host "Usage: .\scripts\video.ps1 --from VIDEO [--save VIS.mp4] [options]"
    exit 2
}

# Locate binary
$BIN = ""
$BIN_PATHS = @(
    "build\Release\fast_sam_3dbody_render.exe",
    "build\Debug\fast_sam_3dbody_render.exe",
    "build\fast_sam_3dbody_render.exe"
)

foreach ($path in $BIN_PATHS) {
    if (Test-Path $path) {
        $BIN = $path
        break
    }
}

if ($BIN -eq "") {
    Write-Error "Could not find fast_sam_3dbody_render.exe in build directories. Build the project first."
    exit 1
}

$FIXED_FLAGS = @(
    "--onnx-dir", ".\onnx",
    "--gguf", ".\onnx\pipeline.gguf",
    "--yolo", ".\onnx\yolo.onnx",
    "--mesh", ".\body_mesh.tri",
    "--lbs", "onnx\body_model.lbs"
)

if (-not $SAVE_REQUESTED) {
    # Normal live mode
    & $BIN --from $FROM_SRC @FIXED_FLAGS @FORWARD_ARGS
    exit $LASTEXITCODE
}

# Save-to-file mode
if ([string]::IsNullOrEmpty($SAVE_OUTPUT)) {
    if (Test-Path $FROM_SRC) {
        $base = Split-Path -Leaf $FROM_SRC
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($base)
        $SAVE_OUTPUT = "${stem}_rendered.mp4"
    } else {
        $SAVE_OUTPUT = "livelastRun3DHiRes.mp4"
    }
    Write-Host "Output: $SAVE_OUTPUT"
}

# Create a temporary directory for the JPEG frames
$TMP_DIR = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "fsb_frames_" + [System.Guid]::NewGuid().ToString().Substring(0,8))
New-Item -ItemType Directory -Path $TMP_DIR | Out-Null
$FRAME_PREFIX = [System.IO.Path]::Combine($TMP_DIR, "colorFrame_0_")

Write-Host "Rendering frames to $TMP_DIR ..."
& $BIN --from $FROM_SRC @FIXED_FLAGS @FORWARD_ARGS --headless --save-frames "$FRAME_PREFIX"
$RENDER_EXIT = $LASTEXITCODE

$ACTUAL_FRAMES = (Get-ChildItem "${FRAME_PREFIX}*.jpg" -ErrorAction SilentlyContinue).Count
if ($ACTUAL_FRAMES -eq 0) {
    Write-Error "Renderer did not produce any frames. Exit code: $RENDER_EXIT"
    Remove-Item -Recurse -Force $TMP_DIR
    exit $RENDER_EXIT
}

# Probe FPS from source
$FPS = 30
if (Test-Path $FROM_SRC) {
    $ffprobe_out = ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $FROM_SRC 2>$null
    if ($ffprobe_out -match "(\d+)/(\d+)") {
        $FPS = [double]$Matches[1] / [double]$Matches[2]
    } elseif ($ffprobe_out -match "^\d+(\.\d+)?$") {
        $FPS = [double]$ffprobe_out
    }
}
Write-Host "Source framerate: $FPS fps"

# Probe Size from first frame
$SIZE_ARG = @()
$FIRST_FRAME = Get-ChildItem "${FRAME_PREFIX}*.jpg" | Sort-Object Name | Select-Object -First 1
if ($null -ne $FIRST_FRAME) {
    $ffprobe_out = ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $FIRST_FRAME.FullName 2>$null
    if ($ffprobe_out -match "(\d+),(\d+)") {
        $FW = [int]$Matches[1]
        $FH = [int]$Matches[2]
        # yuv420p requires even dimensions
        $FW = [int]($FW / 2) * 2
        $FH = [int]($FH / 2) * 2
        $SIZE_ARG = @("-s", "${FW}x${FH}")
        Write-Host "Render size: ${FW}x${FH}"
    }
}

# Check for audio
$AUDIO_ARGS = @()
if (Test-Path $FROM_SRC) {
    $audio_idx = ffprobe -v error -select_streams a:0 -show_entries stream=index -of csv=p=0 $FROM_SRC 2>$null
    if (-not [string]::IsNullOrEmpty($audio_idx)) {
        Write-Host "Copying audio from: $FROM_SRC"
        $AUDIO_ARGS = @("-i", $FROM_SRC, "-map", "0:v", "-map", "1:a", "-c:a", "copy")
    }
}

# Encode
Write-Host "Encoding to $SAVE_OUTPUT ..."
ffmpeg -framerate $FPS -i "${FRAME_PREFIX}%05d.jpg" @AUDIO_ARGS @SIZE_ARG -y -r $FPS -pix_fmt yuv420p -threads 8 $SAVE_OUTPUT
$FFMPEG_EXIT = $LASTEXITCODE

# Cleanup
Remove-Item -Recurse -Force $TMP_DIR

exit $FFMPEG_EXIT

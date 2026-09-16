#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OpenCVDir,
    [string]$OnnxRuntimeDir,
    [string]$GLEWRoot,
    [string]$BuildDir,
    [string]$Generator = 'Visual Studio 17 2022',
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')][string]$Configuration = 'Release',
    [string]$CudaArchitectures,
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [switch]$Gpu,
    [switch]$Headless,
    [switch]$TestOpenGL,
    [switch]$SkipTests,
    [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDir) { $BuildDir = Join-Path $repoRoot 'build\windows' }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'Install CMake 3.21+ and add it to PATH.' }
$opencvPath = (Resolve-Path -LiteralPath $OpenCVDir).Path
$configureArgs = @('-S', $repoRoot, '-B', $BuildDir, '-G', $Generator,
    '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
    "-DOpenCV_DIR=$opencvPath", "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DSAM3D_ONNX_CUDA=$(@{ $true='ON'; $false='OFF' }[[bool]$Gpu])",
    "-DSAM3D_BUILD_RENDERER=$(@{ $true='OFF'; $false='ON' }[[bool]$Headless])",
    "-DSAM3D_TEST_OPENGL=$(@{ $true='ON'; $false='OFF' }[[bool]$TestOpenGL])")
if ($Generator -like 'Visual Studio*') { $configureArgs += @('-A', 'x64') }
if ($OnnxRuntimeDir) { $configureArgs += "-DONNX_RUNTIME_DIR=$((Resolve-Path -LiteralPath $OnnxRuntimeDir).Path)" }
if ($GLEWRoot) { $configureArgs += "-DGLEW_ROOT=$((Resolve-Path -LiteralPath $GLEWRoot).Path)" }
if ($CudaArchitectures) { $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" }
$configureArgs += $CMakeArgs
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)." }
& cmake --build $BuildDir --config $Configuration --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Windows build failed ($LASTEXITCODE)." }
if (-not $SkipTests) {
    & ctest --test-dir $BuildDir -C $Configuration --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "Windows regression tests failed ($LASTEXITCODE)." }
}
Write-Host "Windows $Configuration build is ready in $BuildDir. Models are downloaded separately with tools/fetch_model.ps1."

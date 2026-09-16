#requires -Version 5.1
# Offline checks only: no curl invocation and no model downloads.
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$scriptPath = Join-Path $repository 'tools\fetch_model.ps1'
$testRoot = Join-Path $repository ('build\fetch-model-tests-' + [Guid]::NewGuid().ToString('N'))
$previousAutoFetch = $env:SAM3D_AUTO_FETCH
$previousRevision = $env:SAM3D_HF_REVISION
$checks = 0

function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw "FAIL: $Message" }
    $script:checks++
}

function Assert-Fails([scriptblock] $Action, [string] $Pattern) {
    $caught = $null
    try { & $Action | Out-Null } catch { $caught = $_.Exception.Message }
    Assert-True ($null -ne $caught -and $caught -match $Pattern) "Expected error: $Pattern; received: $caught"
}

try {
    $env:SAM3D_AUTO_FETCH = '0'
    $env:SAM3D_HF_REVISION = 'test-revision'
    $missing = Join-Path $testRoot 'not-created'
    $models = @(& $scriptPath -Profile cpu,trt -OnnxDir $missing -List)
    Assert-True (@($models | Where-Object Name -eq 'pipeline.gguf').Count -eq 1) 'Shared profile is implicit'
    Assert-True (@($models | Where-Object Name -eq 'decoder_fp16.onnx').Count -eq 1) 'CPU/TRT duplicate is fetched once'
    Assert-True (-not (Test-Path -LiteralPath $testRoot)) '-List must not create directories'
    $all = @(& $scriptPath -Profile all -OnnxDir $missing -List)
    Assert-True (@($all | Where-Object { $_.Profiles -contains 'refined' }).Count -eq 0) 'all excludes refined'
    Assert-True (@($all | Where-Object { $_.Profiles -contains 'libreyolo' }).Count -eq 0) 'all matches Bash base profiles'
    $optional = @(& $scriptPath -Profile all,refined,libreyolo -OnnxDir $missing -List)
    Assert-True (@($optional | Where-Object Name -eq 'pipeline_refined.gguf').Count -eq 1) 'refined is available explicitly'
    Assert-True (@($optional | Where-Object Name -eq 'libreyolo9.onnx').Count -eq 1) 'libreyolo is available explicitly'
    Assert-Fails { & $scriptPath -Profile unknown -OnnxDir $missing -List } 'Unknown profile'
    Assert-Fails { & $scriptPath -Profile shared -OnnxDir $missing -Revision "bad`nrevision" -List } 'Revision'
    Assert-Fails { & $scriptPath -Profile shared -OnnxDir $missing -Yes } 'SAM3D_AUTO_FETCH=0'
    Assert-True (-not (Test-Path -LiteralPath $testRoot)) 'Disabled downloads leave directories untouched'

    # A tiny fixture exercises SHA256 checking without creating/downloading models.
    $fixtureTools = Join-Path $testRoot 'tools'
    $fixtureModels = Join-Path $testRoot 'models'
    [IO.Directory]::CreateDirectory($fixtureTools) | Out-Null
    [IO.Directory]::CreateDirectory($fixtureModels) | Out-Null
    $fixtureScript = Join-Path $fixtureTools 'fetch_model.ps1'
    Copy-Item -LiteralPath $scriptPath -Destination $fixtureScript
    $fixtureFile = Join-Path $fixtureModels 'fixture.bin'
    [IO.File]::WriteAllBytes($fixtureFile, [byte[]]@(1, 2, 3, 4))
    $hash = (Get-FileHash -LiteralPath $fixtureFile -Algorithm SHA256).Hash.ToLowerInvariant()
    $fixtureManifest = Join-Path $fixtureTools 'fetch_model.sh'
    $manifest = "MANIFEST=(`n" + '  "shared|fixture.bin|4|' + $hash + '"' + "`n)`n"
    [IO.File]::WriteAllText($fixtureManifest, $manifest)
    $verified = @(& $fixtureScript -Profile shared -OnnxDir $fixtureModels -List)
    Assert-True ($verified[0].Status -eq 'Verified') 'Correct existing file is verified'
    $forced = @(& $fixtureScript -Profile shared -OnnxDir $fixtureModels -Force -List)
    Assert-True ($forced[0].Status -eq 'Forced') '-Force schedules a verified existing file'
    [IO.File]::WriteAllBytes($fixtureFile, [byte[]]@(4, 3, 2, 1))
    $invalid = @(& $fixtureScript -Profile shared -OnnxDir $fixtureModels -List)
    Assert-True ($invalid[0].Status -eq 'Invalid') 'Same-size SHA256 corruption is not accepted'
    Assert-Fails { & $fixtureScript -Profile shared -OnnxDir $fixtureModels -Yes } 'SAM3D_AUTO_FETCH=0'
    Assert-True ([IO.File]::ReadAllBytes($fixtureFile)[0] -eq 4) 'Disabled fetch preserves an invalid existing file'
    [IO.File]::WriteAllText($fixtureManifest, $manifest.Replace('fixture.bin', '..\escape.bin'))
    Assert-Fails { & $fixtureScript -Profile shared -OnnxDir $fixtureModels -List } 'Unsafe model filename'
    Write-Host "PASS: $checks offline fetch_model.ps1 checks."
} finally {
    $env:SAM3D_AUTO_FETCH = $previousAutoFetch
    $env:SAM3D_HF_REVISION = $previousRevision
    if (Test-Path -LiteralPath $testRoot) {
        $resolvedTestRoot = (Resolve-Path -LiteralPath $testRoot).Path
        $expectedPrefix = Join-Path $repository 'build\fetch-model-tests-'
        if (-not $resolvedTestRoot.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing to clean a test path outside the expected build directory.'
        }
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}

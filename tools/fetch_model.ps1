#requires -Version 5.1
<#
.SYNOPSIS
Fetch verified SAM3DBody model files with native Windows PowerShell and curl.exe.
.DESCRIPTION
Reads the authoritative MANIFEST from fetch_model.sh in the same directory.
Shared files are implicit. The all profile selects cpu, cuda and trt; refined
and libreyolo remain opt-in. Existing files are checked by size and SHA256.
Downloads use .partial files; only verified files replace the final model name.
SAM3D_AUTO_FETCH=0 refuses downloads even with -Yes. A value of 1 skips prompting.
HF_TOKEN, when present, is passed to curl through stdin rather than process args.
.EXAMPLE
.\tools\fetch_model.ps1 -Profile cpu -List
.EXAMPLE
.\tools\fetch_model.ps1 -Profile cuda,refined -OnnxDir .\onnx -Yes
.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\fetch_model.ps1 -Profile cpu -Yes
#>
[CmdletBinding()]
param(
    [string[]] $Profile = @('cuda'),
    [string] $OnnxDir,
    [string] $Revision = $env:SAM3D_HF_REVISION,
    [switch] $List,
    [switch] $Force,
    [switch] $Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $PSBoundParameters.ContainsKey('OnnxDir')) {
    $OnnxDir = Join-Path $PSScriptRoot '..\onnx'
}

$allowedProfiles = @('shared', 'cpu', 'cuda', 'trt', 'refined', 'libreyolo')
$selectedProfiles = @('shared')
foreach ($argument in $Profile) {
    # Also accept the comma-separated string passed by powershell.exe -File.
    foreach ($item in $argument.Split(',')) {
        $item = $item.Trim().ToLowerInvariant()
        if ($item -eq 'all') {
            $selectedProfiles += @('cpu', 'cuda', 'trt')
        } elseif ($allowedProfiles -contains $item) {
            $selectedProfiles += $item
        } else {
            throw "Unknown profile '$item'. Use shared, cpu, cuda, trt, refined, libreyolo or all."
        }
    }
}
$selectedProfiles = @($selectedProfiles | Select-Object -Unique)
if ([string]::IsNullOrWhiteSpace($Revision)) { $Revision = 'main' }
if ($Revision -match '[\x00-\x20\x7f]') { throw 'Revision must not contain whitespace or control characters.' }
if ([string]::IsNullOrWhiteSpace($OnnxDir)) { throw 'OnnxDir must not be empty.' }
$destinationRoot = [IO.Path]::GetFullPath(
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OnnxDir))
if ((Test-Path -LiteralPath $destinationRoot) -and
    -not (Test-Path -LiteralPath $destinationRoot -PathType Container)) {
    throw 'OnnxDir must name a directory, not an existing file.'
}

function Get-ModelPath([string] $Name) {
    # Manifest entries are filenames, never commands or paths.
    if ($Name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or $Name.EndsWith('.')) {
        throw "Unsafe model filename in manifest: '$Name'."
    }
    $path = [IO.Path]::GetFullPath([IO.Path]::Combine($destinationRoot, $Name))
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($path).TrimEnd([IO.Path]::DirectorySeparatorChar),
                             $destinationRoot.TrimEnd([IO.Path]::DirectorySeparatorChar),
                             [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Model destination escapes the selected OnnxDir.'
    }
    return $path
}

function Test-ModelFile([string] $Path, [long] $Size, [string] $Sha256) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $Size) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Sha256
}

$manifestPath = Join-Path $PSScriptRoot 'fetch_model.sh'
$inManifest = $false
$foundManifest = $false
$closedManifest = $false
$models = [ordered]@{}
foreach ($line in Get-Content -LiteralPath $manifestPath -Encoding UTF8) {
    $entry = $line.Trim()
    if (-not $inManifest) {
        if ($entry -eq 'MANIFEST=(') { $inManifest = $true; $foundManifest = $true }
        continue
    }
    if ($entry -eq ')') { $closedManifest = $true; break }
    if (-not $entry -or $entry.StartsWith('#')) { continue }
    if ($entry -notmatch '^"([^"|]+)\|([^"|]+)\|([0-9]+)\|([0-9a-fA-F]{64})"$') {
        throw 'Malformed entry in fetch_model.sh MANIFEST.'
    }
    $entryProfile = $Matches[1]
    $name = $Matches[2]
    $size = [long]$Matches[3]
    $sha = $Matches[4].ToLowerInvariant()
    if ($allowedProfiles -notcontains $entryProfile -or $size -le 0) {
        throw 'Invalid profile or size in fetch_model.sh MANIFEST.'
    }
    $path = Get-ModelPath $name
    if ($models.Contains($name)) {
        if ($models[$name].SizeBytes -ne $size -or $models[$name].SHA256 -ne $sha) {
            throw "Conflicting manifest entries for '$name'."
        }
        $models[$name].Profiles += $entryProfile
    } else {
        $models[$name] = [pscustomobject]@{
            Name = $name
            Profiles = @($entryProfile)
            SizeBytes = $size
            SHA256 = $sha
            Path = $path
            Status = 'Missing'
        }
    }
}
if (-not $foundManifest -or -not $closedManifest -or $models.Count -eq 0) {
    throw 'Cannot find a complete MANIFEST in fetch_model.sh.'
}

$selected = @($models.Values | Where-Object {
    $matchingProfiles = @($_.Profiles | Where-Object { $selectedProfiles -contains $_ })
    $matchingProfiles.Count -gt 0
})
foreach ($model in $selected) {
    if (Test-Path -LiteralPath $model.Path -PathType Container) {
        throw "A directory occupies the model filename '$($model.Name)'."
    }
    if ($Force) {
        $model.Status = 'Forced'
    } elseif (Test-ModelFile $model.Path $model.SizeBytes $model.SHA256) {
        $model.Status = 'Verified'
    } elseif (Test-Path -LiteralPath $model.Path) {
        $model.Status = 'Invalid'
    }
}
$pending = @($selected | Where-Object { $_.Status -ne 'Verified' })
$totalBytes = [long]0
foreach ($model in $pending) { $totalBytes += $model.SizeBytes }
Write-Host ('fetch_model.ps1: {0} selected, {1} to fetch ({2:N2} GiB), revision {3}' -f
    $selected.Count, $pending.Count, ($totalBytes / 1GB), $Revision)
if ($List) {
    $selected | Select-Object Name, Profiles, SizeBytes, SHA256, Status, Path
    return
}
if ($pending.Count -eq 0) {
    Write-Host "All selected model files verified in $destinationRoot."
    return
}
foreach ($model in $pending) {
    Write-Host ('  {0,-34} {1,12:N0} bytes  {2}' -f $model.Name, $model.SizeBytes, $model.Status)
}
if ($env:SAM3D_AUTO_FETCH -eq '0') {
    throw 'SAM3D_AUTO_FETCH=0: downloading is disabled. No model files were changed.'
}
if (-not $Yes -and $env:SAM3D_AUTO_FETCH -ne '1') {
    if (-not [Environment]::UserInteractive -or [Console]::IsInputRedirected) {
        throw 'Non-interactive download requires -Yes or SAM3D_AUTO_FETCH=1.'
    }
    $answer = Read-Host "Download $($pending.Count) model files to $destinationRoot ? [y/N]"
    if ($answer -notmatch '^(?i)y(es)?$') { throw 'Download cancelled.' }
}

$curlCommand = Get-Command curl.exe -CommandType Application -ErrorAction Stop
$curlConfig = ''
if ($env:HF_TOKEN) {
    if ($env:HF_TOKEN -match '[\x00-\x1f\x7f]') {
        throw 'HF_TOKEN contains invalid control characters.'
    }
    $escapedToken = $env:HF_TOKEN.Replace('\', '\\').Replace('"', '\"')
    $curlConfig = 'header = "Authorization: Bearer ' + $escapedToken + '"'
}
[IO.Directory]::CreateDirectory($destinationRoot) | Out-Null
$encodedRevision = [Uri]::EscapeDataString($Revision)
$index = 0
foreach ($model in $pending) {
    $index++
    $destination = Get-ModelPath $model.Name
    $partial = Get-ModelPath ($model.Name + '.partial')
    if (Test-Path -LiteralPath $partial -PathType Container) {
        throw "A directory occupies the partial filename for '$($model.Name)'."
    }
    $have = [long]0
    if (Test-Path -LiteralPath $partial -PathType Leaf) {
        $have = (Get-Item -LiteralPath $partial).Length
    }
    Write-Host "[$index/$($pending.Count)] $($model.Name)"
    if (-not $Force -and $have -gt $model.SizeBytes) {
        throw "Partial file for '$($model.Name)' exceeds the expected size. Rerun with -Force to restart it."
    }
    if ($Force -or $have -lt $model.SizeBytes) {
        $url = 'https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/' +
            $encodedRevision + '/' + [Uri]::EscapeDataString($model.Name) + '?download=true'
        # Disable user curl config, including any trace option which could log auth.
        $curlArguments = @('--disable', '--config', '-', '--location', '--fail',
            '--show-error', '--progress-bar', '--retry', '3', '--retry-delay', '2',
            '--connect-timeout', '30', '--proto', '=https', '--proto-redir', '=https',
            '--output', $partial, '--url', $url)
        if (-not $Force -and $have -gt 0) { $curlArguments += @('--continue-at', '-') }
        $curlConfig | & $curlCommand.Source @curlArguments
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed for '$($model.Name)' (exit $LASTEXITCODE). Partial kept for retry."
        }
    }
    if (-not (Test-ModelFile $partial $model.SizeBytes $model.SHA256)) {
        throw "Size or SHA256 verification failed for '$($model.Name)'. Partial kept; rerun with -Force to restart it."
    }
    # Both absolute paths were checked to be direct children of the selected dir.
    # Keep an existing model intact until its replacement has passed verification.
    Move-Item -LiteralPath $partial -Destination $destination -Force
    Write-Host "  Verified $($model.Name)."
}
Write-Host "Done: $($pending.Count) verified model files in $destinationRoot."

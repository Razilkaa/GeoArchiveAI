param(
    [Parameter(Mandatory = $true)]
    [string]$SourceFile,

    [string]$Inbox = "C:\FINAM\Conference\reports_inbox"
)

$ErrorActionPreference = "Stop"

$source = Get-Item -LiteralPath $SourceFile
if ($source.PSIsContainer) {
    throw "SourceFile must point to one archive file."
}

$staging = Join-Path (Split-Path -Parent $Inbox) "transfer_staging"
New-Item -ItemType Directory -Path $staging -Force | Out-Null
New-Item -ItemType Directory -Path $Inbox -Force | Out-Null

$stagedFile = Join-Path $staging $source.Name
$inboxFile = Join-Path $Inbox $source.Name

if (Test-Path -LiteralPath $inboxFile) {
    throw "Destination already exists: $inboxFile"
}

Write-Host "Copying $($source.FullName)"
Write-Host "Staging: $stagedFile"

& robocopy $source.DirectoryName $staging $source.Name /Z /J /R:5 /W:3 /COPY:DAT /DCOPY:T /NP
$robocopyCode = $LASTEXITCODE
if ($robocopyCode -ge 8) {
    throw "robocopy failed with exit code $robocopyCode"
}

$copied = Get-Item -LiteralPath $stagedFile
if ($copied.Length -ne $source.Length) {
    throw "Size mismatch: source=$($source.Length), copied=$($copied.Length)"
}

Move-Item -LiteralPath $stagedFile -Destination $inboxFile
Write-Host "Ready: $inboxFile ($([math]::Round($copied.Length / 1GB, 2)) GB)"

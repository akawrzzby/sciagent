"""Sync helper: copy the local working tree to the GPU host.

Usage (from Windows PowerShell)::

    pwsh scripts/sync.ps1 -Host 179

The script mirrors ``backend/`` and ``scripts/`` to ``~/sciagent`` on the target
host and prints the resulting revision so the developer can confirm what is
running.
"""

param(
    [string]$Target = "dsh-10-184-17-179",
    [string]$RemoteRoot = "sciagent"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $repoRoot "backend"
$scripts = Join-Path $repoRoot "scripts"

Write-Host "==> syncing backend -> ${Target}:~/${RemoteRoot}/backend"
scp -o BatchMode=yes -r "$backend\*" "${Target}:~/${RemoteRoot}/backend/"

Write-Host "==> syncing scripts -> ${Target}:~/${RemoteRoot}/scripts"
scp -o BatchMode=yes -r "$scripts\*" "${Target}:~/${RemoteRoot}/scripts/"

Write-Host "==> remote tree"
ssh -o BatchMode=yes $Target "find ~/${RemoteRoot} -maxdepth 2 -type d | head -30"
Write-Host "done."

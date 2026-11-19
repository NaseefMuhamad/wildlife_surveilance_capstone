# Run from an elevated PowerShell. The temporary rule covers native DLL traffic
# originating from the configured evaluation Python process.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell as Administrator to apply the temporary evaluation firewall rule.'
}

$projectPath = Split-Path -Parent $PSScriptRoot
$interpreterConfig = Join-Path $projectPath '.xpu-python'
$interpreterPath = (Get-Content -LiteralPath $interpreterConfig -Raw).Trim()
$interpreterPath = (Resolve-Path -LiteralPath $interpreterPath).Path
$comparisonScript = Join-Path $PSScriptRoot 'compare_checkpoints.py'
$ruleName = 'ReserveWatch-Evaluation-' + [guid]::NewGuid().ToString('N')
$ruleCreated = $false
$comparisonExitCode = 1

Push-Location -LiteralPath $projectPath
try {
    $inactiveProfiles = @(Get-NetFirewallProfile | Where-Object { -not $_.Enabled })
    if ($inactiveProfiles.Count -gt 0) {
        throw 'All Windows Firewall profiles must be enabled for this evaluation. No profiles were changed.'
    }
    $ruleOptions = @{
        Name = $ruleName
        DisplayName = $ruleName
        Direction = 'Outbound'
        Action = 'Block'
        Enabled = 'True'
        Profile = 'Any'
        Program = $interpreterPath
    }
    New-NetFirewallRule @ruleOptions | Out-Null
    $ruleCreated = $true

    $rule = Get-NetFirewallRule -Name $ruleName
    $filter = $rule | Get-NetFirewallApplicationFilter
    $valid = $rule.Enabled -eq 'True' -and $rule.Direction -eq 'Outbound' -and $rule.Action -eq 'Block'
    if (-not $valid -or $filter.Program -ne $interpreterPath) {
        throw 'The evaluation firewall rule could not be verified.'
    }

    Write-Output 'Verified outbound block for the evaluation interpreter.'
    & $interpreterPath -E $comparisonScript
    $comparisonExitCode = $LASTEXITCODE
}
finally {
    if ($ruleCreated) {
        Remove-NetFirewallRule -Name $ruleName
        Write-Output 'Removed the temporary evaluation firewall rule.'
    }
    Pop-Location
}

exit $comparisonExitCode

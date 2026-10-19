<#
.SYNOPSIS
    Recosys WiFi-Triggered Daily Sync Runner
.DESCRIPTION
    Checks if connected to the target WiFi network (default: M-5G), verifies that
    the sync has not already run today, and executes the Recosys pipeline and Supabase sync.
#>

param (
    [string]$TargetSSID = "M-5G",
    [switch]$Force
)

$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$SyncScript = Join-Path $ProjectDir "scripts\nightly_sync.py"
$LogsDir = Join-Path $ProjectDir "logs"
$LastSyncFile = Join-Path $LogsDir ".last_sync_date"
$RunnerLog = Join-Path $LogsDir "wifi_sync_runner.log"

if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

function Log-Message([string]$msg) {
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp] $msg"
    Write-Host $line
    Add-Content -Path $RunnerLog -Value $line
}

Log-Message "=== WiFi Sync Runner Invoked ==="

# 1. Detect current WiFi SSID
$currentSSID = $null
$wifiLine = (netsh wlan show interfaces) | Where-Object { $_ -match '^\s*SSID\s*:\s*(.+)$' } | Select-Object -First 1
if ($wifiLine -match '^\s*SSID\s*:\s*(.+)$') {
    $currentSSID = $matches[1].Trim()
}

Log-Message "Current connected SSID: '$currentSSID' (Target: '$TargetSSID')"

if ($currentSSID -ne $TargetSSID -and -not $Force) {
    Log-Message "Not connected to target WiFi '$TargetSSID'. Skipping sync."
    exit 0
}

# 2. Check once-per-day enforcement
$today = (Get-Date).ToString("yyyy-MM-dd")
if (Test-Path $LastSyncFile) {
    $lastSync = (Get-Content $LastSyncFile -Raw).Trim()
    if ($lastSync -eq $today -and -not $Force) {
        Log-Message "Already synced today ($today). Enforcing once-per-day policy. Exiting."
        exit 0
    }
}

# 3. Check connectivity to database server 192.168.1.3
Log-Message "Target WiFi matched and not synced today. Verifying database reachability (192.168.1.3)..."
$pingOk = Test-Connection -ComputerName "192.168.1.3" -Count 1 -Quiet
if (-not $pingOk) {
    Log-Message "Warning: 192.168.1.3 did not respond to ICMP ping. Testing TCP port 1433..."
    $tcp = Test-NetConnection -ComputerName "192.168.1.3" -Port 1433 -WarningAction SilentlyContinue
    if (-not $tcp.TcpTestSucceeded) {
        Log-Message "Database server 192.168.1.3 is not reachable. Aborting sync."
        exit 1
    }
}

# 4. Run the sync script
Log-Message "Starting Recosys sync execution..."
$env:PYTHONIOENCODING = "utf-8"
$process = Start-Process -FilePath $PythonExe -ArgumentList "`"$SyncScript`"" -WorkingDirectory $ProjectDir -NoNewWindow -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Set-Content -Path $LastSyncFile -Value $today -Force
    Log-Message "Sync succeeded! Recorded today ($today) as last sync date."
    
    # Windows notification
    try {
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
        $textNodes = $template.GetElementsByTagName("text")
        $textNodes.Item(0).AppendChild($template.CreateTextNode("Recosys Daily Sync")) | Out-Null
        $textNodes.Item(1).AppendChild($template.CreateTextNode("Successfully synced on $TargetSSID for $today.")) | Out-Null
        $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Recosys").Show($toast)
    } catch {}
    exit 0
} else {
    Log-Message "Sync failed with exit code $($process.ExitCode). Today will NOT be marked as synced so it can retry."
    exit $process.ExitCode
}

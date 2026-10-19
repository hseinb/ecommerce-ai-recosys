<#
.SYNOPSIS
    Registers the Recosys WiFi-Triggered Task in Windows Task Scheduler.

.DESCRIPTION
    Creates or updates a scheduled task named 'RecosysWiFiSync' that:
    1. Triggers whenever the user logs in to this device.
    2. Runs an automated background check every 15 minutes.
    3. Executes 'scripts/wifi_sync_runner.ps1', which:
       - Checks if connected to the target WiFi ('M-5G').
       - Enforces a strict ONCE-PER-DAY sync policy (tracked in logs/.last_sync_date).
       - If not on 'M-5G' or already synced today, exits in ~50ms silently.
       - If on 'M-5G' and hasn't synced today, runs the full sync and updates Supabase.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_wifi_task.ps1 -TargetSSID "M-5G"
#>

param (
    [string]$TargetSSID = "M-5G"
)

$TaskName = "RecosysWiFiSync"
$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName
$RunnerScript = Join-Path $ProjectDir "scripts\wifi_sync_runner.ps1"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Registering Recosys WiFi Daily Sync Task in Windows" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Task Name:      $TaskName"
Write-Host "Project Dir:    $ProjectDir"
Write-Host "Runner Script:  $RunnerScript"
Write-Host "Target WiFi:    $TargetSSID"
Write-Host "Execution Rule: Trigger on $TargetSSID once per calendar day"

if (-not (Test-Path $RunnerScript)) {
    Write-Error "Runner script not found at $RunnerScript"
    exit 1
}

# Ensure logs directory exists
$LogsDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Action: runs powershell with hidden window
$ActionArg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$RunnerScript`" -TargetSSID `"$TargetSSID`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionArg -WorkingDirectory $ProjectDir

# Trigger: Repeating check every 15 minutes (active continuously for the current user)
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)

# Settings: Allow running on battery, start when available, don't stop if on battery
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Unregister existing task if present
try {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Unregistering previous scheduled task '$TaskName'..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
} catch {}

# Register scheduled task
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Runs Recosys sync when connected to $TargetSSID once per day" -Force
    
    Write-Host "`n[SUCCESS] Task '$TaskName' registered successfully in Windows Task Scheduler!" -ForegroundColor Green
    Write-Host "Active Behavior:" -ForegroundColor White
    Write-Host "  * Connected to '$TargetSSID': Runs full sync once per day." -ForegroundColor Yellow
    Write-Host "  * Disconnected or another WiFi: Silently exits in ~50ms." -ForegroundColor Yellow
    Write-Host "  * Already synced today: Skips execution automatically." -ForegroundColor Green
    Write-Host "`nTo manually trigger or verify now:" -ForegroundColor White
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\wifi_sync_runner.ps1" -ForegroundColor Yellow
} catch {
    Write-Error "Failed to register scheduled task: $_"
    exit 1
}

<#
.SYNOPSIS
    Registers the Recosys Nightly Sync task in Windows Task Scheduler.

.DESCRIPTION
    Creates or updates a scheduled task named 'RecosysNightlySync' that runs every night at 2:00 AM.
    The task executes 'scripts/nightly_sync.py' using the project's dedicated virtual environment.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_nightly_task.ps1
#>

$TaskName = "RecosysNightlySync"
$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectDir "scripts\nightly_sync.py"
$LogPath = Join-Path $ProjectDir "logs\nightly_task_scheduler.log"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Registering Recosys Nightly Sync in Windows Task Scheduler" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Task Name:      $TaskName"
Write-Host "Project Dir:    $ProjectDir"
Write-Host "Python Path:    $PythonExe"
Write-Host "Script:         $ScriptPath"
Write-Host "Schedule:       Daily at 02:00 AM"

# Verify files exist
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment python.exe not found at $PythonExe"
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "Nightly sync script not found at $ScriptPath"
    exit 1
}

# Ensure logs dir exists
$LogsDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Define Action
$ActionArg = "-NoProfile -WindowStyle Hidden -Command `"`$env:PYTHONIOENCODING='utf-8'; & '$PythonExe' '$ScriptPath' *>> '$LogPath'`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionArg -WorkingDirectory $ProjectDir

# Define Trigger (Daily at 2:00 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At "02:00:00"

# Define Settings (wake computer if supported, allow on battery, restart on failure)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Unregister existing task if present
try {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Unregistering existing scheduled task '$TaskName'..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
} catch {}

# Register new scheduled task
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Runs Recosys ML recommendations and Supabase nightly sync"
    Write-Host "`n[SUCCESS] Task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host "To test run it immediately, execute:" -ForegroundColor White
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host "Or run the script directly in your terminal:" -ForegroundColor White
    Write-Host "  .\.venv\Scripts\python.exe scripts\nightly_sync.py" -ForegroundColor Yellow
} catch {
    Write-Error "Failed to register scheduled task: $_"
    exit 1
}

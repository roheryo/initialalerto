# Kill all Node.js processes
Write-Host "Stopping all Node.js processes..."
Get-Process -Name node -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "All Node.js processes stopped."
Write-Host "You can now run 'npm run dev' again."

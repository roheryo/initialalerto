# Fix Port 3000 Already in Use

## Quick Solutions

### Option 1: Kill All Node Processes (Recommended)

**PowerShell:**
```powershell
Get-Process -Name node -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Or use the script:**
```powershell
.\kill-node-processes.ps1
```

Then restart:
```bash
npm run dev
```

### Option 2: Use a Different Port

Create a file `client/.env` with:
```
PORT=3001
```

Then React will run on port 3001 instead of 3000.

**Note:** If you change the port, access your app at `http://localhost:3001` instead of `http://localhost:3000`

### Option 3: Find and Kill Specific Process

**Find what's using port 3000:**
```powershell
Get-NetTCPConnection -LocalPort 3000 | Select-Object OwningProcess
```

**Kill that specific process (replace PID):**
```powershell
Stop-Process -Id <PID> -Force
```

## Why This Happens

- Previous React dev server didn't shut down properly
- Multiple terminal windows running `npm start`
- Crashed process still holding the port
- Windows sometimes doesn't release ports immediately

## Prevention

Always use `Ctrl+C` to stop the dev server instead of closing the terminal.

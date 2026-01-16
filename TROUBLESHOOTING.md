# Troubleshooting Guide

## Port 5000 Already in Use

If you see `Error: listen EADDRINUSE: address already in use :::5000`:

### Option 1: Kill the Process Using Port 5000

**Windows PowerShell:**
```powershell
# Find the process
Get-NetTCPConnection -LocalPort 5000 | Select-Object OwningProcess

# Kill it (replace PID with the process ID from above)
Stop-Process -Id <PID> -Force
```

**Or use Command Prompt:**
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Option 2: Change the Server Port

Edit `server/index.js` and change:
```javascript
const PORT = process.env.PORT || 5000;
```
to:
```javascript
const PORT = process.env.PORT || 5001;
```

Then update `client/package.json` proxy:
```json
"proxy": "http://localhost:5001"
```

## Client Dependencies Not Installed

If you see `'react-scripts' is not recognized`:

```bash
cd client
npm install
```

## ML Service Not Starting

1. **Check Python is installed:**
   ```bash
   python --version
   ```

2. **Install dependencies:**
   ```bash
   cd ml-service
   pip install -r requirements.txt
   ```

3. **Verify model files exist:**
   - `version-2 DL/davao_bilstm_attention_percentile_outbreak.keras`
   - `version-2 DL/case_scaler.pkl`

4. **Check for errors in the terminal** when starting the service

## All Services Running But Predictions Don't Work

1. **Check ML service is running:**
   - Visit `http://localhost:8000` - should show status
   - Visit `http://localhost:8000/docs` - should show API docs

2. **Check browser console** for errors

3. **Check network tab** - verify API calls are being made

4. **Verify authentication** - make sure you're logged in

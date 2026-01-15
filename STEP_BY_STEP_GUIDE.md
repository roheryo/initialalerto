# Step-by-Step Guide to Run the Application

## Prerequisites
- Node.js installed (version 14 or higher)
- npm (comes with Node.js)

## Step 1: Open Terminal/Command Prompt
- Open PowerShell, Command Prompt, or Terminal in VS Code
- Navigate to the project folder: `c:\Users\madul\.vscode\alertoinitial`

## Step 2: Install Dependencies

### Install root dependencies:
```powershell
npm install
```

### Install client (React) dependencies:
```powershell
cd client
npm install
cd ..
```

**OR use the combined command:**
```powershell
npm run install-all
```

## Step 3: Start the Backend Server

Open a **FIRST terminal window** and run:
```powershell
npm run server
```

Wait until you see: `Server running on port 5000`

## Step 4: Start the Frontend (React App)

Open a **SECOND terminal window** (keep the first one running) and run:
```powershell
cd client
npm start
```

Wait until you see: `webpack compiled successfully` and the browser opens automatically.

## Step 5: Login to the Application

1. The browser should open to `http://localhost:3000`
2. If not, manually open: `http://localhost:3000`
3. You'll see the login page
4. Use these **hardcoded credentials** (no database needed):
   - **Username:** `admin`
   - **Password:** `admin123`
   
   OR
   
   - **Username:** `user`
   - **Password:** `user123`
   
   OR
   
   - **Username:** `test`
   - **Password:** `test123`

5. Click "Login" button

## Step 6: Explore the Application

After login, you can:
- **Dashboard** - View disease statistics and charts
- **Patients** - View and manage patient records
- **Files** - Upload and view files

## Alternative: Run Both Servers at Once

If you want to run both servers in one command, use:
```powershell
npm run dev
```

This will start both backend and frontend together.

## Troubleshooting

### Port Already in Use
If you get "port 3000 or 5000 already in use":
- Close other applications using those ports
- Or kill the process using the port

### Dependencies Not Installing
- Make sure you have Node.js installed: `node --version`
- Try deleting `node_modules` folders and `package-lock.json` files, then reinstall

### Server Not Starting
- Make sure you're in the correct directory
- Check that all files are present in `server/` and `client/` folders

## Quick Commands Reference

```powershell
# Install all dependencies
npm run install-all

# Start both servers together
npm run dev

# Start only backend
npm run server

# Start only frontend (from root)
cd client
npm start
```

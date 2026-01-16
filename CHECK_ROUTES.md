# Quick Route Check

To verify your routes are working:

1. **Check if Express server is running:**
   - Look for "Server running on port 5000" in your terminal
   - If not, restart with `npm run server` or `npm run dev`

2. **Test the root predictions endpoint:**
   - Open browser: `http://localhost:5000/api/predictions`
   - Should show: `{"message":"Predictions API is working",...}`

3. **Test health check (requires auth):**
   - Use Postman or curl with your token
   - Or check browser console when Dashboard loads

4. **Common issues:**
   - Server not restarted after adding routes
   - Port 5000 already in use
   - Route file not saved properly

# PowerShell script to start both backend and frontend
# Run this from: C:\Users\bihar\Desktop\PLANT FRONT END

Write-Host "Starting LeafSense Development Servers..." -ForegroundColor Green
Write-Host ""

# Start Flask backend in a new window
Write-Host "Starting Flask backend (port 5000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; py -3.12 app.py"

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start React frontend
Write-Host "Starting React frontend (port 5173)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Backend: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the frontend server" -ForegroundColor Gray

npm run dev

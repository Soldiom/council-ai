# 🚀 Council AI Platform - Quick Start

Write-Host "🤖 Council AI Platform - Starting..." -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-not (Test-Path "web_app\app.py")) {
    Write-Host "❌ Error: Not in correct directory!" -ForegroundColor Red
    Write-Host "Please run this from: C:\Users\USER\Desktop\council1" -ForegroundColor Yellow
    exit 1
}

# Check Python
Write-Host "🔍 Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python first." -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
Write-Host "(This may take 2-3 minutes first time)" -ForegroundColor Gray
pip install fastapi uvicorn transformers torch 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some dependencies may have failed. Continuing..." -ForegroundColor Yellow
}

# Start server
Write-Host ""
Write-Host "🚀 Starting Council AI Platform..." -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 URL: http://localhost:8000" -ForegroundColor Green
Write-Host "🤖 6 AI models ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

# Change to web_app directory and run
cd web_app
python app.py

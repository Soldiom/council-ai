# 🚀 Deploy Council AI Platform - Step by Step

Your web platform is ready! Users can access all 6 AI models through a beautiful ChatGPT-like interface.

---

## 📋 What You Have

✅ **Backend API** (`app.py`)
- FastAPI server with 6 AI model endpoints
- Auto model selection based on user input
- REST API for chat

✅ **Frontend** (`static/index.html`)
- Beautiful ChatGPT-like interface
- Text input box
- Message history
- Model selector
- Responsive design (mobile + desktop)

✅ **6 AI Models** (will be on HuggingFace after 6 hours)
1. Unified AI - General purpose
2. Forensic AI - Audio/image/video analysis
3. Deepfake Detector - Fake detection
4. Document Verifier - Document verification
5. Agentic Browser - Autonomous research
6. Movie Creator - Movie generation

---

## 🖥️ Option 1: Test Locally (FREE)

**Test the platform on your computer:**

### Step 1: Install Dependencies
```bash
cd C:\Users\USER\Desktop\council1
pip install fastapi uvicorn transformers torch
```

### Step 2: Run the Server
```bash
cd web_app
python app.py
```

### Step 3: Open in Browser
1. Open browser
2. Go to: http://localhost:8000
3. You'll see the beautiful chat interface!
4. Type a message and test it

**Note:** Models will load from HuggingFace when you send first message (may take 1-2 min first time).

---

## ☁️ Option 2: Deploy to Render (FREE + Easy)

**Deploy to the internet for FREE - anyone can use it!**

### Step 1: Push to GitHub
```bash
cd C:\Users\USER\Desktop\council1
git add .
git commit -m "Add web platform"
git push origin main
```

### Step 2: Go to Render
1. Visit: https://render.com
2. Click "Sign Up" (use your GitHub account)
3. Click "New +" → "Web Service"

### Step 3: Connect Repository
1. Click "Connect a repository"
2. Find: `Soldiom/council-ai`
3. Click "Connect"

### Step 4: Configure Service
Fill in these settings:
- **Name**: `council-ai-platform`
- **Region**: `Oregon (US West)` (or closest to you)
- **Branch**: `main`
- **Root Directory**: Leave blank
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `cd web_app && python app.py`
- **Plan**: `Free`

### Step 5: Add Environment Variables (Optional)
If you want to use HuggingFace API token:
- Click "Environment"
- Add: `HF_TOKEN` = your HuggingFace token

### Step 6: Deploy!
1. Click "Create Web Service"
2. Wait 5-10 minutes for deployment
3. You'll get a URL like: `https://council-ai-platform.onrender.com`
4. Share this URL with anyone!

**Free Tier Limits:**
- ✅ 750 hours/month (enough for 24/7)
- ✅ Sleeps after 15 min of inactivity
- ✅ First request after sleep takes 30-60 sec

---

## 🚄 Option 3: Deploy to Railway (FREE + Fast)

**Alternative to Render - also free!**

### Step 1: Go to Railway
1. Visit: https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"

### Step 2: Select Repository
1. Choose: `Soldiom/council-ai`
2. Click "Deploy Now"

### Step 3: Configure
1. Railway auto-detects Python
2. Add start command: `cd web_app && python app.py`
3. Add port: `8000`

### Step 4: Get URL
1. Go to "Settings" → "Domains"
2. Click "Generate Domain"
3. You'll get: `https://council-ai.railway.app`

**Free Tier:**
- ✅ $5 credit/month (enough for 24/7)
- ✅ No sleep mode
- ✅ Faster than Render

---

## 🎨 Option 4: Split Frontend + Backend (Advanced)

**Frontend on Vercel (FREE) + Backend on Render (FREE)**

### Frontend on Vercel:
1. Create `vercel.json`:
```json
{
  "builds": [
    {
      "src": "web_app/static/index.html",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/web_app/static/index.html"
    }
  ]
}
```

2. Deploy to Vercel:
```bash
npm i -g vercel
vercel --prod
```

3. Update `index.html` to point to Render backend URL

### Backend on Render:
- Same as Option 2

**Benefits:**
- Frontend loads SUPER fast (Vercel CDN)
- Backend only pays for API calls
- More scalable

---

## 🧪 Testing Your Deployment

After deploying, test with these messages:

### Test 1: General Question
**Input:** "What is artificial intelligence?"
**Expected:** Unified AI responds with explanation

### Test 2: Forensic Analysis
**Input:** "Analyze this audio file for authenticity"
**Expected:** Forensic AI auto-selected

### Test 3: Deepfake Detection
**Input:** "Is this video a deepfake?"
**Expected:** Deepfake Detector auto-selected

### Test 4: Document Verification
**Input:** "Verify this signature"
**Expected:** Document Verifier auto-selected

### Test 5: Research
**Input:** "Search the web for latest AI news"
**Expected:** Agentic Browser auto-selected

### Test 6: Movie Creation
**Input:** "Create a movie about space exploration"
**Expected:** Movie Creator auto-selected

---

## 🔧 Troubleshooting

### Error: "Model not found"
**Solution:** Models need to be on HuggingFace first. Wait for 6-hour Colab training to complete.

### Error: "Connection refused"
**Solution:** Make sure server is running on port 8000.

### Error: "CORS error"
**Solution:** CORS is already enabled in `app.py`. If still happens, check your browser console.

### Models loading slow
**Solution:** First model load takes 1-2 minutes. After that, models are cached.

---

## 📊 Cost Comparison

| Platform | Free Tier | Best For |
|----------|-----------|----------|
| **Localhost** | ✅ FREE | Testing |
| **Render** | ✅ FREE (750h) | Easy deployment |
| **Railway** | ✅ FREE ($5 credit) | Fast, no sleep |
| **Vercel + Render** | ✅ FREE | Best performance |
| **Google Cloud** | ❌ ~$10/month | Heavy usage |

---

## 🎯 Recommended Steps

**For you right now:**

1. **Test locally first** (Option 1)
   - Make sure everything works
   - Test all 6 models
   - Fix any issues

2. **After 6-hour Colab test passes:**
   - Models will be on HuggingFace
   - Deploy to Render (Option 2)
   - Share URL with friends!

3. **If you get lots of users:**
   - Upgrade to Railway or Vercel + Render
   - Better performance
   - No sleep mode

---

## 🚀 Quick Start (Right Now)

```bash
# 1. Open terminal (PowerShell)
cd C:\Users\USER\Desktop\council1

# 2. Install dependencies
pip install fastapi uvicorn transformers torch

# 3. Run server
cd web_app
python app.py

# 4. Open browser → http://localhost:8000

# 5. Try typing: "Hello, tell me about AI!"
```

**That's it! Your AI platform is live!** 🎉

---

## 📝 Next Steps

After deploying:

1. ✅ Test the platform
2. ✅ Share URL with users
3. ✅ Monitor usage
4. ✅ Collect feedback
5. ✅ Improve models with Colab continuous learning

Your users will see a beautiful ChatGPT-like interface and can use all 6 AI models!

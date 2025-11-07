# How to Get Your FREE Hugging Face Token

## Quick Answer

**Token Access Required:** `Read` access only (the default)

**Time Required:** 2 minutes

**Cost:** $0 (completely free)

---

## Step-by-Step Instructions

### Step 1: Go to Hugging Face Settings
Visit: **https://huggingface.co/settings/tokens**

*(You'll need to create a free Hugging Face account if you don't have one)*

---

### Step 2: Create New Token

1. Click the **"New token"** button

2. Fill in the form:

   **Token name:** `council-agent` (or any name you want)
   
   **Token type:** Select **"Read"** 
   
   *(This is the default and all you need!)*

3. Click **"Generate a token"**

---

### Step 3: Copy Your Token

You'll see a token that looks like:
```
hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890
```

**Copy it!** (You'll only see it once)

---

### Step 4: Add to Your .env File

Open `.env` file in the project folder and add:

```bash
HF_API_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890
```

*(Replace with your actual token)*

---

## Why "Read" Access is Enough

The Hugging Face Inference API only needs **Read** access to:
- ✅ Download public models (for local use)
- ✅ Call Inference API endpoints (for cloud use)
- ✅ Access public model cards and metadata

You **don't need Write access** because:
- ❌ You're not uploading models
- ❌ You're not modifying datasets
- ❌ You're not creating repos

---

## Token Permissions Explained

| Permission | What It Does | Needed? |
|------------|--------------|---------|
| **Read** | Access public models & inference API | ✅ **YES** |
| Write | Upload models, datasets, create repos | ❌ No |
| Manage | Delete repos, change settings | ❌ No |

---

## Security Best Practices

✅ **DO:**
- Keep your token in `.env` file (never commit to git)
- Use Read-only access
- Regenerate token if exposed

❌ **DON'T:**
- Share your token publicly
- Commit `.env` to GitHub
- Use Write access unless needed

---

## Testing Your Token

Once you've added the token to `.env`:

```powershell
# Test with CLI
python -m cli.app run --agent strategist --input "Hello!"

# If it works, you'll see a real AI response (not "[mock]")
```

---

## Troubleshooting

### "Invalid token" error
- Make sure you copied the entire token (starts with `hf_`)
- Check for extra spaces or newlines
- Token might be expired - generate a new one

### "Rate limit reached"
- Free tier has limits (~1000 requests/day for most models)
- Wait a few minutes
- Try a different model
- Or use local provider: `--provider huggingface`

### "Model not available"
- Some models require approval (like Llama 2)
- Use default: `mistralai/Mistral-7B-Instruct-v0.2`
- Or try: `HuggingFaceH4/zephyr-7b-beta`

---

## Alternative: No Token Needed

If you don't want to use a token, you can:

### Option 1: Use Mock Provider (Testing)
```powershell
python -m cli.app run --agent strategist --input "test" --provider mock
```

### Option 2: Use Local Models (Download required)
```bash
# In .env:
DEFAULT_PROVIDER=huggingface
HF_MODEL=gpt2

# Download model
python scripts/download_model.py gpt2

# Run
python -m cli.app run --agent strategist --input "test"
```

### Option 3: Use Paid API (OpenAI, Claude, etc.)
```bash
# In .env:
DEFAULT_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

---

## Summary

**What you need:**
1. Free Hugging Face account
2. **Read access token** (default permission)
3. Add to `.env` file as `HF_API_TOKEN`

**That's it!** 🎉

The system will use free cloud-hosted models via Hugging Face Inference API.

**No GPU needed. No downloads needed. No costs.**

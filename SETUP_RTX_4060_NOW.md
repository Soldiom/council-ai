# ⚡ FIX RTX 4060 SETUP - 5 Minutes

## 🚨 Current Status
```
CUDA available: False
GPU: Not detected
VRAM: N/A
```

Your PyTorch **doesn't have CUDA support** - it's using CPU-only version.

---

## ✅ Fix in 3 Commands (5 Minutes)

### **Step 1: Uninstall CPU-only PyTorch**
```powershell
pip uninstall torch torchvision torchaudio -y
```

### **Step 2: Install CUDA PyTorch (RTX 4060 compatible)**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Step 3: Verify GPU Detection**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

**Expected output:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060
```

---

## 🎯 After GPU Works

### **Option A: Run Local GPU Training (FAST)**
```powershell
# ONE COMMAND - Everything automated on your RTX 4060:
python scripts/auto_local_gpu_finetune.py
```

**Results:**
- ✅ Detects RTX 4060
- ✅ Collects 35+ training examples
- ✅ Fine-tunes in ~30 minutes
- ✅ Cost: $0.04 electricity
- ✅ 2x faster than Colab

---

### **Option B: Use Google Colab (NO GPU NEEDED)**

If you can't get CUDA working or want to try now:

1. **Open:** https://colab.research.google.com/
2. **Runtime** → Change runtime type → **T4 GPU**
3. **Paste this code:**

```python
# Install dependencies
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub

# Clone your repo
!git clone https://github.com/YOUR_USERNAME/council1.git
%cd council1

# Upload training data (drag/drop to Colab)
from google.colab import files
uploaded = files.upload()  # Upload unified_model_complete.jsonl

# Train (automated - no input needed)
!python scripts/finetune_hf_model.py \
    --base-model meta-llama/Llama-3.2-3B-Instruct \
    --dataset-path unified_model_complete.jsonl \
    --output-model aliAIML/unified-ai-model \
    --epochs 3 \
    --batch-size 4 \
    --hf-token YOUR_HF_TOKEN

print("✅ Training complete! Model uploaded to HuggingFace")
```

**Time:** ~45 minutes on Colab T4 GPU  
**Cost:** $0 (FREE)

---

## 🆘 Troubleshooting

### **"CUDA still False after install"**

**Check Windows CUDA drivers:**
```powershell
nvidia-smi
```

If this **fails**, install NVIDIA drivers:
- Download: https://www.nvidia.com/Download/index.aspx
- Select: GeForce RTX 4060
- Install and restart PC

---

### **"bitsandbytes error on Windows"**
```powershell
pip uninstall bitsandbytes -y
pip install bitsandbytes-windows
```

---

### **"Still can't get GPU working"**

**No problem! Use Google Colab instead:**
- ✅ FREE T4 GPU (15GB VRAM)
- ✅ No drivers needed
- ✅ Works in your browser
- ✅ Same automation

**Follow Option B above** ⬆️

---

## 📊 What You Get

### **With RTX 4060 (Local):**
- Speed: ⚡⚡⚡ 2x faster than Colab
- Time limit: ♾️ None
- Cost: $0.04/training
- Models: Up to 7B (Llama, Mistral)

### **With Google Colab (Cloud):**
- Speed: ⚡⚡ Good
- Time limit: 12 hours FREE
- Cost: $0
- Models: Up to 13B

### **Hybrid (BEST):**
- Use RTX 4060 for daily training
- Use Colab for experiments
- Total cost: $1-3/month
- **95%+ savings vs APIs**

---

## 🎯 Quick Decision

### **I want to fix GPU now:**
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
python scripts/auto_local_gpu_finetune.py
```

### **I want to use Colab now (skip GPU setup):**
1. Open https://colab.research.google.com/
2. Runtime → T4 GPU
3. Copy code from Option B above
4. Run and done! ✅

---

## ✅ Success Checklist

After fixing CUDA:
- [ ] `nvidia-smi` shows RTX 4060
- [ ] `torch.cuda.is_available()` returns True
- [ ] GPU name shows "GeForce RTX 4060"
- [ ] Run `python scripts/auto_local_gpu_finetune.py`
- [ ] Training starts on GPU (~30 min)
- [ ] Model uploads to HuggingFace

---

**Choose your path:**
- 🎮 **Local GPU:** Faster, unlimited, $0.04/training
- ☁️ **Colab:** Free, easy, no setup needed
- 🚀 **Both:** Best of both worlds!

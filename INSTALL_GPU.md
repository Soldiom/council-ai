# RTX 4060 Setup - Quick Install

## ✅ 5-Minute Setup for Your RTX 4060

### **Step 1: Install CUDA PyTorch**

```powershell
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not found'); print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else 'N/A')"
```

**Expected output:**
```
GPU: NVIDIA GeForce RTX 4060
VRAM: 8.0 GB
```

### **Step 2: Install Fine-Tuning Dependencies**

```powershell
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually:
pip install transformers datasets peft bitsandbytes accelerate huggingface-hub sentencepiece protobuf scipy
```

### **Step 3: Test GPU Training**

```powershell
# Quick GPU test
python -c "import torch; from transformers import AutoTokenizer, AutoModelForCausalLM; print('✅ All imports successful'); print(f'✅ GPU ready: {torch.cuda.is_available()}')"
```

### **Step 4: Run Automated Fine-Tuning**

```powershell
# Collect data + build + fine-tune on RTX 4060
python scripts/auto_local_gpu_finetune.py
```

**That's it!** 🎉

---

## 🔧 If You Get Errors

### **Error: "CUDA not available"**

```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Error: "CUDA out of memory"**

Edit `scripts/finetune_hf_model.py`:
```python
# Reduce batch size
per_device_train_batch_size=2  # Default is 4

# Or use smaller model
base_model = "meta-llama/Llama-3.2-1B-Instruct"  # Instead of 3B
```

### **Error: "bitsandbytes not working"**

```powershell
# Windows-specific fix
pip uninstall bitsandbytes
pip install bitsandbytes-windows
```

---

## 🚀 After Installation

### **Run Full Pipeline:**

```powershell
# Everything automated:
python scripts/auto_local_gpu_finetune.py
```

**What happens:**
1. ✅ Detects RTX 4060
2. ✅ Collects training data (35+ examples)
3. ✅ Builds datasets
4. ✅ Fine-tunes on your GPU (~30 min)
5. ✅ Uploads to HuggingFace

### **Monitor GPU:**

```powershell
# Watch GPU usage in real-time
nvidia-smi -l 1

# Or use Task Manager → Performance → GPU
```

### **Expected Training Time:**

| Model | Examples | Time (RTX 4060) |
|-------|----------|-----------------|
| Llama 3.2 3B | 100 | ~30 min |
| Llama 3.2 3B | 500 | ~2 hours |
| Mistral 7B | 100 | ~45 min |

---

## 💡 Google Colab Backup

If your PC is busy or you want to experiment:

```python
# Open: https://colab.research.google.com/
# Runtime → T4 GPU
# Copy/paste automated code from: LOCAL_GPU_SETUP.md
```

**Both work together:**
- RTX 4060: Daily production training
- Google Colab: Experiments and backup

---

## ✅ You're Ready!

```powershell
# Test everything
python -c "import torch; print('✅ GPU:', torch.cuda.get_device_name(0))"

# Run automated pipeline
python scripts/auto_local_gpu_finetune.py
```

**Your RTX 4060 is ready for AI fine-tuning!** 🎮🚀

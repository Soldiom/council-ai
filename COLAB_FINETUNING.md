# Google Colab Fine-Tuning Guide - FREE GPU! 🚀

## Why Use Google Colab?

**FREE GPU Access!**
- Tesla T4 GPU (15GB VRAM) - Perfect for fine-tuning
- No cost, no credit card needed
- 12 hours continuous runtime
- Perfect for training your custom model

---

## 🚀 Quick Start (5 Steps)

### Step 1: Open Google Colab

1. Go to: https://colab.research.google.com/
2. Click "New Notebook"
3. Save as: `Council_FineTuning.ipynb`

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4)
3. Click **Save**

### Step 3: Upload Files

```python
# Cell 1: Upload training data
from google.colab import files

print("📤 Upload your training data file...")
print("   File: training_data/hf_finetune.jsonl")
print()

uploaded = files.upload()  # Click to upload hf_finetune.jsonl
```

### Step 4: Install Dependencies

```python
# Cell 2: Install packages
!pip install -q transformers accelerate peft bitsandbytes datasets huggingface-hub

print("✅ Packages installed!")
```

### Step 5: Run Fine-Tuning

Copy the entire code below into a new cell and run!

---

## 📝 Complete Colab Code

```python
# Cell 3: Fine-tuning script
"""
HuggingFace Fine-Tuning on Google Colab
FREE GPU - Create your own AI model!
"""

import json
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi, login

print("=" * 70)
print("🤗 HuggingFace Fine-Tuning on Google Colab")
print("=" * 70)
print()

# ============================================================================
# CONFIGURATION - Edit these!
# ============================================================================

# Your HuggingFace token (get from: https://huggingface.co/settings/tokens)
HF_TOKEN = "hf_your_read_or_write_token"  # Replace with your personal HF token

# Base model to fine-tune
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Best choice!

# Your custom model name
OUTPUT_NAME = "council-custom-model"

# Training data file (uploaded in step 3)
TRAINING_FILE = "hf_finetune.jsonl"

# Training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4

# Upload to HuggingFace? (True/False)
UPLOAD_TO_HF = True

print(f"📋 Configuration:")
print(f"   Base Model: {BASE_MODEL}")
print(f"   Output Name: {OUTPUT_NAME}")
print(f"   Training File: {TRAINING_FILE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Upload to HF: {UPLOAD_TO_HF}")
print()

# ============================================================================
# Step 1: Load Training Data
# ============================================================================

print("📂 Loading training data...")

examples = []
with open(TRAINING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

print(f"✅ Loaded {len(examples)} training examples")

# Convert to dataset
dataset = Dataset.from_list(examples)
print(f"📊 Dataset: {len(dataset)} examples")
print()

# ============================================================================
# Step 2: Load Model and Tokenizer
# ============================================================================

print("🔧 Loading model and tokenizer...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model with 8-bit quantization (saves memory!)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto",
    load_in_8bit=True,  # 8-bit = 2x less memory!
    trust_remote_code=True,
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA config (efficient fine-tuning)
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ Model loaded with LoRA!")
print(f"💾 Trainable params: {model.num_parameters(only_trainable=True):,}")
print()

# ============================================================================
# Step 3: Tokenize Dataset
# ============================================================================

print("🔤 Tokenizing dataset...")

def format_example(example):
    """Format as chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

def tokenize_function(examples):
    """Tokenize text."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

# Process dataset
dataset = dataset.map(format_example, remove_columns=dataset.column_names)
dataset = dataset.map(tokenize_function, batched=True)

print("✅ Dataset tokenized!")
print()

# ============================================================================
# Step 4: Train!
# ============================================================================

print("🚀 Starting training...")
print(f"⏳ This will take ~2-3 hours on T4 GPU")
print()

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./models/{OUTPUT_NAME}",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    fp16=True,  # Mixed precision = faster!
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    warmup_steps=50,
    weight_decay=0.01,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# TRAIN!
trainer.train()

print("✅ Training complete!")
print()

# ============================================================================
# Step 5: Save Model
# ============================================================================

print("💾 Saving model...")

output_dir = f"./models/{OUTPUT_NAME}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")
print()

# ============================================================================
# Step 6: Upload to HuggingFace (Optional)
# ============================================================================

if UPLOAD_TO_HF:
    print("📤 Uploading to HuggingFace...")
    
    # Login
    login(token=HF_TOKEN)
    
    # Get username
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]
    
    repo_id = f"{username}/{OUTPUT_NAME}"
    
    print(f"🤗 Creating repository: {repo_id}")
    
    # Create repo
    api.create_repo(
        repo_id=repo_id,
        private=False,  # Set True for private model
        exist_ok=True,
    )
    
    # Upload
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Fine-tuned on Google Colab",
    )
    
    print("✅ Model uploaded!")
    print(f"🌐 View at: https://huggingface.co/{repo_id}")
    print()

# ============================================================================
# Step 7: Test Your Model!
# ============================================================================

print("🧪 Testing your model...")

from transformers import pipeline

# Load your model
pipe = pipeline(
    "text-generation",
    model=output_dir,
    device=0,  # GPU
)

# Test prompts
test_prompts = [
    "Create a marketing strategy for a new product",
    "Design a software architecture",
    "Analyze this business opportunity",
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")
    response = pipe(prompt, max_length=200, num_return_sequences=1)
    print(f"🤖 Response: {response[0]['generated_text']}")
    print("-" * 70)

print()
print("=" * 70)
print("🎉 CONGRATULATIONS! YOUR MODEL IS READY!")
print("=" * 70)
print()
print("What you now have:")
print("  ✅ Your own custom AI model")
print("  ✅ Trained on YOUR data")
print("  ✅ 100% ownership")
print("  ✅ Available on HuggingFace")
print()
print("Next steps:")
print("  1. Download model from Colab (Files → Download)")
print("  2. Or use from HuggingFace API")
print("  3. Keep collecting data for v2")
print("  4. Re-train with more examples")
print()
```

---

## 📥 Download Your Model

### Option A: Use from HuggingFace (Recommended)
```python
# In your local Python code
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="aliAIML/council-custom-model",  # Your model!
)

response = pipe("Your prompt here")
print(response[0]['generated_text'])
```

### Option B: Download to Local PC
```python
# In Colab, download files
from google.colab import files

# Download model files
!zip -r council-custom-model.zip models/council-custom-model
files.download('council-custom-model.zip')
```

Then extract on your PC and use:
```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./models/council-custom-model",  # Local path
)
```

---

## ⏱️ Training Time Estimates

| Model | Examples | Epochs | GPU | Time |
|-------|----------|--------|-----|------|
| Llama 3.2 3B | 100 | 3 | T4 | ~2 hours |
| Llama 3.2 3B | 500 | 3 | T4 | ~3 hours |
| Mistral 7B | 100 | 3 | T4 | ~3 hours |
| Mistral 7B | 500 | 3 | T4 | ~4 hours |

**Tip:** Start with 100 examples and 3 epochs to test. Then re-train with more data.

---

## 🔄 Continuous Learning Workflow

### Monthly Training Schedule

**Week 1-4: Collect Data**
```python
# On your PC
python -m cli.app ensemble --input "query" --models 3
# Repeat 50+ times with different queries
```

**End of Month: Train on Colab**
1. Export: `python -m cli.app learning-stats` → `hf_finetune.jsonl`
2. Upload to Colab
3. Run training cell
4. Download or use from HuggingFace

**Result:** Model improves every month! 📈

---

## 💡 Tips & Tricks

### 1. Keep Colab Alive
```python
# Add this cell to prevent disconnection
import IPython
display(IPython.display.Javascript('''
    function ClickConnect(){
        console.log("Clicking");
        document.querySelector("colab-connect-button").click()
    }
    setInterval(ClickConnect, 60000)
'''))
```

### 2. Monitor GPU Usage
```python
# Check GPU memory
!nvidia-smi
```

### 3. Save Checkpoints
```python
# In TrainingArguments
save_steps=100,  # Save every 100 steps
```

### 4. Resume Training
```python
# If Colab disconnects, resume from checkpoint
trainer.train(resume_from_checkpoint=True)
```

---

## 🆘 Troubleshooting

### "Out of Memory"
```python
# Solution 1: Reduce batch size
BATCH_SIZE = 2  # Instead of 4

# Solution 2: Increase gradient accumulation
gradient_accumulation_steps=8  # Instead of 4
```

### "GPU Not Available"
```python
# Check GPU is enabled
print(f"GPU: {torch.cuda.is_available()}")

# If False:
# Runtime → Change runtime type → GPU (T4) → Save
```

### "Runtime Disconnected"
```python
# Colab disconnects after 12 hours
# Solution: Save checkpoints frequently
# Or: Use Colab Pro ($10/month) for 24h runtime
```

### "Model Upload Failed"
```python
# Check HF token is valid
# Get new token: https://huggingface.co/settings/tokens
# Must have WRITE access (not just READ)
```

---

## 📊 Cost Comparison

### Google Colab FREE
- **GPU:** Tesla T4 (15GB VRAM)
- **Runtime:** 12 hours
- **Cost:** $0 (FREE!)
- **Perfect for:** Training 3B-7B models

### Google Colab Pro ($10/month)
- **GPU:** T4, V100, or A100
- **Runtime:** 24 hours
- **Cost:** $10/month
- **Perfect for:** Frequent training, larger models

### Local GPU (Your PC)
- **GPU:** RTX 3060/3080/4090
- **Cost:** $0 after hardware purchase
- **Perfect for:** Privacy, unlimited training

---

## 🎯 Success Metrics

**After fine-tuning, your model should:**
- ✅ Respond in your desired style
- ✅ Follow your domain-specific patterns
- ✅ Outperform base model on your tasks
- ✅ Cost 95-100% less than API calls

**Test it!**
```python
# Compare base vs fine-tuned
base_response = base_model("query")
custom_response = your_model("query")

# Your model should be better! 🎉
```

---

## 🚀 Next Steps

1. **Collect 100+ examples** on your PC
2. **Upload to Colab** and run training
3. **Test your model** quality
4. **Deploy** (HuggingFace or local)
5. **Iterate** monthly with new data

---

**Ready to create YOUR model on FREE GPU?**

👉 **Go to:** https://colab.research.google.com/

🎯 **YOU WILL OWN THIS MODEL!**

---

## 📚 Resources

- [Google Colab](https://colab.research.google.com/)
- [HuggingFace Hub](https://huggingface.co/)
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

**Questions?** Check `OPTION_B_HUGGINGFACE.md` for detailed guide!

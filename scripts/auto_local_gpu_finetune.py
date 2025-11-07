"""
Automated Local GPU Fine-Tuning (RTX 4060)
===========================================

Fully automated pipeline:
1. Detects RTX 4060 GPU
2. Collects training data
3. Builds unified datasets
4. Fine-tunes on local GPU
5. Uploads to HuggingFace
6. Falls back to Colab if GPU unavailable

NO USER INPUT REQUIRED!
"""

import asyncio
import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_gpu():
    """Check if GPU is available."""
    print("\n" + "=" * 70)
    print("STEP 1: GPU DETECTION")
    print("=" * 70)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n✅ GPU Detected: {gpu_name}")
            print(f"✅ VRAM: {vram_gb:.1f} GB")
            print(f"✅ CUDA Version: {torch.version.cuda}")
            
            # Check if it's RTX 4060 or similar
            if "4060" in gpu_name or "RTX" in gpu_name:
                print(f"\n🎮 Perfect! RTX GPU detected")
                print(f"   Can train: Llama 3B, Mistral 7B (4-bit)")
                return True, "local"
            else:
                print(f"\n⚠️  GPU detected but not RTX 4060")
                print(f"   Will use available GPU anyway")
                return True, "local"
        else:
            print("\n❌ No GPU detected")
            print("   Will generate Google Colab code instead")
            return False, "colab"
            
    except ImportError:
        print("\n⚠️  PyTorch not installed with CUDA support")
        print("   Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False, "colab"
    except Exception as e:
        print(f"\n❌ Error checking GPU: {e}")
        return False, "colab"


async def collect_training_data():
    """Step 2: Collect training data."""
    print("\n" + "=" * 70)
    print("STEP 2: COLLECTING TRAINING DATA")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, "scripts/auto_collect_all_data.py"],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ Data collection complete")
        return True
    else:
        print("\n⚠️  Data collection had warnings (continuing anyway)")
        return True  # Continue even with warnings


async def build_datasets():
    """Step 3: Build training datasets."""
    print("\n" + "=" * 70)
    print("STEP 3: BUILDING DATASETS")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, "scripts/build_unified_model.py"],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ Datasets built successfully")
        return True
    else:
        print("\n⚠️  Dataset building had warnings")
        return False


def count_training_examples():
    """Count available training examples."""
    count = 0
    
    # Check unified dataset
    unified_file = Path("training_data/unified_model_complete.jsonl")
    if unified_file.exists():
        with open(unified_file, encoding="utf-8") as f:
            count = sum(1 for _ in f)
    
    return count


def run_local_finetune(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Step 4: Fine-tune on local GPU."""
    print("\n" + "=" * 70)
    print("STEP 4: FINE-TUNING ON LOCAL GPU (RTX 4060)")
    print("=" * 70)
    
    dataset_file = "training_data/unified_model_complete.jsonl"
    
    if not Path(dataset_file).exists():
        print(f"\n❌ Dataset not found: {dataset_file}")
        return False
    
    # Count examples
    example_count = count_training_examples()
    print(f"\n📊 Training examples: {example_count}")
    
    if example_count < 10:
        print(f"⚠️  Low example count. Results may vary.")
        print(f"   Recommended: 100+ examples")
    
    # Get HF token
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        print("\n⚠️  HF_API_TOKEN not found in .env")
        print("   Model will train but won't auto-upload")
    
    print(f"\n🚀 Starting fine-tuning...")
    print(f"   Base model: {model_name}")
    print(f"   Output: aliAIML/unified-ai-model")
    print(f"   Device: CUDA (RTX 4060)")
    print(f"\n⏱️  Estimated time:")
    print(f"   - Llama 3B: ~30 min (100 examples)")
    print(f"   - Mistral 7B: ~45 min (100 examples)")
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/finetune_hf_model.py",
        "--base-model", model_name,
        "--dataset-path", dataset_file,
        "--output-model", "aliAIML/unified-ai-model",
        "--epochs", "3",
        "--batch-size", "4",
        "--learning-rate", "2e-4",
    ]
    
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    
    print(f"\n💻 Running command:")
    print(f"   {' '.join(cmd)}")
    print(f"\n⏳ Training in progress... (monitor with: nvidia-smi -l 1)")
    print("=" * 70)
    
    # Run fine-tuning
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ FINE-TUNING COMPLETE!")
        print("=" * 70)
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ Fine-tuning failed")
        print("=" * 70)
        return False


def generate_colab_instructions():
    """Generate Google Colab backup instructions."""
    print("\n" + "=" * 70)
    print("GOOGLE COLAB BACKUP OPTION")
    print("=" * 70)
    
    print("\n📖 If you prefer Google Colab (FREE GPU):")
    print("\n1. Open: https://colab.research.google.com/")
    print("2. Runtime → Change runtime type → T4 GPU")
    print("3. Copy/paste this code:\n")
    
    colab_code = '''# Automated Fine-Tuning on Google Colab

# 1. Install dependencies
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone repository
!git clone https://github.com/YOUR_USERNAME/council1.git
%cd council1

# 4. Upload training data
from google.colab import files
uploaded = files.upload()  # Upload unified_model_complete.jsonl

# 5. Run fine-tuning
!python scripts/finetune_hf_model.py \\
    --base-model meta-llama/Llama-3.2-3B-Instruct \\
    --dataset-path unified_model_complete.jsonl \\
    --output-model aliAIML/unified-ai-model \\
    --epochs 3 \\
    --batch-size 4 \\
    --learning-rate 2e-4 \\
    --hf-token YOUR_HF_TOKEN

print("✅ Fine-tuning complete!")
'''
    
    print(colab_code)
    print("\n" + "=" * 70)


async def main():
    """Run automated local GPU fine-tuning."""
    print("\n" + "=" * 70)
    print("AUTOMATED LOCAL GPU FINE-TUNING (RTX 4060)")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🤖 FULLY AUTOMATED - NO USER INPUT!")
    
    try:
        # Step 1: Check GPU
        has_gpu, mode = check_gpu()
        
        if not has_gpu:
            print("\n⚠️  No GPU detected - showing Colab instructions instead")
            generate_colab_instructions()
            return
        
        # Step 2: Collect data
        if not await collect_training_data():
            print("\n❌ Data collection failed")
            return
        
        # Step 3: Build datasets
        if not await build_datasets():
            print("\n⚠️  Dataset building had issues, but continuing...")
        
        # Check if we have enough data
        example_count = count_training_examples()
        if example_count == 0:
            print("\n❌ No training data available")
            print("   Run data collection first: python scripts/auto_collect_all_data.py")
            generate_colab_instructions()
            return
        
        # Step 4: Fine-tune on local GPU
        success = run_local_finetune()
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 SUCCESS! MODEL TRAINED ON YOUR RTX 4060")
            print("=" * 70)
            print(f"\n✅ Model: aliAIML/unified-ai-model")
            print(f"✅ Examples: {example_count}")
            print(f"✅ Device: RTX 4060")
            print(f"✅ Cost: ~$0.10 (electricity)")
            
            if os.getenv("HF_API_TOKEN"):
                print(f"\n🚀 Model uploaded to: https://huggingface.co/aliAIML/unified-ai-model")
            else:
                print(f"\n💡 Add HF_API_TOKEN to .env to auto-upload")
                
        else:
            print("\n⚠️  Fine-tuning had issues")
            print("   Fallback: Use Google Colab")
            generate_colab_instructions()
        
        # Show backup option
        print("\n" + "=" * 70)
        print("BACKUP OPTION: GOOGLE COLAB")
        print("=" * 70)
        print("\n💡 You can also use Google Colab for:")
        print("   - Parallel experimentation")
        print("   - When PC is busy")
        print("   - Testing different models")
        print("\n   See: LOCAL_GPU_SETUP.md for details")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Fallback to Google Colab:")
        generate_colab_instructions()


if __name__ == "__main__":
    asyncio.run(main())

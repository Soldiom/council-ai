#!/usr/bin/env python
"""
Download and cache Hugging Face models for local, offline inference.

Usage:
    python scripts/download_model.py [model_name]

Examples:
    python scripts/download_model.py microsoft/phi-2
    python scripts/download_model.py gpt2
    python scripts/download_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import sys
import os
from pathlib import Path


def download_model(model_name: str = "microsoft/phi-2"):
    """Download a Hugging Face model and tokenizer to local cache."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: transformers and huggingface-hub are required.")
        print("Install them with: pip install transformers huggingface-hub accelerate")
        sys.exit(1)

    print(f"📥 Downloading model: {model_name}")
    print("   This may take several minutes depending on model size...")
    print()

    try:
        # Download tokenizer
        print("🔧 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer cached: {tokenizer.name_or_path}")

        # Download model
        print("🧠 Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # For large models, consider low_cpu_mem_usage=True or device_map="auto"
        )
        print(f"✅ Model cached: {model.config._name_or_path}")

        # Report cache location
        from transformers.utils import TRANSFORMERS_CACHE
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or TRANSFORMERS_CACHE
        print()
        print(f"📦 Models cached in: {cache_dir}")
        print()
        print("✅ Download complete! You can now use this model offline.")
        print()
        print("To use this model with the Council:")
        print(f"  1. Set HF_MODEL={model_name} in your .env file")
        print("  2. Set DEFAULT_PROVIDER=huggingface")
        print("  3. Run: python -m cli.app run --agent strategist --input 'Your prompt'")
        print()

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print()
        print("Common issues:")
        print("  - Model requires authentication: run `huggingface-cli login` first")
        print("  - Insufficient disk space: large models can be 5-50GB")
        print("  - Network timeout: try again or use a smaller model like 'gpt2'")
        sys.exit(1)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "microsoft/phi-2"
    print()
    print("=" * 70)
    print("  Hugging Face Model Downloader")
    print("  Council of Infinite Innovators - Free Local LLMs")
    print("=" * 70)
    print()
    download_model(model)

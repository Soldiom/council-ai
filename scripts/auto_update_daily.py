"""
Daily Auto-Update Script
========================

Runs daily to:
1. Discover new HuggingFace models
2. Train YOUR unified model on user data
3. Deploy updated model
4. Repeat forever!

Schedule this to run daily (cron, Task Scheduler, cloud function)
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from council.model_hub import get_hub, UnifiedModelTrainer
from dotenv import load_dotenv

load_dotenv()


async def daily_update():
    """Run daily update process."""
    print("=" * 70)
    print(f"🌅 Daily Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    hub = get_hub()
    trainer = UnifiedModelTrainer(hub)
    
    # Step 1: Discover new models
    print("📡 Step 1: Discovering latest HuggingFace models...")
    print()
    
    try:
        capabilities = await hub.discover_all_models(
            top_n_per_category=5,
            min_downloads=1000,
        )
        
        total_models = sum(len(v) for v in capabilities.values())
        print(f"✅ Discovered {total_models} models across {len(capabilities)} categories")
        print()
        
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        print()
    
    # Step 2: Count training data
    print("📊 Step 2: Checking training data...")
    print()
    
    training_files = list(trainer.training_data_dir.glob("interactions_*.jsonl"))
    total_interactions = 0
    
    for file in training_files:
        with open(file) as f:
            count = sum(1 for _ in f)
            total_interactions += count
            print(f"   {file.name}: {count:,} interactions")
    
    print()
    print(f"   Total: {total_interactions:,} interactions")
    print()
    
    # Step 3: Train unified model if enough data
    if total_interactions >= 100:
        print("🎓 Step 3: Training YOUR unified model...")
        print()
        
        try:
            result = await trainer.train_unified_model_daily(
                base_model="meta-llama/Llama-3.2-3B-Instruct",
                output_name="unified-ai-model",
            )
            
            print(f"✅ Training data prepared: {result['training_file']}")
            print(f"   {result['total_examples']:,} examples ready")
            print()
            print("🚀 Next: Run fine-tuning")
            print("   Local: python scripts/finetune_hf_model.py")
            print("   Colab: See COLAB_FINETUNING.md")
            print()
            
        except Exception as e:
            print(f"❌ Training preparation failed: {e}")
            print()
    else:
        print(f"⏳ Step 3: Waiting for more data...")
        print(f"   Have: {total_interactions} interactions")
        print(f"   Need: 100 minimum")
        print(f"   Missing: {100 - total_interactions}")
        print()
    
    # Step 4: Summary
    print("=" * 70)
    print("📈 Daily Update Complete!")
    print("=" * 70)
    print()
    print("📊 Summary:")
    print(f"   Models available: {total_models}")
    print(f"   Capabilities: {len(capabilities)}")
    print(f"   User interactions: {total_interactions:,}")
    print(f"   Ready for training: {'✅ Yes' if total_interactions >= 100 else '❌ No'}")
    print()
    print("🔄 Next update: Tomorrow same time")
    print()


if __name__ == "__main__":
    asyncio.run(daily_update())

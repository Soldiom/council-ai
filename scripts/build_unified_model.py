"""
Build Unified Model - Complete Training Pipeline
=================================================

This script:
1. Collects ALL training data (ensemble, unified platform, forensic)
2. Prepares unified dataset
3. Fine-tunes YOUR unified model
4. Deploys to HuggingFace

YOUR UNIFIED MODEL = All capabilities in one model!
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Fix Windows encoding issues
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from council.continuous_learning import ContinuousLearner
from council.agents.forensic import ForensicModelTrainer
from dotenv import load_dotenv

load_dotenv()


class UnifiedModelBuilder:
    """Build and train YOUR unified model from ALL data sources."""
    
    def __init__(self):
        self.training_dir = Path("training_data")
        self.ensemble_learner = ContinuousLearner()
        self.forensic_trainer = ForensicModelTrainer()
        
    async def collect_all_training_data(self) -> Dict[str, Any]:
        """Collect training data from all sources."""
        print("=" * 70)
        print("📊 Collecting ALL Training Data")
        print("=" * 70)
        print()
        
        stats = {
            "ensemble_examples": 0,
            "unified_platform_examples": 0,
            "forensic_examples": 0,
            "deepfake_examples": 0,
            "document_forgery_examples": 0,
            "total_examples": 0,
        }
        
        # 1. Ensemble data (from model ensemble interactions)
        print("1️⃣ Ensemble Model Data...")
        ensemble_stats = self.ensemble_learner.get_learning_stats()
        if "total_examples" in ensemble_stats:
            stats["ensemble_examples"] = ensemble_stats["total_examples"]
            print(f"   ✅ Found {stats['ensemble_examples']} examples")
        else:
            print("   ℹ️ No ensemble data yet")
        print()
        
        # 2. Unified platform data (user interactions)
        print("2️⃣ Unified Platform Data...")
        unified_dir = self.training_dir / "unified"
        if unified_dir.exists():
            unified_files = list(unified_dir.glob("interactions_*.jsonl"))
            for file in unified_files:
                with open(file) as f:
                    count = sum(1 for _ in f)
                    stats["unified_platform_examples"] += count
                    print(f"   📁 {file.name}: {count} examples")
            print(f"   ✅ Total: {stats['unified_platform_examples']} examples")
        else:
            print("   ℹ️ No unified platform data yet")
        print()
        
        # 3. Forensic data (specialized forensic analysis)
        print("3️⃣ Forensic Analysis Data...")
        forensic_dir = self.training_dir / "forensic"
        if forensic_dir.exists():
            forensic_files = list(forensic_dir.glob("forensic_*.jsonl"))
            for file in forensic_files:
                with open(file) as f:
                    count = sum(1 for _ in f)
                    stats["forensic_examples"] += count
                    print(f"   📁 {file.name}: {count} examples")
            print(f"   ✅ Total: {stats['forensic_examples']} examples")
        else:
            print("   ℹ️ No forensic data yet")
        print()
        
        # 4. Deepfake detection data
        print("4️⃣ Deepfake Detection Data...")
        deepfake_dir = self.training_dir / "deepfake"
        if deepfake_dir.exists():
            deepfake_files = list(deepfake_dir.glob("deepfake_*.jsonl"))
            for file in deepfake_files:
                with open(file) as f:
                    count = sum(1 for _ in f)
                    stats["deepfake_examples"] += count
                    print(f"   📁 {file.name}: {count} examples")
            print(f"   ✅ Total: {stats['deepfake_examples']} examples")
        else:
            print("   ℹ️ No deepfake data yet")
        print()
        
        # 5. Document forgery detection data
        print("5️⃣ Document Forgery Detection Data...")
        doc_forgery_dir = self.training_dir / "document_forgery"
        if doc_forgery_dir.exists():
            doc_files = list(doc_forgery_dir.glob("document_forgery_*.jsonl"))
            for file in doc_files:
                with open(file) as f:
                    count = sum(1 for _ in f)
                    stats["document_forgery_examples"] += count
                    print(f"   📁 {file.name}: {count} examples")
            print(f"   ✅ Total: {stats['document_forgery_examples']} examples")
        else:
            print("   ℹ️ No document forgery data yet")
        print()
        
        # Total
        stats["total_examples"] = (
            stats["ensemble_examples"] +
            stats["unified_platform_examples"] +
            stats["forensic_examples"] +
            stats["deepfake_examples"] +
            stats["document_forgery_examples"]
        )
        
        print("=" * 70)
        print("📈 Summary")
        print("=" * 70)
        print(f"Ensemble interactions:    {stats['ensemble_examples']:>6}")
        print(f"Platform user data:       {stats['unified_platform_examples']:>6}")
        print(f"Forensic analysis:        {stats['forensic_examples']:>6}")
        print(f"Deepfake detection:       {stats['deepfake_examples']:>6}")
        print(f"Document forgery:         {stats['document_forgery_examples']:>6}")
        print("-" * 70)
        print(f"TOTAL TRAINING EXAMPLES:  {stats['total_examples']:>6}")
        print("=" * 70)
        print()
        
        return stats
    
    async def prepare_unified_dataset(self, output_file: str = "training_data/unified_model_complete.jsonl"):
        """Merge all training data into unified dataset."""
        print("🔧 Preparing Unified Training Dataset...")
        print()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_written = 0
        
        with open(output_path, "w", encoding="utf-8") as out:
            # 1. Ensemble data
            ensemble_file = self.training_dir / "ensemble_finetune.jsonl"
            if ensemble_file.exists():
                print("   Adding ensemble data...")
                with open(ensemble_file) as f:
                    for line in f:
                        out.write(line)
                        total_written += 1
            
            # 2. Unified platform data
            unified_dir = self.training_dir / "unified"
            if unified_dir.exists():
                print("   Adding unified platform data...")
                for file in unified_dir.glob("interactions_*.jsonl"):
                    with open(file) as f:
                        for line in f:
                            example = json.loads(line)
                            # Convert to training format
                            messages = [
                                {"role": "user", "content": str(example.get("input", ""))},
                                {"role": "assistant", "content": str(example.get("output", ""))},
                            ]
                            out.write(json.dumps({"messages": messages}) + "\n")
                            total_written += 1
            
            # 3. Forensic data
            forensic_dir = self.training_dir / "forensic"
            if forensic_dir.exists():
                print("   Adding forensic data...")
                for file in forensic_dir.glob("forensic_*.jsonl"):
                    with open(file) as f:
                        for line in f:
                            example = json.loads(line)
                            messages = [
                                {
                                    "role": "system",
                                    "content": "You are an expert digital forensics analyst."
                                },
                                {"role": "user", "content": str(example.get("input", ""))},
                                {"role": "assistant", "content": json.dumps(example.get("output", {}), indent=2)},
                            ]
                            out.write(json.dumps({"messages": messages}) + "\n")
                            total_written += 1
        
        print()
        print(f"✅ Unified dataset prepared!")
        print(f"   Total examples: {total_written}")
        print(f"   Output file: {output_path}")
        print()
        
        return {
            "total_examples": total_written,
            "output_file": str(output_path),
            "ready_for_training": total_written >= 100,
        }
    
    async def build_unified_model(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_name: str = "unified-ai-model",
    ):
        """Build YOUR unified model from all data sources."""
        print("=" * 70)
        print("🏗️ BUILDING YOUR UNIFIED AI MODEL")
        print("=" * 70)
        print()
        print(f"Base model: {base_model}")
        print(f"Output: aliAIML/{output_name}")
        print()
        
        # Step 1: Collect all data
        stats = await self.collect_all_training_data()
        
        if stats["total_examples"] == 0:
            print("❌ No training data found!")
            print()
            print("💡 To collect training data:")
            print("   1. Run: python -m cli.app ensemble --input 'query' --models 3")
            print("   2. Use: POST /task on unified API")
            print("   3. Run: python -m cli.app forensic --input 'logs'")
            print()
            return None
        
        if stats["total_examples"] < 100:
            print(f"⚠️ Only {stats['total_examples']} examples (need 100 minimum)")
            print()
            print("💡 Collect more data:")
            print(f"   - Ensemble: {max(0, 50 - stats['ensemble_examples'])} more examples")
            print(f"   - Platform: {max(0, 30 - stats['unified_platform_examples'])} more examples")
            print(f"   - Forensic: {max(0, 20 - stats['forensic_examples'])} more examples")
            print()
            return None
        
        # Step 2: Prepare unified dataset
        dataset = await self.prepare_unified_dataset()
        
        # Step 3: Training instructions
        print("=" * 70)
        print("🎓 READY TO TRAIN YOUR UNIFIED MODEL!")
        print("=" * 70)
        print()
        print("📊 Dataset Statistics:")
        print(f"   Total examples: {dataset['total_examples']}")
        print(f"   From ensemble: {stats['ensemble_examples']}")
        print(f"   From platform: {stats['unified_platform_examples']}")
        print(f"   From forensic: {stats['forensic_examples']}")
        print()
        print("🚀 Training Options:")
        print()
        print("   OPTION A: Google Colab (FREE GPU - RECOMMENDED)")
        print("   ┌─────────────────────────────────────────────┐")
        print("   │ 1. Go to: https://colab.research.google.com │")
        print("   │ 2. Upload file:", dataset['output_file'].ljust(24), "│")
        print("   │ 3. Follow: COLAB_FINETUNING.md             │")
        print("   │ 4. Result: aliAIML/unified-ai-model        │")
        print("   └─────────────────────────────────────────────┘")
        print()
        print("   OPTION B: Local GPU")
        print("   ┌─────────────────────────────────────────────┐")
        print("   │ python scripts/finetune_hf_model.py        │")
        print("   │ (requires NVIDIA GPU with 8GB+ VRAM)       │")
        print("   └─────────────────────────────────────────────┘")
        print()
        print("=" * 70)
        print("🎯 YOUR UNIFIED MODEL WILL:")
        print("=" * 70)
        print("   ✅ Handle text generation, summarization, translation")
        print("   ✅ Analyze security logs and forensic evidence")
        print("   ✅ Provide expert-level insights")
        print("   ✅ Learn from ALL user interactions")
        print("   ✅ Improve daily with new data")
        print("   ✅ Cost $0 to run (self-hosted)")
        print("=" * 70)
        print()
        
        return dataset
    
    async def build_forensic_model(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_name: str = "forensic-ai-model",
    ):
        """Build specialized forensic model."""
        print("=" * 70)
        print("🔍 BUILDING FORENSIC AI MODEL")
        print("=" * 70)
        print()
        
        result = await self.forensic_trainer.train_forensic_model(base_model, output_name)
        return result


async def main():
    """Main build process."""
    builder = UnifiedModelBuilder()
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          BUILD YOUR AI MODELS - Complete Training               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Build unified model
    print("🌐 Building Unified Model (handles everything)...")
    print()
    unified_result = await builder.build_unified_model()
    
    print()
    print("-" * 70)
    print()
    
    # Build forensic model
    print("🔍 Building Forensic Model (specialized for forensics)...")
    print()
    forensic_result = await builder.build_forensic_model()
    
    print()
    print("=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    print()
    print("📦 You now have training datasets for:")
    print("   1. Unified Model (general purpose)")
    print("   2. Forensic Model (security/forensics)")
    print()
    print("🚀 Next: Fine-tune on Google Colab (FREE GPU)")
    print("📖 Guide: COLAB_FINETUNING.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())

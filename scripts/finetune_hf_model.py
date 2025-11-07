"""
HuggingFace Fine-Tuning Script - Option B: Full Control
========================================================

This script fine-tunes an open-source model using your collected training data.

BENEFITS:
- 100% ownership - model is YOURS
- Full control - customize everything
- No vendor lock-in - run anywhere
- Cost effective - free after training
- Privacy - your data stays with you

REQUIREMENTS:
- GPU recommended (or use Google Colab free GPU)
- ~4-8GB VRAM for 3B models
- ~2-4 hours training time
- HuggingFace account (free)

MODELS TO CHOOSE:
- meta-llama/Llama-3.2-3B-Instruct (BEST - fast + smart)
- mistralai/Mistral-7B-Instruct-v0.3 (excellent quality)
- Qwen/Qwen2.5-7B-Instruct (very good)
- google/gemma-2-2b-it (fast, lightweight)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
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


class HuggingFaceFinetuner:
    """Fine-tune open-source models with LoRA for efficiency."""
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_name: str = "council-custom-model",
        hf_token: Optional[str] = None,
    ):
        self.base_model = base_model
        self.output_name = output_name
        self.hf_token = hf_token or os.getenv("HF_API_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🤗 HuggingFace Fine-Tuning Setup")
        print(f"📦 Base Model: {base_model}")
        print(f"🎯 Output: {output_name}")
        print(f"💻 Device: {self.device}")
        print()
    
    def load_training_data(self, data_path: str = "training_data/hf_finetune.jsonl") -> Dataset:
        """Load and prepare training data."""
        print(f"📂 Loading training data from {data_path}...")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"Training data not found at {data_path}\n"
                f"Run: python -m cli.app ensemble --input 'test' --models 3\n"
                f"Then: python -m cli.app learning-stats"
            )
        
        # Load JSONL data
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f"✅ Loaded {len(examples)} training examples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(examples)
        
        print(f"📊 Dataset info:")
        print(f"   Total examples: {len(dataset)}")
        print(f"   Features: {dataset.features}")
        print()
        
        return dataset
    
    def prepare_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA for efficient fine-tuning."""
        print(f"🔧 Loading model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            token=self.hf_token,
            trust_remote_code=True,
        )
        
        # Set padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with 8-bit quantization for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            token=self.hf_token,
            device_map="auto",
            load_in_8bit=True if self.device == "cuda" else False,
            trust_remote_code=True,
        )
        
        # Prepare for LoRA training
        if self.device == "cuda":
            model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration (efficient fine-tuning)
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "v_proj"],  # Apply to attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print(f"✅ Model loaded with LoRA")
        print(f"📊 Trainable parameters: {model.num_parameters(only_trainable=True):,}")
        print()
        
        return model, tokenizer
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Tokenize the training data."""
        print(f"🔤 Tokenizing dataset...")
        
        def format_example(example):
            """Format as instruction-following conversation."""
            messages = example["messages"]
            # Combine into single text
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
        
        # Format and tokenize
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        dataset = dataset.map(tokenize_function, batched=True)
        
        print(f"✅ Dataset tokenized")
        print()
        
        return dataset
    
    def train(
        self,
        dataset: Dataset,
        model,
        tokenizer,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ):
        """Train the model with your data."""
        print(f"🚀 Starting training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./models/{self.output_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True if self.device == "cuda" else False,
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
        
        # Train!
        print(f"⏳ Training in progress (this may take 2-4 hours)...")
        trainer.train()
        
        print(f"✅ Training complete!")
        print()
        
        return trainer
    
    def save_and_upload(self, model, tokenizer, upload: bool = True):
        """Save model locally and optionally upload to HuggingFace."""
        output_dir = f"./models/{self.output_name}"
        
        print(f"💾 Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Model saved locally")
        print()
        
        if upload:
            print(f"📤 Uploading to HuggingFace...")
            
            # Login
            login(token=self.hf_token)
            
            # Get username
            api = HfApi()
            user_info = api.whoami(token=self.hf_token)
            username = user_info["name"]
            
            repo_id = f"{username}/{self.output_name}"
            
            print(f"🤗 Creating repository: {repo_id}")
            
            # Create repo and upload
            api.create_repo(
                repo_id=repo_id,
                private=False,  # Make it public or set to True
                exist_ok=True,
            )
            
            # Upload model
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                commit_message="Initial fine-tuned model upload",
            )
            
            print(f"✅ Model uploaded!")
            print(f"🌐 View at: https://huggingface.co/{repo_id}")
            print()
            print(f"🎉 YOUR MODEL IS NOW LIVE!")
            print(f"📝 Use it with:")
            print(f"   from transformers import pipeline")
            print(f"   pipe = pipeline('text-generation', model='{repo_id}')")
            print()
        
        return output_dir


def main():
    """Main fine-tuning workflow."""
    print("=" * 70)
    print("HuggingFace Fine-Tuning - Create Your Own Model!")
    print("=" * 70)
    print()
    
    # Configuration
    BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Change to your preferred model
    OUTPUT_NAME = "council-custom-model"
    TRAINING_DATA = "training_data/hf_finetune.jsonl"
    
    # Options for base models:
    # - meta-llama/Llama-3.2-3B-Instruct (BEST - balanced)
    # - mistralai/Mistral-7B-Instruct-v0.3 (excellent, larger)
    # - Qwen/Qwen2.5-7B-Instruct (very good)
    # - google/gemma-2-2b-it (fastest, smaller)
    
    print(f"📋 Configuration:")
    print(f"   Base Model: {BASE_MODEL}")
    print(f"   Output Name: {OUTPUT_NAME}")
    print(f"   Training Data: {TRAINING_DATA}")
    print()
    
    # Check if GPU available
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected!")
        print("   Fine-tuning will be VERY slow on CPU.")
        print("   Consider using Google Colab (free GPU):")
        print("   https://colab.research.google.com/")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Initialize finetuner
    finetuner = HuggingFaceFinetuner(
        base_model=BASE_MODEL,
        output_name=OUTPUT_NAME,
    )
    
    # Step 1: Load training data
    dataset = finetuner.load_training_data(TRAINING_DATA)
    
    # Step 2: Prepare model and tokenizer
    model, tokenizer = finetuner.prepare_model_and_tokenizer()
    
    # Step 3: Tokenize dataset
    tokenized_dataset = finetuner.tokenize_dataset(dataset, tokenizer)
    
    # Step 4: Train
    trainer = finetuner.train(
        dataset=tokenized_dataset,
        model=model,
        tokenizer=tokenizer,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
    )
    
    # Step 5: Save and upload
    output_dir = finetuner.save_and_upload(
        model=model,
        tokenizer=tokenizer,
        upload=True,  # Set to False to only save locally
    )
    
    print("=" * 70)
    print("🎉 CONGRATULATIONS! YOUR MODEL IS READY!")
    print("=" * 70)
    print()
    print("What you now have:")
    print("  ✅ Your own custom AI model")
    print("  ✅ Trained on YOUR data")
    print("  ✅ 100% ownership")
    print("  ✅ Can run locally")
    print("  ✅ Can share/sell access")
    print()
    print("Next steps:")
    print("  1. Test your model: python scripts/use_your_model.py")
    print("  2. Keep collecting data for v2")
    print("  3. Re-train monthly with new examples")
    print("  4. Consider deploying on HuggingFace Spaces")
    print()


if __name__ == "__main__":
    main()

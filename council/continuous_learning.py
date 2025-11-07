"""
Continuous Learning & Custom Model Creation

This system:
1. Collects high-quality interactions from ensemble
2. Creates training datasets automatically
3. Fine-tunes custom models
4. Continuously improves based on feedback
5. Merges knowledge from multiple top models into YOUR model

Path to AGI: Your model learns and improves over time!
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import asyncio


@dataclass
class TrainingExample:
    """Single training example for fine-tuning."""
    messages: List[Dict[str, str]]
    response: str
    quality_score: float  # 0.0 to 1.0
    task_type: str
    models_agreed: bool  # Did multiple models give similar answer?
    user_feedback: Optional[int] = None  # 1-5 rating
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format."""
        return {
            "messages": self.messages + [
                {"role": "assistant", "content": self.response}
            ]
        }
    
    def to_hf_format(self) -> Dict[str, Any]:
        """Convert to HuggingFace format."""
        conversation = ""
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                conversation += f"<|system|>\n{content}\n"
            elif role == "user":
                conversation += f"<|user|>\n{content}\n"
        
        conversation += f"<|assistant|>\n{self.response}\n<|end|>"
        
        return {
            "text": conversation,
            "quality": self.quality_score,
            "task_type": self.task_type,
        }


class ContinuousLearner:
    """
    Continuously learn from interactions and create custom models.
    
    Workflow:
    1. Collect high-quality examples from ensemble
    2. Filter and curate training data
    3. Fine-tune custom model periodically
    4. Deploy and evaluate
    5. Repeat (continuous improvement)
    """
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.examples: List[TrainingExample] = []
        self.min_quality_threshold = 0.7  # Only learn from good examples
        self.min_examples_for_finetune = 100
        
        self._load_examples()
    
    def _load_examples(self):
        """Load existing training examples."""
        examples_file = self.data_dir / "examples.jsonl"
        if examples_file.exists():
            with open(examples_file) as f:
                for line in f:
                    data = json.loads(line)
                    self.examples.append(TrainingExample(**data))
    
    def _save_example(self, example: TrainingExample):
        """Append example to training data."""
        examples_file = self.data_dir / "examples.jsonl"
        with open(examples_file, 'a') as f:
            f.write(json.dumps(example.to_dict()) + "\n")
    
    def collect_example(
        self,
        messages: List[Dict[str, str]],
        response: str,
        quality_score: float,
        task_type: str,
        models_agreed: bool = False,
        user_feedback: Optional[int] = None,
    ):
        """
        Collect a training example from interaction.
        
        This is called after each ensemble query to learn from it.
        """
        # Only collect high-quality examples
        if quality_score < self.min_quality_threshold:
            return
        
        # Bonus points if multiple models agreed
        if models_agreed:
            quality_score = min(quality_score + 0.1, 1.0)
        
        # Bonus for positive user feedback
        if user_feedback and user_feedback >= 4:
            quality_score = min(quality_score + 0.1, 1.0)
        
        example = TrainingExample(
            messages=messages,
            response=response,
            quality_score=quality_score,
            task_type=task_type,
            models_agreed=models_agreed,
            user_feedback=user_feedback,
        )
        
        self.examples.append(example)
        self._save_example(example)
    
    def get_training_dataset(
        self,
        min_quality: float = 0.8,
        task_types: Optional[List[str]] = None,
    ) -> List[TrainingExample]:
        """
        Get curated training dataset.
        
        Args:
            min_quality: Minimum quality score to include
            task_types: Filter by task types (None = all)
        
        Returns:
            Filtered list of high-quality examples
        """
        filtered = [
            ex for ex in self.examples
            if ex.quality_score >= min_quality
        ]
        
        if task_types:
            filtered = [ex for ex in filtered if ex.task_type in task_types]
        
        return filtered
    
    def export_for_finetuning(
        self,
        output_file: str,
        format: str = "openai",
        min_quality: float = 0.8,
    ):
        """
        Export training data for fine-tuning.
        
        Args:
            output_file: Where to save
            format: "openai" or "huggingface"
            min_quality: Minimum quality threshold
        """
        dataset = self.get_training_dataset(min_quality=min_quality)
        
        if len(dataset) < self.min_examples_for_finetune:
            print(f"⚠️  Only {len(dataset)} examples. Need {self.min_examples_for_finetune} for fine-tuning.")
            return False
        
        output_path = Path(output_file)
        
        if format == "openai":
            # OpenAI JSONL format
            with open(output_path, 'w') as f:
                for ex in dataset:
                    f.write(json.dumps(ex.to_openai_format()) + "\n")
        
        elif format == "huggingface":
            # HuggingFace dataset format
            with open(output_path, 'w') as f:
                for ex in dataset:
                    f.write(json.dumps(ex.to_hf_format()) + "\n")
        
        print(f"✅ Exported {len(dataset)} examples to {output_path}")
        print(f"📊 Quality distribution:")
        print(f"   Excellent (>0.9): {sum(1 for ex in dataset if ex.quality_score > 0.9)}")
        print(f"   Good (0.8-0.9): {sum(1 for ex in dataset if 0.8 <= ex.quality_score <= 0.9)}")
        print(f"   Average task types: {len(set(ex.task_type for ex in dataset))}")
        
        return True
    
    async def finetune_openai(
        self,
        model: str = "gpt-3.5-turbo",
        training_file: str = "training_data/openai_finetune.jsonl",
    ) -> Dict[str, Any]:
        """
        Fine-tune an OpenAI model on your data.
        
        Creates YOUR custom GPT model!
        """
        try:
            import openai
            import os
            
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Export data
            self.export_for_finetuning(training_file, format="openai")
            
            # Upload training file
            print("📤 Uploading training data...")
            with open(training_file, "rb") as f:
                file_response = client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            # Create fine-tuning job
            print(f"🚀 Starting fine-tuning job...")
            job = client.fine_tuning.jobs.create(
                training_file=file_response.id,
                model=model,
            )
            
            print(f"✅ Fine-tuning job created: {job.id}")
            print(f"📊 Monitor at: https://platform.openai.com/finetune/{job.id}")
            
            return {
                "job_id": job.id,
                "status": job.status,
                "model": model,
                "training_file": file_response.id,
            }
            
        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            return {"error": str(e)}
    
    async def finetune_huggingface(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_name: str = "council-custom-model",
        training_file: str = "training_data/hf_finetune.jsonl",
    ) -> Dict[str, Any]:
        """
        Fine-tune a HuggingFace model on your data.
        
        Creates YOUR custom open-source model!
        """
        try:
            # Export data
            if not self.export_for_finetuning(training_file, format="huggingface"):
                return {"error": "Not enough training data"}
            
            print(f"🤗 Fine-tuning {base_model}...")
            print(f"📝 This will create: {output_name}")
            print(f"⚠️  Note: Fine-tuning requires GPU and ~2-4 hours")
            print(f"📖 Follow guide: https://huggingface.co/docs/transformers/training")
            
            # In production, would use:
            # - AutoTrain
            # - Transformers Trainer
            # - LoRA/QLoRA for efficient fine-tuning
            
            return {
                "base_model": base_model,
                "output_name": output_name,
                "training_file": training_file,
                "next_steps": [
                    "Install: pip install transformers accelerate peft",
                    "Run: python scripts/finetune_model.py",
                    "Upload: huggingface-cli upload",
                ],
            }
            
        except Exception as e:
            print(f"❌ Fine-tuning setup failed: {e}")
            return {"error": str(e)}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about continuous learning progress."""
        if not self.examples:
            return {"message": "No training data collected yet"}
        
        quality_examples = [ex for ex in self.examples if ex.quality_score >= 0.8]
        
        task_distribution = {}
        for ex in self.examples:
            task_distribution[ex.task_type] = task_distribution.get(ex.task_type, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "high_quality_examples": len(quality_examples),
            "ready_for_finetuning": len(quality_examples) >= self.min_examples_for_finetune,
            "examples_needed": max(0, self.min_examples_for_finetune - len(quality_examples)),
            "avg_quality": sum(ex.quality_score for ex in self.examples) / len(self.examples),
            "task_distribution": task_distribution,
            "examples_with_feedback": sum(1 for ex in self.examples if ex.user_feedback),
        }


class KnowledgeDistillation:
    """
    Distill knowledge from large models into smaller custom model.
    
    This merges GPT-4, Claude, Gemini knowledge into YOUR model!
    """
    
    def __init__(self, learner: ContinuousLearner):
        self.learner = learner
    
    async def distill_from_ensemble(
        self,
        tasks: List[str],
        ensemble,
        task_type: str = "general",
    ):
        """
        Query ensemble on many tasks and learn from responses.
        
        This is knowledge distillation: teach small model what big models know!
        """
        print(f"🎓 Distilling knowledge from ensemble...")
        print(f"📚 Processing {len(tasks)} tasks...")
        
        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{len(tasks)}] {task[:50]}...")
            
            messages = [{"role": "user", "content": task}]
            
            # Get ensemble response
            result = await ensemble.ensemble_query(
                messages=messages,
                task_type=task_type,
                num_models=3,
            )
            
            # Collect as training example
            self.learner.collect_example(
                messages=messages,
                response=result["answer"],
                quality_score=result["confidence"],
                task_type=task_type,
                models_agreed=result.get("consensus", False),
            )
            
            # Rate limit
            await asyncio.sleep(1)
        
        print(f"✅ Distilled {len(tasks)} examples")
        print(f"📊 Total training data: {len(self.learner.examples)}")


# Global learner instance
_learner = None

def get_learner() -> ContinuousLearner:
    """Get global continuous learner."""
    global _learner
    if _learner is None:
        _learner = ContinuousLearner()
    return _learner

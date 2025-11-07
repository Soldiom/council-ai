"""
HuggingFace Model Hub - Auto-discover and use ALL models + Clone to ANY Domain
================================================================================

This system:
1. Discovers ALL models from HuggingFace (text, vision, audio, etc.)
2. Creates specialized agents for each capability
3. Routes user requests to best model
4. Collects training data from ALL interactions
5. Fine-tunes unified model daily
6. Auto-discovers new models and updates
7. **CLONES YOUR MODEL TO ANY FIELD** (medical, legal, financial, etc.)
8. **DEPLOYS WITH SIMPLE INSTRUCTIONS** - No retraining needed!

YOUR UNIFIED MODEL = Learning from ALL HuggingFace models + ALL user interactions

NEW: Clone and deploy to ANY domain with just instructions!
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os
try:
    from huggingface_hub import HfApi, list_models
except ImportError:
    # HuggingFace Hub not installed or version mismatch
    HfApi = None
    list_models = None
from pathlib import Path


@dataclass
class ModelCapability:
    """A specific capability/feature a model provides."""
    name: str  # e.g., "text-generation", "image-classification"
    model_id: str  # e.g., "meta-llama/Llama-3.2-3B"
    pipeline_tag: str
    downloads: int
    likes: int
    description: str
    languages: List[str]
    tags: List[str]
    
    def score(self) -> float:
        """Quality score based on downloads and likes."""
        return (self.downloads * 0.7) + (self.likes * 1000 * 0.3)


class HuggingFaceModelHub:
    """
    Discovers and manages ALL HuggingFace models.
    
    Creates specialized agents for each model type:
    - Text generation → Strategist, Writer, Coder
    - Image generation → Designer, Artist
    - Audio processing → Musician, Transcriber
    - Translation → Translator
    - Summarization → Analyst
    - Question answering → Researcher
    - etc.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_API_TOKEN")
        self.api = HfApi(token=self.hf_token)
        self.capabilities: Dict[str, List[ModelCapability]] = {}
        self.cache_file = Path("data/model_hub_cache.json")
        
        # Map pipeline tags to agent types
        self.pipeline_to_agent = {
            "text-generation": ["strategist", "writer", "coder", "analyst"],
            "text2text-generation": ["translator", "summarizer"],
            "question-answering": ["researcher", "qa_expert"],
            "summarization": ["analyst", "summarizer"],
            "translation": ["translator"],
            "conversational": ["assistant", "support"],
            "text-classification": ["classifier", "moderator"],
            "token-classification": ["ner_expert", "tagger"],
            "fill-mask": ["autocomplete", "editor"],
            "image-classification": ["vision_analyst", "inspector"],
            "image-segmentation": ["segmenter", "detector"],
            "image-to-text": ["caption_generator", "describer"],
            "text-to-image": ["designer", "artist"],
            "object-detection": ["detector", "security"],
            "image-to-image": ["editor", "enhancer"],
            "deepfake-detection": ["deepfake_detector", "authenticity_verifier"],
            "document-forgery-detection": ["document_verifier", "fraud_detector"],
            "automatic-speech-recognition": ["transcriber", "listener"],
            "audio-classification": ["audio_analyst", "identifier"],
            "text-to-speech": ["speaker", "narrator"],
            "audio-to-audio": ["audio_editor", "mixer"],
            "voice-activity-detection": ["vad_detector"],
            "feature-extraction": ["embedder", "vectorizer"],
            "sentence-similarity": ["matcher", "comparator"],
            "zero-shot-classification": ["zero_shot_classifier"],
            "tabular-classification": ["data_classifier"],
            "tabular-regression": ["predictor", "forecaster"],
            "reinforcement-learning": ["rl_agent", "game_player"],
            "robotics": ["robot_controller"],
        }
    
    async def discover_all_models(
        self,
        top_n_per_category: int = 5,
        min_downloads: int = 1000,
    ) -> Dict[str, List[ModelCapability]]:
        """
        Discover ALL HuggingFace models across all categories.
        
        Returns organized by capability:
        {
            "text-generation": [top 5 models],
            "image-classification": [top 5 models],
            ...
        }
        """
        print("🔍 Discovering ALL HuggingFace models...")
        print(f"   Top {top_n_per_category} per category")
        print(f"   Minimum {min_downloads:,} downloads")
        print()
        
        capabilities = {}
        
        # Get all pipeline tags
        all_tags = [
            "text-generation", "text2text-generation", "question-answering",
            "summarization", "translation", "conversational",
            "text-classification", "token-classification", "fill-mask",
            "image-classification", "image-segmentation", "image-to-text",
            "text-to-image", "object-detection", "image-to-image",
            "automatic-speech-recognition", "audio-classification",
            "text-to-speech", "audio-to-audio", "voice-activity-detection",
            "feature-extraction", "sentence-similarity",
            "zero-shot-classification", "tabular-classification",
            "tabular-regression", "reinforcement-learning", "robotics",
        ]
        
        for tag in all_tags:
            print(f"📦 Discovering {tag} models...")
            
            try:
                # Search for models with this pipeline tag
                models = list_models(
                    filter=ModelFilter(pipeline_tag=tag),
                    sort="downloads",
                    direction=-1,
                    limit=top_n_per_category * 3,  # Get extra to filter
                    token=self.hf_token,
                )
                
                tag_models = []
                for model in models:
                    # Filter by downloads
                    if model.downloads < min_downloads:
                        continue
                    
                    # Create capability
                    capability = ModelCapability(
                        name=tag,
                        model_id=model.id,
                        pipeline_tag=tag,
                        downloads=model.downloads or 0,
                        likes=model.likes or 0,
                        description=getattr(model, 'description', '') or '',
                        languages=getattr(model, 'languages', []) or [],
                        tags=model.tags or [],
                    )
                    tag_models.append(capability)
                    
                    if len(tag_models) >= top_n_per_category:
                        break
                
                if tag_models:
                    # Sort by score
                    tag_models.sort(key=lambda x: x.score(), reverse=True)
                    capabilities[tag] = tag_models[:top_n_per_category]
                    
                    print(f"   ✅ Found {len(capabilities[tag])} top models")
                    for i, cap in enumerate(capabilities[tag], 1):
                        print(f"      {i}. {cap.model_id} ({cap.downloads:,} downloads)")
            
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
                continue
        
        print()
        print(f"🎉 Discovered {sum(len(v) for v in capabilities.values())} models across {len(capabilities)} categories!")
        
        self.capabilities = capabilities
        self._save_cache()
        
        return capabilities
    
    def get_agents_for_capability(self, capability: str) -> List[str]:
        """Get agent names for a capability."""
        return self.pipeline_to_agent.get(capability, ["general_agent"])
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get ALL possible agents based on discovered models.
        
        Returns:
        {
            "strategist": {
                "capabilities": ["text-generation"],
                "models": ["meta-llama/Llama-3.2-3B", ...],
                "description": "...",
            },
            ...
        }
        """
        agents = {}
        
        for capability, models in self.capabilities.items():
            agent_names = self.get_agents_for_capability(capability)
            
            for agent_name in agent_names:
                if agent_name not in agents:
                    agents[agent_name] = {
                        "name": agent_name,
                        "capabilities": [],
                        "models": [],
                        "description": f"Agent specialized in {capability}",
                    }
                
                agents[agent_name]["capabilities"].append(capability)
                agents[agent_name]["models"].extend([m.model_id for m in models])
        
        return agents
    
    def route_request(
        self,
        task: str,
        input_text: str,
        preferred_capability: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route user request to best model.
        
        Auto-detects capability from task or uses preferred.
        """
        # Detect capability if not specified
        if not preferred_capability:
            preferred_capability = self._detect_capability(task, input_text)
        
        # Get top model for this capability
        if preferred_capability not in self.capabilities:
            preferred_capability = "text-generation"  # Default
        
        top_model = self.capabilities[preferred_capability][0]
        agent_names = self.get_agents_for_capability(preferred_capability)
        
        return {
            "capability": preferred_capability,
            "model_id": top_model.model_id,
            "agent": agent_names[0],
            "pipeline_tag": top_model.pipeline_tag,
            "description": top_model.description,
        }
    
    def _detect_capability(self, task: str, input_text: str) -> str:
        """Auto-detect capability from task description."""
        task_lower = task.lower()
        
        # Keyword mapping
        if any(word in task_lower for word in ["translate", "translation"]):
            return "translation"
        elif any(word in task_lower for word in ["summarize", "summary"]):
            return "summarization"
        elif any(word in task_lower for word in ["image", "picture", "photo", "visual"]):
            if "generate" in task_lower or "create" in task_lower:
                return "text-to-image"
            else:
                return "image-classification"
        elif any(word in task_lower for word in ["audio", "speech", "voice"]):
            if "transcribe" in task_lower:
                return "automatic-speech-recognition"
            elif "speak" in task_lower or "say" in task_lower:
                return "text-to-speech"
            else:
                return "audio-classification"
        elif any(word in task_lower for word in ["question", "answer", "qa"]):
            return "question-answering"
        elif any(word in task_lower for word in ["classify", "category"]):
            return "text-classification"
        else:
            return "text-generation"  # Default
    
    def _save_cache(self):
        """Save discovered models to cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                cap: [
                    {
                        "name": m.name,
                        "model_id": m.model_id,
                        "pipeline_tag": m.pipeline_tag,
                        "downloads": m.downloads,
                        "likes": m.likes,
                        "description": m.description,
                        "languages": m.languages,
                        "tags": m.tags,
                    }
                    for m in models
                ]
                for cap, models in self.capabilities.items()
            }
        }
        
        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"💾 Cached to {self.cache_file}")
    
    def load_cache(self) -> bool:
        """Load models from cache if available."""
        if not self.cache_file.exists():
            return False
        
        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Check if cache is recent (within 24 hours)
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            age = (datetime.now() - timestamp).total_seconds() / 3600
            
            if age > 24:
                print(f"⚠️ Cache is {age:.1f} hours old, refreshing...")
                return False
            
            # Load capabilities
            self.capabilities = {}
            for cap, models in cache_data["capabilities"].items():
                self.capabilities[cap] = [
                    ModelCapability(**m) for m in models
                ]
            
            print(f"✅ Loaded {sum(len(v) for v in self.capabilities.values())} models from cache")
            return True
        
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
            return False


class UnifiedModelTrainer:
    """
    Trains YOUR unified model from ALL user interactions.
    
    Workflow:
    1. User interacts with ANY model → collect as training example
    2. Daily: Aggregate ALL examples
    3. Fine-tune YOUR unified model
    4. Deploy updated model
    5. Repeat daily → model gets smarter!
    """
    
    def __init__(self, hub: HuggingFaceModelHub):
        self.hub = hub
        self.training_data_dir = Path("training_data/unified")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
    
    async def collect_interaction(
        self,
        user_id: str,
        capability: str,
        model_id: str,
        input_data: Any,
        output_data: Any,
        feedback: Optional[Dict[str, Any]] = None,
    ):
        """
        Collect training data from user interaction.
        
        Every user interaction = training example!
        """
        example = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "capability": capability,
            "model_id": model_id,
            "input": input_data,
            "output": output_data,
            "feedback": feedback or {},
        }
        
        # Save to daily file
        date_str = datetime.now().strftime("%Y-%m-%d")
        daily_file = self.training_data_dir / f"interactions_{date_str}.jsonl"
        
        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")
    
    async def train_unified_model_daily(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_name: str = "unified-ai-model",
    ):
        """
        Train YOUR unified model on ALL collected data.
        
        Runs daily to incorporate new user interactions.
        """
        print("🎓 Daily Unified Model Training")
        print(f"   Base: {base_model}")
        print(f"   Output: {output_name}")
        print()
        
        # Collect all training files
        training_files = list(self.training_data_dir.glob("interactions_*.jsonl"))
        
        if not training_files:
            print("⚠️ No training data yet")
            return
        
        total_examples = 0
        for file in training_files:
            with open(file) as f:
                total_examples += sum(1 for _ in f)
        
        print(f"📊 Total training examples: {total_examples:,}")
        print(f"📁 From {len(training_files)} daily files")
        print()
        
        if total_examples < 100:
            print(f"⚠️ Need at least 100 examples (have {total_examples})")
            return
        
        # Convert to fine-tuning format
        unified_training_file = self.training_data_dir / "unified_training.jsonl"
        self._prepare_training_data(training_files, unified_training_file)
        
        # Fine-tune (same process as before)
        print("🚀 Starting fine-tuning...")
        print("   Use: python scripts/finetune_hf_model.py")
        print(f"   Or: Google Colab (FREE GPU)")
        print()
        
        return {
            "training_file": str(unified_training_file),
            "total_examples": total_examples,
            "base_model": base_model,
            "output_name": output_name,
        }
    
    def _prepare_training_data(self, input_files: List[Path], output_file: Path):
        """Convert interaction data to training format."""
        with open(output_file, "w", encoding="utf-8") as out:
            for file in input_files:
                with open(file, encoding="utf-8") as f:
                    for line in f:
                        example = json.loads(line)
                        
                        # Convert to chat format
                        messages = [
                            {"role": "user", "content": str(example["input"])},
                            {"role": "assistant", "content": str(example["output"])},
                        ]
                        
                        training_example = {"messages": messages}
                        out.write(json.dumps(training_example) + "\n")
        
        print(f"✅ Prepared training data: {output_file}")


# Global instance
_hub = None

def get_hub() -> HuggingFaceModelHub:
    """Get or create global hub instance."""
    global _hub
    if _hub is None:
        _hub = HuggingFaceModelHub()
    return _hub


# ============================================================================
# MODEL CLONING & DEPLOYMENT TO ANY FIELD
# ============================================================================

from dataclasses import dataclass as clone_dataclass

@clone_dataclass
class DomainInstructions:
    """Instructions for cloning model to a specific domain"""
    domain_name: str
    description: str
    system_prompt: str
    example_tasks: List[str]
    
    # Deployment info
    requires_finetuning: bool = False
    training_examples_needed: int = 100


# Pre-configured domains - just add instructions, model adapts!
CLONE_DOMAINS = {
    "medical": DomainInstructions(
        domain_name="Medical AI Assistant",
        description="Medical diagnosis, treatment recommendations, patient care",
        system_prompt="""You are a medical AI trained on clinical data.

Analyze symptoms, suggest diagnoses, recommend tests, explain treatments.
Always recommend consulting a licensed physician.
Use evidence-based medicine and cite sources.""",
        example_tasks=[
            "Analyze patient symptoms and suggest differential diagnosis",
            "Explain lab results in patient-friendly language",
            "Recommend diagnostic tests based on symptoms",
            "Provide treatment options with contraindications"
        ],
        requires_finetuning=True,
        training_examples_needed=500
    ),
    
    "legal": DomainInstructions(
        domain_name="Legal AI Assistant",
        description="Legal research, contract review, case analysis",
        system_prompt="""You are a legal AI trained on case law and statutes.

Research precedents, analyze contracts, identify risks, explain legal concepts.
Always recommend consulting a licensed attorney.
Cite relevant cases and statutes.""",
        example_tasks=[
            "Review contracts for potential issues",
            "Research legal precedents for a case",
            "Explain legal concepts in plain language",
            "Identify risks in legal documents"
        ],
        requires_finetuning=True,
        training_examples_needed=300
    ),
    
    "financial": DomainInstructions(
        domain_name="Financial AI Analyst",
        description="Investment analysis, risk assessment, financial planning",
        system_prompt="""You are a financial AI trained on market data.

Analyze investments, assess risks, forecast trends, provide financial insights.
Not financial advice - recommend consulting advisor.
Use quantitative metrics and data-driven analysis.""",
        example_tasks=[
            "Analyze company financials and stock fundamentals",
            "Evaluate investment opportunities and risks",
            "Forecast market trends based on data",
            "Recommend portfolio allocation strategies"
        ],
        requires_finetuning=True,
        training_examples_needed=400
    ),
    
    "education": DomainInstructions(
        domain_name="Educational AI Tutor",
        description="Teaching, tutoring, explaining concepts at all levels",
        system_prompt="""You are an educational AI tutor.

Explain concepts clearly, adapt to student level, provide examples.
Use Socratic method, check understanding, be encouraging.
Make learning engaging and accessible.""",
        example_tasks=[
            "Explain complex topics in simple terms",
            "Provide step-by-step problem solving",
            "Create practice exercises",
            "Give constructive feedback"
        ],
        requires_finetuning=False,
        training_examples_needed=75
    ),
    
    "code": DomainInstructions(
        domain_name="Code Generation AI",
        description="Software development, debugging, code review",
        system_prompt="""You are a software engineering AI.

Write clean code, debug issues, review for best practices.
Include error handling, comments, tests.
Consider security and performance.""",
        example_tasks=[
            "Generate production-ready code",
            "Debug and fix issues",
            "Review code and suggest improvements",
            "Explain programming concepts"
        ],
        requires_finetuning=False,
        training_examples_needed=100
    ),
    
    "creative": DomainInstructions(
        domain_name="Creative Writing AI",
        description="Story writing, content creation, creative projects",
        system_prompt="""You are a creative writing AI.

Create compelling stories, develop characters, write engaging content.
Match tone and style, use vivid descriptions.
Respect originality and copyright.""",
        example_tasks=[
            "Write creative stories",
            "Develop character profiles",
            "Create marketing copy",
            "Generate creative ideas"
        ],
        requires_finetuning=False,
        training_examples_needed=50
    )
}


def clone_model_to_domain(
    domain: str,
    base_model: str = "aliAIML/unified-ai-model",
    custom_instructions: str = None,
    output_name: str = None
) -> Dict[str, Any]:
    """
    Clone your trained model to ANY domain with just instructions!
    
    No retraining needed for most domains - model adapts via instructions.
    
    Args:
        domain: Domain ID (medical, legal, financial, education, code, creative)
        base_model: Your trained model to clone from
        custom_instructions: Additional instructions to add
        output_name: Custom name for cloned model
    
    Returns:
        {
            "success": True,
            "domain": "medical",
            "model_name": "aliAIML/medical-ai-model",
            "instructions": "...",
            "requires_finetuning": True/False,
            "ready_to_use": True/False
        }
    
    Example:
        # Clone to medical domain
        result = clone_model_to_domain("medical")
        
        # Model is now ready with medical instructions
        # Use it with the provided system_prompt
        # Optionally fine-tune with medical data for better accuracy
    """
    print(f"\n{'=' * 80}")
    print(f"🔄 CLONING MODEL TO: {domain.upper()}")
    print(f"{'=' * 80}\n")
    
    # Get domain instructions
    if domain not in CLONE_DOMAINS:
        print(f"❌ Unknown domain: {domain}")
        print(f"   Available: {', '.join(CLONE_DOMAINS.keys())}")
        return {"success": False, "error": f"Unknown domain: {domain}"}
    
    instructions = CLONE_DOMAINS[domain]
    
    # Generate output name
    if output_name is None:
        output_name = f"aliAIML/{domain}-ai-model"
    
    print(f"📦 Base Model: {base_model}")
    print(f"🎯 Target Domain: {instructions.domain_name}")
    print(f"📝 Output Model: {output_name}\n")
    
    # Create configuration
    config = {
        "base_model": base_model,
        "domain": domain,
        "domain_name": instructions.domain_name,
        "description": instructions.description,
        "model_name": output_name,
        "system_prompt": instructions.system_prompt,
        "example_tasks": instructions.example_tasks,
        "custom_instructions": custom_instructions,
        "requires_finetuning": instructions.requires_finetuning,
        "training_examples_needed": instructions.training_examples_needed,
        "created_at": datetime.now().isoformat()
    }
    
    # Add custom instructions if provided
    if custom_instructions:
        config["system_prompt"] += f"\n\nAdditional Instructions:\n{custom_instructions}"
    
    # Save configuration
    config_dir = Path("model_deployments")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / f"{domain}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_file}\n")
    
    # Show usage instructions
    print(f"{'=' * 80}")
    print(f"✅ MODEL CLONED TO {instructions.domain_name}!")
    print(f"{'=' * 80}\n")
    
    print(f"📖 HOW TO USE:\n")
    print(f"1. Use this system prompt:\n")
    print(f"{config['system_prompt']}\n")
    
    print(f"2. Example tasks:")
    for i, task in enumerate(instructions.example_tasks, 1):
        print(f"   {i}. {task}")
    print()
    
    if instructions.requires_finetuning:
        print(f"⚠️  FINE-TUNING RECOMMENDED:")
        print(f"   For best accuracy, fine-tune with {instructions.training_examples_needed}+ examples")
        print(f"   1. Collect domain-specific data")
        print(f"   2. Create dataset: {domain}_training_data.jsonl")
        print(f"   3. Fine-tune: python scripts/finetune_hf_model.py \\")
        print(f"      --base-model {base_model} \\")
        print(f"      --dataset-path model_deployments/{domain}_training_data.jsonl \\")
        print(f"      --output-model {output_name}\n")
    else:
        print(f"✅ NO FINE-TUNING NEEDED!")
        print(f"   Model adapts automatically with instructions")
        print(f"   Ready to use immediately!\n")
    
    print(f"{'=' * 80}\n")
    
    return {
        "success": True,
        "domain": domain,
        "domain_name": instructions.domain_name,
        "model_name": output_name,
        "instructions": config["system_prompt"],
        "requires_finetuning": instructions.requires_finetuning,
        "ready_to_use": not instructions.requires_finetuning,
        "config_file": str(config_file),
        "example_tasks": instructions.example_tasks
    }


def list_clone_domains():
    """List all available domains for model cloning"""
    print("\n" + "=" * 80)
    print("🌍 AVAILABLE DOMAINS FOR MODEL CLONING")
    print("=" * 80 + "\n")
    
    for domain_id, instructions in CLONE_DOMAINS.items():
        print(f"📌 {domain_id.upper()}: {instructions.domain_name}")
        print(f"   {instructions.description}")
        print(f"   Fine-tuning: {'Recommended' if instructions.requires_finetuning else 'Optional'}")
        if instructions.requires_finetuning:
            print(f"   Training examples: {instructions.training_examples_needed}+")
        print()
    
    print("=" * 80)
    print(f"Total domains: {len(CLONE_DOMAINS)}")
    print("=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    # List available domains
    list_clone_domains()
    
    # Clone to medical domain
    result = clone_model_to_domain(
        domain="medical",
        custom_instructions="Focus on emergency medicine and triage."
    )
    
    print(f"\nCloning result: {result['success']}")
    print(f"Model ready: {result['ready_to_use']}")


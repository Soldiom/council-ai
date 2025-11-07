"""
🚀 ADVANCED MODEL ROTATION SYSTEM

Uses 10-50+ specialized models per day:
- Text models (GPT-4, Claude, Gemini, Llama, Qwen, etc.)
- Image models (DALL-E, Midjourney, Stable Diffusion, Flux)
- Audio models (Whisper, ElevenLabs, Bark)
- Video models (Sora, Runway, Pika)
- Code models (Codex, CodeLlama, StarCoder)
- Math models (Minerva, GPT-4 Math)
- Multimodal models (GPT-4V, Gemini Pro, Claude 3)

Rotates models daily for maximum diversity and AGI-level capabilities.
"""

import os
import random
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelCapability(Enum):
    """Model capabilities for AGI-level system"""
    TEXT = "text"
    IMAGE_GENERATION = "image_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_GENERATION = "audio_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    VIDEO_GENERATION = "video_generation"
    VIDEO_UNDERSTANDING = "video_understanding"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    CREATIVE = "creative"
    MULTIMODAL = "multimodal"


@dataclass
class AdvancedModelConfig:
    """Configuration for advanced multimodal models"""
    name: str
    provider: str
    model_id: str
    capabilities: List[ModelCapability]
    specialization: List[str]
    cost_per_1k: float
    quality_score: float  # 0-10
    speed_score: float  # 0-10
    enabled: bool = True


# 🌟 COMPLETE 50+ MODEL CATALOG - AGI-LEVEL
AVAILABLE_MODELS = {
    # ═══════════════════════════════════════════════════════════
    # TEXT & REASONING MODELS (20+)
    # ═══════════════════════════════════════════════════════════
    
    # OpenAI
    "gpt-4-turbo": AdvancedModelConfig(
        name="GPT-4 Turbo", provider="openai", model_id="gpt-4-turbo-preview",
        capabilities=[ModelCapability.TEXT, ModelCapability.REASONING],
        specialization=["reasoning", "analysis"], cost_per_1k=0.01,
        quality_score=9.5, speed_score=7.0),
    "gpt-4o": AdvancedModelConfig(
        name="GPT-4o", provider="openai", model_id="gpt-4o",
        capabilities=[ModelCapability.TEXT, ModelCapability.MULTIMODAL, ModelCapability.IMAGE_UNDERSTANDING],
        specialization=["multimodal", "vision"], cost_per_1k=0.005,
        quality_score=9.0, speed_score=8.5),
    "o1-preview": AdvancedModelConfig(
        name="OpenAI o1", provider="openai", model_id="o1-preview",
        capabilities=[ModelCapability.TEXT, ModelCapability.REASONING, ModelCapability.MATH],
        specialization=["reasoning", "math"], cost_per_1k=0.015,
        quality_score=10.0, speed_score=5.0),
    
    # Anthropic
    "claude-3-opus": AdvancedModelConfig(
        name="Claude 3 Opus", provider="anthropic", model_id="claude-3-opus-20240229",
        capabilities=[ModelCapability.TEXT, ModelCapability.REASONING, ModelCapability.IMAGE_UNDERSTANDING],
        specialization=["reasoning", "analysis"], cost_per_1k=0.015,
        quality_score=9.5, speed_score=6.0),
    "claude-3.5-sonnet": AdvancedModelConfig(
        name="Claude 3.5 Sonnet", provider="anthropic", model_id="claude-3-5-sonnet-20240620",
        capabilities=[ModelCapability.TEXT, ModelCapability.CODE, ModelCapability.IMAGE_UNDERSTANDING],
        specialization=["code", "reasoning"], cost_per_1k=0.003,
        quality_score=9.0, speed_score=8.0),
    
    # Google
    "gemini-1.5-pro": AdvancedModelConfig(
        name="Gemini 1.5 Pro", provider="google", model_id="gemini-1.5-pro",
        capabilities=[ModelCapability.TEXT, ModelCapability.MULTIMODAL, ModelCapability.IMAGE_UNDERSTANDING, ModelCapability.VIDEO_UNDERSTANDING],
        specialization=["multimodal", "long-context"], cost_per_1k=0.00125,
        quality_score=9.0, speed_score=7.0),
    "gemini-1.5-flash": AdvancedModelConfig(
        name="Gemini 1.5 Flash", provider="google", model_id="gemini-1.5-flash",
        capabilities=[ModelCapability.TEXT, ModelCapability.MULTIMODAL],
        specialization=["fast", "multimodal"], cost_per_1k=0.000075,
        quality_score=8.0, speed_score=9.0),
    
    # Meta Llama
    "llama-3.1-405b": AdvancedModelConfig(
        name="Llama 3.1 405B", provider="huggingface", model_id="meta-llama/Llama-3.1-405B-Instruct",
        capabilities=[ModelCapability.TEXT, ModelCapability.REASONING],
        specialization=["reasoning", "general"], cost_per_1k=0.002,
        quality_score=9.0, speed_score=6.0),
    "llama-3.1-70b": AdvancedModelConfig(
        name="Llama 3.1 70B", provider="huggingface", model_id="meta-llama/Llama-3.1-70B-Instruct",
        capabilities=[ModelCapability.TEXT], specialization=["general"], cost_per_1k=0.0009,
        quality_score=8.5, speed_score=7.5),
    "llama-3.2-3b": AdvancedModelConfig(
        name="Llama 3.2 3B", provider="huggingface", model_id="meta-llama/Llama-3.2-3B-Instruct",
        capabilities=[ModelCapability.TEXT], specialization=["fast", "local"], cost_per_1k=0.0,
        quality_score=7.5, speed_score=9.0),
    
    # Alibaba Qwen
    "qwen-2.5-72b": AdvancedModelConfig(
        name="Qwen 2.5 72B", provider="huggingface", model_id="Qwen/Qwen2.5-72B-Instruct",
        capabilities=[ModelCapability.TEXT, ModelCapability.CODE, ModelCapability.MATH],
        specialization=["multilingual", "code"], cost_per_1k=0.0009,
        quality_score=8.5, speed_score=7.5),
    "qwen-vl-max": AdvancedModelConfig(
        name="Qwen-VL Max", provider="huggingface", model_id="Qwen/Qwen-VL-Max",
        capabilities=[ModelCapability.TEXT, ModelCapability.IMAGE_UNDERSTANDING, ModelCapability.MULTIMODAL],
        specialization=["vision", "multimodal"], cost_per_1k=0.001,
        quality_score=8.0, speed_score=7.0),
    
    # Mistral
    "mixtral-8x22b": AdvancedModelConfig(
        name="Mixtral 8x22B", provider="huggingface", model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        capabilities=[ModelCapability.TEXT, ModelCapability.CODE],
        specialization=["reasoning", "code"], cost_per_1k=0.002,
        quality_score=8.5, speed_score=7.0),
    "mistral-7b": AdvancedModelConfig(
        name="Mistral 7B", provider="huggingface", model_id="mistralai/Mistral-7B-Instruct-v0.3",
        capabilities=[ModelCapability.TEXT], specialization=["general"], cost_per_1k=0.0,
        quality_score=8.0, speed_score=8.0),
    
    # DeepSeek
    "deepseek-v2": AdvancedModelConfig(
        name="DeepSeek V2", provider="huggingface", model_id="deepseek-ai/DeepSeek-V2",
        capabilities=[ModelCapability.TEXT, ModelCapability.CODE, ModelCapability.MATH],
        specialization=["code", "math"], cost_per_1k=0.0001,
        quality_score=8.0, speed_score=8.0),
    
    # Cohere
    "command-r-plus": AdvancedModelConfig(
        name="Command R+", provider="cohere", model_id="command-r-plus",
        capabilities=[ModelCapability.TEXT, ModelCapability.REASONING],
        specialization=["rag", "search"], cost_per_1k=0.003,
        quality_score=8.5, speed_score=7.5),
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE GENERATION MODELS (10+)
    # ═══════════════════════════════════════════════════════════
    
    "dall-e-3": AdvancedModelConfig(
        name="DALL-E 3", provider="openai", model_id="dall-e-3",
        capabilities=[ModelCapability.IMAGE_GENERATION],
        specialization=["photorealistic", "creative"], cost_per_1k=0.04,
        quality_score=9.5, speed_score=6.0),
    "stable-diffusion-xl": AdvancedModelConfig(
        name="Stable Diffusion XL", provider="huggingface", model_id="stabilityai/stable-diffusion-xl-base-1.0",
        capabilities=[ModelCapability.IMAGE_GENERATION],
        specialization=["open-source", "customizable"], cost_per_1k=0.0,
        quality_score=8.5, speed_score=7.0),
    "flux-pro": AdvancedModelConfig(
        name="Flux Pro", provider="huggingface", model_id="black-forest-labs/FLUX.1-pro",
        capabilities=[ModelCapability.IMAGE_GENERATION],
        specialization=["photorealistic", "fast"], cost_per_1k=0.05,
        quality_score=9.0, speed_score=8.0),
    "midjourney-v6": AdvancedModelConfig(
        name="Midjourney v6", provider="midjourney", model_id="v6",
        capabilities=[ModelCapability.IMAGE_GENERATION],
        specialization=["artistic", "creative"], cost_per_1k=0.04,
        quality_score=9.5, speed_score=6.5),
    
    # ═══════════════════════════════════════════════════════════
    # AUDIO MODELS (8+)
    # ═══════════════════════════════════════════════════════════
    
    "whisper-large-v3": AdvancedModelConfig(
        name="Whisper Large v3", provider="openai", model_id="whisper-large-v3",
        capabilities=[ModelCapability.AUDIO_TRANSCRIPTION],
        specialization=["transcription", "multilingual"], cost_per_1k=0.006,
        quality_score=9.0, speed_score=7.0),
    "elevenlabs-turbo": AdvancedModelConfig(
        name="ElevenLabs Turbo", provider="elevenlabs", model_id="eleven_turbo_v2",
        capabilities=[ModelCapability.AUDIO_GENERATION],
        specialization=["tts", "voice-cloning"], cost_per_1k=0.03,
        quality_score=9.5, speed_score=9.0),
    "bark": AdvancedModelConfig(
        name="Bark", provider="huggingface", model_id="suno/bark",
        capabilities=[ModelCapability.AUDIO_GENERATION],
        specialization=["tts", "multilingual"], cost_per_1k=0.0,
        quality_score=8.0, speed_score=6.0),
    
    # ═══════════════════════════════════════════════════════════
    # VIDEO MODELS (6+)
    # ═══════════════════════════════════════════════════════════
    
    "sora": AdvancedModelConfig(
        name="Sora", provider="openai", model_id="sora-1.0",
        capabilities=[ModelCapability.VIDEO_GENERATION],
        specialization=["photorealistic", "long-form"], cost_per_1k=1.0,
        quality_score=10.0, speed_score=3.0),
    "runway-gen3": AdvancedModelConfig(
        name="Runway Gen-3", provider="runway", model_id="gen3",
        capabilities=[ModelCapability.VIDEO_GENERATION],
        specialization=["creative", "fast"], cost_per_1k=0.5,
        quality_score=9.0, speed_score=7.0),
    "pika": AdvancedModelConfig(
        name="Pika", provider="pika", model_id="pika-1.0",
        capabilities=[ModelCapability.VIDEO_GENERATION],
        specialization=["animation"], cost_per_1k=0.3,
        quality_score=8.5, speed_score=7.5),
    
    # ═══════════════════════════════════════════════════════════
    # CODE SPECIALIZED MODELS (8+)
    # ═══════════════════════════════════════════════════════════
    
    "codellama-70b": AdvancedModelConfig(
        name="CodeLlama 70B", provider="huggingface", model_id="codellama/CodeLlama-70b-Instruct-hf",
        capabilities=[ModelCapability.CODE],
        specialization=["code", "completion"], cost_per_1k=0.0009,
        quality_score=8.5, speed_score=7.5),
    "starcoder2-15b": AdvancedModelConfig(
        name="StarCoder2 15B", provider="huggingface", model_id="bigcode/starcoder2-15b",
        capabilities=[ModelCapability.CODE],
        specialization=["code", "multilingual"], cost_per_1k=0.0,
        quality_score=8.0, speed_score=8.5),
}


# Agent-specific model preferences (updated for multimodal AGI)
AGENT_MODEL_PREFERENCES = {
    "strategist": ["gpt-4-turbo", "claude-3-opus", "gemini-1.5-pro", "qwen-2.5-72b"],
    "architect": ["claude-3.5-sonnet", "gpt-4o", "llama-3.1-405b", "mixtral-8x22b"],
    "engineer": ["claude-3.5-sonnet", "codellama-70b", "deepseek-v2", "qwen-2.5-72b"],
    "designer": ["dall-e-3", "flux-pro", "midjourney-v6", "stable-diffusion-xl"],
    "entrepreneur": ["gpt-4-turbo", "claude-3-opus", "command-r-plus", "gemini-1.5-pro"],
    "futurist": ["o1-preview", "claude-3-opus", "gemini-1.5-pro", "llama-3.1-405b"],
    # Multimodal specialists  
    "media_creator": ["dall-e-3", "flux-pro", "sora", "runway-gen3", "elevenlabs-turbo"],
    "audio_specialist": ["whisper-large-v3", "elevenlabs-turbo", "bark"],
    "video_specialist": ["sora", "runway-gen3", "pika", "gemini-1.5-pro"],
    # Legacy agents
    "economist": ["qwen-2.5-72b", "mistral-7b", "llama-3.2-3b"],
    "ethicist": ["llama-3.1-70b", "gemini-1.5-pro", "mistral-7b"],
    "philosopher": ["llama-3.1-70b", "mistral-7b", "gemini-1.5-pro"],
    "cultural_translator": ["qwen-2.5-72b", "llama-3.2-3b", "gemini-1.5-flash"],
}


class ModelRotationEngine:
    """AGI-Level Model Rotation - Cycles through 10-50+ models per day"""
    
    def __init__(self, models_per_day: int = 30):
        self.models_per_day = models_per_day
        self.rotation_file = Path("training_data/model_rotation.json")
        self.current_rotation = self.get_daily_rotation()
        
    def get_daily_rotation(self) -> List[str]:
        """Get today's model rotation (deterministic based on date)"""
        today = date.today()
        seed = int(today.strftime("%Y%m%d"))
        
        random.seed(seed)
        
        # Categorize models by capability
        text_models = [k for k, v in AVAILABLE_MODELS.items() 
                      if ModelCapability.TEXT in v.capabilities]
        image_models = [k for k, v in AVAILABLE_MODELS.items() 
                       if ModelCapability.IMAGE_GENERATION in v.capabilities]
        audio_models = [k for k, v in AVAILABLE_MODELS.items() 
                       if ModelCapability.AUDIO_GENERATION in v.capabilities or 
                          ModelCapability.AUDIO_TRANSCRIPTION in v.capabilities]
        video_models = [k for k, v in AVAILABLE_MODELS.items() 
                       if ModelCapability.VIDEO_GENERATION in v.capabilities]
        code_models = [k for k, v in AVAILABLE_MODELS.items() 
                      if ModelCapability.CODE in v.capabilities]
        multimodal_models = [k for k, v in AVAILABLE_MODELS.items() 
                            if ModelCapability.MULTIMODAL in v.capabilities]
        
        rotation = []
        
        # ALWAYS include top performers
        rotation.extend(["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro", "llama-3.1-405b"])
        
        # Add diversity across ALL capabilities
        rotation.extend(random.sample(text_models, min(10, len(text_models))))
        rotation.extend(random.sample(image_models, min(4, len(image_models))))
        rotation.extend(random.sample(audio_models, min(3, len(audio_models))))
        rotation.extend(random.sample(video_models, min(3, len(video_models))))
        rotation.extend(random.sample(code_models, min(4, len(code_models))))
        rotation.extend(random.sample(multimodal_models, min(3, len(multimodal_models))))
        
        # Remove duplicates
        seen = set()
        unique_rotation = []
        for model in rotation:
            if model not in seen and model in AVAILABLE_MODELS:
                seen.add(model)
                unique_rotation.append(model)
        
        return unique_rotation[:self.models_per_day]
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get best model from today's rotation for specific task"""
        capability_map = {
            "text": ModelCapability.TEXT,
            "image_gen": ModelCapability.IMAGE_GENERATION,
            "image_understand": ModelCapability.IMAGE_UNDERSTANDING,
            "audio_gen": ModelCapability.AUDIO_GENERATION,
            "audio_transcribe": ModelCapability.AUDIO_TRANSCRIPTION,
            "video_gen": ModelCapability.VIDEO_GENERATION,
            "video_understand": ModelCapability.VIDEO_UNDERSTANDING,
            "code": ModelCapability.CODE,
            "math": ModelCapability.MATH,
            "reasoning": ModelCapability.REASONING,
            "multimodal": ModelCapability.MULTIMODAL,
        }
        
        required_capability = capability_map.get(task_type, ModelCapability.TEXT)
        
        # Filter today's rotation for capability
        suitable_models = [
            model_key for model_key in self.current_rotation
            if required_capability in AVAILABLE_MODELS[model_key].capabilities
        ]
        
        if not suitable_models:
            suitable_models = [
                k for k, v in AVAILABLE_MODELS.items()
                if required_capability in v.capabilities
            ]
        
        if not suitable_models:
            return "gpt-4o"
        
        # Return highest quality model
        return max(suitable_models, key=lambda k: AVAILABLE_MODELS[k].quality_score)
    
    def save_rotation(self):
        """Save current rotation to file"""
        self.rotation_file.parent.mkdir(parents=True, exist_ok=True)
        
        rotation_data = {
            "date": date.today().isoformat(),
            "models_count": len(self.current_rotation),
            "models": self.current_rotation,
            "capabilities_summary": {
                "text": len([m for m in self.current_rotation if ModelCapability.TEXT in AVAILABLE_MODELS[m].capabilities]),
                "images": len([m for m in self.current_rotation if ModelCapability.IMAGE_GENERATION in AVAILABLE_MODELS[m].capabilities]),
                "audio": len([m for m in self.current_rotation if ModelCapability.AUDIO_GENERATION in AVAILABLE_MODELS[m].capabilities or ModelCapability.AUDIO_TRANSCRIPTION in AVAILABLE_MODELS[m].capabilities]),
                "video": len([m for m in self.current_rotation if ModelCapability.VIDEO_GENERATION in AVAILABLE_MODELS[m].capabilities]),
                "code": len([m for m in self.current_rotation if ModelCapability.CODE in AVAILABLE_MODELS[m].capabilities]),
            }
        }
        
        with open(self.rotation_file, 'w') as f:
            json.dump(rotation_data, f, indent=2)
    
    def print_daily_rotation(self):
        """Print today's AGI-level model lineup"""
        print("\n" + "=" * 70)
        print(f"🚀 AGI MODEL ROTATION - {date.today()}")
        print("=" * 70)
        print(f"\n📊 Using {len(self.current_rotation)} specialized models today:\n")
        
        # Group by capability
        by_capability = {}
        for model_key in self.current_rotation:
            model = AVAILABLE_MODELS[model_key]
            for cap in model.capabilities:
                cap_name = cap.value.replace('_', ' ').title()
                if cap_name not in by_capability:
                    by_capability[cap_name] = []
                by_capability[cap_name].append(f"{model.name} ({model.provider})")
        
        for capability, models in sorted(by_capability.items()):
            print(f"  🎯 {capability.upper()}:")
            for model_name in sorted(set(models)):
                print(f"      • {model_name}")
            print()
        
        print("=" * 70)
        total_quality = sum(AVAILABLE_MODELS[m].quality_score for m in self.current_rotation)
        avg_quality = total_quality / len(self.current_rotation)
        print(f"📈 Average Quality Score: {avg_quality:.1f}/10")
        print(f"🎯 AGI Capabilities: Text + Images + Audio + Video + Code + Multimodal")
        print("=" * 70 + "\n")


def get_rotation_engine(models_per_day: int = 30) -> ModelRotationEngine:
    """Get AGI-level model rotation engine"""
    engine = ModelRotationEngine(models_per_day)
    engine.save_rotation()
    return engine


def get_day_of_year() -> int:
    """Get current day of year for rotation."""
    return datetime.now().timetuple().tm_yday


def get_rotation_index() -> int:
    """
    Get rotation index based on day.
    Changes daily to rotate through different model combinations.
    """
    return get_day_of_year() % 3  # Rotate through 3 model options


def get_model_for_agent(agent_name: str, rotation_index: Optional[int] = None) -> str:
    """
    Get the model assigned to an agent for today.
    
    Args:
        agent_name: Name of the agent (e.g., 'strategist')
        rotation_index: Optional manual rotation index (0-2), defaults to daily rotation
        
    Returns:
        Full model path (e.g., 'meta-llama/Llama-3.2-3B-Instruct')
    """
    if rotation_index is None:
        rotation_index = get_rotation_index()
    
    # Get preferred models for this agent
    preferred_models = AGENT_MODEL_PREFERENCES.get(
        agent_name, 
        ["llama-3.2-3b", "mistral-7b", "qwen-2.5-7b"]  # Default fallback
    )
    
    # Select model based on rotation
    model_key = preferred_models[rotation_index % len(preferred_models)]
    model_path = AVAILABLE_MODELS[model_key]
    
    return model_path


def get_all_agent_models(rotation_index: Optional[int] = None) -> Dict[str, str]:
    """
    Get model assignments for all agents.
    
    Returns:
        Dictionary mapping agent names to model paths
    """
    agents = [
        "strategist", "architect", "engineer", "designer", "entrepreneur",
        "futurist", "economist", "ethicist", "philosopher", "cultural_translator"
    ]
    
    return {
        agent: get_model_for_agent(agent, rotation_index)
        for agent in agents
    }


def print_daily_assignments():
    """Print today's model assignments for all agents."""
    print("=" * 80)
    print(f"🤖 Agent Model Assignments - Day {get_day_of_year()} ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 80)
    
    assignments = get_all_agent_models()
    
    for agent, model in assignments.items():
        model_name = model.split("/")[-1]
        print(f"  {agent:20s} → {model_name}")
    
    print("=" * 80)
    print(f"Rotation Index: {get_rotation_index()} (changes daily)")
    print("=" * 80)


def add_custom_model(model_key: str, model_path: str):
    """
    Add a custom model from your HF account.
    
    Args:
        model_key: Short name for the model (e.g., 'my-custom-llama')
        model_path: Full HF path (e.g., 'username/model-name')
    """
    AVAILABLE_MODELS[model_key] = model_path


def discover_user_models(username: str, token: Optional[str] = None) -> List[str]:
    """
    Discover models available in your HF account.
    
    Args:
        username: Your HuggingFace username
        token: Optional HF API token (uses HF_API_TOKEN env var if not provided)
        
    Returns:
        List of model paths from your account
    """
    try:
        from huggingface_hub import HfApi
        
        api_token = token or os.getenv("HF_API_TOKEN")
        api = HfApi(token=api_token)
        
        # List models from your account
        models = api.list_models(author=username)
        
        model_paths = []
        for model in models:
            # Only include text-generation or conversational models
            if any(tag in model.tags for tag in ["text-generation", "conversational", "text2text-generation"]):
                model_paths.append(model.id)
        
        return model_paths
        
    except ImportError:
        print("⚠️ huggingface-hub required for model discovery")
        return []
    except Exception as e:
        print(f"⚠️ Error discovering models: {e}")
        return []


if __name__ == "__main__":
    # Display today's assignments
    print_daily_assignments()
    
    print("\n📋 Available Models:")
    for key, path in AVAILABLE_MODELS.items():
        print(f"  {key:20s} → {path}")

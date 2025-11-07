"""
Model Ensemble System - Combine multiple top models for superior performance.

Strategy:
1. Use multiple top models (GPT-4, Claude, Gemini, Llama, Qwen, etc.)
2. Ensemble their outputs (voting, averaging, best-of-N)
3. Learn which models work best for which tasks
4. Continuously improve model selection
5. Merge knowledge into custom fine-tuned model

This creates a "meta-model" that's better than any single model.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime
from pathlib import Path


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HF_INFERENCE = "hf_inference"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for a model in the ensemble."""
    name: str
    provider: ModelProvider
    model_id: str
    cost_per_1k: float  # Cost per 1K tokens
    speed_score: float  # 0-10, higher is faster
    quality_score: float  # 0-10, higher is better
    specialization: List[str]  # ["code", "reasoning", "creative", etc.]
    enabled: bool = True


# Top Models Configuration
TOP_MODELS = {
    # OpenAI GPT Models
    "gpt-4-turbo": ModelConfig(
        name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4-turbo-preview",
        cost_per_1k=0.01,
        speed_score=7.0,
        quality_score=9.5,
        specialization=["reasoning", "analysis", "general"],
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        cost_per_1k=0.005,
        speed_score=8.5,
        quality_score=9.0,
        specialization=["general", "multimodal", "fast"],
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        provider=ModelProvider.OPENAI,
        model_id="gpt-3.5-turbo",
        cost_per_1k=0.0005,
        speed_score=9.5,
        quality_score=7.5,
        specialization=["fast", "general", "cheap"],
    ),
    
    # Anthropic Claude Models
    "claude-3-opus": ModelConfig(
        name="Claude 3 Opus",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-opus-20240229",
        cost_per_1k=0.015,
        speed_score=6.0,
        quality_score=9.8,
        specialization=["reasoning", "analysis", "safety", "long-context"],
    ),
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        cost_per_1k=0.003,
        speed_score=8.0,
        quality_score=9.3,
        specialization=["code", "reasoning", "balanced"],
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        cost_per_1k=0.00025,
        speed_score=9.8,
        quality_score=8.0,
        specialization=["fast", "cheap", "general"],
    ),
    
    # Google Gemini Models
    "gemini-pro": ModelConfig(
        name="Gemini Pro",
        provider=ModelProvider.GOOGLE,
        model_id="gemini-pro",
        cost_per_1k=0.00025,
        speed_score=8.5,
        quality_score=8.5,
        specialization=["general", "multimodal", "fast"],
    ),
    "gemini-1.5-pro": ModelConfig(
        name="Gemini 1.5 Pro",
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-pro",
        cost_per_1k=0.001,
        speed_score=7.5,
        quality_score=9.0,
        specialization=["long-context", "reasoning", "multimodal"],
    ),
    
    # Meta Llama (HF Inference)
    "llama-3.1-405b": ModelConfig(
        name="Llama 3.1 405B",
        provider=ModelProvider.HF_INFERENCE,
        model_id="meta-llama/Llama-3.1-405B-Instruct",
        cost_per_1k=0.0,
        speed_score=4.0,
        quality_score=9.2,
        specialization=["reasoning", "general", "oss"],
    ),
    "llama-3.1-70b": ModelConfig(
        name="Llama 3.1 70B",
        provider=ModelProvider.HF_INFERENCE,
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        cost_per_1k=0.0,
        speed_score=6.0,
        quality_score=8.8,
        specialization=["reasoning", "general", "oss"],
    ),
    "llama-3.2-3b": ModelConfig(
        name="Llama 3.2 3B",
        provider=ModelProvider.HF_INFERENCE,
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        cost_per_1k=0.0,
        speed_score=9.5,
        quality_score=7.5,
        specialization=["fast", "cheap", "oss"],
    ),
    
    # Alibaba Qwen
    "qwen-2.5-72b": ModelConfig(
        name="Qwen 2.5 72B",
        provider=ModelProvider.HF_INFERENCE,
        model_id="Qwen/Qwen2.5-72B-Instruct",
        cost_per_1k=0.0,
        speed_score=6.5,
        quality_score=8.9,
        specialization=["reasoning", "multilingual", "oss"],
    ),
    
    # Mistral
    "mistral-large": ModelConfig(
        name="Mistral Large",
        provider=ModelProvider.HF_INFERENCE,
        model_id="mistralai/Mistral-Large-Instruct-2407",
        cost_per_1k=0.0,
        speed_score=7.0,
        quality_score=8.7,
        specialization=["reasoning", "code", "oss"],
    ),
    
    # DeepSeek (Code specialist)
    "deepseek-coder": ModelConfig(
        name="DeepSeek Coder",
        provider=ModelProvider.HF_INFERENCE,
        model_id="deepseek-ai/deepseek-coder-33b-instruct",
        cost_per_1k=0.0,
        speed_score=7.5,
        quality_score=9.0,
        specialization=["code", "oss"],
    ),
}


@dataclass
class ModelResponse:
    """Response from a single model."""
    model_name: str
    content: str
    confidence: float  # Model's confidence in response
    latency: float  # Response time in seconds
    tokens_used: int
    cost: float


class EnsembleStrategy(Enum):
    """How to combine multiple model outputs."""
    BEST_OF_N = "best_of_n"  # Use highest quality model
    VOTING = "voting"  # Majority vote
    WEIGHTED = "weighted"  # Weight by model quality
    CONSENSUS = "consensus"  # Only if all agree
    SPECIALIZATION = "specialization"  # Use model specialized for task


class ModelEnsemble:
    """
    Ensemble multiple top models for superior performance.
    
    Strategies:
    1. Run same query on multiple models
    2. Combine outputs intelligently
    3. Learn which models work best
    4. Continuously improve selection
    """
    
    def __init__(self, models: Optional[List[str]] = None):
        self.models = models or list(TOP_MODELS.keys())
        self.performance_log = Path("model_performance.json")
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self._load_stats()
    
    def _load_stats(self):
        """Load historical performance data."""
        if self.performance_log.exists():
            try:
                with open(self.performance_log) as f:
                    self.model_stats = json.load(f)
            except Exception:
                self.model_stats = {}
    
    def _save_stats(self):
        """Save performance data."""
        try:
            with open(self.performance_log, 'w') as f:
                json.dump(self.model_stats, f, indent=2)
        except Exception:
            pass
    
    async def query_model(self, model_key: str, messages: List[Dict[str, str]]) -> ModelResponse:
        """Query a single model and return response."""
        config = TOP_MODELS[model_key]
        
        if not config.enabled:
            raise ValueError(f"Model {model_key} is disabled")
        
        start_time = datetime.now()
        
        try:
            # Route to appropriate provider
            if config.provider == ModelProvider.OPENAI:
                content = await self._query_openai(config, messages)
            elif config.provider == ModelProvider.ANTHROPIC:
                content = await self._query_anthropic(config, messages)
            elif config.provider == ModelProvider.GOOGLE:
                content = await self._query_google(config, messages)
            elif config.provider == ModelProvider.HF_INFERENCE:
                content = await self._query_hf(config, messages)
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
            
            latency = (datetime.now() - start_time).total_seconds()
            
            # Estimate tokens (rough)
            tokens = len(content.split()) * 1.3
            cost = (tokens / 1000) * config.cost_per_1k
            
            return ModelResponse(
                model_name=model_key,
                content=content,
                confidence=config.quality_score / 10,
                latency=latency,
                tokens_used=int(tokens),
                cost=cost,
            )
            
        except Exception as e:
            # Return error response
            return ModelResponse(
                model_name=model_key,
                content=f"Error: {str(e)}",
                confidence=0.0,
                latency=0.0,
                tokens_used=0,
                cost=0.0,
            )
    
    async def _query_openai(self, config: ModelConfig, messages: List[Dict]) -> str:
        """Query OpenAI API."""
        try:
            import openai
            import os
            
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await client.chat.completions.create(
                model=config.model_id,
                messages=messages,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI error: {e}")
    
    async def _query_anthropic(self, config: ModelConfig, messages: List[Dict]) -> str:
        """Query Anthropic Claude API."""
        try:
            import anthropic
            import os
            
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # Convert messages format
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]
            
            response = await client.messages.create(
                model=config.model_id,
                max_tokens=4096,
                system=system,
                messages=user_messages,
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise Exception(f"Anthropic error: {e}")
    
    async def _query_google(self, config: ModelConfig, messages: List[Dict]) -> str:
        """Query Google Gemini API."""
        try:
            import google.generativeai as genai
            import os
            
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(config.model_id)
            
            # Combine messages
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            
            response = await model.generate_content_async(prompt)
            return response.text
            
        except Exception as e:
            raise Exception(f"Google error: {e}")
    
    async def _query_hf(self, config: ModelConfig, messages: List[Dict]) -> str:
        """Query HuggingFace Inference API."""
        try:
            from huggingface_hub import InferenceClient
            import os
            
            client = InferenceClient(
                model=config.model_id,
                token=os.getenv("HF_API_TOKEN"),
            )
            
            response = await client.chat_completion(
                messages=messages,
                max_tokens=2048,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"HuggingFace error: {e}")
    
    async def ensemble_query(
        self,
        messages: List[Dict[str, str]],
        strategy: EnsembleStrategy = EnsembleStrategy.BEST_OF_N,
        num_models: int = 3,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query multiple models and ensemble their responses.
        
        Args:
            messages: Chat messages
            strategy: How to combine outputs
            num_models: Number of models to query
            task_type: Type of task ("code", "reasoning", etc.)
        
        Returns:
            Dict with final answer and metadata
        """
        # Select models based on task type
        selected_models = self._select_models(task_type, num_models)
        
        # Query all models in parallel
        tasks = [self.query_model(model, messages) for model in selected_models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_responses = [r for r in responses if isinstance(r, ModelResponse)]
        
        if not valid_responses:
            return {
                "answer": "All models failed to respond",
                "confidence": 0.0,
                "models_used": selected_models,
                "strategy": strategy.value,
            }
        
        # Combine responses based on strategy
        if strategy == EnsembleStrategy.BEST_OF_N:
            result = self._best_of_n(valid_responses)
        elif strategy == EnsembleStrategy.WEIGHTED:
            result = self._weighted_combine(valid_responses)
        elif strategy == EnsembleStrategy.CONSENSUS:
            result = self._consensus(valid_responses)
        else:
            result = self._best_of_n(valid_responses)  # Default
        
        # Log performance for learning
        self._log_performance(task_type, valid_responses)
        
        return result
    
    def _select_models(self, task_type: Optional[str], num: int) -> List[str]:
        """Select best models for task type."""
        if not task_type:
            # Use top-rated models
            sorted_models = sorted(
                TOP_MODELS.items(),
                key=lambda x: x[1].quality_score,
                reverse=True
            )
            return [m[0] for m in sorted_models[:num] if m[1].enabled]
        
        # Filter by specialization
        specialized = [
            (key, config) for key, config in TOP_MODELS.items()
            if task_type in config.specialization and config.enabled
        ]
        
        # Sort by quality
        specialized.sort(key=lambda x: x[1].quality_score, reverse=True)
        
        return [m[0] for m in specialized[:num]]
    
    def _best_of_n(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Select best response based on model quality."""
        best = max(responses, key=lambda r: r.confidence)
        
        return {
            "answer": best.content,
            "confidence": best.confidence,
            "selected_model": best.model_name,
            "total_models": len(responses),
            "total_cost": sum(r.cost for r in responses),
            "avg_latency": sum(r.latency for r in responses) / len(responses),
            "all_responses": [
                {"model": r.model_name, "preview": r.content[:100] + "..."}
                for r in responses
            ],
        }
    
    def _weighted_combine(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Combine responses weighted by confidence."""
        # For text, we can't truly "average", so we use weighted voting
        # In practice, might use LLM to synthesize
        return self._best_of_n(responses)  # Simplified
    
    def _consensus(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Only return if all models agree (high confidence)."""
        # Check similarity (simplified - would use semantic similarity)
        if len(set(r.content[:50] for r in responses)) == 1:
            # All agree
            return {
                "answer": responses[0].content,
                "confidence": 1.0,
                "consensus": True,
                "models": [r.model_name for r in responses],
            }
        else:
            # Disagreement - return best
            result = self._best_of_n(responses)
            result["consensus"] = False
            return result
    
    def _log_performance(self, task_type: Optional[str], responses: List[ModelResponse]):
        """Log model performance for continuous learning."""
        for response in responses:
            key = f"{response.model_name}_{task_type or 'general'}"
            
            if key not in self.model_stats:
                self.model_stats[key] = {
                    "total_queries": 0,
                    "total_latency": 0.0,
                    "total_cost": 0.0,
                    "avg_confidence": 0.0,
                }
            
            stats = self.model_stats[key]
            stats["total_queries"] += 1
            stats["total_latency"] += response.latency
            stats["total_cost"] += response.cost
            stats["avg_confidence"] = (
                (stats["avg_confidence"] * (stats["total_queries"] - 1) + response.confidence)
                / stats["total_queries"]
            )
        
        self._save_stats()
    
    def get_best_models_for_task(self, task_type: str, limit: int = 3) -> List[str]:
        """
        Learn which models perform best for specific tasks.
        
        This is continuous learning in action!
        """
        # Filter stats by task type
        task_stats = {
            k: v for k, v in self.model_stats.items()
            if k.endswith(f"_{task_type}")
        }
        
        if not task_stats:
            # No data yet, use defaults
            return self._select_models(task_type, limit)
        
        # Rank by performance (confidence / latency)
        ranked = sorted(
            task_stats.items(),
            key=lambda x: x[1]["avg_confidence"] / max(x[1]["total_latency"] / x[1]["total_queries"], 0.1),
            reverse=True
        )
        
        # Extract model names
        return [k.rsplit("_", 1)[0] for k, _ in ranked[:limit]]


# Global ensemble instance
_ensemble = None

def get_ensemble() -> ModelEnsemble:
    """Get global ensemble instance."""
    global _ensemble
    if _ensemble is None:
        _ensemble = ModelEnsemble()
    return _ensemble

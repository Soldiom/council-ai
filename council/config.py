from dataclasses import dataclass
import os
from typing import Optional

@dataclass
class Settings:
    """Configuration settings for the Council of Infinite Innovators."""

    # LLM Provider settings
    provider: str = os.getenv("DEFAULT_PROVIDER", "hf_inference")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # Hugging Face Inference API (free cloud)
    hf_api_token: str = os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN", ""))
    hf_inference_model: str = os.getenv("HF_INFERENCE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    # Hugging Face local models (optional)
    hf_model: str = os.getenv("HF_MODEL", "microsoft/phi-2")
    hf_device: int = int(os.getenv("HF_DEVICE", "-1"))  # -1 = CPU, 0+ = GPU

    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")    # Azure OpenAI settings
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # System settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    timeout: int = int(os.getenv("TIMEOUT", "30"))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
        return True

# Global settings instance
SETTINGS = Settings()

# Note: validation is intentionally not run at import time so callers can
# override settings (for example, providing a mock provider for local
# smoke-tests). Call `SETTINGS.validate()` where strict validation is
# required (for production runs).
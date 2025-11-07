"""
LLM Adapters for different providers (OpenAI, Anthropic, Azure OpenAI, etc.)
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import asyncio

# Optional Hugging Face pipeline cache
_HF_PIPELINES: dict = {}

# Load environment variables
load_dotenv()

async def openai_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """OpenAI LLM adapter using langchain-openai."""
    try:
        from langchain_openai import ChatOpenAI
        
        model = kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Convert to langchain message format
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:  # user or any other role
                langchain_messages.append(HumanMessage(content=content))
        
        result = await llm.ainvoke(langchain_messages)
        return result.content
        
    except ImportError:
        raise ImportError("langchain-openai package is required for OpenAI support")
    except Exception as e:
        raise Exception(f"OpenAI LLM error: {str(e)}")

async def anthropic_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """Anthropic Claude LLM adapter."""
    try:
        from langchain_anthropic import ChatAnthropic
        
        model = kwargs.get("model", "claude-3-sonnet-20240229")
        api_key = kwargs.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Convert to langchain message format
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        
        result = await llm.ainvoke(langchain_messages)
        return result.content
        
    except ImportError:
        raise ImportError("langchain-anthropic package is required for Anthropic support")
    except Exception as e:
        raise Exception(f"Anthropic LLM error: {str(e)}")

async def azure_openai_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """Azure OpenAI LLM adapter."""
    try:
        from langchain_openai import AzureChatOpenAI
        
        api_key = kwargs.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        endpoint = kwargs.get("endpoint", os.getenv("AZURE_OPENAI_ENDPOINT"))
        api_version = kwargs.get("api_version", os.getenv("AZURE_OPENAI_API_VERSION"))
        deployment_name = kwargs.get("deployment_name", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        if not all([api_key, endpoint, deployment_name]):
            raise ValueError("Azure OpenAI API key, endpoint, and deployment name are required")
        
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            openai_api_key=api_key,
            openai_api_version=api_version,
            deployment_name=deployment_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Convert to langchain message format
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        
        result = await llm.ainvoke(langchain_messages)
        return result.content
        
    except ImportError:
        raise ImportError("langchain-openai package is required for Azure OpenAI support")
    except Exception as e:
        raise Exception(f"Azure OpenAI LLM error: {str(e)}")

def get_llm_adapter(provider: str = "openai"):
    """Get the appropriate LLM adapter based on provider."""
    adapters = {
        "openai": openai_llm,
        "anthropic": anthropic_llm,
        "azure_openai": azure_openai_llm,
        "azure": azure_openai_llm,  # Alias
        "mock": None,  # placeholder, resolved below
        "huggingface": None,  # placeholder resolved below
        "hf": None,  # alias for local HF
        "hf_inference": None,  # placeholder for cloud API
        "hf_api": None,  # alias for cloud API
    }

    if provider not in adapters:
        raise ValueError(f"Unsupported provider: {provider}. Available: {list(adapters.keys())}")

    # Provide a simple mock adapter for local development / smoke tests
    if provider == "mock":
        return mock_llm

    # Hugging Face (transformers) adapter - local models
    if provider in ("huggingface", "hf"):
        return huggingface_llm

    # Hugging Face Inference API - free cloud models
    if provider in ("hf_inference", "hf_api"):
        return hf_inference_llm

    return adapters[provider]


def get_llm():
    """
    Get default LLM with chat interface for agents.
    Returns an object with async chat() method.
    """
    from council.config import SETTINGS
    
    class LLMWrapper:
        def __init__(self):
            self.provider = SETTINGS.provider
            self.adapter = get_llm_adapter(self.provider)
        
        async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
            """Chat interface compatible with agents."""
            return await self.adapter(messages, **kwargs)
    
    return LLMWrapper()


async def mock_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """A simple mock LLM adapter for local smoke tests.

    It returns deterministic, short responses based on the user prompt content.
    Use this when you don't want to call an external API (local development, CI).
    """
    # Try to extract a short user message for context
    user_texts = [m.get("content", "") for m in messages if m.get("role") in ("user", "assistant")]
    prompt = " ".join(user_texts).strip()
    # Basic canned behavior
    if not prompt:
        return "[mock] No prompt provided."
    if "design" in prompt.lower() and "scalable" in prompt.lower():
        return (
            "[mock] Synthesis: Build a modular microservice architecture, "
            "use async processing for inference, and start with a single-region deployment "
            "with a migration path to multi-region. Prioritize observability and cost controls."
        )
    if "go-to-market" in prompt.lower() or "go to market" in prompt.lower():
        return (
            "[mock] GTM: Validate with a 3-week pilot, target key customers in the GCC, "
            "measure activation and reduce time-to-value."
        )
    # Default fallback
    return f"[mock] Response to: {prompt[:200]}"


async def huggingface_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """Run a local Hugging Face Transformers text-generation pipeline.

    This adapter is optional and will raise ImportError if `transformers` is not installed.
    It caches the pipeline per-model in `_HF_PIPELINES` to avoid reloading.
    For cloud/free API usage, use 'hf_inference' provider instead.
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError("transformers is required for huggingface provider") from e

    model_name = kwargs.get("model", os.getenv("HF_MODEL", os.getenv("LLM_HF_MODEL", "gpt2")))
    max_new_tokens = int(kwargs.get("max_new_tokens", 256))
    temperature = float(kwargs.get("temperature", os.getenv("LLM_TEMPERATURE", 0.2)))

    # Lazy-load and cache pipeline
    if model_name not in _HF_PIPELINES:
        # Use text-generation pipeline (causal LM) for simplicity
        try:
            # Allow CPU/GPU selection via environment or kwargs in advanced setups
            pipe = pipeline("text-generation", model=model_name, device=kwargs.get("device", -1))
        except Exception:
            # Fallback to auto model+tokenizer to provide a clearer error if download fails
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=kwargs.get("device", -1))
        _HF_PIPELINES[model_name] = pipe

    pipe = _HF_PIPELINES[model_name]

    # Prepare a single prompt from messages (simple concatenation)
    user_texts = [m.get("content", "") for m in messages if m.get("role") in ("user", "assistant")]
    prompt = "\n".join(user_texts).strip() or ""

    # Run generation in a thread to avoid blocking the event loop
    def generate():
        return pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature, return_full_text=False)

    try:
        outputs = await asyncio.to_thread(generate)
    except Exception as e:
        raise Exception(f"Hugging Face generation error: {e}")

    if not outputs:
        return ""

    # pipeline returns list of dicts with 'generated_text' for text-generation
    generated = outputs[0].get("generated_text") if isinstance(outputs[0], dict) else str(outputs[0])
    return generated


async def hf_inference_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """Use Hugging Face Inference API (free cloud-hosted models, no downloads).

    This uses HF's serverless inference endpoints - completely free for public models.
    Get a free token at: https://huggingface.co/settings/tokens

    Supports model rotation - different agents can use different models.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise ImportError("huggingface-hub is required for hf_inference provider") from e

    # Configuration - support per-agent model override
    api_token = kwargs.get("api_token", os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN")))
    
    # Check for agent-specific model assignment
    agent_name = kwargs.get("agent_name")
    if agent_name:
        # Try to get agent-specific model from rotation system
        try:
            from council.model_rotation import get_model_for_agent
            model_name = get_model_for_agent(agent_name)
        except Exception:
            # Fallback to default model if rotation fails
            model_name = kwargs.get("model", os.getenv("HF_INFERENCE_MODEL", "meta-llama/Llama-3.2-3B-Instruct"))
    else:
        model_name = kwargs.get("model", os.getenv("HF_INFERENCE_MODEL", "meta-llama/Llama-3.2-3B-Instruct"))
    
    max_new_tokens = int(kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512)))
    temperature = float(kwargs.get("temperature", os.getenv("LLM_TEMPERATURE", 0.7)))

    if not api_token:
        raise ValueError(
            "HF_API_TOKEN is required for Hugging Face Inference API. "
            "Get a free token at: https://huggingface.co/settings/tokens"
        )

    # Initialize inference client
    client = InferenceClient(token=api_token)

    # Prepare prompt from messages
    # For chat models, format as conversation
    system_msgs = [m.get("content", "") for m in messages if m.get("role") == "system"]
    user_msgs = [m.get("content", "") for m in messages if m.get("role") in ("user", "assistant")]

    if system_msgs:
        prompt = f"{system_msgs[0]}\n\n{' '.join(user_msgs)}"
    else:
        prompt = " ".join(user_msgs)

    # Prepare messages
    messages_for_chat = []
    if system_msgs:
        messages_for_chat.append({"role": "system", "content": system_msgs[0]})
    messages_for_chat.append({"role": "user", "content": prompt})

    try:
        # Use chat_completion for instruction/chat models
        response = await asyncio.to_thread(
            client.chat_completion,
            messages=messages_for_chat,
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Provide helpful error messages
        error_msg = str(e)
        if "429" in error_msg or "rate limit" in error_msg.lower():
            raise Exception(
                "Rate limit reached on HF Inference API. "
                "Free tier has limits. Wait a moment or try a different model."
            )
        elif "404" in error_msg or "not found" in error_msg.lower():
            raise Exception(
                f"Model '{model_name}' not found or not available via Inference API. "
                "Try: mistralai/Mistral-7B-Instruct-v0.2"
            )
        else:
            raise Exception(f"HF Inference API error: {error_msg}")
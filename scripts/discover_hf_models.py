#!/usr/bin/env python
"""
Discover and configure models from your Hugging Face account.

This script:
1. Lists all text-generation models from your HF account
2. Tests model availability on HF Inference API
3. Auto-configures model rotation system
4. Can clone model repos for local use
"""

import os
import sys
from typing import List, Dict, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def discover_my_models(username: Optional[str] = None, token: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Discover models available in your HF account.
    
    Args:
        username: Your HF username (optional, will try to detect from token)
        token: HF API token (uses HF_API_TOKEN env var if not provided)
        
    Returns:
        List of dicts with model info: {id, name, tags, downloads, likes}
    """
    try:
        from huggingface_hub import HfApi, whoami
        
        api_token = token or os.getenv("HF_API_TOKEN")
        if not api_token:
            print("❌ Error: HF_API_TOKEN not found")
            print("Set it in .env file or environment")
            return []
        
        api = HfApi(token=api_token)
        
        # Get username from token if not provided
        if not username:
            try:
                user_info = whoami(token=api_token)
                username = user_info["name"]
                print(f"✅ Detected username: {username}")
            except Exception:
                print("⚠️  Could not detect username from token")
                print("Please provide username: python scripts/discover_hf_models.py <username>")
                return []
        
        print(f"\n🔍 Searching for models by {username}...")
        
        # List all models from user
        models = list(api.list_models(author=username))
        
        if not models:
            print(f"⚠️  No models found for user: {username}")
            print("\n💡 You can still use public models!")
            print("See available models at: https://huggingface.co/models")
            return []
        
        print(f"✅ Found {len(models)} model(s)\n")
        
        # Filter and format results
        model_list = []
        for model in models:
            tags = model.tags if hasattr(model, 'tags') else []
            
            model_info = {
                "id": model.id,
                "name": model.id.split("/")[-1],
                "tags": tags[:5] if tags else ["no-tags"],  # Show first 5 tags
                "downloads": getattr(model, 'downloads', 0),
                "likes": getattr(model, 'likes', 0),
            }
            model_list.append(model_info)
        
        return model_list
        
    except ImportError:
        print("❌ Error: huggingface-hub is required")
        print("Install it with: pip install huggingface-hub")
        return []
    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        return []


def test_model_inference(model_id: str, token: Optional[str] = None) -> bool:
    """
    Test if a model works with HF Inference API.
    
    Args:
        model_id: Full model ID (e.g., 'username/model-name')
        token: HF API token
        
    Returns:
        True if model works, False otherwise
    """
    try:
        from huggingface_hub import InferenceClient
        
        api_token = token or os.getenv("HF_API_TOKEN")
        client = InferenceClient(token=api_token)
        
        # Try a simple inference
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model=model_id,
            max_tokens=10,
        )
        
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        if "not supported" in error_str or "not found" in error_str:
            return False
        # Other errors might be temporary
        print(f"  ⚠️  Test error: {e}")
        return False


def print_model_table(models: List[Dict[str, str]], test_inference: bool = False):
    """Pretty print model table."""
    if not models:
        return
    
    print("=" * 100)
    print(f"{'Model ID':<50} {'Tags':<30} {'Downloads':<10} {'Status':<10}")
    print("=" * 100)
    
    token = os.getenv("HF_API_TOKEN")
    
    for model in models:
        model_id = model["id"]
        tags_str = ", ".join(model["tags"][:3]) if model["tags"] else "none"
        downloads = model["downloads"]
        
        # Test inference if requested
        status = "?"
        if test_inference:
            works = test_model_inference(model_id, token)
            status = "✅ Works" if works else "❌ No"
        
        print(f"{model_id:<50} {tags_str:<30} {downloads:<10} {status:<10}")
    
    print("=" * 100)


def add_to_rotation(models: List[Dict[str, str]]):
    """Add discovered models to the rotation system."""
    if not models:
        return
    
    print("\n📝 Adding models to rotation system...")
    
    try:
        from council.model_rotation import add_custom_model, AVAILABLE_MODELS
        
        for model in models:
            model_id = model["id"]
            model_key = model["name"].lower().replace("-", "_")
            
            # Add to available models
            add_custom_model(model_key, model_id)
            print(f"  ✅ Added: {model_key} → {model_id}")
        
        print(f"\n✅ {len(models)} model(s) added to rotation system")
        print("\nYou can now use them in .env:")
        for model in models:
            print(f"  HF_INFERENCE_MODEL={model['id']}")
        
    except Exception as e:
        print(f"⚠️  Could not add to rotation: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("  🤗 Hugging Face Model Discovery Tool")
    print("  Discover and configure models from your HF account")
    print("=" * 100 + "\n")
    
    # Get username from command line or detect from token
    username = sys.argv[1] if len(sys.argv) > 1 else None
    test_api = "--test" in sys.argv or "-t" in sys.argv
    
    # Discover models
    models = discover_my_models(username)
    
    if models:
        print_model_table(models, test_inference=test_api)
        
        # Offer to add to rotation
        print("\n💡 Options:")
        print("  1. Re-run with --test to test each model on Inference API")
        print("  2. Use any model ID in your .env file:")
        print("     HF_INFERENCE_MODEL=<model_id>")
        print("  3. Models will auto-rotate daily based on model_rotation.py")
        
        # Auto-add to rotation
        add_to_rotation(models)
    else:
        print("\n💡 No personal models found. You can:")
        print("  1. Use public models (already configured)")
        print("  2. Fine-tune a model and upload to HF")
        print("  3. Clone/fork a model to your account")
    
    print("\n✨ Done!\n")

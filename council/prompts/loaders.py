"""
Prompt loading utilities for the Council of Infinite Innovators.
"""

from pathlib import Path
from typing import Optional

def get_prompts_directory() -> Path:
    """Get the prompts directory path."""
    return Path(__file__).resolve().parent

def load_archetype(name: str) -> str:
    """Load an archetype prompt by name."""
    prompts_dir = get_prompts_directory()
    archetype_path = prompts_dir / "archetypes" / f"{name}.txt"
    
    if not archetype_path.exists():
        # Return a default prompt if file doesn't exist
        return f"You are {name.title()}, a specialized AI agent providing expert perspective."
    
    try:
        return archetype_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        # Fallback if file can't be read
        return f"You are {name.title()}, a specialized AI agent. (Error loading prompt: {e})"

def load_system_prompt(name: str) -> str:
    """Load a system prompt by name (safety, meta, etc.)."""
    prompts_dir = get_prompts_directory()
    system_path = prompts_dir / "system" / f"{name}.txt"
    
    if not system_path.exists():
        return ""
    
    try:
        return system_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def list_available_archetypes() -> list[str]:
    """List all available archetype prompts."""
    prompts_dir = get_prompts_directory()
    archetypes_dir = prompts_dir / "archetypes"
    
    if not archetypes_dir.exists():
        return []
    
    return [
        f.stem for f in archetypes_dir.glob("*.txt")
        if f.is_file()
    ]

def validate_prompts() -> dict[str, bool]:
    """Validate that all expected prompts exist."""
    expected_archetypes = [
        "architect", "entrepreneur", "strategist", "engineer", 
        "designer", "futurist", "economist", "ethicist", 
        "philosopher", "cultural_translator"
    ]
    
    expected_system = ["safety", "meta"]
    
    results = {}
    
    # Check archetypes
    for archetype in expected_archetypes:
        try:
            prompt = load_archetype(archetype)
            results[f"archetype_{archetype}"] = len(prompt) > 50  # Basic validation
        except Exception:
            results[f"archetype_{archetype}"] = False
    
    # Check system prompts
    for system in expected_system:
        try:
            prompt = load_system_prompt(system)
            results[f"system_{system}"] = len(prompt) > 10
        except Exception:
            results[f"system_{system}"] = False
    
    return results
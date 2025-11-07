"""
Safe file operations for the Council of Infinite Innovators.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Safe directories for file operations (prevent path traversal)
SAFE_DIRECTORIES = [
    "plans",
    "output", 
    "temp",
    "exports"
]

def is_safe_path(file_path: str) -> bool:
    """Check if a file path is safe for operations."""
    path = Path(file_path).resolve()
    
    # Check if path is within safe directories
    for safe_dir in SAFE_DIRECTORIES:
        try:
            safe_path = Path(safe_dir).resolve()
            path.relative_to(safe_path)
            return True
        except ValueError:
            continue
    
    return False

async def read_file_safe(file_path: str) -> Optional[str]:
    """Safely read a file within allowed directories."""
    if not is_safe_path(file_path):
        return None
    
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

async def write_file_safe(file_path: str, content: str) -> bool:
    """Safely write a file within allowed directories."""
    if not is_safe_path(file_path):
        return False
    
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False

async def save_json_safe(file_path: str, data: Dict[str, Any]) -> bool:
    """Safely save JSON data to file."""
    try:
        json_content = json.dumps(data, indent=2, ensure_ascii=False)
        return await write_file_safe(file_path, json_content)
    except Exception:
        return False

async def load_json_safe(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON data from file."""
    content = await read_file_safe(file_path)
    if content is None:
        return None
    
    try:
        return json.loads(content)
    except Exception:
        return None

async def list_files_safe(directory: str) -> List[str]:
    """Safely list files in a directory."""
    if not is_safe_path(directory):
        return []
    
    try:
        path = Path(directory)
        if not path.is_dir():
            return []
        
        return [str(f) for f in path.iterdir() if f.is_file()]
    except Exception:
        return []

async def create_output_file(content: str, filename: str, format: str = "md") -> Optional[str]:
    """Create an output file in the safe output directory."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
    file_path = output_dir / f"{safe_filename}.{format}"
    
    success = await write_file_safe(str(file_path), content)
    return str(file_path) if success else None

# Example usage functions for agents
async def save_analysis_report(agent_name: str, analysis: str, topic: str) -> Optional[str]:
    """Save an agent's analysis as a report file."""
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent_name}_{topic}_{timestamp}"
    return await create_output_file(analysis, filename, "md")

async def load_previous_analysis(topic: str) -> List[str]:
    """Load previous analysis files for a topic."""
    output_files = await list_files_safe("output")
    relevant_files = [f for f in output_files if topic.lower() in f.lower()]
    
    analyses = []
    for file_path in relevant_files[:5]:  # Limit to 5 most recent
        content = await read_file_safe(file_path)
        if content:
            analyses.append(content)
    
    return analyses
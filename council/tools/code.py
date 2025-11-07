"""
Safe code execution tool for the Council of Infinite Innovators.
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Allowed imports for sandboxed execution
ALLOWED_IMPORTS = {
    'math', 'statistics', 'json', 'datetime', 'collections',
    'itertools', 'functools', 're', 'string', 'uuid',
    'pandas', 'numpy', 'matplotlib.pyplot', 'seaborn'
}

def validate_code_safety(code: str) -> tuple[bool, str]:
    """Validate that code is safe for execution."""
    dangerous_patterns = [
        'import os', 'import sys', 'import subprocess', 'import shutil',
        'open(', 'file(', 'exec(', 'eval(', '__import__',
        'globals()', 'locals()', 'dir()', 'vars()',
        'input(', 'raw_input(', 'compile(',
        'reload(', 'delattr(', 'setattr(', 'getattr(',
        'hasattr(', 'isinstance(', 'issubclass(',
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return False, f"Potentially unsafe code pattern detected: {pattern}"
    
    # Check imports
    lines = code.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            # Extract module name
            if line.startswith('import '):
                module = line.split()[1].split('.')[0]
            else:  # from module import
                module = line.split()[1].split('.')[0]
            
            if module not in ALLOWED_IMPORTS:
                return False, f"Import not allowed: {module}"
    
    return True, "Code appears safe"

async def run_python_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    # Validate code safety
    is_safe, message = validate_code_safety(code)
    if not is_safe:
        return {
            "success": False,
            "output": "",
            "error": f"Security check failed: {message}",
            "execution_time": 0
        }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        import time
        start_time = time.time()
        
        # Execute with limited environment
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={'PYTHONPATH': ''}  # Limit Python path
        )
        
        execution_time = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "execution_time": execution_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Code execution timed out after {timeout} seconds",
            "execution_time": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Execution error: {str(e)}",
            "execution_time": 0
        }
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

async def run_data_analysis(data_code: str) -> Dict[str, Any]:
    """
    Run data analysis code with common data science libraries.
    
    Args:
        data_code: Python code for data analysis
        
    Returns:
        Execution results with any generated plots or outputs
    """
    # Prepend common imports for data analysis
    full_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Capture plots
plt.ioff()  # Turn off interactive mode
figures = []

""" + data_code + """

# Save any figures created
if plt.get_fignums():
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        # In a real implementation, you might save these to temporary files
        print(f"Figure {i+1} created with {len(fig.axes)} axes")

plt.close('all')  # Clean up
"""
    
    return await run_python_code(full_code, timeout=30)

async def calculate_metrics(data: Dict[str, Any], formula: str) -> Dict[str, Any]:
    """
    Calculate business metrics using provided data and formula.
    
    Args:
        data: Dictionary of data values
        formula: Python expression to calculate metric
        
    Returns:
        Calculation results
    """
    # Create safe calculation environment
    calc_code = f"""
import math
import statistics

# Data
data = {data}

# Calculation
try:
    result = {formula}
    print(f"Result: {{result}}")
    print(f"Type: {{type(result).__name__}}")
    
    # Additional context if result is numeric
    if isinstance(result, (int, float)):
        print(f"Formatted: {{result:,.2f}}")
except Exception as e:
    print(f"Calculation error: {{e}}")
"""
    
    return await run_python_code(calc_code, timeout=5)

# Example usage functions for agents
async def prototype_feature(description: str, requirements: List[str]) -> str:
    """Generate a code prototype based on description and requirements."""
    prototype_template = f'''
"""
Feature Prototype: {description}

Requirements:
{chr(10).join(f"- {req}" for req in requirements)}
"""

class FeaturePrototype:
    def __init__(self):
        self.name = "{description}"
        self.requirements = {requirements}
        
    def initialize(self):
        """Initialize the feature."""
        print(f"Initializing {{self.name}}")
        return True
        
    def process(self, input_data):
        """Process data through the feature."""
        # TODO: Implement feature logic
        print(f"Processing: {{input_data}}")
        return f"Processed: {{input_data}}"
        
    def validate(self):
        """Validate feature requirements."""
        for req in self.requirements:
            print(f"✓ Requirement: {{req}}")
        return True

# Example usage
if __name__ == "__main__":
    feature = FeaturePrototype()
    feature.initialize()
    result = feature.process("sample_data")
    feature.validate()
    print(f"Prototype ready: {{feature.name}}")
'''
    
    execution_result = await run_python_code(prototype_template)
    
    if execution_result["success"]:
        return f"✅ Prototype executed successfully:\n\n```python\n{prototype_template}\n```\n\nOutput:\n{execution_result['output']}"
    else:
        return f"❌ Prototype execution failed:\n{execution_result['error']}"
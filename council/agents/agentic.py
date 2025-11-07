"""
Agentic AI implementation using ReAct (Reasoning + Acting) pattern.

This transforms simple prompt-based agents into true agentic AI that can:
- Use tools (web search, code execution, file operations)
- Reason iteratively (think → act → observe → repeat)
- Make decisions about which tools to use
- Work toward goals autonomously
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
import re
from .base import BaseAgent, Message

@dataclass
class ToolCall:
    """Represents a tool invocation by an agent."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None


@dataclass
class AgenticStep:
    """One step in the agent's reasoning loop."""
    thought: str  # Agent's reasoning
    action: Optional[str] = None  # Tool to use
    action_input: Optional[Dict[str, Any]] = None  # Tool arguments
    observation: Optional[str] = None  # Tool result
    is_final: bool = False  # Whether this is the final answer


class AgenticAgent(BaseAgent):
    """
    Agentic AI that uses ReAct pattern for iterative problem-solving.
    
    Flow:
    1. THOUGHT: Reason about the task
    2. ACTION: Decide which tool to use
    3. OBSERVATION: See tool result
    4. Repeat until goal achieved
    5. FINAL ANSWER: Return result
    """
    
    max_iterations: int = 5
    tools: Dict[str, Callable] = {}
    
    def __init__(self, llm: Callable, tools: Optional[Dict[str, Callable]] = None):
        super().__init__(llm)
        if tools:
            self.tools = tools
        self._setup_tools()
    
    def _setup_tools(self):
        """Initialize available tools."""
        from ..tools.web import web_search, format_search_results
        from ..tools.code import run_python_code, calculate_metrics
        from ..tools.files import read_file, write_file, list_files
        
        # Register tools
        self.tools = {
            "web_search": web_search,
            "run_code": run_python_code,
            "calculate": calculate_metrics,
            "read_file": read_file,
            "write_file": write_file,
            "list_files": list_files,
        }
    
    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools."""
        descriptions = {
            "web_search": "Search the web for current information. Input: {'query': str, 'max_results': int}",
            "run_code": "Execute Python code safely. Input: {'code': str}",
            "calculate": "Calculate business metrics. Input: {'data': dict, 'formula': str}",
            "read_file": "Read file contents. Input: {'path': str}",
            "write_file": "Write content to file. Input: {'path': str, 'content': str}",
            "list_files": "List files in directory. Input: {'path': str}",
        }
        
        return "\n".join([f"- {name}: {desc}" for name, desc in descriptions.items()])
    
    def _build_react_prompt(self, query: str, history: List[AgenticStep]) -> str:
        """Build ReAct-style prompt with reasoning format."""
        
        tools_desc = self._get_tool_descriptions()
        
        prompt = f"""You are an agentic AI that solves problems iteratively using tools.

AVAILABLE TOOLS:
{tools_desc}

TASK: {query}

Use this EXACT format for each step:

THOUGHT: [Your reasoning about what to do next]
ACTION: [Tool name from the list above, or FINAL_ANSWER if done]
ACTION_INPUT: {{"arg1": "value1", "arg2": "value2"}}

You will see:
OBSERVATION: [Result from the tool]

Then continue with next THOUGHT/ACTION or provide FINAL_ANSWER.

EXAMPLES:

Example 1:
THOUGHT: I need current information about AI trends. Let me search the web.
ACTION: web_search
ACTION_INPUT: {{"query": "latest AI trends 2025", "max_results": 3}}
[After seeing observation...]
THOUGHT: Based on the search results, I can now provide a comprehensive answer.
ACTION: FINAL_ANSWER
ACTION_INPUT: {{"answer": "Based on recent sources, the top AI trends are..."}}

Example 2:
THOUGHT: I need to calculate the ROI. Let me use the calculate tool.
ACTION: calculate
ACTION_INPUT: {{"data": {{"revenue": 100000, "cost": 60000}}, "formula": "(data['revenue'] - data['cost']) / data['cost'] * 100"}}
[After seeing observation...]
THOUGHT: The ROI is 66.67%. I can now provide the final answer.
ACTION: FINAL_ANSWER
ACTION_INPUT: {{"answer": "The ROI is 66.67%, indicating strong profitability..."}}

"""
        
        # Add previous steps if any
        if history:
            prompt += "\nPREVIOUS STEPS:\n"
            for i, step in enumerate(history, 1):
                prompt += f"\nStep {i}:\n"
                prompt += f"THOUGHT: {step.thought}\n"
                if step.action:
                    prompt += f"ACTION: {step.action}\n"
                if step.action_input:
                    prompt += f"ACTION_INPUT: {json.dumps(step.action_input)}\n"
                if step.observation:
                    prompt += f"OBSERVATION: {step.observation}\n"
        
        prompt += "\nNow, what's your next step?\n"
        
        return prompt
    
    def _parse_agent_response(self, response: str) -> AgenticStep:
        """Parse agent's response into structured step."""
        
        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=\nACTION:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Thinking..."
        
        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(\w+)', response)
        action = action_match.group(1).strip() if action_match else None
        
        # Extract ACTION_INPUT
        action_input = None
        input_match = re.search(r'ACTION_INPUT:\s*({.+?})', response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {"raw": input_match.group(1)}
        
        # Check if final answer
        is_final = action == "FINAL_ANSWER" if action else False
        
        return AgenticStep(
            thought=thought,
            action=action,
            action_input=action_input,
            is_final=is_final
        )
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the observation."""
        if tool_name not in self.tools:
            return f"ERROR: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = await tool(**arguments)
            
            # Format result as string
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            elif isinstance(result, list):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except Exception as e:
            return f"ERROR executing {tool_name}: {str(e)}"
    
    async def run(self, messages: List[Message]) -> str:
        """
        Run the agentic loop: Think → Act → Observe → Repeat
        """
        # Get the query from messages
        query = messages[-1].content if messages else "No query provided"
        
        history: List[AgenticStep] = []
        
        for iteration in range(self.max_iterations):
            # Build prompt with history
            react_prompt = self._build_react_prompt(query, history)
            
            # Get agent's reasoning and action
            prompt_messages = [Message(role="user", content=react_prompt)]
            prepared = self.prepare_messages(prompt_messages)
            
            response = await self.llm(prepared, agent_name=self.name)
            
            # Parse the response
            step = self._parse_agent_response(response)
            
            # If final answer, return it
            if step.is_final:
                if step.action_input and "answer" in step.action_input:
                    final_answer = step.action_input["answer"]
                else:
                    final_answer = step.thought
                
                # Add reasoning trail
                trail = self._format_reasoning_trail(history + [step])
                return f"{final_answer}\n\n---\n**Reasoning Trail:**\n{trail}"
            
            # Execute the action (tool call)
            if step.action and step.action_input:
                observation = await self._execute_tool(step.action, step.action_input)
                step.observation = observation
            else:
                step.observation = "No action taken. Please specify ACTION and ACTION_INPUT."
            
            # Add to history
            history.append(step)
        
        # Max iterations reached
        return f"⚠️ Reached maximum iterations ({self.max_iterations}). Last thought: {history[-1].thought if history else 'No steps taken'}"
    
    def _format_reasoning_trail(self, steps: List[AgenticStep]) -> str:
        """Format the reasoning trail for transparency."""
        trail = ""
        for i, step in enumerate(steps, 1):
            trail += f"\n**Step {i}:**\n"
            trail += f"💭 Thought: {step.thought}\n"
            if step.action and not step.is_final:
                trail += f"🛠️ Action: {step.action}({step.action_input})\n"
                if step.observation:
                    trail += f"👁️ Observation: {step.observation[:200]}...\n"
        return trail


class AgenticStrategist(AgenticAgent):
    """Strategist agent with agentic capabilities."""
    name = "strategist"


class AgenticEngineer(AgenticAgent):
    """Engineer agent with agentic capabilities."""
    name = "engineer"


class AgenticResearcher(AgenticAgent):
    """Research-focused agentic agent."""
    name = "futurist"
    
    def _setup_tools(self):
        """Researcher focuses on web search and analysis."""
        from ..tools.web import web_search
        from ..tools.code import calculate_metrics
        
        self.tools = {
            "web_search": web_search,
            "calculate": calculate_metrics,
        }


# Factory function to create agentic versions of any agent
def make_agentic(agent_class: type, llm: Callable) -> AgenticAgent:
    """
    Convert any agent class to agentic version.
    
    Usage:
        agentic_strategist = make_agentic(Strategist, llm)
        response = await agentic_strategist.run([Message(...)])
    """
    
    class AgenticVersion(AgenticAgent):
        name = agent_class.name if hasattr(agent_class, 'name') else 'agent'
    
    return AgenticVersion(llm)

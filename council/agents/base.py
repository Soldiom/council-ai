"""
Base classes for Council agents and messages.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Message:
    """A message in the council conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for LLM."""
        return {"role": self.role, "content": self.content}

class BaseAgent(ABC):
    """Base class for all Council agents."""
    
    name: str = "base"
    system_prompt: str = ""
    
    def __init__(self, llm: Callable):
        self.llm = llm
        self.name = getattr(self, 'name', 'agent')  # Get agent name for model rotation
        self._load_system_prompt()
    
    def _load_system_prompt(self) -> None:
        """Load system prompt from prompts directory."""
        try:
            from ..prompts.loaders import load_archetype, load_system_prompt
            
            # Load safety and meta prompts
            safety = load_system_prompt("safety")
            meta = load_system_prompt("meta")
            
            # Load archetype-specific prompt
            archetype_prompt = load_archetype(self.name)
            
            # Combine prompts
            self.system_prompt = f"{safety}\n\n{meta}\n\n{archetype_prompt}"
            
        except Exception as e:
            # Fallback to default if prompts can't be loaded
            if not self.system_prompt:
                self.system_prompt = f"You are {self.name.title()}, a specialized AI agent."
    
    def prepare_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Prepare messages for LLM by adding system prompt."""
        system_message = Message(role="system", content=self.system_prompt)
        all_messages = [system_message] + messages
        return [msg.to_dict() for msg in all_messages]
    
    @abstractmethod
    async def run(self, messages: List[Message]) -> str:
        """Run the agent with given messages and return response."""
        pass
    
    async def single_query(self, query: str) -> str:
        """Convenience method for single query."""
        messages = [Message(role="user", content=query)]
        return await self.run(messages)

class BaseSpecializedAgent(BaseAgent):
    """Base class for specialized agents with standard LLM execution."""
    
    async def run(self, messages: List[Message]) -> str:
        """Standard implementation using the configured LLM."""
        try:
            prepared_messages = self.prepare_messages(messages)
            # Pass agent name for model rotation
            response = await self.llm(prepared_messages, agent_name=self.name)
            return response
        except Exception as e:
            return f"Error in {self.name}: {str(e)}"

class BaseSynthesizer(BaseAgent):
    """Synthesizer agent that combines council opinions into actionable plans."""
    
    name = "synthesizer"
    
    def __init__(self, llm: Callable):
        super().__init__(llm)
        self.system_prompt = """You are The Council Synthesizer. Your role is to:

1. Analyze multiple expert opinions from the Council of Infinite Innovators
2. Identify key themes, synergies, and conflicts between perspectives
3. Synthesize insights into a coherent, actionable plan
4. Prioritize recommendations based on impact and feasibility
5. Present a clear 30/60/90 day roadmap when appropriate

Format your response with:
- Executive Summary (2-3 sentences)
- Key Insights (numbered list)
- Actionable Recommendations (prioritized)
- Next Steps (specific and time-bound)

Be concise, decisive, and focus on implementation."""
    
    async def run(self, messages: List[Message]) -> str:
        """Synthesize council opinions into actionable plan."""
        try:
            prepared_messages = self.prepare_messages(messages)
            response = await self.llm(prepared_messages)
            return response
        except Exception as e:
            return f"Error in synthesis: {str(e)}"
    
    async def synthesize_opinions(self, question: str, opinions: List[tuple[str, str]]) -> str:
        """Synthesize multiple agent opinions."""
        opinion_text = "\n\n".join([
            f"=== {agent_name.upper()} PERSPECTIVE ===\n{opinion}"
            for agent_name, opinion in opinions
        ])
        
        synthesis_prompt = f"""
ORIGINAL QUESTION: {question}

COUNCIL PERSPECTIVES:
{opinion_text}

Please synthesize these perspectives into a unified, actionable strategy.
"""
        
        messages = [Message(role="user", content=synthesis_prompt)]
        return await self.run(messages)
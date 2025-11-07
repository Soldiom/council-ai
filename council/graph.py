"""
LangGraph-style orchestrator for the Council of Infinite Innovators.

This module provides a simple but powerful state machine for coordinating
multiple AI agents in collaborative problem-solving.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import asyncio
from .agents.base import BaseAgent, Message, BaseSynthesizer

@dataclass
class CouncilState:
    """State object for the Council execution graph."""
    
    # Input
    question: str = ""
    requested_agents: List[str] = field(default_factory=list)
    
    # Execution state
    current_step: str = "init"
    agent_opinions: Dict[str, str] = field(default_factory=dict)
    completed_agents: Set[str] = field(default_factory=set)
    
    # Output
    synthesis: str = ""
    execution_log: List[str] = field(default_factory=list)
    
    # Metadata
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def add_log(self, message: str) -> None:
        """Add a message to the execution log."""
        self.execution_log.append(message)
    
    def is_complete(self) -> bool:
        """Check if all requested agents have completed."""
        return len(self.completed_agents) >= len(self.requested_agents)

class CouncilGraph:
    """
    A simple state machine for orchestrating the Council of Infinite Innovators.
    
    This follows a gather-then-synthesize pattern:
    1. Collect opinions from requested agents in parallel
    2. Synthesize opinions into unified strategy
    3. Return actionable recommendations
    """
    
    def __init__(self, agents: Dict[str, BaseAgent], synthesizer: BaseSynthesizer):
        self.agents = agents
        self.synthesizer = synthesizer
        self.available_agents = set(agents.keys())
    
    async def run(self, question: str, agent_names: List[str]) -> str:
        """
        Run the council with specified agents on a given question.
        
        Args:
            question: The question or problem to address
            agent_names: List of agent names to consult
            
        Returns:
            Synthesized response from the council
        """
        import time
        
        # Initialize state
        state = CouncilState(
            question=question,
            requested_agents=agent_names,
            start_time=time.time()
        )
        
        state.add_log(f"🏛️ Council session started with {len(agent_names)} agents")
        
        # Validate requested agents
        unknown_agents = set(agent_names) - self.available_agents
        if unknown_agents:
            error_msg = f"Unknown agents requested: {unknown_agents}"
            state.add_log(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        # Step 1: Gather opinions from agents
        state.current_step = "gathering"
        state.add_log("📝 Gathering agent opinions...")
        
        await self._gather_opinions(state)
        
        # Step 2: Synthesize opinions
        state.current_step = "synthesizing"
        state.add_log("🔄 Synthesizing council perspectives...")
        
        await self._synthesize_opinions(state)
        
        # Step 3: Finalize
        state.current_step = "complete"
        state.end_time = time.time()
        duration = state.end_time - state.start_time
        state.add_log(f"✅ Council session completed in {duration:.2f}s")
        
        return state.synthesis
    
    async def _gather_opinions(self, state: CouncilState) -> None:
        """Gather opinions from all requested agents in parallel."""
        async def get_agent_opinion(agent_name: str) -> tuple[str, str]:
            """Get opinion from a single agent."""
            try:
                agent = self.agents[agent_name]
                messages = [Message(role="user", content=state.question)]
                
                state.add_log(f"  🤔 Consulting {agent_name}...")
                opinion = await agent.run(messages)
                state.add_log(f"  ✓ {agent_name} completed")
                
                return agent_name, opinion
                
            except Exception as e:
                error_msg = f"Error from {agent_name}: {str(e)}"
                state.add_log(f"  ❌ {error_msg}")
                return agent_name, f"[Error: {error_msg}]"
        
        # Run all agents in parallel
        tasks = [
            get_agent_opinion(agent_name) 
            for agent_name in state.requested_agents
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Store results
        for agent_name, opinion in results:
            state.agent_opinions[agent_name] = opinion
            state.completed_agents.add(agent_name)
    
    async def _synthesize_opinions(self, state: CouncilState) -> None:
        """Synthesize all agent opinions into a unified response."""
        try:
            opinions_list = [
                (agent_name, opinion) 
                for agent_name, opinion in state.agent_opinions.items()
            ]
            
            synthesis = await self.synthesizer.synthesize_opinions(
                state.question, 
                opinions_list
            )
            
            state.synthesis = synthesis
            state.add_log("  ✓ Synthesis completed")
            
        except Exception as e:
            error_msg = f"Synthesis error: {str(e)}"
            state.add_log(f"  ❌ {error_msg}")
            state.synthesis = f"Error during synthesis: {error_msg}"
    
    async def run_single_agent(self, question: str, agent_name: str) -> str:
        """Run a single agent (no synthesis needed)."""
        if agent_name not in self.available_agents:
            return f"Error: Unknown agent '{agent_name}'"
        
        try:
            agent = self.agents[agent_name]
            messages = [Message(role="user", content=question)]
            return await agent.run(messages)
            
        except Exception as e:
            return f"Error from {agent_name}: {str(e)}"
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.available_agents)
    
    def visualize(self) -> str:
        """Generate a simple ASCII visualization of the council structure."""
        agents_list = ", ".join(self.available_agents)
        return f"""
🏛️ Council of Infinite Innovators
┌─────────────────────────────────────┐
│ Available Agents: {len(self.available_agents)}                   │
│ {agents_list:<35} │
├─────────────────────────────────────┤
│ Flow: Question → Agents → Synthesis │
└─────────────────────────────────────┘
"""
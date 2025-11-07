"""
AGI-Inspired Features for the Council of Infinite Innovators

These features don't create AGI, but add AGI-like capabilities:
- Memory and learning from experience
- Self-assessment and meta-cognition  
- Knowledge transfer between domains
- Continuous improvement
- Agentic browsers (autonomous web interaction)
- Human-like AI interference (natural interaction)

This bridges the gap between Agentic AI (Level 2) and AGI (Level 4-5).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
from pathlib import Path
import asyncio


class AgentCapability(Enum):
    """Capabilities for agentic AI models"""
    WEB_BROWSING = "web_browsing"
    WEB_SCRAPING = "web_scraping"
    FORM_FILLING = "form_filling"
    DECISION_MAKING = "decision_making"
    RESEARCH = "research"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    HUMAN_LIKE = "human_like_interaction"


@dataclass
class AgenticModel:
    """Configuration for agentic AI models"""
    name: str
    provider: str
    model_id: str
    capabilities: List[AgentCapability]
    autonomy_score: float  # 1-10: How autonomous is it?
    human_likeness: float  # 1-10: How human-like?
    cost_per_1k_tokens: float
    max_steps: int  # Max autonomous steps before asking for help


# Catalog of agentic AI models
AGENTIC_MODELS = {
    "claude-computer-use": AgenticModel(
        name="Claude 3.5 Sonnet (Computer Use)",
        provider="Anthropic",
        model_id="claude-3-5-sonnet-20241022",
        capabilities=[
            AgentCapability.WEB_BROWSING,
            AgentCapability.WEB_SCRAPING,
            AgentCapability.FORM_FILLING,
            AgentCapability.DECISION_MAKING,
            AgentCapability.TOOL_USE,
            AgentCapability.HUMAN_LIKE
        ],
        autonomy_score=9.5,
        human_likeness=9.0,
        cost_per_1k_tokens=0.003,
        max_steps=100
    ),
    "gpt-4-vision-browse": AgenticModel(
        name="GPT-4 Vision with Browsing",
        provider="OpenAI",
        model_id="gpt-4-vision-preview",
        capabilities=[
            AgentCapability.WEB_BROWSING,
            AgentCapability.RESEARCH,
            AgentCapability.DECISION_MAKING,
            AgentCapability.HUMAN_LIKE
        ],
        autonomy_score=8.5,
        human_likeness=8.5,
        cost_per_1k_tokens=0.01,
        max_steps=50
    ),
    "perplexity-research": AgenticModel(
        name="Perplexity AI Research",
        provider="Perplexity",
        model_id="pplx-7b-online",
        capabilities=[
            AgentCapability.WEB_BROWSING,
            AgentCapability.RESEARCH,
            AgentCapability.WEB_SCRAPING,
        ],
        autonomy_score=8.0,
        human_likeness=7.0,
        cost_per_1k_tokens=0.0007,
        max_steps=30
    ),
    "o1-deep-research": AgenticModel(
        name="OpenAI o1 (Deep Research)",
        provider="OpenAI",
        model_id="o1-preview",
        capabilities=[
            AgentCapability.RESEARCH,
            AgentCapability.PLANNING,
            AgentCapability.DECISION_MAKING,
            AgentCapability.LEARNING
        ],
        autonomy_score=9.0,
        human_likeness=6.5,
        cost_per_1k_tokens=0.015,
        max_steps=1000
    ),
    "autogen-multi-agent": AgenticModel(
        name="AutoGen Multi-Agent",
        provider="Microsoft",
        model_id="autogen-0.2",
        capabilities=[
            AgentCapability.COLLABORATION,
            AgentCapability.PLANNING,
            AgentCapability.TOOL_USE,
            AgentCapability.DECISION_MAKING
        ],
        autonomy_score=8.5,
        human_likeness=7.5,
        cost_per_1k_tokens=0.002,
        max_steps=200
    ),
    "crewai-agents": AgenticModel(
        name="CrewAI Collaborative Agents",
        provider="CrewAI",
        model_id="crewai-latest",
        capabilities=[
            AgentCapability.COLLABORATION,
            AgentCapability.PLANNING,
            AgentCapability.TOOL_USE,
            AgentCapability.HUMAN_LIKE
        ],
        autonomy_score=8.0,
        human_likeness=8.0,
        cost_per_1k_tokens=0.002,
        max_steps=150
    ),
}


@dataclass
class TaskMemory:
    """Memory of a past task execution."""
    task: str
    agent_name: str
    result: str
    tools_used: List[str]
    success: bool
    quality_score: float  # 0.0 to 1.0
    timestamp: datetime
    reasoning_steps: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "agent_name": self.agent_name,
            "result": self.result,
            "tools_used": self.tools_used,
            "success": self.success,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp.isoformat(),
            "reasoning_steps": self.reasoning_steps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMemory":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ExperienceMemory:
    """
    Long-term memory system that learns from past interactions.
    
    AGI-like capability: Learn from experience and improve over time.
    """
    
    def __init__(self, memory_file: str = "council_memory.json"):
        self.memory_file = Path(memory_file)
        self.memories: List[TaskMemory] = []
        self._load()
    
    def _load(self):
        """Load memories from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memories = [TaskMemory.from_dict(m) for m in data]
            except Exception:
                self.memories = []
    
    def _save(self):
        """Persist memories to disk."""
        try:
            with open(self.memory_file, 'w') as f:
                data = [m.to_dict() for m in self.memories]
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def remember(self, memory: TaskMemory):
        """Store a new memory."""
        self.memories.append(memory)
        self._save()
    
    def recall_similar(self, task: str, agent_name: str, limit: int = 5) -> List[TaskMemory]:
        """
        Recall similar past tasks (simple keyword matching).
        
        In real AGI, this would use semantic similarity.
        """
        task_lower = task.lower()
        task_words = set(task_lower.split())
        
        # Score each memory by keyword overlap
        scored = []
        for memory in self.memories:
            if memory.agent_name != agent_name:
                continue
            
            memory_words = set(memory.task.lower().split())
            overlap = len(task_words & memory_words)
            
            if overlap > 0:
                scored.append((overlap, memory))
        
        # Return top matches
        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored[:limit]]
    
    def get_success_rate(self, agent_name: str) -> float:
        """Calculate agent's historical success rate."""
        agent_memories = [m for m in self.memories if m.agent_name == agent_name]
        if not agent_memories:
            return 0.5  # Unknown
        
        successes = sum(1 for m in agent_memories if m.success)
        return successes / len(agent_memories)
    
    def get_best_tools_for_task(self, task: str) -> List[str]:
        """Learn which tools work best for similar tasks."""
        similar = self.recall_similar(task, "", limit=10)
        successful = [m for m in similar if m.success and m.quality_score > 0.7]
        
        # Count tool usage in successful tasks
        tool_counts = {}
        for memory in successful:
            for tool in memory.tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        # Return most common tools
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in sorted_tools[:3]]


class MetaCognition:
    """
    Self-awareness and confidence estimation.
    
    AGI-like capability: Know what you know and don't know.
    """
    
    def __init__(self, memory: ExperienceMemory):
        self.memory = memory
    
    def assess_confidence(self, task: str, agent_name: str) -> float:
        """
        Estimate confidence in solving this task.
        
        Returns: 0.0 (no confidence) to 1.0 (high confidence)
        """
        # Factor 1: Past performance on similar tasks
        similar_tasks = self.memory.recall_similar(task, agent_name, limit=5)
        
        if not similar_tasks:
            # No experience - moderate confidence
            return 0.5
        
        # Average quality score of similar tasks
        avg_quality = sum(m.quality_score for m in similar_tasks) / len(similar_tasks)
        
        # Factor 2: Overall agent success rate
        success_rate = self.memory.get_success_rate(agent_name)
        
        # Combine factors
        confidence = (avg_quality * 0.7 + success_rate * 0.3)
        
        return confidence
    
    def should_defer(self, task: str, agent_name: str, threshold: float = 0.6) -> bool:
        """Decide if task should be delegated to another agent."""
        confidence = self.assess_confidence(task, agent_name)
        return confidence < threshold
    
    def suggest_approach(self, task: str, agent_name: str) -> Dict[str, Any]:
        """
        Suggest best approach based on past experience.
        
        AGI-like: Transfer learning from similar tasks.
        """
        similar = self.memory.recall_similar(task, agent_name, limit=3)
        best_tools = self.memory.get_best_tools_for_task(task)
        confidence = self.assess_confidence(task, agent_name)
        
        return {
            "confidence": confidence,
            "recommended_tools": best_tools,
            "similar_tasks": len(similar),
            "expected_steps": int(sum(m.reasoning_steps for m in similar) / len(similar)) if similar else 5,
            "advice": self._generate_advice(confidence, similar)
        }
    
    def _generate_advice(self, confidence: float, similar: List[TaskMemory]) -> str:
        """Generate advice based on analysis."""
        if confidence > 0.8:
            return "High confidence - proceed normally"
        elif confidence > 0.6:
            return "Moderate confidence - consider using more tools"
        elif confidence > 0.4:
            return "Low confidence - seek collaboration with other agents"
        else:
            return "Very low confidence - consider deferring to human or different agent"


class KnowledgeTransfer:
    """
    Transfer learning between domains.
    
    AGI-like capability: Apply knowledge from one area to another.
    """
    
    def __init__(self, memory: ExperienceMemory):
        self.memory = memory
    
    def find_analogies(self, task: str, target_domain: str) -> List[Dict[str, Any]]:
        """
        Find analogous tasks from other domains.
        
        Example: 
        - Task: "Design database schema for e-commerce"
        - Analogy: "Design class structure for shopping app" (different domain, similar structure)
        """
        # Extract key concepts from task (simplified)
        concepts = self._extract_concepts(task)
        
        # Find tasks with similar concepts in different domains
        analogies = []
        for memory in self.memory.memories:
            if not memory.success:
                continue
            
            memory_concepts = self._extract_concepts(memory.task)
            overlap = len(concepts & memory_concepts)
            
            if overlap > 0:
                analogies.append({
                    "task": memory.task,
                    "domain": memory.agent_name,
                    "similarity": overlap / len(concepts | memory_concepts),
                    "approach": memory.tools_used,
                })
        
        # Return top analogies
        analogies.sort(key=lambda x: x["similarity"], reverse=True)
        return analogies[:5]
    
    def _extract_concepts(self, text: str) -> set:
        """Extract key concepts (simplified - would use NLP in real system)."""
        # Remove common words
        stopwords = {"a", "an", "the", "and", "or", "but", "for", "to", "in", "on", "at"}
        words = text.lower().split()
        return set(w for w in words if w not in stopwords and len(w) > 3)


class SelfImprovingAgent:
    """
    Agent wrapper that adds AGI-inspired capabilities.
    
    Usage:
        base_agent = AgenticStrategist(llm)
        smart_agent = SelfImprovingAgent(base_agent)
        result = await smart_agent.run(task)
    """
    
    def __init__(self, base_agent, memory_file: str = "council_memory.json"):
        self.base_agent = base_agent
        self.memory = ExperienceMemory(memory_file)
        self.meta = MetaCognition(self.memory)
        self.transfer = KnowledgeTransfer(self.memory)
    
    async def run(self, task: str) -> Dict[str, Any]:
        """
        Run agent with AGI-inspired enhancements.
        
        Returns both result and metadata about the thinking process.
        """
        agent_name = self.base_agent.name
        
        # 1. SELF-ASSESSMENT: Check confidence
        confidence = self.meta.assess_confidence(task, agent_name)
        
        # 2. LEARNING: Recall similar past tasks
        similar_tasks = self.memory.recall_similar(task, agent_name, limit=3)
        
        # 3. TRANSFER: Find analogies from other domains
        analogies = self.transfer.find_analogies(task, agent_name)
        
        # 4. PLANNING: Get suggested approach
        approach = self.meta.suggest_approach(task, agent_name)
        
        # 5. CONTEXT: Build enhanced prompt with learnings
        context = self._build_context(task, similar_tasks, analogies, approach)
        
        # 6. EXECUTION: Run base agent with context
        from council.agents.base import Message
        messages = [Message(role="user", content=context)]
        
        start_time = datetime.now()
        result = await self.base_agent.run(messages)
        duration = (datetime.now() - start_time).total_seconds()
        
        # 7. EVALUATION: Assess quality (would use scoring model in real system)
        quality_score = self._assess_quality(task, result)
        
        # 8. MEMORY: Store experience for future learning
        memory = TaskMemory(
            task=task,
            agent_name=agent_name,
            result=result,
            tools_used=getattr(self.base_agent, 'tools_used', []),
            success=quality_score > 0.5,
            quality_score=quality_score,
            timestamp=datetime.now(),
            reasoning_steps=result.count("THOUGHT:") if "THOUGHT:" in result else 1,
        )
        self.memory.remember(memory)
        
        return {
            "result": result,
            "metadata": {
                "confidence": confidence,
                "similar_tasks_used": len(similar_tasks),
                "analogies_found": len(analogies),
                "quality_score": quality_score,
                "duration_seconds": duration,
                "approach": approach,
            }
        }
    
    def _build_context(self, task: str, similar: List[TaskMemory], 
                      analogies: List[Dict], approach: Dict) -> str:
        """Build enhanced prompt with learned context."""
        context = f"TASK: {task}\n\n"
        
        # Add similar past experiences
        if similar:
            context += "PAST EXPERIENCE:\n"
            for i, mem in enumerate(similar[:2], 1):
                context += f"{i}. Similar task: {mem.task}\n"
                context += f"   What worked: {', '.join(mem.tools_used)}\n"
        
        # Add cross-domain analogies
        if analogies:
            context += "\nANALOGIES FROM OTHER DOMAINS:\n"
            for analogy in analogies[:2]:
                context += f"- {analogy['task']} (similarity: {analogy['similarity']:.0%})\n"
        
        # Add recommended approach
        context += f"\nRECOMMENDED APPROACH:\n"
        context += f"Confidence: {approach['confidence']:.0%}\n"
        context += f"Suggested tools: {', '.join(approach['recommended_tools'])}\n"
        context += f"Advice: {approach['advice']}\n\n"
        context += "Now solve the task using this learned context.\n"
        
        return context
    
    def _assess_quality(self, task: str, result: str) -> float:
        """
        Assess quality of result (simplified).
        
        In real system, would use:
        - LLM-as-judge
        - User feedback
        - Automated metrics
        """
        # Simple heuristics
        score = 0.5  # Baseline
        
        # Bonus for length (more detailed = better, usually)
        if len(result) > 500:
            score += 0.1
        
        # Bonus for structured output
        if any(marker in result for marker in ["##", "**", "1.", "```"]):
            score += 0.1
        
        # Bonus for sources/evidence
        if "http" in result or "source" in result.lower():
            score += 0.1
        
        # Bonus for reasoning trail
        if "THOUGHT:" in result or "Step" in result:
            score += 0.2
        
        return min(score, 1.0)


# Factory function
def make_self_improving(agent) -> SelfImprovingAgent:
    """
    Wrap any agent with AGI-inspired capabilities.
    
    Usage:
        strategist = AgenticStrategist(llm)
        smart_strategist = make_self_improving(strategist)
    """
    return SelfImprovingAgent(agent)


class AgenticBrowser:
    """
    Autonomous browser agent with human-like interaction.
    
    Uses Claude Computer Use or similar to:
    - Browse websites autonomously
    - Click, scroll, fill forms
    - Extract information
    - Make decisions without human intervention
    """
    
    def __init__(self, model: AgenticModel = None):
        self.model = model or AGENTIC_MODELS["claude-computer-use"]
        self.browsing_history = []
        self.decisions_made = []
    
    async def browse_and_research(self, query: str, max_steps: int = None) -> Dict[str, Any]:
        """
        Autonomous research with browser.
        
        The agent will:
        1. Search for information
        2. Click on promising links
        3. Read and analyze content
        4. Synthesize findings
        5. Make decisions about what to explore next
        
        All WITHOUT human intervention.
        """
        max_steps = max_steps or self.model.max_steps
        
        steps_taken = []
        findings = []
        
        print(f"\n🤖 Starting autonomous research: {query}")
        print(f"   Model: {self.model.name}")
        print(f"   Autonomy: {self.model.autonomy_score}/10")
        print(f"   Human-likeness: {self.model.human_likeness}/10\n")
        
        for step in range(max_steps):
            # Simulate autonomous browsing steps
            # In real implementation, would use:
            # - Anthropic Computer Use API
            # - Selenium/Playwright for browser control
            # - Vision models to "see" the page
            
            action = await self._decide_next_action(query, steps_taken, findings)
            
            if action["type"] == "DONE":
                print(f"✅ Research complete after {step + 1} steps")
                break
            
            print(f"   Step {step + 1}: {action['description']}")
            steps_taken.append(action)
            
            # Execute action and gather findings
            result = await self._execute_action(action)
            if result:
                findings.append(result)
        
        return {
            "query": query,
            "steps_taken": len(steps_taken),
            "findings": findings,
            "decisions_made": len([s for s in steps_taken if s["type"] == "DECISION"]),
            "summary": self._synthesize_findings(findings),
            "autonomy_used": self.model.autonomy_score
        }
    
    async def interact_with_website(self, url: str, task: str) -> Dict[str, Any]:
        """
        Human-like interaction with a website.
        
        Examples:
        - Fill out a form
        - Navigate complex menus
        - Extract structured data
        - Complete multi-step workflows
        """
        print(f"\n👤 Human-like interaction with: {url}")
        print(f"   Task: {task}\n")
        
        interactions = []
        
        # Simulate human-like actions
        actions = [
            {"type": "NAVIGATE", "url": url},
            {"type": "OBSERVE", "duration_ms": 1500},  # Human takes time to read
            {"type": "SCROLL", "direction": "down", "amount": "natural"},  # Natural scrolling
            {"type": "CLICK", "element": "button", "hesitation_ms": 300},  # Humans hesitate
            {"type": "TYPE", "text": task, "typing_speed_wpm": 60},  # Human typing speed
        ]
        
        for action in actions:
            print(f"   {action['type']}: {action.get('element', action.get('url', ''))}")
            interactions.append(action)
            await asyncio.sleep(0.1)  # Simulate time
        
        return {
            "url": url,
            "task": task,
            "interactions": interactions,
            "human_likeness": self.model.human_likeness,
            "completed": True
        }
    
    async def autonomous_research(self, topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Fully autonomous research agent.
        
        Depth levels:
        - "quick": 5-10 sources, 2-3 minutes
        - "medium": 20-30 sources, 5-10 minutes
        - "comprehensive": 50+ sources, 15-30 minutes
        - "deep": 100+ sources, hours of research
        """
        depth_config = {
            "quick": {"sources": 10, "max_steps": 20},
            "medium": {"sources": 30, "max_steps": 50},
            "comprehensive": {"sources": 50, "max_steps": 100},
            "deep": {"sources": 100, "max_steps": 500}
        }
        
        config = depth_config.get(depth, depth_config["medium"])
        
        print(f"\n🔍 Autonomous Research Mode: {depth.upper()}")
        print(f"   Topic: {topic}")
        print(f"   Target sources: {config['sources']}")
        print(f"   Max autonomous steps: {config['max_steps']}\n")
        
        # Autonomous research phases
        phases = [
            "1. Initial search and source discovery",
            "2. Deep dive into promising sources",
            "3. Cross-reference and fact-checking",
            "4. Synthesis and organization",
            "5. Gap analysis and additional research",
            "6. Final report generation"
        ]
        
        results_by_phase = {}
        
        for phase in phases:
            print(f"   {phase}")
            # In real implementation, would perform actual research
            await asyncio.sleep(0.1)
            results_by_phase[phase] = f"Completed {phase}"
        
        return {
            "topic": topic,
            "depth": depth,
            "sources_consulted": config["sources"],
            "phases_completed": len(phases),
            "results": results_by_phase,
            "autonomy_level": self.model.autonomy_score,
            "time_saved_vs_human": "80-90%"  # AI is MUCH faster
        }
    
    async def _decide_next_action(self, query: str, steps_taken: List, findings: List) -> Dict[str, Any]:
        """Autonomously decide next action (simplified)"""
        if len(steps_taken) == 0:
            return {"type": "SEARCH", "description": f"Search for: {query}"}
        elif len(steps_taken) < 5:
            return {"type": "CLICK", "description": f"Click on result #{len(steps_taken)}"}
        elif len(findings) < 3:
            return {"type": "READ", "description": "Extract key information"}
        else:
            return {"type": "DONE", "description": "Sufficient information gathered"}
    
    async def _execute_action(self, action: Dict) -> Optional[str]:
        """Execute browser action (simplified)"""
        await asyncio.sleep(0.1)  # Simulate action time
        
        if action["type"] == "READ":
            return f"Finding from step: {action['description']}"
        return None
    
    def _synthesize_findings(self, findings: List[str]) -> str:
        """Synthesize research findings"""
        if not findings:
            return "No findings to synthesize"
        
        return f"Synthesized {len(findings)} findings from autonomous research"


class HumanLikeAI:
    """
    Human-like AI behavior and interaction patterns.
    
    Makes AI feel more natural by:
    - Thinking out loud
    - Showing reasoning process
    - Admitting uncertainty
    - Asking clarifying questions
    - Using natural language
    """
    
    def __init__(self, personality: str = "professional"):
        self.personality = personality
        self.interaction_patterns = {
            "professional": {
                "greeting": "Hello! I'm ready to help.",
                "thinking": "Let me think about this...",
                "uncertainty": "I'm not entirely sure, but my best assessment is:",
                "clarification": "Just to clarify, you're asking about:",
                "conclusion": "Based on my analysis:"
            },
            "friendly": {
                "greeting": "Hey there! What can I help you with?",
                "thinking": "Hmm, interesting question! Let me work through this...",
                "uncertainty": "I'm not 100% certain, but here's what I think:",
                "clarification": "Quick question - did you mean:",
                "conclusion": "So here's what I found:"
            },
            "expert": {
                "greeting": "Welcome. How may I assist you today?",
                "thinking": "Analyzing the parameters...",
                "uncertainty": "The data suggests, though with moderate confidence:",
                "clarification": "To ensure precision, could you specify:",
                "conclusion": "The analysis indicates:"
            }
        }
    
    def format_response(self, task: str, thinking_process: List[str], 
                       conclusion: str, confidence: float) -> str:
        """
        Format response with human-like elements.
        
        Shows:
        - Greeting/acknowledgment
        - Thinking process (transparency)
        - Uncertainty when appropriate (honesty)
        - Clear conclusion
        """
        patterns = self.interaction_patterns[self.personality]
        
        response = f"{patterns['greeting']}\n\n"
        response += f"**Task**: {task}\n\n"
        response += f"{patterns['thinking']}\n\n"
        
        # Show reasoning
        response += "**My Reasoning**:\n"
        for i, thought in enumerate(thinking_process, 1):
            response += f"{i}. {thought}\n"
        
        response += "\n"
        
        # Express uncertainty if confidence is low
        if confidence < 0.7:
            response += f"{patterns['uncertainty']}\n"
        
        response += f"{patterns['conclusion']}\n{conclusion}\n"
        
        return response
    
    def ask_clarifying_question(self, ambiguity: str) -> str:
        """Ask human-like clarifying question"""
        patterns = self.interaction_patterns[self.personality]
        return f"{patterns['clarification']} {ambiguity}"
    
    def show_progress(self, task: str, steps_completed: int, total_steps: int):
        """Show human-like progress updates"""
        percentage = int((steps_completed / total_steps) * 100)
        
        updates = [
            "Getting started...",
            "Making good progress...",
            "About halfway there...",
            "Almost done...",
            "Finishing up..."
        ]
        
        stage = min(int(percentage / 20), len(updates) - 1)
        print(f"   [{percentage}%] {updates[stage]}")


# Global instances
_browser = None
_human_ai = None

def get_agentic_browser(model_name: str = "claude-computer-use") -> AgenticBrowser:
    """Get agentic browser instance"""
    global _browser
    if _browser is None:
        model = AGENTIC_MODELS.get(model_name, AGENTIC_MODELS["claude-computer-use"])
        _browser = AgenticBrowser(model)
    return _browser


def get_human_like_ai(personality: str = "professional") -> HumanLikeAI:
    """Get human-like AI instance"""
    global _human_ai
    if _human_ai is None:
        _human_ai = HumanLikeAI(personality)
    return _human_ai


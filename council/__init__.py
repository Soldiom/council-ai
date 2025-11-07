"""
Council of Infinite Innovators - AI Agent Framework

A production-ready multi-agent system for building AI solutions with
specialized archetypes (Architect, Engineer, Strategist, etc.).
"""

from .graph import CouncilGraph
from .llm import openai_llm, get_llm_adapter
from .config import SETTINGS

# Import agents
from .agents.strategist import StrategistAgent
from .agents.architect import ArchitectAgent
from .agents.engineer import EngineerAgent
from .agents.designer import DesignerAgent
from .agents.entrepreneur import EntrepreneurAgent
from .agents.futurist import FuturistAgent
from .agents.economist import EconomistAgent
from .agents.ethicist import EthicistAgent
from .agents.philosopher import PhilosopherAgent
from .agents.cultural_translator import CulturalTranslatorAgent

__version__ = "0.1.0"
__author__ = "Council of Infinite Innovators"

# Available agent types
AGENT_TYPES = {
    "strategist": StrategistAgent,
    "architect": ArchitectAgent,
    "engineer": EngineerAgent,
    "designer": DesignerAgent,
    "entrepreneur": EntrepreneurAgent,
    "futurist": FuturistAgent,
    "economist": EconomistAgent,
    "ethicist": EthicistAgent,
    "philosopher": PhilosopherAgent,
    "cultural_translator": CulturalTranslatorAgent,
}

async def build_default_council():
    """Build a default council with all available agents."""
    llm = get_llm_adapter(SETTINGS.provider)
    
    agents = {}
    for name, agent_class in AGENT_TYPES.items():
        agents[name] = agent_class(llm)
    
    # Create synthesizer
    from .agents.base import BaseSynthesizer
    synthesizer = BaseSynthesizer(llm)
    
    return CouncilGraph(agents, synthesizer)

async def build_council(agent_names: list[str]):
    """Build a council with specific agents."""
    llm = get_llm_adapter(SETTINGS.provider)
    
    agents = {}
    for name in agent_names:
        if name not in AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {name}")
        agents[name] = AGENT_TYPES[name](llm)
    
    # Create synthesizer
    from .agents.base import BaseSynthesizer
    synthesizer = BaseSynthesizer(llm)
    
    return CouncilGraph(agents, synthesizer)
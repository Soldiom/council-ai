"""
Test suite for the Council of Infinite Innovators.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio
async def test_strategist_agent():
    """Test strategist agent basic functionality."""
    from council.agents.strategist import StrategistAgent
    from council.agents.base import Message
    
    # Mock LLM
    mock_llm = AsyncMock(return_value="Strategic recommendation: Focus on market validation.")
    
    # Create agent
    agent = StrategistAgent(mock_llm)
    
    # Test single query
    response = await agent.single_query("How should we launch our AI platform?")
    
    assert isinstance(response, str)
    assert len(response) > 0
    mock_llm.assert_called_once()

@pytest.mark.asyncio
async def test_architect_agent():
    """Test architect agent basic functionality."""
    from council.agents.architect import ArchitectAgent
    from council.agents.base import Message
    
    # Mock LLM
    mock_llm = AsyncMock(return_value="Architecture recommendation: Use microservices with API gateway.")
    
    # Create agent
    agent = ArchitectAgent(mock_llm)
    
    # Test single query
    response = await agent.single_query("Design a scalable AI platform architecture")
    
    assert isinstance(response, str)
    assert len(response) > 0
    mock_llm.assert_called_once()

@pytest.mark.asyncio
async def test_council_graph_single_agent():
    """Test council graph with single agent."""
    from council.graph import CouncilGraph
    from council.agents.strategist import StrategistAgent
    from council.agents.base import BaseSynthesizer
    
    # Mock LLM
    mock_llm = AsyncMock(return_value="Strategic insight about market entry.")
    
    # Create agents
    agents = {"strategist": StrategistAgent(mock_llm)}
    synthesizer = BaseSynthesizer(mock_llm)
    
    # Create graph
    graph = CouncilGraph(agents, synthesizer)
    
    # Test single agent run
    response = await graph.run_single_agent("Market strategy question", "strategist")
    
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_council_graph_multi_agent():
    """Test council graph with multiple agents."""
    from council.graph import CouncilGraph
    from council.agents.strategist import StrategistAgent
    from council.agents.architect import ArchitectAgent
    from council.agents.base import BaseSynthesizer
    
    # Mock LLM with different responses for each agent
    def mock_llm_side_effect(messages):
        # Simple mock that returns different responses based on agent context
        message_content = str(messages)
        if "strategist" in message_content.lower():
            return "Strategic perspective on the problem."
        elif "architect" in message_content.lower():
            return "Technical architecture perspective."
        else:
            return "Synthesized recommendation combining all perspectives."
    
    mock_llm = AsyncMock(side_effect=mock_llm_side_effect)
    
    # Create agents
    agents = {
        "strategist": StrategistAgent(mock_llm),
        "architect": ArchitectAgent(mock_llm)
    }
    synthesizer = BaseSynthesizer(mock_llm)
    
    # Create graph
    graph = CouncilGraph(agents, synthesizer)
    
    # Test multi-agent council
    response = await graph.run("Design a scalable AI platform", ["strategist", "architect"])
    
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Should have called LLM for each agent plus synthesis
    assert mock_llm.call_count >= 3

@pytest.mark.asyncio
async def test_build_default_council():
    """Test building default council with all agents."""
    import os
    
    # Skip if no API key (for CI/CD)
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OPENAI_API_KEY available")
    
    from council import build_default_council
    
    graph = await build_default_council()
    
    # Check that all expected agents are available
    available_agents = graph.get_available_agents()
    expected_agents = [
        "strategist", "architect", "engineer", "designer", 
        "entrepreneur", "futurist", "economist", "ethicist", 
        "philosopher", "cultural_translator"
    ]
    
    for agent in expected_agents:
        assert agent in available_agents

@pytest.mark.asyncio
async def test_build_custom_council():
    """Test building council with specific agents."""
    import os
    
    # Skip if no API key (for CI/CD)
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OPENAI_API_KEY available")
    
    from council import build_council
    
    graph = await build_council(["strategist", "architect"])
    
    available_agents = graph.get_available_agents()
    assert "strategist" in available_agents
    assert "architect" in available_agents
    assert len(available_agents) == 2

def test_config_validation():
    """Test configuration validation."""
    from council.config import Settings
    
    # Test with missing API key
    settings = Settings(provider="openai", openai_api_key="")
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        settings.validate()

def test_prompt_loading():
    """Test prompt loading functionality."""
    from council.prompts.loaders import load_archetype, list_available_archetypes
    
    # Test loading strategist prompt
    prompt = load_archetype("strategist")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    
    # Test listing archetypes
    archetypes = list_available_archetypes()
    assert isinstance(archetypes, list)
    assert "strategist" in archetypes

def test_message_creation():
    """Test message creation and conversion."""
    from council.agents.base import Message
    
    msg = Message(role="user", content="Test question")
    
    assert msg.role == "user"
    assert msg.content == "Test question"
    
    # Test conversion to dict
    msg_dict = msg.to_dict()
    assert msg_dict == {"role": "user", "content": "Test question"}

def test_agent_types_available():
    """Test that all expected agent types are available."""
    from council import AGENT_TYPES
    
    expected_types = [
        "strategist", "architect", "engineer", "designer",
        "entrepreneur", "futurist", "economist", "ethicist",
        "philosopher", "cultural_translator"
    ]
    
    for agent_type in expected_types:
        assert agent_type in AGENT_TYPES
        
    # Check that all agent classes are properly imported
    for agent_name, agent_class in AGENT_TYPES.items():
        assert hasattr(agent_class, "name")
        assert agent_class.name == agent_name
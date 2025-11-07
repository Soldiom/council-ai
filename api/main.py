"""
FastAPI server for the Council of Infinite Innovators.
"""

import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the project root to Python path for imports
sys.path.append('.')

from council import build_council, build_default_council, AGENT_TYPES

app = FastAPI(
    title="Council of Infinite Innovators API",
    description="Multi-agent AI system for collaborative problem-solving",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Council of Infinite Innovators API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Health check endpoint for cloud platforms."""
    return {"status": "healthy", "service": "council-api"}

# Request/Response models
class AgentRunRequest(BaseModel):
    """Request model for single agent consultation."""
    agent: str = Field(..., description="Agent name to consult")
    question: str = Field(..., description="Question or problem to address")
    model: Optional[str] = Field(None, description="Override default LLM model")
    provider: Optional[str] = Field(None, description="Override default LLM provider")

class CouncilRunRequest(BaseModel):
    """Request model for council consultation."""
    question: str = Field(..., description="Question or problem to address")
    agents: List[str] = Field(default=["strategist"], description="List of agents to consult")
    model: Optional[str] = Field(None, description="Override default LLM model")
    provider: Optional[str] = Field(None, description="Override default LLM provider")

class AgentResponse(BaseModel):
    """Response model for agent consultations."""
    agent: str
    question: str
    response: str
    model_used: Optional[str] = None
    provider_used: Optional[str] = None

class CouncilResponse(BaseModel):
    """Response model for council consultations."""
    question: str
    agents: List[str]
    synthesis: str
    model_used: Optional[str] = None
    provider_used: Optional[str] = None

class AgentInfo(BaseModel):
    """Information about an available agent."""
    name: str
    description: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    available_agents: int

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🏛️ Council of Infinite Innovators API",
        "version": "0.1.0",
        "docs": "/docs",
        "agents": "/agents",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from council import __version__
    return HealthResponse(
        status="healthy",
        version=__version__,
        available_agents=len(AGENT_TYPES)
    )

@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """List all available agents."""
    agent_descriptions = {
        "strategist": "Market intelligence and strategic positioning expert",
        "architect": "System design and scalability expert",
        "engineer": "Implementation and production readiness expert",
        "designer": "User experience and interface design expert",
        "entrepreneur": "Business opportunities and market validation expert",
        "futurist": "Trend analysis and future scenarios expert",
        "economist": "Financial modeling and economic analysis expert",
        "ethicist": "Responsible AI and ethical considerations expert",
        "philosopher": "Fundamental assumptions and deeper implications expert",
        "cultural_translator": "Cross-cultural adaptation and localization expert",
    }
    
    return [
        AgentInfo(name=name, description=agent_descriptions.get(name, "Specialized AI agent"))
        for name in sorted(AGENT_TYPES.keys())
    ]

@app.post("/agents/run", response_model=AgentResponse)
async def run_single_agent(request: AgentRunRequest):
    """Run a consultation with a single agent."""
    try:
        # Validate agent
        if request.agent not in AGENT_TYPES:
            available = ", ".join(AGENT_TYPES.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent: {request.agent}. Available: {available}"
            )
        
        # Override settings if provided
        if request.model or request.provider:
            from council.config import SETTINGS
            if request.model:
                SETTINGS.model = request.model
            if request.provider:
                SETTINGS.provider = request.provider
        
        # Build council and run consultation
        graph = await build_council([request.agent])
        response = await graph.run_single_agent(request.question, request.agent)
        
        # Get current settings for response
        from council.config import SETTINGS
        
        return AgentResponse(
            agent=request.agent,
            question=request.question,
            response=response,
            model_used=SETTINGS.model,
            provider_used=SETTINGS.provider
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/council/run", response_model=CouncilResponse)
async def run_council(request: CouncilRunRequest):
    """Run a consultation with multiple agents (full council)."""
    try:
        # Validate agents
        unknown_agents = [a for a in request.agents if a not in AGENT_TYPES]
        if unknown_agents:
            available = ", ".join(AGENT_TYPES.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agents: {unknown_agents}. Available: {available}"
            )
        
        # Override settings if provided
        if request.model or request.provider:
            from council.config import SETTINGS
            if request.model:
                SETTINGS.model = request.model
            if request.provider:
                SETTINGS.provider = request.provider
        
        # Build council and run consultation
        graph = await build_council(request.agents)
        synthesis = await graph.run(request.question, request.agents)
        
        # Get current settings for response
        from council.config import SETTINGS
        
        return CouncilResponse(
            question=request.question,
            agents=request.agents,
            synthesis=synthesis,
            model_used=SETTINGS.model,
            provider_used=SETTINGS.provider
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Development endpoint for testing
@app.post("/test/strategist")
async def test_strategist(question: str = "Draft a go-to-market strategy for an AI platform"):
    """Quick test endpoint for strategist agent."""
    try:
        graph = await build_council(["strategist"])
        response = await graph.run_single_agent(question, "strategist")
        return {"question": question, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
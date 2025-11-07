"""
Unified API - Public endpoint for users to access ALL models
=============================================================

Users interact with YOUR platform → generates training data → YOUR model improves daily

Features:
1. Auto-route to best model for task
2. Track all interactions
3. Collect training data
4. Daily fine-tuning
5. Auto-update with new HF models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
from council.model_hub import get_hub, UnifiedModelTrainer
from council.model_ensemble import get_ensemble
from transformers import pipeline
import uuid

app = FastAPI(
    title="Unified AI Platform",
    description="Access ALL HuggingFace models through one API. YOUR data trains YOUR model!",
    version="1.0.0",
)


class TaskRequest(BaseModel):
    """User task request."""
    task: str = Field(..., description="What you want to do (e.g., 'translate to French', 'generate image')")
    input: str = Field(..., description="Input text/data")
    capability: Optional[str] = Field(None, description="Specific capability (auto-detected if not provided)")
    user_id: Optional[str] = Field(None, description="Your user ID (for personalization)")


class TaskResponse(BaseModel):
    """Task result."""
    output: Any
    model_used: str
    capability: str
    agent: str
    timestamp: str
    interaction_id: str


class FeedbackRequest(BaseModel):
    """User feedback on result."""
    interaction_id: str
    rating: int = Field(..., ge=1, le=5, description="1-5 stars")
    comment: Optional[str] = None


# Initialize
hub = get_hub()
trainer = UnifiedModelTrainer(hub)


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    print("🚀 Starting Unified AI Platform...")
    
    # Load or discover models
    if not hub.load_cache():
        print("🔍 No cache found, discovering models...")
        print("⚠️ This may take a few minutes...")
        # In production, run this in background
        # For now, use cached data or discover on first request
    
    print("✅ Ready!")


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Unified AI Platform",
        "description": "Access ALL HuggingFace models through one API",
        "version": "1.0.0",
        "features": [
            "Auto-route to best model",
            "Support for text, image, audio, video",
            "Continuous learning from your usage",
            "Daily model updates",
            "YOUR data trains YOUR model",
        ],
        "endpoints": {
            "POST /task": "Submit a task",
            "POST /feedback": "Provide feedback",
            "GET /capabilities": "List all capabilities",
            "GET /models": "List all available models",
            "GET /stats": "Platform statistics",
        }
    }


@app.get("/capabilities")
async def list_capabilities():
    """List all available capabilities."""
    if not hub.capabilities:
        await hub.discover_all_models(top_n_per_category=5)
    
    return {
        "capabilities": list(hub.capabilities.keys()),
        "total": len(hub.capabilities),
        "agents": hub.get_all_agents(),
    }


@app.get("/models")
async def list_models():
    """List all available models."""
    if not hub.capabilities:
        await hub.discover_all_models(top_n_per_category=5)
    
    models_by_capability = {}
    for capability, models in hub.capabilities.items():
        models_by_capability[capability] = [
            {
                "model_id": m.model_id,
                "downloads": m.downloads,
                "likes": m.likes,
                "score": m.score(),
            }
            for m in models
        ]
    
    return {
        "models": models_by_capability,
        "total": sum(len(v) for v in hub.capabilities.values()),
    }


@app.post("/task", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
):
    """
    Execute a task using the best model.
    
    Examples:
        - task: "translate", input: "Hello world" → Detects translation model
        - task: "generate image", input: "a cat" → Detects text-to-image
        - task: "summarize", input: "long text..." → Detects summarization
    """
    # Ensure models are discovered
    if not hub.capabilities:
        await hub.discover_all_models(top_n_per_category=3)
    
    # Route to best model
    routing = hub.route_request(
        task=request.task,
        input_text=request.input,
        preferred_capability=request.capability,
    )
    
    # Execute task
    try:
        # Use HuggingFace pipeline
        pipe = pipeline(
            routing["pipeline_tag"],
            model=routing["model_id"],
            token=os.getenv("HF_API_TOKEN"),
        )
        
        # Execute
        if routing["pipeline_tag"] in ["text-generation", "text2text-generation"]:
            result = pipe(request.input, max_length=500, num_return_sequences=1)
            output = result[0]["generated_text"]
        elif routing["pipeline_tag"] == "summarization":
            result = pipe(request.input, max_length=130, min_length=30)
            output = result[0]["summary_text"]
        elif routing["pipeline_tag"] == "translation":
            result = pipe(request.input)
            output = result[0]["translation_text"]
        else:
            result = pipe(request.input)
            output = result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")
    
    # Create response
    interaction_id = str(uuid.uuid4())
    response = TaskResponse(
        output=output,
        model_used=routing["model_id"],
        capability=routing["capability"],
        agent=routing["agent"],
        timestamp=datetime.now().isoformat(),
        interaction_id=interaction_id,
    )
    
    # Collect training data in background
    background_tasks.add_task(
        trainer.collect_interaction,
        user_id=request.user_id or "anonymous",
        capability=routing["capability"],
        model_id=routing["model_id"],
        input_data=request.input,
        output_data=output,
    )
    
    return response


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback on a result.
    
    Feedback improves YOUR model quality!
    """
    # Store feedback (will be used in training)
    feedback_file = trainer.training_data_dir / "feedback.jsonl"
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "interaction_id": feedback.interaction_id,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
    
    return {"status": "success", "message": "Feedback recorded. Thank you!"}


@app.get("/stats")
async def platform_stats():
    """Platform usage statistics."""
    # Count interactions
    interaction_files = list(trainer.training_data_dir.glob("interactions_*.jsonl"))
    total_interactions = 0
    
    for file in interaction_files:
        with open(file) as f:
            total_interactions += sum(1 for _ in f)
    
    return {
        "total_interactions": total_interactions,
        "total_models": sum(len(v) for v in hub.capabilities.values()) if hub.capabilities else 0,
        "total_capabilities": len(hub.capabilities) if hub.capabilities else 0,
        "ready_for_training": total_interactions >= 100,
        "next_training": "Daily at midnight UTC",
        "your_model_status": "Training daily from your usage!",
    }


@app.post("/admin/discover-models")
async def admin_discover_models(background_tasks: BackgroundTasks):
    """
    Admin: Discover new models from HuggingFace.
    
    Run daily to stay up-to-date with latest models.
    """
    background_tasks.add_task(hub.discover_all_models, top_n_per_category=5)
    return {"status": "started", "message": "Model discovery running in background"}


@app.post("/admin/train-unified-model")
async def admin_train_model():
    """
    Admin: Train YOUR unified model on collected data.
    
    Run daily to improve YOUR model.
    """
    result = await trainer.train_unified_model_daily()
    return {"status": "success", "result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

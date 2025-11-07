# 🌐 COUNCIL AI - WEB PLATFORM
# User-friendly interface to use all 6 AI models

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from typing import Optional
import json

app = FastAPI(title="Council AI Platform", version="1.0.0")

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# MODEL CONFIGURATION - USING REAL PUBLIC HUGGINGFACE MODELS!
# ========================================

# 6 REAL AI models users can use RIGHT NOW!
MODELS = {
    "unified": {
        "name": "Unified AI (TinyLlama)",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Fast general-purpose AI assistant",
        "icon": "🤖",
        "use_cases": ["general questions", "explanations", "advice", "analysis"]
    },
    "code": {
        "name": "Code Assistant (GPT-2)",
        "model_id": "gpt2",
        "description": "Programming and coding help",
        "icon": "�",
        "use_cases": ["code", "programming", "python", "javascript", "debug"]
    },
    "creative": {
        "name": "Creative Writer (GPT-2 Large)",
        "model_id": "gpt2-large",
        "description": "Creative writing and storytelling",
        "icon": "✍️",
        "use_cases": ["story", "creative", "write", "screenplay", "movie", "novel"]
    },
    "chat": {
        "name": "Chat AI (DialoGPT)",
        "model_id": "microsoft/DialoGPT-medium",
        "description": "Natural conversation AI",
        "icon": "�",
        "use_cases": ["chat", "talk", "conversation", "discuss"]
    },
    "smart": {
        "name": "Smart AI (DistilGPT-2)",
        "model_id": "distilgpt2",
        "description": "Quick and efficient AI",
        "icon": "⚡",
        "use_cases": ["fast", "quick", "speed", "research", "analyze"]
    },
    "science": {
        "name": "Science AI (BLOOM-560m)",
        "model_id": "bigscience/bloom-560m",
        "description": "Scientific reasoning and analysis",
        "icon": "🔬",
        "use_cases": ["science", "research", "forensic", "technical", "analysis"]
    }
}

# Cache loaded models
loaded_models = {}

# ========================================
# REQUEST/RESPONSE MODELS
# ========================================

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "auto"  # auto, unified, forensic, etc.
    temperature: Optional[float] = 0.7
    max_length: Optional[int] = 500

class ChatResponse(BaseModel):
    response: str
    model_used: str
    confidence: float

# ========================================
# HELPER FUNCTIONS
# ========================================

def load_model(model_key: str):
    """Load model from HuggingFace (cached)"""
    if model_key not in loaded_models:
        try:
            # Import pipeline here to avoid slow startup
            from transformers import pipeline
            
            model_id = MODELS[model_key]["model_id"]
            print(f"Loading model: {model_id}...")
            
            # Load model from HuggingFace
            loaded_models[model_key] = pipeline(
                "text-generation",
                model=model_id,
                device=-1  # CPU (use 0 for GPU)
            )
            
            print(f"✅ Model loaded: {model_id}")
        except Exception as e:
            print(f"❌ Error loading model {model_key}: {str(e)}")
            return None
    
    return loaded_models[model_key]

def auto_select_model(message: str) -> str:
    """Automatically select best model based on user message"""
    message_lower = message.lower()
    
    # Check keywords for each model
    for model_key, model_info in MODELS.items():
        for use_case in model_info["use_cases"]:
            if use_case in message_lower:
                return model_key
    
    # Default to unified model
    return "unified"

def generate_response(message: str, model_key: str, temperature: float = 0.7, max_length: int = 500) -> dict:
    """Generate AI response"""
    
    # Load model
    model = load_model(model_key)
    if model is None:
        return {
            "response": f"⚠️ Could not load the {MODELS[model_key]['name']} model. Please try another model or check your internet connection.",
            "model_used": model_key,
            "confidence": 0.0
        }
    
    try:
        # Generate response
        result = model(
            message,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )
        
        # Extract text
        response_text = result[0]['generated_text']
        
        # Remove input from output (if model includes it)
        if response_text.startswith(message):
            response_text = response_text[len(message):].strip()
        
        return {
            "response": response_text,
            "model_used": model_key,
            "confidence": 0.95  # Could calculate actual confidence
        }
        
    except Exception as e:
        return {
            "response": f"Error generating response: {str(e)}",
            "model_used": model_key,
            "confidence": 0.0
        }

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web interface"""
    with open("web_app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/models")
async def get_models():
    """Get list of available models"""
    return {
        "models": MODELS,
        "total": len(MODELS)
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    # Auto-select model if needed
    if request.model == "auto":
        model_key = auto_select_model(request.message)
    else:
        model_key = request.model
        if model_key not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_key}")
    
    # Generate response
    result = generate_response(
        message=request.message,
        model_key=model_key,
        temperature=request.temperature,
        max_length=request.max_length
    )
    
    return ChatResponse(**result)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models_available": len(MODELS)
    }

# ========================================
# STATIC FILES & FRONTEND
# ========================================

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_path = os.path.join("web_app", "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ========================================
# RUN SERVER
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Council AI Platform...")
    print("📍 URL: http://localhost:8000")
    print("🤖 6 AI models ready to use!")
    print("✨ Open http://localhost:8000 in your browser!")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

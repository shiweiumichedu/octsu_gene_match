#!/usr/bin/env python3
"""
FastAPI Web Application for Precision Oncology Gene Matching System
Provides ChatGPT-style interface for F1 and Tempus gene matching
"""

import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Import the existing matching systems
from f1_matcher import F1MatchingSystem
from tempus_matcher import TempusMatchingSystem

# Load environment variables
load_dotenv('.env', override=True)

# Initialize FastAPI app
app = FastAPI(
    title="Precision Oncology Gene Matching System",
    description="Web interface for matching genomic test results to clinical trials",
    version="1.0.0"
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize matching systems
f1_system = None
tempus_system = None

class ChatMessage(BaseModel):
    message: str
    system_type: str  # "f1" or "tempus"

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    system_type: str

# Store chat history in memory (in production, use a database)
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the matching systems on startup"""
    global f1_system, tempus_system
    
    print("🚀 Starting Precision Oncology Gene Matching System...")
    
    try:
        # Initialize F1 matching system
        print("📊 Initializing F1 matching system...")
        f1_system = F1MatchingSystem()
        await f1_system.initialize()
        print("✅ F1 system ready")
        
        # Initialize Tempus matching system
        print("🧬 Initializing Tempus matching system...")
        tempus_system = TempusMatchingSystem()
        await tempus_system.initialize()
        print("✅ Tempus system ready")
        
        print("🎉 All systems initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        # Continue startup even if systems fail to initialize
        pass

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "f1_system": "ready" if f1_system and f1_system.is_ready() else "not ready",
        "tempus_system": "ready" if tempus_system and tempus_system.is_ready() else "not ready",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, request: Request):
    """Handle chat messages and return responses"""
    
    # Get or create session ID (using IP as simple session identifier)
    session_id = request.client.host
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    try:
        # Route to appropriate matching system
        if message.system_type == "f1":
            if not f1_system or not f1_system.is_ready():
                raise HTTPException(status_code=503, detail="F1 matching system not available")
            
            response_text = await f1_system.process_query(message.message)
            
        elif message.system_type == "tempus":
            if not tempus_system or not tempus_system.is_ready():
                raise HTTPException(status_code=503, detail="Tempus matching system not available")
            
            response_text = await tempus_system.process_query(message.message)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid system_type. Use 'f1' or 'tempus'")
        
        # Store in chat history
        timestamp = datetime.now().isoformat()
        chat_sessions[session_id].append({
            "user_message": message.message,
            "ai_response": response_text,
            "system_type": message.system_type,
            "timestamp": timestamp
        })
        
        # Keep only last 50 messages per session
        if len(chat_sessions[session_id]) > 50:
            chat_sessions[session_id] = chat_sessions[session_id][-50:]
        
        return ChatResponse(
            response=response_text,
            timestamp=timestamp,
            system_type=message.system_type
        )
        
    except Exception as e:
        print(f"❌ Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/chat/history")
async def get_chat_history(request: Request):
    """Get chat history for the current session"""
    session_id = request.client.host
    history = chat_sessions.get(session_id, [])
    return {"history": history}

@app.post("/chat/clear")
async def clear_chat_history(request: Request):
    """Clear chat history for the current session"""
    session_id = request.client.host
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return {"message": "Chat history cleared"}

@app.get("/systems/status")
async def get_systems_status():
    """Get detailed status of both matching systems"""
    return {
        "f1_system": {
            "available": f1_system is not None,
            "ready": f1_system.is_ready() if f1_system else False,
            "stats": f1_system.get_stats() if f1_system and f1_system.is_ready() else None
        },
        "tempus_system": {
            "available": tempus_system is not None,
            "ready": tempus_system.is_ready() if tempus_system else False,
            "stats": tempus_system.get_stats() if tempus_system and tempus_system.is_ready() else None
        }
    }

@app.get("/api/examples")
async def get_example_queries():
    """Get example queries for both systems"""
    return {
        "f1_examples": [
            "What clinical trials are available for BRCA1 mutations?",
            "Match all F1 reports to available studies",
            "What studies match patients with HRD mutations?",
            "Which F1 reports have BRCA1 mutations?",
            "Show study details for NCT05367440"
        ],
        "tempus_examples": [
            "Find trials for patients with KRAS mutations",
            "Match all Tempus reports to available studies", 
            "What mutations are found in Tempus reports?",
            "Show studies for patients with MSI-H status",
            "Which Tempus reports match immunotherapy trials?"
        ]
    }

if __name__ == "__main__":
    # Run the application
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🌐 Starting server on http://{host}:{port}")
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

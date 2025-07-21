#!/usr/bin/env python3
"""
FastAPI application for Precision Oncology Gene Matching System
with tabbed interface for F1 and Tempus matching systems
"""

import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import the matching system components
from f1_matcher_tabbed import F1MatchingSystem
from tempus_matcher_tabbed import TempusMatchingSystem

# Initialize FastAPI app
app = FastAPI(
    title="Precision Oncology Gene Matching System",
    description="Match genomic test results to clinical trial eligibility with tabbed interface",
    version="2.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global instances of matching systems
f1_system: Optional[F1MatchingSystem] = None
tempus_system: Optional[TempusMatchingSystem] = None

# Request models
class ChatRequest(BaseModel):
    message: str
    system_type: str  # "f1" or "tempus"

class ChatResponse(BaseModel):
    response: str
    timestamp: str

class SystemStatus(BaseModel):
    ready: bool
    message: str

# Store conversation history
conversation_history: Dict[str, List] = {
    "f1": [],
    "tempus": []
}

@app.on_event("startup")
async def startup_event():
    """Initialize the matching systems on startup."""
    global f1_system, tempus_system
    
    print("🌐 Starting tabbed server on http://0.0.0.0:8000")
    print("🚀 Starting Precision Oncology Gene Matching System with Tabs...")
    
    try:
        # Initialize F1 system
        print("📊 Initializing F1 matching system...")
        f1_system = F1MatchingSystem()
        await f1_system.initialize()
        print("✅ F1 system ready")
        
        # Initialize Tempus system
        print("🧬 Initializing Tempus matching system...")
        tempus_system = TempusMatchingSystem()
        await tempus_system.initialize()
        print("✅ Tempus system ready")
        
        print("🎉 All systems initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page with tabbed interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests for both systems."""
    try:
        system_type = request.system_type.lower()
        message = request.message.strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if system_type not in ["f1", "tempus"]:
            raise HTTPException(status_code=400, detail="Invalid system type. Must be 'f1' or 'tempus'")
        
        # Get the appropriate system
        if system_type == "f1":
            if not f1_system or not f1_system.is_ready():
                raise HTTPException(status_code=503, detail="F1 system not ready")
            
            # Get chat history for this system
            chat_history = conversation_history["f1"]
            
            # Process the message
            response = await f1_system.process_query(message, chat_history)
            
            # Update conversation history
            conversation_history["f1"].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ])
            
        else:  # tempus
            if not tempus_system or not tempus_system.is_ready():
                raise HTTPException(status_code=503, detail="Tempus system not ready")
            
            # Get chat history for this system
            chat_history = conversation_history["tempus"]
            
            # Process the message
            response = await tempus_system.process_query(message, chat_history)
            
            # Update conversation history
            conversation_history["tempus"].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ])
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat/clear")
async def clear_chat_history():
    """Clear conversation history for both systems."""
    try:
        conversation_history["f1"] = []
        conversation_history["tempus"] = []
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        print(f"❌ Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.get("/systems/status")
async def get_systems_status():
    """Get the status of both matching systems."""
    try:
        f1_status = SystemStatus(
            ready=f1_system is not None and f1_system.is_ready(),
            message="F1 system operational" if f1_system and f1_system.is_ready() else "F1 system not ready"
        )
        
        tempus_status = SystemStatus(
            ready=tempus_system is not None and tempus_system.is_ready(),
            message="Tempus system operational" if tempus_system and tempus_system.is_ready() else "Tempus system not ready"
        )
        
        return {
            "f1_system": f1_status.dict(),
            "tempus_system": tempus_status.dict(),
            "overall_status": "ready" if f1_status.ready and tempus_status.ready else "partial"
        }
        
    except Exception as e:
        print(f"❌ Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "f1": f1_system is not None and f1_system.is_ready(),
            "tempus": tempus_system is not None and tempus_system.is_ready()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

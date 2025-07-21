
# Precision Oncology Gene Matching System

## Setup

### 1. Install Dependencies
```bash
# Install from requirements.txt
C:/repo/github/octsu_gene_match/.venv/Scripts/python.exe -m pip install -r requirements.txt

# Or install individual packages
C:/repo/github/octsu_gene_match/.venv/Scripts/python.exe -m pip install fastapi uvicorn[standard] pydantic python-dotenv langchain langchain-openai langchain-community langchain-core langchain-huggingface azure-openai faiss-cpu jinja2 python-multipart
```

### 2. Environment Setup
Ensure you have a `.env` file with your Azure OpenAI credentials.

## Launch Locally

### Development Mode (with auto-reload):
```bash
C:/repo/github/octsu_gene_match/.venv/Scripts/python.exe -m uvicorn app_tabbed:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode:
```bash
C:/repo/github/octsu_gene_match/.venv/Scripts/python.exe -m uvicorn app_tabbed:app --host 0.0.0.0 --port 8000
```

### Alternative (if .venv is activated):
```bash
uvicorn app_tabbed:app --host 0.0.0.0 --port 8000 --reload
```

## Access
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Features
✅ **Tabbed Interface**: Switch between F1 and Tempus matching systems  
✅ **ChatGPT-style UI**: Modern chat interface with message bubbles  
✅ **Real-time Status**: Shows system readiness indicators  
✅ **Example Queries**: Click buttons for sample questions  
✅ **Responsive Design**: Works on desktop and mobile


# Precision Oncology Gene Matching Web Application

A ChatGPT-style web interface for matching genomic test results to clinical trial eligibility using both F1 (Foundation One) and Tempus systems.

## Features

- **Dual System Support**: Both F1 and Tempus gene matching systems
- **ChatGPT-style Interface**: Interactive conversational interface
- **Real-time Matching**: Instant gene-to-trial matching
- **Visual Status Indicators**: System health monitoring
- **Example Queries**: Quick-start buttons for common searches

## Quick Start

1. **Start the Application**:
   ```bash
   python app.py
   ```

2. **Open in Browser**:
   Navigate to `http://localhost:8000`

3. **Use the Interface**:
   - Choose between F1 or Tempus system
   - Type your gene matching queries
   - Use example buttons for quick testing
   - View real-time system status

## Example Queries

### F1 System
- "What clinical trials are available for BRCA1 mutations?"
- "Match all F1 reports to available studies"
- "What studies match patients with HRD mutations?"
- "Which F1 reports have BRCA1 mutations?"

### Tempus System
- "Find trials for patients with KRAS mutations"
- "Match all Tempus reports to available studies"
- "What mutations are found in Tempus reports?"
- "Show studies for patients with MSI-H status"

## API Endpoints

- `GET /` - Main web interface
- `POST /chat` - Send chat messages
- `GET /health` - Health check
- `GET /systems/status` - Detailed system status
- `GET /chat/history` - Get chat history
- `POST /chat/clear` - Clear chat history
- `GET /api/examples` - Get example queries

## System Status

The web interface shows real-time status for both systems:
- 🟢 **Green**: System ready and operational
- 🔴 **Red**: System not ready or error

## Configuration

The application uses the same `.env` configuration file as the original systems:

```properties
# Azure OpenAI Configuration
OPENAI_API_KEY000=your_api_key
OPENAI_API_BASE000=https://api.umgpt.umich.edu/azure-openai-api
OPENAI_API_VERSION=2024-06-01
CHAT_DEPLOYMENT_NAME=gpt-4o
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Vector Database Paths
OCTSU_INDEX_PATH=faiss2_index
F1_INDEX_PATH=f1_json_index
TEMPUS_INDEX_PATH=tempus_json_index

# Email Configuration (Optional)
MAIL_HOST=mail-04.med.umich.edu
MAIL_PORT=25
MAIL_FROM=tumor-gene-trial-match-support@med.umich.edu
```

## Architecture

- **FastAPI**: Web framework for API endpoints
- **Jinja2**: HTML templating
- **LangChain**: RAG (Retrieval-Augmented Generation) framework
- **FAISS**: Vector database for similarity search
- **Azure OpenAI**: GPT-4o for natural language processing
- **HuggingFace**: Embedding models for F1/Tempus data

## Dependencies

Main packages required:
- `fastapi`
- `uvicorn`
- `jinja2`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-huggingface`
- `faiss-cpu`
- `sentence-transformers`

## Development

To run in development mode with auto-reload:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

For production, consider:
- Using a proper database for chat history
- Adding authentication and authorization
- Implementing proper logging
- Using a production WSGI server
- Adding rate limiting
- Implementing proper error handling and monitoring

## Troubleshooting

1. **Systems Not Ready**: Check vector database paths in `.env`
2. **Connection Errors**: Verify Azure OpenAI credentials
3. **Embedding Issues**: Ensure FAISS indexes are compatible
4. **Memory Issues**: Reduce search result counts if needed

## Support

For issues related to:
- Gene matching algorithms: Check the original F1/Tempus scripts
- Vector databases: Verify index files and embedding models
- Azure OpenAI: Confirm API keys and endpoints
- Web interface: Check browser console for JavaScript errors

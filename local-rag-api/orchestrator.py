import os
import shutil
import logging
import mimetypes
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Service imports
from rag_service import ingest_document, get_raw_retrieval, delete_document_vectors
from agent_service import run_agent_pipeline

# Ensure PDF MIME type is registered
mimetypes.add_type("application/pdf", ".pdf")

# Configure logging
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 1. Console Handler (for terminal)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. File Handler (for Blazor UI logs)
LOG_FILE = os.path.join(BASE_DIR, "data", "app.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)

# Apply handlers to the root logger
logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger("rag_api")

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",            
    openapi_url="/openapi.json"
)

# Enable CORS for frontend clients (Blazor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/documents/{filename}")
async def get_document(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        content_disposition_type="inline"
    )

@app.get("/")
def read_root():
    """Silences the root 404 error by returning a basic status."""
    return {"status": "Local RAG API is running"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silences the favicon 404 error by returning an empty image."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/api/logs")
def get_recent_logs():
    """Returns the last 100 lines of the application log."""
    if not os.path.exists(LOG_FILE):
        return {"logs": ["Log file not created yet..."]}
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        
    return {"logs": lines[-100:]}


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received file upload request: {file.filename}")
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunks_count = ingest_document(file_path)
        logger.info(f"Successfully processed upload: {safe_filename} ({chunks_count} chunks created)")
        return {"filename": safe_filename, "status": "Success", "chunks_created": chunks_count}
    except Exception as e:
        logger.exception(f"Failed to process uploaded file: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def list_documents():
    """Returns a list of all ingested documents."""
    if not os.path.exists(UPLOAD_DIR):
        return {"documents": []}

    docs = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    return {"documents": docs}


@app.delete("/api/documents/{filename}")
def remove_document(filename: str):
    """Deletes a document from both ChromaDB and the file system."""
    logger.info(f"Received deletion request for: {filename}")
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # 1. Delete vectors from ChromaDB
        delete_document_vectors(safe_filename)
        
        # 2. Delete physical file from the disk
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted physical file: {safe_filename}")
            
        return {"status": "Success", "message": f"Completely removed {safe_filename}"}
    except Exception as e:
        logger.exception(f"Failed to delete {filename}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/raw-search")
async def raw_search(request: QueryRequest):
    logger.info(f"Received raw search request for query: '{request.question}'")
    try:
        context_data = get_raw_retrieval(request.question)
        sources_count = len(context_data.get("sources", []))
        logger.info(f"Raw search successful. Retrieved {sources_count} sources.")
        return {
            "context": context_data["context"], 
            "sources": context_data["sources"]
        }
    except Exception as e:
        logger.exception(f"Error during raw search for query: '{request.question}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def ask_question(request: QueryRequest): 
    # Synchronous function runs inside FastAPI's threadpool automatically
    logger.info(f"Received LLM query request: '{request.question}'")
    try:
        result = run_agent_pipeline(request.question)
        logger.info("Agent pipeline executed successfully. Returning answer.")
        return result
    except Exception as e:
        logger.exception(f"Error during LLM query processing for: '{request.question}'")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("orchestrator:app", host="127.0.0.1", port=8000, reload=True)
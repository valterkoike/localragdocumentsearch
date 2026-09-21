"""
Orchestrator API Service (FastAPI)
This is the main entry point and gateway for the backend. It exposes HTTP endpoints
for the frontend (e.g., a Blazor UI) to upload documents, query the RAG pipeline, 
manage files, and retrieve system logs.
"""

import os
import shutil
import logging
import mimetypes
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Internal RAG and LLM service imports
from rag_service import ingest_document, get_raw_retrieval, delete_document_vectors
from agent_service import run_agent_pipeline

# Ensure PDF MIME type is registered so browsers display them inline 
# in the frontend viewer rather than forcing a download.
mimetypes.add_type("application/pdf", ".pdf")

# ---------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 1. Console Handler (for real-time terminal monitoring)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Resolve the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. File Handler (Used to power the frontend's live log viewer)
LOG_FILE = os.path.join(BASE_DIR, "data", "app.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)

# Apply handlers to the root logger
logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------
# FASTAPI APP INITIALIZATION
# ---------------------------------------------------------
app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",            
    openapi_url="/openapi.json"
)

# Enable CORS (Cross-Origin Resource Sharing)
# This is required so the frontend app can make HTTP requests
# to this Python backend without being blocked by the browser's security policies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local directory to store physical copies of uploaded PDFs
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------
# UTILITY & SYSTEM ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def read_root():
    """Silences the root 404 error by returning a basic health check status."""
    return {"status": "Local RAG API is running"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silences the favicon 404 error commonly triggered by browsers."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/api/logs")
def get_recent_logs():
    """
    Returns the last 100 lines of the application log.
    The frontend polls this endpoint to display a live terminal in the UI.
    """
    if not os.path.exists(LOG_FILE):
        return {"logs": ["Log file not created yet..."]}
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        
    return {"logs": lines[-100:]}


# ---------------------------------------------------------
# DOCUMENT MANAGEMENT ENDPOINTS
# ---------------------------------------------------------

@app.get("/documents/{filename}")
async def get_document(filename: str):
    """
    Serves the physical PDF file back to the frontend for inline viewing.
    """
    # os.path.basename prevents directory traversal attacks (e.g., passing "../../../etc/passwd")
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        content_disposition_type="inline" # Tells the browser to display, not download
    )


@app.get("/api/documents")
def list_documents():
    """Returns a list of all currently ingested documents for the frontend sidebar."""
    if not os.path.exists(UPLOAD_DIR):
        return {"documents": []}

    # Filter out directories, returning only files
    docs = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    return {"documents": docs}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives a PDF upload, saves it to disk, and triggers the vector database ingestion process.
    """
    logger.info(f"Received file upload request: {file.filename}")
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # 1. Save the physical file using chunks to avoid memory overflow on large files
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Pass the file to the RAG service to chunk and embed into ChromaDB
        chunks_count = ingest_document(file_path)
        logger.info(f"Successfully processed upload: {safe_filename} ({chunks_count} chunks created)")
        
        return {"filename": safe_filename, "status": "Success", "chunks_created": chunks_count}
        
    except Exception as e:
        logger.exception(f"Failed to process uploaded file: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename}")
def remove_document(filename: str):
    """
    Completely deletes a document from the system. Requires a two-step cleanup.
    """
    logger.info(f"Received deletion request for: {filename}")
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Step 1: Purge the text embeddings from ChromaDB so it stops showing up in searches
        delete_document_vectors(safe_filename)
        
        # Step 2: Delete the physical PDF file from the local disk
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted physical file: {safe_filename}")
            
        return {"status": "Success", "message": f"Completely removed {safe_filename}"}
        
    except Exception as e:
        logger.exception(f"Failed to delete {filename}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# AI & SEARCH ENDPOINTS
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    """Pydantic model for input validation on search/query endpoints."""
    question: str


@app.post("/raw-search")
async def raw_search(request: QueryRequest):
    """
    Performs vector similarity search only. 
    Bypasses the LLM entirely and returns the raw chunks found in the database.
    Useful for debugging or testing retrieval quality.
    """
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
    """
    Full RAG Pipeline Endpoint.
    Takes a user question, finds relevant context, and generates an LLM answer.
    
    Defined as synchronous (`def` instead of `async def`). FastAPI automatically 
    runs sync endpoints in a managed threadpool so the heavy LLM generation doesn't 
    block the main server loop.
    """
    logger.info(f"Received LLM query request: '{request.question}'")
    try:
        result = run_agent_pipeline(request.question)
        logger.info("Agent pipeline executed successfully. Returning answer.")
        return result
    except Exception as e:
        logger.exception(f"Error during LLM query processing for: '{request.question}'")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# SERVER EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # Run the server locally on port 8000. 'reload=True' auto-restarts the server on code changes.
    uvicorn.run("orchestrator:app", host="127.0.0.1", port=8000, reload=True)
# Local RAG Document Search

## Project Overview

This project is a full-stack semantic document search that runs on local architecture to ensure total privacy and zero LLM token costs.

The project features a simple, local Blazor web application where users can add and remove documents to the search index, and perform content searches using natural language.

## Architecture & Technologies Used

* **C#/.NET Blazor/Bootstrap:** Power the local user interface.
* **Python/FastAPI/Uvicorn:** Power the local RESTful backend.
* **LangChain and ChromaDB:** Perform semantic vector searches for smart document retrieval and forces the LLM to answer strictly based on the context.
* **Ollama3.2:** Powers the LLM brain of the search.

## Installation & Setup

* **Install the requirements:** `pip install -r requirements.txt` (from admin command prompt in the _local-rag-api_ folder)
* **Install OllamaSetup.exe for local execution:** https://ollama.com/download
* **Pull 3.2 model for text comprehension:** `ollama pull llama3.2` (from admin command prompt)
* **Pull model for vector embeddings:** `ollama pull nomic-embed-text` (from admin command prompt)

### For Debugging

* **Start the FastAPI:** `python orchestrator.py` (from admin command prompt in the _local-rag-api_ folder)
* **Open Swagger:** Swagger documentation and API testers are enabled by default. You can disable them by modifying the code in _orchestrator.py_:

    FROM:
    ```python
    app = FastAPI(
        docs_url="/docs",
        redoc_url="/redoc",  
        openapi_url="/openapi.json"
    )
    ```

    TO:
    ```python
    app = FastAPI(
        docs_url=None,
        redoc_url=None,   
        openapi_url=None
    )
    ```
* **Start the UI:**  `dotnet restore` and then `dotnet watch` (from admin command prompt in the LocalRagUI folder)

### For Deployment

* **Run all services invisibly:** Double-click on `START_SERVER.cmd` to run it.
* **Stop all services:** Double-click on `STOP_SERVER.cmd` to run it.

## Components of the API

* **orchestrator.py:** The FastAPI web server managing routes.
* **rag_service.py:** The hybrid search engine and document splitter.
* **agent_service.py:** The LangChain pipeline.
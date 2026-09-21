"""
RAG Service Module
This module handles the core Retrieval-Augmented Generation (RAG) operations.
It manages document ingestion (parsing, chunking, and embedding), database 
management (inserting and deleting vectors), and similarity search retrieval 
using ChromaDB and local Ollama models.
"""

import hashlib
import os
import logging
from dotenv import load_dotenv

# LangChain components for document processing and vector storage
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Initialize the module-level logger
logger = logging.getLogger("rag_service")

# Load environment variables from the .env file
load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION & PATH SETUP
# ---------------------------------------------------------
# Define the absolute path for local database storage to ensure it resolves correctly
# regardless of where the script is executed from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
PERSIST_DIRECTORY = os.path.abspath(os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PATH))

# Model configurations (fallback to defaults if not found in .env)
# Using nomic-embed-text for fast, local embedding generation
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "global_knowledge_base")

# ---------------------------------------------------------
# DATABASE UTILITIES
# ---------------------------------------------------------

def get_document_hash(file_path):
    """
    Generates an MD5 hash of the file content for deduplication.
    Reads the file in chunks to prevent memory overload with large PDFs.
    
    Args:
        file_path (str): The absolute path to the file on disk.
        
    Returns:
        str: The hexadecimal MD5 hash of the file.
    """
    logger.debug(f"Generating MD5 hash for {file_path}")
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        # Read in 8KB chunks for memory efficiency
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_vector_store():
    """
    Initializes and returns the Chroma vector database connection.
    Binds the local Ollama embedding model so Chroma knows how to vectorize queries.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

# ---------------------------------------------------------
# CORE RAG PIPELINE FUNCTIONS
# ---------------------------------------------------------

def ingest_document(file_path: str):
    """
    Ingests a physical document into the vector knowledge base.
    Prevents duplicate processing by checking the file's MD5 hash first.
    
    Args:
        file_path (str): Path to the document (.txt or .pdf).
        
    Returns:
        int: The number of vector chunks created and added to the database.
    """
    logger.info(f"Starting ingestion process for: {os.path.basename(file_path)}")
    
    # 1. Deduplication Check
    file_hash = get_document_hash(file_path)
    vector_store = get_vector_store()
    
    # Query ChromaDB to see if this exact file hash already exists
    existing_docs = vector_store.get(where={"file_hash": file_hash})
    if existing_docs['ids']:
        logger.info(f"Document {os.path.basename(file_path)} already indexed. Skipping.")
        return 0 
    
    # 2. File Loading
    logger.info(f"Loading document: {os.path.basename(file_path)}")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        # Fallback for plain text files
        loader = TextLoader(file_path)
    
    docs = loader.load()
    
    # 3. Semantic Chunking
    # Unlike character splitters that cut off mid-sentence, SemanticChunker uses 
    # the embedding model to group sentences that are contextually related.
    logger.info(f"Splitting document using SemanticChunker with model '{EMBEDDING_MODEL}'...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(docs)
    
    # 4. Metadata Tagging
    # Inject traceability metadata into every chunk before saving
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["source"] = os.path.basename(file_path)
        
    # 5. Database Insertion
    logger.info(f"Adding {len(chunks)} chunks to the vector store...")
    vector_store.add_documents(chunks)
    logger.info(f"Successfully indexed {len(chunks)} chunks from {os.path.basename(file_path)}.")
    
    return len(chunks)


def get_raw_retrieval(query: str):
    """
    Performs a similarity search against the vector database for a given user query.
    Applies distance thresholding to prevent hallucination from irrelevant data.
    
    Args:
        query (str): The user's input question.
        
    Returns:
        dict: A payload containing the concatenated context string and a list of citation sources.
    """
    # 1. Get the active vector database instance
    vectorstore = get_vector_store() 
    
    # 2. Grab the top 1 most relevant chunk (k=1) along with its similarity score
    # Note: Increase 'k' if you want the LLM to synthesize answers from multiple chunks
    raw_results = vectorstore.similarity_search_with_score(query, k=1)
    
    valid_chunks = []
    
    # 3. Filter out poor matches using distance thresholds
    for doc, score in raw_results:
        # IMPORTANT CHROMADB QUIRK: 
        # ChromaDB defaults to "L2 Distance" (Euclidean distance) for scoring. 
        # This means a LOWER score is BETTER (0.0 is a perfect identical match).
        # A distance above 1.2 usually indicates the AI is reaching/guessing.
        if score < 1.2:  
            valid_chunks.append(doc)
            
    # 4. If no valid chunks pass the threshold, return an empty context 
    # to trigger the LLM's "Sorry, I don't know" fallback behavior.
    if not valid_chunks:
        return {
            "context": "",
            "sources": []
        }

    # 5. Build the context payload and format citations using ONLY the valid chunks
    context_text = ""
    citations = []
    
    for doc in valid_chunks:
        # Concatenate the text content for the LLM prompt
        context_text += f"{doc.page_content}\n\n"
        
        # Extract metadata for accurate front-end citation
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")
        
        # LangChain's PyPDFLoader is 0-indexed for pages. Add 1 for human readability.
        if isinstance(page, int):
            page += 1
            
        # Format the citation (e.g., "employee_handbook.pdf (Page 4)")
        citation = f"{os.path.basename(source)} (Page {page})"
        
        # Ensure we don't duplicate citations if multiple chunks come from the same page
        if citation not in citations:
            citations.append(citation)
            
    return {
        "context": context_text, 
        "sources": citations
    }
    

def delete_document_vectors(filename: str):
    """
    Deletes all vector chunks associated with a specific document from ChromaDB.
    This ensures that removed files no longer appear in search results.
    
    Args:
        filename (str): The name of the file to remove (e.g., "report.pdf").
        
    Returns:
        bool: True if chunks were found and deleted, False otherwise.
    """
    logger.info(f"Attempting to delete vectors for: {filename}")
    vector_store = get_vector_store()

    # Query ChromaDB specifically for chunks where the 'source' metadata matches the filename
    existing_docs = vector_store.get(where={"source": filename})

    if existing_docs and existing_docs.get('ids'):
        # Extract the unique Chroma DB internal IDs for the matched chunks
        chunk_ids = existing_docs['ids']
        
        # Execute the deletion
        vector_store.delete(ids=chunk_ids)
        logger.info(f"Successfully deleted {len(chunk_ids)} chunks for {filename} from ChromaDB.")
        return True
    else:
        logger.warning(f"No chunks found in ChromaDB for {filename}.")
        return False
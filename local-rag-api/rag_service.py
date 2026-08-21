import hashlib
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Initialize the logger for this service
logger = logging.getLogger("rag_service")

load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
PERSIST_DIRECTORY = os.path.abspath(os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PATH))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "global_knowledge_base")

def get_document_hash(file_path):
    """Generates an MD5 hash of the file content for deduplication."""
    logger.debug(f"Generating MD5 hash for {file_path}")
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def get_vector_store():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def ingest_document(file_path: str):
    """Ingests a document into the global knowledge base if not already present."""
    logger.info(f"Starting ingestion process for: {os.path.basename(file_path)}")
    file_hash = get_document_hash(file_path)
    vector_store = get_vector_store()
    
    existing_docs = vector_store.get(where={"file_hash": file_hash})
    if existing_docs['ids']:
        logger.info(f"Document {os.path.basename(file_path)} already indexed. Skipping.")
        return 0 
    
    logger.info(f"Loading document: {os.path.basename(file_path)}")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    
    logger.info(f"Splitting document using SemanticChunker with model '{EMBEDDING_MODEL}'...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(docs)
    
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["source"] = os.path.basename(file_path)
        
    logger.info(f"Adding {len(chunks)} chunks to the vector store...")
    vector_store.add_documents(chunks)
    logger.info(f"Successfully indexed {len(chunks)} chunks from {os.path.basename(file_path)}.")
    return len(chunks)

def get_raw_retrieval(query: str):
    # 1. Get your vector database instance
    vectorstore = get_vector_store() 
    
    # 2. Grab the top 1 result along with the similarity score
    raw_results = vectorstore.similarity_search_with_score(query, k=1)
    
    valid_chunks = []
    
    # 3. Filter out the bad matches
    for doc, score in raw_results:
        # IMPORTANT CHROMADB QUIRK: 
        # ChromaDB defaults to "L2 Distance" for scoring. 
        # This means a LOWER score is BETTER (0.0 is a perfect match).
        # A distance above 1.2 usually means the AI is hallucinating/guessing.
        if score < 1.2:  
            valid_chunks.append(doc)
            
    # 4. If no valid chunks pass the threshold, return empty context
    if not valid_chunks:
        return {
            "context": "",
            "sources": []
        }

    # 5. Build your context and citations using ONLY the valid chunks
    context_text = ""
    citations = []
    
    for doc in valid_chunks:
        context_text += f"{doc.page_content}\n\n"
        
        # Extract metadata just like you were doing before
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")
        
        # Format the citation (e.g., "filename.pdf (Page 4)")
        if isinstance(page, int):
            page += 1
            
        citation = f"{os.path.basename(source)} (Page {page})"
        if citation not in citations:
            citations.append(citation)
            
    return {
        "context": context_text, 
        "sources": citations
    }
    
def delete_document_vectors(filename: str):
    """Deletes all vector chunks associated with a specific document."""
    logger.info(f"Attempting to delete vectors for: {filename}")
    vector_store = get_vector_store()

    # Query ChromaDB for all chunks where the source matches the filename
    existing_docs = vector_store.get(where={"source": filename})

    if existing_docs and existing_docs.get('ids'):
        chunk_ids = existing_docs['ids']
        vector_store.delete(ids=chunk_ids)
        logger.info(f"Successfully deleted {len(chunk_ids)} chunks for {filename} from ChromaDB.")
        return True
    else:
        logger.warning(f"No chunks found in ChromaDB for {filename}.")
        return False
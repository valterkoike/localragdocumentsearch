import os
import logging
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rag_service import get_raw_retrieval

# Initialize the logger for this service
logger = logging.getLogger("agent_service")

# Load environment variables
load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

logger.info(f"Initializing LLM connection with model: {OLLAMA_MODEL}")

# 1. Connect directly to the local model
local_llm = OllamaLLM(model=OLLAMA_MODEL)

# 2. Define an instruction-focused, action-extraction prompt
qa_prompt = PromptTemplate.from_template(
    "You are a compliance officer. Answer the question using ONLY the provided CONTEXT.\n\n"
    "INSTRUCTIONS:\n"
    "- If you extract an instruction, include the [Source, Page] tag from the context right after the sentence.\n"
    "- Do not truncate: provide the full procedural steps.\n"
    "- If you can't find the answer in the provide context do not hallucinate and just respond with Sorry, I don't know.\n"
    "- If it's a list, use a clean numbered list.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)

# 3. Create the direct chain
rag_chain = qa_prompt | local_llm

def run_agent_pipeline(question: str):
    logger.info(f"Starting agent pipeline for question: '{question}'")
    
    # Step 1: Retrieve context
    logger.info("Step 1: Fetching context from vector store...")
    retrieval_data = get_raw_retrieval(question)
    
    # Step 2: Generate answer
    logger.info("Step 2: Invoking LLM chain with context and question... (This may take a moment)")
    answer = rag_chain.invoke({
        "context": retrieval_data["context"],
        "question": question
    })
    
    if "sorry, i don't know." in answer.lower():
      logger.info("No answer was found in the documents, resetting sources array.")
      retrieval_data["sources"] = []
        
    logger.info("LLM generation complete. Returning response to orchestrator.")    
    
    return {
        "question": question,
        "answer": answer.strip(),
        "sources": retrieval_data["sources"]
    }
import os
import tiktoken
from typing import Generator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

LLM_MODEL = "llama3-8b-8192" 
MAX_CONTEXT_TOKENS = 6000 

SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."

class LLMIntegrationError(Exception):
    pass

def count_tokens(text: str) -> int:
    """Approximate token count using tiktoken (cl100k_base)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4

def truncate_context(context_chunks: list[str], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    combined_context = ""
    current_tokens = 0
    
    for chunk in context_chunks:
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens > max_tokens:
            break
        combined_context += chunk + "\n\n"
        current_tokens += chunk_tokens
        
    return combined_context.strip()

def generate_response(user_query: str, retrieved_context: list[str]) -> Generator[str, None, None]:
    api_key = os.environ.get("RAG_CHATBOT", "")
    if not api_key:
        yield "Error: RAG_CHATBOT environment variable is not set. Please set it in your environment or Streamlit Secrets."
        return

    try:
        llm = ChatGroq(model_name=LLM_MODEL, temperature=0, api_key=api_key)
    except Exception as e:
        raise LLMIntegrationError(f"Failed to initialize Groq client: {str(e)}")

    context_str = truncate_context(retrieved_context)
    prompt = f"Context:\n{context_str}\n\nUser Question: {user_query}\nAnswer:"
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    try:
        for chunk in llm.stream(messages):
            yield chunk.content
    except Exception as e:
        raise LLMIntegrationError(f"Failed to communicate with Groq: {str(e)}")

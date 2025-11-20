import os
from groq import Groq
import tiktoken
from langchain_core.messages import BaseMessage

# Initialize Client (Ensure GROQ_API_KEY is in env)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Cache model -> context_window map
MODEL_CONTEXT = {}

def load_model_context():
    """
    Fetches available models from Groq and caches their context window limits.
    """
    global MODEL_CONTEXT
    try:
        models = client.models.list()
        # Map model ID to context window size
        MODEL_CONTEXT = {m.id: m.context_window for m in models.data}
    except Exception as e:
        print(f"⚠️ Failed to load model context: {e}")
        # Fallback defaults if API fails
        MODEL_CONTEXT = {
            "llama-3.1-8b-instant": 131072,
            "llama-3.3-70b-versatile": 131072,
        }


def count_tokens(messages, model="llama-3.1-8b-instant"):
    """
    Count tokens using tiktoken (fallback simple tokenizer if unknown).
    Handles both Dicts and LangChain Message objects.
    """
    try:
        # Groq uses Llama, which is similar to OpenAI cl100k_base
        enc = tiktoken.get_encoding("cl100k_base")
    except:
        enc = tiktoken.get_encoding("gpt2")

    total = 0
    
    for msg in messages:
        content = ""
        
        # Case 1: Standard Dictionary ({"role": "user", "content": "..."})
        if isinstance(msg, dict):
            content = msg.get("content", "")
            
        # Case 2: LangChain Object (HumanMessage(content="..."))
        elif hasattr(msg, "content"):
            content = msg.content
            
        # Case 3: String fallback
        else:
            content = str(msg)

        total += len(enc.encode(content))
        
    return total


def get_remaining_tokens(model: str, messages: list):
    """
    Returns remaining context tokens before the model hits max window.
    """
    
    # 1. Load context window data if empty
    if not MODEL_CONTEXT:
        load_model_context()

    # 2. Get Max Context (Default to 8k if unknown model)
    max_context = MODEL_CONTEXT.get(model, 8192)

    # 3. Count Used Tokens
    used = count_tokens(messages, model)
    
    # 4. Calculate Remaining
    remaining = max_context - used

    return {
        "model": model,
        "max_context": max_context,
        "used_tokens": used,
        "remaining_tokens": remaining,
    }
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

    
load_dotenv()

def get_groq_llm(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> ChatGroq:
    """
    Returns a LangChain-compatible Groq LLM instance.

    Args:
        model_name (str): LLM to use ("meta-llama/llama-4-scout-17b-16e-instruct" )

    Returns:
        ChatGroq: LangChain ChatGroq object
    """
    return ChatGroq(
        model=model_name,
    )

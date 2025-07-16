import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith.run_helpers import traceable  # ✅ Use LangSmith's auto tracer

# ✅ Load .env variables
load_dotenv()

@traceable(name="Groq LLM Init")  # ✅ Trace this function
def get_groq_llm(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> ChatGroq:
    """
    Returns a LangChain-compatible Groq LLM instance.
    Traced via LangSmith using @traceable.

    Args:
        model_name (str): The model name to use

    Returns:
        ChatGroq: Traced LLM instance
    """
    return ChatGroq(
        model=model_name,
        temperature=0.3,
        max_tokens=1024,
    )

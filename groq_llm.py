from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import LangChainTracer

# Load .env environment variables
load_dotenv()

# Set environment variables explicitly if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-v1-trace")


def get_groq_llm(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> ChatGroq:
    """
    Returns a LangChain-compatible Groq LLM instance with LangSmith tracing.

    Args:
        model_name (str): LLM to use

    Returns:
        ChatGroq: LangChain ChatGroq object
    """
    return ChatGroq(
        model=model_name,
        temperature=0.3,
        max_tokens=1024,
    )


# Optional: Add tracer if you're invoking manually
tracer = LangChainTracer()  # Use in chain.run/config={"callbacks": [tracer]}

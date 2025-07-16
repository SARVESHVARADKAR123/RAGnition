import os
from vector_store import embed_and_store, load_vectorstore
from groq_llm import get_groq_llm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langsmith.run_helpers import traceable  # ✅ Recommended
from dotenv import load_dotenv
load_dotenv()

# ✅ Custom PromptTemplate for RAG
RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

def extract_clean_response(response) -> str:
    """
    Extract plain string response from LLM/chain output, safely.
    """
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()

# ✅ Automatically traced pipeline using LangSmith
@traceable(name="RAGnition RAG Pipeline")
def run_rag_pipeline(user_query: str, source_text: str = None, doc_id: str = None) -> str:
    """
    V1 RAG pipeline using LangChain Runnables, PromptTemplate, and StrOutputParser.
    Traced to LangSmith. Always returns plain string.
    """
    llm = get_groq_llm()
    output_parser = StrOutputParser()

    # 🧠 RAG path
    if doc_id:
        vectorstore = load_vectorstore(doc_id)

        if not vectorstore:
            if not source_text:
                print("⚠️ No source text. Falling back to direct LLM.")
                raw = llm.invoke(user_query)
                return extract_clean_response(raw)

            try:
                print(f"📦 Embedding new vectorstore for doc_id: {doc_id}")
                vectorstore = embed_and_store(text=source_text, doc_id=doc_id)
            except Exception as e:
                print(f"❌ Failed to embed vectorstore: {e}")
                raw = llm.invoke(user_query)
                return extract_clean_response(raw)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        rag_chain = (
            {
                "context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough()
            }
            | RAG_PROMPT
            | llm
            | output_parser
        )

        raw = rag_chain.invoke(user_query)
        return extract_clean_response(raw)

    # 💬 Fallback: Direct LLM
    print("ℹ️ No doc ID provided. Using pure LLM chat.")
    raw = llm.invoke(user_query)
    return extract_clean_response(raw)

from vector_store import embed_and_store, load_vectorstore
from groq_llm import get_groq_llm
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain.callbacks.tracers.langchain import LangChainTracer


# ✅ LangSmith Tracing
tracer = LangChainTracer()

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


def run_rag_pipeline(user_query: str, source_text: str = None, doc_id: str = None) -> str:
    """
    Upgraded V1 RAG pipeline using LangChain Runnables, PromptTemplate, and StrOutputParser.

    Args:
        user_query (str): User's question
        source_text (str, optional): Full source text
        doc_id (str, optional): Document ID to fetch vectorstore

    Returns:
        str: LLM response
    """

    llm = get_groq_llm()
    output_parser = StrOutputParser()

    # ✅ If doc_id is provided, try vector-based retrieval
    if doc_id:
        vectorstore = load_vectorstore(doc_id)

        # If vectorstore doesn't exist, embed and store
        if not vectorstore:
            if not source_text:
                print("⚠️ No source text. Falling back to direct LLM.")
                return llm.invoke(user_query, config={"callbacks": [tracer]})

            try:
                print(f"📦 Embedding new vectorstore for doc_id: {doc_id}")
                vectorstore = embed_and_store(text=source_text, doc_id=doc_id)
            except Exception as e:
                print(f"❌ Failed to embed vectorstore: {e}")
                return llm.invoke(user_query, config={"callbacks": [tracer]})

        if not vectorstore:
            print("❌ Still no vectorstore. Using direct LLM.")
            return llm.invoke(user_query, config={"callbacks": [tracer]})

        # ✅ Build RAG chain using Runnable components
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

        return rag_chain.invoke(user_query, config={"callbacks": [tracer]})

    # ✅ Fallback: Direct LLM chat
    print("ℹ️ No doc ID provided. Using pure LLM chat.")
    return llm.invoke(user_query, config={"callbacks": [tracer]})

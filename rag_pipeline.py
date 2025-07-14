from vector_store import embed_and_store, load_vectorstore
from groq_llm import get_groq_llm
from langchain.chains import RetrievalQA


def run_rag_pipeline(user_query: str, source_text: str = None, doc_id: str = None) -> str:
    """
    Full RAG pipeline:
    - Uses vectorstore if doc_id is provided.
    - Falls back to direct LLM chat if no doc_id or if vectorstore not found and no text.

    Args:
        user_query (str): User's question
        source_text (str, optional): Full source text (needed if vectorstore must be created)
        doc_id (str, optional): Unique document ID

    Returns:
        str: LLM-generated answer (RAG-based or direct)
    """
    llm = get_groq_llm()

    if doc_id:
        # Try to load vectorstore
        vectorstore = load_vectorstore(doc_id)

        if not vectorstore:
            if not source_text:
                print(f"⚠️ No vectorstore or source text. Falling back to LLM for query: {user_query}")
                return llm.invoke(user_query)

            try:
                print(f"📦 Embedding new vectorstore for doc_id: {doc_id}")
                vectorstore = embed_and_store(text=source_text, doc_id=doc_id)
            except Exception as e:
                print(f"❌ Failed to embed: {e}. Falling back to direct LLM.")
                return llm.invoke(user_query)

        if not vectorstore:
            print("❌ Vectorstore is still None. Fallback to LLM.")
            return llm.invoke(user_query)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )
        return rag_chain.run(user_query)

    # No doc_id – pure chat
    print("ℹ️ No doc selected. Using direct LLM.")
    return llm.invoke(user_query)

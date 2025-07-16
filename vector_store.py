import os
import json
import hashlib
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langsmith.run_helpers import traceable  # ✅ Add this
from dotenv import load_dotenv

load_dotenv()

# === Constants ===
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
META_FILE = os.path.join("meta_db", "index.json")

# === Get unique doc ID for caching ===
def get_doc_id(text: str) -> str:
    """
    Generate a unique document ID from input text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

# ✅ Traced function for LangSmith (shows under "Document Embedding")
@traceable(name="Embed and Store")
def embed_and_store(text: str, doc_id: str, filename: Optional[str] = None) -> Chroma:
    """
    Create or reuse a Chroma vectorstore for the given text.
    """
    persist_path = os.path.join(CHROMA_DIR, doc_id)

    if os.path.exists(persist_path) and os.listdir(persist_path):
        print(f"📁 Reusing vectorstore at {persist_path}")
        return Chroma(
            persist_directory=persist_path,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        )

    if not text:
        raise ValueError("❌ Text is required to create a new embedding.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    if filename:
        for doc in docs:
            doc.metadata["source"] = filename

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_path
    )

    vectorstore.persist()
    print(f"✅ New vectorstore saved to {persist_path}")

    if filename:
        save_doc_metadata(doc_id, filename)

    return vectorstore

# === Load Cached Vectorstore Only (Skip Embed)
def load_vectorstore(doc_id: str) -> Optional[Chroma]:
    """
    Load existing Chroma vectorstore for given doc_id.
    """
    persist_path = os.path.join(CHROMA_DIR, doc_id)
    if not os.path.exists(persist_path) or not os.listdir(persist_path):
        return None
    return Chroma(
        persist_directory=persist_path,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

# === Save filename metadata ===
def save_doc_metadata(doc_id: str, filename: str):
    """
    Save the mapping of doc_id to filename in the metadata index.
    """
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    data = {}
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            data = json.load(f)
    data[doc_id] = filename
    with open(META_FILE, "w") as f:
        json.dump(data, f, indent=2)

# === Load metadata for all docs (UI dropdown, etc.) ===
def load_all_doc_metadata() -> dict:
    """
    Load all doc_id → filename metadata mappings.
    """
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}

# === Get most recent doc_id (fallback for startup or default doc) ===
def get_latest_doc_id() -> Optional[str]:
    """
    Return the most recently modified doc_id folder.
    """
    if not os.path.exists(CHROMA_DIR):
        return None

    doc_dirs = sorted(
        [
            d for d in os.listdir(CHROMA_DIR)
            if os.path.isdir(os.path.join(CHROMA_DIR, d)) and os.listdir(os.path.join(CHROMA_DIR, d))
        ],
        reverse=True
    )
    return doc_dirs[0] if doc_dirs else None

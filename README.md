# 🧠 RAGnition — Multimodal RAG System using LangChain + Groq

**RAGnition** is an end-to-end **multimodal RAG (Retrieval-Augmented Generation)** system that allows users to chat with AI using their uploaded files (PDFs, images), online content (YouTube videos, URLs), or search results. Built with LangChain, ChromaDB, Groq’s LLaMA/Mixtral LLMs, and a clean Streamlit interface.

---

## 🚀 Features

- 🧾 **Input Sources**:
  - 📄 PDFs
  - 🖼️ Images (OCR)
  - 📺 YouTube videos (transcripts)
  - 🌐 Web URLs
  - 🔍 DuckDuckGo Search results

- ⚡ **Performance**:
  - Fast inference via **Groq API** (LLaMA 3 / Mixtral)
  - Embedded caching via **ChromaDB**
  - Reuse of **past RAG sessions** without re-uploading
  - Session memory and vectorstore reuse
  - Fallback to default LLM chat (if no document used)

- 🛠️ Modular, clean code using:
  - `LangChain` + `Chroma`
  - `HuggingFace` Embeddings
  - `Streamlit` UI
  - `Python` standard tooling

---

## 🧠 Intelligent Document Handling

> You can chat even **without uploading anything**.

✅ The app supports:
- **Reusing previously embedded documents** via the sidebar dropdown  
- **Default LLM fallback** if no document is selected at all  
- No need to re-upload — RAGnition remembers your documents (`ChromaDB` + metadata caching)

---

## 📁 Directory Structure

```
RAGnition/
├── main.py                   # Streamlit UI
├── rag_pipeline.py           # RAG logic (LLM + retriever)
├── vector_store.py           # Vectorstore caching, metadata
├── groq_llm.py               # Groq LLM loader
│
├── loaders/                  # Input processors
│   ├── pdf_loader.py
│   ├── image_ocr.py
│   ├── youtube_loader.py
│   ├── web_scraper.py
│
├── chroma_db/                # Persisted vectorstores
├── meta_db/                  # Saved doc metadata
│
├── .env                      # (Your Groq API Key)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧪 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/SARVESHVARADKAR123/RAGnition.git
cd RAGnition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Activate
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 API Key Setup

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get your key from [https://console.groq.com](https://console.groq.com)

---

## ▶️ Running the App

```bash
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ How It Works

1. **Upload** or **select** a file / source (or skip to chat freely)
2. Text is extracted using appropriate loaders (PDF, OCR, transcript, scraper)
3. Embeddings are generated using `HuggingFace` models
4. Text chunks are stored in a **ChromaDB** vectorstore
5. On query, top chunks are retrieved and passed to a **Groq-hosted LLM**
6. Results are displayed with caching and session memory

---

## 🧰 Tools Used

| Tool         | Purpose                           |
|--------------|-----------------------------------|
| LangChain    | RAG chain, retrievers             |
| ChromaDB     | Embedding + vectorstore           |
| Groq         | LLM inference (LLaMA/Mixtral)     |
| HuggingFace  | Embedding model                   |
| Streamlit    | UI / front-end                    |
| OCR / Scraper| Image & Web processing            |

---

## 🤝 Contribute

PRs and feedback are welcome! Fork this repo, star it ⭐, and build something epic with it.

🧠 RAGnition V1 — Multimodal RAG System with LangChain + Groq
RAGnition V1 is a production-ready multimodal Retrieval-Augmented Generation (RAG) system that allows users to chat with an AI using uploaded documents (PDFs, images), online content (YouTube videos, URLs), or real-time web search.

Powered by:

💨 Groq's blazing-fast LLaMA/Mixtral models

🧠 LangChain and ChromaDB for document understanding

💻 Streamlit for a clean chat UI

🔍 LangSmith for full tracing and debugging

🚀 Features (V1 Highlights)
🧾 Multimodal Inputs:

📄 PDF documents

🖼️ Images (via OCR)

📺 YouTube video transcripts

🌐 Webpage scraping

🔍 DuckDuckGo Search results

🔁 Smart Document Memory:

Cached via ChromaDB vectorstore

Auto-reuse of previously uploaded content

Fallback to LLM-only if no doc selected

⚡ Powered by Groq:

Uses meta-llama/llama-4-scout-17b-16e-instruct

100x token throughput vs standard LLMs

🧠 LangSmith Tracing (Optional):

Traced inputs, loaders, LLM outputs

Easy debugging and pipeline visibility

📁 Directory Structure
bash
Copy
Edit
RAGnition/
├── main.py                   # Streamlit UI
├── rag_pipeline.py           # RAG chain logic
├── groq_llm.py               # Groq LLM init (with tracing)
├── vector_store.py           # ChromaDB storage/reuse
│
├── loaders/                  # Modular input extractors
│   ├── pdf_loader.py
│   ├── image_ocr.py
│   ├── youtube_loader.py
│   ├── web_scraper.py
│
├── utils/                    # Extras (session, persistence)
│   └── persistence.py
│
├── chroma_db/                # Local vectorstores
├── meta_db/                  # Document metadata
│
├── .env                      # Keys (Groq, LangSmith)
├── .gitignore
├── requirements.txt
└── README.md
🧪 Installation
1. Clone and Set Up Environment
bash
Copy
Edit
git clone https://github.com/SARVESHVARADKAR123/RAGnition.git
cd RAGnition
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔐 Environment Configuration
Create a .env file:

env
Copy
Edit
GROQ_API_KEY=your_groq_key_here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=RAGnition
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
▶️ Run the App
bash
Copy
Edit
streamlit run main.py
Then visit: http://localhost:8501

✨ How It Works
Choose an input (PDF, Image, URL, etc.)

Text is extracted and embedded using HuggingFace + ChromaDB

A Groq LLM retrieves top-relevant chunks and answers your query

Fallback to direct LLM response if no doc is used

📊 Tracing with LangSmith (Built-in)
Every pipeline step is traced via @traceable:

Document loading

Embedding and vectorstore

Retrieval and LLM output

Displayed in LangSmith

✅ No manual config required beyond .env

🧰 Tech Stack
Tool	Purpose
LangChain	RAG logic, retrievers
ChromaDB	Vectorstore + chunk storage
HuggingFace	Embedding model
Groq	LLM inference (LLaMA/Mixtral)
Streamlit	Frontend UI
LangSmith	End-to-end observability

📌 Version Info
yaml
Copy
Edit
🧠 RAGnition Version: v1.0.0
🔁 Multimodal: Yes
⚡ Backend: Groq (LLM4)
🛠️ Framework: LangChain
🎯 Tracing: LangSmith enabled
🤝 Contributing
Have ideas, suggestions, or bugs?
Fork this repo, submit a PR, or reach out!
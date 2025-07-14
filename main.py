import streamlit as st
from loaders.pdf_loader import extract_text_from_pdf
from loaders.image_ocr import extract_text_from_image
from loaders.youtube_loader import get_transcript_from_youtube
from loaders.web_scraper import scrape_text_from_urls, get_duckduckgo_search_context
from rag_pipeline import run_rag_pipeline
from vector_store import get_doc_id, load_all_doc_metadata

# --- PAGE SETUP ---
st.set_page_config(page_title="MultiRAG MVP", layout="centered")
st.title("📚🧠 RAGnition 🧠📚")
st.markdown("Upload content or select a previous doc — or just chat freely with Groq.")

# --- SESSION STATE ---
if "source_text" not in st.session_state:
    st.session_state.source_text = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# --- SIDEBAR ---
st.sidebar.markdown("## ⚙️ Session")
if st.sidebar.button("🧹 Clear Session"):
    st.session_state.clear()
    st.experimental_rerun()

# --- SAVED DOC SELECT ---
saved_docs = load_all_doc_metadata()  # {doc_id: filename}
doc_options = list(saved_docs.items())

if doc_options:
    default_index = 0
    if st.session_state.doc_id:
        for i, (doc_id, _) in enumerate(doc_options):
            if doc_id == st.session_state.doc_id:
                default_index = i
                break

    selected_doc = st.selectbox(
        "📂 Or Select Existing Document",
        doc_options,
        index=default_index,
        format_func=lambda x: f"{x[1]} ({x[0]})"
    )
    if selected_doc:
        st.session_state.doc_id = selected_doc[0]
        st.session_state.source_text = None
        st.session_state.answer = None

# --- INPUT MODE ---
st.markdown("---")
input_mode = st.selectbox(
    "Choose Input Mode",
    ["PDF", "Image (OCR)", "YouTube Link", "Web URL", "DuckDuckGo Search"]
)

uploaded_file = None
input_text = None

if input_mode in ["PDF", "Image (OCR)"]:
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "png", "jpg", "jpeg"])
else:
    input_text = st.text_input("Enter URL or Search Query")

# --- PROCESS INPUT ---
if st.button("📥 Process Input"):
    text = None

    if input_mode == "PDF" and uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
    elif input_mode == "Image (OCR)" and uploaded_file:
        with st.spinner("Running OCR on image..."):
            text = extract_text_from_image(uploaded_file)
    elif input_mode == "YouTube Link" and input_text:
        with st.spinner("Fetching transcript from YouTube..."):
            text = get_transcript_from_youtube(input_text)
    elif input_mode == "Web URL" and input_text:
        with st.spinner("Scraping webpage..."):
            text = scrape_text_from_urls(input_text)
    elif input_mode == "DuckDuckGo Search" and input_text:
        with st.spinner("Searching DuckDuckGo..."):
            text = get_duckduckgo_search_context(input_text)
    else:
        st.warning("⚠️ Please provide valid input.")

    if text:
        doc_id = get_doc_id(text)
        st.session_state.source_text = text
        st.session_state.doc_id = doc_id
        st.session_state.answer = None
        st.success(f"✅ Text processed and cached! ID: `{doc_id}`")

# --- VIEW TEXT IF AVAILABLE ---
if st.session_state.source_text:
    with st.expander("🔍 View Extracted Source Text"):
        st.markdown(f"**🆔 Document ID**: `{st.session_state.doc_id}`")
        st.text_area("Source Text", st.session_state.source_text, height=200)

# --- CHAT MODE: Always Active ---
st.markdown("---")
st.session_state.user_query = st.text_input(
    "💬 Ask a question (RAG if doc selected, LLM if not)",
    value=st.session_state.user_query
)

if st.session_state.user_query:
    with st.spinner("Thinking with Groq..."):
        try:
            answer = run_rag_pipeline(
                user_query=st.session_state.user_query,
                source_text=st.session_state.source_text,
                doc_id=st.session_state.doc_id
            )
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
        st.session_state.answer = answer
        st.markdown("### 🧠 Answer")
        st.success(answer)

# --- SHOW LAST ANSWER IF RELOADED ---
elif st.session_state.answer:
    st.markdown("### 🧠 Last Answer")
    st.success(st.session_state.answer)

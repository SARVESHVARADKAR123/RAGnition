import pdfplumber
from langsmith.run_helpers import traceable  # ✅ Enable tracing

@traceable(name="Extract PDF Text")
def extract_text_from_pdf(file_obj):
    """
    Extracts text from a PDF file object using pdfplumber.
    Traced in LangSmith as 'Extract PDF Text'.
    """
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

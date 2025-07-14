import pdfplumber

def extract_text_from_pdf(file_obj):
    """
    Extracts text from a PDF file object.
    """
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

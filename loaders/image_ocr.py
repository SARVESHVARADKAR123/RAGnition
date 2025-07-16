import pytesseract
from PIL import Image
from langsmith.run_helpers import traceable  # ✅ Add LangSmith tracing

@traceable(name="OCR Image Loader")
def extract_text_from_image(image_file):
    """
    Extracts text from an uploaded image file (PNG, JPG, JPEG).
    Traced in LangSmith as 'OCR Image Loader'.
    """
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

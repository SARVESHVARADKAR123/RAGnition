import pytesseract
from PIL import Image
import io

def extract_text_from_image(image_file):
    """
    Extracts text from an uploaded image file.
    Supports formats like PNG, JPG, JPEG.
    """
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

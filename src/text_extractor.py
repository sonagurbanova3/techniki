import os

import docx
import pdfplumber
import pytesseract
from PIL import Image


def extract_text_from_pdf(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def extract_text_from_docx(file_path):
    document = docx.Document(file_path)
    paragraphs = []

    for paragraph in document.paragraphs:
        paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def extract_text_from_image(file_path):
    """
    OCR dla obrazów. Wymaga zainstalowanego programu Tesseract OCR w systemie.
    PDF/DOCX/TXT działają bez tej funkcji.
    """
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Nie znaleziono Tesseract OCR. Zainstaluj Tesseract albo użyj pliku PDF/DOCX/TXT."
        ) from exc


def extract_text(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    if extension == ".docx":
        return extract_text_from_docx(file_path)

    if extension == ".txt":
        return extract_text_from_txt(file_path)

    if extension in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)

    raise ValueError(f"Nieobsługiwany format pliku: {extension}")

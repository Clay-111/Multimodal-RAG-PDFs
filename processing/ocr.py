# =========================
# File: processing/ocr.py
# =========================

import io
import re
import fitz
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes


def extract_and_clean_text(file_bytes):
    # TesseractOCR path
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe" # If installed elsewhere, update the path

    # OCR each page rendered as an image
    pages_as_images = convert_from_bytes(file_bytes, dpi=300)
    full_text = ""
    for page_img in pages_as_images:
        ocr_result = pytesseract.image_to_string(page_img, lang="ben+eng")
        full_text += ocr_result + "\n"

    # Extract embedded images using PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    extracted_images = []
    for page_index, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                extracted_images.append({
                    "page": page_index,
                    "figure_number": None,
                    "image": pil_img,
                })
            except Exception as e:
                st.warning(f"Image extraction failed on page {page_index}: {e}")

    # Match figures in text (Figure / Fig.)
    figure_pattern = re.compile(r"(?:Figure|Fig\.?)\s*\d+", re.IGNORECASE)
    figure_matches = figure_pattern.findall(full_text)
    fig_counter = 0
    for img_data in extracted_images:
        if fig_counter < len(figure_matches):
            img_data["figure_number"] = figure_matches[fig_counter]
            fig_counter += 1

    # Clean basic artifacts
    text = re.sub(r"\n\s*\d+\s*\n", "\n", full_text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip(), extracted_images
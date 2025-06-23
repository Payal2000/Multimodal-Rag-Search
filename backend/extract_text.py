# backend/extract_text.py

import os
import json
from PyPDF2 import PdfReader

PDF_PATH = "data/input.pdf"
OUTPUT_PATH = "data/text_per_page.json"

def extract_text_per_page(pdf_path):
    reader = PdfReader(pdf_path)
    text_data = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:  # Only store if text is present
            text_data.append({"page": i + 1, "text": text.strip()})
    return text_data

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at {PDF_PATH}")
    else:
        print(f"📄 Extracting text from: {PDF_PATH}")
        results = extract_text_per_page(PDF_PATH)

        os.makedirs("data", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved text chunks to {OUTPUT_PATH}")

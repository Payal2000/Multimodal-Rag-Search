# backend/extract_images.py
import fitz  # PyMuPDF
import os

def save_pdf_pages_as_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        image_path = os.path.join(output_folder, f"slide_{i + 1}.png")
        pix.save(image_path)

    print(f"✅ Saved {len(doc)} slide images to {output_folder}")

if __name__ == "__main__":
    pdf_path = "data/input.pdf"  # <-- your renamed file
    output_folder = "data/slide_images"

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
    else:
        save_pdf_pages_as_images(pdf_path, output_folder)

import os
import json
import faiss
import numpy as np

TEXT_INDEX_PATH = "data/faiss_text.index"
IMAGE_INDEX_PATH = "data/faiss_image.index"
TEXT_META_PATH = "data/text_metadata.json"
IMAGE_META_PATH = "data/image_metadata.json"

def load_embeddings(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def build_faiss_index(embeddings):
    dim = len(embeddings[0]["embedding"])
    index = faiss.IndexFlatL2(dim)
    vectors = np.array([item["embedding"] for item in embeddings]).astype('float32')
    index.add(vectors)
    return index, embeddings

def save_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def upsert_embeddings():
    print("📦 Building FAISS index for text...")
    text_data = load_embeddings("data/text_embeddings.json")
    text_index, text_metadata = build_faiss_index(text_data)
    faiss.write_index(text_index, TEXT_INDEX_PATH)
    save_metadata(text_metadata, TEXT_META_PATH)
    print(f"✅ Saved text index to {TEXT_INDEX_PATH}")

    print("🖼 Building FAISS index for images...")
    image_data = load_embeddings("data/image_embeddings.json")
    image_index, image_metadata = build_faiss_index(image_data)
    faiss.write_index(image_index, IMAGE_INDEX_PATH)
    save_metadata(image_metadata, IMAGE_META_PATH)
    print(f"✅ Saved image index to {IMAGE_INDEX_PATH}")

if __name__ == "__main__":
    upsert_embeddings()

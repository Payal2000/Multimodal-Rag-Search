import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_query(query: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    embedding = response.data[0].embedding
    return np.array([embedding], dtype="float32")

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_metadata(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

def search_faiss(index, query_vector, metadata, top_k=3):
    print(f"🧠 Query vector shape: {query_vector.shape}, FAISS index dim: {index.d}")
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "score": float(distances[0][i]),
                "result": metadata[idx]
            })
    return results

def run_query(user_query):
    query_vector = embed_query(user_query)

    text_index = load_faiss_index("data/faiss_text.index")
    text_meta = load_metadata("data/text_metadata.json")
    text_results = search_faiss(text_index, query_vector, text_meta)

    image_index = load_faiss_index("data/faiss_image.index")
    image_meta = load_metadata("data/image_metadata.json")
    image_results = search_faiss(image_index, query_vector, image_meta)

    print("\n📄 Top Text Matches:")
    for res in text_results:
        print(f"Page {res['result']['page']}: {res['result']['text'][:150]}... (Score: {res['score']:.2f})")

    print("\n🖼 Top Image Matches:")
    for res in image_results:
        print(f"Slide: {res['result']['slide']} | Caption: {res['result']['caption']} (Score: {res['score']:.2f})")

if __name__ == "__main__":
    user_query = input("🔍 Enter your query: ")
    run_query(user_query)

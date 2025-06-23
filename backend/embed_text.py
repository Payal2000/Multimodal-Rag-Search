# backend/embed_text.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["text"]
        )
        embeddings.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding": response.data[0].embedding
        })
    return embeddings

if __name__ == "__main__":
    input_path = "data/text_per_page.json"
    output_path = "data/text_embeddings.json"

    if not os.path.exists(input_path):
        print("❌ Text chunks not found. Run extract_text.py first.")
    else:
        with open(input_path, "r") as f:
            chunks = json.load(f)
        results = embed_texts(chunks)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved text embeddings to {output_path}")

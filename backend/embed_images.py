# backend/embed_images.py
import os
import json
import base64
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def generate_caption(image_path):
    base64_image = encode_image_to_base64(image_path)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this slide image briefly for search indexing."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

def embed_caption(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def process_images(folder_path):
    results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            slide_number = int(filename.split("_")[1].split(".")[0])

            print(f"📸 Processing slide {slide_number}...")

            caption = generate_caption(image_path)
            embedding = embed_caption(caption)

            results.append({
                "slide": slide_number,
                "caption": caption,
                "embedding": embedding
            })

    return results

if __name__ == "__main__":
    folder_path = "data/slide_images"
    output_path = "data/image_embeddings.json"

    if not os.path.exists(folder_path):
        print("❌ No images found. Run extract_images.py first.")
    else:
        results = process_images(folder_path)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved image (caption) embeddings to {output_path}")

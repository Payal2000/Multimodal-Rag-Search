import streamlit as st
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()
client = OpenAI()

TEXT_INDEX_PATH = "data/faiss_text.index"
TEXT_META_PATH = "data/text_metadata.json"
IMAGE_INDEX_PATH = "data/faiss_image.index"
IMAGE_META_PATH = "data/image_metadata.json"


def embed_query(query):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return np.array([response.data[0].embedding], dtype="float32")


def load_faiss_index(path):
    return faiss.read_index(path)


def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def search(index, query_vector, metadata, top_k=3):
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "score": float(distances[0][i]),
                "result": metadata[idx]
            })
    return results


def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return None


def main():
    st.set_page_config(page_title="Multimodal RAG", layout="wide")
    st.title("📄🔍 Multimodal RAG Search Engine")
    st.markdown("Ask questions across PDF slides using both text and image content.")

    user_query = st.text_input("🔍 Enter your query")

    if user_query:
        with st.spinner("🔎 Embedding your query..."):
            query_vector = embed_query(user_query)

        text_index = load_faiss_index(TEXT_INDEX_PATH)
        text_meta = load_metadata(TEXT_META_PATH)

        image_index = load_faiss_index(IMAGE_INDEX_PATH)
        image_meta = load_metadata(IMAGE_META_PATH)

        with st.spinner("🔍 Searching text and image indexes..."):
            text_results = search(text_index, query_vector, text_meta)
            image_results = search(image_index, query_vector, image_meta)

        st.subheader("📄 Top Text Matches")
        for res in text_results:
            page = res['result']['page']
            score = res['score']
            text = res['result']['text']
            st.markdown(f"**📄 Page {page}** — Score: `{score:.2f}`")
            st.write(text)
            st.markdown("---")

        st.subheader("🖼 Top Image Matches")
        for res in image_results:
            slide = res['result']['slide']
            caption = res['result']['caption']
            score = res['score']
            image_path = f"data/slide_images/slide_{slide}.png"
            st.markdown(f"**🖼 Slide {slide}** — Score: `{score:.2f}`")

            base64_img = get_base64_image(image_path)
            if base64_img:
                st.markdown(
                    f'<img src="data:image/png;base64,{base64_img}" width="280" style="border-radius:8px;"/>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"⚠️ Image file for slide {slide} not found.")
            st.caption(caption)
            st.markdown("---")


if __name__ == "__main__":
    main()

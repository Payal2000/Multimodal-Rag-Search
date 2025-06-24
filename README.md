# 📄🔍 Multimodal RAG Search

Multimodal RAG Search is a lightweight Retrieval-Augmented Generation tool that allows users to ask natural language questions over a slide-based PDF (less than 5 pages). It extracts both **text and image context**, making your queries multimodal.

## 🧠 Features

- Text + Image search across slides  
- Captions slides using OpenAI Vision  
- Embeds text and image content using OpenAI  
- Searches with FAISS  
- Simple Streamlit UI

## 🗂️ Project Structure

<pre>
multimodal-rag-search/  
├── backend/  
│   ├── extract_text.py  
│   ├── extract_images.py  
│   ├── embed_text.py  
│   ├── embed_images.py  
│   ├── vector_db.py  
│   └── query_engine.py  
├── streamlit_app.py  
├── data/  
├── .gitignore  
├── requirements.txt  
└── README.md  
</pre>

## 🚀 How to Run

1. Clone this repo  
2. Place your PDF file as `data/input.pdf` (max 5 pages)  
3. Create a `.env` file with your OpenAI key:  
   ```
   OPENAI_API_KEY=your-key-here
   ```
4. Run the following scripts step-by-step:

```
python3 backend/extract_text.py
python3 backend/extract_images.py
python3 backend/embed_text.py
python3 backend/embed_images.py
python3 backend/vector_db.py
streamlit run streamlit_app.py
```

## 💬 Example Query

**Query:**  
Where does the slide mention real-life AI applications?

**Top Match:**  
- **Text:** Page 5 – Self-driving cars, Boston Dynamics, Chatbots...  
- **Image:** Slide 5 – Shows robots, navigation, and chat UI screenshots

## 🧰 Tech Stack

- OpenAI Embeddings & GPT-4-Vision  
- FAISS for vector similarity  
- Streamlit for UI  
- PyMuPDF (fitz) + Pillow + OpenCV  
- Python 3.10+

## 📄 License

MIT License

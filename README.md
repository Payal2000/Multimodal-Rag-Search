# Multimodal RAG Search

A lightweight, GPT-powered search tool that lets you ask natural language questions across both text and image content in a PDF slide deck. It extracts page-wise text and slide images, generates GPT-4o captions for each slide, embeds them using OpenAI’s embedding API, and retrieves relevant results using FAISS — all shown through a clean Streamlit interface.

## Features

- Text + Image search across slides  
- Captions slides using OpenAI gpt 4o 
- Embeds text and image content using OpenAI  
- Searches with FAISS  
- Simple Streamlit UI

## Project Structure

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

## How It Works

This project assumes a presentation PDF (e.g., `input.pdf`) is already saved locally in the `data/` folder.

Here's what happens next:

1. **Text Extraction**  
   Extracts text from each page using `PyPDF2` and stores it with metadata (page number).

2. **Image Extraction**  
   Converts each slide into an image using `PyMuPDF` (`fitz`) for later visual processing.

3. **Image Captioning**  
   Each slide image is interpreted using **OpenAI GPT-4o** to generate a meaningful visual caption.

4. **Embedding Generation**  
   - Text chunks and image captions are embedded using `text-embedding-3-small`.
   - Each modality is saved to a `.json` file with metadata.

5. **FAISS Index Creation**  
   Separate vector indexes are built for:
   - Slide text
   - Slide image captions

6. **Semantic Search via Streamlit**  
   You type a question like:
   > “Where does it talk about real-world AI examples?”

   The app searches both indexes and shows:
   - Top-matching **text chunks** with page numbers and scores
   - Top-matching **slide images** with GPT-generated captions

Run the following scripts step-by-step:

```
python3 backend/extract_text.py
python3 backend/extract_images.py
python3 backend/embed_text.py
python3 backend/embed_images.py
python3 backend/vector_db.py
streamlit run streamlit_app.py
```

## Example Query

**Query:**  
Where does the slide mention real-life AI applications?

**Top Match:**  
- **Text:** Page 5 – Self-driving cars, Boston Dynamics, Chatbots...  
- **Image:** Slide 5 – Shows robots, navigation, and chat UI screenshots

## Tech Stack

- **OpenAI GPT-4o** – For generating captions from slide images
- **OpenAI Embeddings API** – For generating dense vector embeddings
- **FAISS** – For similarity search across text and image captions
- **PyPDF2 / fitz** – For extracting content from the PDF
- **Streamlit** – For interactive UI and display


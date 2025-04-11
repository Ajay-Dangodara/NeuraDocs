import streamlit as st
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract
import cohere
import tempfile
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Initialize Cohere
co = cohere.Client("YOUR-API-KEY") 

# Use a more accurate embedding model
embedder = SentenceTransformer('all-mpnet-base-v2')

stored_docs = {}

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                text += ocr_text
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

def get_chunks(text, max_tokens=300):
    sentences = text.split('.')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len((chunk + sentence).split()) <= max_tokens:
            chunk += sentence.strip() + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence.strip() + '. '
    if chunk:
        chunks.append(chunk.strip())
    # Deduplicate chunks
    seen = set()
    unique_chunks = []
    for c in chunks:
        c_clean = c.strip()
        if c_clean not in seen:
            unique_chunks.append(c_clean)
            seen.add(c_clean)
    return unique_chunks

def get_embeddings(chunks):
    cleaned_chunks = [c.strip() for c in chunks if c.strip()]
    return embedder.encode(cleaned_chunks, convert_to_tensor=True)

def re_rank_chunks(query, chunks):
    rerank_response = co.rerank(query=query, documents=chunks, top_n=min(5, len(chunks)), model="rerank-english-v2.0")
    return [chunks[item.index] for item in rerank_response.results]

def get_most_relevant_chunks(query, chunks, chunk_embeddings, top_k=5):
    query_embedding = embedder.encode(query.strip(), convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = min(top_k, len(chunks))
    top_indices = torch.topk(scores, k=top_k).indices
    top_chunks = [chunks[i] for i in top_indices]
    top_scores = [scores[i].item() for i in top_indices]
    return re_rank_chunks(query, top_chunks), top_scores

def cohere_answer(question, context):
    response = co.generate(
        model='command-r-plus',
        prompt=f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=200
    )
    return response.generations[0].text.strip()

st.set_page_config(page_title="NeuraDocs - Doc Q&A Bot")
st.title("📄 NeuraDocs - AI-Powered Document Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, JPG, PNG)", type=["pdf", "docx", "jpg", "png"])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    with st.spinner("Extracting text from document..."):
        if file_ext == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_ext == "docx":
            text = extract_text_from_docx(uploaded_file)
        elif file_ext in ["jpg", "png"]:
            text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    if not text.strip():
        st.error("No text could be extracted from the document.")
        st.stop()

    st.success("Text extracted successfully!")
    st.subheader("📃 Document Summary")

    if st.button("Generate Summary"):
        if len(text) < 250:
            st.warning("The document is too short to summarize (minimum 250 characters required).")
        else:
            with st.spinner("Generating summary with Cohere..."):
                summary = co.summarize(text=text, model="summarize-xlarge", length="medium").summary
            st.write(summary)

    # Prepare chunks and embeddings
    chunks = get_chunks(text)
    chunk_embeddings = get_embeddings(chunks)
    stored_docs[uploaded_file.name] = (chunks, chunk_embeddings)

    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Type your question:")

    if user_question:
        with st.spinner("Thinking..."):
            top_chunks, top_scores = get_most_relevant_chunks(user_question, chunks, chunk_embeddings)
            context = "\n".join(top_chunks)
            answer = cohere_answer(user_question, context)
        st.markdown(f"**Answer:** {answer}")
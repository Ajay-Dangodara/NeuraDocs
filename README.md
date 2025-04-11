# 📄 NeuraDocs - AI-Powered Document Q&A Chatbot

NeuraDocs is a powerful document question-answering chatbot built with Streamlit and Cohere. It allows users to upload PDFs, DOCX files, and scanned images (JPG/PNG), extracts content using OCR, and intelligently answers questions using semantic search and large language models.

---

## 🚀 Features

- ✅ Upload PDFs, DOCX, JPG, or PNG files
- 🔍 OCR support for scanned PDFs and images (Tesseract)
- 🧠 Document summarization with Cohere
- 🗂️ Smart text chunking and embedding with `all-mpnet-base-v2`
- 🔁 Re-rank context chunks using Cohere's `rerank-english-v2.0`
- 🧾 Question answering using Cohere’s `command-r-plus`

---

## 🧠 How It Works

1. Extract text from files using PyMuPDF, python-docx, or Tesseract
2. Split text into semantic chunks and embed with SentenceTransformers
3. Match question to relevant chunks using cosine similarity
4. Rerank top matches using Cohere's Re-rank model
5. Feed best chunks and question to Cohere’s LLM for final answer

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Cohere API
- SentenceTransformers (`all-mpnet-base-v2`)
- Tesseract OCR
- PyMuPDF
- python-docx
- PIL (Pillow)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ajay-Dangodara/NeuraDocs.git
cd NeuraDocs 
```
### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```
### 3. Install Tesseract OCR

Windows:
Download the installer and add to your system PATH.

macOS:

```bash
brew install tesseract
```

Ubuntu/Linux:

```bash
sudo apt install tesseract-ocr
```

## 🔑 Set Up Cohere API Key

Open app.py and replace:

```python
co = cohere.Client("YOUR_COHERE_API_KEY")
```
with your actual API key from cohere.com.

## ▶️ Run the App
```bash
streamlit run app.py
```

## ✨ Example Use Cases
Upload a research paper and ask for its key findings

Summarize long legal documents

Ask questions from a scanned image or invoice

Extract insights from reports, contracts, or scanned data
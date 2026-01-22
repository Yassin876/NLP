import streamlit as st 
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv
import torch

load_dotenv()

if not torch.cuda.is_available():
    st.error("GPU غير متاح، البرنامج يتطلب GPU للعمل.")
    st.stop()

device = 'cuda'
st.write(f"Using device: {torch.cuda.get_device_name(0)}")

def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ''
    for page in reader.pages:
        if page.extract_text() is not None:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    import re
    text = re.sub('\s+', ' ', text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

uploaded_file = st.file_uploader('Upload your PDF file', type='pdf')

status = st.empty()
status.text("Loading embedding model...")
embedding = SentenceTransformer('all-MiniLM-L6-v2', device=device)
status.text("Embedding model loaded")

if uploaded_file is not None:
    status.text("Loading PDF...")
    text = load_pdf(uploaded_file)
    status.text("PDF loaded")

    status.text("Splitting text into chunks...")
    chunks = chunk_text(text)
    status.text(f"Text split into {len(chunks)} chunks")

    status.text("Creating embeddings...")
    with torch.no_grad():
        embeddings = embedding.encode(chunks, convert_to_numpy=True, device=device)
    status.text("Embeddings created")

    status.text("Building FAISS GPU index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embeddings.astype('float32'))
    status.text("FAISS GPU index ready")

    def retrieve_context(question, k=5):
        q_embedding = embedding.encode([question], convert_to_numpy=True, device=device)
        _, indices = gpu_index.search(q_embedding.astype('float32'), k)
        return '\n'.join([chunks[i] for i in indices[0]])

    def build_prompt(context, question):
        return f"""
You are an AI assistant.
Answer ONLY using the context below.
If the answer is not found, say: "المعلومة غير موجودة في الملف".
Context:
{context}
Question:
{question}
Answer:
"""

    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def ask_llm(prompt):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content

    st.title("LLM Document Search App (Groq)")
    question = st.text_input("Ask a question")
    button = st.button("Ask")

    if button and question:
        status.text("Retrieving context...")
        context = retrieve_context(question)
        status.text("Building prompt...")
        prompt = build_prompt(context, question)
        status.text("Asking LLM...")
        answer = ask_llm(prompt)
        status.text("Done!")
        st.subheader("Answer:")
        st.write(answer)

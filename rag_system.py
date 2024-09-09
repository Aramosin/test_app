import os
import json
import faiss
import streamlit as st

# Assume the exact absolute paths to the files
base_dir = os.getcwd()  # You can hardcode the absolute path here if needed
index_file = os.path.join(base_dir, "faiss_index.bin")
titles_file = os.path.join(base_dir, "document_titles.json")
documents_file = os.path.join(base_dir, "processed_documents.json")

st.title("Chipotle RAG System - File Loading Test")

# Log the exact paths to ensure they're correct
st.write(f"Index file path: {index_file}")
st.write(f"Titles file path: {titles_file}")
st.write(f"Documents file path: {documents_file}")

try:
    # Try loading each file one by one
    st.write("Loading FAISS index...")
    index = faiss.read_index(index_file)
    st.write("FAISS index loaded successfully!")

    st.write("Loading document titles...")
    with open(titles_file, 'r', encoding='utf-8') as f:
        titles = json.load(f)
    st.write(f"Loaded {len(titles)} titles successfully!")

    st.write("Loading processed documents...")
    with open(documents_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    st.write(f"Loaded {len(documents)} documents successfully!")

except Exception as e:
    st.error(f"Error: {e}")

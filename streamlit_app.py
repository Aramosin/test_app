import os
import streamlit as st
import json

# List all files in the current working directory to verify visibility
#st.write("Current Working Directory:", os.getcwd())
#st.write("Files in the directory:", os.listdir(os.getcwd()))

# Check if the specific files exist and can be opened
try:
    with open('faiss_index.bin', 'rb') as f:
        st.write("FAISS index loaded successfully.")

    with open('document_titles.json', 'r') as f:
        titles_data = json.load(f)
        st.write("Titles loaded successfully:", titles_data)

    with open('processed_documents.json', 'r') as f:
        documents_data = json.load(f)
        st.write("Documents loaded successfully:", documents_data)

except Exception as e:
    st.error(f"Error accessing files: {e}")

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import tiktoken
import streamlit as st

class RAGSystem:
    def __init__(self, index_file, titles_file, documents_file, model_name='all-MiniLM-L6-v2'):
        # Use absolute paths based on the script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_file = os.path.join(base_dir, index_file)
        self.titles_file = os.path.join(base_dir, titles_file)
        self.documents_file = os.path.join(base_dir, documents_file)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Load or create data during initialization
        self.load_or_create_data()

    # (Rest of the class remains unchanged)

# Initialize the OpenAI client securely using Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Helper function for counting tokens in a string
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

# Streamlit app
st.title("RAG-based QA System")

# Define relative paths based on the script's location
index_file = 'faiss_index.bin'
titles_file = 'document_titles.json'
documents_file = 'processed_documents.json'

# Optionally, check if files exist and display their paths
st.write("Index file exists:", os.path.exists(index_file))
st.write("Titles file exists:", os.path.exists(titles_file))
st.write("Documents file exists:", os.path.exists(documents_file))

query_input = st.text_input("Ask a question:")
if query_input:
    rag_system = RAGSystem(index_file, titles_file, documents_file)
    answer = rag_system.query(query_input)
    st.write("Answer:")
    st.write(answer)


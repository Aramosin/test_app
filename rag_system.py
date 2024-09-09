import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import tiktoken
import streamlit as st

# Initialize the OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Define folder path for JSON and index files
        self.text_files_dir = "text_files"
        
        # Construct full file paths for JSON and index files
        self.index_file = os.path.join(self.text_files_dir, "faiss_index.bin")
        self.titles_file = os.path.join(self.text_files_dir, "document_titles.json")
        self.documents_file = os.path.join(self.text_files_dir, "processed_documents.json")

        # Check if the files exist before loading or creating them
        if not os.path.exists(self.index_file) or not os.path.exists(self.titles_file) or not os.path.exists(self.documents_file):
            st.write("Files missing. Creating default index and documents...")
            self.create_index()
            self.create_documents()
        else:
            st.write("Loading existing files...")
            self.index = faiss.read_index(self.index_file)
            with open(self.titles_file, 'r', encoding='utf-8') as f:
                self.titles = json.load(f)
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

        # Initialize Sentence Transformer for embeddings
        self.model = SentenceTransformer(model_name)
        
        st.write(f"Loaded {len(self.documents)} documents:")
        for doc in self.documents:
            st.write(f"- {doc['title']} (content length: {len(doc['content'])} characters)")

    def create_index(self):
        """Placeholder for creating FAISS index from raw documents."""
        self.index = faiss.IndexFlatL2(384)  # Assuming 384-dimensional vectors for embeddings
        # Optionally save the FAISS index (you'd need to save this properly later)
        # faiss.write_index(self.index, self.index_file)

    def create_documents(self):
        """Create default documents and save them if the files do not exist."""
        self.documents = [{"title": "Sample Document", "content": "This is a sample document content."}]
        self.titles = ["Sample Document"]

        # Save the created documents
        with open(self.titles_file, 'w', encoding='utf-8') as f:
            json.dump(self.titles, f)
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)
    
    # Add other methods such as search, get_document_content, etc.

# Usage example:
# rag = RAGSystem()


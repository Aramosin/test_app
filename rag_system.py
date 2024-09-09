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
    def __init__(self, index_file, titles_file, documents_file, model_name='all-MiniLM-L6-v2'):
        # Save file paths for later use
        self.index_file = index_file
        self.titles_file = titles_file
        self.documents_file = documents_file

        # Check if the files exist before loading or creating them
        if not os.path.exists(index_file) or not os.path.exists(titles_file) or not os.path.exists(documents_file):
            st.write("Files missing. Creating default index and documents...")
            self.create_index()
            self.create_documents()
        else:
            self.index = faiss.read_index(index_file)
            with open(titles_file, 'r', encoding='utf-8') as f:
                self.titles = json.load(f)
            with open(documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        self.model = SentenceTransformer(model_name)
        
        st.write(f"Loaded {len(self.documents)} documents:")
        for doc in self.documents:
            st.write(f"- {doc['title']} (content length: {len(doc['content'])} characters)")

    def create_index(self):
        """Placeholder for creating FAISS index."""
        self.index = faiss.IndexFlatL2(384)  # Assuming 384-dimensional vectors
        # Optionally save index logic here if needed, e.g., faiss.write_index

    def create_documents(self):
        """Create default documents and save them if the files do not exist."""
        self.documents = [{"title": "Sample Document", "content": "This is a sample document content."}]
        self.titles = ["Sample Document"]

        # Save the created documents
        if self.titles_file:
            with open(self.titles_file, 'w', encoding='utf-8') as f:
                json.dump(self.titles, f)
        else:
            st.error("Error: titles_file is not defined or is invalid.")

        if self.documents_file:
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f)
        else:
            st.error("Error: documents_file is not defined or is invalid.")
    
    # Other methods such as search, get_document_content, etc., go here


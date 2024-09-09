import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import streamlit as st

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Print the current working directory for debugging
        st.write(f"Current working directory: {os.getcwd()}")

        # Base directory for faiss index, titles, and processed documents
        self.base_dir = os.getcwd()  # Ensure we're using the correct working directory
        
        # File paths for faiss index, titles, and processed documents
        self.index_file = os.path.join(self.base_dir, "faiss_index.bin")
        self.titles_file = os.path.join(self.base_dir, "document_titles.json")
        self.documents_file = os.path.join(self.base_dir, "processed_documents.json")

        # Debug: Print the actual paths being used
        st.write(f"Index file path: {self.index_file}")
        st.write(f"Titles file path: {self.titles_file}")
        st.write(f"Documents file path: {self.documents_file}")

        # Check if the individual files exist and log explicit errors
        if not os.path.exists(self.index_file):
            st.error(f"Index file not found: {self.index_file}")
        if not os.path.exists(self.titles_file):
            st.error(f"Titles file not found: {self.titles_file}")
        if not os.path.exists(self.documents_file):
            st.error(f"Documents file not found: {self.documents_file}")

        # If files are missing, create default documents
        if not os.path.exists(self.index_file) or not os.path.exists(self.titles_file) or not os.path.exists(self.documents_file):
            st.write("Files missing. Creating default index and documents...")
            self.create_index()
            self.create_documents()
        else:
            st.write("Loading existing FAISS index and document metadata...")
            self.index = faiss.read_index(self.index_file)
            with open(self.titles_file, 'r', encoding='utf-8') as f:
                self.titles = json.load(f)
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

        # Initialize Sentence Transformer model for embeddings
        self.model = SentenceTransformer(model_name)

        st.write(f"Loaded {len(self.documents)} documents.")
        for doc in self.documents:
            st.write(f"- {doc['title']} (content length: {len(doc['content'])} characters)")

    def create_index(self):
        """Create a default FAISS index."""
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

# Initialize the system
st.title("Chipotle RAG System")
rag_system = RAGSystem()

# Query interface
query = st.text_input("Enter your question here")
if st.button("Ask"):
    st.write(f"Processing query: {query}")
    # (Assuming you have a query method in the system to handle this)
    # result = rag_system.query(query)
    # st.write(result)


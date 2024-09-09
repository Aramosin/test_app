import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import streamlit as st

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Log current working directory for debugging
        st.write(f"Current working directory: {os.getcwd()}")

        # Define base directory for FAISS index, titles, and processed documents
        self.base_dir = os.getcwd()  # Get current working directory

        # Full paths for faiss index, document titles, and processed documents
        self.index_file = os.path.join(self.base_dir, "faiss_index.bin")
        self.titles_file = os.path.join(self.base_dir, "document_titles.json")
        self.documents_file = os.path.join(self.base_dir, "processed_documents.json")

        # Log file paths for debugging
        st.write(f"Index file path: {self.index_file}")
        st.write(f"Titles file path: {self.titles_file}")
        st.write(f"Documents file path: {self.documents_file}")

        # Check if files exist, and log explicitly if they do not
        self.check_files_exist()

        # If files are missing, create default documents
        if not os.path.exists(self.index_file) or not os.path.exists(self.titles_file) or not os.path.exists(self.documents_file):
            st.write("Files missing. Creating default index and documents...")
            self.create_index()
            self.create_documents()
        else:
            st.write("Loading existing FAISS index and document metadata...")
            self.load_existing_files()

        # Initialize Sentence Transformer model for embeddings
        self.model = SentenceTransformer(model_name)

        st.write(f"Loaded {len(self.documents)} documents.")
        for doc in self.documents:
            st.write(f"- {doc['title']} (content length: {len(doc['content'])} characters)")

    def check_files_exist(self):
        """Check if necessary files exist and log errors if they don't."""
        if not os.path.exists(self.index_file):
            st.error(f"Index file not found: {self.index_file}")
        if not os.path.exists(self.titles_file):
            st.error(f"Titles file not found: {self.titles_file}")
        if not os.path.exists(self.documents_file):
            st.error(f"Documents file not found: {self.documents_file}")

    def create_index(self):
        """Create default FAISS index."""
        self.index = faiss.IndexFlatL2(384)  # Assuming 384-dimensional vectors for embeddings
        # You might save the index later with: faiss.write_index(self.index, self.index_file)

    def create_documents(self):
        """Create default documents."""
        self.documents = [{"title": "Sample Document", "content": "This is a sample document content."}]
        self.titles = ["Sample Document"]

        # Save the created documents
        with open(self.titles_file, 'w', encoding='utf-8') as f:
            json.dump(self.titles, f)
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)

    def load_existing_files(self):
        """Load FAISS index and JSON documents."""
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.titles_file, 'r', encoding='utf-8') as f:
                self.titles = json.load(f)
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        except Exception as e:
            st.error(f"Error loading files: {e}")

# Streamlit app interface
st.title("Chipotle RAG System")

# Initialize RAG system
rag_system = RAGSystem()

# Query form
query = st.text_input("Enter your question here")
if st.button("Ask"):
    st.write(f"Processing query: {query}")
    # Process the query if needed
    # result = rag_system.query(query)
    # st.write(result)


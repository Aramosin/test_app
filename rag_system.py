import os
import json
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Assuming the files are located in the current directory
        self.base_dir = "."
        self.index_file = os.path.join(self.base_dir, "faiss_index.bin")
        self.titles_file = os.path.join(self.base_dir, "document_titles.json")
        self.documents_file = os.path.join(self.base_dir, "processed_documents.json")

        # Try to load the files, and if it fails, create new ones
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.titles_file, 'r', encoding='utf-8') as f:
                self.titles = json.load(f)
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            st.write(f"Loaded {len(self.documents)} documents.")
        except Exception as e:
            # If loading fails, create default files
            st.write("Files missing or could not be loaded. Creating default index and documents...")
            self.create_index()
            self.create_documents()

        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)

    def create_index(self):
        # Creating a placeholder FAISS index
        self.index = faiss.IndexFlatL2(384)  # 384 is the dimension of 'all-MiniLM-L6-v2'
        st.write("FAISS index created.")

    def create_documents(self):
        # Creating placeholder documents
        self.documents = [{"title": "Sample Document", "content": "This is a sample document content."}]
        self.titles = ["Sample Document"]
        st.write("Created sample documents.")

        # Saving the new files
        with open(self.titles_file, 'w', encoding='utf-8') as f:
            json.dump(self.titles, f)
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)

# Streamlit UI
st.title("Chipotle RAG System")

# Initialize the system
rag_system = RAGSystem()

# Text input for questions
query = st.text_input("Enter your question here")
if st.button("Ask"):
    st.write(f"Processing query: {query}")



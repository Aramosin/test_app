import os
import streamlit as st
from rag_system import RAGSystem  # Import the RAGSystem class

# Initialize the RAG system
index_file = 'faiss_index.bin'
titles_file = 'document_titles.json'
documents_file = 'processed_documents.json'

# Streamlit app title
st.title("RAG-based QA System")

# Create an instance of the RAG system
rag_system = RAGSystem(index_file, titles_file, documents_file)

# Input field for the query
query_input = st.text_input("Ask a question:")

# Process the query when the user submits it
if query_input:
    answer = rag_system.query(query_input)
    st.write("Answer:")
    st.write(answer)


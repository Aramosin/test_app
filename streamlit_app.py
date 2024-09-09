import streamlit as st
from rag_system import RAGSystem

st.title("Chipotle RAG System")

@st.cache_resource
def load_rag_system():
    index_file = "faiss_index.bin"
    titles_file = "document_titles.json"
    documents_file = "processed_documents.json"
    return RAGSystem(index_file, titles_file, documents_file)

rag = load_rag_system()

question = st.text_input("Enter your question here")

if st.button("Ask"):
    if question:
        with st.spinner("Generating answer..."):
            answer = rag.query(question)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")
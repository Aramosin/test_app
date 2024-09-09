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
        # Base directory for faiss index, titles, and processed documents
        self.base_dir = "."

        # Directory containing the raw text files
        self.text_files_dir = "text_files"
        
        # Paths for faiss index, titles, and processed documents in base directory
        self.index_file = os.path.join(self.base_dir, "faiss_index.bin")
        self.titles_file = os.path.join(self.base_dir, "document_titles.json")
        self.documents_file = os.path.join(self.base_dir, "processed_documents.json")

        # Check if the files exist before loading them
        if not os.path.exists(self.index_file) or not os.path.exists(self.titles_file) or not os.path.exists(self.documents_file):
            st.write("Required files are missing. Please ensure the index and JSON files are present in the base directory.")
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

    def search(self, query, k=3):
        """Search for relevant documents using FAISS."""
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector, k)
        relevant_docs = [self.titles[i] for i in I[0]]
        
        st.write(f"Relevant documents for query '{query}':")
        for doc in relevant_docs:
            st.write(f"- {doc}")
        return relevant_docs

    def get_document_content(self, title, query):
        """Retrieve relevant content from the raw text files in the text_files directory."""
        file_path = os.path.join(self.text_files_dir, title)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = self.split_into_chunks(content)
                relevant_chunks = self.get_relevant_chunks(chunks, query)
                return relevant_chunks
        else:
            st.write(f"Text file for {title} not found in {self.text_files_dir}.")
            return []

    def split_into_chunks(self, text, chunk_size=1000):
        """Split text into manageable chunks."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def get_relevant_chunks(self, chunks, query):
        """Get the most relevant chunks for a query."""
        chunk_embeddings = self.model.encode(chunks)
        query_embedding = self.model.encode([query])
        
        similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(similarities)[::-1][:3]  # Top 3 most similar chunks
        
        return [chunks[i] for i in top_indices]

    def generate_answer(self, query, relevant_docs):
        """Generate an answer based on relevant documents."""
        context = ""
        max_tokens = 14000  # Max tokens including the prompt

        for title in relevant_docs:
            relevant_chunks = self.get_document_content(title, query)
            for chunk in relevant_chunks:
                new_context = f"Document: {title}\nContent: {chunk}\n\n"
                if num_tokens_from_string(context + new_context) <= max_tokens:
                    context += new_context
                else:
                    st.write(f"Reached token limit. Stopping at document: {title}")
                    break
            if num_tokens_from_string(context) > max_tokens:
                break

        context_tokens = num_tokens_from_string(context)
        st.write(f"Context length: {context_tokens} tokens")
        st.write("Context preview:")
        st.write(context[:500] + "..." if len(context) > 500 else context)

        if context_tokens == 0:
            return "I'm sorry, but I couldn't find any relevant information to answer your question."

        prompt = f"Based on the following documents, please answer this question: {query}\n\nContext:\n{context}\n\nAnswer:"
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given documents. Provide accurate information based solely on the context provided."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {str(e)}")
            return "I'm sorry, but an error occurred while generating the answer. Please try again later."

    def query(self, question):
        """Main function to handle a query."""
        relevant_docs = self.search(question)
        answer = self.generate_answer(question, relevant_docs)
        return answer

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import tiktoken
import streamlit as st
import re

# Helper function for counting tokens in a string
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

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

    def load_or_create_data(self):
        try:
            # Load or create the FAISS index
            if not os.path.exists(self.index_file):
                self.create_index()
            else:
                self.index = faiss.read_index(self.index_file)

            # Load document files
            if not os.path.exists(self.titles_file) or not os.path.exists(self.documents_file):
                self.create_documents()
            else:
                with open(self.titles_file, 'r', encoding='utf-8') as f:
                    self.titles = json.load(f)
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            self.create_documents()

    def create_index(self):
        # Placeholder for creating FAISS index
        self.index = faiss.IndexFlatL2(384)  # 384 is the dimension for 'all-MiniLM-L6-v2'

    def create_documents(self):
        # Placeholder for document creation
        self.documents = [
            {"title": "Sample Document", "content": "This is a sample document content."}
        ]
        self.titles = ["Sample Document"]

        # Save the newly created documents
        try:
            with open(self.titles_file, 'w', encoding='utf-8') as f:
                json.dump(self.titles, f, ensure_ascii=False, indent=4)
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=4)
        except Exception as e:
            st.error(f"Error saving documents: {str(e)}")

    def search(self, query, k=3):
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector, k)
        relevant_docs = [self.titles[i] for i in I[0]]

        # Prioritize Employee Handbook for HR-related queries (if needed)
        hr_keywords = ["leave", "vacation", "time off", "sick", "absence", "holiday", "benefits", "policy"]
        handbook_title = "Employee Handbook_Multistate (English)_2024.txt"
        
        if any(keyword in query.lower() for keyword in hr_keywords) and handbook_title in self.titles:
            if handbook_title not in relevant_docs:
                relevant_docs = [handbook_title] + relevant_docs[:2]

        return relevant_docs

    def get_document_content(self, title, query):
        for doc in self.documents:
            if doc['title'] == title:
                content = doc['content']
                chunks = self.split_into_chunks(content)
                relevant_chunks = self.get_relevant_chunks(chunks, query)
                return relevant_chunks
        return []

    def split_into_chunks(self, text, max_chunk_size=500):
        # Split text by paragraphs or newlines
        paragraphs = re.split(r'\n\n|\n', text)  # Splits by double or single newlines

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_chunk_size:
                current_chunk += paragraph + "\n"
            else:
                chunks.append(current_chunk.strip())  # Add the current chunk
                current_chunk = paragraph  # Start a new chunk with the current paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())  # Add the last chunk

        return chunks

    def get_relevant_chunks(self, chunks, query):
        query_embedding = self.model.encode([query])

        # Prioritize chunks that mention key terms from the query (e.g., 'salsa', 'tomato')
        keywords = query.lower().split()
        filtered_chunks = [chunk for chunk in chunks if any(keyword in chunk.lower() for keyword in keywords)]

        if not filtered_chunks:
            # If no chunks match the filter, fall back to the default chunks
            filtered_chunks = chunks

        # Now perform embedding similarity search on the filtered chunks
        chunk_embeddings = self.model.encode(filtered_chunks)
        similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()

        # Get top 3 most similar chunks
        top_indices = np.argsort(similarities)[::-1][:3]
        relevant_chunks = [filtered_chunks[i] for i in top_indices]

        # Debugging: Log similarity scores and selected chunks
        st.write(f"Similarity scores: {similarities}")
        st.write(f"Selected chunks: {relevant_chunks}")

        return relevant_chunks


    def generate_answer(self, query, relevant_docs):
        st.write("Starting to generate answer...")  # Debugging log

        context = ""
        max_tokens = 14000  # Token limit for the context and response

        # Add debugging log for relevant docs
        st.write(f"Relevant documents: {relevant_docs}")

        # Build the context from relevant documents
        for title in relevant_docs:
            relevant_chunks = self.get_document_content(title, query)
            for chunk in relevant_chunks:
                # Use a cleaner and more structured format for adding context
                new_context = f"Document: {title}\nSteps:\n{chunk}\n\n"
                if num_tokens_from_string(context + new_context) <= max_tokens:
                    context += new_context
                else:
                    break
            if num_tokens_from_string(context) > max_tokens:
                break

        # Debugging logs for context
        st.write(f"Context being passed to OpenAI: {context[:500]}...")  # Show the first 500 characters for clarity

        if len(context) == 0:
            st.error("Context is empty, no relevant information found.")
            return "I'm sorry, but I couldn't find any relevant information to answer your question."

        # Build the prompt for OpenAI
        prompt = f"Answer the following question based only on the provided document context:\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"

        st.write(f"Prompt to OpenAI: {prompt[:500]}...")  # Show the first 500 characters for clarity

        try:
            # Corrected method for OpenAI SDK 1.0.0+
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the user’s question based solely on the provided documents. Do not use external knowledge."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000  # Adjust based on your use case
            )

            # Debugging log to show response from OpenAI
            st.write("OpenAI response received.")
        
            # Correctly access the response text using dot notation
            return response.choices[0].message.content  
        except Exception as e:
            st.error(f"An error occurred while calling OpenAI: {str(e)}")
            return f"An error occurred: {str(e)}"


        # Debugging logs for context
        st.write(f"Context: {context[:500]}...")  # Show the first 500 characters for clarity

        if len(context) == 0:
            st.error("Context is empty, no relevant information found.")
            return "I'm sorry, but I couldn't find any relevant information to answer your question."

        # Build the prompt for OpenAI
        prompt = f"Based on the following documents, please answer this question: {query}\n\nContext:\n{context}\n\nAnswer:"

        st.write(f"Prompt to OpenAI: {prompt[:500]}...")  # Show first 500 characters for clarity

        try:
            # Corrected method for OpenAI SDK 1.0.0+
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000  # Adjust based on your use case
            )

            # Debugging log to show response from OpenAI
            st.write("OpenAI response received.")
            return response.choices[0].message.content  # Correct
  # Access the response text correctly
        except Exception as e:
            st.error(f"An error occurred while calling OpenAI: {str(e)}")
            return f"An error occurred: {str(e)}"

    def query(self, question):
        relevant_docs = self.search(question)
        answer = self.generate_answer(question, relevant_docs)
        return answer
# Initialize the OpenAI client securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Debugging log to check which key is being used
st.write(f"API Key in use: {openai.api_key[:6]}...")  # Display first 6 characters (should show "sk-...")


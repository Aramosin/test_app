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

    def load_or_create_data(self):
        try:
            # Load or create the FAISS index
            if not os.path.exists(self.index_file):
                st.write("Index file not found. Creating new index...")
                self.create_index()
            else:
                try:
                    self.index = faiss.read_index(self.index_file)
                    st.write("FAISS index loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading FAISS index: {e}")
                    self.create_index()

            # Load or create document files
            if not os.path.exists(self.titles_file) or not os.path.exists(self.documents_file):
                st.write("Document files not found. Creating new documents...")
                self.create_documents()
            else:
                try:
                    with open(self.titles_file, 'r', encoding='utf-8') as f:
                        self.titles = json.load(f)
                    with open(self.documents_file, 'r', encoding='utf-8') as f:
                        self.documents = json.load(f)
                    st.write(f"Document files loaded successfully. Loaded {len(self.documents)} documents.")
                except Exception as e:
                    st.error(f"Error loading document files: {e}")
                    self.create_documents()

            st.write(f"Loaded {len(self.documents)} documents:")
            for doc in self.documents:
                st.write(f"- {doc['title']} (content length: {len(doc['content'])} characters)")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    def create_index(self):
        # Placeholder for creating FAISS index
        self.index = faiss.IndexFlatL2(384)  # 384 is the dimension for 'all-MiniLM-L6-v2'
        st.write("New FAISS index created.")

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
            st.write("New documents created and saved.")
        except Exception as e:
            st.error(f"Failed to save documents: {e}")

    def search(self, query, k=3):
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector, k)
        relevant_docs = [self.titles[i] for i in I[0]]

        # Prioritize Employee Handbook for HR-related queries
        hr_keywords = ["leave", "vacation", "time off", "sick", "absence", "holiday", "benefits", "policy"]
        handbook_title = "Employee Handbook_Multistate (English)_2024.txt"
        if any(keyword in query.lower() for keyword in hr_keywords) and handbook_title in self.titles:
            if handbook_title not in relevant_docs:
                relevant_docs = [handbook_title] + relevant_docs[:2]

        st.write(f"Relevant documents for query '{query}':")
        for doc in relevant_docs:
            st.write(f"- {doc}")
        return relevant_docs

    def get_document_content(self, title, query):
        for doc in self.documents:
            if doc['title'] == title:
                content = doc['content']
                chunks = self.split_into_chunks(content)
                relevant_chunks = self.get_relevant_chunks(chunks, query)
                return relevant_chunks
        st.write(f"No content found for document: {title}")
        return []

    def split_into_chunks(self, text, chunk_size=1000):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def get_relevant_chunks(self, chunks, query):
        chunk_embeddings = self.model.encode(chunks)
        query_embedding = self.model.encode([query])
        
        similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(similarities)[::-1][:3]  # Get top 3 most similar chunks
        
        return [chunks[i] for i in top_indices]

    def generate_answer(self, query, relevant_docs):
        context = ""
        max_tokens = 14000  # Token limit for the context and response

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
        relevant_docs = self.search(question)
        answer = self.generate_answer(question, relevant_docs)
        return answer

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

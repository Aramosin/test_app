from flask import Flask, render_template, request, jsonify
from rag_system import RAGSystem  # Import your RAG system class

app = Flask(__name__)

# Initialize your RAG system
index_file = r"E:\PROJECT\chipotle\faiss_index.bin"
titles_file = r"E:\PROJECT\chipotle\document_titles.json"
documents_file = r"E:\PROJECT\chipotle\processed_documents.json"
rag = RAGSystem(index_file, titles_file, documents_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    question = request.form['question']
    answer = rag.query(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
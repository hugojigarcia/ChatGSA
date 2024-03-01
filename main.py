from flask import Flask
from libraries.rag import RAG

app = Flask(__name__)

# ===================================
# PROVIONAL
input_path = "vector_db"
rag = RAG(vectordb_path=input_path)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

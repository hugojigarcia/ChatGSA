from flask import Flask
from libraries.rag import RAG

app = Flask(__name__)

# ===================================
# PROVIONAL

from dotenv import load_dotenv
import os
load_dotenv()

@app.route('/')
def hello_world():
    text = "Hello, World! BD count:"
    text += str(os.getenv('OPENAI_API_KEY'))
    return text

if __name__ == '__main__':
    app.run(debug=True)

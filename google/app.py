import sys
import os
config_dir = os.path.join(os.getcwd(), 'config')
sys.path.append(config_dir)

from config import Config
from flask import Flask, render_template, request, jsonify
from libraries.rag import RAG
import json
import argparse
import os
from dotenv import load_dotenv


app = Flask(__name__)



# ===================================
# PROVIONAL
# input_path = "pipeline_files/3_vectordb/coffe_podcasts"
# rag = RAG(vectordb_path=input_path)

 # ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config_file_path', type=str, default="config/config.yaml", help="Path to the configuration file")
args = parser.parse_args()
config = Config(args.config_file_path)

# TASK
llm_model_name = config.get_variable("3_vectordb.llm_model_name")
embeddings_model_name = config.get_variable("3_vectordb.embeddings_model_name")

load_dotenv()
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')
INDEX_ID = os.getenv('INDEX_ID')
ENDPOINT_ID = os.getenv('ENDPOINT_ID')


rag = RAG(PROJECT_ID, REGION, BUCKET_NAME, llm_model_name, embeddings_model_name, INDEX_ID, ENDPOINT_ID)

def get_mock():
    answer = "El escenario que está ocurriendo, el de encontrar el bosón de Higgs y nada más se lo llamaba el escenario pesadillesco, el éxese, porque era decir fantástico, Higgs, Englert, toda la gente de Braut, pues se merecen grandes honores, pero para dónde seguimos, por ese lado yo creo que está clarísimo, para dejarlo claro, yo creo que todos los científicos y científicos deseamos"
    sources = [
        {
            "episode": "ep397_whisper",
            "start": "01:42:46.20",
            "end": "01:43:12.32",
            "text": "el escenario que está ocurriendo, el de encontrar el bosón de Higgs y nada más se lo llamaba el escenario pesadillesco, el éxese, porque era decir fantástico, Higgs, Englert, toda la gente de Braut, pues se merecen grandes honores, pero para dónde seguimos, por ese lado yo creo que está clarísimo, para dejarlo claro, yo creo que todos los científicos y científicos deseamos,"
        },
        {
            "episode": "ep397_whisper",
            "start": "01:42:46.20",
            "end": "01:43:12.32",
            "text": "el escenario que está ocurriendo, el de encontrar el bosón de Higgs y nada más se lo llamaba el escenario pesadillesco, el éxese, porque era decir fantástico, Higgs, Englert, toda la gente de Braut, pues se merecen grandes honores, pero para dónde seguimos, por ese lado yo creo que está clarísimo, para dejarlo claro, yo creo que todos los científicos y científicos deseamos,"
        },
        {
            "episode": "ep397_whisper",
            "start": "01:42:46.20",
            "end": "01:43:12.32",
            "text": "el escenario que está ocurriendo, el de encontrar el bosón de Higgs y nada más se lo llamaba el escenario pesadillesco, el éxese, porque era decir fantástico, Higgs, Englert, toda la gente de Braut, pues se merecen grandes honores, pero para dónde seguimos, por ese lado yo creo que está clarísimo, para dejarlo claro, yo creo que todos los científicos y científicos deseamos,"
        },
        {
            "episode": "ep397_whisper",
            "start": "01:42:46.20",
            "end": "01:43:12.32",
            "text": "el escenario que está ocurriendo, el de encontrar el bosón de Higgs y nada más se lo llamaba el escenario pesadillesco, el éxese, porque era decir fantástico, Higgs, Englert, toda la gente de Braut, pues se merecen grandes honores, pero para dónde seguimos, por ese lado yo creo que está clarísimo, para dejarlo claro, yo creo que todos los científicos y científicos deseamos,"
        },
    ]
    return answer, sources
# ===================================


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    chat_history = json.loads(request.form['chat_history'])
    for i, el in enumerate(chat_history):
        chat_history[i] = (el[0], el[1])

    # MOCK
    answer, sources = get_mock()

    # NO MOCK
    # result = rag.ask(question, chat_history)
    # answer = result["answer"]
    chat_history.extend([(question, answer)])
    # sources = []
    # for el in result["source_documents"]:
    #     source = {}
    #     source["episode"] = el.metadata['filename']
    #     source["start"] = el.metadata['start']
    #     source["end"] = el.metadata['end']
    #     source["text"] = el.page_content
    #     sources.append(source)
    response = {
        "answer": answer,
        "sources": sources,
        "chat_history": chat_history
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

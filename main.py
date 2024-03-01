from flask import Flask, render_template, request, jsonify
# from libraries.rag import RAG
from libraries.mock import MOCK
import json

app = Flask(__name__)



# ===================================
# PROVIONAL
# input_path = "vector_db"
# rag = RAG(vectordb_path=input_path)
mock_obj = MOCK()

def get_mock():
    answer = "El Kraal es el conjunto de responsables del Grupo encargados de liderar y gestionar las actividades educativas y formativas de la organización, y su elección y admisión se realiza mediante consenso del mismo."
    sources = [
        {
            "source": "Estatutos (2022)",
            "text": "La elección y admisión de los nuevos miembros del equipo de Kraal se realizará mediante el consenso del mismo."
        },
        {
            "source": "2. Programa Educativo de Grupo 2022 - 2025",
            "text": "Las familias, antiguas personas del equipo de Kraal y conocidos del Kraal actual tienen la opción de ayudar con las diferentes tareas de los cargos durante el trimestre y los campamentos."
        },
        {
            "source": "Reglamento de Regimen Interno (2023)",
            "text": "``` El Equipo de Kraal es el conjunto de responsables del Grupo encargados del funcionamiento de esta y del cumplimiento de los fines de grupo mediante la preparación, desarrollo y supervisión de las actividades. El Equipo de Kraal actuará en consecuencia a las decisiones que tome la Asamblea General. ```"
        },
        {
            "source": "Reglamento de Regimen Interno (2023)",
            "text": "``` Son aquellas personas asociadas encargadas de liderar y gestionar las actividades educativas y formativas de la organización. A este equipo se le denomina Equipo de Kraal. ```"
        }
    ]
    return answer, sources
# ===================================


@app.route('/')
def home():
    return mock_obj.get_mock()
    # return render_template('index.html')


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
    #     source["source"] = el.metadata['source']
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

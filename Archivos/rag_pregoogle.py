import os
from dotenv import load_dotenv
import openai
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



class RAG:
    def __init__(self, vectordb_path):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # self.embeddings = HuggingFaceEmbeddings()
        self.embeddings = OpenAIEmbeddings()
        self.vectordb_path = vectordb_path
        self.vectordb = Chroma(persist_directory=self.vectordb_path, embedding_function=self.embeddings)
        print(self.vectordb._collection.count())

        self.llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

        self.conversational_retrieval_chain = self.__get_conversational_retrieval_chain()
    

    def __get_conversational_retrieval_chain(self):
        return ConversationalRetrievalChain.from_llm( # Lo que hace ConversationalRetrievalChain que lo diferencia de RetrievalQA.from_chain_type, es que añade un paso extra: coge el historial de chat y la pregunta que se realiza, y lo condensa en una única pregunta para pasásrsela al retriever. Es decir, si en el historial se ha hablado de NVIDIA, y se le pregunta por "esa empresa", lo que hace es juntar esa pregunta, con el historial, y generar la pregunta "la empresa NVIDIA", y le pasa esto último al retriever.
            # 0. Juntar el historial de chat y la pregunta en una única pregunta -> Esto siempre se hace con un modelo LLM
            # 1. Retriever: de qué manera se extraen los documentos de la vectordb para responder a la question Más retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/ 
            retriever=self.vectordb.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 5, "k": 3}), # MMR: usando similaridad, pero se queda con los k que más información diversa aportan de los fetch_k extraídos
            # 2. Juntar los documentos extraídos y responder a la pregunta -> Esto siempre se hace con un modelo LLM
            llm=self.llm,
            chain_type="stuff", # Es el tipo por defecto, que simplemente junta todos los docuemntos extraídos en uno solo (uno a continuación del otro, sin más procesamiento). Pros: Solo hace una llamada al LLM. Contras: si hay muchos documentos, puede que no quepan todos en la ventana de contexto del LLM.
            combine_docs_chain_kwargs={ # Es lo mismo que chain_type_kwargs en RetrievalQA.from_chain_type. Argumentos que recibe la cadena. En el prompt indicamos qué debe hacer (responder a la pregunta usando contexto e historial de chat). Por defecto NO usa el resto de historial para responder. (se puede ver con verbose=True). Esto no se puede usar con chain_type que no sea 'stuff'
                "prompt": self.__get_chain_prompt(),
                "memory": ConversationBufferMemory(
                    memory_key="chat_history", # esto es para alinearlo con la variable de entrada del prompt (pero parece que da igual lo que le pongas, le mete bien el historial de chat)
                    input_key="question",
                    return_messages=True # Para que devuelva el historial como una lista, en lugar de como un único string, según el curso. Aunque parece que en cualquier caso lo devuelve como una lista.
                )
            },
            return_source_documents=True,
            return_generated_question=True,
            verbose=False
        )
    
    def __get_chain_prompt(self):
        template = """
        Utiliza los siguientes elementos de contexto (delimitados por <ctx></ctx>) y el historial de chat (delimitado por <hs></hs>) para responder a la pregunta (delimitada por <qst></qst>). Genera una única respuesta. Si no sabes la respuesta, di simplemente que no la sabes, no intentes inventarte una respuesta. Utiliza tres frases como máximo. La respuesta debe ser lo más concisa posible.:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {chat_history}
        </hs>
        ------
        <qst>
        {question}
        </qst>
        ------
        Respuesta:
        """
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"], # Se declara el prompt, y se define que tiene 3 variables de entrada
            template=template,
        )
    
    def ask(self, question, chat_history):
        return self.conversational_retrieval_chain.invoke({"question": question, "chat_history": chat_history})
        



# input_path = "pipeline_files/3_vectordb_coffee"
# rag = RAG(vectordb_path=input_path)
# chat_history = list([])

# question = "Cómo se descubrió?"
# result = rag.ask(question, chat_history)
# chat_history.extend([(question, result["answer"])])
# print(result["answer"])
import torch
import os
from dotenv import load_dotenv
# import openai
# from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
import vertexai
from langchain_google_vertexai import VertexAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import MatchingEngine


class RAG:
    def __init__(self, project_id, region, bucket_name, llm_model_name, embeddings_model_name, index_id, endpoint_id): # vectordb_path
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.embeddings = HuggingFaceEmbeddings()
        # self.vectordb_path = vectordb_path
        # self.vectordb = Chroma(persist_directory=self.vectordb_path, embedding_function=self.embeddings)
        # openai.api_key = os.getenv('OPENAI_API_KEY')
        # self.llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

        self.embeddings_model_name = embeddings_model_name
        self.embeddings = VertexAIEmbeddings(model_name=self.embeddings_model_name)
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name
        self.llm_model_name = llm_model_name
        self.index_id = index_id
        self.endpoint_id = endpoint_id


        vertexai.init(project=self.project_id, location=self.region)
        self.vectordb = MatchingEngine.from_components(
            project_id=self.project_id,
            region=self.region,
            gcs_bucket_name=self.bucket_name,
            embedding=self.embeddings,
            index_id=self.index_id,
            endpoint_id=self.endpoint_id,
        )
        self.llm = VertexAI(model=self.llm_model_name)

        self.conversational_retrieval_chain = self.__get_conversational_retrieval_chain()
    

    def __get_conversational_retrieval_chain(self):
        # document_content_description, metadata_field_info = self.__get_metadata_info()
        # self_query_retriever = SelfQueryRetriever.from_llm( # LLM Aided retrieval: usa un modelo LLM para hacer la query, ya que usa filtros de metadatos (estos metadatos hay que definirlos en metadata_field_info y document_content_description). Lo que hace el modelo es adaptar la question, para generar una query que filtra por los metadatos según lo que se pregunte.
        #     llm=self.llm,
        #     vectorstore=self.vectordb,
        #     document_contents=document_content_description,
        #     metadata_field_info=metadata_field_info,
        #     enable_limit=True, # esto permite que el modelo LLM pueda limitar el número de documentos que se extraen, por ejemplo preguntando "Dime 2 libros de X autor" y que solo se extraigan 2 documentos
        #     verbose=True
        # )
        return ConversationalRetrievalChain.from_llm( # Lo que hace ConversationalRetrievalChain que lo diferencia de RetrievalQA.from_chain_type, es que añade un paso extra: coge el historial de chat y la pregunta que se realiza, y lo condensa en una única pregunta para pasásrsela al retriever. Es decir, si en el historial se ha hablado de NVIDIA, y se le pregunta por "esa empresa", lo que hace es juntar esa pregunta, con el historial, y generar la pregunta "la empresa NVIDIA", y le pasa esto último al retriever.
            # 0. Juntar el historial de chat y la pregunta en una única pregunta -> Esto siempre se hace con un modelo LLM
            # 1. Retriever: de qué manera se extraen los documentos de la vectordb para responder a la question Más retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/ 
            retriever=self.vectordb.as_retriever(),
            # retriever=self.vectordb.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 5, "k": 3}), # MMR: usando similaridad, pero se queda con los k que más información diversa aportan de los fetch_k extraídos
            # retriever=self_query_retriever,
            # retriever=ContextualCompressionRetriever( # Lo que hace es, de cada documento extraído, extraer sólo la información relevante para responder a la pregunta. Se extraen con base_retriever (puede ser cualquiera) y luego se comprimen usando base_compressor (un LLM)
            #     base_compressor=LLMChainExtractor.from_llm(self.llm), # CUIDADO: esto requiere hacer una llamada al LLM por cada documento extraído, por lo que si se usa la API de OpenAI, se consumen créditos, y si se usa un modelo local, es muy lento (y peor).
            #     base_retriever=self_query_retriever # Aquí se podría usar cualquiera de los otros retrievers (similarity, mmr, SelfQueryRetriever, etc.)
            # ),
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
        Utiliza los siguientes elementos de contexto (delimitados por <ctx></ctx>) y el historial de chat (delimitado por <hs></hs>) para responder a la pregunta (delimitada por <qst></qst>). Genera una única respuesta. Si no sabes la respuesta, di simplemente que no la sabes, no intentes inventarte una respuesta:
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
    
    def __get_metadata_info(self):
        document_content_description = "Episodes of the podcast" # Describe el contenido general de los metadatos.
        metadata_field_info = [ # Se definen cada uno de los atributos de los metadatos.
            AttributeInfo(
                name="filename",
                # description="The podcast episode that must follow this structure: 'epXXX_whisper', where XXX is an integer that represents the episode number. Some examples of podcasts episodes are: 'ep001_whisper', 'ep002_whisper', 'ep003_whisper', 'ep047_whisper', 'ep112_whisper'.",
                description="It's a number that represents the podcast episode. It's a 3 digit number, some podcasts episodes examples are '001', '007', '056', '123'. Keep in mind that some podcasts episodes follow this structure: XXX_Z, where XXX is the 3 digit number, and Z is a capital A or a capital B. Some examples are '003_A', '032_B' and '421_A'",
                type="string",
            ),
        ]
        return document_content_description, metadata_field_info
    

    def ask(self, question, chat_history):
        return self.conversational_retrieval_chain.invoke({"question": question, "chat_history": chat_history})
        



# input_path = "pipeline_files/3_vectordb_coffee"
# rag = RAG(vectordb_path=input_path)
# chat_history = list([])

# question = "Cómo se descubrió?"
# result = rag.ask(question, chat_history)
# chat_history.extend([(question, result["answer"])])
# print(result["answer"])
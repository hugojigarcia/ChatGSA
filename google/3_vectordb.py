import argparse
from config.config import Config
import json
from langchain.docstore.document import Document
# from langchain.vectorstores import Chroma --> Está deprecated el importarlo así
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings --> Está deprecated el importarlo así
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings.vertexai import VertexAIEmbeddings
import os
import shutil
# import torch
from tqdm import tqdm
from google.cloud import aiplatform
from langchain_community.vectorstores import MatchingEngine
from dotenv import load_dotenv


class VectorDBGenerator:
    # def __init__(self, output_path, reset_db=True):
    def __init__(self, project_id, region, bucket_name, llm_model_name, embeddings_model_name, new_index_name, index_id=None, endpoint_id=None):
        # self.output_path = output_path
        # if reset_db:
        #     self.__delete_db()
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.embeddings = HuggingFaceEmbeddings()

        self.embeddings_model_name = embeddings_model_name
        self.embeddings = VertexAIEmbeddings(model_name=self.embeddings_model_name)
        self.project_id = project_id
        self.region = region
        self.bucket_name = bucket_name
        self.llm_model_name = llm_model_name
        self.new_index_name = new_index_name
        self.index_id = index_id
        self.endpoint_id = endpoint_id

    def generate_vectordb(self, input_path, chunk_duration, overlap_duration):
        json_data = self.__read_json_file(input_path)
        chunks = self.__chunk_aggregator(json_data, chunk_duration, overlap_duration)
        documents = self.__generate_documents(chunks)
        vectordb = self.__generate_vectors(documents)
        return vectordb
    

    def __read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data


    # def __chunk_aggregator(self, data, chunk_duration):
    #     duration_chunk, chunk_text, chunk_start, chunk_end = 0, "", "00:00:00,000", "00:00:00,000"
    #     chunks = []
    #     for el in data:
    #         start = float(self.__time_to_seconds(el["start"]))
    #         end = float(self.__time_to_seconds(el["end"]))
    #         duration_chunk += end - start
    #         # chunk_text += el["text"] + " "
    #         chunk_text += el["content"] + " "
    #         if duration_chunk >= chunk_duration:
    #             chunk_end = el["end"]
    #             # chunks.append({"filename": el["filename"], "start": chunk_start, "end": chunk_end, "text": chunk_text})
    #             chunks.append({"filename": el["episode"], "start": chunk_start, "end": chunk_end, "text": chunk_text})
    #             chunk_start = el["end"]
    #             chunk_text = ""
    #             duration_chunk = 0
    #     return chunks


    def __chunk_aggregator(self, data, chunk_duration, overlap_duration):
        duration_chunk, chunk_text, chunk_start, chunk_end = 0, "", "00:00:00.00", "00:00:00.00"
        chunks = []
        for idx, el in enumerate(data):
            start = float(self.__time_to_seconds(el["start"]))
            end = float(self.__time_to_seconds(el["end"]))
            duration_chunk += end - start
            # chunk_text += el["text"] + " "
            chunk_text += el["content"] + " "

            if duration_chunk >= chunk_duration:
                # Add chunk with overlap
                chunk_end = el["end"]
                # chunks.append({"filename": el["filename"], "start": chunk_start, "end": chunk_end, "text": chunk_text})
                chunks.append({"filename": el["episode"], "start": chunk_start, "end": chunk_end, "text": chunk_text})
                
                # Para encontrar donde comienza el overlap, hay que ir hacia atrás desde el fin del chunk actual (chunk_end). Si el chunk actual termina en end=100, y el overlap es 
                # de 10 segundos, el chunk siguiente debe empezar en 90, por lo tanto hay que ir recorriendo el json hacia atrás hasta encontrar un chunk que comience 
                # en un número menor a 90. Pero no es tan trivial como hacer la resta, porque puede que haya segundos de silencio entre chunks, por lo que hay que ir recorriendo
                # hacia atrás, sumando la duración de los chunks hasta que la suma de estos sea mayor o igual al overlap_duration.
                prev_duration = 0
                for prev_id, prev_el in enumerate(reversed(data[:idx+1])):  # Va recorriendo el array de chunks hacia atrás
                    prev_duration += float(self.__time_to_seconds(prev_el["end"])) - float(self.__time_to_seconds(prev_el["start"])) # Va sumando la duración de los chunks
                    if prev_duration >= overlap_duration: # Cuando la suma de la duración de los chunks sea mayor o igual al overlap, se ha encontrado el inicio del overlap
                        chunk_start = prev_el["start"] # Cuando lo encuentra, se le asigna el inicio del siguiente chunk
                        # chunk_text = " ".join([elem["text"] for elem in data[idx-prev_id:idx+1]]) + " "
                        chunk_text = " ".join([elem["content"] for elem in data[idx-prev_id:idx+1]]) + " "
                        # En duration hay que sumar la duración de los elementos que se han recorrido hacia atrás
                        # duration_chunk = sum([float(self.__time_to_seconds(elem["end"])) - float(self.__time_to_seconds(elem["start"])) for elem in data[idx-prev_id:idx+1]])
                        duration_chunk = prev_duration
                        break

        # for chunk in chunks:
        #     print("Start:", chunk["start"], "End:", chunk["end"], "Texto:", chunk["text"])
        #     print()
        #     print()
       
        return chunks

    
    def __time_to_seconds(self, time_str):
        time_components = time_str.split(':')
        hours = int(time_components[0])
        minutes = int(time_components[1])
        # seconds, milliseconds = map(float, time_components[2].split(','))
        seconds, milliseconds = map(float, time_components[2].split('.'))
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return total_seconds
    
    def __seconds_to_time(self, timestamp):
        total_seconds = int(timestamp)
        milliseconds = int((timestamp - total_seconds) * 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)

    def __generate_documents(self, data):
        splits = []
        for el in data:
            metadata = {}
            metadata["filename"] = el["filename"]
            metadata["start"] = el["start"]
            metadata["end"] = el["end"]
            doc =  Document(page_content=el["text"], metadata=metadata)
            splits.append(doc)
        return splits
    
    # def __generate_vectors(self, documents):
    #     vectordb = Chroma.from_documents(
    #         documents=documents,
    #         embedding=self.embeddings,
    #         persist_directory=self.output_path
    #     )
    #     return vectordb
    
    def __generate_vectors(self, documents):
        if (self.index_id is None) or (self.endpoint_id is None):
            index, index_endpoint = self.create_google_vectordb()
            index_id = index.resource_name.split('/')[-1]
            endpoint_id = index_endpoint.resource_name.split('/')[-1]

        vectordb = MatchingEngine.from_components(
            project_id=self.project_id,
            region=self.region,
            gcs_bucket_name=self.bucket_name,
            embedding=self.embeddings,
            index_id=index_id,
            endpoint_id=endpoint_id,
        )

        vectordb.add_documents(documents=documents)

        return vectordb
    
    def create_google_vectordb(self):
        aiplatform.init(project=self.project_id, location=self.region)

        index = aiplatform.MatchingEngineIndex.create_brute_force_index(
            display_name=f'{self.new_index_name}-index',
            contents_delta_uri=self.bucket_name,
            dimensions=768,
            sync=True,
        )

        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name = f"{self.new_index_name}-endpoint",
            public_endpoint_enabled = True
        )
        deployed_index_name = self.new_index_name.replace('-', '_') # Para este parámetro, el nombre no puede incluir '-', pero sí '_'
        index_endpoint.deploy_index(
            index=index, deployed_index_id=f'{deployed_index_name}_deployed_index'
        )

        return index, index_endpoint


    def __delete_db(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)



if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config_file_path', type=str, default="config/config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    config = Config(args.config_file_path)

    # TASK
    input_path = config.get_variable("3_vectordb.input_path")
    chunk_duration = config.get_variable("3_vectordb.chunk_duration")
    overlap_duration = config.get_variable("3_vectordb.overlap_duration")
    new_index_name = config.get_variable("3_vectordb.new_index_name")
    llm_model_name = config.get_variable("3_vectordb.llm_model_name")
    embeddings_model_name = config.get_variable("3_vectordb.embeddings_model_name")
    # output_path = config.get_variable("3_vectordb.output_path")
    # reset_db = config.get_variable("3_vectordb.reset_db")

    load_dotenv()
    PROJECT_ID = os.getenv('PROJECT_ID')
    REGION = os.getenv('REGION')
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    INDEX_ID = os.getenv('INDEX_ID')
    ENDPOINT_ID = os.getenv('ENDPOINT_ID')

    # generator = VectorDBGenerator(output_path, reset_db)
    generator = VectorDBGenerator(PROJECT_ID, REGION, BUCKET_NAME, llm_model_name, embeddings_model_name, new_index_name, INDEX_ID, ENDPOINT_ID)
    if os.path.isdir(input_path):
        for file_name in tqdm(os.listdir(input_path), desc="Processing files"):
            if file_name.endswith(".json"):
                file_path = os.path.join(input_path, file_name)
                generator.generate_vectordb(file_path, chunk_duration, overlap_duration)
    else:
        generator.generate_vectordb(input_path, chunk_duration, overlap_duration)
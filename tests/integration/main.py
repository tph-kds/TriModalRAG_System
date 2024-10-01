import os
from pathlib import Path
import sys

from src.trim_rag.logger import logger  
from src.trim_rag.exception import MyException

from tests.integration import (data_ingestion, 
                               data_processing, 
                               data_embeddings,
                               push_drant_db,
                               data_retriever,
                               data_generation,
                               )
from tests.integration.inference import data_inference
from src.trim_rag.utils import save_tensor,  save_list, load_list, load_tensor

from src.config_params import ROOT_PROJECT_DIR
from langchain.chains.sequential import SequentialChain
from langchain_core.runnables import Runnable


class UpperCaseRunnable(Runnable):
    def invoke(self, input: str) -> str:
        # Converts input to uppercase
        return input.upper()
    
def main( image_url=None, 
         video_url=None, 
         question_str=None,
         query=None
         ):
    
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING ALL SCENARIOS >>>>>>>>")
        logger.log_message("info", "All scenarios started.")

        # inform data  ingestion stage of the pipeline
        # data_ingestion()

        # # inform data  processing stage of the pipeline
        # textprocessing, title_files,  imageprocessing, audioprocessing = data_processing()
        # print(audioprocessing)
        # # inform data  embedding stage of the pipeline
        # text_embeddings, image_embeddings, audio_embeddings = data_embeddings(textprocessing, imageprocessing, audioprocessing)
        # save_list(title_files, ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "title_files.txt"))

        # # save all embeddings data
        # path = ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings")
        # print(path)
        # save_tensor(text_embeddings[0], path / ( "text/" + "text_embeddings.pt"))
        # save_tensor(image_embeddings, path / ("image/" + "image_embeddings.pt"))
        # save_tensor(audio_embeddings, path / ("audio/" + "audio_embeddings.pt"))

        # # push all embeddings data to drant db
        # push_drant_db(text_embeddings, title_files, image_embeddings, audio_embeddings)

        # title_files = load_list(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "title_files.txt"))
        # text_embeddings = load_tensor(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings/text/text_embeddings.pt"))
        # image_embeddings = load_tensor(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings/image/image_embeddings.pt"))
        # audio_embeddings = load_tensor(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings/audio/audio_embeddings.pt"))
        # text_embed_query, image_embed_query, audio_embed_query = data_inference(query, image_url, video_url) 
        # inform data  retrieval stage of the pipeline
        # print(text_embed_query.tolist()[0])
        # _, _,  retriever = data_retriever(text_embed_query, image_embed_query, audio_embed_query)
        # print(retriever)
        retriever = ["hello", "world", "how", "are", "you", "today", "My", "name", "is", "John", "Doe"]
        # inform data  generation stage of the pipeline
        rag_chain, metadata = data_generation(retriever, 
                                    image_url,
                                    video_url,
                                    question_str,
                                    query
                                    )
        print(rag_chain.invoke(metadata))
        
        logger.log_message("info", "All scenarios completed successfully.")
        logger.log_message("info", "<<<<<<<<   END ALL SCENARIOS   >>>>>>>>")
        logger.log_message("info", "")

    except Exception as e:
        logger.log_message("warning", "Failed to run all scenarios: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run all scenarios: " + str(e),
            error_details = sys,
        )
        print(my_exception)

if __name__ == "__main__":
    question_str = "Yagi's devastation and aftermath: A recap of Vietnam's biggest disaster in decades\
            Typhoon Yagi, the most powerful storm to hit Vietnam in 30 years, unleashed devastating floods and landslides across northern provinces, leaving widespread destruction in its wake as shattered communities struggle to recover and rebuild."
    query = "Does Typhoon Yagi have damages in Vietnam country and what is the damage?"
    main(None, None, question_str, query)
    


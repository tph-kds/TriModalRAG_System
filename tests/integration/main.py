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
                               data_generation
                               )

from src.trim_rag.utils import save_tensor,  save_list, load_list, load_tensor

from src.config_params import ROOT_PROJECT_DIR

def main( image_url=None, 
         video_url=None, 
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

        # # inform data  embedding stage of the pipeline
        # text_embeddings, image_embeddings, audio_embeddings = data_embeddings(textprocessing, imageprocessing, audioprocessing)

        # save_list(title_files, ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "title_files.txt"))
        title_files = load_list(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "title_files.txt"))
        text_embeddings = load_tensor(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings/text/text_embeddings.pt"))
        image_embeddings = load_tensor(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings/image/image_embeddings.pt"))
        audio_embeddings = load_tensor(ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings/audio/audio_embeddings.pt"))

        # save all embeddings data
        path = ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings")
        print(path)
        save_tensor(text_embeddings, path / ( "text/" + "text_embeddings.pt"))
        save_tensor(image_embeddings, path / ("image/" + "image_embeddings.pt"))
        save_tensor(audio_embeddings, path / ("audio/" + "audio_embeddings.pt"))

        # push all embeddings data to drant db
        push_drant_db(text_embeddings, title_files, image_embeddings, audio_embeddings)

        # # inform data  retrieval stage of the pipeline
        # retriever = data_retriever()

        # # inform data  generation stage of the pipeline
        # rag_chain = data_generation(retriever, 
        #                             image_url,
        #                             video_url,
        #                             query
        #                             )
        # print(rag_chain.invoke({"query": query}))
        
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
    query = "What weather are you looking for?"
    main("link_image.png", "link_audio.mp3", query)
    


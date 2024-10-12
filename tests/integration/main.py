import os
from pathlib import Path
import sys

from src.trim_rag.logger import logger  
from src.trim_rag.exception import MyException
from src.trim_rag.utils import (
    save_tensor,  
    save_list
)
from src.trim_rag.utils import (
    convert_qdrantdata_tokens, 
    convert_qdrantdata_desc
)
from src.trim_rag.config import ConfiguarationManager

from tests.integration import (
    data_ingestion, 
    data_processing, 
    data_embeddings,
    push_drant_db,
    data_retriever,
    data_generation,
)
from src.config_params import ROOT_PROJECT_DIR

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
        
        config_manager = ConfiguarationManager()
        embed_config = config_manager.get_data_embedding_arguments_config()


        # inform data  ingestion stage of the pipeline
        # data_ingestion()

        # inform data  processing stage of the pipeline
        textprocessing, title_files,  imageprocessing, audioprocessing = data_processing()

        # inform data  embedding stage of the pipeline
        text_embeddings, image_embeddings, audio_embeddings = data_embeddings(textprocessing, imageprocessing, audioprocessing)
        save_list(title_files, ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "title_files.txt"))

        # save all embeddings data
        path = ROOT_PROJECT_DIR /  ("src/"  "artifacts/" + "data/" + "embeddings")
        print(path)
        save_tensor(text_embeddings[0], path / ( "text/" + "text_embeddings.pt"))
        save_tensor(image_embeddings, path / ("image/" + "image_embeddings.pt"))
        save_tensor(audio_embeddings, path / ("audio/" + "audio_embeddings.pt"))

        # push all embeddings data to drant db
        push_drant_db(text_embeddings, image_embeddings, audio_embeddings)

        main_retriever, retriever_text, retriever_image, retriever_audio = data_retriever(text_embeddings[0], image_embeddings, audio_embeddings)

        lir_retriever = convert_qdrantdata_tokens(config=embed_config.text_data, 
                                                  inputs=main_retriever
                                                  )
        lir_retriever_image = convert_qdrantdata_desc(inputs=retriever_image)
        lir_retriever_audio = convert_qdrantdata_desc(inputs=retriever_audio)


        # inform data  generation stage of the pipeline
        rag_chain, metadata = data_generation(lir_retriever, 
                                    lir_retriever_image,
                                    lir_retriever_audio,
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
    query = "Let me know about a thunderstorm from plenty of attached information?"
    text = ROOT_PROJECT_DIR /  ("data/test/file.pdf")
    image = ROOT_PROJECT_DIR / ("data/test/images.jpg")
    audio = ROOT_PROJECT_DIR / ("data/test/audio_lighting.mp3")
    
    main(text, image, audio, query)

    


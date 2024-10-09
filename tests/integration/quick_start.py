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
    
def main( question_str=None,
         image_url=None, 
         video_url=None, 
         query=None
         ):
    
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING A QUICK START SCENARIO >>>>>>>>")
        logger.log_message("info", "A quick start scenarios started.")

        text_embedding, image_embedding, audio_embedding = data_inference(str(text), str(image), str(audio))
        main_retriever, retriever_text, retriever_image, retriever_audio = data_retriever(text_embedding[0], image_embedding, audio_embedding)
        print(main_retriever)
        print("Hung")
        print(retriever_text)
        print("Hung")
        print(retriever_image)
        print("Hung")
        print(retriever_audio)
        # retriever = ["hello", "world", "how", "are", "you", "today", "My", "name", "is", "John", "Doe"]
        # inform data  generation stage of the pipeline
        # rag_chain, metadata = data_generation(retriever, 
        #                             image_url,
        #                             video_url,
        #                             question_str,
        #                             query
        #                             )
        # print(rag_chain.invoke(metadata))
        
        logger.log_message("info", "A quick start scenarios completed successfully.")
        logger.log_message("info", "<<<<<<<<   END A QUICK START SCENARIOS   >>>>>>>>")
        logger.log_message("info", "")

    except Exception as e:
        logger.log_message("warning", "Failed to run a quick start scenarios: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run a quick start scenarios: " + str(e),
            error_details = sys,
        )
        print(my_exception)

if __name__ == "__main__":
    # question_str = "Yagi's devastation and aftermath: A recap of Vietnam's biggest disaster in decades\
    #         Typhoon Yagi, the most powerful storm to hit Vietnam in 30 years, unleashed devastating floods and landslides across northern provinces, leaving widespread destruction in its wake as shattered communities struggle to recover and rebuild."
    query = "Does Typhoon Yagi have damages in Vietnam country and what were the consequences?"

    text = ROOT_PROJECT_DIR /  ("data/test/file.pdf")
    image = ROOT_PROJECT_DIR / ("data/test/images.jpg")
    audio = ROOT_PROJECT_DIR / ("data/test/audio_lighting.mp3")

    main(text, image, audio, query)
    


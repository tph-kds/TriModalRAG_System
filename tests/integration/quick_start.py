import os
import sys
from pathlib import Path
from typing import List, Optional

from src.trim_rag.logger import logger  
from src.trim_rag.exception import MyException

from src.trim_rag.config import ConfiguarationManager
from src.config_params import ROOT_PROJECT_DIR
from src.config_params import (
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
    GOOGLE_API_KEY,
    LANGCHAIN_API_KEY
)
from tests.integration import data_inference
from tests.integration import (
    data_retriever,
    data_generation,
)
from src.trim_rag.utils import (
    convert_qdrantdata_tokens, 
    convert_qdrantdata_desc
)

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

        config_manager = ConfiguarationManager()
        embed_config = config_manager.get_data_embedding_arguments_config()

        api_config = {
            "GOOGLE_API_KEY": GOOGLE_API_KEY,
            "LANGCHAIN_API_KEY" : LANGCHAIN_API_KEY,
            "LANGCHAIN_ENDPOINT" : LANGCHAIN_ENDPOINT, 
            "LANGCHAIN_TRACING_V2" : LANGCHAIN_TRACING_V2, 
            "LANGCHAIN_PROJECT" : LANGCHAIN_PROJECT,
        }
        llm_config = {
            
            "model_name" : "gemini-1.5-flash-001",
            "temperature" : 0,
            "max_tokens" : 128,
            "max_retries" : 6,
            "stop" : None,
        }
        
        text_embedding, image_embedding, audio_embedding = data_inference(
            text = str(question_str), 
            image = str(image_url), 
            audio = str(video_url)
        )
        main_retriever, retriever_text, retriever_image, retriever_audio = data_retriever(text_embedding[0], image_embedding, audio_embedding)

        # retriever_text has been synthesized inside the lir_retriever
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
                                    query,
                                    api_config,
                                    llm_config
                                    )
        print(rag_chain.invoke(metadata))
        
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
    query = "Let me know about a thunderstorm from plenty of attached information?"

    text = ROOT_PROJECT_DIR /  ("data/test/file.pdf")
    image = ROOT_PROJECT_DIR / ("data/test/images.jpg")
    audio = ROOT_PROJECT_DIR / ("data/test/audio_lighting.mp3")

    main(text, image, audio, query)
    


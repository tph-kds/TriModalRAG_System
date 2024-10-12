import sys
from PyPDF2 import PdfReader
from typing import Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import InferencePipeline
from src.config_params import ROOT_PROJECT_DIR


def data_inference(text: Optional[str], 
                   image: Optional[str], 
                   audio: Optional[str]): 

    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA EMBEDDING FOR SEARCHING DATA ON QDRANT DB >>>>>>>>")
        logger.log_message("info", "Data Embedding pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_multimodal_embedding_arguments_config()
        
        config_embed = config_manager.get_data_embedding_arguments_config()

        config_process = config_manager.get_data_processing_arguments_config()

        data_trans = InferencePipeline(config=config, 
                                        config_embedding=config_embed,
                                        config_processing=config_process)
        
        text_embeddings, image_embeddings, audio_embeddings = data_trans.run_inference(
            text=text, 
            image=image, 
            audio=audio, 
            )


        logger.log_message("info", "Data Embedding pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA EMBEDDING  FOR SEARCHING DATA ON QDRANT DB  >>>>>>>>")
        logger.log_message("info", "")

        return text_embeddings, image_embeddings, audio_embeddings

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Embedding pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Embedding pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)


if __name__ == "__main__":

    text = ROOT_PROJECT_DIR /  ("data/test/file.pdf")
    image = ROOT_PROJECT_DIR / ("data/test/images.jpg")
    audio = ROOT_PROJECT_DIR / ("data/test/audio_lighting.mp3")

    text_embedding, image_embedding, audio_embedding = data_inference(str(text), str(image), str(audio))
    print("Text Embedding: ", text_embedding)
    print("Image Embedding: ", image_embedding)
    print("Audio Embedding: ", audio_embedding)


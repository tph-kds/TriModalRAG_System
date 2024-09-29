import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.inference import DataInference

def data_inference(text, image, audio, type_embedding="shared"):

    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA EMBEDDING FOR SEARCHING DATA ON QDRANT DB >>>>>>>>")
        logger.log_message("info", "Data Embedding pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_multimodal_embedding_arguments_config()
        
        config_embed = config_manager.get_data_embedding_arguments_config()

        data_trans = DataInference(config, config_embed)
        text_embeddings, image_embeddings, audio_embeddings = data_trans._embed_data(
            text_data=text, 
            image_data=image, 
            audio_data=audio, 
            type_embed=type_embedding
            )


        # print(config)
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


# if __name__ == "__main__":
#     data_embedding()



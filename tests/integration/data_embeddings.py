import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataEmbeddingPipeline

def data_embeddings(text, image, audio, type_embedding="shared"):

    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA EMBEDDING SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Embedding pipeline started.")
        config_manager = ConfiguarationManager()
        print("HUng")
        config = config_manager.get_multimodal_embedding_arguments_config()
        print("HUng1")
        
        config_embed = config_manager.get_data_embedding_arguments_config()
        print("HUn2")
        
        pipeline = DataEmbeddingPipeline(config, config_embed)
        text_embeddings, image_embeddings, audio_embeddings = pipeline.run_data_embedding_pipeline(
            text=text, 
            image=image, 
            audio=audio, 
            type_embedding=type_embedding
            )
        # text_embeddings, image_embeddings, audio_embeddings = pipeline.run_data_embedding_pipeline(type="shared")

        # print(text_embeddings)


        # print(config)
        logger.log_message("info", "Data Embedding pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA EMBEDDING SCENARIO   >>>>>>>>")
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



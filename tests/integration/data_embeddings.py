import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataEmbeddingPipeline

if __name__ == "__main__":
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA EMBEDDING SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Embedding pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_multimodal_embedding_arguments_config()
        config_embed = config_manager.get_data_embedding_arguments_config()
        pipeline = DataEmbeddingPipeline(config, config_embed)
        text_embeddings = pipeline.run_data_embedding_pipeline(type="shared")
        # text_embeddings, image_embeddings, audio_embeddings = pipeline.run_data_embedding_pipeline(type="shared")

        print(text_embeddings)


        # print(config)
        logger.log_message("info", "Data Embedding pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA EMBEDDING SCENARIO   >>>>>>>>")
        logger.log_message("info", "")

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Embedding pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Embedding pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)




import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import QdrantVectorDBPipeline
from src.config_params import QDRANT_DB_URL, QDRANT_API_KEY


def push_drant_db(text_embeds, image_embeds, audio_embeds):
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING TO PUSH QDRANT DATABASE SCENARIO >>>>>>>>")
        logger.log_message("info", "PUSHING QDRANT DATABASE pipeline started.")
        manager_config = ConfiguarationManager()
        config = manager_config.get_qdrant_vectordb_arguments_config()
        prepare_db_config = manager_config.get_init_embedding_qdrant_arguments_config()
        pipeline = QdrantVectorDBPipeline(config, 
                                          prepare_db_config, 
                                          text_embeds, 
                                          image_embeds, 
                                          audio_embeds,
                                          QDRANT_API_KEY=QDRANT_API_KEY,
                                          QDRANT_DB_URL=QDRANT_DB_URL)
        ## delete all vectors from QDRANT DATABASE


        ## run upload  the new vector and index embedding data pipeline to QDRANT DATABASE
        pipeline.run_qdrant_vector_db_pipeline()

        # print(config)
        logger.log_message("info", "PUSH QDRANT DATABASE pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END PUSHING QDRANT DATABASE SCENARIO   >>>>>>>>")
        logger.log_message("info", "")
        return None

    except Exception as e:
        logger.log_message("warning", "Failed to run to push QDRANT DATABASE pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run to push QDRANT DATABASE pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)


if __name__ == "__main__":
    push_drant_db()





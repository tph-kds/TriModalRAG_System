import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import GenerationPipeline


def data_generation(retriever=None, 
                    image_url=None, 
                    video_url=None, 
                    query=None
                    ):
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA GENERATION SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Generation pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_generation_config()
        pipeline = GenerationPipeline(config)

        rag_chain = pipeline.run_generation_pipeline(
            retriever=retriever,
            image_url=image_url,
            video_url=video_url,
            query=query
        )
        # print(config)
        logger.log_message("info", "Data Generation pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA GENERATION SCENARIO   >>>>>>>>")
        logger.log_message("info", "")
        return rag_chain

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Generation pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Generation pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)







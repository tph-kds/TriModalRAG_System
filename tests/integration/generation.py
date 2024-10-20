import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import GenerationPipeline



def data_generation(retriever=None, 
                    image_str=None, 
                    video_str=None, 
                    question_str=None,
                    query=None,
                    api_config=None,
                    llm_config=None
                    ):
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA GENERATION SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Generation pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_multimodal_generation_config()
        models_config = config_manager.get_model_arguments_config()
        embeddings_config = config_manager.get_data_embedding_arguments_config()

        pipeline = GenerationPipeline(config, models_config, embeddings_config, api_config, llm_config)

        rag_chain = pipeline.run_generation_pipeline(
            retriever=retriever,
            image_url=image_str,
            video_url=video_str,
            question_str=question_str,
            query_str=query
        )

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







import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataRetrievalPipeline


def data_retriever():
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA RETRIEVER SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Retriever pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_retrieval_config()
        pipeline = DataRetrievalPipeline(config)
        
        chain_retrieval = pipeline.run_data_retrieval_pipeline()

        logger.log_message("info", "Data Retriever pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA RETRIEVER SCENARIO   >>>>>>>>")
        logger.log_message("info", "")
        return chain_retrieval

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Retriever pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Retriever pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)


if __name__ == "__main__":
    retriever = data_retriever()
    print(retriever)




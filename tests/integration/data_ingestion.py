import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataIngestionPipeline


def data_ingestion():
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA INGESTION SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Ingestion pipeline started.")
        config = ConfiguarationManager().get_data_ingestion_arguments_config()
        pipeline = DataIngestionPipeline(config)
        pipeline.run_data_ingestion_pipeline()

        logger.log_message("info", "Data Ingestion pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA INGESTION SCENARIO   >>>>>>>>")
        logger.log_message("info", "")

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Ingestion pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Ingestion pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)




if __name__ == "__main__":
    data_ingestion()
import sys
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataTransformPipeline


def data_processing():
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA TRANSFORMATION SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Transformation pipeline started.")
        config = ConfiguarationManager().get_data_processing_arguments_config()
        pipeline = DataTransformPipeline(config)
        textprocessing, title_files, imageprocessing, audioprocessing = pipeline.run_data_processing_pipeline()


        logger.log_message("info", "Data Transformation pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA TRANSFORMATION SCENARIO   >>>>>>>>")
        logger.log_message("info", "")
        return textprocessing, title_files, imageprocessing, audioprocessing

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Transformation pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Transformation pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)


if __name__ == "__main__":
    textprocessing, imageprocessing, audioprocessing = data_processing()
    print(textprocessing)
    print(imageprocessing)
    print(audioprocessing)




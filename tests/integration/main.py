import os
import sys

from src.trim_rag.logger import logger  
from src.trim_rag.exception import MyException

from tests.integration import (data_ingestion, 
                               data_processing, 
                               data_embeddings,
                               push_drant_db
                               )

def main():
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING ALL SCENARIOS >>>>>>>>")
        logger.log_message("info", "All scenarios started.")
        # inform data  ingestion stage of the pipeline
        # data_ingestion()

        # inform data  processing stage of the pipeline
        textprocessing, imageprocessing, audioprocessing = data_processing()

        # inform data  embedding stage of the pipeline
        text_embeddings, image_embeddings, audio_embeddings = data_embeddings(textprocessing, imageprocessing, audioprocessing)

        # push all embeddings data to drant db
        push_drant_db(text_embeddings, image_embeddings, audio_embeddings)


        
        logger.log_message("info", "All scenarios completed successfully.")
        logger.log_message("info", "<<<<<<<<   END ALL SCENARIOS   >>>>>>>>")
        logger.log_message("info", "")

    except Exception as e:
        logger.log_message("warning", "Failed to run all scenarios: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run all scenarios: " + str(e),
            error_details = sys,
        )
        print(my_exception)

if __name__ == "__main__":
    main()

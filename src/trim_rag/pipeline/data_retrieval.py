import os
import sys

from langchain.chains import SimpleChain

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.retrieval import TriModalRetrieval

class DataRetrievalPipeline:

    def __init__(self, ) -> None:
        super(DataRetrievalPipeline, self).__init__()

        # self.config = config


    def run_data_retrieval_pipeline(self) -> SimpleChain:
        try: 
            logger.log_message("info", "Running data retrieval pipeline...")
            tri_modal_retrieval = TriModalRetrieval()
            chain_retrieval = tri_modal_retrieval.trimodal_retrieval()
            logger.log_message("info", "Data retrieval pipeline completed")

            return chain_retrieval

        except Exception as e:
            logger.log_message("warning", "Failed to run data retrieval pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data retrieval pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

class DataRetrievalPipeline:

    def __init__(self, config) -> None:
        super(DataRetrievalPipeline, self).__init__()

        self.config = config


    def run_data_retrieval_pipeline(self) -> None:
        pass
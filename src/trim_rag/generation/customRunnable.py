import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from langchain_core.runnables.base import Runnable


class StringFormatterRunnable(Runnable):
    def __init__(self, prefix="Result: "):
        self.prefix = prefix

    def invoke(self, input_data):
        # Process input data and format it as a string
        result = self.prefix + str(input_data)

        return result

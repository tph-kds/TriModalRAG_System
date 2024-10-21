import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

# from src.trim_rag.config import PineconeVectorDBArgumentsConfig


class PineconeVectorDB:
    def __init__(self, config) -> None:
        super(PineconeVectorDB, self).__init__()

        self.config = config

    def pinecone_setting(self) -> None:
        pass

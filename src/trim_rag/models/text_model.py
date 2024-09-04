import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import TextModelArgumentsConfig

class TextModel:
    def __init__(self, config: TextModelArgumentsConfig) -> None:
        super(TextModel, self).__init__()

        self.config = config

    def text_model(self) -> None:
        pass
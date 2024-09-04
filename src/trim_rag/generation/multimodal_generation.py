import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import MultimodalGenerationArgumentsConfig

class MultimodalGeneration:
    def __init__(self, config: MultimodalGenerationArgumentsConfig) -> None:
        super(MultimodalGeneration, self).__init__()

        self.config = config

    def multimodal_generation(self) -> None:
        pass

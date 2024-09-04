import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import ImageModelArgumentsConfig

class ImageModel:
    def __init__(self, config: ImageModelArgumentsConfig) -> None:
        super(ImageModel, self).__init__()

        self.config = config

    def image_model(self) -> None:
        pass
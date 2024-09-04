import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import AudioModelArgumentsConfig

class AudioModel:
    def __init__(self, config: AudioModelArgumentsConfig) -> None:
        super(AudioModel, self).__init__()

        self.config = config

    def audio_model(self) -> None:
        pass
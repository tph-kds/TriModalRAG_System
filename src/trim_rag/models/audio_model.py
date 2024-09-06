import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import AudioModelArgumentsConfig
from src.trim_rag.embedding import AudioEmbedding

class AudioModel:
    def __init__(self, 
                 config: AudioModelArgumentsConfig,
                 audio_embedding: AudioEmbedding) -> None:
        super(AudioModel, self).__init__()

        self.config = config
        self.name_of_model = self.config.name_of_model
        self.audio_embedding = audio_embedding

    def audio_model(self) -> None:
        try:
            logger.log_message("info", "Initializing audio model...")
            audio_embedding = AudioEmbedding(self.audio_embedding)
            models = audio_embedding._get_model()
            logger.log_message("info", "audio model initialized successfully.")
            return models
        
        except Exception as e:
            logger.log_message("warning", "Failed to initialize audio model: " + str(e))
            my_exception = MyException(
                error_message = "Failed to initialize audio model: " + str(e),
                error_details = sys,
            )
            print(my_exception)
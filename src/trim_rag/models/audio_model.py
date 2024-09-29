import os
import sys
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import (
    AudioModelArgumentsConfig,
    AudioEmbeddingArgumentsConfig,
)
from src.trim_rag.embedding import AudioEmbedding
from transformers import Wav2Vec2Model

class AudioModel:
    def __init__(self, 
                 config: AudioModelArgumentsConfig,
                 embed_config: AudioEmbeddingArgumentsConfig) -> None:
        super(AudioModel, self).__init__()

        self.config = config
        self.name_of_model = self.config.name_of_model
        self.audio_embedding = AudioEmbedding(embed_config)

    def audio_model(self) -> Optional[Wav2Vec2Model]:
        try:
            logger.log_message("info", "Initializing audio model...")
            models = self.audio_embedding._get_model()
            logger.log_message("info", "audio model initialized successfully.")
            return models
        
        except Exception as e:
            logger.log_message("warning", "Failed to initialize audio model: " + str(e))
            my_exception = MyException(
                error_message = "Failed to initialize audio model: " + str(e),
                error_details = sys,
            )
            print(my_exception)
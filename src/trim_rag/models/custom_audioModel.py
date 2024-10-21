import os
import sys
from typing import Optional

import torch

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import AudioEmbeddingArgumentsConfig, AudioModelArgumentsConfig
from src.trim_rag.models.audio_model import AudioModel
from transformers import BertModel, BertTokenizer
from langchain_core.runnables import Runnable
from transformers import Wav2Vec2Model


class AudioModelRunnable(Runnable):
    def __init__(
        self,
        config: AudioModelArgumentsConfig,
        embed_config: AudioEmbeddingArgumentsConfig,
    ) -> None:
        super(AudioModelRunnable, self).__init__()

        # self.model = model
        # self.tokenizer = tokenizer
        self.config = config
        self.name_of_model = self.config.name_of_model
        self.audio_model_main = AudioModel(config=config, embed_config=embed_config)

    def invoke(self, input: str) -> torch.Tensor:
        try:
            logger.log_message("info", "Initializing audio model...")
            embed_audio = self.audio_model_main.embedding_audio(input)
            logger.log_message("info", "audio model initialized successfully.")
            return embed_audio

        except Exception as e:
            logger.log_message("warning", "Failed to initialize audio model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize audio model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def audio_model_runnable(self) -> Optional[Wav2Vec2Model]:
        try:
            logger.log_message("info", "Initializing audio model...")
            models = self.audio_model_main.audio_model()
            logger.log_message("info", "audio model initialized successfully.")
            return models

        except Exception as e:
            logger.log_message("warning", "Failed to initialize audio model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize audio model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

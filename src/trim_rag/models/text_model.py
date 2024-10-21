import os
import sys
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import TextModelArgumentsConfig, TextEmbeddingArgumentsConfig
from src.trim_rag.embedding import TextEmbedding
from transformers import BertModel


class TextModel:
    def __init__(
        self,
        config: TextModelArgumentsConfig,
        embed_config: TextEmbeddingArgumentsConfig,
    ) -> None:
        super(TextModel, self).__init__()

        self.config = config
        self.name_of_model = self.config.name_of_model
        self.text_embedding = TextEmbedding(embed_config)

    def text_model(self) -> Optional[BertModel]:
        try:
            logger.log_message("info", "Initializing text model...")
            models = self.text_embedding._get_model()
            logger.log_message("info", "Text model initialized successfully.")
            return models

        except Exception as e:
            logger.log_message("warning", "Failed to initialize text model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize text model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

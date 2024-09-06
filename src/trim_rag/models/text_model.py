import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import TextModelArgumentsConfig
from src.trim_rag.embedding import TextEmbedding

class TextModel:
    def __init__(self, 
                 config: TextModelArgumentsConfig,
                 text_embedding: TextEmbedding) -> None:
        super(TextModel, self).__init__()

        self.config = config
        self.name_of_model = self.config.name_of_model
        self.text_embedding = text_embedding

    def text_model(self) -> None:
        try:
            logger.log_message("info", "Initializing text model...")
            text_embedding = TextEmbedding(self.text_embedding)
            models = text_embedding._get_model()
            logger.log_message("info", "Text model initialized successfully.")
            return models
        
        except Exception as e:
            logger.log_message("warning", "Failed to initialize text model: " + str(e))
            my_exception = MyException(
                error_message = "Failed to initialize text model: " + str(e),
                error_details = sys,
            )
            print(my_exception)
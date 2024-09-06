import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import ImageModelArgumentsConfig
from src.trim_rag.embedding import ImageEmbedding

class ImageModel:
    def __init__(self, 
                 config: ImageModelArgumentsConfig,
                 image_embedding: ImageEmbedding) -> None:
        super(ImageModel, self).__init__()

        self.config = config
        self.name_of_model = self.config.name_of_model
        self.image_embedding = image_embedding

    def image_model(self) -> None:
        try:
            logger.log_message("info", "Initializing image model...")
            image_embedding = ImageEmbedding(self.image_embedding)
            models = image_embedding._get_model()
            logger.log_message("info", "image model initialized successfully.")
            return models
        
        except Exception as e:
            logger.log_message("warning", "Failed to initialize image model: " + str(e))
            my_exception = MyException(
                error_message = "Failed to initialize image model: " + str(e),
                error_details = sys,
            )
            print(my_exception)
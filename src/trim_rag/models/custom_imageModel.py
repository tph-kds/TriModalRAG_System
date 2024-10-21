import os
import sys
from typing import Optional

import torch

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ImageEmbeddingArgumentsConfig, ImageModelArgumentsConfig
from src.trim_rag.models.image_model import ImageModel
from transformers import CLIPModel
from langchain_core.runnables import Runnable

from src.trim_rag.embedding import ImageEmbedding


class ImageModelRunnable(Runnable):
    def __init__(
        self,
        config: ImageModelArgumentsConfig,
        embed_config: ImageEmbeddingArgumentsConfig,
    ) -> None:
        super(ImageModelRunnable, self).__init__()

        # self.model = model
        # self.tokenizer = tokenizer
        self.config = config
        self.name_of_model = self.config.name_of_model
        self.image_model_main = ImageModel(config=config, embed_config=embed_config)

    def invoke(self, input: str) -> torch.Tensor:
        try:
            logger.log_message("info", "Initializing image model...")
            embed_image = self.image_model_main.embedding_image(input)

            logger.log_message("info", "image model initialized successfully.")
            return embed_image

        except Exception as e:
            logger.log_message("warning", "Failed to initialize image model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize image model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def image_model_runnable(self) -> Optional[CLIPModel]:
        try:
            logger.log_message("info", "Initializing image model...")
            models = self.image_model_main.image_model()
            logger.log_message("info", "image model initialized successfully.")
            return models

        except Exception as e:
            logger.log_message("warning", "Failed to initialize image model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize image model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

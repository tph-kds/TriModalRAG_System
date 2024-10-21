import os
import sys
from typing import Optional

import torch

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import TextModelArgumentsConfig
from src.trim_rag.models.text_model import TextModel
from transformers import BertModel, BertTokenizer
from langchain_core.runnables import Runnable

from src.trim_rag.config import TextEmbeddingArgumentsConfig
from src.trim_rag.embedding import TextEmbedding


class TextModelRunnable(Runnable):
    def __init__(
        self,
        config: TextModelArgumentsConfig,
        embed_config: TextEmbeddingArgumentsConfig,
    ) -> None:
        super(TextModelRunnable, self).__init__()

        # self.model = model
        # self.tokenizer = tokenizer
        self.config = config
        self.name_of_model = self.config.name_of_model
        self.text_model_main = TextModel(config=config, embed_config=embed_config)

    def __call__(self, input: str) -> torch.Tensor:
        return self.invoke(input)

    def invoke(self, input: str) -> torch.Tensor:
        try:
            logger.log_message("info", "Initializing text model and tokenizer ...")
            embed_text, _ = self.text_model_main.embedding_text(input)
            logger.log_message("info", "Embedding text successfully.")
            return embed_text

        except Exception as e:
            logger.log_message("warning", "Failed to initialize text model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize text model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def text_model_runnable(self) -> Optional[BertModel]:
        try:
            logger.log_message("info", "Initializing text model...")
            models = self.text_model_main.text_model()
            logger.log_message("info", "Text model initialized successfully.")
            return models

        except Exception as e:
            logger.log_message("warning", "Failed to initialize text model: " + str(e))
            my_exception = MyException(
                error_message="Failed to initialize text model: " + str(e),
                error_details=sys,
            )
            print(my_exception)

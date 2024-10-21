import os
import sys
from typing import List, Optional, Tuple, Union

import torch

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import (
    DataTransformationArgumentsConfig,
    MultimodalEmbeddingArgumentsConfig,
)
from src.trim_rag.pipeline import DataEmbeddingPipeline

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


class DataInference(Embeddings):
    def __init__(
        self,
        multi_embed_config: MultimodalEmbeddingArgumentsConfig,
        embed_config: DataTransformationArgumentsConfig,
    ) -> None:
        self.embed_config = embed_config
        self.multi_embed_config = multi_embed_config
        self.embed = DataEmbeddingPipeline(multi_embed_config, embed_config)
        # self.data = self.config.data

        # self.text_data = self.data.text_data
        # self.image_data = self.data.image_data
        # self.audio_data = self.data.audio_data

        # self.size_text = self.text_data.size_text
        # self.size_image = self.image_data.size_image
        # self.size_audio = self.audio_data.size_audio

        # self.collection_text_name = self.text_data.collection_text_name
        # self.collection_image_name = self.image_data.collection_image_name
        # self.collection_audio_name = self.audio_data.collection_audio_name

    def data_inference(self):
        pass

    def _embed_data(
        self,
        text_data: Optional[List[str]],
        image_data: Optional[List[str]],
        audio_data: Optional[List[str]],
        type_embed: Optional[str],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        try:
            logger.log_message("info", "Embedding data...")

            logger.log_message("info", "Data embedding pipeline started.")
            if text_data is not None:
                text_embeds = self.embed.text_embedding(texts=text_data)
            else:
                text_embeds = None
            if image_data is not None:
                image_embeds = self.embed.image_embedding(images=image_data)
            else:
                image_embeds = None
            if audio_data is not None:
                audio_embeds = self.embed.audio_embedding(audios=audio_data)
            else:
                audio_embeds = None

            if type_embed == "all":
                embeddings = self.embed._multimodal_embedding()
                return embeddings

            (
                text_new_embeddings,
                image_new_embeddings,
                audio_new_embeddings,
            ) = self.embed.shared_embedding_space(
                text_embeds[0], image_embeds, audio_embeds
            )

            logger.log_message(
                "info", "Data embedding pipeline completed successfully."
            )
            return text_new_embeddings, image_new_embeddings, audio_new_embeddings

        except Exception as e:
            logger.log_message(
                "warning",
                "Data type not supported. Please select 'text', 'image', or 'audio'",
            )
            my_exception = MyException(
                error_message="Data type not supported. Please select 'text', 'image', or 'audio'",
                error_details=sys,
            )
            print(my_exception)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        pass

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        pass

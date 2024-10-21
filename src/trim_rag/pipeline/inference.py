import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

# from src.trim_rag.config import InferenceArgumentsConfig

import os
import sys
import torch
from torch import nn

from typing import List, Optional, Union, Tuple
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger

from src.trim_rag.config import (
    MultimodalEmbeddingArgumentsConfig,
    EmbeddingArgumentsConfig,
    DataTransformationArgumentsConfig,
)
from src.trim_rag.inference import Inference


class InferencePipeline:
    def __init__(
        self,
        config: MultimodalEmbeddingArgumentsConfig,
        config_embedding: EmbeddingArgumentsConfig,
        config_processing: DataTransformationArgumentsConfig,
    ):
        super(InferencePipeline, self).__init__()

        self.infer = Inference(
            config=config,
            config_embedding=config_embedding,
            config_processing=config_processing,
        )

    def run_inference(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
    ) -> Union[
        Optional[List[float]],
        Optional[List[str]],
        Optional[List[float]],
        Optional[List[float]],
    ]:
        try:
            logger.log_message("info", "Inference pipeline started.")
            (
                text_infer_processing,
                image_infer_processing,
                audio_infer_processing,
            ) = self.infer.run_data_processing_pipeline(
                text_path=text, image_path=image, audio_path=audio
            )

            (
                text_embeddings,
                image_embeddings,
                audio_embeddings,
            ) = self.infer.run_data_embedding_pipeline(
                text_infer_processing, image_infer_processing, audio_infer_processing
            )

            logger.log_message("info", "Inference pipeline completed successfully.")

            return text_embeddings, image_embeddings, audio_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run inference pipeline: " + str(e))
            my_exception = MyException(
                error_message="Failed to run inference pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

import os
import sys
import torch
from typing import List, Optional, Tuple

from langchain.chains.sequential import SequentialChain
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import (
    TrimodalRetrievalPipelineArgumentsConfig,
    QdrantVectorDBArgumentsConfig,
)
from src.trim_rag.retrieval.fusion_mechanism import FusionMechanism
from src.trim_rag.retrieval.text_retrieval import TextRetrieval
from src.trim_rag.retrieval.audio_retrieval import AudioRetrieval
from src.trim_rag.retrieval.image_retrieval import ImageRetrieval
from src.trim_rag.components import QdrantVectorDB


class TriModalRetrieval:
    def __init__(
        self,
        config: TrimodalRetrievalPipelineArgumentsConfig,
        config_qdrant: QdrantVectorDBArgumentsConfig,
        QDRANT_API_KEY: Optional[str] = None,
        QDRANT_DB_URL: Optional[str] = None,
    ) -> None:
        super(TriModalRetrieval, self).__init__()

        self.config = config
        self.fusion_mechanism = self.config.fusion_method
        self.text_retrieval = self.config.trimodal_retrieval.text_retrieval
        self.image_retrieval = self.config.trimodal_retrieval.image_retrieval
        self.audio_retrieval = self.config.trimodal_retrieval.audio_retrieval
        self.config_qdrant = config_qdrant
        self.client = QdrantVectorDB(
            self.config_qdrant, QDRANT_API_KEY, QDRANT_DB_URL
        )._connect_qdrant()

    def _init_func(self):
        try:
            logger.log_message("info", "Initializing trimodal retrieval...")
            fusion_mechanism = FusionMechanism(self.fusion_mechanism)
            tex_retrieval = TextRetrieval(
                self.text_retrieval, self.config_qdrant, self.client
            )
            img_retrieval = ImageRetrieval(
                self.image_retrieval, self.config_qdrant, self.client
            )
            aud_retrieval = AudioRetrieval(
                self.audio_retrieval, self.config_qdrant, self.client
            )

            return fusion_mechanism, tex_retrieval, img_retrieval, aud_retrieval

        except Exception as e:
            logger.log_message("warning", f"Error initializing trimodal retrieval: {e}")
            my_exception = MyException(
                error_message=f"Error initializing trimodal retrieval: {e}",
                error_details=sys,
            )
            print(my_exception)

    def trimodal_retrieval(
        self,
        text_embedding_query: Optional[List[float]] = None,
        image_embedding_query: Optional[List[float]] = None,
        audio_embedding_query: Optional[List[float]] = None,
    ) -> Tuple[SequentialChain, FusionMechanism]:
        try:
            logger.log_message("info", "Starting to retrieve text embeddings...")
            (
                fusion_mechanism,
                tex_retrieval,
                img_retrieval,
                aud_retrieval,
            ) = self._init_func()
            vector_text, vector_image, vector_audio = None, None, None

            if text_embedding_query is not None:
                # Retrieve text embeddings
                print(text_embedding_query.shape)
                text_embedding = tex_retrieval.text_retrieval(text_embedding_query)
                vector_text = text_embedding[
                    0
                ].vector  # get the vector after retrieval from qdrant
                vector_text = torch.tensor(vector_text)  # convert to tensor
                logger.log_message("info", "Retrieved text embeddings successfully")

            else:
                vector_text = None

            if image_embedding_query is not None:
                # Retrieve image embeddings
                image_embedding = img_retrieval.image_retrieval(image_embedding_query)
                vector_image = image_embedding[
                    0
                ].vector  # get the vector after retrieval from qdrant
                vector_image = torch.tensor(vector_image)  # convert to tensor
                logger.log_message("info", "Retrieved image embeddings successfully")

            else:
                vector_image = None
            if audio_embedding_query is not None:
                # Retrieve audio embeddings
                audio_embedding = aud_retrieval.audio_retrieval(audio_embedding_query)
                vector_audio = audio_embedding[
                    0
                ].vector  # get the vector after retrieval from qdrant
                vector_audio = torch.tensor(vector_audio)  # convert to tensor
                logger.log_message("info", "Retrieved audio embeddings successfully")

            else:
                vector_audio = None

            # Fusion
            # print(text_embedding[0].vector)
            # print(text_embedding[0].points.vector)
            # print(vector_text)

            fusion = fusion_mechanism(
                text_results=vector_text,
                image_results=vector_image,
                audio_results=vector_audio,
            )

            # chain = SequentialChain(input_variables=["text", "image", "audio"],
            #                         chains = [tex_retrieval, img_retrieval, aud_retrieval, fusion_mechanism],
            #                         output_variables=["fusion_matrix"],)
            # Change shape of fusion to torch.Size([n_sample, 512])
            fusion = torch.mean(fusion, dim=1)  # =>> torch.Size([3, 512])
            # print(fusion.shape)
            return fusion, text_embedding, image_embedding, audio_embedding

        except Exception as e:
            logger.log_message("warning", f"Error retrieving embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving embeddings: {e}",
                error_details=sys,
            )
            print(my_exception)

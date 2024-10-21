import os
import sys
import torch
from typing import Optional, List
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import (
    QdrantVectorDBArgumentsConfig,
    PrepareDataQdrantArgumentsConfig,
)
from src.trim_rag.components.qdrant_db import QdrantVectorDB
from src.trim_rag.processing import PrepareDataQdrant


class QdrantVectorDBPipeline:
    def __init__(
        self,
        config: QdrantVectorDBArgumentsConfig,
        prepare_config: PrepareDataQdrantArgumentsConfig,
        text_embeddings: Optional[List] = None,
        image_embeddings: Optional[List] = None,
        audio_embeddings: Optional[List] = None,
        QDRANT_API_KEY: Optional[str] = None,
        QDRANT_DB_URL: Optional[str] = None,
    ) -> None:
        super(QdrantVectorDBPipeline, self).__init__()
        self.config = config
        self.prepare_config = prepare_config

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.audio_embeddings = audio_embeddings

        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.QDRANT_DB_URL = QDRANT_DB_URL

        self.image_collection = self.config.image_data.collection_image_name
        self.text_collection = self.config.text_data.collection_text_name
        self.audio_collection = self.config.audio_data.collection_audio_name

        # self.titles = titles
        # self.input_ids = input_ids

    def run_qdrant_vector_db_pipeline(self) -> None:
        try:
            logger.log_message(
                "info", "Running upload embeddings to qdrant pipeline..."
            )
            text_records, image_records, audio_records = self.records_all_to_qdrant()
            qdrant = QdrantVectorDB(
                self.config, self.QDRANT_API_KEY, self.QDRANT_DB_URL
            )
            ### Delete text embeddings from Qdrant
            self.delete_all_from_qdrant(qdrant)

            ### Connect to Qdrant server
            qdrant.qdrant_setting_version_2()

            ### Upload text embeddings to Qdrant
            qdrant._upload_records_v1(self.text_collection, text_records)
            ### Upload image embeddings to Qdrant
            qdrant._upload_records_v1(self.image_collection, image_records)
            ### Upload audio embeddings to Qdrant
            qdrant._upload_records_v1(self.audio_collection, audio_records)

            return None
            # return text_records, image_records, audio_records

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run to upload embeddings pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run to upload embeddings pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    # def _handle_input_ids(self, input: Optional[List[torch.Tensor]]) -> Optional[List[int]]:
    #     try:
    #         # convert input_ids to list, avoid an error about the type mismatch
    #         out_input_ids = convert_tensor_to_list(input)

    #         return out_input_ids

    #     except Exception as e:
    #         logger.log_message("warning", "Failed to handle input_ids: " + str(e))
    #         my_exception = MyException(
    #             error_message = "Failed to handle input_ids: " + str(e),
    #             error_details = sys,
    #         )
    #         print(my_exception)

    def records_all_to_qdrant(self):
        try:
            prepare_data_qdrant = PrepareDataQdrant(
                self.prepare_config,
                self.text_embeddings,
                self.image_embeddings,
                self.audio_embeddings,
            )

            (
                text_records,
                image_records,
                audio_records,
            ) = prepare_data_qdrant.run_prepare_data_qdrant_pipeline()

            return text_records, image_records, audio_records

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run text preparation pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run text preparation pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def delete_all_from_qdrant(self, qdrant: QdrantVectorDB):
        try:
            logger.log_message(
                "info", "Running Reset embeddings from qdrant pipeline..."
            )
            # qdrant = QdrantVectorDB(self.config,
            #                         self.QDRANT_API_KEY,
            #                         self.QDRANT_DB_URL)
            # ### Connect to Qdrant server
            # qdrant.qdrant_setting_version_2()
            ### Delete text embeddings from Qdrant
            qdrant._delete_collection(self.text_collection)
            ### Delete image embeddings from Qdrant
            qdrant._delete_collection(self.image_collection)
            ### Delete audio embeddings from Qdrant
            qdrant._delete_collection(self.audio_collection)

            return None
        except Exception as e:
            logger.log_message(
                "warning", "Failed to run to reset embeddings pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run to reset embeddings pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

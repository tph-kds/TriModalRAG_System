import os
import sys
from typing import List, Optional, Tuple
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.processing import TextQdrantDB, ImageQdrantDB, AudioQdrantDB

from src.trim_rag.config import PrepareDataQdrantArgumentsConfig

class PrepareDataQdrant:
    def __init__(self,
                 config: PrepareDataQdrantArgumentsConfig,
                 text_embeddings: Optional[List] = None,
                 image_embeddings: Optional[List] = None,
                 audio_embeddings: Optional[List] = None,
                 ) -> None:
        super(PrepareDataQdrant, self).__init__()
        self.config = config
        self.text_data = self.config.text_data
        self.image_data = self.config.image_data
        self.audio_data = self.config.audio_data

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.audio_embeddings = audio_embeddings



    def run_prepare_data_qdrant_pipeline(self) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        try:
            logger.log_message("info", "Running text preparation for data embeddings to qdrant pipeline...")
            text_records = self.records_text_to_qdrant()
            image_records = self.records_image_to_qdrant()
            audio_records = self.records_audio_to_qdrant()

            return text_records, image_records, audio_records
        

        except Exception as e:
            logger.log_message("warning", "Failed to run text preparation pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text preparation pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def records_text_to_qdrant(self):
        try:
            textqdrant = TextQdrantDB(self.text_data)
            text_records = textqdrant.create_records(
                processing_embedding=self.text_embeddings,
            )
            return text_records


        except Exception as e:
            logger.log_message("warning", "Failed to run text preparation pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text preparation pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def records_image_to_qdrant(self):
        try:
            imageqdrant = ImageQdrantDB(self.image_data)
            image_records = imageqdrant.create_records(processing_embedding=self.image_embeddings)
            return image_records
        

        except Exception as e:
            logger.log_message("warning", "Failed to run image preparation pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run image preparation pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def records_audio_to_qdrant(self):
        try:
            audioqdrant = AudioQdrantDB(self.audio_data)
            audio_records = audioqdrant.create_records(processing_embedding=self.audio_embeddings)
            return audio_records

        except Exception as e:
            logger.log_message("warning", "Failed to run audio preparation pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run audio preparation pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

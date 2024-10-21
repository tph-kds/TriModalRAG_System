import os
import sys
from typing import List
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.data import TextIngestion, ImageIngestion, AudioIngestion
from src.trim_rag.config import DataIngestionArgumentsConfig
from src.trim_rag.data.images import DuplicateImageProcessing


class DataIngestionPipeline:
    def __init__(self, config: DataIngestionArgumentsConfig):
        super(DataIngestionPipeline, self).__init__()
        self.config = config
        self.text_data = self.config.textdata
        self.image_data = self.config.imagedata
        self.audio_data = self.config.audiodata
        self.image_access_key = self.config.image_access_key
        self.audio_access_key = self.config.audio_access_key

    def run_data_ingestion_pipeline(self) -> None:
        try:
            logger.log_message("info", "Data ingestion pipeline started.")

            self.text_ingestion()
            self.image_ingestion()
            self.audio_ingestion()
            # self.data_ingestion()

            logger.log_message(
                "info", "Data ingestion pipeline completed successfully."
            )
            return None

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run data ingestion pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run data ingestion pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def text_ingestion(self) -> None:
        try:
            logger.log_message("info", "Text ingestion pipeline started.")
            textIngestion = TextIngestion(self.text_data)
            textIngestion.download_file()

            logger.log_message(
                "info", "Text ingestion pipeline completed successfully."
            )

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run text ingestion pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run text ingestion pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def image_ingestion(self) -> List[str]:
        try:
            logger.log_message("info", "Image ingestion pipeline started.")
            imageIngestion = ImageIngestion(
                self.image_data, self.config.image_access_key
            )
            image_links, image_destinations = imageIngestion.image_ingestion()
            logger.log_message("info", "Removing duplicate images...")
            print(image_destinations)
            duplicateImageProcessing = DuplicateImageProcessing(image_destinations)
            checkpoints = duplicateImageProcessing.run_check_duplicate_images()

            print(checkpoints)
            for checkpoint in checkpoints:
                os.remove(image_destinations[checkpoint])

            logger.log_message("info", "End removing duplicate images.")

            logger.log_message(
                "info", "Image ingestion pipeline completed successfully."
            )
            return image_links

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run image ingestion pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run image ingestion pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def audio_ingestion(self) -> None:
        try:
            logger.log_message("info", "Audio ingestion pipeline started.")
            audioIngestion = AudioIngestion(
                self.audio_data, self.config.audio_access_key
            )
            audioIngestion.audio_ingestion()

            logger.log_message(
                "info", "Audio ingestion pipeline completed successfully."
            )

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run audio ingestion pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run audio ingestion pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

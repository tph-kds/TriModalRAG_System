import os
import sys
from typing import List
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config  import DataTransformArgumentsConfig
from src.trim_rag.data.images import DuplicateImageProcessing
class DataTransformPipeline:
    def __init__(self, config: DataTransformArgumentsConfig):
        super(DataTransformPipeline, self).__init__()
        self.config = config
        self.text_data = self.config.textdata
        self.image_data = self.config.imagedata
        self.audio_data = self.config.audiodata
        self.image_access_key = self.config.image_access_key
        self.audio_access_key = self.config.audio_access_key



    def run_data_processing_pipeline(self) -> None:
        try:
            logger.log_message("info", "Data processing pipeline started.")

            self.text_processing()
            self.image_processing()
            self.audio_processing()
            # self.data_processing()

            logger.log_message("info", "Data processing pipeline completed successfully.")
            return None

        except Exception as e:
            logger.log_message("warning", "Failed to run data processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def text_processing(self) -> None:
       try:
            logger.log_message("info", "Text processing pipeline started.")
            textprocessing = TextTransform(self.text_data)
            textprocessing.download_file()

            logger.log_message("info", "Text processing pipeline completed successfully.")

       except Exception as e:
            logger.log_message("warning", "Failed to run text processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def image_processing(self) -> List[str]:
        try:
            logger.log_message("info", "Image processing pipeline started.")
            imageIngestion = ImageTransform(self.image_data, self.config.image_access_key)

            return image_links


        except Exception as e:
            logger.log_message("warning", "Failed to run image processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run image processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def audio_processing(self) -> None:
        try:
            logger.log_message("info", "Audio processing pipeline started.")
            audioprocessing = AudioTransform(self.audio_data, self.config.audio_access_key)
            audioprocessing.audio_processing()

            logger.log_message("info", "Audio processing pipeline completed successfully.")

        except Exception as e:
            logger.log_message("warning", "Failed to run audio processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run audio processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


import os
import sys
from typing import List, Optional
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config  import DataTransformationArgumentsConfig
from src.trim_rag.processing import TextTransform, ImageTransform, AudioTransform

class DataTransformPipeline:
    def __init__(self, config: DataTransformationArgumentsConfig):
        super(DataTransformPipeline, self).__init__()
        self.config = config
        self.root_dir = self.config.root_dir
        self.processed_dir = self.config.processed_dir
        self.text_data = self.config.text_data
        self.image_data = self.config.image_data
        self.audio_data = self.config.audio_data

        self.list_text_processeds: List[int] = []
        self.list_image_processeds: List[int] = []
        self.list_audio_processeds: List[int] = []


    def run_data_processing_pipeline(self) -> Optional[List[int]]:
        try:
            logger.log_message("info", "Data processing pipeline started.")

            # textprocessing = None
            # imageprocessing = None
            # audioprocessing = None
            textprocessing = self.text_processing()
            imageprocessing = self.image_processing()
            audioprocessing = self.audio_processing()
            # self.data_processing()

            logger.log_message("info", "Data Processing pipeline completed successfully.")
            return textprocessing, imageprocessing, audioprocessing

        except Exception as e:
            logger.log_message("warning", "Failed to run Data Processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run Data Processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def text_processing(self) -> None:
       try:
            logger.log_message("info", "Text processing pipeline started.")
            # access data folder before transforming
            dir_textdata = os.getcwd() + "/" + self.text_data.text_dir
            list_textdata = os.listdir(dir_textdata)
            # print(list_textdata)
            for listr_textdata in list_textdata:
                link_textdata = dir_textdata + "/" + listr_textdata
                link_textdata = link_textdata.replace("\\", "/")
                # print(link_textdata)
                # print(self.text_data.text_path +  "hi")
                # self.text_data.text_path = link_textdata
                textprocessing = TextTransform(self.text_data, 
                                               text_path = link_textdata
                                               )
                text_data_prcessed = textprocessing.text_processing()

                self.list_text_processeds.append(text_data_prcessed)
            
            logger.log_message("info", "Text processing pipeline completed successfully.")
            return self.list_text_processeds

       except Exception as e:
            logger.log_message("warning", "Failed to run text processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def image_processing(self) -> List[str]:
        try:
            logger.log_message("info", "Image Processing pipeline started.")
            # access data folder before transforming
            dir_imagedata = os.getcwd() + "/" + self.image_data.image_dir
            list_imagedata = os.listdir(dir_imagedata)
            for listr_imagedata in list_imagedata:
                link_imagedata = dir_imagedata + "/" + listr_imagedata
                # self.image_data.image_path = link_imagedata
                imageprocessing = ImageTransform(self.image_data, 
                                                 image_path=link_imagedata
                                                 )
                image_data_prcessed = imageprocessing.image_processing()

                self.list_image_processeds.append(image_data_prcessed)
            
            logger.log_message("info", "Image Processing pipeline completed successfully.")
            return self.list_image_processeds


        except Exception as e:
            logger.log_message("warning", "Failed to run Image Processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run Image Processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def audio_processing(self) -> None:
        try:
            logger.log_message("info", "Audio processing pipeline started.")
            # access data folder before transforming
            dir_audiodata = os.getcwd() + "/" + self.audio_data.audio_dir
            list_audiodata = os.listdir(dir_audiodata)
            for listr_audiodata in list_audiodata:
                link_audiodata = dir_audiodata + "/" + listr_audiodata
                # self.audio_data.audio_path = link_audiodata
                audioprocessing = AudioTransform(self.audio_data, 
                                                 audio_path=link_audiodata
                                                 )
                audio_data_prcessed = audioprocessing.audio_processing()

                self.list_audio_processeds.append(audio_data_prcessed)
            
            logger.log_message("info", "Audio Processing pipeline completed successfully.")
            return self.list_audio_processeds

            # logger.log_message("info", "Audio processing pipeline completed successfully.")

        except Exception as e:
            logger.log_message("warning", "Failed to run Audio Processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run Audio Processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


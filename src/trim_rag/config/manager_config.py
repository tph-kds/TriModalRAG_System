import os
from src.trim_rag.utils import read_yaml, create_directories
from src.config_params import CONFIG_FILE_PATH
from src.trim_rag.config import (LoggerArgumentsConfig, 
                                 ExceptionArgumentsConfig,
                                 DataIngestionArgumentsConfig, 
                                 TextDataIngestionArgumentsConfig,
                                 ImageDataIngestionArgumentsConfig,
                                 AudioDataIngestionArgumentsConfig
                                )
from src.config_params.constants import image_access_key, audio_access_key

class ConfiguarationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 image_access_key = image_access_key,
                 audio_access_key = audio_access_key 
                 ):
        
        super(ConfiguarationManager, self).__init__()

        self.config = read_yaml(config_filepath)
        self.image_access_key = image_access_key
        self.audio_access_key = audio_access_key
        create_directories([self.config.artifacts_root])

    
    def get_logger_arguments_config(self) -> LoggerArgumentsConfig:
        config = self.config.logger

        create_directories([config.root_dir])

        data_logger_config = LoggerArgumentsConfig(
            name = config.name,
            log_dir = config.log_dir,
            name_file_logs = config.name_file_logs,
            format_logging = config.format_logging,
            datefmt_logging = config.datefmt_logging
        )

        return data_logger_config
    

    def get_exception_arguments_config(self) -> ExceptionArgumentsConfig:
        config = self.config.exception

        # create_directories([config.root_dir])

        data_exception_config = ExceptionArgumentsConfig(
            error_message = config.exception.error_message,
            error_details = config.exception.error_details
        )

        return data_exception_config
    
    def _get_textdata_arguments_config(self) -> TextDataIngestionArgumentsConfig:
        config = self.config.data_ingestion.text_data

        create_directories([config.text_dir])

        text_data_ingestion_config = TextDataIngestionArgumentsConfig(
            text_dir = config.text_dir,
            query = config.query,
            max_results = config.max_results,
            destination = config.destination
        )

        return text_data_ingestion_config
    
    def _get_imagedata_arguments_config(self) -> ImageDataIngestionArgumentsConfig:
        config = self.config.data_ingestion.image_data

        create_directories([config.image_dir])

        image_data_ingestion_config = ImageDataIngestionArgumentsConfig(
            image_dir = config.image_dir,
            url = config.url,
            query = config.query,
            max_results = config.max_results,
            max_pages = config.max_pages,
            destination = config.destination,
        )

        return image_data_ingestion_config

    def _get_audiodata_arguments_config(self) -> AudioDataIngestionArgumentsConfig:
        config = self.config.data_ingestion.audio_data

        create_directories([config.audio_dir])

        audio_data_ingestion_config = AudioDataIngestionArgumentsConfig(
            audio_dir = config.audio_dir,
            url = config.url,
            query = config.query,
            max_results = config.max_results,
            destination = config.destination
        )

        return audio_data_ingestion_config
    
    def get_data_ingestion_arguments_config(self) -> DataIngestionArgumentsConfig:
        config = self.config.data_ingestion
        config.image_access_key = self.image_access_key
        config.audio_access_key = self.audio_access_key

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionArgumentsConfig(
            root_dir = config.root_dir,
            image_access_key = config.image_access_key,
            audio_access_key = config.audio_access_key,
            textdata = self._get_textdata_arguments_config(),
            audiodata = self._get_audiodata_arguments_config(),
            imagedata = self._get_imagedata_arguments_config()
        )

        return data_ingestion_config

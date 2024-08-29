import os
from src.trim_rag.utils import read_yaml, create_directories
from src.config_params import CONFIG_FILE_PATH
from src.trim_rag.config import (LoggerArgumentsConfig, 
                                 ExceptionArgumentsConfig)

class ConfiguarationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

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
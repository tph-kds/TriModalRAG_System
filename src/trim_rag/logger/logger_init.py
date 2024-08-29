from src.trim_rag.logger.logger_setting import MainLoggerHandler
from src.trim_rag.config import ConfiguarationManager
from src.config_params import CONFIG_FILE_PATH

logger_config = ConfiguarationManager(CONFIG_FILE_PATH).get_logger_arguments_config()

logger = MainLoggerHandler(logger_config= logger_config)

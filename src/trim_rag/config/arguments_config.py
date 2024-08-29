import sys
from pathlib import Path
from dataclasses import dataclass, field

@dataclass(frozen=True)
class LoggerArgumentsConfig():
    """
    Logger arguments config class for the logger.
    
    """

    name: str = field(
        default="trimRag_logger",
        metadata={"help": "Name of the logger."}
    )
    log_dir: str  = field(
        default="logs", 
        metadata={"help": "Directory to save logs."}
    )
    name_file_logs: str = field(
        default="running_logs.log",
        metadata={"help": "Name of the log file."}
    )
    format_logging: str = field(
        default="[%(asctime)s - { %(levelname)s } - { %(module)s } - %(message)s]",
        metadata={"help": "Format of the log file."}
    )
    datefmt_logging: str = field(       
        default="%m/%d/%Y %H:%M:%S",
        metadata={"help": "Date format of the log file."}
    )
@dataclass(frozen=True)
class ExceptionArgumentsConfig():
    error_message:str = field(
        default="Error occured in python script name [{file_name}] line number [{line_number}] error message [{error_message}]",
        metadata={"help": "Error message for exception."}
    )

    error_details:sys = field(  
        default=None,
        metadata={"help": "Error details for exception."}
    )



@dataclass(frozen=True)
class ArgumentsConfig():
    logger: LoggerArgumentsConfig
    exception: ExceptionArgumentsConfig

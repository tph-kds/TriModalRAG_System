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
class AudioDataIngestionArgumentsConfig():
    audio_dir: str  = field(
        default="data/audio",
        metadata={"help": "Directory to save audio data."}
    )
    query: str = field(
        default="weather",
        metadata={"help": "Query for calling api from freesound.org website for images."}
    )
    max_results: int = field(
        default=50,
        metadata={"help": "Maximum number of audio to return."}
    )
    destination: str = field(
        default="data/audio",
        metadata={"help": "Destination directory to save audio data."}
    )   
    url: str = field(
        default="https://freesound.org/apiv2/search/text/",
        metadata={"help": "Url for calling api from freesound.org website for audio."}
    )
@dataclass(frozen=True)
class ImageDataIngestionArgumentsConfig():
    image_dir: str  = field(
        default="data/image",
        metadata={"help": "Directory to save image data."}
    )
    url: str = field(
        default="https://api.unsplash.com/search/photos",
        metadata={"help": "Url for calling api from unsplash.com website for images."}
    )
    query: str = field(
        default="weather",
        metadata={"help": "Query for calling api from unsplash.com website for images."}
    )
    max_results: int = field(
        default=50,
        metadata={"help": "Maximum number of image to return."}
    ) 
    destination : str = field(
        default="data/image",
        metadata={"help": "Destination directory to save image data."}
    )

@dataclass(frozen=True)
class TextDataIngestionArgumentsConfig():
    text_dir: str  = field(
        default="data/text",
        metadata={"help": "Directory to save text data."}
    )
    query: str = field(
        default="machine learning weather prediction",
        metadata={"help": "Query for calling api from openweathermap.org."}
    )
    max_results: int = field(
        default=50,
        metadata={"help": "Maximum number of pdf file to return."}
    )
    destination: str = field(
        default="data/text",
        metadata={"help": "Destination directory to save text data."}
    )


@dataclass(frozen=True)
class DataIngestionArgumentsConfig():
    root_dir: str  = field(
        default="data",
        metadata={"help": "Root directory to save data."}
    )
    textdata :TextDataIngestionArgumentsConfig
    audiodata :AudioDataIngestionArgumentsConfig
    imagedata :ImageDataIngestionArgumentsConfig

@dataclass(frozen=True)
class ArgumentsConfig():
    logger: LoggerArgumentsConfig
    exception: ExceptionArgumentsConfig
    data_ingestion: DataIngestionArgumentsConfig

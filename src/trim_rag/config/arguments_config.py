import sys
from pathlib import Path
from dataclasses import dataclass, field


# LOGGER PARAMS | EXCEPTION PARAMS
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

# # DATA INGESTION PARAMS

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
    max_pages: int = field(
        default=10,
        metadata={"help": "Maximum number of pages to return."}
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
    image_access_key: str = field(
        default="API_IMAGE_DATA",
        metadata={"help": "Access key for calling api from unsplash.com website for images."}
    )
    audio_access_key: str = field(  
        default="API_AUDIO_DATA",
        metadata={"help": "Access key for calling api from freesound.org website for audio."}
    )
    textdata :TextDataIngestionArgumentsConfig = field(
        default_factory=TextDataIngestionArgumentsConfig
    )
    audiodata :AudioDataIngestionArgumentsConfig = field(
        default_factory=AudioDataIngestionArgumentsConfig
    )
    imagedata :ImageDataIngestionArgumentsConfig = field(
        default_factory=ImageDataIngestionArgumentsConfig
    )

# @dataclass(frozen=True)
# class ArgumentsConfig():
#     logger: LoggerArgumentsConfig
#     exception: ExceptionArgumentsConfig
#     data_ingestion: DataIngestionArgumentsConfig

# DATA TRANSFORMATION PARAMS


@dataclass(frozen=True)
class TextDataTransformArgumentsConfig():
    processed_dir: str  = field(
        default="processed_data/text",
        metadata={"help": "Root directory to save data."}
    )
    
    text_dir: str  = field(
        default="data/text",
        metadata={"help": "Root directory to save data."}
    )



@dataclass(frozen=True)
class AudioDataTransformArgumentsConfig():
    processed_dir: str  = field(
        default="processed_data/audio",
        metadata={"help": "Root directory to save data."}
    )
    audio_dir: str  = field(
        default="data/audio",
        metadata={"help": "Root directory to save data."}
    )

@dataclass(frozen=True)
class ImageDataTransformArgumentsConfig():
    processed_dir: str  = field(
        default="processed_data/image",
        metadata={"help": "Root directory to save data."}
    )
    image_dir: str  = field(
        default="data/image",
        metadata={"help": "Root directory to save data."}
    )
    image_path : str = field(
        default="",
        metadata={"help": "Root directory to save data."}
    )
    size: int = field(
        default=224,
        metadata={"help": "Size of the image."}
    )
    rotate: int = field(
        default=10,
        metadata={"help": "Rotation of the image."}
    )
    horizontal_flip: bool = field(
        default=True,   
        metadata={"help": "Horizontal flip of the image."}
    )
    rotation: int = field(
        default=10,
        metadata={"help": "Rotation of the image."}
    )
    brightness: float = field(
        default=0.2,
        metadata={"help": "Brightness of the image."}
    )
    contrast: float = field(
        default=0.2,
        metadata={"help": "Contrast of the image."}
    )
    scale: float = field(
        default=0.8,
        metadata={"help": "Scale of the image."}
    )
    ratio: float = field(
        default=0.8,
        metadata={"help": "Ratio of the image."}
    )
    saturation: float = field(
        default=0.2,
        metadata={"help": "Saturation of the image."}
    )
    hue: float = field(
        default=0.1,
        metadata={"help": "Hue of the image."}
    )
    format: str = field(
        default="JPEG",
        metadata={"help": "Format of the image."}
    )   

@dataclass(frozen=True)
class DataTransformationArgumentsConfig():
    root_dir: str  = field(
        default="src/artifacts/data",
        metadata={"help": "Root directory to save data."}
    )
    processed_dir: str  = field(
        default="processed_data",
        metadata={"help": "Root directory to save data."}
    )
    text_data :TextDataTransformArgumentsConfig = field(
        default_factory=TextDataTransformArgumentsConfig
    )
    audio_data :AudioDataTransformArgumentsConfig = field(
        default_factory=AudioDataTransformArgumentsConfig
    )
    image_data :ImageDataTransformArgumentsConfig = field(
        default_factory=ImageDataTransformArgumentsConfig
    )
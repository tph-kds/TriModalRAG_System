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
    text_path: str = field(
        default="",
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
    audio_path: str = field(
        default="",
        metadata={"help": "Root directory to save data."}
    )
    target_sr: int = field(
        default=16000,
        metadata={"help": "Target sampling rate."}
    )
    top_db: int = field(
        default=80,
        metadata={"help": "Top db for mel spectrogram"}
    )
    scale: int = field(
        default=1,
        metadata={"help": "Scale for mel spectrogram"}
    )
    fix: bool = field(
        default=True,
        metadata={"help": "Fix for mel spectrogram"}
    )
    mono: bool = field(
        default=True,
        metadata={"help": "Mono for mel spectrogram"}
    )
    pad_mode: str = field(
        default="reflect",
        metadata={"help": "Pad mode for mel spectrogram"}
    )

    frame_length: int = field(
        default=1024,
        metadata={"help": "Frame length for mel spectrogram"}
    )
    hop_length: int = field(    
        default=512,
        metadata={"help": "Hop length for mel spectrogram"}
    )

    n_steps: int = field(
        default=128,
        metadata={"help": "N steps for mel spectrogram"}
    )
    bins_per_octave: int = field(
        default=12,
        metadata={"help": "Bins per octave for mel spectrogram"}
    )
    res_type: str = field(
        default="kaiser_fast",
        metadata={"help": "Res type for mel spectrogram"}
    )
    rate: int = field(
        default=12,
        metadata={"help": "Rate for mel spectrogram"}
    )
    noise: float = field(
        default=0.005,
        metadata={"help": "Noise for mel spectrogram"}
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


@dataclass(frozen=True)
class TextEmbeddingArgumentsConfig():
    embedding_dir: str  = field(
        default="src/artifacts/data",
        metadata={"help": "Root directory to save data."}
    )
    pretrained_model_name: str = field(
        default="bert_base_cased",
        metadata={"help": "Pretrained model name."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device to use."}
    )
    return_dict: bool = field(
        default=True,
        metadata={"help": "Return dictionary."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max length."}
    )
    return_hidden_states: bool = field(
        default=True,
        metadata={"help": "Return hidden states."}
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Lower case."}
    )
    truncation: bool = field(
        default=True,
        metadata={"help": "Truncation for tokenization."}
    )
    return_tensor: bool = field(
        default=True,
        metadata={"help": "Return tensor."}
    )
    padding: bool = field(
        default=True,
        metadata={"help": "Padding for tokenization."}
    )
    add_special_tokens: bool = field(
        default=True,
        metadata={"help": "Add special tokens for tokenization."}
    )
    return_token_type_ids: bool = field(
        default=False,
        metadata={"help": "Return token type ids."}
    )
    return_attention_mask: bool = field(
        default=False,
        metadata={"help": "Return attention mask."}
    )
    return_overflowing_tokens: bool = field(
        default=False,
        metadata={"help": "Return overflowing tokens ."}
    )
    return_special_tokens_mask: bool = field(
        default=False,
        metadata={"help": "Return special tokens mask ."}
    )

@dataclass(frozen=True)
class ImageEmbeddingArgumentsConfig():
    embedding_dir: str  = field(
        default="src/artifacts/data",
        metadata={"help": "Root directory to save data."}
    )
    pretrained_model_name: str = field(
        default="",
        metadata={"help": "Pretrained model name."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device to use."}
    )
    output_hidden_states: bool = field(
        default=True,
        metadata={"help": "Output hidden states."}
    )
    output_attentions: bool = field(
        default=True,
        metadata={"help": "Output attention."}
    )
    return_dict:  bool = field(
        default=True,   
        metadata={"help": "Return dictionary."}
    ) 
    revision: str = field(
        default="",
        metadata={"help": "Revision."}
    )
    use_safetensors: bool = field(
        default=False,
        metadata={"help": "Use safetensors."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Ignore mismatched sizes."}
    )
    return_tensors: str = field(
        default="pt",
        metadata={"help": "Return tensors."}
    )
    return_overflowing_tokens: bool = field(
        default=False,
        metadata={"help": "Return overflowing tokens ."}
    )
    return_special_tokens_mask: bool = field(
        default=False,
        metadata={"help": "Return special tokens mask ."}
    )

@dataclass(frozen=True)
class AudioEmbeddingArgumentsConfig():
    embedding_dir: str  = field(
        default="src/artifacts/data",
        metadata={"help": "Root directory to save data."}
    )
    pretrained_model_name: str = field(
        default="",
        metadata={"help": "Pretrained model name."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device to use."}
    )
    revision: str = field(
        default="",
        metadata={"help": "Revision."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Ignore mismatched sizes."}
    )
    return_tensors: str = field(
        default="pt",
        metadata={"help": "Return tensors."}
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code."}
    )
    n_components: int = field(
        default=2,
        metadata={"help": "Number of components."}
    )

@dataclass(frozen=True)
class EmbeddingArgumentsConfig():
    root_dir: str  = field(
        default="src/artifacts/data",
        metadata={"help": "Root directory to save data."}
    )
    embedding_dir: str  = field(
        default="embeddings",
        metadata={"help": "Root directory to save data."}
    )
    text_data :TextEmbeddingArgumentsConfig = field(
        default_factory=TextEmbeddingArgumentsConfig
    )
    audio_data :AudioEmbeddingArgumentsConfig = field(
        default_factory=AudioEmbeddingArgumentsConfig
    )
    image_data :ImageEmbeddingArgumentsConfig = field(
        default_factory=ImageEmbeddingArgumentsConfig
    )

@dataclass(frozen=True)
class CrossModalEmbeddingArgumentsConfig():

    dim_hidden: int = field(
        default=512,
        metadata={"help": "Dimension of hidden layer."}
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device to use."}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability."}
    )
    batch_first : bool = field(
        default=False,
        metadata={"help": "Batch first."}
    )
    eps:    float = field(
        default=1e-6,
        metadata={"help": "Epsilon for layer normalization."}
    )
    bias: bool = field(
        default=True,
        metadata={"help": "Bias for layer normalization."}
    )
    num_layers: int = field(
        default=3,
        metadata={"help": "Number of layers."}
    )
    training: bool = field(
        default=True,
        metadata={"help": "Training mode ."}
    )
    inplace: bool = field(
        default=True,
        metadata={"help": "Inplace mode."}
    )   

@dataclass(frozen=True)
class SharedEmbeddingSpaceArgumentsConfig():
    dim_text : int = field(
        default=512,
        metadata={"help": "Dimension of text space."}
    )
    dim_image:  int = field(
        default=512,
        metadata={"help": "Dimension of image space."}
    )
    dim_sound:  int = field(
        default=512,
        metadata={"help": "Dimension of sound space."}
    )
    dim_shared : int = field(
        default=512,
        metadata={"help": "Dimension of shared space."}
    )
    device : str = field(
        default="cpu",
        metadata={"help": "Device to use."}
    )
    eps : float = field(
        default=1e-6,
        metadata={"help": "Epsilon for layer normalization."}
    )
    bias : bool = field(
        default=True,
        metadata={"help": "Bias for layer normalization."}
    )   

@dataclass(frozen=True)
class MultimodalEmbeddingArgumentsConfig():
    root_dir: str  = field(
        default="src/artifacts/data",
        metadata={"help": "Root directory to save data."}
    )
    embedding_dir: str  = field(
        default="embeddings",
        metadata={"help": "Root directory to save data."}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability."}
    )
    num_layers: int = field(
        default=3,
        metadata={"help": "Number of layers."}
    )
    crossmodal_embedding :CrossModalEmbeddingArgumentsConfig = field(
        default_factory=CrossModalEmbeddingArgumentsConfig
    )
    sharedspace_embedding :SharedEmbeddingSpaceArgumentsConfig = field(
        default_factory=SharedEmbeddingSpaceArgumentsConfig
    )

import os
from src.trim_rag.utils import read_yaml, create_directories
from src.config_params import CONFIG_FILE_PATH
from src.trim_rag.config import (LoggerArgumentsConfig, 
                                 ExceptionArgumentsConfig,
                                 DataIngestionArgumentsConfig, 
                                 TextDataIngestionArgumentsConfig,
                                 ImageDataIngestionArgumentsConfig,
                                 AudioDataIngestionArgumentsConfig,
                                 TextDataTransformArgumentsConfig,
                                 AudioDataTransformArgumentsConfig,
                                 ImageDataTransformArgumentsConfig,
                                 DataTransformationArgumentsConfig,
                                 TextEmbeddingArgumentsConfig,
                                 ImageEmbeddingArgumentsConfig,
                                 AudioEmbeddingArgumentsConfig,
                                 EmbeddingArgumentsConfig
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
    
    ### GETTING ALL DATA INGESTION PARAMS  ###  
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
    
    ### GETTING ALL DATA TRANSFORMATION PARAMS  ###  
    def _get_textdata_transform_arguments_config(self) -> TextDataTransformArgumentsConfig:
        config = self.config.data_processing.text_data

        create_directories([config.processed_dir])

        text_data_processing_config = TextDataTransformArgumentsConfig(
            processed_dir = config.processed_dir,
            text_dir= config.text_dir,
            text_path = config.text_path
        )

        return text_data_processing_config
    
    def _get_imagedata_transform_arguments_config(self) -> ImageDataTransformArgumentsConfig:
        config = self.config.data_processing.image_data

        create_directories([config.processed_dir])

        image_data_processing_config = ImageDataTransformArgumentsConfig(
            processed_dir = config.processed_dir,
            image_dir = config.image_dir,
            image_path = config.image_path,
            size = config.size,
            rotate = config.rotate,
            horizontal_flip = config.horizontal_flip,
            rotation = config.rotation,
            brightness = config.brightness,
            contrast = config.contrast,
            scale = config.scale,
            ratio = config.ratio,
            saturation = config.saturation,
            hue = config.hue,
            format = config.format,
        )

        return image_data_processing_config
    def _get_audiodata_transform_arguments_config(self) -> AudioDataTransformArgumentsConfig:
        config = self.config.data_processing.audio_data

        create_directories([config.processed_dir])

        audio_data_processing_config = AudioDataTransformArgumentsConfig(
            processed_dir = config.processed_dir,
            audio_dir = config.audio_dir,
            audio_path = config.audio_path,
            target_sr = config.target_sr,
            top_db = config.top_db,
            scale = config.scale,
            fix = config.fix,
            mono = config.mono,
            pad_mode = config.pad_mode,
            frame_length = config.frame_length,
            hop_length = config.hop_length,
            n_steps = config.n_steps,
            bins_per_octave = config.bins_per_octave,
            res_type = config.res_type,
            rate = config.rate,
            noise = config.noise
        )
        return audio_data_processing_config
    
    def get_data_processing_arguments_config(self) -> DataTransformationArgumentsConfig:
        config = self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataTransformationArgumentsConfig(
            root_dir = config.root_dir,
            processed_dir = config.processed_dir,
            text_data = self._get_textdata_transform_arguments_config(),
            audio_data = self._get_audiodata_transform_arguments_config(),
            image_data = self._get_imagedata_transform_arguments_config()
        )

        return data_processing_config
    
    def _get_textdata_embedding_arguments_config(self) -> TextEmbeddingArgumentsConfig:
        config = self.config.embedding.text_data

        create_directories([config.embedding_dir])

        text_data_processing_config = TextEmbeddingArgumentsConfig(
            embedding_dir = config.embedding_dir,
            pretrained_model_name = config.pretrained_model_name,
            device = config.device,
            return_dict = config.return_dict,
            max_length = config.max_length,
            return_hidden_states = config.return_hidden_states,
            do_lower_case = config.do_lower_case,
            truncation = config.truncation,
            return_tensor = config.return_tensor,
            padding = config.padding,
            max_length = config.max_length,
            add_special_tokens = config.add_special_tokens,
            return_token_type_ids = config.return_token_type_ids,
            return_attention_mask = config.return_attention_mask,
            return_overflowing_tokens = config.return_overflowing_tokens,
            return_special_tokens_mask = config.return_special_tokens_mask,
        )
        return text_data_processing_config
    
    def _get_imagedata_embedding_arguments_config(self) -> ImageEmbeddingArgumentsConfig:
        config = self.config.embedding.image_data

        create_directories([config.embedding_dir])

        image_data_processing_config = ImageEmbeddingArgumentsConfig(
            embedding_dir = config.embedding_dir,
            pretrained_model_name = config.pretrained_model_name,
            device = config.device,
            output_hidden_states = config.output_hidden_states,
            output_attentions = config.output_attentions,
            return_dict = config.return_dict,
            revision = config.revision,
            use_safetensors = config.use_safetensors,
            ignore_mismatched_sizes = config.ignore_mismatched_sizes,
            return_tensors = config.return_tensors,
            return_overflowing_tokens = config.return_overflowing_tokens,
            return_special_tokens_mask = config.return_special_tokens_mask,
        )
        return image_data_processing_config
    
    def _get_audiodata_embedding_arguments_config(self) -> AudioEmbeddingArgumentsConfig:
        config = self.config.embedding.audio_data

        create_directories([config.embedding_dir])

        audio_data_processing_config = AudioEmbeddingArgumentsConfig(
            embedding_dir = config.embedding_dir,
            pretrained_model_name = config.pretrained_model_name,
            device = config.device,
            revision = config.revision,
            ignore_mismatched_sizes = config.ignore_mismatched_sizes,
            return_tensors = config.return_tensors,
            trust_remote_code = config.trust_remote_code,
            n_components = config.n_components,
        )
        return audio_data_processing_config
    
    def get_data_embedding_arguments_config(self) -> EmbeddingArgumentsConfig:
        config = self.config.embedding

        create_directories([config.root_dir])

        data_embedding_config = EmbeddingArgumentsConfig(
            root_dir = config.root_dir,
            embedding_dir = config.embedding_dir,
            text_data = self._get_textdata_embedding_arguments_config(),
            audio_data = self._get_audiodata_embedding_arguments_config(),
            image_data = self._get_imagedata_embedding_arguments_config()
        )

        return data_embedding_config

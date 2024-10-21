import os
import sys
import torch
import numpy as np
import soundfile as sf

from typing import Optional
from sklearn.decomposition import PCA
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config import AudioEmbeddingArgumentsConfig

from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model


class AudioEmbedding:
    def __init__(self, config: AudioEmbeddingArgumentsConfig):
        super(AudioEmbedding, self).__init__()
        self.config = config
        self.audio_data = self.config

        self.pretrained_model_name = self.audio_data.pretrained_model_name
        self.device = self.audio_data.device
        self.revision = self.audio_data.revision
        self.ignore_mismatched_sizes = self.audio_data.ignore_mismatched_sizes
        self.return_tensors = self.audio_data.return_tensors
        self.trust_remote_code = self.audio_data.trust_remote_code
        self.n_components = self.audio_data.n_components

    def _get_model(self) -> Optional[Wav2Vec2Model]:
        try:
            logger.log_message("info", "Getting model for embedding audio started.")
            model = Wav2Vec2Model.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
                revision=self.revision,
                ignore_mismatched_sizes=self.ignore_mismatched_sizes,
            )

            model.eval()
            model.to(self.device)

            logger.log_message(
                "info", "Getting model for embedding audio completed successfully."
            )
            return model

        except Exception as e:
            logger.log_message(
                "warning", "Failed to get model for embedding audio: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to get model for embedding audio: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _get_tokenizer(self) -> Optional[Wav2Vec2Tokenizer]:
        try:
            logger.log_message("info", "Getting tokenizer for embedding audio started.")
            # Load the tokenizer
            tokenizer = Wav2Vec2Tokenizer.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
            )

            logger.log_message(
                "info", "Getting tokenizer for embedding audio completed successfully."
            )
            return tokenizer

        except Exception as e:
            logger.log_message(
                "warning", "Failed to get tokenizer for embedding audio: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to get tokenizer for embedding audio: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    # Example of dimensionality reduction (optional)
    def _reduce_dimensions(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            logger.log_message(
                "info", "Reducing dimensions for embedding audio started."
            )
            pca = PCA(n_components=self.n_components)
            reduced_embeddings = pca.fit_transform(embeddings.squeeze(0))

            logger.log_message(
                "info",
                "Reducing dimensions for embedding audio completed successfully.",
            )
            return reduced_embeddings

        except Exception as e:
            logger.log_message(
                "warning", "Failed to reduce dimensions for embedding audio: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to reduce dimensions for embedding audio: "
                + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _get_features(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            logger.log_message("info", "Getting features for embedding audio started.")
            # Tokenize audio (convert to input format expected by the model)
            tokenizer = self._get_tokenizer()
            inputs = tokenizer(audio, return_tensors=self.return_tensors).input_values
            inputs = inputs.to(self.device)

            # Get embeddings from the model
            with torch.no_grad():
                model = self._get_model()
                outputs = model(inputs)
                embeddings = outputs.last_hidden_state

            logger.log_message(
                "info", "Getting features for embedding audio completed successfully."
            )
            # embeddings = embeddings.mean(dim=1)  # Average over sequence length
            # embeddings = embeddings[:, :self.target_dim]
            return embeddings.cpu().numpy()

        except Exception as e:
            logger.log_message(
                "warning", "Failed to get features for embedding audio: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to get features for embedding audio: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def embedding_audio(self, audios) -> Optional[torch.Tensor]:
        try:
            logger.log_message("info", "Embedding audio started.")
            audio = self._get_features(audio=audios)
            logger.log_message("info", "Embedding audio completed successfully.")
            return audio

        except Exception as e:
            logger.log_message("warning", "Failed to embed audio: " + str(e))
            my_exception = MyException(
                error_message="Failed to embed audio: " + str(e),
                error_details=sys,
            )
            print(my_exception)

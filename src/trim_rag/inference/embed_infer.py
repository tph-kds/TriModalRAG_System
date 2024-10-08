import os
import sys
import torch
import numpy as np

from typing import Optional, List, Union
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.embedding import (
    TextEmbedding,
    ImageEmbedding,
    AudioEmbedding
)

class EmbeddingTextInference(TextEmbedding):

    def __init__(self, **kwargs) -> None:
        super(EmbeddingTextInference, self).__init__(**kwargs)

    def text_embedding(self, input: str) -> Optional[torch.Tensor]:
        try:
            logger.log_message("info", "Embedding text in the Inference Phase started.")
            tokenizer = self._get_tokenizer()
            embeddings, input_ids = self.get_bert_embeddings(input)
            logger.log_message("info", "Embedding text in the Inference Phase completed successfully.")
            # Convert input IDs back to tokens (for verification)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            # Print tokens and their corresponding embeddings
            tokens_list = [token for token, embedding in zip(tokens, embeddings[0])]
            return embeddings, tokens_list

        except Exception as e:
            logger.log_message("warning", "Failed to embed text in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to embed text in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)

class EmbeddingImageInference(ImageEmbedding):

    def __init__(self, **kwargs) -> None:
        super(EmbeddingImageInference, self).__init__(**kwargs)

    def image_embedding(self, input) -> Optional[np.ndarray]:
        try: 
            logger.log_message("info", "Embedding Images in the Inference Phase started.")
            # Preprocess image
            image_tensor = self._preprocess_image(input)
            # Get image features
            image_features = self.get_features(image_tensor)

            logger.log_message("info", "Embedding Images in the Inference Phase completed successfully.")
            return image_features

        except Exception as e:
            logger.log_message("warning", "Failed to Embedding Images in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to Embedding Images in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception) 


class EmbeddingAudioInference(AudioEmbedding):

    def __init__(self, **kwargs) -> None:
        super(EmbeddingAudioInference, self).__init__(**kwargs)

    def audio_embedding(self, input) -> Optional[torch.Tensor]:
        try:
            logger.log_message("info", "Embedding audio in the Inference Phase started.")
            audio = self._get_features(audio=input)
            logger.log_message("info", "Embedding audio in the Inference Phase completed successfully.")
            return audio

        except Exception as e:
            logger.log_message("warning", "Failed to embed audio in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to embed audio in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)

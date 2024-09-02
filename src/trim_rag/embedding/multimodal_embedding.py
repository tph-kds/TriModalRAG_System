import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import MultimodalEmbeddingArgumentsConfig
from src.trim_rag.embedding.sharedSpace import SharedEmbeddingSpace
from src.trim_rag.embedding.crossAttention import CrossModalEmbedding
from src.trim_rag.config import (
    SharedEmbeddingSpaceArgumentsConfig,
    CrossModalEmbeddingArgumentsConfig
)



class MultimodalEmbedding(nn.Module):

    def __init__(self, 
                 config: MultimodalEmbeddingArgumentsConfig,
                 dim_text: int = 512,
                 dim_image: int = 512,
                 dim_sound: int = 512,
                 ):
        super(MultimodalEmbedding, self).__init__()
        
        self.config = config
        self.num_layers = self.config.num_layers
        self.dropout = self.config.dropout
        
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_sound = dim_sound

        self.shared_space = self._get_init_shared_space()
        self.crossmodal_embedding = self._get_init_crossmodal_embedding()

    def _get_init_shared_space(self) -> Optional[SharedEmbeddingSpaceArgumentsConfig]:
        return self.config.sharedspace_embedding
    
    def _get_init_crossmodal_embedding(self) -> Optional[CrossModalEmbeddingArgumentsConfig]:
        return self.config.crossmodal_embedding
        

    def forward(self, 
                text: torch.Tensor , 
                image: torch.Tensor, 
                sound: torch.Tensor) -> torch.Tensor:
        
        text_out_shared, image_out_shared, sound_out_shared = self._get_shared_space(text, image, sound)
        combined_output = self._get_crossmodal_embedding(text_out_shared, 
                                                         image_out_shared, 
                                                         sound_out_shared
                                                         )
        for _ in range(self.num_layers):
            combined_output = self._FullyConnectedLayer(combined_output)

        combined_output = nn.Dropout(self.dropout)(combined_output)
        
        return combined_output

    def _get_shared_space(self, 
                          text: torch.Tensor, 
                          image: torch.Tensor, 
                          sound: torch.Tensor) -> torch.Tensor:
        try:
            logger.log_message("info", "Loading Shared Embedding Space for Multimodal Embedding ...")
            shared_space = SharedEmbeddingSpace(self.shared_space)
            shared_space.to(self.config.device)
            shared_space.eval()
            shared_space._init_weights_and_bias()
            text_out_shared, image_out_shared, sound_out_shared = shared_space(text, image, sound)

            logger.log_message("info", "Shared Embedding Space for Multimodal Embedding loaded successfully.")
            return text_out_shared, image_out_shared, sound_out_shared

        except Exception as e:
            logger.log_message("warning", "Failed to load Shared Embedding Space for Multimodal Embedding: " + str(e))
            my_exception = MyException(
                error_message = "Failed to load Shared Embedding Space for Multimodal Embedding: " + str(e),
                error_details = sys
            )   
            print(my_exception)
        
    def  _get_crossmodal_embedding(self, 
                                   text: torch.Tensor, 
                                   image: torch.Tensor, 
                                   sound: torch.Tensor) -> torch.Tensor:
        try:
            logger.log_message("info", "Loading CrossModal Embedding for Multimodal Embedding ...")
            crossmodal_embedding = CrossModalEmbedding(self.crossmodal_embedding)
            crossmodal_embedding.to(self.config.device)
            crossmodal_embedding.eval()
            crossmodal_embedding._init_weights_and_bias()
            combined_embeddings = crossmodal_embedding(text, image, sound)

            logger.log_message("info", "CrossModal Embedding for Multimodal Embedding loaded successfully.")
            return combined_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to load CrossModal Embedding for Multimodal Embedding: " + str(e))    
            my_exception = MyException(
                error_message = "Failed to load CrossModal Embedding for Multimodal Embedding: " + str(e),
                error_details = sys
            )   
            print(my_exception) 

    def _FullyConnectedLayer(self, 
                             dim_input: int, 
                             dim_output: int) -> Optional[nn.Linear]:
        try:
            logger.log_message("info", " Initializing Linear layers  ...")
            linear_layer = nn.Linear(in_features=dim_input, 
                             out_features=dim_output)

            logger.log_message("info", " Initialized Linear layers")

            return linear_layer

        except Exception as e:
            logger.log_message("warning", "Failed to load MultimodalEmbedding: _FullyConnectedLayer: " + str(e))
            my_exception = MyException(
                error_message = "Failed to load MultimodalEmbedding: _FullyConnectedLayer: " + str(e),
                error_details = sys
            )   
            print(my_exception)



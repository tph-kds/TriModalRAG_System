import os
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import SharedEmbeddingSpaceArgumentsConfig


class SharedEmbeddingSpace(nn.Module):

    def __init__(self, config: SharedEmbeddingSpaceArgumentsConfig):
        super(SharedEmbeddingSpace, self).__init__()
        
        self.config = config
        self.dim_text = self.config.dim_text # 512
        self.dim_image = self.config.dim_image # 512
        self.dim_sound = self.config.dim_sound # 512
        self.dim_shared = self.config.dim_shared # 512
        self.device = self.config.device # cpu
        self.eps = self.config.eps # 1e-6
        self.bias = self.config.bias # True

        # Activation function (optional, can be any activation like ReLU, GELU)
        self.activation = nn.ReLU()

    def forward(self, text, image, sound):
        # Apply linear layers
        text_emb = self._linear_layers(self.dim_text)
        image_emb = self._linear_layers(self.dim_image)
        sound_emb = self._linear_layers(self.dim_sound)

        # Project each modality to the shared space
        text_emb = self.activation(text_emb(text))
        image_emb = self.activation(image_emb(image))
        sound_emb = self.activation(sound_emb(sound))
        

        # Normalize embeddings (optional)
        text_emb = self._LayerNorm(text_emb)
        image_emb = self._LayerNorm(image_emb)
        sound_emb = self._LayerNorm(sound_emb)

        
        return text_emb, image_emb, sound_emb
    
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.xavier_uniform_(self.sound_proj.weight)

    def _init_bias(self) -> None:
        nn.init.zeros_(self.text_proj.bias)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.zeros_(self.sound_proj.bias)

    def _init_weights_and_bias(self) -> None:
        self._init_weights()
        self._init_bias()
    
    def _linear_layers(self, dim_input) -> Optional[nn.Linear]:
        try:
            logger.log_message("info", " Initializing Linear layers  ...")
            self._init_weights_and_bias()

            proj = nn.Linear(in_features=dim_input, 
                             out_features=self.dim_shared,
                             device=self.device,
                             dtype=torch.float32,
                             bias=self.bias,)

            logger.log_message("info", " Initialized Linear layers")

            return proj

        except Exception as e:
            logger.log_message("warning", "Failed to load SharedEmbeddingSpace: _linear_layers: " + str(e))
            my_exception = MyException(
                error_message = "Failed to load SharedEmbeddingSpace: _linear_layers: " + str(e),
                error_details = sys
            )   
            print(my_exception)

    def _LayerNorm(self) -> Optional[nn.LayerNorm]:
        try:
            logger.log_message("info", " Initializing LayerNorm ...")
            layer_norm = nn.LayerNorm(self.dim_shared, 
                                      device=self.device,
                                      dtype=torch.float32,
                                      eps=self.eps)

            logger.log_message("info", " Initialized LayerNorm")

            return layer_norm
        
        except Exception as e:
            logger.log_message("warning", "Failed to load SharedEmbeddingSpace: _LayerNorm: " + str(e))
            my_exception = MyException(
                error_message = "Failed to load SharedEmbeddingSpace: _LayerNorm: " + str(e),
                error_details = sys
            )   
            print(my_exception)





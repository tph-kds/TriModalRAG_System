import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import CrossModalEmbeddingArgumentsConfig

class CrossModalEmbedding(nn.Module):

    def __init__(self, config: CrossModalEmbeddingArgumentsConfig):
        super(CrossModalEmbedding, self).__init__()
        self.config = config
        self.crossmodal_embedding = self.config.crossmodal_embedding
        self.dim_hidden = self.config.dim_hidden # 512
        self.num_heads = self.config.num_heads # 8
        self.device = self.config.device # cpu
        self.dropout = self.config.dropout # 0.1
        self.batch_first = self.config.batch_first # False
        self.eps = self.config.eps # 1e-6
        self.dtype = self.config.dtype # torch.float32
        self.bias = self.config.bias # True
        self.num_layers = self.config.num_layers # 3
        self.training = self.config.training # True
        self.inplace = self.config.inplace # True
        
        # Project each modality to a common space
        self.text_proj = self._LayerNorm()
        self.image_proj = self._LayerNorm()
        self.sound_proj = self._LayerNorm()
        

    def forward(self, 
                text: torch.Tensor , 
                image: torch.Tensor, 
                sound: torch.Tensor) -> torch.Tensor:
        
        # Project each modality to a common space
        text_proj = self.text_proj(text)
        image_proj = self.image_proj(image)
        audio_proj = self.sound_proj(sound)

        # Apply attention mechanisms here
        text_out, _ = self._MultiHeadAttention(text_proj, image_proj, image_proj)
        image_out, _ = self._MultiHeadAttention(image_proj, audio_proj, audio_proj)
        audio_out, _ = self._MultiHeadAttention(audio_proj, text_proj, text_proj)
        
        # Stack the modalities
        modalities = torch.stack([text_out, image_out, audio_out], dim=0)
        
        # Apply multi-head attention         
        attn_output, _ = self._MultiHeadAttention(modalities, modalities, modalities)
        
        # Normalize and project to output
        output = self._LayerNorm(attn_output)

        ### loop for 3 time with FullyConectedLayer
        for _ in range(self.num_layers):
            output = self._FullyConectedLayer(output)

        # Apply dropout
        output = F.dropout(input=output, 
                           p=self.dropout, 
                           training=self.training,
                           inplace=self.inplace)

        # Aggregate the modalities (e.g., sum, mean, or concatenation)
        output = output.sum(dim=0)  # Example: sum the modalities

        return output
    
    def _FullyConectedLayer(self) -> nn.Linear:
        try:
            logger.log_message("info", "Loading CrossModalEmbedding: _FullyConectedLayer ...")
            # Output projection
            linear_layer = nn.Linear(in_features=self.dim_hidden, 
                                            out_features=self.dim_hidden,
                                            device=self.device,
                                            dtype=self.dtype,
                                            bias=self.bias)

            logger.log_message("info", "Loaded CrossModalEmbedding: _FullyConectedLayer")

            return linear_layer

        except Exception as e:
            logger.log_message("warning", "Failed to load CrossModalEmbedding: _FullyConectedLayer: " + str(e))
            my_exception = MyException(
                error_message = "Failed to load CrossModalEmbedding: _FullyConectedLayer: " + str(e),
                error_details = sys
            )   
            print(my_exception)
            

    def _MultiHeadAttention(self)  -> nn.MultiheadAttention:
        try:
            logger.log_message("info", "Loading CrossModalEmbedding: _MultiHeadAttention ...")
            # Multi-head attention for each modality
            multihead_attn = nn.MultiheadAttention(embed_dim=self.dim_hidden, 
                                                        num_heads=self.num_heads,
                                                        batch_first=self.batch_first,
                                                        dropout=self.dropout,
                                                        device=self.device,
                                                        dtype=self.dtype)
            logger.log_message("info", "Loaded CrossModalEmbedding: _MultiHeadAttention")

            return multihead_attn

        except Exception as e:
            logger.log_message("warning", "Failed to load CrossModalEmbedding: _MultiHeadAttention: " + str(e))
            my_exception = MyException( 
                error_message = "Failed to load CrossModalEmbedding: _MultiHeadAttention: " + str(e),
                error_details = sys
            )
            print(my_exception)

    def _LayerNorm(self) -> nn.LayerNorm:
        try:
            logger.log_message("info", "Loading CrossModalEmbedding: _LayerNorm ...")
            # Layer normalization
            norm = nn.LayerNorm(self.dim_hidden,
                                    eps=self.eps,
                                    device=self.device,
                                    dtype=self.dtype)

            logger.log_message("info", "Loaded CrossModalEmbedding: _LayerNorm")

            return norm

        except Exception as e:
            logger.log_message("warning", "Failed to load CrossModalEmbedding: _LayerNorm: " + str(e))
            my_exception = MyException(
                error_message = "Failed to load CrossModalEmbedding: _LayerNorm: " + str(e),
                error_details = sys
            )
            print(my_exception)

        


import os
import sys
import torch
import torch.nn as nn

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import AttentionFusionArgumentsConfig


class AttentionFusion(nn.Module):
    def __init__(self, config: AttentionFusionArgumentsConfig):
        super(AttentionFusion, self).__init__()

        self.config = config
        
        self.input_dim = self.config.input_dim # 512
        self.embed_dim = self.config.embed_dim # 512
        self.num_heads = self.config.num_heads # 8
        self.dropout = self.config.dropout # 0.1

    def forward(self, 
                text: torch.Tensor, 
                image: torch.Tensor, 
                audio: torch.Tensor
                ):
        # Stack features for attention fusion
        x = torch.stack((text, image, audio), dim=0)
        x , _ = self._attention_fusion()(x, x, x)

        return self._attention_fusion()(x)

    def _attention_fusion(self) -> nn.MultiheadAttention:
        try:
            attention = nn.MultiheadAttention(
                embed_dim=self.input_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
            return attention

        except Exception as e:
            logger.log_message("info", f"Error creating attention layer: {e}")
            my_exception = MyException(
                error_message=f"Error creating attention layer: {e}",
                error_details= sys,
            )
            print(my_exception)
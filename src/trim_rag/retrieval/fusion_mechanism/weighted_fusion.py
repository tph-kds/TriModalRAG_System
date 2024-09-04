import os
import sys
import torch
import torch.nn as nn

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import WeightedFusionArgumentsConfig


class WeightedFusion(nn.Module):
    def __init__(self, config: WeightedFusionArgumentsConfig):
        super(WeightedFusion, self).__init__()

        self.config = config

    def forward(self, 
                text: torch.Tensor, 
                image: torch.Tensor, 
                audio: torch.Tensor
                ):
        weighted_sum = (self.weights[0] * text) + (self.weights[1] * image) + (self.weights[2] * audio)
        

        return weighted_sum / torch.sum(self.weights)
    
    def weights_fusion(self):
        try:
            weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

            return weights

        except Exception as e:
            logger.log_message("info", f"Error creating weighted fusion: {e}")
            my_exception = MyException(
                error_message=f"Error creating weighted fusion: {e}",
                error_details= sys,
            )
            print(my_exception)
            


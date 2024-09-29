import os
import sys
import torch
import torch.nn as nn

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import ModalityAlignerArgumentsConfig


class ModalityAligner(nn.Module):
    def __init__(self, config: ModalityAlignerArgumentsConfig):
        super(ModalityAligner, self).__init__()

        self.config = config
        self.input_dim = self.config.input_dim # 512
        self.output_dim = self.config.output_dim # 512

    def forward(self, x):
        linear_layer = self._modality_aligner()  
        return linear_layer(x)

    def _modality_aligner(self) -> nn.Linear:
        try:
            linear_layer = nn.Linear(in_features=self.input_dim, 
                                    out_features=self.output_dim
                                    )
            return linear_layer

        except Exception as e:
            logger.log_message("warning", f"Error creating linear layer: {e}")
            my_exception = MyException(
                error_message=f"Error creating linear layer: {e}",
                error_details= sys,
            )
            print(my_exception)
        
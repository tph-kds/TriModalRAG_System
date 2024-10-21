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

        self.weights = self._weights_fusion()

    def forward(
        self, text: torch.Tensor, image: torch.Tensor, audio: torch.Tensor
    ) -> torch.Tensor:
        print("Hung")
        none_count = 0
        if text is None:
            none_count += 1
            text = 0
        if image is None:
            none_count += 1
            image = 0
        if audio is None:
            none_count += 1
            audio = 0

        weighted_sum = (
            (self.weights[0] * text)
            + (self.weights[1] * image)
            + (self.weights[2] * audio)
        )

        return (
            weighted_sum / (torch.sum(self.weights) - none_count)
            if torch.sum(self.weights) - none_count > 0
            else 0
        )

    def _weights_fusion(self) -> nn.Parameter:
        try:
            weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

            return weights

        except Exception as e:
            logger.log_message("warning", f"Error creating weighted fusion: {e}")
            my_exception = MyException(
                error_message=f"Error creating weighted fusion: {e}",
                error_details=sys,
            )
            print(my_exception)

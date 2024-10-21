import os
import sys
from typing import List, Optional
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
        self.dim_text = self.config.dim_text  # 512
        self.dim_image = self.config.dim_image  # 512
        self.dim_sound = self.config.dim_sound  # 512
        self.dim_shared = self.config.dim_shared  # 512
        self.device = self.config.device  # cpu
        self.eps = self.config.eps  # 1e-6
        self.bias = self.config.bias  # True
        self.text_proj = nn.Linear(self.dim_text, self.dim_shared)

        # Activation function (optional, can be any activation like ReLU, GELU)
        self.activation = nn.ReLU()

        # Apply linear layers
        self.text_proj = self._linear_layers(self.dim_text)
        self.image_proj = self._linear_layers(self.dim_image)
        self.sound_proj = self._linear_layers(self.dim_sound)

        self._init_weights_and_bias()

        self._layer_norm = self._LayerNorm()

    def forward(self, text, image, sound):
        # # Project each modality to the shared space
        # text_emb = self.activation(self.text_proj(text))
        # image_emb = self.activation(self.image_proj(image))
        # sound_emb = self.activation(self.sound_proj(sound))

        # # Normalize embeddings (optional)
        # text_emb = self._layer_norm(text_emb)
        # image_emb = self._layer_norm(image_emb)
        # sound_emb = self._layer_norm(sound_emb)

        # Project each modality to the shared space
        if text is not None:
            text_emb = self.activation(self.text_proj(text))
            text_emb = self._layer_norm(text_emb)
        else:
            text_emb = None
        if image is not None:
            image_emb = self.activation(self.image_proj(image))
            image_emb = self._layer_norm(image_emb)
        else:
            image_emb = None
        if sound is not None:
            sound_emb = self.activation(self.sound_proj(sound))
            # Normalize embeddings (optional)
            sound_emb = self._layer_norm(sound_emb)
        else:
            sound_emb = None

        return text_emb, image_emb, sound_emb

    def _convert_format_inputs(
        self,
        text_input: Optional[List],
        image_input: Optional[List],
        sound_input: Optional[List],
    ) -> Optional[torch.Tensor]:
        # Convert inputs to tensors if they are lists
        if isinstance(text_input, list):
            text_input = torch.tensor(text_input, device=self.device)
        if isinstance(image_input, list):
            image_input = torch.tensor(image_input, device=self.device)
        if isinstance(sound_input, list):
            sound_input = torch.tensor(sound_input, device=self.device)

        # Ensure the inputs are tensors
        if not isinstance(text_input, torch.Tensor):
            raise TypeError(f"Expected Tensor for text_input, got {type(text_input)}")
        if not isinstance(image_input, torch.Tensor):
            raise TypeError(f"Expected Tensor for image_input, got {type(image_input)}")
        if not isinstance(sound_input, torch.Tensor):
            raise TypeError(f"Expected Tensor for sound_input, got {type(sound_input)}")

        return text_input, image_input, sound_input

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
            # self._init_weights_and_bias()

            proj = nn.Linear(
                in_features=dim_input,
                out_features=self.dim_shared,
                device=self.device,
                dtype=torch.float32,
                bias=self.bias,
            )

            logger.log_message("info", " Initialized Linear layers")

            return proj

        except Exception as e:
            logger.log_message(
                "warning",
                "Failed to load SharedEmbeddingSpace: _linear_layers: " + str(e),
            )
            my_exception = MyException(
                error_message="Failed to load SharedEmbeddingSpace: _linear_layers: "
                + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _LayerNorm(self) -> Optional[nn.LayerNorm]:
        try:
            logger.log_message("info", " Initializing LayerNorm ...")
            layer_norm = nn.LayerNorm(
                self.dim_shared, device=self.device, dtype=torch.float32, eps=self.eps
            )

            logger.log_message("info", " Initialized LayerNorm")

            return layer_norm

        except Exception as e:
            logger.log_message(
                "warning", "Failed to load SharedEmbeddingSpace: _LayerNorm: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to load SharedEmbeddingSpace: _LayerNorm: "
                + str(e),
                error_details=sys,
            )
            print(my_exception)

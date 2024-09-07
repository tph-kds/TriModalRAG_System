import os
import sys
import numpy as np
import torch
from typing import Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ImageEmbeddingArgumentsConfig

# import clip
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


class ImageEmbedding:
    def __init__(self, config: ImageEmbeddingArgumentsConfig):
        super(ImageEmbedding, self).__init__()
        self.config = config
        self.image_data = self.config

        self.pretrained_model_name = self.image_data.pretrained_model_name
        self.device = self.image_data.device
        self.output_hidden_states=self.image_data.output_hidden_states
        self.output_attentions= self.image_data.output_attentions
        self.return_dict= self.image_data.return_dict
        self.revision = self.image_data.revision
        self.use_safetensors = self.image_data.use_safetensors
        self.ignore_mismatched_sizes = self.image_data.ignore_mismatched_sizes
        self.return_tensors = self.image_data.return_tensors
        self.return_overflowing_tokens = self.image_data.return_overflowing_tokens
        self.return_special_tokens_mask = self.image_data.return_special_tokens_mask



    def _get_model(self) -> Optional[CLIPModel]:
        try:
            logger.log_message("info", "Getting model started.")
            # Load the model and processor
            clipmodel = CLIPModel.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name,
                                                ignore_mismatched_sizes=self.ignore_mismatched_sizes,
                                                use_safetensors=self.use_safetensors,
                                                attn_implementation="eager",
                                                )

            clipmodel.to(self.device)
            clipmodel.eval()

            logger.log_message("info", "Getting model completed successfully.")
            return clipmodel
        
        except Exception as e:
            logger.log_message("warning", "Failed to get model: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get model: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_processor(self) -> Optional[CLIPProcessor]:
        try:
            logger.log_message("info", "Getting processor started.")
            image_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name,
                                                revision=self.revision,
                                                )

            logger.log_message("info", "Getting processor completed successfully.")
            return image_processor
        
        except Exception as e:
            logger.log_message("warning", "Failed to get processor: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get processor: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _preprocess_image(self, converted_image) -> Optional[torch.Tensor]:
        # Load and preprocess image
        image = converted_image.convert("RGB")
        processor_obj = self._get_processor()
        return processor_obj(images=image, 
                               return_tensors=self.return_tensors, 
                               return_overflowing_tokens=self.return_overflowing_tokens,
                               return_special_tokens_mask=self.return_special_tokens_mask).pixel_values
            
    def get_features(self, image_tensor) -> Optional[np.ndarray]:
        try:
            logger.log_message("info", "Getting features Images started.")
            # Move tensors to device
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                # Extract image features
                image_model = self._get_model()
                image_features = image_model.get_image_features(pixel_values=image_tensor,
                                                                output_hidden_states=self.output_hidden_states,
                                                                output_attentions=self.output_attentions,
                                                                return_dict=self.return_dict,
                                                                )

                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Move features to CPU
                image_features = image_features.to(self.device)

            logger.log_message("info", "Getting features Images completed successfully.")
            return image_features.cpu().numpy()   
        
        except Exception as e:
            logger.log_message("warning", "Failed to get features Images: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get features Images: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def embedding_image(self, image_data) -> Optional[np.ndarray]:
        try: 
            logger.log_message("info", "Embedding Images started.")
            # Preprocess image
            image_tensor = self._preprocess_image(image_data)

            # Get image features
            image_features = self.get_features(image_tensor)

            logger.log_message("info", "Embedding Images completed successfully.")
            return image_features

        except Exception as e:
            logger.log_message("warning", "Failed to Embedding Images: " + str(e))
            my_exception = MyException(
                error_message = "Failed to Embedding Images: " + str(e),
                error_details = sys,
            )
            print(my_exception) 
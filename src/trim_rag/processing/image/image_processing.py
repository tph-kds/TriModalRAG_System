
import os
import sys
import io
import numpy as np

from PIL import Image
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ImageDataTransformArgumentsConfig
from torchvision.transforms import (
    ToTensor,
    RandomHorizontalFlip,
    RandomRotation,
    RandomResizedCrop,
    ColorJitter,
    Compose, 

)


class ImageTransform:
    def __init__(self, 
                 config: ImageDataTransformArgumentsConfig, 
                 image_path: str= None
                 ):
        
        super(ImageTransform, self).__init__()
        self.config = config
        self.image_data = self.config
        self.image_path = image_path
        self.size = self.image_data.size
        self.rotate = self.image_data.rotate
        self.horizontal_flip = self.image_data.horizontal_flip
        self.rotation = self.image_data.rotation
        self.brightness = self.image_data.brightness
        self.contrast = self.image_data.contrast
        self.scale = self.image_data.scale
        self.ratio = self.image_data.ratio
        self.saturation = self.image_data.saturation
        self.hue = self.image_data.hue
        self.format = self.image_data.format




    def _resize_image(self) -> Optional[Image.Image]:
        try:
            image = Image.open(self.image_path)
            resized_image = image.resize((self.size, self.size))
            return resized_image

        except Exception as e:
            logger.log_message("warning", "Failed to resize image: " + str(e))
            my_exception = MyException(
                error_message = "Failed to resize image: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _normalize_image(self, image) -> Optional[Image.Image]:
        try:
            image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

            return image_array

        except Exception as e:
            logger.log_message("warning", "Failed to normalize image: " + str(e))
            my_exception = MyException(
                error_message = "Failed to normalize image: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _augment_image(self, image) -> Optional[Image.Image]:
        try:
            augmentation = Compose([
                RandomHorizontalFlip(),
                RandomRotation(self.rotation),
                RandomResizedCrop(self.size, 
                                  scale=(self.scale, self.scale+ 0.2), 
                                  ratio=(self.ratio, self.ratio)),
                ColorJitter(brightness=self.brightness, 
                            contrast=self.contrast, 
                            saturation=self.saturation, 
                            hue=self.hue
                            ),
            ])
            augmented_image = augmentation(image)
            return augmented_image

        except Exception as e:
            logger.log_message("warning", "Failed to augment image: " + str(e))
            my_exception = MyException(
                error_message = "Failed to augment image: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _convert_format(self, image) -> Optional[Image.Image]:
        try:
            converted_image = image.convert('RGB')  # Ensure RGB format
            image_bytes = io.BytesIO()
            converted_image.save(image_bytes, format=self.format)
            image_bytes.seek(0)
            return Image.open(image_bytes)

        except Exception as e:
            logger.log_message("warning", "Failed to convert image format: " + str(e))
            my_exception = MyException(
                error_message = "Failed to convert image format: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def image_processing(self) -> None:
        try: 
            resized_image = self._resize_image()
            augmented_image = self._augment_image(resized_image)
            converted_image = self._convert_format(augmented_image)
            normalized_image = self._normalize_image(converted_image)

            return normalized_image

        except Exception as e:
            logger.log_message("warning", "Failed to process image: " + str(e))
            my_exception = MyException(
                error_message = "Failed to process image: " + str(e),
                error_details = sys,
            )
            print(my_exception)




        
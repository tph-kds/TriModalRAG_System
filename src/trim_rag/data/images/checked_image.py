import os
from typing import List
import numpy as np

from PIL import Image
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger


class DuplicateImageProcessing:
    def __init__(self, image_destinations):
        super(DuplicateImageProcessing, self).__init__()
        # self.image1_path = image1_path
        # self.image2_path = image2_path
        self.image_destinations = image_destinations
        self.checkpoints: List[int] = []

    def _load_image(self, image_path) -> np.ndarray:
        """Load an image and convert it to grayscale for comparison."""
        with Image.open(image_path) as img:
            img = img.convert("L")  # Convert to grayscale
            return np.array(img)  # Convert to a NumPy array

    def check_compare_images(self, image1_path, image2_path) -> bool:
        """Compare two images and return whether they are identical."""
        img1 = self._load_image(image1_path)
        img2 = self._load_image(image2_path)

        # Check if the images are of the same size
        if img1.shape != img2.shape:
            return False

        # Compare pixel values
        return np.array_equal(img1, img2)

    def delete_duplicate_images(self, image1_path, image2_path) -> bool:
        """Compare two images and delete the second image if they are identical."""
        are_identical = self.check_compare_images(image1_path, image2_path)
        if are_identical:
            # logger.log_message("info", "The images are identical. Deleting the second image.")
            os.remove(image2_path)
            # logger.log_message("info", "The second image has been deleted.")
            return True
        else:
            return False

    def run_check_duplicate_images(
        self,
    ) -> List[int]:
        """Compare two images and return whether they are identical."""
        for link_1 in self.image_destinations[:-1]:
            for link_2 in self.image_destinations[
                self.image_destinations.index(link_1) + 1 :
            ]:
                duplicate = self.check_compare_images(link_1, link_2)
                if duplicate:
                    self.checkpoints.append(self.image_destinations.index(link_2))

        return self.checkpoints

import os
from pathlib import Path
import sys
import io
import base64
from PIL import Image
import pandas as pd
from typing import Optional, List

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import ImagePrepareDataQdrantArgumentsConfig
from qdrant_client import models
# from src.trim_rag.processing.image.image_processing import ImageTransform

from src.config_params import ROOT_PROJECT_DIR
from src.trim_rag.utils import load_csv


### Handle on all files of this folder
class ImageQdrantDB:
    def __init__(self, config: ImagePrepareDataQdrantArgumentsConfig):
        super(ImageQdrantDB, self).__init__()
        self.config = config
        self.image_dir = self.config.image_dir
        self.format = self.config.format # "*.png"
        self.path_description = self.config.path_description

        self._image_urls: List = []
        self._types: List = []
        self._base64_strings: List = []
        self._descriptions: List = []

    def _create_image_url(self, image_p) -> Optional[str]:
        try:
            logger.log_message("info", "Creating image url started.")

            image_url = os.path.basename(image_p)

            logger.log_message("info", "Creating image url completed.")

            return image_url

        except Exception as e:
            logger.log_message("warning", "Failed to create image url: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create image url: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _create_base64_strings(self, image_path: str) -> str:
        # Convert image to base64 string
        def open_image(image_path: str) -> Image.Image:
            img = Image.open(image_path)
            return img
            
        def convert_to_base64(image: Image.Image) -> str:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format=self.format.split('.')[-1])
            image_bytes.seek(0)
            return base64.b64encode(image_bytes.getvalue()).decode('utf-8')

        try:
            logger.log_message("info", "Creating base64 string started.")
            image = open_image(image_path)
            image = image.resize((224, 224))
            # print(image)
            base64_string = convert_to_base64(image)

            # print(base64_string)

            logger.log_message("info", "Creating base64 string completed.")

            return base64_string

        except Exception as e:
            logger.log_message("warning", "Failed to create base64 string: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create base64 string: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _create_types(self) -> Optional[str]:
        try:
            logger.log_message("info", "Creating type started.")

            type = "weather"

            logger.log_message("info", "Creating type completed.")

            return type

        except Exception as e:
            logger.log_message("warning", "Failed to create type: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create type: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _create_description(self, 
                            type: str,
                            url: str) -> Optional[str]:
        try:
            logger.log_message("info", "Creating description started.")
            df_path = ROOT_PROJECT_DIR / self.path_description
            df = load_csv(Path(df_path))
            df = df[df["type"] == type]
            desc = df[df["path"] == url]["describe"]
            # convert series to string
            desc = desc.values[0]

            logger.log_message("info", "Creating description completed.")

            return desc

        except Exception as e:
            logger.log_message("warning", "Failed to create description: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create description: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def create_pyload(self, processing_embedding) -> Optional[dict]:
        try:
            logger.log_message("info", "Creating pyload started for preparing upload to qdrant.")
            path_image_embeddings = ROOT_PROJECT_DIR / self.image_dir

            for image_p in path_image_embeddings.glob(self.format):
                # image_url = self._create_image_url(image_p)
                
                type = self._create_types()
                # print(image_p)
                base64_string = self._create_base64_strings(image_p)
                path_image = os.path.basename(str(image_p))
                url = self.image_dir + "/" + path_image
                desc = self._create_description("image", url)

                self._image_urls.append(image_p)
                self._types.append(type)
                self._base64_strings.append(base64_string)
                self._descriptions.append(desc)
            # print(self._image_urls)
            # print(len(self._image_urls))
            # print(self._types)
            # print(len(self._types))
            # print(len(self._base64_strings))

            pyloads = pd.DataFrame({"image_url": self._image_urls,
                                    "type": self._types,
                                    "base64": self._base64_strings,
                                    "description": self._descriptions})

            pyload_dicts = pyloads.to_dict(orient="records")

            logger.log_message("info", "Creating pyload completed for preparing upload to qdrant.")
            return pyload_dicts

        except Exception as e:
            logger.log_message("warning", "Failed to create pyload: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create pyload: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    
    def create_records(self, processing_embedding) -> Optional[dict]:
        try:
            logger.log_message("info", "Creating records started for preparing upload to qdrant.")
            payloads = self.create_pyload(processing_embedding)
            records = [
                models.Record(
                    id = idx,
                    payload = payloads[idx],
                    vector = processing_embedding[idx]
                )
                for idx in range(len(payloads))
            ]

            logger.log_message("info", "Creating records completed for preparing upload to qdrant.")
            return records

        except Exception as e:
            logger.log_message("warning", "Failed to create records: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create records: " + str(e),
                error_details = sys,
            )
            print(my_exception)
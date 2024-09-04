import os
import sys
import io
import base64
import pandas as pd
from typing import Optional, List

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import ImagePrepareDataQdrantArgumentsConfig
from qdrant_client import models


### Handle on all files of this folder
class ImageQdrantDB:
    def __init__(self, config: ImagePrepareDataQdrantArgumentsConfig):
        super(ImageQdrantDB, self).__init__()
        self.config = config
        self.image_dir = self.config.image_dir
        self.format = self.config.format # "*.png"

        self._image_urls: List = []
        self._types: List = []
        self._base64_strings: List = []

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

    def _create_base64_strings(self, image) -> Optional[str]:
        # Convert image to base64 string
        def convert_to_base64(image):
            image_bytes = io.BytesIO()
            image.save(image_bytes, format=self.format)
            image_bytes.seek(0)
            return base64.b64encode(image_bytes.getvalue()).decode('utf-8')

        try:
            logger.log_message("info", "Creating base64 string started.")

            base64_string = convert_to_base64(image)

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


    def create_pyload(self, processing_embedding) -> Optional[dict]:
        try:
            logger.log_message("info", "Creating pyload started for preparing upload to qdrant.")

            for image_p in self.image_dir.glob(self.format):
                image_url = self._create_image_url(image_p)
                type = self._create_types()
                base64_string = self._create_base64_strings(processing_embedding)

                self._image_urls.append(image_url)
                self._types.append(type)
                self._base64_strings.append(base64_string)
            

            pyloads = pd.DataFrame.from_records([{"image_url": self._image_urls,
                                                  "type": self._types,
                                                  "base64": self._base64_strings}])

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

            records = [
                models.Record(
                    id = idx,
                    payload = self.create_pyload(processing_embedding)[idx],
                    vector = processing_embedding[idx]
                )
                for idx in range(len(self.create_pyload(processing_embedding)))
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
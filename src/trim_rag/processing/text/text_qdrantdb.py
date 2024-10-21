import os
import sys
import io
import base64
import pandas as pd
from pathlib import Path
from typing import Optional, List


from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import TextPrepareDataQdrantArgumentsConfig

from qdrant_client import models
from src.config_params import ROOT_PROJECT_DIR
from src.trim_rag.utils import convert_tensor_to_list


### Handle on all files of this folder
class TextQdrantDB:
    def __init__(self, config: TextPrepareDataQdrantArgumentsConfig):
        super(TextQdrantDB, self).__init__()
        self.config = config
        self.text_dir = self.config.text_dir
        self.format = self.config.format  # "*.pdf"

        self._text_urls: List = []
        self._types: List = []
        self._token: List = []
        self._titles: List = []

    def _create_text_url(self, text_p) -> Optional[str]:
        try:
            logger.log_message("info", "Creating text url started.")

            text_url = os.path.basename(text_p)

            logger.log_message("info", "Creating text url completed.")

            return text_url

        except Exception as e:
            logger.log_message("warning", "Failed to create text url: " + str(e))
            my_exception = MyException(
                error_message="Failed to create text url: " + str(e),
                error_details=sys,
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
                error_message="Failed to create type: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    # def _create_titles(self) -> Optional[str]:
    #     try:
    #         logger.log_message("info", "Creating title started.")

    #         title = "weather"

    #         logger.log_message("info", "Creating title completed.")

    #         return title

    #     except Exception as e:
    #         logger.log_message("warning", "Failed to create title: " + str(e))
    #         my_exception = MyException(
    #             error_message = "Failed to create title: " + str(e),
    #             error_details = sys,
    #         )
    #         print(my_exception)

    def create_pyload(self, text_embeddings) -> Optional[dict]:
        try:
            logger.log_message(
                "info", "Creating pyload started for preparing upload to qdrant."
            )
            path_text_embeddings = ROOT_PROJECT_DIR / self.text_dir
            input_ids = text_embeddings[2]
            input_ids_to_qdrant = convert_tensor_to_list(input=input_ids)
            flattened = [item for sublist in input_ids_to_qdrant for item in sublist]
            # print(len(flattened))

            for i, text_p in enumerate(path_text_embeddings.glob(self.format)):
                # text_url = self._create_text_url(text_p)
                titles = self._create_text_url(text_p)
                for _ in range(text_embeddings[3][i]):
                    type = self._create_types()

                    self._text_urls.append(text_p)
                    self._types.append(type)
                    self._titles.append(titles)

            # print(len(self._text_urls))
            # print(len(self._types))
            # print(len(self._titles))
            # print(text_embeddings[1])
            # print(len(input_ids_to_qdrant))
            # print(len(input_ids_to_qdrant[0]))
            # print(len(input_ids_to_qdrant[0][0]))
            # print(len(flattened))
            # print(flattened)
            # print(input_ids_to_qdrant)
            # print(text_embeddings[1])
            # print(input_ids)
            pyloads = pd.DataFrame(
                {
                    "text_url": self._text_urls,
                    "type": self._types,
                    "title": self._titles,
                    # "token": text_embeddings[1],
                    "input_ids": flattened,
                }
            )

            pyload_dicts = pyloads.to_dict(orient="records")

            logger.log_message(
                "info", "Creating pyload completed for preparing upload to qdrant."
            )
            return pyload_dicts

        except Exception as e:
            logger.log_message("warning", "Failed to create pyload: " + str(e))
            my_exception = MyException(
                error_message="Failed to create pyload: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def create_records(
        self,
        processing_embedding: Optional[List],
    ) -> Optional[dict]:
        try:
            logger.log_message(
                "info", "Creating records started for preparing upload to qdrant."
            )
            payloads = self.create_pyload(processing_embedding)
            # print(payloads)
            # print("Hung")
            records = [
                models.Record(
                    id=idx, payload=payloads[idx], vector=processing_embedding[0][idx]
                )
                for idx in range(len(payloads))
            ]

            logger.log_message(
                "info", "Creating records completed for preparing upload to qdrant."
            )
            return records

        except Exception as e:
            logger.log_message("warning", "Failed to create records: " + str(e))
            my_exception = MyException(
                error_message="Failed to create records: " + str(e),
                error_details=sys,
            )
            print(my_exception)

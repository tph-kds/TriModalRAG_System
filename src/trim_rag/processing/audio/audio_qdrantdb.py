import os
import sys
import io
import base64
import pandas as pd
from typing import Optional, List

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import AudioPrepareDataQdrantArgumentsConfig
from qdrant_client import models


### Handle on all files of this folder
class AudioQdrantDB:
    def __init__(self, config: AudioPrepareDataQdrantArgumentsConfig):
        super(AudioQdrantDB, self).__init__()
        self.config = config
        self.audio_dir = self.config.audio_dir


        self._audio_urls: List = []
        self._types: List = []
        self._names: List = []


    def _create_audio_url(self, audio_p) -> Optional[str]:
        try:
            logger.log_message("info", "Creating audio url started.")

            audio_url = os.path.basename(audio_p)

            logger.log_message("info", "Creating audio url completed.")

            return audio_url

        except Exception as e:
            logger.log_message("warning", "Failed to create audio url: " + str(e))
            my_exception = MyException(
                error_message = "Failed to create audio url: " + str(e),
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

            for audio_p in self.audio_dir.glob(self.format):
                audio_url = self._create_audio_url(audio_p)
                type = self._create_types()
                extract_name = os.path.basename(audio_p.split(".")[0])

                self._audio_urls.append(audio_url)
                self._types.append(type)
                self._names.append(extract_name)
            

            pyloads = pd.DataFrame.from_records([{"audio_url": self._audio_urls,
                                                  "type": self._types,
                                                  "base64": self._names}])

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
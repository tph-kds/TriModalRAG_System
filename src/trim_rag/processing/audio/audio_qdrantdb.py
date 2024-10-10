import os
from pathlib import Path
import sys
import io
import base64
import pandas as pd
from typing import Optional, List

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.config_params import ROOT_PROJECT_DIR


from src.trim_rag.config import AudioPrepareDataQdrantArgumentsConfig
from qdrant_client import models
from src.trim_rag.utils import load_csv


### Handle on all files of this folder
class AudioQdrantDB:
    def __init__(self, config: AudioPrepareDataQdrantArgumentsConfig):
        super(AudioQdrantDB, self).__init__()
        self.config = config
        self.audio_dir = self.config.audio_dir

        self.path_description = self.config.path_description


        self._audio_urls: List = []
        self._types: List = []
        self._names: List = []
        self.format = self.config.format # "*.mp3"
        self._descriptions: List = []


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

    def _create_description(self, 
                            type: str,
                            url: str) -> Optional[str]:
        try:
            logger.log_message("info", "Creating description started.")
            df_path = ROOT_PROJECT_DIR / self.path_description
            
            df = load_csv(Path(df_path))
            df = df[df["type"] == type]
            desc = df[df["path"] == url]["describe"]
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
            path_audio_embeddings = ROOT_PROJECT_DIR / self.audio_dir

            for audio_p in path_audio_embeddings.glob(self.format):
                # audio_url = self._create_audio_url(audio_p)
                type = self._create_types()
                extract_name = os.path.basename(str(audio_p).split(".")[0])
                # extract_name = os.path.basename(audio_p.split(".")[0])
                path_audio = os.path.basename(str(audio_p))
                url = self.audio_dir + "/" + path_audio
                desc = self._create_description("audio", url)

                self._audio_urls.append(audio_p)
                self._types.append(type)
                self._names.append(extract_name)
                self._descriptions.append(desc)
            
            # print(self._audio_urls)
            # print(len(self._audio_urls))
            # print(self._types)
            # print(len(self._types))
            # print(self._names)
            # print(len(self._names))
            pyloads = pd.DataFrame({"audio_url": self._audio_urls,
                                    "type": self._types,
                                    "name": self._names,
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
            print(processing_embedding.shape)
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
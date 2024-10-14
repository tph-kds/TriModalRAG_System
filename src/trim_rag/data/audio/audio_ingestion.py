import os
import sys
import wget
import requests
import xml.etree.ElementTree as ET

from typing import Optional, List, Dict, Any
from src.trim_rag.utils.config import read_yaml, create_directories
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config import (
    ConfiguarationManager,
    TextDataIngestionArgumentsConfig,
    ImageDataIngestionArgumentsConfig,
    AudioDataIngestionArgumentsConfig,
)




class AudioIngestion:
    def __init__(self, config: AudioDataIngestionArgumentsConfig, access_key: str):
        
        super(AudioIngestion, self).__init__()
        self.config = config
        self.query = config.query
        self.max_results = config.max_results
        self.destination = config.destination
        self.access_key = access_key
        self.url = config.url

        self.headers = {
            'Authorization': f'Token {self.access_key}'
        }
        self.params = {
            'query': self.query,
            'page_size': self.max_results
        }

        self.results = self._search_freesound()
        self.sounds_dict: Dict[int, Dict[str, Any]] = {}

    def audio_ingestion(self) -> None:

        for idx, result in enumerate(self.results['results']):
            id = result['id']
            name = result['name']
            tags = result["tags"]
            self.sounds_dict[idx] = {
                'id': id,
                'name': name,
                'tags': tags,
            }
        # print(sounds_dict)
        logger.log_message("info", "Crawled " + str(len(self.sounds_dict)) + " sounds information.")

        try:
            logger.log_message("info", "Start downloading audio files...")
            for sound_id, sound_item in self.sounds_dict.items():
                # print(sound_item)
                link_download = self._download_audio(sound_item["id"])
                # print(link_download)
                # break
                # print( link_download['previews']["preview-hq-mp3"])
                audio_url = link_download['previews']['preview-hq-mp3']  # URL for high-quality preview
                description = link_download['description']
                self.sounds_dict[idx] = {
                    'id': id,
                    'name': name,
                    'tags': tags,
                    'audio_url': audio_url,
                    'description' : description,
                }
                print(f"{idx}. Downloading from: {audio_url}")
                logger.log_message("info", f"{idx}. Downloading from: {audio_url}")
                # Download the audio file
                    # Open the destination file in binary write mode
                get_sound = requests.get(audio_url, stream=True)
                sound_name = os.path.join(self.destination, f"/{self.query}_audio_{sound_id}.mp3")
                with open(sound_name, 'wb') as file:
                    # Write the content in chunks
                    for chunk in get_sound.iter_content(chunk_size=8192):
                        file.write(chunk)

                # print(f"Sound downloaded successfully and saved to audio_{sound_id}")
                logger.log_message("info", f"Sound downloaded successfully and saved to audio_{sound_id}")
            

            logger.log_message("info", "Downloaded audio: " + str(len(self.sounds_dict)) + " audio files successfully.")
            return self.sounds_dict

        except Exception as e:
            logger.log_message("warning", "Failed to download audio: " + str(e))
            my_exception = MyException(
                error_message = "Failed to download audio: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _search_freesound(self) -> Optional[Dict[str, Any]]:

        response = requests.get(self.url, headers=self.headers, params=self.params)

        if response.status_code == 200:
            logger.log_message("info", "Fetched sounds successfully.")
            return response.json()
        else:
            # print(f"Failed to retrieve sounds: {response.status_code}")
            logger.log_message("warning", "Failed to retrieve sounds: " + str(response.status_code))
            my_exception = MyException(
                error_message = "Failed to retrieve sounds: " + str(response.status_code),
                error_details = sys,
            )
            print(my_exception)
            return None

    def _download_audio(self, id, stream=True) -> Optional[Dict[str, Any]]:
        url = f'https://freesound.org/apiv2/sounds/{id}' #https://freesound.org/apiv2/sounds/123456/

        response = requests.get(url, headers=self.headers, stream=stream)

        if response.status_code == 200:
            return response.json()
        else:
            # print(f"Failed to download audio: {response.status_code}")
            logger.log_message("warning", "Failed to download audio: " + str(response.status_code))
            my_exception = MyException(
                error_message = "Failed to download audio: " + str(response.status_code),
                error_details = sys,
            )
            print(my_exception)







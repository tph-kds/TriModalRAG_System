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

)




class ImageIngestion:
    def __init__(self, config: ImageDataIngestionArgumentsConfig, access_key: str):
        
        super(ImageIngestion, self).__init__()
        self.config = config
        self.query = config.query
        self.max_results = config.max_results
        self.destination = config.destination
        self.access_key = access_key
        self.url = config.url

        self.image_links: List[str] = []
        self.params = {
            'query': self.query,          # Search query
            'client_id': self.access_key, # Your Unsplash Access Key
            'per_page': self.max_results     # Number of results per page
        }
        self.images = self._get_unsplash_images()


    def image_ingestion(self) -> None:
        for idx, link in enumerate(self.image_links):
            try:
                response = requests.get(link)
                if response.status_code == 200:
                    with open(f"{self.destination}/weather_images_{idx}", 'wb') as file:
                        file.write(response.content)

            except Exception as e:
                logger.log_message("warning", "Failed to download image: " + str(e))
                my_exception = MyException(
                    error_message = "Failed to download image: " + str(e),
                    error_details = sys,
                )
                print(my_exception)
        
        logger.log_message("info", "Downloaded image: " + str(len(self.image_links)) + " images successfully.")


    def _get_unsplash_images(self) -> Optional[Dict[str, Any]]:

        response = requests.get(self.url, params=self.params)

        if response.status_code == 200:
            data = response.json()
            images = data.get('results', [])
            return images
        else:
            print(f"Failed to retrieve images: {response.status_code}")
            return None

    def _print_image_urls(self) -> None:
        try:
            if self.images:
                for idx, image in enumerate(self.images, start=1):
                    urls = image.get('urls', {})
                    # print(f"{idx}. {urls.get('regular', 'No URL available')}")
                    logger.log_message("info", f"{idx}. {urls.get('regular', 'No URL available')}")
                    self.image_links.append(urls.get('regular', 'No URL available'))
            else:
                logger.log_message("warning", "No images found.")

        except Exception as e:
            logger.log_message("warning", "Failed to print image URLs: " + str(e))
            my_exception = MyException(
                error_message = "Failed to print image URLs: " + str(e),
                error_details = sys,
            )
            print(my_exception)



    

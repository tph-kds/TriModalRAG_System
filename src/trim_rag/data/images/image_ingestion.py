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
        self.max_pages = config.max_pages
        self.destination = config.destination
        self.access_key = access_key
        self.url = config.url

        self.params: Dict[str, Any] = {}
        self.images: List[Dict[str, Any]] = []

        self.image_links: List[str] = []
        self.image_destination: List[str] = []

    def _get_params(self, page) -> Dict[str, Any]:
        self.params = {
            'query': self.query,          # Search query
            'client_id': self.access_key, # Your Unsplash Access Key
            'per_page': self.max_results,     # Number of results per page
            'page': page                   # Page number
        }
        return self.params

    def image_ingestion(self) -> None:
        # Crawling images link from Unsplash
        self._print_image_urls()
        # Remove duplicate links
        self.image_links = list(dict.fromkeys(self.image_links))
        # Download images to destination folder
        for idx, link in enumerate(self.image_links):
            try:
                response = requests.get(link)
                if response.status_code == 200:
                    self.image_destination.append(f"{self.destination}/{self.query}_images_{idx}.png")
                    # with open(f"{self.destination}/{self.query}_images_{idx}.png", 'wb') as file:
                    with open(self.image_destination[idx], 'wb') as file:
                        file.write(response.content)

            except Exception as e:
                logger.log_message("warning", "Failed to download image: " + str(e))
                my_exception = MyException(
                    error_message = "Failed to download image: " + str(e),
                    error_details = sys,
                )
                print(my_exception)
        logger.log_message("info", "Downloaded image: " + str(len(self.image_links)) + " images successfully.")
        
        return self.image_links, self.image_destination


    def _get_unsplash_images(self) -> Optional[Dict[str, Any]]:
        
        for pagei in range(1, (self.max_results // self.max_pages) + 2):
            self.params = self._get_params(pagei)
            response = requests.get(self.url, params=self.params)

            if response.status_code == 200:
                data = response.json()
                # images = data.get('results', [])
                self.images.extend(data.get('results', []))
                if len(self.images) >= self.max_results:
                    self.images = self.images[:self.max_results]  # Limit to the required total_images
                    return self.images
            else:
                # print(f"Failed to retrieve images: {response.status_code}")
                logger.log_message("warning", "Failed to retrieve images: " + str(response.status_code))
                my_exception = MyException(
                    error_message = "Failed to retrieve images: " + str(response.status_code),
                    error_details = sys,
                )
                print(my_exception)
                return None

    def _print_image_urls(self) -> None:
        try:
            self.images = self._get_unsplash_images()
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



    

import os
import sys
import wget
import requests
import xml.etree.ElementTree as ET

from typing import Optional, List, Dict, Any
from src.trim_rag.utils.config import read_yaml, create_directories
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config import ConfiguarationManager, TextDataIngestionArgumentsConfig



class TextIngestion:
    def __init__(self, config: TextDataIngestionArgumentsConfig):
        
        super(TextIngestion, self).__init__()
        self.config = config
        self.query = config.query
        self.max_results = config.max_results
        self.destination = config.destination
        # Construct the request URL
        self.url = f"http://export.arxiv.org/api/query?search_query={self.query}&max_results={self.max_results}"
        
        self.papers, self.response = self._text_ingestion()
        self.infos = {}

    def _text_ingestion(self) -> Optional[None, [ET.Element, Any]]:
        """
        This function performs text ingestion

        """

        # Make the HTTP GET request
        response = requests.get(self.url)

        if response.status_code == 200:
            # Parse the XML response
            root = ET.fromstring(response.text)
            # Extract the titles of the papers
            return root, response
        else:
            return None
    
    def get_infos(self, papers) -> Optional[Dict[str, Any]]:
        if papers is None:
            logger.log_message("info", "Empty response. No papers found. Please check your query.")
            return None
    # Extract paper titles and authors
        for idx, entry in enumerate(papers.findall('{http://www.w3.org/2005/Atom}entry')):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            authors = [author.find('{http://www.w3.org/2005/Atom}name').text.strip() for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
            links = [link.get('href') for link in entry.findall('{http://www.w3.org/2005/Atom}link') if link.get('rel') == 'alternate']
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            # Save the paper information

            self.infos[idx] = {
                'title': title,
                'authors': [author.replace("\n", "") for author in authors],
                'links': links,
                'summary': summary.replace("\n", ""),
            }

        return self.infos
    
    def download_file(self) -> None:
        try:
            logger.log_message("info", f"Downloading pdf file from: {url}.")
            for idx, info in self.infos.items():
                url = info['links']
                url = url.replace("abs", "pdf")
                dir_name = os.path.join(self.destination, f"/{idx}.")
                name_file = os.path.join(dir_name, (info['title'].replace(" ", "_")).lower() + ".pdf")
                wget.download(url, name_file)

            logger.log_message("info", f"Downloaded pdf file to {self.destination} successfully with {len(self.infos)} files.")
            
            return None

        except Exception as e:
            logger.log_message("warning", f"Error downloading pdf file to {self.destination}.")
            my_exception = MyException(e, sys)
            print(my_exception)

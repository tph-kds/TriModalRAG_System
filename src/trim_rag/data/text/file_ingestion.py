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
        self.infos: Dict[str, Any] = {}

    def _text_ingestion(self) -> Optional[Dict[str, Any]]:
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

    def _get_infos(self, papers) -> Optional[Dict[str, Any]]:
        if papers is None:
            logger.log_message(
                "info", "Empty response. No papers found. Please check your query."
            )
            return None
        # Extract paper titles and authors
        for idx, entry in enumerate(
            papers.findall("{http://www.w3.org/2005/Atom}entry")
        ):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            authors = [
                author.find("{http://www.w3.org/2005/Atom}name").text.strip()
                for author in entry.findall("{http://www.w3.org/2005/Atom}author")
            ]
            links = [
                link.get("href")
                for link in entry.findall("{http://www.w3.org/2005/Atom}link")
                if link.get("rel") == "alternate"
            ]
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            # Save the paper information

            self.infos[idx] = {
                "title": title,
                "authors": [author.replace("\n", "") for author in authors],
                "links": links,
                "summary": summary.replace("\n", ""),
            }

        return self.infos

    def download_file(self) -> None:
        try:
            self.infos = self._get_infos(self.papers)
            logger.log_message("info", f"Downloading pdf file from: {self.url}.")
            for idx, info in self.infos.items():
                links_url = info["links"][0].replace("abs", "pdf")
                title = f"/{idx}_" + info["links"][0].split("/")[-1].replace(".", "_")
                # name_file = os.path.join(self.destination, title)
                dir_root = (
                    os.path.dirname(os.getcwd()).replace("\\", "/")
                    + "/"
                    + os.path.basename(os.getcwd()).split("/")[0]
                    + "/"
                    + self.destination
                )
                file_name = f"{title}.pdf"
                dir_path = (dir_root + file_name).replace("\\", "/")
                # Download the PDF using requests
                logger.log_message(
                    "info", f"Downloading pdf file from: {links_url} at {dir_path}."
                )
                response = requests.get(links_url)

                # Save the file with the desired extension
                with open(dir_path, "wb") as file:
                    file.write(response.content)

                # logger.log_message("info", f"Downloaded pdf file to {dir_path} successfully.")

            logger.log_message(
                "info",
                f"Downloaded pdf file to {self.destination} successfully with {len(self.infos)} files.",
            )

        except Exception as e:
            logger.log_message(
                "warning", f"Error downloading pdf file to {self.destination}."
            )
            my_exception = MyException(e, sys)
            print(my_exception)

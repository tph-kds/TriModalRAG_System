import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import VectorDBArgumentsConfig
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class VectorDB:
    def __init__(self, config: VectorDBArgumentsConfig) -> None:
        super(VectorDB, self).__init__()

        self.config = config


    def store_vector_db(self, embed_vector) -> None:
        # try: 
        #     logger.log_message("info", "Starting to store vectors to Qdrant server...")

        #     # Store embedding
        #     client.upsert(
        #         collection_name=self.name_text_collection,
        #         points=[{
        #             "id": "unique_image_id",
        #             "vector": embed_vector,
        #             "payload": {"type": self.type_text, "file_path": self.file_path}
        #         }]
        #     )
        pass
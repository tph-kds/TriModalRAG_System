import os
import sys
from typing import List

import torch
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import QdrantVectorDBArgumentsConfig
from src.trim_rag.components import QdrantVectorDB
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from src.config_params import QDRANT_DB_URL, QDRANT_API_KEY
from src.trim_rag.embedding import TextEmbedding
from src.trim_rag.config import TextEmbeddingArgumentsConfig
from qdrant_client import models


class Retrieval_VectorStore:
    def __init__(self, config: QdrantVectorDBArgumentsConfig,
                 embed_config: TextEmbeddingArgumentsConfig) -> None:
        self.config = config
        self.client = QdrantVectorDB(config,
                                     QDRANT_API_KEY,
                                     QDRANT_DB_URL
                                     )._connect_qdrant()
        
        # self.embeddings = TextEmbedding(config= embed_config)

    # def get_embedding_vector_store(self) -> HuggingFaceEmbeddings:
    #     # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #     embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased",
    #                                        model_kwargs={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    #                                        )
    #     return embeddings
    def get_embedding_vector_store(self) -> None:
        return None

    def _get_vector_store(self, 
                          name_collection: str) -> QdrantVectorStore:
        try:
            logger.log_message("info", "Starting to get vector store...")
            # self.client = self.client._connect_qdrant()
            embeddings = self.get_embedding_vector_store()
            # embeddings = self.embeddings()
            self.client.create_collection(
                collection_name=name_collection,
                vectors_config=VectorParams(size=768,
                                            distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
                shard_number=2,
            )
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name= name_collection,
                embedding=embeddings,
                

            )
            # Qdrant = QdrantVectorStore.from_documents(
            #     docs,
            #     embeddings,
            #     url=url,
            #     prefer_grpc=True,
            #     api_key=api_key,
            #     collection_name="my_documents",
            # )
            logger.log_message("info", "Retrieved vector store successfully")
            return vector_store

        except Exception as e:
            logger.log_message("warning", f"Error retrieving vector store: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving vector store: {e}",
                error_details= sys,
            )
            print(my_exception)
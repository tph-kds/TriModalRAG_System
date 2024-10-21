import os
import sys
from typing import List, Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import (
    TextRetrievalArgumentsConfig,
    QdrantVectorDBArgumentsConfig,
)

from src.trim_rag.components import QdrantVectorDB
from qdrant_client import QdrantClient
from qdrant_client import models

# from langchain.chains.sequential import SequentialChain
import torch


class TextRetrieval:
    def __init__(
        self,
        config: TextRetrievalArgumentsConfig,
        config_qdrant: QdrantVectorDBArgumentsConfig,
        client: QdrantClient,
    ) -> None:
        super(TextRetrieval, self).__init__()

        self.config = config
        self.client = client
        self.config_qdrant = config_qdrant
        self.name_collection = self.config_qdrant.text_data.collection_text_name

        # custom chain
        self.input_keys = ["text"]
        self.output_keys = ["text_retrieval"]

    def __call__(self, inputs):
        input = inputs["text"]
        vector_text = None
        if input is not None:
            # Retrieve text embeddings
            text_embedding = self.text_retrieval(input)
            vector_text = text_embedding[
                0
            ].vector  # get the vector after retrieval from qdrant
            vector_text = torch.tensor(vector_text)  # convert to tensor

        else:
            vector_text = None

        return {"text_retrieval": vector_text}

    def text_retrieval(self, query_embedding: Optional[List[float]]) -> None:
        try:
            logger.log_message("info", "Starting to retrieve text embeddings...")
            # print(self.client.get_collection(self.name_collection))
            # Retrieve text embeddings

            results = self.client.search(
                collection_name=self.name_collection,
                query_vector=query_embedding.tolist()[0],
                # with_payload=["text"],
                limit=5,  # Get top 5 most similar texts,
                with_vectors=True,
            )
            # results = self.client.query_points(
            #     collection_name=self.name_collection,
            #     query=query_embedding.tolist()[0],
            #     # using= "text",
            #     limit=1,
            # ).points
            # print(results.points)
            # print(results)

            # for hit in hits:
            #     print(hit.payload, "score:", hit.score)
            logger.log_message("info", "Retrieved text embeddings successfully")
            return results

        except Exception as e:
            logger.log_message("warning", f"Error retrieving text embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving text embeddings: {e}",
                error_details=sys,
            )
            print(my_exception)

    def delete_text_embeddings(self, ids) -> None:
        try:
            logger.log_message("info", "Starting to delete text embeddings...")

            # Delete text embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_text_collection,
                points_selector={"ids": ids},
            )
            logger.log_message("info", "Deleted text embeddings successfully")

        except Exception as e:
            logger.log_message("warning", f"Error deleting text embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting text embeddings: {e}",
                error_details=sys,
            )
            print(my_exception)

    def delete_all_text_embeddings(self) -> None:
        try:
            logger.log_message("info", "Starting to delete all text embeddings...")

            # Delete all text embeddings
            self.client.delete(collection_name=self.config_qdrant.name_text_collection)
            logger.log_message("info", "Deleted all text embeddings successfully")

        except Exception as e:
            logger.log_message("warning", f"Error deleting all text embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting all text embeddings: {e}",
                error_details=sys,
            )
            print(my_exception)

    def recreate_text_collection(self) -> None:
        try:
            logger.log_message("info", "Starting to recreate text collection...")

            # Recreate text collection
            self.client.recreate_collection(
                collection_name=self.config_qdrant.name_text_collection
            )
            logger.log_message("info", "Recreated text collection successfully")

        except Exception as e:
            logger.log_message("warning", f"Error recreating text collection: {e}")
            my_exception = MyException(
                error_message=f"Error recreating text collection: {e}",
                error_details=sys,
            )
            print(my_exception)

    ## Function checking assertion
    def assert_text_retrieval(self, results) -> None:
        logger.log_message("info", "Assert text retrieval...")
        # assert len of results
        assert len(results) == 5
        assert len(results[0].matches) == 5
        # assert type of query_embedding is Sequence[float] | Tuple[str, List[float]] | NamedVector | NamedSparseVector | NumpyArray
        for top_k in results:
            if not isinstance(top_k, list):
                raise AssertionError
            for match in top_k:
                if not isinstance(match, dict):
                    raise AssertionError
                if not isinstance(match["id"], str):
                    raise AssertionError
                if not isinstance(match["score"], float):
                    raise AssertionError

        logger.log_message("info", "Assert text retrieval successfully")

        return None

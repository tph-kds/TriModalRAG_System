import os
import sys
from typing import List, Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.retrieval.text_retrieval import TextRetrieval


class RetrieverQuery(TextRetrieval):
    def __init__(self, **kwargs):
        super(RetrieverQuery, self).__init__(**kwargs)

    def as_retriever(
        self,
        collection_name: str,
        query_embedding: Optional[List[float]],
        k: Optional[int] = 5,
    ) -> None:
        try:
            logger.log_message(
                "info", "Starting to retrieve final information embeddings..."
            )

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist()[0],
                # with_payload=["text"],
                limit=k,  # Get top 5 most similar texts,
                with_vectors=True,
            )

            logger.log_message(
                "info", "Retrieved final information embeddings successfully"
            )
            return results

        except Exception as e:
            logger.log_message(
                "warning", f"Error retrieving final information embeddings: {e}"
            )
            my_exception = MyException(
                error_message=f"Error retrieving final information embeddings: {e}",
                error_details=sys,
            )
            print(my_exception)

import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import TextRetrievalArgumentsConfig, QdrantVectorDBArgumentsConfig

from src.trim_rag.components import QdrantVectorDB

class TextRetrieval:
    def __init__(self, 
                 config: TextRetrievalArgumentsConfig,
                 config_qdrant: QdrantVectorDBArgumentsConfig) -> None:
        super(TextRetrieval, self).__init__()

        self.config = config
        self.config_qdrant = config_qdrant
        self.client = QdrantVectorDB(config_qdrant)._connect_qdrant()


    def text_retrieval(self, query_embedding) -> None:
        try: 
            logger.log_message("info", "Starting to retrieve text embeddings...")

            # Retrieve text embeddings
            results = self.client.search(
                collection_name=self.config_qdrant.name_text_collection,
                query_vector=query_embedding,
                limit=5  # Get top 5 most similar texts,

            )
            logger.log_message("info", "Retrieved text embeddings successfully")
            return results
        
        except Exception as e:
            logger.log_message("info", f"Error retrieving text embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving text embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)

    def delete_text_embeddings(self, ids) -> None:
        try: 
            logger.log_message("info", "Starting to delete text embeddings...")

            # Delete text embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_text_collection,
                points_selector= {
                    "ids": ids
                }
            )
            logger.log_message("info", "Deleted text embeddings successfully")

        except Exception as e:
            logger.log_message("info", f"Error deleting text embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting text embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)

    def delete_all_text_embeddings(self) -> None:
        try: 
            logger.log_message("info", "Starting to delete all text embeddings...")

            # Delete all text embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_text_collection
            )
            logger.log_message("info", "Deleted all text embeddings successfully")

        except Exception as e:
            logger.log_message("info", f"Error deleting all text embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting all text embeddings: {e}",
                error_details= sys,
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
            logger.log_message("info", f"Error recreating text collection: {e}")
            my_exception = MyException(
                error_message=f"Error recreating text collection: {e}",
                error_details= sys,
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


        
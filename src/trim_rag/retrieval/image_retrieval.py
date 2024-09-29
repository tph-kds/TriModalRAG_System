import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import ImageRetrievalArgumentsConfig, QdrantVectorDBArgumentsConfig

from src.trim_rag.components import QdrantVectorDB

class ImageRetrieval:
    def __init__(self, 
                 config: ImageRetrievalArgumentsConfig,
                 config_qdrant: QdrantVectorDBArgumentsConfig,
                 client: QdrantVectorDB) -> None:
        super(ImageRetrieval, self).__init__()

        self.config = config
        self.client = client
        self.config_qdrant = config_qdrant
        self.name_collection = self.config_qdrant.image_data.collection_image_name


    def image_retrieval(self, query_embedding) -> None:
        try: 
            logger.log_message("info", "Starting to retrieve image embeddings...")

            # Retrieve image embeddings
            # results = self.client.search(
            #     collection_name=self.name_collection,
            #     query_vector=(self.name_collection, query_embedding),
            #     limit=5  # Get top 5 most similar images,

            # )
            results = self.client.query_points(
                collection_name=self.name_collection,
                query=query_embedding.tolist(),
                limit=3,
            ).points
            logger.log_message("info", "Retrieved image embeddings successfully")
            return results
        
        except Exception as e:
            logger.log_message("warning", f"Error retrieving image embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving image embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)

    def delete_image_embeddings(self, ids) -> None:
        try: 
            logger.log_message("info", "Starting to delete image embeddings...")

            # Delete image embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_image_collection,
                points_selector= {
                    "ids": ids
                }
                
            )
            logger.log_message("info", "Deleted image embeddings successfully")

        except Exception as e:
            logger.log_message("warning", f"Error deleting image embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting image embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)

    def delete_all_image_embeddings(self) -> None:
        try: 
            logger.log_message("info", "Starting to delete all image embeddings...")

            # Delete all image embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_image_collection
            )
            logger.log_message("info", "Deleted all image embeddings successfully")

        except Exception as e:
            logger.log_message("warning", f"Error deleting all image embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting all image embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)
    def recreate_image_collection(self) -> None:
        try: 
            logger.log_message("info", "Starting to recreate image collection...")

            # Recreate image collection
            self.client.recreate_collection(
                collection_name=self.config_qdrant.name_image_collection
            )
            logger.log_message("info", "Recreated image collection successfully")

        except Exception as e:
            logger.log_message("warning", f"Error recreating image collection: {e}")
            my_exception = MyException(
                error_message=f"Error recreating image collection: {e}",
                error_details= sys,
            )
            print(my_exception)

    ## Function checking assertion
    def assert_image_retrieval(self, results) -> None:
        logger.log_message("info", "Assert image retrieval...")
        # assert len of results
        assert len(results) == 5
        assert len(results[0].matches) == 5

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
        
        logger.log_message("info", "Assert image retrieval successfully")

        return None


        
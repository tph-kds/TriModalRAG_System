import os
import sys

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import AudioRetrievalArgumentsConfig, QdrantVectorDBArgumentsConfig

from src.trim_rag.components import QdrantVectorDB

class AudioRetrieval:
    def __init__(self, 
                 config: AudioRetrievalArgumentsConfig,
                 config_qdrant: QdrantVectorDBArgumentsConfig,
                 client: QdrantVectorDB) -> None:
        super(AudioRetrieval, self).__init__()

        self.config = config
        self.client = client
        self.config_qdrant = config_qdrant
        self.name_collection = self.config_qdrant.audio_data.collection_audio_name


    def audio_retrieval(self, query_embedding) -> None:
        try: 
            logger.log_message("info", "Starting to retrieve audio embeddings...")

            # Retrieve audio embeddings
            # results = self.client.search(
            #     collection_name=self.name_collection,
            #     query_vector=(self.name_collection, query_embedding),
            #     limit=5  # Get top 5 most similar audios,

            # )
            results = self.client.search(
                collection_name=self.name_collection,
                query_vector=query_embedding.tolist()[0],
                # with_payload=["text"],
                limit=1,  # Get top 5 most similar texts,
                with_vectors=True,

            )
            logger.log_message("info", "Retrieved audio embeddings successfully")
            return results
        
        except Exception as e:
            logger.log_message("warning", f"Error retrieving audio embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving audio embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)

    def delete_audio_embeddings(self, ids) -> None:
        try: 
            logger.log_message("info", "Starting to delete audio embeddings...")

            # Delete audio embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_audio_collection,
                points_selector= {
                    "ids": ids
                }
                
            )
            logger.log_message("info", "Deleted audio embeddings successfully")

        except Exception as e:
            logger.log_message("warning", f"Error deleting audio embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting audio embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)

    def delete_all_audio_embeddings(self) -> None:
        try: 
            logger.log_message("info", "Starting to delete all audio embeddings...")

            # Delete all audio embeddings
            self.client.delete(
                collection_name=self.config_qdrant.name_audio_collection
            )
            logger.log_message("info", "Deleted all audio embeddings successfully")

        except Exception as e:
            logger.log_message("warning", f"Error deleting all audio embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error deleting all audio embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)
    def recreate_audio_collection(self) -> None:
        try: 
            logger.log_message("info", "Starting to recreate audio collection...")

            # Recreate audio collection
            self.client.recreate_collection(
                collection_name=self.config_qdrant.name_audio_collection
            )
            logger.log_message("info", "Recreated audio collection successfully")

        except Exception as e:
            logger.log_message("warning", f"Error recreating audio collection: {e}")
            my_exception = MyException(
                error_message=f"Error recreating audio collection: {e}",
                error_details= sys,
            )
            print(my_exception)

    ## Function checking assertion
    def assert_audio_retrieval(self, results) -> None:
        logger.log_message("info", "Assert audio retrieval...")
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
        
        logger.log_message("info", "Assert audio retrieval successfully")

        return None


        
import os
import sys

from typing import List, Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import QdrantVectorDBArgumentsConfig
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams



class QdrantVectorDB:
    def __init__(self, 
                 config: QdrantVectorDBArgumentsConfig,
                 QDRANT_API_KEY: str,
                 QDRANT_DB_URL: str
                 ) -> None:
        super(QdrantVectorDB, self).__init__()

        self.config = config
        self.host = self.config.qdrant_host
        self.port = self.config.qdrant_port
        # self.api_key = self.config.api_key

        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.QDRANT_DB_URL = QDRANT_DB_URL
        
        self.text_data = self.config.text_data
        self.image_data = self.config.image_data
        self.audio_data = self.config.audio_data
        self.size_text = self.text_data.size_text
        self.size_image = self.image_data.size_image
        self.size_audio = self.audio_data.size_audio
        self.collection_text_name = self.text_data.collection_text_name
        self.collection_image_name = self.image_data.collection_image_name
        self.collection_audio_name = self.audio_data.collection_audio_name




    def _connect_qdrant(self) -> Optional[QdrantClient]:
        try:
            logger.log_message("info", "Connecting to Qdrant server...")
            self.client = QdrantClient(
                # host=self.host,
                # port=self.port,
                url=self.QDRANT_DB_URL,
                api_key=self.QDRANT_API_KEY,
            )
            logger.log_message("info", "Connected to Qdrant server")

            # self.client.wait_connection()

            logger.log_message("info", "Qdrant server connected successfully")

            return self.client
        
        except Exception as e:
            logger.log_message("warning", f"Error connecting to Qdrant server: {e}")
            my_exception = MyException(
                error_message=f"Error connecting to Qdrant server: {e}",
                error_details= sys,
            )
            print(my_exception)

    
    def _init_collection(self, 
                         collection_name: str, 
                         embeddings_lenght: int,
                         ) -> None:
        try:
            logger.log_message("info", "Initializing collection...")

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embeddings_lenght,
                    distance=Distance.COSINE,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
                shard_number=2,
            )
    #             vectors_config=models.VectorParams(
    #     size=1024,
    #     distance=models.Distance.DOT,
    # ),

            logger.log_message("info", "Initializing created successfully")

            # self.client.wait_for_loading_collection()

            return None

        except Exception as e:
            logger.log_message("warning", f"Error Initializing collection: {e}")
            my_exception = MyException(
                error_message=f"Error Initializing collection: {e}",
                error_details= sys,
            )
            print(my_exception)

    def _create_collection(self) -> None:
        try:
            logger.log_message("info", "Creating Text Data Collection collection...")
            self._init_collection(self.collection_text_name, 
                                  self.size_text
                                  )
            
            logger.log_message("info", "Text Data Collection created successfully")

            logger.log_message("info", "Creating Image Data Collection collection...")
            self._init_collection(self.collection_image_name, 
                                  self.size_image
                                  )

            logger.log_message("info", "Image Data Collection created successfully")
            logger.log_message("info", "Creating Audio Data Collection collection...")
            self._init_collection(self.collection_audio_name, 
                                  self.size_audio
                                  )

            logger.log_message("info", "Audio Data Collection created successfully")

            return None

            
        
        except Exception as e:
            logger.log_message("warning", f"Error creating collection: {e}")
            my_exception = MyException(
                error_message=f"Error creating collection: {e}",
                error_details= sys,
            )
            print(my_exception)

    def _insert_data(self, 
                     text_embeddings: Optional[List], 
                     image_embeddings: Optional[List], 
                     audio_embeddings: Optional[List],
                     content: Optional[List],
                     file_image_path: Optional[List],
                     audio_file_path: Optional[List]
                     ) -> None:
        try:
            logger.log_message("info", "Inserting text data to Qdrant...")
            # Replace with your embedding generation function
            text_ids = list(range(len(text_embeddings)))
            for i in range(len(text_ids)):
                text_id = str(text_ids[i])

                self.qdrant_client.upsert(
                    collection_name=self.collection_text_name,
                    points=[{
                        "id": text_id,
                        "vector": text_embeddings[i],
                        "payload": {"type": "text", "content": content[i]}
                    }]
                )

            logger.log_message("info", "Text data inserted successfully")

            logger.log_message("info", "Inserting image data to Qdrant...")

            image_ids = list(range(len(image_embeddings)))
            for i in range(len(image_ids)):
                image_id = str(image_ids[i])

                self.qdrant_client.upsert(
                    collection_name=self.collection_image_name,
                    points=[{
                        "id": image_id,
                        "vector": image_embeddings[i],
                        "payload": {"type": "image", "file_path": file_image_path[i]}
                    }]
                )

            logger.log_message("info", "Image data inserted successfully")

            logger.log_message("info", "Inserting audio data to Qdrant...")

            audio_ids = list(range(len(audio_embeddings)))
            for i in range(len(audio_ids)):
                audio_id = str(audio_ids[i])

                self.qdrant_client.upsert(
                    collection_name=self.collection_audio_name,
                    points=[{
                        "id": audio_id,
                        "vector": audio_embeddings[i],
                        "payload": {"type": "audio", "file_path": audio_file_path[i]}
                    }]
                    
                )

            logger.log_message("info", "Audio data inserted successfully")

            return None

        except Exception as e:
            logger.log_message("warning", f"Error inserting text data: {e}")
            my_exception = MyException(
                error_message=f"Error inserting text data: {e}",
                error_details= sys,
            )
            print(my_exception)
    
    def _delete_collection(self, collection_name) -> None:
        try:
            logger.log_message("info", f"Deleting collection {collection_name}...")
            self.client = self._connect_qdrant()
            self.client.delete_collection(collection_name)
            logger.log_message("info", "Collection deleted successfully")

            return None

        except Exception as e:
            logger.log_message("warning", f"Error deleting collection: {e}")
            my_exception = MyException(
                error_message=f"Error deleting collection: {e}",
                error_details= sys,
            )
            print(my_exception)

    def _upload_collection(self, 
                           collection_name: str,
                           data: List,
                           batch_size: int,
                           ) -> None:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                logger.log_message("info", "Uploading collection...")
                self.client.upload_collection(collection_name=collection_name, 
                                              vectors=batch)
                logger.log_message("info", f"Uploaded batch {i // batch_size + 1}")

            except Exception as e:
                logger.log_message("warning", f"Error uploading collection: {e}")
                my_exception = MyException(
                    error_message=f"Error uploading collection: {e}",
                    error_details= sys,
                )
                print(my_exception)

    def qdrant_setting_version_1(self, 
                     text_embeddings: Optional[List], 
                     image_embeddings: Optional[List], 
                     audio_embeddings: Optional[List],
                     content: Optional[List],
                     file_image_path: Optional[List],
                     audio_file_path: Optional[List]
                     )  -> None:
        try:   
            logger.log_message("info", "Connecting to Qdrant server...")
            self.client = self._connect_qdrant()
            logger.log_message("info", "Connected to Qdrant server")
            self._create_collection()
            self._insert_data(text_embeddings, 
                              image_embeddings, 
                              audio_embeddings,
                              content,
                              file_image_path,
                              audio_file_path
                              )
            logger.log_message("info", "Qdrant setting completed successfully")

            return None

        except Exception as e:
            logger.log_message("warning", f"Error connecting to Qdrant server: {e}")
            my_exception = MyException(
                error_message=f"Error connecting to Qdrant server: {e}",
                error_details= sys,
            )
            print(my_exception)

    def _upload_records(self, 
                        collection_name: str,  
                        records, 
                        batch_size) -> None:

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                logger.log_message("info", f"Uploading records with {collection_name} name to Qdrant...")
                self.client.upload_records(
                    collection_name=collection_name,
                    records=batch
                )

                logger.log_message("info", f"Uploaded batch {i // batch_size + 1}")


            except Exception as e:
                logger.log_message("warning", f"Error uploading records: {e}")
                my_exception = MyException(
                    error_message=f"Error uploading records: {e}",
                    error_details= sys,
                )
                print(my_exception)

        logger.log_message("info", f"Records with {collection_name} name uploaded successfully")

    def _upload_records_v1(self, 
                        collection_name: str,  
                        records, ) -> None:

        try:
            logger.log_message("info", f"Uploading records with {collection_name} name to Qdrant...")
            self.client.upload_records(
                collection_name=collection_name,
                records=records,
                batch_size=32,
                wait=True
            )

            logger.log_message("info", f"Records with {collection_name} name uploaded successfully")


        except Exception as e:
            logger.log_message("warning", f"Error uploading records: {e}")
            my_exception = MyException(
                error_message=f"Error uploading records: {e}",
                error_details= sys,
            )
            print(my_exception)

    ## Only conect to Qdrant server and if you want to update or add data, you need to update outside of class
    def qdrant_setting_version_2(self) -> None:
        try:   
            logger.log_message("info", "Connecting to Qdrant server...")
            self.client = self._connect_qdrant()
            logger.log_message("info", "Connected to Qdrant server")
            self._create_collection()
            logger.log_message("info", "Qdrant setting completed successfully")

            return None

        except Exception as e:
            logger.log_message("warning", f"Error connecting to Qdrant server: {e}")
            my_exception = MyException(
                error_message=f"Error connecting to Qdrant server: {e}",
                error_details= sys,
            )
            print(my_exception)
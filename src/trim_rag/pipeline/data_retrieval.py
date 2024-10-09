import os
import sys

from typing import List, Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.retrieval import TriModalRetrieval
from src.trim_rag.config import (
    TrimodalRetrievalPipelineArgumentsConfig,
    QdrantVectorDBArgumentsConfig
)
from langchain.chains.sequential import SequentialChain

class DataRetrievalPipeline:

    def __init__(self, config: TrimodalRetrievalPipelineArgumentsConfig,
                 qdrant_config: QdrantVectorDBArgumentsConfig,
                 QDRANT_API_KEY:  Optional[str] = None,
                 QDRANT_DB_URL:  Optional[str] = None,
                   ) -> None:
        super(DataRetrievalPipeline, self).__init__()

        self.config = config
        self.qdrant_config = qdrant_config

        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.QDRANT_DB_URL = QDRANT_DB_URL


    def run_data_retrieval_pipeline(self,
                                    text_embedding_query: Optional[List[float]] = None,
                                    image_embedding_query: Optional[List[float]] = None,
                                    audio_embedding_query: Optional[List[float]] = None
                                    ) -> SequentialChain:
        try: 
            logger.log_message("info", "Running data retrieval pipeline...")
            tri_modal_retrieval = TriModalRetrieval( config=self.config, 
                                                     config_qdrant=self.qdrant_config,
                                                     QDRANT_API_KEY=self.QDRANT_API_KEY,
                                                     QDRANT_DB_URL=self.QDRANT_DB_URL
                                                    )
            retrieved_fusion, retrieved_text, retrieved_image, retrieved_audio = tri_modal_retrieval.trimodal_retrieval(
                text_embedding_query=text_embedding_query,
                image_embedding_query=image_embedding_query,
                audio_embedding_query=audio_embedding_query
            )
            logger.log_message("info", "Data retrieval pipeline completed")
        
            return retrieved_fusion, retrieved_text, retrieved_image, retrieved_audio

        except Exception as e:
            logger.log_message("warning", "Failed to run data retrieval pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data retrieval pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

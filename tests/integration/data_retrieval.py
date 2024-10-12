import sys
from typing import List, Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataRetrievalPipeline
from src.config_params import QDRANT_DB_URL, QDRANT_API_KEY
from src.trim_rag.retrieval import Retrieval_VectorStore, RetrieverQuery
from src.trim_rag.components import QdrantVectorDB



def data_retriever(
    text_embedding_query: Optional[List[float]] = None,
    image_embedding_query: Optional[List[float]] = None,
    audio_embedding_query: Optional[List[float]] = None
):
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING DATA RETRIEVER SCENARIO >>>>>>>>")
        logger.log_message("info", "Data Retriever pipeline started.")
        config_manager = ConfiguarationManager()
        config = config_manager.get_retrieval_config()
        qdrant_config = config_manager.get_qdrant_vectordb_arguments_config()
        embed_config = config_manager.get_data_embedding_arguments_config()
        client = QdrantVectorDB(qdrant_config,
                                     QDRANT_API_KEY,
                                     QDRANT_DB_URL
                                     )._connect_qdrant()
        
        pipeline = DataRetrievalPipeline(config, 
                                         qdrant_config,
                                         QDRANT_API_KEY=QDRANT_API_KEY,
                                         QDRANT_DB_URL=QDRANT_DB_URL
                                         )
        
        retrieved_fusion, retriever_text, retriever_image, retriever_audio = pipeline.run_data_retrieval_pipeline(text_embedding_query=text_embedding_query,
                                                               image_embedding_query=image_embedding_query,
                                                               audio_embedding_query=audio_embedding_query
                                                               )
        # vector_store = Retrieval_VectorStore(qdrant_config, embed_config.text_data)
        vector_store = RetrieverQuery(config=config.trimodal_retrieval.text_retrieval, 
                                      config_qdrant= qdrant_config,
                                      client=client
                                      )
        main_retriever = vector_store.as_retriever(
            collection_name="text", 
            query_embedding=retrieved_fusion, 
            k=5
        )

        logger.log_message("info", "Data Retriever pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA RETRIEVER SCENARIO   >>>>>>>>")
        logger.log_message("info", "")
        return main_retriever, retriever_text, retriever_image, retriever_audio

    except Exception as e:
        logger.log_message("warning", "Failed to run Data Retriever pipeline: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run Data Retriever pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)


if __name__ == "__main__":
    retriever = data_retriever()
    print(retriever)




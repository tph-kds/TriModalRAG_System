import sys
from typing import List, Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.pipeline import DataRetrievalPipeline
from src.config_params import QDRANT_DB_URL, QDRANT_API_KEY
from src.trim_rag.retrieval import Retrieval_VectorStore


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
        pipeline = DataRetrievalPipeline(config, 
                                         qdrant_config,
                                         QDRANT_API_KEY=QDRANT_API_KEY,
                                         QDRANT_DB_URL=QDRANT_DB_URL
                                         )
        
        chain_retrieval, fusion= pipeline.run_data_retrieval_pipeline(text_embedding_query=text_embedding_query,
                                                               image_embedding_query=image_embedding_query,
                                                               audio_embedding_query=audio_embedding_query
                                                               )
        print("Hung")
        vector_store = Retrieval_VectorStore(qdrant_config, embed_config.text_data)
        print("Hung123")
    
        retriever_text = vector_store._get_vector_store(name_collection="retriever_text")
        retriever = retriever_text.as_retriever(search_type="similarity", search_kwargs={"k": 5})


        logger.log_message("info", "Data Retriever pipeline completed successfully.")
        logger.log_message("info", "<<<<<<<<   END DATA RETRIEVER SCENARIO   >>>>>>>>")
        logger.log_message("info", "")
        return chain_retrieval, fusion, retriever

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




import os
import sys

from langchain.chains import SimpleChain
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import TriModalRetrievalArgumentsConfig
from src.trim_rag.retrieval.fusion_mechanism import FusionMechanism
from src.trim_rag.retrieval.text_retrieval import TextRetrieval
from src.trim_rag.retrieval.audio_retrieval import AudioRetrieval
from src.trim_rag.retrieval.image_retrieval import ImageRetrieval

class TriModalRetrieval:
    def __init__(self, config: TriModalRetrievalArgumentsConfig):
        super(TriModalRetrieval, self).__init__()

        self.config = config

    def _init_func(self):
        try:
            logger.log_message("info", "Initializing trimodal retrieval...")
            fusion_mechanism = FusionMechanism(self.config.fusion_mechanism)
            tex_retrieval = TextRetrieval(self.config.text_retrieval)
            img_retrieval = ImageRetrieval(self.config.image_retrieval)
            aud_retrieval = AudioRetrieval(self.config.audio_retrieval)

            return fusion_mechanism, tex_retrieval, img_retrieval, aud_retrieval

        except Exception as e:
            logger.log_message("info", f"Error initializing trimodal retrieval: {e}")
            my_exception = MyException(
                error_message=f"Error initializing trimodal retrieval: {e}",
                error_details= sys,
            )
            print(my_exception)


    def trimodal_retrieval(self) -> SimpleChain:
        try:
            logger.log_message("info", "Starting to retrieve text embeddings...")
            fusion_mechanism, tex_retrieval, img_retrieval, aud_retrieval = self._init_func()

            # Retrieve text embeddings
            text_embedding = tex_retrieval.text_retrieval()
            logger.log_message("info", "Retrieved text embeddings successfully")

            # Retrieve image embeddings
            image_embedding = img_retrieval.image_retrieval(text_embedding)
            logger.log_message("info", "Retrieved image embeddings successfully")

            # Retrieve audio embeddings
            audio_embedding = aud_retrieval.audio_retrieval(text_embedding)
            logger.log_message("info", "Retrieved audio embeddings successfully")

            # Fusion
            fusion = fusion_mechanism(text_embedding, image_embedding, audio_embedding)

            chain = SimpleChain.from_chains([tex_retrieval, img_retrieval, aud_retrieval, fusion])

            return chain

        except Exception as e:
            logger.log_message("info", f"Error retrieving embeddings: {e}")
            my_exception = MyException(
                error_message=f"Error retrieving embeddings: {e}",
                error_details= sys,
            )
            print(my_exception)


            
            


import os
import sys
from typing import List
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger

from src.trim_rag.config  import MultimodalEmbeddingArgumentsConfig, EmbeddingArgumentsConfig
from src.trim_rag.embedding import (TextEmbedding,
    ImageEmbedding,
    AudioEmbedding,
    MultimodalEmbedding,
    SharedEmbeddingSpace,
    CrossModalEmbedding
)

class DataEmbeddingPipeline:
    def __init__(self, 
                 config: MultimodalEmbeddingArgumentsConfig,
                 config_embedding: EmbeddingArgumentsConfig):
        super(DataEmbeddingPipeline, self).__init__()
        self.config = config
        self.config_embedding = config_embedding

        self.text_data = self.config_embedding.text_data
        self.image_data = self.config_embedding.image_data
        self.audio_data = self.config_embedding.audio_data

        self.text_embeddings: List[str] = []
        self.image_embeddings: List[str] = []
        self.audio_embeddings: List[str] = []
        
        self.multimodal_embedding = MultimodalEmbedding(self.config)
        self.shared_embeddings =  SharedEmbeddingSpace(self.config.sharedspace_embedding)


    def run_data_embedding_pipeline(self, type: str = "shared") -> List[str]:
        try:
            logger.log_message("info", "Data embedding pipeline started.")

            self.text_embeddings = self.text_embedding()
            # self.image_embeddings = self.image_embedding()
            # self.audio_embeddings = self.audio_embedding()
            # if type == "all":
            #     embeddings = self._multimodal_embedding()
            #     return embeddings

            # text_new_embeddings, image_new_embeddings, audio_new_embeddings = self.shared_embedding_space()

            # logger.log_message("info", "Data embedding pipeline completed successfully.")
            # return text_new_embeddings, image_new_embeddings, audio_new_embeddings
            return self.text_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run data embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def text_embedding(self, texts: List[str]) -> List[str]:
        try:
            logger.log_message("info", "Text embedding pipeline started.")
            textEmbedding = TextEmbedding(self.text_data)
            for text in texts:
                self.text_embeddings.append(textEmbedding.embedding_text(text))

            logger.log_message("info", "Text embedding pipeline completed successfully.")

            return self.text_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run text embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)
    

    def image_embedding(self, images: List[str]) -> List[str]:
        try:
            logger.log_message("info", "Image embedding pipeline started.")
            imageEmbedding = ImageEmbedding(self.image_data)
            for image in images:
               self.image_embeddings.append(imageEmbedding.embedding_image(image))
            
            logger.log_message("info", "Image embedding pipeline completed successfully.")  

            return self.image_embeddings


        except Exception as e:
            logger.log_message("warning", "Failed to run image embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run image embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def audio_embedding(self, audios: List[str]) -> List[str]:
        try:
            logger.log_message("info", "Audio embedding pipeline started.")
            audioEmbedding = AudioEmbedding(self.audio_data)
            for audio in audios:
                self.audio_embeddings.append(audioEmbedding.audio_embedding(audio))

            logger.log_message("info", "Audio embedding pipeline completed successfully.")

            return self.audio_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run audio embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run audio embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    
    def _multimodal_embedding(self) -> List[str]:
        try:
            logger.log_message("info", "Multimodal embedding pipeline started.")
            multimodalEmbedding = self.multimodal_embedding(self.text_embeddings, 
                                                            self.image_embeddings, 
                                                            self.audio_embeddings)

            logger.log_message("info", "Multimodal embedding pipeline completed successfully.")

            return multimodalEmbedding

        except Exception as e:
            logger.log_message("warning", "Failed to run multimodal embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run multimodal embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def shared_embedding_space(self) -> List[str]:
        try:
            logger.log_message("info", "Shared embedding space pipeline started.")
            sharedEmbeddingSpace = self.shared_embeddings(self.text_embeddings, 
                                                          self.image_embeddings, 
                                                          self.audio_embeddings)
            
            text_new_embeddings, image_new_embeddings, audio_new_embeddings = sharedEmbeddingSpace

            logger.log_message("info", "Shared embedding space pipeline completed successfully.")

            return text_new_embeddings, image_new_embeddings, audio_new_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run shared embedding space pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run shared embedding space pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    



import os
import sys
import torch
from torch import nn

from typing import List, Optional
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
        self.device = self.config_embedding.device

        self.text_embeddings: List[str] = []
        self.image_embeddings: List[str] = []
        self.audio_embeddings: List[str] = []

        self.multimodal_embedding = MultimodalEmbedding(self.config)
        self.shared_embeddings =  SharedEmbeddingSpace(self.config.sharedspace_embedding)


    def run_data_embedding_pipeline(self, 
                                    text: List[str], 
                                    image: List[str], 
                                    audio: List[str], 
                                    type_embedding: str = "shared") -> List[str]:
        try:
            logger.log_message("info", "Data embedding pipeline started.")

            text_embeds = self.text_embedding(texts=text)
            image_embeds = self.image_embedding(images=image)
            audio_embeds = self.audio_embedding(audios=audio)
            print(f" TEXT EMBEDS:  {text_embeds.shape}", 
                  f" IMAGE EMBEDS: {image_embeds.shape}", 
                  f" AUDIO EMBEDS: {audio_embeds.shape}")
            
            if type_embedding == "all":
                embeddings = self._multimodal_embedding()
                return embeddings

            text_new_embeddings, image_new_embeddings, audio_new_embeddings = self.shared_embedding_space(text_embeds,
                                                                                                          image_embeds,
                                                                                                          audio_embeds)

            logger.log_message("info", "Data embedding pipeline completed successfully.")
            return text_new_embeddings, image_new_embeddings, audio_new_embeddings
            # return self.text_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run data embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def text_embedding(self, texts: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Text embedding pipeline started.")
            textEmbedding = TextEmbedding(self.text_data)
            for text in texts:
                self.text_embeddings.append(textEmbedding.embedding_text(text))

            logger.log_message("info", "Text embedding pipeline completed successfully.")
            
            text_tensors = torch.stack(self.text_embeddings)
            text_tensors_flatten = text_tensors.squeeze(1) # torch.Size([n_texts, 512, 768])
            pooled_text_tensors = torch.mean(text_tensors_flatten, dim=1) # torch.Size([n_texts, 768])
            return pooled_text_tensors

        except Exception as e:
            logger.log_message("warning", "Failed to run text embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)
    

    def image_embedding(self, images: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Image embedding pipeline started.")
            imageEmbedding = ImageEmbedding(self.image_data)
            for image in images:
               self.image_embeddings.append(imageEmbedding.embedding_image(image))
            
            logger.log_message("info", "Image embedding pipeline completed successfully.")  
            image_embed = torch.tensor(self.image_embeddings) # torch.Size([n_images, 1, 512])
            pooled_image_embed = torch.mean(image_embed, dim=1) # torch.Size([n_images, 512])
            pooled_image_embed = pooled_image_embed.to(self.device)
            return pooled_image_embed


        except Exception as e:
            logger.log_message("warning", "Failed to run image embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run image embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def audio_embedding(self, audios: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Audio embedding pipeline started.")
            audioEmbedding = AudioEmbedding(self.audio_data)
            for audio in audios:
                self.audio_embeddings.append(audioEmbedding.embedding_audio(audio))

            logger.log_message("info", "Audio embedding pipeline completed successfully.")
            audio_embed = [ae for ae in self.audio_embeddings if ae is not None]
            # print(len(audio_embed))
            audio_embed = torch.tensor(audio_embed) # torch.Size([n_audios, 1, 8169, 768])
            pooled_audio_embed = audio_embed.squeeze(1) # torch.Size([n_audios,  1,  768])
            pooled_audio_embed = torch.mean(pooled_audio_embed, dim=1) # torch.Size([n_audios, 768])
            pooled_audio_embed = pooled_audio_embed.to(self.device)
            return pooled_audio_embed

        except Exception as e:
            logger.log_message("warning", "Failed to run audio embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run audio embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    
    def _multimodal_embedding(self, 
                              text_embeds: torch.Tensor,
                              image_embeds: torch.Tensor,
                              audio_embeds: torch.Tensor
                              ) -> torch.Tensor:
        try:
            logger.log_message("info", "Multimodal embedding pipeline started.")
            multimodalEmbedding = self.multimodal_embedding(text_embeds,
                                                             image_embeds,
                                                             audio_embeds
                                                             )

            logger.log_message("info", "Multimodal embedding pipeline completed successfully.")

            return multimodalEmbedding

        except Exception as e:
            logger.log_message("warning", "Failed to run multimodal embedding pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run multimodal embedding pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def shared_embedding_space(self, 
                               text_embeds: torch.Tensor, 
                               image_embeds: torch.Tensor, 
                               audio_embeds: torch.Tensor
                               ) -> torch.Tensor:
        try:
            logger.log_message("info", "Shared embedding space pipeline started.")
            sharedEmbeddingSpace = self.shared_embeddings(text_embeds, 
                                                          image_embeds, 
                                                          audio_embeds
                                                          )
            
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

    



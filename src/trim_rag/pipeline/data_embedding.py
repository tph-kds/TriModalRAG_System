import os
import sys
import torch
from torch import nn

from typing import List, Optional, Union, Tuple
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger

from src.trim_rag.config import (
    MultimodalEmbeddingArgumentsConfig,
    EmbeddingArgumentsConfig,
)
from src.trim_rag.embedding import (
    TextEmbedding,
    ImageEmbedding,
    AudioEmbedding,
    MultimodalEmbedding,
    SharedEmbeddingSpace,
    CrossModalEmbedding,
)


class DataEmbeddingPipeline:
    def __init__(
        self,
        config: MultimodalEmbeddingArgumentsConfig,
        config_embedding: EmbeddingArgumentsConfig,
    ):
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
        self.shared_embeddings = SharedEmbeddingSpace(self.config.sharedspace_embedding)

    def run_data_embedding_pipeline(
        self,
        text: List[str],
        image: List[str],
        audio: List[str],
        type_embedding: str = "shared",
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            logger.log_message("info", "Data embedding pipeline started.")

            text_embeds = self.text_embedding(texts=text)
            image_embeds = self.image_embedding(images=image)
            audio_embeds = self.audio_embedding(audios=audio)
            # print(f" TEXT EMBEDS:  {text_embeds.shape}",
            #       f" IMAGE EMBEDS: {image_embeds.shape}",
            #       f" AUDIO EMBEDS: {audio_embeds.shape}")

            if type_embedding == "all":
                embeddings = self._multimodal_embedding()
                return embeddings

            (
                text_new_embeddings,
                image_new_embeddings,
                audio_new_embeddings,
            ) = self.shared_embedding_space(text_embeds[0], image_embeds, audio_embeds)
            textnew_embed = (
                text_new_embeddings,
                text_embeds[1],
                text_embeds[2],
                text_embeds[3],
            )
            logger.log_message(
                "info", "Data embedding pipeline completed successfully."
            )
            return textnew_embed, image_new_embeddings, audio_new_embeddings
            # return self.text_embeddings

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run data embedding pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run data embedding pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def text_embedding(self, texts: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Text embedding pipeline started.")
            textEmbedding = TextEmbedding(self.text_data)
            tokens = []
            token_ids = []
            shape_ids = []
            for i, text in enumerate(texts):
                embedding_text, token_list, input_ids = textEmbedding.embedding_text(
                    text
                )
                tokens.append(token_list)
                token_ids.append(input_ids)
                shape_ids.append(embedding_text.shape[0])
                if i == 0:
                    self.text_embeddings.append(embedding_text)
                else:
                    # torch.Size([20, 512, 768]) + torch.Size([18, 512, 768]) = torch.Size([38, 512, 768])
                    self.text_embeddings = torch.cat(
                        (self.text_embeddings[0], embedding_text), dim=0
                    )
                if len(texts) == 1:
                    self.text_embeddings = self.text_embeddings[0]
            logger.log_message(
                "info", "Text embedding pipeline completed successfully."
            )
            text_tensors_flatten = torch.tensor(self.text_embeddings).squeeze(
                1
            )  # torch.Size([n_texts, 512, 768])
            pooled_text_tensors = torch.mean(
                text_tensors_flatten, dim=1
            )  # torch.Size([n_texts, 768])
            text_tensors = (
                pooled_text_tensors,
                tokens,
                token_ids,
                shape_ids,
            )  # 0 : pooled_text_tensors, 1: token_list, 2:input_ids , 3:shape_texts
            return text_tensors

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run text embedding pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run text embedding pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def image_embedding(self, images: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Image embedding pipeline started.")
            imageEmbedding = ImageEmbedding(self.image_data)
            for image in images:
                self.image_embeddings.append(imageEmbedding.embedding_image(image))

            logger.log_message(
                "info", "Image embedding pipeline completed successfully."
            )
            image_embed = torch.tensor(
                self.image_embeddings
            )  # torch.Size([n_images, 1, 512])
            pooled_image_embed = torch.mean(
                image_embed, dim=1
            )  # torch.Size([n_images, 512])
            pooled_image_embed = pooled_image_embed.to(self.device)
            return pooled_image_embed

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run image embedding pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run image embedding pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def audio_embedding(self, audios: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Audio embedding pipeline started.")
            audioEmbedding = AudioEmbedding(self.audio_data)
            for i, audio in enumerate(audios):
                embedding_audio = audioEmbedding.embedding_audio(audio)
                mean_tensor = torch.mean(
                    torch.tensor(embedding_audio), dim=1
                )  # [1, 768]
                if i == 0:
                    self.audio_embeddings.append(mean_tensor)
                    # print(torch.tensor((self.audio_embeddings[0])).shape)
                else:
                    # torch.Size([20, 512, 768]) + torch.Size([18, 512, 768]) = torch.Size([38, 512, 768])
                    self.audio_embeddings = torch.cat(
                        (self.audio_embeddings[0], mean_tensor), dim=0
                    )
                    print(self.audio_embeddings.shape)
                if len(audios) == 1:
                    self.audio_embeddings = self.audio_embeddings[0]

            # print(self.audio_embeddings)
            # print(self.audio_embeddings.shape)

            logger.log_message(
                "info", "Audio embedding pipeline completed successfully."
            )
            # audio_embed = [ae for ae in self.audio_embeddings if ae is not None]
            # print(len(audio_embed))

            # audio_embed = torch.tensor(self.audio_embeddings) # torch.Size([n_audios, 1, 8169, 768])
            # pooled_audio_embed = audio_embed.squeeze(1) # torch.Size([n_audios,  1,  768])
            # pooled_audio_embed = torch.mean(pooled_audio_embed, dim=1) # torch.Size([n_audios, 768])
            pooled_audio_embed = self.audio_embeddings.to(self.device)
            return pooled_audio_embed

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run audio embedding pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run audio embedding pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _multimodal_embedding(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        try:
            logger.log_message("info", "Multimodal embedding pipeline started.")
            multimodalEmbedding = self.multimodal_embedding(
                text_embeds, image_embeds, audio_embeds
            )

            logger.log_message(
                "info", "Multimodal embedding pipeline completed successfully."
            )

            return multimodalEmbedding

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run multimodal embedding pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run multimodal embedding pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def shared_embedding_space(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        try:
            logger.log_message("info", "Shared embedding space pipeline started.")
            sharedEmbeddingSpace = self.shared_embeddings(
                text_embeds, image_embeds, audio_embeds
            )

            (
                text_new_embeddings,
                image_new_embeddings,
                audio_new_embeddings,
            ) = sharedEmbeddingSpace

            logger.log_message(
                "info", "Shared embedding space pipeline completed successfully."
            )

            return text_new_embeddings, image_new_embeddings, audio_new_embeddings

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run shared embedding space pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run shared embedding space pipeline: "
                + str(e),
                error_details=sys,
            )
            print(my_exception)

import os
import sys
import torch
from torch import nn

from typing import List, Optional, Union, Tuple
from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger

from src.trim_rag.config  import (
    MultimodalEmbeddingArgumentsConfig, 
    EmbeddingArgumentsConfig,
    DataTransformationArgumentsConfig
)
from src.trim_rag.embedding import (
    SharedEmbeddingSpace,
)
from src.trim_rag.inference.embed_infer import (
    EmbeddingTextInference, 
    EmbeddingImageInference, 
    EmbeddingAudioInference
)

from src.trim_rag.inference.extract_inputIfer import (
    TextDataInference,
    ImageDataInference,
    AudioDataInference
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
class Inference:
    def __init__(self, 
                 config: MultimodalEmbeddingArgumentsConfig,
                 config_embedding: EmbeddingArgumentsConfig,
                 config_processing: DataTransformationArgumentsConfig) -> None:
        super(Inference, self).__init__()
        self.config = config
        self.config_embedding = config_embedding
        self.config_processing = config_processing

        self.text_data = self.config_embedding.text_data
        self.image_data = self.config_embedding.image_data
        self.audio_data = self.config_embedding.audio_data
        self.device = self.config_embedding.device

        self.text_embeddings: List[str] = []
        self.image_embeddings: List[str] = []
        self.audio_embeddings: List[str] = []
        
        self.processing_text_data = self.config_processing.text_data
        self.root_dir = self.config_processing.root_dir
        self.processed_dir = self.config_processing.processed_dir
        self.text_data_processing = self.config_processing.text_data
        self.image_data_processing = self.config_processing.image_data
        self.audio_data_processing = self.config_processing.audio_data
        self.chunk_size = self.config_processing.chunk_size # 512
        self.chunk_overlap = self.config_processing.chunk_overlap # 64

        self.list_text_processeds: List[int] = []
        self.list_image_processeds: List[int] = []
        self.list_audio_processeds: List[int] = []

        # self.multimodal_embedding = MultimodalEmbedding(self.config)
        self.shared_embeddings =  SharedEmbeddingSpace(self.config.sharedspace_embedding)

    def run_data_processing_pipeline(self,
                                     text_path: str,
                                     image_path: str,
                                     audio_path: str
                                     ) -> Union[Optional[List[float]], 
                                                    Optional[List[str]],
                                                    Optional[List[float]],
                                                    Optional[List[float]]]:
        try:
            logger.log_message("info", "Data processing in the Inference Phase started.")

            # textprocessing = None
            # imageprocessing = None
            # audioprocessing = None
            textprocessing= self.text_processing(text_path)
            imageprocessing = self.image_processing(image_path)
            audioprocessing = self.audio_processing(audio_path)
            # self.data_processing()

            logger.log_message("info", "Data Processing in the Inference Phase completed successfully.")
            return textprocessing, imageprocessing, audioprocessing

        except Exception as e:
            logger.log_message("warning", "Failed to run Data Processing in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run Data Processing in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def text_processing(self, input: str) -> Union[Optional[List[float]], Optional[List[str]]]:
       try:
            logger.log_message("info", "Text processing in the Inference Phase started.")
            # access data folder before transforming
            # dir_textdata = os.getcwd() + "/" + self.text_data.text_dir

            # link_textdata = dir_textdata + "/" + input
            # link_textdata = link_textdata.replace("\\", "/")
            link_textdata = input
            # config_vars = vars(self.text_data_processing)
            textprocessing = TextDataInference(config = self.text_data_processing)
            text_data_prcessed = textprocessing.text_processing(input=link_textdata)
            ## create small chunk text from big text
            # Example for word-based splitting (Recursive splitting)
            word_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                            chunk_overlap=self.chunk_overlap
                                                            )
            text_data_chunks = word_splitter.split_text(text_data_prcessed)

            self.list_text_processeds.append(text_data_chunks)

            logger.log_message("info", "Text processing in the Inference Phase completed successfully.")
            return self.list_text_processeds

       except Exception as e:
            logger.log_message("warning", "Failed to run text processing in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text processing in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def image_processing(self, input: str) -> List[str]:
        try:
            logger.log_message("info", "Image Processing in the Inference Phase started.")
            # access data folder before transforming
            # dir_imagedata = os.getcwd() + "/" + self.image_data.image_dir
            # link_imagedata = dir_imagedata + "/" + input
            # self.image_data.image_path = link_imagedata

            link_imagedata = input
            imageprocessing = ImageDataInference(config=self.image_data_processing
                                                )
            image_data_prcessed = imageprocessing.image_processing(input=link_imagedata)

            self.list_image_processeds.append(image_data_prcessed)
            
            logger.log_message("info", "Image Processing in the Inference Phase completed successfully.")
            return self.list_image_processeds


        except Exception as e:
            logger.log_message("warning", "Failed to run Image Processing in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run Image Processing in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def audio_processing(self, input: str) -> None:
        try:
            logger.log_message("info", "Audio processing in the Inference Phase started.")
            # access data folder before transforming
            # dir_audiodata = os.getcwd() + "/" + self.audio_data.audio_dir
            # link_audiodata = dir_audiodata + "/" + input
            # self.audio_data.audio_path = link_audiodata

            link_audiodata = input
            audioprocessing = AudioDataInference(config=self.audio_data_processing, 
                                                )
            audio_data_prcessed = audioprocessing.audio_processing(input=link_audiodata)

            self.list_audio_processeds.append(audio_data_prcessed)
            
            logger.log_message("info", "Audio Processing in the Inference Phase completed successfully.")
            return self.list_audio_processeds
        
        except Exception as e:
            logger.log_message("warning", "Failed to run Audio Processing in the Inference Phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run Audio Processing in the Inference Phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)
    
        
    def run_data_embedding_pipeline(self, 
                                    text: List[str], 
                                    image: List[str], 
                                    audio: List[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            logger.log_message("info", "Data embedding in the Inference phase started.")

            text_embeds = self.text_embedding(text=text)
            image_embeds = self.image_embedding(image=image)
            audio_embeds = self.audio_embedding(audio=audio)
            print(f" TEXT EMBEDS:  {text_embeds.shape}", 
                  f" IMAGE EMBEDS: {image_embeds.shape}", 
                  f" AUDIO EMBEDS: {audio_embeds.shape}")


            text_new_embeddings, image_new_embeddings, audio_new_embeddings = self.shared_embedding_space(text_embeds[0],
                                                                                                          image_embeds,
                                                                                                          audio_embeds)
            textnew_embed = (text_new_embeddings, text_embeds[1])
            logger.log_message("info", "Data embedding in the Inference phase completed successfully.")
            return textnew_embed,  image_new_embeddings, audio_new_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run data embedding in the Inference phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data embedding in the Inference phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def text_embedding(self, text: List[str]) -> torch.Tensor:
        try:
            logger.log_message("info", "Text embedding in the Inference phase started.")
            textEmbedding = EmbeddingTextInference(config=self.text_data)
            tokens = []
            embedding_text, token_list = textEmbedding.embedding_text(text)
            tokens.append(token_list)
            # self.text_embeddings.append(embedding_text)
            # self.text_embeddings = self.text_embeddings[0]
            self.text_embeddings = embedding_text
            logger.log_message("info", "Text embedding in the Inference phase completed successfully.")
            text_tensors_flatten = torch.tensor(self.text_embeddings).squeeze(1) # torch.Size([n_texts, 512, 768])
            pooled_text_tensors = torch.mean(text_tensors_flatten, dim=1) # torch.Size([n_texts, 768])
            text_tensors = (pooled_text_tensors, tokens) # 0 : pooled_text_tensors, 1: token_list, 2:shape_texts 
            return text_tensors

        except Exception as e:
            logger.log_message("warning", "Failed to run text embedding in the Inference phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run text embedding in the Inference phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)
    

    def image_embedding(self, image) -> torch.Tensor:
        try:
            logger.log_message("info", "Image embedding in the Inference phase started.")
            imageEmbedding = EmbeddingImageInference(config=self.image_data)
            self.image_embeddings.append(imageEmbedding.embedding_image(image))
            
            logger.log_message("info", "Image embedding in the Inference phase completed successfully.")  
            image_embed = torch.tensor(self.image_embeddings) # torch.Size([n_images, 1, 512])
            pooled_image_embed = torch.mean(image_embed, dim=1) # torch.Size([n_images, 512])
            pooled_image_embed = pooled_image_embed.to(self.device)
            return pooled_image_embed


        except Exception as e:
            logger.log_message("warning", "Failed to run image embedding in the Inference phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run image embedding in the Inference phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def audio_embedding(self, audio) -> torch.Tensor:
        try:
            logger.log_message("info", "Audio embedding in the Inference phase started.")
            audioEmbedding = EmbeddingAudioInference(config=self.audio_data)
            embedding_audio = audioEmbedding.embedding_audio(audio)  
            mean_tensor = torch.mean(torch.tensor(embedding_audio), dim=1) # [1, 768]
            # self.audio_embeddings.append(mean_tensor)
            # self.audio_embeddings = self.audio_embeddings[0]
            self.audio_embeddings = mean_tensor

            logger.log_message("info", "Audio embedding in the Inference phase completed successfully.")
            pooled_audio_embed = self.audio_embeddings.to(self.device)
            return pooled_audio_embed

        except Exception as e:
            logger.log_message("warning", "Failed to run audio embedding in the Inference phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run audio embedding in the Inference phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def shared_embedding_space(self, 
                               text_embeds: torch.Tensor, 
                               image_embeds: torch.Tensor, 
                               audio_embeds: torch.Tensor
                               ) -> torch.Tensor:
        try:
            logger.log_message("info", "Shared embedding space in the Inference phase started.")
            sharedEmbeddingSpace = self.shared_embeddings(text_embeds, 
                                                          image_embeds, 
                                                          audio_embeds
                                                          )
            
            text_new_embeddings, image_new_embeddings, audio_new_embeddings = sharedEmbeddingSpace

            logger.log_message("info", "Shared embedding space in the Inference phase completed successfully.")

            return text_new_embeddings, image_new_embeddings, audio_new_embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to run shared embedding space in the Inference phase: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run shared embedding space in the Inference phase: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    



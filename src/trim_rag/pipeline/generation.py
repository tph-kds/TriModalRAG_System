import os
import sys
from typing import Optional

from langchain.chains import SimpleChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import MultimodalGenerationArgumentsConfig

from src.trim_rag.generation import (
    PromptFlows, 
    MultimodalGeneration,
    PostProcessing
)

class GenerationPipeline:

    def __init__(self, config: MultimodalGenerationArgumentsConfig) -> None:
        super(GenerationPipeline, self).__init__()

        self.config = config
        self.multimodal_generation_config = self.config.multimodal_generation
        self.prompt_flows_config = self.config.prompt_flows
        self.post_processing_config = self.config.post_processing


        self.prompt_flows = PromptFlows(config=self.prompt_flows_config)
        self.multimodal_generation = MultimodalGeneration(config=self.multimodal_generation_config,
                                                            llm_text=config.llm_text,
                                                            llm_image=config.llm_image,
                                                            llm_audio=config.llm_audio)
        
        self.post_processing = PostProcessing(config=self.post_processing_config)

    def run_generation_pipeline(self, 
                                retriever: Optional[SimpleChain],
                                image_url: Optional[str],
                                video_url: Optional[str],
                                query: Optional[str]
                                ) -> SimpleChain:
        try: 
            logger.log_message("info", "Running generation pipeline...")
            prompt = self._get_prompt_flows()
            multimodal_generation = self._get_multimodal_generation(
                prompt=prompt,
                image_url=image_url,
                video_url=video_url
            )
            p_chain = self.post_processing.post_processing(retriever=retriever,
                                                            query=query)
            

            
            logger.log_message("info", "Generation pipeline completed")

            return multimodal_generation, p_chain

        except Exception as e:
            logger.log_message("warning", "Failed to run generation pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run generation pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_prompt_flows(self) -> Optional[ChatPromptTemplate]:
        try:

            logger.log_message("info", "Getting prompt flows started.")

            prompt = self.prompt_flows.prompt()

            logger.log_message("info", "Getting prompt flows completed successfully.")
            return prompt
        
        except Exception as e:
            logger.log_message("warning", "Failed to get prompt flows: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get prompt flows: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_multimodal_generation(self, 
                                   prompt: Optional[ChatPromptTemplate],
                                   image_url: Optional[str],
                                   video_url: Optional[str]
                                   ) -> Optional[ChatPromptTemplate]:
        try:

            logger.log_message("info", "Getting multimodal generation started.")

            multimodal_generation = self.multimodal_generation.multimodal_generation(
                prompt=prompt,
                image_url=image_url,
                video_url=video_url
            )

            logger.log_message("info", "Getting multimodal generation completed successfully.")
            return multimodal_generation
        
        except Exception as e:
            logger.log_message("warning", "Failed to get multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_post_processing(self, 
                             retriever, 
                             query
                             ) -> Optional[ChatPromptTemplate]:
        try:

            logger.log_message("info", "Getting post processing started.")

            post_processing = self.post_processing.post_processing(retriever=retriever, 
                                                                   query=query
                                                                   )

            logger.log_message("info", "Getting post processing completed successfully.")
            return post_processing
        
        except Exception as e:
            logger.log_message("warning", "Failed to get post processing: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get post processing: " + str(e),
                error_details = sys,
            )
            print(my_exception)

\




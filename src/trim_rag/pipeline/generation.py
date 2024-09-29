import os
import sys
from typing import List, Optional, Tuple, Union

from langchain.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import (
    MultimodalGenerationPipelineArgumentsConfig,
    MultiModelsArgumentsConfig,
    EmbeddingArgumentsConfig
)

from src.trim_rag.generation import (
    PromptFlows, 
    MultimodalGeneration,
    PostProcessing
)

from langchain_core.runnables.base import RunnableSerializable

class GenerationPipeline:

    def __init__(self, config: MultimodalGenerationPipelineArgumentsConfig,
                 model_config: MultiModelsArgumentsConfig,
                 embed_config: EmbeddingArgumentsConfig) -> None:
        super(GenerationPipeline, self).__init__()

        self.config = config
        self.embed_config = embed_config
        self.multimodal_generation_config = self.config.multimodal_generation
        self.prompt_flows_config = self.config.prompts
        self.post_processing_config = self.config.post_processing

        self.llm_text = model_config.text_model
        self.llm_image = model_config.image_model
        self.llm_audio = model_config.audio_model


        self.prompt_flows = PromptFlows(config=self.prompt_flows_config)
        self.multimodal_generation = MultimodalGeneration(config=self.multimodal_generation_config,
                                                          model_config=model_config,
                                                          embed_config=self.embed_config
                                                          )
        
        self.post_processing = PostProcessing(config=self.post_processing_config)

    def run_generation_pipeline(self, 
                                retriever: Optional[SequentialChain],
                                image_url: Optional[str],
                                video_url: Optional[str],
                                query: Optional[str]
                                ) -> SequentialChain:
        try: 
            logger.log_message("info", "Running generation pipeline...")
            prompt, data_prompt = self._get_prompt_flows(system_str=self.multimodal_generation_config.system_str,
                                            context_str=self.multimodal_generation_config.context_str,
                                            question_str=query,
                                            image_str=image_url,
                                            video_str=video_url
                                            )
            
            prompt_llm = self._get_prompt_llm(), self._get_prompt_llm(), self._get_prompt_llm()

            full_chain, messages_question = self._get_multimodal_generation(
                prompt=prompt,
                prompt_llm=prompt_llm,
                image_url=image_url,
                video_url=video_url,
                retriever=retriever
            )
            p_chain = self.post_processing.post_processing(retriever=retriever,
                                                            query=query)
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])
            
            rag_chain = (
                  messages_question
                | full_chain
                # | p_chain
                | StrOutputParser()
            )
            print("RAG CHAIN: ", rag_chain)
            
            logger.log_message("info", "Generation pipeline completed")

            return rag_chain

        except Exception as e:
            logger.log_message("warning", "Failed to run generation pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run generation pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_prompt_flows(self, 
                          system_str: str, 
                          context_str: str, 
                          question_str: str, 
                          image_str: str, 
                          video_str: str) -> Optional[ChatPromptTemplate]:
        try:

            logger.log_message("info", "Getting prompt flows started.")

            prompt, data_prompt = self.prompt_flows.prompt_flows(
                system_str=system_str,
                context_str=context_str,
                question_str=question_str,
                image_str=image_str,
                video_str=video_str
            )

            logger.log_message("info", "Getting prompt flows completed successfully.")
            # print("\n\n")
            # print(prompt)
            # print("\n\n")
            # print(data_prompt)
            return prompt, data_prompt
        
        except Exception as e:
            logger.log_message("warning", "Failed to get prompt flows: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get prompt flows: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_prompt_llm(self, ) -> Optional[ChatPromptTemplate]:
        try:

            logger.log_message("info", "Getting prompt llm started.")

            prompt, data_prompt = self.prompt_flows.prompt_llm()

            logger.log_message("info", "Getting prompt llm completed successfully.")

            return prompt
        
        except Exception as e:
            logger.log_message("warning", "Failed to get prompt llm: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get prompt llm: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_multimodal_generation(self, 
                                   prompt: Optional[ChatPromptTemplate],
                                   prompt_llm: Optional[ChatPromptTemplate],
                                   image_url: Optional[str],
                                   video_url: Optional[str],
                                   retriever: Optional[RunnablePassthrough]
                                   ) -> Optional[RunnableSerializable]:
        try:

            logger.log_message("info", "Getting multimodal generation started.")

            multimodal_generation = self.multimodal_generation.multimodal_generation(
                prompt=prompt,
                prompt_llm=prompt_llm,
                image_url=image_url,
                video_url=video_url,
                retriever=retriever
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
                             ) -> None:
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






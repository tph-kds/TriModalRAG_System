import os
import sys

from typing import Dict, List, Optional, Tuple, Union

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import (
    MultimodalGenerationPipelineArgumentsConfig,
    MultiModelsArgumentsConfig,
    EmbeddingArgumentsConfig,
)
from src.trim_rag.generation import PromptFlows, MultimodalGeneration, PostProcessing
from langchain.chains.sequential import SequentialChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableSerializable


class GenerationPipeline:
    def __init__(
        self,
        config: MultimodalGenerationPipelineArgumentsConfig,
        model_config: MultiModelsArgumentsConfig,
        embed_config: EmbeddingArgumentsConfig,
        api_config: Dict,
        llm_config: Dict,
    ) -> None:
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
        self.multimodal_generation = MultimodalGeneration(
            config=self.multimodal_generation_config,
            model_config=model_config,
            embed_config=self.embed_config,
            api_config=api_config,
            llm_config=llm_config,
        )

        self.post_processing = PostProcessing(
            config=self.post_processing_config,
            COHERE_API_KEY=api_config["COHERE_API_KEY"],
        )
        self.chat_history: BaseMessage = [
            HumanMessage(content="Hello, How about you with a weather, today?"),
            AIMessage(
                content="Hi, I'm doing well, but more colder for the previous day. How about you?"
            ),
        ]

    def run_generation_pipeline(
        self,
        retriever: Optional[SequentialChain],
        image_url: Optional[str],
        video_url: Optional[str],
        question_str: Optional[str],
        query_str: Optional[str],
    ) -> SequentialChain:
        try:
            logger.log_message("info", "Running generation pipeline...")
            prompt = self._get_prompt_flows()

            prompt_llm = (
                self._get_prompt_llm(),
                self._get_prompt_llm(),
                self._get_prompt_llm(),
            )

            full_message = self._get_multimodal_generation(
                prompt=prompt,
                prompt_llm=prompt_llm,
                question_str=query_str,
                image_url=image_url,
                video_url=video_url,
                retriever=retriever,
                chat_history=self.chat_history,
            )

            # update chat history
            def format_history(chat_history, input) -> List[BaseMessage]:
                for key, text in input.items():
                    key_str = key.split("_")[0]

                    if key_str == "answer":
                        chat_history.append(AIMessage(content=str(text)))
                    else:
                        chat_history.append(HumanMessage(content=str(text)))
                return chat_history

            self.chat_history = format_history(self.chat_history, full_message)

            ## Get meta data for the question and response
            meta_data_main = self.meta_data(
                question_str=query_str,
                # image_str=image_url,
                # audio_str=video_url,
                response_retriever=full_message,
                chat_history=self.chat_history,
            )

            print("hung")
            # Get RAG CHAIN
            rag_chain = self.post_processing.post_processing(prompt=prompt)

            print("RAG CHAIN: ", rag_chain)

            logger.log_message("info", "Generation pipeline completed")
            print(rag_chain.invoke(meta_data_main))

            return rag_chain, meta_data_main, full_message

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run generation pipeline: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run generation pipeline: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def meta_data(
        self,
        question_str: Optional[str],
        #   image_str: Optional[str],
        #   audio_str: Optional[str],
        response_retriever: Optional[Dict],
        chat_history: Optional[List[BaseMessage]],
    ) -> Dict:
        try:
            logger.log_message("info", "Getting meta data started.")

            meta_data = self.post_processing.meta_data(
                question_str=question_str,
                # image_str=image_str,
                # audio_str=audio_str,
                response_retriever=response_retriever,
                chat_history=chat_history,
            )

            logger.log_message("info", "Getting meta data completed successfully.")
            return meta_data

        except Exception as e:
            logger.log_message("warning", "Failed to get meta data: " + str(e))
            my_exception = MyException(
                error_message="Failed to get meta data: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _get_prompt_flows(self) -> Optional[ChatPromptTemplate]:
        try:
            logger.log_message("info", "Getting prompt flows started.")

            prompt = self.prompt_flows.prompt_flows()

            logger.log_message("info", "Getting prompt flows completed successfully.")
            return prompt

        except Exception as e:
            logger.log_message("warning", "Failed to get prompt flows: " + str(e))
            my_exception = MyException(
                error_message="Failed to get prompt flows: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _get_prompt_llm(
        self,
    ) -> Optional[ChatPromptTemplate]:
        try:
            logger.log_message("info", "Getting prompt llm started.")

            prompt = self.prompt_flows.prompt_llm()

            logger.log_message("info", "Getting prompt llm completed successfully.")

            return prompt

        except Exception as e:
            logger.log_message("warning", "Failed to get prompt llm: " + str(e))
            my_exception = MyException(
                error_message="Failed to get prompt llm: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _get_multimodal_generation(
        self,
        prompt: Optional[ChatPromptTemplate],
        prompt_llm: Optional[ChatPromptTemplate],
        question_str: Optional[str],
        image_url: Optional[str],
        video_url: Optional[str],
        retriever: Optional[RunnablePassthrough],
        chat_history: Optional[BaseChatMessageHistory],
    ) -> Optional[RunnableSerializable]:
        try:
            logger.log_message("info", "Getting multimodal generation started.")

            multimodal_generation = self.multimodal_generation.multimodal_generation(
                prompt=prompt,
                prompt_llm=prompt_llm,
                question_str=question_str,
                image_url=image_url,
                video_url=video_url,
                retriever=retriever,
                chat_history=chat_history,
            )

            logger.log_message(
                "info", "Getting multimodal generation completed successfully."
            )
            return multimodal_generation

        except Exception as e:
            logger.log_message(
                "warning", "Failed to get multimodal generation: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to get multimodal generation: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    def _get_post_processing(self, prompt: Optional[ChatPromptTemplate]) -> Runnable:
        try:
            logger.log_message("info", "Getting post processing started.")

            post_processing = self.post_processing.post_processing(
                prompt=prompt,
            )

            logger.log_message(
                "info",
                "Getting post processing completed successfully. Output is a main Runnable for the generation pipeline.",
            )
            return post_processing

        except Exception as e:
            logger.log_message("warning", "Failed to get post processing: " + str(e))
            my_exception = MyException(
                error_message="Failed to get post processing: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    # def meta_data(self,
    #               question_str: str,
    #               image_url: str,
    #               audio_url: str,
    #               response_retriever: Dict) -> Dict:

    #     logger.log_message("info", "Getting meta data in generation pipeline started.")
    #     def concat_function(docs):
    #         return " ".join(str(key + "\n")  for doc, key in docs.items())
    #     logger.log_message("info", "Getting pre prompt in generation pipeline started.")

    #     return {
    #                 # "chat_history":  chat_history,
    #                 "context_str": concat_function(response_retriever) ,
    #                 "question_str":  question_str[0],
    #                 "image_url": image_url,
    #                 "audio_url":  audio_url
    #             }

    # def pre_prompt(self) -> RunnableParallel:

    #     logger.log_message("info", "Getting pre prompt in generation pipeline started.")

    #     pre_prompt = RunnableParallel({
    #         "context_str": RunnablePassthrough(itemgetter("context_str")),
    #         "question_str": RunnablePassthrough(itemgetter("question_str")),
    #         "image_url": RunnablePassthrough(itemgetter("image_url")),
    #         "audio_url": RunnablePassthrough(itemgetter("audio_url")),
    #         "chat_history": itemgetter("chat_history"),
    #     } )

    #     return pre_prompt

import os
import sys
from operator import itemgetter
from typing import Dict, Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import PostProcessingArgumentsConfig
from src.trim_rag.generation.customRunnable import StringFormatterRunnable

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_cohere.llms import Cohere
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# https://python.langchain.com/v0.2/docs/integrations/retrievers/cohere-reranker/
## Use rerank model for post processing ##
class PostProcessing:
    def __init__(
        self,
        config: PostProcessingArgumentsConfig,
        COHERE_API_KEY: Optional[str] = None,
    ) -> None:
        super(PostProcessing, self).__init__()

        self.config = config
        self.temperature = self.config.temperature  # default = 0
        self.verbose = self.config.verbose  # default = False
        self.frequency_penalty = self.config.frequency_penalty  # default = 0
        self.max_tokens = self.config.max_tokens  # default = 128
        self.model_cohere = self.config.model_cohere  # default = "rerank-english-v3.0"

        # get a new token: https://dashboard.cohere.ai/
        self.cohere_api_key = COHERE_API_KEY

    def _get_llm(self) -> Cohere:
        try:
            logger.log_message("info", "Getting LLM is Cohere started.")
            llms = Cohere(
                temperature=self.temperature,
                verbose=self.verbose,
                cohere_api_key=self.cohere_api_key,
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_tokens,
            )
            logger.log_message("info", "Getting LLM is Cohere completed successfully.")
            return llms

        except Exception as e:
            logger.log_message("warning", "Failed to get LLM is Cohere: " + str(e))
            my_exception = MyException(
                error_message="Failed to get LLM is Cohere: " + str(e),
                error_details=sys,
            )
            print(my_exception)

    # def _get_retriever(self, retriever):
    #     # Older version of Cohere, so I don't use it for this project
    #     try:
    #         logger.log_message("info", "Getting retriever in post processing started.")
    #         compressor = CohereRerank(model=self.model_cohere)

    #         compression_retriever = ContextualCompressionRetriever(
    #             base_compressor=compressor,
    #             base_retriever=retriever
    #         )

    #         logger.log_message("info", "Getting retriever in post processing completed successfully.")
    #         return compression_retriever

    #     except Exception as e:
    #         logger.log_message("warning", "Failed to get retriever in post processing: " + str(e))
    #         my_exception = MyException(
    #             error_message = "Failed to get retriever in post processing: " + str(e),
    #             error_details = sys,
    #         )
    #         print(my_exception)

    def pre_prompt(self) -> RunnableParallel:
        logger.log_message("info", "Getting pre prompt for post processing started.")

        pre_prompt = RunnableParallel(
            {
                "context_str": RunnablePassthrough(itemgetter("context_str")),
                "question_str": RunnablePassthrough(itemgetter("question_str")),
                # "image_str": RunnablePassthrough(itemgetter("image_str")),
                # "audio_str": RunnablePassthrough(itemgetter("audio_str")),
                "chat_history": itemgetter("chat_history"),
            }
        )

        return pre_prompt

    def meta_data(
        self,
        question_str: str,
        #   image_str: str,
        #   audio_str: str,
        response_retriever: Dict,
        chat_history: str,
    ) -> Dict:
        logger.log_message("info", "Getting meta data in post processing started.")

        def concat_function(docs):
            return " ".join(
                str(doc) + " : " + str(key + "\n")
                for doc, key in docs.items()
                if key != None
            )

        logger.log_message(
            "info", "Getting pre prompt in post processing completed successfully."
        )

        return {
            "chat_history": chat_history,
            "context_str": concat_function(response_retriever),
            "question_str": question_str,
            # "image_str": image_str,
            # "audio_str":  audio_str
        }

    def post_processing(self, prompt: ChatPromptTemplate) -> Runnable:
        try:
            logger.log_message("info", "Post processing started.")
            llm = self._get_llm()
            # compression_retriever = self._get_retriever(retriever)
            # chain = RetrievalQA.from_chain_type(
            #     llm=llm,
            #     retriever=compression_retriever
            # )
            pre_prompt = self.pre_prompt()

            final_chain = pre_prompt | prompt | llm | StrOutputParser()

            logger.log_message("info", "Post processing completed successfully.")

            # return chain.run({"query": query})
            return final_chain

        except Exception as e:
            logger.log_message("warning", "Failed to post processing: " + str(e))
            my_exception = MyException(
                error_message="Failed to post processing: " + str(e),
                error_details=sys,
            )
            print(my_exception)

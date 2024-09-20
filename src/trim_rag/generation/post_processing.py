import os
import sys
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import PostProcessingArgumentsConfig

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langhchain_cohere import CohereTextSplitter, CohereRerank
from langchain.chains import RetrievalQA

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.llms import Cohere


# https://python.langchain.com/v0.2/docs/integrations/retrievers/cohere-reranker/
## Use rerank model for post processing ##
class PostProcessing:

    def __init__(self, 
                 config: PostProcessingArgumentsConfig,
                COHERE_API_KEY: Optional[str] = None
                ) -> None:
        super(PostProcessing, self).__init__()

        self.config = config
        self.temperature = self.config.temperature # default = 0
        self.verbose = self.config.verbose # default = False
        self.frequency_penalty = self.config.frequency_penalty # default = 0
        self.max_tokens = self.config.max_tokens # default = 128
        self.model_cohere = self.config.model_cohere # default = "rerank-english-v3.0"
        
        # get a new token: https://dashboard.cohere.ai/
        self.cohere_api_key = COHERE_API_KEY

    def _get_llm(self):
        try:
            logger.log_message("info", "Getting LLM is Cohere started.")
            llms = Cohere(temperature=self.temperature, 
                        verbose=self.verbose,
                        cohere_api_key= self.cohere_api_key,
                        frequency_penalty= self.frequency_penalty,
                        max_tokens = self.max_tokens,
                        )
            logger.log_message("info", "Getting LLM is Cohere completed successfully.")
            return  llms

        except Exception as e:
            logger.log_message("warning", "Failed to get LLM is Cohere: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get LLM is Cohere: " + str(e),
                error_details = sys,
            )
            print(my_exception) 
        
    def _get_retriever(self, retriever):
        try:
            logger.log_message("info", "Getting retriever in post processing started.")
            compressor = CohereRerank(model=self.model_cohere)

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=retriever
            )

            logger.log_message("info", "Getting retriever in post processing completed successfully.")
            return compression_retriever
        
        except Exception as e:
            logger.log_message("warning", "Failed to get retriever in post processing: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get retriever in post processing: " + str(e),
                error_details = sys,
            )
            print(my_exception)
    

    def post_processing(self, retriever, query) -> None:
        try: 
            logger.log_message("info", "Post processing started.")
            llm = self._get_llm()
            compression_retriever = self._get_retriever(retriever)
            chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=compression_retriever
            )

            logger.log_message("info", "Post processing completed successfully.")

            # return chain.run({"query": query})
            return chain
        
        except Exception as e:
            logger.log_message("warning", "Failed to post processing: " + str(e))
            my_exception = MyException(
                error_message = "Failed to post processing: " + str(e),
                error_details = sys,
            )
            print(my_exception)


# Step 5: Simulate a conversation
# query_1 = "What is the weather forecast for tomorrow?"
# response_1 = tri_modal_rag.run(query_1, message_history=memory.load_memory_variables({}))
# print(f"User: {query_1}\nSystem: {response_1}\n")
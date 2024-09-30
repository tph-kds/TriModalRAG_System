import os
import sys
import getpass
from typing import Dict, List, Optional, Union


from src.config_params.constants import COHERE_API_KEY
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import (
    MultimodalGenerationArgumentsConfig,
    MultiModelsArgumentsConfig,
    EmbeddingArgumentsConfig
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.trim_rag.models import TextModel, ImageModel, AudioModel
from langchain_qdrant import QdrantVectorStore
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from src.trim_rag.models import (
    TextModelRunnable, 
    ImageModelRunnable, 
    AudioModelRunnable
)
from langchain_cohere import ChatCohere
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from operator import itemgetter

# from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config_params import (
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
    GOOGLE_API_KEY,
    LANGCHAIN_API_KEY
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory


class MultimodalGeneration:
    def __init__(self, 
                 config: MultimodalGenerationArgumentsConfig, 
                 model_config: MultiModelsArgumentsConfig,
                 embed_config: EmbeddingArgumentsConfig,
                ) -> None:
        super(MultimodalGeneration, self).__init__()

        self.config = config
        self.system_str = self.config.system_str
        self.model_config = model_config
        self.embed_config = embed_config

        # self.llm_text = TextModel(config=model_config.text_model, 
        #                           embed_config=embed_config.text_data
        #                           )
        # self.llm_image = ImageModel(config=model_config.image_model,
        #                             embed_config=embed_config.image_data
        #                             )
        # self.llm_audio = AudioModel(config=model_config.audio_model,
        #                             embed_config=embed_config.audio_data
        #                             )
        self.llm_text = TextModelRunnable(config=model_config.text_model, 
                                  embed_config=embed_config.text_data
                                  )
        self.llm_image = ImageModelRunnable(config=model_config.image_model,
                                    embed_config=embed_config.image_data
                                    )
        self.llm_audio = AudioModelRunnable(config=model_config.audio_model,
                                    embed_config=embed_config.audio_data
                                    )
        
        self.model = "gemini-1.5-flash-001"
        self.temperature = 0
        self.max_tokens = 128
        self.max_retries = 6
        self.stop = None


    def multimodal_generation(self, 
                              prompt: Optional[str], 
                              prompt_llm: Optional[ChatPromptTemplate],
                              question_str: Optional[str],
                              image_url: Optional[str], 
                              video_url: Optional[str],
                              retriever: Optional[RunnablePassthrough],
                              chat_history: Optional[BaseChatMessageHistory]
                              ) -> None:
        try:

            logger.log_message("info", "Getting multimodal generation started.")
            
            meta_data_text = self.meta_data( retriever=retriever,
                                            question_str=question_str,
                                            system_str=self.system_str,
                                            type_str="text",
                                            info_str="",
                                            variable_name="Halihallo",
                                            chat_history=chat_history)

            meta_data_image = self.meta_data( retriever=retriever,
                                            question_str=question_str,
                                            system_str=self.system_str,
                                            type_str="image",
                                            info_str=image_url,
                                            variable_name="Halihallo",
                                            chat_history=chat_history)

            meta_data_audio = self.meta_data( retriever=retriever,
                                            question_str=question_str,
                                            system_str=self.system_str,
                                            type_str="audio",
                                            info_str=video_url,
                                            variable_name="Halihallo",
                                            chat_history=chat_history)
            
            # init
            self._init_model()
            llm_model = self._get_llm(name_model=self.model,
                                      temperature=self.temperature,
                                      max_tokens=self.max_tokens,
                                      max_retries=self.max_retries,
                                      stop=self.stop
                                      )
            
            text_rag_chain = self._rag_chain(
                                              prompt = prompt_llm[0],
                                             llm_model = llm_model,
                                             type_data = "text")

            image_rag_chain = self._rag_chain(
                                              prompt = prompt_llm[1],
                                              llm_model = llm_model,
                                              type_data = "image")
            
            audio_rag_chain = self._rag_chain(
                                              prompt = prompt_llm[2],
                                              llm_model = llm_model,
                                              type_data = "audio")
            def create_invoke_dict(meta_data) -> dict:
                return {
                    "chat_history": chat_history,
                    **meta_data
                }

            ## COmplete rag for each part of multimodal
            result_text = text_rag_chain.invoke(create_invoke_dict(meta_data_text))
            result_image = image_rag_chain.invoke(create_invoke_dict(meta_data_image))
            result_audio = audio_rag_chain.invoke(create_invoke_dict(meta_data_audio))

            ## give them to additional chains in the main defined prompt above.
            full_response = {
                "text": result_text,
                "image": result_image,
                "audio": result_audio
            }
            
            # text_rag_chain: find embed relevant text from Qdrant
            # image_rag_chain: find embed relevant image from Qdrant
            # audio_rag_chain: find embed relevant audio from Qdrant

            # After completing above rag chain, we can get the full chain and use to generate


            logger.log_message("info", "Getting multimodal generation completed successfully.")
            return full_response

        except Exception as e:
            logger.log_message("warning", "Failed to get multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)
            

    # def _create_messages(self, 
    #                      prompt: Optional[str], 
    #                      image_url: Optional[str], 
    #                      video_url: Optional[str]
    #                      ) -> Optional[Union[SystemMessage, HumanMessage]]:
    #     try:

    #         logger.log_message("info", "Getting messages in multimodal generation started.")
    #         system_message = SystemMessage(content= self.system_str)
    #         messages_reponse = HumanMessage(
    #             content= [
    #                 {
    #                     "type": "text",
    #                     "text": prompt,
    #                 },
    #                 {
    #                     "type": "image",
    #                     "image": image_url
    #                 },
    #                 {
    #                     "type": "video",
    #                     "video": video_url
    #                 },
    #             ]
    #         )

    #         logger.log_message("info", "Getting messages in multimodal generation completed successfully.")
    #         return system_message, messages_reponse
        
    #     except Exception as e:
    #         logger.log_message("warning", "Failed to get messages in multimodal generation: " + str(e))
    #         my_exception = MyException(
    #             error_message = "Failed to get messages in multimodal generation: " + str(e),
    #             error_details = sys,
    #         )
    #         print(my_exception)

    def _rag_chain(self, 
                    prompt: Optional[ChatPromptTemplate],
                    llm_model: Runnable,
                    type_data: str) -> Runnable:
        try:

            logger.log_message("info", f"Getting rag chain for {type_data.upper()} in multimodal generation started.")
            pre_prompt = self.pre_prompt()
            # print(pre_prompt | prompt)
            rag_chain: Runnable = ( pre_prompt | prompt | llm_model | StrOutputParser())

            logger.log_message("info", f"Getting rag chain for {type_data.upper()} in multimodal generation completed successfully.")
            return rag_chain

        except Exception as e:
            logger.log_message("warning", f"Failed to get rag chain for {type_data.upper()} in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = f"Failed to get rag chain for {type_data.upper()} in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def meta_data(self, 
                  retriever: QdrantVectorStore,
                  question_str: str,
                  system_str: str,
                  type_str: str,
                  info_str: str,
                  variable_name: str,
                  chat_history: Optional[BaseChatMessageHistory]) -> Dict:

        logger.log_message("info", "Getting meta data in multimodal generation started.")
        # return {
        #             "context_str": RunnablePassthrough( lambda: test_function(retriever)) , 
        #             "question_str": RunnablePassthrough(lambda: question_str),
        #             "system_str": RunnablePassthrough( lambda: system_str),
        #             "type_str": RunnablePassthrough(lambda: type_str),
        #             "info_str": RunnablePassthrough(lambda : info_str),
        #             "variable_name": RunnablePassthrough(lambda : variable_name),
        #         } 
        def test_function(docs):
            return " ".join(doc for doc in docs)
        logger.log_message("info", "Getting pre prompt in multimodal generation started.")

        return {
                    # "chat_history":  chat_history,
                    "context_str": test_function(retriever) , 
                    "question_str":  question_str[0],
                    "system_str": system_str,
                    "type_str":  type_str,
                    "info_str":  info_str,
                    "variable_name":  variable_name,
                } 
    def pre_prompt(self) -> RunnableParallel:
        
        def test_function(docs):
            return " ".join(doc for doc in docs)
        logger.log_message("info", "Getting pre prompt in multimodal generation started.")

        pre_prompt = RunnableParallel({
            "context_str": RunnablePassthrough(itemgetter("context_str")),   
            "question_str": RunnablePassthrough(itemgetter("question_str")),
            "system_str": RunnablePassthrough(itemgetter("system_str")),
            "type_str": RunnablePassthrough(itemgetter("type_str")),
            "info_str": RunnablePassthrough(itemgetter("info_str")),
            "variable_name": RunnablePassthrough(itemgetter("variable_name")), 
            "chat_history": itemgetter("chat_history"),
        } ) 

        return pre_prompt
    
    def _get_llm(self, 
                 name_model: str, # Name of ChatVertexAI model 
                 temperature: float, # Sampling temperature.
                 max_tokens: int, # Max number of tokens to generate.
                 max_retries: int, # Max number of retries.
                 stop: Optional[List[str]] # Default stop sequences.
                 ) -> Runnable:
        try:

            logger.log_message("info", "Getting llm distributed by Google in multimodal generation started.")
            # gemini-1.0-pro-vision-001
            llm = ChatGoogleGenerativeAI(
                model = name_model,
                temperature = temperature,
                max_tokens = max_tokens,
                max_retries = max_retries,
                stop = stop
            )

            logger.log_message("info", "Getting llm distributed by Google in multimodal generation completed successfully.")

            return llm

        except Exception as e:
            logger.log_message("warning", "Failed to get llm distributed by Google in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get llm distributed by Google in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _init_model(self):
            # init 
        
        if "GOOGLE_API_KEY" not in os.environ:
            # os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["LANGSMITH_API_KEY"] = LANGCHAIN_API_KEY
        os.environ["LANGSMITH_ENDPOINT"] = LANGCHAIN_ENDPOINT
        os.environ["LANGSMITH_TRACING"] = LANGCHAIN_TRACING_V2
        os.environ["LANGSMITH_PROJECT"] = LANGCHAIN_PROJECT
                        
    def contextualize_question(self, 
                               llm, 
                               retriever, 
                               prompt: ChatPromptTemplate) -> Runnable:
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        ### Statefully manage chat history ###
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain
    


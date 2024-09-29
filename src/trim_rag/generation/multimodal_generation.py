import os
import sys
from typing import Dict, Optional, Union

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
from langchain_core.runnables import Runnable
from src.trim_rag.models import (
    TextModelRunnable, 
    ImageModelRunnable, 
    AudioModelRunnable
)
from langchain_cohere import ChatCohere
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

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


    def multimodal_generation(self, 
                              prompt: Optional[str], 
                              prompt_llm: Optional[ChatPromptTemplate],
                              image_url: Optional[str], 
                              video_url: Optional[str],
                              retriever: Optional[RunnablePassthrough]
                              ) -> None:
        try:

            logger.log_message("info", "Getting multimodal generation started.")
            
            meta_data_text = self.meta_data(retriever,
                                            question_str=prompt,
                                            system_str=self.system_str,
                                            type_str="text",
                                            info_str=prompt_llm[0],
                                            variable_name="chat_history")

            meta_data_image = self.meta_data(retriever,
                                            question_str=prompt,
                                            system_str=self.system_str,
                                            type_str="image",
                                            info_str=prompt_llm[1],
                                            variable_name="chat_history")

            meta_data_audio = self.meta_data(retriever,
                                            question_str=prompt,
                                            system_str=self.system_str,
                                            type_str="audio",
                                            info_str=prompt_llm[2],
                                            variable_name="chat_history")
                                  
            _, messages_question = self._create_messages(prompt, image_url, video_url)
            text_rag_chain = self._rag_chain_for_text(prompt_llm[0], retriever)
            image_rag_chain = self._rag_chain_for_image(prompt_llm[1])
            audio_rag_chain = self._rag_chain_for_audio(prompt_llm[2])

            ## COmplete rag for each part of multimodal
            result_text = text_rag_chain.invoke(meta_data_text)
            result_image = image_rag_chain.invoke(meta_data_image)
            result_audio = audio_rag_chain.invoke(meta_data_audio)

            ## give them to additional chains in the main defined prompt above.
            
            
            # text_rag_chain: find embed relevant text from Qdrant
            # image_rag_chain: find embed relevant image from Qdrant
            # audio_rag_chain: find embed relevant audio from Qdrant

            # After completing above rag chain, we can get the full chain and use to generate

            # full_chain = (
            #       text_rag_chain 
            #     | image_rag_chain 
            #     | audio_rag_chain
            # )
            # print(full_chain)
            # print(messages_question)
            # response = full_chain.invoke({"chat_history": [messages_question]})
            # print(response)
            full_chain = None
            logger.log_message("info", "Getting multimodal generation completed successfully.")
            return full_chain, messages_question

        except Exception as e:
            logger.log_message("warning", "Failed to get multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)
            

    def _create_messages(self, 
                         prompt: Optional[str], 
                         image_url: Optional[str], 
                         video_url: Optional[str]
                         ) -> Optional[Union[SystemMessage, HumanMessage]]:
        try:

            logger.log_message("info", "Getting messages in multimodal generation started.")
            system_message = SystemMessage(content= self.system_str)
            messages_reponse = HumanMessage(
                content= [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                        "image": image_url
                    },
                    {
                        "type": "video",
                        "video": video_url
                    },
                ]
            )

            logger.log_message("info", "Getting messages in multimodal generation completed successfully.")
            return system_message, messages_reponse
        
        except Exception as e:
            logger.log_message("warning", "Failed to get messages in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get messages in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _create_messages_text(self, 
                         prompt: Optional[str], 
                         ) -> Optional[Union[SystemMessage, HumanMessage]]:
        try:

            logger.log_message("info", "Getting messages in multimodal generation started.")
            system_message = SystemMessage(content= self.system_str)
            messages_reponse = HumanMessage(
                content= [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ]
            )

            logger.log_message("info", "Getting messages in multimodal generation completed successfully.")
            return system_message, messages_reponse
        
        except Exception as e:
            logger.log_message("warning", "Failed to get messages in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get messages in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _rag_chain_for_text(self, 
                            prompt: Optional[ChatPromptTemplate],) -> Optional[Runnable]:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        def test_function(docs):
            return " ".join(doc for doc in docs)

        try:

            logger.log_message("info", "Getting rag chain for TEXT in multimodal generation started.")
            # message_text = self._create_messages_text(prompt)
            llm_text = self.llm_text.text_model_runnable()
            # llm_text =  ChatCohere(model="command-r-plus", cohere_api_key= COHERE_API_KEY, temperature= 0.0)
            print("Hung")
            
            rag_chain = prompt |  llm_text | StrOutputParser()

            logger.log_message("info", "Getting rag chain for TEXT in multimodal generation completed successfully.")
            return rag_chain

        except Exception as e:
            logger.log_message("warning", "Failed to get rag chain for TEXT in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get rag chain for TEXT in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _rag_chain_for_image(self) -> Runnable:
        try:

            logger.log_message("info", "Getting rag chain for IMAGE in multimodal generation started.")

            llm_image = self.llm_image.image_model_runnable()
            # llm_image =  ChatCohere(model="command-r-plus", cohere_api_key= COHERE_API_KEY, temperature= 0.0)
            
            rag_chain: Runnable = (
                    RunnablePassthrough()
                    | llm_image
                    | StrOutputParser()
            )

            logger.log_message("info", "Getting rag chain for IMAGE in multimodal generation completed successfully.")
            return rag_chain

        except Exception as e:
            logger.log_message("warning", "Failed to get rag chain for IMAGE in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get rag chain for IMAGE in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _rag_chain_for_audio(self) -> Runnable:
        try:

            logger.log_message("info", "Getting rag chain for AUDIO in multimodal generation started.")

            llm_audio = self.llm_audio.audio_model_runnable()
            # llm_audio =  ChatCohere(model="command-r-plus", cohere_api_key= COHERE_API_KEY, temperature= 0.0)
            
            rag_chain: Runnable = (
                    RunnablePassthrough()
                    | llm_audio
                    | StrOutputParser()
            )

            logger.log_message("info", "Getting rag chain for AUDIO in multimodal generation completed successfully.")
            return rag_chain

        except Exception as e:
            logger.log_message("warning", "Failed to get rag chain for AUDIO in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get rag chain for AUDIO in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def meta_data(self, 
                  retriever: QdrantVectorStore,
                  question_str: str,
                  system_str: str,
                  type_str: str,
                  info_str: str,
                  variable_name: str) -> Dict:
        def test_function(docs):
            return " ".join(doc for doc in docs)
        logger.log_message("info", "Getting meta data in multimodal generation started.")
        return {
                    "context_str": RunnablePassthrough( lambda: test_function(retriever)) , 
                    "question_str": RunnablePassthrough(lambda: question_str),
                    "system_str": RunnablePassthrough( lambda: system_str),
                    "type_str": RunnablePassthrough(lambda: type_str),
                    "info_str": RunnablePassthrough(lambda : info_str),
                    "variable_name": RunnablePassthrough(lambda : variable_name),
                } 

            


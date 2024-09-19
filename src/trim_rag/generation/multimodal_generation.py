import os
import sys
from typing import Optional, Union

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

from src.trim_rag.config import MultimodalGenerationArgumentsConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.trim_rag.models import TextModel, ImageModel, AudioModel

class MultimodalGeneration:
    def __init__(self, 
                 config: MultimodalGenerationArgumentsConfig, 
                 llm_text: TextModel, 
                 llm_image: ImageModel, 
                 llm_audio: AudioModel) -> None:
        super(MultimodalGeneration, self).__init__()

        self.config = config
        self.system_str = self.config.system_str

        self.llm_text = llm_text
        self.llm_image = llm_image
        self.llm_audio = llm_audio

    def multimodal_generation(self, 
                              prompt: Optional[str], 
                              image_url: Optional[str], 
                              video_url: Optional[str]
                              ) -> None:
        try:

            logger.log_message("info", "Getting multimodal generation started.")

            _, messages_question = self._create_messages(prompt, image_url, video_url)

            text_rag_chain = self._rag_chain_for_text(prompt, self.llm_text)
            image_rag_chain = self._rag_chain_for_image()
            audio_rag_chain = self._rag_chain_for_audio()
            full_chain = text_rag_chain | image_rag_chain | audio_rag_chain

            response = full_chain.invoke([messages_question])
            logger.log_message("info", "Getting multimodal generation completed successfully.")
            return response

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

    def _rag_chain_for_text(self, 
                            prompt: Optional[str], 
                            retriever) -> Optional[RunnablePassthrough]:
        try:

            logger.log_message("info", "Getting rag chain for TEXT in multimodal generation started.")

            llm_text = self.llm_text()
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm_text
                | StrOutputParser()
            )

            logger.log_message("info", "Getting rag chain for TEXT in multimodal generation completed successfully.")
            return rag_chain

        except Exception as e:
            logger.log_message("warning", "Failed to get rag chain for TEXT in multimodal generation: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get rag chain for TEXT in multimodal generation: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _rag_chain_for_image(self) -> None:
        try:

            logger.log_message("info", "Getting rag chain for IMAGE in multimodal generation started.")

            llm_image = self.llm_image
            rag_chain = (
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

    def _rag_chain_for_audio(self) -> None:
        try:

            logger.log_message("info", "Getting rag chain for AUDIO in multimodal generation started.")

            llm_audio = self.llm_audio
            rag_chain = (
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

            


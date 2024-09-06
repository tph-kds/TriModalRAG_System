import os
import sys
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import PromptFlowsArgumentsConfig

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = "Here is a question that you should answer based on the given context. Write a response that answers the question using only information provided in the context. Provide the answer in Spanish."

# context = """Water boils at 100°C (212°F) at standard atmospheric pressure, which is at sea level.
# However, this boiling point can vary depending on altitude and atmospheric pressure.
# At higher altitudes, where atmospheric pressure is lower, water boils at a lower temperature.
# For example, at 2,000 meters (about 6,600 feet) above sea level, water boils at around 93°C (199°F).
# """
# instruction = "At what temperature does water boil?"


class PromptFlows:

    def __init__(self, config: PromptFlowsArgumentsConfig) -> None:
        super(PromptFlows, self).__init__()

        self.config = config
        self.variable_name = self.config.variable_name

    def prompt_flows(self) -> Optional[ChatPromptTemplate]:
        try:
            logger.log_message("info", "Getting prompt flows started.")
            template =[
                (   "system",
                 
                    "System: {system_str}"
                    "---------------------\n"
                    "Context: {context_str}\n"
                    "---------------------\n"
                ),
                    MessagesPlaceholder(variable_name=self.variable_name),

                (    "---------------------\n"
                    "Human:", 

                    "Metadata for image: {image_str}\n"
                    "---------------------\n"
                    "Metadata for video: {video_str} \n"
                    "---------------------\n"
                    "Question: {question_str}\n"
                ),(
                    "Answer[Bot]: "
            ),]

            prompt = ChatPromptTemplate.from_messages(messages=template)

            logger.log_message("info", "Getting prompt flows completed successfully.")
            return prompt

        except Exception as e:
            logger.log_message("warning", "Failed to get prompt flows: " + str(e))
            my_exception = MyException(
                error_message = "Failed to get prompt flows: " + str(e),
                error_details = sys,
            )
            print(my_exception)


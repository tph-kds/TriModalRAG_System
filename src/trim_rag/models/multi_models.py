from dataclasses import dataclass
import os
import sys

from typing import Optional
from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import MultiModelsArgumentsConfig
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PerceiverModel,
    PerceiverTokenizer,
)

from PIL import Image
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)  # gemini-pro || gemini-pro-vision
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


# model = PerceiverForMultimodal.from_pretrained("deepmind/perceiver-io")
# outputs = model(inputs={'text': text_inputs, 'image': image_inputs, 'audio': audio_inputs})

# tokenizer = GPT2Tokenizer.from_pretrained("openai/multimodal-gpt")
# model = GPT2LMHeadModel.from_pretrained("openai/multimodal-gpt")

# # Assuming embeddings from text, image, and audio are fused here
# input_text = "Generated response from multimodal inputs"
# inputs = tokenizer(input_text, return_tensors="pt")
# outputs = model.generate(**inputs)


@dataclass
class MultiModels:
    def __init__(self, config: MultiModelsArgumentsConfig) -> None:
        super(MultiModels, self).__init__()

        self.config = config
        self.name_of_model = self.config.model_name

    # @classmethod
    def multi_models(self, name: Optional[str] = "gpt2") -> None:
        try:
            logger.log_message("info", "Initializing multimodal models...")
            if name == "gpt2":
                logger.log_message("info", "Initializing GPT-2 multimodal model...")
                tokenizer = GPT2Tokenizer.from_pretrained(self.name_of_model)
                model = GPT2LMHeadModel.from_pretrained(self.name_of_model)

            elif name == "perceiver":
                logger.log_message("info", "Initializing Perceiver multimodal model...")
                tokenizer = PerceiverTokenizer.from_pretrained(self.name_of_model)
                model = PerceiverModel.from_pretrained(self.name_of_model)
            else:
                logger.log_message(
                    "info", "Initializing ChatGoogleGenerativeAI multimodal model..."
                )
                tokenizer = GoogleGenerativeAIEmbeddings.from_pretrained(
                    self.name_of_model
                )
                model = ChatGoogleGenerativeAI.from_pretrained(self.name_of_model)

            logger.log_message("info", "Initialized multimodal models successfully")
            return tokenizer, model

        except Exception as e:
            logger.log_message("info", f"Error initializing multimodal models: {e}")
            my_exception = MyException(
                error_message=f"Error initializing multimodal models: {e}",
                error_details=sys,
            )
            print(my_exception)

    def assert_multi_models(self) -> None:
        try:
            logger.log_message("info", "Asserting multimodal models...")
            if self.name_of_model == None:
                logger.log_message(
                    "info", "No model specified. Please specify a model."
                )
                assert (
                    self.name_of_model != None
                ), "No model specified. Please specify a model."

        except Exception as e:
            logger.log_message("info", f"Error asserting multimodal models: {e}")
            my_exception = MyException(
                error_message=f"Error asserting multimodal models: {e}",
                error_details=sys,
            )
            print(my_exception)

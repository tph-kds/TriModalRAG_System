import os
from pathlib import Path
import sys
from typing import List, Optional

from src.trim_rag.logger import logger  
from src.trim_rag.exception import MyException

from tests.integration import (data_ingestion, 
                               data_processing, 
                               data_embeddings,
                               push_drant_db,
                               data_retriever,
                               data_generation,
                               )
from src.trim_rag.config import ConfiguarationManager
from src.trim_rag.utils import save_tensor,  save_list, load_list, load_tensor

from src.config_params import ROOT_PROJECT_DIR

from tests.integration import data_inference
from langchain_core.runnables import Runnable
from src.trim_rag.embedding import TextEmbedding
from src.trim_rag.config import TextEmbeddingArgumentsConfig
import torch

def convert_qdrantdata_tokens(
        config: TextEmbeddingArgumentsConfig,
        inputs: List
    ) -> Optional[List]:
    """convert qdrant data tokens to list of strings

    Args:
        input (List): list of tokens

    Returns:
        List: list of strings
    """
    text_emebd = TextEmbedding(config=config)
    tokenizer = text_emebd._get_tokenizer()
    list_tokens_id = [torch.Tensor(x.payload["input_ids"]) for x in inputs]
    # list_tokens_id = list_tokens_id[ list_tokens_id != 0]
    # list_tokens_id = list_tokens_id[ list_tokens_id != 101.0]

    list_text = [tokenizer.decode(list_tokens_id[i]) for i in range(len(list_tokens_id))]
    # Remove the unwanted tokens ['[CLS]', '[PAD]', 'SEP']
    list_texts = [lt.replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', '').strip() for lt in list_text]
    # print(list_texts)
    # create a format for top k retriever
    retriever_text = "".join([str(f"{i + 1}. ") + token + "\n" for i, token in enumerate(list_texts)]) 
    return retriever_text

class UpperCaseRunnable(Runnable):
    def invoke(self, input: str) -> str:
        # Converts input to uppercase
        return input.upper()
    
def main( question_str=None,
         image_url=None, 
         video_url=None, 
         query=None
         ):
    
    try:
        logger.log_message("info", "")
        logger.log_message("info", "<<<<<<<< RUNNING A QUICK START SCENARIO >>>>>>>>")
        logger.log_message("info", "A quick start scenarios started.")

        config_manager = ConfiguarationManager()
        embed_config = config_manager.get_data_embedding_arguments_config()

        text_embedding, image_embedding, audio_embedding = data_inference(str(text), str(image), str(audio))
        main_retriever, retriever_text, retriever_image, retriever_audio = data_retriever(text_embedding[0], image_embedding, audio_embedding)
        # print(main_retriever)
        lir_retriever = convert_qdrantdata_tokens(config=embed_config.text_data, 
                                                  inputs=main_retriever
                                                  )
        print(lir_retriever)
        # print(retriever_text)
        # print("Hung")
        # print(retriever_image)
        # print("Hung")
        # print(retriever_audio)

        # retriever = ["hello", "world", "how", "are", "you", "today", "My", "name", "is", "John", "Doe"]
        # inform data  generation stage of the pipeline
        rag_chain, metadata = data_generation(lir_retriever, 
                                    image_url,
                                    video_url,
                                    question_str,
                                    query
                                    )
        print(rag_chain.invoke(metadata))
        
        logger.log_message("info", "A quick start scenarios completed successfully.")
        logger.log_message("info", "<<<<<<<<   END A QUICK START SCENARIOS   >>>>>>>>")
        logger.log_message("info", "")

    except Exception as e:
        logger.log_message("warning", "Failed to run a quick start scenarios: " + str(e))
        my_exception = MyException(
            error_message = "Failed to run a quick start scenarios: " + str(e),
            error_details = sys,
        )
        print(my_exception)

if __name__ == "__main__":
    # question_str = "Yagi's devastation and aftermath: A recap of Vietnam's biggest disaster in decades\
    #         Typhoon Yagi, the most powerful storm to hit Vietnam in 30 years, unleashed devastating floods and landslides across northern provinces, leaving widespread destruction in its wake as shattered communities struggle to recover and rebuild."
    query = "Does Typhoon Yagi have damages in Vietnam country and what were the consequences?"

    text = ROOT_PROJECT_DIR /  ("data/test/file.pdf")
    image = ROOT_PROJECT_DIR / ("data/test/images.jpg")
    audio = ROOT_PROJECT_DIR / ("data/test/audio_lighting.mp3")

    main(text, image, audio, query)
    


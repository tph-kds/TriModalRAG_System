import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


from src.trim_rag.config import ConfiguarationManager
from tests.integration import (
    data_inference,
    data_retriever,
    data_generation,
)
from src.trim_rag.utils import (
    convert_qdrantdata_tokens, 
    convert_qdrantdata_desc
)

def settup_config_llm(inputs: Dict) -> Dict:
    """set up config for llm

    Args:
        inputs (Dict): config for llm
    
        "model_name": inputs["model_name"], 
        "temperature": inputs["temperature"], 
        "max_tokens": inputs["max_tokens"],
        "max_retries": inputs["max_retries"],
        "stop": inputs["stop"]

    Returns:
        Dict: config for llm
    """
    return {
                "model_name": inputs["model"] if "model" in inputs else "gemini-1.0",
                "temperature": inputs["temperature"] if "temperature" in inputs else 0.3,
                "max_tokens": inputs["max_tokens"] if "max_tokens" in inputs else 128,
                "max_retries": inputs["max_retries"] if "max_retries" in inputs else 3,
                "stop": inputs["stop"] if "stop" in inputs else None
        }

def set_up_api_config(inputs: Dict) -> Dict:
    """set up api config

    Args:
        inputs (Dict): api config

        "GOOGLE_API_KEY": inputs["GOOGLE_API_KEY"], 
        "LANGCHAIN_API_KEY" : inputs["LANGCHAIN_API_KEY"], 
        "LANGCHAIN_ENDPOINT" : inputs["LANGCHAIN_ENDPOINT"], 
        "LANGCHAIN_TRACING_V2" : inputs["LANGCHAIN_TRACING_V2"],
        "LANGCHAIN_PROJECT" : inputs["LANGCHAIN_PROJECT"]

    Returns:
        Dict: api config
    """
    return {
        "GOOGLE_API_KEY": inputs["GOOGLE_API_KEY"], 
        "LANGCHAIN_API_KEY" : inputs["LANGCHAIN_API_KEY"], 
        "LANGCHAIN_ENDPOINT" : inputs["LANGCHAIN_ENDPOINT"], 
        "LANGCHAIN_TRACING_V2" : inputs["LANGCHAIN_TRACING_V2"],
        "LANGCHAIN_PROJECT" : inputs["LANGCHAIN_PROJECT"]
    }



    
def result_scenarios( question_str: str = None,
         image_url: str = None, 
         video_url: str = None, 
         query: str = None,
         serving_format: str= None,
         api_config: Dict = None,
         llm_setup: Dict = None
         ) -> Union[str, Tuple[str, List[str]]]:
    
    # try:
        # logger.log_message("info", "")
        # logger.log_message("info", f"<<<<<<<< RUNNING APPLICATION SCENARIOS USING {serving_format.upper()} >>>>>>>>")

        config_manager = ConfiguarationManager()
        embed_config = config_manager.get_data_embedding_arguments_config()

        text_embedding, image_embedding, audio_embedding = data_inference(
            text = str(question_str), 
            image = str(image_url), 
            audio = str(video_url)
        )
        main_retriever, retriever_text, retriever_image, retriever_audio = data_retriever(text_embedding[0], image_embedding, audio_embedding)

        # retriever_text has been synthesized inside the lir_retriever
        lir_retriever = convert_qdrantdata_tokens(config=embed_config.text_data, 
                                                  inputs=main_retriever
                                                  )
        lir_retriever_image = convert_qdrantdata_desc(inputs=retriever_image)
        lir_retriever_audio = convert_qdrantdata_desc(inputs=retriever_audio)

        # inform data  generation stage of the pipeline
        rag_chain, metadata = data_generation(lir_retriever, 
                                    lir_retriever_image,
                                    lir_retriever_audio,
                                    question_str,
                                    query,
                                    api_config=api_config,
                                    llm_config=llm_setup
                                    )
        ai_answer = rag_chain.invoke(metadata)

        # logger.log_message("info", "")
        return ai_answer, metadata

    # except Exception as e:
    #     logger.log_message("warning", f"Failed to run an application scenarios by using {serving_format.upper()}: " + str(e))
    #     my_exception = MyException(
    #         error_message = f"Failed to run a quick start scenarios by using {serving_format.upper()}: " + str(e),
    #         error_details = sys,
    #     )
    #     print(my_exception)


    


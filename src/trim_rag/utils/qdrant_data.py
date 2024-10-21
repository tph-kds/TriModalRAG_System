import os
from pathlib import Path
import sys
from typing import List, Optional

from src.trim_rag.embedding import TextEmbedding
from src.trim_rag.config import TextEmbeddingArgumentsConfig
import torch


def convert_qdrantdata_tokens(
    config: TextEmbeddingArgumentsConfig, inputs: List
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

    list_text = [
        tokenizer.decode(list_tokens_id[i]) for i in range(len(list_tokens_id))
    ]
    # Remove the unwanted tokens ['[CLS]', '[PAD]', 'SEP']
    list_texts = [
        lt.replace("[CLS]", "").replace("[PAD]", "").replace("[SEP]", "").strip()
        for lt in list_text
    ]

    # create a format for top k retriever
    retriever_text = "".join(
        [str(f"{i + 1}. ") + token + "\n" for i, token in enumerate(list_texts)]
    )
    return retriever_text


def convert_qdrantdata_desc(inputs: List) -> Optional[List]:
    """convert qdrant data tokens to list of strings

    Args:
        input (List): list of tokens

    Returns:
        List: list of strings
    """

    list_desc = [x.payload["description"] for x in inputs]

    # create a format for top k retriever
    retriever = "".join(
        [str(f"{i + 1}. ") + des + "\n" for i, des in enumerate(list_desc)]
    )
    return retriever

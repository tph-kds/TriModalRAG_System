import os
import sys
from typing import List, Optional, Tuple

from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config import TextEmbeddingArgumentsConfig
from transformers import BertModel, BertTokenizer
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

# # Initialize BERT model and tokenizer
# model = BertModel.from_pretrained('bert-base-uncased')
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextEmbedding:
    def __init__(self, config: TextEmbeddingArgumentsConfig):
        super(TextEmbedding, self).__init__()
        self.config = config
        self.text_data = self.config

        self.pretrained_model_name = self.text_data.pretrained_model_name
        self.device = self.text_data.device
        self.return_dict = self.text_data.return_dict
        self.max_length = self.text_data.max_length
        self.output_hidden_states = self.text_data.output_hidden_states
        self.do_lower_case = self.text_data.do_lower_case # 
        self.truncation = self.text_data.truncation
        self.return_tensors = self.text_data.return_tensors
        self.padding = self.text_data.padding
        self.max_length = self.text_data.max_length
        self.add_special_tokens = self.text_data.add_special_tokens
        self.return_token_type_ids = self.text_data.return_token_type_ids
        self.return_attention_mask = self.text_data.return_attention_mask
        self.return_overflowing_tokens = self.text_data.return_overflowing_tokens
        self.return_special_tokens_mask = self.text_data.return_special_tokens_mask



    def _get_model(self) -> Optional[BertModel]:
        try:
            logger.log_message("info", f"Loading {self.pretrained_model_name.split('-')[0].upper()} model ...")
            model = BertModel.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name,
                                            return_dict=self.return_dict, 
                                            output_hidden_states=self.output_hidden_states,)
            model = model.to(self.device)
            model.eval()

            logger.log_message("info", f"{self.pretrained_model_name.split('-')[0].upper()} model loaded successfully.")
            return model
        
        except Exception as e:
            logger.log_message("warning", f"Failed to load {self.pretrained_model_name.split('-')[0].upper()} model: " + str(e))
            my_exception = MyException(
                error_message = f"Failed to load {self.pretrained_model_name.split('-')[0].upper()} model: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def _get_tokenizer(self) -> Optional[BertTokenizer]:
        try:
            logger.log_message("info", f"Loading {self.pretrained_model_name.split('-')[0].upper()} tokenizer ...")
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name,
                                                    do_lower_case=self.do_lower_case,
                                                    return_dict=self.return_dict)
            
            logger.log_message("info", f"{self.pretrained_model_name.split('-')[0].upper()} tokenizer loaded successfully.")
            return tokenizer
        
        except Exception as e:
            logger.log_message("warning", f"Failed to load {self.pretrained_model_name.split('-')[0].upper()} tokenizer: " + str(e))
            my_exception = MyException(
                error_message = f"Failed to load  {self.pretrained_model_name.split('-')[0].upper()} tokenizer: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def get_bert_embeddings(self, text: str) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]] :
        try:
            logger.log_message("info", f"Getting {self.pretrained_model_name.split('-')[0].upper()} embeddings ...")
            tokenizer = self._get_tokenizer()
            inputs = tokenizer(text, 
                            return_tensors=self.return_tensors, 
                            truncation=self.truncation, 
                            padding=self.padding,
                            max_length=self.max_length,
                            add_special_tokens=self.add_special_tokens,
                            return_token_type_ids=self.return_token_type_ids,
                            return_attention_mask=self.return_attention_mask,
                            return_overflowing_tokens=self.return_overflowing_tokens,
                            return_special_tokens_mask=self.return_special_tokens_mask
                            )
            input_ids = inputs["input_ids"]
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                text_model = self._get_model()
                outputs = text_model(**inputs)
                embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            # embeddings = embeddings.mean(dim=1)  # Average over sequence length
            # embeddings = embeddings[:, :self.target_dim]
            return embeddings, input_ids

        except Exception as e:
            logger.log_message("warning", f"Failed to get {self.pretrained_model_name.split('-')[0].upper()} embeddings: " + str(e))
            my_exception = MyException(
                error_message = f"Failed to get {self.pretrained_model_name.split('-')[0].upper()} embeddings: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def embedding_text(self, text) -> Optional[torch.Tensor]:
        try:
            logger.log_message("info", "Embedding text started.")
            tokenizer = self._get_tokenizer()
            embeddings, input_ids = self.get_bert_embeddings(text)
            logger.log_message("info", "Embedding text completed successfully.")
            # Convert input IDs back to tokens (for verification)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            # Print tokens and their corresponding embeddings
            tokens_list = [token for token, embedding in zip(tokens, embeddings[0])]
            return embeddings, tokens_list, input_ids

        except Exception as e:
            logger.log_message("warning", "Failed to embed text: " + str(e))
            my_exception = MyException(
                error_message = "Failed to embed text: " + str(e),
                error_details = sys,
            )
            print(my_exception)
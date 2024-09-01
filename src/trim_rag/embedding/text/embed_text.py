import os
import sys

from src.trim_rag.exception import MyException
from src.trim_rag.logger import logger
from src.trim_rag.config import TextEmbeddingArgumentsConfig
from transformers import BertModel, BertTokenizer
import torch

# # Initialize BERT model and tokenizer
# model = BertModel.from_pretrained('bert-base-uncased')
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextEmbedding:
    def __init__(self, config: TextEmbeddingArgumentsConfig):
        super(TextEmbedding, self).__init__()
        self.config = config
        self.text_data = self.config.text_data

        self.pretrained_model_name = self.text_data.pretrained_model_name
        self.device = self.text_data.device
        self.return_dict = self.text_data.return_dict
        self.max_length = self.text_data.max_length
        self.return_hidden_states = self.text_data.return_hidden_states
        self.do_lower_case = self.text_data.do_lower_case
        self.truncation = self.text_data.truncation
        self.return_tensor = self.text_data.return_tensor
        self.padding = self.text_data.padding
        self.max_length = self.text_data.max_length,
        self.add_special_tokens = self.text_data.add_special_tokens,
        self.return_token_type_ids = self.text_data.return_token_type_ids,
        self.return_attention_mask = self.text_data.return_attention_mask,
        self.return_overflowing_tokens = self.text_data.return_overflowing_tokens,
        self.return_special_tokens_mask = self.text_data.return_special_tokens_mask



    def _get_model(self):
        try:
            logger.log_message("info", f"Loading {self.pretrained_model_name.split('-')[0].upper()} model ...")
            model = BertModel.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name,
                                            return_dict=True, 
                                            output_hidden_states=True)
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

    def _get_tokenizer(self):
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

    def get_bert_embeddings(self, text) -> torch.Tensor:
        try:
            logger.log_message("info", f"Getting {self.pretrained_model_name.split('-')[0].upper()} embeddings ...")
            inputs = self._get_tokenizer(text, 
                            return_tensors=self.return_tensor, 
                            truncation=self.truncation, 
                            padding=self.padding,
                            max_length=self.max_length,
                            add_special_tokens=self.add_special_tokens,
                            return_token_type_ids=self.return_token_type_ids,
                            return_attention_mask=self.return_attention_mask,
                            return_overflowing_tokens=self.return_overflowing_tokens,
                            return_special_tokens_mask=self.return_special_tokens_mask
                            )
            with torch.no_grad():
                outputs = self._get_model(**inputs)
                embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            return embeddings

        except Exception as e:
            logger.log_message("warning", f"Failed to get {self.pretrained_model_name.split('-')[0].upper()} embeddings: " + str(e))
            my_exception = MyException(
                error_message = f"Failed to get {self.pretrained_model_name.split('-')[0].upper()} embeddings: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def embedding_text(self, text) -> None:
        try:
            logger.log_message("info", "Embedding text started.")
            embeddings = self.get_bert_embeddings(text)
            logger.log_message("info", "Embedding text completed successfully.")
            return embeddings

        except Exception as e:
            logger.log_message("warning", "Failed to embed text: " + str(e))
            my_exception = MyException(
                error_message = "Failed to embed text: " + str(e),
                error_details = sys,
            )
            print(my_exception)
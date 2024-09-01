
import os
import re
import sys
import nltk
import PyPDF2
import torch
import torch.nn.functional as F

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from typing import Optional
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import TextDataTransformArgumentsConfig


class TextTransform:
    def __init__(self, 
                 config: TextDataTransformArgumentsConfig, 
                 text_path: str= None
                 ):
        
        super(TextTransform, self).__init__()
        self.config = config
        self.text_data = self.config
        # self.text_path = self.text_data.text_path
        self.text_path = text_path

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def _extract_text_from_pdf(self, pdf_path) -> Optional[str]:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ''
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
            return text
        
        except Exception as e:
            logger.log_message("warning", "Failed to extract text from PDF: " + str(e))
            my_exception = MyException(
                
                error_message = "Failed to extract text from PDF: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    ## Normalization 
    def _normalize_text(self, text) -> Optional[str]:
        try: 
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.strip()  # Remove leading/trailing spaces
            return text

        except Exception as e:
            logger.log_message("warning", "Failed to normalize text: " + str(e))
            my_exception = MyException(
                error_message = "Failed to normalize text: " + str(e),
                error_details = sys,
            )
            print(my_exception)
    

    def _remove_stopwords(self, text) -> Optional[str]:
        try:
            tokens = text.split()
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            return ' '.join(filtered_tokens)

        except Exception as e:
            logger.log_message("warning", "Failed to remove stopwords: " + str(e))
            my_exception = MyException(
                error_message = "Failed to remove stopwords: " + str(e),
                error_details = sys,
            )    
            print(my_exception)

    ## Stemming Lemmatization
    def _lemmatize_text(self, text) -> Optional[str]:
        try:
            tokens = text.split()
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized_tokens)
        except Exception as e:
            logger.log_message("warning", "Failed to lemmatize text: " + str(e))
            my_exception = MyException(
                error_message = "Failed to lemmatize text: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    def text_processing(self) -> Optional[str]:
        try:
            logger.log_message("info", "Data processing pipeline started.")
            text = self._extract_text_from_pdf(self.text_path)
            text = self._normalize_text(text)
            text = self._remove_stopwords(text)
            text = self._lemmatize_text(text)
            
            logger.log_message("info", "Data processing pipeline completed successfully.")
            return text


        except Exception as e:
            logger.log_message("warning", "Failed to run data processing pipeline: " + str(e))
            my_exception = MyException(
                error_message = "Failed to run data processing pipeline: " + str(e),
                error_details = sys,
            )
            print(my_exception)




        
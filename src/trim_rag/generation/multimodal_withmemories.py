import os
import sys
import torch
from torch import nn
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException

# from src.trim_rag.config import MultimodalWithMemoriesArgumentsConfig
from src.trim_rag.generation import MultimodalGeneration
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Qdrant





class MultimodalWithMemories(MultimodalGeneration):
    def __init__(self, 
                #  config: MultimodalWithMemoriesArgumentsConfig,
                 qdrant_client) -> None:
        super(MultimodalWithMemories, self).__init__()

        # self.config = config
        # Initialize Memory
        self.memory = ConversationBufferMemory()
        self.qdrant_client = qdrant_client 

    def run_with_memory(self,
                        text_input, 
                        image_input,  
                        audio_input, 
                        **kwargs) -> None:
        """Run the query through TriModal RAG with message history."""

        # Access message history
        past_history = kwargs.get('message_history', [])
        #         Using past_history for Condition-Based Logic
        # You could also use the history to handle specific conditions, such as repeating a query or clarifying based on past responses:
        if len(past_history) > 0:
            # Check if the current query is too similar to the previous one
            last_query = past_history[-1]['input']
            if text_input == last_query:
                return "You just asked the same question. Do you want more details?"
        
        # Step 2: Process the Text Input
        text_embeddings = self.text_embeddings(text_input, )

        # Step 3: Process the Image Input
        image_embeddings = self.image_processor(image_input)

        # Step 4: Process the Audio Input
        audio_embeddings = self.audio_processor(audio_input)

        # Step 5: Fusion of Embeddings
        fused_embeddings = self._fuse_embeddings(text_embeddings, image_embeddings, audio_embeddings)

        # Step 6: Store and Retrieve from Vector Database (Qdrant)
        self.qdrant_client.add_embedding(embedding=fused_embeddings.detach().numpy())

        # Step 7: Generate Final Output using LLM (with fusion as context)
        final_response = self.generative_model(fused_embeddings)

        # Store response in history
        self.memory.save_context(prompt, {"output": final_response})
        
        # Example:

            # [
            #     {"input": "What is the weather?", "output": "It is sunny today."},
            #     {"input": "Show me a picture of the sun.", "output": "<image data>"},
            #     {"input": "What sound does it make?", "output": "<audio data>"}
            # ]


        return final_response

    def _fuse_embeddings(self, text_embed, image_embed, audio_embed):
        """Fuse the embeddings from all three modalities."""
        return torch.cat((text_embed, image_embed, audio_embed), dim=-1)
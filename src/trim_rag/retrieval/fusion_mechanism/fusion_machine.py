import os
import sys
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import FusionMechanismArgumentsConfig

from src.trim_rag.retrieval.fusion_mechanism import (
    AttentionFusion, 
    ModalityAligner, 
    WeightedFusion
)

class FusionMechanism:
    def __init__(self, config: FusionMechanismArgumentsConfig):
        super(FusionMechanism, self).__init__()

        self.config = config
        self.dropout = self.config.dropout

        self.modality_aligner = ModalityAligner(self.config.modality_aligner)
        self.attention_fusion = AttentionFusion(self.config.attention_fusion)
        self.weighted_fusion = WeightedFusion(self.config.weighted_fusion)

    def forward(self, 
                text_results: Optional[List[Tuple[str, float]]] , 
                image_results: Optional[List[Tuple[str, float]]] , 
                audio_results: Optional[List[Tuple[str, float]]]):
        """
        Perform fusion of the retrieval results.
        :param text_results: List of (id, score) tuples for text retrieval.
        :param image_results: List of (id, score) tuples for image retrieval.
        :param audio_results: List of (id, score) tuples for audio retrieval.
        :return: Fused results as a list of (id, fused_score) tuples.
        """
        # Perform modality aligner
        text = self.modality_aligner(text_results)
        image = self.modality_aligner(image_results)
        audio = self.modality_aligner(audio_results)

        # Perform attention fusion
        text = self.attention_fusion(text)
        image = self.attention_fusion(image)
        audio = self.attention_fusion(audio)

        # Perform weighted fusion
        fused_results = self.weighted_fusion(text, image, audio)

        finally_fused_results = self.modality_aligner(fused_results)
        finally_fused_results = nn.Dropout(self.dropout)(finally_fused_results)

        return finally_fused_results

    # def weighted_average(self, text_results, image_results, audio_results):
    #     """
    #     Perform weighted average fusion of the retrieval results.
    #     :param text_results: List of (id, score) tuples for text retrieval.
    #     :param image_results: List of (id, score) tuples for image retrieval.
    #     :param audio_results: List of (id, score) tuples for audio retrieval.
    #     :return: Fused results as a list of (id, fused_score) tuples.
    #     """
    #     results_dict = {}
        
    #     # Aggregate scores with weights
    #     for result in text_results:
    #         id, score = result
    #         if id not in results_dict:
    #             results_dict[id] = 0.0
    #         results_dict[id] += self.weight_text * score
        
    #     for result in image_results:
    #         id, score = result
    #         if id not in results_dict:
    #             results_dict[id] = 0.0
    #         results_dict[id] += self.weight_image * score
        
    #     for result in audio_results:
    #         id, score = result
    #         if id not in results_dict:
    #             results_dict[id] = 0.0
    #         results_dict[id] += self.weight_audio * score
        
    #     # Sort by fused score
    #     fused_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    #     return fused_results
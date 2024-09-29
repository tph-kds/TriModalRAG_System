import os
import sys
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import FusionMechanismArgumentsConfig

from src.trim_rag.retrieval.fusion_mechanism.attention import     AttentionFusion
from src.trim_rag.retrieval.fusion_mechanism.modality_aligner import   ModalityAligner
from src.trim_rag.retrieval.fusion_mechanism.weighted_fusion import WeightedFusion

class FusionMechanism(nn.Module):
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
        print("test")
        if text_results != None:
            # Perform modality aligner
            text = self.modality_aligner(text_results)
            # Perform attention fusion
            text = self.attention_fusion(text, text, text)
        else:
            text = text_results

        if image_results != None:
            image = self.modality_aligner(image_results)
            image = self.attention_fusion(image, image, image)

        else:
            image = image_results

        if audio_results != None:
            audio = self.modality_aligner(audio_results)
            audio = self.attention_fusion(audio)

        else:   
            audio = audio_results

        attention_fusion_results = self.attention_fusion(text, image, audio)
        # Perform weighted fusion
        fused_results = self.weighted_fusion(text, image, audio)    
        if attention_fusion_results != None:
            fused_results = attention_fusion_results + fused_results

        finally_fused_results = self.modality_aligner(fused_results)
        finally_fused_results = nn.Dropout(self.dropout)(finally_fused_results)
        print("Húngdsdsd")

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
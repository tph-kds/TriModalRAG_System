<h1 align="center">TriModal Ritrieval Augmented Generation - TriModalRAG</h1>
<p align="center">
  <img align="center" src="https://github.com/tph-kds/image_storages/blob/ac4db01000f4603602be502753adea6724183f1b/images/svgs/TriModal_RAG/logo.png" width="1200">
</p>

# Tripple Model + Langchain: Find and support data scientists in providing solutions for weather data

...

## Contents
* [Model Overview](#model-overview)
    * [Introduction](#introduction)
    * [Architecture](#architecture)
* [Getting Started](#getting-started)
    * [Install Required Packages](#install-required-packages)
    * [Prepare the Training Data](#prepare-the-training-data)
    * [Models](#models)
* [Inference and Demo](#infer-and-demo)
    * [Results](#results)
    * [Deployment](#deployment)
* [Acknowledgements](#acknowledgements)
* [Future Plans](#future-plans)
* [References](#references)


## Model Overview

### Introduction

The TriModal Retrieval-Augmented Generation (T-RAG) Project is an advanced AI system that combines the power of text, image, and audio data for multi-modal retrieval and generation tasks. This project leverages state-of-the-art deep learning models, and cutting-edge supportive frameworks such as Langchain, DVC, and ZenML. Consequently, a shared embedding space can be built more efficiently where data from all three modalities can be processed, retrieved, and used in a generative pipeline.

The primary goal of this system is to enhance traditional information retrieval by integrating cross-modal knowledge, a fusion mechanism enabling the model to retrieve and generate accurate, context-aware responses that span multiple data types. Whether the task involves answering questions based on text, recognizing patterns in images, or interpreting sounds, the TriModal RAG framework is designed to handle and fuse these distinct types of data into a unified response.

<p align="center">
  <img align="center" src="" width="800">
  
</p>

### Architecture


<p align="center">
  <img align="center" src="https://github.com/tph-kds/image_storages/blob/409dbcf62bbb3ad13e7e02ab323d332d85702487/images/svgs/TriModal_RAG/architecture.png" width="800">
  
</p>

## Getting Started

### Install Required Packages
(It is recommended that the dependencies be installed under the Conda environment.)
```
pip install -r requirements.txt
```
or run [`init_setup.sh`]() file in the project's folder:
```
<!-- Run this command to give the script execution rights: -->

chmod +x init_setup.sh

<!-- Right now, you can execute the script by typing: -->

bash init_setup.sh
```

To be detailed requirements on [Pypi Website](https://pypi.org/project/pip/)

**The required supportive environment uses a hardware accelerator GPUs such as T4 of Colab, GPU A100, etc.**

### Prepare the Training Data

Name | #Text(PDF) | #Image | #Audio
| :------:| :------: | :------: | :-----: |
Quantity | 100 | 100 | 100
Topic | "Machine Learning Weather Prediction" | "Weather" | "Weather" 
Type | API | API | API
Supportive Website | [Arxiv](https://arxiv.org/) | [Unsplash](https://api.unsplash.com/) | [FreeSound](https://freesound.org/)
Feature | Text in research papers | Natural Object - (Non-human) | Natural Sound - (As Rain, Lighting)  

### Models
- The ``BERT`` (Bidirectional Encoder Representations from Transformers) model is used in the TriModal Retrieval-Augmented Generation (RAG) Project to generate high-quality text embeddings. BERT is a transformer-based model pre-trained on vast amounts of text data, which allows it to capture contextual information from both directions (left-to-right and right-to-left) of a sentence. This makes BERT highly effective at understanding the semantic meaning of text, even in complex multi-sentence inputs. [Available on this link](https://huggingface.co/google-bert/bert-base-uncased)

-  The ``CLIP`` (Contrastive Language–Image Pretraining) model, specifically the `openai/clip-vit-base-patch32` variant, is utilized in the TriModal Retrieval-Augmented Generation (RAG) Project. CLIP is a powerful model trained on both images and their textual descriptions, allowing it to learn shared representations between visual and textual modalities. This capability is crucial for multi-modal tasks where text and image data need to be compared and fused effectively.  [Available on this link](https://huggingface.co/openai/clip-vit-base-patch32)

- The ``Wav2Vec 2.0`` Model - `(facebook/wav2vec2-base-960h)`,  is a state-of-the-art speech representation learning framework developed by Facebook AI Research. Applying supportive Embedding to advanced models processes raw audio signals to produce rich, context-aware embeddings, having been pre-trained on a vast corpus of speech data.  Its ability to seamlessly integrate with text and image modalities enhances the project's overall functionality and versatility in handling diverse data types. [Available on this link](https://huggingface.co/facebook/wav2vec2-base-960h)

## Inference And Demo


### Results
...


### Deployment

...

<p align="center">
  <img align="center" src="" width="800">
  
</p>

## Acknowledgements
- [Langchain](https://python.langchain.com/docs/introduction/)

- [Qdrant Vector Database.](https://qdrant.tech/documentation/)

- Use Langchain in multimodal chains connected  to each other. [Read to be more](https://python.langchain.com/v0.1/docs/expression_language/cookbook/multiple_chains/) 

- Apply Rerank Methods by LangChain Cohere library. [Github Here](https://github.com/langchain-ai/langchain-cohere)


- Logo is generated by [@tranphihung](https://github.com/tph-kds)

## Future Plans
- Overall and comprehensive assessment of project performance.
- Upgrade usually and integrate plenty of new positive models.
- Optimal Response capabilities result higher than currently available. 
- Experiment with increasing and expanding larger dataset inputs.


Stay tuned for future releases as we are continuously working on improving the model, expanding the dataset, and adding new features.

Thank you for your interest in my project. We hope you find it useful. If you have any questions, please feel free and don't hesitate to contact me at [tranphihung8383@gmail.com](tranphihung8383@gmail.com)

## References
- Chen, Wenhu, et al. "Murag: Multimodal retrieval-augmented generator for open question answering over images and text." arXiv preprint arXiv:2210.02928 (2022). [Available on this link.](https://arxiv.org/pdf/2210.02928)
<!-- https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization -->



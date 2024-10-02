from setuptools import find_packages,setup
from typing import List


with open("README.md", "r", encoding="utf-8") as f:
    long_descriptions = f.read()

AUTHOR_USER_NAME = "tph-kds"
__version__ = "0.0.1"
NAME_PROJECT = "TriModal_RAG"
PACKAGE_NAME = "trimodal-rag"
description = "A python package for applying TriModal Retrieval-Augmented Generation (RAG) system."
REPO_NAME = "TriModalRAG_System"
AUTHOR_EMAIL = "tranphihung8383@gmail.com"

setup(
    name= PACKAGE_NAME,
    version= __version__ ,
    author= AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description= description,
    long_description= long_descriptions, # open("README.md").read()
    long_description_content_type="text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls= {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues", 
    },
    package_dir={"": "src"},
    install_requires=["transformers","pandas","numpy", "torch", "huggingface-hub", "langchain"],
    packages=find_packages(where="src"),
)
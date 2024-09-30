from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

CONFIG_FILE_PATH = Path("src/config_params/params.yaml")
VECTOR_FILE_PATH = Path("src/config_params/vectordb.yaml")


image_access_key = os.getenv("API_IMAGE_DATA")
audio_access_key = os.getenv("API_AUDIO_DATA")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_DB_URL = os.getenv("QDRANT_DB_URL")

__file__ = os.getcwd()
ROOT_PROJECT_DIR = Path(__file__)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")








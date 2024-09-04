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






from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

CONFIG_FILE_PATH = Path("src/config_params/params.yaml")
image_access_key = os.getenv("API_IMAGE_DATA")
audio_access_key = os.getenv("API_AUDIO_DATA")




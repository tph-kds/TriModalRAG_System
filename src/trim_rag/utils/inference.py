import os
import yaml
import json
import joblib
import base64
import sys
import torch
import pandas as pd
from src.trim_rag.exception import MyException

from box.exceptions import BoxValueError
from ensure import (
    ensure_annotations,
)  # Đảm bảo kiểm tra khi thực hiện sai cú pháp: 1 + "1" == ?
from box import ConfigBox  # truy cập được đối tượng vd: a.ensure thay vì a["ensure"]
from pathlib import Path
from typing import Any, List, Optional


@ensure_annotations
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)
        f.close()


@ensure_annotations
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

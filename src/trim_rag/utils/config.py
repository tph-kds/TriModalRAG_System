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
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Target:
        read yaml file and return

    With Args:
        path_to_yaml (str) : path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml, "r") as f:
            content = yaml.safe_load(f)
            # logger.log_message("info", f"Yaml File: {path_to_yaml} loaded successfully.")

            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        # logger.log_message("warning", f"Error reading yaml file: {e}")
        # my_exception = MyException("Error reading yaml file", sys)
        # print(my_exception)
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Target: Create list of the directories

    With Args:
        path_to_directories (list): list of path of the directories
        verbose (boolean, optional): ignore if multiple dirs is to be created. Defaults to False.

    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        # if verbose:
        #     logger.log_message("info", f"created directory at: {path}")


@ensure_annotations
def save_tensor(data: torch.Tensor, path: Path):
    """save tensor file

    Args:
        data (torch.Tensor): data to be saved as tensor
        path (Path): path to tensor file
    """
    # Save the tensor as a .pt file
    torch.save(data, path)


@ensure_annotations
def load_tensor(path: Path) -> torch.Tensor:
    """load tensor file

    Args:
        path (Path): path to tensor file

    Returns:
        torch.Tensor: data as tensor
    """
    # Load the tensor from the .pt file
    data = torch.load(path)

    return data


@ensure_annotations
def save_list(data: list, path: Path):
    """save list file

    Args:
        data (list): data to be saved as list
        path (Path): path to list file
    """
    # Save the list as a .json file
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


@ensure_annotations
def load_list(path: Path) -> list:
    """load list file

    Args:
        path (Path): path to list file

    Returns:
        list: data as list
    """
    # Load the list from the .json file
    with open(path, "r") as f:
        data = json.load(f)

    return data


@ensure_annotations
def save_csv(data: pd.DataFrame, path: Path):
    """save csv file

    Args:
        data (pd.DataFrame): data to be saved as csv
        path (Path): path to csv file
    """
    data.to_csv(path, index=False)


@ensure_annotations
def load_csv(path: Path) -> pd.DataFrame:
    """load csv file

    Args:
        path (Path): path to csv file

    Returns:
        pd.DataFrame: data as dataframe
    """
    data = pd.read_csv(path)

    return data


# @ensure_annotations
# def load_json(path: Path) -> ConfigBox:
#     """
#         Target: Load Json files data

#         With Args:
#             path (Path): path to json file

#         Returns:
#             ConfigBox: data as class attributes instead of dict
#     """
#     with open(path) as f:
#         content = json.load(f)

#     logger.info(f"Json file loaded successfully from: {path}")
#     return ConfigBox(content)

# @ensure_annotations
# def save_json(path: Path, data: dict):
#     """
#         Target: Save json data

#         With Args:
#             path (Path): path to json file
#             data (dict): data to be saved in json file

#     """
#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)

#     logger.info(f"Json file saved at: {path}")

# @ensure_annotations
# def save_bin(data: Any, path: Path):
#     """save binary file

#     Args:
#         data (Any): data to be saved as binary
#         path (Path): path to binary file
#     """
#     joblib.dump(value=data, filename=path)
#     logger.info(f"binary file saved at: {path}")


# @ensure_annotations
# def load_bin(path: Path) -> Any:
#     """load binary data

#     Args:
#         path (Path): path to binary file

#     Returns:
#         Any: object stored in the file
#     """
#     data = joblib.load(path)
#     logger.info(f"binary file loaded from: {path}")
#     return data

# @ensure_annotations
# def get_size(path: Path) -> str:
#     """get size in KB

#     Args:
#         path (Path): path of the file

#     Returns:
#         str: size in KB
#     """
#     size_in_kb = round(os.path.getsize(path)/1024)
#     return f"~ {size_in_kb} KB"


# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, 'wb') as f:
#         f.write(imgdata)
#         f.close()


# def encodeImageIntoBase64(croppedImagePath):
#     with open(croppedImagePath, "rb") as f:
#         return base64.b64encode(f.read())


# @ensure_annotations
def convert_tensor_to_list(input: Optional[List[torch.Tensor]]) -> Optional[List]:
    """convert tensor to list

    Args:
        x List[(torch.Tensor)]: List of tensor

    Returns:
        list: list
    """
    output_list = [i.tolist() for i in input]
    return output_list


@ensure_annotations
def convert_list_to_tensor(input: List[int]) -> Optional[List[torch.Tensor]]:
    """convert list to tensor

    Args:
        x (List[list]): list

    Returns:
        torch.Tensor: tensor
    """
    output_tensor = [torch.tensor(i) for i in input]

    return output_tensor

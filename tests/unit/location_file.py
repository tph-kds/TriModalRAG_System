import os
import sys
import numpy as np

if __name__ == "__main__":
    text_dir = "data/text" 
    dir_data = os.getcwd()+ "/" + text_dir
    dir_data = dir_data + "/" + os.listdir(dir_data)[0]
    print(dir_data)
    assert os.path.exists(dir_data) == True 
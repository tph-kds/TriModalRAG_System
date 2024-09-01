import os
import sys

if __name__ == "__main__":
    text_dir = "data/text" 
    dir_data = os.getcwd()+ "/" + text_dir
    print(dir_data + "/" + os.listdir(dir_data)[0])

PYTHON_VERSION = 3.9

echo [$(date)]: "START INIT SETUP"


echo [$(date)]: "creating env file with python ${PYTHON_VERSION} version"


conda create -p ./rag_env python==${PYTHON_VERSION} -y


source avtivate ./rag_env


echo [$(date)]: "installing the requirements"


pip install -r requirements.txt


echo [$(date)]: "Setup and install GPU-accelerated libraries if available in the system"


if command -v nvidia-smi &> /dev/null
then
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -U nvidia-cuda-cudnn-cu11.8 nvidia-cuda-cupti-cudnn-cu11.8 nvidia-cuda-nvrtc-cu11-cudnn-cu11.8 nvidia-cuda-runtime-cu11-cudnn-cu11.8
    pip install -U torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

else
    echo "GPU not found"
fi

echo [$(date)]: "END INIT SETUP"
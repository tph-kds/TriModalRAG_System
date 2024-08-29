import os
from pathlib import Path

package_name = "trim_rag"

list_of_files = [
    # Root folder
    ".github/workflows/ci.yaml",
    ".github/workflows/python_publish.yaml",
    ".github/workflows/update_hf.yaml",
    "src/__init__.py",
    f"src/{package_name}/__init__.py",
    # Data folder
    f"src/{package_name}/data/__init__.py",
    f"src/{package_name}/data/images/__init__.py",
    f"src/{package_name}/data/audio/__init__.py",
    f"src/{package_name}/data/text/__init__.py",
    # Config folder
    f"src/{package_name}/config/__init__.py",
    # Inference folder
    f"src/{package_name}/inference/__init__.py",
    # Components folder
    f"src/{package_name}/components/__init__.py",
    # Exception folder
    f"src/{package_name}/exception/__init__.py",
    # Logger folder
    f"src/{package_name}/logger/__init__.py",
    # Models folder
    f"src/{package_name}/models/__init__.py",
    f"src/{package_name}/models/text_model.py",
    f"src/{package_name}/models/image_model.py",
    f"src/{package_name}/models/audio_model.py",
    # Processing folder
    f"src/{package_name}/processing/__init__.py",
    f"src/{package_name}/processing/text/__init__.py",
    f"src/{package_name}/processing/image/__init__.py",
    f"src/{package_name}/processing/audio/__init__.py",
    # Embedding folder
    f"src/{package_name}/embedding/__init__.py",
    f"src/{package_name}/embedding/multimodal_embedding.py",
    # Utils folder
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/utils/config.py",
    f"src/{package_name}/utils/logger.py",
    f"src/{package_name}/utils/exception.py",
    # Retrieval folder
    f"src/{package_name}/retrieval/__init__.py",
    f"src/{package_name}/retrieval/text_retrieval.py",
    f"src/{package_name}/retrieval/image_retrieval.py",
    f"src/{package_name}/retrieval/audio_retrieval.py",
    # Generation folder
    f"src/{package_name}/generation/__init__.py",
    f"src/{package_name}/generation/multimodal_generation.py",

    # Main folder 
    # Inference folder
    "serving/flask_app/app.py",
    "serving/streamlit_app/app.py",
    
    # Tests folder
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    # Init folder
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "Dockerfile",
    "docker-compose.yaml",
    "Makefile",
    # Experiments folder
    "experiments/experiments.ipynb",

]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass # Create a empty file
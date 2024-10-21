from src.config_params import   LANGCHAIN_PROJECT
from dotenv import load_dotenv
load_dotenv()
def test_file():
    
    assert LANGCHAIN_PROJECT == "trimodal_rag"
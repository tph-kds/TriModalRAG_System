
from src.trim_rag.exception import MyException
import sys
def test_log_message():
    try: 

        x = 1/0

    except Exception as e:

        my_exception = MyException(
            error_message = "Failed to run Data Embedding pipeline: " + str(e),
            error_details = sys,
        )
        print(my_exception)
        assert my_exception is not None 
        assert my_exception.error_message == "Failed to run Data Embedding pipeline: " + str(e)



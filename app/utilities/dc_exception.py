from abc import abstractmethod
from fastapi import status

class DcException(Exception):

    @abstractmethod
    def get_code(self):
        pass
    
    @abstractmethod
    def get_message(self):
        pass
class LLMFactoryException(Exception):
    def __init__(self,message: str=None) -> None:
        self.code = status.HTTP_406_NOT_ACCEPTABLE
        if message is None:
            self.message="Exception in LLM factory."
        else:
            self.message=message
    def get_code(self):
        return self.code
    
    def get_messsage(self):
        return self.message

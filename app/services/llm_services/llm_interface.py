from abc import abstractmethod
from app.utilities.singletons_factory import DcSingleton

class LLMInterface(metaclass=DcSingleton):
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_llm(self):
        pass
    
    @abstractmethod
    def generate(self) -> str:
        pass
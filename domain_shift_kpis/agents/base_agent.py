from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class BaseAgent(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        
    @abstractmethod
    def load(self, path: Union[str, Path]):
        pass
    
    @abstractmethod
    def train(self, **kwargs):
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs):
        pass
    
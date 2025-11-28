from abc import ABC, abstractmethod

class DomainShiftBaseClass(ABC):
    def __init__(self, agent, env):
        super().__init__()
        self.agent = agent
        self.env = env
        
    @abstractmethod
    def compute(self):
        pass
    
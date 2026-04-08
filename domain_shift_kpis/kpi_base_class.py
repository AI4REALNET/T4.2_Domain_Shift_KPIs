from abc import ABC, abstractmethod

class DomainShiftBaseClass(ABC):
    def __init__(self, agent, env, env_shift):
        super().__init__()
        self.agent = agent
        self.env = env
        self.env_shift = env_shift
        
    @abstractmethod
    def compute(self):
        pass
    
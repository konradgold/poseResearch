from abc import ABC

class Estimation(ABC):

    @abstractmethod
    def _forward(self, batch) -> torch.Tensor:
        pass
    
    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        state = self._forward(batch)
        if self.output_check(state):
            return state
        else:
            raise RuntimeError(f"{self.identifier} did not return expected output")

    @abstractmethod
    def output_check(self, state) -> bool:
        pass

    @abstractmethod
    @property
    def config(self) -> dict:
        pass

    @abstractmethod
    @property
    def identifier(self) -> str:
        pass
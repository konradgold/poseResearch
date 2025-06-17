from abc import ABC


class StateSaver(ABC):

    @abstractmethod
    def handle(self, state, config):
        pass
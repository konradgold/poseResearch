from abc import ABC


class OutputSaver(ABC):

    @abstractmethod
    def handle(self, output, config):
        pass
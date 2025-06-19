from abc import ABC, abstractmethod


class OutputSaver(ABC):

    @abstractmethod
    def handle(self, output, config):
        pass

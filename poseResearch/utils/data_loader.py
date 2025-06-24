from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def handle(self, output, config):
        pass

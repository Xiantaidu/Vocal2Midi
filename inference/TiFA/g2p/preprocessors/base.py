from abc import ABC, abstractmethod


class Preprocessor(ABC):
    @abstractmethod
    def process(self, tokens: list[str]) -> list[str]:
        ...

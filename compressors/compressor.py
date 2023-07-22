from __future__ import annotations
from abc import ABC, abstractmethod

ENCODING = "UTF-8"

class Compressor(ABC):

    @staticmethod
    @abstractmethod
    def new_instance() -> Compressor:
        raise NotImplementedError()

    @abstractmethod
    def fit(self, data: str) -> Compressor:
        raise NotImplementedError()

    @abstractmethod
    def get_compressed_len(self, text: str):
        raise NotImplementedError()
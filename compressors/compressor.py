from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

ENCODING = "UTF-8"

class Compressor(ABC):

    @abstractmethod
    def fit(self, texts: List[str]) -> Compressor:
        raise NotImplementedError()

    @abstractmethod
    def get_compressed_len(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def dictionary_size(self) -> int:
        raise NotImplementedError()
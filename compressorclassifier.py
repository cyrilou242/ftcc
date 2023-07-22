from collections import defaultdict
from typing import Tuple, Dict, Callable, List

from compressors.compressor import Compressor


class CompressorClassifier:

    # if top_k is set to > 1, cheat as described in https://github.com/bazingagin/npc_gzip/issues/3
    # only set top_k=2 to show how cheating improves accuracy performance by a crazy amount
    def __init__(self, compressor_provider: Callable[[], Compressor], top_k=1):
        self.compressor_provider = compressor_provider
        if top_k < 1:
            raise ValueError("Invalid top_k value. Correct value is 1. Cheat is 2 or more.")
        self.top_k = top_k

    # train_pair is a list of [(label, observation), ...
    # todo see if string concatenation can be improved
    def fit(self, train_pair: List[Tuple[str, str]]):
        # concatenate strings that have the same labels
        label_to_texts = defaultdict(list)
        for label, observation in train_pair:
            label_to_texts[label].append(observation)

        self.label_to_compressor = {label: self.compressor_provider().fit('\n'.join(texts)) for label, texts in
                                    label_to_texts.items()}

    def predict(self, text):
        label_to_score = {label: compressor.get_compressed_len(text) for label, compressor in
                          self.label_to_compressor.items()}

        res = set({})
        for _ in range(self.top_k):
            predicted = min(label_to_score, key=label_to_score.get)
            label_to_score.pop(predicted)
            res.add(predicted)
        return res

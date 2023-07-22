from collections import defaultdict
from math import ceil
from typing import Tuple, Callable, List

from compressors.compressor import Compressor


class CompressorClassifier:

    # if top_k is set to > 1, cheat as described in https://github.com/bazingagin/npc_gzip/issues/3
    # only set top_k=2 to show how cheating improves accuracy performance by a crazy amount
    def __init__(self, compressor_provider: Callable[[], Compressor], top_k=1, num_compressors_per_class=1):
        self.compressor_provider = compressor_provider
        if top_k < 1:
            raise ValueError("Invalid top_k value. Correct value is 1. Cheat is 2 or more.")
        self.top_k = top_k
        self.num_compressors_per_class = num_compressors_per_class

    # train_pair is a list of [(label, observation), ...
    # todo see if string concatenation can be improved
    def fit(self, train_pair: List[Tuple[str, str]]):
        # concatenate strings that have the same labels
        label_to_texts = defaultdict(list)
        for label, observation in train_pair:
            label_to_texts[label].append(observation)

        self.label_to_compressors = {}
        for label, texts in label_to_texts.items():
            compressors = []
            step = ceil(len(texts) / self.num_compressors_per_class)
            for i in range(0, len(texts), step):
                compressor = self.compressor_provider().fit('\n'.join(texts[i:i + step]))
                compressors.append(compressor)
            self.label_to_compressors[label] = compressors

    def predict(self, text):
        label_to_scores = {label: [c.get_compressed_len(text) for c in compressors] for label, compressors in
                           self.label_to_compressors.items()}

        # TODO CYRIL add DI for the strategy on how to pick
        # reduced scores could be a vote, a sum, etc. not sure for the moment take the sum
        label_to_score = {label: sum(scores) for label, scores in label_to_scores.items()}
        res = []
        for _ in range(self.top_k):
            predicted = min(label_to_score, key=label_to_score.get)
            label_to_score.pop(predicted)
            res.append(predicted)
        return res

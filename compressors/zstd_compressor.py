from __future__ import annotations

from typing import List

import zstandard

from compressors.compressor import Compressor, ENCODING


class ZstdCompressor(Compressor):

    def __init__(self, size: int = -1, compression_level=9):
        self.compression_level = compression_level
        self.size = size

    def fit(self, data: List[str]) -> ZstdCompressor:
        if self.size < -1:
            raise ValueError("size must be -1, 0 or an integer")
        if self.size == -1:
            # -1: special value - the whole dataset is set maintained in memory and set as prefix for compression
            combined_texts = '\n'.join(data)
            self.dictionary = zstandard.ZstdCompressionDict(combined_texts.encode(ENCODING),
                                                            dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        else:
            # 0: special value - the dictionary size is unbounded, but optimized
            # we set unbounded at ~10Gb. Should be enough for the moment
            size_limit = int(1e10) if self.size == 0 else self.size
            try:
              self.dictionary = zstandard.train_dictionary(size_limit, [e.encode(ENCODING) for e in data], split_point=1,
                                                         level=self.compression_level)
            except Exception as e:
                if "Src size is incorrect" in str(e):
                    print("WARNING - Could not train dictionary. Not enough data. Using the whole training data as compressor prefix.")
                    combined_texts = '\n'.join(data)
                    self.dictionary = zstandard.ZstdCompressionDict(combined_texts.encode(ENCODING),
                                                                    dict_type=zstandard.DICT_TYPE_RAWCONTENT)
                else:
                    raise e
        # can be improved for perf - params = zstandard.ZstdCompressionParameters(...)
        self.dictionary.precompute_compress(level=self.compression_level)
        self.compressor = zstandard.ZstdCompressor(dict_data=self.dictionary)

        return self

    def get_compressed_len(self, text: str):
        compressed = self.compressor.compress(text.encode(ENCODING))
        return len(compressed)

    def dictionary_size(self):
        return len(self.dictionary.as_bytes())


if __name__ == '__main__':
    # fixme this is broken
    compressor = ZstdCompressor(size=0)
    compressor.fit(
        "Cras in porta nunc, non volutpat nisi. Nunc quis euismod dui, sit amet convallis turpis. Nulla nisl massa, faucibus vitae mauris eget, placerat porta enim. Aenean molestie pulvinar dolor vitae ultrices. Nulla quis mi convallis, malesuada est eu, tempus arcu. Donec aliquet libero tristique orci ornare porttitor. Etiam cursus fermentum ultricies. Ut purus nisl, maximus in elit in, tempor luctus nulla. Aenean mattis auctor blandit. Aenean fringilla turpis massa, a faucibus libero gravida sed. Quisque sit amet tortor interdum, fermentum orci nec, suscipit sem. Praesent eu dapibus lacus.".split("."))
    print(compressor.dictionary_size())

    l1 = compressor.get_compressed_len(
        "Sed in diam nec lacus hendrerit mattis. Sed non vulputate eros. Vestibulum aliquam massa et sollicitudin porta. Cras justo nisl, lacinia ornare enim ut, ornare maximus libero. Sed consectetur commodo nisi vel lobortis. Sed at tempus enim, in condimentum velit. Praesent vestibulum consectetur laoreet. Donec congue accumsan metus molestie lacinia. Pellentesque pharetra rhoncus ipsum, nec porta sem cursus sed. Nulla ut magna ultrices justo scelerisque tincidunt. Curabitur sit amet risus porta, placerat diam vel, vulputate magna. Aliquam vitae mattis turpis, quis malesuada enim.")
    print(f"Size of similar text: {l1}")

    l2 = compressor.get_compressed_len(
        "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.")
    print(f"Size of different text: {l2}")

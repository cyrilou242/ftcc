from __future__ import annotations
import zstandard

from compressors.compressor import Compressor, ENCODING


class ZstdCompressor(Compressor):

    @staticmethod
    def new_instance() -> ZstdCompressor:
        return ZstdCompressor()

    def fit(self, data: str) -> ZstdCompressor:
        # size can be controlled <--> the size/performance ratio can be tuned based on technical requirements - see zstandard.train_dictionary(size, samples)
        self.dictionary = zstandard.ZstdCompressionDict(data.encode(ENCODING), dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        # can be improved for perf - see precompute_compress
        self.compressor = zstandard.ZstdCompressor(dict_data=self.dictionary)

        return self

    def get_compressed_len(self, text: str):
        compressed = self.compressor.compress(text.encode(ENCODING))
        return len(compressed)


if __name__ == '__main__':
    compressor = ZstdCompressor()
    compressor.fit("Cras in porta nunc, non volutpat nisi. Nunc quis euismod dui, sit amet convallis turpis. Nulla nisl massa, faucibus vitae mauris eget, placerat porta enim. Aenean molestie pulvinar dolor vitae ultrices. Nulla quis mi convallis, malesuada est eu, tempus arcu. Donec aliquet libero tristique orci ornare porttitor. Etiam cursus fermentum ultricies. Ut purus nisl, maximus in elit in, tempor luctus nulla. Aenean mattis auctor blandit. Aenean fringilla turpis massa, a faucibus libero gravida sed. Quisque sit amet tortor interdum, fermentum orci nec, suscipit sem. Praesent eu dapibus lacus.")

    l1 = compressor.get_compressed_len("Sed in diam nec lacus hendrerit mattis. Sed non vulputate eros. Vestibulum aliquam massa et sollicitudin porta. Cras justo nisl, lacinia ornare enim ut, ornare maximus libero. Sed consectetur commodo nisi vel lobortis. Sed at tempus enim, in condimentum velit. Praesent vestibulum consectetur laoreet. Donec congue accumsan metus molestie lacinia. Pellentesque pharetra rhoncus ipsum, nec porta sem cursus sed. Nulla ut magna ultrices justo scelerisque tincidunt. Curabitur sit amet risus porta, placerat diam vel, vulputate magna. Aliquam vitae mattis turpis, quis malesuada enim.")
    print(f"Size of similar text: {l1}")

    l2 = compressor.get_compressed_len(
        "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.")
    print(f"Size of different text: {l2}")







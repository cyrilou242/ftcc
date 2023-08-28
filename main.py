import os
import time
import csv

import click
import numpy as np
from py_markdown_table.markdown_table import markdown_table
from torchtext.datasets import AG_NEWS, IMDB, AmazonReviewPolarity, DBpedia, SogouNews, YahooAnswers, YelpReviewPolarity

from compressorclassifier import CompressorClassifier
from compressors.zstd_compressor import ZstdCompressor
from data import load_20news, load_ohsumed_single_23, load_reuters, load_kinnews_kirnews

def write_csv(filename, data):
    headers = [x for x in data[0].keys()]
    with open(os.path.join(os.getcwd(), filename), mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

DATA_DIR = "data"

DATASET_TO_LOADER = {
    "AG_NEWS": lambda: AG_NEWS(root=DATA_DIR),
    "IMDB": lambda: IMDB(root=DATA_DIR),
    # for google drive links, you may have to comment line 22 to line 31 in torchtext._download_hooks.py
    "AmazonReviewPolarity": lambda: AmazonReviewPolarity(root=DATA_DIR),
    "DBpedia": lambda: DBpedia(root=DATA_DIR),
    # FIXME CYRIL - loading is broken
    # "SogouNews": lambda: SogouNews(root=DATA_DIR),
    "YahooAnswers": lambda: YahooAnswers(root=DATA_DIR),
    "YelpReviewPolarity": lambda: YelpReviewPolarity(root=DATA_DIR),
    # on mac OS, you may have to run something like: ln -s /etc/ssl/* /Library/Frameworks/Python.framework/Versions/3.7/etc/openssl
    # to fix urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1091)>
    "20News": load_20news,
    # to compare with Max Halford's article https://maxhalford.github.io/blog/text-classification-by-compression/
    "4News": lambda: load_20news(categories=['alt.atheism', 'talk.religion.misc', 'comp.graphics','sci.space']),
    "R8": lambda: load_reuters(os.path.join(DATA_DIR, "R8")),
    "R52": lambda: load_reuters(os.path.join(DATA_DIR, "R52")),
    "Ohsumed": lambda: load_ohsumed_single_23(os.path.join(DATA_DIR, "ohsumed_single_23")),
    # FIXME need to implement downloader from https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/dengue/dengue_raw.zip
    # FIXME CYRIL - hugging face dataset is contaminated - https://github.com/bazingagin/npc_gzip/issues/13
    "kinnews": lambda: load_kinnews_kirnews(
        dataset_name="kinnews_kirnews", data_split="kirnews_cleaned"
    ),
    # "swahili": load_swahili,
    # FIXME CYRIL - dataset is contaminated - https://github.com/bazingagin/npc_gzip/issues/13
    # "filipino": load_filipino,
}

COMPRESSOR_PROVIDERS = {
    "ZSTD_CL15": lambda size: ZstdCompressor(size=size, compression_level=15),
    "ZSTD_CL12": lambda size: ZstdCompressor(size=size, compression_level=12),
    "ZSTD_CL10": lambda size: ZstdCompressor(size=size, compression_level=10),
    "ZSTD_CL9": lambda size: ZstdCompressor(size=size, compression_level=9),
    "ZSTD_CL6": lambda size: ZstdCompressor(size=size, compression_level=6),
    "ZSTD_CL3": lambda size: ZstdCompressor(size=size, compression_level=3)
}


@click.command()
@click.option("-d", "--dataset", help='Dataset', type=click.Choice(list(DATASET_TO_LOADER.keys())), multiple=True,
              default=list(DATASET_TO_LOADER.keys()))
@click.option("-c", "--compressor",
              help='Compressor implementation/configuration.',
              type=click.Choice(list(COMPRESSOR_PROVIDERS.keys())), multiple=True,
              default=["ZSTD_CL9"])
@click.option("--top_k_accuracy",
              help='Top k cheat as described in https://github.com/bazingagin/npc_gzip/issues/3. Set it to 2 to cheat on the accuracy measure like in the original paper. Should be kept to 1 for proper performance analysis.',
              type=click.Choice(["1", "2"]), multiple=True,
              default=["1"])
@click.option("-cpc", "--compressors_per_class",
              help="Number of compressors per class. A value of 3 or 5 can make the model more stable. Should be kept small when the number of classes is big to keep inference fast.",
              multiple=True,
              default=[1, 3, 5])
@click.option("-s", "--size",
              help="Constraint on the size of the created dictionaries, in bytes. Each generated dictionary will have a size smaller or equal to this value. Special value -1 means the whole training dataset is maintained in memory. Special value 0 means the size of the dictionary is unbounded, but optimized.",
              multiple=True,
              default=[-1])
def run_experiment(dataset, compressor, top_k_accuracy, compressors_per_class, size):
    # convert k to int - see click issue https://github.com/pallets/click/issues/784
    top_k_accuracy = [int(k) for k in top_k_accuracy]

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    results = []
    speed_results = []
    size_results = []
    for s in size:
        for k in top_k_accuracy:
            for cpc in compressors_per_class:
                for c in compressor:
                    size_message = "dataset_prefixed" if s == -1 else (
                        "size_unbounded_optimized" if s == 0 else f"size_bounded_{s}")
                    method_name = f"FFTC {c} {size_message} CPC_{cpc}" + (f" top_{k} accuracy" if k > 1 else "")
                    method_result = {"Method": method_name}
                    speed_result = {"Method": method_name}
                    size_result = {"Method": method_name}
                    for d in dataset:
                        loader = DATASET_TO_LOADER[d]
                        print(f"Loading dataset {d}. It will be downloaded if not available in the {DATA_DIR} folder.")
                        dataset_pair = loader()
                        train_pair, test_pair = dataset_pair[0], dataset_pair[1]

                        print(f"Training classifier {method_name} for dataset {d}.")
                        compressor_provider = COMPRESSOR_PROVIDERS[c]
                        classifier = CompressorClassifier(lambda: compressor_provider(s), k,
                                                          num_compressors_per_class=cpc)
                        start = time.monotonic()
                        classifier.fit(train_pair)
                        training_time = time.monotonic() - start

                        # todo extract this
                        print(f"Running evaluation for classifier {method_name} for dataset {d}.")
                        run_times_millis = []
                        obs_count = 0
                        correct_obs_count = 0
                        for (label, observation) in test_pair:
                            start = time.monotonic()
                            predicted = classifier.predict(observation)
                            end = time.monotonic()
                            run_times_millis.append((end - start) * 1000)
                            obs_count += 1
                            if label in predicted:
                                # if predicted == label:
                                correct_obs_count += 1

                        accuracy = correct_obs_count / obs_count
                        print(
                            f"Accuracy on dataset {d}: {accuracy * 100}%. \nTraining time: {training_time}s. \nPrediction times: p50: {np.percentile(run_times_millis, 50)}ms, p90: {np.percentile(run_times_millis, 90)}ms, p99: {np.percentile(run_times_millis, 99)}ms.")
                        method_result[d] = accuracy
                        size_result[d] = f"{classifier.dictionaries_size() / 1e6} Mb"
                        speed_result[d + "_train"] = f"{round(training_time, 1)}s"
                        speed_result[d + "_predict_p90"] = f"{round(np.percentile(run_times_millis, 90), 3)}ms"
                    results.append(method_result)
                    speed_results.append(speed_result)
                    size_results.append(size_result)

    write_csv('accuracy_results.csv', results)
    write_csv('speed_results.csv', speed_results)
    write_csv('size_results.csv', size_results)

    accuracy_table = markdown_table(results).set_params(float_rounding=3).get_markdown()
    print(accuracy_table)
    speed_table = markdown_table(speed_results).set_params(float_rounding=3).get_markdown()
    print(speed_table)
    size_table = markdown_table(size_results).set_params(float_rounding=0).get_markdown()
    print(size_table)



if __name__ == '__main__':
    run_experiment()

import os
import time

import click
import numpy as np
from py_markdown_table.markdown_table import markdown_table
from torchtext.datasets import AG_NEWS, IMDB, AmazonReviewPolarity, DBpedia, SogouNews, YahooAnswers, YelpReviewPolarity

from compressorclassifier import CompressorClassifier
from compressors.zstd_compressor import ZstdCompressor
from data import load_20news, load_ohsumed, load_ohsumed_single, load_r8, load_kinnews_kirnews, load_swahili, \
    load_filipino

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
    # FIXME CYRIL need to implement a proper downloader
    # "Ohsumed": lambda: load_ohsumed(os.path.join(DATA_DIR, "Ohsumed")),
    # "Ohsumed_single": lambda: load_ohsumed_single(os.path.join(DATA_DIR, "Ohsumed_single")),
    # FIXME CYRIL need to implement a proper downloader
    # "R8": lambda: load_r8(os.path.join(DATA_DIR, "R8")),
    # fixme cyril not sure to understand why this is the same loader
    # FIXME CYRIL need to implement a proper downloader
    # "R52": lambda: load_r8(os.path.join(DATA_DIR, "R52")),
    # FIXME need to implement downloader from https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/dengue/dengue_raw.zip
    # FIXME CYRIL - dataset is contaminated - https://github.com/bazingagin/npc_gzip/issues/13
    "kinnews": lambda: load_kinnews_kirnews(
        dataset_name="kinnews_kirnews", data_split="kirnews_cleaned"
    ),
    # "swahili": load_swahili,
    # FIXME CYRIL - dataset is contaminated - https://github.com/bazingagin/npc_gzip/issues/13
    # "filipino": load_filipino,
}

COMPRESSOR_PROVIDERS = {
    "ZSTD_CL18": lambda: ZstdCompressor(compression_level=18),
    "ZSTD_CL15": lambda: ZstdCompressor(compression_level=15),
    "ZSTD_CL12": lambda: ZstdCompressor(compression_level=12),
    "ZSTD_CL9": lambda: ZstdCompressor(compression_level=9),
    "ZSTD_CL6": lambda: ZstdCompressor(compression_level=6),
    "ZSTD_CL3": lambda: ZstdCompressor(compression_level=3)
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
def run_experiment(dataset, compressor, top_k_accuracy, compressors_per_class):
    # convert k to int - see click issue https://github.com/pallets/click/issues/784
    top_k_accuracy = [int(k) for k in top_k_accuracy]

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    results = []
    speed_results = []
    for k in top_k_accuracy:
        for cpc in compressors_per_class:
            for c in compressor:
                method_name = f"FFTC {c} CPC_{cpc}" + (f" top_{k} accuracy" if k > 1 else "")
                method_result = {"Method": method_name}
                speed_result = {"Method": method_name}
                for d in dataset:
                    loader = DATASET_TO_LOADER[d]
                    print(f"Loading dataset {d}. It will be downloaded if not available in the {DATA_DIR} folder.")
                    dataset_pair = loader()
                    train_pair, test_pair = dataset_pair[0], dataset_pair[1]

                    print(f"Training classifier for dataset {d}.")
                    classifier = CompressorClassifier(lambda: ZstdCompressor(), k, num_compressors_per_class=cpc)
                    start = time.monotonic()
                    classifier.fit(train_pair)
                    training_time = time.monotonic() - start

                    print(f"Running evaluation for dataset {d}.")
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
                    speed_result[d + "_train"] = f"{round(training_time, 1)}s"
                    speed_result[d + "_predict_p90"] = f"{round(np.percentile(run_times_millis, 90), 3)}ms"
                results.append(method_result)
                speed_results.append(speed_result)

    # todo add save in proper tabular format
    accuracy_table = markdown_table(results).set_params(float_rounding=3).get_markdown()
    print(accuracy_table)
    speed_table = markdown_table(speed_results).set_params(float_rounding=3).get_markdown()
    print(speed_table)


if __name__ == '__main__':
    run_experiment()

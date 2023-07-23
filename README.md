# FTCC

**FTCC**: **F**ast **T**ext **C**lassification with **C**ompressors dictionary

This repository implements a compression based classification technique that is fast at training **AND** 
at inference. 
It follows [Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip) (referred as the *gzip method* from now on)
that received a lot of attention.
The paper mentions different techniques based on compression, but does not mention the technique implemented in 
this repository.  
I suspect the **FTCC** technique in this repository has already been tested in the industry, but I could not find an 
implementation online, so here you go. I think it is a reasonable baseline for compression based classification. 
It is **multiple orders of magnitudes faster** than *the gzip method* and the other compression-based classifiers mentioned in the paper, 
and has comparable accuracy (*yet to validate*).
I'll try to make this package easy to use in the future.

## Principle
Some compressor implementation, such as `zstd`, can generate a compression dictionary from data.

> A compression dictionary is essentially data used to seed the compressor state so it can achieve better compression. 
> The idea is that if you are compressing a lot of similar pieces of data (e.g. JSON documents or anything 
> sharing similar structure), then you can find common patterns across multiple objects then leverage 
> those common patterns during compression and decompression operations to achieve better compression ratios.
> Dictionary compression is generally only useful for small inputs - data no larger than a few kilobytes.

*[source](https://python-zstandard.readthedocs.io/en/latest/concepts.html#dictionaries)*

Intuitively, this means: if I "train" a compressor on a specific topic, it will get better at compressing this 
particular topic. I can train a different compressor for each topic.
Then given an input text, I can compare all compressor. The topic of the compressor that has the 
best compression ratio is most likely the topic of the input text.
This collection of compressors is the text classification model.

Algorithm:
##### training
- Given a dataset of `[(text, class)]`
- Join texts with the same label together:
  ```
  { 
    class_1: ct_1,
    class_2: ct_2,
    ...
    class_n: ct_n, 
  } 
  ```
  With `ct_i`  all texts of label `i`, joined together with the separator `\n`.
- For each label, generate a compression dictionary
  ```
  { 
    class_1: class1_compressor,
    class_2: class2_compressor,
    ...
    class_n: class3_compressor, 
  }
  ```
- save the above as `class_to_compressor`.

##### inference
- Given a new input `text`
- For each compression dictionary, compress the input, and get the length. Return the class of the compressor that achieved the best compression ratio.
   ```
   predicted_class = None
   minimum_size = INF
   for class, compressor in class_to_compressor:
      size = length(compressor.compress(text))
      if size < minimum_size:
        predicted_class = class
        minimum_size = size
  return predicted_class
  ```

To further improve the algorithm, instead of building a single compressor per class, we build a few 
of them. For instance, we split a given class text in 3 chunks, and train 1 compressor for each chunk.
At inference, we take the average of the compression ratio (we could also use a vote approach). 
This stabilizes the inferences, and small values between 2 and 5 are enough. 
Below we call the number of **c**ompressors **p**er **c**lass the CPC.

### Properties
This technique has the following nice properties:
- runs on low resources machines easily. 
  **Training takes a few seconds, inferences between 0.1 and 20 milliseconds - on commodity hardware.**
- you can control the accuracy/model size trade-off easily. 
  *Because most dictionary based compressor allow to control the size of the compression dictionary.*
- you can control the accuracy/inference speed trade-off easily
  *By setting the compression level of the compressor. A higher compression level has higher accuracy but is slower.*
- you can control the accuracy/inference speed trade-off by setting the CPC.
  *A bigger CPC improves the accuracy and does not make the training slower, only the inference.*
- you can perform partial - per-class - re-training. 
  *Compressors are trained per class. If a class often has false negative, you can improve the compressor of this class independently of the other compressors.*
- it is fast and different from traditional approaches, so a good candidate for ensembling
- reproducible. No `random()` here and there.

## Performance analysis

### `dataset_prefixed` mode
`dataset_prefixed` means the whole dataset is kept in memory by the compressor and used as a prefix. This makes the "training" very fast, because nothing is done. 
This means the model has the same size as the training input. This corresponds to the setup of the *gzip method*.

#### Accuracy
```
+------------------------------------------------------------------------------------------------------------------------------------------+
|               Method               |AG_NEWS| IMDB|AmazonReviewPolarity|DBpedia|YahooAnswers|YelpReviewPolarity|20News|  R8 | R52 |kinnews|
+------------------------------------+-------+-----+--------------------+-------+------------+------------------+------+-----+-----+-------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_1| 0.863 |0.691|        0.716       | 0.931 |    0.534   |       0.771      | 0.783|0.917|0.846| 0.892 |
+------------------------------------+-------+-----+--------------------+-------+------------+------------------+------+-----+-----+-------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_3| 0.896 |0.773|        0.799       | 0.959 |    0.628   |       0.841      | 0.795|0.935|0.007| 0.883 |
+------------------------------------+-------+-----+--------------------+-------+------------+------------------+------+-----+-----+-------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_5| 0.901 | 0.8 |        0.83        | 0.965 |    0.655   |       0.859      | 0.79 |0.937|0.001| 0.881 |
+------------------------------------------------------------------------------------------------------------------------------------------+
```

*Some datasets are not included because they have train/test split issues. See [here](https://github.com/bazingagin/npc_gzip/issues/13).*  
*Some other datasets are not included yet because the downloader function seemed broken. Help me fix this! See [contribute](#extend-and-contribute)*

Comparison with [A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip) (referred as *gzip method* below) on test accuracy: 

| method                                              | AGNews | DBpedia | YahooAnswers   | 20News        | kinnews    | R8    | R52    |
|-----------------------------------------------------|--------|---------|----------------|---------------|------------|-------|--------|
| gzip method top1                                    |        |         |                |               | 0.835 [1\] |       |        |
| FFTC ZSTD_CL9 dataset_prefixed CPC_5 (this project) | 0.901  | 0.965   | **0.655** [2\] | **0.79** [2\] | **0.881**  | 0.937 | NA [3] |
| FFTC ZSTD_CL9 dataset_prefixed CPC_1 (this project) |        |         |                |               |            |       | 0.846  |


**CAUTION** 
I am not showing top 2 scores from the paper because the implementation leaks the labels at prediction time and overestimates accuracy. See [issue](https://github.com/bazingagin/npc_gzip/issues/3).
I am currently recomputing top 1 with the provided source code. The gzip method takes hours to run, I'll update the numbers here incrementally. 
If you have the resources to run for one dataset (16Gb ram VM, 48 hours), please reach out!  

[1] taken from knn1 [here](https://kenschutte.com/gzip-knn-paper/)  
[2] means *FFTC ZSTD_CP9 CPC_5* already outperforms *gzip*, even with the accuracy overestimate in the *gzip* paper.  
[3] having CPC > 1 is not relevant for 52 categories with the little amount of data for some class, or requires a more subtle implementation, with cpc depending on the number of observations for each class

#### Speed
*Below is just to give an idea. Run on my 2021 intel MacBook Pro, with tons of apps running at the same time. Do your own microbenchmark.*
The computation is multiple orders of magnitudes faster [A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip).

```
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|               Method               |AG_NEWS_train|AG_NEWS_predict_p90|IMDB_train|IMDB_predict_p90|AmazonReviewPolarity_train|AmazonReviewPolarity_predict_p90|DBpedia_train|DBpedia_predict_p90|YahooAnswers_train|YahooAnswers_predict_p90|YelpReviewPolarity_train|YelpReviewPolarity_predict_p90|20News_train|20News_predict_p90|R8_train|R8_predict_p90|R52_train|R52_predict_p90|kinnews_train|kinnews_predict_p90|
+------------------------------------+-------------+-------------------+----------+----------------+--------------------------+--------------------------------+-------------+-------------------+------------------+------------------------+------------------------+------------------------------+------------+------------------+--------+--------------+---------+---------------+-------------+-------------------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_1|     2.4s    |      0.123ms      |   1.3s   |     0.368ms    |          111.7s          |             0.136ms            |    15.8s    |      0.878ms      |       48.1s      |         1.363ms        |          18.1s         |            0.328ms           |    0.6s    |      6.537ms     |  0.0s  |    0.768ms   |   0.1s  |    5.846ms    |     0.1s    |      3.885ms      |
+------------------------------------+-------------+-------------------+----------+----------------+--------------------------+--------------------------------+-------------+-------------------+------------------+------------------------+------------------------+------------------------------+------------+------------------+--------+--------------+---------+---------------+-------------+-------------------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_3|     2.6s    |      0.358ms      |   1.2s   |     1.462ms    |           93.9s          |             0.451ms            |    13.3s    |      2.547ms      |       52.4s      |         3.861ms        |          21.1s         |            0.939ms           |    0.4s    |     15.202ms     |  0.0s  |    2.408ms   |   0.1s  |    13.566ms   |     0.1s    |       8.92ms      |
+------------------------------------+-------------+-------------------+----------+----------------+--------------------------+--------------------------------+-------------+-------------------+------------------+------------------------+------------------------+------------------------------+------------+------------------+--------+--------------+---------+---------------+-------------+-------------------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_5|     3.0s    |      0.696ms      |   1.3s   |     2.343ms    |          109.8s          |             0.877ms            |    16.6s    |      3.867ms      |       50.5s      |         6.263ms        |          21.8s         |             1.9ms            |    0.4s    |     28.908ms     |  0.0s  |    5.401ms   |   0.1s  |    24.708ms   |     0.1s    |      16.786ms     |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
As expected, the bigger the CPC, the bigger the prediction time, especially when the number of classes is big.
Training time is almost not impacted.

#### Model size
Reminder: `dataset_prefixed` mode keeps the whole training dataset in memory.  In an age where 
multi-gigabytes NN model are welcome for classification, this can be considered as a viable solution. Most models are 
a few megabytes only. Also, this corresponds to the *gzip method* setup.

```
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|               Method               |   AG_NEWS  |    IMDB    |AmazonReviewPolarity|   DBpedia   | YahooAnswers|YelpReviewPolarity|   20News   |     R8    |    R52    |  kinnews  |
+------------------------------------+------------+------------+--------------------+-------------+-------------+------------------+------------+-----------+-----------+-----------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_1|28.497299 Mb|33.157821 Mb|    1553.96443 Mb   |170.588956 Mb|730.885685 Mb|   407.052479 Mb  |22.065911 Mb|3.320973 Mb|4.239561 Mb|6.578101 Mb|
+------------------------------------+------------+------------+--------------------+-------------+-------------+------------------+------------+-----------+-----------+-----------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_3|28.497291 Mb|33.157817 Mb|   1553.964426 Mb   |170.588928 Mb|730.885665 Mb|   407.052475 Mb  |22.065871 Mb|3.320957 Mb|4.239464 Mb|6.578077 Mb|
+------------------------------------+------------+------------+--------------------+-------------+-------------+------------------+------------+-----------+-----------+-----------+
|FFTC ZSTD_CL9 dataset_prefixed CPC_5|28.497283 Mb|33.157813 Mb|   1553.964422 Mb   | 170.5889 Mb |730.885645 Mb|   407.052471 Mb  |22.065831 Mb|3.320941 Mb|4.239384 Mb|6.578053 Mb|
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

### `size_unbounded_optimized` mode 
`size_unbounded_optimized` means an algorithm automatically optimizes the dictionary, without bound on the size of the dictionary. See [algorithm here](https://python-zstandard.readthedocs.io/en/latest/dictionaries.html#training-dictionaries).
Performance are a bit worse, while staying reasonable. The training time is bigger, but the model size is smaller than 
the whole training set size.

**WIP - benchmark coming soon.**

#### Accuracy
**WIP - benchmark coming soon.**

#### Speed
*Below is just to give an idea. Run on my 2021 intel MacBook Pro. Do your own microbenchmark.*
**WIP - benchmark coming soon.**

#### Model size

### Notes on performance
In the benchmark above we used a compression level of 9. Compression level can go up to 22. 
I have observed that compression level up to 18 will give significant performance improvements. It 
makes training and inference slower though. Try and benchmark yourself if need be.

## Reproduce
Requirements
```
python 3.7
```

Install
```
pip install -r requirements.txt
```

Reproduce: with whole training dataset maintained in memory (`dataset_prefixed` mode).  
Recommended to compare performance with the *gzip method*, or with setups that can afford to maintain the whole training dataset in memory (eg small dataset).
```
python main.py -s -1
```

Reproduce - with unbounded model size (`size_unbounded_optimized`).  
Recommended to compare performance with setups that can't afford to keep the whole training data in memory (big datasets, memory constraints or interested by the "learning" side of compression).  
```
python main.py -s 0
```

This will train and evaluate for 30 models, so this takes some time - around 40 minutes on commodity hardware.
The slow evaluation is caused by the AmazonReviewPolarity dataset that has 400000k.
See how to select the dataset, compressors, cpc and size constraint below. 

Run on specific datasets
```
python main.py -d AG_NEWS -d IMDB
```

Run with specific compressors
```
python main.py -d AG_NEWS -c ZSTD_CL12
```

Run with specific CPC 
```
python main.py -d AG_NEWS -cpc 1 -cpc 3
```

Run with specific memory constraints 
```
python main.py -s -1 -s 0 -s 10000 
```
`-1` means the full dataset is used as a prefix by the dictionary (`dataset_prefixed` mode)
`-0` means the dictionary size is unbounded but optimized automatically (`size_unbounded_optimized` mode)
`10000` means each generated dictionary size is bounded to 10000 bytes. 
See [Accuracy performance](#accuracy-performance) for more info.

You can combine all the parameters together. For instance:
```
python main.py -d AG_NEWS -d IMDB -cpc 1 -cpc 3 -c ZSTD_CL9 -c ZSTD_CL12 -s -1 -s 0
```

To get the full help and see possible values for each parameters, run: 
```
python main.py --help
```

## Extend and Contribute
- ADD: return results as csv, not a printed table lol 
- ADD: implement dataset download functions. Some seemed broken in https://github.com/bazingagin/npc_gzip
- pytorch is extremely slow and not necessary in this project, we should remove it
- once the above is done, we can make this a library
- add remove stopwords
- add more compressors! 
- add more datasets!
- improve the evaluation loop, it's slow! 
- improve logs! 
- predictions are cheap: ensemble!
- optimize the prediction time when CPC > 1
- improve decision making when CPC > 1
- the string concatenation is the slowest part in the training. I suspect it could greatly be improved.
- change the loop order to avoid re-loading datasets - make dataset the outerloop


## Troubleshooting
#### Google drive download does not work  
For google drive links, you may have to comment line 22 to line 31 in torchtext._download_hooks.py

#### 20News download does not work
If you have the following error on mac OS: 
```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1091)>
```
You may have to run something like  
```
ln -s /etc/ssl/* /Library/Frameworks/Python.framework/Versions/3.7/etc/openssl
```

## Citation

If you use FTCC in your publication, please cite it by using the following BibTeX entry.

```
@Misc{peft,
  title =        {FTCC: Fast Text Classification with Compressors dictionary},
  author =       {Cyril de Catheu},
  howpublished = {\url{https://github.com/cyrilou242/peft}},
  year =         {2023}
}
```
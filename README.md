# FTCC

**FTCC**: **F**ast **T**ext **C**lassification with **C**ompressors dictionary

This repository implements a compression based classification technique that is fast at training **AND** 
at inference. 
It is an answer to [Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip).
The paper mentions different techniques based on compression, but does not mention the technique implemented in 
this repository.  
I suspect the **FTCC** technique in this repository has already been tested in the industry, but I could not find an 
implementation online, so here you go. I think it is a reasonable baseline for compression based classification.
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
Then given a input text, I can compare all compressor. The topic of the compressor that has the 
best compression ratio is most likely the topic of the input text.
This collection of compressors is the text classification model.

Formal algorithm:
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
  With `ct_i`  all texts of label `i`, joined together with the separator `/n`.
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

### Properties
This technique has the following nice properties:
- runs on low resources machines easily. 
  **Training takes a few seconds, inferences less than a milliseconds - on commodity hardware.**
- you can control the accuracy/model size trade-off easily. 
  *Because most dictionary based compressor allow to control the size of the compression dictionary.* 
- you can perform partial - per-class - re-training. 
  *There is one compressor per class. If a class often has false negative, you can improve the compressor of this class independently of the other compressors.* 

## Accuracy performance

```
+--------------------------------------------------------------------------------------------------------------------------+
|             Method             |AG_NEWS| IMDB|AmazonReviewPolarity|DBpedia|YahooAnswers|YelpReviewPolarity|20News|kinnews|
+--------------------------------+-------+-----+--------------------+-------+------------+------------------+------+-------+
|FFTC DEFAULT_ZSTD top_1 accuracy| 0.812 |0.663|        0.688       | 0.907 |    0.488   |       0.742      | 0.744| 0.868 |
+--------------------------------+-------+-----+--------------------+-------+------------+------------------+------+-------+
|FFTC DEFAULT_ZSTD top_2 accuracy| 0.941 | 1.0 |         1.0        | 0.968 |    0.641   |        1.0       | 0.838| 0.937 |
+--------------------------------------------------------------------------------------------------------------------------+```
```

*Some dataset are not included because they have train/test split issues. See [here](https://github.com/bazingagin/npc_gzip/issues/13).*  
*Some other datasets are not included because the downloader function seemed broken. Help me fix this! See [contribute](#extend-and-contribute)*

**CAUTION**   
`top_k=2` corresponds to the accuracy formula used in [A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip).  
It should only be used to compare performance with this paper. I hope the author will share results for k=1.
See [issue](https://github.com/bazingagin/npc_gzip/issues/3).
`top_k=1` correspond to the standard accuracy formula.

**CAUTION**
Obviously `top_k=2` has a perfect accuracy for binary classification datasets `IMDB`, `AmazonReviewPolarity` and `YelpReviewPolarity`.

Performance are similar or better than the [A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip) for 
top 2 accuracy.

| method  | AGNews  | DBpedia  |  YahooAnswers | 20News  |
|---|---|---|---|---|
| gzip top2 accuracy  | 0.937  | **0.970**  | 0.638   | 0.685  |
| FFTC zstd top2  (ours)| **0.941**  | 0.968  |  **0.641**  | **0.838**  |

**CAUTION** 
Again, I'd prefer to compare top_k=1 accuracy, but the numbers are not provided in the paper.

## Speed
*Below is just to give an idea. Run on my 2021 intel MacBook Pro. Do your own microbenchmark.*
The computation is multiple orders of magnitudes faster [A Parameter-Free Classification Method with Compressors](https://github.com/bazingagin/npc_gzip).

```
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|             Method             |                             AG_NEWS                            |                              IMDB                              |                       AmazonReviewPolarity                      |                             DBpedia                             |                           YahooAnswers                          |                        YelpReviewPolarity                       |                             20News                             |                             kinnews                            |
+--------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
|FFTC DEFAULT_ZSTD top_1 accuracy|Training: 2.0s.Inference: p50: 0.022ms,p90: 0.034ms,p99: 0.057ms|Training: 0.9s.Inference: p50: 0.032ms,p90: 0.061ms,p99: 0.112ms| Training: 86.4s.Inference: p50: 0.018ms,p90: 0.034ms,p99: 0.06ms|Training: 11.2s.Inference: p50: 0.178ms,p90: 0.286ms,p99: 0.507ms|Training: 35.9s.Inference: p50: 0.117ms,p90: 0.304ms,p99: 0.779ms|Training: 15.7s.Inference: p50: 0.029ms,p90: 0.074ms,p99: 0.167ms|Training: 0.0s.Inference: p50: 0.728ms,p90: 1.691ms,p99: 6.805ms|Training: 0.0s.Inference: p50: 0.317ms,p90: 0.692ms,p99: 1.424ms|
+--------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
|FFTC DEFAULT_ZSTD top_2 accuracy| Training: 2.2s.Inference: p50: 0.032ms,p90: 0.046ms,p99: 0.08ms|Training: 0.9s.Inference: p50: 0.038ms,p90: 0.079ms,p99: 0.161ms|Training: 88.5s.Inference: p50: 0.018ms,p90: 0.033ms,p99: 0.065ms|Training: 11.1s.Inference: p50: 0.161ms,p90: 0.246ms,p99: 0.314ms|Training: 35.8s.Inference: p50: 0.115ms,p90: 0.296ms,p99: 0.727ms|Training: 15.5s.Inference: p50: 0.025ms,p90: 0.057ms,p99: 0.127ms|Training: 0.0s.Inference: p50: 0.746ms,p90: 1.774ms,p99: 7.031ms|Training: 0.0s.Inference: p50: 0.317ms,p90: 0.753ms,p99: 1.512ms|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+```


## Reproduce
Requirements
```
python 3.7
```

Install
```
pip install -r requirements.txt
```

Reproduce
```
python main.py --top_k_accuracy 1 --top_k_accuracy 2
```

Run on a few datasets
```
python main.py -d AG_NEWS -d IMDB
```

For more configuration knobs, run 
```
python main.py --help
```

## Extend and Contribute
- implement dataset download functions. Some seemed broken in https://github.com/bazingagin/npc_gzip
- pytorch is extremely slow and not necessary in this project, we should remove it
- once the above is done, we can make this a library
- add more compressors! 
- add more datasets!
- tune the zstd compressor parameters!
- the string concatenation is the slowest part. I suspect it could greatly be improved.


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
# BPE-Dropout
An official implementation of "BPE-Dropout" algorithm, introduced in [BPE-Dropout: Simple and Effective Subword Regularization](https://www.aclweb.org/anthology/2020.acl-main.170/)

### Usage example

With [Subword-nmt](https://github.com/rsennrich/subword-nmt) merge table:

```
merge_table_path = './example/subword_nmt.voc'

from bpe import load_subword_nmt_table, BpeOnlineTokenizer

merge_table = load_subword_nmt_table(merge_table_path)

subword_nmt_tokenizer = BpeOnlineTokenizer(
    bpe_dropout_rate=0.1, 
    merge_table=merge_table)

for i in range(10):
    print(subword_nmt_tokenizer("Some example sentence to show segmentation", 
                                sentinels=['', '</w>'],
                                regime='end',
                                bpe_symbol='@@'))
```

Example output:

```
S@@ ome ex@@ am@@ ple senten@@ ce t@@ o sh@@ ow seg@@ ment@@ ation
S@@ ome exam@@ ple senten@@ ce to sh@@ ow seg@@ ment@@ ation
S@@ ome ex@@ am@@ ple sen@@ te@@ nc@@ e to show seg@@ ment@@ ation
S@@ ome exam@@ ple s@@ enten@@ ce to show seg@@ ment@@ ation
S@@ ome exam@@ ple senten@@ ce to sh@@ ow seg@@ ment@@ ation
S@@ ome exam@@ ple senten@@ ce t@@ o sho@@ w s@@ eg@@ ment@@ ation
S@@ ome exam@@ ple senten@@ ce to s@@ how seg@@ m@@ ent@@ ation
S@@ ome exam@@ ple senten@@ ce to show seg@@ ment@@ ation
S@@ om@@ e exam@@ ple senten@@ ce to show se@@ g@@ ment@@ ation
S@@ om@@ e exam@@ ple senten@@ ce to s@@ how seg@@ ment@@ ation
```

With our merge table:

```
merge_table_path = './example/bpe.voc'

from bpe import load_merge_table, BpeOnlineTokenizer

merge_table = load_merge_table(merge_table_path)

tokenizer = BpeOnlineTokenizer(bpe_dropout_rate=0.1, merge_table=merge_table)

for i in range(10):
    print(tokenizer("some example sentence to show segmentation"))
```

Example output:
```
som `e exam `ple s `ent `ence to sh `ow s `eg `mentation
so `me example sentence to show seg `men `tation
some example sentence to show se `g `mentation
some example sentence to show seg `mentation
some example sentence to s `h `ow s `eg `mentation
some ex `am `ple s `ent `ence to show seg `mentation
s `ome example sentence to show s `eg `men `tation
some example sent `ence to show se `g `mentation
some example sen `ten `ce to show seg `mentation
some example sentence to show s `eg `mentation
```

Unfortunately, we do not provide code of building our merge table as it is internal.

#### Our algorithm should be applied on each new batch, or new epoch, to obtain multiple segmentations of the same sentence.

### Additional functions

* BpeOnlineParallelApplier -- performs segmentation for parallel sentences

### Other implementations
The following repositories have implemented BPE-dropout
* [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) -- fastest bpe implementation
* [Sentencepiece](https://github.com/google/sentencepiece) -- original subword regularization repository
* [Subword-nmt](https://github.com/rsennrich/subword-nmt) -- original bpe repository

### Speed

In order to achive high speed of segmentation you can either use faster implementation like [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe), or use Python multiprocessing.

### Reference

```
@inproceedings{provilkov-etal-2020-bpe,
    title = "{BPE}-Dropout: Simple and Effective Subword Regularization",
    author = "Provilkov, Ivan  and
      Emelianenko, Dmitrii  and
      Voita, Elena",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.170",
    pages = "1882--1892",
    abstract = "Subword segmentation is widely used to address the open vocabulary problem in machine translation. The dominant approach to subword segmentation is Byte Pair Encoding (BPE), which keeps the most frequent words intact while splitting the rare ones into multiple tokens. While multiple segmentations are possible even with the same vocabulary, BPE splits words into unique sequences; this may prevent a model from better learning the compositionality of words and being robust to segmentation errors. So far, the only way to overcome this BPE imperfection, its deterministic nature, was to create another subword segmentation algorithm (Kudo, 2018). In contrast, we show that BPE itself incorporates the ability to produce multiple segmentations of the same word. We introduce BPE-dropout - simple and effective subword regularization method based on and compatible with conventional BPE. It stochastically corrupts the segmentation procedure of BPE, which leads to producing multiple segmentations within the same fixed BPE framework. Using BPE-dropout during training and the standard BPE during inference improves translation quality up to 2.3 BLEU compared to BPE and up to 0.9 BLEU compared to the previous subword regularization.",
}
```

# BPE-Dropout
An official implementation of "BPE-Dropout" algorithm, introduced in [BPE-Dropout: Simple and Effective Subword Regularization](https://www.aclweb.org/anthology/2020.acl-main.170/)

### Introduction

This repository contains a reference implementation of BPE-Dropout algorithm, which was used in the original paper. This repository does not contain the code for building a BPE merge table, relying on the external BPE merge table for the sake of simplicity. Note that BPE-dropout algorithm is already implemented in several most used packages that perform subword segmentation (see [Other Implementations](#other-implementations)). We encourage using these implementations as they provide merge table building and other useful features. 

#### While using these, keep in mind, that our algorithm should be applied on each new batch, or new epoch, to obtain multiple segmentations of the same sentence.

### Usage example

BPE (and BPE-dropout inherits this) needs a merge table to operate. 
For this example we will use a merge table, produced by 
 [Subword-nmt](https://github.com/rsennrich/subword-nmt):

```
merge_table_path = './example/subword_nmt.voc'

from bpe import load_subword_nmt_table, BpeOnlineTokenizer

merge_table = load_subword_nmt_table(merge_table_path)

subword_nmt_tokenizer = BpeOnlineTokenizer(
    bpe_dropout_rate=0.1, 
    merge_table=merge_table)

for i in range(10):
    print(subword_nmt_tokenizer("some example sentence to show segmentation", 
                                sentinels=['', '</w>'],
                                regime='end',
                                bpe_symbol='@@'))
```

Example output:

```
some ex@@ ample sentence to show segmentation
some exam@@ ple sentence to show segmentation
so@@ me example sentence to show segmentation
some ex@@ ample sentence to show segmentation
some ex@@ ample sen@@ te@@ nce to show segmentation
some example sentence to show segmentation
some ex@@ ample sentence to show segmentation
some example sen@@ tence to show seg@@ men@@ ta@@ tion
some example sentence to s@@ how segmentation
some example sen@@ tence t@@ o show segmentation
```


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

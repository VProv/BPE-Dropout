# BPE-Dropout
An official implementation of "BPE-Dropout" algorithm, introduced in [BPE-Dropout: Simple and Effective Subword Regularization](https://www.aclweb.org/anthology/2020.acl-main.170/)

### Usage example
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

### Additional functions

* load_subword_nmt_table -- allows to work with subword-nmt merge table
* BpeOnlineParallelApplier -- performs segmentation for parallel sentences

### Other implementations
The following repositories have implemented BPE-dropout
* [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) -- fastest bpe implementation
* [Sentencepiece](https://github.com/google/sentencepiece) -- original subword regularization repository
* [Subword-nmt](https://github.com/rsennrich/subword-nmt) -- original bpe repository


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

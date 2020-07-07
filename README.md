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

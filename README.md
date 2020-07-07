# BPE-Dropout
An official implementation of "BPE-Dropout: Simple and Effective Subword Regularization" algorithm.

### Usage example:
```
merge_table_path = './example/bpe.voc'

from bpe import load_merge_table, BpeOnlineTokenizer

merge_table = load_merge_table(merge_table_path)

tokenizer = BpeOnlineTokenizer(bpe_dropout_rate=0.1, merge_table=merge_table)

for i in range(10):
    print(tokenizer("Some example sentence to show segmentation"))
```

Example output:
```
S `ome example sentence t `o show seg `mentation
S `ome example sentence to s `h `ow se `g `mentation
S `ome example sen `t `ence to show seg `mentation
S `o `me example sent `ence to sh `o `w seg `mentation
S `o `me example sentence t `o show seg `mentation
S `ome exam `ple sentence to show seg `men `tation
S `ome ex `a `m `ple sentence to show se `g `mentation
S `ome example sentence to show se `g `mentation
S `ome example sentence to show seg `mentation
S `ome example sentence to show seg `men `tation
```

### Additional functions

* load_subword_nmt_table -- allows to work with subword-nmt merge table
* BpeOnlineParallelApplier -- performs segmentation for parallel sentences

### Other implementations:

* [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) -- fastest bpe implementation
* [Sentencepiece](https://github.com/google/sentencepiece) -- original subword regularization repository
* [Subword-nmt](https://github.com/rsennrich/subword-nmt) -- original bpe repository

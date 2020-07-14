# make sure bpe from the root of the repository is in PYTHONPATH environment variable
# or in the same directory to be able to import it
from bpe import load_subword_nmt_table, BpeOnlineTokenizer

merge_table_path = './subword_nmt.voc'
merge_table = load_subword_nmt_table(merge_table_path)

subword_nmt_tokenizer = BpeOnlineTokenizer(
    bpe_dropout_rate=0.1, 
    merge_table=merge_table)

for i in range(10):
    print(subword_nmt_tokenizer("some example sentence to show segmentation", 
                                sentinels=['', '</w>'],
                                regime='end',
                                bpe_symbol='@@'))

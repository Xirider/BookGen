# goal is to summarize the fragments and summarize these again

import torch
BART_HUB = 'pytorch/fairseq'
BART_CNN_MODEL = 'bart.large.cnn'
SUM_LEVELS = ["sum1", "sum2", "sum3", "sum4", "sum5", "sum6", "sum7", "sum8", "sum9", "sum10", "sum_chapters"]


def summarize_with_bart(text_list, batch_size=4):
    bart = torch.hub.load(BART_HUB,BART_CNN_MODEL)

    # add leading space for gpt tokenizer for bart
    text_list = ["".join([" ", text]) for text in text_list]
    summarized = []
    for i in range(0,len(text_list), batch_size):

        batch = text_list[i: i+ batch_size]
        result_batch = bart.sample(batch, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        summarized.append(result_batch)
    return summarized

def summarize_books(books, max_sum_level, sum_ratio, sum_high_level_chapters, sum_model):

    for bookid in range(len(books)):
        for chapterid in range(int(books[bookid]["Chapters"])):

            # chapter_fragments = books[bookid][chapterid]["fragments"]
            fragment_texts = [fragment["text"] for fragment in books[bookid][chapterid]["fragments"]]

            for level in range(max_sum_level):
                






    pass


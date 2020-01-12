# goal is to summarize the fragments and summarize these again
from joblib import Memory
import torch
BART_HUB = 'pytorch/fairseq'
BART_CNN_MODEL = 'bart.large.cnn'
SUM_LEVELS = ["sum1", "sum2", "sum3", "sum4", "sum5", "sum6", "sum7", "sum8", "sum9", "sum10", "sum_chapters"]

cachedir = "cache"
memory = Memory(cachedir, verbose=10)

class BartModel:
    def __init__(self, gpu=True):
        self.bart = torch.hub.load(BART_HUB,BART_CNN_MODEL)
        print("model loaded")
        if gpu:
            self.bart.cuda()
        self.bart.eval()
    def summarize_text(self, text_list, batch_size=8):
        text_list = ["".join([" ", text]) for text in text_list]
        text_list_len = len(text_list)
        summarized = []
        for i in range(0,text_list_len, batch_size):

            batch = text_list[i: i+ batch_size]
            result_batch = self.bart.sample(batch, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            summarized.extend(result_batch)
            # print(f"bart batch finished {i} of {text_list_len}")
        return summarized


# @memory.cache
def summarize_books(books, max_chapter_level,  sum_ratio, sum_model, sum_ratio_high = 10000, treat_text_equally=False):

    if sum_model == "bart":
        summarizer = BartModel()

    booknum = len(books)
    for bookid in range(len(books)):
        if bookid % 100 == 0:
            print(f"summarizing book {bookid} now of a total of {booknum} books")
        high_level_chapters = []


        for chapterid in range(int(books[bookid]["Chapters"])):

            # chapter_fragments = books[bookid][chapterid]["fragments"]
            fragment_texts = [fragment["text"] for fragment in books[bookid][chapterid]["fragments"]]

            for level in range(max_chapter_level):
                
                if level == 0:
                    cur_sum =  summarizer.summarize_text(fragment_texts)
                    
                
                else:
                    joined_summaries, unjoined_summaries ,relations = join_summaries(cur_sum, sum_ratio)
                    cur_sum = summarizer.summarize_text(joined_summaries)
                
                cur_sum_level = f"sum{level+1}"
                relations_name = f"relations{level+1}"
                before_join_name = f"unjoined_before_{level+1}"
                books[bookid][chapterid][cur_sum_level] = cur_sum

                if level != 0:
                    books[bookid][chapterid][before_join_name] = unjoined_summaries
                    books[bookid][chapterid][relations_name] = relations

                if len(cur_sum) == 1:
                    break
            
            high_level_chapters.extend(cur_sum)
        if len(high_level_chapters) > 1:

            # print("book high level chapters:")
            # print(len(high_level_chapters))            
            high_level_text, _,  _  = join_summaries(high_level_chapters, sum_ratio_high)
            books[bookid]["before_sum_chapters"] = high_level_chapters
            books[bookid]["sum_chapters"] = summarizer.summarize_text(high_level_text)
            


    return books


def join_summaries(texts, sum_ratio, join_character= " "):
    joined = []
    unjoined = []
    relations = []
    for textid, text in enumerate(texts):
        step = textid % sum_ratio

        if step == 0:
            joined_text = ""
            unjoined_text = []
            relation = []

        joined_text = join_character.join([joined_text, text])
        relation.append(textid)
        unjoined_text.append(text)

        if (step == sum_ratio -1) or (textid == len(texts) -1):
            joined.append(joined_text)
            unjoined.append(unjoined_text)
            relations.append(relation)
    
    return joined,unjoined, relations


if __name__ == "__main__":
    tt = ["1", "2", "3","4", "5", "6", "7 ", "8", "9 ", "10", "11"]
    abc = join_summaries(tt, 3)
    print(abc)
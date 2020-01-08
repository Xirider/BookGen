from pipeline.getstories import get_ff_files
from pipeline.filterstories import filter_ff_stories
from pipeline.split import split
from pipeline.summarize import summarize_books
from pipeline.lmprep import prepare_for_lm
from pipeline.utils import get_tokenizer, save_datasets


def run():
    books = get_ff_files("allbooks")
    books = filter_ff_stories(books, max_rating="M", min_words= 400, max_words= 10000, max_chapters= 3, min_chapters= 1, max_books=10)

    tokenizer = get_tokenizer("gpt2-large")
    books = split(books, tokenizer, max_tokens = 200, max_prev_tokens = 100)
    books = summarize_books(books, max_chapter_level=10, sum_ratio=3, sum_model="bart")
    datasets = prepare_for_lm(books, tokenizer, max_seq_len=400)
    save_datasets(datasets)

    
    return datasets




if __name__ == "__main__":
    datasets = run()
    from pprint import pprint as pp
    import pdb; pdb.set_trace()
    #print(books)

from pipeline.getstories import get_text_paths, parse_fanfiction
from pipeline.filterstories import filter_ff_stories
from pipeline.split import split
from pipeline.summarize import summarize_books
from pipeline.lmprep import prepare_for_lm
from pipeline.utils import get_tokenizer, save_datasets, split_examples, get_shard


def run():

    paths = get_text_paths("allbooks", shuffle = True)
    
    shard_count = 150

    for i in range(shard_count):
        
        print(f"starting with shard {i}")
        path_shard = get_shard(paths, i, shard_count)
        books = parse_fanfiction(path_shard)
        books = filter_ff_stories(books, max_rating="M", min_words= 2000, max_words= 200000, max_chapters= 30, min_chapters= 0, max_books=10000000)

        tokenizer = get_tokenizer("gpt2-medium")
        books = split(books, tokenizer, max_tokens = 150, max_prev_tokens = 150)

        books = summarize_books(books, max_chapter_level=10, sum_ratio=3, sum_model="bart")

        datasets = prepare_for_lm(books, tokenizer, max_seq_len=400)

        # train_datasets, eval_datasets = split_examples(datasets, ratio=0.05)

        if i == shard_count - 1:
            dataset_name = "eval"
        else:
            dataset_name = f"train_{i}"

        save_datasets(datasets, dataset_name)


    




if __name__ == "__main__":
    run()
    # from pprint import pprint as pp
    # import pdb; pdb.set_trace()
    #print(books)

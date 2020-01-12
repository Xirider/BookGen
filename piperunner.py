from pipeline.getstories import get_text_paths, parse_fanfiction
from pipeline.filterstories import filter_ff_stories
from pipeline.split import split
from pipeline.summarize import summarize_books
from pipeline.lmprep import prepare_for_lm
from pipeline.utils import get_tokenizer, save_datasets, split_examples, get_shard, save_book_shard
import argparse

def run():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--shard_start", default=0, type=int, required=False
    )
    # Required parameters
    parser.add_argument(
        "--shard_end", default=-1, type=int, required=False
    )

    parser.add_argument(
        "--shard_count", default=2000, type=int, required=False
    )
    args = parser.parse_args()

    if args.shard_end == -1:
        args.shard_end = args.shard_count 

    paths = get_text_paths("allbooks", shuffle = True)
    shard_count = args.shard_count




    for i in range(shard_count):

        if args.shard_start > i or args.shard_end <= i:
            continue
        
        print(f"starting with shard {i}")
        path_shard = get_shard(paths, i, shard_count)
        books = parse_fanfiction(path_shard)
        books = filter_ff_stories(books, max_rating="M", min_words= 2000, max_words= 50000, max_chapters= 20, min_chapters= 0, max_books=10000000)

        tokenizer = get_tokenizer("gpt2-medium")
        books = split(books, tokenizer, max_tokens = 150, max_prev_tokens = 150)

        books = summarize_books(books, max_chapter_level=10, sum_ratio=3, sum_model="bart")

        save_book_shard(books, i)


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

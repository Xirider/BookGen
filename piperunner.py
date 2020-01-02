from pipeline.getstories import get_ff_files
from pipeline.filterstories import filter_ff_stories
from pipeline.split import split
from pipeline.summarize import summarize_books


def run():
    books = get_ff_files("allbooks")
    books = filter_ff_stories(books, max_rating="M", min_words= 400, max_words= 10000, max_chapters= 1, min_chapters= 1, max_books=10)
    books = split(books, max_tokens = 200, max_prev_tokens = 100)
    # books = summarize_books(books)
    
    return books




if __name__ == "__main__":
    books = run()

    import pdb; pdb.set_trace()
    #print(books)

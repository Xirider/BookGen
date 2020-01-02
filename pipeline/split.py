
# create fragments with the right size for gpt2
from pipeline.utils import get_tokenizer
from joblib import Memory

cachedir = "cache"
memory = Memory(cachedir, verbose=10)

@memory.cache
def split(books, max_tokens, max_prev_tokens, tokenizer_name = "gpt2-large"):

    tokenizer = get_tokenizer(tokenizer_name)
    for bookid, book in enumerate(books):

        chapter_count = int(books[bookid]["Chapters"].replace(",", ""))

        for chapterid in range(chapter_count):
            chapter_text = book[chapterid]["content"]

            chapter_text = "\n".join(chapter_text)
            chapter_tokens  = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chapter_text, add_prefix_space= True))

            fragments = []
            fragment_count = 0
            for i in range(0, len(chapter_tokens) - max_tokens + 1, max_tokens):
                tokens = chapter_tokens[i : i + max_tokens]
                if fragment_count == 0:
                    prev_tokens = [tokenizer.pad_token_id] * max_prev_tokens
                else:
                    prev_tokens = chapter_tokens[i - max_prev_tokens:i]
                text = tokenizer.decode(tokens)

                while len(tokens) < max_tokens:
                    tokens.append(tokenizer.pad_token_id)
                fragments.append({"tokens": tokens, "prev_tokens": prev_tokens, "text":text,
                 "fragment_id": fragment_count, "token_start": i})
                fragment_count += 1
            
            books[bookid][chapterid]["fragments"] = fragments
            
            
# add_prefix_space = True, 



    return books
























if __name__ == "__main__":
    print("hello")
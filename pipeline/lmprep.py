from joblib import Memory

IGNORE_TOKEN = -100
cachedir = "cache"
memory = Memory(cachedir, verbose=10)

@memory.cache
def prepare_for_lm(books, tokenizer, max_seq_len):

    # each dataset need a list of tokens, a list of segment ids, a list of labels with ignored -100 indixes
    # input, token_type_ids, labels (with ignoring)
    ignore_start = "ignore_start"
    ignore_end = "ignore_end"

    # text_0
    out_dict = {}
    text_0 = []
    print("prepping level 0 now")
    for book in books:
        for chapterid in range(int(book["Chapters"])):
            chapter = book[chapterid]
            for fragmentid, fragment in enumerate(chapter["fragments"]):
                cur_schema = [ignore_start,"<sum1>", chapter["sum1"][fragmentid], "<prevtext>", fragment["prev_tokens"], "<genstart>", ignore_end,fragment["tokens"] ]
                text_0.append(create_single_example(cur_schema, tokenizer, max_seq_len))
    out_dict["text_0"] = text_0


    # sum_1 to sum_10
    for level in range(1, 11):
        print(f"prepping level {level} now")
        gen_level = f"unjoined_before_{level+1}"
        gen_token = f"<sum{level}>"
        context_level = f"sum{level+1}"
        context_token = f"<sum{level+1}>"
        sum_cur_level = []
        for book in books:
            for chapterid in range(int(book["Chapters"])):
                chapter = book[chapterid]
                if context_level in chapter:
                    for contextid , context in enumerate(chapter[context_level]):
                        
                        unjoined = chapter[gen_level][contextid]
                        multiple_gen = []
                        for partid, part in enumerate(unjoined):
                            multiple_gen.append(gen_token)
                            if partid == 0:
                                multiple_gen.append(ignore_end)
                            multiple_gen.append(part)

                        cur_schema = [ignore_start, context_token, context]
                        cur_schema.extend(multiple_gen)
                        cur_schema.append("<sum_end>")
                        sum_cur_level.append(create_single_example(cur_schema, tokenizer, max_seq_len))

        out_dict[f"sum_{level}"] = sum_cur_level
        


    # chapters_11
    chapters_11 = []
    print("prepping high level chapters now")
    for book in books:
        if "sum_chapters" in book:
            before_sum = book["before_sum_chapters"]
            sum_chapters = book["sum_chapters"][0]

            cur_schema = [ignore_start, "<sum_chapters>", sum_chapters]
            for partid, part in enumerate(before_sum):
                cur_schema.append("<chapter>")
                if partid == 0:
                    cur_schema.append(ignore_end)
                cur_schema.append(part)

            cur_schema.append("<sum_end>")
            chapters_11.append(create_single_example(cur_schema, tokenizer, max_seq_len))
    
    out_dict["chapters_11"] = chapters_11


    return out_dict




# if __name__ == "__main__":
#     from utils import get_tokenizer

#     tokenizer = get_tokenizer()
#     datasets= prepare_for_lm(books, tokenizer)

def create_single_example(schema, tokenizer, max_seq_len):
    input_ids = []
    token_type_ids = []
    labels = []
    ignore = False
    active_segment = []

    special_tokens = tokenizer.additional_special_tokens
    for group in schema:
        # remove and use ignore tokens
        if group == "ignore_start":
            ignore = True
            continue
        if group == "ignore_end":
            ignore = False
            continue
        
        # add special tokens
        if group in special_tokens:
            token = tokenizer.convert_tokens_to_ids(group)
            active_segment = [token]

            input_ids.append(token)
            token_type_ids.append(token)
            if ignore == False:
                labels.append(token)
            else:
                labels.append(IGNORE_TOKEN)

        # add text
        else:
            
            assert len(active_segment) == 1
            if isinstance(group, str):
                group = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(group, add_prefix_space= True))
            assert isinstance(group, list)

            input_ids.extend(group)
            token_type_ids.extend(active_segment * len(group))
            if ignore == False:
                labels.extend(group)
            else:
                labels.extend([IGNORE_TOKEN]* len(group))
            
            
    assert len(input_ids) == len(token_type_ids) == len(labels) 

    # pad and slice to max_seq_len
    while len(input_ids) < max_seq_len:
                    input_ids.append(tokenizer.pad_token_id)
                    labels.append(IGNORE_TOKEN)
                    token_type_ids.append(tokenizer.pad_token_id)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        token_type_ids = token_type_ids[:max_seq_len]
    

    return [input_ids, token_type_ids, labels]
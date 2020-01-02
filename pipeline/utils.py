import os
import pickle
from transformers import GPT2Tokenizer

def get_tokenizer(tokenizer_name = "gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    special_tokens_dict = {"pad_token": "<pad>", "additional_special_tokens":["<genstart>", "<prevtext>"
     "<sum1>", "<sum2>", "<sum3>", "<sum4>", "<sum5>", "<sum6>", "<sum7>", "<sum8>", "<sum9>", "<sum10>", "<sum_chapters>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer



# def cache_object(obj, cache_name, no_cache_flag=False, reset_cache=False):
#     if not no_cache_flag:
#         if not os.path.exists("cache"):
#             os.mkdir("cache")
#         pathname = "cache/" + cache_name + ".p"
#         if not os.path.exists(pathname) or reset_cache:
#             pickle.dump(obj, open(pathname), protocol=pickle.HIGHEST_PROTOCOL)
        

        
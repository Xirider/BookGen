import os
import pickle
import json
from transformers import GPT2Tokenizer
import random
from pathlib import Path

def get_tokenizer(tokenizer_name = "gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    special_tokens_dict = {"pad_token": "<pad>", "additional_special_tokens":["<genstart>", "<prevtext>",
    "<sum1>", "<sum2>", "<sum3>", "<sum4>", "<sum5>", "<sum6>", "<sum7>", "<sum8>",
    "<sum9>", "<sum10>", "<sum_chapters>", "<chapter>", "<sum_end>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def save_datasets(datasets, name="datasets"):
    start = Path("pipeline/data/")
    name = name + ".json"
    name = start / name
    with open(name, "w") as fp:
        json.dump(datasets, fp)



def split_examples(datasets, ratio):
    ratio = 1 - ratio
    train_datasets = {}
    eval_datasets = {}
    for key in datasets:
        train_datasets[key] = []
        eval_datasets[key] = []

    for key in datasets:
        random.shuffle(datasets[key])
        item_number = len(datasets[key])
        
        cutoff = int(item_number * ratio)
        if item_number > 0:
            if cutoff == 0:
                cutoff += 1
            
            train_datasets[key] = datasets[key][:cutoff]

            eval_datasets[key] = datasets[key][cutoff:]

    datasets = []
    return train_datasets, eval_datasets
            

def get_shard(full_list, shard_index, total_shards):

    total_len = len(full_list)

    shard_size = total_len // total_shards

    return full_list[shard_index * shard_size:(shard_index + 1)*shard_size]

# def cache_object(obj, cache_name, no_cache_flag=False, reset_cache=False):
#     if not no_cache_flag:
#         if not os.path.exists("cache"):
#             os.mkdir("cache")
#         pathname = "cache/" + cache_name + ".p"
#         if not os.path.exists(pathname) or reset_cache:
#             pickle.dump(obj, open(pathname), protocol=pickle.HIGHEST_PROTOCOL)
        


def save_book_shard(books, shard_index):
    start = Path("pipeline/data/bookcache")
    if not os.path.exists(start):
        os.makedirs(start)
    name = f"bookcache_{shard_index}.json"
    name = start / name
    with open(name, "w") as fp:
        json.dump(books, fp)
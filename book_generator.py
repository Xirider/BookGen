
import numpy as np
import torch




def generate_text(model, tokenizer,  inputs, labels,  token_type_ids,  max_length, temperature, top_k, top_p, repetition_penalty, start_token, stop_token):

    input_ids, type_ids = create_prompt(inputs, labels, token_type_ids )

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        token_type_ids = type_ids,
    )

    text_batch = ids2text(tokenizer, output_sequences, start_token, stop_token)
    return text_batch

def generation_wrap(model,input_ids, token_type_ids, max_seq_len):
    output_sequences = model.generate(
    input_ids=input_ids,
    max_length=max_seq_len,
    temperature=0.7,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.2,
    token_type_ids = token_type_ids,
    )
    return output_sequences

def extract_generation(tokenizer, ids, start_token, repeat_token, stop_token, remove_after= "<|endoftext|>"):
    
    list_ids = ids[0].tolist()
    special = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
    start_t = tokenizer.convert_tokens_to_ids([start_token])[0]
    remove_t = tokenizer.convert_tokens_to_ids([remove_after])[0]
    if repeat_token:
        repeat_t = tokenizer.convert_tokens_to_ids([repeat_token])[0]
    if stop_token:
        stop_t = tokenizer.convert_tokens_to_ids([stop_token])[0]
    out = []
    cur= []
    active = False
    for tok in list_ids:
        if stop_token:
            if tok == stop_t:
                active = False
        if repeat_token:
            if tok == repeat_token:
                if len(cur) > 0:
                    out.append(cur)
                    cur = []
        if tok in special and tok != start_t:
            if repeat_token:
                if tok != repeat_t:
                    active = False
            else:
                active = False
        if tok == remove_t:
            break
        if active and tok not in special:
            cur.append(tok)

        
        if tok == start_t:
            active = True

    if len(cur) > 0:
        out.append(cur)
    for sq in range(len(out)):
        out[sq] = tokenizer.decode(out[sq], clean_up_tokenization_spaces=True)
    return out





def ids2text(tokenizer, input_ids, start_token, stop_token, is_label=False, ignore_id = -100):
    text_batch = []
    for dim in range(input_ids.size(0)):

        generated_sequence = input_ids[dim].tolist()
        if is_label:
            generated_sequence = [value for value in generated_sequence if value != ignore_id]
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        if stop_token:
            pos = text.find(stop_token)
            if pos != -1:
                text = text[: pos]
        if start_token:
            pos = text.find(start_token)
            if pos != -1:
                text = text[pos+len(start_token) + 1:]
        text_batch.append(text)
    return text_batch

def create_prompt(input_ids, labels, token_type_ids,  ignore_id=-100):
    device = input_ids.get_device()
    prompt_batch = []
    prompt_t_batch = []
    for dim in range(input_ids.size(0)):
        input_l = input_ids[dim].tolist()
        label_l = labels[dim].tolist()
        type_l = token_type_ids[dim].tolist()
        batch = []
        t_batch = []
        for i, l, t in zip(input_l, label_l, type_l):
            if l != ignore_id:
                break
            batch.append(i)
            t_batch.append(t)
        prompt_batch.append(batch)
        prompt_t_batch.append(t_batch)


    return torch.tensor(prompt_batch, device=device), torch.tensor(prompt_t_batch, device=device) 

def compare_text(generated, reference, bleu, rouge):
    rouge_l_score = rouge.rouge_l( summary=generated, references=reference)
    bleu_score = bleu.bleu(summary= generated, references= reference)
    return bleu_score, rouge_l_score
    


def generate_book(model, tokenizer, high_level_prompt, max_chapters, start_level, device, mid_start_level, max_input_len = 300, max_seq_len=400, sum_factor=3, prev_tokens_len=150):
    # for each level we inputids and token type ids. after generation we need to cut out the generation and put it in a dict.
    book = {"chapter_0": []}


    if start_level == 11:
        
        print("creating chapter level of book")
        # generate chapter summaries from book summary
        schema = ["<sum_chapters>", high_level_prompt, "<chapter>"]
        in_11 = create_gen_schema(schema, tokenizer, max_input_len, device)
        uncleaned_gen = generation_wrap(model, in_11[0], in_11[1], max_seq_len=max_seq_len)
        print(f"lvl 11 - ")
        print(tokenizer.decode(uncleaned_gen[0]))
        chapter_sums = extract_generation(tokenizer, uncleaned_gen, start_token="<chapter>", 
        repeat_token="<chapter>", stop_token="<sum_end>")

        # chapter_sums = chapter_sums[:max_chapters]
        for cid, chapter in enumerate(chapter_sums):
            book[f"chapter_{cid}"] = {f"level_{mid_start_level}":[chapter]}
    else:
        book["chapter_0"][f"level_{start_level}"] = [high_level_prompt]
    


    c_count = len(chapter_sums) if start_level == 11 else 1

    print("creating mid level of book")
    for chapterid in range(c_count):
        for level in range(11, 1, - 1):
            if f"level_{level}" not in book[f"chapter_{chapterid}"]:
                continue
            # generate x to 1 summaries
            if level > start_level:
                continue
            cur_level_prompts = book[f"chapter_{chapterid}"][f"level_{level}"]
            book[f"chapter_{chapterid}"][f"level_{level -1}"] = []
            for pid, prompt in enumerate(cur_level_prompts):
                if level == 11:
                    start  = "<sum_chapters>"
                else:
                    start = f"<sum{level}>"
                schema = [start, prompt, f"<sum{level-1}>"]
                created_schema = create_gen_schema(schema, tokenizer, max_input_len, device)
                uncleaned_gen = generation_wrap(model, created_schema[0], created_schema[1], max_seq_len=max_seq_len)
                print(f"lvl {level} - ")
                print(tokenizer.decode(uncleaned_gen[0]))
                lower_level_prompts = extract_generation(tokenizer, uncleaned_gen,
                 start_token=f"<sum{level-1}>", repeat_token=f"<sum{level-1}>", stop_token="<sum_end>")
                lower_level_prompts = lower_level_prompts[:sum_factor]
                book[f"chapter_{chapterid}"][f"level_{level -1}"].extend(lower_level_prompts)
    
    # generate final text
    print("creating low level text of book")
    for chapterid in range(c_count):
        cur_level_prompts = book[f"chapter_{chapterid}"][f"level_1"]
        prev_tokens = "<pad>" * prev_tokens_len
        book[f"chapter_{chapterid}"]["text_fragments"] = []
        for pid, prompt in enumerate(cur_level_prompts):
            schema = ["<sum1>" , prompt, "<prevtext>" ,prev_tokens, "<genstart>"]
            created_schema = create_gen_schema(schema, tokenizer, max_input_len, device)
            uncleaned_gen = generation_wrap(model, created_schema[0], created_schema[1], max_seq_len=max_seq_len)
            print(f"lvl text - ")
            print(tokenizer.decode(uncleaned_gen[0]))
            gen_text = extract_generation(tokenizer, uncleaned_gen,
            start_token=f"<genstart>", repeat_token=False, stop_token=False)
            if len(gen_text) == 0:
                continue
            
            gen_text = gen_text[0]
            book[f"chapter_{chapterid}"]["text_fragments"].append(gen_text)
            prev_tokens = prompt
    
    readable_book = []
    print("join text of book")
    for chapterid in range(c_count):
        readable_book.append(" ".join(book[f"chapter_{chapterid}"]["text_fragments"]))
    return readable_book, book


    

def create_gen_schema(schema, tokenizer, max_input_len, device):
    input_ids = []
    token_type_ids = []

    special_tokens = tokenizer.additional_special_tokens
    for group in schema:
        
        # add special tokens
        if group in special_tokens:
            token = tokenizer.convert_tokens_to_ids(group)
            active_segment = [token]

            input_ids.append(token)
            token_type_ids.append(token)


        # add text
        else:

            assert len(active_segment) == 1

            if isinstance(group, str):
                group = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(group, add_prefix_space= True))
            assert isinstance(group, list)

            input_ids.extend(group)
            token_type_ids.extend(active_segment * len(group))

            
            
    assert len(input_ids) == len(token_type_ids)

    max_seq_len = max_input_len
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        token_type_ids = token_type_ids[:max_seq_len]
    

    return [torch.tensor(input_ids,device=device).unsqueeze(0), torch.tensor(token_type_ids, device=device).unsqueeze(0)]
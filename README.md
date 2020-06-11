# BookGen

BookGen is a research projects that aims to generate whole books with a language model by iteratively expanding summaries.

## How it works

This is how the general approach works:
1. books are downloaded and extracted from the book-corpus and fanfiction.net
2. books are preprocessed and split into chapters
3. smaller parts of each chapter are summarized by a BART model that was trained on summarization
4. the summaries of the smaller text parts are joined together and summarized again by BART
5. summarization is repeated until only one summary is left for the book (for a maximum of 10 levels of summaries)
6. a large GPT-2 language model is then finetuned to create text based on the summaries. For this the text is joined with summaries in this format: "SUMMARY1 DELIMITER TEXT1 DELIMITER" and "SUMMARYLEVEL2 DELIMITER SUMMARYLEVEL1_0 DELIMTER SUMMARYLEVEL1_1 DELIMITER"
7. for inference the model is prompted with a single summary to generate lower level summaries. this are then fed into the model repeatedly until the lowest level is reached, at which point the model generates the text of the book


## Usage

1. clone the repo
2. download data by running download_data.py
3. preprocess the data by running piperunner.py
4. train the model with this command, it will calculate BLEU and ROUGE-L scores every few epochs:
python run_lm_tuning.py --cache_dir model_cache --train_data_file pipeline/data/train.json --eval_data_file pipeline/data/eval.json --output_dir data/output_dir --do_train --do_eval --evaluate_during_training --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 4 --gradient_accumulation_steps 1 --datasets text_0 sum_1 sum_2 sum_3 sum_4 sum_5 sum_6 sum_7 sum_8 sum_9 sum_10 chapters_11 --model_type gpt2 --overwrite_output_dir --model_name_or_path gpt2-medium --block_size 400 --fp16 --num_train_epochs 1 --save_steps 200 --logging_steps 200
5. you can prompt the model by running the generate_book function:
generate_book(model, tokenizer, high_level_prompt= "This is a funny fantasy story about a lama that works as a wizard detective", max_chapters= 20, start_level=5, device, mid_start_level=5, max_input_len = 300, max_seq_len=400, sum_factor=3, prev_tokens_len=150):

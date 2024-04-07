import json
import os
import re
from json import JSONDecoder
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from connections.google_sheet import GoogleSheet
from utility.clean_gpu import clear_hardwares
from utility.preprocess import get_chunked
from datetime import datetime

regex = r"[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)*"
#

model = AutoModelForCausalLM.from_pretrained('Mohammadreza/llama-7b-lora-bslm-entity-attributes', return_dict=True, device_map='auto',
                                             token='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')
tokenizer = AutoTokenizer.from_pretrained('Mohammadreza/llama-7b-lora-bslm-entity-attributes', max_length=1024, token='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')


def get_json(output: str) -> dict:
    try:
        try:
            start_index = output.index('{')
            end_index = output.index('}')
            assert start_index < end_index
        except Exception as e:
            print(e)
            match = re.search(regex, output)
            persian_phrase = match.group()
            return persian_phrase, 0
        candid_str = output[start_index:end_index + 1]
        my_str_with_double_quotes = candid_str.replace("'", '"')
        json_data = json.loads(my_str_with_double_quotes)
        return json_data.get('product_type'), 1
    except Exception as e:
        print(e)
        return None, 0


def generate(prompt: str):
    batch = tokenizer(prompt, return_tensors='pt').to(model.device)

    prompt_length = len(batch.get('input_ids')[0])
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=80,
                                       do_sample=True,
                                       num_beams=2,
                                       temperature=0.1,
                                       top_k=10,
                                       top_p=.5,
                                       length_penalty=-1
                                       )
        output = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)
        # print(output)
    return output


def generate_batch(prompt: list, gen_kwargs: dict):
    s = datetime.now()
    batch = tokenizer(prompt, return_tensors='pt', padding=True)
    batch = {k: v.to('cuda') for k, v in batch.items()}
    prompt_lengths = [len(input_ids) for input_ids in batch['input_ids']]
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, **gen_kwargs) if gen_kwargs else model.generate(**batch)
    outputs = []
    for i, output_token in enumerate(output_tokens):
        print('output length: ', len(output_token[prompt_lengths[i]:]))
        output = tokenizer.decode(output_token[prompt_lengths[i]:], skip_special_tokens=True)
        outputs.append(output)

    clear_hardwares()
    e = (datetime.now() - s).seconds
    print('time to generate batch: ', e)
    return outputs


def extract_json_objects(text, decoder=JSONDecoder()):
    results = []
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


def to_sheet(df, tab_name):
    gs = GoogleSheet('Zero-Shot-CLS')
    gs.set_worksheet(tab_name=tab_name)
    gs.write_data(df, row=1, col=1)


def run():
    current_directory = os.getcwd()
    dataset_path = os.path.join(current_directory, 'entity_attribute_dataset')
    dataset = load_from_disk(dataset_path)

    split_dataset = dataset.train_test_split(train_size=300_000, seed=19, shuffle=False)
    valid_dataset = split_dataset["test"]
    # valid_dataset = valid_dataset.select(range(1000))

    df_valid_ds = valid_dataset.to_pandas()
    valid_ds = df_valid_ds.to_dict(orient='records')
    chunked_data = get_chunked(valid_ds, chunk_size=5, _type=list)

    product_predicts = list()
    input_length_list = list()
    gen_kwargs = {
        "max_new_tokens": 200}
    for ind, chunk in enumerate(tqdm(chunked_data)):
        if ind >= 0:
            chunked_input = [f"""### Question: {i.get('instruction')}\n ### Answer:""" for i in chunk]
            # instruct = """here is a product title from a Iranian marketplace.  \n         give me the Product Entity and Attributes of this product in Persian language.\n         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give Json not a single word more.\n         \nproduct title:"""
            # chunked_input = [f"""### Question: {instruct}{i}\n ### Answer:""" for i in chunk]
            batch = tokenizer(chunked_input, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            seq_length = len(batch.get('input_ids')[0])
            print('input length: ', seq_length)
            input_length_list.append(seq_length)
            if seq_length < 1024:
                generated_texts = generate_batch(prompt=chunked_input, gen_kwargs=gen_kwargs)
                # jsonformer = Jsonformer(model, tokenizer, json_schema, chunked_input)
                # generated_texts = jsonformer()
                for idx, (output, product) in enumerate(zip(generated_texts, chunk)):
                    title = product.get('instruction').split('product title')[-1]
                    extracted_json = extract_json_objects(output)
                    print(extracted_json)
                    if not extracted_json:
                        print('There is a None!')
                    product_predicts.append({'product_title': title, 'real': product.get('output'), 'predict': extracted_json if extracted_json else None})

    # df = pd.DataFrame(product_predicts)
    # # result = calculate(df)
    # # result1 = calculate_wighted(df)
    # to_sheet(df, 'Our-Llama-2-v0.02')

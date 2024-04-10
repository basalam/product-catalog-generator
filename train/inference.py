from json import JSONDecoder
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from utility.clean_gpu import clear_hardwares
from utility.preprocess import get_chunked
from datetime import datetime
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained('BaSalam/Llama2-7b-entity-attr-v1', return_dict=True, device_map='auto', token='')
tokenizer = AutoTokenizer.from_pretrained('BaSalam/Llama2-7b-entity-attr-v1', max_length=1024, token='')


def generate_batch(prompt: list, gen_kwargs: dict):
    s = datetime.now()
    batch = tokenizer(prompt, return_tensors='pt', padding=True)
    batch = {k: v.to('cuda') for k, v in batch.items()}
    prompt_lengths = [len(input_ids) for input_ids in batch['input_ids']]
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, **gen_kwargs) if gen_kwargs else model.generate(**batch)
    outputs = []
    for i, output_token in enumerate(output_tokens):
        output = tokenizer.decode(output_token[prompt_lengths[i]:], skip_special_tokens=True)
        outputs.append(output)

    clear_hardwares()
    e = (datetime.now() - s).seconds
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


def run():
    dataset = load_dataset("BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1")['train']
    split_dataset = dataset.train_test_split(train_size=300_000, seed=19, shuffle=False)
    valid_dataset = split_dataset["test"]

    df_valid_ds = valid_dataset.to_pandas()
    valid_ds = df_valid_ds.to_dict(orient='records')
    chunked_data = get_chunked(valid_ds, chunk_size=5, _type=list)

    product_predicts = list()
    gen_kwargs = {"max_new_tokens": 300}
    for ind, chunk in enumerate(tqdm(chunked_data)):
        chunked_input = [f"""### Question: {i.get('instruction')}\n ### Answer:""" for i in chunk]
        batch = tokenizer(chunked_input, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        seq_length = len(batch.get('input_ids')[0])
        if seq_length < 1024:
            generated_texts = generate_batch(prompt=chunked_input, gen_kwargs=gen_kwargs)
            for idx, (output, product) in enumerate(zip(generated_texts, chunk)):
                title = product.get('instruction').split('product title')[-1]
                extracted_json = extract_json_objects(output)
                product_predicts.append({'product_title': title, 'real': product.get('output'), 'predict': extracted_json if extracted_json else None})

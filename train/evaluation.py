from utility.postprocess import extract_json_objects
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from utility.clean_gpu import clear_hardwares
from utility.preprocess import get_chunked
from datasets import load_dataset
import argparse


def generate_batch(prompt: list, tokenizer, model, gen_kwargs: dict):
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
    return outputs



def run(**kwargs):
    fine_tuned_model = kwargs.get('fine_tuned_model')
    dataset_name_or_path = kwargs.get('dataset_name_or_path')
    response_template = kwargs.get('response_template')
    user_prompt_template = kwargs.get('user_prompt_template')

    model = AutoModelForCausalLM.from_pretrained(fine_tuned_model, return_dict=True, device_map='auto', token='')
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model, max_length=1024, token='')

    dataset = load_dataset(dataset_name_or_path)['train']
    split_dataset = dataset.train_test_split(train_size=300_000, seed=19, shuffle=False)
    valid_dataset = split_dataset["test"]

    df_valid_ds = valid_dataset.to_pandas()
    valid_ds = df_valid_ds.to_dict(orient='records')
    chunked_data = get_chunked(valid_ds, chunk_size=5, _type=list)

    product_predicts = list()
    gen_kwargs = {"max_new_tokens": 300}
    for ind, chunk in enumerate(tqdm(chunked_data)):
        chunked_input = [f"""{user_prompt_template} {i.get('instruction')}\n{response_template} """ for i in chunk]
        generated_texts = generate_batch(prompt=chunked_input, tokenizer=tokenizer, model=model, gen_kwargs=gen_kwargs)
        for idx, (output, product) in enumerate(zip(generated_texts, chunk)):
            title = product.get('instruction').split('product title')[-1]
            extracted_json = extract_json_objects(output)
            product_predicts.append({'product_title': title, 'real': product.get('output'), 'predict': extracted_json if extracted_json else None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_tuned_model', type=str, default='Llama2-7b-entity-attr-v1', help='The name of path of the base model')
    parser.add_argument('--dataset_name_or_path', type=str, default='BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1', help='The name or path of the dataset')
    parser.add_argument('--response_template', type=str, default=' ### Answer:', help='response template for LLM')
    parser.add_argument('--user_prompt_template', type=str, default='### Question: ', help='response template for LLM')
    parser.add_argument('--prompt', type=str,
                        default="""instruction': "here is a product title from a Iranian marketplace.  \n         give me the Product Entity and Attributes of this product in Persian language.\n         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give Json not a single word more.\n         \nproduct title:""",
                        help='Our prompt')

    args = parser.parse_args()
    args_dict = vars(args)
    run(**args_dict)


if __name__ == "__main__":
    main()

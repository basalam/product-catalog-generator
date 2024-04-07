from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import itertools
import os


# from models.chatgpt_kw_extract.fine_tune_llm.data_generator import generate_data


def create_datasets():
    # dataset = load_from_disk('/home/basalam1676/Desktop/projects/product_knowledge_base/instruct_bslm_all')
    current_directory = os.getcwd()
    # dataset_path = os.path.join(current_directory, 'entity_attribute_dataset_text_based')
    dataset_path = os.path.join(current_directory, 'entity_attribute_dataset')
    dataset = load_from_disk(dataset_path)
    print('process data for Mistral Model...')

    def replace_newline_with_tab(row):
        row['text'] = row['text'].replace('<<SYS>>\n', '').replace('\n<</SYS>>\n', '')
        return row

    # dataset = dataset.map(replace_newline_with_tab)
    # top_m = list(itertools.islice(dataset, 2))
    # dataset = generate_data(save=True)
    other_columns = [i for i in dataset.column_names if i not in ['instruction', 'output', 'text']]
    dataset = dataset.remove_columns(other_columns)
    split_dataset = dataset.train_test_split(train_size=300001, seed=19, shuffle=False)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")

    # chars_per_token = chars_token_ratio(train_data, tokenizer)
    # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    #
    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_data,
    #     formatting_func=prepare_sample_text,
    #     infinite=True,
    #     seq_length=seq_length,
    #     chars_per_token=chars_per_token,
    # )
    # valid_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     valid_data,
    #     formatting_func=prepare_sample_text,
    #     infinite=False,
    #     seq_length=seq_length,
    #     chars_per_token=chars_per_token,
    # )
    return train_dataset, valid_dataset


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

from datasets import load_dataset


def create_datasets():
    dataset = load_dataset("BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1")['train']
    other_columns = [i for i in dataset.column_names if i not in ['instruction', 'output', 'text']]
    dataset = dataset.remove_columns(other_columns)
    split_dataset = dataset.train_test_split(train_size=300001, seed=19, shuffle=False)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")
    return train_dataset, valid_dataset

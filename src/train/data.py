from datasets import load_dataset


def create_datasets(args, version):
    ds_path = args.dataset_name_or_path
    percent_of_train_dataset = args.percent_of_train_dataset
    dataset = load_dataset(ds_path)
    if version == 'v1':
        dataset = dataset['train']
        other_columns = [i for i in dataset.column_names if i not in ['instruction', 'output', 'text']]
        dataset = dataset.remove_columns(other_columns)
        split_dataset = dataset.train_test_split(train_size=int(dataset.num_rows * percent_of_train_dataset), seed=19, shuffle=False)
        train_dataset = split_dataset["train"]
        valid_dataset = split_dataset["test"]
    elif version == 'v2':
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    else:
        raise Exception(f'The version is wrong.')
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")
    return train_dataset, valid_dataset

from data import *
from training import *
from transformers import set_seed
from configs.train.config import Config

set_seed(19)


def main():
    config_args = Config()
    train_data, val_data = create_datasets(config_args)
    run_training(train_data, val_data, config_args)


if __name__ == "__main__":
    main()

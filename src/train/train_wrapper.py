import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore",
                        message=".*using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method.*")

import argparse
import traceback
import sys
import yaml
from transformers import set_seed

from train.arguments.sft_arguments import SFTTrainArguments
from train.data import create_datasets
from train.training import run_training


def load_config(version_):
    with open(f'train/{version_}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument('--version', type=str, default='v1', help='Version of the configuration to use.')

    args = parser.parse_args()
    version_ = args.version

    try:
        config_args = load_config(version_)

    except Exception as e:
        print("Failed to load configuration:", e)
        print(traceback.format_exc())

    else:
        config_args = SFTTrainArguments(**config_args)
        set_seed(config_args.seed)
        train_data, val_data = create_datasets(config_args, version=version_)
        run_training(train_data, val_data, config_args)
        print('Python %s on %s' % (sys.version, sys.platform))


if __name__ == '__main__':
    main()

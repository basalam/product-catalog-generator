import argparse
import yaml
import traceback
from vision.training import run


def load_config(version_):
    with open(f'vision/{version_}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument('--version', type=str, default='v1', help='Version of the configuration to use.')

    args = parser.parse_args()
    _version = args.version

    try:
        config_args = load_config(_version)

    except Exception as e:
        print("Failed to load configuration:", e)
        print(traceback.format_exc())

    run(config_args)


if __name__ == '__main__':
    main()

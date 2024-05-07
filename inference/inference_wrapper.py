from inference.vllm_engine import inference_model
from configs.inference.config import Config


def main():
    inference_args = Config()
    product_list = ['برنج طارم هاشمی ممتاز شمال امساله (10 کیلوگرم)']
    generation_config = {'temperature': 0.0, 'max_tokens': 300}
    inference_model(product_list, generation_config, inference_args)


if __name__ == "__main__":
    main()

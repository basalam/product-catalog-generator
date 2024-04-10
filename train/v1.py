from data import *
from training import *
from transformers import set_seed
import argparse

set_seed(19)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.00015, help='The learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='The number of epochs for training')
    parser.add_argument('--rank', type=int, default=128, help='The rank for LoRA')
    parser.add_argument('--base_model', type=str, default='NousResearch/Llama-2-7b-chat-hf', help='The name of path of the base model')
    parser.add_argument('--project_name', type=str, default='Llama2-7b-entity-attr-v1', help='The name of the project')
    parser.add_argument('--dataset_name_or_path', type=str, default='BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1', help='The name or path of the dataset')
    parser.add_argument('--response_template', type=str, default=' ### Answer:', help='response template for LLM')
    parser.add_argument('--user_prompt_template', type=str, default='### Question: ', help='response template for LLM')
    parser.add_argument('--prompt', type=str,
                        default="""instruction': "here is a product title from a Iranian marketplace.  \n         give me the Product Entity and Attributes of this product in Persian language.\n         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give Json not a single word more.\n         \nproduct title:""",
                        help='Our prompt')

    args = parser.parse_args()
    args_dict = vars(args)
    train_data, val_data = create_datasets(**args_dict)
    run_training(train_data=train_data, val_data=val_data, **args_dict)


if __name__ == "__main__":
    main()

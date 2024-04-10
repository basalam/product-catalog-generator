from inference.vllm_engine import inference_model
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fine_tuned_model', type=str, default='Llama2-7b-entity-attr-v1', help='The name of path of the base model')
    parser.add_argument('--response_template', type=str, default=' ### Answer:', help='response template for LLM')
    parser.add_argument('--user_prompt_template', type=str, default='### Question: ', help='response template for LLM')
    parser.add_argument('--prompt', type=str,
                        default="""instruction': "here is a product title from a Iranian marketplace.  \n         give me the Product Entity and Attributes of this product in Persian language.\n         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give Json not a single word more.\n         \nproduct title:""",
                        help='Our prompt')
    product_list = ['برنج طارم هاشمی ممتاز شمال امساله (10 کیلوگرم)']
    generation_config = {'temperature': 0.0, 'max_tokens': 300}

    args = parser.parse_args()
    args_dict = vars(args)
    inference_model(product_list=product_list, generation_config=generation_config, **args_dict)


if __name__ == "__main__":
    main()

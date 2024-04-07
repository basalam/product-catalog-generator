# model_name_or_path = 'HuggingFaceH4/zephyr-7b-beta'
model_name_or_path = 'NousResearch/Llama-2-7b-chat-hf'
# model_name_or_path = 'mostafaamiri/base_7B'
# model_name_or_path = 'MaralGPT/Maral-7B-alpha-1'
# model_name_or_path = 'mistralai/Mistral-7B-Instruct-v0.1'
project_name = 'bslm_entity_attributes_v2'

prompt = """here is a product title from a Iranian marketplace.  
         give me the product type of this product title in Persian language. 
         give the output in this json format: {'product_type': <your output>}.
         Don't make assumptions about what values to plug into json. Just give Json and not a single word more.
         \nproduct title:"""

prompt_type_list = """here is a product title from a Iranian marketplace.  
         give me the possible product types of this product title in Persian language. 
         give the output in this json format: {'product_types_list': [list of objects of product types in Persian].
         Don't make assumptions about what values to plug into json. Just give Json and not a single word more.
         \nproduct title:"""

prompt_en_attr = """here is a product title from a Iranian marketplace. give me the Product Entity and Attributes of this product in Persian language. give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}. Don't make assumptions about what values to plug into json. Just give Json not a single word more. \nproduct title:"""

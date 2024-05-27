class Config:
    def __init__(self,
                 fine_tuned_model='BaSaLam/Llama2-7b-entity-attr-v1',
                 response_template=' ### Answer:',
                 user_prompt_template=  '### Question: ',
                 prompt = """instruction': "here is a product title from a Iranian marketplace.  \n         give me 
                 the Product Entity and Attributes of this product in Persian language.\n         give the output in 
                 this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': 
                 '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give 
                 Json not a single word more.\n         \nproduct title:"""):
        self.fine_tuned_model = fine_tuned_model
        self.response_template = response_template
        self.user_prompt_template = user_prompt_template
        self.prompt = prompt

class Config:
    def __init__(self,
                 learning_rate=0.00015,
                 batch_size=16,
                 num_epochs=2,
                 rank=128,
                 base_model="NousResearch/Llama-2-7b-chat-hf",
                 project_name="Llama2-7b-entity-attr-v1",
                 dataset_name_or_path="BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1",
                 response_template=" ### Answer:",
                 user_prompt_template="### Question: ",
                 percent_of_train_dataset=0.985,
                 prompt="""instruction': \"here is a product title from a Iranian marketplace.  \n         give me 
                 the Product Entity and Attributes of this product in Persian language.\n         give the output in 
                 this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': 
                 '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give 
                 Json not a single word more.\n         \nproduct title:"""):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.rank = rank
        self.base_model = base_model
        self.project_name = project_name
        self.dataset_name_or_path = dataset_name_or_path
        self.response_template = response_template
        self.user_prompt_template = user_prompt_template
        self.percent_of_train_dataset = percent_of_train_dataset
        self.prompt = prompt

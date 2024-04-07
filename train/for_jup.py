import json
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_name_or_path = 'mistralai/Mistral-7B-Instruct-v0.1'
prompt_en_attr = """here is a product title from a Iranian marketplace.  
         give me the Product Entity and Attributes of this product in Persian language.
         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.
         Don't make assumptions about what values to plug into json. Just give Json not a single word more.
         \nproduct title:"""

storing_path = '/home/basalam1676/Desktop/projects/product_knowledge_base/gpt_3.5_turbo/'
product_list = os.listdir(storing_path)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

products = list()
for product in tqdm(product_list):
    with open(storing_path + product, 'r') as f:
        data = json.load(f)
        products.append(data)

instructed_products = list()
for idx, product in tqdm(enumerate(products)):
    title = product.get('title')
    product_entity = product.get('product_entity', None)
    if product_entity:
        attribute_keys = [i for i in product if 'attribute' in i]
        structure = {'attributes': {}}
        for attribute_key in attribute_keys:
            structure['attributes'][list(product[attribute_key].keys())[0]] = list(product[attribute_key].values())[0]
        structure['product_entity'] = product.get('product_entity')

        prompt = """here is a product title from a Iranian marketplace. give me the Product Entity and Attributes of this product in Persian language. give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}. Don't make assumptions about what values to plug into json. Just give Json not a single word more."""
        chat_temp = [{"role": "system", "content": prompt},
                     {"role": "user", "content": f'product title: {title}'},
                     {"role": "assistant", "content": json.dumps(structure, ensure_ascii=False)}]
        text = tokenizer.apply_chat_template(chat_temp, tokenize=False)
        obj = {'text': text}

        # obj = {'instruction': f'{prompt_en_attr} {title}',
        #        'output': json.dumps(structure, ensure_ascii=False)}
        instructed_products.append(obj)

df = pd.DataFrame(instructed_products)
shuffled_df = df.sample(frac=1, random_state=42)
shuffled_df = shuffled_df.reset_index(drop=True)
dataset = Dataset.from_pandas(shuffled_df)


# text_based_path = '/home/basalam1676/Desktop/projects/product_knowledge_base/entity_attribute_dataset_text_based'
# path = '/home/basalam1676/Desktop/projects/product_knowledge_base/entity_attribute_dataset'
# ds.save_to_disk(text_based_path)


# ---------------------------------------------------------------------------------------------
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def replace_newline_with_tab(row):
    row['text'] = row['text'].replace('<<SYS>>\n', '').replace('\n<</SYS>>\n', '')
    return row


# dataset = load_from_disk('/home/user01/fine_tune_llm/products-knowledge-base/entity_attribute_dataset_text_based/')
dataset = dataset.map(replace_newline_with_tab)
other_columns = [i for i in dataset.column_names if i not in ['instruction', 'output', 'text']]
dataset = dataset.remove_columns(other_columns)
split_dataset = dataset.train_test_split(train_size=300000, seed=19, shuffle=False)
train_data = split_dataset["train"]
val_data = split_dataset["test"]
print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(val_data)}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

project_name = 'bslm_entity_attributes'

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", 'k_proj'],
    task_type="CAUSAL_LM"
)

print("Starting main loop")

training_args = TrainingArguments(
    output_dir='output_dir',
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    num_train_epochs=1,
    # max_steps=11625,
    eval_steps=37500,
    # dataloader_num_workers = 4,
    save_steps=0,
    logging_steps=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # learning_rate=1.41e-5,
    learning_rate=0.00015,
    # lr_scheduler_type=lr_scheduler_type,
    warmup_steps=100,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    weight_decay=0.01,
    fp16=False,
    bf16=False
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             quantization_config=bnb_config,
                                             # load_in_8bit=True,
                                             device_map={"": Accelerator().process_index})
model.config.use_cache = False

trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    # formatting_func=formatting_prompts_func,
    # data_collator=collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    # neftune_noise_alpha=5,
    packing=False,
    max_seq_length=1024
    # dataset_num_Sproc=4
)

print_trainable_parameters(trainer.model)

print("Training...")
trainer.train()

print("Saving last checkpoint of the model")
trainer.model.save_pretrained(os.path.join('output_dir', project_name))

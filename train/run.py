from data import *
from training import *
from transformers import AutoTokenizer, logging, set_seed
from config import model_name_or_path, project_name
import os

set_seed(19)
current_directory = os.getcwd()
output_dir = os.path.join(current_directory, "outputs")

# seq_length = 256
# logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# tokenizer.decode([673, 29901])
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


# project_name = 'bslm_honey'


# import torch
# chat = [{"role": "user", "content": "act as a information extractor and extract product entity and product attributes for given product. product: عسل بهارنارنج اردبیل"},
#         {'role': 'assistant', 'content': 'ok hello'}]
# chat = [{"role": "system", "content": "You are a hunan who loves to dance."},
#         {"role": "user", "content": "What do you like to do in your spare time?"}]
# tokenizer.use_default_system_prompt = True
# en = tokenizer.apply_chat_template(chat, tokenize=False, return_tensors="pt", add_generation_prompt=False)
# with torch.cuda.amp.autocast(): output_tokens = model.generate(en, max_new_tokens=250, do_sample=True, num_beams=2, temperature=0.1, top_k=10, top_p=.5, length_penalty=-1)
# tokenizer.decode(output_tokens[0], skip_special_tokens=False)


def run():
    train_data, val_data = create_datasets()
    run_training(output_dir=output_dir, model_name_or_path=model_name_or_path, train_data=train_data, val_data=val_data,
                 tokenizer=tokenizer, project_name=project_name)

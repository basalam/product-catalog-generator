from data import *
from training import *
from transformers import AutoTokenizer, set_seed
from config import model_name_or_path, project_name
import os

set_seed(19)
current_directory = os.getcwd()
output_dir = os.path.join(current_directory, "outputs")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


def run():
    train_data, val_data = create_datasets()
    run_training(output_dir=output_dir, model_name_or_path=model_name_or_path, train_data=train_data, val_data=val_data,
                 tokenizer=tokenizer, project_name=project_name)

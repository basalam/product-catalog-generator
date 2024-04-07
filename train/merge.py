import os
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from config import project_name
from peft import PeftModel, PeftConfig
from huggingface_hub import notebook_login

base_model = AutoModelForCausalLM.from_pretrained('Mohammadreza/llama-7b-lora-bslm-entity-attributes', return_dict=True, device_map='auto',
                                                  token='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')
tokenizer = AutoTokenizer.from_pretrained('Mohammadreza/llama-7b-lora-bslm-entity-attributes', max_length=1024, token='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')

notebook_login(write_permission=True)
current_directory = os.getcwd()
output_dir = os.path.join(current_directory, "outputs")
peft_model_id = os.path.join(output_dir, project_name)
config = PeftConfig.from_pretrained(peft_model_id)
model_to_merge = PeftModel.from_pretrained(base_model, peft_model_id)
model = model_to_merge.merge_and_unload()
model.save_pretrained('llama-2-7b-lora-bslm-entity-attributes')
model.push_to_hub("Mohammadreza/llama-7b-lora-bslm-entity-attributes", token='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')

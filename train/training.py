from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer
from peft import PeftModel
from utility.clean_gpu import clear_hardwares


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


def run_training(train_data, val_data, **kwargs):
    base_model = kwargs.get('base_model')
    rank = kwargs.get('rank')
    project_name = kwargs.get('project_name')
    num_epochs = kwargs.get('num_epochs')
    batch_size = kwargs.get('batch_size')
    learning_rate = kwargs.get('learning_rate')
    response_template = kwargs.get('response_template')
    user_prompt_template = kwargs.get('user_prompt_template')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            out = example['output'][i]
            text = f"{user_prompt_template}{example['instruction'][i]}\n{response_template} {out}"
            output_texts.append(text)
        return output_texts

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", 'k_proj'],
        task_type="CAUSAL_LM"
    )

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=project_name,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        num_train_epochs=num_epochs,
        eval_steps=75000,
        save_steps=0,
        logging_steps=30,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # learning_rate=1.41e-5,
        learning_rate=learning_rate,
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

    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 quantization_config=bnb_config,
                                                 # load_in_8bit=True,
                                                 device_map={"": Accelerator().process_index})
    model.config.use_cache = False

    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=False,
        max_seq_length=1024
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(project_name)
    # merge LoRa Adaptor with base model and push it in HF
    model_to_merge = PeftModel.from_pretrained(model, project_name)
    del model
    clear_hardwares()
    model = model_to_merge.merge_and_unload()
    model.push_to_hub(f'BaSalam/{project_name}')

from accelerate import Accelerator
from peft import LoraConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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


def run_training(train_data, val_data, args):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = args.padding_side

    def formatting_prompts_func_v1(example):
        output_texts = []
        instruction = example['instruction']
        output = example['output']
        for i in range(len(instruction)):
            out = output[i]
            text = f"{args.user_prompt_template}{instruction[i]}\n{args.response_template} {out}"
            output_texts.append(text)
        return output_texts

    def formatting_prompts_func_v2(example):
        output_texts = []
        instruction = example['system_prompt']
        output = example['product_data']
        for i in range(len(instruction)):
            out = output[i]
            text = f"{args.user_prompt_template}{instruction[i]}\n{args.response_template} {out}"
            output_texts.append(text)
        return output_texts

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=args.lora_target_modules,
        task_type=args.lora_task_type
    )

    print("Starting main loop")

    training_args = TrainingArguments(**args.to_dict())

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 quantization_config=bnb_config,
                                                 load_in_8bit=args.load_in_8bit,
                                                 device_map={"": Accelerator().process_index})
    model.config.use_cache = False

    response_template_ids = tokenizer.encode(args.response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=args.mlm)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func_v1 if args.version == '1.0' else formatting_prompts_func_v2,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=False,
        max_seq_length=args.max_seq_length
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(args.project_name)
    # merge LoRa Adaptor with base model and push it in HF
    model_to_merge = PeftModel.from_pretrained(model, args.project_name)
    del model
    clear_hardwares()
    model = model_to_merge.merge_and_unload()
    model.push_to_hub(f'BaSalam/{args.project_name}')

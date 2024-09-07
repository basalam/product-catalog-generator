from datasets import load_dataset
from peft import LoraConfig
from transformers import LlavaForConditionalGeneration, AutoProcessor
from trl import SFTTrainer, SFTConfig
import torch


def run(args):
    dataset_name = args.dataset_name_or_path
    model_id = args.model_name_or_path
    experiment_name = args.project_name

    model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                          torch_dtype=torch.float16,
                                                          device_map='auto')
    model.config.use_cache = False

    LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
    processor = AutoProcessor.from_pretrained(model_id)
    processor.chat_template = LLAVA_CHAT_TEMPLATE
    processor.tokenizer.padding_side = 'right'

    def get_prompt(product):
        product = product["text"]
        json_format = """attributes': {'attribute_name_1' : <list of attribute values>, 'attribute_name_2': <list of attribute values>, ...}"""
        entity = product['entity']
        final_prompt = f"""برای محصول داده شده، ویژگی‌های تصویری محصول را در قالب جیسون (json) استخراج کن. ساختار JSON باید به این شکل باشد: {json_format}. محصول از یک بازار اینترنتی ایرانی است پس خروجی Json باید به زبان فارسی باشد.
    محصول: '{entity}'."""
        return final_prompt

    class LLavaDataCollator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, examples):
            texts = []
            images = []
            for example in examples:
                prompt = get_prompt(example)
                real_output = example['text']['gpt_output']

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": real_output}
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                images.append(example["image"])

            batch = self.processor(texts, images, return_tensors="pt", padding=True)

            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch

    data_collator = LLavaDataCollator(processor)
    raw_datasets = load_dataset(dataset_name)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        # use_dora=True,
        target_modules=args.target_modules)

    sft_config = SFTConfig(
        max_seq_length=args.max_seq_length,
        dataset_text_field='text',
        dataset_kwargs={"skip_prepare_dataset": True},
        output_dir=experiment_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=args.fp16,
        bf16=args.bf16,
        lr_scheduler_type='linear',
        weight_decay=args.weight_decay,
        # torch_compile=True,
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        save_strategy='steps',
        save_steps=args.save_steps
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
        tokenizer=processor.tokenizer,
        data_collator=data_collator)

    trainer.train()
    trainer.model.save_pretrained(experiment_name)
    trainer.push_to_hub(experiment_name)

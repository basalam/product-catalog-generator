import inspect
from transformers import TrainingArguments

class SFTTrainArguments(TrainingArguments):
    """
    Extends the TrainingArguments class to include additional parameters specific to Sparse Fine-Tuning (SFT)
    of language models. This class provides structured management of training parameters to ensure they
    are easily adjustable and maintainable, accommodating specific requirements like LoRA layers.
    """

    def __init__(self,
                 # model specifications
                 fine_tuning_version: str = '1.0',
                 max_seq_length: int = 1024,
                 prompt: str= '',
                 dataset_name_or_path: str = "",
                 model_name_or_path: str = "",
                 load_in_8bit: bool = True,
                 # TrainingArguments specifications
                 project_name: str = "Llama2-7b-entity-attr",
                 # LoRA specifications
                 lora_bias: [float] = "none",
                 lora_rank: int = 128,
                 lora_task_type: str = "CAUSAL_LM",
                 lora_dataloader_drop_last: bool = True,
                 lora_target_modules=None,
                 lora_dropout: float = 0.05,
                 lora_alpha: int = 256,
                 # BitsAndBytes specifications
                 load_in_4bit: bool = True,
                 bnb_4bit_quant_type: str = 'nf4',
                 bnb_4bit_compute_dtype: str = 'float16',
                 bnb_4bit_use_double_quant: bool = False,
                 # tokenizer specifications
                 response_template: str = " ### Answer:",
                 user_prompt_template: str = "### Question: ",
                 percent_of_train_dataset: float = 0.985,
                 padding_side: str = "right",
                 mlm: bool = False,
                 **kwargs):
        """
        Initializes SFT training arguments with additional sparse fine-tuning specific parameters.

        Additional parameters:
        :param lora_bias: Bias value for LoRA layers, defaults to None.
        :param training_seed: Seed for random number generators to ensure reproducibility.
        :param lora_task_type: Specifies the task type for LoRA configuration.
        :param lora_dataloader_drop_last: Whether to drop the last incomplete batch during training.
        :param eval_steps: Evaluation frequency in terms of steps.
        :param bnb_4bit_quant_type: Quantization type for BitsAndBytes.
        :param bnb_4bit_compute_dtype: Compute data type for quantization.
        :param bnb_4bit_use_double_quant: Whether to use double quantization.
        :param mlm: Whether the Masked Language Modeling is used.
        """
        # Initialize the superclass with all the parameters it needs
        supported_params = {key: kwargs[key] for key in kwargs if
                            key in inspect.signature(TrainingArguments).parameters}

        # Initialize the superclass with only supported parameters
        super().__init__(output_dir=f'{project_name}_v{fine_tuning_version}',
                         **supported_params)

        # Initialize new attributes specific to SFT
        # model specifications
        self.version = fine_tuning_version
        self.prompt = prompt
        self.dataset_name_or_path = dataset_name_or_path
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.load_in_8bit = load_in_8bit
        # LoRA specifications
        self.lora_bias = lora_bias
        self.lora_rank = lora_rank
        self.lora_task_type = lora_task_type
        self.lora_dataloader_drop_last = lora_dataloader_drop_last
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        # BitsAndBytes specifications
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        # tokenizer specifications
        self.response_template = response_template
        self.user_prompt_template = user_prompt_template
        self.percent_of_train_dataset = percent_of_train_dataset
        self.mlm = mlm
        self.padding_side = padding_side

    @classmethod
    def get_valid_keys_for_class(cls, target_class):
        """ Retrieves valid constructor parameter names for the specified class, excluding 'self'. """
        init_signature = inspect.signature(target_class.__init__)
        return [name for name, param in init_signature.parameters.items() if name != 'self']

    def to_dict(self):
        """ Converts attributes to a dictionary filtering by valid keys of TrainingArguments. """
        valid_keys = self.get_valid_keys_for_class(TrainingArguments)
        return {k: v for k, v in vars(self).items() if k in valid_keys and not k.startswith('_')}

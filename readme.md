# Product catalog generator 

This repo is the source code for a custom LLM and VLM fine tuned on [LLama 2](https://huggingface.co/docs/transformers/en/model_doc/llama2) and [Llava1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) based on [Basalam](https://basalam.com/) products to infer enitty (product types) and attributes based on product data. You can use it on any similar dataset.


## Datasets

### Dataset V1 generated using GPT-3.5
[GPT-3.5 generated product data](https://huggingface.co/datasets/BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1)
### Dataset V2 generated using GPT-4
[GPT-4 generated product data](https://huggingface.co/datasets/BaSalam/entity-attribute-sft-dataset-GPT-4.0-generated-v1)

### Dataset for Vision catalog generated using GPT-4 (✅New)
[GPT-4 generated product data](https://huggingface.co/datasets/BaSalam/vision-catalogs-llava-format-v3)

## Models

### Sft model version 1 based on llama 2 and GPT-3.5 data.
[Model V1](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1)

### Sft model version 2 based on llama 2 and GPT-4 data.
[Model V2](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v2)

### Sft model based on Llava1.5 and GPT-4 data (✅New)
[Model](https://huggingface.co/BaSalam/Llava-1.5-7b-hf-bslm-product-attributes-v0)
### LoRA file for Vision Catalog
[Model](https://huggingface.co/BaSalam/llava1.5-7b-bslm-products-vision-catalog-lora)


## Evaluation

| model    | train loss | val loss | download                                                                                                                          
|----------|------------|----------|-----------------------------------------------------------------------------------------------------------------------------------|
| Model V1 | 0.07       | 0.08     | [Sft model version 1 based on llama 2 and GPT-3.5 data.](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1)                 
| Model V2 | 0.1        | 0.12     | [Sft model version 2 based on llama 2 and GPT-4 data.](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v2)                   |
| vision   | 0.11       | 0.13     | [Sft model based on llava1.5 and GPT-4 data.](https://huggingface.co/BaSalam/Llava-1.5-7b-hf-bslm-product-attributes-v0)          |

## Motivations

Problem definition and roadmap to solve it (in Persian). [Virgool link](https://experience.basalam.com/%D9%85%D8%B3%D8%A7%D9%84%D9%87-%D8%AA%D8%B4%D8%AE%DB%8C%D8%B5-%D9%85%D8%AD%D8%B5%D9%88%D9%84%D8%A7%D8%AA-%D8%A8%D8%A7%D8%B3%D9%84%D8%A7%D9%85-%DB%8C%DA%A9-%D8%AA%D8%AC%D8%B1%D8%A8%D9%87-%D8%B9%D9%85%D9%84%DB%8C-%D8%A7%D8%B2-%D8%A8%D9%87-%DA%A9%D8%A7%D8%B1%DA%AF%DB%8C%D8%B1%DB%8C-llm%D9%87%D8%A7-m8sr2xt1dhdk).


## How to use


**Train:**

To finetune a new model, you can either create a new YAML configuration file with your specific parameters or modify an existing one. You'll find example configuration files in the src/train/ directory (**[config](https://github.com/basalam/product-catalog-generator/blob/main/src/train/v1.yaml), **[config](https://github.com/basalam/product-catalog-generator/blob/main/src/train/v2.yaml)). The default base model is `NousResearch/Llama-2-7b-chat-hf`, but you are free to change it. It's advisable to adjust the LoRA parameters accordingly if you do. Tailor other parameters to the needs of your task and dataset.

To initiate finetuning, navigate to the src directory and start the process with:

    python -m train.train_wrapper --version v1

Here, --version v1 corresponds to the version of the finetuning configuration, which should match the name of your YAML file.

The training process includes several steps:

    1- Parameter Initialization: Loads parameters from the specified YAML file.
    2- Dataset Loading:
        - Retrieves the dataset from the Hugging Face hub using the _create_datasets_ function.
    3- Training Execution: Handled by the _run_training_ method from the **[training_module](https://github.com/basalam/product-catalog-generator/blob/main/src/train/training.py):
        - Initializes configuration settings.
        - Prepares the model and tokenizer.
        - Sets up training arguments.
        - Begins the training cycle.
        - After completing the last iteration, saves the model, merging LoRA configurations with the base model using peft.
        - Uploads the trained model to the Hugging Face hub.
        - Concludes the process!

This structured approach ensures comprehensive management and execution of the model training process.

**Inference:**

For inference we use llm inference engine [vllm](https://github.com/vllm-project/vllm).

Inference config such as model, prompt and response templates are located at ````BASE_DIR/configs/inference/````.
Start by running inference_wrapper in inference directory. The process is as follows:
1.  Reading config from config file
2.  Running _inference_model_ from **[vllm_engine](https://github.com/basalam/product-catalog-generator/blob/main/inference/vllm_engine.py)** module 
    - Args is read
    - LLM inference engine is built 
    - For each sample input (prompt + input values (typically a product information)), a response is generated

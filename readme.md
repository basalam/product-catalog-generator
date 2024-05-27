# Product catalog generator 

This repo is the source code for a custom LLM fine tuned on [LLama 2](https://huggingface.co/docs/transformers/en/model_doc/llama2) based on [Basalam](https://basalam.com/) products to infer enitty (product types) and attributes based on product data. You can use it on any similar dataset.


## Datasets

### Dataset V1 generated using GPT-3.5
[GPT-3.5 generated product data](https://huggingface.co/datasets/BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1)
### Dataset V2 generated using GPT-4
[GPT-4 generated product data](https://huggingface.co/datasets/BaSalam/entity-attribute-sft-dataset-GPT-4.0-generated-v1)

## Models

### Sft model version 1 based on llama 2 and GPT-3.5 data.
[Model V1](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1)

### Sft model version 2 based on llama 2 and GPT-4 data.
[Model V2](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v2)


## Evaluation

| model | train loss | val loss | download
| --- | --- | --- | --- |
| Model V1 | 1.0 | 1.297 | [Sft model version 1 based on llama 2 and GPT-3.5 data.](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1)
| Model V2 | 1.0 | 1.072 | [Sft model version 2 based on llama 2 and GPT-4 data.](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v2) |

## Motivations

Problem definition and roadmap to solve it (in Persian). [Dropbox link](https://www.dropbox.com/scl/fi/xjr81mna7ae5tlwco461q/LLM.paper?rlkey=fpimc6mm2hqrke31t7bqs7e38&dl=0).

Medium story introducing problem and devised solution. [Medium story](https://medium.com/p/72bf6abd22eb/) (in progress)


## How to use


**Train:**

To finetune a new model, you can either create a new YAML configuration file with your specific parameters or modify an existing one. You'll find example configuration files in the src/train/ directory (**[config](https://github.com/basalam/product-catalog-generator/blob/main/src/train/v1.yaml), **[config](https://github.com/basalam/product-catalog-generator/blob/main/src/train/v2.yaml)). The default base model is NousResearch/Llama-2-7b-chat-hf, but you are free to change it. It's advisable to adjust the LoRA parameters accordingly if you do. Tailor other parameters to the needs of your task and dataset.

To initiate finetuning, navigate to the src directory and start the process with:
bash
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

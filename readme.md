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

For finetuning a new model you should set some parameters in **[config](https://github.com/basalam/product-catalog-generator/blob/main/configs/train/config.py)** file (located at ````BASE_DIR/configs/train/````).
Base model is ````NousResearch/Llama-2-7b-chat-hf````, but you may change it if you prefer, it is recommended to change Lora parameters accordingly. The other parameters are set based on the task and/or dataset.

Start by running training_wrapper in train directory. Training process is as follows:
1.  Reading config from config file
2.  Loading dataset
    - Dataset is loaded from hugging face (_create_datasets_ function)
3.  Running _run_training_ from **[training](https://github.com/basalam/product-catalog-generator/blob/main/train/training.py)** module 
    - Config is read
    - Model and tokenizer are loaded
    - Training args are set
    - Training begins
    - After the last iteration model is saved (Merging Lora config with base model using **peft**)
    - Pushing model to hub
    - Finish!

**Inference:**

For inference we use llm inference engine [vllm](https://github.com/vllm-project/vllm).

Inference config such as model, prompt and response templates are located at ````BASE_DIR/configs/inference/````.
Start by running inference_wrapper in inference directory. The process is as follows:
1.  Reading config from config file
2.  Running _inference_model_ from **[vllm_engine](https://github.com/basalam/product-catalog-generator/blob/main/inference/vllm_engine.py)** module 
    - Args is read
    - LLM inference engine is built 
    - For each sample input (prompt + input values (typically a product information)), a response is generated

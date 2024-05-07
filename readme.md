![stack Overflow](https://biaupload.com/do.php?imgf=org-76318f19e01f1.png)




# Product catalog generator 

This repo is the source code for a custom LLM fine tuned on [LLama 2](https://huggingface.co/docs/transformers/en/model_doc/llama2) based on [Basalam](https://basalam.com/) products to infer enitty (product types) and attributes based on product data. You can use it on any similar dataset.


## Datasets

[GPT4 generated product data](https://huggingface.co/datasets/BaSalam/entity-attribute-sft-dataset-GPT-4.0-generated-v1)

## Models

### Sft model version 1 based on lama2 and GPT4 data.
[Model V1](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1)

### V2 model

Will be uploaded soon.

## Motivations

Problem definition and roadmap to solve it (in Persian). [Dropbox link](https://www.dropbox.com/scl/fi/xjr81mna7ae5tlwco461q/LLM.paper?rlkey=fpimc6mm2hqrke31t7bqs7e38&dl=0).

Medium story introducing problem and devised solution. [Medium story](https://medium.com/p/72bf6abd22eb/) (in progress)


## How to use


**Train:**

For finetuning a new model you should set some parameters in **config** file (located at _BASE_DIR/configs/train/_).
Base model is NousResearch/Llama-2-7b-chat-hf, but you may change it if you prefer, it is recommended to change Lora parameters accordingly. The other parameters are set based on the task and/or dataset.

Start by running training_wrapper in train directory. Training process is as follows:
1.  > Reading config from config file
2.  > Loading dataset
    > Dataset is loaded from hugging face (_create_datasets_ function)
3.  > Running _run_training_ from training module 
    1. > Config is read
       > Model and tokenizer are loaded
       > Training args are set
       > Training begins
       > After the last iteration model is saved (Merging Lora config with base model using **peft**)
       > Pushing model to hub
       > Finish!

from utility.postprocess import extract_json_objects
from vllm import LLM, SamplingParams
from huggingface_hub import login

login(token='')


def get_attributes_entity(catalog):
    must_keys = ['attributes', 'product_entity']
    if isinstance(catalog, dict):
        keys = list(catalog.keys())
        if keys == must_keys:
            attributes = catalog['attributes']
            product_entity = catalog['product_entity']
        else:
            attributes = catalog
            product_entity = None
        return {'attributes': attributes, 'product_entity': product_entity}
    else:
        ds = {'attributes': list(), 'product_entity': list()}
        for cat in catalog:
            content = get_attributes_entity(cat)
            ds['attributes'].append(content['attributes'])
            ds['product_entity'].append(content['product_entity'])
        return ds


def processed_catalog(catalog):
    if isinstance(catalog, list):
        if len(catalog) == 1:
            catalog_content = get_attributes_entity(catalog[0])
        elif len(catalog) >= 2:
            content_list = [get_attributes_entity(i) for i in catalog]
            final_product_entity = None
            final_attributes = dict()
            for cnt in content_list:
                attributes = cnt['attributes']
                product_entity = cnt['product_entity']
                if product_entity:
                    final_product_entity = product_entity
                final_attributes.update(attributes)
            catalog_content = {'attributes': final_attributes, 'product_entity': final_product_entity}
        else:
            return []
        return catalog_content
    elif isinstance(catalog, dict):
        catalog_content = get_attributes_entity(catalog)
        return catalog_content


def inference_model(product_list: list, generation_config, args):
    fine_tuned_model = args.fine_tuned_model
    response_template = args.response_template
    user_prompt_template = args.user_prompt_template
    prompt = args.prompt
    llm = LLM(model=fine_tuned_model, gpu_memory_utilization=0.9, trust_remote_code=True)
    sampling_params = SamplingParams(**generation_config)
    prompts = []
    for product in product_list:
        prompts.append(f'{user_prompt_template} {prompt}{product}\n {response_template}')
    outputs = llm.generate(prompts, sampling_params)
    results = list()
    for ind, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        generated_obj = extract_json_objects(generated_text)
        generated_obj = processed_catalog(generated_obj)
        results.append(generated_obj)
    return results

from json import JSONDecoder
from vllm import LLM, SamplingParams
from inference.config import config
from huggingface_hub import login

login(token='')

prompt = """### Question: here is a product title from a Iranian marketplace.  \n         give me the Product Entity and Attributes of this product in Persian language.\n         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.\n         Don't make assumptions about what values to plug into json. Just give Json not a single word more.\n         \nproduct title:"""

llm = LLM(model='BaSalam/Llama2-7b-entity-attr-v1', gpu_memory_utilization=0.9, trust_remote_code=True)


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


def extract_json_objects(text, decoder=JSONDecoder()):
    results = []
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


def inference_model(product_list: list, config):
    sampling_params = SamplingParams(**config)
    prompts = []
    for product in product_list:
        prompts.append(f'{prompt} {product}\n ### Answer:')
    outputs = llm.generate(prompts, sampling_params)
    results = list()
    for ind, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        generated_obj = extract_json_objects(generated_text)
        generated_obj = processed_catalog(generated_obj)
        results.append(generated_obj)
    return results

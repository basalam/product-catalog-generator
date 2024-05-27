import itertools
import re
import json


def read_json(fp):
    with open(fp, 'r') as f:
        ds = json.load(f)
    return ds


REGEX_PRUNE1 = r'[A-z]|(?<![\w\d])مدل(?![\w\d])|(?<![\w\d])و(?![\w\d])|(?<![\w\d])کیلو(?![\w\d])|(?<![\w\d])گرم(?![\w\d])|(?<![\w\d])وزن(?![\w\d])|(?<![\w\d])کد(?![\w\d])|(?<![\w\d])تازه(?![\w\d])|تخفیف|ارسال|رایگان|تضمینی|درجه|سانتی|سانت|درصدی|درصد|(?<![\w\d])فروشی(?![\w\d])|(?<![\w\d])فروش(?![\w\d])|(?<![\w\d])ویژه(?![\w\d])|\W|(?<![\w\d])انواع(?![\w\d])|(?<![\w\d])عمده(?![\w\d])|(?<![\w\d])گرمی(?![\w\d])|(?<![\w\d])بسته بندی(?![\w\d])|(?<![\w\d])بسته(?![\w\d])|(?<![\w\d])پک(?![\w\d])|مجموعه|بسته‌ای|بسته ای|بسته|عددی|تایی|(?<![\w\d])تیکه ای(?![\w\d])|(?<![\w\d])جدید(?![\w\d])|یکاسه|قوطی|(?<![\w\d])اصلی(?![\w\d])|(?<![\w\d])اصل(?![\w\d])|طبیعی|[0-9]|[۰۱۲۳۴۵۶۷۸۹]|\([^)]*\)'
REGEX_PRUNE2 = r' {2,}'


def prune(text):
    text = str(text)
    text = text.replace('#', ' ').strip()
    _text = text
    text = re.sub(r'\([^()]*\)', '', text)
    if not text:
        text = re.sub(r'\([^()]*\)', '', _text[1:-1])
    text = re.sub(REGEX_PRUNE1, ' ', text)
    text = re.sub(REGEX_PRUNE2, ' ', text)
    return text.strip()


def get_chunked(iterable, chunk_size, _type):
    return [iterable[x:x + chunk_size] for x in range(0, len(iterable), chunk_size)] if _type == list else [
        dict(itertools.islice(iterable.items(), x, x + chunk_size)) for x in range(0, len(iterable), chunk_size)]


def clean_text(text):
    text = text.replace('_', ' ')
    text = text.replace('#', '')
    # Include the Persian comma in the pattern to be removed
    pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u200c\u200d0-9a-zA-Z\s]+|،')

    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = cleaned_text.replace('\u200c', ' ')

    return cleaned_text


def remove_non_character(text):
    text = re.sub(r'[\n\r]', ' ', text)

    # Step 2: Remove the long line of underscores (used as a separator)
    text = re.sub(r'_+', ' ', text)

    # Step 3: Remove the asterisks (used for bold formatting)
    text = re.sub(r'\*\*', '', text)

    # Remove any extra spaces created by the replacements
    text = re.sub(r'\s+', ' ', text).strip()

    return text

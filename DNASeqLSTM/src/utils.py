import json
import re
import string
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F

# text-preprocessing


def lower(text):
    return text.lower()


def remove_hashtags(text):
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_urls(text):
    clean_text = re.sub(r'(https|http):\/\/[A-Za-z0-9_\.]+(?:\/)?[A-Za-z0-9_\.]+[\r\n]*', '', text, flags=re.MULTILINE)
    return clean_text

def striphtml(text):
    p = re.compile(r'<.*?>')
    return p.sub(' ', text)

def remove_punctuation(text):
    exclude = set(string.punctuation)
    punc_free = ''.join(ch if ch not in exclude else ' ' for ch in text)
    return punc_free

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    ADD_TAGS_RE = re.compile('id|vk|com|video|wifi')
    DIGIT_SYMBOLS_RE = re.compile('\d+')

    text = DIGIT_SYMBOLS_RE.sub('', text)
    text = ADD_TAGS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if len(word)>3)
    return text

preprocessing_setps = {
    'remove_hashtags': remove_hashtags,
    'remove_urls': remove_urls,
    'remove_user_mentions': remove_user_mentions,
    'remove_html': striphtml,
    'text_prepare': text_prepare,
    'remove_punctuation': remove_punctuation,
    'lower': lower
}

def normalize_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)
    return text

def process_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)

    processed_text = ""
    for word in text.split():
        processed_text += ' '.join(list(word)) + " "
    return processed_text

# metrics // model evaluations


def get_evaluation(labels: torch.Tensor, y_prob: torch.Tensor, list_metrics: list) -> dict:

    y_true = labels.cpu().numpy()
    probabilities = F.softmax(y_prob, dim=1)
    proba = probabilities.cpu().detach().numpy()
    y_pred = np.argmax(proba, -1)

    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred, average='micro')

    return output


# preprocess input for prediction

def preprocess_input(args):
    collection=[]
    corpora = args.text
    steps = args.steps
    for text in corpora:
        if steps != None:
            for step in steps:
                raw_text = preprocessing_setps[step](text)
        else:
            raw_text = text

        number_of_characters = args.number_of_characters + len(args.extra_characters)
        identity_mat = np.identity(number_of_characters)
        vocabulary = list(args.alphabet) + list(args.extra_characters)
        max_length = args.max_length

        processed_output = np.array([identity_mat[vocabulary.index(i)] for i in list(
            raw_text) if i in vocabulary], dtype=np.float32)
        if len(processed_output) > max_length:
            processed_output = processed_output[:max_length]
        elif 0 < len(processed_output) < max_length:
            processed_output = np.concatenate((processed_output, np.zeros(
                (max_length - len(processed_output), number_of_characters), dtype=np.float32)))
        elif len(processed_output) == 0:
            processed_output = np.zeros(
                (max_length, number_of_characters), dtype=np.float32)
        collection.append(processed_output)
    return collection

import jsonlines
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

OPTS = None

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, type=str)
    args = parser.parse_args()

    with jsonlines.open(args.input ,'r') as f:
        data = [line for line in f.iter()]

    em = 0.
    f1 = 0.
    count = 0.
    key = 'answer:'
    wrong_type = 0.

    for i, line in enumerate(data):
        # if i % 3 != 0:
        #     continue

        if i % 1000 == 0:
            print(line["tgt"].replace(key, '').lstrip())
            print(line["gen"].replace(key, '').lstrip())

        if key not in line["tgt"]:
            continue

        count += 1

        if key in line["gen"]:
            em += compute_exact(line["tgt"].replace(key, '').lstrip(), line["gen"].replace(key, '').lstrip())
            f1 += compute_f1(line["tgt"].replace(key, '').lstrip(), line["gen"].replace(key, '').lstrip())
        else:
            em += 0.
            f1 += 0.
            wrong_type += 1

    print(f"EM accuracy is: {em / count}")

    print(f"F1 is: {f1 / count}")

    print(f"Wrong generated type: {wrong_type / count}")

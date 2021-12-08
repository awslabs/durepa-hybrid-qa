import torch
from transformers.data.processors.utils import DataProcessor
from torch.utils.data import IterableDataset, Dataset, TensorDataset, DataLoader
from dataclasses import dataclass
import jsonlines
import os
from typing import List, Optional, Union
import pickle
from tqdm import tqdm
import pdb
import copy
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
import multiprocessing as mp
from multiprocessing import Pool
import random
from itertools import cycle

import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: str
    label: Optional[int] = None


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label: Optional[int] = None


class RerankingProcessor(DataProcessor):
    """Processor for the reranking task."""

    def _read_jsonl(self, data_dir, file_name):
        print(f"loading {file_name}")
        lines = []
        with jsonlines.open(os.path.join(data_dir, file_name)) as f:
            for line in f.iter():
                lines.append(line)
        return lines

    def get_data_examples(self, data_dir, num_cand, mode, question_type='wikisql_denotation'):
        """See base class."""
        return self._create_examples(self._parse_data(self._read_jsonl(data_dir, f"reranking/{question_type}.{mode}.es_retrieved.{num_cand}_cands.both_index.jsonl"), num_cand), f"{mode}", num_cand)

    def get_inference_examples(self, data_dir, mode, question_type='wikisql_denotation'):
        return self._create_inference_examples(self._parse_inference_data(self._read_jsonl(data_dir, f"{question_type}.dev.es_retrieved.processed.jsonl")), f"{mode}")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _parse_data(self, all_data, num_cand, index='both'):
        outputs = []

        for data in tqdm(all_data, total=len(all_data), desc="converting aes candidates"):
            # process one question

            if 'candidates' in data:
                # data has been processed
                logger.info("Data has been processed")
                outputs = all_data
                break

            if index == 'both':
                all_aes = data['passages'] + data['tables']
            elif index in ['passages', 'tables']:
                all_aes = data[index]
            else:
                raise NotImplementedError()

            if len(all_aes) < num_cand:
                logger.info("you do not have enough aes retrievals")
                continue

            # get indices for positive candidates
            if index == 'both':
                pos_indices = data['pos_index_passages'] + [int(x)+len(data['passages']) for x in data['pos_index_tables']]
            elif index == 'passages':
                pos_indices = data['pos_index_passages']
            else:
                pos_indices = data['pos_index_tables']
            random.shuffle(pos_indices)
            if len(pos_indices) == 0:
                continue

            # probability of sampling negative candidates
            prob = [0. if i in pos_indices else 1./(len(all_aes)-len(pos_indices)) for i in range(len(all_aes))]

            num_pos = 1
            for pos_idx in pos_indices[:num_pos]:
                # one example
                try:
                    neg_indices = list(np.random.choice(len(all_aes), num_cand-1, replace=False, p=prob))
                except:
                    logger.info(f"question: {data['question']}")
                    logger.info(f"answer: {data['answers']}") if 'answers' in data else logger.info(f"answer: {data['denotation']}")
                    break
                all_indices = [pos_idx] + neg_indices
                outputs.append(
                    {'qid': data['qid'],
                     'question': data['question'],
                     'candidates': [all_aes[i] for i in all_indices],
                    }
                )

        return outputs

    def _parse_inference_data(self, all_data):
        outputs = []

        for data in tqdm(all_data, total=len(all_data), desc="converting aes candidates"):
            outputs.append(
                {'qid': data['qid'],
                 'question': data['question'],
                 'candidates': data['passages'][:100] + data['tables'][:100],
                }
            )

        return outputs

    def _create_examples(self, all_data, set_type, num_cand):
        """Creates examples for the training and dev sets. In this method, text_b includes all the candidates."""

        examples = []

        for idx_i, data in tqdm(enumerate(all_data), total=len(all_data), desc="Creating examples"):

            batch_examples = []
            for idx_j, line in enumerate(data['candidates']):

                guid = "%s-%s" % (set_type, f"example_{idx_i}_index_{idx_j}") # example_1_index_7
                text_a = data['question']
                text_b = line['article_title'] + '[title]' + line['text']
                label = line['judge']['judge_contain_some'] if isinstance(line['judge'], dict) else line['judge']
                if idx_j == 0:
                    if label != 1:
                        pdb.set_trace()
                    assert label == 1
                else:
                    if label != 0:
                        pdb.set_trace()
                    assert label == 0
                batch_examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=0))

            assert len(batch_examples) == num_cand, f"you need {num_cand} candidates, but you have {len(batch_examples)}"

            examples.append(batch_examples.copy())

        return examples

    def _create_inference_examples(self, all_data, set_type):

        examples = []

        for idx_i, data in tqdm(enumerate(all_data), total=len(all_data), desc="Creating examples"):
            for idx_j, line in enumerate(data['candidates']):
                guid = "%s-%s" % (set_type, f"example_{idx_i}_index_{idx_j}") # example_1_index_7
                text_a = data['question']
                text_b = line['article_title'] + '[title]' + line['text']

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b))

        return examples

def convert_examples_to_features(
    all_examples,
    tokenizer,
    max_length,
    num_cand,
):
    """
    Loads a data file into a list of ``InputFeatures``
    """
    global process_one # TODO: this is a bit hacky

    features = []

    def process_one(examples):

        input_ids, token_type_ids, attention_mask = [], [], []

        assert len(examples) == num_cand

        for (ex_index, example) in enumerate(examples):

            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation='only_second')

            if example.guid.split('_')[1] == '0' and ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in inputs["input_ids"]]))
                logger.info("input_tokens: %s" % " ".join([str(tokenizer.convert_ids_to_tokens(x)) for x in inputs["input_ids"]]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in inputs["attention_mask"]]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in inputs["token_type_ids"]]))

            input_ids.extend(inputs["input_ids"])
            token_type_ids.extend(inputs["token_type_ids"])
            attention_mask.extend(inputs["attention_mask"])

        assert len(input_ids) == max_length * num_cand, f"actual length {len(input_ids)}; required {max_length * num_cand}"
        assert len(token_type_ids) == max_length * num_cand
        assert len(attention_mask) == max_length * num_cand

        return InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=0
            )

    n_cpus= max(1, mp.cpu_count()-2)
    logger.info(f"Using {n_cpus} cpus")
    with Pool(processes=n_cpus) as pool:
        for feature in tqdm(pool.imap_unordered(process_one, all_examples), total=len(all_examples), desc="converting examples to features"):
            features.append(
                feature
            )

    return features


def convert_examples_to_features_inference(
    all_examples,
    tokenizer,
    max_length,
):
    """
    Loads a data file into a list of ``InputFeatures``
    """
    global process_one # TODO: this is a bit hacky

    features = []

    def process_one(examples):

        input_ids, token_type_ids, attention_mask = [], [], []
        output = []

        for (ex_index, example) in enumerate(examples):

            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation='only_second')

            if example.guid.split('_')[1] == '0' and ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in inputs["input_ids"]]))
                logger.info("input_tokens: %s" % " ".join([str(tokenizer.convert_ids_to_tokens(x)) for x in inputs["input_ids"]]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in inputs["attention_mask"]]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in inputs["token_type_ids"]]))

            output.append(
                InputFeatures(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    label=0
                )
            )

        return output

    n_cpus= max(1, int(mp.cpu_count()-2))
    logger.info(f"Using {n_cpus} cpus")
    split_len = 64
    batched_examples = []
    start = 0
    while start < len(all_examples):
        batched_examples.append(all_examples[start:start+split_len])
        start += split_len

    with Pool(processes=n_cpus) as pool:
        for feature in tqdm(pool.imap(process_one, batched_examples), total=len(batched_examples), desc="converting examples to features"):
            features.extend(
                feature
            )

    return features


class RankDataset(Dataset):
    """
    Create dataset.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        num_cand: int,
        mode: str,
        question_type: str,
        overwrite_cache=False,
    ):
        # Load data features from cache or dataset file
        cache_dir = os.path.join(data_dir, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cached_features_file = os.path.join(
            cache_dir, "cached_{}_{}_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length), str(num_cand), question_type),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            # self.features = torch.load(cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
            logger.info(f"Total number of data: {len(self.features)}")
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            processor = RerankingProcessor()
            if mode == 'test':
                examples = processor.get_inference_examples(data_dir, 'test', question_type)
                self.features = convert_examples_to_features_inference(
                    examples,
                    tokenizer,
                    max_seq_length,
                )
            else:
                examples = processor.get_data_examples(data_dir, num_cand, mode, question_type)
                self.features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_seq_length,
                    num_cand,
                )
            logger.info(f"Total number of data: {len(self.features)}")
            logger.info(f"Saving features into cached file {cached_features_file}")
            # torch.save(self.features, cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class IterableRankDataset(IterableDataset):
    """ For large file."""

    def __init__(self, data_dir, tokenizer, mode, num_cand, max_length, question_type='opensquad', index='both'):
        self.filepath = os.path.join(data_dir, f"{question_type}.{mode}.es_retrieved.{num_cand}_cands.{index}_index.jsonl")
        self.set_type = mode
        self.num_cand = num_cand
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def _create_examples(self):
        """Creates examples for the training and dev sets. In this method, text_b includes all the candidates."""

        with jsonlines.open(self.filepath, 'r') as reader:

            for idx_i, data in enumerate(reader.iter()):

                input_ids, token_type_ids, attention_mask = [], [], []

                for idx_j, line in enumerate(data['candidates']):

                    guid = "%s-%s" % (self.mode, f"example_{idx_i}_index_{idx_j}") # example_1_index_7
                    text_a = data['question']
                    text_b = line['article_title'] + '[title]' + line['text']
                    label = line['judge']['judge_contain_some'] if type(line['judge']) == dict else line['judge']
                    if idx_j == 0:
                        assert label == 1
                    else:
                        assert label == 0

                    inputs = self.tokenizer.encode_plus(
                        text_a,
                        text_b,
                        add_special_tokens=True,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation='only_second')

                    input_ids.extend(inputs["input_ids"])
                    token_type_ids.extend(inputs["token_type_ids"])
                    attention_mask.extend(inputs["attention_mask"])

                assert len(input_ids) == self.max_length * self.num_cand, f"actual length {len(input_ids)}; required {max_length * num_cand}"
                assert len(token_type_ids) == self.max_length * self.num_cand
                assert len(attention_mask) == self.max_length * self.num_cand

                yield InputFeatures(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        label=0
                        )

    def _get_stream(self):
        return cycle(self._create_examples())

    def __iter__(self):
        return self._get_stream()


def generate_dataloader(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    mode: str,
    num_cand: int,
    overwrite_cache=False,
    batch_size: int=8,
    question_type: str='opensquad',
) -> DataLoader:

    def collate_fn(features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        labels = torch.tensor([f.label for f in features], dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, labels

    if mode == '':
        # File is too large, we will use IterableDataset
        dataset = IterableRankDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            mode=mode,
            num_cand=num_cand,
            max_length=max_seq_length,
            question_type=question_type,
            )
    else:
        dataset = RankDataset(data_dir=data_dir,
                             tokenizer=tokenizer,
                             max_seq_length=max_seq_length,
                             num_cand=num_cand,
                             overwrite_cache=overwrite_cache,
                             mode=mode,
                             question_type=question_type)
    # pdb.set_trace()
    try:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                shuffle=True if mode=="train" else False)
    except:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                shuffle=False)

    return dataloader


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    additional_special_tokens_dict = {'additional_special_tokens': ['[title]']}
    tokenizer.add_special_tokens(additional_special_tokens_dict) # add classification tokens

    dataloader = generate_dataloader('/home/ec2-user/efs/hybridQA/squad', tokenizer, 128, 'test', 64, False, 1)

    for data in dataloader:
        break

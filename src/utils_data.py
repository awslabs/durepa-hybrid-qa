from transformers.data.processors.glue import DataProcessor
from transformers import T5Tokenizer, PreTrainedTokenizer
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import jsonlines
import os
from typing import List, Optional
from tqdm import tqdm
import logging
import multiprocessing as mp
from multiprocessing import Pool

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

import pdb


@dataclass
class InputExamples:
    guid: str
    source: str
    target: str


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    decoder_input_ids: Optional[List[int]] = None
    decoder_attention_mask: Optional[List[int]] = None
    labels: Optional[List[int]] = None


class T5Processor(DataProcessor):

    def _read_jsonl(self, input_file):
        with jsonlines.open(input_file, 'r') as f:
            data = [x for x in f.iter()]
        return data

    def get_data_examples(self, data_dir, mode):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, f"{mode}.jsonl")), f"{mode}")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), total=len(lines)):
            guid = "%s-%s" % (set_type, i)
            src = line["src"] #+ " </s>"
            tgt = line["tgt"] #+ " </s>"
            examples.append(InputExamples(guid=guid, source=src, target=tgt))
        return examples


def convert_examples_to_features(
    examples: List[InputExamples],
    max_source_length: int,
    max_target_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:

    global process_one # TODO: this is a bit hacky

    features = []

    def process_one(example):

        inputs = tokenizer.encode_plus(
            example.source,
            max_length=max_source_length,
            add_special_tokens=True,
            padding='max_length',
            truncation='longest_first',
            # return_tensors="pt"
        )
        labels = tokenizer.encode(
            example.target,
            max_length=max_target_length,
            add_special_tokens=True,
            padding='max_length',
            truncation='longest_first',
            # return_tensors="pt"
        )

        assert len(inputs['input_ids']) == max_source_length
        assert len(labels) == max_target_length

        if int(example.guid.split('-')[-1]) < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in inputs["input_ids"]]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in inputs["attention_mask"]]))
            logger.info("input_tokens: %s" % tokenizer.decode(inputs['input_ids']))
            logger.info("labels: %s" % tokenizer.decode(labels))

        labels = [x if x > 0 else -100 for x in labels]

        return InputFeatures(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels,
            )

    n_cpus= int(max(1, mp.cpu_count()/2))
    logger.info(f"Using {n_cpus} cpus")
    with Pool(processes=n_cpus) as pool:
        for feature in tqdm(pool.imap(process_one, examples), total=len(examples), desc="converting examples to features"):
            features.append(
                feature
            )

    return features


class T5Dataset(Dataset):

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int,
        max_target_length: int,
        mode: str,
        overwrite_cache=False,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_source_length), str(max_target_length)),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            processor = T5Processor()
            examples = processor.get_data_examples(data_dir, mode)

            self.features = convert_examples_to_features(
                examples,
                max_source_length,
                max_target_length,
                tokenizer,
            )
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def generate_dataloader(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_source_length: int,
    max_target_length: int,
    mode: str,
    overwrite_cache=False,
    batch_size: int=8,
) -> DataLoader:

    def collate_fn(features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        labels = torch.tensor([f.labels for f in features], dtype=torch.long)
        return input_ids, attention_mask, labels

    dataset = T5Dataset(data_dir=data_dir,
                         tokenizer=tokenizer,
                         max_source_length=max_source_length,
                         max_target_length=max_target_length,
                         overwrite_cache=overwrite_cache,
                         mode=mode)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            num_workers=mp.cpu_count(),
                            shuffle=True if mode=="train" else False)

    return dataloader


if __name__ == "__main__":
    # processor = SnliProcessor()
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # examples = processor.get_data_examples('/home/ec2-user/efs/glue_data/SNLI', 'dev')
    # features = convert_examples_to_features(examples, 100, 100, tokenizer)

    dataset = T5Dataset(data_dir="/home/ec2-user/efs/nel_data/zeshel/zeshel/",
                          tokenizer=tokenizer,
                          max_source_length=110,
                          max_target_length=128,
                          overwrite_cache=True,
                          mode='dev')

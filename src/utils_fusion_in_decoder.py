from transformers.data.processors.glue import DataProcessor
from transformers import T5Tokenizer, PreTrainedTokenizer
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import jsonlines
import os
import random
from typing import List, Optional
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

import pdb

@dataclass
class InputExamples:
    guid: str
    source: List[str]
    target: str


@dataclass
class InputFeatures:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: Optional[List[int]] = None


# cand_for_each_source = 50

# prefix_to_folder = {
#     'wikisql_denotation': 'regular',
#     'opensquad': 'regular',
#     'NQ-open': 'open-nq',
#     'ott-qa': 'ott-qa',
# }


class T5Processor(DataProcessor):

    def _read_jsonl(self, input_file):
        with jsonlines.open(input_file, 'r') as f:
            data = [x for x in f.iter()]
        return data

    def get_data_examples(self, data_dir, mode, question_type, passage_type, enable_sql_supervision, cand_for_each_source):
        """See base class."""
        if question_type == 'wikisql_question':
            prefix = ['wikisql_denotation']
        elif question_type == 'opensquad_question':
            prefix = ['opensquad']
        elif question_type == 'mixed':
            prefix = ['wikisql_denotation', 'opensquad']
        elif question_type == 'nq':
            prefix = ['NQ-open']
        elif question_type == 'nq_wikisql':
            prefix = ['NQ-open', 'wikisql_denotation']
        elif question_type == 'ott-qa':
            prefix = ['ott-qa']
        elif question_type == 'ottqa_wikisql':
            prefix = ['ott-qa', 'wikisql_denotation']
        elif question_type == 'all':
            prefix = ['ott-qa', 'opensquad', 'NQ-open', 'wikisql_denotation']
        else:
            raise NotImplementedError()

        data = []
        if mode == 'train':
            for q_type in prefix:
                if q_type == 'wikisql_denotation':
                    data.extend(self._read_jsonl(os.path.join(data_dir, f"{q_type}.train.es_retrieved.true_sql.jsonl")))
                else:
                    data.extend(self._read_jsonl(os.path.join(data_dir, f"{q_type}.train.es_retrieved.jsonl")))
            random.shuffle(data)
            return self._create_examples(data, mode, passage_type, enable_sql_supervision, cand_for_each_source)
        else:
            for q_type in prefix:
                if q_type == 'wikisql_denotation':
                    data.extend(self._read_jsonl(os.path.join(data_dir, f"scores", f"{q_type}.dev.es_retrieved.scores.sorted.true_sql.jsonl")))
                else:
                    data.extend(self._read_jsonl(os.path.join(data_dir, f"scores", f"{q_type}.dev.es_retrieved.processed.scores.sorted.jsonl")))
                    # data.extend(self._read_jsonl(os.path.join(data_dir, "nq.dev.wikigq_siamese_512.wiki2016.wikitable.jsonl")))
                    # data.extend(self._read_jsonl(os.path.join(data_dir, "scores", f"{q_type}.dev.es_retrieved.scores.sorted.jsonl")))
            return self._create_examples(data, mode, passage_type, enable_sql_supervision, cand_for_each_source)

    def _create_examples(self, lines, set_type, passage_type, enable_sql_supervision, cand_for_each_source):
        """Creates examples for the training and dev sets."""

        examples = []
        total_answerable = 0

        for (i, line) in tqdm(enumerate(lines), total=len(lines)):
            guid = "%s-%s-%s" % (set_type, line['qid'], str(i))
            src = []
            has_pos = 0

            if passage_type == 'both' and set_type != 'train':

                # rank by bert score in val and test
                score_key = 'rank_score'
                # score_key = "nr_score"
                both_sources = line["passages"][:2*cand_for_each_source] + line["tables"][:2*cand_for_each_source]
                all_scores = np.array([x[score_key] for x in both_sources])
                sorted_index = all_scores.argsort()[::-1][:2*cand_for_each_source]

                for idx in sorted_index:
                    pas = both_sources[idx]
                    if idx < 2 * cand_for_each_source:
                        pas_type = 'passage'
                        title = pas["article_title"]
                    else:
                        pas_type = 'table'
                        title = "table_" + pas["uid"].split('-split')[0]

                    src.append("question: " + line["question"].strip() + f" </s> {pas_type} title: " + title + f" </s> {pas_type} content: " + pas["text"] + " </s>")

                    if isinstance(pas['judge'], dict):
                        has_pos += int(pas['judge']['judge_contain_all'])
                    else:
                        has_pos += int(pas['judge'])

            else:

                if passage_type in ['textual', 'hybrid', 'both']:
                    for pas in line["passages"][:cand_for_each_source]:
                        # 10 passages
                        src.append("question: " + line["question"].strip() + " </s> passage title: " + pas["article_title"] + " </s> passage content: " + pas["text"] + " </s>") # e.g. question: Tell me what the notes are for South Australia </s> passage title: Strictly Commercial </s> passage content: album \"ZAPPAtite\".  All songs written and performed ... </s>
                        if isinstance(pas['judge'], dict):
                            has_pos += int(pas['judge']['judge_contain_all'])
                        else:
                            has_pos += int(pas['judge'])

                if passage_type in ['tabular', 'hybrid', 'both']:
                    for tab in line["tables"][:cand_for_each_source]:
                        # 10 tables
                        src.append("question: " + line["question"].strip() + " </s> table title: "   + "table_" + tab["uid"].split('-split')[0] + " </s> table content: "   + tab["text"] + " </s>") # e.g. question: Tell me what the notes are for South Australia </s> table title: From Nashville to Memphis: The Essential '60s Masters ; Disc Two ; Disc Tw </s> table content: ... </s>
                        if isinstance(tab['judge'], dict):
                            has_pos += int(tab['judge']['judge_contain_all'])
                        else:
                            has_pos += int(tab['judge'])

            const = 2 if passage_type in ['hybrid', 'both'] else 1
            if len(src) != const * cand_for_each_source:
                logger.info(line)
                src = src + [""] * (const * cand_for_each_source - len(src))

            if has_pos > 0:
                total_answerable += 1

            if enable_sql_supervision and 'true_sql' in line:
                tgt = "sql: " + line["true_sql"] + " </s>" # e.g. sql: SELECT Position FROM table_1-10015132-11 WHERE School/Club Team = \"Butler CC (KS)\" </s>
                examples.append(InputExamples(guid=guid, source=src, target=tgt))
                if i % 1000 == 0:
                    logger.info(src[0] + " " + tgt)
            if 'denotation' in line:
                tgt = "answer: " + str(line["denotation"][0]) + " </s>"
                examples.append(InputExamples(guid=guid, source=src, target=tgt))
                if i % 1000 == 0:
                    logger.info(src[0] + " " + tgt)
            if 'answers' in line:
                tgt = "answer: " + str(line["answers"][0]) + " </s>" # e.g. answer: no slogan on current series </s>
                examples.append(InputExamples(guid=guid, source=src, target=tgt))
                if i % 1000 == 0:
                    logger.info(src[0] + " " + tgt + "\n")

        logger.info(f"Total answerable in {set_type} split is {total_answerable / len(lines)}")

        return examples

def convert_examples_to_features(
    examples: List[InputExamples],
    max_source_length: int,
    max_target_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:

    global process_one

    features = []

    def process_one(examples):

        output = []

        for (ex_index, example) in enumerate(examples):

            inputs = tokenizer.batch_encode_plus(
                example.source, # source is a list of str
                max_length=max_source_length,
                add_special_tokens=True,
                padding='max_length',
                truncation='longest_first',
                # return_tensors="pt"
            )

            labels = tokenizer.encode(
                example.target,
                max_length=max_source_length,
                add_special_tokens=True,
                padding='max_length',
                truncation='longest_first',
                # return_tensors="pt"
            )

            if int(example.guid.split('-')[-1]) < 10:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in inputs["input_ids"][0]]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in inputs["attention_mask"][0]]))
                logger.info("input_tokens: %s" % tokenizer.decode(inputs["input_ids"][0]))
                logger.info("labels: %s" % tokenizer.decode(labels))

            for input in inputs['input_ids']:
                assert len(input) == max_source_length

            labels = [x if x > 0 else -100 for x in labels]

            output.append(
                InputFeatures(
                    input_ids=inputs['input_ids'], # list of lists
                    attention_mask=inputs['attention_mask'], # list of lists
                    labels=labels, # list
                )
            )

        return output

    n_cpus= max(1, int(mp.cpu_count()/1.1))
    logger.info(f"Using {n_cpus} cpus")
    split_len = 100
    batched_examples = []
    start = 0
    while start < len(examples):
        batched_examples.append(examples[start:start+split_len])
        start += split_len

    with Pool(processes=n_cpus) as pool:
        for feature in tqdm(pool.imap(process_one, batched_examples), total=len(batched_examples), desc="converting examples to features"):
            features.extend(
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
        question_type='wikisql_question',
        passage_type='textual',
        enable_sql_supervision=False,
        cand_for_each_source=50,
    ):
        # Load data features from cache or dataset file
        cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_features_file = os.path.join(
            cache_dir, "cached_{}_{}_{}_{}_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_source_length), str(max_target_length), str(2*cand_for_each_source), question_type, passage_type),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            processor = T5Processor()
            examples = processor.get_data_examples(data_dir, mode, question_type, passage_type, enable_sql_supervision, cand_for_each_source)

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
    enable_sql_supervision: bool,
    cand_for_each_source: int,
    overwrite_cache=False,
    batch_size: int=8,
    question_type: str='wikisql_question',
    passage_type: str='textual',
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
                         mode=mode,
                         question_type=question_type,
                         passage_type=passage_type,
                         enable_sql_supervision=enable_sql_supervision,
                         cand_for_each_source=cand_for_each_source,
                        )

    n_cpus= max(1, int(mp.cpu_count()/2))

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            num_workers=n_cpus,
                            shuffle=True if mode=="train" else False)

    return dataloader

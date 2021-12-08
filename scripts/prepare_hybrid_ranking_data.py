import jsonlines
from tqdm import tqdm
import os
import random
import multiprocessing as mp
from multiprocessing import Pool
import pdb
import numpy as np
import argparse

import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def read_one(line):
    return line


def read_jsonlines(file_path):
    n_cpus= max(1, mp.cpu_count()-2)
    logger.info(f"Using {n_cpus} CPUs")
    reader = jsonlines.open(file_path, 'r')
    data = []
    with Pool(processes=n_cpus) as pool:
        for line in tqdm(pool.imap(read_one, reader), desc='reading jsonlines'):
            data.append(line)
    return data


def save_jsonlines(lines, file_path):
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(lines)


def parse_data(all_data, num_cand, index='both'):
    outputs = []

    for data in tqdm(all_data, total=len(all_data), desc="converting aes candidates"):
        # process one question
        all_aes = data['passages'] + data['tables']

        if len(all_aes) < num_cand:
            logger.info("you do not have enough aes retrievals")
            continue

        pos_indices = data['pos_index_passages'] + [int(x)+len(data['passages']) for x in data['pos_index_tables']]
        random.shuffle(pos_indices)
        if len(pos_indices) == 0:
            continue
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


def _determine_judge(doc, judge_key):
    if judge_key is None:
        # vanilla mode
        return doc['judge']
    else:
        return doc['judge'][judge_key]


if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    argp.add_argument('--data_dir', default='/home/ec2-user/efs/hybridQA/NQ/hystruct_es_retrieval-nqopen-wikitext-wikitable-20200929T195611-Lug', type=str)
    argp.add_argument('--num_cand', default=64)
    argp.add_argument('--index', default='both')
    argp.add_argument('--question_type', default='NQ-open', choices=['opensquad', 'wikisql_denotation', 'NQ-open', 'ott-qa'])
    args = argp.parse_args()

    if args.question_type.lower() in ['opensquad', 'nq-open', 'ott-qa']:
        judge_key = None
    elif args.question_type.lower() == 'wikisql_denotation':
        judge_key = "judge_contain_some"
    else:
        raise NotImplementedError()

    for mode in ['dev', 'train', 'test']:
        os.makedirs(os.path.join(args.data_dir, 'reranking'), exist_ok=True)

        if os.path.exists(os.path.join(args.data_dir, f'reranking/{args.question_type}.{mode}.es_retrieved.{args.num_cand}_cands.{args.index}_index.jsonl')):
            continue

        print(f"loading {mode} data")

        if os.path.exists(os.path.join(args.data_dir, f'{args.question_type}.{mode}.es_retrieved.processed.jsonl')):
            all_data = read_jsonlines(os.path.join(args.data_dir, f'{args.question_type}.{mode}.es_retrieved.processed.jsonl'))
        else:
            all_data = read_jsonlines(os.path.join(args.data_dir, f'{args.question_type}.{mode}.es_retrieved.jsonl'))
            for data in tqdm(list(all_data), total=len(all_data)):
                passages = data['passages']
                tables = data['tables']

                # passages
                pos_index = []
                for i, pas in enumerate(passages):
                    if _determine_judge(pas, judge_key) == 1:
                        pos_index.append(i)
                data['pos_index_passages'] = list(pos_index)

                # tables
                pos_index = []
                for i, tab in enumerate(tables):
                    if _determine_judge(tab, judge_key) == 1:
                        pos_index.append(i)
                data['pos_index_tables'] = list(pos_index)

            save_jsonlines(all_data, os.path.join(args.data_dir, f'{args.question_type}.{mode}.es_retrieved.processed.jsonl'))

        final_results = parse_data(all_data, args.num_cand, args.index)

        save_jsonlines(final_results, os.path.join(args.data_dir, f'reranking/{args.question_type}.{mode}.es_retrieved.{args.num_cand}_cands.{args.index}_index.jsonl'))

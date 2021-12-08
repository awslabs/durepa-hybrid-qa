import glob
import argparse
import json
import sys
import jsonlines
from tqdm import tqdm
import numpy as np


def _determine_judge(doc, judge_key):
    if judge_key is None:
        # vanilla mode
        return doc['judge']
    else:
        return doc['judge'][judge_key]


def sort_scores(data, sort_on='both', rank_ulimit=200):
    # sort_index: ['both', 'passages', 'tables']
    for line in tqdm(data, total=len(data), desc='sorting'):
        line['passages'] = line['passages'][:rank_ulimit]
        line['tables'] = line['tables'][:rank_ulimit]
        rank_scores = []
        if sort_on == 'both':
            line['both'] = line['passages'] + line['tables']
        for cand in line[sort_on]:
            rank_scores.append(cand['rank_score'])
        sort_index = np.array(rank_scores).argsort()[::-1] # descending
        passages_copy = line[sort_on].copy()
        line[sort_on] = [passages_copy[i] for i in sort_index]


def evaluation(in_prefix, *, eval_type, rank_ulimit):
    k_values = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    total = 0
    correct = {}
    for k in k_values:
        correct[k] = 0
    map_result = 0.0
    mrr_result = 0.0

    judge_key = 'NOT_INTIALIZED'

    for inpath in sorted(glob.glob(in_prefix + "*")):
        print(inpath)
        infile = open(inpath, 'rt', encoding='utf-8')
        for i, line in enumerate(infile):
            line = line.strip()
            question = json.loads(line)

            if i == 0:
                if isinstance(question['passages'][0]['judge'], dict):
                    judge_key = 'judge_contain_some'
                else:
                    judge_key = None

            total += 1
            if len(question[eval_type]) ==0:
                print("NONE")
                continue
            rank_list = [("", _determine_judge(doc, judge_key)) for doc in question[eval_type]]
            rank_list = rank_list[:rank_ulimit]
            # calculate recall
            for k in k_values:
                if k > rank_ulimit:
                    break
                for j in range(k):
                    if j >= len(rank_list): break
                    if rank_list[j][1] == 1:
                        correct[k] += 1
                        break
            # calculate MAP and MRR
            cur_map = 0.0
            cur_map_total = 0.0
            cur_mrr = 0.0
            scores = []
            for j in range(len(rank_list)):
                cur_dist, cur_label = rank_list[j]
                scores.append(cur_dist)
                if cur_label == 1:
                    cur_map_total += 1
                    cur_map += cur_map_total / (1 + j)
                    if cur_mrr == 0.0:
                        cur_mrr = 1 / float(1 + j)

            if cur_map_total != 0.0: cur_map = cur_map / cur_map_total
            map_result += cur_map
            mrr_result += cur_mrr

    # output accuracy
    out_report = ''
    for k in k_values:
        cur_accuracy = correct[k] / float(total) * 100
        out_report += 'Top-%d:\t%.2f\n' % (k, cur_accuracy)
    map_result = map_result / float(total) * 100
    mrr_result = mrr_result / float(total) * 100
    out_report += 'MAP:\t%.2f\n' % (map_result)
    out_report += 'MRR:\t%.2f\n' % (mrr_result)
    out_report += 'Total:\t%d\n' % (total)
    out_report += 'Size of rank list:\t%d\n' % (rank_ulimit)
    acc = correct[1] / float(total)

    print(f'eval_type: {eval_type}')
    print(f'judge_key: {judge_key}')
    print(out_report)
    print()

    return acc


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--input', help='', required=True)
    argp.add_argument('--eval_rank', action='store_true', help='whether to evaluate ranked results')
    argp.add_argument('--rank_ulimit', help='upper limit on count of rank_list', default=200, type=int)
    args = argp.parse_args()

    input_file = args.input

    if args.eval_rank:
        with jsonlines.open(args.input) as f:
            lines = [line for line in f.iter()]

        sort_scores(lines, 'both')
        sort_scores(lines, 'passages')
        sort_scores(lines, 'tables')

        input_file = args.input.replace('jsonl', 'sorted.jsonl')
        with jsonlines.open(input_file, 'w') as writer:
            writer.write_all(lines)

    all_keys = ['passages', 'tables', 'both'] if args.eval_rank else ['passages', 'tables']
    for dict_key in all_keys:
        evaluation(input_file, eval_type=dict_key, rank_ulimit=args.rank_ulimit)

    print('DONE!')

import jsonlines
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
from tqdm import tqdm
import random

import glob
import pdb
from lib.dbengine import DBEngine, DBEngineSqlite3
from lib.query import Query
from lib.common import count_lines

import multiprocessing as mp
from multiprocessing import Pool


agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']


tables = []
table_id_owner = {}

# wikisql tables
for file in glob.iglob("/home/ec2-user/efs/nel_data/WikiSQL/data/*.tables.jsonl"):
    with jsonlines.open(file) as f:
        curr_table = [l for l in f.iter()]
        tables.extend(curr_table)
        all_ids = list(set([x['id'] for x in curr_table]))
        split = re.search(r'/home/ec2-user/efs/nel_data/WikiSQL/data/(.*?)\.tables\.jsonl', file).group(1)
        for id in all_ids:
            table_id_owner[id] = split + '.db'
tables = {x["id"]: x for x in tables}

# wikitables
# with jsonlines.open('/home/ec2-user/efs/hybridQA/wikitables_sqlite/wikitables.tables.jsonl') as f:
#     for line in f.iter():
#         tables[line['id']] = line
# wikitable_split = json.load(open('/home/ec2-user/efs/hybridQA/wikitables_sqlite/table_id_db_dict.json'))
# new_wikitable_split = {}
# for key in wikitable_split:
#     value = wikitable_split[key]
#     key = key.replace('_', '-').replace('table-', '')
#     new_wikitable_split[key] = value
# table_id_owner = {**table_id_owner, **new_wikitable_split}


def parse_sql_str(sql):

    # e.g. SELECT Position FROM table_1-10015132-11 WHERE School/Club Team = "Butler CC (KS)"
    # e.g. SELECT COUNT(Position) FROM table_1-10015132-9 WHERE Years in Toronto = \"2006-07\"

    parsed_sql = {"phase": 1, "query": {}}

    # from table
    try:
        if "WHERE" in sql:
            from_table = re.search(r'FROM(.*?)WHERE', sql).group(1).strip().lower() # 'table_1-10015132-11'
        else:
            from_table = re.search(r'(?<=FROM).*$', sql).group(0).strip().lower()
    except:
        # print(sql)
        parsed_sql["error"] = "Generated SQL is not valid"
        return False, parsed_sql
    try:
        table_id = from_table.replace("table_", "") # '1-10015132-11'
        table = tables[table_id]
        header = [x.lower() for x in table["header"]]
        parsed_sql["table_id"] = table_id
    except:
        parsed_sql["error"] = "Generated SQL is not valid"
        return False, parsed_sql

    # select column
    sel_col = re.search(r'SELECT(.*?)FROM', sql).group(1).strip() # COUNT(Position)
    agg_idx = 0
    for idx, op in enumerate(agg_ops):
        if op in sel_col and op != '':
            sel_col = re.search(f'{op}\((.*?)\)', sel_col)
            if sel_col:
                sel_col = sel_col.group(1).strip()
                agg_idx = idx
            break

    try:
        sel = header.index(sel_col.lower()) # 3
        parsed_sql["query"]["sel"] = sel
        parsed_sql["query"]["agg"] = agg_idx
    except:
        parsed_sql["error"] = "Generated SQL is not valid"
        return False, parsed_sql

    # where clause
    conds = []
    if "WHERE" in sql:
        where = re.search(r'(?<=WHERE).*$', sql).group(0).strip() # 'Position = "Guard" AND Years in Toronto = "1996-97"'
        wheres = where.split("AND")
        for cond in wheres:
            cond = cond.strip() # 'Position = "Guard"'
            for idx, op in enumerate(cond_ops):
                if op in cond:
                    try:
                        col, value = cond.split(op, 1) # split by first occurance
                    except:
                        pdb.set_trace()
                    col = col.strip()
                    value = value.strip()
                    try:
                        sel = header.index(col.lower())
                        conds.append([sel, idx, json.loads(value)])
                    except:
                        pass
                    break

    parsed_sql["query"]["conds"] = conds # can be empty list
    parsed_sql["error"] = ""
    parsed_sql["split"] = table_id_owner[table_id]

    return True, parsed_sql


def execute_line(line):
    if "sql" not in line['gen_top1']:
        return line

    success, ep = parse_sql_str(line['gen_top1'].replace("sql:", "").lstrip())
    if success:
        if ep['split'].endswith('.sqlite'):
            engine = DBEngine(f"/home/ec2-user/efs/hybridQA/wikitables_sqlite/{ep['split']}")
        else:
            engine = DBEngine(f"/home/ec2-user/efs/nel_data/WikiSQL/data/{ep['split']}")
        try:
            qp = Query.from_dict(ep['query'], ordered=False)
            pred_str_list = engine.execute_query(ep['table_id'], qp, lower=True)
        except Exception as e:
            pred_str_list = [repr(e)]
    else:
        pred_str_list = [""]

    line['success'] = success
    line['exe_results'] = pred_str_list
    line['parsed_sql'] = ep

    return line


def execute_line_gth(line):
    success, ep = parse_sql_str(line['true_sql'].lstrip())
    if success:
        if ep['split'].endswith('.sqlite'):
            engine = DBEngine(f"/home/ec2-user/efs/hybridQA/wikitables_sqlite/{ep['split']}")
        else:
            engine = DBEngine(f"/home/ec2-user/efs/nel_data/WikiSQL/data/{ep['split']}")
        try:
            qp = Query.from_dict(ep['query'], ordered=False)
            pred_str_list = engine.execute_query(ep['table_id'], qp, lower=True)
        except Exception as e:
            pred_str_list = [repr(e)]
    else:
        pred_str_list = [""]

    line['success'] = success
    line['exe_results'] = pred_str_list
    line['parsed_sql'] = ep

    return line


def execute_lines(data):
    n_cpus= max(1, int(mp.cpu_count()-2))
    # n_cpus = 1
    print(f"Using {n_cpus} cpus")

    new_data = []
    with Pool(processes=n_cpus) as pool:
        for line in tqdm(pool.imap(execute_line, data), total=len(data)):
            new_data.append(line)
    # for line in tqdm(data, total=len(data)):
    #     new_data.append(execute_line(line))

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, type=str)
    args = parser.parse_args()

    with jsonlines.open(args.input) as f:
        data = [l for l in f.iter()]

    data = execute_lines(data)

    with jsonlines.open(args.input + '.executed', 'w') as writer:
        writer.write_all(data)


if __name__ == "__main__":
    main()

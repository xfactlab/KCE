import json
from argparse import ArgumentParser

from src.eval_funcs import *

parser = ArgumentParser(description="Evaluate the results")
parser.add_argument("--dataset", type=str, help="nq, strategyqa, qasc, or hotpotqa", required=True)
parser.add_argument("--result_path", type=str, help="path to the results which are the output of read.py", required=True)
args = parser.parse_args()

# read results (after read)
with open(args.result_path, "r") as f:
    data = [json.loads(line) for line in f]

# set eval function according to dataset
if args.dataset == "strategyqa":
    eval_func = bool_accuracy
elif args.dataset == "qasc":
    eval_func = eight_way_accuracy
elif args.dataset in ["hotpotqa", "nq"]:
    eval_func = exact_match
else:
    raise ValueError("Dataset not supported")

# eval
answer_keys = ['a_org2gpt', 'a_org2claude', 'a_gpt2gpt', 'a_gpt2claude', 'a_claude2gpt', 'a_claude2claude']
count = {key: 0 for key in answer_keys}

for line in data:
    for key in answer_keys:
        if eval_func(line['a'], line[key]):
            count[key] += 1

for key in answer_keys:
    print(f"{key}: {count[key]}/{len(data)} ({count[key] / len(data) * 100:.2f}%)")

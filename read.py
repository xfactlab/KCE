import json
import yaml
from argparse import ArgumentParser
from datetime import datetime

from src.data import QADataset
from src.template import ReadTemplate
from src.generate import OpenaiComplete, AnthropicComplete

# args
parser = ArgumentParser(description="Generate answer to the question with given gold/paraphrased context")
parser.add_argument("--dataset", type=str, help="nq, strategyqa, qasc, or hotpotqa", required=True)
parser.add_argument("--paraph_path", type=str, help="path to the paraphrased contexts which are the output of paraphrase.py", required=True)
parser.add_argument("--buffer_size", type=int, default=50, help="number of results to save at once. Default: 50")
parser.add_argument("--data_path", type=str, default=None, help="path to the dataset file. If not given, use the path in config.yaml")
parser.add_argument("--result_path", type=str, default=None, help="path to save the results. If not given, automatically set")
args = parser.parse_args()

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# configure result path
if args.result_path is None:
    NOW = datetime.now().strftime("%m%d-%H%M")
    RESULT_PATH = f"read_{args.dataset}_{NOW}.jsonl"
else:
    RESULT_PATH = args.result_path


# read paraphrased contexts into memory
PARAPH_PATH = args.paraph_path
paraphrased = {}
with open(PARAPH_PATH, "r") as f:
    for line in f:
        line = json.loads(line)
        paraphrased[line['id']] = line

# dataset, template, and LLMs
if args.data_path is None:  # follow config.yaml
    dataset = QADataset(config['dataset_path'][args.dataset], args.dataset)
else:
    dataset = QADataset(args.data_path, args.dataset)
template = ReadTemplate(args.dataset)
gpt = OpenaiComplete(config['openai_key_path'])
claude = AnthropicComplete(config['anthropic_key_path'])


# run
BUFFER_SIZE = args.buffer_size
buffer = []
for i, line in enumerate(dataset):
    # original context
    content_org = template.format(line['q'], line['c'])

    # gpt paraphrased context
    content_gpt = template.format(line['q'], paraphrased[line['id']]['c_gpt'])

    # claude paraphrased context
    content_claude = template.format(line['q'], paraphrased[line['id']]['c_claude'])

    # complete
    try:
        a_org2gpt = gpt.complete(content_org, model='gpt-3.5-turbo', max_tokens=10)
        a_org2claude = claude.complete(content_org, model='claude-1', max_tokens_to_sample=10)
        a_gpt2gpt = gpt.complete(content_gpt, model='gpt-3.5-turbo', max_tokens=10)
        a_gpt2claude = claude.complete(content_gpt, model='claude-1', max_tokens_to_sample=10)
        a_claude2gpt = gpt.complete(content_claude, model='gpt-3.5-turbo', max_tokens=10)
        a_claude2claude = claude.complete(content_claude, model='claude-1', max_tokens_to_sample=10)
    except Exception as e:
        print(e)
        break

    # save
    result = {'id': line['id'], 'a': line['a'], 'a_org2gpt': a_org2gpt, 'a_org2claude': a_org2claude,
              'a_gpt2gpt': a_gpt2gpt, 'a_gpt2claude': a_gpt2claude, 'a_claude2gpt': a_claude2gpt,
              'a_claude2claude': a_claude2claude}
    buffer.append(result)

    # print progress
    print(f"Ran {i + 1}/{len(dataset)}")

    # write
    if (i + 1) % BUFFER_SIZE == 0:
        with open(RESULT_PATH, "a") as f:
            for res in buffer:
                f.write(json.dumps(res) + "\n")
        buffer = []
    break

# write any remaining results in the buffer
if buffer:
    with open(RESULT_PATH, "a") as f:
        for res in buffer:
            f.write(json.dumps(res) + "\n")

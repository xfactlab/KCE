import json
import yaml
from argparse import ArgumentParser
from datetime import datetime

from src.data import QADataset
from src.template import ParaphraseTemplate
from src.generate import OpenaiComplete, AnthropicComplete


# args
parser = ArgumentParser(description="Paraphrase human-annotated gold context using LLMs")
parser.add_argument("--dataset", type=str, help="nq, strategyqa, qasc, or hotpotqa", required=True)
parser.add_argument("--buffer_size", type=int, default=50, help="number of results to save at once. Default: 50")
parser.add_argument("--data_path", type=str, default=None, help="path to the dataset file. If not given, use the path in config.yaml")
parser.add_argument("--result_path", type=str, default=None, help="path to save the results. If not given, automatically set")
args = parser.parse_args()

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# configure result path
if args.result_path is None:
    NOW = datetime.now().strftime("%m%d-%H%M")
    RESULT_PATH = f"paraph_{args.dataset}_{NOW}.jsonl"
else:
    RESULT_PATH = args.result_path


# dataset, template, and LLMs
if args.data_path is None:  # follow config.yaml
    dataset = QADataset(config['dataset_path'][args.dataset], args.dataset)
else:
    dataset = QADataset(args.data_path, args.dataset)
template = ParaphraseTemplate(args.dataset)
gpt = OpenaiComplete(config['openai_key_path'])
claude = AnthropicComplete(config['anthropic_key_path'])


# run
BUFFER_SIZE = args.buffer_size  # prevent losing results in case of crash
buffer = []
for i, line in enumerate(dataset):
    content = template.format(line['q'], line['c'])
    try:
        c_gpt = gpt.complete(content, model='gpt-3.5-turbo', max_tokens=300)
        c_claude = claude.complete(content, model='claude-1', max_tokens_to_sample=300)
    except Exception as e:
        print(e)
        break

    result = {'id': line['id'], 'q': line['q'], 'c': line['c'], 'c_gpt': c_gpt, 'c_claude': c_claude}
    buffer.append(result)

    # print progress
    print(f"Saved {i + 1}/{len(dataset)}")

    # write
    if (i + 1) % BUFFER_SIZE == 0:
        with open(RESULT_PATH, "a") as f:
            for res in buffer:
                f.write(json.dumps(res) + "\n")
        buffer = []

# Write any remaining results in the buffer
if buffer:
    with open(RESULT_PATH, "a") as f:
        for res in buffer:
            f.write(json.dumps(res) + "\n")

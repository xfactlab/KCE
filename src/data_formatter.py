import json
import yaml
from typing import List


# helper functions
def concat_title_context(title: str, body: str) -> str:
    return f"Title: {title}\n\n{body}"


def concat_contexts(contexts: List[str]) -> str:
    return "\n\n".join(contexts)


# formatter for each dataset
def nq(line: dict) -> dict:
    #
    question: str
    context: str
    answers: List[str]
    id_: int

    question = line['question']
    context_title = line['ctxs'][0]['title']
    context_body = line['ctxs'][0]['text']
    context = concat_title_context(context_title, context_body)
    answers = line['answers']
    id_ = line['id'] if 'id' in line.keys() else None

    return {"q": question, "c": context, "a": answers, "id": id_}


def hotpotqa(line: dict) -> dict:
    # Dataset: https://hotpotqa.github.io/
    # We used dev distractor split: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    question: str
    context: str
    answer: bool
    id_: str

    id_ = line['_id']
    question = line['question']
    answer = line['answer']

    # all contexts including distractors
    context_dict = {c[0]: c[1] for c in line['context']}

    # contexts
    contexts = []
    for title in set([title for title, index in line['supporting_facts']]):
        body = "".join(context_dict[title])
        context = concat_title_context(title, body)
        contexts.append(context)
    context = concat_contexts(contexts)

    return {'id': id_, 'q': question, 'c': context, 'a': answer}


def strategyqa(line: dict) -> dict:
    # Dataset: https://allenai.org/data/strategyqa
    # We used train split ("strategyqa_train.json")
    question: str
    context: str
    answer: bool
    id_: str

    with open("../config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    PATH_PARAGRAPHS = config['strategyqa_paragraph_path']

    question = line['question']
    answer = line['answer']
    id_ = line['qid']

    # contexts
    context_ids = []
    for step in line['evidence'][0]:
        for e in step:
            if e != 'no_evidence' and e != 'operation':
                for i in e:
                    context_ids.append(i)

    with open(PATH_PARAGRAPHS, "r") as f:
        paragraphs = json.load(f)
    contexts = [paragraphs[i] for i in context_ids]
    contexts = [concat_title_context(p['title'], p['content']) for p in contexts]
    context = concat_contexts(contexts)

    return {"q": question, "c": context, "a": answer, "id": id_}


def qasc(line: dict) -> dict:
    # Dataset: https://allenai.org/data/qasc
    # We used dev split ("dev.jsonl")
    question: str
    context: str
    answer: str
    id_: str

    question = line['formatted_question']
    answer = line['answerKey']
    id_ = line['id']

    # contexts
    contexts = [line['fact1'], line['fact2']]
    context = concat_contexts(contexts)

    return {"q": question, "c": context, "a": answer, "id": id_}

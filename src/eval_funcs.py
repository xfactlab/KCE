import re
from typing import List, Union
import string


# helper function (private)
def _normalize(s):
    # code from GenRead (Yu et al., 2023)
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# main eval functions (public)

# for NQ and HotpotQA
def exact_match(answers: Union[List[str], str], prediction: str):
    # type check
    if isinstance(answers, str):
        answers = [answers]

    # eval
    for a in answers:
        if _normalize(a) == _normalize(prediction):
            return True
    return False


# for StrategyQA
def bool_accuracy(answer: bool, prediction: str):
    # code from zero-shot CoT (Kojima et al., 2022)
    prediction = prediction.lower()
    prediction = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", prediction)
    prediction = prediction.split(" ")
    prediction = [i for i in prediction if i in ("yes", "no")]

    if prediction:
        if answer:
            return prediction[0] == 'yes'
        else:
            return prediction[0] == 'no'
    else:
        return False


# for QASC
def eight_way_accuracy(answer: str, prediction: str):
    """Check the first character out of 8 options."""
    prediction = re.findall(r'A|B|C|D|E|F|G|H', prediction)
    if prediction:
        return answer == prediction[0]
    return False


# test
if __name__ == "__main__":
    s = "tHiS    is 'a test..! string 32."
    assert (_normalize(s) == "this is test string 32")

    s2 = "yes. true"
    assert bool_accuracy(True, s2)

    s3 = "The answer is (A)."
    assert eight_way_accuracy("A", s3)

from abc import abstractmethod, ABC


class Template(ABC):
    def __init__(self, dataset: str):
        # dataset: one of ['nq', 'strategyqa', 'qasc', 'hotpotqa']
        self.dataset = dataset

    @abstractmethod
    def format(self, q: str, c: str) -> str:
        """
        Format the given question and context into a prompt, according to the dataset.
        Args:
            q: question
            c: context

        Returns: formatted prompt
        """
        pass


class ParaphraseTemplate(Template):
    def format(self, q: str, c: str) -> str:
        if self.dataset in ['nq']:  # single-hop
            return f"Paraphrase a background document in your own words to answer the given question.\n\nQuestion: {q}\n\nDocument: {c}"
        elif self.dataset in ['strategyqa', 'qasc', 'hotpotqa']:  # multi-hop
            return f"Paraphrase the background documents into a single document in your own words to answer the given question.\n\nQuestion: {q}\n\nDocuments: {c}\n\nParaphrased: "
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")


class ReadTemplate(Template):
    def format(self, q: str, c: str) -> str:
        if self.dataset == 'strategyqa':
            return f"Read the passage and answer the question with yes or no.\n\n{c}\n\n{q}"
        elif self.dataset == 'nq':
            content = "Referring to the passage, find the correct answer (just one entity) to the given question. I will first show you few examples.\n\n"
            content += "\n\n".join([f"Passage: {demo['c']}\nQuestion: {demo['q']}\nAnswer: {demo['a']}" for demo in NQ_DEMOS])
            content += f"\n\nPassage: {c}\nQuestion: {q}\nAnswer:"
            return content
        elif self.dataset == 'qasc':
            return f"Read the passage and answer the question with one of A, B, C, D, E, F, G, or H.\n\n{c}\n\n{q}"
        elif self.dataset == 'hotpotqa':
            return f"Referring `to the passage, find the correct answer (just one entity) to the given question.\n\n{c}\n\n{q}"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")


# demos for few-shot prompting.
# these are generated from GPT-3.5-turbo, from questions in the NQ dev set. Note that these questions were excluded from the experiment.
NQ_DEMOS = [
    {"q": "how i.met your mother who is the mother", "c": "Tracy McConnell, also known as \"The Mother,\" is the main character in the TV show How I Met Your Mother on CBS. The show follows the story of how Ted Mosby met The Mother, with Future Ted narrating. Though she is only heard of in eight episodes from \"Lucky Penny\" to \"The Time Travelers,\" it is not until the episode \"Something New\" that she is fully seen, and in season 9 she is promoted to a main character. Cristin Milioti plays the role of The Mother.", "a": "Tracy McConnell"},
    {"q": "which is the most common use of opt-in e-mail marketing", "c": "Permission marketing involves sending newsletters to a company's customers which inform them about new products, upcoming events, or promotions. The customers have to give consent to receive the emails, which is usually asked for at the point of purchase.", "a": "a newsletter sent to an advertising firm 's customers"},
    {"q": "who had the most wins in the nfl", "c": "Tom Brady, a current quarterback, has set three impressive records: he has the most wins overall (220), the most regular season wins (195), and the most postseason wins (25) as of Week 16 in the 2017 NFL season. He achieved all of these records while playing for the New England Patriots exclusively, which means his team also holds the record for the most wins by a single team in each category.", "a": "Tom Brady"},
]


# test
if __name__ == '__main__':
    q = "Did King Sejong speak French?"
    c = "King Sejong was a king of Chosun dynasty. He invented Korean alphabet, Hangul. He was born in 1397 and died in 1450."

    print("\n=======Paraphrase Template for NQ=======\n")
    print(ParaphraseTemplate('nq').format(q, c))

    print("\n=======Read Template for NQ=======\n")
    print(ReadTemplate('nq').format(q, c))

    print("\n=======Paraphrase Template for StrategyQA=======\n")
    print(ParaphraseTemplate('strategyqa').format(q, c))

    print("\n=======Read Template for StrategyQA=======\n")
    print(ReadTemplate('strategyqa').format(q, c))

import json

import src.data_formatter as formatter


class QADataset:
    def __init__(self, path: str, dataset: str):
        """
        An iterator for the QA dataset.
        Args:
            path (str): path to the dataset. Either json or jsonl file
            dataset (str): name of the dataset. Either "nq", "strategyqa", "qasc", or "hotpotqa"
        """
        self.path = path

        # set formatter to unify data format
        if dataset == "nq":
            self.formatter = formatter.nq
        elif dataset == "strategyqa":
            self.formatter = formatter.strategyqa
        elif dataset == "qasc":
            self.formatter = formatter.qasc
        elif dataset == "hotpotqa":
            self.formatter = formatter.hotpotqa
        else:
            raise ValueError("Dataset not supported")

    def __iter__(self):
        with open(self.path, "r") as f:
            # jsonl file
            if self.path.endswith(".jsonl"):
                for line in f:
                    yield self.formatter(json.loads(line))

            # json file
            elif self.path.endswith(".json"):
                data = json.load(f)
                for line in data:
                    yield self.formatter(line)

            else:
                raise ValueError("Dataset file format not supported")

    def __len__(self):
        if self.path.endswith(".jsonl"):
            with open(self.path, "r") as f:
                return sum(1 for _ in f)
        elif self.path.endswith(".json"):
            return len(json.load(open(self.path, "r")))
        else:
            raise ValueError("Dataset file format not supported")


# test
if __name__ == '__main__':
    import yaml
    with open("../config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for dataset in config['dataset_path'].keys():
        print(f"Test {dataset}")
        dataset = QADataset(config['dataset_path'][dataset], dataset)
        for i, line in enumerate(dataset):
            print(line)
            if i == 5:
                break


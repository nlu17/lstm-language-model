import numpy as np

class DataSource:
    def __init__(self, data_file, pad_idx):
        self.start = 0
        self.dataset = {}
        with open(data_file, "r") as f:
            lines = f.readlines()[:100]  # TODO: remove the :100 part
            lines = [line.strip("\n").split() for line in lines]
            lines = [np.array([int(x) for x in line]) for line in lines]
            targets = [np.append(line[1:], pad_idx) for line in lines]

            self.dataset["input"] = lines
            self.dataset["target"] = targets

        self.size = len(self.dataset["input"])

    def next_train_batch(self, batch_size):
        batch_inputs = []
        batch_targets = []
        end = self.start + batch_size
        batch_inputs = self.dataset["input"][self.start:end]
        batch_targets = self.dataset["target"][self.start:end]

        self.start = end

        if (len(batch_inputs) < batch_size):
            rest = batch_size - len(batch_inputs)
            batch_inputs += self.dataset["input"][:rest]
            batch_targets += self.dataset["target"][:rest]
            self.start = rest

        return batch_inputs, batch_targets

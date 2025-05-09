from torch.utils.data import Dataset

class Datasett(Dataset):
    def __init__(self, inputs, pos, labels):
        self.inputs = inputs
        self.pos = pos
        self.labels = labels


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        numerical_inputs = self.inputs[index]
        numerical_pos = self.pos[index]
        numerical_label = self.labels[index]

        return numerical_inputs, numerical_pos, numerical_label
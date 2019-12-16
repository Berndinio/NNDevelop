from utils import get_dataset_generator, get_dataset_length
from torch.utils.data import Dataset


class AmazonDataset(Dataset):
    def __init__(self, filename, mode=0):
        # get tokenized strings
        self.dataset_length = get_dataset_length(filename)
        self.mode = mode
        self.cuts = [int(0.0*self.dataset_length),
                     int(0.7*self.dataset_length),
                     int(0.9*self.dataset_length),
                     int(1.0*self.dataset_length)-1]
        self.dataset_length = self.cuts[self.mode+1] - self.cuts[self.mode]
        # load into ram
        generator = get_dataset_generator(filename, self.cuts[self.mode], self.cuts[self.mode+1])
        self.data = []
        for sample in generator:
            self.data.append(sample)
        self.dataset_length = len(self.data)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        return self.data[index]

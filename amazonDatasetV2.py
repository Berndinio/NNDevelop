from torch.utils.data import Dataset
import torch
import tables
import numpy as np


class AmazonDataset(Dataset):
    '''
    Highly optimized version of the amazonDatasetV1
    '''

    def __init__(self, filename, scaling=1.0, mode=0, p_map=False):
        # get tokenized strings
        h5_file = tables.open_file(filename, mode='r')

        # calculate dataset indices - train - test - validation
        # dataset is not ordered anyway, so we just can take the first scaling*originalLength
        self.dataset_length = int(len(h5_file.root.data) * scaling)
        # these values cut them 70% train, 20% test, 10% validation
        # the strange scalers come from the equal distribution of the class which is done afterwards, so we do not have
        # to load the whole dataset from the h5f file
        self.cuts = [int(0.0 * self.dataset_length),
                     int(0.7 * self.dataset_length),
                     int(0.9 * self.dataset_length),
                     int(1.0 * self.dataset_length) - 1]
        # cut the data
        # noinspection PyArgumentList
        data = h5_file.root.data[self.cuts[mode]:self.cuts[mode + 1]]
        # conclude classes
        if p_map:
            data[:, 0] = np.where(data[:, 0] == 1, 0, data[:, 0])
            data[:, 0] = np.where(data[:, 0] == 2, 1, data[:, 0])
            data[:, 0] = np.where(data[:, 0] == 3, 2, data[:, 0])
            data[:, 0] = np.where(data[:, 0] == 4, 2, data[:, 0])
        # get the minimum count to make classes equally distributed
        data = torch.LongTensor(data)
        counts = np.unique(data[:, 0].numpy(), return_counts=True)
        min_count = counts[1].min()
        self.data = None
        for label in range(len(counts[0])):
            idxs = np.where(data[:, 0] == label)[0]
            subdata = data[idxs[:min_count]]
            if self.data is None:
                self.data = subdata
            else:
                self.data = torch.cat((self.data, subdata), dim=0)
        self.dataset_length = self.data.shape[0]
        h5_file.close()

        # check if equally distributed classes
        check_unique = np.unique(self.data[:, 0], return_counts=True)
        for i in range(len(check_unique[0]) - 1):
            assert (check_unique[1][i] == check_unique[1][i + 1])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1:]

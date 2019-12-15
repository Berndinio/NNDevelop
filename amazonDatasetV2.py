from torch.utils.data import Dataset
import torch
import tables

class AmazonDataset(Dataset):
    '''
    Highly optimized version of the amazonDatasetV1
    '''
    def __init__(self, filename, scaling=1.0, mode=0):
        # get tokenized strings
        h5_file = tables.open_file(filename, mode='r')
        # calculate dataset indices - train - test - validation
        # dataset is not ordered anyway, so we just can take the first scaling*originalLength
        self.dataset_length = int(len(h5_file.root.data)*scaling)
        self.cuts = [int(0.0*self.dataset_length),
                     int(0.7*self.dataset_length),
                     int(0.9*self.dataset_length),
                     int(1.0*self.dataset_length)-1]
        # cut the data
        # noinspection PyArgumentList
        self.data = torch.LongTensor(h5_file.root.data[self.cuts[mode]:self.cuts[mode+1]])
        self.dataset_length = self.data.shape[0]
        h5_file.close()
        print(self.data.shape)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1:]


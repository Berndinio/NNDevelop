import multiprocessing
from multiprocessing import Process, Semaphore, Manager
import json
import torch
from transformers import *
import time
import gc
import numpy as np
from tempfile import TemporaryFile
import tables




# HDF5 save format prepare
filename = 'data/test_dataset.h5'
ROW_SIZE = 265
h5_file = tables.open_file(filename, mode='a')
atom = tables.Float32Atom()
array_c = h5_file.create_earray(h5_file.root, 'data', atom, (0, ROW_SIZE+1))

labels = np.random.randint(0, 5, (1128437, 1))
features = np.random.randint(100, 300, (1128437, ROW_SIZE))
data = np.concatenate((labels, features), axis=1)

for i, sample in enumerate(data):
    print(i/float(1128437) * 100)
    h5_file.root.data.append(sample[None, :])
h5_file.close()

# transform to torch
h5_file = tables.open_file(filename, mode='r')
final_mat = h5_file.root.data
final_mat = torch.tensor(final_mat)
print(final_mat[0:], "\n", final_mat.shape)
h5_file.close()
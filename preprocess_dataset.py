'''
This file is for EFFIZIENT MULTITHREADED preprocessing of the amazon data.
'''

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
outfile = TemporaryFile()
filename = 'data/Cell_Phones_and_Accessories_5_preprocessed.h5'
ROW_SIZE = 265
h5_file = tables.open_file(filename, mode='a')
atom = tables.Float32Atom()
array_c = h5_file.create_earray(h5_file.root, 'data', atom, (0, ROW_SIZE+1))

# prepare dataset needed variables
utils_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Loading dataset file")
get_dataset_idx_file = open("data/Cell_Phones_and_Accessories_5.json", "r")
get_dataset_idx_file_lines = get_dataset_idx_file.readlines()
print("Loaded dataset file")


def get_dataset_length(f_name):
    with open(f_name, "r") as file:
        return len(file.readlines())


def get_dataset_idx(idx, return_dict, semaphore, max_input_length=ROW_SIZE):
    if True:
        line = get_dataset_idx_file_lines[idx]
        # start parsing
        json_object = json.loads(line)
        keys = json_object.keys()
        if "overall" in keys and "reviewText" in keys:
            # text tokenization
            tokenized = utils_bert_tokenizer.tokenize(json_object["reviewText"])
            # handle too long reviews
            if len(tokenized) > max_input_length - 2:
                tokenized = tokenized[:max_input_length - 2]
            # PAD the things
            tokenized += ["[PAD]"] * (max_input_length - 2 - len(tokenized))
            # here are start and end appended
            text = utils_bert_tokenizer.encode(tokenized)
            text = torch.LongTensor(text)
            # one hot labels
            label = int(json_object["overall"]) - 1
            # try to catch error
            if label is None:
                label = 1
            if text is None:
                text = [101]
                text += [0] * (max_input_length - 2)
                text += [102]
            # put all together
            yield_value = np.concatenate(([label], text), 0)[None, :]
            return_dict[str(idx)] = yield_value
        else:
            return_dict[str(idx)] = None
        semaphore.release()
        return


d_length = get_dataset_length("data/Cell_Phones_and_Accessories_5.json")
# print(d_length) = 1128437
# d_length = 1000
walk_stepwidth = 1000

semaphore = Semaphore(8)
walk_idx = 0
manager = Manager()
return_dic = manager.dict()

while walk_idx < d_length - 1:
    start_time = time.time()
    jobs = []
    for idx in range(walk_idx, min(d_length, walk_idx + walk_stepwidth)):
        # start conversion jobs
        semaphore.acquire()
        p = multiprocessing.Process(target=get_dataset_idx, args=(idx, return_dic, semaphore))
        jobs.append(p)
        p.start()
    # increase counter
    walk_idx += walk_stepwidth

    # wait for jobs
    for j in jobs:
        j.join()
    for j in jobs:
        j.terminate()
    # get results, append them to h5 file and clear dict
    gc.disable()
    for key in return_dic.keys():
        ret = return_dic[key]
        if ret is not None:
            h5_file.root.data.append(return_dic[key])
    return_dic.clear()
    gc.enable()

    print(str(walk_idx / float(d_length) * 100.0) + "%              " + str(
        time.time() - start_time) + "                      ")  # , end="\r")

# transform to torch
final_mat = h5_file.root.data
final_mat = torch.tensor(final_mat)
print(final_mat[0:], "\n", final_mat.shape)
h5_file.close()

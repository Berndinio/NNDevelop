import json

import torch
from transformers import *


def get_dataset_length(filename):
    with open(filename, "r") as file:
        return len(file.readlines())


utils_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Loading dataset file")
get_dataset_idx_file = open("data/Cell_Phones_and_Accessories_5.json", "r")
get_dataset_idx_file_lines = get_dataset_idx_file.readlines()
print("Loaded dataset file")
def get_dataset_idx(idx, max_input_length=256):
    if True:
        line = get_dataset_idx_file_lines[idx]
        print(line)
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
            yield_value = (label, text)
            return yield_value


def get_dataset_generator(filename, start, stop, max_input_length=256):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(filename, "r") as file:
        try:
            num_line = 0
            while True:
                line = file.readline()
                if num_line < start:
                    continue
                if num_line >= stop:
                    break
                num_line += 1
                # start parsing
                json_object = json.loads(line)
                keys = json_object.keys()
                if "overall" in keys and "reviewText" in keys:
                    # text tokenization
                    tokenized = tokenizer.tokenize(json_object["reviewText"])
                    # handle too long reviews
                    if len(tokenized) > max_input_length - 2:
                        continue
                    tokenized += ["[PAD]"] * (max_input_length - 2 - len(tokenized))
                    # here are start and end appended
                    text = tokenizer.encode(tokenized)
                    text = torch.LongTensor(text)
                    # one hot labels
                    label = int(json_object["overall"]) - 1
                    # put all together
                    yield_value = (num_line, label, text)
                    yield yield_value
        except Exception as e:
            # try to read even the next line - maybe there is just some faulty line
            try:
                # exc_info = sys.exc_info()
                line = file.readline()
                json_object = json.loads(line)
                print("There may be a faulty line in your json file.")
                print(e)
            except:
                # both were faulty ==> end of file
                print("End of file reached.")

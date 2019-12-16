import pickle
import os
import torch
from torch.utils.data import DataLoader
from amazonDatasetV2 import AmazonDataset
import torch.nn as nn
from constants import Constants
from transformers import *
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from get_model import build_model
import numpy as np

# parameters which can be adjusted
version_number = 1
bert_filename = ""


# begin loading parameters
print("Loading parameters at: " + "model_saves/v" + str(version_number) + "/parameters.pkl")
parameters = pickle.load(open("model_saves/v" + str(version_number) + "/parameters.pkl", 'rb'))
print(parameters)

# begin loading bert
if bert_filename == "":
    # if no specific file should be loaded, load the latest epoch
    f = None
    for (dirpath, dirnames, filenames) in os.walk("model_saves/v" + str(version_number) + "/"):
        f = filenames
        break
    # sort
    f = sorted(f)
    # use all elements
    berts_to_load = ["model_saves/v" + str(version_number) + "/" + x for x in f][:-2]
    berts_to_load.reverse()
    print(berts_to_load)
else:
    berts_to_load = ["model_saves/v" + str(version_number) + "/" + bert_filename]

# load the Datasets
test_loader = DataLoader(
    dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 1),
    batch_size=parameters["batch_size"], shuffle=False, num_workers=8)
print("Finished Loading Testing Data...")
valid_loader = DataLoader(
    dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 2),
    batch_size=parameters["batch_size"], shuffle=False, num_workers=8)
print("Finished Loading Validation Data...")



for bert_to_load in berts_to_load:
    # replace the classification layer
    model = build_model(parameters)

    # test
    laenge = len(test_loader)
    criterion = nn.CrossEntropyLoss()
    # for one-hot encoding
    auc_roc_output, auc_roc_target = None, None

    # [<cross-entropy-loss>, <auc-roc-score>, <precent of True-positive>]
    stats = [0.0, 0.0, 0.0]
    # begin testing
    for i, (target, net_input) in enumerate(test_loader):
        print(str(i / float(laenge) * 100.0) + "              ", end="\r")
        model.eval()
        target, net_input = target.to(Constants.device), net_input.to(Constants.device)
        # forward + compute statistics
        output = model(net_input)[0]
        # normal cross entropy loss
        stats[0] += criterion(output, target).item()
        # roc-auc-score
        if auc_roc_target is None:
            auc_roc_output, auc_roc_target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        else:
            auc_roc_output = np.concatenate((auc_roc_output, output.cpu().detach().numpy()), axis=0)
            auc_roc_target = np.concatenate((auc_roc_target, target.cpu().detach().numpy()), axis=0)

    # scale stats data
    stats[0] /= (len(test_loader) * parameters["batch_size"])
    stats[1] = roc_auc_score(auc_roc_target, auc_roc_output, average="weighted", multi_class="ovo")
    stats[2] = ((np.argmax(auc_roc_output, axis=1) == auc_roc_target).sum()/auc_roc_target.shape[0])*100.0
    # print results
    print("Bert: ", bert_to_load)
    print("Test stat results: ", stats)

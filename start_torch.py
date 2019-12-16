# zuerst wollte ich:
# https://arxiv.org/pdf/1902.09492.pdf
# https://arxiv.org/pdf/1903.03243.pdf

# Da Code zum Teil nicht offen (fürs mapping)
# ==> Amazon review classification mit attention analyse.
# https://nijianmo.github.io/amazon/index.html
# Vielleicht Vergleich BERT-ELMO?

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import *
from constants import Constants
import torch
from amazonDatasetV2 import AmazonDataset
from os import path, mkdir
import pickle
from get_model import build_model
from sklearn.metrics import roc_auc_score


# create directory automatically
next_save_path = 0
while path.exists("model_saves/v" + str(next_save_path)):
    next_save_path += 1
mkdir("model_saves/v" + str(next_save_path))

# some variables
parameters = {
    "batch_size": 6,
    "num_labels": 5,
    "num_epochs": 100,
    "lr_classifier": 0.0001,
    "lr_bert": 0.00001,
    "dataset_scaling": 0.01
}
pickle.dump(parameters, open("model_saves/v" + str(next_save_path) + "/parameters.pkl", 'wb'))


# load the raw dataset
train_loader = DataLoader(
    dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 0),
    batch_size=parameters["batch_size"], shuffle=True, num_workers=8)
print("Finished Loading Training Data...")
test_loader = DataLoader(
    dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 1),
    batch_size=parameters["batch_size"], shuffle=False, num_workers=8)
print("Finished Loading Testing Data...")

# replace the classification layer
model = build_model(parameters)

optimizer = optim.Adam(
    [
        {"params": model.bert.parameters(), "lr": parameters["lr_bert"]},
        {"params": model.classifier.parameters(), "lr": parameters["lr_classifier"]},
    ],
    weight_decay=0.01
)
criterion = nn.CrossEntropyLoss()
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("Training dataset length: ", len(train_loader))
print("Testing dataset length: ", len(test_loader))

# open losses pickle file
all_losses = [[], [], [], []]
for epoch in range(parameters["num_epochs"]):
    # train
    running_loss = 0.0
    laenge = len(train_loader)
    for i, (target, net_input) in enumerate(train_loader):
        print(str(i / float(laenge) * 100.0) + "              ", end="\r")
        target, net_input = target.to(Constants.device), net_input.to(Constants.device)
        model.train()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(net_input)[0]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (int(len(train_loader) / 10)) == 0 or ((i + 1) % int(len(train_loader) / 10)) == 0:
            print('[%d, %5d] Training loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / ((i + 1)*parameters["batch_size"])))
    all_losses[0].append(running_loss / (len(train_loader) * parameters["batch_size"]))
    pickle.dump(all_losses, open("model_saves/v" + str(next_save_path) + "/losses.pkl", 'wb'))
    print("\n")

    # test
    running_loss = 0.0
    auc_roc_output, auc_roc_target = None, None
    laenge = len(test_loader)
    for i, (target, net_input) in enumerate(test_loader):
        print(str(i / float(laenge) * 100.0) + "              ", end="\r")
        target, net_input = target.to(Constants.device), net_input.to(Constants.device)
        model.eval()
        # forward + backward + optimize
        output = model(net_input)[0]
        loss = criterion(output, target)
        # print statistics
        running_loss += loss.item()
        # roc-auc-score
        if auc_roc_target is None:
            auc_roc_output, auc_roc_target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        else:
            auc_roc_output = np.concatenate((auc_roc_output, output.cpu().detach().numpy()), axis=0)
            auc_roc_target = np.concatenate((auc_roc_target, target.cpu().detach().numpy()), axis=0)
    all_losses[1].append(running_loss / (len(test_loader) * parameters["batch_size"]))
    all_losses[2].append(((np.argmax(auc_roc_output, axis=1) == auc_roc_target).sum()/auc_roc_target.shape[0])*100.0)
    all_losses[3].append(roc_auc_score(auc_roc_target, auc_roc_output, average="weighted", multi_class="ovo"))

    pickle.dump(all_losses, open("model_saves/v" + str(next_save_path) + "/losses.pkl", 'wb'))
    print("\n")
    # noinspection PyUnboundLocalVariable
    print('[%d, %5d] Test loss: %.3f' % (epoch + 1, i + 1, all_losses[1][-1]))
    torch.save(model.state_dict(),
               "model_saves/v" + str(next_save_path) + "/epoch_" + str(epoch).zfill(3) + "-loss_" + str(
                all_losses[1][-1]) + ".pt")

    # lr decay
    if epoch < 31:
        lr_scheduler.step()


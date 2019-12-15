# zuerst wollte ich:
# https://arxiv.org/pdf/1902.09492.pdf
# https://arxiv.org/pdf/1903.03243.pdf

# Da Code zum Teil nicht offen (fürs mapping)
# ==> Amazon review classification mit attention analyse.
# https://nijianmo.github.io/amazon/index.html
# Vielleicht Vergleich BERT-ELMO?

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

# create directory automatically
next_save_path = 0
while path.exists("model_saves/v" + str(next_save_path)):
    next_save_path += 1
mkdir("model_saves/v" + str(next_save_path))

# some variables
parameters = {
    "batch_size": 6,
    "num_labels": 5,
    "num_epochs": 20,
    "lr_classifier": 0.001,
    "lr_bert": 0.00001,
    "dataset_scaling": 0.001
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
    ])
criterion = nn.CrossEntropyLoss()
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

print("Training dataset length: ", len(train_loader))
print("Testing dataset length: ", len(test_loader))

# open losses pickle file
all_losses = [[], [], []]
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
    all_losses[1].append(running_loss / (len(test_loader) * parameters["batch_size"]))
    pickle.dump(all_losses, open("model_saves/v" + str(next_save_path) + "/losses.pkl", 'wb'))
    print("\n")
    # noinspection PyUnboundLocalVariable
    print('[%d, %5d] Test loss: %.3f' % (epoch + 1, i + 1, all_losses[1][-1]))
    torch.save(model.state_dict(),
               "model_saves/v" + str(next_save_path) + "/epoch_" + str(epoch).zfill(3) + "-loss_" + str(
                all_losses[1][-1]) + ".pt")

    # lr decay
    lr_scheduler.step()

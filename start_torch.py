# zuerst wollte ich:
# https://arxiv.org/pdf/1902.09492.pdf
# https://arxiv.org/pdf/1903.03243.pdf

# Da Code zum Teil nicht offen (fürs mapping)
# ==> Amazon review classification mit attention analyse.
# https://nijianmo.github.io/amazon/index.html
# Vielleicht Vergleich BERT-ELMO?

import os
import pickle
from os import path, mkdir

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from amazonDatasetV2 import AmazonDataset
from constants import Constants
from get_model import build_model
from torch.optim import lr_scheduler as torch_lr_scheduler


# None = plain from beginning
# <int> = version number of folder in model_saves
def main_train_loop(continue_training=None):

    # create directory automatically
    if continue_training is None:
        next_save_path = 0
        while path.exists("model_saves/v" + str(next_save_path)):
            next_save_path += 1
        mkdir("model_saves/v" + str(next_save_path))

        # some variables
        parameters = {
            "batch_size": 6,
            "num_labels": 5,
            "lr_classifier": 0.001,
            "lr_bert": 0.00001,
            "dataset_scaling": 0.001,

            "scheduler_stepsize": 5,
            "scheduler_gamma": 5,
            "weight_decay": 0.01,

            "next_epoch": 0,
            "scheduler_max_epoch": 31
        }
        pickle.dump(parameters, open("model_saves/v" + str(next_save_path) + "/parameters.pkl", 'wb'))
        # create variable for losses
        all_losses = [[], [], [], []]
        pickle.dump(parameters, open("model_saves/v" + str(next_save_path) + "/losses.pkl", 'wb'))
        # get the model
        model = build_model(parameters)
    else:
        next_save_path = continue_training
        # load parameters
        parameters = pickle.load(open("model_saves/v" + str(next_save_path) + "/parameters.pkl", 'rb'))
        # load variable for losses
        all_losses = pickle.load(open("model_saves/v" + str(next_save_path) + "/losses.pkl", 'rb'))

        # GET THE MODEL
        # if no specific file should be loaded, load the latest epoch
        f = None
        for (dirpath, dirnames, filenames) in os.walk("model_saves/v" + str(next_save_path) + "/"):
            f = filenames
            break
        # sort
        f = sorted(f)
        # use all elements
        berts_to_load = ["model_saves/v" + str(next_save_path) + "/" + x for x in f][:-2]
        # load the last/latest model
        model = build_model(parameters)
        model.load_state_dict(torch.load(berts_to_load[-1]))
        print("Loading Bert model \n", berts_to_load[-1])

    print("Beginning with parameters: \n", parameters)

    # some constant init stuff - only dependent on parameters
    optimizer = optim.Adam(
        [
            {"params": model.bert.parameters(), "lr": parameters["lr_bert"]},
            {"params": model.classifier.parameters(), "lr": parameters["lr_classifier"]},
        ],
        weight_decay=parameters["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch_lr_scheduler.StepLR(
        optimizer, step_size=parameters["scheduler_stepsize"], gamma=parameters["scheduler_gamma"]
    )

    # load the raw dataset
    train_loader = DataLoader(
        dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 0),
        batch_size=parameters["batch_size"], shuffle=True, num_workers=8)
    print("Finished Loading Training Data...")
    test_loader = DataLoader(
        dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 1),
        batch_size=parameters["batch_size"], shuffle=False, num_workers=8)
    print("Finished Loading Testing Data...")
    validation_loader = DataLoader(
        dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 2),
        batch_size=parameters["batch_size"], shuffle=False, num_workers=8)
    print("Finished Loading Testing Data...")
    print("Training dataset length: ", len(train_loader))
    print("Testing dataset length: ", len(test_loader))
    print("Testing dataset length: ", len(validation_loader))

    # open losses pickle file
    for epoch in range(parameters["next_epoch"], 10000):
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
        # append metrics
        all_losses[1].append(running_loss / (len(test_loader) * parameters["batch_size"]))
        all_losses[2].append(((np.argmax(auc_roc_output, axis=1) == auc_roc_target).sum()/auc_roc_target.shape[0])*100.0)
        all_losses[3].append(roc_auc_score(auc_roc_target, auc_roc_output, average="weighted", multi_class="ovo"))

        # dump everything
        pickle.dump(all_losses, open("model_saves/v" + str(next_save_path) + "/losses.pkl", 'wb'))
        parameters["next_epoch"] = 1 + parameters["next_epoch"]
        pickle.dump(parameters, open("model_saves/v" + str(next_save_path) + "/parameters.pkl", 'wb'))

        # print things
        print("\n")
        # noinspection PyUnboundLocalVariable
        print("Losses: ", all_losses)
        torch.save(model.state_dict(),
                   "model_saves/v" + str(next_save_path) + "/epoch_" + str(epoch).zfill(3) + "-loss_" + str(
                    all_losses[1][-1]) + ".pt")

        # lr decay
        if epoch < parameters["scheduler_max_epoch"]:
            lr_scheduler.step(epoch)
        else:
            lr_scheduler.step(parameters["scheduler_max_epoch"])


if __name__ == "__main__":
    main_train_loop()
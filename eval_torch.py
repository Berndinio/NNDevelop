import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import *

from amazonDatasetV2 import AmazonDataset
from constants import Constants
from get_model import build_model

# parameters which can be adjusted
version_number = 0
bert_filename = ""
prefix = ""

# just print the losses from training
print("Loading parameters at: " + "model_saves/"+prefix+"v" + str(version_number) + "/losses.pkl")
losses = pickle.load(open("model_saves/"+prefix+"v" + str(version_number) + "/losses.pkl", 'rb'))
print(losses)

# begin loading parameters
print("Loading parameters at: " + "model_saves/"+prefix+"v" + str(version_number) + "/parameters.pkl")
parameters = pickle.load(open("model_saves/"+prefix+"v" + str(version_number) + "/parameters.pkl", 'rb'))
print(parameters)

bert_tokenizer = BertTokenizer.from_pretrained('data/new_tokenizer')
def plot_attention_mat(attention_mat, label, input):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    # cut initial tag
    # input = input[1:]
    # attention_mat = attention_mat[-2:, 0, 1:]

    input = input[:]
    until_idx = np.where(input == 0)[0][0]
    attention_mat = attention_mat[-6:, :until_idx, 0]
    # attention_mat = np.sum(attention_mat, axis=1)

    # input = input[1:]
    # attention_mat = attention_mat[:2, 1:, 1:]
    # attention_mat = np.sum(attention_mat, axis=1)

    print(attention_mat.shape)

    # some drawing
    s = attention_mat.shape
    labels = list(range(s[1]))
    fig, ax = plt.subplots(figsize=(20, 20))
    cax = ax.matshow(attention_mat, interpolation='nearest')
    ax.grid(True)
    plt.xticks(range(s[1]), labels, rotation=90)
    plt.yticks(range(s[0]), labels)
    fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])

    # output
    zipped = list(zip(bert_tokenizer.convert_ids_to_tokens(input), list(range(len(input)))))
    print(label, zipped)
    plt.show()







# begin loading bert
if bert_filename == "":
    # if no specific file should be loaded, load the latest epoch
    f = None
    for (dirpath, dirnames, filenames) in os.walk("model_saves/"+prefix+"v" + str(version_number) + "/"):
        f = filenames
        break
    # sort
    f = sorted(f)
    # use all elements
    berts_to_load = ["model_saves/"+prefix+"v" + str(version_number) + "/" + x for x in f][:-2]
    # berts_to_load.reverse()
else:
    berts_to_load = ["model_saves/"+prefix+"v" + str(version_number) + "/" + bert_filename]

# load the Datasets
test_loader = DataLoader(
    dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 1, parameters["num_labels"] < 5),
    batch_size=parameters["batch_size"], shuffle=True, num_workers=8)
print("Finished Loading Testing Data...")
valid_loader = DataLoader(
    dataset=AmazonDataset("data/Cell_Phones_and_Accessories_5_preprocessed.h5", parameters["dataset_scaling"], 2, parameters["num_labels"] < 5),
    batch_size=parameters["batch_size"], shuffle=True, num_workers=8)
print("Finished Loading Validation Data...")

# dimension 1 ==> train, test, validation
# dimension 2 ==> [<cross-entropy-loss>, <precent of correct>, <auc-roc-score>, <precent of correct relaxed>]
# dimension 3 ==> epochs
overall_stats = [[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]]
overall_stats[0] = losses
for bert_i, bert_to_load in enumerate(berts_to_load):
    print("Bert: ", bert_to_load)

    # load the model
    model = build_model(parameters).to(Constants.device)
    model.load_state_dict(torch.load(bert_to_load, map_location=Constants.device))
    model.output_attentions = True

    #########################################################
    # TEST DATASET
    #########################################################
    laenge = len(test_loader)
    criterion = nn.CrossEntropyLoss()
    # for saving test vectors
    auc_roc_output, auc_roc_target = None, None
    # [<cross-entropy-loss>, <auc-roc-score>, <precent of correct>, <precent of correct relaxed>]
    loss = 0.0
    # begin testing
    for i, (target, net_input) in enumerate(test_loader):
        print(str(i / float(laenge) * 100.0) + "              ", end="\r")
        model.eval()
        target, net_input = target.to(Constants.device), net_input.to(Constants.device)
        # forward + compute statistics
        output, attentions = model(net_input)
        # print(attentions[0].shape)
        # attention_mat = attentions[0][0]
        # plot_attention_mat(attention_mat.cpu().detach().numpy(),
        #                    target[0].cpu().detach().numpy(), net_input[0].cpu().detach().numpy())
        # normal cross entropy loss
        loss += criterion(output, target).item()
        # roc-auc-score
        if auc_roc_target is None:
            auc_roc_output, auc_roc_target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        else:
            auc_roc_output = np.concatenate((auc_roc_output, output.cpu().detach().numpy()), axis=0)
            auc_roc_target = np.concatenate((auc_roc_target, target.cpu().detach().numpy()), axis=0)
    # get stats data
    overall_stats[1][0].append(loss / (len(test_loader) * parameters["batch_size"]))
    prediction = np.argmax(auc_roc_output, axis=1)
    overall_stats[1][1].append(((prediction == auc_roc_target).sum()/auc_roc_target.shape[0])*100.0)
    overall_stats[1][2].append(roc_auc_score(auc_roc_target, auc_roc_output, average="weighted", multi_class="ovo"))
    overall_stats[1][3].append(((((prediction == auc_roc_target) + (prediction == auc_roc_target+1) + (prediction == auc_roc_target-1)) >= 1).sum()
                /auc_roc_target.shape[0]
               )*100.0)

    # print results
    print("Bert: ", bert_to_load)
    print("Test stat results: ", overall_stats[1][0][-1], overall_stats[1][1][-1], overall_stats[1][2][-1], overall_stats[1][3][-1])

    #########################################################
    # VALIDATION DATASET
    #########################################################
    laenge = len(valid_loader)
    criterion = nn.CrossEntropyLoss()
    # for saving test vectors
    auc_roc_output, auc_roc_target = None, None
    # [<cross-entropy-loss>, <precent of True-positive>, <auc-roc-score>]
    loss = 0.0
    # begin testing
    for i, (target, net_input) in enumerate(valid_loader):
        print(str(i / float(laenge) * 100.0) + "              ", end="\r")
        model.eval()
        target, net_input = target.to(Constants.device), net_input.to(Constants.device)
        # forward + compute statistics
        output, attentions = model(net_input)
        attention_mat = attentions[0][0][1]
        # plot_attention_mat(attention_mat.cpu().detach().numpy(), target[0], net_input[0])
        # normal cross entropy loss
        loss += criterion(output, target).item()
        # roc-auc-score
        if auc_roc_target is None:
            auc_roc_output, auc_roc_target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        else:
            auc_roc_output = np.concatenate((auc_roc_output, output.cpu().detach().numpy()), axis=0)
            auc_roc_target = np.concatenate((auc_roc_target, target.cpu().detach().numpy()), axis=0)
    # get stats data
    overall_stats[2][0].append(loss / (len(valid_loader) * parameters["batch_size"]))
    prediction = np.argmax(auc_roc_output, axis=1)
    overall_stats[2][1].append(((prediction == auc_roc_target).sum()/auc_roc_target.shape[0])*100.0)
    overall_stats[2][2].append(roc_auc_score(auc_roc_target, auc_roc_output, average="weighted", multi_class="ovo"))
    overall_stats[2][3].append(((((prediction == auc_roc_target) + (prediction == auc_roc_target+1) + (prediction == auc_roc_target-1)) >= 1).sum()
                /auc_roc_target.shape[0]
               )*100.0)

    # print results
    print("Bert: ", bert_to_load)
    print("Validation stat results: ", overall_stats[2][0][-1], overall_stats[2][1][-1], overall_stats[2][2][-1], overall_stats[2][3][-1])
    del auc_roc_output, auc_roc_target, model, output, attentions


    #########################################################
    # PLOT each time to avoid data loss if we crash
    #########################################################
    # dimension 2 ==> [<cross-entropy-loss>, <precent of correct>, <auc-roc-score>, <precent of correct relaxed>]
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    s = len(overall_stats[1][0])
    # plot CE loss
    plt.title("Cross Entropy Loss")
    plt.plot(overall_stats[0][0][:s], label="train")
    plt.plot(overall_stats[1][0][:s], label="test")
    plt.plot(overall_stats[2][0][:s], label="validation")
    plt.legend()
    plt.ylabel('CE loss')
    plt.xlabel('Epoch')
    plt.savefig("plots/CE-loss.png")
    plt.close()

    plt.title("Precentage of correct classifications")
    plt.plot(overall_stats[1][1][:s], label="test")
    plt.plot(overall_stats[2][1][:s], label="validation")
    plt.legend()
    plt.ylabel('% correct')
    plt.xlabel('Epoch')
    plt.savefig("plots/correct.png")
    plt.close()

    plt.title("AUC-ROC score")
    plt.plot(overall_stats[1][2][:s], label="test")
    plt.plot(overall_stats[2][2][:s], label="validation")
    plt.legend()
    plt.ylabel('AUC-ROC')
    plt.xlabel('Epoch')
    plt.savefig("plots/AUC-ROC.png")
    plt.close()

    plt.title("Precentage of correct classifications (relaxed)")
    plt.plot(overall_stats[1][3][:s], label="test")
    plt.plot(overall_stats[2][3][:s], label="validation")
    plt.legend()
    plt.ylabel('% correct (relaxed)')
    plt.xlabel('Epoch')
    plt.savefig("plots/correct-(relaxed).png")
    plt.close()

    pickle.dump(overall_stats, open("plots/plot_data.pkl", 'wb'))

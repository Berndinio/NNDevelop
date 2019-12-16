import torch.nn as nn
from transformers import *

from constants import Constants


def build_model(parameters):
    # begin dataset preparation
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(Constants.device)

    # replace classifier with linear + softmax along axis 1
    model.classifier = nn.Sequential(
            #nn.Linear(model.config.hidden_size, model.config.hidden_size),
            nn.Linear(model.config.hidden_size, parameters["num_labels"]),
            nn.Softmax(1)
        ).to(Constants.device)

    # update parameters
    model.num_labels = parameters["num_labels"]
    model.config.num_labels = parameters["num_labels"]
    return model

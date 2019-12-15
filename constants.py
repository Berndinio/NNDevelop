import torch

class Constants:
    device = None


if torch.cuda.is_available():
    Constants.device = torch.device('cuda')
else:
    Constants.device = torch.device('cpu')